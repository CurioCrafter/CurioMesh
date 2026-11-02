// SPDX-License-Identifier: GPL-3.0-or-later
//! Numeric kernels: generalized winding number and cross-field relaxation loops.

use numpy::Complex64;
use rayon::prelude::*;

/// `patch.winding_number`: Van Oosterom & Strackee solid angles summed per point.
/// Points run in parallel; triangles whose bounding box is far from the point are
/// still evaluated (the result must stay exact), so the cull lives in `inside_volume`.
pub fn winding_number(points: &[f64], tris: &[f64]) -> Vec<f64> {
    let tau = 2.0 * std::f64::consts::PI;
    let one = |p: &[f64]| {
        let mut total = 0.0;
        for t in tris.chunks_exact(9) {
            let a = [t[0] - p[0], t[1] - p[1], t[2] - p[2]];
            let b = [t[3] - p[0], t[4] - p[1], t[5] - p[2]];
            let c = [t[6] - p[0], t[7] - p[1], t[8] - p[2]];
            let norm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let dot = |u: &[f64; 3], v: &[f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            let (la, lb, lc) = (norm(&a), norm(&b), norm(&c));
            let bxc = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ];
            let det = dot(&a, &bxc);
            let den = la * lb * lc + dot(&a, &b) * lc + dot(&b, &c) * la + dot(&c, &a) * lb;
            total += det.atan2(den);
        }
        total / tau
    };
    if (points.len() / 3) * (tris.len() / 9) >= crate::topo::PARALLEL_MIN {
        points.par_chunks(3).map(one).collect()
    } else {
        points.chunks(3).map(one).collect()
    }
}

#[inline]
fn mul(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// Inner loop of `engine._estimate_field` (plain Jacobi diffusion).
pub fn cross_jacobi(
    cross: &mut [Complex64],
    face_a: &[i64],
    face_b: &[i64],
    transport: &[Complex64],
    pinned: &[bool],
    degree: &[f64],
    steps: usize,
) {
    let n = cross.len();
    let zero = Complex64::new(0.0, 0.0);
    let (mut acc_a, mut acc_b) = (vec![zero; n], vec![zero; n]);
    for _ in 0..steps {
        acc_a.fill(zero);
        acc_b.fill(zero);
        for k in 0..face_a.len() {
            let (fa, fb) = (face_a[k] as usize, face_b[k] as usize);
            let into_a = mul(cross[fb], transport[k]);
            let into_b = mul(cross[fa], transport[k].conj());
            acc_a[fa].re += into_a.re;
            acc_a[fa].im += into_a.im;
            acc_b[fb].re += into_b.re;
            acc_b[fb].im += into_b.im;
        }
        for f in 0..n {
            if pinned[f] {
                continue;
            }
            let scale = 1.0 / degree[f];
            let re = cross[f].re + (acc_a[f].re + acc_b[f].re);
            let im = cross[f].im + (acc_a[f].im + acc_b[f].im);
            cross[f] = Complex64::new(re * scale, im * scale);
        }
    }
}

/// Inner loop of `field.estimate_field_gl` (Ginzburg-Landau relaxation).
#[allow(clippy::too_many_arguments)]
pub fn cross_gl(
    cross: &mut [Complex64],
    face_a: &[i64],
    face_b: &[i64],
    transport: &[Complex64],
    pinned: &[bool],
    denom: &[f64],
    pin_term: &[Complex64],
    mu: f64,
    dt: f64,
    steps: usize,
) {
    let n = cross.len();
    let zero = Complex64::new(0.0, 0.0);
    let mut acc = vec![zero; n];
    let mut next = cross.to_vec();
    for step in 0..steps {
        acc.fill(zero);
        // np.bincount over concat((face_a, face_b)): all face_a terms first.
        for k in 0..face_a.len() {
            let v = mul(cross[face_b[k] as usize], transport[k]);
            let a = &mut acc[face_a[k] as usize];
            a.re += v.re;
            a.im += v.im;
        }
        for k in 0..face_b.len() {
            let v = mul(cross[face_a[k] as usize], transport[k].conj());
            let a = &mut acc[face_b[k] as usize];
            a.re += v.re;
            a.im += v.im;
        }
        let sharpen = step >= steps / 2;
        for f in 0..n {
            if pinned[f] {
                next[f] = cross[f];
                continue;
            }
            let scale = 1.0 / denom[f];
            let re = acc[f].re + mu * cross[f].re + pin_term[f].re;
            let im = acc[f].im + mu * cross[f].im + pin_term[f].im;
            let mut r = Complex64::new(re * scale, im * scale);
            if sharpen {
                let s = dt * (1.0 - r.re.hypot(r.im).powi(2));
                r = Complex64::new(r.re + s * r.re, r.im + s * r.im);
            }
            next[f] = r;
        }
        cross.copy_from_slice(&next);
    }
}

/// Two pictures along `axis`, looking down it from each side: one pass fills both, since the far side's
/// nearest surface is this side's farthest. `tris` indexes `points`. Returns per view, row by row:
/// the triangle seen at each pixel (`u32::MAX` for none) and its depth.
fn pictures_along(
    points: &[f64],
    tris: &[i64],
    center: [f64; 3],
    half: f64,
    pixels: usize,
    axis: usize,
) -> [(Vec<u32>, Vec<f64>); 2] {
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let cells = pixels * pixels;
    let mut front = (vec![u32::MAX; cells], vec![f64::INFINITY; cells]);
    let mut back = (vec![u32::MAX; cells], vec![f64::INFINITY; cells]);
    let (near, far) = (center[axis] + 2.0 * half, center[axis] - 2.0 * half);
    let scale = pixels as f64 / (2.0 * half);
    // The pixel a coordinate falls on, in units where pixel k's center is exactly k.
    let at = |value: f64, mid: f64| (value - mid + half) * scale - 0.5;
    let last = pixels as f64 - 1.0;
    for (id, t) in tris.chunks_exact(3).enumerate() {
        let corner = |k: usize, c: usize| points[t[k] as usize * 3 + c];
        let (x0, y0, z0) = (
            at(corner(0, u), center[u]),
            at(corner(0, v), center[v]),
            corner(0, axis),
        );
        let (x1, y1, z1) = (
            at(corner(1, u), center[u]),
            at(corner(1, v), center[v]),
            corner(1, axis),
        );
        let (x2, y2, z2) = (
            at(corner(2, u), center[u]),
            at(corner(2, v), center[v]),
            corner(2, axis),
        );
        let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if area == 0.0 {
            continue; // edge-on: a ray along the axis can't hit it
        }
        // Barycentric weights as a * row + b * column + c; inside means all three are >= 0.
        let edges = [
            (
                (y1 - y2) / area,
                (x2 - x1) / area,
                (x1 * y2 - x2 * y1) / area,
            ),
            (
                (y2 - y0) / area,
                (x0 - x2) / area,
                (x2 * y0 - x0 * y2) / area,
            ),
            (
                (y0 - y1) / area,
                (x1 - x0) / area,
                (x0 * y1 - x1 * y0) / area,
            ),
        ];
        let mut row = x0.min(x1).min(x2).ceil().max(0.0);
        let last_row = x0.max(x1).max(x2).floor().min(last);
        while row <= last_row {
            // A row crosses a triangle in one run, so solve for its ends instead of testing every pixel.
            let (mut low, mut high) = (0.0f64, last);
            for &(a, b, c) in &edges {
                let rest = a * row + c;
                if b > 0.0 {
                    low = low.max(-rest / b - 1e-9);
                } else if b < 0.0 {
                    high = high.min(-rest / b + 1e-9);
                } else if rest < -1e-12 {
                    high = -1.0;
                }
            }
            let base = row as usize * pixels;
            let (mut col, end) = (low.ceil(), high.floor());
            while col <= end {
                let w0 = edges[0].0 * row + edges[0].1 * col + edges[0].2;
                let w1 = edges[1].0 * row + edges[1].1 * col + edges[1].2;
                let z = w0 * z0 + w1 * z1 + (1.0 - w0 - w1) * z2;
                let cell = base + col as usize;
                let (from_near, from_far) = (near - z, z - far);
                if from_near >= 0.0 && from_near < front.1[cell] {
                    front.0[cell] = id as u32;
                    front.1[cell] = from_near;
                }
                if from_far >= 0.0 && from_far < back.1[cell] {
                    back.0[cell] = id as u32;
                    back.1[cell] = from_far;
                }
                col += 1.0;
            }
            row += 1.0;
        }
    }
    [front, back]
}

/// Six orthographic pictures, one sample at each pixel center: what `autoretopo.pictures` gets by casting a ray
/// per pixel. All rays of a view are parallel to an axis, so a depth buffer finds the same first hits with no BVH.
/// Views run +x, -x, +y, -y, +z, -z, packed one after another: (hit, normal xyz, depth).
pub fn pictures(
    points: &[f64],
    tris: &[i64],
    normals: &[f64],
    center: [f64; 3],
    half: f64,
    pixels: usize,
) -> (Vec<bool>, Vec<f64>, Vec<f64>, Vec<i64>) {
    let run = |axis: usize| pictures_along(points, tris, center, half, pixels, axis);
    // ponytail: one thread per axis tops out at 3x; bin triangles by row if a huge mesh ever needs more.
    let views: Vec<_> = if tris.len() / 3 >= 2000 {
        (0..3usize).into_par_iter().map(run).collect()
    } else {
        (0..3usize).map(run).collect()
    };
    let cells = pixels * pixels;
    let mut hit = vec![false; 6 * cells];
    let mut normal = vec![0.0; 18 * cells];
    let mut depth = vec![0.0; 6 * cells];
    let mut face = vec![0_i64; 6 * cells];
    let unit = |id: u32| {
        let n = &normals[id as usize * 3..id as usize * 3 + 3];
        let size = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
            .sqrt()
            .max(f64::MIN_POSITIVE);
        [n[0] / size, n[1] / size, n[2] / size]
    };
    for (index, view) in views.iter().flatten().enumerate() {
        for (cell, &id) in view.0.iter().enumerate() {
            if id != u32::MAX {
                let at = index * cells + cell;
                hit[at] = true;
                depth[at] = view.1[cell];
                face[at] = id as i64;
                normal[at * 3..at * 3 + 3].copy_from_slice(&unit(id));
            }
        }
    }
    (hit, normal, depth, face)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn minus(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scaled_unit(v: [f64; 3], length: f64) -> [f64; 3] {
    let by = if length > 0.0 { length } else { 1.0 };
    [v[0] / by, v[1] / by, v[2] / by]
}

/// `topology._quad_points_valid` for one quad: finite, convex, non-crossed, low warp.
pub fn quad_valid(p: &[[f64; 3]; 4], epsilon: f64, warp_cosine: f64) -> bool {
    if !p.iter().flatten().all(|x| x.is_finite()) {
        return false;
    }
    let mut normal = [0.0; 3];
    for k in 0..4 {
        let c = cross(p[k], p[(k + 1) % 4]);
        normal = [normal[0] + c[0], normal[1] + c[1], normal[2] + c[2]];
    }
    let normal_length = dot(normal, normal).sqrt();
    if !(normal_length > epsilon) {
        return false;
    }
    let normal = scaled_unit(normal, normal_length);
    for (first, second) in [([0, 1, 2], [0, 2, 3]), ([1, 2, 3], [1, 3, 0])] {
        let side = |t: [usize; 3]| {
            let n = cross(minus(p[t[1]], p[t[0]]), minus(p[t[2]], p[t[0]]));
            (n, dot(n, n).sqrt())
        };
        let ((a, la), (b, lb)) = (side(first), side(second));
        if !(la > epsilon && lb > epsilon)
            || dot(scaled_unit(a, la), scaled_unit(b, lb)) < warp_cosine
        {
            return false;
        }
    }
    for k in 0..4 {
        let following = minus(p[(k + 1) % 4], p[k]);
        let previous = minus(p[(k + 3) % 4], p[k]);
        let denominator = dot(following, following).sqrt() * dot(previous, previous).sqrt();
        if !(denominator > epsilon) {
            return false;
        }
        let jacobian = dot(cross(following, previous), normal) / denominator;
        if !(jacobian.is_finite() && jacobian > 1e-8) {
            return false;
        }
    }
    let size = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let axis = if size[1] > size[0] && size[1] >= size[2] {
        1
    } else if size[2] > size[0] {
        2
    } else {
        0
    };
    let (u, v) = [(1, 2), (0, 2), (0, 1)][axis];
    let flat = |k: usize| (p[k][u], p[k][v]);
    let turn = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    };
    let crossing = |a, b, c, d| {
        let (t1, t2, t3, t4) = (turn(a, b, c), turn(a, b, d), turn(c, d, a), turn(c, d, b));
        ((t1 > epsilon && t2 < -epsilon) || (t1 < -epsilon && t2 > epsilon))
            && ((t3 > epsilon && t4 < -epsilon) || (t3 < -epsilon && t4 > epsilon))
    };
    !crossing(flat(0), flat(1), flat(2), flat(3)) && !crossing(flat(1), flat(2), flat(3), flat(0))
}

/// `topology._parity_template_validity`: per face, how many templates it has and whether each is valid, in the order
/// they are tried. `edges` holds the split edges as sorted pairs with their midpoints in `mids`.
#[allow(clippy::too_many_arguments)]
pub fn parity_templates(
    indptr: &[i64],
    flat: &[i64],
    edges: &[i64],
    mids: &[i64],
    vertices: &[f64],
    base: &[f64],
    warp_cosine: f64,
) -> (Vec<i64>, Vec<bool>) {
    let keys: Vec<(i64, i64)> = edges.chunks_exact(2).map(|e| (e[0], e[1])).collect();
    let rows = crate::topo::collect(indptr.len() - 1, |index| {
        let face = &flat[indptr[index] as usize..indptr[index + 1] as usize];
        let mut augmented = Vec::with_capacity(face.len() * 2);
        for (corner, &vertex) in face.iter().enumerate() {
            augmented.push(vertex);
            let following = face[(corner + 1) % face.len()];
            if let Ok(at) = keys.binary_search(&(vertex.min(following), vertex.max(following))) {
                augmented.push(mids[at]);
            }
        }
        const CENTER: i64 = -1;
        let mut center = [0.0; 3];
        for &vertex in face {
            for c in 0..3 {
                center[c] += vertices[vertex as usize * 3 + c];
            }
        }
        let center = center.map(|x| x / face.len() as f64);
        let cell_ok = |cell: [i64; 4]| {
            let p = cell.map(|i| {
                if i == CENTER {
                    center
                } else {
                    [
                        base[i as usize * 3],
                        base[i as usize * 3 + 1],
                        base[i as usize * 3 + 2],
                    ]
                }
            });
            let (mut size, mut c) = (0.0, 0);
            while c < 3 {
                let (mut low, mut high) = (p[0][c], p[0][c]);
                for q in &p[1..] {
                    low = low.min(q[c]);
                    high = high.max(q[c]);
                }
                size += (high - low) * (high - low);
                c += 1;
            }
            quad_valid(&p, (size * 1e-14).max(1e-20), warp_cosine)
        };
        let (mut patches, count) = (Vec::with_capacity(5), augmented.len());
        let split = count - face.len();
        if face.len() == 4 && split == 0 {
            patches.push(cell_ok([face[0], face[1], face[2], face[3]]));
        }
        if face.len() == 4 && split == 2 {
            for offset in 0..3 {
                let at = |step: usize| augmented[(offset + step) % 6];
                patches.push(
                    cell_ok([at(0), at(1), at(2), at(3)]) && cell_ok([at(3), at(4), at(5), at(6)]),
                );
            }
        }
        if count % 2 == 0 && count >= 4 {
            for offset in 0..2 {
                let at = |step: usize| augmented[(offset + step) % count];
                patches.push((0..count / 2).all(|cell| {
                    cell_ok([CENTER, at(2 * cell), at(2 * cell + 1), at(2 * cell + 2)])
                }));
            }
        }
        patches
    });
    let counts = rows.iter().map(|row| row.len() as i64).collect();
    (counts, rows.into_iter().flatten().collect())
}

/// NumPy's `floor_divide` for floats.
fn floor_div(a: f64, b: f64) -> f64 {
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div == 0.0 {
        return 0.0f64.copysign(a / b);
    }
    let mut floored = div.floor();
    if div - floored > 0.5 {
        floored += 1.0;
    }
    floored
}

/// `parts._cluster`: leader clustering. Each point joins the first leader in reach, looking through the nine nearby cells
/// in order, or becomes a leader itself.
pub fn leader_clusters(values: &[f64], tol: f64) -> Vec<i64> {
    let mut cells: std::collections::HashMap<(i64, i64), Vec<u32>, crate::topo::FxBuild> = Default::default();
    let mut leaders: Vec<(f64, f64)> = Vec::new();
    let mut ids = Vec::with_capacity(values.len() / 2);
    for p in values.chunks_exact(2) {
        let (cx, cy) = (floor_div(p[0], tol) as i64, floor_div(p[1], tol) as i64);
        let mut hit = -1i64;
        'look: for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(row) = cells.get(&(cx + dx, cy + dy)) {
                    for &k in row {
                        let leader = leaders[k as usize];
                        if (leader.0 - p[0]).hypot(leader.1 - p[1]) <= tol {
                            hit = k as i64;
                            break 'look;
                        }
                    }
                }
            }
        }
        if hit < 0 {
            hit = leaders.len() as i64;
            leaders.push((p[0], p[1]));
            cells.entry((cx, cy)).or_default().push(hit as u32);
        }
        ids.push(hit);
    }
    ids
}
