// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Even rows of quads over one part, from a direction field and a grid laid along it: the idea of Jakob, Tarini,
//! Panozzo and Sorkine-Hornung, "Instant Field-Aligned Meshes" (2015), as the add-on's Field Rows (`fieldmesh.py`)
//! does it, with its sweeps (`field.rs`). This is a leaner own version: cells shrink where the surface turns fast,
//! crease and border lines are pinned, points whose nearest grid corner is the same form one corner (the add-on's
//! way), corners one cell apart along the field are linked and the faces walked (Instant Meshes' way), and at the end
//! every face is split into quads (their pure-quad mode).

use crate::field;
use std::collections::HashMap;

type V3 = [f64; 3];
const SPLIT_SHARE: f64 = 0.4; // no edge of the worked copy longer than this share of a cell
const SAME_CORNER: f64 = 0.35; // grid corners closer than this share of a cell are one corner
const ORIENT_SWEEPS: usize = 8;
const POSITION_SWEEPS: usize = 8;
const ROBUST: f64 = 0.05;
const SETTLED: f64 = 0.02;
const COARSEST: usize = 64; // points: stop coarsening here
const MOST_POINTS: usize = 2_000_000; // a worked copy bigger than this isn't tried

fn at(v: &[f64], i: usize) -> V3 {
    [v[3 * i], v[3 * i + 1], v[3 * i + 2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: V3) -> V3 {
    let n = dot(a, a).sqrt().max(1e-30);
    [a[0] / n, a[1] / n, a[2] / n]
}
fn tangent(v: V3, n: V3) -> V3 {
    let d = dot(v, n);
    [v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]]
}

/// Holes the face walk left (a link missing near a pole): each border loop of 3 to 8 corners becomes a face, wound
/// against its border so it fits. A loop that is the part's own outline in the source stays open.
fn fill_small_holes(indptr: &mut Vec<i64>, flat: &mut Vec<i64>, fine_tris: &[i64], label: &[u32]) {
    let mut directed = std::collections::HashSet::new();
    for f in 0..indptr.len() - 1 {
        let face = &flat[indptr[f] as usize..indptr[f + 1] as usize];
        for k in 0..face.len() {
            directed.insert((face[k], face[(k + 1) % face.len()]));
        }
    }
    // Corners on the source's own outline: their regions hold points on a source border.
    let mut uses: HashMap<(i64, i64), i32> = HashMap::new();
    for t in fine_tris.chunks_exact(3) {
        for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *uses.entry((x.min(y), x.max(y))).or_default() += 1;
        }
    }
    let outline: std::collections::HashSet<i64> = uses.iter().filter(|(_, &n)| n == 1).flat_map(|(&(x, y), _)| [label[x as usize] as i64, label[y as usize] as i64]).collect();
    let mut from: HashMap<i64, Vec<i64>> = HashMap::new(); // hole edge b -> a for every border edge a -> b
    for &(a, b) in &directed {
        if !directed.contains(&(b, a)) {
            from.entry(b).or_default().push(a);
        }
    }
    let mut done = std::collections::HashSet::new();
    let mut starts: Vec<i64> = from.keys().copied().collect();
    starts.sort_unstable();
    for start in starts {
        if done.contains(&start) || from[&start].len() != 1 {
            continue;
        }
        let mut face = vec![start];
        let mut v = from[&start][0];
        while v != start && face.len() <= 8 {
            match from.get(&v) {
                Some(next) if next.len() == 1 && !face.contains(&v) => {
                    face.push(v);
                    v = next[0];
                }
                _ => break,
            }
        }
        done.extend(face.iter().copied());
        if v == start && (3..=8).contains(&face.len()) && !face.iter().all(|c| outline.contains(c)) {
            flat.extend_from_slice(&face);
            indptr.push(flat.len() as i64);
        }
    }
}

/// All quads at about `cell`: rows extracted at twice the size, then every face of n sides split into n quads through
/// its middle and its sides' middles (Instant Meshes' pure-quad mode). No closing paths are needed, so no face count
/// blows up, and poles sit only where the field has them.
pub fn field_quads(points: &[f64], tris: &[i64], cell: f64, seed: u64) -> Option<(Vec<f64>, Vec<i64>, Vec<i64>)> {
    let (mut out, indptr, flat) = field_polygons(points, tris, 2.0 * cell, seed)?;
    let mut middle: HashMap<(i64, i64), i64> = HashMap::new();
    let (mut q_indptr, mut q_flat) = (vec![0i64], Vec::with_capacity(flat.len() * 4));
    for f in 0..indptr.len() - 1 {
        let face: Vec<i64> = flat[indptr[f] as usize..indptr[f + 1] as usize].to_vec();
        let n = face.len();
        let centre = (0..n).fold([0.0; 3], |s, k| {
            let p = at(&out, face[k] as usize);
            [s[0] + p[0] / n as f64, s[1] + p[1] / n as f64, s[2] + p[2] / n as f64]
        });
        out.extend(centre);
        let c = (out.len() / 3 - 1) as i64;
        let mids: Vec<i64> = (0..n).map(|k| {
            let (x, y) = (face[k], face[(k + 1) % n]);
            *middle.entry((x.min(y), x.max(y))).or_insert_with(|| {
                let (p, q) = (at(&out, x as usize), at(&out, y as usize));
                out.extend([0, 1, 2].map(|i| (p[i] + q[i]) / 2.0));
                (out.len() / 3 - 1) as i64
            })
        }).collect();
        for k in 0..n {
            q_flat.extend([face[k], mids[k], c, mids[(k + n - 1) % n]]);
            q_indptr.push(q_flat.len() as i64);
        }
    }
    let q_flat = finish(&mut out, q_flat, points, tris);
    let q_indptr = (0..=(q_flat.len() / 4) as i64).map(|f| 4 * f).collect();
    Some((out, q_indptr, q_flat))
}

const MOST_FIT: f64 = 0.75; // share of a side a corner may move while fitting
const FIT_STRONG: f64 = 0.05; // a plane direction counts if this strong against the strongest: flat moves only along the normal
const GUARD_QUALITY: f64 = 0.1;
const UNTANGLE_BELOW: f64 = 0.05;
const FINISH_ROUNDS: usize = 4;

/// `patch.quad_quality`: > 0 means convex and warped under 60 degrees. min(minimum scaled Jacobian about the mean
/// normal, (cos warp - 0.5) / 0.5), the warp being the larger fold of the two diagonal splits.
pub fn quad_quality(q: &[V3; 4]) -> f64 {
    let normal = unit(cross(sub(q[2], q[0]), sub(q[3], q[1])));
    let mut jac = f64::INFINITY;
    for k in 0..4 {
        let following = sub(q[(k + 1) % 4], q[k]);
        let previous = sub(q[(k + 3) % 4], q[k]);
        let length = (dot(following, following) * dot(previous, previous)).sqrt();
        jac = jac.min(if length > 1e-30 { dot(cross(following, previous), normal) / length } else { -1.0 });
    }
    let fold_a = dot(unit(cross(sub(q[1], q[0]), sub(q[2], q[0]))), unit(cross(sub(q[2], q[0]), sub(q[3], q[0]))));
    let fold_b = dot(unit(cross(sub(q[2], q[1]), sub(q[3], q[1]))), unit(cross(sub(q[3], q[1]), sub(q[0], q[1]))));
    jac.min((fold_a.min(fold_b) - 0.5) * 2.0)
}

fn quad_at(points: &[f64], flat: &[i64], f: usize) -> [V3; 4] {
    [0, 1, 2, 3].map(|k| at(points, flat[f * 4 + k] as usize))
}

/// Eigenvalues (ascending) and unit eigenvectors (as rows) of a symmetric 3x3 matrix, by Jacobi rotations.
fn eigh3(mut m: [[f64; 3]; 3]) -> ([f64; 3], [V3; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let (mut p, mut q, mut big) = (0, 1, m[0][1].abs());
        for (i, j) in [(0, 2), (1, 2)] {
            if m[i][j].abs() > big {
                (p, q, big) = (i, j, m[i][j].abs());
            }
        }
        if big < 1e-300 {
            break;
        }
        let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
        for k in 0..3 {
            let (mkp, mkq) = (m[k][p], m[k][q]);
            m[k][p] = c * mkp - s * mkq;
            m[k][q] = s * mkp + c * mkq;
        }
        for k in 0..3 {
            let (mpk, mqk) = (m[p][k], m[q][k]);
            m[p][k] = c * mpk - s * mqk;
            m[q][k] = s * mpk + c * mqk;
        }
        for row in v.iter_mut() {
            let (vp, vq) = (row[p], row[q]);
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&x, &y| m[x][x].total_cmp(&m[y][y]));
    (order.map(|k| m[k][k]), order.map(|k| [v[0][k], v[1][k], v[2][k]]))
}

/// The quads' corners onto the source, the add-on's way (`fieldmesh.quads_from`'s last steps): `refit` gives each
/// corner the patch of source nearest to it and fits it to the planes there, `untangle` slides the corners of nearly
/// folded quads toward their neighbours along the surface, and `_even_out_folds` irons out what is still folded.
fn finish(points: &mut Vec<f64>, mut quads: Vec<i64>, source: &[f64], source_tris: &[i64]) -> Vec<i64> {
    if quads.is_empty() {
        return quads;
    }
    let side = {
        let mut sides: Vec<f64> = quads.chunks_exact(4).map(|q| {
            let d = sub(at(points, q[0] as usize), at(points, q[1] as usize));
            dot(d, d).sqrt()
        }).collect();
        sides.sort_by(f64::total_cmp);
        sides[sides.len() / 2]
    };
    refit(points, &quads, source, source_tris, side, 3);
    quads = clean(points, quads);
    untangle(points, &quads, 12);
    refit(points, &quads, source, source_tris, side, 1);
    for _ in 0..FINISH_ROUNDS {
        quads = clean(points, quads);
        untangle(points, &quads, 24);
        if !(0..quads.len() / 4).any(|f| quad_quality(&quad_at(points, &quads, f)) <= 0.0) {
            break;
        }
    }
    even_out_folds(points, &quads, 40);
    clean(points, quads)
}

/// `fieldmesh.clean`: a corner in only two quads adds nothing, so those two become one (`_dissolve_doublets`); a folded
/// quad goes by merging two of its opposite corners, which leaves the faces round it quads (`_merge_folded`).
fn clean(points: &mut [f64], mut quads: Vec<i64>) -> Vec<i64> {
    for _ in 0..6 {
        let count = quads.len() / 4;
        // Doublets.
        let mut around: HashMap<i64, Vec<usize>> = HashMap::new();
        for f in 0..count {
            for &v in &quads[f * 4..f * 4 + 4] {
                around.entry(v).or_default().push(f);
            }
        }
        let mut corners: Vec<i64> = around.iter().filter(|(_, f)| f.len() == 2).map(|(&v, _)| v).collect();
        corners.sort_unstable();
        let mut gone = vec![false; count];
        let mut added: Vec<[i64; 4]> = Vec::new();
        let turned = |f: usize, v: i64, quads: &[i64]| -> [i64; 4] {
            let q = &quads[f * 4..f * 4 + 4];
            let k = q.iter().position(|&x| x == v).unwrap();
            [0, 1, 2, 3].map(|i| q[(k + i) % 4])
        };
        for v in corners {
            let ids = &around[&v];
            if gone[ids[0]] || gone[ids[1]] {
                continue;
            }
            let (first, second) = (turned(ids[0], v, &quads), turned(ids[1], v, &quads));
            if first[1] != second[3] || first[3] != second[1] {
                continue; // not two faces sharing both edges at this corner
            }
            let merged = [first[1], first[2], first[3], second[2]];
            let mut sorted = merged;
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w[0] == w[1]) {
                continue;
            }
            gone[ids[0]] = true;
            gone[ids[1]] = true;
            added.push(merged);
        }
        let doublets = added.len();
        let mut next: Vec<i64> = (0..count).filter(|&f| !gone[f]).flat_map(|f| quads[f * 4..f * 4 + 4].to_vec()).collect();
        next.extend(added.iter().flatten());
        quads = next;
        // Folded quads.
        let count = quads.len() / 4;
        let mut neighbours: HashMap<i64, std::collections::BTreeSet<i64>> = HashMap::new();
        for q in quads.chunks_exact(4) {
            for k in 0..4 {
                neighbours.entry(q[k]).or_default().insert(q[(k + 1) % 4]);
                neighbours.entry(q[(k + 1) % 4]).or_default().insert(q[k]);
            }
        }
        let mut rename: HashMap<i64, i64> = HashMap::new();
        let mut touched = std::collections::HashSet::new();
        let mut gone = vec![false; count];
        for f in 0..count {
            if quad_quality(&quad_at(points, &quads, f)) > 0.0 {
                continue;
            }
            let [a, b, c, d] = [0, 1, 2, 3].map(|k| quads[f * 4 + k]);
            if [a, b, c, d].iter().any(|v| touched.contains(v)) {
                continue;
            }
            let distance = |x: i64, y: i64| {
                let e = sub(at(points, x as usize), at(points, y as usize));
                dot(e, e)
            };
            let options = if distance(a, c) <= distance(b, d) { [(a, c, [b, d]), (b, d, [a, c])] } else { [(b, d, [a, c]), (a, c, [b, d])] };
            for (x, y, side) in options {
                let shared: std::collections::BTreeSet<i64> = neighbours[&x].intersection(&neighbours[&y]).copied().collect();
                let wanted: std::collections::BTreeSet<i64> = side.into_iter().collect();
                if shared == wanted && neighbours[&side[0]].len() > 2 && neighbours[&side[1]].len() > 2 {
                    rename.insert(y, x);
                    let middle = [0, 1, 2].map(|i| 0.5 * (points[x as usize * 3 + i] + points[y as usize * 3 + i]));
                    points[x as usize * 3..x as usize * 3 + 3].copy_from_slice(&middle);
                    touched.extend([a, b, c, d]);
                    touched.extend(neighbours[&x].iter().copied());
                    touched.extend(neighbours[&y].iter().copied());
                    gone[f] = true;
                    break;
                }
            }
        }
        let folded = gone.iter().filter(|&&g| g).count();
        quads = (0..count).filter(|&f| !gone[f]).flat_map(|f| quads[f * 4..f * 4 + 4].iter().map(|v| *rename.get(v).unwrap_or(v)).collect::<Vec<_>>()).collect();
        if doublets == 0 && folded == 0 {
            break;
        }
    }
    quads
}

/// `fieldmesh.refit`: each corner takes the source points nearer to it than to any other corner (measured along the
/// source), and moves to where it best fits their triangles' planes; a move that makes a quad bad goes back.
fn refit(points: &mut [f64], quads: &[i64], source: &[f64], source_tris: &[i64], side: f64, rounds: usize) {
    let n_out = points.len() / 3;
    let n_src = source.len() / 3;
    let indptr: Vec<i64> = (0..=(source_tris.len() / 3) as i64).map(|f| 3 * f).collect();
    let tree = crate::near::FaceTree::new(source, &indptr, source_tris);
    let (a, b) = edges_of(source_tris);
    let mut around: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_src];
    for (&x, &y) in a.iter().zip(&b) {
        let d = sub(at(source, x as usize), at(source, y as usize));
        let length = dot(d, d).sqrt();
        around[x as usize].push((y as usize, length));
        around[y as usize].push((x as usize, length));
    }
    let face_normal = |t: &[i64]| {
        let (p, q, r) = (at(source, t[0] as usize), at(source, t[1] as usize), at(source, t[2] as usize));
        cross(sub(q, p), sub(r, p)) // length is twice the area
    };
    let stop = std::sync::atomic::AtomicBool::new(false);
    for _ in 0..rounds {
        // Seeds: each corner's nearest source point, then every source point to the corner whose seed is nearest.
        let (_, found) = crate::near::nearest_faces(&tree, points, &stop);
        let mut owner = vec![usize::MAX; n_src];
        let mut far = vec![f64::INFINITY; n_src];
        let mut heap = std::collections::BinaryHeap::new();
        for (v, hit) in found.iter().enumerate() {
            let t = &source_tris[hit.1 as usize * 3..hit.1 as usize * 3 + 3];
            let seed = *t.iter().min_by(|&&x, &&y| {
                let (dx, dy) = (sub(at(source, x as usize), hit.0), sub(at(source, y as usize), hit.0));
                dot(dx, dx).total_cmp(&dot(dy, dy))
            }).unwrap() as usize;
            if owner[seed] == usize::MAX {
                owner[seed] = v;
                far[seed] = 0.0;
                heap.push((std::cmp::Reverse(0u64), seed));
            }
        }
        while let Some((std::cmp::Reverse(d), x)) = heap.pop() {
            let d = f64::from_bits(d);
            if d > far[x] {
                continue;
            }
            for &(y, length) in &around[x] {
                if d + length < far[y] {
                    far[y] = d + length;
                    owner[y] = owner[x];
                    heap.push((std::cmp::Reverse((d + length).to_bits()), y));
                }
            }
        }
        // Fit: planes and pull per corner from the triangles its source points own, a third each.
        let mut planes = vec![[[0.0; 3]; 3]; n_out];
        let mut pull = vec![[0.0; 3]; n_out];
        for t in source_tris.chunks_exact(3) {
            let doubled = face_normal(t);
            let area = dot(doubled, doubled).sqrt() / 2.0;
            if area <= 0.0 {
                continue;
            }
            let nrm = unit(doubled);
            let middle = [0, 1, 2].map(|k| (source[t[0] as usize * 3 + k] + source[t[1] as usize * 3 + k] + source[t[2] as usize * 3 + k]) / 3.0);
            for &c in t {
                let o = owner[c as usize];
                if o == usize::MAX {
                    continue;
                }
                for i in 0..3 {
                    for j in 0..3 {
                        planes[o][i][j] += area / 3.0 * nrm[i] * nrm[j];
                    }
                }
                let reach = area / 3.0 * dot(nrm, sub(middle, at(points, o)));
                for i in 0..3 {
                    pull[o][i] += reach * nrm[i];
                }
            }
        }
        let mut moved = points.to_vec();
        for v in 0..n_out {
            if planes[v][0][0] + planes[v][1][1] + planes[v][2][2] <= 0.0 {
                continue; // no source of its own
            }
            let (values, vectors) = eigh3(planes[v]);
            let strongest = values[2].max(1e-30);
            let mut step = [0.0; 3];
            for k in 0..3 {
                if values[k] > FIT_STRONG * strongest {
                    let along = dot(vectors[k], pull[v]) / values[k].max(1e-30);
                    for i in 0..3 {
                        step[i] += vectors[k][i] * along;
                    }
                }
            }
            let length = dot(step, step).sqrt();
            let scale = if length > 1e-30 { (MOST_FIT * side / length).min(1.0) } else { 1.0 };
            for i in 0..3 {
                moved[v * 3 + i] += step[i] * scale;
            }
        }
        // Any corner of a quad the move made bad goes back.
        let before: Vec<f64> = (0..quads.len() / 4).map(|f| quad_quality(&quad_at(points, quads, f))).collect();
        for _ in 0..8 {
            let mut reverted = false;
            for f in 0..quads.len() / 4 {
                if quad_quality(&quad_at(&moved, quads, f)) < before[f].min(GUARD_QUALITY) {
                    for &v in &quads[f * 4..f * 4 + 4] {
                        let v = v as usize;
                        if moved[v * 3..v * 3 + 3] != points[v * 3..v * 3 + 3] {
                            moved[v * 3..v * 3 + 3].copy_from_slice(&points[v * 3..v * 3 + 3]);
                            reverted = true;
                        }
                    }
                }
            }
            if !reverted {
                break;
            }
        }
        points.copy_from_slice(&moved);
    }
}

/// Neighbours' middle per corner over the quads' sides, and how many sides each corner has.
fn neighbour_middles(points: &[f64], quads: &[i64]) -> Vec<V3> {
    let n = points.len() / 3;
    let (mut sum, mut uses) = (vec![[0.0; 3]; n], vec![0.0; n]);
    for q in quads.chunks_exact(4) {
        for k in 0..4 {
            for (x, y) in [(q[k], q[(k + 1) % 4]), (q[(k + 1) % 4], q[k])] {
                let p = at(points, y as usize);
                for i in 0..3 {
                    sum[x as usize][i] += p[i];
                }
                uses[x as usize] += 1.0;
            }
        }
    }
    sum.iter().zip(&uses).map(|(s, &u)| if u > 0.0 { s.map(|x| x / u) } else { [0.0; 3] }).collect()
}

/// `fieldmesh.untangle`: only the corners of folded or nearly folded quads move, toward their neighbours' middle,
/// along the surface.
fn untangle(points: &mut [f64], quads: &[i64], rounds: usize) {
    let n = points.len() / 3;
    for _ in 0..rounds {
        let mut moving = vec![false; n];
        let mut any = false;
        for f in 0..quads.len() / 4 {
            if quad_quality(&quad_at(points, quads, f)) < UNTANGLE_BELOW {
                for &v in &quads[f * 4..f * 4 + 4] {
                    moving[v as usize] = true;
                }
                any = true;
            }
        }
        if !any {
            return;
        }
        let mut normal = vec![[0.0; 3]; n];
        for f in 0..quads.len() / 4 {
            let q = quad_at(points, quads, f);
            let area = cross(sub(q[2], q[0]), sub(q[3], q[1]));
            for &v in &quads[f * 4..f * 4 + 4] {
                for i in 0..3 {
                    normal[v as usize][i] += area[i];
                }
            }
        }
        let target = neighbour_middles(points, quads);
        for v in (0..n).filter(|&v| moving[v]) {
            let step = tangent(sub(target[v], at(points, v)), unit(normal[v]));
            for i in 0..3 {
                points[v * 3 + i] += 0.7 * step[i];
            }
        }
    }
}

/// `fieldmesh._even_out_folds`: the last resort for what is still folded. Each corner of a folded quad, and every
/// corner reached before, goes halfway to its neighbours' middle in all directions, until none is folded.
fn even_out_folds(points: &mut [f64], quads: &[i64], rounds: usize) {
    let n = points.len() / 3;
    let mut reach = vec![false; n];
    for _ in 0..rounds {
        let mut any = false;
        for f in 0..quads.len() / 4 {
            if quad_quality(&quad_at(points, quads, f)) <= 0.0 {
                for &v in &quads[f * 4..f * 4 + 4] {
                    reach[v as usize] = true;
                }
                any = true;
            }
        }
        if !any {
            return;
        }
        let target = neighbour_middles(points, quads);
        for v in (0..n).filter(|&v| reach[v]) {
            for i in 0..3 {
                points[v * 3 + i] += 0.5 * (target[v][i] - points[v * 3 + i]);
            }
        }
    }
}

/// splitmix64: a seeded stream, so the same part gives the same rows every run.
struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn colors(&mut self, n: usize) -> Vec<i64> {
        (0..n).map(|_| (self.next() % 3) as i64).collect()
    }
    fn unit_ish(&mut self) -> V3 {
        [0, 1, 2].map(|_| (self.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
    }
}

struct Level {
    points: Vec<f64>,
    normals: Vec<f64>,
    a: Vec<i64>,
    b: Vec<i64>,
    parent: Vec<u32>, // into the next coarser level; empty on the coarsest
    pinned: Vec<i64>, // 0 free, 1 on a crease or border line (field along it, slides only along it), 2 a corner of lines
    along: Vec<f64>,  // the line's direction where pinned is 1
    sizes: Vec<f64>,  // the cell size wanted at each point
}

fn edges_of(tris: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut e: Vec<(i64, i64)> = tris.chunks_exact(3).flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]).map(|(x, y)| (x.min(y), x.max(y))).collect();
    e.sort_unstable();
    e.dedup();
    e.into_iter().unzip()
}

/// Split edges longer than `most` at their middle, both triangles beside an edge alike so the surface stays closed,
/// until none is left: a triangle with one long edge becomes two, with two three, with three four.
fn split_until(mut points: Vec<f64>, mut tris: Vec<i64>, most: f64) -> Option<(Vec<f64>, Vec<i64>)> {
    let most2 = most * most;
    loop {
        let long = |x: i64, y: i64, points: &[f64]| {
            let d = sub(at(points, x as usize), at(points, y as usize));
            dot(d, d) > most2
        };
        let mut middle: HashMap<(i64, i64), i64> = HashMap::new();
        for t in tris.chunks_exact(3) {
            for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                if long(x, y, &points) && !middle.contains_key(&(x.min(y), x.max(y))) {
                    let (p, q) = (at(&points, x as usize), at(&points, y as usize));
                    points.extend([0, 1, 2].map(|k| (p[k] + q[k]) / 2.0));
                    middle.insert((x.min(y), x.max(y)), (points.len() / 3 - 1) as i64);
                }
            }
        }
        if middle.is_empty() {
            return Some((points, tris));
        }
        if points.len() / 3 > MOST_POINTS {
            return None;
        }
        let mid = |x: i64, y: i64| middle.get(&(x.min(y), x.max(y))).copied();
        let mut out = Vec::with_capacity(tris.len() * 2);
        for t in tris.chunks_exact(3) {
            // Start at the corner before the first split edge, so the cases below cover every pattern.
            let splits = [mid(t[0], t[1]), mid(t[1], t[2]), mid(t[2], t[0])];
            let k = (0..3).find(|&k| splits[k].is_some() && splits[(k + 2) % 3].is_none()).unwrap_or(0);
            let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
            let (ab, bc, ca) = (splits[k], splits[(k + 1) % 3], splits[(k + 2) % 3]);
            match (ab, bc, ca) {
                (None, None, None) => out.extend([a, b, c]),
                (Some(m), None, None) => out.extend([a, m, c, m, b, c]),
                (Some(m), Some(n), None) => out.extend([a, m, c, m, b, n, m, n, c]),
                (Some(m), Some(n), Some(o)) => out.extend([a, m, o, m, b, n, o, n, c, m, n, o]),
                _ => unreachable!("a split edge always has an unsplit one before it unless all three are split"),
            }
        }
        tris = out;
    }
}

const SMALLEST: f64 = 0.5; // no cell under this share of the even size (the add-on's)
const TURN: f64 = 0.5; // radians the surface may turn across one cell, so a quad never bends far round a tight curve

/// The cell size at each point: `cell` where the surface is gentle, smaller where it turns faster than TURN per cell
/// (down to SMALLEST of it), then all scaled together so the part still gets about as many cells as even ones would.
fn cell_sizes(points: &[f64], tris: &[i64], normals: &[f64], a: &[i64], b: &[i64], cell: f64) -> Vec<f64> {
    let n = points.len() / 3;
    let mut bend = vec![0.0f64; n]; // radians per unit length: the fastest turn along any edge at the point
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as usize, y as usize);
        let d = sub(at(points, x), at(points, y));
        let length = dot(d, d).sqrt().max(1e-30);
        let turn = dot(at(normals, x), at(normals, y)).clamp(-1.0, 1.0).acos() / length;
        bend[x] = bend[x].max(turn);
        bend[y] = bend[y].max(turn);
    }
    let mut area = vec![0.0; n];
    for t in tris.chunks_exact(3) {
        let (p, q, r) = (at(points, t[0] as usize), at(points, t[1] as usize), at(points, t[2] as usize));
        let c = cross(sub(q, p), sub(r, p));
        for &v in t {
            area[v as usize] += dot(c, c).sqrt() / 6.0;
        }
    }
    let mut sizes: Vec<f64> = bend.iter().map(|&k| if k > 0.0 { (TURN / k).clamp(SMALLEST * cell, cell) } else { cell }).collect();
    // A point's size is never more than a quarter over its neighbours', so sizes change smoothly.
    for _ in 0..3 {
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as usize, y as usize);
            sizes[x] = sizes[x].min(1.25 * sizes[y]);
            sizes[y] = sizes[y].min(1.25 * sizes[x]);
        }
    }
    let even: f64 = area.iter().sum::<f64>() / (cell * cell);
    let wanted: f64 = area.iter().zip(&sizes).map(|(a, s)| a / (s * s)).sum();
    let scale = if even > 0.0 { (wanted / even).sqrt() } else { 1.0 };
    sizes.iter().map(|s| s * scale).collect()
}

const LINE_DEG: f64 = 60.0; // faces meeting sharper than this make a crease the rows should follow

/// Crease and border lines of the worked copy (the add-on's `_features`, simpler): a point on exactly two line edges
/// that run on from each other is pinned along them (1); where lines meet or turn sharply it is a fixed corner (2).
fn feature_lines(points: &[f64], tris: &[i64], _normals: &[f64]) -> (Vec<i64>, Vec<f64>) {
    let n = points.len() / 3;
    let face_normal = |t: &[i64]| unit(cross(sub(at(points, t[1] as usize), at(points, t[0] as usize)), sub(at(points, t[2] as usize), at(points, t[0] as usize))));
    let mut faces_at: HashMap<(i64, i64), Vec<V3>> = HashMap::new();
    for t in tris.chunks_exact(3) {
        let fnrm = face_normal(t);
        for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            faces_at.entry((x.min(y), x.max(y))).or_default().push(fnrm);
        }
    }
    let limit = LINE_DEG.to_radians().cos();
    let mut lines: Vec<Vec<V3>> = vec![Vec::new(); n];
    let mut keys: Vec<_> = faces_at.keys().copied().collect();
    keys.sort_unstable();
    for (x, y) in keys {
        let f = &faces_at[&(x, y)];
        let line = f.len() == 1 || (f.len() == 2 && dot(f[0], f[1]) < limit);
        if line {
            let d = unit(sub(at(points, y as usize), at(points, x as usize)));
            lines[x as usize].push(d);
            lines[y as usize].push(d);
        }
    }
    let mut pinned = vec![0i64; n];
    let mut along = vec![0.0; n * 3];
    for v in 0..n {
        match lines[v].len() {
            0 | 1 => {}
            2 if dot(lines[v][0], lines[v][1]).abs() > 0.7 => {
                pinned[v] = 1;
                along[v * 3..v * 3 + 3].copy_from_slice(&lines[v][0]);
            }
            _ => pinned[v] = 2,
        }
    }
    (pinned, along)
}

fn vertex_normals(points: &[f64], tris: &[i64]) -> Vec<f64> {
    let mut n = vec![0.0; points.len()];
    for t in tris.chunks_exact(3) {
        let (a, b, c) = (at(points, t[0] as usize), at(points, t[1] as usize), at(points, t[2] as usize));
        let face = cross(sub(b, a), sub(c, a)); // area-weighted
        for &v in t {
            for k in 0..3 {
                n[v as usize * 3 + k] += face[k];
            }
        }
    }
    n.chunks_exact(3).flat_map(|c| unit([c[0], c[1], c[2]])).collect()
}

/// Pair neighbours facing the same way (in a seeded order) into one coarser point each.
fn coarsen(level: &mut Level, random: &mut Random) -> Option<Level> {
    let n = level.points.len() / 3;
    let mut order: Vec<usize> = (0..level.a.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, (random.next() % (i as u64 + 1)) as usize);
    }
    let mut parent = vec![u32::MAX; n];
    let mut count = 0u32;
    for e in order {
        let (x, y) = (level.a[e] as usize, level.b[e] as usize);
        if parent[x] == u32::MAX && parent[y] == u32::MAX && dot(at(&level.normals, x), at(&level.normals, y)) > 0.0 {
            parent[x] = count;
            parent[y] = count;
            count += 1;
        }
    }
    for p in parent.iter_mut().filter(|p| **p == u32::MAX) {
        *p = count;
        count += 1;
    }
    let m = count as usize;
    if m * 10 > n * 9 {
        return None; // hardly coarser: stop
    }
    let (mut points, mut normals, mut weight) = (vec![0.0; m * 3], vec![0.0; m * 3], vec![0.0; m]);
    for v in 0..n {
        let c = parent[v] as usize;
        weight[c] += 1.0;
        for k in 0..3 {
            points[c * 3 + k] += level.points[v * 3 + k];
            normals[c * 3 + k] += level.normals[v * 3 + k];
        }
    }
    for c in 0..m {
        for k in 0..3 {
            points[c * 3 + k] /= weight[c];
        }
        let u = unit(at(&normals, c));
        normals[c * 3..c * 3 + 3].copy_from_slice(&u);
    }
    let mut e: Vec<(i64, i64)> = level.a.iter().zip(&level.b).map(|(&x, &y)| (parent[x as usize] as i64, parent[y as usize] as i64)).filter(|(x, y)| x != y).map(|(x, y)| (x.min(y), x.max(y))).collect();
    e.sort_unstable();
    e.dedup();
    let (a, b) = e.into_iter().unzip();
    // A coarse point is as pinned as its most pinned child, and runs along that child's line; its size is their mean.
    let (mut pinned, mut along, mut sizes) = (vec![0i64; m], vec![0.0; m * 3], vec![0.0; m]);
    for v in 0..n {
        let c = parent[v] as usize;
        sizes[c] += level.sizes[v] / weight[c];
        if level.pinned[v] > pinned[c] {
            pinned[c] = level.pinned[v];
            along[c * 3..c * 3 + 3].copy_from_slice(&level.along[v * 3..v * 3 + 3]);
        }
    }
    level.parent = parent;
    Some(Level { points, normals, a, b, parent: vec![], pinned, along, sizes })
}

fn nearest_grid(o: V3, q: V3, t: V3, s: f64, target: V3) -> V3 {
    let d = sub(target, o);
    let (u, w) = ((dot(d, q) / s).round_ties_even(), (dot(d, t) / s).round_ties_even());
    [0, 1, 2].map(|k| o[k] + q[k] * u * s + t[k] * w * s)
}

/// `_place`: onto the point's tangent plane, then the grid corner nearest the point.
fn place(p: V3, n: V3, q: V3, s: f64, o: V3) -> V3 {
    let lift = dot(sub(o, p), n);
    let on = [o[0] - lift * n[0], o[1] - lift * n[1], o[2] - lift * n[2]];
    nearest_grid(on, q, cross(n, q), s, p)
}

/// Rows of quads over one part (closed or open, triangles only) with cells of about `cell`: (points, indptr, flat),
/// mostly quads, with some triangles and faces of up to 8 sides where the field has poles.
/// None when the part is too big to work on at that cell size.
pub fn field_polygons(points: &[f64], tris: &[i64], cell: f64, seed: u64) -> Option<(Vec<f64>, Vec<i64>, Vec<i64>)> {
    let (fine_points, fine_tris) = split_until(points.to_vec(), tris.to_vec(), SPLIT_SHARE * SMALLEST * cell)?;
    if fine_points.len() / 3 > MOST_POINTS {
        return None;
    }
    let mut random = Random(seed ^ 0x5EED);
    let (a, b) = edges_of(&fine_tris);
    let normals = vertex_normals(&fine_points, &fine_tris);
    let (pinned, along) = feature_lines(&fine_points, &fine_tris, &normals);
    let sizes = cell_sizes(&fine_points, &fine_tris, &normals, &a, &b, cell);
    let mut levels = vec![Level { points: fine_points, normals, a, b, parent: vec![], pinned, along, sizes }];
    while levels.last().unwrap().points.len() / 3 > COARSEST {
        let last = levels.last_mut().unwrap();
        match coarsen(last, &mut random) {
            Some(next) => levels.push(next),
            None => break,
        }
    }
    // The direction field, coarse to fine; each level starts from its parents' directions. Kept per level for the grid.
    let mut qs: Vec<Vec<f64>> = vec![Vec::new(); levels.len()];
    for index in (0..levels.len()).rev() {
        let level = &levels[index];
        let n = level.points.len() / 3;
        let mut here: Vec<f64> = (0..n).flat_map(|v| {
            let from = if index + 1 < levels.len() { at(&qs[index + 1], level.parent[v] as usize) } else { random.unit_ish() };
            let from = if level.pinned[v] == 1 { at(&level.along, v) } else { from };
            unit(tangent(from, at(&level.normals, v)))
        }).collect();
        for _ in 0..(if index == 0 { ORIENT_SWEEPS / 2 } else { ORIENT_SWEEPS }) {
            field::orient_sweep(&mut here, &level.normals, &level.pinned, &level.a, &level.b, &random.colors(n));
        }
        qs[index] = here;
    }
    // The grid's positions, coarse to fine.
    let mut o: Vec<f64> = Vec::new();
    for index in (0..levels.len()).rev() {
        let level = &levels[index];
        let n = level.points.len() / 3;
        let pinned = &level.pinned;
        let qv = &qs[index];
        o = (0..n).flat_map(|v| {
            let start = if index + 1 < levels.len() { at(&o, level.parent[v] as usize) } else { at(&level.points, v) };
            place(at(&level.points, v), at(&level.normals, v), at(qv, v), level.sizes[v], start)
        }).collect();
        let view = field::Level { points: &level.points, normals: &level.normals, q: qv, sizes: &level.sizes, pinned };
        let mut none: Vec<f64> = Vec::new();
        for _ in 0..(if index == 0 { POSITION_SWEEPS / 2 } else { POSITION_SWEEPS }) {
            field::position_sweep(&view, &mut o, &mut none, &level.a, &level.b, &random.colors(n), false, SETTLED, ROBUST, true);
        }
        let mut misfit = vec![1.0; n];
        for _ in 0..(if index == 0 { 6 } else if index == 1 { 16 } else { 4 }) {
            field::position_sweep(&view, &mut o, &mut misfit, &level.a, &level.b, &random.colors(n), true, SETTLED, ROBUST, true);
        }
    }
    let fine = &levels[0];
    // Corners: points joined where their grid corners agree.
    let n = fine.points.len() / 3;
    let mut sets = crate::topo::Dsu::new(n);
    for (&x, &y) in fine.a.iter().zip(&fine.b) {
        let d = sub(at(&o, x as usize), at(&o, y as usize));
        if dot(d, d).sqrt() < SAME_CORNER * fine.sizes[x as usize].min(fine.sizes[y as usize]) {
            sets.union(x as u32, y as u32);
        }
    }
    // Points with the same grid corner that the surface doesn't join directly (a corner's region pinched in two) are
    // still one corner: join regions whose corners coincide, on the same side of the surface.
    {
        let mut sum: HashMap<u32, (V3, V3, f64, f64)> = HashMap::new();
        for v in 0..n {
            let e = sum.entry(sets.find(v as u32)).or_insert(([0.0; 3], [0.0; 3], 0.0, f64::INFINITY));
            let (p, nv) = (at(&o, v), at(&fine.normals, v));
            for k in 0..3 {
                e.0[k] += p[k];
                e.1[k] += nv[k];
            }
            e.2 += 1.0;
            e.3 = e.3.min(fine.sizes[v]);
        }
        let mut roots: Vec<(u32, V3, V3, f64)> = sum.into_iter().map(|(r, (p, nv, w, size))| (r, p.map(|x| x / w), unit(nv), size)).collect();
        roots.sort_unstable_by_key(|x| x.0);
        let key = |p: V3| p.map(|x| (x / cell).floor() as i64);
        let mut buckets: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for (i, r) in roots.iter().enumerate() {
            buckets.entry(key(r.1)).or_default().push(i);
        }
        for (i, r) in roots.iter().enumerate() {
            let k = key(r.1);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        for &j in buckets.get(&[k[0] + dx, k[1] + dy, k[2] + dz]).map_or(&[][..], |v| &v[..]) {
                            let d = sub(roots[j].1, r.1);
                            if j > i && dot(d, d).sqrt() < SAME_CORNER * r.3.min(roots[j].3) && dot(roots[j].2, r.2) > 0.5 {
                                sets.union(r.0, roots[j].0);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut label = vec![u32::MAX; n];
    let (mut out_points, mut weight): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for v in 0..n {
        let root = sets.find(v as u32) as usize;
        if label[root] == u32::MAX {
            label[root] = weight.len() as u32;
            weight.push(0.0);
            out_points.extend([0.0; 3]);
        }
        let l = label[root] as usize;
        label[v] = label[root];
        weight[l] += 1.0;
        let p = at(&o, v);
        for k in 0..3 {
            out_points[l * 3 + k] += p[k];
        }
    }
    for (l, w) in weight.iter().enumerate() {
        for k in 0..3 {
            out_points[l * 3 + k] /= w;
        }
    }
    // The corners' graph: two corners are linked when the surface joins their regions and their grid points lie one
    // cell apart along the field (Instant Meshes' extraction). Diagonal neighbours are left out.
    let count = weight.len();
    let mut frame = vec![([0.0; 3], [0.0; 3]); count]; // (normal, field direction) per corner
    let mut region_size = vec![0.0; count];
    for v in 0..n {
        let l = label[v] as usize;
        region_size[l] += fine.sizes[v] / weight[l];
        let nv = at(&fine.normals, v);
        frame[l].0 = [frame[l].0[0] + nv[0], frame[l].0[1] + nv[1], frame[l].0[2] + nv[2]];
        if frame[l].1 == [0.0; 3] {
            frame[l].1 = at(&qs[0], v);
        }
    }
    for f in frame.iter_mut() {
        f.0 = unit(f.0);
        f.1 = unit(tangent(f.1, f.0));
    }
    let mut links: Vec<(u32, u32)> = fine.a.iter().zip(&fine.b).map(|(&x, &y)| (label[x as usize], label[y as usize])).filter(|(x, y)| x != y).map(|(x, y)| (x.min(y), x.max(y))).collect();
    links.sort_unstable();
    links.dedup();
    let mut around: Vec<Vec<(f64, u32)>> = vec![Vec::new(); count];
    for (x, y) in links {
        let d = sub(at(&out_points, y as usize), at(&out_points, x as usize));
        // In each end's own frame, the offset in whole cells: a link is one cell straight along the field, (1, 0) or
        // (0, 1) up to sign, seen from at least one end (near a pole the two ends' frames disagree).
        let mut straight = false;
        let mut angle = [0.0; 2];
        for (side, (from, sign)) in [(x, 1.0), (y, -1.0)].into_iter().enumerate() {
            let (nv, q) = frame[from as usize];
            let dd = d.map(|c| c * sign);
            let (u, w) = (dot(dd, q), dot(dd, cross(nv, q)));
            let size = region_size[from as usize];
            straight |= (u / size).round().abs() + (w / size).round().abs() == 1.0;
            angle[side] = w.atan2(u);
        }
        if straight {
            around[x as usize].push((angle[0], y));
            around[y as usize].push((angle[1], x));
        }
    }
    for a in around.iter_mut() {
        a.sort_by(|p, q| p.0.total_cmp(&q.0));
    }
    // Faces: walk each link with the face on its left (counter-clockwise round the normal).
    let mut used: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let (mut indptr, mut flat): (Vec<i64>, Vec<i64>) = (vec![0i64], Vec::new());
    for a in 0..count as u32 {
        for &(_, b) in &around[a as usize] {
            if used.contains(&(a, b)) {
                continue;
            }
            let mut face = vec![a];
            let (mut from, mut to) = (a, b);
            let mut closed = false;
            for _ in 0..10 {
                used.insert((from, to));
                let ring = &around[to as usize];
                let Some(k) = ring.iter().position(|x| x.1 == from) else { break };
                let next = ring[(k + ring.len() - 1) % ring.len()].1; // the link just clockwise of the way back
                from = to;
                to = next;
                if from == a && to == b {
                    closed = true;
                    break;
                }
                face.push(from);
            }
            if closed && (3..=8).contains(&face.len()) {
                let mut sorted = face.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted.len() == face.len() {
                    flat.extend(face.iter().map(|&v| v as i64));
                    indptr.push(flat.len() as i64);
                }
            }
        }
    }
    fill_small_holes(&mut indptr, &mut flat, &fine_tris, &label);
    Some((out_points, indptr, flat))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An icosahedron split three times (1280 triangles) on the unit sphere.
    fn sphere() -> (Vec<f64>, Vec<i64>) {
        let t = (1.0 + 5f64.sqrt()) / 2.0;
        let mut points: Vec<f64> = [[-1., t, 0.], [1., t, 0.], [-1., -t, 0.], [1., -t, 0.], [0., -1., t], [0., 1., t], [0., -1., -t], [0., 1., -t], [t, 0., -1.], [t, 0., 1.], [-t, 0., -1.], [-t, 0., 1.]].iter().flat_map(|p| unit(*p)).collect();
        let mut tris: Vec<i64> = vec![0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1];
        for _ in 0..3 {
            let mut middle: HashMap<(i64, i64), i64> = HashMap::new();
            let mut out = Vec::new();
            let mut mid = |x: i64, y: i64, points: &mut Vec<f64>| *middle.entry((x.min(y), x.max(y))).or_insert_with(|| {
                let m = unit([0, 1, 2].map(|k| at(points, x as usize)[k] + at(points, y as usize)[k]));
                points.extend(m);
                (points.len() / 3 - 1) as i64
            });
            for c in tris.chunks_exact(3) {
                let (ab, bc, ca) = (mid(c[0], c[1], &mut points), mid(c[1], c[2], &mut points), mid(c[2], c[0], &mut points));
                out.extend([c[0], ab, ca, ab, c[1], bc, ca, bc, c[2], ab, bc, ca]);
            }
            tris = out;
        }
        (points, tris)
    }

    #[test]
    fn a_sphere_gets_closed_rows_of_mostly_regular_quads() {
        let (points, tris) = sphere();
        let (quad_points, indptr, flat) = field_quads(&points, &tris, 0.15, 1).expect("rows");
        let faces = indptr.len() - 1;
        let summary = crate::topo::summary(&crate::topo::Faces { indptr: &indptr, flat: &flat });
        assert!(faces > 200, "faces {faces}");
        assert_eq!(summary.quads as usize, faces, "all quads");
        assert_eq!(summary.boundary_edges, 0, "closed");
        let mut valence = vec![0usize; quad_points.len() / 3];
        flat.iter().for_each(|&v| valence[v as usize] += 1);
        let used: Vec<usize> = valence.into_iter().filter(|&n| n > 0).collect();
        let irregular = used.iter().filter(|&&n| n != 4).count() as f64 / used.len() as f64;
        eprintln!("sphere: {faces} quads, irregular share {irregular:.3}");
        assert!(irregular < 0.15, "irregular share {irregular}");
    }
}
