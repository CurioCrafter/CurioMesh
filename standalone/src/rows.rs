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
    Some((out, q_indptr, q_flat))
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
