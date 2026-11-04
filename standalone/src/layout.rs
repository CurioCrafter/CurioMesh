// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Evening out the quads after closing, without moving the shape: spin edges where that leaves fewer poles (edge
//! rotation, Tarini et al. 2010, as `ChordReducer.turn_poles` does in the add-on), then slide corners toward their
//! neighbours' middle along the surface and put them back on the source (as `ChordReducer.relax` with `project`).

use crate::io::Mesh;
use crate::{geom, near};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

type V3 = [f64; 3];
const SPIN_PASSES: usize = 8;
const RELAX_PASSES: usize = 3;
const RELAX_STEP: f64 = 0.5; // share of the way to the neighbours' middle per pass
const BEND_DEG: f64 = 25.0; // a corner whose faces turn more than this sits on a bend or crease, and stays put
const MIN_JACOBIAN: f64 = 0.2; // a moved corner may not leave a quad less square than this (or than it was)

fn p(points: &[f64], v: i64) -> V3 {
    [points[v as usize * 3], points[v as usize * 3 + 1], points[v as usize * 3 + 2]]
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
    let n = dot(a, a).sqrt();
    if n > 1e-30 { [a[0] / n, a[1] / n, a[2] / n] } else { [0.0; 3] }
}
fn normal(q: &[V3; 4]) -> V3 {
    unit(cross(sub(q[2], q[0]), sub(q[3], q[1])))
}

/// `engine._min_scaled_jacobians` for one quad: 1 for a square, <= 0 when folded.
fn jacobian(q: &[V3; 4]) -> f64 {
    let n = normal(q);
    (0..4).map(|k| {
        let (a, b) = (sub(q[(k + 1) % 4], q[k]), sub(q[(k + 3) % 4], q[k]));
        let d = (dot(a, a) * dot(b, b)).sqrt();
        if d > 1e-30 { dot(cross(a, b), n) / d } else { -1.0 }
    }).fold(1.0, f64::min)
}

struct Limits {
    epsilon: f64,
    warp: f64,
}

impl Limits {
    fn of(points: &[f64]) -> Limits {
        let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for c in points.chunks_exact(3) {
            for k in 0..3 {
                low[k] = low[k].min(c[k]);
                high[k] = high[k].max(c[k]);
            }
        }
        let span = sub(high, low);
        Limits { epsilon: (dot(span, span) * 1e-14).max(1e-20), warp: crate::retopo::MAX_QUAD_WARP_DEGREES.to_radians().cos() }
    }
    fn ok(&self, q: &[V3; 4]) -> bool {
        geom::quad_valid(q, self.epsilon, self.warp)
    }
}

fn key(a: i64, b: i64) -> (i64, i64) {
    (a.min(b), a.max(b))
}

/// Spin edges between two quads where that brings corners closer to four edges. Corners don't move. Returns how many
/// edges were spun. Faces that aren't quads, border corners and material borders are left alone.
pub fn spin_edges(mesh: &mut Mesh) -> usize {
    let nf = mesh.faces();
    if (0..nf).any(|f| mesh.face(f).len() != 4) {
        return 0; // ponytail: all-quad output only, which is what closing gives; mixed meshes would need CSR edits
    }
    let limits = Limits::of(&mesh.points);
    let mut faces: Vec<[i64; 4]> = (0..nf).map(|f| mesh.face(f).try_into().unwrap()).collect();
    let mut edges: HashMap<(i64, i64), Vec<usize>> = HashMap::with_capacity(nf * 2);
    let mut valence = vec![0i64; mesh.points.len() / 3];
    for (f, q) in faces.iter().enumerate() {
        for k in 0..4 {
            edges.entry(key(q[k], q[(k + 1) % 4])).or_default().push(f);
            valence[q[k] as usize] += 1;
        }
    }
    let border: Vec<bool> = {
        let mut b = vec![false; valence.len()];
        for (&(x, y), users) in &edges {
            if users.len() != 2 {
                b[x as usize] = true;
                b[y as usize] = true;
            }
        }
        b
    };
    let energy = |v: i64, change: i64, valence: &[i64]| if border[v as usize] { 0 } else { (valence[v as usize] + change - 4).pow(2) };
    let mut spun = 0;
    for _ in 0..SPIN_PASSES {
        let mut list: Vec<(i64, i64)> = edges.iter().filter(|(_, u)| u.len() == 2).map(|(&e, _)| e).collect();
        list.sort_unstable(); // the same result every run
        let before = spun;
        for (x, y) in list {
            let Some(users) = edges.get(&(x, y)) else { continue };
            if users.len() != 2 || border[x as usize] || border[y as usize] {
                continue;
            }
            let (f, g) = (users[0], users[1]);
            if mesh.material[f] != mesh.material[g] {
                continue;
            }
            // f runs u -> v at corner i; g runs v -> u. Their outline is the hexagon v, f2, f3, u, g2, g3.
            let (qf, qg) = (faces[f], faces[g]);
            let Some(i) = (0..4).find(|&k| key(qf[k], qf[(k + 1) % 4]) == (x, y)) else { continue };
            let (u, v) = (qf[i], qf[(i + 1) % 4]);
            let Some(j) = (0..4).find(|&k| qg[k] == v && qg[(k + 1) % 4] == u) else { continue };
            let h = [v, qf[(i + 2) % 4], qf[(i + 3) % 4], u, qg[(j + 2) % 4], qg[(j + 3) % 4]];
            let mut sorted = h;
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w[0] == w[1]) {
                continue;
            }
            let now: i64 = h.iter().map(|&w| energy(w, 0, &valence)).sum();
            let mut best: Option<(i64, [[i64; 4]; 2], [usize; 2])> = None; // (energy after, quads, the corners that gain)
            for s in [1usize, 2] {
                let (a, b) = (s, s + 3); // the new diagonal h[a]-h[b]; h[0]-h[3] goes
                if edges.contains_key(&key(h[a], h[b])) {
                    continue;
                }
                let after: i64 = (0..6).map(|k| {
                    let change = if k == 0 || k == 3 { -1 } else if k == a || k == b { 1 } else { 0 };
                    energy(h[k], change, &valence)
                }).sum();
                let quads = [[h[a], h[(a + 1) % 6], h[(a + 2) % 6], h[b]], [h[b], h[(b + 1) % 6], h[(b + 2) % 6], h[a]]];
                if after < now && best.as_ref().is_none_or(|x| after < x.0) && quads.iter().all(|q| limits.ok(&q.map(|w| p(&mesh.points, w)))) {
                    best = Some((after, quads, [a, b]));
                }
            }
            let Some((_, quads, [a, b])) = best else { continue };
            for (slot, q) in [(f, quads[0]), (g, quads[1])] {
                for k in 0..4 {
                    let e = key(faces[slot][k], faces[slot][(k + 1) % 4]);
                    if let Some(list) = edges.get_mut(&e) {
                        list.retain(|&o| o != slot);
                        if list.is_empty() {
                            edges.remove(&e);
                        }
                    }
                }
                faces[slot] = q;
            }
            for slot in [f, g] {
                for k in 0..4 {
                    edges.entry(key(faces[slot][k], faces[slot][(k + 1) % 4])).or_default().push(slot);
                }
            }
            valence[h[0] as usize] -= 1;
            valence[h[3] as usize] -= 1;
            valence[h[a] as usize] += 1;
            valence[h[b] as usize] += 1;
            spun += 1;
        }
        if spun == before {
            break;
        }
    }
    mesh.flat = faces.iter().flatten().copied().collect();
    spun
}

/// Slide corners on smooth ground toward their neighbours' middle, put them back on `source`, and keep a move only if
/// every quad round the corner stays valid and about as square. Returns how many moves were kept.
pub fn relax(mesh: &mut Mesh, source: &near::FaceTree) -> usize {
    let nf = mesh.faces();
    let nv = mesh.points.len() / 3;
    let limits = Limits::of(&mesh.points);
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); nv];
    let mut neighbours: Vec<Vec<i64>> = vec![Vec::new(); nv];
    let mut edge_uses: HashMap<(i64, i64), usize> = HashMap::with_capacity(nf * 2);
    for f in 0..nf {
        let face = mesh.face(f);
        for k in 0..face.len() {
            let (a, b) = (face[k], face[(k + 1) % face.len()]);
            users[a as usize].push(f);
            neighbours[a as usize].push(b);
            neighbours[b as usize].push(a);
            *edge_uses.entry(key(a, b)).or_default() += 1;
        }
    }
    let mut fixed = vec![false; nv];
    for (&(a, b), &n) in &edge_uses {
        if n != 2 {
            fixed[a as usize] = true;
            fixed[b as usize] = true;
        }
    }
    let bend = BEND_DEG.to_radians().cos();
    let (indptr, flat) = (mesh.indptr.clone(), mesh.flat.clone()); // corners move below, faces don't
    let quad_at = |points: &[f64], f: usize| -> Option<[V3; 4]> {
        let face = &flat[indptr[f] as usize..indptr[f + 1] as usize];
        (face.len() == 4).then(|| [0, 1, 2, 3].map(|k| p(points, face[k])))
    };
    for v in 0..nv {
        neighbours[v].sort_unstable();
        neighbours[v].dedup();
        let materials = users[v].iter().map(|&f| mesh.material[f]);
        let normals: Vec<V3> = users[v].iter().filter_map(|&f| quad_at(&mesh.points, f)).map(|q| normal(&q)).collect();
        fixed[v] |= users[v].is_empty()
            || normals.len() != users[v].len()
            || materials.clone().any(|m| m != mesh.material[users[v][0]])
            || normals.iter().enumerate().any(|(k, n)| normals[..k].iter().any(|m| dot(*n, *m) < bend));
    }
    let stop = AtomicBool::new(false);
    let mut kept = 0;
    for _ in 0..RELAX_PASSES {
        let movable: Vec<usize> = (0..nv).filter(|&v| !fixed[v]).collect();
        let mut asks = Vec::with_capacity(movable.len() * 3);
        for &v in &movable {
            let here = p(&mesh.points, v as i64);
            let n = neighbours[v].len() as f64;
            let middle = neighbours[v].iter().fold([0.0; 3], |s, &w| {
                let q = p(&mesh.points, w);
                [s[0] + q[0] / n, s[1] + q[1] / n, s[2] + q[2] / n]
            });
            let up = unit(users[v].iter().filter_map(|&f| quad_at(&mesh.points, f)).fold([0.0; 3], |s, q| {
                let m = normal(&q);
                [s[0] + m[0], s[1] + m[1], s[2] + m[2]]
            }));
            let step = sub(middle, here);
            let along = sub(step, [up[0] * dot(step, up), up[1] * dot(step, up), up[2] * dot(step, up)]);
            asks.extend([0, 1, 2].map(|k| here[k] + RELAX_STEP * along[k]));
        }
        let (_, found) = near::nearest_faces(source, &asks, &stop);
        let before = kept;
        for (i, &v) in movable.iter().enumerate() {
            let to = found[i].0;
            let old = p(&mesh.points, v as i64);
            let quads_before: Vec<[V3; 4]> = users[v].iter().filter_map(|&f| quad_at(&mesh.points, f)).collect();
            mesh.points[v * 3..v * 3 + 3].copy_from_slice(&to);
            let fine = users[v].iter().zip(&quads_before).all(|(&f, was)| {
                let q = quad_at(&mesh.points, f).unwrap();
                limits.ok(&q) && jacobian(&q) >= jacobian(was).min(MIN_JACOBIAN) && dot(normal(&q), normal(was)) > bend
            });
            if fine {
                kept += 1;
            } else {
                mesh.points[v * 3..v * 3 + 3].copy_from_slice(&old);
            }
        }
        if kept == before {
            break;
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat 4x4 grid of quads with the middle corner's edges spun into a 3-pole/5-pole pair should spin back.
    #[test]
    fn spinning_takes_away_a_pole_pair_on_a_flat_grid() {
        let n = 5i64;
        let mut mesh = Mesh { indptr: vec![0], ..Default::default() };
        for j in 0..n {
            for i in 0..n {
                mesh.points.extend([i as f64, j as f64, 0.0]);
            }
        }
        let at = |i: i64, j: i64| j * n + i;
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                mesh.flat.extend([at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
                mesh.indptr.push(mesh.flat.len() as i64);
                mesh.material.push(0);
            }
        }
        let clean = mesh.flat.clone();
        // Spin the edge (2,2)-(2,1) between quads (1,1) and (2,1) once: the hexagon's other diagonal.
        let (f, g) = ((1 * 4 + 1) as usize, (1 * 4 + 2) as usize);
        let h = [at(2, 2), at(1, 2), at(1, 1), at(2, 1), at(3, 1), at(3, 2)];
        mesh.flat[f * 4..f * 4 + 4].copy_from_slice(&[h[1], h[2], h[3], h[4]]);
        mesh.flat[g * 4..g * 4 + 4].copy_from_slice(&[h[4], h[5], h[0], h[1]]);
        assert_ne!(mesh.flat, clean);
        assert!(spin_edges(&mut mesh) >= 1);
        let valence = |v: i64| mesh.flat.iter().filter(|&&w| w == v).count();
        assert!((0..n * n).filter(|&v| { let (i, j) = (v % n, v / n); i > 0 && j > 0 && i < n - 1 && j < n - 1 }).all(|v| valence(v) == 4));
    }

    #[test]
    fn relaxing_keeps_corners_on_the_source_plane() {
        let mut mesh = Mesh { points: vec![0., 0., 0., 2., 0., 0., 2., 2., 0., 0., 2., 0., 0.3, 1.4, 0., 1., 0., 0., 2., 1., 0., 1., 2., 0., 0., 1., 0.], indptr: vec![0], material: vec![0; 4], ..Default::default() };
        // A 2x2 grid whose middle corner (4) sits off centre.
        for q in [[0, 5, 4, 8], [5, 1, 6, 4], [4, 6, 2, 7], [8, 4, 7, 3]] {
            mesh.flat.extend(q);
            mesh.indptr.push(mesh.flat.len() as i64);
        }
        let plane = near::FaceTree::new(&[-5., -5., 0., 5., -5., 0., 5., 5., 0., -5., 5., 0.], &[0, 4], &[0, 1, 2, 3]);
        assert!(relax(&mut mesh, &plane) > 0);
        let middle = p(&mesh.points, 4);
        assert!(middle[2].abs() < 1e-6);
        assert!((middle[0] - 1.0).abs() < 0.5 && (middle[1] - 1.0).abs() < 0.3, "moved toward the middle: {middle:?}");
    }
}
