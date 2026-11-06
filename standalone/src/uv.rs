// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! The source's UVs carried over to a result. Each result corner takes the UV of the nearest point on the source. A
//! face whose corners land on different UV islands (or far apart in UV for their size in 3D, as across a wrapped seam)
//! takes the island under its centre, and its other corners are looked up again on that island only; a corner that
//! still jumps is placed by the centre triangle's own UV slope. UVs are per corner, so a corner on a seam gets one UV per
//! face.

use crate::export::triangulate;
use crate::io::Mesh;
use crate::near::{nearest_faces, FaceTree};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn at(points: &[f64], v: i64) -> V3 {
    let i = v as usize * 3;
    [points[i], points[i + 1], points[i + 2]]
}

/// Barycentric weights of `p` (on or near the triangle's plane) in triangle a, b, c. A triangle without area gives a.
pub fn barycentric(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let (e1, e2, d) = (sub(b, a), sub(c, a), sub(p, a));
    let (d11, d12, d22, d1, d2) = (dot(e1, e1), dot(e1, e2), dot(e2, e2), dot(d, e1), dot(d, e2));
    let under = d11 * d22 - d12 * d12;
    if under.abs() <= 1e-30 * (d11 * d22).max(1e-300) {
        return [1.0, 0.0, 0.0];
    }
    let v = (d22 * d1 - d12 * d2) / under;
    let w = (d11 * d2 - d12 * d1) / under;
    [1.0 - v - w, v, w]
}

/// The source cut in triangles with their UV islands: triangles joined across edges whose UVs match on both sides.
pub struct Islands {
    pub tris: Mesh,
    pub island: Vec<u32>,
    pub scale: Vec<f64>, // per island, UV length per unit of 3D length
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

impl Islands {
    pub fn new(source: &Mesh) -> Islands {
        let tris = triangulate(source);
        let n = tris.faces();
        let same = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5;
        // Every edge as (low vertex, high vertex, triangle, UV at low, UV at high), sorted so shared edges sit together.
        let mut edges: Vec<(i64, i64, u32, [f64; 2], [f64; 2])> = Vec::with_capacity(n * 3);
        for t in 0..n {
            for i in 0..3 {
                let (j, k) = (3 * t + i, 3 * t + (i + 1) % 3);
                let (a, b) = (tris.flat[j], tris.flat[k]);
                let (ua, ub) = (tris.uv[j], tris.uv[k]);
                edges.push(if a < b { (a, b, t as u32, ua, ub) } else { (b, a, t as u32, ub, ua) });
            }
        }
        edges.par_sort_unstable_by_key(|e| (e.0, e.1));
        let mut parent: Vec<u32> = (0..n as u32).collect();
        let mut start = 0;
        while start < edges.len() {
            let mut end = start + 1;
            while end < edges.len() && (edges[end].0, edges[end].1) == (edges[start].0, edges[start].1) {
                end += 1;
            }
            for x in start..end {
                for y in x + 1..end {
                    let (p, q) = (&edges[x], &edges[y]);
                    if tris.material[p.2 as usize] == tris.material[q.2 as usize] && same(p.3, q.3) && same(p.4, q.4) {
                        let (a, b) = (find(&mut parent, p.2), find(&mut parent, q.2));
                        parent[a.max(b) as usize] = a.min(b);
                    }
                }
            }
            start = end;
        }
        let mut number: HashMap<u32, u32> = HashMap::new();
        let island: Vec<u32> = (0..n as u32).map(|t| {
            let root = find(&mut parent, t);
            let next = number.len() as u32;
            *number.entry(root).or_insert(next)
        }).collect();
        let mut sums = vec![(0.0f64, 0.0f64); number.len()];
        for t in 0..n {
            let s = &mut sums[island[t] as usize];
            s.0 += uv_area(&tris.uv[3 * t..3 * t + 3]);
            s.1 += area(&tris, t);
        }
        let scale = sums.iter().map(|&(uv, a)| if a > 0.0 { (uv / a).sqrt() } else { 0.0 }).collect();
        Islands { tris, island, scale }
    }

    /// The UV at `p`, a point on triangle `t`.
    pub fn uv_at(&self, t: usize, p: V3) -> [f64; 2] {
        let f = &self.tris.flat[3 * t..3 * t + 3];
        let w = barycentric(p, at(&self.tris.points, f[0]), at(&self.tris.points, f[1]), at(&self.tris.points, f[2]));
        let u = &self.tris.uv[3 * t..3 * t + 3];
        [w[0] * u[0][0] + w[1] * u[1][0] + w[2] * u[2][0], w[0] * u[0][1] + w[1] * u[1][1] + w[2] * u[2][1]]
    }

    /// The UV `d` away from `uv` along triangle `t`'s own UV slope (`d` is projected onto the triangle's plane).
    fn slide(&self, t: usize, uv: [f64; 2], d: V3) -> [f64; 2] {
        let f = &self.tris.flat[3 * t..3 * t + 3];
        let p0 = at(&self.tris.points, f[0]);
        let (e1, e2) = (sub(at(&self.tris.points, f[1]), p0), sub(at(&self.tris.points, f[2]), p0));
        let u = &self.tris.uv[3 * t..3 * t + 3];
        let (d11, d12, d22) = (dot(e1, e1), dot(e1, e2), dot(e2, e2));
        let under = d11 * d22 - d12 * d12;
        if under.abs() < 1e-30 {
            return uv;
        }
        let (r1, r2) = (dot(d, e1), dot(d, e2));
        let (a, b) = ((d22 * r1 - d12 * r2) / under, (d11 * r2 - d12 * r1) / under);
        [uv[0] + a * (u[1][0] - u[0][0]) + b * (u[2][0] - u[0][0]), uv[1] + a * (u[1][1] - u[0][1]) + b * (u[2][1] - u[0][1])]
    }
}

/// Area of a polygon in UV.
pub fn uv_area(uv: &[[f64; 2]]) -> f64 {
    let n = uv.len();
    (0..n).map(|i| uv[i][0] * uv[(i + 1) % n][1] - uv[(i + 1) % n][0] * uv[i][1]).sum::<f64>().abs() * 0.5
}

/// Area of face `f` in 3D.
pub fn area(mesh: &Mesh, f: usize) -> f64 {
    let face = mesh.face(f);
    let mut sum = [0.0; 3];
    for i in 0..face.len() {
        let c = cross(at(&mesh.points, face[i]), at(&mesh.points, face[(i + 1) % face.len()]));
        sum = [sum[0] + c[0], sum[1] + c[1], sum[2] + c[2]];
    }
    dot(sum, sum).sqrt() * 0.5
}

/// What `transfer` did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    pub faces: usize,
    pub seam_faces: usize, // faces whose corners were looked up again on one island
    pub off: usize,        // faces whose UV area is over 4 times off their island's (stretched or squashed)
}

impl Report {
    pub fn text(&self) -> String {
        let share = |n: usize| 100.0 * n as f64 / self.faces.max(1) as f64;
        format!("UVs kept: {} faces, {} across a UV seam ({:.1}%), {} stretched over 4x ({:.1}%)", self.faces, self.seam_faces, share(self.seam_faces), self.off, share(self.off))
    }
}

/// Give `result` the UVs of `source` (which must have UVs), and each face the material under its centre.
pub fn transfer(source: &Mesh, result: &mut Mesh) -> Report {
    let isl = Islands::new(source);
    let tree = FaceTree::new(&isl.tris.points, &isl.tris.indptr, &isl.tris.flat);
    let faces = result.faces();
    let corners = result.flat.len();
    let centres: Vec<V3> = (0..faces).map(|f| {
        let face = result.face(f);
        let mut c = [0.0; 3];
        for &v in face {
            let p = at(&result.points, v);
            c = [c[0] + p[0], c[1] + p[1], c[2] + p[2]];
        }
        c.map(|x| x / face.len() as f64)
    }).collect();
    let mut queries: Vec<f64> = result.flat.iter().flat_map(|&v| at(&result.points, v)).collect();
    queries.extend(centres.iter().flatten());
    let never = AtomicBool::new(false);
    let (_, hits) = nearest_faces(&tree, &queries, &never);
    let tri_of = |k: usize| hits[k].1.max(0) as usize;

    // First pass: each corner's UV, and whether its face needs a second look.
    let mut uv: Vec<[f64; 2]> = (0..corners).into_par_iter().map(|k| isl.uv_at(tri_of(k), hits[k].0)).collect();
    let centre_uv: Vec<[f64; 2]> = (0..faces).into_par_iter().map(|f| isl.uv_at(tri_of(corners + f), hits[corners + f].0)).collect();
    let jumps = |f: usize, k: usize, uvk: [f64; 2]| {
        let c = isl.island[tri_of(corners + f)] as usize;
        let reach = dot(sub(at(&result.points, result.flat[k]), centres[f]), sub(at(&result.points, result.flat[k]), centres[f])).sqrt();
        let gap = ((uvk[0] - centre_uv[f][0]).powi(2) + (uvk[1] - centre_uv[f][1]).powi(2)).sqrt();
        gap > 3.0 * isl.scale[c] * reach + 1e-9
    };
    let seam: Vec<bool> = (0..faces).into_par_iter().map(|f| {
        let c = isl.island[tri_of(corners + f)];
        (result.indptr[f] as usize..result.indptr[f + 1] as usize).any(|k| isl.island[tri_of(k)] != c || jumps(f, k, uv[k]))
    }).collect();

    // Second pass: those faces' corners again, on their centre's island only.
    let mut needed: Vec<u32> = (0..faces).filter(|&f| seam[f]).map(|f| isl.island[tri_of(corners + f)]).collect();
    needed.sort_unstable();
    needed.dedup();
    let mut members: HashMap<u32, Vec<u32>> = needed.iter().map(|&i| (i, Vec::new())).collect();
    for (t, i) in isl.island.iter().enumerate() {
        if let Some(m) = members.get_mut(i) {
            m.push(t as u32);
        }
    }
    let trees: HashMap<u32, (FaceTree, Vec<u32>)> = members.into_par_iter().map(|(i, list)| {
        let flat: Vec<i64> = list.iter().flat_map(|&t| isl.tris.face(t as usize).to_vec()).collect();
        let indptr: Vec<i64> = (0..=list.len() as i64).map(|k| 3 * k).collect();
        (i, (FaceTree::new(&isl.tris.points, &indptr, &flat), list))
    }).collect();
    let again: Vec<(usize, [f64; 2])> = (0..faces).into_par_iter().filter(|&f| seam[f]).flat_map_iter(|f| {
        let tc = tri_of(corners + f);
        let (tree, list) = &trees[&isl.island[tc]];
        let range = result.indptr[f] as usize..result.indptr[f + 1] as usize;
        range.map(|k| {
            let p = at(&result.points, result.flat[k]);
            let mut found = tree.nearest(p).map_or(centre_uv[f], |(q, t, _)| isl.uv_at(list[t as usize] as usize, q));
            if jumps(f, k, found) {
                found = isl.slide(tc, centre_uv[f], sub(p, centres[f]));
            }
            (k, found)
        }).collect::<Vec<_>>()
    }).collect();
    for (k, value) in again {
        uv[k] = value;
    }

    // Each face takes the material under its centre; faces with none share one new material, so maps can be baked.
    let mut material: Vec<i64> = (0..faces).map(|f| isl.tris.material[tri_of(corners + f)]).collect();
    if material.iter().any(|&m| m < 0) {
        let blank = result.materials.len() as i64;
        result.materials.push("material".into());
        material.iter_mut().filter(|m| **m < 0).for_each(|m| *m = blank);
    }
    result.material = material;
    result.uv = uv;

    let off = (0..faces).filter(|&f| {
        let s = isl.scale[isl.island[tri_of(corners + f)] as usize];
        let a = area(result, f) * s * s;
        let u = uv_area(&result.uv[result.indptr[f] as usize..result.indptr[f + 1] as usize]);
        a > 0.0 && (u > 4.0 * a || u < a / 4.0)
    }).count();
    Report { faces, seam_faces: seam.iter().filter(|&&s| s).count(), off }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane of n x n quads in x, z with UVs from x and z, split down the middle into two UV islands.
    fn plane(n: usize) -> Mesh {
        let mut m = Mesh { indptr: vec![0], materials: vec!["m".into()], ..Default::default() };
        for j in 0..=n {
            for i in 0..=n {
                m.points.extend([i as f64 / n as f64, 0.0, j as f64 / n as f64]);
            }
        }
        let id = |i: usize, j: usize| (j * (n + 1) + i) as i64;
        for j in 0..n {
            for i in 0..n {
                let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
                let right = i >= n / 2; // the right half sits higher up in UV: an island of its own
                m.flat.extend(corners.map(|(a, b)| id(a, b)));
                m.uv.extend(corners.map(|(a, b)| [a as f64 / n as f64 * 0.5, b as f64 / n as f64 * 0.45 + if right { 0.5 } else { 0.0 }]));
                m.indptr.push(m.flat.len() as i64);
                m.material.push(0);
            }
        }
        m
    }

    #[test]
    fn uvs_follow_the_source_and_seams_stay_clean() {
        let source = plane(8);
        assert_eq!(*Islands::new(&source).island.iter().max().unwrap(), 1);
        // A coarser plane whose middle column of quads straddles the seam.
        let mut result = plane(3);
        result.uv.clear();
        let report = transfer(&source, &mut result);
        assert_eq!(report.off, 0, "{}", report.text());
        assert_eq!(report.seam_faces, 3);
        // A corner away from the seam has exactly its source UV.
        assert!((result.uv[0][0] - 0.0).abs() < 1e-9 && (result.uv[0][1] - 0.0).abs() < 1e-9);
        // Every face's UVs stay on one island: within half a unit of each other.
        for f in 0..result.faces() {
            let vs: Vec<f64> = result.uv[result.indptr[f] as usize..result.indptr[f + 1] as usize].iter().map(|u| u[1]).collect();
            assert!(vs.iter().all(|&v| (v >= 0.5) == (vs[0] >= 0.5)), "face {f} mixes islands: {vs:?}");
        }
    }
}
