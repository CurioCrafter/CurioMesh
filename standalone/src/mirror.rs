// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Mirror: solve one half of a mesh and reflect it, so both sides are identical and the seam is closed. A port of the
//! add-on's `symmetry.py` (`find_plane`, `cut_half`, `mirror_result`, `_clean_seam`).

use crate::io::Mesh;
use crate::near::FaceTree;
use std::collections::HashMap;

pub const AXES: [&str; 3] = ["X", "Y", "Z"];
const MATCH_TOLERANCE: f64 = 0.004; // a point matches when its mirror lies this near the surface (share of the size)
const ENOUGH: f64 = 0.90; // share of points that must match for Auto to call a mesh symmetric
const ENOUGH_CHOSEN: f64 = 0.70; // ... when the axis was picked by hand
const SAMPLES: usize = 4000;
const SNAP: f64 = 1e-5; // vertices this near the plane (share of the size) go exactly on it before cutting

#[derive(Clone, Debug)]
pub struct Plane {
    pub axis: usize,
    pub offset: f64,
    pub matched: f64, // share of the sampled points whose mirror lies on the surface
    pub worst: f64,   // 99th percentile distance of a point's mirror from the surface, as a share of the size
}

impl Plane {
    pub fn ok(&self, chosen: bool) -> bool {
        self.matched >= if chosen { ENOUGH_CHOSEN } else { ENOUGH }
    }
    pub fn text(&self) -> String {
        format!("{}: {:.0}% of the surface matches its mirror (99% within {:.1}% of the size)", AXES[self.axis], self.matched * 100.0, self.worst * 100.0)
    }
}

fn size(points: &[f64]) -> ([f64; 3], [f64; 3], f64) {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in points.chunks_exact(3) {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    let diag = (0..3).map(|c| (hi[c] - lo[c]).powi(2)).sum::<f64>().sqrt();
    (lo, hi, if diag > 0.0 { diag } else { 1.0 })
}

/// (matched share, 99th percentile distance, per sample: distance, the axis coordinate of the nearest point, the
/// mirrored axis coordinate).
fn score(tree: &FaceTree, samples: &[[f64; 3]], axis: usize, offset: f64, diag: f64) -> (f64, f64, Vec<(f64, f64, f64)>) {
    let found: Vec<(f64, f64, f64)> = samples.iter().map(|p| {
        let mut m = *p;
        m[axis] = 2.0 * offset - m[axis];
        match tree.nearest(m) {
            Some((at, _, d2)) => (d2.sqrt() / diag, at[axis], m[axis]),
            None => (1.0, 0.0, m[axis]),
        }
    }).collect();
    let matched = found.iter().filter(|f| f.0 < MATCH_TOLERANCE).count() as f64 / found.len().max(1) as f64;
    let mut d: Vec<f64> = found.iter().map(|f| f.0).collect();
    d.sort_by(f64::total_cmp);
    let worst = d.get(((d.len() as f64 - 1.0) * 0.99).round() as usize).copied().unwrap_or(0.0);
    (matched, worst, found)
}

/// The best mirror plane on each of `axes`, best first: tries the middle of the mesh and the origin, then nudges the
/// plane toward where the mirrored points land.
pub fn find_plane(mesh: &Mesh, axes: &[usize]) -> Vec<Plane> {
    let (lo, hi, diag) = size(&mesh.points);
    let tree = FaceTree::new(&mesh.points, &mesh.indptr, &mesh.flat);
    let count = mesh.points.len() / 3;
    let step = (count / SAMPLES).max(1);
    let samples: Vec<[f64; 3]> = (0..count).step_by(step).map(|v| [mesh.points[v * 3], mesh.points[v * 3 + 1], mesh.points[v * 3 + 2]]).collect();
    let mut planes: Vec<Plane> = axes.iter().map(|&axis| {
        let middle = (lo[axis] + hi[axis]) / 2.0;
        let mut offset = middle;
        if middle != 0.0 && score(&tree, &samples, axis, 0.0, diag).0 > score(&tree, &samples, axis, middle, diag).0 {
            offset = 0.0;
        }
        for _ in 0..2 {
            let (_, _, found) = score(&tree, &samples, axis, offset, diag);
            let used: Vec<&(f64, f64, f64)> = found.iter().filter(|f| f.0 < 3.0 * MATCH_TOLERANCE).collect();
            if used.len() < 16 {
                break;
            }
            offset += used.iter().map(|f| f.1 - f.2).sum::<f64>() / used.len() as f64 / 2.0;
        }
        let (matched, worst, _) = score(&tree, &samples, axis, offset, diag);
        Plane { axis, offset, matched, worst }
    }).collect();
    planes.sort_by(|a, b| b.matched.total_cmp(&a.matched));
    planes
}

/// The faces clipped to the side of the plane with larger coordinates (keep = 1) or smaller (keep = -1). New points
/// on the plane are exactly on it, and faces sharing a cut edge share its new point.
pub fn cut_half(mesh: &Mesh, axis: usize, offset: f64, keep: f64) -> Mesh {
    let mut points = mesh.points.clone();
    let diag = size(&points).2;
    let mut side: Vec<f64> = points.chunks_exact(3).map(|p| keep * (p[axis] - offset)).collect();
    for (v, s) in side.iter_mut().enumerate() {
        if s.abs() <= SNAP * diag {
            points[v * 3 + axis] = offset;
            *s = 0.0;
        }
    }
    let mut cuts: HashMap<(i64, i64), i64> = HashMap::new();
    let mut out = Mesh { indptr: vec![0], materials: mesh.materials.clone(), ..Default::default() };
    for f in 0..mesh.faces() {
        let face = mesh.face(f);
        let d: Vec<f64> = face.iter().map(|&v| side[v as usize]).collect();
        if d.iter().all(|&x| x < 0.0) || d.iter().all(|&x| x == 0.0) {
            continue;
        }
        let poly: Vec<i64> = if d.iter().all(|&x| x >= 0.0) {
            face.to_vec()
        } else {
            // Sutherland-Hodgman against one plane: the corners on the kept side, plus the crossings.
            let mut poly = Vec::new();
            for k in 0..face.len() {
                let (a, b) = (face[k], face[(k + 1) % face.len()]);
                let (da, db) = (side[a as usize], side[b as usize]);
                if da >= 0.0 {
                    poly.push(a);
                }
                if (da > 0.0 && db < 0.0) || (da < 0.0 && db > 0.0) {
                    let key = (a.min(b), a.max(b));
                    let made = *cuts.entry(key).or_insert_with(|| {
                        let t = da / (da - db);
                        let mut p = [0.0; 3];
                        for c in 0..3 {
                            p[c] = points[a as usize * 3 + c] + (points[b as usize * 3 + c] - points[a as usize * 3 + c]) * t;
                        }
                        p[axis] = offset;
                        points.extend_from_slice(&p);
                        (points.len() / 3 - 1) as i64
                    });
                    poly.push(made);
                }
            }
            poly
        };
        if poly.len() >= 3 {
            out.flat.extend(poly);
            out.indptr.push(out.flat.len() as i64);
            out.material.push(mesh.material[f]);
        }
    }
    out.points = points;
    crate::io::compact(&out)
}

fn edge_uses(mesh: &Mesh) -> HashMap<(i64, i64), usize> {
    let mut uses = HashMap::new();
    for f in 0..mesh.faces() {
        let face = mesh.face(f);
        for k in 0..face.len() {
            let (a, b) = (face[k], face[(k + 1) % face.len()]);
            *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    uses
}

/// The solved half plus its reflection. The half's open border near the plane is put exactly on it and shared by both
/// halves; corners that strayed over the plane go back; seam corners joined by a very short edge merge; and faces left
/// with no area, or lying wholly on the seam, go. Returns the mesh and how many corners sit on the seam.
pub fn mirror_result(half: &Mesh, axis: usize, offset: f64, keep: f64) -> (Mesh, usize) {
    let mut points = half.points.clone();
    let count = points.len() / 3;
    let size = size(&points).2;
    let mut near = vec![false; count];
    for ((a, b), n) in edge_uses(half) {
        if n == 1 {
            for v in [a, b] {
                if (points[v as usize * 3 + axis] - offset).abs() <= 0.02 * size {
                    near[v as usize] = true;
                }
            }
        }
    }
    let tiny = 1e-6 * size;
    for v in 0..count {
        let at = &mut points[v * 3 + axis];
        if near[v] {
            *at = offset;
            continue;
        }
        let side = *at - offset;
        if keep * side < 0.0 {
            *at = offset - side; // back over the plane: its mirror would overlap it
        }
        if (*at - offset).abs() < tiny {
            *at = offset + keep * tiny; // on the plane but not on the seam: it must not weld to its mirror
        }
    }
    // Merge seam corners joined by an edge under a tenth of the usual length.
    let mut lengths = Vec::new();
    let step = (half.faces() / 2000).max(1);
    for f in (0..half.faces()).step_by(step) {
        let face = half.face(f);
        for k in 0..face.len() {
            lengths.push(dist(&points, face[k], face[(k + 1) % face.len()]));
        }
    }
    lengths.sort_by(f64::total_cmp);
    let short = 0.1 * lengths.get(lengths.len() / 2).copied().unwrap_or(0.0);
    let mut parent: Vec<usize> = (0..count).collect();
    fn root(parent: &mut [usize], mut v: usize) -> usize {
        while parent[v] != v {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        v
    }
    for f in 0..half.faces() {
        let face = half.face(f);
        for k in 0..face.len() {
            let (a, b) = (face[k] as usize, face[(k + 1) % face.len()] as usize);
            if near[a] && near[b] && dist(&points, a as i64, b as i64) < short {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
    }
    let mut kept = Mesh { points, indptr: vec![0], materials: half.materials.clone(), ..Default::default() };
    for f in 0..half.faces() {
        let mut g: Vec<i64> = half.face(f).iter().map(|&v| root(&mut parent, v as usize) as i64).collect();
        g = (0..g.len()).filter(|&k| g[k] != g[(k + g.len() - 1) % g.len()]).map(|k| g[k]).collect();
        let mut unique = g.clone();
        unique.sort();
        unique.dedup();
        if unique.len() < 3 || g.iter().all(|&v| near[v as usize]) {
            continue;
        }
        kept.flat.extend(g);
        kept.indptr.push(kept.flat.len() as i64);
        kept.material.push(half.material[f]);
    }
    // Reflect: seam corners are shared, every other corner gets a mirrored twin.
    let (mut number, mut used_points, mut kept_near) = (vec![-1i64; count], Vec::new(), Vec::new());
    for v in kept.flat.iter_mut() {
        let old = *v as usize;
        if number[old] < 0 {
            number[old] = kept_near.len() as i64;
            used_points.extend_from_slice(&kept.points[old * 3..old * 3 + 3]);
            kept_near.push(near[old]);
        }
        *v = number[old];
    }
    kept.points = used_points;
    let n = kept.points.len() / 3;
    let mut twin = vec![0i64; n];
    let mut next = n as i64;
    let mut out = kept.clone();
    for v in 0..n {
        if kept_near[v] {
            twin[v] = v as i64;
        } else {
            twin[v] = next;
            next += 1;
            let mut p = [kept.points[v * 3], kept.points[v * 3 + 1], kept.points[v * 3 + 2]];
            p[axis] = 2.0 * offset - p[axis];
            out.points.extend_from_slice(&p);
        }
    }
    for f in 0..kept.faces() {
        out.flat.extend(kept.face(f).iter().rev().map(|&v| twin[v as usize]));
        out.indptr.push(out.flat.len() as i64);
        out.material.push(kept.material[f]);
    }
    (out, kept_near.iter().filter(|&&s| s).count())
}

fn dist(points: &[f64], a: i64, b: i64) -> f64 {
    (0..3).map(|c| (points[a as usize * 3 + c] - points[b as usize * 3 + c]).powi(2)).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x1 strip of quads across x = 0, cut and mirrored, comes back as the same strip with a shared seam.
    #[test]
    fn cut_and_mirror_a_strip() {
        let mesh = Mesh {
            points: vec![-1., 0., 0., 0., 0., 0., 1., 0., 0., -1., 1., 0., 0., 1., 0., 1., 1., 0.],
            indptr: vec![0, 4, 8],
            flat: vec![0, 1, 4, 3, 1, 2, 5, 4],
            material: vec![0, 0],
            materials: vec![],
        };
        let plane = &find_plane(&mesh, &[0])[0];
        assert!(plane.offset.abs() < 1e-9 && plane.ok(false), "{}", plane.text());
        let half = cut_half(&mesh, 0, 0.0, 1.0);
        assert_eq!(half.faces(), 1);
        let (whole, seam) = mirror_result(&half, 0, 0.0, 1.0);
        assert_eq!((whole.faces(), whole.points.len() / 3, seam), (2, 6, 2));
    }

    /// A quad crossing the plane is clipped at it.
    #[test]
    fn cut_clips_a_crossing_face() {
        let mesh = Mesh { points: vec![-1., 0., 0., 1., 0., 0., 1., 1., 0., -1., 1., 0.], indptr: vec![0, 4], flat: vec![0, 1, 2, 3], material: vec![0], materials: vec![] };
        let half = cut_half(&mesh, 0, 0.0, 1.0);
        assert_eq!(half.face(0).len(), 4);
        assert!(half.points.chunks_exact(3).all(|p| p[0] >= 0.0));
    }
}
