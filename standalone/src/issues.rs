// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Checks on a result and the Issue Finder's spots (`triadq/checks.py`, `triadq/issues.py`): faces on broken edges,
//! bent or concave quads, slivers and poles, grouped into spots that touch, worst first.

use crate::io::Mesh;
use crate::{geom, near, topo};
use serde_json::{json, Value};

const SLIVER_ANGLE: f64 = 10.0; // degrees: a corner sharper than this makes a sliver
const SLIVER_ASPECT: f64 = 8.0; // longest side over shortest side
const POLE_HIGH: usize = 7; // edges at an inner vertex of quads
const POLE_HIGH_TRIS: usize = 9; // the same where triangles meet
const POLE_LOW: usize = 2;

pub const KINDS: [&str; 4] = ["broken", "bent", "sliver", "pole"];
const WEIGHTS: [f64; 4] = [10.0, 3.0, 1.0, 0.5]; // as in issues.WEIGHTS; broken = on a bad edge, flipped or duplicate
const SERIOUS: f64 = 3.0; // kinds weighing this much or more are defects ("bad faces")

pub struct Check {
    pub faces: usize,
    pub quads: usize,
    pub tris: usize,
    pub bad: usize,
    pub non_manifold_edges: i64,
    pub flipped_edges: i64,
    pub duplicate_faces: i64,
    pub holes: i64,
    pub poles: i64,
    pub spots: Vec<Spot>,
}

pub struct Spot {
    pub score: f64,
    pub center: [f64; 3],
    pub radius: f64,
    pub faces: Vec<usize>,
    pub kinds: [usize; 4],
}

fn point(points: &[f64], v: i64) -> [f64; 3] {
    [points[v as usize * 3], points[v as usize * 3 + 1], points[v as usize * 3 + 2]]
}

pub fn check(mesh: &Mesh) -> Check {
    let faces = topo::Faces { indptr: &mesh.indptr, flat: &mesh.flat };
    let nf = mesh.faces();
    let summary = topo::summary(&faces);
    let (broken, nm, flipped, dups, holes, _) = near::mesh_checks(&faces, &vec![0; nf]);
    let span = {
        let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for p in mesh.points.chunks_exact(3) {
            for c in 0..3 {
                low[c] = low[c].min(p[c]);
                high[c] = high[c].max(p[c]);
            }
        }
        (0..3).map(|c| (high[c] - low[c]).powi(2)).sum::<f64>()
    };
    let epsilon = (span * 1e-14).max(1e-20);
    let warp = crate::retopo::MAX_QUAD_WARP_DEGREES.to_radians().cos();
    let mut flags = vec![[false; 4]; nf];
    // Edges per vertex, and whether it is on a border, for poles.
    let nv = mesh.points.len() / 3;
    let mut neighbours: Vec<Vec<i64>> = vec![Vec::new(); nv];
    let mut quad_corner = vec![false; nv];
    for f in 0..nf {
        let face = mesh.face(f);
        for (k, &v) in face.iter().enumerate() {
            neighbours[v as usize].push(face[(k + 1) % face.len()]);
            neighbours[v as usize].push(face[(k + face.len() - 1) % face.len()]);
            quad_corner[v as usize] |= face.len() == 4;
        }
    }
    let pole: Vec<bool> = neighbours.iter_mut().enumerate().map(|(v, n)| {
        let around = n.len() / 2; // corners: faces at v
        n.sort_unstable();
        n.dedup();
        let inner = n.len() == around; // a closed fan has as many edges as faces
        let high = if quad_corner[v] { POLE_HIGH } else { POLE_HIGH_TRIS };
        inner && around > 0 && (n.len() >= high || n.len() <= POLE_LOW)
    }).collect();
    for f in 0..nf {
        let face = mesh.face(f);
        flags[f][0] = broken[f];
        if face.len() == 4 {
            let q = [0, 1, 2, 3].map(|k| point(&mesh.points, face[k]));
            flags[f][1] = !geom::quad_valid(&q, epsilon, warp);
        }
        let ps: Vec<[f64; 3]> = face.iter().map(|&v| point(&mesh.points, v)).collect();
        let n = ps.len();
        let side = |k: usize| {
            let (a, b) = (ps[k], ps[(k + 1) % n]);
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        let lengths: Vec<f64> = (0..n).map(side).collect();
        let (short, long) = lengths.iter().fold((f64::INFINITY, 0.0f64), |(s, l), &x| (s.min(x), l.max(x)));
        let sharpest = (0..n).map(|k| {
            let (a, b, c) = (ps[(k + n - 1) % n], ps[k], ps[(k + 1) % n]);
            let u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            let w = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
            let d = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / ((u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt() * (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt()).max(1e-30);
            d.clamp(-1.0, 1.0).acos().to_degrees()
        }).fold(180.0, f64::min);
        flags[f][2] = sharpest < SLIVER_ANGLE || long > SLIVER_ASPECT * short;
        flags[f][3] = face.iter().any(|&v| pole[v as usize]);
    }
    let bad = flags.iter().filter(|k| (0..4).any(|i| k[i] && WEIGHTS[i] >= SERIOUS)).count();
    // Spots: flagged faces that share a vertex.
    let mut sets = topo::Dsu::new(nf);
    let mut owner = vec![usize::MAX; nv];
    for f in 0..nf {
        if !flags[f].iter().any(|&x| x) {
            continue;
        }
        for &v in mesh.face(f) {
            match owner[v as usize] {
                usize::MAX => owner[v as usize] = f,
                g => sets.union(f as u32, g as u32),
            }
        }
    }
    let mut groups: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
    for f in 0..nf {
        if flags[f].iter().any(|&x| x) {
            groups.entry(sets.find(f as u32)).or_default().push(f);
        }
    }
    let mut spots: Vec<Spot> = groups.into_values().map(|faces| {
        let mut kinds = [0usize; 4];
        let mut center = [0.0; 3];
        let mut corners = 0.0;
        for &f in &faces {
            for i in 0..4 {
                kinds[i] += flags[f][i] as usize;
            }
            for &v in mesh.face(f) {
                let p = point(&mesh.points, v);
                for c in 0..3 {
                    center[c] += p[c];
                }
                corners += 1.0;
            }
        }
        center = center.map(|x| x / corners);
        let radius = faces.iter().flat_map(|&f| mesh.face(f).iter()).map(|&v| {
            let p = point(&mesh.points, v);
            ((p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) + (p[2] - center[2]).powi(2)).sqrt()
        }).fold(0.0, f64::max);
        let score = (0..4).filter(|&i| kinds[i] > 0).map(|i| WEIGHTS[i] * (1.0 + (kinds[i] as f64).log2())).sum();
        Spot { score, center, radius, faces, kinds }
    }).collect();
    spots.sort_by(|a, b| b.score.total_cmp(&a.score));
    Check {
        faces: nf,
        quads: summary.quads as usize,
        tris: (0..nf).filter(|&f| mesh.face(f).len() == 3).count(),
        bad,
        non_manifold_edges: nm,
        flipped_edges: flipped,
        duplicate_faces: dups,
        holes,
        poles: pole.iter().filter(|&&p| p).count() as i64,
        spots,
    }
}

impl Check {
    pub fn to_json(&self, most_spots: usize) -> Value {
        let spots: Vec<Value> = self.spots.iter().take(most_spots).enumerate().map(|(rank, s)| {
            let kinds: serde_json::Map<String, Value> = KINDS.iter().zip(s.kinds).filter(|(_, n)| *n > 0).map(|(k, n)| (k.to_string(), json!(n))).collect();
            json!({"rank": rank + 1, "score": (s.score * 100.0).round() / 100.0, "center": s.center, "radius": s.radius, "kinds": kinds, "faces": s.faces})
        }).collect();
        json!({
            "faces": self.faces, "quads": self.quads, "tris": self.tris, "bad_faces": self.bad,
            "non_manifold_edges": self.non_manifold_edges, "flipped_edges": self.flipped_edges,
            "duplicate_faces": self.duplicate_faces, "holes": self.holes, "poles": self.poles,
            "spot_count": self.spots.len(), "spots": spots,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bow_tie_quad_is_bad_and_makes_a_spot() {
        let mesh = Mesh {
            points: vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 2., 0., 0., 2., 1., 0.],
            indptr: vec![0, 4, 8],
            flat: vec![0, 1, 2, 3, 1, 4, 2, 5], // the second quad crosses itself
            material: vec![0, 0],
            ..Default::default()
        };
        let c = check(&mesh);
        assert_eq!(c.bad, 1);
        assert_eq!(c.spots.len(), 1);
        assert_eq!(c.spots[0].kinds[1], 1);
    }
}
