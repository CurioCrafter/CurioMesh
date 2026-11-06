// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Seam-aware quadric edge collapse (package K2): the rules `qem::run_seams` asks about UV seams. Byte-identical on
//! `main` and `standalone`; each branch's `qem.rs` (they differ) only calls in here.
//!
//! Hoppe 1996 (Progressive Meshes): half-edge collapses keep one endpoint, so a kept seam vertex keeps its source
//! per-side UVs exactly. Liu, Ferguson, Jacobson, Gingold 2017 (Seamless: seam-aware decimation): a collapse along a
//! seam must keep the seam's image in UV space on both sides. The rules here are written from those papers' ideas:
//! - a seam vertex goes only along a seam edge of its own chain, onto the other end's position;
//! - a vertex off the seams may go onto a seam vertex, taking the seam vertex's position;
//! - seam corners (seam valence other than 2, open border, a chain end from K1's runs) never go;
//! - a closed loop with no corner keeps at least 3 vertices, spread evenly by 3D length;
//! - UV-side check: the new seam segment stays within `seam_px` of the source UV polyline it replaces, on both sides.
//! Seam edges also get the border plane quadric in `qem.rs` (their `BORDER_WEIGHT` block), so the 3D line holds.

use crate::uvseam::{SeamUv, TEXTURE_PX_DEFAULT};
use std::collections::HashMap;

// Constants table.
pub const SEAM_PX: f64 = 0.5; // default UV-side tolerance, pixels at TEXTURE_PX_DEFAULT
pub const LOOP_KEEP: usize = 3; // a closed seam loop with no corner keeps at least this many vertices

/// The live seam edges of a reduction and what they replace.
pub struct Seams {
    near: Vec<Vec<u32>>,                      // live seam neighbours per vertex
    corner: Vec<bool>,                        // never goes (seam corners and loop anchors)
    runs: HashMap<(u32, u32), Vec<u32>>,      // live seam edge -> source run positions end to end (when K1's runs cover it)
    first: Vec<(u32, u32)>,                   // the source seam edges, for `reduced`
    uv_a: Vec<f64>,
    uv_b: Vec<f64>,
    run_vertex: Vec<u32>,
    tol: f64, // UV units
}

fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

impl Seams {
    /// `seam_pairs` are welded vertex pairs (2 per edge). None when there are no usable seam edges, so the caller runs
    /// exactly as without seams.
    pub fn new(points: &[f64], seam_pairs: &[u32], seam_uv: Option<&SeamUv>, seam_px: f64) -> Option<Seams> {
        let nv = points.len() / 3;
        let mut near: Vec<Vec<u32>> = vec![Vec::new(); nv];
        let mut first = Vec::new();
        for p in seam_pairs.chunks_exact(2) {
            let (a, b) = (p[0], p[1]);
            if a == b || a as usize >= nv || b as usize >= nv || near[a as usize].contains(&b) {
                continue;
            }
            near[a as usize].push(b);
            near[b as usize].push(a);
            first.push(key(a, b));
        }
        if first.is_empty() {
            return None;
        }
        let mut corner: Vec<bool> = near.iter().map(|n| !n.is_empty() && n.len() != 2).collect();
        let mut runs = HashMap::new();
        let (mut uv_a, mut uv_b, mut run_vertex) = (Vec::new(), Vec::new(), Vec::new());
        if let Some(s) = seam_uv {
            run_vertex = s.run_vertex.iter().map(|&v| v as u32).collect();
            uv_a = s.uv_a.to_vec();
            uv_b = s.uv_b.to_vec();
            let ok = uv_a.len() >= 2 * run_vertex.len() && uv_b.len() >= 2 * run_vertex.len();
            for c in 0..s.run_indptr.len().saturating_sub(1) {
                let (lo, hi) = (s.run_indptr[c].max(0) as usize, (s.run_indptr[c + 1].max(0) as usize).min(run_vertex.len()));
                if lo >= hi {
                    continue;
                }
                let (v0, v1) = (run_vertex[lo], run_vertex[hi - 1]);
                if v0 != v1 {
                    for v in [v0, v1] {
                        if (v as usize) < nv {
                            corner[v as usize] = true; // a chain end (material corner, border) never goes
                        }
                    }
                }
                for i in lo..hi.saturating_sub(1) {
                    let k = key(run_vertex[i], run_vertex[i + 1]);
                    if ok && k.0 != k.1 && (k.1 as usize) < nv && near[k.0 as usize].contains(&k.1) {
                        runs.insert(k, vec![i as u32, i as u32 + 1]);
                    }
                }
            }
        }
        let mut s = Seams { near, corner, runs, first, uv_a, uv_b, run_vertex, tol: seam_px.max(0.0) / TEXTURE_PX_DEFAULT };
        s.anchor_loops(points);
        Some(s)
    }

    /// Closed loops with fewer than LOOP_KEEP fixed vertices get anchors spread evenly by 3D length.
    fn anchor_loops(&mut self, points: &[f64]) {
        let nv = self.near.len();
        let mut seen = vec![false; nv];
        for start in 0..nv {
            if seen[start] || self.near[start].len() != 2 {
                continue;
            }
            // Walk the component; it is a loop only when every vertex has seam valence 2.
            let mut ring = vec![start as u32];
            seen[start] = true;
            let (mut prev, mut at) = (start as u32, self.near[start][0]);
            let mut closed = true;
            while at as usize != start {
                if self.near[at as usize].len() != 2 || seen[at as usize] {
                    closed = false;
                    break;
                }
                seen[at as usize] = true;
                ring.push(at);
                let n = &self.near[at as usize];
                let next = if n[0] == prev { n[1] } else { n[0] };
                prev = at;
                at = next;
            }
            if !closed {
                continue; // open chains end in corners; their middles are reached from those, nothing to anchor
            }
            let fixed = ring.iter().filter(|&&v| self.corner[v as usize]).count();
            if fixed >= LOOP_KEEP {
                continue;
            }
            let p = |v: u32| [points[3 * v as usize], points[3 * v as usize + 1], points[3 * v as usize + 2]];
            let mut along = vec![0.0f64; ring.len() + 1];
            for i in 0..ring.len() {
                let (a, b) = (p(ring[i]), p(ring[(i + 1) % ring.len()]));
                along[i + 1] = along[i] + ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            }
            let total = along[ring.len()];
            // Start from a fixed vertex when there is one, so the anchors spread around it.
            let shift = ring.iter().position(|&v| self.corner[v as usize]).unwrap_or(0);
            let base = along[shift];
            let mut taken = 0;
            for k in 0..LOOP_KEEP {
                let goal = base + total * k as f64 / LOOP_KEEP as f64;
                let mut best = (f64::INFINITY, 0usize);
                for (i, &v) in ring.iter().enumerate() {
                    let mut d = (along[i] - goal).abs();
                    d = d.min((along[i] + total - goal).abs()).min((along[i] - total - goal).abs());
                    if !self.corner[v as usize] && d < best.0 {
                        best = (d, i);
                    }
                }
                if best.0.is_finite() && fixed + taken < LOOP_KEEP {
                    self.corner[ring[best.1] as usize] = true;
                    taken += 1;
                }
            }
        }
    }

    pub fn is_seam(&self, v: u32) -> bool {
        !self.near[v as usize].is_empty()
    }

    pub fn is_edge(&self, a: u32, b: u32) -> bool {
        self.near.get(a as usize).is_some_and(|n| n.contains(&b))
    }

    /// Seam corners, open-border seam vertices and loop anchors join `locked`.
    pub fn lock(&self, locked: &mut [bool], border: &[bool]) {
        for v in 0..self.near.len().min(locked.len()) {
            if self.corner[v] || (self.is_seam(v as u32) && border.get(v).copied().unwrap_or(false)) {
                locked[v] = true;
            }
        }
    }

    /// May edge (a, b) go with this way (0 best point, 1 middle, 2 at a keeping a, 3 at b keeping b)?
    pub fn way_ok(&self, a: u32, b: u32, way: u8) -> bool {
        let (sa, sb) = (self.is_seam(a), self.is_seam(b));
        if !sa && !sb {
            return true;
        }
        if way < 2 {
            return false; // a half-edge collapse only: the seam vertex stays where its UVs are
        }
        let (keep, gone) = if way == 2 { (a, b) } else { (b, a) };
        if !self.is_seam(gone) {
            return self.is_seam(keep);
        }
        if self.corner[gone as usize] || !self.near[keep as usize].contains(&gone) {
            return false; // a seam vertex goes only along its own chain
        }
        // The far neighbour must not already be next to the kept end (a loop would close up into a double edge).
        match self.other(gone, keep) {
            Some(far) => far != keep && !self.near[keep as usize].contains(&far),
            None => false,
        }
    }

    fn other(&self, v: u32, not: u32) -> Option<u32> {
        let n = &self.near[v as usize];
        (n.len() == 2).then(|| if n[0] == not { n[1] } else { n[0] })
    }

    /// The run positions the merged edge keep..far would cover (keep end first), when K1's runs cover both halves.
    fn merged(&self, keep: u32, gone: u32) -> Option<Vec<u32>> {
        let far = self.other(gone, keep)?;
        let mut l1 = self.runs.get(&key(keep, gone))?.clone();
        let mut l2 = self.runs.get(&key(gone, far))?.clone();
        if self.run_vertex[l1[0] as usize] != keep {
            l1.reverse();
        }
        if self.run_vertex[l2[0] as usize] != gone {
            l2.reverse();
        }
        if self.run_vertex[*l1.last()? as usize] != gone || self.run_vertex[*l2.last()? as usize] != far {
            return None;
        }
        // The run position of `gone` must be the same on both halves (one chain, not two meeting there).
        if l1.last() != l2.first() {
            return None;
        }
        l1.extend_from_slice(&l2[1..]);
        Some(l1)
    }

    /// The UV-side check (Liu et al. 2017): every replaced source UV stays within the tolerance of the new segment, on
    /// both sides. A polyline whose vertices all lie within the tolerance of a segment between its own ends lies within
    /// it entirely (distance to a segment is convex), and covers it, so this bounds the deviation both ways.
    pub fn uv_ok(&self, a: u32, b: u32, way: u8) -> bool {
        if way < 2 {
            return true;
        }
        let (keep, gone) = if way == 2 { (a, b) } else { (b, a) };
        if !self.is_seam(gone) {
            return true;
        }
        let Some(run) = self.merged(keep, gone) else {
            return true; // no source UVs given for this stretch: topology rules only
        };
        let (s, e) = (run[0] as usize, *run.last().unwrap() as usize);
        [&self.uv_a, &self.uv_b].iter().all(|uv| {
            let at = |i: usize| [uv[2 * i], uv[2 * i + 1]];
            let (p, q) = (at(s), at(e));
            run[1..run.len() - 1].iter().all(|&i| segment_distance(at(i as usize), p, q) <= self.tol)
        })
    }

    /// Records the collapse of `gone` onto `keep` (call after the caller applied it).
    pub fn collapse(&mut self, keep: u32, gone: u32) {
        if !self.is_seam(gone) {
            return;
        }
        let merged = self.merged(keep, gone);
        let far = self.other(gone, keep);
        if let Some(far) = far {
            self.runs.remove(&key(keep, gone));
            self.runs.remove(&key(gone, far));
            for v in [keep, far] {
                let n = &mut self.near[v as usize];
                n.retain(|&x| x != gone);
            }
            self.near[keep as usize].push(far);
            self.near[far as usize].push(keep);
            if let Some(run) = merged {
                self.runs.insert(key(keep, far), run);
            }
        }
        self.near[gone as usize].clear();
    }

    /// The source seam edges after a reduction, renumbered by `number` (-1 for a vertex that is not in the output),
    /// as flat pairs without repeats.
    pub fn reduced(&self, mut number: impl FnMut(u32) -> i64) -> Vec<i64> {
        let mut out: Vec<(i64, i64)> = self
            .first
            .iter()
            .filter_map(|&(a, b)| {
                let (x, y) = (number(a), number(b));
                (x >= 0 && y >= 0 && x != y).then(|| if x < y { (x, y) } else { (y, x) })
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out.into_iter().flat_map(|(x, y)| [x, y]).collect()
    }
}

fn segment_distance(x: [f64; 2], p: [f64; 2], q: [f64; 2]) -> f64 {
    let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
    let l = dx * dx + dy * dy;
    let t = if l > 0.0 { (((x[0] - p[0]) * dx + (x[1] - p[1]) * dy) / l).clamp(0.0, 1.0) } else { 0.0 };
    ((x[0] - p[0] - t * dx).powi(2) + (x[1] - p[1] - t * dy).powi(2)).sqrt()
}

/// Compile probe: builds a seam set over a 4-vertex chain. Returns 1 when it worked.
pub fn probe() -> usize {
    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
    let s = Seams::new(&points, &[0, 1, 1, 2, 2, 3], None, SEAM_PX);
    usize::from(s.is_some_and(|s| s.way_ok(0, 1, 2) && !s.way_ok(0, 1, 3) && !s.way_ok(0, 2, 2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_builds() {
        assert_eq!(probe(), 1);
    }

    #[test]
    fn loop_without_corner_gets_three_anchors() {
        let n = 12u32;
        let points: Vec<f64> = (0..n).flat_map(|i| {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            [t.cos(), t.sin(), 0.0]
        }).collect();
        let pairs: Vec<u32> = (0..n).flat_map(|i| [i, (i + 1) % n]).collect();
        let s = Seams::new(&points, &pairs, None, SEAM_PX).unwrap();
        let mut locked = vec![false; n as usize];
        s.lock(&mut locked, &[]);
        let at: Vec<usize> = (0..n as usize).filter(|&i| locked[i]).collect();
        assert_eq!(at, vec![0, 4, 8]);
    }

    #[test]
    fn uv_side_check_refuses_a_bend() {
        // A straight chain in 3D whose side-A UVs bend at vertex 1: by 2 px both inner drops are refused (1 px off for
        // vertex 2), by 0.4 px they pass. Side B is straight.
        let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let px = 1.0 / TEXTURE_PX_DEFAULT;
        let uv_b = [0.5, 0.0, 0.6, 0.0, 0.7, 0.0, 0.8, 0.0];
        for (bend, ok) in [(2.0, false), (0.4, true)] {
            let uv_a = [0.0, 0.0, 0.1, bend * px, 0.2, 0.0, 0.3, 0.0];
            let seam = SeamUv { run_indptr: &[0, 4], run_vertex: &[0, 1, 2, 3], uv_a: &uv_a, uv_b: &uv_b, class: &[2], gap_px: &[0.0; 4] };
            let s = Seams::new(&points, &[0, 1, 1, 2, 2, 3], Some(&seam), SEAM_PX).unwrap();
            assert_eq!(s.uv_ok(0, 1, 2), ok); // 1 onto 0: segment 0..2 against vertex 1's UV
            assert_eq!(s.uv_ok(2, 3, 3), ok); // 2 onto 3: segment 1..3 against vertex 2's UV (half the bend off)
            assert!(s.uv_ok(0, 1, 1)); // not a half-edge way: way_ok refuses it, nothing to check here
        }
    }

    /// A 32x32 bumpy grid with an open chain across it (border to border, wavy on side A by 3 px) and a closed
    /// rectangular loop with no corner, reduced by `qem::run_seams` to a tenth of its faces.
    #[test]
    fn seams_survive_a_tenth() {
        let n = 32u32;
        let id = |i: u32, j: u32| j * (n + 1) + i;
        let mut points = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let (x, y) = (i as f64 / n as f64, j as f64 / n as f64);
                points.extend_from_slice(&[x, y, 0.03 * (6.0 * x).sin() * (4.0 * y).cos()]);
            }
        }
        let mut tris = Vec::new();
        for j in 0..n {
            for i in 0..n {
                tris.extend_from_slice(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
            }
        }
        let px = 1.0 / TEXTURE_PX_DEFAULT;
        let mut run_vertex: Vec<i64> = (0..=n).map(|i| id(i, 12) as i64).collect();
        let mut ring: Vec<u32> = (6..20).map(|i| id(i, 18)).collect();
        ring.extend((18..26).map(|j| id(20, j)));
        ring.extend((7..=20).rev().map(|i| id(i, 26)));
        ring.extend((19..=26).rev().map(|j| id(6, j)));
        ring.push(ring[0]);
        let chain_len = run_vertex.len() as i64;
        run_vertex.extend(ring.iter().map(|&v| v as i64));
        let (mut uv_a, mut uv_b) = (Vec::new(), Vec::new());
        for (k, &v) in run_vertex.iter().enumerate() {
            let (x, y) = (points[3 * v as usize], points[3 * v as usize + 1]);
            let wave = if (k as i64) < chain_len { 3.0 * px * (x * 18.0).sin() } else { 0.0 };
            uv_a.extend_from_slice(&[0.5 * x, 0.5 * y + wave]);
            uv_b.extend_from_slice(&[0.5 * x + 0.5, 0.5 * y]);
        }
        let run_indptr = [0, chain_len, run_vertex.len() as i64];
        let gap = vec![0.0; run_vertex.len()];
        let seam = SeamUv { run_indptr: &run_indptr, run_vertex: &run_vertex, uv_a: &uv_a, uv_b: &uv_b, class: &[2, 2], gap_px: &gap };
        let pairs: Vec<u32> = (0..2)
            .flat_map(|c| (run_indptr[c] as usize..run_indptr[c + 1] as usize - 1).flat_map(|k| [run_vertex[k] as u32, run_vertex[k + 1] as u32]))
            .collect();
        let faces = (tris.len() / 3) as u32;
        let run = crate::qem::run_seams(&points, &tris, &[], &pairs, Some(&seam), SEAM_PX, 4);
        let (out, out_tris, _, kept) = run.level_seams(faces / 10);
        assert!(out_tris.len() / 3 <= faces as usize / 10 + 2, "reduced to {} of {faces}", out_tris.len() / 3);
        // Every seam vertex left sits exactly on a source seam vertex (half-edge collapses only).
        let at: HashMap<[u64; 3], u32> = (0..points.len() / 3).map(|v| ([0, 1, 2].map(|k| points[3 * v + k].to_bits()), v as u32)).collect();
        let source = |o: i64| at.get(&[0, 1, 2].map(|k| out[3 * o as usize + k].to_bits())).copied();
        let edges: std::collections::HashSet<(u32, u32)> = kept.chunks_exact(2).map(|e| key(source(e[0]).unwrap(), source(e[1]).unwrap())).collect();
        let alive: std::collections::HashSet<u32> = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
        let mut worst: f64 = 0.0;
        for c in 0..2 {
            let span: Vec<usize> = (run_indptr[c] as usize..run_indptr[c + 1] as usize).collect();
            let left: Vec<usize> = span.iter().copied().filter(|&k| alive.contains(&(run_vertex[k] as u32))).collect();
            let closed = run_vertex[span[0]] == run_vertex[*span.last().unwrap()];
            if closed {
                assert!(left.len() - 1 >= LOOP_KEEP, "loop kept {} edges", left.len() - 1); // last repeats the first
            } else {
                assert_eq!((left[0], *left.last().unwrap()), (span[0], *span.last().unwrap()), "chain ends kept");
            }
            for w in left.windows(2) {
                let (s, e) = (w[0], w[1]);
                assert!(edges.contains(&key(run_vertex[s] as u32, run_vertex[e] as u32)), "chain {c} broken between {s} and {e}");
                for uv in [&uv_a, &uv_b] {
                    let p = |i: usize| [uv[2 * i], uv[2 * i + 1]];
                    for i in s + 1..e {
                        worst = worst.max(segment_distance(p(i), p(s), p(e)) / px);
                    }
                }
            }
            println!("chain {c}: {} of {} vertices left", left.len(), span.len());
        }
        println!("faces {} -> {}, seam edges {} -> {}, worst UV-side deviation {worst:.3} px", faces, out_tris.len() / 3, pairs.len() / 2, edges.len());
        assert!(worst <= SEAM_PX);
        // Without K1's runs (topology only) the loop still keeps its 3 anchors and the chain its two ends.
        let run = crate::qem::run_seams(&points, &tris, &[], &pairs, None, SEAM_PX, 4);
        let (out, _, _, kept) = run.level_seams(faces / 10);
        let source = |o: i64| at.get(&[0, 1, 2].map(|k| out[3 * o as usize + k].to_bits())).copied().unwrap();
        let on_loop = kept.chunks_exact(2).filter(|e| ring.contains(&source(e[0])) && ring.contains(&source(e[1]))).count();
        println!("topology only: {} seam edges, {on_loop} on the loop", kept.len() / 2);
        assert!(on_loop >= LOOP_KEEP);
    }
}
