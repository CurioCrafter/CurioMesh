// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! UV seam kernel (package K1): source islands, side regions, seam chains, chain classes, Hard triage with the
//! face-cost model, side-aware per-side transfer and the fold fix (SPEC 4.1, 4.2, 4.8). Byte-identical on `main` and
//! `standalone`: uses only `topo`, `geom`, `close` and the `near` items both branches have.
//!
//! Our own design and code. Textbook pieces: closest point on a triangle with its barycentric weights (Ericson,
//! "Real-Time Collision Detection", 2005, 5.1.5); Gauss-Seidel relaxation of a harmonic fill (Tutte 1963 / Floater 1997
//! idea of a convex-combination fill, here with inverse-length weights). The side rules (side regions, the area-majority
//! reference corner, the walk that never crosses a seam, per-corner whole-tile offsets on Shift chains, ring-growing
//! fold fix) are ours.

#![allow(dead_code)] // some outputs are read only by other kernels' packages and the standalone wiring

use crate::near::{nearest_faces, FaceTree};
use crate::topo::{edges, halves, Dsu, Faces, FxBuild};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};

// Constants table (mirrored in triadq/uvseams.py; tests/test_uv_scaffold.py compares the two).
/// Two corners' UVs are equal when within this many pixels at the texture size (islands).
pub const UV_TOL_PX: f64 = 0.01;
/// Texture size when the caller gives none (pixels).
pub const TEXTURE_PX_DEFAULT: f64 = 1024.0;
/// Ignore: both sides' UVs within this many pixels everywhere along the chain.
pub const IGNORE_PX: f64 = 1.0;
/// Shift: the sides differ by a whole-tile vector within this many pixels everywhere along the chain.
pub const SHIFT_TOL_PX: f64 = 1.0;
/// Hard candidates: islands fewer than this many result edges across.
pub const HARD_ACROSS_EDGES: f64 = 3.0;
/// Face-cost model of one Hard island: `COST_A * perimeter / edge_est + COST_B` faces (start values; P0 fits them).
pub const COST_A: f64 = 1.0;
pub const COST_B: f64 = 4.0;
/// Hard islands may spend this share of the allowance; split leftovers spend the rest.
pub const HARD_BUDGET_SHARE: f64 = 0.5;
/// The per-corner walk stays within this many result-face diameters (plus the reference hit's distance).
pub const WALK_REACH: f64 = 2.0;
/// A corner is out of reach (slid along its triangle's plane instead of clamped to the seam) when its reachable point is
/// farther than its nearest point by more than this share of the result face's mean edge.
pub const SLIDE_EDGE: f64 = 0.02;
/// Share of a result face's area its centre sample stands for (the corners share the rest equally).
pub const CENTRE_WEIGHT: f64 = 0.5;
/// Fold fix stop rule (SPEC 3): no UV moves more than this share of a texel in one sweep.
pub const STOP_TEXEL_SHARE: f64 = 0.02;
/// Fold fix stall rule: the largest move must drop by `STALL_DROP` within `STALL_WINDOW` sweeps (reported).
pub const STALL_WINDOW: usize = 50;
pub const STALL_DROP: f64 = 0.01;

/// What a seam chain asks of the remesh (spec 4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ChainClass {
    Ignore = 0,
    Shift = 1,
    Soft = 2,
    Hard = 3,
}

/// Frozen cross-kernel input (K1's output, read by K1b, K2, K3, K6): seam chains as flat slices.
/// Chain `c` runs over `run_vertex[run_indptr[c]..run_indptr[c + 1]]` (welded vertex ids). `uv_a` / `uv_b` hold the
/// UV (2 values) of each run vertex on side A and side B; `gap_px` one value per run vertex; `class` one
/// `ChainClass` code per chain.
pub struct SeamUv<'a> {
    pub run_indptr: &'a [i64],
    pub run_vertex: &'a [i64],
    pub uv_a: &'a [f64],
    pub uv_b: &'a [f64],
    pub class: &'a [u8],
    pub gap_px: &'a [f64],
}

/// The source's seams. A closed chain (a loop with no seam corner) repeats its first vertex at the end.
pub struct SeamSet {
    pub texture_px: f64,
    /// Welded vertex id per input point (points at exactly the same place share one), and the welded count.
    pub weld: Vec<i64>,
    pub n_welded: usize,
    /// Per source face: its island (faces joined across an edge where material and both end UVs agree) and its side
    /// region (islands joined across Ignore and Shift chains). Ids in order of first face.
    pub island: Vec<i64>,
    pub region: Vec<i64>,
    pub n_islands: usize,
    pub n_regions: usize,
    /// Seam edges (welded vertex pairs), their chain and their side A / side B source face.
    pub edge_a: Vec<i64>,
    pub edge_b: Vec<i64>,
    pub edge_chain: Vec<i64>,
    pub edge_face: Vec<[i64; 2]>,
    /// Per welded vertex: true at a seam corner (seam valence other than 2, open border, other-than-two-faces edge,
    /// a change of island or material pair along the seam).
    pub corner: Vec<bool>,
    pub run_indptr: Vec<i64>,
    pub run_vertex: Vec<i64>,
    pub uv_a: Vec<f64>,
    pub uv_b: Vec<f64>,
    pub gap_px: Vec<f64>,
    pub class: Vec<u8>,
    pub chain_island: Vec<[i64; 2]>,
    pub chain_material: Vec<[i64; 2]>,
    pub chain_length: Vec<f64>,
    /// Per island: its tile (floor of the centre of its UV box), u and v.
    pub island_tile: Vec<[i64; 2]>,
}

impl SeamSet {
    pub fn view(&self) -> SeamUv<'_> {
        SeamUv { run_indptr: &self.run_indptr, run_vertex: &self.run_vertex, uv_a: &self.uv_a, uv_b: &self.uv_b, class: &self.class, gap_px: &self.gap_px }
    }
    pub fn chains(&self) -> usize {
        self.class.len()
    }
}

type V3 = [f64; 3];

fn p3(points: &[f64], v: i64) -> V3 {
    let v = v as usize * 3;
    [points[v], points[v + 1], points[v + 2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn dist(a: V3, b: V3) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}
fn quotient(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        0.0
    } else {
        a / b
    }
}
/// Rounds half away from zero (the twin uses the same rule).
fn whole(x: f64) -> f64 {
    x.round()
}

/// Closest point on triangle abc to p as barycentric weights (Ericson 2005, 5.1.5; zero divisors give 0).
pub fn closest_bary(p: V3, a: V3, b: V3, c: V3) -> [f64; 3] {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = quotient(d1, d1 - d3);
        return [1.0 - v, v, 0.0];
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = quotient(d2, d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = quotient(d4 - d3, (d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let under = va + vb + vc;
    let (v, w) = (quotient(vb, under), quotient(vc, under));
    [1.0 - v - w, v, w]
}

/// Barycentric weights of p's projection onto the plane of abc, not clamped to the triangle (an affine extension).
fn plane_bary(p: V3, a: V3, b: V3, c: V3) -> Option<[f64; 3]> {
    let (e1, e2, d) = (sub(b, a), sub(c, a), sub(p, a));
    let (g11, g12, g22) = (dot(e1, e1), dot(e1, e2), dot(e2, e2));
    let det = g11 * g22 - g12 * g12;
    if det <= 1e-30 * (g11 * g22).max(1e-300) {
        return None;
    }
    let (r1, r2) = (dot(d, e1), dot(d, e2));
    let s = (g22 * r1 - g12 * r2) / det;
    let t = (g11 * r2 - g12 * r1) / det;
    Some([1.0 - s - t, s, t])
}

fn mix(w: [f64; 3], a: V3, b: V3, c: V3) -> V3 {
    [w[0] * a[0] + w[1] * b[0] + w[2] * c[0], w[0] * a[1] + w[1] * b[1] + w[2] * c[1], w[0] * a[2] + w[1] * b[2] + w[2] * c[2]]
}

fn newell(points: &[f64], face: &[i64]) -> V3 {
    let mut n = [0.0; 3];
    for i in 0..face.len() {
        let (a, b) = (p3(points, face[i]), p3(points, face[(i + 1) % face.len()]));
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    n
}

fn weld(points: &[f64]) -> (Vec<i64>, usize) {
    let n = points.len() / 3;
    let mut seen: HashMap<[u64; 3], i64, FxBuild> = HashMap::with_capacity_and_hasher(n, FxBuild::default());
    let mut out = Vec::with_capacity(n);
    for v in 0..n {
        let key = [(points[3 * v] + 0.0).to_bits(), (points[3 * v + 1] + 0.0).to_bits(), (points[3 * v + 2] + 0.0).to_bits()];
        let next = seen.len() as i64;
        out.push(*seen.entry(key).or_insert(next));
    }
    let count = seen.len();
    (out, count)
}

fn relabel(root: impl Iterator<Item = u32>, out: &mut Vec<i64>) -> usize {
    let mut ids: HashMap<u32, i64, FxBuild> = HashMap::default();
    for r in root {
        let next = ids.len() as i64;
        out.push(*ids.entry(r).or_insert(next));
    }
    ids.len()
}

/// Corner -> next corner of the same face.
fn next_corners(faces: &Faces) -> Vec<usize> {
    let mut nxt = vec![0usize; faces.flat.len()];
    for f in 0..faces.count() {
        let (s, e) = (faces.indptr[f] as usize, faces.indptr[f + 1] as usize);
        for h in s..e {
            nxt[h] = if h + 1 == e { s } else { h + 1 };
        }
    }
    nxt
}

/// Islands, seam edges and chains of a source mesh (SPEC 2.2, 4.1). `corner_uv` holds 2 values per corner; `material`
/// one id per face (empty: all 0); `repeat[m]` says whether material `m`'s texture repeats (missing: true, the glTF
/// default). Classes are set by `classify` (called here with no "Cut here" chains).
pub fn extract(points: &[f64], faces: &Faces, corner_uv: &[f64], material: &[i64], repeat: &[bool], texture_px: f64, uv_tol_px: f64) -> SeamSet {
    let tex = if texture_px > 0.0 { texture_px } else { TEXTURE_PX_DEFAULT };
    let tol = uv_tol_px.max(0.0) / tex;
    let nf = faces.count();
    let mat = |f: usize| if material.is_empty() { 0 } else { material[f] };
    let (weld_of, n_welded) = weld(points);
    let wflat: Vec<i64> = faces.flat.iter().map(|&v| weld_of[v as usize]).collect();
    let wfaces = Faces { indptr: faces.indptr, flat: &wflat };
    let half = halves(&wfaces);
    let all = edges(&half);
    let nxt = next_corners(faces);
    let uv = |h: usize| [corner_uv[2 * h], corner_uv[2 * h + 1]];
    // UV of the face that owns corner h, at welded vertex w (an end of h's edge).
    let uv_at = |h: usize, w: i64| if wflat[h] == w { uv(h) } else { uv(nxt[h]) };
    let same = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() <= tol;
    let joined = |h: usize, g: usize, a: i64, b: i64| {
        mat(half.face[h] as usize) == mat(half.face[g] as usize) && same(uv_at(h, a), uv_at(g, a)) && same(uv_at(h, b), uv_at(g, b))
    };

    let mut dsu = Dsu::new(nf);
    let mut odd = vec![false; n_welded]; // open border or an edge used by other than two faces
    let mut seam_edges: Vec<usize> = Vec::new();
    for e in 0..all.len() {
        let uses = all.uses(e);
        let (a, b) = (all.a[e], all.b[e]);
        if uses.len() == 2 && half.face[uses[0] as usize] != half.face[uses[1] as usize] {
            let (h, g) = (uses[0] as usize, uses[1] as usize);
            if joined(h, g, a, b) {
                dsu.union(half.face[h], half.face[g]);
            } else {
                seam_edges.push(e);
            }
            continue;
        }
        odd[a as usize] = true;
        odd[b as usize] = true;
        for k in 1..uses.len() {
            let (h, g) = (uses[k - 1] as usize, uses[k] as usize);
            if half.face[h] != half.face[g] && joined(h, g, a, b) {
                dsu.union(half.face[h], half.face[g]);
            }
        }
    }
    let mut island = Vec::with_capacity(nf);
    let n_islands = relabel((0..nf as u32).map(|f| dsu.find(f)), &mut island);

    // Seam edges: faces (by orientation where possible), island and material pairs, adjacency per welded vertex.
    let ne = seam_edges.len();
    let mut valence = vec![0u32; n_welded];
    for &e in &seam_edges {
        valence[all.a[e] as usize] += 1;
        valence[all.b[e] as usize] += 1;
    }
    let mut adj_ptr = vec![0usize; n_welded + 1];
    for v in 0..n_welded {
        adj_ptr[v + 1] = adj_ptr[v] + valence[v] as usize;
    }
    let mut fill = adj_ptr.clone();
    let mut adj = vec![(0i64, 0usize); adj_ptr[n_welded]];
    for (k, &e) in seam_edges.iter().enumerate() {
        let (a, b) = (all.a[e], all.b[e]);
        adj[fill[a as usize]] = (b, k);
        fill[a as usize] += 1;
        adj[fill[b as usize]] = (a, k);
        fill[b as usize] += 1;
    }
    for v in 0..n_welded {
        adj[adj_ptr[v]..adj_ptr[v + 1]].sort_unstable();
    }
    let pair = |x: i64, y: i64| if x <= y { [x, y] } else { [y, x] };
    let key_of = |k: usize| {
        let uses = all.uses(seam_edges[k]);
        let (f, g) = (half.face[uses[0] as usize] as usize, half.face[uses[1] as usize] as usize);
        (pair(island[f], island[g]), pair(mat(f), mat(g)))
    };
    let mut corner: Vec<bool> = (0..n_welded).map(|v| odd[v] || (valence[v] != 2 && valence[v] > 0)).collect();
    for v in 0..n_welded {
        if valence[v] == 2 && !corner[v] && key_of(adj[adj_ptr[v]].1) != key_of(adj[adj_ptr[v] + 1].1) {
            corner[v] = true;
        }
    }

    // Walk chains: from corners in vertex order, then the closed loops from their lowest vertex.
    let mut visited = vec![false; ne];
    let mut runs: Vec<(Vec<i64>, Vec<usize>)> = Vec::new();
    let walk = |start: i64, first: usize, visited: &mut Vec<bool>| {
        let (mut run, mut used) = (vec![start], Vec::new());
        let (mut cur, mut k) = (start, first);
        loop {
            visited[k] = true;
            used.push(k);
            let e = seam_edges[k];
            let nxt_v = if all.a[e] == cur { all.b[e] } else { all.a[e] };
            run.push(nxt_v);
            if corner[nxt_v as usize] || nxt_v == start {
                break;
            }
            let around = &adj[adj_ptr[nxt_v as usize]..adj_ptr[nxt_v as usize + 1]];
            match around.iter().find(|&&(_, j)| j != k && !visited[j]) {
                Some(&(_, j)) => {
                    cur = nxt_v;
                    k = j;
                }
                None => break,
            }
        }
        (run, used)
    };
    for v in 0..n_welded {
        if corner[v] {
            for i in adj_ptr[v]..adj_ptr[v + 1] {
                let k = adj[i].1;
                if !visited[k] {
                    runs.push(walk(v as i64, k, &mut visited));
                }
            }
        }
    }
    for v in 0..n_welded {
        for i in adj_ptr[v]..adj_ptr[v + 1] {
            let k = adj[i].1;
            if !visited[k] {
                runs.push(walk(v as i64, k, &mut visited));
            }
        }
    }

    // Per chain: side faces per edge, UVs per run vertex, gaps, islands, materials, 3D length.
    let mut set = SeamSet {
        texture_px: tex,
        weld: weld_of,
        n_welded,
        island,
        region: Vec::new(),
        n_islands,
        n_regions: 0,
        edge_a: vec![0; ne],
        edge_b: vec![0; ne],
        edge_chain: vec![-1; ne],
        edge_face: vec![[0, 0]; ne],
        corner,
        run_indptr: vec![0],
        run_vertex: Vec::new(),
        uv_a: Vec::new(),
        uv_b: Vec::new(),
        gap_px: Vec::new(),
        class: Vec::new(),
        chain_island: Vec::new(),
        chain_material: Vec::new(),
        chain_length: Vec::new(),
        island_tile: Vec::new(),
    };
    let mut wpos = vec![[0.0; 3]; n_welded];
    for (v, &w) in set.weld.iter().enumerate() {
        wpos[w as usize] = p3(points, v as i64);
    }
    for (c, (run, used)) in runs.iter().enumerate() {
        let mut side_a: Option<i64> = None;
        let mut faces_ab: Vec<[usize; 2]> = Vec::with_capacity(used.len());
        for (j, &k) in used.iter().enumerate() {
            let e = seam_edges[k];
            let uses = all.uses(e);
            let (h, g) = (uses[0] as usize, uses[1] as usize);
            let (from, to) = (run[j], run[j + 1]);
            let along = |x: usize| wflat[x] == from && wflat[nxt[x]] == to;
            let (fh, fg) = (half.face[h] as i64, half.face[g] as i64);
            let pick_h = match (along(h), along(g)) {
                (true, false) => true,
                (false, true) => false,
                _ => match side_a {
                    Some(isl) if set.island[fh as usize] != set.island[fg as usize] => set.island[fh as usize] == isl,
                    _ => fh <= fg,
                },
            };
            let (ha, hb) = if pick_h { (h, g) } else { (g, h) };
            if side_a.is_none() {
                side_a = Some(set.island[half.face[ha] as usize]);
            }
            faces_ab.push([ha, hb]);
            set.edge_a[k] = all.a[e];
            set.edge_b[k] = all.b[e];
            set.edge_chain[k] = c as i64;
            set.edge_face[k] = [half.face[ha] as i64, half.face[hb] as i64];
        }
        let mut length = 0.0;
        for (j, &w) in run.iter().enumerate() {
            let [ha, hb] = faces_ab[j.min(used.len() - 1)];
            let (a, b) = (uv_at(ha, w), uv_at(hb, w));
            set.run_vertex.push(w);
            set.uv_a.extend_from_slice(&a);
            set.uv_b.extend_from_slice(&b);
            set.gap_px.push(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() * tex);
            if j > 0 {
                length += dist(wpos[run[j - 1] as usize], wpos[w as usize]);
            }
        }
        set.run_indptr.push(set.run_vertex.len() as i64);
        let [fa, fb] = set.edge_face[used[0]];
        set.chain_island.push([set.island[fa as usize], set.island[fb as usize]]);
        set.chain_material.push([mat(fa as usize), mat(fb as usize)]);
        set.chain_length.push(length);
    }
    // Island tiles from each island's UV box.
    let mut lo = vec![[f64::INFINITY; 2]; n_islands];
    let mut hi = vec![[f64::NEG_INFINITY; 2]; n_islands];
    for f in 0..nf {
        let i = set.island[f] as usize;
        for h in faces.indptr[f] as usize..faces.indptr[f + 1] as usize {
            for k in 0..2 {
                lo[i][k] = lo[i][k].min(corner_uv[2 * h + k]);
                hi[i][k] = hi[i][k].max(corner_uv[2 * h + k]);
            }
        }
    }
    set.island_tile = (0..n_islands).map(|i| [(0.5 * (lo[i][0] + hi[i][0])).floor() as i64, (0.5 * (lo[i][1] + hi[i][1])).floor() as i64]).collect();
    classify(&mut set, repeat, &[]);
    set
}

/// Chain classes (SPEC 4.1) and the side regions they give. `cut_here` lists chains the user made Hard.
pub fn classify(set: &mut SeamSet, repeat: &[bool], cut_here: &[i64]) {
    let tex = set.texture_px;
    let repeats = |m: i64| m < 0 || (m as usize) >= repeat.len() || repeat[m as usize];
    set.class.clear();
    for c in 0..set.chain_island.len() {
        let (s, e) = (set.run_indptr[c] as usize, set.run_indptr[c + 1] as usize);
        let [ma, mb] = set.chain_material[c];
        let [ia, ib] = set.chain_island[c];
        let gap = set.gap_px[s..e].iter().cloned().fold(0.0, f64::max);
        let class = if ma != mb || cut_here.contains(&(c as i64)) {
            ChainClass::Hard
        } else if gap <= IGNORE_PX {
            ChainClass::Ignore
        } else {
            let n = (e - s) as f64;
            let mean = |k: usize| (s..e).map(|j| set.uv_b[2 * j + k] - set.uv_a[2 * j + k]).sum::<f64>() / n;
            let shift = [whole(mean(0)), whole(mean(1))];
            let fits = (s..e).all(|j| {
                let (du, dv) = (set.uv_b[2 * j] - set.uv_a[2 * j] - shift[0], set.uv_b[2 * j + 1] - set.uv_a[2 * j + 1] - shift[1]);
                (du * du + dv * dv).sqrt() * tex <= SHIFT_TOL_PX
            });
            let other_tile = set.island_tile[ia as usize] != set.island_tile[ib as usize];
            if fits && shift != [0.0, 0.0] && repeats(ma) {
                ChainClass::Shift
            } else if other_tile && !repeats(ma) {
                ChainClass::Hard
            } else {
                ChainClass::Soft
            }
        };
        set.class.push(class as u8);
    }
    regions(set);
}

/// Sets every chain's class (codes of `ChainClass`, e.g. from the caller's cache) and redoes the side regions.
pub fn set_classes(set: &mut SeamSet, classes: &[u8]) {
    set.class = classes.to_vec();
    regions(set);
}

fn regions(set: &mut SeamSet) {
    let mut dsu = Dsu::new(set.n_islands);
    for (c, &class) in set.class.iter().enumerate() {
        if class == ChainClass::Ignore as u8 || class == ChainClass::Shift as u8 {
            let [a, b] = set.chain_island[c];
            dsu.union(a as u32, b as u32);
        }
    }
    let mut region = Vec::with_capacity(set.island.len());
    set.n_regions = relabel(set.island.iter().map(|&i| dsu.find(i as u32)), &mut region);
    set.region = region;
}

/// What the Hard triage chose (SPEC 4.2): islands made Hard, the face estimate spent, the budget, the candidates
/// the budget left (reported, offered to "Bake those spots"), and the edge estimate used.
pub struct Triage {
    pub hard_islands: Vec<i64>,
    pub hard_cost: Vec<f64>,
    pub left_islands: Vec<i64>,
    pub left_cost: Vec<f64>,
    pub spent: f64,
    pub budget: f64,
    pub edge_est: f64,
}

/// Pre-search Hard choice (SPEC 4.2 step 1): islands fewer than `HARD_ACROSS_EDGES` result edges across, ranked by
/// texture at risk (3D area x largest Soft gap px x seen share), made Hard while the summed face estimate fits in
/// `HARD_BUDGET_SHARE` of the allowance. `allowance` is a share of `f_target` faces; `seen_face` per source face or None.
pub fn triage(set: &mut SeamSet, points: &[f64], faces: &Faces, f_target: f64, allowance: f64, seen_face: Option<&[f64]>, cost_a: f64, cost_b: f64) -> Triage {
    let ni = set.n_islands;
    let (mut area, mut seen_area, mut perimeter) = (vec![0.0; ni], vec![0.0; ni], vec![0.0; ni]);
    let mut lo = vec![[f64::INFINITY; 3]; ni];
    let mut hi = vec![[f64::NEG_INFINITY; 3]; ni];
    let mut total = 0.0;
    for f in 0..faces.count() {
        let face = faces.face(f);
        let i = set.island[f] as usize;
        let a = 0.5 * dot(newell(points, face), newell(points, face)).sqrt();
        area[i] += a;
        seen_area[i] += a * seen_face.map_or(1.0, |s| s[f]);
        total += a;
        for &v in face {
            let p = p3(points, v);
            for k in 0..3 {
                lo[i][k] = lo[i][k].min(p[k]);
                hi[i][k] = hi[i][k].max(p[k]);
            }
        }
    }
    // Perimeter: seam edges count for both sides' islands, open borders for theirs.
    let wflat: Vec<i64> = faces.flat.iter().map(|&v| set.weld[v as usize]).collect();
    let half = halves(&Faces { indptr: faces.indptr, flat: &wflat });
    let all = edges(&half);
    let mut wpos = vec![[0.0; 3]; set.n_welded];
    for (v, &w) in set.weld.iter().enumerate() {
        wpos[w as usize] = p3(points, v as i64);
    }
    for e in 0..all.len() {
        let uses = all.uses(e);
        let len = dist(wpos[all.a[e] as usize], wpos[all.b[e] as usize]);
        if uses.len() == 1 {
            perimeter[set.island[half.face[uses[0] as usize] as usize] as usize] += len;
        }
    }
    for k in 0..set.edge_a.len() {
        let len = dist(wpos[set.edge_a[k] as usize], wpos[set.edge_b[k] as usize]);
        for &f in &set.edge_face[k] {
            perimeter[set.island[f as usize] as usize] += len;
        }
    }
    let mut gap = vec![0.0f64; ni];
    let mut soft = vec![false; ni];
    for c in 0..set.chains() {
        if set.class[c] == ChainClass::Soft as u8 {
            let g = set.gap_px[set.run_indptr[c] as usize..set.run_indptr[c + 1] as usize].iter().cloned().fold(0.0, f64::max);
            for &i in &set.chain_island[c] {
                gap[i as usize] = gap[i as usize].max(g);
                soft[i as usize] = true;
            }
        }
    }
    let edge_est = if f_target > 0.0 { (total / f_target).sqrt() } else { 0.0 };
    let mut ranked: Vec<(f64, i64, f64)> = Vec::new();
    for i in 0..ni {
        if !soft[i] || perimeter[i] <= 0.0 || edge_est <= 0.0 {
            continue;
        }
        let diag = dist(lo[i], hi[i]);
        let across = diag.min(4.0 * area[i] / perimeter[i]);
        if across / edge_est < HARD_ACROSS_EDGES {
            let seen = if area[i] > 0.0 { seen_area[i] / area[i] } else { 1.0 };
            ranked.push((area[i] * gap[i] * seen, i as i64, cost_a * perimeter[i] / edge_est + cost_b));
        }
    }
    ranked.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal).then(x.1.cmp(&y.1)));
    let budget = HARD_BUDGET_SHARE * allowance.max(0.0) * f_target.max(0.0);
    let mut out = Triage { hard_islands: Vec::new(), hard_cost: Vec::new(), left_islands: Vec::new(), left_cost: Vec::new(), spent: 0.0, budget, edge_est };
    let mut hard = vec![false; ni];
    for &(_, i, cost) in &ranked {
        if out.spent + cost <= budget {
            out.spent += cost;
            hard[i as usize] = true;
            out.hard_islands.push(i);
            out.hard_cost.push(cost);
        } else {
            out.left_islands.push(i);
            out.left_cost.push(cost);
        }
    }
    for c in 0..set.chains() {
        let [a, b] = set.chain_island[c];
        if set.class[c] == ChainClass::Soft as u8 && (hard[a as usize] || hard[b as usize]) {
            set.class[c] = ChainClass::Hard as u8;
        }
    }
    out
}

/// Where every result corner reads the source (SPEC 4.8): three source corners and their weights per corner (an
/// affine combination, so it serves any UV layer), whether the corner's lookup crossed a Shift chain (then its UV takes
/// the whole-tile offset nearest the reference's), whether it slid out of reach, and per face the reference read the
/// same way, the reference source face, the orientation sign against it and the side region. `done` is false when
/// stopped; unfinished faces have `ref_face` -1.
pub struct Lookup {
    pub ids: Vec<i64>,
    pub w: Vec<f64>,
    pub shift: Vec<bool>,
    pub slid: Vec<bool>,
    pub ref_ids: Vec<i64>,
    pub ref_w: Vec<f64>,
    pub ref_face: Vec<i64>,
    pub orient: Vec<f64>,
    pub region: Vec<i64>,
    /// Per side region: how far apart (UV units) two corners of one vertex may be and still share an unknown in the
    /// fold fix (half the region's smallest self-seam gap; infinite without one).
    pub join: Vec<f64>,
    pub done: bool,
}

struct Source<'a> {
    points: &'a [f64],
    faces: &'a Faces<'a>,
    /// Fan triangles per face, as source corner ids.
    tri: Vec<[usize; 3]>,
    tri_ptr: Vec<usize>,
    centre: Vec<V3>,
    normal: Vec<V3>,
    /// Per corner: the face across its edge (or -1) and the crossing: 0 free, 1 Shift chain, 2 blocked.
    across: Vec<i64>,
    crossing: Vec<u8>,
}

impl Source<'_> {
    fn corner_point(&self, h: usize) -> V3 {
        p3(self.points, self.faces.flat[h])
    }
    /// Closest point on face f: (squared distance, triangle id, weights).
    fn on_face(&self, f: usize, q: V3) -> (f64, usize, [f64; 3]) {
        let mut best = (f64::INFINITY, usize::MAX, [0.0; 3]);
        for t in self.tri_ptr[f]..self.tri_ptr[f + 1] {
            let [a, b, c] = self.tri[t];
            let (pa, pb, pc) = (self.corner_point(a), self.corner_point(b), self.corner_point(c));
            let w = closest_bary(q, pa, pb, pc);
            let d = sub(mix(w, pa, pb, pc), q);
            let d2 = dot(d, d);
            if d2 < best.0 {
                best = (d2, t, w);
            }
        }
        best
    }
}

fn source<'a>(points: &'a [f64], faces: &'a Faces<'a>, set: &SeamSet) -> Source<'a> {
    let nf = faces.count();
    let (mut tri, mut tri_ptr) = (Vec::new(), vec![0usize]);
    let (mut centre, mut normal) = (Vec::with_capacity(nf), Vec::with_capacity(nf));
    for f in 0..nf {
        let (s, e) = (faces.indptr[f] as usize, faces.indptr[f + 1] as usize);
        for k in s + 1..e.saturating_sub(1) {
            tri.push([s, k, k + 1]);
        }
        tri_ptr.push(tri.len());
        let face = faces.face(f);
        let mut c = [0.0; 3];
        for &v in face {
            let p = p3(points, v);
            for k in 0..3 {
                c[k] += p[k] / face.len() as f64;
            }
        }
        centre.push(c);
        normal.push(newell(points, face));
    }
    let wflat: Vec<i64> = faces.flat.iter().map(|&v| set.weld[v as usize]).collect();
    let half = halves(&Faces { indptr: faces.indptr, flat: &wflat });
    let all = edges(&half);
    let mut seam_class: HashMap<(i64, i64), u8, FxBuild> = HashMap::default();
    for k in 0..set.edge_a.len() {
        let (a, b) = (set.edge_a[k].min(set.edge_b[k]), set.edge_a[k].max(set.edge_b[k]));
        seam_class.insert((a, b), set.class[set.edge_chain[k] as usize]);
    }
    let (mut across, mut crossing) = (vec![-1i64; faces.flat.len()], vec![2u8; faces.flat.len()]);
    for e in 0..all.len() {
        let uses = all.uses(e);
        if uses.len() != 2 || half.face[uses[0] as usize] == half.face[uses[1] as usize] {
            continue;
        }
        let kind = match seam_class.get(&(all.a[e], all.b[e])) {
            None => 0,
            Some(&c) if c == ChainClass::Ignore as u8 => 0,
            Some(&c) if c == ChainClass::Shift as u8 => 1,
            Some(_) => 2,
        };
        let (h, g) = (uses[0] as usize, uses[1] as usize);
        across[h] = half.face[g] as i64;
        across[g] = half.face[h] as i64;
        crossing[h] = kind;
        crossing[g] = kind;
    }
    Source { points, faces, tri, tri_ptr, centre, normal, across, crossing }
}

struct FaceRead {
    ids: Vec<i64>,
    w: Vec<f64>,
    shift: Vec<bool>,
    slid: Vec<bool>,
    ref_ids: [i64; 3],
    ref_w: [f64; 3],
    ref_face: i64,
    orient: f64,
    region: i64,
}

fn read_face(src: &Source, set: &SeamSet, tree: &FaceTree, res_points: &[f64], face: &[i64]) -> FaceRead {
    let n = face.len();
    let corners: Vec<V3> = face.iter().map(|&v| p3(res_points, v)).collect();
    let mut c = [0.0; 3];
    for p in &corners {
        for k in 0..3 {
            c[k] += p[k] / n as f64;
        }
    }
    let mut diameter = 0.0f64;
    let mut edge = 0.0;
    for i in 0..n {
        edge += dist(corners[i], corners[(i + 1) % n]) / n as f64;
        for j in i + 1..n {
            diameter = diameter.max(dist(corners[i], corners[j]));
        }
    }
    // Samples: the corners, then the centre; each one's nearest source face over the whole source.
    let mut samples = corners.clone();
    samples.push(c);
    let hits: Vec<(usize, f64, usize, [f64; 3])> = samples
        .iter()
        .map(|&q| {
            let f = tree.nearest(q).map_or(0, |h| h.1 as usize);
            let (d2, t, w) = src.on_face(f, q);
            (f, d2.sqrt(), t, w)
        })
        .collect();
    let weight = |s: usize| if s == n { CENTRE_WEIGHT } else { (1.0 - CENTRE_WEIGHT) / n as f64 };
    let mut votes: Vec<(i64, f64)> = Vec::new();
    for (s, hit) in hits.iter().enumerate() {
        let r = set.region[hit.0];
        match votes.iter_mut().find(|x| x.0 == r) {
            Some(x) => x.1 += weight(s),
            None => votes.push((r, weight(s))),
        }
    }
    let mut win = votes[0];
    for &v in &votes[1..] {
        if v.1 > win.1 || (v.1 == win.1 && v.0 < win.0) {
            win = v;
        }
    }
    let ref_sample = if set.region[hits[n].0] == win.0 { n } else { (0..n).find(|&s| set.region[hits[s].0] == win.0).unwrap_or(n) };
    let (rf, rd, rt, rw) = hits[ref_sample];
    // Walk from the reference face across free and Shift edges only, within reach of the face.
    let reach = WALK_REACH * diameter + rd;
    let mut order: Vec<(usize, bool)> = vec![(rf, false)];
    let mut seen: HashMap<usize, (), FxBuild> = HashMap::default();
    seen.insert(rf, ());
    let mut queue = VecDeque::from([0usize]);
    while let Some(at) = queue.pop_front() {
        let (f, shifted) = order[at];
        for h in src.faces.indptr[f] as usize..src.faces.indptr[f + 1] as usize {
            let g = src.across[h];
            if g < 0 || src.crossing[h] == 2 || seen.contains_key(&(g as usize)) || dist(src.centre[g as usize], c) > reach {
                continue;
            }
            seen.insert(g as usize, ());
            order.push((g as usize, shifted || src.crossing[h] == 1));
            queue.push_back(order.len() - 1);
        }
    }
    let mut out = FaceRead {
        ids: Vec::with_capacity(3 * n),
        w: Vec::with_capacity(3 * n),
        shift: Vec::with_capacity(n),
        slid: Vec::with_capacity(n),
        ref_ids: src.tri[rt].map(|h| h as i64),
        ref_w: rw,
        ref_face: rf as i64,
        orient: if dot(newell(res_points, face), src.normal[rf]) < 0.0 { -1.0 } else { 1.0 },
        region: win.0,
    };
    for (i, &q) in corners.iter().enumerate() {
        let mut best = (f64::INFINITY, 0usize, [0.0; 3], false);
        for &(f, shifted) in &order {
            let (d2, t, w) = src.on_face(f, q);
            if d2 < best.0 {
                best = (d2, t, w, shifted);
            }
        }
        let tri = src.tri[best.1];
        let mut w = best.2;
        let slid = best.0.sqrt() > hits[i].1 + SLIDE_EDGE * edge;
        if slid {
            let (a, b, cc) = (src.corner_point(tri[0]), src.corner_point(tri[1]), src.corner_point(tri[2]));
            if let Some(x) = plane_bary(q, a, b, cc) {
                w = x;
            }
        }
        out.ids.extend(tri.iter().map(|&h| h as i64));
        out.w.extend_from_slice(&w);
        out.shift.push(best.3);
        out.slid.push(slid);
    }
    out
}

/// Half the smallest median gap of the Soft and Hard chains whose two sides lie in one side region (self-seams), per
/// region, in UV units; infinite for a region with none. At least `IGNORE_PX`.
pub fn join_tolerance(set: &SeamSet) -> Vec<f64> {
    let mut join = vec![f64::INFINITY; set.n_regions];
    let region_of_island = {
        let mut r = vec![0i64; set.n_islands];
        for (f, &i) in set.island.iter().enumerate() {
            r[i as usize] = set.region[f];
        }
        r
    };
    for c in 0..set.chains() {
        let [a, b] = set.chain_island[c];
        let (ra, rb) = (region_of_island[a as usize], region_of_island[b as usize]);
        if ra != rb || set.class[c] < ChainClass::Soft as u8 {
            continue;
        }
        let mut gaps = set.gap_px[set.run_indptr[c] as usize..set.run_indptr[c + 1] as usize].to_vec();
        gaps.sort_by(|x, y| x.total_cmp(y));
        let median = gaps[gaps.len() / 2];
        let tol = (0.5 * median).max(IGNORE_PX) / set.texture_px;
        join[ra as usize] = join[ra as usize].min(tol);
    }
    join
}

/// Per-side lookup for a result mesh against the source (SPEC 4.8 steps 1-3). Runs on all cores; checks `stop` per face.
pub fn lookup(src_points: &[f64], src_faces: &Faces, set: &SeamSet, res_points: &[f64], res_faces: &Faces, stop: &AtomicBool) -> Lookup {
    let src = source(src_points, src_faces, set);
    let tree = FaceTree::new(src_points, src_faces.indptr, src_faces.flat);
    let nf = res_faces.count();
    let reads: Vec<Option<FaceRead>> = (0..nf)
        .into_par_iter()
        .map(|f| {
            if stop.load(Ordering::Relaxed) || src_faces.count() == 0 || res_faces.face(f).len() < 3 {
                return None;
            }
            Some(read_face(&src, set, &tree, res_points, res_faces.face(f)))
        })
        .collect();
    let corners = res_faces.flat.len();
    let mut out = Lookup {
        ids: vec![0; 3 * corners],
        w: vec![0.0; 3 * corners],
        shift: vec![false; corners],
        slid: vec![false; corners],
        ref_ids: vec![0; 3 * nf],
        ref_w: vec![0.0; 3 * nf],
        ref_face: vec![-1; nf],
        orient: vec![1.0; nf],
        region: vec![-1; nf],
        join: join_tolerance(set),
        done: !stop.load(Ordering::Relaxed),
    };
    for (f, read) in reads.into_iter().enumerate() {
        let Some(r) = read else { continue };
        let s = res_faces.indptr[f] as usize;
        out.ids[3 * s..3 * s + r.ids.len()].copy_from_slice(&r.ids);
        out.w[3 * s..3 * s + r.w.len()].copy_from_slice(&r.w);
        out.shift[s..s + r.shift.len()].copy_from_slice(&r.shift);
        out.slid[s..s + r.slid.len()].copy_from_slice(&r.slid);
        out.ref_ids[3 * f..3 * f + 3].copy_from_slice(&r.ref_ids);
        out.ref_w[3 * f..3 * f + 3].copy_from_slice(&r.ref_w);
        out.ref_face[f] = r.ref_face;
        out.orient[f] = r.orient;
        out.region[f] = r.region;
    }
    out
}

/// UVs per result corner for one source UV layer (2 per source corner), and the expected UV winding per result face
/// (the reference source face's winding times the orientation sign; 0 when unknown). NaN where the lookup is unfinished.
pub fn evaluate(look: &Lookup, src_faces: &Faces, layer: &[f64], res_faces: &Faces) -> (Vec<f64>, Vec<f64>) {
    let at = |ids: &[i64], w: &[f64]| {
        let mut uv = [0.0; 2];
        for k in 0..3 {
            uv[0] += w[k] * layer[2 * ids[k] as usize];
            uv[1] += w[k] * layer[2 * ids[k] as usize + 1];
        }
        uv
    };
    let nf = res_faces.count();
    let mut uv = vec![f64::NAN; 2 * res_faces.flat.len()];
    let mut expected = vec![0.0; nf];
    for f in 0..nf {
        let rf = look.ref_face[f];
        if rf < 0 {
            continue;
        }
        let r = at(&look.ref_ids[3 * f..3 * f + 3], &look.ref_w[3 * f..3 * f + 3]);
        for h in res_faces.indptr[f] as usize..res_faces.indptr[f + 1] as usize {
            let mut x = at(&look.ids[3 * h..3 * h + 3], &look.w[3 * h..3 * h + 3]);
            if look.shift[h] {
                x = [x[0] + whole(r[0] - x[0]), x[1] + whole(r[1] - x[1])];
            }
            uv[2 * h] = x[0];
            uv[2 * h + 1] = x[1];
        }
        let face = src_faces.face(rf as usize);
        let s = src_faces.indptr[rf as usize] as usize;
        let mut area = 0.0;
        for i in 0..face.len() {
            let j = (i + 1) % face.len();
            area += layer[2 * (s + i)] * layer[2 * (s + j) + 1] - layer[2 * (s + j)] * layer[2 * (s + i) + 1];
        }
        expected[f] = if area > 0.0 { look.orient[f] } else if area < 0.0 { -look.orient[f] } else { 0.0 };
    }
    (uv, expected)
}

/// True when any fan triangle of face f winds against `expected` in UV (or has no area). Faces with expected 0 or NaN
/// UVs are never folded.
pub fn folded(res_faces: &Faces, uv: &[f64], expected: &[f64], f: usize) -> bool {
    if expected[f] == 0.0 {
        return false;
    }
    let s = res_faces.indptr[f] as usize;
    let n = res_faces.indptr[f + 1] as usize - s;
    let q = |i: usize| [uv[2 * (s + i)], uv[2 * (s + i) + 1]];
    if q(0)[0].is_nan() {
        return false;
    }
    (1..n.saturating_sub(1)).any(|i| {
        let (a, b, c) = (q(0), q(i), q(i + 1));
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        area * expected[f] <= 0.0
    })
}

/// What the fold fix did: folded faces before and after, rings grown (the most any spot needed), sweeps run, folded
/// faces left alone because no ring can fix them, and whether a solve stalled (stopped by the stall rule) or the stop flag ended it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FoldReport {
    pub before: usize,
    pub after: usize,
    pub rings: usize,
    pub sweeps: usize,
    /// Folded faces no ring can fix (a folded triangle whose corners all sit on the region's border).
    pub stuck: usize,
    pub stalled: bool,
    pub stopped: bool,
}

/// Fold fix (SPEC 4.8 step 4): folded faces plus their 1-ring, within their side region, get a harmonic fill (own
/// Gauss-Seidel, inverse-length weights) with the spot's border corners pinned; while a spot still folds it grows by one
/// ring, up to its region's border. A spot that ends with no fewer folds than it started with gets its UVs back. Only
/// corners of faces inside a spot change; every other UV stays bit for bit.
///
/// Corners of one vertex in one region share an unknown unless their UVs are farther apart than `join[region]` (UV
/// units; half the region's smallest self-seam gap, so the two sides of a seam inside one island stay apart). An empty
/// `join` joins all.
#[allow(clippy::too_many_arguments)]
pub fn fold_fix(res_points: &[f64], res_faces: &Faces, uv: &mut [f64], region: &[i64], expected: &[f64], join: &[f64], texture_px: f64, stop: &AtomicBool) -> FoldReport {
    let nf = res_faces.count();
    let bad: Vec<bool> = (0..nf).map(|f| folded(res_faces, uv, expected, f)).collect();
    let mut report = FoldReport { before: bad.iter().filter(|&&x| x).count(), ..FoldReport::default() };
    if report.before == 0 {
        return report;
    }
    let tex = if texture_px > 0.0 { texture_px } else { TEXTURE_PX_DEFAULT };
    let original = uv.to_vec();
    let n_points = res_points.len() / 3;
    // Faces per vertex.
    let mut vf_ptr = vec![0usize; n_points + 1];
    for &v in res_faces.flat {
        vf_ptr[v as usize + 1] += 1;
    }
    for v in 0..n_points {
        vf_ptr[v + 1] += vf_ptr[v];
    }
    let mut vf = vec![0usize; res_faces.flat.len()];
    let mut fill = vf_ptr.clone();
    for f in 0..nf {
        for &v in res_faces.face(f) {
            vf[fill[v as usize]] = f;
            fill[v as usize] += 1;
        }
    }
    let ring = |set: &[bool], out: &mut Vec<bool>| {
        out.clone_from_slice(set);
        for f in 0..nf {
            if set[f] {
                for &v in res_faces.face(f) {
                    for &g in &vf[vf_ptr[v as usize]..vf_ptr[v as usize + 1]] {
                        if region[g] == region[f] && region[g] >= 0 {
                            out[g] = true;
                        }
                    }
                }
            }
        }
    };
    // A corner on its region's border (a face of another region at its vertex, an open mesh edge, or the other side
    // of a self-seam) never moves; a face with a folded fan triangle made only of such corners cannot be fixed by any
    // ring and is left out (reported as still folded).
    let mut open_edge: HashMap<(i64, i64), usize, FxBuild> = HashMap::default();
    for f in 0..nf {
        let face = res_faces.face(f);
        for i in 0..face.len() {
            let (a, b) = (face[i], face[(i + 1) % face.len()]);
            *open_edge.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let join_of = |r: i64| if r >= 0 && (r as usize) < join.len() { join[r as usize] } else { f64::INFINITY };
    let fixed = |f: usize, i: usize| {
        let face = res_faces.face(f);
        let (v, h) = (face[i], res_faces.indptr[f] as usize + i);
        let (prev, next) = (face[(i + face.len() - 1) % face.len()], face[(i + 1) % face.len()]);
        let on_open = |a: i64, b: i64| open_edge.get(&(a.min(b), a.max(b))).is_none_or(|&n| n == 1);
        on_open(prev, v)
            || on_open(v, next)
            || vf[vf_ptr[v as usize]..vf_ptr[v as usize + 1]].iter().any(|&g| {
                region[g] != region[f]
                    || (res_faces.indptr[g] as usize..res_faces.indptr[g + 1] as usize).any(|c| res_faces.flat[c] == v && uv_gap(uv, c, h) > join_of(region[f]))
            })
    };
    let stuck: Vec<bool> = (0..nf)
        .map(|f| {
            if !bad[f] {
                return false;
            }
            let s = res_faces.indptr[f] as usize;
            let n = res_faces.indptr[f + 1] as usize - s;
            let q = |i: usize| [uv[2 * (s + i)], uv[2 * (s + i) + 1]];
            (1..n - 1).any(|i| {
                let (a, b, c) = (q(0), q(i), q(i + 1));
                let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
                area * expected[f] <= 0.0 && fixed(f, 0) && fixed(f, i) && fixed(f, i + 1)
            })
        })
        .collect();
    report.stuck = stuck.iter().filter(|&&x| x).count();
    let fixable: Vec<bool> = (0..nf).map(|f| bad[f] && !stuck[f]).collect();
    if report.stuck == report.before {
        report.after = report.before;
        return report;
    }
    let mut active = vec![false; nf];
    ring(&fixable, &mut active);
    let mut touched = active.clone();
    let mut rings = 1;
    loop {
        report.rings = report.rings.max(rings);
        let (sweeps, stalled) = harmonic_fill(res_points, res_faces, uv, region, &active, &vf, &vf_ptr, join, tex, stop);
        report.sweeps += sweeps;
        report.stalled |= stalled;
        if stop.load(Ordering::Relaxed) {
            report.stopped = true;
            break;
        }
        // Spots (active faces joined through shared vertices in one region) that still fold grow by one ring.
        let spot = spots(res_faces, region, &active, &vf, &vf_ptr);
        let mut still = vec![false; nf];
        for f in 0..nf {
            if active[f] && !stuck[f] && folded(res_faces, uv, expected, f) {
                still[spot[f] as usize] = true;
            }
        }
        let keep: Vec<bool> = (0..nf).map(|f| active[f] && still[spot[f] as usize]).collect();
        if !keep.iter().any(|&x| x) {
            break;
        }
        let mut grown = vec![false; nf];
        ring(&keep, &mut grown);
        if grown == keep {
            break; // reached the region's border: what still folds is reported
        }
        for f in 0..nf {
            touched[f] |= grown[f];
        }
        active = grown;
        rings += 1;
    }
    // Spots that did not get better get their UVs back.
    let spot = spots(res_faces, region, &touched, &vf, &vf_ptr);
    let (mut was, mut now) = (vec![0usize; nf], vec![0usize; nf]);
    for f in 0..nf {
        if touched[f] {
            was[spot[f] as usize] += bad[f] as usize;
            now[spot[f] as usize] += folded(res_faces, uv, expected, f) as usize;
        }
    }
    for f in 0..nf {
        if touched[f] && now[spot[f] as usize] >= was[spot[f] as usize] {
            let (s, e) = (res_faces.indptr[f] as usize, res_faces.indptr[f + 1] as usize);
            uv[2 * s..2 * e].copy_from_slice(&original[2 * s..2 * e]);
        }
    }
    report.after = (0..nf).filter(|&f| folded(res_faces, uv, expected, f)).count();
    report
}

/// Spot id per active face: the lowest face id of its group (faces of one region sharing a vertex).
fn spots(res_faces: &Faces, region: &[i64], active: &[bool], vf: &[usize], vf_ptr: &[usize]) -> Vec<i64> {
    let nf = res_faces.count();
    let mut dsu = Dsu::new(nf);
    for f in 0..nf {
        if active[f] {
            for &v in res_faces.face(f) {
                for &g in &vf[vf_ptr[v as usize]..vf_ptr[v as usize + 1]] {
                    if active[g] && region[g] == region[f] {
                        dsu.union(f as u32, g as u32);
                    }
                }
            }
        }
    }
    (0..nf as u32).map(|f| dsu.find(f) as i64).collect()
}

fn uv_gap(uv: &[f64], h: usize, g: usize) -> f64 {
    (uv[2 * h] - uv[2 * g]).abs().max((uv[2 * h + 1] - uv[2 * g + 1]).abs())
}

#[allow(clippy::too_many_arguments)]
fn harmonic_fill(points: &[f64], faces: &Faces, uv: &mut [f64], region: &[i64], active: &[bool], vf: &[usize], vf_ptr: &[usize], join: &[f64], tex: f64, stop: &AtomicBool) -> (usize, bool) {
    let join_of = |r: i64| if r >= 0 && (r as usize) < join.len() { join[r as usize] } else { f64::INFINITY };
    // Corner of face g at vertex v.
    let corner_at = |g: usize, v: i64| (faces.indptr[g] as usize..faces.indptr[g + 1] as usize).find(|&h| faces.flat[h] == v);
    // Nodes: corners of one vertex in one region within `join` of the node's first corner, in order of first use. A
    // node with such a corner in a face outside the spot is pinned.
    let mut groups: HashMap<(i64, i64), Vec<usize>, FxBuild> = HashMap::default();
    let (mut value, mut vertex, mut free, mut first): (Vec<[f64; 2]>, Vec<i64>, Vec<bool>, Vec<usize>) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut corner_node = vec![usize::MAX; faces.flat.len()];
    for f in 0..faces.count() {
        if !active[f] {
            continue;
        }
        for h in faces.indptr[f] as usize..faces.indptr[f + 1] as usize {
            let v = faces.flat[h];
            let tol = join_of(region[f]);
            let list = groups.entry((v, region[f])).or_default();
            let id = match list.iter().find(|&&n| uv_gap(uv, first[n], h) <= tol) {
                Some(&n) => n,
                None => {
                    value.push([uv[2 * h], uv[2 * h + 1]]);
                    vertex.push(v);
                    first.push(h);
                    let pinned = vf[vf_ptr[v as usize]..vf_ptr[v as usize + 1]]
                        .iter()
                        .any(|&g| !active[g] && region[g] == region[f] && corner_at(g, v).is_some_and(|c| uv_gap(uv, c, h) <= tol));
                    free.push(!pinned);
                    list.push(value.len() - 1);
                    value.len() - 1
                }
            };
            corner_node[h] = id;
        }
    }
    // Edges of the spot; an edge only one active face uses is on the spot's border and pins both ends.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for f in 0..faces.count() {
        if active[f] {
            let (s, e) = (faces.indptr[f] as usize, faces.indptr[f + 1] as usize);
            for h in s..e {
                let g = if h + 1 == e { s } else { h + 1 };
                let (a, b) = (corner_node[h], corner_node[g]);
                if a != b {
                    pairs.push((a.min(b), a.max(b)));
                }
            }
        }
    }
    pairs.sort_unstable();
    let nn = value.len();
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j] == pairs[i] {
            j += 1;
        }
        if j - i == 1 {
            free[pairs[i].0] = false;
            free[pairs[i].1] = false;
        }
        i = j;
    }
    pairs.dedup();
    let mut deg = vec![0usize; nn + 1];
    for &(a, b) in &pairs {
        deg[a + 1] += 1;
        deg[b + 1] += 1;
    }
    for i in 0..nn {
        deg[i + 1] += deg[i];
    }
    let mut nbr = vec![(0usize, 0.0f64); deg[nn]];
    let mut put = deg.clone();
    for &(a, b) in &pairs {
        let w = 1.0 / dist(p3(points, vertex[a]), p3(points, vertex[b])).max(1e-12);
        nbr[put[a]] = (b, w);
        put[a] += 1;
        nbr[put[b]] = (a, w);
        put[b] += 1;
    }
    let goal = STOP_TEXEL_SHARE / tex;
    let mut history: Vec<f64> = Vec::new();
    let mut stalled = false;
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let mut most = 0.0f64;
        for i in 0..nn {
            if !free[i] {
                continue;
            }
            let (mut su, mut sv, mut sw) = (0.0, 0.0, 0.0);
            for &(j, w) in &nbr[deg[i]..deg[i + 1]] {
                su += w * value[j][0];
                sv += w * value[j][1];
                sw += w;
            }
            if sw > 0.0 {
                let next = [su / sw, sv / sw];
                most = most.max((next[0] - value[i][0]).abs().max((next[1] - value[i][1]).abs()));
                value[i] = next;
            }
        }
        history.push(most);
        if most <= goal {
            break;
        }
        let it = history.len();
        if it > STALL_WINDOW && most > (1.0 - STALL_DROP) * history[it - 1 - STALL_WINDOW] {
            stalled = true;
            break;
        }
    }
    for h in 0..faces.flat.len() {
        let id = corner_node[h];
        if id != usize::MAX && free[id] {
            uv[2 * h] = value[id][0];
            uv[2 * h + 1] = value[id][1];
        }
    }
    (history.len(), stalled)
}

/// What one `transfer` call hands back: UVs per result corner (layer given), the lookup (for other layers), the
/// expected winding per face and the fold fix report.
pub struct Transfer {
    pub uv: Vec<f64>,
    pub look: Lookup,
    pub expected: Vec<f64>,
    pub fold: FoldReport,
}

/// Per-side transfer in one call: lookup, UVs of `layer`, fold fix. With `stop` set the lookup ends early
/// (`look.done` false) and the fold fix is skipped.
#[allow(clippy::too_many_arguments)]
pub fn transfer(src_points: &[f64], src_faces: &Faces, set: &SeamSet, layer: &[f64], res_points: &[f64], res_faces: &Faces, fix_folds: bool, stop: &AtomicBool) -> Transfer {
    let look = lookup(src_points, src_faces, set, res_points, res_faces, stop);
    let (mut uv, expected) = evaluate(&look, src_faces, layer, res_faces);
    let fold = if fix_folds && look.done { fold_fix(res_points, res_faces, &mut uv, &look.region, &expected, &look.join, set.texture_px, stop) } else { FoldReport::default() };
    Transfer { uv, look, expected, fold }
}

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let flat = [0_i64, 1, 2, 3, 1, 4, 5, 2];
    let indptr = [0_i64, 4, 8];
    let faces = Faces { indptr: &indptr, flat: &flat };
    let n_edges = edges(&halves(&faces)).len();
    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0];
    let tree = FaceTree::new(&points, &indptr, &flat);
    let stop = AtomicBool::new(false);
    let (done, rows) = nearest_faces(&tree, &[0.5, 0.5, 0.5], &stop);
    usize::from(n_edges == 7 && done && rows[0].1 >= 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (nx x ny) quad grid on [0, w] x [0, h] in the plane z = 0; points per corner-vertex (welded).
    fn grid(nx: usize, ny: usize, w: f64, h: f64) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut points = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                points.extend_from_slice(&[w * i as f64 / nx as f64, h * j as f64 / ny as f64, 0.0]);
            }
        }
        let (mut indptr, mut flat) = (vec![0i64], Vec::new());
        for j in 0..ny {
            for i in 0..nx {
                let v = (j * (nx + 1) + i) as i64;
                flat.extend_from_slice(&[v, v + 1, v + nx as i64 + 2, v + nx as i64 + 1]);
                indptr.push(flat.len() as i64);
            }
        }
        (points, indptr, flat)
    }

    /// Two islands split at x = 4 on the 8x8 plane: left UV = (x, y)/8, right UV = (x/8 + 0.25, y/8).
    fn two_islands() -> (Vec<f64>, Vec<i64>, Vec<i64>, Vec<f64>) {
        let (points, indptr, flat) = grid(8, 8, 8.0, 8.0);
        let mut uv = Vec::new();
        for f in 0..indptr.len() - 1 {
            let right = (f % 8) >= 4;
            for h in indptr[f] as usize..indptr[f + 1] as usize {
                let v = flat[h] as usize;
                uv.push(points[3 * v] / 8.0 + if right { 0.25 } else { 0.0 });
                uv.push(points[3 * v + 1] / 8.0);
            }
        }
        (points, indptr, flat, uv)
    }

    /// Open tube around z: `nu` faces around, `nv` up; UV u = angle / 2pi (seam at angle 0, u from 0 to `wrap`), v = z.
    /// Points are welded (one per place); the seam is in the per-corner UVs.
    fn tube(nu: usize, nv: usize, turn: f64, wrap: f64) -> (Vec<f64>, Vec<i64>, Vec<i64>, Vec<f64>) {
        let mut points = Vec::new();
        for j in 0..=nv {
            for i in 0..nu {
                let a = std::f64::consts::TAU * (i as f64 + turn) / nu as f64;
                points.extend_from_slice(&[a.cos(), a.sin(), j as f64 * 0.5]);
            }
        }
        let (mut indptr, mut flat, mut uv) = (vec![0i64], Vec::new(), Vec::new());
        for j in 0..nv {
            for i in 0..nu {
                let i1 = (i + 1) % nu;
                let ids = [j * nu + i, j * nu + i1, (j + 1) * nu + i1, (j + 1) * nu + i];
                let us = [i as f64, (i + 1) as f64, (i + 1) as f64, i as f64];
                let vs = [j as f64, j as f64, (j + 1) as f64, (j + 1) as f64];
                for k in 0..4 {
                    flat.push(ids[k] as i64);
                    uv.push(wrap * (us[k] + turn) / nu as f64);
                    uv.push(vs[k] / nv as f64);
                }
                indptr.push(flat.len() as i64);
            }
        }
        (points, indptr, flat, uv)
    }

    fn faces<'a>(indptr: &'a [i64], flat: &'a [i64]) -> Faces<'a> {
        Faces { indptr, flat }
    }

    #[test]
    fn probe_builds() {
        assert_eq!(probe(), 1);
    }

    #[test]
    fn plane_two_islands_one_chain_per_side_uvs() {
        let (points, indptr, flat, uv) = two_islands();
        let f = faces(&indptr, &flat);
        let set = extract(&points, &f, &uv, &[], &[], 1024.0, UV_TOL_PX);
        assert_eq!(set.n_islands, 2);
        assert_eq!(set.chains(), 1);
        assert_eq!(set.run_vertex.len(), 9);
        assert_eq!(set.class[0], ChainClass::Soft as u8);
        assert!((set.gap_px[0] - 256.0).abs() < 1e-9);
        // Result = the same grid: every face on one side, corners on the seam read their own side's UVs.
        let stop = AtomicBool::new(false);
        let t = transfer(&points, &f, &set, &uv, &points, &f, true, &stop);
        let drift = t.uv.iter().zip(&uv).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) * 1024.0;
        assert!(drift < 1e-6, "drift {drift} px");
        assert_eq!(t.fold.before, 0);
    }

    #[test]
    fn plane_offset_result_reads_its_side() {
        let (points, indptr, flat, uv) = two_islands();
        let f = faces(&indptr, &flat);
        let set = extract(&points, &f, &uv, &[], &[], 1024.0, UV_TOL_PX);
        // Result: the same grid lifted by 0.2 edge and stretched a little in x (seam still on a grid line).
        let res: Vec<f64> = points.chunks(3).flat_map(|p| [4.0 + (p[0] - 4.0) * 1.01, p[1], 0.2]).collect();
        let stop = AtomicBool::new(false);
        let t = transfer(&points, &f, &set, &uv, &res, &f, true, &stop);
        for h in 0..flat.len() {
            let fid = (0..indptr.len() - 1).find(|&k| indptr[k + 1] as usize > h).unwrap();
            let x = res[3 * flat[h] as usize].clamp(0.0, 8.0);
            let right = fid % 8 >= 4;
            let want = x / 8.0 + if right { 0.25 } else { 0.0 };
            assert!((t.uv[2 * h] - want).abs() * 1024.0 < 1e-6, "corner {h}");
        }
    }

    #[test]
    fn self_seam_tube_reads_each_side() {
        // One island cut once (u from 0 to 1, no repeat): the seam corners must take their own face's side.
        let (points, indptr, flat, uv) = tube(12, 4, 0.0, 1.0);
        let f = faces(&indptr, &flat);
        let set = extract(&points, &f, &uv, &[], &[false], 1024.0, UV_TOL_PX);
        assert_eq!(set.n_islands, 1);
        assert_eq!(set.chains(), 1);
        assert_eq!(set.class[0], ChainClass::Soft as u8);
        let stop = AtomicBool::new(false);
        let t = transfer(&points, &f, &set, &uv, &points, &f, true, &stop);
        let drift = t.uv.iter().zip(&uv).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) * 1024.0;
        assert!(drift <= 1e-6, "drift {drift} px");
        assert_eq!(t.fold.after, 0);
    }

    #[test]
    fn wrap_tube_is_shift_on_repeat_and_reads_continuous() {
        let (points, indptr, flat, uv) = tube(12, 4, 0.0, 1.0);
        let f = faces(&indptr, &flat);
        // u jumps by exactly one tile at the seam: Shift with repeat, Soft without.
        let set = extract(&points, &f, &uv, &[], &[true], 1024.0, UV_TOL_PX);
        assert_eq!(set.class[0], ChainClass::Shift as u8);
        assert_eq!(set.n_regions, 1);
        let soft = extract(&points, &f, &uv, &[], &[false], 1024.0, UV_TOL_PX);
        assert_eq!(soft.class[0], ChainClass::Soft as u8);
        // Result turned by half a face: faces straddle the wrap and must read continuous texture (span < 1 tile).
        let (rp, ri, rf, _) = tube(12, 4, 0.5, 1.0);
        let r = faces(&ri, &rf);
        let stop = AtomicBool::new(false);
        let t = transfer(&points, &f, &set, &uv, &rp, &r, true, &stop);
        for k in 0..r.count() {
            let us: Vec<f64> = (ri[k] as usize..ri[k + 1] as usize).map(|h| t.uv[2 * h]).collect();
            let span = us.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - us.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(span < 0.1, "face {k} spans {span} in u");
        }
        // The same texture as the source (UV modulo 1), within a pixel at 1024 on the chord-vs-arc gap.
        for h in 0..rf.len() {
            let a = std::f64::consts::TAU * ((h / 4 % 12) as f64 + [0.5, 1.5, 1.5, 0.5][h % 4]) / 12.0;
            let want = (a / std::f64::consts::TAU).rem_euclid(1.0);
            let got = t.uv[2 * h].rem_euclid(1.0);
            let d = (got - want).abs().min(1.0 - (got - want).abs());
            assert!(d * 1024.0 < 1.0, "corner {h}: {got} vs {want}");
        }
        assert_eq!(t.fold.after, 0);
    }

    #[test]
    fn fold_fix_on_non_convex_ring_folds_no_neighbour() {
        // 8x8 plane; the middle vertex (4, 4) gets a non-convex 1-ring (its left neighbour pushed in, its right one
        // pulled down) and a UV thrown outside that ring, so its faces fold.
        let (mut points, indptr, flat) = grid(8, 8, 8.0, 8.0);
        points[3 * 39] += 0.8;
        points[3 * 41 + 1] -= 0.6;
        let f = faces(&indptr, &flat);
        let mut uv: Vec<f64> = flat.iter().flat_map(|&v| [points[3 * v as usize] / 8.0, points[3 * v as usize + 1] / 8.0]).collect();
        let before = uv.clone();
        for h in 0..flat.len() {
            if flat[h] == 40 {
                uv[2 * h] = 0.4;
                uv[2 * h + 1] = 0.7;
            }
        }
        let region = vec![0i64; f.count()];
        let expected = vec![1.0; f.count()];
        let folds = (0..f.count()).filter(|&k| folded(&f, &uv, &expected, k)).count();
        assert!(folds > 0);
        let stop = AtomicBool::new(false);
        let report = fold_fix(&points, &f, &mut uv, &region, &expected, &[], 1024.0, &stop);
        assert_eq!(report.before, folds);
        assert_eq!(report.after, 0, "{report:?}");
        assert!(!report.stalled);
        // Faces three or more rings away keep their UVs bit for bit.
        for k in 0..f.count() {
            let far = (indptr[k] as usize..indptr[k + 1] as usize).all(|h| {
                let v = flat[h] as i64;
                (v % 9 - 4).abs().max((v / 9 - 4).abs()) >= 3
            });
            if far {
                for h in indptr[k] as usize..indptr[k + 1] as usize {
                    assert_eq!((uv[2 * h], uv[2 * h + 1]), (before[2 * h], before[2 * h + 1]), "face {k} moved");
                }
            }
        }
    }

    #[test]
    fn triage_makes_small_islands_hard_within_budget() {
        // 8x8 plane, a 1x1-face island in the middle with its own UVs: smaller than 3 edges across.
        let (points, indptr, flat) = grid(8, 8, 8.0, 8.0);
        let mut uv = Vec::new();
        for k in 0..indptr.len() - 1 {
            let small = k == 27;
            for h in indptr[k] as usize..indptr[k + 1] as usize {
                let v = flat[h] as usize;
                uv.push(points[3 * v] / 8.0 + if small { 0.5 } else { 0.0 });
                uv.push(points[3 * v + 1] / 8.0);
            }
        }
        let f = faces(&indptr, &flat);
        let mut set = extract(&points, &f, &uv, &[], &[], 1024.0, UV_TOL_PX);
        assert_eq!(set.n_islands, 2);
        assert_eq!(set.chains(), 1); // a closed loop with no corner
        assert_eq!(set.run_vertex.len(), 5);
        let t = triage(&mut set, &points, &f, 64.0, 0.5, None, COST_A, COST_B);
        assert_eq!(t.hard_islands, vec![1]);
        assert!((t.hard_cost[0] - 8.0).abs() < 1e-9); // 4 edges of perimeter / edge 1 + 4
        assert_eq!(set.class[0], ChainClass::Hard as u8);
        let mut tight = extract(&points, &f, &uv, &[], &[], 1024.0, UV_TOL_PX);
        let t = triage(&mut tight, &points, &f, 64.0, 0.1, None, COST_A, COST_B);
        assert!(t.hard_islands.is_empty() && t.left_islands == vec![1]);
    }

    #[test]
    fn stop_hands_back_unfinished() {
        let (points, indptr, flat, uv) = two_islands();
        let f = faces(&indptr, &flat);
        let set = extract(&points, &f, &uv, &[], &[], 1024.0, UV_TOL_PX);
        let stop = AtomicBool::new(true);
        let t = transfer(&points, &f, &set, &uv, &points, &f, true, &stop);
        assert!(!t.look.done);
        assert!(t.look.ref_face.iter().all(|&r| r < 0));
    }
}
