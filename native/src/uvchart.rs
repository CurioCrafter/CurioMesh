// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Row Charts (package K6, spec 6.1-6.6): patches, merge tree, disk cuts and Keep going cut moves on a quad result.
//! Our own code, written from the papers named at each piece:
//!
//! - Irregular points and patches: one trace per outgoing edge of every irregular point, all advancing one edge per
//!   step and stopping at a border, a forced line or another trace (motorcycle graphs: Eppstein, Goodrich, Kim,
//!   Tamstorf 2008, "Motorcycle graphs: canonical quad mesh partitioning"). Turning vertices of forced lines count as
//!   irregular so patches stay rectangles (ours).
//! - Topology check: Euler characteristic per patch over corner wedges (V' - E' + F); an annulus is cut along its
//!   cheapest straight crossing row, anything else by tree-cotree (Eppstein 2003, "Dynamic generators of topologically
//!   embedded graphs") with a maximum dual spanning tree under the cut cost, dangling branches pruned (ours: the cost).
//! - Cut cost: `length_px`, or `seen_px + length_px / L` with seams out of sight (idea of visibility-weighted seams:
//!   Sheffer, Hart 2002, "Seamster"; the two measured terms are ours).
//! - Merge tree: greedy pair merges ranked by cost removed per stretch added (Garland, Willmott, Heckbert 2001,
//!   "Hierarchical face clustering on polygonal surfaces"; bounded distortion after Sorkine, Cohen-Or, Goldenthal,
//!   Lischinski 2002, "Bounded-distortion piecewise mesh parameterization"). Grid unions are measured exactly from the
//!   glued grid, no solve; pole unions sit in a lazy queue and get a band-only ARAP (`uvflat::quick_stretch`) when they
//!   reach the top (ours). `level(x)` applies every recorded merge whose stretch fits, skipping the others (ours).
//!   A union is refused when its layout would lie over itself (a cap closing around a pole of valence 5 or more):
//!   a flip-free disk layout is one-to-one exactly when its border does not cross itself (the border condition for
//!   one-to-one piecewise linear maps, Floater 2003, "One-to-one piecewise linear mappings over triangulations"),
//!   so only border sides that moved are tested, against a cell grid (ours).
//! - Keep going: one discrete cut move per pass on patch borders, accepted only when the score drops (idea of
//!   discrete cut moves: Li, Kaufman, Kim, Sorkine-Hornung, Jacobson, Sheffer 2018, "OptCuts"; our moves and score).

use std::cmp::Ordering as Cmp;
use std::collections::{BTreeSet, BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;

use crate::close::STOPPED;
use crate::topo::{edges, halves, Dsu, Faces, FxBuild, PARALLEL_MIN};
use crate::uvflat;

/// Frozen cross-kernel input (K6): the part of every face. Passed in, never computed here, because `segment.rs` is
/// not compiled into the standalone. Material borders are passed in the same array (one label per part and material).
pub type PartLabel<'a> = &'a [u32];

// Constants table (the NumPy twin `triadq/uvchart.py` holds the same names and values).
/// L in `cost = seen_px + length_px / L` (SPEC 6.3 start value; P1 tunes).
pub const HIDDEN_L: f64 = 20.0;
/// Default Stretch allowed (percent): the merge-tree level x.
pub const STRETCH_ALLOWED_PCT: f64 = 10.0;
/// The merge tree records merges up to a chart stretch of `1 + BUILD_STRETCH`. A setting, not a cap: a level above it
/// says so and Again with a higher value records more ("Higher can take longer").
pub const BUILD_STRETCH: f64 = 1.0;
/// The merge tree runs in rounds of rising stretch: the default Stretch allowed times powers of this factor, from
/// `ROUND_LOWEST` up to the build stretch. A level at a round's stretch is then that round's greedy result, with no
/// patch left alone because its merge came after a chart had grown past the level (ours).
pub const ROUND_FACTOR: f64 = 2.0;
pub const ROUND_LOWEST: f64 = 0.01;
/// Rings around the glued border that a pole union's band ARAP frees.
pub const BAND_RINGS: usize = 2;
/// Pixels per mesh unit when the caller gives none: this many across the bounding box diagonal.
pub const PX_PER_DIAGONAL: f64 = 1024.0;
/// A grid layout is kept while its stretch is within this share of the best layout (else ARAP lays the chart out).
pub const GRID_SLACK: f64 = 0.01;
/// Stretch changes this small count as none (rounding noise must not reorder the queue).
pub const SAME_STRETCH: f64 = 1e-9;
/// Numbers in `Tree::numbers` (see `num`).
pub const NUMBERS_LEN: usize = 21;

/// Columns of `Tree::numbers`.
pub mod num {
    pub const PATCHES: usize = 0;
    pub const IRREGULAR: usize = 1;
    pub const TURNING: usize = 2;
    pub const TRACES: usize = 3;
    pub const ANNULI: usize = 4; // patches cut by the Euler check (annulus or other)
    pub const EULER_CUT_EDGES: usize = 5;
    pub const NOT_DISK_PATCHES: usize = 6; // left after the Euler check (0 expected)
    pub const SEGMENTS: usize = 7;
    pub const MERGES: usize = 8;
    pub const GRID_MERGES: usize = 9;
    pub const POLE_MERGES: usize = 10;
    pub const QUICK_CALLS: usize = 11; // uvflat::quick_stretch calls (pole candidates that reached the top)
    pub const EXACT_EVALS: usize = 12;
    pub const BUILD_MS: usize = 13;
    pub const STOPPED: usize = 14;
    pub const GRID_PATCHES: usize = 15;
    pub const DROPPED: usize = 16; // candidates above 1 + build_stretch (not recorded; Again with more records them)
    pub const TOP_STRETCH: usize = 17; // largest recorded merge stretch
    pub const FORCED_EDGES: usize = 18;
    pub const PX_PER_UNIT: usize = 19;
    pub const LAY_OVER: usize = 20; // candidates refused because the union would lie over itself (pole cones)
}

const NONE: u32 = u32::MAX;

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let flat = [0_i64, 1, 2, 3, 1, 4, 5, 2];
    let indptr = [0_i64, 4, 8];
    let n_edges = edges(&halves(&Faces { indptr: &indptr, flat: &flat })).len();
    let mut dsu = Dsu::new(3);
    dsu.union(0, 1);
    usize::from(n_edges == 7 && dsu.find(0) == dsu.find(1) && dsu.find(2) != dsu.find(0) && !STOPPED.is_empty())
}

// ---------------------------------------------------------------------------------------------------------------
// Quad topology view
// ---------------------------------------------------------------------------------------------------------------

/// Corners, edges (in `topo::edges` order, the order of `seen_edge_px`), and edges per vertex.
pub struct Topo {
    pub nv: usize,
    pub nf: usize,
    start: Vec<usize>,
    pub flat: Vec<u32>,
    pub cface: Vec<u32>,
    pub cedge: Vec<u32>,
    pub ea: Vec<u32>,
    pub eb: Vec<u32>,
    /// The two corners of a manifold edge (`NONE` second on a border).
    pub euse: Vec<[u32; 2]>,
    /// Border or non-manifold: never glued, never traced.
    pub wall: Vec<bool>,
    vstart: Vec<usize>,
    vedge: Vec<u32>,
    vfstart: Vec<usize>,
    vface: Vec<u32>,
    pub border_v: Vec<bool>,
}

impl Topo {
    pub fn new(faces: &Faces, nv: usize) -> Topo {
        let ed = edges(&halves(faces));
        let ne = ed.len();
        let nf = faces.count();
        let start: Vec<usize> = faces.indptr.iter().map(|&x| x as usize).collect();
        let flat: Vec<u32> = faces.flat.iter().map(|&v| v as u32).collect();
        let mut cface = vec![0u32; flat.len()];
        for f in 0..nf {
            for c in &mut cface[start[f]..start[f + 1]] {
                *c = f as u32;
            }
        }
        let mut cedge = vec![NONE; flat.len()];
        let (mut euse, mut wall) = (vec![[NONE; 2]; ne], vec![false; ne]);
        for e in 0..ne {
            let uses = ed.uses(e);
            for &c in uses {
                cedge[c as usize] = e as u32;
            }
            euse[e] = [uses[0], if uses.len() > 1 { uses[1] } else { NONE }];
            wall[e] = uses.len() != 2 || cface[uses[0] as usize] == cface[uses[1] as usize];
        }
        let (ea, eb): (Vec<u32>, Vec<u32>) = (ed.a.iter().map(|&v| v as u32).collect(), ed.b.iter().map(|&v| v as u32).collect());
        let mut vstart = vec![0usize; nv + 1];
        for e in 0..ne {
            vstart[ea[e] as usize + 1] += 1;
            vstart[eb[e] as usize + 1] += 1;
        }
        for v in 0..nv {
            vstart[v + 1] += vstart[v];
        }
        let mut fill = vstart.clone();
        let mut vedge = vec![0u32; vstart[nv]];
        for e in 0..ne {
            for v in [ea[e], eb[e]] {
                vedge[fill[v as usize]] = e as u32;
                fill[v as usize] += 1;
            }
        }
        let mut border_v = vec![false; nv];
        for e in 0..ne {
            if wall[e] {
                border_v[ea[e] as usize] = true;
                border_v[eb[e] as usize] = true;
            }
        }
        let mut vfstart = vec![0usize; nv + 1];
        for &v in &flat {
            vfstart[v as usize + 1] += 1;
        }
        for v in 0..nv {
            vfstart[v + 1] += vfstart[v];
        }
        let mut fill = vfstart.clone();
        let mut vface = vec![0u32; flat.len()];
        for (h, &v) in flat.iter().enumerate() {
            vface[fill[v as usize]] = cface[h];
            fill[v as usize] += 1;
        }
        Topo { nv, nf, start, flat, cface, cedge, ea, eb, euse, wall, vstart, vedge, vfstart, vface, border_v }
    }

    pub fn ne(&self) -> usize {
        self.ea.len()
    }
    pub fn corners(&self) -> usize {
        self.flat.len()
    }
    pub fn face_corners(&self, f: usize) -> std::ops::Range<usize> {
        self.start[f]..self.start[f + 1]
    }
    fn next(&self, h: usize) -> usize {
        let f = self.cface[h] as usize;
        if h + 1 == self.start[f + 1] { self.start[f] } else { h + 1 }
    }
    /// The corner at vertex `v` of side `h` (which runs between `v` and another vertex).
    fn at(&self, h: usize, v: u32) -> usize {
        if self.flat[h] == v { h } else { self.next(h) }
    }
    pub fn vedges(&self, v: usize) -> &[u32] {
        &self.vedge[self.vstart[v]..self.vstart[v + 1]]
    }
    /// Faces around vertex v (once per corner).
    pub fn vfaces(&self, v: usize) -> &[u32] {
        &self.vface[self.vfstart[v]..self.vfstart[v + 1]]
    }
    fn faces_of(&self, e: usize) -> [u32; 2] {
        let [a, b] = self.euse[e];
        [self.cface[a as usize], if b == NONE { NONE } else { self.cface[b as usize] }]
    }
    fn other(&self, e: usize, v: u32) -> u32 {
        if self.ea[e] == v { self.eb[e] } else { self.ea[e] }
    }
    /// The edge that continues `e` straight through a regular interior vertex `v` (the one sharing no face with it).
    fn straight_next(&self, v: u32, e: u32) -> Option<u32> {
        let at = self.vedges(v as usize);
        if at.len() != 4 || self.border_v[v as usize] {
            return None;
        }
        let fe = self.faces_of(e as usize);
        let mut found = None;
        for &x in at {
            if x == e {
                continue;
            }
            let fx = self.faces_of(x as usize);
            if fx.iter().any(|f| *f != NONE && fe.contains(f)) {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(x);
        }
        found
    }
}

/// Union-find with undo (union by size, no path compression), over corners: the wedges of a cut state.
struct Undo {
    parent: Vec<u32>,
    size: Vec<u32>,
    log: Vec<u32>,
}

impl Undo {
    fn new(n: usize) -> Undo {
        Undo { parent: (0..n as u32).collect(), size: vec![1; n], log: Vec::new() }
    }
    fn find(&self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            x = self.parent[x as usize];
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) -> bool {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
        self.log.push(rb);
        true
    }
    fn mark(&self) -> usize {
        self.log.len()
    }
    fn undo(&mut self, mark: usize) {
        while self.log.len() > mark {
            let rb = self.log.pop().unwrap();
            let ra = self.parent[rb as usize];
            self.size[ra as usize] -= self.size[rb as usize];
            self.parent[rb as usize] = rb;
        }
    }
    fn commit(&mut self) {
        self.log.clear();
    }
}

/// The union-find a glue runs on: the build's own `Undo`, or an `Overlay` of tentative unions over it.
trait Uf {
    fn find(&self, x: u32) -> u32;
    fn union(&mut self, a: u32, b: u32) -> bool;
    fn mark(&self) -> usize;
    fn undo(&mut self, mark: usize);
}

impl Uf for Undo {
    fn find(&self, x: u32) -> u32 {
        Undo::find(self, x)
    }
    fn union(&mut self, a: u32, b: u32) -> bool {
        Undo::union(self, a, b)
    }
    fn mark(&self) -> usize {
        Undo::mark(self)
    }
    fn undo(&mut self, mark: usize) {
        Undo::undo(self, mark)
    }
}

/// Tentative unions over a fixed `Undo`, kept in a small map, so measures can run on worker threads.
struct Overlay<'a> {
    base: &'a Undo,
    par: HashMap<u32, u32, FxBuild>,
    log: Vec<u32>,
}

impl<'a> Overlay<'a> {
    fn new(base: &'a Undo) -> Overlay<'a> {
        Overlay { base, par: HashMap::default(), log: Vec::new() }
    }
}

impl Uf for Overlay<'_> {
    fn find(&self, x: u32) -> u32 {
        let mut r = self.base.find(x);
        while let Some(&p) = self.par.get(&r) {
            r = p;
        }
        r
    }
    fn union(&mut self, a: u32, b: u32) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.par.insert(rb, ra);
        self.log.push(rb);
        true
    }
    fn mark(&self) -> usize {
        self.log.len()
    }
    fn undo(&mut self, mark: usize) {
        while self.log.len() > mark {
            let rb = self.log.pop().unwrap();
            self.par.remove(&rb);
        }
    }
}

/// Glue edges (join the corner wedges on both sides): (edges glued, wedge unions that joined two classes).
fn glue<D: Uf>(t: &Topo, dsu: &mut D, list: &[u32]) -> (i64, i64) {
    let mut m = 0i64;
    for &e in list {
        let e = e as usize;
        let [h0, h1] = t.euse[e];
        for v in [t.ea[e], t.eb[e]] {
            m += i64::from(dsu.union(t.at(h0 as usize, v) as u32, t.at(h1 as usize, v) as u32));
        }
    }
    (list.len() as i64, m)
}

/// Wedge roots per corner of a cut state (corners joined across every edge that is not cut and not a wall).
fn wedges(t: &Topo, cut: &[bool]) -> Vec<u32> {
    let mut d = Dsu::new(t.corners());
    for e in 0..t.ne() {
        if !cut[e] && !t.wall[e] {
            let [h0, h1] = t.euse[e];
            for v in [t.ea[e], t.eb[e]] {
                d.union(t.at(h0 as usize, v) as u32, t.at(h1 as usize, v) as u32);
            }
        }
    }
    (0..t.corners() as u32).map(|c| d.find(c)).collect()
}

/// A chart as its own mesh: vertices are wedges, so a cut seam inside the chart is open.
struct Local {
    pts: Vec<f64>,
    ip: Vec<i64>,
    fl: Vec<i64>,
    /// Global corner of each local corner.
    corner: Vec<u32>,
    /// Global face of each local face.
    faces: Vec<u32>,
}

impl Local {
    fn new(t: &Topo, points: &[f64], faces: &[u32], root: impl Fn(u32) -> u32) -> Local {
        let mut id: HashMap<u32, i64, FxBuild> = HashMap::default();
        let (mut pts, mut ip, mut fl, mut corner) = (Vec::new(), vec![0i64], Vec::new(), Vec::new());
        for &f in faces {
            for h in t.face_corners(f as usize) {
                let k = *id.entry(root(h as u32)).or_insert_with(|| {
                    let v = t.flat[h] as usize;
                    pts.extend_from_slice(&points[3 * v..3 * v + 3]);
                    (pts.len() / 3 - 1) as i64
                });
                fl.push(k);
                corner.push(h as u32);
            }
            ip.push(fl.len() as i64);
        }
        Local { pts, ip, fl, corner, faces: faces.to_vec() }
    }
    fn n(&self) -> usize {
        self.pts.len() / 3
    }
    fn chart(&self) -> uvflat::Chart<'_> {
        uvflat::Chart { points: &self.pts, faces: Faces { indptr: &self.ip, flat: &self.fl } }
    }
    /// Per local vertex: the mean of the given per-corner UVs.
    fn gather(&self, corner_uv: &[f64]) -> Vec<f64> {
        let n = self.n();
        let (mut s, mut c) = (vec![0f64; 2 * n], vec![0f64; n]);
        for (k, &h) in self.corner.iter().enumerate() {
            let v = self.fl[k] as usize;
            s[2 * v] += corner_uv[2 * h as usize];
            s[2 * v + 1] += corner_uv[2 * h as usize + 1];
            c[v] += 1.0;
        }
        for v in 0..n {
            s[2 * v] /= c[v].max(1.0);
            s[2 * v + 1] /= c[v].max(1.0);
        }
        s
    }
    fn scatter(&self, uv: &[f64], corner_uv: &mut [f64]) {
        for (k, &h) in self.corner.iter().enumerate() {
            let v = self.fl[k] as usize;
            corner_uv[2 * h as usize] = uv[2 * v];
            corner_uv[2 * h as usize + 1] = uv[2 * v + 1];
        }
    }
}

/// Rigid 2D fit `q ≈ R p + t` (rotation and translation, least squares; Kabsch's 2D case).
fn align(pairs: &[([f64; 2], [f64; 2])]) -> ([f64; 2], [f64; 2]) {
    let k = pairs.len().max(1) as f64;
    let (mut pc, mut qc) = ([0f64; 2], [0f64; 2]);
    for (p, q) in pairs {
        for d in 0..2 {
            pc[d] += p[d] / k;
            qc[d] += q[d] / k;
        }
    }
    let (mut sc, mut ss) = (0.0, 0.0);
    for (p, q) in pairs {
        let (a, b) = ([p[0] - pc[0], p[1] - pc[1]], [q[0] - qc[0], q[1] - qc[1]]);
        sc += a[0] * b[0] + a[1] * b[1];
        ss += a[0] * b[1] - a[1] * b[0];
    }
    let th = ss.atan2(sc);
    let (c, s) = (th.cos(), th.sin());
    ([c, s], [qc[0] - (c * pc[0] - s * pc[1]), qc[1] - (s * pc[0] + c * pc[1])])
}

fn face_area(points: &[f64], t: &Topo, f: usize) -> f64 {
    let r = t.face_corners(f);
    let mut n = [0f64; 3];
    for h in r.clone() {
        let (a, b) = (t.flat[h] as usize, t.flat[t.next(h)] as usize);
        let (p, q) = (&points[3 * a..3 * a + 3], &points[3 * b..3 * b + 3]);
        n[0] += (p[1] - q[1]) * (p[2] + q[2]);
        n[1] += (p[2] - q[2]) * (p[0] + q[0]);
        n[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

// ---------------------------------------------------------------------------------------------------------------
// Input, tree and build
// ---------------------------------------------------------------------------------------------------------------

pub struct Input<'a> {
    pub points: &'a [f64],
    pub faces: Faces<'a>,
    /// Per face (empty = one part). Borders between labels are forced lines.
    pub part_label: PartLabel<'a>,
    /// Forced edges as vertex pairs: hard edges, kept source seams, user "Cut here".
    pub forced: &'a [i64],
    /// Inside pixels per edge from `uvsight` (`topo::edges` order); None = every edge fully seen.
    pub seen_edge_px: Option<&'a [f64]>,
    /// Pixels per mesh unit for `length_px` (0 = `PX_PER_DIAGONAL` across the bounding box diagonal).
    pub px_per_unit: f64,
    /// "Seams out of sight": cost = seen_px + length_px / L; off: cost = length_px.
    pub out_of_sight: bool,
    pub hidden_l: f64,
    /// Merges are recorded up to a chart stretch of `1 + build_stretch`.
    pub build_stretch: f64,
}

/// A run of gluable patch-border edges between two junctions, with one patch on each side (`p == q`: a cut inside
/// one patch, made by the Euler check).
pub struct Segment {
    pub edges: Vec<u32>,
    pub p: u32,
    pub q: u32,
    pub cost: f64,
    pub seen: f64,
}

/// One recorded merge: the segments it glued and the resulting chart stretch (worst 1 %).
pub struct Merge {
    pub segs: Vec<u32>,
    pub stretch: f64,
    pub cost: f64,
    pub pole: bool,
}

/// The charts of one level.
pub struct Level {
    pub chart: Vec<u32>,
    pub cut: Vec<bool>,
    pub charts: usize,
    pub not_disk: usize,
    pub merges_used: usize,
    pub segments_glued: usize,
    pub above_build: bool,
    pub ms: f64,
}

pub struct Tree {
    pub points: Vec<f64>,
    pub topo: Topo,
    pub seen: Vec<f64>,
    pub cost: Vec<f64>,
    pub irregular: Vec<bool>,
    pub turning: Vec<bool>,
    /// Patch borders: traces, forced lines, part borders, Euler cuts.
    pub sep: Vec<bool>,
    /// Never glued: walls, forced lines, part borders.
    pub fixed: Vec<bool>,
    /// Cut in the patch state (sep or wall).
    pub cut_base: Vec<bool>,
    pub patch: Vec<u32>,
    pub n_patches: usize,
    pub patch_chi: Vec<i64>,
    pub segments: Vec<Segment>,
    pub merges: Vec<Merge>,
    /// The merge tree's layout per corner when it ended (flip-free per chart): every level's charts start from it.
    pub start_uv: Vec<f64>,
    pub build_stretch: f64,
    pub numbers: [f64; NUMBERS_LEN],
    cache: Mutex<HashMap<u64, Arc<Level>, FxBuild>>,
}

#[derive(Clone, Copy, PartialEq)]
struct Key {
    class: u8,
    a: f64,
    b: f64,
}

struct Cand {
    key: Key,
    a: u32,
    b: u32,
    va: u32,
    vb: u32,
    exact: bool,
    seq: u64,
}

impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Cmp::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Cmp> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    /// Greater = better (BinaryHeap pops the greatest): lower class, then lower a, then lower b, then lower ids.
    fn cmp(&self, o: &Self) -> Cmp {
        o.key.class
            .cmp(&self.key.class)
            .then(o.key.a.total_cmp(&self.key.a))
            .then(o.key.b.total_cmp(&self.key.b))
            .then(o.a.cmp(&self.a))
            .then(o.b.cmp(&self.b))
            .then(o.seq.cmp(&self.seq))
    }
}

/// Rank of a merge: stretch added <= 0 first, by the share of the smaller chart's border cost that the merge removes
/// (largest first); then by that share / stretch added. The share, not the bare cost, keeps charts compact: a merge that
/// closes most of a chart's border goes before one that glues a short stretch of a long border, so charts do not grow
/// branches (compactness term after Garland, Willmott, Heckbert 2001; the share form is ours). A lower stretch from the
/// merge is not ranked: it is the quantile thinning out, not a better chart.
fn key_of(added: f64, share: f64) -> Key {
    if !added.is_finite() {
        Key { class: 2, a: 0.0, b: -share }
    } else if added <= SAME_STRETCH {
        Key { class: 0, a: 0.0, b: -share }
    } else {
        Key { class: 1, a: -(share / added), b: 0.0 }
    }
}

/// A candidate measured exactly (tentative glue undone afterwards). Grid union: `local` is the whole union laid out
/// again. Pole union: `local` is the band, and `rigid` maps chart b onto chart a.
struct Eval {
    glued: Vec<u32>,
    cost: f64,
    stretch: f64,
    pole: bool,
    local: Local,
    uv: Vec<f64>,
    rigid: Option<([f64; 2], [f64; 2])>,
    /// Stretch of the faces of `local`.
    face_str: Vec<f64>,
}

impl Tree {
    /// Separatrices, patches, Euler check, segments and the merge tree (6.1-6.4). A stop returns the tree reached so
    /// far (`numbers[STOPPED]` = 1): every level of it is valid, and Again resumes from the patches.
    pub fn build(input: &Input, stop: &AtomicBool) -> Result<Tree, String> {
        let started = Instant::now();
        let points = input.points;
        let nv = points.len() / 3;
        let faces = &input.faces;
        let nf = faces.count();
        if !input.part_label.is_empty() && input.part_label.len() != nf {
            return Err("part_label must hold one label per face".into());
        }
        if input.forced.len() % 2 != 0 {
            return Err("forced must hold vertex pairs".into());
        }
        let t = Topo::new(faces, nv);
        let ne = t.ne();
        if let Some(s) = input.seen_edge_px {
            if s.len() != ne {
                return Err(format!("seen_edge_px must hold one number per edge ({ne})"));
            }
        }
        let mut numbers = [0f64; NUMBERS_LEN];
        // Cut cost per edge (6.3).
        let ppu = if input.px_per_unit > 0.0 {
            input.px_per_unit
        } else {
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for v in 0..nv {
                for d in 0..3 {
                    lo[d] = lo[d].min(points[3 * v + d]);
                    hi[d] = hi[d].max(points[3 * v + d]);
                }
            }
            let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
            if diag > 0.0 && diag.is_finite() { PX_PER_DIAGONAL / diag } else { 1.0 }
        };
        numbers[num::PX_PER_UNIT] = ppu;
        let l = if input.hidden_l > 0.0 { input.hidden_l } else { HIDDEN_L };
        let mut length_px = vec![0f64; ne];
        for e in 0..ne {
            let (a, b) = (t.ea[e] as usize, t.eb[e] as usize);
            let d: f64 = (0..3).map(|k| (points[3 * a + k] - points[3 * b + k]).powi(2)).sum();
            length_px[e] = d.sqrt() * ppu;
        }
        let seen: Vec<f64> = match input.seen_edge_px {
            Some(s) => s.iter().map(|x| x.max(0.0)).collect(),
            None => length_px.clone(),
        };
        let cost: Vec<f64> = (0..ne).map(|e| if input.out_of_sight { seen[e] + length_px[e] / l } else { length_px[e] }).collect();
        // Forced lines (6.2.1): walls, given pairs, part borders.
        let mut fixed = t.wall.clone();
        let mut forced_line = vec![false; ne];
        if !input.forced.is_empty() {
            let lookup: HashMap<(u32, u32), u32, FxBuild> = (0..ne).map(|e| ((t.ea[e], t.eb[e]), e as u32)).collect();
            for p in input.forced.chunks(2) {
                if p[0] < 0 || p[1] < 0 {
                    continue;
                }
                let (a, b) = (p[0].min(p[1]) as u32, p[0].max(p[1]) as u32);
                if let Some(&e) = lookup.get(&(a, b)) {
                    forced_line[e as usize] = !t.wall[e as usize];
                }
            }
        }
        if !input.part_label.is_empty() {
            for e in 0..ne {
                let [f0, f1] = t.faces_of(e);
                if !t.wall[e] && input.part_label[f0 as usize] != input.part_label[f1 as usize] {
                    forced_line[e] = true;
                }
            }
        }
        for e in 0..ne {
            fixed[e] |= forced_line[e];
        }
        numbers[num::FORCED_EDGES] = forced_line.iter().filter(|&&x| x).count() as f64;
        // Irregular points (6.1).
        let mut irregular = vec![false; nv];
        for f in 0..nf {
            let r = t.face_corners(f);
            if r.len() != 4 {
                for h in r {
                    irregular[t.flat[h] as usize] = true;
                }
            }
        }
        let mut fcount = vec![0u32; nv];
        for e in 0..ne {
            if forced_line[e] {
                fcount[t.ea[e] as usize] += 1;
                fcount[t.eb[e] as usize] += 1;
            }
        }
        let mut turning = vec![false; nv];
        for v in 0..nv {
            let at = t.vedges(v);
            let val = at.len();
            let nonmanifold = at.iter().any(|&e| t.wall[e as usize] && t.euse[e as usize][1] != NONE);
            if (if t.border_v[v] { val != 3 } else { val != 4 }) || nonmanifold {
                irregular[v] = true;
            }
            let k = fcount[v];
            let turns = match k {
                0 => false,
                2 => {
                    // Interior vertices: the two forced edges must continue each other.
                    if t.border_v[v] || irregular[v] {
                        false
                    } else {
                        let pair: Vec<u32> = at.iter().copied().filter(|&e| forced_line[e as usize]).collect();
                        t.straight_next(v as u32, pair[0]) != Some(pair[1])
                    }
                }
                _ => !t.border_v[v],
            };
            if turns && !irregular[v] {
                turning[v] = true;
                irregular[v] = true;
            }
        }
        numbers[num::IRREGULAR] = irregular.iter().filter(|&&x| x).count() as f64;
        numbers[num::TURNING] = turning.iter().filter(|&&x| x).count() as f64;
        // Traces (6.2.2): every outgoing edge of every irregular point, all one edge per step.
        let blocked: Vec<bool> = (0..nv).map(|v| t.border_v[v] || irregular[v] || fcount[v] > 0).collect();
        let mut claimed = vec![false; nv];
        let mut traced = vec![false; ne];
        let mut active: Vec<(u32, u32)> = Vec::new();
        for v in 0..nv {
            if irregular[v] {
                claimed[v] = true;
                for &e in t.vedges(v) {
                    if !fixed[e as usize] {
                        active.push((v as u32, e));
                    }
                }
            }
        }
        numbers[num::TRACES] = active.len() as f64;
        let mut hits = vec![0u32; nv];
        while !active.is_empty() {
            for &(v, e) in &active {
                hits[t.other(e as usize, v) as usize] += 1;
            }
            let mut next = Vec::with_capacity(active.len());
            for &(v, e) in &active {
                traced[e as usize] = true;
                let w = t.other(e as usize, v);
                if blocked[w as usize] || claimed[w as usize] || hits[w as usize] > 1 {
                    continue;
                }
                if let Some(ne2) = t.straight_next(w, e) {
                    if !fixed[ne2 as usize] {
                        next.push((w, ne2));
                    }
                }
            }
            for &(v, e) in &active {
                let w = t.other(e as usize, v) as usize;
                claimed[w] = true;
                hits[w] = 0;
            }
            active = next;
        }
        let mut sep: Vec<bool> = (0..ne).map(|e| (traced[e] || forced_line[e]) && !t.wall[e]).collect();
        let cut_of = |sep: &[bool]| -> Vec<bool> { (0..ne).map(|e| sep[e] || t.wall[e]).collect() };
        let mut cut_base = cut_of(&sep);
        // Patches (6.2.2) and the topology check (6.2.3).
        let patches_of = |cut: &[bool]| -> (Vec<u32>, usize) {
            let mut d = Dsu::new(nf);
            for e in 0..ne {
                if !cut[e] {
                    let [f0, f1] = t.faces_of(e);
                    d.union(f0, f1);
                }
            }
            let mut id: HashMap<u32, u32, FxBuild> = HashMap::default();
            let p: Vec<u32> = (0..nf as u32).map(|f| {
                let r = d.find(f);
                let k = id.len() as u32;
                *id.entry(r).or_insert(k)
            }).collect();
            (p, id.len())
        };
        let (patch, n_patches) = patches_of(&cut_base);
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_patches];
        for f in 0..nf {
            members[patch[f] as usize].push(f as u32);
        }
        let mut euler_cut = vec![false; ne];
        let mut cut_patches = 0usize;
        for round in 0..2 {
            let chi = chi_per(&t, &cut_base, &patch, n_patches);
            let bad: Vec<usize> = (0..n_patches).filter(|&p| chi[p] != 1).collect();
            if bad.is_empty() {
                break;
            }
            for &p in &bad {
                let path = if round == 0 { annulus_row(&t, &cut_base, &patch, p as u32, &members[p], &cost) } else { None };
                let path = path.unwrap_or_else(|| tree_cotree(&t, &cut_base, &patch, p as u32, &members[p], &cost));
                if !path.is_empty() {
                    cut_patches += usize::from(round == 0);
                }
                for e in path {
                    sep[e as usize] = true;
                    euler_cut[e as usize] = true;
                }
            }
            cut_base = cut_of(&sep);
        }
        let patch_chi = chi_per(&t, &cut_base, &patch, n_patches);
        numbers[num::PATCHES] = n_patches as f64;
        numbers[num::ANNULI] = cut_patches as f64;
        numbers[num::EULER_CUT_EDGES] = euler_cut.iter().filter(|&&x| x).count() as f64;
        numbers[num::NOT_DISK_PATCHES] = patch_chi.iter().filter(|&&c| c != 1).count() as f64;
        // Segments (6.3): gluable patch-border runs between junctions with one patch pair.
        let gluable: Vec<bool> = (0..ne).map(|e| sep[e] && !fixed[e]).collect();
        let mut cut_deg = vec![0u32; nv];
        for e in 0..ne {
            if cut_base[e] {
                cut_deg[t.ea[e] as usize] += 1;
                cut_deg[t.eb[e] as usize] += 1;
            }
        }
        let junction = |v: usize| irregular[v] || cut_deg[v] != 2;
        let pair = |e: usize| {
            let [f0, f1] = t.faces_of(e);
            let (a, b) = (patch[f0 as usize], patch[f1 as usize]);
            (a.min(b), a.max(b))
        };
        let mut seg_of = vec![NONE; ne];
        let mut segments: Vec<Segment> = Vec::new();
        for e0 in 0..ne {
            if !gluable[e0] || seg_of[e0] != NONE {
                continue;
            }
            let id = segments.len() as u32;
            let pr = pair(e0);
            let mut list = vec![e0 as u32];
            seg_of[e0] = id;
            for end in [t.ea[e0], t.eb[e0]] {
                let (mut v, mut e) = (end, e0 as u32);
                while !junction(v as usize) {
                    let ne2 = t.vedges(v as usize).iter().copied().find(|&x| x != e && cut_base[x as usize]);
                    match ne2 {
                        Some(x) if gluable[x as usize] && seg_of[x as usize] == NONE && pair(x as usize) == pr => {
                            seg_of[x as usize] = id;
                            list.push(x);
                            v = t.other(x as usize, v);
                            e = x;
                        }
                        _ => break,
                    }
                }
            }
            let c: f64 = list.iter().map(|&x| cost[x as usize]).sum();
            let s: f64 = list.iter().map(|&x| seen[x as usize]).sum();
            segments.push(Segment { edges: list, p: pr.0, q: pr.1, cost: c, seen: s });
        }
        numbers[num::SEGMENTS] = segments.len() as f64;
        let mut tree = Tree {
            points: points.to_vec(),
            topo: t,
            seen,
            cost,
            irregular,
            turning,
            sep,
            fixed,
            cut_base,
            patch,
            n_patches,
            patch_chi,
            segments,
            merges: Vec::new(),
            start_uv: Vec::new(),
            build_stretch: if input.build_stretch > 0.0 { input.build_stretch } else { BUILD_STRETCH },
            numbers,
            cache: Mutex::new(HashMap::default()),
        };
        let (merges, found, start_uv) = tree.merge_tree(&members, stop);
        tree.merges = merges;
        tree.start_uv = start_uv;
        for (k, x) in found {
            tree.numbers[k] = x;
        }
        tree.numbers[num::BUILD_MS] = started.elapsed().as_secs_f64() * 1e3;
        tree.numbers[num::STOPPED] = f64::from(u8::from(stop.load(Ordering::Relaxed)));
        Ok(tree)
    }

    /// Lay one patch out for the merge tree: its exact grid when that is within `GRID_SLACK` of 1 or of ARAP, else
    /// ARAP (from the grid, or from LSCM when it is no grid). A quad patch that is no rectangle (a trapezoid, a
    /// rhombus) is therefore no grid chart.
    fn patch_layout(&self, local: &Local, stop: &AtomicBool) -> (bool, Vec<f64>) {
        let ch = local.chart();
        let grid = uvflat::grid_layout(&ch).ok();
        let q_grid = grid.as_ref().map(|g| quantile(&self.face_stretch(local, &g.uv), uvflat::STRETCH_QUANTILE));
        if let (Some(g), Some(q)) = (&grid, q_grid) {
            if q <= 1.0 + GRID_SLACK {
                return (true, g.uv.clone());
            }
        }
        let start = if grid.is_some() { uvflat::Start::Given } else { uvflat::Start::Free };
        let opts = uvflat::FlatOpts { rows: uvflat::Rows::Off, start, ..Default::default() };
        let arap = uvflat::flatten_chart(&ch, grid.as_ref().map(|g| &g.uv[..]), None, &opts, stop).ok().map(|o| o.uv);
        match (grid, arap) {
            (Some(g), Some(a)) => {
                let q_arap = quantile(&self.face_stretch(local, &a), uvflat::STRETCH_QUANTILE);
                if q_grid.unwrap() <= q_arap * (1.0 + GRID_SLACK) { (true, g.uv) } else { (false, a) }
            }
            (Some(g), None) => (true, g.uv),
            (None, Some(a)) => (false, a),
            (None, None) => (false, vec![f64::NAN; 2 * local.n()]),
        }
    }

    /// The merge tree (6.4): the merges in merge order and the numbers it measured.
    fn merge_tree(&self, members: &[Vec<u32>], stop: &AtomicBool) -> (Vec<Merge>, Vec<(usize, f64)>, Vec<f64>) {
        let t = &self.topo;
        let np = self.n_patches;
        let mut dsu = Undo::new(t.corners());
        for e in 0..t.ne() {
            if !self.cut_base[e] {
                glue(t, &mut dsu, &[e as u32]);
            }
        }
        dsu.commit();
        let mut b = Build {
            tree: self,
            dsu,
            corner_uv: vec![f64::NAN; 2 * t.corners()],
            face_str: vec![f64::INFINITY; t.nf],
            owner: self.patch.clone(),
            cpar: (0..np as u32).collect(),
            faces: members.to_vec(),
            segs: vec![Vec::new(); np],
            stretch: vec![f64::INFINITY; np],
            grid: vec![false; np],
            chi: self.patch_chi.clone(),
            ver: vec![0; np],
            glued_seg: vec![false; self.segments.len()],
            quick: 0,
            vals: vec![BTreeSet::new(); np],
            last: HashMap::default(),
            bnd: vec![Vec::new(); np],
            refused: HashMap::default(),
            cap: f64::INFINITY,
            last_st: HashMap::default(),
            parked: Vec::new(),
            cloc: vec![0; t.corners()],
            eglue: vec![0; t.ne()],
        };
        for (p, m) in members.iter().enumerate() {
            for &f in m {
                for h in t.face_corners(f as usize) {
                    if self.cut_base[t.cedge[h] as usize] {
                        b.bnd[p].push(h as u32);
                    }
                }
            }
        }
        // Patch layouts.
        let layouts: Vec<(Local, bool, Vec<f64>)> = {
            let dsu = &b.dsu;
            let one = |p: usize| {
                let local = Local::new(t, &self.points, &members[p], |c| dsu.find(c));
                let (g, uv) = self.patch_layout(&local, stop);
                (local, g, uv)
            };
            if t.nf >= PARALLEL_FACES { (0..np).into_par_iter().map(one).collect() } else { (0..np).map(one).collect() }
        };
        for (p, (local, g, uv)) in layouts.into_iter().enumerate() {
            local.scatter(&uv, &mut b.corner_uv);
            let per = self.face_stretch(&local, &uv);
            for (k, &f) in local.faces.iter().enumerate() {
                b.face_str[f as usize] = per[k];
                b.vals[p].insert((bits(per[k]), f));
            }
            b.stretch[p] = quantile(&per, uvflat::STRETCH_QUANTILE);
            b.grid[p] = g;
        }
        let grid_patches = b.grid.iter().filter(|&&g| g).count() as f64;
        for (s, seg) in self.segments.iter().enumerate() {
            b.segs[seg.p as usize].push(s as u32);
            if seg.q != seg.p {
                b.segs[seg.q as usize].push(s as u32);
            }
        }
        let mut heap = Queue::default();
        let mut seq = 0u64;
        let (mut exact_evals, mut dropped, mut grid_merges, mut pole_merges, mut lay_over) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut top = 1.0f64;
        let mut merges: Vec<Merge> = Vec::new();
        for p in 0..np as u32 {
            b.push_for(&mut heap, &mut seq, p);
        }
        // Measured candidates by chart pair, with the versions they were measured at: a measure stays valid while both
        // charts are unchanged, so a candidate that comes back (queued again, or parked to a later round) is not
        // measured twice. Entries of charts that changed are dropped when the store outgrows the live pairs.
        let mut memo: Memo = HashMap::default();
        let mut memo_live = 0usize;
        // Candidates measured above the round's cap wait for the round whose cap they fit (their measure stays valid
        // while both charts are unchanged).
        for (round, &cap) in round_caps(self.build_stretch).iter().enumerate() {
            b.cap = cap;
            if round > 0 {
                for (c, st) in std::mem::take(&mut b.parked) {
                    if b.find(c.a) != c.a || b.find(c.b) != c.b || b.ver[c.a as usize] != c.va || b.ver[c.b as usize] != c.vb {
                        continue;
                    }
                    if st <= 1.0 + cap { heap.push(c) } else { b.parked.push((c, st)) }
                }
            }
            while let Some(c) = heap.pop() {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                if b.find(c.a) != c.a || b.find(c.b) != c.b || b.ver[c.a as usize] != c.va || b.ver[c.b as usize] != c.vb {
                    continue; // stale: the new chart pushed its own candidates
                }
                let hit = matches!(memo.get(&(c.a, c.b)), Some(&(va, vb, _)) if (va, vb) == (c.va, c.vb));
                let ev = if hit {
                    memo.remove(&(c.a, c.b)).unwrap().2
                } else {
                    exact_evals += b.prefetch(&mut heap, &c, &mut memo, stop);
                    match memo.remove(&(c.a, c.b)) {
                        Some((va, vb, ev)) if (va, vb) == (c.va, c.vb) => ev,
                        _ => continue, // the segments cannot be glued into a disk
                    }
                };
                let added = ev.stretch - b.stretch[c.a as usize].max(b.stretch[c.b as usize]);
                if !c.exact {
                    seq += 1;
                    let key = key_of(added, b.share(c.a, c.b, ev.cost));
                    b.last.insert((c.a, c.b), key);
                    heap.push(Cand { key, exact: true, seq, ..c });
                    memo.insert((c.a, c.b), (c.va, c.vb, ev));
                    continue;
                }
                if !(ev.stretch <= 1.0 + self.build_stretch) {
                    dropped += 1;
                    continue;
                }
                b.last_st.insert((c.a, c.b), ev.stretch);
                if ev.stretch > 1.0 + cap {
                    memo.insert((c.a, c.b), (c.va, c.vb, ev));
                    let st = memo[&(c.a, c.b)].2.stretch;
                    b.parked.push((c, st));
                    if memo.len() > 2 * memo_live + 4096 {
                        let ok = |b: &mut Build, k: &(u32, u32), v: &(u32, u32, Eval)| b.find(k.0) == k.0 && b.find(k.1) == k.1 && b.ver[k.0 as usize] == v.0 && b.ver[k.1 as usize] == v.1;
                        memo.retain(|k, v| ok(&mut b, k, v));
                        memo_live = memo.len();
                    }
                    continue;
                }
                if b.lies_over(c.a, c.b, &ev) {
                    lay_over += 1;
                    // Not tried again until the smaller chart changes: the bigger one growing elsewhere seldom frees it.
                    let small = if b.faces[c.a as usize].len() <= b.faces[c.b as usize].len() { c.a } else { c.b };
                    b.refused.insert((c.a, c.b), (small, b.ver[small as usize]));
                    continue;
                }
                top = top.max(ev.stretch);
                if ev.pole { pole_merges += 1 } else { grid_merges += 1 }
                let (stretch, cost, pole, glued) = (ev.stretch, ev.cost, ev.pole, ev.glued.clone());
                let r = b.apply(c.a, c.b, ev);
                merges.push(Merge { segs: glued, stretch, cost, pole });
                b.push_for(&mut heap, &mut seq, r);
            }
            if stop.load(Ordering::Relaxed) {
                break;
            }
        }
        let found = vec![
            (num::GRID_PATCHES, grid_patches),
            (num::MERGES, merges.len() as f64),
            (num::GRID_MERGES, grid_merges as f64),
            (num::POLE_MERGES, pole_merges as f64),
            (num::QUICK_CALLS, b.quick as f64),
            (num::EXACT_EVALS, exact_evals as f64),
            (num::DROPPED, dropped as f64),
            (num::TOP_STRETCH, top),
            (num::LAY_OVER, lay_over as f64),
        ];
        (merges, found, b.corner_uv)
    }

    /// Per-face stretch of a laid-out chart (`uvflat::stretch`; INF for every face when a UV is missing).
    fn face_stretch(&self, local: &Local, uv: &[f64]) -> Vec<f64> {
        if !uv.iter().all(|x| x.is_finite()) {
            return vec![f64::INFINITY; local.faces.len()];
        }
        uvflat::stretch(&local.chart(), uv).1
    }
}

/// Stretch caps of the merge rounds (as `stretch - 1`), rising, the last one `build`.
pub fn round_caps(build: f64) -> Vec<f64> {
    let mut x = STRETCH_ALLOWED_PCT / 100.0;
    while x / ROUND_FACTOR >= ROUND_LOWEST {
        x /= ROUND_FACTOR;
    }
    let mut caps = Vec::new();
    while x < build {
        caps.push(x);
        x *= ROUND_FACTOR;
    }
    caps.push(build);
    caps
}

fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    let at = ((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
    v.select_nth_unstable_by(at, |a, b| a.total_cmp(b));
    v[at]
}

/// Merge-tree state: charts are patch ids (`cpar` points to the surviving one), `owner` is the chart per face.
struct Build<'t> {
    tree: &'t Tree,
    dsu: Undo,
    corner_uv: Vec<f64>,
    face_str: Vec<f64>,
    owner: Vec<u32>,
    cpar: Vec<u32>,
    faces: Vec<Vec<u32>>,
    segs: Vec<Vec<u32>>,
    stretch: Vec<f64>,
    grid: Vec<bool>,
    chi: Vec<i64>,
    ver: Vec<u32>,
    glued_seg: Vec<bool>,
    quick: usize,
    /// Per chart: (stretch bits, face) of its faces, so the worst 1 % is read from the top end only.
    vals: Vec<BTreeSet<(u64, u32)>>,
    /// The last measured key of a chart pair: a pair whose chart grew is queued again under it, not re-measured
    /// at once.
    last: HashMap<(u32, u32), Key, FxBuild>,
    /// Per chart: its border sides (corner h = the side from h to the next corner of its face).
    bnd: Vec<Vec<u32>>,
    /// Pairs refused because the union lay over itself: (smaller chart, its version then).
    refused: HashMap<(u32, u32), (u32, u32), FxBuild>,
    /// The round's cap (as `stretch - 1`), the last measured stretch per pair, and the candidates waiting for a later
    /// round. A pair measured above the cap waits again when one of its charts grows, without a new measure: unions
    /// only stretch more as charts grow, and the round whose cap it fits measures it again.
    cap: f64,
    last_st: HashMap<(u32, u32), f64, FxBuild>,
    parked: Vec<(Cand, f64)>,
    /// Per corner: local vertex + 1 in the union under test (0 = not among the measured faces); per edge: 1 glued by
    /// the union under test, 2 glued for good.
    cloc: Vec<u32>,
    eglue: Vec<u32>,
}

/// The candidate queue in two heaps, not measured yet (`open`) and measured (`done`), popped as one: measuring ahead
/// then takes the top of `open` without wading through measured candidates.
#[derive(Default)]
struct Queue {
    open: BinaryHeap<Cand>,
    done: BinaryHeap<Cand>,
}

impl Queue {
    fn push(&mut self, c: Cand) {
        if c.exact { self.done.push(c) } else { self.open.push(c) }
    }
    fn pop(&mut self) -> Option<Cand> {
        match (self.open.peek(), self.done.peek()) {
            (Some(x), Some(y)) => if x > y { self.open.pop() } else { self.done.pop() },
            (Some(_), None) => self.open.pop(),
            _ => self.done.pop(),
        }
    }
}

/// Measured candidates by chart pair: (versions of a and b when measured, the measure).
type Memo = HashMap<(u32, u32), (u32, u32, Eval), FxBuild>;

/// Candidates measured ahead together on worker threads (`Build::prefetch`).
const PREFETCH: usize = 24;
/// Meshes from this many faces measure ahead and lay out patches on worker threads (each measure is a small band
/// solve, so the crate-wide `PARALLEL_MIN` for array-wide loops is far too high here).
const PARALLEL_FACES: usize = 2048;

/// Per-thread stamps of one measure (vertices and faces of its band).
#[derive(Default)]
struct Scratch {
    vstamp: Vec<u32>,
    fstamp: Vec<u32>,
    stamp: u32,
}

impl Scratch {
    fn fit(&mut self, nv: usize, nf: usize) {
        if self.vstamp.len() != nv || self.fstamp.len() != nf {
            *self = Scratch { vstamp: vec![0; nv], fstamp: vec![0; nf], stamp: 0 };
        }
        if self.stamp == u32::MAX {
            self.vstamp.iter_mut().for_each(|x| *x = 0);
            self.fstamp.iter_mut().for_each(|x| *x = 0);
            self.stamp = 0;
        }
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// Order key of a stretch value (>= 0, or +inf): its bits.
fn bits(x: f64) -> u64 {
    if x.is_nan() { f64::INFINITY.to_bits() } else { x.max(0.0).to_bits() }
}

impl Build<'_> {
    /// `quantile(STRETCH_QUANTILE)` over the union of charts a and b where the faces stamped now take `fresh`.
    fn union_quantile(&self, a: u32, b: u32, fresh: &[f64], fstamp: &[u32], stamp: u32) -> f64 {
        let n = self.faces[a as usize].len() + self.faces[b as usize].len();
        if n == 0 {
            return f64::NAN;
        }
        let idx = ((uvflat::STRETCH_QUANTILE * (n - 1) as f64).round() as usize).min(n - 1);
        let r = n - idx; // the r-th largest
        let mut top: Vec<f64> = fresh.to_vec();
        for c in [a, b] {
            top.extend(self.vals[c as usize].iter().rev().filter(|x| fstamp[x.1 as usize] != stamp).take(r).map(|x| f64::from_bits(x.0)));
        }
        let at = r - 1;
        top.select_nth_unstable_by(at, |x, y| y.total_cmp(x));
        top[at]
    }

    fn find(&mut self, mut p: u32) -> u32 {
        let mut r = p;
        while self.cpar[r as usize] != r {
            r = self.cpar[r as usize];
        }
        while self.cpar[p as usize] != r {
            let next = self.cpar[p as usize];
            self.cpar[p as usize] = r;
            p = next;
        }
        r
    }

    /// Open segments of chart r grouped by the neighbouring chart, each list sorted by cost (highest first).
    fn neighbours(&mut self, r: u32) -> Vec<(u32, Vec<u32>)> {
        let list = self.segs[r as usize].clone();
        let mut by: HashMap<u32, Vec<u32>, FxBuild> = HashMap::default();
        for s in list {
            if self.glued_seg[s as usize] {
                continue;
            }
            let sg = &self.tree.segments[s as usize];
            let (p, q) = (sg.p, sg.q);
            let (rp, rq) = (self.find(p), self.find(q));
            let o = if rp == r { rq } else { rp };
            if o != r {
                by.entry(o).or_default().push(s);
            }
        }
        let segs = &self.tree.segments;
        let mut out: Vec<(u32, Vec<u32>)> = by.into_iter().collect();
        for (_, l) in &mut out {
            l.sort_unstable();
            l.dedup();
            l.sort_by(|&x, &y| segs[y as usize].cost.total_cmp(&segs[x as usize].cost).then(x.cmp(&y)));
        }
        out.sort_by_key(|x| x.0);
        out
    }

    fn between(&mut self, a: u32, b: u32) -> Vec<u32> {
        let (x, y) = if self.segs[a as usize].len() <= self.segs[b as usize].len() { (a, b) } else { (b, a) };
        let list = self.segs[x as usize].clone();
        let mut out: Vec<u32> = Vec::new();
        for s in list {
            if self.glued_seg[s as usize] {
                continue;
            }
            let sg = &self.tree.segments[s as usize];
            let (p, q) = (sg.p, sg.q);
            let (rp, rq) = (self.find(p), self.find(q));
            if (rp == x && rq == y) || (rp == y && rq == x) {
                out.push(s);
            }
        }
        let segs = &self.tree.segments;
        out.sort_unstable();
        out.dedup();
        out.sort_by(|&x, &y| segs[y as usize].cost.total_cmp(&segs[x as usize].cost).then(x.cmp(&y)));
        out
    }

    /// Estimate: the relative length mismatch of the glued sides in the two layouts.
    fn estimate(&self, list: &[u32]) -> (f64, f64) {
        let t = &self.tree.topo;
        let uv = &self.corner_uv;
        let (mut diff, mut total, mut c) = (0.0, 0.0, 0.0);
        for &s in list {
            let sg = &self.tree.segments[s as usize];
            c += sg.cost;
            for &e in &sg.edges {
                let e = e as usize;
                let mut len = [0f64; 2];
                for (k, &h) in t.euse[e].iter().enumerate() {
                    let (h0, h1) = (t.at(h as usize, t.ea[e]), t.at(h as usize, t.eb[e]));
                    len[k] = ((uv[2 * h0] - uv[2 * h1]).powi(2) + (uv[2 * h0 + 1] - uv[2 * h1 + 1]).powi(2)).sqrt();
                }
                diff += (len[0] - len[1]).abs();
                total += len[0].max(len[1]);
            }
        }
        (if total > 0.0 && diff.is_finite() { diff / total } else { f64::INFINITY }, c)
    }

    /// Cost of a chart's open border (its segments not glued yet).
    fn border(&self, r: u32) -> f64 {
        self.segs[r as usize].iter().filter(|&&s| !self.glued_seg[s as usize]).map(|&s| self.tree.segments[s as usize].cost).sum()
    }

    /// Share of the smaller border that gluing `cost` removes (`key_of`).
    fn share(&self, a: u32, b: u32, cost: f64) -> f64 {
        cost / self.border(a).min(self.border(b)).max(f64::MIN_POSITIVE)
    }

    fn push_for(&mut self, heap: &mut Queue, seq: &mut u64, r: u32) {
        if self.chi[r as usize] != 1 {
            return;
        }
        for (o, list) in self.neighbours(r) {
            if self.chi[o as usize] != 1 {
                continue;
            }
            let (a, b) = (r.min(o), r.max(o));
            if matches!(self.refused.get(&(a, b)), Some(&(c, v)) if self.ver[c as usize] == v) {
                continue;
            }
            let key = match self.last.get(&(a, b)) {
                Some(k) => *k,
                None => {
                    let (mis, c) = self.estimate(&list);
                    key_of(mis, self.share(r, o, c))
                }
            };
            *seq += 1;
            let c = Cand { key, a, b, va: self.ver[a as usize], vb: self.ver[b as usize], exact: false, seq: *seq };
            match self.last_st.get(&(a, b)) {
                Some(&st) if st > 1.0 + self.cap => self.parked.push((Cand { exact: true, ..c }, st)),
                _ => heap.push(c),
            }
        }
    }

    /// Measure ahead: the candidate `first` and up to `PREFETCH - 1` more from the top of the open queue (current, not
    /// measured yet) are measured on worker threads into `memo`; the queue keeps every candidate. A measure depends only
    /// on its two charts, so the build makes the same merges as measuring one at a time (a measure whose chart changes
    /// first is simply not used).
    /// Returns the number measured.
    fn prefetch(&mut self, heap: &mut Queue, first: &Cand, memo: &mut Memo, stop: &AtomicBool) -> usize {
        let parallel = self.tree.topo.nf >= PARALLEL_FACES;
        let want = if parallel { PREFETCH } else { 1 };
        let mut batch = vec![(first.a, first.b, first.va, first.vb)];
        let mut back: Vec<Cand> = Vec::new();
        while batch.len() < want && back.len() < want {
            let Some(c) = heap.open.pop() else { break };
            if self.find(c.a) != c.a || self.find(c.b) != c.b || self.ver[c.a as usize] != c.va || self.ver[c.b as usize] != c.vb {
                continue; // stale for good
            }
            let measured = matches!(memo.get(&(c.a, c.b)), Some(&(va, vb, _)) if (va, vb) == (c.va, c.vb));
            if !measured && !batch.iter().any(|x| (x.0, x.1) == (c.a, c.b)) {
                batch.push((c.a, c.b, c.va, c.vb));
            }
            back.push(c);
        }
        heap.open.extend(back);
        let lists: Vec<Vec<u32>> = batch.iter().map(|&(a, b, _, _)| self.between(a, b)).collect();
        let me = &*self;
        let one = |k: usize| {
            SCRATCH.with(|sc| {
                let mut sc = sc.borrow_mut();
                sc.fit(me.tree.topo.nv, me.tree.topo.nf);
                let mut dsu = Overlay::new(&me.dsu);
                let (a, b, _, _) = batch[k];
                me.measure(&mut dsu, &mut sc, a, b, &lists[k], stop)
            })
        };
        let out: Vec<(Option<Eval>, usize)> = if parallel && batch.len() > 1 { (0..batch.len()).into_par_iter().map(one).collect() } else { (0..batch.len()).map(one).collect() };
        let n = out.len();
        for ((a, b, va, vb), (ev, quick)) in batch.into_iter().zip(out) {
            self.quick += quick;
            if let Some(ev) = ev {
                memo.insert((a, b), (va, vb, ev));
            }
        }
        n
    }

    /// Measure the union of charts a and b glued along `list` (tentatively on `dsu`; undone before returning), and
    /// the number of band ARAP calls it made.
    fn measure<D: Uf>(&self, dsu: &mut D, sc: &mut Scratch, a: u32, b: u32, list: &[u32], stop: &AtomicBool) -> (Option<Eval>, usize) {
        let tree = self.tree;
        let t = &tree.topo;
        let mark = dsu.mark();
        let (glued, cost) = glue_disk(t, dsu, &tree.segments, list, 2);
        if glued.is_empty() {
            dsu.undo(mark);
            return (None, 0);
        }
        if self.grid[a as usize] && self.grid[b as usize] {
            // Grid union: the glued grid measured exactly, no solve.
            let union: Vec<u32> = self.faces[a as usize].iter().chain(self.faces[b as usize].iter()).copied().collect();
            let local = Local::new(t, &tree.points, &union, |c| dsu.find(c));
            if let Ok(g) = uvflat::grid_layout(&local.chart()) {
                let per = tree.face_stretch(&local, &g.uv);
                let q = quantile(&per, uvflat::STRETCH_QUANTILE);
                let parts = self.stretch[a as usize].max(self.stretch[b as usize]);
                if q <= 1.0 + GRID_SLACK || q <= parts + GRID_SLACK {
                    dsu.undo(mark);
                    return (Some(Eval { glued, cost, stretch: q, pole: false, local, uv: g.uv, rigid: None, face_str: per }), 0);
                }
                // The glued grid stretches: measure the band ARAP too and keep the better.
                let grid_eval = Eval { glued: glued.clone(), cost, stretch: q, pole: false, local, uv: g.uv, rigid: None, face_str: per };
                let band = self.band_eval(&*dsu, sc, a, b, glued, cost, stop);
                dsu.undo(mark);
                return (Some(if band.stretch < grid_eval.stretch * (1.0 - GRID_SLACK) { band } else { grid_eval }), 1);
            }
        }
        let ev = self.band_eval(&*dsu, sc, a, b, glued, cost, stop);
        dsu.undo(mark);
        (Some(ev), 1)
    }

    /// Pole union (with the glue in place): B onto A by a rigid fit along the glued edges, then ARAP on the band.
    #[allow(clippy::too_many_arguments)]
    fn band_eval<D: Uf>(&self, dsu: &D, sc: &mut Scratch, a: u32, b: u32, glued: Vec<u32>, cost: f64, stop: &AtomicBool) -> Eval {
        let tree = self.tree;
        let t = &tree.topo;
        let mut pairs = Vec::new();
        for &s in &glued {
            for &e in &tree.segments[s as usize].edges {
                let e = e as usize;
                let [h0, h1] = t.euse[e];
                let (ha, hb) = if self.owner[t.cface[h0 as usize] as usize] == a { (h0, h1) } else { (h1, h0) };
                for v in [t.ea[e], t.eb[e]] {
                    let (ca, cb) = (t.at(ha as usize, v), t.at(hb as usize, v));
                    pairs.push(([self.corner_uv[2 * cb], self.corner_uv[2 * cb + 1]], [self.corner_uv[2 * ca], self.corner_uv[2 * ca + 1]]));
                }
            }
        }
        let rigid = align(&pairs);
        sc.stamp += 1;
        let stamp = sc.stamp;
        let in_union = |owner: &[u32], f: u32| owner[f as usize] == a || owner[f as usize] == b;
        let mut front: Vec<u32> = Vec::new();
        for &s in &glued {
            for &e in &tree.segments[s as usize].edges {
                for v in [t.ea[e as usize], t.eb[e as usize]] {
                    if sc.vstamp[v as usize] != stamp {
                        sc.vstamp[v as usize] = stamp;
                        front.push(v);
                    }
                }
            }
        }
        let mut band_v = front.clone();
        for _ in 0..BAND_RINGS {
            let mut next = Vec::new();
            for &v in &front {
                for &f in t.vfaces(v as usize) {
                    if !in_union(&self.owner, f) {
                        continue;
                    }
                    for h in t.face_corners(f as usize) {
                        let w = t.flat[h];
                        if sc.vstamp[w as usize] != stamp {
                            sc.vstamp[w as usize] = stamp;
                            next.push(w);
                        }
                    }
                }
            }
            band_v.extend_from_slice(&next);
            front = next;
        }
        let mut sub: Vec<u32> = Vec::new();
        for &v in &band_v {
            for &f in t.vfaces(v as usize) {
                if in_union(&self.owner, f) && sc.fstamp[f as usize] != stamp {
                    sc.fstamp[f as usize] = stamp;
                    sub.push(f);
                }
            }
        }
        let local = Local::new(t, &tree.points, &sub, |c| dsu.find(c));
        let n = local.n();
        let (mut uv, mut cnt) = (vec![0f64; 2 * n], vec![0f64; n]);
        let mut free = vec![false; n];
        for (k, &h) in local.corner.iter().enumerate() {
            let v = local.fl[k] as usize;
            let (x, y) = (self.corner_uv[2 * h as usize], self.corner_uv[2 * h as usize + 1]);
            let (x, y) = if self.owner[t.cface[h as usize] as usize] == b { apply_rigid(rigid, x, y) } else { (x, y) };
            uv[2 * v] += x;
            uv[2 * v + 1] += y;
            cnt[v] += 1.0;
            free[v] = sc.vstamp[t.flat[h as usize] as usize] == stamp;
        }
        for v in 0..n {
            uv[2 * v] /= cnt[v].max(1.0);
            uv[2 * v + 1] /= cnt[v].max(1.0);
        }
        let ok = uvflat::quick_stretch(&local.chart(), &mut uv, &free, 0.0, stop).is_ok();
        if !ok {
            uv.iter_mut().for_each(|x| *x = f64::NAN);
        }
        let per = tree.face_stretch(&local, &uv);
        let q = self.union_quantile(a, b, &per, &sc.fstamp, stamp);
        Eval { glued, cost, stretch: q, pole: true, local, uv, rigid: Some(rigid), face_str: per }
    }

    /// Would the union of charts a and b, laid out as `ev` says, lie over itself? True when more band faces are
    /// turned over than before, or two border sides cross. Unmoved sides of a alone (or b alone) are not tested
    /// against each other: each chart was tested when it was made.
    fn lies_over(&mut self, a: u32, b: u32, ev: &Eval) -> bool {
        let tree = self.tree;
        let t = &tree.topo;
        let local = &ev.local;
        for (k, &h) in local.corner.iter().enumerate() {
            self.cloc[h as usize] = local.fl[k] as u32 + 1;
        }
        let old = |owner: &[u32], uv: &[f64], h: usize| -> [f64; 2] {
            match ev.rigid {
                Some(r) if owner[t.cface[h] as usize] == b => {
                    let (x, y) = apply_rigid(r, uv[2 * h], uv[2 * h + 1]);
                    [x, y]
                }
                _ => [uv[2 * h], uv[2 * h + 1]],
            }
        };
        // Band faces turned over now and before (b placed rigidly), so a face that was already turned does not
        // refuse the merge.
        let (mut now, mut before) = (0usize, 0usize);
        for k in 0..local.faces.len() {
            let (lo, hi) = (local.ip[k] as usize, local.ip[k + 1] as usize);
            let (mut s_now, mut s_before) = (0.0, 0.0);
            for i in lo..hi {
                let j = if i + 1 == hi { lo } else { i + 1 };
                let (p, q) = (local.fl[i] as usize, local.fl[j] as usize);
                s_now += ev.uv[2 * p] * ev.uv[2 * q + 1] - ev.uv[2 * q] * ev.uv[2 * p + 1];
                let (u, w) = (old(&self.owner, &self.corner_uv, local.corner[i] as usize), old(&self.owner, &self.corner_uv, local.corner[j] as usize));
                s_before += u[0] * w[1] - w[0] * u[1];
            }
            now += usize::from(hi - lo >= 3 && !(s_now > 0.0));
            before += usize::from(hi - lo >= 3 && !(s_before > 0.0));
        }
        let mut sides: Vec<Side> = Vec::new();
        if now <= before {
            // Sides meeting at one UV point (one wedge once the union is glued) touch, they do not cross; two sides of
            // a cut share mesh vertices but not wedges, and are tested (a cone closing too far crosses there).
            let mark = self.dsu.mark();
            for &s in &ev.glued {
                glue(t, &mut self.dsu, &tree.segments[s as usize].edges);
                for &e in &tree.segments[s as usize].edges {
                    self.eglue[e as usize] = 1;
                }
            }
            for (c, cls) in [(a, 0u8), (b, 1u8)] {
                for &h in &self.bnd[c as usize] {
                    let h = h as usize;
                    if self.eglue[t.cedge[h] as usize] != 0 {
                        continue;
                    }
                    let n = t.next(h);
                    let at = |x: usize| {
                        let k = self.cloc[x];
                        if k > 0 { [ev.uv[2 * (k - 1) as usize], ev.uv[2 * (k - 1) as usize + 1]] } else { old(&self.owner, &self.corner_uv, x) }
                    };
                    let moved = self.cloc[h] > 0 || self.cloc[n] > 0;
                    sides.push(Side { p: at(h), q: at(n), v: [self.dsu.find(h as u32), self.dsu.find(n as u32)], class: if moved { 2 } else { cls } });
                }
            }
            self.dsu.undo(mark);
            for &s in &ev.glued {
                for &e in &tree.segments[s as usize].edges {
                    self.eglue[e as usize] = 0;
                }
            }
        }
        for &h in &local.corner {
            self.cloc[h as usize] = 0;
        }
        now > before || sides_cross(&mut sides)
    }

    /// Glue an evaluated union for good and take its layout; returns the surviving chart.
    fn apply(&mut self, a: u32, b: u32, ev: Eval) -> u32 {
        let tree = self.tree;
        let t = &tree.topo;
        let mut chi_new = self.chi[a as usize] + self.chi[b as usize];
        for &s in &ev.glued {
            let (g, m) = glue(t, &mut self.dsu, &tree.segments[s as usize].edges);
            chi_new += g - m;
            self.glued_seg[s as usize] = true;
        }
        self.dsu.commit();
        debug_assert_eq!(chi_new, 1);
        if let Some(rigid) = ev.rigid {
            for &f in &self.faces[b as usize] {
                for h in t.face_corners(f as usize) {
                    let (x, y) = apply_rigid(rigid, self.corner_uv[2 * h], self.corner_uv[2 * h + 1]);
                    self.corner_uv[2 * h] = x;
                    self.corner_uv[2 * h + 1] = y;
                }
            }
        }
        ev.local.scatter(&ev.uv, &mut self.corner_uv);
        let (r, o) = if self.faces[a as usize].len() >= self.faces[b as usize].len() { (a, b) } else { (b, a) };
        let small = std::mem::take(&mut self.vals[o as usize]);
        self.vals[r as usize].extend(small);
        for (k, &f) in ev.local.faces.iter().enumerate() {
            let old = self.face_str[f as usize];
            self.vals[r as usize].remove(&(bits(old), f));
            self.face_str[f as usize] = ev.face_str[k];
            self.vals[r as usize].insert((bits(ev.face_str[k]), f));
        }
        self.cpar[o as usize] = r;
        let moved = std::mem::take(&mut self.faces[o as usize]);
        for &f in &moved {
            self.owner[f as usize] = r;
        }
        self.faces[r as usize].extend(moved);
        let moved = std::mem::take(&mut self.segs[o as usize]);
        self.segs[r as usize].extend(moved);
        let gs = &self.glued_seg;
        self.segs[r as usize].retain(|&s| !gs[s as usize]);
        let moved = std::mem::take(&mut self.bnd[o as usize]);
        self.bnd[r as usize].extend(moved);
        for &s in &ev.glued {
            for &e in &tree.segments[s as usize].edges {
                self.eglue[e as usize] = 2; // glued for good
            }
        }
        let eg = &self.eglue;
        self.bnd[r as usize].retain(|&h| eg[t.cedge[h as usize] as usize] != 2);
        self.stretch[r as usize] = ev.stretch;
        self.grid[r as usize] = !ev.pole;
        self.chi[r as usize] = 1;
        self.ver[a as usize] += 1;
        self.ver[b as usize] += 1;
        r
    }
}

/// One border side of a union under test: its UV ends, the wedges of its ends, and whether it moved (2) or is an
/// unmoved side of chart a (0) or b (1).
struct Side {
    p: [f64; 2],
    q: [f64; 2],
    v: [u32; 2],
    class: u8,
}

fn side_box<'s>(it: impl Iterator<Item = &'s Side>) -> [f64; 4] {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for s in it {
        for p in [s.p, s.q] {
            b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
        }
    }
    b
}

/// Do two sides cross (each one's ends strictly on both sides of the other, beyond rounding)? Sides that share a
/// wedge, or are unmoved sides of the same chart, are not tested. Broad phase: only sides inside the box of the
/// sides they could meet, then a cell grid no finer than the longest side.
fn sides_cross(sides: &mut Vec<Side>) -> bool {
    let not0 = side_box(sides.iter().filter(|s| s.class != 0));
    let not1 = side_box(sides.iter().filter(|s| s.class != 1));
    let inside = |b: &[f64; 4], s: &Side| s.p[0].max(s.q[0]) >= b[0] && s.p[0].min(s.q[0]) <= b[2] && s.p[1].max(s.q[1]) >= b[1] && s.p[1].min(s.q[1]) <= b[3];
    sides.retain(|s| match s.class {
        0 => inside(&not0, s),
        1 => inside(&not1, s),
        _ => true,
    });
    if sides.len() < 2 {
        return false;
    }
    if !sides.iter().all(|s| s.p.iter().chain(s.q.iter()).all(|x| x.is_finite())) {
        return true; // a side without a position: the layout failed
    }
    let all = side_box(sides.iter());
    let longest = sides.iter().map(|s| (s.q[0] - s.p[0]).abs().max((s.q[1] - s.p[1]).abs())).fold(0.0, f64::max);
    let span = (all[2] - all[0]).max(all[3] - all[1]);
    // cells no smaller than the longest side, and about as many cells across as the square root of the side count
    let cell = longest.max(span / (sides.len() as f64).sqrt().max(1.0)).max(1e-300);
    let nx = ((all[2] - all[0]) / cell).floor() as i64 + 1;
    let mut entries: Vec<(i64, u32)> = Vec::with_capacity(4 * sides.len());
    for (i, s) in sides.iter().enumerate() {
        let x0 = ((s.p[0].min(s.q[0]) - all[0]) / cell).floor() as i64;
        let x1 = ((s.p[0].max(s.q[0]) - all[0]) / cell).floor() as i64;
        let y0 = ((s.p[1].min(s.q[1]) - all[1]) / cell).floor() as i64;
        let y1 = ((s.p[1].max(s.q[1]) - all[1]) / cell).floor() as i64;
        for y in y0..=y1 {
            for x in x0..=x1 {
                entries.push((y * nx + x, i as u32));
            }
        }
    }
    entries.sort_unstable();
    let mut lo = 0;
    while lo < entries.len() {
        let mut hi = lo + 1;
        while hi < entries.len() && entries[hi].0 == entries[lo].0 {
            hi += 1;
        }
        for i in lo..hi {
            for j in i + 1..hi {
                let (s, u) = (&sides[entries[i].1 as usize], &sides[entries[j].1 as usize]);
                if (s.class == u.class && s.class != 2) || s.v.iter().any(|v| u.v.contains(v)) {
                    continue;
                }
                if cross_proper(s, u) {
                    return true;
                }
            }
        }
        lo = hi;
    }
    false
}

fn cross_proper(s: &Side, u: &Side) -> bool {
    let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let ls = ((s.q[0] - s.p[0]).powi(2) + (s.q[1] - s.p[1]).powi(2)).sqrt();
    let lu = ((u.q[0] - u.p[0]).powi(2) + (u.q[1] - u.p[1]).powi(2)).sqrt();
    let tol = 1e-7 * ls * lu;
    let apart = |x: f64, y: f64| (x > tol && y < -tol) || (x < -tol && y > tol);
    apart(orient(s.p, s.q, u.p), orient(s.p, s.q, u.q)) && apart(orient(u.p, u.q, s.p), orient(u.p, u.q, s.q))
}

fn apply_rigid(r: ([f64; 2], [f64; 2]), x: f64, y: f64) -> (f64, f64) {
    let ([c, s], tr) = r;
    (c * x - s * y + tr[0], s * x + c * y + tr[1])
}

impl Tree {
    // -----------------------------------------------------------------------------------------------------------
    // Levels (6.4: Stretch allowed)
    // -----------------------------------------------------------------------------------------------------------

    /// The charts at Stretch allowed `x` percent: every recorded merge with stretch <= 1 + x/100 in recorded order,
    /// skipping (not stopping at) the others; a segment whose glue would leave a chart that is not a disk stays cut.
    /// Cached per x.
    pub fn level(&self, x_pct: f64) -> Arc<Level> {
        let key = x_pct.to_bits();
        if let Some(l) = self.cache.lock().unwrap().get(&key) {
            return l.clone();
        }
        let l = Arc::new(self.compute_level(x_pct));
        self.cache.lock().unwrap().insert(key, l.clone());
        l
    }

    pub fn compute_level(&self, x_pct: f64) -> Level {
        let started = Instant::now();
        let t = &self.topo;
        let thr = 1.0 + x_pct / 100.0;
        let mut dsu = Undo::new(t.corners());
        for e in 0..t.ne() {
            if !self.cut_base[e] {
                glue(t, &mut dsu, &[e as u32]);
            }
        }
        dsu.commit();
        let mut pd = Dsu::new(self.n_patches);
        let mut chi = self.patch_chi.clone();
        let mut glued = vec![false; t.ne()];
        let (mut used, mut segs_glued) = (0usize, 0usize);
        for m in &self.merges {
            if !(m.stretch <= thr) {
                continue;
            }
            used += 1;
            for &s in &m.segs {
                let sg = &self.segments[s as usize];
                let (rp, rq) = (pd.find(sg.p), pd.find(sg.q));
                let chi0 = if rp != rq { chi[rp as usize] + chi[rq as usize] } else { chi[rp as usize] };
                let mk = dsu.mark();
                let (g, mm) = glue(t, &mut dsu, &sg.edges);
                if chi0 + g - mm == 1 {
                    dsu.commit();
                    pd.union(rp, rq);
                    chi[pd.find(rp) as usize] = 1;
                    for &e in &sg.edges {
                        glued[e as usize] = true;
                    }
                    segs_glued += 1;
                } else {
                    dsu.undo(mk);
                }
            }
        }
        let cut: Vec<bool> = (0..t.ne()).map(|e| self.cut_base[e] && !glued[e]).collect();
        let mut id: HashMap<u32, u32, FxBuild> = HashMap::default();
        let chart: Vec<u32> = (0..t.nf).map(|f| {
            let r = pd.find(self.patch[f]);
            let k = id.len() as u32;
            *id.entry(r).or_insert(k)
        }).collect();
        let not_disk = id.keys().filter(|&&r| chi[r as usize] != 1).count();
        Level {
            chart,
            cut,
            charts: id.len(),
            not_disk,
            merges_used: used,
            segments_glued: segs_glued,
            above_build: x_pct / 100.0 > self.build_stretch,
            ms: started.elapsed().as_secs_f64() * 1e3,
        }
    }

    // -----------------------------------------------------------------------------------------------------------
    // Flatten a level (6.5) and Keep going (6.6)
    // -----------------------------------------------------------------------------------------------------------

    /// Flatten the charts of a cut state: exact grid start (and row labels) where a chart is a grid, LSCM otherwise,
    /// then quad ARAP with Straight rows (`uvflat::flatten_chart`). Returns (UVs per corner, stats rows of
    /// `uvflat::STATS_LEN` per chart, per-face stretch). `start` per corner (NaN = none) is used where it is
    /// flip-free and the chart is not a grid.
    pub fn flatten_cut(&self, cut: &[bool], chart: &[u32], only: Option<&[u32]>, start: Option<&[f64]>, opts: &uvflat::FlatOpts, stop: &AtomicBool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let t = &self.topo;
        let wr = wedges(t, cut);
        let nc = chart.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); nc];
        for f in 0..t.nf {
            members[chart[f] as usize].push(f as u32);
        }
        let ids: Vec<u32> = match only {
            Some(o) => o.to_vec(),
            None => (0..nc as u32).filter(|&c| !members[c as usize].is_empty()).collect(),
        };
        let one = |c: u32| {
            let local = Local::new(t, &self.points, &members[c as usize], |h| wr[h as usize]);
            let ch = local.chart();
            let n = local.n();
            let grid = uvflat::grid_layout(&ch).ok();
            let given = start.map(|s| local.gather(s)).filter(|g| g.iter().all(|x| x.is_finite()) && uvflat::count_flips(&ch.faces, g) == 0);
            let (st, labels, mode) = match (&grid, &given) {
                (Some(g), _) => (Some(g.uv.clone()), Some((g.col.clone(), g.row.clone())), uvflat::Start::Given),
                (None, Some(g)) => (Some(g.clone()), None, uvflat::Start::Given),
                _ => (None, None, uvflat::Start::Free),
            };
            let o = uvflat::FlatOpts { start: mode, ..*opts };
            let lab = labels.as_ref().map(|(a, b)| (&a[..], &b[..]));
            match uvflat::flatten_chart(&ch, st.as_deref(), lab, &o, stop) {
                Ok(mut out) => {
                    out.stats[uvflat::stat::CHART] = c as f64;
                    let per = self.face_stretch(&local, &out.uv);
                    (local, out.uv, out.stats, per)
                }
                Err(_) => {
                    let mut st = [f64::NAN; uvflat::STATS_LEN];
                    st[uvflat::stat::CHART] = c as f64;
                    st[uvflat::stat::STEPS] = -1.0;
                    let k = members[c as usize].len();
                    (local, vec![f64::NAN; 2 * n], st, vec![f64::INFINITY; k])
                }
            }
        };
        let results: Vec<_> = if t.nf >= PARALLEL_MIN && ids.len() > 1 { ids.par_iter().map(|&c| one(c)).collect() } else { ids.iter().map(|&c| one(c)).collect() };
        let mut uv = vec![f64::NAN; 2 * t.corners()];
        let mut face_str = vec![f64::NAN; t.nf];
        let mut stats = Vec::with_capacity(results.len() * uvflat::STATS_LEN);
        for (k, (local, cuv, st, per)) in results.into_iter().enumerate() {
            local.scatter(&cuv, &mut uv);
            for (i, &f) in members[ids[k] as usize].iter().enumerate() {
                face_str[f as usize] = per[i];
            }
            stats.extend_from_slice(&st);
        }
        (uv, stats, face_str)
    }

    /// Start Keep going from a level: its charts flattened.
    pub fn start_cuts(&self, x_pct: f64, opts: &uvflat::FlatOpts, stop: &AtomicBool) -> Cuts {
        let level = self.level(x_pct);
        let (uv, stats, face_str) = self.flatten_cut(&level.cut, &level.chart, None, Some(&self.start_uv), opts, stop);
        let t = &self.topo;
        let area: Vec<f64> = (0..t.nf).map(|f| face_area(&self.points, t, f)).collect();
        let mut cuts = Cuts { cut: level.cut.clone(), chart: level.chart.clone(), uv, face_str, chart_str: Vec::new(), chart_area: Vec::new(), area, opts: *opts, score: 0.0, passes: 0, moves: 0 };
        let _ = stats;
        cuts.chart_str = vec![f64::INFINITY; level.charts];
        let all: Vec<u32> = (0..level.charts as u32).collect();
        cuts.refresh_stretch(&all);
        cuts.refresh_area();
        cuts.score = self.score(&cuts);
        cuts
    }

    /// `stretch excess + seen cut px / sqrt(area px)` (6.6). Excess: area-weighted share above Stretch allowed.
    pub fn score(&self, cuts: &Cuts) -> f64 {
        let allowed = 1.0 + cuts.opts.stretch_allowed;
        let total: f64 = cuts.chart_area.iter().sum::<f64>().max(1e-300);
        let excess: f64 = cuts.chart_str.iter().zip(&cuts.chart_area).map(|(&s, &a)| if a > 0.0 { a * if s.is_finite() { (s - allowed).max(0.0) } else { 1e6 } } else { 0.0 }).sum::<f64>() / total;
        let seen_cut: f64 = (0..self.topo.ne()).filter(|&e| cuts.cut[e] && !self.topo.wall[e]).map(|e| self.seen[e]).sum();
        let ppu = self.numbers[num::PX_PER_UNIT];
        excess + seen_cut / (total * ppu * ppu).sqrt()
    }

    /// One Keep going pass (6.6): the better of "cut from the worst-stretch vertex to the chart border along patch
    /// borders" and "close the cut segment that keeps Stretch allowed and saves the most seen px", taken only when
    /// the score drops. Returns what happened; `accepted == false` means the pass gained nothing (Keep going stops).
    pub fn pass(&self, cuts: &mut Cuts, stop: &AtomicBool) -> Pass {
        let started = Instant::now();
        let before = cuts.score;
        let mut best: Option<(f64, u8, Cuts)> = None;
        if let Some(c) = self.cut_move(cuts, stop) {
            let s = self.score(&c);
            best = Some((s, 1, c));
        }
        if let Some(c) = self.close_move(cuts, stop) {
            let s = self.score(&c);
            if best.as_ref().is_none_or(|b| s < b.0) {
                best = Some((s, 2, c));
            }
        }
        cuts.passes += 1;
        let mut out = Pass { kind: 0, accepted: false, before, after: before, ms: 0.0 };
        if let Some((s, kind, mut c)) = best {
            if s < before && !stop.load(Ordering::Relaxed) {
                c.score = s;
                c.passes = cuts.passes;
                c.moves = cuts.moves + 1;
                *cuts = c;
                out.kind = kind;
                out.accepted = true;
                out.after = s;
            }
        }
        out.ms = started.elapsed().as_secs_f64() * 1e3;
        out
    }

    fn reflatten(&self, c: &mut Cuts, charts: &[u32], start: Option<&[f64]>, stop: &AtomicBool) {
        let (uv, stats, face_str) = self.flatten_cut(&c.cut, &c.chart, Some(charts), start, &c.opts, stop);
        for f in 0..self.topo.nf {
            if charts.contains(&c.chart[f]) {
                for h in self.topo.face_corners(f) {
                    c.uv[2 * h] = uv[2 * h];
                    c.uv[2 * h + 1] = uv[2 * h + 1];
                }
                c.face_str[f] = face_str[f];
            }
        }
        let _ = stats;
        c.refresh_stretch(charts);
    }

    fn cut_move(&self, cuts: &Cuts, stop: &AtomicBool) -> Option<Cuts> {
        let t = &self.topo;
        let allowed = 1.0 + cuts.opts.stretch_allowed;
        let worst = (0..cuts.chart_str.len()).filter(|&c| cuts.chart_area[c] > 0.0 && !(cuts.chart_str[c] <= allowed)).max_by(|&a, &b| cuts.chart_str[a].total_cmp(&cuts.chart_str[b]))? as u32;
        let in_chart = |e: usize| {
            let [f0, f1] = t.faces_of(e);
            f1 != NONE && cuts.chart[f0 as usize] == worst && cuts.chart[f1 as usize] == worst
        };
        let on_border = |v: usize| t.vedges(v).iter().any(|&e| {
            let e = e as usize;
            (cuts.cut[e] || t.wall[e]) && t.faces_of(e).iter().any(|&f| f != NONE && cuts.chart[f as usize] == worst)
        });
        let wf = (0..t.nf).filter(|&f| cuts.chart[f] == worst).max_by(|&a, &b| cuts.face_str[a].total_cmp(&cuts.face_str[b]))?;
        // Nearest interior vertex with an uncut patch border edge, from the worst face.
        let mut seenv = vec![false; t.nv];
        let mut queue: VecDeque<u32> = VecDeque::new();
        for h in t.face_corners(wf) {
            seenv[t.flat[h] as usize] = true;
            queue.push_back(t.flat[h]);
        }
        let mut start = None;
        while let Some(v) = queue.pop_front() {
            if !on_border(v as usize) && t.vedges(v as usize).iter().any(|&e| self.sep[e as usize] && !cuts.cut[e as usize] && in_chart(e as usize)) {
                start = Some(v);
                break;
            }
            for &e in t.vedges(v as usize) {
                if !cuts.cut[e as usize] && in_chart(e as usize) {
                    let w = t.other(e as usize, v);
                    if !seenv[w as usize] {
                        seenv[w as usize] = true;
                        queue.push_back(w);
                    }
                }
            }
        }
        let start = start?;
        // Dijkstra along uncut patch borders to the chart border.
        let mut dist: HashMap<u32, (f64, u32), FxBuild> = HashMap::default();
        let mut heap: BinaryHeap<(std::cmp::Reverse<u64>, u32)> = BinaryHeap::new();
        dist.insert(start, (0.0, NONE));
        heap.push((std::cmp::Reverse(0f64.to_bits()), start));
        let mut end = None;
        while let Some((std::cmp::Reverse(d), v)) = heap.pop() {
            let d = f64::from_bits(d);
            if d > dist[&v].0 {
                continue;
            }
            if v != start && on_border(v as usize) {
                end = Some(v);
                break;
            }
            for &e in t.vedges(v as usize) {
                let e2 = e as usize;
                if !self.sep[e2] || cuts.cut[e2] || !in_chart(e2) {
                    continue;
                }
                let w = t.other(e2, v);
                let nd = d + self.cost[e2].max(0.0);
                if dist.get(&w).is_none_or(|x| nd < x.0) {
                    dist.insert(w, (nd, e));
                    heap.push((std::cmp::Reverse(nd.to_bits()), w));
                }
            }
        }
        let mut v = end?;
        let mut c = cuts.clone();
        while v != start {
            let e = dist[&v].1 as usize;
            c.cut[e] = true;
            v = t.other(e, v);
        }
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let old = cuts.uv.clone();
        self.reflatten(&mut c, &[worst], Some(&old), stop);
        if !(c.chart_str[worst as usize] < cuts.chart_str[worst as usize]) {
            // The slit start did not help: try the merge tree's layout of the cut chart.
            let mut d = cuts.clone();
            d.cut = c.cut.clone();
            self.reflatten(&mut d, &[worst], Some(&self.start_uv), stop);
            if d.chart_str[worst as usize] < c.chart_str[worst as usize] {
                return Some(d);
            }
        }
        Some(c)
    }

    fn close_move(&self, cuts: &Cuts, stop: &AtomicBool) -> Option<Cuts> {
        let t = &self.topo;
        let allowed = 1.0 + cuts.opts.stretch_allowed;
        let mut cands: Vec<u32> = (0..self.segments.len() as u32).filter(|&s| self.segments[s as usize].edges.iter().all(|&e| cuts.cut[e as usize])).collect();
        cands.sort_by(|&a, &b| self.segments[b as usize].seen.total_cmp(&self.segments[a as usize].seen).then(a.cmp(&b)));
        let mut dsu = Undo::new(t.corners());
        for e in 0..t.ne() {
            if !cuts.cut[e] && !t.wall[e] {
                glue(t, &mut dsu, &[e as u32]);
            }
        }
        dsu.commit();
        for s in cands {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let sg = &self.segments[s as usize];
            let [f0, f1] = t.faces_of(sg.edges[0] as usize);
            let (c0, c1) = (cuts.chart[f0 as usize], cuts.chart[f1 as usize]);
            let chi0 = if c0 != c1 { 2 } else { 1 };
            let mk = dsu.mark();
            let (g, m) = glue(t, &mut dsu, &sg.edges);
            dsu.undo(mk);
            if chi0 + g - m != 1 {
                continue;
            }
            let mut c = cuts.clone();
            for &e in &sg.edges {
                c.cut[e as usize] = false;
            }
            if c0 != c1 {
                for f in 0..t.nf {
                    if c.chart[f] == c1 {
                        c.chart[f] = c0;
                    }
                }
                c.chart_str[c1 as usize] = f64::NAN;
                c.chart_area[c1 as usize] = 0.0;
            }
            self.reflatten(&mut c, &[c0], Some(&self.start_uv), stop);
            c.refresh_area();
            if c.chart_str[c0 as usize] <= allowed {
                return Some(c);
            }
        }
        None
    }
}

/// Glue segments one by one (in the given order), keeping each only when the charts stay one disk (Euler
/// characteristic 1); retried until nothing changes (a segment between two kept ones may fit later). `chi` is the
/// characteristic before (2 for two separate disks). Returns (glued segments, their cost).
fn glue_disk<D: Uf>(t: &Topo, dsu: &mut D, segments: &[Segment], list: &[u32], mut chi: i64) -> (Vec<u32>, f64) {
    let mut pending: Vec<u32> = list.to_vec();
    let mut glued = Vec::new();
    let mut cost = 0.0;
    loop {
        let mut left = Vec::new();
        let before = glued.len();
        for &s in &pending {
            let mk = dsu.mark();
            let (g, m) = glue(t, dsu, &segments[s as usize].edges);
            if chi + g - m == 1 {
                chi = 1;
                glued.push(s);
                cost += segments[s as usize].cost;
            } else {
                dsu.undo(mk);
                left.push(s);
            }
        }
        pending = left;
        if glued.len() == before || pending.is_empty() {
            break;
        }
    }
    (glued, cost)
}

/// Euler characteristic per patch: wedges - (corners - glued edges) + faces.
fn chi_per(t: &Topo, cut: &[bool], patch: &[u32], np: usize) -> Vec<i64> {
    let wr = wedges(t, cut);
    let mut chi = vec![0i64; np];
    let mut first = vec![false; t.corners()];
    for f in 0..t.nf {
        let p = patch[f] as usize;
        chi[p] += 1;
        for h in t.face_corners(f) {
            chi[p] -= 1;
            if !first[wr[h] as usize] {
                first[wr[h] as usize] = true;
                chi[p] += 1;
            }
        }
    }
    for e in 0..t.ne() {
        if !cut[e] && !t.wall[e] {
            chi[patch[t.cface[t.euse[e][0] as usize] as usize] as usize] += 1;
        }
    }
    chi
}

/// Vertices on the border of patch `p` (a side of one of its faces is cut), grouped into loops.
fn patch_loops(t: &Topo, cut: &[bool], members: &[u32]) -> HashMap<u32, u32, FxBuild> {
    let mut id: HashMap<u32, u32, FxBuild> = HashMap::default();
    let mut pairs = Vec::new();
    for &f in members {
        for h in t.face_corners(f as usize) {
            if cut[t.cedge[h] as usize] {
                let (a, b) = (t.flat[h], t.flat[t.next(h)]);
                for v in [a, b] {
                    let k = id.len() as u32;
                    id.entry(v).or_insert(k);
                }
                pairs.push((id[&a], id[&b]));
            }
        }
    }
    let mut d = Dsu::new(id.len());
    for (a, b) in pairs {
        d.union(a, b);
    }
    id.iter().map(|(&v, &k)| (v, d.find(k))).collect()
}

/// An annulus patch: the cheapest straight row from one border loop to the other (keeps the patch a grid).
fn annulus_row(t: &Topo, cut: &[bool], patch: &[u32], p: u32, members: &[u32], cost: &[f64]) -> Option<Vec<u32>> {
    let loops = patch_loops(t, cut, members);
    let mut ids: Vec<u32> = loops.values().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != 2 {
        return None;
    }
    let inside = |e: usize| {
        let [f0, f1] = t.faces_of(e);
        !cut[e] && f1 != NONE && patch[f0 as usize] == p && patch[f1 as usize] == p
    };
    let mut best: Option<(f64, Vec<u32>)> = None;
    let mut starts: Vec<u32> = loops.iter().filter(|(_, &l)| l == ids[0]).map(|(&v, _)| v).collect();
    starts.sort_unstable();
    for v in starts {
        let inward: Vec<u32> = t.vedges(v as usize).iter().copied().filter(|&e| inside(e as usize)).collect();
        if inward.len() != 1 {
            continue;
        }
        let (mut cur, mut e) = (v, inward[0]);
        let (mut path, mut c) = (Vec::new(), 0.0);
        loop {
            path.push(e);
            c += cost[e as usize];
            let w = t.other(e as usize, cur);
            if let Some(&l) = loops.get(&w) {
                if l == ids[1] && best.as_ref().is_none_or(|b| c < b.0) {
                    best = Some((c, path.clone()));
                }
                break;
            }
            match t.straight_next(w, e) {
                Some(x) if inside(x as usize) && path.len() <= members.len() => {
                    cur = w;
                    e = x;
                }
                _ => break,
            }
        }
    }
    best.map(|b| b.1)
}

/// Any other non-disk patch: tree-cotree. Keep a maximum-cost dual spanning tree glued, cut every other inside edge,
/// then glue back dangling branches (an end with one cut edge and no border).
fn tree_cotree(t: &Topo, cut: &[bool], patch: &[u32], p: u32, members: &[u32], cost: &[f64]) -> Vec<u32> {
    let mut inner: Vec<u32> = Vec::new();
    for &f in members {
        for h in t.face_corners(f as usize) {
            let e = t.cedge[h] as usize;
            let [f0, f1] = t.faces_of(e);
            if !cut[e] && !t.wall[e] && f0 == f && patch[f1 as usize] == p {
                inner.push(e as u32);
            }
            let _ = f1;
        }
    }
    inner.sort_unstable();
    inner.dedup();
    inner.sort_by(|&a, &b| cost[b as usize].total_cmp(&cost[a as usize]).then(a.cmp(&b)));
    let local: HashMap<u32, u32, FxBuild> = members.iter().enumerate().map(|(k, &f)| (f, k as u32)).collect();
    let mut d = Dsu::new(members.len());
    let mut cutset: Vec<u32> = Vec::new();
    for &e in &inner {
        let [f0, f1] = t.faces_of(e as usize);
        let (a, b) = (local[&f0], local[&f1]);
        if d.find(a) == d.find(b) {
            cutset.push(e);
        } else {
            d.union(a, b);
        }
    }
    let mut on: HashMap<u32, bool, FxBuild> = cutset.iter().map(|&e| (e, true)).collect();
    let mut deg: HashMap<u32, u32, FxBuild> = HashMap::default();
    for &f in members {
        for h in t.face_corners(f as usize) {
            if cut[t.cedge[h] as usize] {
                *deg.entry(t.flat[h]).or_insert(0) += 2; // on the border: never a dangling end
            }
        }
    }
    for &e in &cutset {
        *deg.entry(t.ea[e as usize]).or_insert(0) += 1;
        *deg.entry(t.eb[e as usize]).or_insert(0) += 1;
    }
    let mut queue: Vec<u32> = deg.iter().filter(|(_, &k)| k == 1).map(|(&v, _)| v).collect();
    while let Some(v) = queue.pop() {
        if deg[&v] != 1 {
            continue;
        }
        let e = t.vedges(v as usize).iter().copied().find(|e| on.get(e) == Some(&true));
        let Some(e) = e else { continue };
        on.insert(e, false);
        for w in [t.ea[e as usize], t.eb[e as usize]] {
            let k = deg.get_mut(&w).unwrap();
            *k -= 1;
            if *k == 1 {
                queue.push(w);
            }
        }
    }
    let mut out: Vec<u32> = on.into_iter().filter(|(_, x)| *x).map(|(e, _)| e).collect();
    out.sort_unstable();
    out
}

/// Keep going state: the cut, the charts and their flattened UVs.
#[derive(Clone)]
pub struct Cuts {
    pub cut: Vec<bool>,
    pub chart: Vec<u32>,
    pub uv: Vec<f64>,
    pub face_str: Vec<f64>,
    pub chart_str: Vec<f64>,
    pub chart_area: Vec<f64>,
    area: Vec<f64>,
    pub opts: uvflat::FlatOpts,
    pub score: f64,
    pub passes: usize,
    pub moves: usize,
}

impl Cuts {
    /// Chart stretch (worst 1 % of the faces) of the given charts.
    fn refresh_stretch(&mut self, charts: &[u32]) {
        let mut per: HashMap<u32, Vec<f64>, FxBuild> = charts.iter().map(|&c| (c, Vec::new())).collect();
        for (f, &c) in self.chart.iter().enumerate() {
            if let Some(v) = per.get_mut(&c) {
                v.push(self.face_str[f]);
            }
        }
        for (c, v) in per {
            if !v.is_empty() {
                self.chart_str[c as usize] = quantile(&v, uvflat::STRETCH_QUANTILE);
            }
        }
    }

    fn refresh_area(&mut self) {
        self.chart_area = vec![0.0; self.chart_str.len()];
        for (f, &c) in self.chart.iter().enumerate() {
            self.chart_area[c as usize] += self.area[f];
        }
    }
}

pub struct Pass {
    pub kind: u8, // 0 nothing, 1 cut added, 2 cut closed
    pub accepted: bool,
    pub before: f64,
    pub after: f64,
    pub ms: f64,
}

// ---------------------------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------------------------

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::uvflat::tests::grid;

    fn never() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn input<'a>(pts: &'a [f64], ip: &'a [i64], fl: &'a [i64]) -> Input<'a> {
        Input { points: pts, faces: Faces { indptr: ip, flat: fl }, part_label: &[], forced: &[], seen_edge_px: None, px_per_unit: 0.0, out_of_sight: false, hidden_l: 0.0, build_stretch: 0.0 }
    }

    /// A cube of side 1 with `k` x `k` quads per side, outward faces.
    pub fn cube(k: usize) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut id: HashMap<[i64; 3], i64> = HashMap::new();
        let (mut pts, mut ip, mut fl) = (Vec::new(), vec![0i64], Vec::new());
        let k = k as i64;
        // (normal axis, sign): u, v axes with u x v = normal.
        for (ax, sg) in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)] {
            let (mut u, mut v) = ((ax + 1) % 3, (ax + 2) % 3);
            if sg < 0 {
                std::mem::swap(&mut u, &mut v);
            }
            let mut vid = |i: i64, j: i64| {
                let mut c = [0i64; 3];
                c[ax] = if sg > 0 { k } else { 0 };
                c[u] = i;
                c[v] = j;
                *id.entry(c).or_insert_with(|| {
                    pts.extend(c.iter().map(|&x| x as f64 / k as f64));
                    (pts.len() / 3 - 1) as i64
                })
            };
            for j in 0..k {
                for i in 0..k {
                    let q = [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)];
                    fl.extend_from_slice(&q);
                    ip.push(fl.len() as i64);
                }
            }
        }
        (pts, ip, fl)
    }

    /// An open tube: `around` quads around, `rows` along, radius r.
    pub fn tube(around: usize, rows: usize, r: f64) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut pts = Vec::new();
        for j in 0..=rows {
            for i in 0..around {
                let a = std::f64::consts::TAU * i as f64 / around as f64;
                pts.extend_from_slice(&[r * a.cos(), r * a.sin(), j as f64 * 0.3]);
            }
        }
        let (mut ip, mut fl) = (vec![0i64], Vec::new());
        let id = |i: usize, j: usize| (j * around + i % around) as i64;
        for j in 0..rows {
            for i in 0..around {
                fl.extend_from_slice(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
                ip.push(fl.len() as i64);
            }
        }
        (pts, ip, fl)
    }

    #[test]
    fn border_sides_that_cross_are_found_and_touching_ones_are_not() {
        let side = |p: [f64; 2], q: [f64; 2], v: [u32; 2]| Side { p, q, v, class: 2 };
        // A border folded over itself: the side 2-3 crosses the side 0-1.
        let mut crossing = vec![side([0.0, 0.0], [2.0, 0.0], [0, 1]), side([1.0, -1.0], [1.0, 1.0], [2, 3])];
        assert!(sides_cross(&mut crossing));
        // Sides meeting at a shared wedge touch, they do not cross.
        let mut touching = vec![side([0.0, 0.0], [2.0, 0.0], [0, 1]), side([2.0, 0.0], [1.0, 1.0], [1, 3])];
        assert!(!sides_cross(&mut touching));
        // Unmoved sides of one chart are not tested against each other (each chart was tested when it was made).
        let mut same = vec![Side { class: 0, ..side([0.0, 0.0], [2.0, 0.0], [0, 1]) }, Side { class: 0, ..side([1.0, -1.0], [1.0, 1.0], [2, 3]) }];
        assert!(!sides_cross(&mut same));
    }

    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }

    #[test]
    fn cube_gives_six_patches_and_disks_at_every_level() {
        let (pts, ip, fl) = cube(4);
        let tree = Tree::build(&input(&pts, &ip, &fl), &never()).unwrap();
        assert_eq!(tree.n_patches, 6);
        assert_eq!(tree.numbers[num::IRREGULAR], 8.0);
        assert_eq!(tree.numbers[num::NOT_DISK_PATCHES], 0.0);
        assert_eq!(tree.segments.len(), 12);
        for x in [0.0, 10.0, 50.0, 100.0] {
            let l = tree.level(x);
            assert_eq!(l.not_disk, 0, "level {x}");
        }
        // A closed cube can never become one disk chart without a cut: the last glue is refused.
        assert!(tree.level(100.0).charts >= 1);
    }

    #[test]
    fn cylinder_gets_exactly_one_cut() {
        let (pts, ip, fl) = tube(16, 5, 1.0);
        let tree = Tree::build(&input(&pts, &ip, &fl), &never()).unwrap();
        assert_eq!(tree.n_patches, 1);
        assert_eq!(tree.numbers[num::ANNULI], 1.0);
        assert_eq!(tree.numbers[num::EULER_CUT_EDGES], 5.0, "one straight row across 5 rows");
        assert_eq!(tree.patch_chi, vec![1]);
        let l = tree.level(10.0);
        assert_eq!((l.charts, l.not_disk), (1, 0));
    }

    #[test]
    fn forced_line_turning_vertex_starts_traces() {
        let (pts, ip, fl) = grid(8, 8, 1.0, |_, _| 0.0);
        let id = |i: i64, j: i64| j * 9 + i;
        // An L: from (2, 2) right to (5, 2), then up to (5, 6).
        let mut forced = Vec::new();
        for i in 2..5 {
            forced.extend_from_slice(&[id(i, 2), id(i + 1, 2)]);
        }
        for j in 2..6 {
            forced.extend_from_slice(&[id(5, j), id(5, j + 1)]);
        }
        let mut inp = input(&pts, &ip, &fl);
        inp.forced = &forced;
        let tree = Tree::build(&inp, &never()).unwrap();
        assert!(tree.turning[id(5, 2) as usize], "the corner of the L is a turning vertex");
        assert!(tree.irregular[id(2, 2) as usize] && tree.irregular[id(5, 6) as usize], "line ends start traces");
        // Traces out of the corner continue both legs straight to the border.
        let t = &tree.topo;
        let traced_from = |a: i64, b: i64| (0..t.ne()).any(|e| (t.ea[e] as i64, t.eb[e] as i64) == (a.min(b), a.max(b)) && tree.sep[e]);
        assert!(traced_from(id(5, 2), id(6, 2)) && traced_from(id(5, 2), id(5, 1)));
        for l in [tree.level(0.0), tree.level(10.0)] {
            assert_eq!(l.not_disk, 0);
        }
        // Every patch is a grid rectangle.
        assert_eq!(tree.numbers[num::GRID_PATCHES], tree.n_patches as f64);
    }

    fn arap_vs_grid(pts: &[f64], ip: &[i64], fl: &[i64]) -> (f64, f64) {
        let ch = uvflat::Chart { points: pts, faces: Faces { indptr: ip, flat: fl } };
        let g = uvflat::grid_layout(&ch).unwrap();
        let sg = uvflat::stretch(&ch, &g.uv).0;
        let opts = uvflat::FlatOpts { rows: uvflat::Rows::Off, ..Default::default() };
        let out = uvflat::flatten_chart(&ch, Some(&g.uv), None, &opts, &never()).unwrap();
        (sg, uvflat::stretch(&ch, &out.uv).0)
    }

    #[test]
    fn grid_stretch_at_least_arap_and_equal_on_flat_and_cylinder() {
        let (pts, ip, fl) = grid(8, 5, 0.5, |_, _| 0.0);
        let (g, a) = arap_vs_grid(&pts, &ip, &fl);
        assert!(g >= a - 1e-9 && (g - a).abs() <= 0.01 * a, "flat {g} {a}");
        // The tube's patch after its Euler cut, as the merge tree lays it out.
        let (pts, ip, fl) = tube(16, 5, 1.0);
        let tree = Tree::build(&input(&pts, &ip, &fl), &never()).unwrap();
        let l = tree.level(0.0);
        let wr = wedges(&tree.topo, &l.cut);
        let all: Vec<u32> = (0..tree.topo.nf as u32).collect();
        let local = Local::new(&tree.topo, &tree.points, &all, |h| wr[h as usize]);
        let (g, a) = arap_vs_grid(&local.pts, &local.ip, &local.fl);
        assert!(g >= a - 1e-9 && (g - a).abs() <= 0.01 * a, "cylinder {g} {a}");
        // Curved (not developable): the grid is never better than ARAP.
        let (pts, ip, fl) = grid(8, 8, 0.25, |x, y| 0.3 * (x * x - y * y));
        let (g, a) = arap_vs_grid(&pts, &ip, &fl);
        assert!(g >= a - 1e-9, "curved {g} {a}");
    }

    #[test]
    fn level_skips_over_threshold_merges_instead_of_stopping() {
        // Three patches in a row (forced columns), the two columns then made gluable segments by hand, and two
        // hand-made merges with the expensive one recorded first.
        let (pts, ip, fl) = grid(6, 2, 1.0, |_, _| 0.0);
        let column = |c: i64| -> Vec<i64> { (0..2).flat_map(|j| [j * 7 + c, (j + 1) * 7 + c]).collect() };
        let forced: Vec<i64> = [2, 4].iter().flat_map(|&c| column(c)).collect();
        let mut inp = input(&pts, &ip, &fl);
        inp.forced = &forced;
        let mut tree = Tree::build(&inp, &never()).unwrap();
        assert_eq!(tree.n_patches, 3);
        let t = &tree.topo;
        let mut segs = Vec::new();
        for c in [2u32, 4] {
            let edges: Vec<u32> = (0..t.ne() as u32).filter(|&e| t.ea[e as usize] % 7 == c && t.eb[e as usize] == t.ea[e as usize] + 7).collect();
            let [f0, f1] = t.faces_of(edges[0] as usize);
            let (p, q) = (tree.patch[f0 as usize], tree.patch[f1 as usize]);
            segs.push(Segment { edges, p: p.min(q), q: p.max(q), cost: 1.0, seen: 1.0 });
        }
        tree.segments = segs;
        tree.merges = vec![Merge { segs: vec![0], stretch: 1.5, cost: 1.0, pole: false }, Merge { segs: vec![1], stretch: 1.01, cost: 1.0, pole: false }];
        let l = tree.compute_level(10.0);
        assert_eq!((l.merges_used, l.segments_glued, l.charts), (1, 1, 2), "the cheap later merge is applied although an earlier one is skipped");
        assert_eq!(tree.compute_level(60.0).charts, 1);
    }

    #[test]
    fn curved_patches_merge_lazily_and_keep_going_only_drops_the_score() {
        // A bumpy sheet with poles: a grid with one extra diagonal split (two triangles) in the middle.
        let (pts, ip, mut fl) = grid(10, 10, 0.2, |x, y| 0.4 * (2.0 * x).sin() * (1.5 * y).cos());
        let mut ip = ip;
        // Replace quad 55 by two triangles (its four vertices become irregular).
        let q: Vec<i64> = fl[4 * 55..4 * 56].to_vec();
        let mut nfl = Vec::new();
        let mut nip = vec![0i64];
        for f in 0..ip.len() - 1 {
            if f == 55 {
                nfl.extend_from_slice(&[q[0], q[1], q[2]]);
                nip.push(nfl.len() as i64);
                nfl.extend_from_slice(&[q[0], q[2], q[3]]);
            } else {
                nfl.extend_from_slice(&fl[ip[f] as usize..ip[f + 1] as usize]);
            }
            nip.push(nfl.len() as i64);
        }
        fl = nfl;
        ip = nip;
        let tree = Tree::build(&input(&pts, &ip, &fl), &never()).unwrap();
        assert!(tree.n_patches > 4);
        assert!(tree.numbers[num::MERGES] > 0.0);
        assert!(tree.numbers[num::QUICK_CALLS] <= tree.numbers[num::EXACT_EVALS]);
        for x in [0.0, 5.0, 10.0, 40.0] {
            assert_eq!(tree.level(x).not_disk, 0);
        }
        let opts = uvflat::FlatOpts { stretch_allowed: 0.02, ..Default::default() };
        let mut cuts = tree.start_cuts(5.0, &opts, &never());
        let (uv, _, _) = tree.flatten_cut(&cuts.cut, &cuts.chart, None, None, &opts, &never());
        assert!(uv.iter().all(|x| x.is_finite()));
        for _ in 0..6 {
            let snapshot = cuts.cut.clone();
            let p = tree.pass(&mut cuts, &never());
            if p.accepted {
                assert!(p.after < p.before);
            } else {
                assert_eq!(cuts.cut, snapshot, "a refused pass leaves the cut as it was");
                break;
            }
        }
    }
}
