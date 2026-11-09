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
//! - Compact charts (R4): the merge queue ranks a union by its irregularity, outline^2 / (4 pi area) (a disk is 1),
//!   with the outline weighted by what a cut there costs: its length, less on a fold (shading changes there anyway),
//!   more where the camera sees it. That is the shape term of hierarchical face clustering (Garland, Willmott,
//!   Heckbert 2001) and the compactness/boundary terms of chart growing (D-Charts: Julius, Kraevoy, Sheffer 2005;
//!   variational shape approximation: Cohen-Steiner, Alliez, Desbrun 2004), used here as a merge order. A union that
//!   is more irregular than `SHAPE_MAX` (plain lengths) and than both its parts is refused: branching, fractal charts
//!   waste the page in gap rings and box corners, and their long borders are seen (our rule).
//! - Thin tubes (R5): a part that wraps round and is a long strip of surface (springs, chain links, rope, wire) is cut
//!   across into pieces and each piece once along its length on its least-seen side (`tube_cuts`), started from Tutte's
//!   embedding on a convex outline of the strip's shape (Tutte 1963; Floater 1997, "Parametrization and smooth
//!   approximation of surface triangulations") and laid out by quad ARAP; a piece is judged by the UV Check's own
//!   stretch (area-weighted L2, Sander, Snyder, Gortler, Hoppe 2001). A union refused for its shape is still made when it
//!   is so small that its gap ring outweighs its box and the union's box with its ring takes less page than its parts'
//!   (smallest-area box by rotating calipers, Toussaint 1983; convex hull by Andrew's monotone chain, 1979). Ours.
//! - Fewer, smoother charts (R6): a last merge round in which every chart, smallest first, joins a neighbour glued
//!   along all it shares or hinged on its costliest shared segment (small compact charts free of the shape rule), and
//!   an outline smoothing of the full tree (`Tree::smooth`): a patch moves to the chart it lies against when that lowers
//!   the cut's outline weight, both charts stay disks, no cone closes and its fitted start crosses no border sides
//!   (boundary smoothing of chart growing: Sander, Wood, Gortler, Snyder, Hoppe 2003, "Multi-chart geometry images";
//!   Julius, Kraevoy, Sheffer 2005, "D-Charts"; our move, gain and guards). Every level replays the moves, so lower
//!   levels' charts stay inside higher ones (the redo of a chart that lies over itself relies on it).
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
/// Outline weight of a cut (R4): `length_px * (1 - FOLD_DISCOUNT * (1 - cos fold)) + SEEN_WEIGHT * seen_px`, the
/// fold angle between the edge's two faces (90 deg and more: the most discount), seen pixels only with seams out of
/// sight. Measured on 13 models (bench/uvplan/uv_defaults.md, R4chart).
pub const FOLD_DISCOUNT: f64 = 0.9;
pub const SEEN_WEIGHT: f64 = 4.0;
/// A merge whose union's irregularity (plain lengths, outline^2 / (4 pi area)) is above this and above both parts' is
/// refused (a square is 1.27, a 10:1 strip 3.85).
pub const SHAPE_MAX: f64 = 4.0;
/// Gap ring around a chart's box on the page, in the tree's pixels (1024 across the diagonal, about the page at
/// 1024: 8 texels at 1024, clean down to 256).
pub const GAP_PX: f64 = 8.0;
/// A part whose surface is a strip at least this many times as long as it is wide (length^2 / area) is tried as one
/// chart cut once along its length (`tube_cuts`).
pub const TUBE_ASPECT: f64 = 3.0;
/// Tube pieces are about this many times as long as the tube is round (`tube_cuts`).
pub const PIECE_ASPECT: f64 = 6.0;
/// A tube piece is kept when its stretch by the UV Check's own measure (area-weighted L2, SPEC 9.1) has a median within
/// the New UVs gate (SPEC 12.3, 1.08), flip-free and its border not crossing itself. The merge tree's per-face measure
/// (worst test triangle over both diagonals) reads above 2 on quads that wrap round a thin tube whatever the layout, so
/// it cannot judge these pieces. Faces above `TUBE_L2_BAD` share one budget, `TUBE_BAD_SHARE` of the mesh's area, the
/// cleanest pieces first: a fifth of the 1 % the worst 1 % gate (1.6) looks at, counted from well below 1.6 (measured
/// on 13 models, bench/uvplan/uv_defaults.md: 0.5 % counted from 1.6 let the worst 1 % reach 1.96 after the full layout).
pub const TUBE_L2_MEDIAN: f64 = 1.08;
pub const TUBE_L2_BAD: f64 = 1.3;
pub const TUBE_BAD_SHARE: f64 = 0.002;

/// In the last merge round (R6) a chart of at most this many faces joins a neighbour whatever the union's shape (its
/// gap ring outweighs its box, so joining always saves page); larger charts keep the shape rule (`SHAPE_MAX`).
/// Measured on 13 models (bench/uvplan/uv_defaults.md, R6chart).
pub const FREE_FACES: usize = 64;

/// Smoother outlines (R6, `Tree::smooth`): a patch moves to the neighbouring chart only when every vertex the move
/// closes inside that chart has an angle defect (360 deg less its corner angles) within this many degrees.
pub const SMOOTH_DEFECT_DEG: f64 = 10.0;
/// ... and only when every face of the moved patch, fitted onto the chart's start layout, keeps its area within this
/// factor of its surface area (the start layouts are isometric up to the chart's stretch).
pub const SMOOTH_AREA: f64 = 1.5;

/// Numbers in `Tree::numbers` (see `num`).
pub const NUMBERS_LEN: usize = 22;

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
    pub const TUBE_FACES: usize = 21; // faces laid out as tube strips (`tube_cuts`)
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
    /// `new` with the local numbering kept in the scratch arrays (the measure's hot path; same result).
    fn new_in(t: &Topo, points: &[f64], faces: &[u32], root: impl Fn(u32) -> u32, sc: &mut Scratch) -> Local {
        let stamp = sc.stamp; // the band's stamp: corners are stamped once per measure
        let (mut pts, mut ip, mut fl, mut corner) = (Vec::new(), vec![0i64], Vec::new(), Vec::new());
        for &f in faces {
            for h in t.face_corners(f as usize) {
                let r = root(h as u32) as usize;
                if sc.cstamp[r] != stamp {
                    sc.cstamp[r] = stamp;
                    let v = t.flat[h] as usize;
                    pts.extend_from_slice(&points[3 * v..3 * v + 3]);
                    sc.cid[r] = (pts.len() / 3 - 1) as u32;
                }
                fl.push(i64::from(sc.cid[r]));
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
    /// Outline weight (`outline_w`) and length (px) of the segment: what gluing it takes off the charts' outlines.
    pub weight: f64,
    pub length: f64,
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
    /// Start layout per corner when outline smoothing moved patches (`Tree::smooth`): each moved patch fitted rigidly
    /// onto its new chart along the glue. Empty: the tree's `start_uv`.
    pub start: Vec<f64>,
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
    /// Outline weight per edge (`outline_w`) and length (px), for the charts' irregularity in the merge tree.
    pub outline: Vec<f64>,
    pub length_px: Vec<f64>,
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
    smooth_cache: Mutex<Option<Arc<Smooth>>>,
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

/// Rank of a merge (R4): the union's irregularity by outline weight (most compact first), then cost removed; a union
/// that cannot be measured comes last. Stretch decides through the rounds' caps, not the order inside a round.
fn key_of(added: f64, cost: f64, irregular: f64) -> Key {
    if !added.is_finite() {
        Key { class: 2, a: 0.0, b: -cost }
    } else {
        Key { class: 0, a: irregular, b: -cost }
    }
}

/// Outline weight of one edge (px): see `FOLD_DISCOUNT`.
fn outline_w(length_px: f64, seen_px: f64, cos: f64, out_of_sight: bool) -> f64 {
    length_px * (1.0 - FOLD_DISCOUNT * (1.0 - cos.max(0.0))) + if out_of_sight { SEEN_WEIGHT * seen_px } else { 0.0 }
}

/// Irregularity of a chart: outline^2 / (4 pi area); a disk is 1.
fn irregularity(outline: f64, area: f64) -> f64 {
    if area > 0.0 { outline * outline / (4.0 * std::f64::consts::PI * area) } else { f64::INFINITY }
}

/// Per edge: cosine of the angle between its two faces' normals (Newell normals; 1 on a border edge).
fn fold_cos(points: &[f64], t: &Topo) -> Vec<f64> {
    let normal = |f: usize| {
        let mut n = [0f64; 3];
        for h in t.face_corners(f) {
            let (a, b) = (t.flat[h] as usize, t.flat[t.next(h)] as usize);
            let (p, q) = (&points[3 * a..3 * a + 3], &points[3 * b..3 * b + 3]);
            n[0] += (p[1] - q[1]) * (p[2] + q[2]);
            n[1] += (p[2] - q[2]) * (p[0] + q[0]);
            n[2] += (p[0] - q[0]) * (p[1] + q[1]);
        }
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l > 0.0 { [n[0] / l, n[1] / l, n[2] / l] } else { [0.0; 3] }
    };
    let fnorm: Vec<[f64; 3]> = (0..t.nf).map(normal).collect();
    (0..t.ne())
        .map(|e| {
            let [h0, h1] = t.euse[e];
            if h1 == NONE {
                return 1.0;
            }
            let (a, b) = (fnorm[t.cface[h0 as usize] as usize], fnorm[t.cface[h1 as usize] as usize]);
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        })
        .collect()
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
        let fold = fold_cos(points, &t);
        let outline: Vec<f64> = (0..ne).map(|e| outline_w(length_px[e], seen[e], fold[e], input.out_of_sight)).collect();
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
        // Thin tubes (R5): a part that is a long strip of surface becomes one patch, cut once along its length.
        let tubes = tube_cuts(&t, points, &cost, &forced_line, stop);
        for Tube { faces, cut, .. } in &tubes {
            for &f in faces {
                for h in t.face_corners(f as usize) {
                    sep[t.cedge[h] as usize] = false;
                }
            }
            for &e in cut {
                sep[e as usize] = true;
            }
        }
        numbers[num::TUBE_FACES] = tubes.iter().map(|x| x.faces.len()).sum::<usize>() as f64;
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
            let w: f64 = list.iter().map(|&x| outline[x as usize]).sum();
            let len: f64 = list.iter().map(|&x| length_px[x as usize]).sum();
            segments.push(Segment { edges: list, p: pr.0, q: pr.1, cost: c, seen: s, weight: w, length: len });
        }
        numbers[num::SEGMENTS] = segments.len() as f64;
        let mut tree = Tree {
            points: points.to_vec(),
            topo: t,
            seen,
            cost,
            outline,
            length_px,
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
            smooth_cache: Mutex::new(None),
        };
        let tube_uv: HashMap<u32, [f64; 2], FxBuild> = tubes.into_iter().flat_map(|x| x.corner_uv).collect();
        let (merges, found, start_uv) = tree.merge_tree(&members, &tube_uv, stop);
        tree.merges = merges;
        tree.start_uv = start_uv;
        for (k, x) in found {
            tree.numbers[k] = x;
        }
        tree.numbers[num::BUILD_MS] = started.elapsed().as_secs_f64() * 1e3;
        tree.numbers[num::STOPPED] = f64::from(u8::from(stop.load(Ordering::Relaxed)));
        Ok(tree)
    }

    fn patch_layout(&self, local: &Local, stop: &AtomicBool) -> (bool, Vec<f64>) {
        patch_layout(local, stop)
    }

    /// Per-face stretch of a laid-out chart (`uvflat::stretch`; INF for every face when a UV is missing).
    fn face_stretch(&self, local: &Local, uv: &[f64]) -> Vec<f64> {
        face_stretch(local, uv)
    }
}

/// Lay one patch out for the merge tree: its exact grid when that is within `GRID_SLACK` of 1 or of ARAP, else ARAP
/// (from the grid, or from LSCM when it is no grid). A quad patch that is no rectangle (a trapezoid, a rhombus) is
/// therefore no grid chart.
fn patch_layout(local: &Local, stop: &AtomicBool) -> (bool, Vec<f64>) {
        let ch = local.chart();
        let grid = uvflat::grid_layout(&ch).ok();
        let q_grid = grid.as_ref().map(|g| quantile(&face_stretch(local, &g.uv), uvflat::STRETCH_QUANTILE));
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
                let q_arap = quantile(&face_stretch(local, &a), uvflat::STRETCH_QUANTILE);
                if q_grid.unwrap() <= q_arap * (1.0 + GRID_SLACK) { (true, g.uv) } else { (false, a) }
            }
            (Some(g), None) => (true, g.uv),
            (None, Some(a)) => (false, a),
            (None, None) => (false, vec![f64::NAN; 2 * local.n()]),
        }
}

/// Per-face stretch of a laid-out chart (`uvflat::stretch`; INF for every face when a UV is missing).
fn face_stretch(local: &Local, uv: &[f64]) -> Vec<f64> {
    if !uv.iter().all(|x| x.is_finite()) {
        return vec![f64::INFINITY; local.faces.len()];
    }
    uvflat::stretch(&local.chart(), uv).1
}

impl Tree {

    /// The merge tree (6.4): the merges in merge order and the numbers it measured.
    fn merge_tree(&self, members: &[Vec<u32>], tube_uv: &HashMap<u32, [f64; 2], FxBuild>, stop: &AtomicBool) -> (Vec<Merge>, Vec<(usize, f64)>, Vec<f64>) {
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
            bnd: vec![Vec::new(); np],
            refused: HashMap::default(),
            failed: HashMap::default(),
            cap: f64::INFINITY,
            last_st: HashMap::default(),
            parked: Vec::new(),
            cloc: vec![0; t.corners()],
            eglue: vec![0; t.ne()],
            area: vec![0.0; np],
            weight: vec![0.0; np],
            length: vec![0.0; np],
            hull: vec![Vec::new(); np],
        };
        let ppu2 = self.numbers[num::PX_PER_UNIT] * self.numbers[num::PX_PER_UNIT];
        for (p, m) in members.iter().enumerate() {
            for &f in m {
                b.area[p] += face_area(&self.points, t, f as usize) * ppu2;
                for h in t.face_corners(f as usize) {
                    let e = t.cedge[h] as usize;
                    if self.cut_base[e] {
                        b.bnd[p].push(h as u32);
                        b.weight[p] += self.outline[e];
                        b.length[p] += self.length_px[e];
                    }
                }
            }
        }
        // Patch layouts.
        let layouts: Vec<(Local, bool, Vec<f64>)> = {
            let dsu = &b.dsu;
            let one = |p: usize| {
                let local = Local::new(t, &self.points, &members[p], |c| dsu.find(c));
                if let Some(uv) = tube_layout(&local, tube_uv) {
                    return (local, false, uv);
                }
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
            b.hull[p] = hull_of(local.corner.iter().map(|&h| (h, [b.corner_uv[2 * h as usize], b.corner_uv[2 * h as usize + 1]])));
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
                let (irr, plain, parts) = b.shape(c.a, c.b, &ev.glued);
                if !c.exact {
                    seq += 1;
                    let key = key_of(added, ev.cost, irr);
                    heap.push(Cand { key, exact: true, seq, ..c });
                    memo.insert((c.a, c.b), (c.va, c.vb, ev));
                    continue;
                }
                if !(ev.stretch <= 1.0 + self.build_stretch) {
                    dropped += 1;
                    b.failed.insert((c.a, c.b), (c.va, c.vb, ev.glued.len()));
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
                if plain > SHAPE_MAX && plain > parts && !b.saves_page(c.a, c.b, &ev) {
                    // Too irregular: not tried again until the smaller chart changes (as a union that lies over).
                    let small = if b.faces[c.a as usize].len() <= b.faces[c.b as usize].len() { c.a } else { c.b };
                    b.refused.insert((c.a, c.b), (small, b.ver[small as usize]));
                    continue;
                }
                if b.lies_over(c.a, c.b, &ev) {
                    lay_over += 1;
                    b.failed.insert((c.a, c.b), (c.va, c.vb, ev.glued.len()));
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
        // R6: a last round in which every chart, smallest first, tries to join a neighbour (the neighbour with the
        // longest shared border first), glued along all it shares or else hinged on the costliest shared segment (the
        // two unfold flat about it; the rest of the shared border stays cut). It runs after every other merge so the
        // compact merge order above is not disturbed. A chart of at most `FREE_FACES` faces joins whatever its shape
        // (its gap ring outweighs its box: joining always saves page); a larger one keeps the shape rule. Passes repeat
        // until one joins nothing (a pair is tried again only when one of its charts changed).
        let mut tried: HashMap<(u32, u32), (u32, u32), FxBuild> = HashMap::default();
        while !stop.load(Ordering::Relaxed) {
            let mut order: Vec<u32> = (0..np as u32).filter(|&p| b.cpar[p as usize] == p && b.chi[p as usize] == 1).collect();
            order.sort_by_key(|&p| (b.faces[p as usize].len(), p));
            let mut joined = 0usize;
            for p in order {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                if b.find(p) != p {
                    continue;
                }
                let mut nb: Vec<(u32, f64)> = b.neighbours(p).into_iter().filter(|(o, _)| b.chi[*o as usize] == 1)
                    .map(|(o, l)| (o, l.iter().map(|&s| self.segments[s as usize].length).sum::<f64>())).collect();
                nb.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
                // Each neighbour's full glue is measured up front (on worker threads on a big mesh; a measure depends only
                // on the two charts, so the joins are those of measuring one at a time), the hinge only when it is needed.
                let cands: Vec<u32> = nb.iter().map(|x| x.0).filter(|&o| tried.get(&(p, o)) != Some(&(b.ver[p as usize], b.ver[o as usize]))).collect();
                let lists: Vec<Vec<u32>> = cands.iter().map(|&o| b.between(o, p)).collect();
                // (whether the pair's full union is known to fail, and how many segments it glued)
                let known = |b: &Build, o: u32| {
                    let (x, y) = (o.min(p), o.max(p));
                    match b.failed.get(&(x, y)) {
                        Some(&(vx, vy, n)) if (vx, vy) == (b.ver[x as usize], b.ver[y as usize]) => Some(n),
                        _ => None,
                    }
                };
                let failed: Vec<Option<usize>> = cands.iter().map(|&o| known(&b, o)).collect();
                let skip: Vec<bool> = failed.iter().map(|x| x.is_some()).collect();
                let full: Vec<(Option<Eval>, usize)> = {
                    let me = &b;
                    let one = |k: usize| if skip[k] { (None, 0) } else { me.measure_one(cands[k], p, &lists[k], stop) };
                    if t.nf >= PARALLEL_FACES && cands.len() > 1 { (0..cands.len()).into_par_iter().map(one).collect() } else { (0..cands.len()).map(one).collect() }
                };
                for (k, (first, q)) in full.into_iter().enumerate() {
                    let o = cands[k];
                    tried.insert((p, o), (b.ver[p as usize], b.ver[o as usize]));
                    b.quick += q;
                    exact_evals += usize::from(!skip[k]);
                    let mut next = first;
                    let mut take = None;
                    for hinge in [false, true] {
                        if !hinge && skip[k] {
                            if failed[k] <= Some(1) {
                                break; // measured already in the rounds, glued along one segment: nothing to hinge
                            }
                            continue; // measured already in the rounds: over the build stretch or lying over itself
                        }
                        if hinge {
                            let (ev, q) = b.measure_one(o, p, &lists[k][..1], stop);
                            b.quick += q;
                            exact_evals += 1;
                            next = ev;
                        }
                        let Some(ev) = next.take() else { break };
                        let one = ev.glued.len() <= 1;
                        let free = b.faces[p as usize].len() <= FREE_FACES && irregularity(b.length[p as usize], b.area[p as usize]) <= SHAPE_MAX;
                        let shaped = free || {
                            let (_, plain, parts) = b.shape(o, p, &ev.glued);
                            !(plain > SHAPE_MAX && plain > parts) || b.saves_page(o, p, &ev)
                        };
                        if shaped && ev.stretch <= 1.0 + self.build_stretch && !b.lies_over(o, p, &ev) {
                            take = Some(ev);
                            break;
                        }
                        if one {
                            break;
                        }
                    }
                    if let Some(ev) = take {
                        top = top.max(ev.stretch);
                        if ev.pole { pole_merges += 1 } else { grid_merges += 1 }
                        let (stretch, cost, pole, glued) = (ev.stretch, ev.cost, ev.pole, ev.glued.clone());
                        b.apply(o, p, ev);
                        merges.push(Merge { segs: glued, stretch, cost, pole });
                        joined += 1;
                        break;
                    }
                }
            }
            if joined == 0 {
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

/// The UV Check's stretch of one chart (SPEC 9.1): per triangle of each face's fan, the root mean square of the
/// singular values of the map from the triangle (in its own plane) to the UV, the UVs scaled so the chart's UV area
/// equals its surface area; returns its median weighted by surface area and the surface area above `TUBE_L2_BAD`.
fn l2_stretch(local: &Local, uv: &[f64]) -> (f64, f64) {
    let (mut vals, mut w) = (Vec::new(), Vec::new());
    let (mut a3, mut a2) = (0.0, 0.0);
    let mut tris: Vec<([f64; 2], [f64; 2], f64, [f64; 2], [f64; 2])> = Vec::new();
    for f in 0..local.ip.len() - 1 {
        let (lo, hi) = (local.ip[f] as usize, local.ip[f + 1] as usize);
        for i in lo + 1..hi.saturating_sub(1) {
            let [p0, p1, p2] = [local.fl[lo] as usize, local.fl[i] as usize, local.fl[i + 1] as usize];
            let q = |v: usize| [local.pts[3 * v], local.pts[3 * v + 1], local.pts[3 * v + 2]];
            let (x0, x1, x2) = (q(p0), q(p1), q(p2));
            let e1 = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
            let e2 = [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]];
            let l1 = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
            let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let area = 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            if !(l1 > 0.0 && area > 0.0) {
                continue;
            }
            // the triangle in its own plane: (0, 0), (l1, 0), (bx, by)
            let bx = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / l1;
            let by = 2.0 * area / l1;
            let u = |v: usize| [uv[2 * v], uv[2 * v + 1]];
            let (u0, u1, u2) = (u(p0), u(p1), u(p2));
            let (d1, d2) = ([u1[0] - u0[0], u1[1] - u0[1]], [u2[0] - u0[0], u2[1] - u0[1]]);
            a3 += area;
            a2 += 0.5 * (d1[0] * d2[1] - d1[1] * d2[0]).abs();
            tris.push(([l1, 0.0], [bx, by], area, d1, d2));
        }
    }
    if !(a3 > 0.0 && a2 > 0.0) {
        return (f64::INFINITY, f64::INFINITY);
    }
    let s = (a3 / a2).sqrt();
    for (e1, e2, area, d1, d2) in tris {
        // J = [d1 d2] [e1 e2]^-1 (2x2); L2 = sqrt((s1^2 + s2^2) / 2) = |J|_F / sqrt(2)
        let det = e1[0] * e2[1] - e1[1] * e2[0];
        let inv = [[e2[1] / det, -e2[0] / det], [-e1[1] / det, e1[0] / det]];
        let j = [
            [d1[0] * inv[0][0] + d2[0] * inv[1][0], d1[0] * inv[0][1] + d2[0] * inv[1][1]],
            [d1[1] * inv[0][0] + d2[1] * inv[1][0], d1[1] * inv[0][1] + d2[1] * inv[1][1]],
        ];
        let fro = j[0][0] * j[0][0] + j[0][1] * j[0][1] + j[1][0] * j[1][0] + j[1][1] * j[1][1];
        vals.push(s * (fro / 2.0).sqrt());
        w.push(area);
    }
    let bad: f64 = vals.iter().zip(&w).filter(|(&x, _)| !(x <= TUBE_L2_BAD)).map(|(_, &a)| a).sum();
    (weighted_quantile(&vals, &w, 0.5), bad)
}

/// The `q` quantile of `values` weighted by `w` (the smallest value with at least that share of the weight at or
/// below it).
fn weighted_quantile(values: &[f64], w: &[f64], q: f64) -> f64 {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let total: f64 = w.iter().sum();
    let mut acc = 0.0;
    for &i in &idx {
        acc += w[i];
        if acc >= q * total {
            return values[i];
        }
    }
    idx.last().map_or(f64::NAN, |&i| values[i])
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
    /// Per chart: its border sides (corner h = the side from h to the next corner of its face).
    bnd: Vec<Vec<u32>>,
    /// Pairs refused because the union lay over itself: (smaller chart, its version then).
    refused: HashMap<(u32, u32), (u32, u32), FxBuild>,
    /// Pairs whose whole-border union was measured over the build stretch or lying over itself, with both versions
    /// then: the last round goes straight to the hinge for them.
    failed: HashMap<(u32, u32), (u32, u32, usize), FxBuild>,
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
    /// Per chart: area (px^2), outline weight and outline length (px) of its border sides, for its irregularity.
    area: Vec<f64>,
    weight: Vec<f64>,
    length: Vec<f64>,
    /// Per chart: the corners on the convex hull of its layout (for its box on the page).
    hull: Vec<Vec<u32>>,
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
    /// Per corner (a wedge root): its local vertex in the `Local` being built, valid where `cstamp` is `stamp`.
    cid: Vec<u32>,
    cstamp: Vec<u32>,
}

impl Scratch {
    fn fit(&mut self, nv: usize, nf: usize, nc: usize) {
        if self.vstamp.len() != nv || self.fstamp.len() != nf || self.cid.len() != nc {
            *self = Scratch { vstamp: vec![0; nv], fstamp: vec![0; nf], stamp: 0, cid: vec![0; nc], cstamp: vec![0; nc] };
        }
        if self.stamp == u32::MAX {
            self.vstamp.iter_mut().for_each(|x| *x = 0);
            self.fstamp.iter_mut().for_each(|x| *x = 0);
            self.cstamp.iter_mut().for_each(|x| *x = 0);
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

    /// Irregularity of the union of a and b glued along `glued` by outline weight; and by plain length, with the larger
    /// of its parts' (for `SHAPE_MAX`).
    fn shape(&self, a: u32, b: u32, glued: &[u32]) -> (f64, f64, f64) {
        let segs = &self.tree.segments;
        let (a, b) = (a as usize, b as usize);
        let area = self.area[a] + self.area[b];
        let w: f64 = glued.iter().map(|&s| segs[s as usize].weight).sum();
        let l: f64 = glued.iter().map(|&s| segs[s as usize].length).sum();
        let by_weight = irregularity((self.weight[a] + self.weight[b] - 2.0 * w).max(0.0), area);
        let plain = irregularity((self.length[a] + self.length[b] - 2.0 * l).max(0.0), area);
        let parts = irregularity(self.length[a], self.area[a]).max(irregularity(self.length[b], self.area[b]));
        (by_weight, plain, parts)
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
            // The shape changes whenever a chart grows, so the key is estimated afresh (all open segments glued).
            let (mis, c) = self.estimate(&list);
            let key = key_of(mis, c, self.shape(a, b, &list).0);
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
                sc.fit(me.tree.topo.nv, me.tree.topo.nf, me.tree.topo.corners());
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

    /// `measure` on this thread over the committed glue.
    fn measure_one(&self, a: u32, b: u32, list: &[u32], stop: &AtomicBool) -> (Option<Eval>, usize) {
        SCRATCH.with(|sc| {
            let mut sc = sc.borrow_mut();
            sc.fit(self.tree.topo.nv, self.tree.topo.nf, self.tree.topo.corners());
            let mut dsu = Overlay::new(&self.dsu);
            self.measure(&mut dsu, &mut sc, a, b, list, stop)
        })
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
        let local = Local::new_in(t, &tree.points, &sub, |c| dsu.find(c), sc);
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

    /// Page px of the union's box with its gap ring against the two parts' (`GAP_PX`): does the merge save page?
    fn saves_page(&mut self, a: u32, b: u32, ev: &Eval) -> bool {
        let t = &self.tree.topo;
        let ppu = self.tree.numbers[num::PX_PER_UNIT];
        let local = &ev.local;
        for (k, &h) in local.corner.iter().enumerate() {
            self.cloc[h as usize] = local.fl[k] as u32 + 1;
        }
        let mut pts: Vec<[f64; 2]> = Vec::new();
        for (c, moved) in [(a, false), (b, true)] {
            for &h in &self.hull[c as usize] {
                let h = h as usize;
                let k = self.cloc[h];
                pts.push(if k > 0 {
                    [ev.uv[2 * (k - 1) as usize], ev.uv[2 * (k - 1) as usize + 1]]
                } else {
                    let (x, y) = (self.corner_uv[2 * h], self.corner_uv[2 * h + 1]);
                    match ev.rigid {
                        Some(r) if moved && self.owner[t.cface[h] as usize] == b => {
                            let (x, y) = apply_rigid(r, x, y);
                            [x, y]
                        }
                        _ => [x, y],
                    }
                });
            }
        }
        for v in 0..local.n() {
            pts.push([ev.uv[2 * v], ev.uv[2 * v + 1]]);
        }
        for &h in &local.corner {
            self.cloc[h as usize] = 0;
        }
        let page = |p: &[[f64; 2]]| {
            let (w, h) = min_box(p);
            (w * ppu + GAP_PX) * (h * ppu + GAP_PX)
        };
        let (wu, hu) = min_box(&pts);
        let (wu, hu) = (wu * ppu, hu * ppu);
        if wu * hu > 2.0 * GAP_PX * (wu + hu) {
            return false; // not a chart whose gap ring outweighs its box
        }
        let part = |c: u32| -> Vec<[f64; 2]> { self.hull[c as usize].iter().map(|&h| [self.corner_uv[2 * h as usize], self.corner_uv[2 * h as usize + 1]]).collect() };
        page(&pts) <= page(&part(a)) + page(&part(b))
    }

    /// Glue an evaluated union for good and take its layout; returns the surviving chart.
    fn apply(&mut self, a: u32, b: u32, ev: Eval) -> u32 {
        let tree = self.tree;
        let t = &tree.topo;
        let mut chi_new = self.chi[a as usize] + self.chi[b as usize];
        let (ua, ub) = (a as usize, b as usize);
        let segs = &tree.segments;
        let area = self.area[ua] + self.area[ub];
        let weight = (self.weight[ua] + self.weight[ub] - 2.0 * ev.glued.iter().map(|&s| segs[s as usize].weight).sum::<f64>()).max(0.0);
        let length = (self.length[ua] + self.length[ub] - 2.0 * ev.glued.iter().map(|&s| segs[s as usize].length).sum::<f64>()).max(0.0);
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
        let ids: Vec<u32> = self.hull[a as usize].iter().chain(self.hull[b as usize].iter()).chain(ev.local.corner.iter()).copied().collect();
        let uv = &self.corner_uv;
        let new_hull = hull_of(ids.iter().map(|&h| (h, [uv[2 * h as usize], uv[2 * h as usize + 1]])));
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
        self.hull[o as usize] = Vec::new();
        self.hull[r as usize] = new_hull;
        self.area[r as usize] = area;
        self.weight[r as usize] = weight;
        self.length[r as usize] = length;
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

/// Corners on the convex hull of the given (corner, UV) points (Andrew's monotone chain).
fn hull_of(pts: impl Iterator<Item = (u32, [f64; 2])>) -> Vec<u32> {
    let mut p: Vec<(u32, [f64; 2])> = pts.filter(|x| x.1[0].is_finite() && x.1[1].is_finite()).collect();
    p.sort_by(|x, y| x.1[0].total_cmp(&y.1[0]).then(x.1[1].total_cmp(&y.1[1])));
    p.dedup_by(|x, y| x.1 == y.1);
    if p.len() < 3 {
        return p.into_iter().map(|x| x.0).collect();
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut out: Vec<(u32, [f64; 2])> = Vec::with_capacity(2 * p.len());
    for pass in 0..2 {
        let start = out.len();
        let it: Box<dyn Iterator<Item = &(u32, [f64; 2])>> = if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &q in it {
            while out.len() >= start + 2 && cross(out[out.len() - 2].1, out[out.len() - 1].1, q.1) <= 0.0 {
                out.pop();
            }
            out.push(q);
        }
        out.pop();
    }
    out.into_iter().map(|x| x.0).collect()
}

/// Sides (w, h) of the smallest-area box around the points, one side along a hull edge (Toussaint 1983).
fn min_box(pts: &[[f64; 2]]) -> (f64, f64) {
    let ids = hull_of(pts.iter().enumerate().map(|(i, &q)| (i as u32, q)));
    let h: Vec<[f64; 2]> = ids.iter().map(|&i| pts[i as usize]).collect();
    if h.len() < 3 {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for q in &h {
            for d in 0..2 {
                lo[d] = lo[d].min(q[d]);
                hi[d] = hi[d].max(q[d]);
            }
        }
        return if h.is_empty() { (0.0, 0.0) } else { (hi[0] - lo[0], hi[1] - lo[1]) };
    }
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for i in 0..h.len() {
        let (p, q) = (h[i], h[(i + 1) % h.len()]);
        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
        let l = (dx * dx + dy * dy).sqrt();
        if l <= 0.0 {
            continue;
        }
        let (ux, uy) = (dx / l, dy / l);
        let (mut a0, mut a1, mut b0, mut b1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for r in &h {
            let (x, y) = (r[0] * ux + r[1] * uy, -r[0] * uy + r[1] * ux);
            a0 = a0.min(x);
            a1 = a1.max(x);
            b0 = b0.min(y);
            b1 = b1.max(y);
        }
        if (a1 - a0) * (b1 - b0) < best.0 {
            best = ((a1 - a0) * (b1 - b0), a1 - a0, b1 - b0);
        }
    }
    (best.1, best.2)
}

/// Dijkstra over the vertices of one part (edges with `inside`), from the given sources: (distance, edge taken) per
/// vertex reached.
fn dijkstra(t: &Topo, sources: &[u32], weight: impl Fn(usize) -> f64, inside: impl Fn(usize) -> bool) -> HashMap<u32, (f64, u32), FxBuild> {
    let mut dist: HashMap<u32, (f64, u32), FxBuild> = HashMap::default();
    let mut heap: BinaryHeap<(std::cmp::Reverse<u64>, u32)> = BinaryHeap::new();
    for &s in sources {
        dist.insert(s, (0.0, NONE));
        heap.push((std::cmp::Reverse(0f64.to_bits()), s));
    }
    while let Some((std::cmp::Reverse(d), v)) = heap.pop() {
        let d = f64::from_bits(d);
        if d > dist[&v].0 {
            continue;
        }
        for &e in t.vedges(v as usize) {
            if !inside(e as usize) {
                continue;
            }
            let w = t.other(e as usize, v);
            let nd = d + weight(e as usize).max(0.0);
            if dist.get(&w).is_none_or(|x| nd < x.0) {
                dist.insert(w, (nd, e));
                heap.push((std::cmp::Reverse(nd.to_bits()), w));
            }
        }
    }
    dist
}

/// The strip layout of a tube patch (`tube_cuts`) per local vertex, when the patch is one.
fn tube_layout(local: &Local, tube_uv: &HashMap<u32, [f64; 2], FxBuild>) -> Option<Vec<f64>> {
    if local.corner.is_empty() || !tube_uv.contains_key(&local.corner[0]) {
        return None;
    }
    let mut uv = vec![f64::NAN; 2 * local.n()];
    for (k, &h) in local.corner.iter().enumerate() {
        let x = tube_uv.get(&h)?;
        uv[2 * local.fl[k] as usize] = x[0];
        uv[2 * local.fl[k] as usize + 1] = x[1];
    }
    Some(uv)
}

/// One thin tube laid out as one strip chart: its faces, its cut, and its layout per corner.
struct Tube {
    faces: Vec<u32>,
    cut: Vec<u32>,
    corner_uv: Vec<(u32, [f64; 2])>,
}

/// Start layout of a strip chart (R5, ours): Tutte's embedding (Tutte 1963, Floater 1997) with the border on an ellipse
/// as long as the strip (`aspect` = length / width), its two tips at the strip's ends (the middle of the border runs
/// through the corners of `ends`, global vertices), each side of the border spread over its half by arc length. A long
/// strip on a circle (plain Tutte) starts squashed and ARAP takes very long to open it.
fn strip_start(t: &Topo, local: &Local, ends: [u32; 2], area: f64, stop: &AtomicBool) -> Option<Vec<f64>> {
    let n = local.n();
    let nf = local.ip.len() - 1;
    let mut dir: HashMap<(i64, i64), (), FxBuild> = HashMap::default();
    for f in 0..nf {
        let (lo, hi) = (local.ip[f] as usize, local.ip[f + 1] as usize);
        for i in lo..hi {
            let j = if i + 1 == hi { lo } else { i + 1 };
            dir.insert((local.fl[i], local.fl[j]), ());
        }
    }
    let mut next = vec![-1i64; n];
    let mut glob = vec![NONE; n];
    for (k, &h) in local.corner.iter().enumerate() {
        glob[local.fl[k] as usize] = t.flat[h as usize];
    }
    for &(a, b) in dir.keys() {
        if !dir.contains_key(&(b, a)) {
            next[a as usize] = b;
        }
    }
    // The longest border loop.
    let mut seen = vec![false; n];
    let mut border: Vec<usize> = Vec::new();
    for s in 0..n {
        if next[s] < 0 || seen[s] {
            continue;
        }
        let mut lp = Vec::new();
        let mut v = s;
        while !seen[v] && next[v] >= 0 {
            seen[v] = true;
            lp.push(v);
            v = next[v] as usize;
        }
        if v == s && lp.len() > border.len() {
            border = lp;
        }
    }
    let m = border.len();
    if m < 3 {
        return None;
    }
    let p = |v: usize| [local.pts[3 * v], local.pts[3 * v + 1], local.pts[3 * v + 2]];
    let mut at = vec![0f64; m + 1];
    for i in 0..m {
        let (a, b) = (p(border[i]), p(border[(i + 1) % m]));
        at[i + 1] = at[i] + ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    }
    let total = at[m];
    if !(total > 0.0) {
        return None;
    }
    // The border's corners: the corners of the two ends (one where an end is a point, two where the cut meets a border
    // loop there). Runs between corners go onto the sides of a convex outline: a rectangle as long as the cut and as
    // wide as the loops, its short side shrunk to a point (with a bevel) at an end that is a point.
    let occ = |g: u32| -> Vec<usize> { (0..m).filter(|&i| glob[border[i]] == g).collect() };
    let (oa, ob) = (occ(ends[0]), occ(ends[1]));
    if oa.is_empty() || ob.is_empty() || oa.len() > 2 || ob.len() > 2 || ends[0] == ends[1] {
        return None;
    }
    let fwd = |i: usize, j: usize| (at[j] - at[i]).rem_euclid(total);
    // a1 -> a2 is the short run at end a (no corner of b inside); likewise b1 -> b2.
    let pick = |o: &[usize], other: &[usize]| -> (usize, usize) {
        if o.len() == 1 {
            return (o[0], o[0]);
        }
        let inside = |i: usize, j: usize| other.iter().any(|&k| fwd(i, k) < fwd(i, j));
        if !inside(o[0], o[1]) { (o[0], o[1]) } else { (o[1], o[0]) }
    };
    let (a1, a2) = pick(&oa, &ob);
    let (b1, b2) = pick(&ob, &oa);
    let (ra, rb) = (fwd(a1, a2), fwd(b1, b2));
    let (bottom, top) = (fwd(a2, b1), fwd(b2, a1));
    let long = 0.5 * (bottom + top);
    let wide = match (oa.len(), ob.len()) {
        (2, 2) => 0.5 * (ra + rb),
        (2, _) => ra,
        (_, 2) => rb,
        _ => area / long.max(1e-300),
    };
    if !(long > 0.0 && wide > 0.0) {
        return None;
    }
    let bevel = (0.5 * wide).min(0.25 * long);
    let (ha, hb) = (if oa.len() == 2 { 0.0 } else { bevel }, if ob.len() == 2 { 0.0 } else { bevel });
    let left: Vec<[f64; 2]> = if oa.len() == 2 { vec![[0.0, wide], [0.0, 0.0]] } else { vec![[0.0, 0.5 * wide]] };
    let right: Vec<[f64; 2]> = if ob.len() == 2 { vec![[long, 0.0], [long, wide]] } else { vec![[long, 0.5 * wide]] };
    let mut lower = vec![*left.last().unwrap()];
    if ha > 0.0 {
        lower.push([ha, 0.0]);
    }
    if hb > 0.0 {
        lower.push([long - hb, 0.0]);
    }
    lower.push(right[0]);
    let mut upper = vec![*right.last().unwrap()];
    if hb > 0.0 {
        upper.push([long - hb, wide]);
    }
    if ha > 0.0 {
        upper.push([ha, wide]);
    }
    upper.push(left[0]);
    // A point at share x (0..1) of a polyline's length.
    let along = |line: &[[f64; 2]], x: f64| -> [f64; 2] {
        let seg: Vec<f64> = line.windows(2).map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt()).collect();
        let tot: f64 = seg.iter().sum();
        if line.len() == 1 || !(tot > 0.0) {
            return line[0];
        }
        let mut want = x.clamp(0.0, 1.0) * tot;
        for (k, &l) in seg.iter().enumerate() {
            if want <= l || k + 1 == seg.len() {
                let r = if l > 0.0 { (want / l).min(1.0) } else { 0.0 };
                return [line[k][0] + r * (line[k + 1][0] - line[k][0]), line[k][1] + r * (line[k + 1][1] - line[k][1])];
            }
            want -= l;
        }
        *line.last().unwrap()
    };
    let runs: [(usize, f64, &[[f64; 2]]); 4] = [(a1, ra, &left), (a2, bottom, &lower), (b1, rb, &right), (b2, top, &upper)];
    let mut uv = vec![0f64; 2 * n];
    let mut pins = Vec::with_capacity(m);
    for i in 0..m {
        // the run this border vertex lies on: the last run start at or before it
        let (start, len, line) = runs.iter().filter(|r| r.1 > 0.0 || r.0 == i).min_by(|x, y| fwd(x.0, i).total_cmp(&fwd(y.0, i))).copied()?;
        let x = if len > 0.0 { fwd(start, i) / len } else { 0.0 };
        let q = along(line, x);
        let v = border[i];
        uv[2 * v] = q[0];
        uv[2 * v + 1] = q[1];
        pins.push(v);
    }
    let scale = long.max(wide);
    let mut b = uvflat::Builder::new(n);
    for f in 0..nf {
        let (lo, hi) = (local.ip[f] as usize, local.ip[f + 1] as usize);
        for i in lo..hi {
            let j = if i + 1 == hi { lo } else { i + 1 };
            b.edge(local.fl[i] as usize, local.fl[j] as usize, 0.5);
        }
    }
    let lin = uvflat::Lin::new(b.build(), &pins, 0.0, false, stop).ok()?;
    let mut rhs = vec![0f64; 2 * n];
    for &v in &pins {
        rhs[2 * v] = uv[2 * v];
        rhs[2 * v + 1] = uv[2 * v + 1];
    }
    let mut x = uv;
    lin.solve::<2>(&mut rhs, &mut x, scale * 1e-9, stop);
    if x.iter().all(|v| v.is_finite()) { Some(x) } else { None }
}

/// Thin tubes (R5, our design): a whole part (faces joined across manifold edges, no forced line on it) that wraps round
/// (not a disk) and whose surface is a long strip, length^2 / area >= `TUBE_ASPECT`, is laid out as strips: cut across
/// into pieces about `PIECE_ASPECT` times as long as the tube is round (bands of the distance from one far end, so a
/// closed loop such as a chain link or a ring of rope gives two pieces side by side per band), and each piece cut once
/// along its length on the cheapest edges by the cut cost (6.3), so that cut runs on the side the camera sees least.
/// The length is the part's longest shortest path (twice that around a closed loop, found as two separate bands half
/// way). Each piece starts from `strip_start`; a piece is kept when it lays out within the stretch rule of
/// `TUBE_L2_MEDIAN`, flip-free and without its border crossing itself, else its faces keep their ordinary patches.
/// Springs, chain links, rope and wire then give a few strips instead of dozens of charts too small to pay for their
/// gap ring on the page.
fn tube_cuts(t: &Topo, points: &[f64], cost: &[f64], forced_line: &[bool], stop: &AtomicBool) -> Vec<Tube> {
    let ne = t.ne();
    let mut d = Dsu::new(t.nf);
    for e in 0..ne {
        if !t.wall[e] && !forced_line[e] {
            let [f0, f1] = t.faces_of(e);
            d.union(f0, f1);
        }
    }
    let mut id: HashMap<u32, u32, FxBuild> = HashMap::default();
    let comp: Vec<u32> = (0..t.nf as u32)
        .map(|f| {
            let r = d.find(f);
            let k = id.len() as u32;
            *id.entry(r).or_insert(k)
        })
        .collect();
    let nc = id.len();
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); nc];
    let mut bad = vec![false; nc];
    for f in 0..t.nf {
        members[comp[f] as usize].push(f as u32);
        for h in t.face_corners(f) {
            bad[comp[f] as usize] |= forced_line[t.cedge[h] as usize];
        }
    }
    let len3 = |e: usize| {
        let (a, b) = (t.ea[e] as usize, t.eb[e] as usize);
        (0..3).map(|k| (points[3 * a + k] - points[3 * b + k]).powi(2)).sum::<f64>().sqrt()
    };
    // A tube wraps round: a part that is already a disk (a plate, a ribbon) is left to the merge tree.
    let part_chi = chi_per(t, &t.wall, &comp, nc);
    // Per candidate part: its pieces as (faces, cut path, ends) and the edges between pieces.
    type Piece = (Vec<u32>, Vec<u32>, [u32; 2]);
    let one = |c: usize| -> Option<(Vec<Piece>, Vec<u32>)> {
        if bad[c] || part_chi[c] == 1 || stop.load(Ordering::Relaxed) || members[c].len() < 4 {
            return None;
        }
        let faces = &members[c];
        let in_part = |e: usize| {
            let [f0, f1] = t.faces_of(e);
            comp[f0 as usize] as usize == c && (f1 == NONE || comp[f1 as usize] as usize == c)
        };
        let area: f64 = faces.iter().map(|&f| face_area(points, t, f as usize)).sum();
        if !(area > 0.0) {
            return None;
        }
        let v0 = t.flat[t.face_corners(faces[0] as usize).start];
        let farthest = |dist: &HashMap<u32, (f64, u32), FxBuild>| dist.iter().map(|(&v, &(x, _))| (v, x)).max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
        let (u, _) = farthest(&dijkstra(t, &[v0], len3, in_part))?;
        let su = dijkstra(t, &[u], len3, in_part);
        let (_, diameter) = farthest(&su)?;
        if !(diameter > 0.0) {
            return None;
        }
        let s_face = |f: u32| -> f64 {
            let r = t.face_corners(f as usize);
            let k = r.len() as f64;
            r.map(|h| su.get(&t.flat[h]).map_or(0.0, |x| x.0)).sum::<f64>() / k
        };
        let sf: HashMap<u32, f64, FxBuild> = faces.iter().map(|&f| (f, s_face(f))).collect();
        // Bands of faces by distance: their pieces are the faces of one band joined across uncut edges.
        let pieces_of = |band: &dyn Fn(u32) -> i64| -> HashMap<u32, u32, FxBuild> {
            let mut loc: HashMap<u32, u32, FxBuild> = faces.iter().enumerate().map(|(k, &f)| (f, k as u32)).collect();
            let mut dd = Dsu::new(faces.len());
            for &f in faces {
                for h in t.face_corners(f as usize) {
                    let e = t.cedge[h] as usize;
                    if t.wall[e] {
                        continue;
                    }
                    let [f0, f1] = t.faces_of(e);
                    if band(f0) == band(f1) {
                        dd.union(loc[&f0], loc[&f1]);
                    }
                }
            }
            for x in loc.values_mut() {
                *x = dd.find(*x);
            }
            loc
        };
        // A closed loop: the faces half way along form two separate bands.
        let mid = |f: u32| -> i64 { i64::from((0.4 * diameter..=0.6 * diameter).contains(&sf[&f])) };
        let mid_pieces = pieces_of(&mid);
        let mut halves: Vec<u32> = faces.iter().filter(|&&f| mid(f) == 1).map(|f| mid_pieces[f]).collect();
        halves.sort_unstable();
        halves.dedup();
        // (a part with a border is no closed loop: on a short wide tube the half-way band also splits in two)
        let closed = faces.iter().all(|&f| t.face_corners(f as usize).all(|h| !t.wall[t.cedge[h] as usize]));
        let length = if closed && halves.len() >= 2 { 2.0 * diameter } else { diameter };
        let aspect = length * length / area;
        if !(aspect >= TUBE_ASPECT) {
            return None;
        }
        // Pieces: bands of `PIECE_ASPECT` times the mean round (area / length) along the distance.
        let round = area / length;
        let width = PIECE_ASPECT * round;
        let bins = ((diameter / width).round() as i64).max(1);
        let bin = |f: u32| -> i64 { ((sf[&f] / diameter * bins as f64).floor() as i64).clamp(0, bins - 1) };
        let piece = pieces_of(&bin);
        let mut groups: HashMap<u32, Vec<u32>, FxBuild> = HashMap::default();
        for &f in faces {
            groups.entry(piece[&f]).or_default().push(f);
        }
        let mut rings: Vec<u32> = Vec::new();
        for &f in faces {
            for h in t.face_corners(f as usize) {
                let e = t.cedge[h] as usize;
                if !t.wall[e] {
                    let [f0, f1] = t.faces_of(e);
                    if f0 == f && piece[&f0] != piece[&f1] {
                        rings.push(e as u32);
                    }
                }
            }
        }
        let is_ring: HashMap<u32, (), FxBuild> = rings.iter().map(|&e| (e, ())).collect();
        let mut keys: Vec<u32> = groups.keys().copied().collect();
        keys.sort_unstable();
        let mut out: Vec<Piece> = Vec::new();
        for k in keys {
            let pf = &groups[&k];
            let inside = |e: usize| {
                let [f0, f1] = t.faces_of(e);
                piece.get(&f0) == Some(&k) && (f1 == NONE || piece.get(&f1) == Some(&k))
            };
            // Border loops of the piece (walls and rings).
            let border: Vec<usize> = pf.iter().flat_map(|&f| t.face_corners(f as usize)).map(|h| t.cedge[h] as usize).filter(|&e| t.wall[e] || is_ring.contains_key(&(e as u32))).collect();
            let mut ld = Dsu::new(t.nv);
            for &e in &border {
                ld.union(t.ea[e], t.eb[e]);
            }
            let mut roots: Vec<u32> = border.iter().map(|&e| ld.find(t.ea[e])).collect();
            roots.sort_unstable();
            roots.dedup();
            let root: HashMap<u32, u32, FxBuild> = border.iter().flat_map(|&e| [t.ea[e], t.eb[e]]).map(|v| (v, ld.find(v))).collect();
            let on = |r: u32| -> Vec<u32> {
                let mut v: Vec<u32> = root.iter().filter(|(_, &x)| x == r).map(|(&v, _)| v).collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            let inner = |e: usize| inside(e) && !t.wall[e] && !is_ring.contains_key(&(e as u32));
            let path_back = |dist: &HashMap<u32, (f64, u32), FxBuild>, mut v: u32| -> (Vec<u32>, u32) {
                let mut path = Vec::new();
                while let Some(&(_, e)) = dist.get(&v) {
                    if e == NONE {
                        break;
                    }
                    path.push(e);
                    v = t.other(e as usize, v);
                }
                (path, v)
            };
            let (from, to): (Vec<u32>, Vec<u32>) = match roots.len() {
                0 => {
                    // The whole closed part in one piece: between its two far ends.
                    let lp = dijkstra(t, &[u], len3, inside);
                    (vec![u], vec![farthest(&lp)?.0])
                }
                1 => {
                    // A cap: from the point farthest from its border loop to the loop.
                    let rim = on(roots[0]);
                    let lp = dijkstra(t, &rim, len3, inner);
                    (vec![farthest(&lp)?.0], rim)
                }
                2 => (on(roots[0]), on(roots[1])),
                _ => return None,
            };
            let dist = dijkstra(t, &from, |e| cost[e], inner);
            let end = to.iter().filter_map(|&v| dist.get(&v).map(|x| (v, x.0))).min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))?;
            let (path, start) = path_back(&dist, end.0);
            out.push((pf.clone(), path, [start, end.0]));
        }
        Some((out, rings))
    };
    let cands: Vec<(usize, (Vec<Piece>, Vec<u32>))> =
        if t.nf >= PARALLEL_FACES { (0..nc).into_par_iter().filter_map(|c| one(c).map(|x| (c, x))).collect() } else { (0..nc).filter_map(|c| one(c).map(|x| (c, x))).collect() };
    if cands.is_empty() {
        return Vec::new();
    }
    let mut cut = t.wall.clone();
    let mut label = vec![NONE; t.nf];
    let mut np = 0u32;
    for (_, (pieces, rings)) in &cands {
        for &e in rings {
            cut[e as usize] = true;
        }
        for (pf, path, _) in pieces {
            for &e in path {
                cut[e as usize] = true;
            }
            for &f in pf {
                label[f as usize] = np;
            }
            np += 1;
        }
    }
    // Faces outside every candidate get their own label (not checked).
    let mut lab = label.clone();
    for f in 0..t.nf {
        if lab[f] == NONE {
            lab[f] = np;
        }
    }
    let chi = chi_per(t, &cut, &lab, np as usize + 1);
    let wr = wedges(t, &cut);
    let lay = |pf: &[u32], ends: [u32; 2]| -> Option<(Vec<(u32, [f64; 2])>, f64, f64)> {
        if chi[label[pf[0] as usize] as usize] != 1 {
            return None;
        }
        let local = Local::new(t, points, pf, |h| wr[h as usize]);
        let area: f64 = pf.iter().map(|&f| face_area(points, t, f as usize)).sum();
        let start = strip_start(t, &local, ends, area, stop)?;
        let opts = uvflat::FlatOpts { rows: uvflat::Rows::Off, start: uvflat::Start::Given, ..Default::default() };
        let uv = uvflat::flatten_chart(&local.chart(), Some(&start), None, &opts, stop).ok()?.uv;
        let (l2_med, l2_bad) = l2_stretch(&local, &uv);
        if !(l2_med <= TUBE_L2_MEDIAN) || uvflat::count_flips(&local.chart().faces, &uv) > 0 {
            return None;
        }
        let mut cuv = HashMap::<u32, [f64; 2], FxBuild>::default();
        for (k, &h) in local.corner.iter().enumerate() {
            let v = local.fl[k] as usize;
            cuv.insert(h, [uv[2 * v], uv[2 * v + 1]]);
        }
        let mut sides: Vec<Side> = Vec::new();
        for &f in pf {
            for h in t.face_corners(f as usize) {
                if cut[t.cedge[h] as usize] {
                    let n = t.next(h);
                    sides.push(Side { p: cuv[&(h as u32)], q: cuv[&(n as u32)], v: [wr[h], wr[n]], class: 2 });
                }
            }
        }
        if sides_cross(&mut sides) {
            return None;
        }
        Some((cuv.into_iter().collect::<Vec<_>>(), l2_bad, area))
    };
    // Each piece on its own: one that does not lay out well keeps the ordinary patches. Faces stretching more than
    // `TUBE_L2_BAD` take from one budget, `TUBE_BAD_SHARE` of the mesh's area, the cleanest pieces first.
    let jobs: Vec<(usize, usize)> = cands.iter().enumerate().flat_map(|(i, c)| (0..c.1 .0.len()).map(move |k| (i, k))).collect();
    let run = |&(i, k): &(usize, usize)| lay(&cands[i].1 .0[k].0, cands[i].1 .0[k].2);
    let laid: Vec<Option<(Vec<(u32, [f64; 2])>, f64, f64)>> = if t.nf >= PARALLEL_FACES { jobs.par_iter().map(run).collect() } else { jobs.iter().map(run).collect() };
    let total: f64 = (0..t.nf).map(|f| face_area(points, t, f)).sum();
    let mut order: Vec<usize> = (0..jobs.len()).filter(|&j| laid[j].is_some()).collect();
    let share = |j: usize| laid[j].as_ref().map_or(f64::INFINITY, |x| x.1 / x.2.max(1e-300));
    order.sort_by(|&a, &b| share(a).total_cmp(&share(b)).then(a.cmp(&b)));
    let mut keep = vec![false; jobs.len()];
    let mut spent = 0.0;
    for j in order {
        let bad = laid[j].as_ref().unwrap().1;
        if spent + bad <= TUBE_BAD_SHARE * total {
            spent += bad;
            keep[j] = true;
        }
    }
    let mut out: Vec<Tube> = Vec::new();
    let mut laid = laid;
    let mut j = 0;
    for (_, (pieces, rings)) in cands.into_iter() {
        let mut tube = Tube { faces: Vec::new(), cut: Vec::new(), corner_uv: Vec::new() };
        let mut kept = HashMap::<u32, (), FxBuild>::default();
        for (pf, path, _) in pieces.into_iter() {
            j += 1;
            let j = j - 1;
            if !keep[j] {
                continue;
            }
            tube.corner_uv.extend(laid[j].take().unwrap().0);
            for &f in &pf {
                kept.insert(f, ());
            }
            tube.faces.extend(pf);
            tube.cut.extend(path);
        }
        if tube.faces.is_empty() {
            continue;
        }
        tube.cut.extend(rings.into_iter().filter(|&e| t.faces_of(e as usize).iter().any(|f| kept.contains_key(f))));
        out.push(tube);
    }
    out
}

/// Union-find over corners and over faces for `Tree::smooth`, reset by a stamp (an entry is valid where its stamp is
/// the current one).
struct SmoothScratch {
    cpar: Vec<u32>,
    cst: Vec<u32>,
    fpar: Vec<u32>,
    fst: Vec<u32>,
    stamp: u32,
}

/// The outline smoothing of the full tree (`Tree::smooth`): the moves in order (patch, a patch of the chart it
/// joined), and the start layout with the moved patches fitted onto their charts (empty: no move).
pub struct Smooth {
    pub moves: Vec<(u32, u32)>,
    pub start: Vec<f64>,
}

/// Faces and (two-patch) segments per patch.
struct SmoothCtx {
    pfaces: Vec<Vec<u32>>,
    psegs: Vec<Vec<u32>>,
}

impl SmoothCtx {
    fn members(&self, root: &[u32]) -> Vec<Vec<u32>> {
        let mut m: Vec<Vec<u32>> = vec![Vec::new(); root.len()];
        for (p, &r) in root.iter().enumerate() {
            m[r as usize].push(p as u32);
        }
        m
    }
}

impl SmoothScratch {
    fn new(corners: usize, faces: usize) -> Self {
        SmoothScratch { cpar: vec![0; corners], cst: vec![0; corners], fpar: vec![0; faces], fst: vec![0; faces], stamp: 0 }
    }
    fn find(par: &mut [u32], st: &mut [u32], stamp: u32, mut x: u32) -> u32 {
        if st[x as usize] != stamp {
            st[x as usize] = stamp;
            par[x as usize] = x;
            return x;
        }
        while par[x as usize] != x {
            let up = par[par[x as usize] as usize];
            par[x as usize] = up;
            x = up;
        }
        x
    }
    fn cfind(&mut self, x: u32) -> u32 {
        Self::find(&mut self.cpar, &mut self.cst, self.stamp, x)
    }
    fn join(par: &mut [u32], st: &mut [u32], stamp: u32, a: u32, b: u32) -> bool {
        let (ra, rb) = (Self::find(par, st, stamp, a), Self::find(par, st, stamp, b));
        if ra == rb {
            return false;
        }
        par[ra.max(rb) as usize] = ra.min(rb);
        true
    }
    fn cjoin(&mut self, a: u32, b: u32) -> bool {
        Self::join(&mut self.cpar, &mut self.cst, self.stamp, a, b)
    }
    fn fjoin(&mut self, a: u32, b: u32) -> bool {
        Self::join(&mut self.fpar, &mut self.fst, self.stamp, a, b)
    }
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
        let mut l = self.compute_level(x_pct);
        crate::uvhide::apply(self, &mut l); // R7hide: borders slid out of sight
        let l = Arc::new(l);
        self.cache.lock().unwrap().insert(key, l.clone());
        l
    }

    /// The start layout to flatten a level's charts from (`Level::start`, else the tree's `start_uv`).
    pub fn start_of<'a>(&'a self, l: &'a Level) -> &'a [f64] {
        if l.start.is_empty() { &self.start_uv } else { &l.start }
    }

    pub fn compute_level(&self, x_pct: f64) -> Level {
        let started = Instant::now();
        let t = &self.topo;
        let (mut root, mut glued, chi, used, segs_glued) = self.replay(1.0 + x_pct / 100.0);
        let sm = self.smoothing();
        let moved = !sm.moves.is_empty();
        if moved {
            let ctx = self.smooth_ctx();
            let mut sc = SmoothScratch::new(t.corners(), t.nf);
            let mut members = ctx.members(&root);
            for &(p, q) in &sm.moves {
                let (a, b) = (root[p as usize], root[q as usize]);
                if a != b {
                    let mut undo = Vec::new();
                    self.move_patch(&ctx, &mut sc, p, b, &mut root, &mut glued, &mut members, &mut undo);
                }
            }
        }
        let cut: Vec<bool> = (0..t.ne()).map(|e| self.cut_base[e] && !glued[e]).collect();
        let mut id: HashMap<u32, u32, FxBuild> = HashMap::default();
        let (chart, not_disk) = if moved {
            // A chart that gave a patch away can fall apart in pieces at a level below the full tree: one chart per piece.
            let mut d = Dsu::new(t.nf);
            for e in 0..t.ne() {
                if !cut[e] {
                    let [f0, f1] = t.faces_of(e);
                    d.union(f0, f1);
                }
            }
            let chart: Vec<u32> = (0..t.nf as u32).map(|f| {
                let r = d.find(f);
                let k = id.len() as u32;
                *id.entry(r).or_insert(k)
            }).collect();
            let nd = chi_per(t, &cut, &chart, id.len()).iter().filter(|&&c| c != 1).count();
            (chart, nd)
        } else {
            let chart: Vec<u32> = (0..t.nf).map(|f| {
                let r = root[self.patch[f] as usize];
                let k = id.len() as u32;
                *id.entry(r).or_insert(k)
            }).collect();
            let nd = id.keys().filter(|&&r| chi[r as usize] != 1).count();
            (chart, nd)
        };
        Level {
            chart,
            start: if moved { sm.start.clone() } else { Vec::new() },
            cut,
            charts: id.len(),
            not_disk,
            merges_used: used,
            segments_glued: segs_glued,
            above_build: x_pct / 100.0 > self.build_stretch,
            ms: started.elapsed().as_secs_f64() * 1e3,
        }
    }

    /// The recorded merges with stretch <= `thr`, in order: (chart root per patch, glued per edge, Euler
    /// characteristic per root, merges used, segments glued).
    fn replay(&self, thr: f64) -> (Vec<u32>, Vec<bool>, Vec<i64>, usize, usize) {
        let t = &self.topo;
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
        let root: Vec<u32> = (0..self.n_patches as u32).map(|p| pd.find(p)).collect();
        (root, glued, chi, used, segs_glued)
    }

    /// The outline smoothing of the full tree (`smooth`), made once and kept.
    fn smoothing(&self) -> Arc<Smooth> {
        if let Some(s) = self.smooth_cache.lock().unwrap().as_ref() {
            return s.clone();
        }
        let (mut root, mut glued, chi, _, _) = self.replay(f64::INFINITY);
        let s = Arc::new(self.smooth(&mut root, &mut glued, &chi));
        *self.smooth_cache.lock().unwrap() = Some(s.clone());
        s
    }

    fn smooth_ctx(&self) -> SmoothCtx {
        let t = &self.topo;
        let np = self.n_patches;
        let mut pfaces: Vec<Vec<u32>> = vec![Vec::new(); np];
        for f in 0..t.nf {
            pfaces[self.patch[f] as usize].push(f as u32);
        }
        let mut psegs: Vec<Vec<u32>> = vec![Vec::new(); np];
        for (s, sg) in self.segments.iter().enumerate() {
            if sg.p != sg.q {
                psegs[sg.p as usize].push(s as u32);
                psegs[sg.q as usize].push(s as u32);
            }
        }
        SmoothCtx { pfaces, psegs }
    }

    /// What a cut along segment s costs in the outline smoothing: its outline weight (`outline_w`). The cut cost
    /// (seen pixels first) was measured as a gain too and changed nothing (bench/uvplan/uv_defaults.md, R6chart).
    fn gain_w(&self, s: u32) -> f64 {
        self.segments[s as usize].weight
    }

    fn seg_glued(&self, glued: &[bool], s: u32) -> bool {
        glued[self.segments[s as usize].edges[0] as usize]
    }

    fn set_seg(&self, glued: &mut [bool], s: u32, on: bool, undo: &mut Vec<(u32, bool)>) {
        if self.seg_glued(glued, s) != on {
            undo.push((s, !on));
            for &e in &self.segments[s as usize].edges {
                glued[e as usize] = on;
            }
        }
    }

    /// Move patch p into chart b: its glued segments to its own chart are cut, and its segments to b glued, all of
    /// them when b and p then make one disk, else only the heaviest (`weight`, then the lowest id). The changed
    /// segments go to `undo` (with their old state). Returns (outline weight glued to b, is b with p one disk); the
    /// corner wedges of b with p are then in `sc`.
    #[allow(clippy::too_many_arguments)]
    fn move_patch(&self, ctx: &SmoothCtx, sc: &mut SmoothScratch, p: u32, b: u32, root: &mut [u32], glued: &mut [bool], members: &mut [Vec<u32>], undo: &mut Vec<(u32, bool)>) -> (f64, bool) {
        let to_b = self.shift_patch(ctx, p, b, root, glued, members, undo);
        let all: f64 = to_b.iter().map(|&s| self.gain_w(s)).sum();
        if self.is_disk(sc, &members[b as usize], &ctx.pfaces, root, glued, b) {
            return (all, true);
        }
        let Some(&best) = to_b.iter().max_by(|&&x, &&y| self.gain_w(x).total_cmp(&self.gain_w(y)).then(y.cmp(&x))) else {
            return (0.0, false);
        };
        for &s in &to_b {
            self.set_seg(glued, s, s == best, undo);
        }
        (self.gain_w(best), self.is_disk(sc, &members[b as usize], &ctx.pfaces, root, glued, b))
    }

    /// The first half of `move_patch`: p's glue to its own chart cut, all its segments to b glued, p labelled b.
    /// Returns its segments to b.
    #[allow(clippy::too_many_arguments)]
    fn shift_patch(&self, ctx: &SmoothCtx, p: u32, b: u32, root: &mut [u32], glued: &mut [bool], members: &mut [Vec<u32>], undo: &mut Vec<(u32, bool)>) -> Vec<u32> {
        let a = root[p as usize];
        let mut to_b: Vec<u32> = Vec::new();
        for &s in &ctx.psegs[p as usize] {
            let sg = &self.segments[s as usize];
            let q = if sg.p == p { sg.q } else { sg.p };
            let c = root[q as usize];
            if c == a {
                self.set_seg(glued, s, false, undo);
            } else if c == b {
                to_b.push(s);
            }
        }
        root[p as usize] = b;
        members[a as usize].retain(|&x| x != p);
        members[b as usize].push(p);
        for &s in &to_b {
            self.set_seg(glued, s, true, undo);
        }
        to_b
    }

    /// Smoother chart outlines (R6) on the full tree: a patch that would glue more outline weight (`outline_w`: length,
    /// less on a fold, more where it is seen) to a neighbouring chart than it has glued to its own chart moves to that
    /// neighbour. Every move lowers the cut's total outline weight, so the sweeps end by themselves. Spikes, saw teeth
    /// and the sticks of "lollipops" go to the chart they lie against (the boundary smoothing step of chart growing,
    /// after Sander, Wood, Gortler, Snyder, Hoppe 2003, "Multi-chart geometry images", and Julius, Kraevoy, Sheffer
    /// 2005, "D-Charts"; on our patch borders, with our gain and rules). A move is made only when both charts stay one
    /// disk, every vertex it closes inside the new chart lies flat (`SMOOTH_DEFECT_DEG`), and the patch, fitted rigidly
    /// onto the new chart's start layout along the glue, crosses none of its border sides (the one-to-one border
    /// condition, as `lies_over`). Every level replays the moves on its own charts (`compute_level`), so a level's
    /// charts lie inside the charts of every level above it.
    fn smooth(&self, root: &mut [u32], glued: &mut [bool], chi: &[i64]) -> Smooth {
        let t = &self.topo;
        let np = self.n_patches;
        let ctx = self.smooth_ctx();
        let mut members = ctx.members(root);
        let defect_max = SMOOTH_DEFECT_DEG.to_radians();
        // Corner angles summed per vertex (3D).
        let mut angle_sum = vec![0f64; t.nv];
        for f in 0..t.nf {
            let r = t.face_corners(f);
            let n = r.len();
            for (i, h) in r.clone().enumerate() {
                let v = t.flat[h] as usize;
                let (pv, nx) = (t.flat[r.start + (i + n - 1) % n] as usize, t.flat[r.start + (i + 1) % n] as usize);
                let d = |w: usize| [self.points[3 * w] - self.points[3 * v], self.points[3 * w + 1] - self.points[3 * v + 1], self.points[3 * w + 2] - self.points[3 * v + 2]];
                let (x, y) = (d(pv), d(nx));
                let dot = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
                let cr = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
                angle_sum[v] += (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt().atan2(dot);
            }
        }
        let mut sc = SmoothScratch::new(t.corners(), t.nf);
        let mut work: Vec<f64> = Vec::new();
        let mut moves: Vec<(u32, u32)> = Vec::new();
        loop {
            let before = moves.len();
            for p in 0..np as u32 {
                let a = root[p as usize];
                if chi[a as usize] != 1 {
                    continue;
                }
                let mut wa = 0.0;
                let mut wb: Vec<(u32, f64, u32)> = Vec::new(); // chart, weight, a patch of it across the heaviest segment
                for &s in &ctx.psegs[p as usize] {
                    let sg = &self.segments[s as usize];
                    let q = if sg.p == p { sg.q } else { sg.p };
                    let c = root[q as usize];
                    if c == a {
                        if self.seg_glued(glued, s) {
                            wa += self.gain_w(s);
                        }
                    } else if let Some(x) = wb.iter_mut().find(|x| x.0 == c) {
                        x.1 += self.gain_w(s);
                    } else {
                        wb.push((c, self.gain_w(s), q));
                    }
                }
                let Some(&(b, w, q)) = wb.iter().max_by(|x, y| x.1.total_cmp(&y.1).then(y.0.cmp(&x.0))) else { continue };
                if chi[b as usize] != 1 || !(w - wa > 1e-9 * (w + wa)) {
                    continue;
                }
                let mut undo: Vec<(u32, bool)> = Vec::new();
                // The patch glued along all it shares with b (no slit left between them); the cheap tests first.
                self.shift_patch(&ctx, p, b, root, glued, &mut members, &mut undo);
                let mut ok = true;
                // A vertex the move closes inside b must lie flat (no cone): its angle defect stays within the limit.
                ok = ok && undo.iter().all(|&(s, was)| {
                    was || self.segments[s as usize].edges.iter().all(|&e| {
                        [t.ea[e as usize], t.eb[e as usize]].iter().all(|&v| {
                            let v = v as usize;
                            let inside = !t.border_v[v]
                                && t.vfaces(v).iter().all(|&f| root[self.patch[f as usize] as usize] == b)
                                && t.vedges(v).iter().all(|&x| !self.cut_base[x as usize] || glued[x as usize]);
                            !inside || (std::f64::consts::TAU - angle_sum[v]).abs() <= defect_max
                        })
                    })
                });
                ok = ok && (members[a as usize].is_empty() || self.is_disk(&mut sc, &members[a as usize], &ctx.pfaces, root, glued, a));
                ok = ok && self.is_disk(&mut sc, &members[b as usize], &ctx.pfaces, root, glued, b);
                let mut old: Vec<(usize, f64, f64)> = Vec::new();
                if ok {
                    // The patch fitted rigidly onto b's start layout along the glue; its border sides must cross none of b's.
                    if work.is_empty() {
                        work = self.start_uv.clone();
                    }
                    let mut pairs = Vec::new();
                    for &s in &ctx.psegs[p as usize] {
                        let sg = &self.segments[s as usize];
                        let o = if sg.p == p { sg.q } else { sg.p };
                        if root[o as usize] != b || !self.seg_glued(glued, s) {
                            continue;
                        }
                        for &e in &sg.edges {
                            let [h0, h1] = t.euse[e as usize];
                            let (hp, hb) = if self.patch[t.cface[h0 as usize] as usize] == p { (h0, h1) } else { (h1, h0) };
                            for v in [t.ea[e as usize], t.eb[e as usize]] {
                                let (cp, cb) = (t.at(hp as usize, v), t.at(hb as usize, v));
                                pairs.push(([work[2 * cp], work[2 * cp + 1]], [work[2 * cb], work[2 * cb + 1]]));
                            }
                        }
                    }
                    let rigid = align(&pairs);
                    for &f in &ctx.pfaces[p as usize] {
                        for h in t.face_corners(f as usize) {
                            old.push((h, work[2 * h], work[2 * h + 1]));
                            let (x, y) = apply_rigid(rigid, work[2 * h], work[2 * h + 1]);
                            work[2 * h] = x;
                            work[2 * h + 1] = y;
                        }
                    }
                    // The patch's corners on the glue take b's positions, so the start is one layout; none of the
                    // patch's triangles may turn over by that.
                    let mut snap: HashMap<u32, [f64; 2], FxBuild> = HashMap::default();
                    for &s in &ctx.psegs[p as usize] {
                        let sg = &self.segments[s as usize];
                        let o = if sg.p == p { sg.q } else { sg.p };
                        if root[o as usize] != b || !self.seg_glued(glued, s) {
                            continue;
                        }
                        for &e in &sg.edges {
                            let [h0, h1] = t.euse[e as usize];
                            let hb = if self.patch[t.cface[h0 as usize] as usize] == p { h1 } else { h0 };
                            for v in [t.ea[e as usize], t.eb[e as usize]] {
                                let cb = t.at(hb as usize, v);
                                snap.insert(v, [work[2 * cb], work[2 * cb + 1]]);
                            }
                        }
                    }
                    for &f in &ctx.pfaces[p as usize] {
                        for h in t.face_corners(f as usize) {
                            if let Some(q) = snap.get(&t.flat[h]) {
                                work[2 * h] = q[0];
                                work[2 * h + 1] = q[1];
                            }
                        }
                    }
                    // Corners of p at a vertex shared with b through a corner only (no glued edge) keep the fit.
                    // (every corner turns left: a convex face, so both diagonals of a quad stay unturned; and the snap
                    // keeps each face's area within `SMOOTH_AREA` of its surface area)
                    ok = ctx.pfaces[p as usize].iter().all(|&f| {
                        let r = t.face_corners(f as usize);
                        let n = r.len();
                        let mut a2 = 0.0;
                        for i in 0..n {
                            let (m, c) = (r.start + i, r.start + (i + 1) % n);
                            a2 += work[2 * m] * work[2 * c + 1] - work[2 * c] * work[2 * m + 1];
                        }
                        let ratio = 0.5 * a2 / face_area(&self.points, t, f as usize).max(1e-300);
                        ratio <= SMOOTH_AREA && ratio * SMOOTH_AREA >= 1.0
                            && (0..n).all(|i| {
                                let (a, m, c) = (r.start + (i + n - 1) % n, r.start + i, r.start + (i + 1) % n);
                                (work[2 * m] - work[2 * a]) * (work[2 * c + 1] - work[2 * a + 1]) - (work[2 * c] - work[2 * a]) * (work[2 * m + 1] - work[2 * a + 1]) > 0.0
                            })
                    });
                    let mut sides: Vec<Side> = Vec::new();
                    for &m in &members[b as usize] {
                        for &f in &ctx.pfaces[m as usize] {
                            for h in t.face_corners(f as usize) {
                                let e = t.cedge[h] as usize;
                                let [h0, h1] = t.euse[e];
                                let o = if h as u32 == h0 { h1 } else { h0 };
                                if o != NONE && !t.wall[e] && !(self.cut_base[e] && !glued[e]) && root[self.patch[t.cface[o as usize] as usize] as usize] == b {
                                    continue;
                                }
                                let n = t.next(h);
                                let v = [sc.cfind(h as u32), sc.cfind(n as u32)];
                                sides.push(Side { p: [work[2 * h], work[2 * h + 1]], q: [work[2 * n], work[2 * n + 1]], v, class: if m == p { 2 } else { 0 } });
                            }
                        }
                    }
                    ok = ok && !sides_cross(&mut sides);
                }
                if ok {
                    moves.push((p, q));
                } else {
                    for (h, x, y) in old {
                        work[2 * h] = x;
                        work[2 * h + 1] = y;
                    }
                    for &(s, was) in undo.iter().rev() {
                        for &e in &self.segments[s as usize].edges {
                            glued[e as usize] = was;
                        }
                    }
                    root[p as usize] = a;
                    members[b as usize].retain(|&x| x != p);
                    members[a as usize].push(p);
                }
            }
            if moves.len() == before {
                break;
            }
        }
        Smooth { moves, start: work }
    }

    /// Are the faces of these patches (all with root `r`) one disk under the glue (Euler characteristic over wedges,
    /// one piece)? Leaves the wedges of these faces in `sc` (`cfind`).
    #[allow(clippy::too_many_arguments)]
    fn is_disk(&self, sc: &mut SmoothScratch, patches: &[u32], pfaces: &[Vec<u32>], root: &[u32], glued: &[bool], r: u32) -> bool {
        let t = &self.topo;
        sc.stamp += 1;
        let (mut chi, mut comps) = (0i64, 0i64);
        for &p in patches {
            let n = pfaces[p as usize].len() as i64;
            chi += n; // a face; its sides and corners cancel until corners join
            comps += n;
        }
        for &p in patches {
            for &f in &pfaces[p as usize] {
                for h in t.face_corners(f as usize) {
                    let e = t.cedge[h] as usize;
                    let [h0, h1] = t.euse[e];
                    if h as u32 != h0 || h1 == NONE || t.wall[e] || (self.cut_base[e] && !glued[e]) {
                        continue;
                    }
                    let g = t.cface[h1 as usize];
                    if root[self.patch[g as usize] as usize] != r {
                        continue;
                    }
                    chi += 1; // two sides are one edge
                    for v in [t.ea[e], t.eb[e]] {
                        chi -= i64::from(sc.cjoin(t.at(h0 as usize, v) as u32, t.at(h1 as usize, v) as u32));
                    }
                    comps -= i64::from(sc.fjoin(f, g));
                }
            }
        }
        comps == 1 && chi == 1
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
        let (uv, stats, face_str) = self.flatten_cut(&level.cut, &level.chart, None, Some(self.start_of(&level)), opts, stop);
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
    fn thin_tube_becomes_two_strip_pieces() {
        // R5: 40 rows of 6 quads around a thin tube: two pieces, each cut once along its length, every face in them.
        let (pts, ip, fl) = tube(6, 40, 0.2);
        let tree = Tree::build(&input(&pts, &ip, &fl), &never()).unwrap();
        assert_eq!(tree.numbers[num::TUBE_FACES], 240.0);
        assert_eq!((tree.n_patches, tree.numbers[num::NOT_DISK_PATCHES]), (2, 0.0));
        let l = tree.level(120.0);
        assert_eq!((l.charts, l.not_disk), (2, 0));
        // a short wide tube is no tube
        let (pts, ip, fl) = tube(16, 5, 1.0);
        assert_eq!(Tree::build(&input(&pts, &ip, &fl), &never()).unwrap().numbers[num::TUBE_FACES], 0.0);
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
            segs.push(Segment { edges, p: p.min(q), q: p.max(q), cost: 1.0, seen: 1.0, weight: 1.0, length: 1.0 });
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
