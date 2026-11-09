// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Seams out of sight (R7): once the charts of a level are made, every border between two charts slides onto nearby
//! edges where it shows less, by giving faces from one of the two charts to the other. Our own code, written from:
//!
//! - Where a seam goes between two fixed regions: the cheapest set of edges that separates them is a minimum cut of
//!   the face graph (graph cuts for region borders: Boykov, Jolly 2001, "Interactive graph cuts for optimal boundary
//!   and region segmentation"; seams placed by a minimum cut where they show least: Kwatra, Schödl, Essa, Turk, Bobick
//!   2003, "Graphcut textures"). Maximum flow by blocking flows on a layered graph (Dinic 1970); the cut is the side
//!   the source still reaches in the residual graph (max-flow min-cut theorem, Ford, Fulkerson 1956), which is the
//!   same for every maximum flow, so the kernel and its NumPy twin agree.
//! - What an edge costs: the merge tree's cut cost, `seen_px + length_px / L` (seams out of sight) or `length_px`
//!   (visibility-weighted seams: Sheffer, Hart 2002, "Seamster"; the measured terms are the tree's).
//! - Charts stay disks: Euler characteristic over corner wedges and one connected piece (as the merge tree checks).
//! - Charts keep their stretch: both new charts are laid out (exact grid or ARAP, `uvflat`) and kept only when neither
//!   is more stretched than the worse of the two before, or than `HIDE_STRETCH` (merge tree's per-face measure).
//! - Ours: the band of rings around the border that may change sides (widest first, narrower when the widest breaks a
//!   rule), the pairs worked in rounds (all pairs measured against one state, the best gains kept that share no
//!   chart), and the replay at every level: a face that changed sides follows the chart of its anchor (the nearest
//!   face of its new chart that stayed), so the charts of a lower level stay inside those of a higher one.
//!
//! One call per level (`apply`, from `Tree::level`); the faces' moves are made once per tree on its full level and kept.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::topo::{Dsu, Faces, FxBuild};
use crate::uvchart::{num, Level, Tree, FREE_FACES};
use crate::uvflat;

// Constants table (the NumPy twin `triadq/uvhide.py` holds the same names and values).
/// Rings of faces on each side of a border that may change sides, widest first; a narrower band is tried when the
/// widest one's cut leaves a chart that is not a disk or too stretched.
pub const RINGS: [usize; 6] = [32, 16, 8, 4, 2, 1];
/// Edge costs become whole numbers for the cut (`round(cost * CAP_SCALE) + 1`: exact sums in both twins, and among
/// equal costs the cut with fewer edges).
pub const CAP_SCALE: f64 = 1024.0;
/// A move may leave a chart this stretched by the UV Check's measure (area-weighted L2, median and worst 1 %; Sander,
/// Snyder, Gortler, Hoppe 2001) even when both charts were less; else no more than the more stretched of the two.
pub const L2_MEDIAN: f64 = 1.08;
pub const L2_WORST: f64 = 1.3;
/// The whole mesh may have this share of its surface area above `L2_WORST` before moves stop adding to it (so the
/// mesh's worst 1 % stays near `L2_WORST`); a move that lowers it is always fine.
pub const OVER_SHARE: f64 = 0.005;
/// Rings of vertices around the faces that changed sides that ARAP lays out again (the rest of the chart held).
pub const FREE_RINGS: usize = 2;
/// A move may let the two charts' smallest boxes grow by this share in area and in their longer side (page they take).
pub const PAGE_SLACK: f64 = 0.1;
pub const FULL_STEPS: usize = 10;
/// Every fan triangle of a kept layout has at least this share of its surface area (so the flatten's next small step
/// cannot turn one over).
pub const FAN_MARGIN: f64 = 0.01;
/// Trees whose moves are kept (most recent first).
pub const KEEP_TREES: usize = 8;

const NONE: u32 = u32::MAX;

/// The moves of one tree, made on its full level.
pub struct Hide {
    /// Per face: the face whose chart it follows (it changed sides), or `NONE`.
    pub anchor: Vec<u32>,
    /// Per face: its chart changed (its start layout is `start`).
    pub changed: Vec<bool>,
    /// Start layout per corner for the faces of changed charts (NaN elsewhere).
    pub start: Vec<f64>,
    /// Per edge: cut inside a chart so that it stays one disk (a slit the moves added).
    pub slit: Vec<bool>,
    pub moved: usize,
}

type Slot = Arc<OnceLock<Arc<Hide>>>;
static KEPT: Mutex<Vec<(usize, [u64; 4], Slot)>> = Mutex::new(Vec::new());

/// The moves of `tree`, made on its first level and kept.
// ponytail: kept by the tree's address plus a fingerprint (faces, merges, build time, start length) because the tree
// itself is not ours to extend; a new tree at a freed address with the same build time to the bit would reuse them.
pub fn of(tree: &Tree) -> Arc<Hide> {
    slot(tree).get_or_init(|| Arc::new(make(tree))).clone()
}

/// The moves of `tree` are made (a level with `hide` then costs only their replay; R8late).
pub fn ready(tree: &Tree) -> bool {
    slot(tree).get().is_some()
}

fn slot(tree: &Tree) -> Slot {
    let key = tree as *const Tree as usize;
    let print = [tree.topo.nf as u64, tree.merges.len() as u64, tree.numbers[num::BUILD_MS].to_bits(), tree.start_uv.len() as u64];
    {
        let mut kept = KEPT.lock().unwrap();
        match kept.iter().position(|(k, p, _)| *k == key && *p == print) {
            Some(i) => kept[i].2.clone(),
            None => {
                let s: Slot = Arc::new(OnceLock::new());
                kept.insert(0, (key, print, s.clone()));
                kept.truncate(KEEP_TREES);
                s
            }
        }
    }
}

/// Slide the borders of a level's charts (the one call from `Tree::level`).
pub fn apply(tree: &Tree, level: &mut Level) {
    let h = of(tree);
    if h.moved == 0 {
        return;
    }
    let t = &tree.topo;
    let raw = &level.chart;
    let lab: Vec<u32> = (0..t.nf).map(|f| if h.anchor[f] == NONE { raw[f] } else { raw[h.anchor[f] as usize] }).collect();
    let mut cut = level.cut.clone();
    for e in 0..t.ne() {
        if let Some([f0, f1]) = inner(tree, e) {
            let slit = level.cut[e] && raw[f0 as usize] == raw[f1 as usize];
            cut[e] = tree.fixed[e] || slit || h.slit[e] || lab[f0 as usize] != lab[f1 as usize];
        }
    }
    let (mut chart, mut n) = pieces(tree, &cut);
    let mut chi = chi_all(tree, &cut, &chart, n);
    if chi.iter().any(|&c| c != 1) {
        // below the level the moves were made on, a chart can close round a hole: there it keeps this level's own
        // borders as well (still inside the charts above)
        for e in 0..t.ne() {
            if let Some([f0, f1]) = inner(tree, e) {
                let bad = chi[chart[f0 as usize] as usize] != 1 || chi[chart[f1 as usize] as usize] != 1;
                if bad && raw[f0 as usize] != raw[f1 as usize] {
                    cut[e] = true;
                }
            }
        }
        (chart, n) = pieces(tree, &cut);
        chi = chi_all(tree, &cut, &chart, n);
    }
    let not_disk = chi.iter().filter(|&&c| c != 1).count();
    let mut start = tree.start_of(level).to_vec();
    for f in 0..t.nf {
        if h.changed[f] {
            for c in t.face_corners(f) {
                start[2 * c] = h.start[2 * c];
                start[2 * c + 1] = h.start[2 * c + 1];
            }
        }
    }
    level.chart = chart;
    level.cut = cut;
    level.start = start;
    level.charts = n;
    level.not_disk = not_disk;
}

/// Charts as the pieces the cut leaves: (chart per face numbered by first face, count).
fn pieces(tree: &Tree, cut: &[bool]) -> (Vec<u32>, usize) {
    let t = &tree.topo;
    let mut d = Dsu::new(t.nf);
    for e in 0..t.ne() {
        if let Some([f0, f1]) = inner(tree, e) {
            if !cut[e] {
                d.union(f0, f1);
            }
        }
    }
    let mut id: HashMap<u32, u32, FxBuild> = HashMap::default();
    let chart: Vec<u32> = (0..t.nf as u32).map(|f| {
        let r = d.find(f);
        let k = id.len() as u32;
        *id.entry(r).or_insert(k)
    }).collect();
    (chart, id.len())
}

/// The two faces of a manifold edge that may be glued, else None (border, non-manifold, wall).
fn inner(tree: &Tree, e: usize) -> Option<[u32; 2]> {
    let t = &tree.topo;
    let [h0, h1] = t.euse[e];
    if h1 == NONE || t.wall[e] {
        return None;
    }
    Some([t.cface[h0 as usize], t.cface[h1 as usize]])
}

/// The corner at vertex `v` of side `h` (which runs between `v` and the next corner's vertex).
fn corner_at(tree: &Tree, h: usize, v: u32) -> usize {
    let t = &tree.topo;
    if t.flat[h] == v {
        return h;
    }
    let r = t.face_corners(t.cface[h] as usize);
    if h + 1 == r.end { r.start } else { h + 1 }
}

/// Euler characteristic per chart over corner wedges (corners joined across every edge that is not cut).
fn chi_all(tree: &Tree, cut: &[bool], chart: &[u32], n: usize) -> Vec<i64> {
    let t = &tree.topo;
    let mut d = Dsu::new(t.corners());
    let mut chi = vec![0i64; n];
    for e in 0..t.ne() {
        if let Some(_) = inner(tree, e) {
            if !cut[e] {
                let [h0, h1] = t.euse[e];
                for v in [t.ea[e], t.eb[e]] {
                    d.union(corner_at(tree, h0 as usize, v) as u32, corner_at(tree, h1 as usize, v) as u32);
                }
                chi[chart[t.cface[h0 as usize] as usize] as usize] += 1; // a glued edge is one edge, not two sides
            }
        }
    }
    for f in 0..t.nf {
        let c = chart[f] as usize;
        chi[c] += 1;
        for h in t.face_corners(f) {
            chi[c] -= 1;
            if d.find(h as u32) == h as u32 {
                chi[c] += 1;
            }
        }
    }
    chi
}

// ---------------------------------------------------------------------------------------------------------------
// The moves on the full level
// ---------------------------------------------------------------------------------------------------------------

struct Mesh<'a> {
    tree: &'a Tree,
    /// Per face: its inner edges (CSR).
    fptr: Vec<usize>,
    fedge: Vec<u32>,
    /// Per edge: whole-number cut cost (0 when the edge stays cut whatever happens: forced, wall, a slit).
    w: Vec<i64>,
    free: Vec<bool>,
}

impl Mesh<'_> {
    fn other(&self, e: u32, f: u32) -> u32 {
        let [a, b] = inner(self.tree, e as usize).unwrap();
        if a == f { b } else { a }
    }
    fn edges(&self, f: u32) -> &[u32] {
        &self.fedge[self.fptr[f as usize]..self.fptr[f as usize + 1]]
    }
}

/// A chart laid out on its own: its corners, the local vertex of each and the layout per local vertex.
struct Laid {
    corners: Vec<u32>,
    vert: Vec<u32>,
    uv: Vec<f64>,
    /// Stretch by the UV Check's measure: (median, worst 1 %, surface area above `L2_WORST`).
    q: [f64; 3],
    /// Smallest-area box around the layout: (area, longer side).
    bx: (f64, f64),
    /// The chart as its own mesh (points, indptr, flat), its border sides, and whether `uv` is its exact grid.
    mesh: (Vec<f64>, Vec<i64>, Vec<i64>),
    sides: Vec<(u32, u32)>,
    grid: bool,
    /// Inner edges cut so the chart is one disk, and what they cost.
    slit: Vec<u32>,
    slit_cost: i64,
}

impl Laid {
    fn new(corners: Vec<u32>, vert: Vec<u32>, uv: Vec<f64>, q: [f64; 3], mesh: (Vec<f64>, Vec<i64>, Vec<i64>), sides: Vec<(u32, u32)>, grid: bool) -> Laid {
        let bx = min_box(&uv);
        Laid { corners, vert, uv, q, bx, mesh, sides, grid, slit: Vec::new(), slit_cost: 0 }
    }

    /// Lay the chart out the way the level's flatten will (from this layout: ARAP with soft straight rows, or without
    /// them when they break Stretch allowed, until it settles) and keep that layout when it is still one-to-one and
    /// within `limit`; false when it is not.
    fn confirm(&mut self, allowed: f64, limit: [f64; 2]) -> bool {
        if self.grid {
            return true;
        }
        let ch = uvflat::Chart { points: &self.mesh.0, faces: Faces { indptr: &self.mesh.1, flat: &self.mesh.2 } };
        let never = AtomicBool::new(false);
        let opts = uvflat::FlatOpts { rows: uvflat::Rows::Auto, start: uvflat::Start::Given, stretch_allowed: allowed, ..Default::default() };
        let Ok(out) = uvflat::flatten_chart(&ch, Some(&self.uv), None, &opts, &never) else { return false };
        let uv = out.uv;
        if !uv.iter().all(|x| x.is_finite()) || uvflat::count_flips(&ch.faces, &uv) != 0 || !fans_ok(&ch, &uv, FAN_MARGIN) { return false; }
        if border_crosses(&uv, &self.sides) {
            return false;
        }
        let q = l2(&ch, &uv);
        if !within(q, limit) {
            return false;
        }
        self.bx = min_box(&uv);
        self.uv = uv;
        self.q = q;
        true
    }
}

/// Surface area of a face (its fan).
fn face_area(tree: &Tree, f: usize) -> f64 {
    let t = &tree.topo;
    let r = t.face_corners(f);
    let x = |h: usize| { let v = t.flat[h] as usize; [tree.points[3 * v], tree.points[3 * v + 1], tree.points[3 * v + 2]] };
    let x0 = x(r.start);
    let mut a = 0.0;
    for h in r.start + 1..r.end.saturating_sub(1) {
        let (x1, x2) = (x(h), x(h + 1));
        let (e1, e2) = ([x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]], [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]]);
        let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        a += 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    }
    a
}

/// Smallest-area box around points (x, y pairs) by rotating calipers on their convex hull (Toussaint 1983; hull by
/// Andrew's monotone chain, 1979): (area, longer side).
fn min_box(uv: &[f64]) -> (f64, f64) {
    let mut p: Vec<[f64; 2]> = uv.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return (0.0, 0.0);
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut h: Vec<[f64; 2]> = Vec::new();
    for pass in 0..2 {
        let start = h.len();
        let it: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &x in it {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], x) <= 0.0 {
                h.pop();
            }
            h.push(x);
        }
        h.pop();
    }
    let mut best = (f64::INFINITY, 0.0);
    for i in 0..h.len() {
        let (a, b) = (h[i], h[(i + 1) % h.len()]);
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        if len <= 0.0 {
            continue;
        }
        let d = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
        let (mut lo, mut hi, mut top) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
        for x in &h {
            let r = [x[0] - a[0], x[1] - a[1]];
            let s = r[0] * d[0] + r[1] * d[1];
            lo = lo.min(s);
            hi = hi.max(s);
            top = top.max((r[0] * d[1] - r[1] * d[0]).abs());
        }
        let area = (hi - lo) * top;
        if area < best.0 {
            best = (area, (hi - lo).max(top));
        }
    }
    best
}

/// A chart as its own mesh when it is one disk (else None), laid out: exact grid when it is one, ARAP (rows off) from
/// the grid or from LSCM; the lower stretch of the two.
#[allow(clippy::too_many_arguments)]
fn lay(m: &Mesh, faces: &[u32], l: u32, cur: &[f64], before: &[u32], touched: &HashSet<u32, FxBuild>, limit: [f64; 2]) -> Option<Laid> {
    let tree = m.tree;
    let t = &tree.topo;
    let mut slot: HashMap<u32, u32, FxBuild> = HashMap::default();
    let mut corners = Vec::new();
    for &f in faces {
        for h in t.face_corners(f as usize) {
            slot.insert(h as u32, corners.len() as u32);
            corners.push(h as u32);
        }
    }
    let fidx: HashMap<u32, u32, FxBuild> = faces.iter().enumerate().map(|(i, &f)| (f, i as u32)).collect();
    // the chart glued along its inner edges; a chart that is a disk with holes (several border loops) gets slits, each
    // the cheapest path of inner edges from one border loop to another (Dijkstra 1959), until one loop is left
    let mut slit: Vec<u32> = Vec::new();
    let (cd, glued, inside, vid, vert, pts) = loop {
        let mut cd = Dsu::new(corners.len());
        let mut fd = Dsu::new(faces.len());
        let mut glued = 0i64;
        let mut inside: HashSet<u32, FxBuild> = HashSet::default();
        for &f in faces {
            for &e in m.edges(f) {
                let o = m.other(e, f);
                if o > f && !m.free[e as usize] && fidx.contains_key(&o) && !slit.contains(&e) {
                    glued += 1;
                    inside.insert(e);
                    fd.union(fidx[&f], fidx[&o]);
                    let [h0, h1] = t.euse[e as usize];
                    for v in [t.ea[e as usize], t.eb[e as usize]] {
                        let a = slot[&(corner_at(tree, h0 as usize, v) as u32)];
                        let b = slot[&(corner_at(tree, h1 as usize, v) as u32)];
                        cd.union(a, b);
                    }
                }
            }
        }
        let r0 = fd.find(0);
        if (1..faces.len() as u32).any(|i| fd.find(i) != r0) {
            return None;
        }
        let mut vid: HashMap<u32, u32, FxBuild> = HashMap::default();
        let mut vert = Vec::with_capacity(corners.len());
        let mut pts = Vec::new();
        for (k, &h) in corners.iter().enumerate() {
            let r = cd.find(k as u32);
            let n = vid.len() as u32;
            let v = *vid.entry(r).or_insert_with(|| {
                let p = t.flat[h as usize] as usize;
                pts.extend_from_slice(&tree.points[3 * p..3 * p + 3]);
                n
            });
            vert.push(v);
        }
        let chi = vid.len() as i64 - (corners.len() as i64 - glued) + faces.len() as i64;
        if chi == 1 {
            break (cd, glued, inside, vid, vert, pts);
        }
        match bridge(m, faces, &slot, &vert, vid.len(), &inside, chi) {
            Some(path) => slit.extend(path),
            None => {
                return None;
            }
        }
    };
    let _ = (cd, glued);
    let slit_cost: i64 = slit.iter().map(|&e| m.w[e as usize]).sum();
    let fin = |mut l: Laid| {
        l.slit = slit.clone();
        l.slit_cost = slit_cost;
        l
    };
    let mut ip = vec![0i64];
    let mut fl = Vec::with_capacity(corners.len());
    let mut k = 0usize;
    for &f in faces {
        for _ in t.face_corners(f as usize) {
            fl.push(i64::from(vert[k]));
            k += 1;
        }
        ip.push(fl.len() as i64);
    }
    let ch = uvflat::Chart { points: &pts, faces: Faces { indptr: &ip, flat: &fl } };
    let nv = vid.len();
    // one-to-one: a flip-free disk layout is one exactly when its border does not cross itself (Floater 2003)
    let mut sides = Vec::new();
    let mut k = 0usize;
    for &f in faces {
        let r = t.face_corners(f as usize);
        let first = k;
        for h in r.clone() {
            let nk = if h + 1 == r.end { first } else { k + 1 };
            if !inside.contains(&t.cedge[h]) {
                sides.push((vert[k], vert[nk]));
            }
            k += 1;
        }
    }
    let good = |uv: &[f64]| -> Option<[f64; 3]> {
        if !uv.iter().all(|x| x.is_finite()) || uvflat::count_flips(&ch.faces, uv) != 0 || !fans_ok(&ch, uv, FAN_MARGIN) {
            return None;
        }
        if border_crosses(uv, &sides) {
            return None;
        }
        Some(l2(&ch, uv)).filter(|q| q[1].is_finite())
    };
    // a grid chart is laid out from its exact grid (rows on where they fit), as the level's flatten will
    if let Ok(g) = uvflat::grid_layout(&ch) {
        let never = AtomicBool::new(false);
        let opts = uvflat::FlatOpts { rows: uvflat::Rows::Auto, start: uvflat::Start::Given, stretch_allowed: m.tree.build_stretch, ..Default::default() };
        let uv = uvflat::flatten_chart(&ch, Some(&g.uv), Some((&g.col, &g.row)), &opts, &never).ok()?.uv;
        let q = good(&uv).filter(|&q| within(q, limit))?;
        return Some(fin(Laid::new(corners, vert, uv, q, (pts, ip, fl), sides, true)));
    }
    // the current layouts, faces that came from the other chart turned onto this one (rigid fit on the vertices they
    // share with the faces that stayed), then ARAP near what changed sides with the rest held
    let came: Vec<bool> = faces.iter().map(|&f| before[f as usize] != l).collect();
    let (mut sa, mut sb) = (vec![[0.0f64; 3]; nv], vec![[0.0f64; 3]; nv]);
    let mut k = 0usize;
    for (i, &f) in faces.iter().enumerate() {
        for h in t.face_corners(f as usize) {
            let s = if came[i] { &mut sb[vert[k] as usize] } else { &mut sa[vert[k] as usize] };
            s[0] += cur[2 * h];
            s[1] += cur[2 * h + 1];
            s[2] += 1.0;
            k += 1;
        }
    }
    let pairs: Vec<([f64; 2], [f64; 2])> = (0..nv)
        .filter(|&v| sa[v][2] > 0.0 && sb[v][2] > 0.0)
        .map(|v| ([sb[v][0] / sb[v][2], sb[v][1] / sb[v][2]], [sa[v][0] / sa[v][2], sa[v][1] / sa[v][2]]))
        .collect();
    let any_came = came.iter().any(|&c| c);
    if !any_came || pairs.len() >= 2 {
        let (c, tr) = if any_came { align(&pairs) } else { ([1.0, 0.0], [0.0, 0.0]) };
        let mut uv = vec![0.0f64; 2 * nv];
        let mut cnt = vec![0.0f64; nv];
        let mut k = 0usize;
        for (i, &f) in faces.iter().enumerate() {
            for h in t.face_corners(f as usize) {
                let mut p = [cur[2 * h], cur[2 * h + 1]];
                if came[i] {
                    p = [c[0] * p[0] - c[1] * p[1] + tr[0], c[1] * p[0] + c[0] * p[1] + tr[1]];
                }
                let v = vert[k] as usize;
                uv[2 * v] += p[0];
                uv[2 * v + 1] += p[1];
                cnt[v] += 1.0;
                k += 1;
            }
        }
        for v in 0..nv {
            uv[2 * v] /= cnt[v];
            uv[2 * v + 1] /= cnt[v];
        }
        if uv.iter().all(|x| x.is_finite()) && uvflat::count_flips(&ch.faces, &uv) == 0 {
            let seeds: Vec<u32> = (0..corners.len()).filter(|&k| touched.contains(&t.flat[corners[k] as usize])).map(|k| vert[k]).collect();
            let ok = seeds.is_empty() || {
                let free = uvflat::band(&ch.faces, nv, &seeds, FREE_RINGS);
                let never = AtomicBool::new(false);
                uvflat::quick_stretch(&ch, &mut uv, &free, 0.0, &never).is_ok()
            };
            if ok {
                match good(&uv) {
                    Some(q) if within(q, limit) => return Some(fin(Laid::new(corners, vert, uv, q, (pts.clone(), ip.clone(), fl.clone()), sides, false))),
                    _ => {}
                }
            }
        }
    }
    // else the whole chart laid out again: ARAP from LSCM (Tutte when that flips)
    let never = AtomicBool::new(false);
    let opts = uvflat::FlatOpts { rows: uvflat::Rows::Off, start: uvflat::Start::Free, steps: FULL_STEPS, ..Default::default() };
    let uv = uvflat::flatten_chart(&ch, None, None, &opts, &never).ok()?.uv;
    let q = good(&uv)?;
    Some(fin(Laid::new(corners, vert, uv, q, (pts.clone(), ip.clone(), fl.clone()), sides, false)))
}

/// For a chart glued along `inside` that is a disk with holes (genus 0, two or more border loops; `chi` = 2 - loops):
/// the cheapest path of inner edges from the border loop of the lowest vertex to another loop, through no other border
/// vertex (ties to the lower vertex); None for any other shape.
fn bridge(m: &Mesh, faces: &[u32], slot: &HashMap<u32, u32, FxBuild>, vert: &[u32], nv: usize, inside: &HashSet<u32, FxBuild>, chi: i64) -> Option<Vec<u32>> {
    let tree = m.tree;
    let t = &tree.topo;
    let mut loops = Dsu::new(nv);
    let mut on = vec![false; nv];
    let mut k = 0usize;
    for &f in faces {
        let r = t.face_corners(f as usize);
        let first = k;
        for h in r.clone() {
            let nk = if h + 1 == r.end { first } else { k + 1 };
            if !inside.contains(&t.cedge[h]) {
                loops.union(vert[k], vert[nk]);
                on[vert[k] as usize] = true;
                on[vert[nk] as usize] = true;
            }
            k += 1;
        }
    }
    let roots: HashSet<u32, FxBuild> = (0..nv as u32).filter(|&v| on[v as usize]).map(|v| loops.find(v)).collect();
    let b = roots.len() as i64;
    if b < 2 || chi != 2 - b {
        return None;
    }
    // inner edges as a vertex graph
    let mut adj: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nv];
    let mut list: Vec<u32> = inside.iter().copied().collect();
    list.sort_unstable();
    for e in list {
        let h0 = t.euse[e as usize][0] as usize;
        let va = vert[slot[&(corner_at(tree, h0, t.ea[e as usize]) as u32)] as usize];
        let vb = vert[slot[&(corner_at(tree, h0, t.eb[e as usize]) as u32)] as usize];
        adj[va as usize].push((vb, e));
        adj[vb as usize].push((va, e));
    }
    let start = (0..nv as u32).find(|&v| on[v as usize])?;
    let home = loops.find(start);
    let mut dist = vec![i64::MAX; nv];
    let mut prev: Vec<(u32, u32)> = vec![(NONE, NONE); nv];
    let mut heap = std::collections::BinaryHeap::new();
    for v in 0..nv as u32 {
        if on[v as usize] && loops.find(v) == home {
            dist[v as usize] = 0;
            heap.push(std::cmp::Reverse((0i64, v)));
        }
    }
    while let Some(std::cmp::Reverse((d, v))) = heap.pop() {
        if d > dist[v as usize] {
            continue;
        }
        if on[v as usize] && loops.find(v) != home {
            let mut path = Vec::new();
            let mut x = v;
            while prev[x as usize].0 != NONE {
                path.push(prev[x as usize].1);
                x = prev[x as usize].0;
            }
            return Some(path);
        }
        if on[v as usize] && d > 0 {
            continue;
        }
        for &(o, e) in &adj[v as usize] {
            if on[o as usize] && loops.find(o) == home {
                continue;
            }
            let nd = d + m.w[e as usize];
            if nd < dist[o as usize] {
                dist[o as usize] = nd;
                prev[o as usize] = (v, e);
                heap.push(std::cmp::Reverse((nd, o)));
            }
        }
    }
    None
}

/// Every triangle of every face's fan (the export diagonal) keeps a positive area, as the layout check after the
/// flatten asks, and at least `margin` of its surface area (UVs scaled to the chart's surface area), so that the
/// flatten's next small step cannot turn it over.
fn fans_ok(ch: &uvflat::Chart, uv: &[f64], margin: f64) -> bool {
    let faces = &ch.faces;
    let pts = ch.points;
    let mut tris: Vec<(f64, f64)> = Vec::new();
    for f in 0..faces.count() {
        let p = faces.face(f);
        let at = |i: usize| [uv[2 * p[i] as usize], uv[2 * p[i] as usize + 1]];
        let x = |i: usize| { let v = p[i] as usize; [pts[3 * v], pts[3 * v + 1], pts[3 * v + 2]] };
        for i in 1..p.len().saturating_sub(1) {
            let (a, b, c) = (at(0), at(i), at(i + 1));
            let a2 = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
            if !(a2 > 0.0) {
                return false;
            }
            let (x0, x1, x2) = (x(0), x(i), x(i + 1));
            let (e1, e2) = ([x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]], [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]]);
            let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            tris.push((a2, 0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt()));
        }
    }
    let (s2, s3): (f64, f64) = tris.iter().fold((0.0, 0.0), |(p, q), t| (p + t.0, q + t.1));
    let scale = if s3 > 0.0 { s2 / s3 } else { 0.0 };
    tris.iter().all(|&(a2, a3)| a2 >= margin * scale * a3)
}

/// Whether stretch `q` is within `limit` (both entries).
fn within(q: [f64; 3], limit: [f64; 2]) -> bool {
    q[0] <= limit[0] && q[1] <= limit[1]
}

/// The UV Check's stretch of a laid-out chart, per triangle of each face's fan: Sander, Snyder, Gortler, Hoppe 2001
/// L2 of the map from the layout to the surface (sqrt of half its squared Frobenius norm, so a triangle squeezed flat
/// in the layout reads as huge), the layout's area scaled to the surface's; area-weighted median,
/// worst 1 % and the surface area above `L2_WORST`.
fn l2(ch: &uvflat::Chart, uv: &[f64]) -> [f64; 3] {
    let (mut vals, mut w) = (Vec::new(), Vec::new());
    let (mut a3, mut a2) = (0.0, 0.0);
    let mut tris = Vec::new();
    let pts = ch.points;
    for f in 0..ch.faces.count() {
        let poly = ch.faces.face(f);
        for i in 1..poly.len().saturating_sub(1) {
            let [p0, p1, p2] = [poly[0] as usize, poly[i] as usize, poly[i + 1] as usize];
            let x = |v: usize| [pts[3 * v], pts[3 * v + 1], pts[3 * v + 2]];
            let (x0, x1, x2) = (x(p0), x(p1), x(p2));
            let e1 = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
            let e2 = [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]];
            let l1 = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
            let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let area = 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            if !(l1 > 0.0 && area > 0.0) {
                continue;
            }
            let (bx, by) = ((e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / l1, 2.0 * area / l1);
            let u = |v: usize| [uv[2 * v], uv[2 * v + 1]];
            let (u0, u1, u2) = (u(p0), u(p1), u(p2));
            let (d1, d2) = ([u1[0] - u0[0], u1[1] - u0[1]], [u2[0] - u0[0], u2[1] - u0[1]]);
            a3 += area;
            a2 += 0.5 * (d1[0] * d2[1] - d1[1] * d2[0]).abs();
            tris.push((l1, bx, by, area, d1, d2));
        }
    }
    if !(a3 > 0.0 && a2 > 0.0) {
        return [f64::INFINITY; 3];
    }
    let s = (a2 / a3).sqrt();
    for (l1, bx, by, area, d1, d2) in tris {
        // M = E D^-1: E = [e1 e2] with e1 = (l1, 0), e2 = (bx, by); D = [d1 d2]
        let det = d1[0] * d2[1] - d1[1] * d2[0];
        let v = if det > 0.0 {
            let inv = [[d2[1] / det, -d2[0] / det], [-d1[1] / det, d1[0] / det]];
            let m = [[l1 * inv[0][0] + bx * inv[1][0], l1 * inv[0][1] + bx * inv[1][1]], [by * inv[1][0], by * inv[1][1]]];
            s * ((m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[1][0] * m[1][0] + m[1][1] * m[1][1]) / 2.0).sqrt()
        } else {
            f64::INFINITY
        };
        vals.push(v);
        w.push(area);
    }
    let over: f64 = vals.iter().zip(&w).filter(|(v, _)| **v > L2_WORST).map(|(_, a)| a).sum();
    [weighted_quantile(&vals, &w, 0.5), weighted_quantile(&vals, &w, 0.99), over]
}

/// The `q` quantile of `values` weighted by `w`: the smallest value with at least that share of the weight at or below it.
fn weighted_quantile(values: &[f64], w: &[f64], q: f64) -> f64 {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let total: f64 = w.iter().sum();
    let mut acc = 0.0;
    for &i in &order {
        acc += w[i];
        if acc >= q * total {
            return values[i];
        }
    }
    order.last().map_or(f64::NAN, |&i| values[i])
}

/// Rigid 2D fit `q = R p + t` by least squares (Kabsch's 2D case): ((cos, sin), t).
fn align(pairs: &[([f64; 2], [f64; 2])]) -> ([f64; 2], [f64; 2]) {
    let k = pairs.len() as f64;
    let (mut cp, mut cq) = ([0.0; 2], [0.0; 2]);
    for (p, q) in pairs {
        for r in 0..2 {
            cp[r] += p[r] / k;
            cq[r] += q[r] / k;
        }
    }
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for (p, q) in pairs {
        let (a, b) = ([p[0] - cp[0], p[1] - cp[1]], [q[0] - cq[0], q[1] - cq[1]]);
        sxx += a[0] * b[0] + a[1] * b[1];
        sxy += a[0] * b[1] - a[1] * b[0];
    }
    let r = sxx.hypot(sxy);
    let c = if r > 0.0 { [sxx / r, sxy / r] } else { [1.0, 0.0] };
    ([c[0], c[1]], [cq[0] - (c[0] * cp[0] - c[1] * cp[1]), cq[1] - (c[1] * cp[0] + c[0] * cp[1])])
}

/// Whether two border sides that share no vertex cross (touching does not count), found through a grid of cells.
fn border_crosses(uv: &[f64], sides: &[(u32, u32)]) -> bool {
    if sides.len() < 4 {
        return false;
    }
    let p = |v: u32| [uv[2 * v as usize], uv[2 * v as usize + 1]];
    let (mut lo, mut hi, mut len) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2], 0.0);
    for &(a, b) in sides {
        let (x, y) = (p(a), p(b));
        for r in 0..2 {
            lo[r] = lo[r].min(x[r]).min(y[r]);
            hi[r] = hi[r].max(x[r]).max(y[r]);
        }
        len += ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2)).sqrt();
    }
    let cell = (len / sides.len() as f64).max(1e-12);
    let cells = |x: f64, r: usize| ((x - lo[r]) / cell).floor() as i64;
    let mut grid: HashMap<(i64, i64), Vec<u32>, FxBuild> = HashMap::default();
    for (i, &(a, b)) in sides.iter().enumerate() {
        let (x, y) = (p(a), p(b));
        for cx in cells(x[0].min(y[0]), 0)..=cells(x[0].max(y[0]), 0) {
            for cy in cells(x[1].min(y[1]), 1)..=cells(x[1].max(y[1]), 1) {
                grid.entry((cx, cy)).or_default().push(i as u32);
            }
        }
    }
    let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let _ = hi;
    for list in grid.values() {
        for (n, &i) in list.iter().enumerate() {
            let (a, b) = sides[i as usize];
            for &j in &list[n + 1..] {
                let (c, d) = sides[j as usize];
                if a == c || a == d || b == c || b == d {
                    continue;
                }
                let (pa, pb, pc, pd) = (p(a), p(b), p(c), p(d));
                let (o1, o2, o3, o4) = (orient(pa, pb, pc), orient(pa, pb, pd), orient(pc, pd, pa), orient(pc, pd, pb));
                if o1 * o2 < 0.0 && o3 * o4 < 0.0 {
                    return true;
                }
            }
        }
    }
    false
}

struct Prop {
    a: u32,
    b: u32,
    /// Band faces whose chart changes, with their new chart.
    swap: Vec<(u32, u32)>,
    la: Option<Laid>,
    lb: Option<Laid>,
    /// Surface area above `L2_WORST` the move adds (negative: takes off).
    grow: f64,
}

/// Max flow by blocking flows (Dinic 1970) on an undirected graph with whole-number capacities; returns the flow and
/// the nodes the source reaches in the residual graph.
struct Flow {
    head: Vec<u32>,
    to: Vec<u32>,
    next: Vec<u32>,
    cap: Vec<i64>,
}

impl Flow {
    fn new(n: usize) -> Flow {
        Flow { head: vec![NONE; n], to: Vec::new(), next: Vec::new(), cap: Vec::new() }
    }
    fn add(&mut self, a: u32, b: u32, c: i64) {
        for (x, y) in [(a, b), (b, a)] {
            self.to.push(y);
            self.cap.push(c);
            self.next.push(self.head[x as usize]);
            self.head[x as usize] = (self.to.len() - 1) as u32;
        }
    }
    fn levels(&self, s: u32) -> Vec<u32> {
        let mut lv = vec![NONE; self.head.len()];
        lv[s as usize] = 0;
        let mut q = VecDeque::from([s]);
        while let Some(x) = q.pop_front() {
            let mut a = self.head[x as usize];
            while a != NONE {
                let y = self.to[a as usize];
                if self.cap[a as usize] > 0 && lv[y as usize] == NONE {
                    lv[y as usize] = lv[x as usize] + 1;
                    q.push_back(y);
                }
                a = self.next[a as usize];
            }
        }
        lv
    }
    fn run(&mut self, s: u32, t: u32) -> (i64, Vec<bool>) {
        let mut total = 0i64;
        loop {
            let lv = self.levels(s);
            if lv[t as usize] == NONE {
                return (total, lv.iter().map(|&l| l != NONE).collect());
            }
            let mut it = self.head.clone();
            loop {
                // one augmenting path along the levels (iterative; arcs that lead nowhere are skipped for good)
                let mut path: Vec<u32> = Vec::new();
                let mut x = s;
                let found = loop {
                    if x == t {
                        break true;
                    }
                    let mut a = it[x as usize];
                    while a != NONE {
                        let y = self.to[a as usize];
                        if self.cap[a as usize] > 0 && lv[y as usize] == lv[x as usize] + 1 {
                            break;
                        }
                        a = self.next[a as usize];
                    }
                    it[x as usize] = a;
                    if a == NONE {
                        match path.pop() {
                            Some(back) => {
                                x = self.to[(back ^ 1) as usize];
                                it[x as usize] = self.next[it[x as usize] as usize];
                            }
                            None => break false,
                        }
                    } else {
                        path.push(a);
                        x = self.to[a as usize];
                    }
                };
                if !found {
                    break;
                }
                let push = path.iter().map(|&a| self.cap[a as usize]).min().unwrap_or(0);
                for &a in &path {
                    self.cap[a as usize] -= push;
                    self.cap[(a ^ 1) as usize] += push;
                }
                total += push;
            }
        }
    }
}

/// One way to slide the border between charts `a` and `b`: the cut of one band, cleaned.
struct Cut {
    gain: i64,
    /// Faces whose chart changes, with their new chart.
    swap: Vec<(u32, u32)>,
    fa: Vec<u32>,
    fb: Vec<u32>,
}

/// The slides of the border between charts `a` and `b` that lower the cut cost, one per band width, widest first.
fn cuts(m: &Mesh, label: &[u32], members: &[Vec<u32>], a: u32, b: u32) -> Vec<Cut> {
    let mut found: Vec<Cut> = Vec::new();
    let union: Vec<u32> = members[a as usize].iter().chain(members[b as usize].iter()).copied().collect();
    let idx: HashMap<u32, u32, FxBuild> = union.iter().enumerate().map(|(i, &f)| (f, i as u32)).collect();
    let n = union.len();
    let mut old = 0i64;
    let mut dist = vec![usize::MAX; n];
    let mut queue = VecDeque::new();
    for &f in &members[a as usize] {
        for &e in m.edges(f) {
            let o = m.other(e, f);
            if label[o as usize] == b && !m.free[e as usize] {
                old += m.w[e as usize];
                for x in [f, o] {
                    let i = idx[&x] as usize;
                    if dist[i] == usize::MAX {
                        dist[i] = 0;
                        queue.push_back(i);
                    }
                }
            }
        }
    }
    if old == 0 {
        return found;
    }
    // a small chart may also join the other one whole (the border goes away)
    for (x, y) in [(a, b), (b, a)] {
        if members[x as usize].len() <= FREE_FACES {
            let swap: Vec<(u32, u32)> = members[x as usize].iter().map(|&f| (f, y)).collect();
            let (fa, fb) = if x == a { (Vec::new(), union.clone()) } else { (union.clone(), Vec::new()) };
            found.push(Cut { gain: old, swap, fa, fb });
        }
    }
    while let Some(i) = queue.pop_front() {
        for &e in m.edges(union[i]) {
            if let Some(&j) = idx.get(&m.other(e, union[i])) {
                if dist[j as usize] == usize::MAX {
                    dist[j as usize] = dist[i] + 1;
                    queue.push_back(j as usize);
                }
            }
        }
    }
    let mut last_band = usize::MAX;
    for &r in &RINGS {
        let band: Vec<u32> = (0..n as u32).filter(|&i| dist[i as usize] < r).collect();
        if band.len() == last_band {
            continue; // the same band as a wider one
        }
        last_band = band.len();
        // node 0 = source (chart a beyond the band), 1 = sink (chart b beyond the band), 2.. = band faces
        let mut node = vec![NONE; n];
        for (k, &i) in band.iter().enumerate() {
            node[i as usize] = k as u32 + 2;
        }
        let mut g = Flow::new(band.len() + 2);
        let (mut to_s, mut to_t) = (false, false);
        for &i in &band {
            let f = union[i as usize];
            for &e in m.edges(f) {
                let o = m.other(e, f);
                let Some(&j) = idx.get(&o) else { continue };
                if m.free[e as usize] {
                    continue;
                }
                let w = m.w[e as usize];
                if node[j as usize] != NONE {
                    if f < o {
                        g.add(node[i as usize], node[j as usize], w);
                    }
                } else if label[o as usize] == a {
                    g.add(0, node[i as usize], w);
                    to_s = true;
                } else {
                    g.add(1, node[i as usize], w);
                    to_t = true;
                }
            }
        }
        if !(to_s && to_t) {
            continue; // one chart lies wholly in the band: nothing holds it
        }
        let (flow, reach) = g.run(0, 1);
        if flow >= old {
            break; // a narrower band cannot cut cheaper
        }
        let mut new: Vec<u32> = (0..n).map(|i| if node[i] != NONE { if reach[node[i] as usize] { a } else { b } } else { label[union[i] as usize] }).collect();
        // each side keeps only what joins its faces beyond the band; a piece cut off (or a hole) goes to the other side
        for c in [a, b] {
            let mut keep = vec![false; n];
            let mut q2: VecDeque<usize> = (0..n).filter(|&i| node[i] == NONE && new[i] == c).collect();
            for &i in &q2 {
                keep[i] = true;
            }
            while let Some(i) = q2.pop_front() {
                for &e in m.edges(union[i]) {
                    if m.free[e as usize] {
                        continue;
                    }
                    if let Some(&j) = idx.get(&m.other(e, union[i])) {
                        if !keep[j as usize] && new[j as usize] == c {
                            keep[j as usize] = true;
                            q2.push_back(j as usize);
                        }
                    }
                }
            }
            let o = if c == a { b } else { a };
            for i in 0..n {
                if new[i] == c && !keep[i] {
                    new[i] = o;
                }
            }
        }
        let mut cost = 0i64;
        for i in 0..n {
            if new[i] != a {
                continue;
            }
            for &e in m.edges(union[i]) {
                if let Some(&j) = idx.get(&m.other(e, union[i])) {
                    if new[j as usize] == b && !m.free[e as usize] {
                        cost += m.w[e as usize];
                    }
                }
            }
        }
        if cost >= old {
            continue;
        }
        let (mut fa, mut fb, mut swap) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let f = union[i];
            if new[i] == a { fa.push(f) } else { fb.push(f) }
            if new[i] != label[f as usize] {
                swap.push((f, new[i]));
            }
        }
        if fa.is_empty() || fb.is_empty() || swap.is_empty() || found.last().is_some_and(|x: &Cut| x.swap == swap && !x.fa.is_empty() && !x.fb.is_empty()) {
            continue;
        }
        found.push(Cut { gain: old - cost, swap, fa, fb });
    }
    found
}

/// The first of a pair's cuts whose two charts stay disks, lie flat one-to-one, keep their stretch and get no longer
/// boxes than before; None when no cut does.
#[allow(clippy::too_many_arguments)]
fn evaluate(m: &Mesh, label: &[u32], q: &[[f64; 3]], bx: &[(f64, f64)], cur: &[f64], room: f64, a: u32, b: u32, list: Vec<Cut>) -> Option<Prop> {
    let known = |x: f64| if x.is_finite() { x } else { 0.0 };
    let (qa, qb) = (q[a as usize], q[b as usize]);
    let limit = [known(qa[0]).max(known(qb[0])).max(L2_MEDIAN), known(qa[1]).max(known(qb[1])).max(L2_WORST)];
    let t = &m.tree.topo;
    for cut in list {
        let touched: HashSet<u32, FxBuild> = cut.swap.iter().flat_map(|&(f, _)| t.face_corners(f as usize).map(|h| t.flat[h])).collect();
        let side = |faces: &[u32], c: u32| -> Option<Option<Laid>> {
            if faces.is_empty() {
                return Some(None);
            }
            let mut l = lay(m, faces, c, cur, label, &touched, limit)?;
            if !within(l.q, limit) {
                return None;
            }
            if !l.confirm(m.tree.build_stretch, limit) {
                return None;
            }
            Some(Some(l))
        };
        // the smaller side first: when it breaks a rule the larger one is never laid out (both must pass, so the
        // order changes nothing else; R8late)
        let (la, lb) = if cut.fa.len() <= cut.fb.len() {
            let Some(la) = side(&cut.fa, a) else { continue };
            let Some(lb) = side(&cut.fb, b) else { continue };
            (la, lb)
        } else {
            let Some(lb) = side(&cut.fb, b) else { continue };
            let Some(la) = side(&cut.fa, a) else { continue };
            (la, lb)
        };
        // the surface area stretched past `L2_WORST` does not grow (a stretched chart that grows would weigh more in the
        // whole mesh's worst 1 %)
        let over = la.iter().chain(lb.iter()).map(|l| l.q[2]).sum::<f64>();
        let grow = if qa[2].is_finite() && qb[2].is_finite() { over - (qa[2] + qb[2]) } else { 0.0 };
        if grow > 0.0 && grow > room {
            continue;
        }
        // page: neither chart gets a longer box than the longer of the two had (when both were measured)
        let (oa, ob) = (bx[a as usize], bx[b as usize]);
        let long = la.iter().chain(lb.iter()).map(|l| l.bx.1).fold(0.0, f64::max);
        if oa.1.is_finite() && ob.1.is_finite() && long > oa.1.max(ob.1) * (1.0 + PAGE_SLACK) {
            continue;
        }
        let gain = cut.gain - la.iter().chain(lb.iter()).map(|l| l.slit_cost).sum::<i64>();
        if gain <= 0 {
            continue;
        }
        return Some(Prop { a, b, swap: cut.swap, la, lb, grow });
    }
    None
}

fn make(tree: &Tree) -> Hide {
    let t = &tree.topo;
    let raw = tree.compute_level(100.0 * tree.build_stretch);
    let nf = t.nf;
    let ne = t.ne();
    let mut fptr = vec![0usize; nf + 1];
    let mut fedge = Vec::new();
    for f in 0..nf {
        for h in t.face_corners(f) {
            let e = t.cedge[h] as usize;
            if inner(tree, e).is_some() {
                fedge.push(e as u32);
            }
        }
        fptr[f + 1] = fedge.len();
    }
    let mut free = vec![true; ne];
    let mut w = vec![0i64; ne];
    for e in 0..ne {
        if let Some([f0, f1]) = inner(tree, e) {
            let slit = raw.cut[e] && raw.chart[f0 as usize] == raw.chart[f1 as usize];
            free[e] = tree.fixed[e] || slit;
            if !free[e] {
                w[e] = (tree.cost[e] * CAP_SCALE).round() as i64 + 1;
            }
        }
    }
    let mut m = Mesh { tree, fptr, fedge, w, free };
    let mut slit = vec![false; ne];
    let mut label = raw.chart.clone();
    let nc = raw.charts;
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); nc];
    for f in 0..nf {
        members[label[f] as usize].push(f as u32);
    }
    let mut cur = tree.start_of(&raw).to_vec();
    let mut changed = vec![false; nf];
    let none: HashSet<u32, FxBuild> = HashSet::default();
    // stretch per chart before any move, laid out the same way as after one
    let first: Vec<([f64; 3], (f64, f64))> = (0..nc).into_par_iter().map(|c| {
        let unknown = ([f64::INFINITY; 3], (f64::NAN, f64::NAN));
        if members[c].is_empty() {
            return unknown;
        }
        let l = &label;
        lay(&m, &members[c], c as u32, &cur, l, &none, [f64::INFINITY; 2]).and_then(|mut x| x.confirm(tree.build_stretch, [f64::INFINITY; 2]).then_some(x)).map_or(unknown, |x| (x.q, x.bx))
    }).collect();
    let mut q: Vec<[f64; 3]> = first.iter().map(|x| x.0).collect();
    let mut bx: Vec<(f64, f64)> = first.iter().map(|x| x.1).collect();
    let mut over: f64 = q.iter().map(|x| x[2]).filter(|x| x.is_finite()).sum();
    let budget = OVER_SHARE * (0..nf).map(|f| face_area(tree, f)).sum::<f64>();
    let mut dirty = vec![true; nc];
    loop {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for e in 0..ne {
            if m.free[e] {
                continue;
            }
            if let Some([f0, f1]) = inner(tree, e) {
                let (x, y) = (label[f0 as usize], label[f1 as usize]);
                if x != y && (dirty[x as usize] || dirty[y as usize]) {
                    pairs.push((x.min(y), x.max(y)));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        if pairs.is_empty() {
            break;
        }
        // every pair's cuts, ranked by their best gain; then batches of pairs that share no chart, best first, each
        // pair laid out once: a pair whose chart changed in this round waits for the next one
        let mut ranked: Vec<(i64, u32, u32, Vec<Cut>)> = pairs
            .par_iter()
            .map(|&(a, b)| (a, b, cuts(&m, &label, &members, a, b)))
            .filter(|x| !x.2.is_empty())
            .map(|(a, b, c)| (c.iter().map(|x| x.gain).max().unwrap_or(0), a, b, c))
            .collect();
        ranked.sort_by(|x, y| y.0.cmp(&x.0).then((x.1, x.2).cmp(&(y.1, y.2))));
        dirty = vec![false; nc];
        let mut any = false;
        let mut left: Vec<Option<(u32, u32, Vec<Cut>)>> = ranked.into_iter().map(|(_, a, b, c)| Some((a, b, c))).collect();
        loop {
            let mut busy = dirty.clone();
            let mut batch = Vec::new();
            for slot in left.iter_mut() {
                let Some((a, b, _)) = slot.as_ref() else { continue };
                let (a, b) = (*a, *b);
                if dirty[a as usize] || dirty[b as usize] {
                    *slot = None; // its chart changed: next round
                    continue;
                }
                if busy[a as usize] || busy[b as usize] {
                    continue;
                }
                busy[a as usize] = true;
                busy[b as usize] = true;
                batch.push(slot.take().unwrap());
            }
            if batch.is_empty() {
                break;
            }
            let props: Vec<Option<Prop>> = batch.into_par_iter().map(|(a, b, c)| evaluate(&m, &label, &q, &bx, &cur, budget - over, a, b, c)).collect();
            for p in props.into_iter().flatten() {
                if p.grow > 0.0 && over + p.grow > budget {
                    continue; // the batch before it used the room
                }
                over += p.grow;
                dirty[p.a as usize] = true;
                dirty[p.b as usize] = true;
                any = true;
                for &(f, s) in &p.swap {
                    label[f as usize] = s;
                }
                for (c, l) in [(p.a, &p.la), (p.b, &p.lb)] {
                    let Some(l) = l else { continue };
                    for &e in &l.slit {
                        slit[e as usize] = true;
                        m.free[e as usize] = true;
                    }
                    q[c as usize] = l.q;
                    bx[c as usize] = l.bx;
                    for (k, &h) in l.corners.iter().enumerate() {
                        let v = l.vert[k] as usize;
                        cur[2 * h as usize] = l.uv[2 * v];
                        cur[2 * h as usize + 1] = l.uv[2 * v + 1];
                    }
                }
                for c in [p.a, p.b] {
                    let mut list = std::mem::take(&mut members[c as usize]);
                    list.retain(|&f| label[f as usize] == c);
                    members[c as usize] = list;
                }
                for &(f, s) in &p.swap {
                    members[s as usize].push(f);
                }
                for c in [p.a, p.b] {
                    members[c as usize].sort_unstable();
                    for &f in &members[c as usize] {
                        changed[f as usize] = true;
                    }
                }
            }
        }
        if !any {
            break;
        }
    }
    // anchors: every face that changed sides follows the nearest face of its new chart that stayed (breadth first
    // from the faces that stayed, in face order)
    let mut anchor = vec![NONE; nf];
    let mut queue = VecDeque::new();
    let mut moved = 0usize;
    for f in 0..nf {
        if label[f] == raw.chart[f] {
            anchor[f] = f as u32;
            queue.push_back(f as u32);
        } else {
            moved += 1;
        }
    }
    while let Some(f) = queue.pop_front() {
        for &e in m.edges(f) {
            if m.free[e as usize] {
                continue;
            }
            let o = m.other(e, f);
            if anchor[o as usize] == NONE && label[o as usize] == label[f as usize] {
                anchor[o as usize] = anchor[f as usize];
                queue.push_back(o);
            }
        }
    }
    for f in 0..nf {
        if anchor[f] == f as u32 {
            anchor[f] = NONE;
        }
    }
    let start: Vec<f64> = (0..2 * t.corners()).map(|i| if changed[t.cface[i / 2] as usize] { cur[i] } else { f64::NAN }).collect();
    Hide { anchor, changed, start, slit, moved }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_cuts_the_cheap_side() {
        // s - x - t with a cheap arc x-t: the source reaches x, not t
        let mut g = Flow::new(3);
        g.add(0, 2, 5);
        g.add(2, 1, 2);
        let (flow, reach) = g.run(0, 1);
        assert_eq!(flow, 2);
        assert_eq!(reach, vec![true, false, true]);
    }
}
