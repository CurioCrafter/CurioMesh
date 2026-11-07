// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! UV Check numbers (package K3), SPEC 9.1. Our own code. Cited ideas: Sander, Snyder, Gortler, Hoppe 2001, "Texture
//! mapping progressive meshes" (L2 stretch from the Jacobian of the 3D to UV map); closest point on a triangle is
//! textbook geometry (Ericson 2005, "Real-Time Collision Detection"); the pixel-centre fill rule with a shared-edge
//! tie break is the usual rasteriser rule (Pineda 1988, "A parallel algorithm for polygon rasterization").
//!
//! One call, `check`, scores a result mesh (and, when given, the source it came from): faces across seams, texture
//! drift, seam fit, stretch, angle, flips, overlap per page and tile, space used, pixel spread, seams in sight,
//! straight rows, islands. Definitions are the ones in `bench/uvplan/uvstats.py` (P0), so baseline and result are
//! scored alike. Quads are cut as a fan from corner 0. The NumPy twin `triadq/uvscore.py` follows the same steps.

#![allow(dead_code)] // the frozen input types are read by other kernels' packages

use crate::topo::{edges, halves, summary, Dsu, Faces, FxBuild};
use rayon::prelude::*;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

// Constants table (the NumPy twin `triadq/uvscore.py` holds the same names and values).
/// Texture size used when the caller gives none (pixels).
pub const TEXTURE_PX: f64 = 1024.0;
/// Two end UVs are the same when they differ by less than this many pixels at the texture size.
pub const UV_TOL_PX: f64 = 0.01;
/// A corner on a seam belongs to the sides whose source UV there is within this many pixels of its own UV.
pub const OWNER_UV_PX: f64 = 2.0;
/// A face counts as seen when its seen value is at least this.
pub const SEEN_MIN: f64 = 0.5;
/// Drift looks at source points within (nearest distance + max(face error, this share of the first edge)).
pub const DRIFT_EDGE_SLACK: f64 = 0.25;
/// Owner ball: eps = max(this share of the source diagonal, `EPS_EDGE` of its median edge).
pub const EPS_DIAG: f64 = 1e-4;
pub const EPS_EDGE: f64 = 0.05;
/// Positions closer than this share of the largest extent are one welded vertex.
pub const WELD: f64 = 1e-6;
/// A row is straight when its UV image stays within this many texels of one line.
pub const ROW_LINE_TEXEL: f64 = 0.25;
/// Overlap and space-used raster size ceiling (memory guard only: one page is this squared bytes).
pub const RASTER_MAX_PX: f64 = 8192.0;
/// Query grid cell edge, in median source edges (the cell grows only when the extent would pass `GRID_MAX_DIM` cells).
pub const GRID_EDGES: f64 = 2.0;
pub const GRID_MAX_DIM: f64 = 4096.0;
/// A nearest-distance probe starts at this share of the cell edge and doubles until it finds something within its radius
/// (the answer is exact either way; most corners are far closer than the cell, so a small start gathers far fewer triangles).
pub const PROBE_START: f64 = 0.0625;
/// Occupied-cell blocks: this many cells per side (4^3 = 64 cells = one 64-bit mask).
const BLOCK: i64 = 4;
/// Stretch share counted as "too stretched".
pub const STRETCH_BAD: f64 = 4.0;
/// Drift above this many pixels is counted in `drift_over8_pct`.
pub const DRIFT_BAD_PX: f64 = 8.0;
/// Below this area a triangle has no usable UV Jacobian.
pub const AREA_MIN: f64 = 1e-14;

/// Frozen cross-kernel inputs (K4's outputs). `None` means "every face and edge fully seen" (synthetic seen = 1).
pub type SeenFace<'a> = Option<&'a [f64]>;
pub type SeenEdgePx<'a> = Option<&'a [f64]>;

pub const NAMES: [&str; 29] = [
    "faces", "tris", "seen_faces_pct", "straddle_faces", "straddle_pct", "drift_p50", "drift_p99", "drift_over8_pct",
    "seam_fit_max", "seam_fit_p99", "extra_faces", "split_faces", "stretch_med", "stretch_worst1", "stretch_over4_pct",
    "stretch_worst1_seen", "angle_p90", "flipped_faces", "flipped_seen_faces", "flipped_vs_source", "overlap_pct",
    "space_used_pct", "pixel_spread", "seams_in_sight", "cut_per_sqrt_area", "straight_rows_pct", "row_bend_p95",
    "islands", "island_min_px",
];
const I_FACES: usize = 0;
const I_TRIS: usize = 1;
const I_SEEN_PCT: usize = 2;
const I_STRADDLE_N: usize = 3;
const I_STRADDLE: usize = 4;
const I_DRIFT50: usize = 5;
const I_DRIFT99: usize = 6;
const I_DRIFT8: usize = 7;
const I_FIT_MAX: usize = 8;
const I_FIT_99: usize = 9;
const I_EXTRA: usize = 10;
const I_SPLIT: usize = 11;
const I_ST_MED: usize = 12;
const I_ST_W1: usize = 13;
const I_ST_4: usize = 14;
const I_ST_W1_SEEN: usize = 15;
const I_ANGLE: usize = 16;
const I_FLIPS: usize = 17;
const I_FLIPS_SEEN: usize = 18;
const I_FLIPS_SRC: usize = 19;
const I_OVERLAP: usize = 20;
const I_SPACE: usize = 21;
const I_SPREAD: usize = 22;
const I_SIGHT: usize = 23;
const I_CUT_LEN: usize = 24;
const I_ROWS: usize = 25;
const I_BEND: usize = 26;
const I_ISLANDS: usize = 27;
const I_ISL_MIN: usize = 28;

/// A polygon mesh with per-corner UVs (2 per corner) and an optional material per face (empty = all 0).
pub struct Mesh<'a> {
    pub points: &'a [f64],
    pub faces: Faces<'a>,
    pub uv: &'a [f64],
    pub material: &'a [i64],
}

/// Optional inputs; every empty slice means "not given".
#[derive(Default)]
pub struct Extras<'a> {
    /// Source side region per source face (K1's joined islands); empty = the source's own islands.
    pub src_region: &'a [i64],
    pub seen_face: &'a [f64],
    pub seen_edge_px: &'a [f64],
    /// Result faces whose overlap is intended (mirrored or stacked on purpose): not counted as overlap.
    pub allow_overlap: &'a [u8],
    /// K5's grid column and row per result corner (-1 = none), and which result faces sit on an On chart.
    pub col: &'a [i64],
    pub row: &'a [i64],
    pub on_chart: &'a [u8],
    /// Intended pixel density factor per result face (Hidden parts weighting); empty = 1.
    pub density: &'a [f64],
    /// Faces of the same run with Seam Fit off (0 = unknown) and per-face "split" flags.
    pub faces_off: i64,
    pub split: &'a [u8],
}

pub struct Out {
    pub numbers: Vec<f64>,
    /// Per result face: 1 when it straddles a seam / is flipped (for the preview).
    pub straddle: Vec<u8>,
    pub flipped: Vec<u8>,
}

type V3 = [f64; 3];

const BARY7: [[f64; 3]; 7] = [
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [0.6, 0.2, 0.2],
    [0.2, 0.6, 0.2],
    [0.2, 0.2, 0.6],
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Closest point on triangle abc to p: barycentric (u, v, w) and distance (Ericson 2005).
fn closest_on_tri(p: V3, a: V3, b: V3, c: V3) -> ([f64; 3], f64) {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    let bary = if d1 <= 0.0 && d2 <= 0.0 {
        [1.0, 0.0, 0.0]
    } else {
        let bp = sub(p, b);
        let (d3, d4) = (dot(ab, bp), dot(ac, bp));
        if d3 >= 0.0 && d4 <= d3 {
            [0.0, 1.0, 0.0]
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                let t = d1 / (d1 - d3);
                [1.0 - t, t, 0.0]
            } else {
                let cp = sub(p, c);
                let (d5, d6) = (dot(ab, cp), dot(ac, cp));
                if d6 >= 0.0 && d5 <= d6 {
                    [0.0, 0.0, 1.0]
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                        let t = d2 / (d2 - d6);
                        [1.0 - t, 0.0, t]
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
                            let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            [0.0, 1.0 - t, t]
                        } else {
                            let den = va + vb + vc;
                            if den.abs() < 1e-300 {
                                [1.0, 0.0, 0.0]
                            } else {
                                let (v, w) = (vb / den, vc / den);
                                [1.0 - v - w, v, w]
                            }
                        }
                    }
                }
            }
        }
    };
    let q = [
        bary[0] * a[0] + bary[1] * b[0] + bary[2] * c[0],
        bary[0] * a[1] + bary[1] * b[1] + bary[2] * c[1],
        bary[0] * a[2] + bary[1] * b[2] + bary[2] * c[2],
    ];
    (bary, norm(sub(p, q)))
}

/// One triangle of the fan: welded-free point ids, polygon corner ids (index into `flat` and `uv`) and its face.
#[derive(Clone, Copy)]
struct Tri {
    c: [u32; 3],
    f: u32,
}

fn fan(m: &Mesh) -> Vec<Tri> {
    let mut out = Vec::with_capacity(m.faces.flat.len());
    for f in 0..m.faces.count() {
        let (a, b) = (m.faces.indptr[f] as u32, m.faces.indptr[f + 1] as u32);
        for j in 1..b.saturating_sub(a).saturating_sub(1) {
            out.push(Tri { c: [a, a + j, a + j + 1], f: f as u32 });
        }
    }
    out
}

fn pt(m: &Mesh, corner: u32) -> V3 {
    let v = m.faces.flat[corner as usize] as usize;
    [m.points[3 * v], m.points[3 * v + 1], m.points[3 * v + 2]]
}
fn uvc(m: &Mesh, corner: u32) -> [f64; 2] {
    [m.uv[2 * corner as usize], m.uv[2 * corner as usize + 1]]
}
fn tri_pts(m: &Mesh, t: &Tri) -> [V3; 3] {
    [pt(m, t.c[0]), pt(m, t.c[1]), pt(m, t.c[2])]
}
fn tri_uvs(m: &Mesh, t: &Tri) -> [[f64; 2]; 3] {
    [uvc(m, t.c[0]), uvc(m, t.c[1]), uvc(m, t.c[2])]
}
fn material(m: &Mesh, f: u32) -> i64 {
    m.material.get(f as usize).copied().unwrap_or(0)
}

fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if n >= 2048 {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// numpy.percentile (linear) of an ascending array.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q / 100.0 * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let t = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * t
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

/// Weighted quantile: the first value (ascending) whose cumulative weight share reaches `q`.
fn wquantile(values: &[f64], weights: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)));
    let mut cum = Vec::with_capacity(order.len());
    let mut run = 0.0;
    for &i in &order {
        run += weights[i];
        cum.push(run);
    }
    let total = run;
    let k = cum.iter().position(|&c| c / total >= q).unwrap_or(order.len() - 1);
    values[order[k.min(order.len() - 1)]]
}

// ------------------------------------------------------------------------------------------------ islands

/// Welded vertex ids: positions closer than `WELD` of the largest extent are one vertex.
fn weld(points: &[f64]) -> Vec<u32> {
    let n = points.len() / 3;
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for i in 0..n {
        for k in 0..3 {
            lo[k] = lo[k].min(points[3 * i + k]);
            hi[k] = hi[k].max(points[3 * i + k]);
        }
    }
    let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
    let q = if ext > 0.0 { ext * WELD } else { 1.0 };
    let mut ids: HashMap<[i64; 3], u32, FxBuild> = HashMap::default();
    (0..n)
        .map(|i| {
            let key = [0, 1, 2].map(|k| (points[3 * i + k] / q).round_ties_even() as i64);
            let next = ids.len() as u32;
            *ids.entry(key).or_insert(next)
        })
        .collect()
}

struct Islands {
    /// Island per triangle (0..count).
    label: Vec<u32>,
    count: usize,
    /// Cut edges: `halves[start[e]..start[e + 1]]` are (triangle, side) of the uses of cut edge e.
    cut_halves: Vec<(u32, u8)>,
    cut_start: Vec<usize>,
}

fn islands(m: &Mesh, tris: &[Tri], tex: f64) -> Islands {
    let tol = UV_TOL_PX / tex;
    let w = weld(m.points);
    let vert = |c: u32| w[m.faces.flat[c as usize] as usize];
    let mut keys: Vec<(u32, u32, u32)> = Vec::with_capacity(tris.len() * 3);
    for (t, tri) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (vert(tri.c[k]), vert(tri.c[(k + 1) % 3]));
            if a != b {
                keys.push((a.min(b), a.max(b), (3 * t + k) as u32));
            }
        }
    }
    keys.sort_unstable();
    let mut dsu = Dsu::new(tris.len());
    let near = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).abs() <= tol && (p[1] - q[1]).abs() <= tol;
    let (mut cut_halves, mut cut_start) = (Vec::new(), vec![0usize]);
    let mut i = 0;
    while i < keys.len() {
        let mut j = i + 1;
        while j < keys.len() && keys[j].0 == keys[i].0 && keys[j].1 == keys[i].1 {
            j += 1;
        }
        let group = &keys[i..j];
        i = j;
        if group.len() < 2 {
            continue;
        }
        let end_uvs = |h: u32| {
            let (t, k) = ((h / 3) as usize, (h % 3) as usize);
            let c = &tris[t].c;
            (vert(c[k]), uvc(m, c[k]), uvc(m, c[(k + 1) % 3]))
        };
        let mut all_joined = true;
        for x in 0..group.len() {
            for y in x + 1..group.len() {
                let (hx, hy) = (group[x].2, group[y].2);
                let ((ax, ux0, ux1), (ay, uy0, uy1)) = (end_uvs(hx), end_uvs(hy));
                let same_uv = if ax == ay { near(ux0, uy0) && near(ux1, uy1) } else { near(ux0, uy1) && near(ux1, uy0) };
                let same_mat = material(m, tris[(hx / 3) as usize].f) == material(m, tris[(hy / 3) as usize].f);
                if same_uv && same_mat {
                    dsu.union(hx / 3, hy / 3);
                } else {
                    all_joined = false;
                }
            }
        }
        if !all_joined {
            for g in group {
                cut_halves.push((g.2 / 3, (g.2 % 3) as u8));
            }
            cut_start.push(cut_halves.len());
        }
    }
    let mut label = vec![0u32; tris.len()];
    let mut ids: HashMap<u32, u32, FxBuild> = HashMap::default();
    for t in 0..tris.len() {
        let root = dsu.find(t as u32);
        let next = ids.len() as u32;
        label[t] = *ids.entry(root).or_insert(next);
    }
    Islands { label, count: ids.len(), cut_halves, cut_start }
}

// ------------------------------------------------------------------------------------------------ query grid

struct Grid {
    lo: V3,
    h: f64,
    dim: [i64; 3],
    cells: HashMap<u64, (u32, u32), FxBuild>,
    /// Occupied cells in blocks of `BLOCK`^3: block id -> one bit per cell (a ball skips empty space a block at a time).
    blocks: HashMap<u64, u64, FxBuild>,
    bdim: [i64; 3],
    items: Vec<u32>,
    tlo: Vec<[i32; 3]>,
    /// Box and corners of each triangle side by side (one cache line pair per candidate test).
    rec: Vec<([f64; 6], [V3; 3])>,
}

impl Grid {
    fn new(a: Vec<V3>, b: Vec<V3>, c: Vec<V3>) -> Grid {
        let n = a.len();
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for t in 0..n {
            for p in [a[t], b[t], c[t]] {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        if n == 0 {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
        let mut edge: Vec<f64> = (0..n).map(|t| norm(sub(b[t], a[t]))).collect();
        edge.sort_by(|x, y| x.total_cmp(y));
        let median = if n == 0 { 0.0 } else if n % 2 == 1 { edge[n / 2] } else { (edge[n / 2 - 1] + edge[n / 2]) / 2.0 };
        let h = (GRID_EDGES * median).max(ext / GRID_MAX_DIM).max(1e-12);
        let dim = [0, 1, 2].map(|k| ((hi[k] - lo[k]) / h).floor() as i64 + 1);
        let cell = |p: V3| [0, 1, 2].map(|k| (((p[k] - lo[k]) / h).floor() as i64).clamp(0, dim[k] - 1) as i32);
        let key = |c: [i32; 3]| ((c[2] as i64 * dim[1] + c[1] as i64) * dim[0] + c[0] as i64) as u64;
        let mut tlo = Vec::with_capacity(n);
        let mut pairs: Vec<(u64, u32)> = Vec::new();
        for t in 0..n {
            let (mut l, mut u) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for p in [a[t], b[t], c[t]] {
                for k in 0..3 {
                    l[k] = l[k].min(p[k]);
                    u[k] = u[k].max(p[k]);
                }
            }
            let (cl, cu) = (cell(l), cell(u));
            tlo.push(cl);
            for z in cl[2]..=cu[2] {
                for y in cl[1]..=cu[1] {
                    for x in cl[0]..=cu[0] {
                        pairs.push((key([x, y, z]), t as u32));
                    }
                }
            }
        }
        pairs.sort_unstable();
        let items: Vec<u32> = pairs.iter().map(|p| p.1).collect();
        let mut cells: HashMap<u64, (u32, u32), FxBuild> = HashMap::default();
        let mut i = 0;
        while i < pairs.len() {
            let mut j = i;
            while j < pairs.len() && pairs[j].0 == pairs[i].0 {
                j += 1;
            }
            cells.insert(pairs[i].0, (i as u32, j as u32));
            i = j;
        }
        let bdim = [0, 1, 2].map(|k| (dim[k] + BLOCK - 1) / BLOCK);
        let mut blocks: HashMap<u64, u64, FxBuild> = HashMap::default();
        for &k in cells.keys() {
            let k = k as i64;
            let (x, y, z) = (k % dim[0], (k / dim[0]) % dim[1], k / (dim[0] * dim[1]));
            let id = (((z / BLOCK) * bdim[1] + y / BLOCK) * bdim[0] + x / BLOCK) as u64;
            *blocks.entry(id).or_insert(0) |= 1u64 << (((z % BLOCK) * BLOCK + y % BLOCK) * BLOCK + x % BLOCK);
        }
        let tbox: Vec<[f64; 6]> = (0..n)
            .map(|t| {
                let mut o = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                for p in [a[t], b[t], c[t]] {
                    for k in 0..3 {
                        o[k] = o[k].min(p[k]);
                        o[k + 3] = o[k + 3].max(p[k]);
                    }
                }
                o
            })
            .collect();
        let rec = (0..n).map(|t| (tbox[t], [a[t], b[t], c[t]])).collect();
        Grid { lo, h, dim, cells, blocks, bdim, items, tlo, rec }
    }

    fn cell_of(&self, p: V3) -> [i32; 3] {
        [0, 1, 2].map(|k| (((p[k] - self.lo[k]) / self.h).floor() as i64).clamp(0, self.dim[k] - 1) as i32)
    }

    /// One candidate triangle: skipped by its box, by `pre`, or by its exact distance, else handed to `f`.
    #[inline]
    fn visit(&self, t: u32, q: V3, r: f64, pre: &mut impl FnMut(u32) -> bool, f: &mut impl FnMut(u32, [f64; 3], f64)) {
        let (bx, v) = &self.rec[t as usize];
        let gap = (0..3).map(|k| (bx[k] - q[k]).max(q[k] - bx[k + 3]).max(0.0).powi(2)).sum::<f64>();
        if gap > r * r || !pre(t) {
            return;
        }
        let (bary, d) = closest_on_tri(q, v[0], v[1], v[2]);
        if d <= r {
            f(t, bary, d);
        }
    }

    /// Every triangle whose cells meet the box `lo..hi` (each once): the shared candidate list of several nearby queries.
    fn collect(&self, lo: V3, hi: V3) -> Vec<u32> {
        let mut out = Vec::new();
        self.walk(lo, hi, |t| out.push(t));
        out
    }

    fn walk(&self, lo: V3, hi: V3, mut each: impl FnMut(u32)) {
        let (ql, qu) = (self.cell_of(lo), self.cell_of(hi));
        self.cells_in(ql, qu, |x, y, z, from, to| {
            for &t in &self.items[from as usize..to as usize] {
                let first = self.tlo[t as usize];
                if first[0].max(ql[0]) == x && first[1].max(ql[1]) == y && first[2].max(ql[2]) == z {
                    each(t);
                }
            }
        });
    }

    /// Calls `f(triangle, barycentric, distance)` for every triangle within `r` of `q` that `pre(triangle)` lets through
    /// (`pre` runs before the exact distance, so a caller that can tell a triangle does not matter saves that work).
    fn ball(&self, q: V3, r: f64, mut pre: impl FnMut(u32) -> bool, mut f: impl FnMut(u32, [f64; 3], f64)) {
        let ql = self.cell_of([q[0] - r, q[1] - r, q[2] - r]);
        let qu = self.cell_of([q[0] + r, q[1] + r, q[2] + r]);
        self.cells_in(ql, qu, |x, y, z, from, to| {
            for &t in &self.items[from as usize..to as usize] {
                let first = self.tlo[t as usize];
                // a triangle in several cells is handled in the first cell of the overlap only
                if first[0].max(ql[0]) == x && first[1].max(ql[1]) == y && first[2].max(ql[2]) == z {
                    self.visit(t, q, r, &mut pre, &mut f);
                }
            }
        });
    }

    /// Calls `each(x, y, z, from, to)` for every occupied cell in the box `ql..=qu` (empty space is skipped a block at a time).
    fn cells_in(&self, ql: [i32; 3], qu: [i32; 3], mut each: impl FnMut(i32, i32, i32, u32, u32)) {
        let (bl, bu) = ([0, 1, 2].map(|k| ql[k] as i64 / BLOCK), [0, 1, 2].map(|k| qu[k] as i64 / BLOCK));
        for bz in bl[2]..=bu[2] {
            for by in bl[1]..=bu[1] {
                for bx in bl[0]..=bu[0] {
                    let Some(&mask) = self.blocks.get(&(((bz * self.bdim[1] + by) * self.bdim[0] + bx) as u64)) else {
                        continue;
                    };
                    let mut bits = mask;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as i64;
                        bits &= bits - 1;
                        let (x, y, z) = (bx * BLOCK + bit % BLOCK, by * BLOCK + (bit / BLOCK) % BLOCK, bz * BLOCK + bit / (BLOCK * BLOCK));
                        if x < ql[0] as i64 || x > qu[0] as i64 || y < ql[1] as i64 || y > qu[1] as i64 || z < ql[2] as i64 || z > qu[2] as i64 {
                            continue;
                        }
                        let Some(&(from, to)) = self.cells.get(&(((z * self.dim[1] + y) * self.dim[0] + x) as u64)) else {
                            continue;
                        };
                        each(x as i32, y as i32, z as i32, from, to);
                    }
                }
            }
        }
    }

    fn gather(&self, q: V3, r: f64) -> Vec<Hit> {
        let mut v = Vec::new();
        self.ball(q, r, |_| true, |t, b, d| v.push((t, b, d)));
        v
    }

    /// The nearest distance (exact: rings of doubling radius), the radius that found it and every hit within that radius.
    fn probe(&self, q: V3) -> (f64, f64, Vec<Hit>) {
        let mut r = PROBE_START * self.h;
        loop {
            let hits = self.gather(q, r);
            let best = hits.iter().map(|h| h.2).fold(f64::INFINITY, f64::min);
            if best <= r || r > 1e30 {
                return (best, r, hits);
            }
            r *= 2.0;
        }
    }

    /// Every hit within `rad` of `q`, reusing the hits a probe of radius `r` already found when they cover it.
    fn upto(&self, q: V3, r: f64, hits: Vec<Hit>, rad: f64) -> Vec<Hit> {
        if rad <= r {
            hits.into_iter().filter(|h| h.2 <= rad).collect()
        } else {
            self.gather(q, rad)
        }
    }
}

type Hit = (u32, [f64; 3], f64);

// ------------------------------------------------------------------------------------------------ the check

fn face_tri_area(p: &[V3; 3]) -> f64 {
    0.5 * norm(cross(sub(p[1], p[0]), sub(p[2], p[0])))
}

fn uv_area_signed(t: &[[f64; 2]; 3]) -> f64 {
    let (e1, e2) = ([t[1][0] - t[0][0], t[1][1] - t[0][1]], [t[2][0] - t[0][0], t[2][1] - t[0][1]]);
    0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
}

/// Sander L2 squared and the two Jacobian invariants (a, b, c) of a triangle; None when the UV area is degenerate.
fn jacobian(q: &[V3; 3], t: &[[f64; 2]; 3], a2s: f64) -> Option<(f64, f64, f64, f64)> {
    let d = 2.0 * a2s;
    let ss: V3 = {
        let mut o = [0.0; 3];
        for k in 0..3 {
            o[k] = (q[0][k] * (t[1][1] - t[2][1]) + q[1][k] * (t[2][1] - t[0][1]) + q[2][k] * (t[0][1] - t[1][1])) / d;
        }
        o
    };
    let st: V3 = {
        let mut o = [0.0; 3];
        for k in 0..3 {
            o[k] = (q[0][k] * (t[2][0] - t[1][0]) + q[1][k] * (t[0][0] - t[2][0]) + q[2][k] * (t[1][0] - t[0][0])) / d;
        }
        o
    };
    let (ga, gb, gc) = (dot(ss, ss), dot(ss, st), dot(st, st));
    Some((0.5 * (ga + gc), ga, gb, gc))
}

pub fn check(res: &Mesh, src: Option<&Mesh>, ex: &Extras, texture_px: f64, stop: &AtomicBool) -> Option<Out> {
    let tex = if texture_px > 0.0 { texture_px } else { TEXTURE_PX };
    let nf = res.faces.count();
    let rt = fan(res);
    let mut num = vec![f64::NAN; NAMES.len()];
    num[I_FACES] = nf as f64;
    num[I_TRIS] = rt.len() as f64;
    let seen_of = |f: usize| ex.seen_face.is_empty() || ex.seen_face.get(f).copied().unwrap_or(1.0) >= SEEN_MIN;
    num[I_SEEN_PCT] = if nf == 0 { 100.0 } else { 100.0 * (0..nf).filter(|&f| seen_of(f)).count() as f64 / nf as f64 };
    let rq: Vec<[V3; 3]> = rt.iter().map(|t| tri_pts(res, t)).collect();
    let ru: Vec<[[f64; 2]; 3]> = rt.iter().map(|t| tri_uvs(res, t)).collect();
    let a3: Vec<f64> = rq.iter().map(face_tri_area).collect();
    let a2s: Vec<f64> = ru.iter().map(uv_area_signed).collect();
    let a2: Vec<f64> = a2s.iter().map(|x| x.abs()).collect();
    let ri = islands(res, &rt, tex);
    let mut straddle = vec![0u8; nf];
    let mut flipped_tri = vec![false; rt.len()];
    let mut flips_vs_source = false;

    // ---- source: straddle by owner sets, drift, flips
    if let Some(src) = src {
        let st = fan(src);
        let si = islands(src, &st, tex);
        let region: Vec<u32> = if ex.src_region.len() == src.faces.count() {
            st.iter().map(|t| ex.src_region[t.f as usize].max(0) as u32).collect()
        } else {
            si.label.clone()
        };
        let sq: Vec<[V3; 3]> = st.iter().map(|t| tri_pts(src, t)).collect();
        let su: Vec<[[f64; 2]; 3]> = st.iter().map(|t| tri_uvs(src, t)).collect();
        let grid = Grid::new(sq.iter().map(|t| t[0]).collect(), sq.iter().map(|t| t[1]).collect(), sq.iter().map(|t| t[2]).collect());
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for i in 0..src.points.len() / 3 {
            for k in 0..3 {
                lo[k] = lo[k].min(src.points[3 * i + k]);
                hi[k] = hi[k].max(src.points[3 * i + k]);
            }
        }
        let diag = norm([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]);
        let mut edge: Vec<f64> = sq.iter().map(|t| norm(sub(t[1], t[0]))).collect();
        edge.sort_by(|x, y| x.total_cmp(y));
        let n = edge.len();
        let median = if n == 0 { 0.0 } else if n % 2 == 1 { edge[n / 2] } else { (edge[n / 2 - 1] + edge[n / 2]) / 2.0 };
        let eps = (EPS_DIAG * diag).max(EPS_EDGE * median);
        let interp = |t: u32, b: [f64; 3]| {
            let u = &su[t as usize];
            [b[0] * u[0][0] + b[1] * u[1][0] + b[2] * u[2][0], b[0] * u[0][1] + b[1] * u[1][1] + b[2] * u[2][1]]
        };
        // owner sets per result corner
        let owners: Vec<Vec<u32>> = par_map(res.faces.flat.len(), |c| {
            let p = pt(res, c as u32);
            let (d0, r, hits) = grid.probe(p);
            // (region, source uv, tri)
            let near: Vec<(u32, [f64; 2], u32)> = grid.upto(p, r, hits, d0 + eps).into_iter().map(|(t, b, _)| (region[t as usize], interp(t, b), t)).collect();
            let mut ids: Vec<u32> = near.iter().map(|x| x.0).collect();
            ids.sort_unstable();
            ids.dedup();
            if ids.len() > 1 {
                let own = uvc(res, c as u32);
                ids = near.iter().filter(|x| ((x.1[0] - own[0]).powi(2) + (x.1[1] - own[1]).powi(2)).sqrt() * tex <= OWNER_UV_PX).map(|x| x.0).collect();
                ids.sort_unstable();
                ids.dedup();
            }
            ids
        });
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        for f in 0..nf {
            let (a, b) = (res.faces.indptr[f] as usize, res.faces.indptr[f + 1] as usize);
            let mut inter = owners[a].clone();
            for o in &owners[a + 1..b] {
                inter.retain(|x| o.contains(x));
            }
            straddle[f] = u8::from(inter.is_empty());
        }
        let nstr = straddle.iter().filter(|&&x| x == 1).count();
        num[I_STRADDLE_N] = nstr as f64;
        num[I_STRADDLE] = if nf == 0 { 0.0 } else { 100.0 * nstr as f64 / nf as f64 };
        // drift and flips: 7 samples per result triangle
        let sn: Vec<V3> = sq.iter().map(|t| cross(sub(t[1], t[0]), sub(t[2], t[0]))).collect();
        let sa: Vec<f64> = su.iter().map(uv_area_signed).collect();
        // UV box of each source triangle: every point of it has a UV inside, which bounds a match's drift from below.
        let ubox: Vec<[f64; 4]> = su.iter().map(|u| [u[0][0].min(u[1][0]).min(u[2][0]), u[0][1].min(u[1][1]).min(u[2][1]), u[0][0].max(u[1][0]).max(u[2][0]), u[0][1].max(u[1][1]).max(u[2][1])]).collect();
        let per: Vec<([f64; 7], bool)> = par_map(rt.len(), |t| {
            let (q, u) = (&rq[t], &ru[t]);
            let nr = cross(sub(q[1], q[0]), sub(q[2], q[0]));
            let sample = |j: usize| {
                let b = BARY7[j];
                let p = [0, 1, 2].map(|k| b[0] * q[0][k] + b[1] * q[1][k] + b[2] * q[2][k]);
                let uv = [b[0] * u[0][0] + b[1] * u[1][0] + b[2] * u[2][0], b[0] * u[0][1] + b[1] * u[1][1] + b[2] * u[2][1]];
                (p, uv)
            };
            let samples: Vec<(V3, [f64; 2])> = (0..7).map(sample).collect();
            let mut probes: Vec<(f64, f64, Vec<Hit>)> = samples.iter().map(|s| grid.probe(s.0)).collect();
            let d0: Vec<f64> = probes.iter().map(|x| x.0).collect();
            let face_err = d0.iter().sum::<f64>() / 7.0;
            let slack = face_err.max(DRIFT_EDGE_SLACK * norm(sub(q[1], q[0])));
            let mut drift = [f64::INFINITY; 7];
            let mut matched = false;
            // One candidate list for the seven samples of this triangle, gathered when the first one needs it.
            let mut shared: Option<Vec<u32>> = None;
            for j in 0..7 {
                let (p, uv) = samples[j];
                let (_, r, hits) = std::mem::take(&mut probes[j]);
                let rad = d0[j] + slack;
                let best = Cell::new(f64::INFINITY);
                let found = Cell::new(false);
                // Sample 0 also asks whether any source triangle in the ball has the result's winding.
                let winds = |ts: u32| {
                    let s = sign(sa[ts as usize]) * sign(dot(sn[ts as usize], nr));
                    s == sign(a2s[t]) || sa[ts as usize].abs() < AREA_MIN
                };
                let take = |ts: u32, b: [f64; 3]| {
                    let c = interp(ts, b);
                    let d = ((c[0] - uv[0]).powi(2) + (c[1] - uv[1]).powi(2)).sqrt() * tex;
                    best.set(best.get().min(d));
                    if j == 0 && winds(ts) {
                        found.set(true);
                    }
                };
                for &(ts, b, d) in &hits {
                    if d <= rad {
                        take(ts, b);
                    }
                }
                if rad > r {
                    // Only a triangle whose UV box is nearer than the best match so far (or that could still show the
                    // winding) can change the answer; the rest skip their exact distance.
                    let list = shared.get_or_insert_with(|| {
                        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
                        for k in 0..7 {
                            let reach = d0[k] + slack;
                            for c in 0..3 {
                                lo[c] = lo[c].min(samples[k].0[c] - reach);
                                hi[c] = hi[c].max(samples[k].0[c] + reach);
                            }
                        }
                        grid.collect(lo, hi)
                    });
                    let mut pre = |ts: u32| {
                        let u = &ubox[ts as usize];
                        let (dx, dy) = ((u[0] - uv[0]).max(uv[0] - u[2]).max(0.0), (u[1] - uv[1]).max(uv[1] - u[3]).max(0.0));
                        ((dx * dx + dy * dy).sqrt() * tex) * (1.0 - 1e-9) < best.get() || (j == 0 && !found.get() && winds(ts))
                    };
                    let mut hit = |ts: u32, b: [f64; 3], _: f64| take(ts, b);
                    for &ts in list.iter() {
                        grid.visit(ts, p, rad, &mut pre, &mut hit);
                    }
                }
                drift[j] = best.get();
                if j == 0 {
                    matched = found.get();
                }
            }
            (drift, !matched && a2s[t].abs() >= AREA_MIN)
        });
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let all = sorted(per.iter().flat_map(|x| x.0).collect());
        num[I_DRIFT50] = percentile(&all, 50.0);
        num[I_DRIFT99] = percentile(&all, 99.0);
        num[I_DRIFT8] = if all.is_empty() { 0.0 } else { 100.0 * all.iter().filter(|&&d| d > DRIFT_BAD_PX).count() as f64 / all.len() as f64 };
        for (t, x) in per.iter().enumerate() {
            flipped_tri[t] = x.1;
        }
        flips_vs_source = true;
        // seam fit: result seam samples against the source seam UV segments of the same material
        let seg = |m: &Mesh, tris: &[Tri], is: &Islands| {
            let mut v: Vec<(i64, [f64; 4])> = Vec::new();
            for &(t, k) in &is.cut_halves {
                let tri = &tris[t as usize];
                let (p, q) = (uvc(m, tri.c[k as usize]), uvc(m, tri.c[(k as usize + 1) % 3]));
                v.push((material(m, tri.f), [p[0], p[1], q[0], q[1]]));
            }
            v
        };
        let (ss, rs) = (seg(src, &st, &si), seg(res, &rt, &ri));
        if !ss.is_empty() && !rs.is_empty() {
            let fit = seam_fit(&ss, &rs, tex, stop)?;
            num[I_FIT_MAX] = fit.iter().cloned().fold(0.0, f64::max);
            num[I_FIT_99] = percentile(&sorted(fit), 99.0);
        } else if rs.is_empty() || ss.is_empty() {
            num[I_FIT_MAX] = 0.0;
            num[I_FIT_99] = 0.0;
        }
    }

    // ---- stretch, angle (per result chart)
    let nisl = ri.count;
    let (mut a2i, mut a3i, mut sgn) = (vec![0.0; nisl], vec![0.0; nisl], vec![0.0; nisl]);
    for t in 0..rt.len() {
        let l = ri.label[t] as usize;
        a2i[l] += a2[t];
        a3i[l] += a3[t];
        sgn[l] += a2s[t];
    }
    let (mut tcv, mut wv, mut angv, mut fv, mut seenw) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for t in 0..rt.len() {
        if !(a2[t] > AREA_MIN && a3[t] > 0.0) {
            continue;
        }
        let (l2sq, ga, gb, gc) = jacobian(&rq[t], &ru[t], a2s[t]).unwrap();
        let scale = a2i[ri.label[t] as usize] / a3i[ri.label[t] as usize].max(1e-30);
        tcv.push((l2sq * scale).sqrt());
        wv.push(a3[t]);
        let r = (((ga - gc) / 2.0).powi(2) + gb * gb).sqrt();
        let (l1, l2) = ((ga + gc) / 2.0 + r, ((ga + gc) / 2.0 - r).max(1e-30));
        angv.push((l1 / l2).sqrt());
        fv.push(t);
        seenw.push(if seen_of(rt[t].f as usize) { a3[t] } else { 0.0 });
        if src.is_none() {
            flipped_tri[t] = sign(a2s[t]) != sign(sgn[ri.label[t] as usize]);
        }
    }
    if !tcv.is_empty() {
        num[I_ST_MED] = wquantile(&tcv, &wv, 0.5);
        num[I_ST_W1] = wquantile(&tcv, &wv, 0.99);
        let total: f64 = wv.iter().sum();
        num[I_ST_4] = 100.0 * tcv.iter().zip(&wv).filter(|(c, _)| **c > STRETCH_BAD).map(|(_, w)| *w).sum::<f64>() / total;
        num[I_ANGLE] = wquantile(&angv, &wv, 0.9);
        if seenw.iter().any(|&w| w > 0.0) {
            num[I_ST_W1_SEEN] = wquantile(&tcv, &seenw, 0.99);
        }
    }
    num[I_FLIPS_SRC] = if flips_vs_source { 1.0 } else { 0.0 };
    let mut flipped = vec![0u8; nf];
    for t in 0..rt.len() {
        if flipped_tri[t] {
            flipped[rt[t].f as usize] = 1;
        }
    }
    num[I_FLIPS] = flipped.iter().filter(|&&x| x == 1).count() as f64;
    num[I_FLIPS_SEEN] = (0..nf).filter(|&f| flipped[f] == 1 && seen_of(f)).count() as f64;

    // ---- islands and pixel spread
    num[I_ISLANDS] = nisl as f64;
    let isl_px: Vec<f64> = a2i.iter().filter(|&&a| a > 0.0).map(|a| a.sqrt() * tex).collect();
    num[I_ISL_MIN] = isl_px.iter().cloned().fold(f64::INFINITY, f64::min);
    if isl_px.is_empty() {
        num[I_ISL_MIN] = f64::NAN;
    }
    {
        let (mut dsum, mut dw) = (vec![0.0; nisl], vec![0.0; nisl]);
        for t in 0..rt.len() {
            let d = ex.density.get(rt[t].f as usize).copied().unwrap_or(1.0);
            dsum[ri.label[t] as usize] += d * a3[t];
            dw[ri.label[t] as usize] += a3[t];
        }
        let (mut v, mut w) = (Vec::new(), Vec::new());
        for l in 0..nisl {
            if a3i[l] > 0.0 && a2i[l] > 0.0 && dw[l] > 0.0 {
                v.push((a2i[l] / a3i[l]).sqrt() * tex / (dsum[l] / dw[l]).max(1e-30));
                w.push(a3i[l]);
            }
        }
        if !v.is_empty() {
            num[I_SPREAD] = wquantile(&v, &w, 0.95) / wquantile(&v, &w, 0.05).max(1e-30);
        }
    }

    // ---- overlap and space used per page (material, integer tile)
    raster(res, &rt, &ru, &a3, ex, tex, &mut num);

    // ---- seams in sight, cut length
    let total_area: f64 = a3.iter().sum();
    {
        let welded = weld(res.points);
        let vert = |c: u32| welded[res.faces.flat[c as usize] as usize];
        let seg_len = |t: u32, k: u8| norm(sub(pt(res, rt[t as usize].c[k as usize]), pt(res, rt[t as usize].c[(k as usize + 1) % 3])));
        let _ = vert;
        let mut cut_len = 0.0;
        for e in 0..ri.cut_start.len() - 1 {
            let (t, k) = ri.cut_halves[ri.cut_start[e]];
            cut_len += seg_len(t, k);
        }
        if total_area > 0.0 {
            num[I_CUT_LEN] = cut_len / total_area.sqrt();
        }
        let ed = edges(&halves(&res.faces));
        if ex.seen_edge_px.len() == ed.len() && ed.len() > 0 {
            let key = |a: i64, b: i64| (a.min(b), a.max(b));
            let index: HashMap<(i64, i64), usize, FxBuild> = (0..ed.len()).map(|e| ((ed.a[e], ed.b[e]), e)).collect();
            let len3: Vec<f64> = (0..ed.len())
                .map(|e| {
                    let (a, b) = (ed.a[e] as usize, ed.b[e] as usize);
                    norm(sub([res.points[3 * a], res.points[3 * a + 1], res.points[3 * a + 2]], [res.points[3 * b], res.points[3 * b + 1], res.points[3 * b + 2]]))
                })
                .collect();
            // px per unit of length: seen pixels over the length that should be seen (edges weighted by the faces' seen values)
            let (mut px, mut want) = (0.0, 0.0);
            for e in 0..ed.len() {
                let uses = ed.uses(e);
                let frac = if ex.seen_face.is_empty() { 1.0 } else { uses.iter().map(|&h| ex.seen_face.get(h2f(res, h)).copied().unwrap_or(1.0)).sum::<f64>() / uses.len() as f64 };
                px += ex.seen_edge_px[e];
                want += len3[e] * frac;
            }
            let scale = if want > 0.0 { px / want } else { f64::NAN };
            let (mut seen_cut, mut all_cut) = (0.0, 0.0);
            for e in 0..ri.cut_start.len() - 1 {
                let (t, k) = ri.cut_halves[ri.cut_start[e]];
                let tri = &rt[t as usize];
                let (va, vb) = (res.faces.flat[tri.c[k as usize] as usize], res.faces.flat[tri.c[(k as usize + 1) % 3] as usize]);
                if let Some(&idx) = index.get(&key(va, vb)) {
                    seen_cut += ex.seen_edge_px[idx];
                    all_cut += len3[idx] * scale;
                }
            }
            if all_cut > 0.0 {
                num[I_SIGHT] = seen_cut / all_cut;
            }
        }
    }

    // ---- extra faces, split faces
    if ex.faces_off > 0 {
        num[I_EXTRA] = nf as f64 - ex.faces_off as f64;
    }
    if !ex.split.is_empty() {
        num[I_SPLIT] = ex.split.iter().filter(|&&x| x != 0).count() as f64;
    }

    // ---- straight rows
    rows(res, &rt, &ri, ex, tex, &mut num);
    let _ = (fv, summary);
    if stop.load(Ordering::Relaxed) {
        return None;
    }
    Some(Out { numbers: num, straddle, flipped })
}

fn h2f(m: &Mesh, corner_in_flat: u32) -> usize {
    // face of the polygon corner `corner_in_flat` (a position in `flat`)
    m.faces.indptr.partition_point(|&s| s <= corner_in_flat as i64) - 1
}

/// Largest distance (px) of each result seam sample to the source seam segments of the same material.
fn seam_fit(src: &[(i64, [f64; 4])], res: &[(i64, [f64; 4])], tex: f64, stop: &AtomicBool) -> Option<Vec<f64>> {
    let dist = |p: [f64; 2], s: &[f64; 4]| {
        let (d, w) = ([s[2] - s[0], s[3] - s[1]], [p[0] - s[0], p[1] - s[1]]);
        let l2 = d[0] * d[0] + d[1] * d[1];
        let t = if l2 > 0.0 { ((w[0] * d[0] + w[1] * d[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
        ((w[0] - t * d[0]).powi(2) + (w[1] - t * d[1]).powi(2)).sqrt()
    };
    // uniform grid over the source segments (cells cover segment boxes)
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for (_, s) in src {
        for k in 0..2 {
            lo[k] = lo[k].min(s[k]).min(s[k + 2]);
            hi[k] = hi[k].max(s[k]).max(s[k + 2]);
        }
    }
    let mut lens: Vec<f64> = src.iter().map(|(_, s)| ((s[2] - s[0]).powi(2) + (s[3] - s[1]).powi(2)).sqrt()).collect();
    lens.sort_by(|a, b| a.total_cmp(b));
    let med = lens[lens.len() / 2];
    let ext = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    let h = (2.0 * med).max(ext / 256.0).max(1e-9);
    let dim = [((hi[0] - lo[0]) / h).floor() as i64 + 1, ((hi[1] - lo[1]) / h).floor() as i64 + 1];
    let cell = |x: f64, k: usize| (((x - lo[k]) / h).floor() as i64).clamp(0, dim[k] - 1);
    let mut cells: HashMap<(i64, i64, i64), Vec<u32>, FxBuild> = HashMap::default();
    for (i, (m, s)) in src.iter().enumerate() {
        let (x0, x1, y0, y1) = (cell(s[0].min(s[2]), 0), cell(s[0].max(s[2]), 0), cell(s[1].min(s[3]), 1), cell(s[1].max(s[3]), 1));
        for x in x0..=x1 {
            for y in y0..=y1 {
                cells.entry((*m, x, y)).or_default().push(i as u32);
            }
        }
    }
    let mut samples: Vec<(i64, [f64; 2])> = Vec::with_capacity(res.len() * 3);
    for (m, s) in res {
        samples.push((*m, [s[0], s[1]]));
        samples.push((*m, [(s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0]));
        samples.push((*m, [s[2], s[3]]));
    }
    let out = par_map(samples.len(), |i| {
        let (m, p) = samples[i];
        let (cx, cy) = (cell(p[0], 0), cell(p[1], 1));
        let mut r = 1i64;
        loop {
            let mut best = f64::INFINITY;
            for x in (cx - r).max(0)..=(cx + r).min(dim[0] - 1) {
                for y in (cy - r).max(0)..=(cy + r).min(dim[1] - 1) {
                    if let Some(list) = cells.get(&(m, x, y)) {
                        for &s in list {
                            best = best.min(dist(p, &src[s as usize].1));
                        }
                    }
                }
            }
            // everything within (r) cells of the sample's own cell has been looked at; a farther segment is at least r*h away
            // (the sample may lie outside the grid: then the true distance can only be larger, so grow until covered)
            if best <= r as f64 * h || (cx - r <= 0 && cy - r <= 0 && cx + r >= dim[0] - 1 && cy + r >= dim[1] - 1) {
                return best * tex;
            }
            r *= 2;
        }
    });
    if stop.load(Ordering::Relaxed) {
        return None;
    }
    // a material with no source seam at all: nothing to compare to, report 0 for those samples
    Some(out.into_iter().map(|d| if d.is_finite() { d } else { 0.0 }).collect())
}

// ------------------------------------------------------------------------------------------------ raster

fn raster(res: &Mesh, rt: &[Tri], ru: &[[[f64; 2]; 3]], a3: &[f64], ex: &Extras, tex: f64, num: &mut [f64]) {
    let size = tex.round().clamp(16.0, RASTER_MAX_PX) as usize;
    // triangles of each page (material, tile): (triangle, intended overlap)
    let mut pages: HashMap<(i64, i64, i64), (Vec<(usize, bool)>, f64), FxBuild> = HashMap::default();
    for (t, tri) in rt.iter().enumerate() {
        let u = &ru[t];
        let (cu, cv) = ((u[0][0] + u[1][0] + u[2][0]) / 3.0, (u[0][1] + u[1][1] + u[2][1]) / 3.0);
        let page = pages.entry((material(res, tri.f), cu.floor() as i64, cv.floor() as i64)).or_default();
        page.0.push((t, ex.allow_overlap.get(tri.f as usize).copied().unwrap_or(0) != 0));
        page.1 += a3[t];
    }
    let (mut covered, mut over, mut used, mut weight) = (0.0, 0.0, 0.0, 0.0);
    let s = size as f64;
    for (key, (list, area)) in &pages {
        let mut buf = vec![0u8; size * size];
        let band = 32usize;
        // bin the triangles by the bands of rows they touch, then fill the bands in parallel
        let mut bins: Vec<Vec<(usize, bool)>> = vec![Vec::new(); size.div_ceil(band)];
        let shaped: Vec<[[f64; 2]; 3]> = list.iter().map(|&(t, _)| [0, 1, 2].map(|k| [(ru[t][k][0] - key.1 as f64) * s, (ru[t][k][1] - key.2 as f64) * s])).collect();
        for (i, p) in shaped.iter().enumerate() {
            let (y0, y1) = (p[0][1].min(p[1][1]).min(p[2][1]), p[0][1].max(p[1][1]).max(p[2][1]));
            if !(y0.is_finite() && y1.is_finite()) {
                continue;
            }
            let (r0, r1) = (((y0 - 0.5).floor().max(0.0)) as usize / band, (((y1 + 0.5).ceil().min(s)).max(0.0)) as usize);
            for b in r0..=(r1.saturating_sub(1) / band).min(bins.len() - 1) {
                bins[b].push((i, list[i].1));
            }
        }
        buf.par_chunks_mut(size * band).enumerate().for_each(|(b, chunk)| {
            for &(i, flag) in &bins[b] {
                fill(chunk, size, b * band, shaped[i], flag);
            }
        });
        let n = buf.iter().filter(|&&b| b & 3 > 0).count();
        let o = buf.iter().filter(|&&b| b & 3 >= 2).count();
        let any = buf.iter().filter(|&&b| b != 0).count();
        covered += n as f64;
        over += o as f64;
        used += *area * any as f64 / (size * size) as f64;
        weight += *area;
    }
    if !pages.is_empty() {
        num[I_OVERLAP] = if covered > 0.0 { 100.0 * over / covered } else { 0.0 };
        num[I_SPACE] = if weight > 0.0 { 100.0 * used / weight } else { 0.0 };
    }
}

/// Fill the texels whose centre lies in triangle `p` (pixel units). Shared edges give each centre to exactly one
/// triangle: the edge function is evaluated on the endpoints in a fixed order, and a centre exactly on an edge goes
/// to the triangle on one chosen side. `flag` marks intended-overlap texels instead of counting them.
fn fill(buf: &mut [u8], size: usize, row0: usize, mut p: [[f64; 2]; 3], flag: bool) {
    let rows = buf.len() / size;
    let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
    if area == 0.0 || !area.is_finite() {
        return;
    }
    if area < 0.0 {
        p.swap(1, 2);
    }
    let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for q in &p {
        x0 = x0.min(q[0]);
        x1 = x1.max(q[0]);
        y0 = y0.min(q[1]);
        y1 = y1.max(q[1]);
    }
    let lim = size as f64;
    let (ix0, ix1) = ((x0 - 0.5).floor().max(0.0) as usize, ((x1 + 0.5).ceil().min(lim)) as usize);
    let (iy0, iy1) = ((y0 - 0.5).floor().max(0.0) as usize, ((y1 + 0.5).ceil().min(lim)) as usize);
    // edge value of point c against directed edge a->b, evaluated on the endpoints in sorted order so the two triangles
    // sharing the edge get exactly opposite values
    let edge = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        let (lo, hi, s) = if (a[0], a[1]) <= (b[0], b[1]) { (a, b, 1.0) } else { (b, a, -1.0) };
        s * ((hi[0] - lo[0]) * (c[1] - lo[1]) - (hi[1] - lo[1]) * (c[0] - lo[0]))
    };
    let owns = |a: [f64; 2], b: [f64; 2]| b[1] > a[1] || (b[1] == a[1] && b[0] < a[0]);
    for y in iy0.max(row0)..iy1.min(row0 + rows) {
        for x in ix0..ix1 {
            let c = [x as f64 + 0.5, y as f64 + 0.5];
            let mut inside = true;
            for k in 0..3 {
                let (a, b) = (p[k], p[(k + 1) % 3]);
                let e = edge(a, b, c);
                if !(e > 0.0 || (e == 0.0 && owns(a, b))) {
                    inside = false;
                    break;
                }
            }
            if inside {
                let cell = &mut buf[(y - row0) * size + x];
                if flag {
                    *cell |= 4;
                } else if *cell & 3 < 2 {
                    *cell += 1;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ rows

fn rows(res: &Mesh, rt: &[Tri], ri: &Islands, ex: &Extras, tex: f64, num: &mut [f64]) {
    let corners = res.faces.flat.len();
    if ex.row.len() != corners || ex.col.len() != corners {
        return;
    }
    let mut face_isl = vec![0u32; res.faces.count()];
    for (t, tri) in rt.iter().enumerate() {
        face_isl[tri.f as usize] = ri.label[t];
    }
    // (island, row) -> points by column
    let mut groups: HashMap<(u32, i64), Vec<(i64, [f64; 2], bool)>, FxBuild> = HashMap::default();
    for f in 0..res.faces.count() {
        for c in res.faces.indptr[f] as usize..res.faces.indptr[f + 1] as usize {
            if ex.row[c] >= 0 && ex.col[c] >= 0 {
                let on = ex.on_chart.get(f).copied().unwrap_or(0) != 0;
                groups.entry((face_isl[f], ex.row[c])).or_default().push((ex.col[c], uvc(res, c as u32), on));
            }
        }
    }
    let mut keys: Vec<_> = groups.keys().cloned().collect();
    keys.sort_unstable();
    let (mut on_rows, mut on_straight, mut bends) = (0usize, 0usize, Vec::new());
    for k in keys {
        let mut pts = groups[&k].clone();
        pts.sort_by(|a, b| a.0.cmp(&b.0));
        pts.dedup_by_key(|p| p.0);
        if pts.len() < 3 {
            continue;
        }
        let on = pts[0].2;
        let (a, b) = (pts[0].1, pts[pts.len() - 1].1);
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        let off = pts.iter().map(|p| {
            let w = [p.1[0] - a[0], p.1[1] - a[1]];
            if len > 0.0 { (w[0] * d[1] - w[1] * d[0]).abs() / len } else { (w[0] * w[0] + w[1] * w[1]).sqrt() }
        });
        let worst = off.fold(0.0, f64::max) * tex;
        if on {
            on_rows += 1;
            on_straight += usize::from(worst <= ROW_LINE_TEXEL);
        } else {
            let mut bend = 0.0f64;
            for w in pts.windows(3) {
                let (u, v) = ([w[1].1[0] - w[0].1[0], w[1].1[1] - w[0].1[1]], [w[2].1[0] - w[1].1[0], w[2].1[1] - w[1].1[1]]);
                let (lu, lv) = ((u[0] * u[0] + u[1] * u[1]).sqrt(), (v[0] * v[0] + v[1] * v[1]).sqrt());
                if lu > 0.0 && lv > 0.0 {
                    let c = ((u[0] * v[0] + u[1] * v[1]) / (lu * lv)).clamp(-1.0, 1.0);
                    bend = bend.max(c.acos().to_degrees());
                }
            }
            bends.push(bend);
        }
    }
    if on_rows > 0 {
        num[I_ROWS] = 100.0 * on_straight as f64 / on_rows as f64;
    }
    if !bends.is_empty() {
        num[I_BEND] = percentile(&sorted(bends), 95.0);
    }
}

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let flat = [0_i64, 1, 2, 3, 1, 4, 5, 2];
    let indptr = [0_i64, 4, 8];
    usize::from(summary(&Faces { indptr: &indptr, flat: &flat }).edges == 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// n x n grid of unit quads with UV = xy / n, optionally cut into two islands at column n/2 (the right half shifted).
    fn plane(n: usize, split: bool) -> (Vec<f64>, Vec<i64>, Vec<i64>, Vec<f64>) {
        let mut pts = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                pts.extend([i as f64, j as f64, 0.0]);
            }
        }
        let (mut flat, mut ptr, mut uv) = (Vec::new(), vec![0i64], Vec::new());
        for j in 0..n {
            for i in 0..n {
                let v = |a: usize, b: usize| (b * (n + 1) + a) as i64;
                flat.extend([v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)]);
                ptr.push(flat.len() as i64);
                for (a, b) in [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)] {
                    let shift = if split && i >= n / 2 { 0.05 } else { 0.0 };
                    uv.extend([a as f64 / n as f64 * 0.9 + shift, b as f64 / n as f64 * 0.9]);
                }
            }
        }
        (pts, ptr, flat, uv)
    }

    fn mesh<'a>(p: &'a [f64], ptr: &'a [i64], flat: &'a [i64], uv: &'a [f64]) -> Mesh<'a> {
        Mesh { points: p, faces: Faces { indptr: ptr, flat }, uv, material: &[] }
    }

    #[test]
    fn clean_plane_has_no_straddle_and_no_drift() {
        let (p, ptr, flat, uv) = plane(8, true);
        let m = mesh(&p, &ptr, &flat, &uv);
        let out = check(&m, Some(&m), &Extras::default(), 1024.0, &AtomicBool::new(false)).unwrap();
        // the source against itself: every face sits inside one island, no drift
        assert_eq!(out.numbers[I_STRADDLE_N], 0.0);
        assert!(out.numbers[I_DRIFT99] < 1e-6);
        assert_eq!(out.numbers[I_FLIPS], 0.0);
        assert_eq!(out.numbers[I_ISLANDS], 2.0);
    }

    #[test]
    fn faces_across_a_seam_straddle() {
        let (p, ptr, flat, uv) = plane(8, true);
        let src = mesh(&p, &ptr, &flat, &uv);
        // result: the same plane in one island (a smooth UV): faces touching the old seam have corners on both sides
        let (_, _, _, uv2) = plane(8, false);
        let res = mesh(&p, &ptr, &flat, &uv2);
        let out = check(&res, Some(&src), &Extras::default(), 1024.0, &AtomicBool::new(false)).unwrap();
        assert!(out.numbers[I_STRADDLE_N] >= 8.0, "{}", out.numbers[I_STRADDLE_N]);
        assert!(out.numbers[I_DRIFT99] > 10.0);
    }

    #[test]
    fn overlap_and_space() {
        let (p, ptr, flat, uv) = plane(4, false);
        let mut stacked = uv.clone();
        for c in stacked.chunks_mut(2) {
            c[0] = 0.45 - (c[0] - 0.45).abs(); // fold the right half onto the left: overlap
        }
        let m = mesh(&p, &ptr, &flat, &stacked);
        let out = check(&m, None, &Extras::default(), 256.0, &AtomicBool::new(false)).unwrap();
        assert!(out.numbers[I_OVERLAP] > 5.0);
        let m = mesh(&p, &ptr, &flat, &uv);
        let out = check(&m, None, &Extras::default(), 256.0, &AtomicBool::new(false)).unwrap();
        assert_eq!(out.numbers[I_OVERLAP], 0.0);
        assert!((out.numbers[I_SPACE] - 81.0).abs() < 1.0, "{}", out.numbers[I_SPACE]);
    }

    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }
}
