// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Visibility from face-id pictures (package K4): how much each face is seen, how many pixels of each edge show
//! inside the picture, and how many sit on an outline.
//!
//! Pictures are orthographic depth buffers, one per view direction, one sample at each pixel center. The first six
//! directions are the axes, then the 8 cube corners, then the 12 cube edge middles (so 14 and 26 are prefixes), then an
//! evenly spread spiral for any larger count (golden-angle spiral, Saff and Kuijlaars 1997). The pictures are kept in a
//! `Pictures` value: more views can be added later and `sight` can run again on them with no new rendering.
//!
//! Method (ours, standard depth-buffer visibility):
//! - seen face = weighted hits / weighted expected hits, expected = `area * max(0, n.d) / pixel^2`, clamped to 0..1;
//!   a face too small to be hit reliably borrows the mean of its edge neighbours, two rounds.
//! - edge pixels: for every pixel the right and the lower neighbour. Different faces at near-equal depth (the
//!   `DEPTH_PIXELS` rule of `views.py`) = inside pixels of the edge those two faces share; a depth jump or an empty
//!   neighbour = outline pixels of the nearest edge of the nearer face. A line of length l at angle a crosses
//!   `l (|cos a| + |sin a|)` such pairs, so each pair counts `1 / (|cos a| + |sin a|)` and the totals are lengths in
//!   pixels. Values are the weighted mean over the views.

#![allow(dead_code)] // the cross-kernel types are used by other kernels' packages

use crate::geom::pictures;
use crate::topo::{edges, halves, Edges, Faces, FxBuild};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

// Constants table (mirrored in triadq/uvsight.py).
pub const VIEWS_DEFAULT: usize = 6;
pub const PIXELS_MIN: usize = 512;
pub const PIXELS_PER_EDGE: f64 = 4.0;
pub const DEPTH_PIXELS: f64 = 2.0;
pub const SUBPIXEL_HITS: f64 = 1.0;
pub const SMOOTH_ROUNDS: usize = 2;
pub const UNDERSIDE_WEIGHT: f64 = 0.3;
pub const FRAME_PAD: f64 = 1.02;
pub const MAX_PICTURE_BYTES: u64 = 4_294_967_296;

/// Frozen cross-kernel output (read by K3, K6): how much each face and each edge is seen.
/// `seen_edge_px` is the inside pixel length of each edge (weighted mean over views).
pub struct Seen {
    pub seen_face: Vec<f64>,
    pub seen_edge_px: Vec<f64>,
}

impl Seen {
    /// Synthetic "everything seen once", for tests of the kernels that read it.
    pub fn all_seen(faces: &Faces) -> Seen {
        Seen { seen_face: vec![1.0; faces.count()], seen_edge_px: vec![1.0; edges(&halves(faces)).len()] }
    }
}

/// `Seen` plus the outline pixel length of each edge.
pub struct Sight {
    pub seen: Seen,
    pub outline_px: Vec<f64>,
}

const EMPTY: u32 = u32::MAX;

fn unit(v: [f64; 3]) -> [f64; 3] {
    let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if s > 0.0 {
        [v[0] / s, v[1] / s, v[2] / s]
    } else {
        v
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// The first `n` view directions (pointing from the surface toward the camera): axes +x -x +y -y +z -z, the 8 corners,
/// the 12 edge middles, then a golden-angle spiral. Any `n`, no cap.
pub fn view_dirs(n: usize) -> Vec<[f64; 3]> {
    let mut out: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    if n > 6 {
        for i in 0..8 {
            let s = |bit: usize| if (i >> bit) & 1 == 1 { -1.0 } else { 1.0 };
            out.push(unit([s(0), s(1), s(2)]));
        }
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            for sa in [1.0, -1.0] {
                for sb in [1.0, -1.0] {
                    let mut d = [0.0; 3];
                    d[a] = sa;
                    d[b] = sb;
                    out.push(unit(d));
                }
            }
        }
    }
    if n > 26 {
        let extra = n - 26;
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        for k in 0..extra {
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / extra as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let a = golden * k as f64;
            out.push([r * a.cos(), r * a.sin(), z]);
        }
    }
    out.truncate(n);
    out
}

/// View weights: all 1; with `all_sides` false the straight-up view from below (camera under the model) gets
/// `UNDERSIDE_WEIGHT`, for game assets that are never seen from beneath.
pub fn view_weights(dirs: &[[f64; 3]], all_sides: bool) -> Vec<f64> {
    dirs.iter().map(|d| if !all_sides && d[2] < -0.99 { UNDERSIDE_WEIGHT } else { 1.0 }).collect()
}

/// About `PIXELS_PER_EDGE` across the median edge, at least `PIXELS_MIN`. No upper cap here: `Pictures::new` refuses
/// only what would not fit in memory.
pub fn default_pixels(points: &[f64], faces: &Faces) -> usize {
    let (center, radius) = frame(points, faces);
    let _ = center;
    let mut lengths: Vec<f64> = Vec::new();
    for f in 0..faces.count() {
        let p = faces.face(f);
        for k in 0..p.len() {
            let (a, b) = (p[k] as usize * 3, p[(k + 1) % p.len()] as usize * 3);
            if a + 3 <= points.len() && b + 3 <= points.len() {
                let d = [points[a] - points[b], points[a + 1] - points[b + 1], points[a + 2] - points[b + 2]];
                lengths.push(dot(d, d).sqrt());
            }
        }
    }
    if lengths.is_empty() {
        return PIXELS_MIN;
    }
    let mid = lengths.len() / 2;
    let median = *lengths.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
    if median <= 0.0 {
        return PIXELS_MIN;
    }
    ((PIXELS_PER_EDGE * 2.0 * radius / median).ceil() as usize).max(PIXELS_MIN)
}

/// Bounding-box center and the radius of the sphere around it that holds every used vertex.
fn frame(points: &[f64], faces: &Faces) -> ([f64; 3], f64) {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &v in faces.flat {
        let b = v as usize * 3;
        if b + 3 <= points.len() {
            for c in 0..3 {
                lo[c] = lo[c].min(points[b + c]);
                hi[c] = hi[c].max(points[b + c]);
            }
        }
    }
    if lo[0] > hi[0] {
        return ([0.0; 3], 1.0);
    }
    let center = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let mut r2 = 0.0f64;
    for &v in faces.flat {
        let b = v as usize * 3;
        if b + 3 <= points.len() {
            let d = [points[b] - center[0], points[b + 1] - center[1], points[b + 2] - center[2]];
            r2 = r2.max(dot(d, d));
        }
    }
    (center, r2.sqrt().max(1e-12))
}

/// One view's frame: image axes `u`, `v` and the direction `d` toward the camera, right-handed.
#[derive(Clone, Copy)]
pub struct Basis {
    pub u: [f64; 3],
    pub v: [f64; 3],
    pub d: [f64; 3],
}

pub fn basis(d: [f64; 3]) -> Basis {
    let d = unit(d);
    let axis = (0..3).min_by(|&a, &b| d[a].abs().total_cmp(&d[b].abs())).unwrap();
    let mut helper = [0.0; 3];
    helper[axis] = 1.0;
    let u = unit(cross(helper, d));
    let v = cross(d, u);
    Basis { u, v, d }
}

/// Pictures of one mesh from many directions, kept between passes. `cell = iu * pixels + iv`.
pub struct Pictures {
    pub dirs: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub pixels: usize,
    pub center: [f64; 3],
    pub half: f64,
    pub points: Vec<f64>,
    pub indptr: Vec<i64>,
    pub flat: Vec<i64>,
    /// Face id (`u32::MAX` = empty) and depth (smaller = nearer) per view, `pixels^2` cells each; empty until rendered.
    pub face: Vec<Vec<u32>>,
    pub depth: Vec<Vec<f32>>,
    /// Count of views drawn since creation: stays put when `sight` runs again.
    pub renders: usize,
}

impl Pictures {
    /// Err when the pictures would not fit in memory (`MAX_PICTURE_BYTES`) or the arrays disagree.
    pub fn new(points: &[f64], indptr: &[i64], flat: &[i64], dirs: Vec<[f64; 3]>, weights: Vec<f64>, pixels: usize) -> Result<Pictures, String> {
        if dirs.len() != weights.len() || pixels == 0 || points.len() % 3 != 0 || indptr.is_empty() || indptr[0] != 0 || *indptr.last().unwrap() as usize != flat.len() {
            return Err("sight arrays are inconsistent".into());
        }
        if flat.iter().any(|&v| v < 0 || v as usize * 3 + 3 > points.len()) {
            return Err("sight faces use a missing vertex".into());
        }
        let bytes = dirs.len() as u64 * (pixels as u64) * (pixels as u64) * 8;
        if bytes > MAX_PICTURE_BYTES {
            return Err(format!("{} views at {} px need {} MB, more than fits in memory", dirs.len(), pixels, bytes >> 20));
        }
        let faces = Faces { indptr, flat };
        let (center, radius) = frame(points, &faces);
        let views = dirs.len();
        Ok(Pictures {
            dirs,
            weights,
            pixels,
            center,
            half: radius * FRAME_PAD,
            points: points.to_vec(),
            indptr: indptr.to_vec(),
            flat: flat.to_vec(),
            face: vec![Vec::new(); views],
            depth: vec![Vec::new(); views],
            renders: 0,
        })
    }

    pub fn views(&self) -> usize {
        self.dirs.len()
    }

    /// Views that are drawn so far.
    pub fn drawn(&self) -> usize {
        self.face.iter().filter(|f| !f.is_empty()).count()
    }

    /// Add views (directions and weights); the ones already drawn are kept. Err when memory would run out.
    pub fn extend(&mut self, dirs: &[[f64; 3]], weights: &[f64]) -> Result<(), String> {
        if dirs.len() != weights.len() {
            return Err("sight arrays are inconsistent".into());
        }
        let bytes = (self.dirs.len() + dirs.len()) as u64 * (self.pixels as u64) * (self.pixels as u64) * 8;
        if bytes > MAX_PICTURE_BYTES {
            return Err(format!("{} views at {} px need {} MB, more than fits in memory", self.dirs.len() + dirs.len(), self.pixels, bytes >> 20));
        }
        self.dirs.extend_from_slice(dirs);
        self.weights.extend_from_slice(weights);
        self.face.resize(self.dirs.len(), Vec::new());
        self.depth.resize(self.dirs.len(), Vec::new());
        Ok(())
    }

    /// Replace the view weights (no re-render: weights only matter when `sight` reads the pictures).
    pub fn set_weights(&mut self, weights: &[f64]) -> bool {
        if weights.len() != self.weights.len() {
            return false;
        }
        self.weights.copy_from_slice(weights);
        true
    }

    fn triangles(&self) -> (Vec<[u32; 3]>, Vec<u32>) {
        let mut tris = Vec::new();
        let mut owner = Vec::new();
        for f in 0..self.indptr.len() - 1 {
            let p = &self.flat[self.indptr[f] as usize..self.indptr[f + 1] as usize];
            for k in 1..p.len().saturating_sub(1) {
                tris.push([p[0] as u32, p[k] as u32, p[k + 1] as u32]);
                owner.push(f as u32);
            }
        }
        (tris, owner)
    }

    /// Projected vertices of one view: (pixel x, pixel y, depth) with the depth in model units.
    fn project(&self, b: &Basis) -> Vec<[f64; 3]> {
        let scale = self.pixels as f64 / (2.0 * self.half);
        self.points
            .chunks_exact(3)
            .map(|p| {
                let q = [p[0] - self.center[0], p[1] - self.center[1], p[2] - self.center[2]];
                [(dot(q, b.u) + self.half) * scale - 0.5, (dot(q, b.v) + self.half) * scale - 0.5, 2.0 * self.half - dot(q, b.d)]
            })
            .collect()
    }

    /// Draw every view that is not drawn yet. False when `stop` ended it early; what is drawn stays, so the next call
    /// carries on.
    pub fn fill(&mut self, stop: &AtomicBool) -> bool {
        let todo: Vec<usize> = (0..self.views()).filter(|&k| self.face[k].is_empty()).collect();
        if todo.is_empty() {
            return true;
        }
        let (tris, owner) = self.triangles();
        let this = &*self;
        let run = |&k: &usize| {
            let proj = this.project(&basis(this.dirs[k]));
            (k, raster(&proj, &tris, &owner, this.pixels, stop))
        };
        let done: Vec<(usize, Option<(Vec<u32>, Vec<f32>)>)> = if todo.len() > 1 && tris.len() >= 2000 { todo.par_iter().map(run).collect() } else { todo.iter().map(run).collect() };
        let mut complete = true;
        for (k, got) in done {
            match got {
                Some((f, d)) => {
                    self.face[k] = f;
                    self.depth[k] = d;
                    self.renders += 1;
                }
                None => complete = false,
            }
        }
        complete && !stop.load(Ordering::Relaxed)
    }

    /// Seen per face and edge pixels from the pictures drawn so far, with no drawing. `None` when stopped.
    pub fn sight(&self, stop: &AtomicBool) -> Option<Sight> {
        let faces = Faces { indptr: &self.indptr, flat: &self.flat };
        let nf = faces.count();
        let ed = edges(&halves(&faces));
        let views: Vec<usize> = (0..self.views()).filter(|&k| !self.face[k].is_empty()).collect();
        let pixel = 2.0 * self.half / self.pixels as f64;
        // Face areas and normals (Newell).
        let mut area = vec![0.0; nf];
        let mut normal = vec![[0.0; 3]; nf];
        for f in 0..nf {
            let p = faces.face(f);
            let mut m = [0.0; 3];
            for k in 0..p.len() {
                let (a, b) = (p[k] as usize * 3, p[(k + 1) % p.len()] as usize * 3);
                let (pa, pb) = ([self.points[a], self.points[a + 1], self.points[a + 2]], [self.points[b], self.points[b + 1], self.points[b + 2]]);
                let c = cross(pa, pb);
                m = [m[0] + c[0], m[1] + c[1], m[2] + c[2]];
            }
            let s = dot(m, m).sqrt();
            area[f] = s / 2.0;
            normal[f] = if s > 0.0 { [m[0] / s, m[1] / s, m[2] / s] } else { [0.0; 3] };
        }
        // Edge lookups: vertex pair -> edge, and face pair -> edge.
        let mut by_vertices: HashMap<u64, u32, FxBuild> = HashMap::default();
        let mut by_faces: HashMap<u64, u32, FxBuild> = HashMap::default();
        let hf = halves(&faces);
        for e in 0..ed.len() {
            by_vertices.insert(key(ed.a[e], ed.b[e]), e as u32);
            let uses = ed.uses(e);
            for i in 0..uses.len() {
                for j in i + 1..uses.len() {
                    let (fa, fb) = (hf.face[uses[i] as usize] as i64, hf.face[uses[j] as usize] as i64);
                    if fa != fb {
                        by_faces.entry(key(fa, fb)).or_insert(e as u32);
                    }
                }
            }
        }
        let ctx = EdgeCtx { by_vertices: &by_vertices, by_faces: &by_faces, ed: &ed };
        struct PerView {
            hits: Vec<u32>,
            inside: Vec<f64>,
            outline: Vec<f64>,
        }
        let one = |&k: &usize| -> Option<PerView> {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let b = basis(self.dirs[k]);
            let proj = self.project(&b);
            let (fid, dep) = (&self.face[k], &self.depth[k]);
            let mut hits = vec![0u32; nf];
            let front: Vec<bool> = (0..nf).map(|f| dot(normal[f], b.d) > 1e-9).collect();
            for &f in fid.iter() {
                if f != EMPTY && front[f as usize] {
                    hits[f as usize] += 1;
                }
            }
            let mut inside = vec![0.0; ed.len()];
            let mut outline = vec![0.0; ed.len()];
            let n = self.pixels;
            let scale = n as f64 / (2.0 * self.half);
            let tol = (DEPTH_PIXELS * pixel) as f32;
            let weight_of = |e: u32, acc: &mut Vec<f64>| {
                let (pa, pb) = (&proj[ed.a[e as usize] as usize], &proj[ed.b[e as usize] as usize]);
                let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
                let len = dx.hypot(dy);
                if len > 1e-9 {
                    acc[e as usize] += len / (dx.abs() + dy.abs());
                }
            };
            for iu in 0..n {
                for iv in 0..n {
                    let c = iu * n + iv;
                    let a = fid[c];
                    if a == EMPTY {
                        continue;
                    }
                    for (ju, jv) in [(iu + 1, iv), (iu, iv + 1)] {
                        let (nb, nd) = if ju < n && jv < n { (fid[ju * n + jv], dep[ju * n + jv]) } else { (EMPTY, 0.0) };
                        if nb == a {
                            continue;
                        }
                        if nb != EMPTY {
                            let near_equal = (dep[c] - nd).abs() <= tol;
                            if near_equal && front[a as usize] && front[nb as usize] {
                                if let Some(&e) = ctx.by_faces.get(&key(a as i64, nb as i64)) {
                                    weight_of(e, &mut inside);
                                }
                                continue;
                            }
                        }
                        // Outline of the nearer face (this pixel when the neighbour is empty or farther).
                        let (cu, cv, cd, who) = if nb == EMPTY || dep[c] <= nd { (iu, iv, dep[c], a) } else { (ju, jv, nd, nb) };
                        if let Some(e) = nearest_edge(&faces, &ctx, &proj, who as usize, [cu as f64, cv as f64, cd as f64 * scale], scale) {
                            weight_of(e, &mut outline);
                        }
                    }
                }
            }
            Some(PerView { hits, inside, outline })
        };
        let run: Vec<Option<PerView>> = if views.len() > 1 && nf >= 2000 { views.par_iter().map(one).collect() } else { views.iter().map(one).collect() };
        if run.iter().any(|r| r.is_none()) {
            return None;
        }
        let total_w: f64 = views.iter().map(|&k| self.weights[k]).sum::<f64>().max(1e-300);
        let mut got = vec![0.0; nf];
        let mut expected = vec![0.0; nf];
        let mut inside = vec![0.0; ed.len()];
        let mut outline = vec![0.0; ed.len()];
        for (r, &k) in run.iter().flatten().zip(&views) {
            let w = self.weights[k];
            let dir = basis(self.dirs[k]).d;
            for f in 0..nf {
                let facing = dot(normal[f], dir);
                if facing > 1e-9 {
                    expected[f] += w * area[f] * facing / (pixel * pixel);
                    got[f] += w * r.hits[f] as f64;
                }
            }
            for e in 0..ed.len() {
                inside[e] += w * r.inside[e] / total_w;
                outline[e] += w * r.outline[e] / total_w;
            }
        }
        let mut seen: Vec<f64> = (0..nf).map(|f| if expected[f] > 0.0 { (got[f] / expected[f]).clamp(0.0, 1.0) } else { 0.0 }).collect();
        // Faces too small to be hit reliably borrow from their edge neighbours.
        let small: Vec<bool> = (0..nf).map(|f| expected[f] < SUBPIXEL_HITS * total_w / views.len().max(1) as f64).collect();
        let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); nf];
        for e in 0..ed.len() {
            let uses = ed.uses(e);
            for i in 0..uses.len() {
                for j in 0..uses.len() {
                    let (fa, fb) = (hf.face[uses[i] as usize], hf.face[uses[j] as usize]);
                    if fa != fb && small[fa as usize] {
                        neighbours[fa as usize].push(fb);
                    }
                }
            }
        }
        for _ in 0..SMOOTH_ROUNDS {
            let before = seen.clone();
            for f in 0..nf {
                if small[f] && !neighbours[f].is_empty() {
                    seen[f] = neighbours[f].iter().map(|&g| before[g as usize]).sum::<f64>() / neighbours[f].len() as f64;
                }
            }
        }
        Some(Sight { seen: Seen { seen_face: seen, seen_edge_px: inside }, outline_px: outline })
    }
}

struct EdgeCtx<'a> {
    by_vertices: &'a HashMap<u64, u32, FxBuild>,
    by_faces: &'a HashMap<u64, u32, FxBuild>,
    ed: &'a Edges,
}

fn key(a: i64, b: i64) -> u64 {
    ((a.min(b) as u64) << 32) | a.max(b) as u64
}

/// The edge of face `f` closest to the picture point `q` (x, y, depth * scale), measured in the projected frame.
fn nearest_edge(faces: &Faces, ctx: &EdgeCtx, proj: &[[f64; 3]], f: usize, q: [f64; 3], scale: f64) -> Option<u32> {
    let p = faces.face(f);
    let mut best: Option<(f64, u32)> = None;
    for k in 0..p.len() {
        let (va, vb) = (p[k], p[(k + 1) % p.len()]);
        let Some(&e) = ctx.by_vertices.get(&key(va, vb)) else { continue };
        let (a, b) = (proj[va as usize], proj[vb as usize]);
        let (a, b) = ([a[0], a[1], a[2] * scale], [b[0], b[1], b[2] * scale]);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let aq = [q[0] - a[0], q[1] - a[1], q[2] - a[2]];
        let len2 = dot(ab, ab);
        let t = if len2 > 0.0 { (dot(aq, ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
        let r = [aq[0] - ab[0] * t, aq[1] - ab[1] * t, aq[2] - ab[2] * t];
        let d2 = dot(r, r);
        if best.map_or(true, |(bd, _)| d2 < bd) {
            best = Some((d2, e));
        }
    }
    best.map(|b| b.1)
}

/// One depth-buffer picture. `None` when stopped.
fn raster(proj: &[[f64; 3]], tris: &[[u32; 3]], owner: &[u32], pixels: usize, stop: &AtomicBool) -> Option<(Vec<u32>, Vec<f32>)> {
    let cells = pixels * pixels;
    let mut face = vec![EMPTY; cells];
    let mut depth = vec![f32::INFINITY; cells];
    let last = pixels as f64 - 1.0;
    for (id, t) in tris.iter().enumerate() {
        if id % 4096 == 0 && stop.load(Ordering::Relaxed) {
            return None;
        }
        let (p0, p1, p2) = (proj[t[0] as usize], proj[t[1] as usize], proj[t[2] as usize]);
        let area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
        if area == 0.0 || !area.is_finite() {
            continue; // edge-on: a ray along the view can't hit it
        }
        // Barycentric weights w_i = a x + b y + c (inside when all >= 0).
        let lines = [
            ((p1[1] - p2[1]) / area, (p2[0] - p1[0]) / area, (p1[0] * p2[1] - p2[0] * p1[1]) / area),
            ((p2[1] - p0[1]) / area, (p0[0] - p2[0]) / area, (p2[0] * p0[1] - p0[0] * p2[1]) / area),
        ];
        let w2 = |x: f64, y: f64| 1.0 - (lines[0].0 * x + lines[0].1 * y + lines[0].2) - (lines[1].0 * x + lines[1].1 * y + lines[1].2);
        // Rows run along the first image axis (x); solve the column span of each row from the three lines.
        // The third weight is 1 - w0 - w1, a line too: a = -(a0 + a1), b = -(b0 + b1), c = 1 - c0 - c1.
        let all = [lines[0], lines[1], (-(lines[0].0 + lines[1].0), -(lines[0].1 + lines[1].1), 1.0 - lines[0].2 - lines[1].2)];
        let (lo, hi) = (p0[0].min(p1[0]).min(p2[0]).ceil().max(0.0), p0[0].max(p1[0]).max(p2[0]).floor().min(last));
        let mut row = lo;
        while row <= hi {
            let (mut low, mut high) = (0.0f64, last);
            for &(a, b, c) in &all {
                // weight = a * row + b * col + c >= 0
                let rest = a * row + c;
                if b > 0.0 {
                    low = low.max(-rest / b - 1e-9);
                } else if b < 0.0 {
                    high = high.min(-rest / b + 1e-9);
                } else if rest < -1e-12 {
                    high = -1.0;
                }
            }
            let base = row as usize * pixels;
            let (mut col, end) = (low.ceil(), high.floor());
            while col <= end {
                let w0 = lines[0].0 * row + lines[0].1 * col + lines[0].2;
                let w1 = lines[1].0 * row + lines[1].1 * col + lines[1].2;
                let z = (w0 * p0[2] + w1 * p1[2] + w2(row, col) * p2[2]) as f32;
                let cell = base + col as usize;
                if z < depth[cell] {
                    depth[cell] = z;
                    face[cell] = owner[id];
                }
                col += 1.0;
            }
            row += 1.0;
        }
    }
    Some((face, depth))
}

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let flat = [0_i64, 1, 2, 3, 1, 4, 5, 2];
    let indptr = [0_i64, 4, 8];
    let seen = Seen::all_seen(&Faces { indptr: &indptr, flat: &flat });
    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let up = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let (hit, _, _, _) = pictures(&points, &[0, 1, 2], &up, [0.3, 0.3, 0.0], 1.0, 4);
    usize::from(seen.seen_face.len() == 2 && seen.seen_edge_px.len() == 7 && hit.len() == 6 * 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(c: f64) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut pts = Vec::new();
        for i in 0..8 {
            pts.extend([(i & 1) as f64 * c, ((i >> 1) & 1) as f64 * c, ((i >> 2) & 1) as f64 * c]);
        }
        // outward quads
        let quads: [[i64; 4]; 6] = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        let flat: Vec<i64> = quads.iter().flatten().copied().collect();
        (pts, (0..=6).map(|i| i * 4).collect(), flat)
    }

    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }

    #[test]
    fn cube_is_seen_and_reuse_renders_nothing() {
        let (p, ip, fl) = cube(1.0);
        let stop = AtomicBool::new(false);
        let mut pic = Pictures::new(&p, &ip, &fl, view_dirs(6), vec![1.0; 6], 256).unwrap();
        assert!(pic.fill(&stop));
        let s = pic.sight(&stop).unwrap();
        assert!(s.seen.seen_face.iter().all(|&v| v > 0.97), "{:?}", s.seen.seen_face);
        let drawn = pic.renders;
        assert!(pic.fill(&stop) && pic.sight(&stop).is_some());
        assert_eq!(pic.renders, drawn);
    }

    #[test]
    fn dirs_are_prefixes() {
        assert_eq!(view_dirs(14), view_dirs(26)[..14].to_vec());
        assert_eq!(view_dirs(40).len(), 40);
    }
}
