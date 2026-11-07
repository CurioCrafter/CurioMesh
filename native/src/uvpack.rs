// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Island packer (package K7). Our own code, written from the papers named at each piece:
//!
//! - Rectangles: MaxRects with best-short-side-fit (Jylänki 2010, "A thousand ways to pack the bin"); 90 degree steps
//!   only; the order, the gap rule, the page scale search and the order moves are ours.
//! - Bottom-left rule (Baker, Coffman and Rivest 1980, "Orthogonal packings in two dimensions"): the free place whose
//!   box ends lowest, then leftmost, as a second placement rule beside best short side fit; large sets of many small boxes
//!   tile 1-2 % tighter with it, small sets of few big boxes with the first. Both are tried and the better kept (ours).
//! - Gap and mip levels (our rule, derived in `gap_axis`): a mip-k texel is a block of 2^k texels; two charts are mip-clean
//!   when their touched blocks, each grown by one block (the bilinear footprint), do not meet.
//! - Irregular charts: conservative row intervals per texel row (no bitmap), dilated by the gap in both axes (Minkowski
//!   sum with a square), then a bottom-left free-interval search (Chazelle 1983, "The bottom-left bin-packing
//!   heuristic"); score: least new page height, then least wasted interval length (ours).
//! - Fill the page: the largest uniform scale at which everything fits, found by a bracket that only ever tightens
//!   (growth/shrink steps, then bisection), so the best feasible layout is kept and nothing oscillates (ours).
//! - Keep going: other orders and orientation policies, then adjacent swaps accepted only when the page then fits a
//!   larger scale; a pass that gains nothing ends it (ours).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::topo::sort;

// Constants table (the NumPy twin `triadq/uvpack.py` holds the same names and values).
/// Fill the page: first guess, boxes cover this share of the page.
pub const START_FILL: f64 = 0.5;
/// Fill the page: scale step while the bracket is not closed yet.
pub const GROW: f64 = 1.25;
/// Fill the page: stop when the bracket is narrower than this share of the scale.
pub const SCALE_TOL: f64 = 0.00001;
/// Keep going: a move or a pass counts only when the scale rises by at least this share.
pub const MIN_GAIN: f64 = 0.0001;
/// Number of box orders (long side, area, seen-weighted area, height, width).
pub const ORDERS: usize = 5;
/// Number of orientation policies (free, wide, tall, as given).
pub const ORIENTS: usize = 4;
/// Number of placement rules (0 best short side fit, 1 bottom-left).
pub const RULES: usize = 2;
/// A policy is an orientation policy plus a placement rule: `policy % ORIENTS` and `policy / ORIENTS` (so ORIENTS * RULES).
pub const POLICIES: usize = 8;
/// Largest mip level the gap rule accepts (a gap of 2^(k+1) texels).
pub const MAX_K: i64 = 20;

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let mut v = [3_i64, 1, 2];
    sort(&mut v);
    usize::from(v == [1, 2, 3])
}

fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1).div_euclid(b)
}

/// Texel size of a box side at scale `s`: whole texels, at least one (the tiny tolerance keeps exact products exact).
pub fn snap(x: f64) -> i64 {
    ((x - 1e-9).ceil() as i64).max(1)
}

/// `G` of a mip level: 2^k texels; 0 (no gap at all) for a negative level.
pub fn block(k: i64) -> i64 {
    if k < 0 {
        0
    } else {
        1_i64 << k.min(MAX_K)
    }
}

// ---------------------------------------------------------------- MaxRects

/// One free rectangle `[x0, y0, x1, y1)`.
type Rect = [i64; 4];

pub struct MaxRects {
    g: i64,
    rule: usize,
    free: Vec<Rect>,
}

/// One axis of a placement inside a free span `[f0, f0 + flen)`: (footprint start, footprint end, content start).
///
/// Mip-k blocks are `g` = 2^k texels. Two neighbours are clean when the blocks they touch, each grown by one block, do not
/// meet. With A before B, that holds when `trail(A) + lead(B) >= 2g` if B starts on a block edge, and `>= 3g - 1` for any
/// start (A ends at `e`, B starts at `s >= e + 3g - 1`, so `floor(s/g) >= ceil(e/g) + 2`). So a box *aligned* on a block
/// edge needs `g` of gap on each side, and any other (an outline, a chart of an existing layout) `2g - 1` on each side:
/// aligned + aligned = 2g, aligned + loose = 3g - 1, loose + loose = 4g - 2. The box packer always aligns: an aligned
/// footprint never ends later than a loose one in the same span.
fn gap_axis(f0: i64, flen: i64, len: i64, g: i64) -> Option<(i64, i64, i64)> {
    let cs = if g == 0 { f0 } else { ceil_div(f0 + g, g) * g };
    let end = cs + len + g;
    (end <= f0 + flen).then_some((cs - g, end, cs))
}

impl MaxRects {
    pub fn new(w: i64, h: i64, k: i64, rule: usize) -> Self {
        MaxRects { g: block(k), rule, free: vec![[0, 0, w, h]] }
    }

    /// Place a box of `w` x `h` texels (content). Returns (content x, content y, turned).
    pub fn insert(&mut self, w: i64, h: i64, can_turn: bool) -> Option<(i64, i64, bool)> {
        let mut best: Option<((i64, i64, i64, i64), Rect, (i64, i64), bool)> = None;
        for f in &self.free {
            for turned in [false, true] {
                if turned && (!can_turn || w == h) {
                    continue;
                }
                let (bw, bh) = if turned { (h, w) } else { (w, h) };
                let (Some(ax), Some(ay)) = (gap_axis(f[0], f[2] - f[0], bw, self.g), gap_axis(f[1], f[3] - f[1], bh, self.g)) else {
                    continue;
                };
                let (rx, ry) = (f[2] - f[0] - (ax.1 - f[0]), f[3] - f[1] - (ay.1 - f[1]));
                let score = match self.rule {
                    1 => (ay.1, ax.1, 0, 0),
                    _ => (rx.min(ry), rx.max(ry), ay.1, ax.1),
                };
                if best.as_ref().map_or(true, |b| score < b.0) {
                    best = Some((score, [ax.0, ay.0, ax.1, ay.1], (ax.2, ay.2), turned));
                }
            }
        }
        let (_, p, (cx, cy), turned) = best?;
        self.split(p);
        Some((cx, cy, turned))
    }

    /// Forget free rectangles in which no box of `side` texels or more can sit, whichever way it is turned: a box needs its
    /// side plus the gap on both sides in each direction. They could never be chosen, so the layout is the same, only faster.
    fn prune(&mut self, side: i64) {
        let need = side + 2 * self.g;
        self.free.retain(|f| f[2] - f[0] >= need && f[3] - f[1] >= need);
    }

    fn split(&mut self, p: Rect) {
        let mut keep = Vec::with_capacity(self.free.len() + 4);
        let mut news: Vec<Rect> = Vec::new();
        for &f in &self.free {
            if p[2] <= f[0] || p[0] >= f[2] || p[3] <= f[1] || p[1] >= f[3] {
                keep.push(f);
                continue;
            }
            if p[0] > f[0] {
                news.push([f[0], f[1], p[0], f[3]]);
            }
            if p[2] < f[2] {
                news.push([p[2], f[1], f[2], f[3]]);
            }
            if p[1] > f[1] {
                news.push([f[0], f[1], f[2], p[1]]);
            }
            if p[3] < f[3] {
                news.push([f[0], p[3], f[2], f[3]]);
            }
        }
        let inside = |a: &Rect, b: &Rect| a[0] >= b[0] && a[1] >= b[1] && a[2] <= b[2] && a[3] <= b[3];
        for (i, n) in news.iter().enumerate() {
            let covered = keep.iter().any(|r| inside(n, r)) || news.iter().enumerate().any(|(j, m)| j != i && inside(n, m) && (m != n || j < i));
            if !covered {
                keep.push(*n);
            }
        }
        self.free = keep;
    }
}

// ---------------------------------------------------------------- order, policies, one pack

/// Boxes in unit size (the page scale multiplies them): width, height, may turn by 90 degrees, weight for order 2.
pub struct Boxes<'a> {
    pub w: &'a [f64],
    pub h: &'a [f64],
    pub turn_ok: &'a [u8],
    pub weight: &'a [f64],
}

impl Boxes<'_> {
    pub fn len(&self) -> usize {
        self.w.len()
    }

    pub fn is_empty(&self) -> bool {
        self.w.is_empty()
    }
}

/// The order of a mode: 0 long side then area, 1 area, 2 area times weight, 3 height then width, 4 width then height;
/// every key largest first, ties by index.
pub fn make_order(b: &Boxes, mode: usize) -> Vec<usize> {
    let key = |i: usize| -> (f64, f64) {
        let (w, h) = (b.w[i], b.h[i]);
        match mode {
            0 => (w.max(h), w * h),
            1 => (w * h, w.max(h)),
            2 => (w * h * b.weight[i], w * h),
            3 => (h, w),
            _ => (w, h),
        }
    };
    let mut o: Vec<usize> = (0..b.len()).collect();
    o.sort_by(|&a, &c| {
        let (ka, kc) = (key(a), key(c));
        kc.0.total_cmp(&ka.0).then(kc.1.total_cmp(&ka.1)).then(a.cmp(&c))
    });
    o
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Place {
    pub x: i64,
    pub y: i64,
    pub turned: bool,
}

/// Pack every box at scale `s` in `order` (policy 0 free turning, 1 wide first, 2 tall first, 3 as given). None when one fails.
pub fn pack(b: &Boxes, order: &[usize], pol: usize, s: f64, page: (i64, i64), k: i64) -> Option<Vec<Place>> {
    let mut mr = MaxRects::new(page.0, page.1, k, (pol / ORIENTS).min(RULES - 1));
    let pol = pol % ORIENTS;
    let mut out = vec![Place { x: 0, y: 0, turned: false }; b.len()];
    // shortest side among the boxes still to place (from each position on), for the pruning of useless free rectangles
    let mut rest = vec![i64::MAX; order.len() + 1];
    for (n, &i) in order.iter().enumerate().rev() {
        rest[n] = rest[n + 1].min(snap(b.w[i] * s).min(snap(b.h[i] * s)));
    }
    for (n, &i) in order.iter().enumerate() {
        let (mut w, mut h) = (snap(b.w[i] * s), snap(b.h[i] * s));
        let ok = b.turn_ok[i] != 0;
        let mut pre = false;
        if ok && ((pol == 1 && h > w) || (pol == 2 && w > h)) {
            std::mem::swap(&mut w, &mut h);
            pre = true;
        }
        let (x, y, t) = mr.insert(w, h, ok && pol == 0)?;
        out[i] = Place { x, y, turned: pre ^ t };
        if rest[n + 1] != i64::MAX {
            mr.prune(rest[n + 1]);
        }
    }
    Some(out)
}

/// Content area of the packed boxes over the page area.
pub fn used(b: &Boxes, s: f64, page: (i64, i64)) -> f64 {
    let a: f64 = (0..b.len()).map(|i| (snap(b.w[i] * s) * snap(b.h[i] * s)) as f64).sum();
    a / (page.0 * page.1) as f64
}

// ---------------------------------------------------------------- fill the page and keep going

pub struct Fill {
    pub scale: f64,
    pub places: Vec<Place>,
    pub order: Vec<usize>,
    pub pol: usize,
    pub used: f64,
    pub passes: usize,
    pub budget_hit: bool,
    pub stopped: bool,
}

struct Clock<'a> {
    start: Instant,
    budget: f64,
    stop: &'a AtomicBool,
}

impl Clock<'_> {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
    fn over(&self) -> bool {
        self.budget > 0.0 && self.start.elapsed().as_secs_f64() > self.budget
    }
    fn halt(&self) -> bool {
        self.stopped() || self.over()
    }
}

/// Best feasible scale for one order and policy. `hint` is where the bracket starts (0: from the area). None: nothing fits.
fn search(b: &Boxes, order: &[usize], pol: usize, page: (i64, i64), k: i64, hint: f64, clock: &Clock) -> Option<(f64, Vec<Place>)> {
    let total: f64 = (0..b.len()).map(|i| b.w[i] * b.h[i]).sum();
    if b.is_empty() || total <= 0.0 {
        return None;
    }
    let mut s = if hint > 0.0 { hint } else { (START_FILL * (page.0 * page.1) as f64 / total).sqrt() };
    let (mut lo, mut hi, mut best): (f64, f64, Option<Vec<Place>>) = (0.0, f64::INFINITY, None);
    // bracket
    loop {
        match pack(b, order, pol, s, page, k) {
            Some(p) => {
                lo = s;
                best = Some(p);
                if hi.is_finite() || clock.halt() {
                    break;
                }
                s *= GROW;
            }
            None => {
                hi = s;
                if best.is_some() || clock.halt() {
                    break;
                }
                if (0..b.len()).all(|i| snap(b.w[i] * s) == 1 && snap(b.h[i] * s) == 1) {
                    return None;
                }
                s /= GROW;
            }
        }
    }
    best.as_ref()?;
    // bisection: lo is always feasible and hi never is, so the bracket only tightens
    while hi.is_finite() && (hi - lo) > SCALE_TOL * lo && !clock.halt() {
        let mid = 0.5 * (lo + hi);
        match pack(b, order, pol, mid, page, k) {
            Some(p) => {
                lo = mid;
                best = Some(p);
            }
            None => hi = mid,
        }
    }
    best.map(|p| (lo, p))
}

/// Fill one page. With no start state every order x policy is tried and the best kept; with one, it is the incumbent
/// (so the result never has a smaller scale). Then passes of adjacent swaps: a swap is kept only when the page fits a
/// scale `MIN_GAIN` larger. `passes`: swap passes after the variants (0 = none, negative = until a pass gains nothing). A time budget (seconds, 0 = none) or the stop flag
/// returns the best so far with the flag set.
#[allow(clippy::too_many_arguments)]
pub fn fill_page(b: &Boxes, page: (i64, i64), k: i64, start: Option<(&[usize], usize, f64)>, passes: i64, budget: f64, stop: &AtomicBool) -> Option<Fill> {
    let clock = Clock { start: Instant::now(), budget, stop };
    let mut inc: Option<Fill> = None;
    // a variant is worth a full search only when it fits a scale `MIN_GAIN` above the incumbent
    let consider = |inc: &mut Option<Fill>, order: Vec<usize>, pol: usize, hint: f64, clock: &Clock| {
        let floor = inc.as_ref().map_or(0.0, |f| f.scale * (1.0 + MIN_GAIN));
        if floor > 0.0 && pack(b, &order, pol, floor, page, k).is_none() {
            return;
        }
        if let Some((s, places)) = search(b, &order, pol, page, k, hint.max(floor), clock) {
            if inc.as_ref().map_or(true, |f| s > f.scale) {
                *inc = Some(Fill { scale: s, used: used(b, s, page), places, order, pol, passes: 0, budget_hit: false, stopped: false });
            }
        }
    };
    if let Some((order, pol, hint)) = start {
        if order.len() == b.len() {
            consider(&mut inc, order.to_vec(), pol.min(POLICIES - 1), hint, &clock);
        }
    }
    if inc.is_none() {
        'variants: for mode in 0..ORDERS {
            let order = make_order(b, mode);
            for pol in 0..POLICIES {
                if clock.halt() {
                    break 'variants;
                }
                consider(&mut inc, order.clone(), pol, 0.0, &clock);
            }
        }
    }
    let mut f = inc?;
    let mut done = 0_i64;
    while (passes < 0 || done < passes) && !clock.halt() && b.len() > 1 {
        let before = f.scale;
        for i in 0..f.order.len() - 1 {
            if clock.halt() {
                break;
            }
            f.order.swap(i, i + 1);
            let target = f.scale * (1.0 + MIN_GAIN);
            match pack(b, &f.order, f.pol, target, page, k) {
                Some(_) => match search(b, &f.order, f.pol, page, k, target, &clock) {
                    Some((s, places)) if s > f.scale => {
                        f.scale = s;
                        f.places = places;
                    }
                    _ => f.order.swap(i, i + 1),
                },
                None => f.order.swap(i, i + 1),
            }
        }
        done += 1;
        f.used = used(b, f.scale, page);
        if f.scale < before * (1.0 + MIN_GAIN) {
            break;
        }
    }
    f.passes = done as usize;
    f.budget_hit = clock.over();
    f.stopped = clock.stopped();
    Some(f)
}

// ---------------------------------------------------------------- irregular charts: row intervals

/// Sorted, merged half-open intervals per texel row.
pub type Rows = Vec<Vec<(i64, i64)>>;

fn merge(mut v: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    v.sort_unstable();
    let mut out: Vec<(i64, i64)> = Vec::with_capacity(v.len());
    for (a, b) in v {
        match out.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// x range of the triangle clipped to the strip `y0 <= y <= y1` (None when the strip misses it).
fn strip_range(t: &[f64; 6], y0: f64, y1: f64) -> Option<(f64, f64)> {
    let mut poly: Vec<(f64, f64)> = vec![(t[0], t[1]), (t[2], t[3]), (t[4], t[5])];
    for (keep_above, edge) in [(true, y0), (false, y1)] {
        let inside = |p: (f64, f64)| if keep_above { p.1 >= edge } else { p.1 <= edge };
        let mut out = Vec::with_capacity(poly.len() + 2);
        for i in 0..poly.len() {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            if inside(p) {
                out.push(p);
            }
            if inside(p) != inside(q) {
                let u = (edge - p.1) / (q.1 - p.1);
                out.push((p.0 + u * (q.0 - p.0), edge));
            }
        }
        poly = out;
        if poly.is_empty() {
            return None;
        }
    }
    let lo = poly.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let hi = poly.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    Some((lo, hi))
}

/// Conservative raster: every texel a triangle touches. Rows start at `y_origin`, x is absolute.
pub fn raster(tris: &[f64], y_origin: i64, rows: usize) -> Rows {
    let mut acc: Vec<Vec<(i64, i64)>> = vec![Vec::new(); rows];
    for t in tris.chunks_exact(6) {
        let t: [f64; 6] = [t[0], t[1], t[2], t[3], t[4], t[5]];
        let (ylo, yhi) = (t[1].min(t[3]).min(t[5]), t[1].max(t[3]).max(t[5]));
        let (r0, r1) = (ylo.floor() as i64, (yhi.ceil() as i64).max(ylo.floor() as i64 + 1));
        for r in r0.max(y_origin)..r1.min(y_origin + rows as i64) {
            if let Some((a, b)) = strip_range(&t, r as f64, r as f64 + 1.0) {
                let (a, b) = (a.floor() as i64, (b.ceil() as i64).max(a.floor() as i64 + 1));
                acc[(r - y_origin) as usize].push((a, b));
            }
        }
    }
    acc.into_iter().map(merge).collect()
}

/// Grow row intervals by `r` texels in both axes (Minkowski sum with a square): `rows.len() + 2r` rows; the result's
/// corner is `r` up and left of the input's, so x is shifted by `+r` (the content then sits at offset `r`).
pub fn dilate(rows: &Rows, r: i64) -> Rows {
    let h = rows.len() as i64;
    let wide: Vec<Vec<(i64, i64)>> = rows.iter().map(|v| v.iter().map(|&(a, b)| (a, b + 2 * r)).collect()).collect();
    (0..h + 2 * r)
        .map(|y| {
            let (j0, j1) = ((y - 2 * r).max(0), y.min(h - 1));
            merge((j0..=j1).flat_map(|j| wide[j as usize].iter().copied()).collect())
        })
        .collect()
}

/// A chart as row intervals relative to its own corner, plus its extent.
#[derive(Clone)]
pub struct Shape {
    pub w: i64,
    pub rows: Rows,
}

impl Shape {
    pub fn h(&self) -> i64 {
        self.rows.len() as i64
    }
}

/// The page's occupied texels per row.
pub struct Page {
    pub w: i64,
    pub rows: Rows,
}

impl Page {
    pub fn new(w: i64, h: i64) -> Self {
        Page { w, rows: vec![Vec::new(); h as usize] }
    }

    pub fn h(&self) -> i64 {
        self.rows.len() as i64
    }

    /// Put absolute row intervals (rows from `y0`) onto the page, clipped to it.
    pub fn add(&mut self, y0: i64, rows: &Rows, dx: i64) {
        for (j, v) in rows.iter().enumerate() {
            let y = y0 + j as i64;
            if y < 0 || y >= self.h() {
                continue;
            }
            let row = &mut self.rows[y as usize];
            row.extend(v.iter().map(|&(a, b)| (a + dx, b + dx)));
            *row = merge(std::mem::take(row));
        }
    }

    /// Height used so far (rows up to the last occupied one).
    pub fn height(&self) -> i64 {
        self.rows.iter().rposition(|r| !r.is_empty()).map_or(0, |p| p as i64 + 1)
    }

    /// Leftmost x >= `x0` (a multiple of `step`) where the shape fits at row `y`, or None.
    fn leftmost(&self, s: &Shape, y: i64, x0: i64, step: i64) -> Option<i64> {
        let mut x = x0;
        'again: loop {
            if x + s.w > self.w {
                return None;
            }
            for (r, v) in s.rows.iter().enumerate() {
                let occ = &self.rows[(y + r as i64) as usize];
                for &(a, b) in v {
                    for &(c, d) in occ {
                        if c < x + b && d > x + a {
                            x = ceil_div(d - a, step) * step;
                            continue 'again;
                        }
                    }
                }
            }
            return Some(x);
        }
    }

    /// Texels of free span to the left of the shape in each of its rows (the wasted interval length).
    fn waste(&self, s: &Shape, x: i64, y: i64) -> i64 {
        s.rows
            .iter()
            .enumerate()
            .filter_map(|(r, v)| v.first().map(|f| (r, f.0)))
            .map(|(r, a)| {
                let left = x + a;
                let end = self.rows[(y + r as i64) as usize].iter().map(|c| c.1).filter(|&d| d <= left).max().unwrap_or(0);
                left - end
            })
            .sum()
    }

    /// Best spot for a shape: inside the used height first (least waste), else the lowest row. `step` = grid of the origin.
    pub fn find(&self, s: &Shape, step: i64) -> Option<(i64, i64, i64)> {
        let (h, used_h) = (s.h(), self.height());
        if h > self.h() || s.w > self.w {
            return None;
        }
        let mut best: Option<((i64, i64, i64), i64, i64)> = None;
        let mut y = 0;
        while y + h <= self.h() {
            if let Some(x) = self.leftmost(s, y, 0, step) {
                let top = (y + h).max(used_h);
                let score = (top, self.waste(s, x, y), y);
                if best.as_ref().map_or(true, |b| score < b.0) {
                    best = Some((score, x, y));
                }
                if top > used_h {
                    break; // lowest row that grows the page: nothing below beats it
                }
            }
            y += step;
        }
        best.map(|(sc, x, y)| (x, y, sc.1))
    }
}

/// Rotate triangles by `quarter` x 90 degrees ((x, y) -> (-y, x)).
pub fn turn_tris(tris: &[f64], quarter: usize) -> Vec<f64> {
    tris.chunks_exact(2)
        .flat_map(|p| match quarter % 4 {
            0 => [p[0], p[1]],
            1 => [-p[1], p[0]],
            2 => [-p[0], -p[1]],
            _ => [p[1], -p[0]],
        })
        .collect()
}

/// Result per chart: placed, shift to add to the (turned) triangles, quarter turns.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChartPlace {
    pub ok: bool,
    pub dx: i64,
    pub dy: i64,
    pub quarter: usize,
}

/// Pack charts into a page that may already hold others (`seed` triangles in texels; they stay where they are).
/// `tri_chart` gives each triangle's chart; `as_box` charts are reserved as their bounding box (and may sit on block
/// edges); the rest follow their outline with the loose gap. Order: long side, then area. Charts may turn when `turn_ok`.
#[allow(clippy::too_many_arguments)]
pub fn pack_into(page: (i64, i64), k: i64, seed: &[f64], tris: &[f64], tri_chart: &[i64], charts: usize, as_box: &[u8], turn_ok: &[u8], stop: &AtomicBool) -> Vec<ChartPlace> {
    let g = block(k);
    let (loose, tight) = if g == 0 { (0, 0) } else { (2 * g - 1, g) };
    let mut occ = Page::new(page.0, page.1);
    if !seed.is_empty() {
        let (ylo, yhi) = seed.chunks_exact(2).fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
        let y0 = ylo.floor() as i64;
        let rows = (yhi.ceil() as i64 + 1 - y0).max(1) as usize;
        occ.add(y0 - loose, &dilate(&raster(seed, y0, rows), loose), -loose);
    }
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); charts];
    for (t, &c) in tri_chart.iter().enumerate() {
        if c >= 0 && (c as usize) < charts {
            per[c as usize].extend_from_slice(&tris[6 * t..6 * t + 6]);
        }
    }
    let extent = |t: &[f64]| -> (f64, f64, f64, f64) {
        t.chunks_exact(2).fold((f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[0]), a.1.min(p[1]), a.2.max(p[0]), a.3.max(p[1])))
    };
    let mut order: Vec<usize> = (0..charts).filter(|&c| !per[c].is_empty()).collect();
    let key = |c: usize| {
        let e = extent(&per[c]);
        let (w, h) = (e.2 - e.0, e.3 - e.1);
        (w.max(h), w * h)
    };
    order.sort_by(|&a, &c| key(c).0.total_cmp(&key(a).0).then(key(c).1.total_cmp(&key(a).1)).then(a.cmp(&c)));
    let mut out = vec![ChartPlace::default(); charts];
    for c in order {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let quarters: &[usize] = if turn_ok.get(c).copied().unwrap_or(0) != 0 { &[0, 1, 2, 3] } else { &[0] };
        let mut best: Option<((i64, i64, i64, usize), Shape, i64, i64, i64, usize, i64, i64)> = None;
        for &q in quarters {
            let t = turn_tris(&per[c], q);
            let e = extent(&t);
            let (fx, fy) = (e.0.floor() as i64, e.1.floor() as i64);
            let rows = ((e.3.ceil() as i64).max(fy + 1) - fy) as usize;
            let local: Vec<f64> = t.chunks_exact(2).flat_map(|p| [p[0] - fx as f64, p[1] - fy as f64]).collect();
            let mut core = raster(&local, 0, rows);
            let boxed = as_box.get(c).copied().unwrap_or(0) != 0;
            let cw = core.iter().flatten().map(|i| i.1).max().unwrap_or(1);
            if boxed {
                core = vec![vec![(0, cw)]; rows];
            }
            // loose: follows the outline, any position; aligned: a box on a block edge with less gap
            let mut variants: Vec<(i64, i64, Rows, i64)> = vec![(loose, 1, core.clone(), cw)];
            if boxed && g > 0 {
                variants.insert(0, (tight, g, core, cw));
            }
            for (r, step, shape_rows, w) in variants {
                let shape = Shape { w: w + 2 * r, rows: dilate(&shape_rows, r) };
                if let Some((x, y, waste)) = occ.find(&shape, step) {
                    let top = (y + shape.h()).max(occ.height());
                    let score = (top, waste, y, q);
                    if best.as_ref().map_or(true, |b| score < b.0) {
                        best = Some((score, shape, x, y, r, q, fx, fy));
                    }
                }
            }
        }
        if let Some((_, shape, x, y, r, q, fx, fy)) = best {
            occ.add(y, &shape.rows, x);
            out[c] = ChartPlace { ok: true, dx: x + r - fx, dy: y + r - fy, quarter: q };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn never() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn boxes<'a>(w: &'a [f64], h: &'a [f64], t: &'a [u8], wt: &'a [f64]) -> Boxes<'a> {
        Boxes { w, h, turn_ok: t, weight: wt }
    }

    fn dims(b: &Boxes, f: &Fill, i: usize) -> (i64, i64) {
        let (w, h) = (snap(b.w[i] * f.scale), snap(b.h[i] * f.scale));
        if f.places[i].turned {
            (h, w)
        } else {
            (w, h)
        }
    }

    /// Blocks of 2^j texels a content rectangle touches, grown by one block.
    fn padded(x: i64, y: i64, w: i64, h: i64, j: u32) -> (i64, i64, i64, i64) {
        let s = 1_i64 << j;
        (x.div_euclid(s) - 1, y.div_euclid(s) - 1, (x + w - 1).div_euclid(s) + 1, (y + h - 1).div_euclid(s) + 1)
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / (1u64 << 31) as f64
    }

    #[test]
    fn four_equal_squares_fill_the_page() {
        let (w, h, t, wt) = ([1.0; 4], [1.0; 4], [1u8; 4], [1.0; 4]);
        let b = boxes(&w, &h, &t, &wt);
        let f = fill_page(&b, (1024, 1024), -1, None, 1, 0.0, &never()).unwrap();
        assert!((f.used - 1.0).abs() < 1e-12, "used {}", f.used);
    }

    #[test]
    fn gap_holds_at_every_mip_level_and_nothing_overlaps() {
        let mut seed = 7u64;
        for k in [0_i64, 1, 3] {
            let n = 60;
            let w: Vec<f64> = (0..n).map(|_| 0.2 + lcg(&mut seed)).collect();
            let h: Vec<f64> = (0..n).map(|_| 0.2 + lcg(&mut seed)).collect();
            let (t, wt) = (vec![1u8; n], vec![1.0; n]);
            let b = boxes(&w, &h, &t, &wt);
            let f = fill_page(&b, (512, 512), k, None, 1, 0.0, &never()).unwrap();
            for i in 0..n {
                let (bw, bh) = dims(&b, &f, i);
                let p = f.places[i];
                assert!(p.x >= 0 && p.y >= 0 && p.x + bw <= 512 && p.y + bh <= 512);
                for j in 0..i {
                    let (cw, ch) = dims(&b, &f, j);
                    let q = f.places[j];
                    assert!(p.x + bw <= q.x || q.x + cw <= p.x || p.y + bh <= q.y || q.y + ch <= p.y, "overlap");
                    for lvl in 0..=k as u32 {
                        let (a, c) = (padded(p.x, p.y, bw, bh, lvl), padded(q.x, q.y, cw, ch, lvl));
                        assert!(a.2 < c.0 || c.2 < a.0 || a.3 < c.1 || c.3 < a.1, "k {k} level {lvl}: boxes {i} {j} meet");
                    }
                }
            }
        }
    }

    #[test]
    fn keep_going_never_lowers_the_scale() {
        let mut seed = 11u64;
        let n = 40;
        let w: Vec<f64> = (0..n).map(|_| 0.1 + lcg(&mut seed)).collect();
        let h: Vec<f64> = (0..n).map(|_| 0.1 + lcg(&mut seed)).collect();
        let (t, wt) = (vec![1u8; n], vec![1.0; n]);
        let b = boxes(&w, &h, &t, &wt);
        let mut f = fill_page(&b, (1024, 1024), 1, None, 1, 0.0, &never()).unwrap();
        for _ in 0..4 {
            let g = fill_page(&b, (1024, 1024), 1, Some((&f.order, f.pol, f.scale)), 1, 0.0, &never()).unwrap();
            assert!(g.scale >= f.scale && g.used >= f.used, "{} < {}", g.scale, f.scale);
            f = g;
        }
        // a start far too small or too large settles on the same bracket (no oscillation)
        let lo = fill_page(&b, (1024, 1024), 1, Some((&f.order, f.pol, f.scale * 0.1)), 1, 0.0, &never()).unwrap();
        let hi = fill_page(&b, (1024, 1024), 1, Some((&f.order, f.pol, f.scale * 10.0)), 1, 0.0, &never()).unwrap();
        assert!(lo.scale >= f.scale * 0.999 && hi.scale >= f.scale * 0.999);
    }

    #[test]
    fn rows_raster_is_conservative_and_dilation_is_a_square() {
        let rows = raster(&[0.5, 0.5, 3.5, 0.5, 0.5, 3.5], 0, 4);
        assert_eq!(rows[0], vec![(0, 4)]);
        assert_eq!(rows[3], vec![(0, 1)]);
        let d = dilate(&vec![vec![(5, 6)]], 2);
        assert_eq!(d.len(), 5);
        assert!(d.iter().all(|r| *r == vec![(5, 10)]));
    }

    #[test]
    fn pack_into_never_overlaps_the_seed_or_each_other() {
        // seed: two big squares; charts: a few triangles and boxes
        let seed = vec![10.0, 10.0, 90.0, 10.0, 10.0, 90.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0];
        let mut tris = Vec::new();
        let mut tc = Vec::new();
        for c in 0..6_i64 {
            let s = 10.0 + 4.0 * c as f64;
            tris.extend_from_slice(&[0.0, 0.0, s, 0.0, 0.0, s]);
            tc.push(c);
        }
        let flags = vec![0u8, 1, 0, 1, 0, 1];
        let placed = pack_into((128, 128), 1, &seed, &tris, &tc, 6, &flags, &[1; 6], &never());
        assert!(placed.iter().all(|p| p.ok));
        let mut occ_tex = vec![false; 128 * 128];
        let mut mark = |rows: &Rows, y0: i64, dx: i64| {
            for (j, r) in rows.iter().enumerate() {
                for &(a, b) in r {
                    for x in a + dx..b + dx {
                        let y = y0 + j as i64;
                        assert!(x >= 0 && x < 128 && y >= 0 && y < 128, "outside page");
                        assert!(!occ_tex[(y * 128 + x) as usize], "overlap at {x},{y}");
                        occ_tex[(y * 128 + x) as usize] = true;
                    }
                }
            }
        };
        mark(&raster(&seed, 10, 80), 10, 0);
        for (c, p) in placed.iter().enumerate() {
            let t: Vec<f64> = turn_tris(&tris[6 * c..6 * c + 6], p.quarter).chunks_exact(2).flat_map(|q| [q[0] + p.dx as f64, q[1] + p.dy as f64]).collect();
            let ylo = t.chunks_exact(2).map(|q| q[1]).fold(f64::INFINITY, f64::min).floor() as i64;
            let yhi = t.chunks_exact(2).map(|q| q[1]).fold(f64::NEG_INFINITY, f64::max).ceil() as i64;
            mark(&raster(&t, ylo, (yhi - ylo) as usize), ylo, 0);
        }
    }
}
