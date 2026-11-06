// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Edge-collapse reduction with Garland and Heckbert's quadrics, for pieces too big for Blender's Decimate to run on the
//! main thread. It follows Decimate's rules: area-weighted planes, heavy planes along borders, the best point of the
//! merged quadric, no collapse that flips a face or breaks the surface.
//!
//! One run takes a piece all the way down and records every collapse. Which edge goes next doesn't depend on how far
//! the run is going, so every smaller size is a prefix of the same list, and a size costs one replay instead of one run.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

const BORDER_WEIGHT: f64 = 100.0; // Decimate's BOUNDARY_PRESERVE_WEIGHT, per unit of edge length squared here
const FLAT_COST: f64 = 1e-12; // where every error is about zero, the shortest edge goes first (Decimate's TOPOLOGY_FALLBACK_EPS)
const SINGULAR: f64 = 1e-8; // Decimate's OPTIMIZE_EPS
const MOST_VALENCE: usize = 64; // a collapse past this builds a fan nothing good comes out of
// Stricter than Decimate, whose test lets a face turn almost 90 degrees per collapse and so, over many collapses, fold over.
const STEP_TURN: f64 = 0.5; // a face may turn at most 60 degrees in one collapse
const SOURCE_TURN: f64 = 0.0; // nor ever end up more than 90 degrees from where it faced in the source
const NEAREST_TURN: f64 = -0.5; // nor face more than 120 degrees from the source nearest its middle (triadq.checks.FACING)
const FOLD: f64 = -0.866; // two faces around the new point more than 150 degrees apart fold over each other (triadq.checks.FOLD)
const THIN_REACH: f64 = 0.03; // of the piece's size: a surface facing the other way this near makes a spot thin
const THIN_DIRECTIONS: usize = 12; // ways the triangles of one cell face, kept to tell

type Q = [f64; 10]; // the upper triangle of the symmetric 4x4 matrix, row by row

fn q_add(a: &mut Q, b: &Q) {
    for k in 0..10 {
        a[k] += b[k];
    }
}

fn q_plane(n: [f64; 3], d: f64, weight: f64) -> Q {
    let [a, b, c] = n;
    [a * a * weight, a * b * weight, a * c * weight, a * d * weight, b * b * weight, b * c * weight, b * d * weight, c * c * weight, c * d * weight, d * d * weight]
}

fn q_eval(q: &Q, p: [f64; 3]) -> f64 {
    let [x, y, z] = p;
    x * (q[0] * x + 2.0 * (q[1] * y + q[2] * z + q[3])) + y * (q[4] * y + 2.0 * (q[5] * z + q[6])) + z * (q[7] * z + 2.0 * q[8]) + q[9]
}

/// The point where this quadric is smallest, if it has one.
fn q_best(q: &Q) -> Option<[f64; 3]> {
    let (a, b, c, e, f, h) = (q[0], q[1], q[2], q[4], q[5], q[7]);
    let (m00, m01, m02) = (e * h - f * f, c * f - b * h, b * f - c * e);
    let det = a * m00 + b * m01 + c * m02;
    if det.abs() <= SINGULAR * (a + e + h).abs().powi(3).max(1e-300) {
        return None;
    }
    let (m11, m12, m22) = (a * h - c * c, b * c - a * f, a * e - b * b);
    let (dx, dy, dz) = (-q[3], -q[6], -q[8]);
    Some([(m00 * dx + m01 * dy + m02 * dz) / det, (m01 * dx + m11 * dy + m12 * dz) / det, (m02 * dx + m12 * dy + m22 * dz) / det])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Default)]
struct Mix(u64); // a small multiply-rotate hash for grid cells: the std one is many times slower on three integers

impl std::hash::Hasher for Mix {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_i64(&mut self, n: i64) {
        self.write_u64(n as u64);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

type Cells = std::collections::HashMap<[i64; 3], Vec<u32>, std::hash::BuildHasherDefault<Mix>>;

/// The live triangles by where they are, so a collapse can ask which ones its new triangles might pass through. A
/// loose grid in levels: a triangle sits in one cell, at its lowest corner, of the level whose cells are at least as big
/// as it, so it reaches at most into the next cell along each axis. Triangles grow as the piece shrinks and move up.
struct Grid {
    base: f64,
    levels: Vec<Cells>,
    home: Vec<(usize, [i64; 3])>,
}

impl Grid {
    const NOWHERE: usize = usize::MAX;

    fn insert(&mut self, t: u32, low: [f64; 3], high: [f64; 3]) {
        let extent = (0..3).map(|k| high[k] - low[k]).fold(0.0, f64::max);
        let (mut level, mut size) = (0, self.base);
        while size < extent && level < 60 {
            size *= 2.0;
            level += 1;
        }
        let key = [(low[0] / size).floor() as i64, (low[1] / size).floor() as i64, (low[2] / size).floor() as i64];
        while self.levels.len() <= level {
            self.levels.push(Cells::default());
        }
        self.levels[level].entry(key).or_default().push(t);
        self.home[t as usize] = (level, key);
    }

    fn remove(&mut self, t: u32) {
        let (level, key) = self.home[t as usize];
        if level == Self::NOWHERE {
            return;
        }
        let cells = &mut self.levels[level];
        if let Some(cell) = cells.get_mut(&key) {
            if let Some(at) = cell.iter().position(|&x| x == t) {
                cell.swap_remove(at);
            }
            if cell.is_empty() {
                cells.remove(&key);
            }
        }
        self.home[t as usize] = (Self::NOWHERE, key);
    }

    /// Calls `each` with every triangle that may reach into this box.
    fn visit(&self, low: [f64; 3], high: [f64; 3], mut each: impl FnMut(u32)) {
        let mut size = self.base;
        for cells in &self.levels {
            if !cells.is_empty() {
                let from = [(low[0] / size).floor() as i64 - 1, (low[1] / size).floor() as i64 - 1, (low[2] / size).floor() as i64 - 1];
                let to = [(high[0] / size).floor() as i64, (high[1] / size).floor() as i64, (high[2] / size).floor() as i64];
                let span = (0..3).map(|k| (to[k] - from[k] + 1) as f64).product::<f64>();
                if span > cells.len() as f64 {
                    for (key, cell) in cells {
                        if (0..3).all(|k| from[k] <= key[k] && key[k] <= to[k]) {
                            cell.iter().for_each(|&t| each(t));
                        }
                    }
                } else {
                    for x in from[0]..=to[0] {
                        for y in from[1]..=to[1] {
                            for z in from[2]..=to[2] {
                                if let Some(cell) = cells.get(&[x, y, z]) {
                                    cell.iter().for_each(|&t| each(t));
                                }
                            }
                        }
                    }
                }
            }
            size *= 2.0;
        }
    }
}

#[derive(Clone, Copy)]
struct Entry {
    cost: f64,
    a: u32,
    b: u32,
    stamp_a: u32,
    stamp_b: u32,
    way: u8, // 0 the best point, 1 the middle, 2 at a, 3 at b
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed, for a smallest-first heap; ties by edge, so the order never depends on anything but the mesh.
        other.cost.total_cmp(&self.cost).then_with(|| (other.a, other.b, other.way).cmp(&(self.a, self.b, self.way)))
    }
}

pub struct Collapse {
    keep: u32,
    drop: u32,
    at: [f64; 3],
    faces_left: u32,
}

/// A piece and every collapse found for it.
pub struct Run {
    points: Vec<f64>,
    tris: Vec<u32>,
    collapses: Vec<Collapse>,
    faces: u32,
}

struct State {
    p: Vec<[f64; 3]>,
    q: Vec<Q>,
    tris: Vec<[u32; 3]>,
    around: Vec<Vec<u32>>, // live triangles at each vertex
    locked: Vec<bool>,
    border: Vec<bool>,
    stamp: Vec<u32>,
    dead: Vec<bool>,
    blocked: Vec<Vec<u32>>, // edges refused because of a flip, tried again when something near them changes
    facing: Vec<[f64; 3]>,  // each triangle's unit normal in the source (zero for a degenerate one)
    source: crate::near::FaceTree, // the source's triangles, which say which way is out near a moved one
    heap: BinaryHeap<Entry>,
    scale: f64,
    mark: Vec<u32>, // scratch: neighbour marks by generation
    generation: u32,
    grid: Grid,          // the live triangles by place, for the crossing test
    tri_mark: Vec<u32>,  // scratch: the triangles a collapse changes, by generation
    thin: Vec<bool>,     // near a surface facing the other way: only here can a collapse push through the piece
}

impl State {
    fn target(&self, a: u32, b: u32, way: u8) -> Option<[f64; 3]> {
        let (pa, pb) = (self.p[a as usize], self.p[b as usize]);
        let middle = || [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5];
        if self.locked[a as usize] {
            return (way == 2).then_some(pa);
        }
        if self.locked[b as usize] {
            return (way == 3).then_some(pb);
        }
        match way {
            0 => {
                let mut q = self.q[a as usize];
                q_add(&mut q, &self.q[b as usize]);
                // A far-off best point means the quadric is nearly flat in some direction: the middle is safer.
                q_best(&q).filter(|p| {
                    let (m, v) = (middle(), sub(pa, pb));
                    let off = sub(*p, m);
                    dot(off, off) <= 4.0 * dot(v, v).max(self.scale * self.scale * 1e-12) + 1e-30
                })
            }
            1 => Some(middle()),
            2 => Some(pa),
            _ => Some(pb),
        }
    }

    fn entry(&self, a: u32, b: u32, first_way: u8) -> Option<Entry> {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        if self.locked[a as usize] && self.locked[b as usize] {
            return None;
        }
        let mut q = self.q[a as usize];
        q_add(&mut q, &self.q[b as usize]);
        let side = sub(self.p[a as usize], self.p[b as usize]);
        for way in first_way..4 {
            if let Some(at) = self.target(a, b, way) {
                let error = q_eval(&q, at).abs();
                return Some(Entry {
                    cost: error.max(dot(side, side) * FLAT_COST),
                    a,
                    b,
                    stamp_a: self.stamp[a as usize],
                    stamp_b: self.stamp[b as usize],
                    way,
                });
            }
        }
        None
    }

    fn push(&mut self, a: u32, b: u32) {
        if let Some(entry) = self.entry(a, b, 0) {
            self.heap.push(entry);
        }
    }

    /// Does merging a and b keep the surface a surface? Returns how many triangles it takes away, or 0 for no.
    fn can_collapse(&mut self, a: u32, b: u32) -> usize {
        if self.around[a as usize].len() + self.around[b as usize].len() > MOST_VALENCE {
            return 0;
        }
        self.generation += 1;
        let now = self.generation;
        let mut shared = 0usize;
        for &t in &self.around[a as usize] {
            let tri = self.tris[t as usize];
            if tri.contains(&b) {
                shared += 1;
            }
            for v in tri {
                self.mark[v as usize] = now;
            }
        }
        if shared == 0 || shared > 2 {
            return 0;
        }
        let mut common = 0usize;
        for &t in &self.around[b as usize] {
            for v in self.tris[t as usize] {
                if v != a && v != b && self.mark[v as usize] == now {
                    self.mark[v as usize] = now.wrapping_sub(1);
                    common += 1;
                }
            }
        }
        if common != shared {
            return 0; // the link condition: anything else pinches the surface or closes a tunnel
        }
        // A chord between two borders would tear the piece in two; an inner edge with one end on a border is fine.
        if shared == 2 && self.border[a as usize] && self.border[b as usize] {
            return 0;
        }
        shared
    }

    /// Would moving these two to `at` turn a triangle over or squash it (Decimate's degenerate-flip test), turn it far
    /// in one go, away from the source or the source nearest it, or fold two of the moved triangles over each other?
    fn flips(&self, a: u32, b: u32, at: [f64; 3]) -> bool {
        let unit = |n: [f64; 3]| {
            let l = dot(n, n).sqrt();
            if l > 0.0 { [n[0] / l, n[1] / l, n[2] / l] } else { [0.0; 3] }
        };
        let mut fan: Vec<([u32; 2], [f64; 3], [f64; 3])> = Vec::new(); // (far corners, old normal, new normal)
        for (v, other) in [(a, b), (b, a)] {
            let old = self.p[v as usize];
            for &t in &self.around[v as usize] {
                let tri = self.tris[t as usize];
                if tri.contains(&other) {
                    continue;
                }
                let k = tri.iter().position(|&x| x == v).unwrap();
                let (p, n) = (self.p[tri[(k + 2) % 3] as usize], self.p[tri[(k + 1) % 3] as usize]);
                let side = sub(p, n); // so these are the triangle's own normals, which way it faces
                let before = cross(side, sub(old, p));
                let after = cross(side, sub(at, p));
                let both = dot(before, before) * dot(after, after);
                if dot(before, after) <= (dot(before, before) + dot(after, after)) * 0.01 || dot(before, after) < STEP_TURN * both.sqrt() {
                    return true;
                }
                let after = unit(after);
                if dot(after, self.facing[t as usize]) < SOURCE_TURN && self.facing[t as usize] != [0.0; 3] {
                    return true;
                }
                // Slid over to another side of the source (a thin plate's back, a wall behind): inside out there.
                let middle = [(at[0] + p[0] + n[0]) / 3.0, (at[1] + p[1] + n[1]) / 3.0, (at[2] + p[2] + n[2]) / 3.0];
                if let Some((_, f, _)) = self.source.nearest(middle) {
                    if dot(after, self.facing[f as usize]) < NEAREST_TURN {
                        return true;
                    }
                }
                fan.push(([tri[(k + 1) % 3], tri[(k + 2) % 3]], unit(before), after));
            }
        }
        // Around the new point, triangles next to each other share a far corner. A pair already that far apart (a thin
        // blade's edge) may stay so.
        fan.iter().enumerate().any(|(i, x)| {
            fan[i + 1..].iter().any(|y| (x.0[0] == y.0[1] || x.0[1] == y.0[0]) && dot(x.2, y.2) < FOLD && dot(x.1, y.1) >= FOLD)
        })
    }

    fn bounds(corners: &[[f64; 3]; 3]) -> ([f64; 3], [f64; 3]) {
        let mut low = corners[0];
        let mut high = corners[0];
        for c in &corners[1..] {
            for k in 0..3 {
                low[k] = low[k].min(c[k]);
                high[k] = high[k].max(c[k]);
            }
        }
        (low, high)
    }

    fn corners(&self, t: u32) -> [[f64; 3]; 3] {
        let tri = self.tris[t as usize];
        [self.p[tri[0] as usize], self.p[tri[1] as usize], self.p[tri[2] as usize]]
    }

    /// How many of the `near` triangles (with their corners in single precision and boxes) this one passes through,
    /// leaving out those sharing a corner with it.
    fn crossings(&self, corners: &[[f64; 3]; 3], ends: [u32; 2], near: &[(u32, [[f32; 3]; 3], [f64; 3], [f64; 3])]) -> usize {
        let (low, high) = Self::bounds(corners);
        let small = corners.map(|c| c.map(|x| x as f32));
        near.iter()
            .filter(|(t, other, o_low, o_high)| {
                let tri = self.tris[*t as usize];
                (0..3).all(|k| low[k] <= o_high[k] && o_low[k] <= high[k]) && !tri.contains(&ends[0]) && !tri.contains(&ends[1]) && crate::near::triangles_cross(&small, other)
            })
            .count()
    }

    /// The triangles around a and b, with a moved to `at` (or left, for None), as (corners, far ends), and the box round them.
    fn fan(&self, a: u32, b: u32, at: Option<[f64; 3]>) -> (Vec<([[f64; 3]; 3], [u32; 2])>, [f64; 3], [f64; 3]) {
        let mut out = Vec::with_capacity(self.around[a as usize].len() + self.around[b as usize].len());
        let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for (v, other) in [(a, b), (b, a)] {
            for &t in &self.around[v as usize] {
                let tri = self.tris[t as usize];
                if tri.contains(&other) && (at.is_some() || v == b) {
                    continue; // moved, it goes away; left, it counts once
                }
                let k = tri.iter().position(|&x| x == v).unwrap();
                let mut corners = self.corners(t);
                if let Some(p) = at {
                    corners[k] = p;
                }
                let (l, h) = Self::bounds(&corners);
                for c in 0..3 {
                    low[c] = low[c].min(l[c]);
                    high[c] = high[c].max(h[c]);
                }
                out.push((corners, [tri[(k + 1) % 3], tri[(k + 2) % 3]]));
            }
        }
        (out, low, high)
    }

    /// Would moving these two to `at` make more crossings with the rest of the piece than their triangles have now? QEM
    /// alone collapses a thin ring or flange right through the tube it sits on, and the repair later spends its whole
    /// time limit on those spots (the crane's ringed posts, the drill press flange).
    fn crosses(&mut self, a: u32, b: u32, at: [f64; 3]) -> bool {
        self.generation += 1;
        let now = self.generation;
        for v in [a, b] {
            for &t in &self.around[v as usize] {
                self.tri_mark[t as usize] = now;
            }
        }
        let count = |s: &Self, fan: &[([[f64; 3]; 3], [u32; 2])], low: [f64; 3], high: [f64; 3]| -> usize {
            let mut near = Vec::new();
            s.grid.visit(low, high, |t| {
                if s.tri_mark[t as usize] != now {
                    let corners = s.corners(t);
                    let (l, h) = Self::bounds(&corners);
                    near.push((t, corners.map(|c| c.map(|x| x as f32)), l, h));
                }
            });
            if near.is_empty() {
                return 0;
            }
            fan.iter().map(|(corners, ends)| s.crossings(corners, *ends, &near)).sum()
        };
        let (moved, low, high) = self.fan(a, b, Some(at));
        let after = count(self, &moved, low, high);
        if after == 0 {
            return false;
        }
        let (left, low, high) = self.fan(a, b, None); // only asked when needed: most collapses cross nothing
        after > count(self, &left, low, high)
    }

    fn apply(&mut self, a: u32, b: u32, at: [f64; 3], collapses: &mut Vec<Collapse>, alive: &mut u32, shared: usize) {
        let (au, bu) = (a as usize, b as usize);
        let held = self.locked[au]; // a locked vertex stays exactly where it was, to the last digit: chunks join on them
        for v in [au, bu] {
            for &t in &self.around[v] {
                self.grid.remove(t);
            }
        }
        let moved = std::mem::take(&mut self.around[bu]);
        let mut gone: [u32; 2] = [u32::MAX; 2];
        let mut count = 0;
        for &t in &moved {
            let tri = &mut self.tris[t as usize];
            if tri.contains(&a) {
                gone[count] = t;
                count += 1;
            } else {
                for v in tri.iter_mut() {
                    if *v == b {
                        *v = a;
                    }
                }
            }
        }
        for &t in &gone[..count] {
            for v in self.tris[t as usize] {
                if v != b {
                    let row = &mut self.around[v as usize];
                    let at_row = row.iter().position(|&x| x == t).unwrap();
                    row.swap_remove(at_row);
                }
            }
        }
        let keep_row = &mut self.around[au];
        keep_row.extend(moved.iter().copied().filter(|t| !gone[..count].contains(t)));
        *alive -= shared as u32;
        let qb = self.q[bu];
        q_add(&mut self.q[au], &qb);
        self.p[au] = at;
        self.border[au] |= self.border[bu];
        self.thin[au] |= self.thin[bu];
        self.locked[au] |= self.locked[bu];
        self.dead[bu] = true;
        self.stamp[au] += 1;
        self.stamp[bu] += 1;
        for i in 0..self.around[au].len() {
            let t = self.around[au][i];
            let (low, high) = Self::bounds(&self.corners(t));
            self.grid.insert(t, low, high);
        }
        collapses.push(Collapse {
            keep: a,
            drop: b,
            at: if held { [f64::NAN; 3] } else { at },
            faces_left: *alive,
        });
        self.generation += 1;
        let now = self.generation;
        let mut near: Vec<u32> = Vec::with_capacity(self.around[au].len() + 2);
        for &t in &self.around[au] {
            for v in self.tris[t as usize] {
                if v != a && self.mark[v as usize] != now {
                    self.mark[v as usize] = now;
                    near.push(v);
                }
            }
        }
        for &n in &near {
            self.push(a, n);
            let waiting = std::mem::take(&mut self.blocked[n as usize]);
            for packed in waiting {
                if !self.dead[packed as usize] {
                    self.push(n, packed);
                }
            }
        }
        let waiting = std::mem::take(&mut self.blocked[au]);
        for packed in waiting {
            if !self.dead[packed as usize] {
                self.push(a, packed);
            }
        }
    }
}

/// Vertices with a surface facing the other way within THIN_REACH of the piece's size: thin rods, rings, flanges and
/// plates, and parts close to each other. Elsewhere a collapse can't reach through the piece, so the crossing test is
/// skipped there; on a big smooth scan that is nearly everywhere. Each cell keeps a few directions its triangles face.
fn thin_vertices(p: &[[f64; 3]], tris: &[[u32; 3]]) -> Vec<bool> {
    let mut cells: std::collections::HashMap<[i64; 3], Vec<[f64; 3]>, std::hash::BuildHasherDefault<Mix>> = Default::default();
    let mut placed: Vec<Option<([i64; 3], [f64; 3])>> = Vec::with_capacity(tris.len());
    for t in tris {
        let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
        let n = cross(sub(b, a), sub(c, a));
        let l = dot(n, n).sqrt();
        if l <= 0.0 {
            placed.push(None);
            continue;
        }
        let n = [n[0] / l, n[1] / l, n[2] / l];
        let key = [0, 1, 2].map(|k| (((a[k] + b[k] + c[k]) / 3.0) / THIN_REACH).floor() as i64);
        let row = cells.entry(key).or_default();
        if row.len() < THIN_DIRECTIONS && !row.iter().any(|&m| dot(m, n) > 0.9) {
            row.push(n);
        }
        placed.push(Some((key, n)));
    }
    let mut thin = vec![false; p.len()];
    for (t, at) in tris.iter().zip(&placed) {
        let Some((key, n)) = at else { continue };
        let mut found = false;
        'look: for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if let Some(row) = cells.get(&[key[0] + x, key[1] + y, key[2] + z]) {
                        if row.iter().any(|&m| dot(m, *n) < -0.5) {
                            found = true;
                            break 'look;
                        }
                    }
                }
            }
        }
        if found {
            for &v in t {
                thin[v as usize] = true;
            }
        }
    }
    thin
}

/// Run everything down to `least` triangles. `locked` vertices (may be empty) never move or go.
pub fn run(points: &[f64], tris: &[u32], locked: &[bool], least: u32) -> Run {
    let watch = crate::near::Watch::now(); // from the moment the call starts
    let nv = points.len() / 3;
    let nf = tris.len() / 3;
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in points.chunks_exact(3) {
        for c in 0..3 {
            low[c] = low[c].min(p[c]);
            high[c] = high[c].max(p[c]);
        }
    }
    let span = dot(sub(high, low), sub(high, low)).sqrt().max(1e-300);
    // Worked out in a box of size one, so the constants mean the same on a bolt and on a cliff.
    let p: Vec<[f64; 3]> = points.chunks_exact(3).map(|c| [(c[0] - low[0]) / span, (c[1] - low[1]) / span, (c[2] - low[2]) / span]).collect();
    let source = {
        let flat: Vec<f64> = p.iter().flatten().copied().collect();
        let indptr: Vec<i64> = (0..=nf as i64).map(|f| 3 * f).collect();
        let corners: Vec<i64> = tris.iter().map(|&v| v as i64).collect();
        crate::near::FaceTree::new(&flat, &indptr, &corners)
    };
    let mut s = State {
        p,
        q: vec![[0.0; 10]; nv],
        tris: tris.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect(),
        around: vec![Vec::new(); nv],
        locked: if locked.len() == nv { locked.to_vec() } else { vec![false; nv] },
        border: vec![false; nv],
        stamp: vec![0; nv],
        dead: vec![false; nv],
        blocked: vec![Vec::new(); nv],
        facing: Vec::new(),
        source,
        heap: BinaryHeap::new(),
        scale: 1.0,
        mark: vec![0; nv],
        generation: 0,
        grid: Grid { base: 1.0, levels: Vec::new(), home: vec![(Grid::NOWHERE, [0; 3]); nf] },
        tri_mark: vec![0; nf],
        thin: Vec::new(),
    };
    // Cells about as big as the piece's triangles are at the start.
    let edge: f64 = s.tris.iter().map(|t| dot(sub(s.p[t[1] as usize], s.p[t[0] as usize]), sub(s.p[t[1] as usize], s.p[t[0] as usize])).sqrt()).sum::<f64>() / nf.max(1) as f64;
    s.grid.base = edge.max(1e-9);
    for t in 0..nf as u32 {
        let (low, high) = State::bounds(&s.corners(t));
        s.grid.insert(t, low, high);
    }
    s.thin = thin_vertices(&s.p, &s.tris);
    s.facing = s
        .tris
        .iter()
        .map(|t| {
            let n = cross(sub(s.p[t[1] as usize], s.p[t[0] as usize]), sub(s.p[t[2] as usize], s.p[t[0] as usize]));
            let l = dot(n, n).sqrt();
            if l > 0.0 { [n[0] / l, n[1] / l, n[2] / l] } else { [0.0; 3] }
        })
        .collect();
    let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(tris.len());
    for (t, tri) in s.tris.iter().enumerate() {
        let (pa, pb, pc) = (s.p[tri[0] as usize], s.p[tri[1] as usize], s.p[tri[2] as usize]);
        let n = cross(sub(pb, pa), sub(pc, pa));
        let twice = dot(n, n).sqrt();
        if twice > 0.0 {
            let unit = [n[0] / twice, n[1] / twice, n[2] / twice];
            let plane = q_plane(unit, -dot(unit, pa), twice * 0.5);
            for v in tri {
                q_add(&mut s.q[*v as usize], &plane);
            }
        }
        for k in 0..3 {
            let (x, y) = (tri[k], tri[(k + 1) % 3]);
            s.around[x as usize].push(t as u32);
            edges.push(if x < y { (x, y, t as u32) } else { (y, x, t as u32) });
        }
    }
    crate::topo::sort(&mut edges);
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 && edges[j].1 == edges[i].1 {
            j += 1;
        }
        let (x, y) = (edges[i].0 as usize, edges[i].1 as usize);
        match j - i {
            1 => {
                s.border[x] = true;
                s.border[y] = true;
                let tri = s.tris[edges[i].2 as usize];
                let (pa, pb, pc) = (s.p[tri[0] as usize], s.p[tri[1] as usize], s.p[tri[2] as usize]);
                let normal = cross(sub(pb, pa), sub(pc, pa));
                let along = sub(s.p[y], s.p[x]);
                let side = cross(along, normal);
                let length = dot(side, side).sqrt();
                if length > 0.0 {
                    let unit = [side[0] / length, side[1] / length, side[2] / length];
                    let plane = q_plane(unit, -dot(unit, s.p[x]), BORDER_WEIGHT * dot(along, along));
                    q_add(&mut s.q[x], &plane);
                    q_add(&mut s.q[y], &plane);
                }
            }
            2 => {}
            _ => {
                s.locked[x] = true;
                s.locked[y] = true;
            }
        }
        i = j;
    }
    let mut start = Vec::with_capacity(edges.len() / 2);
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 && edges[j].1 == edges[i].1 {
            j += 1;
        }
        if let Some(entry) = s.entry(edges[i].0, edges[i].1, 0) {
            start.push(entry);
        }
        i = j;
    }
    drop(edges);
    s.heap = BinaryHeap::from(start);
    let mut collapses = Vec::with_capacity(nf / 2);
    let mut alive = nf as u32;
    while alive > least {
        if collapses.len() % 4096 == 4095 && watch.stopped() {
            break; // the job was stopped: what is reduced so far is all there is
        }
        let Some(entry) = s.heap.pop() else {
            break;
        };
        let (a, b) = (entry.a, entry.b);
        if s.dead[a as usize] || s.dead[b as usize] || s.stamp[a as usize] != entry.stamp_a || s.stamp[b as usize] != entry.stamp_b {
            continue;
        }
        let shared = s.can_collapse(a, b);
        if shared == 0 {
            continue;
        }
        let Some(at) = s.target(a, b, entry.way) else {
            continue;
        };
        if s.flips(a, b, at) || ((s.thin[a as usize] || s.thin[b as usize]) && s.crosses(a, b, at)) {
            match s.entry(a, b, entry.way + 1) {
                Some(next) if entry.way < 3 => s.heap.push(next),
                _ => {
                    s.blocked[a as usize].push(b);
                }
            }
            continue;
        }
        // Stay on the kept end when it is locked.
        let (keep, gone) = if s.locked[b as usize] { (b, a) } else { (a, b) };
        s.apply(keep, gone, at, &mut collapses, &mut alive, shared);
    }
    Run {
        points: points.to_vec(),
        tris: tris.to_vec(),
        collapses,
        faces: nf as u32,
    }
    .with_span(low, span)
}

impl Run {
    fn with_span(mut self, low: [f64; 3], span: f64) -> Run {
        for c in &mut self.collapses {
            c.at = [c.at[0] * span + low[0], c.at[1] * span + low[1], c.at[2] * span + low[2]];
        }
        self
    }

    pub fn smallest(&self) -> u32 {
        self.collapses.last().map_or(self.faces, |c| c.faces_left)
    }

    /// The first collapses that bring the piece down to `wanted` triangles: (points, triangles, the source triangle each is).
    pub fn level(&self, wanted: u32) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        let nv = self.points.len() / 3;
        let mut parent: Vec<u32> = (0..nv as u32).collect();
        let mut points = self.points.clone();
        for c in &self.collapses {
            parent[c.drop as usize] = c.keep;
            if !c.at[0].is_nan() {
                let at = c.keep as usize * 3;
                points[at..at + 3].copy_from_slice(&c.at);
            }
            if c.faces_left <= wanted {
                break;
            }
        }
        fn root(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                let up = parent[parent[x as usize] as usize];
                parent[x as usize] = up;
                x = up;
            }
            x
        }
        let mut number = vec![-1i64; nv];
        let mut flat = Vec::with_capacity(wanted as usize * 3 + 3);
        let mut owner = Vec::with_capacity(wanted as usize + 1);
        for (t, tri) in self.tris.chunks_exact(3).enumerate() {
            let (x, y, z) = (root(&mut parent, tri[0]), root(&mut parent, tri[1]), root(&mut parent, tri[2]));
            if x != y && y != z && x != z {
                flat.extend_from_slice(&[x as i64, y as i64, z as i64]);
                owner.push(t as i64);
                number[x as usize] = 0;
                number[y as usize] = 0;
                number[z as usize] = 0;
            }
        }
        let mut out = Vec::with_capacity(flat.len());
        let mut next = 0i64;
        for v in 0..nv {
            if number[v] == 0 {
                number[v] = next;
                next += 1;
                out.extend_from_slice(&points[v * 3..v * 3 + 3]);
            }
        }
        for v in &mut flat {
            *v = number[*v as usize];
        }
        (out, flat, owner)
    }
}
