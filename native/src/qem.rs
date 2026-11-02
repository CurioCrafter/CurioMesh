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
    heap: BinaryHeap<Entry>,
    scale: f64,
    mark: Vec<u32>, // scratch: neighbour marks by generation
    generation: u32,
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

    /// Would moving these two to `at` turn a triangle over or squash it? (Decimate's degenerate-flip test.)
    fn flips(&self, a: u32, b: u32, at: [f64; 3]) -> bool {
        for (v, other) in [(a, b), (b, a)] {
            let old = self.p[v as usize];
            for &t in &self.around[v as usize] {
                let tri = self.tris[t as usize];
                if tri.contains(&other) {
                    continue;
                }
                let k = tri.iter().position(|&x| x == v).unwrap();
                let (p, n) = (self.p[tri[(k + 2) % 3] as usize], self.p[tri[(k + 1) % 3] as usize]);
                let side = sub(n, p);
                let before = cross(side, sub(old, p));
                let after = cross(side, sub(at, p));
                if dot(before, after) <= (dot(before, before) + dot(after, after)) * 0.01 {
                    return true;
                }
            }
        }
        false
    }

    fn apply(&mut self, a: u32, b: u32, at: [f64; 3], collapses: &mut Vec<Collapse>, alive: &mut u32, shared: usize) {
        let (au, bu) = (a as usize, b as usize);
        let held = self.locked[au]; // a locked vertex stays exactly where it was, to the last digit: chunks join on them
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
        self.locked[au] |= self.locked[bu];
        self.dead[bu] = true;
        self.stamp[au] += 1;
        self.stamp[bu] += 1;
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
        heap: BinaryHeap::new(),
        scale: 1.0,
        mark: vec![0; nv],
        generation: 0,
    };
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
        if s.flips(a, b, at) {
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

    /// The first collapses that bring the piece down to `wanted` triangles: (points, triangles, the vertex each came from).
    pub fn level(&self, wanted: u32) -> (Vec<f64>, Vec<i64>) {
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
        for tri in self.tris.chunks_exact(3) {
            let (x, y, z) = (root(&mut parent, tri[0]), root(&mut parent, tri[1]), root(&mut parent, tri[2]));
            if x != y && y != z && x != z {
                flat.extend_from_slice(&[x as i64, y as i64, z as i64]);
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
        (out, flat)
    }
}
