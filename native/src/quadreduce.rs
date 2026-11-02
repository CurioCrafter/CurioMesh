// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! `quadreduce.QuadReducer` on flat arrays, a line-by-line mirror so the same quads go in the same order.

use crate::join::{cross, dot, normal, quality, sub};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

type P = [f64; 3];
type Quadric = [f64; 11];

const MIN_QUALITY: f64 = 0.15;
const MOST_FACES_AT_CORNER: usize = 8;

#[derive(Clone, Copy)]
pub struct Entry {
    key: f64,
    index: u32,
    corner: u8,
    a: u32,
    c: u32,
    stamp_a: u32,
    stamp_c: u32,
    place: u8,
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
    // Reversed: the heap gives the smallest tuple first, as `heapq` does.
    fn cmp(&self, o: &Self) -> Ordering {
        o.key.total_cmp(&self.key).then_with(|| (o.index, o.corner, o.a, o.c, o.stamp_a, o.stamp_c, o.place).cmp(&(self.index, self.corner, self.a, self.c, self.stamp_a, self.stamp_c, self.place)))
    }
}

/// What `save` keeps. Locks and borders are not in it, as in the Python class.
#[derive(Clone)]
pub struct Saved {
    points: Vec<P>,
    faces: Vec<Vec<u32>>, // empty: the face is gone
    users: Vec<Vec<u32>>, // sorted
    quadrics: Vec<Quadric>,
    stamps: Vec<u32>,
    heap: BinaryHeap<Entry>,
    refused: Vec<Entry>,
    alive: usize,
}

pub struct Reducer {
    s: Saved,
    locked: Vec<bool>,
    border: Vec<bool>,
    dissolve_limit: f64,
    most_turned: f64,
    result_ids: Vec<u32>,
}

fn add_unique(row: &mut Vec<u32>, x: u32) {
    if let Err(at) = row.binary_search(&x) {
        row.insert(at, x);
    }
}

fn discard(row: &mut Vec<u32>, x: u32) {
    if let Ok(at) = row.binary_search(&x) {
        row.remove(at);
    }
}

fn error(q: &Quadric, p: P) -> f64 {
    let [x, y, z] = p;
    x * (q[0] * x + 2.0 * (q[1] * y + q[2] * z + q[3])) + y * (q[4] * y + 2.0 * (q[5] * z + q[6])) + z * (q[7] * z + 2.0 * q[8]) + q[9]
}

impl Reducer {
    pub fn new(indptr: &[i64], flat: &[i64], vertices: &[f64], material: &[i64], dissolve_limit: f64, most_turned: f64) -> Self {
        let count = indptr.len() - 1;
        let nv = vertices.len() / 3;
        let points: Vec<P> = vertices.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        let faces: Vec<Vec<u32>> = (0..count).map(|f| flat[indptr[f] as usize..indptr[f + 1] as usize].iter().map(|&v| v as u32).collect()).collect();
        let mut users: Vec<Vec<u32>> = vec![Vec::new(); nv];
        let mut locked = vec![false; nv];
        let mut first_material = vec![-1i64; nv];
        let mut mixed = vec![false; nv];
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(flat.len());
        for (index, face) in faces.iter().enumerate() {
            for (k, &v) in face.iter().enumerate() {
                add_unique(&mut users[v as usize], index as u32);
                if first_material[v as usize] == -1 {
                    first_material[v as usize] = material[index];
                } else if first_material[v as usize] != material[index] {
                    mixed[v as usize] = true;
                }
                let w = face[(k + 1) % face.len()];
                edges.push(if v < w { (v, w) } else { (w, v) });
            }
            if face.len() != 4 {
                for &v in face {
                    locked[v as usize] = true;
                }
            }
        }
        crate::topo::sort(&mut edges);
        let mut border = vec![false; nv];
        let mut i = 0;
        while i < edges.len() {
            let mut j = i + 1;
            while j < edges.len() && edges[j] == edges[i] {
                j += 1;
            }
            if j - i != 2 {
                border[edges[i].0 as usize] = true;
                border[edges[i].1 as usize] = true;
            }
            i = j;
        }
        for v in 0..nv {
            if border[v] || mixed[v] {
                locked[v] = true;
            }
        }
        let mut quadrics = vec![[0.0f64; 11]; nv];
        for face in &faces {
            for k in 1..face.len() - 1 {
                let (a, b, c) = (points[face[0] as usize], points[face[k] as usize], points[face[k + 1] as usize]);
                let n = cross(sub(b, a), sub(c, a));
                let twice_area = dot(n, n).sqrt();
                if twice_area < 1e-30 {
                    continue;
                }
                let (x, y, z) = (n[0] / twice_area, n[1] / twice_area, n[2] / twice_area);
                let d = -(x * a[0] + y * a[1] + z * a[2]);
                let row = [x * x, x * y, x * z, x * d, y * y, y * z, y * d, z * z, z * d, d * d, 1.0];
                for v in [face[0], face[k], face[k + 1]] {
                    let q = &mut quadrics[v as usize];
                    for i in 0..11 {
                        q[i] += row[i] * twice_area;
                    }
                }
            }
        }
        let alive = count;
        let mut out = Reducer {
            s: Saved {
                points,
                faces,
                users,
                quadrics,
                stamps: vec![0; nv],
                heap: BinaryHeap::new(),
                refused: Vec::new(),
                alive,
            },
            locked,
            border,
            dissolve_limit,
            most_turned,
            result_ids: (0..nv as u32).collect(),
        };
        let mut start = Vec::with_capacity(count * 2);
        for index in 0..count as u32 {
            if out.s.faces[index as usize].len() == 4 {
                for corner in 0..2 {
                    if let Some(entry) = out.entry(index, corner) {
                        start.push(entry);
                    }
                }
            }
        }
        out.s.heap = BinaryHeap::from(start);
        out
    }

    fn entry(&self, index: u32, corner: u8) -> Option<Entry> {
        let face = &self.s.faces[index as usize];
        let (a, c) = (face[corner as usize], face[corner as usize + 2]);
        let (la, lc) = (self.locked[a as usize], self.locked[c as usize]);
        if la && lc {
            return None;
        }
        let (pa, pc) = (self.s.points[a as usize], self.s.points[c as usize]);
        let mut q = self.s.quadrics[a as usize];
        for (x, y) in q.iter_mut().zip(&self.s.quadrics[c as usize]) {
            *x += *y;
        }
        let (best, place) = if la {
            (error(&q, pa), 0u8)
        } else if lc {
            (error(&q, pc), 1u8)
        } else {
            let middle = [(pa[0] + pc[0]) * 0.5, (pa[1] + pc[1]) * 0.5, (pa[2] + pc[2]) * 0.5];
            let mut low = error(&q, pa);
            let mut at = 0u8;
            let e = error(&q, pc);
            if e < low {
                low = e;
                at = 1;
            }
            let e = error(&q, middle);
            if e < low {
                low = e;
                at = 2;
            }
            (low, at)
        };
        let side = sub(pa, pc);
        let floor = if best > 0.0 { best } else { 0.0 }; // max(error, 0.0)
        let divisor = if q[10] > 1e-30 { q[10] } else { 1e-30 };
        Some(Entry {
            key: (floor / divisor).sqrt() + 1e-4 * dot(side, side).sqrt(),
            index,
            corner,
            a,
            c,
            stamp_a: self.s.stamps[a as usize],
            stamp_c: self.s.stamps[c as usize],
            place,
        })
    }

    fn push(&mut self, index: u32, corner: u8) {
        if let Some(entry) = self.entry(index, corner) {
            self.s.heap.push(entry);
        }
    }

    fn around(&self, vertex: u32) -> Vec<u32> {
        let mut out = Vec::with_capacity(12);
        for &f in &self.s.users[vertex as usize] {
            let face = &self.s.faces[f as usize];
            let at = face.iter().position(|&v| v == vertex).unwrap();
            out.push(face[(at + face.len() - 1) % face.len()]);
            out.push(face[(at + 1) % face.len()]);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn try_collapse(&mut self, q: u32, corner: u8, place: u8) -> bool {
        let corner = corner as usize;
        let face = &self.s.faces[q as usize];
        let (a, b, c, d) = (face[corner], face[(corner + 1) % 4], face[(corner + 2) % 4], face[(corner + 3) % 4]);
        let (ua, uc) = (&self.s.users[a as usize], &self.s.users[c as usize]);
        // (ua & uc) != {q}: q uses both, so it is enough that nothing else does
        if ua.iter().filter(|f| uc.binary_search(f).is_ok()).count() != 1 || ua.len() + uc.len() - 2 > MOST_FACES_AT_CORNER {
            return false;
        }
        let (keep, drop) = if place == 1 { (c, a) } else { (a, c) };
        let (around_a, around_c) = (self.around(a), self.around(c));
        let mut common = around_a.iter().filter(|v| around_c.binary_search(v).is_ok()).copied();
        let mut wanted = [b, d];
        wanted.sort_unstable();
        let matches = if b == d {
            common.next() == Some(b) && common.next().is_none()
        } else {
            common.next() == Some(wanted[0]) && common.next() == Some(wanted[1]) && common.next().is_none()
        };
        if !matches {
            return false;
        }
        let (pa, pc) = (self.s.points[a as usize], self.s.points[c as usize]);
        let position = match place {
            0 => pa,
            1 => pc,
            _ => [(pa[0] + pc[0]) * 0.5, (pa[1] + pc[1]) * 0.5, (pa[2] + pc[2]) * 0.5],
        };
        let renamed = |corners: &[u32]| -> Vec<u32> { corners.iter().map(|&v| if v == drop { keep } else { v }).collect() };

        let mut moved: Vec<(u32, Vec<u32>)> = Vec::with_capacity(8);
        for &f in ua.iter().chain(uc.iter()) {
            if f != q {
                moved.push((f, renamed(&self.s.faces[f as usize])));
            }
        }
        let mut merged_with: Vec<(u32, u32, u32)> = Vec::new(); // kept face, face merged into it, corner that left
        for side in [b, d] {
            let row = &self.s.users[side as usize];
            let count = row.len();
            if self.border[side as usize] {
                if count < 2 {
                    return false;
                }
            } else if count < 3 {
                return false;
            } else if count == 3 {
                // It would be left in two quads: they become one.
                let mut others = row.iter().copied().filter(|&f| f != q);
                let (first, second) = (others.next().unwrap(), others.next().unwrap());
                if merged_with.iter().any(|m| m.0 == first || m.0 == second) {
                    return false; // both sides share a quad, which can't merge twice in one removal
                }
                let in_moved = |f: u32, moved: &Vec<(u32, Vec<u32>)>| moved.iter().position(|m| m.0 == f);
                let (Some(first_at), Some(second_at)) = (in_moved(first, &moved), in_moved(second, &moved)) else {
                    return false;
                };
                if self.locked[side as usize] || self.s.faces[first as usize].len() != 4 || self.s.faces[second as usize].len() != 4 {
                    return false;
                }
                let turn = |corners: &[u32]| -> Vec<u32> {
                    let mut out = renamed(corners);
                    let at = out.iter().position(|&v| v == side).unwrap();
                    out.rotate_left(at);
                    out
                };
                let (one, two) = (turn(&self.s.faces[first as usize]), turn(&self.s.faces[second as usize]));
                if one[1] != two[3] || one[3] != two[1] || one[2] == two[2] {
                    return false;
                }
                let other = if one[1] == keep { one[3] } else { one[1] };
                if !self.border[other as usize] && self.s.users[other as usize].len() <= 3 {
                    return false;
                }
                moved[first_at].1 = vec![one[1], one[2], one[3], two[2]];
                moved.remove(second_at);
                match merged_with.iter_mut().find(|m| m.0 == first) {
                    Some(m) => *m = (first, second, side),
                    None => merged_with.push((first, second, side)),
                }
            }
        }
        if !self.border[keep as usize] && moved.iter().filter(|m| m.1.contains(&keep)).count() < 3 {
            return false;
        }
        let mut seen: Vec<Vec<u32>> = Vec::with_capacity(moved.len());
        for (f, corners) in &moved {
            let mut shape = corners.clone();
            shape.sort_unstable();
            if shape.windows(2).any(|w| w[0] == w[1]) || seen.contains(&shape) {
                return false;
            }
            seen.push(shape);
            if corners.len() != 4 {
                continue;
            }
            let point = |v: u32| if v == keep { position } else { self.s.points[v as usize] };
            let new = [point(corners[0]), point(corners[1]), point(corners[2]), point(corners[3])];
            if quality(&new) < MIN_QUALITY {
                return false;
            }
            let turned = normal(&new);
            let was = |face: &Vec<u32>| normal(&[self.s.points[face[0] as usize], self.s.points[face[1] as usize], self.s.points[face[2] as usize], self.s.points[face[3] as usize]]);
            if dot(turned, was(&self.s.faces[*f as usize])) < self.most_turned {
                return false;
            }
            if let Some(m) = merged_with.iter().find(|m| m.0 == *f) {
                if dot(turned, was(&self.s.faces[m.1 as usize])) < self.most_turned {
                    return false;
                }
                if dot(turned, sub(self.s.points[m.2 as usize], new[0])).abs() > self.dissolve_limit {
                    return false;
                }
            }
        }

        // Everything holds: apply it.
        let mut total = self.s.quadrics[a as usize];
        for (x, y) in total.iter_mut().zip(&self.s.quadrics[c as usize]) {
            *x += *y;
        }
        let gone: Vec<u32> = std::iter::once(q).chain(merged_with.iter().map(|m| m.1)).collect();
        for f in gone {
            let face = std::mem::take(&mut self.s.faces[f as usize]);
            for v in face {
                discard(&mut self.s.users[v as usize], f);
            }
            self.s.alive -= 1;
        }
        for (f, corners) in moved {
            let face = std::mem::replace(&mut self.s.faces[f as usize], corners);
            for v in face {
                discard(&mut self.s.users[v as usize], f);
            }
            let row = self.s.faces[f as usize].clone();
            for v in row {
                add_unique(&mut self.s.users[v as usize], f);
            }
        }
        for m in &merged_with {
            for (x, y) in total.iter_mut().zip(&self.s.quadrics[m.2 as usize]) {
                *x = *y + *x;
            }
        }
        self.s.quadrics[keep as usize] = total;
        self.s.points[keep as usize] = position;
        self.s.stamps[keep as usize] += 1;
        self.s.stamps[drop as usize] += 1;
        let row = self.s.users[keep as usize].clone();
        for f in row {
            let face = &self.s.faces[f as usize];
            if face.len() == 4 {
                let at = face.iter().position(|&v| v == keep).unwrap();
                self.push(f, (at % 2) as u8);
            }
        }
        for m in &merged_with {
            self.push(m.0, 0);
            self.push(m.0, 1);
        }
        true
    }

    /// `remove`: take away up to `count` quads, cheapest first.
    pub fn remove(&mut self, count: usize) {
        let goal = self.s.alive.saturating_sub(count);
        while self.s.alive > goal {
            let Some(entry) = self.s.heap.pop() else {
                break;
            };
            let (q, corner, a, c) = (entry.index, entry.corner as usize, entry.a, entry.c);
            let face = &self.s.faces[q as usize];
            if face.is_empty() || face[corner] != a || face[corner + 2] != c || self.s.stamps[a as usize] != entry.stamp_a || self.s.stamps[c as usize] != entry.stamp_c {
                continue;
            }
            let (la, lc) = (self.locked[a as usize], self.locked[c as usize]);
            if la && lc {
                continue;
            }
            // On a flat grid only the middle works, though every place has the same error.
            let mut done = self.try_collapse(q, entry.corner, entry.place);
            if !done && !la && !lc {
                for k in [2u8, 0, 1] {
                    if k != entry.place && self.try_collapse(q, entry.corner, k) {
                        done = true;
                        break;
                    }
                }
            }
            if !done {
                self.s.refused.push(entry);
            }
        }
    }

    pub fn alive(&self) -> usize {
        self.s.alive
    }

    pub fn refused(&self) -> usize {
        self.s.refused.len()
    }

    pub fn retry_refused(&mut self) {
        let refused = std::mem::take(&mut self.s.refused);
        self.s.heap.extend(refused);
    }

    pub fn original_ids(&self, vertices: &[i64]) -> Vec<i64> {
        vertices.iter().map(|&v| self.result_ids[v as usize] as i64).collect()
    }

    /// Nothing more comes off the connected parts holding these vertices (numbered in this reducer's own numbering).
    pub fn freeze(&mut self, vertices: &[i64]) {
        let mut seen = vec![false; self.s.points.len()];
        let mut stack: Vec<u32> = vertices.iter().map(|&v| v as u32).collect();
        for &v in &stack {
            seen[v as usize] = true;
        }
        while let Some(vertex) = stack.pop() {
            self.locked[vertex as usize] = true;
            for &f in &self.s.users[vertex as usize] {
                for &other in &self.s.faces[f as usize] {
                    if !seen[other as usize] {
                        seen[other as usize] = true;
                        stack.push(other);
                    }
                }
            }
        }
    }

    pub fn save(&self) -> Saved {
        self.s.clone()
    }

    pub fn restore(&mut self, saved: &Saved) {
        self.s = saved.clone();
    }

    /// `result`: (vertices, indptr, flat, the input face each face stands for).
    pub fn result(&mut self) -> (Vec<f64>, Vec<i64>, Vec<i64>, Vec<i64>) {
        let mut used = vec![false; self.s.points.len()];
        let (mut indptr, mut flat, mut owner) = (vec![0i64], Vec::new(), Vec::new());
        for (k, face) in self.s.faces.iter().enumerate() {
            if !face.is_empty() {
                for &v in face {
                    used[v as usize] = true;
                    flat.push(v as i64);
                }
                indptr.push(flat.len() as i64);
                owner.push(k as i64);
            }
        }
        let mut table = vec![0i64; used.len()];
        let mut ids = Vec::new();
        let mut vertices = Vec::new();
        for (v, &on) in used.iter().enumerate() {
            if on {
                table[v] = ids.len() as i64;
                ids.push(v as u32);
                vertices.extend_from_slice(&self.s.points[v]);
            }
        }
        for v in &mut flat {
            *v = table[*v as usize];
        }
        self.result_ids = ids;
        (vertices, indptr, flat, owner)
    }
}
