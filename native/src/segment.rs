// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! `segment._merge_all`'s loop: merge the cheapest neighbouring patches first, until each loose piece is one. The same
//! prices, in the same arithmetic (`x ** 0.5` as `powf`), popped in the same order as a heap of Python tuples.

use std::cmp::Ordering as Order;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Weights {
    pub convex: f64,
    pub usual: f64,
    pub curve_step: f64,
    pub thick_step: f64,
    pub neck: f64,
    pub base: f64,
}

#[derive(PartialEq)]
struct Item(f64, u32, u32, u32, u32);

impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Order> {
        Some(self.cmp(other))
    }
}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> Order {
        // Reversed: BinaryHeap pops the largest, heapq the smallest tuple.
        other.0.total_cmp(&self.0).then((other.1, other.2, other.3, other.4).cmp(&(self.1, self.2, self.3, self.4)))
    }
}

type Near = Vec<HashMap<u32, [f64; 3]>>;

/// (merges as (kept, absorbed), cost after each), or None when stopped.
#[allow(clippy::too_many_arguments)]
pub fn merge_all(count: usize, pa: &[i64], pb: &[i64], rows: &[[f64; 3]], area: &[f64], curve: &[f64], thick: &[f64], whole: f64, w: &Weights, stop: &AtomicBool) -> Option<(Vec<(i64, i64)>, Vec<f64>)> {
    let mut near: Near = vec![HashMap::new(); count];
    for ((&x, &y), row) in pa.iter().zip(pb).zip(rows) {
        near[x as usize].insert(y as u32, *row);
        near[y as usize].insert(x as u32, *row);
    }
    let (mut area, mut curve, mut thick) = (area.to_vec(), curve.to_vec(), thick.to_vec());
    let mut version = vec![0u32; count];
    let half = std::hint::black_box(0.5); // a plain pow, as Python calls it: the compiler would make pow(x, 0.5) a square root
    let cost = |near: &Near, area: &[f64], curve: &[f64], thick: &[f64], i: usize, j: usize| {
        let row = near[i][&(j as u32)];
        let (ai, aj) = (area[i].max(1e-30), area[j].max(1e-30));
        let (inside_i, inside_j) = (curve[i] / ai, curve[j] / aj);
        let bend = ((row[1] + w.convex * row[2]) / row[0].max(1e-30) - w.usual * inside_i.min(inside_j)).max(0.0);
        let step = w.curve_step * (inside_i - inside_j).abs() + w.thick_step * (thick[i] / ai - thick[j] / aj).abs();
        let small = ai.min(aj);
        let neck = 1.05 - (row[0] / (w.neck * small.powf(half))).min(1.0);
        (w.base + bend + step) * (small / whole.max(1e-30)).powf(half) * neck
    };
    let mut heap: BinaryHeap<Item> = pa.iter().zip(pb).map(|(&x, &y)| Item(cost(&near, &area, &curve, &thick, x as usize, y as usize), x as u32, y as u32, 0, 0)).collect();
    let (mut merges, mut costs, mut top) = (Vec::new(), Vec::new(), 0.0f64);
    while let Some(Item(price, i, j, vi, vj)) = heap.pop() {
        let (mut i, mut j) = (i as usize, j as usize);
        if vi != version[i] || vj != version[j] || !near[i].contains_key(&(j as u32)) {
            continue;
        }
        if merges.len() % 512 == 0 && stop.load(Ordering::Relaxed) {
            return None;
        }
        top = top.max(price);
        if near[i].len() < near[j].len() {
            std::mem::swap(&mut i, &mut j);
        }
        merges.push((i as i64, j as i64));
        costs.push(top);
        near[i].remove(&(j as u32));
        near[j].remove(&(i as u32));
        area[i] += area[j];
        curve[i] += curve[j];
        thick[i] += thick[j];
        version[i] += 1;
        for (k, row) in std::mem::take(&mut near[j]) {
            let k = k as usize;
            near[k].remove(&(j as u32));
            let joined = match near[i].get(&(k as u32)) {
                None => row,
                Some(mine) => [mine[0] + row[0], mine[1] + row[1], mine[2] + row[2]],
            };
            near[i].insert(k as u32, joined);
            near[k].insert(i as u32, joined);
        }
        let around: Vec<u32> = near[i].keys().copied().collect();
        for k in around {
            let k = k as usize;
            heap.push(Item(cost(&near, &area, &curve, &thick, i, k), i as u32, k as u32, version[i], version[k]));
        }
    }
    Some((merges, costs))
}
