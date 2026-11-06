// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! The field mesher's sweeps (`triadq/fieldmesh.py`: `_orient_sweep`, `_position_sweep`). Each vertex sums what its
//! edges say in the order NumPy's `bincount` would, so the result is the same to the last bit, on all cores.

use rayon::prelude::*;

type V3 = [f64; 3];

#[inline]
fn at(v: &[f64], i: usize) -> V3 {
    [v[3 * i], v[3 * i + 1], v[3 * i + 2]]
}
#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
#[inline]
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn tangent(v: V3, n: V3) -> V3 {
    let d = dot(v, n);
    [v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]]
}
#[inline]
fn unit(v: V3) -> V3 {
    let l = norm(v).max(1e-30);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// The directed edges (head = a then b, tail = b then a) grouped by head, each group in edge order.
struct ByHead {
    start: Vec<usize>,
    tail: Vec<u32>,
}

fn by_head(count: usize, a: &[i64], b: &[i64]) -> ByHead {
    let mut start = vec![0usize; count + 1];
    for &v in a.iter().chain(b) {
        start[v as usize + 1] += 1;
    }
    for k in 0..count {
        start[k + 1] += start[k];
    }
    let mut fill = start.clone();
    let mut tail = vec![0u32; a.len() + b.len()];
    for (&h, &t) in a.iter().zip(b).chain(b.iter().zip(a)) {
        tail[fill[h as usize]] = t as u32;
        fill[h as usize] += 1;
    }
    ByHead { start, tail }
}

/// `_best_match(qi, ni, qj, nj)[0]`.
fn best_match(qi: V3, ni: V3, qj: V3, nj: V3) -> V3 {
    let (ti, tj) = (cross(ni, qi), cross(nj, qj));
    let dots = [dot(qi, qj), dot(qi, tj), dot(ti, qj), dot(ti, tj)];
    let mut best = 0;
    for k in 1..4 {
        if dots[k].abs() > dots[best].abs() {
            best = k;
        }
    }
    let sign = if dots[best] > 0.0 {
        1.0
    } else if dots[best] < 0.0 {
        -1.0
    } else {
        0.0
    };
    let other = scale(if best % 2 == 0 { qj } else { tj }, sign);
    if best < 2 {
        other
    } else {
        cross(other, ni)
    }
}

fn each_vertex<T: Send>(count: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if count >= 4096 {
        (0..count).into_par_iter().map(f).collect()
    } else {
        (0..count).map(f).collect()
    }
}

/// One `_orient_sweep`, in place on `q`.
pub fn orient_sweep(q: &mut [f64], normals: &[f64], pinned: &[i64], a: &[i64], b: &[i64], colors: &[i64]) {
    let count = colors.len();
    let edges = by_head(count, a, b);
    for color in 0..3 {
        let new: Vec<Option<V3>> = each_vertex(count, |v| {
            if colors[v] != color {
                return None;
            }
            let (qi, ni) = (at(q, v), at(normals, v));
            let mut total = [0.0; 3];
            for &j in &edges.tail[edges.start[v]..edges.start[v + 1]] {
                let w = best_match(qi, ni, at(q, j as usize), at(normals, j as usize));
                total = [total[0] + w[0], total[1] + w[1], total[2] + w[2]];
            }
            (pinned[v] != 1 && norm(total) > 1e-12).then(|| unit(tangent(total, ni)))
        });
        for (v, value) in new.into_iter().enumerate() {
            if let Some(value) = value {
                q[3 * v..3 * v + 3].copy_from_slice(&value);
            }
        }
    }
}

#[inline]
fn nearest_grid(o: V3, q: V3, t: V3, s: f64, target: V3) -> V3 {
    let d = [target[0] - o[0], target[1] - o[1], target[2] - o[2]];
    let (u, w) = ((dot(d, q) / s).round_ties_even(), (dot(d, t) / s).round_ties_even());
    [o[0] + q[0] * u * s + t[0] * w * s, o[1] + q[1] * u * s + t[1] * w * s, o[2] + q[2] * u * s + t[2] * w * s]
}

pub struct Level<'a> {
    pub points: &'a [f64],
    pub normals: &'a [f64],
    pub q: &'a [f64],
    pub sizes: &'a [f64],
    pub pinned: &'a [i64],
}

/// `_place` for one corner.
fn place(level: &Level, v: usize, o: V3) -> V3 {
    let (p, n, q) = (at(level.points, v), at(level.normals, v), at(level.q, v));
    let t = cross(n, q);
    let lift = dot([o[0] - p[0], o[1] - p[1], o[2] - p[2]], n);
    let mut out = [o[0] - lift * n[0], o[1] - lift * n[1], o[2] - lift * n[2]];
    if level.pinned[v] == 1 {
        let along = dot([out[0] - p[0], out[1] - p[1], out[2] - p[2]], t);
        out = [out[0] - t[0] * along, out[1] - t[1] * along, out[2] - t[2] * along];
    }
    if level.pinned[v] == 2 {
        out = p;
    }
    nearest_grid(out, q, t, level.sizes[v], p)
}

/// One `_position_sweep`, in place on `o` (and on `misfit` when robust). `misfit` is empty when it is None.
#[allow(clippy::too_many_arguments)]
pub fn position_sweep(level: &Level, o: &mut [f64], misfit: &mut [f64], a: &[i64], b: &[i64], colors: &[i64], robust: bool, settled_below: f64, give: f64, confidence: bool) {
    let count = colors.len();
    let edges = by_head(count, a, b);
    let unsettled: Option<Vec<bool>> = (robust && !misfit.is_empty()).then(|| {
        let mut out: Vec<bool> = misfit.iter().map(|&m| m > settled_below).collect();
        for v in 0..count {
            if misfit[v] > settled_below {
                for &j in &edges.tail[edges.start[v]..edges.start[v + 1]] {
                    out[j as usize] = true;
                }
            }
        }
        out
    });
    let gap_of = |v: usize, j: usize, o: &[f64]| {
        let middle = scale([level.points[3 * v] + level.points[3 * j], level.points[3 * v + 1] + level.points[3 * j + 1], level.points[3 * v + 2] + level.points[3 * j + 2]], 0.5);
        let (qi, qj) = (at(level.q, v), at(level.q, j));
        let mine = nearest_grid(at(o, v), qi, cross(at(level.normals, v), qi), level.sizes[v], middle);
        let theirs = nearest_grid(at(o, j), qj, cross(at(level.normals, j), qj), level.sizes[j], mine);
        let gap = [theirs[0] - mine[0], theirs[1] - mine[1], theirs[2] - mine[2]];
        (gap, norm(gap) / level.sizes[v].min(level.sizes[j]))
    };
    for color in 0..3 {
        let active = |v: usize| colors[v] == color && unsettled.as_ref().is_none_or(|u| u[v]) && edges.start[v] < edges.start[v + 1];
        if robust && confidence {
            let heard: Vec<Option<f64>> = each_vertex(count, |v| {
                if !active(v) {
                    return None;
                }
                let tails = &edges.tail[edges.start[v]..edges.start[v + 1]];
                let total = tails.iter().fold(0.0, |sum, &j| sum + gap_of(v, j as usize, o).1);
                Some(total / tails.len() as f64)
            });
            for (v, value) in heard.into_iter().enumerate() {
                if let Some(value) = value {
                    misfit[v] = value;
                }
            }
        }
        let misfit_now: &[f64] = misfit;
        let o_now: &[f64] = o;
        let new: Vec<Option<V3>> = each_vertex(count, |v| {
            if !active(v) {
                return None;
            }
            let (mut total, mut uses) = ([0.0; 3], 0.0);
            for &j in &edges.tail[edges.start[v]..edges.start[v + 1]] {
                let (gap, relative) = gap_of(v, j as usize, o_now);
                let mut trust = if robust { 1.0 / (give + relative) } else { 1.0 };
                if robust && confidence {
                    trust /= give + misfit_now[j as usize];
                }
                total = [total[0] + gap[0] * trust, total[1] + gap[1] * trust, total[2] + gap[2] * trust];
                uses += trust;
            }
            (uses > 0.0).then(|| {
                let p = at(o_now, v);
                place(level, v, [p[0] + total[0] / uses, p[1] + total[1] / uses, p[2] + total[2] / uses])
            })
        });
        for (v, value) in new.into_iter().enumerate() {
            if let Some(value) = value {
                o[3 * v..3 * v + 3].copy_from_slice(&value);
            }
        }
    }
}

/// `fieldmesh.nearest_points` after its grid is made: per ask, the nearest point in the 27 cells around it (all points
/// if those are empty), a point facing away counting `far_side` times as far. Ties go to the first in the same order.
#[allow(clippy::too_many_arguments)]
pub fn nearest_points(points: &[f64], normals: &[f64], sorted_keys: &[i64], order: &[i64], centre: &[i64], around: &[i64], asks: &[f64], ask_normals: &[f64], far_side: f64) -> Vec<i64> {
    let one = |k: usize| {
        let (ask, facing) = (at(asks, k), at(ask_normals, k));
        let mut best = (f64::INFINITY, -1i64);
        let mut seen = false;
        let mut consider = |f: i64| {
            let p = at(points, f as usize);
            let (dx, dy, dz) = (p[0] - ask[0], p[1] - ask[1], p[2] - ask[2]);
            let mut away = dx * dx + dy * dy + dz * dz;
            if dot(at(normals, f as usize), facing) < 0.0 {
                away *= far_side;
            }
            if away < best.0 || best.1 < 0 {
                best = (away, f);
            }
        };
        for &step in around {
            let wanted = centre[k] + step;
            let left = sorted_keys.partition_point(|&x| x < wanted);
            let right = sorted_keys.partition_point(|&x| x <= wanted);
            for &f in &order[left..right] {
                seen = true;
                consider(f);
            }
        }
        if !seen {
            order.iter().for_each(|&f| consider(f));
        }
        best.1
    };
    each_vertex(centre.len(), one)
}
