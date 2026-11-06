// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Guided edge loops: strokes drawn on the model (round an eye, along a joint, across a panel) that the result must
//! carry as edge loops. The add-on's `guides.py`: each stroke is laid on the surface, the reduced triangles near it are
//! split until they are small, the cheapest chain of edges that hugs it is found and its corners pulled onto it, and
//! those edges go to the pairing as guide edges, which no quad may cross.

use crate::near::{FaceTree, PointTree};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

type V3 = [f64; 3];
pub type Edge = (i64, i64);

const STEP: f64 = 1.0 / 200.0; // strokes are laid on the surface at points this share of the model's size apart
const OFF_MODEL: f64 = 0.02; // a stroke point farther than this share of the size from the surface is off the model
const SHORTEST: f64 = 0.02; // a stroke (or what is left of it on the model) shorter than this share of the size is ignored
const REFINE_PASSES: usize = 6;
const LONG: f64 = 1.5; // triangles near a stroke are split until their edges are under this many spacings
const LEG: usize = 6; // samples between the waypoints a chain must pass
const HUG: f64 = 8.0; // how much an edge costs extra per spacing it lies away from the stroke
const PULL: f64 = 0.45; // a corner moves onto the stroke by at most this share of its shortest edge: no triangle flips
const FOLLOW_STEP: f64 = 1.0 / 400.0; // points along a stroke the follow numbers are measured at

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dist(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}
fn at(points: &[f64], v: i64) -> V3 {
    let i = v as usize * 3;
    [points[i], points[i + 1], points[i + 2]]
}
fn key(a: i64, b: i64) -> Edge {
    (a.min(b), a.max(b))
}

/// The model's size: its bounding box's diagonal (what the follow numbers are a share of).
pub fn size_of(points: &[f64]) -> f64 {
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in points.chunks_exact(3) {
        for c in 0..3 {
            low[c] = low[c].min(p[c]);
            high[c] = high[c].max(p[c]);
        }
    }
    let d = dist(low, high);
    if d.is_finite() && d > 0.0 { d } else { 1.0 }
}

fn length(line: &[V3]) -> f64 {
    line.windows(2).map(|w| dist(w[0], w[1])).sum()
}

/// Points along a polyline, `step` apart, the last point kept.
pub fn resample(line: &[V3], step: f64) -> Vec<V3> {
    let mut out = Vec::new();
    let Some(&first) = line.first() else { return out };
    out.push(first);
    let step = step.max(1e-12);
    let mut left = step; // distance to the next sample
    for w in line.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mut along = 0.0;
        let seg = dist(a, b);
        while seg - along >= left {
            along += left;
            let t = along / seg;
            out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
            left = step;
        }
        left -= seg - along;
    }
    let last = *line.last().unwrap();
    if dist(*out.last().unwrap(), last) > 1e-12 * step {
        out.push(last);
    }
    out
}

/// The strokes laid onto the surface, `size` being the model's size: (pieces, a note per stroke left out). A stroke is
/// cut where it leaves the model or jumps across a gap; pieces too short to carry a loop are dropped. A stroke whose ends
/// meet is closed, so its loop goes all the way round.
pub fn lay(strokes: &[Vec<V3>], tree: &FaceTree, size: f64) -> (Vec<Vec<V3>>, Vec<String>) {
    let step = STEP * size;
    let (mut out, mut notes) = (Vec::new(), Vec::new());
    for (i, stroke) in strokes.iter().enumerate() {
        let points = resample(stroke, step);
        let (mut pieces, mut run, mut off) = (Vec::new(), Vec::new(), 0usize);
        for &p in &points {
            let Some((q, _, d2)) = tree.nearest(p) else { break };
            let jump = run.last().is_some_and(|&last| dist(last, q) > 6.0 * step);
            if d2.sqrt() > OFF_MODEL * size || jump {
                pieces.push(std::mem::take(&mut run));
                if !jump {
                    off += 1;
                    continue;
                }
            }
            run.push(q);
        }
        pieces.push(run);
        let kept: Vec<Vec<V3>> = pieces.into_iter().filter(|p| p.len() > 1 && length(p) >= SHORTEST * size).collect();
        if kept.is_empty() {
            let why = if 2 * off >= points.len() { "it is off the model" } else { "it is too short" };
            notes.push(format!("Stroke {} left out: {why}", i + 1));
            continue;
        }
        for mut piece in kept {
            let (first, last) = (piece[0], *piece.last().unwrap());
            if dist(first, last) < 3.0 * step && length(&piece) > 12.0 * step {
                piece.push(first);
            }
            out.push(piece);
        }
    }
    (out, notes)
}

/// The strokes moved to the side of a mirror plane that is solved (larger coordinates on `axis`): each stroke is cut where
/// it crosses the plane, and the pieces on the other side are reflected.
pub fn to_side(strokes: &[Vec<V3>], axis: usize, offset: f64) -> Vec<Vec<V3>> {
    let mut out = Vec::new();
    for stroke in strokes {
        let mut run: Vec<V3> = Vec::new();
        let mut side = None;
        for &p in stroke {
            let here = p[axis] >= offset;
            if side.is_some_and(|s| s != here) && !run.is_empty() {
                let last = *run.last().unwrap();
                let t = (offset - last[axis]) / (p[axis] - last[axis]);
                let mut cut = [last[0] + (p[0] - last[0]) * t, last[1] + (p[1] - last[1]) * t, last[2] + (p[2] - last[2]) * t];
                cut[axis] = offset;
                run.push(cut);
                out.push(std::mem::replace(&mut run, vec![cut]));
            }
            side = Some(here);
            run.push(p);
        }
        out.push(run);
    }
    out.into_iter().filter(|r| r.len() > 1).map(|r| r.into_iter().map(|mut p| {
        if p[axis] < offset {
            p[axis] = 2.0 * offset - p[axis];
        }
        p
    }).collect()).collect()
}

// ------------------------------------------------------------------------------------- cutting the strokes in

fn samples_of(lines: &[Vec<V3>], step: f64) -> Vec<V3> {
    lines.iter().flat_map(|l| resample(l, step)).collect()
}

fn tree_of(points: &[V3]) -> PointTree {
    PointTree::new(&points.iter().flatten().copied().collect::<Vec<f64>>())
}

/// Split the given edges at their middles (put back on the surface unless on a border); the triangles beside them are
/// split to match, keeping their winding.
fn split(points: &mut Vec<f64>, tris: &[i64], tags: &[i64], keys: &HashSet<Edge>, surface: &FaceTree) -> (Vec<i64>, Vec<i64>) {
    let mut uses: HashMap<Edge, usize> = keys.iter().map(|&k| (k, 0)).collect();
    for t in tris.chunks_exact(3) {
        for c in 0..3 {
            if let Some(n) = uses.get_mut(&key(t[c], t[(c + 1) % 3])) {
                *n += 1;
            }
        }
    }
    let mut sorted: Vec<Edge> = keys.iter().copied().collect();
    sorted.sort_unstable();
    let mut mid = HashMap::with_capacity(sorted.len());
    for k in sorted {
        let (a, b) = (at(points, k.0), at(points, k.1));
        let mut p = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0];
        if uses[&k] > 1 {
            if let Some((q, _, _)) = surface.nearest(p) {
                p = q;
            }
        }
        mid.insert(k, (points.len() / 3) as i64);
        points.extend(p);
    }
    let (mut out, mut out_tags) = (Vec::with_capacity(tris.len() * 2), Vec::with_capacity(tags.len() * 2));
    for (t, &tag) in tris.chunks_exact(3).zip(tags) {
        let vm: Vec<Option<i64>> = (0..3).map(|k| mid.get(&key(t[k], t[(k + 1) % 3])).copied()).collect();
        let n = vm.iter().filter(|m| m.is_some()).count();
        let k = match n {
            0 => {
                out.extend_from_slice(t);
                out_tags.push(tag);
                continue;
            }
            1 => vm.iter().position(|m| m.is_some()).unwrap(),
            2 => (vm.iter().position(|m| m.is_none()).unwrap() + 1) % 3,
            _ => 0,
        };
        let (p0, p1, p2) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
        let (m01, m12, m20) = (vm[k], vm[(k + 1) % 3], vm[(k + 2) % 3]);
        let new: Vec<[i64; 3]> = match n {
            1 => vec![[p0, m01.unwrap(), p2], [m01.unwrap(), p1, p2]],
            2 => { let (a, b) = (m01.unwrap(), m12.unwrap()); vec![[a, p1, b], [p0, a, p2], [a, b, p2]] }
            _ => { let (a, b, c) = (m01.unwrap(), m12.unwrap(), m20.unwrap()); vec![[p0, a, c], [a, p1, b], [c, b, p2], [a, b, c]] }
        };
        for tri in new {
            out.extend(tri);
            out_tags.push(tag);
        }
    }
    (out, out_tags)
}

/// Cheapest path of corners from `start` to `goal`; edges in `banned` cost 40 times more (a chain doesn't run back on itself).
fn path(adj: &HashMap<i64, Vec<(i64, f64)>>, start: i64, goal: i64, banned: &HashSet<Edge>) -> Option<Vec<i64>> {
    let mut best: HashMap<i64, f64> = HashMap::from([(start, 0.0)]);
    let mut came: HashMap<i64, i64> = HashMap::new();
    let mut heap = BinaryHeap::from([Reverse((0u64, start))]);
    while let Some(Reverse((bits, u))) = heap.pop() {
        let c = f64::from_bits(bits);
        if u == goal {
            let mut out = vec![u];
            while *out.last().unwrap() != start {
                out.push(came[out.last().unwrap()]);
            }
            out.reverse();
            return Some(out);
        }
        if c > best.get(&u).copied().unwrap_or(f64::INFINITY) {
            continue;
        }
        for &(w, cost) in adj.get(&u).map_or(&[][..], |v| v.as_slice()) {
            let step = c + cost * if banned.contains(&key(u, w)) { 40.0 } else { 1.0 };
            if step < best.get(&w).copied().unwrap_or(f64::INFINITY) {
                best.insert(w, step);
                came.insert(w, u);
                heap.push(Reverse((step.to_bits(), w))); // costs are positive, so their bits sort like the numbers
            }
        }
    }
    None
}

/// The reduced triangles with the strokes cut in: (points, triangles, tags, guide edges, strokes that found no chain).
/// `surface` is the source, which new corners are put back on.
pub fn cut_in(mut points: Vec<f64>, mut tris: Vec<i64>, mut tags: Vec<i64>, lines: &[Vec<V3>], surface: &FaceTree) -> (Vec<f64>, Vec<i64>, Vec<i64>, HashSet<Edge>, usize) {
    let mut lengths: Vec<f64> = tris.chunks_exact(3).flat_map(|t| (0..3).map(|k| dist(at(&points, t[k]), at(&points, t[(k + 1) % 3])))).collect();
    if lengths.is_empty() || lines.is_empty() {
        return (points, tris, tags, HashSet::new(), lines.len());
    }
    let middle = lengths.len() / 2;
    let spacing = *lengths.select_nth_unstable_by(middle, f64::total_cmp).1;
    // Split the triangles near the strokes until their longest edge is under LONG spacings.
    let near_tree = tree_of(&samples_of(lines, spacing / 2.0));
    for _ in 0..REFINE_PASSES {
        let mut keys = HashSet::new();
        for t in tris.chunks_exact(3) {
            let c = [at(&points, t[0]), at(&points, t[1]), at(&points, t[2])];
            let (k, longest) = (0..3).map(|k| (k, dist(c[k], c[(k + 1) % 3]))).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            let centre = [(c[0][0] + c[1][0] + c[2][0]) / 3.0, (c[0][1] + c[1][1] + c[2][1]) / 3.0, (c[0][2] + c[1][2] + c[2][2]) / 3.0];
            if longest > LONG * spacing && near_tree.nearest(centre).1.sqrt() < longest {
                keys.insert(key(t[k], t[(k + 1) % 3]));
            }
        }
        if keys.is_empty() {
            break;
        }
        (tris, tags) = split(&mut points, &tris, &tags, &keys, surface);
    }
    // Every edge once, with how many triangles use it.
    let mut uses: HashMap<Edge, usize> = HashMap::with_capacity(tris.len());
    for t in tris.chunks_exact(3) {
        for k in 0..3 {
            *uses.entry(key(t[k], t[(k + 1) % 3])).or_default() += 1;
        }
    }
    let mut edges: Vec<Edge> = uses.keys().copied().collect();
    edges.sort_unstable();
    let mut shortest = HashMap::<i64, f64>::new();
    let mut border = HashSet::new();
    for &(a, b) in &edges {
        let l = dist(at(&points, a), at(&points, b));
        for v in [a, b] {
            let s = shortest.entry(v).or_insert(f64::INFINITY);
            *s = s.min(l);
        }
        if uses[&(a, b)] == 1 {
            border.extend([a, b]);
        }
    }
    let (mut guide, mut missed) = (HashSet::new(), 0);
    for line in lines {
        let samples = resample(line, spacing / 2.0);
        let tree = tree_of(&samples);
        // The edges near the stroke, costing more the farther they lie from it.
        let mut adj: HashMap<i64, Vec<(i64, f64)>> = HashMap::new();
        for &(a, b) in &edges {
            let (p, q) = (at(&points, a), at(&points, b));
            let away = tree.nearest([(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0, (p[2] + q[2]) / 2.0]).1.sqrt();
            if away < 3.0 * spacing {
                let cost = dist(p, q) * (1.0 + HUG * away / spacing) + 1e-12;
                adj.entry(a).or_default().push((b, cost));
                adj.entry(b).or_default().push((a, cost));
            }
        }
        if adj.is_empty() {
            missed += 1;
            continue;
        }
        let mut corners: Vec<i64> = adj.keys().copied().collect();
        corners.sort_unstable();
        let corner_tree = tree_of(&corners.iter().map(|&v| at(&points, v)).collect::<Vec<_>>());
        let mut stops: Vec<usize> = (0..samples.len()).step_by(LEG).collect();
        if *stops.last().unwrap() != samples.len() - 1 {
            stops.push(samples.len() - 1);
        }
        let waypoints: Vec<i64> = stops.iter().map(|&i| corners[corner_tree.nearest(samples[i]).0 as usize]).collect();
        // The chain: cheapest paths from waypoint to waypoint (a leg with no path leaves a gap).
        let mut banned = HashSet::new();
        for w in waypoints.windows(2) {
            if w[0] != w[1] {
                for x in path(&adj, w[0], w[1], &banned).unwrap_or_default().windows(2) {
                    banned.insert(key(x[0], x[1]));
                }
            }
        }
        if banned.is_empty() {
            missed += 1;
            continue;
        }
        guide.extend(banned.iter().copied());
        // Pull the chain's corners onto the stroke, a little at a time at most, borders left alone.
        let fine = resample(line, spacing / 4.0);
        let fine_tree = tree_of(&fine);
        let mut moved = HashSet::new();
        for &(a, b) in &banned {
            for v in [a, b] {
                if border.contains(&v) || !moved.insert(v) {
                    continue;
                }
                let p = at(&points, v);
                let (i, d2) = fine_tree.nearest(p);
                let d = d2.sqrt();
                let share = (PULL * shortest[&v] / d.max(1e-30)).min(1.0);
                let target = fine[i as usize];
                for c in 0..3 {
                    points[v as usize * 3 + c] += (target[c] - p[c]) * share;
                }
            }
        }
    }
    (points, tris, tags, guide, missed)
}

// ------------------------------------------------------------------------------------- how closely the result follows

fn to_segment(x: V3, p: V3, q: V3) -> f64 {
    let d = sub(q, p);
    let dd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).max(1e-30);
    let w = sub(x, p);
    let t = ((w[0] * d[0] + w[1] * d[1] + w[2] * d[2]) / dd).clamp(0.0, 1.0);
    dist(x, [p[0] + d[0] * t, p[1] + d[1] * t, p[2] + d[2] * t])
}

/// (mean, worst) distance from points along the strokes to the nearest edge of `mesh`, as shares of `size`.
/// ponytail: every point against every edge near the strokes' box; a tree over the edges if strokes get long on huge meshes.
pub fn follow(lines: &[Vec<V3>], mesh: &crate::io::Mesh, size: f64) -> (f64, f64) {
    use rayon::prelude::*;
    let pts: Vec<V3> = samples_of(lines, FOLLOW_STEP * size);
    if pts.is_empty() {
        return (0.0, 0.0);
    }
    let mut seen = HashSet::new();
    for f in 0..mesh.faces() {
        let face = mesh.face(f);
        for k in 0..face.len() {
            seen.insert(key(face[k], face[(k + 1) % face.len()]));
        }
    }
    let segs: Vec<(V3, V3)> = seen.into_iter().map(|(a, b)| (at(&mesh.points, a), at(&mesh.points, b))).collect();
    let best: Vec<f64> = pts.par_iter().map(|&x| segs.iter().map(|&(p, q)| to_segment(x, p, q)).fold(f64::INFINITY, f64::min)).collect();
    let mean = best.iter().sum::<f64>() / best.len() as f64;
    let worst = best.iter().cloned().fold(0.0, f64::max);
    (mean / size, worst / size)
}

/// "Loops followed: mean 0.3%, worst 1.2% of the model's size".
pub fn follow_text(follow: (f64, f64)) -> String {
    format!("Loops followed: mean {:.1}%, worst {:.1}% of the model's size", 100.0 * follow.0, 100.0 * follow.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_the_ends_and_the_spacing() {
        let line = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let r = resample(&line, 0.25);
        assert_eq!(r.len(), 9);
        assert_eq!(*r.last().unwrap(), [1.0, 1.0, 0.0]);
        assert!(r.windows(2).all(|w| (dist(w[0], w[1]) - 0.25).abs() < 1e-9));
    }

    #[test]
    fn strokes_across_a_mirror_plane_are_cut_and_reflected() {
        let line = vec![[-1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let out = to_side(&[line], 0, 0.0);
        assert_eq!(out.len(), 2);
        assert!(out.iter().flatten().all(|p| p[0] >= 0.0));
    }
}
