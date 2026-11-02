// SPDX-License-Identifier: GPL-3.0-or-later
//! Our own lookup trees, in place of Blender's one-call-at-a-time KDTree and BVHTree: nearest points, nearest faces and
//! faces passing through each other, all cores, a batch at a time, and each batch stops early when the stop flag is set.
//!
//! Exact ties go to the lowest index, so a result never depends on the thread count or the tree's layout. Blender breaks
//! ties by its own tree order. `FaceTree` keeps coordinates in single precision, as Blender's BVH does.

use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

const BLOCK: usize = 512; // queries between looks at the stop flag
const PARALLEL_MIN: usize = 4096; // fewer queries than this aren't worth waking the thread pool: see `topo::PARALLEL_MIN`
const SIDE_BY_SIDE: usize = 8192; // ranges below this are built on one thread
const LEAF_POINTS: usize = 8;
const LEAF_TRIS: usize = 4;

type V3 = [f64; 3];

/// Run `one(index, row)` for each row of `out` (rows of `width`), all cores. False when stopped: the output is then incomplete.
pub fn each_row<T: Send>(out: &mut [T], width: usize, stop: &AtomicBool, one: impl Fn(usize, &mut [T]) + Sync) -> bool {
    let run = |(block, rows): (usize, &mut [T])| {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        for (k, row) in rows.chunks_mut(width).enumerate() {
            one(block * BLOCK + k, row);
        }
    };
    if out.len() / width >= PARALLEL_MIN {
        out.par_chunks_mut(width * BLOCK).enumerate().for_each(run);
    } else {
        out.chunks_mut(width * BLOCK).enumerate().for_each(run);
    }
    !stop.load(Ordering::Relaxed)
}

#[inline]
fn point(flat: &[f64], i: usize) -> V3 {
    [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]]
}

#[inline]
fn gap2(a: V3, b: V3) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

// --------------------------------------------------------------------------- points

/// A balanced kd-tree in one array: a range's middle item is its node, its two sides are the rest of the range.
pub struct PointTree {
    points: Vec<V3>,
    ids: Vec<u32>,
    axis: Vec<u8>,
}

fn sort_points(order: &mut [u32], axis: &mut [u8], flat: &[f64]) {
    if order.len() <= LEAF_POINTS {
        return;
    }
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &i in order.iter() {
        let p = point(flat, i as usize);
        for k in 0..3 {
            low[k] = low[k].min(p[k]);
            high[k] = high[k].max(p[k]);
        }
    }
    let wide = (0..3).fold(0, |best, k| if high[k] - low[k] > high[best] - low[best] { k } else { best });
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |&a, &b| flat[3 * a as usize + wide].total_cmp(&flat[3 * b as usize + wide]));
    axis[mid] = wide as u8;
    let (order_a, order_b) = order.split_at_mut(mid);
    let (axis_a, axis_b) = axis.split_at_mut(mid);
    let (order_b, axis_b) = (&mut order_b[1..], &mut axis_b[1..]);
    if order_a.len() >= SIDE_BY_SIDE {
        rayon::join(|| sort_points(order_a, axis_a, flat), || sort_points(order_b, axis_b, flat));
    } else {
        sort_points(order_a, axis_a, flat);
        sort_points(order_b, axis_b, flat);
    }
}

impl PointTree {
    pub fn new(flat: &[f64]) -> Self {
        let count = flat.len() / 3;
        let mut ids: Vec<u32> = (0..count as u32).collect();
        let mut axis = vec![0u8; count];
        sort_points(&mut ids, &mut axis, flat);
        let points = ids.iter().map(|&i| point(flat, i as usize)).collect();
        PointTree { points, ids, axis }
    }

    fn nearest_in(&self, lo: usize, hi: usize, q: V3, best: &mut (f64, u32)) {
        let mut consider = |k: usize| {
            let d = gap2(self.points[k], q);
            if d < best.0 || (d == best.0 && self.ids[k] < best.1) {
                *best = (d, self.ids[k]);
            }
        };
        if hi - lo <= LEAF_POINTS {
            (lo..hi).for_each(consider);
            return;
        }
        let mid = lo + (hi - lo) / 2;
        consider(mid);
        let axis = self.axis[mid] as usize;
        let off = q[axis] - self.points[mid][axis];
        let (first, second) = if off < 0.0 { ((lo, mid), (mid + 1, hi)) } else { ((mid + 1, hi), (lo, mid)) };
        self.nearest_in(first.0, first.1, q, best);
        if off * off <= best.0 {
            self.nearest_in(second.0, second.1, q, best);
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// (index, squared distance), or (u32::MAX, infinity) on an empty tree.
    pub fn nearest(&self, q: V3) -> (u32, f64) {
        let mut best = (f64::INFINITY, u32::MAX);
        self.nearest_in(0, self.ids.len(), q, &mut best);
        (best.1, best.0)
    }

    fn k_nearest_in(&self, lo: usize, hi: usize, q: V3, found: &mut Vec<(f64, u32)>, k: usize) {
        let mut consider = |at: usize| {
            let item = (gap2(self.points[at], q), self.ids[at]);
            if found.len() == k {
                if (item.0, item.1) >= *found.last().unwrap() {
                    return;
                }
                found.pop();
            }
            let place = found.partition_point(|have| *have < item);
            found.insert(place, item);
        };
        if hi - lo <= LEAF_POINTS {
            (lo..hi).for_each(consider);
            return;
        }
        let mid = lo + (hi - lo) / 2;
        consider(mid);
        let axis = self.axis[mid] as usize;
        let off = q[axis] - self.points[mid][axis];
        let (first, second) = if off < 0.0 { ((lo, mid), (mid + 1, hi)) } else { ((mid + 1, hi), (lo, mid)) };
        self.k_nearest_in(first.0, first.1, q, found, k);
        if found.len() < k || off * off <= found.last().unwrap().0 {
            self.k_nearest_in(second.0, second.1, q, found, k);
        }
    }

    /// The `k` nearest, closest first, as (squared distance, index).
    pub fn k_nearest(&self, q: V3, k: usize, found: &mut Vec<(f64, u32)>) {
        found.clear();
        if k > 0 {
            self.k_nearest_in(0, self.ids.len(), q, found, k);
        }
    }
}

// --------------------------------------------------------------------------- triangles of a polygon

/// Triangles of one polygon as corner numbers, cut as Blender's BVH cuts them: a quad loses its first corner (the diagonal
/// from 1 to 3), or, when corner 0 or 2 turns inward seen along its normal, the other diagonal. Bigger polygons lose ears.
fn cut_polygon(corner: &[V3], out: &mut Vec<[u32; 3]>) {
    let n = corner.len();
    match n {
        0..=2 => {}
        3 => out.push([0, 1, 2]),
        4 => {
            let (a, b, c, d) = (corner[0], corner[1], corner[2], corner[3]);
            let facing = cross(sub(c, a), sub(d, b));
            let inward = dot(cross(sub(a, d), sub(b, a)), facing) < 0.0 || dot(cross(sub(c, b), sub(d, c)), facing) < 0.0;
            if inward {
                out.extend([[0, 1, 2], [2, 3, 0]]);
            } else {
                out.extend([[3, 0, 1], [1, 2, 3]]);
            }
        }
        _ => {
            let mut facing = [0.0; 3]; // Newell's sum
            for i in 0..n {
                let c = cross(corner[i], corner[(i + 1) % n]);
                facing = [facing[0] + c[0], facing[1] + c[1], facing[2] + c[2]];
            }
            let mut left: Vec<u32> = (0..n as u32).collect();
            while left.len() > 3 {
                let m = left.len();
                let at = |k: usize| corner[left[k % m] as usize];
                let convex = |k: usize| dot(cross(sub(at(k + 1), at(k)), sub(at(k + 2), at(k + 1))), facing) >= 0.0;
                let holds = |k: usize, p: V3| {
                    let (a, b, c) = (at(k), at(k + 1), at(k + 2));
                    dot(cross(sub(b, a), sub(p, a)), facing) >= 0.0 && dot(cross(sub(c, b), sub(p, b)), facing) >= 0.0 && dot(cross(sub(a, c), sub(p, c)), facing) >= 0.0
                };
                let ear = (0..m).find(|&k| convex(k) && (3..m).all(|o| !holds(k, at(k + o)))).unwrap_or(0);
                out.push([left[ear], left[(ear + 1) % m], left[(ear + 2) % m]]);
                left.remove((ear + 1) % m);
            }
            out.push([left[0], left[1], left[2]]);
        }
    }
}

// --------------------------------------------------------------------------- faces

#[derive(Clone, Copy, Default)]
struct Node {
    low: [f32; 3],
    high: [f32; 3],
    a: u32, // inside: where the second child is. Leaf: its first triangle
    b: u32, // Leaf: its triangle count. Inside: 0
}

/// A bounding-box tree over a mesh's polygons, cut in triangles, the first child right after its parent.
pub struct FaceTree {
    nodes: Vec<Node>,
    corners: Vec<[[f32; 3]; 3]>,
    owner: Vec<u32>,
    starts: Vec<u32>,
    verts: Vec<u32>,
}

fn node_count(tris: usize) -> usize {
    if tris <= LEAF_TRIS {
        1
    } else {
        1 + node_count(tris / 2) + node_count(tris - tris / 2)
    }
}

type Box6 = ([f32; 3], [f32; 3]);

fn union(a: Box6, b: Box6) -> Box6 {
    ([a.0[0].min(b.0[0]), a.0[1].min(b.0[1]), a.0[2].min(b.0[2])], [a.1[0].max(b.1[0]), a.1[1].max(b.1[1]), a.1[2].max(b.1[2])])
}

fn tri_box(t: &[[f32; 3]; 3]) -> Box6 {
    union(union((t[0], t[0]), (t[1], t[1])), (t[2], t[2]))
}

fn build_nodes(order: &mut [u32], nodes: &mut [Node], first: usize, node_at: usize, corners: &[[[f32; 3]; 3]]) -> Box6 {
    let centre = |i: u32, k: usize| {
        let t = &corners[i as usize];
        t[0][k] + t[1][k] + t[2][k]
    };
    if order.len() <= LEAF_TRIS {
        let mut hull = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for &i in order.iter() {
            hull = union(hull, tri_box(&corners[i as usize]));
        }
        nodes[0] = Node { low: hull.0, high: hull.1, a: first as u32, b: order.len() as u32 };
        return hull;
    }
    let (mut low, mut high) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for &i in order.iter() {
        for k in 0..3 {
            let c = centre(i, k);
            low[k] = low[k].min(c);
            high[k] = high[k].max(c);
        }
    }
    let wide = (0..3).fold(0, |best, k| if high[k] - low[k] > high[best] - low[best] { k } else { best });
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |&a, &b| centre(a, wide).total_cmp(&centre(b, wide)));
    let (order_a, order_b) = order.split_at_mut(mid);
    let left = node_count(mid);
    let (head, rest) = nodes.split_at_mut(1);
    let (nodes_a, nodes_b) = rest.split_at_mut(left);
    let (box_a, box_b) = if mid >= SIDE_BY_SIDE {
        rayon::join(
            || build_nodes(order_a, nodes_a, first, node_at + 1, corners),
            || build_nodes(order_b, nodes_b, first + mid, node_at + 1 + left, corners),
        )
    } else {
        (build_nodes(order_a, nodes_a, first, node_at + 1, corners), build_nodes(order_b, nodes_b, first + mid, node_at + 1 + left, corners))
    };
    let hull = union(box_a, box_b);
    head[0] = Node { low: hull.0, high: hull.1, a: (node_at + 1 + left) as u32, b: 0 };
    hull
}

#[inline]
fn to_f64(p: [f32; 3]) -> V3 {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

/// The closest point of a triangle to `p` (Ericson, Real-Time Collision Detection).
fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let along = |from: V3, dir: V3, t: f64| [from[0] + t * dir[0], from[1] + t * dir[1], from[2] + t * dir[2]];
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return along(a, ab, d1 / (d1 - d3));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return along(a, ac, d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return along(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let under = va + vb + vc;
    if under == 0.0 {
        return a; // a triangle without area, all of it on this corner's line
    }
    let (v, w) = (vb / under, vc / under);
    [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w]
}

/// Do two triangles pass through each other? As Blender's `isect_tri_tri_v3`: coplanar ones don't, and touching ones do.
fn triangles_cross(ta: &[[f32; 3]; 3], tb: &[[f32; 3]; 3]) -> bool {
    let a = [to_f64(ta[0]), to_f64(ta[1]), to_f64(ta[2])];
    let b = [to_f64(tb[0]), to_f64(tb[1]), to_f64(tb[2])];
    let plane = |t: &[V3; 3]| {
        let n = cross(sub(t[0], t[1]), sub(t[2], t[1]));
        (n, -dot(n, t[1]))
    };
    let sides = |t: &[V3; 3], n: V3, d: f64| [(dot(n, t[0]) + d) as f32, (dot(n, t[1]) + d) as f32, (dot(n, t[2]) + d) as f32];
    let apart = |s: &[f32; 3]| (s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0) || (s[0] != 0.0 && s[1] != 0.0 && s[2] != 0.0 && (s[0] < 0.0) == (s[1] < 0.0) && (s[0] < 0.0) == (s[2] < 0.0));
    let (na, da) = plane(&a);
    let side_b = sides(&b, na, da);
    if apart(&side_b) {
        return false;
    }
    let (nb, db) = plane(&b);
    let side_a = sides(&a, nb, db);
    if apart(&side_a) {
        return false;
    }
    let line = cross(na, nb);
    let span = |t: &[V3; 3], s: &[f32; 3]| {
        // Reorder so the corner alone on its side of the plane is the middle one.
        let same = |i: usize, j: usize| s[i] != 0.0 && s[j] != 0.0 && (s[i] < 0.0) == (s[j] < 0.0);
        let (i0, i1, i2) = if same(0, 1) {
            (1, 2, 0)
        } else if same(1, 2) {
            (2, 0, 1)
        } else {
            (0, 1, 2)
        };
        let base = dot(line, t[i1]);
        let lone = s[i1] as f64;
        let at0 = (dot(line, t[i0]) - base) * (lone / (lone - s[i0] as f64)) + base;
        let at1 = (dot(line, t[i2]) - base) * (lone / (lone - s[i2] as f64)) + base;
        if at0 > at1 {
            (at1, at0)
        } else {
            (at0, at1)
        }
    };
    let (ra, rb) = (span(&a, &side_a), span(&b, &side_b));
    !(ra.0 > rb.1 || ra.1 < rb.0)
}

impl FaceTree {
    pub fn new(points: &[f64], indptr: &[i64], flat: &[i64]) -> Self {
        let polys = indptr.len() - 1;
        let (mut corners, mut owner) = (Vec::with_capacity(polys * 2), Vec::with_capacity(polys * 2));
        let (mut cuts, mut around) = (Vec::new(), Vec::new());
        for f in 0..polys {
            let face = &flat[indptr[f] as usize..indptr[f + 1] as usize];
            around.clear();
            around.extend(face.iter().map(|&v| point(points, v as usize)));
            cuts.clear();
            cut_polygon(&around, &mut cuts);
            for t in &cuts {
                let single = |c: u32| {
                    let p = around[c as usize];
                    [p[0] as f32, p[1] as f32, p[2] as f32]
                };
                corners.push([single(t[0]), single(t[1]), single(t[2])]);
                owner.push(f as u32);
            }
        }
        let mut order: Vec<u32> = (0..corners.len() as u32).collect();
        let mut nodes = vec![Node::default(); if corners.is_empty() { 0 } else { node_count(corners.len()) }];
        if !corners.is_empty() {
            build_nodes(&mut order, &mut nodes, 0, 0, &corners);
        }
        FaceTree {
            nodes,
            corners: order.iter().map(|&i| corners[i as usize]).collect(),
            owner: order.iter().map(|&i| owner[i as usize]).collect(),
            starts: indptr.iter().map(|&i| i as u32).collect(),
            verts: flat.iter().map(|&v| v as u32).collect(),
        }
    }

    #[inline]
    fn box_gap2(node: &Node, q: V3) -> f64 {
        let mut total = 0.0;
        for k in 0..3 {
            let off = (node.low[k] as f64 - q[k]).max(q[k] - node.high[k] as f64).max(0.0);
            total += off * off;
        }
        total
    }

    /// (closest point, polygon, squared distance), or None on an empty tree.
    pub fn nearest(&self, q: V3) -> Option<(V3, u32, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = (q, u32::MAX, f64::INFINITY);
        let mut stack = [(0u32, 0.0f64); 64];
        let mut depth = 1;
        while depth > 0 {
            depth -= 1;
            let (at, near) = stack[depth];
            if near > best.2 {
                continue;
            }
            let node = &self.nodes[at as usize];
            if node.b > 0 {
                for t in node.a as usize..(node.a + node.b) as usize {
                    let c = &self.corners[t];
                    let found = closest_on_triangle(q, to_f64(c[0]), to_f64(c[1]), to_f64(c[2]));
                    let d = gap2(found, q);
                    if d < best.2 || (d == best.2 && self.owner[t] < best.1) {
                        best = (found, self.owner[t], d);
                    }
                }
                continue;
            }
            let (first, second) = (at + 1, node.a);
            let (gap_a, gap_b) = (Self::box_gap2(&self.nodes[first as usize], q), Self::box_gap2(&self.nodes[second as usize], q));
            let (near_child, far_child) = if gap_a <= gap_b { ((first, gap_a), (second, gap_b)) } else { ((second, gap_b), (first, gap_a)) };
            stack[depth] = far_child;
            stack[depth + 1] = near_child;
            depth += 2;
        }
        Some(best)
    }

    fn share_a_vertex(&self, f: u32, g: u32) -> bool {
        let of = |x: u32| &self.verts[self.starts[x as usize] as usize..self.starts[x as usize + 1] as usize];
        let other = of(g);
        of(f).iter().any(|v| other.contains(v))
    }

    /// Polygon pairs (lower first, sorted, each once) that pass through each other and share no vertex. None when stopped.
    pub fn crossing_pairs(&self, stop: &AtomicBool) -> Option<Vec<(u32, u32)>> {
        let touches = |a: &Box6, n: &Node| (0..3).all(|k| a.0[k] <= n.high[k] && a.1[k] >= n.low[k]);
        let blocks = self.corners.len().div_ceil(BLOCK);
        let block = |b: usize| {
            let mut found = Vec::new();
            if stop.load(Ordering::Relaxed) || self.nodes.is_empty() {
                return found;
            }
            let mut stack = [0u32; 64];
            for t in b * BLOCK..((b + 1) * BLOCK).min(self.corners.len()) {
                let (mine, me) = (tri_box(&self.corners[t]), self.owner[t]);
                let mut depth = 1;
                stack[0] = 0;
                while depth > 0 {
                    depth -= 1;
                    let at = stack[depth];
                    let node = &self.nodes[at as usize];
                    if !touches(&mine, node) {
                        continue;
                    }
                    if node.b == 0 {
                        stack[depth] = at + 1;
                        stack[depth + 1] = node.a;
                        depth += 2;
                        continue;
                    }
                    for o in node.a as usize..(node.a + node.b) as usize {
                        let them = self.owner[o];
                        if o <= t || them == me {
                            continue;
                        }
                        let other = tri_box(&self.corners[o]);
                        if (0..3).all(|k| mine.0[k] <= other.1[k] && mine.1[k] >= other.0[k]) && triangles_cross(&self.corners[t], &self.corners[o]) && !self.share_a_vertex(me, them) {
                            found.push(if me < them { (me, them) } else { (them, me) });
                        }
                    }
                }
            }
            found
        };
        let mut pairs: Vec<(u32, u32)> = if self.corners.len() >= PARALLEL_MIN {
            (0..blocks).into_par_iter().flat_map_iter(block).collect()
        } else {
            (0..blocks).flat_map(block).collect()
        };
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        crate::topo::sort(&mut pairs);
        pairs.dedup();
        Some(pairs)
    }
}

// --------------------------------------------------------------------------- batches

/// Index and distance of the nearest point for every query. False when stopped.
pub fn nearest_points(tree: &PointTree, queries: &[f64], stop: &AtomicBool) -> (bool, Vec<(i64, f64)>) {
    let mut rows = vec![(-1, f64::INFINITY); queries.len() / 3];
    let done = each_row(&mut rows, 1, stop, |i, row| {
        let (id, d) = tree.nearest(point(queries, i));
        row[0] = (if id == u32::MAX { -1 } else { id as i64 }, d.sqrt());
    });
    (done, rows)
}

/// Closest point, polygon and distance for every query. False when stopped.
pub fn nearest_faces(tree: &FaceTree, queries: &[f64], stop: &AtomicBool) -> (bool, Vec<(V3, i64, f64)>) {
    let mut rows = vec![([0.0; 3], -1, f64::INFINITY); queries.len() / 3];
    let done = each_row(&mut rows, 1, stop, |i, row| {
        if let Some((at, face, d)) = tree.nearest(point(queries, i)) {
            row[0] = (at, face as i64, d.sqrt());
        }
    });
    (done, rows)
}

/// For each query, the first of its `k` nearest points whose kind matches the query's kind, or -1. False when stopped.
pub fn first_of_kind(tree: &PointTree, queries: &[f64], query_kind: &[i64], point_kind: &[i64], k: usize, stop: &AtomicBool) -> (bool, Vec<i64>) {
    let mut out = vec![-1; queries.len() / 3];
    let done = each_row(&mut out, 1, stop, |i, row| {
        thread_local!(static FOUND: std::cell::RefCell<Vec<(f64, u32)>> = const { std::cell::RefCell::new(Vec::new()) });
        FOUND.with(|found| {
            let mut found = found.borrow_mut();
            tree.k_nearest(point(queries, i), k, &mut found);
            row[0] = found.iter().find(|item| point_kind[item.1 as usize] == query_kind[i]).map_or(-1, |item| item.1 as i64);
        });
    });
    (done, out)
}

// --------------------------------------------------------------------------- checks

/// `checks.edge_flags` and `checks.hole_counts` in one go: (broken per face, non-manifold edges, flipped edges,
/// duplicate faces, boundary loops, loops per piece as (piece, count) sorted by piece).
pub fn mesh_checks(faces: &crate::topo::Faces, piece: &[i64]) -> (Vec<bool>, i64, i64, i64, i64, Vec<(i64, i64)>) {
    let count = faces.count();
    let mut broken = vec![false; count];
    let mut keys: Vec<(Vec<i64>, u32)> = (0..count as u32)
        .into_par_iter()
        .map(|f| {
            let mut key = faces.face(f as usize).to_vec();
            key.sort_unstable();
            (key, f)
        })
        .collect();
    keys.par_sort_unstable();
    let mut dups = 0;
    for pair in keys.windows(2) {
        if pair[0].0 == pair[1].0 {
            dups += 1;
            broken[pair[1].1 as usize] = true;
        }
    }
    drop(keys);
    let halves = crate::topo::halves(faces);
    let edges = crate::topo::edges(&halves);
    let (mut bad_edges, mut flipped) = (0, 0);
    let mut loose: Vec<u32> = Vec::new(); // the one use of each boundary edge, in the order the edges first turn up
    for e in 0..edges.len() {
        let uses = edges.uses(e);
        match uses.len() {
            1 => loose.push(uses[0]),
            2 => {
                let (x, y) = (uses[0] as usize, uses[1] as usize);
                if (halves.from[x] < halves.to[x]) == (halves.from[y] < halves.to[y]) {
                    flipped += 1;
                    broken[halves.face[x] as usize] = true;
                    broken[halves.face[y] as usize] = true;
                }
            }
            _ => {
                bad_edges += 1;
                for &u in uses {
                    broken[halves.face[u as usize] as usize] = true;
                }
            }
        }
    }
    let highest = halves.from.iter().chain(&halves.to).copied().max().unwrap_or(-1);
    let mut sets = crate::topo::Dsu::new((highest + 1) as usize);
    for &u in &loose {
        sets.union(halves.from[u as usize] as u32, halves.to[u as usize] as u32);
    }
    let mut roots = std::collections::HashMap::new();
    for &u in &loose {
        roots.insert(sets.find(halves.from[u as usize] as u32), piece[halves.face[u as usize] as usize]);
    }
    let mut per = std::collections::BTreeMap::new();
    for p in roots.values() {
        *per.entry(*p).or_insert(0i64) += 1;
    }
    (broken, bad_edges, flipped, dups, roots.len() as i64, per.into_iter().collect())
}
