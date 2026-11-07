// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! UV seam snap and split leftovers (package K1b, SPEC 4.5 and 4.7). Byte-identical on `main` and `standalone`: uses
//! only `topo`, `geom` and the frozen seam input of `uvseam`.
//!
//! Our own design and code. Textbook pieces: Dijkstra's shortest paths (Dijkstra 1959) inside a corridor around the
//! chain; closest point on a segment and a segment crossing a plane (Ericson, "Real-Time Collision Detection", 2005,
//! 5.1.2 and 5.3.1); a uniform grid for neighbour queries; Newell's polygon normal (Newell 1972, as in Sutherland,
//! Sproull, Schumacker 1974). The corridor rule, the staircase test, the move guard, the per-edge crossings with
//! vertex reuse, the per-side UVs from the chain polylines, fan and successive cuts and the damage-ordered face
//! budget are ours. We never add a strip or row of faces along an island border: only single faces the remesh and
//! the snap could not fit are cut, once per chain, along the source polyline.

#![allow(dead_code)] // RELAX_RINGS and the state codes are read by the callers (the twin and the bindings use them by value)

use crate::geom::quad_valid;
use crate::topo::{edges, halves, Dsu, Faces};
use crate::uvseam::{closest_bary, SeamUv};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};

// Constants table (mirrored in triadq/uvsnap.py; tests/test_uv_scaffold.py compares the two).
/// Snap corridor: a result vertex may be on a snap path when it is within this many of its own mean edge lengths of
/// the chain (about the half-diagonal plus margin, so a staircase on a 5-11 deg chain stays inside).
pub const SNAP_REACH: f64 = 1.0;
/// Snap path edge cost: `length * (1 + SNAP_COST_K * distance / edge)` (bench-tuned start value).
pub const SNAP_COST_K: f64 = 4.0;
/// Accept a path only when its vertices' mean distance to the chain is below this share of their edge length...
pub const SNAP_MEAN_DIST: f64 = 0.35;
/// ...and it turns away from the chain direction at most once per this many path edges (a staircase shears quads).
pub const SNAP_TURN_EVERY: f64 = 5.0;
/// A path edge "turns" when its angle to the chain direction is above 45 deg (cosine below this) or it runs back.
pub const SNAP_TURN_COS: f64 = 0.7071067811865476;
/// A chain is already on result edges when each of its vertices is within this share of an edge of a result vertex.
pub const ON_EDGE_SHARE: f64 = 1e-6;
/// Tangential relax after the snap: move each ring vertex this share of the way to its neighbours' mean.
pub const RELAX_WEIGHT: f64 = 0.5;
/// Rings relaxed around snapped vertices by default (a setting: higher evens out more).
pub const RELAX_RINGS: usize = 1;
/// Split only chains whose two sides differ by at least this many pixels somewhere.
pub const SPLIT_GAP_PX: f64 = 1.0;
/// A chain crosses a result edge when its crossing point is within this share of the edge length of the edge.
pub const CROSS_REACH: f64 = 0.5;
/// A crossing closer than this share of the edge to an end reuses that vertex (no sliver faces).
pub const REUSE_EDGE: f64 = 0.1;
/// A crossing point belongs to another surface (the far side of a thin part) when another result face is nearer to it
/// than this share of the crossed edge's own faces (bench: tried sparing faces at the edge's ends and an absolute floor;
/// both measured worse on fire_hydrant).
pub const NEARER_SHARE: f64 = 0.5;
/// Quad warp limit of the add-on's quad test (60 deg).
pub const WARP_COSINE: f64 = 0.5;

/// Chain class of a part contact (the twin's `part_contacts`: where two unconnected source pieces cut through each
/// other). Only this kernel reads it: the snap fits it like a Soft chain, the split leaves it alone.
pub const CONTACT: u8 = 4;

/// Frozen cross-kernel input (K1b): per vertex, the farthest it may move in model units (0 = held).
pub type MaxMove<'a> = &'a [f64];

/// Snap chain states.
pub const STATE_SKIPPED: u8 = 0;
pub const STATE_ACCEPTED: u8 = 1;
pub const STATE_REFUSED: u8 = 2;
pub const STATE_NO_PATH: u8 = 3;
pub const STATE_ON_EDGES: u8 = 4;
pub const STATE_SHORT: u8 = 5;
pub const STATE_PARTLY_HELD: u8 = 6;

/// Split face states (per input face).
pub const FACE_KEPT: u8 = 0;
pub const FACE_SPLIT: u8 = 1;
pub const FACE_LEFT: u8 = 2;
pub const FACE_CANNOT: u8 = 3;
pub const FACE_BAKE: u8 = 4;

/// The source's seam chains with their 3D polylines: `xyz` holds 3 values per welded vertex (`run_vertex` ids).
pub struct Chains<'a> {
    pub seams: SeamUv<'a>,
    pub xyz: &'a [f64],
}

type V3 = [f64; 3];

fn p3(points: &[f64], v: usize) -> V3 {
    [points[3 * v], points[3 * v + 1], points[3 * v + 2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
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
fn lerp(a: V3, b: V3, t: f64) -> V3 {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
}
fn unit(a: V3) -> V3 {
    let l = norm(a);
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [0.0; 3]
    }
}

/// Newell's normal of a polygon (length = twice its area).
fn newell(pos: &[V3]) -> V3 {
    let mut n = [0.0; 3];
    for k in 0..pos.len() {
        n = add(n, cross(pos[k], pos[(k + 1) % pos.len()]));
    }
    n
}

fn tolerance(pos: &[V3]) -> f64 {
    let mut size = 0.0;
    for c in 0..3 {
        let (mut lo, mut hi) = (pos[0][c], pos[0][c]);
        for q in &pos[1..] {
            lo = lo.min(q[c]);
            hi = hi.max(q[c]);
        }
        size += (hi - lo) * (hi - lo);
    }
    (size * 1e-14).max(1e-20)
}

/// A face is fine when it is a valid quad (the add-on's quad test) or, for other sizes, every fan triangle turns with
/// the polygon's normal.
fn face_ok(pos: &[V3]) -> bool {
    if pos.len() < 3 || !pos.iter().flatten().all(|x| x.is_finite()) {
        return false;
    }
    let eps = tolerance(pos);
    if pos.len() == 4 {
        return quad_valid(&[pos[0], pos[1], pos[2], pos[3]], eps, WARP_COSINE);
    }
    let n = newell(pos);
    if !(norm(n) > eps) {
        return false;
    }
    (1..pos.len() - 1).all(|k| dot(cross(sub(pos[k], pos[0]), sub(pos[k + 1], pos[0])), n) > 0.0)
}

/// Total order on finite floats for the heap.
#[derive(Clone, Copy, PartialEq)]
struct Key(f64, usize);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> CmpOrdering {
        // reversed: BinaryHeap pops the smallest (cost, vertex)
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// Uniform grid over boxes (vertices or edges) for neighbour queries. Queries return every item whose box touches the
/// query box, sorted and unique, so the exact tests after it see the same candidates as a scan of everything would.
struct Grid {
    cell: f64,
    map: HashMap<[i64; 3], Vec<u32>>,
}

impl Grid {
    fn new(cell: f64) -> Self {
        Grid { cell: if cell > 0.0 && cell.is_finite() { cell } else { 1.0 }, map: HashMap::new() }
    }
    fn key(&self, p: V3) -> [i64; 3] {
        [(p[0] / self.cell).floor() as i64, (p[1] / self.cell).floor() as i64, (p[2] / self.cell).floor() as i64]
    }
    fn insert(&mut self, id: u32, lo: V3, hi: V3) {
        let (a, b) = (self.key(lo), self.key(hi));
        for i in a[0]..=b[0] {
            for j in a[1]..=b[1] {
                for k in a[2]..=b[2] {
                    self.map.entry([i, j, k]).or_default().push(id);
                }
            }
        }
    }
    fn query(&self, lo: V3, hi: V3) -> Vec<u32> {
        let (a, b) = (self.key(lo), self.key(hi));
        let mut out = Vec::new();
        let cells = ((b[0] - a[0] + 1) as f64) * ((b[1] - a[1] + 1) as f64) * ((b[2] - a[2] + 1) as f64);
        if cells > 4.0 * self.map.len() as f64 {
            for ids in self.map.values() {
                out.extend_from_slice(ids);
            }
        } else {
            for i in a[0]..=b[0] {
                for j in a[1]..=b[1] {
                    for k in a[2]..=b[2] {
                        if let Some(ids) = self.map.get(&[i, j, k]) {
                            out.extend_from_slice(ids);
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

fn bbox(pts: &[V3], pad: f64) -> (V3, V3) {
    let (mut lo, mut hi) = (pts[0], pts[0]);
    for p in pts {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    ([lo[0] - pad, lo[1] - pad, lo[2] - pad], [hi[0] + pad, hi[1] + pad, hi[2] + pad])
}

/// The result mesh's topology: undirected edges, vertex neighbours and faces, mean edge length per vertex.
struct Mesh {
    ea: Vec<usize>,
    eb: Vec<usize>,
    efaces: Vec<Vec<usize>>,
    nbr_ptr: Vec<usize>,
    nbr: Vec<usize>,
    vf_ptr: Vec<usize>,
    vf: Vec<usize>,
    vedge: Vec<f64>,
    mean_edge: f64,
    max_edge: f64,
}

impl Mesh {
    fn new(points: &[f64], faces: &Faces) -> Mesh {
        let n = points.len() / 3;
        let half = halves(faces);
        let all = edges(&half);
        let ne = all.len();
        let (mut ea, mut eb, mut efaces) = (Vec::with_capacity(ne), Vec::with_capacity(ne), Vec::with_capacity(ne));
        let (mut sum, mut cnt) = (vec![0.0; n], vec![0usize; n]);
        let (mut total, mut max_edge) = (0.0, 0.0f64);
        let mut deg = vec![0usize; n];
        for e in 0..ne {
            let (a, b) = (all.a[e] as usize, all.b[e] as usize);
            let l = norm(sub(p3(points, b), p3(points, a)));
            sum[a] += l;
            sum[b] += l;
            cnt[a] += 1;
            cnt[b] += 1;
            total += l;
            max_edge = max_edge.max(l);
            ea.push(a);
            eb.push(b);
            let mut fs: Vec<usize> = all.uses(e).iter().map(|&h| half.face[h as usize] as usize).collect();
            fs.dedup();
            efaces.push(fs);
            if a != b {
                deg[a] += 1;
                deg[b] += 1;
            }
        }
        let mean_edge = if ne > 0 { total / ne as f64 } else { 1.0 };
        let vedge = (0..n).map(|v| if cnt[v] > 0 { sum[v] / cnt[v] as f64 } else { mean_edge }).collect();
        let mut nbr_ptr = vec![0usize; n + 1];
        for v in 0..n {
            nbr_ptr[v + 1] = nbr_ptr[v] + deg[v];
        }
        let mut fill = nbr_ptr.clone();
        let mut nbr = vec![0usize; nbr_ptr[n]];
        for e in 0..ne {
            let (a, b) = (ea[e], eb[e]);
            if a != b {
                nbr[fill[a]] = b;
                fill[a] += 1;
                nbr[fill[b]] = a;
                fill[b] += 1;
            }
        }
        for v in 0..n {
            nbr[nbr_ptr[v]..nbr_ptr[v + 1]].sort_unstable();
        }
        let mut vcount = vec![0usize; n + 1];
        for f in 0..faces.count() {
            for &v in faces.face(f) {
                vcount[v as usize + 1] += 1;
            }
        }
        for v in 0..n {
            vcount[v + 1] += vcount[v];
        }
        let mut fill = vcount.clone();
        let mut vf = vec![0usize; vcount[n]];
        for f in 0..faces.count() {
            for &v in faces.face(f) {
                vf[fill[v as usize]] = f;
                fill[v as usize] += 1;
            }
        }
        Mesh { ea, eb, efaces, nbr_ptr, nbr, vf_ptr: vcount, vf, vedge, mean_edge, max_edge }
    }
    fn nbrs(&self, v: usize) -> &[usize] {
        &self.nbr[self.nbr_ptr[v]..self.nbr_ptr[v + 1]]
    }
    fn faces_of(&self, v: usize) -> &[usize] {
        &self.vf[self.vf_ptr[v]..self.vf_ptr[v + 1]]
    }
}

/// One chain's polyline and per-side UVs.
struct Chain {
    pts: Vec<V3>,
    uv_a: Vec<[f64; 2]>,
    uv_b: Vec<[f64; 2]>,
    gap: Vec<f64>,
    closed: bool,
    start_id: i64,
    end_id: i64,
}

fn read_chains(ch: &Chains) -> Vec<Chain> {
    let s = &ch.seams;
    (0..s.class.len())
        .map(|c| {
            let (a, b) = (s.run_indptr[c] as usize, s.run_indptr[c + 1] as usize);
            let ids = &s.run_vertex[a..b];
            Chain {
                pts: ids.iter().map(|&w| p3(ch.xyz, w as usize)).collect(),
                uv_a: (a..b).map(|i| [s.uv_a[2 * i], s.uv_a[2 * i + 1]]).collect(),
                uv_b: (a..b).map(|i| [s.uv_b[2 * i], s.uv_b[2 * i + 1]]).collect(),
                gap: s.gap_px[a..b].to_vec(),
                closed: ids.len() > 2 && ids[0] == ids[ids.len() - 1],
                start_id: ids[0],
                end_id: ids[ids.len() - 1],
            }
        })
        .collect()
}

impl Chain {
    /// Closest point on segments `lo..hi`: (squared distance, parameter = segment + share). Ties keep the first.
    fn closest(&self, lo: usize, hi: usize, q: V3) -> (f64, f64) {
        let mut best = (f64::INFINITY, lo as f64);
        for j in lo..hi {
            let (a, b) = (self.pts[j], self.pts[j + 1]);
            let ab = sub(b, a);
            let l2 = dot(ab, ab);
            let mut t = if l2 > 0.0 { dot(sub(q, a), ab) / l2 } else { 0.0 };
            t = t.clamp(0.0, 1.0);
            let d = sub(q, lerp(a, b, t));
            let d2 = dot(d, d);
            if d2 < best.0 {
                best = (d2, j as f64 + t);
            }
        }
        best
    }
    fn seg(&self, param: f64) -> (usize, f64) {
        let last = self.pts.len().saturating_sub(2);
        let j = (param.floor().max(0.0) as usize).min(last);
        (j, param - j as f64)
    }
    fn at(&self, param: f64) -> V3 {
        if self.pts.len() < 2 {
            return self.pts[0];
        }
        let (j, t) = self.seg(param);
        lerp(self.pts[j], self.pts[j + 1], t)
    }
    fn tangent(&self, param: f64) -> V3 {
        if self.pts.len() < 2 {
            return [0.0; 3];
        }
        let (j, _) = self.seg(param);
        unit(sub(self.pts[j + 1], self.pts[j]))
    }
    fn uv(&self, param: f64, side_b: bool) -> [f64; 2] {
        let uv = if side_b { &self.uv_b } else { &self.uv_a };
        if uv.len() < 2 {
            return uv[0];
        }
        let (j, t) = self.seg(param);
        [uv[j][0] + t * (uv[j + 1][0] - uv[j][0]), uv[j][1] + t * (uv[j + 1][1] - uv[j][1])]
    }
    fn gap_at(&self, param: f64) -> f64 {
        if self.gap.len() < 2 {
            return self.gap[0];
        }
        let (j, t) = self.seg(param);
        self.gap[j] + t * (self.gap[j + 1] - self.gap[j])
    }
    fn max_gap(&self) -> f64 {
        self.gap.iter().cloned().fold(0.0, f64::max)
    }
    fn end(&self) -> f64 {
        (self.pts.len() - 1) as f64
    }
}

fn face_pos(pos: &[V3], faces: &Faces, f: usize) -> Vec<V3> {
    faces.face(f).iter().map(|&v| pos[v as usize]).collect()
}

/// Would moving `v` to `to` break or flip a face around it? (A face that was already broken may stay broken.)
fn move_ok(pos: &mut [V3], faces: &Faces, mesh: &Mesh, v: usize, to: V3) -> bool {
    let from = pos[v];
    let before: Vec<(bool, V3)> = mesh
        .faces_of(v)
        .iter()
        .map(|&f| {
            let p = face_pos(pos, faces, f);
            (face_ok(&p), newell(&p))
        })
        .collect();
    pos[v] = to;
    let ok = mesh.faces_of(v).iter().zip(&before).all(|(&f, &(was, n0))| {
        let p = face_pos(pos, faces, f);
        (face_ok(&p) || !was) && dot(newell(&p), n0) > 0.0
    });
    pos[v] = from;
    ok
}

/// Result of [`snap`].
pub struct Snap {
    pub points: Vec<f64>,
    pub held: Vec<bool>,
    pub snapped: Vec<bool>,
    pub vchain: Vec<i64>,
    pub vparam: Vec<f64>,
    pub state: Vec<u8>,
    pub mean_dist: Vec<f64>,
    pub turns: Vec<i64>,
    pub path_edges: Vec<i64>,
    pub moved: usize,
    pub relaxed: usize,
    pub stopped: bool,
}

fn nearest_vertex(pos: &[V3], grid: &Grid, q: V3) -> usize {
    let mut r = grid.cell;
    loop {
        let ids = grid.query(sub(q, [r; 3]), add(q, [r; 3]));
        let mut best = (f64::INFINITY, usize::MAX);
        for &v in &ids {
            let d = sub(pos[v as usize], q);
            let d2 = dot(d, d);
            if d2 < best.0 {
                best = (d2, v as usize);
            }
        }
        if best.1 != usize::MAX && best.0 <= r * r {
            return best.1;
        }
        if ids.len() >= pos.len() {
            return if best.1 == usize::MAX { 0 } else { best.1 };
        }
        r *= 2.0;
    }
}

/// Seam snap (SPEC 4.5): moves result vertices onto Soft chains where a short, straight path of result edges runs
/// along the chain. Never flips or breaks a face, never moves a vertex further than `max_move` (empty: no limit); a
/// refused vertex is held. `vchain_in` / `vparam_in` (empty: none) are earlier tags, so Again resumes on its own result.
#[allow(clippy::too_many_arguments)]
pub fn snap(points: &[f64], faces: &Faces, chains: &Chains, max_move: MaxMove, vchain_in: &[i64], vparam_in: &[f64], rings: usize, stop: &AtomicBool) -> Snap {
    let n = points.len() / 3;
    let mesh = Mesh::new(points, faces);
    let orig: Vec<V3> = (0..n).map(|v| p3(points, v)).collect();
    let mut pos = orig.clone();
    let chs = read_chains(chains);
    let nc = chs.len();
    let mut out = Snap {
        points: Vec::new(),
        held: vec![false; n],
        snapped: vec![false; n],
        vchain: if vchain_in.len() == n { vchain_in.to_vec() } else { vec![-1; n] },
        vparam: if vparam_in.len() == n { vparam_in.to_vec() } else { vec![0.0; n] },
        state: vec![STATE_SKIPPED; nc],
        mean_dist: vec![0.0; nc],
        turns: vec![0; nc],
        path_edges: vec![0; nc],
        moved: 0,
        relaxed: 0,
        stopped: false,
    };
    let limit = |v: usize| if max_move.len() == n { max_move[v] } else { f64::INFINITY };
    let mut grid = Grid::new(mesh.mean_edge);
    for (v, &p) in pos.iter().enumerate() {
        grid.insert(v as u32, p, p);
    }
    let max_vedge = mesh.vedge.iter().cloned().fold(0.0, f64::max);
    for c in 0..nc {
        if stop.load(Ordering::Relaxed) {
            out.stopped = true;
            break;
        }
        let ch = &chs[c];
        if (chains.seams.class[c] != 2 && chains.seams.class[c] != CONTACT) || ch.pts.len() < 2 || n == 0 {
            continue;
        }
        // Already on result edges: every chain vertex sits on a result vertex, consecutive ones joined by an edge.
        let near: Vec<usize> = ch.pts.iter().map(|&p| nearest_vertex(&pos, &grid, p)).collect();
        let on = near.iter().zip(&ch.pts).all(|(&v, &p)| norm(sub(pos[v], p)) <= ON_EDGE_SHARE * mesh.vedge[v])
            && near.windows(2).all(|w| w[0] == w[1] || mesh.nbrs(w[0]).binary_search(&w[1]).is_ok());
        if on {
            for (i, &v) in near.iter().enumerate() {
                if out.vchain[v] < 0 {
                    out.vchain[v] = c as i64;
                    out.vparam[v] = i as f64;
                }
            }
            out.state[c] = STATE_ON_EDGES;
            continue;
        }
        let last = ch.pts.len() - 1;
        let halves_of: Vec<(usize, usize)> = if ch.closed && last >= 2 { vec![(0, last / 2), (last / 2, last)] } else { vec![(0, last)] };
        let mut states = Vec::new();
        for &(j0, j1) in &halves_of {
            let (a, b) = (ch.pts[j0], ch.pts[j1]);
            let (s, t) = (nearest_vertex(&pos, &grid, a), nearest_vertex(&pos, &grid, b));
            if s == t {
                states.push(STATE_SHORT);
                continue;
            }
            // Corridor.
            let (lo, hi) = bbox(&ch.pts[j0..=j1], SNAP_REACH * max_vedge);
            let mut dist: HashMap<usize, (f64, f64)> = HashMap::new();
            for &v in &grid.query(lo, hi) {
                let v = v as usize;
                let blocked = out.vchain[v] >= 0 && out.vchain[v] != c as i64 && v != s && v != t;
                if blocked {
                    continue;
                }
                let (d2, prm) = ch.closest(j0, j1, pos[v]);
                let d = d2.sqrt();
                if d <= SNAP_REACH * mesh.vedge[v] || v == s || v == t {
                    dist.insert(v, (d, prm));
                }
            }
            for v in [s, t] {
                dist.entry(v).or_insert_with(|| {
                    let (d2, prm) = ch.closest(j0, j1, pos[v]);
                    (d2.sqrt(), prm)
                });
            }
            // Dijkstra inside the corridor.
            let mut best: HashMap<usize, (f64, usize)> = HashMap::new();
            let mut heap = BinaryHeap::new();
            best.insert(s, (0.0, usize::MAX));
            heap.push(Key(0.0, s));
            while let Some(Key(cost, u)) = heap.pop() {
                if cost > best[&u].0 {
                    continue;
                }
                if u == t {
                    break;
                }
                let du = dist[&u].0 / mesh.vedge[u];
                for &w in mesh.nbrs(u) {
                    let Some(&(dw, _)) = dist.get(&w) else { continue };
                    let l = norm(sub(pos[w], pos[u]));
                    let step = cost + l * (1.0 + SNAP_COST_K * (du + dw / mesh.vedge[w]) / 2.0);
                    if best.get(&w).is_none_or(|&(old, _)| step < old) {
                        best.insert(w, (step, u));
                        heap.push(Key(step, w));
                    }
                }
            }
            if !best.contains_key(&t) {
                states.push(STATE_NO_PATH);
                continue;
            }
            let mut path = vec![t];
            while *path.last().unwrap() != s {
                path.push(best[path.last().unwrap()].1);
            }
            path.reverse();
            // Acceptance: close on average and no staircase.
            let mean = path.iter().map(|&v| dist[&v].0 / mesh.vedge[v]).sum::<f64>() / path.len() as f64;
            let mut turns = 0i64;
            for w in path.windows(2) {
                let (pu, pw) = (pos[w[0]], pos[w[1]]);
                let (_, prm) = ch.closest(j0, j1, scale(add(pu, pw), 0.5));
                let along = dot(sub(pw, pu), ch.tangent(prm));
                if along < SNAP_TURN_COS * norm(sub(pw, pu)) {
                    turns += 1;
                }
            }
            let edges_n = (path.len() - 1) as i64;
            out.mean_dist[c] = out.mean_dist[c].max(mean);
            out.turns[c] += turns;
            out.path_edges[c] += edges_n;
            if !(mean < SNAP_MEAN_DIST) || turns as f64 * SNAP_TURN_EVERY > edges_n as f64 {
                states.push(STATE_REFUSED);
                continue;
            }
            // Move path vertices onto the chain; chain ends go to the seam corners.
            let mut any_held = false;
            for (i, &v) in path.iter().enumerate() {
                if out.vchain[v] >= 0 {
                    continue;
                }
                let prm = if i == 0 {
                    j0 as f64
                } else if i == path.len() - 1 {
                    j1 as f64
                } else {
                    dist[&v].1
                };
                let to = ch.at(prm);
                if norm(sub(to, orig[v])) > limit(v) || !move_ok(&mut pos, faces, &mesh, v, to) {
                    out.held[v] = true;
                    any_held = true;
                    continue;
                }
                pos[v] = to;
                grid.insert(v as u32, to, to); // queries test current positions; the old cell entry stays (a superset)
                out.held[v] = false;
                out.snapped[v] = true;
                out.vchain[v] = c as i64;
                out.vparam[v] = prm;
                out.moved += 1;
            }
            states.push(if any_held { STATE_PARTLY_HELD } else { STATE_ACCEPTED });
        }
        out.state[c] = if states.iter().all(|&s| s == STATE_ACCEPTED) {
            STATE_ACCEPTED
        } else if states.iter().any(|&s| s == STATE_ACCEPTED || s == STATE_PARTLY_HELD) {
            STATE_PARTLY_HELD
        } else {
            states[0]
        };
    }
    // Tangential relax of the rings around snapped vertices (snapped and held vertices stay where they are).
    let mut ring: Vec<usize> = (0..n).filter(|&v| out.snapped[v]).collect();
    let mut seen: Vec<bool> = out.snapped.clone();
    for _ in 0..rings {
        if out.stopped || stop.load(Ordering::Relaxed) {
            out.stopped = true;
            break;
        }
        let mut next = Vec::new();
        for &u in &ring {
            for &w in mesh.nbrs(u) {
                if !seen[w] {
                    seen[w] = true;
                    next.push(w);
                }
            }
        }
        next.sort_unstable();
        for &v in &next {
            if out.held[v] || out.vchain[v] >= 0 || mesh.nbrs(v).is_empty() || !(limit(v) > 0.0) {
                continue;
            }
            let mut m = [0.0; 3];
            for &w in mesh.nbrs(v) {
                m = add(m, pos[w]);
            }
            m = scale(m, 1.0 / mesh.nbrs(v).len() as f64);
            let mut nv = [0.0; 3];
            for &f in mesh.faces_of(v) {
                nv = add(nv, unit(newell(&face_pos(&pos, faces, f))));
            }
            let nv = unit(nv);
            let d = sub(m, pos[v]);
            let dt = sub(d, scale(nv, dot(d, nv)));
            let to = add(pos[v], scale(dt, RELAX_WEIGHT));
            if norm(sub(to, orig[v])) <= limit(v) && move_ok(&mut pos, faces, &mesh, v, to) {
                pos[v] = to;
                out.relaxed += 1;
            }
        }
        ring = next;
    }
    out.points = pos.iter().flatten().cloned().collect();
    out
}

/// A point of a chain on a face: an existing vertex, a crossing on an edge, or a seam corner inside the face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Node {
    Vertex(usize),
    Edge(usize, usize), // (edge, chain)
    Corner(i64),        // welded corner id
}

#[derive(Clone, Copy)]
struct Cross {
    s: f64,
    param: f64,
    reuse: Option<usize>,
}

/// Result of [`split`].
pub struct Split {
    pub points: Vec<f64>,
    pub indptr: Vec<i64>,
    pub flat: Vec<i64>,
    pub origin: Vec<i64>,
    pub face_state: Vec<u8>,
    pub vchain: Vec<i64>,
    pub vparam: Vec<f64>,
    pub corner_uv: Vec<f64>,
    pub corner_side: Vec<i8>,
    pub faces_split: usize,
    pub extra_faces: i64,
    pub triangles_left: usize,
    pub faces_left: usize,
    pub faces_cannot: usize,
    pub faces_bake: usize,
    pub spent: f64,
    pub wanted: f64,
    pub stopped: bool,
}

/// Inside face `f` (its edges' inward planes) and near its plane.
fn inside(fp: &[V3], x: V3) -> bool {
    let n = unit(newell(fp));
    let k = fp.len();
    let mut c = [0.0; 3];
    let mut perim = 0.0;
    for i in 0..k {
        c = add(c, fp[i]);
        perim += norm(sub(fp[(i + 1) % k], fp[i]));
    }
    let c = scale(c, 1.0 / k as f64);
    if dot(sub(x, c), n).abs() > CROSS_REACH * perim / k as f64 {
        return false;
    }
    (0..k).all(|i| dot(sub(x, fp[i]), cross(n, sub(fp[(i + 1) % k], fp[i]))) >= 0.0)
}

/// Distance from `x` to a polygon (its fan triangles).
fn face_dist(fp: &[V3], x: V3) -> f64 {
    let mut best = f64::INFINITY;
    for k in 1..fp.len().saturating_sub(1) {
        let (a, b, c) = (fp[0], fp[k], fp[k + 1]);
        let w = closest_bary(x, a, b, c);
        let p = [w[0] * a[0] + w[1] * b[0] + w[2] * c[0], w[0] * a[1] + w[1] * b[1] + w[2] * c[1], w[0] * a[2] + w[1] * b[2] + w[2] * c[2]];
        best = best.min(norm(sub(x, p)));
    }
    best
}

/// Splits a polygon (node list) into quads and triangles by diagonals only: best valid quad first, else an ear.
fn to_quads(poly: &[usize], at: &[V3], normal: V3) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = poly.to_vec();
    while cur.len() > 4 {
        let m = cur.len();
        let mut best: Option<(f64, usize)> = None;
        for i in 0..m {
            let q = [cur[i], cur[(i + 1) % m], cur[(i + 2) % m], cur[(i + 3) % m]];
            let p: Vec<V3> = q.iter().map(|&x| at[x]).collect();
            if !face_ok(&p) || dot(newell(&p), normal) <= 0.0 {
                continue;
            }
            let mut score = f64::INFINITY;
            for k in 0..4 {
                let (e1, e2) = (sub(p[(k + 1) % 4], p[k]), sub(p[(k + 3) % 4], p[k]));
                score = score.min(dot(cross(e1, e2), normal) / (norm(e1) * norm(e2)));
            }
            if best.is_none_or(|(b, _)| score > b) {
                best = Some((score, i));
            }
        }
        match best {
            Some((_, i)) => {
                out.push(vec![cur[i], cur[(i + 1) % m], cur[(i + 2) % m], cur[(i + 3) % m]]);
                let (r1, r2) = ((i + 1) % m, (i + 2) % m);
                cur = cur.iter().enumerate().filter(|&(k, _)| k != r1 && k != r2).map(|(_, &x)| x).collect();
            }
            None => {
                let mut ear = 0;
                for i in 0..m {
                    let p = [at[cur[i]], at[cur[(i + 1) % m]], at[cur[(i + 2) % m]]];
                    if dot(cross(sub(p[1], p[0]), sub(p[2], p[0])), normal) > 0.0 {
                        ear = i;
                        break;
                    }
                }
                out.push(vec![cur[ear], cur[(ear + 1) % m], cur[(ear + 2) % m]]);
                let r = (ear + 1) % m;
                cur.remove(r);
            }
        }
    }
    out.push(cur);
    out
}

/// One face's cut plan.
struct Plan {
    /// Chords (node pairs, by chain) and the fan corner with its spokes.
    chords: Vec<(Node, Node, usize, f64, f64)>,
    fan: Option<(Node, Vec<(Node, usize, f64)>)>,
    cannot: bool,
    whole: bool,
}

fn node_edges(node: Node, fv: &[usize], fe: &[usize]) -> Vec<usize> {
    let k = fv.len();
    match node {
        Node::Vertex(v) => (0..k).filter(|&i| fv[i] == v).flat_map(|i| [fe[i], fe[(i + k - 1) % k]]).collect(),
        Node::Edge(e, _) => vec![e],
        Node::Corner(_) => Vec::new(),
    }
}

/// Split leftovers (SPEC 4.7): cuts result faces still crossed by a Soft chain (gap >= `SPLIT_GAP_PX`) along the
/// chain, sharing one vertex per crossed edge, reusing a vertex closer than `REUSE_EDGE`, fanning at seam corners and
/// cutting several chains one after another. Never moves a vertex. Spends at most `budget` extra faces (negative: no
/// limit) on whole crossed runs, most damage first (area x seen x gap px). `vchain` / `vparam` (empty: none) tag
/// vertices already on a chain (from [`snap`]); `seen_face` per input face (empty: all seen).
#[allow(clippy::too_many_arguments)]
pub fn split(points: &[f64], faces: &Faces, chains: &Chains, vchain: &[i64], vparam: &[f64], seen_face: &[f64], budget: f64, stop: &AtomicBool) -> Split {
    let n = points.len() / 3;
    let nf = faces.count();
    let mesh = Mesh::new(points, faces);
    let pos: Vec<V3> = (0..n).map(|v| p3(points, v)).collect();
    let chs = read_chains(chains);
    let nc = chs.len();
    let mut tag_c: Vec<i64> = if vchain.len() == n { vchain.to_vec() } else { vec![-1; n] };
    let mut tag_p: Vec<f64> = if vparam.len() == n { vparam.to_vec() } else { vec![0.0; n] };
    let active: Vec<bool> = (0..nc).map(|c| chains.seams.class[c] == 2 && chs[c].pts.len() >= 2 && chs[c].max_gap() >= SPLIT_GAP_PX).collect();
    let fverts = |f: usize| -> Vec<usize> { faces.face(f).iter().map(|&v| v as usize).collect() };
    let mut stopped = false;

    // Face edges in boundary order (edge id per face side), from the vertex pairs.
    let mut edge_of: HashMap<(usize, usize), usize> = HashMap::with_capacity(mesh.ea.len());
    for e in 0..mesh.ea.len() {
        edge_of.insert((mesh.ea[e], mesh.eb[e]), e);
    }
    let side_edge = |a: usize, b: usize| edge_of[&(a.min(b), a.max(b))];
    let fnormal: Vec<V3> = (0..nf).map(|f| unit(newell(&face_pos(&pos, faces, f)))).collect();
    let enormal: Vec<V3> = (0..mesh.ea.len())
        .map(|e| {
            let mut s = [0.0; 3];
            for &f in &mesh.efaces[e] {
                s = add(s, fnormal[f]);
            }
            s
        })
        .collect();

    // 1. Crossings per (edge, chain), once.
    // Candidates: an edge's box padded by `CROSS_REACH` of its length (a crossing is never farther), a face's box padded
    // by the `inside` plane tolerance. The grids are only a first cut; the exact box test decides.
    let ebox: Vec<(V3, V3)> = (0..mesh.ea.len())
        .map(|e| {
            let (p, q) = (pos[mesh.ea[e]], pos[mesh.eb[e]]);
            bbox(&[p, q], CROSS_REACH * norm(sub(q, p)))
        })
        .collect();
    let fbox: Vec<(V3, V3)> = (0..nf)
        .map(|f| {
            let fp = face_pos(&pos, faces, f);
            let k = fp.len();
            let perim: f64 = (0..k).map(|i| norm(sub(fp[(i + 1) % k], fp[i]))).sum();
            bbox(&fp, CROSS_REACH * perim / k as f64)
        })
        .collect();
    let mut egrid = Grid::new(mesh.mean_edge);
    for (e, &(lo, hi)) in ebox.iter().enumerate() {
        egrid.insert(e as u32, lo, hi);
    }
    let mut fgrid = Grid::new(mesh.mean_edge);
    for (f, &(lo, hi)) in fbox.iter().enumerate() {
        fgrid.insert(f as u32, lo, hi);
    }
    let hit = |b: &(V3, V3), lo: V3, hi: V3| (0..3).all(|c| b.0[c] <= hi[c] && b.1[c] >= lo[c]);
    let near_edges = |lo: V3, hi: V3| -> Vec<usize> { egrid.query(lo, hi).into_iter().map(|e| e as usize).filter(|&e| hit(&ebox[e], lo, hi)).collect() };
    let faces_at = |x: V3| -> Vec<usize> { fgrid.query(x, x).into_iter().map(|f| f as usize).filter(|&f| hit(&fbox[f], x, x)).collect() };
    let mut corner_map: HashMap<i64, Vec<usize>> = HashMap::new(); // welded corner id -> open chains ending there
    for (c, ch) in chs.iter().enumerate() {
        if !ch.closed {
            corner_map.entry(ch.start_id).or_default().push(c);
            if ch.end_id != ch.start_id {
                corner_map.entry(ch.end_id).or_default().push(c);
            }
        }
    }
    let mut crosses: HashMap<(usize, usize), (f64, Cross)> = HashMap::new();
    for c in 0..nc {
        if !active[c] {
            continue;
        }
        if stop.load(Ordering::Relaxed) {
            stopped = true;
            break;
        }
        let ch = &chs[c];
        for j in 0..ch.pts.len() - 1 {
            let (pa, pb) = (ch.pts[j], ch.pts[j + 1]);
            let (lo, hi) = bbox(&[pa, pb], 0.0);
            for e in near_edges(lo, hi) {
                let (a, b) = (mesh.ea[e], mesh.eb[e]);
                if tag_c[a] == c as i64 && tag_c[b] == c as i64 {
                    continue;
                }
                let (p, q) = (pos[a], pos[b]);
                let m = cross(sub(q, p), enormal[e]);
                let (d0, d1) = (dot(sub(pa, p), m), dot(sub(pb, p), m));
                if (d0 > 0.0) == (d1 > 0.0) {
                    continue;
                }
                let lam = d0 / (d0 - d1);
                let x = lerp(pa, pb, lam);
                let pq = sub(q, p);
                let l2 = dot(pq, pq);
                if !(l2 > 0.0) {
                    continue;
                }
                let s = dot(sub(x, p), pq) / l2;
                if !(0.0..=1.0).contains(&s) {
                    continue;
                }
                let gapd = norm(sub(x, lerp(p, q, s)));
                if gapd > CROSS_REACH * l2.sqrt() {
                    continue;
                }
                if matches!(crosses.get(&(e, c)), Some(&(old, _)) if old <= gapd) {
                    continue;
                }
                let own = mesh.efaces[e].iter().map(|&f| face_dist(&face_pos(&pos, faces, f), x)).fold(f64::INFINITY, f64::min);
                if faces_at(x).into_iter().any(|f| !mesh.efaces[e].contains(&f) && face_dist(&face_pos(&pos, faces, f), x) < NEARER_SHARE * own) {
                    continue;
                }
                let reuse = if s < REUSE_EDGE {
                    Some(a)
                } else if s > 1.0 - REUSE_EDGE {
                    Some(b)
                } else {
                    None
                };
                crosses.insert((e, c), (gapd, Cross { s, param: j as f64 + lam, reuse }));
            }
        }
    }
    let mut cross_keys: Vec<(usize, usize)> = crosses.keys().cloned().collect();
    cross_keys.sort_unstable();

    // 2. Points per face: tagged vertices, crossings (reused or new), corners inside.
    let mut fpts: HashMap<usize, Vec<(usize, Node, f64)>> = HashMap::new();
    let push = |fpts: &mut HashMap<usize, Vec<(usize, Node, f64)>>, f: usize, c: usize, node: Node, prm: f64| {
        let list = fpts.entry(f).or_default();
        if !list.iter().any(|&(cc, nn, _)| cc == c && nn == node) {
            list.push((c, node, prm));
        }
    };
    // Tagged vertices; one on a seam corner stands for every chain that ends there.
    let mut end_tagged = vec![[false; 2]; nc];
    for v in 0..n {
        if tag_c[v] < 0 {
            continue;
        }
        let (c, prm) = (tag_c[v] as usize, tag_p[v]);
        let mut on: Vec<(usize, f64)> = vec![(c, prm)];
        if !chs[c].closed && (prm == 0.0 || prm == chs[c].end()) {
            let id = if prm == 0.0 { chs[c].start_id } else { chs[c].end_id };
            for &cc in &corner_map[&id] {
                if cc != c {
                    on.push((cc, if chs[cc].start_id == id { 0.0 } else { chs[cc].end() }));
                }
            }
        }
        for (cc, pp) in on {
            if pp == 0.0 {
                end_tagged[cc][0] = true;
            }
            if pp == chs[cc].end() {
                end_tagged[cc][1] = true;
            }
            if active[cc] {
                for &f in mesh.faces_of(v) {
                    push(&mut fpts, f, cc, Node::Vertex(v), pp);
                }
            }
        }
    }
    let mut vgrid = Grid::new(mesh.mean_edge);
    for (v, &p) in pos.iter().enumerate() {
        vgrid.insert(v as u32, p, p);
    }
    // Vertices a crossing reuses become seam vertices (per-side UVs) even where no face needs a cut.
    let mut reuse_tags: Vec<(usize, usize, f64)> = Vec::new();
    for &(e, c) in &cross_keys {
        let cr = crosses[&(e, c)].1;
        if let Some(v) = cr.reuse {
            reuse_tags.push((v, c, cr.param));
        }
        let node = match cr.reuse {
            Some(v) => Node::Vertex(v),
            None => Node::Edge(e, c),
        };
        let fs = match cr.reuse {
            Some(v) => mesh.faces_of(v).to_vec(),
            None => mesh.efaces[e].clone(),
        };
        for f in fs {
            push(&mut fpts, f, c, node, cr.param);
        }
    }
    let mut corner_at: HashMap<i64, (usize, V3)> = HashMap::new(); // welded corner id -> (face, point)
    for c in 0..nc {
        if !active[c] || chs[c].closed {
            continue;
        }
        let ch = &chs[c];
        for (k, (id, prm)) in [(ch.start_id, 0.0), (ch.end_id, ch.end())].into_iter().enumerate() {
            let x = ch.at(prm);
            if end_tagged[c][k] {
                continue;
            }
            let near = nearest_vertex(&pos, &vgrid, x);
            if norm(sub(pos[near], x)) < REUSE_EDGE * mesh.vedge[near] {
                reuse_tags.push((near, c, prm));
                for &g in mesh.faces_of(near) {
                    push(&mut fpts, g, c, Node::Vertex(near), prm);
                }
                continue;
            }
            let Some(f) = faces_at(x).into_iter().find(|&f| inside(&face_pos(&pos, faces, f), x)) else { continue };
            let fv = fverts(f);
            // A corner on (or within `REUSE_EDGE` of) one of the face's edges is a crossing of that edge, not a point
            // inside the face.
            let k = fv.len();
            let mut on_edge: Option<(f64, usize, f64)> = None;
            for i in 0..k {
                let e = side_edge(fv[i], fv[(i + 1) % k]);
                let (p, q) = (pos[mesh.ea[e]], pos[mesh.eb[e]]);
                let pq = sub(q, p);
                let l2 = dot(pq, pq);
                if !(l2 > 0.0) {
                    continue;
                }
                let s = (dot(sub(x, p), pq) / l2).clamp(0.0, 1.0);
                let d = norm(sub(x, lerp(p, q, s)));
                if d < REUSE_EDGE * l2.sqrt() && on_edge.is_none_or(|(b, _, _)| d < b) {
                    on_edge = Some((d, e, s));
                }
            }
            if let Some((_, e, s)) = on_edge {
                if let std::collections::hash_map::Entry::Vacant(slot) = crosses.entry((e, c)) {
                    slot.insert((0.0, Cross { s, param: prm, reuse: None }));
                    for &g in &mesh.efaces[e] {
                        push(&mut fpts, g, c, Node::Edge(e, c), prm);
                    }
                }
            } else {
                corner_at.entry(id).or_insert((f, x));
                if corner_at[&id].0 == f {
                    push(&mut fpts, f, c, Node::Corner(id), prm);
                }
            }
        }
    }

    // 3. Plans per face.
    let mut face_list: Vec<usize> = fpts.keys().cloned().collect();
    face_list.sort_unstable();
    let mut plans: HashMap<usize, Plan> = HashMap::new();
    for &f in &face_list {
        let fv = fverts(f);
        let k = fv.len();
        let fe: Vec<usize> = (0..k).map(|i| side_edge(fv[i], fv[(i + 1) % k])).collect();
        let fp = face_pos(&pos, faces, f);
        let mut list = fpts[&f].clone();
        list.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.total_cmp(&b.2)).then(a.1.cmp(&b.1)));
        let mut plan = Plan { chords: Vec::new(), fan: None, cannot: false, whole: false };
        let mut corners: Vec<Node> = Vec::new();
        let mut i = 0;
        while i < list.len() {
            let c = list[i].0;
            let mut j = i;
            while j < list.len() && list[j].0 == c {
                j += 1;
            }
            let pts = &list[i..j];
            let ch = &chs[c];
            let mut pairs: Vec<(usize, usize, f64)> = (0..pts.len().saturating_sub(1)).map(|q| (q, q + 1, 0.5 * (pts[q].2 + pts[q + 1].2))).collect();
            if ch.closed && pts.len() >= 2 {
                let mut mid = 0.5 * (pts[pts.len() - 1].2 + pts[0].2 + ch.end());
                if mid > ch.end() {
                    mid -= ch.end();
                }
                pairs.push((pts.len() - 1, 0, mid));
            }
            // A point is explained by a chord through it, or by a neighbour on the same edge (the chain runs along it).
            let mut explained = vec![false; pts.len()];
            for (x, y, mid) in pairs {
                let (na, nb) = (pts[x].1, pts[y].1);
                if na == nb {
                    continue;
                }
                let (ea, eb) = (node_edges(na, &fv, &fe), node_edges(nb, &fv, &fe));
                if ea.iter().any(|e| eb.contains(e)) {
                    explained[x] = true;
                    explained[y] = true;
                    continue;
                }
                if !inside(&fp, ch.at(mid)) {
                    continue;
                }
                explained[x] = true;
                explained[y] = true;
                if plan.chords.iter().any(|&(a, b, _, _, _)| (a == na && b == nb) || (a == nb && b == na)) {
                    continue;
                }
                plan.chords.push((na, nb, c, pts[x].2, pts[y].2));
            }
            for p in pts {
                if let Node::Corner(_) = p.1 {
                    if !corners.contains(&p.1) {
                        corners.push(p.1);
                    }
                }
            }
            if pts.iter().zip(&explained).any(|(p, &ok)| !ok && matches!(p.1, Node::Edge(..))) {
                plan.cannot = true;
            }
            i = j;
        }
        if let Some(&cn) = corners.first() {
            let spokes: Vec<(Node, usize, f64)> = plan
                .chords
                .iter()
                .filter_map(|&(a, b, c, pa, pb)| {
                    if a == cn {
                        Some((b, c, pb))
                    } else if b == cn {
                        Some((a, c, pa))
                    } else {
                        None
                    }
                })
                .collect();
            let has_edge = list.iter().any(|p| matches!(p.1, Node::Edge(..)));
            if corners.len() == 1 && spokes.len() >= 2 {
                plan.fan = Some((cn, spokes));
                plan.chords.retain(|&(a, b, _, _, _)| a != cn && b != cn);
            } else if plan.chords.is_empty() && !has_edge {
                plan.whole = true; // nothing leaves the face: an island (or a piece of seam) inside it
            } else {
                plan.cannot = true;
            }
        }
        if plan.cannot {
            plan.chords.clear();
            plan.fan = None;
        }
        plans.insert(f, plan);
    }

    // 4. Used crossing vertices and the runs of faces that must be split together.
    let mut used: Vec<Node> = Vec::new();
    for &f in &face_list {
        let p = &plans[&f];
        for &(a, b, _, _, _) in &p.chords {
            used.push(a);
            used.push(b);
        }
        if let Some((cn, sp)) = &p.fan {
            used.push(*cn);
            used.extend(sp.iter().map(|x| x.0));
        }
    }
    used.sort_unstable();
    used.dedup();
    let is_used = |node: &Node| used.binary_search(node).is_ok();
    let mut dsu = Dsu::new(nf);
    for node in &used {
        if let Node::Edge(e, _) = node {
            let fs = &mesh.efaces[*e];
            for w in fs.windows(2) {
                dsu.union(w[0] as u32, w[1] as u32);
            }
        }
    }
    // Members: faces with a plan that cuts, and faces that only get a used crossing on an edge.
    let mut touched: Vec<usize> = Vec::new();
    for &f in &face_list {
        let p = &plans[&f];
        if !p.chords.is_empty() || p.fan.is_some() {
            touched.push(f);
        }
    }
    for node in &used {
        if let Node::Edge(e, _) = node {
            touched.extend(mesh.efaces[*e].iter().cloned());
        }
    }
    touched.sort_unstable();
    touched.dedup();

    // Node positions: existing vertices, then used crossings and corners (ids assigned in order).
    let mut at: Vec<V3> = pos.clone();
    let mut node_id: HashMap<Node, usize> = HashMap::new();
    let mut new_tag: Vec<(i64, f64)> = Vec::new();
    for node in &used {
        match *node {
            Node::Vertex(v) => {
                node_id.insert(*node, v);
            }
            Node::Edge(e, c) => {
                let cr = crosses[&(e, c)].1;
                node_id.insert(*node, at.len());
                at.push(lerp(pos[mesh.ea[e]], pos[mesh.eb[e]], cr.s));
                new_tag.push((c as i64, cr.param));
            }
            Node::Corner(_) => {}
        }
    }
    for node in &used {
        if let Node::Corner(id) = *node {
            let (f, x) = corner_at[&id];
            let fp = face_pos(&pos, faces, f);
            let nrm = fnormal[f];
            let cen = scale(fp.iter().fold([0.0; 3], |s, &p| add(s, p)), 1.0 / fp.len() as f64);
            let onto = sub(x, scale(nrm, dot(sub(x, cen), nrm)));
            node_id.insert(*node, at.len());
            at.push(onto);
            let c = corner_map[&id].iter().cloned().find(|&c| active[c]).unwrap();
            new_tag.push((c as i64, if chs[c].start_id == id { 0.0 } else { chs[c].end() }));
        }
    }

    // Pieces per touched face.
    let mut cross_by_edge: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut all_keys: Vec<(usize, usize)> = crosses.keys().cloned().collect();
    all_keys.sort_unstable();
    for (e, c) in all_keys {
        cross_by_edge.entry(e).or_default().push(c);
    }
    let pieces_of = |f: usize| -> (Vec<Vec<usize>>, bool) {
        let fv = fverts(f);
        let k = fv.len();
        let mut ring: Vec<usize> = Vec::new();
        for i in 0..k {
            ring.push(fv[i]);
            let (a, b) = (fv[i], fv[(i + 1) % k]);
            let e = side_edge(a, b);
            let mut on: Vec<(f64, usize)> = Vec::new();
            for &c in cross_by_edge.get(&e).map(|v| v.as_slice()).unwrap_or(&[]) {
                let cr = crosses[&(e, c)].1;
                let node = Node::Edge(e, c);
                if cr.reuse.is_none() && is_used(&node) {
                    let s = if mesh.ea[e] == a { cr.s } else { 1.0 - cr.s };
                    on.push((s, node_id[&node]));
                }
            }
            on.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
            ring.extend(on.into_iter().map(|x| x.1));
        }
        let plan = plans.get(&f);
        let mut pieces: Vec<Vec<usize>> = vec![ring.clone()];
        let mut cut = false;
        if let Some(p) = plan {
            if let Some((cn, spokes)) = &p.fan {
                let cid = node_id[cn];
                let mut ends: Vec<(usize, usize)> = spokes
                    .iter()
                    .map(|s| {
                        let id = node_id[&s.0];
                        (ring.iter().position(|&x| x == id).unwrap(), id)
                    })
                    .collect();
                ends.sort_unstable();
                pieces.clear();
                for q in 0..ends.len() {
                    let (s0, e0) = (ends[q].0, ends[(q + 1) % ends.len()].0);
                    let mut piece = vec![cid];
                    let mut i = s0;
                    loop {
                        piece.push(ring[i]);
                        if i == e0 {
                            break;
                        }
                        i = (i + 1) % ring.len();
                    }
                    pieces.push(piece);
                }
                cut = true;
            }
            for &(a, b, _, _, _) in &p.chords {
                let (ia, ib) = (node_id[&a], node_id[&b]);
                let Some(w) = pieces.iter().position(|pc| pc.contains(&ia) && pc.contains(&ib)) else { continue };
                let pc = pieces.remove(w);
                let (x, y) = (pc.iter().position(|&v| v == ia).unwrap(), pc.iter().position(|&v| v == ib).unwrap());
                let (x, y) = (x.min(y), x.max(y));
                if y - x == 1 || (x == 0 && y == pc.len() - 1) {
                    pieces.push(pc);
                    continue;
                }
                pieces.push(pc[x..=y].to_vec());
                let mut other: Vec<usize> = pc[y..].to_vec();
                other.extend_from_slice(&pc[..=x]);
                pieces.push(other);
                cut = true;
            }
        }
        let nrm = fnormal[f];
        let mut out = Vec::new();
        for pc in pieces {
            out.extend(to_quads(&pc, &at, nrm));
        }
        (out, cut)
    };

    // 5. Budget: runs (faces joined through a new crossing vertex), most damage first.
    let mut runs: HashMap<u32, Vec<usize>> = HashMap::new();
    for &f in &touched {
        runs.entry(dsu.find(f as u32)).or_default().push(f);
    }
    let mut run_list: Vec<(f64, usize, Vec<usize>, i64)> = Vec::new();
    let mut piece_cache: HashMap<usize, (Vec<Vec<usize>>, bool)> = HashMap::new();
    let mut keys: Vec<u32> = runs.keys().cloned().collect();
    keys.sort_unstable();
    for r in keys {
        let members = runs[&r].clone();
        let (mut damage, mut cost) = (0.0, 0i64);
        for &f in &members {
            let pc = pieces_of(f);
            cost += pc.0.len() as i64 - 1;
            if let Some(p) = plans.get(&f) {
                let mut g: f64 = 0.0;
                for &(_, _, c, pa, pb) in &p.chords {
                    g = g.max(chs[c].gap_at(pa)).max(chs[c].gap_at(pb));
                }
                if let Some((Node::Corner(id), _)) = &p.fan {
                    for &c in &corner_map[id] {
                        if active[c] {
                            g = g.max(chs[c].max_gap());
                        }
                    }
                }
                let area = 0.5 * norm(newell(&face_pos(&pos, faces, f)));
                let seen = if seen_face.len() == nf { seen_face[f] } else { 1.0 };
                damage += area * seen * g;
            }
            piece_cache.insert(f, pc);
        }
        run_list.push((damage, members[0], members, cost));
    }
    run_list.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let wanted: f64 = run_list.iter().map(|r| r.3 as f64).sum();
    let mut chosen = vec![false; nf];
    let mut spent = 0.0;
    for (_, _, members, cost) in &run_list {
        if budget < 0.0 || spent + *cost as f64 <= budget {
            spent += *cost as f64;
            for &f in members {
                chosen[f] = true;
            }
        }
    }

    // States.
    let mut state = vec![FACE_KEPT; nf];
    for &f in &face_list {
        let p = &plans[&f];
        if p.whole {
            state[f] = FACE_BAKE;
        } else if p.cannot {
            state[f] = FACE_CANNOT;
        } else if !p.chords.is_empty() || p.fan.is_some() {
            state[f] = if chosen[f] { FACE_SPLIT } else { FACE_LEFT };
        }
    }
    // Whole islands: closed chains that touch no face at all.
    let mut chain_touched = vec![false; nc];
    for f in &face_list {
        for p in &fpts[f] {
            chain_touched[p.0] = true;
        }
    }
    for c in 0..nc {
        if !active[c] || !chs[c].closed || chain_touched[c] {
            continue;
        }
        let ch = &chs[c];
        let cen = scale(ch.pts[..ch.pts.len() - 1].iter().fold([0.0; 3], |s, &p| add(s, p)), 1.0 / (ch.pts.len() - 1) as f64);
        if let Some(f) = faces_at(cen).into_iter().find(|&f| inside(&face_pos(&pos, faces, f), cen)) {
            state[f] = FACE_BAKE;
        }
    }

    // 6. Output mesh: kept faces copied, chosen faces replaced by their pieces (new vertices only from chosen runs).
    let mut keep_node = vec![false; at.len()];
    for v in 0..n {
        keep_node[v] = true;
    }
    for f in 0..nf {
        if chosen[f] {
            for pc in &piece_cache[&f].0 {
                for &v in pc {
                    keep_node[v] = true;
                }
            }
        }
    }
    let mut remap = vec![usize::MAX; at.len()];
    let mut out_pts: Vec<V3> = Vec::new();
    for (i, &k) in keep_node.iter().enumerate() {
        if k {
            remap[i] = out_pts.len();
            out_pts.push(at[i]);
        }
    }
    for i in n..at.len() {
        if keep_node[i] {
            let (c, prm) = new_tag[i - n];
            tag_c.push(c);
            tag_p.push(prm);
        }
    }
    for &(v, c, prm) in &reuse_tags {
        if tag_c[v] < 0 {
            tag_c[v] = c as i64;
            tag_p[v] = prm;
        }
    }
    // Reused vertices that end a chord in a chosen face become seam vertices.
    for &f in &face_list {
        if !chosen[f] {
            continue;
        }
        let p = &plans[&f];
        let mut ends: Vec<(Node, usize, f64)> = p.chords.iter().flat_map(|&(a, b, c, pa, pb)| [(a, c, pa), (b, c, pb)]).collect();
        if let Some((_, sp)) = &p.fan {
            ends.extend(sp.iter().cloned());
        }
        for (node, c, prm) in ends {
            if let Node::Vertex(v) = node {
                if tag_c[v] < 0 {
                    tag_c[v] = c as i64;
                    tag_p[v] = prm;
                }
            }
        }
    }
    let mut polys: Vec<(Vec<usize>, usize, bool)> = Vec::new(); // (vertices, origin, from split)
    for f in 0..nf {
        if chosen[f] {
            for pc in &piece_cache[&f].0 {
                polys.push((pc.iter().map(|&v| remap[v]).collect(), f, true));
            }
        } else {
            polys.push((fverts(f), f, false));
        }
    }
    // Pair split triangles with a same-side neighbour (not across a cut) when the quad is valid.
    let mut seam_edge: Vec<(usize, usize)> = Vec::new();
    for &f in &face_list {
        if !chosen[f] {
            continue;
        }
        let p = &plans[&f];
        for &(a, b, _, _, _) in &p.chords {
            let (x, y) = (remap[node_id[&a]], remap[node_id[&b]]);
            seam_edge.push((x.min(y), x.max(y)));
        }
        if let Some((cn, sp)) = &p.fan {
            let x = remap[node_id[cn]];
            for s in sp {
                let y = remap[node_id[&s.0]];
                seam_edge.push((x.min(y), x.max(y)));
            }
        }
    }
    seam_edge.sort_unstable();
    let mut by_edge: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, (pv, _, from)) in polys.iter().enumerate() {
        if *from && pv.len() == 3 {
            for k in 0..3 {
                let (a, b) = (pv[k], pv[(k + 1) % 3]);
                by_edge.entry((a.min(b), a.max(b))).or_default().push(i);
            }
        }
    }
    let mut gone = vec![false; polys.len()];
    for i in 0..polys.len() {
        if gone[i] || !polys[i].2 || polys[i].0.len() != 3 {
            continue;
        }
        let tri = polys[i].0.clone();
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            if seam_edge.binary_search(&key).is_ok() {
                continue;
            }
            let Some(&j) = by_edge[&key].iter().find(|&&j| j != i && !gone[j] && polys[j].0.len() == 3) else { continue };
            let other = &polys[j].0;
            let apex = *other.iter().find(|&&v| v != a && v != b).unwrap();
            let opp = tri[(k + 2) % 3];
            let quad = vec![a, apex, b, opp];
            let p: Vec<V3> = quad.iter().map(|&v| out_pts[v]).collect();
            let want = add(newell(&tri.iter().map(|&v| out_pts[v]).collect::<Vec<_>>()), newell(&other.iter().map(|&v| out_pts[v]).collect::<Vec<_>>()));
            // The shared edge runs b -> a in the other triangle, so the quad is a, apex, b, opp.
            if face_ok(&p) && dot(newell(&p), want) > 0.0 {
                polys[i].0 = quad;
                gone[j] = true;
                break;
            }
        }
    }
    let mut indptr = vec![0i64];
    let mut flat: Vec<i64> = Vec::new();
    let mut origin: Vec<i64> = Vec::new();
    let mut triangles_left = 0;
    for (i, (pv, f, from)) in polys.iter().enumerate() {
        if gone[i] {
            continue;
        }
        if *from && pv.len() == 3 {
            triangles_left += 1;
        }
        flat.extend(pv.iter().map(|&v| v as i64));
        indptr.push(flat.len() as i64);
        origin.push(*f as i64);
    }

    // 7. Per-side UVs at seam vertices, straight from the chain polylines (side A = left of the chain direction).
    let mut corner_uv = vec![f64::NAN; 2 * flat.len()];
    let mut corner_side = vec![-1i8; flat.len()];
    for f in 0..indptr.len() - 1 {
        let (s, e) = (indptr[f] as usize, indptr[f + 1] as usize);
        let fp: Vec<V3> = flat[s..e].iter().map(|&v| out_pts[v as usize]).collect();
        let nrm = newell(&fp);
        let cen = scale(fp.iter().fold([0.0; 3], |a, &p| add(a, p)), 1.0 / fp.len() as f64);
        for h in s..e {
            let v = flat[h] as usize;
            if tag_c[v] < 0 {
                continue;
            }
            let (mut c, mut prm) = (tag_c[v] as usize, tag_p[v]);
            let x = out_pts[v];
            if !chs[c].closed && (prm == 0.0 || prm == chs[c].end()) {
                let id = if prm == 0.0 { chs[c].start_id } else { chs[c].end_id };
                let mut best = (f64::INFINITY, c, prm);
                for &cc in &corner_map[&id] {
                    let p0 = if chs[cc].start_id == id { 0.0 } else { chs[cc].end() };
                    let (d2, _) = chs[cc].closest(0, chs[cc].pts.len() - 1, cen);
                    if d2 < best.0 {
                        best = (d2, cc, p0);
                    }
                }
                c = best.1;
                prm = best.2;
            }
            let ch = &chs[c];
            let side_b = dot(cross(ch.tangent(prm), sub(cen, x)), nrm) <= 0.0;
            let uv = ch.uv(prm, side_b);
            corner_uv[2 * h] = uv[0];
            corner_uv[2 * h + 1] = uv[1];
            corner_side[h] = side_b as i8;
        }
    }
    let faces_split = (0..nf).filter(|&f| state[f] == FACE_SPLIT).count();
    let new_faces = indptr.len() as i64 - 1;
    Split {
        points: out_pts.iter().flatten().cloned().collect(),
        indptr,
        flat,
        origin,
        faces_split,
        extra_faces: new_faces - nf as i64,
        triangles_left,
        faces_left: state.iter().filter(|&&s| s == FACE_LEFT).count(),
        faces_cannot: state.iter().filter(|&&s| s == FACE_CANNOT).count(),
        faces_bake: state.iter().filter(|&&s| s == FACE_BAKE).count(),
        face_state: state,
        vchain: tag_c,
        vparam: tag_p,
        corner_uv,
        corner_side,
        spent,
        wanted,
        stopped,
    }
}

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let unit = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    usize::from(quad_valid(&unit, 1e-9, 0.5))
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }
}
