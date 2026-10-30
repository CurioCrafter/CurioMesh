// SPDX-License-Identifier: GPL-3.0-or-later
//! Combinatorial mesh topology: edge occurrences, summaries, closures, parity paths.
//!
//! Every function mirrors a Python function in `triadq/topology.py` and keeps its
//! iteration order wherever the order is observable (dict insertion order, error
//! messages, tie-breaking), so results are identical, not merely equivalent.

use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// Polygon faces in CSR form: face `f` is `flat[indptr[f]..indptr[f + 1]]`.
pub struct Faces<'a> {
    pub indptr: &'a [i64],
    pub flat: &'a [i64],
}

impl<'a> Faces<'a> {
    pub fn count(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }
    pub fn face(&self, f: usize) -> &'a [i64] {
        &self.flat[self.indptr[f] as usize..self.indptr[f + 1] as usize]
    }
}

/// Below this many items the rayon pool costs more than it saves (worker wake-up
/// is ~1 ms on a busy machine, and triadq calls these on many small patches).
pub const PARALLEL_MIN: usize = 1 << 15;

pub fn sort<T: Ord + Send>(items: &mut [T]) {
    if items.len() >= PARALLEL_MIN {
        items.par_sort_unstable()
    } else {
        items.sort_unstable()
    }
}

pub fn collect<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if n >= PARALLEL_MIN {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

pub struct Dsu(Vec<u32>);

impl Dsu {
    pub fn new(n: usize) -> Self {
        Dsu((0..n as u32).collect())
    }
    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            let grand = self.0[self.0[x as usize] as usize];
            self.0[x as usize] = grand;
            x = grand;
        }
        x
    }
    pub fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra < rb {
            self.0[rb as usize] = ra;
        } else if rb < ra {
            self.0[ra as usize] = rb;
        }
    }
}

/// One entry per polygon corner `h`: its face and directed edge from -> to.
pub struct Halves {
    pub face: Vec<u32>,
    pub from: Vec<i64>,
    pub to: Vec<i64>,
}

pub fn halves(faces: &Faces) -> Halves {
    let n = faces.flat.len();
    let (mut face, mut from, mut to) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for f in 0..faces.count() {
        let poly = faces.face(f);
        for (c, &v) in poly.iter().enumerate() {
            face.push(f as u32);
            from.push(v);
            to.push(poly[(c + 1) % poly.len()]);
        }
    }
    Halves { face, from, to }
}

/// Unique undirected edges in first-occurrence order (Python dict insertion order);
/// `occ[start[e]..start[e + 1]]` are the corners using edge `e`, in corner order.
pub struct Edges {
    pub a: Vec<i64>,
    pub b: Vec<i64>,
    pub start: Vec<usize>,
    pub occ: Vec<u32>,
}

impl Edges {
    pub fn len(&self) -> usize {
        self.a.len()
    }
    pub fn uses(&self, e: usize) -> &[u32] {
        &self.occ[self.start[e]..self.start[e + 1]]
    }
}

pub fn edges(h: &Halves) -> Edges {
    let n = h.from.len();
    let mut keys: Vec<(i64, i64, u32)> = collect(n, |i| {
        let (x, y) = (h.from[i], h.to[i]);
        if x <= y {
            (x, y, i as u32)
        } else {
            (y, x, i as u32)
        }
    });
    sort(&mut keys);
    let mut groups: Vec<(u32, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && keys[j].0 == keys[i].0 && keys[j].1 == keys[i].1 {
            j += 1;
        }
        groups.push((keys[i].2, i, j));
        i = j;
    }
    sort(&mut groups); // first corners are unique, so this orders by first occurrence
    let mut out = Edges {
        a: Vec::with_capacity(groups.len()),
        b: Vec::with_capacity(groups.len()),
        start: Vec::with_capacity(groups.len() + 1),
        occ: Vec::with_capacity(n),
    };
    out.start.push(0);
    for &(_, s, e) in &groups {
        out.a.push(keys[s].0);
        out.b.push(keys[s].1);
        out.occ.extend(keys[s..e].iter().map(|k| k.2));
        out.start.push(out.occ.len());
    }
    out
}

#[derive(Default)]
pub struct Summary {
    pub used_vertices: i64,
    pub edges: i64,
    pub faces: i64,
    pub quads: i64,
    pub boundary_edges: i64,
    pub boundary_components: i64,
    pub branched_boundary_vertices: i64,
    pub non_manifold_edges: i64,
    pub non_manifold_vertices: i64,
    pub inconsistent_edges: i64,
    pub duplicate_faces: i64,
    pub euler_characteristic: i64,
    pub extraordinary_vertices: i64,
    pub defect_charge: i64,
    /// Sorted (euler, boundary cycles, face count) per connected component.
    pub component_detail: Vec<(i64, i64, i64)>,
}

fn canonical(face: &[i64]) -> Vec<i64> {
    let n = face.len();
    let at = |start: usize, rev: bool, k: usize| {
        let i = (start + k) % n;
        if rev {
            face[n - 1 - i]
        } else {
            face[i]
        }
    };
    let (mut best_s, mut best_r) = (0usize, false);
    for rev in [false, true] {
        for s in 0..n {
            for k in 0..n {
                let (x, y) = (at(s, rev, k), at(best_s, best_r, k));
                if x != y {
                    if x < y {
                        best_s = s;
                        best_r = rev;
                    }
                    break;
                }
            }
        }
    }
    (0..n).map(|k| at(best_s, best_r, k)).collect()
}

/// Mirrors the combinatorial part of `topology_summary` (everything but the
/// geometric `degenerate_faces` / `invalid_quads`), including
/// `_non_manifold_vertex_count` and `_component_topology_signatures`.
/// Vertex ids must be non-negative.
pub fn summary(faces: &Faces) -> Summary {
    let fcount = faces.count();
    let mut out = Summary {
        faces: fcount as i64,
        ..Default::default()
    };
    if fcount == 0 {
        return out;
    }
    let h = halves(faces);
    let e = edges(&h);
    let nv = faces.flat.iter().copied().max().unwrap_or(-1).max(-1) as usize + 1;
    out.edges = e.len() as i64;
    out.quads = (0..fcount).filter(|&f| faces.face(f).len() == 4).count() as i64;

    let mut used = vec![false; nv];
    for &v in faces.flat {
        used[v as usize] = true;
    }
    out.used_vertices = used.iter().filter(|&&u| u).count() as i64;

    let mut boundary_degree = vec![0u32; nv];
    let mut valence = vec![0u32; nv];
    let mut boundary_dsu = Dsu::new(nv);
    for i in 0..e.len() {
        let (a, b) = (e.a[i] as usize, e.b[i] as usize);
        let uses = e.uses(i);
        valence[a] += 1;
        if b != a {
            valence[b] += 1;
        }
        match uses.len() {
            1 => {
                out.boundary_edges += 1;
                boundary_degree[a] += 1;
                if b != a {
                    boundary_degree[b] += 1;
                }
                boundary_dsu.union(a as u32, b as u32);
            }
            2 => {
                let (h0, h1) = (uses[0] as usize, uses[1] as usize);
                if h.from[h0] == h.from[h1] && h.to[h0] == h.to[h1] {
                    out.inconsistent_edges += 1;
                }
            }
            n if n > 2 => out.non_manifold_edges += 1,
            _ => {}
        }
    }
    for v in 0..nv {
        if boundary_degree[v] > 0 {
            if boundary_dsu.find(v as u32) == v as u32 {
                out.boundary_components += 1;
            }
            if boundary_degree[v] != 2 {
                out.branched_boundary_vertices += 1;
            }
        }
        if used[v] {
            let ideal = if boundary_degree[v] > 0 { 3 } else { 4 };
            if valence[v] != ideal {
                out.extraordinary_vertices += 1;
            }
            out.defect_charge += ideal as i64 - valence[v] as i64;
        }
    }

    let mut canon: Vec<Vec<i64>> = collect(fcount, |f| canonical(faces.face(f)));
    sort(&mut canon);
    canon.dedup();
    out.duplicate_faces = (fcount - canon.len()) as i64;

    // Distinct (vertex, face) incidences; a vertex is non-manifold when its
    // incident faces split into several fans linked by two-face edges.
    let mut vf: Vec<(i64, u32)> = h.from.iter().zip(&h.face).map(|(&v, &f)| (v, f)).collect();
    sort(&mut vf);
    vf.dedup();
    let index = |v: i64, f: u32| vf.binary_search(&(v, f)).unwrap() as u32;
    let mut fan = Dsu::new(vf.len());
    let mut face_dsu = Dsu::new(fcount);
    let mut linked: Vec<u32> = Vec::new();
    for i in 0..e.len() {
        linked.clear();
        linked.extend(e.uses(i).iter().map(|&c| h.face[c as usize]));
        linked.sort_unstable();
        linked.dedup();
        for &f in &linked[1..] {
            face_dsu.union(linked[0], f);
        }
        if linked.len() == 2 {
            for v in [e.a[i], e.b[i]] {
                fan.union(index(v, linked[0]), index(v, linked[1]));
            }
        }
    }
    let mut i = 0;
    while i < vf.len() {
        let first = fan.find(i as u32);
        let mut j = i + 1;
        let mut split = false;
        while j < vf.len() && vf[j].0 == vf[i].0 {
            split |= fan.find(j as u32) != first;
            j += 1;
        }
        out.non_manifold_vertices += split as i64;
        i = j;
    }

    // Components by edge-sharing faces.
    let mut label = vec![u32::MAX; fcount];
    let mut comp_of = vec![0u32; fcount];
    let mut ncomp = 0u32;
    for f in 0..fcount {
        let r = face_dsu.find(f as u32) as usize;
        if label[r] == u32::MAX {
            label[r] = ncomp;
            ncomp += 1;
        }
        comp_of[f] = label[r];
    }
    let nc = ncomp as usize;
    let (mut fc, mut ec, mut vc, mut cycles) = (
        vec![0i64; nc],
        vec![0i64; nc],
        vec![0i64; nc],
        vec![0i64; nc],
    );
    for f in 0..fcount {
        fc[comp_of[f] as usize] += 1;
    }
    for i in 0..e.len() {
        ec[comp_of[h.face[e.occ[e.start[i]] as usize] as usize] as usize] += 1;
    }
    let mut cv: Vec<(u32, i64)> = vf.iter().map(|&(v, f)| (comp_of[f as usize], v)).collect();
    sort(&mut cv);
    cv.dedup();
    for &(c, _) in &cv {
        vc[c as usize] += 1;
    }
    let cindex = |c: u32, v: i64| cv.binary_search(&(c, v)).unwrap() as u32;
    let mut ring = Dsu::new(cv.len());
    let mut marked = vec![false; cv.len()];
    for i in 0..e.len() {
        let uses = e.uses(i);
        if uses.len() != 1 {
            continue;
        }
        let c = comp_of[h.face[uses[0] as usize] as usize];
        let (x, y) = (cindex(c, e.a[i]), cindex(c, e.b[i]));
        marked[x as usize] = true;
        marked[y as usize] = true;
        ring.union(x, y);
    }
    for n in 0..cv.len() {
        if marked[n] && ring.find(n as u32) == n as u32 {
            cycles[cv[n].0 as usize] += 1;
        }
    }
    out.component_detail = (0..nc)
        .map(|c| (vc[c] - ec[c] + fc[c], cycles[c], fc[c]))
        .collect();
    out.component_detail.sort_unstable();
    out.euler_characteristic = out.used_vertices - out.edges + out.faces;
    out
}

/// Faces with fan area <= `epsilon` or a repeated vertex (`_face_areas(...) <= eps`).
pub fn degenerate_count(faces: &Faces, vertices: &[f64], epsilon: f64) -> i64 {
    collect(faces.count(), |f| {
        let poly = faces.face(f);
        if poly.len() < 3 {
            return true; // area stays 0 in Python, and epsilon >= 1e-20
        }
        let p = |k: usize| {
            let v = poly[k] as usize * 3;
            [vertices[v], vertices[v + 1], vertices[v + 2]]
        };
        let p0 = p(0);
        let rel = |k: usize| {
            let q = p(k);
            [q[0] - p0[0], q[1] - p0[1], q[2] - p0[2]]
        };
        let mut sum = [0.0f64; 3];
        for k in 1..poly.len() - 1 {
            let (a, b) = (rel(k), rel(k + 1));
            let c = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            for j in 0..3 {
                sum[j] = if k == 1 { c[j] } else { sum[j] + c[j] };
            }
        }
        let mut area = 0.5 * (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
        let mut sorted = poly.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            area = 0.0;
        }
        area <= epsilon
    })
    .into_iter()
    .filter(|&degenerate| degenerate)
    .count() as i64
}

pub struct Closure {
    pub vertices: Vec<f64>,
    pub quads: Vec<i64>,
    pub uvs: Option<Vec<f64>>,
    pub face_uvs: Option<Vec<i64>>,
}

/// Mirrors `quad_only_closure` up to (not including) building MeshData and validating it.
pub fn quad_closure(
    faces: &Faces,
    vertices: &[f64],
    frozen: &[(i64, i64, i64)],
    uvs: Option<&[f64]>,
    face_uvs: Option<&Faces>,
) -> Result<Closure, String> {
    let nv = (vertices.len() / 3) as i64;
    let mut out_v = vertices.to_vec();
    let mut mids: HashMap<(i64, i64), i64> = frozen.iter().map(|&(a, b, m)| ((a, b), m)).collect();
    let mut quads = Vec::with_capacity(faces.flat.len() * 4);
    let mut out_uvs = uvs.map(|u| u.to_vec());
    let mut out_fuv: Option<Vec<i64>> = if uvs.is_some() && face_uvs.is_some() {
        Some(Vec::new())
    } else {
        None
    };
    let nuv = uvs.map_or(0, |u| (u.len() / 2) as i64);
    let mut local: Vec<i64> = Vec::new();
    let mut seen: Vec<i64> = Vec::new();
    let mut uv_mids: Vec<i64> = Vec::new();
    for f in 0..faces.count() {
        let face = faces.face(f);
        let n = face.len();
        if n < 3 {
            return Err(format!("Face {f} has fewer than three corners."));
        }
        seen.clear();
        seen.extend_from_slice(face);
        seen.sort_unstable();
        if seen.windows(2).any(|w| w[0] == w[1]) {
            return Err(format!(
                "Face {f} repeats a vertex and cannot be closed safely."
            ));
        }
        if face.iter().any(|&v| v < 0 || v >= nv) {
            return Err(format!("Face {f} has an out-of-range vertex index."));
        }
        local.clear();
        for c in 0..n {
            let (v, w) = (face[c], face[(c + 1) % n]);
            let key = if v <= w { (v, w) } else { (w, v) };
            let next = (out_v.len() / 3) as i64;
            let m = *mids.entry(key).or_insert_with(|| {
                for j in 0..3 {
                    let x = (vertices[v as usize * 3 + j] + vertices[w as usize * 3 + j]) * 0.5;
                    out_v.push(x);
                }
                next
            });
            local.push(m);
        }
        let center = (out_v.len() / 3) as i64;
        for j in 0..3 {
            let mut s = vertices[face[0] as usize * 3 + j];
            for &v in &face[1..] {
                s += vertices[v as usize * 3 + j];
            }
            out_v.push(s / n as f64);
        }

        let mut face_uv: Option<&[i64]> = None;
        let mut uv_center = -1;
        if let (Some(fuv_out), Some(fuv), Some(src), Some(dst)) =
            (&out_fuv, face_uvs, uvs, out_uvs.as_mut())
        {
            let _ = fuv_out;
            let cand: &[i64] = if f < fuv.count() { fuv.face(f) } else { &[] };
            if cand.len() == n && cand.iter().all(|&u| 0 <= u && u < nuv) {
                face_uv = Some(cand);
                uv_mids.clear();
                for c in 0..n {
                    let (u, w) = (cand[c] as usize, cand[(c + 1) % n] as usize);
                    uv_mids.push((dst.len() / 2) as i64);
                    for j in 0..2 {
                        dst.push((src[u * 2 + j] + src[w * 2 + j]) * 0.5);
                    }
                }
                uv_center = (dst.len() / 2) as i64;
                for j in 0..2 {
                    let mut s = src[cand[0] as usize * 2 + j];
                    for &u in &cand[1..] {
                        s += src[u as usize * 2 + j];
                    }
                    dst.push(s / n as f64);
                }
            }
        }
        for c in 0..n {
            let prev = (c + n - 1) % n;
            quads.extend_from_slice(&[face[c], local[c], center, local[prev]]);
            if out_fuv.is_some() {
                match face_uv {
                    None => out_fuv = None,
                    Some(fu) => out_fuv.as_mut().unwrap().extend_from_slice(&[
                        fu[c],
                        uv_mids[c],
                        uv_center,
                        uv_mids[prev],
                    ]),
                }
            }
        }
    }
    Ok(Closure {
        vertices: out_v,
        quads,
        uvs: out_uvs,
        face_uvs: out_fuv,
    })
}

type Edge = (i64, i64);

fn key(a: i64, b: i64) -> Edge {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn toggle(set: &mut HashSet<Edge>, edge: Edge) {
    if !set.remove(&edge) {
        set.insert(edge);
    }
}

/// Mirrors `_parity_path_edges`: returns the selected (split) edges, sorted.
pub fn parity_path_edges(faces: &Faces, frozen: &[Edge]) -> Result<Vec<Edge>, String> {
    let fcount = faces.count();
    let h = halves(faces);
    let e = edges(&h);
    let frozen: HashSet<Edge> = frozen.iter().copied().collect();
    let mut selected: HashSet<Edge> = frozen.clone();
    for f in 0..fcount {
        let face = faces.face(f);
        if face.len() == 3 {
            for c in 0..3 {
                selected.insert(key(face[c], face[(c + 1) % 3]));
            }
        }
    }
    let is_quad = |f: u32| faces.face(f as usize).len() == 4;
    let mut adjacency: Vec<Vec<(u32, Edge)>> = vec![Vec::new(); fcount];
    let mut boundary: Vec<Vec<Edge>> = vec![Vec::new(); fcount];
    for i in 0..e.len() {
        let edge = (e.a[i], e.b[i]);
        let uses = e.uses(i);
        match uses.len() {
            1 => {
                let f = h.face[uses[0] as usize];
                if is_quad(f) && !frozen.contains(&edge) {
                    boundary[f as usize].push(edge);
                }
            }
            2 => {
                let (first, second) = (h.face[uses[0] as usize], h.face[uses[1] as usize]);
                if is_quad(first) && is_quad(second) {
                    adjacency[first as usize].push((second, edge));
                    adjacency[second as usize].push((first, edge));
                }
            }
            n => {
                return Err(format!(
                    "Edge ({}, {}) has {} incident faces.",
                    edge.0, edge.1, n
                ))
            }
        }
    }
    for linked in adjacency.iter_mut() {
        linked.sort_unstable();
    }
    for b in boundary.iter_mut() {
        b.sort_unstable();
    }
    let mut terminals: BTreeSet<u32> = (0..fcount as u32)
        .filter(|&f| {
            let face = faces.face(f as usize);
            face.len() == 4
                && (0..4)
                    .filter(|&c| selected.contains(&key(face[c], face[(c + 1) % 4])))
                    .count()
                    % 2
                    == 1
        })
        .collect();

    let mut dist = vec![0i64; fcount];
    let mut parent: Vec<(u32, Edge)> = vec![(0, (0, 0)); fcount];
    let mut stamp = vec![0u32; fcount];
    let mut generation = 0u32;
    let mut visited: Vec<u32> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    while let Some(&start) = terminals.iter().next() {
        generation += 1;
        visited.clear();
        queue.clear();
        stamp[start as usize] = generation;
        dist[start as usize] = 0;
        visited.push(start);
        queue.push_back(start);
        let mut best = if boundary[start as usize].is_empty() {
            i64::MAX
        } else {
            1
        };
        while let Some(current) = queue.pop_front() {
            if dist[current as usize] >= best {
                break;
            }
            for &(linked, edge) in &adjacency[current as usize] {
                if stamp[linked as usize] == generation {
                    continue;
                }
                let d = dist[current as usize] + 1;
                stamp[linked as usize] = generation;
                dist[linked as usize] = d;
                parent[linked as usize] = (current, edge);
                queue.push_back(linked);
                visited.push(linked);
                if terminals.contains(&linked) {
                    best = best.min(d);
                } else if !boundary[linked as usize].is_empty() {
                    best = best.min(d + 1);
                }
            }
        }
        // (cost, kind, target, exterior edge): kind 0 pairs with a terminal.
        let mut choice: Option<(i64, u8, u32, Edge)> = None;
        for &target in &visited {
            let d = dist[target as usize];
            if target != start && terminals.contains(&target) {
                let c = (d, 0u8, target, (-1, -1));
                if choice.map_or(true, |b| c < b) {
                    choice = Some(c);
                }
            }
            for &edge in &boundary[target as usize] {
                let c = (d + 1, 1u8, target, edge);
                if choice.map_or(true, |b| c < b) {
                    choice = Some(c);
                }
            }
        }
        let Some((_cost, kind, target, exterior)) = choice else {
            return Err("Transition parity cannot be discharged through quad-only adjacency or the mesh boundary."
                .to_string());
        };
        let mut current = target;
        while current != start {
            let (p, edge) = parent[current as usize];
            toggle(&mut selected, edge);
            current = p;
        }
        terminals.remove(&start);
        if kind == 1 {
            toggle(&mut selected, exterior);
        } else {
            terminals.remove(&target);
        }
    }
    for f in 0..fcount {
        let face = faces.face(f);
        let n = face.len();
        let split = (0..n)
            .filter(|&c| selected.contains(&key(face[c], face[(c + 1) % n])))
            .count();
        if (n + split) % 2 == 1 {
            return Err(format!("Parity solve failed on face {f}."));
        }
    }
    let mut out: Vec<Edge> = selected.into_iter().collect();
    out.sort_unstable();
    Ok(out)
}
