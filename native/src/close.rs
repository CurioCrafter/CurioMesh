// SPDX-License-Identifier: GPL-3.0-or-later
//! Closing paired meshes into quads, piece by piece on all cores: `flatquad.make_closable`, `flatquad._close_pieces` and
//! `topology.parity_path_closure_unpairing`. Each mirrors its Python function line by line, so the faces come out the
//! same and in the same order. Each face carries `origin`, the input face it came from, so Python can look up materials.

use crate::geom::{parity_templates, quad_valid};
use crate::topo::{self, Dsu, Faces, Summary};
use rayon::prelude::*;
use std::collections::HashSet;

pub struct Mesh {
    pub vertices: Vec<f64>,
    pub indptr: Vec<i64>,
    pub flat: Vec<i64>,
    pub origin: Vec<i64>,
}

impl Mesh {
    fn faces(&self) -> Faces<'_> {
        Faces {
            indptr: &self.indptr,
            flat: &self.flat,
        }
    }
    fn count(&self) -> usize {
        self.indptr.len() - 1
    }
}

fn point(vertices: &[f64], v: i64) -> [f64; 3] {
    let at = v as usize * 3;
    [vertices[at], vertices[at + 1], vertices[at + 2]]
}

/// `_face_frames`' area: half the length of Newell's normal.
fn newell_area(vertices: &[f64], face: &[i64]) -> f64 {
    let mut total = [0.0f64; 3];
    for (k, &a) in face.iter().enumerate() {
        let (p, q) = (point(vertices, a), point(vertices, face[(k + 1) % face.len()]));
        total[0] += p[1] * q[2] - p[2] * q[1];
        total[1] += p[2] * q[0] - p[0] * q[2];
        total[2] += p[0] * q[1] - p[1] * q[0];
    }
    0.5 * (total[0] * total[0] + total[1] * total[1] + total[2] * total[2]).sqrt()
}

fn span_of(vertices: &[f64]) -> [f64; 3] {
    let mut span = [0.0; 3];
    for (c, out) in span.iter_mut().enumerate() {
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in vertices.chunks_exact(3) {
            low = low.min(p[c]);
            high = high.max(p[c]);
        }
        *out = high - low;
    }
    span
}

/// `make_closable`: `span` is the bounding box's diagonal. Returns (mesh, vertices added).
pub fn make_closable(mesh: &Mesh, span: f64) -> (Mesh, usize) {
    let faces = mesh.faces();
    let nf = faces.count();
    let areas = topo::collect(nf, |f| newell_area(&mesh.vertices, faces.face(f)));
    let limit = 1e-14 * span * span;
    let mut seen: HashSet<Vec<i64>, topo::FxBuild> = HashSet::with_capacity_and_hasher(nf, Default::default());
    let (mut kept, mut indptr, mut flat) = (Vec::with_capacity(nf), vec![0i64], Vec::with_capacity(mesh.flat.len()));
    let mut key: Vec<i64> = Vec::new();
    for f in 0..nf {
        let face = faces.face(f);
        key.clear();
        key.extend_from_slice(face);
        key.sort_unstable();
        if key.windows(2).any(|w| w[0] == w[1]) || areas[f] <= limit || seen.contains(&key) {
            continue;
        }
        seen.insert(key.clone());
        kept.push(f);
        flat.extend_from_slice(face);
        indptr.push(flat.len() as i64);
    }
    drop(seen);
    let origin: Vec<i64> = kept.iter().map(|&f| mesh.origin[f]).collect();
    let nv = mesh.vertices.len() / 3;
    let count = indptr.len() - 1;

    // Per vertex: its faces (ascending), and its outgoing corners as (next vertex, face) in face order.
    let mut uses = vec![0u32; nv + 1];
    for &v in &flat {
        uses[v as usize + 1] += 1;
    }
    for v in 0..nv {
        uses[v + 1] += uses[v];
    }
    let mut filled = uses.clone();
    let mut around = vec![(0i64, 0u32); flat.len()];
    let mut visit_order = Vec::new(); // vertices in the order `users_of` meets them
    for f in 0..count {
        let face = &flat[indptr[f] as usize..indptr[f + 1] as usize];
        for (k, &v) in face.iter().enumerate() {
            let at = &mut filled[v as usize];
            if *at == uses[v as usize] {
                visit_order.push(v);
            }
            around[*at as usize] = (face[(k + 1) % face.len()], f as u32);
            *at += 1;
        }
    }
    let around_of = |v: i64| &around[uses[v as usize] as usize..uses[v as usize + 1] as usize];

    // For each vertex with more than one fan: the fan (by rank in sorted root order) of each of its faces.
    let split_fans = |v: i64| -> Option<(Vec<u32>, Vec<u32>, usize)> {
        let row = around_of(v);
        if row.len() < 2 {
            return None;
        }
        let users: Vec<u32> = row.iter().map(|x| x.1).collect();
        let at = |face: u32| users.binary_search(&face).unwrap();
        let mut parent: Vec<usize> = (0..users.len()).collect();
        fn root(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for (i, &(following, _)) in row.iter().enumerate() {
            let out = row.iter().filter(|x| x.0 == following).count();
            let mut back_count = 0;
            let mut back = 0u32;
            for &(to, face) in around_of(following) {
                if to == v {
                    back_count += 1;
                    if back_count == 1 {
                        back = face;
                    }
                }
            }
            if out == 1 && back_count == 1 {
                let r = root(&mut parent, i);
                let s = root(&mut parent, at(back));
                parent[r] = s;
            }
        }
        let roots: Vec<usize> = (0..users.len()).map(|i| root(&mut parent, i)).collect();
        let mut fans: Vec<u32> = roots.iter().map(|&r| users[r]).collect(); // a root's identity is its face id
        fans.sort_unstable();
        fans.dedup();
        if fans.len() < 2 {
            return None;
        }
        let rank: Vec<u32> = roots
            .iter()
            .map(|&r| fans.binary_search(&users[r]).unwrap() as u32)
            .collect();
        Some((users, rank, fans.len()))
    };
    let splits: Vec<Option<(Vec<u32>, Vec<u32>, usize)>> =
        topo::collect(visit_order.len(), |i| split_fans(visit_order[i]));

    let mut out_vertices = mesh.vertices.clone();
    let mut added = 0usize;
    let mut out_flat = flat.clone();
    for (i, split) in splits.iter().enumerate() {
        let Some((users, rank, fan_count)) = split else {
            continue;
        };
        let v = visit_order[i];
        let first_new = (nv + added) as i64;
        let p = point(&mesh.vertices, v);
        for _ in 1..*fan_count {
            out_vertices.extend_from_slice(&p);
        }
        added += fan_count - 1;
        for (&face, &fan) in users.iter().zip(rank) {
            if fan > 0 {
                let (a, b) = (indptr[face as usize] as usize, indptr[face as usize + 1] as usize);
                for c in &mut out_flat[a..b] {
                    if *c == v {
                        *c = first_new + fan as i64 - 1;
                    }
                }
            }
        }
    }
    (
        Mesh {
            vertices: out_vertices,
            indptr,
            flat: out_flat,
            origin,
        },
        added,
    )
}

/// `_span_epsilon`.
fn span_epsilon(vertices: &[f64]) -> f64 {
    if vertices.is_empty() {
        return 1e-20;
    }
    let s = span_of(vertices);
    (((s[0] * s[0] + s[1] * s[1]) + s[2] * s[2]) * 1e-14).max(1e-20)
}

struct Full {
    counts: Summary,
    degenerate: i64,
    invalid: i64,
}

impl Full {
    fn all_quads(&self) -> bool {
        let c = &self.counts;
        c.faces > 0
            && c.quads == c.faces
            && self.degenerate == 0
            && c.non_manifold_edges == 0
            && c.non_manifold_vertices == 0
            && c.branched_boundary_vertices == 0
            && c.inconsistent_edges == 0
            && c.duplicate_faces == 0
            && self.invalid == 0
    }
    fn certifiable(&self) -> bool {
        let c = &self.counts;
        self.degenerate == 0
            && self.invalid == 0
            && c.non_manifold_edges == 0
            && c.non_manifold_vertices == 0
            && c.branched_boundary_vertices == 0
            && c.inconsistent_edges == 0
            && c.duplicate_faces == 0
    }
}

/// `topology_summary`, as far as closure looks at it.
fn full_summary(faces: &Faces, vertices: &[f64], warp_cosine: f64) -> Full {
    let counts = topo::summary(faces);
    let epsilon = span_epsilon(vertices);
    let degenerate = topo::degenerate_count(faces, vertices, epsilon);
    let invalid = topo::collect(faces.count(), |f| {
        let face = faces.face(f);
        if face.len() != 4 {
            return false;
        }
        let distinct = face[0] != face[1] && face[0] != face[2] && face[0] != face[3] && face[1] != face[2] && face[1] != face[3] && face[2] != face[3];
        let p = [point(vertices, face[0]), point(vertices, face[1]), point(vertices, face[2]), point(vertices, face[3])];
        !(distinct && quad_valid(&p, epsilon, warp_cosine))
    })
    .into_iter()
    .filter(|&bad| bad)
    .count() as i64;
    Full {
        counts,
        degenerate,
        invalid,
    }
}

/// The midpoints of these sorted edges appended to the vertices: (coordinates, flat edges, midpoint ids).
fn with_midpoints(vertices: &[f64], selected: &[(i64, i64)]) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
    let mut base = Vec::with_capacity(vertices.len() + selected.len() * 3);
    base.extend_from_slice(vertices);
    let (mut edges, mut mids) = (Vec::with_capacity(selected.len() * 2), Vec::with_capacity(selected.len()));
    for &(a, b) in selected {
        let (p, q) = (point(vertices, a), point(vertices, b));
        mids.push((base.len() / 3) as i64);
        base.extend_from_slice(&[(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5, (p[2] + q[2]) * 0.5]);
        edges.extend_from_slice(&[a, b]);
    }
    (base, edges, mids)
}

/// `_unclosable_quads`: quads where every template comes out warped or folded, and the split edges they were judged with.
#[allow(clippy::type_complexity)]
fn unclosable_quads(mesh: &Mesh, warp_cosine: f64) -> Result<(Vec<usize>, Vec<(i64, i64)>), String> {
    let selected = topo::parity_path_edges(&mesh.faces(), &[])?;
    let (base, edges, mids) = with_midpoints(&mesh.vertices, &selected);
    let (counts, valid) = parity_templates(&mesh.indptr, &mesh.flat, &edges, &mids, &mesh.vertices, &base, warp_cosine);
    let mut out = Vec::new();
    let mut at = 0usize;
    for f in 0..mesh.count() {
        let count = counts[f] as usize;
        if !valid[at..at + count].iter().any(|&v| v) && mesh.faces().face(f).len() == 4 {
            out.push(f);
        }
        at += count;
    }
    Ok((out, selected))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn minus(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `_unpair_quads`: these quads as two triangles each, cut along the diagonal that bends the surface least.
fn unpair_quads(mesh: &Mesh, quads: &[usize]) -> Mesh {
    let mut chosen = vec![false; mesh.count()];
    for &q in quads {
        chosen[q] = true;
    }
    let mut out = Mesh {
        vertices: mesh.vertices.clone(),
        indptr: vec![0],
        flat: Vec::with_capacity(mesh.flat.len() + quads.len() * 2),
        origin: Vec::with_capacity(mesh.count() + quads.len()),
    };
    let faces = mesh.faces();
    for f in 0..mesh.count() {
        let face = faces.face(f);
        if !chosen[f] {
            out.flat.extend_from_slice(face);
            out.indptr.push(out.flat.len() as i64);
            out.origin.push(mesh.origin[f]);
            continue;
        }
        let (a, b, c, d) = (face[0], face[1], face[2], face[3]);
        let p = [point(&mesh.vertices, a), point(&mesh.vertices, b), point(&mesh.vertices, c), point(&mesh.vertices, d)];
        let bend_ac = dot(cross(minus(p[1], p[0]), minus(p[2], p[0])), cross(minus(p[2], p[0]), minus(p[3], p[0])));
        let bend_bd = dot(cross(minus(p[2], p[1]), minus(p[3], p[1])), cross(minus(p[3], p[1]), minus(p[0], p[1])));
        let cut: [i64; 6] = if bend_ac >= bend_bd { [a, b, c, a, c, d] } else { [b, c, d, b, d, a] };
        for tri in cut.chunks_exact(3) {
            out.flat.extend_from_slice(tri);
            out.indptr.push(out.flat.len() as i64);
            out.origin.push(mesh.origin[f]);
        }
    }
    out
}

/// `parity_path_quad_closure` without frozen edges, UVs or a second geometry. `known` is this mesh's parity solution, if
/// it has been worked out already: it is the same every time, and takes as long as all the rest.
fn parity_closure(mesh: &Mesh, known: Option<Vec<(i64, i64)>>, warp_cosine: f64) -> Result<Mesh, String> {
    let faces = mesh.faces();
    let nf = mesh.count();
    if nf == 0 || mesh.indptr.windows(2).any(|w| w[1] - w[0] != 3 && w[1] - w[0] != 4) {
        return Err("not triangles and quads".to_string());
    }
    let source = full_summary(&faces, &mesh.vertices, warp_cosine);
    if !source.certifiable() {
        return Err("source is not certifiable".to_string());
    }
    let selected = match known {
        Some(edges) => edges,
        None => topo::parity_path_edges(&faces, &[])?,
    };
    let (mut base, edges, mids) = with_midpoints(&mesh.vertices, &selected);
    let (counts, valid) = parity_templates(&mesh.indptr, &mesh.flat, &edges, &mids, &mesh.vertices, &base, warp_cosine);
    let mut out_flat: Vec<i64> = Vec::with_capacity(mesh.flat.len() * 2);
    let mut origin: Vec<i64> = Vec::with_capacity(nf * 2);
    let mut at = 0usize;
    let mut augmented: Vec<i64> = Vec::with_capacity(8);
    for f in 0..nf {
        let face = faces.face(f);
        let mut next_valid = {
            let row = &valid[at..at + counts[f] as usize];
            at += counts[f] as usize;
            row.iter().copied()
        };
        let n = face.len();
        augmented.clear();
        for (corner, &vertex) in face.iter().enumerate() {
            augmented.push(vertex);
            let following = face[(corner + 1) % n];
            if let Ok(k) = selected.binary_search(&(vertex.min(following), vertex.max(following))) {
                augmented.push(mids[k]);
            }
        }
        let split = augmented.len() - n;
        let mut done = false;
        let before = out_flat.len();
        if n == 4 && split == 0 && next_valid.next() == Some(true) {
            out_flat.extend_from_slice(face);
            done = true;
        }
        if !done && n == 4 && split == 2 {
            for offset in 0..3 {
                if next_valid.next() != Some(true) {
                    continue;
                }
                for start in [offset, offset + 3] {
                    out_flat.extend((0..4).map(|step| augmented[(start + step) % 6]));
                }
                done = true;
                break;
            }
        }
        if !done {
            let count = augmented.len();
            if count % 2 == 1 || count < 4 {
                return Err(format!("Face {f} has no even transition boundary"));
            }
            let center = (base.len() / 3) as i64;
            let mut sum = point(&mesh.vertices, face[0]);
            for &v in &face[1..] {
                let p = point(&mesh.vertices, v);
                sum = [sum[0] + p[0], sum[1] + p[1], sum[2] + p[2]];
            }
            let size = n as f64;
            base.extend_from_slice(&[sum[0] / size, sum[1] / size, sum[2] / size]);
            for offset in 0..2 {
                if next_valid.next() != Some(true) {
                    continue;
                }
                for cell in 0..count / 2 {
                    out_flat.push(center);
                    for step in 0..3 {
                        out_flat.push(augmented[(offset + 2 * cell + step) % count]);
                    }
                }
                done = true;
                break;
            }
        }
        if !done {
            return Err(format!("Face {f} has no locally valid ParityPath quad template."));
        }
        for _ in 0..(out_flat.len() - before) / 4 {
            origin.push(mesh.origin[f]);
        }
    }
    let quads = out_flat.len() / 4;
    let result = Mesh {
        vertices: base,
        indptr: (0..=quads as i64).map(|q| q * 4).collect(),
        flat: out_flat,
        origin,
    };
    let summary = full_summary(&result.faces(), &result.vertices, warp_cosine);
    if !summary.all_quads() {
        return Err("ParityPath closure failed validation".to_string());
    }
    let signature = |detail: &[(i64, i64, i64)]| {
        let mut rows: Vec<(i64, i64)> = detail.iter().map(|row| (row.0, row.1)).collect();
        rows.sort_unstable();
        rows
    };
    if signature(&source.counts.component_detail) != signature(&summary.counts.component_detail) {
        return Err("ParityPath closure changed the component signature".to_string());
    }
    Ok(result)
}

/// `parity_path_closure_unpairing`.
fn closure_unpairing(mesh: &Mesh, warp_cosine: f64) -> Result<Mesh, String> {
    let mut changed: Option<Mesh> = None;
    let mut known = None;
    for _ in 0..8 {
        let (quads, selected) = unclosable_quads(changed.as_ref().unwrap_or(mesh), warp_cosine)?;
        if quads.is_empty() {
            known = Some(selected);
            break;
        }
        changed = Some(unpair_quads(changed.as_ref().unwrap_or(mesh), &quads));
    }
    parity_closure(changed.as_ref().unwrap_or(mesh), known, warp_cosine)
}

/// `quad_only_closure`; `checked` validates the result.
fn corner_closure(mesh: &Mesh, checked: bool, warp_cosine: f64) -> Result<Mesh, String> {
    let closed = topo::quad_closure(&mesh.faces(), &mesh.vertices, &[], None, None)?;
    let mut origin = Vec::with_capacity(closed.quads.len() / 4);
    for f in 0..mesh.count() {
        for _ in 0..mesh.faces().face(f).len() {
            origin.push(mesh.origin[f]);
        }
    }
    let quads = closed.quads.len() / 4;
    let result = Mesh {
        vertices: closed.vertices,
        indptr: (0..=quads as i64).map(|q| q * 4).collect(),
        flat: closed.quads,
        origin,
    };
    if checked && !full_summary(&result.faces(), &result.vertices, warp_cosine).all_quads() {
        return Err("Quad closure failed validation".to_string());
    }
    Ok(result)
}

fn diagonal(vertices: &[f64]) -> f64 {
    if vertices.is_empty() {
        return 1.0;
    }
    let s = span_of(vertices);
    (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
}

/// `_close_piece`: (closed piece, how: 0 ParityPath, 1 corner closure, 2 all quads left as they were, 3 forced).
fn close_piece(piece: Mesh, warp_cosine: f64) -> Result<(Mesh, usize), String> {
    let (opened, added) = make_closable(&piece, diagonal(&piece.vertices));
    let attempts: &[&Mesh] = if added > 0 { &[&piece, &opened] } else { &[&piece] };
    for attempt in attempts {
        if let Ok(done) = closure_unpairing(attempt, warp_cosine) {
            return Ok((done, 0));
        }
        if let Ok(done) = corner_closure(attempt, true, warp_cosine) {
            return Ok((done, 1));
        }
    }
    if piece.indptr.windows(2).all(|w| w[1] - w[0] == 4) {
        return Ok((piece, 2));
    }
    Ok((corner_closure(&opened, false, warp_cosine)?, 3))
}

/// `_close_pieces`: every edge-connected piece closed on its own, in order of their first faces. Pieces run side by
/// side; each result is complete before they are joined, so the output doesn't depend on the number of threads.
pub fn close_pieces(faces: &Faces, vertices: &[f64], warp_cosine: f64) -> Result<(Mesh, [i64; 4]), String> {
    let nf = faces.count();
    let h = topo::halves(faces);
    let e = topo::edges(&h);
    let mut dsu = Dsu::new(nf);
    for i in 0..e.len() {
        let uses = e.uses(i);
        let first = h.face[uses[0] as usize];
        for &c in &uses[1..] {
            dsu.union(first, h.face[c as usize]);
        }
    }
    drop((h, e));
    let mut group_of = vec![u32::MAX; nf];
    let mut members: Vec<Vec<u32>> = Vec::new();
    for f in 0..nf as u32 {
        let r = dsu.find(f) as usize;
        if group_of[r] == u32::MAX {
            group_of[r] = members.len() as u32;
            members.push(Vec::new());
        }
        members[group_of[r] as usize].push(f);
    }
    let close_group = |ids: &Vec<u32>| -> Result<(Mesh, usize), String> {
        let mut used: Vec<i64> = Vec::new();
        for &f in ids {
            used.extend_from_slice(faces.face(f as usize));
        }
        topo::sort(&mut used);
        used.dedup();
        let mut piece = Mesh {
            vertices: Vec::with_capacity(used.len() * 3),
            indptr: Vec::with_capacity(ids.len() + 1),
            flat: Vec::new(),
            origin: ids.iter().map(|&f| f as i64).collect(),
        };
        for &v in &used {
            piece.vertices.extend_from_slice(&point(vertices, v));
        }
        piece.indptr.push(0);
        for &f in ids {
            for v in faces.face(f as usize) {
                piece.flat.push(used.binary_search(v).unwrap() as i64);
            }
            piece.indptr.push(piece.flat.len() as i64);
        }
        if piece.indptr.windows(2).all(|w| w[1] - w[0] == 4) && full_summary(&piece.faces(), &piece.vertices, warp_cosine).all_quads() {
            return Ok((piece, 0));
        }
        close_piece(piece, warp_cosine)
    };
    let closed: Vec<Result<(Mesh, usize), String>> = if nf >= topo::PARALLEL_MIN {
        members.par_iter().map(close_group).collect()
    } else {
        members.iter().map(close_group).collect()
    };
    let mut merged = Mesh {
        vertices: Vec::new(),
        indptr: vec![0],
        flat: Vec::new(),
        origin: Vec::new(),
    };
    let mut counts = [0i64; 4];
    for row in closed {
        let (mesh, kind) = row?;
        counts[kind] += 1;
        let offset = (merged.vertices.len() / 3) as i64;
        merged.vertices.extend_from_slice(&mesh.vertices);
        merged.flat.extend(mesh.flat.iter().map(|&v| v + offset));
        let end = *merged.indptr.last().unwrap();
        merged.indptr.extend(mesh.indptr[1..].iter().map(|&at| at + end));
        merged.origin.extend_from_slice(&mesh.origin);
    }
    Ok((merged, counts))
}
