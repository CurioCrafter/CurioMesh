// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Fixing one spot of a result without redoing the model. The faces round a spot (the region) are remade and the rest
//! is left exactly as it is: the region's outline keeps its corners and sides, so nothing outside moves and no crack
//! or T-junction opens.
//! - Redo this spot: every region quad is cut into smaller quads, the new corners are put on the source, then evened
//!   out on it (`layout::relax`, `layout::spin_edges`).
//! - Keep original here: the source's own triangles under the region, stitched to the outline by a strip of
//!   triangles, paired into quads and closed to all quads.
//! Closing to all quads with the outline kept: a face of n corners with s cut sides becomes (n + s) / 2 quads round a
//! middle point, so n + s must be even. Which inner sides are cut is fixed by flipping cuts along paths between faces
//! whose count is odd; each path flips the count of only its two ends.

use crate::io::Mesh;
use crate::{issues, layout, near, pairing, retopo, topo};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::AtomicBool;

type V3 = [f64; 3];
const RINGS: usize = 1; // rings of neighbouring faces taken in with the spot's own faces
const SMOOTH_PASSES: usize = 10;
const MOST_TRIMS: usize = 50;
const SNAP_TURN: f64 = 60.0; // degrees: a corner isn't put on source faces turned further than this from its quads

/// A fixed result and what changed.
pub struct Fixed {
    pub mesh: Mesh,
    pub faces_before: usize, // faces in the region before
    pub faces_after: usize,  // and after
    pub bad_before: usize,   // bad faces in the whole result
    pub bad_after: usize,
    pub new_faces: Vec<usize>, // the region's faces in `mesh`
}

fn p(points: &[f64], v: i64) -> V3 {
    [points[v as usize * 3], points[v as usize * 3 + 1], points[v as usize * 3 + 2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn dist(a: V3, b: V3) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}
/// A face's normal times twice its area (Newell).
fn newell(points: &[f64], face: &[i64]) -> V3 {
    face.iter().enumerate().fold([0.0; 3], |s, (k, &v)| {
        let c = cross(p(points, v), p(points, face[(k + 1) % face.len()]));
        [s[0] + c[0], s[1] + c[1], s[2] + c[2]]
    })
}
fn key(a: i64, b: i64) -> (i64, i64) {
    (a.min(b), a.max(b))
}

/// How square a quad is: 1 for a square, 0 or less when folded.
fn squareness(q: [V3; 4]) -> f64 {
    let n = cross(sub(q[2], q[0]), sub(q[3], q[1]));
    (0..4).map(|k| {
        let (a, b) = (sub(q[(k + 1) % 4], q[k]), sub(q[(k + 3) % 4], q[k]));
        let d = (dot(a, a) * dot(b, b) * dot(n, n)).sqrt();
        if d > 1e-300 { dot(cross(a, b), n) / d } else { -1.0 }
    }).fold(1.0, f64::min)
}

/// The spot's faces and `rings` rings round them, as a mask. Where the outline would touch itself at a corner, every
/// face round that corner is taken in too, so the outline is plain loops.
fn region(mesh: &Mesh, spot: &[usize], rings: usize) -> Vec<bool> {
    let nf = mesh.faces();
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); mesh.points.len() / 3];
    for f in 0..nf {
        for &v in mesh.face(f) {
            users[v as usize].push(f);
        }
    }
    let mut inside = vec![false; nf];
    for &f in spot {
        inside[f] = true;
    }
    let grow = |inside: &mut Vec<bool>, corners: &[i64]| {
        for &v in corners {
            for &g in &users[v as usize] {
                inside[g] = true;
            }
        }
    };
    for _ in 0..rings {
        let corners: Vec<i64> = (0..nf).filter(|&f| inside[f]).flat_map(|f| mesh.face(f).to_vec()).collect();
        grow(&mut inside, &corners);
    }
    for _ in 0..8 {
        let (outline, _) = outline(mesh, &inside);
        let mut ways: HashMap<i64, usize> = HashMap::new();
        for &(a, _) in &outline {
            *ways.entry(a).or_default() += 1;
        }
        let pinched: Vec<i64> = outline.iter().map(|e| e.0).filter(|a| ways[a] > 1).collect();
        if pinched.is_empty() {
            break;
        }
        grow(&mut inside, &pinched);
    }
    inside
}

/// The region's outline sides as they run in its faces, and the same as undirected keys. A side is inside the region
/// only when exactly two faces use it and both are in the region.
fn outline(mesh: &Mesh, inside: &[bool]) -> (Vec<(i64, i64)>, HashSet<(i64, i64)>) {
    let mut uses: HashMap<(i64, i64), (usize, usize)> = HashMap::new(); // (all faces, region faces)
    for f in 0..mesh.faces() {
        let face = mesh.face(f);
        for k in 0..face.len() {
            let u = uses.entry(key(face[k], face[(k + 1) % face.len()])).or_default();
            u.0 += 1;
            u.1 += inside[f] as usize;
        }
    }
    let mut sides = Vec::new();
    for f in (0..mesh.faces()).filter(|&f| inside[f]) {
        let face = mesh.face(f);
        for k in 0..face.len() {
            let (a, b) = (face[k], face[(k + 1) % face.len()]);
            if uses[&key(a, b)] != (2, 2) {
                sides.push((a, b));
            }
        }
    }
    let keys = sides.iter().map(|&(a, b)| key(a, b)).collect();
    (sides, keys)
}

/// Directed sides joined into closed loops of corners. Errs where a corner has two ways on or a loop is open.
fn loops(sides: &[(i64, i64)]) -> Result<Vec<Vec<i64>>, String> {
    let mut next = HashMap::new();
    for &(a, b) in sides {
        if next.insert(a, b).is_some() {
            return Err("its edge touches itself (it runs along a broken edge)".into());
        }
    }
    let (mut out, mut seen) = (Vec::new(), HashSet::new());
    for &(start, _) in sides {
        if seen.contains(&start) {
            continue;
        }
        let (mut ring, mut v) = (Vec::new(), start);
        loop {
            seen.insert(v);
            ring.push(v);
            v = *next.get(&v).ok_or("its outline is open")?;
            if v == start {
                break;
            }
            if seen.contains(&v) {
                return Err("its edge touches itself (it runs along a broken edge)".into());
            }
        }
        out.push(ring);
    }
    Ok(out)
}

/// Cut `faces` into quads (see the top of the file); `cut_all` starts with every inner side cut (Redo), otherwise none
/// (Keep original: only what closing needs). Sides used once, or in `fixed`, are never cut. New points go on the end
/// of `points`. Returns the quads and the face each came from.
fn split_to_quads(points: &mut Vec<f64>, faces: &[Vec<i64>], fixed: &HashSet<(i64, i64)>, cut_all: bool) -> Result<(Vec<[i64; 4]>, Vec<usize>), String> {
    let mut uses: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (f, face) in faces.iter().enumerate() {
        for k in 0..face.len() {
            uses.entry(key(face[k], face[(k + 1) % face.len()])).or_default().push(f);
        }
    }
    let mut inner: Vec<(i64, i64)> = uses.iter().filter(|(e, u)| u.len() == 2 && u[0] != u[1] && !fixed.contains(e)).map(|(e, _)| *e).collect();
    inner.sort_unstable(); // HashMap order is random: keep the result the same run to run
        let mut next_to: Vec<Vec<(usize, usize)>> = vec![Vec::new(); faces.len()]; // (side, face across it)
    for (i, e) in inner.iter().enumerate() {
        let u = &uses[e];
        next_to[u[0]].push((i, u[1]));
        next_to[u[1]].push((i, u[0]));
    }
    let mut cut = vec![cut_all; inner.len()];
    let mut odd: Vec<bool> = (0..faces.len()).map(|f| (faces[f].len() + next_to[f].iter().filter(|s| cut[s.0]).count()) % 2 == 1).collect();
    // ponytail: one breadth-first search per odd face, O(region^2) at worst; fine for spots of a few thousand faces.
    for f in 0..faces.len() {
        if !odd[f] {
            continue;
        }
        let mut came: Vec<Option<(usize, usize)>> = vec![None; faces.len()];
        let mut queue = VecDeque::from([f]);
        let mut found = None;
        came[f] = Some((usize::MAX, f));
        while let Some(g) = queue.pop_front() {
            if g != f && odd[g] {
                found = Some(g);
                break;
            }
            for &(s, h) in &next_to[g] {
                if came[h].is_none() {
                    came[h] = Some((s, g));
                    queue.push_back(h);
                }
            }
        }
        let mut g = found.ok_or("it can't be closed to quads with its outline kept")?;
        (odd[f], odd[g]) = (false, false);
        while g != f {
            let (s, from) = came[g].unwrap();
            cut[s] = !cut[s];
            g = from;
        }
    }
    let mut middle = HashMap::new();
    let first_new = (points.len() / 3) as i64;
    for (i, &(a, b)) in inner.iter().enumerate() {
        if cut[i] {
            let (x, y) = (p(points, a), p(points, b));
            points.extend((0..3).map(|c| (x[c] + y[c]) / 2.0));
            middle.insert((a, b), (points.len() / 3 - 1) as i64);
        }
    }
    let (mut quads, mut from) = (Vec::new(), Vec::new());
    for (f, face) in faces.iter().enumerate() {
        let mut ring = Vec::with_capacity(face.len() * 2);
        for k in 0..face.len() {
            ring.push(face[k]);
            if let Some(&m) = middle.get(&key(face[k], face[(k + 1) % face.len()])) {
                ring.push(m);
            }
        }
        if face.len() == 4 && ring.len() == 4 {
            quads.push([face[0], face[1], face[2], face[3]]);
            from.push(f);
            continue;
        }
        let cut_at = |k: usize| ring[k] >= first_new;
        if face.len() == 4 && ring.len() == 6 && ((cut_at(1) && cut_at(4)) || (cut_at(2) && cut_at(5))) {
            // Two opposite sides cut: two quads side by side (a fan round a middle would bend at a cut).
            let (x, y) = if cut_at(1) { (1, 4) } else { (2, 5) };
            quads.push([ring[x], ring[(x + 1) % 6], ring[(x + 2) % 6], ring[y]]);
            quads.push([ring[y], ring[(y + 1) % 6], ring[(y + 2) % 6], ring[x]]);
            from.extend([f, f]);
            continue;
        }
        let n = face.len() as f64;
        let centre: V3 = face.iter().fold([0.0; 3], |s, &v| {
            let q = p(points, v);
            [s[0] + q[0] / n, s[1] + q[1] / n, s[2] + q[2] / n]
        });
        if face.len() == 3 && ring.len() == 4 {
            // A triangle with one side cut: its middle lies on the line from the cut to the far corner, so a fan would
            // fold. A quad inside it instead, joined to the four outer points (the cut's inner point pushed out further
            // so the inner quad stays convex).
            let r = |k: usize| ring[(ring.iter().position(|&v| v >= first_new).unwrap() + 2 + k) % 4]; // r(2) is the cut
            let inner: Vec<i64> = (0..4).map(|k| {
                let (x, t) = (p(points, r(k)), if k == 2 { 0.75 } else { 0.5 });
                points.extend((0..3).map(|c| centre[c] + t * (x[c] - centre[c])));
                (points.len() / 3 - 1) as i64
            }).collect();
            for k in 0..4 {
                quads.push([r(k), r((k + 1) % 4), inner[(k + 1) % 4], inner[k]]);
            }
            quads.push([inner[0], inner[1], inner[2], inner[3]]);
            from.extend([f; 5]);
            continue;
        }
        points.extend(centre);
        let c = (points.len() / 3 - 1) as i64;
        let l = ring.len();
        let fan = |o: usize| -> Vec<[i64; 4]> { (0..l / 2).map(|i| [c, ring[(2 * i + o) % l], ring[(2 * i + 1 + o) % l], ring[(2 * i + 2 + o) % l]]).collect() };
        let worst = |qs: &[[i64; 4]]| qs.iter().map(|q| squareness(q.map(|v| p(points, v)))).fold(1.0, f64::min);
        let (a, b) = (fan(0), fan(1));
        let best = if worst(&b) > worst(&a) { b } else { a };
        from.extend(std::iter::repeat_n(f, best.len()));
        quads.extend(best);
    }
    Ok((quads, from))
}

/// Whether a quad passes the checks' bent test (`issues::check`), with the same limits for the same mesh.
struct Valid {
    epsilon: f64,
    warp: f64,
}

impl Valid {
    fn of(points: &[f64]) -> Valid {
        let (low, high) = points.chunks_exact(3).fold(([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]), |(l, h), c| ([0, 1, 2].map(|k| l[k].min(c[k])), [0, 1, 2].map(|k| h[k].max(c[k]))));
        let span = sub(high, low);
        Valid { epsilon: (dot(span, span) * 1e-14).max(1e-20), warp: retopo::MAX_QUAD_WARP_DEGREES.to_radians().cos() }
    }
    fn quad(&self, points: &[f64], q: &[i64]) -> bool {
        q.len() == 4 && crate::geom::quad_valid(&[0, 1, 2, 3].map(|k| p(points, q[k])), self.epsilon, self.warp)
    }
}

/// Move the corners in `movers` to their neighbours' middle in `faces`, a few times over (Laplace smoothing): it
/// undoes folds.
fn smooth(points: &mut [f64], faces: &[Vec<i64>], movers: &HashSet<i64>) {
    let mut neighbours: HashMap<i64, Vec<i64>> = HashMap::new();
    for face in faces {
        for k in 0..face.len() {
            let (a, b) = (face[k], face[(k + 1) % face.len()]);
            neighbours.entry(a).or_default().push(b);
            neighbours.entry(b).or_default().push(a);
        }
    }
    let mut free: Vec<(i64, Vec<i64>)> = neighbours.into_iter().filter(|(v, _)| movers.contains(v)).collect();
    free.sort_unstable();
    for (_, n) in &mut free {
        n.sort_unstable();
        n.dedup();
    }
    for _ in 0..SMOOTH_PASSES {
        let moved: Vec<V3> = free.iter().map(|(_, n)| {
            let k = n.len() as f64;
            n.iter().fold([0.0; 3], |s, &w| {
                let q = p(points, w);
                [s[0] + q[0] / k, s[1] + q[1] / k, s[2] + q[2] / k]
            })
        }).collect();
        for ((v, _), to) in free.iter().zip(moved) {
            points[*v as usize * 3..*v as usize * 3 + 3].copy_from_slice(&to);
        }
    }
}

/// Put `ids` (points of `points`) on the source's surface, or half way there, where that leaves no more of the
/// `quads` round it bent than before; otherwise the point stays.
fn snap(points: &mut [f64], ids: &[i64], quads: &[[i64; 4]], source: &Mesh, tree: &near::FaceTree, valid: &Valid) {
    let mut users: HashMap<i64, Vec<usize>> = HashMap::new();
    for (q, quad) in quads.iter().enumerate() {
        for &v in quad {
            users.entry(v).or_default().push(q);
        }
    }
    let asks: Vec<f64> = ids.iter().flat_map(|&v| p(points, v)).collect();
    let (_, found) = near::nearest_faces(tree, &asks, &AtomicBool::new(false));
    for (&v, hit) in ids.iter().zip(found) {
        let around = users.get(&v).map_or(&[][..], |u| u.as_slice());
        // Only onto source faces turned about the same way and near enough: not onto the back of a thin part or the
        // side of something standing on it.
        let facing = around.iter().fold([0.0; 3], |s, &q| {
            let c = quads[q].map(|w| p(points, w));
            let n = cross(sub(c[2], c[0]), sub(c[3], c[1]));
            [s[0] + n[0], s[1] + n[1], s[2] + n[2]]
        });
        let reach = around.iter().flat_map(|&q| quads[q].iter().map(|&w| dist(p(points, w), p(points, v)))).sum::<f64>() / (4 * around.len()).max(1) as f64;
        let turned = |n: V3| dot(facing, n) / (dot(facing, facing) * dot(n, n)).sqrt().max(1e-300);
        if hit.1 < 0 || turned(newell(&source.points, source.face(hit.1 as usize))) < SNAP_TURN.to_radians().cos() || dist(hit.0, p(points, v)) > reach {
            continue;
        }
        let bent = |points: &[f64]| around.iter().filter(|&&q| !valid.quad(points, &quads[q])).count();
        let (was, at) = (bent(points), p(points, v));
        for share in [1.0, 0.5] {
            let to: V3 = [0, 1, 2].map(|k| at[k] + share * (hit.0[k] - at[k]));
            points[v as usize * 3..v as usize * 3 + 3].copy_from_slice(&to);
            if bent(points) <= was {
                break;
            }
            points[v as usize * 3..v as usize * 3 + 3].copy_from_slice(&at);
        }
    }
}

/// `mesh` without its region faces, with `quads` (corners into `points`) on the end.
fn splice(mesh: &Mesh, inside: &[bool], points: Vec<f64>, quads: &[[i64; 4]], material: &[i64]) -> (Mesh, Vec<usize>) {
    let mut out = Mesh { points, indptr: vec![0], materials: mesh.materials.clone(), ..Default::default() };
    for f in (0..mesh.faces()).filter(|&f| !inside[f]) {
        out.flat.extend_from_slice(mesh.face(f));
        out.indptr.push(out.flat.len() as i64);
        out.material.push(mesh.material[f]);
    }
    let first = out.faces();
    for (q, &m) in quads.iter().zip(material) {
        out.flat.extend_from_slice(q);
        out.indptr.push(out.flat.len() as i64);
        out.material.push(m);
    }
    (crate::io::compact(&out), (first..first + quads.len()).collect())
}

/// Refuse a fix that leaves anything but quads, breaks the surface, or adds bad faces.
fn judge(before: &issues::Check, mesh: Mesh, faces_before: usize, new_faces: Vec<usize>) -> Result<Fixed, String> {
    let after = issues::check(&mesh);
    if after.quads != after.faces {
        return Err(format!("it would leave {} faces that aren't quads", after.faces - after.quads));
    }
    if after.non_manifold_edges > before.non_manifold_edges || after.flipped_edges > before.flipped_edges || after.duplicate_faces > before.duplicate_faces || after.holes > before.holes {
        return Err("it would break the surface (a broken or flipped edge, or a hole)".into());
    }
    if after.bad > before.bad {
        return Err(format!("it would add {} bad face{}", after.bad - before.bad, if after.bad - before.bad == 1 { "" } else { "s" }));
    }
    Ok(Fixed { faces_before, faces_after: new_faces.len(), bad_before: before.bad, bad_after: after.bad, new_faces, mesh })
}

fn source_tree(source: &Mesh) -> near::FaceTree {
    near::FaceTree::new(&source.points, &source.indptr, &source.flat)
}

/// Redo this spot: the region's quads cut into smaller ones that lie on the source. `spot` is the spot's faces.
pub fn redo_spot(source: &Mesh, result: &Mesh, spot: &[usize]) -> Result<Fixed, String> {
    let before = issues::check(result);
    let inside = region(result, spot, RINGS);
    let (_, fixed) = outline(result, &inside);
    let faces: Vec<Vec<i64>> = (0..result.faces()).filter(|&f| inside[f]).map(|f| result.face(f).to_vec()).collect();
    let region_material: Vec<i64> = (0..result.faces()).filter(|&f| inside[f]).map(|f| result.material[f]).collect();
    let mut points = result.points.clone();
    // Corners any outside face uses stay exactly where they are. Corners of bent quads are evened out (folds undone),
    // then every quad is cut, and the free corners are put on the source and evened out again there.
    let mut locked = vec![false; points.len() / 3];
    for f in (0..result.faces()).filter(|&f| !inside[f]) {
        for &v in result.face(f) {
            locked[v as usize] = true;
        }
    }
    let valid = Valid::of(&result.points);
    let movers: HashSet<i64> = faces.iter().filter(|f| !valid.quad(&points, f)).flatten().copied().filter(|&v| !locked[v as usize]).collect();
    smooth(&mut points, &faces, &movers);
    let (quads, from) = split_to_quads(&mut points, &faces, &fixed, true)?;
    locked.resize(points.len() / 3, false);
    let mut free: Vec<i64> = quads.iter().flatten().copied().filter(|&v| !locked[v as usize]).collect();
    free.sort_unstable();
    free.dedup();
    let tree = source_tree(source);
    let bent = |points: &[f64], quads: &[i64]| quads.chunks_exact(4).filter(|q| !valid.quad(points, q)).count();
    snap(&mut points, &free, &quads, source, &tree, &valid);
    let material: Vec<i64> = from.iter().map(|&f| region_material[f]).collect();
    let mut part = Mesh { points, indptr: (0..=quads.len() as i64).map(|f| 4 * f).collect(), flat: quads.iter().flatten().copied().collect(), material, materials: result.materials.clone() };
    // Evening out is kept only where it bends no more quads; corners outside faces use never move.
    let steps: [&dyn Fn(&mut Mesh); 2] = [&|m| _ = layout::spin_edges(m), &|m| _ = layout::relax(m, &tree)];
    for step in steps {
        let mut tried = part.clone();
        step(&mut tried);
        for v in (0..result.points.len() / 3).filter(|&v| locked[v]) {
            tried.points[v * 3..v * 3 + 3].copy_from_slice(&part.points[v * 3..v * 3 + 3]);
        }
        if bent(&tried.points, &tried.flat) <= bent(&part.points, &part.flat) {
            part = tried;
        }
    }
    let quads: Vec<[i64; 4]> = part.flat.chunks_exact(4).map(|q| [q[0], q[1], q[2], q[3]]).collect();
    let (mesh, new_faces) = splice(result, &inside, part.points, &quads, &part.material);
    judge(&before, mesh, faces.len(), new_faces)
}

/// Keep original here: the source's own faces put back in the region, as quads where its triangles pair up and closed
/// to all quads, stitched to the region's outline. `spot` is the spot's faces.
pub fn keep_original(source: &Mesh, result: &Mesh, spot: &[usize]) -> Result<Fixed, String> {
    let before = issues::check(result);
    let inside = region(result, spot, RINGS + 1);
    let (sides, fixed) = outline(result, &inside);
    let outline_loops = loops(&sides)?;
    // The source's triangles whose middle lies nearest a region face off the outline, facing the same way: the strip
    // between them and the outline then has some width everywhere, so it doesn't fold.
    let on_outline: HashSet<i64> = sides.iter().map(|e| e.0).collect();
    let inner: Vec<bool> = (0..result.faces()).map(|f| inside[f] && !result.face(f).iter().any(|v| on_outline.contains(v))).collect();
    let result_tree = near::FaceTree::new(&result.points, &result.indptr, &result.flat);
    let (tris, _) = retopo::triangles(source);
    let middles: Vec<f64> = tris.chunks_exact(3).flat_map(|t| {
        let (a, b, c) = (p(&source.points, t[0]), p(&source.points, t[1]), p(&source.points, t[2]));
        (0..3).map(move |k| (a[k] + b[k] + c[k]) / 3.0)
    }).collect();
    let stop = AtomicBool::new(false);
    let (_, nearest) = near::nearest_faces(&result_tree, &middles, &stop);
    let face_normal = |f: usize| newell(&result.points, result.face(f));
    let mut points = result.points.clone();
    let mut number: HashMap<i64, i64> = HashMap::new();
    let mut patch = Vec::new();
    for (t, hit) in nearest.iter().enumerate() {
        let f = hit.1 as usize;
        let c = &tris[t * 3..t * 3 + 3];
        let n = cross(sub(p(&source.points, c[1]), p(&source.points, c[0])), sub(p(&source.points, c[2]), p(&source.points, c[0])));
        if !inner[f] || dot(n, face_normal(f)) <= 0.0 {
            continue;
        }
        for &v in c {
            let id = *number.entry(v).or_insert_with(|| {
                points.extend(p(&source.points, v));
                (points.len() / 3 - 1) as i64
            });
            patch.push(id);
        }
    }
    // Ragged edges off: triangles at a corner where the border touches itself, and spikes (two sides on the border).
    for _ in 0..MOST_TRIMS {
        let border = border_sides(&patch);
        let mut ways: HashMap<i64, usize> = HashMap::new();
        for &(a, _) in &border {
            *ways.entry(a).or_default() += 1;
        }
        let edge: HashSet<(i64, i64)> = border.iter().copied().collect();
        let before = patch.len();
        patch = patch.chunks_exact(3).filter(|t| t.iter().all(|v| ways.get(v).is_none_or(|&n| n < 2)) && (0..3).filter(|&k| edge.contains(&(t[k], t[(k + 1) % 3]))).count() < 2).flatten().copied().collect();
        if patch.len() == before {
            break;
        }
    }
    if patch.is_empty() {
        return Err("no source faces lie under this spot".into());
    }
    // The patch's border loops, one for each outline loop; bits of the patch with none of them are left out.
    let np = patch.len() / 3;
    let mut uses: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for t in 0..np {
        for k in 0..3 {
            uses.entry(key(patch[t * 3 + k], patch[t * 3 + (k + 1) % 3])).or_default().push(t);
        }
    }
    if uses.values().any(|u| u.len() > 2) {
        return Err("the source is tangled here (an edge with more than two faces)".into());
    }
    let mut parts = topo::Dsu::new(np);
    for u in uses.values().filter(|u| u.len() == 2) {
        parts.union(u[0] as u32, u[1] as u32);
    }
    let mut border = Vec::new();
    let mut border_part = HashMap::new();
    for t in 0..np {
        for k in 0..3 {
            let (a, b) = (patch[t * 3 + k], patch[t * 3 + (k + 1) % 3]);
            if uses[&key(a, b)].len() == 1 {
                border.push((a, b));
                border_part.insert(a, parts.find(t as u32));
            }
        }
    }
    let patch_loops = loops(&border).map_err(|e| format!("the source's faces here can't be put back: {}", e.replace("its edge", "their border")))?;
    // Closest pairs first: the mean distance from each loop's corners to the other loop, both ways.
    let one_way = |o: &[i64], b: &[i64]| o.iter().map(|&v| b.iter().map(|&w| dist(p(&points, v), p(&points, w))).fold(f64::INFINITY, f64::min)).sum::<f64>() / o.len() as f64;
    let gap = |o: &[i64], b: &[i64]| one_way(o, b) + one_way(b, o);
    let mut options: Vec<(f64, usize, usize)> = outline_loops.iter().enumerate().flat_map(|(i, o)| patch_loops.iter().enumerate().map(move |(j, b)| (gap(o, b), i, j))).collect();
    options.sort_by(|x, y| x.0.total_cmp(&y.0));
    let (mut matched, mut used) = (vec![None; outline_loops.len()], vec![false; patch_loops.len()]);
    for (_, i, j) in options {
        if matched[i].is_none() && !used[j] {
            (matched[i], used[j]) = (Some(j), true);
        }
    }
    if matched.iter().any(|m| m.is_none()) {
        return Err("the source here has fewer borders than the spot's outline".into());
    }
    let kept_parts: HashSet<u32> = if outline_loops.is_empty() { (0..np as u32).map(|t| parts.find(t)).collect() } else { matched.iter().map(|j| border_part[&patch_loops[j.unwrap()][0]]).collect() };
    // Region triangles: the kept patch, a strip zipped between each outline loop and its patch loop, and any other
    // border loop of the kept patch (a gap, or a hole the source has) closed with a fan.
    let made_from = points.len() / 3;
    let mut region_tris = Vec::new();
    for t in (0..np).filter(|&t| kept_parts.contains(&parts.find(t as u32))) {
        region_tris.extend_from_slice(&patch[t * 3..t * 3 + 3]);
    }
    for (o, j) in outline_loops.iter().zip(&matched) {
        zip(&points, o, &patch_loops[j.unwrap()], &mut region_tris);
    }
    for (_, b) in patch_loops.iter().enumerate().filter(|(j, b)| !used[*j] && kept_parts.contains(&border_part[&b[0]])) {
        let n = b.len() as f64;
        let centre = b.iter().fold([0.0; 3], |s, &v| [0, 1, 2].map(|k| s[k] + p(&points, v)[k] / n));
        points.extend(centre);
        let c = (points.len() / 3 - 1) as i64;
        for k in 0..b.len() {
            region_tris.extend([b[(k + 1) % b.len()], b[k], c]);
        }
    }
    // Each triangle takes the material of the result face nearest it.
    let middles: Vec<f64> = region_tris.chunks_exact(3).flat_map(|t| {
        let (a, b, c) = (p(&points, t[0]), p(&points, t[1]), p(&points, t[2]));
        (0..3).map(move |k| (a[k] + b[k] + c[k]) / 3.0)
    }).collect();
    let (_, nearest) = near::nearest_faces(&result_tree, &middles, &stop);
    let tri_material: Vec<i64> = nearest.iter().map(|h| result.material[h.1 as usize]).collect();
    let (fa, fb, cands, scores) = retopo::candidates(&points, &region_tris, &tri_material);
    let graph = pairing::allowed(tri_material.len(), &fa, &fb, &scores, None);
    let paired = pairing::pair(&graph, &cands, 0, &near::Watch::now()).ok_or("pairing was stopped")?;
    let mut faces: Vec<Vec<i64>> = paired.quads.chunks_exact(4).map(|q| q.to_vec()).collect();
    faces.extend(paired.tris.iter().map(|&t| region_tris[t as usize * 3..t as usize * 3 + 3].to_vec()));
    let face_material: Vec<i64> = paired.owner.iter().map(|&t| tri_material[t as usize]).collect();
    let (quads, from) = split_to_quads(&mut points, &faces, &fixed, false)?;
    let added: Vec<i64> = (made_from as i64..(points.len() / 3) as i64).collect();
    snap(&mut points, &added, &quads, source, &source_tree(source), &Valid::of(&result.points));
    let material: Vec<i64> = from.iter().map(|&f| face_material[f]).collect();
    let (mesh, new_faces) = splice(result, &inside, points, &quads, &material);
    judge(&before, mesh, inside.iter().filter(|&&x| x).count(), new_faces)
}

/// Sides of triangles `tris` that only one of them uses, as they run in it.
fn border_sides(tris: &[i64]) -> Vec<(i64, i64)> {
    let mut uses: HashMap<(i64, i64), usize> = HashMap::new();
    for t in tris.chunks_exact(3) {
        for k in 0..3 {
            *uses.entry(key(t[k], t[(k + 1) % 3])).or_default() += 1;
        }
    }
    tris.chunks_exact(3).flat_map(|t| (0..3).map(move |k| (t[k], t[(k + 1) % 3]))).filter(|&(a, b)| uses[&key(a, b)] == 1).collect()
}

/// Stitch outline loop `o` (outside) to patch loop `b` (inside) with triangles, each step taking the shorter new
/// side. Both loops run as their faces do, so the strip faces the same way as both.
fn zip(points: &[f64], o: &[i64], b: &[i64], out: &mut Vec<i64>) {
    let (no, nb) = (o.len(), b.len());
    let start = (0..nb).min_by(|&i, &j| dist(p(points, o[0]), p(points, b[i])).total_cmp(&dist(p(points, o[0]), p(points, b[j])))).unwrap();
    let ov = |i: usize| o[i % no];
    let bv = |j: usize| b[(start + j) % nb];
    let (mut i, mut j) = (0, 0);
    while i < no || j < nb {
        let along_outline = j == nb || (i < no && dist(p(points, ov(i + 1)), p(points, bv(j))) <= dist(p(points, ov(i)), p(points, bv(j + 1))));
        if along_outline {
            out.extend([ov(i), ov(i + 1), bv(j)]);
            i += 1;
        } else {
            out.extend([ov(i), bv(j + 1), bv(j)]);
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ball of quads: a cube's sides cut n x n and pushed out onto the unit sphere.
    fn ball(n: i64) -> Mesh {
        let mut mesh = Mesh { indptr: vec![0], ..Default::default() };
        let mut index: HashMap<[i64; 3], i64> = HashMap::new();
        for axis in 0..3 {
            for side in [0, n] {
                for i in 0..n {
                    for j in 0..n {
                        let corner = |di: i64, dj: i64| {
                            let mut c = [0; 3];
                            c[axis] = side;
                            c[(axis + 1) % 3] = i + di;
                            c[(axis + 2) % 3] = j + dj;
                            c
                        };
                        let mut q = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
                        if side == 0 {
                            q.reverse();
                        }
                        for c in q {
                            let v = *index.entry(c).or_insert_with(|| {
                                let x = c.map(|k| k as f64 / n as f64 * 2.0 - 1.0);
                                let l = dot(x, x).sqrt();
                                mesh.points.extend(x.map(|k| k / l));
                                (mesh.points.len() / 3 - 1) as i64
                            });
                            mesh.flat.push(v);
                        }
                        mesh.indptr.push(mesh.flat.len() as i64);
                        mesh.material.push(0);
                    }
                }
            }
        }
        mesh
    }

    /// A fine ball as triangles (the source) and a coarse ball of quads with one corner pulled sideways (the result).
    fn source_and_bad_result() -> (Mesh, Mesh, Vec<usize>) {
        let fine = ball(12);
        let mut source = Mesh { points: fine.points.clone(), indptr: vec![0], ..Default::default() };
        for f in 0..fine.faces() {
            let q = fine.face(f);
            for t in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
                source.flat.extend(t);
                source.indptr.push(source.flat.len() as i64);
                source.material.push(0);
            }
        }
        let mut result = ball(4);
        // Pull a corner in the middle of one side far over, folding the quads round it.
        let v = (0..result.points.len() / 3).find(|&v| (0..result.faces()).filter(|&f| result.face(f).contains(&(v as i64))).count() == 4).unwrap();
        let f = (0..result.faces()).find(|&f| result.face(f).contains(&(v as i64))).unwrap();
        let k = result.face(f).iter().position(|&w| w == v as i64).unwrap();
        let across = p(&result.points, result.face(f)[(k + 2) % 4]);
        let here = p(&result.points, v as i64);
        for c in 0..3 {
            result.points[v * 3 + c] = here[c] + 0.8 * (across[c] - here[c]);
        }
        let check = issues::check(&result);
        assert!(check.bad > 0, "the pulled corner makes bad faces");
        let spot = check.spots[0].faces.clone();
        (source, result, spot)
    }

    fn outside_kept(result: &Mesh, fixed: &Fixed, spot: &[usize], rings: usize) {
        let inside = region(result, spot, rings);
        let outside = (0..result.faces()).filter(|&f| !inside[f]).count();
        assert_eq!(fixed.mesh.faces(), outside + fixed.faces_after);
        for (g, f) in (0..result.faces()).filter(|&f| !inside[f]).enumerate() {
            let was: Vec<V3> = result.face(f).iter().map(|&v| p(&result.points, v)).collect();
            let now: Vec<V3> = fixed.mesh.face(g).iter().map(|&v| p(&fixed.mesh.points, v)).collect();
            assert_eq!(was, now, "outside faces stay exactly as they were");
        }
        let c = issues::check(&fixed.mesh);
        assert_eq!((c.quads, c.non_manifold_edges, c.holes), (c.faces, 0, 0), "all quads, closed and manifold");
    }

    #[test]
    fn redo_makes_smaller_quads_on_the_source_and_clears_the_fold() {
        let (source, result, spot) = source_and_bad_result();
        let fixed = redo_spot(&source, &result, &spot).unwrap();
        outside_kept(&result, &fixed, &spot, RINGS);
        assert!(fixed.faces_after > fixed.faces_before);
        assert!(fixed.bad_after < fixed.bad_before, "{} -> {}", fixed.bad_before, fixed.bad_after);
    }

    #[test]
    fn keep_original_puts_the_source_back() {
        let (source, result, spot) = source_and_bad_result();
        let fixed = keep_original(&source, &result, &spot).unwrap();
        outside_kept(&result, &fixed, &spot, RINGS + 1);
        assert!(fixed.faces_after > fixed.faces_before);
        assert!(fixed.bad_after < fixed.bad_before, "{} -> {}", fixed.bad_before, fixed.bad_after);
    }
}
