// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! The retopology itself: reduce (QEM), pair triangles into quads, close every piece to all quads, and look at the
//! result from six sides to keep the fewest faces whose view changes no more than asked (`AutoJob._run` in the add-on,
//! for the whole object at once; see PLAN.md for what the add-on does per part that this doesn't yet).

use crate::io::Mesh;
use crate::{close, geom, gpu, near, pairing, qem, topo};
use std::collections::HashMap;
use std::time::Instant;

pub const TURNED_DEG: f64 = 20.0; // a pixel whose surface turned more than this counts as changed
pub const MAX_QUAD_WARP_DEGREES: f64 = 60.0;
const FEATURE_DEG: f64 = 60.0; // pair_and_close: triangles meeting sharper than this aren't paired
const PAIRED_SHARE: f64 = 0.6; // first guess of faces after pairing per triangle before it
const CLOSE_ENOUGH: f64 = 0.10; // Reduce To: this near the face count asked for is a hit
const ROUNDS: usize = 4;
const MOST_LOOKS: usize = 12;
const TURN_LOSS: f64 = 1.0; // points of view change evening the quads out may add

#[derive(Clone, Debug)]
pub struct Settings {
    pub detail: f64,               // percent of the view that may change
    pub reduce_to: Option<usize>,  // a face count instead
    pub pixels: usize,             // a side of one picture (the add-on's Normal setting: 192)
    pub gpu: bool,
    pub rows: bool,                // offer even rows of quads to smooth parts
    pub baked: bool,               // judge the result as it will look with a normal map baked from the source
}

pub struct Outcome {
    pub mesh: Mesh,
    pub loss: f64,
    pub looks: usize,
    pub seconds: f64,
}

// ------------------------------------------------------------------------------------------------------ the camera

pub struct View {
    center: [f64; 3],
    half: f64,
    pixels: usize,
    pixel: f64,
    gpu: bool,
}

pub struct Pictures {
    hit: Vec<bool>,
    normal: Vec<f64>,
    depth: Vec<f64>,
}

/// The six views for the source's bounding box, as in `AutoJob._run`.
pub fn view_of(points: &[f64], pixels: usize, gpu: bool) -> View {
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in points.chunks_exact(3) {
        for c in 0..3 {
            low[c] = low[c].min(p[c]);
            high[c] = high[c].max(p[c]);
        }
    }
    let extent = (0..3).map(|c| high[c] - low[c]).fold(0.0, f64::max);
    let half = if extent > 0.0 { extent } else { 1.0 } * 0.55;
    View { center: [0, 1, 2].map(|c| (low[c] + high[c]) / 2.0), half, pixels, pixel: 2.0 * half / pixels as f64, gpu }
}

/// Fan triangles of every face: (corners, 3 per triangle; the face each came from).
pub fn triangles(mesh: &Mesh) -> (Vec<i64>, Vec<i64>) {
    let (mut tris, mut owner) = (Vec::new(), Vec::new());
    for f in 0..mesh.faces() {
        let face = mesh.face(f);
        for k in 1..face.len().saturating_sub(1) {
            tris.extend_from_slice(&[face[0], face[k], face[k + 1]]);
            owner.push(f as i64);
        }
    }
    (tris, owner)
}

fn face_normals(points: &[f64], tris: &[i64]) -> Vec<f64> {
    let p = |v: i64| [points[v as usize * 3], points[v as usize * 3 + 1], points[v as usize * 3 + 2]];
    tris.chunks_exact(3).flat_map(|t| {
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        cross(sub(b, a), sub(c, a))
    }).collect()
}

pub fn pictures(points: &[f64], tris: &[i64], view: &View) -> Pictures {
    let normals = face_normals(points, tris);
    let made = if view.gpu {
        gpu::pictures(points, tris, &normals, view.center, view.half, view.pixels).ok()
    } else {
        None
    };
    let (hit, normal, depth, _face) = made.unwrap_or_else(|| geom::pictures(points, tris, &normals, view.center, view.half, view.pixels));
    Pictures { hit, normal, depth }
}

/// `picture_counts` as a percent: pixels that differ in outline, direction or depth, of those showing either mesh.
pub fn loss(reference: &Pictures, other: &Pictures, view: &View) -> f64 {
    let limit = TURNED_DEG.to_radians().cos();
    let (mut wrong, mut total) = (0usize, 0usize);
    for i in 0..reference.hit.len() {
        let (a, b) = (reference.hit[i], other.hit[i]);
        if !(a || b) {
            continue;
        }
        total += 1;
        if a != b {
            wrong += 1;
            continue;
        }
        let dot: f64 = (0..3).map(|k| reference.normal[i * 3 + k] * other.normal[i * 3 + k]).sum();
        if dot.abs() < limit || (reference.depth[i] - other.depth[i]).abs() > 2.0 * view.pixel {
            wrong += 1;
        }
    }
    100.0 * wrong as f64 / total.max(1) as f64
}

// ------------------------------------------------------------------------------------------------------ pairing

type V3 = [f64; 3];
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V3) -> V3 {
    let n = norm(a);
    if n > 1e-20 { [a[0] / n, a[1] / n, a[2] / n] } else { [0.0; 3] }
}

/// `engine._min_scaled_jacobians` for one quad.
fn min_scaled_jacobian(p: &[V3; 4]) -> f64 {
    let mut normal = [0.0; 3];
    for k in 0..4 {
        let c = cross(p[k], p[(k + 1) % 4]);
        normal = [normal[0] + c[0], normal[1] + c[1], normal[2] + c[2]];
    }
    let length = norm(normal);
    if length <= 1e-20 {
        return -1.0;
    }
    let normal = [normal[0] / length, normal[1] / length, normal[2] / length];
    let mut least = 1.0f64;
    for k in 0..4 {
        let (following, previous) = (sub(p[(k + 1) % 4], p[k]), sub(p[(k + 3) % 4], p[k]));
        let denominator = norm(following) * norm(previous);
        if denominator <= 1e-20 {
            return -1.0;
        }
        least = least.min(dot(cross(following, previous), normal) / denominator);
    }
    least
}

/// `engine._pair_candidates` + `_pair_scores` with the default weights: (face_a, face_b, quads 4 per row, scores).
/// ponytail: the cross-field term aligns quad sides with face_a's longest edge, not the add-on's smoothed cross field
/// (`engine._estimate_field`); port that when quad rows on curved parts need to line up better.
fn candidates(points: &[f64], tris: &[i64], material: &[i64]) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<f64>) {
    let (w_normal, w_field, w_material, w_aspect, w_diagonal) = (2.2, 0.90, 0.15, 0.90, 0.35);
    let p = |v: i64| [points[v as usize * 3], points[v as usize * 3 + 1], points[v as usize * 3 + 2]];
    let nf = tris.len() / 3;
    let normals: Vec<V3> = (0..nf).map(|f| {
        let t = &tris[f * 3..f * 3 + 3];
        unit(cross(sub(p(t[1]), p(t[0])), sub(p(t[2]), p(t[0]))))
    }).collect();
    let mut uses: HashMap<(i64, i64), Vec<(usize, usize)>> = HashMap::with_capacity(tris.len());
    for f in 0..nf {
        for c in 0..3 {
            let (x, y) = (tris[f * 3 + c], tris[f * 3 + (c + 1) % 3]);
            uses.entry((x.min(y), x.max(y))).or_default().push((f, c));
        }
    }
    let span = {
        let v = crate::retopo::view_of(points, 1, false);
        2.0 * v.half / 0.55
    };
    let epsilon = (span * span * 1e-14).max(1e-20);
    let warp = MAX_QUAD_WARP_DEGREES.to_radians().cos();
    let sharp = FEATURE_DEG.to_radians().cos();
    let (mut fa, mut fb, mut quads, mut scores) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut edges: Vec<_> = uses.into_iter().filter(|(_, u)| u.len() == 2).map(|(_, u)| (u[0], u[1])).collect();
    edges.sort_unstable(); // HashMap order is random: keep the result the same run to run
    for ((a, ca), (b, cb)) in edges {
        let x = tris[a * 3 + ca];
        let y = tris[a * 3 + (ca + 1) % 3];
        if a == b || tris[b * 3 + cb] != y || tris[b * 3 + (cb + 1) % 3] != x {
            continue; // the shared edge must run opposite ways (consistent winding)
        }
        if material[a] != material[b] || dot(normals[a], normals[b]) < sharp {
            continue;
        }
        let quad = [tris[a * 3 + (ca + 2) % 3], x, tris[b * 3 + (cb + 2) % 3], y];
        let mut sorted = quad;
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            continue;
        }
        let q = quad.map(p);
        if !geom::quad_valid(&q, epsilon, warp) {
            continue;
        }
        let sides: Vec<V3> = (0..4).map(|k| sub(q[(k + 1) % 4], q[k])).collect();
        let lengths: Vec<f64> = sides.iter().map(|s| norm(*s)).collect();
        let longest = lengths.iter().cloned().fold(0.0, f64::max);
        let aspect = longest / lengths.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-12);
        let diagonal = norm(sub(q[3], q[1])) / (lengths.iter().sum::<f64>() * 0.25).max(1e-9);
        let along = {
            let t = &tris[a * 3..a * 3 + 3];
            let e = (0..3).map(|k| sub(p(t[(k + 1) % 3]), p(t[k]))).max_by(|u, v| norm(*u).total_cmp(&norm(*v))).unwrap();
            unit(e)
        };
        let across = unit(cross(normals[a], along));
        let field = sides.iter().map(|s| {
            let u = unit(*s);
            dot(u, along).abs().max(dot(u, across).abs())
        }).sum::<f64>() / 4.0;
        let score = dot(normals[a], normals[b]).clamp(-1.0, 1.0) * w_normal + field * w_field + min_scaled_jacobian(&q) * 3.0 + w_material
            - aspect.max(1.0).ln() * w_aspect
            - diagonal * w_diagonal;
        fa.push(a as i64);
        fb.push(b as i64);
        quads.extend_from_slice(&quad);
        scores.push(if aspect.is_finite() && aspect > 0.0 && score.is_finite() { score } else { -100.0 });
    }
    (fa, fb, quads, scores)
}

/// `flatquad.pair_and_close` for triangles: repair, pair into quads, then close every piece to all quads.
pub fn pair_and_close(points: &[f64], tris: &[i64], material: &[i64], materials: &[String]) -> Result<Mesh, String> {
    let indptr: Vec<i64> = (0..=(tris.len() / 3) as i64).map(|f| f * 3).collect();
    close_polygons(points, &indptr, tris, material, materials)
}

/// The same for polygons, with `keep_quads`: quads stay whole, larger faces are cut into triangles from their first
/// corner, and only the triangles are paired before every piece is closed to all quads.
pub fn close_polygons(points: &[f64], indptr: &[i64], flat: &[i64], material: &[i64], materials: &[String]) -> Result<Mesh, String> {
    let span = 2.0 * view_of(points, 1, false).half / 0.55 * 3f64.sqrt();
    let source = close::Mesh { vertices: points.to_vec(), indptr: indptr.to_vec(), flat: flat.to_vec(), origin: material.to_vec() };
    let (fixed, _added) = close::make_closable(&source, span);
    let count = fixed.indptr.len() - 1;
    if count == 0 {
        return Ok(Mesh { indptr: vec![0], materials: materials.to_vec(), ..Default::default() });
    }
    let (mut kept_flat, mut kept_tag) = (Vec::new(), Vec::new());
    let (mut tris, mut tri_tag) = (Vec::new(), Vec::new());
    for f in 0..count {
        let face = &fixed.flat[fixed.indptr[f] as usize..fixed.indptr[f + 1] as usize];
        if face.len() == 4 {
            kept_flat.extend_from_slice(face);
            kept_tag.push(fixed.origin[f]);
        } else {
            for k in 1..face.len() - 1 {
                tris.extend_from_slice(&[face[0], face[k], face[k + 1]]);
                tri_tag.push(fixed.origin[f]);
            }
        }
    }
    let (mut out_indptr, mut out_flat, mut out_tag) = (vec![0i64], kept_flat, kept_tag);
    for _ in 0..out_tag.len() {
        out_indptr.push(out_indptr.last().unwrap() + 4);
    }
    if !tris.is_empty() {
        let (fa, fb, quads, scores) = candidates(&fixed.vertices, &tris, &tri_tag);
        let graph = pairing::allowed(tri_tag.len(), &fa, &fb, &scores, None);
        // One order is enough: the augmenting search pairs as many as can be, so other orders leave as many triangles.
        let paired = pairing::pair(&graph, &quads, 0, &near::Watch::now()).ok_or("pairing was stopped")?;
        for q in paired.quads.chunks_exact(4) {
            out_flat.extend_from_slice(q);
            out_indptr.push(out_flat.len() as i64);
        }
        for &t in &paired.tris {
            out_flat.extend_from_slice(&tris[t as usize * 3..t as usize * 3 + 3]);
            out_indptr.push(out_flat.len() as i64);
        }
        out_tag.extend(paired.owner.iter().map(|&f| tri_tag[f as usize]));
    }
    let (closed, _counts) = close::close_pieces(&topo::Faces { indptr: &out_indptr, flat: &out_flat }, &fixed.vertices, MAX_QUAD_WARP_DEGREES.to_radians().cos())?;
    Ok(Mesh {
        material: closed.origin.iter().map(|&f| out_tag[f as usize]).collect(),
        points: closed.vertices,
        indptr: closed.indptr,
        flat: closed.flat,
        materials: materials.to_vec(),
    })
}

// ------------------------------------------------------------------------------------------------------ the search

pub fn retopo(source: &Mesh, settings: &Settings, mut log: impl FnMut(&str)) -> Result<Outcome, String> {
    let started = Instant::now();
    let prepared = Prepared::new(source, settings.pixels, settings.gpu, settings.baked, &mut log);
    let mut out = prepared.solve(settings, log)?;
    out.seconds = started.elapsed().as_secs_f64();
    Ok(out)
}

/// What every search on one source shares: its camera pictures and its one reduction order. Made once, it answers
/// any number of Detail or face-count searches (the Quad Size ladder, `ladder.rs`).
pub struct Prepared<'a> {
    source: &'a Mesh,
    tris: Vec<i64>,
    tri_piece: Vec<i64>,
    tags: Vec<i64>,
    view: View,
    reference: Pictures,
    run: qem::Run,
    source_tree: near::FaceTree,
    baked: bool,
    source_normals: Vec<[f64; 3]>, // per source face, for what a bake paints
}

impl<'a> Prepared<'a> {
    pub fn new(source: &'a Mesh, pixels: usize, gpu: bool, baked: bool, log: &mut dyn FnMut(&str)) -> Prepared<'a> {
        let (tris, owner) = triangles(source);
        let tri_material: Vec<i64> = owner.iter().map(|&f| source.material[f as usize]).collect();
        let tri_piece = pieces(&tris, source.points.len() / 3);
        // Faces carry (material, part) packed in one tag through pairing and closing; parts never touch, so pairing,
        // which won't join faces with different tags, is unchanged.
        let tags: Vec<i64> = tri_material.iter().zip(&tri_piece).map(|(&m, &p)| pack(m, p)).collect();
        let view = view_of(&source.points, pixels, gpu);
        let reference = pictures(&source.points, &tris, &view);
        log(&format!("Reducing {} triangles", tris.len() / 3));
        let tris_u32: Vec<u32> = tris.iter().map(|&v| v as u32).collect();
        let run = qem::run(&source.points, &tris_u32, &[], 4);
        let source_tree = near::FaceTree::new(&source.points, &source.indptr, &source.flat);
        let source_normals = face_normals_of(source);
        Prepared { source, tris, tri_piece, tags, view, reference, run, source_tree, baked, source_normals }
    }

    /// The view change of `mesh` against the source, in percent: as it is, or as it will look once baked.
    pub fn look(&self, mesh: &Mesh) -> f64 {
        let seen = pictures(&mesh.points, &triangles(mesh).0, &self.view);
        if self.baked {
            loss_baked(&self.reference, &painted(seen, &self.view, &self.source_tree, &self.source_normals), &self.view)
        } else {
            loss(&self.reference, &seen, &self.view)
        }
    }

    /// One search. `settings.pixels` and `settings.gpu` were fixed by `new`; the rest is used.
    pub fn solve(&self, settings: &Settings, mut log: impl FnMut(&str)) -> Result<Outcome, String> {
        let started = Instant::now();
        let (source, tris, tri_piece, tags, run) = (self.source, &self.tris, &self.tri_piece, &self.tags, &self.run);
        let total = tris.len() / 3;
        let smallest = (run.smallest() as usize).max(12);
        let mut looks = 0;
        let mut make = |wanted: usize| -> Result<(Mesh, f64), String> {
            let (points, level_tris, from) = run.level(wanted as u32);
            let material: Vec<i64> = from.iter().map(|&t| tags[t as usize]).collect();
            let out = pair_and_close(&points, &level_tris, &material, &source.materials)?;
            looks += 1;
            let change = self.look(&out);
            Ok((out, change))
        };
        let (mesh, change) = if let Some(goal) = settings.reduce_to {
            let mut wanted = ((goal as f64 / PAIRED_SHARE) as usize).clamp(smallest, total);
            let mut best: Option<(f64, Mesh, f64)> = None;
            for _ in 0..ROUNDS {
                let (out, change) = make(wanted)?;
                let miss = (out.faces() as f64 - goal as f64).abs() / goal.max(1) as f64;
                log(&format!("  {} triangles kept -> {} faces, {:.2}% of the view changed", wanted, out.faces(), change));
                let share = out.faces() as f64 / wanted.max(1) as f64;
                if best.as_ref().is_none_or(|b| miss < b.0) {
                    best = Some((miss, out, change));
                }
                if miss <= CLOSE_ENOUGH || share <= 0.0 {
                    break;
                }
                wanted = ((goal as f64 / share) as usize).clamp(smallest, total);
            }
            let (_, mesh, change) = best.unwrap();
            (mesh, change)
        } else {
            // The fewest triangles kept whose closed quads change no more than `detail` of the view, halving in log space.
            let (mut low, mut high) = (smallest as f64, total as f64);
            let mut best: Option<(Mesh, f64)> = None;
            for _ in 0..MOST_LOOKS {
                if high / low < 1.05 {
                    break;
                }
                let middle = (low * high).sqrt() as usize;
                let (out, change) = make(middle)?;
                log(&format!("  {} triangles kept -> {} faces, {:.2}% of the view changed", middle, out.faces(), change));
                if change <= settings.detail {
                    high = middle as f64;
                    let near = change >= 0.9 * settings.detail;
                    best = Some((out, change));
                    if near {
                        break;
                    }
                } else {
                    low = middle as f64;
                }
            }
            match best {
                Some(found) => found,
                None => make(high as usize)?,
            }
        };
        let (mut mesh, mut change) = (mesh, change);
        // Rows and evening out together may add at most TURN_LOSS points, and stay within 110% of the detail asked for
        // unless the search result was already over (the add-on's rule for spinning edges at poles).
        let limit = if settings.reduce_to.is_some() { change + TURN_LOSS } else { (change + TURN_LOSS).min(change.max(1.1 * settings.detail)) };
        if settings.rows {
            let mut look = |m: &Mesh| {
                looks += 1;
                self.look(m)
            };
            let swapped = rows_where_they_hold(&mut mesh, &mut change, limit, source, tris, tri_piece, tags, &mut look, &mut log);
            if swapped > 0 {
                log(&format!("  Rows of quads on {swapped} smooth part(s): {change:.2}% of the view changed"));
            }
        }
        // Even the quads out.
        let steps: [(&str, &dyn Fn(&mut Mesh) -> usize); 2] = [("Spinning edges at poles", &|m| crate::layout::spin_edges(m)), ("Relaxing the quads on the source", &|m| crate::layout::relax(m, &self.source_tree))];
        for (name, step) in steps {
            let mut tried = mesh.clone();
            if step(&mut tried) == 0 {
                continue;
            }
            let after = self.look(&tried);
            looks += 1;
            let keep = after <= limit;
            log(&format!("  {name}: {:.2}% of the view changed, {}", after, if keep { "kept" } else { "not kept" }));
            if keep {
                (mesh, change) = (tried, after);
            }
        }
        mesh.material = mesh.material.iter().map(|&t| unpack(t).0).collect();
        Ok(Outcome { mesh: crate::io::compact(&mesh), loss: change, looks, seconds: started.elapsed().as_secs_f64() })
    }
}

/// Part number per triangle: triangles sharing a corner are one part.
fn pieces(tris: &[i64], points: usize) -> Vec<i64> {
    let mut sets = topo::Dsu::new(points);
    for t in tris.chunks_exact(3) {
        sets.union(t[0] as u32, t[1] as u32);
        sets.union(t[1] as u32, t[2] as u32);
    }
    let mut number = HashMap::new();
    tris.chunks_exact(3).map(|t| {
        let root = sets.find(t[0] as u32);
        let next = number.len() as i64;
        *number.entry(root).or_insert(next)
    }).collect()
}

const PART_BITS: i64 = 1 << 32;
fn pack(material: i64, part: i64) -> i64 {
    (material + 1) * PART_BITS + part
}
fn unpack(tag: i64) -> (i64, i64) {
    (tag / PART_BITS - 1, tag % PART_BITS)
}

const ROWS_SMALLEST: usize = 1000; // source triangles: a smaller part isn't worth a grid (the add-on's FIELD_SMALLEST)
const ROWS_FEWEST: usize = 24; // faces a part must have in the result
const ROWS_MOST_MISS: f64 = 0.25; // rows more than this share off the part's face count are refused

/// Offer each big part even rows of quads (`rows.rs`) at about the faces it has now, biggest first, and keep a part's
/// rows when the whole view stays within `limit`. Returns how many parts took rows.
/// ponytail: one tag (material) per part's rows; a part with several materials gets its first one.
#[allow(clippy::too_many_arguments)]
fn rows_where_they_hold(mesh: &mut Mesh, change: &mut f64, limit: f64, source: &Mesh, tris: &[i64], tri_piece: &[i64], tags: &[i64], look: &mut dyn FnMut(&Mesh) -> f64, log: &mut dyn FnMut(&str)) -> usize {
    use rayon::prelude::*;
    let parts = tri_piece.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut source_tris = vec![Vec::new(); parts];
    for (t, &p) in tri_piece.iter().enumerate() {
        source_tris[p as usize].push(t);
    }
    let mut faces_of = vec![Vec::new(); parts];
    for f in 0..mesh.faces() {
        faces_of[unpack(mesh.material[f]).1 as usize].push(f);
    }
    let mut order: Vec<usize> = (0..parts).filter(|&p| source_tris[p].len() >= ROWS_SMALLEST && faces_of[p].len() >= ROWS_FEWEST).collect();
    order.sort_by_key(|&p| std::cmp::Reverse(source_tris[p].len()));
    // Every part's rows are built side by side (the add-on does them one at a time on a clock, to keep Blender
    // responsive). Parts never share faces, so each is judged against the part as it is now.
    let made: Vec<Option<(usize, Mesh, usize, usize)>> = order.par_iter().map(|&p| {
        let mine = &faces_of[p];
        let mut local = HashMap::new();
        let (mut points, mut part_tris) = (Vec::new(), Vec::new());
        let mut area = 0.0;
        for &t in &source_tris[p] {
            let c = &tris[t * 3..t * 3 + 3];
            let corner = |v: i64| [source.points[v as usize * 3], source.points[v as usize * 3 + 1], source.points[v as usize * 3 + 2]];
            area += 0.5 * norm(cross(sub(corner(c[1]), corner(c[0])), sub(corner(c[2]), corner(c[0]))));
            for &v in c {
                let id = *local.entry(v).or_insert_with(|| {
                    points.extend_from_slice(&source.points[v as usize * 3..v as usize * 3 + 3]);
                    (points.len() / 3 - 1) as i64
                });
                part_tris.push(id);
            }
        }
        // Cleaning merges some faces, so a count far off is tried once more with the cell corrected by what came out;
        // the try nearer the part's count is kept, and rows far off it are refused, so the face count stays honest.
        let miss = |made: &Option<(Vec<f64>, Vec<i64>, Vec<i64>)>| made.as_ref().map_or(f64::INFINITY, |m| ((m.1.len() - 1) as f64 / mine.len() as f64 - 1.0).abs());
        let cell = (area / mine.len() as f64).sqrt();
        let mut made = crate::rows::field_quads(&points, &part_tris, cell, p as u64);
        if (CLOSE_ENOUGH..f64::INFINITY).contains(&miss(&made)) {
            let faces = (made.as_ref().unwrap().1.len() - 1) as f64;
            let again = crate::rows::field_quads(&points, &part_tris, cell * (faces / mine.len() as f64).sqrt(), p as u64);
            if miss(&again) < miss(&made) {
                made = again;
            }
        }
        if miss(&made) > ROWS_MOST_MISS {
            return None;
        }
        let (grid_points, grid_indptr, grid_flat) = made?;
        let tag = tags[source_tris[p][0]];
        let quads = Mesh { material: vec![tag; grid_indptr.len() - 1], points: grid_points, indptr: grid_indptr, flat: grid_flat, materials: source.materials.clone() };
        // Rows must not bring in faces the checks call bad: compare with the part as it is.
        let bad_now = crate::issues::check(&sub_mesh(mesh, mine)).bad;
        let bad_rows = crate::issues::check(&quads).bad;
        Some((p, quads, bad_now, bad_rows))
    }).collect();
    // Then each one is looked at in the whole view, biggest part first, and kept while the view holds.
    let mut swapped = 0;
    for (p, quads, bad_now, bad_rows) in made.into_iter().flatten() {
        if bad_rows > bad_now {
            log(&format!("  Rows on part {p}: {bad_rows} bad faces where it has {bad_now}, not kept"));
            continue;
        }
        let mine: Vec<usize> = (0..mesh.faces()).filter(|&f| unpack(mesh.material[f]).1 == p as i64).collect();
        let tried = replace_faces(mesh, &mine, &quads);
        let after = look(&tried);
        let keep = after <= limit;
        log(&format!("  Rows on part {} ({} faces -> {}): {:.2}% of the view changed, {}", p, mine.len(), quads.faces(), after, if keep { "kept" } else { "not kept" }));
        if keep {
            *mesh = tried;
            *change = after;
            swapped += 1;
        }
    }
    swapped
}

/// Just these faces of `mesh`, with all its points.
fn sub_mesh(mesh: &Mesh, faces: &[usize]) -> Mesh {
    let mut out = Mesh { points: mesh.points.clone(), indptr: vec![0], materials: mesh.materials.clone(), ..Default::default() };
    for &f in faces {
        out.flat.extend_from_slice(mesh.face(f));
        out.indptr.push(out.flat.len() as i64);
        out.material.push(mesh.material[f]);
    }
    out
}

/// `mesh` without the faces `gone`, with `added`'s faces (and points) appended.
fn replace_faces(mesh: &Mesh, gone: &[usize], added: &Mesh) -> Mesh {
    let mut drop = vec![false; mesh.faces()];
    for &f in gone {
        drop[f] = true;
    }
    let mut out = Mesh { points: mesh.points.clone(), indptr: vec![0], materials: mesh.materials.clone(), ..Default::default() };
    for f in 0..mesh.faces() {
        if !drop[f] {
            out.flat.extend_from_slice(mesh.face(f));
            out.indptr.push(out.flat.len() as i64);
            out.material.push(mesh.material[f]);
        }
    }
    let offset = (out.points.len() / 3) as i64;
    out.points.extend_from_slice(&added.points);
    for f in 0..added.faces() {
        out.flat.extend(added.face(f).iter().map(|&v| v + offset));
        out.indptr.push(out.flat.len() as i64);
        out.material.push(added.material[f]);
    }
    out
}

/// The view change of `result` against `source`, with the same camera as `retopo`; `baked`: as it will look once a
/// normal map is baked onto it from the source.
pub fn measure(source: &Mesh, result: &Mesh, pixels: usize, gpu: bool, baked: bool) -> f64 {
    let view = view_of(&source.points, pixels, gpu);
    let reference = pictures(&source.points, &triangles(source).0, &view);
    let seen = pictures(&result.points, &triangles(result).0, &view);
    if baked {
        let tree = near::FaceTree::new(&source.points, &source.indptr, &source.flat);
        loss_baked(&reference, &painted(seen, &view, &tree, &face_normals_of(source)), &view)
    } else {
        loss(&reference, &seen, &view)
    }
}

// ------------------------------------------------------------------------------------------------ judged after baking

// The add-on's bakescore.py: a bake repaints the normals but moves nothing, so fine bumps don't count. What counts is an
// outline that moved, a surface far enough from the source that the map's detail slides as the view turns, and shading
// still wrong over a few pixels once the source's normals are painted on.
const BAKED_DEPTH_PIXELS: f64 = 3.0;
const BAKED_OUTLINE_PIXELS: usize = 1;
const BAKED_BLOCK: usize = 4; // shading compared over squares of this many pixels: shifted fine bumps cancel out there
const BAKED_TURNED_DEG: f64 = 30.0;
const BAKED_OUTLINE_WEIGHT: f64 = 0.5; // every outline pixel moved: half the view counts as changed (thin parts)

fn face_normals_of(mesh: &Mesh) -> Vec<[f64; 3]> {
    (0..mesh.faces()).map(|f| {
        let face = mesh.face(f);
        let p = |k: usize| { let v = face[k % face.len()] as usize; [mesh.points[v * 3], mesh.points[v * 3 + 1], mesh.points[v * 3 + 2]] };
        let mut n = [0.0; 3];
        for k in 0..face.len() {
            let (a, b) = (p(k), p(k + 1));
            n = [n[0] + (a[1] - b[1]) * (a[2] + b[2]), n[1] + (a[2] - b[2]) * (a[0] + b[0]), n[2] + (a[0] - b[0]) * (a[1] + b[1])];
        }
        unit(n)
    }).collect()
}

/// The pictures with each hit's normal replaced by the source's nearest face normal: about what a bake paints there.
fn painted(mut seen: Pictures, view: &View, tree: &near::FaceTree, normals: &[[f64; 3]]) -> Pictures {
    let cells = view.pixels * view.pixels;
    let scale = view.pixels as f64 / (2.0 * view.half);
    let hits: Vec<usize> = (0..seen.hit.len()).filter(|&i| seen.hit[i]).collect();
    let mut asks = Vec::with_capacity(hits.len() * 3);
    for &i in &hits {
        // Views run +x, -x, +y, -y, +z, -z; a pixel's row runs along the next axis, its column along the one after.
        let (picture, cell) = (i / cells, i % cells);
        let (axis, back) = (picture / 2, picture % 2 == 1);
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let (row, col) = ((cell / view.pixels) as f64, (cell % view.pixels) as f64);
        let mut p = [0.0; 3];
        p[u] = view.center[u] - view.half + (row + 0.5) / scale;
        p[v] = view.center[v] - view.half + (col + 0.5) / scale;
        p[axis] = if back { view.center[axis] - 2.0 * view.half + seen.depth[i] } else { view.center[axis] + 2.0 * view.half - seen.depth[i] };
        asks.extend(p);
    }
    let (_, found) = near::nearest_faces(tree, &asks, &std::sync::atomic::AtomicBool::new(false));
    for (&i, hit) in hits.iter().zip(found) {
        seen.normal[i * 3..i * 3 + 3].copy_from_slice(&normals[hit.1 as usize]);
    }
    seen
}

fn grow(mask: &[bool], pixels: usize, steps: usize) -> Vec<bool> {
    let mut out = mask.to_vec();
    for _ in 0..steps {
        let was = out.clone();
        for r in 0..pixels {
            for c in 0..pixels {
                let i = r * pixels + c;
                out[i] = was[i] || (r > 0 && was[i - pixels]) || (r + 1 < pixels && was[i + pixels]) || (c > 0 && was[i - 1]) || (c + 1 < pixels && was[i + 1]);
            }
        }
    }
    out
}

/// `bakescore.picture_counts` as a percent of the pixels showing either mesh.
fn loss_baked(reference: &Pictures, other: &Pictures, view: &View) -> f64 {
    let (n, cells) = (view.pixels, view.pixels * view.pixels);
    let limit = BAKED_TURNED_DEG.to_radians().cos();
    let (mut wrong, mut total, mut moved, mut outline) = (0usize, 0usize, 0usize, 0usize);
    for picture in 0..6 {
        let at = picture * cells;
        let a: &[bool] = &reference.hit[at..at + cells];
        let b: &[bool] = &other.hit[at..at + cells];
        let normal = |pics: &Pictures, i: usize| [pics.normal[(at + i) * 3], pics.normal[(at + i) * 3 + 1], pics.normal[(at + i) * 3 + 2]];
        let mut changed: Vec<bool> = (0..cells).map(|i| a[i] && b[i] && (reference.depth[at + i] - other.depth[at + i]).abs() > BAKED_DEPTH_PIXELS * view.pixel).collect();
        for br in 0..n / BAKED_BLOCK {
            for bc in 0..n / BAKED_BLOCK {
                let (mut sa, mut sb) = ([0.0; 3], [0.0; 3]);
                let cells_here: Vec<usize> = (0..BAKED_BLOCK * BAKED_BLOCK).map(|k| (br * BAKED_BLOCK + k / BAKED_BLOCK) * n + bc * BAKED_BLOCK + k % BAKED_BLOCK).collect();
                for &i in &cells_here {
                    if a[i] && b[i] {
                        let (na, mut nb) = (normal(reference, i), normal(other, i));
                        if na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2] < 0.0 {
                            nb = [-nb[0], -nb[1], -nb[2]]; // winding isn't shape
                        }
                        sa = [sa[0] + na[0], sa[1] + na[1], sa[2] + na[2]];
                        sb = [sb[0] + nb[0], sb[1] + nb[1], sb[2] + nb[2]];
                    }
                }
                let size = (sa[0] * sa[0] + sa[1] * sa[1] + sa[2] * sa[2]).sqrt() * (sb[0] * sb[0] + sb[1] * sb[1] + sb[2] * sb[2]).sqrt();
                if sa[0] * sb[0] + sa[1] * sb[1] + sa[2] * sb[2] < limit * size {
                    for &i in &cells_here {
                        changed[i] |= a[i] && b[i];
                    }
                }
            }
        }
        wrong += changed.iter().filter(|&&c| c).count();
        total += (0..cells).filter(|&i| a[i] || b[i]).count();
        let (wide_a, wide_b) = (grow(a, n, BAKED_OUTLINE_PIXELS), grow(b, n, BAKED_OUTLINE_PIXELS));
        moved += (0..cells).filter(|&i| (a[i] && !wide_b[i]) || (b[i] && !wide_a[i])).count();
        let outside: Vec<bool> = a.iter().map(|&x| !x).collect();
        let edge = grow(&outside, n, 1);
        outline += (0..cells).filter(|&i| a[i] && edge[i]).count();
    }
    let counted = wrong as f64 + BAKED_OUTLINE_WEIGHT * total as f64 * moved as f64 / outline.max(1) as f64;
    100.0 * counted / total.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed box of 2x2 quads per side, as triangles.
    fn cube() -> Mesh {
        let mut mesh = Mesh { indptr: vec![0], ..Default::default() };
        let mut index: HashMap<[i64; 3], i64> = HashMap::new();
        let mut at = |p: [i64; 3], mesh: &mut Mesh| *index.entry(p).or_insert_with(|| {
            mesh.points.extend(p.map(|x| x as f64));
            (mesh.points.len() / 3 - 1) as i64
        });
        for axis in 0..3 {
            for side in [0i64, 2] {
                for i in 0..2 {
                    for j in 0..2 {
                        let corner = |di: i64, dj: i64| {
                            let mut p = [0i64; 3];
                            p[axis] = side;
                            p[(axis + 1) % 3] = i + di;
                            p[(axis + 2) % 3] = j + dj;
                            p
                        };
                        let mut q = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
                        if side == 0 {
                            q.reverse();
                        }
                        let v = q.map(|p| at(p, &mut mesh));
                        for t in [[v[0], v[1], v[2]], [v[0], v[2], v[3]]] {
                            mesh.flat.extend(t);
                            mesh.indptr.push(mesh.flat.len() as i64);
                            mesh.material.push(0);
                        }
                    }
                }
            }
        }
        mesh
    }

    #[test]
    fn a_box_comes_out_all_quads_and_unchanged_in_view() {
        let source = cube();
        let settings = Settings { detail: 6.0, reduce_to: None, pixels: 64, gpu: false, rows: true, baked: false };
        let out = retopo(&source, &settings, |_| {}).unwrap();
        assert!(out.mesh.faces() > 0);
        assert!((0..out.mesh.faces()).all(|f| out.mesh.face(f).len() == 4), "all quads");
        assert!(out.loss <= 6.0, "loss {}", out.loss);
        assert!(out.mesh.faces() < 24, "fewer faces than the 24 source quads: {}", out.mesh.faces());
    }

    #[test]
    fn the_same_mesh_measures_no_change() {
        let source = cube();
        assert_eq!(measure(&source, &source, 64, false, false), 0.0);
        assert_eq!(measure(&source, &source, 64, false, true), 0.0);
    }
}
