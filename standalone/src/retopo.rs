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

pub struct Settings {
    pub detail: f64,               // percent of the view that may change
    pub reduce_to: Option<usize>,  // a face count instead
    pub pixels: usize,             // a side of one picture (the add-on's Normal setting: 192)
    pub gpu: bool,
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
    let nf = tris.len() / 3;
    let span = 2.0 * view_of(points, 1, false).half / 0.55 * 3f64.sqrt();
    let source = close::Mesh { vertices: points.to_vec(), indptr: (0..=nf as i64).map(|f| f * 3).collect(), flat: tris.to_vec(), origin: material.to_vec() };
    let (fixed, _added) = close::make_closable(&source, span);
    let count = fixed.indptr.len() - 1;
    if count == 0 {
        return Ok(Mesh { indptr: vec![0], materials: materials.to_vec(), ..Default::default() });
    }
    let (fa, fb, quads, scores) = candidates(&fixed.vertices, &fixed.flat, &fixed.origin);
    let graph = pairing::allowed(count, &fa, &fb, &scores, None);
    let paired = pairing::pair(&graph, &quads, 0, &near::Watch::now()).ok_or("pairing was stopped")?;
    let mut indptr = vec![0i64];
    let mut flat = Vec::with_capacity(fixed.flat.len());
    for q in paired.quads.chunks_exact(4) {
        flat.extend_from_slice(q);
        indptr.push(flat.len() as i64);
    }
    for &t in &paired.tris {
        flat.extend_from_slice(&fixed.flat[t as usize * 3..t as usize * 3 + 3]);
        indptr.push(flat.len() as i64);
    }
    let paired_material: Vec<i64> = paired.owner.iter().map(|&f| fixed.origin[f as usize]).collect();
    let (closed, _counts) = close::close_pieces(&topo::Faces { indptr: &indptr, flat: &flat }, &fixed.vertices, MAX_QUAD_WARP_DEGREES.to_radians().cos())?;
    Ok(Mesh {
        material: closed.origin.iter().map(|&f| paired_material[f as usize]).collect(),
        points: closed.vertices,
        indptr: closed.indptr,
        flat: closed.flat,
        materials: materials.to_vec(),
    })
}

// ------------------------------------------------------------------------------------------------------ the search

pub fn retopo(source: &Mesh, settings: &Settings, mut log: impl FnMut(&str)) -> Result<Outcome, String> {
    let started = Instant::now();
    let (tris, owner) = triangles(source);
    let tri_material: Vec<i64> = owner.iter().map(|&f| source.material[f as usize]).collect();
    let view = view_of(&source.points, settings.pixels, settings.gpu);
    let reference = pictures(&source.points, &tris, &view);
    log(&format!("Reducing {} triangles", tris.len() / 3));
    let tris_u32: Vec<u32> = tris.iter().map(|&v| v as u32).collect();
    let run = qem::run(&source.points, &tris_u32, &[], 4);
    let total = tris.len() / 3;
    let smallest = (run.smallest() as usize).max(12);
    let mut looks = 0;
    let mut make = |wanted: usize| -> Result<(Mesh, f64), String> {
        let (points, level_tris, from) = run.level(wanted as u32);
        let material: Vec<i64> = from.iter().map(|&t| tri_material[t as usize]).collect();
        let out = pair_and_close(&points, &level_tris, &material, &source.materials)?;
        let seen = pictures(&out.points, &triangles(&out).0, &view);
        looks += 1;
        Ok((out, loss(&reference, &seen, &view)))
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
    Ok(Outcome { mesh, loss: change, looks, seconds: started.elapsed().as_secs_f64() })
}

/// The view change of `result` against `source`, with the same camera as `retopo`.
pub fn measure(source: &Mesh, result: &Mesh, pixels: usize, gpu: bool) -> f64 {
    let view = view_of(&source.points, pixels, gpu);
    let reference = pictures(&source.points, &triangles(source).0, &view);
    let seen = pictures(&result.points, &triangles(result).0, &view);
    loss(&reference, &seen, &view)
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
        let settings = Settings { detail: 6.0, reduce_to: None, pixels: 64, gpu: false };
        let out = retopo(&source, &settings, |_| {}).unwrap();
        assert!(out.mesh.faces() > 0);
        assert!((0..out.mesh.faces()).all(|f| out.mesh.face(f).len() == 4), "all quads");
        assert!(out.loss <= 6.0, "loss {}", out.loss);
        assert!(out.mesh.faces() < 24, "fewer faces than the 24 source quads: {}", out.mesh.faces());
    }

    #[test]
    fn the_same_mesh_measures_no_change() {
        let source = cube();
        assert_eq!(measure(&source, &source, 64, false), 0.0);
    }
}
