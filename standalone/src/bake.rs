// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Baking the source onto a result's UVs: a tangent-space normal map from the source's smooth normals, and the base
//! colour sampled from the source's own colour textures.
//!
//! Every result triangle is drawn in UV space. For each texel inside it: its point on the result, then the source
//! surface along the result's normal (the nearer hit of a ray each way, within a cage distance), or the nearest source
//! point when no ray hits. The source normal and UV are blended there from the hit triangle's corners.
//!
//! Tangents: per triangle from its UVs (the usual dP/du, dP/dv solve), averaged over the corners that share a vertex
//! and a UV, blended across the triangle like the normals and made square to the normal (Gram-Schmidt); the bitangent
//! is normal x tangent, flipped where the triangle's UVs are mirrored. That is close to MikkTSpace, which engines use,
//! where UVs are smooth; small differences can show along UV seams and very stretched triangles. Maps are OpenGL style
//! (green = towards +v), what glTF, Blender, Unity and Godot expect. Islands are padded by a few texels so mip-maps
//! don't pull in the empty background.

use crate::export::{triangulate, vertex_normals, Maps};
use crate::io::{Mesh, Texture};
use crate::near::{nearest_faces, FaceTree};
use crate::uv::barycentric;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;

type V3 = [f64; 3];

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
fn unit(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 0.0 { scale(a, 1.0 / l) } else { a }
}
fn point(points: &[f64], v: i64) -> V3 {
    let i = v as usize * 3;
    [points[i], points[i + 1], points[i + 2]]
}
fn blend<const N: usize>(w: V3, a: [f64; N], b: [f64; N], c: [f64; N]) -> [f64; N] {
    std::array::from_fn(|k| w[0] * a[k] + w[1] * b[k] + w[2] * c[k])
}

// ------------------------------------------------------------------------------------------------------ ray tree

/// A bounding-box tree over triangles for rays (`near::FaceTree` answers nearest-point questions only).
struct RayTree {
    nodes: Vec<(V3, V3, u32, u32)>, // box low, box high; leaf: first, count; inside: second child, 0 (first child follows)
    order: Vec<u32>,
    tris: Vec<[V3; 3]>,
}

impl RayTree {
    fn new(tris: Vec<[V3; 3]>) -> RayTree {
        let mut tree = RayTree { nodes: Vec::with_capacity(tris.len() / 2 + 1), order: (0..tris.len() as u32).collect(), tris };
        if !tree.tris.is_empty() {
            let mut order = std::mem::take(&mut tree.order);
            tree.build(&mut order, 0);
            tree.order = order;
        }
        tree
    }

    fn build(&mut self, order: &mut [u32], first: usize) {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &t in order.iter() {
            for p in &self.tris[t as usize] {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        let at = self.nodes.len();
        if order.len() <= 4 {
            self.nodes.push((lo, hi, first as u32, order.len() as u32));
            return;
        }
        self.nodes.push((lo, hi, 0, 0));
        let wide = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap();
        let centre = |t: u32| self.tris[t as usize].iter().map(|p| p[wide]).sum::<f64>();
        let mid = order.len() / 2;
        order.select_nth_unstable_by(mid, |&a, &b| centre(a).total_cmp(&centre(b)));
        let (left, right) = order.split_at_mut(mid);
        self.build(left, first);
        self.nodes[at].2 = self.nodes.len() as u32;
        self.build(right, first + mid);
    }

    /// The first triangle the ray from `o` along `d` (unit) meets within `reach`: (triangle, distance, weights).
    fn cast(&self, o: V3, d: V3, reach: f64) -> Option<(u32, f64, V3)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = d.map(|x| 1.0 / x);
        let mut best: Option<(u32, f64, V3)> = None;
        let mut stack = vec![0u32];
        while let Some(n) = stack.pop() {
            let (lo, hi, a, count) = self.nodes[n as usize];
            let limit = best.map_or(reach, |b| b.1);
            let (mut near, mut far) = (0.0f64, limit);
            for k in 0..3 {
                let (t1, t2) = ((lo[k] - o[k]) * inv[k], (hi[k] - o[k]) * inv[k]);
                near = near.max(t1.min(t2));
                far = far.min(t1.max(t2));
            }
            if near > far {
                continue;
            }
            if count == 0 {
                stack.push(a);
                stack.push(n + 1);
                continue;
            }
            for &t in &self.order[a as usize..(a + count) as usize] {
                // Moller and Trumbore's ray-triangle test.
                let [p0, p1, p2] = self.tris[t as usize];
                let (e1, e2) = (sub(p1, p0), sub(p2, p0));
                let h = cross(d, e2);
                let det = dot(e1, h);
                if det.abs() < 1e-30 {
                    continue;
                }
                let s = sub(o, p0);
                let u = dot(s, h) / det;
                let q = cross(s, e1);
                let v = dot(d, q) / det;
                let dist = dot(e2, q) / det;
                if u >= 0.0 && v >= 0.0 && u + v <= 1.0 && dist >= 0.0 && dist <= best.map_or(reach, |b| b.1) {
                    best = Some((t, dist, [1.0 - u - v, u, v]));
                }
            }
        }
        best
    }
}

// ------------------------------------------------------------------------------------------------------ images

/// The colour of `image` at `uv` (v upwards), blended between the four nearest texels; the image repeats.
fn sample(image: &image::RgbaImage, uv: [f64; 2]) -> [f64; 3] {
    let (w, h) = (image.width() as i64, image.height() as i64);
    let (x, y) = (uv[0] * w as f64 - 0.5, (1.0 - uv[1]) * h as f64 - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let texel = |i: i64, j: i64| image.get_pixel(i.rem_euclid(w) as u32, j.rem_euclid(h) as u32).0;
    let mut out = [0.0; 3];
    for (i, j, weight) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
        let t = texel(x0 as i64 + i, y0 as i64 + j);
        for k in 0..3 {
            out[k] += weight * t[k] as f64;
        }
    }
    out
}

/// Fill empty texels next to filled ones with their filled neighbours' average, `rings` times.
fn pad(pixels: &mut [[u8; 3]], filled: &mut [bool], size: usize, rings: usize) {
    for _ in 0..rings {
        let before = filled.to_vec();
        let grown: Vec<(usize, [u8; 3])> = (0..size * size).into_par_iter().filter(|&i| !before[i]).filter_map(|i| {
            let (x, y) = ((i % size) as i64, (i / size) as i64);
            let (mut sum, mut n) = ([0u32; 3], 0u32);
            for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx >= 0 && ny >= 0 && nx < size as i64 && ny < size as i64 && before[ny as usize * size + nx as usize] {
                    let p = pixels[ny as usize * size + nx as usize];
                    for k in 0..3 {
                        sum[k] += p[k] as u32;
                    }
                    n += 1;
                }
            }
            (n > 0).then(|| (i, sum.map(|s| ((s + n / 2) / n) as u8)))
        }).collect();
        if grown.is_empty() {
            break;
        }
        for (i, p) in grown {
            pixels[i] = p;
            filled[i] = true;
        }
    }
}

fn save(pixels: Vec<[u8; 3]>, size: usize, path: &Path) -> Result<(), String> {
    let raw: Vec<u8> = pixels.into_iter().flatten().collect();
    image::RgbImage::from_raw(size as u32, size as u32, raw).unwrap().save(path).map_err(|e| format!("{}: {e}", path.display()))
}

// ------------------------------------------------------------------------------------------------------ bake

/// Bake `source` onto `result`'s UVs at `size` x `size`, one image per map and per material the result uses, saved as
/// PNGs next to `out` (`<stem>_normal.png`, `<stem>_color.png`; with several materials `<stem>_<material>_normal.png`).
/// Returns each material's maps. The colour map is skipped when the source has no colour texture that loads.
pub fn bake(source: &Mesh, result: &Mesh, size: usize, normal: bool, color: bool, out: &Path, log: &mut dyn FnMut(&str)) -> Result<Vec<Maps>, String> {
    let mut maps = vec![Maps::default(); result.materials.len()];
    if result.uv.is_empty() || (!normal && !color) {
        return Ok(maps);
    }
    let size = size.clamp(16, 16384);
    let started = std::time::Instant::now();
    let src = triangulate(source);
    let src_normals = vertex_normals(source);
    let textures: Vec<Option<image::RgbaImage>> = if color && !src.uv.is_empty() {
        source.textures.iter().map(|t| t.as_ref().and_then(|t| t.load().map_err(|e| log(&format!("Colour texture not read: {e}"))).ok())).collect()
    } else {
        vec![]
    };
    let color = textures.iter().any(Option::is_some);
    if !normal && !color {
        log("Nothing to bake: the source has no colour texture that could be read.");
        return Ok(maps);
    }
    let low = triangulate(result);
    let low_normals = vertex_normals(result);

    // How far the result is from the source: rays reach a bit further than nearly all of the result's corners are.
    let tree = FaceTree::new(&src.points, &src.indptr, &src.flat);
    let (_, gaps) = nearest_faces(&tree, &result.points, &AtomicBool::new(false));
    let mut gap: Vec<f64> = gaps.iter().map(|g| g.2).filter(|d| d.is_finite()).collect();
    gap.sort_by(f64::total_cmp);
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in source.points.chunks_exact(3) {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diagonal = dot(sub(hi, lo), sub(hi, lo)).sqrt();
    let cage = 1.5 * gap.get(gap.len() * 99 / 100).copied().unwrap_or(0.0) + 0.002 * diagonal;
    let rays = RayTree::new((0..src.faces()).map(|t| [0, 1, 2].map(|c| point(&src.points, src.flat[3 * t + c]))).collect());

    // Tangent and bitangent per result triangle from its UVs, then averaged per (vertex, UV).
    let tri_frame: Vec<(V3, V3)> = (0..low.faces()).map(|t| {
        let [p0, p1, p2] = [0, 1, 2].map(|c| point(&low.points, low.flat[3 * t + c]));
        let [u0, u1, u2] = [0, 1, 2].map(|c| low.uv[3 * t + c]);
        let (e1, e2) = (sub(p1, p0), sub(p2, p0));
        let (s1, t1, s2, t2) = (u1[0] - u0[0], u1[1] - u0[1], u2[0] - u0[0], u2[1] - u0[1]);
        let r = s1 * t2 - s2 * t1;
        if r.abs() < 1e-20 {
            return ([0.0; 3], [0.0; 3]);
        }
        (unit(scale(sub(scale(e1, t2), scale(e2, t1)), 1.0 / r)), unit(scale(sub(scale(e2, s1), scale(e1, s2)), 1.0 / r)))
    }).collect();
    let mut shared: HashMap<(i64, [u64; 2]), V3> = HashMap::new();
    for c in 0..low.flat.len() {
        let e = shared.entry((low.flat[c], low.uv[c].map(f64::to_bits))).or_insert([0.0; 3]);
        *e = add(*e, tri_frame[c / 3].0);
    }
    let corner_tangent: Vec<V3> = (0..low.flat.len()).map(|c| shared[&(low.flat[c], low.uv[c].map(f64::to_bits))]).collect();

    let mut used: Vec<i64> = low.material.clone();
    used.sort_unstable();
    used.dedup();
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let outside = low.uv.iter().filter(|u| u[0] < -0.01 || u[0] > 1.01 || u[1] < -0.01 || u[1] > 1.01).count();
    if outside * 20 > low.uv.len() {
        log("Note: the UVs reach outside the 0-1 square (a repeating texture); the maps only cover the square.");
    }
    let rings = (size / 256).clamp(2, 16);
    for &m in &used {
        // Which result triangle covers each texel (texel centres inside the triangle).
        let mut owner = vec![u32::MAX; size * size];
        let s = size as f64;
        for t in (0..low.faces()).filter(|&t| low.material[t] == m) {
            let q = [0, 1, 2].map(|c| { let u = low.uv[3 * t + c]; [u[0] * s - 0.5, (1.0 - u[1]) * s - 0.5] });
            let area = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
            if area.abs() < 1e-12 {
                continue;
            }
            let x0 = q.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min).ceil().max(0.0) as usize;
            let x1 = q.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max).floor().min(s - 1.0);
            let y0 = q.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min).ceil().max(0.0) as usize;
            let y1 = q.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max).floor().min(s - 1.0);
            if x1 < 0.0 || y1 < 0.0 {
                continue;
            }
            for y in y0..=y1 as usize {
                for x in x0..=x1 as usize {
                    let (px, py) = (x as f64, y as f64);
                    let edge = |a: [f64; 2], b: [f64; 2]| ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) / area;
                    if edge(q[1], q[2]) >= -1e-9 && edge(q[2], q[0]) >= -1e-9 && edge(q[0], q[1]) >= -1e-9 {
                        owner[y * size + x] = t as u32;
                    }
                }
            }
        }
        // Each covered texel: its point and frame on the result, and what lies under it on the source.
        let texel = |i: usize| -> Option<([u8; 3], [u8; 3])> {
            let t = owner[i];
            if t == u32::MAX {
                return None;
            }
            let t = t as usize;
            let (px, py) = ((i % size) as f64, (i / size) as f64);
            let q = [0, 1, 2].map(|c| { let u = low.uv[3 * t + c]; [u[0] * s - 0.5, (1.0 - u[1]) * s - 0.5] });
            let area = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
            let edge = |a: [f64; 2], b: [f64; 2]| ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) / area;
            let w = [edge(q[1], q[2]), edge(q[2], q[0]), edge(q[0], q[1])];
            let corner = |c: usize| low.flat[3 * t + c];
            let p = blend(w, point(&low.points, corner(0)), point(&low.points, corner(1)), point(&low.points, corner(2)));
            let n = unit(blend(w, low_normals[corner(0) as usize], low_normals[corner(1) as usize], low_normals[corner(2) as usize]));
            // A hit on a source face turned the other way (the inside of a thin part) only counts when nothing else does.
            let facing = |h: &(u32, f64, V3)| {
                let [a, b, c] = rays.tris[h.0 as usize];
                dot(cross(sub(b, a), sub(c, a)), n) > 0.0
            };
            let (mut ahead, mut behind) = (rays.cast(p, n, cage), rays.cast(p, scale(n, -1.0), cage));
            if ahead.as_ref().is_some_and(facing) || behind.as_ref().is_some_and(facing) {
                ahead = ahead.filter(facing);
                behind = behind.filter(facing);
            }
            let (st, sw) = match (ahead, behind) {
                (Some(a), Some(b)) => if a.1 <= b.1 { (a.0, a.2) } else { (b.0, b.2) },
                (Some(a), None) | (None, Some(a)) => (a.0, a.2),
                (None, None) => {
                    let (at, f, _) = tree.nearest(p)?;
                    let c = [0, 1, 2].map(|k| point(&src.points, src.flat[3 * f as usize + k]));
                    (f, barycentric(at, c[0], c[1], c[2]))
                }
            };
            let st = st as usize;
            let sv = |k: usize| src.flat[3 * st + k] as usize;
            let mut normal_texel = [128, 128, 255];
            if normal {
                let source_normal = unit(blend(sw, src_normals[sv(0)], src_normals[sv(1)], src_normals[sv(2)]));
                let (tangent, bitangent) = tri_frame[t];
                let mut tg = blend(w, corner_tangent[3 * t], corner_tangent[3 * t + 1], corner_tangent[3 * t + 2]);
                tg = sub(tg, scale(n, dot(n, tg)));
                if dot(tg, tg) < 1e-24 {
                    tg = if n[0].abs() < 0.9 { cross(n, [1.0, 0.0, 0.0]) } else { cross(n, [0.0, 1.0, 0.0]) };
                }
                let tg = unit(tg);
                let flip = if dot(cross(n, tangent), bitangent) < 0.0 { -1.0 } else { 1.0 };
                let bt = scale(cross(n, tg), flip);
                let local = [dot(source_normal, tg), dot(source_normal, bt), dot(source_normal, n)];
                normal_texel = local.map(|x| ((x.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8);
            }
            let mut color_texel = [255, 255, 255];
            if color {
                let image = usize::try_from(src.material[st]).ok().and_then(|k| textures.get(k)).and_then(Option::as_ref);
                if let Some(image) = image {
                    let suv = blend(sw, src.uv[3 * st], src.uv[3 * st + 1], src.uv[3 * st + 2]);
                    color_texel = sample(image, suv).map(|x| x.round().clamp(0.0, 255.0) as u8);
                }
            }
            Some((normal_texel, color_texel))
        };
        let values: Vec<Option<([u8; 3], [u8; 3])>> = (0..size * size).into_par_iter().map(texel).collect();
        let filled: Vec<bool> = values.iter().map(Option::is_some).collect();
        let name = if used.len() > 1 {
            let raw = usize::try_from(m).ok().and_then(|k| result.materials.get(k)).cloned().unwrap_or_else(|| "material".into());
            format!("_{}", raw.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect::<String>())
        } else {
            String::new()
        };
        let slot = usize::try_from(m).ok().filter(|&k| k < maps.len());
        for (wanted, which, background) in [(normal, "normal", [128u8, 128, 255]), (color, "color", [0, 0, 0])] {
            if !wanted {
                continue;
            }
            let mut pixels: Vec<[u8; 3]> = values.iter().map(|v| v.map_or(background, |v| if which == "normal" { v.0 } else { v.1 })).collect();
            let mut mask = filled.clone();
            pad(&mut pixels, &mut mask, size, rings);
            let path = out.with_file_name(format!("{stem}{name}_{which}.png"));
            save(pixels, size, &path)?;
            if let Some(k) = slot {
                if which == "normal" {
                    maps[k].normal = Some(Texture::File(path));
                } else {
                    maps[k].color = Some(Texture::File(path));
                }
            }
        }
    }
    log(&format!("Baked {} at {size} x {size} for {} material(s) in {:.1} s (rays reach {:.4})", match (normal, color) { (true, true) => "normal and colour maps", (true, false) => "a normal map", _ => "a colour map" }, used.len(), started.elapsed().as_secs_f64(), cage));
    Ok(maps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(n: usize, height: f64) -> Mesh {
        let mut m = Mesh { indptr: vec![0], materials: vec!["m".into()], ..Default::default() };
        for j in 0..=n {
            for i in 0..=n {
                m.points.extend([i as f64 / n as f64, height, j as f64 / n as f64]);
            }
        }
        let id = |i: usize, j: usize| (j * (n + 1) + i) as i64;
        for j in 0..n {
            for i in 0..n {
                let corners = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]; // facing +y
                m.flat.extend(corners.map(|(a, b)| id(a, b)));
                m.uv.extend(corners.map(|(a, b)| [a as f64 / n as f64, 1.0 - b as f64 / n as f64]));
                m.indptr.push(m.flat.len() as i64);
                m.material.push(0);
            }
        }
        m
    }

    #[test]
    fn a_flat_plane_bakes_a_flat_normal_map() {
        let (source, result) = (plane(10, 0.0), plane(2, 0.01));
        let dir = std::env::temp_dir().join("curiomesh_test_bake");
        std::fs::create_dir_all(&dir).unwrap();
        let maps = bake(&source, &result, 32, true, false, &dir.join("flat.obj"), &mut |_| {}).unwrap();
        let Some(Texture::File(path)) = &maps[0].normal else { panic!("no normal map") };
        let image = image::open(path).unwrap().to_rgb8();
        assert_eq!(image.dimensions(), (32, 32));
        for p in image.pixels() {
            assert!(p.0.iter().zip([128i32, 128, 255]).all(|(&a, b)| (a as i32 - b).abs() <= 1), "{:?}", p.0);
        }
    }
}
