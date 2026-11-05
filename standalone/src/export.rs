// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Writing a result for a game engine: OBJ or binary glTF (.glb), triangulated if asked, cut into pieces under a
//! triangle limit, scaled and turned. The engine presets follow the add-on's `game_export.py`; textures and FBX are
//! left out here (the standalone bakes nothing, and has no well-trusted FBX writer).

use crate::io::Mesh;
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    Obj,
    Glb,
}

impl Format {
    pub fn ext(&self) -> &'static str {
        match self {
            Format::Obj => "obj",
            Format::Glb => "glb",
        }
    }
    pub fn parse(word: &str) -> Result<Format, String> {
        match word.to_ascii_lowercase().trim_start_matches('.') {
            "obj" => Ok(Format::Obj),
            "glb" | "gltf" => Ok(Format::Glb),
            other => Err(format!("{other}: the file type is obj or glb")),
        }
    }
}

/// How a result is written.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub format: Format,
    pub triangulate: bool,
    pub max_tris: usize, // 0: no limit; else meshes are cut into pieces of at most this many triangles
    pub scale: f64,
    pub z_up: bool, // OBJ only: turn Y-up into Z-up (a .glb is always Y-up, as glTF says)
}

impl Default for Target {
    fn default() -> Self {
        Target { format: Format::Obj, triangulate: false, max_tris: 0, scale: 1.0, z_up: false }
    }
}

/// (key, label, what it does, target), as the add-on's Export for Game Engine presets.
pub fn engines() -> Vec<(&'static str, &'static str, &'static str, Target)> {
    let t = |format, max_tris| Target { format, triangulate: true, max_tris, scale: 1.0, z_up: false };
    vec![
        ("unreal", "Unreal", "One .glb in triangles; Unreal 5 reads glTF and turns it to its own axes and centimetres", t(Format::Glb, 0)),
        ("unity", "Unity", "OBJ in triangles, Y up in metres, as Unity reads it", t(Format::Obj, 0)),
        ("godot", "Godot", "One .glb in triangles", t(Format::Glb, 0)),
        ("roblox", "Roblox", "OBJ in triangles, cut into pieces of at most 20,000 triangles (Roblox's limit per mesh)", t(Format::Obj, 20_000)),
        ("gltf", "Web and glTF", "One .glb in triangles for three.js, Babylon.js, Bevy, Sketchfab and other glTF readers", t(Format::Glb, 0)),
    ]
}

pub fn engine(name: &str) -> Result<Target, String> {
    let list = engines();
    let keys: Vec<&str> = list.iter().map(|e| e.0).collect();
    list.into_iter().find(|e| e.0 == name.to_ascii_lowercase()).map(|e| e.3).ok_or(format!("{name}: no such engine ({}, or custom options)", keys.join(", ")))
}

impl Target {
    pub fn to_json(&self) -> Value {
        json!({"format": self.format.ext(), "triangulate": self.triangulate, "max_tris": self.max_tris, "scale": self.scale, "up": if self.z_up { "z" } else { "y" }})
    }

    pub fn apply(&mut self, map: &serde_json::Map<String, Value>) -> Result<(), String> {
        for (k, v) in map {
            let bad = || format!("settings: export {k} has a wrong value");
            match k.as_str() {
                "format" => self.format = Format::parse(v.as_str().ok_or_else(bad)?)?,
                "triangulate" => self.triangulate = v.as_bool().ok_or_else(bad)?,
                "max_tris" => self.max_tris = v.as_u64().ok_or_else(bad)? as usize,
                "scale" => self.scale = v.as_f64().filter(|s| *s > 0.0).ok_or_else(bad)?,
                "up" => self.z_up = match v.as_str() { Some("z") | Some("Z") => true, Some("y") | Some("Y") => false, _ => return Err(bad()) },
                "engine" => *self = engine(v.as_str().ok_or_else(bad)?)?,
                other => return Err(format!("settings: unknown export key {other}")),
            }
        }
        Ok(())
    }
}

/// Every face as triangles. A quad is cut along its shorter diagonal; larger faces are fanned from their first corner.
pub fn triangulate(mesh: &Mesh) -> Mesh {
    let mut out = Mesh { points: mesh.points.clone(), indptr: vec![0], materials: mesh.materials.clone(), ..Default::default() };
    let gap = |a: i64, b: i64| (0..3).map(|c| (mesh.points[a as usize * 3 + c] - mesh.points[b as usize * 3 + c]).powi(2)).sum::<f64>();
    let push = |out: &mut Mesh, t: [i64; 3], m: i64| {
        out.flat.extend_from_slice(&t);
        out.indptr.push(out.flat.len() as i64);
        out.material.push(m);
    };
    for f in 0..mesh.faces() {
        let (face, m) = (mesh.face(f), mesh.material[f]);
        if face.len() == 4 && gap(face[1], face[3]) < gap(face[0], face[2]) {
            push(&mut out, [face[0], face[1], face[3]], m);
            push(&mut out, [face[1], face[2], face[3]], m);
            continue;
        }
        for k in 1..face.len().saturating_sub(1) {
            push(&mut out, [face[0], face[k], face[k + 1]], m);
        }
    }
    out
}

/// Per face, which piece it goes in so no piece has more than `limit` triangles and no piece mixes materials. Each
/// material's faces are cut along their longest side until they fit, then packed biggest first into the first piece
/// with room (the add-on's `texsets.split_under`).
pub fn split_under(mesh: &Mesh, limit: usize) -> Vec<usize> {
    let n = mesh.faces();
    let mut out = vec![0usize; n];
    let weight: Vec<f64> = (0..n).map(|f| mesh.face(f).len().saturating_sub(2) as f64).collect();
    let mut groups: std::collections::BTreeMap<i64, Vec<usize>> = Default::default();
    for f in 0..n {
        groups.entry(mesh.material[f]).or_default().push(f);
    }
    if limit == 0 || (weight.iter().sum::<f64>() <= limit as f64 && groups.len() <= 1) {
        return out;
    }
    let centre = |f: usize| {
        let face = mesh.face(f);
        let mut c = [0.0; 3];
        for &v in face {
            for k in 0..3 {
                c[k] += mesh.points[v as usize * 3 + k] / face.len() as f64;
            }
        }
        c
    };
    let centres: Vec<[f64; 3]> = (0..n).map(centre).collect();
    let most = limit as f64;
    let mut made = 0;
    for group in groups.into_values() {
        let mut pieces = Vec::new();
        let mut todo = vec![group];
        while let Some(mut g) = todo.pop() {
            if g.iter().map(|&f| weight[f]).sum::<f64>() <= most || g.len() < 2 {
                pieces.push(g);
                continue;
            }
            let axis = (0..3).max_by(|&a, &b| {
                let span = |k: usize| { let (lo, hi) = g.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &f| (lo.min(centres[f][k]), hi.max(centres[f][k]))); hi - lo };
                span(a).total_cmp(&span(b))
            }).unwrap();
            g.sort_by(|&a, &b| centres[a][axis].total_cmp(&centres[b][axis]));
            let (mut sum, mut half) = (0.0, 0);
            while half < g.len() && sum + weight[g[half]] <= most {
                sum += weight[g[half]];
                half += 1;
            }
            let half = half.clamp(1, g.len() - 1);
            let rest = g.split_off(half);
            todo.push(g);
            todo.push(rest);
        }
        let sum = |p: &Vec<usize>| p.iter().map(|&f| weight[f]).sum::<f64>();
        pieces.sort_by(|a, b| sum(b).total_cmp(&sum(a)));
        let mut room: Vec<f64> = Vec::new();
        for p in &pieces {
            let need = sum(p);
            let k = room.iter().position(|&r| r >= need).unwrap_or_else(|| {
                room.push(most);
                room.len() - 1
            });
            room[k] -= need;
            for &f in p {
                out[f] = made + k;
            }
        }
        made += room.len();
    }
    out
}

/// Write `mesh` to `path` as `target` says. Returns how many pieces were written.
pub fn write(mesh: &Mesh, path: &Path, target: &Target) -> Result<usize, String> {
    let mut mesh = if target.triangulate { triangulate(mesh) } else { mesh.clone() };
    for p in mesh.points.chunks_exact_mut(3) {
        let [x, y, z] = [p[0], p[1], p[2]].map(|c| c * target.scale);
        // Y up to Z up, keeping the model's front: (x, y, z) -> (x, -z, y).
        p.copy_from_slice(&if target.z_up && target.format == Format::Obj { [x, -z, y] } else { [x, y, z] });
    }
    let piece = split_under(&mesh, target.max_tris);
    let pieces = piece.iter().max().map_or(0, |m| m + 1);
    match target.format {
        Format::Obj => crate::io::write_obj_pieces(&mesh, &piece, path)?,
        Format::Glb => write_glb(&mesh, &piece, pieces, path)?,
    }
    Ok(pieces.max(1))
}

/// A binary glTF 2.0 file: one node and mesh per piece, positions and triangle indices, one named material per piece.
fn write_glb(mesh: &Mesh, piece: &[usize], pieces: usize, path: &Path) -> Result<(), String> {
    let tris = triangulate(mesh); // glTF holds triangles; a no-op when already triangulated
    let mut bin: Vec<u8> = Vec::new();
    let (mut nodes, mut meshes, mut accessors, mut views, mut materials) = (vec![], vec![], vec![], vec![], vec![]);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh");
    // tris keeps the source face order, so each triangle's piece is its face's piece.
    let mut face_of = Vec::with_capacity(tris.faces());
    for f in 0..mesh.faces() {
        face_of.extend(std::iter::repeat(f).take(mesh.face(f).len().saturating_sub(2)));
    }
    for k in 0..pieces.max(1) {
        let mut number = vec![u32::MAX; tris.points.len() / 3];
        let (mut positions, mut indices): (Vec<f32>, Vec<u32>) = (vec![], vec![]);
        let mut material = -1;
        for t in 0..tris.faces() {
            if piece.get(face_of[t]).copied().unwrap_or(0) != k {
                continue;
            }
            material = tris.material[t];
            for &v in tris.face(t) {
                if number[v as usize] == u32::MAX {
                    number[v as usize] = (positions.len() / 3) as u32;
                    positions.extend(tris.points[v as usize * 3..v as usize * 3 + 3].iter().map(|&c| c as f32));
                }
                indices.push(number[v as usize]);
            }
        }
        if indices.is_empty() {
            continue;
        }
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in positions.chunks_exact(3) {
            for c in 0..3 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
        let start = bin.len();
        bin.extend(positions.iter().flat_map(|c| c.to_le_bytes()));
        let middle = bin.len();
        bin.extend(indices.iter().flat_map(|i| i.to_le_bytes()));
        let (pos_view, idx_view) = (views.len(), views.len() + 1);
        views.push(json!({"buffer": 0, "byteOffset": start, "byteLength": middle - start, "target": 34962}));
        views.push(json!({"buffer": 0, "byteOffset": middle, "byteLength": bin.len() - middle, "target": 34963}));
        accessors.push(json!({"bufferView": pos_view, "componentType": 5126, "count": positions.len() / 3, "type": "VEC3", "min": lo, "max": hi}));
        accessors.push(json!({"bufferView": idx_view, "componentType": 5125, "count": indices.len(), "type": "SCALAR"}));
        let name = if pieces > 1 { format!("{stem}_{:02}", k + 1) } else { stem.to_string() };
        let material_name = usize::try_from(material).ok().and_then(|m| mesh.materials.get(m).cloned()).unwrap_or_else(|| "material".into());
        materials.push(json!({"name": material_name, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.8}}));
        meshes.push(json!({"name": name, "primitives": [{"attributes": {"POSITION": accessors.len() - 2}, "indices": accessors.len() - 1, "material": materials.len() - 1, "mode": 4}]}));
        nodes.push(json!({"name": name, "mesh": meshes.len() - 1}));
    }
    let doc = json!({
        "asset": {"version": "2.0", "generator": concat!("CurioMesh standalone ", env!("CARGO_PKG_VERSION"))},
        "scene": 0,
        "scenes": [{"nodes": (0..nodes.len()).collect::<Vec<_>>()}],
        "nodes": nodes, "meshes": meshes, "materials": materials, "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": bin.len()}],
    });
    let mut text = serde_json::to_vec(&doc).unwrap();
    while text.len() % 4 != 0 {
        text.push(b' ');
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let total = 12 + 8 + text.len() + 8 + bin.len();
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?);
    let mut put = |bytes: &[u8]| file.write_all(bytes).map_err(|e| format!("{}: {e}", path.display()));
    put(b"glTF")?;
    put(&2u32.to_le_bytes())?;
    put(&(total as u32).to_le_bytes())?;
    put(&(text.len() as u32).to_le_bytes())?;
    put(b"JSON")?;
    put(&text)?;
    put(&(bin.len() as u32).to_le_bytes())?;
    put(b"BIN\0")?;
    put(&bin)?;
    file.flush().map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Mesh {
        let mut m = Mesh { indptr: vec![0], ..Default::default() };
        for j in 0..=n {
            for i in 0..=n {
                m.points.extend([i as f64, 0.0, j as f64]);
            }
        }
        let at = |i: usize, j: usize| (j * (n + 1) + i) as i64;
        for j in 0..n {
            for i in 0..n {
                m.flat.extend([at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
                m.indptr.push(m.flat.len() as i64);
                m.material.push(0);
            }
        }
        m
    }

    #[test]
    fn split_keeps_every_piece_under_the_limit() {
        let tris = triangulate(&grid(20)); // 800 triangles
        let piece = split_under(&tris, 150);
        let pieces = piece.iter().max().unwrap() + 1;
        assert!(pieces >= 6);
        for k in 0..pieces {
            assert!(piece.iter().filter(|&&p| p == k).count() <= 150);
        }
    }

    #[test]
    fn glb_reads_back_with_the_gltf_reader() {
        let path = std::env::temp_dir().join("curiomesh_test_export.glb");
        let target = Target { format: Format::Glb, triangulate: true, max_tris: 50, scale: 2.0, z_up: false };
        assert_eq!(write(&grid(6), &path, &target).unwrap(), 2);
        let back = crate::io::read(&path).unwrap();
        assert_eq!(back.faces(), 72);
        assert!(back.points.chunks_exact(3).any(|p| (p[0] - 12.0).abs() < 1e-6));
    }
}
