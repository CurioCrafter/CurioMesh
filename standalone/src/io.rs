// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Reading OBJ (tobj), glTF/GLB (gltf) and PLY (own reader, ASCII and binary), and writing OBJ.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Polygons as CSR (`indptr`, `flat`), one material index per face into `materials`.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub points: Vec<f64>,
    pub indptr: Vec<i64>,
    pub flat: Vec<i64>,
    pub material: Vec<i64>,
    pub materials: Vec<String>,
}

impl Mesh {
    pub fn faces(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }
    pub fn face(&self, f: usize) -> &[i64] {
        &self.flat[self.indptr[f] as usize..self.indptr[f + 1] as usize]
    }
    fn push_face(&mut self, face: impl IntoIterator<Item = i64>, material: i64) {
        self.flat.extend(face);
        self.indptr.push(self.flat.len() as i64);
        self.material.push(material);
    }
    fn new() -> Mesh {
        Mesh { indptr: vec![0], ..Default::default() }
    }
}

pub fn read(path: &Path) -> Result<Mesh, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let mesh = match ext.as_str() {
        "obj" => read_obj(path)?,
        "gltf" | "glb" => read_gltf(path)?,
        "ply" => read_ply(path)?,
        _ => return Err(format!("{}: only .obj, .gltf, .glb and .ply can be read", path.display())),
    };
    let nv = (mesh.points.len() / 3) as i64;
    if mesh.flat.iter().any(|&v| v < 0 || v >= nv) {
        return Err(format!("{}: a face uses a vertex that isn't there", path.display()));
    }
    if mesh.faces() == 0 {
        return Err(format!("{}: no faces", path.display()));
    }
    Ok(mesh)
}

fn read_obj(path: &Path) -> Result<Mesh, String> {
    let options = tobj::LoadOptions { single_index: false, triangulate: false, ignore_points: true, ignore_lines: true };
    let (models, materials) = tobj::load_obj(path, &options).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut mesh = Mesh::new();
    mesh.materials = materials.map(|m| m.into_iter().map(|m| m.name).collect()).unwrap_or_default();
    for model in models {
        let m = &model.mesh;
        let offset = (mesh.points.len() / 3) as i64;
        mesh.points.extend(m.positions.iter().map(|&x| x as f64));
        let material = m.material_id.map_or(-1, |k| k as i64);
        let mut at = 0usize;
        let count = if m.face_arities.is_empty() { m.indices.len() / 3 } else { m.face_arities.len() };
        for f in 0..count {
            let n = if m.face_arities.is_empty() { 3 } else { m.face_arities[f] as usize };
            mesh.push_face(m.indices[at..at + n].iter().map(|&v| v as i64 + offset), material);
            at += n;
        }
    }
    Ok(mesh)
}

fn read_gltf(path: &Path) -> Result<Mesh, String> {
    let err = |e: gltf::Error| format!("{}: {e}", path.display());
    let (doc, buffers, _images) = gltf::import(path).map_err(err)?;
    let mut mesh = Mesh::new();
    mesh.materials = doc.materials().map(|m| m.name().map_or_else(|| format!("material{}", m.index().unwrap_or(0)), str::to_string)).collect();
    let scene = doc.default_scene().or_else(|| doc.scenes().next()).ok_or_else(|| format!("{}: no scene", path.display()))?;
    let mut stack: Vec<(gltf::Node, [[f32; 4]; 4])> = scene.nodes().map(|n| (n, IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let world = mul(parent, node.transform().matrix());
        for child in node.children() {
            stack.push((child, world));
        }
        let Some(m) = node.mesh() else { continue };
        for prim in m.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let Some(positions) = reader.read_positions() else { continue };
            let offset = (mesh.points.len() / 3) as i64;
            let mut count = 0i64;
            for p in positions {
                let w = |r: usize| world[0][r] * p[0] + world[1][r] * p[1] + world[2][r] * p[2] + world[3][r];
                mesh.points.extend([w(0) as f64, w(1) as f64, w(2) as f64]);
                count += 1;
            }
            let indices: Vec<i64> = match reader.read_indices() {
                Some(i) => i.into_u32().map(|v| v as i64).collect(),
                None => (0..count).collect(),
            };
            let material = prim.material().index().map_or(-1, |k| k as i64);
            for tri in indices.chunks_exact(3) {
                mesh.push_face(tri.iter().map(|&v| v + offset), material);
            }
        }
    }
    Ok(mesh)
}

const IDENTITY: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// Column-major 4x4 product a * b, as glTF stores matrices.
fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (c, col) in out.iter_mut().enumerate() {
        for (r, cell) in col.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------------------ PLY

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Ascii,
    Little,
    Big,
}

struct Property {
    name: String,
    kind: String,
    list: Option<String>, // the count's type, for a list property
}

struct Element {
    name: String,
    count: usize,
    props: Vec<Property>,
}

fn size_of(kind: &str) -> Result<usize, String> {
    Ok(match kind {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "int" | "uint" | "int32" | "uint32" | "float" | "float32" => 4,
        "double" | "float64" => 8,
        _ => return Err(format!("PLY: unknown type {kind}")),
    })
}

fn binary_value(bytes: &[u8], kind: &str, big: bool) -> f64 {
    macro_rules! num {
        ($t:ty) => {{
            let mut raw = [0u8; std::mem::size_of::<$t>()];
            raw.copy_from_slice(&bytes[..std::mem::size_of::<$t>()]);
            (if big { <$t>::from_be_bytes(raw) } else { <$t>::from_le_bytes(raw) }) as f64
        }};
    }
    match kind {
        "char" | "int8" => num!(i8),
        "uchar" | "uint8" => num!(u8),
        "short" | "int16" => num!(i16),
        "ushort" | "uint16" => num!(u16),
        "int" | "int32" => num!(i32),
        "uint" | "uint32" => num!(u32),
        "float" | "float32" => num!(f32),
        _ => num!(f64),
    }
}

fn read_ply(path: &Path) -> Result<Mesh, String> {
    let bad = |what: &str| format!("{}: {what}", path.display());
    let mut input = BufReader::new(fs::File::open(path).map_err(|e| bad(&e.to_string()))?);
    let mut line = String::new();
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    loop {
        line.clear();
        if input.read_line(&mut line).map_err(|e| bad(&e.to_string()))? == 0 {
            return Err(bad("PLY header has no end_header"));
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["format", f, ..] => {
                format = Some(match *f {
                    "ascii" => Format::Ascii,
                    "binary_little_endian" => Format::Little,
                    "binary_big_endian" => Format::Big,
                    _ => return Err(bad("unknown PLY format")),
                })
            }
            ["element", name, count] => elements.push(Element { name: name.to_string(), count: count.parse().map_err(|_| bad("bad element count"))?, props: vec![] }),
            ["property", "list", count, item, name] => {
                let e = elements.last_mut().ok_or_else(|| bad("property before element"))?;
                e.props.push(Property { name: name.to_string(), kind: item.to_string(), list: Some(count.to_string()) });
            }
            ["property", kind, name] => {
                let e = elements.last_mut().ok_or_else(|| bad("property before element"))?;
                e.props.push(Property { name: name.to_string(), kind: kind.to_string(), list: None });
            }
            ["end_header"] => break,
            _ => {}
        }
    }
    let format = format.ok_or_else(|| bad("PLY has no format line"))?;
    let mut rest = Vec::new();
    input.read_to_end(&mut rest).map_err(|e| bad(&e.to_string()))?;
    let mut words = std::str::from_utf8(if format == Format::Ascii { &rest } else { &[] }).map_err(|_| bad("PLY text isn't UTF-8"))?.split_ascii_whitespace();
    let mut at = 0usize;
    let mut next = |kind: &str| -> Result<f64, String> {
        if format == Format::Ascii {
            return words.next().and_then(|w| w.parse().ok()).ok_or_else(|| bad("PLY data ends early or isn't a number"));
        }
        let n = size_of(kind)?;
        let bytes = rest.get(at..at + n).ok_or_else(|| bad("PLY data ends early"))?;
        at += n;
        Ok(binary_value(bytes, kind, format == Format::Big))
    };
    let mut mesh = Mesh::new();
    for e in &elements {
        let xyz: Vec<Option<usize>> = ["x", "y", "z"].iter().map(|c| e.props.iter().position(|p| p.name == *c)).collect();
        for _ in 0..e.count {
            let mut point = [0.0; 3];
            for (k, p) in e.props.iter().enumerate() {
                match &p.list {
                    Some(count_kind) => {
                        let n = next(count_kind)? as usize;
                        let mut items = Vec::with_capacity(n);
                        for _ in 0..n {
                            items.push(next(&p.kind)? as i64);
                        }
                        if e.name == "face" && (p.name == "vertex_indices" || p.name == "vertex_index") && n >= 3 {
                            mesh.push_face(items, -1);
                        }
                    }
                    None => {
                        let value = next(&p.kind)?;
                        if e.name == "vertex" {
                            if let Some(c) = xyz.iter().position(|&i| i == Some(k)) {
                                point[c] = value;
                            }
                        }
                    }
                }
            }
            if e.name == "vertex" {
                mesh.points.extend(point);
            }
        }
    }
    Ok(mesh)
}

// ------------------------------------------------------------------------------------------------------------ OBJ out

pub fn write_obj(mesh: &Mesh, path: &Path) -> Result<(), String> {
    let file = fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = BufWriter::new(file);
    let w = |e: std::io::Error| e.to_string();
    writeln!(out, "# CurioMesh standalone").map_err(w)?;
    for p in mesh.points.chunks_exact(3) {
        writeln!(out, "v {} {} {}", p[0], p[1], p[2]).map_err(w)?;
    }
    let mut last = i64::MIN;
    for f in 0..mesh.faces() {
        let m = mesh.material[f];
        if m != last && m >= 0 {
            let name = mesh.materials.get(m as usize).cloned().unwrap_or_else(|| format!("material{m}"));
            writeln!(out, "usemtl {name}").map_err(w)?;
        }
        last = m;
        write!(out, "f").map_err(w)?;
        for v in mesh.face(f) {
            write!(out, " {}", v + 1).map_err(w)?;
        }
        writeln!(out).map_err(w)?;
    }
    out.flush().map_err(w)
}

/// Vertices at exactly the same place become one (glTF and some OBJ exports split them at UV seams and hard edges,
/// which would cut the surface into loose pieces). Unused vertices are dropped.
pub fn weld(mesh: &Mesh) -> Mesh {
    let mut index: HashMap<[u64; 3], i64> = HashMap::with_capacity(mesh.points.len() / 3);
    let mut points = Vec::with_capacity(mesh.points.len());
    let mut remap = vec![-1i64; mesh.points.len() / 3];
    let mut used = vec![false; remap.len()];
    for &v in &mesh.flat {
        used[v as usize] = true;
    }
    for (v, p) in mesh.points.chunks_exact(3).enumerate() {
        if !used[v] {
            continue;
        }
        let key = [p[0] + 0.0, p[1] + 0.0, p[2] + 0.0].map(f64::to_bits); // + 0.0 makes -0.0 equal 0.0
        remap[v] = *index.entry(key).or_insert_with(|| {
            points.extend_from_slice(p);
            (points.len() / 3 - 1) as i64
        });
    }
    Mesh { points, flat: mesh.flat.iter().map(|&v| remap[v as usize]).collect(), ..mesh.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ply_ascii_and_binary_read_the_same_quad() {
        let dir = std::env::temp_dir();
        let ascii = dir.join("curiomesh_test_ascii.ply");
        fs::write(&ascii, "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n").unwrap();
        let mut bin = b"ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n".to_vec();
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]] {
            for x in p {
                bin.extend(x.to_le_bytes());
            }
        }
        bin.push(4);
        for v in 0..4i32 {
            bin.extend(v.to_le_bytes());
        }
        let binary = dir.join("curiomesh_test_binary.ply");
        fs::write(&binary, bin).unwrap();
        let (a, b) = (read(&ascii).unwrap(), read(&binary).unwrap());
        assert_eq!(a.points, b.points);
        assert_eq!(a.flat, vec![0, 1, 2, 3]);
        assert_eq!(b.flat, vec![0, 1, 2, 3]);
    }

    #[test]
    fn weld_joins_split_vertices() {
        let mesh = Mesh { points: vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.], indptr: vec![0, 3, 6], flat: vec![0, 1, 2, 3, 5, 4], material: vec![0, 0], materials: vec![] };
        let welded = weld(&mesh);
        assert_eq!(welded.points.len(), 12);
        assert_eq!(welded.flat, vec![0, 1, 2, 1, 3, 2]);
    }
}
