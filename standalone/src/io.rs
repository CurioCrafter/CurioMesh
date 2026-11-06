// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Reading OBJ (tobj), glTF/GLB (gltf) and PLY (own reader, ASCII and binary), and writing OBJ (`export.rs` writes .glb).

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Polygons as CSR (`indptr`, `flat`), one material index per face into `materials`. UVs are per corner (one per entry
/// of `flat`, OBJ's way up: v grows upwards) so a vertex on a UV seam can have one UV per face; empty when there are none.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub points: Vec<f64>,
    pub indptr: Vec<i64>,
    pub flat: Vec<i64>,
    pub material: Vec<i64>,
    pub materials: Vec<String>,
    pub uv: Vec<[f64; 2]>,
    pub textures: Vec<Option<Texture>>, // per material, its colour image when the file names one
}

/// Where a material's colour image is: a file, or bytes inside a .glb.
#[derive(Clone)]
pub enum Texture {
    File(std::path::PathBuf),
    Bytes(std::sync::Arc<Vec<u8>>),
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Texture::File(p) => write!(f, "File({})", p.display()),
            Texture::Bytes(b) => write!(f, "Bytes({} bytes)", b.len()),
        }
    }
}

impl Texture {
    pub fn load(&self) -> Result<image::RgbaImage, String> {
        match self {
            Texture::File(p) => image::open(p).map(|i| i.to_rgba8()).map_err(|e| format!("{}: {e}", p.display())),
            Texture::Bytes(b) => image::load_from_memory(b).map(|i| i.to_rgba8()).map_err(|e| format!("texture in the model: {e}")),
        }
    }
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
    if !mesh.uv.is_empty() && mesh.uv.len() != mesh.flat.len() {
        return Err(format!("{}: UVs don't match the faces", path.display()));
    }
    Ok(mesh)
}

fn read_obj(path: &Path) -> Result<Mesh, String> {
    let options = tobj::LoadOptions { single_index: false, triangulate: false, ignore_points: true, ignore_lines: true };
    let (models, materials) = tobj::load_obj(path, &options).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut mesh = Mesh::new();
    let folder = path.parent().unwrap_or(Path::new("."));
    for m in materials.unwrap_or_default() {
        mesh.textures.push(m.diffuse_texture.filter(|t| !t.trim().is_empty()).map(|t| Texture::File(folder.join(t.trim()))));
        mesh.materials.push(m.name);
    }
    let mut every_uv = true;
    for model in models {
        let m = &model.mesh;
        let offset = (mesh.points.len() / 3) as i64;
        mesh.points.extend(m.positions.iter().map(|&x| x as f64));
        let material = m.material_id.map_or(-1, |k| k as i64);
        let mut at = 0usize;
        let count = if m.face_arities.is_empty() { m.indices.len() / 3 } else { m.face_arities.len() };
        let has_uv = m.texcoord_indices.len() == m.indices.len() && !m.texcoords.is_empty();
        every_uv &= has_uv;
        if has_uv {
            mesh.uv.extend(m.texcoord_indices.iter().map(|&t| [m.texcoords[2 * t as usize] as f64, m.texcoords[2 * t as usize + 1] as f64]));
        }
        for f in 0..count {
            let n = if m.face_arities.is_empty() { 3 } else { m.face_arities[f] as usize };
            mesh.push_face(m.indices[at..at + n].iter().map(|&v| v as i64 + offset), material);
            at += n;
        }
    }
    if !every_uv {
        mesh.uv.clear(); // UVs on some parts only: treated as none
    }
    Ok(mesh)
}

fn read_gltf(path: &Path) -> Result<Mesh, String> {
    let err = |e: gltf::Error| format!("{}: {e}", path.display());
    // Textures are only located here (a bake loads them), so a model whose texture files are missing still opens.
    let gltf::Gltf { document: doc, blob } = gltf::Gltf::open(path).map_err(err)?;
    let buffers = gltf::import_buffers(&doc, path.parent(), blob).map_err(err)?;
    let mut mesh = Mesh::new();
    mesh.materials = doc.materials().map(|m| m.name().map_or_else(|| format!("material{}", m.index().unwrap_or(0)), str::to_string)).collect();
    let folder = path.parent().unwrap_or(Path::new("."));
    mesh.textures = doc.materials().map(|m| {
        let image = m.pbr_metallic_roughness().base_color_texture()?.texture().source();
        match image.source() {
            gltf::image::Source::Uri { uri, .. } if !uri.starts_with("data:") => Some(Texture::File(folder.join(uri.replace("%20", " ")))),
            gltf::image::Source::View { view, .. } => {
                let bytes = buffers[view.buffer().index()].get(view.offset()..view.offset() + view.length())?;
                Some(Texture::Bytes(std::sync::Arc::new(bytes.to_vec())))
            }
            _ => None, // ponytail: data: URIs are skipped; decode their base64 here if such files turn up
        }
    }).collect();
    let mut every_uv = true;
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
            // glTF's v grows downwards; here it grows upwards, as in OBJ.
            let uvs: Option<Vec<[f64; 2]>> = reader.read_tex_coords(0).map(|t| t.into_f32().map(|[u, v]| [u as f64, 1.0 - v as f64]).collect());
            let uvs = uvs.filter(|t| t.len() == count as usize);
            every_uv &= uvs.is_some();
            for tri in indices.chunks_exact(3) {
                mesh.push_face(tri.iter().map(|&v| v + offset), material);
                if let Some(t) = &uvs {
                    mesh.uv.extend(tri.iter().map(|&v| t[v as usize]));
                }
            }
        }
    }
    if !every_uv {
        mesh.uv.clear();
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
    let mut vertex_uv: Vec<[f64; 2]> = Vec::new();
    for e in &elements {
        // x, y, z, then u, v when the vertices have them (as u v, s t or texture_u texture_v).
        let mut xyz: Vec<Option<usize>> = ["x", "y", "z"].iter().map(|c| e.props.iter().position(|p| p.name == *c)).collect();
        let named = |a: &str, b: &str| e.props.iter().position(|p| p.name == a).zip(e.props.iter().position(|p| p.name == b));
        if let (true, Some((u, v))) = (e.name == "vertex", named("u", "v").or_else(|| named("s", "t")).or_else(|| named("texture_u", "texture_v"))) {
            xyz.extend([Some(u), Some(v)]);
        }
        for _ in 0..e.count {
            let mut point = [0.0; 5];
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
                mesh.points.extend(&point[..3]);
                if xyz.len() == 5 {
                    vertex_uv.push([point[3], point[4]]);
                }
            }
        }
    }
    if !vertex_uv.is_empty() && vertex_uv.len() == mesh.points.len() / 3 {
        mesh.uv = mesh.flat.iter().map(|&v| vertex_uv.get(v as usize).copied().unwrap_or_default()).collect();
    }
    Ok(mesh)
}

// ------------------------------------------------------------------------------------------------------------ OBJ out

pub fn write_obj(mesh: &Mesh, path: &Path) -> Result<(), String> {
    write_obj_pieces(mesh, &[], path, &[])
}

/// OBJ with each face in the object `piece[face]` (`o name_01`, ...). An empty or all-zero `piece` writes one object.
/// With UVs it also writes `vt` and smooth `vn` (`f v/vt/vn`), and with `maps` a .mtl next to it that names each
/// material's colour (`map_Kd`) and normal map (`norm` and `map_Bump`).
pub fn write_obj_pieces(mesh: &Mesh, piece: &[usize], path: &Path, maps: &[crate::export::Maps]) -> Result<(), String> {
    let pieces = piece.iter().max().map_or(1, |m| m + 1);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh");
    let w = |e: std::io::Error| e.to_string();
    let has_uv = !mesh.uv.is_empty();
    let maps_used = has_uv && maps.iter().any(|m| m.color.is_some() || m.normal.is_some());
    if maps_used {
        let folder = path.parent().unwrap_or(Path::new(""));
        // A file next to the OBJ by its name, anything else by its full path; images inside a .glb can't be named.
        let name = |t: &Option<Texture>| match t {
            Some(Texture::File(p)) if p.parent() == Some(folder) => p.file_name().map(|n| n.to_string_lossy().to_string()),
            Some(Texture::File(p)) => Some(std::path::absolute(p).unwrap_or(p.clone()).display().to_string()),
            _ => None,
        };
        let mut text = String::from("# CurioMesh standalone\n");
        for (m, material) in mesh.materials.iter().enumerate() {
            text += &format!("\nnewmtl {material}\nKd 1 1 1\n");
            let found = maps.get(m).cloned().unwrap_or_default();
            if let Some(file) = name(&found.color) {
                text += &format!("map_Kd {file}\n");
            }
            if let Some(file) = name(&found.normal) {
                text += &format!("norm {file}\nmap_Bump {file}\n");
            }
        }
        let mtl = path.with_extension("mtl");
        fs::write(&mtl, text).map_err(|e| format!("{}: {e}", mtl.display()))?;
    }
    let file = fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "# CurioMesh standalone").map_err(w)?;
    if maps_used {
        writeln!(out, "mtllib {stem}.mtl").map_err(w)?;
    }
    for p in mesh.points.chunks_exact(3) {
        writeln!(out, "v {} {} {}", p[0], p[1], p[2]).map_err(w)?;
    }
    // One vt per different UV, numbered per corner.
    let mut uv_number: Vec<usize> = Vec::with_capacity(mesh.uv.len());
    if has_uv {
        let mut seen: HashMap<[u64; 2], usize> = HashMap::new();
        for uv in &mesh.uv {
            let next = seen.len() + 1;
            let n = *seen.entry(uv.map(f64::to_bits)).or_insert_with(|| {
                let _ = writeln!(out, "vt {} {}", uv[0], uv[1]);
                next
            });
            uv_number.push(n);
        }
        for n in crate::export::vertex_normals(mesh) {
            writeln!(out, "vn {:.6} {:.6} {:.6}", n[0], n[1], n[2]).map_err(w)?;
        }
    }
    for k in 0..pieces {
        if pieces > 1 {
            writeln!(out, "o {stem}_{:02}", k + 1).map_err(w)?;
        }
        let mut last = i64::MIN;
        for f in (0..mesh.faces()).filter(|&f| piece.get(f).copied().unwrap_or(0) == k) {
            let m = mesh.material[f];
            if m != last && m >= 0 {
                let name = mesh.materials.get(m as usize).cloned().unwrap_or_else(|| format!("material{m}"));
                writeln!(out, "usemtl {name}").map_err(w)?;
            }
            last = m;
            write!(out, "f").map_err(w)?;
            for c in mesh.indptr[f] as usize..mesh.indptr[f + 1] as usize {
                let v = mesh.flat[c] + 1;
                if has_uv {
                    write!(out, " {v}/{}/{v}", uv_number[c]).map_err(w)?;
                } else {
                    write!(out, " {v}").map_err(w)?;
                }
            }
            writeln!(out).map_err(w)?;
        }
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

/// The same mesh without unused vertices.
pub fn compact(mesh: &Mesh) -> Mesh {
    let mut number = vec![-1i64; mesh.points.len() / 3];
    let mut points = Vec::new();
    let flat = mesh.flat.iter().map(|&v| {
        if number[v as usize] < 0 {
            number[v as usize] = (points.len() / 3) as i64;
            points.extend_from_slice(&mesh.points[v as usize * 3..v as usize * 3 + 3]);
        }
        number[v as usize]
    }).collect();
    Mesh { points, flat, ..mesh.clone() }
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
    fn obj_uvs_are_per_corner_and_survive_weld() {
        let dir = std::env::temp_dir().join("curiomesh_test_uv_read");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("m.mtl"), "newmtl skin\nmap_Kd tex/skin.png\n").unwrap();
        // Two triangles sharing an edge whose UVs differ on each side (a seam), and vertex 4 a copy of vertex 2.
        fs::write(dir.join("m.obj"), "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvt 0.5 0.5\nvt 0.75 0.75\nusemtl skin\nf 1/1 2/2 3/3\nf 2/4 4/5 5/3\n").unwrap();
        let mesh = weld(&read(&dir.join("m.obj")).unwrap());
        assert_eq!(mesh.flat, vec![0, 1, 2, 1, 3, 2]);
        assert_eq!(mesh.uv, vec![[0., 0.], [1., 0.], [0., 1.], [0.5, 0.5], [0.75, 0.75], [0., 1.]]);
        assert!(matches!(&mesh.textures[0], Some(Texture::File(p)) if p.ends_with("tex/skin.png")));
    }

    #[test]
    fn weld_joins_split_vertices() {
        let mesh = Mesh { points: vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.], indptr: vec![0, 3, 6], flat: vec![0, 1, 2, 3, 5, 4], material: vec![0, 0], ..Default::default() };
        let welded = weld(&mesh);
        assert_eq!(welded.points.len(), 12);
        assert_eq!(welded.flat, vec![0, 1, 2, 1, 3, 2]);
    }
}
