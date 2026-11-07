// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! CurioMesh Turbo: native kernels for the CurioMesh add-on. The Python side
//! (`turbo.py`) converts meshes to flat arrays and falls back to pure Python
//! whenever this module is missing. All heavy work runs with the GIL released.

mod chords;
mod close;
mod field;
mod geom;
mod gpu;
mod join;
mod matching;
mod near;
mod pairing;
mod qem;
mod quadreduce;
mod segment;
mod topo;

use numpy::{Complex64, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

type I64Array<'py> = Bound<'py, PyArray1<i64>>;
type F64Array<'py> = Bound<'py, PyArray1<f64>>;

fn slice<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

fn check_faces(indptr: &[i64], flat: &[i64]) -> PyResult<()> {
    let ok = !indptr.is_empty()
        && indptr[0] == 0
        && *indptr.last().unwrap() as usize == flat.len()
        && indptr.windows(2).all(|w| w[0] <= w[1]);
    if ok {
        Ok(())
    } else {
        Err(PyValueError::new_err("invalid face CSR (indptr)"))
    }
}

/// Combinatorial topology summary (see `topo::summary`); vertex ids must be >= 0.
#[pyfunction]
fn topology_counts<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (indptr, flat) = (slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    check_faces(indptr, flat)?;
    if flat.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err("negative vertex index"));
    }
    let s = py.detach(|| topo::summary(&topo::Faces { indptr, flat }));
    let d = PyDict::new(py);
    d.set_item("used_vertices", s.used_vertices)?;
    d.set_item("edges", s.edges)?;
    d.set_item("faces", s.faces)?;
    d.set_item("quads", s.quads)?;
    d.set_item("boundary_edges", s.boundary_edges)?;
    d.set_item("boundary_components", s.boundary_components)?;
    d.set_item("branched_boundary_vertices", s.branched_boundary_vertices)?;
    d.set_item("non_manifold_edges", s.non_manifold_edges)?;
    d.set_item("non_manifold_vertices", s.non_manifold_vertices)?;
    d.set_item("inconsistent_edges", s.inconsistent_edges)?;
    d.set_item("duplicate_faces", s.duplicate_faces)?;
    d.set_item("euler_characteristic", s.euler_characteristic)?;
    d.set_item("extraordinary_vertices", s.extraordinary_vertices)?;
    d.set_item("defect_charge", s.defect_charge)?;
    d.set_item("component_detail", s.component_detail)?;
    Ok(d)
}

/// Faces whose fan area is <= epsilon (`topology._face_areas(...) <= eps`).
#[pyfunction]
fn degenerate_faces(
    py: Python<'_>,
    indptr: PyReadonlyArray1<'_, i64>,
    flat: PyReadonlyArray1<'_, i64>,
    vertices: PyReadonlyArray1<'_, f64>,
    epsilon: f64,
) -> PyResult<i64> {
    let (indptr, flat, vertices) = (
        slice(&indptr, "indptr")?,
        slice(&flat, "flat")?,
        slice(&vertices, "vertices")?,
    );
    check_faces(indptr, flat)?;
    let nv = (vertices.len() / 3) as i64;
    if flat.iter().any(|&v| v < 0 || v >= nv) {
        return Err(PyValueError::new_err("vertex index out of range"));
    }
    Ok(py.detach(|| topo::degenerate_count(&topo::Faces { indptr, flat }, vertices, epsilon)))
}

/// Edge occurrences in Python dict order: (edges (2E,), start (E+1,), face, from, to).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn edge_occurrences<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    I64Array<'py>,
    I64Array<'py>,
    I64Array<'py>,
    I64Array<'py>,
    I64Array<'py>,
)> {
    let (indptr, flat) = (slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    check_faces(indptr, flat)?;
    let (edges, start, face, from, to) = py.detach(|| {
        let h = topo::halves(&topo::Faces { indptr, flat });
        let e = topo::edges(&h);
        let edges: Vec<i64> = e.a.iter().zip(&e.b).flat_map(|(&a, &b)| [a, b]).collect();
        let start: Vec<i64> = e.start.iter().map(|&s| s as i64).collect();
        let face: Vec<i64> = e.occ.iter().map(|&c| h.face[c as usize] as i64).collect();
        let from: Vec<i64> = e.occ.iter().map(|&c| h.from[c as usize]).collect();
        let to: Vec<i64> = e.occ.iter().map(|&c| h.to[c as usize]).collect();
        (edges, start, face, from, to)
    });
    Ok((
        edges.into_pyarray(py),
        start.into_pyarray(py),
        face.into_pyarray(py),
        from.into_pyarray(py),
        to.into_pyarray(py),
    ))
}

/// `quad_only_closure` core: (vertices (3M,), quads (4Q,), uvs | None, face_uvs | None).
#[pyfunction]
#[pyo3(signature = (indptr, flat, vertices, frozen, uvs=None, uv_indptr=None, uv_flat=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn quad_closure<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    vertices: PyReadonlyArray1<'py, f64>,
    frozen: PyReadonlyArray1<'py, i64>,
    uvs: Option<PyReadonlyArray1<'py, f64>>,
    uv_indptr: Option<PyReadonlyArray1<'py, i64>>,
    uv_flat: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(
    F64Array<'py>,
    I64Array<'py>,
    Option<F64Array<'py>>,
    Option<I64Array<'py>>,
)> {
    let (indptr, flat, vertices, frozen) = (
        slice(&indptr, "indptr")?,
        slice(&flat, "flat")?,
        slice(&vertices, "vertices")?,
        slice(&frozen, "frozen")?,
    );
    check_faces(indptr, flat)?;
    let frozen: Vec<(i64, i64, i64)> = frozen.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect();
    let uvs = uvs.as_ref().map(|u| slice(u, "uvs")).transpose()?;
    let uv_indptr = uv_indptr
        .as_ref()
        .map(|u| slice(u, "uv_indptr"))
        .transpose()?;
    let uv_flat = uv_flat.as_ref().map(|u| slice(u, "uv_flat")).transpose()?;
    let face_uvs = match (uv_indptr, uv_flat) {
        (Some(i), Some(f)) => {
            check_faces(i, f)?;
            Some(topo::Faces { indptr: i, flat: f })
        }
        _ => None,
    };
    let result = py.detach(|| {
        topo::quad_closure(
            &topo::Faces { indptr, flat },
            vertices,
            &frozen,
            uvs,
            face_uvs.as_ref(),
        )
    });
    let c = result.map_err(PyValueError::new_err)?;
    Ok((
        c.vertices.into_pyarray(py),
        c.quads.into_pyarray(py),
        c.uvs.map(|u| u.into_pyarray(py)),
        c.face_uvs.map(|u| u.into_pyarray(py)),
    ))
}

/// `_parity_path_edges`: selected edges as a sorted (2S,) array.
#[pyfunction]
fn parity_path_edges<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    frozen: PyReadonlyArray1<'py, i64>,
) -> PyResult<I64Array<'py>> {
    let (indptr, flat, frozen) = (
        slice(&indptr, "indptr")?,
        slice(&flat, "flat")?,
        slice(&frozen, "frozen")?,
    );
    check_faces(indptr, flat)?;
    let frozen: Vec<(i64, i64)> = frozen.chunks_exact(2).map(|c| (c[0], c[1])).collect();
    let result = py.detach(|| topo::parity_path_edges(&topo::Faces { indptr, flat }, &frozen));
    let edges = result.map_err(PyValueError::new_err)?;
    Ok(edges
        .into_iter()
        .flat_map(|(a, b)| [a, b])
        .collect::<Vec<i64>>()
        .into_pyarray(py))
}

/// `_augment_matching` on a CSR graph; `mate` (-1 = free) is updated in place.
#[pyfunction]
fn augment_matching(
    py: Python<'_>,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    order: PyReadonlyArray1<'_, i64>,
    mut mate: PyReadwriteArray1<'_, i64>,
) -> PyResult<i64> {
    let (indptr, indices, order) = (
        slice(&indptr, "indptr")?,
        slice(&indices, "indices")?,
        slice(&order, "order")?,
    );
    let mate = mate
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("mate must be C-contiguous"))?;
    let n = mate.len() as i64;
    let bad = indptr.len() as i64 != n + 1
        || indptr[n as usize] as usize != indices.len()
        || indices
            .iter()
            .chain(order)
            .chain(mate.iter().filter(|&&m| m != -1))
            .any(|&v| v < 0 || v >= n);
    if bad {
        return Err(PyValueError::new_err("matching arrays are inconsistent"));
    }
    let watch = near::Watch::now();
    py.detach(|| matching::augment(indptr, indices, order, mate, &watch)).ok_or_else(|| pyo3::exceptions::PyInterruptedError::new_err("stopped"))
}

#[pyfunction]
fn winding_number<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    triangles: PyReadonlyArray1<'py, f64>,
) -> PyResult<F64Array<'py>> {
    let (points, triangles) = (slice(&points, "points")?, slice(&triangles, "triangles")?);
    Ok(py
        .detach(|| geom::winding_number(points, triangles))
        .into_pyarray(py))
}

type Pictures<'py> = (Bound<'py, PyArray1<bool>>, F64Array<'py>, F64Array<'py>, Bound<'py, PyArray1<i64>>);

/// `gpu`: draw on the graphics card (raises RuntimeError when there is none or it fails; the caller falls back).
#[pyfunction]
#[pyo3(signature = (points, triangles, normals, center, half, pixels, gpu=false))]
#[allow(clippy::too_many_arguments)]
fn pictures<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    triangles: PyReadonlyArray1<'py, i64>,
    normals: PyReadonlyArray1<'py, f64>,
    center: [f64; 3],
    half: f64,
    pixels: usize,
    gpu: bool,
) -> PyResult<Pictures<'py>> {
    let (points, triangles) = (slice(&points, "points")?, slice(&triangles, "triangles")?);
    let normals = slice(&normals, "normals")?;
    let vertices = (points.len() / 3) as i64;
    if points.len() % 3 != 0
        || triangles.len() % 3 != 0
        || normals.len() != triangles.len()
        || half.is_nan()
        || half <= 0.0
        || pixels == 0
        || triangles.iter().any(|&v| v < 0 || v >= vertices)
    {
        return Err(PyValueError::new_err("picture arrays are inconsistent"));
    }
    let (hit, normal, depth, face) = if gpu {
        py.detach(|| gpu::pictures(points, triangles, normals, center, half, pixels))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
    } else {
        py.detach(|| geom::pictures(points, triangles, normals, center, half, pixels))
    };
    Ok((
        hit.into_pyarray(py),
        normal.into_pyarray(py),
        depth.into_pyarray(py),
        face.into_pyarray(py),
    ))
}

/// The graphics card pictures would be drawn on ("NVIDIA GeForce RTX 3080 Ti (Vulkan)"), or the reason there is none.
/// The first call opens the card, which can take a second or two.
#[pyfunction]
fn gpu_info() -> (bool, String) {
    match gpu::describe() {
        Ok(name) => (true, name),
        Err(why) => (false, why),
    }
}

#[pyfunction]
fn quad_points_valid<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, f64>,
    warp_cosine: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let (points, epsilon) = (slice(&points, "points")?, slice(&epsilon, "epsilon")?);
    if points.len() != epsilon.len() * 12 {
        return Err(PyValueError::new_err("quad arrays are inconsistent"));
    }
    let out = py.detach(|| {
        topo::collect(epsilon.len(), |k| {
            let p = &points[k * 12..k * 12 + 12];
            let corner = |i: usize| [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]];
            geom::quad_valid(
                &[corner(0), corner(1), corner(2), corner(3)],
                epsilon[k],
                warp_cosine,
            )
        })
    });
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn parity_templates<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    edges: PyReadonlyArray1<'py, i64>,
    mids: PyReadonlyArray1<'py, i64>,
    vertices: PyReadonlyArray1<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
    warp_cosine: f64,
) -> PyResult<(I64Array<'py>, Bound<'py, PyArray1<bool>>)> {
    let (indptr, flat) = (slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    let (edges, mids) = (slice(&edges, "edges")?, slice(&mids, "mids")?);
    let (vertices, base) = (slice(&vertices, "vertices")?, slice(&base, "base")?);
    check_faces(indptr, flat)?;
    let inside = |v: i64, n: usize| v >= 0 && (v as usize) < n / 3;
    let ok = edges.len() == mids.len() * 2
        && flat
            .iter()
            .all(|&v| inside(v, vertices.len()) && inside(v, base.len()))
        && mids.iter().all(|&v| inside(v, base.len()));
    if !ok {
        return Err(PyValueError::new_err("template arrays are inconsistent"));
    }
    let (counts, valid) = py
        .detach(|| geom::parity_templates(indptr, flat, edges, mids, vertices, base, warp_cosine));
    Ok((counts.into_pyarray(py), valid.into_pyarray(py)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cross_field_jacobi(
    py: Python<'_>,
    mut cross: PyReadwriteArray1<'_, Complex64>,
    face_a: PyReadonlyArray1<'_, i64>,
    face_b: PyReadonlyArray1<'_, i64>,
    transport: PyReadonlyArray1<'_, Complex64>,
    pinned: PyReadonlyArray1<'_, bool>,
    degree: PyReadonlyArray1<'_, f64>,
    steps: usize,
) -> PyResult<()> {
    let (fa, fb, tr, pin, deg) = (
        slice(&face_a, "face_a")?,
        slice(&face_b, "face_b")?,
        slice(&transport, "transport")?,
        slice(&pinned, "pinned")?,
        slice(&degree, "degree")?,
    );
    let cross = cross
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("cross must be C-contiguous"))?;
    check_field(cross.len(), fa, fb, tr.len(), &[pin.len(), deg.len()])?;
    py.detach(|| geom::cross_jacobi(cross, fa, fb, tr, pin, deg, steps));
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cross_field_gl(
    py: Python<'_>,
    mut cross: PyReadwriteArray1<'_, Complex64>,
    face_a: PyReadonlyArray1<'_, i64>,
    face_b: PyReadonlyArray1<'_, i64>,
    transport: PyReadonlyArray1<'_, Complex64>,
    pinned: PyReadonlyArray1<'_, bool>,
    denom: PyReadonlyArray1<'_, f64>,
    pin_term: PyReadonlyArray1<'_, Complex64>,
    mu: f64,
    dt: f64,
    steps: usize,
) -> PyResult<()> {
    let (fa, fb, tr, pin, den, pt) = (
        slice(&face_a, "face_a")?,
        slice(&face_b, "face_b")?,
        slice(&transport, "transport")?,
        slice(&pinned, "pinned")?,
        slice(&denom, "denom")?,
        slice(&pin_term, "pin_term")?,
    );
    let cross = cross
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("cross must be C-contiguous"))?;
    check_field(
        cross.len(),
        fa,
        fb,
        tr.len(),
        &[pin.len(), den.len(), pt.len()],
    )?;
    py.detach(|| geom::cross_gl(cross, fa, fb, tr, pin, den, pt, mu, dt, steps));
    Ok(())
}

fn check_field(n: usize, fa: &[i64], fb: &[i64], links: usize, per_face: &[usize]) -> PyResult<()> {
    let ok = fa.len() == links
        && fb.len() == links
        && per_face.iter().all(|&l| l == n)
        && fa.iter().chain(fb).all(|&f| f >= 0 && (f as usize) < n);
    if ok {
        Ok(())
    } else {
        Err(PyValueError::new_err("cross field arrays are inconsistent"))
    }
}

fn vertex_ids_ok(flat: &[i64], vertices: usize) -> bool {
    flat.iter().all(|&v| v >= 0 && (v as usize) < vertices / 3)
}

/// `flatquad.make_closable`: (vertices, indptr, flat, kept input face per face, vertices added).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn make_closable<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    vertices: PyReadonlyArray1<'py, f64>,
    span: f64,
) -> PyResult<(F64Array<'py>, I64Array<'py>, I64Array<'py>, I64Array<'py>, usize)> {
    let (indptr, flat, vertices) = (slice(&indptr, "indptr")?, slice(&flat, "flat")?, slice(&vertices, "vertices")?);
    check_faces(indptr, flat)?;
    if !vertex_ids_ok(flat, vertices.len()) {
        return Err(PyValueError::new_err("vertex index out of range"));
    }
    let (out, added) = py.detach(|| {
        let mesh = close::Mesh {
            vertices: vertices.to_vec(),
            indptr: indptr.to_vec(),
            flat: flat.to_vec(),
            origin: (0..indptr.len() as i64 - 1).collect(),
        };
        close::make_closable(&mesh, span)
    });
    Ok((out.vertices.into_pyarray(py), out.indptr.into_pyarray(py), out.flat.into_pyarray(py), out.origin.into_pyarray(py), added))
}

/// `flatquad._close_pieces`: (vertices, indptr, flat, input face per face, how many pieces went each of the four ways).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn close_pieces<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    vertices: PyReadonlyArray1<'py, f64>,
    warp_cosine: f64,
) -> PyResult<(F64Array<'py>, I64Array<'py>, I64Array<'py>, I64Array<'py>, [i64; 4])> {
    let (indptr, flat, vertices) = (slice(&indptr, "indptr")?, slice(&flat, "flat")?, slice(&vertices, "vertices")?);
    check_faces(indptr, flat)?;
    if !vertex_ids_ok(flat, vertices.len()) {
        return Err(PyValueError::new_err("vertex index out of range"));
    }
    let (out, counts) = py
        .detach(|| close::close_pieces(&topo::Faces { indptr, flat }, vertices, warp_cosine))
        .map_err(|e| if e == close::STOPPED { pyo3::exceptions::PyInterruptedError::new_err(e) } else { PyValueError::new_err(e) })?;
    Ok((out.vertices.into_pyarray(py), out.indptr.into_pyarray(py), out.flat.into_pyarray(py), out.origin.into_pyarray(py), counts))
}

/// `quadreduce.join_loose_triangles`: (vertices, indptr, flat, input face whose material each face takes, triangles joined).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn join_loose_triangles<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    vertices: PyReadonlyArray1<'py, f64>,
    material: PyReadonlyArray1<'py, i64>,
    farthest: usize,
) -> PyResult<(F64Array<'py>, I64Array<'py>, I64Array<'py>, I64Array<'py>, i64)> {
    let (indptr, flat, vertices, material) = (slice(&indptr, "indptr")?, slice(&flat, "flat")?, slice(&vertices, "vertices")?, slice(&material, "material")?);
    check_faces(indptr, flat)?;
    if !vertex_ids_ok(flat, vertices.len()) || material.len() + 1 != indptr.len() {
        return Err(PyValueError::new_err("join arrays are inconsistent"));
    }
    let out = py.detach(|| join::join_loose_triangles(indptr, flat, vertices, material, farthest));
    Ok((out.vertices.into_pyarray(py), out.indptr.into_pyarray(py), out.flat.into_pyarray(py), out.owner.into_pyarray(py), out.joined))
}

/// `random.Random(seed).shuffle(list(range(count)))`.
#[pyfunction]
fn shuffled<'py>(py: Python<'py>, count: usize, seed: u32) -> I64Array<'py> {
    py.detach(|| pairing::shuffled(count, seed)).into_pyarray(py)
}

/// `engine._pair_triangles` for each seed, side by side: a list of (quads, leftover triangles, material face per face, rescued).
#[pyfunction]
#[pyo3(signature = (faces, face_a, face_b, quads, scores, minimum, seeds))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn pair_seeds<'py>(
    py: Python<'py>,
    faces: usize,
    face_a: PyReadonlyArray1<'py, i64>,
    face_b: PyReadonlyArray1<'py, i64>,
    quads: PyReadonlyArray1<'py, i64>,
    scores: PyReadonlyArray1<'py, f64>,
    minimum: Option<f64>,
    seeds: Vec<u32>,
) -> PyResult<Vec<(I64Array<'py>, I64Array<'py>, I64Array<'py>, i64)>> {
    let (fa, fb, quads, scores) = (slice(&face_a, "face_a")?, slice(&face_b, "face_b")?, slice(&quads, "quads")?, slice(&scores, "scores")?);
    let ok = fa.len() == fb.len()
        && fa.len() == scores.len()
        && quads.len() == fa.len() * 4
        && fa.iter().chain(fb).all(|&f| f >= 0 && (f as usize) < faces)
        && !scores.iter().any(|s| s.is_nan());
    if !ok {
        return Err(PyValueError::new_err("pairing arrays are inconsistent"));
    }
    let watch = near::Watch::now();
    let rows = py.detach(|| {
        let graph = pairing::allowed(faces, fa, fb, scores, minimum);
        pairing::pair_seeds(&graph, quads, &seeds, &watch)
    });
    let rows = rows.ok_or_else(|| pyo3::exceptions::PyInterruptedError::new_err("stopped"))?;
    Ok(rows
        .into_iter()
        .map(|p| (p.quads.into_pyarray(py), p.tris.into_pyarray(py), p.owner.into_pyarray(py), p.rescued))
        .collect())
}

/// `QuadReducer.save`.
#[pyclass(frozen)]
struct QuadSaved(quadreduce::Saved);

/// `quadreduce.QuadReducer`.
#[pyclass]
struct QuadReducer(quadreduce::Reducer);

#[pymethods]
impl QuadReducer {
    #[new]
    fn new(py: Python<'_>, indptr: PyReadonlyArray1<'_, i64>, flat: PyReadonlyArray1<'_, i64>, vertices: PyReadonlyArray1<'_, f64>, material: PyReadonlyArray1<'_, i64>, dissolve_limit: f64, most_turned: f64) -> PyResult<Self> {
        let (indptr, flat, vertices, material) = (slice(&indptr, "indptr")?, slice(&flat, "flat")?, slice(&vertices, "vertices")?, slice(&material, "material")?);
        check_faces(indptr, flat)?;
        if !vertex_ids_ok(flat, vertices.len()) || material.len() + 1 != indptr.len() || indptr.windows(2).any(|w| w[1] - w[0] < 3) {
            return Err(PyValueError::new_err("reducer arrays are inconsistent"));
        }
        Ok(QuadReducer(py.detach(|| quadreduce::Reducer::new(indptr, flat, vertices, material, dissolve_limit, most_turned))))
    }

    fn remove(&mut self, py: Python<'_>, count: usize) {
        py.detach(|| self.0.remove(count));
    }

    #[getter]
    fn alive(&self) -> usize {
        self.0.alive()
    }

    #[getter]
    fn refused(&self) -> usize {
        self.0.refused()
    }

    fn retry_refused(&mut self) {
        self.0.retry_refused();
    }

    fn original_ids<'py>(&self, py: Python<'py>, vertices: PyReadonlyArray1<'py, i64>) -> PyResult<I64Array<'py>> {
        Ok(self.0.original_ids(slice(&vertices, "vertices")?).into_pyarray(py))
    }

    fn freeze(&mut self, vertices: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        self.0.freeze(slice(&vertices, "vertices")?);
        Ok(())
    }

    fn save(&self, py: Python<'_>) -> QuadSaved {
        QuadSaved(py.detach(|| self.0.save()))
    }

    fn restore(&mut self, py: Python<'_>, saved: &QuadSaved) {
        py.detach(|| self.0.restore(&saved.0));
    }

    #[allow(clippy::type_complexity)]
    fn result<'py>(&mut self, py: Python<'py>) -> (F64Array<'py>, I64Array<'py>, I64Array<'py>, I64Array<'py>) {
        let (vertices, indptr, flat, owner) = py.detach(|| self.0.result());
        (vertices.into_pyarray(py), indptr.into_pyarray(py), flat.into_pyarray(py), owner.into_pyarray(py))
    }
}

/// One piece taken all the way down: every size of it is then a quick `level(faces)`.
#[pyclass(frozen)]
struct Reducer(qem::Run);

#[pymethods]
impl Reducer {
    #[new]
    fn new(py: Python<'_>, points: PyReadonlyArray1<'_, f64>, tris: PyReadonlyArray1<'_, i64>, locked: PyReadonlyArray1<'_, bool>, least: u32) -> PyResult<Self> {
        let (points, tris, locked) = (slice(&points, "points")?, slice(&tris, "tris")?, slice(&locked, "locked")?);
        if tris.len() % 3 != 0 || points.len() % 3 != 0 || !vertex_ids_ok(tris, points.len()) {
            return Err(PyValueError::new_err("reducer arrays are inconsistent"));
        }
        Ok(Reducer(py.detach(|| {
            let tris: Vec<u32> = tris.iter().map(|&v| v as u32).collect();
            qem::run(points, &tris, locked, least)
        })))
    }

    /// (points, triangles) of the piece at `wanted` triangles, or as near as the run could get.
    fn level<'py>(&self, py: Python<'py>, wanted: u32) -> (F64Array<'py>, I64Array<'py>) {
        let (points, tris, _) = py.detach(|| self.0.level(wanted));
        (points.into_pyarray(py), tris.into_pyarray(py))
    }

    /// `level`, and the input triangle each output triangle is.
    fn level_owned<'py>(&self, py: Python<'py>, wanted: u32) -> (F64Array<'py>, I64Array<'py>, I64Array<'py>) {
        let (points, tris, owner) = py.detach(|| self.0.level(wanted));
        (points.into_pyarray(py), tris.into_pyarray(py), owner.into_pyarray(py))
    }

    fn smallest(&self) -> u32 {
        self.0.smallest()
    }
}

/// `parts._cluster`: a cluster id per 2D point.
#[pyfunction]
fn leader_clusters<'py>(py: Python<'py>, values: PyReadonlyArray1<'py, f64>, tol: f64) -> PyResult<I64Array<'py>> {
    let values = slice(&values, "values")?;
    if values.len() % 2 != 0 || !(tol > 0.0) {
        return Err(PyValueError::new_err("cluster arrays are inconsistent"));
    }
    Ok(py.detach(|| geom::leader_clusters(values, tol)).into_pyarray(py))
}

/// Set to end the calls that were given it early, with `InterruptedError`.
#[pyclass(frozen)]
struct StopFlag(std::sync::atomic::AtomicBool);

#[pymethods]
impl StopFlag {
    #[new]
    fn new() -> Self {
        StopFlag(std::sync::atomic::AtomicBool::new(false))
    }

    fn set(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn finished(done: bool) -> PyResult<()> {
    if done {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyInterruptedError::new_err("stopped"))
    }
}

fn check_queries(queries: &[f64]) -> PyResult<()> {
    if queries.len() % 3 != 0 || queries.iter().any(|x| x.is_nan()) {
        return Err(PyValueError::new_err("queries must be finite points"));
    }
    Ok(())
}

/// Nearest points, a whole array at a time.
#[pyclass(frozen)]
struct PointLookup(near::PointTree);

#[pymethods]
impl PointLookup {
    #[new]
    fn new(py: Python<'_>, points: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let points = slice(&points, "points")?;
        check_queries(points)?;
        Ok(PointLookup(py.detach(|| near::PointTree::new(points))))
    }

    /// (index, distance) per query.
    fn nearest<'py>(&self, py: Python<'py>, queries: PyReadonlyArray1<'py, f64>, stop: &StopFlag) -> PyResult<(I64Array<'py>, F64Array<'py>)> {
        let queries = slice(&queries, "queries")?;
        check_queries(queries)?;
        let (done, rows) = py.detach(|| near::nearest_points(&self.0, queries, &stop.0));
        finished(done)?;
        let (ids, away): (Vec<i64>, Vec<f64>) = rows.into_iter().unzip();
        Ok((ids.into_pyarray(py), away.into_pyarray(py)))
    }

    /// For each query, the closest of its `count` nearest points of the same kind, or -1.
    fn first_of_kind<'py>(&self, py: Python<'py>, queries: PyReadonlyArray1<'py, f64>, query_kind: PyReadonlyArray1<'py, i64>, point_kind: PyReadonlyArray1<'py, i64>, count: usize, stop: &StopFlag) -> PyResult<I64Array<'py>> {
        let (queries, query_kind, point_kind) = (slice(&queries, "queries")?, slice(&query_kind, "query_kind")?, slice(&point_kind, "point_kind")?);
        check_queries(queries)?;
        if query_kind.len() != queries.len() / 3 || point_kind.len() != self.0.len() {
            return Err(PyValueError::new_err("kinds don't match the points"));
        }
        let (done, out) = py.detach(|| near::first_of_kind(&self.0, queries, query_kind, point_kind, count, &stop.0));
        finished(done)?;
        Ok(out.into_pyarray(py))
    }
}

/// Nearest faces and faces passing through each other, a whole array at a time.
#[pyclass(frozen)]
struct FaceLookup(near::FaceTree);

#[pymethods]
impl FaceLookup {
    #[new]
    fn new(py: Python<'_>, points: PyReadonlyArray1<'_, f64>, indptr: PyReadonlyArray1<'_, i64>, flat: PyReadonlyArray1<'_, i64>) -> PyResult<Self> {
        let (points, indptr, flat) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?);
        check_queries(points)?;
        check_faces(indptr, flat)?;
        if !vertex_ids_ok(flat, points.len()) {
            return Err(PyValueError::new_err("vertex index out of range"));
        }
        Ok(FaceLookup(py.detach(|| near::FaceTree::new(points, indptr, flat))))
    }

    /// (closest points (3N,), polygon, distance) per query.
    fn nearest<'py>(&self, py: Python<'py>, queries: PyReadonlyArray1<'py, f64>, stop: &StopFlag) -> PyResult<(F64Array<'py>, I64Array<'py>, F64Array<'py>)> {
        let queries = slice(&queries, "queries")?;
        check_queries(queries)?;
        let (done, rows) = py.detach(|| near::nearest_faces(&self.0, queries, &stop.0));
        finished(done)?;
        let at: Vec<f64> = rows.iter().flat_map(|row| row.0).collect();
        let face: Vec<i64> = rows.iter().map(|row| row.1).collect();
        let away: Vec<f64> = rows.iter().map(|row| row.2).collect();
        Ok((at.into_pyarray(py), face.into_pyarray(py), away.into_pyarray(py)))
    }

    /// Polygon pairs (2M,) that pass through each other and share no vertex.
    fn crossing_pairs<'py>(&self, py: Python<'py>, stop: &StopFlag) -> PyResult<I64Array<'py>> {
        let found = py.detach(|| self.0.crossing_pairs(&stop.0));
        finished(found.is_some())?;
        let flat: Vec<i64> = found.unwrap_or_default().iter().flat_map(|p| [p.0 as i64, p.1 as i64]).collect();
        Ok(flat.into_pyarray(py))
    }
}

/// `checks.edge_flags` and `checks.hole_counts`: (broken per face, non-manifold edges, flipped edges, duplicate faces, holes, pieces, holes per piece).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn mesh_checks<'py>(py: Python<'py>, indptr: PyReadonlyArray1<'py, i64>, flat: PyReadonlyArray1<'py, i64>, piece: PyReadonlyArray1<'py, i64>) -> PyResult<(Bound<'py, PyArray1<bool>>, i64, i64, i64, i64, I64Array<'py>, I64Array<'py>)> {
    let (indptr, flat, piece) = (slice(&indptr, "indptr")?, slice(&flat, "flat")?, slice(&piece, "piece")?);
    check_faces(indptr, flat)?;
    if piece.len() != indptr.len() - 1 || flat.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err("check arrays are inconsistent"));
    }
    let (broken, bad_edges, flipped, dups, holes, per) = py.detach(|| near::mesh_checks(&topo::Faces { indptr, flat }, piece));
    let (pieces, counts): (Vec<i64>, Vec<i64>) = per.into_iter().unzip();
    Ok((broken.into_pyarray(py), bad_edges, flipped, dups, holes, pieces.into_pyarray(py), counts.into_pyarray(py)))
}

/// `chords._closed_walk` up to its turns: (price, turn index per corner), or None.
#[pyfunction]
fn closed_walk(py: Python<'_>, costs: Vec<[f64; 3]>, limit: f64) -> Option<(f64, Vec<usize>)> {
    py.detach(|| chords::closed_walk(&costs, limit))
}

/// `chords.PictureJudge` with its pictures held natively.
#[pyclass(frozen)]
struct PictureJudgeCore(chords::Judge);

#[pymethods]
impl PictureJudgeCore {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(hit: PyReadonlyArray1<'_, bool>, normal: PyReadonlyArray1<'_, f64>, depth: PyReadonlyArray1<'_, f64>, center: [f64; 3], half: f64, pixels: usize, cosine: f64, behind: f64) -> PyResult<Self> {
        let (hit, normal, depth) = (slice(&hit, "hit")?, slice(&normal, "normal")?, slice(&depth, "depth")?);
        let cells = 6 * pixels * pixels;
        if hit.len() != cells || depth.len() != cells || normal.len() != 3 * cells || !(half > 0.0) {
            return Err(PyValueError::new_err("picture arrays are inconsistent"));
        }
        Ok(PictureJudgeCore(chords::Judge { hit: hit.to_vec(), normal: normal.to_vec(), depth: depth.to_vec(), center, half, pixels, cosine, behind }))
    }

    /// Pixels a quad gets wrong.
    fn wrong(&self, corners: [[f64; 3]; 4]) -> i64 {
        self.0.wrong(&corners)
    }
}

fn field_arrays(count: usize, arrays: &[(&[f64], usize)], ids: &[&[i64]], a: &[i64], b: &[i64]) -> PyResult<()> {
    let ok = arrays.iter().all(|(x, w)| x.len() == count * w) && ids.iter().all(|x| x.len() == count) && a.len() == b.len() && a.iter().chain(b).all(|&v| v >= 0 && (v as usize) < count);
    if ok {
        Ok(())
    } else {
        Err(PyValueError::new_err("field arrays are inconsistent"))
    }
}

/// `fieldmesh._orient_sweep`, in place on `q` (3N,), with the colors drawn in Python.
#[pyfunction]
fn field_orient_sweep(py: Python<'_>, mut q: PyReadwriteArray1<'_, f64>, normals: PyReadonlyArray1<'_, f64>, pinned: PyReadonlyArray1<'_, i64>, a: PyReadonlyArray1<'_, i64>, b: PyReadonlyArray1<'_, i64>, colors: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
    let (normals, pinned, a, b, colors) = (slice(&normals, "normals")?, slice(&pinned, "pinned")?, slice(&a, "a")?, slice(&b, "b")?, slice(&colors, "colors")?);
    let q = q.as_slice_mut().map_err(|_| PyValueError::new_err("q must be C-contiguous"))?;
    field_arrays(colors.len(), &[(q, 3), (normals, 3)], &[pinned], a, b)?;
    py.detach(|| field::orient_sweep(q, normals, pinned, a, b, colors));
    Ok(())
}

/// `fieldmesh._position_sweep`, in place on `o` (3N,) and `misfit` (N,, or empty for None).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn field_position_sweep(py: Python<'_>, points: PyReadonlyArray1<'_, f64>, normals: PyReadonlyArray1<'_, f64>, q: PyReadonlyArray1<'_, f64>, sizes: PyReadonlyArray1<'_, f64>, pinned: PyReadonlyArray1<'_, i64>, mut o: PyReadwriteArray1<'_, f64>, mut misfit: PyReadwriteArray1<'_, f64>, a: PyReadonlyArray1<'_, i64>, b: PyReadonlyArray1<'_, i64>, colors: PyReadonlyArray1<'_, i64>, robust: bool, settled: f64, give: f64, confidence: bool) -> PyResult<()> {
    let (points, normals, q, sizes, pinned) = (slice(&points, "points")?, slice(&normals, "normals")?, slice(&q, "q")?, slice(&sizes, "sizes")?, slice(&pinned, "pinned")?);
    let (a, b, colors) = (slice(&a, "a")?, slice(&b, "b")?, slice(&colors, "colors")?);
    let o = o.as_slice_mut().map_err(|_| PyValueError::new_err("o must be C-contiguous"))?;
    let misfit = misfit.as_slice_mut().map_err(|_| PyValueError::new_err("misfit must be C-contiguous"))?;
    let count = colors.len();
    field_arrays(count, &[(points, 3), (normals, 3), (q, 3), (sizes, 1), (o, 3)], &[pinned], a, b)?;
    if !misfit.is_empty() && misfit.len() != count {
        return Err(PyValueError::new_err("field arrays are inconsistent"));
    }
    let level = field::Level { points, normals, q, sizes, pinned };
    py.detach(|| field::position_sweep(&level, o, misfit, a, b, colors, robust, settled, give, confidence));
    Ok(())
}

/// `fieldmesh.nearest_points`' loop over the asks (the grid is made in Python).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn field_nearest_points<'py>(py: Python<'py>, points: PyReadonlyArray1<'py, f64>, normals: PyReadonlyArray1<'py, f64>, sorted_keys: PyReadonlyArray1<'py, i64>, order: PyReadonlyArray1<'py, i64>, centre: PyReadonlyArray1<'py, i64>, around: PyReadonlyArray1<'py, i64>, asks: PyReadonlyArray1<'py, f64>, ask_normals: PyReadonlyArray1<'py, f64>, far_side: f64) -> PyResult<I64Array<'py>> {
    let (points, normals, sorted_keys, order) = (slice(&points, "points")?, slice(&normals, "normals")?, slice(&sorted_keys, "sorted_keys")?, slice(&order, "order")?);
    let (centre, around, asks, ask_normals) = (slice(&centre, "centre")?, slice(&around, "around")?, slice(&asks, "asks")?, slice(&ask_normals, "ask_normals")?);
    let count = points.len() / 3;
    if points.len() != normals.len() || sorted_keys.len() != count || order.len() != count || asks.len() != 3 * centre.len() || ask_normals.len() != asks.len() || order.iter().any(|&f| f < 0 || f as usize >= count) || count == 0 {
        return Err(PyValueError::new_err("nearest point arrays are inconsistent"));
    }
    Ok(py.detach(|| field::nearest_points(points, normals, sorted_keys, order, centre, around, asks, ask_normals, far_side)).into_pyarray(py))
}

/// `segment._merge_all`'s merge loop: (merges (2M,), costs). Weights: (convex, usual, curve_step, thick_step, neck, base).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn segment_merges<'py>(py: Python<'py>, count: usize, pa: PyReadonlyArray1<'py, i64>, pb: PyReadonlyArray1<'py, i64>, rows: PyReadonlyArray1<'py, f64>, area: PyReadonlyArray1<'py, f64>, curve: PyReadonlyArray1<'py, f64>, thick: PyReadonlyArray1<'py, f64>, whole: f64, weights: [f64; 6], stop: &StopFlag) -> PyResult<(I64Array<'py>, F64Array<'py>)> {
    let (pa, pb, rows) = (slice(&pa, "pa")?, slice(&pb, "pb")?, slice(&rows, "rows")?);
    let (area, curve, thick) = (slice(&area, "area")?, slice(&curve, "curve")?, slice(&thick, "thick")?);
    if pa.len() != pb.len() || rows.len() != 3 * pa.len() || area.len() != count || curve.len() != count || thick.len() != count || pa.iter().chain(pb).any(|&v| v < 0 || v as usize >= count) {
        return Err(PyValueError::new_err("merge arrays are inconsistent"));
    }
    let rows: Vec<[f64; 3]> = rows.chunks_exact(3).map(|r| [r[0], r[1], r[2]]).collect();
    let w = segment::Weights { convex: weights[0], usual: weights[1], curve_step: weights[2], thick_step: weights[3], neck: weights[4], base: weights[5] };
    let found = py.detach(|| segment::merge_all(count, pa, pb, &rows, area, curve, thick, whole, &w, &stop.0));
    let (merges, costs) = found.ok_or_else(|| pyo3::exceptions::PyInterruptedError::new_err("stopped"))?;
    let flat: Vec<i64> = merges.iter().flat_map(|m| [m.0, m.1]).collect();
    Ok((flat.into_pyarray(py), costs.into_pyarray(py)))
}

/// Faces as a list of tuples with `offset` added to every vertex: from (indptr, flat), or from any sequence of sequences.
#[pyfunction]
#[pyo3(signature = (polys, offset, indptr=None, flat=None))]
fn shifted_faces<'py>(py: Python<'py>, polys: &Bound<'py, PyAny>, offset: i64, indptr: Option<PyReadonlyArray1<'py, i64>>, flat: Option<PyReadonlyArray1<'py, i64>>) -> PyResult<Bound<'py, PyList>> {
    if let (Some(indptr), Some(flat)) = (indptr, flat) {
        let (indptr, flat) = (slice(&indptr, "indptr")?, slice(&flat, "flat")?);
        check_faces(indptr, flat)?;
        let rows = indptr.windows(2).map(|w| PyTuple::new(py, flat[w[0] as usize..w[1] as usize].iter().map(|&v| v + offset)));
        return PyList::new(py, rows.collect::<PyResult<Vec<_>>>()?);
    }
    let out = PyList::empty(py);
    let mut values = Vec::with_capacity(8);
    for item in polys.try_iter()? {
        values.clear();
        for v in item?.try_iter()? {
            values.push(v?.extract::<i64>()? + offset);
        }
        out.append(PyTuple::new(py, &values)?)?;
    }
    Ok(out)
}

/// `shading.corner_normals`: a unit smooth-shading normal per corner (3C,), on all cores.
#[pyfunction]
fn corner_normals<'py>(py: Python<'py>, points: PyReadonlyArray1<'py, f64>, indptr: PyReadonlyArray1<'py, i64>, flat: PyReadonlyArray1<'py, i64>, limit: f64) -> PyResult<F64Array<'py>> {
    let (points, indptr, flat) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    check_faces(indptr, flat)?;
    if !vertex_ids_ok(flat, points.len()) || indptr.windows(2).any(|w| w[0] == w[1]) {
        return Err(PyValueError::new_err("vertex index out of range"));
    }
    Ok(py.detach(|| geom::corner_normals(points, indptr, flat, limit)).into_pyarray(py))
}

/// Every native call running now that watches for it (reductions, closing pieces) ends early.
#[pyfunction]
fn stop_running() {
    near::stop_running();
}

// main-only: uv begin seam
mod uvseam;

/// Compile probe of the `uvseam` kernel: 1 when every helper it needs is there.
#[pyfunction]
fn uvseam_probe() -> usize {
    uvseam::probe()
}

struct UvseamSource<'a> {
    points: &'a [f64],
    indptr: &'a [i64],
    flat: &'a [i64],
    uv: &'a [f64],
    material: &'a [i64],
}

fn uvseam_mesh<'a>(points: &'a [f64], indptr: &'a [i64], flat: &'a [i64]) -> PyResult<topo::Faces<'a>> {
    check_faces(indptr, flat)?;
    let n = points.len() / 3;
    if points.len() % 3 != 0 || flat.iter().any(|&v| v < 0 || v as usize >= n) || points.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("points or face vertex ids are invalid"));
    }
    Ok(topo::Faces { indptr, flat })
}

fn uvseam_source<'a>(src: &UvseamSource<'a>) -> PyResult<topo::Faces<'a>> {
    let faces = uvseam_mesh(src.points, src.indptr, src.flat)?;
    if src.uv.len() != 2 * src.flat.len() || !(src.material.is_empty() || src.material.len() == faces.count()) || src.uv.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("corner UVs need 2 finite values per corner and materials one per face (or none)"));
    }
    Ok(faces)
}

fn uvseam_set(src: &UvseamSource, faces: &topo::Faces, repeat: &[bool], texture_px: f64, uv_tol_px: f64, cut_here: &[i64], classes: &[i64]) -> PyResult<uvseam::SeamSet> {
    let mut set = uvseam::extract(src.points, faces, src.uv, src.material, repeat, texture_px, uv_tol_px);
    if !cut_here.is_empty() {
        uvseam::classify(&mut set, repeat, cut_here);
    }
    if !classes.is_empty() {
        if classes.len() != set.chains() || classes.iter().any(|&c| !(0..=3).contains(&c)) {
            return Err(PyValueError::new_err("chain classes need one code 0..3 per chain"));
        }
        uvseam::set_classes(&mut set, &classes.iter().map(|&c| c as u8).collect::<Vec<_>>());
    }
    Ok(set)
}

fn uvseam_put_set<'py>(py: Python<'py>, d: &Bound<'py, PyDict>, set: uvseam::SeamSet) -> PyResult<()> {
    d.set_item("n_welded", set.n_welded)?;
    d.set_item("n_islands", set.n_islands)?;
    d.set_item("n_regions", set.n_regions)?;
    d.set_item("texture_px", set.texture_px)?;
    d.set_item("weld", set.weld.into_pyarray(py))?;
    d.set_item("island", set.island.into_pyarray(py))?;
    d.set_item("region", set.region.into_pyarray(py))?;
    d.set_item("edge_a", set.edge_a.into_pyarray(py))?;
    d.set_item("edge_b", set.edge_b.into_pyarray(py))?;
    d.set_item("edge_chain", set.edge_chain.into_pyarray(py))?;
    d.set_item("edge_face", set.edge_face.concat().into_pyarray(py))?;
    d.set_item("corner", set.corner.into_pyarray(py))?;
    d.set_item("run_indptr", set.run_indptr.into_pyarray(py))?;
    d.set_item("run_vertex", set.run_vertex.into_pyarray(py))?;
    d.set_item("uv_a", set.uv_a.into_pyarray(py))?;
    d.set_item("uv_b", set.uv_b.into_pyarray(py))?;
    d.set_item("gap_px", set.gap_px.into_pyarray(py))?;
    d.set_item("class", set.class.iter().map(|&c| c as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("chain_island", set.chain_island.concat().into_pyarray(py))?;
    d.set_item("chain_material", set.chain_material.concat().into_pyarray(py))?;
    d.set_item("chain_length", set.chain_length.into_pyarray(py))?;
    d.set_item("island_tile", set.island_tile.concat().into_pyarray(py))?;
    Ok(())
}

fn uvseam_put_fold(d: &Bound<'_, PyDict>, fold: uvseam::FoldReport) -> PyResult<()> {
    d.set_item("folded_before", fold.before)?;
    d.set_item("folded_after", fold.after)?;
    d.set_item("fold_rings", fold.rings)?;
    d.set_item("fold_sweeps", fold.sweeps)?;
    d.set_item("fold_stuck", fold.stuck)?;
    d.set_item("fold_stalled", fold.stalled)?;
    d.set_item("fold_stopped", fold.stopped)?;
    Ok(())
}

/// Islands, side regions, seam chains and classes of a source (`uvseam::extract` + `classify`) as a dict of arrays
/// (pairs flattened: edge_face, chain_island, chain_material, island_tile hold 2 values per row). `repeat` per
/// material (missing: repeats); `cut_here` chain ids made Hard.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uvseam_extract<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    corner_uv: PyReadonlyArray1<'py, f64>,
    material: PyReadonlyArray1<'py, i64>,
    repeat: PyReadonlyArray1<'py, bool>,
    texture_px: f64,
    uv_tol_px: f64,
    cut_here: PyReadonlyArray1<'py, i64>,
    stop: &StopFlag,
) -> PyResult<Bound<'py, PyDict>> {
    let src = UvseamSource { points: slice(&points, "points")?, indptr: slice(&indptr, "indptr")?, flat: slice(&flat, "flat")?, uv: slice(&corner_uv, "corner_uv")?, material: slice(&material, "material")? };
    let (repeat, cut_here) = (slice(&repeat, "repeat")?, slice(&cut_here, "cut_here")?);
    let faces = uvseam_source(&src)?;
    let set = py.detach(|| uvseam_set(&src, &faces, repeat, texture_px, uv_tol_px, cut_here, &[]))?;
    finished(!stop.0.load(std::sync::atomic::Ordering::Relaxed))?;
    let d = PyDict::new(py);
    uvseam_put_set(py, &d, set)?;
    Ok(d)
}

/// Pre-search Hard choice (`uvseam::triage`): the extract dict (with the new classes) plus hard_islands, hard_cost,
/// left_islands, left_cost, spent, budget, edge_est. `seen_face` per source face, or empty for "all seen".
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uvseam_triage<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    corner_uv: PyReadonlyArray1<'py, f64>,
    material: PyReadonlyArray1<'py, i64>,
    repeat: PyReadonlyArray1<'py, bool>,
    texture_px: f64,
    uv_tol_px: f64,
    cut_here: PyReadonlyArray1<'py, i64>,
    f_target: f64,
    allowance: f64,
    seen_face: PyReadonlyArray1<'py, f64>,
    cost_a: f64,
    cost_b: f64,
    stop: &StopFlag,
) -> PyResult<Bound<'py, PyDict>> {
    let src = UvseamSource { points: slice(&points, "points")?, indptr: slice(&indptr, "indptr")?, flat: slice(&flat, "flat")?, uv: slice(&corner_uv, "corner_uv")?, material: slice(&material, "material")? };
    let (repeat, cut_here, seen) = (slice(&repeat, "repeat")?, slice(&cut_here, "cut_here")?, slice(&seen_face, "seen_face")?);
    let faces = uvseam_source(&src)?;
    if !(seen.is_empty() || seen.len() == faces.count()) {
        return Err(PyValueError::new_err("seen_face needs one value per face (or none)"));
    }
    let (set, t) = py.detach(|| {
        let mut set = uvseam_set(&src, &faces, repeat, texture_px, uv_tol_px, cut_here, &[])?;
        let t = uvseam::triage(&mut set, src.points, &faces, f_target, allowance, if seen.is_empty() { None } else { Some(seen) }, cost_a, cost_b);
        Ok::<_, PyErr>((set, t))
    })?;
    finished(!stop.0.load(std::sync::atomic::Ordering::Relaxed))?;
    let d = PyDict::new(py);
    uvseam_put_set(py, &d, set)?;
    d.set_item("hard_islands", t.hard_islands.into_pyarray(py))?;
    d.set_item("hard_cost", t.hard_cost.into_pyarray(py))?;
    d.set_item("left_islands", t.left_islands.into_pyarray(py))?;
    d.set_item("left_cost", t.left_cost.into_pyarray(py))?;
    d.set_item("spent", t.spent)?;
    d.set_item("budget", t.budget)?;
    d.set_item("edge_est", t.edge_est)?;
    Ok(d)
}

fn uvseam_put_lookup<'py>(py: Python<'py>, d: &Bound<'py, PyDict>, look: uvseam::Lookup) -> PyResult<()> {
    d.set_item("done", look.done)?;
    d.set_item("ids", look.ids.into_pyarray(py))?;
    d.set_item("w", look.w.into_pyarray(py))?;
    d.set_item("shift", look.shift.into_pyarray(py))?;
    d.set_item("slid", look.slid.into_pyarray(py))?;
    d.set_item("ref_ids", look.ref_ids.into_pyarray(py))?;
    d.set_item("ref_w", look.ref_w.into_pyarray(py))?;
    d.set_item("ref_face", look.ref_face.into_pyarray(py))?;
    d.set_item("orient", look.orient.into_pyarray(py))?;
    d.set_item("region", look.region.into_pyarray(py))?;
    d.set_item("join", look.join.into_pyarray(py))?;
    Ok(())
}

/// Per-side transfer (`uvseam::transfer`) of the source's corner UVs onto a result mesh: dict with uv (2 per result
/// corner), expected (winding per face), the lookup arrays (for other layers) and the fold report. `classes` per chain
/// (empty: as classified; e.g. after triage). A stop hands back done = False and the faces read so far (no error).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uvseam_transfer<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    corner_uv: PyReadonlyArray1<'py, f64>,
    material: PyReadonlyArray1<'py, i64>,
    repeat: PyReadonlyArray1<'py, bool>,
    texture_px: f64,
    uv_tol_px: f64,
    classes: PyReadonlyArray1<'py, i64>,
    res_points: PyReadonlyArray1<'py, f64>,
    res_indptr: PyReadonlyArray1<'py, i64>,
    res_flat: PyReadonlyArray1<'py, i64>,
    fix_folds: bool,
    stop: &StopFlag,
) -> PyResult<Bound<'py, PyDict>> {
    let src = UvseamSource { points: slice(&points, "points")?, indptr: slice(&indptr, "indptr")?, flat: slice(&flat, "flat")?, uv: slice(&corner_uv, "corner_uv")?, material: slice(&material, "material")? };
    let (repeat, classes) = (slice(&repeat, "repeat")?, slice(&classes, "classes")?);
    let faces = uvseam_source(&src)?;
    let (rp, ri, rf) = (slice(&res_points, "res_points")?, slice(&res_indptr, "res_indptr")?, slice(&res_flat, "res_flat")?);
    let res = uvseam_mesh(rp, ri, rf)?;
    let t = py.detach(|| {
        let set = uvseam_set(&src, &faces, repeat, texture_px, uv_tol_px, &[], classes)?;
        Ok::<_, PyErr>(uvseam::transfer(src.points, &faces, &set, src.uv, rp, &res, fix_folds, &stop.0))
    })?;
    let d = PyDict::new(py);
    d.set_item("uv", t.uv.into_pyarray(py))?;
    d.set_item("expected", t.expected.into_pyarray(py))?;
    uvseam_put_lookup(py, &d, t.look)?;
    uvseam_put_fold(&d, t.fold)?;
    Ok(d)
}

fn uvseam_item<T: numpy::Element + Copy>(lookup: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<T>> {
    let item = lookup.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("lookup misses {key}")))?;
    let array = item.cast_into::<PyArray1<T>>().map_err(|_| PyValueError::new_err(format!("lookup {key} has the wrong type")))?;
    let read = numpy::PyArrayMethods::readonly(&array);
    Ok(slice(&read, key)?.to_vec())
}

/// Another UV layer through an existing lookup (`uvseam::evaluate` + fold fix): (uv, expected, fold dict).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn uvseam_layer<'py>(
    py: Python<'py>,
    src_indptr: PyReadonlyArray1<'py, i64>,
    src_flat: PyReadonlyArray1<'py, i64>,
    layer: PyReadonlyArray1<'py, f64>,
    res_points: PyReadonlyArray1<'py, f64>,
    res_indptr: PyReadonlyArray1<'py, i64>,
    res_flat: PyReadonlyArray1<'py, i64>,
    lookup: &Bound<'py, PyDict>,
    texture_px: f64,
    fix_folds: bool,
    stop: &StopFlag,
) -> PyResult<(F64Array<'py>, F64Array<'py>, Bound<'py, PyDict>)> {
    let (si, sf, layer) = (slice(&src_indptr, "src_indptr")?, slice(&src_flat, "src_flat")?, slice(&layer, "layer")?);
    check_faces(si, sf)?;
    let (rp, ri, rf) = (slice(&res_points, "res_points")?, slice(&res_indptr, "res_indptr")?, slice(&res_flat, "res_flat")?);
    let res = uvseam_mesh(rp, ri, rf)?;
    let look = uvseam::Lookup {
        ids: uvseam_item(lookup, "ids")?,
        w: uvseam_item(lookup, "w")?,
        shift: uvseam_item(lookup, "shift")?,
        slid: uvseam_item(lookup, "slid")?,
        ref_ids: uvseam_item(lookup, "ref_ids")?,
        ref_w: uvseam_item(lookup, "ref_w")?,
        ref_face: uvseam_item(lookup, "ref_face")?,
        orient: uvseam_item(lookup, "orient")?,
        region: uvseam_item(lookup, "region")?,
        join: uvseam_item(lookup, "join")?,
        done: true,
    };
    let (nc, nf, n_src) = (rf.len(), res.count(), sf.len());
    let ok = look.ids.len() == 3 * nc
        && look.w.len() == 3 * nc
        && look.shift.len() == nc
        && look.ref_ids.len() == 3 * nf
        && look.ref_w.len() == 3 * nf
        && look.ref_face.len() == nf
        && look.orient.len() == nf
        && look.region.len() == nf
        && layer.len() == 2 * n_src
        && look.ids.iter().chain(&look.ref_ids).all(|&h| h >= 0 && (h as usize) < n_src.max(1))
        && look.ref_face.iter().all(|&f| f < si.len() as i64 - 1);
    if !ok {
        return Err(PyValueError::new_err("lookup, layer and meshes do not match"));
    }
    let src_faces = topo::Faces { indptr: si, flat: sf };
    let (uv, expected, fold) = py.detach(|| {
        let (mut uv, expected) = uvseam::evaluate(&look, &src_faces, layer, &res);
        let fold = if fix_folds { uvseam::fold_fix(rp, &res, &mut uv, &look.region, &expected, &look.join, texture_px, &stop.0) } else { uvseam::FoldReport::default() };
        (uv, expected, fold)
    });
    let d = PyDict::new(py);
    uvseam_put_fold(&d, fold)?;
    Ok((uv.into_pyarray(py), expected.into_pyarray(py), d))
}

fn register_uvseam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uvseam_probe, m)?)?;
    m.add_function(wrap_pyfunction!(uvseam_extract, m)?)?;
    m.add_function(wrap_pyfunction!(uvseam_triage, m)?)?;
    m.add_function(wrap_pyfunction!(uvseam_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(uvseam_layer, m)?)?;
    Ok(())
}
// main-only: uv end seam

// main-only: uv begin snap
mod uvsnap;

/// Compile probe of the `uvsnap` kernel: 1 when every helper it needs is there.
#[pyfunction]
fn uvsnap_probe() -> usize {
    uvsnap::probe()
}

/// Seam chain arrays (as `uvseam_extract` returns them, `xyz` = 3 per welded vertex) checked and held for a call.
struct UvsnapChains {
    run_indptr: Vec<i64>,
    run_vertex: Vec<i64>,
    uv_a: Vec<f64>,
    uv_b: Vec<f64>,
    class: Vec<u8>,
    gap_px: Vec<f64>,
    xyz: Vec<f64>,
}

impl UvsnapChains {
    #[allow(clippy::too_many_arguments)]
    fn read(run_indptr: &[i64], run_vertex: &[i64], uv_a: &[f64], uv_b: &[f64], class: &[i64], gap_px: &[f64], xyz: &[f64]) -> PyResult<Self> {
        let nw = xyz.len() / 3;
        let nr = run_vertex.len();
        let ok = xyz.len() % 3 == 0
            && run_indptr.len() == class.len() + 1
            && run_indptr.first() == Some(&0)
            && *run_indptr.last().unwrap() as usize == nr
            && run_indptr.windows(2).all(|w| w[1] - w[0] >= 1)
            && run_vertex.iter().all(|&v| v >= 0 && (v as usize) < nw)
            && uv_a.len() == 2 * nr
            && uv_b.len() == 2 * nr
            && gap_px.len() == nr
            && class.iter().all(|&c| (0..=3).contains(&c))
            && xyz.iter().chain(uv_a).chain(uv_b).chain(gap_px).all(|x| x.is_finite());
        if !ok {
            return Err(PyValueError::new_err("seam chains are invalid (run_indptr, run_vertex, uv_a, uv_b, class, gap_px, xyz)"));
        }
        Ok(UvsnapChains {
            run_indptr: run_indptr.to_vec(),
            run_vertex: run_vertex.to_vec(),
            uv_a: uv_a.to_vec(),
            uv_b: uv_b.to_vec(),
            class: class.iter().map(|&c| c as u8).collect(),
            gap_px: gap_px.to_vec(),
            xyz: xyz.to_vec(),
        })
    }
    fn view(&self) -> uvsnap::Chains<'_> {
        uvsnap::Chains {
            seams: uvseam::SeamUv { run_indptr: &self.run_indptr, run_vertex: &self.run_vertex, uv_a: &self.uv_a, uv_b: &self.uv_b, class: &self.class, gap_px: &self.gap_px },
            xyz: &self.xyz,
        }
    }
}

fn uvsnap_tags(n: usize, nc: usize, vchain: &[i64], vparam: &[f64]) -> PyResult<()> {
    let sizes = (vchain.is_empty() && vparam.is_empty()) || (vchain.len() == n && vparam.len() == n);
    if sizes && vparam.iter().all(|x| x.is_finite()) && vchain.iter().all(|&c| c < nc as i64) {
        Ok(())
    } else {
        Err(PyValueError::new_err("vchain and vparam need one value per vertex (or none), naming chains that are there"))
    }
}

/// Seam snap (`uvsnap::snap`) of a result mesh onto the Soft chains: dict with points (moved), held, snapped, vchain,
/// vparam (per vertex), state, mean_dist, turns, path_edges (per chain), moved, relaxed, done. `max_move` per vertex
/// (empty: no limit); `vchain` / `vparam` earlier tags (empty: none); `rings` relaxed around snapped vertices.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uvsnap_snap<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    run_indptr: PyReadonlyArray1<'py, i64>,
    run_vertex: PyReadonlyArray1<'py, i64>,
    uv_a: PyReadonlyArray1<'py, f64>,
    uv_b: PyReadonlyArray1<'py, f64>,
    class: PyReadonlyArray1<'py, i64>,
    gap_px: PyReadonlyArray1<'py, f64>,
    xyz: PyReadonlyArray1<'py, f64>,
    max_move: PyReadonlyArray1<'py, f64>,
    vchain: PyReadonlyArray1<'py, i64>,
    vparam: PyReadonlyArray1<'py, f64>,
    rings: usize,
    stop: &StopFlag,
) -> PyResult<Bound<'py, PyDict>> {
    let (p, i, f) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    let faces = uvseam_mesh(p, i, f)?;
    let chains = UvsnapChains::read(slice(&run_indptr, "run_indptr")?, slice(&run_vertex, "run_vertex")?, slice(&uv_a, "uv_a")?, slice(&uv_b, "uv_b")?, slice(&class, "class")?, slice(&gap_px, "gap_px")?, slice(&xyz, "xyz")?)?;
    let (mm, vc, vp) = (slice(&max_move, "max_move")?, slice(&vchain, "vchain")?, slice(&vparam, "vparam")?);
    let n = p.len() / 3;
    if !(mm.is_empty() || mm.len() == n) || mm.iter().any(|x| x.is_nan() || *x < 0.0) {
        return Err(PyValueError::new_err("max_move needs one value >= 0 per vertex (or none)"));
    }
    uvsnap_tags(n, chains.class.len(), vc, vp)?;
    let r = py.detach(|| uvsnap::snap(p, &faces, &chains.view(), mm, vc, vp, rings, &stop.0));
    let d = PyDict::new(py);
    d.set_item("done", !r.stopped)?;
    d.set_item("moved", r.moved)?;
    d.set_item("relaxed", r.relaxed)?;
    d.set_item("points", r.points.into_pyarray(py))?;
    d.set_item("held", r.held.into_pyarray(py))?;
    d.set_item("snapped", r.snapped.into_pyarray(py))?;
    d.set_item("vchain", r.vchain.into_pyarray(py))?;
    d.set_item("vparam", r.vparam.into_pyarray(py))?;
    d.set_item("state", r.state.iter().map(|&x| x as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("mean_dist", r.mean_dist.into_pyarray(py))?;
    d.set_item("turns", r.turns.into_pyarray(py))?;
    d.set_item("path_edges", r.path_edges.into_pyarray(py))?;
    Ok(d)
}

/// Split leftovers (`uvsnap::split`): dict with the new mesh (points, indptr, flat, origin per new face), face_state
/// per input face, vchain / vparam per new vertex, corner_uv (2 per new corner, NaN off seams) and corner_side (-1 off
/// seams, 0 side A, 1 side B), and the counts. `budget` extra faces (negative: no limit); `seen_face` per input face
/// (empty: all seen).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uvsnap_split<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    run_indptr: PyReadonlyArray1<'py, i64>,
    run_vertex: PyReadonlyArray1<'py, i64>,
    uv_a: PyReadonlyArray1<'py, f64>,
    uv_b: PyReadonlyArray1<'py, f64>,
    class: PyReadonlyArray1<'py, i64>,
    gap_px: PyReadonlyArray1<'py, f64>,
    xyz: PyReadonlyArray1<'py, f64>,
    vchain: PyReadonlyArray1<'py, i64>,
    vparam: PyReadonlyArray1<'py, f64>,
    seen_face: PyReadonlyArray1<'py, f64>,
    budget: f64,
    stop: &StopFlag,
) -> PyResult<Bound<'py, PyDict>> {
    let (p, i, f) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    let faces = uvseam_mesh(p, i, f)?;
    let chains = UvsnapChains::read(slice(&run_indptr, "run_indptr")?, slice(&run_vertex, "run_vertex")?, slice(&uv_a, "uv_a")?, slice(&uv_b, "uv_b")?, slice(&class, "class")?, slice(&gap_px, "gap_px")?, slice(&xyz, "xyz")?)?;
    let (vc, vp, seen) = (slice(&vchain, "vchain")?, slice(&vparam, "vparam")?, slice(&seen_face, "seen_face")?);
    uvsnap_tags(p.len() / 3, chains.class.len(), vc, vp)?;
    if !(seen.is_empty() || seen.len() == faces.count()) || budget.is_nan() {
        return Err(PyValueError::new_err("seen_face needs one value per face (or none) and budget a number"));
    }
    let r = py.detach(|| uvsnap::split(p, &faces, &chains.view(), vc, vp, seen, budget, &stop.0));
    let d = PyDict::new(py);
    d.set_item("done", !r.stopped)?;
    for (k, v) in [("faces_split", r.faces_split), ("triangles_left", r.triangles_left), ("faces_left", r.faces_left), ("faces_cannot", r.faces_cannot), ("faces_bake", r.faces_bake)] {
        d.set_item(k, v)?;
    }
    d.set_item("extra_faces", r.extra_faces)?;
    d.set_item("spent", r.spent)?;
    d.set_item("wanted", r.wanted)?;
    d.set_item("points", r.points.into_pyarray(py))?;
    d.set_item("indptr", r.indptr.into_pyarray(py))?;
    d.set_item("flat", r.flat.into_pyarray(py))?;
    d.set_item("origin", r.origin.into_pyarray(py))?;
    d.set_item("face_state", r.face_state.iter().map(|&x| x as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("vchain", r.vchain.into_pyarray(py))?;
    d.set_item("vparam", r.vparam.into_pyarray(py))?;
    d.set_item("corner_uv", r.corner_uv.into_pyarray(py))?;
    d.set_item("corner_side", r.corner_side.iter().map(|&x| x as i64).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(d)
}

fn register_uvsnap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uvsnap_probe, m)?)?;
    m.add_function(wrap_pyfunction!(uvsnap_snap, m)?)?;
    m.add_function(wrap_pyfunction!(uvsnap_split, m)?)?;
    Ok(())
}
// main-only: uv end snap

// main-only: uv begin qem
mod uvqem;

/// Compile probe of the `uvqem` kernel: 1 when every helper it needs is there.
#[pyfunction]
fn uvqem_probe() -> usize {
    uvqem::probe()
}

fn register_uvqem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uvqem_probe, m)?)?;
    Ok(())
}
// main-only: uv end qem

// main-only: uv begin score
mod uvscore;

/// Compile probe of the `uvscore` kernel: 1 when every helper it needs is there.
#[pyfunction]
fn uvscore_probe() -> usize {
    uvscore::probe()
}

/// The card names of `uvscore_check`'s numbers, in order.
#[pyfunction]
fn uvscore_names() -> Vec<String> {
    uvscore::NAMES.iter().map(|s| s.to_string()).collect()
}

fn uvscore_mesh<'a>(points: &'a [f64], indptr: &'a [i64], flat: &'a [i64], uv: &'a [f64], material: &'a [i64], name: &str) -> PyResult<uvscore::Mesh<'a>> {
    check_faces(indptr, flat)?;
    let n = points.len() / 3;
    let faces = indptr.len() - 1;
    if points.len() % 3 != 0 || flat.iter().any(|&v| v < 0 || v as usize >= n) || points.iter().any(|x| !x.is_finite()) || uv.len() != 2 * flat.len() || !(material.is_empty() || material.len() == faces) {
        return Err(PyValueError::new_err(format!("{name} arrays are invalid")));
    }
    Ok(uvscore::Mesh { points, faces: topo::Faces { indptr, flat }, uv, material })
}

/// UV Check numbers (`uvscore::check`): the result mesh and, when `src_points` is not empty, the source it came from.
/// Every empty array means "not given". Returns (numbers in the order of `uvscore_names`, straddling faces, flipped faces).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn uvscore_check<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    uv: PyReadonlyArray1<'py, f64>,
    material: PyReadonlyArray1<'py, i64>,
    src_points: PyReadonlyArray1<'py, f64>,
    src_indptr: PyReadonlyArray1<'py, i64>,
    src_flat: PyReadonlyArray1<'py, i64>,
    src_uv: PyReadonlyArray1<'py, f64>,
    src_material: PyReadonlyArray1<'py, i64>,
    src_region: PyReadonlyArray1<'py, i64>,
    seen_face: PyReadonlyArray1<'py, f64>,
    seen_edge_px: PyReadonlyArray1<'py, f64>,
    allow_overlap: PyReadonlyArray1<'py, bool>,
    col: PyReadonlyArray1<'py, i64>,
    row: PyReadonlyArray1<'py, i64>,
    on_chart: PyReadonlyArray1<'py, bool>,
    density: PyReadonlyArray1<'py, f64>,
    faces_off: i64,
    split: PyReadonlyArray1<'py, bool>,
    texture_px: f64,
    stop: &StopFlag,
) -> PyResult<(F64Array<'py>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<bool>>)> {
    let res = uvscore_mesh(slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?, slice(&uv, "uv")?, slice(&material, "material")?, "result")?;
    let nf = res.faces.count();
    let src_given = !slice(&src_points, "src_points")?.is_empty();
    let src = if src_given {
        Some(uvscore_mesh(slice(&src_points, "src_points")?, slice(&src_indptr, "src_indptr")?, slice(&src_flat, "src_flat")?, slice(&src_uv, "src_uv")?, slice(&src_material, "src_material")?, "source")?)
    } else {
        None
    };
    let (seen_face, seen_edge_px, density) = (slice(&seen_face, "seen_face")?, slice(&seen_edge_px, "seen_edge_px")?, slice(&density, "density")?);
    let (col, row) = (slice(&col, "col")?, slice(&row, "row")?);
    let flags = |a: &PyReadonlyArray1<'py, bool>, name: &str| -> PyResult<Vec<u8>> {
        let v = slice(a, name)?;
        if !(v.is_empty() || v.len() == nf) {
            return Err(PyValueError::new_err(format!("{name} needs one value per result face")));
        }
        Ok(v.iter().map(|&b| u8::from(b)).collect())
    };
    let (allow, on, split) = (flags(&allow_overlap, "allow_overlap")?, flags(&on_chart, "on_chart")?, flags(&split, "split")?);
    if !(seen_face.is_empty() || seen_face.len() == nf) || !(density.is_empty() || density.len() == nf) {
        return Err(PyValueError::new_err("seen_face and density need one value per result face"));
    }
    let ex = uvscore::Extras { src_region: slice(&src_region, "src_region")?, seen_face, seen_edge_px, allow_overlap: &allow, col, row, on_chart: &on, density, faces_off, split: &split };
    let out = py.detach(|| uvscore::check(&res, src.as_ref(), &ex, texture_px, &stop.0));
    let out = out.ok_or_else(|| pyo3::exceptions::PyInterruptedError::new_err("stopped"))?;
    let flags = |v: Vec<u8>| v.into_iter().map(|x| x != 0).collect::<Vec<bool>>().into_pyarray(py);
    Ok((out.numbers.into_pyarray(py), flags(out.straddle), flags(out.flipped)))
}

fn register_uvscore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uvscore_probe, m)?)?;
    m.add_function(wrap_pyfunction!(uvscore_names, m)?)?;
    m.add_function(wrap_pyfunction!(uvscore_check, m)?)?;
    Ok(())
}
// main-only: uv end score

// main-only: uv begin sight
mod uvsight;

/// Compile probe of the `uvsight` kernel: 1 when every helper it needs is there.
#[pyfunction]
fn uvsight_probe() -> usize {
    uvsight::probe()
}

/// Pictures of one mesh from many directions, held between passes. Methods never wait for a running kernel: while
/// `uvsight_fill` or `uvsight_seen` works on it they raise RuntimeError("busy"); the counters are always readable.
#[pyclass(frozen)]
struct SightPictures {
    inner: std::sync::Mutex<uvsight::Pictures>,
    drawn: std::sync::atomic::AtomicUsize,
    renders: std::sync::atomic::AtomicUsize,
}

impl SightPictures {
    fn busy<T>(&self, work: impl FnOnce(&mut uvsight::Pictures) -> T) -> PyResult<T> {
        match self.inner.try_lock() {
            Ok(mut guard) => Ok(work(&mut guard)),
            Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err("busy")),
        }
    }
    fn counts(&self, p: &uvsight::Pictures) {
        self.drawn.store(p.drawn(), std::sync::atomic::Ordering::Relaxed);
        self.renders.store(p.renders, std::sync::atomic::Ordering::Relaxed);
    }
}

#[pymethods]
impl SightPictures {
    /// `pixels` 0 = about 4 px across the median edge, at least 512. Draws nothing yet.
    #[new]
    #[pyo3(signature = (points, indptr, flat, views=6, pixels=0, all_sides=true))]
    fn new<'py>(points: PyReadonlyArray1<'py, f64>, indptr: PyReadonlyArray1<'py, i64>, flat: PyReadonlyArray1<'py, i64>, views: usize, pixels: usize, all_sides: bool) -> PyResult<Self> {
        let (points, indptr, flat) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?);
        check_faces(indptr, flat)?;
        if flat.iter().any(|&v| v < 0 || v as usize * 3 + 3 > points.len()) {
            return Err(PyValueError::new_err("sight faces use a missing vertex"));
        }
        let pixels = if pixels == 0 { uvsight::default_pixels(points, &topo::Faces { indptr, flat }) } else { pixels };
        let dirs = uvsight::view_dirs(views);
        let weights = uvsight::view_weights(&dirs, all_sides);
        let pictures = uvsight::Pictures::new(points, indptr, flat, dirs, weights, pixels).map_err(PyValueError::new_err)?;
        Ok(SightPictures { inner: std::sync::Mutex::new(pictures), drawn: 0.into(), renders: 0.into() })
    }

    #[getter]
    fn drawn(&self) -> usize {
        self.drawn.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Views drawn since creation (stays put when `uvsight_seen` runs again).
    #[getter]
    fn renders(&self) -> usize {
        self.renders.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Make the view count `views` (new views are drawn by the next fill; drawn ones are kept) and set the weights.
    fn set_views(&self, views: usize, all_sides: bool) -> PyResult<()> {
        self.busy(|p| {
            let dirs = uvsight::view_dirs(views);
            let weights = uvsight::view_weights(&dirs, all_sides);
            if views < p.views() {
                p.dirs.truncate(views);
                p.weights.truncate(views);
                p.face.truncate(views);
                p.depth.truncate(views);
            } else {
                let at = p.views();
                p.extend(&dirs[at..], &weights[at..]).map_err(PyValueError::new_err)?;
            }
            p.set_weights(&weights);
            self.counts(p);
            Ok(())
        })?
    }

    fn set_all_sides(&self, all_sides: bool) -> PyResult<()> {
        self.busy(|p| {
            let weights = uvsight::view_weights(&p.dirs, all_sides);
            p.set_weights(&weights);
        })
    }

    /// (pixels, center xyz, half): the picture frame.
    fn frame(&self) -> PyResult<(usize, [f64; 3], f64)> {
        self.busy(|p| (p.pixels, p.center, p.half))
    }

    /// (directions flat xyz, weights).
    fn views<'py>(&self, py: Python<'py>) -> PyResult<(F64Array<'py>, F64Array<'py>)> {
        self.busy(|p| (p.dirs.iter().flatten().copied().collect::<Vec<f64>>().into_pyarray(py), p.weights.clone().into_pyarray(py)))
    }

    /// Face id per cell of one view (-1 empty); a view that is not drawn gives an empty array.
    fn face<'py>(&self, py: Python<'py>, view: usize) -> PyResult<I64Array<'py>> {
        let out = self.busy(|p| p.face.get(view).map(|f| f.iter().map(|&v| if v == u32::MAX { -1 } else { v as i64 }).collect::<Vec<i64>>()))?;
        out.map(|v| v.into_pyarray(py)).ok_or_else(|| PyValueError::new_err("no such view"))
    }

    /// Depth per cell of one view (smaller = nearer; 2 half minus the distance toward the camera).
    fn depth<'py>(&self, py: Python<'py>, view: usize) -> PyResult<F64Array<'py>> {
        let out = self.busy(|p| p.depth.get(view).map(|f| f.iter().map(|&v| v as f64).collect::<Vec<f64>>()))?;
        out.map(|v| v.into_pyarray(py)).ok_or_else(|| PyValueError::new_err("no such view"))
    }
}

/// Draw the views that are not drawn yet. A stop ends it with InterruptedError; what is drawn stays for the next call.
#[pyfunction]
fn uvsight_fill(py: Python<'_>, pictures: &SightPictures, stop: &StopFlag) -> PyResult<()> {
    let done = py.detach(|| {
        let mut p = pictures.inner.lock().map_err(|_| "poisoned")?;
        let ok = p.fill(&stop.0);
        pictures.counts(&p);
        Ok::<bool, &str>(ok)
    });
    finished(done.map_err(pyo3::exceptions::PyRuntimeError::new_err)?)
}

/// (seen per face, inside px per edge, outline px per edge) from the drawn pictures, with no drawing.
#[pyfunction]
fn uvsight_seen<'py>(py: Python<'py>, pictures: &SightPictures, stop: &StopFlag) -> PyResult<(F64Array<'py>, F64Array<'py>, F64Array<'py>)> {
    let found = py.detach(|| pictures.inner.lock().map(|p| p.sight(&stop.0)).map_err(|_| "poisoned"));
    let sight = found.map_err(pyo3::exceptions::PyRuntimeError::new_err)?.ok_or_else(|| pyo3::exceptions::PyInterruptedError::new_err("stopped"))?;
    Ok((sight.seen.seen_face.into_pyarray(py), sight.seen.seen_edge_px.into_pyarray(py), sight.outline_px.into_pyarray(py)))
}

/// The first `count` view directions (flat xyz) and their weights.
#[pyfunction]
fn uvsight_dirs<'py>(py: Python<'py>, count: usize, all_sides: bool) -> (F64Array<'py>, F64Array<'py>) {
    let dirs = uvsight::view_dirs(count);
    let weights = uvsight::view_weights(&dirs, all_sides);
    (dirs.iter().flatten().copied().collect::<Vec<f64>>().into_pyarray(py), weights.into_pyarray(py))
}

fn register_uvsight(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uvsight_probe, m)?)?;
    m.add_function(wrap_pyfunction!(uvsight_fill, m)?)?;
    m.add_function(wrap_pyfunction!(uvsight_seen, m)?)?;
    m.add_function(wrap_pyfunction!(uvsight_dirs, m)?)?;
    m.add_class::<SightPictures>()?;
    Ok(())
}
// main-only: uv end sight

// main-only: uv begin flat
mod uvflat;

/// Compile probe of the `uvflat` kernel: 1 when every helper it needs is there.
#[pyfunction]
fn uvflat_probe() -> usize {
    uvflat::probe()
}

fn uvflat_mesh<'a>(points: &'a [f64], indptr: &'a [i64], flat: &'a [i64]) -> PyResult<topo::Faces<'a>> {
    check_faces(indptr, flat)?;
    let n = points.len() / 3;
    if points.len() % 3 != 0 || flat.iter().any(|&v| v < 0 || v as usize >= n) || points.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("points or face vertex ids are invalid"));
    }
    Ok(topo::Faces { indptr, flat })
}

fn uvflat_stopped(stop: &StopFlag) -> PyResult<()> {
    finished(!stop.0.load(std::sync::atomic::Ordering::Relaxed))
}

/// Flatten every chart of a mesh in one call (`uvflat::flatten`): returns (UVs per corner, stats rows of
/// `uvflat::STATS_LEN`). rows: 0 Off, 1 Soft, 2 On, 3 Auto; start: 0 given, 1 LSCM pulled toward the given, 2 free.
/// A stop returns the flip-free state each chart had reached (STOPPED = 1 in its row), so Keep going can resume.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uvflat_flatten<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    chart: PyReadonlyArray1<'py, i64>,
    start: PyReadonlyArray1<'py, f64>,
    col: PyReadonlyArray1<'py, i64>,
    row: PyReadonlyArray1<'py, i64>,
    rows: u8,
    start_mode: u8,
    stretch_allowed: f64,
    steps: usize,
    max_factor_bytes: f64,
    texture_px: f64,
    stop: &StopFlag,
) -> PyResult<(F64Array<'py>, F64Array<'py>)> {
    let (points, indptr, flat) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    let (chart, start, col, row) = (slice(&chart, "chart")?, slice(&start, "start")?, slice(&col, "col")?, slice(&row, "row")?);
    let faces = uvflat_mesh(points, indptr, flat)?;
    let corners = flat.len();
    if chart.len() != faces.count() || !(start.is_empty() || start.len() == 2 * corners) || col.len() != row.len() || !(col.is_empty() || col.len() == corners) {
        return Err(PyValueError::new_err("flatten arrays are inconsistent"));
    }
    let rows = match rows {
        0 => uvflat::Rows::Off,
        1 => uvflat::Rows::Soft,
        2 => uvflat::Rows::On,
        _ => uvflat::Rows::Auto,
    };
    let start_mode = match start_mode {
        0 => uvflat::Start::Given,
        1 => uvflat::Start::Pull,
        _ => uvflat::Start::Free,
    };
    let opts = uvflat::FlatOpts { rows, start: start_mode, stretch_allowed, steps, max_factor_bytes, texture_px, force_iterative: false };
    let (uv, stats) = py.detach(|| uvflat::flatten(points, &faces, chart, start, col, row, &opts, &stop.0));
    Ok((uv.into_pyarray(py), stats.into_pyarray(py)))
}

/// Exact grid UVs of a pole-free quad chart (the given faces): (UVs per vertex, column, row), NaN / -1 for vertices
/// the faces do not use; None when the faces are not a grid.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn uvflat_grid_layout<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    stop: &StopFlag,
) -> PyResult<Option<(F64Array<'py>, I64Array<'py>, I64Array<'py>)>> {
    let (points, indptr, flat) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    let faces = uvflat_mesh(points, indptr, flat)?;
    let n = points.len() / 3;
    let found = py.detach(|| {
        let ids: Vec<u32> = (0..faces.count() as u32).collect();
        let (pts, ip, fl, global) = uvflat::sub_chart(points, &faces, &ids);
        let g = uvflat::grid_layout(&uvflat::Chart { points: &pts, faces: topo::Faces { indptr: &ip, flat: &fl } }).ok()?;
        let (mut uv, mut col, mut row) = (vec![f64::NAN; 2 * n], vec![-1i64; n], vec![-1i64; n]);
        for (k, &v) in global.iter().enumerate() {
            let v = v as usize;
            uv[2 * v] = g.uv[2 * k];
            uv[2 * v + 1] = g.uv[2 * k + 1];
            col[v] = g.col[k];
            row[v] = g.row[k];
        }
        Some((uv, col, row))
    });
    uvflat_stopped(stop)?;
    Ok(found.map(|(uv, col, row)| (uv.into_pyarray(py), col.into_pyarray(py), row.into_pyarray(py))))
}

/// Stretch of a merge candidate (`uvflat::quick_stretch`): the given faces, UVs per vertex (2 per vertex of
/// `points`), `free` per vertex (the band). Returns (band stretch, chart stretch, UVs per vertex after the band moved).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uvflat_quick_stretch<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    uv: PyReadonlyArray1<'py, f64>,
    free: PyReadonlyArray1<'py, bool>,
    max_factor_bytes: f64,
    stop: &StopFlag,
) -> PyResult<(f64, f64, F64Array<'py>)> {
    let (points, indptr, flat) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    let (uv, free) = (slice(&uv, "uv")?, slice(&free, "free")?);
    let faces = uvflat_mesh(points, indptr, flat)?;
    let n = points.len() / 3;
    if uv.len() != 2 * n || free.len() != n {
        return Err(PyValueError::new_err("quick_stretch arrays are inconsistent"));
    }
    let found = py.detach(|| {
        let ids: Vec<u32> = (0..faces.count() as u32).collect();
        let (pts, ip, fl, global) = uvflat::sub_chart(points, &faces, &ids);
        let mut luv: Vec<f64> = global.iter().flat_map(|&v| [uv[2 * v as usize], uv[2 * v as usize + 1]]).collect();
        let lfree: Vec<bool> = global.iter().map(|&v| free[v as usize]).collect();
        let chart = uvflat::Chart { points: &pts, faces: topo::Faces { indptr: &ip, flat: &fl } };
        let (band, all) = uvflat::quick_stretch(&chart, &mut luv, &lfree, max_factor_bytes, &stop.0)?;
        let mut out = uv.to_vec();
        for (k, &v) in global.iter().enumerate() {
            out[2 * v as usize] = luv[2 * k];
            out[2 * v as usize + 1] = luv[2 * k + 1];
        }
        Ok::<_, String>((band, all, out))
    });
    uvflat_stopped(stop)?;
    let (band, all, out) = found.map_err(PyValueError::new_err)?;
    Ok((band, all, out.into_pyarray(py)))
}

/// Chart stretch (`uvflat::stretch`) of the given faces with UVs per vertex: (quantile, per face).
#[pyfunction]
fn uvflat_stretch<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    flat: PyReadonlyArray1<'py, i64>,
    uv: PyReadonlyArray1<'py, f64>,
    stop: &StopFlag,
) -> PyResult<(f64, F64Array<'py>)> {
    let (points, indptr, flat, uv) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?, slice(&uv, "uv")?);
    let faces = uvflat_mesh(points, indptr, flat)?;
    if uv.len() != 2 * (points.len() / 3) {
        return Err(PyValueError::new_err("uv must hold 2 numbers per point"));
    }
    let (q, per) = py.detach(|| uvflat::stretch(&uvflat::Chart { points, faces }, uv));
    uvflat_stopped(stop)?;
    Ok((q, per.into_pyarray(py)))
}

fn register_uvflat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uvflat_probe, m)?)?;
    m.add_function(wrap_pyfunction!(uvflat_flatten, m)?)?;
    m.add_function(wrap_pyfunction!(uvflat_grid_layout, m)?)?;
    m.add_function(wrap_pyfunction!(uvflat_quick_stretch, m)?)?;
    m.add_function(wrap_pyfunction!(uvflat_stretch, m)?)?;
    m.add("UVFLAT_STATS_LEN", uvflat::STATS_LEN)?;
    Ok(())
}
// main-only: uv end flat

// main-only: uv begin chart
mod uvchart;

/// Compile probe of the `uvchart` kernel: 1 when every helper it needs is there.
#[pyfunction]
fn uvchart_probe() -> usize {
    uvchart::probe()
}

type BoolArray<'py> = Bound<'py, PyArray1<bool>>;

/// The Row Charts tree of one quad result (`uvchart::Tree`): patches, segments and the merge tree. `level(x)` is
/// instant after the build.
#[pyclass(frozen)]
struct UvChartTree(uvchart::Tree);

#[pymethods]
impl UvChartTree {
    /// `uvchart::num` columns: patches, irregular, turning, traces, Euler-cut patches and edges, ..., build ms.
    fn numbers<'py>(&self, py: Python<'py>) -> F64Array<'py> {
        self.0.numbers.to_vec().into_pyarray(py)
    }

    /// Patch per face.
    fn patches<'py>(&self, py: Python<'py>) -> I64Array<'py> {
        self.0.patch.iter().map(|&p| p as i64).collect::<Vec<_>>().into_pyarray(py)
    }

    /// Per edge in `seen_edge_px` order: (a, b, patch border, never glued, cut cost).
    #[allow(clippy::type_complexity)]
    fn edges<'py>(&self, py: Python<'py>) -> (I64Array<'py>, I64Array<'py>, BoolArray<'py>, BoolArray<'py>, F64Array<'py>) {
        let t = &self.0.topo;
        (
            t.ea.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py),
            t.eb.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py),
            self.0.sep.clone().into_pyarray(py),
            self.0.fixed.clone().into_pyarray(py),
            self.0.cost.clone().into_pyarray(py),
        )
    }

    /// Per vertex: 0 regular, 1 irregular, 2 turning vertex of a forced line.
    fn irregular<'py>(&self, py: Python<'py>) -> I64Array<'py> {
        (0..self.0.topo.nv).map(|v| if self.0.turning[v] { 2 } else { i64::from(self.0.irregular[v]) }).collect::<Vec<_>>().into_pyarray(py)
    }

    /// Per segment: (patch p, patch q, cost, seen px, edge ids CSR indptr, edge ids).
    #[allow(clippy::type_complexity)]
    fn segments<'py>(&self, py: Python<'py>) -> (I64Array<'py>, I64Array<'py>, F64Array<'py>, F64Array<'py>, I64Array<'py>, I64Array<'py>) {
        let sg = &self.0.segments;
        let mut ptr = vec![0i64];
        let mut ids = Vec::new();
        for s in sg {
            ids.extend(s.edges.iter().map(|&e| e as i64));
            ptr.push(ids.len() as i64);
        }
        (
            sg.iter().map(|s| s.p as i64).collect::<Vec<_>>().into_pyarray(py),
            sg.iter().map(|s| s.q as i64).collect::<Vec<_>>().into_pyarray(py),
            sg.iter().map(|s| s.cost).collect::<Vec<_>>().into_pyarray(py),
            sg.iter().map(|s| s.seen).collect::<Vec<_>>().into_pyarray(py),
            ptr.into_pyarray(py),
            ids.into_pyarray(py),
        )
    }

    /// Per recorded merge, in merge order: (resulting stretch, cost removed, pole union, segments CSR indptr, ids).
    #[allow(clippy::type_complexity)]
    fn merges<'py>(&self, py: Python<'py>) -> (F64Array<'py>, F64Array<'py>, BoolArray<'py>, I64Array<'py>, I64Array<'py>) {
        let ms = &self.0.merges;
        let mut ptr = vec![0i64];
        let mut ids = Vec::new();
        for m in ms {
            ids.extend(m.segs.iter().map(|&s| s as i64));
            ptr.push(ids.len() as i64);
        }
        (
            ms.iter().map(|m| m.stretch).collect::<Vec<_>>().into_pyarray(py),
            ms.iter().map(|m| m.cost).collect::<Vec<_>>().into_pyarray(py),
            ms.iter().map(|m| m.pole).collect::<Vec<_>>().into_pyarray(py),
            ptr.into_pyarray(py),
            ids.into_pyarray(py),
        )
    }

    /// The charts at Stretch allowed `x` percent (cached): (chart per face, cut per edge, [charts, not disk, merges
    /// used, segments glued, above the build stretch, ms]).
    fn level<'py>(&self, py: Python<'py>, x: f64) -> (I64Array<'py>, BoolArray<'py>, F64Array<'py>) {
        uvchart_level_out(py, &self.0, x)
    }
}

fn uvchart_level_out<'py>(py: Python<'py>, tree: &uvchart::Tree, x: f64) -> (I64Array<'py>, BoolArray<'py>, F64Array<'py>) {
    let l = py.detach(|| tree.level(x));
    let numbers = vec![l.charts as f64, l.not_disk as f64, l.merges_used as f64, l.segments_glued as f64, f64::from(u8::from(l.above_build)), l.ms];
    (l.chart.iter().map(|&c| c as i64).collect::<Vec<_>>().into_pyarray(py), l.cut.clone().into_pyarray(py), numbers.into_pyarray(py))
}

/// Build the Row Charts tree (`uvchart::Tree::build`): separatrices, patches, Euler check, segments, merge tree.
/// `part_label` per face (empty = one part; materials folded in by the caller), `forced` vertex pairs, `seen_edge_px`
/// per edge from uvsight (empty = all seen). A stop returns the tree reached so far (numbers[STOPPED] = 1).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uvchart_build(
    py: Python<'_>,
    points: PyReadonlyArray1<'_, f64>,
    indptr: PyReadonlyArray1<'_, i64>,
    flat: PyReadonlyArray1<'_, i64>,
    part_label: PyReadonlyArray1<'_, i64>,
    forced: PyReadonlyArray1<'_, i64>,
    seen_edge_px: PyReadonlyArray1<'_, f64>,
    px_per_unit: f64,
    out_of_sight: bool,
    hidden_l: f64,
    build_stretch: f64,
    stop: &StopFlag,
) -> PyResult<UvChartTree> {
    let (points, indptr, flat) = (slice(&points, "points")?, slice(&indptr, "indptr")?, slice(&flat, "flat")?);
    let (part_label, forced, seen) = (slice(&part_label, "part_label")?, slice(&forced, "forced")?, slice(&seen_edge_px, "seen_edge_px")?);
    let faces = uvflat_mesh(points, indptr, flat)?;
    if part_label.iter().any(|&p| p < 0 || p > u32::MAX as i64) {
        return Err(PyValueError::new_err("part labels must be 0 .. 2^32-1"));
    }
    let labels: Vec<u32> = part_label.iter().map(|&p| p as u32).collect();
    let input = uvchart::Input {
        points,
        faces,
        part_label: &labels,
        forced,
        seen_edge_px: if seen.is_empty() { None } else { Some(seen) },
        px_per_unit,
        out_of_sight,
        hidden_l,
        build_stretch,
    };
    let tree = py.detach(|| uvchart::Tree::build(&input, &stop.0)).map_err(PyValueError::new_err)?;
    Ok(UvChartTree(tree))
}

/// `UvChartTree.level` as a module function, so it can run as a job.
#[pyfunction]
fn uvchart_level<'py>(py: Python<'py>, tree: &UvChartTree, x: f64, stop: &StopFlag) -> PyResult<(I64Array<'py>, BoolArray<'py>, F64Array<'py>)> {
    let out = uvchart_level_out(py, &tree.0, x);
    uvflat_stopped(stop)?;
    Ok(out)
}

fn uvchart_opts(rows: u8, stretch_allowed: f64, steps: usize, max_factor_bytes: f64, texture_px: f64) -> uvflat::FlatOpts {
    let rows = match rows {
        0 => uvflat::Rows::Off,
        1 => uvflat::Rows::Soft,
        2 => uvflat::Rows::On,
        _ => uvflat::Rows::Auto,
    };
    uvflat::FlatOpts { rows, stretch_allowed, steps, max_factor_bytes, texture_px, ..Default::default() }
}

/// New UVs of a level (6.5): (UVs per corner, chart per face, stats rows of `uvflat::STATS_LEN` per chart, stretch
/// per face).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn uvchart_uvs<'py>(py: Python<'py>, tree: &UvChartTree, x: f64, rows: u8, stretch_allowed: f64, steps: usize, max_factor_bytes: f64, texture_px: f64, stop: &StopFlag) -> PyResult<(F64Array<'py>, I64Array<'py>, F64Array<'py>, F64Array<'py>)> {
    let opts = uvchart_opts(rows, stretch_allowed, steps, max_factor_bytes, texture_px);
    let (uv, chart, stats, ratio) = py.detach(|| {
        let l = tree.0.level(x);
        let (uv, stats, ratio) = tree.0.flatten_cut(&l.cut, &l.chart, None, Some(&tree.0.start_uv), &opts, &stop.0);
        (uv, l.chart.iter().map(|&c| c as i64).collect::<Vec<_>>(), stats, ratio)
    });
    uvflat_stopped(stop)?;
    Ok((uv.into_pyarray(py), chart.into_pyarray(py), stats.into_pyarray(py), ratio.into_pyarray(py)))
}

/// Keep going state for New UVs (`uvchart::Cuts`).
#[pyclass(frozen)]
struct UvChartCuts(std::sync::Mutex<uvchart::Cuts>);

#[pymethods]
impl UvChartCuts {
    /// (UVs per corner, chart per face, cut per edge, stretch per chart id, score, passes, moves).
    #[allow(clippy::type_complexity)]
    fn state<'py>(&self, py: Python<'py>) -> (F64Array<'py>, I64Array<'py>, BoolArray<'py>, F64Array<'py>, f64, usize, usize) {
        let c = self.0.lock().unwrap();
        (
            c.uv.clone().into_pyarray(py),
            c.chart.iter().map(|&x| x as i64).collect::<Vec<_>>().into_pyarray(py),
            c.cut.clone().into_pyarray(py),
            c.chart_str.clone().into_pyarray(py),
            c.score,
            c.passes,
            c.moves,
        )
    }
}

/// Start Keep going from level x: its charts flattened.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uvchart_cuts(py: Python<'_>, tree: &UvChartTree, x: f64, rows: u8, stretch_allowed: f64, steps: usize, max_factor_bytes: f64, texture_px: f64, stop: &StopFlag) -> PyResult<UvChartCuts> {
    let opts = uvchart_opts(rows, stretch_allowed, steps, max_factor_bytes, texture_px);
    let cuts = py.detach(|| tree.0.start_cuts(x, &opts, &stop.0));
    uvflat_stopped(stop)?;
    Ok(UvChartCuts(std::sync::Mutex::new(cuts)))
}

/// One Keep going pass: (kind 0 none / 1 cut added / 2 cut closed, accepted, score before, score after, ms).
/// Not accepted = the pass gained nothing and Keep going stops by itself.
#[pyfunction]
fn uvchart_pass(py: Python<'_>, tree: &UvChartTree, cuts: &UvChartCuts, stop: &StopFlag) -> PyResult<(u8, bool, f64, f64, f64)> {
    let p = py.detach(|| {
        let mut c = cuts.0.lock().unwrap();
        tree.0.pass(&mut c, &stop.0)
    });
    Ok((p.kind, p.accepted, p.before, p.after, p.ms))
}

fn register_uvchart(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uvchart_probe, m)?)?;
    m.add_class::<UvChartTree>()?;
    m.add_class::<UvChartCuts>()?;
    m.add_function(wrap_pyfunction!(uvchart_build, m)?)?;
    m.add_function(wrap_pyfunction!(uvchart_level, m)?)?;
    m.add_function(wrap_pyfunction!(uvchart_uvs, m)?)?;
    m.add_function(wrap_pyfunction!(uvchart_cuts, m)?)?;
    m.add_function(wrap_pyfunction!(uvchart_pass, m)?)?;
    m.add("UVCHART_NUMBERS_LEN", uvchart::NUMBERS_LEN)?;
    m.add("UVCHART_STRETCH_ALLOWED_PCT", uvchart::STRETCH_ALLOWED_PCT)?;
    Ok(())
}
// main-only: uv end chart

// main-only: uv begin pack
mod uvpack;

/// Compile probe of the `uvpack` kernel: 1 when every helper it needs is there.
#[pyfunction]
fn uvpack_probe() -> usize {
    uvpack::probe()
}

fn pack_stopped(stop: &StopFlag) -> PyResult<()> {
    finished(!stop.0.load(std::sync::atomic::Ordering::Relaxed))
}

fn pack_flags(a: &[bool]) -> Vec<u8> {
    a.iter().map(|&b| u8::from(b)).collect()
}

fn pack_ints(a: &[usize]) -> Vec<i64> {
    a.iter().map(|&i| i as i64).collect()
}

/// Fill one page with boxes (`uvpack::fill_page`): the largest scale at which the boxes (unit w x h) fit a
/// `page_w` x `page_h` texel page with the mip-`k` gap (k < 0: no gap), and Keep going on it. `order` (a permutation,
/// empty = none), `policy` and `scale` resume an earlier call. `passes` = swap passes after the variants (0 none, negative until a pass gains nothing). `budget` seconds,
/// 0 = none. Returns (info, x, y, turned, order); info = [scale, used, policy, passes, budget_hit, stopped, ok].
/// A stop returns the best layout so far (never an error), so Keep going can resume.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn uvpack_fill_page<'py>(
    py: Python<'py>,
    w: PyReadonlyArray1<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    turn_ok: PyReadonlyArray1<'py, bool>,
    weight: PyReadonlyArray1<'py, f64>,
    page_w: i64,
    page_h: i64,
    k: i64,
    order: PyReadonlyArray1<'py, i64>,
    policy: usize,
    scale: f64,
    passes: i64,
    budget: f64,
    stop: &StopFlag,
) -> PyResult<(F64Array<'py>, I64Array<'py>, I64Array<'py>, I64Array<'py>, I64Array<'py>)> {
    let (w, h, weight, order) = (slice(&w, "w")?, slice(&h, "h")?, slice(&weight, "weight")?, slice(&order, "order")?);
    let turn_ok = pack_flags(slice(&turn_ok, "turn_ok")?);
    let n = w.len();
    if h.len() != n || turn_ok.len() != n || weight.len() != n || !(order.is_empty() || order.len() == n) || page_w < 1 || page_h < 1 {
        return Err(PyValueError::new_err("pack arrays are inconsistent"));
    }
    if w.iter().chain(h).chain(weight).any(|x| !x.is_finite() || *x < 0.0) {
        return Err(PyValueError::new_err("box sizes must be finite and not negative"));
    }
    let start: Vec<usize> = order.iter().map(|&i| i as usize).collect();
    let mut seen = vec![false; n];
    if start.iter().any(|&i| i >= n || std::mem::replace(&mut seen[i], true)) {
        return Err(PyValueError::new_err("order is not a permutation"));
    }
    let boxes = uvpack::Boxes { w, h, turn_ok: &turn_ok, weight };
    let found = py.detach(|| {
        let start = (!start.is_empty()).then(|| (start.as_slice(), policy, scale));
        uvpack::fill_page(&boxes, (page_w, page_h), k, start, passes, budget, &stop.0)
    });
    let Some(f) = found else {
        let stopped = f64::from(stop.0.load(std::sync::atomic::Ordering::Relaxed));
        let zeros = || vec![0i64; n].into_pyarray(py);
        return Ok((vec![0.0, 0.0, 0.0, 0.0, 0.0, stopped, 0.0].into_pyarray(py), zeros(), zeros(), zeros(), (0..n as i64).collect::<Vec<_>>().into_pyarray(py)));
    };
    let info = vec![f.scale, f.used, f.pol as f64, f.passes as f64, f64::from(f.budget_hit), f64::from(f.stopped), 1.0];
    let x: Vec<i64> = f.places.iter().map(|p| p.x).collect();
    let y: Vec<i64> = f.places.iter().map(|p| p.y).collect();
    let t: Vec<i64> = f.places.iter().map(|p| i64::from(p.turned)).collect();
    Ok((info.into_pyarray(py), x.into_pyarray(py), y.into_pyarray(py), t.into_pyarray(py), pack_ints(&f.order).into_pyarray(py)))
}

/// Pack boxes at one fixed scale in the given order (`uvpack::pack`): (ok, x, y, turned). Nothing placed when ok is false.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn uvpack_pack_boxes<'py>(
    py: Python<'py>,
    w: PyReadonlyArray1<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    turn_ok: PyReadonlyArray1<'py, bool>,
    order: PyReadonlyArray1<'py, i64>,
    policy: usize,
    scale: f64,
    page_w: i64,
    page_h: i64,
    k: i64,
    stop: &StopFlag,
) -> PyResult<(bool, I64Array<'py>, I64Array<'py>, I64Array<'py>)> {
    let (w, h, order) = (slice(&w, "w")?, slice(&h, "h")?, slice(&order, "order")?);
    let turn_ok = pack_flags(slice(&turn_ok, "turn_ok")?);
    let n = w.len();
    let order: Vec<usize> = order.iter().map(|&i| i as usize).collect();
    if h.len() != n || turn_ok.len() != n || order.len() != n || order.iter().any(|&i| i >= n) || page_w < 1 || page_h < 1 || !(scale > 0.0 && scale.is_finite()) {
        return Err(PyValueError::new_err("pack arrays are inconsistent"));
    }
    let weight = vec![1.0; n];
    let boxes = uvpack::Boxes { w, h, turn_ok: &turn_ok, weight: &weight };
    let placed = py.detach(|| uvpack::pack(&boxes, &order, policy.min(uvpack::POLICIES - 1), scale, (page_w, page_h), k));
    pack_stopped(stop)?;
    let (ok, p) = match placed {
        Some(p) => (true, p),
        None => (false, vec![uvpack::Place { x: 0, y: 0, turned: false }; n]),
    };
    let x: Vec<i64> = p.iter().map(|q| q.x).collect();
    let y: Vec<i64> = p.iter().map(|q| q.y).collect();
    let t: Vec<i64> = p.iter().map(|q| i64::from(q.turned)).collect();
    Ok((ok, x.into_pyarray(py), y.into_pyarray(py), t.into_pyarray(py)))
}

/// Pack charts (triangles in texels, `tri_chart` per triangle) into a page that may already hold `seed` triangles
/// (`uvpack::pack_into`). Returns (ok, dx, dy, quarter turns) per chart: add (dx, dy) after turning the chart's points
/// `quarter` x 90 degrees, (x, y) -> (-y, x). A stop returns what was placed so far (the rest ok = 0).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn uvpack_pack_into<'py>(
    py: Python<'py>,
    seed: PyReadonlyArray1<'py, f64>,
    tris: PyReadonlyArray1<'py, f64>,
    tri_chart: PyReadonlyArray1<'py, i64>,
    charts: usize,
    as_box: PyReadonlyArray1<'py, bool>,
    turn_ok: PyReadonlyArray1<'py, bool>,
    page_w: i64,
    page_h: i64,
    k: i64,
    stop: &StopFlag,
) -> PyResult<(I64Array<'py>, I64Array<'py>, I64Array<'py>, I64Array<'py>)> {
    let (seed, tris, tri_chart) = (slice(&seed, "seed")?, slice(&tris, "tris")?, slice(&tri_chart, "tri_chart")?);
    let (as_box, turn_ok) = (pack_flags(slice(&as_box, "as_box")?), pack_flags(slice(&turn_ok, "turn_ok")?));
    if seed.len() % 6 != 0 || tris.len() % 6 != 0 || tri_chart.len() * 6 != tris.len() || as_box.len() != charts || turn_ok.len() != charts || page_w < 1 || page_h < 1 {
        return Err(PyValueError::new_err("pack arrays are inconsistent"));
    }
    if seed.iter().chain(tris).any(|x| !x.is_finite() || x.abs() > 1.0e9) {
        return Err(PyValueError::new_err("triangle corners must be finite texel positions"));
    }
    let placed = py.detach(|| uvpack::pack_into((page_w, page_h), k, seed, tris, tri_chart, charts, &as_box, &turn_ok, &stop.0));
    let ok: Vec<i64> = placed.iter().map(|p| i64::from(p.ok)).collect();
    let dx: Vec<i64> = placed.iter().map(|p| p.dx).collect();
    let dy: Vec<i64> = placed.iter().map(|p| p.dy).collect();
    let q: Vec<i64> = placed.iter().map(|p| p.quarter as i64).collect();
    Ok((ok.into_pyarray(py), dx.into_pyarray(py), dy.into_pyarray(py), q.into_pyarray(py)))
}

fn register_uvpack(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uvpack_probe, m)?)?;
    m.add_function(wrap_pyfunction!(uvpack_fill_page, m)?)?;
    m.add_function(wrap_pyfunction!(uvpack_pack_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(uvpack_pack_into, m)?)?;
    m.add("UVPACK_ORDERS", uvpack::ORDERS)?;
    m.add("UVPACK_POLICIES", uvpack::POLICIES)?;
    Ok(())
}
// main-only: uv end pack

#[pymodule]
fn _curiomesh_turbo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("threads", rayon::current_num_threads())?;
    m.add_function(wrap_pyfunction!(topology_counts, m)?)?;
    m.add_function(wrap_pyfunction!(degenerate_faces, m)?)?;
    m.add_function(wrap_pyfunction!(edge_occurrences, m)?)?;
    m.add_function(wrap_pyfunction!(quad_closure, m)?)?;
    m.add_function(wrap_pyfunction!(parity_path_edges, m)?)?;
    m.add_function(wrap_pyfunction!(augment_matching, m)?)?;
    m.add_function(wrap_pyfunction!(winding_number, m)?)?;
    m.add_function(wrap_pyfunction!(pictures, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(quad_points_valid, m)?)?;
    m.add_function(wrap_pyfunction!(parity_templates, m)?)?;
    m.add_function(wrap_pyfunction!(cross_field_jacobi, m)?)?;
    m.add_function(wrap_pyfunction!(cross_field_gl, m)?)?;
    m.add_function(wrap_pyfunction!(make_closable, m)?)?;
    m.add_function(wrap_pyfunction!(close_pieces, m)?)?;
    m.add_function(wrap_pyfunction!(join_loose_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(shuffled, m)?)?;
    m.add_function(wrap_pyfunction!(pair_seeds, m)?)?;
    m.add_function(wrap_pyfunction!(leader_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_checks, m)?)?;
    m.add_function(wrap_pyfunction!(closed_walk, m)?)?;
    m.add_function(wrap_pyfunction!(stop_running, m)?)?;
    m.add_function(wrap_pyfunction!(corner_normals, m)?)?;
    m.add_function(wrap_pyfunction!(shifted_faces, m)?)?;
    m.add_function(wrap_pyfunction!(segment_merges, m)?)?;
    m.add_function(wrap_pyfunction!(field_orient_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(field_nearest_points, m)?)?;
    m.add_function(wrap_pyfunction!(field_position_sweep, m)?)?;
    m.add_class::<PictureJudgeCore>()?;
    m.add_class::<StopFlag>()?;
    m.add_class::<PointLookup>()?;
    m.add_class::<FaceLookup>()?;
    m.add_class::<Reducer>()?;
    m.add_class::<QuadReducer>()?;
    m.add_class::<QuadSaved>()?;
    // main-only: uv begin init
    register_uvseam(m)?;
    register_uvsnap(m)?;
    register_uvqem(m)?;
    register_uvscore(m)?;
    register_uvsight(m)?;
    register_uvflat(m)?;
    register_uvchart(m)?;
    register_uvpack(m)?;
    // main-only: uv end init
    Ok(())
}
