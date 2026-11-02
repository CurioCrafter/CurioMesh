// SPDX-License-Identifier: GPL-3.0-or-later
//! CurioMesh Turbo: native kernels for the CurioMesh add-on. The Python side
//! (`turbo.py`) converts meshes to flat arrays and falls back to pure Python
//! whenever this module is missing. All heavy work runs with the GIL released.

mod geom;
mod matching;
mod topo;

use numpy::{Complex64, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    Ok(py.detach(|| matching::augment(indptr, indices, order, mate)))
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

#[pyfunction]
#[allow(clippy::type_complexity)]
fn pictures<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f64>,
    triangles: PyReadonlyArray1<'py, i64>,
    normals: PyReadonlyArray1<'py, f64>,
    center: [f64; 3],
    half: f64,
    pixels: usize,
) -> PyResult<(Bound<'py, PyArray1<bool>>, F64Array<'py>, F64Array<'py>, Bound<'py, PyArray1<i64>>)> {
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
    let (hit, normal, depth, face) =
        py.detach(|| geom::pictures(points, triangles, normals, center, half, pixels));
    Ok((
        hit.into_pyarray(py),
        normal.into_pyarray(py),
        depth.into_pyarray(py),
        face.into_pyarray(py),
    ))
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
    m.add_function(wrap_pyfunction!(quad_points_valid, m)?)?;
    m.add_function(wrap_pyfunction!(parity_templates, m)?)?;
    m.add_function(wrap_pyfunction!(cross_field_jacobi, m)?)?;
    m.add_function(wrap_pyfunction!(cross_field_gl, m)?)?;
    Ok(())
}
