# SPDX-License-Identifier: GPL-3.0-or-later
"""CurioMesh Turbo: optional native (Rust) kernels with pure-Python fallbacks.

The compiled module ships as ``_native/<platform tag>/_curiomesh_turbo.*`` next
to this file, one stable-ABI (abi3, Python >= 3.10) binary per OS/arch, so a
single add-on zip serves every Blender 3.6 - 5.x on every platform. Loading
never raises: a missing, blocked or broken binary is logged once and every
wrapper returns ``None``, which tells the caller to run its Python path.

Set ``CURIOMESH_TURBO=0`` (or call ``set_enabled(False)``) to force Python.
Wrappers take and return exactly what their Python counterparts do.
"""

from __future__ import annotations

import gc
import importlib.machinery
import importlib.util
from itertools import chain
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


_log = logging.getLogger(__name__)
MODULE_NAME = "_curiomesh_turbo"
NATIVE_DIR = Path(__file__).resolve().parent / "_native"

_module: Any = None
_tried = False
_error = ""
_enabled = True


def platform_tag() -> str:
    """Folder name for this interpreter: win_amd64, linux_x86_64, macos_arm64, ..."""

    if sys.platform == "win32":  # platform.machine() queries WMI on Python 3.12+ (~1 s)
        machine = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "")
    else:
        machine = os.uname().machine
    machine = machine.lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    if sys.platform == "win32":
        return {"x86_64": "win_amd64", "arm64": "win_arm64"}.get(machine, f"win_{machine}")
    if sys.platform == "darwin":
        return f"macos_{machine}"
    return f"linux_{'aarch64' if machine == 'arm64' else machine}"


def _load() -> Any:
    global _module, _tried, _error
    if _tried:
        return _module
    _tried = True
    folder = NATIVE_DIR / platform_tag()
    try:
        candidates = sorted(p for p in folder.glob(f"{MODULE_NAME}*") if p.suffix in {".pyd", ".so"})
        if not candidates:
            raise ImportError(f"no native build for {platform_tag()} in {folder}")
        # The add-on can be loaded under several package names; reuse one copy.
        module = sys.modules.get(MODULE_NAME)
        if module is None:
            loader = importlib.machinery.ExtensionFileLoader(MODULE_NAME, str(candidates[0]))
            spec = importlib.util.spec_from_file_location(MODULE_NAME, candidates[0], loader=loader)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            sys.modules[MODULE_NAME] = module
        _module = module
    except Exception as exc:  # ImportError, OSError (blocked/quarantined DLL), anything else
        _error = f"{type(exc).__name__}: {exc}"
        _log.info("CurioMesh Turbo unavailable, using Python: %s", _error)
    return _module


def set_enabled(flag: bool) -> None:
    """Preference hook: False forces the Python paths (same as CURIOMESH_TURBO=0)."""

    global _enabled
    _enabled = bool(flag)


def native() -> Any:
    """The native module, or None when disabled or not loadable."""

    if not _enabled or os.environ.get("CURIOMESH_TURBO", "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    return _load()


def available() -> bool:
    return native() is not None


def version() -> str | None:
    module = native()
    return None if module is None else str(module.__version__)


def status() -> str:
    """One line for the UI / logs."""

    module = native()
    if module is not None:
        return f"Turbo {module.__version__} ({platform_tag()}, {module.threads} threads)"
    if not _enabled or os.environ.get("CURIOMESH_TURBO", "1").strip().lower() in {"0", "false", "off", "no"}:
        return "Turbo disabled (Python)"
    return f"Turbo unavailable (Python): {_error}"


# --------------------------------------------------------------------------- helpers


def _topology():
    try:
        from .triadq import topology
    except ImportError:
        from triadq import topology
    return topology


def _csr(faces) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.fromiter(map(len, faces), dtype=np.int64, count=len(faces))
    indptr = np.zeros(len(faces) + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    flat = np.fromiter(chain.from_iterable(faces), dtype=np.int64, count=int(indptr[-1]))
    return indptr, flat


def _edge_array(edges) -> np.ndarray:
    return np.asarray(sorted(tuple(e) for e in edges), dtype=np.int64).reshape(-1)


# --------------------------------------------------------------------------- topology


def topology_counts(faces) -> dict | None:
    """Combinatorial half of `topology_summary` as a dict (no geometry)."""

    module = native()
    if module is None or not faces:
        return None
    indptr, flat = _csr(faces)
    if flat.min() < 0:
        return None
    return module.topology_counts(indptr, flat)


def topology_summary(mesh):
    """Native `triadq.topology.topology_summary`; None means "use Python"."""

    module = native()
    if module is None or not mesh.faces:
        return None
    T = _topology()
    indptr, flat = _csr(mesh.faces)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    if flat.min() < 0 or flat.max() >= len(vertices):
        return None  # let Python produce its own error / semantics
    counts = module.topology_counts(indptr, flat)
    epsilon = T._span_epsilon(vertices)
    degenerate = module.degenerate_faces(indptr, flat, vertices.reshape(-1), epsilon)
    sizes = np.diff(indptr)
    quad_starts = indptr[:-1][sizes == 4]
    invalid = 0
    if len(quad_starts):
        quads = flat[quad_starts[:, None] + np.arange(4)]
        invalid = int(np.count_nonzero(~T._quads_valid(vertices, quads, epsilon)))
    detail = tuple(tuple(item) for item in counts["component_detail"])
    signature = tuple(sorted((item[0], item[1]) for item in detail))
    c = counts
    return T.TopologySummary(
        used_vertices=c["used_vertices"],
        edges=c["edges"],
        faces=c["faces"],
        quads=c["quads"],
        boundary_edges=c["boundary_edges"],
        boundary_components=c["boundary_components"],
        connected_components=len(detail),
        component_euler=tuple(item[0] for item in signature),
        component_signature=signature,
        component_detail=detail,
        non_manifold_edges=c["non_manifold_edges"],
        non_manifold_vertices=c["non_manifold_vertices"],
        branched_boundary_vertices=c["branched_boundary_vertices"],
        inconsistent_edges=c["inconsistent_edges"],
        duplicate_faces=c["duplicate_faces"],
        euler_characteristic=c["euler_characteristic"],
        extraordinary_vertices=c["extraordinary_vertices"],
        defect_charge=c["defect_charge"],
        degenerate_faces=degenerate,
        invalid_quads=invalid,
        all_quads=(
            c["quads"] == c["faces"]
            and degenerate == 0
            and c["non_manifold_edges"] == 0
            and c["non_manifold_vertices"] == 0
            and c["branched_boundary_vertices"] == 0
            and c["inconsistent_edges"] == 0
            and c["duplicate_faces"] == 0
            and invalid == 0
        ),
    )


def non_manifold_vertex_count(faces) -> int | None:
    """`topology._non_manifold_vertex_count` (the occurrence table is rebuilt natively)."""

    counts = topology_counts(faces)
    return None if counts is None else counts["non_manifold_vertices"]


def edge_occurrences(faces) -> dict | None:
    """`topology._edge_occurrences`: same keys, insertion order and (face, a, b) lists."""

    module = native()
    if module is None:
        return None
    indptr, flat = _csr(faces)
    edges, start, face, a, b = module.edge_occurrences(indptr, flat)
    # Millions of small tuples would trigger repeated cyclic-GC passes; none can form cycles.
    paused = gc.isenabled()
    gc.disable()
    try:
        uses = list(zip(face.tolist(), a.tolist(), b.tolist()))
        bounds = start.tolist()
        keys = map(tuple, edges.reshape(-1, 2).tolist())
        return {key: uses[bounds[k] : bounds[k + 1]] for k, key in enumerate(keys)}
    finally:
        if paused:
            gc.enable()


def quad_only_closure(mesh, *, name: str | None = None, frozen_midpoints: dict | None = None):
    """Native `topology.quad_only_closure` (same output, same ValueErrors)."""

    module = native()
    if module is None:
        return None
    T = _topology()
    indptr, flat = _csr(mesh.faces)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    frozen = np.asarray(
        [(int(a), int(b), int(m)) for (a, b), m in (frozen_midpoints or {}).items()], dtype=np.int64
    ).reshape(-1)
    uv_args = {}
    if mesh.uvs is not None:
        uv_args["uvs"] = np.ascontiguousarray(mesh.uvs, dtype=np.float64).reshape(-1)
        if mesh.face_uvs is not None:
            uv_args["uv_indptr"], uv_args["uv_flat"] = _csr(mesh.face_uvs)
    out_vertices, quads, uvs, face_uvs = module.quad_closure(indptr, flat, vertices.reshape(-1), frozen, **uv_args)
    sizes = np.diff(indptr).tolist()
    materials = mesh.face_materials
    # MeshData would re-tuple millions of faces in __post_init__; the lists below
    # already have its normalized form, so they are assigned after construction.
    result = T.MeshData(
        vertices=out_vertices.reshape(-1, 3),
        faces=[],
        uvs=None if uvs is None else uvs.reshape(-1, 2),
        name=name or f"{mesh.name}_quad_closure",
    )
    result.faces = list(map(tuple, quads.reshape(-1, 4).tolist()))
    result.face_materials = [
        materials[index] if index < len(materials) else "default"
        for index, size in enumerate(sizes)
        for _ in range(size)
    ]
    result.face_uvs = None if face_uvs is None else list(map(tuple, face_uvs.reshape(-1, 4).tolist()))
    summary = topology_summary(result) or T.topology_summary(result)
    if not summary.all_quads:
        raise ValueError(
            "Quad closure failed validation: "
            f"quads={summary.quads}/{summary.faces}, degenerate={summary.degenerate_faces}, "
            f"invalid_quads={summary.invalid_quads}, non_manifold_edges={summary.non_manifold_edges}, "
            f"non_manifold_vertices={summary.non_manifold_vertices}, "
            f"branched_boundary_vertices={summary.branched_boundary_vertices}, "
            f"inconsistent_edges={summary.inconsistent_edges}, duplicate_faces={summary.duplicate_faces}."
        )
    return result


def parity_path_edges(mesh, frozen=frozenset()) -> set | None:
    """Native `topology._parity_path_edges` (same set, same ValueErrors)."""

    module = native()
    if module is None or not mesh.faces:
        return None
    indptr, flat = _csr(mesh.faces)
    edges = module.parity_path_edges(indptr, flat, _edge_array(frozen))
    return set(map(tuple, edges.reshape(-1, 2).tolist()))


# --------------------------------------------------------------------------- matching


def augment_matching(mate: dict, allowed: dict, order: list) -> int | None:
    """Native `engine._augment_matching`: grows `mate` in place, returns the rescued count."""

    module = native()
    if module is None:
        return None
    ids = list(chain(allowed, order, mate))
    for linked in allowed.values():
        ids.extend(item[1] for item in linked)
    count = max(ids, default=-1) + 1
    if not count or min(ids) < 0:
        return 0 if not count else None
    sizes = np.zeros(count, dtype=np.int64)
    for face, linked in allowed.items():
        sizes[face] = len(linked)
    indptr = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    for face, linked in allowed.items():
        indices[indptr[face] : indptr[face + 1]] = [item[1] for item in linked]
    mate_array = np.full(count, -1, dtype=np.int64)
    if mate:
        mate_array[np.fromiter(mate.keys(), np.int64)] = np.fromiter(mate.values(), np.int64)
    before = mate_array.copy()
    rescued = module.augment_matching(indptr, indices, np.asarray(order, dtype=np.int64), mate_array)
    for face in np.flatnonzero(mate_array != before).tolist():
        mate[face] = int(mate_array[face])
    return int(rescued)


# --------------------------------------------------------------------------- geometry


def winding_number(points, triangles) -> np.ndarray | None:
    """Native `patch.winding_number` (parallel over points)."""

    module = native()
    if module is None:
        return None
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.ascontiguousarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if not len(points) or not len(triangles):
        return np.zeros(len(points))
    return module.winding_number(points.reshape(-1), triangles.reshape(-1))


def inside_volume(points, triangles) -> np.ndarray | None:
    """Native `patch.inside_volume` (same bounding-box cull, native winding number)."""

    module = native()
    if module is None:
        return None
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    inside = np.zeros(len(points), dtype=bool)
    if not len(triangles):
        return inside
    low, high = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
    pad = 1e-9 * float(np.linalg.norm(high - low))
    near = np.all((points >= low - pad) & (points <= high + pad), axis=1)
    inside[near] = np.abs(winding_number(points[near], triangles)) > 0.5
    return inside


def cross_field_jacobi(cross, face_a, face_b, transport, pinned, degree, steps: int) -> np.ndarray | None:
    """The smoothing loop of `engine._estimate_field`; returns the relaxed complex field."""

    module = native()
    if module is None:
        return None
    out = np.array(cross, dtype=np.complex128)
    module.cross_field_jacobi(
        out,
        np.ascontiguousarray(face_a, dtype=np.int64),
        np.ascontiguousarray(face_b, dtype=np.int64),
        np.ascontiguousarray(transport, dtype=np.complex128),
        np.ascontiguousarray(pinned, dtype=bool),
        np.ascontiguousarray(degree, dtype=np.float64),
        int(steps),
    )
    return out


def cross_field_gl(
    cross, face_a, face_b, transport, pinned, denom, pin_term, steps: int, mu: float = 1.0, dt: float = 0.4
) -> np.ndarray | None:
    """The relaxation loop of `field.estimate_field_gl`; returns the relaxed complex field."""

    module = native()
    if module is None:
        return None
    out = np.array(cross, dtype=np.complex128)
    module.cross_field_gl(
        out,
        np.ascontiguousarray(face_a, dtype=np.int64),
        np.ascontiguousarray(face_b, dtype=np.int64),
        np.ascontiguousarray(transport, dtype=np.complex128),
        np.ascontiguousarray(pinned, dtype=bool),
        np.ascontiguousarray(denom, dtype=np.float64),
        np.ascontiguousarray(pin_term, dtype=np.complex128),
        float(mu),
        float(dt),
        int(steps),
    )
    return out
