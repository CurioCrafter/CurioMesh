# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import astuple, dataclass, field, replace
import math
import random
import time
from typing import Any

import numpy as np

from .field import estimate_field_gl
from .topology import (
    _turbo,
    parity_path_quad_closure,
    quad_only_closure,
    quads_locally_valid,
    split_non_manifold_vertex_fans,
    topology_summary,
)
from .types import Face, MeshData


@dataclass(slots=True)
class RemeshOptions:
    target_faces: int = 4000
    mode: str = "AUTO"
    seed_count: int = 8
    feature_angle_deg: float = 35.0
    force_quads: bool = False
    smooth_field_iters: int = 4
    preserve_material_boundaries: bool = True
    preserve_uv_seams: bool = True
    # "SMOOTH": diffused 4-RoSy field (_estimate_field). "CURVATURE": Ginzburg-Landau
    # relaxation pulled towards principal curvature directions (field.estimate_field_gl).
    field_mode: str = "SMOOTH"
    # Accept every locally valid pair, however poorly it scores: a leftover triangle costs 3+ quads at closure.
    max_pairing: bool = False


@dataclass(slots=True)
class RemeshReport:
    success: bool
    mode: str
    selected_seed: int = 0
    input_faces: int = 0
    output_faces: int = 0
    quads: int = 0
    tris: int = 0
    quad_ratio: float = 0.0
    face_count_error: float = 0.0
    extraordinary_vertices: int = 0
    extraordinary_ratio: float = 0.0
    boundary_edges: int = 0
    non_manifold_edges: int = 0
    feature_edges: int = 0
    feature_breaks: int = 0
    aspect_penalty: float = 0.0
    closure_applied: bool = False
    closure_method: str = ""
    vertex_fan_splits: int = 0
    rescued_pairs: int = 0
    quad_only_verified: bool = False
    degenerate_faces: int = 0
    euler_characteristic: int = 0
    defect_charge: int = 0
    score: float = 0.0
    elapsed_ms: float = 0.0
    message: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "mode": self.mode,
            "selected_seed": self.selected_seed,
            "input_faces": self.input_faces,
            "output_faces": self.output_faces,
            "quads": self.quads,
            "tris": self.tris,
            "quad_ratio": self.quad_ratio,
            "face_count_error": self.face_count_error,
            "extraordinary_vertices": self.extraordinary_vertices,
            "extraordinary_ratio": self.extraordinary_ratio,
            "boundary_edges": self.boundary_edges,
            "non_manifold_edges": self.non_manifold_edges,
            "feature_edges": self.feature_edges,
            "feature_breaks": self.feature_breaks,
            "aspect_penalty": self.aspect_penalty,
            "closure_applied": self.closure_applied,
            "closure_method": self.closure_method,
            "vertex_fan_splits": self.vertex_fan_splits,
            "rescued_pairs": self.rescued_pairs,
            "quad_only_verified": self.quad_only_verified,
            "degenerate_faces": self.degenerate_faces,
            "euler_characteristic": self.euler_characteristic,
            "defect_charge": self.defect_charge,
            "score": self.score,
            "elapsed_ms": self.elapsed_ms,
            "message": self.message,
            "candidates": self.candidates,
        }


@dataclass(slots=True)
class _TriMesh:
    vertices: np.ndarray
    faces: list[tuple[int, int, int]]
    face_materials: list[str]
    face_uvs: list[tuple[int, int, int]] | None
    uvs: np.ndarray | None
    name: str
    # Per-mesh invariants (adjacency, features, field, pair scores) shared by
    # every seed of one remesh call.
    cache: dict[Any, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _Candidate:
    mesh: MeshData
    report: RemeshReport


@dataclass(frozen=True, slots=True)
class _PairWeights:
    normal: float
    field: float
    material: float
    aspect: float
    diagonal: float
    minimum_score: float


def remesh_mesh(mesh: MeshData, options: RemeshOptions | None = None) -> tuple[MeshData, RemeshReport]:
    start = time.perf_counter()
    options = options or RemeshOptions()
    normalized_mesh = mesh
    vertex_fan_splits = 0
    if options.force_quads:
        normalized_mesh, vertex_fan_splits = split_non_manifold_vertex_fans(mesh)
    tri_mesh = _triangulate(normalized_mesh)
    if not tri_mesh.faces:
        return mesh, RemeshReport(
            False,
            mode="EMPTY",
            vertex_fan_splits=vertex_fan_splits,
            message="Input mesh has no faces.",
        )

    mode = _classify_mesh(tri_mesh, options)
    best_mesh: MeshData | None = None
    best_report: RemeshReport | None = None
    candidate_reports: list[dict[str, Any]] = []
    seed_count = max(1, int(options.seed_count))

    for seed in range(seed_count):
        candidate = _run_candidate(tri_mesh, options, mode, seed)
        candidate.report.vertex_fan_splits = vertex_fan_splits
        candidate.report.candidates = []
        candidate_reports.append(candidate.report.to_dict())
        candidate_is_better = (
            best_report is None
            or (candidate.report.success and not best_report.success)
            or (
                candidate.report.success == best_report.success
                and candidate.report.score > best_report.score
            )
        )
        if candidate_is_better:
            best_mesh = candidate.mesh
            best_report = candidate.report

    assert best_mesh is not None and best_report is not None
    elapsed = (time.perf_counter() - start) * 1000.0
    report = best_report
    report.elapsed_ms = elapsed
    report.candidates = candidate_reports[:8]
    if report.success:
        closure_label = {
            "PARITY_PATH": "ParityPath",
            "GLOBAL_CORNER_TARGET": "global corner target closure",
        }.get(report.closure_method, "PairClosure")
        report.message = (
            f"TRIAD-Q {closure_label} completed with a locally validated all-quad manifold."
            if report.quad_only_verified
            else "TRIAD-Q Lite remesh completed."
        )
    return best_mesh, report


def _triangulate(mesh: MeshData) -> _TriMesh:
    faces: list[tuple[int, int, int]] = []
    materials: list[str] = []
    face_uvs: list[tuple[int, int, int]] = []
    has_uv_faces = mesh.face_uvs is not None

    for face_index, face in enumerate(mesh.faces):
        material = mesh.face_materials[face_index] if face_index < len(mesh.face_materials) else "default"
        uv_face = mesh.face_uvs[face_index] if has_uv_faces and mesh.face_uvs is not None else None
        if len(face) == 3:
            faces.append((face[0], face[1], face[2]))
            materials.append(material)
            if uv_face is not None:
                face_uvs.append((uv_face[0], uv_face[1], uv_face[2]))
        else:
            for i in range(1, len(face) - 1):
                faces.append((face[0], face[i], face[i + 1]))
                materials.append(material)
                if uv_face is not None:
                    face_uvs.append((uv_face[0], uv_face[i], uv_face[i + 1]))

    return _TriMesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=faces,
        face_materials=materials,
        face_uvs=face_uvs if has_uv_faces else None,
        uvs=mesh.uvs,
        name=mesh.name,
    )


def _classify_mesh(mesh: _TriMesh, options: RemeshOptions) -> str:
    requested = (options.mode or "AUTO").upper()
    if requested not in {"AUTO", "BALANCED"}:
        aliases = {
            "ORGANIC": "OrganicFlow",
            "PATCH": "PatchFlow",
            "DIRTY": "DirtyFlow",
            "TEXTURE": "TextureFlow",
        }
        return aliases.get(requested, requested)

    adj = _adjacency(mesh)
    edge_count = max(1, len(adj.edges))
    boundary = int(np.count_nonzero(adj.counts == 1))
    non_manifold = int(np.count_nonzero(adj.counts > 2))
    material_edges = _material_edge_mask(adj)
    sharp_edges = _sharp_edge_mask(adj, options.feature_angle_deg)
    face_count = max(1, len(mesh.faces))
    sharp_ratio = int(np.count_nonzero(sharp_edges | material_edges)) / edge_count
    boundary_ratio = boundary / edge_count

    if non_manifold > 0 and non_manifold / edge_count > 0.01:
        return "DirtyFlow"
    if mesh.face_uvs is not None and bool(material_edges.any()):
        return "TextureFlow"
    if sharp_ratio > 0.08 or boundary_ratio > 0.18:
        return "PatchFlow"
    if face_count > 100 and sharp_ratio < 0.03:
        return "OrganicFlow"
    return "BalancedFlow"


def _run_candidate(mesh: _TriMesh, options: RemeshOptions, mode: str, seed: int) -> _Candidate:
    prep = _pairing_prep(mesh, _options_for_mode(options, mode), mode)
    output_mesh, feature_breaks, rescued_pairs = _pair_triangles(mesh, prep, seed)
    closure_applied = bool(options.force_quads)
    closure_method = ""
    closure_error = ""
    if closure_applied:
        parity_mesh: MeshData | None = None
        global_mesh: MeshData | None = None
        parity_error = ""
        global_error = ""
        try:
            parity_mesh = parity_path_quad_closure(
                output_mesh,
                name=f"{mesh.name}_paritypath",
            )
        except ValueError as exc:
            parity_error = str(exc)
        try:
            global_mesh = quad_only_closure(
                output_mesh,
                name=f"{mesh.name}_pairclosure_global",
            )
        except ValueError as exc:
            global_error = str(exc)

        closure_candidates = [
            (candidate, method)
            for candidate, method in (
                (parity_mesh, "PARITY_PATH"),
                (global_mesh, "GLOBAL_CORNER_TARGET"),
            )
            if candidate is not None
        ]
        if closure_candidates:
            output_mesh, closure_method = min(
                closure_candidates,
                key=lambda item: (
                    abs(item[0].face_count - int(options.target_faces)),
                    item[0].face_count,
                    item[1],
                ),
            )
        else:
            closure_error = (
                f"ParityPath: {parity_error or 'no candidate'}; "
                f"global fallback: {global_error or 'no candidate'}"
            )
    report = _score_output(output_mesh, mesh, options, mode, seed, prep.feature_edge_count, feature_breaks)
    report.input_faces = len(mesh.faces)
    report.success = bool(output_mesh.faces)
    report.feature_edges = prep.feature_edge_count
    report.closure_applied = closure_applied
    report.closure_method = closure_method
    report.rescued_pairs = rescued_pairs
    if closure_error:
        report.success = False
        report.message = f"PairClosure rejected the base topology: {closure_error}"
    if _adjacency(mesh).areas.size == 0:
        report.success = False
        report.message = "No valid face areas."
    return _Candidate(output_mesh, report)


def _options_for_mode(options: RemeshOptions, mode: str) -> RemeshOptions:
    key = mode.upper()
    if key == "ORGANICFLOW":
        return replace(
            options,
            smooth_field_iters=max(6, int(options.smooth_field_iters) + 3),
            feature_angle_deg=min(175.0, float(options.feature_angle_deg) + 12.0),
        )
    if key == "PATCHFLOW":
        return replace(
            options,
            smooth_field_iters=min(2, max(0, int(options.smooth_field_iters))),
            feature_angle_deg=max(5.0, float(options.feature_angle_deg) - 10.0),
        )
    if key == "DIRTYFLOW":
        return replace(
            options,
            smooth_field_iters=1,
            feature_angle_deg=min(175.0, float(options.feature_angle_deg) + 15.0),
        )
    if key == "TEXTUREFLOW":
        return replace(
            options,
            smooth_field_iters=max(3, int(options.smooth_field_iters)),
            feature_angle_deg=max(5.0, float(options.feature_angle_deg) - 5.0),
            preserve_material_boundaries=True,
            preserve_uv_seams=True,
        )
    return options


def _pair_weights(mode: str) -> _PairWeights:
    return {
        "ORGANICFLOW": _PairWeights(2.6, 1.35, 0.15, 0.70, 0.25, -10.0),
        "PATCHFLOW": _PairWeights(2.1, 0.70, 0.35, 2.00, 0.05, -10.0),
        "DIRTYFLOW": _PairWeights(3.0, 0.35, 0.10, 1.35, 0.60, -3.0),
        "TEXTUREFLOW": _PairWeights(2.2, 0.80, 0.80, 0.95, 0.35, -10.0),
    }.get(mode.upper(), _PairWeights(2.2, 0.90, 0.15, 0.90, 0.35, -10.0))


@dataclass(slots=True)
class _Adjacency:
    """Seed-independent triangle adjacency, built once per mesh."""

    faces: np.ndarray  # (F, 3) vertex ids
    normals: np.ndarray  # (F, 3) unit normals, zero for degenerate faces
    areas: np.ndarray  # (F,)
    edges: np.ndarray  # (E, 2) sorted vertex pairs
    counts: np.ndarray  # (E,) face uses per edge
    edge_halves: np.ndarray  # (E, 2) first two half-edge ids (3 * face + corner), -1 if absent
    face_edges: np.ndarray  # (F, 3) edge id of corner i -> i + 1
    material_codes: np.ndarray  # (F,) integer material ids


@dataclass(slots=True)
class _PairingPrep:
    """Seed-independent pairing inputs for one (mode, options) combination."""

    feature_edge_count: int
    allowed: dict[int, list[tuple[float, int, tuple[int, ...]]]]
    quads_by_pair: dict[tuple[int, int], tuple[int, ...]]


def _adjacency(mesh: _TriMesh) -> _Adjacency:
    cached = mesh.cache.get("adjacency")
    if cached is None:
        cached = mesh.cache["adjacency"] = _build_adjacency(mesh)
    return cached


def _build_adjacency(mesh: _TriMesh) -> _Adjacency:
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    normals, areas = _face_normals(mesh.vertices, faces)
    halves = np.stack((faces, np.roll(faces, -1, axis=1)), axis=2).reshape(-1, 2)
    edges, inverse, counts = np.unique(
        np.sort(halves, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).reshape(-1)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    edge_halves = np.full((len(edges), 2), -1, dtype=np.int64)
    edge_halves[:, 0] = order[starts]
    multi = counts >= 2
    edge_halves[multi, 1] = order[starts[multi] + 1]
    codes: dict[str, int] = {}
    material_codes = np.asarray(
        [codes.setdefault(material, len(codes)) for material in mesh.face_materials], dtype=np.int64
    )
    return _Adjacency(
        faces=faces,
        normals=normals,
        areas=areas,
        edges=edges,
        counts=counts,
        edge_halves=edge_halves,
        face_edges=inverse.reshape(-1, 3),
        material_codes=material_codes,
    )


def _edge_face_pair(adj: _Adjacency) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (interior mask, first face, second face) per edge; second == first off interior."""

    interior = adj.counts == 2
    first = adj.edge_halves[:, 0] // 3
    second = np.where(interior, adj.edge_halves[:, 1] // 3, first)
    return interior, first, second


def _face_normals(vertices: np.ndarray, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    points = vertices[faces]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    length = np.linalg.norm(cross, axis=1)
    valid = length > 1e-12
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / length[valid, None]
    return normals, np.where(valid, length * 0.5, 0.0)


def _feature_mask(mesh: _TriMesh, adj: _Adjacency, options: RemeshOptions) -> np.ndarray:
    features = adj.counts != 2
    features |= _sharp_edge_mask(adj, options.feature_angle_deg)
    if options.preserve_material_boundaries:
        features |= _material_edge_mask(adj)
    if options.preserve_uv_seams and mesh.face_uvs is not None:
        features |= _uv_seam_mask(mesh, adj)
    return features


def _sharp_edge_mask(adj: _Adjacency, feature_angle_deg: float) -> np.ndarray:
    interior, first, second = _edge_face_pair(adj)
    threshold = math.cos(math.radians(max(0.0, min(180.0, feature_angle_deg))))
    dots = np.einsum("ij,ij->i", adj.normals[first], adj.normals[second])
    return interior & (dots < threshold)


def _material_edge_mask(adj: _Adjacency) -> np.ndarray:
    interior, first, second = _edge_face_pair(adj)
    return interior & (adj.material_codes[first] != adj.material_codes[second])


def _uv_seam_mask(mesh: _TriMesh, adj: _Adjacency) -> np.ndarray:
    if mesh.face_uvs is None:
        return np.zeros(len(adj.edges), dtype=bool)
    interior, _first, _second = _edge_face_pair(adj)
    face_uvs = np.asarray(mesh.face_uvs, dtype=np.int64).reshape(-1, 3)
    vertex_from = adj.faces.reshape(-1)
    vertex_to = np.roll(adj.faces, -1, axis=1).reshape(-1)
    uv_from = face_uvs.reshape(-1)
    uv_to = np.roll(face_uvs, -1, axis=1).reshape(-1)
    # UV stamp of each half-edge, listed in sorted-vertex order.
    ascending = vertex_from <= vertex_to
    stamps = np.stack(
        (np.where(ascending, uv_from, uv_to), np.where(ascending, uv_to, uv_from)), axis=1
    )
    halves = np.where(interior[:, None], adj.edge_halves, 0)
    return interior & np.any(stamps[halves[:, 0]] != stamps[halves[:, 1]], axis=1)


def _paired_face_uvs(
    mesh: _TriMesh,
    face_a: int,
    face_b: int,
    quad: tuple[int, ...],
) -> Face | None:
    if mesh.uvs is None or mesh.face_uvs is None:
        return None

    maps: list[dict[int, int]] = []
    for face_index in (face_a, face_b):
        face = mesh.faces[face_index]
        uv_face = mesh.face_uvs[face_index]
        if len(uv_face) != len(face):
            return None
        maps.append(dict(zip(face, uv_face, strict=True)))

    quad_uvs: list[int] = []
    for vertex in quad:
        values = [mapping[vertex] for mapping in maps if vertex in mapping]
        if not values or any(value < 0 or value >= len(mesh.uvs) for value in values):
            return None
        if any(value != values[0] for value in values[1:]):
            return None
        quad_uvs.append(values[0])
    return tuple(quad_uvs)


def _source_face_uvs(mesh: _TriMesh, face_index: int) -> Face | None:
    if mesh.uvs is None or mesh.face_uvs is None:
        return None
    face = mesh.faces[face_index]
    uv_face = mesh.face_uvs[face_index]
    if len(uv_face) != len(face):
        return None
    if any(value < 0 or value >= len(mesh.uvs) for value in uv_face):
        return None
    return tuple(uv_face)


def _unit_rows(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Normalize rows; rows shorter than 1e-12 take the (unit) fallback rows."""

    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = length > 1e-12
    return np.where(safe, vectors / np.where(safe, length, 1.0), fallback)


def _estimate_field(
    mesh: _TriMesh,
    adj: _Adjacency,
    feature_mask: np.ndarray,
    smooth_iters: int,
) -> np.ndarray:
    """Smooth a 4-RoSy cross field and return one representative direction per face.

    Each face gets a tangent frame (e1 along its first side, e2 = n x e1). A
    cross at angle theta is stored as the complex number exp(4i theta), so its
    four equivalent directions share one value. Neighbors are averaged across
    non-feature interior edges after parallel transport: a value in face j is
    rotated by exp(4i (alpha_i - alpha_j)), alpha_* being the shared edge's
    angle in each frame, which unfolds the two faces about that edge. Faces
    touching a feature edge are pinned to their longest feature edge; free
    faces start from a weak longest-edge guess that propagated values override.
    """

    faces = adj.faces
    normals = adj.normals
    face_count = len(faces)
    points = mesh.vertices[faces]
    sides = np.roll(points, -1, axis=1) - points  # side c runs corner c -> c + 1
    x_axis = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (face_count, 3))
    e1 = sides[:, 0] - normals * np.einsum("ij,ij->i", sides[:, 0], normals)[:, None]
    e1 = _unit_rows(e1, x_axis)
    e2 = np.cross(normals, e1)
    side_angle = np.arctan2(
        np.einsum("fcj,fj->fc", sides, e2), np.einsum("fcj,fj->fc", sides, e1)
    )
    lengths = np.linalg.norm(sides, axis=2)
    is_feature = feature_mask[adj.face_edges]
    priority = lengths + is_feature * (float(lengths.max(initial=0.0)) + 1.0)
    pick = np.argmax(priority, axis=1)
    initial = np.exp(4j * side_angle[np.arange(face_count), pick])
    pinned = is_feature.any(axis=1)
    cross = np.where(pinned, initial, 0.05 * initial)

    link = (adj.counts == 2) & ~feature_mask
    half_a, half_b = adj.edge_halves[link].T
    face_a, corner_a = np.divmod(half_a, 3)
    face_b, corner_b = np.divmod(half_b, 3)
    # Rotation carrying a cross value from face_b's frame into face_a's frame.
    transport = np.exp(4j * (side_angle[face_a, corner_a] - side_angle[face_b, corner_b]))
    degree = 1.0 + np.bincount(face_a, minlength=face_count) + np.bincount(face_b, minlength=face_count)

    # ponytail: fixed-count Jacobi diffusion; switch to a sparse linear solve if
    # very large scans need the field converged across hundreds of rings.
    steps = 24 * max(0, int(smooth_iters))
    fast = (
        _turbo.cross_field_jacobi(cross, face_a, face_b, transport, pinned, degree, steps)
        if _turbo is not None
        else None
    )
    for _ in range(0 if fast is not None else steps):
        into_a = cross[face_b] * transport
        into_b = cross[face_a] * np.conj(transport)
        real = np.bincount(face_a, weights=into_a.real, minlength=face_count) + np.bincount(
            face_b, weights=into_b.real, minlength=face_count
        )
        imag = np.bincount(face_a, weights=into_a.imag, minlength=face_count) + np.bincount(
            face_b, weights=into_b.imag, minlength=face_count
        )
        cross = np.where(pinned, cross, (cross + real + 1j * imag) / degree)
    if fast is not None:
        cross = fast

    cross = np.where(np.abs(cross) > 1e-12, cross, initial)
    theta = np.angle(cross) / 4.0
    return np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2


def _pairing_prep(mesh: _TriMesh, options: RemeshOptions, mode: str) -> _PairingPrep:
    key = ("pairing", mode, astuple(options))
    cached = mesh.cache.get(key)
    if cached is not None:
        return cached
    adj = _adjacency(mesh)
    weights = _pair_weights(mode)
    feature_mask = _feature_mask(mesh, adj, options)
    if str(options.field_mode).upper() == "CURVATURE":
        cross_field = estimate_field_gl(mesh, adj, feature_mask, options.smooth_field_iters)
    else:
        cross_field = _estimate_field(mesh, adj, feature_mask, options.smooth_field_iters)
    face_a, face_b, quads, scores = _pair_candidates(mesh, adj, feature_mask, cross_field, weights)

    allowed: dict[int, list[tuple[float, int, tuple[int, ...]]]] = defaultdict(list)
    quads_by_pair: dict[tuple[int, int], tuple[int, ...]] = {}
    for a, b, quad, score in zip(face_a.tolist(), face_b.tolist(), quads.tolist(), scores.tolist()):
        if not options.max_pairing and not score > weights.minimum_score:
            continue
        quad = tuple(quad)
        allowed[a].append((score, b, quad))
        allowed[b].append((score, a, quad))
        quads_by_pair[(a, b)] = quad
        quads_by_pair[(b, a)] = quad
    for linked in allowed.values():
        linked.sort(key=lambda item: item[0], reverse=True)
    prep = _PairingPrep(int(np.count_nonzero(feature_mask)), dict(allowed), quads_by_pair)
    mesh.cache[key] = prep
    return prep


def _pair_candidates(
    mesh: _TriMesh,
    adj: _Adjacency,
    feature_mask: np.ndarray,
    cross_field: np.ndarray,
    weights: _PairWeights,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (face_a, face_b, quad, score) for every locally valid pair."""

    link = (adj.counts == 2) & ~feature_mask
    half_a, half_b = adj.edge_halves[link].T
    face_a, corner_a = np.divmod(half_a, 3)
    face_b, corner_b = np.divmod(half_b, 3)
    faces = adj.faces
    x = faces[face_a, corner_a]
    y = faces[face_a, (corner_a + 1) % 3]
    # The shared edge must run opposite ways in the two faces (consistent winding).
    keep = (face_a != face_b) & (faces[face_b, corner_b] == y) & (faces[face_b, (corner_b + 1) % 3] == x)
    quads = np.stack((faces[face_a, (corner_a + 2) % 3], x, faces[face_b, (corner_b + 2) % 3], y), axis=1)
    ordered = np.sort(quads, axis=1)
    keep &= np.all(ordered[:, 1:] != ordered[:, :-1], axis=1)
    face_a, face_b, quads = face_a[keep], face_b[keep], quads[keep]
    valid = quads_locally_valid(mesh.vertices, quads) if len(quads) else np.zeros(0, dtype=bool)
    face_a, face_b, quads = face_a[valid], face_b[valid], quads[valid]
    scores = _pair_scores(mesh.vertices, adj, face_a, face_b, quads, cross_field, weights)
    return face_a, face_b, quads, scores


def _pair_scores(
    vertices: np.ndarray,
    adj: _Adjacency,
    face_a: np.ndarray,
    face_b: np.ndarray,
    quads: np.ndarray,
    cross_field: np.ndarray,
    weights: _PairWeights,
) -> np.ndarray:
    if not len(quads):
        return np.zeros(0, dtype=np.float64)
    points = vertices[quads]
    sides = np.roll(points, -1, axis=1) - points
    lengths = np.linalg.norm(sides, axis=2)
    with np.errstate(all="ignore"):
        aspect = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-12)
        normal_alignment = np.clip(
            np.einsum("ij,ij->i", adj.normals[face_a], adj.normals[face_b]), -1.0, 1.0
        )
        # quad = (opp_a, x, opp_b, y): the shared diagonal runs corner 1 -> 3.
        shared_len = np.linalg.norm(points[:, 3] - points[:, 1], axis=1)
        diagonal_penalty = shared_len / np.maximum(lengths.sum(axis=1) * 0.25, 1e-9)
        material_bonus = np.where(
            adj.material_codes[face_a] == adj.material_codes[face_b], weights.material, -1.5
        )
        field_score = _cross_alignment(sides, cross_field[face_a], adj.normals[face_a])
        jacobian = _min_scaled_jacobians(points)
        score = (
            normal_alignment * weights.normal
            + field_score * weights.field
            + jacobian * 3.0
            + material_bonus
            - np.log(np.maximum(aspect, 1.0)) * weights.aspect
            - diagonal_penalty * weights.diagonal
        )
    return np.where(np.isfinite(aspect) & (aspect > 0.0), score, -100.0)


def _cross_alignment(sides: np.ndarray, direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Mean alignment of each quad side with the nearest of the four cross directions."""

    count = len(sides)
    tangent = direction - normal * np.einsum("ij,ij->i", direction, normal)[:, None]
    tangent = _unit_rows(tangent, np.broadcast_to(np.array([1.0, 0.0, 0.0]), (count, 3)))
    bitangent = _unit_rows(
        np.cross(normal, tangent), np.broadcast_to(np.array([0.0, 1.0, 0.0]), (count, 3))
    )
    unit_sides = _unit_rows(sides, np.broadcast_to(tangent[:, None, :], sides.shape))
    along = np.abs(np.einsum("mkj,mj->mk", unit_sides, tangent))
    across = np.abs(np.einsum("mkj,mj->mk", unit_sides, bitangent))
    return np.maximum(along, across).mean(axis=1)


def _min_scaled_jacobians(points: np.ndarray) -> np.ndarray:
    following_points = np.roll(points, -1, axis=1)
    normal = np.cross(points, following_points).sum(axis=1)
    normal_length = np.linalg.norm(normal, axis=1)
    normal = normal / np.where(normal_length > 1e-20, normal_length, 1.0)[:, None]
    following = following_points - points
    previous = np.roll(points, 1, axis=1) - points
    denominator = np.linalg.norm(following, axis=2) * np.linalg.norm(previous, axis=2)
    corners = np.einsum("mkj,mj->mk", np.cross(following, previous), normal) / np.where(
        denominator > 1e-20, denominator, 1.0
    )
    minimum = np.minimum(corners.min(axis=1), 1.0)
    degenerate = (normal_length <= 1e-20) | np.any(denominator <= 1e-20, axis=1)
    return np.where(degenerate, -1.0, minimum)


def _pair_triangles(mesh: _TriMesh, prep: _PairingPrep, seed: int) -> tuple[MeshData, int, int]:
    allowed = prep.allowed
    feature_breaks = 0
    order = list(range(len(mesh.faces)))
    random.Random(seed).shuffle(order)
    mate: dict[int, int] = {}
    for face_index in order:
        if face_index in mate:
            continue
        for _score, other, _quad in allowed.get(face_index, ()):
            if other not in mate:
                mate[face_index] = other
                mate[other] = face_index
                break
    rescued_pairs = _augment_matching(mate, allowed, order)

    used: set[int] = set()
    output_faces: list[Face] = []
    output_materials: list[str] = []
    output_face_uvs: list[Face] | None = [] if mesh.uvs is not None and mesh.face_uvs is not None else None
    for face_index in order:
        if face_index in used or face_index not in mate:
            continue
        other = mate[face_index]
        quad = prep.quads_by_pair[(face_index, other)]
        used.add(face_index)
        used.add(other)
        output_faces.append(tuple(quad))
        output_materials.append(_majority_material(mesh.face_materials[face_index], mesh.face_materials[other]))
        if output_face_uvs is not None:
            paired_uvs = _paired_face_uvs(mesh, face_index, other, quad)
            if paired_uvs is None:
                output_face_uvs = None
            else:
                output_face_uvs.append(paired_uvs)

    for face_index, tri in enumerate(mesh.faces):
        if face_index in used:
            continue
        output_faces.append(tuple(tri))
        output_materials.append(mesh.face_materials[face_index])
        if output_face_uvs is not None:
            source_uvs = _source_face_uvs(mesh, face_index)
            if source_uvs is None:
                output_face_uvs = None
            else:
                output_face_uvs.append(source_uvs)

    out = MeshData(
        vertices=mesh.vertices.copy(),
        faces=output_faces,
        face_materials=output_materials,
        uvs=mesh.uvs,
        face_uvs=output_face_uvs,
        name=f"{mesh.name}_triadq",
    )
    return out, feature_breaks, rescued_pairs


def _augment_matching(
    mate: dict[int, int],
    allowed: dict[int, list[tuple[float, int, tuple[int, ...]]]],
    order: list[int],
) -> int:
    """Grow `mate` in place to a maximum-cardinality matching (Edmonds' blossoms).

    The greedy start keeps high-score pairs; each free root, in seed order,
    then runs one alternating-tree search that contracts odd cycles
    (blossoms) so paths through them are not missed. A root without an
    augmenting path stays unmatchable after later augmentations, and its whole
    search tree can be frozen (Edmonds), so every root is searched once.
    Returns the number of augmentations, i.e. extra pairs beyond the start.
    """

    adjacency = {face: [item[1] for item in linked] for face, linked in allowed.items()}
    frozen: set[int] = set()
    rescued = 0
    for root in order:
        if root in mate or root in frozen or not adjacency.get(root):
            continue
        end, parent, tree = _blossom_search(root, mate, adjacency, frozen)
        if end < 0:
            frozen.update(tree)
            continue
        vertex = end
        while vertex != -1:
            previous = parent[vertex]
            following = mate.get(previous, -1)
            mate[vertex] = previous
            mate[previous] = vertex
            vertex = following
        rescued += 1
    return rescued


def _blossom_search(
    root: int,
    mate: dict[int, int],
    adjacency: dict[int, list[int]],
    frozen: set[int],
) -> tuple[int, dict[int, int], set[int]]:
    """One BFS of Edmonds' algorithm from `root`.

    Returns (free end or -1, parent links for the augmenting walk, tree
    vertices). Contracted blossoms live in a union-find whose root is the
    blossom base, so a contraction costs only the cycle length.
    """

    base: dict[int, int] = {}
    parent: dict[int, int] = {}
    even = {root}

    def find(vertex: int) -> int:
        top = vertex
        while top in base:
            top = base[top]
        while vertex != top:
            following = base[vertex]
            base[vertex] = top
            vertex = following
        return top

    def common_base(first: int, second: int) -> int:
        seen: set[int] = set()
        while True:
            first = find(first)
            seen.add(first)
            partner = mate.get(first)
            if partner is None:
                break
            first = parent[partner]
        while True:
            second = find(second)
            if second in seen:
                return second
            second = parent[mate[second]]

    def mark_path(vertex: int, stop: int, child: int, marked: set[int], odd: list[int]) -> None:
        while find(vertex) != stop:
            partner = mate[vertex]
            marked.add(find(vertex))
            marked.add(find(partner))
            parent[vertex] = child
            child = partner
            odd.append(partner)
            vertex = parent[partner]

    queue = deque([root])
    while queue:
        vertex = queue.popleft()
        for other in adjacency.get(vertex, ()):
            if other in frozen or mate.get(vertex) == other or find(vertex) == find(other):
                continue
            partner = mate.get(other)
            if other == root or (partner is not None and partner in parent):
                # Both ends are even: an odd cycle. Contract it onto its base.
                blossom_base = common_base(vertex, other)
                marked: set[int] = set()
                odd: list[int] = []
                mark_path(vertex, blossom_base, other, marked, odd)
                mark_path(other, blossom_base, vertex, marked, odd)
                for member in marked:
                    if member != blossom_base:
                        base[member] = blossom_base
                for member in odd:
                    if member not in even:
                        even.add(member)
                        queue.append(member)
            elif other not in parent:
                parent[other] = vertex
                if partner is None:
                    return other, parent, even | set(parent)
                even.add(partner)
                queue.append(partner)
    return -1, parent, even | set(parent)


def _majority_material(a: str, b: str) -> str:
    return a if a == b else a


def _score_output(
    output: MeshData,
    source: _TriMesh,
    options: RemeshOptions,
    mode: str,
    seed: int,
    feature_edge_count: int,
    feature_breaks: int,
) -> RemeshReport:
    face_count = len(output.faces)
    quads = sum(1 for face in output.faces if len(face) == 4)
    tris = sum(1 for face in output.faces if len(face) == 3)
    quad_ratio = quads / max(1, face_count)
    summary = topology_summary(output)
    boundary = summary.boundary_edges
    non_manifold = summary.non_manifold_edges
    extraordinary = summary.extraordinary_vertices
    extraordinary_ratio = extraordinary / max(1, summary.used_vertices)
    aspect_penalty = _aspect_penalty(output)
    target_error = abs(face_count - int(options.target_faces)) / max(1.0, float(options.target_faces))
    score = (
        quad_ratio * 8.0
        - target_error * 1.2
        - extraordinary_ratio * 0.65
        - aspect_penalty * 0.45
        - feature_breaks * 0.2
        - non_manifold * 0.02
    )
    return RemeshReport(
        success=face_count > 0,
        mode=mode,
        selected_seed=seed,
        output_faces=face_count,
        quads=quads,
        tris=tris,
        quad_ratio=quad_ratio,
        face_count_error=target_error,
        extraordinary_vertices=extraordinary,
        extraordinary_ratio=extraordinary_ratio,
        boundary_edges=boundary,
        non_manifold_edges=non_manifold,
        feature_edges=feature_edge_count,
        feature_breaks=feature_breaks,
        aspect_penalty=aspect_penalty,
        quad_only_verified=summary.all_quads,
        degenerate_faces=summary.degenerate_faces,
        euler_characteristic=summary.euler_characteristic,
        defect_charge=summary.defect_charge,
        score=score,
        input_faces=len(source.faces),
    )


def _aspect_penalty(mesh: MeshData) -> float:
    quads = [face for face in mesh.faces if len(face) == 4]
    if not quads:
        return 0.0
    points = mesh.vertices[np.asarray(quads, dtype=np.int64)]
    lengths = np.linalg.norm(np.roll(points, -1, axis=1) - points, axis=2)
    aspect = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-12)
    return float(np.mean(np.log(np.maximum(aspect, 1.0))))
