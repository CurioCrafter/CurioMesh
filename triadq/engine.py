from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import random
import time
from typing import Any

import numpy as np

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


@dataclass(slots=True)
class _Candidate:
    mesh: MeshData
    report: RemeshReport


def remesh_mesh(mesh: MeshData, options: RemeshOptions | None = None) -> tuple[MeshData, RemeshReport]:
    start = time.perf_counter()
    options = options or RemeshOptions()
    tri_mesh = _triangulate(mesh)
    if not tri_mesh.faces:
        return mesh, RemeshReport(False, mode="EMPTY", message="Input mesh has no faces.")

    mode = _classify_mesh(tri_mesh, options)
    best: _Candidate | None = None
    seed_count = max(1, int(options.seed_count))

    for seed in range(seed_count):
        candidate = _run_candidate(tri_mesh, options, mode, seed)
        candidate.report.candidates = []
        if best is None or candidate.report.score > best.report.score:
            best = candidate

    assert best is not None
    elapsed = (time.perf_counter() - start) * 1000.0
    report = best.report
    report.elapsed_ms = elapsed
    report.candidates = [
        _run_candidate(tri_mesh, options, mode, seed).report.to_dict()
        for seed in range(min(seed_count, 8))
    ]
    report.message = "TRIAD-Q Lite remesh completed."
    return best.mesh, report


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

    edge_faces = _edge_faces(mesh.faces)
    boundary = sum(1 for faces in edge_faces.values() if len(faces) == 1)
    non_manifold = sum(1 for faces in edge_faces.values() if len(faces) > 2)
    material_edges = _material_boundary_edges(mesh, edge_faces)
    sharp_edges = _sharp_edges(mesh, edge_faces, options.feature_angle_deg)
    face_count = max(1, len(mesh.faces))
    sharp_ratio = len(sharp_edges | material_edges) / max(1, len(edge_faces))
    boundary_ratio = boundary / max(1, len(edge_faces))

    if non_manifold > 0 and non_manifold / max(1, len(edge_faces)) > 0.01:
        return "DirtyFlow"
    if mesh.face_uvs is not None and len(material_edges) > 0:
        return "TextureFlow"
    if sharp_ratio > 0.08 or boundary_ratio > 0.18:
        return "PatchFlow"
    if face_count > 100 and sharp_ratio < 0.03:
        return "OrganicFlow"
    return "BalancedFlow"


def _run_candidate(mesh: _TriMesh, options: RemeshOptions, mode: str, seed: int) -> _Candidate:
    edge_faces = _edge_faces(mesh.faces)
    normals, areas = _face_normals(mesh.vertices, mesh.faces)
    feature_edges = _feature_edges(mesh, options, edge_faces, normals)
    field = _estimate_field(mesh, edge_faces, feature_edges, normals, options)
    output_mesh, feature_breaks = _pair_triangles(mesh, options, edge_faces, feature_edges, normals, field, seed)
    report = _score_output(output_mesh, mesh, options, mode, seed, feature_edges, feature_breaks)
    report.input_faces = len(mesh.faces)
    report.success = bool(output_mesh.faces)
    report.feature_edges = len(feature_edges)
    if areas.size == 0:
        report.success = False
        report.message = "No valid face areas."
    return _Candidate(output_mesh, report)


def _edge_faces(faces: list[tuple[int, int, int]]) -> dict[tuple[int, int], list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for edge in _tri_edges(face):
            edge_faces[edge].append(face_index)
    return edge_faces


def _tri_edges(face: tuple[int, int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    a, b, c = face
    return tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))


def _face_normals(vertices: np.ndarray, faces: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros((len(faces), 3), dtype=np.float64)
    areas = np.zeros(len(faces), dtype=np.float64)
    for index, face in enumerate(faces):
        a, b, c = vertices[list(face)]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length > 1e-12:
            normals[index] = normal / length
            areas[index] = length * 0.5
    return normals, areas


def _feature_edges(
    mesh: _TriMesh,
    options: RemeshOptions,
    edge_faces: dict[tuple[int, int], list[int]],
    normals: np.ndarray,
) -> set[tuple[int, int]]:
    features: set[tuple[int, int]] = set()
    features.update(edge for edge, faces in edge_faces.items() if len(faces) != 2)
    features.update(_sharp_edges(mesh, edge_faces, options.feature_angle_deg, normals=normals))
    if options.preserve_material_boundaries:
        features.update(_material_boundary_edges(mesh, edge_faces))
    if options.preserve_uv_seams and mesh.face_uvs is not None:
        features.update(_uv_seam_edges(mesh, edge_faces))
    return features


def _sharp_edges(
    mesh: _TriMesh,
    edge_faces: dict[tuple[int, int], list[int]],
    feature_angle_deg: float,
    *,
    normals: np.ndarray | None = None,
) -> set[tuple[int, int]]:
    normals = normals if normals is not None else _face_normals(mesh.vertices, mesh.faces)[0]
    threshold = math.cos(math.radians(max(0.0, min(180.0, feature_angle_deg))))
    sharp = set()
    for edge, faces in edge_faces.items():
        if len(faces) != 2:
            continue
        if float(np.dot(normals[faces[0]], normals[faces[1]])) < threshold:
            sharp.add(edge)
    return sharp


def _material_boundary_edges(mesh: _TriMesh, edge_faces: dict[tuple[int, int], list[int]]) -> set[tuple[int, int]]:
    out = set()
    for edge, faces in edge_faces.items():
        if len(faces) == 2 and mesh.face_materials[faces[0]] != mesh.face_materials[faces[1]]:
            out.add(edge)
    return out


def _uv_seam_edges(mesh: _TriMesh, edge_faces: dict[tuple[int, int], list[int]]) -> set[tuple[int, int]]:
    if mesh.face_uvs is None:
        return set()
    seams = set()
    for edge, faces in edge_faces.items():
        if len(faces) != 2:
            continue
        stamps = []
        for face_index in faces:
            face = mesh.faces[face_index]
            uv_face = mesh.face_uvs[face_index]
            uv_by_vertex = {face[i]: uv_face[i] for i in range(3)}
            stamps.append(tuple(uv_by_vertex.get(v, -1) for v in edge))
        if stamps[0] != stamps[1]:
            seams.add(edge)
    return seams


def _estimate_field(
    mesh: _TriMesh,
    edge_faces: dict[tuple[int, int], list[int]],
    feature_edges: set[tuple[int, int]],
    normals: np.ndarray,
    options: RemeshOptions,
) -> np.ndarray:
    field = np.zeros((len(mesh.faces), 3), dtype=np.float64)
    for face_index, face in enumerate(mesh.faces):
        field[face_index] = _initial_face_direction(mesh.vertices, face, feature_edges, normals[face_index])

    neighbors = [[] for _ in mesh.faces]
    for edge, faces in edge_faces.items():
        if len(faces) == 2 and edge not in feature_edges:
            neighbors[faces[0]].append(faces[1])
            neighbors[faces[1]].append(faces[0])

    for _ in range(max(0, int(options.smooth_field_iters))):
        next_field = field.copy()
        for face_index, linked in enumerate(neighbors):
            if not linked:
                continue
            acc = field[face_index].copy() * 2.0
            normal = normals[face_index]
            for other in linked:
                projected = _project_to_plane(field[other], normal)
                if np.linalg.norm(projected) < 1e-12:
                    continue
                if np.dot(projected, field[face_index]) < 0:
                    projected = -projected
                acc += projected
            next_field[face_index] = _safe_normalize(_project_to_plane(acc, normal), fallback=field[face_index])
        field = next_field
    return field


def _initial_face_direction(
    vertices: np.ndarray,
    face: tuple[int, int, int],
    feature_edges: set[tuple[int, int]],
    normal: np.ndarray,
) -> np.ndarray:
    edges = [(face[i], face[(i + 1) % 3]) for i in range(3)]
    scored = []
    for edge in edges:
        vec = vertices[edge[1]] - vertices[edge[0]]
        length = float(np.linalg.norm(vec))
        is_feature = tuple(sorted(edge)) in feature_edges
        scored.append((1 if is_feature else 0, length, vec))
    _feature, _length, vec = max(scored, key=lambda item: (item[0], item[1]))
    return _safe_normalize(_project_to_plane(vec, normal), fallback=np.array([1.0, 0.0, 0.0]))


def _project_to_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vec - normal * float(np.dot(vec, normal))


def _safe_normalize(vec: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length <= 1e-12:
        fallback_length = float(np.linalg.norm(fallback))
        return fallback / fallback_length if fallback_length > 1e-12 else np.array([1.0, 0.0, 0.0])
    return vec / length


def _pair_triangles(
    mesh: _TriMesh,
    options: RemeshOptions,
    edge_faces: dict[tuple[int, int], list[int]],
    feature_edges: set[tuple[int, int]],
    normals: np.ndarray,
    field: np.ndarray,
    seed: int,
) -> tuple[MeshData, int]:
    candidates: dict[int, list[tuple[float, int, tuple[int, ...]]]] = defaultdict(list)
    feature_breaks = 0
    for edge, linked_faces in edge_faces.items():
        if len(linked_faces) != 2:
            continue
        if edge in feature_edges:
            continue
        a, b = linked_faces
        quad = _quad_from_pair(mesh.faces[a], mesh.faces[b], edge)
        if quad is None:
            continue
        score = _pair_score(mesh, quad, edge, a, b, normals, field)
        candidates[a].append((score, b, quad))
        candidates[b].append((score, a, quad))

    order = list(range(len(mesh.faces)))
    random.Random(seed).shuffle(order)
    used: set[int] = set()
    output_faces: list[Face] = []
    output_materials: list[str] = []
    for face_index in order:
        if face_index in used:
            continue
        options_for_face = sorted(candidates.get(face_index, []), key=lambda item: item[0], reverse=True)
        selected = None
        for score, other, quad in options_for_face:
            if other not in used and score > -10.0:
                selected = (other, quad)
                break
        if selected is None:
            continue
        other, quad = selected
        used.add(face_index)
        used.add(other)
        output_faces.append(tuple(quad))
        output_materials.append(_majority_material(mesh.face_materials[face_index], mesh.face_materials[other]))

    vertices = mesh.vertices.tolist()
    for face_index, tri in enumerate(mesh.faces):
        if face_index in used:
            continue
        if options.force_quads:
            quad, vertices = _split_triangle_to_quad(tri, np.asarray(vertices, dtype=np.float64))
            output_faces.append(tuple(quad))
        else:
            output_faces.append(tuple(tri))
        output_materials.append(mesh.face_materials[face_index])

    out = MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=output_faces,
        face_materials=output_materials,
        name=f"{mesh.name}_triadq",
    )
    return out, feature_breaks


def _quad_from_pair(
    face_a: tuple[int, int, int],
    face_b: tuple[int, int, int],
    shared_edge: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    shared = set(shared_edge)
    opp_a = [v for v in face_a if v not in shared]
    opp_b = [v for v in face_b if v not in shared]
    if len(opp_a) != 1 or len(opp_b) != 1:
        return None
    a, b = shared_edge
    quad = (opp_a[0], a, opp_b[0], b)
    if len(set(quad)) != 4:
        return None
    return quad


def _pair_score(
    mesh: _TriMesh,
    quad: tuple[int, int, int, int],
    shared_edge: tuple[int, int],
    face_a: int,
    face_b: int,
    normals: np.ndarray,
    field: np.ndarray,
) -> float:
    normal_alignment = max(-1.0, min(1.0, float(np.dot(normals[face_a], normals[face_b]))))
    aspect = _quad_aspect(mesh.vertices, quad)
    if not math.isfinite(aspect) or aspect <= 0.0:
        return -100.0
    shared_len = float(np.linalg.norm(mesh.vertices[shared_edge[1]] - mesh.vertices[shared_edge[0]]))
    perimeter = _quad_perimeter(mesh.vertices, quad)
    diagonal_penalty = shared_len / max(perimeter * 0.25, 1e-9)
    material_bonus = 0.15 if mesh.face_materials[face_a] == mesh.face_materials[face_b] else -0.75
    field_score = _field_alignment(mesh.vertices, quad, field[face_a], normals[face_a])
    return (
        normal_alignment * 2.2
        + field_score * 0.9
        + material_bonus
        - math.log(max(aspect, 1.0)) * 0.9
        - diagonal_penalty * 0.35
    )


def _quad_aspect(vertices: np.ndarray, quad: tuple[int, int, int, int]) -> float:
    lengths = []
    for i in range(4):
        a = vertices[quad[i]]
        b = vertices[quad[(i + 1) % 4]]
        lengths.append(float(np.linalg.norm(b - a)))
    min_len = max(min(lengths), 1e-12)
    return max(lengths) / min_len


def _quad_perimeter(vertices: np.ndarray, quad: tuple[int, int, int, int]) -> float:
    return sum(
        float(np.linalg.norm(vertices[quad[(i + 1) % 4]] - vertices[quad[i]]))
        for i in range(4)
    )


def _field_alignment(vertices: np.ndarray, quad: tuple[int, int, int, int], direction: np.ndarray, normal: np.ndarray) -> float:
    tangent = _safe_normalize(_project_to_plane(direction, normal), fallback=np.array([1.0, 0.0, 0.0]))
    bitangent = _safe_normalize(np.cross(normal, tangent), fallback=np.array([0.0, 1.0, 0.0]))
    score = 0.0
    for i in range(4):
        edge_vec = vertices[quad[(i + 1) % 4]] - vertices[quad[i]]
        edge_vec = _safe_normalize(edge_vec, fallback=tangent)
        score += max(abs(float(np.dot(edge_vec, tangent))), abs(float(np.dot(edge_vec, bitangent))))
    return score / 4.0


def _majority_material(a: str, b: str) -> str:
    return a if a == b else a


def _split_triangle_to_quad(tri: tuple[int, int, int], vertices: np.ndarray) -> tuple[tuple[int, int, int, int], list[list[float]]]:
    a, b, c = tri
    pairs = [((a, b), c), ((b, c), a), ((c, a), b)]
    edge, opposite = max(
        pairs,
        key=lambda item: float(np.linalg.norm(vertices[item[0][1]] - vertices[item[0][0]])),
    )
    midpoint = (vertices[edge[0]] + vertices[edge[1]]) * 0.5
    new_vertices = vertices.tolist()
    new_index = len(new_vertices)
    new_vertices.append(midpoint.tolist())
    return (opposite, edge[0], new_index, edge[1]), new_vertices


def _score_output(
    output: MeshData,
    source: _TriMesh,
    options: RemeshOptions,
    mode: str,
    seed: int,
    feature_edges: set[tuple[int, int]],
    feature_breaks: int,
) -> RemeshReport:
    face_count = len(output.faces)
    quads = sum(1 for face in output.faces if len(face) == 4)
    tris = sum(1 for face in output.faces if len(face) == 3)
    quad_ratio = quads / max(1, face_count)
    edge_counts = _poly_edge_counts(output.faces)
    boundary = sum(1 for count in edge_counts.values() if count == 1)
    non_manifold = sum(1 for count in edge_counts.values() if count > 2)
    valence = Counter(v for face in output.faces for v in face)
    extraordinary = sum(1 for count in valence.values() if count != 4)
    extraordinary_ratio = extraordinary / max(1, output.vertex_count)
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
        feature_edges=len(feature_edges),
        feature_breaks=feature_breaks,
        aspect_penalty=aspect_penalty,
        score=score,
        input_faces=len(source.faces),
    )


def _poly_edge_counts(faces: list[Face]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for face in faces:
        for i in range(len(face)):
            counts[tuple(sorted((face[i], face[(i + 1) % len(face)])))] += 1
    return counts


def _aspect_penalty(mesh: MeshData) -> float:
    penalties = []
    for face in mesh.faces:
        if len(face) == 4:
            penalties.append(max(0.0, math.log(max(_quad_aspect(mesh.vertices, face), 1.0))))
    return float(np.mean(penalties)) if penalties else 0.0
