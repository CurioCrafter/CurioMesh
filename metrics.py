from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import bpy


@dataclass(slots=True)
class MeshDiagnostics:
    faces: int = 0
    quads: int = 0
    tris: int = 0
    ngons: int = 0
    quad_ratio: float = 0.0
    face_count_error: float = 0.0
    extraordinary: int = 0
    extraordinary_ratio: float = 0.0
    non_manifold_edges: int = 0
    boundary_edges: int = 0
    uv_valid: bool = False


def mesh_diagnostics(obj: bpy.types.Object, target_faces: int = 0) -> MeshDiagnostics:
    mesh = obj.data
    face_count = len(mesh.polygons)
    quads = sum(1 for poly in mesh.polygons if len(poly.vertices) == 4)
    tris = sum(1 for poly in mesh.polygons if len(poly.vertices) == 3)
    ngons = max(0, face_count - quads - tris)

    valence = [0] * len(mesh.vertices)
    for edge in mesh.edges:
        a, b = edge.vertices
        valence[a] += 1
        valence[b] += 1

    extraordinary = sum(1 for count in valence if count != 4)
    edge_use = Counter()
    for poly in mesh.polygons:
        for key in poly.edge_keys:
            edge_use[tuple(sorted(key))] += 1

    boundary_edges = sum(1 for count in edge_use.values() if count == 1)
    non_manifold_edges = sum(1 for count in edge_use.values() if count != 2)
    face_error = 0.0
    if target_faces > 0:
        face_error = abs(face_count - int(target_faces)) / float(max(1, int(target_faces)))

    return MeshDiagnostics(
        faces=face_count,
        quads=quads,
        tris=tris,
        ngons=ngons,
        quad_ratio=(quads / float(face_count)) if face_count else 0.0,
        face_count_error=face_error,
        extraordinary=extraordinary,
        extraordinary_ratio=(extraordinary / float(len(mesh.vertices))) if mesh.vertices else 0.0,
        non_manifold_edges=non_manifold_edges,
        boundary_edges=boundary_edges,
        uv_valid=uv_mapping_is_valid(obj),
    )


def uv_mapping_is_valid(obj: bpy.types.Object, min_area: float = 1e-10) -> bool:
    mesh = obj.data
    if not getattr(mesh, "uv_layers", None) or not mesh.uv_layers:
        return False
    uv_layer = mesh.uv_layers.active
    if uv_layer is None or len(uv_layer.data) == 0:
        return False

    coords = [slot.uv for slot in uv_layer.data]
    if not coords:
        return False
    if any(not (math.isfinite(uv.x) and math.isfinite(uv.y)) for uv in coords):
        return False

    span_x = max(uv.x for uv in coords) - min(uv.x for uv in coords)
    span_y = max(uv.y for uv in coords) - min(uv.y for uv in coords)
    if span_x <= 1e-8 or span_y <= 1e-8:
        return False

    try:
        mesh.calc_loop_triangles()
    except Exception:
        return True

    area = 0.0
    for tri in mesh.loop_triangles:
        u0 = uv_layer.data[tri.loops[0]].uv
        u1 = uv_layer.data[tri.loops[1]].uv
        u2 = uv_layer.data[tri.loops[2]].uv
        area += abs((u1.x - u0.x) * (u2.y - u0.y) - (u1.y - u0.y) * (u2.x - u0.x)) * 0.5
    return area > min_area
