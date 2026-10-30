# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

from dataclasses import dataclass
import math

import bpy
from mathutils import Vector
import numpy as np

from .triadq import MeshData, topology_summary


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
    non_manifold_vertices: int = 0
    branched_boundary_vertices: int = 0
    inconsistent_edges: int = 0
    duplicate_faces: int = 0
    boundary_edges: int = 0
    boundary_loops: int = 0
    connected_components: int = 0
    component_euler: tuple[int, ...] = ()
    component_signature: tuple[tuple[int, int], ...] = ()
    component_detail: tuple[tuple[int, int, int], ...] = ()
    euler_characteristic: int = 0
    degenerate_faces: int = 0
    invalid_quads: int = 0
    min_scaled_jacobian: float = 0.0
    p05_scaled_jacobian: float = 0.0
    p95_aspect_ratio: float = 0.0
    quad_only: bool = False
    uv_valid: bool = False


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _quad_quality(obj: bpy.types.Object, max_percentile_samples: int = 200_000) -> tuple[int, float, float, float]:
    mesh = obj.data
    if not mesh.vertices:
        return 0, 0.0, 0.0, 0.0
    bounds = [obj.matrix_world @ Vector(corner) for corner in (getattr(obj, "bound_box", ()) or ())]
    if bounds:
        xs = [corner[0] for corner in bounds]
        ys = [corner[1] for corner in bounds]
        zs = [corner[2] for corner in bounds]
        diagonal_squared = (
            (max(xs) - min(xs)) ** 2
            + (max(ys) - min(ys)) ** 2
            + (max(zs) - min(zs)) ** 2
        )
    else:
        diagonal_squared = 0.0
    area_epsilon = max(diagonal_squared * 1e-14, 1e-20)
    jacobians: list[float] = []
    aspects: list[float] = []
    degenerate = 0
    quad_count = sum(1 for poly in mesh.polygons if len(poly.vertices) == 4)
    corner_stride = max(1, math.ceil(max(1, quad_count * 4) / max(1, max_percentile_samples)))
    aspect_stride = max(1, math.ceil(max(1, quad_count) / max(1, max_percentile_samples)))
    corner_number = 0
    quad_number = 0
    minimum_jacobian = float("inf")
    matrix = obj.matrix_world

    for poly in mesh.polygons:
        coords = [matrix @ mesh.vertices[index].co for index in poly.vertices]
        origin = coords[0] if coords else None
        world_area = (
            0.5
            * sum(
                (coords[corner] - origin).cross(coords[corner + 1] - origin).length
                for corner in range(1, len(coords) - 1)
            )
            if origin is not None
            else 0.0
        )
        poly_degenerate = (
            not math.isfinite(world_area)
            or world_area <= area_epsilon
            or len(set(poly.vertices)) != len(poly.vertices)
        )
        if poly_degenerate:
            degenerate += 1
        if len(poly.vertices) != 4:
            continue
        lengths = [(coords[(index + 1) % 4] - coords[index]).length for index in range(4)]
        shortest = min(lengths)
        if shortest <= 1e-12:
            if not poly_degenerate:
                degenerate += 1
            minimum_jacobian = -1.0
            if corner_number % corner_stride == 0:
                jacobians.append(-1.0)
            corner_number += 4
            quad_number += 1
            continue
        aspect = max(lengths) / shortest
        if quad_number % aspect_stride == 0:
            aspects.append(aspect)
        reference_normal = Vector((0.0, 0.0, 0.0))
        center = sum(coords, Vector((0.0, 0.0, 0.0))) / 4.0
        for index in range(4):
            reference_normal += (coords[index] - center).cross(coords[(index + 1) % 4] - center)
        if reference_normal.length <= 1e-20:
            reference_normal = Vector((0.0, 0.0, 0.0))
        else:
            reference_normal.normalize()
        for index in range(4):
            current = coords[index]
            following = coords[(index + 1) % 4] - current
            previous = coords[(index - 1) % 4] - current
            denominator = following.length * previous.length
            if denominator <= 1e-20:
                jacobian = -1.0
            else:
                jacobian = following.cross(previous).dot(reference_normal) / denominator
            minimum_jacobian = min(minimum_jacobian, jacobian)
            if corner_number % corner_stride == 0:
                jacobians.append(jacobian)
            corner_number += 1
        quad_number += 1

    finite_aspects = [value for value in aspects if math.isfinite(value)]
    return (
        degenerate,
        minimum_jacobian if math.isfinite(minimum_jacobian) else 0.0,
        _percentile(jacobians, 0.05),
        _percentile(finite_aspects, 0.95) if finite_aspects else 0.0,
    )


def mesh_diagnostics(obj: bpy.types.Object, target_faces: int = 0) -> MeshDiagnostics:
    mesh = obj.data
    face_count = len(mesh.polygons)
    quads = sum(1 for poly in mesh.polygons if len(poly.vertices) == 4)
    tris = sum(1 for poly in mesh.polygons if len(poly.vertices) == 3)
    ngons = max(0, face_count - quads - tris)

    coordinates = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    if len(mesh.vertices):
        mesh.vertices.foreach_get("co", coordinates)
    coordinates = coordinates.reshape((-1, 3))
    world_matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    coordinates = coordinates @ world_matrix[:3, :3].T + world_matrix[:3, 3]
    faces = [tuple(poly.vertices) for poly in mesh.polygons]
    topology = topology_summary(MeshData(vertices=coordinates, faces=faces))
    degenerate, min_jacobian, p05_jacobian, p95_aspect = _quad_quality(obj)
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
        extraordinary=topology.extraordinary_vertices,
        extraordinary_ratio=(
            topology.extraordinary_vertices / float(topology.used_vertices)
            if topology.used_vertices
            else 0.0
        ),
        non_manifold_edges=topology.non_manifold_edges,
        non_manifold_vertices=topology.non_manifold_vertices,
        branched_boundary_vertices=topology.branched_boundary_vertices,
        inconsistent_edges=topology.inconsistent_edges,
        duplicate_faces=topology.duplicate_faces,
        boundary_edges=topology.boundary_edges,
        boundary_loops=topology.boundary_components,
        connected_components=topology.connected_components,
        component_euler=topology.component_euler,
        component_signature=topology.component_signature,
        component_detail=topology.component_detail,
        euler_characteristic=topology.euler_characteristic,
        degenerate_faces=degenerate,
        invalid_quads=topology.invalid_quads,
        min_scaled_jacobian=min_jacobian,
        p05_scaled_jacobian=p05_jacobian,
        p95_aspect_ratio=p95_aspect,
        quad_only=(
            topology.all_quads
            and min_jacobian > 0.0
        ),
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
