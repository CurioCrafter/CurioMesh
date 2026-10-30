# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import random
import time
import traceback
from typing import Any

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np

from .atlasq import (
    CandidateEvaluation,
    LARGE_SOURCE_FACES,
    CandidateSpec,
    ScanProfile,
    analyze_object,
    build_candidate_specs,
    candidate_gate_failures,
    evaluate_candidate,
    split_soft_failures,
)
from .metrics import MeshDiagnostics, mesh_diagnostics, uv_mapping_is_valid
from .textures import bake_selected_to_active, ensure_target_uvs
from .triadq import (
    MeshData,
    RemeshOptions as TriadRemeshOptions,
    remesh_mesh as triad_remesh_mesh,
    split_non_manifold_vertex_fans,
)


MAX_FIDELITY_REFERENCE_TRIANGLES = 450_000
MAX_EVALUATED_SNAPSHOT_ESTIMATED_BYTES = 6 * 1024**3
# Above this many source faces the Data Transfer modifier takes tens of minutes; use the Python BVH projection.
UV_MODIFIER_MAX_FACES = 300_000

_solver_errors: list[str] = []


def _take_solver_error() -> str:
    """Why the last solver call failed ('' if it did not say); clears the note."""

    message = "; ".join(_solver_errors)
    _solver_errors.clear()
    return message


@dataclass(slots=True)
class RemeshConfig:
    target_faces: int = 8000
    engine: str = "QUADRIFLOW"
    quality: str = "BALANCED"
    seed: int = 0
    triad_seed_count: int = 6
    triad_feature_angle: float = 35.0
    triad_force_quads: bool = False
    pair_closure_base_faces: int = 0
    pair_closure_output_cap: int = 0
    triad_flow_mode: str = "AUTO"
    triad_field_mode: str = "SMOOTH"
    atlas_strategy: str = "AUTO"
    atlas_candidate_count: int = 3
    topology_policy: str = "PRESERVE"
    require_quads: bool = True
    analysis_sample_faces: int = 120000
    preserve_sharp: bool = True
    preserve_boundary: bool = True
    preserve_seams: bool = True
    use_symmetry: bool = False
    preserve_attributes: bool = True
    smooth_normals: bool = True
    cleanup_strength: float = 0.25
    voxel_repair: bool = True
    shrinkwrap_project: bool = True
    texture_mode: str = "PROJECT"
    bake_fallback: bool = False
    bake_resolution: int = 2048
    bake_margin: int = 8
    output_mode: str = "NEW"
    apply_source_modifiers: bool = True
    keep_debug_objects: bool = False
    debug_logs: bool = False


@dataclass(slots=True)
class RemeshResult:
    success: bool
    output_object: bpy.types.Object | None = None
    diagnostics: MeshDiagnostics = field(default_factory=MeshDiagnostics)
    engine: str = "NONE"
    elapsed_ms: float = 0.0
    message: str = ""
    materials_preserved: bool = False
    uv_status: str = "NONE"
    strategy: str = ""
    profile: str = ""
    candidate_count: int = 0
    geometry_rms: float = 0.0
    geometry_p95: float = 0.0
    geometry_max: float = 0.0
    sampling_status: str = "NOT_MEASURED"
    fidelity_status: str = "NOT_MEASURED"
    quad_only_verified: bool = False
    topology_status: str = "NOT_CHECKED"
    candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class _SurfaceReference:
    bvh: BVHTree
    sample_points: list[Vector]
    diagonal: float
    component_sampling: str = "NOT_REQUESTED"
    fidelity_status: str = "SOURCE_FULL"


@dataclass(slots=True)
class _BoundedSourceEvidence:
    sample_points: list[Vector]
    component_sampling: str
    diagonal: float
    source_triangle_floor: int


@dataclass(frozen=True, slots=True)
class _SurfaceError:
    rms: float
    p95: float
    maximum: float
    component_sampling: str
    fidelity_status: str


class _ContextSnapshot:
    def __init__(self) -> None:
        self.active = bpy.context.view_layer.objects.active
        self.selected = list(bpy.context.selected_objects)
        self.mode = getattr(self.active, "mode", "OBJECT") if self.active else "OBJECT"
        self.final_active: bpy.types.Object | None = None

    def __enter__(self) -> _ContextSnapshot:
        try:
            if self.active is not None and self.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        try:
            for selected in bpy.context.selected_objects:
                selected.select_set(False)
            if self.final_active is not None and self.final_active.name in bpy.data.objects:
                self.final_active.select_set(True)
                bpy.context.view_layer.objects.active = self.final_active
                return
            for selected in self.selected:
                if selected.name in bpy.data.objects:
                    selected.select_set(True)
            if self.active and self.active.name in bpy.data.objects:
                bpy.context.view_layer.objects.active = self.active
                if self.mode != "OBJECT":
                    try:
                        bpy.ops.object.mode_set(mode=self.mode)
                    except Exception:
                        pass
        except Exception:
            pass


def _log(config: RemeshConfig, *parts: object) -> None:
    if config.debug_logs:
        print("[CurioMesh]", *parts)


def _collection_for(obj: bpy.types.Object) -> bpy.types.Collection:
    if obj.users_collection:
        return obj.users_collection[0]
    return bpy.context.collection


def _link_object_like(source: bpy.types.Object, obj: bpy.types.Object) -> None:
    try:
        _collection_for(source).objects.link(obj)
    except Exception:
        bpy.context.collection.objects.link(obj)


def _set_active_only(obj: bpy.types.Object) -> None:
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _remove_object(obj: bpy.types.Object | None, *, remove_mesh: bool = True) -> None:
    if obj is None or obj.name not in bpy.data.objects:
        return
    mesh = obj.data if getattr(obj, "type", None) == "MESH" else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if remove_mesh and mesh is not None and mesh.users == 0:
        try:
            bpy.data.meshes.remove(mesh)
        except Exception:
            pass


def _object_diagonal(obj: bpy.types.Object) -> float:
    if not getattr(obj.data, "vertices", None) or not obj.data.vertices:
        return 1.0
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minv = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maxv = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return max((maxv - minv).length, 1e-6)


def _copy_material_slots(source: bpy.types.Object, target: bpy.types.Object) -> int:
    target.data.materials.clear()
    copied = 0
    for material in getattr(source.data, "materials", []):
        try:
            target.data.materials.append(material)
            copied += 1
        except Exception:
            pass
    return copied


def _estimated_mesh_copy_bytes(mesh: bpy.types.Mesh) -> int:
    """Conservative custom-data-aware estimate for one evaluated mesh copy."""

    return (
        len(mesh.vertices) * 96
        + len(mesh.edges) * 48
        + len(mesh.loops) * 32
        + len(mesh.polygons) * 96
    )


def _make_evaluated_snapshot(source: bpy.types.Object, config: RemeshConfig) -> bpy.types.Object:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = source.evaluated_get(depsgraph)
    estimated_bytes = _estimated_mesh_copy_bytes(evaluated.data)
    if estimated_bytes > MAX_EVALUATED_SNAPSHOT_ESTIMATED_BYTES:
        gib = estimated_bytes / float(1024**3)
        limit_gib = MAX_EVALUATED_SNAPSHOT_ESTIMATED_BYTES / float(1024**3)
        raise RuntimeError(
            f"The evaluated scan needs an estimated {gib:.1f} GiB copy, above CurioMesh's "
            f"{limit_gib:.0f} GiB safety guard. Split or pre-decimate the scan before retopology."
        )
    _log(config, f"Evaluated mesh copy estimate: {estimated_bytes / float(1024**2):.1f} MiB")
    try:
        mesh = bpy.data.meshes.new_from_object(
            evaluated,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
    except TypeError:
        mesh = bpy.data.meshes.new_from_object(evaluated)

    mesh.name = f"{source.name}_CurioMeshSourceMesh"
    snapshot = bpy.data.objects.new(f"{source.name}_CurioMeshSource", mesh)
    snapshot.matrix_world = source.matrix_world.copy()
    snapshot.hide_render = True
    snapshot.hide_select = True
    snapshot.display_type = "WIRE"
    _link_object_like(source, snapshot)
    _copy_material_slots(source, snapshot)
    _log(config, f"Created source snapshot with {len(mesh.polygons)} faces")
    return snapshot


def _make_working_copy(source_snapshot: bpy.types.Object, source_obj: bpy.types.Object) -> bpy.types.Object:
    mesh = source_snapshot.data.copy()
    mesh.name = f"{source_obj.name}_CurioMeshWorkMesh"
    work_obj = bpy.data.objects.new(f"{source_obj.name}_CurioMeshWork", mesh)
    work_obj.matrix_world = source_snapshot.matrix_world.copy()
    _link_object_like(source_obj, work_obj)
    _copy_material_slots(source_snapshot, work_obj)
    return work_obj


def _cleanup_mesh(mesh: bpy.types.Mesh, config: RemeshConfig) -> None:
    strength = max(0.0, min(1.0, float(config.cleanup_strength)))
    if strength <= 0.0 and not config.preserve_seams:
        return

    diagonal = 1.0
    if mesh.vertices:
        xs = [vert.co.x for vert in mesh.vertices]
        ys = [vert.co.y for vert in mesh.vertices]
        zs = [vert.co.z for vert in mesh.vertices]
        diagonal = max(
            math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2),
            1e-6,
        )
    merge_distance = diagonal * (1e-7 + strength * 2e-5)

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        if bm.verts and strength > 0.0:
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=merge_distance)
            try:
                bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=merge_distance * 0.5)
            except Exception:
                pass
        if bm.faces:
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.normal_update()
        bm.to_mesh(mesh)
    finally:
        bm.free()

    if config.preserve_seams:
        for edge in mesh.edges:
            try:
                if edge.use_seam:
                    edge.use_edge_sharp = True
            except Exception:
                pass

    try:
        mesh.validate(clean_customdata=False)
    except Exception:
        pass
    mesh.update(calc_edges=True)


def _auto_voxel_size(obj: bpy.types.Object, config: RemeshConfig) -> float:
    diag = _object_diagonal(obj)
    target = max(16, int(config.target_faces))
    density = math.sqrt(float(target))
    quality_factor = {"DRAFT": 0.8, "BALANCED": 1.0, "HERO": 1.25}.get(config.quality, 1.0)
    return max(diag / max(32.0, density * 2.0 * quality_factor), diag * 0.002)


def _voxel_repair(obj: bpy.types.Object, config: RemeshConfig) -> bool:
    _set_active_only(obj)
    voxel_size = _auto_voxel_size(obj, config)
    try:
        if hasattr(obj.data, "remesh_voxel_size"):
            obj.data.remesh_voxel_size = voxel_size
        if hasattr(obj.data, "remesh_voxel_adaptivity"):
            obj.data.remesh_voxel_adaptivity = 0.0
        bpy.ops.object.voxel_remesh()
        _log(config, f"Voxel repair succeeded at voxel_size={voxel_size:.5f}")
        return True
    except Exception:
        traceback.print_exc()
    return False


def _quadriflow_kwargs(config: RemeshConfig) -> dict[str, Any]:
    return {
        "use_mesh_symmetry": bool(config.use_symmetry),
        "use_preserve_sharp": bool(config.preserve_sharp),
        "use_preserve_boundary": bool(config.preserve_boundary),
        "preserve_attributes": bool(config.preserve_attributes),
        "smooth_normals": bool(config.smooth_normals),
        "mode": "FACES",
        "target_faces": max(1, int(config.target_faces)),
        "seed": max(0, int(config.seed)),
    }


def _run_quadriflow(obj: bpy.types.Object, config: RemeshConfig) -> bool:
    _set_active_only(obj)
    kwargs = _quadriflow_kwargs(config)
    try:
        try:
            result = bpy.ops.object.quadriflow_remesh(**kwargs)
        except TypeError:
            minimal = {
                "mode": "FACES",
                "target_faces": max(1, int(config.target_faces)),
                "seed": max(0, int(config.seed)),
            }
            result = bpy.ops.object.quadriflow_remesh(**minimal)
    except Exception as exc:
        traceback.print_exc()
        _solver_errors.append(f"QuadriFlow error: {exc}"[:200])
        return False
    if "FINISHED" not in result:
        _solver_errors.append(f"QuadriFlow returned {sorted(result)}")
        return False
    if not len(obj.data.polygons):
        _solver_errors.append("QuadriFlow produced an empty mesh")
        return False
    return True


def _meshdata_from_object(obj: bpy.types.Object) -> MeshData:
    mesh = obj.data
    vertices = [vert.co[:] for vert in mesh.vertices]
    faces = [tuple(poly.vertices[:]) for poly in mesh.polygons]
    materials = [str(poly.material_index) for poly in mesh.polygons]

    uvs = None
    face_uvs = None
    if mesh.uv_layers:
        uv_layer = mesh.uv_layers.active
        uv_lookup: dict[tuple[float, float], int] = {}
        uv_values: list[tuple[float, float]] = []
        face_uvs = []
        for poly in mesh.polygons:
            uv_face = []
            for loop_index in poly.loop_indices:
                uv = uv_layer.data[loop_index].uv
                key = (round(float(uv.x), 8), round(float(uv.y), 8))
                uv_index = uv_lookup.get(key)
                if uv_index is None:
                    uv_index = len(uv_values)
                    uv_lookup[key] = uv_index
                    uv_values.append(key)
                uv_face.append(uv_index)
            face_uvs.append(tuple(uv_face))
        if uv_values:
            import numpy as np

            uvs = np.asarray(uv_values, dtype=float)

    return MeshData(
        vertices=vertices,
        faces=faces,
        face_materials=materials,
        uvs=uvs,
        face_uvs=face_uvs,
        name=obj.name,
    )


def _apply_meshdata_to_object(obj: bpy.types.Object, data: MeshData) -> None:
    old_mesh = obj.data
    mesh = bpy.data.meshes.new(name=f"{obj.name}_TriadQLiteMesh")
    mesh.from_pydata(data.vertices.tolist(), [], [tuple(face) for face in data.faces])
    mesh.update(calc_edges=True)
    for index, material in enumerate(data.face_materials[: len(mesh.polygons)]):
        try:
            mesh.polygons[index].material_index = max(0, int(material))
        except Exception:
            mesh.polygons[index].material_index = 0
    if data.uvs is not None and data.face_uvs is not None:
        try:
            uv_layer = mesh.uv_layers.new(name="UVMap")
            for poly_index, poly in enumerate(mesh.polygons):
                if poly_index >= len(data.face_uvs):
                    break
                uv_face = data.face_uvs[poly_index]
                for corner, loop_index in enumerate(poly.loop_indices):
                    if corner >= len(uv_face):
                        continue
                    uv_index = uv_face[corner]
                    if 0 <= uv_index < len(data.uvs):
                        uv_layer.data[loop_index].uv = data.uvs[uv_index][:2]
        except Exception:
            traceback.print_exc()
    obj.data = mesh
    try:
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    except Exception:
        pass


def _run_triadq_lite(obj: bpy.types.Object, config: RemeshConfig) -> bool:
    source = _meshdata_from_object(obj)
    mode = str(config.triad_flow_mode or "AUTO").upper()
    options = TriadRemeshOptions(
        target_faces=max(1, int(config.target_faces)),
        mode=mode,
        seed_count=max(1, int(config.triad_seed_count)),
        feature_angle_deg=float(config.triad_feature_angle),
        force_quads=bool(config.triad_force_quads),
        preserve_material_boundaries=True,
        preserve_uv_seams=bool(config.preserve_seams),
        field_mode=str(config.triad_field_mode or "SMOOTH").upper(),
    )
    result, report = triad_remesh_mesh(source, options)
    if not report.success or not result.faces:
        _log(config, f"TRIAD-Q Lite failed: {report.message}")
        _solver_errors.append(f"TRIAD-Q Lite: {report.message}"[:300])
        return False
    _apply_meshdata_to_object(obj, result)
    obj["curiomesh_triadq_mode"] = report.mode
    obj["curiomesh_triadq_seed"] = int(report.selected_seed)
    obj["curiomesh_triadq_score"] = float(report.score)
    obj["curiomesh_triadq_feature_edges"] = int(report.feature_edges)
    obj["curiomesh_vertex_fan_splits"] = int(report.vertex_fan_splits)
    obj["curiomesh_triadq_rescued_pairs"] = int(report.rescued_pairs)
    _log(
        config,
        (
            f"TRIAD-Q Lite: mode={report.mode} seed={report.selected_seed} "
            f"faces={report.output_faces} quad_ratio={report.quad_ratio:.3f} "
            f"vertex_fan_splits={report.vertex_fan_splits} "
            f"rescued_pairs={report.rescued_pairs}"
        ),
    )
    return True


def _apply_decimate_proxy(
    obj: bpy.types.Object,
    target_faces: int,
    config: RemeshConfig,
    *,
    preserve_features: bool,
) -> bool:
    current_faces = len(obj.data.polygons)
    current_triangles = sum(max(1, len(poly.vertices) - 2) for poly in obj.data.polygons)
    target = max(4, int(target_faces))
    if current_triangles <= target:
        return False
    _set_active_only(obj)
    modifier = obj.modifiers.new("CurioMesh target-scale QEM proxy", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = max(0.00001, min(1.0, target / max(1.0, float(current_triangles))))
    try:
        modifier.use_collapse_triangulate = True
    except Exception:
        pass
    if preserve_features:
        try:
            modifier.delimit = {"SHARP", "SEAM", "MATERIAL"}
        except Exception:
            pass
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        obj["curiomesh_proxy_input_faces"] = int(current_faces)
        obj["curiomesh_proxy_faces"] = int(len(obj.data.polygons))
        _log(config, f"Built QEM proxy: {current_faces} -> {len(obj.data.polygons)} faces")
        return True
    except Exception:
        traceback.print_exc()
        try:
            obj.modifiers.remove(modifier)
        except Exception:
            pass
        return False


def _triangle_count(obj: bpy.types.Object) -> int:
    return sum(max(1, len(poly.vertices) - 2) for poly in obj.data.polygons)


def _rotate_independent_triangle_edges(
    obj: bpy.types.Object,
    seed: int,
    *,
    fraction: float = 0.18,
) -> tuple[int, int]:
    """Rotate a deterministic independent set of interior triangle edges.

    Blender's collapse decimator can stop on an irreducible triangulation even
    when a much smaller triangulation of the same surface topology exists.
    Edge rotations change only the diagonalization.  Keeping selected edges
    face-disjoint makes the operation deterministic and avoids interacting
    rotations within a round.
    """

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        candidates = [
            edge
            for edge in bm.edges
            if len(edge.link_faces) == 2
            and all(len(face.verts) == 3 for face in edge.link_faces)
        ]
        candidates.sort(key=lambda edge: tuple(sorted(vertex.index for vertex in edge.verts)))
        random.Random(int(seed)).shuffle(candidates)
        used_faces: set[bmesh.types.BMFace] = set()
        selected: list[bmesh.types.BMEdge] = []
        limit = max(1, int(round(len(bm.faces) * max(0.01, min(0.45, fraction)))))
        for edge in candidates:
            if any(face in used_faces for face in edge.link_faces):
                continue
            selected.append(edge)
            used_faces.update(edge.link_faces)
            if len(selected) >= limit:
                break
        if not selected:
            return 0, 0
        result = bmesh.ops.rotate_edges(
            bm,
            edges=selected,
            use_ccw=bool(int(seed) & 1),
        )
        rotated = len(result.get("edges", ()))
        if rotated:
            bm.to_mesh(obj.data)
            obj.data.update(calc_edges=True)
        return len(selected), rotated
    finally:
        bm.free()


def _proxy_topology_certifiable(diagnostics: MeshDiagnostics) -> bool:
    return bool(
        diagnostics.faces > 0
        and diagnostics.component_signature
        and diagnostics.non_manifold_edges == 0
        and diagnostics.non_manifold_vertices == 0
        and diagnostics.branched_boundary_vertices == 0
        and diagnostics.inconsistent_edges == 0
        and diagnostics.duplicate_faces == 0
        and diagnostics.degenerate_faces == 0
    )


def _proxy_topology_matches(
    reference: MeshDiagnostics,
    candidate: MeshDiagnostics,
) -> bool:
    return bool(
        _proxy_topology_certifiable(candidate)
        and candidate.connected_components == reference.connected_components
        and candidate.euler_characteristic == reference.euler_characteristic
        and candidate.component_signature == reference.component_signature
        and candidate.ngons == 0
    )


def _normalize_proxy_vertex_fans(
    obj: bpy.types.Object,
    output_cap: int,
    config: RemeshConfig,
) -> int:
    """Split coordinate-identical bow-tie fans created by QEM contractions."""

    diagnostics = mesh_diagnostics(obj, output_cap)
    if diagnostics.non_manifold_vertices <= 0:
        return 0
    if (
        diagnostics.non_manifold_edges > 0
        or diagnostics.branched_boundary_vertices > 0
        or diagnostics.inconsistent_edges > 0
        or diagnostics.duplicate_faces > 0
        or diagnostics.degenerate_faces > 0
    ):
        return 0
    normalized, splits = split_non_manifold_vertex_fans(_meshdata_from_object(obj))
    if splits <= 0:
        return 0
    _apply_meshdata_to_object(obj, normalized)
    obj["curiomesh_pair_unlock_vertex_fan_splits"] = int(
        obj.get("curiomesh_pair_unlock_vertex_fan_splits", 0)
    ) + int(splits)
    _log(config, f"Normalized {splits} QEM-created disconnected vertex fan(s)")
    return int(splits)


def _unlock_pair_closure_proxy(
    obj: bpy.types.Object,
    base_target: int,
    output_cap: int,
    config: RemeshConfig,
    *,
    max_rounds: int = 112,
) -> int:
    """Unlock topology-preserving QEM reductions with gated edge rotations.

    Each round is transactional.  A rotated/QEM candidate is retained only
    when it remains a certifiable manifold with the exact same per-component
    Euler/boundary signature.  Invalid or topology-changing rounds restore the
    previous mesh.  The 2.8 closure estimate is conservative for the observed
    PairClosure pairing range; final candidate gates still enforce the actual
    quad count and topology.
    """

    cap = max(4, int(output_cap))
    target = max(4, int(base_target))
    _normalize_proxy_vertex_fans(obj, cap, config)
    baseline = mesh_diagnostics(obj, cap)
    if not _proxy_topology_certifiable(baseline):
        _log(config, "Skipped QEM unlock because the proxy topology is not certifiable")
        return 0

    initial_triangles = _triangle_count(obj)
    if math.ceil(initial_triangles * 2.8) <= cap:
        return 0

    accepted_rounds = 0
    rejected_rounds = 0
    rotated_edges = 0
    no_reduction_rounds = 0
    rounds = max(0, min(128, int(max_rounds)))
    for round_index in range(1, rounds + 1):
        before_triangles = _triangle_count(obj)
        previous_mesh = obj.data.copy()
        previous_mesh.name = f"{obj.data.name}_PreUnlock{round_index:03d}"
        try:
            _selected, rotated = _rotate_independent_triangle_edges(
                obj,
                int(config.seed) + 1000 + round_index,
            )
            if rotated <= 0:
                if previous_mesh.users == 0:
                    bpy.data.meshes.remove(previous_mesh)
                break
            _apply_decimate_proxy(
                obj,
                target,
                config,
                preserve_features=False,
            )
            _normalize_proxy_vertex_fans(obj, cap, config)
            candidate = mesh_diagnostics(obj, cap)
            candidate_triangles = _triangle_count(obj)
            accepted = bool(
                candidate_triangles <= before_triangles
                and _proxy_topology_matches(baseline, candidate)
            )
            if accepted:
                if previous_mesh.users == 0:
                    bpy.data.meshes.remove(previous_mesh)
                accepted_rounds += 1
                rotated_edges += rotated
                baseline = candidate
                if candidate_triangles < before_triangles:
                    no_reduction_rounds = 0
                else:
                    no_reduction_rounds += 1
            else:
                rejected_mesh = obj.data
                obj.data = previous_mesh
                if rejected_mesh.users == 0:
                    bpy.data.meshes.remove(rejected_mesh)
                rejected_rounds += 1
                no_reduction_rounds += 1

            if math.ceil(_triangle_count(obj) * 2.8) <= cap:
                break
            if no_reduction_rounds >= 12:
                break
        except Exception:
            traceback.print_exc()
            rejected_mesh = obj.data
            obj.data = previous_mesh
            if rejected_mesh.users == 0:
                bpy.data.meshes.remove(rejected_mesh)
            rejected_rounds += 1
            break

    final_triangles = _triangle_count(obj)
    obj["curiomesh_pair_unlock_initial_triangles"] = int(initial_triangles)
    obj["curiomesh_pair_unlock_final_triangles"] = int(final_triangles)
    obj["curiomesh_pair_unlock_rounds"] = int(accepted_rounds)
    obj["curiomesh_pair_unlock_rejected_rounds"] = int(rejected_rounds)
    obj["curiomesh_pair_unlock_rotated_edges"] = int(rotated_edges)
    obj["curiomesh_pair_unlock_output_estimate"] = int(math.ceil(final_triangles * 2.8))
    _log(
        config,
        (
            "Topology-gated QEM unlock: "
            f"triangles={initial_triangles}->{final_triangles}, "
            f"rounds={accepted_rounds}, rejected={rejected_rounds}, "
            f"estimated_closure={math.ceil(final_triangles * 2.8)}/{cap}"
        ),
    )
    return accepted_rounds


def _prepare_flow_proxy(obj: bpy.types.Object, config: RemeshConfig) -> None:
    factor, cap = {
        "DRAFT": (6, 150_000),
        "BALANCED": (12, 400_000),
        "HERO": (20, 400_000),
    }.get(config.quality, (12, 400_000))
    budget = min(cap, max(2_000, int(config.target_faces) * factor))
    _apply_decimate_proxy(obj, budget, config, preserve_features=config.preserve_sharp or config.preserve_seams)


def _prepare_pair_closure_proxy(obj: bpy.types.Object, config: RemeshConfig) -> None:
    closure_upper_bound = sum(len(poly.vertices) for poly in obj.data.polygons)
    if closure_upper_bound <= max(4, int(config.target_faces)) * 1.10:
        return
    # With pairing fraction p, face-corner closure emits approximately
    # N_base * (3 - p) quads. p=0.8 is a conservative first estimate.
    configured_base = int(config.pair_closure_base_faces)
    base_target = (
        max(4, configured_base)
        if configured_base > 0
        else max(4, int(round(max(4, config.target_faces) / 2.2)))
    )
    obj["curiomesh_pair_closure_base_target"] = int(base_target)
    _apply_decimate_proxy(
        obj,
        base_target,
        config,
        preserve_features=bool(config.preserve_sharp or config.preserve_seams),
    )
    configured_cap = int(config.pair_closure_output_cap)
    output_cap = (
        max(4, configured_cap)
        if configured_cap > 0
        else max(4, int(math.ceil(config.target_faces * (1.75 if config.quality == "DRAFT" else 1.35))))
    )
    obj["curiomesh_pair_closure_output_cap"] = int(output_cap)
    if configured_base > 0 or not (config.preserve_sharp or config.preserve_seams):
        _unlock_pair_closure_proxy(
            obj,
            base_target,
            output_cap,
            config,
        )


def _clone_candidate(base: bpy.types.Object, index: int) -> bpy.types.Object:
    mesh = base.data.copy()
    mesh.name = f"{base.data.name}_AtlasCandidate{index}"
    candidate = bpy.data.objects.new(f"{base.name}_AtlasCandidate{index}", mesh)
    candidate.matrix_world = base.matrix_world.copy()
    _link_object_like(base, candidate)
    _copy_material_slots(base, candidate)
    for key in base.keys():
        candidate[key] = base[key]
    return candidate


def _apply_shrinkwrap(target: bpy.types.Object, source: bpy.types.Object, config: RemeshConfig) -> bool:
    _set_active_only(target)
    try:
        modifier = target.modifiers.new("CurioMesh detail projection", "SHRINKWRAP")
        modifier.target = source
        modifier.wrap_method = "NEAREST_SURFACEPOINT"
        modifier.offset = 0.0
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        _log(config, "Shrinkwrap detail projection applied")
        return True
    except Exception:
        traceback.print_exc()
        try:
            target.modifiers.remove(modifier)
        except Exception:
            pass
    return False


def _build_source_bvh(source: bpy.types.Object) -> tuple[BVHTree, list[tuple[int, int, int]], list[int]]:
    mesh = source.data
    mesh.calc_loop_triangles()
    verts_world = [source.matrix_world @ vert.co for vert in mesh.vertices]
    triangles = [tuple(tri.vertices) for tri in mesh.loop_triangles]
    polygon_indices = [tri.polygon_index for tri in mesh.loop_triangles]
    return BVHTree.FromPolygons(verts_world, triangles, all_triangles=True), triangles, polygon_indices


def _distance_bvh(obj: bpy.types.Object) -> BVHTree:
    # All surface-error queries are in world space. Blender's FromObject BVH
    # stores object-local coordinates, so using it here silently corrupts
    # scores for translated, rotated, or scaled objects.
    return _build_source_bvh(obj)[0]


def _bounded_triangle_count(obj: bpy.types.Object, limit: int) -> tuple[int, bool]:
    """Count tessellated faces without allocating a loop-triangle collection.

    The returned count stops immediately after ``limit``. This makes the
    fidelity-reference decision proportional to the cap instead of the full
    scan size and, importantly, avoids materializing Python triangle tuples.
    """

    cap = max(1, int(limit))
    count = 0
    for polygon in obj.data.polygons:
        count += max(1, len(polygon.vertices) - 2)
        if count > cap:
            return count, True
    return count, False


def _bounded_source_sample_points(
    obj: bpy.types.Object,
    limit: int,
) -> tuple[list[Vector], str]:
    """Sample a huge source in bounded memory without loop triangulation.

    Face-center stride samples retain source-to-candidate fidelity evidence,
    while a small world-space grid keeps broad spatial coverage. This is not an
    exhaustive component proof, so the status deliberately says ``BOUNDED``.
    """

    face_count = len(obj.data.polygons)
    budget = max(0, int(limit))
    if face_count == 0 or budget == 0:
        return [], "BOUNDED_SOURCE_EMPTY"

    world_bounds = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in world_bounds) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in world_bounds) for axis in range(3)))
    span = maximum - minimum
    grid_resolution = max(2, min(8, int(round(budget ** (1.0 / 3.0)))))
    grid_budget = min(budget // 4, grid_resolution**3)
    stride_budget = max(1, budget - grid_budget)
    stride = max(1, math.ceil(face_count / stride_budget))
    matrix = obj.matrix_world
    grid_points: dict[tuple[int, int, int], Vector] = {}
    stride_points: list[Vector] = []

    for index, polygon in enumerate(obj.data.polygons):
        center = matrix @ polygon.center
        if len(grid_points) < grid_budget:
            cell = tuple(
                min(
                    grid_resolution - 1,
                    max(0, int(((center[axis] - minimum[axis]) / max(span[axis], 1e-20)) * grid_resolution)),
                )
                for axis in range(3)
            )
            grid_points.setdefault(cell, center)
        if index % stride == stride // 2 and len(stride_points) < stride_budget:
            stride_points.append(center)

    points = (list(grid_points.values()) + stride_points)[:budget]
    return points, f"BOUNDED_SOURCE_FACES_{len(points)}_OF_{face_count}"


def _capture_bounded_source_evidence(
    source: bpy.types.Object,
    sample_limit: int,
    source_triangle_floor: int,
) -> _BoundedSourceEvidence:
    points, component_sampling = _bounded_source_sample_points(source, sample_limit)
    return _BoundedSourceEvidence(
        sample_points=points,
        component_sampling=component_sampling,
        diagonal=_object_diagonal(source),
        source_triangle_floor=max(1, int(source_triangle_floor)),
    )


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


def _component_sample_points(
    obj: bpy.types.Object,
    limit: int,
    *,
    max_vertices: int = 2_000_000,
    max_edges: int = 6_000_000,
) -> tuple[list[Vector], str]:
    """Return representatives biased toward small connected components.

    NumPy-backed union-find avoids Python adjacency sets on scan-scale meshes.
    Above the explicit memory guard, spatial and area strata remain active and
    the returned status records that component coverage was not exhaustive.
    """

    mesh = obj.data
    vertex_count = len(mesh.vertices)
    edge_count = len(mesh.edges)
    if limit <= 0 or vertex_count == 0 or edge_count == 0:
        return [], "NO_SURFACE_COMPONENTS"
    if vertex_count > max_vertices or edge_count > max_edges:
        return [], f"SKIPPED_LIMIT_{vertex_count}_VERTICES_{edge_count}_EDGES"

    dtype = np.int32 if vertex_count < np.iinfo(np.int32).max else np.int64
    parents = np.arange(vertex_count, dtype=dtype)
    sizes = np.ones(vertex_count, dtype=np.int32)
    edge_vertices = np.empty(edge_count * 2, dtype=dtype)
    mesh.edges.foreach_get("vertices", edge_vertices)

    def root(raw: int) -> int:
        current = int(raw)
        while int(parents[current]) != current:
            parents[current] = parents[int(parents[current])]
            current = int(parents[current])
        return current

    for raw_a, raw_b in edge_vertices.reshape((-1, 2)):
        a = root(int(raw_a))
        b = root(int(raw_b))
        if a == b:
            continue
        if sizes[a] < sizes[b]:
            a, b = b, a
        parents[b] = a
        sizes[a] += sizes[b]

    while True:
        compressed = parents[parents]
        if np.array_equal(compressed, parents):
            break
        parents = compressed
    roots, first_indices, counts = np.unique(parents, return_index=True, return_counts=True)
    del roots, edge_vertices
    surface_mask = counts >= 2
    first_indices = first_indices[surface_mask]
    counts = counts[surface_mask]
    component_count = len(first_indices)
    if component_count == 0:
        return [], "NO_SURFACE_COMPONENTS"

    if component_count > limit:
        # The area strata already favor large islands. Spend the component
        # quota on the small pieces most likely to vanish during QEM collapse.
        selected = np.argsort(counts, kind="stable")[:limit]
        first_indices = first_indices[selected]
        status = f"CAPPED_{len(first_indices)}_OF_{component_count}"
    else:
        status = f"ALL_{component_count}"
    matrix = obj.matrix_world
    return [matrix @ mesh.vertices[int(index)].co for index in first_indices], status


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    current = max(1, int(index))
    while current:
        current, digit = divmod(current, base)
        denominator *= base
        value += digit / denominator
    return value


def _triangle_point(a: Vector, b: Vector, c: Vector, index: int) -> Vector:
    root_u = math.sqrt(_radical_inverse(index + 1, 2))
    v = _radical_inverse(index + 1, 3)
    return (1.0 - root_u) * a + root_u * (1.0 - v) * b + root_u * v * c


def _stratified_surface_points(
    obj: bpy.types.Object,
    limit: int,
    *,
    include_components: bool,
) -> tuple[list[Vector], str]:
    mesh = obj.data
    if limit <= 0 or not mesh.polygons:
        return [], "EMPTY"
    mesh.calc_loop_triangles()
    if not mesh.loop_triangles:
        return [], "EMPTY"

    component_points: list[Vector] = []
    component_status = "NOT_REQUESTED"
    if include_components:
        component_points, component_status = _component_sample_points(
            obj,
            min(1024, max(1, limit // 4)),
        )

    world_bounds = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector(
        (
            min(point.x for point in world_bounds),
            min(point.y for point in world_bounds),
            min(point.z for point in world_bounds),
        )
    )
    maximum = Vector(
        (
            max(point.x for point in world_bounds),
            max(point.y for point in world_bounds),
            max(point.z for point in world_bounds),
        )
    )
    span = maximum - minimum
    grid_resolution = 8
    grid_points: dict[tuple[int, int, int], Vector] = {}
    total_area = 0.0
    matrix = obj.matrix_world
    for triangle in mesh.loop_triangles:
        a, b, c = (matrix @ mesh.vertices[index].co for index in triangle.vertices)
        area = 0.5 * (b - a).cross(c - a).length
        if not math.isfinite(area) or area <= 0.0:
            continue
        total_area += area
        center = (a + b + c) / 3.0
        cell = tuple(
            min(
                grid_resolution - 1,
                max(0, int(((center[axis] - minimum[axis]) / max(span[axis], 1e-20)) * grid_resolution)),
            )
            for axis in range(3)
        )
        grid_points.setdefault(cell, center)

    points = component_points + list(grid_points.values())
    if len(points) >= limit:
        return points[:limit], component_status
    area_budget = limit - len(points)
    if total_area <= 0.0:
        return points, component_status

    interval = total_area / max(1, area_budget)
    threshold = interval * 0.5
    cumulative = 0.0
    sample_index = 0
    for triangle in mesh.loop_triangles:
        a, b, c = (matrix @ mesh.vertices[index].co for index in triangle.vertices)
        area = 0.5 * (b - a).cross(c - a).length
        if not math.isfinite(area) or area <= 0.0:
            continue
        upper = cumulative + area
        while threshold <= upper and sample_index < area_budget:
            points.append(_triangle_point(a, b, c, sample_index))
            sample_index += 1
            threshold += interval
        cumulative = upper
        if sample_index >= area_budget:
            break
    return points[:limit], component_status


def _make_surface_reference(
    source: bpy.types.Object,
    sample_limit: int = 5000,
    *,
    reference_proxy: bpy.types.Object | None = None,
    max_reference_triangles: int = MAX_FIDELITY_REFERENCE_TRIANGLES,
    bounded_source_evidence: _BoundedSourceEvidence | None = None,
) -> _SurfaceReference:
    cap = max(1, int(max_reference_triangles))
    if bounded_source_evidence is None:
        source_triangles, source_exceeds_cap = _bounded_triangle_count(source, cap)
    else:
        source_triangles = bounded_source_evidence.source_triangle_floor
        source_exceeds_cap = True
    reference_object = source

    if source_exceeds_cap:
        if reference_proxy is None:
            raise RuntimeError(
                "The source exceeds the bounded fidelity-reference budget and no reduced proxy was supplied."
            )
        proxy_triangles, proxy_exceeds_cap = _bounded_triangle_count(reference_proxy, cap)
        if proxy_exceeds_cap:
            raise RuntimeError(
                f"The fidelity proxy exceeds the {cap:,}-triangle memory budget; lower the target face count."
            )
        reference_object = reference_proxy
        if bounded_source_evidence is None:
            points, component_status = _bounded_source_sample_points(source, sample_limit)
            diagonal = _object_diagonal(source)
        else:
            points = bounded_source_evidence.sample_points
            component_status = bounded_source_evidence.component_sampling
            diagonal = bounded_source_evidence.diagonal
        fidelity_status = (
            f"BOUNDED_PROXY_BVH_{proxy_triangles}_TRIS_SOURCE_GE_{source_triangles}_TRIS"
        )
    else:
        points, component_status = _stratified_surface_points(
            source,
            sample_limit,
            include_components=True,
        )
        fidelity_status = f"SOURCE_FULL_BVH_{source_triangles}_TRIS"
        diagonal = _object_diagonal(source)

    return _SurfaceReference(
        bvh=_distance_bvh(reference_object),
        sample_points=points,
        diagonal=diagonal,
        component_sampling=component_status,
        fidelity_status=fidelity_status,
    )


def _sampled_surface_error_stats(
    source: bpy.types.Object,
    target: bpy.types.Object,
    *,
    sample_limit: int = 5000,
    reference: _SurfaceReference | None = None,
) -> _SurfaceError:
    if not target.data.polygons or (reference is None and not source.data.polygons):
        return _SurfaceError(float("inf"), float("inf"), float("inf"), "EMPTY", "EMPTY")
    if reference is None:
        reference = _make_surface_reference(source, sample_limit)
    target_bvh = _distance_bvh(target)
    distances: list[float] = []
    target_points, _component_status = _stratified_surface_points(
        target,
        sample_limit,
        include_components=False,
    )
    for point in target_points:
        hit = reference.bvh.find_nearest(point)
        if hit is not None and hit[3] is not None and math.isfinite(hit[3]):
            distances.append(float(hit[3]))
    for point in reference.sample_points:
        hit = target_bvh.find_nearest(point)
        if hit is not None and hit[3] is not None and math.isfinite(hit[3]):
            distances.append(float(hit[3]))
    if not distances:
        return _SurfaceError(
            float("inf"),
            float("inf"),
            float("inf"),
            reference.component_sampling,
            reference.fidelity_status,
        )
    rms = math.sqrt(sum(distance * distance for distance in distances) / len(distances)) / reference.diagonal
    p95 = _percentile(distances, 0.95) / reference.diagonal
    maximum = max(distances) / reference.diagonal
    return _SurfaceError(
        rms,
        p95,
        maximum,
        reference.component_sampling,
        reference.fidelity_status,
    )


def _sampled_surface_error(
    source: bpy.types.Object,
    target: bpy.types.Object,
    *,
    sample_limit: int = 5000,
    reference: _SurfaceReference | None = None,
) -> tuple[float, float]:
    """Compatibility wrapper returning RMS and p95 normalized error."""

    stats = _sampled_surface_error_stats(
        source,
        target,
        sample_limit=sample_limit,
        reference=reference,
    )
    return stats.rms, stats.p95


def _run_atlas_candidate(
    candidate: bpy.types.Object,
    source: bpy.types.Object,
    config: RemeshConfig,
    profile: ScanProfile,
    spec: CandidateSpec,
    reference: _SurfaceReference,
) -> CandidateEvaluation:
    started = time.perf_counter()
    candidate_config = replace(
        config,
        seed=spec.seed,
        preserve_sharp=config.preserve_sharp and spec.preserve_sharp,
        preserve_boundary=config.preserve_boundary and spec.preserve_boundary,
        triad_force_quads=True if spec.pair_closure else config.triad_force_quads,
    )
    _take_solver_error()
    if spec.pair_closure:
        _prepare_pair_closure_proxy(candidate, candidate_config)
        ok = _run_triadq_lite(candidate, candidate_config)
    else:
        if spec.repair and candidate_config.voxel_repair:
            ok = _voxel_repair(candidate, candidate_config)
        else:
            ok = True
        if ok:
            ok = _run_quadriflow(candidate, candidate_config)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not ok:
        return CandidateEvaluation(
            strategy=spec.strategy,
            label=spec.label,
            valid=False,
            score=-1.0,
            reason=f"solver failed: {_take_solver_error() or 'no reason reported'}",
            faces=0,
            quad_ratio=0.0,
            target_error=1.0,
            geometry_rms=float("inf"),
            geometry_p95=float("inf"),
            geometry_max=float("inf"),
            min_scaled_jacobian=0.0,
            p95_aspect_ratio=0.0,
            extraordinary_ratio=1.0,
            elapsed_ms=elapsed_ms,
        )

    diagnostics = mesh_diagnostics(candidate, config.target_faces)
    geometry = _sampled_surface_error_stats(source, candidate, reference=reference)
    return evaluate_candidate(
        spec,
        diagnostics,
        target_faces=config.target_faces,
        geometry_rms=geometry.rms,
        geometry_p95=geometry.p95,
        geometry_max=geometry.maximum,
        target_edge_normalized=max(profile.target_edge_length / profile.diagonal, 1e-9),
        require_quads=config.require_quads,
        topology_policy=config.topology_policy,
        profile=profile,
        quality=config.quality,
        elapsed_ms=elapsed_ms,
    )


def _run_atlasq(
    base_obj: bpy.types.Object,
    source: bpy.types.Object,
    config: RemeshConfig,
    profile: ScanProfile,
    reference: _SurfaceReference | None = None,
) -> tuple[bpy.types.Object | None, CandidateEvaluation | None, list[CandidateEvaluation]]:
    specs = build_candidate_specs(
        profile,
        strategy=config.atlas_strategy,
        quality=config.quality,
        seed=config.seed,
        candidate_count=config.atlas_candidate_count,
        topology_policy=config.topology_policy,
        voxel_repair=config.voxel_repair,
    )
    if not all(spec.pair_closure for spec in specs):
        _prepare_flow_proxy(base_obj, config)
    pristine_mesh = base_obj.data.copy()
    if reference is None:
        reference = _make_surface_reference(
            source,
            5000,
            reference_proxy=base_obj,
        )
    evaluations: list[CandidateEvaluation] = []
    candidate_objects: list[tuple[bpy.types.Object, CandidateEvaluation]] = []

    def run_spec(spec: CandidateSpec, index: int) -> None:
        candidate = base_obj if index == 0 else _clone_candidate(base_obj, index)
        if index > 0:
            replacement = pristine_mesh.copy()
            old_mesh = candidate.data
            candidate.data = replacement
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
        try:
            evaluation = _run_atlas_candidate(candidate, source, config, profile, spec, reference)
        except Exception as exc:
            traceback.print_exc()
            evaluation = CandidateEvaluation(
                strategy=spec.strategy,
                label=spec.label,
                valid=False,
                score=-1.0,
                reason=f"candidate exception: {type(exc).__name__}: {exc}"[:240],
                faces=0,
                quad_ratio=0.0,
                target_error=1.0,
                geometry_rms=float("inf"),
                geometry_p95=float("inf"),
                geometry_max=float("inf"),
                min_scaled_jacobian=0.0,
                p95_aspect_ratio=0.0,
                extraordinary_ratio=1.0,
                elapsed_ms=0.0,
            )
        evaluations.append(evaluation)
        candidate_objects.append((candidate, evaluation))
        _log(
            config,
            (
                f"Atlas-Q {evaluation.label}: valid={evaluation.valid} score={evaluation.score:.4f} "
                f"faces={evaluation.faces} p95={evaluation.geometry_p95:.6f} reason={evaluation.reason}"
            ),
        )

    for index, spec in enumerate(specs):
        run_spec(spec, index)

    valid = [(obj, evaluation) for obj, evaluation in candidate_objects if evaluation.valid]
    pair_attempted = any(spec.pair_closure for spec in specs)
    if not valid and config.require_quads and not pair_attempted:
        fallback = CandidateSpec(
            strategy="PAIR_CLOSURE",
            label="PairClosure exact all-quad fallback",
            seed=config.seed + 104729,
            pair_closure=True,
        )
        run_spec(fallback, len(specs))
        valid = [(obj, evaluation) for obj, evaluation in candidate_objects if evaluation.valid]

    selected_obj = None
    selected_evaluation = None
    if valid:
        selected_obj, selected_evaluation = max(valid, key=lambda item: item[1].score)

    for candidate, _evaluation in candidate_objects:
        if candidate is not selected_obj:
            _remove_object(candidate)
    if pristine_mesh.users == 0:
        bpy.data.meshes.remove(pristine_mesh)
    return selected_obj, selected_evaluation, evaluations


def _transfer_material_indices_with_modifier(
    source: bpy.types.Object,
    target: bpy.types.Object,
) -> bool:
    _set_active_only(target)
    modifier = target.modifiers.new("CurioMesh material transfer", "DATA_TRANSFER")
    modifier.object = source
    try:
        modifier.use_poly_data = True
        modifier.data_types_polys = {"MATERIAL_INDEX"}
        modifier.poly_mapping = "NEAREST"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        return True
    except Exception:
        traceback.print_exc()
        try:
            target.modifiers.remove(modifier)
        except Exception:
            pass
        return False


def _map_material_indices(
    source: bpy.types.Object,
    target: bpy.types.Object,
    *,
    allow_python_bvh: bool = True,
) -> bool:
    if not source.data.materials or not target.data.materials or not source.data.polygons:
        return False
    if not allow_python_bvh:
        return _transfer_material_indices_with_modifier(source, target)

    bvh, _triangles, polygon_indices = _build_source_bvh(source)
    assigned = 0
    for poly in target.data.polygons:
        center = Vector((0.0, 0.0, 0.0))
        for loop_index in poly.loop_indices:
            vertex_index = target.data.loops[loop_index].vertex_index
            center += target.matrix_world @ target.data.vertices[vertex_index].co
        center *= 1.0 / max(1, len(poly.loop_indices))
        hit = bvh.find_nearest(center)
        if hit is None or hit[2] is None:
            continue
        source_poly = source.data.polygons[polygon_indices[hit[2]]]
        poly.material_index = min(source_poly.material_index, max(0, len(target.data.materials) - 1))
        assigned += 1
    return assigned > 0


def _transfer_uvs_with_modifier(source: bpy.types.Object, target: bpy.types.Object) -> bool:
    if not source.data.uv_layers:
        return False
    if not target.data.uv_layers:
        target.data.uv_layers.new(name=source.data.uv_layers.active.name or "UVMap")

    _set_active_only(target)
    modifier = target.modifiers.new("CurioMesh UV transfer", "DATA_TRANSFER")
    modifier.object = source
    modifier.use_loop_data = True
    try:
        modifier.data_types_loops = {"UV"}
    except Exception:
        modifier.loop_data_types = {"UV"}
    for attr, value in (
        ("loop_mapping", "NEAREST_FACE_INTERPOLATED"),
        ("layers_uv_select_src", "ACTIVE"),
        ("layers_uv_select_dst", "ACTIVE"),
        ("mix_mode", "REPLACE"),
    ):
        try:
            setattr(modifier, attr, value)
        except Exception:
            pass

    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    except Exception:
        try:
            target.modifiers.remove(modifier)
        except Exception:
            pass
        return False
    return uv_mapping_is_valid(target)


def _barycentric(point: Vector, a: Vector, b: Vector, c: Vector) -> tuple[float, float, float]:
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return 1.0, 0.0, 0.0
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def _project_uvs_bvh(source: bpy.types.Object, target: bpy.types.Object) -> bool:
    if not source.data.uv_layers:
        return False
    source_uv = source.data.uv_layers.active
    if source_uv is None:
        return False

    source.data.calc_loop_triangles()
    source_vertices = [source.matrix_world @ vert.co for vert in source.data.vertices]
    triangles = [tuple(tri.vertices) for tri in source.data.loop_triangles]
    triangle_loop_uvs = [
        [source_uv.data[loop_index].uv.copy() for loop_index in tri.loops]
        for tri in source.data.loop_triangles
    ]
    bvh = BVHTree.FromPolygons(source_vertices, triangles, all_triangles=True)

    if not target.data.uv_layers:
        target.data.uv_layers.new(name=source_uv.name or "UVMap")
    target.data.uv_layers.active.name = source_uv.name or target.data.uv_layers.active.name
    target_uv = target.data.uv_layers.active

    assigned = 0
    for poly in target.data.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = target.data.loops[loop_index].vertex_index
            point = target.matrix_world @ target.data.vertices[vertex_index].co
            hit = bvh.find_nearest(point)
            if hit is None or hit[2] is None:
                continue
            hit_point, _normal, tri_index, _distance = hit
            a_index, b_index, c_index = triangles[tri_index]
            u, v, w = _barycentric(
                hit_point,
                source_vertices[a_index],
                source_vertices[b_index],
                source_vertices[c_index],
            )
            uv0, uv1, uv2 = triangle_loop_uvs[tri_index]
            target_uv.data[loop_index].uv = uv0 * u + uv1 * v + uv2 * w
            assigned += 1
    return assigned > 0 and uv_mapping_is_valid(target)


def _preserve_textures(
    source: bpy.types.Object,
    target: bpy.types.Object,
    config: RemeshConfig,
    *,
    allow_python_bvh: bool = True,
) -> str:
    mode = str(config.texture_mode or "PROJECT").upper()
    if mode == "NONE":
        return "SKIPPED"
    if not source.data.uv_layers and mode != "BAKE":
        return "NO_SOURCE_UV"

    transferred = False
    projected = False
    baked = False

    huge_source = len(source.data.polygons) > UV_MODIFIER_MAX_FACES
    if mode in {"TRANSFER", "PROJECT"} and not huge_source:
        transferred = _transfer_uvs_with_modifier(source, target)
    if mode in {"TRANSFER", "PROJECT"} and (huge_source or (mode == "PROJECT" and allow_python_bvh)):
        if not uv_mapping_is_valid(target):
            projected = _project_uvs_bvh(source, target)
    if mode == "BAKE" or (config.bake_fallback and not uv_mapping_is_valid(target)):
        try:
            ensure_target_uvs(target)
            bake_selected_to_active(
                source,
                target,
                maps={"COLOR"},
                resolution=int(config.bake_resolution),
                margin=int(config.bake_margin),
            )
            baked = True
        except Exception:
            traceback.print_exc()

    if baked:
        return "BAKED"
    if projected:
        return "PROJECTED"
    if transferred:
        return "TRANSFERRED"
    return "FAILED" if mode != "NONE" else "SKIPPED"


def _strip_source_modifiers(obj: bpy.types.Object) -> None:
    for modifier in list(obj.modifiers):
        try:
            obj.modifiers.remove(modifier)
        except Exception:
            pass


def _unique_object_name(base: str) -> str:
    if base not in bpy.data.objects:
        return base
    index = 1
    while True:
        candidate = f"{base}.{index:03d}"
        if candidate not in bpy.data.objects:
            return candidate
        index += 1


def _finalize_output(
    source_obj: bpy.types.Object,
    work_obj: bpy.types.Object,
    config: RemeshConfig,
) -> tuple[bpy.types.Object, bpy.types.Object | None]:
    if config.output_mode == "REPLACE":
        result_mesh = work_obj.data.copy()
        result_mesh.name = f"{source_obj.name}_CurioMesh"
        if config.apply_source_modifiers:
            _strip_source_modifiers(source_obj)
        old_source_mesh = source_obj.data
        source_obj.data = result_mesh
        _remove_object(work_obj)
        if old_source_mesh.users == 0:
            try:
                bpy.data.meshes.remove(old_source_mesh)
            except Exception:
                pass
        output_obj = source_obj
        work_obj = None
    else:
        output_obj = work_obj
        output_obj.name = _unique_object_name(f"{source_obj.name}_CurioMesh")
        output_obj.data.name = f"{output_obj.name}_Mesh"
        output_obj.hide_render = False
        output_obj.hide_select = False
        work_obj = None

    _set_active_only(output_obj)
    return output_obj, work_obj


def _annotate_output(output_obj: bpy.types.Object, result: RemeshResult) -> None:
    output_obj["curiomesh_engine"] = result.engine
    output_obj["curiomesh_elapsed_ms"] = round(result.elapsed_ms, 2)
    output_obj["curiomesh_quad_ratio"] = round(result.diagnostics.quad_ratio, 4)
    output_obj["curiomesh_face_count_error"] = round(result.diagnostics.face_count_error, 4)
    output_obj["curiomesh_uv_status"] = result.uv_status
    output_obj["curiomesh_materials_preserved"] = bool(result.materials_preserved)
    output_obj["curiomesh_strategy"] = result.strategy
    output_obj["curiomesh_scan_profile"] = result.profile
    output_obj["curiomesh_candidate_count"] = int(result.candidate_count)
    output_obj["curiomesh_geometry_rms"] = round(result.geometry_rms, 8)
    output_obj["curiomesh_geometry_p95"] = round(result.geometry_p95, 8)
    output_obj["curiomesh_geometry_max"] = round(result.geometry_max, 8)
    output_obj["curiomesh_component_sampling"] = result.sampling_status
    output_obj["curiomesh_fidelity_reference"] = result.fidelity_status
    output_obj["curiomesh_quad_only_verified"] = bool(result.quad_only_verified)
    output_obj["curiomesh_topology_status"] = result.topology_status
    if result.candidates:
        output_obj["curiomesh_candidates_json"] = json.dumps(
            result.candidates,
            separators=(",", ":"),
            allow_nan=False,
        )


def _merge_proxy_topology_profile(source: ScanProfile, proxy: ScanProfile) -> ScanProfile:
    """Route and validate from an exact reduced proxy without overstating source proof."""

    if not proxy.exact_topology:
        return source
    return replace(
        source,
        category=proxy.category,
        recommended_strategy=proxy.recommended_strategy,
        topology_basis="PROXY_EXACT",
        boundary_edges=proxy.boundary_edges,
        boundary_loops=proxy.boundary_loops,
        connected_components=proxy.connected_components,
        component_euler=proxy.component_euler,
        component_signature=proxy.component_signature,
        component_detail=proxy.component_detail,
        non_manifold_edges=proxy.non_manifold_edges,
        non_manifold_vertices=proxy.non_manifold_vertices,
        branched_boundary_vertices=proxy.branched_boundary_vertices,
        inconsistent_edges=proxy.inconsistent_edges,
        duplicate_faces=proxy.duplicate_faces,
        euler_characteristic=proxy.euler_characteristic,
        warnings=source.warnings
        + (
            "Topology gates use the exact reduced proxy; full-resolution source topology remains unverified.",
        ),
    )


def _topology_status(profile: ScanProfile | None, topology_policy: str) -> str:
    if topology_policy == "REPAIR":
        return "REPAIR_ALLOWED"
    if profile is None:
        return "NOT_CHECKED"
    if profile.topology_basis == "SOURCE_EXACT":
        return "SOURCE_EXACT_PRESERVED"
    if profile.topology_basis == "PROXY_EXACT":
        return "PROXY_PRESERVED_SOURCE_UNVERIFIED"
    return "SOURCE_UNVERIFIED"


def run_curiomesh(obj: bpy.types.Object, config: RemeshConfig) -> RemeshResult:
    start = time.perf_counter()
    source_snapshot: bpy.types.Object | None = None
    source_surface: bpy.types.Object | None = None
    work_obj: bpy.types.Object | None = None
    engine = str(config.engine or "QUADRIFLOW").upper()
    if engine == "AUTO":
        engine = "ATLAS_Q"
    uv_status = "NONE"
    materials_preserved = False
    profile: ScanProfile | None = None
    selected_evaluation: CandidateEvaluation | None = None
    candidate_evaluations: list[CandidateEvaluation] = []
    strategy = ""
    geometry_rms = 0.0
    geometry_p95 = 0.0
    geometry_max = 0.0
    sampling_status = "NOT_MEASURED"
    fidelity_status = "NOT_MEASURED"
    fidelity_reference: _SurfaceReference | None = None
    bounded_source_evidence: _BoundedSourceEvidence | None = None

    if obj is None or obj.type != "MESH":
        return RemeshResult(False, message="Select a mesh object.")

    with _ContextSnapshot() as context_state:
        try:
            source_snapshot = _make_evaluated_snapshot(obj, config)
            source_surface = source_snapshot
            source_triangle_probe, source_exceeds_reference_cap = _bounded_triangle_count(
                source_snapshot,
                MAX_FIDELITY_REFERENCE_TRIANGLES,
            )
            if source_exceeds_reference_cap:
                bounded_source_evidence = _capture_bounded_source_evidence(
                    source_snapshot,
                    5000,
                    source_triangle_probe,
                )
                if engine == "ATLAS_Q":
                    profile = analyze_object(
                        source_snapshot,
                        config.target_faces,
                        feature_angle_deg=config.triad_feature_angle,
                        max_sample_faces=config.analysis_sample_faces,
                    )
                # Reuse the sole evaluated mesh as the native QEM work object.
                # Candidate copies are created only after this object is below
                # the reference cap, avoiding a second unreduced scan copy.
                work_obj = source_snapshot
                source_snapshot = None
                source_surface = obj
                work_obj.name = f"{obj.name}_CurioMeshWork"
                work_obj.data.name = f"{obj.name}_CurioMeshWorkMesh"
                work_obj.hide_select = False
                work_obj.hide_render = False
                work_obj.hide_set(False)
            else:
                work_obj = _make_working_copy(source_snapshot, obj)
            atlas_strategy = str(config.atlas_strategy or "AUTO").upper()
            pair_closure_only = engine == "ATLAS_Q" and atlas_strategy == "PAIR_CLOSURE"
            # Atlas-Q sends huge sources straight to PairClosure, so skip the flow-sized proxy for them.
            large_pair_first = (
                engine == "ATLAS_Q"
                and atlas_strategy == "AUTO"
                and len(obj.data.polygons) > LARGE_SOURCE_FACES
            )
            if engine == "ATLAS_Q" and not pair_closure_only and not large_pair_first:
                # Keep BMesh cleanup and Python adjacency off the full-resolution
                # scan whenever Blender's native QEM proxy can reduce it first.
                _prepare_flow_proxy(work_obj, config)
            elif engine == "TRIAD_Q_LITE" or large_pair_first:
                # PairClosure is an exact closure, not a simplifier. Reduce to
                # its expected output scale before Python pairing/adjacency so a
                # dense scan cannot expand by roughly three times.
                _prepare_pair_closure_proxy(work_obj, config)
            elif source_exceeds_reference_cap:
                # Direct QuadriFlow remains available for compatibility, but a
                # scan-scale input still needs a bounded fidelity reference.
                # Reusing its native QEM proxy avoids another full mesh copy.
                _prepare_flow_proxy(work_obj, config)
            _cleanup_mesh(work_obj.data, config)
            fidelity_reference = _make_surface_reference(
                source_surface,
                5000,
                reference_proxy=work_obj,
                bounded_source_evidence=bounded_source_evidence,
            )

            if engine == "ATLAS_Q":
                if profile is None:
                    profile = analyze_object(
                        source_surface,
                        config.target_faces,
                        feature_angle_deg=config.triad_feature_angle,
                        max_sample_faces=config.analysis_sample_faces,
                    )
                if not profile.exact_topology:
                    proxy_profile = analyze_object(
                        work_obj,
                        config.target_faces,
                        feature_angle_deg=config.triad_feature_angle,
                        max_exact_faces=450_000,
                        max_sample_faces=config.analysis_sample_faces,
                    )
                    profile = _merge_proxy_topology_profile(profile, proxy_profile)
                work_obj, selected_evaluation, candidate_evaluations = _run_atlasq(
                    work_obj,
                    source_surface,
                    config,
                    profile,
                    fidelity_reference,
                )
                ok = work_obj is not None and selected_evaluation is not None
                strategy = selected_evaluation.strategy if selected_evaluation is not None else "NONE"
            elif engine == "TRIAD_Q_LITE":
                _take_solver_error()
                triad_config = replace(
                    config,
                    triad_force_quads=bool(config.require_quads),
                )
                ok = _run_triadq_lite(work_obj, triad_config)
                strategy = "PAIR_CLOSURE" if triad_config.triad_force_quads else "TRIAD_Q_LITE"
            else:
                ok = _run_quadriflow(work_obj, config)
                strategy = "QUADRIFLOW"
                if ok and config.require_quads:
                    ok = mesh_diagnostics(work_obj, config.target_faces).quad_only

            if (
                not ok
                and config.voxel_repair
                and config.topology_policy == "REPAIR"
                and engine == "QUADRIFLOW"
            ):
                _log(config, "QuadriFlow failed; retrying after voxel repair")
                _remove_object(work_obj)
                if source_snapshot is not None:
                    work_obj = _make_working_copy(source_snapshot, obj)
                else:
                    work_obj = _make_evaluated_snapshot(obj, config)
                    work_obj.name = f"{obj.name}_CurioMeshWork"
                    work_obj.hide_select = False
                    work_obj.hide_render = False
                    work_obj.hide_set(False)
                    if bounded_source_evidence is not None:
                        # The retry snapshot is full resolution again. Keep its
                        # Python BMesh cleanup behind the same native QEM cap as
                        # the first scan-scale attempt.
                        _prepare_flow_proxy(work_obj, config)
                _cleanup_mesh(work_obj.data, config)
                repaired = _voxel_repair(work_obj, config)
                ok = repaired and _run_quadriflow(work_obj, config)
                engine = "VOXEL_REPAIR_QUADRIFLOW"
                strategy = "DIRTY_FLOW"
                if ok and config.require_quads:
                    ok = mesh_diagnostics(work_obj, config.target_faces).quad_only

            if not ok:
                elapsed = (time.perf_counter() - start) * 1000.0
                if candidate_evaluations:
                    # The candidate with the fewest rejection reasons is the closest miss.
                    closest = min(candidate_evaluations, key=lambda item: item.reason.count(";"))
                    why = f" Closest candidate ({closest.label}): {closest.reason[:400]}."
                else:
                    why = f" {_take_solver_error()}." if _solver_errors else ""
                return RemeshResult(
                    False,
                    engine=engine,
                    elapsed_ms=elapsed,
                    message=(
                        f"{engine} produced no candidate that passed the topology and quad-quality gates.{why} "
                        "Try Repair topology for a closed dirty scan, preserve topology for an open fragment, "
                        "or raise the target face count."
                    ),
                    strategy=strategy,
                    profile=profile.category if profile else "",
                    candidate_count=len(candidate_evaluations),
                    candidates=[evaluation.to_dict() for evaluation in candidate_evaluations],
                )

            projection_snapshot: bpy.types.Mesh | None = None
            projection_status = "NOT_REQUESTED"
            if config.shrinkwrap_project:
                projection_snapshot = work_obj.data.copy()
                projection_status = "APPLIED" if _apply_shrinkwrap(work_obj, source_surface, config) else "FAILED"

            if projection_snapshot is not None and projection_status == "FAILED":
                rejected_mesh = work_obj.data
                work_obj.data = projection_snapshot
                projection_snapshot = None
                if rejected_mesh.users == 0:
                    bpy.data.meshes.remove(rejected_mesh)
                projection_status = "ROLLED_BACK_APPLY_FAILED"
                _log(config, "Shrinkwrap projection failed to apply; restored validated candidate")

            def evaluate_final_mesh() -> tuple[MeshDiagnostics, _SurfaceError, list[str]]:
                diagnostics = mesh_diagnostics(work_obj, config.target_faces)
                measured = _sampled_surface_error_stats(
                    source_surface,
                    work_obj,
                    reference=fidelity_reference,
                )
                failures: list[str] = []
                if engine == "ATLAS_Q" and profile is not None:
                    failures = candidate_gate_failures(
                        diagnostics,
                        target_faces=config.target_faces,
                        geometry_rms=measured.rms,
                        geometry_p95=measured.p95,
                        geometry_max=measured.maximum,
                        target_edge_normalized=max(profile.target_edge_length / profile.diagonal, 1e-9),
                        require_quads=config.require_quads,
                        topology_policy=config.topology_policy,
                        profile=profile,
                        quality=config.quality,
                    )
                    if strategy == "PAIR_CLOSURE" and selected_evaluation is not None and selected_evaluation.warnings:
                        failures = split_soft_failures(failures)[0]
                elif config.require_quads and not diagnostics.quad_only:
                    failures = ["local quad/manifold gate failed"]
                return diagnostics, measured, failures

            post_projection, geometry, post_failures = evaluate_final_mesh()
            if post_failures and projection_snapshot is not None and projection_status == "APPLIED":
                rejected_mesh = work_obj.data
                work_obj.data = projection_snapshot
                projection_snapshot = None
                if rejected_mesh.users == 0:
                    bpy.data.meshes.remove(rejected_mesh)
                post_projection, geometry, post_failures = evaluate_final_mesh()
                if not post_failures:
                    projection_status = "ROLLED_BACK_FAILED_GATES"
                    _log(config, "Shrinkwrap projection failed release gates; restored validated candidate")

            if projection_snapshot is not None:
                if projection_snapshot.users == 0:
                    bpy.data.meshes.remove(projection_snapshot)
                projection_snapshot = None

            geometry_rms = geometry.rms
            geometry_p95 = geometry.p95
            geometry_max = geometry.maximum
            sampling_status = geometry.component_sampling
            fidelity_status = geometry.fidelity_status
            if post_failures:
                elapsed = (time.perf_counter() - start) * 1000.0
                return RemeshResult(
                    False,
                    engine=engine,
                    elapsed_ms=elapsed,
                    message=(
                        "Final projection violated the release gates: "
                        + "; ".join(post_failures[:5])
                    ),
                    strategy=strategy,
                    profile=profile.category if profile else "",
                    candidate_count=len(candidate_evaluations),
                    candidates=[evaluation.to_dict() for evaluation in candidate_evaluations],
                )

            work_obj["curiomesh_projection_status"] = projection_status

            material_count = _copy_material_slots(source_surface, work_obj)
            materials_preserved = material_count > 0
            if materials_preserved:
                _map_material_indices(
                    source_surface,
                    work_obj,
                    allow_python_bvh=bounded_source_evidence is None,
                )

            uv_status = _preserve_textures(
                source_surface,
                work_obj,
                config,
                allow_python_bvh=bounded_source_evidence is None,
            )

            output_obj, work_obj = _finalize_output(obj, work_obj, config)
            diagnostics = mesh_diagnostics(output_obj, config.target_faces)
            elapsed = (time.perf_counter() - start) * 1000.0
            topology_status = _topology_status(profile, config.topology_policy)
            topology_note = (
                " Full-resolution source topology was not exhaustively verified."
                if "UNVERIFIED" in topology_status
                else ""
            )
            sampling_note = (
                " Exhaustive connected-component sampling was skipped at the scan-scale memory guard."
                if sampling_status.startswith(("SKIPPED_LIMIT", "BOUNDED_"))
                else ""
            )
            fidelity_note = (
                " Fidelity error used bounded source samples and a target-scale proxy BVH."
                if fidelity_status.startswith("BOUNDED_PROXY_BVH_")
                else ""
            )
            projection_note = (
                {
                    "ROLLED_BACK_FAILED_GATES": (
                        " Shrinkwrap projection was rolled back because it violated quad-quality gates."
                    ),
                    "ROLLED_BACK_APPLY_FAILED": (
                        " Shrinkwrap projection could not be applied; the validated candidate was retained."
                    ),
                }.get(projection_status, "")
            )
            warning_note = (
                f" Warning: {selected_evaluation.warnings}."
                if selected_evaluation is not None and selected_evaluation.warnings
                else ""
            )
            result = RemeshResult(
                True,
                output_object=output_obj,
                diagnostics=diagnostics,
                engine=engine,
                elapsed_ms=elapsed,
                message=(
                    f"{strategy} completed: {diagnostics.faces:,} locally validated manifold quads; "
                    f"p95 scan error {geometry_p95:.4%} of the source diagonal."
                    f"{warning_note}{topology_note}{sampling_note}{fidelity_note}{projection_note}"
                    if diagnostics.quad_only
                    else f"{strategy} remesh completed."
                ),
                materials_preserved=materials_preserved,
                uv_status=uv_status,
                strategy=strategy,
                profile=profile.category if profile else "",
                candidate_count=max(1, len(candidate_evaluations)),
                geometry_rms=geometry_rms,
                geometry_p95=geometry_p95,
                geometry_max=geometry_max,
                sampling_status=sampling_status,
                fidelity_status=fidelity_status,
                quad_only_verified=diagnostics.quad_only,
                topology_status=topology_status,
                candidates=[evaluation.to_dict() for evaluation in candidate_evaluations],
            )
            _annotate_output(output_obj, result)
            context_state.final_active = output_obj
            _log(
                config,
                (
                    f"Done: faces={diagnostics.faces} quad_ratio={diagnostics.quad_ratio:.3f} "
                    f"face_error={diagnostics.face_count_error:.3f} uv={uv_status}"
                ),
            )
            return result
        except Exception as exc:
            traceback.print_exc()
            elapsed = (time.perf_counter() - start) * 1000.0
            return RemeshResult(False, engine=engine, elapsed_ms=elapsed, message=str(exc))
        finally:
            if not config.keep_debug_objects:
                _remove_object(source_snapshot)
            elif source_snapshot is not None:
                source_snapshot.name = f"{obj.name}_CurioMesh_SourceDebug"
            if work_obj is not None:
                _remove_object(work_obj)
