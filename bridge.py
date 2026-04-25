from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
import traceback
from typing import Any

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .metrics import MeshDiagnostics, mesh_diagnostics, uv_mapping_is_valid
from .textures import bake_selected_to_active, ensure_target_uvs
from .triadq import MeshData, RemeshOptions as TriadRemeshOptions, remesh_mesh as triad_remesh_mesh


@dataclass(slots=True)
class RemeshConfig:
    target_faces: int = 8000
    engine: str = "QUADRIFLOW"
    quality: str = "BALANCED"
    seed: int = 0
    triad_seed_count: int = 6
    triad_feature_angle: float = 35.0
    triad_force_quads: bool = False
    triad_flow_mode: str = "AUTO"
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
    if not getattr(obj.data, "vertices", None):
        return 1.0
    if not obj.data.vertices:
        return 1.0
    points = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
    minv = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maxv = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return max((maxv - minv).length, 1e-6)


def _copy_material_slots(source: bpy.types.Object, target: bpy.types.Object) -> int:
    target.data.materials.clear()
    materials: list[bpy.types.Material] = []

    for slot in getattr(source, "material_slots", []):
        if slot.material and slot.material not in materials:
            materials.append(slot.material)
    for material in getattr(source.data, "materials", []):
        if material and material not in materials:
            materials.append(material)

    for material in materials:
        target.data.materials.append(material)
    if materials:
        target.active_material = materials[0]
    return len(materials)


def _make_evaluated_snapshot(source: bpy.types.Object, config: RemeshConfig) -> bpy.types.Object:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = source.evaluated_get(depsgraph)
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
        bpy.ops.object.voxel_remesh(voxel_size=voxel_size, adaptivity=0.0)
        _log(config, f"Voxel repair succeeded at voxel_size={voxel_size:.5f}")
        return True
    except TypeError:
        try:
            bpy.ops.object.voxel_remesh(voxel_size=voxel_size)
            _log(config, f"Voxel repair succeeded at voxel_size={voxel_size:.5f}")
            return True
        except Exception:
            traceback.print_exc()
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
        result = bpy.ops.object.quadriflow_remesh(**kwargs)
    except TypeError:
        minimal = {
            "mode": "FACES",
            "target_faces": max(1, int(config.target_faces)),
            "seed": max(0, int(config.seed)),
        }
        result = bpy.ops.object.quadriflow_remesh(**minimal)
    except Exception:
        traceback.print_exc()
        return False
    return "FINISHED" in result and len(obj.data.polygons) > 0


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
    )
    result, report = triad_remesh_mesh(source, options)
    if not report.success or not result.faces:
        _log(config, f"TRIAD-Q Lite failed: {report.message}")
        return False
    _apply_meshdata_to_object(obj, result)
    obj["curiomesh_triadq_mode"] = report.mode
    obj["curiomesh_triadq_seed"] = int(report.selected_seed)
    obj["curiomesh_triadq_score"] = float(report.score)
    obj["curiomesh_triadq_feature_edges"] = int(report.feature_edges)
    _log(
        config,
        (
            f"TRIAD-Q Lite: mode={report.mode} seed={report.selected_seed} "
            f"faces={report.output_faces} quad_ratio={report.quad_ratio:.3f}"
        ),
    )
    return True


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


def _map_material_indices(source: bpy.types.Object, target: bpy.types.Object) -> bool:
    if not source.data.materials or not target.data.materials or not source.data.polygons:
        return False

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


def _preserve_textures(source: bpy.types.Object, target: bpy.types.Object, config: RemeshConfig) -> str:
    mode = str(config.texture_mode or "PROJECT").upper()
    if mode == "NONE":
        return "SKIPPED"
    if not source.data.uv_layers and mode != "BAKE":
        return "NO_SOURCE_UV"

    transferred = False
    projected = False
    baked = False

    if mode in {"TRANSFER", "PROJECT"}:
        transferred = _transfer_uvs_with_modifier(source, target)
    if mode == "PROJECT" and not uv_mapping_is_valid(target):
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
        source_obj.data = result_mesh
        _remove_object(work_obj)
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


def run_curiomesh(obj: bpy.types.Object, config: RemeshConfig) -> RemeshResult:
    start = time.perf_counter()
    source_snapshot: bpy.types.Object | None = None
    work_obj: bpy.types.Object | None = None
    engine = str(config.engine or "QUADRIFLOW").upper()
    if engine == "AUTO":
        engine = "QUADRIFLOW"
    uv_status = "NONE"
    materials_preserved = False

    if obj is None or obj.type != "MESH":
        return RemeshResult(False, message="Select a mesh object.")

    with _ContextSnapshot() as context_state:
        try:
            source_snapshot = _make_evaluated_snapshot(obj, config)
            work_obj = _make_working_copy(source_snapshot, obj)
            _cleanup_mesh(work_obj.data, config)

            if engine == "TRIAD_Q_LITE":
                ok = _run_triadq_lite(work_obj, config)
            else:
                ok = _run_quadriflow(work_obj, config)

            if not ok and config.voxel_repair and engine == "QUADRIFLOW":
                _log(config, "QuadriFlow failed; retrying after voxel repair")
                _remove_object(work_obj)
                work_obj = _make_working_copy(source_snapshot, obj)
                _cleanup_mesh(work_obj.data, config)
                _voxel_repair(work_obj, config)
                ok = _run_quadriflow(work_obj, config)
                engine = "VOXEL_REPAIR_QUADRIFLOW"

            if not ok:
                elapsed = (time.perf_counter() - start) * 1000.0
                return RemeshResult(
                    False,
                    engine=engine,
                    elapsed_ms=elapsed,
                    message="QuadriFlow failed on this mesh.",
                )

            if config.shrinkwrap_project:
                _apply_shrinkwrap(work_obj, source_snapshot, config)

            material_count = _copy_material_slots(source_snapshot, work_obj)
            materials_preserved = material_count > 0
            if materials_preserved:
                _map_material_indices(source_snapshot, work_obj)

            uv_status = _preserve_textures(source_snapshot, work_obj, config)

            output_obj, work_obj = _finalize_output(obj, work_obj, config)
            diagnostics = mesh_diagnostics(output_obj, config.target_faces)
            elapsed = (time.perf_counter() - start) * 1000.0
            result = RemeshResult(
                True,
                output_object=output_obj,
                diagnostics=diagnostics,
                engine=engine,
                elapsed_ms=elapsed,
                message="Remesh completed.",
                materials_preserved=materials_preserved,
                uv_status=uv_status,
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
