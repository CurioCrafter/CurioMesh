# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Region retopology: remesh part of a mesh on a fixed seam, preview it, apply it.

The math lives in triadq/patch.py (pure NumPy). This module turns Blender
state (edit selection, face sets, vertex groups, box / lattice cages and
mesh volumes) into a face mask, shows the patched result as a preview object
with a tinted region and highlighted seam, and writes it back with bmesh.
"""

from __future__ import annotations

import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import overlay
from .triadq import patch

PREVIEW_NAME = overlay.REGION_PREVIEW_NAME
LIVE_REGION_FACE_CAP = 6_000
DEBOUNCE_SECONDS = 0.4
POLL_SECONDS = 0.25
MAX_PATCH_FACES = 40_000

REGION_TINT = overlay.REGION_TINT
SEAM_COLOR = overlay.SEAM_COLOR

SOURCE_ITEMS = [
    ("SELECTION", "Selection", "The faces you selected in Edit Mode", "RESTRICT_SELECT_OFF", 0),
    ("FACE_SET", "Face Set", "Faces that share one Face Set number (sculpt Face Sets, or any whole-number face attribute)", "FACE_MAPS", 1),
    ("VERTEX_GROUP", "Vertex Group", "Faces whose average vertex-group weight is at least 0.5", "GROUP_VERTEX", 2),
    ("VOLUMES", "Cages", "Faces inside the cages and volumes in the list (a box or lattice you move over the part)", "CUBE", 3),
]
OPERATION_ITEMS = [
    ("UNION", "Add", "Add the faces inside this cage to the region", "SELECT_EXTEND", 0),
    ("SUBTRACT", "Remove", "Remove the faces inside this cage from the region", "SELECT_SUBTRACT", 1),
    ("INTERSECT", "Overlap", "Keep only region faces that are also inside this cage", "SELECT_INTERSECT", 2),
]
STRATEGY_ITEMS = [
    ("FLOW", "Flow", "Quad loops that follow the surface and the seam (best for organic shapes)", "MOD_WAVE", 0),
    ("GRID", "Grid", "A regular grid. Cleanest when the region has four clear sides; otherwise it is laid out and closed to quads", "MESH_GRID", 1),
]
APPLY_ITEMS = [
    ("SOURCE", "In the Mesh", "Replace the region inside your mesh (Undo restores it)"),
    ("COPY", "In a Copy", "Leave your mesh untouched and write the result into a copy"),
]

state = {
    "source": "",
    "sig": None,
    "changed_at": 0.0,
    "message": "",
    "cache": None,
    "draw": None,
    "batches": None,
    "last_removed": [],
}


def status() -> str:
    return state["message"]


# --------------------------------------------------------------------------- mesh data


def _mesh_arrays(obj) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray, np.ndarray]:
    """(local vertex positions, polygons, loop starts, loop sizes)."""

    mesh = obj.data
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    starts = np.empty(len(mesh.polygons), dtype=np.int64)
    sizes = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", sizes)
    loops = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loops)
    faces = [tuple(loops[s : s + n].tolist()) for s, n in zip(starts.tolist(), sizes.tolist())]
    return co.reshape(-1, 3), faces, starts, sizes


class _CornerUVs:
    """faces[i] corner UVs on demand, as patch.remesh_region expects."""

    def __init__(self, mesh, starts, sizes):
        layer = mesh.uv_layers.active
        uv = np.empty(len(mesh.loops) * 2, dtype=np.float64)
        layer.data.foreach_get("uv", uv)
        self.uv, self.starts, self.sizes = uv.reshape(-1, 2), starts, sizes

    def __getitem__(self, face_index):
        start = self.starts[face_index]
        return self.uv[start : start + self.sizes[face_index]]


def _world_centroids(obj, co, starts, sizes) -> np.ndarray:
    mesh = obj.data
    loops = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loops)
    owner = np.repeat(np.arange(len(starts)), sizes)
    sums = np.stack([np.bincount(owner, co[loops, k], len(starts)) for k in range(3)], axis=1)
    local = sums / np.maximum(sizes, 1)[:, None]
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def _world_scale(obj) -> float:
    return max(float(np.mean(obj.matrix_world.to_scale())), 1e-12)


# --------------------------------------------------------------------------- region masks


def volume_triangles(vol, depsgraph) -> np.ndarray:
    """World-space closed triangle hull of a lattice or mesh volume."""

    matrix = np.asarray(vol.matrix_world, dtype=np.float64)
    if vol.type == "LATTICE":
        lat = vol.data
        points = np.asarray([p.co_deform[:] for p in lat.points], dtype=np.float64)
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
        grid = points.reshape(lat.points_w, lat.points_v, lat.points_u, 3)
        return patch.lattice_hull(grid)
    evaluated = vol.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tris)
    finally:
        evaluated.to_mesh_clear()
    co = co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
    return co[tris.reshape(-1, 3)]


def volume_mask(vol, points: np.ndarray, depsgraph) -> np.ndarray:
    if vol.type == "EMPTY":
        inverse = np.asarray(vol.matrix_world.inverted_safe(), dtype=np.float64)
        return patch.inside_box(points, inverse, float(vol.empty_display_size))
    if vol.type in {"LATTICE", "MESH"}:
        return patch.inside_volume(points, volume_triangles(vol, depsgraph))
    return np.zeros(len(points), dtype=bool)


def build_mask(context, obj, settings) -> tuple[np.ndarray, str]:
    """Boolean face mask on `obj` for the chosen region source."""

    mesh = obj.data
    count = len(mesh.polygons)
    source = settings.source
    if source == "SELECTION":
        if obj.mode == "EDIT":
            obj.update_from_editmode()
        mask = np.zeros(count, dtype=bool)
        mesh.polygons.foreach_get("select", mask)
        return mask, "" if mask.any() else "Select faces in Edit Mode first"
    if source == "FACE_SET":
        attribute = mesh.attributes.get(settings.face_set_attribute)
        if attribute is None or attribute.domain != "FACE" or attribute.data_type not in {"INT", "INT8"}:
            return np.zeros(count, dtype=bool), f"No integer face attribute '{settings.face_set_attribute}'"
        values = np.empty(count, dtype=np.int64)
        attribute.data.foreach_get("value", values)
        mask = values == int(settings.face_set_id)
        return mask, "" if mask.any() else f"No faces in face set {settings.face_set_id}"
    if source == "VERTEX_GROUP":
        group = obj.vertex_groups.get(settings.vertex_group)
        if group is None:
            return np.zeros(count, dtype=bool), "Pick a vertex group"
        weights = np.zeros(len(mesh.vertices))
        index = group.index
        for vertex in mesh.vertices:
            for element in vertex.groups:
                if element.group == index:
                    weights[vertex.index] = element.weight
        _co, faces, _starts, sizes = _mesh_arrays(obj)
        owner = np.repeat(np.arange(count), sizes)
        flat = np.fromiter((v for face in faces for v in face), dtype=np.int64, count=int(sizes.sum()))
        mask = np.bincount(owner, weights[flat], count) / np.maximum(sizes, 1) >= 0.5
        return mask, "" if mask.any() else "No faces with weight >= 0.5"
    entries = [(item.operation, item.obj) for item in settings.volumes if item.obj is not None and item.obj != obj]
    if not entries:
        return np.zeros(count, dtype=bool), "Add a cage or volume first"
    co, _faces, starts, sizes = _mesh_arrays(obj)
    points = _world_centroids(obj, co, starts, sizes)
    depsgraph = context.evaluated_depsgraph_get()
    mask = patch.combine_masks(count, [(op, volume_mask(vol, points, depsgraph)) for op, vol in entries])
    return mask, "" if mask.any() else "No faces inside the cages"


# --------------------------------------------------------------------------- compute


def _projector(bvh: BVHTree):
    def project(points: np.ndarray) -> np.ndarray:
        out = points.copy()
        for k, point in enumerate(points):
            hit = bvh.find_nearest(Vector(point))
            if hit[0] is not None:
                out[k] = hit[0][:]
        return out

    return project


def _fit_result(co: np.ndarray, result, project, iterations: int) -> None:
    """Relax a patch's new vertices on the original surface (seam pinned)."""

    if iterations <= 0 or not len(result.points):
        return
    seam = sorted({v for face in result.faces for v in face if v < result.base})
    local = {v: k for k, v in enumerate(seam)}
    offset = len(seam)
    faces = [tuple(local[v] if v < result.base else offset + v - result.base for v in face) for face in result.faces]
    points = np.vstack((co[seam], result.points))
    pinned = np.arange(len(points)) < offset
    result.points = patch.fit_to_surface(points, faces, pinned, project, iterations, 0.5)[offset:]


def compute(context, obj, settings) -> dict:
    """Mask -> disk components -> patches. Returns a cache dict (results, arrays, message)."""

    started = time.perf_counter()
    mask, message = build_mask(context, obj, settings)
    co, faces, starts, sizes = _mesh_arrays(obj)
    out = {"results": [], "co": co, "faces": faces, "mask": mask, "message": message, "region_faces": int(mask.sum())}
    if not mask.any():
        return out
    scale = _world_scale(obj)
    components, warnings = patch.prepare_regions(
        co, faces, mask, min_island=settings.min_island, max_hole=settings.min_island, smooth=2 if settings.smooth_seam else 0
    )
    corner_uvs = _CornerUVs(obj.data, starts, sizes) if obj.data.uv_layers.active is not None else None
    bvh = BVHTree.FromPolygons(co.tolist(), faces) if settings.fit_iterations > 0 else None
    for component in components:
        result = patch.remesh_region(
            co,
            faces,
            component,
            target_length=settings.target_edge / scale,
            strategy=settings.strategy,
            falloff=settings.falloff / scale,
            corner_uvs=corner_uvs,
            max_faces=MAX_PATCH_FACES,
            seam_layer=settings.seam_layer,
            field_mode=settings.field_mode,
        )
        if result.ok and bvh is not None:
            _fit_result(co, result, _projector(bvh), settings.fit_iterations)
        if not result.ok:
            warnings.append(f"Region ({len(component)} faces) skipped: {result.message}")
        out["results"].append(result)
    good = [r for r in out["results"] if r.ok]
    quads = sum(r.quads for r in good)
    new_faces = sum(len(r.faces) for r in good)
    seconds = time.perf_counter() - started
    out["message"] = (
        f"{len(good)} region(s): {sum(len(r.removed_faces) for r in good):,} faces -> {new_faces:,} "
        f"({quads / max(new_faces, 1) * 100:.0f}% quads), {seconds:.1f} s"
    )
    if warnings:
        out["message"] += " | " + warnings[0]
    out["warnings"] = warnings
    return out


# --------------------------------------------------------------------------- write back


def write_patches(obj, cache) -> int:
    """Write the cached patches into obj's mesh (object mode). Returns the number of new faces."""

    results = [r for r in cache["results"] if r.ok]
    if not results:
        return 0
    mesh = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        verts = list(bm.verts)
        removed = sorted({int(f) for r in results for f in r.removed_faces})
        state["last_removed"] = removed
        material = [bm.faces[i].material_index for i in range(len(bm.faces))]
        smooth = [bm.faces[i].smooth for i in range(len(bm.faces))]
        old_edges = {e for i in removed for e in bm.faces[i].edges}
        old_verts = {v for i in removed for v in bm.faces[i].verts}
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in removed], context="FACES_ONLY")
        for edge in old_edges:
            if edge.is_valid and not edge.link_faces:
                bm.edges.remove(edge)
        needed = {verts[v] for r in results for face in r.faces for v in face if v < r.base}
        for vert in old_verts:
            if vert.is_valid and not vert.link_edges and vert not in needed:
                bm.verts.remove(vert)
        uv_layer = bm.loops.layers.uv.active
        created = 0
        for result in results:
            fresh = [bm.verts.new(p) for p in result.points.tolist()]
            for index, face in enumerate(result.faces):
                corners = [verts[v] if v < result.base else fresh[v - result.base] for v in face]
                try:
                    new_face = bm.faces.new(corners)
                except ValueError:
                    continue
                source = result.face_source[index] if index < len(result.face_source) else -1
                if source >= 0:
                    new_face.material_index = material[source]
                    new_face.smooth = smooth[source]
                if uv_layer is not None and result.face_uvs is not None:
                    for loop, uv in zip(new_face.loops, result.face_uvs[index]):
                        loop[uv_layer].uv = uv
                new_face.select = True
                created += 1
        bm.normal_update()
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()
    return created


# --------------------------------------------------------------------------- preview


def _preview_object():
    return bpy.data.objects.get(PREVIEW_NAME)


def remove_preview(restore_source: bool = True) -> None:
    obj = _preview_object()
    if obj is not None:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    source = bpy.data.objects.get(state["source"])
    if restore_source and source is not None:
        try:
            source.hide_set(False)
        except RuntimeError:
            pass
    state.update(draw=None, batches=None)
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    overlay.tag_redraw()


def _merged_arrays(cache):
    vertices, faces, new = patch.apply_patches(cache["co"], cache["faces"], cache["results"])
    used = np.zeros(len(vertices), dtype=bool)
    for face in faces:
        used[list(face)] = True
    remap = np.cumsum(used) - 1
    return vertices[used], [tuple(remap[list(face)].tolist()) for face in faces], new


def _build_preview(context, source, cache) -> None:
    vertices, faces, new = _merged_arrays(cache)
    obj = _preview_object()
    mesh = bpy.data.meshes.new(PREVIEW_NAME + "_Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()
    for material in source.data.materials:
        mesh.materials.append(material)
    if obj is None:
        obj = bpy.data.objects.new(PREVIEW_NAME, mesh)
        collection = source.users_collection[0] if source.users_collection else context.scene.collection
        collection.objects.link(obj)
    else:
        old = obj.data
        obj.data = mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)
    obj.matrix_world = source.matrix_world.copy()
    obj.show_wire = True
    obj.show_all_edges = True
    obj.hide_select = True
    matrix = np.asarray(source.matrix_world, dtype=np.float64)
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    tris = [(face[0], face[k], face[k + 1]) for face, is_new in zip(faces, new) if is_new for k in range(1, len(face) - 1)]
    co_world = cache["co"] @ matrix[:3, :3].T + matrix[:3, 3]
    lines = []
    for result in cache["results"]:
        if result.ok:
            seam = result.seam
            lines.extend((co_world[seam[k]], co_world[seam[(k + 1) % len(seam)]]) for k in range(len(seam)))
    tri_pos = world[np.asarray(tris, dtype=np.int64).reshape(-1)] if tris else np.zeros((0, 3))
    state["draw"] = (
        np.ascontiguousarray(tri_pos, dtype=np.float32),
        np.ascontiguousarray(np.asarray(lines, dtype=np.float32).reshape(-1, 3)),
    )
    state["batches"] = None


def _signature(context, obj, settings) -> tuple:
    parts: list = [
        obj.name,
        len(obj.data.vertices),
        len(obj.data.polygons),
        settings.source,
        settings.face_set_attribute,
        settings.face_set_id,
        settings.vertex_group,
        round(settings.target_edge, 6),
        settings.strategy,
        round(settings.falloff, 6),
        settings.smooth_seam,
        settings.min_island,
        settings.fit_iterations,
        tuple(round(x, 5) for row in obj.matrix_world for x in row),
    ]
    if settings.source == "VOLUMES":
        for item in settings.volumes:
            vol = item.obj
            parts.append((item.operation, vol.name if vol else ""))
            if vol is None:
                continue
            parts.append(tuple(round(x, 5) for row in vol.matrix_world for x in row))
            if vol.type == "EMPTY":
                parts.append(round(vol.empty_display_size, 5))
            elif vol.type == "LATTICE":
                parts.append(round(sum(sum(p.co_deform) for p in vol.data.points), 5))
            elif vol.type == "MESH":
                parts.append((len(vol.data.vertices), vol.data.as_pointer()))
    elif settings.source == "SELECTION":
        select = np.zeros(len(obj.data.polygons), dtype=bool)
        obj.data.polygons.foreach_get("select", select)
        parts.append(hash(select.tobytes()))
    return tuple(parts)


def run_preview(context, obj=None) -> bool:
    """Compute and show the preview synchronously. Returns True when a patch was produced."""

    settings = context.scene.curiomesh_region
    obj = obj or bpy.data.objects.get(state["source"])
    if obj is None or obj.type != "MESH" or obj.name == PREVIEW_NAME:
        state["message"] = "Select a mesh"
        return False
    if obj.mode == "EDIT":
        obj.update_from_editmode()
        bpy.ops.object.mode_set(mode="OBJECT")
    state["source"] = obj.name
    cache = compute(context, obj, settings)
    cache["sig"] = _signature(context, obj, settings)
    state["cache"] = cache
    state["sig"] = cache["sig"]
    state["message"] = cache["message"]
    if not any(r.ok for r in cache["results"]):
        remove_preview()
        state["cache"] = cache
        return False
    _build_preview(context, obj, cache)
    try:
        obj.hide_set(True)
    except RuntimeError:
        pass
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=POLL_SECONDS)
    overlay.tag_redraw()
    return True


def _poll():
    """Recompute the preview after cages or settings stop changing for DEBOUNCE_SECONDS."""

    try:
        context = bpy.context
        settings = getattr(context.scene, "curiomesh_region", None)
        source = bpy.data.objects.get(state["source"])
        if settings is None or source is None or _preview_object() is None:
            return None
        if not settings.live:
            return POLL_SECONDS
        sig = _signature(context, source, settings)
        now = time.perf_counter()
        if sig != state["sig"]:
            state["sig"] = sig
            state["changed_at"] = now
            return POLL_SECONDS
        cache = state["cache"]
        if cache is not None and cache.get("sig") != sig and now - state["changed_at"] >= DEBOUNCE_SECONDS:
            mask, _message = build_mask(context, source, settings)
            if int(mask.sum()) > LIVE_REGION_FACE_CAP:
                cache["sig"] = sig
                state["message"] = f"Live paused: region has {int(mask.sum()):,} faces (limit {LIVE_REGION_FACE_CAP:,}); press Preview"
            else:
                run_preview(context, source)
            overlay.tag_redraw()
    except Exception as exc:
        state["message"] = f"Region preview error: {exc}"[:120]
    return POLL_SECONDS


def _draw() -> None:
    try:
        data = state["draw"]
        if data is None or _preview_object() is None:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader

        if state["batches"] is None:
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            tri_pos, line_pos = data
            tris = batch_for_shader(shader, "TRIS", {"pos": tri_pos}) if len(tri_pos) else None
            lines = batch_for_shader(shader, "LINES", {"pos": line_pos}) if len(line_pos) else None
            state["batches"] = (shader, tris, lines)
        shader, tris, lines = state["batches"]
        prefs = overlay.get_prefs()
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)
        shader.bind()
        if tris is not None:
            shader.uniform_float("color", overlay.pref_color(prefs, "region_tint"))
            tris.draw(shader)
        if lines is not None:
            gpu.state.line_width_set(float(prefs.region_seam_width))
            gpu.state.depth_test_set("ALWAYS")
            shader.uniform_float("color", overlay.pref_color(prefs, "region_seam_color"))
            lines.draw(shader)
            gpu.state.line_width_set(1.0)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


# --------------------------------------------------------------------------- properties


def _volume_poll(self, obj) -> bool:
    return obj.type in {"EMPTY", "LATTICE", "MESH"} and obj.name != PREVIEW_NAME


class CURIOMESH_PG_region_volume(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(name="Volume", type=bpy.types.Object, poll=_volume_poll)
    operation: bpy.props.EnumProperty(name="Operation", items=OPERATION_ITEMS, default="UNION")


FIELD_ITEMS = (
    ("SMOOTH", "Smooth", "Quads follow a smoothed flow pinned to sharp edges and borders (fast, default)"),
    ("CURVATURE", "Follow Curvature", "Quads also line up with the main bending directions of the surface; about 1.5x slower"),
)
SEAM_LAYER_ITEMS = (
    ("AUTO", "Auto", "Add a ring of quads along jagged borders so quads next to the seam stay well shaped"),
    ("ON", "Always", "Always add a ring of quads along the border"),
    ("OFF", "Off", "Fill straight up to the border (the 0.6 preview behaviour before this option)"),
)


class CURIOMESH_PG_region(bpy.types.PropertyGroup):
    source: bpy.props.EnumProperty(name="Region From", description="How the part to redo is chosen", items=SOURCE_ITEMS, default="SELECTION")
    face_set_attribute: bpy.props.StringProperty(name="Attribute", description="Name of the whole-number face attribute (sculpt Face Sets use .sculpt_face_set)", default=".sculpt_face_set")
    face_set_id: bpy.props.IntProperty(name="Face Set", description="Which Face Set number to redo", default=1)
    vertex_group: bpy.props.StringProperty(name="Vertex Group", description="Faces weighted 0.5 or more in this group form the region", default="")
    volumes: bpy.props.CollectionProperty(type=CURIOMESH_PG_region_volume)
    volume_index: bpy.props.IntProperty(default=0)
    target_edge: bpy.props.FloatProperty(
        name="Quad Size",
        description="Edge length of the new quads inside the region. 0 matches the quads around the region",
        default=0.0,
        min=0.0,
        soft_max=1.0,
        subtype="DISTANCE",
    )
    strategy: bpy.props.EnumProperty(name="Fill Style", description="Pattern of the new quads", items=STRATEGY_ITEMS, default="FLOW")
    falloff: bpy.props.FloatProperty(
        name="Seam Blend",
        description="Distance over which quad size changes from the surrounding quads to the target size. 0 uses twice the quad size",
        default=0.0,
        min=0.0,
        soft_max=1.0,
        subtype="DISTANCE",
    )
    smooth_seam: bpy.props.BoolProperty(
        name="Straighten Seam", description="Even out jagged region borders by whole quads, with no notches or spikes", default=True
    )
    min_island: bpy.props.IntProperty(
        name="Ignore Specks", description="Ignore region specks and fill holes up to this many faces", default=4, min=1, max=1000
    )
    fit_iterations: bpy.props.IntProperty(
        name="Surface Fit Passes", description="Relax-and-project passes that settle the new quads onto the original surface", default=3, min=0, max=50
    )
    seam_layer: bpy.props.EnumProperty(
        name="Border Ring", description="Ring of quads along the region border that keeps seam quads well shaped", items=SEAM_LAYER_ITEMS, default="AUTO"
    )
    field_mode: bpy.props.EnumProperty(name="Flow", description="Direction the new quads follow", items=FIELD_ITEMS, default="SMOOTH")
    apply_mode: bpy.props.EnumProperty(name="Result", description="Where the new quads are written", items=APPLY_ITEMS, default="SOURCE")
    live: bpy.props.BoolProperty(
        name="Auto-Update", description="Rebuild the preview automatically when you move a cage or change a setting", default=True
    )
    fit_target: bpy.props.PointerProperty(
        name="Fit Target",
        type=bpy.types.Object,
        description="Surface that Fit to Surface snaps to. Empty: another selected mesh, otherwise the mesh itself",
        poll=lambda self, obj: obj.type == "MESH",
    )


# --------------------------------------------------------------------------- operators


def _active_mesh(context):
    obj = context.object
    if obj is not None and obj.type == "MESH" and obj.name != PREVIEW_NAME:
        return obj
    return None


class CURIOMESH_OT_region_preview(bpy.types.Operator):
    bl_idname = "curiomesh.region_preview"
    bl_label = "Preview Region"
    bl_description = "Build the new quads for the region and show them as a preview. The seam (orange) never moves; nothing is changed until you Apply"

    def execute(self, context):
        obj = _active_mesh(context) or bpy.data.objects.get(state["source"])
        if obj is None:
            self.report({"ERROR"}, "Select a mesh")
            return {"CANCELLED"}
        ok = run_preview(context, obj)
        self.report({"INFO"} if ok else {"WARNING"}, state["message"][:200])
        return {"FINISHED"} if ok else {"CANCELLED"}


class CURIOMESH_OT_region_apply(bpy.types.Operator):
    bl_idname = "curiomesh.region_apply"
    bl_label = "Apply Region"
    bl_description = "Write the previewed quads into your mesh (or a copy, see Result). Undo restores it"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.curiomesh_region
        source = bpy.data.objects.get(state["source"]) if _preview_object() else None
        source = source or _active_mesh(context)
        if source is None:
            self.report({"ERROR"}, "Select a mesh")
            return {"CANCELLED"}
        was_edit = source.mode == "EDIT"
        if was_edit:
            source.update_from_editmode()
        cache = state["cache"]
        if cache is None or cache.get("sig") != _signature(context, source, settings) or state["source"] != source.name:
            cache = compute(context, source, settings)
        remove_preview()
        if was_edit:
            bpy.ops.object.mode_set(mode="OBJECT")
        target = source
        if settings.apply_mode == "COPY":
            target = source.copy()
            target.data = source.data.copy()
            target.name = f"{source.name}_Region"
            for collection in source.users_collection:
                collection.objects.link(target)
            target["curiomesh_region_source"] = source.name
            source.hide_set(True)
            context.view_layer.objects.active = target
        created = write_patches(target, cache)
        state["cache"] = None
        state["message"] = cache["message"]
        if was_edit and target is source:
            bpy.ops.object.mode_set(mode="EDIT")
        if not created:
            self.report({"WARNING"}, cache["message"] or "Nothing to apply")
            return {"CANCELLED"}
        target["curiomesh_region_faces"] = int(created)
        overlay.invalidate()
        self.report({"INFO"}, f"Region applied: {created:,} faces. {cache['message']}"[:200])
        return {"FINISHED"}


class CURIOMESH_OT_region_cancel(bpy.types.Operator):
    bl_idname = "curiomesh.region_cancel"
    bl_label = "Cancel Region Preview"
    bl_description = "Discard the preview and show your original mesh again"

    def execute(self, context):
        remove_preview()
        state["cache"] = None
        state["message"] = ""
        return {"FINISHED"}


class CURIOMESH_OT_region_add_cage(bpy.types.Operator):
    bl_idname = "curiomesh.region_add_cage"
    bl_label = "Add Region Cage"
    bl_description = "Add a box or lattice around your selection (or the 3D cursor). Move and scale it so it covers the part to redo"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(items=[("BOX", "Box", ""), ("LATTICE", "Lattice", "")], default="BOX")

    def execute(self, context):
        settings = context.scene.curiomesh_region
        obj = _active_mesh(context)
        center, size = Vector(context.scene.cursor.location), None
        if obj is not None:
            if obj.mode == "EDIT":
                obj.update_from_editmode()
            co, _faces, _starts, _sizes = _mesh_arrays(obj)
            select = np.zeros(len(obj.data.vertices), dtype=bool)
            obj.data.vertices.foreach_get("select", select)
            matrix = np.asarray(obj.matrix_world, dtype=np.float64)
            if obj.mode == "EDIT" and select.any():
                world = co[select] @ matrix[:3, :3].T + matrix[:3, 3]
                low, high = world.min(axis=0), world.max(axis=0)
                center = Vector(((low + high) * 0.5).tolist())
                size = np.maximum((high - low) * 1.1, 1e-3 * max(obj.dimensions))
            else:
                size = np.full(3, 0.25 * max(max(obj.dimensions), 1e-3))
        if size is None:
            size = np.full(3, 0.5)
        collection = context.collection or context.scene.collection
        if self.kind == "BOX":
            cage = bpy.data.objects.new("CurioRegion_Box", None)
            cage.empty_display_type = "CUBE"
            cage.empty_display_size = 1.0
            cage.scale = (size * 0.5).tolist()
        else:
            lattice = bpy.data.lattices.new("CurioRegion_Lattice")
            lattice.points_u = lattice.points_v = lattice.points_w = 3
            lattice.interpolation_type_u = lattice.interpolation_type_v = lattice.interpolation_type_w = "KEY_LINEAR"
            cage = bpy.data.objects.new("CurioRegion_Lattice", lattice)
            cage.scale = size.tolist()
        cage.location = center
        cage.show_in_front = True
        collection.objects.link(cage)
        item = settings.volumes.add()
        item.obj = cage
        item.operation = "UNION"
        settings.volume_index = len(settings.volumes) - 1
        settings.source = "VOLUMES"
        if obj is not None:
            state["source"] = obj.name
        self.report({"INFO"}, f"Added {cage.name}; move or scale it, then press Preview")
        return {"FINISHED"}


class CURIOMESH_OT_region_volume_add(bpy.types.Operator):
    bl_idname = "curiomesh.region_volume_add"
    bl_label = "Add Volume"
    bl_description = "Add the other selected empties, lattices or closed meshes as region volumes (or an empty slot)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.curiomesh_region
        picked = [o for o in context.selected_objects if o != context.object and _volume_poll(None, o)]
        for vol in picked or [None]:
            item = settings.volumes.add()
            if vol is not None:
                item.obj = vol
        settings.volume_index = len(settings.volumes) - 1
        settings.source = "VOLUMES"
        return {"FINISHED"}


class CURIOMESH_OT_region_volume_remove(bpy.types.Operator):
    bl_idname = "curiomesh.region_volume_remove"
    bl_label = "Remove Volume"
    bl_description = "Remove the highlighted volume from the list (the object stays)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.curiomesh_region
        if 0 <= settings.volume_index < len(settings.volumes):
            settings.volumes.remove(settings.volume_index)
            settings.volume_index = max(0, settings.volume_index - 1)
        return {"FINISHED"}


def _surface_bvh(obj, depsgraph) -> BVHTree:
    """World-space BVH of an object's evaluated surface."""

    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = obj.matrix_world
        verts = [(matrix @ v.co)[:] for v in mesh.vertices]
        polys = [tuple(p.vertices) for p in mesh.polygons]
    finally:
        evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys)


class CURIOMESH_OT_fit_to_surface(bpy.types.Operator):
    bl_idname = "curiomesh.fit_to_surface"
    bl_label = "Fit to Surface"
    bl_description = (
        "Relax vertices along the surface and snap them back onto it; open borders (and unselected vertices "
        "in Edit Mode) stay put"
    )
    bl_options = {"REGISTER", "UNDO"}

    iterations: bpy.props.IntProperty(name="Iterations", default=5, min=1, max=100)
    strength: bpy.props.FloatProperty(name="Strength", default=0.5, min=0.0, max=1.0)

    def execute(self, context):
        obj = _active_mesh(context)
        if obj is None:
            self.report({"ERROR"}, "Select a mesh")
            return {"CANCELLED"}
        settings = context.scene.curiomesh_region
        depsgraph = context.evaluated_depsgraph_get()
        others = [o for o in context.selected_objects if o != obj and o.type == "MESH"]
        target = others[0] if others else settings.fit_target
        if target is None and obj.get("curiomesh_region_source"):
            target = bpy.data.objects.get(str(obj["curiomesh_region_source"]))
        edit = obj.mode == "EDIT"
        if edit:
            bpy.ops.object.mode_set(mode="OBJECT")
        target = target if target is not None and target != obj else obj
        bvh = _surface_bvh(target, depsgraph)
        co, faces, _starts, _sizes = _mesh_arrays(obj)
        matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        inverse = np.linalg.inv(matrix)
        world = co @ matrix[:3, :3].T + matrix[:3, 3]
        pinned = np.zeros(len(co), dtype=bool)
        _face_of, a, b = patch._half_edges(faces)
        key = np.minimum(a, b) * len(co) + np.maximum(a, b)
        unique, counts = np.unique(key, return_counts=True)
        open_edges = unique[counts == 1]
        pinned[open_edges // len(co)] = True
        pinned[open_edges % len(co)] = True
        if edit:
            select = np.zeros(len(co), dtype=bool)
            obj.data.vertices.foreach_get("select", select)
            pinned |= ~select
        fitted = patch.fit_to_surface(world, faces, pinned, _projector(bvh), self.iterations, self.strength)
        local = fitted @ inverse[:3, :3].T + inverse[:3, 3]
        obj.data.vertices.foreach_set("co", local.astype(np.float32).ravel())
        obj.data.update()
        if edit:
            bpy.ops.object.mode_set(mode="EDIT")
        moved = int(np.count_nonzero(~pinned))
        self.report({"INFO"}, f"Fitted {moved:,} vertices onto {target.name}")
        return {"FINISHED"}


def _hole_loops(bm) -> list[list]:
    """Closed loops of selected open edges, ordered the way the new faces must run."""

    neighbours: dict = {}
    for edge in bm.edges:
        if edge.select and edge.is_boundary:
            a, b = edge.verts
            neighbours.setdefault(a, []).append(b)
            neighbours.setdefault(b, []).append(a)
    loops, seen = [], set()
    for start in neighbours:
        if start in seen or len(neighbours[start]) != 2:
            continue
        order, previous, current = [start], start, neighbours[start][0]
        seen.add(start)
        while current is not start:
            links = neighbours.get(current, ())
            if len(links) != 2 or current in seen:
                order = []
                break
            order.append(current)
            seen.add(current)
            previous, current = current, (links[1] if links[0] is previous else links[0])
        if len(order) < 3:
            continue
        edge = bm.edges.get((order[0], order[1]))
        face_loop = next(loop for loop in edge.link_faces[0].loops if loop.edge == edge)
        if face_loop.vert is order[0]:  # the old face runs 0 -> 1, so the fill runs 1 -> 0
            order.reverse()
        loops.append(order)
    return loops


class CURIOMESH_OT_fill_loop(bpy.types.Operator):
    bl_idname = "curiomesh.fill_loop"
    bl_label = "Fill Loop"
    bl_description = "Fill each selected closed hole loop with an all-quad patch that follows the surrounding curvature"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == "MESH" and obj.mode == "EDIT"

    def execute(self, context):
        settings = context.scene.curiomesh_region
        obj = context.object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        loops = _hole_loops(bm)
        if not loops:
            self.report({"ERROR"}, "Select a closed loop of open (hole) edges")
            return {"CANCELLED"}
        co = np.asarray([v.co[:] for v in bm.verts], dtype=np.float64)
        scale = _world_scale(obj)
        made, skipped = 0, []
        for loop in loops:
            ids = [v.index for v in loop]
            members = set(ids)
            ring = sorted({(v.index, e.other_vert(v).index) for v in loop for e in v.link_edges if e.other_vert(v).index not in members})
            result = patch.fill_loop(
                co,
                ids,
                ring_edges=ring,
                target_length=settings.target_edge / scale,
                strategy=settings.strategy,
                seam_layer=settings.seam_layer,
                field_mode=settings.field_mode,
            )
            if not result.ok:
                skipped.append(result.message)
                continue
            neighbour = next((f for v in loop for f in v.link_faces), None)
            verts = list(bm.verts)
            fresh = [bm.verts.new(p) for p in result.points.tolist()]
            for face in result.faces:
                corners = [verts[v] if v < result.base else fresh[v - result.base] for v in face]
                try:
                    new_face = bm.faces.new(corners)
                except ValueError:
                    continue
                if neighbour is not None:
                    new_face.material_index = neighbour.material_index
                    new_face.smooth = neighbour.smooth
                new_face.select = True
                made += 1
        bm.normal_update()
        bmesh.update_edit_mesh(obj.data)
        if not made:
            self.report({"ERROR"}, skipped[0] if skipped else "Nothing filled")
            return {"CANCELLED"}
        note = f" ({len(skipped)} loop(s) skipped: {skipped[0]})" if skipped else ""
        self.report({"INFO"}, f"Filled {len(loops) - len(skipped)} loop(s) with {made:,} quads{note}"[:200])
        return {"FINISHED"}


# --------------------------------------------------------------------------- panel


class CURIOMESH_UL_region_volumes(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "operation", text="", icon_only=True)
        row.prop(item, "obj", text="")


def _is_problem(message: str) -> bool:
    return any(word in message for word in ("skipped", "error", "Error", "paused")) or message.startswith(
        ("Select", "No ", "Add", "Pick", "Nothing")
    )


def _selected_face_count(obj) -> int | None:
    """Faces selected in Edit Mode, or None outside it."""
    try:
        if obj is not None and obj.type == "MESH" and obj.mode == "EDIT":
            return int(obj.data.total_face_sel)
    except Exception:
        pass
    return None


def _draw_pick_part(layout, settings, obj, has_mesh, compact):
    layout.label(text="1. Pick the part to redo", icon="RESTRICT_SELECT_OFF")
    layout.grid_flow(row_major=True, columns=2, even_columns=True, align=True).prop(settings, "source", expand=True)
    box = layout.box()
    if settings.source == "SELECTION":
        count = _selected_face_count(obj)
        if count is None:
            box.label(text="Select faces in Edit Mode", icon="EDITMODE_HLT")
            row = box.row()
            row.enabled = has_mesh
            row.operator("object.mode_set", text="Enter Edit Mode", icon="EDITMODE_HLT").mode = "EDIT"
        elif count == 0:
            box.alert = True
            box.label(text="No faces selected yet", icon="ERROR")
        else:
            box.label(text=f"{count:,} faces selected", icon="CHECKMARK")
    elif settings.source == "FACE_SET":
        col = box.column(align=True)
        col.prop(settings, "face_set_id")
        col.prop(settings, "face_set_attribute", text="Attribute")
        if not compact:
            col.label(text="Sculpt Face Sets work out of the box.")
    elif settings.source == "VERTEX_GROUP":
        if has_mesh:
            box.prop_search(settings, "vertex_group", obj, "vertex_groups", text="")
        if not settings.vertex_group:
            box.label(text="Pick a vertex group", icon="INFO")
        elif not compact:
            box.label(text="Faces with weight 0.5 or more")
    else:
        row = box.row(align=True)
        row.scale_y = 1.3
        row.operator("curiomesh.region_add_cage", text="Box Cage", icon="CUBE").kind = "BOX"
        row.operator("curiomesh.region_add_cage", text="Lattice Cage", icon="LATTICE_DATA").kind = "LATTICE"
        row = box.row()
        row.template_list("CURIOMESH_UL_region_volumes", "", settings, "volumes", settings, "volume_index", rows=3)
        col = row.column(align=True)
        col.operator("curiomesh.region_volume_add", text="", icon="ADD")
        col.operator("curiomesh.region_volume_remove", text="", icon="REMOVE")
        if not len(settings.volumes):
            box.label(text="Add a cage, then move it over the part", icon="INFO")
        elif not compact:
            box.label(text="Cages combine from top to bottom")


class CURIOMESH_PT_regions(bpy.types.Panel):
    bl_label = "Regions"
    bl_idname = "CURIOMESH_PT_regions"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return bool(overlay.get_prefs().show_regions)

    def draw_header(self, context):
        self.layout.label(icon="SELECT_SUBTRACT")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.curiomesh_region
        compact = bool(overlay.get_prefs().compact_mode)
        obj = context.object
        has_mesh = obj is not None and obj.type == "MESH" and obj.name != PREVIEW_NAME
        previewing = _preview_object() is not None

        if not has_mesh and not previewing:
            box = layout.box()
            box.label(text="Select a mesh to begin", icon="INFO")

        _draw_pick_part(layout, settings, obj, has_mesh, compact)

        layout.label(text="2. Choose the new quads", icon="MESH_GRID")
        layout.row(align=True).prop(settings, "strategy", expand=True)
        col = layout.column(align=True)
        col.prop(settings, "target_edge")
        if settings.target_edge == 0.0 and not compact:
            col.label(text="0 = match the surrounding quads")

        layout.label(text="3. Preview, then apply", icon="HIDE_OFF")
        ready = has_mesh or previewing
        # One primary action at a time: Preview first, Apply once there is something to apply.
        preview_row = layout.row(align=True)
        preview_row.enabled = ready
        preview_row.scale_y = 1.2 if previewing else 1.9
        preview_row.operator(
            "curiomesh.region_preview", text="Refresh Preview" if previewing else "Preview", icon="HIDE_OFF"
        )
        row = layout.row(align=True)
        row.scale_y = 1.9 if previewing else 1.2
        apply_row = row.row(align=True)
        apply_row.enabled = previewing or has_mesh
        apply_row.operator("curiomesh.region_apply", text="Apply", icon="CHECKMARK")
        sub = row.row(align=True)
        sub.enabled = previewing
        sub.operator("curiomesh.region_cancel", text="Cancel", icon="X")
        layout.prop(settings, "apply_mode", text="")

        message = status()
        if message:
            box = layout.box()
            box.alert = _is_problem(message)
            for i, chunk in enumerate(message.split(" | ")[:2]):
                overlay.wrap_label(box, chunk, icon=("ERROR" if box.alert else "INFO") if i == 0 else "BLANK1", width=36, limit=2)
        elif not previewing and not compact:
            layout.label(text="Nothing changes until you press Apply.", icon="LOCKED")
        if previewing:
            if compact:
                row = layout.row(align=True)
                overlay.pref_swatch(row, "region_tint")
                overlay.pref_swatch(row, "region_seam_color")
            else:
                legend = layout.column(align=True)
                legend.use_property_split = True
                legend.use_property_decorate = False
                overlay.pref_swatch(legend, "region_tint", "New faces")
                overlay.pref_swatch(legend, "region_seam_color", "Kept seam")

        layout.label(text="Tools", icon="TOOL_SETTINGS")
        row = layout.row(align=True)
        row.scale_y = 1.2
        row.operator("curiomesh.fit_to_surface", icon="SNAP_FACE")
        row.operator("curiomesh.fill_loop", icon="MOD_TRIANGULATE")


class CURIOMESH_PT_regions_advanced(bpy.types.Panel):
    bl_label = "Advanced"
    bl_idname = "CURIOMESH_PT_regions_advanced"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"
    bl_parent_id = "CURIOMESH_PT_regions"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.label(icon="PREFERENCES")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.curiomesh_region
        layout.prop(settings, "seam_layer")
        layout.prop(settings, "field_mode")
        layout.prop(settings, "falloff")
        layout.prop(settings, "fit_iterations")
        layout.prop(settings, "min_island")
        layout.prop(settings, "smooth_seam")
        layout.prop(settings, "live")
        layout.prop(settings, "fit_target")


PANEL_CLASSES = (CURIOMESH_PT_regions, CURIOMESH_PT_regions_advanced)


CLASSES = (
    CURIOMESH_PG_region_volume,
    CURIOMESH_PG_region,
    CURIOMESH_OT_region_preview,
    CURIOMESH_OT_region_apply,
    CURIOMESH_OT_region_cancel,
    CURIOMESH_OT_region_add_cage,
    CURIOMESH_OT_region_volume_add,
    CURIOMESH_OT_region_volume_remove,
    CURIOMESH_OT_fit_to_surface,
    CURIOMESH_OT_fill_loop,
    CURIOMESH_UL_region_volumes,
    CURIOMESH_PT_regions,
    CURIOMESH_PT_regions_advanced,
)

_draw_handles: list = []


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_region = bpy.props.PointerProperty(type=CURIOMESH_PG_region)
    _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW"))


def unregister() -> None:
    while _draw_handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handles.pop(), "WINDOW")
        except Exception:
            pass
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    try:
        remove_preview()
    except Exception:
        pass
    state.update(source="", sig=None, message="", cache=None, draw=None, batches=None)
    if hasattr(bpy.types.Scene, "curiomesh_region"):
        del bpy.types.Scene.curiomesh_region
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
