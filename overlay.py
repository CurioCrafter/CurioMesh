"""Viewport HUD dashboard and triangle/pole heatmap (gpu + blf draw handlers)."""

from __future__ import annotations

from dataclasses import dataclass, field

import bpy
import numpy as np

PREVIEW_NAME = "CurioMesh_Preview"
HEATMAP_MAX_FACES = 300_000
HEATMAP_MAX_POLE_POINTS = 60_000

TRI_COLOR = (1.0, 0.18, 0.2, 0.45)
NGON_COLOR = (0.62, 0.25, 0.95, 0.45)
POLE3_COLOR = (0.25, 0.55, 1.0, 1.0)
POLE5_COLOR = (1.0, 0.68, 0.1, 1.0)

_draw_handles: list = []
_stats_cache: dict[str, "_CacheEntry"] = {}
_geometry_hooks: list = []
LAST_DRAW_ERROR = ""


@dataclass
class MeshStats:
    faces: int = 0
    verts: int = 0
    tris: int = 0
    quads: int = 0
    ngons: int = 0
    poles: int = 0
    sizes: np.ndarray | None = None
    valence: np.ndarray | None = None
    pole_mask: np.ndarray | None = None

    @property
    def quad_ratio(self) -> float:
        return self.quads / self.faces if self.faces else 0.0


@dataclass
class _CacheEntry:
    key: tuple
    stats: MeshStats
    heat: object = None  # (tri_batch | None, point_batch | None) once uploaded
    heat_built: bool = False
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- stats


def compute_stats(mesh: bpy.types.Mesh) -> MeshStats:
    """Cheap numpy face-size / valence stats. Poles exclude boundary and non-manifold vertices."""
    nf, nv, ne, nl = len(mesh.polygons), len(mesh.vertices), len(mesh.edges), len(mesh.loops)
    if nf == 0:
        return MeshStats(verts=nv)
    starts = np.empty(nf, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", starts)
    sizes = np.diff(np.append(starts, nl))
    edge_verts = np.empty(ne * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edge_verts)
    loop_edges = np.empty(nl, dtype=np.int32)
    mesh.loops.foreach_get("edge_index", loop_edges)
    edge_faces = np.bincount(loop_edges, minlength=ne)
    valence = np.bincount(edge_verts, minlength=nv)
    open_edge = edge_faces != 2
    bad = np.bincount(edge_verts.reshape(-1, 2)[open_edge].ravel(), minlength=nv) > 0
    pole_mask = (valence > 0) & (valence != 4) & ~bad
    return MeshStats(
        faces=nf,
        verts=nv,
        tris=int(np.count_nonzero(sizes == 3)),
        quads=int(np.count_nonzero(sizes == 4)),
        ngons=int(np.count_nonzero(sizes > 4)),
        poles=int(np.count_nonzero(pole_mask)),
        sizes=sizes,
        valence=valence,
        pole_mask=pole_mask,
    )


def _cache_key(obj: bpy.types.Object) -> tuple:
    mesh = obj.data
    return (mesh.as_pointer(), len(mesh.vertices), len(mesh.polygons), len(mesh.loops))


def _entry(obj: bpy.types.Object) -> _CacheEntry | None:
    if obj is None or obj.type != "MESH":
        return None
    key = _cache_key(obj)
    entry = _stats_cache.get(obj.name)
    if entry is None or entry.key != key:
        entry = _CacheEntry(key, compute_stats(obj.data))
        _stats_cache[obj.name] = entry
    return entry


def get_stats(obj: bpy.types.Object | None) -> MeshStats | None:
    try:
        entry = _entry(obj)
    except Exception:
        return None
    return entry.stats if entry else None


def invalidate(name: str | None = None) -> None:
    if name is None:
        _stats_cache.clear()
    else:
        _stats_cache.pop(name, None)


def add_geometry_hook(fn) -> None:
    if fn not in _geometry_hooks:
        _geometry_hooks.append(fn)


def remove_geometry_hook(fn) -> None:
    if fn in _geometry_hooks:
        _geometry_hooks.remove(fn)


def _on_depsgraph_update(scene, depsgraph) -> None:
    try:
        for update in depsgraph.updates:
            if not update.is_updated_geometry:
                continue
            idb = update.id
            if isinstance(idb, bpy.types.Object):
                invalidate(idb.name)
                names = [idb.name]
            elif isinstance(idb, bpy.types.Mesh):
                names = [n for n, e in _stats_cache.items() if e.key[0] == idb.as_pointer()]
                for n in names:
                    invalidate(n)
                names += [o.name for o in bpy.data.objects if o.type == "MESH" and o.data == idb]
            else:
                continue
            for hook in tuple(_geometry_hooks):
                for n in names:
                    hook(n)
    except Exception:
        pass


def tag_redraw() -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# --------------------------------------------------------------------------- hints


def next_step_hint(obj, stats: MeshStats | None, settings) -> str:
    """One-line guidance; pure so it can be tested without a GPU."""
    if obj is None or obj.type != "MESH":
        return "Select a mesh"
    if obj.mode == "EDIT":
        return "Exit Edit Mode to refresh stats"
    if stats is None or stats.faces == 0:
        return "Mesh has no faces"
    if "curiomesh_engine" not in obj.keys():
        if stats.quad_ratio >= 0.999:
            return "Already all quads - Remesh to change density"
        return "Press Remesh"
    if stats.tris:
        if settings is not None and not settings.require_quads:
            return f"{stats.tris} triangles left - enable Require Quads"
        return f"{stats.tris} triangles left - try Hero quality or PairClosure"
    if stats.ngons:
        return f"{stats.ngons} n-gons left - raise Cleanup Strength"
    target = int(settings.target_faces) if settings is not None else 0
    if target and abs(stats.faces - target) / target > 0.25:
        return "Face count is off target - adjust Target Faces"
    if stats.faces >= 100 and stats.poles > 0.3 * stats.faces:
        return "Many poles - try Hero quality or a lower target"
    return "Looks clean - ready to export"


# --------------------------------------------------------------------------- heatmap data


def build_heat_arrays(obj: bpy.types.Object, stats: MeshStats):
    """Return (tri_pos, tri_col, pt_pos, pt_col) float32 arrays in object space, or None."""
    mesh = obj.data
    nv = stats.verts
    co = np.empty(nv * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    nrm = np.empty(nv * 3, dtype=np.float32)
    mesh.vertex_normals.foreach_get("vector", nrm)
    nrm = nrm.reshape(-1, 3)
    extent = float(np.ptp(co, axis=0).max()) if nv else 1.0
    lift = nrm * (extent * 0.0007)

    tri_pos = np.zeros((0, 3), dtype=np.float32)
    tri_col = np.zeros((0, 4), dtype=np.float32)
    if stats.tris or stats.ngons:
        mesh.calc_loop_triangles()
        nt = len(mesh.loop_triangles)
        tv = np.empty(nt * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tv)
        tp = np.empty(nt, dtype=np.int32)
        mesh.loop_triangles.foreach_get("polygon_index", tp)
        face_size = stats.sizes[tp]
        keep = (face_size == 3) | (face_size > 4)
        tv = tv.reshape(-1, 3)[keep]
        colors = np.where((face_size[keep] == 3)[:, None], TRI_COLOR, NGON_COLOR).astype(np.float32)
        flat = tv.ravel()
        tri_pos = np.ascontiguousarray(co[flat] + lift[flat], dtype=np.float32)
        tri_col = np.ascontiguousarray(np.repeat(colors, 3, axis=0), dtype=np.float32)

    idx = np.flatnonzero(stats.pole_mask)
    if len(idx) > HEATMAP_MAX_POLE_POINTS:
        idx = idx[:: len(idx) // HEATMAP_MAX_POLE_POINTS + 1]
    pt_pos = np.ascontiguousarray(co[idx] + lift[idx] * 2.0, dtype=np.float32)
    pt_col = np.where((stats.valence[idx] == 3)[:, None], POLE3_COLOR, POLE5_COLOR).astype(np.float32)
    # valence 1 or 2 (rare, non-manifold-ish) share the blue "low" colour
    pt_col[stats.valence[idx] < 3] = POLE3_COLOR
    return tri_pos, tri_col, pt_pos, np.ascontiguousarray(pt_col)


# --------------------------------------------------------------------------- drawing


def _settings():
    scene = bpy.context.scene
    return getattr(scene, "curiomesh_settings", None) if scene else None


def _ui_scale() -> float:
    try:
        return max(0.5, float(bpy.context.preferences.system.ui_scale))
    except Exception:
        return 1.0


def _heat_batches(entry: _CacheEntry, obj: bpy.types.Object):
    import gpu
    from gpu_extras.batch import batch_for_shader

    if entry.heat_built:
        return entry.heat
    entry.heat_built = True
    entry.heat = (None, None)
    arrays = build_heat_arrays(obj, entry.stats)
    if arrays is None:
        return entry.heat
    tri_pos, tri_col, pt_pos, pt_col = arrays
    shader = gpu.shader.from_builtin("FLAT_COLOR")
    tri_batch = (
        batch_for_shader(shader, "TRIS", {"pos": tri_pos, "color": tri_col}) if len(tri_pos) else None
    )
    pt_batch = batch_for_shader(shader, "POINTS", {"pos": pt_pos, "color": pt_col}) if len(pt_pos) else None
    entry.heat = (tri_batch, pt_batch)
    return entry.heat


def heatmap_targets(context, settings) -> list[bpy.types.Object]:
    targets = []
    obj = context.object
    if settings.heatmap_enabled and obj is not None and obj.type == "MESH" and obj.mode == "OBJECT":
        targets.append(obj)
    if settings.live_enabled:
        preview = bpy.data.objects.get(PREVIEW_NAME)
        if preview is not None and preview not in targets and preview.visible_get():
            targets.append(preview)
    return targets


def _draw_heatmap() -> None:
    global LAST_DRAW_ERROR
    try:
        import gpu

        context = bpy.context
        settings = _settings()
        if settings is None:
            return
        targets = heatmap_targets(context, settings)
        if not targets:
            return
        shader = gpu.shader.from_builtin("FLAT_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)
        gpu.state.point_size_set(7.0 * _ui_scale())
        try:
            for obj in targets:
                # Cheap size check first: stats on a huge mesh cost seconds per draw.
                if len(obj.data.polygons) > HEATMAP_MAX_FACES:
                    continue
                entry = _entry(obj)
                if entry is None or entry.stats.faces > HEATMAP_MAX_FACES:
                    continue
                tri_batch, pt_batch = _heat_batches(entry, obj)
                gpu.matrix.push()
                gpu.matrix.multiply_matrix(obj.matrix_world)
                shader.bind()
                if tri_batch:
                    tri_batch.draw(shader)
                if pt_batch:
                    pt_batch.draw(shader)
                gpu.matrix.pop()
        finally:
            gpu.state.depth_mask_set(True)
            gpu.state.blend_set("NONE")
    except Exception as exc:
        LAST_DRAW_ERROR = repr(exc)


def _rounded_rect(x, y, w, h, r, seg=6):
    import math

    pts = [(x + w / 2, y + h / 2)]
    corners = ((x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180), (x + w - r, y + r, 270))
    for cx, cy, start in corners:
        for i in range(seg + 1):
            a = math.radians(start + 90.0 * i / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append(pts[1])
    return pts


def _fill(shader, pts, color, mode="TRI_FAN"):
    from gpu_extras.batch import batch_for_shader

    shader.uniform_float("color", color)
    batch_for_shader(shader, mode, {"pos": pts}).draw(shader)


def _text(x, y, size, color, text):
    import blf

    blf.size(0, size)
    blf.color(0, *color)
    blf.position(0, x, y, 0)
    blf.draw(0, text)
    return blf.dimensions(0, text)[0]


def accent_for_ratio(ratio: float | None) -> tuple[float, float, float]:
    if ratio is None:
        return (0.45, 0.55, 1.0)
    if ratio >= 0.95:
        return (0.22, 0.86, 0.46)
    if ratio >= 0.80:
        return (1.0, 0.72, 0.16)
    return (0.96, 0.3, 0.32)


def hud_model(context) -> dict:
    """Everything the HUD shows, as plain data (testable without a GPU)."""
    settings = _settings()
    obj = context.object
    model = {"name": "", "ratio": None, "faces": 0, "target": 0, "tris": 0, "poles": 0, "info": "", "batch": ""}
    if settings is not None:
        model["target"] = int(settings.target_faces)
        model["batch"] = settings.batch_status
    is_mesh = obj is not None and obj.type == "MESH"
    too_big = is_mesh and len(obj.data.polygons) > HEATMAP_MAX_FACES
    stats = get_stats(obj) if is_mesh and not too_big else None
    if is_mesh:
        model["name"] = obj.name
    if too_big:
        model.update(faces=len(obj.data.polygons), hint=f"Over {HEATMAP_MAX_FACES:,} faces - live stats off")
        return model
    if stats is not None and stats.faces:
        model.update(ratio=stats.quad_ratio, faces=stats.faces, tris=stats.tris, poles=stats.poles)
    elif obj is not None and "curiomesh_quad_ratio" in obj.keys():
        model.update(ratio=float(obj["curiomesh_quad_ratio"]), faces=len(obj.data.polygons))
    if obj is not None and "curiomesh_engine" in obj.keys():
        parts = [str(obj.get("curiomesh_engine", "")), str(obj.get("curiomesh_strategy", ""))]
        info = " / ".join(dict.fromkeys(p for p in parts if p))
        elapsed = obj.get("curiomesh_elapsed_ms")
        if elapsed is not None:
            info += f"  {float(elapsed) / 1000.0:.2f} s"
        model["info"] = info
    model["hint"] = next_step_hint(obj, stats, settings)
    return model


def _draw_hud() -> None:
    global LAST_DRAW_ERROR
    try:
        import gpu

        settings = _settings()
        if settings is None or not settings.hud_enabled:
            return
        context = bpy.context
        model = hud_model(context)
        s = _ui_scale()
        accent = accent_for_ratio(model["ratio"])
        w, h = 330 * s, 168 * s
        x, y = 70 * s, 40 * s
        pad = 14 * s

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        shader.bind()
        _fill(shader, _rounded_rect(x, y, w, h, 12 * s), (0.07, 0.08, 0.11, 0.86))
        _fill(shader, _rounded_rect(x, y + 12 * s, 5 * s, h - 24 * s, 2.5 * s, 3), (*accent, 1.0))

        left = x + pad + 6 * s
        top = y + h - pad
        _text(left, top - 10 * s, 11 * s, (*accent, 1.0), "CURIOMESH")
        name = model["name"][:26] or "no mesh"
        _text(left + 95 * s, top - 10 * s, 11 * s, (0.7, 0.72, 0.8, 1.0), name)

        ratio = model["ratio"]
        big = f"{ratio * 100:.1f}%" if ratio is not None else "--"
        bw = _text(left, top - 52 * s, 32 * s, (1, 1, 1, 1), big)
        _text(left + bw + 8 * s, top - 52 * s, 11 * s, (0.7, 0.72, 0.8, 1.0), "quads")

        bar_x, bar_y, bar_w, bar_h = left, top - 66 * s, w - 2 * pad - 6 * s, 7 * s
        shader.bind()
        _fill(shader, _rounded_rect(bar_x, bar_y, bar_w, bar_h, 3.5 * s, 3), (0.2, 0.22, 0.28, 1.0))
        if ratio:
            fill_w = max(bar_h, bar_w * min(1.0, ratio))
            _fill(shader, _rounded_rect(bar_x, bar_y, fill_w, bar_h, 3.5 * s, 3), (*accent, 1.0))

        col_w = (w - 2 * pad - 6 * s) / 3
        cells = (
            ("FACES", f"{model['faces']:,} / {model['target']:,}", (1, 1, 1, 1)),
            ("TRIS", f"{model['tris']:,}", (1.0, 0.45, 0.45, 1) if model["tris"] else (0.6, 0.9, 0.7, 1)),
            ("POLES", f"{model['poles']:,}", (1.0, 0.78, 0.3, 1)),
        )
        for i, (label, value, color) in enumerate(cells):
            cx = left + i * col_w
            _text(cx, top - 90 * s, 9 * s, (0.55, 0.58, 0.68, 1.0), label)
            _text(cx, top - 106 * s, 14 * s, color, value)

        if model["batch"]:
            _text(left, top - 124 * s, 10 * s, (0.5, 0.8, 1.0, 1.0), model["batch"][:46])
        elif model["info"]:
            _text(left, top - 124 * s, 10 * s, (0.6, 0.63, 0.72, 1.0), model["info"][:46])
        _text(left, top - 144 * s, 12 * s, (*accent, 1.0), "> " + model["hint"][:44])
        gpu.state.blend_set("NONE")
    except Exception as exc:
        LAST_DRAW_ERROR = repr(exc)


# --------------------------------------------------------------------------- operators


class CURIOMESH_OT_toggle_heatmap(bpy.types.Operator):
    bl_idname = "curiomesh.toggle_heatmap"
    bl_label = "Toggle Triangle / Pole Heatmap"
    bl_description = "Show leftover triangles (red), n-gons (purple) and poles (blue valence 3, orange 5+)"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        settings = context.scene.curiomesh_settings
        settings.heatmap_enabled = not settings.heatmap_enabled
        obj = context.object
        if settings.heatmap_enabled and obj is not None and obj.type == "MESH":
            stats = get_stats(obj)
            if stats is not None and stats.faces > HEATMAP_MAX_FACES:
                self.report({"WARNING"}, f"Heatmap skipped: over {HEATMAP_MAX_FACES:,} faces.")
        return {"FINISHED"}


CLASSES = (CURIOMESH_OT_toggle_heatmap,)


def register() -> None:
    unregister()
    import bpy.app.handlers as handlers

    handlers.depsgraph_update_post.append(_on_depsgraph_update)
    _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_heatmap, (), "WINDOW", "POST_VIEW"))
    _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_hud, (), "WINDOW", "POST_PIXEL"))


def unregister() -> None:
    import bpy.app.handlers as handlers

    if _on_depsgraph_update in handlers.depsgraph_update_post:
        handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    while _draw_handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handles.pop(), "WINDOW")
        except Exception:
            pass
    _stats_cache.clear()
    _geometry_hooks.clear()
