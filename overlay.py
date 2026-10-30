# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Viewport HUD dashboard and triangle/pole heatmap (gpu + blf draw handlers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import bpy
import numpy as np

PREVIEW_NAME = "CurioMesh_Preview"
REGION_PREVIEW_NAME = "CurioMesh_RegionPreview"
HEATMAP_MAX_FACES = 300_000
HEATMAP_MAX_POLE_POINTS = 60_000

TRI_COLOR = (1.0, 0.18, 0.2, 0.45)
NGON_COLOR = (0.62, 0.25, 0.95, 0.45)
POLE3_COLOR = (0.25, 0.55, 1.0, 1.0)
POLE5_COLOR = (1.0, 0.68, 0.1, 1.0)
REGION_TINT = (0.2, 0.75, 1.0, 0.28)
SEAM_COLOR = (1.0, 0.62, 0.1, 1.0)
POLE5_SIZE_BOOST = 1.45  # 5+ poles are bigger as well as orange, so colour is never the only cue

HUD_CORNERS = (
    ("BOTTOM_LEFT", "Bottom Left", "Lower left of the viewport (clear of the toolbar)"),
    ("BOTTOM_RIGHT", "Bottom Right", "Lower right of the viewport (clear of the sidebar)"),
    ("TOP_LEFT", "Top Left", "Upper left of the viewport"),
    ("TOP_RIGHT", "Top Right", "Upper right of the viewport (clear of the navigation gizmo)"),
)

# Single source of truth for the add-on preferences (also used when the add-on is not enabled, e.g. in tests).
PREF_DEFAULTS = {
    "panel_tab": "CurioMesh",
    "compact_mode": False,
    "show_scan": True,
    "show_atlas": True,
    "show_shape": True,
    "show_pipeline": True,
    "show_pairclosure": True,
    "show_materials": True,
    "show_batch": True,
    "show_diagnostics": True,
    "show_debug": True,
    "show_regions": True,
    "hud_default": True,
    "hud_corner": "BOTTOM_LEFT",
    "hud_offset_x": 70,
    "hud_offset_y": 40,
    "hud_scale": 1.0,
    "hud_opacity": 0.86,
    "heat_tri_color": TRI_COLOR,
    "heat_ngon_color": NGON_COLOR,
    "heat_pole3_color": POLE3_COLOR,
    "heat_pole5_color": POLE5_COLOR,
    "heat_point_size": 7.0,
    "region_tint": REGION_TINT,
    "region_seam_color": SEAM_COLOR,
    "region_seam_width": 3.0,
}
_DEFAULT_PREFS = SimpleNamespace(**PREF_DEFAULTS)
_prefs_override = None  # tests may point this at any object with the PREF_DEFAULTS attributes

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


def get_prefs():
    """The add-on preferences, or the built-in defaults when the add-on is not enabled."""
    if _prefs_override is not None:
        return _prefs_override
    try:
        addon = bpy.context.preferences.addons.get(__package__)
        if addon is not None and addon.preferences is not None:
            return addon.preferences
    except Exception:
        pass
    return _DEFAULT_PREFS


def pref_color(prefs, name) -> tuple:
    default = PREF_DEFAULTS[name]
    try:
        value = tuple(float(x) for x in getattr(prefs, name))
    except Exception:
        return default
    return value if len(value) == 4 else default


def heat_colors(prefs=None) -> tuple:
    """(triangle, n-gon, 3-pole, 5+-pole) RGBA tuples from the preferences."""
    prefs = prefs or get_prefs()
    return tuple(pref_color(prefs, n) for n in ("heat_tri_color", "heat_ngon_color", "heat_pole3_color", "heat_pole5_color"))


def pref_swatch(layout, name: str, text: str = "") -> None:
    """A colour swatch bound to a preference (plain label when the preferences are not a live struct)."""
    prefs = get_prefs()
    if isinstance(prefs, bpy.types.AddonPreferences):
        layout.prop(prefs, name, text=text)
    elif text:
        layout.label(text=text)


def wrap_label(layout, text: str, icon: str = "NONE", width: int = 38, limit: int = 4) -> None:
    """Word-wrapped label lines (Blender labels never wrap on their own)."""
    import textwrap

    lines = textwrap.wrap(str(text), width=width) or [""]
    for i, line in enumerate(lines[:limit]):
        layout.label(text=line if i < limit - 1 or len(lines) <= limit else line[: width - 3] + "...", icon=icon if i == 0 else "BLANK1")


def apply_hud_default(scene=None) -> None:
    """Give each scene's HUD toggle the preferred default once (new scenes and files that never had it)."""
    scenes = [scene] if scene is not None else list(bpy.data.scenes)
    for sc in scenes:
        settings = getattr(sc, "curiomesh_settings", None)
        if settings is None or sc.get("curiomesh_hud_init"):
            continue
        sc["curiomesh_hud_init"] = True
        if not get_prefs().hud_default:  # the property default is already on
            settings.hud_enabled = False


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
    if obj.name == REGION_PREVIEW_NAME:
        return "Region preview - press Apply in Regions"
    if obj.mode == "EDIT":
        return "Exit Edit Mode to refresh stats"
    if stats is None or stats.faces == 0:
        return "Mesh has no faces"
    if "curiomesh_engine" not in obj.keys():
        if stats.quad_ratio >= 0.999:
            return "Already all quads - Remesh only to change density"
        return "Press Remesh to build clean quads"
    if stats.tris:
        if settings is not None and not settings.require_quads:
            return f"{stats.tris} triangles left - turn on Pure Quads"
        return f"{stats.tris} triangles left - try Hero quality or PairClosure"
    if stats.ngons:
        return f"{stats.ngons} n-gons left - raise Cleanup"
    target = int(settings.target_faces) if settings is not None else 0
    if target and abs(stats.faces - target) / target > 0.25:
        return "Face count is off target - adjust Target Faces"
    if stats.faces >= 100 and stats.poles > 0.3 * stats.faces:
        return "Many poles - try Hero quality or a lower target"
    return "Looks clean - ready to export"


# --------------------------------------------------------------------------- heatmap data


def pole_indices(stats: MeshStats) -> np.ndarray:
    idx = np.flatnonzero(stats.pole_mask)
    if len(idx) > HEATMAP_MAX_POLE_POINTS:
        idx = idx[:: len(idx) // HEATMAP_MAX_POLE_POINTS + 1]
    return idx


def build_heat_arrays(obj: bpy.types.Object, stats: MeshStats, colors: tuple | None = None):
    """Return (tri_pos, tri_col, pt_pos, pt_col) float32 arrays in object space, or None."""
    tri_c, ngon_c, pole3_c, pole5_c = colors or heat_colors()
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
        colors = np.where((face_size[keep] == 3)[:, None], tri_c, ngon_c).astype(np.float32)
        flat = tv.ravel()
        tri_pos = np.ascontiguousarray(co[flat] + lift[flat], dtype=np.float32)
        tri_col = np.ascontiguousarray(np.repeat(colors, 3, axis=0), dtype=np.float32)

    idx = pole_indices(stats)
    pt_pos = np.ascontiguousarray(co[idx] + lift[idx] * 2.0, dtype=np.float32)
    pt_col = np.where((stats.valence[idx] == 3)[:, None], pole3_c, pole5_c).astype(np.float32)
    # valence 1 or 2 (rare, non-manifold-ish) share the blue "low" colour
    pt_col[stats.valence[idx] < 3] = pole3_c
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
    """(triangle batch, pole batch, big-pole batch); rebuilt when the preference colours change."""
    import gpu
    from gpu_extras.batch import batch_for_shader

    colors = heat_colors()
    if entry.heat_built and entry.extra.get("heat_colors") == colors:
        return entry.heat
    entry.heat_built = True
    entry.extra["heat_colors"] = colors
    entry.heat = (None, None, None)
    arrays = build_heat_arrays(obj, entry.stats, colors)
    if arrays is None:
        return entry.heat
    tri_pos, tri_col, pt_pos, pt_col = arrays
    big = entry.stats.valence[pole_indices(entry.stats)] >= 5
    shader = gpu.shader.from_builtin("FLAT_COLOR")
    tri_batch = batch_for_shader(shader, "TRIS", {"pos": tri_pos, "color": tri_col}) if len(tri_pos) else None

    def points(mask):
        if not mask.any():
            return None
        pos, col = np.ascontiguousarray(pt_pos[mask]), np.ascontiguousarray(pt_col[mask])
        return batch_for_shader(shader, "POINTS", {"pos": pos, "color": col})

    entry.heat = (tri_batch, points(~big), points(big))
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
        point_size = float(get_prefs().heat_point_size) * _ui_scale()
        try:
            for obj in targets:
                # Cheap size check first: stats on a huge mesh cost seconds per draw.
                if len(obj.data.polygons) > HEATMAP_MAX_FACES:
                    continue
                entry = _entry(obj)
                if entry is None or entry.stats.faces > HEATMAP_MAX_FACES:
                    continue
                tri_batch, pt_batch, big_batch = _heat_batches(entry, obj)
                gpu.matrix.push()
                gpu.matrix.multiply_matrix(obj.matrix_world)
                shader.bind()
                if tri_batch:
                    tri_batch.draw(shader)
                if pt_batch:
                    gpu.state.point_size_set(point_size)
                    pt_batch.draw(shader)
                if big_batch:
                    gpu.state.point_size_set(point_size * POLE5_SIZE_BOOST)
                    big_batch.draw(shader)
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


def quality_word(ratio: float | None) -> str:
    """Words as well as colours, so the HUD never relies on red vs green alone."""
    if ratio is None:
        return "NO DATA"
    if ratio >= 0.999:
        return "ALL QUADS"
    if ratio >= 0.95:
        return "CLEAN"
    if ratio >= 0.80:
        return "FAIR"
    return "NEEDS WORK"


def hud_origin(corner, region_w, region_h, w, h, off_x, off_y, right_inset=0.0):
    """Lower-left pixel of the HUD box for a corner; everything in pixels, clamped inside the region."""
    right = str(corner).endswith("RIGHT")
    top = str(corner).startswith("TOP")
    x = region_w - right_inset - off_x - w if right else off_x
    y = region_h - off_y - h if top else off_y
    return max(0.0, x), max(0.0, y)


def _sidebar_inset() -> float:
    """Width of the N-panel when it overlaps the viewport, so right-hand corners stay clear of it."""
    try:
        if not bpy.context.preferences.system.use_region_overlap:
            return 0.0
        space = bpy.context.space_data
        if space is not None and space.type == "VIEW_3D" and space.show_region_ui:
            for region in bpy.context.area.regions:
                if region.type == "UI" and region.width > 1:
                    return float(region.width)
    except Exception:
        pass
    return 0.0


def hud_model(context) -> dict:
    """Everything the HUD shows, as plain data (testable without a GPU)."""
    settings = _settings()
    obj = context.object
    if obj is None or obj.type != "MESH":
        # Working on a region (the source is hidden and a cage is active): report the preview instead.
        obj = bpy.data.objects.get(REGION_PREVIEW_NAME) or obj
    model = {"name": "", "ratio": None, "faces": 0, "target": 0, "tris": 0, "poles": 0, "info": "", "batch": ""}
    if settings is not None:
        model["target"] = int(settings.target_faces)
        model["batch"] = settings.batch_status
    is_mesh = obj is not None and obj.type == "MESH"
    too_big = is_mesh and len(obj.data.polygons) > HEATMAP_MAX_FACES
    stats = get_stats(obj) if is_mesh and not too_big else None
    if is_mesh:
        model["name"] = "Region preview" if obj.name == REGION_PREVIEW_NAME else obj.name
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


def _text_right(xr, y, size, color, text):
    import blf

    blf.size(0, size)
    width = blf.dimensions(0, text)[0]
    return _text(xr - width, y, size, color, text)


def _draw_hud() -> None:
    global LAST_DRAW_ERROR
    try:
        import gpu

        settings = _settings()
        if settings is None or not settings.hud_enabled:
            return
        context = bpy.context
        prefs = get_prefs()
        model = hud_model(context)
        ui = _ui_scale()
        s = ui * float(prefs.hud_scale)
        accent = accent_for_ratio(model["ratio"])
        w, h = 330 * s, 168 * s
        pad = 14 * s
        corner = prefs.hud_corner
        safe_top = (36 + (140 if corner == "TOP_RIGHT" else 0)) * ui if corner.startswith("TOP") else 0.0
        x, y = hud_origin(
            corner,
            context.region.width,
            context.region.height,
            w,
            h,
            float(prefs.hud_offset_x) * ui,
            float(prefs.hud_offset_y) * ui + safe_top,
            _sidebar_inset(),
        )

        muted = (0.72, 0.74, 0.82, 1.0)
        label = (0.62, 0.65, 0.75, 1.0)
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        shader.bind()
        _fill(shader, _rounded_rect(x, y, w, h, 12 * s), (0.07, 0.08, 0.11, float(prefs.hud_opacity)))
        _fill(shader, _rounded_rect(x, y + 12 * s, 5 * s, h - 24 * s, 2.5 * s, 3), (*accent, 1.0))

        left = x + pad + 6 * s
        right = x + w - pad
        top = y + h - pad
        _text(left, top - 10 * s, 11 * s, (*accent, 1.0), "CURIOMESH")
        _text(left + 95 * s, top - 10 * s, 11 * s, muted, model["name"][:18] or "no mesh")
        _text_right(right, top - 10 * s, 10 * s, (*accent, 1.0), quality_word(model["ratio"]))

        ratio = model["ratio"]
        big = f"{ratio * 100:.1f}%" if ratio is not None else "--"
        bw = _text(left, top - 52 * s, 32 * s, (1, 1, 1, 1), big)
        _text(left + bw + 8 * s, top - 52 * s, 11 * s, muted, "quads")

        bar_x, bar_y, bar_w, bar_h = left, top - 66 * s, w - 2 * pad - 6 * s, 7 * s
        shader.bind()
        _fill(shader, _rounded_rect(bar_x, bar_y, bar_w, bar_h, 3.5 * s, 3), (0.2, 0.22, 0.28, 1.0))
        if ratio:
            fill_w = max(bar_h, bar_w * min(1.0, ratio))
            _fill(shader, _rounded_rect(bar_x, bar_y, fill_w, bar_h, 3.5 * s, 3), (*accent, 1.0))
        # Goal tick at 95 % quads.
        tick_x = bar_x + bar_w * 0.95
        _fill(shader, [(tick_x, bar_y - 2 * s), (tick_x + 1.5 * s, bar_y - 2 * s), (tick_x + 1.5 * s, bar_y + bar_h + 2 * s), (tick_x, bar_y + bar_h + 2 * s)], (0.85, 0.87, 0.95, 0.9))

        col_w = (w - 2 * pad - 6 * s) / 3
        cells = (
            ("FACES", f"{model['faces']:,} / {model['target']:,}", (1, 1, 1, 1)),
            ("TRIS", f"! {model['tris']:,}" if model["tris"] else "0", (1.0, 0.5, 0.5, 1) if model["tris"] else (0.6, 0.9, 0.7, 1)),
            ("POLES", f"{model['poles']:,}", (1.0, 0.78, 0.3, 1)),
        )
        for i, (cell_label, value, color) in enumerate(cells):
            cx = left + i * col_w
            _text(cx, top - 90 * s, 10 * s, label, cell_label)
            _text(cx, top - 106 * s, 14 * s, color, value)

        if model["batch"]:
            _text(left, top - 124 * s, 10 * s, (0.5, 0.8, 1.0, 1.0), model["batch"][:46])
        elif model["info"]:
            _text(left, top - 124 * s, 10 * s, label, model["info"][:46])
        _text(left, top - 144 * s, 12 * s, (*accent, 1.0), "> " + model["hint"][:44])
        gpu.state.blend_set("NONE")
    except Exception as exc:
        LAST_DRAW_ERROR = repr(exc)


# --------------------------------------------------------------------------- operators


class CURIOMESH_OT_toggle_heatmap(bpy.types.Operator):
    bl_idname = "curiomesh.toggle_heatmap"
    bl_label = "Show Quality Heatmap"
    bl_description = "Colour the mesh: triangles red, n-gons purple, 3-edge poles blue dots, 5+-edge poles larger orange dots (colours editable in Preferences)"
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


def _init_hud_default():
    try:
        apply_hud_default()
    except Exception:
        pass
    return None


@bpy.app.handlers.persistent
def _on_load_post(*_args):
    bpy.app.timers.register(_init_hud_default, first_interval=0.2)


CLASSES = (CURIOMESH_OT_toggle_heatmap,)


def register() -> None:
    unregister()
    import bpy.app.handlers as handlers

    handlers.depsgraph_update_post.append(_on_depsgraph_update)
    handlers.load_post.append(_on_load_post)
    bpy.app.timers.register(_init_hud_default, first_interval=0.2)
    _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_heatmap, (), "WINDOW", "POST_VIEW"))
    _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_hud, (), "WINDOW", "POST_PIXEL"))


def unregister() -> None:
    import bpy.app.handlers as handlers

    if _on_depsgraph_update in handlers.depsgraph_update_post:
        handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    if _on_load_post in handlers.load_post:
        handlers.load_post.remove(_on_load_post)
    if bpy.app.timers.is_registered(_init_hud_default):
        bpy.app.timers.unregister(_init_hud_default)
    while _draw_handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handles.pop(), "WINDOW")
        except Exception:
            pass
    _stats_cache.clear()
    _geometry_hooks.clear()
