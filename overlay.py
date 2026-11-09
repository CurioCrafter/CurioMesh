# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Viewport HUD dashboard and triangle/pole heatmap (gpu + blf draw handlers)."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from types import SimpleNamespace

import bpy
import numpy as np

from .translations import clip, tip, tr

PREVIEW_NAME = "CurioMesh_Preview"  # the name a new preview object gets; it is found by its tag, never by its name
REGION_PREVIEW_NAME = "CurioMesh_RegionPreview"
PREVIEW_TAG = "curiomesh_preview"  # on objects (and spare meshes) the add-on made itself: which tool made them
LIVE, REGION, GUIDED = "LIVE", "REGION", "GUIDED"
_previews: dict[str, str] = {}  # tool -> the name Blender gave the preview object it made this session


def tag_preview(obj, kind: str) -> None:
    """Mark `obj` as the preview `kind` made, so it is found (and removed) by that mark, not by its name."""
    obj[PREVIEW_TAG] = kind
    _previews[kind] = obj.name


def preview_object(kind: str):
    """The preview object `kind` made, or None. Only an object carrying that tool's mark counts."""
    obj = bpy.data.objects.get(_previews.get(kind, ""))
    return obj if obj is not None and obj.get(PREVIEW_TAG) == kind else None


def is_preview(obj, kind: str | None = None) -> bool:
    if obj is None:
        return False
    tag = obj.get(PREVIEW_TAG)
    return tag is not None if kind is None else tag == kind


def remove_saved_previews() -> None:
    """After a file loads: previews saved in it can't be applied any more, so the ones the add-on made go (a source
    the Region preview hid is shown again). Objects without the add-on's mark are never touched."""
    _previews.clear()
    for obj in [o for o in bpy.data.objects if o.get(PREVIEW_TAG) is not None]:
        source = bpy.data.objects.get(str(obj.get("curiomesh_hid", ""))) if obj.get(PREVIEW_TAG) == REGION else None
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if isinstance(mesh, bpy.types.Mesh) and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        if source is not None:
            try:
                source.hide_set(False)
            except RuntimeError:  # not in this view layer
                pass
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

# Single source of truth for the add-on preferences (also used while the add-on's preferences are not registered).
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
    "show_debug": False,
    "show_regions": True,
    "show_guided": True,
    "advanced_mode": False,
    "flow_editor": False,
    "allow_helpers": False,
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
    "guided_detail_color": (1.0, 0.55, 0.1, 1.0),
    "guided_loop_color": (0.1, 0.85, 0.75, 1.0),
    "guided_sharp_color": (0.95, 0.25, 0.65, 1.0),
    "guided_calm_color": (0.35, 0.55, 1.0, 1.0),
    "guided_outline_color": (1.0, 0.95, 0.4, 1.0),
    "guided_lock_color": (0.3, 0.5, 1.0, 0.4),
}
_DEFAULT_PREFS = SimpleNamespace(**PREF_DEFAULTS)

_draw_handles: list = []
_stats_cache: dict[str, "_CacheEntry"] = {}
_draw_errors: set[str] = set()
_built: dict[str, tuple] = {}  # name -> (key, arrays) a draw callback shows
_pending: dict[str, tuple] = {}  # name -> (key, build) for the next timer tick


def drawn_arrays(name: str, key, build):
    """The arrays a draw callback shows under `name`. When `key` changes, `build()` runs in a one-shot timer, never inside
    the draw; until then the arrays last built are returned (None before the first build)."""
    have = _built.get(name)
    if have is None or have[0] != key:
        _pending[name] = (key, build)
        if not bpy.app.timers.is_registered(_build_pending):
            bpy.app.timers.register(_build_pending, first_interval=0.0)
    return have[1] if have is not None else None


def _build_pending():
    while _pending:
        name, (key, build) = _pending.popitem()
        try:
            _built[name] = (key, build())
        except Exception:
            _built[name] = (key, None)  # not tried again until the key changes
            draw_failed(name)
    tag_redraw()
    return None


def draw_failed(where: str) -> None:
    """Called from a draw callback's except block: print the traceback once per callback, not on every redraw."""
    if where not in _draw_errors:
        _draw_errors.add(where)
        traceback.print_exc()


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
    heat_arrays: tuple | None = None  # (colours, arrays) built by the watch timer for the heatmap
    heat: tuple | None = None  # (arrays, batches): the GPU batches the draw callback made from those arrays


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
    """Changes when the mesh is edited: its sizes, and the sum of its corners for a move that keeps the sizes."""
    mesh = obj.data
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    # Limitation: a topology edit that keeps every count and corner (an edge spun in place) keeps the old stats until the next real change
    return (mesh.as_pointer(), len(mesh.vertices), len(mesh.polygons), len(mesh.loops), float(co.sum(dtype=np.float64)))


def _entry(obj: bpy.types.Object) -> _CacheEntry | None:
    if obj is None or obj.type != "MESH":
        return None
    key = _cache_key(obj)
    entry = _stats_cache.get(obj.name)
    if entry is None or entry.key != key:
        entry = _CacheEntry(key, compute_stats(obj.data))
        _stats_cache[obj.name] = entry
    return entry


def cached_stats(obj: bpy.types.Object | None) -> MeshStats | None:
    """The stats the watch timer last measured for this object (draw callbacks read only this; they measure nothing)."""
    entry = _stats_cache.get(obj.name) if obj is not None else None
    return entry.stats if entry else None


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


def get_prefs():
    """The add-on preferences, or the built-in defaults in a helper Blender (it imports the add-on without enabling it
    in its preferences) and while the add-on is being enabled or disabled."""
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
        layout.prop(prefs, name, text=tr(text))
    elif text:
        layout.label(text=tr(text))


def wrap_label(layout, text: str, icon: str = "NONE", width: int = 38, limit: int = 4) -> None:
    """Word-wrapped label lines (Blender labels never wrap on their own), in the user's language.
    A translation longer than the English gets as many more lines as it needs, so nothing is cut off."""
    from . import translations as T

    shown = T.tr(str(text))
    if text and shown != text:
        limit = max(limit, -(-limit * T.text_width(shown) // max(T.text_width(str(text)), 1)))
    lines = T.wrap(shown, width) or [""]
    for i, line in enumerate(lines[:limit]):
        layout.label(text=line if i < limit - 1 or len(lines) <= limit else T.clip(line, width - 3) + "...", icon=icon if i == 0 else "BLANK1", translate=False)


def tag_redraw() -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    if not bpy.app.timers.is_registered(_hud_tick):
        ensure_ticker()
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# --------------------------------------------------------------------------- hints


def next_step_hint(obj, stats: MeshStats | None, settings) -> str:
    """One-line guidance, from plain data (no drawing)."""
    if obj is None or obj.type != "MESH":
        return "Select a mesh"
    if is_preview(obj, REGION):
        return "Region preview - press Apply in Regions"
    if obj.mode == "EDIT":
        return "Exit Edit Mode to refresh stats"
    if stats is None or stats.faces == 0:
        return "Mesh has no faces"
    if "curiomesh_engine" not in obj.keys():
        if stats.quad_ratio >= 0.999:
            return "Already all quads - Auto or Remesh to lighten it"
        return "Press Auto or Remesh for clean quads"
    if stats.tris:
        if settings is not None and not settings.require_quads:
            return f"{stats.tris} triangles left - turn on All Quads"
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


def _heat_batches(entry: _CacheEntry):
    """(triangle batch, pole batch, big-pole batch), uploaded from the arrays the watch timer built."""
    import gpu
    from gpu_extras.batch import batch_for_shader

    arrays = entry.heat_arrays[1] if entry.heat_arrays else None
    if entry.heat is not None and entry.heat[0] is arrays:
        return entry.heat[1]
    entry.heat = (arrays, (None, None, None))
    if arrays is None:
        return entry.heat[1]
    tri_pos, tri_col, pt_pos, pt_col = arrays
    big = entry.stats.valence[pole_indices(entry.stats)] >= 5
    shader = gpu.shader.from_builtin("FLAT_COLOR")
    tri_batch = batch_for_shader(shader, "TRIS", {"pos": tri_pos, "color": tri_col}) if len(tri_pos) else None

    def points(mask):
        if not mask.any():
            return None
        pos, col = np.ascontiguousarray(pt_pos[mask]), np.ascontiguousarray(pt_col[mask])
        return batch_for_shader(shader, "POINTS", {"pos": pos, "color": col})

    entry.heat = (arrays, (tri_batch, points(~big), points(big)))
    return entry.heat[1]


def heatmap_targets(context, settings) -> list[bpy.types.Object]:
    targets = []
    obj = context.object
    if settings.heatmap_enabled and obj is not None and obj.type == "MESH" and obj.mode == "OBJECT":
        targets.append(obj)
    if settings.live_enabled:
        preview = preview_object(LIVE)
        if preview is not None and preview not in targets and preview.visible_get():
            targets.append(preview)
    return targets


def _draw_heatmap() -> None:
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
                entry = _stats_cache.get(obj.name)
                if entry is None or entry.heat_arrays is None:
                    continue
                tri_batch, pt_batch, big_batch = _heat_batches(entry)
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
    except Exception:
        draw_failed("overlay._draw_heatmap")


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

    text = tr(text)
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


HUD_FULL_H, HUD_JOB_EXTRA, HUD_STRIP_H = 168, 36, 30  # box heights before scaling: full, extra while a job runs, collapsed strip
HUD_TICK_SECONDS = 0.25


def clock(seconds) -> str:
    minutes, secs = divmod(int(round(max(0.0, seconds))), 60)
    return f"{minutes}:{secs:02d}"


def job_info():
    """The job running now as plain data for the dashboard, or None. Reads guided.progress_info and the job's own counters."""
    import math
    import time

    try:
        from . import guided  # late: guided imports this module

        info = guided.progress_info()
        if info is None:
            return None
        frac, words = info
        runner = guided.state["work"]["runner"]
    except Exception:
        return None
    done, total = getattr(runner, "progress", None) or (0, 0)
    started = getattr(runner, "started", None)
    left_of = getattr(runner, "seconds_left", None)
    try:
        left = left_of() if left_of else None
    except Exception:
        left = None
    now = time.perf_counter()
    return {
        "frac": min(1.0, max(0.0, float(frac))),
        "stage": str(getattr(runner, "stage", "") or words),
        "done": int(done),
        "total": int(total),
        "elapsed": now - started if started else None,
        "left": left,
        "alive_tick": math.modf(now * 1.5)[0],  # 0 to 1, repeating: the draw turns it into the pulse
    }


def job_text(job: dict) -> str:
    """'42%  piece 3 of 8  1:12 elapsed  about 2:00 left' (parts left out when not known)."""
    parts = [f"{job['frac'] * 100:.0f}%"]
    if job["total"]:
        parts.append(f"piece {min(job['done'] + 1, job['total'])} of {job['total']}")
    if job["elapsed"] is not None:
        parts.append(f"{clock(job['elapsed'])} elapsed")
    if job["left"] is not None:
        parts.append(f"about {clock(job['left'])} left")
    return "  ".join(parts)


def _hud_tick():
    """Redraw while a job runs so the box never looks frozen. Does nothing else; stops by itself when the job ends."""
    settings = _settings()
    live = settings is not None and settings.hud_enabled and job_info() is not None
    tag_redraw()
    return HUD_TICK_SECONDS if live else None


def ensure_ticker() -> None:
    """Start the redraw ticker if the dashboard is on and a job runs. Safe to call often (guided and the panel may call it)."""
    if bpy.app.timers.is_registered(_hud_tick):
        return
    settings = _settings()
    if settings is not None and settings.hud_enabled and job_info() is not None:
        bpy.app.timers.register(_hud_tick, first_interval=HUD_TICK_SECONDS)


def hud_object(context):
    """(object the HUD reports on, whether it is a result waiting for Apply)."""
    from . import guided  # late: guided imports this module

    obj = context.object
    if obj is None or obj.type != "MESH":
        # Working on a region (the source is hidden and a cage is active): report the preview instead.
        obj = preview_object(REGION) or guided.state_source() or obj  # a job can leave nothing active for a moment
    # While an Auto Segments result waits for Apply, the HUD reports the result, not the original it came from.
    preview = preview_object(GUIDED)
    showing_result = preview is not None and preview.type == "MESH" and obj is not None and obj in (guided.state_source(), preview)
    if showing_result:
        obj = preview
    return obj, showing_result


def hud_model(context) -> dict:
    """Everything the HUD shows, as plain data (no drawing). Reads the cached stats only."""
    settings = _settings()
    obj, showing_result = hud_object(context)
    model = {"name": "", "ratio": None, "faces": 0, "target": 0, "tris": 0, "poles": 0, "info": "", "batch": "", "original": False, "job": job_info()}
    if settings is not None:
        model["target"] = int(settings.target_faces)
        model["batch"] = settings.batch_status
    is_mesh = obj is not None and obj.type == "MESH"
    too_big = is_mesh and len(obj.data.polygons) > HEATMAP_MAX_FACES
    stats = cached_stats(obj) if is_mesh and not too_big else None
    if is_mesh:
        model["name"] = "Region preview" if is_preview(obj, REGION) else "Result preview" if showing_result else obj.name
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
    if showing_result:
        auto = getattr(getattr(context.scene, "curiomesh_guided", None), "scope", "") == "AUTO"
        model["target"] = 0  # Classic Remesh's Target Faces is not what this result aimed for
        model["hint"] = "Check the parts, then press Apply" if auto else "Press Apply when it looks right"
        from . import ladder

        model["hint"] = ladder.slider_words() or model["hint"]  # while Quad Size or Rows moves: what it changed
    else:
        model["hint"] = next_step_hint(obj, stats, settings)
    # A mesh CurioMesh has not touched yet is the starting point, not a failure: no red, no NEEDS WORK
    model["original"] = is_mesh and not showing_result and not is_preview(obj, REGION) and "curiomesh_engine" not in obj.keys()
    return model


def _text_right(xr, y, size, color, text):
    import blf

    blf.size(0, size)
    width = blf.dimensions(0, tr(text))[0]
    return _text(xr - width, y, size, color, text)


def _draw_hud() -> None:
    try:
        import gpu

        context = bpy.context
        settings = _settings()
        if settings is None or not settings.hud_enabled:
            return
        prefs = get_prefs()
        model = hud_model(context)
        job = model["job"]
        collapsed = bool(settings.hud_collapsed)
        ui = _ui_scale()
        s = ui * float(prefs.hud_scale)
        accent = (0.45, 0.55, 1.0) if model["original"] else accent_for_ratio(model["ratio"])
        if job is not None:
            accent = (0.35, 0.7, 1.0)
        w = 330 * s
        h = (HUD_STRIP_H if collapsed else HUD_FULL_H + (HUD_JOB_EXTRA if job is not None else 0)) * s
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
        top = y + h - (pad if not collapsed else 15 * s)
        mid = top - 6 * s
        bx = right
        _text(left, top - 10 * s, 11 * s, (*accent, 1.0), "CURIOMESH")
        if job is not None:  # the alive dot: it pulses, so a slow step never looks frozen
            pulse = 0.35 + 0.65 * abs(2.0 * job["alive_tick"] - 1.0)
            dot = 4 * s
            shader.bind()
            _fill(shader, _rounded_rect(bx - dot * 1.5, mid - dot, dot * 2, dot * 2, dot, 4), (0.4, 0.95, 0.6, pulse))
        if collapsed:
            tail = f"{job['frac'] * 100:.0f}%  {job['stage']}" if job is not None else (f"{model['ratio'] * 100:.0f}% quads" if model["ratio"] is not None else "")
            room = max(8, int((bx - 2 * s * 4 - left - 82 * s) / (5.4 * s)))
            _text(left + 82 * s, top - 10 * s, 10 * s, muted, clip((model["name"][:10] + "  " if model["name"] else "") + tr(tail), room))
            gpu.state.blend_set("NONE")
            return
        _text(left + 95 * s, top - 10 * s, 11 * s, muted, model["name"][:8] or "no mesh")

        ratio = model["ratio"]
        if job is not None:
            big = f"{job['frac'] * 100:.0f}%"
            bar_ratio, bar_colour, big_word = job["frac"], (0.35, 0.7, 1.0), "done"
        else:
            big = f"{ratio * 100:.1f}%" if ratio is not None else "--"
            bar_ratio, bar_colour, big_word = ratio, accent, "quads"
        bw = _text(left, top - 52 * s, 32 * s, (1, 1, 1, 1), big)
        _text(left + bw + 8 * s, top - 52 * s, 11 * s, muted, big_word)
        _text_right(right, top - 52 * s, 10 * s, (*accent, 1.0), "WORKING" if job is not None else "ORIGINAL" if model["original"] else quality_word(model["ratio"]))

        bar_x, bar_y, bar_w, bar_h = left, top - 66 * s, w - 2 * pad - 6 * s, 7 * s
        shader.bind()
        _fill(shader, _rounded_rect(bar_x, bar_y, bar_w, bar_h, 3.5 * s, 3), (0.2, 0.22, 0.28, 1.0))
        if bar_ratio:
            fill_w = max(bar_h, bar_w * min(1.0, bar_ratio))
            _fill(shader, _rounded_rect(bar_x, bar_y, fill_w, bar_h, 3.5 * s, 3), (*bar_colour[:3], 1.0))
        if job is None:  # goal tick at 95 % quads
            tick_x = bar_x + bar_w * 0.95
            _fill(shader, [(tick_x, bar_y - 2 * s), (tick_x + 1.5 * s, bar_y - 2 * s), (tick_x + 1.5 * s, bar_y + bar_h + 2 * s), (tick_x, bar_y + bar_h + 2 * s)], (0.85, 0.87, 0.95, 0.9))

        col_w = (w - 2 * pad - 6 * s) / 3
        cells = (
            ("FACES", f"{model['faces']:,} / {model['target']:,}" if model["target"] else f"{model['faces']:,}", (1, 1, 1, 1)),
            ("TRIS", f"! {model['tris']:,}" if model["tris"] else "0", (1.0, 0.5, 0.5, 1) if model["tris"] else (0.6, 0.9, 0.7, 1)),
            ("POLES", f"{model['poles']:,}", (1.0, 0.78, 0.3, 1)),
        )
        for (cell_label, value, color), cx in zip(cells, (left, left + col_w * 1.5, left + col_w * 2.25)):  # FACES holds "17,456 / 8,000", so it gets a wider column
            _text(cx, top - 90 * s, 10 * s, label, cell_label)
            _text(cx, top - 106 * s, 14 * s, color, value)

        if job is not None:
            _text(left, top - 124 * s, 10 * s, (0.5, 0.8, 1.0, 1.0), clip(tr(job_text(job)), 56))
            _text(left, top - 144 * s, 12 * s, (*accent, 1.0), "> " + clip(tr(job["stage"]), 44))
            _text(left, top - 164 * s, 10 * s, label, f"Quads so far: {ratio * 100:.1f}%" if ratio is not None else "Quads so far: --")
        else:
            if model["batch"]:
                _text(left, top - 124 * s, 10 * s, (0.5, 0.8, 1.0, 1.0), clip(tr(model["batch"]), 46))
            elif model["info"]:
                _text(left, top - 124 * s, 10 * s, label, clip(tr(model["info"]), 46))
            _text(left, top - 144 * s, 12 * s, (*accent, 1.0), "> " + clip(tr(model["hint"]), 44))
        gpu.state.blend_set("NONE")
    except Exception:
        draw_failed("overlay._draw_hud")


# --------------------------------------------------------------------------- operators


class CURIOMESH_OT_toggle_heatmap(bpy.types.Operator):
    bl_idname = "curiomesh.toggle_heatmap"
    bl_label = "Show Quality Heatmap"
    bl_description = "Colour the mesh: triangles red, n-gons purple, 3-edge poles blue dots, 5+-edge poles larger orange dots (colours editable in Preferences)"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        settings = context.scene.curiomesh_settings
        settings.heatmap_enabled = not settings.heatmap_enabled
        ensure_watch()
        obj = context.object
        if settings.heatmap_enabled and obj is not None and obj.type == "MESH":
            stats = get_stats(obj)
            if stats is not None and stats.faces > HEATMAP_MAX_FACES:
                self.report({"WARNING"}, tr(f"Heatmap skipped: over {HEATMAP_MAX_FACES:,} faces."))
        return {"FINISHED"}


class CURIOMESH_OT_hud_toggle(bpy.types.Operator):
    bl_idname = "curiomesh.hud_toggle"
    bl_label = "Show Dashboard"
    bl_description = "Show or hide the dashboard in the 3D view. It stays off until you turn it on"
    bl_options = {"INTERNAL"}

    @classmethod
    def description(cls, context, properties):
        on = getattr(getattr(context.scene, "curiomesh_settings", None), "hud_enabled", False)
        return tip("Hide the dashboard from the 3D view" if on else "Show the dashboard (quad quality and job progress) in the 3D view")

    def execute(self, context):
        settings = context.scene.curiomesh_settings
        settings.hud_enabled = not settings.hud_enabled
        tag_redraw()
        return {"FINISHED"}


CLASSES = (CURIOMESH_OT_toggle_heatmap, CURIOMESH_OT_hud_toggle)
WATCH_SECONDS = 0.5


def _watched(settings) -> bool:
    return settings is not None and bool(settings.hud_enabled or settings.heatmap_enabled or settings.live_enabled)


def _watch():
    """Timer: measures what the HUD and heatmap show, so their draw callbacks only draw. Ends when both (and Live) are off.
    Meshes in Edit or Sculpt Mode are skipped: their stats wait until you are back in Object Mode."""
    settings = _settings()
    if not _watched(settings):
        return None
    context = SimpleNamespace(scene=bpy.context.scene, object=bpy.context.view_layer.objects.active)
    heat = heatmap_targets(context, settings)
    shown = heat + [hud_object(context)[0]] if settings.hud_enabled else heat
    colors = heat_colors()
    changed = False
    for obj in dict.fromkeys(o for o in shown if o is not None and o.type == "MESH"):
        if obj.mode != "OBJECT" or len(obj.data.polygons) > HEATMAP_MAX_FACES:
            continue
        old = _stats_cache.get(obj.name)
        entry = _entry(obj)
        changed |= entry is not old
        if obj in heat and (entry.heat_arrays is None or entry.heat_arrays[0] != colors):
            entry.heat_arrays = (colors, build_heat_arrays(obj, entry.stats, colors))
            changed = True
    if changed:
        tag_redraw()
    return WATCH_SECONDS


def ensure_watch(_self=None, _context=None) -> None:
    """Start the watch timer when the HUD, Heatmap or Live is on (their switches, and a loaded file whose scene has one
    on). Nothing runs while all three are off. Usable as an update callback."""
    try:
        settings = _settings()
    except AttributeError:  # no scene yet
        return
    if _watched(settings) and not bpy.app.timers.is_registered(_watch):
        bpy.app.timers.register(_watch, first_interval=0.0, persistent=True)


def register() -> None:
    unregister()
    _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_heatmap, (), "WINDOW", "POST_VIEW"))
    _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_hud, (), "WINDOW", "POST_PIXEL"))


def unregister() -> None:
    while _draw_handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handles.pop(), "WINDOW")
        except Exception:
            pass
    for timer in (_watch, _build_pending, _hud_tick):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    _built.clear()
    _pending.clear()
    _stats_cache.clear()
    _draw_errors.clear()
