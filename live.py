"""Live preview: debounced TRIAD-Q (PairClosure) re-run into one reusable preview object."""

from __future__ import annotations

import time
from dataclasses import replace

import bpy
import numpy as np

from . import overlay
from .bridge import run_curiomesh
from .operators import config_from_owner

LIVE_FACE_CAP = 8_000
DEBOUNCE_SECONDS = 0.4

state = {"source": "", "sig": None, "message": "", "faces": 0, "ms": 0.0}


def status() -> str:
    return state["message"]


def _signature(obj: bpy.types.Object) -> tuple:
    mesh = obj.data
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    return (len(mesh.vertices), len(mesh.polygons), round(float(co.sum()), 4))


def _is_preview(obj) -> bool:
    return obj is not None and obj.name == overlay.PREVIEW_NAME


def remove_preview() -> None:
    obj = bpy.data.objects.get(overlay.PREVIEW_NAME)
    if obj is None:
        return
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def _pick_source(context) -> bpy.types.Object | None:
    obj = context.object
    if obj is not None and obj.type == "MESH" and not _is_preview(obj):
        state["source"] = obj.name
    return bpy.data.objects.get(state["source"])


def schedule(delay: float = DEBOUNCE_SECONDS) -> None:
    """(Re)start the debounce timer; later calls within `delay` replace earlier ones."""
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    bpy.app.timers.register(_tick, first_interval=delay)


def on_setting_changed(settings, context=None) -> None:
    """Property update callback for the Live toggle and the settings that drive the preview."""
    if settings.live_enabled:
        schedule()


def on_live_toggled(settings, context) -> None:
    if settings.live_enabled:
        _pick_source(context)
        state["sig"] = None
        schedule(0.05)
    else:
        if bpy.app.timers.is_registered(_tick):
            bpy.app.timers.unregister(_tick)
        remove_preview()
        state["message"] = ""
    overlay.tag_redraw()


def _on_geometry(name: str) -> None:
    settings = bpy.context.scene.curiomesh_settings
    if not settings.live_enabled or name != state["source"]:
        return
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "MESH" or obj.mode != "OBJECT":
        return
    if _signature(obj) != state["sig"]:
        schedule()


def run_preview(context) -> bool:
    """Synchronous preview rebuild. Returns True when a preview was produced."""
    settings = context.scene.curiomesh_settings
    source = _pick_source(context)
    if source is None or source.type != "MESH":
        state["message"] = "Live: select a mesh"
        return False
    faces = len(source.data.polygons)
    if faces > LIVE_FACE_CAP:
        remove_preview()
        state["message"] = f"Live off for this mesh: {faces:,} faces (limit {LIVE_FACE_CAP:,})"
        return False
    if source.mode != "OBJECT":
        state["message"] = "Live: exit Edit Mode"
        return False

    config = replace(
        config_from_owner(settings),
        engine="TRIAD_Q_LITE",
        output_mode="NEW",
        texture_mode="NONE",
        bake_fallback=False,
        keep_debug_objects=False,
        debug_logs=False,
    )
    view_layer = context.view_layer
    selected = [o for o in view_layer.objects if o.select_get()]
    active = view_layer.objects.active
    started = time.perf_counter()
    result = run_curiomesh(source, config)
    # Restore the user's selection: the pipeline selects its output.
    for o in view_layer.objects:
        o.select_set(o in selected)
    view_layer.objects.active = active

    out = result.output_object
    if not result.success or out is None or out is source:
        if out is not None and out is not source:
            bpy.data.objects.remove(out, do_unlink=True)
        state["message"] = f"Live: {result.message or 'preview failed'}"[:90]
        return False

    remove_preview()
    out.name = overlay.PREVIEW_NAME
    out.data.name = overlay.PREVIEW_NAME + "_Mesh"
    out.select_set(False)
    out.location = source.location.copy()
    out.location.x += max(source.dimensions.x, 0.1) * 1.25
    state["sig"] = _signature(source)
    state["faces"] = len(out.data.polygons)
    state["ms"] = (time.perf_counter() - started) * 1000.0
    state["message"] = (
        f"Preview: {state['faces']:,} faces, {result.diagnostics.quad_ratio * 100:.0f}% quads, "
        f"{state['ms'] / 1000.0:.1f} s"
    )
    return True


def _tick():
    try:
        context = bpy.context
        settings = context.scene.curiomesh_settings
        if settings.live_enabled:
            run_preview(context)
    except Exception as exc:
        state["message"] = f"Live error: {exc}"[:90]
    overlay.tag_redraw()
    return None


def register() -> None:
    overlay.add_geometry_hook(_on_geometry)


def unregister() -> None:
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    overlay.remove_geometry_hook(_on_geometry)
    try:
        remove_preview()
    except Exception:
        pass
    state.update(source="", sig=None, message="")
