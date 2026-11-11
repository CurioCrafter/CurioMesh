# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""You can always go back: Ctrl+Z, Ctrl+Shift+Z and the History list for what CurioMesh keeps outside Blender.

Blender's undo brings back the scene and the preview object, but not the results the add-on keeps in Python (Tidy Up
passes, the size ladder, tool results). So every step that changes them gets a number on the scene
(`scene.curiomesh_tools.step`), which Blender's undo restores, and `triadq.history.Steps` keeps what was on screen
under that number. After an undo or redo, `sync` puts back the entry of the number Blender brought back.

How a change records itself:
- An operator with no UNDO flag (Tidy Again, Keep Tidying from its timer, Undo Pass, a History jump):
  `before(context)`, change things, `record(context, name, push=True)`.
- An operator with {"REGISTER", "UNDO"} (every Tidy Tool's Apply): the same with `push=False`; Blender pushes.
- A slider or switch: `note(context, kind, name)` from its update function, before the change shows (Blender pushes
  when the mouse lets go), then `settled(kind, numbers)` once the result on screen is the one it asked for.
"""

from __future__ import annotations

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import overlay, profiles
from .triadq import history as H
from .triadq import uvbake
from .translations import tr

LOW_GB = 1.5  # free memory under this: the oldest half of the steps is let go, and the panel says so
ORG_KEYS = ("base", "origin", "report", "levels", "shown", "seg_piece", "passes", "note")
LAD_KEYS = ("key", "base", "rungs", "plan", "job", "doing", "shown", "failed", "want", "settings")
SEG_KEYS = ("base", "piece", "parts", "sel", "ref", "whole", "src", "tree", "tree_job", "res_src", "history")
SLIDERS = ("quad_size", "quad_order", "organize")
BAKE_KEY = "curio_uv_step"  # triadq.uvbake.HISTORY_KEY: its image states follow this scene counter

steps = H.Steps()


def _g():
    from . import guided  # late: guided imports modules that import this one

    return guided


def settings(scene=None):
    scene = scene or bpy.context.scene
    return getattr(scene, "curiomesh_tools", None)


def current(scene=None) -> int:
    ts = settings(scene)
    return int(ts.step) if ts is not None else 0


def _set_step(scene, step: int) -> None:
    ts = settings(scene)
    if ts is not None:
        ts["step"] = step  # no update function: nothing else runs


def snapshot(context=None) -> dict:
    """References to everything that says what is on screen: the result, Tidy Up's passes, the ladder, the parts, the
    strip picked with Face Loops."""
    from . import faceloops, ladder, organize, segments

    guided = _g()
    base = guided.state["auto_base"]
    gs = getattr((context or bpy.context).scene, "curiomesh_guided", None)
    return {
        "base": base,
        "out": None if base is None else base.get("out"),
        "id": None if base is None else base.get("id"),  # the parts table is keyed by it: an undone part redo keeps its table
        "report": None if base is None else base.get("report"),
        "res": guided.state["res"],
        "org": {k: list(organize.org[k]) if k == "passes" else organize.org[k] for k in ORG_KEYS},
        "lad": {k: ladder.lad.get(k) for k in LAD_KEYS},
        "seg": {k: _seg_copy(k, v) for k, v in ((k, segments.seg.get(k)) for k in SEG_KEYS)},
        "sliders": None if gs is None else {k: float(getattr(gs, k)) for k in SLIDERS},
        "curve_sig": guided.state.get("curve_sig"),  # the search's inputs for this result: an undone Tweak does not restart it
        "strip_sel": faceloops.fl["sel"],  # (result, element): only shown while that result is on screen
        "bake_step": int((context or bpy.context).scene.get(BAKE_KEY, 0)),  # which pixels baked images had (triadq.uvbake)
    }


def _seg_copy(key, value):
    """A copy later changes cannot reach. Part records are changed in place (which try is shown, its face count), so
    each one is copied too; the tries themselves are only added to, so they are shared."""
    if key == "parts" and isinstance(value, list):
        return [dict(r, tries=list(r["tries"])) if isinstance(r, dict) and "tries" in r else r for r in value]
    return list(value) if isinstance(value, list) else set(value) if isinstance(value, set) else value


def restore(context, snap: dict, sliders: bool = False) -> bool:
    """Put a snapshot back on screen. `sliders`: also set the sliders to what they were (a History jump; after an undo
    Blender has already put them back). True when the result on screen changed."""
    from . import faceloops, ladder, organize, segments

    guided = _g()
    shown = guided.state["res"]
    base = snap["base"]
    if base is not None:
        base["out"], base["report"] = snap["out"], snap["report"]
        if snap.get("id") is not None:
            base["id"] = snap["id"]
    guided.state["auto_base"] = base
    guided.state["res"] = snap["res"]
    org = snap["org"]
    if organize.org["base"] is not org["base"] or organize.org["levels"] is not org["levels"]:
        organize._drop_work()  # a pass still being worked out belongs to another step
    organize.org.update({k: list(v) if k == "passes" else v for k, v in org.items()}, keep=False)
    ladder.lad.update(snap["lad"])
    new_parts = segments.seg.get("parts") is not snap["seg"]["parts"]
    segments.seg.update({k: _seg_copy(k, v) for k, v in snap["seg"].items()})
    if "curve_sig" in snap:
        guided.state["curve_sig"] = snap["curve_sig"]
    faceloops.fl.update(sel=snap.get("strip_sel"), live=None, hover=None)
    segments.seg.update(batch_key=None, bad_key=None, tris=None, hover=None, lines_key=None, outline_key=None, high_key=None)
    if new_parts:
        segments.refresh_list()
    gs = getattr(context.scene, "curiomesh_guided", None)
    if sliders and gs is not None and snap["sliders"]:
        for k, v in snap["sliders"].items():
            gs[k] = v  # no update: the snapshot already holds what they show
    changed = snap["res"] is not shown
    source = guided.state_source()
    if changed and snap["res"] is not None and source is not None:
        guided._build_preview(context, source, snap["res"], uvs=False, mesh=ladder._make_mesh(guided.PREVIEW_NAME + "_Mesh", snap["res"], source))
    guided.state["draw_key"] = None
    overlay.tag_redraw()
    return changed


def _resync(context) -> None:
    """The sliders were put back by Blender: show what they ask for (a drag undone before its result settled)."""
    from . import ladder, organize

    gs = getattr(context.scene, "curiomesh_guided", None)
    if gs is None:
        return
    if ladder.active() and ladder.preview(context):
        ladder.settle(context)
    if organize.active():
        organize.show(context, gs)


def _push(name: str) -> bool:
    """One Blender undo step, also from a timer (which has no window in its context)."""
    try:
        bpy.ops.ed.undo_push(message=name)
        return True
    except Exception:
        pass
    wm = bpy.context.window_manager
    for window in getattr(wm, "windows", ()):
        try:
            with bpy.context.temp_override(window=window):
                bpy.ops.ed.undo_push(message=name)
            return True
        except Exception:
            continue
    return False


def _mind_memory() -> None:
    if profiles.free_memory_gb() < LOW_GB and len(steps.entries) > 1:
        steps.let_go_oldest(steps.size() // 2)


def before(context) -> None:
    """Make sure the step on screen has an entry (the state as it is now), before something changes it."""
    step = current(context.scene)
    if steps.at(step) is None:
        steps.add(step - 1, "Before", "", snapshot(context))


def record(context, name: str, numbers: str = "", push: bool = False, kind: str = "") -> int:
    """Keep what is on screen now as a new step and return its number. `push`: also push Blender's undo step (for
    operators without the UNDO flag and for timers)."""
    step = steps.add(current(context.scene), name, numbers, snapshot(context), kind)
    _set_step(context.scene, step)
    _mind_memory()
    if push:
        _push(name)
    return step


def note(context, kind: str, name: str) -> None:
    """A slider or switch moved: a step whose result is not on screen yet (`settled` fills it in). One drag is one
    step. Blender pushes its own undo step for the move."""
    before(context)
    step = steps.add(current(context.scene), name, "", None, kind)
    _set_step(context.scene, step)


def settled(kind: str, numbers: str = "") -> bool:
    """The result a slider asked for is on screen: keep it in that slider's step. False when the step on screen is not
    one of that kind (nothing was kept)."""
    if steps.entries and steps.entries[-1].kind == kind and steps.entries[-1].step == current():
        entry = steps.entries[-1]
        entry.state, entry.numbers = snapshot(), numbers or entry.numbers
        entry.size = H.size_of(entry.state, steps._seen)
        return True
    return False


def refresh(context) -> None:
    """Something finished in the background on the step on screen (the whole-result check after a part swap): its
    entry keeps what is shown now, so Ctrl+Z and Ctrl+Shift+Z come back to the checked result."""
    if steps.entries and steps.entries[-1].step == current(context.scene) and steps.entries[-1].state is not None:
        entry = steps.entries[-1]
        entry.state = snapshot(context)
        entry.size = H.size_of(entry.state, steps._seen)


def _state_for(step: int):
    """(snapshot, settled) for this step: a step still waiting for its slider takes the one before it."""
    i = steps.index_of(step)
    while i >= 0 and steps.entries[i].state is None:
        i -= 1
    return (steps.entries[i].state if i >= 0 else None), steps.at(step) is not None and steps.at(step).state is not None


def sync(context) -> bool:
    """After an undo or redo: put back what CurioMesh had at the step Blender brought back. True when it did."""
    from . import tidylive

    tidylive.close("The result changed, so this preview was closed. Open the tool again.")
    step = current(context.scene)
    if steps.at(step) is None:
        return False
    snap, settled_ = _state_for(step)
    if snap is None:
        return False
    gs = getattr(context.scene, "curiomesh_guided", None)
    if settled_ and gs is not None and snap.get("sliders") and any(abs(float(getattr(gs, k)) - v) > 1e-9 for k, v in snap["sliders"].items()):
        settled_ = False  # two drags within COALESCE share one step: Blender brought back the first drag's value, so show what it asks for
    restore(context, snap)
    if not settled_:
        _resync(context)
    else:
        from . import organize

        gs = getattr(context.scene, "curiomesh_guided", None)
        if gs is not None and organize.active():
            organize.show(context, gs)
    return True


def _read_uvs(mesh) -> dict:
    """Every UV layer of `mesh` as float32 corner arrays, and which one is active: what a History jump writes back."""
    layers = {}
    for layer in mesh.uv_layers:
        uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        layer.data.foreach_get("uv", uv)
        layers[layer.name] = uv
    return {"loops": len(mesh.loops), "active": mesh.uv_layers.active_index, "layers": layers}


def _write_uvs(mesh, kept: dict) -> bool:
    """Put `_read_uvs`'s layers back on `mesh` (made, removed or overwritten to match). False when the mesh changed shape."""
    if len(mesh.loops) != kept["loops"]:
        return False
    for name in [layer.name for layer in mesh.uv_layers if layer.name not in kept["layers"]]:
        mesh.uv_layers.remove(mesh.uv_layers[name])
    for name, uv in kept["layers"].items():
        layer = mesh.uv_layers.get(name) or mesh.uv_layers.new(name=name)
        layer.data.foreach_set("uv", uv)
    if 0 <= kept["active"] < len(mesh.uv_layers):
        mesh.uv_layers.active_index = kept["active"]
    mesh.update()
    return True


def keep_uvs(context, obj) -> None:
    """A UV tool writes `obj`'s UVs (or bakes into its images): the step on screen keeps what the mesh and images have
    now, so a History jump to it brings them back. Call it before the write, and again after `record`."""
    before(context)
    entry = steps.at(current(context.scene))
    if entry is None or not isinstance(entry.state, dict) or obj is None or obj.type != "MESH":
        return
    uvs = dict(entry.state.get("uvs") or {})
    if obj.data.name not in uvs:
        uvs[obj.data.name] = _read_uvs(obj.data)
    entry.state = dict(entry.state, uvs=uvs, bake_step=int(context.scene.get(BAKE_KEY, 0)))
    entry.size += H.size_of(uvs, steps._seen)


def _uvs_at(index: int) -> dict:
    """The UVs each tracked mesh had at entry `index`: the newest kept at or before it, else the oldest kept after it
    (what the mesh had before a UV tool first changed it)."""
    out: dict = {}
    for i, entry in enumerate(steps.entries):
        kept = entry.state.get("uvs") if isinstance(entry.state, dict) else None
        for name, uvs in (kept or {}).items():
            if i <= index or name not in out:
                out[name] = uvs
    return out


def jump(context, index: int) -> bool:
    """Go back (or forward) to History entry `index`, as a new step, so Ctrl+Z undoes the jump. The meshes' UVs and
    the baked images' pixels a UV tool changed come back too."""
    if not 0 <= index < len(steps.entries):
        return False
    before(context)
    entry = steps.entries[index]
    snap, _ = _state_for(entry.step)
    if snap is None:
        return False
    restore(context, snap, sliders=True)
    uvs = _uvs_at(index)
    for name, kept in uvs.items():
        mesh = bpy.data.meshes.get(name)
        if mesh is not None:
            _write_uvs(mesh, kept)
    uvbake.bring_back(context.scene, snap.get("bake_step", 0))
    record(context, f"Back to: {entry.name}", entry.numbers, push=True)
    if uvs:
        new = steps.entries[-1]
        new.state = dict(new.state, uvs=uvs)  # the same arrays: later jumps read this step's UVs from here
    return True


@persistent
def _curiomesh_history_undo(*_args):
    try:
        sync(bpy.context)
    except Exception:  # a handler must never stop Blender's undo
        overlay.draw_failed("history._curiomesh_history_undo")


@persistent
def _curiomesh_history_load(*_args):
    from . import tidylive

    tidylive.close("")
    steps.clear()


# --------------------------------------------------------------------------- panel


def draw(layout, context) -> None:
    """Every step of this result, newest first; click one to go back to it."""
    ts = settings(context.scene)
    if not steps.entries:
        sub = layout.row()
        sub.active = False
        sub.label(text="Steps show here as you tidy, adjust and use tools", icon="INFO")
        return
    step = current(context.scene)
    at = steps.index_of(step)
    shown = range(len(steps.entries) - 1, -1, -1)
    if ts is not None and not ts.history_all:
        shown = list(shown)[:12]
    col = layout.column(align=True)
    for i in shown:
        entry = steps.entries[i]
        row = col.row(align=True)
        row.active = entry.step <= step  # undone steps are grey until something new replaces them
        text = entry.name + (f": {entry.numbers}" if entry.numbers else "")
        row.operator("curiomesh.history_jump", text=tr(text), depress=i == at, icon="RADIOBUT_ON" if i == at else "BLANK1").index = i
    if ts is not None and len(steps.entries) > 12:
        layout.prop(ts, "history_all", text=tr(f"Show all {len(steps.entries)} steps"), toggle=True)
    if steps.let_go:
        overlay.wrap_label(layout, (f"{steps.let_go} oldest steps were let go to free memory" if steps.let_go != 1 else "The oldest step was let go to free memory"), icon="INFO", width=36, limit=2)


class CURIOMESH_OT_history_jump(bpy.types.Operator):
    bl_idname = "curiomesh.history_jump"
    bl_label = "Go to Step"
    bl_description = "Puts the result back as it was at this step. Ctrl+Z takes the jump back again"
    bl_options = {"INTERNAL"}

    index: bpy.props.IntProperty(name="Step", description="Which History entry to go to", default=-1, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        from . import tidylive

        return not tidylive.busy()

    def execute(self, context):
        return {"FINISHED"} if jump(context, self.index) else {"CANCELLED"}


class CURIOMESH_PG_tools(bpy.types.PropertyGroup):
    step: bpy.props.IntProperty(name="Step", description="Which History step is on screen. Blender's undo puts it back", default=0, options={"HIDDEN"})
    history_all: bpy.props.BoolProperty(name="Show All Steps", description="Shows every step in History, not only the newest twelve. Long lists take more room", default=False)
    tool_scope: bpy.props.EnumProperty(
        name="Work on",
        description="Which faces the tools may change: your Edit Mode selection, the picked part, or the whole result. The edge of that area never moves",
        items=(
            ("SELECTION", "Selection", "Only the faces of the result you selected in Edit Mode. Use it for one patch"),
            ("PART", "Part", "Only the part picked in the Fix list or the viewport. Use it to tidy one piece at a time"),
            ("WHOLE", "Whole Result", "Every face of the result. Slower on big results; you still see the change before keeping it"),
        ),
        default="WHOLE",
    )
    last_tool: bpy.props.StringProperty(name="Last Tool", description="The operator of the last tool you kept", default="")
    last_name: bpy.props.StringProperty(name="Last Tool Name", description="The name of the last tool you kept", default="")
    last_numbers: bpy.props.StringProperty(name="Last Numbers", description="What the last tool you kept changed", default="")
    last_settings: bpy.props.StringProperty(name="Last Settings", description="The settings the last tool was kept with", default="")


CLASSES = (CURIOMESH_PG_tools, CURIOMESH_OT_history_jump)
HANDLERS = ((bpy.app.handlers.undo_post, _curiomesh_history_undo), (bpy.app.handlers.redo_post, _curiomesh_history_undo), (bpy.app.handlers.load_post, _curiomesh_history_load))


def _drop_handlers() -> None:
    """Remove ours, also copies left by an earlier load of this module (a reload makes new function objects)."""
    for handlers, fn in HANDLERS:
        for old in [h for h in handlers if getattr(h, "__name__", "") == fn.__name__]:
            handlers.remove(old)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_tools = bpy.props.PointerProperty(type=CURIOMESH_PG_tools)
    _drop_handlers()
    for handlers, fn in HANDLERS:
        handlers.append(fn)


def unregister() -> None:
    _drop_handlers()
    steps.clear()
    if hasattr(bpy.types.Scene, "curiomesh_tools"):
        del bpy.types.Scene.curiomesh_tools
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
