# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Live segments: each mesh's own segment settings, and a coloured preview that follows them while you drag.

A change asks for a preview. When the asking has been quiet for a moment (or every tenth of a second during a long
drag), a task (`triadq.tasks`) cuts the merge tree kept from the last find at the new settings (Crease Strength merges it again
first) and marks the result's faces through a map of the nearest face of your mesh, worked out once per result. That
is all a preview needs, so it lands in well under a second; one preview runs at a time, and the newest change is
always the next one shown. When the changes have stopped for longer,
`FindJob` finds the segments for real (borders settled on the creases, seams laid on the result) and keeps them.

Paint Size is the `CM_SegmentSize` vertex group: red makes smaller segments there, blue larger. While your mesh is in
Weight Paint, a timer compares a sample of its paint every quarter second; a stroke asks for a preview after a longer
pause so painting stays smooth. The timer starts from the viewport's redraw and ends when you leave Weight Paint.
"""

from __future__ import annotations

import time

import bmesh
import bpy
import numpy as np

from . import overlay, parallel
from . import segments as view
from .triadq import segment as S
from .triadq import tasks

SIZE_GROUP = "CM_SegmentSize"
PREVIEW_WAIT = 0.03  # quiet before a preview: a drag sends many changes a second
PREVIEW_MOST = 0.1  # a drag that never pauses still gets a preview this often
PAINT_WAIT = 0.3  # after a stroke: reading the paint of a big mesh takes a moment, so not after every dab
KEEP_WAIT = 0.8  # quiet before the segments are found for real and kept
TICK = 0.02
PAINT_POLL = 0.25  # how often the paint is compared while in Weight Paint
PAINT_SAMPLES = 20_000  # corners whose paint is compared: enough for any stroke to touch some

live = {
    "preview": parallel.Debounce(PREVIEW_WAIT, PREVIEW_MOST),
    "keep": parallel.Debounce(KEEP_WAIT),
    "task": None,
    "paint_sig": None,  # the sampled paint last seen in Weight Paint
    "box": None,
    "poked": 0.0,  # when the newest change came
    "ms": None,  # from the last change to its preview on screen
    "count": None,
    "paint": None,  # (key, per face of your mesh)
    "paint_serial": 0,
    "error": "",
    "shown": 0,  # ticket of the preview on screen
}


def _changed(self, context) -> None:
    request()


class CURIOMESH_PG_segsettings(bpy.types.PropertyGroup):
    """Kept on each mesh, so every mesh has its own segments."""

    detail: bpy.props.FloatProperty(name="Part Detail", description="How many parts the mesh is cut into. Left finds only the big parts, right also the small ones; locked parts stay", default=0.65, min=0.0, max=1.0, subtype="FACTOR", update=_changed)
    largest: bpy.props.FloatProperty(name="Largest Part", description="No part grows past this share of its loose piece. 100% leaves the size to Part Detail", default=100.0, min=2.0, max=100.0, subtype="PERCENTAGE", update=_changed)
    smallest: bpy.props.FloatProperty(name="Smallest Part", description="A part smaller than this share of its loose piece joins its neighbour", default=0.4, min=0.0, max=10.0, subtype="PERCENTAGE", update=_changed)
    creases: bpy.props.FloatProperty(name="Crease Strength", description="How much sharp edges split the mesh into parts. More: every crease becomes a part border", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=_changed)
    smooth: bpy.props.FloatProperty(name="Border Smoothness", description="How much part borders are rounded off where no crease holds them. Right: smoother, rounder borders", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=_changed)
    size_paint: bpy.props.BoolProperty(name="Paint Size", description=f"Use the '{SIZE_GROUP}' vertex group: red makes smaller parts there, blue larger ones, green leaves it to the sliders", default=False, update=_changed)


def snapshot(source) -> dict:
    """The settings of this mesh, as plain numbers a thread can use."""
    st = source.curiomesh_seg
    return {"detail": st.detail, "largest": st.largest / 100.0, "smallest": st.smallest / 100.0, "creases": S.crease_factor(st.creases), "smooth": st.smooth, "paint": st.size_paint}


def paint_for(source, src) -> np.ndarray | None:
    """Per face of your mesh: its Paint Size (0.5 where unpainted), or None when Paint Size is off or has no group."""
    group = source.vertex_groups.get(SIZE_GROUP)
    if not source.curiomesh_seg.size_paint or group is None:
        return None
    key = (src["sig"], live["paint_serial"], group.index)
    if live["paint"] is not None and live["paint"][0] == key:
        return live["paint"][1]
    bm = bmesh.new()
    try:
        bm.from_mesh(source.data)
        layer = bm.verts.layers.deform.active
        if layer is None:
            return None
        index = group.index
        weight = np.fromiter((v[layer].get(index, 0.5) for v in bm.verts), dtype=np.float64, count=len(bm.verts))
    finally:
        bm.free()
    total = len(src["co"])
    welded = np.bincount(src["weld"], weight, total) / np.maximum(np.bincount(src["weld"], minlength=total), 1)
    faces = S.face_paint(src["faces"], welded)
    live["paint"] = (key, faces)
    return faces


def request(paint: bool = False) -> None:
    """A setting or the paint changed: preview it soon, keep it a little later."""
    if not view.ready() or view.seg.get("src") is None:
        return
    now = time.perf_counter()
    if paint:
        live["paint_serial"] += 1
    live["preview"].wait = PAINT_WAIT if paint else PREVIEW_WAIT
    live["preview"].most = None if paint else PREVIEW_MOST  # not mid-stroke: reading the paint would make painting stutter
    live["poked"] = now
    live["preview"].poke(now)
    live["keep"].poke(now)
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.0)


def res_map() -> np.ndarray | None:
    """Per face of the result on screen: the face of your mesh nearest to it. Worked out once per result."""
    guided = view._g()
    base, source = guided.state["auto_base"], guided.state_source()
    if base is None or source is None:
        return None
    faces = base["out"]["faces"]
    held = view.seg.get("res_src")
    if held is not None and held[0] is faces:
        return held[1]
    src = guided._source_data(source)
    found = view.segment_goals.nearest_faces(src, guided._bvh(src), base["out"]["co"], faces)
    view.seg["res_src"] = (faces, found)
    return found


def _start() -> None:
    guided = view._g()
    source, base = guided.state_source(), guided.state["auto_base"]
    tree, old = view.seg.get("tree"), view.seg.get("src")
    ticket = live["preview"].take()
    if source is None or base is None or tree is None or old is None or not view.ready():
        return
    src = guided._source_data(source)
    if len(src["faces"]) != len(old):
        return
    settings = snapshot(source)
    paint = paint_for(source, src) if settings["paint"] else None
    mapping = res_map()
    locked = {r["id"] for r in view.seg["parts"] if r.get("locked")}
    box = {"ticket": ticket, "asked": live["poked"], "base": base, "faces": base["out"]["faces"]}

    def work():
        try:
            cut_tree = tree
            if abs(cut_tree.creases - settings["creases"]) > 1e-9:
                cut_tree = box["tree"] = S.remerge(tree, settings["creases"])
            scale = S.patch_scale(cut_tree, paint) if paint is not None else None
            labels = S.cut(cut_tree, settings["detail"], scale, settings["largest"], settings["smallest"])[cut_tree.leaf]
            labels = S.keep_locked(labels, old, locked, None, None, tidy_up=False)
            values, labels = np.unique(labels, return_inverse=True)
            box.update(count=len(values), res=labels.reshape(-1)[mapping])
        except Exception as exc:  # a thread has nobody above it to report to
            if str(exc) != "cancelled":
                box["error"] = str(exc)[:100]

    live["box"] = box
    live["task"] = tasks.start(work, "CurioMeshLiveSegments")


def _finished(box) -> None:
    guided = view._g()
    base = guided.state["auto_base"]
    if box.get("tree") is not None and base is box["base"]:
        view.seg["tree"] = box["tree"]  # merged again for this Crease Strength: the next drag starts from it
    live["error"] = box.get("error", "")
    # Shown even when a newer change came in meanwhile: it is newer than what is on screen, and the newer one follows.
    if "res" not in box or box["ticket"] <= live["shown"] or base is not box["base"] or base["out"]["faces"] is not box["faces"]:
        return
    live["shown"] = box["ticket"]
    view.seg.update(live=box["res"], live_count=box["count"], batch_key=None)
    live.update(count=box["count"], ms=1000.0 * (time.perf_counter() - box["asked"]))
    overlay.tag_redraw()


def _tick():
    try:
        task = live["task"]
        if task is not None and task.done():
            live["task"] = None
            _finished(live["box"])
        now = time.perf_counter()
        if live["task"] is None:
            if live["preview"].due(now):
                _start()
            elif live["keep"].due(now) and live["preview"].asked is None:
                live["keep"].take()
                work = view._g().state["work"]
                # the same settings, found for real; locked segments stay. A redo still running is never stopped for it: ask again later
                if not view.segment_goals.start_find(bpy.context, fresh=True) and work is not None and work["runner"].busy:
                    live["keep"].poke(now)
        busy = live["task"] is not None or live["preview"].asked is not None or live["keep"].asked is not None
        return TICK if busy else None
    except Exception as exc:  # a timer has nobody above it to report to
        live["error"] = str(exc)[:100]
        return None


def clear() -> None:
    """The preview is done with: the kept segments are on screen again."""
    view.seg.update(live=None, live_count=None, batch_key=None)


def _painting():
    """Your mesh, when it is in Weight Paint with Paint Size on and segments to preview; else None."""
    source = view._g().state_source()
    if source is None or source.mode != "WEIGHT_PAINT" or not source.curiomesh_seg.size_paint or view.seg.get("src") is None:
        return None
    return source


def _paint_sig(source) -> float | None:
    """A sum over a sample of the Paint Size weights, weighted by corner: a stroke changes it."""
    group = source.vertex_groups.get(SIZE_GROUP)
    if group is None:
        return None
    verts, index, total = source.data.vertices, group.index, 0.0
    for i in range(0, len(verts), max(1, len(verts) // PAINT_SAMPLES)):
        for g in verts[i].groups:
            if g.group == index:
                total += (i + 1) * g.weight
    return total


def _paint_tick():
    """While you paint: a changed sample asks for a preview. Ends when you leave Weight Paint."""
    try:
        source = _painting()
        if source is None:
            live["paint_sig"] = None
            return None
        sig = _paint_sig(source)
        if live["paint_sig"] is not None and sig != live["paint_sig"]:
            request(paint=True)
        live["paint_sig"] = sig
        return PAINT_POLL
    except (ReferenceError, AttributeError):
        live["paint_sig"] = None
        return None


def _watch() -> None:
    """From the viewport's redraw: start watching the paint once your mesh is in Weight Paint."""
    try:
        if not bpy.app.timers.is_registered(_paint_tick) and _painting() is not None:
            bpy.app.timers.register(_paint_tick, first_interval=PAINT_POLL)
    except (ReferenceError, AttributeError):
        pass


def _on_undo(*_args) -> None:
    """Undo and Redo bring back the segments saved on the mesh at that step."""
    try:
        view.segment_goals.reload_saved()
    except Exception as exc:  # a handler must never stop the undo
        live["error"] = str(exc)[:100]


class CURIOMESH_OT_seg_paint_size(bpy.types.Operator):
    bl_idname = "curiomesh.seg_paint_size"
    bl_label = "Paint Size"
    bl_description = f"Go to Weight Paint on your mesh with the '{SIZE_GROUP}' group: red makes smaller segments there, blue larger ones. The colours follow each stroke. Tab back when done. Turns on unified weight in the tool settings and sets the brush weight to 1"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return view._g().state_source() is not None

    def execute(self, context):
        source = view._g().state_source()
        group = source.vertex_groups.get(SIZE_GROUP)
        if group is None:
            group = source.vertex_groups.new(name=SIZE_GROUP)
            group.add(range(len(source.data.vertices)), 0.5, "REPLACE")  # green: the sliders' own size
        source.vertex_groups.active_index = group.index
        source.curiomesh_seg.size_paint = True
        for obj in context.selected_objects:
            obj.select_set(False)
        source.hide_set(False)
        source.select_set(True)
        context.view_layer.objects.active = source
        if source.mode != "WEIGHT_PAINT":
            bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
        try:  # a full red brush; the setting moved between versions
            paint = context.tool_settings.unified_paint_settings
            paint.use_unified_weight = True
            paint.weight = 1.0
        except AttributeError:
            pass
        return {"FINISHED"}


def draw(layout, source) -> None:
    """The size controls of this mesh, with the live count."""
    st = source.curiomesh_seg
    col = layout.column(align=True)
    count = view.seg.get("live_count") if view.seg.get("live") is not None else len(view.seg["parts"])
    busy = live["task"] is not None or live["preview"].asked is not None or live["keep"].asked is not None
    text = f"{count:,} parts" + (f"  ({live['ms']:.0f} ms)" if live["ms"] is not None and view.seg.get("live") is not None else "") + ("  ..." if busy else "")
    col.label(text=text, icon="MESH_GRID")
    col.prop(st, "detail", slider=True)
    col.prop(st, "largest", slider=True)
    col.prop(st, "smallest", slider=True)
    col.prop(st, "creases", slider=True)
    col.prop(st, "smooth", slider=True)
    row = col.row(align=True)
    row.prop(st, "size_paint", toggle=True, icon="BRUSH_DATA")
    row.operator("curiomesh.seg_paint_size", text="Paint", icon="MOD_VERTEX_WEIGHT")
    if live["error"]:
        overlay.wrap_label(layout, live["error"], icon="ERROR", width=32, limit=2)


PRE_CLASSES = (CURIOMESH_PG_segsettings,)
CLASSES = (CURIOMESH_OT_seg_paint_size,)


def register() -> None:
    bpy.types.Object.curiomesh_seg = bpy.props.PointerProperty(type=CURIOMESH_PG_segsettings)
    overlay.add_geometry_hook(_watch)
    add_handlers()


def add_handlers() -> None:
    """Undo and Redo handlers. Not kept over a file load by Blender: the add-on's load handler puts them back."""
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _on_undo not in handlers:
            handlers.append(_on_undo)


def unregister() -> None:
    overlay.remove_geometry_hook(_watch)
    for handlers, fn in ((bpy.app.handlers.undo_post, _on_undo), (bpy.app.handlers.redo_post, _on_undo)):
        if fn in handlers:
            handlers.remove(fn)
    for timer in (_tick, _paint_tick):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    live.update(task=None, paint_sig=None, box=None, ms=None, count=None, paint=None)
    if hasattr(bpy.types.Object, "curiomesh_seg"):
        del bpy.types.Object.curiomesh_seg
