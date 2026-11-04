# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Live segments: each mesh's own segment settings, and a coloured preview that follows them while you drag.

A change asks for a preview. When the asking has been quiet for a moment (or every tenth of a second during a long
drag), a thread cuts the merge tree kept from the last find at the new settings (Crease Strength merges it again
first) and marks the result's faces through a map of the nearest face of your mesh, worked out once per result. That
is all a preview needs, so it lands in well under a second; one preview runs at a time, and the newest change is
always the next one shown. When the changes have stopped for longer,
`FindJob` finds the segments for real (borders settled on the creases, seams laid on the result) and keeps them.

Paint Size is the `CM_SegmentSize` vertex group: red makes smaller segments there, blue larger. Strokes in Weight
Paint ask for a preview through the depsgraph handler, after a longer pause so painting stays smooth.
"""

from __future__ import annotations

import threading
import time

import bmesh
import bpy
import numpy as np

from . import overlay, parallel
from . import segments as view
from .triadq import segment as S

SIZE_GROUP = "CM_SegmentSize"
PREVIEW_WAIT = 0.03  # quiet before a preview: a drag sends many changes a second
PREVIEW_MOST = 0.1  # a drag that never pauses still gets a preview this often
PAINT_WAIT = 0.3  # after a stroke: reading the paint of a big mesh takes a moment, so not after every dab
KEEP_WAIT = 0.8  # quiet before the segments are found for real and kept
TICK = 0.02

live = {
    "preview": parallel.Debounce(PREVIEW_WAIT, PREVIEW_MOST),
    "keep": parallel.Debounce(KEEP_WAIT),
    "thread": None,
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

    detail: bpy.props.FloatProperty(name="Segment Detail", description="How many segments the mesh splits into. Left finds only the big parts, right also the smaller ones. The colours follow while you drag; locked segments stay as they are", default=0.65, min=0.0, max=1.0, subtype="FACTOR", update=_changed)
    largest: bpy.props.FloatProperty(name="Largest Segment", description="No segment grows past this share of its loose piece. 100% leaves the size to Segment Detail", default=100.0, min=2.0, max=100.0, subtype="PERCENTAGE", update=_changed)
    smallest: bpy.props.FloatProperty(name="Smallest Segment", description="A segment smaller than this share of its loose piece joins its neighbour", default=0.4, min=0.0, max=10.0, subtype="PERCENTAGE", update=_changed)
    creases: bpy.props.FloatProperty(name="Crease Strength", description="How strongly sharp edges and creases cut segments apart. Left: segments run over soft creases. Right: creases become borders", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=_changed)
    smooth: bpy.props.FloatProperty(name="Border Smoothness", description="How much a border is rounded off where no crease holds it. Left: borders follow the faces closely. Right: smoother, rounder borders. Shows when the segments are kept, a moment after you stop", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=_changed)
    size_paint: bpy.props.BoolProperty(name="Paint Size", description=f"Use the '{SIZE_GROUP}' vertex group: red makes smaller segments there, blue larger ones, green leaves the size to the sliders", default=False, update=_changed)


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
    live["thread"] = threading.Thread(target=work, daemon=True, name="CurioMeshLiveSegments")
    live["thread"].start()


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
        thread = live["thread"]
        if thread is not None and not thread.is_alive():
            live["thread"] = None
            _finished(live["box"])
        now = time.perf_counter()
        if live["thread"] is None:
            if live["preview"].due(now):
                _start()
            elif live["keep"].due(now) and live["preview"].asked is None:
                live["keep"].take()
                work = view._g().state["work"]
                # the same settings, found for real; locked segments stay. A redo still running is never stopped for it: ask again later
                if not view.segment_goals.start_find(bpy.context, fresh=True) and work is not None and work["runner"].busy:
                    live["keep"].poke(now)
        busy = live["thread"] is not None or live["preview"].asked is not None or live["keep"].asked is not None
        return TICK if busy else None
    except Exception as exc:  # a timer has nobody above it to report to
        live["error"] = str(exc)[:100]
        return None


def clear() -> None:
    """The preview is done with: the kept segments are on screen again."""
    view.seg.update(live=None, live_count=None, batch_key=None)


@bpy.app.handlers.persistent
def _on_depsgraph(scene, depsgraph) -> None:
    """A stroke in Weight Paint on the mesh with Paint Size on: preview the new sizes after the stroke."""
    try:
        source = view._g().state_source()
        if source is None or source.mode != "WEIGHT_PAINT" or not source.curiomesh_seg.size_paint or view.seg.get("src") is None:
            return
        mesh = source.data
        for update in depsgraph.updates:
            if update.id.original in (source, mesh):
                request(paint=True)
                return
    except (ReferenceError, AttributeError):
        pass


@bpy.app.handlers.persistent
def _on_undo(*_args) -> None:
    """Undo and Redo bring back the segments saved on the mesh at that step."""
    try:
        view.segment_goals.reload_saved()
    except Exception as exc:  # a handler must never stop the undo
        live["error"] = str(exc)[:100]


class CURIOMESH_OT_seg_paint_size(bpy.types.Operator):
    bl_idname = "curiomesh.seg_paint_size"
    bl_label = "Paint Size"
    bl_description = f"Go to Weight Paint on your mesh with the '{SIZE_GROUP}' group: red makes smaller segments there, blue larger ones. The colours follow each stroke. Tab back when done"
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
    busy = live["thread"] is not None or live["preview"].asked is not None or live["keep"].asked is not None
    text = f"{count:,} segments" + (f"  ({live['ms']:.0f} ms)" if live["ms"] is not None and view.seg.get("live") is not None else "") + ("  ..." if busy else "")
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
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    bpy.app.handlers.undo_post.append(_on_undo)
    bpy.app.handlers.redo_post.append(_on_undo)


def unregister() -> None:
    for handlers, fn in ((bpy.app.handlers.depsgraph_update_post, _on_depsgraph), (bpy.app.handlers.undo_post, _on_undo), (bpy.app.handlers.redo_post, _on_undo)):
        if fn in handlers:
            handlers.remove(fn)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    live.update(thread=None, box=None, ms=None, count=None, paint=None)
    if hasattr(bpy.types.Object, "curiomesh_seg"):
        del bpy.types.Object.curiomesh_seg
