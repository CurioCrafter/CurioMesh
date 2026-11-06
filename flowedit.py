# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Flow Editor: draw how the loops should run on a finished result, and only the quads around each stroke are redone.

The strokes are ordinary curves in the guide list (saved with the .blend, used again by Start), marked with their kind
(`flowstroke`: Flow, Ring or Edge). Draw recognises the kind itself; Draw Flow, Draw Ring and Draw Edge force one;
Place Circle and the face shapes are clicked. After each change the faces around the strokes are worked out again in
timer slices (the parts' redo runner, `segment_goals.in_step`), from the result as it was before any stroke;
strokes that did not change reuse their patch, and the timer only steps the job and swaps the preview mesh. With a mirrored result every stroke also works on the other side.
"""

from __future__ import annotations

import time

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import flowstroke as F
from . import guides, overlay, region
from .triadq import checks, issues, loopscore
from .triadq import segment as S

KIND_KEY = "curiomesh_flow"  # custom property of a stroke's curve: FLOW, RING or EDGE
RAW_KEY = "curiomesh_flow_raw"  # the stroke as drawn (flat x y z list), so "change to" starts from the hand's line
LIST_KIND = {"FLOW": "GUIDE", "EDGE": "GUIDE", "RING": "LOOP"}
COLOURS = {"FLOW": (0.25, 0.6, 1.0, 1.0), "RING": (0.3, 0.95, 0.4, 1.0), "EDGE": (1.0, 0.55, 0.1, 1.0)}
GAP = 0.15  # seconds between timer slices of the job and looks at the strokes (moved in Edit Mode, say)
SHAPE_SIZE = {"EYE": 0.06, "MOUTH": 0.09, "NOSE_MOUTH": 0.09}  # of the mesh's diagonal, when Shape Size is 0
CREASE_DEG = 40.0  # a source edge folding more than this is a crease Draw can lay an Edge on
CREASE_NEAR = 0.006  # of the diagonal: how near a stroke has to run to a crease to be on it

fe = {
    "base": None,  # the result (guided.state["auto_base"]) being edited
    "origin": None,  # its output before any stroke
    "shown": None,  # the output on screen
    "centres": None,
    "adjacency": None,
    "cache": {},  # (face ids, stroke signature) -> (points, faces)
    "box": None,
    "job": None,  # the generator working out the latest strokes
    "sig": None,
    "last": None,  # (curve name, kind) of the last stroke drawn
    "meter": "",
    "crease": None,  # (source signature, sample points, directions)
    "batch": None,
}


def _g():
    from . import guided  # late: guided imports this module

    return guided


def cancel() -> None:
    fe.update(job=None, base=None, origin=None, shown=None, centres=None, adjacency=None, cache={}, box=None, sig=None, meter="", batch=None)


def _curves(gs):
    return [g for g in gs.guide_curves if g.obj is not None and KIND_KEY in g.obj]


def strokes(context, gs) -> list[dict]:
    """Every Flow Editor stroke as `flowstroke.record`s in the source's space, mirrored when the result is."""
    guided = _g()
    source = guided.state_source()
    if source is None:
        return []
    depsgraph = context.evaluated_depsgraph_get()
    out = []
    for g in _curves(gs):
        for line in guides.gather([g.obj], source, depsgraph):
            out.append(F.record(str(g.obj[KIND_KEY]), line))
    base = guided.state["auto_base"]
    mirror = base and getattr(base["job"], "mirror", None)
    if mirror is not None:
        axis, offset = mirror[:2]
        for s in list(out):
            p = s["points"].copy()
            p[:, axis] = 2.0 * offset - p[:, axis]
            out.append(F.record(s["kind"], p))
    return out


def _sig(items) -> tuple:
    return tuple((s["kind"], round(float(s["points"].sum()), 6), len(s["points"])) for s in items)


def active() -> bool:
    return fe["base"] is not None and _g().state["auto_base"] is fe["base"]


def _begin() -> bool:
    guided = _g()
    base = guided.state["auto_base"]
    if base is None:
        return False
    if fe["base"] is not base or (base["out"] is not fe["shown"] and base["out"] is not fe["origin"]):  # a new result, or Organize swapped it
        cancel()
        out = base["out"]
        from .triadq import patch

        fe.update(base=base, origin=out, shown=out, centres=np.asarray([out["co"][list(f)].mean(axis=0) for f in out["faces"]]), adjacency=patch.face_adjacency(out["faces"]))
    return True


def request(context) -> bool:
    """Work out the strokes again (only the patches whose strokes changed), in timer slices."""
    gs = context.scene.curiomesh_guided
    if not gs.flow_live or not _begin():
        return False
    items = strokes(context, gs)
    fe["sig"] = _sig(items)
    origin = fe["origin"]
    plan = F.areas(origin["co"], origin["faces"], items, fe["centres"], fe["adjacency"])
    guided = _g()
    src = guided._source_data(guided.state_source())
    project = region._projector(guided._bvh(src))
    box = {"stop": False, "out": None, "meter": "", "error": None, "started": time.perf_counter()}
    fe["box"] = box
    fe["job"] = _job(box, origin, plan, project, src["co"], src["tris"])  # a newer request drops the older job
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=GAP)
    return True


def _local(co, faces, ids):
    sub = [faces[i] for i in ids]
    used = np.unique(np.fromiter((v for f in sub for v in f), dtype=np.int64))
    remap = np.full(len(co), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return co[used], [tuple(int(remap[v]) for v in f) for f in sub]


def _issues(co, faces, mask, src_co, src_tris) -> int:
    """Issue Finder spots among the masked faces (where the strokes redid the result, seams included)."""
    found = checks.defects(co, faces, src_co, [tuple(t) for t in src_tris.tolist()])
    hit = np.any([getattr(found, k) for k in checks.Defects.KINDS], axis=0)
    _spots, record = issues.analyse(np.asarray(co), faces, {"here": hit & mask}, None, {"source_faces": len(src_tris)})
    return int(sum(record["totals"].values()))


def _job(box, origin, plan, project, src_co, src_tris):
    """Generator, stepped by the timer in short slices: each patch redone (or taken from the cache), then put in and
    measured, a slice at a time."""
    from .segment_goals import in_step

    co, faces = origin["co"], origin["faces"]
    builds, timings, failed = [], [], []
    for ids, items in plan:
        key = (ids.tobytes(), _sig(items))
        made = fe["cache"].get(key)
        if made is None:
            t = time.perf_counter()
            done = yield from in_step(lambda ids=ids, items=items: F.redo(co, faces, ids, items, project))
            if "out" not in done:
                failed.append(done.get("error", "no result"))
                continue
            made = fe["cache"][key] = done["out"]
            timings.append(time.perf_counter() - t)
        builds.append((ids, made))
    done = yield from in_step(lambda: _finish(origin, builds, failed, sum(timings) if timings else None, src_co, src_tris))
    if "out" not in done:
        box["error"] = f"Flow Editor failed: {done.get('error')}"
        return
    box.update(done["out"])


def _finish(origin, builds, failed, seconds, src_co, src_tris) -> dict:
    """The result with the patches put in, and what changed where they went."""
    co, faces, mat = origin["co"], origin["faces"], np.asarray(origin["mat"])
    if builds:
        new_co, new_faces, new_mat, new = F.splice(co, faces, mat, builds)
        if S.open_edges(new_faces) > S.open_edges(faces):
            return {"error": "A redo did not close on its seam, so the result stays as it was"}
    else:
        new_co, new_faces, new_mat, new = co, faces, mat, np.zeros(len(faces), dtype=bool)
    out = dict(origin, co=new_co, faces=new_faces, mat=new_mat, new=new, locked=np.zeros(len(new_faces), dtype=bool))
    for name in ("parts", "piece", "chart", "_seg"):
        out.pop(name, None)
    if not builds:
        return {"out": out, "seconds": 0.0, "meter": "Draw on the result: the quads around each stroke are redone" + (f". Could not redo: {failed[0]}" if failed else "")}
    old_ids = np.concatenate([ids for ids, _m in builds])
    before = loopscore.loop_quality(*_local(co, faces, old_ids))
    after = loopscore.loop_quality(*_local(new_co, new_faces, np.flatnonzero(new)))
    was = np.zeros(len(faces), dtype=bool)
    was[old_ids] = True
    spots = (_issues(co, faces, was, src_co, src_tris), _issues(new_co, new_faces, new, src_co, src_tris))
    text = f"Redone where you drew: faces {len(old_ids):,} -> {int(new.sum()):,}" + (f" in {seconds:.1f} s" if seconds else "")
    text += f". {loopscore.meter(before, after)}. Issues there {spots[0]} -> {spots[1]}"
    if failed:
        text += f". Could not redo one area: {failed[0]}"
    return {"out": out, "seconds": seconds or 0.0, "meter": text, "numbers": {"before": before, "after": after, "issues": spots, "faces": (len(old_ids), int(new.sum())), "seconds": seconds or 0.0}}


def _tick():
    if not active():
        return None
    context = bpy.context
    gs = getattr(context.scene, "curiomesh_guided", None)
    job = fe.get("job")
    if job is not None:
        ends = time.perf_counter() + 0.02  # one short slice of the job per tick
        try:
            while time.perf_counter() < ends:
                next(job)
        except StopIteration:
            fe["job"] = None
    box = fe["box"]
    if box is not None and box["out"] is not None and box["out"] is not fe["shown"]:
        guided = _g()
        out = box["out"]
        fe["shown"] = out
        fe["base"]["out"] = out
        guided.state["res"] = out
        guided._build_preview(context, guided.state_source(), out, uvs=False)
        guided.state["draw_key"] = None
        overlay.tag_redraw()
    if box is not None:
        fe["meter"] = box["error"] or box["meter"] or "Working out the strokes..."
    if gs is not None and gs.flow_live:  # a stroke moved in Edit Mode, deleted, or its kind changed
        try:
            now = _sig(strokes(context, gs))
        except Exception:
            now = fe["sig"]
        if now != fe["sig"]:
            request(context)
    fe["batch"] = None
    return GAP if fe["job"] is not None or (gs is not None and gs.flow_live) else None  # idle and not Live: the timer ends


# --------------------------------------------------------------------------- creases of your mesh (for Draw)


def _crease(src):
    """(points, directions) sampled along the source's sharp creases, cached per source."""
    if fe["crease"] is not None and fe["crease"][0] == src["sig"]:
        return fe["crease"][1:]
    co, tris = src["co"], src["tris"]
    n = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    a = tris.reshape(-1)
    b = np.roll(tris, -1, axis=1).reshape(-1)
    face = np.repeat(np.arange(len(tris)), 3)
    key = np.minimum(a, b) * len(co) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    k = key[order]
    same = np.flatnonzero(k[1:] == k[:-1])
    f1, f2 = face[order][same], face[order][same + 1]
    sharp = (n[f1] * n[f2]).sum(axis=1) < np.cos(np.radians(CREASE_DEG))
    e = order[same][sharp]
    p, q = co[a[e]], co[b[e]]
    d = q - p
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    fe["crease"] = (src["sig"], 0.5 * (p + q), d)
    return fe["crease"][1:]


def crease_test(src):
    points, dirs = _crease(src)
    reach = CREASE_NEAR * src["diag"]

    def test(stroke):
        if not len(points):
            return np.zeros(len(stroke), dtype=bool), np.zeros((len(stroke), 3))
        dist, index = guides._nearest_sample(stroke, points)
        return dist < reach, dirs[index]

    return test


# --------------------------------------------------------------------------- making and changing strokes


def add_stroke(context, source, kind: str, line, raw=None):
    """A stroke laid on the source as a curve in the guide list. Returns the curve object."""
    guided = _g()
    gs = context.scene.curiomesh_guided
    src = guided._source_data(source)
    bvh = guided._bvh(src)
    hits = [bvh.find_nearest(Vector(p)) for p in np.asarray(line).tolist()]
    points = [np.asarray(h[0] if h[0] is not None else p) for h, p in zip(hits, np.asarray(line).tolist())]
    normals = [np.asarray(h[1] if h[1] is not None else (0, 0, 1)) for h in hits]
    obj = guided.new_curve(context, source, LIST_KIND[kind], points, normals, name="CM_" + F.NAMES[kind])
    obj[KIND_KEY] = kind
    obj[RAW_KEY] = np.asarray(line if raw is None else raw, dtype=np.float64).ravel().tolist()
    item = gs.guide_curves.add()
    item.obj, item.kind = obj, LIST_KIND[kind]
    gs.guide_index = len(gs.guide_curves) - 1
    if kind == "RING":
        gs.loop_use = True  # Start uses it too
    else:
        gs.guide_use = True
    fe["last"] = (obj.name, kind)
    fe["batch"] = None
    return obj


def remove_stroke(gs, obj) -> None:
    for k, g in enumerate(gs.guide_curves):
        if g.obj is obj:
            gs.guide_curves.remove(k)
            break
    gs.guide_index = max(0, min(gs.guide_index, len(gs.guide_curves) - 1))
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        bpy.data.curves.remove(data)
    fe["batch"] = None


def _nearest_surface(source):
    guided = _g()
    bvh = guided._bvh(guided._source_data(source))

    def nearest(points):
        return np.asarray([(bvh.find_nearest(Vector(p))[0] or Vector(p))[:] for p in np.asarray(points).tolist()], dtype=np.float64)

    return nearest


def _hit(op, context, event):
    view = op._view_region(context, event)
    return (None, None) if view is None else (view, op._ray(view, event))


def _view_axes(view):
    """(right, up) of the 3D view under the mouse, in world space."""
    r = view.data.view_rotation
    return np.asarray(r @ Vector((1, 0, 0))), np.asarray(r @ Vector((0, 1, 0)))


def _local_dir(source, d):
    return np.asarray(source.matrix_world.inverted().to_3x3() @ Vector(d))


class _FlowTool:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        guided = _g()
        o = context.object
        return o is not None and o.type == "MESH" and o.name != guided.PREVIEW_NAME and o.mode == "OBJECT" and context.area is not None and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        guided = _g()
        self.obj = context.object
        guided.state["source"] = self.obj.name
        self.stroke, self.last, self.anchor = None, None, None
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(self.HINT)
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        context.workspace.status_text_set(None)
        overlay.tag_redraw()
        return {"FINISHED"}

    def _changed(self, context):
        request(context)
        overlay.tag_redraw()


def _surface_base():
    return _g()._SurfaceTool


KIND_ITEMS = [
    ("AUTO", "Draw", "Draw a stroke and it is recognised: a closed loop becomes a Ring, a line along a sharp crease an Edge, any other line a Flow line"),
    ("FLOW", "Draw Flow", "Draw a line the rows of quads should run along (across a forehead, along an arm)"),
    ("RING", "Draw Ring", "Draw a loop round something (an eye, the mouth, a hole, a bolt): it becomes a closed edge loop with rings of quads round it"),
    ("EDGE", "Draw Edge", "Draw a line that must become a run of edges exactly (a crease, a panel line)"),
]


def _make_draw():
    base = _surface_base()

    class CURIOMESH_OT_flow_edit(_FlowTool, base, bpy.types.Operator):
        bl_idname = "curiomesh.flow_edit"
        bl_label = "Draw"
        bl_description = "Draw on your mesh with the left mouse button held. Each stroke reshapes the quads around it. F, R, E right after a stroke change it to Flow, Ring or Edge. Right click, Enter or Esc: done"
        HINT = "Flow Editor: hold left mouse and draw on the mesh  F/R/E: change the last stroke to Flow/Ring/Edge  Right click/Enter/Esc: done"

        kind: bpy.props.EnumProperty(name="Kind", items=KIND_ITEMS, default="AUTO", options={"SKIP_SAVE"})

        @classmethod
        def description(cls, context, properties):
            return next(d for k, _n, d in KIND_ITEMS if k == properties.kind)

        def invoke(self, context, event):
            return _FlowTool.invoke(self, context, event)

        def _end_stroke(self, context):
            points, self.stroke = self.stroke, None
            if points is None or len(points) < 3:
                return
            raw = np.asarray([p for p, _n in points])
            normals = np.asarray([n for _p, n in points])
            guided = _g()
            src = guided._source_data(self.obj)
            kind = self.kind if self.kind != "AUTO" else F.recognise(raw, normals, crease_test(src))
            add_stroke(context, self.obj, kind, F.tidy(kind, raw), raw)
            context.workspace.status_text_set(f"Recognised: {F.NAMES[kind]}.  F/R/E: change to Flow/Ring/Edge  Keep drawing, or Right click/Enter/Esc: done")
            self._changed(context)

        def modal(self, context, event):
            if context.area is not None:
                context.area.tag_redraw()
            if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
                self._end_stroke(context)
                return self._finish(context)
            if event.type in {"F", "R", "E"} and event.value == "PRESS" and self.stroke is None:
                change_last(context, {"F": "FLOW", "R": "RING", "E": "EDGE"}[event.type])
                return {"RUNNING_MODAL"}
            view = self._view_region(context, event)
            if view is None:
                return {"PASS_THROUGH"}
            if event.type == "RIGHTMOUSE" and event.value == "PRESS":
                self._end_stroke(context)
                return self._finish(context)
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                self.stroke, self.last = [], None
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._end_stroke(context)
                return {"RUNNING_MODAL"}
            if event.type in {"LEFTMOUSE", "MOUSEMOVE"} and self.stroke is not None:
                at = (event.mouse_x, event.mouse_y)
                if self.last is None or (at[0] - self.last[0]) ** 2 + (at[1] - self.last[1]) ** 2 >= 25:
                    hit = self._ray(view, event)
                    if hit is not None:
                        self.stroke.append(hit)
                        self.last = at
                return {"RUNNING_MODAL"}
            if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt:
                return {"PASS_THROUGH"}
            return {"RUNNING_MODAL"}

    class CURIOMESH_OT_flow_erase(_FlowTool, base, bpy.types.Operator):
        bl_idname = "curiomesh.flow_erase"
        bl_label = "Erase"
        bl_description = "Click a stroke on your mesh to take it away; the quads there go back to how they were. Right click, Enter or Esc: done"
        HINT = "Flow Editor: click a stroke to erase it  Right click/Enter/Esc: done"

        def invoke(self, context, event):
            return _FlowTool.invoke(self, context, event)

        def modal(self, context, event):
            if event.type in {"RET", "NUMPAD_ENTER", "ESC", "RIGHTMOUSE"} and event.value == "PRESS":
                return self._finish(context)
            view, hit = _hit(self, context, event)
            if view is None:
                return {"PASS_THROUGH"}
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                if hit is not None:
                    gs = context.scene.curiomesh_guided
                    guided = _g()
                    reach = 0.03 * guided._source_data(self.obj)["diag"]
                    best = None
                    for g in _curves(gs):
                        for line in guides.gather([g.obj], self.obj, context.evaluated_depsgraph_get()):
                            d = float(guides._to_line(hit[0][None], line)[0])
                            if d < reach and (best is None or d < best[0]):
                                best = (d, g.obj)
                    if best is not None:
                        remove_stroke(gs, best[1])
                        self._changed(context)
                return {"RUNNING_MODAL"}
            if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt:
                return {"PASS_THROUGH"}
            return {"RUNNING_MODAL"}

    class CURIOMESH_OT_flow_circle(_FlowTool, base, bpy.types.Operator):
        bl_idname = "curiomesh.flow_circle"
        bl_label = "Place Circle"
        bl_description = "Click the middle of a hole, bolt or eye and drag out to size: a clean circle laid on your mesh becomes a Ring. Hold Shift while dragging for an ellipse. Right click, Enter or Esc: done"
        HINT = "Flow Editor: click the middle and drag to size (Shift: ellipse)  Right click/Enter/Esc: done"

        def invoke(self, context, event):
            return _FlowTool.invoke(self, context, event)

        def modal(self, context, event):
            if event.type in {"RET", "NUMPAD_ENTER", "ESC", "RIGHTMOUSE"} and event.value == "PRESS":
                return self._finish(context)
            view = self._view_region(context, event)
            if view is None:
                return {"PASS_THROUGH"}
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                hit = self._ray(view, event)
                self.anchor = (hit, (event.mouse_x, event.mouse_y)) if hit is not None else None
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE" and self.anchor is not None:
                (centre, normal), start = self.anchor
                self.anchor = None
                from bpy_extras import view3d_utils

                coord = (event.mouse_x - view.x, event.mouse_y - view.y)
                origin = view3d_utils.region_2d_to_origin_3d(view, view.data, coord)
                ray = view3d_utils.region_2d_to_vector_3d(view, view.data, coord)
                world_c = self.obj.matrix_world @ Vector(centre)
                wn = (self.obj.matrix_world.to_3x3() @ Vector(normal)).normalized()
                denom = ray.dot(wn)
                end = origin + ray * ((world_c - origin).dot(wn) / denom) if abs(denom) > 1e-6 else world_c
                right, up = _view_axes(view)
                drag = np.asarray(end - world_c)
                if event.shift:
                    rx, ry = abs(float(drag @ right)), abs(float(drag @ up))
                else:
                    rx = ry = float(np.linalg.norm(drag))
                scale = float(np.mean(self.obj.matrix_world.to_scale()))
                if max(rx, ry) > 1e-9:
                    ring = F.ellipse(centre, normal, _local_dir(self.obj, right), max(rx, 1e-6) / scale, max(ry, 1e-6) / scale, 48)
                    add_stroke(context, self.obj, "RING", _nearest_surface(self.obj)(ring))
                    self._changed(context)
                return {"RUNNING_MODAL"}
            if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt:
                return {"PASS_THROUGH"}
            return {"RUNNING_MODAL"}

    class CURIOMESH_OT_flow_shape(_FlowTool, base, bpy.types.Operator):
        bl_idname = "curiomesh.flow_shape"
        bl_label = "Place Shape"
        HINT = "Flow Editor: click where the shape goes  Right click/Enter/Esc: done"

        shape: bpy.props.EnumProperty(name="Shape", items=[
            ("EYE", "Eye Ring", "Click the middle of an eye: a closed ring round it, and one on the other eye when Both Sides is on"),
            ("MOUTH", "Mouth Ring", "Click the middle of the mouth: a closed ring round the lips"),
            ("NOSE_MOUTH", "Nose to Mouth", "Click the middle of the mouth: two lines from beside the nose round the mouth corners, the fold that creases in a smile"),
        ], default="EYE", options={"SKIP_SAVE"})

        @classmethod
        def description(cls, context, properties):
            return {"EYE": "Click the middle of an eye: a closed ring round it (and the other eye with Both Sides)", "MOUTH": "Click the middle of the mouth: a closed ring round the lips", "NOSE_MOUTH": "Click the middle of the mouth: lines from beside the nose round the mouth corners"}[properties.shape]

        def invoke(self, context, event):
            return _FlowTool.invoke(self, context, event)

        def modal(self, context, event):
            if event.type in {"RET", "NUMPAD_ENTER", "ESC", "RIGHTMOUSE"} and event.value == "PRESS":
                return self._finish(context)
            view, hit = _hit(self, context, event)
            if view is None:
                return {"PASS_THROUGH"}
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                if hit is not None:
                    place_shape(context, self.obj, self.shape, hit[0], hit[1], _local_dir(self.obj, _view_axes(view)[0]))
                    self._changed(context)
                return {"RUNNING_MODAL"}
            if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt:
                return {"PASS_THROUGH"}
            return {"RUNNING_MODAL"}

    return [CURIOMESH_OT_flow_edit, CURIOMESH_OT_flow_erase, CURIOMESH_OT_flow_circle, CURIOMESH_OT_flow_shape]


def place_shape(context, source, shape: str, centre, normal, right) -> list:
    """A clicked shape laid on the source; an eye also on the other side with Both Sides (unless the result mirrors anyway)."""
    guided = _g()
    gs = context.scene.curiomesh_guided
    src = guided._source_data(source)
    size = gs.flow_size or SHAPE_SIZE[shape] * src["diag"]
    nearest = _nearest_surface(source)
    made = []
    for kind, line in F.template(shape, centre, normal, right, size):
        made.append(add_stroke(context, source, kind, nearest(line)))
    base = guided.state["auto_base"]
    mirrored = base is not None and getattr(base["job"], "mirror", None) is not None
    if shape == "EYE" and gs.flow_pair and not mirrored:
        middle = 0.5 * (src["co"].min(axis=0) + src["co"].max(axis=0))
        axis = 0
        other = np.asarray(centre, dtype=np.float64).copy()
        other[axis] = 2.0 * middle[axis] - other[axis]
        hit = guided._bvh(src).find_nearest(Vector(other))
        if hit[0] is not None:
            flip = np.asarray(right, dtype=np.float64).copy()
            flip[axis] = -flip[axis]
            for kind, line in F.template(shape, np.asarray(hit[0]), np.asarray(hit[1]), -flip, size):
                made.append(add_stroke(context, source, kind, nearest(line)))
    return made


def change_last(context, kind: str) -> bool:
    """The last stroke (or the chosen one in the list) made again as `kind` from the line as drawn."""
    gs = context.scene.curiomesh_guided
    obj = None
    if fe["last"] is not None:
        obj = bpy.data.objects.get(fe["last"][0])
    if obj is None and 0 <= gs.guide_index < len(gs.guide_curves):
        obj = gs.guide_curves[gs.guide_index].obj
    if obj is None or KIND_KEY not in obj:
        return False
    guided = _g()
    source = guided.state_source()
    raw = np.asarray(obj.get(RAW_KEY, []), dtype=np.float64).reshape(-1, 3)
    if source is None or len(raw) < 3:
        return False
    remove_stroke(gs, obj)
    add_stroke(context, source, kind, F.tidy(kind, raw), raw)
    request(context)
    return True


class CURIOMESH_OT_flow_change(bpy.types.Operator):
    bl_idname = "curiomesh.flow_change"
    bl_label = "Change To"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(name="Kind", items=[(k, n, "") for k, n, _d in KIND_ITEMS[1:]], default="FLOW", options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        return {"FLOW": "Make the last stroke a Flow line: the rows run along it", "RING": "Make the last stroke a Ring: a closed edge loop with rings round it", "EDGE": "Make the last stroke an Edge: a run of edges exactly on it"}[properties.kind]

    def execute(self, context):
        return {"FINISHED"} if change_last(context, self.kind) else {"CANCELLED"}


class CURIOMESH_OT_flow_clear(bpy.types.Operator):
    bl_idname = "curiomesh.flow_clear"
    bl_label = "Erase All"
    bl_description = "Take away every Flow Editor stroke; the result goes back to how it was before you drew"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        gs = context.scene.curiomesh_guided
        for g in list(_curves(gs)):
            remove_stroke(gs, g.obj)
        request(context)
        return {"FINISHED"}


def draw_tools(layout, gs) -> None:
    """The Flow Editor's tools, under the Draw on the mesh button (`panel.draw_shape_flow`)."""
    col = layout.column(align=True)
    row = col.row(align=True)
    row.operator("curiomesh.flow_edit", text="Flow").kind = "FLOW"
    row.operator("curiomesh.flow_edit", text="Ring").kind = "RING"
    row.operator("curiomesh.flow_edit", text="Edge").kind = "EDGE"
    row = col.row(align=True)
    row.operator("curiomesh.flow_circle", icon="MESH_CIRCLE")
    row.operator("curiomesh.flow_erase", icon="TRASH")
    row = col.row(align=True)
    row.operator("curiomesh.flow_shape", text="Eye").shape = "EYE"
    row.operator("curiomesh.flow_shape", text="Mouth").shape = "MOUTH"
    row.operator("curiomesh.flow_shape", text="Nose-Mouth").shape = "NOSE_MOUTH"
    row = col.row(align=True)
    row.prop(gs, "flow_size")
    row.prop(gs, "flow_pair", toggle=True)
    row = col.row(align=True)
    row.prop(gs, "flow_live", toggle=True)
    row.operator("curiomesh.flow_clear", icon="X")
    last = fe["last"]
    if last is not None and bpy.data.objects.get(last[0]) is not None:
        row = col.row(align=True)
        row.label(text=f"Last: {F.NAMES[last[1]]}")
        for kind in F.KINDS:
            if kind != last[1]:
                row.operator("curiomesh.flow_change", text=F.NAMES[kind]).kind = kind
    if active() and fe["meter"]:
        overlay.wrap_label(col, fe["meter"], icon="MESH_GRID", width=36, limit=5)


# --------------------------------------------------------------------------- strokes in colour


def _draw_strokes() -> None:
    try:
        context = bpy.context
        gs = getattr(context.scene, "curiomesh_guided", None)
        if gs is None:
            return
        curves = _curves(gs)
        if not curves:
            return
        if fe["batch"] is None:
            ends, colours = [], []
            for g in curves:
                obj = g.obj
                for sp in obj.data.splines:
                    pts = [obj.matrix_world @ Vector(p.co[:3]) for p in sp.points]
                    for a, b in zip(pts[:-1], pts[1:]):
                        ends += [a[:], b[:]]
                        colours += [COLOURS.get(str(obj[KIND_KEY]), COLOURS["FLOW"])] * 2
            shader = gpu.shader.from_builtin("FLAT_COLOR")
            fe["batch"] = batch_for_shader(shader, "LINES", {"pos": np.asarray(ends, dtype=np.float32).reshape(-1, 3), "color": np.asarray(colours, dtype=np.float32).reshape(-1, 4)})
        shader = gpu.shader.from_builtin("FLAT_COLOR")
        gpu.state.line_width_set(3.0)
        shader.bind()
        fe["batch"].draw(shader)
        gpu.state.line_width_set(1.0)
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


_classes: list = []


def classes() -> list:
    """The operators; the drawing tools share the guided mode's surface tool, so they are made once it is loaded."""
    if not _classes:
        _classes.extend(_make_draw() + [CURIOMESH_OT_flow_change, CURIOMESH_OT_flow_clear])
    return _classes


_handles: list = []


def register() -> None:
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_strokes, (), "WINDOW", "POST_VIEW"))


def unregister() -> None:
    while _handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), "WINDOW")
        except Exception:
            pass
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    cancel()
