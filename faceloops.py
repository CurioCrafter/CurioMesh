# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Face Loops: pick one strip of quads of the result on screen and adjust it (the Blender side of `triadq.strips`).

Pick a Strip shows the strip under the mouse and selects it on click. Slide (drag the mouse), Straighten, Even Width,
Add Strip and Remove Strip change it on the result: each edit is one Ctrl+Z step and one History entry. Straighten,
Even Width, Remove Strip and Redraw in Rows run the Tidy Tools of `triadq.stripfix` (the same maths Auto Fix uses), and
Fix Every Strip is the "Fix every spot like this" run of Straighten and Even Width in Look First. The result
lives in Python, so each edit is kept by the add-on's shared undo history (`history.before` / `history.record`), the
same steps Tidy Up and Look First use; the selected strip is part of each step.
"""

from __future__ import annotations

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import autoretopo, history, icons, overlay, region, segments, tidylive
from . import organize as ORG
from .triadq import autofix, loopscore, patch
from .triadq import stripfix as SF
from .triadq import trial as TT
from .triadq import organize as O
from .triadq import rings as RG
from .triadq import strips as S
from .triadq.patterns import Topology

WIGGLY, UNEVEN, SHORT = SF.WIGGLY, SF.UNEVEN, SF.SHORT  # Fix This: straighten, even out, or take away a dead end
SHARP_DEG = 60.0  # an edge folding more than this holds its vertices: Tidy Up's 35 held 40% of a coarse bust's face
LASSO_FACING = 0.25  # a lasso takes faces turned at least this much toward the view (cosine): not the sides it only grazes
RING_VIEW = 3.0  # points of view change Ring Round This may spend
POLE_PIXELS = 14  # how near the mouse has to be to a pole dot to drag it
DRAG_PIXELS = 250.0  # mouse travel that slides a strip a whole quad over
NAMES = {"slide": "Slide Strip", "straighten": "Straighten Strip", "even": "Even Width", "fix": "Fix This Strip", "add": "Add Strip", "remove": "Remove Strip"}
SEL_COLOUR = (1.0, 0.55, 0.1, 0.4)
HOVER_COLOUR = (1.0, 1.0, 1.0, 0.25)

fl = {
    "res": None,  # the result the caches below belong to
    "strips": None,
    "tree": None,
    "stay": None,
    "go": None,
    "walks": {},  # strip label -> walk
    "hover": None,  # element under the mouse while picking
    "sel": None,  # (result, element)
    "live": None,  # points shown while sliding
    "batch": None,
    "note": "",
    "lasso": None,  # region pixels of the Redraw in Rows lasso while it is drawn
    "look": None,  # (source signature, its pictures, centre, reach) for the camera check
    "ring_show": None,  # (centre, u, v, [(width, height), ...]): ovals drawn while Ring Round This is dragged
}


def _g():
    from . import guided  # late: guided imports this module

    return guided


def cancel() -> None:
    fl.update(res=None, strips=None, tree=None, stay=None, go=None, walks={}, hover=None, sel=None, live=None, batch=None, note="", lasso=None, ring_show=None)


def _cache(res) -> None:
    if fl["res"] is not res:
        fl.update(res=res, strips=None, tree=None, stay=None, go=None, walks={}, hover=None, batch=None)


def strips(res) -> S.Strips:
    _cache(res)
    if fl["strips"] is None:
        fl["strips"] = S.Strips(res["co"], res["faces"])
    return fl["strips"]


def walk(res, element: int) -> dict:
    st = strips(res)
    label = int(st.labels[element])
    if label not in fl["walks"]:
        fl["walks"][label] = st.walk(element)
    return fl["walks"][label]


def selected():
    """(result, element, strip) of the selected strip on the result on screen, or None."""
    res = _g().state["res"]
    sel = fl["sel"]
    if res is None or sel is None or sel[0] is not res or sel[1] is None:
        return None
    return res, sel[1], walk(res, sel[1])


# --------------------------------------------------------------------------- one Ctrl+Z step per edit


def commit(context, out, element, parent=None, name: str = "Face Loops", numbers: str = "") -> None:
    """Show `out` as the result and keep it as a History step (Ctrl+Z goes back to the one before). `parent`: the old
    face of each face, when faces changed. Operators with the UNDO flag get Blender's step pushed for them; the strip
    drags push their own."""
    guided = _g()
    base = guided.state["auto_base"]
    history.before(context)
    if parent is not None:
        piece = segments.seg.get("piece")
        if piece is not None and len(piece) == len(base["out"]["faces"]):
            segments.seg.update(piece=np.asarray(piece)[parent], batch_key=None, bad_key=None, tris=None, hover=None)
    base["out"] = guided.state["res"] = out
    guided._build_preview(context, guided.state_source(), out, uvs=False)
    guided.state["draw_key"] = None
    fl["sel"], fl["live"] = (out, element), None
    history.record(context, name, numbers)
    overlay.tag_redraw()


def _faces_changed(out, co, faces, parent) -> dict:
    """`out` with new faces; the per-face numbers follow each face's parent."""
    made = dict(out, co=np.asarray(co, dtype=np.float64), faces=[tuple(f) for f in faces])
    old = len(out["faces"])
    for name in (*ORG.FACE_ARRAYS, "bad", "_seg"):
        value = out.get(name)
        if value is not None and len(value) == old:
            made[name] = np.asarray(value)[parent]
    made.pop("issue_spots", None)
    return made


# --------------------------------------------------------------------------- edits


def _plan(res, strip, reach: int = S.REACH):
    _cache(res)
    co = np.asarray(res["co"], dtype=np.float64)
    if fl["stay"] is None:
        fl["stay"] = O.pinned(co, res["faces"], sharp_deg=SHARP_DEG)
        fl["go"] = loopscore._continuations(Topology(co, res["faces"]))
    return S.rung_lines(strip, fl["go"], len(co), fl["stay"], reach), fl["stay"]


def _project():
    guided = _g()
    source = guided.state_source()
    return None if source is None else region._projector(guided._bvh(guided._source_data(source)))


def _mirror(res, strip):
    """((axis, offset, partner per vertex), side the strip is on) for a mirrored result, else (None, 1)."""
    base = _g().state["auto_base"]
    mirror = getattr(base and base.get("job"), "mirror", None)
    if mirror is None:
        return None, 1.0
    axis, offset = mirror[0], mirror[1]
    co = np.asarray(res["co"], dtype=np.float64)
    side = float(np.mean(co[strip["a"] + strip["b"], axis]) - offset)
    return (axis, offset, S.partners(co, axis, offset)), (1.0 if side >= 0 else -1.0)


def moved_points(res, strip, kind: str, amount: float, project=None) -> np.ndarray:
    co = np.asarray(res["co"], dtype=np.float64)
    lines, stay = _plan(res, strip)
    goals = S.targets(co, lines, kind, amount, strip["closed"])
    mirror, side = _mirror(res, strip)
    return S.finish(co, S.move(co, lines, goals, stay), res["faces"], project, mirror, side)


def _view(points, faces):
    """Share (percent) of what a viewer sees that differs from your mesh: the camera check Organize uses."""
    guided = _g()
    source = guided.state_source()
    if source is None:
        return 0.0
    src = guided._source_data(source)
    if fl.get("look") is None or fl["look"][0] != src["sig"]:
        low, high = src["co"].min(axis=0), src["co"].max(axis=0)
        centre, reach = (low + high) / 2.0, (float(np.ptp(src["co"], axis=0).max()) or 1.0) * 0.55
        fl["look"] = (src["sig"], ORG._drain(autoretopo.pictures(src["co"], src["faces"], centre, reach, autoretopo.MAX_PIXELS)), centre, reach)
    _sig, reference, centre, reach = fl["look"]
    pixels = autoretopo.MAX_PIXELS
    wrong, shown = autoretopo.picture_counts(reference, ORG._drain(autoretopo.pictures(np.asarray(points, dtype=np.float64), faces, centre, reach, pixels)), 2.0 * reach / pixels)
    return 100.0 * wrong / max(shown, 1)


def _base(res, project=None, scope=None) -> TT.Base:
    """A Tidy Tool's Base for the result on screen, with Face Loops' own holds (borders, edges folding over SHARP_DEG,
    non-quads) and the mirror plane of a mirrored result."""
    co = np.asarray(res["co"], dtype=np.float64)
    _plan(res, {"a": [], "b": []})  # the pins
    job = (_g().state["auto_base"] or {}).get("job")
    mirror = getattr(job, "mirror", None)
    scope = np.ones(len(res["faces"]), dtype=bool) if scope is None else scope
    return TT.Base(co, res["faces"], scope, fl["stay"].copy(), project, TT.mean_edge(co, res["faces"]), None if mirror is None else (mirror[0], mirror[1]))


def _trial(res, trial, settings: dict, project=None, scope=None):
    """Run a `triadq.stripfix` tool on the result to the end: its Candidate (a refusal says why it can't)."""
    return TT.finish(trial.run(_base(res, project, scope), dict(trial.defaults, **settings)))


def lasso_faces(res, poly, to_screen, toward) -> np.ndarray:
    """Faces of the result inside a lasso drawn over it: their middles fall inside `poly` (screen points) as
    `to_screen(points)` places them, and they face the viewer and are not hidden, `toward(points)` being the way the
    view looks at each point. Faces seen nearly edge on are left out: rows can't be laid flat over them."""
    co = np.asarray(res["co"], dtype=np.float64)
    middle = np.asarray([co[list(f)].mean(axis=0) for f in res["faces"]])
    normal = np.asarray([np.cross(co[f[2]] - co[f[0]], co[f[-1]] - co[f[1]]) for f in res["faces"]])
    look = np.asarray(toward(middle), dtype=np.float64)
    facing = -(normal * look).sum(axis=1) > LASSO_FACING * np.linalg.norm(normal, axis=1) * np.linalg.norm(look, axis=1)
    keep = np.flatnonzero(facing & patch._inside_polygon(np.asarray(to_screen(middle), dtype=np.float64), np.asarray(poly, dtype=np.float64)))
    size = float(np.linalg.norm(np.ptp(co, axis=0))) or 1.0
    seen = []
    for i in keep.tolist():  # only what the viewer sees: the first face along the view through its middle
        found = hit(res, middle[i] - look[i] * size, look[i])
        if found is not None and (found[1] == i or np.linalg.norm(found[0] - middle[i]) < 0.02 * size):
            seen.append(i)
    return np.asarray(seen, dtype=np.int64)


def _strokes_in(context, res, ids):
    """The Flow Editor strokes that run over the area: they steer its rows."""
    from . import flowedit

    gs = context.scene.curiomesh_guided
    co = np.asarray(res["co"], dtype=np.float64)
    middle = np.asarray([co[list(res["faces"][i])].mean(axis=0) for i in ids])
    edge = float(np.sqrt(np.mean([np.linalg.norm(np.cross(co[f[2]] - co[f[0]], co[f[-1]] - co[f[1]])) for f in (res["faces"][i] for i in ids)]) / 2.0))
    out = []
    for s in flowedit.strokes(context, gs):
        d = np.linalg.norm(s["points"][:, None, :] - middle[None], axis=2).min(axis=1)
        if np.median(d) < 2.0 * edge:
            out.append(s)
    return out


def redraw_rows(context, ids, across=None) -> str:
    """Remake the faces `ids` of the result on screen as rows of quads at their own size, tied to the faces round them
    (`triadq.stripfix.RedrawInRows`), as one Undo step. `across`: the view's sideways direction in the result's space,
    which the rows follow. Returns an error in words, or ""."""
    res = _g().state["res"]
    if res is None:
        return "No result yet"
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) < SF.SMALLEST:
        return "Mark a bigger area: rows need at least a few quads each way"
    cand = _trial(res, SF.RedrawInRows(), {"faces": ids, "strokes": _strokes_in(context, res, ids), "across": across}, _project())
    if cand.refusal:
        return cand.refusal
    parent = np.asarray(cand.parent, dtype=np.int64)
    old = res["faces"]
    new = np.array([tuple(f) != tuple(old[q]) for f, q in zip(cand.faces, parent.tolist())], dtype=bool)
    out = _faces_changed(res, cand.co, cand.faces, parent)
    out["new"] = new
    before = loopscore.loop_quality(res["co"], res["faces"])["quality"]
    after = loopscore.loop_quality(out["co"], out["faces"])["quality"]
    commit(context, out, None, parent, name="Redraw in Rows")
    poles = Topology(out["co"], out["faces"]).poles
    inside = np.unique([v for f, n in zip(cand.faces, new) if n for v in f])
    fl["note"] = f"Redraw in Rows: {cand.note}, poles there {100.0 * poles[inside].mean():.0f}%. Loop Quality {before:.0f} -> {after:.0f}"
    return ""


def edit(context, kind: str, amount: float = 1.0):
    """Do one edit on the selected strip. Returns an error in words, or ""."""
    found = selected()
    if found is None:
        return "Pick a strip first"
    res, element, strip = found
    project = _project()
    if kind == "slide":
        commit(context, dict(res, co=moved_points(res, strip, kind, amount, project)), element, name=NAMES[kind])
        return ""
    if kind == "fix":
        numbers = S.stats(res["co"], strip)
        if not strip["closed"] and numbers["faces"] <= SHORT and not edit(context, "remove"):
            return ""  # a short dead end: gone, its neighbours joined
        kind, amount = ("straighten" if numbers["turn"] > WIGGLY else "even" if numbers["width"] > UNEVEN else ""), 1.0
        if not kind:
            return "This strip already looks clean"
        name = NAMES["fix"]
    else:
        name = NAMES[kind]
    if kind in ("straighten", "even"):
        cand = _trial(res, SF.Straighten() if kind == "straighten" else SF.EvenWidth(), {"strip": element, "amount": amount}, project)
        if cand.refusal:
            return cand.refusal
        commit(context, dict(res, co=cand.co), element, name=name)
        return ""
    if kind == "remove":
        cand = _trial(res, SF.RemoveDeadEnd(), {"strip": element})
        if cand.refusal:
            return cand.refusal
        commit(context, _faces_changed(res, cand.co, cand.faces, cand.parent), None, cand.parent, name=name)
        return ""
    strip = strips(res).walk(element, past_poles=True)  # a strip is split whole, or a face at its end breaks
    mirror, _side = _mirror(res, strip)
    group = [strip]
    if mirror is not None:
        other = S.mirror_element(strips(res), strip, mirror[2])
        if other >= 0:
            group.append(strips(res).walk(other, past_poles=True))
    _plan(res, strip)  # the pins
    try:
        co, faces, parent, (face, edge) = S.cut(res["co"], res["faces"], group, project)
    except ValueError as exc:
        return f"Can't add a strip here: {exc}"
    out = _faces_changed(res, co, faces, parent)
    element = strips(out).element(face, edge)
    new_strip = walk(out, element)
    lines, stay = _plan(out, new_strip, reach=2)  # the new strip and two quads each side spaced evenly
    out["co"] = S.finish(out["co"], S.move(out["co"], lines, [(0.0, 0.0)] * len(lines), stay, spread=True), out["faces"], project, *_mirror(out, new_strip))
    commit(context, out, element, parent, name=name)
    return ""


# --------------------------------------------------------------------------- picking


def _ray(context, event):
    """(origin, direction) of the mouse ray in the result's own space, or None off the 3D view."""
    guided = _g()
    preview = guided.preview_object()
    view = guided._SurfaceTool._view_region(None, context, event)
    if preview is None or view is None:
        return None
    from bpy_extras import view3d_utils

    coord = (event.mouse_x - view.x, event.mouse_y - view.y)
    inv = preview.matrix_world.inverted()
    origin = inv @ view3d_utils.region_2d_to_origin_3d(view, view.data, coord)
    return origin, (inv.to_3x3() @ view3d_utils.region_2d_to_vector_3d(view, view.data, coord)).normalized()


def hit(res, origin, ray):
    """(point, face) where the ray meets the result, or None."""
    _cache(res)
    if fl["tree"] is None:
        fl["tree"] = BVHTree.FromPolygons(np.asarray(res["co"]).tolist(), [list(f) for f in res["faces"]], all_triangles=False, epsilon=0.0)
    loc, _n, face, _d = fl["tree"].ray_cast(Vector(origin), Vector(ray))
    return None if face is None else (np.asarray(loc), int(face))


def element_at(res, origin, ray):
    """The element of the strip under a ray: the strip running along the quad edge nearest the hit, like Alt+click."""
    found = hit(res, origin, ray)
    if found is None or len(res["faces"][found[1]]) != 4:
        return None
    at, face = found
    p = np.asarray(res["co"])[list(res["faces"][face])]
    q = np.roll(p, -1, axis=0)
    t = np.clip(((at - p) * (q - p)).sum(axis=1) / np.maximum(((q - p) ** 2).sum(axis=1), 1e-30), 0.0, 1.0)
    k = int(np.argmin(np.linalg.norm(p + t[:, None] * (q - p) - at, axis=1)))
    return strips(res).element(face, k + 1)


def _under_mouse(context, event):
    res, ray = _g().state["res"], _ray(context, event)
    return None if res is None or ray is None else element_at(res, *ray)


def _navigation(event) -> bool:
    return event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt


def _has_result(context) -> bool:
    guided = _g()
    return guided.state["res"] is not None and guided.state["auto_base"] is not None and context.area is not None and context.area.type == "VIEW_3D"


def move_pole(context, vertex: int, goal) -> None:
    """Put a pole of the result at `goal` (a point on it); the quads round it follow."""
    res = _g().state["res"]
    _plan(res, {"a": [], "b": []})  # the pins
    new = S.move_vertex(res["co"], res["faces"], vertex, goal, fl["stay"])
    keep = fl["sel"][1] if fl["sel"] is not None and fl["sel"][0] is res else None
    commit(context, dict(res, co=S.finish(res["co"], new, res["faces"], _project(), *_mirror(res, {"a": [vertex], "b": [vertex]}))), keep, name="Move Pole")


def _screen(context, preview, points):
    """Window pixel positions of points of the result, or None when one is behind the view."""
    from bpy_extras import view3d_utils

    view = next((r for r in context.area.regions if r.type == "WINDOW"), None)
    rv3d = getattr(context.space_data, "region_3d", None)
    if view is None or rv3d is None:
        return None
    out = [view3d_utils.location_3d_to_region_2d(view, rv3d, preview.matrix_world @ Vector(p)) for p in np.asarray(points).tolist()]
    return None if any(p is None for p in out) else np.asarray([(p.x + view.x, p.y + view.y) for p in out])


def _pole_near(context, event):
    """The pole of the result within POLE_PIXELS of the mouse, or None."""
    res, preview = _g().state["res"], _g().preview_object()
    ray = _ray(context, event)
    if res is None or preview is None or ray is None:
        return None
    at = hit(res, *ray)
    poles = np.flatnonzero(Topology(res["co"], res["faces"]).poles)
    if at is None or not len(poles):
        return None
    near = poles[np.argsort(np.linalg.norm(np.asarray(res["co"])[poles] - at[0], axis=1))[:8]]  # only poles by the hit: none behind
    xy = _screen(context, preview, np.asarray(res["co"])[near])
    if xy is None:
        return None
    d = np.linalg.norm(xy - (event.mouse_x, event.mouse_y), axis=1)
    return int(near[np.argmin(d)]) if d.min() <= POLE_PIXELS else None


class CURIOMESH_OT_strip_pick(bpy.types.Operator):
    bl_idname = "curiomesh.strip_pick"
    bl_label = "Pick a Strip"
    bl_description = "Point at the result to light up a strip of quads, and click to select it. Drag the selected strip to move it, or drag a pole dot to move it along the surface"
    bl_options = {"REGISTER"}
    HINT = "Click a strip to select it. Drag the selected strip to move it, drag a dot to move a pole. Right-click or Enter: done"

    @classmethod
    def poll(cls, context):
        return _has_result(context)

    def invoke(self, context, event):
        context.scene.curiomesh_guided.organize_flow = True
        self.drag, self.goal = None, None
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(self.HINT)
        return {"RUNNING_MODAL"}

    def _done(self, context):
        fl["hover"] = fl["live"] = None
        context.workspace.status_text_set(None)
        overlay.tag_redraw()
        return {"FINISHED"}

    def _show(self, points) -> None:
        mesh = _g().preview_object().data
        mesh.vertices.foreach_set("co", np.asarray(points, dtype=np.float32).ravel())
        mesh.update()
        fl["live"] = points
        overlay.tag_redraw()

    def _start_drag(self, context, event) -> bool:
        res = _g().state["res"]
        preview = _g().preview_object()
        if res is None or preview is None or len(preview.data.vertices) != len(res["co"]):
            return False
        pole = _pole_near(context, event)
        if pole is not None:
            self.drag = {"kind": "pole", "res": res, "vertex": pole}
            context.workspace.status_text_set("Drag to move the pole. Let go to keep it, right-click to cancel.")
            return True
        found, under = selected(), _under_mouse(context, event)
        if found is None or under is None or strips(res).labels[under] != strips(res).labels[found[1]]:
            return False
        strip = found[2]
        co = np.asarray(res["co"])
        xy = _screen(context, preview, [co[strip["a"]].mean(axis=0), co[strip["b"]].mean(axis=0)])
        if xy is None:
            return False
        across = xy[1] - xy[0]
        width = max(float(np.linalg.norm(across)), 4.0)
        self.drag = {"kind": "strip", "res": res, "strip": strip, "start": np.asarray((event.mouse_x, event.mouse_y), dtype=np.float64), "along": across / width, "width": width, "project": _project(), "amount": 0.0}
        context.workspace.status_text_set("Drag to move the strip. Let go to keep it, right-click to cancel.")
        return True

    def _follow(self, context, event) -> None:
        d = self.drag
        if d["kind"] == "pole":
            ray = _ray(context, event)
            at = hit(d["res"], *ray) if ray is not None else None
            if at is not None:
                _plan(d["res"], {"a": [], "b": []})
                self._show(S.move_vertex(d["res"]["co"], d["res"]["faces"], d["vertex"], at[0], fl["stay"]))
                self.goal = at[0]
            return
        d["amount"] = float(np.clip((np.asarray((event.mouse_x, event.mouse_y)) - d["start"]) @ d["along"] / d["width"], -0.95, 0.95))
        self._show(moved_points(d["res"], d["strip"], "slide", d["amount"], d["project"]))

    def _end_drag(self, context, keep: bool) -> None:
        d, self.drag = self.drag, None
        self._show(d["res"]["co"])  # the shared preview mesh as it was; commit makes a new one
        fl["live"] = None
        if keep and d["kind"] == "pole" and self.goal is not None:
            move_pole(context, d["vertex"], self.goal)
            bpy.ops.ed.undo_push(message="Move Pole")
        elif keep and d["kind"] == "strip" and abs(d["amount"]) > 1e-3:
            edit(context, "slide", d["amount"])
            bpy.ops.ed.undo_push(message="Slide Strip")
        self.goal = None
        context.workspace.status_text_set(self.HINT)

    def modal(self, context, event):
        if self.drag is not None:
            if event.type == "MOUSEMOVE":
                self._follow(context, event)
            elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self._end_drag(context, True)
            elif event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
                self._end_drag(context, False)
            return {"RUNNING_MODAL"}
        if event.type in {"RET", "NUMPAD_ENTER", "ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            return self._done(context)
        if _navigation(event):
            return {"PASS_THROUGH"}
        if _g()._SurfaceTool._view_region(None, context, event) is None:
            return {"PASS_THROUGH"}
        if event.type == "MOUSEMOVE":
            hover = _under_mouse(context, event)
            if hover != fl["hover"]:
                fl["hover"] = hover
                overlay.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            if self._start_drag(context, event):
                return {"RUNNING_MODAL"}
            element = _under_mouse(context, event)
            fl["sel"] = None if element is None else (_g().state["res"], element)
            overlay.tag_redraw()
        return {"RUNNING_MODAL"}


def opening(res, centre, right, rx=None, ry=None):
    """(normal, width, height, quad width) of the opening at `centre` on the result: the given oval, else the one read
    off the source there, else one 3 quads wide and 2 high."""
    h = RG.quad_size(res["co"], res["faces"], centre, 0.05 * float(np.linalg.norm(np.ptp(np.asarray(res["co"]), axis=0))))
    n = RG.facing(res["co"], res["faces"], centre, 3.0 * h)
    if rx is None or ry is None or max(rx, ry) <= 0.0:
        source = _g().state_source()
        src = None if source is None else _g()._source_data(source)
        found = None if src is None else RG.find_rim(src["co"], src["tris"], centre, n, right, h)
        rx, ry = found or (1.5 * h, 1.0 * h)
    return n, float(rx), float(ry), h


def ring(context, centre, normal, right, rx=None, ry=None, band: float = RG.BAND, rings: int = 0) -> str:
    """Rings of quads round the opening at `centre` (a point on the result), built into it as one step: the rim is
    the `rx` by `ry` oval (None: read off the source), the faces up to `band` quad widths past it are redone. Returns
    an error in words, or ""."""
    res = _g().state["res"]
    if res is None:
        return "No result yet"
    centre = np.asarray(centre, dtype=np.float64)
    n, rx, ry, _h = opening(res, centre, right, rx, ry)
    try:
        made = RG.build(res["co"], res["faces"], centre, n if normal is None else normal, right, rx, ry, band, rings or None, _project())
    except ValueError as exc:
        return f"Can't ring round this: {exc}"
    out = _faces_changed(res, made["co"], made["faces"], made["parent"])
    seen, now = _view(res["co"], res["faces"]), _view(out["co"], out["faces"])
    if now > seen + RING_VIEW:
        return f"Rings here would change the look too much (view change {seen:.1f}% -> {now:.1f}%); set a smaller reach"
    commit(context, out, None, made["parent"], name="Ring Round This")
    fl["note"] = f"Ring Round This: {made['rings']} rings of {made['count']} quads, {made['corners']} poles in the band; faces {len(res['faces'])} -> {len(out['faces'])}"
    return ""


class CURIOMESH_OT_strip_ring(bpy.types.Operator):
    bl_idname = "curiomesh.strip_ring"
    bl_label = "Ring Round This"
    bl_description = "Click an eye, a mouth or a hole (or drag from its middle out to its edge), then move out to set how far the new faces reach and click. Its quads are rebuilt as clean rings round it, joined to the faces round them"
    bl_options = {"REGISTER", "UNDO"}
    HINT = "Click the middle of an eye or mouth, or drag out to its edge (sideways: width, up: height). Right-click: cancel"
    BAND_HINT = "Move the mouse out or in to set how far the new rings reach, then click. Right-click: cancel"

    centre: bpy.props.FloatVectorProperty(size=3, options={"HIDDEN", "SKIP_SAVE"})
    normal: bpy.props.FloatVectorProperty(size=3, options={"HIDDEN", "SKIP_SAVE"})
    right: bpy.props.FloatVectorProperty(size=3, default=(1.0, 0.0, 0.0), options={"HIDDEN", "SKIP_SAVE"})
    width: bpy.props.FloatProperty(name="Width", description="Half the opening's width. 0 finds it on your mesh", default=0.0, min=0.0, subtype="DISTANCE")
    height: bpy.props.FloatProperty(name="Height", description="Half the opening's height. 0 finds it on your mesh", default=0.0, min=0.0, subtype="DISTANCE")
    band: bpy.props.FloatProperty(name="Reach", description="How far past the opening's edge the faces are redone, in quad widths. Faces further out are never touched", default=RG.BAND, min=1.0, max=6.0)
    rings: bpy.props.IntProperty(name="Rings", description="How many rings go round the opening. 0 fits as many as there is room for, 2 to 4", default=0, min=0, max=4)

    @classmethod
    def poll(cls, context):
        return _has_result(context) and _g().state_source() is not None

    def execute(self, context):
        normal = tuple(self.normal) if any(self.normal) else None
        error = ring(context, tuple(self.centre), normal, tuple(self.right), self.width or None, self.height or None, self.band, self.rings)
        fl["ring_show"] = None
        overlay.tag_redraw()
        if error:
            self.report({"WARNING"}, error)
            return {"CANCELLED"}
        return {"FINISHED"}

    def invoke(self, context, event):
        self.stage, self.found = 0, None
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(self.HINT)
        return {"RUNNING_MODAL"}

    def _stop(self, context, result):
        fl["ring_show"] = None
        context.workspace.status_text_set(None)
        overlay.tag_redraw()
        return result

    def _show(self):
        _n, rx, ry, h = self.found
        reach = self.band * h
        fl["ring_show"] = (np.asarray(self.centre), self.u, self.v, [(rx, ry), (rx + reach, ry + reach)])
        overlay.tag_redraw()

    def _drag(self, event):
        dx, dy = abs(event.mouse_x - self.start[0]) * self.per_pixel, abs(event.mouse_y - self.start[1]) * self.per_pixel
        size = max(dx, dy)
        return None if size < 6.0 * self.per_pixel else (max(dx, 0.4 * size), max(dy, 0.4 * size))

    def _press(self, context, event, res):
        ray = _ray(context, event)
        at = None if ray is None else hit(res, *ray)
        if at is None:
            return
        preview = _g().preview_object()
        inv = preview.matrix_world.inverted().to_3x3()
        right = np.asarray(inv @ (context.space_data.region_3d.view_rotation @ Vector((1, 0, 0))))
        right /= max(float(np.linalg.norm(right)), 1e-30)
        one = _screen(context, preview, [at[0], at[0] + right])
        if one is None:
            return
        self.per_pixel = 1.0 / max(float(np.linalg.norm(one[1] - one[0])), 1e-9)  # the result's units per pixel here
        self.centre, self.right, self.start = tuple(at[0]), tuple(right), (event.mouse_x, event.mouse_y)
        self.found = opening(res, at[0], right, 1.0, 1.0)
        n = self.found[0]
        self.normal = tuple(n)
        u = right - (right @ n) * n
        self.u = u / max(float(np.linalg.norm(u)), 1e-30)
        self.v = np.cross(n, self.u)
        self.band, self.stage = RG.BAND, 1

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            return self._stop(context, {"CANCELLED"})
        if _navigation(event):
            return {"PASS_THROUGH"}
        res = _g().state["res"]
        if res is None:
            return self._stop(context, {"CANCELLED"})
        if self.stage == 0:
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                self._press(context, event, res)
            return {"RUNNING_MODAL"}
        if self.stage == 1:
            oval = self._drag(event)
            if event.type == "MOUSEMOVE" and oval is not None:
                self.found = (self.found[0], *oval, self.found[3])
                self._show()
            elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
                if oval is None:  # a click: read the opening off the source
                    self.found = opening(res, np.asarray(self.centre), np.asarray(self.right))
                self.width, self.height = self.found[1], self.found[2]
                self.stage = 2
                self.out = float(np.hypot(event.mouse_x - self.start[0], event.mouse_y - self.start[1]))
                self._show()
                context.workspace.status_text_set(self.BAND_HINT)
            return {"RUNNING_MODAL"}
        if event.type == "MOUSEMOVE":
            far = float(np.hypot(event.mouse_x - self.start[0], event.mouse_y - self.start[1]))
            self.band = float(np.clip(RG.BAND + (far - self.out) * self.per_pixel / self.found[3], 1.0, 6.0))
            self._show()
            context.workspace.status_text_set(f"Reach: {self.band:.1f} quad widths. " + self.BAND_HINT)
        elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            context.workspace.status_text_set(None)
            return self.execute(context)
        return {"RUNNING_MODAL"}


def _view_mappers(context, preview):
    """(to_screen, toward) for `lasso_faces`: result-space points to region pixels, and the way the view looks at them."""
    region, rv3d = context.region, context.space_data.region_3d
    matrix = np.asarray(rv3d.perspective_matrix @ preview.matrix_world)
    inv = preview.matrix_world.inverted()
    size = np.array((region.width, region.height), dtype=np.float64)

    def to_screen(points):
        clip = np.c_[points, np.ones(len(points))] @ matrix.T
        return (clip[:, :2] / np.maximum(np.abs(clip[:, 3:]), 1e-30) * np.sign(clip[:, 3:]) + 1.0) * 0.5 * size

    if rv3d.is_perspective:
        eye = np.asarray(inv @ rv3d.view_matrix.inverted().translation)
        return to_screen, lambda points: (points - eye) / np.maximum(np.linalg.norm(points - eye, axis=1, keepdims=True), 1e-30)
    look = np.asarray(inv.to_3x3() @ (rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))))
    look /= max(float(np.linalg.norm(look)), 1e-30)
    return to_screen, lambda points: np.tile(look, (len(points), 1))


def _across(context):
    """The view's sideways direction in the result's space (None off a 3D view): Redraw in Rows lays rows along it."""
    preview = _g().preview_object()
    rv3d = getattr(context.space_data, "region_3d", None)
    if preview is None or rv3d is None:
        return None
    right = np.asarray(preview.matrix_world.inverted().to_3x3() @ (rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))))
    return right / max(float(np.linalg.norm(right)), 1e-30)


def _edit_selection(context):
    """Faces of the result selected in Edit Mode on the preview, or None (back in Object Mode either way)."""
    preview = _g().preview_object()
    res = _g().state["res"]
    if preview is None or res is None or context.mode != "EDIT_MESH" or context.edit_object is not preview:
        return None
    bpy.ops.object.mode_set(mode="OBJECT")
    picked = np.zeros(len(preview.data.polygons), dtype=bool)
    preview.data.polygons.foreach_get("select", picked)
    return np.flatnonzero(picked) if len(picked) == len(res["faces"]) and picked.any() else None


def _part_at(res, face: int) -> np.ndarray:
    """The faces of the part a face is on (the result's parts, else its loose piece)."""
    piece = segments.seg.get("piece")
    if piece is not None and len(piece) == len(res["faces"]):
        return np.flatnonzero(np.asarray(piece) == piece[face])
    return next(c for c in patch.mask_components(res["faces"], np.ones(len(res["faces"]), dtype=bool)) if face in c)


class CURIOMESH_OT_redraw_rows(bpy.types.Operator):
    bl_idname = "curiomesh.redraw_rows"
    bl_label = "Redraw in Rows"
    bl_description = "Draw round an area of the result (or click a part, or select faces in Edit Mode first) and its quads are laid out again as long straight rows at the same size. The rows run across the screen the way you look at it, bent by the surface and any lines you drew, and the faces round it stay as they are"
    bl_options = {"REGISTER", "UNDO"}
    HINT = "Drag round the area to redraw in rows, or click a part for all of it. Right-click or Esc: cancel"

    @classmethod
    def poll(cls, context):
        return _has_result(context) or (context.mode == "EDIT_MESH" and context.edit_object is _g().preview_object())

    def _run(self, context, ids):
        fl["lasso"] = None
        context.workspace.status_text_set(None)
        overlay.tag_redraw()
        error = redraw_rows(context, ids, _across(context))
        if error:
            self.report({"WARNING"}, error)
            return {"CANCELLED"}
        return {"FINISHED"}

    def execute(self, context):
        """From a script or the redo panel: the faces selected in Edit Mode."""
        ids = _edit_selection(context)
        if ids is None:
            self.report({"WARNING"}, "Select the faces to redraw in Edit Mode on the result first")
            return {"CANCELLED"}
        return self._run(context, ids)

    def invoke(self, context, event):
        ids = _edit_selection(context)
        if ids is not None:
            return self._run(context, ids)
        fl["lasso"] = []
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(self.HINT)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            fl["lasso"] = None
            context.workspace.status_text_set(None)
            overlay.tag_redraw()
            return {"CANCELLED"}
        if fl["lasso"] is None:
            return {"CANCELLED"}
        if _navigation(event) and not fl["lasso"]:
            return {"PASS_THROUGH"}
        here = (event.mouse_region_x, event.mouse_region_y)
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            fl["lasso"] = [here]
        elif event.type == "MOUSEMOVE" and fl["lasso"]:
            fl["lasso"].append(here)
            overlay.tag_redraw()
        elif event.type == "LEFTMOUSE" and event.value == "RELEASE" and fl["lasso"]:
            res, preview = _g().state["res"], _g().preview_object()
            path = np.asarray(fl["lasso"], dtype=np.float64)
            if float(np.ptp(path, axis=0).max()) < 8.0:  # a click: the whole part under it
                ray = _ray(context, event)
                found = hit(res, *ray) if ray is not None else None
                if found is None:
                    fl["lasso"] = []
                    return {"RUNNING_MODAL"}
                return self._run(context, _part_at(res, found[1]))
            return self._run(context, lasso_faces(res, path, *_view_mappers(context, preview)))
        return {"RUNNING_MODAL"}


class _Edit:
    bl_options = {"REGISTER", "UNDO"}
    KIND = ""

    @classmethod
    def poll(cls, context):
        return _has_result(context) and fl["sel"] is not None

    def execute(self, context):
        error = edit(context, self.KIND, getattr(self, "amount", 1.0))
        if error:
            self.report({"WARNING"}, error)
            return {"CANCELLED"}
        return {"FINISHED"}


class CURIOMESH_OT_strip_slide(_Edit, bpy.types.Operator):
    bl_idname = "curiomesh.strip_slide"
    bl_label = "Slide"
    bl_description = "Drag the mouse left or right to move the selected strip toward one of its neighbours, then click. The quads beside it widen or narrow to make room"
    KIND = "slide"

    amount: bpy.props.FloatProperty(name="Amount", description="How far, as a share of the next quad's width (minus: the other way)", default=0.0, min=-0.95, max=0.95, subtype="FACTOR")

    def invoke(self, context, event):
        found = selected()
        preview = _g().preview_object()
        if found is None or preview is None or len(preview.data.vertices) != len(found[0]["co"]):
            return self.execute(context)
        self.res, _e, self.strip = found
        self.project = _project()
        self.start = event.mouse_x
        self.sign = self._screen_sign(context, preview)
        self.mesh = preview.data
        self.amount = 0.0
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _screen_sign(self, context, preview) -> float:
        """+1 when the strip's b side is to the right of its a side on screen, so the strip follows the mouse."""
        from bpy_extras import view3d_utils

        view = next((r for r in context.area.regions if r.type == "WINDOW"), None)
        rv3d = getattr(context.space_data, "region_3d", None)
        if view is None or rv3d is None:
            return 1.0
        co = np.asarray(self.res["co"])
        at = [view3d_utils.location_3d_to_region_2d(view, rv3d, preview.matrix_world @ Vector(co[s].mean(axis=0))) for s in (self.strip["a"], self.strip["b"])]
        return 1.0 if None in at or at[1].x >= at[0].x else -1.0

    def _status(self, context):
        context.workspace.status_text_set(f"Slide: {self.amount:+.2f}  Move the mouse left/right  Click/Enter: done  Right click/Esc: cancel")

    def _show(self, points) -> None:
        self.mesh.vertices.foreach_set("co", np.asarray(points, dtype=np.float32).ravel())
        self.mesh.update()
        fl["live"] = points
        overlay.tag_redraw()

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            self.amount = float(np.clip(self.sign * (event.mouse_x - self.start) / DRAG_PIXELS, -0.95, 0.95))
            self._show(moved_points(self.res, self.strip, "slide", self.amount, self.project))
            self._status(context)
            return {"RUNNING_MODAL"}
        if _navigation(event):
            return {"PASS_THROUGH"}
        done = event.value == "PRESS" and event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"}
        stop = event.value == "PRESS" and event.type in {"RIGHTMOUSE", "ESC"}
        if done or stop:
            self._show(self.res["co"])  # the shared preview mesh as it was; commit makes a new one
            fl["live"] = None
            context.workspace.status_text_set(None)
            return self.execute(context) if done and abs(self.amount) > 1e-4 else {"CANCELLED"}
        return {"RUNNING_MODAL"}


class CURIOMESH_OT_strip_straighten(_Edit, bpy.types.Operator):
    bl_idname = "curiomesh.strip_straighten"
    bl_label = "Straighten"
    bl_description = "Smooths the wiggles out of the selected strip, so it runs in a clean line across the surface. Only the quads beside it move"
    KIND = "straighten"

    amount: bpy.props.FloatProperty(name="Amount", description="How much to straighten: 1 takes out every wiggle", default=1.0, min=0.0, max=1.0, subtype="FACTOR")


class CURIOMESH_OT_strip_even(_Edit, bpy.types.Operator):
    bl_idname = "curiomesh.strip_even"
    bl_label = "Even Width"
    bl_description = "Makes the selected strip the same width all along. The quads beside it widen or narrow to make room"
    KIND = "even"

    amount: bpy.props.FloatProperty(name="Amount", description="How much: 1 makes every quad of the strip the same width", default=1.0, min=0.0, max=1.0, subtype="FACTOR")


class CURIOMESH_OT_strip_fix(_Edit, bpy.types.Operator):
    bl_idname = "curiomesh.strip_fix"
    bl_label = "Fix This"
    bl_description = "Picks the fix the selected strip needs and does it: straightens a wiggly strip, else evens out an uneven one, or takes away a short dead end. One step, one Undo"
    KIND = "fix"


class CURIOMESH_OT_strip_fix_all(tidylive.LiveTool, bpy.types.Operator):
    bl_idname = "curiomesh.strip_fix_all"
    bl_label = "Fix Every Strip"
    bl_description = "Finds every wiggly strip and every strip of uneven width in Work On and shows the fix of each first. Enter keeps the ones that got better"
    bl_options = {"REGISTER", "UNDO"}
    trial = autofix.every(SF.FixStrips)

    amount: bpy.props.FloatProperty(name="Amount", description="How far each strip goes toward straight and even. 1 goes all the way", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    keep_going: bpy.props.BoolProperty(name="Keep Going", description="Runs again on the kept result while a run still makes a strip better. Each run is one step in History", default=False)

    @classmethod
    def poll(cls, context):
        return _g().state["res"] is not None and _g().state["auto_base"] is not None  # no 3D view needed for F9 and scripts


class CURIOMESH_OT_strip_add(_Edit, bpy.types.Operator):
    bl_idname = "curiomesh.strip_add"
    bl_label = "Add Strip"
    bl_description = "Splits the selected strip down its middle, so there is one more strip of quads there. Every face stays a quad"
    KIND = "add"


class CURIOMESH_OT_strip_remove(_Edit, bpy.types.Operator):
    bl_idname = "curiomesh.strip_remove"
    bl_label = "Remove Strip"
    bl_description = "Takes the selected strip away by joining its two sides. Refused where a face next to it would stop being a quad"
    KIND = "remove"


CLASSES = (CURIOMESH_OT_strip_pick, CURIOMESH_OT_strip_ring, CURIOMESH_OT_redraw_rows, CURIOMESH_OT_strip_fix, CURIOMESH_OT_strip_fix_all, CURIOMESH_OT_strip_slide, CURIOMESH_OT_strip_straighten, CURIOMESH_OT_strip_even, CURIOMESH_OT_strip_add, CURIOMESH_OT_strip_remove)


def draw(layout, gs) -> None:
    """The Face Loops tools, under Tidy Up in Adjust."""
    col = layout.column(align=True)
    col.label(text="Face Loops", **icons.kw("view_flow", "FORCE_CURVE"))
    row = col.row(align=True)
    row.operator("curiomesh.strip_pick", icon="RESTRICT_SELECT_OFF")
    row.operator("curiomesh.strip_ring", **icons.kw("tool_ring", "MESH_CIRCLE"))
    row = col.row(align=True)
    row.operator("curiomesh.strip_fix_all", icon="SHADERFX")
    row.operator("curiomesh.redraw_rows", icon="MOD_LATTICE")
    if fl["note"]:
        overlay.wrap_label(col, fl["note"], width=36, limit=1)
    found = selected()
    if found is None:
        return
    res, _element, strip = found
    col.operator("curiomesh.strip_fix", icon="SHADERFX")
    row = col.row(align=True)
    row.operator("curiomesh.strip_slide", icon="ARROW_LEFTRIGHT")
    row.operator("curiomesh.strip_straighten", icon="IPO_LINEAR")
    row.operator("curiomesh.strip_even", icon="ALIGN_JUSTIFY")
    row = col.row(align=True)
    row.operator("curiomesh.strip_add", **icons.kw("tool_more", "ADD"))
    row.operator("curiomesh.strip_remove", **icons.kw("tool_less", "REMOVE"))
    sub = layout.column(align=True)
    sub.active = False
    overlay.wrap_label(sub, S.describe(S.stats(res["co"], strip)), width=36, limit=2)


# --------------------------------------------------------------------------- the picked strip on screen


def _strip_batches(points, strip, matrix, colour):
    import gpu
    from gpu_extras.batch import batch_for_shader

    guided = _g()
    q = np.asarray([[a0, b0, b1, a1] for a0, b0, a1, b1 in zip(strip["a"], strip["b"], strip["a"][1:] + strip["a"][:1], strip["b"][1:] + strip["b"][:1])][: len(strip["faces"])], dtype=np.int64).reshape(-1, 4)
    p = np.asarray(points, dtype=np.float64)[q]
    n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    lift = 0.02 * float(np.linalg.norm(p[:, 1] - p[:, 0], axis=1).mean())  # just off the surface, so it isn't hidden in it
    p = p + lift * n[:, None, :]
    tris = p[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
    rails = p[:, [0, 3, 1, 2]].reshape(-1, 3)
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    world = lambda x: guided._world(matrix, x).astype(np.float32)  # noqa: E731
    return batch_for_shader(shader, "TRIS", {"pos": world(tris)}), batch_for_shader(shader, "LINES", {"pos": world(rails)}), colour


def _draw_ring(preview) -> None:
    """The opening's rim and how far the new rings reach, as two ovals while Ring Round This is dragged."""
    import gpu
    from gpu_extras.batch import batch_for_shader

    centre, u, v, ovals = fl["ring_show"]
    t = np.linspace(0.0, 2.0 * np.pi, 65)[:, None]
    n = np.cross(u, v)
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.line_width_set(2.0)
    shader.bind()
    for (a, b), colour in zip(ovals, ((1.0, 0.55, 0.1, 1.0), (1.0, 1.0, 1.0, 0.8))):
        line = centre + a * np.cos(t) * u + b * np.sin(t) * v + 0.1 * min(a, b) * n
        pts = _g()._world(preview.matrix_world, line).astype(np.float32)
        shader.uniform_float("color", colour)
        batch_for_shader(shader, "LINE_STRIP", {"pos": pts}).draw(shader)
    gpu.state.line_width_set(1.0)


def _draw() -> None:
    try:
        guided = _g()
        res, preview = guided.state["res"], guided.preview_object()
        if res is None or preview is None:
            return
        if fl["ring_show"] is not None:
            _draw_ring(preview)
        found = selected()
        shown = []
        if fl["hover"] is not None and fl["res"] is res and (found is None or fl["hover"] != found[1]):
            shown.append((fl["hover"], res["co"], HOVER_COLOUR))
        if found is not None:
            shown.append((found[1], res["co"] if fl["live"] is None else fl["live"], SEL_COLOUR))
        if not shown:
            return
        import gpu

        matrix = tuple(round(x, 5) for row in preview.matrix_world for x in row)
        key = (id(res), matrix, tuple((e, id(p)) for e, p, _c in shown))
        if fl["batch"] is None or fl["batch"][0] != key:
            fl["batch"] = (key, [_strip_batches(p, walk(res, e), preview.matrix_world, c) for e, p, c in shown])
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.line_width_set(3.0)
        shader.bind()
        for tris, rails, colour in fl["batch"][1]:
            shader.uniform_float("color", colour)
            tris.draw(shader)
            shader.uniform_float("color", (*colour[:3], 1.0))
            rails.draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


def _draw_lasso() -> None:
    try:
        path = fl["lasso"]
        if not path or len(path) < 2:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        line = [(float(x), float(y)) for x, y in path + path[:1]]
        gpu.state.line_width_set(2.0)
        shader.bind()
        shader.uniform_float("color", SEL_COLOUR[:3] + (1.0,))
        batch_for_shader(shader, "LINE_STRIP", {"pos": line}).draw(shader)
        gpu.state.line_width_set(1.0)
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


_handles: list = []


def register() -> None:
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW"))
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_lasso, (), "WINDOW", "POST_PIXEL"))


def unregister() -> None:
    while _handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), "WINDOW")
        except Exception:
            pass
    cancel()
