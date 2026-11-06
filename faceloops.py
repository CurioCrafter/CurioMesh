# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Face Loops: pick one strip of quads of the result on screen and adjust it (the Blender side of `triadq.strips`).

Pick a Strip shows the strip under the mouse and selects it on click. Slide (drag the mouse), Straighten, Even Width,
Add Strip and Remove Strip change it on the result: each edit is one Ctrl+Z step. The result lives in Python, so each
edit is kept in a short history keyed by a number on the scene (`strip_step`) that Blender's undo restores; the next
draw or edit puts the result of that step back on screen.
"""

from __future__ import annotations

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import autoretopo, icons, overlay, region, segments
from . import organize as ORG
from .triadq import loopscore, patch, segment
from .triadq import organize as O
from .triadq import strips as S
from .triadq.patterns import Topology

WIGGLY = 4.0  # degrees per quad: Fix This straightens a strip that turns more
UNEVEN = 0.15  # Fix This evens out a strip whose width varies more than this
SHORT = 3  # quads: Fix This takes away an open strip this short (a dead end between two poles)
VIEW_COST = 1.0  # points of view change Fix Every Strip may spend
SHARP_DEG = 60.0  # an edge folding more than this holds its vertices: Tidy Up's 35 held 40% of a coarse bust's face
RING_REACH = 2.5  # quad widths round a ring that are redone: the Flow Editor's 6 took the nose with an eye on a coarse result
LASSO_FACING = 0.25  # a lasso takes faces turned at least this much toward the view (cosine): not the sides it only grazes
POLE_PIXELS = 14  # how near the mouse has to be to a pole dot to drag it
DRAG_PIXELS = 250.0  # mouse travel that slides a strip a whole quad over
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
    "base": None,  # the result (guided.state["auto_base"]) the history belongs to
    "hist": {},  # strip_step -> (result, selected element)
    "batch": None,
    "note": "",
    "lasso": None,  # region pixels of the Redraw in Rows lasso while it is drawn
    "look": None,  # (source signature, its pictures, centre, reach) for the camera check  # the last Fix Every Strip's Loop Quality change
}


def _g():
    from . import guided  # late: guided imports this module

    return guided


def cancel() -> None:
    fl.update(res=None, strips=None, tree=None, stay=None, go=None, walks={}, hover=None, sel=None, live=None, base=None, hist={}, batch=None, note="", lasso=None)


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


def sync(context) -> None:
    """Put the result of the scene's `strip_step` back on screen after an undo or redo."""
    guided = _g()
    gs = getattr(context.scene, "curiomesh_guided", None)
    base = guided.state["auto_base"]
    if gs is None or base is None:
        return
    if fl["base"] is not base:
        fl.update(base=base, hist={gs.strip_step: (base["out"], None)})
        return
    entry = fl["hist"].get(gs.strip_step)
    if entry is None or entry[0] is base["out"]:
        return
    if not any(e[0] is base["out"] for e in fl["hist"].values()):  # another tool changed the result: start from it
        fl["hist"] = {gs.strip_step: (base["out"], None)}
        return
    base["out"] = guided.state["res"] = entry[0]
    guided.state["draw_key"] = None
    fl["sel"] = (entry[0], entry[1])
    overlay.tag_redraw()


def commit(context, out, element, parent=None) -> None:
    """Show `out` as the result and keep it as the next undo step. `parent`: the old face of each face, when faces changed."""
    guided = _g()
    gs = context.scene.curiomesh_guided
    base = guided.state["auto_base"]
    if parent is not None:
        piece = segments.seg.get("piece")
        if piece is not None and len(piece) == len(base["out"]["faces"]):
            segments.seg.update(piece=np.asarray(piece)[parent], batch_key=None, bad_key=None, tris=None, hover=None)
    step = gs.strip_step
    fl["hist"] = {k: v for k, v in fl["hist"].items() if k <= step}
    fl["hist"][step + 1] = (out, element)
    gs.strip_step = step + 1
    base["out"] = guided.state["res"] = out
    guided._build_preview(context, guided.state_source(), out, uvs=False)
    guided.state["draw_key"] = None
    fl["sel"], fl["live"] = (out, element), None
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


def fix_all(context, rounds: int = 3, bend: float = 40.0, most: float = 0.75) -> str:
    """Straighten every wiggly strip of the result, worst first, a few times over, then even out the uneven ones half
    way, as one step. Vertices where the surface bends stay, no vertex moves more than 3/4 of an edge, and it is kept only
    when Loop Quality goes up."""
    sync(context)
    guided = _g()
    res = guided.state["res"]
    if res is None:
        return "No result yet"
    st = strips(res)
    co = np.asarray(res["co"], dtype=np.float64)
    _plan(res, {"a": [], "b": []})
    stay, go = fl["stay"] | S.bent(co, res["faces"], bend), fl["go"]
    plans = []
    for label in np.unique(st.labels).tolist():
        if st.sizes[label] >= 2:
            strip = st.walk(int(np.flatnonzero(st.labels == label)[0]))
            plans.append((strip, S.rung_lines(strip, go, len(co), stay)))
    for _ in range(rounds):
        for strip, lines in sorted(plans, key=lambda p: -S.stats(co, p[0])["turn"]):
            if lines and S.stats(co, strip)["turn"] > WIGGLY:
                co = S.move(co, lines, S.targets(co, lines, "straighten", 1.0, strip["closed"]), stay)
    for strip, lines in plans:
        if lines and S.stats(co, strip)["width"] > UNEVEN:
            co = S.move(co, lines, S.targets(co, lines, "even", 0.5, strip["closed"]), stay)
    co = S.capped(res["co"], co, res["faces"], most)
    mirror, side = _mirror(res, {"a": [0], "b": [0]})
    co = S.finish(res["co"], co, res["faces"], _project(), mirror, side)
    before = loopscore.loop_quality(res["co"], res["faces"])["quality"]
    seen = _view(res["co"], res["faces"])
    full = co
    for part in (1.0, 0.5, 0.25):  # as much of the fix as keeps the look within VIEW_COST points
        co = res["co"] + part * (full - res["co"])
        after = loopscore.loop_quality(co, res["faces"])["quality"]
        if _view(co, res["faces"]) <= seen + VIEW_COST:
            break
    else:
        return "Straightening the strips here would change the look too much (the rows run over detail)"
    if after <= before:
        return f"Straightening every strip would not make the rows better here (Loop Quality {before:.0f} -> {after:.0f})"
    keep = fl["sel"][1] if fl["sel"] is not None and fl["sel"][0] is res else None
    commit(context, dict(res, co=co), keep)
    fl["note"] = f"Fix Every Strip: Loop Quality {before:.0f} -> {after:.0f}"
    return ""


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


def redraw_rows(context, ids) -> str:
    """Remake the faces `ids` of the result on screen as rows of quads at their own size, tied to the faces round them
    (`triadq.rowfill`), as one Undo step. Returns an error in words, or ""."""
    from . import flowstroke as F

    sync(context)
    res = _g().state["res"]
    if res is None:
        return "No result yet"
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) < 8:
        return "Mark a bigger area: rows need at least a few quads each way"
    try:
        co, faces, mat, new, area, bad = F.rows(res["co"], res["faces"], res["mat"], ids, _strokes_in(context, res, ids), _project())
    except ValueError as exc:
        return f"Can't redraw this area in rows: {exc}"
    if segment.open_edges(faces) > segment.open_edges(res["faces"]):
        return "The rows did not close on the faces round them, so the result stays as it was"
    old = np.asarray([np.asarray(res["co"])[list(res["faces"][i])].mean(axis=0) for i in area])
    kept = np.setdiff1d(np.arange(len(res["faces"])), area)
    born = np.asarray([co[list(f)].mean(axis=0) for f, n in zip(faces, new) if n]).reshape(-1, 3)
    parent = np.empty(len(faces), dtype=np.int64)
    parent[~new] = kept
    parent[new] = area[np.argmin(np.linalg.norm(born[:, None] - old[None], axis=2), axis=1)] if len(born) else 0
    out = _faces_changed(res, co, faces, parent)
    out["mat"], out["new"] = np.asarray(mat), new
    before = loopscore.loop_quality(res["co"], res["faces"])["quality"]
    after = loopscore.loop_quality(out["co"], out["faces"])["quality"]
    commit(context, out, None, parent)
    poles = Topology(out["co"], out["faces"]).poles
    inside = np.unique([v for f, n in zip(faces, new) if n for v in f])
    fl["note"] = f"Redraw in Rows: {len(area)} faces -> {int(new.sum())}, poles there {100.0 * poles[inside].mean():.0f}%. Loop Quality {before:.0f} -> {after:.0f}" + (f". {bad} folded faces to check" if bad else "")
    return ""


def edit(context, kind: str, amount: float = 1.0):
    """Do one edit on the selected strip. Returns an error in words, or ""."""
    if kind == "fix_all":
        return fix_all(context)
    sync(context)
    found = selected()
    if found is None:
        return "Pick a strip first"
    res, element, strip = found
    project = _project()
    if kind in ("slide", "straighten", "even"):
        new = moved_points(res, strip, kind, amount, project)
        commit(context, dict(res, co=new), element)
        return ""
    if kind == "fix":
        numbers = S.stats(res["co"], strip)
        if not strip["closed"] and numbers["faces"] <= SHORT and not edit(context, "remove"):
            return ""  # a short dead end: gone, its neighbours joined
        out = res
        if numbers["turn"] > WIGGLY:
            out = dict(out, co=moved_points(out, strip, "straighten", 1.0, project))
        elif numbers["width"] > UNEVEN:
            out = dict(out, co=moved_points(out, strip, "even", 1.0, project))
        if out is res:
            return "This strip already looks clean"
        commit(context, out, element)
        return ""
    strip = strips(res).walk(element, past_poles=True)  # a strip is split or joined whole, or a face at its end breaks
    mirror, _side = _mirror(res, strip)
    group = [strip]
    if mirror is not None:
        other = S.mirror_element(strips(res), strip, mirror[2])
        if other >= 0:
            group.append(strips(res).walk(other, past_poles=True))
    _plan(res, strip)  # the pins
    try:
        if kind == "add":
            co, faces, parent, (face, edge) = S.cut(res["co"], res["faces"], group, project)
        else:
            co, faces, parent = S.collapse(res["co"], res["faces"], group, fl["stay"])
    except ValueError as exc:
        return f"Can't {'add a strip here' if kind == 'add' else 'remove this strip'}: {exc}"
    out = _faces_changed(res, co, faces, parent)
    element = None
    if kind == "add":
        element = strips(out).element(face, edge)
        new_strip = walk(out, element)
        lines, stay = _plan(out, new_strip, reach=2)  # the new strip and two quads each side spaced evenly
        out["co"] = S.finish(out["co"], S.move(out["co"], lines, [(0.0, 0.0)] * len(lines), stay, spread=True), out["faces"], project, *_mirror(out, new_strip))
    commit(context, out, element, parent)
    return ""


# --------------------------------------------------------------------------- picking


def _ray(context, event):
    """(origin, direction) of the mouse ray in the result's own space, or None off the 3D view."""
    guided = _g()
    preview = bpy.data.objects.get(guided.PREVIEW_NAME)
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
    sync(context)
    res = _g().state["res"]
    _plan(res, {"a": [], "b": []})  # the pins
    new = S.move_vertex(res["co"], res["faces"], vertex, goal, fl["stay"])
    keep = fl["sel"][1] if fl["sel"] is not None and fl["sel"][0] is res else None
    commit(context, dict(res, co=S.finish(res["co"], new, res["faces"], _project(), *_mirror(res, {"a": [vertex], "b": [vertex]}))), keep)


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
    res, preview = _g().state["res"], bpy.data.objects.get(_g().PREVIEW_NAME)
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
        sync(context)
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
        mesh = bpy.data.objects[_g().PREVIEW_NAME].data
        mesh.vertices.foreach_set("co", np.asarray(points, dtype=np.float32).ravel())
        mesh.update()
        fl["live"] = points
        overlay.tag_redraw()

    def _start_drag(self, context, event) -> bool:
        res = _g().state["res"]
        preview = bpy.data.objects.get(_g().PREVIEW_NAME)
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
        sync(context)
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


def ring(context, centre, normal, right, rx: float, ry: float) -> bool:
    """Closed rings of strips round `centre` (a point on the result): an oval `rx` by `ry` laid on your mesh and handed
    to the Flow Editor as a Ring, which redoes the quads round it."""
    from . import flowedit
    from . import flowstroke as F

    source = _g().state_source()
    if source is None:
        return False
    context.scene.curiomesh_guided.flow_live = True
    n = np.asarray(normal, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    r = r - (r @ n) * n
    line = F.ellipse(np.asarray(centre, dtype=np.float64), n, r / max(float(np.linalg.norm(r)), 1e-30), rx, ry, 48)
    obj = flowedit.add_stroke(context, source, "RING", flowedit._nearest_surface(source)(line))
    obj[flowedit.REACH_KEY] = RING_REACH
    return flowedit.request(context)


class CURIOMESH_OT_strip_ring(bpy.types.Operator):
    bl_idname = "curiomesh.strip_ring"
    bl_label = "Ring Round This"
    bl_description = "Press on the middle of an eye, a mouth or a hole and drag out to its edge. The quads there are redone as closed rings of strips round it"
    bl_options = {"REGISTER", "UNDO"}
    HINT = "Press on the middle of an eye or mouth and drag out to its edge (sideways: width, up: height). Right-click: cancel"

    @classmethod
    def poll(cls, context):
        return _has_result(context) and _g().state_source() is not None

    def invoke(self, context, event):
        sync(context)
        self.centre = None
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(self.HINT)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            context.workspace.status_text_set(None)
            return {"CANCELLED"}
        if _navigation(event):
            return {"PASS_THROUGH"}
        res = _g().state["res"]
        ray = _ray(context, event)
        if ray is None or res is None:
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            found = hit(res, *ray)
            if found is not None:
                p = np.asarray(res["co"])[list(res["faces"][found[1]])]
                n = np.cross(p[2] - p[0], p[-1] - p[1])
                self.centre, self.normal = found[0], n / max(float(np.linalg.norm(n)), 1e-30)
                self.start = (event.mouse_x, event.mouse_y)
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE" and self.centre is not None:
            context.workspace.status_text_set(None)
            preview = bpy.data.objects[_g().PREVIEW_NAME]
            rot = context.space_data.region_3d.view_rotation
            inv = preview.matrix_world.inverted().to_3x3()
            right = np.asarray(inv @ (rot @ Vector((1, 0, 0))))
            unit = np.linalg.norm(right)
            one = _screen(context, preview, [self.centre, self.centre + right / max(unit, 1e-30)])
            if one is None:
                return {"CANCELLED"}
            per_pixel = 1.0 / max(float(np.linalg.norm(one[1] - one[0])), 1e-9)  # the result's units per pixel here
            dx, dy = abs(event.mouse_x - self.start[0]) * per_pixel, abs(event.mouse_y - self.start[1]) * per_pixel
            size = max(dx, dy)
            if size <= 0.0:
                return {"CANCELLED"}
            ring(context, self.centre, self.normal, right, max(dx, 0.4 * size), max(dy, 0.4 * size))
            return {"FINISHED"}
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


def _edit_selection(context):
    """Faces of the result selected in Edit Mode on the preview, or None (back in Object Mode either way)."""
    preview = bpy.data.objects.get(_g().PREVIEW_NAME)
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
    bl_description = "Draw round an area of the result (or click a part, or select faces in Edit Mode first) and its quads are laid out again as long straight rows at the same size. The rows follow the surface's bends and any lines you drew, and the faces round it stay as they are"
    bl_options = {"REGISTER", "UNDO"}
    HINT = "Drag round the area to redraw in rows, or click a part for all of it. Right-click or Esc: cancel"

    @classmethod
    def poll(cls, context):
        return _has_result(context) or (context.mode == "EDIT_MESH" and context.edit_object is bpy.data.objects.get(_g().PREVIEW_NAME))

    def _run(self, context, ids):
        fl["lasso"] = None
        context.workspace.status_text_set(None)
        overlay.tag_redraw()
        error = redraw_rows(context, ids)
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
        sync(context)
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
            res, preview = _g().state["res"], bpy.data.objects[_g().PREVIEW_NAME]
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
        sync(context)
        found = selected()
        preview = bpy.data.objects.get(_g().PREVIEW_NAME)
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


class CURIOMESH_OT_strip_fix_all(_Edit, bpy.types.Operator):
    bl_idname = "curiomesh.strip_fix_all"
    bl_label = "Fix Every Strip"
    bl_description = "Straightens every wiggly strip of the result and evens out the uneven ones, all at once. Kept only when the rows come out better; one step, one Undo"
    KIND = "fix_all"

    @classmethod
    def poll(cls, context):
        return _has_result(context)


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


def _draw() -> None:
    try:
        guided = _g()
        res, preview = guided.state["res"], bpy.data.objects.get(guided.PREVIEW_NAME)
        if res is None or preview is None:
            return
        context = bpy.context
        if getattr(context.scene, "curiomesh_guided", None) is not None:
            sync(context)
            res = guided.state["res"]
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
