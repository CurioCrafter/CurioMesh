# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The Organize slider: a tidier layout of the result on screen, at about the same faces, and a meter of what it costs.

Moving the slider the first time works out a few strengths on a worker thread (`triadq.organize`: messy patches around
poles redrawn as grids and edges spun, then the quads settled on your mesh step by step). Each strength is measured (Tidiness, poles, how much of the view changed, the
Issue Finder) and gets its own preview mesh, so dragging only swaps meshes. With a mirrored result one half is organized
and mirrored again, so the seam stays exact. Most Detail Loss holds the slider back at the strongest one within it.

Loop Flow draws the edge loops of the result in a colour each, and its poles as dots.
"""

from __future__ import annotations

import colorsys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from . import autoretopo, overlay, segments, symmetry, turbo
from .triadq import checks, issues, patterns, tasks
from .triadq import organize as O
from .triadq.chords import PictureJudge

STRENGTHS = (0.25, 0.5, 0.75, 1.0)  # worked out once; the slider shows the nearest one at or below it. 0.25: poles cleaned, then the quads settle more and more
GAP = 0.1  # seconds between looks at the worker thread
LONG_LOOP = 4  # edges: shorter loops are not drawn in Loop Flow
FACE_ARRAYS = ("mat", "piece", "parts", "new", "locked", "chart")  # per-face numbers of a result that organizing carries along
SETTLE_PULL = "smooth"  # neighbours' middle; "settle" (rectangles and straight loops) measured the same view change with a few more slivers on five test models
REDRAW = True  # redraw messy patches around poles as grids (False: only spin edges, faces stay as they are)

org = {
    "base": None,  # the result (guided.state["auto_base"]) these strengths belong to
    "origin": None,  # its output at strength 0
    "report": None,  # its report at strength 0
    "levels": {},  # strength -> {"out", "tidy", "poles", "view", "issues"}
    "box": None,  # what the worker thread hands over
    "meshes": {},  # strength -> preview mesh
    "shown": 0.0,
    "flow": None,  # (id of the result, line batch data, pole points)
    "seg_piece": None,  # the segment of each face at strength 0 (the parts table's colours)
}


def _g():
    from . import guided  # late: guided imports this module

    return guided


def active() -> bool:
    return org["base"] is not None and _g().state["auto_base"] is org["base"]


def cancel() -> None:
    """Forget the strengths (a new result, another mesh, or the add-on going away)."""
    if org["box"] is not None:
        org["box"]["stop"] = True
    for mesh in org["meshes"].values():
        try:
            mesh.use_fake_user = False
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        except ReferenceError:
            pass
    org.update(base=None, origin=None, report=None, levels={}, box=None, meshes={}, shown=0.0, flow=None, seg_piece=None)


def _nearest(tree, points: np.ndarray) -> np.ndarray:
    if isinstance(tree, BVHTree):
        return np.asarray([tree.find_nearest(p)[0] or p for p in points.tolist()], dtype=np.float64).reshape(-1, 3)
    return tree.nearest(points)[0]


def _drain(work):
    while True:
        try:
            next(work)
        except StopIteration as done:
            return done.value


def start(context) -> bool:
    """Start working out the strengths of the result on screen. False when it has none (no search, Blockout, Reduce To)."""
    guided = _g()
    base = guided.state["auto_base"]
    job = base and base["job"]
    if not isinstance(job, autoretopo.AutoJob) or job.output_kind == "TRIS":
        return False
    cancel()
    out = base["out"]
    tree = turbo.face_tree(job.co, job.faces) or BVHTree.FromPolygons(job.co.tolist(), [list(f) for f in job.faces], all_triangles=False, epsilon=0.0)
    box = {"stop": False, "levels": {}, "stage": "Getting ready", "error": None}
    seg_piece = segments.seg.get("piece")
    org.update(base=base, origin=out, report=dict(base["report"]), box=box, seg_piece=seg_piece)
    org["meshes"][0.0] = _mesh_for(out)
    tasks.start(lambda: _work(box, out, job.co, job.faces, job.mirror, tree, seg_piece), "CurioMeshOrganize")
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=GAP)
    return True


def _kinds(out, faces: int, seg_piece):
    """(kind per face, table): faces with the same material, part, segment and flags share a kind, so new faces made
    from a patch carry the patch's numbers. The table rebuilds each per-face array from kinds."""
    names = [n for n in FACE_ARRAYS if out.get(n) is not None and len(out[n]) == faces]
    cols = [np.asarray(out[n]) for n in names] + ([np.asarray(seg_piece)] if seg_piece is not None and len(seg_piece) == faces else [])
    found: dict[tuple, int] = {}
    kind = [found.setdefault(row, len(found)) for row in zip(*(c.tolist() for c in cols))] if cols else [0] * faces
    rows = list(found) or [()]
    table = {n: np.asarray([r[k] for r in rows], dtype=np.asarray(out[n]).dtype) for k, n in enumerate(names)}
    if len(cols) > len(names):
        table["_seg"] = np.asarray([r[-1] for r in rows], dtype=np.asarray(seg_piece).dtype)
    return kind, table


def _work(box, out, src_co, src_faces, mirror, tree, seg_piece) -> None:
    """On a worker thread: every strength, measured. Hands each over in `box["levels"]` as it is ready."""
    try:
        co, faces = np.asarray(out["co"], dtype=np.float64), [tuple(f) for f in out["faces"]]
        kind, table = _kinds(out, len(faces), seg_piece)
        if mirror is not None:
            axis, offset, size, _whole, keep = mirror
            half, half_faces, ids = O.half_of(co, faces, axis, offset, keep)
            plane = (axis, offset)
        else:
            half, half_faces, ids, plane = co, faces, np.arange(len(faces)), None
        half_kinds = [kind[i] for i in ids]
        box["stage"] = "Looking at your mesh"
        low, high = src_co.min(axis=0), src_co.max(axis=0)
        centre, reach = (low + high) / 2.0, (float(np.ptp(src_co, axis=0).max()) or 1.0) * 0.55
        pixels = autoretopo.MAX_PIXELS
        reference = _drain(autoretopo.pictures(src_co, src_faces, centre, reach, pixels))
        judge = PictureJudge(reference, centre, reach, pixels, autoretopo.TURNED_DEG) if turbo.available() else None

        def whole_of(points, these, kinds):
            """The full result of an organized half (or of the whole): its arrays rebuilt from the kinds."""
            if mirror is None:
                full_co, full_faces, kept = points, these, np.arange(len(these))
                per_face = np.asarray(kinds, dtype=np.int64)
            else:
                full_co, full_faces, _m, _snapped, kept = symmetry.mirror_result(points, these, None, axis, offset, size, keep)
                part = np.asarray(kinds, dtype=np.int64)[kept]
                per_face = np.concatenate((part, part))
            made = dict(out, co=np.asarray(full_co, dtype=np.float64), faces=[tuple(f) for f in full_faces], **{n: arr[per_face] for n, arr in table.items()})
            return made, kept

        def measured(points, these, whole, kept):
            wrong, shown = autoretopo.picture_counts(reference, _drain(autoretopo.pictures(points, these, centre, reach, pixels)), 2.0 * reach / pixels)
            full_co, full_faces = whole["co"], whole["faces"]
            score = patterns.score(patterns.measure(full_co, full_faces))["score"]
            found = checks.defects(points, these, src_co, src_faces)
            masks = {k: getattr(found, k) for k in checks.Defects.KINDS}
            if mirror is not None:  # the half was checked: its mirror has the same faces
                masks = {k: np.concatenate((m[kept], m[kept])) for k, m in masks.items()}
            spots, record = issues.analyse(np.asarray(full_co), full_faces, masks, whole.get("piece"), {"source_faces": len(src_faces)})
            whole.update(issue_spots=spots, issues=record, bad=np.any(list(masks.values()), axis=0))
            return {"view": 100.0 * wrong / max(shown, 1), "tidy": score, "poles": O.poles(full_co, full_faces), "faces": len(full_faces), "issues": dict(record["totals"])}

        box["stage"] = "Measuring the result as it is"
        first, kept = whole_of(half, half_faces, half_kinds)
        box["levels"][0.0] = dict(measured(half, half_faces, first, kept), out=None)
        if box["stop"]:
            return
        box["stage"] = "Redrawing patches around poles"
        stay = O.pinned(half, half_faces, plane=plane)
        clean, clean_faces, clean_kinds, stay = O.clean_poles(half, half_faces, half_kinds, stay, judge, redraw=REDRAW)
        if box["stop"]:
            return
        box["stage"] = "Settling the quads"
        steps = [round((s - STRENGTHS[0]) / (1.0 - STRENGTHS[0]) * O.SETTLE_STEPS) for s in STRENGTHS]  # the weakest only cleans the poles
        stay |= O.pinned(clean, clean_faces, plane=plane)
        shots = O.settle(clean, clean_faces, lambda p: _nearest(tree, p), stay, plane, snapshots=steps[:-1], method=SETTLE_PULL)
        for strength, (_step, points) in zip(STRENGTHS, shots):
            if box["stop"]:
                return
            box["stage"] = f"Measuring strength {strength:g}"
            whole, kept = whole_of(points, clean_faces, clean_kinds)
            row = measured(points, clean_faces, whole, kept)
            box["levels"][strength] = dict(row, out=whole)
        box["stage"] = ""
    except Exception as exc:  # a thread has nobody above it to report to
        box["error"] = f"Organize failed: {exc}"


def _mesh_for(out):
    from .ladder import _make_mesh  # late: ladder imports this module

    source = _g().state_source()
    if source is None:
        return None
    mesh = _make_mesh(_g().PREVIEW_NAME + "_Organize", out, source)
    mesh.use_fake_user = True  # kept while another strength is on screen
    return mesh


def _tick():
    if not active() or org["box"] is None:
        return None
    box = org["box"]
    for strength, row in list(box["levels"].items()):
        if strength not in org["levels"]:
            org["levels"][strength] = row
            if row["out"] is not None and strength not in org["meshes"]:
                org["meshes"][strength] = _mesh_for(row["out"])  # made now, so showing it later is a swap
            overlay.tag_redraw()
    context = bpy.context
    gs = getattr(context.scene, "curiomesh_guided", None)
    if gs is not None:
        show(context, gs)
    done = box["error"] is not None or len(org["levels"]) > len(STRENGTHS)
    return None if done else GAP


def cap(gs):
    return gs.organize_cap if gs.organize_cap_use else None


def show(context, gs) -> bool:
    """Put the strength the slider asks for on screen (the nearest ready one at or below it, within the cap)."""
    if not active():
        return False
    strength = O.pick(org["levels"], gs.organize, cap(gs)) if 0.0 in org["levels"] else 0.0
    if strength == org["shown"]:
        return False
    guided = _g()
    base, row = org["base"], org["levels"].get(strength)
    out = org["origin"] if strength == 0.0 else row["out"]
    mesh = org["meshes"].get(strength)
    try:
        if mesh is not None and not mesh.name:
            mesh = None
    except ReferenceError:
        mesh = None
    if mesh is None:
        mesh = org["meshes"][strength] = _mesh_for(out)
    base["out"] = out
    report = dict(org["report"])
    zero = org["levels"].get(0.0)
    if strength and zero is not None:
        report.update(loss=report["loss"] + row["view"] - zero["view"], topology=row["tidy"])
    base["report"] = report
    if org["seg_piece"] is not None:  # the parts table stays; its faces are the organized ones now
        segments.seg.update(piece=out.get("_seg", org["seg_piece"]), batch_key=None, bad_key=None, tris=None, hover=None)
    guided.state["res"] = out
    guided._build_preview(context, guided.state_source(), out, uvs=False, mesh=mesh)
    guided.state["draw_key"] = None
    org["shown"] = strength
    overlay.tag_redraw()
    return True


def on_slider(self, context) -> None:
    if not active() and self.organize > 0.0 and not start(context):
        return
    show(context, self)


def status() -> str:
    box = org["box"]
    if box is None:
        return ""
    if box["error"]:
        return box["error"]
    left = len(STRENGTHS) + 1 - len(org["levels"])
    return f"Working out Organize: {len(STRENGTHS) - min(left, len(STRENGTHS))} of {len(STRENGTHS)} ready ({box['stage']})" if left > 0 else ""


def meter_text(gs) -> str:
    zero, row = org["levels"].get(0.0), org["levels"].get(org["shown"])
    if zero is None or row is None:
        return ""
    text = O.meter(zero, row)
    held = O.pick(org["levels"], gs.organize, None)
    if cap(gs) is not None and held > org["shown"]:
        text += f". Held at {org['shown']:g}: more would change the view by over {cap(gs):g}%"
    return text


def draw(layout, gs) -> None:
    """The slider, its meter and the switches, under the result."""
    box = layout.box()
    col = box.column(align=True)
    col.prop(gs, "organize", slider=True)
    row = col.row(align=True)
    row.prop(gs, "organize_cap_use", toggle=True)
    sub = row.row(align=True)
    sub.active = gs.organize_cap_use
    sub.prop(gs, "organize_cap", text="")
    col.prop(gs, "organize_flow", toggle=True, icon="FORCE_CURVE")
    if active():
        text = meter_text(gs)
        if text:
            overlay.wrap_label(box, text, icon="MESH_GRID", width=36, limit=3)
    note = status() if active() else ""
    if note:
        overlay.wrap_label(box, note, icon="TIME", width=36, limit=2)


def on_cap(self, context) -> None:
    show(context, self)


# --------------------------------------------------------------------------- Loop Flow


def loop_colours(labels: np.ndarray) -> np.ndarray:
    """A colour per loop, far apart for neighbouring numbers (golden-ratio hues)."""
    hue = (labels * 0.6180339887) % 1.0
    return np.asarray([(*colorsys.hsv_to_rgb(h, 0.75, 1.0), 1.0) for h in hue.tolist()], dtype=np.float32).reshape(-1, 4)


def flow_lines(co, faces):
    """(edge ends 2 per edge, colour per end, pole points, pole colours): long loops in a colour each, poles as dots
    (blue: three edges, red: five or more)."""
    topo = patterns.Topology(co, faces)
    if not len(topo.quads):
        return np.zeros((0, 3)), np.zeros((0, 4)), np.zeros((0, 3)), np.zeros((0, 4))
    labels = topo.loop_labels()
    sizes = np.bincount(labels)[labels]
    keep = np.flatnonzero(sizes >= LONG_LOOP)
    ends = np.stack((topo.edge_a[keep], topo.edge_b[keep]), axis=1).reshape(-1)
    colours = np.repeat(loop_colours(labels[keep]), 2, axis=0)
    poles = np.flatnonzero(topo.poles)
    low = topo.code[poles] < 4
    pole_colours = np.where(low[:, None], np.float32((0.2, 0.5, 1.0, 1.0)), np.float32((1.0, 0.15, 0.1, 1.0)))
    return topo.points[ends], colours, topo.points[poles], pole_colours


def _draw_flow() -> None:
    try:
        guided = _g()
        gs = getattr(bpy.context.scene, "curiomesh_guided", None)
        res, preview = guided.state["res"], bpy.data.objects.get(guided.PREVIEW_NAME)
        if gs is None or not gs.organize_flow or res is None or preview is None:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader

        matrix = tuple(round(x, 5) for row in preview.matrix_world for x in row)
        key = (id(res), matrix)
        if org["flow"] is None or org["flow"][0] != key:
            lines, colours, dots, dot_colours = flow_lines(res["co"], res["faces"])
            shader = gpu.shader.from_builtin("FLAT_COLOR")
            org["flow"] = (key, batch_for_shader(shader, "LINES", {"pos": guided._world(preview.matrix_world, lines).astype(np.float32), "color": colours}), batch_for_shader(shader, "POINTS", {"pos": guided._world(preview.matrix_world, dots).astype(np.float32), "color": dot_colours}), res)
        shader = gpu.shader.from_builtin("FLAT_COLOR")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.line_width_set(2.0)
        gpu.state.point_size_set(7.0)
        shader.bind()
        org["flow"][1].draw(shader)
        org["flow"][2].draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
        gpu.state.depth_test_set("NONE")
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


_handles: list = []


def register() -> None:
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_flow, (), "WINDOW", "POST_VIEW"))


def unregister() -> None:
    while _handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), "WINDOW")
        except Exception:
            pass
    cancel()
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)

