# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The Organize slider: a tidier layout of the result on screen, at about the same faces, and a meter of what it costs.

Moving the slider the first time works out a few strengths on a worker thread (`triadq.organize`: pole pairs walked
together until they cancel and messy patches around poles redrawn as grids, then the quads settled on your mesh step by
step, then the loops straightened and stacked evenly). The switches under the slider (Fewer Poles, Straighter Rows,
Even Sizes, Square Corners, Selection Only) pick which of those steps run and where. Each strength is measured (Tidiness, Loop Quality, poles, how much of the view changed, the Issue
Finder) and gets its own preview mesh, so dragging only swaps meshes. With a mirrored result one half is organized
and mirrored again, so the seam stays exact. Most Detail Loss holds the slider back at the strongest one within it.

Nothing stops at one pass. Tidy Again takes what is on screen as the new start and works out the strengths again from
there; Keep Tidying does that pass after pass, showing each, while a pass still makes the mesh tidier (`O.gained`) within
Most Detail Loss (counted from the result before the first pass), and stops by itself when one gains nothing. Each pass
is one step back for Undo Pass.

Loop Flow draws the face loops (strips of quads, `triadq.strips`) of the result in a colour each, and its poles as dots.
The strip picked with Face Loops is drawn by `faceloops`.
"""

from __future__ import annotations

import colorsys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from . import autoretopo, history, overlay, segments, symmetry, turbo
from . import guides
from .triadq import checks, issues, loopscore, patterns, tasks
from .triadq import organize as O
from .triadq.chords import PictureJudge

STRENGTHS = (0.25, 0.5, 0.75, 1.0)  # worked out once; the slider shows the nearest one at or below it. 0.25: poles cleaned, 0.5: quads settled, 0.75 and 1: loops straightened half and fully
GAP = 0.1  # seconds between looks at the worker thread
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
    "passes": [],  # earlier passes, oldest first: {"origin", "report", "seg_piece", "levels", "shown"}
    "keep": False,  # Keep Tidying is running
    "note": "",  # why Keep Tidying stopped
    "spent": [],  # preview meshes of earlier passes, removed once nothing shows them
}


def _g():
    from . import guided  # late: guided imports this module

    return guided


def active() -> bool:
    return org["base"] is not None and _g().state["auto_base"] is org["base"]


def _drop_work() -> None:
    """Stop the worker and let go of this pass's preview meshes (the one on screen stays until something replaces it)."""
    if org["box"] is not None:
        org["box"]["stop"] = True
    left = []
    for mesh in list(org["meshes"].values()) + org["spent"]:
        try:
            if mesh is None:
                continue
            mesh.use_fake_user = False
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            else:
                left.append(mesh)
        except ReferenceError:
            pass
    org.update(box=None, meshes={}, spent=left)


def cancel() -> None:
    """Forget the strengths and every pass (a new result, another mesh, or the add-on going away)."""
    _drop_work()
    org.update(base=None, origin=None, report=None, levels={}, shown=0.0, flow=None, seg_piece=None, passes=[], keep=False, note="")


def _first() -> dict | None:
    """Strength 0 of the first pass: the result before any Tidy Up. The meter and Most Detail Loss count from it."""
    return org["passes"][0]["levels"].get(0.0) if org["passes"] else org["levels"].get(0.0)


def _first_view():
    first = _first()
    return None if first is None else first.get("view")


def ready() -> bool:
    """Every strength of this pass is worked out (or its work ended)."""
    box = org["box"]
    return active() and (box is None or box["error"] is not None or box["stop"] or len(org["levels"]) > len(STRENGTHS))


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


def start(context, again: bool = False) -> bool:
    """Start working out the strengths of the result on screen. False when it has none (no search, Blockout, Reduce To).
    `again`: a new pass that starts from what is on screen now; the pass before it is kept for Undo Pass."""
    guided = _g()
    base = guided.state["auto_base"]
    job = base and base["job"]
    if not isinstance(job, autoretopo.AutoJob) or job.output_kind == "TRIS":
        return False
    if again and active():
        shown = org["shown"]
        # ponytail: only strength 0 and the picked strength are kept per pass, so Undo Pass shows the picked one
        org["passes"].append({"origin": org["origin"], "report": org["report"], "seg_piece": org["seg_piece"], "shown": shown, "levels": {s: r for s, r in org["levels"].items() if s in (0.0, shown)}})
        _drop_work()
        org["levels"], org["shown"] = {}, 0.0
    else:
        cancel()
    out = base["out"]
    tree = turbo.face_tree(job.co, job.faces) or BVHTree.FromPolygons(job.co.tolist(), [list(f) for f in job.faces], all_triangles=False, epsilon=0.0)
    box = {"stop": False, "levels": {}, "stage": "Getting ready", "error": None}
    seg_piece = segments.seg.get("piece")
    org.update(base=base, origin=out, report=dict(base["report"]), box=box, seg_piece=seg_piece)
    org["meshes"][0.0] = _mesh_for(out)
    strokes = list(getattr(job, "guide_lines", None) or []) + [line for _tag, rows in (getattr(job, "bend_groups", None) or []) for line in rows]
    gap = float(getattr(job, "gap", 0.0) or 0.0)
    choice = _choice(context, out)
    tasks.start(lambda: _work(box, out, job.co, job.faces, job.mirror, tree, seg_piece, (strokes, gap), choice), "CurioMeshOrganize")
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=GAP)
    return True


def _choice(context, out) -> dict:
    """The Tidy Up switches, and with Selection Only the faces of the result selected in Edit Mode (None: all)."""
    gs = context.scene.curiomesh_guided
    picked = None
    if gs.tidy_selected:
        preview = bpy.data.objects.get(_g().PREVIEW_NAME)
        if preview is not None and preview.type == "MESH":
            if preview.mode == "EDIT":
                preview.update_from_editmode()
            mesh = preview.data
            if len(mesh.polygons) == len(out["faces"]):
                picked = np.zeros(len(mesh.polygons), dtype=bool)
                mesh.polygons.foreach_get("select", picked)
                if not picked.any():
                    picked = None
    return {"poles": gs.tidy_poles, "rows": gs.tidy_rows, "sizes": gs.tidy_sizes, "square": gs.tidy_square, "picked": picked}


def on_options(self, context) -> None:
    """A switch changed: work the strengths out again (the slider keeps its place). One step back with Ctrl+Z."""
    if not active() and self.organize <= 0.0:
        return
    history.before(context)
    passes = org["passes"] if active() else []
    if active():  # back to the result as it was, so the new strengths start from it
        org["base"]["out"], org["base"]["report"] = org["origin"], dict(org["report"])
    started = start(context)
    if started:
        org["passes"] = passes  # earlier passes stay: only this one is worked out again
        org["shown"] = -1.0  # whatever is on screen is stale: the next `show` swaps the preview
    # a step of its own, made in the update so Blender's undo push for the switch keeps its number; after an undo or
    # redo `history.sync` shows the strength the slider asks for from these strengths
    history.record(context, "Tidy Up switches")
    if started:
        show(context, self)


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


def _held_by_strokes(points: np.ndarray, strokes) -> np.ndarray:
    """Vertices on a guide curve, drawn edge loop or deformation line of the search: Organize leaves them where they are."""
    lines, gap = strokes
    out = np.zeros(len(points), dtype=bool)
    if gap <= 0.0:
        gap = 0.01 * (float(np.linalg.norm(np.ptp(points, axis=0))) or 1.0)
    for line in lines:
        if len(line) > 1:
            out |= guides._to_line(points, np.asarray(line, dtype=np.float64)) < 0.3 * gap
    return out


def _work(box, out, src_co, src_faces, mirror, tree, seg_piece, strokes=((), 0.0), choice=None) -> None:
    """On a worker thread: every strength, measured. Hands each over in `box["levels"]` as it is ready."""
    choice = choice or {"poles": True, "rows": True, "sizes": True, "square": False, "picked": None}
    try:
        co, faces = np.asarray(out["co"], dtype=np.float64), [tuple(f) for f in out["faces"]]
        kind, table = _kinds(out, len(faces), seg_piece)
        picked = choice.get("picked")
        if mirror is not None:
            axis, offset, size, _whole, keep = mirror
            half, half_faces, ids = O.half_of(co, faces, axis, offset, keep)
            plane = (axis, offset)
        else:
            half, half_faces, ids, plane = co, faces, np.arange(len(faces)), None
        half_kinds = [kind[i] for i in ids]
        held_out, tidy_kinds = None, half_kinds
        if picked is not None:  # outside the selection: its own key, so nothing is redrawn across, and its corners pinned
            inside = np.asarray(picked, dtype=bool)[ids]
            tidy_kinds = [(k, bool(i)) for k, i in zip(half_kinds, inside)]
            held_out = np.zeros(len(half), dtype=bool)
            for f, i in zip(half_faces, inside):
                if not i:
                    held_out[list(f)] = True
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
            whole.update(issue_spots=spots, issues=record, bad=np.any([masks[k] for k in checks.Defects.BROKEN], axis=0))
            loops = loopscore.loop_quality(full_co, full_faces)
            loops.pop("numbers")
            return {"view": 100.0 * wrong / max(shown, 1), "tidy": score, "loops": loops, "poles": O.poles(full_co, full_faces), "faces": len(full_faces), "issues": dict(record["totals"])}

        box["stage"] = "Measuring the result as it is"
        first, kept = whole_of(half, half_faces, half_kinds)
        box["levels"][0.0] = dict(measured(half, half_faces, first, kept), out=None)
        if box["stop"]:
            return
        box["stage"] = "Walking poles together"
        stay = O.pinned(half, half_faces, plane=plane) | _held_by_strokes(half, strokes)
        if held_out is not None:
            stay |= held_out
        project = lambda p: _nearest(tree, p)  # noqa: E731
        if choice["poles"]:
            clean, clean_faces, clean_kinds, stay = O.fewer_poles(half, half_faces, tidy_kinds, stay, judge, project, plane, redraw=REDRAW, stopped=lambda: box["stop"])
        else:
            clean, clean_faces, clean_kinds = half, half_faces, tidy_kinds
        if box["stop"]:
            return
        box["stage"] = "Settling the quads"
        stay |= O.pinned(clean, clean_faces, plane=plane)
        if held_out is not None:  # the selection's edge: corners touching a face outside it
            outside = np.zeros(len(clean), dtype=bool)
            for f, k in zip(clean_faces, clean_kinds):
                if not k[1]:
                    outside[list(f)] = True
            stay |= outside
            clean_kinds = [k[0] for k in clean_kinds]
        if choice["sizes"] or choice["square"]:
            shots = O.settle(clean, clean_faces, project, stay, plane, snapshots=(0,), method="settle" if choice["square"] else SETTLE_PULL)  # the weakest only cleans the poles
        else:
            shots = [(0, clean), (0, clean)]
        box["stage"] = "Straightening the loops"
        settled = shots[-1][1]
        if choice["rows"]:
            straight = O.settle(settled, clean_faces, project, stay, plane, snapshots=(O.STRAIGHT_STEPS // 2,), method="straighten", steps=O.STRAIGHT_STEPS)
        else:
            straight = [(0, settled), (0, settled)]
        for strength, (_step, points) in zip(STRENGTHS, shots + straight):
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
    done = box["error"] is not None or box["stop"] or len(org["levels"]) > len(STRENGTHS)
    if done and org["keep"]:
        if box["error"] is None and not box["stop"] and gs is not None and _keep_next(context, gs):
            return GAP
        org["keep"] = False
    return None if done else GAP


def _keep_next(context, gs) -> bool:
    """Keep Tidying, once this pass is ready: show its strongest strength that made the mesh tidier within Most Detail
    Loss and start the next pass from it. False when none did: the pass is dropped and the last good result stays."""
    zero = org["levels"][0.0]
    better = [s for s, row in org["levels"].items() if s > 0.0 and O.gained(zero, row) and O.pick(org["levels"], s, cap(gs), _first_view()) == s]
    if not better:
        count = len(org["passes"])
        if not undo_pass(context, gs):  # the first pass gained nothing: the result as found stays
            gs["organize"] = 0.0
            show(context, gs)
        org["note"] = f"Keep Tidying stopped after {count} pass{'es' if count != 1 else ''}: another pass made it no tidier" + (" within Most Detail Loss" if cap(gs) is not None else "")
        history.record(context, "Tidy: Keep Tidying stopped", f"{count} pass{'es' if count != 1 else ''}", push=True)
        return False
    gs["organize"] = max(better)
    show(context, gs)
    start(context, again=True)
    gs["organize"] = 1.0  # the next pass shows each strength as it comes in
    history.record(context, f"Tidy: Keep Tidying pass {len(org['passes'])}", meter_text(gs), push=True)  # each pass is one Ctrl+Z
    return True


def undo_pass(context, gs) -> bool:
    """Back to the pass before, with the strength it had picked on screen. False when this is the first pass."""
    if not active() or not org["passes"]:
        return False
    _drop_work()
    back = org["passes"].pop()
    org.update(origin=back["origin"], report=back["report"], seg_piece=back["seg_piece"], levels=back["levels"], shown=-1.0, keep=False, note="")
    gs["organize"] = back["shown"]
    show(context, gs)
    return True


def keep_tidying(context, gs) -> bool:
    """Start Keep Tidying: from the first pass if there is none yet, else from the pass on screen once it is ready."""
    org["note"] = ""
    if not active():
        if not start(context):
            return False
        gs["organize"] = 1.0
    org["keep"] = True
    if ready() and not _keep_next(context, gs):
        org["keep"] = False
        return True
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=GAP)
    return True


def tidy_again(context, gs) -> bool:
    """One more pass from what is on screen, at the same slider strength."""
    wanted = gs.organize or 1.0
    org["note"] = ""
    if not start(context, again=True):
        return False
    gs["organize"] = wanted
    return True


def cap(gs):
    return gs.organize_cap if gs.organize_cap_use else None


def show(context, gs) -> bool:
    """Put the strength the slider asks for on screen (the nearest ready one at or below it, within the cap)."""
    if not active():
        return False
    strength = O.pick(org["levels"], gs.organize, cap(gs), _first_view()) if 0.0 in org["levels"] else 0.0
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
    history.settled("tidy", meter_text(gs) or f"strength {strength:g}")
    overlay.tag_redraw()
    return True


def on_slider(self, context) -> None:
    history.note(context, "tidy", "Tidy Up")  # Blender pushes the undo step when the slider lets go
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
    if left <= 0 or box["stop"]:
        return ""
    what = f"Keep Tidying, pass {len(org['passes']) + 1}" if org["keep"] else f"Working out Tidy Up pass {len(org['passes']) + 1}" if org["passes"] else "Working out Tidy Up"
    return f"{what}: {len(STRENGTHS) - min(left, len(STRENGTHS))} of {len(STRENGTHS)} ready ({box['stage']})"


def meter_text(gs) -> str:
    zero, row = _first(), org["levels"].get(org["shown"])
    if zero is None or row is None:
        return org["note"]
    text = O.meter(zero, row)
    if org["passes"]:
        text = f"Pass {len(org['passes']) + 1}: {text}"
    held = O.pick(org["levels"], gs.organize, None)
    if cap(gs) is not None and held > org["shown"]:
        text += f". Held at {org['shown']:g}: more would change the view by over {cap(gs):g}%"
    return text + (f". {org['note']}" if org["note"] else "")


def draw_again(layout, gs) -> None:
    """Again, Keep Tidying (or Stop) and Undo Pass, in one row under the slider."""
    row = layout.row(align=True)
    row.operator("curiomesh.tidy_again", text="Again", icon="FILE_REFRESH")
    if org["keep"] and active():
        row.operator("curiomesh.tidy_stop", text="Stop", icon="CANCEL")
    else:
        row.operator("curiomesh.tidy_keep", text="Keep Tidying", icon="PLAY")
    row.operator("curiomesh.tidy_back", text="", icon="LOOP_BACK")


def draw(layout, gs) -> None:
    """The slider, its meter and the switches, under the result."""
    box = layout.box()
    col = box.column(align=True)
    col.prop(gs, "organize", slider=True)
    draw_again(col, gs)
    row = col.row(align=True)
    row.prop(gs, "organize_cap_use", toggle=True)
    sub = row.row(align=True)
    sub.active = gs.organize_cap_use
    sub.prop(gs, "organize_cap", text="")
    col.prop(gs, "organize_flow", toggle=True, icon="FORCE_CURVE")
    if active():
        text = meter_text(gs)
        if text:
            overlay.wrap_label(box, text, icon="MESH_GRID", width=36, limit=4)
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


def flow_lines(co, faces, strips=None):
    """(line ends 2 per quad crossed, colour per end, pole points, pole colours): each strip of quads drawn as a line
    through the middles of the edges it crosses, a colour per strip; poles as dots (blue: three edges, red: five or more)."""
    from .triadq import strips as S

    topo = patterns.Topology(co, faces)
    if not len(topo.quads):
        return np.zeros((0, 3)), np.zeros((0, 4)), np.zeros((0, 3)), np.zeros((0, 4))
    ends, labels = (strips or S.Strips(co, faces)).segments()
    colours = np.repeat(loop_colours(labels), 2, axis=0)
    poles = np.flatnonzero(topo.poles)
    low = topo.code[poles] < 4
    pole_colours = np.where(low[:, None], np.float32((0.2, 0.5, 1.0, 1.0)), np.float32((1.0, 0.15, 0.1, 1.0)))
    return ends, colours, topo.points[poles], pole_colours


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
            from . import faceloops  # late: faceloops imports this module

            lines, colours, dots, dot_colours = flow_lines(res["co"], res["faces"], faceloops.strips(res))
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


# --------------------------------------------------------------------------- Again, Keep Tidying, Stop, Undo Pass


def _settings(context):
    return getattr(context.scene, "curiomesh_guided", None)


class CURIOMESH_OT_tidy_again(bpy.types.Operator):
    bl_idname = "curiomesh.tidy_again"
    bl_label = "Tidy Again"
    bl_description = "Tidies the result on screen once more, starting from where the last pass left it. Drag the slider to pick how strong this pass is"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return active() and ready() and org["shown"] > 0.0 and not org["keep"]

    def execute(self, context):
        history.before(context)
        if not tidy_again(context, _settings(context)):
            return {"CANCELLED"}
        history.record(context, "Tidy: Again", f"pass {len(org['passes']) + 1}", push=True)
        return {"FINISHED"}


class CURIOMESH_OT_tidy_keep(bpy.types.Operator):
    bl_idname = "curiomesh.tidy_keep"
    bl_label = "Keep Tidying"
    bl_description = "Tidies pass after pass, showing each, while a pass still makes the mesh tidier within Most Detail Loss. It stops by itself when a pass gains nothing, and Stop ends it any time"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _settings(context) is not None and not org["keep"] and (active() or _g().state.get("auto_base") is not None)

    def execute(self, context):
        history.before(context)
        if not keep_tidying(context, _settings(context)):
            self.report({"WARNING"}, "Tidy Up needs a result from Start")
            return {"CANCELLED"}
        return {"FINISHED"}


class CURIOMESH_OT_tidy_stop(bpy.types.Operator):
    bl_idname = "curiomesh.tidy_stop"
    bl_label = "Stop Tidying"
    bl_description = "Stops Keep Tidying. The passes done so far stay, and Undo Pass steps back through them"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return org["keep"]

    def execute(self, context):
        org["keep"] = False
        if org["box"] is not None and not ready():
            org["box"]["stop"] = True
        overlay.tag_redraw()
        return {"FINISHED"}


class CURIOMESH_OT_tidy_back(bpy.types.Operator):
    bl_idname = "curiomesh.tidy_back"
    bl_label = "Undo Pass"
    bl_description = "Goes back to the result before the last Tidy Again or Keep Tidying pass. Press it again to step further back"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return active() and bool(org["passes"])

    def execute(self, context):
        org["keep"] = False
        history.before(context)
        if not undo_pass(context, _settings(context)):
            return {"CANCELLED"}
        history.record(context, "Tidy: Undo Pass", f"back to pass {len(org['passes']) + 1}", push=True)  # Ctrl+Z brings the pass back
        return {"FINISHED"}


CLASSES = (CURIOMESH_OT_tidy_again, CURIOMESH_OT_tidy_keep, CURIOMESH_OT_tidy_stop, CURIOMESH_OT_tidy_back)
_handles: list = []


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
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
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass

