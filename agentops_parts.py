# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""Agent commands for Auto Fix issues, Fix Selection and Parts (AGENT_TOOLS.md section 2).

Each command calls the function the matching button calls (issue_panel, fix_select, segments, segment_goals, ladder),
so it gets the same History step and the same Ctrl+Z. An agent keeps its changes directly: Look First is a person's
step, and the reply carries the numbers instead. A redo that has to run in the background returns at once with
`running: true`; the agent polls `job.status`.

No bpy and no add-on import at the top, so the shaping and checking helpers run in plain tests.
"""

from __future__ import annotations

USES = ("KEEP", "REDUCE", "LOWPOLY", "ANIMATION", "SUBD", "TRIS")  # segment_goals.GOAL_ITEMS
USE_ALIASES = {"LOW_POLY": "LOWPOLY", "SUBDIV": "SUBD", "SUBDIVISION": "SUBD"}  # the Use names Auto uses
METHODS = ("AUTO", "PAIRING", "FLAT", "QUADRIFLOW", "GRID", "FLOW", "FIELD", "TRIS")  # profiles.METHOD_ITEMS


try:
    from .agentcmd import AgentError as PartsError
except ImportError:  # the unit tests import this module on its own
    from agentcmd import AgentError as PartsError


# --------------------------------------------------------------------------- pure helpers (tested without Blender)


def check_ids(ids, count: int, what: str = "part") -> list:
    """Whole numbers 0..count-1, without repeats, in the order given; NOT_FOUND names the ones that are not there."""
    out, seen = [], set()
    for i in ids:
        if isinstance(i, bool) or not isinstance(i, int):
            raise PartsError("BAD_ARGS", f"{what} ids are whole numbers, got {i!r}", details={"argument": what})
        if i not in seen:
            seen.add(i)
            out.append(i)
    bad = [i for i in out if not 0 <= i < count]
    if bad:
        raise PartsError("NOT_FOUND", f"no {what} {bad[0]}", f"The {what}s are numbered 0 to {count - 1}" if count else f"There are no {what}s", {"missing": bad, "count": count})
    return out


def use_of(name: str) -> str:
    """A part Use (KEEP, REDUCE, LOWPOLY, ANIMATION, SUBD, TRIS), also from Auto's names (LOW_POLY, SUBDIV)."""
    key = USE_ALIASES.get(str(name).upper(), str(name).upper())
    if key not in USES:
        raise PartsError("BAD_ARGS", f"use must be one of {list(USES)}", details={"argument": "use", "allowed": list(USES) + list(USE_ALIASES)})
    return key


def shape_part(record: dict, bad_spots: int = 0, selected: bool = False) -> dict:
    """One row of parts.list from a part record of `segments.seg["parts"]`."""
    return {
        "id": record["id"], "name": record.get("name") or f"Part {record['id'] + 1}", "kind": record["kind"],
        "faces_before": record["before"], "faces_after": record["after"],
        "view_change": record["change"], "tidiness": record.get("topology"),
        "method": record["method"], "use": record.get("goal", "REDUCE"), "detail": record.get("goal_detail", 0.5),
        "locked": bool(record.get("locked")), "tries": len(record["tries"]), "current_try": record["cur"],
        "bad_spots": bad_spots, "selected": selected,
    }


def shape_parts(records: list, spots: list, selected, judged=None) -> dict:
    """parts.list: every part (by number) and the numbers worst first (largest change first, as `judged` says; unmeasured last)."""
    bad = {}
    for s in spots:
        p = int(s.get("part", -1))
        bad[p] = bad.get(p, 0) + 1
    rows = [shape_part(r, bad.get(r["id"], 0), r["id"] in selected) for r in records]
    score = [judged(r) if judged else r["change"] for r in records]
    order = sorted(range(len(rows)), key=lambda i: (score[i] is None, -(score[i] or 0.0), i))
    return {"parts": rows, "worst_first": order, "selected": sorted(selected)}


def shape_spot(index: int, spot: dict, weights: dict) -> dict:
    """One row of issues.list: the spot's worst kind, all its kinds, its part, faces, centre and score."""
    kinds = spot.get("kinds", {})
    worst = max(kinds, key=lambda k: weights.get(k, 0) * kinds[k]) if kinds else ""
    return {"index": index, "kind": worst, "kinds": dict(kinds), "part": int(spot.get("part", -1)), "faces": list(spot["faces"]), "face_count": len(spot["faces"]), "centre": list(spot["center"]), "radius": spot.get("radius"), "score": spot.get("score")}


def shape_tries(record: dict) -> list:
    return [{"index": i, "method": t["method"], "detail": t.get("detail"), "faces": t["after"], "view_change": t["change"], "tidiness": t.get("topology"), "current": i == record["cur"]} for i, t in enumerate(record["tries"])]


def shape_diff(diff: dict) -> dict:
    """issues.compare: counts and the spots, without face lists."""
    slim = lambda spots: [{k: s.get(k) for k in ("kinds", "part", "center", "score")} for s in spots]  # noqa: E731
    return {"fixed": len(diff["fixed"]), "still": len(diff["still"]), "new": len(diff["new"]), "fixed_spots": slim(diff["fixed"]), "still_spots": slim(diff["still"]), "new_spots": slim(diff["new"]), "totals": diff["totals"]}


# --------------------------------------------------------------------------- guards (these need the add-on)


def _g():
    from . import guided

    return guided


def _running() -> bool:
    work = _g().state["work"]
    return work is not None and bool(work["runner"].busy)


def _busy_check() -> None:
    info = _g().progress_info()
    if info is not None:
        raise PartsError("BUSY", f"CurioMesh is busy: {info[1]}", "Poll job.status, or stop it with job.stop", {"fraction": info[0], "words": info[1]})


def _result() -> dict:
    res = _g().state["res"]
    if res is None:
        raise PartsError("NO_RESULT", "There is no result on screen", "Run auto.start and wait for it to finish")
    return res


def _table(edit: bool = False, redo: bool = False):
    """The parts table of the result on screen. `edit`: nothing may be running. `redo`: the result must allow part redos."""
    from . import segments

    _result()
    base = _g().state["auto_base"]
    if base is None or base["job"] is None:
        raise PartsError("NOT_READY", "This result has no parts (it was imported, not made by Auto)", "Run auto.start on the mesh")
    segments.ensure()
    if not segments.ready():
        raise PartsError("NOT_READY", "The parts of this result are not ready yet", "Poll job.status and try again when nothing runs")
    if edit:
        _busy_check()
    if redo and segments.mirrored():
        raise PartsError("NOT_READY", "Part redos are off on this result (mirror, Even Rows or an unsplit loose piece)", "Redo the whole mesh with auto.start instead")
    return segments.seg


def _found() -> None:
    from . import segments

    if segments.seg.get("src") is None:
        raise PartsError("NOT_READY", "The parts are not found yet", "Wait for the background find to finish (job.status), then try again")


def _numbers(context) -> str:
    return f"{len(_g().state['res']['faces']):,} faces"


def _edit(context, name: str, change) -> int:
    """Run a part edit (`change()` returns a problem text or ''), keep it as one History step and one Ctrl+Z, and return the step.
    A refused edit changes nothing."""
    from . import history, segments

    history.before(context)
    selected = set(segments.seg["sel"])
    problem = change()
    if problem:
        segments.seg["sel"], segments.seg["batch_key"] = selected, None
        raise PartsError("REFUSED", problem, "Nothing changed")
    return history.record(context, name, _numbers(context), push=True)


def _plan_done(context, plan: dict, kept: bool) -> dict:
    if not kept:
        raise PartsError("REFUSED", plan["why"] or f"{plan['name']}: nothing changed", "Nothing changed")
    return {"name": plan["name"], "parts": sorted(set(plan.get("tries", {})) | set(plan.get("todo", []))), "started_in_background": list(plan.get("todo", [])), "running": _running(), "faces": len(_g().state["res"]["faces"])}


# --------------------------------------------------------------------------- issues


def issues_list(context, object=None):
    from . import issue_panel
    from .triadq import issues

    _result()
    spots = issue_panel.spots()
    rec = _g().state["res"].get("issues") or {}
    return {"count": len(spots), "totals": rec.get("totals", {}), "spots": [shape_spot(i, s, issues.WEIGHTS) for i, s in enumerate(spots)]}


def _spot(index: int) -> int:
    from . import issue_panel

    _table(edit=True, redo=True)
    spots = issue_panel.spots()
    check_ids([index], len(spots), "spot")
    p = issue_panel.spot_part(index)
    if p < 0:
        raise PartsError("NOT_FOUND", f"spot {index} is not on a part", "Use fix.run on its faces instead", {"spot": index})
    return p


def issues_redo(context, index, object=None):
    from . import issue_panel, ladder

    p = _spot(index)
    plan = issue_panel.redo_plan(p, index)
    if not plan["tries"] and not plan["todo"] and ladder.active():  # no smaller size is worked out yet (headless, or just started): remake the part for a smaller size in the background, as when there is no ladder
        plan.update(todo=[p], why="")
    if not plan["tries"] and not plan["todo"]:
        return _plan_done(context, plan, False)
    return dict(_plan_done(context, plan, ladder.apply_sizes(context, plan["sizes"], plan["todo"], False, plan["name"]) > 0), spot=index, part=p)


def issues_keep_original(context, index, object=None):
    from . import issue_panel, segments

    p = _spot(index)
    plan = issue_panel.original_plan(p, index)
    return dict(_plan_done(context, plan, bool(plan["tries"]) and segments.use_tries(context, plan["tries"], plan["name"])), spot=index, part=p)


def issues_put_back(context, part=-1, object=None):
    from . import issue_panel, segments

    _table(edit=True, redo=True)
    if part >= 0:
        check_ids([part], len(segments.seg["parts"]))
    plan = issue_panel.put_back_plan(part)
    return _plan_done(context, plan, bool(plan["tries"]) and segments.use_tries(context, plan["tries"], plan["name"]))


def issues_save(context, path, object=None):
    from . import issue_panel

    _result()
    if _g().state["res"].get("issues") is None:
        raise PartsError("NO_RESULT", "This result has no issue record", "Run auto.start again")
    try:
        rec = issue_panel.save_report(path)
    except OSError as exc:
        raise PartsError("FILE", f"Could not save the issue report: {exc}", "Check that the folder exists and can be written", {"path": path})
    return {"path": path, "spots": len(rec.get("spots", [])), "totals": rec.get("totals", {})}


def issues_compare(context, path, object=None):
    from . import issue_panel

    _result()
    if _g().state["res"].get("issues") is None:
        raise PartsError("NO_RESULT", "This result has no issue record", "Run auto.start again")
    try:
        diff = issue_panel.compare_report(path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PartsError("FILE", f"Could not read that report: {exc}", "Pass a report written by issues.save", {"path": path})
    return shape_diff(diff)


# --------------------------------------------------------------------------- Fix Selection


def _mesh_object(name):
    import bpy

    obj = bpy.data.objects.get(name) if name else bpy.context.view_layer.objects.active
    if obj is None or obj.type != "MESH":
        raise PartsError("NO_OBJECT" if not name else "NOT_FOUND", f"no mesh object {name!r}" if name else "There is no active mesh", "Pass object, or make a mesh active with scene.select")
    return obj


def _in_edit_mode(context, obj, faces, work):
    """Run `work(bm)` with `obj` in Edit Mode and `faces` (or, with None, nothing) selected; leave the mode as it was."""
    import bmesh
    import bpy

    from . import overlay

    if overlay.is_preview(obj, overlay.GUIDED):
        raise PartsError("REFUSED", "This is the preview: a fix there is lost when it is rebuilt", "Apply first (apply), then fix the result")
    count = len(obj.data.polygons)
    if faces is not None:
        faces = check_ids(faces, count, "face")
    view_layer = context.view_layer
    was_active, was_mode = view_layer.objects.active, obj.mode
    if was_active is not obj or was_mode != "OBJECT":
        if was_active is not None and was_active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_mode = {"FACE"}
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            f.select_set(False)
        for i in faces or ():
            bm.faces[i].select_set(True)
        bm.select_flush_mode()
        text = work(obj)
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.index_update()
        bad = [f.index for f in bm.faces if f.select]
        total = len(bm.faces)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT" if was_mode != "EDIT" else "EDIT")
        if was_active is not None and was_active is not obj:
            view_layer.objects.active = was_active
    return text, bad, total


def _grow(context, grow):
    return context.scene.curiomesh_fix.grow if grow is None else grow


def fix_find(context, faces=None, grow=None, object=None):
    from . import fix_select

    obj = _mesh_object(object)
    text, bad, total = _in_edit_mode(context, obj, faces, lambda o: fix_select.find_bad(context, o, _grow(context, grow)))
    return {"object": obj.name, "report": text, "bad_faces": bad, "bad_count": len(bad), "faces": total}


def fix_run(context, how, faces=None, grow=None, object=None):
    from . import fix_select, history

    if how != "ALL" and how not in fix_select.FIXES:
        raise PartsError("BAD_ARGS", f"how must be one of {['ALL', *fix_select.FIXES]}", details={"argument": "how", "allowed": ["ALL", *fix_select.FIXES]})
    obj = _mesh_object(object)
    history.before(context)
    done = {}

    def work(o):
        done["text"], done["warning"] = fix_select.run_fix(context, o, how, _grow(context, grow))
        return done["text"]

    text, bad, total = _in_edit_mode(context, obj, faces, work)
    if done["warning"]:
        raise PartsError("REFUSED", done["warning"], "Nothing changed. Set a Source (settings.set fix source) or pick another fix")
    label = fix_select.FIXES[how][1] if how in fix_select.FIXES else "Fix All"
    history.record(context, f"Fix Selection: {label}", text, push=True)
    return {"object": obj.name, "report": text, "bad_faces": bad, "bad_count": len(bad), "faces": total}


# --------------------------------------------------------------------------- parts


def parts_list(context, object=None):
    from . import issue_panel, segments

    seg = _table()
    return dict(shape_parts(seg["parts"], issue_panel.spots(), seg["sel"], segments.judged), mirrored=segments.mirrored(), editable=seg.get("src") is not None)


def parts_select(context, parts, extend=False, object=None):
    from . import segments

    seg = _table()
    segments.select(check_ids(parts, len(seg["parts"])), extend)
    return {"selected": sorted(seg["sel"])}


def parts_worst(context, count=10, object=None):
    from . import segments

    seg = _table()
    rows = shape_parts(seg["parts"], [], set(), segments.judged)["worst_first"]
    picked = rows[: max(int(count), 0)]
    segments.select(picked)
    return {"selected": picked}


def _one(part: int, seg) -> int:
    return check_ids([part], len(seg["parts"]))[0]


def parts_rename(context, part, name, object=None):
    from . import segment_goals

    seg = _table(edit=True)
    p = _one(part, seg)
    _found()
    if not name.strip():
        raise PartsError("BAD_ARGS", "name is empty", details={"argument": "name"})
    step = _edit(context, "Rename Part", lambda: "" if segment_goals.rename_part(p, name) else "name is empty")
    return {"part": p, "name": seg["parts"][p]["name"], "step": step}


def parts_lock(context, parts, on=None, object=None):
    from . import segment_goals

    seg = _table(edit=True)
    picked = check_ids(parts, len(seg["parts"]))
    if not picked:
        raise PartsError("BAD_ARGS", "parts is empty", details={"argument": "parts"})
    _found()
    on = (not all(seg["parts"][p].get("locked") for p in picked)) if on is None else bool(on)
    step = _edit(context, "Lock Parts" if on else "Unlock Parts", lambda: segment_goals.set_locked(picked, on) or "")
    return {"parts": picked, "locked": on, "step": step}


def parts_merge(context, parts, object=None):
    from . import segment_goals, segments

    seg = _table(edit=True, redo=True)
    picked = check_ids(parts, len(seg["parts"]))
    if len(picked) < 2:
        raise PartsError("BAD_ARGS", "merge needs two or more different parts", details={"argument": "parts"})
    _found()
    segments.select(picked)
    step = _edit(context, "Merge Parts", segment_goals.merge_selected)
    return {"part": sorted(seg["sel"])[0], "parts_now": len(seg["parts"]), "step": step}


def parts_halve(context, part, object=None):
    from . import segment_goals, segments

    seg = _table(edit=True, redo=True)
    p = _one(part, seg)
    _found()
    if seg.get("tree") is None:
        raise PartsError("NOT_READY", "The parts are not found yet", "Wait for the background find to finish (job.status), then try again")
    segments.select([p])
    step = _edit(context, "Split Part in Two", lambda: segment_goals.halve(p))
    return {"parts_now": len(seg["parts"]), "selected": sorted(seg["sel"]), "step": step}


def parts_from_faces(context, faces, object=None):
    from . import segment_goals

    seg = _table(edit=True, redo=True)
    _found()
    faces = check_ids(faces, len(_g().state["res"]["faces"]), "face")
    if not faces:
        raise PartsError("BAD_ARGS", "faces is empty", details={"argument": "faces"})
    step = _edit(context, "Make a Part", lambda: segment_goals.claim_result_faces(faces))
    return {"part": sorted(seg["sel"])[0], "parts_now": len(seg["parts"]), "step": step}


def parts_redo(context, parts, use=None, method=None, detail=None, object=None):
    from . import segment_goals, segments

    seg = _table(edit=True, redo=True)
    picked = check_ids(parts, len(seg["parts"]))
    if not picked:
        raise PartsError("BAD_ARGS", "parts is empty", details={"argument": "parts"})
    if method is not None and method not in METHODS:
        raise PartsError("BAD_ARGS", f"method must be one of {list(METHODS)}", details={"argument": "method", "allowed": list(METHODS)})
    goal = use_of(use) if use is not None else None
    segments.select(picked)
    groups = {}  # parts with the same Use, Detail and Method are redone together
    for p in picked:
        r = seg["parts"][p]
        key = (goal or r.get("goal", "REDUCE"), r.get("goal_detail", 0.5) if detail is None else float(detail), method or r.get("choice", "AUTO"))
        groups.setdefault(key, []).append(p)
    switched, started, waiting = [], [], []
    for key, group in groups.items():
        if started:  # one background redo at a time: a new one would stop this one
            waiting += group
            continue
        done, todo = segment_goals.redo_parts(context, group, *key)
        switched += done
        started += todo
    return {"parts": picked, "switched_to_existing_try": switched, "started_in_background": started, "not_started": waiting, "running": _running(), "hint": "Call parts.redo again for not_started when job.status says nothing runs" if waiting else ""}


def parts_tries(context, part, object=None):
    seg = _table()
    return {"part": _one(part, seg), "tries": shape_tries(seg["parts"][part])}


def parts_use_try(context, part, index, object=None):
    from . import segments

    seg = _table(edit=True, redo=True)
    p = _one(part, seg)
    tries = seg["parts"][p]["tries"]
    check_ids([index], len(tries), "try")
    if index == seg["parts"][p]["cur"]:
        return {"part": p, "index": index, "changed": False}
    if not segments.use_try(context, p, index):
        raise PartsError("REFUSED", "That try could not be put in", "Nothing changed")
    return {"part": p, "index": index, "changed": True, "running": _running(), "faces": len(_g().state["res"]["faces"])}


def parts_keep_original(context, parts, object=None):
    from . import segments

    seg = _table(edit=True, redo=True)
    picked = check_ids(parts, len(seg["parts"]))
    changed = segments.keep_original(context, picked, "Keep Original") if picked else 0
    if not changed:
        raise PartsError("REFUSED", "No part could be put back", "Nothing changed")
    return {"parts": picked, "changed": changed, "running": _running(), "faces": len(_g().state["res"]["faces"])}


# --------------------------------------------------------------------------- the command table

_OBJECT = {"type": "string", "description": "Object name; default the active mesh"}
_PARTS = {"type": "array", "items": {"type": "integer"}, "description": "Part numbers, as parts.list gives them"}
_FACES = {"type": "array", "items": {"type": "integer", "minimum": 0}, "description": "Face numbers"}


def _schema(props: dict, required=()) -> dict:
    return {"type": "object", "properties": dict(props, object=_OBJECT), "required": list(required)}


COMMANDS = {
    "issues.list": {"fn": issues_list, "args": _schema({}), "edits": False, "help": "The bad spots of the result, worst first: index, kind, part, faces, centre, score"},
    "issues.redo": {"fn": issues_redo, "args": _schema({"index": {"type": "integer", "minimum": 0}}, ["index"]), "edits": True, "help": "Remake the part under this spot with smaller quads and keep it. The old version stays in the part's tries"},
    "issues.keep_original": {"fn": issues_keep_original, "args": _schema({"index": {"type": "integer", "minimum": 0}}, ["index"]), "edits": True, "help": "Put your own faces back on the part under this spot and keep them"},
    "issues.put_back": {"fn": issues_put_back, "args": _schema({"part": {"type": "integer", "minimum": -1, "description": "Part to switch back; -1 (default) for every part showing your own faces"}}), "edits": True, "help": "Switch parts that got your own faces back to the reduced faces they had before"},
    "issues.save": {"fn": issues_save, "args": _schema({"path": {"type": "string"}}, ["path"]), "edits": False, "help": "Save the issue report as a json file"},
    "issues.compare": {"fn": issues_compare, "args": _schema({"path": {"type": "string"}}, ["path"]), "edits": False, "help": "Compare the bad spots with an older report: fixed, still there, new"},
    "fix.find": {"fn": fix_find, "args": _schema({"faces": _FACES, "grow": {"type": "integer", "minimum": 0, "description": "Rings of faces around the faces that are checked too; default the Fix Selection setting"}}), "edits": False, "help": "Find the bad faces among these faces (and the rings around them); no faces means the whole mesh. Not for the preview"},
    "fix.run": {"fn": fix_run, "args": _schema({"how": {"type": "string", "enum": ["ALL", "CLEAN", "FLIP", "UNTANGLE", "RELAX", "PATCH"]}, "faces": _FACES, "grow": {"type": "integer", "minimum": 0}}, ["how"]), "edits": True, "help": "Fix only the bad faces among these faces (no faces: the whole mesh) this way. One History step. Not for the preview"},
    "parts.list": {"fn": parts_list, "args": _schema({}), "edits": False, "help": "Every part: id, name, kind, faces before and after, view change, tidiness, method, Use, locked, tries, bad spots; and the ids worst first"},
    "parts.select": {"fn": parts_select, "args": _schema({"parts": _PARTS, "extend": {"type": "boolean", "description": "Toggle these parts in the marking instead of replacing it"}}, ["parts"]), "edits": False, "help": "Mark parts, as a click in the viewport does. An empty list clears the marking"},
    "parts.worst": {"fn": parts_worst, "args": _schema({"count": {"type": "integer", "minimum": 0}}), "edits": False, "help": "Mark the parts whose look changed most"},
    "parts.rename": {"fn": parts_rename, "args": _schema({"part": {"type": "integer", "minimum": 0}, "name": {"type": "string"}}, ["part", "name"]), "edits": True, "help": "Give a part a name of your own"},
    "parts.lock": {"fn": parts_lock, "args": _schema({"parts": _PARTS, "on": {"type": "boolean", "description": "Default: lock, or unlock when all are locked"}}, ["parts"]), "edits": True, "help": "Lock parts so size changes keep them face for face"},
    "parts.merge": {"fn": parts_merge, "args": _schema({"parts": _PARTS}, ["parts"]), "edits": True, "help": "Make two or more parts one (locked). Parts are renumbered"},
    "parts.halve": {"fn": parts_halve, "args": _schema({"part": {"type": "integer", "minimum": 0}}, ["part"]), "edits": True, "help": "Split a part in two along its most natural border. Both halves are locked. Parts are renumbered"},
    "parts.from_faces": {"fn": parts_from_faces, "args": _schema({"faces": _FACES}, ["faces"]), "edits": True, "help": "The faces of the result become a new part (stands in for painting one)"},
    "parts.redo": {"fn": parts_redo, "args": _schema({"parts": _PARTS, "use": {"type": "string", "enum": list(USES) + list(USE_ALIASES), "description": "Remake for this Use; default the part's own"}, "method": {"type": "string", "enum": list(METHODS)}, "detail": {"type": "number", "minimum": -4, "maximum": 5, "description": "0.5 keeps the quad size the part has now"}}, ["parts"]), "edits": True, "help": "Remake parts for a Use or with a method; each result is kept as a try. Runs in the background: poll job.status"},
    "parts.tries": {"fn": parts_tries, "args": _schema({"part": {"type": "integer", "minimum": 0}}, ["part"]), "edits": False, "help": "The results this part has had, and which one is shown"},
    "parts.use_try": {"fn": parts_use_try, "args": _schema({"part": {"type": "integer", "minimum": 0}, "index": {"type": "integer", "minimum": 0}}, ["part", "index"]), "edits": True, "help": "Switch a part to one of its tries"},
    "parts.keep_original": {"fn": parts_keep_original, "args": _schema({"parts": _PARTS}, ["parts"]), "edits": True, "help": "Put your own faces back on these parts (one more try each)"},
}
