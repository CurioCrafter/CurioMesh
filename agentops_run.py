# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""Agent commands for a whole run: session and diagnostics, every setting, Auto and the result, Tidy Up, History and undo,
Bake, LOD and export (AGENT_TOOLS.md section 2). `COMMANDS` is read by `agentcmd`: {name: {"fn": f(context, **args), "args":
json-schema, "help": str, "edits": bool}}.

Every edit goes through the operator or function a click uses, so History and Ctrl+Z come for free. The helpers at the top
need no Blender and are unit tested; everything that touches Blender is below and imports the add-on's modules lazily.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from types import SimpleNamespace


class AgentError(Exception):
    """An error `agentcmd` turns into a coded reply (section 5)."""

    def __init__(self, code, message, hint="", details=None):
        super().__init__(message)
        self.agent_code, self.hint, self.details = code, hint, details or {}


# --------------------------------------------------------------------------- pure helpers

UNBOUNDED = 1e30  # RNA gives -3.4e38 / 3.4e38 (floats) and +-2**31 (ints) for "no limit"
USES = ("AUTO", "GAME", "LOW_POLY", "ANIMATION", "SUBDIV", "SCAN", "SCULPT", "BLOCKOUT", "HARD", "HULL")
EFFORTS = ("AUTO", "LIGHT", "NORMAL", "FULL")
DEFECT_KINDS = ("inward", "sunk", "crossing", "folded_over", "bent", "flat", "duplicate", "overlap")


def to_json(value):
    """A property value as plain JSON: vectors and arrays become lists, sets sorted lists."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return [to_json(v) for v in value]
    except TypeError:
        return str(value)


def _limit(value, ints: bool):
    """A range end, or None when RNA means 'no limit'."""
    if value is None or abs(value) >= (2**31 - 1 if ints else UNBOUNDED):
        return None
    return int(value) if ints else float(value)


def prop_info(prop, value, default=None) -> dict:
    """One property as an agent sees it, from Blender's own description of it (`prop` is a bpy Property or anything with the
    same attributes). Nothing is listed by hand, so new options appear on their own."""
    kind = prop.type
    info = {"name": prop.identifier, "label": prop.name, "help": prop.description, "type": kind, "value": to_json(value), "default": to_json(default)}
    if getattr(prop, "is_readonly", False):
        info["readonly"] = True
    if "HIDDEN" in getattr(prop, "options", ()):
        info["hidden"] = True
    if getattr(prop, "array_length", 0):
        info["array"] = prop.array_length
    if kind in ("INT", "FLOAT"):
        ints = kind == "INT"
        info.update(min=_limit(prop.hard_min, ints), max=_limit(prop.hard_max, ints), soft_min=_limit(prop.soft_min, ints), soft_max=_limit(prop.soft_max, ints))
    elif kind == "ENUM":
        info["items"] = [{"id": i.identifier, "label": i.name, "help": i.description} for i in prop.enum_items]
        if getattr(prop, "is_enum_flag", False):
            info["flags"] = True
    elif kind == "STRING" and getattr(prop, "length_max", 0):
        info["max_length"] = prop.length_max
    return info


def bad_value(name, value, why, **details):
    return AgentError("BAD_ARGS", f"{name}: {why}", "Use a value inside the range or one of the choices that settings.list shows", dict(details, name=name, got=to_json(value)))


def _number(info, value):
    ints = info["type"] == "INT"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise bad_value(info["name"], value, "must be a number", min=info.get("min"), max=info.get("max"))
    if ints and float(value) != int(value):
        raise bad_value(info["name"], value, "must be a whole number", min=info.get("min"), max=info.get("max"))
    value = int(value) if ints else float(value)
    lo, hi = info.get("min"), info.get("max")
    if value != value or (lo is not None and value < lo) or (hi is not None and value > hi):
        raise bad_value(info["name"], value, f"must be between {lo} and {hi}", min=lo, max=hi, soft_min=info.get("soft_min"), soft_max=info.get("soft_max"))
    return value


def check_setting(info: dict, value):
    """The value ready for setattr, or BAD_ARGS naming the allowed range or choices. Also refuses read-only and hidden (internal) options."""
    name, kind = info["name"], info["type"]
    if kind in ("POINTER", "COLLECTION"):
        raise bad_value(name, value, "is a link to other data and cannot be set here")
    if info.get("readonly") or info.get("hidden"):
        raise bad_value(name, value, "is internal and cannot be set")
    n = info.get("array", 0)
    if n:
        if not isinstance(value, (list, tuple)) or len(value) != n:
            raise bad_value(name, value, f"must be a list of {n} numbers")
        one = dict(info, array=0)
        if kind == "BOOLEAN":
            return [bool(v) if isinstance(v, bool) else _raise(bad_value(name, v, "must be true or false")) for v in value]
        return [_number(one, v) for v in value]
    if kind == "BOOLEAN":
        if not isinstance(value, bool):
            raise bad_value(name, value, "must be true or false")
        return value
    if kind in ("INT", "FLOAT"):
        return _number(info, value)
    if kind == "STRING":
        if not isinstance(value, str):
            raise bad_value(name, value, "must be text")
        if info.get("max_length") and len(value) > info["max_length"]:
            raise bad_value(name, value, f"is longer than {info['max_length']} characters", max_length=info["max_length"])
        return value
    if kind == "ENUM":
        ids = [i["id"] for i in info.get("items", [])]
        if info.get("flags"):
            if not isinstance(value, (list, tuple)) or any(v not in ids for v in value):
                raise bad_value(name, value, "must be a list of choices", choices=ids)
            return set(value)
        if value not in ids:
            raise bad_value(name, value, "is not one of the choices", choices=ids)
        return value
    raise bad_value(name, value, f"has a kind ({kind}) that cannot be set here")


def _raise(exc):
    raise exc


def history_rows(rows: list[dict], current_step: int) -> dict:
    """The History reply: rows as `history.listing` makes them, plus the step on screen and what undo and redo would reach."""
    here = next((r for r in reversed(rows) if r["step"] == current_step), None)
    i = None if here is None else here["index"]
    return {
        "steps": rows,
        "current": None if here is None else {"index": i, "step": here["step"], "name": here["name"], "faces": here["faces"]},
        "can_undo": i is not None and i > 0,
        "can_redo": i is not None and i + 1 < len(rows) and rows[i + 1]["step"] > current_step,
    }


def find_step(rows: list[dict], step=None, index=None) -> dict:
    """The row for a step number (the newest entry with it) or a list index. NOT_FOUND names the steps there are."""
    if (step is None) == (index is None):
        raise AgentError("BAD_ARGS", "give either step or index", "Both are in history.list", {"allowed": ["step", "index"]})
    if index is not None:
        hit = rows[index] if isinstance(index, int) and 0 <= index < len(rows) else None
    else:
        hit = next((r for r in reversed(rows) if r["step"] == step), None)
    if hit is None:
        raise AgentError("NOT_FOUND", f"no History entry for {'index' if index is not None else 'step'} {index if index is not None else step}", "Call history.list", {"count": len(rows), "steps": [r["step"] for r in rows][-50:]})
    return hit


def status_shape(kinds: list, fraction=None, words: str = "", seconds: float = 0.0, extra=None) -> dict:
    """The job.status reply. `kinds`: what is running now, the first being the main one ("auto", "redo", "tidy", "autofix", "lod"...)."""
    out = {"running": bool(kinds), "kind": kinds[0] if kinds else None, "jobs": list(kinds), "percent": None if fraction is None else round(100.0 * fraction, 1), "words": words, "seconds": round(seconds, 1)}
    out.update(extra or {})
    return out


def issues_summary(record, top: int = 10) -> dict:
    """Totals per kind, the spot count, bad kinds per part, and the worst spots (no face lists)."""
    record = record or {}
    spots = record.get("spots", [])
    per_part: dict = {}
    for s in spots:
        bucket = per_part.setdefault(str(s.get("part", -1)), {})
        for kind, count in s.get("kinds", {}).items():
            bucket[kind] = bucket.get(kind, 0) + int(count)
    worst = [{k: s.get(k) for k in ("part", "kinds", "face_count", "score", "center", "radius")} for s in spots[:top]]
    return {"totals": dict(record.get("totals", {})), "spots": len(spots), "per_part": per_part, "worst": worst}


def face_counts(faces) -> dict:
    sizes = [len(f) for f in faces]
    n = max(len(sizes), 1)
    return {"faces": len(sizes), "quads": sizes.count(4), "tris": sizes.count(3), "ngons": sum(1 for s in sizes if s > 4), "quad_pct": round(100.0 * sizes.count(4) / n, 1)}


def tidy_levels(levels: dict, shown: float) -> list[dict]:
    """The scored Tidy Up strengths, as plain rows."""
    return [{"strength": s, "shown": abs(s - shown) < 1e-9, **{k: (round(float(row[k]), 2) if isinstance(row.get(k), (int, float)) else row.get(k)) for k in ("tidy", "view", "poles", "faces") if row.get(k) is not None}} for s, row in sorted(levels.items())]


# --------------------------------------------------------------------------- Blender side


def _m() -> SimpleNamespace:
    """The add-on's modules, imported when first needed (they import bpy; this file must load without it)."""
    names = ("guided", "history", "organize", "ladder", "segments", "autofix_live", "tidylive", "overlay", "hardware", "jobpack", "profiles", "verdict", "autoretopo", "blockout", "game_export")
    return SimpleNamespace(**{n: importlib.import_module("." + n, __package__) for n in names})


LOG: list[str] = []  # CurioMesh's own messages as the agent met them (diag.log)


def _log(text: str) -> None:
    text = str(text).strip()
    if text and (not LOG or LOG[-1].split(" ", 1)[-1] != text):
        LOG.append(f"{time.strftime('%H:%M:%S')} {text}")
        del LOG[:-500]


def _cm():
    """The agent scripts' loaders (skills/curiomesh-retopo/scripts/cm_agent.py), imported by path. They sit outside the add-on zip."""
    import importlib.util

    path = Path(__file__).resolve().parent / "skills" / "curiomesh-retopo" / "scripts" / "cm_agent.py"
    if not path.exists():
        raise AgentError("NOT_READY", "file loading and exporting need skills/curiomesh-retopo/scripts/cm_agent.py next to the add-on", "Use a git checkout of CurioMesh, or open the file in Blender yourself", {"missing": str(path)})
    spec = importlib.util.spec_from_file_location("curiomesh_cm_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ops_error(exc: Exception, code: str = "REFUSED") -> AgentError:
    text = str(exc).replace("Error: ", "", 1).strip()
    if "made from a mesh" in text:
        return AgentError("MISMATCH", text, "Import the result into the mesh the job was exported from", {})
    if "damaged" in text or "Cannot read" in text or "No such file" in text:
        return AgentError("FILE", text, "Check the path and that the file is complete", {})
    return AgentError(code, text or "the operation was refused", "", {})


def _obj(context, name=None):
    """The mesh to work on: the named one, or the active mesh (a CurioMesh preview never counts)."""
    import bpy

    a = _m()
    if name:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise AgentError("NO_OBJECT", f"no mesh called {name!r}", "scene.objects lists the meshes", {"object": name})
    else:
        obj = a.guided.state_source() or context.object
        if obj is None or obj.type != "MESH" or a.overlay.is_preview(obj):
            raise AgentError("NO_OBJECT", "no mesh is active", "Pass object, or call scene.select first", {})
    if context.object is not obj or not obj.select_get():
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.scene.objects:
            o.select_set(o is obj)
        context.view_layer.objects.active = obj
    return obj


def _gs(context):
    return context.scene.curiomesh_guided


def _step(context) -> int:
    return _m().history.current(context.scene)


def _need_result():
    a = _m()
    if a.guided.state["res"] is None or a.guided.state["auto_base"] is None:
        raise AgentError("NO_RESULT", "there is no result on screen", "Run auto.start and job.wait first", {})
    return a.guided.state["res"], a.guided.state["auto_base"]


def _runner_kind(runner, lod) -> str:
    if lod is not None:
        return "lod"
    a = _m()
    if isinstance(runner, a.autoretopo.AutoJob):
        return "auto"
    if isinstance(runner, a.blockout.BlockoutJob):
        return "blockout"
    if isinstance(runner, a.segments.PartJob):
        return "part_redo"
    return "find_parts" if "Find" in type(runner).__name__ or "Show" in type(runner).__name__ else "redo"


def status(context=None) -> dict:
    a = _m()
    kinds, fraction, words, seconds = [], None, "", 0.0
    work = a.guided.state["work"]
    info = a.guided.progress_info()
    if work is not None and work["runner"].busy:
        runner = work["runner"]
        kinds.append(_runner_kind(runner, a.guided.state["lod"]))
        if info:
            fraction, words = info
        seconds = max(0.0, time.perf_counter() - getattr(runner, "started", time.perf_counter()))
    org = a.organize.org
    if org["keep"] or (a.organize.active() and org["box"] is not None and not a.organize.ready()):
        kinds.append("tidy")
        words = words or a.organize.status() or "Keep Tidying"
    if a.autofix_live.busy():
        kinds.append("autofix")
        words = words or "Auto Fix"
    extra = {}
    if not kinds:
        extra["message"] = a.guided.state["message"]
        if a.guided.state["lod_report"]:
            extra["lod_report"] = a.guided.state["lod_report"]
        extra["has_result"] = a.guided.state["res"] is not None
    if a.ladder.active():
        extra["sizes_working"] = bool(a.ladder.lad["job"] is not None or a.ladder.lad["plan"])
    return status_shape(kinds, fraction, words, seconds, extra)


def _idle(context=None) -> None:
    st = status()
    if st["running"]:
        raise AgentError("BUSY", f"{st['kind']} is running", "Poll job.status, use job.wait, or job.stop", st)


def _pump(slice_s: float = 0.05) -> None:
    """One look at everything that works in the background, stepping what a timer would (a background Blender has no timers)."""
    import bpy

    a = _m()
    context = bpy.context
    work = a.guided.state["work"]
    waited = False
    if work is not None:
        runner = work["runner"]
        if runner.busy:
            if hasattr(runner, "step"):
                runner.step(slice_s)
            else:
                waited = True
        a.guided._collect_work(context)
    if a.organize.org["box"] is not None:
        a.organize._tick()
    if a.autofix_live.busy():
        a.autofix_live.fx["job"].step(slice_s)
    if waited or a.organize.org["box"] is not None:
        time.sleep(0.02)  # a worker thread is doing the work


def _settle_sliders() -> None:
    import bpy

    a = _m()
    if a.ladder.active() and a.ladder.drag["pending"]:
        a.ladder.drag["pending"] = False
        a.ladder.show(bpy.context)


# ----- commands: the registry


COMMANDS: dict = {}
OBJ = {"object": {"type": "string", "description": "Mesh name; without it the active mesh"}}


def command(name, help, props=None, required=(), edits=False):
    def wrap(fn):
        COMMANDS[name] = {"fn": fn, "help": help, "edits": edits, "args": {"type": "object", "properties": {**OBJ, **(props or {})}, "required": list(required)}}
        return fn

    return wrap


# ----- session and diagnostics


@command("diag.check_pc", "Cores, memory, graphics card, a short timing test and the loose pieces of the mesh, as Check This Computer.")
def diag_check_pc(context, object=None):
    import bpy

    a = _m()
    try:
        _obj(context, object)
    except AgentError:
        pass
    bpy.ops.curiomesh.guided_check_pc()
    machine = a.guided.state["machine"] or {}
    return {"machine": to_json({k: v for k, v in machine.items() if isinstance(v, (str, int, float, bool, list, tuple, type(None)))}), "words": a.hardware.describe(machine), "rays_per_second": a.guided.state["rays"], "loose_pieces": (a.guided.state["parts"] or (None, None))[1]}


@command("diag.log", "The last lines of CurioMesh's own messages: what the panel said, refusals, warnings.", {"lines": {"type": "integer", "minimum": 1}})
def diag_log(context, object=None, lines=50):
    a = _m()
    now = [t for t in (a.guided.state["message"], a.overlay.LAST_DRAW_ERROR, a.organize.org["note"]) if t]
    return {"lines": LOG[-lines:], "now": now}


@command("scene.objects", "The meshes in the scene with their face counts, and which is active.")
def scene_objects(context, object=None):
    a = _m()
    rows = [{"name": o.name, "faces": len(o.data.polygons), "verts": len(o.data.vertices), "active": o is context.object, "selected": o.select_get(), "hidden": o.hide_get()} for o in context.scene.objects if o.type == "MESH" and not a.overlay.is_preview(o)]
    return {"objects": rows}


@command("scene.select", "Make one mesh active and selected.", {}, ("object",))
def scene_select(context, object=None):
    obj = _obj(context, object)
    return {"object": obj.name, "faces": len(obj.data.polygons)}


@command("file.load", "Import obj, fbx, glb, gltf, ply, stl, blend or npz as ONE mesh and make it active.", {"path": {"type": "string"}, "name": {"type": "string"}}, ("path",), edits=True)
def file_load(context, path, name=None, object=None):
    _idle()
    p = Path(path)
    if not p.exists():
        raise AgentError("FILE", f"no file at {path}", "Pass the full path", {"path": path})
    cm = _cm()
    try:
        obj = cm.load_mesh(str(p), name or p.stem)
    except (ValueError, RuntimeError) as exc:
        raise AgentError("FILE", str(exc), "Formats: " + ", ".join(sorted(cm.IMPORTERS)), {"path": path}) from exc
    return {"object": obj.name, "faces": len(obj.data.polygons)}


@command("file.save", "Save the .blend.", {"path": {"type": "string"}, "copy": {"type": "boolean"}}, ("path",))
def file_save(context, path, copy=False, object=None):
    import bpy

    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(path).with_suffix(".blend")), copy=copy)
    except RuntimeError as exc:
        raise _ops_error(exc, "FILE") from exc
    return {"saved": str(Path(path).with_suffix(".blend"))}


# ----- settings: every option, read from Blender's own description of it


def _group_owner(context, group, object=None):
    scene = context.scene
    if group == "part":
        return _obj(context, object).curiomesh_seg
    if group == "prefs":
        return context.preferences.addons[__package__].preferences
    attr = {"guided": "curiomesh_guided", "classic": "curiomesh_settings", "tools": "curiomesh_tools", "autofix": "curiomesh_autofix", "fix": "curiomesh_fix", "region": "curiomesh_region"}.get(group)
    if attr is None:
        raise AgentError("BAD_ARGS", f"unknown group {group!r}", "settings.groups lists them", {"allowed": list(GROUPS)})
    owner = getattr(scene, attr, None)
    if owner is None:
        raise AgentError("NOT_FOUND", f"the {group} settings are not registered", "", {})
    return owner


GROUPS = {"guided": "Auto Segments, the Use, Detail, Mirror, Tidy Up, Bake and every search option", "classic": "The classic remesh settings", "tools": "Tidy Tools scope and History view", "autofix": "The seven Auto Fix groups", "fix": "Fix Selection", "region": "Regions", "part": "Part settings of one mesh (segments)", "prefs": "Add-on preferences"}


def _props(owner):
    return [p for p in owner.bl_rna.properties if p.identifier not in ("rna_type", "name") and p.type not in ("POINTER", "COLLECTION")]


def _default(owner, prop):
    try:
        if getattr(prop, "array_length", 0):
            return list(prop.default_array)
        if prop.type == "ENUM" and prop.is_enum_flag:
            return sorted(prop.default_flag)
        return prop.default
    except Exception:
        return None


def _infos(owner, names=None):
    rows = []
    for p in _props(owner):
        if names is None or p.identifier in names:
            rows.append(prop_info(p, getattr(owner, p.identifier), _default(owner, p)))
    return rows


@command("settings.groups", "The groups of settings and what each holds.")
def settings_groups(context, object=None):
    return {"groups": [{"group": g, "help": h} for g, h in GROUPS.items()]}


@command("settings.list", "Every property of a group, read from Blender: label, help, type, value, default, soft and hard range, choices with their help.", {"group": {"type": "string"}}, ("group",))
def settings_list(context, group, object=None):
    owner = _group_owner(context, group, object)
    return {"group": group, "settings": _infos(owner)}


@command("settings.get", "Read settings of a group (all, or the names given).", {"group": {"type": "string"}, "names": {"type": "array", "items": {"type": "string"}}}, ("group",))
def settings_get(context, group, names=None, object=None):
    owner = _group_owner(context, group, object)
    infos = {i["name"]: i for i in _infos(owner, names)}
    missing = [n for n in names or () if n not in infos]
    if missing:
        raise AgentError("BAD_ARGS", "unknown setting: " + ", ".join(missing), "settings.list shows the names", {"unknown": missing, "allowed": sorted(i for i in infos)[:200]})
    return {"group": group, "values": {n: i["value"] for n, i in infos.items()}}


def set_values(context, group, values, object=None) -> dict:
    """Check every value first, then set them with setattr, so each property's own update runs (preview, History note). One bad value changes nothing."""
    owner = _group_owner(context, group, object)
    infos = {i["name"]: i for i in _infos(owner)}
    unknown = [n for n in values if n not in infos]
    if unknown:
        raise AgentError("BAD_ARGS", "unknown setting: " + ", ".join(unknown), "settings.list shows the names", {"unknown": unknown, "allowed": sorted(infos)[:300]})
    ready = {n: check_setting(infos[n], v) for n, v in values.items()}
    order = sorted(ready, key=lambda n: n != "auto_profile")  # the Use first: it sets Detail, Method and Output
    old = {n: getattr(owner, n) for n in order}
    done = []
    try:
        for n in order:
            setattr(owner, n, ready[n])
            done.append(n)
    except Exception as exc:
        for n in reversed(done):
            setattr(owner, n, old[n])
        raise AgentError("REFUSED", f"{n}: {exc}", "Nothing was changed", {"name": n}) from exc
    return {"group": group, "changed": {n: {"old": to_json(old[n]), "new": to_json(getattr(owner, n))} for n in order}}


@command("settings.set", "Set options of a group. Goes through each property's own update, so sliders reshape the preview and write their History note as a drag does. A bad value changes nothing.", {"group": {"type": "string"}, "values": {"type": "object"}}, ("group", "values"), edits=True)
def settings_set(context, group, values, object=None):
    out = set_values(context, group, values, object)
    out["step"] = _step(context)
    _log(f"settings.set {group} {sorted(values)}")
    return out


@command("use.pick", "Pick the Use: " + ", ".join(USES) + ". AUTO lets Start look at the mesh and choose.", {"use": {"type": "string", "enum": list(USES)}}, ("use",), edits=True)
def use_pick(context, use, object=None):
    _idle()
    gs = _gs(context)
    if use == "AUTO":
        gs.use_auto = True
        return {"use": "AUTO", "note": "auto.start_for_me picks the Use and says why"}
    ids = [i.identifier for i in gs.bl_rna.properties["auto_profile"].enum_items]
    if use not in ids:
        raise AgentError("BAD_ARGS", f"{use} is not a Use in this version", "", {"allowed": ids})
    gs.auto_profile = use
    gs.use_auto = False
    return {"use": gs.auto_profile, "detail": gs.auto_detail, "method": gs.auto_method, "output": gs.auto_output}


# ----- Auto and the job


def _started(ok, context) -> dict:
    if not ok:
        msg = _m().guided.state["message"] or "Start was refused"
        raise AgentError("NO_OBJECT" if "mesh" in msg.lower() and "face" in msg.lower() else "REFUSED", msg, "", {})
    _log(_m().guided.state["message"])
    return {"started": True, "running": True, "status": status()}


@command("auto.start", "Start Auto Segments on the mesh with the Use and Detail set now. Returns at once; poll job.status or call job.wait.", {"use": {"type": "string", "enum": list(USES[1:])}, "detail": {"type": "number", "minimum": 0.1, "maximum": 80}}, edits=True)
def auto_start(context, object=None, use=None, detail=None):
    import bpy

    _idle()
    _obj(context, object)
    gs = _gs(context)
    if use:
        use_pick(context, use)
    if detail is not None:
        gs.auto_detail = detail
    try:
        r = bpy.ops.curiomesh.guided_auto_start()
    except RuntimeError as exc:
        raise _ops_error(exc) from exc
    return _started(r == {"FINISHED"}, context)


@command("auto.start_for_me", "Start with Auto: it looks at the mesh and this computer, picks the Use and settings, says why, and starts.", edits=True)
def auto_start_for_me(context, object=None):
    import bpy

    a = _m()
    _idle()
    _obj(context, object)
    asked = a.guided.state["no_ask"]
    a.guided.state["no_ask"] = True  # a long job would open a question dialog, which an agent cannot answer
    try:
        r = bpy.ops.curiomesh.guided_auto_all("INVOKE_DEFAULT")
    except RuntimeError as exc:
        raise _ops_error(exc) from exc
    finally:
        a.guided.state["no_ask"] = asked
    out = _started(r == {"FINISHED"}, context)
    gs = _gs(context)
    out.update(use=gs.auto_profile, why=gs.auto_why)
    return out


@command("job.status", "What is running now: kind, percent, plain words (\"Part 3 of 12\"), seconds. Auto, a redo, Auto Fix, Keep Tidying, an LOD set.")
def job_status(context, object=None):
    return status(context)


@command("job.wait", "Headless: step the running work until it is done or this many seconds have passed, then answer with the status. Any number of seconds.", {"seconds": {"type": "number", "minimum": 0}, "sizes": {"type": "boolean", "description": "Also work out the other Quad Size stops"}})
def job_wait(context, seconds=600.0, sizes=False, object=None):
    a = _m()
    ends = time.perf_counter() + seconds
    while status()["running"] and time.perf_counter() < ends:
        _pump()
    if sizes and not status()["running"]:
        a.ladder.finish(timeout=max(0.0, ends - time.perf_counter()))
    st = status()
    st["timed_out"] = st["running"]
    _log(f"job.wait: {a.guided.state['message']}")
    return st


@command("job.stop", "Stop what is running now. Keep Tidying and Auto Fix keep the passes done; stopping the search throws its partial result away.", edits=True)
def job_stop(context, object=None):
    import bpy

    a = _m()
    stopped = []
    if a.organize.org["keep"]:
        bpy.ops.curiomesh.tidy_stop()
        stopped.append("tidy")
    if a.autofix_live.busy():
        a.autofix_live.stop(context)
        stopped.append("autofix")
    work = a.guided.state["work"]
    if work is not None and work["runner"].busy:
        bpy.ops.curiomesh.guided_stop_work()
        stopped.append("work")
    return {"stopped": stopped, "status": status()}


# ----- the result


@command("result.numbers", "Faces, quads, tris, n-gons, quad %, how much the view changed, Most Detail Loss in use, seconds and the verdict.")
def result_numbers(context, object=None):
    a = _m()
    res, base = _need_result()
    report = base["report"]
    gs = _gs(context)
    out = face_counts(res["faces"])
    out.update(source_faces=report.get("before"), view_change_pct=round(float(report["loss"]), 2), tidiness=report.get("topology"), shading_change_pct=report.get("shading"), seconds=round(float(report.get("seconds", 0)), 1), parts=report.get("pieces"))
    org = a.organize
    if org.active() and org.org["shown"] > 0 and org.org["shown"] in org.org["levels"]:
        row, zero = org.org["levels"][org.org["shown"]], org._first() or {}
        if row.get("view") is not None and zero.get("view") is not None:
            out["view_change_pct"] = round(float(report["loss"]) + row["view"] - zero["view"], 2)
        out.update(tidiness=row.get("tidy"), poles=row.get("poles"), tidy={"strength": org.org["shown"], "passes": len(org.org["passes"]) + 1, "text": org.meter_text(gs)})
    out["most_detail_loss"] = {"on": bool(gs.organize_cap_use), "percent": gs.organize_cap}
    asked = gs.auto_detail if gs.auto_aim == "KEEP" and gs.auto_profile not in a.profiles.BLOCKOUT_USES and not gs.multires_on else None
    spots = [s for s in res.get("issue_spots") or [] if any(k in a.verdict.SERIOUS for k in s.get("kinds", {}))]
    level, title = a.verdict.verdict(gs.auto_profile if gs.auto_profile != "CUSTOM" else "", out["faces"], report["loss"], asked, a.verdict.serious_count((res.get("issues") or {}).get("totals")), len(spots), report.get("before"))
    out.update(verdict={"level": level, "title": title}, use=gs.auto_profile, detail_asked=asked, step=_step(context))
    return out


@command("result.report", "The full report: issues (totals, spots, per part, worst), the notes, why Auto picked what it did, the part table summary.", {"worst": {"type": "integer", "minimum": 1}})
def result_report(context, worst=10, object=None):
    a = _m()
    res, base = _need_result()
    report = {k: v for k, v in base["report"].items() if k not in ("guides", "timings", "memory_mb")}
    gs = _gs(context)
    return {"numbers": result_numbers(context), "issues": issues_summary(res.get("issues"), worst), "notes": report.get("notes", []), "auto_why": gs.auto_why, "report": to_json(report), "message": a.guided.state["message"], "applied": a.guided.state["applied"]}


@command("result.update", "Rebuild the preview now (the Update button).", edits=True)
def result_update(context, object=None):
    import bpy

    _idle()
    _obj(context, object)
    try:
        r = bpy.ops.curiomesh.guided_update()
    except RuntimeError as exc:
        raise _ops_error(exc) from exc
    return {"updated": r == {"FINISHED"}, "message": _m().guided.state["message"], "status": status()}


@command("result.discard", "Cancel: remove the preview and keep the mesh as it was.", edits=True)
def result_discard(context, object=None):
    import bpy

    a = _m()
    if a.guided.state["work"] is not None and a.guided.state["work"]["runner"].busy:
        bpy.ops.curiomesh.guided_stop_work()
    bpy.ops.curiomesh.guided_stop()
    return {"discarded": True, "has_result": a.guided.state["res"] is not None}


# ----- size


@command(
    "size.set",
    "Set sizes. detail: the Detail slider (percent of the view that may change; takes effect on the next start, or now with restart). quad_size and rows: the live Quad Size and Rows sliders (0 to 1), once those stops are worked out.",
    {"detail": {"type": "number", "minimum": 0.1, "maximum": 80}, "quad_size": {"type": "number", "minimum": 0, "maximum": 1}, "rows": {"type": "number", "minimum": 0, "maximum": 1}, "restart": {"type": "boolean"}},
    edits=True,
)
def size_set(context, detail=None, quad_size=None, rows=None, restart=False, object=None):
    a = _m()
    if detail is None and quad_size is None and rows is None:
        raise AgentError("BAD_ARGS", "give detail, quad_size or rows", "", {"allowed": ["detail", "quad_size", "rows"]})
    out = {}
    if detail is not None:
        _idle()
        out.update(set_values(context, "guided", {"auto_detail": detail, "use_auto": False, "auto_aim": "KEEP"}, object))
        if restart:
            out["restart"] = auto_start(context, object)
    if quad_size is not None or rows is not None:
        _need_result()
        if not a.ladder.active():
            raise AgentError("NOT_READY", "Quad Size and Rows need a finished search that was not Reduce To, Blockout or Mirror-less chunked", "Use detail with restart instead", {})
        if status()["running"]:
            _idle()
        values = {k: v for k, v in (("quad_size", quad_size), ("quad_order", rows)) if v is not None}
        out.update(set_values(context, "guided", values, object))
        _settle_sliders()
        out["faces"] = len(a.guided.state["res"]["faces"])
        out["sizes_ready"] = {style: a.ladder.ready(style) for style in ("DETAIL", "BALANCED", "ROWS")}
    out["step"] = _step(context)
    return out


@command("size.parts", "Bigger or smaller quads on chosen parts only; the rest stays. Kept at once as one History step.", {"parts": {"type": "array", "items": {"type": "integer"}}, "bigger": {"type": "boolean"}}, ("parts",), edits=True)
def size_parts(context, parts, bigger=True, object=None):
    a = _m()
    _need_result()
    _idle()
    a.segments.ensure()
    if not a.segments.ready():
        raise AgentError("NOT_READY", "this result has no part table to change", "", {})
    if a.segments.mirrored():
        raise AgentError("NOT_READY", "Bigger and Smaller Quads don't work on a mirrored, Even Rows or imported result", "", {})
    count = len(a.segments.seg["parts"])
    bad = [p for p in parts if not 0 <= p < count]
    if bad:
        raise AgentError("NOT_FOUND", "no such part: " + ", ".join(map(str, bad)), "parts.list shows the ids", {"parts": count})
    changed, why = a.ladder.part_sizes(context, sorted(set(parts)), bigger)
    if not changed:
        raise AgentError("REFUSED", why[0] if why else "nothing could change", "Nothing was changed", {"why": why})
    return {"changed": changed, "why": why, "running": status()["running"], "step": _step(context)}


# ----- Tidy Up


def _tidy_ready(op) -> None:
    import bpy

    if not getattr(bpy.ops.curiomesh, op).poll():
        a = _m()
        reason = "Keep Tidying is running" if a.organize.org["keep"] else "the strengths are still being worked out" if a.organize.active() and not a.organize.ready() else "there is no earlier pass" if op == "tidy_back" and a.organize.active() else "Tidy Up needs a finished search that is not Reduce To or Blockout"
        raise AgentError("NOT_READY" if a.guided.state["res"] is not None else "NO_RESULT", reason, "Poll job.status", status())


def _tidy_reply(context) -> dict:
    a = _m()
    return {"running": status()["running"], "passes": len(a.organize.org["passes"]) + (1 if a.organize.active() else 0), "text": a.organize.meter_text(_gs(context)), "note": a.organize.org["note"], "step": _step(context)}


@command("tidy.again", "One more Tidy Up pass from the result on screen (the first pass if none yet, at strength). Run it again for as long as it keeps improving.", {"strength": {"type": "number", "minimum": 0.25, "maximum": 1}}, edits=True)
def tidy_again(context, strength=1.0, object=None):
    import bpy

    a = _m()
    _need_result()
    _idle()
    gs = _gs(context)
    if not a.organize.active():
        set_values(context, "guided", {"organize": strength})  # the slider starts the first pass, as dragging it does
    else:
        _tidy_ready("tidy_again")
        try:
            bpy.ops.curiomesh.tidy_again()
        except RuntimeError as exc:
            raise _ops_error(exc) from exc
    return _tidy_reply(context)


@command("tidy.keep_going", "Keep Tidying: pass after pass while a pass still makes the mesh tidier within Most Detail Loss. Stops by itself when a pass gains nothing.", edits=True)
def tidy_keep_going(context, object=None):
    import bpy

    _need_result()
    _idle()
    _tidy_ready("tidy_keep")
    try:
        r = bpy.ops.curiomesh.tidy_keep()
    except RuntimeError as exc:
        raise _ops_error(exc) from exc
    if r != {"FINISHED"}:
        raise AgentError("NOT_READY", "Tidy Up needs a result from Start", "", {})
    return _tidy_reply(context)


@command("tidy.stop", "Stop Keep Tidying. The passes done so far stay.", edits=True)
def tidy_stop(context, object=None):
    import bpy

    a = _m()
    was = bool(a.organize.org["keep"])
    if was:
        bpy.ops.curiomesh.tidy_stop()
    return dict(_tidy_reply(context), stopped=was)


@command("tidy.back", "Undo Pass: back to the result before the last Tidy pass.", edits=True)
def tidy_back(context, object=None):
    import bpy

    _need_result()
    _idle()
    _tidy_ready("tidy_back")
    bpy.ops.curiomesh.tidy_back()
    return _tidy_reply(context)


@command("tidy.levels", "The scored Tidy Up strengths of the pass on screen: tidiness, view change, poles, faces.")
def tidy_levels_cmd(context, object=None):
    a = _m()
    _need_result()
    if not a.organize.active():
        return {"levels": [], "note": "no Tidy Up pass yet: tidy.again starts one"}
    return {"levels": tidy_levels(a.organize.org["levels"], a.organize.org["shown"]), "ready": a.organize.ready(), "text": a.organize.meter_text(_gs(context))}


# ----- apply, LODs, bake, export


def _write_stats(obj) -> dict:
    sizes = [len(p.vertices) for p in obj.data.polygons]
    return {"object": obj.name, "faces": len(sizes), "quads": sizes.count(4), "tris": sizes.count(3), "ngons": sum(1 for s in sizes if s > 4)}


@command(
    "apply",
    "Apply the result (writes <name>_Retopo, or into your object). The same options as the panel.",
    {"mode": {"type": "string", "enum": ["NEW", "REPLACE"]}, "bake": {"type": "string", "enum": ["OFF", "NORMAL", "NORMAL_COLOR"]}, "bake_size": {"type": "string"}, "uvs": {"type": "boolean", "description": "Keep UV seams"}, "multires": {"type": "integer", "minimum": 0, "description": "Multires levels, 0 off"}},
    edits=True,
)
def apply_result(context, mode=None, bake=None, bake_size=None, uvs=None, multires=None, object=None):
    import bpy

    a = _m()
    _need_result()
    _idle()
    values = {k: v for k, v in (("apply_mode", mode), ("bake_detail", bake), ("bake_size", bake_size), ("keep_uvs", uvs)) if v is not None}
    if multires is not None:
        values["multires_on"] = multires > 0
        if multires > 0:
            values["multires_levels"] = multires
    if values:
        set_values(context, "guided", values)
    try:
        r = bpy.ops.curiomesh.guided_apply()
    except RuntimeError as exc:
        raise _ops_error(exc) from exc
    if r != {"FINISHED"}:
        raise AgentError("REFUSED", a.guided.state["message"] or "Apply was refused", "", {})
    out = bpy.context.view_layer.objects.active
    return dict(_write_stats(out), applied=a.guided.state["applied"], step=_step(context))


@command("lod.make", "Make the LOD set: three objects named _LOD0, _LOD1, _LOD2, each a Reduce To of the mesh at these percents. Returns at once; job.wait finishes it.", {"lod0": {"type": "number", "minimum": 0.1, "maximum": 100}, "lod1": {"type": "number", "minimum": 0.1, "maximum": 100}, "lod2": {"type": "number", "minimum": 0.1, "maximum": 100}}, edits=True)
def lod_make(context, lod0=None, lod1=None, lod2=None, object=None):
    import bpy

    _idle()
    _obj(context, object)
    values = {k: v for k, v in (("lod0_percent", lod0), ("lod1_percent", lod1), ("lod2_percent", lod2)) if v is not None}
    if values:
        set_values(context, "guided", values)
    try:
        r = bpy.ops.curiomesh.guided_make_lods()
    except RuntimeError as exc:
        raise _ops_error(exc) from exc
    if r != {"FINISHED"}:
        raise AgentError("REFUSED", _m().guided.state["message"] or "the LOD set was not started", "", {})
    return {"started": True, "running": True, "status": status()}


@command("bake.set", "Bake settings (the bake happens on apply): detail OFF, NORMAL or NORMAL_COLOR, the texture size, and the number of texture sets.", {"detail": {"type": "string", "enum": ["OFF", "NORMAL", "NORMAL_COLOR"]}, "size": {"type": "string", "enum": ["512", "1024", "2048", "4096"]}, "sets": {"type": "integer", "minimum": 1}}, edits=True)
def bake_set(context, detail=None, size=None, sets=None, object=None):
    values = {k: v for k, v in (("bake_detail", detail), ("bake_size", size), ("bake_sets", sets)) if v is not None}
    if not values:
        gs = _gs(context)
        return {"detail": gs.bake_detail, "size": gs.bake_size, "sets": gs.bake_sets}
    return set_values(context, "guided", values)


def _files(folder: Path) -> set:
    return {p for p in folder.rglob("*") if p.is_file()} if folder.exists() else set()


@command(
    "export.game",
    "Export the mesh and its baked textures for a game engine: Unreal, Unity, Godot, Roblox, glTF, DirectX or Custom (then form, normals, tris and texture count).",
    {"engine": {"type": "string"}, "form": {"type": "string", "enum": ["FBX", "GLB", "OBJ"]}, "normals": {"type": "string", "enum": ["OPENGL", "DIRECTX"]}, "tris": {"type": "integer", "minimum": 0}, "texture": {"type": "string"}, "folder": {"type": "string"}},
    ("engine", "folder"),
    edits=True,
)
def export_game(context, engine, folder, form=None, normals=None, tris=None, texture=None, object=None):
    import bpy

    a = _m()
    _idle()
    obj = _obj(context, object)
    if engine not in a.game_export.ENGINES:
        raise AgentError("BAD_ARGS", f"unknown engine {engine!r}", "", {"allowed": list(a.game_export.ENGINES)})
    kind = (form or "FBX") if engine == "CUSTOM" else a.game_export.ENGINES[engine].form
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    before = _files(folder)
    extra = {k: v for k, v in (("form", form), ("normals", normals), ("tris", tris), ("texture", texture)) if v is not None and engine == "CUSTOM"}
    try:
        bpy.ops.curiomesh.export_game(filepath=str(folder / f"{obj.name}.{kind.lower()}"), engine=engine, **extra)
    except RuntimeError as exc:
        raise _ops_error(exc, "FILE") from exc
    return {"engine": engine, "folder": str(folder), "files": sorted(str(p.relative_to(folder)) for p in _files(folder) - before)}


@command("export.file", "Write the mesh as obj, fbx, glb or blend. A glb is triangles: quads are lost, and the reply says so.", {"path": {"type": "string"}}, ("path",))
def export_file(context, path, object=None):
    obj = _obj(context, object)
    ext = Path(path).suffix.lower()
    if ext not in (".glb", ".gltf", ".fbx", ".obj", ".blend"):
        raise AgentError("BAD_ARGS", f"cannot write {ext or 'a file without an extension'}", "", {"allowed": [".glb", ".gltf", ".fbx", ".obj", ".blend"]})
    stats = _write_stats(obj)
    try:
        written = _cm().export_mesh(obj, path)
    except (ValueError, RuntimeError, OSError) as exc:
        raise AgentError("FILE", str(exc), "", {"path": path}) from exc
    out = {"file": written, **stats}
    if ext in (".glb", ".gltf"):
        out["note"] = f"glTF holds triangles only: the {stats['quads']:,} quads are written as {stats['quads'] * 2 + stats['tris']:,}+ triangles. Use fbx or obj to keep quads"
    return out


@command("job.export_folder", "Export Job: write a folder (mesh, settings, run_job.py) to run the search on another computer. effort: AUTO, LIGHT, NORMAL or FULL on that computer.", {"folder": {"type": "string"}, "effort": {"type": "string", "enum": list(EFFORTS)}}, ("folder",))
def job_export_folder(context, folder, effort="FULL", object=None):
    import bpy

    a = _m()
    obj = _obj(context, object)
    try:
        bpy.ops.curiomesh.guided_export_job(directory=str(folder))
    except RuntimeError as exc:
        raise _ops_error(exc, "FILE") from exc
    made = a.jobpack.job_folder(folder, obj.name)
    if not made.exists():
        raise AgentError("REFUSED", a.guided.state["message"] or "the job was not written", "", {})
    if effort != "FULL":
        a.jobpack.set_effort(made, effort)
    return {"folder": str(made), "effort": effort, "files": sorted(p.name for p in made.iterdir()), "how": "Copy the folder to the other computer and run: blender --background --python run_job.py"}


@command("job.import_result", "Import Job Result: bring result.npz back as the preview of the mesh it was made from. One History step; apply keeps it.", {"path": {"type": "string"}}, ("path",), edits=True)
def job_import_result(context, path, object=None):
    import bpy

    a = _m()
    _idle()
    _obj(context, object)
    if not Path(path).exists():
        raise AgentError("FILE", f"no file at {path}", "", {"path": path})
    a.history.before(context)
    try:
        r = bpy.ops.curiomesh.guided_import_result(filepath=str(path))
    except RuntimeError as exc:
        raise _ops_error(exc, "FILE") from exc
    if r != {"FINISHED"}:
        raise AgentError("FILE", a.guided.state["message"] or "the result could not be imported", "", {"path": path})
    res = a.guided.state["res"]
    step = a.history.record(context, "Import Job Result", f"{len(res['faces']):,} faces", push=True)
    return dict(face_counts(res["faces"]), step=step, message=a.guided.state["message"])


# ----- History and undo


@command("history.list", "Every History step, oldest first: index, step number, name, numbers, faces at that step, which one is on screen, which are undone.")
def history_list(context, object=None):
    a = _m()
    return history_rows(a.history.listing(context.scene), a.history.current(context.scene))


@command("history.jump", "Go to any History step (by step number or list index), as the panel does. It is a new step, so undo takes the jump back. Works headless.", {"step": {"type": "integer"}, "index": {"type": "integer"}}, edits=True)
def history_jump(context, step=None, index=None, object=None):
    a = _m()
    _idle()
    row = find_step(a.history.listing(context.scene), step, index)
    if not a.history.jump(context, row["index"]):
        raise AgentError("NOT_FOUND", f"History entry {row['index']} cannot be shown", "Its result was let go to free memory, or it is waiting for a slider", {"index": row["index"]})
    return dict(history_rows(a.history.listing(context.scene), a.history.current(context.scene)), faces=len(a.guided.state["res"]["faces"]) if a.guided.state["res"] is not None else None)


def _walk(context, back: bool) -> dict:
    import bpy

    a = _m()
    _idle()
    rows = a.history.listing(context.scene)
    info = history_rows(rows, a.history.current(context.scene))
    if not (info["can_undo"] if back else info["can_redo"]):
        raise AgentError("NOT_FOUND", "nothing to undo" if back else "nothing to redo", "history.list shows the steps", {"current": info["current"]})
    target = rows[info["current"]["index"] + (-1 if back else 1)]
    window = None if bpy.app.background else next(iter(context.window_manager.windows), None)
    if window is not None:
        try:
            with context.temp_override(window=window):
                (bpy.ops.ed.undo if back else bpy.ops.ed.redo)()  # Blender's own; its handler puts CurioMesh's state back
            window = True
        except RuntimeError:
            window = None
    if window is None:
        a.history.step_to(context, target["step"])  # the same, for a Blender with no undo stack
    res = a.guided.state["res"]
    return dict(history_rows(a.history.listing(context.scene), a.history.current(context.scene)), faces=None if res is None else len(res["faces"]))


@command("history.undo", "Ctrl+Z. In a window it is Blender's own undo; headless it steps back to the entry before.", edits=True)
def history_undo(context, object=None):
    return _walk(context, True)


@command("history.redo", "Ctrl+Shift+Z. In a window it is Blender's own redo; headless it steps forward to the next entry.", edits=True)
def history_redo(context, object=None):
    return _walk(context, False)
