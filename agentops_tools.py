# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""Agent commands for the Tidy Tools, Face Loops, the Flow Editor and Auto Fix (bench/tools_design/AGENT_TOOLS.md 2 and 5).

Every command here calls the code a click calls: the tool's own `Trial` through `tidylive.work_out` (the straight-through
path F9 uses), `faceloops.edit` / `ring` / `redraw_rows` / `move_pole`, `flowedit.add_stroke`, `autofix_live.start`. Nothing
is copied. A kept run is one History step and one Blender undo step; a refused change keeps nothing.

COMMANDS = {name: {"fn": f(context, **args), "args": json-schema, "help": str, "edits": bool}}, as `agentcmd` reads it.
Errors are `ToolError`s: any exception with `agent_code`, `hint` and `details` becomes a coded reply, so this file needs
nothing from `agentcmd`. No bpy at import (the helpers at the top are plain Python and unit tested); every command
imports the add-on modules when it runs.
"""

from __future__ import annotations

import difflib
import json
import math
import time
from types import SimpleNamespace

import numpy as np

SETTLED = 0.005  # of an edge: a pass that moves no corner further than this gained nothing (as Source Relax's own Keep going)
SECONDS = 600.0  # time a Keep going run works before it says so and offers to go on (it never stops silently)
HARD_EDGES = "curiomesh.tool_hard_edges"
EVERY = "curiomesh.tool_every"
HUGE = 1.0e30  # Blender's "no limit" for a float property is about 3.4e38, for an int 2147483647
SCOPES = ("SELECTION", "PART", "WHOLE")


class ToolError(Exception):
    """Handlers raise this; `agentcmd` turns any exception with `agent_code` into the reply."""

    def __init__(self, code: str, message: str, hint: str = "", details: dict | None = None):
        super().__init__(message)
        self.agent_code, self.hint, self.details = code, hint, details or {}


# --------------------------------------------------------------------------- plain helpers (no bpy)


def _py(x):
    """JSON-safe copy: numpy to plain numbers and lists, NaN and inf to None."""
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, np.generic):
        return _py(x.item())
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _limit(value, whole: bool):
    """A property's hard limit, or None when Blender means 'no limit'."""
    if value is None:
        return None
    if abs(float(value)) >= (2**31 - 1 if whole else HUGE):
        return None
    return int(value) if whole else float(value)


def settings_schema(props) -> list:
    """One entry per setting from Blender's own description of an operator's properties (`bl_rna.properties`): name,
    label, help, type, default, soft and hard range, enum choices. Hidden and read-only properties are left out."""
    kinds = {"FLOAT": "float", "INT": "int", "BOOLEAN": "bool", "STRING": "string", "ENUM": "enum"}
    out = []
    for p in props:
        if p.identifier == "rna_type" or getattr(p, "is_hidden", False) or getattr(p, "is_readonly", False) or p.type not in kinds:
            continue
        kind = kinds[p.type]
        entry = {"name": p.identifier, "label": p.name, "help": p.description, "type": kind, "default": _py(p.default)}
        if kind in ("float", "int"):
            whole = kind == "int"
            entry.update(min=_limit(p.hard_min, whole), max=_limit(p.hard_max, whole), soft_min=_limit(p.soft_min, whole), soft_max=_limit(p.soft_max, whole))
        if kind == "enum":
            entry["choices"] = [{"value": c.identifier, "label": c.name, "help": c.description} for c in p.enum_items]
        out.append(entry)
    return out


def check_settings(schema: list, given: dict | None) -> dict:
    """The settings an agent gave, checked against `schema`: unknown names, wrong types, values outside the hard range or
    not among the enum choices give BAD_ARGS with what is allowed. Returns them with floats and ints made plain."""
    given = {} if given is None else given
    if not isinstance(given, dict):
        raise ToolError("BAD_ARGS", "settings must be an object of name: value", details={"got": type(given).__name__})
    by = {e["name"]: e for e in schema}
    out = {}
    for name, value in given.items():
        e = by.get(name)
        if e is None:
            close = difflib.get_close_matches(str(name), list(by), n=3, cutoff=0.5)
            raise ToolError("BAD_ARGS", f"unknown setting {name!r}", "Use tools.list to see this tool's settings", {"argument": name, "allowed": sorted(by), "close": close})
        kind = e["type"]
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if kind == "bool" and not isinstance(value, bool):
            raise ToolError("BAD_ARGS", f"{name} must be true or false", details={"argument": name, "type": "boolean"})
        if kind in ("string", "enum") and not isinstance(value, str):
            raise ToolError("BAD_ARGS", f"{name} must be text", details={"argument": name, "type": "string"})
        if kind == "float" and not (number and math.isfinite(value)):
            raise ToolError("BAD_ARGS", f"{name} must be a number", details={"argument": name, "type": "number"})
        if kind == "int" and not (number and math.isfinite(value) and float(value) == int(value)):
            raise ToolError("BAD_ARGS", f"{name} must be a whole number", details={"argument": name, "type": "integer"})
        if kind == "enum" and value not in {c["value"] for c in e["choices"]}:
            raise ToolError("BAD_ARGS", f"{name} must be one of {[c['value'] for c in e['choices']]}", details={"argument": name, "allowed": [c["value"] for c in e["choices"]]})
        if kind in ("float", "int"):
            lo, hi = e.get("min"), e.get("max")
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                raise ToolError("BAD_ARGS", f"{name} must be between {lo} and {hi}", "Soft limits only mark the slider's usual range; the hard range is the allowed one", {"argument": name, "minimum": lo, "maximum": hi, "soft_minimum": e.get("soft_min"), "soft_maximum": e.get("soft_max")})
            value = int(value) if kind == "int" else float(value)
        out[name] = value
    return out


def _trial_mod():
    try:
        from .triadq import trial
    except ImportError:  # run outside the add-on package (the unit tests)
        from triadq import trial
    return trial


def gained(cand, least: float = SETTLED) -> bool:
    """True when a pass made something better: any of the card's numbers on the up side (poles gone, Tidiness, Loop Quality,
    fewer bad faces, the tool's own number). With no numbers measured, new faces or a corner moved further than `least` of an
    edge. A pass that only moves things about, or makes them worse, gained nothing."""
    if cand is None or cand.co is None or cand.refusal:
        return False
    numbers = getattr(cand, "numbers", None) or {}
    if numbers:
        number_text = _trial_mod().number_text
        return any(number_text(k, v)[1] > 0 for k, v in numbers.items())
    if cand.faces is not None:
        return True
    moved = getattr(cand, "moved", None)
    return moved is not None and len(moved) > 0 and float(np.nanmax(moved)) > least


def run_passes(do_pass, keep_going: bool, seconds: float | None = None, clock=time.monotonic):
    """Run `do_pass()` -> {"gain": bool, ...} once, or with `keep_going` again until a pass gains nothing (or the time
    budget `seconds` is used up). Returns (passes, why it ended): "single", "gained nothing" or "time"."""
    began, passes = clock(), []
    while True:
        p = do_pass()
        passes.append(p)
        if not keep_going:
            return passes, "single"
        if not p.get("gain"):
            return passes, "gained nothing"
        if seconds is not None and clock() - began >= seconds:
            return passes, "time"


def tool_entry(tool, rna_help: str, settings: list, group_label: str, keep_going: bool, every: bool, picks) -> dict:
    """One `tools.list` row from a `tidytools.Tool` and what Blender says about its operator."""
    return {
        "tool": tool.idname.partition(".")[2],
        "label": tool.label,
        "group": tool.group,
        "group_label": group_label,
        "help": rna_help,
        "developer_only": bool(tool.developer),
        "keep_going": keep_going,
        "every_spot": every,
        "picks": list(picks),
        "settings": settings,
    }


def _short(name: str) -> str:
    return name.partition(".")[2] if "." in name else name


# --------------------------------------------------------------------------- gates and lookups (need the add-on)


def _need(context):
    """The result on screen and nothing else working; raises NO_RESULT or BUSY."""
    from . import autofix_live, guided, tidylive, tidytools

    if not tidytools.result_ready():
        raise ToolError("NO_RESULT", "There is no result on screen", "Start Auto first (auto.start), then use the tools on its result")
    if guided.progress_info("AUTO") is not None or tidylive.busy():
        raise ToolError("BUSY", "Something is still working", "Wait for it (job.wait) or stop it (job.stop, autofix.stop)", {"auto_fix": bool(autofix_live.busy()), "tool_preview_open": tidylive.live["session"] is not None})
    return guided.state["res"]


def _object(name):
    """Every command may name the object; the tools work on the result on screen, so a different name is NOT_FOUND."""
    if not name:
        return
    from . import guided

    have = guided.state.get("source")
    if have and have != name:
        raise ToolError("NOT_FOUND", f"The result on screen is for {have!r}, not {name!r}", "Start Auto on that object first", {"object": have})


def _step(context) -> int:
    from . import history

    return history.current(context.scene)


def _op_class(idname: str):
    import bpy

    return getattr(bpy.types, "CURIOMESH_OT_" + _short(idname), None)


def _tool_for(name: str):
    """The `tidytools.Tool` for 'tool_source_relax' or 'curiomesh.tool_source_relax'; NOT_FOUND names the nearest."""
    from . import tidytools

    full = name if "." in name else "curiomesh." + name
    for t in tidytools.TOOLS:
        if t.idname == full and tidytools._op(t.idname) is not None:
            return t
    names = [_short(t.idname) for t in tidytools.TOOLS if tidytools._op(t.idname) is not None]
    raise ToolError("NOT_FOUND", f"no tool called {name!r}", "tools.list shows every tool", {"close": difflib.get_close_matches(str(name), names, n=3, cutoff=0.4), "tools": names})


def _props(idname: str):
    """Blender's description of an operator's properties (what its panel and F9 show), not of the class."""
    import bpy

    return getattr(bpy.ops.curiomesh, _short(idname)).get_rna_type()


def _schema_of(idname: str) -> list:
    return settings_schema(_props(idname).properties)


def _scope_now(context) -> str:
    from . import history

    ts = history.settings(context.scene)
    return ts.tool_scope if ts is not None else "WHOLE"


def _plan(context, name: str, given: dict | None, every: bool = False, keep_going: bool = False) -> SimpleNamespace:
    """What a run needs: the Trial class, its settings (checked), the scope, a pick function and what Again remembers."""
    from . import tidytools

    tool = _tool_for(name)
    cls = _op_class(tool.idname)
    schema = _schema_of(tool.idname)
    cleaned = check_settings(schema, given)
    if keep_going and not every and any(e["name"] == "keep_going" for e in schema):
        cleaned.setdefault("keep_going", True)  # the Relax tools' own Keep going: one pass runs on until it settles, one History step
    if every:
        if not tidytools.finds_spots(tool.idname):
            raise ToolError("BAD_ARGS", f"{tool.label} has no spots to find", "Use tools.run for it, or pick a tool with every_spot true in tools.list")
        trial = tidytools._every_class(tool.idname)
        base = tidytools.trial_of(tool.idname)
        settings = dict(base().defaults, **{k: v for k, v in cleaned.items() if k in base().defaults})
        settings.update(tool=tool.idname, keep_going=False)
        return SimpleNamespace(tool=tool, cls=cls, trial=trial, settings=settings, scope=_scope_now(context), pick=lambda: {}, idname=EVERY, stub=None)
    values = {p.identifier: p.default for p in _props(tool.idname).properties if p.identifier != "rna_type" and p.type in ("FLOAT", "INT", "BOOLEAN", "STRING", "ENUM")}
    if tool.idname == HARD_EDGES:
        values["degrees"] = context.scene.curiomesh_guided.hs_angle  # opens on the angle Apply would use now, as the button does
    values.update(cleaned)
    stub = SimpleNamespace(trial=cls.trial, **values)  # the operator's own _settings and _pick read these, as they read the operator
    return SimpleNamespace(tool=tool, cls=cls, trial=cls.trial, settings=cls._settings(stub), scope="WHOLE" if tool.idname == HARD_EDGES else _scope_now(context), pick=lambda: cls._pick(stub, context), idname=tool.idname, stub=stub)


def _info(s, before_faces: int) -> dict:
    """The readout card as numbers: each measured number with its words and whether it is better (1), worse (-1) or the same (0)."""
    from .triadq import trial as TT

    c = s.cand
    out = {"numbers": [], "faces": [before_faces, before_faces], "moved_most": None, "poles_gone": 0, "poles_new": 0, "held": 0, "note": "", "readout": ""}
    if c is None:
        return out
    for key, value in c.numbers.items():
        words, sign = TT.number_text(key, value, s.cap)
        out["numbers"].append({"key": key, "value": _py(value), "text": words, "better": sign})
    if c.faces is not None:
        out["faces"][1] = len(c.faces)
    if c.moved is not None and len(c.moved):
        out["moved_most"] = float(np.nanmax(c.moved))
    out["poles_gone"] = 0 if c.poles_gone is None else len(c.poles_gone)
    out["poles_new"] = 0 if c.poles_new is None else len(c.poles_new)
    out["held"] = len(c.held)
    out["note"] = c.note
    from . import tidylive

    out["readout"] = tidylive.numbers_text(c, s.trial, s.cap)
    return out


def _pick_details(s) -> dict:
    kind = s.waiting()
    ids, labels = s.targets(kind) if s.base is not None else (None, None)
    return {"pick": kind, "ask": s.ask(), "targets": None if ids is None else [{"id": int(i), "label": str(l)} for i, l in zip(np.asarray(ids).tolist(), labels)]}


def _refusal(s, allow_worse: bool):
    """(code, words, hint, details) why this finished run is not kept, or None when it can be."""
    from . import tidylive
    from .triadq import autofix

    why = tidylive.why_not_kept(s)
    if why:
        if s.error:
            return ("NO_RESULT" if s.res is None else "NOT_READY", why, "Use tools.scope to choose what the tool may change", {})
        if s.waiting():
            return ("BAD_ARGS", why, f"Give the setting {s.waiting()!r} (a number); targets lists what can be picked", _pick_details(s))
        if s.cand is not None and s.cand.co is None and s.cand.rows:
            return ("BAD_ARGS", why, "Pass settings.row: one of the keys in rows", {"rows": [{"key": r["key"], "words": r["words"]} for r in s.cand.rows]})
        return ("REFUSED", why, "Nothing changed", {})
    reason = s.proposal.reason if s.proposal is not None else ""
    if reason in ("bad faces", "worse", "look") and not allow_worse:
        hint = "Raise Most Detail Loss to allow it, or pass allow_worse" if reason == "look" else "Pass allow_worse to keep it anyway"
        return ("REFUSED", f"Not kept: {autofix.WHY[reason]}", hint, {"reason": reason})
    return None


def _one_pass(context, plan, keep: bool, keep_going: bool, allow_worse: bool, before_faces: int) -> dict:
    """One run of the plan on the result as it is now. Kept (one History step, one Blender undo step) unless `keep` is off,
    it is refused, or Keep going finds it gains nothing. Returns the pass as a dict; `code` is set when it was refused."""
    from . import guided, tidylive

    s = tidylive.work_out(context, plan.trial(), dict(plan.settings), plan.scope, plan.pick())
    p = dict(_info(s, before_faces), name=s.trial.name, kept=False, gain=False, code="", why="", hint="", details={}, ran=True)
    refused = _refusal(s, allow_worse)
    if refused:
        p.update(code=refused[0], why=refused[1], hint=refused[2], details=refused[3])
        return p
    p["gain"] = gained(s.cand)
    if not keep or (keep_going and not p["gain"]):
        p["why"] = "" if not keep else "gained nothing"
        return p
    if not tidylive.commit(context, s.cand, f"Tidy: {s.trial.name}", p["readout"], push=True):
        p.update(code="REFUSED", why="The result changed while the tool was working", hint="Run it again")
        return p
    tidylive._remember(context, plan.idname, s.trial.name, p["readout"], {k: v for k, v in s.settings.items() if k not in s.trial.picks})
    p.update(kept=True, step=_step(context), faces=[before_faces, len(guided.state["res"]["faces"])])
    return p


def _passes_reply(context, plan, passes, why_ended: str, before_faces: int, seconds: float) -> dict:
    from . import guided

    kept = [p for p in passes if p["kept"]]
    first = passes[0]
    if first["code"]:  # nothing was kept: say why, with the details
        raise ToolError(first["code"], first["why"], first["hint"], first["details"])
    out = {
        "tool": _short(plan.idname) if plan.idname != EVERY else "every:" + _short(plan.settings["tool"]),
        "kept": len(kept),
        "ended": why_ended,
        "faces": [before_faces, len(guided.state["res"]["faces"])],
        "passes": [{k: p[k] for k in ("kept", "gain", "readout", "numbers", "faces", "moved_most", "poles_gone", "poles_new", "note", "why", "code") if k in p} for p in passes],
    }
    if why_ended == "time":
        out["hint"] = f"Stopped after {seconds:.0f} s so Blender is not held up. tools.again with keep_going goes on from here"
    return out


def _run(context, plan, keep_going: bool, allow_worse: bool, seconds, keep: bool = True) -> dict:
    from . import guided

    before = len(guided.state["res"]["faces"])
    limit = SECONDS if seconds is None else float(seconds)
    t0 = time.monotonic()
    passes, why = run_passes(lambda: _one_pass(context, plan, keep, keep_going, allow_worse, len(guided.state["res"]["faces"])), keep_going and keep, limit)
    return _passes_reply(context, plan, passes, why, before, time.monotonic() - t0 if why == "time" else limit)


# --------------------------------------------------------------------------- tools.*

_SETTINGS = {"type": "object", "description": "Tool settings by name; tools.list gives each tool's names and ranges"}
_OBJECT = {"type": "string"}


def tools_list(context, object=None):
    from . import history, tidytools

    _object(object)
    groups = [{"key": k, "label": label, "help": tip} for k, label, _i, _s, tip in tidytools.GROUPS]
    labels = {g["key"]: g["label"] for g in groups}
    rows = []
    for t in tidytools.TOOLS:
        cls = _op_class(t.idname)
        if cls is None or tidytools._op(t.idname) is None:
            continue
        rna = _props(t.idname)
        schema = settings_schema(rna.properties)
        rows.append(tool_entry(t, rna.description or cls.bl_description, schema, labels[t.group], any(e["name"] == "keep_going" for e in schema), tidytools.finds_spots(t.idname), getattr(cls.trial, "picks", ())))
    ts = history.settings(context.scene)
    return {"tools": rows, "groups": groups, "scope": ts.tool_scope if ts is not None else "WHOLE", "scopes": list(SCOPES), "last": ts.last_tool if ts is not None else ""}


def tools_scope(context, scope, faces=None, parts=None, object=None):
    import bpy

    from . import history, segments, tidylive

    _object(object)
    res = _need(context)
    if scope not in SCOPES:
        raise ToolError("BAD_ARGS", f"scope must be one of {list(SCOPES)}", details={"argument": "scope", "allowed": list(SCOPES)})
    count = len(res["faces"])
    if scope == "SELECTION":
        ids = sorted({int(i) for i in (faces or [])})
        if not ids:
            raise ToolError("BAD_ARGS", "SELECTION needs faces: the face numbers of the result", "Pass faces: [0, 1, 2]")
        if ids[0] < 0 or ids[-1] >= count:
            raise ToolError("BAD_ARGS", f"face numbers must be 0 to {count - 1}", details={"argument": "faces", "minimum": 0, "maximum": count - 1})
        obj = tidylive._preview()
        if obj is None or len(obj.data.polygons) != count:
            raise ToolError("NOT_READY", "The result on screen has no mesh to select on")
        mask = np.zeros(count, dtype=bool)
        mask[ids] = True
        if obj.mode == "EDIT":
            import bmesh

            bm = bmesh.from_edit_mesh(obj.data)
            bm.faces.ensure_lookup_table()
            for f in bm.faces:
                f.select_set(False)
            for i in ids:
                bm.faces[i].select_set(True)
            bmesh.update_edit_mesh(obj.data)
        else:
            obj.data.polygons.foreach_set("select", mask)
            obj.data.update()
    elif scope == "PART":
        if not segments.ready():
            raise ToolError("NOT_READY", "This result has no parts table yet", "Use Whole Result, or start Auto first")
        have = len(segments.seg["parts"])
        ids = sorted({int(i) for i in (parts or [])})
        if not ids or ids[0] < 0 or ids[-1] >= have:
            raise ToolError("NOT_FOUND", f"PART needs parts: numbers 0 to {have - 1}", "parts.list shows them", {"parts": have})
        segments.select(ids)
    ts = history.settings(context.scene)
    ts.tool_scope = scope
    mask, _why = tidylive.scope_faces(context, res, scope)
    return {"scope": scope, "faces_in_scope": int(mask.sum()) if mask is not None else 0, "faces": count}


def tools_preview(context, tool, settings=None, every=False, object=None):
    """Work out the change without keeping it: the readout card's numbers. Nothing on the result moves."""
    from . import guided, tidylive

    _object(object)
    _need(context)
    plan = _plan(context, tool, settings, every)
    n = len(guided.state["res"]["faces"])
    s = tidylive.work_out(context, plan.trial(), dict(plan.settings), plan.scope, plan.pick())
    out = dict(_info(s, n), tool=plan.tool.idname.partition(".")[2], name=s.trial.name, scope=plan.scope, step=_step(context), kept=False)
    out["rows"] = [] if s.cand is None else [{"key": r["key"], "words": r["words"], "cost": _py(r.get("cost"))} for r in s.cand.rows]
    refused = _refusal(s, False)
    out["would_keep"] = refused is None and s.cand is not None and s.cand.co is not None
    out["problem"] = None if refused is None else {"code": refused[0], "message": refused[1], "details": refused[3]}
    out["gain"] = gained(s.cand)
    out["held_back"] = s.status if s.status.startswith("held") else ""
    if plan.tool.idname == HARD_EDGES:
        out["would_keep"], out["problem"] = True, None
    return out


def tools_run(context, tool, settings=None, keep_going=False, allow_worse=False, seconds=None, object=None):
    """Run a tool and keep it: one History step per kept pass. With keep_going it repeats until a pass gains nothing."""
    _object(object)
    _need(context)
    plan = _plan(context, tool, settings, keep_going=bool(keep_going))
    if plan.tool.idname == HARD_EDGES:
        return _hard_edges(context, plan)
    return _run(context, plan, bool(keep_going), bool(allow_worse), seconds)


def _hard_edges(context, plan) -> dict:
    """Hard Edges moves nothing: it keeps the angle for Apply (the operator's own `_set`)."""
    from . import tidylive

    degrees = float(plan.stub.degrees)
    plan.cls._set(plan.stub, context, degrees)
    tidylive._remember(context, HARD_EDGES, "Hard Edges", f"angle {degrees:.0f} degrees", {})
    return {"tool": "tool_hard_edges", "kept": 0, "ended": "single", "passes": [], "angle": degrees, "note": f"Apply will mark edges over {degrees:.0f} degrees as hard. The result's corners did not move"}


def tools_every(context, tool, settings=None, keep_going=False, allow_worse=False, seconds=None, object=None):
    """Fix every spot like this: the tool on every spot it finds in Work On; spots that get no better are put back."""
    _object(object)
    _need(context)
    return _run(context, _plan(context, tool, settings, every=True), bool(keep_going), bool(allow_worse), seconds)


def tools_again(context, keep_going=False, allow_worse=False, seconds=None, object=None):
    """Run the last tool again on the current result, with the settings it was kept with."""
    from . import history

    _object(object)
    _need(context)
    ts = history.settings(context.scene)
    if ts is None or not ts.last_tool:
        raise ToolError("NOT_FOUND", "No tool has been kept yet", "Use tools.run first")
    try:
        kept = json.loads(ts.last_settings or "{}")
    except ValueError:
        kept = {}
    kept = kept if isinstance(kept, dict) else {}
    if ts.last_tool == EVERY:
        tool = kept.pop("tool", "")
        known = {e["name"] for e in _schema_of(tool)} if tool else set()
        return _run(context, _plan(context, tool, {k: v for k, v in kept.items() if k in known}, every=True), bool(keep_going), bool(allow_worse), seconds)
    if ts.last_tool.startswith("curiomesh.strip_fix_all"):
        return loops_fix_all(context, amount=kept.get("amount"), keep_going=keep_going, seconds=seconds)
    known = {e["name"] for e in _schema_of(ts.last_tool)}
    return _run(context, _plan(context, ts.last_tool, {k: v for k, v in kept.items() if k in known}, keep_going=bool(keep_going)), bool(keep_going), bool(allow_worse), seconds)


# --------------------------------------------------------------------------- loops.*


def _strip_ok(res, strip: int):
    from . import faceloops

    st = faceloops.strips(res)
    if not 0 <= int(strip) < 2 * len(st.quads):
        raise ToolError("NOT_FOUND", f"no strip {strip}", "loops.strips lists strips by number", {"strips": 2 * len(st.quads)})
    return st


def loops_strips(context, kind="ALL", limit=50, object=None):
    """The strips with their number, length, wiggle and width variation, worst first."""
    from . import faceloops
    from .triadq import strips as S
    from .triadq import stripfix as SF

    _object(object)
    res = _need(context)
    st = faceloops.strips(res)
    rows = []
    if len(st.quads):
        labels, first = np.unique(st.labels, return_index=True)
        for element in first.tolist():
            walk = faceloops.walk(res, element)
            n = S.stats(res["co"], walk)
            score = max(n["turn"] / SF.WIGGLY, n["width"] / SF.UNEVEN)
            if kind == "WIGGLY" and n["turn"] <= SF.WIGGLY or kind == "UNEVEN" and n["width"] <= SF.UNEVEN or kind == "CLOSED" and not n["closed"] or kind == "OPEN" and n["closed"]:
                continue
            rows.append({"strip": int(element), "faces": n["faces"], "closed": n["closed"], "wiggle_degrees_per_quad": round(n["turn"], 3), "width_variation_percent": round(100.0 * n["width"], 2), "score": round(score, 3)})
    rows.sort(key=lambda r: -r["score"])
    return {"strips": rows[: int(limit)], "total": len(rows), "wiggly_over": SF.WIGGLY, "uneven_over_percent": 100.0 * SF.UNEVEN}


def loops_poles(context, limit=50, object=None):
    """The poles (corners where other than four edges meet) of the result, for loops.move_pole."""
    from .triadq.patterns import Topology

    _object(object)
    res = _need(context)
    t = Topology(np.asarray(res["co"], dtype=np.float64), res["faces"])
    ids = np.flatnonzero(t.poles)
    return {"poles": [{"vertex": int(v), "edges": int(t.valence[v]), "border": bool(t.border[v]), "at": _py(t.points[v])} for v in ids[: int(limit)].tolist()], "total": int(len(ids))}


def _done(context, name: str, error: str, before: tuple) -> dict:
    """Close a Face Loops edit: an error in words means nothing changed; else one Blender undo step."""
    from . import faceloops, guided, history

    if error:
        raise ToolError("REFUSED", error, "Nothing changed", {})
    history._push(name)
    return {"done": name, "note": faceloops.fl["note"], "faces": [before[0], len(guided.state["res"]["faces"])], "step": _step(context)}


def loops_edit(context, strip, how, amount=None, object=None):
    """One strip tool on one strip: the same call the buttons make."""
    from . import faceloops, guided

    _object(object)
    res = _need(context)
    kinds = {"STRAIGHTEN": "straighten", "EVEN": "even", "FIX": "fix", "SLIDE": "slide", "ADD": "add", "REMOVE": "remove"}
    if how not in kinds:
        raise ToolError("BAD_ARGS", f"how must be one of {list(kinds)}", details={"argument": "how", "allowed": list(kinds)})
    _strip_ok(res, strip)
    if how == "SLIDE":
        if amount is None or not -0.95 <= amount <= 0.95:
            raise ToolError("BAD_ARGS", "SLIDE needs amount between -0.95 and 0.95", "The share of the next quad's width; minus goes the other way", {"argument": "amount", "minimum": -0.95, "maximum": 0.95})
    elif amount is not None and not 0.0 <= amount <= 1.0:
        raise ToolError("BAD_ARGS", "amount must be between 0 and 1", details={"argument": "amount", "minimum": 0.0, "maximum": 1.0})
    faceloops.fl["sel"] = (res, int(strip))
    n = len(res["faces"])
    out = _done(context, faceloops.NAMES[kinds[how]], faceloops.edit(context, kinds[how], 1.0 if amount is None else float(amount)), (n,))
    sel = faceloops.fl["sel"]
    out["strip"] = int(sel[1]) if sel is not None and sel[1] is not None and sel[0] is guided.state["res"] else None
    return out


def loops_fix_all(context, amount=None, keep_going=False, allow_worse=False, seconds=None, object=None):
    """Fix every wiggly or uneven strip in Work On; it keeps only the strips that got better."""
    from . import faceloops

    _object(object)
    _need(context)
    trial = faceloops.CURIOMESH_OT_strip_fix_all.trial
    settings = dict(trial().defaults, keep_going=False)
    if amount is not None:
        settings.update(check_settings([{"name": "amount", "type": "float", "min": 0.0, "max": 1.0}], {"amount": amount}))
    plan = SimpleNamespace(tool=SimpleNamespace(idname="curiomesh.strip_fix_all"), trial=trial, settings=settings, scope=_scope_now(context), pick=lambda: {}, idname="curiomesh.strip_fix_all", stub=None)
    return _run(context, plan, bool(keep_going), bool(allow_worse), seconds)


def _vec(value, what: str, optional=False):
    if value is None and optional:
        return None
    try:
        v = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        v = np.zeros(0)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ToolError("BAD_ARGS", f"{what} must be three numbers [x, y, z]", details={"argument": what})
    return v


def loops_ring(context, centre, right, normal=None, width=None, height=None, band=None, rings=0, object=None):
    """Ring Round This as numbers: rings of quads round the opening at `centre` (a point on the result). width and height
    are the oval's full size; left out, they are read off your original mesh."""
    from . import faceloops, guided
    from .triadq import rings as RG

    _object(object)
    res = _need(context)
    n = len(res["faces"])
    for what, v in (("width", width), ("height", height)):
        if v is not None and not (isinstance(v, (int, float)) and v > 0):
            raise ToolError("BAD_ARGS", f"{what} must be a number above 0", details={"argument": what, "minimum": 0})
    error = faceloops.ring(context, _vec(centre, "centre"), _vec(normal, "normal", True), _vec(right, "right"), None if width is None else width / 2.0, None if height is None else height / 2.0, RG.BAND if band is None else float(band), int(rings))
    return _done(context, "Ring Round This", error, (n,))


def loops_redraw(context, faces=None, part=None, across=None, object=None):
    """Redraw as Rows on the given faces (or all faces of a part): the same call the lasso makes."""
    from . import faceloops, segments

    _object(object)
    res = _need(context)
    n = len(res["faces"])
    if (faces is None) == (part is None):
        raise ToolError("BAD_ARGS", "Give either faces or part", "faces: [face numbers]; part: a part number from parts.list")
    if faces is not None:
        ids = np.asarray(sorted({int(i) for i in faces}), dtype=np.int64)
        if not len(ids) or ids[0] < 0 or ids[-1] >= n:
            raise ToolError("BAD_ARGS", f"face numbers must be 0 to {n - 1}", details={"argument": "faces", "minimum": 0, "maximum": n - 1})
    else:
        piece = segments.seg.get("piece")
        if piece is None or len(piece) != n:
            raise ToolError("NOT_READY", "This result has no parts table", "Pass faces instead")
        ids = np.flatnonzero(np.asarray(piece) == int(part))
        if not len(ids):
            raise ToolError("NOT_FOUND", f"no part {part}", "parts.list shows them")
    return _done(context, "Redraw in Rows", faceloops.redraw_rows(context, ids, _vec(across, "across", True)), (n,))


def loops_move_pole(context, vertex, to, object=None):
    """Put a pole at a point on the result; the quads round it follow. Refused when nothing could move (a held corner)."""
    from . import faceloops, guided, history

    _object(object)
    res = _need(context)
    n = len(res["faces"])
    if not isinstance(vertex, int) or isinstance(vertex, bool) or not 0 <= vertex < len(res["co"]):
        raise ToolError("BAD_ARGS", f"vertex must be a corner number 0 to {len(res['co']) - 1}", "loops.poles lists the poles", {"argument": "vertex"})
    before = np.asarray(res["co"], dtype=np.float64).copy()
    faceloops.move_pole(context, vertex, _vec(to, "to"))
    if np.array_equal(before, np.asarray(guided.state["res"]["co"], dtype=np.float64)):
        raise ToolError("REFUSED", "That pole is held (a border, a hard edge or the mirror plane) and did not move", "Nothing changed")
    history._push("Move Pole")
    return {"done": "Move Pole", "faces": [n, len(guided.state["res"]["faces"])], "step": _step(context)}


# --------------------------------------------------------------------------- flow.*

FLOW_KINDS = ("FLOW", "RING", "EDGE")


def _settle(seconds: float = 600.0) -> bool:
    """Headless Blender has no timers: step the Flow Editor's worker until it is done. True when it finished."""
    import bpy

    from . import flowedit

    if not bpy.app.background:
        return flowedit.fe["job"] is None
    ends = time.monotonic() + seconds
    while flowedit.fe["job"] is not None and time.monotonic() < ends:
        flowedit._tick()
        time.sleep(0.01)
    flowedit._tick()
    return flowedit.fe["job"] is None


def _strokes(context) -> list:
    from . import flowedit

    return [g for g in flowedit._curves(context.scene.curiomesh_guided)]


def _stroke_row(i: int, g) -> dict:
    from . import flowedit
    from . import flowstroke as F

    obj = g.obj
    pts = np.asarray(obj.get(flowedit.RAW_KEY, []), dtype=np.float64).reshape(-1, 3)
    rec = F.record(str(obj[flowedit.KIND_KEY]), pts) if len(pts) >= 2 else None
    return {"index": i, "kind": str(obj[flowedit.KIND_KEY]), "object": obj.name, "points": int(len(pts)), "closed": None if rec is None else rec["closed"]}


def flow_add(context, kind, points, object=None):
    """A Flow Editor stroke from 3D points in the source's space (they are laid on the source mesh). RING closes itself."""
    from . import flowedit, guided
    from . import flowstroke as F

    _object(object)
    if kind not in FLOW_KINDS:
        raise ToolError("BAD_ARGS", f"kind must be one of {list(FLOW_KINDS)}", details={"argument": "kind", "allowed": list(FLOW_KINDS)})
    pts = np.asarray(points, dtype=np.float64) if isinstance(points, (list, tuple)) and points else np.zeros(0)
    least = 3 if kind == "RING" else 2
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < least or not np.isfinite(pts).all():
        raise ToolError("BAD_ARGS", f"points must be a list of at least {least} points [x, y, z]", details={"argument": "points", "minimum_points": least})
    source = guided.state_source()
    if source is None:
        raise ToolError("NO_OBJECT", "There is no source mesh to lay the stroke on", "Start Auto first")
    F.record(kind, pts)  # the same plain-data form the solver reads; it rejects nothing, so a bad stroke fails above
    obj = flowedit.add_stroke(context, source, kind, F.tidy(kind, pts), pts)
    gs = context.scene.curiomesh_guided
    flowedit.request(context)
    from . import history

    history._push("Flow Editor stroke")
    done = _settle() if gs.flow_live else True
    items = _strokes(context)
    return {"index": len(items) - 1, "object": obj.name, "kind": kind, "strokes": len(items), "live": bool(gs.flow_live), "running": not done, "hint": "" if gs.flow_live else "Flow Live is off: the strokes steer the next Start. Turn Live on to redo the result now"}


def flow_list(context, object=None):
    _object(object)
    return {"strokes": [_stroke_row(i, g) for i, g in enumerate(_strokes(context))]}


def flow_remove(context, index, object=None):
    from . import flowedit, history

    _object(object)
    items = _strokes(context)
    if not 0 <= int(index) < len(items):
        raise ToolError("NOT_FOUND", f"no stroke {index}", "flow.list shows them", {"strokes": len(items)})
    gs = context.scene.curiomesh_guided
    flowedit.remove_stroke(gs, items[int(index)].obj)
    flowedit.request(context)
    history._push("Remove Flow Stroke")
    done = _settle() if gs.flow_live else True
    return {"strokes": len(_strokes(context)), "running": not done}


def flow_clear(context, object=None):
    from . import flowedit, history

    _object(object)
    gs = context.scene.curiomesh_guided
    for g in list(flowedit._curves(gs)):
        flowedit.remove_stroke(gs, g.obj)
    flowedit.request(context)
    history._push("Erase All Strokes")
    done = _settle() if gs.flow_live else True
    return {"strokes": 0, "running": not done}


# --------------------------------------------------------------------------- autofix.*


def autofix_groups(context, object=None):
    from . import autofix_live
    from .triadq import autofix

    _object(object)
    pg = context.scene.curiomesh_autofix
    props = pg.bl_rna.properties  # a property group's own class: this one is the real list
    out = []
    for key in autofix.GROUPS:
        kinds = [many for cls, only in autofix.plan({key}) for k, _o, many in cls.kinds if only is None or k in only]
        out.append({"group": key, "label": props[key].name, "help": autofix_live.GROUP_TIPS.get(key) or props[key].description, "on": bool(getattr(pg, key)), "fixes": kinds})
    return {"groups": out, "auto_fix": bool(context.scene.curiomesh_guided.auto_fix)}


def autofix_set(context, ticks, object=None):
    from .triadq import autofix

    _object(object)
    if not isinstance(ticks, dict) or not ticks:
        raise ToolError("BAD_ARGS", "ticks must be an object of group: true or false", details={"allowed": list(autofix.GROUPS)})
    for key, value in ticks.items():
        if key not in autofix.GROUPS:
            raise ToolError("BAD_ARGS", f"unknown group {key!r}", "autofix.groups lists them", {"argument": key, "allowed": list(autofix.GROUPS), "close": difflib.get_close_matches(str(key), autofix.GROUPS, n=3, cutoff=0.4)})
        if not isinstance(value, bool):
            raise ToolError("BAD_ARGS", f"{key} must be true or false", details={"argument": key, "type": "boolean"})
    pg = context.scene.curiomesh_autofix
    for key, value in ticks.items():
        setattr(pg, key, value)
    return {"groups": {g: bool(getattr(pg, g)) for g in autofix.GROUPS}}


def _pump_headless(context, seconds: float) -> None:
    """A background Blender has no timers: run the job here, to its end or `seconds`."""
    import bpy

    from . import autofix_live

    if bpy.app.background and autofix_live.busy():
        autofix_live.finish_now(context, seconds)


def _reply(context) -> dict:
    from . import autofix_live as AL
    from . import guided

    rep = AL.fx["report"]
    out = {"running": bool(AL.busy()), "faces": len(guided.state["res"]["faces"]) if guided.state["res"] is not None else 0}
    if rep is None:
        return dict(out, how="", words="", passes=0, kept=0)
    kept = [e for e in rep["passes"] if e["kept"]]
    out.update(how=rep.get("how", ""), words=AL.head_words(rep) if rep["passes"] else rep.get("words", ""), stop=rep.get("stop", ""), passes=len(rep["passes"]), kept=len(kept), seconds=round(float(rep.get("seconds", 0.0)), 2), can_keep_going=rep.get("how") == "limit" and not AL.busy(), can_undo=rep.get("step0") is not None and not AL.busy())
    if rep.get("how") == "limit":
        out["hint"] = "The time limit ended Auto Fix while it was still improving. autofix.keep_going carries on"
    return out


def autofix_run(context, here=True, time_limit=None, seconds=3600.0, object=None):
    """Start Auto Fix on the result on screen. In a live window it returns at once with running true (poll job.status);
    a background Blender runs it to its end here. Every kept pass is one History step."""
    from . import autofix_live as AL
    from . import guided

    _object(object)
    if guided.state["auto_base"] is None or guided.state["res"] is None:
        raise ToolError("NO_RESULT", "There is no result on screen", "Start Auto first (auto.start)")
    if AL.busy():
        raise ToolError("BUSY", "Auto Fix is already working", "autofix.stop keeps what it fixed so far")
    if time_limit is not None and not (isinstance(time_limit, (int, float)) and time_limit > 0):
        raise ToolError("BAD_ARGS", "time_limit must be a number of seconds above 0", details={"argument": "time_limit", "minimum": 0})
    if not AL.start(context, force=True, limit=None if time_limit is None else float(time_limit)):
        raise ToolError("NOT_READY", "Auto Fix could not start on this result")
    _pump_headless(context, float(seconds))
    return _reply(context)


def autofix_keep_going(context, seconds=3600.0, object=None):
    from . import autofix_live as AL

    _object(object)
    if AL.busy():
        raise ToolError("BUSY", "Auto Fix is still working")
    if not AL.keep_going(context):
        raise ToolError("NOT_READY", "Auto Fix did not stop at its time limit, so there is nothing to carry on", "autofix.run starts it again")
    _pump_headless(context, float(seconds))
    return _reply(context)


def autofix_stop(context, object=None):
    from . import autofix_live as AL

    _object(object)
    stopped = AL.stop(context)
    return dict(_reply(context), stopped=bool(stopped))


def autofix_undo(context, object=None):
    """Back to the result as it was before Auto Fix, in one History step. Ctrl+Z brings the fixes back."""
    from . import autofix_live as AL

    _object(object)
    if AL.busy():
        raise ToolError("BUSY", "Auto Fix is still working", "autofix.stop first")
    if not AL.undo_all(context):
        raise ToolError("NOT_READY", "Auto Fix has not kept anything to take back", "autofix.run first")
    return dict(_reply(context), undone=True, step=_step(context))


def autofix_findings(context, object=None):
    """Each finding with its place, kind and what was done; then each pass with its numbers."""
    from . import autofix_live as AL

    _object(object)
    rep = AL.fx["report"]
    if rep is None:
        raise ToolError("NOT_FOUND", "Auto Fix has not run on this result", "autofix.run first")
    findings = [{"tool": f["tool"], "text": f["text"], "kept": f["kept"], "at": _py(f["at"])} for f in AL.lines(rep)]
    passes = []
    for e in rep["passes"]:
        p = e["proposal"]
        row = {"tool": e["name"], "round": e["round"], "kept": e["kept"], "reason": "" if p is None else p.reason}
        if p is not None:
            row.update(fixed=len(p.fixed), left=len(p.left), before=_py(vars(p.before)), after=_py(vars(p.after)))
        passes.append(row)
    return {"findings": findings, "passes": passes, "words": AL.head_words(rep) if rep["passes"] else rep.get("words", "")}


# --------------------------------------------------------------------------- the command table


def _cmd(fn, help, props=None, required=(), edits=False):
    props = dict(props or {}, object=_OBJECT)
    return {"fn": fn, "help": help, "edits": edits, "args": {"type": "object", "properties": props, "required": list(required)}}


_RUN = {"settings": _SETTINGS, "keep_going": {"type": "boolean"}, "allow_worse": {"type": "boolean"}, "seconds": {"type": "number", "minimum": 0}}
_POINT = {"type": "array", "items": {"type": "number"}}
_FACES = {"type": "array", "items": {"type": "integer"}}

COMMANDS = {
    "tools.list": _cmd(tools_list, "Every Tidy Tool with its group, help, settings (type, default, soft and hard range, choices), whether it has Keep going, finds spots (every_spot) or needs a pick. Developer-only tools are marked."),
    "tools.scope": _cmd(tools_scope, "Choose what the tools may change: SELECTION (with faces), PART (with parts) or WHOLE.", {"scope": {"type": "string", "enum": list(SCOPES)}, "faces": _FACES, "parts": _FACES}, ["scope"], True),
    "tools.preview": _cmd(tools_preview, "Work out a tool's change without keeping it. Returns the readout card's numbers (better or worse, look change, poles, points moved). Nothing changes.", {"tool": {"type": "string"}, "settings": _SETTINGS, "every": {"type": "boolean"}}, ["tool"]),
    "tools.run": _cmd(tools_run, "Run a tool and keep it: one History step per kept pass. keep_going repeats until a pass gains nothing (or seconds run out, and it says so). A change that makes things worse is REFUSED and nothing changes.", dict(_RUN, tool={"type": "string"}), ["tool"], True),
    "tools.again": _cmd(tools_again, "Run the last kept tool again on the current result with its last settings.", {k: v for k, v in _RUN.items() if k != "settings"}, edits=True),
    "tools.every": _cmd(tools_every, "Fix every spot like this: the tool on every spot it finds in Work On; spots that get no better are put back.", dict(_RUN, tool={"type": "string"}), ["tool"], True),
    "loops.strips": _cmd(loops_strips, "The strips of quads with number, length, wiggle and width variation, worst first.", {"kind": {"type": "string", "enum": ["ALL", "WIGGLY", "UNEVEN", "CLOSED", "OPEN"]}, "limit": {"type": "integer", "minimum": 1}}),
    "loops.poles": _cmd(loops_poles, "The poles of the result (corner number, edges meeting, position), for loops.move_pole.", {"limit": {"type": "integer", "minimum": 1}}),
    "loops.edit": _cmd(loops_edit, "A strip tool on one strip: STRAIGHTEN, EVEN, FIX, SLIDE, ADD or REMOVE. amount 0 to 1 (SLIDE: -0.95 to 0.95).", {"strip": {"type": "integer", "minimum": 0}, "how": {"type": "string", "enum": ["STRAIGHTEN", "EVEN", "FIX", "SLIDE", "ADD", "REMOVE"]}, "amount": {"type": "number"}}, ["strip", "how"], True),
    "loops.fix_all": _cmd(loops_fix_all, "Fix every wiggly or uneven strip in Work On. Keeps only the strips that got better.", {"amount": {"type": "number", "minimum": 0, "maximum": 1}, **{k: v for k, v in _RUN.items() if k != "settings"}}, edits=True),
    "loops.ring": _cmd(loops_ring, "Ring Round This as numbers: rings of quads round an opening at centre (a point on the result). width and height are the oval's full size.", {"centre": _POINT, "right": _POINT, "normal": _POINT, "width": {"type": "number"}, "height": {"type": "number"}, "band": {"type": "number", "minimum": 0}, "rings": {"type": "integer", "minimum": 0}}, ["centre", "right"], True),
    "loops.redraw": _cmd(loops_redraw, "Redraw as Rows on faces (or on a whole part). across is the direction the rows run in, in the result's space.", {"faces": _FACES, "part": {"type": "integer", "minimum": 0}, "across": _POINT}, edits=True),
    "loops.move_pole": _cmd(loops_move_pole, "Put a pole at a point on the result; the quads round it follow.", {"vertex": {"type": "integer", "minimum": 0}, "to": _POINT}, ["vertex", "to"], True),
    "flow.add": _cmd(flow_add, "Add a Flow Editor stroke (FLOW, RING or EDGE) from 3D points in the source's space.", {"kind": {"type": "string", "enum": list(FLOW_KINDS)}, "points": {"type": "array", "items": _POINT}}, ["kind", "points"], True),
    "flow.list": _cmd(flow_list, "The Flow Editor strokes."),
    "flow.remove": _cmd(flow_remove, "Remove one Flow Editor stroke by its index in flow.list.", {"index": {"type": "integer", "minimum": 0}}, ["index"], True),
    "flow.clear": _cmd(flow_clear, "Erase every Flow Editor stroke; the result goes back to how it was before.", edits=True),
    "autofix.groups": _cmd(autofix_groups, "The seven Auto Fix groups, whether each is ticked, and what each fixes."),
    "autofix.set": _cmd(autofix_set, "Tick or untick Auto Fix groups: ticks is {group: true or false}.", {"ticks": {"type": "object"}}, ["ticks"], True),
    "autofix.run": _cmd(autofix_run, "Start Auto Fix on the result on screen. Live: returns at once with running true. Background Blender: runs to the end.", {"here": {"type": "boolean"}, "time_limit": {"type": "number", "minimum": 0}, "seconds": {"type": "number", "minimum": 0}}, edits=True),
    "autofix.keep_going": _cmd(autofix_keep_going, "Carry on after Auto Fix stopped at its time limit.", {"seconds": {"type": "number", "minimum": 0}}, edits=True),
    "autofix.stop": _cmd(autofix_stop, "Stop Auto Fix now and keep what it fixed so far.", edits=True),
    "autofix.undo": _cmd(autofix_undo, "Put the result back as it was before Auto Fix, in one History step.", edits=True),
    "autofix.findings": _cmd(autofix_findings, "Each finding with its place and kind, and what Auto Fix did with it, then every pass with its numbers."),
}
