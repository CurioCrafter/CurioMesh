# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""One command layer for agents: a name plus JSON arguments in, one JSON reply out.

`run(name, args, context)` is the only entry point; the mailbox, `cm_do.py` and the MCP server all call it. Handler modules
(HANDLER_MODULES) each expose COMMANDS = {name: {"fn": f(context, **args), "args": json-schema, "help": str, "edits": bool}}
and raise any exception with an `agent_code` attribute (plus optional `hint`, `details`) to give a coded error. They need
nothing from here.

No bpy at import, so the tests run outside Blender.
"""

from __future__ import annotations

import difflib
import importlib
import time
import traceback

HANDLER_MODULES = ("agentops_run", "agentops_tools", "agentops_parts", "parts_export", "parts_import")

CODES = {
    "BAD_ARGS": "Missing or wrong argument. details gives the allowed values or range.",
    "UNKNOWN_COMMAND": "No such command. details.close names the nearest ones.",
    "NO_OBJECT": "No mesh to work on.",
    "NO_RESULT": "Needs a result on screen (run auto.start).",
    "BUSY": "Something is running. details holds job.status.",
    "NOT_READY": "The result does not allow this (mirror, Even Rows, imported job...).",
    "NOT_FOUND": "Part, spot, strip, tool or History step not there.",
    "REFUSED": "The tool would make things worse. Nothing changed.",
    "FILE": "Can't read or write a file; bad or damaged file.",
    "VERSION": "Part file from a newer format.",
    "MISMATCH": "Part file made from a different result.",
    "BORDER_MOVED": "A returned part moved a border point.",
    "HOLE": "A returned part would open new holes.",
    "SHAPE_LOST": "A returned part no longer follows the source shape.",
    "VIEW_CHANGED": "A returned part changes the look more than Most Detail Loss allows.",
    "NEEDS_WINDOW": "Only works in a live window. The hint gives the headless way.",
    "INTERNAL": "A bug. details.trace holds the last lines of the traceback.",
}

_PY = {"string": str, "object": dict, "array": (list, tuple)}


class AgentError(Exception):
    """Convenience for code in this package; handlers may raise any exception with an `agent_code` attribute."""

    def __init__(self, code, message, hint="", details=None):
        super().__init__(message)
        self.agent_code, self.hint, self.details = code, hint, details or {}


def _err(code, message, hint="", details=None):
    return {"code": code, "message": str(message), "hint": hint, "details": details or {}}


def _is(value, kind):
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    return isinstance(value, _PY.get(kind, object))


def check_args(schema, args):
    """None when `args` fit the JSON-schema subset (type, required, properties, enum, minimum, maximum, items), else an error dict."""
    schema = schema or {}
    if not isinstance(args, dict):
        return _err("BAD_ARGS", "args must be an object", details={"got": type(args).__name__})
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return _err("BAD_ARGS", "missing argument: " + ", ".join(missing), "Pass every required argument", {"required": schema.get("required", []), "missing": missing, "allowed": sorted(props)})
    unknown = [k for k in args if props and k not in props and schema.get("additionalProperties") is not True]
    if unknown:
        return _err("BAD_ARGS", "unknown argument: " + ", ".join(unknown), "Use only the listed arguments", {"unknown": unknown, "allowed": sorted(props)})
    for key, value in args.items():
        bad = _check_value(props.get(key) or {}, value, key)
        if bad:
            return bad
    return None


def _check_value(spec, value, where):
    kind = spec.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if kinds and not any(_is(value, k) for k in kinds):
        return _err("BAD_ARGS", f"{where} must be {' or '.join(kinds)}", details={"argument": where, "type": kinds, "got": type(value).__name__})
    if "enum" in spec and value not in spec["enum"]:
        return _err("BAD_ARGS", f"{where} must be one of {spec['enum']}", details={"argument": where, "allowed": list(spec["enum"])})
    if _is(value, "number"):
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            return _err("BAD_ARGS", f"{where} must be between {lo} and {hi}", details={"argument": where, "minimum": lo, "maximum": hi})
    if isinstance(value, (list, tuple)) and "items" in spec:
        for i, item in enumerate(value):
            bad = _check_value(spec["items"], item, f"{where}[{i}]")
            if bad:
                return bad
    return None


# --------------------------------------------------------------------------- registry

_registry: dict | None = None


def commands(reload: bool = False) -> dict:
    """Every command: the built-ins plus those of each handler module that exists. A missing module is skipped quietly."""
    global _registry
    if _registry is not None and not reload:
        return _registry
    reg = dict(BUILTIN)
    for mod in HANDLER_MODULES:
        try:
            m = importlib.import_module("." + mod if __package__ else mod, __package__ or None)
        except ModuleNotFoundError as e:
            if e.name and e.name.split(".")[-1] == mod:
                continue
            raise
        reg.update(getattr(m, "COMMANDS", {}))
    _registry = reg
    return reg


def _step(context):
    try:
        import bpy

        from . import history

        return history.current((context or bpy.context).scene)
    except Exception:
        return None


def run(name, args=None, context=None) -> dict:
    """Run one command and return the reply dict {ok, result | error, step, seconds}."""
    t0 = time.time()
    try:
        cmd = commands().get(name)
        if cmd is None:
            close = difflib.get_close_matches(str(name), list(commands()), n=5, cutoff=0.5)
            err = _err("UNKNOWN_COMMAND", f"no command called {name!r}", "Call commands.list to see every command", {"close": close})
        else:
            err = check_args(cmd.get("args"), {} if args is None else args)
        if err is None:
            if context is None:
                import bpy

                context = bpy.context
            result = cmd["fn"](context, **(args or {}))
            reply = {"ok": True, "result": {} if result is None else result}
        else:
            reply = {"ok": False, "error": err}
    except Exception as e:  # noqa: BLE001 - every failure becomes a reply
        code = getattr(e, "agent_code", None)
        if code:
            reply = {"ok": False, "error": _err(code, e, getattr(e, "hint", ""), getattr(e, "details", None))}
        else:
            trace = traceback.format_exc().strip().splitlines()[-8:]
            reply = {"ok": False, "error": _err("INTERNAL", f"{type(e).__name__}: {e}", "This is a bug in CurioMesh. Please report it with the trace", {"trace": trace})}
    reply["step"] = _step(context)
    reply["seconds"] = round(time.time() - t0, 3)
    return reply


# --------------------------------------------------------------------------- built-ins


def _commands_list(context):
    return {"commands": [{"name": n, "help": c.get("help", ""), "args": c.get("args", {"type": "object"}), "edits": bool(c.get("edits", False))} for n, c in sorted(commands().items())], "codes": CODES}


def _session_info(context):
    import sys

    import bpy

    from . import guided

    try:
        version = ".".join(str(x) for x in sys.modules[__package__].bl_info["version"])
    except Exception:
        version = "unknown"
    obj = context.object
    return {
        "curiomesh": version,
        "build": "main",
        "blender": bpy.app.version_string,
        "os": sys.platform,
        "object": obj.name if obj is not None else None,
        "has_result": guided.state.get("res") is not None,
        "commands": len(commands()),
    }


def _session_quit(context):
    return {"quitting": True}  # the mailbox / cm_do loop sees the command name and stops


_NONE = {"type": "object", "properties": {}}
BUILTIN = {
    "commands.list": {"fn": _commands_list, "args": _NONE, "help": "Every command with its plain-words help, argument schema and whether it edits, plus the error codes.", "edits": False},
    "session.info": {"fn": _session_info, "args": _NONE, "help": "CurioMesh and Blender versions, build, OS, active object, whether a result is on screen.", "edits": False},
    "session.quit": {"fn": _session_quit, "args": _NONE, "help": "Stop a headless mailbox loop (cm_do.py --serve). In a live window it does nothing.", "edits": False},
}
