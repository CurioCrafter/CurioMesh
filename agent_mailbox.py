# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""File mailbox for agents: <folder>/inbox/<name>.json in, <folder>/outbox/<name>.json out. Files only, no sockets.

Live Blender: the switch `WindowManager.curiomesh_agent_on` ("Let an Agent Drive CurioMesh", off by default) runs a timer
that handles one message per tick, in name order, on the main thread. Headless: `serve(folder)` runs the same loop.
A message is {"id"?, "command", "args"?}; write it as NAME.json.tmp and rename it so it is never read half-written.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

INTERVAL = 0.2  # seconds between ticks in a live window


# --------------------------------------------------------------------------- pure helpers


def pending(names) -> list[str]:
    """The inbox file names to handle, in order: *.json only (so *.tmp and anything else is skipped), sorted by name."""
    return sorted(n for n in names if n.endswith(".json") and not n.startswith("."))


def outbox_name(inbox_name: str) -> str:
    return inbox_name  # the reply carries the same name as the message


def parse(text: str, name: str = ""):
    """(command, args, id) from a message, or (None, None, id) plus an error reply dict when it is not usable."""
    stem = name[:-5] if name.endswith(".json") else name
    try:
        msg = json.loads(text)
    except ValueError as e:
        return None, None, {"id": stem, "ok": False, "error": {"code": "FILE", "message": f"{name}: not valid JSON ({e})", "hint": "Write one JSON object: {\"command\": ..., \"args\": {...}}", "details": {}}}
    if not isinstance(msg, dict) or not isinstance(msg.get("command"), str):
        return None, None, {"id": stem, "ok": False, "error": {"code": "BAD_ARGS", "message": "a message needs a string \"command\"", "hint": "Write {\"command\": \"session.info\", \"args\": {}}", "details": {}}}
    return msg["command"], msg.get("args") or {}, str(msg.get("id", stem))


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    os.replace(tmp, path)


def handle_one(folder, context=None):
    """Handle the first pending message. Returns the command name handled, or None when the inbox is empty."""
    try:
        from . import agentcmd
    except ImportError:  # loaded as a plain module (unit tests)
        import agentcmd

    folder = Path(folder)
    inbox, outbox = folder / "inbox", folder / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    names = pending(os.listdir(inbox))
    if not names:
        return None
    name = names[0]
    path = inbox / name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None  # still being written or locked: try again next tick
    command, args, extra = parse(text, name)
    if command is None:
        reply = extra
    else:
        reply = {"id": extra, **agentcmd.run(command, args, context)}
    _write_json(outbox / outbox_name(name), reply)
    try:
        path.unlink()
    except OSError:
        pass
    return command or ""


def serve(folder, context=None, poll: float = 0.05, idle_limit: float | None = None) -> None:
    """Headless loop: handle messages until a session.quit arrives (or nothing arrived for idle_limit seconds)."""
    last = time.time()
    while True:
        done = handle_one(folder, context)
        if done == "session.quit":
            return
        if done is None:
            if idle_limit is not None and time.time() - last > idle_limit:
                return
            time.sleep(poll)
        else:
            last = time.time()


def default_folder() -> Path | None:
    from . import hardware

    base = hardware.user_folder("agent")  # extension user folder, or the add-on's config folder
    return base


# --------------------------------------------------------------------------- live Blender


def _tick():
    import bpy

    if not bpy.context.window_manager.curiomesh_agent_on:
        return None
    folder = default_folder()
    if folder is not None:
        try:
            handle_one(folder)
        except Exception as e:  # never let the timer die
            print("CurioMesh agent mailbox:", e)
    return INTERVAL


def _on_change(wm, _context):
    import bpy

    if wm.curiomesh_agent_on and not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=INTERVAL, persistent=True)
    elif not wm.curiomesh_agent_on and bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)


def register():
    import bpy

    bpy.types.WindowManager.curiomesh_agent_on = bpy.props.BoolProperty(
        name="Let an Agent Drive CurioMesh",
        description="Let an agent send commands by dropping files in CurioMesh's agent folder. Off unless you turn it on. Files only, no network",
        default=False,
        update=_on_change,
    )


def unregister():
    import bpy

    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if hasattr(bpy.types.WindowManager, "curiomesh_agent_on"):
        del bpy.types.WindowManager.curiomesh_agent_on
