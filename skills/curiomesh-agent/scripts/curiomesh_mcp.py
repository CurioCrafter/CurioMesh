# SPDX-License-Identifier: GPL-3.0-or-later
# main-only: agent tools (not in the extensions.blender.org build)
"""CurioMesh MCP server (stdio, JSON-RPC 2.0, Python standard library only).

It sits outside the add-on and talks to a CurioMesh mailbox folder (inbox/ and outbox/ with one JSON file per message).

  python curiomesh_mcp.py --folder FOLDER     use the mailbox of a Blender that is open (Pro tab > This Computer >
                                              Let an Agent Drive CurioMesh shows the folder)
  python curiomesh_mcp.py                     start its own headless Blender running cm_do.py --serve, in a temp folder

Every command from `commands.list` becomes one MCP tool (a dot in the name becomes an underscore: auto.start is
auto_start). `curiomesh_call {command, args}` reaches any command by its real name. Each tool also takes `object`
(a mesh name) and `timeout` (seconds to wait for the answer; leave it out to wait as long as Blender is working).
"""

import argparse
import glob
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROTOCOL = "2024-11-05"
HERE = Path(__file__).resolve().parent


class MailboxError(Exception):
    pass


class Mailbox:
    """Writes inbox/<id>.json (tmp, then rename), waits for outbox/<id>.json. The only limit is the caller's timeout."""

    def __init__(self, folder, proc=None):
        self.folder = Path(folder)
        self.inbox = self.folder / "inbox"
        self.outbox = self.folder / "outbox"
        self.proc = proc
        self.seq = itertools.count(1)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)

    def _id(self):
        # time first, so names sort in the order we wrote them, even across restarts
        return f"{int(time.time() * 1000):013d}-{next(self.seq):04d}"

    def call(self, command, args=None, obj=None, timeout=None):
        mid = self._id()
        message = {"id": mid, "command": command, "args": dict(args or {})}
        if obj:
            message["args"]["object"] = obj
        tmp = self.inbox / f"{mid}.json.tmp"
        tmp.write_text(json.dumps(message), encoding="utf-8")
        os.replace(tmp, self.inbox / f"{mid}.json")
        reply_path = self.outbox / f"{mid}.json"
        start = time.time()
        pause = 0.02
        while True:
            if reply_path.exists():
                try:
                    return json.loads(reply_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass  # half written or locked: look again
            if self.proc is not None and self.proc.poll() is not None:
                raise MailboxError(f"Blender stopped (exit code {self.proc.returncode}) before it answered {command}")
            if timeout is not None and time.time() - start > timeout:
                raise MailboxError(f"no answer to {command} after {timeout:g} s; Blender may still be working: ask job.status")
            time.sleep(pause)
            pause = min(pause * 1.5, 0.25)

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


def find_blender():
    """$BLENDER, then PATH, then the usual install places. Never a build or bench folder."""
    env = os.environ.get("BLENDER")
    if env and Path(env).exists():
        return env
    found = shutil.which("blender")
    if found:
        return found
    patterns = [
        r"C:\Program Files\Blender Foundation\Blender *\blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/snap/bin/blender",
        os.path.expanduser("~/blender*/blender"),
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def find_cm_do(explicit=None):
    candidates = [explicit] if explicit else []
    repo = os.environ.get("CURIOMESH_REPO")
    if repo:
        candidates += [Path(repo) / "cm_do.py", Path(repo) / "skills" / "curiomesh-retopo" / "scripts" / "cm_do.py"]
    candidates += [HERE / "cm_do.py", HERE.parent.parent / "curiomesh-retopo" / "scripts" / "cm_do.py"]
    for path in candidates:
        if path and Path(path).is_file():
            return str(path)
    return None


def start_blender(args):
    blender = args.blender or find_blender()
    if not blender:
        raise MailboxError("Blender not found: set $BLENDER or pass --blender, or pass --folder of a Blender that is open")
    script = find_cm_do(args.cm_do)
    if not script:
        raise MailboxError("cm_do.py not found: pass --cm-do, or set CURIOMESH_REPO to the CurioMesh folder")
    folder = args.workdir or tempfile.mkdtemp(prefix="curiomesh-mailbox-")
    Path(folder).mkdir(parents=True, exist_ok=True)
    cmd = [blender, "-b", "--factory-startup", "--python", script, "--", "--serve", folder]
    if args.addon:
        cmd += ["--addon", args.addon]
    log = open(Path(folder) / "blender.log", "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return Mailbox(folder, proc)


# ---------------------------------------------------------------- MCP side


def tool_name(command):
    return command.replace(".", "_")


def input_schema(entry):
    """Turn a commands.list entry's argument description into a JSON schema; always allow object and timeout."""
    raw = entry.get("schema") or entry.get("args") or entry.get("arguments") or {}
    if isinstance(raw, dict) and ("properties" in raw or raw.get("type") == "object"):
        schema = json.loads(json.dumps(raw))
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
    else:
        props = {}
        if isinstance(raw, dict):
            for key, desc in raw.items():
                props[key] = desc if isinstance(desc, dict) else {"description": str(desc)}
        schema = {"type": "object", "properties": props}
    schema["properties"].setdefault("object", {"type": "string", "description": "Mesh object name; the active mesh if left out"})
    schema["properties"].setdefault("timeout", {"type": "number", "description": "Seconds to wait for the answer; leave out to wait until Blender answers"})
    return schema


class Server:
    def __init__(self, mailbox, start_wait=120.0):
        self.mailbox = mailbox
        self.start_wait = start_wait
        self.commands = None  # name -> entry, filled by the first commands.list that works

    def load_commands(self):
        if self.commands is None:
            reply = self.mailbox.call("commands.list", timeout=self.start_wait)
            if not reply.get("ok"):
                raise MailboxError(f"commands.list failed: {reply.get('error')}")
            result = reply.get("result")
            entries = result.get("commands", []) if isinstance(result, dict) else (result or [])
            self.commands = {e["name"]: e for e in entries if isinstance(e, dict) and "name" in e}
        return self.commands

    def tools(self):
        out = [{
            "name": "curiomesh_call",
            "description": "Run any CurioMesh command by its real name (for example auto.start) with a JSON args object.",
            "inputSchema": {"type": "object", "required": ["command"], "properties": {
                "command": {"type": "string"}, "args": {"type": "object"},
                "object": {"type": "string"}, "timeout": {"type": "number"}}},
        }]
        try:
            commands = self.load_commands()
        except MailboxError as exc:
            out[0]["description"] += f" (command list not read yet: {exc})"
            return out
        for name, entry in sorted(commands.items()):
            text = entry.get("help") or name
            if entry.get("edits"):
                text += " (changes the result; one History step)"
            out.append({"name": tool_name(name), "description": text, "inputSchema": input_schema(entry)})
        return out

    def real_name(self, tool):
        for name in self.load_commands():
            if tool_name(name) == tool:
                return name
        return tool.replace("_", ".", 1)  # let CurioMesh answer UNKNOWN_COMMAND with its close matches

    def call_tool(self, tool, arguments):
        arguments = dict(arguments or {})
        timeout = arguments.pop("timeout", None)
        obj = arguments.pop("object", None)
        if tool == "curiomesh_call":
            command, args = arguments.get("command"), arguments.get("args") or {}
            if not command:
                return {"ok": False, "error": {"code": "BAD_ARGS", "message": "command is required", "hint": "for example {\"command\": \"session.info\"}"}}
        else:
            command, args = self.real_name(tool), arguments
        return self.mailbox.call(command, args, obj, timeout)

    def handle(self, msg):
        """One JSON-RPC message in, the reply dict out (None for notifications)."""
        mid, method = msg.get("id"), msg.get("method", "")

        def ok(result):
            return {"jsonrpc": "2.0", "id": mid, "result": result}

        if mid is None:
            return None  # notifications/initialized and friends
        if method == "initialize":
            return ok({"protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
                       "serverInfo": {"name": "curiomesh", "version": "1"}})
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": self.tools()})
        if method == "tools/call":
            params = msg.get("params") or {}
            try:
                reply = self.call_tool(params.get("name", ""), params.get("arguments"))
            except MailboxError as exc:
                reply = {"ok": False, "error": {"code": "NO_ANSWER", "message": str(exc)}}
            return ok({"content": [{"type": "text", "text": json.dumps(reply, indent=2)}], "isError": not reply.get("ok", False)})
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve(server, stdin, stdout):
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
            stdout.flush()
            continue
        reply = server.handle(msg)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="curiomesh_mcp.py", description=__doc__.split("\n")[0])
    ap.add_argument("--folder", help="mailbox folder of a running CurioMesh (inbox/ and outbox/ inside)")
    ap.add_argument("--blender", help="Blender executable (default: $BLENDER, PATH, usual install places)")
    ap.add_argument("--cm-do", dest="cm_do", help="path to cm_do.py")
    ap.add_argument("--addon", help="CurioMesh repo folder, when it is not installed as an add-on")
    ap.add_argument("--workdir", help="mailbox folder for the headless Blender (default: a temp folder)")
    ap.add_argument("--start-wait", type=float, default=120.0, help="seconds to wait for the first command list")
    args = ap.parse_args(argv)
    try:
        mailbox = Mailbox(args.folder) if args.folder else start_blender(args)
    except MailboxError as exc:
        sys.stderr.write(f"curiomesh_mcp: {exc}\n")
        return 2
    try:
        stdin = open(sys.stdin.fileno(), "r", encoding="utf-8", newline="\n", closefd=False)
        stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", newline="\n", closefd=False)
        serve(Server(mailbox, args.start_wait), stdin, stdout)
    finally:
        mailbox.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
