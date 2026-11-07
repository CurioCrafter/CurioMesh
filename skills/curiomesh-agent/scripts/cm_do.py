# SPDX-License-Identifier: GPL-3.0-or-later
# main-only: agent tools (not in the extensions.blender.org build)
"""Run CurioMesh agent commands headless. Every reply is one JSON line, the same shape as the mailbox and the MCP server.

blender -b --factory-startup --python cm_do.py -- --commands job.json --out replies.jsonl [--input MESH] [--addon REPO_FOLDER]
blender -b --factory-startup --python cm_do.py -- --serve FOLDER [--input MESH]     (until a {"command": "session.quit"} arrives)

job.json is a list of {"command": ..., "args": {...}}. Exit code 1 when any reply has "ok": false.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "curiomesh-retopo" / "scripts"))
import cm_agent as cm  # noqa: E402


def run_list(package, items, out=None):
    """Run the commands in order; returns the replies. Prints each as a line and appends it to `out` when given."""
    import bpy

    agentcmd = cm.sub(package, "agentcmd")
    replies, lines = [], []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "command" not in item:
            reply = {"ok": False, "error": {"code": "BAD_ARGS", "message": f"item {i} needs a \"command\"", "hint": "", "details": {}}}
        else:
            reply = agentcmd.run(item["command"], item.get("args") or {}, bpy.context)
            if "id" in item:
                reply = {"id": item["id"], **reply}
        replies.append(reply)
        line = json.dumps(reply, default=str)
        lines.append(line)
        print(line, flush=True)
        if isinstance(item, dict) and item.get("command") == "session.quit":
            break
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return replies


def main() -> int:
    ap = argparse.ArgumentParser(prog="cm_do.py")
    ap.add_argument("--commands", help="JSON file: a list of {command, args}")
    ap.add_argument("--serve", help="mailbox folder to serve until session.quit")
    ap.add_argument("--out", help="write the replies here, one JSON per line")
    ap.add_argument("--input", help="mesh file to import first (as curiomesh_run.py does)")
    ap.add_argument("--addon", default=None)
    args = ap.parse_args(cm.script_args())
    if bool(args.commands) == bool(args.serve):
        ap.error("give exactly one of --commands or --serve")
    package = cm.load_addon(args.addon)
    if args.input:
        cm.select_only(cm.load_mesh(args.input, Path(args.input).stem))
    if args.serve:
        cm.sub(package, "agent_mailbox").serve(args.serve)
        return 0
    items = json.loads(Path(args.commands).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        ap.error("--commands file must hold a JSON list")
    replies = run_list(package, items, args.out)
    return 1 if any(not r.get("ok") for r in replies) else 0


if __name__ == "__main__":
    sys.exit(main())
