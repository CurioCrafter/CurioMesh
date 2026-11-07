# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""cm_do.py --commands on a fixture mesh: session.info and commands.list answer ok, nope.cmd gives UNKNOWN_COMMAND.

bench/rough/lowblender.sh OUT.txt tests/blender_agent_cmd.py     (one background Blender, Idle priority)
"""

import json
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="cm_agent_cmd_"))
fx = json.loads((ROOT / "tests" / "fixtures" / "turned_tube.json").read_text())
obj = tmp / "tube.obj"
obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in fx["source_co"]) + "".join("f " + " ".join(str(i + 1) for i in f) + "\n" for f in fx["source_faces"]))
job = tmp / "job.json"
job.write_text(json.dumps([{"command": "session.info"}, {"command": "commands.list"}, {"command": "nope.cmd"}]))
out = tmp / "replies.jsonl"

sys.argv = ["cm_do.py", "--", "--commands", str(job), "--out", str(out), "--input", str(obj), "--addon", str(ROOT)]
code = 0
try:
    runpy.run_path(str(ROOT / "skills" / "curiomesh-agent" / "scripts" / "cm_do.py"), run_name="__main__")
except SystemExit as e:
    code = e.code or 0

replies = [json.loads(line) for line in out.read_text().splitlines()]
ok = [r["ok"] for r in replies]
assert code == 1, f"exit code {code}, wanted 1 (one reply is not ok)"
assert ok == [True, True, False], ok
assert replies[2]["error"]["code"] == "UNKNOWN_COMMAND", replies[2]
assert replies[0]["result"]["object"] == "tube", replies[0]
assert any(c["name"] == "session.info" for c in replies[1]["result"]["commands"])
print("AGENT_CMD_OK", json.dumps(replies[0]["result"]))
