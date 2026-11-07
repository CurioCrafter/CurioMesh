# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""An agent-style session through agentcmd.run only: load a mesh, Auto, mark the two worst parts, Send to Agent,
put the files back unchanged, Bring Back. Checks both parts are kept, the camera check passes, one History step is
added, and Ctrl+Z (history.undo; Blender's own undo is not there headless) gives the mesh from before back.

    blender -b --factory-startup --python tests/blender_agent_roundtrip.py -- --input MESH [--addon FOLDER]

On the owner's PC start Blender only through bench/rough/lowblender.sh. Exits 1 on any failure.
"""

import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ADDON = os.path.abspath(ARGV[ARGV.index("--addon") + 1]) if "--addon" in ARGV else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH = ARGV[ARGV.index("--input") + 1]
sys.path.insert(0, os.path.dirname(ADDON))
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
hooked = [importlib.import_module(f"{pkg.__name__}.{m}") for m in ("agent_mailbox", "parts_export", "parts_import")]
for _ in range(2):  # by hand until the __init__.py hook lines land (H1); twice, as a reload would
    for m in hooked:
        try:
            m.unregister()
        except Exception:  # noqa: BLE001 - not registered yet
            pass
        m.register()
agentcmd, H, guided = pkg.agentcmd, pkg.history, pkg.guided
FAILS = []


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def do(_command, /, **args):
    reply = agentcmd.run(_command, args, bpy.context)
    print(f"CMD {_command} {json.dumps(args)} -> {json.dumps(reply, default=str)[:400]}", flush=True)
    return reply


def mesh_key():
    res = guided.state["res"]
    return len(res["faces"]), np.asarray(res["co"], dtype=np.float64).copy()


for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
ok(do("file.load", path=MESH, name="Bust")["ok"], "file.load")
ok(do("auto.start")["ok"], "auto.start")
st = do("job.wait", seconds=3600)
ok(st["ok"] and not st["result"]["running"], "Auto finished")
listing = do("parts.list")
ok(listing["ok"], "parts.list")
ok(do("parts.worst", count=2)["ok"], "two parts marked")
folder = Path(tempfile.mkdtemp(prefix="cm_agent_rt_"))
exp = do("parts.export", folder=str(folder), note="round trip, unchanged")
ok(exp["ok"] and len(exp["result"]["parts"]) == 2, f"two parts sent: {exp.get('result')}")
out = Path(exp["result"]["folder"])
for f in out.glob("part_*.obj"):
    shutil.copy(f, out / "returned" / f.name)  # the agent gives them back as they were

faces0, co0 = mesh_key()
steps0 = len(H.steps.entries)
imp = do("parts.import", folder=str(out))
r = imp.get("result", {})
ok(imp["ok"] and len(r.get("kept", [])) == 2 and not r.get("refused"), f"both parts kept, none refused: {r.get('refused')}")
ok(len(H.steps.entries) == steps0 + 1, f"one History step: {[e.name for e in H.steps.entries[steps0:]]}")
num = do("result.numbers")
print(f"INFO numbers {json.dumps(num.get('result'), default=str)[:600]}", flush=True)
faces1, co1 = mesh_key()

w = do("job.wait", seconds=3600)  # the swap starts a whole-result check; an agent waits for it as job.status says
ok(w["ok"] and not w["result"]["running"], "the whole-result check after the swap finished")
ok(len(H.steps.entries) == steps0 + 1, "still one History step after the check")
und = do("history.undo")
ok(und["ok"], "Ctrl+Z (history.undo)")
faces2, co2 = mesh_key()


def far(a, b):
    """Largest distance from a point of one set to the nearest point of the other (point order may differ)."""
    d = lambda p, q: max(np.sqrt(((p[i : i + 512, None] - q[None]) ** 2).sum(-1)).min(1).max() for i in range(0, len(p), 512))  # noqa: E731
    return max(d(a, b), d(b, a))


size = float(np.ptp(co0, axis=0).max())
print(f"INFO size {size:.3g}; points {len(co0)} {len(co1)} {len(co2)}; Bring Back moved {far(co1, co0):.3g}; after Ctrl+Z {far(co2, co0):.3g}; same order {co2.shape == co0.shape and np.allclose(co2, co0)}", flush=True)
ok(faces1 == faces0 and far(co1, co0) < 1e-5 * size, "unchanged parts come back in the same place")
ok(faces2 == faces0 and far(co2, co0) < 1e-5 * size, f"Ctrl+Z gives the mesh back: {faces2} faces")

shutil.rmtree(folder, ignore_errors=True)
for m in reversed(hooked):
    m.unregister()
left = [n for n in ("CURIOMESH_OT_parts_export", "CURIOMESH_OT_parts_report", "CURIOMESH_OT_parts_import") if hasattr(bpy.types, n)]
ok(not left and not hasattr(bpy.types.WindowManager, "curiomesh_agent_on"), f"the hooked modules unregister clean: {left}")
print("RESULT", "FAIL" if FAILS else "PASS", FAILS, flush=True)
sys.exit(1 if FAILS else 0)
