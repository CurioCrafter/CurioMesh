# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""parts.export and parts.report in background Blender: Auto on a subdivided monkey, mark the two worst parts, export them.
Registers parts_export by hand (the __init__ hook is not in yet).

    blender -b --factory-startup --python tests/blender_parts_export.py -- --addon <folder of the add-on, a valid module name>

On the owner's PC start Blender only through bench/rough/lowblender.sh. Exits 1 on any failure.
"""

import importlib
import json
import os
import sys
import tempfile
import zipfile

import bpy
import numpy as np

ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ADDON = os.path.abspath(ARGV[ARGV.index("--addon") + 1]) if "--addon" in ARGV else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ADDON))
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
PE = importlib.import_module(pkg.__name__ + ".parts_export")
PE.register()
guided, H, segments = pkg.guided, pkg.history, pkg.segments
FAILS = []


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_monkey_add()
src = bpy.context.object
src.name = "Monkey"
mod = src.modifiers.new("sub", "SUBSURF")
mod.levels = 2
bpy.ops.object.modifier_apply(modifier=mod.name)
gs = bpy.context.scene.curiomesh_guided
gs.scope, gs.auto_update = "AUTO", False
ok(bpy.ops.curiomesh.guided_auto_start() == {"FINISHED"}, "Auto starts")
for _ in range(60):
    work = guided.state["work"]
    if work is None:
        break
    runner = work["runner"]
    while runner.busy:
        runner.step(0.05)
    guided._collect_work(bpy.context)
    if guided.state["work"] is work or guided.state["work"] is None or not guided.state["work"]["runner"].busy:
        break
ok(guided.state["res"] is not None, f"a result: {guided.state['message']}")
segments.ensure()
ok(segments.ready() and len(segments.seg["parts"]) >= 2, f"{len(segments.seg['parts'])} parts in the table")

order = segments.P.worst_first([segments.judged(r) for r in segments.seg["parts"]], 2)
segments.select(order[:2])
step0, names0 = H.current(bpy.context.scene), [e.name for e in H.steps.entries]
out_dir = tempfile.mkdtemp(prefix="cm_parts_")
reply = PE.COMMANDS["parts.export"]["fn"](bpy.context, folder=out_dir, note="test note")
folder = reply["folder"]
print("INFO reply", {k: v for k, v in reply.items()}, flush=True)
ok(len(reply["parts"]) + len(reply["skipped"]) == 2, "both marked parts are exported or skipped with a reason")
ok(os.path.isfile(os.path.join(folder, "parts.json")) and os.path.isdir(os.path.join(folder, "returned")), "parts.json and returned/ exist")
man = json.load(open(os.path.join(folder, "parts.json")))
ok(man["format"] == "curiomesh-parts" and man["version"] == [1, 0] and man["note"] == "test note", "manifest header")
for e in man["parts"]:
    for suffix in ("", "_source", "_around"):
        ok(os.path.isfile(os.path.join(folder, f"part_{e['id']}{suffix}.obj")), f"part_{e['id']}{suffix}.obj exists")
    ok(e["faces"] > 0 and (not e["border"] or e["border"][0][0] == e["border"][0][-1]), f"part {e['id']} faces and closed border loops")
ok(len(man["parts"]) >= 1, "at least one part written")
ok(H.current(bpy.context.scene) == step0 and [e.name for e in H.steps.entries] == names0, "History step count unchanged")

zpath = os.path.join(out_dir, "r.zip")
PE.COMMANDS["parts.report"]["fn"](bpy.context, path=zpath, note="why")
names = zipfile.ZipFile(zpath).namelist()
ok("report.md" in names and "settings.json" in names and any(n.endswith("/parts.json") for n in names), "report zip has report.md, settings.json and the parts")
md = zipfile.ZipFile(zpath).read("report.md").decode()
ok("Check them before you share it" in md and "why" in md, "report.md carries the warning and the note")
try:
    PE.COMMANDS["parts.export"]["fn"](bpy.context, parts=[9999])
    ok(False, "unknown part refused")
except PE.PartsError as exc:
    ok(exc.agent_code == "NOT_FOUND", "unknown part gives NOT_FOUND")
ok(H.current(bpy.context.scene) == step0, "History still unchanged")

PE.unregister()
print("FAILS:", FAILS, flush=True)
sys.exit(1 if FAILS else 0)
