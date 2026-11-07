# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""Bring Back in background Blender: Auto on a self-made mesh (a subdivided monkey), the worst part written out in the
part folder format, given back with its inside changed, and parts.import. Checks one History step, the From Outside
try, a History jump back, and a refused part that leaves everything as it was.

    blender -b --factory-startup --python tests/blender_parts_import.py -- --addon <folder of the add-on, a valid module name>

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
sys.path.insert(0, os.path.dirname(ADDON))
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
PI = importlib.import_module(pkg.__name__ + ".parts_import")
PI.register()  # by hand: __init__.py gets its hook line after the UV merge
guided, H, segments, L = pkg.guided, pkg.history, pkg.segments, PI.partload
P, S = PI.P, PI.S
boundary_loops = importlib.import_module(pkg.__name__ + ".triadq.patch").boundary_loops
FAILS = []


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def settle():
    """Step whatever runs (the search, the part finder, a check) to the end, as the panel's timer would."""
    for _ in range(60):
        work = guided.state["work"]
        if work is None:
            return
        runner = work["runner"]
        while runner.busy:
            runner.step(0.05)
        guided._collect_work(bpy.context)
        if guided.state["work"] is work or guided.state["work"] is None or not guided.state["work"]["runner"].busy:
            return


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
settle()
segments.ensure()
ok(segments.ready() and not segments.mirrored(), f"part swaps allowed: ready {segments.ready()}, mirrored {segments.mirrored()}")

# --- the worst part, written out as an exporter would
base = guided.state["auto_base"]
out, piece = base["out"], np.asarray(segments.seg["piece"])
co, faces = np.asarray(out["co"], dtype=np.float64), list(out["faces"])
parts = segments.seg["parts"]
def has_border(q):
    try:
        return bool(S.seam_loops(co, faces, np.flatnonzero(piece == q)))
    except ValueError:
        return False


p = next(q for q in P.worst_first([segments.judged(r) for r in parts]) if (piece == q).sum() >= 8 and has_border(q))
pts, local, mats = P.cut(co, faces, out["mat"], piece, p)
loops = S.seam_loops(co, faces, np.flatnonzero(piece == p))
print(f"INFO part {p}: {len(local)} faces, {len(loops)} border loop(s), {len(parts)} parts", flush=True)
folder = Path(tempfile.mkdtemp(prefix="cm_parts_"))
(folder / "returned").mkdir()


def obj(points, fs, ms=None):
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in np.asarray(points, dtype=np.float32).astype(np.float64).tolist()]
    last = None
    for i, f in enumerate(fs):
        if ms is not None and int(ms[i]) != last:
            last = int(ms[i])
            lines.append(f"usemtl m{last}")
        lines.append("f " + " ".join(str(v + 1) for v in f))
    return "\n".join(lines) + "\n"


index = {tuple(q): i for i, q in enumerate(pts.tolist())}
manifest = {
    "format": "curiomesh-parts", "version": [1, 0], "made_by": "CurioMesh 0.9.9", "object": "Monkey",
    "result": {"result_faces": len(faces), "result_key": L.result_key(len(faces), np.vstack(loops) if loops else np.zeros((0, 3)))},
    "parts": [{"id": int(p), "name": segments.name_of(p), "border": [[index[tuple(q)] for q in loop.tolist()] for loop in loops], "border_points": np.vstack(loops).tolist() if loops else [], "materials": [int(mats[0])]}],
}
(folder / "parts.json").write_text(json.dumps(manifest))
(folder / f"part_{p}.obj").write_text(obj(pts, local, mats))
edge = {v for loop in boundary_loops(local, range(len(local)))[0] for v in loop}
size = float(np.ptp(co, axis=0).max())
moved = pts.copy()
inside = [v for v in range(len(pts)) if v not in edge]
moved[inside] += size * 0.002 * np.sin(np.arange(len(inside)))[:, None] * np.asarray([[0.3, 0.5, 0.8]])
split = [(f[0], f[i], f[i + 1]) for f in local for i in range(1, len(f) - 1)]  # every face as a fan of triangles: the border stays
(folder / "returned" / f"part_{p}.obj").write_text(obj(moved, split))

# --- a border point moved: refused, nothing changes
steps_now, faces_now = len(H.steps.entries), len(guided.state["res"]["faces"])
far = moved.copy()
far[next(iter(edge))] += size * 0.01
(folder / "returned" / f"part_{p}.obj").write_text(obj(far, split))
reply = PI.cmd_import(bpy.context, str(folder))
settle()
ok(not reply["kept"] and reply["refused"] and reply["refused"][0]["error"]["code"] == "BORDER_MOVED", f"moved border refused: {reply['refused'][:1]}")
ok(len(H.steps.entries) == steps_now and len(guided.state["res"]["faces"]) == faces_now, "a refused part changes nothing")

(folder / "returned" / f"part_{p}.obj").write_text(obj(moved, split))

# --- bring it back
faces_before, steps_before = len(guided.state["res"]["faces"]), len(H.steps.entries)
reply = PI.cmd_import(bpy.context, str(folder))
print(f"INFO reply: {json.dumps(reply, default=str)[:600]}", flush=True)
settle()  # the whole-object check after the swap
ok([k["part"] for k in reply["kept"]] == [p] and not reply["refused"], "the part is kept, nothing refused")
ok(len(H.steps.entries) == steps_before + 1, f"exactly one new History step: {[e.name for e in H.steps.entries[steps_before:]]}")
ok(H.steps.entries[-1].name == PI.NAME, f"named {PI.NAME!r}")
record = segments.seg["parts"][p]
ok(record["tries"][record["cur"]]["method"] == "OUTSIDE" and record["method"] == "OUTSIDE", f"the try is From Outside: {segments.method_name(record['method'])}")
ok(len(guided.state["res"]["faces"]) == faces_before + len(split) - len(local), f"the faces came in: {faces_before} -> {len(guided.state['res']['faces'])}")

# --- History jump back one step restores the face count
ok(H.jump(bpy.context, len(H.steps.entries) - 2), "History jump back")
settle()
ok(len(guided.state["res"]["faces"]) == faces_before, f"face count back: {len(guided.state['res']['faces'])} (was {faces_before})")
segments.ensure()
print(f"INFO after the jump: {len(segments.seg['parts'])} parts in the table (was {len(parts)})", flush=True)

try:
    PI.cmd_import(bpy.context, str(folder), look_first=True)
    ok(False, "look_first headless is refused")
except PI.NeedsWindow as exc:
    ok(exc.agent_code == "NEEDS_WINDOW", "look_first headless gives NEEDS_WINDOW")

shutil.rmtree(folder, ignore_errors=True)
PI.unregister()
print("RESULT", "FAIL" if FAILS else "PASS", FAILS, flush=True)
sys.exit(1 if FAILS else 0)
