# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""The agent tool commands (agentops_tools.py) in background Blender, on a subdivided monkey: Auto, tools.list and preview,
tools.run Source Relax with keep_going (one History step per kept pass), undo and redo, loops.fix_all, a strip edit, a Flow
Editor stroke, then autofix.run and autofix.undo (the face count and corners come back exactly).

    blender -b --factory-startup --python tests/blender_agent_tools.py -- [--addon <add-on folder>] [--model <file.gltf>] [--blender-undo]

On the owner's PC start Blender only through bench/rough/lowblender.sh. Exits 1 on any failure.
"""

import importlib
import os
import sys
import time

import bpy
import numpy as np

ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def arg(name, default=None):
    return ARGV[ARGV.index(name) + 1] if name in ARGV else default


ADDON = os.path.abspath(arg("--addon", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(ADDON))
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
guided, H, AL = pkg.guided, pkg.history, pkg.autofix_live
A = importlib.import_module(pkg.__name__ + ".agentops_tools")
FAILS = []


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def res_co():
    return np.asarray(guided.state["res"]["co"], dtype=np.float64).copy()


def faces_n():
    return len(guided.state["res"]["faces"])


def step():
    return H.current(bpy.context.scene)


def entries():
    return len(H.steps.entries)


def call(name, **args):
    """Run a command the way agentcmd would, minus its schema check: the handlers' own checks are what is tested."""
    return A.COMMANDS[name]["fn"](bpy.context, **args)


def fails(name, code, **args):
    try:
        call(name, **args)
    except A.ToolError as e:
        return e.agent_code == code
    return False


def start(src):
    bpy.context.view_layer.objects.active = src
    src.select_set(True)
    gs = bpy.context.scene.curiomesh_guided
    gs.scope, gs.auto_update = "AUTO", False
    r = bpy.ops.curiomesh.guided_auto_start()
    for _ in range(200):
        work = guided.state["work"]
        if work is None:
            break
        runner = work["runner"]
        while runner.busy:
            runner.step(0.05)
        guided._collect_work(bpy.context)
        if guided.state["work"] is work or guided.state["work"] is None or not guided.state["work"]["runner"].busy:
            break
    return r


def can_undo():
    bpy.ops.ed.undo_push(message="probe")
    bpy.ops.mesh.primitive_cube_add()
    probe = bpy.context.object.name
    can = True
    try:
        bpy.ops.ed.undo()
        can = probe not in bpy.data.objects
    except RuntimeError:
        can = False
    for o in list(bpy.data.objects):
        if o.name.startswith("Cube"):
            bpy.data.objects.remove(o, do_unlink=True)
    return can


def undo(can):
    if can:
        bpy.ops.ed.undo()
    else:
        H.settings()["step"] = H.steps.entries[H.steps.index_of(step()) - 1].step
        H._curiomesh_history_undo(bpy.context.scene)


def redo(can, to):
    if can:
        bpy.ops.ed.redo()
    else:
        H.settings()["step"] = to
        H._curiomesh_history_undo(bpy.context.scene)


# --- Auto on the fixture
model = arg("--model")
if model:
    bpy.ops.import_scene.gltf(filepath=model)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    src = bpy.context.view_layer.objects.active
else:
    bpy.ops.mesh.primitive_monkey_add()
    src = bpy.context.object
    mod = src.modifiers.new("sub", "SUBSURF")
    mod.levels = 2
    bpy.ops.object.modifier_apply(modifier=mod.name)
src.name = "Source"
ok(fails("tools.run", "NO_RESULT", tool="tool_source_relax"), "before Auto, a tool run says NO_RESULT")
t0 = time.perf_counter()
start(src)
ok(guided.state["res"] is not None, f"Auto made a result: {faces_n()} faces in {time.perf_counter() - t0:.0f} s")
can = can_undo() if "--blender-undo" in ARGV else False  # the probe (a cube add, then Ctrl+Z) crashes this Blender build in background: History's own undo stands in
H.steps.clear()
H.settings()["step"] = 0
bpy.ops.ed.undo_push(message="test start")

# --- listing, scope, preview
listing = call("tools.list")
names = [t["tool"] for t in listing["tools"]]
ok("tool_source_relax" in names and all({"label", "group", "help", "settings", "developer_only", "keep_going", "every_spot"} <= set(t) for t in listing["tools"]), f"tools.list: {len(names)} tools")
relax = next(t for t in listing["tools"] if t["tool"] == "tool_source_relax")
ok(relax["keep_going"] and any(s["name"] == "strength" and s["max"] == 1.0 for s in relax["settings"]), "tools.list: Source Relax settings carry their ranges")
ok(any(t["developer_only"] for t in listing["tools"]), "tools.list: developer-only tools are marked")
ok(call("tools.scope", scope="WHOLE")["faces_in_scope"] == faces_n(), "tools.scope WHOLE")
ok(fails("tools.scope", "BAD_ARGS", scope="SELECTION"), "tools.scope SELECTION without faces is BAD_ARGS")
call("tools.scope", scope="SELECTION", faces=list(range(min(40, faces_n()))))
ok(call("tools.preview", tool="tool_source_relax")["faces"][0] == faces_n(), "tools.preview on a selection")
call("tools.scope", scope="WHOLE")
ok(fails("tools.run", "BAD_ARGS", tool="tool_source_relax", settings={"strength": 2.0}), "a setting past its hard range is BAD_ARGS")
ok(fails("tools.run", "NOT_FOUND", tool="tool_nothing"), "an unknown tool is NOT_FOUND")
co0, step0, n0 = res_co(), step(), entries()
pv = call("tools.preview", tool="tool_source_relax", settings={"strength": 0.9})
ok(np.allclose(res_co(), co0) and step() == step0 and entries() == n0, "tools.preview changes nothing")
ok(pv["numbers"] and pv["moved_most"] is not None and (pv["would_keep"] or pv["problem"]["code"] == "REFUSED"), f"tools.preview gives the readout: {pv['readout']} (would keep: {pv['would_keep']})")

# --- every Tidy Tool has a path with no window: preview each one (nothing kept)
for t in listing["tools"]:
    try:
        p = call("tools.preview", tool=t["tool"])
        print(f"INFO {t['tool']}: would_keep {p['would_keep']} problem {p['problem'] and p['problem']['code']} readout {p['readout']!r}", flush=True)
    except A.ToolError as e:
        print(f"INFO {t['tool']}: {e.agent_code} {e}", flush=True)
    except Exception as e:  # a crash is the failure this loop looks for
        ok(False, f"{t['tool']}: preview raised {e!r}")
ok(np.allclose(res_co(), co0) and step() == step0, "no tool preview changed the result")

# --- tools.run: a change that adds bad faces is REFUSED and changes nothing; with allow_worse it is one pass, undo, redo
if not pv["would_keep"]:
    ok(fails("tools.run", "REFUSED", tool="tool_source_relax", settings={"strength": 0.9}) and np.allclose(res_co(), co0) and step() == step0 and entries() == n0, "a refused run changes nothing and adds no History step")
r = call("tools.run", tool="tool_source_relax", settings={"strength": 0.9}, allow_worse=True)
ok(r["kept"] == 1 and step() == step0 + 1 and entries() - n0 in (1, 2),  # the first change also stores a "Before" entry
    f"tools.run: one History step ({r['passes'][0]['readout']})")
after = res_co()
ok(not np.allclose(after, co0) and faces_n() == r["faces"][1], "tools.run changed the corners")
undo(can)
ok(np.allclose(res_co(), co0), "Ctrl+Z takes the run back")
redo(can, step0 + 1)
ok(np.allclose(res_co(), after), "Ctrl+Shift+Z brings it back")

# --- keep_going: every kept pass is one History step, and it stops by itself
n1, s1 = entries(), step()
t0 = time.perf_counter()
r = call("tools.run", tool="tool_source_relax", keep_going=True, allow_worse=True, seconds=300)
ok(r["kept"] >= 1 and entries() - n1 == r["kept"] and step() - s1 == r["kept"], f"keep_going: {r['kept']} kept passes = {entries() - n1} History steps ({time.perf_counter() - t0:.0f} s, ended: {r['ended']})")
ok(r["ended"] == "gained nothing" and not r["passes"][-1]["kept"], "keep_going stops on a pass that gains nothing")
n2 = entries()
a = call("tools.again", keep_going=False, allow_worse=True)
print(f"INFO tools.again: kept {a['kept']}", flush=True)
ok(entries() - n2 == a["kept"], "tools.again is one History step too")

# --- Fix every strip
n3 = entries()
try:
    f = call("loops.fix_all", keep_going=True, allow_worse=True, seconds=300)
    ok(entries() - n3 == f["kept"], f"loops.fix_all: {f['kept']} kept passes = {entries() - n3} History steps")
except A.ToolError as e:
    ok(e.agent_code == "REFUSED" and entries() == n3, f"loops.fix_all: nothing to fix here ({e})")
try:
    ev = call("tools.every", tool="tool_square_up", allow_worse=True)
    print(f"INFO tools.every square_up: kept {ev['kept']}", flush=True)
except A.ToolError as e:
    ok(e.agent_code == "REFUSED", f"tools.every: {e.agent_code} {e}")

# --- strips and poles
strips = call("loops.strips", limit=5)
ok(strips["total"] > 0 and strips["strips"] == sorted(strips["strips"], key=lambda r: -r["score"]), f"loops.strips: {strips['total']} strips, worst first")
poles = call("loops.poles", limit=5)
print(f"INFO loops.poles: {poles['total']}", flush=True)
ok(fails("loops.edit", "NOT_FOUND", strip=10**9, how="STRAIGHTEN"), "a strip that is not there is NOT_FOUND")
n4, s4, c4 = entries(), step(), res_co()
try:
    call("loops.edit", strip=strips["strips"][0]["strip"], how="STRAIGHTEN")
    ok(entries() == n4 + 1 and step() == s4 + 1, "loops.edit: one History step")
except A.ToolError as e:
    ok(e.agent_code == "REFUSED" and entries() == n4 and np.allclose(res_co(), c4), f"loops.edit refused and changed nothing: {e}")

# --- Flow Editor stroke
look = np.asarray(res_co())
a_, b_ = look[0], look[len(look) // 3]
try:
    fa = call("flow.add", kind="FLOW", points=[a_.tolist(), ((a_ + b_) / 2).tolist(), b_.tolist()])
    ok(fa["strokes"] == 1 and len(call("flow.list")["strokes"]) == 1, "flow.add then flow.list")
    ok(fails("flow.remove", "NOT_FOUND", index=5), "flow.remove past the end is NOT_FOUND")
    ok(call("flow.remove", index=0)["strokes"] == 0, "flow.remove")
except A.ToolError as e:
    ok(False, f"flow.add: {e.agent_code} {e}")
ok(fails("flow.add", "BAD_ARGS", kind="FLOW", points=[[0, 0, 0]]), "flow.add with one point is BAD_ARGS")

# --- Auto Fix, then undo it
g = call("autofix.groups")
ok(len(g["groups"]) == 7 and all(x["on"] for x in g["groups"]), "autofix.groups: seven, all ticked")
ok(fails("autofix.set", "BAD_ARGS", ticks={"nonsense": True}), "autofix.set with an unknown group is BAD_ARGS")
ok(fails("autofix.undo", "NOT_READY"), "autofix.undo before a run is NOT_READY")
faces_before, co_before, n5 = faces_n(), res_co(), entries()
t0 = time.perf_counter()
rep = call("autofix.run")
ok(not rep["running"] and rep["how"] in ("done", "limit"), f"autofix.run: {rep['words']} ({rep['kept']} of {rep['passes']} passes kept, {time.perf_counter() - t0:.0f} s)")
if rep["how"] == "limit":
    rep = call("autofix.keep_going")
fixes = [e.name for e in H.steps.entries[n5:] if e.name.startswith("Auto Fix: ")]
ok(len(fixes) == rep["kept"], f"autofix: one History step per kept pass ({len(fixes)} of {rep['kept']})")
fnd = call("autofix.findings")
ok(isinstance(fnd["findings"], list) and len(fnd["passes"]) == rep["passes"], f"autofix.findings: {len(fnd['findings'])} findings")
if rep["kept"]:
    u = call("autofix.undo")
    ok(faces_n() == faces_before, f"autofix.undo: face count back ({faces_n()} == {faces_before})")
    ok(np.allclose(res_co(), co_before), "autofix.undo: corners back")
else:
    print("INFO autofix kept nothing on this model: undo check skipped", flush=True)
ok(not AL.busy() and not bpy.app.timers.is_registered(AL._tick), "nothing left running")

print("FAILED: " + "; ".join(FAILS) if FAILS else "ALL PASSED", flush=True)
sys.exit(1 if FAILS else 0)
