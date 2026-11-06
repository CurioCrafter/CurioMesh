# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Look First and History in background Blender: a tool preview, a handle drag, Apply, Ctrl+Z, redo, a History jump,
and Tidy Up passes going back and forth. Makes its own test mesh (a subdivided monkey), so it needs no models.

    blender -b --factory-startup --python tests/blender_lookfirst.py -- --addon <folder of the add-on, a valid module name>

On the owner's PC start Blender only through bench/rough/lowblender.sh. Exits 1 on any failure.
"""

import importlib
import os
import sys
import time

import bpy
import numpy as np

ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ADDON = os.path.abspath(ARGV[ARGV.index("--addon") + 1]) if "--addon" in ARGV else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ADDON))
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
guided, H, TL, organize = pkg.guided, pkg.history, pkg.tidylive, pkg.organize
from importlib import import_module  # noqa: E402

TRIAL = import_module(pkg.__name__ + ".triadq.tidytools")
FAILS = []


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def ctx():
    return bpy.context


def res_co():
    return np.asarray(guided.state["res"]["co"], dtype=np.float64)


def names():
    return [e.name for e in H.steps.entries]


def step():
    return H.current(bpy.context.scene)


def undo(can):
    """Ctrl+Z: Blender's own when this Blender keeps an undo stack, else the same path by hand (the handler)."""
    if can:
        bpy.ops.ed.undo()
    else:
        ts = H.settings()
        ts["step"] = H.steps.entries[H.steps.index_of(step()) - 1].step
        H._curiomesh_history_undo(bpy.context.scene)


def redo(can, to):
    if can:
        bpy.ops.ed.redo()
    else:
        H.settings()["step"] = to
        H._curiomesh_history_undo(bpy.context.scene)


# --- a source mesh and a real result
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
r = bpy.ops.curiomesh.guided_auto_start()
ok(r == {"FINISHED"}, f"Start: {r}")
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
ok("New result" in names(), f"a new result is a History step: {names()}")

# can this Blender undo? (the feature tour's probe)
bpy.ops.ed.undo_push(message="probe")
bpy.ops.mesh.primitive_cube_add()
probe = bpy.context.object.name
can = True
try:
    bpy.ops.ed.undo()
    can = probe not in bpy.data.objects
except RuntimeError:
    can = False
print(f"INFO Blender undo in this run: {can}", flush=True)
for o in list(bpy.data.objects):
    if o.name.startswith("Cube"):
        bpy.data.objects.remove(o, do_unlink=True)
bpy.context.view_layer.objects.active = src
src.select_set(True)
H.steps.clear()
H.settings()["step"] = 0
bpy.ops.ed.undo_push(message="test start")

# --- 1. a tool preview: sliced, instant drag, nothing written before Apply
before = res_co()
preview = bpy.data.objects[guided.PREVIEW_NAME]
mesh_before = np.empty(len(preview.data.vertices) * 3)
preview.data.vertices.foreach_get("co", mesh_before)
s = TL.Session(bpy.context, TRIAL.SmoothCorners(), {"strength": 0.6}, "WHOLE")
ok(not s.error, f"session opens: {s.error}")
pumps, longest = 0, 0.0
t0 = time.perf_counter()
while not s.ready_to_keep() and time.perf_counter() - t0 < 120:
    a = time.perf_counter()
    s.pump(TL.BUDGET)
    longest = max(longest, time.perf_counter() - a)
    pumps += 1
ok(s.ready_to_keep(), f"preview ready after {pumps} slices ({time.perf_counter() - t0:.2f} s, longest slice {1000 * longest:.0f} ms)")
ok(pumps > 3, "the preview is worked out in slices")
print(f"INFO card: {TL.card_lines(s)}", flush=True)
a = time.perf_counter()
s.set(0.9)  # the handle dragged
drag_ms = 1000 * (time.perf_counter() - a)
ok(s.shown is not None and not s.shown.exact and drag_ms < 50, f"dragging shows a blend at once ({drag_ms:.1f} ms)")
s.moved_at -= 1.0  # the mouse rested
t0 = time.perf_counter()
while not s.ready_to_keep() and time.perf_counter() - t0 < 120:
    s.pump(TL.BUDGET)
ok(s.ready_to_keep() and abs(s.cand.strength - 0.9) < 1e-9, "the exact run at the new strength follows")
mesh_now = np.empty_like(mesh_before)
bpy.data.objects[guided.PREVIEW_NAME].data.vertices.foreach_get("co", mesh_now)
ok(np.allclose(mesh_now, mesh_before) and np.allclose(res_co(), before), "nothing changed before Apply")
try:
    import gpu  # noqa: F401

    layers = s._batches(preview.matrix_world)
    ok("ghost" in layers and "wash" in layers, f"preview layers built: {sorted(layers)}")
except Exception as exc:
    print(f"INFO no GPU in this run, layers not built: {exc!r}", flush=True)

# --- 2. Apply: one step
moved_free = s.base.free.copy()
kept = TL.commit(bpy.context, s.cand, s.trial.name, TL.numbers_text(s.cand, s.trial))
bpy.ops.ed.undo_push(message="Smooth Corners")  # what the operator's UNDO flag does in a window
after = res_co()
ok(kept and not np.allclose(after, before), "Apply changed the result")
ok(np.allclose(after[~moved_free], before[~moved_free]), "held corners stayed")
ok(names()[-1] == "Tidy: Smooth Corners" and step() == 1, f"Apply is one History step: {names()} at {step()}")

# --- 3. Ctrl+Z and Ctrl+Shift+Z
undo(can)
ok(np.allclose(res_co(), before), f"Ctrl+Z brings back the result before the tool (step {step()})")
redo(can, 1)
ok(np.allclose(res_co(), after), f"Ctrl+Shift+Z brings the tool's result back (step {step()})")

# --- 4. the F9 / script path, then a History jump back to the start
r = bpy.ops.curiomesh.tool_smooth_corners("EXEC_DEFAULT", strength=0.3)
ok(r == {"FINISHED"}, f"Smooth Corners run inline: {r}")
third = res_co()
ok(not np.allclose(third, after) and step() == 2, f"a second step: {names()}")
first = H.steps.index_of(0)
ok(first >= 0 and H.jump(bpy.context, first), "History jump to the start")
ok(np.allclose(res_co(), before) and names()[-1].startswith("Back to:") and step() == 3, f"the jump is a step of its own: {names()}")
undo(can)
ok(np.allclose(res_co(), third), "Ctrl+Z takes the jump back")

# --- 5. Tidy Up: the slider, Again and Undo Pass go back and forth
gs.organize = 0.5
t0 = time.time()
while not organize.ready() and time.time() - t0 < 300:
    organize._tick()
    time.sleep(0.05)
organize._tick()
ok(organize.ready() and organize.org["shown"] > 0.0, f"Tidy Up shows strength {organize.org['shown']}")
ok(names()[-1] == "Tidy Up", f"the slider is a step: {names()}")
bpy.ops.ed.undo_push(message="Tidy Up")  # Blender's own push when the slider lets go
tidied, tidied_step = res_co(), step()
r = bpy.ops.curiomesh.tidy_again()
ok(r == {"FINISHED"} and names()[-1] == "Tidy: Again" and len(organize.org["passes"]) == 1, f"Tidy Again is a step: {r} {names()}")
t0 = time.time()
while not organize.ready() and time.time() - t0 < 300:
    organize._tick()
    time.sleep(0.05)
r = bpy.ops.curiomesh.tidy_back()
ok(r == {"FINISHED"} and names()[-1] == "Tidy: Undo Pass" and not organize.org["passes"], f"Undo Pass is a step: {r} {names()}")
undo(can)
ok(len(organize.org["passes"]) == 1, f"Ctrl+Z brings the pass back ({len(organize.org['passes'])} pass kept)")
undo(can)
ok(not organize.org["passes"] and np.allclose(res_co(), tidied), f"and again goes back to before Tidy Again (step {step()}, wanted {tidied_step})")

# --- 6. clean up: no handler, no session; registering twice is fine
TL.close("")
ok(not TL.busy() and not TL.live["handles"], "no preview and no draw handler left")
undo_handlers = [h for h in bpy.app.handlers.undo_post if h.__name__ == "_curiomesh_history_undo"]
pkg.register()
again = [h for h in bpy.app.handlers.undo_post if h.__name__ == "_curiomesh_history_undo"]
ok(len(undo_handlers) == 1 and len(again) == 1, f"one undo handler, also after registering twice ({len(undo_handlers)}, {len(again)})")
pkg.unregister()
ok(not [h for h in bpy.app.handlers.undo_post if h.__name__ == "_curiomesh_history_undo"], "unregister removes the undo handler")
print(f"LOOKFIRST {'FAILED ' + str(len(FAILS)) if FAILS else 'OK'}", flush=True)
sys.exit(1 if FAILS else 0)
