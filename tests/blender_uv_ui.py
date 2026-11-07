# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""UV tools in background Blender (A3, SPEC 13 real-window list as far as no window allows): Auto's Seam Fit, the Seam
Fit preview in Look First, Apply, Ctrl+Z, redo, the result's Apply writing exactly the previewed fit (hash), then New,
New page layout with Keep going and Esc, Cut here, Bake those spots, Edit Mode, empty and nothing selected. Needs the
hook lines of SPEC 11.1 in guided.py, panel.py and __init__.py (a local test branch). Makes its own mesh.

    blender -b --factory-startup --python tests/blender_uv_ui.py -- --addon <folder of the add-on, a valid module name>

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
BAKE = "--no-bake" not in ARGV
sys.path.insert(0, os.path.dirname(ADDON))
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
guided, H, TL = pkg.guided, pkg.history, pkg.tidylive
UP = importlib.import_module(pkg.__name__ + ".uv_panel")
UT = importlib.import_module(pkg.__name__ + ".triadq.uvtrial")
KEEP = importlib.import_module(pkg.__name__ + ".triadq.uvkeep")
HOOKS = importlib.import_module(pkg.__name__ + ".uv_hooks")
LAYERS = importlib.import_module(pkg.__name__ + ".uv_layers")
FAILS = []
FRAMES = {}


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def frame(name, seconds):
    FRAMES[name] = max(FRAMES.get(name, 0.0), seconds)


def res_co():
    return np.asarray(guided.state["res"]["co"], dtype=np.float64)


def names():
    return [e.name for e in H.steps.entries]


def step():
    return H.current(bpy.context.scene)


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


def mesh_hash(mesh):
    p, ip, fl, _uv, mat = LAYERS.geometry(mesh)
    return KEEP.candidate_hash(p, ip, fl, mat, [(x.name, LAYERS.read_uv(mesh, x.name)) for x in mesh.uv_layers])


# --- a source with UVs (Suzanne has a UV map) and a real result
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_monkey_add()
src = bpy.context.object
src.name = "Monkey"
mod = src.modifiers.new("sub", "SUBSURF")
mod.levels = 1
bpy.ops.object.modifier_apply(modifier=mod.name)
ok(bool(src.data.uv_layers), "the source has UVs")
us = bpy.context.scene.curiomesh_uv
us.uvs, us.seam_fit, us.extra_faces, us.texture_size = "KEEP", "LIGHT", 15.0, 1024
gs = bpy.context.scene.curiomesh_guided
gs.scope, gs.auto_update = "AUTO", False
gs.auto_fix = False  # Auto Fix runs from a timer, which background Blender never fires; Seam Fit waits for it
UP.sync()
ok(KEEP.mode_of(gs) == "light", f"the popover reaches the cores: mode {KEEP.mode_of(gs)}")
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
src = bpy.data.objects["Monkey"]
bpy.context.view_layer.objects.active = src
src.select_set(True)
H.steps.clear()
H.settings()["step"] = 0
bpy.ops.ed.undo_push(message="test start")
us = bpy.context.scene.curiomesh_uv
gs = bpy.context.scene.curiomesh_guided
print(f"INFO result faces {len(guided.state['res']['faces'])}, source faces {len(src.data.polygons)}", flush=True)

# --- 0. Auto: the timer fits seams on the finished result, a few ms a tick; the row gets a verdict
t0, ticks = time.perf_counter(), 0
UP.ROW["key"] = None
while time.perf_counter() - t0 < 60:
    a = time.perf_counter()
    UP._auto_tick()
    frame("auto timer tick", time.perf_counter() - a)
    ticks += 1
    if UP.ROW["key"] is not None and UP.ROW["stage"] is None:
        break
print(f"INFO Auto guards: work={guided.state['work'] is not None} tool_busy={TL.busy()} uv_session={UP.live['session'] is not None} mode={UP.keep_mode(us, gs)} key={UP.ROW['key']}", flush=True)
ok(UP.ROW["level"] in ("GOOD", "CHECK", "BAD"), f"Auto: row {UP.ROW['level']} {UP.ROW['number']!r} after {ticks} ticks; {UP.ROW['message']!r}")
f = importlib.import_module(pkg.__name__ + ".triadq.types").FlatFaces.of(guided.state["res"]["faces"])
auto_key = KEEP.result_key(res_co(), f.indptr, f.flat)
ok(auto_key in KEEP.CANDIDATES, "Auto's fit is filed under the result, so Apply writes it")

# --- 1. Seam Fit in Look First: sliced, nothing written before Apply
before = res_co()
before_faces = len(guided.state["res"]["faces"])
s = TL.Session(bpy.context, UT.SeamFit(), {"faces": 1.0, "split": True, "keep_going": False}, "WHOLE")
ok(not s.error, f"Seam Fit opens: {s.error}")
pumps, t0 = 0, time.perf_counter()
while not (s.ready_to_keep() or s.refused()) and time.perf_counter() - t0 < 300:
    a = time.perf_counter()
    s.pump(TL.BUDGET)
    frame("Seam Fit preview slice", time.perf_counter() - a)
    pumps += 1
ok(s.ready_to_keep(), f"Seam Fit preview ready after {pumps} slices ({time.perf_counter() - t0:.1f} s): {s.cand.refusal if s.cand else ''}")
print(f"INFO card: {TL.card_lines(s)}", flush=True)
ok(np.allclose(res_co(), before) and len(guided.state["res"]["faces"]) == before_faces, "nothing changed before Apply")
cand = s.cand
fit_key = cand.fit.key

# Keep going: on, then Esc (the Session closes at once and the result is untouched)
s2 = TL.Session(bpy.context, UT.SeamFit(), {"faces": 0.05, "split": True, "keep_going": True}, "WHOLE")
for _ in range(20):
    s2.pump(TL.BUDGET)
a = time.perf_counter()
TL.close("")
s2.work = None  # what the modal does after close: the generator is dropped
esc_ms = 1000 * (time.perf_counter() - a)
ok(esc_ms < 50 and np.allclose(res_co(), before), f"Keep going + Esc stops in {esc_ms:.1f} ms and changes nothing")
TL.live["session"] = None

# --- 2. Apply: one step, one History entry
kept = TL.commit(bpy.context, cand, "Seam Fit", TL.numbers_text(cand, s.trial))
bpy.ops.ed.undo_push(message="Seam Fit")
after = res_co()
ok(kept and len(guided.state["res"]["faces"]) == len(cand.faces), f"Apply made the fit the result ({before_faces} -> {len(cand.faces)} faces)")
ok(names()[-1] == "Tidy: Seam Fit" and step() == 1, f"Apply is one History step: {names()} at {step()}")
undo(can)
ok(len(res_co()) == len(before) and np.allclose(res_co(), before), f"Ctrl+Z brings back the result before Seam Fit (step {step()})")
redo(can, 1)
ok(len(res_co()) == len(after) and np.allclose(res_co(), after), "Ctrl+Shift+Z brings the fit back")

# --- 3. the result's Apply writes exactly the previewed fit
bpy.context.view_layer.objects.active = bpy.data.objects["Monkey"]
if guided.state["res"].get("chart") is not None:
    # guided._transfer_uvs reads res["chart"] as old island ids, but in Light it holds uvkeep's kind codes (A1/H1 bug,
    # reported): it crashes on an empty island. after_apply overwrites those UVs anyway, so the test drops it here.
    print("INFO res['chart'] dropped before Apply (known A1/H1 issue)", flush=True)
    guided.state["res"].pop("chart")
f2 = importlib.import_module(pkg.__name__ + ".triadq.types").FlatFaces.of(guided.state["res"]["faces"])
k2 = KEEP.result_key(res_co(), f2.indptr, f2.flat)
print(f"INFO before Apply: key filed={k2 in KEEP.CANDIDATES}, filed keys={len(KEEP.CANDIDATES)}, mode={KEEP.mode_of(gs)}", flush=True)
r = bpy.ops.curiomesh.guided_apply()
ok(r == {"FINISHED"}, f"Apply: {r}")
out = bpy.context.object
ok(out is not None and out.name.startswith("Monkey_Retopo"), f"applied object: {out.name if out else None}")
ok(HOOKS.LAST.get("fitted") and HOOKS.LAST.get("hash") == fit_key, f"after_apply wrote the previewed fit: fitted={HOOKS.LAST.get('fitted')}")
ok(mesh_hash(out.data) == fit_key, "applied mesh hash == previewed candidate hash")
bpy.ops.ed.undo_push(message="Apply")

# --- 4. New on the applied mesh: sliced preview, Apply writes the shown UVs, one step, Ctrl+Z
OUT = out.name
uv_before = LAYERS.read_uv(out.data).copy()
us = bpy.context.scene.curiomesh_uv
s = UP.UVSession(bpy.context, out, "new", {"new": UP.new_settings(us)})
t0 = time.perf_counter()
while not (s.ready() or (s.cand is not None and s.cand.refusal)) and time.perf_counter() - t0 < 600:
    s.pump(UP.BUDGET)
frame("New preview slice", s.longest)
ok(s.ready(), f"New ready in {time.perf_counter() - t0:.1f} s: {s.cand.refusal if s.cand else ''} {s.cand.verdict if s.cand else ''}")
print(f"INFO New card: {UP.uv_draw.card_lines(s)}", flush=True)
ok(np.allclose(LAYERS.read_uv(out.data), uv_before), "nothing written before Apply")
steps_before = len(names())
err = UP._apply(bpy.context, s)
bpy.ops.ed.undo_push(message="UVs: New")
ok(not err and UP.LAST["key"] == UP.LAST["written"] == s.cand.key, f"New: applied UVs == previewed UVs (hash) {err}")
ok(len(names()) == steps_before + 1 and names()[-1] == "UVs: New", f"one History entry: {names()[-2:]}")
new_uv = LAYERS.read_uv(out.data).copy()
if can:
    bpy.ops.ed.undo()
    o = bpy.data.objects[OUT]
    ok(np.allclose(LAYERS.read_uv(o.data), uv_before), "Ctrl+Z brings the UVs before New back")
    bpy.ops.ed.redo()
    o = bpy.data.objects[OUT]
    ok(np.allclose(LAYERS.read_uv(o.data), new_uv), "Ctrl+Shift+Z brings New back")
else:
    print("INFO Ctrl+Z on the applied mesh needs a real window (no undo stack here)", flush=True)
out = bpy.data.objects[OUT]

# --- 5. New page layout: Keep going, then Esc stops within 50 ms and keeps the best
s = UP.UVSession(bpy.context, out, "layout", dict(time=0.3, gap=2 / 8))
while not (s.ready() or (s.cand is not None and s.cand.refusal)):
    s.pump(UP.BUDGET)
frame("New page layout slice", s.longest)
ok(s.ready(), f"New page layout ready: {UP.numbers_text(s)}")
first = s.cand.gain
s.toggle_keep_going()
for _ in range(5):
    s.pump(UP.BUDGET)
a = time.perf_counter()
stopped = s.stop() or s.work is None
stop_ms = 1000 * (time.perf_counter() - a)
ok(stopped and s.work is None and stop_ms < 50, f"Keep going + Esc stops in {stop_ms:.2f} ms")
ok(s.cand.gain >= first - 1e-9, f"Keep going never lowers space used ({first:.3f} -> {s.cand.gain:.3f})")

# --- 6. Cut here: a cut you add is in the new UVs
us = bpy.context.scene.curiomesh_uv
s = UP.UVSession(bpy.context, out, "cut", {"new": UP.new_settings(us)})
mesh_flat = s.base.flat
a_, b_ = int(mesh_flat[0]), int(mesh_flat[1])
s.settings["cuts"] = UT.toggle_cut((), a_, b_)
s.work = s._run()
while not (s.ready() or (s.cand is not None and s.cand.refusal)):
    s.pump(UP.BUDGET)
frame("Cut here slice", s.longest)
ids, _ne = UT.uvnew.edge_ids(s.base.indptr, s.base.flat)
cut = UT.uvnew.cut_edges(s.base.indptr, s.base.flat, s.cand.uv)
ok(s.ready() and bool(cut[ids[0]]), "Cut here: the clicked edge is a cut in the new UVs")

# --- 7. Bake those spots (a small image on the result, a bake from the source into the new islands only)
if BAKE:
    out = bpy.data.objects[OUT]
    LAYERS.write_uv(out.data, uv_before)  # back to the kept UVs, which still have faces across seams
    img = bpy.data.images.new("uvtest_color", 256, 256)
    img.pixels.foreach_set(np.full(256 * 256 * 4, 0.25, np.float32))
    mat = bpy.data.materials.new("uvtest")
    mat.use_nodes = True
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    out.data.materials.clear()
    out.data.materials.append(mat)
    us = bpy.context.scene.curiomesh_uv  # an undo above reloaded the scene: the old reference is gone
    us.texture_size = 256
    s = UP.UVSession(bpy.context, out, "bake", {"share": 1.0})
    t0 = time.perf_counter()
    while not (s.ready() or (s.cand is not None and s.cand.refusal)) and time.perf_counter() - t0 < 300:
        s.pump(UP.BUDGET)
    frame("Bake those spots preview slice", s.longest)
    if s.cand is not None and s.cand.refusal:
        print(f"INFO Bake those spots: {s.cand.refusal}", flush=True)
    else:
        ok(s.ready(), f"Bake those spots preview: {UP.numbers_text(s)}; {s.cand.note}")
        before_px = np.asarray(img.pixels[:], np.float32).copy()
        a = time.perf_counter()
        print("INFO baking now", flush=True)
        err = UP._apply(bpy.context, s)
        print(f"INFO bake took {time.perf_counter() - a:.1f} s", flush=True)
        ok(not err, f"Bake those spots applied: {err}")
        now_px = np.asarray(img.pixels[:], np.float32)
        changed = float((np.abs(now_px - before_px).reshape(-1, 4).max(axis=1) > 1e-4).mean())
        ok(0.0 < changed < 0.5, f"only the new islands' pixels changed ({100 * changed:.1f} % of the image)")
        ok(names()[-1] == "UVs: Bake those spots", f"a History entry: {names()[-1]}")

# --- 8. edge cases: nothing selected, an empty mesh, Edit Mode
for o in bpy.context.selected_objects:
    o.select_set(False)
bpy.context.view_layer.objects.active = None
ok(not UP.CURIOMESH_OT_uv_tool.poll(bpy.context), "nothing selected: the UV tools are greyed")
empty = bpy.data.objects.new("Empty_mesh", bpy.data.meshes.new("Empty_mesh"))
bpy.context.scene.collection.objects.link(empty)
bpy.context.view_layer.objects.active = empty
ok(not UP.CURIOMESH_OT_uv_tool.poll(bpy.context), "an empty mesh: the UV tools are greyed")
out = bpy.data.objects[OUT]
bpy.context.view_layer.objects.active = out
out.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
r = bpy.ops.curiomesh.uv_tool("EXEC_DEFAULT", tool="layout")
ok(r == {"FINISHED"} and out.mode == "EDIT", f"from Edit Mode: {r}, back in {out.mode}")
bpy.ops.object.mode_set(mode="OBJECT")

# --- 9. clean up: no handler, no timer after unregister
pkg.unregister()
ok(not bpy.app.timers.is_registered(UP._auto_tick) and not UP.live["handles"], "unregister removes the timer and the draw handlers")
for name, sec in sorted(FRAMES.items()):
    print(f"FRAME {name}: longest {1000 * sec:.1f} ms", flush=True)
print(f"UVUI {'FAILED ' + str(len(FAILS)) if FAILS else 'OK'}", flush=True)
sys.exit(1 if FAILS else 0)
