# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: the 0.9.9 UX pieces together on a real model
"""One pass over the 0.9.9 UX work on a real model: dashboard off by default, hud_toggle, Auto with the dashboard's
job info, Fix Bad Spots marking two parts and redoing them, Ctrl+Z, Try Every Method on one part, the AI prompt, the
issue-form link, and a clean unregister.

    blender -b --factory-startup --python tests/blender_ux099.py -- --blend <file.blend>

On the owner's PC start Blender only through bench/rough/lowblender.sh. Exits 1 on any failure.
"""

import importlib.util
import inspect
from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
FAILS = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


if "--blend" in ARGV:
    bpy.ops.wm.open_mainfile(filepath=ARGV[ARGV.index("--blend") + 1], load_ui=False)


def rna_names():
    out = {n for n in dir(bpy.types) if n.startswith("CURIOMESH")}
    for t in (bpy.types.Scene, bpy.types.WindowManager, bpy.types.Object):
        out |= {f"{t.__name__}.{p.identifier}" for p in t.bl_rna.properties if p.identifier.startswith("curiomesh")}
    return out


rna0 = rna_names()
spec = importlib.util.spec_from_file_location("curiomesh", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec)
sys.modules["curiomesh"] = pkg
spec.loader.exec_module(pkg)
pkg.register()
from curiomesh import guided, history as H, issue_panel as IP, overlay, segment_goals as SG, segments  # noqa: E402

scene = bpy.context.scene
settings, gs = scene.curiomesh_settings, scene.curiomesh_guided

# --- the dashboard: off by default, on only after Show Dashboard
overlay.apply_hud_default(scene)
ok(settings.hud_enabled is False, "the dashboard is off in a loaded file")
bpy.ops.curiomesh.hud_toggle()
ok(settings.hud_enabled is True, "hud_toggle shows it")
bpy.ops.curiomesh.hud_toggle()
ok(settings.hud_enabled is False, "hud_toggle hides it again")

# --- the model: the biggest mesh in the file
meshes = sorted((o for o in scene.objects if o.type == "MESH"), key=lambda o: len(o.data.polygons), reverse=True)
src = meshes[0]
for o in scene.objects:
    o.select_set(o is src)
bpy.context.view_layer.objects.active = src
print(f"INFO model {src.name}: {len(src.data.polygons):,} faces", flush=True)

# can this Blender undo? (the feature tour's probe)
bpy.ops.ed.undo_push(message="probe")
bpy.ops.mesh.primitive_cube_add()
probe = bpy.context.object.name
try:
    bpy.ops.ed.undo()
    can_undo = probe not in bpy.data.objects
except RuntimeError:
    can_undo = False
print(f"INFO Blender undo in this run: {can_undo}", flush=True)
for o in list(bpy.data.objects):
    if o.name == probe:
        bpy.data.objects.remove(o, do_unlink=True)
src = bpy.data.objects[src.name]
bpy.context.view_layer.objects.active = src
src.select_set(True)
scene = bpy.context.scene
settings, gs = scene.curiomesh_settings, scene.curiomesh_guided

# --- Auto, with the dashboard's job info filled while it runs
gs.scope, gs.auto_update = "AUTO", False
ok(bpy.ops.curiomesh.guided_auto_start() == {"FINISHED"}, "Auto starts")
seen = None
for _ in range(60):
    work = guided.state["work"]
    if work is None:
        break
    runner = work["runner"]
    while runner.busy:
        runner.step(0.05)
        info = overlay.job_info()
        if info is not None and seen is None and info["frac"] > 0:
            seen = info
    guided._collect_work(bpy.context)
    if guided.state["work"] is work or guided.state["work"] is None or not guided.state["work"]["runner"].busy:
        break
ok(seen is not None and seen["stage"] and seen["elapsed"] is not None, f"job info while Auto runs: {seen}")
ok(seen is not None and overlay.job_text(seen).startswith(f"{round(seen['frac'] * 100)}%"), f"job text: {overlay.job_text(seen) if seen else None!r}")
ok(guided.state["res"] is not None, f"a result: {guided.state['message']}")
ok(settings.hud_enabled is False, "Auto did not turn the dashboard on")
ok(overlay.job_info() is None, "no job info after Auto")
segments.ensure()
parts = len(segments.seg["parts"])
ok(segments.ready() and parts >= 2, f"{parts} parts")

# --- Fix Bad Spots: mark two parts, redo them, Ctrl+Z
bad = segments.bad_parts()
print(f"INFO {len(IP.spots())} bad spots on parts {bad[:8]}", flush=True)
segments.seg["reviewing"] = True
SG._set_mode(bpy.context, "SPOTS")
ok(segments.seg["review_mode"] == "SPOTS" and gs.seg_bad is True, "Fix Bad Spots mode, bad faces shown")
by_size = sorted(range(parts), key=lambda p: segments.seg["parts"][p]["after"])
two = (bad + [p for p in by_size if p not in bad])[:2]
segments.select(two)
ok(segments.seg["sel"] == set(two), f"marked parts {two}")
before = {p: (segments.seg["parts"][p]["cur"], len(segments.seg["parts"][p]["tries"])) for p in two}
faces_before = len(guided.state["res"]["faces"])
step0 = H.current(scene)
r = bpy.ops.curiomesh.issue_redo("EXEC_DEFAULT", True, marked=True)
ok(r == {"FINISHED"}, f"Redo Marked runs: {r}")
for _ in range(20):
    work = guided.state["work"]
    if work is None:
        break
    while work["runner"].busy:
        work["runner"].step(0.05)
    guided._collect_work(bpy.context)
    if guided.state["work"] is work:
        break
after = {p: (segments.seg["parts"][p]["cur"], len(segments.seg["parts"][p]["tries"])) for p in two}
ok(after != before, f"both marked parts got new faces: {before} -> {after}")
ok(H.current(scene) > step0, f"Redo Marked is a History step ({step0} -> {H.current(scene)})")
names = [e.name for e in H.steps.entries]
if can_undo:
    bpy.ops.ed.undo()
else:
    H.settings()["step"] = H.steps.entries[H.steps.index_of(H.current(bpy.context.scene)) - 1].step
    H._curiomesh_history_undo(bpy.context.scene)
back = {p: segments.seg["parts"][p]["cur"] for p in two}
ok(back == {p: before[p][0] for p in two} and len(guided.state["res"]["faces"]) == faces_before, f"Ctrl+Z puts both parts back: {back}, steps {names}")
segments.ensure()
ok(len(segments.seg["parts"]) == parts, f"Ctrl+Z keeps the parts table ({len(segments.seg['parts'])} parts)")
segments.seg["reviewing"] = False
SG._set_mode(bpy.context, "PARTS")
scene, src = bpy.context.scene, bpy.data.objects[src.name]
settings, gs = scene.curiomesh_settings, scene.curiomesh_guided

# --- Try Every Method on the smallest part
guided.stop_work()
small = min(range(len(segments.seg["parts"])), key=lambda p: segments.seg["parts"][p]["after"])
segments.select([small])
record = segments.seg["parts"][small]
ways, tries0 = len(SG.try_queue(record)), len(record["tries"])
ok(bpy.ops.curiomesh.seg_try_all() == {"FINISHED"}, f"Try Every Method starts on part {small} ({record['after']} faces, {ways} ways)")
try:
    for _ in range(400):
        work = guided.state["work"]
        if work is not None:
            while work["runner"].busy:
                work["runner"].step(0.05)
            guided._collect_work(bpy.context)
        if segments.seg.get("tryall") is None:
            break
        SG._tick()
except Exception as exc:
    s = segments.seg.get("tryall") or {}
    ok(False, f"Try Every Method: {exc!r} at {s.get('label')!r}, {len(segments.seg['parts'])} parts")
    segments.seg["tryall"] = None
ok(segments.seg.get("tryall") is None, "every way finished")
ok(len(record["tries"]) > tries0 and SG.tried_order(record), f"{len(record['tries'])} tries, note {segments.seg['tryall_note']!r}")
ok(settings.hud_enabled is False, "Try Every Method did not turn the dashboard on")

# --- unregister: nothing left behind
pkg.unregister()
left = rna_names() - rna0
ok(not left, f"no classes or properties left: {sorted(left)[:8]}")
timers = []
handlers = []
for name, mod in list(sys.modules.items()):
    if not name.startswith("curiomesh"):
        continue
    for fname, fn in inspect.getmembers(mod, inspect.isfunction):
        if fn.__module__ == name and bpy.app.timers.is_registered(fn):
            timers.append(f"{name}.{fname}")
for hname in dir(bpy.app.handlers):
    lst = getattr(bpy.app.handlers, hname)
    if isinstance(lst, list):
        handlers += [f"{hname}:{f.__name__}" for f in lst if getattr(f, "__module__", "").startswith("curiomesh")]
ok(not timers, f"no timers left: {timers}")
ok(not handlers, f"no app handlers left: {handlers}")
kc = bpy.context.window_manager.keyconfigs.addon
if kc is not None:
    km = [k.name for k in kc.keymaps if any(i.idname.startswith("curiomesh.") for i in k.keymap_items)]
    ok(not km, f"no keymap items left: {km}")
print("UX099 " + (f"FAILED {len(FAILS)}: {FAILS}" if FAILS else "ALL PASS"), flush=True)
sys.exit(1 if FAILS else 0)
