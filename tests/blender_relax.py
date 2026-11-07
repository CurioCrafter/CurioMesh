# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""The four Relax tools in background Blender, the way a person uses them: open the preview, drag, flip a line kind,
Apply, Again, Ctrl+Z, redo; then nothing selected and a tiny mesh. Saves one before/after still per tool.

    blender -b --factory-startup --python tests/blender_relax.py -- --addon <add-on folder> --model <file.gltf> \
        [--tools source,keep,square,detail] [--stills <folder>] [--tag <name>]

With no --model it makes its own (a subdivided monkey). On the owner's PC start Blender only through
bench/rough/lowblender.sh. Exits 1 on any failure.
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
MODEL = arg("--model")
STILLS = arg("--stills")
TAG = arg("--tag", "model")
WANTED = arg("--tools", "source,keep,square,detail").split(",")
sys.path.insert(0, os.path.dirname(ADDON))
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
guided, H, TL = pkg.guided, pkg.history, pkg.tidylive
R = importlib.import_module(pkg.__name__ + ".triadq.relax")
TOOLS = {
    "source": ("tool_source_relax", R.SourceRelax),
    "keep": ("tool_keep_lines_relax", R.KeepLinesRelax),
    "square": ("tool_square_up", R.SquareUp),
    "detail": ("tool_relax_toward_detail", R.RelaxTowardDetail),
}
FAILS = []


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def res_co():
    return np.asarray(guided.state["res"]["co"], dtype=np.float64)


def step():
    return H.current(bpy.context.scene)


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


def pump(s, limit=600.0):
    t0, longest = time.perf_counter(), 0.0
    while not (s.ready_to_keep() or s.refused() or s.error) and time.perf_counter() - t0 < limit:
        a = time.perf_counter()
        s.pump(TL.BUDGET)
        longest = max(longest, time.perf_counter() - a)
    return time.perf_counter() - t0, longest


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


def still(name, before, after, faces):
    """Before (left) and after (right): the result shaded with its wire, rendered by Workbench."""
    if not STILLS:
        return
    os.makedirs(STILLS, exist_ok=True)
    scene = bpy.context.scene
    hidden = [o for o in scene.objects if not o.hide_render]
    for o in hidden:
        o.hide_render = True
    size = float(np.ptp(before, axis=0).max()) or 1.0
    made = []
    for k, co in enumerate((before, after)):
        mesh = bpy.data.meshes.new(f"still_{k}")
        mesh.from_pydata((co - before.mean(axis=0) + np.array([(k - 0.5) * 1.15 * size, 0.0, 0.0])).tolist(), [], [list(f) for f in faces])
        obj = bpy.data.objects.new(f"still_{k}", mesh)
        scene.collection.objects.link(obj)
        wire = obj.copy()
        wire.data = mesh
        scene.collection.objects.link(wire)
        m = wire.modifiers.new("w", "WIREFRAME")
        m.thickness = 0.0025 * size
        m.use_replace = True
        mat = bpy.data.materials.new("still_wire")
        mat.diffuse_color = (0.02, 0.05, 0.08, 1.0)
        wire.data = mesh.copy()
        wire.data.materials.append(mat)
        obj.data.materials.append(bpy.data.materials.new("still_body"))
        obj.data.materials[0].diffuse_color = (0.8, 0.78, 0.74, 1.0)
        made += [obj, wire]
    centre = np.zeros(3)
    cam_data = bpy.data.cameras.new("still_cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.4 * size
    cam = bpy.data.objects.new("still_cam", cam_data)
    scene.collection.objects.link(cam)
    direction = np.array([0.0, 1.0, 0.0])  # from the front (-Y), looking along +Y
    cam.location = (centre - 3.0 * size * direction).tolist()
    cam.rotation_mode = "QUATERNION"
    from mathutils import Vector

    cam.rotation_quaternion = Vector(direction.tolist()).to_track_quat("-Z", "Z")
    scene.camera = cam
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x, scene.render.resolution_y = 1600, 800
    scene.render.film_transparent = False
    scene.render.filepath = os.path.join(STILLS, name)
    scene.render.image_settings.file_format = "PNG"
    try:
        bpy.ops.render.render(write_still=True)
        print(f"INFO still {scene.render.filepath}", flush=True)
    except RuntimeError as exc:
        print(f"INFO no still: {exc}", flush=True)
    for o in made + [cam]:
        bpy.data.objects.remove(o, do_unlink=True)
    for o in hidden:
        o.hide_render = False


# --- the source
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
if MODEL:
    bpy.ops.import_scene.gltf(filepath=MODEL)
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
print(f"INFO source {len(src.data.polygons)} faces", flush=True)
t0 = time.perf_counter()
r = start(src)
ok(guided.state["res"] is not None, f"a result ({time.perf_counter() - t0:.0f} s): {r} {guided.state['message']}")
res = guided.state["res"]
print(f"INFO result {len(res['faces'])} faces", flush=True)
if STILLS:
    os.makedirs(STILLS, exist_ok=True)
    flat = [list(f) for f in res["faces"]]
    np.savez(os.path.join(STILLS, f"result_{TAG}.npz"), co=np.asarray(res["co"]), sizes=np.array([len(f) for f in flat]), corners=np.array([v for f in flat for v in f]))

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
print(f"INFO Blender undo in this run: {can}", flush=True)
H.settings().tool_scope = "WHOLE"
bpy.ops.ed.undo_push(message="test start")

# --- each tool
for key in WANTED:
    idname, trial = TOOLS[key]
    name = trial.name
    before = res_co()
    faces = [tuple(f) for f in guided.state["res"]["faces"]]
    s = TL.Session(bpy.context, trial(), dict(trial.defaults), "WHOLE")
    ok(not s.error, f"{name}: opens ({s.error})")
    took, longest = pump(s)
    if s.refused():
        print(f"INFO {name}: refused here: {s.cand.refusal}", flush=True)
        TL.close("")
        continue
    ok(s.ready_to_keep(), f"{name}: preview ready in {took:.1f} s, longest slice {1000 * longest:.0f} ms")
    print(f"INFO {name} card: {TL.card_lines(s)}", flush=True)
    if trial.toggles:
        first = s.cand
        s.toggle(trial.toggles[1][0])
        s.moved_at -= 1.0
        pump(s)
        ok(s.ready_to_keep() and s.cand is not first and not s.settings[trial.toggles[1][0]], f"{name}: a line-kind chip re-runs the preview")
        s.toggle(trial.toggles[1][0])
    a = time.perf_counter()
    s.set(0.9)
    ms = 1000 * (time.perf_counter() - a)
    ok(s.shown is not None and ms < 100, f"{name}: dragging the ring answers at once ({ms:.1f} ms, blended {not s.shown.exact})")
    s.moved_at -= 1.0
    took, _ = pump(s)
    ok(s.ready_to_keep(), f"{name}: exact run at 0.9 in {took:.1f} s")
    ok(np.allclose(res_co(), before), f"{name}: nothing changed before Apply")
    try:
        layers = s._batches(bpy.data.objects[guided.PREVIEW_NAME].matrix_world)
        print(f"INFO {name} layers: {sorted(layers)}", flush=True)
        if trial.toggles:
            ok("kept" in layers or not len(s.cand.lines), f"{name}: kept lines drawn")
        del layers  # GPU batches freed now, not at Python exit after the GPU is gone (that crashes background Blender)
    except Exception as exc:
        print(f"INFO {name}: no GPU here, layers not built: {exc!r}", flush=True)
    print(f"INFO {name} numbers: {TL.numbers_text(s.cand, s.trial, s.cap)}", flush=True)
    free = s.base.free.copy()
    kept = TL.commit(bpy.context, s.cand, name, TL.numbers_text(s.cand, s.trial, s.cap))
    bpy.ops.ed.undo_push(message=name)
    after = res_co()
    one = step()
    ok(kept and not np.allclose(after, before), f"{name}: Apply changed the result")
    if trial is not R.KeepLinesRelax:
        ok(np.allclose(after[~free], before[~free]), f"{name}: held corners stayed")
    still(f"relax_{key}_{TAG}.png", before, after, faces)
    r = getattr(bpy.ops.curiomesh, idname)("EXEC_DEFAULT")
    again = res_co()
    ok(r == {"FINISHED"} and step() == one + 1, f"{name}: Again from the result is one more step ({r}, step {step()})")
    bpy.ops.ed.undo_push(message=name + " again")
    undo(can)
    ok(np.allclose(res_co(), after), f"{name}: Ctrl+Z goes back to the first Apply")
    undo(can)
    ok(np.allclose(res_co(), before), f"{name}: Ctrl+Z again goes back to before the tool")
    redo(can, one)
    ok(np.allclose(res_co(), after), f"{name}: Ctrl+Shift+Z brings the Apply back")
    redo(can, one + 1)
    ok(np.allclose(res_co(), again), f"{name}: and the Again")

# --- the first-run button: Source Relax on the roughest part
if "source" in WANTED:
    seg = importlib.import_module(pkg.__name__ + ".segments").seg
    before, was = res_co(), step()
    r = bpy.ops.curiomesh.tool_source_relax("EXEC_DEFAULT", first=True)
    part = seg.get("sel")
    ok(r == {"FINISHED"} and step() == was + 1 and bool(part), f"Even out the roughest patch: one step on part {part} ({r})")
    if part and seg.get("piece") is not None:
        outside = np.ones(len(before), dtype=bool)
        for f, p in zip(guided.state["res"]["faces"], seg["piece"]):
            if int(p) in part:
                outside[list(f)] = False
        ok(np.allclose(res_co()[outside], before[outside]), "Even out the roughest patch: nothing outside that part moved")
    H.settings().tool_scope = "WHOLE"

# --- nothing selected, then a tiny mesh
print("INFO nothing selected", flush=True)
H.settings().tool_scope = "SELECTION"  # fetched again: an undo reloads the scene, so older handles to it are gone
preview = bpy.data.objects[guided.PREVIEW_NAME]
preview.data.polygons.foreach_set("select", [False] * len(preview.data.polygons))
for key in WANTED:
    idname, trial = TOOLS[key]
    s = TL.Session(bpy.context, trial(), dict(trial.defaults), "SELECTION")
    ok(s.error.startswith("Nothing selected"), f"{trial.name}: nothing selected says so ({s.error})")
    ok(not getattr(bpy.ops.curiomesh, idname).poll(), f"{trial.name}: button greyed with nothing selected")
H.settings().tool_scope = "WHOLE"
print("INFO tiny mesh", flush=True)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_cube_add()
tiny = bpy.context.object
tiny.name = "Tiny"
r = start(tiny)
print(f"INFO tiny result: {guided.state['res'] is not None} ({guided.state['message']})", flush=True)
for key in WANTED:
    idname, trial = TOOLS[key]
    try:
        r = getattr(bpy.ops.curiomesh, idname)("EXEC_DEFAULT")
        ok(True, f"{trial.name}: tiny mesh, no crash ({r})")
    except RuntimeError as exc:
        ok("poll" in str(exc) or "context" in str(exc), f"{trial.name}: tiny mesh refused cleanly ({exc})")
TL.close("")
ok(not TL.busy() and not TL.live["handles"], "no preview and no draw handler left")
pkg.unregister()
print(f"RELAX {'FAILED ' + str(len(FAILS)) if FAILS else 'OK'}", flush=True)
sys.exit(1 if FAILS else 0)
