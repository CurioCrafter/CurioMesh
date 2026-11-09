# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""The three long Apply UV steps in timer slices, on the barrier test mesh (run with CURIOMESH_TURBO=0 for the Python path):

1. Classic Guided Apply's transfer when no Seam Fit finished (uv_hooks.after_apply_steps)
2. the bake unwrap inside Classic Guided Apply (textures.bake_detail_steps, New UVs)
3. Classic Remesh's UV transfer (bridge.run_curiomesh_steps)

Each runs through its real operator. `sliced.finish` is swapped for a driver that does what the modal does on each
timer event (one `sliced.run_slice`), times every slice and, for the Stop runs, presses Stop 2 s in. Checks: the
longest slice after the first (the first also makes the result), UVs at the end, a valid mesh after Stop, and Ctrl+Z
then Ctrl+Shift+Z with the History step counter in step.

    blender -b --factory-startup --python tests/blender_uv_sliced.py -- [--addon <folder>]

On the owner's PC start Blender only through bench/rough/lowblender.sh. Exits 1 on any failure.
"""

import copy
import importlib
import os
import sys
import time

import bpy
import numpy as np

ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ADDON = os.path.abspath(ARGV[ARGV.index("--addon") + 1]) if "--addon" in ARGV else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ADDON))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "bench", "assets")))
import uvm  # noqa: E402

src = uvm.load_model("concrete_road_barrier")  # factory settings first, then the add-on
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
guided, H, SL = pkg.guided, pkg.history, pkg.sliced
UP = importlib.import_module(pkg.__name__ + ".uv_panel")
KEEP = importlib.import_module(pkg.__name__ + ".triadq.uvkeep")
BAKE = importlib.import_module(pkg.__name__ + ".triadq.uvbake")
HOOKS = importlib.import_module(pkg.__name__ + ".uv_hooks")
LAYERS = importlib.import_module(pkg.__name__ + ".uv_layers")
FAILS = []
REAL_FINISH = SL.finish
RUN = {"stop_after": None, "times": [], "stopped": False}


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def timed_finish(steps):
    """What the modal does, one timer event per slice, timed. Presses Stop once the UV slices passed `stop_after` s."""
    if SL.JOB is not None:
        return REAL_FINISH(steps)
    SL.JOB = SL.Job("test")
    times = []
    try:
        while True:
            a = time.perf_counter()
            done, value = SL.run_slice(steps)
            times.append(time.perf_counter() - a)
            if done:
                return value
            if RUN["stop_after"] is not None and sum(times[1:]) > RUN["stop_after"]:
                SL.JOB.stop()
    finally:
        RUN["times"], RUN["stopped"] = times, SL.JOB.stopped
        SL.JOB = None


SLOW = []  # steps over 80 ms: (class, ms, where it was, where it went, inner stage)


def where(gen):
    loc = []
    while gen is not None and getattr(gen, "gi_frame", None) is not None:
        loc.append(f"{gen.gi_frame.f_code.co_name}:{gen.gi_frame.f_lineno}")
        gen = gen.gi_yieldfrom
    return " > ".join(loc[-4:])


def watch(cls, gen_attr):
    orig = cls.step

    def step(self, *a):
        before = where(getattr(self, gen_attr, None))
        t = time.perf_counter()
        r = orig(self, *a)
        dt = time.perf_counter() - t
        if dt > 0.08:
            inner = getattr(self, "_now", None) or getattr(self, "_active", None)
            SLOW.append((cls.__name__, round(1000 * dt), before, where(getattr(self, gen_attr, None)), type(inner).__name__))
        return r

    cls.step = step


watch(KEEP.FitStage, "_steps")  # SLOW lines: which step of which stage ran past 80 ms
DUMP = os.environ.get("UVSLICED_DUMP")  # a file: the first transfer's inputs, for profiling outside Blender
if DUMP:
    _SEAMS = importlib.import_module(pkg.__name__ + ".triadq.uvseams")
    _init = _SEAMS.Transfer.__init__

    def _dump_init(self, *a, **k):
        if not os.path.exists(DUMP):
            np.savez(DUMP, *[np.asarray(x) for x in a[:7]], material=np.asarray(k.get("material") if k.get("material") is not None else []))
        _init(self, *a, **k)

    _SEAMS.Transfer.__init__ = _dump_init
    _NEW = importlib.import_module(pkg.__name__ + ".triadq.uvnew")
    _ninit = _NEW.NewUVs.__init__

    def _dump_new(self, points, indptr, flat, material=None, forced=None, *a, **k):
        out = DUMP.replace(".npz", "_new.npz")
        if not os.path.exists(out):
            np.savez(out, points=points, indptr=indptr, flat=flat, material=np.asarray([] if material is None else material), forced=np.asarray([] if forced is None else forced))
        _ninit(self, points, indptr, flat, material, forced, *a, **k)

    _NEW.NewUVs.__init__ = _dump_new
watch(importlib.import_module(pkg.__name__ + ".triadq.uvnew").NewUVs, "_work")
watch(importlib.import_module(pkg.__name__ + ".triadq.uvseams").Transfer, "_steps")
watch(importlib.import_module(pkg.__name__ + ".triadq.uvsight").Sight, "_gen")
watch(importlib.import_module(pkg.__name__ + ".triadq.uvflat").Flatten, "_work")
watch(importlib.import_module(pkg.__name__ + ".triadq.uvchart").Charts, "_work")


def report(label):
    for row in sorted(SLOW, key=lambda r: -r[1])[:6]:
        print(f"SLOW {label}: {row}", flush=True)
    SLOW.clear()
    t = RUN["times"]
    rest = max(t[1:-1]) if len(t) > 2 else 0.0
    print(f"TIME {label}: {len(t)} slices, first {1000 * t[0]:.0f} ms, last {1000 * t[-1]:.0f} ms, longest UV slice between {1000 * rest:.1f} ms, total {sum(t):.1f} s", flush=True)
    return rest


def mesh_hash(mesh):
    p, ip, fl, _uv, mat = LAYERS.geometry(mesh)
    return KEEP.candidate_hash(p, ip, fl, mat, [(x.name, LAYERS.read_uv(mesh, x.name)) for x in mesh.uv_layers])


def uv_ok(mesh):
    return bool(mesh.uv_layers) and len(mesh.polygons) > 0 and all(np.isfinite(LAYERS.read_uv(mesh, x.name)).all() for x in mesh.uv_layers)


def undo_redo(label, name):
    """Ctrl+Z removes the written object and keeps the History step; Ctrl+Shift+Z brings it back, same mesh and UVs."""
    if not CAN_UNDO:
        print(f"INFO {label}: no undo stack in this Blender run", flush=True)
        return
    global SHAPE
    m = bpy.data.objects[name].data
    SHAPE = (len(m.vertices), len(m.polygons), [x.name for x in m.uv_layers], [float(np.abs(LAYERS.read_uv(m, x.name)).sum()) for x in m.uv_layers])
    after, step_after, entries = mesh_hash(m), H.current(bpy.context.scene), len(H.steps.entries)
    bpy.ops.ed.undo()
    ok(name not in bpy.data.objects and H.current(bpy.context.scene) == STEP0, f"{label}: Ctrl+Z takes the result away (step {H.current(bpy.context.scene)})")
    print(f"INFO {label} after undo: objects {sorted(bpy.data.objects.keys())}", flush=True)
    bpy.ops.ed.redo()
    back = bpy.data.objects.get(name)
    print(f"INFO {label} after redo: objects {sorted(bpy.data.objects.keys())}", flush=True)
    if back is not None and mesh_hash(back.data) != after:
        print(f"INFO {label} after redo: {len(back.data.vertices)} verts, {len(back.data.polygons)} faces, layers {[x.name for x in back.data.uv_layers]} {[float(np.abs(LAYERS.read_uv(back.data, x.name)).sum()) for x in back.data.uv_layers]}, was {SHAPE}", flush=True)
    ok(back is not None and mesh_hash(back.data) == after, f"{label}: Ctrl+Shift+Z brings the same result and UVs back")
    ok(H.current(bpy.context.scene) == step_after and len(H.steps.entries) == entries, f"{label}: History step {H.current(bpy.context.scene)} == {step_after}")


# --- the result to apply: Auto on the barrier, UVs kept with Light Seam Fit, no Look (so no fit is finished)
us, gs = bpy.context.scene.curiomesh_uv, bpy.context.scene.curiomesh_guided
us.uvs, us.seam_fit = "KEEP", "LIGHT"
gs.scope, gs.auto_update, gs.auto_fix = "AUTO", False, False
UP.sync()
src.select_set(True)
bpy.context.view_layer.objects.active = src
t0 = time.perf_counter()
r = bpy.ops.curiomesh.guided_auto_start()
for _ in range(60):
    work = guided.state["work"]
    if work is None:
        break
    while work["runner"].busy:
        work["runner"].step(0.05)
    guided._collect_work(bpy.context)
    if guided.state["work"] is work or guided.state["work"] is None or not guided.state["work"]["runner"].busy:
        break
ok(guided.state["res"] is not None, f"a result in {time.perf_counter() - t0:.0f} s: {guided.state['message']}")
print(f"INFO turbo: {pkg.turbo.status_line() if hasattr(pkg.turbo, 'status_line') else os.environ.get('CURIOMESH_TURBO')}", flush=True)
SAVED = {k: copy.copy(guided.state[k]) for k in ("res", "auto_base")}
KEEP.CANDIDATES.clear()  # no finished fit: Apply runs the whole transfer

bpy.ops.ed.undo_push(message="probe")
bpy.ops.mesh.primitive_cube_add()
probe = bpy.context.object.name
try:
    bpy.ops.ed.undo()
    CAN_UNDO = probe not in bpy.data.objects
except RuntimeError:
    CAN_UNDO = False
for o in [o for o in bpy.data.objects if o.name.startswith("Cube")]:
    bpy.data.objects.remove(o, do_unlink=True)
print(f"INFO Blender undo in this run: {CAN_UNDO}", flush=True)
SL.finish = timed_finish


def ready(bake="OFF", chart=True):
    """The saved result back on show, as if Apply had not run yet."""
    global STEP0
    gs = bpy.context.scene.curiomesh_guided
    res = dict(SAVED["res"])
    if not chart:
        res["chart"] = None  # the bake unwraps new UVs, as without kept seams
    guided.state.update(res=res, auto_base=SAVED["auto_base"], source="Model", work=None)
    gs["enabled"] = True
    gs.bake_detail, gs.bake_size = bake, "1024"
    model = bpy.data.objects["Model"]
    for o in bpy.context.selected_objects:
        o.select_set(False)
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    bpy.ops.ed.undo_push(message="before Apply")
    STEP0 = H.current(bpy.context.scene)


def apply(label, stop_after=None):
    RUN["stop_after"] = stop_after
    names = set(bpy.data.objects.keys())
    r = bpy.ops.curiomesh.guided_apply()
    bpy.ops.ed.undo_push(message="Apply")  # a script's operator call pushes no step of its own here (as in blender_uv_ui)
    made = [n for n in bpy.data.objects.keys() if n not in names]
    longest = report(label)
    ok(r == {"FINISHED"} and len(made) == 1, f"{label}: Apply {r}, wrote {made}: {guided.state['applied']}")
    return (made[0] if made else None), longest


# --- 1. after_apply: the whole transfer, sliced
ready()
name, longest = apply("1 transfer")
ok(longest < 0.1, f"1 transfer: longest slice {1000 * longest:.0f} ms < 100 ms")
mesh = bpy.data.objects[name].data
ok(uv_ok(mesh) and HOOKS.LAST.get("hash") and not RUN["stopped"], f"1 transfer: finished with UVs ({HOOKS.LAST.get('messages')})")
undo_redo("1 transfer", name)

ready()
name, longest = apply("1 transfer, Stop", stop_after=2.0)
ok(longest < 0.1, f"1 transfer, Stop: longest slice {1000 * longest:.0f} ms")
ok(RUN["stopped"] and name and uv_ok(bpy.data.objects[name].data) and "Stopped" in guided.state["applied"], f"1 transfer, Stop: a valid mesh with UVs, and it says so: {HOOKS.LAST.get('messages')}")

# --- 2. the bake unwrap (New UVs into the bake layer), sliced
bpy.context.scene.curiomesh_uv.uvs = "NEW"  # no kept seams: the bake makes its own layer with New UVs
UP.sync()
ready("NORMAL", chart=False)
name, longest = apply("2 bake unwrap")
ok(longest < 0.1, f"2 bake unwrap: longest slice {1000 * longest:.0f} ms (the bake itself is Blender's, in the last slice)")
mesh = bpy.data.objects[name].data
ok(uv_ok(mesh) and pkg.textures.BAKE_LAYER in mesh.uv_layers and "Smart Project" not in BAKE.LAST.get("message", ""), f"2 bake unwrap: New UVs made the bake layer ({BAKE.LAST.get('message')}, {BAKE.LAST.get('seconds', 0):.1f} s)")
undo_redo("2 bake unwrap", name)

BAKE._cache.clear()
ready("NORMAL", chart=False)
name, longest = apply("2 bake unwrap, Stop", stop_after=2.0)
ok(RUN["stopped"] and name and uv_ok(bpy.data.objects[name].data) and "Smart Project" in BAKE.LAST.get("message", "") and "Stopped" in guided.state["applied"], f"2 bake unwrap, Stop: Smart Project made the layer and it says so: {guided.state['applied']}")

# --- 3. Classic Remesh: the UV transfer after the remesh, sliced
settings = bpy.context.scene.curiomesh_settings
for o in bpy.context.selected_objects:
    o.select_set(False)
model = bpy.data.objects["Model"]
model.select_set(True)
bpy.context.view_layer.objects.active = model


def remesh(label, stop_after=None):
    global STEP0
    bpy.ops.ed.undo_push(message="before Remesh")
    STEP0 = H.current(bpy.context.scene)
    RUN["stop_after"] = stop_after
    names = set(bpy.data.objects.keys())
    r = bpy.ops.curiomesh.remesh(texture_mode="PROJECT", output_mode="NEW", target_faces=2000)
    bpy.ops.ed.undo_push(message="Remesh")
    made = [n for n in bpy.data.objects.keys() if n not in names]
    longest = report(label)
    ok(r == {"FINISHED"} and len(made) == 1, f"{label}: {r}, wrote {made}, UVs {bpy.context.scene.curiomesh_settings.metrics_uv_status}")
    return (made[0] if made else None), longest


name, longest = remesh("3 remesh transfer")
ok(longest < 0.1, f"3 remesh transfer: longest slice {1000 * longest:.0f} ms (the remesh itself is the first slice)")
ok(name and uv_ok(bpy.data.objects[name].data) and bpy.context.scene.curiomesh_settings.metrics_uv_status == "TRANSFERRED", "3 remesh transfer: finished with the side-by-side transfer")
undo_redo("3 remesh transfer", name)
bpy.context.view_layer.objects.active = bpy.data.objects["Model"]
name, longest = remesh("3 remesh transfer, Stop", stop_after=2.0)
ok(RUN["stopped"] and name and uv_ok(bpy.data.objects[name].data), f"3 remesh transfer, Stop: a valid mesh with Blender's transfer ({bpy.context.scene.curiomesh_settings.metrics_uv_status})")

SL.finish = REAL_FINISH
pkg.unregister()
print(f"UVSLICED {'FAILED ' + str(len(FAILS)) if FAILS else 'OK'}", flush=True)
sys.exit(1 if FAILS else 0)
