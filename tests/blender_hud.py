# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""The dashboard (HUD): off by default, live job progress, clickable buttons, nothing left after unregister.

    blender -b --factory-startup --python tests/blender_hud.py
"""

import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import bpy

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def ok(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


spec = importlib.util.spec_from_file_location("curiomesh", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec)
sys.modules["curiomesh"] = pkg
spec.loader.exec_module(pkg)
pkg.register()
from curiomesh import guided, overlay  # noqa: E402

# 0. off by default; the once-per-scene default forces it off and Start does not touch it
scene = bpy.data.scenes.new("hudtest")
ok(scene.curiomesh_settings.hud_enabled is False, "a fresh scene starts with the dashboard off")
scene.curiomesh_settings.hud_enabled = True  # a file saved while it was on by default
overlay.apply_hud_default(scene)
ok(scene.curiomesh_settings.hud_enabled is False, "apply_hud_default switches an old True off")
scene.curiomesh_settings.hud_enabled = True
overlay.apply_hud_default(scene)
ok(scene.curiomesh_settings.hud_enabled is True, "and only once per scene")
scene.curiomesh_settings.hud_enabled = False

# 1. pure pieces
rects = [(10, 10, 50, 30, "hide"), (60, 10, 90, 30, "stop")]
ok(overlay.hit_test(rects, 20, 20) == "hide" and overlay.hit_test(rects, 70, 20) == "stop" and overlay.hit_test(rects, 55, 20) is None, "hit_test finds the right button")
ok(overlay.clock(72) == "1:12" and overlay.clock(-3) == "0:00", "clock")
job = {"frac": 0.42, "stage": "x", "done": 2, "total": 8, "elapsed": 72.0, "left": 120.0, "alive_tick": 0.0}
ok(overlay.job_text(job) == "42%  piece 3 of 8  1:12 elapsed  about 2:00 left", f"job_text: {overlay.job_text(job)!r}")
ok(overlay.job_text({**job, "total": 0, "elapsed": None, "left": None}) == "42%", "job_text leaves out what is not known")

# 2. job model from a running job, ticker only while it runs
settings = bpy.context.scene.curiomesh_settings
ok(overlay.job_info() is None, "no job: no job info")
runner = SimpleNamespace(busy=True, progress=(2, 8), started=time.perf_counter() - 72, seconds_left=lambda: 120.0, stage="Redoing piece 3", stop=lambda: None)
guided.state["work"] = {"key": ("t", "t"), "runner": runner}
info = overlay.job_info()
ok(info is not None and info["done"] == 2 and info["total"] == 8 and 71 < info["elapsed"] < 80 and info["left"] == 120.0 and info["stage"] == "Redoing piece 3", f"job info from the runner: {info}")
ok(overlay.hud_model(bpy.context)["job"] is not None, "hud_model carries the job")
settings.hud_enabled = False
overlay.ensure_ticker()
ok(not bpy.app.timers.is_registered(overlay._hud_tick), "no ticker while the dashboard is hidden")
settings.hud_enabled = True
overlay.ensure_ticker()
ok(bpy.app.timers.is_registered(overlay._hud_tick), "ticker runs while a job runs and the dashboard is on")
runner.busy = False
ok(overlay._hud_tick() is None, "the ticker stops itself when the job ends")
guided.state["work"] = None

# 3. buttons
ok(hasattr(bpy.ops.curiomesh, "hud_click") and hasattr(bpy.ops.curiomesh, "hud_toggle"), "hud_click and hud_toggle are registered")
bpy.ops.curiomesh.hud_toggle()
ok(settings.hud_enabled is False, "hud_toggle hides it")
bpy.ops.curiomesh.hud_toggle()
ok(settings.hud_enabled is True, "hud_toggle shows it")
overlay.run_hud_action(bpy.context, "collapse")
ok(settings.hud_collapsed is True, "collapse")
overlay.run_hud_action(bpy.context, "collapse")
ok(settings.hud_collapsed is False, "expand")
overlay.run_hud_action(bpy.context, "hide")
ok(settings.hud_enabled is False, "hide")
ok(not overlay.CURIOMESH_OT_hud_click.poll(bpy.context), "hud_click does nothing when no dashboard is drawn")

# 4. everything removed on unregister
kc = bpy.context.window_manager.keyconfigs.addon
if kc is not None:
    ok(len(overlay._keymap_items) == 2, "keymap items added")
settings.hud_enabled = True
runner.busy = True
guided.state["work"] = {"key": ("t", "t"), "runner": runner}
overlay.ensure_ticker()
pkg.unregister()
ok(not bpy.app.timers.is_registered(overlay._hud_tick) and not overlay._keymap_items and not overlay.HUD_RECTS, "unregister removes the ticker, keymap items and rects")
guided.state["work"] = None
print(f"HUD {'FAILED ' + str(len(FAILS)) if FAILS else 'OK'}", flush=True)
sys.exit(1 if FAILS else 0)
