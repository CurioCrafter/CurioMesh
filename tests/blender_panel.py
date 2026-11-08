# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Panel pass: registers cleanly, dashboard stays off, step strip, "?" and prompt panel draw, unregister leaves nothing.

    blender -b --factory-startup --python tests/blender_panel.py
"""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import MagicMock

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
from curiomesh import panel  # noqa: E402

ctx = bpy.context
ok(ctx.scene.curiomesh_settings.hud_enabled is False, "dashboard is off in a fresh scene")
ok(not hasattr(pkg.overlay.get_prefs(), "hud_default"), "hud_default pref is gone")
ok(hasattr(bpy.types, "CURIOMESH_PT_seg_card"), "seg_card panel is registered")
ok(hasattr(bpy.types, "CURIOMESH_PT_pro_prompt"), "prompt panel is registered")
ok(hasattr(bpy.types.WindowManager, "curiomesh_prompt_goal"), "prompt goal property exists")
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ok(panel.current_step(ctx) == 0, "no mesh = step 0")
bpy.ops.mesh.primitive_cube_add()
ctx.view_layer.objects.active = ctx.object
ok(panel.current_step(ctx) in (1, 2), "a mesh and no result = step 1 or 2")
for fn in (panel.draw_dashboard_button, panel.draw_steps):
    fn(MagicMock(), ctx)
    ok(True, f"{fn.__name__} draws")
panel.draw_review_button(MagicMock())
panel.draw_spots_line(MagicMock())
panel.draw_empty(MagicMock(), ctx)
panel.CURIOMESH_PT_pro_prompt.draw(MagicMock(), ctx)
ok(True, "review button, spots line, empty state and prompt panel draw")
pkg.unregister()
ok(not hasattr(bpy.types, "CURIOMESH_PT_pro_prompt"), "prompt panel removed")
ok(not hasattr(bpy.types.WindowManager, "curiomesh_prompt_goal"), "prompt property removed")
ok(not hasattr(bpy.types, "CURIOMESH_OT_help"), "help operator removed")
print("FAILS:", FAILS, flush=True)
sys.exit(1 if FAILS else 0)
