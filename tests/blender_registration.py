# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors
"""Every Blender class CurioMesh defines must be registered once the add-on is enabled, and gone after it's disabled.

Catches the 0.9.0-0.9.7 bug where the "5. Apply" panel was defined but never registered, so the Apply button was missing
for everyone who kept the default sidebar tab. Run headless against an installed or folder copy of the add-on:

    blender -b --factory-startup --python tests/blender_registration.py -- <module name>

<module name> is how the add-on is enabled, e.g. bl_ext.user_default.curiomesh for an installed extension. Exits 1 on
any failure.
"""

import inspect
import sys

import addon_utils
import bpy

KINDS = (bpy.types.Panel, bpy.types.Operator, bpy.types.Menu, bpy.types.UIList, bpy.types.PropertyGroup,
         bpy.types.AddonPreferences, bpy.types.Gizmo, bpy.types.GizmoGroup, bpy.types.WorkSpaceTool)


def defined_classes(package: str) -> list[type]:
    found = {}
    for name, module in list(sys.modules.items()):
        if module is None or not (name == package or name.startswith(package + ".")):
            continue
        for _n, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != name or not issubclass(obj, KINDS) or obj in KINDS:
                continue
            if obj.__name__.startswith("_") or not (getattr(obj, "bl_idname", "") or getattr(obj, "bl_label", "")):
                continue  # mixins and bases
            found[obj.__name__] = obj
    return list(found.values())


def is_registered(cls: type) -> bool:
    return bool(getattr(cls, "is_registered", False)) or hasattr(bpy.types, getattr(cls, "bl_idname", "") or cls.__name__)


def main() -> int:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    module = args[0] if args else next(m.__name__ for m in addon_utils.modules() if m.__name__.endswith("curiomesh"))
    addon_utils.enable(module, default_set=True, handle_error=None)
    classes = defined_classes(module)
    missing = [c.__name__ for c in classes if not is_registered(c)]
    panels = [c for c in classes if issubclass(c, bpy.types.Panel)]
    orphan_children = [c.__name__ for c in panels if getattr(c, "bl_parent_id", "") and not hasattr(bpy.types, c.bl_parent_id)]
    print(f"REG classes={len(classes)} panels={len(panels)} missing={missing} orphan_children={orphan_children}")
    for must in ("CURIOMESH_PT_guided", "CURIOMESH_PT_guided_apply"):
        if not hasattr(bpy.types, must):
            missing.append(must)
    guided, overlay = sys.modules[module + ".guided"], sys.modules[module + ".overlay"]
    guided._enabled_set(bpy.context.scene.curiomesh_guided, True)  # Guided Preview on (its flag only), as if Start had run
    try:  # a second register() (scripts, reloads) must start clean, not raise "already registered"
        sys.modules[module].register()
        sys.modules[module].register()
        again = []
    except Exception as exc:
        again = [f"{type(exc).__name__}: {exc}"]
    print(f"REG twice={again}")
    missing += again
    addon_utils.disable(module, default_set=True)
    left = [c.__name__ for c in classes if getattr(c, "is_registered", False)]
    if bpy.app.timers.is_registered(overlay._watch):
        left.append("overlay._watch timer")
    print(f"UNREG left={left}")
    # Disable then enable in one session: the preview flag must not survive, or Start would skip starting its timer.
    addon_utils.enable(module, default_set=True, handle_error=None)
    stale = bpy.context.scene.curiomesh_guided.enabled or "enabled" in (bpy.context.scene.get("curiomesh_guided") or {})
    print(f"REENABLE stale_preview_flag={stale}")
    addon_utils.disable(module, default_set=True)
    ok = not missing and not orphan_children and not left and not stale
    print("REGISTRATION PASS" if ok else "REGISTRATION FAIL")
    return 0 if ok else 1


sys.exit(main())
