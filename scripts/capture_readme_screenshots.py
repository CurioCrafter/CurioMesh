# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Capture the README screenshots in a real Blender window.

    blender --factory-startup --window-geometry 0 0 1920 1080       --python scripts/capture_readme_screenshots.py

Builds a triangulated Suzanne, runs TRIAD-Q with the rescue pass disabled and
enabled, paints leftover triangles red, shows the HUD and heatmap, then builds a
region preview (box cage around the forehead) and opens the add-on preferences.
Saves docs/images/*.png and quits. The add-on is installed from a temporary
copy through Blender's own add-on system so the Preferences window shows it;
preferences are never saved.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import bpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
sys.path.insert(0, str(ROOT / "tests"))
from blender_smoke import load_addon  # noqa: E402

bpy.context.preferences.use_preferences_save = False


def short_path(folder: Path) -> Path:
    """On Windows, mount the folder as a spare drive so screenshots show X:\ instead of a user path."""
    if sys.platform != "win32":
        return folder
    for letter in "XYZ":
        if not Path(f"{letter}:/").exists() and subprocess.run(["subst", f"{letter}:", str(folder)]).returncode == 0:
            return Path(f"{letter}:/")
    return folder


def install_addon():
    """Enable the add-on like a user would; fall back to a plain load if that fails."""
    try:
        tmp = short_path(Path(tempfile.mkdtemp(prefix="curiomesh_capture_")))
        shutil.copytree(
            ROOT,
            tmp / "addons" / "curiomesh",
            ignore=shutil.ignore_patterns("tests", "docs", "scripts", "__pycache__", ".git*", "dist", ".tools"),
        )
        entry = bpy.context.preferences.filepaths.script_directories.new()
        entry.name = "curiomesh_capture"
        entry.directory = str(tmp)
        bpy.utils.refresh_script_paths()
        bpy.ops.preferences.addon_enable(module="curiomesh")
        if "curiomesh" not in bpy.context.preferences.addons:
            raise RuntimeError("add-on did not enable")
        print("CAPTURE addon enabled via preferences")
        return sys.modules["curiomesh"]
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"CAPTURE addon fallback load_addon ({exc!r})")
        return load_addon()


addon = install_addon()
# Open panels by default so one screenshot shows them (panel state is created on the first draw of the tab).
from curiomesh import ui  # noqa: E402

_panels = ui.panel_classes()
for _cls in reversed(_panels):
    bpy.utils.unregister_class(_cls)
for _cls in _panels:
    if _cls.bl_idname == "CURIOMESH_PT_regions":
        _cls.bl_options = set()
    if _cls in ui.region.PANEL_CLASSES:
        _cls.bl_category = "Regions"  # its own tab, so the whole panel fits in one screenshot
    bpy.utils.register_class(_cls)
engine = sys.modules["curiomesh.triadq.engine"]
real_augment = engine._augment_matching


def make_source():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_monkey_add()
    obj = bpy.context.object
    obj.name = "Source"
    mod = obj.modifiers.new("sub", "SUBSURF")
    mod.levels = 1
    bpy.ops.object.modifier_apply(modifier=mod.name)
    mod = obj.modifiers.new("tri", "TRIANGULATE")
    mod.quad_method = "SHORTEST_DIAGONAL"
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


def remesh(source, name, rescue):
    engine._augment_matching = real_augment if rescue else (lambda *args, **kwargs: 0)
    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    bpy.context.view_layer.objects.active = source
    bpy.ops.curiomesh.remesh(
        engine="TRIAD_Q_LITE",
        target_faces=len(source.data.polygons) // 2,
        quality="BALANCED",
        output_mode="NEW",
        texture_mode="NONE",
        triad_seed_count=2,
        triad_flow_mode="ORGANIC",
        require_quads=False,
        triad_force_quads=False,
        shrinkwrap_project=False,
    )
    out = bpy.context.view_layer.objects.active
    out.name = name
    return out


def paint_triangles(obj):
    quad = bpy.data.materials.get("Quad") or bpy.data.materials.new("Quad")
    quad.diffuse_color = (0.62, 0.64, 0.68, 1.0)
    tri = bpy.data.materials.get("Leftover Triangle") or bpy.data.materials.new("Leftover Triangle")
    tri.diffuse_color = (0.9, 0.12, 0.1, 1.0)
    obj.data.materials.clear()
    obj.data.materials.append(quad)
    obj.data.materials.append(tri)
    for poly in obj.data.polygons:
        poly.material_index = 1 if len(poly.vertices) == 3 else 0
    return sum(1 for poly in obj.data.polygons if len(poly.vertices) == 3)


def label(text, x):
    bpy.ops.object.text_add(location=(x, 0, 1.25), rotation=(1.5708, 0, 0))
    obj = bpy.context.object
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.size = 0.22


source = make_source()
before = remesh(source, "Before", rescue=False)
after = remesh(source, "After", rescue=True)
before_tris = paint_triangles(before)
after_tris = paint_triangles(after)
rescued = after.get("curiomesh_triadq_rescued_pairs", 0)
print(f"CAPTURE before_tris={before_tris} after_tris={after_tris} rescued={rescued}")

for obj, x in ((source, -2.6), (before, 0.0), (after, 2.6)):
    obj.location = (x, 0, 0)
label(f"Source  {len(source.data.polygons):,} tris", -2.6)
label(f"Before  {before_tris} leftover tris", 0.0)
label(f"After  {after_tris} leftover tris", 2.6)

bpy.ops.object.select_all(action="DESELECT")
for obj in (source, before, after):
    obj.select_set(True)
bpy.context.view_layer.objects.active = after
settings = bpy.context.scene.curiomesh_settings
settings.engine = "TRIAD_Q_LITE"
settings.require_quads = False
settings.target_faces = len(source.data.polygons) // 2
settings.triad_seed_count = 2
settings.triad_flow_mode = "ORGANIC"
settings.hud_enabled = False
settings.heatmap_enabled = False

window = bpy.context.window_manager.windows[0]
view = next(area for area in window.screen.areas if area.type == "VIEW_3D")
space = view.spaces.active
space.shading.type = "SOLID"
space.shading.color_type = "MATERIAL"
space.overlay.show_wireframes = True
space.overlay.show_floor = False
space.overlay.show_axis_x = False
space.overlay.show_axis_y = False
space.show_region_ui = False
ui_region = next(region for region in view.regions if region.type == "UI")
window_region = next(region for region in view.regions if region.type == "WINDOW")


def frame():
    # View bounds and panel categories only exist after the first redraw.
    with bpy.context.temp_override(window=window, area=view, region=window_region):
        bpy.ops.view3d.view_axis(type="FRONT")
        bpy.ops.view3d.view_selected()
    space.region_3d.view_distance *= 0.6  # fill the frame with the three heads
    for obj in (source, before):
        obj.select_set(False)
    after.select_set(False)
    bpy.app.timers.register(capture_viewport, first_interval=1.0)


def capture_viewport():
    OUT.mkdir(parents=True, exist_ok=True)
    with bpy.context.temp_override(window=window, area=view):
        bpy.ops.screen.screenshot_area(filepath=str(OUT / "rescue_before_after.png"))
    # Second shot: HUD and heatmap on the rescued result alone.
    for obj in bpy.context.scene.objects:
        obj.hide_set(obj is not after)
    space.shading.color_type = "OBJECT"
    bpy.ops.object.select_all(action="DESELECT")
    after.select_set(True)
    bpy.context.view_layer.objects.active = after
    with bpy.context.temp_override(window=window, area=view, region=window_region):
        bpy.ops.view3d.view_selected()
    after.select_set(False)
    settings.hud_enabled = True
    bpy.ops.curiomesh.toggle_heatmap()
    bpy.app.timers.register(capture_hud, first_interval=1.0)


def capture_hud():
    with bpy.context.temp_override(window=window, area=view):
        bpy.ops.screen.screenshot_area(filepath=str(OUT / "hud_heatmap.png"))
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
    after.select_set(True)
    space.show_region_ui = True
    bpy.app.timers.register(open_panel, first_interval=0.5)


def open_panel():
    ui_region.active_panel_category = "CurioMesh"
    # A scroll forces the sidebar to redraw on the new tab.
    with bpy.context.temp_override(window=window, area=view, region=ui_region):
        bpy.ops.view2d.scroll_up()
    with bpy.context.temp_override(window=window, area=view, region=window_region):
        bpy.ops.view3d.view_all()
    bpy.app.timers.register(capture_panel, first_interval=1.0)


def capture_panel():
    with bpy.context.temp_override(window=window):
        bpy.ops.screen.screenshot(filepath=str(OUT / "panel_quickstart.png"))
    bpy.app.timers.register(setup_region, first_interval=0.3)


# --------------------------------------------------------------------------- regions


def setup_region():
    """A subdivided Suzanne with a box cage over the forehead and a live region preview."""
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_monkey_add()
    head = bpy.context.object
    head.name = "Head"
    mod = head.modifiers.new("sub", "SUBSURF")
    mod.levels = 2
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True)
    bpy.context.view_layer.objects.active = head
    rs = bpy.context.scene.curiomesh_region
    bpy.ops.curiomesh.region_add_cage(kind="BOX")
    cage = rs.volumes[0].obj
    cage.name = "Forehead Cage"
    cage.location = (0.0, -0.62, 0.42)
    cage.scale = (0.5, 0.32, 0.22)
    rs.source = "VOLUMES"
    rs.strategy = "FLOW"
    rs.target_edge = 0.05
    bpy.context.scene.curiomesh_settings.heatmap_enabled = False
    bpy.context.scene.curiomesh_settings.hud_enabled = True
    bpy.context.scene.curiomesh_settings.target_faces = len(head.data.polygons)
    bpy.context.scene.curiomesh_settings.engine = "ATLAS_Q"
    result = bpy.ops.curiomesh.region_preview()
    print(f"CAPTURE region_preview={result} message={addon.region.status()}")
    bpy.ops.object.select_all(action="DESELECT")
    cage.select_set(True)
    bpy.context.view_layer.objects.active = cage
    space.shading.color_type = "OBJECT"
    space.overlay.show_wireframes = False
    space.show_region_ui = False
    region3d = space.region_3d
    from mathutils import Euler, Vector

    region3d.view_perspective = "PERSP"
    region3d.view_rotation = Euler((1.25, 0.0, 0.42), "XYZ").to_quaternion()
    region3d.view_location = Vector((0.0, -0.45, 0.3))
    region3d.view_distance = 3.2
    bpy.app.timers.register(capture_region_view, first_interval=1.5)


def capture_region_view():
    with bpy.context.temp_override(window=window, area=view):
        bpy.ops.screen.screenshot_area(filepath=str(OUT / "region_cage_preview.png"))
    space.show_region_ui = True
    bpy.app.timers.register(open_region_panel, first_interval=0.5)


def open_region_panel():
    ui_region.active_panel_category = "Regions"
    with bpy.context.temp_override(window=window, area=view, region=ui_region):
        bpy.ops.view2d.scroll_up()
    bpy.app.timers.register(capture_region_panel, first_interval=1.2)


def capture_region_panel():
    with bpy.context.temp_override(window=window):
        bpy.ops.screen.screenshot(filepath=str(OUT / "panel_regions.png"))
    bpy.app.timers.register(open_preferences, first_interval=0.3)


# --------------------------------------------------------------------------- preferences


def open_preferences():
    """Show the Preferences editor in the big viewport area so the whole add-on section fits."""
    prefs = bpy.context.preferences
    prefs.active_section = "ADDONS"
    bpy.context.window_manager.addon_search = "CurioMesh"
    try:
        bpy.ops.preferences.addon_expand(module="curiomesh")
    except RuntimeError as exc:
        print(f"CAPTURE addon_expand failed: {exc}")
    view.ui_type = "PREFERENCES"
    with bpy.context.temp_override(window=window, area=view):
        bpy.ops.screen.screen_full_area(use_hide_panels=False)
    bpy.app.timers.register(capture_preferences, first_interval=2.0)


def capture_preferences():
    area = next(a for a in window.screen.areas if a.type == "PREFERENCES")
    with bpy.context.temp_override(window=window, area=area):
        bpy.ops.screen.screenshot_area(filepath=str(OUT / "preferences.png"))
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(frame, first_interval=2.0)
