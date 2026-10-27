"""Capture the README screenshots in a real Blender window.

    blender --factory-startup --window-geometry 0 0 1920 1080 \
      --python scripts/capture_readme_screenshots.py

Builds a triangulated Suzanne, runs TRIAD-Q with the rescue pass disabled and
enabled, paints leftover triangles red, then saves docs/images/*.png and quits.
"""

from __future__ import annotations

from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
sys.path.insert(0, str(ROOT / "tests"))
from blender_smoke import load_addon  # noqa: E402

addon = load_addon()
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
    for obj in (source, before):
        obj.select_set(False)
    after.select_set(False)
    bpy.app.timers.register(capture_viewport, first_interval=1.0)


def capture_viewport():
    OUT.mkdir(parents=True, exist_ok=True)
    with bpy.context.temp_override(window=window, area=view):
        bpy.ops.screen.screenshot_area(filepath=str(OUT / "rescue_before_after.png"))
    after.select_set(True)
    space.show_region_ui = True
    bpy.app.timers.register(open_panel, first_interval=0.5)


def open_panel():
    ui_region.active_panel_category = "CurioMesh"
    with bpy.context.temp_override(window=window, area=view, region=ui_region):
        for _ in range(14):
            bpy.ops.view2d.scroll_down()
    bpy.app.timers.register(capture_panel, first_interval=1.0)


def capture_panel():
    with bpy.context.temp_override(window=window):
        bpy.ops.screen.screenshot(filepath=str(OUT / "panel_triadq.png"))
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(frame, first_interval=2.0)
