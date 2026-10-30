# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

# ruff: noqa: E402

bl_info = {
    "name": "CurioMesh Quad Remesher",
    "author": "CurioCrafter",
    "version": (0, 6, 0),
    "blender": (3, 6, 0),
    "location": "3D View > N-panel > CurioMesh",
    "category": "Mesh",
    "description": "Atlas-Q scan-aware quad retopology with explicit validation",
    "doc_url": "https://github.com/CurioCrafter/CurioMesh",
}

import bpy

from .operators import (
    CURIOMESH_OT_apply_preset,
    CURIOMESH_OT_analyze,
    CURIOMESH_OT_compute_metrics,
    CURIOMESH_OT_remesh,
)
from . import batch, live, overlay, region
from .ui import CURIOMESH_PG_settings, PANEL_CLASSES, apply_category

D = overlay.PREF_DEFAULTS


def _tab_update(self, context):
    apply_category(self.panel_tab)


def _redraw(self, context):
    overlay.tag_redraw()


def _color(name: str, label: str, description: str):
    return bpy.props.FloatVectorProperty(
        name=label,
        description=description,
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=D[name],
        update=_redraw,
    )


# (property, label) of the optional panels.
SUBPANEL_TOGGLES = (
    ("show_scan", "Scan"),
    ("show_atlas", "Smart Solver"),
    ("show_shape", "Features to Keep"),
    ("show_pipeline", "Cleanup & Output"),
    ("show_pairclosure", "PairClosure"),
    ("show_materials", "Materials & UVs"),
    ("show_batch", "Batch"),
    ("show_diagnostics", "Diagnostics"),
    ("show_debug", "Debug"),
    ("show_regions", "Regions"),
)


class CURIOMESH_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    panel_tab: bpy.props.StringProperty(
        name="Sidebar Tab",
        description="Name of the tab in the 3D Viewport sidebar (N panel) that holds CurioMesh",
        default=D["panel_tab"],
        update=_tab_update,
    )
    compact_mode: bpy.props.BoolProperty(
        name="Compact Panels",
        description="Smaller panels: fewer section labels and no explanatory text. Handy on small screens",
        default=D["compact_mode"],
    )
    show_scan: bpy.props.BoolProperty(name="Scan", description="Show the Scan panel", default=True)
    show_atlas: bpy.props.BoolProperty(
        name="Smart Solver", description="Show the Atlas-Q solver options (when the method is Smart)", default=True
    )
    show_shape: bpy.props.BoolProperty(name="Features to Keep", description="Show the features-to-keep panel", default=True)
    show_pipeline: bpy.props.BoolProperty(
        name="Cleanup & Output", description="Show the cleanup and output panel", default=True
    )
    show_pairclosure: bpy.props.BoolProperty(
        name="PairClosure", description="Show the PairClosure options (when the method is PairClosure)", default=True
    )
    show_materials: bpy.props.BoolProperty(name="Materials & UVs", description="Show the materials and UVs panel", default=True)
    show_batch: bpy.props.BoolProperty(name="Batch", description="Show the batch remesh panel", default=True)
    show_diagnostics: bpy.props.BoolProperty(name="Diagnostics", description="Show the measurements panel", default=True)
    show_debug: bpy.props.BoolProperty(name="Debug", description="Show the debug panel", default=True)
    show_regions: bpy.props.BoolProperty(name="Regions", description="Show the Regions panel", default=True)

    hud_default: bpy.props.BoolProperty(
        name="HUD On in New Scenes",
        description="Whether the viewport dashboard starts on in scenes that have not used CurioMesh yet",
        default=D["hud_default"],
    )
    hud_corner: bpy.props.EnumProperty(
        name="Corner",
        description="Which corner of the viewport the dashboard sits in",
        items=overlay.HUD_CORNERS,
        default=D["hud_corner"],
        update=_redraw,
    )
    hud_offset_x: bpy.props.IntProperty(
        name="Offset X",
        description="Distance from the left or right edge, in pixels. 70 keeps clear of the toolbar",
        default=D["hud_offset_x"],
        min=0,
        max=2000,
        subtype="PIXEL",
        update=_redraw,
    )
    hud_offset_y: bpy.props.IntProperty(
        name="Offset Y",
        description="Distance from the top or bottom edge, in pixels",
        default=D["hud_offset_y"],
        min=0,
        max=2000,
        subtype="PIXEL",
        update=_redraw,
    )
    hud_scale: bpy.props.FloatProperty(
        name="Scale",
        description="Size of the dashboard (on top of Blender's Resolution Scale)",
        default=D["hud_scale"],
        min=0.5,
        max=2.5,
        update=_redraw,
    )
    hud_opacity: bpy.props.FloatProperty(
        name="Background Opacity",
        description="How solid the dashboard background is. Text always stays fully opaque",
        default=D["hud_opacity"],
        min=0.2,
        max=1.0,
        subtype="FACTOR",
        update=_redraw,
    )

    heat_tri_color: _color("heat_tri_color", "Triangles", "Colour of leftover triangles in the heatmap")
    heat_ngon_color: _color("heat_ngon_color", "N-gons", "Colour of faces with five or more sides in the heatmap")
    heat_pole3_color: _color("heat_pole3_color", "3-Edge Poles", "Colour of the dots where three edges meet")
    heat_pole5_color: _color("heat_pole5_color", "5+-Edge Poles", "Colour of the larger dots where five or more edges meet")
    heat_point_size: bpy.props.FloatProperty(
        name="Dot Size",
        description="Size of the pole dots in pixels (5+-edge poles are drawn larger still)",
        default=D["heat_point_size"],
        min=2.0,
        max=20.0,
        update=_redraw,
    )

    region_tint: _color("region_tint", "New Faces", "Tint laid over the new quads in a region preview")
    region_seam_color: _color("region_seam_color", "Kept Seam", "Colour of the seam outline that stays fixed in a region preview")
    region_seam_width: bpy.props.FloatProperty(
        name="Seam Width",
        description="Thickness of the seam outline in pixels",
        default=D["region_seam_width"],
        min=1.0,
        max=10.0,
        update=_redraw,
    )

    def draw(self, context):
        layout = self.layout.split(factor=0.72).column()  # keep the form readable on wide windows
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Panels", icon="WINDOW")
        col = box.column()
        col.prop(self, "panel_tab")
        col.prop(self, "compact_mode")
        shown = box.column()
        shown.use_property_split = False
        shown.label(text="Show these panels")
        flow = shown.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
        for name, label in SUBPANEL_TOGGLES:
            flow.prop(self, name, text=label, toggle=True)

        box = layout.box()
        box.label(text="Dashboard (HUD)", icon="OVERLAY")
        col = box.column()
        col.prop(self, "hud_default")
        col.prop(self, "hud_corner")
        offsets = col.row(align=True)
        offsets.use_property_split = False
        offsets.prop(self, "hud_offset_x", text="Offset X")
        offsets.prop(self, "hud_offset_y", text="Offset Y")
        col.prop(self, "hud_scale")
        col.prop(self, "hud_opacity", slider=True)

        box = layout.box()
        box.label(text="Heatmap", icon="SHADING_RENDERED")
        col = box.column()
        col.prop(self, "heat_tri_color")
        col.prop(self, "heat_ngon_color")
        col.prop(self, "heat_pole3_color")
        col.prop(self, "heat_pole5_color")
        col.prop(self, "heat_point_size")

        box = layout.box()
        box.label(text="Region Preview", icon="SELECT_SUBTRACT")
        col = box.column()
        col.prop(self, "region_tint")
        col.prop(self, "region_seam_color")
        col.prop(self, "region_seam_width")

        layout.operator("curiomesh.reset_preferences", icon="LOOP_BACK")


class CURIOMESH_OT_reset_preferences(bpy.types.Operator):
    bl_idname = "curiomesh.reset_preferences"
    bl_label = "Reset CurioMesh Preferences"
    bl_description = "Put every CurioMesh preference (tab name, panels, dashboard, colours) back to its default"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        prefs = overlay.get_prefs()
        for name in D:
            try:
                prefs.property_unset(name)
            except Exception:
                pass
        apply_category()
        overlay.tag_redraw()
        return {"FINISHED"}


classes = (
    CURIOMESH_AP_preferences,
    CURIOMESH_OT_reset_preferences,
    batch.CURIOMESH_PG_batch_item,
    CURIOMESH_PG_settings,
    batch.CURIOMESH_OT_batch_remesh,
    *overlay.CLASSES,
    CURIOMESH_OT_analyze,
    CURIOMESH_OT_remesh,
    CURIOMESH_OT_apply_preset,
    CURIOMESH_OT_compute_metrics,
    *PANEL_CLASSES,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_settings = bpy.props.PointerProperty(type=CURIOMESH_PG_settings)
    overlay.register()
    live.register()
    region.register()
    apply_category()


def unregister():
    region.unregister()
    live.unregister()
    overlay.unregister()
    if hasattr(bpy.types.Scene, "curiomesh_settings"):
        del bpy.types.Scene.curiomesh_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
