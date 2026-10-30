# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

import bpy

from .operators import (
    ATLAS_STRATEGY_ITEMS,
    ENGINE_ITEMS,
    FLOW_ITEMS,
    OUTPUT_ITEMS,
    QUALITY_ITEMS,
    TEXTURE_ITEMS,
    TOPOLOGY_ITEMS,
)
from . import guided, live, overlay, region
from .batch import CURIOMESH_PG_batch_item


def _live_update(self, context):
    live.on_setting_changed(self)


def _live_toggle_update(self, context):
    live.on_live_toggled(self, context)


def _redraw_update(self, context):
    overlay.tag_redraw()


class CURIOMESH_PG_settings(bpy.types.PropertyGroup):
    target_faces: bpy.props.IntProperty(name="Target Faces", description="Roughly how many faces the new mesh should have (a quad counts as one face)", default=8000, min=4, soft_max=250000, update=_live_update)
    engine: bpy.props.EnumProperty(name="Method", description="How the new quads are built. Smart (Atlas-Q) measures the mesh, then tries several solvers and keeps the best", items=ENGINE_ITEMS, default="ATLAS_Q")
    quality: bpy.props.EnumProperty(name="Quality", description="Draft is fast with light cleanup; Hero spends more time on cleanup and checks", items=QUALITY_ITEMS, default="BALANCED")
    seed: bpy.props.IntProperty(name="Random Seed", description="Change this for a different but repeatable result", default=0, min=0, soft_max=100000)
    triad_seed_count: bpy.props.IntProperty(name="PairClosure Starts", description="How many starting points PairClosure tries. More is slower and usually cleaner", default=6, min=1, max=32, update=_live_update)
    triad_feature_angle: bpy.props.FloatProperty(name="Feature Angle", description="Edges sharper than this angle (in degrees) count as creases and are kept", default=35.0, min=0.0, max=180.0, update=_live_update)
    triad_force_quads: bpy.props.BoolProperty(name="PairClosure Pure Quads", description="Force every face to be a quad, even if shape accuracy suffers", default=False)
    triad_flow_mode: bpy.props.EnumProperty(name="PairClosure Style", description="Tune PairClosure for the kind of model: organic, hard-surface, messy scan or textured", items=FLOW_ITEMS, default="AUTO", update=_live_update)
    flat_tolerance: bpy.props.FloatProperty(name="Flatness", description="How far a surface may bump away from flat and still become one big quad, as a percent of the mesh size. Higher gives fewer faces and less detail", default=0.1, min=0.0, soft_max=2.0, precision=3, subtype="PERCENTAGE", update=_live_update)
    triad_field_mode: bpy.props.EnumProperty(name="Flow", description="Direction the quads follow: smoothed flow, or also the surface curvature (about 1.5x slower)", items=(("SMOOTH", "Smooth", "Smoothed flow pinned to sharp edges and borders"), ("CURVATURE", "Follow Curvature", "Also line up with the main bending directions of the surface")), default="SMOOTH", update=_live_update)
    atlas_strategy: bpy.props.EnumProperty(name="Scan Type", description="Which plan Smart (Atlas-Q) follows. Auto picks one from the measured mesh", items=ATLAS_STRATEGY_ITEMS, default="AUTO")
    atlas_candidate_count: bpy.props.IntProperty(name="Attempts", description="How many candidate results are built and compared; the best one wins. More is slower", default=3, min=1, max=8)
    topology_policy: bpy.props.EnumProperty(name="Holes & Shells", description="Preserve keeps holes, open borders and separate parts as they are. Repair may close holes and drop thin sheets", items=TOPOLOGY_ITEMS, default="PRESERVE")
    require_quads: bpy.props.BoolProperty(name="Pure Quads", description="Reject results that still contain triangles or badly shaped quads. Turn off to keep the best attempt even with a few triangles", default=True, update=_live_update)
    analysis_sample_faces: bpy.props.IntProperty(
        name="Analysis Detail", description="How many faces are sampled when scanning very large meshes. More is slower and more exact",
        default=120000,
        min=1000,
        max=500000,
    )
    preserve_sharp: bpy.props.BoolProperty(name="Keep Sharp Edges", description="Keep marked sharp edges and hard creases as quad edge lines", default=True)
    preserve_boundary: bpy.props.BoolProperty(name="Keep Open Borders", description="Keep the outline of open surfaces in place", default=True)
    preserve_seams: bpy.props.BoolProperty(name="Keep UV Seams", description="Treat UV seams as edges to preserve so textures stay clean", default=True)
    use_symmetry: bpy.props.BoolProperty(name="Mirror Symmetry", description="Use the mesh's symmetry so both halves match", default=False)
    preserve_attributes: bpy.props.BoolProperty(name="Keep Attributes", description="Carry colours, weights and other mesh data over to the new mesh", default=True)
    smooth_normals: bpy.props.BoolProperty(name="Smooth Shading", description="Shade the result smooth", default=True)
    cleanup_strength: bpy.props.FloatProperty(name="Cleanup", description="How hard to tidy leftover triangles, n-gons and bad corners (0 = off, 1 = strongest)", default=0.25, min=0.0, max=1.0)
    voxel_repair: bpy.props.BoolProperty(name="Voxel Repair Fallback", description="If the mesh is too broken to remesh, rebuild it through voxels first (can close small holes)", default=True)
    shrinkwrap_project: bpy.props.BoolProperty(name="Snap to Original", description="Project the new vertices back onto the original surface to keep fine detail", default=True)
    texture_mode: bpy.props.EnumProperty(name="Textures", description="How UVs and textures are carried over to the new mesh", items=TEXTURE_ITEMS, default="PROJECT")
    bake_fallback: bpy.props.BoolProperty(name="Bake If UVs Fail", description="If UV transfer fails, bake the colour into a new texture instead", default=False)
    bake_resolution: bpy.props.IntProperty(name="Bake Size", description="Texture size in pixels used when baking", default=2048, min=64, soft_max=8192)
    bake_margin: bpy.props.IntProperty(name="Bake Margin", description="Pixels of padding around UV islands when baking", default=8, min=0, max=64)
    output_mode: bpy.props.EnumProperty(name="Result", description="New Object keeps your original; Replace overwrites its mesh", items=OUTPUT_ITEMS, default="NEW")
    apply_source_modifiers: bpy.props.BoolProperty(name="Apply Modifiers First", description="Bake the source's modifiers into the mesh before remeshing", default=True)
    keep_debug_objects: bpy.props.BoolProperty(name="Keep Debug Objects", description="Keep the intermediate meshes the solver creates, for troubleshooting", default=False)
    debug_logs: bpy.props.BoolProperty(name="Debug Console Logs", description="Print step-by-step details to the system console", default=False)

    hud_enabled: bpy.props.BoolProperty(
        name="HUD", description="Show the quad-quality dashboard in the 3D viewport (position and size: add-on Preferences)", default=True, update=_redraw_update
    )
    heatmap_enabled: bpy.props.BoolProperty(
        name="Heatmap",
        description="Colour triangles and n-gons on the active mesh and mark poles with dots (colours: add-on Preferences)",
        default=False,
        update=_redraw_update,
    )
    live_enabled: bpy.props.BoolProperty(
        name="Live",
        description=f"Keep a PairClosure quad preview beside the source that updates as you change settings (meshes up to {live.LIVE_FACE_CAP:,} faces)",
        default=False,
        update=_live_toggle_update,
    )
    batch_results: bpy.props.CollectionProperty(type=CURIOMESH_PG_batch_item)
    batch_index: bpy.props.IntProperty(default=0)
    batch_status: bpy.props.StringProperty(default="")
    batch_report_path: bpy.props.StringProperty(default="")

    metrics_faces: bpy.props.IntProperty(name="Faces", default=0)
    metrics_quads: bpy.props.IntProperty(name="Quads", default=0)
    metrics_tris: bpy.props.IntProperty(name="Triangles", default=0)
    metrics_ngons: bpy.props.IntProperty(name="N-gons", default=0)
    metrics_quad_ratio: bpy.props.FloatProperty(name="Quad Ratio", default=0.0, precision=3, subtype="FACTOR")
    metrics_face_error: bpy.props.FloatProperty(name="Target Error", default=0.0, precision=3, subtype="FACTOR")
    metrics_extraordinary: bpy.props.IntProperty(name="Extraordinary Verts", default=0)
    metrics_extraordinary_ratio: bpy.props.FloatProperty(name="Extraordinary Ratio", default=0.0, precision=3)
    metrics_non_manifold_edges: bpy.props.IntProperty(name="Non-Manifold Edges", default=0)
    metrics_non_manifold_vertices: bpy.props.IntProperty(name="Non-Manifold Vertices", default=0)
    metrics_duplicate_faces: bpy.props.IntProperty(name="Duplicate Faces", default=0)
    metrics_invalid_quads: bpy.props.IntProperty(name="Invalid Quads", default=0)
    metrics_components: bpy.props.IntProperty(name="Connected Components", default=0)
    metrics_component_euler: bpy.props.StringProperty(name="Component Euler Values", default="")
    metrics_boundary_edges: bpy.props.IntProperty(name="Boundary Edges", default=0)
    metrics_boundary_loops: bpy.props.IntProperty(name="Boundary Loops", default=0)
    metrics_euler: bpy.props.IntProperty(name="Euler Characteristic", default=0)
    metrics_degenerate: bpy.props.IntProperty(name="Degenerate Faces", default=0)
    metrics_min_jacobian: bpy.props.FloatProperty(name="Min Scaled Jacobian", default=0.0, precision=4)
    metrics_p95_aspect: bpy.props.FloatProperty(name="p95 Aspect Ratio", default=0.0, precision=3)
    metrics_quad_only: bpy.props.BoolProperty(name="Local Quad Manifold Valid", default=False)
    metrics_uv_valid: bpy.props.BoolProperty(name="UV Valid", default=False)
    metrics_materials_preserved: bpy.props.BoolProperty(name="Materials Preserved", default=False)
    metrics_elapsed_ms: bpy.props.FloatProperty(name="Elapsed ms", default=0.0, precision=1)
    metrics_engine: bpy.props.StringProperty(name="Engine", default="")
    metrics_strategy: bpy.props.StringProperty(name="Strategy", default="")
    metrics_profile: bpy.props.StringProperty(name="Scan Profile", default="")
    metrics_candidates: bpy.props.IntProperty(name="Candidates", default=0)
    metrics_geometry_rms: bpy.props.FloatProperty(name="RMS Error / Diagonal", default=0.0, precision=6)
    metrics_geometry_p95: bpy.props.FloatProperty(name="p95 Error / Diagonal", default=0.0, precision=6)
    metrics_geometry_max: bpy.props.FloatProperty(name="Max Sample Error / Diagonal", default=0.0, precision=6)
    metrics_sampling_status: bpy.props.StringProperty(name="Component Sampling", default="NOT_MEASURED")
    metrics_topology_status: bpy.props.StringProperty(name="Topology Proof", default="NOT_CHECKED")
    metrics_status: bpy.props.StringProperty(name="Status", default="")
    metrics_uv_status: bpy.props.StringProperty(name="UV Status", default="")
    metrics_message: bpy.props.StringProperty(name="Message", default="")

    analysis_profile: bpy.props.StringProperty(name="Profile", default="Not analyzed")
    analysis_strategy: bpy.props.StringProperty(name="Recommended", default="")
    analysis_faces: bpy.props.IntProperty(name="Input Faces", default=0)
    analysis_sampled_faces: bpy.props.IntProperty(name="Sampled Faces", default=0)
    analysis_exact: bpy.props.BoolProperty(name="Exact Topology", default=False)
    analysis_boundary_loops: bpy.props.IntProperty(name="Boundary Loops", default=-1)
    analysis_non_manifold: bpy.props.IntProperty(name="Non-Manifold Edges", default=-1)
    analysis_sharp_ratio: bpy.props.FloatProperty(name="Sharp Coverage", default=0.0, subtype="FACTOR")
    analysis_target_edge: bpy.props.FloatProperty(name="Target Edge Scale", default=0.0, precision=6)
    analysis_warning: bpy.props.StringProperty(name="Warning", default="")


def _fill_remesh_operator(op, settings) -> None:
    """Copy every shared setting onto the remesh operator (names match by design)."""
    for prop in op.bl_rna.properties:
        if prop.identifier != "rna_type" and prop.identifier in settings.bl_rna.properties:
            setattr(op, prop.identifier, getattr(settings, prop.identifier))


def _prefs():
    return overlay.get_prefs()


def _names(items) -> dict:
    return {item[0]: item[1] for item in items}


def _split(layout):
    """Label-left / value-right rows, the layout Blender's own settings panels use."""
    layout.use_property_split = True
    layout.use_property_decorate = False
    return layout


def _is_preview(obj) -> bool:
    return obj is not None and obj.name in (overlay.PREVIEW_NAME, region.PREVIEW_NAME)


# --------------------------------------------------------------------------- quick start


def _draw_result(layout, settings, compact: bool) -> None:
    """What the last remesh did, in plain words (the missing feedback of the old panel)."""
    status = settings.metrics_status
    if not status:
        if not compact:
            layout.label(text="No result yet. Press Remesh.", icon="DOT")
        return
    ok = status == "OK"
    box = layout.box()
    box.alert = not ok
    col = box.column(align=True)
    col.label(text="Last result" if ok else "Last attempt failed", icon="CHECKMARK" if ok else "ERROR")
    if not ok:
        overlay.wrap_label(col, settings.metrics_message or "The remesh did not finish.", width=40)
        return
    col.label(text=f"{settings.metrics_quad_ratio * 100:.1f}% quads   {settings.metrics_faces:,} faces")
    if not compact:
        leftovers = []
        if settings.metrics_tris:
            leftovers.append(f"{settings.metrics_tris:,} triangles")
        if settings.metrics_ngons:
            leftovers.append(f"{settings.metrics_ngons:,} n-gons")
        col.label(text=("Left over: " + ", ".join(leftovers)) if leftovers else "No triangles or n-gons left")
        col.label(text=f"Took {settings.metrics_elapsed_ms / 1000.0:.1f} s")
    if "UNVERIFIED" in settings.metrics_topology_status:
        col.alert = True
        col.label(text="Topology not verified (see Diagnostics)", icon="ERROR")


class CURIOMESH_PT_panel(bpy.types.Panel):
    bl_label = "Quick Start"
    bl_idname = "CURIOMESH_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"

    def draw_header(self, context: bpy.types.Context):
        self.layout.label(icon="MOD_REMESH")

    def draw_header_preset(self, context: bpy.types.Context):
        op = self.layout.operator("preferences.addon_show", text="", icon="PREFERENCES", emboss=False)
        op.module = __package__

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = context.scene.curiomesh_settings
        compact = bool(_prefs().compact_mode)
        obj = context.object
        has_mesh = obj is not None and obj.type == "MESH"
        usable = has_mesh and not _is_preview(obj)

        if not has_mesh and bpy.data.objects.get(region.PREVIEW_NAME) is not None:
            box = layout.box()
            box.label(text="Region preview in progress", icon="INFO")
            if not compact:
                box.label(text="Use the Regions panel to apply it.", icon="BLANK1")
        elif not has_mesh:
            box = layout.box()
            box.label(text="Select a mesh to begin", icon="INFO")
            if not compact:
                box.label(text="Click an object in the viewport.", icon="BLANK1")
        elif not usable:
            box = layout.box()
            box.label(text="This is a preview object", icon="INFO")
            if not compact:
                box.label(text="Click your own mesh to remesh it.", icon="BLANK1")

        # The one primary action.
        big = layout.column(align=True)
        big.enabled = usable
        big.scale_y = 1.6 if compact else 2.0
        text = "Remesh" if compact else f"Remesh to ~{settings.target_faces:,} faces"
        _fill_remesh_operator(big.operator("curiomesh.remesh", text=text, icon="MOD_REMESH"), settings)

        # Quality presets.
        if not compact:
            layout.label(text="Quality")
        presets = layout.row(align=True)
        presets.scale_y = 1.3
        for key, label, icon in (
            ("DRAFT", "Draft", "SORTTIME"),
            ("BALANCED", "Balanced", "SHADING_SOLID"),
            ("HERO", "Hero", "SOLO_ON"),
        ):
            op = presets.operator("curiomesh.apply_preset", text=label, icon=icon, depress=settings.quality == key)
            op.preset = key

        # Size and method.
        col = layout.column(align=True)
        col.prop(settings, "target_faces")
        row = col.split(factor=0.3, align=True)
        row.label(text="Method")
        row.prop(settings, "engine", text="")
        if settings.engine == "FLAT_QUADS":
            layout.prop(settings, "flat_tolerance")

        _draw_result(layout, settings, compact)

        # Visual helpers.
        if not compact:
            layout.label(text="See it")
        toggles = layout.row(align=True)
        toggles.scale_y = 1.2
        toggles.prop(settings, "hud_enabled", text="HUD", icon="OVERLAY", toggle=True)
        toggles.operator(
            "curiomesh.toggle_heatmap", text="Heatmap", icon="SHADING_RENDERED", depress=settings.heatmap_enabled
        )
        live_row = toggles.row(align=True)
        live_row.alert = settings.live_enabled and live.status().startswith(("Live off", "Live error"))
        live_row.prop(settings, "live_enabled", text="Live", icon="PLAY", toggle=True)

        if settings.heatmap_enabled:
            legend = layout.column(align=True)
            legend.use_property_split = True
            legend.use_property_decorate = False
            if compact:
                row = legend.row(align=True)
                for name in ("heat_tri_color", "heat_ngon_color", "heat_pole3_color", "heat_pole5_color"):
                    overlay.pref_swatch(row, name)
            else:
                overlay.pref_swatch(legend, "heat_tri_color", "Triangles")
                overlay.pref_swatch(legend, "heat_ngon_color", "N-gons")
                overlay.pref_swatch(legend, "heat_pole3_color", "3-edge pole")
                overlay.pref_swatch(legend, "heat_pole5_color", "5+ pole (big)")
        if settings.live_enabled:
            box = layout.box()
            box.label(text=live.status() or "Live: waiting for a change...", icon="TIME")
            if has_mesh and len(obj.data.polygons) > live.LIVE_FACE_CAP and obj.name != overlay.PREVIEW_NAME:
                box.label(text=f"Live needs <= {live.LIVE_FACE_CAP:,} faces", icon="ERROR")


# --------------------------------------------------------------------------- sub-panels


class _SubPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"
    bl_parent_id = "CURIOMESH_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    pref_key = ""

    @classmethod
    def poll(cls, context):
        return bool(getattr(_prefs(), cls.pref_key, True)) and cls.extra_poll(context)

    @classmethod
    def extra_poll(cls, context):
        return True


class CURIOMESH_PT_scan(_SubPanel, bpy.types.Panel):
    bl_label = "Scan"
    bl_idname = "CURIOMESH_PT_scan"
    pref_key = "show_scan"

    def draw_header(self, context):
        self.layout.label(icon="VIEWZOOM")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.curiomesh_settings
        compact = bool(_prefs().compact_mode)
        obj = context.object
        usable = obj is not None and obj.type == "MESH" and not _is_preview(obj)
        row = layout.row()
        row.enabled = usable
        row.scale_y = 1.3
        row.operator("curiomesh.analyze", text="Analyze Mesh", icon="VIEWZOOM")
        if not settings.analysis_faces:
            if not compact:
                layout.label(text="Checks holes, sharp edges and scale.")
                layout.label(text="Never changes your mesh.")
            return
        names = _names(ATLAS_STRATEGY_ITEMS)
        box = layout.box()
        col = box.column(align=True)
        col.label(text=f"{settings.analysis_profile}  ({settings.analysis_faces:,} faces)")
        col.label(text=f"Suggested: {names.get(settings.analysis_strategy, settings.analysis_strategy)}", icon="LIGHT")
        qualifier = "exact topology" if settings.analysis_exact else f"sampled {settings.analysis_sampled_faces:,} faces"
        col.label(text=f"Measured from {qualifier}")
        if settings.analysis_warning:
            warn = box.column(align=True)
            warn.alert = True
            for chunk in settings.analysis_warning.split(" | ")[:3]:
                overlay.wrap_label(warn, chunk, icon="ERROR", width=38, limit=2)


class CURIOMESH_PT_atlas(_SubPanel, bpy.types.Panel):
    bl_label = "Smart Solver (Atlas-Q)"
    bl_idname = "CURIOMESH_PT_atlas"
    pref_key = "show_atlas"

    @classmethod
    def extra_poll(cls, context):
        return context.scene.curiomesh_settings.engine in {"ATLAS_Q", "AUTO"}

    def draw_header(self, context):
        self.layout.label(icon="MOD_MESHDEFORM")

    def draw(self, context):
        layout = _split(self.layout)
        settings = context.scene.curiomesh_settings
        compact = bool(_prefs().compact_mode)
        layout.prop(settings, "atlas_strategy")
        layout.prop(settings, "topology_policy")
        layout.prop(settings, "require_quads")
        layout.prop(settings, "atlas_candidate_count")
        layout.prop(settings, "triad_feature_angle")
        if settings.topology_policy == "REPAIR":
            col = layout.column(align=True)
            col.alert = True
            overlay.wrap_label(col, "Repair may close holes or erase thin sheets.", icon="ERROR")
        if not compact:
            col = layout.column(align=True)
            if settings.atlas_strategy == "PAIR_CLOSURE":
                overlay.wrap_label(col, "All quads, but expect more poles and a looser face count.", icon="INFO")
            else:
                overlay.wrap_label(col, "Each attempt is scored on accuracy, shape, face count and poles. The best one wins.", icon="INFO")


class CURIOMESH_PT_shape(_SubPanel, bpy.types.Panel):
    bl_label = "Features to Keep"
    bl_idname = "CURIOMESH_PT_shape"
    pref_key = "show_shape"

    def draw_header(self, context):
        self.layout.label(icon="MOD_SMOOTH")

    def draw(self, context):
        layout = _split(self.layout)
        settings = context.scene.curiomesh_settings
        col = layout.column(heading="Keep")
        for name in ("preserve_sharp", "preserve_boundary", "preserve_seams", "use_symmetry"):
            col.prop(settings, name)
        layout.prop(settings, "smooth_normals")


class CURIOMESH_PT_pipeline(_SubPanel, bpy.types.Panel):
    bl_label = "Cleanup & Output"
    bl_idname = "CURIOMESH_PT_pipeline"
    pref_key = "show_pipeline"

    def draw_header(self, context):
        self.layout.label(icon="SETTINGS")

    def draw(self, context):
        layout = _split(self.layout)
        settings = context.scene.curiomesh_settings
        layout.prop(settings, "output_mode")
        layout.prop(settings, "apply_source_modifiers")
        layout.prop(settings, "cleanup_strength")
        layout.prop(settings, "voxel_repair")
        layout.prop(settings, "shrinkwrap_project")
        layout.prop(settings, "preserve_attributes")
        layout.prop(settings, "seed")


class CURIOMESH_PT_pairclosure(_SubPanel, bpy.types.Panel):
    bl_label = "PairClosure Options"
    bl_idname = "CURIOMESH_PT_pairclosure"
    pref_key = "show_pairclosure"

    @classmethod
    def extra_poll(cls, context):
        return context.scene.curiomesh_settings.engine == "TRIAD_Q_LITE"

    def draw_header(self, context):
        self.layout.label(icon="MESH_GRID")

    def draw(self, context):
        layout = _split(self.layout)
        settings = context.scene.curiomesh_settings
        if not _prefs().compact_mode:
            overlay.wrap_label(layout, "Pairs triangles into quads along the surface flow, then closes leftovers.", icon="INFO")
        layout.prop(settings, "triad_flow_mode")
        layout.prop(settings, "triad_field_mode")
        layout.prop(settings, "triad_seed_count")
        layout.prop(settings, "triad_feature_angle")
        layout.prop(settings, "require_quads")
        if not settings.require_quads:
            col = layout.column(align=True)
            col.alert = True
            overlay.wrap_label(col, "Pure Quads is off: triangles may be kept.", icon="ERROR")
        rescued = context.object.get("curiomesh_triadq_rescued_pairs") if context.object else None
        if rescued is not None:
            layout.label(text=f"Rescued pairs: {rescued}", icon="CHECKMARK")


class CURIOMESH_PT_materials(_SubPanel, bpy.types.Panel):
    bl_label = "Materials & UVs"
    bl_idname = "CURIOMESH_PT_materials"
    pref_key = "show_materials"

    def draw_header(self, context):
        self.layout.label(icon="MATERIAL")

    def draw(self, context):
        layout = _split(self.layout)
        settings = context.scene.curiomesh_settings
        layout.prop(settings, "texture_mode")
        layout.prop(settings, "bake_fallback")
        if settings.texture_mode == "BAKE" or settings.bake_fallback:
            layout.prop(settings, "bake_resolution")
            layout.prop(settings, "bake_margin")


class CURIOMESH_UL_batch(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.alert = not item.ok
        row.label(text=item.name, icon="CHECKMARK" if item.ok else "ERROR")
        if item.ok:
            row.label(text=f"{item.quad_ratio * 100:.0f}% quads  {item.faces:,}f")
        else:
            row.label(text=("Failed: " + item.error)[:26] if item.error else "Failed")


class CURIOMESH_PT_batch(_SubPanel, bpy.types.Panel):
    bl_label = "Batch"
    bl_idname = "CURIOMESH_PT_batch"
    pref_key = "show_batch"

    def draw_header(self, context):
        self.layout.label(icon="OUTLINER_COLLECTION")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.curiomesh_settings
        count = len([o for o in context.selected_objects if o.type == "MESH"])
        row = layout.row()
        row.enabled = count > 0
        row.scale_y = 1.4
        row.operator("curiomesh.batch_remesh", text=f"Remesh {count} Selected", icon="OUTLINER_COLLECTION")
        if count == 0 and not settings.batch_results:
            layout.label(text="Select several meshes, then press the button.")
        if settings.batch_status:
            layout.label(text=settings.batch_status)
        if settings.batch_results:
            layout.template_list(
                "CURIOMESH_UL_batch", "", settings, "batch_results", settings, "batch_index", rows=3
            )
            if 0 <= settings.batch_index < len(settings.batch_results):
                item = settings.batch_results[settings.batch_index]
                col = layout.column(align=True)
                col.label(text=f"{item.name} -> {item.output or '-'}")
                col.label(text=f"Triangles {item.tris}   Poles {item.poles}   {item.ms / 1000.0:.1f} s")
                if item.error:
                    col.alert = True
                    overlay.wrap_label(col, item.error, icon="ERROR", width=40, limit=2)
            if settings.batch_report_path:
                layout.label(text=settings.batch_report_path[-48:], icon="FILE_TEXT")


SUMMARY_PROPS = (
    "metrics_status",
    "metrics_quad_ratio",
    "metrics_faces",
    "metrics_face_error",
    "metrics_topology_status",
)

DETAIL_PROPS = (
    "metrics_engine",
    "metrics_strategy",
    "metrics_profile",
    "metrics_candidates",
    "metrics_quad_only",
    "metrics_geometry_rms",
    "metrics_geometry_p95",
    "metrics_geometry_max",
    "metrics_sampling_status",
    "metrics_min_jacobian",
    "metrics_p95_aspect",
    "metrics_extraordinary",
    "metrics_non_manifold_edges",
    "metrics_non_manifold_vertices",
    "metrics_duplicate_faces",
    "metrics_invalid_quads",
    "metrics_components",
    "metrics_component_euler",
    "metrics_boundary_edges",
    "metrics_boundary_loops",
    "metrics_euler",
    "metrics_degenerate",
    "metrics_uv_valid",
    "metrics_materials_preserved",
    "metrics_uv_status",
    "metrics_elapsed_ms",
)


class CURIOMESH_PT_diagnostics(_SubPanel, bpy.types.Panel):
    bl_label = "Diagnostics"
    bl_idname = "CURIOMESH_PT_diagnostics"
    pref_key = "show_diagnostics"

    def draw_header(self, context):
        self.layout.label(icon="GRAPH")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.curiomesh_settings
        obj = context.object
        row = layout.row()
        row.enabled = obj is not None and obj.type == "MESH"
        row.scale_y = 1.3
        row.operator("curiomesh.compute_metrics", text="Measure Active Mesh", icon="FILE_REFRESH")
        if not settings.metrics_status:
            layout.label(text="Nothing measured yet.")
            return
        col = _split(layout.column(align=True))
        topology_row = col.row()
        topology_row.alert = "UNVERIFIED" in settings.metrics_topology_status
        for name in SUMMARY_PROPS:
            target = topology_row if name == "metrics_topology_status" else col
            target.prop(settings, name, emboss=False)
        if settings.metrics_message:
            overlay.wrap_label(layout, settings.metrics_message, icon="INFO", width=40, limit=3)


class CURIOMESH_PT_diagnostics_all(bpy.types.Panel):
    bl_label = "All Measurements"
    bl_idname = "CURIOMESH_PT_diagnostics_all"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"
    bl_parent_id = "CURIOMESH_PT_diagnostics"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.curiomesh_settings.metrics_status)

    def draw(self, context):
        settings = context.scene.curiomesh_settings
        col = _split(self.layout.column(align=True))
        for name in DETAIL_PROPS:
            col.prop(settings, name, emboss=False)


class CURIOMESH_PT_debug(_SubPanel, bpy.types.Panel):
    bl_label = "Debug"
    bl_idname = "CURIOMESH_PT_debug"
    pref_key = "show_debug"

    def draw_header(self, context):
        self.layout.label(icon="CONSOLE")

    def draw(self, context):
        layout = _split(self.layout)
        settings = context.scene.curiomesh_settings
        layout.prop(settings, "debug_logs")
        layout.prop(settings, "keep_debug_objects")


PANEL_CLASSES = (
    CURIOMESH_PT_panel,
    CURIOMESH_PT_scan,
    CURIOMESH_PT_atlas,
    CURIOMESH_PT_shape,
    CURIOMESH_PT_pipeline,
    CURIOMESH_PT_pairclosure,
    CURIOMESH_PT_materials,
    CURIOMESH_UL_batch,
    CURIOMESH_PT_batch,
    CURIOMESH_PT_diagnostics,
    CURIOMESH_PT_diagnostics_all,
    CURIOMESH_PT_debug,
)


def panel_classes() -> list:
    """Every CurioMesh sidebar panel in display order (Quick Start and its folds, then Regions)."""
    return [cls for cls in (*PANEL_CLASSES, *region.PANEL_CLASSES, *guided.PANEL_CLASSES) if issubclass(cls, bpy.types.Panel)]


def apply_category(name: str | None = None) -> str:
    """Move every CurioMesh panel to the sidebar tab `name` (default: the preference)."""
    name = (name if name is not None else _prefs().panel_tab or "").strip() or overlay.PREF_DEFAULTS["panel_tab"]
    panels = panel_classes()
    if all(cls.bl_category == name for cls in panels):
        return name
    for cls in reversed(panels):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    for cls in panels:
        cls.bl_category = name
        bpy.utils.register_class(cls)
    return name
