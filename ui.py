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


class CURIOMESH_PG_settings(bpy.types.PropertyGroup):
    target_faces: bpy.props.IntProperty(name="Target Faces", default=8000, min=4, soft_max=250000)
    engine: bpy.props.EnumProperty(name="Engine", items=ENGINE_ITEMS, default="ATLAS_Q")
    quality: bpy.props.EnumProperty(name="Quality", items=QUALITY_ITEMS, default="BALANCED")
    seed: bpy.props.IntProperty(name="Seed", default=0, min=0, soft_max=100000)
    triad_seed_count: bpy.props.IntProperty(name="TRIAD-Q Seeds", default=6, min=1, max=32)
    triad_feature_angle: bpy.props.FloatProperty(name="TRIAD-Q Feature Angle", default=35.0, min=0.0, max=180.0)
    triad_force_quads: bpy.props.BoolProperty(name="TRIAD-Q Pure Quads", default=False)
    triad_flow_mode: bpy.props.EnumProperty(name="TRIAD-Q Flow", items=FLOW_ITEMS, default="AUTO")
    atlas_strategy: bpy.props.EnumProperty(name="Scan Strategy", items=ATLAS_STRATEGY_ITEMS, default="AUTO")
    atlas_candidate_count: bpy.props.IntProperty(name="Candidate Budget", default=3, min=1, max=8)
    topology_policy: bpy.props.EnumProperty(name="Topology", items=TOPOLOGY_ITEMS, default="PRESERVE")
    require_quads: bpy.props.BoolProperty(name="Require Locally Valid Quads", default=True)
    analysis_sample_faces: bpy.props.IntProperty(
        name="Analysis Sample Faces",
        default=120000,
        min=1000,
        max=500000,
    )
    preserve_sharp: bpy.props.BoolProperty(name="Preserve Sharp", default=True)
    preserve_boundary: bpy.props.BoolProperty(name="Preserve Boundary", default=True)
    preserve_seams: bpy.props.BoolProperty(name="Treat UV Seams As Sharp", default=True)
    use_symmetry: bpy.props.BoolProperty(name="Use Mesh Symmetry", default=False)
    preserve_attributes: bpy.props.BoolProperty(name="Preserve Attributes", default=True)
    smooth_normals: bpy.props.BoolProperty(name="Smooth Normals", default=True)
    cleanup_strength: bpy.props.FloatProperty(name="Cleanup Strength", default=0.25, min=0.0, max=1.0)
    voxel_repair: bpy.props.BoolProperty(name="Voxel Repair Fallback", default=True)
    shrinkwrap_project: bpy.props.BoolProperty(name="Project Details", default=True)
    texture_mode: bpy.props.EnumProperty(name="Texture Preservation", items=TEXTURE_ITEMS, default="PROJECT")
    bake_fallback: bpy.props.BoolProperty(name="Bake If UVs Fail", default=False)
    bake_resolution: bpy.props.IntProperty(name="Bake Resolution", default=2048, min=64, soft_max=8192)
    bake_margin: bpy.props.IntProperty(name="Bake Margin", default=8, min=0, max=64)
    output_mode: bpy.props.EnumProperty(name="Output", items=OUTPUT_ITEMS, default="NEW")
    apply_source_modifiers: bpy.props.BoolProperty(name="Apply Source Modifiers", default=True)
    keep_debug_objects: bpy.props.BoolProperty(name="Keep Debug Objects", default=False)
    debug_logs: bpy.props.BoolProperty(name="Debug Console Logs", default=False)

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


class CURIOMESH_PT_panel(bpy.types.Panel):
    bl_label = "CurioMesh"
    bl_idname = "CURIOMESH_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"

    @classmethod
    def poll(cls, context: bpy.types.Context):
        obj = context.object
        return obj is not None and obj.type == "MESH"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = context.scene.curiomesh_settings

        main = layout.column(align=True)
        main.prop(settings, "target_faces")
        main.prop(settings, "engine")
        main.prop(settings, "quality")
        row = main.row(align=True)
        row.operator("curiomesh.apply_preset", text="Draft").preset = "DRAFT"
        row.operator("curiomesh.apply_preset", text="Balanced").preset = "BALANCED"
        row.operator("curiomesh.apply_preset", text="Hero").preset = "HERO"

        scan = layout.box()
        scan.label(text="1. Scan Intelligence")
        scan.operator("curiomesh.analyze", text="Analyze Without Changing Mesh", icon="VIEWZOOM")
        if settings.analysis_faces:
            scan.label(text=f"{settings.analysis_profile}: {settings.analysis_faces:,} input faces")
            scan.label(text=f"Recommended: {settings.analysis_strategy}")
            qualifier = "exact topology" if settings.analysis_exact else f"sampled {settings.analysis_sampled_faces:,} faces"
            scan.label(text=qualifier)
            if settings.analysis_warning:
                scan.label(text=settings.analysis_warning[:100], icon="ERROR")

        if settings.engine in {"ATLAS_Q", "AUTO"}:
            atlas = layout.box()
            atlas.label(text="2. Atlas-Q Solver Portfolio")
            atlas.prop(settings, "atlas_strategy")
            atlas.prop(settings, "topology_policy")
            atlas.prop(settings, "require_quads")
            atlas.prop(settings, "atlas_candidate_count")
            atlas.prop(settings, "triad_feature_angle", text="Feature Angle")
            if settings.topology_policy == "REPAIR":
                atlas.label(text="Repair may close holes or erase thin sheets.", icon="ERROR")
            if settings.atlas_strategy == "PAIR_CLOSURE":
                atlas.label(text="Exact quads; expect more poles and a looser face target.")
            else:
                atlas.label(text="Candidates are scored by error, Jacobian, aspect, count, and poles.")

        preserve = layout.box()
        preserve.label(text="Shape Preservation")
        preserve.prop(settings, "preserve_sharp")
        preserve.prop(settings, "preserve_boundary")
        preserve.prop(settings, "preserve_seams")
        preserve.prop(settings, "use_symmetry")
        preserve.prop(settings, "smooth_normals")

        pipeline = layout.box()
        pipeline.label(text="Pipeline")
        pipeline.prop(settings, "cleanup_strength")
        pipeline.prop(settings, "voxel_repair")
        pipeline.prop(settings, "shrinkwrap_project")
        pipeline.prop(settings, "preserve_attributes")
        pipeline.prop(settings, "apply_source_modifiers")
        pipeline.prop(settings, "output_mode")
        pipeline.prop(settings, "seed")

        if settings.engine == "TRIAD_Q_LITE":
            triad = layout.box()
            triad.label(text="PairClosure Standalone")
            triad.prop(settings, "triad_flow_mode")
            triad.prop(settings, "triad_seed_count")
            triad.prop(settings, "triad_feature_angle")
            triad.prop(settings, "require_quads")
            if not settings.require_quads:
                triad.label(text="Debug mode may retain triangles.", icon="ERROR")
            rescued = context.object.get("curiomesh_triadq_rescued_pairs")
            if rescued is not None:
                triad.label(text=f"Rescued pairs: {rescued}", icon="CHECKMARK")

        textures = layout.box()
        textures.label(text="Materials And UVs")
        textures.prop(settings, "texture_mode")
        if settings.texture_mode == "BAKE" or settings.bake_fallback:
            row = textures.row(align=True)
            row.prop(settings, "bake_resolution")
            row.prop(settings, "bake_margin")
        textures.prop(settings, "bake_fallback")

        debug = layout.box()
        debug.label(text="Debug")
        debug.prop(settings, "debug_logs")
        debug.prop(settings, "keep_debug_objects")

        op = layout.operator("curiomesh.remesh", text="Remesh")
        op.target_faces = settings.target_faces
        op.engine = settings.engine
        op.quality = settings.quality
        op.seed = settings.seed
        op.triad_seed_count = settings.triad_seed_count
        op.triad_feature_angle = settings.triad_feature_angle
        op.triad_force_quads = settings.triad_force_quads
        op.triad_flow_mode = settings.triad_flow_mode
        op.atlas_strategy = settings.atlas_strategy
        op.atlas_candidate_count = settings.atlas_candidate_count
        op.topology_policy = settings.topology_policy
        op.require_quads = settings.require_quads
        op.analysis_sample_faces = settings.analysis_sample_faces
        op.preserve_sharp = settings.preserve_sharp
        op.preserve_boundary = settings.preserve_boundary
        op.preserve_seams = settings.preserve_seams
        op.use_symmetry = settings.use_symmetry
        op.preserve_attributes = settings.preserve_attributes
        op.smooth_normals = settings.smooth_normals
        op.cleanup_strength = settings.cleanup_strength
        op.voxel_repair = settings.voxel_repair
        op.shrinkwrap_project = settings.shrinkwrap_project
        op.texture_mode = settings.texture_mode
        op.bake_fallback = settings.bake_fallback
        op.bake_resolution = settings.bake_resolution
        op.bake_margin = settings.bake_margin
        op.output_mode = settings.output_mode
        op.apply_source_modifiers = settings.apply_source_modifiers
        op.keep_debug_objects = settings.keep_debug_objects
        op.debug_logs = settings.debug_logs

        metrics = layout.box()
        metrics.label(text="Diagnostics")
        metrics.operator("curiomesh.compute_metrics", text="Compute Current Mesh")
        col = metrics.column(align=True)
        col.prop(settings, "metrics_status")
        col.prop(settings, "metrics_engine")
        col.prop(settings, "metrics_strategy")
        col.prop(settings, "metrics_profile")
        topology_row = col.row()
        topology_row.alert = "UNVERIFIED" in settings.metrics_topology_status
        topology_row.prop(settings, "metrics_topology_status")
        col.prop(settings, "metrics_candidates")
        col.prop(settings, "metrics_faces")
        col.prop(settings, "metrics_quad_ratio")
        col.prop(settings, "metrics_quad_only")
        col.prop(settings, "metrics_face_error")
        col.prop(settings, "metrics_geometry_rms")
        col.prop(settings, "metrics_geometry_p95")
        col.prop(settings, "metrics_geometry_max")
        col.prop(settings, "metrics_sampling_status")
        col.prop(settings, "metrics_min_jacobian")
        col.prop(settings, "metrics_p95_aspect")
        col.prop(settings, "metrics_extraordinary")
        col.prop(settings, "metrics_non_manifold_edges")
        col.prop(settings, "metrics_non_manifold_vertices")
        col.prop(settings, "metrics_duplicate_faces")
        col.prop(settings, "metrics_invalid_quads")
        col.prop(settings, "metrics_components")
        col.prop(settings, "metrics_component_euler")
        col.prop(settings, "metrics_boundary_edges")
        col.prop(settings, "metrics_boundary_loops")
        col.prop(settings, "metrics_euler")
        col.prop(settings, "metrics_degenerate")
        col.prop(settings, "metrics_uv_valid")
        col.prop(settings, "metrics_materials_preserved")
        col.prop(settings, "metrics_uv_status")
        col.prop(settings, "metrics_elapsed_ms")
        if settings.metrics_message:
            col.label(text=settings.metrics_message[:80])
