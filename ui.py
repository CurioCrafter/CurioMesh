from __future__ import annotations

import bpy

from .operators import OUTPUT_ITEMS, QUALITY_ITEMS, TEXTURE_ITEMS


class CURIOMESH_PG_settings(bpy.types.PropertyGroup):
    target_faces: bpy.props.IntProperty(name="Target Faces", default=8000, min=4, soft_max=250000)
    quality: bpy.props.EnumProperty(name="Quality", items=QUALITY_ITEMS, default="BALANCED")
    seed: bpy.props.IntProperty(name="Seed", default=0, min=0, soft_max=100000)
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
    metrics_boundary_edges: bpy.props.IntProperty(name="Boundary Edges", default=0)
    metrics_uv_valid: bpy.props.BoolProperty(name="UV Valid", default=False)
    metrics_materials_preserved: bpy.props.BoolProperty(name="Materials Preserved", default=False)
    metrics_elapsed_ms: bpy.props.FloatProperty(name="Elapsed ms", default=0.0, precision=1)
    metrics_engine: bpy.props.StringProperty(name="Engine", default="")
    metrics_status: bpy.props.StringProperty(name="Status", default="")
    metrics_uv_status: bpy.props.StringProperty(name="UV Status", default="")
    metrics_message: bpy.props.StringProperty(name="Message", default="")


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
        main.prop(settings, "quality")
        row = main.row(align=True)
        row.operator("curiomesh.apply_preset", text="Draft").preset = "DRAFT"
        row.operator("curiomesh.apply_preset", text="Balanced").preset = "BALANCED"
        row.operator("curiomesh.apply_preset", text="Hero").preset = "HERO"

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
        op.quality = settings.quality
        op.seed = settings.seed
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
        col.prop(settings, "metrics_faces")
        col.prop(settings, "metrics_quad_ratio")
        col.prop(settings, "metrics_face_error")
        col.prop(settings, "metrics_extraordinary")
        col.prop(settings, "metrics_non_manifold_edges")
        col.prop(settings, "metrics_boundary_edges")
        col.prop(settings, "metrics_uv_valid")
        col.prop(settings, "metrics_materials_preserved")
        col.prop(settings, "metrics_uv_status")
        col.prop(settings, "metrics_elapsed_ms")
        if settings.metrics_message:
            col.label(text=settings.metrics_message[:80])
