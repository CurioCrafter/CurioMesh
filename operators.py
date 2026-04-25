from __future__ import annotations

import bpy

from .bridge import RemeshConfig, RemeshResult, run_curiomesh
from .metrics import mesh_diagnostics


QUALITY_ITEMS = [
    ("DRAFT", "Draft", "Fast preview remesh with light cleanup"),
    ("BALANCED", "Balanced", "General-purpose quality and speed"),
    ("HERO", "Hero", "Higher-quality preservation and validation"),
]

ENGINE_ITEMS = [
    ("QUADRIFLOW", "QuadriFlow", "Production path using Blender's built-in QuadriFlow remesher"),
    ("TRIAD_Q_LITE", "TRIAD-Q Lite", "Experimental standalone CurioMesh quad-dominant engine"),
    ("AUTO", "Auto", "Use the strongest available production engine"),
]

FLOW_ITEMS = [
    ("AUTO", "Auto", "Classify the mesh and choose a TRIAD-Q mode"),
    ("BALANCED", "Balanced", "General feature-aware pairing"),
    ("ORGANIC", "OrganicFlow", "Prioritize smooth organic flow"),
    ("PATCH", "PatchFlow", "Prioritize hard features and patch boundaries"),
    ("DIRTY", "DirtyFlow", "Prefer robustness on messy inputs"),
    ("TEXTURE", "TextureFlow", "Favor material and UV seam preservation"),
]

TEXTURE_ITEMS = [
    ("PROJECT", "Project UVs", "Transfer UVs, then use BVH projection if needed"),
    ("TRANSFER", "Transfer Only", "Use Blender data transfer only"),
    ("BAKE", "Bake Color", "Bake base color from source to result"),
    ("NONE", "None", "Skip UV and texture preservation"),
]

OUTPUT_ITEMS = [
    ("NEW", "New Object", "Create a remeshed object and keep the original"),
    ("REPLACE", "Replace", "Replace the selected object's mesh data"),
]


def config_from_owner(owner: object) -> RemeshConfig:
    return RemeshConfig(
        target_faces=int(owner.target_faces),
        engine=str(owner.engine),
        quality=str(owner.quality),
        seed=int(owner.seed),
        triad_seed_count=int(owner.triad_seed_count),
        triad_feature_angle=float(owner.triad_feature_angle),
        triad_force_quads=bool(owner.triad_force_quads),
        triad_flow_mode=str(owner.triad_flow_mode),
        preserve_sharp=bool(owner.preserve_sharp),
        preserve_boundary=bool(owner.preserve_boundary),
        preserve_seams=bool(owner.preserve_seams),
        use_symmetry=bool(owner.use_symmetry),
        preserve_attributes=bool(owner.preserve_attributes),
        smooth_normals=bool(owner.smooth_normals),
        cleanup_strength=float(owner.cleanup_strength),
        voxel_repair=bool(owner.voxel_repair),
        shrinkwrap_project=bool(owner.shrinkwrap_project),
        texture_mode=str(owner.texture_mode),
        bake_fallback=bool(owner.bake_fallback),
        bake_resolution=int(owner.bake_resolution),
        bake_margin=int(owner.bake_margin),
        output_mode=str(owner.output_mode),
        apply_source_modifiers=bool(owner.apply_source_modifiers),
        keep_debug_objects=bool(owner.keep_debug_objects),
        debug_logs=bool(owner.debug_logs),
    )


def apply_result_to_settings(settings: bpy.types.PropertyGroup, result: RemeshResult) -> None:
    diagnostics = result.diagnostics
    settings.metrics_faces = int(diagnostics.faces)
    settings.metrics_quads = int(diagnostics.quads)
    settings.metrics_tris = int(diagnostics.tris)
    settings.metrics_ngons = int(diagnostics.ngons)
    settings.metrics_quad_ratio = float(diagnostics.quad_ratio)
    settings.metrics_face_error = float(diagnostics.face_count_error)
    settings.metrics_extraordinary = int(diagnostics.extraordinary)
    settings.metrics_extraordinary_ratio = float(diagnostics.extraordinary_ratio)
    settings.metrics_non_manifold_edges = int(diagnostics.non_manifold_edges)
    settings.metrics_boundary_edges = int(diagnostics.boundary_edges)
    settings.metrics_uv_valid = bool(diagnostics.uv_valid)
    settings.metrics_materials_preserved = bool(result.materials_preserved)
    settings.metrics_elapsed_ms = float(result.elapsed_ms)
    settings.metrics_engine = str(result.engine)
    settings.metrics_status = "OK" if result.success else "FAILED"
    settings.metrics_uv_status = str(result.uv_status)
    settings.metrics_message = str(result.message)


class CURIOMESH_OT_remesh(bpy.types.Operator):
    bl_idname = "curiomesh.remesh"
    bl_label = "CurioMesh Remesh"
    bl_description = "Run CurioMesh's QuadriFlow retopology pipeline"
    bl_options = {"REGISTER", "UNDO"}

    target_faces: bpy.props.IntProperty(name="Target Faces", default=8000, min=4, soft_max=250000)
    engine: bpy.props.EnumProperty(name="Engine", items=ENGINE_ITEMS, default="QUADRIFLOW")
    quality: bpy.props.EnumProperty(name="Quality", items=QUALITY_ITEMS, default="BALANCED")
    seed: bpy.props.IntProperty(name="Seed", default=0, min=0, soft_max=100000)
    triad_seed_count: bpy.props.IntProperty(name="TRIAD-Q Seeds", default=6, min=1, max=32)
    triad_feature_angle: bpy.props.FloatProperty(name="TRIAD-Q Feature Angle", default=35.0, min=0.0, max=180.0)
    triad_force_quads: bpy.props.BoolProperty(name="TRIAD-Q Pure Quads", default=False)
    triad_flow_mode: bpy.props.EnumProperty(name="TRIAD-Q Flow", items=FLOW_ITEMS, default="AUTO")
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

    def execute(self, context: bpy.types.Context):
        obj = context.object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object.")
            return {"CANCELLED"}

        result = run_curiomesh(obj, config_from_owner(self))
        settings = context.scene.curiomesh_settings
        apply_result_to_settings(settings, result)

        if not result.success:
            self.report({"ERROR"}, result.message or "CurioMesh remesh failed.")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            (
                f"CurioMesh: {result.diagnostics.faces} faces, "
                f"{result.diagnostics.quad_ratio:.1%} quads, UV {result.uv_status.lower()}"
            ),
        )
        return {"FINISHED"}


class CURIOMESH_OT_apply_preset(bpy.types.Operator):
    bl_idname = "curiomesh.apply_preset"
    bl_label = "Apply CurioMesh Preset"
    bl_options = {"INTERNAL"}

    preset: bpy.props.EnumProperty(name="Preset", items=QUALITY_ITEMS, default="BALANCED")

    def execute(self, context: bpy.types.Context):
        settings = context.scene.curiomesh_settings
        preset = str(self.preset)
        settings.quality = preset

        if preset == "DRAFT":
            settings.target_faces = max(250, settings.target_faces // 2)
            settings.cleanup_strength = 0.15
            settings.shrinkwrap_project = False
            settings.texture_mode = "TRANSFER"
            settings.bake_fallback = False
            settings.smooth_normals = True
        elif preset == "HERO":
            settings.cleanup_strength = 0.35
            settings.shrinkwrap_project = True
            settings.texture_mode = "PROJECT"
            settings.bake_fallback = True
            settings.preserve_sharp = True
            settings.preserve_boundary = True
            settings.preserve_attributes = True
            settings.smooth_normals = True
        else:
            settings.cleanup_strength = 0.25
            settings.shrinkwrap_project = True
            settings.texture_mode = "PROJECT"
            settings.bake_fallback = False
            settings.preserve_sharp = True
            settings.preserve_boundary = True
            settings.preserve_attributes = True
            settings.smooth_normals = True

        return {"FINISHED"}


class CURIOMESH_OT_compute_metrics(bpy.types.Operator):
    bl_idname = "curiomesh.compute_metrics"
    bl_label = "Compute CurioMesh Metrics"
    bl_options = {"INTERNAL"}

    def execute(self, context: bpy.types.Context):
        obj = context.object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object.")
            return {"CANCELLED"}

        settings = context.scene.curiomesh_settings
        diagnostics = mesh_diagnostics(obj, settings.target_faces)
        result = RemeshResult(
            True,
            output_object=obj,
            diagnostics=diagnostics,
            engine="METRICS_ONLY",
            elapsed_ms=0.0,
            message="Metrics computed.",
            materials_preserved=bool(obj.data.materials),
            uv_status="VALID" if diagnostics.uv_valid else "INVALID",
        )
        apply_result_to_settings(settings, result)
        return {"FINISHED"}
