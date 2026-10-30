# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

import bpy

from .atlasq import analyze_object
from .bridge import RemeshConfig, RemeshResult, run_curiomesh
from .metrics import mesh_diagnostics


QUALITY_ITEMS = [
    ("DRAFT", "Draft", "Fast preview remesh with light cleanup. Halves the face target"),
    ("BALANCED", "Balanced", "Good quality at a sensible speed. The right choice most of the time"),
    ("HERO", "Hero", "Slowest, tries more candidates and cleans up harder. Use for final assets"),
]

ENGINE_ITEMS = [
    (
        "ATLAS_Q",
        "Smart (Atlas-Q)",
        "Recommended. Atlas-Q measures your mesh, builds a simplified copy at the target size, tries several solvers and keeps the best result that passes the quad checks",
    ),
    ("QUADRIFLOW", "QuadriFlow", "Blender's built-in QuadriFlow remesher. Fast and tidy on clean, closed meshes"),
    (
        "TRIAD_Q_LITE",
        "PairClosure (pure quads)",
        "Pairs triangles into quads along the surface flow, then closes any leftovers so every face is a quad. Always finishes, but makes more poles",
    ),
    (
        "FLAT_QUADS",
        "Flat Quads (huge quads on flat areas)",
        "Turns flat areas into one big quad each and spends the faces on detail instead. Best on ships, hard-surface props and architecture. The target limits what's left after the flat areas: a flat plate becomes one quad whatever the target",
    ),
    ("AUTO", "Auto (same as Atlas-Q)", "Older name for Smart: Atlas-Q, kept so old files still work"),
]

FLOW_ITEMS = [
    ("AUTO", "Auto", "Look at the mesh and choose a style"),
    ("BALANCED", "Balanced", "General purpose pairing that follows sharp features"),
    ("ORGANIC", "Organic", "Favour smooth, flowing edge loops (characters, creatures, scans)"),
    ("PATCH", "Hard Surface", "Favour crisp features and patch borders (machines, props)"),
    ("DIRTY", "Messy Scan", "Favour robustness on noisy or broken input"),
    ("TEXTURE", "Textured", "Favour keeping material and UV seam borders intact"),
]

ATLAS_STRATEGY_ITEMS = [
    ("AUTO", "Auto", "Choose from the measured topology, borders, sharp features, scale and texture state"),
    ("FLOW_ENSEMBLE", "Organic / General", "Simplify to the target size, then compare several QuadriFlow candidates"),
    ("FEATURE_FLOW", "Hard Surface / Features", "Keep persistent sharp edges, UV seams and material borders"),
    ("DIRTY_FLOW", "Dirty Closed Scan", "Allow voxel repair before remeshing a leaky or noisy scan"),
    ("OPEN_FLOW", "Open Fragment", "Keep intentional open borders and never close the mesh with voxels"),
    (
        "PAIR_CLOSURE",
        "PairClosure (pure quads)",
        "Reliable all-quad fallback. Splits bow-tie corners safely, but makes more poles and a looser face count",
    ),
]

TOPOLOGY_ITEMS = [
    (
        "PRESERVE",
        "Preserve",
        "Reject any result that changes the number of holes, open borders or separate parts, or whether the mesh is closed",
    ),
    ("REPAIR", "Repair", "Allow rebuilding of dirty closed scans. Small holes and thin sheets may change"),
]

TEXTURE_ITEMS = [
    ("PROJECT", "Project UVs", "Copy UVs across, then project from the original surface where needed"),
    ("TRANSFER", "Transfer Only", "Use Blender's data transfer only (fastest)"),
    ("BAKE", "Bake Colour", "Bake the base colour of the source onto the result"),
    ("NONE", "None", "Skip UVs and textures"),
]

OUTPUT_ITEMS = [
    ("NEW", "New Object", "Create a remeshed object and keep the original"),
    ("REPLACE", "Replace", "Overwrite the selected object's mesh"),
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
        triad_field_mode=str(getattr(owner, "triad_field_mode", "SMOOTH")),
        flat_tolerance=float(getattr(owner, "flat_tolerance", 0.1)),
        atlas_strategy=str(owner.atlas_strategy),
        atlas_candidate_count=int(owner.atlas_candidate_count),
        topology_policy=str(owner.topology_policy),
        require_quads=bool(owner.require_quads),
        analysis_sample_faces=int(owner.analysis_sample_faces),
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
    settings.metrics_non_manifold_vertices = int(diagnostics.non_manifold_vertices)
    settings.metrics_duplicate_faces = int(diagnostics.duplicate_faces)
    settings.metrics_invalid_quads = int(diagnostics.invalid_quads)
    settings.metrics_components = int(diagnostics.connected_components)
    settings.metrics_component_euler = ", ".join(str(value) for value in diagnostics.component_euler)
    settings.metrics_boundary_edges = int(diagnostics.boundary_edges)
    settings.metrics_boundary_loops = int(diagnostics.boundary_loops)
    settings.metrics_euler = int(diagnostics.euler_characteristic)
    settings.metrics_degenerate = int(diagnostics.degenerate_faces)
    settings.metrics_min_jacobian = float(diagnostics.min_scaled_jacobian)
    settings.metrics_p95_aspect = float(diagnostics.p95_aspect_ratio)
    settings.metrics_quad_only = bool(result.quad_only_verified)
    settings.metrics_uv_valid = bool(diagnostics.uv_valid)
    settings.metrics_materials_preserved = bool(result.materials_preserved)
    settings.metrics_elapsed_ms = float(result.elapsed_ms)
    settings.metrics_engine = str(result.engine)
    settings.metrics_strategy = str(result.strategy)
    settings.metrics_profile = str(result.profile)
    settings.metrics_candidates = int(result.candidate_count)
    settings.metrics_geometry_rms = float(result.geometry_rms)
    settings.metrics_geometry_p95 = float(result.geometry_p95)
    settings.metrics_geometry_max = float(result.geometry_max)
    settings.metrics_sampling_status = str(result.sampling_status)
    settings.metrics_topology_status = str(result.topology_status)
    settings.metrics_status = "OK" if result.success else "FAILED"
    settings.metrics_uv_status = str(result.uv_status)
    settings.metrics_message = str(result.message)


class CURIOMESH_OT_remesh(bpy.types.Operator):
    bl_idname = "curiomesh.remesh"
    bl_label = "Remesh to Quads"
    bl_description = "Turn the active mesh into clean quads at the target face count. Undo restores the original"
    bl_options = {"REGISTER", "UNDO"}

    target_faces: bpy.props.IntProperty(name="Target Faces", description="Roughly how many faces the new mesh should have (a quad counts as one face)", default=8000, min=4, soft_max=250000)
    engine: bpy.props.EnumProperty(name="Method", description="How the new quads are built. Smart (Atlas-Q) measures the mesh, then tries several solvers and keeps the best", items=ENGINE_ITEMS, default="ATLAS_Q")
    quality: bpy.props.EnumProperty(name="Quality", description="Draft is fast with light cleanup; Hero spends more time on cleanup and checks", items=QUALITY_ITEMS, default="BALANCED")
    seed: bpy.props.IntProperty(name="Random Seed", description="Change this for a different but repeatable result", default=0, min=0, soft_max=100000)
    triad_seed_count: bpy.props.IntProperty(name="PairClosure Starts", description="How many starting points PairClosure tries. More is slower and usually cleaner", default=6, min=1, max=32)
    triad_feature_angle: bpy.props.FloatProperty(name="Feature Angle", description="Edges sharper than this angle (in degrees) count as creases and are kept", default=35.0, min=0.0, max=180.0)
    triad_force_quads: bpy.props.BoolProperty(name="PairClosure Pure Quads", description="Force every face to be a quad, even if shape accuracy suffers", default=False)
    triad_flow_mode: bpy.props.EnumProperty(name="PairClosure Style", description="Tune PairClosure for the kind of model: organic, hard-surface, messy scan or textured", items=FLOW_ITEMS, default="AUTO")
    flat_tolerance: bpy.props.FloatProperty(name="Flatness", description="How far a surface may bump away from flat and still become one big quad, as a percent of the mesh size. Higher gives fewer faces and less detail", default=0.1, min=0.0, soft_max=2.0, precision=3, subtype="PERCENTAGE")
    triad_field_mode: bpy.props.EnumProperty(name="Flow", description="Direction the quads follow: smoothed flow, or also the surface curvature (about 1.5x slower)", items=(("SMOOTH", "Smooth", "Smoothed flow pinned to sharp edges and borders"), ("CURVATURE", "Follow Curvature", "Also line up with the main bending directions of the surface")), default="SMOOTH")
    atlas_strategy: bpy.props.EnumProperty(name="Scan Type", description="Which plan Smart (Atlas-Q) follows. Auto picks one from the measured mesh", items=ATLAS_STRATEGY_ITEMS, default="AUTO")
    atlas_candidate_count: bpy.props.IntProperty(name="Attempts", description="How many candidate results are built and compared; the best one wins. More is slower", default=3, min=1, max=8)
    topology_policy: bpy.props.EnumProperty(name="Holes & Shells", description="Preserve keeps holes, open borders and separate parts as they are. Repair may close holes and drop thin sheets", items=TOPOLOGY_ITEMS, default="PRESERVE")
    require_quads: bpy.props.BoolProperty(name="Pure Quads", description="Reject results that still contain triangles or badly shaped quads. Turn off to keep the best attempt even with a few triangles", default=True)
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

    def execute(self, context: bpy.types.Context):
        obj = context.object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object first.")
            return {"CANCELLED"}

        result = run_curiomesh(obj, config_from_owner(self))
        settings = context.scene.curiomesh_settings
        apply_result_to_settings(settings, result)

        if not result.success:
            self.report({"ERROR"}, result.message or "CurioMesh remesh failed.")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            result.message,
        )
        return {"FINISHED"}


class CURIOMESH_OT_apply_preset(bpy.types.Operator):
    bl_idname = "curiomesh.apply_preset"
    bl_label = "Quality Preset"
    bl_description = "Pick a quality preset"
    bl_options = {"INTERNAL"}

    @classmethod
    def description(cls, context, properties):
        return {
            "DRAFT": "Draft: fast preview with light cleanup. Halves the face target",
            "BALANCED": "Balanced: good quality at a sensible speed. The right choice most of the time",
            "HERO": "Hero: slowest. Tries more candidates and cleans up harder, for final assets",
        }.get(str(properties.preset), cls.bl_description)

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
            settings.atlas_candidate_count = 1
        elif preset == "HERO":
            settings.cleanup_strength = 0.35
            settings.shrinkwrap_project = True
            settings.texture_mode = "PROJECT"
            settings.bake_fallback = True
            settings.preserve_sharp = True
            settings.preserve_boundary = True
            settings.preserve_attributes = True
            settings.smooth_normals = True
            settings.atlas_candidate_count = 5
        else:
            settings.cleanup_strength = 0.25
            settings.shrinkwrap_project = True
            settings.texture_mode = "PROJECT"
            settings.bake_fallback = False
            settings.preserve_sharp = True
            settings.preserve_boundary = True
            settings.preserve_attributes = True
            settings.smooth_normals = True
            settings.atlas_candidate_count = 3

        return {"FINISHED"}


class CURIOMESH_OT_analyze(bpy.types.Operator):
    bl_idname = "curiomesh.analyze"
    bl_label = "Analyze Mesh"
    bl_description = "Check holes, sharp features and scale, and recommend a method. Does not change the mesh"
    bl_options = {"INTERNAL"}

    def execute(self, context: bpy.types.Context):
        obj = context.object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object first.")
            return {"CANCELLED"}
        settings = context.scene.curiomesh_settings
        profile = analyze_object(
            obj,
            settings.target_faces,
            feature_angle_deg=settings.triad_feature_angle,
            max_sample_faces=settings.analysis_sample_faces,
        )
        settings.analysis_profile = profile.category
        settings.analysis_strategy = profile.recommended_strategy
        settings.analysis_faces = profile.face_count
        settings.analysis_sampled_faces = profile.sampled_faces
        settings.analysis_exact = profile.exact_topology
        settings.analysis_boundary_loops = profile.boundary_loops
        settings.analysis_non_manifold = profile.non_manifold_edges
        settings.analysis_sharp_ratio = profile.sharp_ratio
        settings.analysis_target_edge = profile.target_edge_length
        settings.analysis_warning = " | ".join(profile.warnings)
        self.report({"INFO"}, profile.summary)
        return {"FINISHED"}


class CURIOMESH_OT_compute_metrics(bpy.types.Operator):
    bl_idname = "curiomesh.compute_metrics"
    bl_label = "Measure Mesh"
    bl_description = "Measure the active mesh (quad ratio, poles, holes, UVs) and list the results below"
    bl_options = {"INTERNAL"}

    def execute(self, context: bpy.types.Context):
        obj = context.object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object first.")
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
            quad_only_verified=diagnostics.quad_only,
        )
        apply_result_to_settings(settings, result)
        return {"FINISHED"}
