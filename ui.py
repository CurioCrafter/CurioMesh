import bpy


class CURIOMESH_PG_settings(bpy.types.PropertyGroup):
    target_faces: bpy.props.IntProperty(name="Target Faces", default=8000, min=100)
    feature_weight: bpy.props.FloatProperty(name="Feature Weight", default=0.7, min=0.0, max=1.0)
    adaptivity: bpy.props.FloatProperty(name="Adaptivity", default=0.1, min=0.0, max=1.0)
    engine: bpy.props.EnumProperty(
        name="Engine",
        items=[
            ("AUTO", "Auto", "Autodetect"),
            ("CPU", "CPU", "CPU only"),
            ("CUDA", "CUDA", "NVIDIA CUDA"),
            ("METAL", "Metal", "Apple Metal"),
            ("VULKAN", "Vulkan", "Generic GPU"),
        ],
        default="AUTO",
    )
    use_vdb_pre: bpy.props.BoolProperty(name="VDB Preprocess", default=True)
    vdb_voxel_size: bpy.props.FloatProperty(name="Voxel Size", default=0.0, min=0.0, description="VDB voxel size; 0=auto")
    feature_angle: bpy.props.FloatProperty(name="Feature Angle (deg)", default=35.0, min=0.0, max=180.0)
    smooth_iters: bpy.props.IntProperty(name="Smooth Iters", default=5, min=0)
    smooth_lambda: bpy.props.FloatProperty(name="Smooth Strength", default=0.2, min=0.0, max=1.0)
    preserve_seams: bpy.props.BoolProperty(name="Preserve Seams", default=True)
    preserve_sharp: bpy.props.BoolProperty(name="Preserve Sharps", default=True)
    output_mode: bpy.props.EnumProperty(
        name="Output",
        items=[("REPLACE", "Replace", "Replace object geometry"), ("NEW", "New Object", "Create new object")],
        default="REPLACE",
    )
    # Metrics (read-only display)
    metrics_faces: bpy.props.IntProperty(name="Faces", default=0)
    metrics_extraordinary: bpy.props.IntProperty(name="Extraordinary", default=0)
    metrics_ratio: bpy.props.FloatProperty(name="Extra Ratio", default=0.0, precision=4)
    # Baking
    bake_pbr: bpy.props.BoolProperty(name="Bake PBR Textures", default=False)
    bake_resolution: bpy.props.IntProperty(name="Bake Resolution", default=2048, min=64)
    bake_margin: bpy.props.IntProperty(name="Bake Margin", default=8, min=0, max=64)
    bake_maps: bpy.props.EnumProperty(
        name="Maps",
        items=[
            ("COLOR", "Base Color", "Base Color"),
            ("ROUGHNESS", "Roughness", "Roughness"),
            ("METALLIC", "Metallic", "Metallic"),
            ("NORMAL", "Normal", "Normal"),
            ("EMIT", "Emission", "Emission"),
        ],
        options={'ENUM_FLAG'},
        default={"COLOR", "ROUGHNESS", "METALLIC", "NORMAL"},
    )
    uv_method: bpy.props.EnumProperty(
        name="UV Method",
        items=[("SMART", "Smart Project", "Smart UV Project"), ("UNWRAP", "Unwrap", "Standard unwrap")],
        default="SMART",
    )
    debug_logs: bpy.props.BoolProperty(name="Debug Console Logs", default=True)
    force_bake_fallback: bpy.props.BoolProperty(name="Force Bake Fallback", default=False, description="If UV transfer fails, bake from source to target")
    projection_mode: bpy.props.EnumProperty(
        name="Projection",
        description="How to transfer textures/UVs from source to remesh",
        items=[
            ("AUTO", "Auto", "Try Transfer → Nearest → Cage"),
            ("TRANSFER", "Transfer", "Blender Data Transfer only"),
            ("NEAREST", "Nearest", "BVH nearest surface projection"),
            ("CAGE", "Cage", "Ray project from auto cage"),
            ("BAKE", "Bake", "Skip projection and bake maps"),
        ],
        default="CAGE",
    )
    projection_expand: bpy.props.FloatProperty(
        name="Cage Expand", description="Expand/shrink auto cage by fraction of bounds diagonal (negative shrinks)",
        default=0.01, min=-1.0, soft_max=1.0
    )
    cage_mode: bpy.props.EnumProperty(
        name="Cage Mode",
        description="Geometry used for projection cage",
        items=[
            ("BOX", "Box", "Axis-aligned bounding box"),
            ("SPHERE", "Sphere", "Bounding sphere"),
            ("MESH", "Mesh Copy", "Copy of source mesh offset along normals"),
        ],
        default="MESH",
    )
    cage_inflate: bpy.props.EnumProperty(
        name="Inflate Method",
        description="How the Mesh cage inflates",
        items=[
            ("WEDGE", "Wedge-Normal", "Bisector-based per-vertex offset"),
            ("SOLIDIFY", "Solidify", "Use solidify thickness along face normals"),
            ("SDF", "Volumetric (SDF)", "Voxel remesh + smooth + solidify"),
        ], default="WEDGE"
    )
    wedge_angle_thresh: bpy.props.FloatProperty(name="Wedge Angle (deg)", default=30.0, min=0.0, max=180.0)
    solidify_thickness_factor: bpy.props.FloatProperty(name="Solidify Factor", description="Thickness as fraction of bounds diagonal", default=0.01, min=0.0, max=0.1)
    sdf_voxel_factor: bpy.props.FloatProperty(name="SDF Voxel Factor", description="Voxel size as fraction of bounds diagonal", default=0.01, min=0.001, max=0.1)
    sdf_smooth_iters: bpy.props.IntProperty(name="SDF Smooth Iters", default=5, min=0, max=50)
    # Advanced projection controls
    backface_thresh: bpy.props.FloatProperty(
        name="Backface Threshold", description="Reject hits whose surface normal opposes the ray by more than this (0=off, 1=strict)",
        default=0.10, min=0.0, max=1.0
    )
    cage_blend_samples: bpy.props.IntProperty(
        name="Cage Blend Samples", description="Extra samples blended per polygon in Cage mode",
        default=4, min=1, max=16
    )
    neighbor_blend: bpy.props.BoolProperty(
        name="Neighbor Blend (Nearest)", description="Blend multiple nearby triangle samples in Nearest mode to reduce banding", default=False)
    neighbor_blend_samples: bpy.props.IntProperty(
        name="Neighbor Samples", default=3, min=1, max=8
    )
    ray_dir_mode: bpy.props.EnumProperty(
        name="Ray Direction", items=[
            ("NORMALS_AXES", "Normals+Axes", "Use face normals plus axis directions"),
            ("NORMALS_ONLY", "Normals Only", "Use only destination polygon normals"),
            ("CENTER", "Center Vector", "Use vector from cage/scene center"),
        ], default="NORMALS_AXES"
    )
    ray_aim_mode: bpy.props.EnumProperty(
        name="Ray Aim",
        description="Where cage rays aim on the target mesh",
        items=[
            ("POLY_CENTER", "Poly Centers", "Aim rays at polygon centers"),
            ("VERTEX", "Vertices", "Aim rays at vertex positions"),
            ("EDGE", "Boundary Edges", "Aim rays at boundary/feature edges"),
        ], default="POLY_CENTER"
    )
    ray_aim_object: bpy.props.PointerProperty(name="Ray Aim Object", description="Optional object to bias/aim rays toward (its origin)", type=bpy.types.Object)
    ray_density: bpy.props.IntProperty(name="Ray Count", description="Number of ray directions sampled around each polygon", default=64, min=8, max=512)
    cage_center_offset: bpy.props.FloatVectorProperty(name="Cage Center Offset", description="World offset applied to cage center for ray emission", size=3, default=(0.0,0.0,0.0))
    axis_bias: bpy.props.EnumProperty(name="Axis Bias", description="Bias ray directions toward an axis", items=[
        ("NONE","None","No axis bias"),
        ("+X","+X","Bias toward +X"), ("-X","-X","Bias toward -X"),
        ("+Y","+Y","Bias toward +Y"), ("-Y","-Y","Bias toward -Y"),
        ("+Z","+Z","Bias toward +Z"), ("-Z","-Z","Bias toward -Z"),
    ], default="NONE")
    split_sharp: bpy.props.BoolProperty(name="Back Off At Sharps", description="Back off ray start near sharp features", default=True)
    sharp_angle: bpy.props.FloatProperty(name="Sharp Angle (deg)", default=35.0, min=0.0, max=180.0)
    sharp_backoff: bpy.props.FloatProperty(name="Sharp Backoff", description="Fraction of bounds diagonal to back off from sharp polys", default=0.01, min=0.0, max=0.1)
    uv_consistency_check: bpy.props.BoolProperty(name="UV Consistency Check", description="Fallback to base triangle when blended UV variance is high", default=True)
    uv_consistency_thresh: bpy.props.FloatProperty(name="UV Variance Threshold", default=0.02, min=0.0, max=0.2)
    projection_solver: bpy.props.EnumProperty(
        name="Solver",
        description="How UVs are computed for Cage mode",
        items=[
            ("AUTO", "Auto", "Heuristic"),
            ("LOOP_NEAREST", "Loop Nearest", "Per-loop nearest triangle (smoothest)"),
            ("POLY_PLANE", "Poly Plane", "Per-polygon plane projection (sharper edges)"),
            ("KNN_BLEND", "KNN Blend", "Blend several ray/triangle samples per loop"),
            ("RBF_FIELD", "RBF Field", "Thin-plate-spline vector field (displacement-based)"),
            ("SPECTRAL", "Spectral (LBO)", "Functional maps via Laplace–Beltrami bases"),
            ("OT", "Optimal Transport", "Wasserstein-preserving material statistics"),
            ("SGMM_NDF", "SGMM NDF", "Spherical-Gaussian mixture for normal distributions"),
            ("TENSOR_ANISO", "Anisotropic Tensor", "Covariance tensor baking for anisotropic BRDFs"),
            ("PRT", "PRT Transport", "Spherical-harmonics light transport"),
            ("DR", "Inverse Render", "Differentiable rendering optimization"),
        ], default="AUTO"
    )
    rbf_samples: bpy.props.IntProperty(name="RBF Samples", default=200, min=20, max=1000)
    rbf_lambda: bpy.props.FloatProperty(name="RBF Lambda", default=1e-3, min=1e-6, max=1e-1)
    spectral_basis: bpy.props.IntProperty(name="Basis Size", default=64, min=16, max=256)
    spectral_lambda: bpy.props.FloatProperty(name="Reg λ", default=1e-4, min=0.0, max=1e-1)
    ot_iters: bpy.props.IntProperty(name="OT Iters", default=50, min=10, max=500)
    sgmm_lobes: bpy.props.IntProperty(name="SGMM Lobes", default=3, min=1, max=6)
    sgmm_em_iters: bpy.props.IntProperty(name="EM Iters", default=15, min=5, max=100)
    tensor_iters: bpy.props.IntProperty(name="Tensor Iters", default=10, min=1, max=100)
    prt_order: bpy.props.IntProperty(name="SH Order", default=3, min=2, max=5)
    prt_samples: bpy.props.IntProperty(name="PRT Samples", default=8000, min=1000, max=20000)
    dr_steps: bpy.props.IntProperty(name="DR Steps", default=400, min=50, max=5000)
    dr_lr: bpy.props.FloatProperty(name="DR LR", default=1e-2, min=1e-5, max=1e-1)
    relax_uvs: bpy.props.BoolProperty(name="Relax UVs", default=True)
    relax_iters: bpy.props.IntProperty(name="Relax Iters", default=2, min=1, max=20)
    relax_alpha: bpy.props.FloatProperty(name="Relax Strength", default=0.35, min=0.0, max=1.0)
    relax_threshold: bpy.props.FloatProperty(name="Relax Threshold", default=0.15, min=0.0, max=1.0)
    lock_cage: bpy.props.BoolProperty(name="Lock Cage (Keep Edits)", default=False)
    # Quality: post-bake base color for perfect projection
    post_bake_color: bpy.props.BoolProperty(name="Post-Bake Base Color", description="Bake Base Color after projection to remove remaining artifacts", default=False)
    post_bake_res: bpy.props.IntProperty(name="Post-Bake Resolution", default=2048, min=256, max=8192)
    post_bake_margin: bpy.props.IntProperty(name="Post-Bake Margin", default=16, min=0, max=64)
    pure_quads: bpy.props.BoolProperty(name="Pure Quads", default=False)
    use_preview_cage: bpy.props.BoolProperty(name="Use Preview Cage", default=False)
    preview_cage_name: bpy.props.StringProperty(name="Preview Cage Name", default="CurioMesh_Cage")


class CURIOMESH_PT_panel(bpy.types.Panel):
    bl_label = "CurioMesh"
    bl_idname = "CURIOMESH_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == "MESH"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        s = context.scene.curiomesh_settings

        col.prop(s, "target_faces")
        col.label(text="Note: very low targets may be clamped by topology")
        col.prop(s, "feature_weight")
        col.prop(s, "adaptivity")
        col.prop(s, "engine")
        col.prop(s, "use_vdb_pre")
        if s.use_vdb_pre:
            row = col.row(align=True)
            row.prop(s, "vdb_voxel_size")
        col.separator()
        col.prop(s, "feature_angle")
        col.separator()
        col.prop(s, "smooth_iters")
        col.prop(s, "smooth_lambda")
        col.prop(s, "preserve_seams")
        col.prop(s, "preserve_sharp")
        col.prop(s, "output_mode")
        col.prop(s, "pure_quads")
        layout.separator()
        boxb = layout.box()
        boxb.label(text="Bake PBR")
        boxb.prop(s, "bake_pbr")
        if s.bake_pbr:
            boxb.prop(s, "bake_resolution")
            boxb.prop(s, "bake_margin")
            boxb.prop(s, "uv_method")
            boxb.prop(s, "bake_maps")
        layout.prop(s, "debug_logs")
        layout.prop(s, "force_bake_fallback")
        layout.separator()
        layout.label(text="Projection")
        layout.prop(s, "projection_mode")
        # Keep cage settings under Projection as requested
        if s.projection_mode == 'CAGE':
            boxc = layout.box()
            boxc.prop(s, "projection_expand")
            boxc.prop(s, "cage_mode")
            if s.cage_mode == 'MESH':
                row = boxc.row(align=True)
                row.prop(s, "cage_inflate")
                if s.cage_inflate == 'WEDGE':
                    boxc.prop(s, "wedge_angle_thresh")
                elif s.cage_inflate == 'SOLIDIFY':
                    boxc.prop(s, "solidify_thickness_factor")
                elif s.cage_inflate == 'SDF':
                    row = boxc.row(align=True)
                    row.prop(s, "sdf_voxel_factor")
                    row.prop(s, "sdf_smooth_iters")
            row = boxc.row(align=True)
            row.prop(s, "use_preview_cage")
            row.prop(s, "preview_cage_name")
            row = boxc.row(align=True)
            row.prop(s, "lock_cage")
            row = boxc.row(align=True)
            opv = row.operator("curiomesh.preview_cage", text="Build/Refresh Cage")
            opv.visualize_rays = False
            row = boxc.row(align=True)
            opv2 = row.operator("curiomesh.preview_cage", text="Build + Visualize Rays")
            opv2.visualize_rays = True
            opv2.visualize_count = 256
            boxc.prop(s, "cage_blend_samples")
            boxc.prop(s, "backface_thresh")
            boxc.prop(s, "ray_dir_mode")
            boxc.prop(s, "ray_aim_mode")
            boxc.prop(s, "ray_aim_object")
            boxc.prop(s, "ray_density")
            boxc.prop(s, "axis_bias")
            boxc.prop(s, "cage_center_offset")
            boxc.prop(s, "projection_solver")
            if s.projection_solver == 'RBF_FIELD':
                row = boxc.row(align=True)
                row.prop(s, "rbf_samples")
                row.prop(s, "rbf_lambda")
            elif s.projection_solver == 'SPECTRAL':
                row = boxc.row(align=True)
                row.prop(s, "spectral_basis")
                row.prop(s, "spectral_lambda")
            elif s.projection_solver == 'OT':
                boxc.prop(s, "ot_iters")
            elif s.projection_solver == 'SGMM_NDF':
                row = boxc.row(align=True)
                row.prop(s, "sgmm_lobes")
                row.prop(s, "sgmm_em_iters")
            elif s.projection_solver == 'TENSOR_ANISO':
                boxc.prop(s, "tensor_iters")
            elif s.projection_solver == 'PRT':
                row = boxc.row(align=True)
                row.prop(s, "prt_order")
                row.prop(s, "prt_samples")
            elif s.projection_solver == 'DR':
                row = boxc.row(align=True)
                row.prop(s, "dr_steps")
                row.prop(s, "dr_lr")
            row = boxc.row(align=True)
            row.prop(s, "split_sharp")
            if s.split_sharp:
                row = boxc.row(align=True)
                row.prop(s, "sharp_angle")
                row.prop(s, "sharp_backoff")
            row = boxc.row(align=True)
            row.prop(s, "uv_consistency_check")
            if s.uv_consistency_check:
                row.prop(s, "uv_consistency_thresh")
        boxa = layout.box()
        boxa.label(text="Advanced Anti-Banding")
        boxa.prop(s, "neighbor_blend")
        if s.neighbor_blend:
            boxa.prop(s, "neighbor_blend_samples")
        boxa.prop(s, "relax_uvs")
        if s.relax_uvs:
            row = boxa.row(align=True)
            row.prop(s, "relax_iters")
            row.prop(s, "relax_alpha")
            row.prop(s, "relax_threshold")
        boxq = layout.box()
        boxq.label(text="Quality")
        boxq.prop(s, "post_bake_color")
        if s.post_bake_color:
            row = boxq.row(align=True)
            row.prop(s, "post_bake_res")
            row.prop(s, "post_bake_margin")

        op = col.operator("curiomesh.remesh", text="Remesh (CurioMesh)")
        op.target_faces = s.target_faces
        op.feature_weight = s.feature_weight
        op.adaptivity = s.adaptivity
        op.engine = s.engine
        op.use_vdb_pre = s.use_vdb_pre
        op.vdb_voxel_size = s.vdb_voxel_size
        op.feature_angle = s.feature_angle
        op.smooth_iters = s.smooth_iters
        op.smooth_lambda = s.smooth_lambda
        op.preserve_seams = s.preserve_seams
        op.preserve_sharp = s.preserve_sharp
        op.output_mode = s.output_mode
        op.bake_pbr = s.bake_pbr
        op.bake_resolution = s.bake_resolution
        op.bake_margin = s.bake_margin
        op.bake_maps = s.bake_maps
        op.uv_method = s.uv_method
        op.debug_logs = s.debug_logs
        op.force_bake_fallback = s.force_bake_fallback
        op.projection_mode = s.projection_mode
        op.projection_expand = s.projection_expand
        op.pure_quads = s.pure_quads
        op.use_preview_cage = s.use_preview_cage
        op.preview_cage_name = s.preview_cage_name
        # Advanced
        op.backface_thresh = s.backface_thresh
        op.cage_blend_samples = s.cage_blend_samples
        op.neighbor_blend = s.neighbor_blend
        op.neighbor_blend_samples = s.neighbor_blend_samples
        op.ray_dir_mode = s.ray_dir_mode
        op.ray_aim_mode = s.ray_aim_mode
        op.ray_aim_object_name = (s.ray_aim_object.name if s.ray_aim_object else "")
        op.ray_density = s.ray_density
        op.axis_bias = s.axis_bias
        op.cage_center_offset = s.cage_center_offset
        op.split_sharp = s.split_sharp
        op.sharp_angle = s.sharp_angle
        op.sharp_backoff = s.sharp_backoff
        op.uv_consistency_check = s.uv_consistency_check
        op.uv_consistency_thresh = s.uv_consistency_thresh
        op.projection_solver = s.projection_solver
        op.rbf_samples = s.rbf_samples
        op.rbf_lambda = s.rbf_lambda
        op.spectral_basis = s.spectral_basis
        op.spectral_lambda = s.spectral_lambda
        op.ot_iters = s.ot_iters
        op.sgmm_lobes = s.sgmm_lobes
        op.sgmm_em_iters = s.sgmm_em_iters
        op.tensor_iters = s.tensor_iters
        op.prt_order = s.prt_order
        op.prt_samples = s.prt_samples
        op.dr_steps = s.dr_steps
        op.dr_lr = s.dr_lr
        op.relax_uvs = s.relax_uvs
        op.relax_iters = s.relax_iters
        op.relax_alpha = s.relax_alpha
        op.relax_threshold = s.relax_threshold
        op.post_bake_color = s.post_bake_color
        op.post_bake_res = s.post_bake_res
        op.post_bake_margin = s.post_bake_margin
        op.cage_mode = s.cage_mode

        # Presets box near the Remesh button for quick access
        boxp = layout.box()
        boxp.label(text="Presets")
        row = boxp.row(align=True)
        row.operator("curiomesh.apply_preset", text="Draft").preset = 'DRAFT'
        row.operator("curiomesh.apply_preset", text="Balanced").preset = 'BALANCED'
        row.operator("curiomesh.apply_preset", text="Hero").preset = 'HERO'
        row = boxp.row(align=True)
        row.operator("curiomesh.apply_preset", text="Mesh Detail").preset = 'CAGE_MESH_DETAIL'
        row.operator("curiomesh.apply_preset", text="Sphere Soft").preset = 'CAGE_SPHERE_SOFT'
        row = boxp.row(align=True)
        row.operator("curiomesh.apply_preset", text="Nearest kNN").preset = 'NEAREST_KNN_FAST'
        row.operator("curiomesh.apply_preset", text="Bake Hero").preset = 'BAKE_HERO'

        layout.separator()
        box = layout.box()
        box.label(text="Metrics")
        row = box.row(align=True)
        row.operator("curiomesh.compute_metrics", text="Compute")
        colm = box.column(align=True)
        colm.prop(s, "metrics_faces")
        colm.prop(s, "metrics_extraordinary")
        colm.prop(s, "metrics_ratio")


