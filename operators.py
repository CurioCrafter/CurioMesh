import bpy
import bmesh
from mathutils import Vector

from .bridge import run_curiomesh


class CURIOMESH_OT_remesh(bpy.types.Operator):
    bl_idname = "curiomesh.remesh"
    bl_label = "CurioMesh Remesh"
    bl_description = "Fast field-aligned quad remesher (experimental)"
    bl_options = {"REGISTER", "UNDO"}

    target_faces: bpy.props.IntProperty(
        name="Target Faces", default=8000, min=10, soft_max=1000000
    )
    feature_weight: bpy.props.FloatProperty(
        name="Feature Weight", default=0.7, min=0.0, max=1.0
    )
    adaptivity: bpy.props.FloatProperty(
        name="Adaptivity", default=0.1, min=0.0, max=1.0
    )
    engine: bpy.props.EnumProperty(
        name="Engine",
        items=[
            ("AUTO", "Auto", "Autodetect best engine"),
            ("CPU", "CPU", "CPU only"),
            ("CUDA", "CUDA", "NVIDIA CUDA"),
            ("METAL", "Metal", "Apple Metal"),
            ("VULKAN", "Vulkan", "Generic GPU via Vulkan"),
        ],
        default="AUTO",
    )
    use_vdb_pre: bpy.props.BoolProperty(name="VDB Preprocess", default=True)
    vdb_voxel_size: bpy.props.FloatProperty(
        name="Voxel Size", description="Voxel size for preprocess; 0 = auto",
        default=0.0, min=0.0, soft_max=0.5
    )
    preserve_seams: bpy.props.BoolProperty(name="Preserve Seams", default=True)
    preserve_sharp: bpy.props.BoolProperty(name="Preserve Sharp Edges", default=True)
    output_mode: bpy.props.EnumProperty(
        name="Output",
        items=[("REPLACE", "Replace", "Replace object geometry"), ("NEW", "New Object", "Create new object")],
        default="REPLACE",
    )
    feature_angle: bpy.props.FloatProperty(
        name="Feature Angle", description="Dihedral angle in degrees to detect sharp edges",
        default=35.0, min=0.0, max=180.0
    )
    smooth_iters: bpy.props.IntProperty(
        name="Smooth Iters", description="Post-smoothing iterations",
        default=5, min=0, soft_max=50
    )
    smooth_lambda: bpy.props.FloatProperty(
        name="Smooth Strength", description="Laplacian smoothing factor",
        default=0.2, min=0.0, max=1.0
    )
    # PBR baking/preservation
    bake_pbr: bpy.props.BoolProperty(name="Bake PBR Textures", default=False)
    bake_resolution: bpy.props.IntProperty(name="Bake Resolution", default=2048, min=64, soft_max=8192)
    bake_margin: bpy.props.IntProperty(name="Bake Margin", default=8, min=0, max=64)
    bake_maps: bpy.props.EnumProperty(
        name="Maps",
        items=[
            ("COLOR", "Base Color", "Bake base color (albedo)", 0),
            ("ROUGHNESS", "Roughness", "Bake roughness", 1),
            ("METALLIC", "Metallic", "Bake metallic", 2),
            ("NORMAL", "Normal", "Bake normal (tangent)", 3),
            ("EMIT", "Emission", "Bake emission", 4),
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
    force_bake_fallback: bpy.props.BoolProperty(name="Force Bake Fallback", default=False)
    pure_quads: bpy.props.BoolProperty(name="Pure Quads", default=False, description="Attempt to output only quads (may add verts on leftover tris)")
    use_preview_cage: bpy.props.BoolProperty(name="Use Preview Cage", default=False)
    preview_cage_name: bpy.props.StringProperty(name="Preview Cage Name", default="CurioMesh_Cage")
    cage_mode: bpy.props.EnumProperty(
        name="Cage Mode",
        items=[("BOX", "Box", "Axis-aligned bounding box"), ("SPHERE", "Sphere", "Bounding sphere"), ("MESH", "Mesh Copy", "Copy of source mesh offset along normals")],
        default="BOX",
    )
    # Advanced projection controls
    backface_thresh: bpy.props.FloatProperty(name="Backface Threshold", default=0.10, min=0.0, max=1.0)
    cage_blend_samples: bpy.props.IntProperty(name="Cage Blend Samples", default=4, min=1, max=16)
    neighbor_blend: bpy.props.BoolProperty(name="Neighbor Blend (Nearest)", default=False)
    neighbor_blend_samples: bpy.props.IntProperty(name="Neighbor Samples", default=3, min=1, max=8)
    ray_dir_mode: bpy.props.EnumProperty(name="Ray Direction", items=[("NORMALS_AXES","Normals+Axes",""),("NORMALS_ONLY","Normals Only",""),("CENTER","Center Vector","")], default="NORMALS_AXES")
    ray_aim_mode: bpy.props.EnumProperty(name="Ray Aim", items=[("POLY_CENTER","Poly Centers",""),("VERTEX","Vertices",""),("EDGE","Boundary Edges","")], default="POLY_CENTER")
    ray_aim_object_name: bpy.props.StringProperty(name="Ray Aim Object Name", default="")
    ray_density: bpy.props.IntProperty(name="Ray Count", default=64, min=8, max=512)
    axis_bias: bpy.props.EnumProperty(name="Axis Bias", items=[("NONE","None",""),("+X","+X",""),("-X","-X",""),("+Y","+Y",""),("-Y","-Y",""),("+Z","+Z",""),("-Z","-Z","")], default="NONE")
    cage_center_offset: bpy.props.FloatVectorProperty(name="Cage Center Offset", size=3, default=(0.0,0.0,0.0))
    split_sharp: bpy.props.BoolProperty(name="Back Off At Sharps", default=True)
    sharp_angle: bpy.props.FloatProperty(name="Sharp Angle (deg)", default=35.0, min=0.0, max=180.0)
    sharp_backoff: bpy.props.FloatProperty(name="Sharp Backoff", default=0.01, min=0.0, max=0.1)
    uv_consistency_check: bpy.props.BoolProperty(name="UV Consistency Check", default=True)
    uv_consistency_thresh: bpy.props.FloatProperty(name="UV Variance Threshold", default=0.02, min=0.0, max=0.2)
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
    projection_solver: bpy.props.EnumProperty(
        name="Solver",
        items=[
            ("AUTO","Auto",""),
            ("LOOP_NEAREST","Loop Nearest",""),
            ("POLY_PLANE","Poly Plane",""),
            ("KNN_BLEND","KNN Blend",""),
            ("RBF_FIELD","RBF Field",""),
            ("SPECTRAL","Spectral (LBO)",""),
            ("OT","Optimal Transport",""),
            ("SGMM_NDF","SGMM NDF",""),
            ("TENSOR_ANISO","Anisotropic Tensor",""),
            ("PRT","PRT Transport",""),
            ("DR","Inverse Render",""),
        ],
        default="AUTO"
    )
    relax_uvs: bpy.props.BoolProperty(name="Relax UVs", default=True)
    relax_iters: bpy.props.IntProperty(name="Relax Iters", default=2, min=1, max=20)
    relax_alpha: bpy.props.FloatProperty(name="Relax Strength", default=0.35, min=0.0, max=1.0)
    relax_threshold: bpy.props.FloatProperty(name="Relax Threshold", default=0.15, min=0.0, max=1.0)
    lock_cage: bpy.props.BoolProperty(name="Lock Cage (Keep Edits)", default=False)
    post_bake_color: bpy.props.BoolProperty(name="Post-Bake Base Color", default=False)
    post_bake_res: bpy.props.IntProperty(name="Post-Bake Resolution", default=2048, min=256, max=8192)
    post_bake_margin: bpy.props.IntProperty(name="Post-Bake Margin", default=16, min=0, max=64)
    projection_mode: bpy.props.EnumProperty(
        name="Projection",
        items=[
            ("AUTO", "Auto", "Try Transfer → Nearest → Cage"),
            ("TRANSFER", "Transfer", "Blender Data Transfer only"),
            ("NEAREST", "Nearest", "BVH nearest surface projection"),
            ("CAGE", "Cage", "Ray project from auto cage"),
            ("BAKE", "Bake", "Skip projection and bake maps"),
        ],
        default="AUTO",
    )
    projection_expand: bpy.props.FloatProperty(name="Cage Expand", default=0.25, min=0.0, max=1.0)

    def execute(self, context: bpy.types.Context):
        obj = context.object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object.")
            return {"CANCELLED"}

        # Ensure mesh is up-to-date
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(eval_obj)

        try:
            ok = run_curiomesh(
                obj,
                mesh,
                target_faces=self.target_faces,
                feature_weight=self.feature_weight,
                adaptivity=self.adaptivity,
                engine=self.engine,
                use_vdb=self.use_vdb_pre,
                vdb_voxel_size=self.vdb_voxel_size,
                feature_angle_deg=self.feature_angle,
                smooth_iters=self.smooth_iters,
                smooth_lambda=self.smooth_lambda,
                preserve_seams=self.preserve_seams,
                preserve_sharp=self.preserve_sharp,
                output_mode=self.output_mode,
                bake_pbr=self.bake_pbr,
                bake_resolution=self.bake_resolution,
                bake_margin=self.bake_margin,
                bake_maps=set(self.bake_maps),
                uv_method=self.uv_method,
                debug_logs=self.debug_logs,
                force_bake_fallback=self.force_bake_fallback,
                projection_mode=self.projection_mode,
                projection_expand=self.projection_expand,
                pure_quads=self.pure_quads,
                use_preview_cage=self.use_preview_cage,
                preview_cage_name=self.preview_cage_name,
                # Advanced
                backface_thresh=self.backface_thresh,
                cage_blend_samples=self.cage_blend_samples,
                neighbor_blend=self.neighbor_blend,
                neighbor_blend_samples=self.neighbor_blend_samples,
                ray_dir_mode=self.ray_dir_mode,
                ray_aim_mode=self.ray_aim_mode,
                ray_aim_object_name=self.ray_aim_object_name,
                ray_density=self.ray_density,
                axis_bias=self.axis_bias,
                cage_center_offset=self.cage_center_offset,
                split_sharp=bool(self.split_sharp),
                sharp_angle=float(self.sharp_angle),
                sharp_backoff=float(self.sharp_backoff),
                uv_consistency_check=bool(self.uv_consistency_check),
                uv_consistency_thresh=float(self.uv_consistency_thresh),
                projection_solver=self.projection_solver,
                rbf_samples=int(self.rbf_samples),
                rbf_lambda=float(self.rbf_lambda),
                spectral_basis=int(self.spectral_basis),
                spectral_lambda=float(self.spectral_lambda),
                ot_iters=int(self.ot_iters),
                sgmm_lobes=int(self.sgmm_lobes),
                sgmm_em_iters=int(self.sgmm_em_iters),
                tensor_iters=int(self.tensor_iters),
                prt_order=int(self.prt_order),
                prt_samples=int(self.prt_samples),
                dr_steps=int(self.dr_steps),
                dr_lr=float(self.dr_lr),
                relax_uvs=self.relax_uvs,
                relax_iters=self.relax_iters,
                relax_alpha=self.relax_alpha,
                relax_threshold=self.relax_threshold,
                lock_cage=self.lock_cage,
                post_bake_color=self.post_bake_color,
                post_bake_res=self.post_bake_res,
                post_bake_margin=self.post_bake_margin,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"CurioMesh error: {exc}")
            ok = False
        finally:
            # Preserve original mesh if output=NEW for possible UV transfer reference
            try:
                bpy.data.meshes.remove(mesh)
            except Exception:
                pass

        if not ok:
            self.report({"ERROR"}, "Remesh failed or produced empty output. In Preferences > Add-ons, re-enable 'Debug Console Logs' and check Window > Toggle System Console.")

        return {"FINISHED" if ok else "CANCELLED"}


class CURIOMESH_OT_apply_preset(bpy.types.Operator):
    bl_idname = "curiomesh.apply_preset"
    bl_label = "Apply CurioMesh Preset"
    bl_options = {"INTERNAL"}

    preset: bpy.props.EnumProperty(
        name="Preset",
        items=[
            ("DRAFT", "Draft", "Max speed"),
            ("BALANCED", "Balanced", "Balanced quality/perf"),
            ("HERO", "Hero", "Higher quality"),
            ("CAGE_MESH_DETAIL", "Cage Mesh Detail", "Mesh cage, higher density and relax"),
            ("CAGE_SPHERE_SOFT", "Cage Sphere Soft", "Sphere cage, soft projection"),
            ("NEAREST_KNN_FAST", "Nearest kNN Fast", "Nearest projection with kNN blend"),
            ("BAKE_HERO", "Bake Hero", "Cage Mesh + 4K post-bake"),
        ],
        default="BALANCED",
    )

    def execute(self, context: bpy.types.Context):
        s = context.scene.curiomesh_settings
        p = self.preset
        if p == "DRAFT":
            s.engine = "AUTO"
            s.target_faces = max(1000, s.target_faces // 2)
            s.feature_weight = 0.5
            s.adaptivity = 0.3
            s.use_vdb_pre = False
            s.vdb_voxel_size = 0.0
            s.smooth_iters = 2
            s.smooth_lambda = 0.1
            # Projection preset (fast)
            s.projection_mode = 'CAGE'
            s.cage_mode = 'BOX'
            s.projection_expand = 0.1
            s.cage_blend_samples = 3
            s.backface_thresh = 0.1
            s.ray_dir_mode = 'NORMALS_AXES'
            s.relax_uvs = True
            s.relax_iters = 1
            s.relax_alpha = 0.25
            s.relax_threshold = 0.1
        elif p == "HERO":
            s.engine = "AUTO"
            s.target_faces = s.target_faces
            s.feature_weight = 0.85
            s.adaptivity = 0.05
            s.use_vdb_pre = True
            s.vdb_voxel_size = 0.0
            s.smooth_iters = 10
            s.smooth_lambda = 0.25
            # Projection preset (high quality)
            s.projection_mode = 'CAGE'
            s.cage_mode = 'MESH'
            s.projection_expand = -0.05
            s.cage_blend_samples = 8
            s.backface_thresh = 0.25
            s.ray_dir_mode = 'NORMALS_ONLY'
            s.relax_uvs = True
            s.relax_iters = 4
            s.relax_alpha = 0.45
            s.relax_threshold = 0.15
            s.post_bake_color = True
            s.post_bake_res = 4096
            s.post_bake_margin = 16
        elif p == "CAGE_MESH_DETAIL":
            s.projection_mode = 'CAGE'
            s.cage_mode = 'MESH'
            s.projection_expand = 0.01
            s.cage_blend_samples = 10
            s.backface_thresh = 0.25
            s.ray_dir_mode = 'NORMALS_ONLY'
            s.ray_aim_mode = 'VERTEX'
            s.ray_density = 128
            s.axis_bias = 'NONE'
            s.relax_uvs = True
            s.relax_iters = 3
            s.relax_alpha = 0.4
            s.relax_threshold = 0.15
            s.post_bake_color = False
        elif p == "CAGE_SPHERE_SOFT":
            s.projection_mode = 'CAGE'
            s.cage_mode = 'SPHERE'
            s.projection_expand = 0.2
            s.cage_blend_samples = 8
            s.backface_thresh = 0.15
            s.ray_dir_mode = 'NORMALS_AXES'
            s.ray_aim_mode = 'POLY_CENTER'
            s.ray_density = 64
            s.axis_bias = 'NONE'
            s.relax_uvs = True
            s.relax_iters = 2
            s.relax_alpha = 0.3
            s.relax_threshold = 0.12
            s.post_bake_color = False
        elif p == "NEAREST_KNN_FAST":
            s.projection_mode = 'NEAREST'
            s.neighbor_blend = True
            s.neighbor_blend_samples = 5
            s.relax_uvs = True
            s.relax_iters = 2
            s.relax_alpha = 0.35
            s.relax_threshold = 0.12
            s.post_bake_color = False
        elif p == "BAKE_HERO":
            s.projection_mode = 'CAGE'
            s.cage_mode = 'MESH'
            s.projection_expand = 0.01
            s.cage_blend_samples = 10
            s.backface_thresh = 0.25
            s.ray_dir_mode = 'NORMALS_ONLY'
            s.ray_aim_mode = 'VERTEX'
            s.ray_density = 128
            s.axis_bias = 'NONE'
            s.relax_uvs = True
            s.relax_iters = 3
            s.relax_alpha = 0.4
            s.relax_threshold = 0.15
            s.post_bake_color = True
            s.post_bake_res = 4096
            s.post_bake_margin = 16
        else:  # BALANCED
            s.engine = "AUTO"
            s.feature_weight = 0.7
            s.adaptivity = 0.1
            s.use_vdb_pre = True
            s.vdb_voxel_size = 0.0
            s.smooth_iters = 5
            s.smooth_lambda = 0.2
            # Projection preset (balanced)
            s.projection_mode = 'CAGE'
            s.cage_mode = 'SPHERE'
            s.projection_expand = 0.15
            s.cage_blend_samples = 6
            s.backface_thresh = 0.2
            s.ray_dir_mode = 'NORMALS_AXES'
            s.relax_uvs = True
            s.relax_iters = 2
            s.relax_alpha = 0.35
            s.relax_threshold = 0.15
        return {"FINISHED"}


class CURIOMESH_OT_compute_metrics(bpy.types.Operator):
    bl_idname = "curiomesh.compute_metrics"
    bl_label = "Compute Metrics"
    bl_options = {"INTERNAL"}

    def execute(self, context: bpy.types.Context):
        obj = context.object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object.")
            return {"CANCELLED"}
        from .metrics import quick_metrics

        m = quick_metrics(obj.data)
        s = context.scene.curiomesh_settings
        s.metrics_faces = int(m["faces"])  # type: ignore[attr-defined]
        s.metrics_extraordinary = int(m["extraordinary"])  # type: ignore[attr-defined]
        s.metrics_ratio = float(m["extraordinary_ratio"])  # type: ignore[attr-defined]
        return {"FINISHED"}


class CURIOMESH_OT_preview_cage(bpy.types.Operator):
    bl_idname = "curiomesh.preview_cage"
    bl_label = "Preview Projection Cage"
    bl_options = {"REGISTER"}
    visualize_rays: bpy.props.BoolProperty(name="Visualize Rays", default=False)
    visualize_count: bpy.props.IntProperty(name="Ray Samples", default=128, min=1, max=5000)

    def execute(self, context: bpy.types.Context):
        s = context.scene.curiomesh_settings
        obj = context.object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, 'Select a mesh object.')
            return {'CANCELLED'}
        # Ensure the depsgraph is valid at the moment we grab it
        deps = context.evaluated_depsgraph_get()
        try:
            obj_eval = obj.evaluated_get(deps)
        except ReferenceError:
            deps = context.view_layer.depsgraph
            obj_eval = obj.evaluated_get(deps)
        name = s.preview_cage_name or "CurioMesh_Cage"
        # Remove existing
        if name in bpy.data.objects:
            try:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
            except Exception:
                pass
        # Create cage based on mode
        mode = s.cage_mode if hasattr(s, 'cage_mode') else 'BOX'
        me = bpy.data.meshes.new(name+"Mesh")
        o = bpy.data.objects.new(name, me)
        try:
            context.collection.objects.link(o)
        except Exception:
            try:
                context.scene.collection.objects.link(o)
            except Exception:
                bpy.context.collection.objects.link(o)

        # Re-evaluate in case object changed after removal step
        deps = context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(deps)
        bb = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]
        xs=[p.x for p in bb]; ys=[p.y for p in bb]; zs=[p.z for p in bb]
        minv = Vector((min(xs),min(ys),min(zs)))
        maxv = Vector((max(xs),max(ys),max(zs)))
        center = (minv+maxv)*0.5
        diag = (maxv-minv).length
        expand = max(diag * float(context.scene.curiomesh_settings.projection_expand), 1e-4)

        if mode == 'SPHERE':
            # Build an actual UV sphere with radius that encloses the object
            import bmesh
            radius = max((maxv - minv).x, (maxv - minv).y, (maxv - minv).z) * 0.5 + expand
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=radius)
            bm.to_mesh(me)
            bm.free()
            o.location = center
            try:
                o["curiomesh_cage_mode"] = "SPHERE"
            except Exception:
                pass
        elif mode == 'MESH':
            # Copy evaluated mesh and move verts along their normals by expand
            src_mesh = obj_eval.to_mesh()
            mesh_copy = src_mesh.copy()
            o.data = mesh_copy
            # Match transforms so world-space size/position aligns
            try:
                o.matrix_world = obj.matrix_world.copy()
            except Exception:
                pass
            try:
                bpy.data.meshes.remove(me)
            except Exception:
                pass
            me = o.data
            # Apply advanced inflation methods
            import bmesh
            from math import radians, cos
            bm = bmesh.new()
            bm.from_mesh(me)
            bm.normal_update()
            
            # Convert world expand to approximate local expand based on average scale
            try:
                sx, sy, sz = obj.scale
                scale_avg = (abs(sx)+abs(sy)+abs(sz))/3.0
                offset = expand / max(scale_avg, 1e-6)
            except Exception:
                offset = expand
            
            # Apply different inflation methods (match UI: WEDGE, SOLIDIFY, SDF)
            if getattr(s, 'cage_inflate', 'WEDGE') == 'WEDGE':
                # Wedge-Normal inflation: bisector of incident face normal clusters
                wedge_angle = radians(getattr(s, 'wedge_angle_thresh', 30.0))
                
                # Build vertex -> face normal mapping
                vert_normals = {}
                for vert in bm.verts:
                    face_normals = []
                    for face in vert.link_faces:
                        face_normals.append(face.normal.copy())
                    
                    if not face_normals:
                        vert_normals[vert] = vert.normal
                        continue
                    
                    # Cluster normals based on angle threshold
                    clusters = []
                    for fn in face_normals:
                        added = False
                        for cluster in clusters:
                            if fn.dot(cluster[0]) > cos(wedge_angle):
                                cluster.append(fn)
                                added = True
                                break
                        if not added:
                            clusters.append([fn])
                    
                    # For each cluster, compute average normal
                    cluster_dirs = []
                    for cluster in clusters:
                        avg = sum(cluster, Vector((0,0,0))) / len(cluster)
                        avg.normalize()
                        cluster_dirs.append(avg)
                    
                    # Compute bisector if multiple clusters, else use single direction
                    if len(cluster_dirs) == 1:
                        vert_normals[vert] = cluster_dirs[0]
                    elif len(cluster_dirs) == 2:
                        # Bisector of two directions
                        bisector = (cluster_dirs[0] + cluster_dirs[1]).normalized()
                        vert_normals[vert] = bisector
                    else:
                        # Average all cluster directions
                        avg_dir = sum(cluster_dirs, Vector((0,0,0))) / len(cluster_dirs)
                        avg_dir.normalize()
                        vert_normals[vert] = avg_dir
                
                # Apply offset along computed directions
                for vert in bm.verts:
                    direction = vert_normals.get(vert, vert.normal)
                    vert.co = vert.co + (direction * offset)
            
            elif getattr(s, 'cage_inflate', 'WEDGE') == 'SOLIDIFY':
                # Simple uniform thickness along vertex normals
                factor = float(getattr(s, 'solidify_thickness_factor', 0.01)) / max(1e-6, diag)
                for vert in bm.verts:
                    vert.co = vert.co + (vert.normal * offset * factor)
            
            elif getattr(s, 'cage_inflate', 'WEDGE') == 'SDF':
                # Volumetric SDF-based offset using voxel remesh
                bm.to_mesh(me)
                o.data = me
                
                # Temporarily make cage active for voxel remesh
                ctx = bpy.context
                view_layer = ctx.view_layer
                prev_active = view_layer.objects.active
                prev_sel = [ob for ob in ctx.selected_objects]
                
                try:
                    for ob in ctx.selected_objects:
                        ob.select_set(False)
                    o.select_set(True)
                    view_layer.objects.active = o
                    
                    # Voxel remesh for clean topology
                    voxel_size = max(0.001, abs(expand) * float(getattr(s, 'sdf_voxel_factor', 0.01)))
                    try:
                        bpy.ops.object.voxel_remesh(voxel_size=voxel_size)
                    except:
                        pass
                    
                    # Smooth to get cleaner offset surface
                    if int(getattr(s, 'sdf_smooth_iters', 5)) > 0:
                        try:
                            mod = o.modifiers.new("Smooth", 'SMOOTH')
                            mod.factor = 0.5
                            mod.iterations = max(1, int(getattr(s, 'sdf_smooth_iters', 5)))
                            bpy.ops.object.modifier_apply(modifier=mod.name)
                        except:
                            pass
                    
                    # Apply solidify for final offset
                    try:
                        mod = o.modifiers.new("Solidify", 'SOLIDIFY')
                        mod.thickness = abs(expand) * 2
                        mod.offset = 1.0 if expand > 0 else -1.0  # Direction based on expand sign
                        bpy.ops.object.modifier_apply(modifier=mod.name)
                    except:
                        pass
                finally:
                    # Restore selection
                    for ob in ctx.selected_objects:
                        ob.select_set(False)
                    for ob in prev_sel:
                        ob.select_set(True)
                    view_layer.objects.active = prev_active
                
                # Skip bmesh operations as mesh is already updated
                me = o.data
            
            else:  # Default radial inflation
                # Compute center in local space
                if len(bm.verts) > 0:
                    c = sum((v.co for v in bm.verts), Vector((0.0, 0.0, 0.0))) * (1.0/len(bm.verts))
                else:
                    c = Vector((0.0, 0.0, 0.0))
                
                for v in bm.verts:
                    d = v.co - c
                    if d.length > 1e-12:
                        v.co = v.co + d.normalized() * offset
            
            # Write back to mesh if not volumetric (which already updated)
            if getattr(s, 'cage_inflate', 'WEDGE') != 'SDF':
                bm.to_mesh(me)
                bm.free()
            else:
                bm.free()
            try:
                obj_eval.to_mesh_clear()
            except Exception:
                pass
            try:
                o["curiomesh_cage_mode"] = "MESH_COPY"
            except Exception:
                pass
            # Keep cage selectable and transformable; disable symmetry widgets if any
            o.hide_select = False
            o.show_in_front = True
        else:  # BOX
            minv -= Vector((expand,expand,expand))
            maxv += Vector((expand,expand,expand))
            v = [
                (minv.x,minv.y,minv.z), (maxv.x,minv.y,minv.z), (maxv.x,maxv.y,minv.z), (minv.x,maxv.y,minv.z),
                (minv.x,minv.y,maxv.z), (maxv.x,minv.y,maxv.z), (maxv.x,maxv.y,maxv.z), (minv.x,maxv.y,maxv.z)
            ]
            e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            me.from_pydata(v,e,[])
            try:
                o["curiomesh_cage_mode"] = "BOX"
            except Exception:
                pass

        o.display_type = 'WIRE'
        o.show_in_front = True
        # Ensure sphere/mesh preview stays wireframe
        try:
            o.display.show_shaded_solid = False
        except Exception:
            pass
        try:
            o.hide_render = True
            o.visible_camera = False
        except Exception:
            pass
        # Ensure the cage is visible in viewport overlays
        o.hide_set(False)
        o.hide_viewport = False
        o.hide_render = True
        try:
            s.use_preview_cage = True
        except Exception:
            pass

        # Optional: build a helper collection of ray gizmos for visualization
        if bool(getattr(self, 'visualize_rays', False)):
            coll_name = name + "_Rays"
            if coll_name in bpy.data.collections:
                try:
                    bpy.data.collections.remove(bpy.data.collections[coll_name])
                except Exception:
                    pass
            rayc = bpy.data.collections.new(coll_name)
            try:
                context.scene.collection.children.link(rayc)
            except Exception:
                pass
            # sample points on cage and draw rays toward chosen aim (poly centers / vertices / boundary edges)
            verts = [o.matrix_world @ v.co for v in o.data.vertices]
            if verts:
                import random
                random.seed(0)
                center = sum(verts, Vector((0,0,0))) * (1.0/len(verts))
                count = int(getattr(self, 'visualize_count', 128))
                step = max(1, len(verts)//count)
                aim_mode = getattr(context.scene.curiomesh_settings, 'ray_aim_mode', 'POLY_CENTER')
                # Choose aims on the target mesh
                tgt = obj
                tgt_me = tgt.data
                tgt_me.calc_loop_triangles()
                aims = []
                if aim_mode == 'VERTEX' and len(tgt_me.vertices) > 0:
                    aims = [tgt.matrix_world @ v.co for v in tgt_me.vertices]
                elif aim_mode == 'EDGE' and len(tgt_me.edges) > 0:
                    # Build edge -> polys adjacency from polygon edge_keys
                    edge_to_polys: dict[tuple[int,int], list[int]] = {}
                    for p in tgt_me.polygons:
                        for ek in p.edge_keys:
                            key = (min(ek[0], ek[1]), max(ek[0], ek[1]))
                            lst = edge_to_polys.get(key)
                            if lst is None:
                                edge_to_polys[key] = [p.index]
                            else:
                                lst.append(p.index)
                    # boundary edges (single poly) or feature edges (normal angle threshold)
                    for (v0, v1), polys in edge_to_polys.items():
                        add = False
                        if len(polys) == 1:
                            add = True
                        elif len(polys) >= 2:
                            try:
                                n0 = tgt_me.polygons[polys[0]].normal
                                n1 = tgt_me.polygons[polys[1]].normal
                                add = abs(n0.dot(n1)) < 0.85
                            except Exception:
                                add = False
                        if add:
                            a = tgt.matrix_world @ tgt_me.vertices[v0].co
                            b = tgt.matrix_world @ tgt_me.vertices[v1].co
                            aims.append((a + b) * 0.5)
                else:
                    for p in tgt_me.polygons:
                        c = Vector((0,0,0))
                        for li in p.loop_indices:
                            vi = tgt_me.loops[li].vertex_index
                            c += tgt.matrix_world @ tgt_me.vertices[vi].co
                        c *= (1.0/len(p.loop_indices))
                        aims.append(c)
                for i in range(0, len(verts), step):
                    p = verts[i]
                    # pick nearest aim
                    if aims:
                        closest = min(aims, key=lambda a: (a - p).length)
                    else:
                        closest = center
                    me_line = bpy.data.meshes.new(f"Ray{i}")
                    obj_line = bpy.data.objects.new(f"Ray{i}", me_line)
                    try:
                        rayc.objects.link(obj_line)
                    except Exception:
                        context.collection.objects.link(obj_line)
                    me_line.from_pydata([p, closest], [(0,1)], [])
                    obj_line.display_type = 'WIRE'
                    obj_line.show_in_front = True
                    obj_line.hide_select = True
        return {'FINISHED'}


class CURIOMESH_OT_exploded_bake(bpy.types.Operator):
    bl_idname = "curiomesh.exploded_bake"
    bl_label = "Exploded Bake Helper"
    bl_description = "Separate mesh by loose parts, bake individually to prevent crosstalk, then reassemble"
    bl_options = {'REGISTER', 'UNDO'}
    
    distance: bpy.props.FloatProperty(
        name="Explode Distance",
        description="How far to separate parts before baking",
        default=2.0,
        min=0.1,
        max=10.0,
    )
    
    reassemble: bpy.props.BoolProperty(
        name="Reassemble After Bake",
        description="Move parts back to original positions after baking",
        default=True,
    )
    
    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}
        
        # Ensure we're in object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Duplicate the object to work on
        work_obj = obj.copy()
        work_obj.data = obj.data.copy()
        context.collection.objects.link(work_obj)
        
        # Make work object active
        view_layer = context.view_layer
        view_layer.objects.active = work_obj
        work_obj.select_set(True)
        obj.select_set(False)
        
        # Enter edit mode and separate by loose parts
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Get all separated parts
        parts = [o for o in context.selected_objects if o.type == 'MESH']
        
        if len(parts) <= 1:
            self.report({'INFO'}, "Mesh has no loose parts to separate")
            # Clean up
            if work_obj in parts:
                bpy.data.objects.remove(work_obj, do_unlink=True)
            return {'FINISHED'}
        
        # Calculate bounding box center
        all_verts = []
        for part in parts:
            all_verts.extend([part.matrix_world @ v.co for v in part.data.vertices])
        
        if all_verts:
            xs = [v.x for v in all_verts]
            ys = [v.y for v in all_verts]
            zs = [v.z for v in all_verts]
            center = Vector((
                (min(xs) + max(xs)) * 0.5,
                (min(ys) + max(ys)) * 0.5,
                (min(zs) + max(zs)) * 0.5
            ))
        else:
            center = Vector((0, 0, 0))
        
        # Store original positions
        original_positions = {}
        
        # Explode parts outward from center
        for i, part in enumerate(parts):
            # Calculate part center
            part_verts = [part.matrix_world @ v.co for v in part.data.vertices]
            if part_verts:
                part_center = sum(part_verts, Vector((0, 0, 0))) / len(part_verts)
            else:
                part_center = part.location
            
            # Store original position
            original_positions[part] = part.location.copy()
            
            # Calculate explosion direction
            direction = (part_center - center)
            if direction.length > 0.001:
                direction.normalize()
            else:
                # Use a default direction based on index
                import math
                angle = (i / len(parts)) * 2 * math.pi
                direction = Vector((math.cos(angle), math.sin(angle), 0))
            
            # Move part
            part.location += direction * self.distance
        
        # TODO: Trigger baking here if integrated with main baking system
        # For now, just report success
        self.report({'INFO'}, f"Exploded {len(parts)} parts by {self.distance} units")
        
        # Reassemble if requested
        if self.reassemble:
            for part, orig_pos in original_positions.items():
                part.location = orig_pos
            
            # Join parts back together
            view_layer.objects.active = parts[0]
            for part in parts:
                part.select_set(True)
            bpy.ops.object.join()
            
            # Copy result back to original object
            obj.data = parts[0].data.copy()
            
            # Clean up
            bpy.data.objects.remove(parts[0], do_unlink=True)
        
        # Select original object
        obj.select_set(True)
        view_layer.objects.active = obj
        
        return {'FINISHED'}


def naive_tritoquad_on_bmesh(bm: bmesh.types.BMesh, protected_edges: set | None = None):
    """Simple tri-to-quad via pairing adjacent triangles.

    This is a placeholder so the add-on does something without native core.
    It greedily pairs triangles that share an edge and have a reasonable shape.
    """
    bm.faces.ensure_lookup_table()
    visited = set()
    pairs = []
    protected_edges = protected_edges or set()
    for f in bm.faces:
        if len(f.verts) != 3 or f.index in visited:
            continue
        found = None
        for e in f.edges:
            if e.is_boundary:
                continue
            key = tuple(sorted((e.verts[0].index, e.verts[1].index)))
            if key in protected_edges:
                continue
            other = next((lf for lf in e.link_faces if lf != f), None)
            if other and len(other.verts) == 3 and other.index not in visited:
                found = other
                break
        if found is not None:
            pairs.append((f, found))
            visited.add(f.index)
            visited.add(found.index)

    faces_to_join = [f for pair in pairs for f in pair]
    if faces_to_join:
        try:
            bmesh.ops.join_triangles(
                bm,
                faces=faces_to_join,
                angle_face_threshold=3.14159265,
                angle_shape_threshold=3.14159265,
            )
        except TypeError:
            # Older/newer Blender API variations may differ; fallback to edge dissolve + face make
            pass


