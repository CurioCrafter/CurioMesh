from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import traceback

import bpy


ADDON_ROOT = Path(__file__).resolve().parents[1]


def load_addon():
    for name in list(sys.modules):
        if name == "curiomesh" or name.startswith("curiomesh."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "curiomesh",
        ADDON_ROOT / "__init__.py",
        submodule_search_locations=[str(ADDON_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["curiomesh"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.register()
    return module


def reset_scene():
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object else None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def bake_settings_state(scene):
    bake = scene.render.bake
    return (
        scene.render.engine,
        scene.cycles.samples,
        bake.use_selected_to_active,
        bake.cage_extrusion,
        bake.margin,
        bake.use_pass_direct,
        bake.use_pass_indirect,
        bake.use_pass_color,
    )


def add_material_and_uv(obj, color=(0.7, 0.35, 0.1, 1.0)):
    material = bpy.data.materials.new(f"{obj.name}_Material")
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
    obj.data.materials.append(material)

    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active
    xs = [vertex.co.x for vertex in mesh.vertices] or [0.0]
    ys = [vertex.co.y for vertex in mesh.vertices] or [0.0]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    for poly in mesh.polygons:
        poly.material_index = 0
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            co = mesh.vertices[vertex_index].co
            uv_layer.data[loop_index].uv = ((co.x - min_x) / span_x, (co.y - min_y) / span_y)
    mesh.update()


def make_uv_sphere(name: str, segments=40, rings=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=1.0)
    obj = bpy.context.object
    obj.name = name
    add_material_and_uv(obj)
    return obj


def make_torus(name: str):
    bpy.ops.mesh.primitive_torus_add(major_segments=48, minor_segments=16, major_radius=1.2, minor_radius=0.28)
    obj = bpy.context.object
    obj.name = name
    add_material_and_uv(obj, color=(0.25, 0.5, 0.9, 1.0))
    return obj


def make_large_triangulated_grid(name: str, subdivisions: int = 513):
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=subdivisions,
        y_subdivisions=subdivisions,
        size=2.0,
    )
    obj = bpy.context.object
    obj.name = name
    modifier = obj.modifiers.new("CurioSmoke triangulate", "TRIANGULATE")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return obj


def make_open_grid(name: str, size=8, height=0.08):
    verts = []
    faces = []
    for y in range(size + 1):
        for x in range(size + 1):
            verts.append((x / size - 0.5, y / size - 0.5, height * ((x + y) % 2),))
    for y in range(size):
        for x in range(size):
            a = y * (size + 1) + x
            faces.append((a, a + 1, a + size + 2, a + size + 1))
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    add_material_and_uv(obj, color=(0.2, 0.8, 0.35, 1.0))
    return obj


def make_cube(name: str):
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    obj.name = name
    add_material_and_uv(obj, color=(0.9, 0.55, 0.2, 1.0))
    return obj


def make_bad_normals_sphere(name: str):
    obj = make_uv_sphere(name, segments=28, rings=14)
    obj.data.flip_normals()
    return obj


def make_non_manifold_cube(name: str):
    verts = [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (0, 3, 2, 1),
    ]
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    add_material_and_uv(obj, color=(0.55, 0.42, 0.28, 1.0))
    return obj


def run_remesh_case(
    name: str,
    obj,
    kwargs,
    *,
    min_quad_ratio=0.55,
    require_uv=True,
    require_material=True,
    max_face_error=None,
    require_quad_only=False,
):
    reset_selection(obj)
    result = bpy.ops.curiomesh.remesh(**kwargs)
    if "FINISHED" not in result:
        raise AssertionError(f"{name}: operator returned {result}")

    output = bpy.context.view_layer.objects.active
    if output is None or output.type != "MESH":
        raise AssertionError(f"{name}: no active mesh output")
    if len(output.data.polygons) == 0:
        raise AssertionError(f"{name}: empty output mesh")
    if require_material and len(output.data.materials) == 0:
        raise AssertionError(f"{name}: materials were not preserved")
    if require_uv and not output.data.uv_layers:
        raise AssertionError(f"{name}: UV layer missing")

    settings = bpy.context.scene.curiomesh_settings
    if settings.metrics_faces <= 0:
        raise AssertionError(f"{name}: diagnostics did not update")
    if settings.metrics_quad_ratio < min_quad_ratio:
        raise AssertionError(
            f"{name}: quad ratio {settings.metrics_quad_ratio:.3f} below {min_quad_ratio:.3f}"
        )
    if settings.metrics_status != "OK":
        raise AssertionError(f"{name}: status {settings.metrics_status}")
    if require_quad_only and not settings.metrics_quad_only:
        raise AssertionError(f"{name}: exact quad-only validation did not pass")
    if require_uv and settings.metrics_uv_status == "FAILED":
        raise AssertionError(f"{name}: UV preservation failed")
    if max_face_error is not None and settings.metrics_face_error > max_face_error:
        raise AssertionError(
            f"{name}: face error {settings.metrics_face_error:.3f} above {max_face_error:.3f}"
        )

    print(
        f"PASS {name}: faces={settings.metrics_faces} "
        f"quad_ratio={settings.metrics_quad_ratio:.3f} uv={settings.metrics_uv_status}"
    )
    return output


def reset_selection(obj):
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object else None
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def run_bake_rollback_case():
    source = make_cube("CurioSmoke_BakeRollbackSource")
    target = make_cube("CurioSmoke_BakeRollbackTarget")
    reset_selection(source)

    scene = bpy.context.scene
    expected_settings = bake_settings_state(scene)
    original_materials = list(target.data.materials)
    original_material_names = set(bpy.data.materials.keys())
    original_image_names = set(bpy.data.images.keys())
    original_nodes = [
        (node.name, node.label, node.type)
        for material in original_materials
        if material is not None and material.use_nodes
        for node in material.node_tree.nodes
    ]

    textures = sys.modules["curiomesh.textures"]
    execute_bake = textures._execute_bake
    textures._execute_bake = lambda _bake_type: {"CANCELLED"}
    try:
        try:
            textures.bake_selected_to_active(
                source,
                target,
                maps=("COLOR",),
                resolution=64,
                margin=3,
            )
        except RuntimeError as exc:
            if "did not finish" not in str(exc):
                raise AssertionError(f"bake_rollback: unexpected failure: {exc}") from exc
        else:
            raise AssertionError("bake_rollback: CANCELLED bake was accepted")
    finally:
        textures._execute_bake = execute_bake

    if len(target.data.materials) != len(original_materials) or any(
        restored is not original for restored, original in zip(target.data.materials, original_materials)
    ):
        raise AssertionError("bake_rollback: target material slots were not restored")
    if set(bpy.data.materials.keys()) != original_material_names:
        raise AssertionError("bake_rollback: a private bake material leaked")
    if set(bpy.data.images.keys()) != original_image_names:
        raise AssertionError("bake_rollback: a bake image leaked")
    restored_nodes = [
        (node.name, node.label, node.type)
        for material in target.data.materials
        if material is not None and material.use_nodes
        for node in material.node_tree.nodes
    ]
    if restored_nodes != original_nodes:
        raise AssertionError("bake_rollback: original material nodes were mutated")
    if bake_settings_state(scene) != expected_settings:
        raise AssertionError("bake_rollback: render settings were not restored")
    if bpy.context.view_layer.objects.active is not source or [obj.name for obj in bpy.context.selected_objects] != [
        source.name
    ]:
        raise AssertionError("bake_rollback: active object or selection was not restored")
    print("PASS bake_rollback: cancelled bake left no image, material, node, or settings state")


def run_transformed_surface_error_case():
    source = make_cube("CurioSmoke_TransformSource")
    source.location = (17.0, -4.5, 3.25)
    source.rotation_euler = (0.37, -0.21, 1.13)
    source.scale = (2.0, 0.75, 1.4)
    bpy.context.view_layer.update()
    target = bpy.data.objects.new("CurioSmoke_TransformTarget", source.data.copy())
    bpy.context.collection.objects.link(target)
    target.matrix_world = source.matrix_world.copy()
    bpy.context.view_layer.update()

    bridge = sys.modules["curiomesh.bridge"]
    rms, p95 = bridge._sampled_surface_error(source, target)
    if rms > 1e-6 or p95 > 1e-6:
        raise AssertionError(
            f"transformed_surface_error: identical world surfaces scored rms={rms:.12f} p95={p95:.12f}"
        )
    print("PASS transformed_surface_error: world-space BVH scoring is transform invariant")


def run_bounded_fidelity_reference_case():
    source = make_open_grid("CurioSmoke_BoundedReferenceSource", size=8, height=0.0)
    source.location = (9.0, -3.0, 2.0)
    source.rotation_euler = (0.21, -0.33, 0.71)
    source.scale = (2.5, 0.8, 1.25)
    bpy.context.view_layer.update()

    proxy_mesh = bpy.data.meshes.new("CurioSmoke_BoundedReferenceProxyMesh")
    proxy_mesh.from_pydata(
        [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    proxy_mesh.update(calc_edges=True)
    proxy = bpy.data.objects.new("CurioSmoke_BoundedReferenceProxy", proxy_mesh)
    bpy.context.collection.objects.link(proxy)
    proxy.matrix_world = source.matrix_world.copy()
    target = bpy.data.objects.new("CurioSmoke_BoundedReferenceTarget", proxy_mesh.copy())
    bpy.context.collection.objects.link(target)
    target.matrix_world = source.matrix_world.copy()
    bpy.context.view_layer.update()

    bridge = sys.modules["curiomesh.bridge"]
    original_builder = bridge._build_source_bvh

    def reject_full_source_bvh(obj):
        if obj is source:
            raise AssertionError("bounded_fidelity_reference: attempted a full-source Python BVH")
        return original_builder(obj)

    bridge._build_source_bvh = reject_full_source_bvh
    try:
        reference = bridge._make_surface_reference(
            source,
            sample_limit=64,
            reference_proxy=proxy,
            max_reference_triangles=8,
        )
        stats = bridge._sampled_surface_error_stats(source, target, reference=reference)
    finally:
        bridge._build_source_bvh = original_builder

    if not reference.fidelity_status.startswith("BOUNDED_PROXY_BVH_2_TRIS_SOURCE_GE_"):
        raise AssertionError(f"bounded_fidelity_reference: wrong status {reference.fidelity_status!r}")
    if not reference.component_sampling.startswith("BOUNDED_SOURCE_FACES_"):
        raise AssertionError(
            f"bounded_fidelity_reference: source sampling was not bounded: {reference.component_sampling!r}"
        )
    if len(reference.sample_points) > 64:
        raise AssertionError("bounded_fidelity_reference: sample budget was exceeded")
    if stats.rms > 1e-6 or stats.p95 > 1e-6:
        raise AssertionError(
            f"bounded_fidelity_reference: planar world-space proxy scored rms={stats.rms} p95={stats.p95}"
        )
    if stats.fidelity_status != reference.fidelity_status:
        raise AssertionError("bounded_fidelity_reference: status was lost during scoring")
    print("PASS bounded_fidelity_reference: forced cap used bounded source samples and proxy BVH")


def run_large_retry_proxy_order_case():
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    source = bpy.context.object
    source.name = "CurioSmoke_LargeRetryOrder"
    bridge = sys.modules["curiomesh.bridge"]
    events = []
    quadriflow_calls = 0

    original_count = bridge._bounded_triangle_count
    original_prepare = bridge._prepare_flow_proxy
    original_cleanup = bridge._cleanup_mesh
    original_quadriflow = bridge._run_quadriflow
    original_voxel = bridge._voxel_repair

    def force_large_source(obj, limit):
        if obj.name.startswith(f"{source.name}_CurioMeshSource"):
            return limit + 1, True
        return original_count(obj, limit)

    def record_prepare(obj, config):
        events.append("prepare")
        return original_prepare(obj, config)

    def record_cleanup(mesh, config):
        events.append("cleanup")
        return original_cleanup(mesh, config)

    def fail_then_succeed(_obj, _config):
        nonlocal quadriflow_calls
        quadriflow_calls += 1
        events.append(f"quadriflow_{quadriflow_calls}")
        return quadriflow_calls == 2

    def record_voxel(_obj, _config):
        events.append("voxel")
        return True

    bridge._bounded_triangle_count = force_large_source
    bridge._prepare_flow_proxy = record_prepare
    bridge._cleanup_mesh = record_cleanup
    bridge._run_quadriflow = fail_then_succeed
    bridge._voxel_repair = record_voxel
    try:
        result = bridge.run_curiomesh(
            source,
            bridge.RemeshConfig(
                engine="QUADRIFLOW",
                target_faces=24,
                topology_policy="REPAIR",
                require_quads=False,
                shrinkwrap_project=False,
                texture_mode="NONE",
                output_mode="NEW",
            ),
        )
    finally:
        bridge._bounded_triangle_count = original_count
        bridge._prepare_flow_proxy = original_prepare
        bridge._cleanup_mesh = original_cleanup
        bridge._run_quadriflow = original_quadriflow
        bridge._voxel_repair = original_voxel

    expected = [
        "prepare",
        "cleanup",
        "quadriflow_1",
        "prepare",
        "cleanup",
        "voxel",
        "quadriflow_2",
    ]
    if events != expected:
        raise AssertionError(f"large_retry_proxy_order: expected {expected}, got {events}")
    if not result.success:
        raise AssertionError(f"large_retry_proxy_order: retry did not complete: {result.message}")
    print("PASS large_retry_proxy_order: retry QEM reduction precedes BMesh cleanup")


def run_quality_gate_case():
    atlasq = sys.modules["curiomesh.atlasq"]
    metrics = sys.modules["curiomesh.metrics"]
    diagnostics = metrics.MeshDiagnostics(
        faces=100,
        quads=100,
        quad_ratio=1.0,
        min_scaled_jacobian=0.8,
        p05_scaled_jacobian=0.7,
        p95_aspect_ratio=1.4,
        quad_only=True,
        connected_components=1,
    )
    failures = atlasq.candidate_gate_failures(
        diagnostics,
        target_faces=100,
        geometry_rms=0.05,
        geometry_p95=0.10,
        geometry_max=0.12,
        target_edge_normalized=0.01,
        require_quads=True,
        topology_policy="REPAIR",
        profile=None,
        quality="BALANCED",
    )
    if not any("p95 error" in failure for failure in failures):
        raise AssertionError(f"quality_gate: catastrophic finite p95 error was accepted: {failures}")
    max_failures = atlasq.candidate_gate_failures(
        diagnostics,
        target_faces=100,
        geometry_rms=0.002,
        geometry_p95=0.005,
        geometry_max=0.05,
        target_edge_normalized=0.01,
        require_quads=True,
        topology_policy="REPAIR",
        profile=None,
        quality="BALANCED",
    )
    if not any("max sampled error" in failure for failure in max_failures):
        raise AssertionError(f"quality_gate: isolated maximum error was accepted: {max_failures}")
    print("PASS quality_gate: finite but excessive geometry error is rejected")


def run_component_sampling_case():
    source_mesh = bpy.data.meshes.new("CurioSmoke_ComponentSourceMesh")
    main_vertices = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    tiny_vertices = [(6.0, 0.0, 0.0), (6.05, 0.0, 0.0), (6.0, 0.05, 0.0)]
    source_mesh.from_pydata(main_vertices + tiny_vertices, [], [(0, 1, 2, 3), (4, 5, 6)])
    source_mesh.update(calc_edges=True)
    source = bpy.data.objects.new("CurioSmoke_ComponentSource", source_mesh)
    bpy.context.collection.objects.link(source)

    target_mesh = bpy.data.meshes.new("CurioSmoke_ComponentTargetMesh")
    target_mesh.from_pydata(main_vertices, [], [(0, 1, 2, 3)])
    target_mesh.update(calc_edges=True)
    target = bpy.data.objects.new("CurioSmoke_ComponentTarget", target_mesh)
    bpy.context.collection.objects.link(target)

    bridge = sys.modules["curiomesh.bridge"]
    stats = bridge._sampled_surface_error_stats(source, target, sample_limit=32)
    if stats.component_sampling != "ALL_2":
        raise AssertionError(f"component_sampling: expected both islands, got {stats.component_sampling}")
    if stats.maximum < 0.5:
        raise AssertionError(f"component_sampling: deleted small island was missed, max={stats.maximum}")
    print("PASS component_sampling: omitted tiny disconnected island contributes to max error")


def run_world_scale_metrics_case():
    mesh = bpy.data.meshes.new("CurioSmoke_ScaledMetricsMesh")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new("CurioSmoke_ScaledMetrics", mesh)
    bpy.context.collection.objects.link(obj)
    obj.scale = (100.0, 0.01, 1.0)
    bpy.context.view_layer.update()

    metrics = sys.modules["curiomesh.metrics"].mesh_diagnostics(obj, 100)
    if metrics.p95_aspect_ratio < 5_000.0:
        raise AssertionError(f"world_scale_metrics: rendered aspect was underreported: {metrics.p95_aspect_ratio}")
    profile = sys.modules["curiomesh.atlasq"].analyze_object(obj, 100)
    normalized_edge = profile.target_edge_length / profile.diagonal
    if not 0.0009 <= normalized_edge <= 0.0011:
        raise AssertionError(f"world_scale_metrics: target-edge normalization is not world-space: {normalized_edge}")
    print("PASS world_scale_metrics: non-uniform scale uses world-space area, diagonal, and quad quality")


def main():
    module = load_addon()
    reset_scene()
    run_transformed_surface_error_case()
    run_bounded_fidelity_reference_case()
    run_large_retry_proxy_order_case()
    run_quality_gate_case()
    run_component_sampling_case()
    run_world_scale_metrics_case()
    reset_scene()

    sphere = make_uv_sphere("CurioSmoke_Sphere")
    out = run_remesh_case(
        "sphere_new",
        sphere,
        {
            "engine": "QUADRIFLOW",
            "target_faces": 260,
            "quality": "BALANCED",
            "output_mode": "NEW",
            "texture_mode": "PROJECT",
            "seed": 7,
        },
        min_quad_ratio=0.75,
        max_face_error=0.35,
    )
    if out.name == sphere.name:
        raise AssertionError("sphere_new: NEW mode reused original object")

    torus = make_torus("CurioSmoke_Torus")
    run_remesh_case(
        "torus_replace",
        torus,
        {
            "engine": "QUADRIFLOW",
            "target_faces": 220,
            "quality": "BALANCED",
            "output_mode": "REPLACE",
            "texture_mode": "PROJECT",
            "seed": 11,
        },
        min_quad_ratio=0.70,
        max_face_error=0.35,
    )

    grid = make_open_grid("CurioSmoke_OpenGrid")
    run_remesh_case(
        "open_grid_boundary",
        grid,
        {
            "engine": "QUADRIFLOW",
            "target_faces": 90,
            "quality": "DRAFT",
            "output_mode": "NEW",
            "texture_mode": "PROJECT",
            "preserve_boundary": True,
            "require_quads": False,
            "seed": 3,
        },
        min_quad_ratio=0.50,
    )

    bad_normals = make_bad_normals_sphere("CurioSmoke_BadNormals")
    run_remesh_case(
        "bad_normals_repair",
        bad_normals,
        {
            "engine": "QUADRIFLOW",
            "target_faces": 180,
            "quality": "BALANCED",
            "output_mode": "REPLACE",
            "texture_mode": "PROJECT",
            "voxel_repair": True,
            "seed": 5,
        },
        min_quad_ratio=0.65,
        max_face_error=0.35,
    )

    bake_source = make_uv_sphere("CurioSmoke_Bake", segments=20, rings=10)
    source_material = bake_source.data.materials[0]
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.cycles.samples = 7
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.cage_extrusion = 0.123
    scene.render.bake.margin = 19
    scene.render.bake.use_pass_direct = True
    scene.render.bake.use_pass_indirect = True
    scene.render.bake.use_pass_color = False
    expected_bake_settings = bake_settings_state(scene)
    bake_output = run_remesh_case(
        "bake_color",
        bake_source,
        {
            "engine": "QUADRIFLOW",
            "target_faces": 96,
            "quality": "DRAFT",
            "output_mode": "NEW",
            "texture_mode": "BAKE",
            "bake_resolution": 64,
            "bake_margin": 2,
            "seed": 17,
        },
        min_quad_ratio=0.60,
        max_face_error=0.60,
    )
    if bake_settings_state(scene) != expected_bake_settings:
        raise AssertionError(
            f"bake_color: render settings leaked; expected={expected_bake_settings!r} "
            f"actual={bake_settings_state(scene)!r}"
        )
    if bake_output.data.materials[0] == source_material:
        raise AssertionError("bake_color: bake output still shares its source material")
    if any(node.label.startswith("CurioMesh_") for node in source_material.node_tree.nodes):
        raise AssertionError("bake_color: source material nodes were mutated")
    baked_nodes = [
        node
        for material in bake_output.data.materials
        if material is not None and material.use_nodes
        for node in material.node_tree.nodes
        if node.label == "CurioMesh_BaseColor" and node.image is not None
    ]
    if not baked_nodes:
        raise AssertionError("bake_color: baked image node was not retained on the output")
    run_bake_rollback_case()

    triad_cube = make_cube("CurioSmoke_TriadCube")
    run_remesh_case(
        "triad_cube_new",
        triad_cube,
        {
            "engine": "TRIAD_Q_LITE",
            "target_faces": 24,
            "quality": "BALANCED",
            "output_mode": "NEW",
            "texture_mode": "PROJECT",
            "triad_seed_count": 4,
            "triad_flow_mode": "PATCH",
            "triad_feature_angle": 35.0,
            "debug_logs": True,
        },
        min_quad_ratio=0.85,
        max_face_error=0.20,
        require_quad_only=True,
    )

    dirty_source = make_non_manifold_cube("CurioSmoke_Dirty")
    run_remesh_case(
        "atlas_dirty_repair",
        dirty_source,
        {
            "engine": "ATLAS_Q",
            "target_faces": 48,
            "quality": "DRAFT",
            "atlas_strategy": "DIRTY_FLOW",
            "atlas_candidate_count": 1,
            "topology_policy": "REPAIR",
            "require_quads": True,
            "output_mode": "NEW",
            "texture_mode": "NONE",
            "seed": 29,
        },
        min_quad_ratio=1.0,
        require_uv=False,
        max_face_error=0.75,
        require_quad_only=True,
    )

    large_source = make_large_triangulated_grid("CurioSmoke_LargeProxy")
    large_input_faces = len(large_source.data.polygons)
    if large_input_faces <= 500_000:
        raise AssertionError(f"atlas_target_scale_proxy: test did not enter sampled branch ({large_input_faces})")
    large_output = run_remesh_case(
        "atlas_target_scale_proxy",
        large_source,
        {
            "engine": "ATLAS_Q",
            "target_faces": 128,
            "quality": "DRAFT",
            "atlas_strategy": "FLOW_ENSEMBLE",
            "atlas_candidate_count": 1,
            "topology_policy": "PRESERVE",
            "require_quads": True,
            "output_mode": "NEW",
            "texture_mode": "NONE",
            "seed": 31,
        },
        min_quad_ratio=1.0,
        require_uv=False,
        require_material=False,
        max_face_error=0.45,
        require_quad_only=True,
    )
    proxy_faces = int(large_output.get("curiomesh_proxy_faces", 0))
    if proxy_faces <= 0 or proxy_faces >= large_input_faces:
        raise AssertionError(
            f"atlas_target_scale_proxy: expected reduced proxy, input={large_input_faces} proxy={proxy_faces}"
        )
    topology_status = str(large_output.get("curiomesh_topology_status", ""))
    if topology_status != "PROXY_PRESERVED_SOURCE_UNVERIFIED":
        raise AssertionError(f"atlas_target_scale_proxy: unexpected topology status {topology_status!r}")
    if bpy.context.scene.curiomesh_settings.metrics_topology_status != topology_status:
        raise AssertionError("atlas_target_scale_proxy: proxy-only topology status was not surfaced in UI metrics")
    reset_scene()

    triad_grid = make_open_grid("CurioSmoke_TriadGrid", size=10, height=0.0)
    triad_input_faces = len(triad_grid.data.polygons)
    triad_grid_output = run_remesh_case(
        "triad_grid_proxy",
        triad_grid,
        {
            "engine": "TRIAD_Q_LITE",
            "target_faces": 196,
            "quality": "DRAFT",
            "output_mode": "NEW",
            "texture_mode": "PROJECT",
            "triad_seed_count": 4,
            "triad_flow_mode": "AUTO",
            "triad_feature_angle": 35.0,
            "debug_logs": True,
        },
        min_quad_ratio=0.70,
        require_quad_only=True,
    )
    triad_proxy_faces = int(triad_grid_output.get("curiomesh_proxy_faces", 0))
    if triad_proxy_faces <= 0 or triad_proxy_faces >= triad_input_faces:
        raise AssertionError(
            f"triad_grid_proxy: expected target-scale QEM proxy before closure, got {triad_proxy_faces}"
        )

    atlas_source = make_uv_sphere("CurioSmoke_Atlas", segments=20, rings=10)
    reset_selection(atlas_source)
    analysis_result = bpy.ops.curiomesh.analyze()
    if "FINISHED" not in analysis_result:
        raise AssertionError(f"atlas_analyze: operator returned {analysis_result}")
    settings = bpy.context.scene.curiomesh_settings
    if not settings.analysis_profile or settings.analysis_faces <= 0:
        raise AssertionError("atlas_analyze: scan profile was not populated")
    run_remesh_case(
        "atlas_auto",
        atlas_source,
        {
            "engine": "ATLAS_Q",
            "target_faces": 96,
            "quality": "BALANCED",
            "atlas_strategy": "AUTO",
            "atlas_candidate_count": 2,
            "topology_policy": "PRESERVE",
            "require_quads": True,
            "output_mode": "NEW",
            "texture_mode": "NONE",
            "seed": 23,
        },
        min_quad_ratio=1.0,
        require_uv=False,
        max_face_error=0.40,
        require_quad_only=True,
    )

    module.unregister()
    print("CurioMesh Blender smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
