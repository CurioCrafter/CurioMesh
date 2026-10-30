# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import traceback

import bmesh
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


def run_pairclosure_skips_flow_proxy_case():
    reset_scene()
    bpy.ops.mesh.primitive_cube_add()
    source = bpy.context.object
    source.name = "CurioSmoke_PairClosureDirectProxy"
    bridge = sys.modules["curiomesh.bridge"]
    original_prepare = bridge._prepare_flow_proxy
    calls = []

    def fail_if_called(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("explicit PairClosure entered the flow-proxy path")

    bridge._prepare_flow_proxy = fail_if_called
    try:
        result = bridge.run_curiomesh(
            source,
            bridge.RemeshConfig(
                engine="ATLAS_Q",
                atlas_strategy="PAIR_CLOSURE",
                target_faces=24,
                atlas_candidate_count=1,
                voxel_repair=False,
                shrinkwrap_project=False,
                texture_mode="NONE",
                output_mode="NEW",
            ),
        )
    finally:
        bridge._prepare_flow_proxy = original_prepare

    if calls:
        raise AssertionError("pairclosure_direct_proxy: flow preprocessing was called")
    if not result.success or result.output_object is None:
        raise AssertionError(f"pairclosure_direct_proxy: {result.message}")
    print("PASS pairclosure_direct_proxy: explicit PairClosure bypassed redundant flow QEM")


def run_pairclosure_unlock_topology_gate_case():
    bridge = sys.modules["curiomesh.bridge"]

    reset_scene()
    bpy.ops.mesh.primitive_torus_add(major_segments=12, minor_segments=6)
    rotated_source = bpy.context.object
    rotated_source.name = "CurioSmoke_PairUnlockRotation"
    triangulate = rotated_source.modifiers.new("Triangulate", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)
    before = bridge.mesh_diagnostics(rotated_source, 128)
    selected, rotated = bridge._rotate_independent_triangle_edges(rotated_source, 1001)
    after = bridge.mesh_diagnostics(rotated_source, 128)
    if selected <= 0 or rotated <= 0:
        raise AssertionError("pair_unlock_gate: deterministic rotation selected no usable edges")
    if after.component_signature != before.component_signature:
        raise AssertionError("pair_unlock_gate: edge rotation changed the component topology signature")
    if not bridge._proxy_topology_certifiable(after):
        raise AssertionError("pair_unlock_gate: edge rotation introduced a topology defect")

    reset_scene()
    bpy.ops.mesh.primitive_cube_add()
    rollback_source = bpy.context.object
    rollback_source.name = "CurioSmoke_PairUnlockRollback"
    triangulate = rollback_source.modifiers.new("Triangulate", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)
    rollback_before = bridge.mesh_diagnostics(rollback_source, 24)
    original_rotate = bridge._rotate_independent_triangle_edges
    original_decimate = bridge._apply_decimate_proxy

    def damage_topology(obj, _seed, *, fraction=0.18):
        del fraction
        bm = bridge.bmesh.new()
        try:
            bm.from_mesh(obj.data)
            bm.faces.ensure_lookup_table()
            bridge.bmesh.ops.delete(bm, geom=[bm.faces[0]], context="FACES")
            bm.to_mesh(obj.data)
            obj.data.update(calc_edges=True)
        finally:
            bm.free()
        return 1, 1

    def keep_damaged_proxy(_obj, _target, _config, *, preserve_features):
        del preserve_features
        return True

    bridge._rotate_independent_triangle_edges = damage_topology
    bridge._apply_decimate_proxy = keep_damaged_proxy
    try:
        accepted = bridge._unlock_pair_closure_proxy(
            rollback_source,
            4,
            4,
            bridge.RemeshConfig(seed=9, debug_logs=True),
            max_rounds=1,
        )
    finally:
        bridge._rotate_independent_triangle_edges = original_rotate
        bridge._apply_decimate_proxy = original_decimate

    rollback_after = bridge.mesh_diagnostics(rollback_source, 24)
    if accepted != 0:
        raise AssertionError("pair_unlock_gate: topology-changing round was accepted")
    if rollback_after.component_signature != rollback_before.component_signature:
        raise AssertionError("pair_unlock_gate: rejected round did not restore exact topology")
    if rollback_after.faces != rollback_before.faces:
        raise AssertionError("pair_unlock_gate: rejected round did not restore the prior mesh")
    print("PASS pair_unlock_gate: deterministic rotations preserve topology and invalid rounds roll back")


def run_projection_rollback_case():
    bridge = sys.modules["curiomesh.bridge"]
    original_projection = bridge._apply_shrinkwrap
    cases = (
        (True, "ROLLED_BACK_FAILED_GATES"),
        (False, "ROLLED_BACK_APPLY_FAILED"),
    )
    for reports_applied, expected_status in cases:
        reset_scene()
        bpy.ops.mesh.primitive_cube_add()
        source = bpy.context.object
        source.name = f"CurioSmoke_ProjectionRollback_{expected_status}"

        def collapse_projection(target, _source, _config):
            origin = target.data.vertices[0].co.copy()
            for vertex in target.data.vertices:
                vertex.co = origin
            target.data.update()
            return reports_applied

        bridge._apply_shrinkwrap = collapse_projection
        try:
            result = bridge.run_curiomesh(
                source,
                bridge.RemeshConfig(
                    engine="TRIAD_Q_LITE",
                    target_faces=24,
                    triad_force_quads=True,
                    shrinkwrap_project=True,
                    texture_mode="NONE",
                    output_mode="NEW",
                ),
            )
        finally:
            bridge._apply_shrinkwrap = original_projection

        if not result.success or result.output_object is None:
            raise AssertionError(f"projection_rollback: validated pre-projection mesh was lost: {result.message}")
        if result.output_object.get("curiomesh_projection_status") != expected_status:
            raise AssertionError(
                f"projection_rollback: expected {expected_status}, got "
                f"{result.output_object.get('curiomesh_projection_status')}"
            )
        diagnostics = bridge.mesh_diagnostics(result.output_object, 24)
        if not diagnostics.quad_only:
            raise AssertionError("projection_rollback: restored output no longer passes quad validation")
    print("PASS projection_rollback: invalid and failed shrinkwrap results restored the validated candidate")


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


def run_atlas_pragmatic_gates_case():
    atlasq = sys.modules["curiomesh.atlasq"]
    metrics = sys.modules["curiomesh.metrics"]

    def diag(**overrides):
        values = dict(
            faces=1000,
            quads=1000,
            quad_ratio=1.0,
            min_scaled_jacobian=0.8,
            p05_scaled_jacobian=0.7,
            p95_aspect_ratio=1.4,
            quad_only=True,
            connected_components=1,
        )
        values.update(overrides)
        return metrics.MeshDiagnostics(**values)

    def profile(**overrides):
        values = dict(
            category="OPEN_SURFACE", recommended_strategy="OPEN_FLOW", face_count=6000, vertex_count=3000,
            sampled_faces=6000, exact_topology=True, topology_basis="SOURCE_EXACT", boundary_edges=40,
            boundary_loops=2, connected_components=2, component_euler=(1, 1), component_signature=((1, 1), (1, 1)),
            non_manifold_edges=0, non_manifold_vertices=0, branched_boundary_vertices=0, inconsistent_edges=0,
            duplicate_faces=0, euler_characteristic=2, degenerate_ratio=0.0, sharp_ratio=0.0, triangle_ratio=1.0,
            target_edge_length=0.05, diagonal=1.0, texture_critical=False, warnings=(),
            component_detail=((1, 1, 5990), (1, 1, 10)),
        )
        values.update(overrides)
        return atlasq.ScanProfile(**values)

    def gate(diagnostics, policy="REPAIR", source=None):
        return atlasq.candidate_gate_failures(
            diagnostics, target_faces=1000, geometry_rms=0.0, geometry_p95=0.0, geometry_max=0.0,
            target_edge_normalized=0.05, require_quads=True, topology_policy=policy, profile=source,
            quality="BALANCED",
        )

    if gate(diag(invalid_quads=20, quad_only=False, min_scaled_jacobian=-0.2)):
        raise AssertionError("atlas_gates: 2% invalid quads should be tolerated")
    if not any("invalid quads" in item for item in gate(diag(invalid_quads=50, quad_only=False))):
        raise AssertionError("atlas_gates: 5% invalid quads must still fail")
    kept = diag(faces=1000, component_detail=((1, 1, 990), (1, 1, 1)), component_signature=((1, 1), (1, 1)))
    if any("shell" in item or "per-component" in item for item in gate(kept, "PRESERVE", profile())):
        raise AssertionError("atlas_gates: a vanished tiny shell must not fail PRESERVE")
    lost = diag(faces=1000, component_detail=((1, 0, 990), (1, 1, 1)), component_signature=((1, 0), (1, 1)))
    if not gate(lost, "PRESERVE", profile()):
        raise AssertionError("atlas_gates: a changed significant shell must fail PRESERVE")

    dirty = profile(category="DIRTY_SCAN", recommended_strategy="DIRTY_FLOW")
    for policy, expected in (("PRESERVE", ["PAIR_CLOSURE"]), ("REPAIR", ["DIRTY_FLOW"])):
        specs = atlasq.build_candidate_specs(
            dirty, strategy="AUTO", quality="BALANCED", seed=0, candidate_count=3, topology_policy=policy,
        )
        if specs[0].strategy != expected[0]:
            raise AssertionError(f"atlas_gates: {policy} dirty scan routed to {specs[0].strategy}")
    huge = profile(face_count=300_000, connected_components=1, boundary_edges=0, recommended_strategy="FLOW_ENSEMBLE")
    if not atlasq.build_candidate_specs(
        huge, strategy="AUTO", quality="BALANCED", seed=0, candidate_count=3, topology_policy="REPAIR",
    )[0].pair_closure:
        raise AssertionError("atlas_gates: a >200k-face source must go PairClosure first")
    shells = profile(connected_components=5, boundary_edges=9)
    specs = atlasq.build_candidate_specs(
        shells, strategy="FLOW_ENSEMBLE", quality="BALANCED", seed=0, candidate_count=3, topology_policy="REPAIR",
    )
    if [spec.pair_closure for spec in specs] != [True]:
        raise AssertionError("atlas_gates: multi-shell open mesh must not run QuadriFlow")

    pair = atlasq.CandidateSpec("PAIR_CLOSURE", "pair", 0, pair_closure=True)
    flow = atlasq.CandidateSpec("FLOW_ENSEMBLE", "flow", 0)
    wide = diag(p95_aspect_ratio=10.0)
    for spec, valid in ((pair, True), (flow, False)):
        evaluation = atlasq.evaluate_candidate(
            spec, wide, target_faces=1000, geometry_rms=0.0, geometry_p95=0.0, geometry_max=0.0,
            target_edge_normalized=0.05, require_quads=True, topology_policy="REPAIR", profile=profile(),
            quality="BALANCED", elapsed_ms=0.0,
        )
        if evaluation.valid != valid or bool(evaluation.warnings) != valid:
            raise AssertionError(f"atlas_gates: {spec.label} aspect handling wrong: {evaluation.reason}")
    print("PASS atlas_gates: bad-quad tolerance, tiny shells, pair-first routing, soft fallback")


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


def run_ui_features_case(module):
    from curiomesh import batch, live, overlay

    reset_scene()
    settings = bpy.context.scene.curiomesh_settings

    # Toggles must not raise.
    cube = make_cube("CurioSmoke_UiCube")
    reset_selection(cube)
    for name in ("hud_enabled", "heatmap_enabled"):
        setattr(settings, name, True)
        setattr(settings, name, False)
    if "FINISHED" not in bpy.ops.curiomesh.toggle_heatmap() or not settings.heatmap_enabled:
        raise AssertionError("ui: toggle_heatmap did not enable the heatmap")
    bpy.ops.curiomesh.toggle_heatmap()

    # Stats, heatmap arrays and hints on a mesh with a triangle and an n-gon.
    stats = overlay.get_stats(cube)
    if (stats.faces, stats.quads, stats.tris, stats.poles) != (6, 6, 0, 8):
        raise AssertionError(f"ui: cube stats {stats}")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")
    overlay.invalidate()
    stats = overlay.get_stats(cube)
    if stats.tris != 12 or stats.quads != 0:
        raise AssertionError(f"ui: triangulated cube stats {stats}")
    tri_pos, tri_col, pt_pos, pt_col = overlay.build_heat_arrays(cube, stats)
    if len(tri_pos) != 36 or len(tri_col) != 36 or len(pt_pos) != len(pt_col):
        raise AssertionError("ui: heatmap arrays have wrong sizes")
    hint = overlay.next_step_hint(cube, stats, settings)
    if "Remesh" not in hint:
        raise AssertionError(f"ui: unexpected hint {hint!r}")
    if overlay.next_step_hint(None, None, settings) != "Select a mesh":
        raise AssertionError("ui: no-object hint")
    model = overlay.hud_model(bpy.context)
    if model["tris"] != 12 or model["target"] != settings.target_faces:
        raise AssertionError(f"ui: hud model {model}")

    # Batch on two cubes (QuadriFlow keeps this fast).
    reset_scene()
    first, second = make_cube("CurioSmoke_BatchA"), make_cube("CurioSmoke_BatchB")
    bpy.ops.object.select_all(action="DESELECT")
    first.select_set(True)
    second.select_set(True)
    bpy.context.view_layer.objects.active = second
    settings.engine = "QUADRIFLOW"
    settings.target_faces = 96
    settings.texture_mode = "NONE"
    if "FINISHED" not in bpy.ops.curiomesh.batch_remesh():
        raise AssertionError("batch: operator did not finish")
    if len(settings.batch_results) != 2 or not all(item.ok for item in settings.batch_results):
        raise AssertionError(f"batch: results {[(i.name, i.ok, i.error) for i in settings.batch_results]}")
    report = Path(settings.batch_report_path)
    if not report.is_file() or len(json.loads(report.read_text(encoding="utf-8"))["objects"]) != 2:
        raise AssertionError("batch: JSON report missing or incomplete")
    report.unlink()
    bad = batch.report_row("Ghost", None, 1.0, "boom")
    if bad["success"] or bad["error"] != "boom":
        raise AssertionError("batch: failure row")

    # Live preview: cap message and a real preview on a small mesh.
    reset_scene()
    source = make_uv_sphere("CurioSmoke_LiveSrc", segments=16, rings=8)
    reset_selection(source)
    settings.live_enabled = True
    if not live.run_preview(bpy.context):
        raise AssertionError(f"live: preview failed: {live.status()}")
    preview = bpy.data.objects.get(overlay.PREVIEW_NAME)
    if preview is None or bpy.context.object is not source:
        raise AssertionError("live: preview missing or selection not restored")
    live.run_preview(bpy.context)
    if len([o for o in bpy.data.objects if o.name.startswith(overlay.PREVIEW_NAME)]) != 1:
        raise AssertionError("live: preview object was not reused")
    original_cap = live.LIVE_FACE_CAP
    live.LIVE_FACE_CAP = 10
    try:
        if live.run_preview(bpy.context) or "Live off" not in live.status():
            raise AssertionError("live: cap was not enforced")
    finally:
        live.LIVE_FACE_CAP = original_cap
    settings.live_enabled = False
    if bpy.data.objects.get(overlay.PREVIEW_NAME) is not None:
        raise AssertionError("live: preview was not removed")

    # Register / unregister / register leaves no stale handlers or timers.
    module.unregister()
    if overlay._draw_handles or overlay._on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        raise AssertionError("ui: unregister left handlers behind")
    module.register()
    if bpy.app.timers.is_registered(live._tick):
        raise AssertionError("ui: live timer survived re-register")
    print("PASS ui_features")


class _MockOp:
    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)


class _MockLayout:
    """Stand-in for UILayout that validates icons, property names and operator ids while a panel draws."""

    ICONS = None

    def __init__(self, log):
        self._log = log

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def _sub(self, *args, **kwargs):
        return _MockLayout(self._log)

    row = column = box = split = grid_flow = column_flow = _sub

    def _icon(self, icon):
        if icon not in self.ICONS:
            raise AssertionError(f"ui draw: unknown icon {icon!r}")

    def label(self, text="", icon="NONE", **kwargs):
        self._icon(icon)
        self._log.append(("label", text))

    def separator(self, *args, **kwargs):
        pass

    def prop(self, data, name, text=None, icon="NONE", **kwargs):
        self._icon(icon)
        if name not in data.bl_rna.properties:
            raise AssertionError(f"ui draw: {data.bl_rna.identifier} has no property {name!r}")
        self._log.append(("prop", name))

    def prop_search(self, data, name, search_data, search_prop, **kwargs):
        if name not in data.bl_rna.properties or search_prop not in search_data.bl_rna.properties:
            raise AssertionError(f"ui draw: bad prop_search {name!r}")

    def operator(self, idname, text="", icon="NONE", **kwargs):
        self._icon(icon)
        group, _, op = idname.partition(".")
        try:
            rna = getattr(getattr(bpy.ops, group), op).get_rna_type()
        except Exception as exc:
            raise AssertionError(f"ui draw: unknown operator {idname!r}") from exc
        self._log.append(("operator", idname))
        mock = _MockOp()
        mock.bl_rna = rna
        return mock

    def template_list(self, listtype, list_id, data, propname, active_data, active_propname, **kwargs):
        if propname not in data.bl_rna.properties or active_propname not in active_data.bl_rna.properties:
            raise AssertionError(f"ui draw: bad template_list {propname!r}")


def _draw_all_panels(label):
    """Draw every CurioMesh panel (and its header) against the mock layout; return how many drew."""
    from curiomesh import ui

    _MockLayout.ICONS = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())
    drawn = 0
    for cls in ui.panel_classes():
        poll = getattr(cls, "poll", None)
        if poll is not None and not poll(bpy.context):
            continue
        log: list = []
        holder = type("Holder", (), {"layout": _MockLayout(log)})()
        for hook in ("draw_header", "draw_header_preset", "draw"):
            fn = getattr(cls, hook, None)
            if fn is None or (hook != "draw" and fn is getattr(bpy.types.Panel, hook, None)):
                continue
            try:
                fn(holder, bpy.context)
            except Exception as exc:
                raise AssertionError(f"ui draw [{label}] {cls.__name__}.{hook}: {exc!r}") from exc
        drawn += 1
    return drawn


def run_ui_prefs_case(module):
    """Preferences, tab rename, HUD placement, heatmap colours and every panel drawing in several states."""
    from types import SimpleNamespace

    from curiomesh import overlay, region, ui

    reset_scene()
    settings = bpy.context.scene.curiomesh_settings
    region_settings = bpy.context.scene.curiomesh_region

    # A real preferences entry so the AddonPreferences class and its update callbacks run.
    addons = bpy.context.preferences.addons
    entry = addons.get("curiomesh") or addons.new()
    entry.module = "curiomesh"
    prefs = overlay.get_prefs()
    if not isinstance(prefs, bpy.types.AddonPreferences):
        raise AssertionError("ui prefs: get_prefs did not return the add-on preferences")
    for name, default in overlay.PREF_DEFAULTS.items():
        value = getattr(prefs, name)
        value = tuple(round(x, 4) for x in value) if not isinstance(value, (str, bool, int, float)) else value
        want = tuple(round(x, 4) for x in default) if isinstance(default, tuple) else default
        if value != want and not (isinstance(want, float) and abs(value - want) < 1e-6):
            raise AssertionError(f"ui prefs: default of {name} is {value!r}, expected {want!r}")

    # Tab rename re-registers every panel (Quick Start, its folds and Regions).
    prefs.panel_tab = "Retopo"
    for cls in ui.panel_classes():
        if cls.bl_category != "Retopo" or getattr(bpy.types, cls.__name__).bl_category != "Retopo":
            raise AssertionError(f"ui prefs: {cls.__name__} did not move to the Retopo tab")
    prefs.panel_tab = "   "
    if bpy.types.CURIOMESH_PT_panel.bl_category != "CurioMesh":
        raise AssertionError("ui prefs: blank tab name should fall back to CurioMesh")
    prefs.panel_tab = "CurioMesh"

    # HUD corner math.
    w, h = 330.0, 168.0
    cases = {
        "BOTTOM_LEFT": (70.0, 40.0),
        "BOTTOM_RIGHT": (1600.0 - 70.0 - w, 40.0),
        "TOP_LEFT": (70.0, 900.0 - 40.0 - h),
        "TOP_RIGHT": (1600.0 - 70.0 - w, 900.0 - 40.0 - h),
    }
    for corner, want in cases.items():
        got = overlay.hud_origin(corner, 1600.0, 900.0, w, h, 70.0, 40.0)
        if got != want:
            raise AssertionError(f"ui prefs: hud corner {corner} -> {got}, expected {want}")
    if overlay.hud_origin("BOTTOM_RIGHT", 1600.0, 900.0, w, h, 70.0, 40.0, right_inset=300.0)[0] != 1600.0 - 300.0 - 70.0 - w:
        raise AssertionError("ui prefs: sidebar inset ignored")
    if overlay.hud_origin("TOP_RIGHT", 200.0, 100.0, w, h, 70.0, 40.0) != (0.0, 0.0):
        raise AssertionError("ui prefs: HUD origin must stay inside a tiny viewport")
    if overlay.quality_word(None) != "NO DATA" or overlay.quality_word(1.0) != "ALL QUADS" or overlay.quality_word(0.5) != "NEEDS WORK":
        raise AssertionError("ui prefs: quality words")

    # Heatmap colours come from the preferences and trigger a rebuild when they change.
    cube = make_cube("CurioSmoke_PrefCube")
    reset_selection(cube)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")
    overlay.invalidate()
    stats = overlay.get_stats(cube)
    prefs.heat_tri_color = (0.0, 1.0, 0.0, 0.5)
    prefs.heat_pole5_color = (0.0, 0.0, 1.0, 1.0)
    colors = overlay.heat_colors()
    if colors[0] != (0.0, 1.0, 0.0, 0.5) or colors[3] != (0.0, 0.0, 1.0, 1.0):
        raise AssertionError(f"ui prefs: heat_colors ignored the preferences: {colors}")
    _tri_pos, tri_col, _pt_pos, pt_col = overlay.build_heat_arrays(cube, stats)
    if tuple(tri_col[0].tolist()) != (0.0, 1.0, 0.0, 0.5):
        raise AssertionError(f"ui prefs: heatmap triangle colour not applied: {tri_col[0]}")
    if not any(tuple(c.tolist()) == (0.0, 0.0, 1.0, 1.0) for c in pt_col):
        raise AssertionError("ui prefs: heatmap pole colour not applied")
    custom = overlay.build_heat_arrays(cube, stats, colors=((1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)))
    if tuple(custom[1][0].tolist()) != (1.0, 1.0, 0.0, 1.0):
        raise AssertionError("ui prefs: explicit heatmap colours ignored")
    # Overlay never raises when a preferences object is incomplete.
    overlay._prefs_override = SimpleNamespace(**{**overlay.PREF_DEFAULTS, "heat_tri_color": "bad"})
    try:
        if overlay.heat_colors()[0] != overlay.TRI_COLOR:
            raise AssertionError("ui prefs: a malformed colour should fall back to the default")
    finally:
        overlay._prefs_override = None
    if any(abs(a - b) > 1e-5 for a, b in zip(region.SEAM_COLOR, overlay.pref_color(overlay.get_prefs(), "region_seam_color"))):
        raise AssertionError("ui prefs: region seam colour default mismatch")

    # HUD default applies once per scene and only ever turns the HUD off.
    scene = bpy.context.scene
    if "curiomesh_hud_init" in scene.keys():
        del scene["curiomesh_hud_init"]
    settings.hud_enabled = True
    prefs.hud_default = False
    overlay.apply_hud_default(scene)
    if settings.hud_enabled:
        raise AssertionError("ui prefs: hud_default False did not switch the HUD off")
    settings.hud_enabled = True
    overlay.apply_hud_default(scene)
    if not settings.hud_enabled:
        raise AssertionError("ui prefs: hud default applied twice")
    prefs.hud_default = True

    # Every panel draws, in several states, with no unknown icon / property / operator.
    states = []
    for compact in (False, True):
        prefs.compact_mode = compact
        # nothing selected
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = None
        states.append(f"none/{compact}")
        _draw_all_panels(states[-1])
        reset_scene()
        sphere = make_uv_sphere("CurioSmoke_DrawSphere", segments=12, rings=6)
        reset_selection(sphere)
        for engine in ("ATLAS_Q", "TRIAD_Q_LITE", "QUADRIFLOW"):
            settings.engine = engine
            settings.heatmap_enabled = True
            settings.live_enabled = False
            settings.metrics_status = "OK"
            settings.metrics_message = "A fairly long message that has to wrap over several lines of the panel"
            settings.analysis_faces = 100
            settings.analysis_warning = "Open boundary loops | Thin sheets detected"
            settings.batch_status = "Batch done"
            for source in ("SELECTION", "FACE_SET", "VERTEX_GROUP", "VOLUMES"):
                region_settings.source = source
                states.append(f"{engine}/{source}/{compact}")
                if _draw_all_panels(states[-1]) < 8:
                    raise AssertionError(f"ui draw: too few panels drew in {states[-1]}")
        settings.metrics_status = "FAILED"
        _draw_all_panels(f"failed/{compact}")
        settings.metrics_status = ""
        settings.heatmap_enabled = False
        bpy.ops.object.mode_set(mode="EDIT")
        region_settings.source = "SELECTION"
        _draw_all_panels(f"edit/{compact}")
        bpy.ops.object.mode_set(mode="OBJECT")
        region_settings.volumes.add()
        region_settings.source = "VOLUMES"
        _draw_all_panels(f"cages/{compact}")
        region_settings.volumes.clear()
    # A region preview object (previewing state) and a live preview object as the active object.
    source = bpy.data.objects["CurioSmoke_DrawSphere"]
    source.select_set(True)
    ghost = bpy.data.objects.new(region.PREVIEW_NAME, bpy.data.meshes.new("CurioSmoke_GhostMesh"))
    bpy.context.scene.collection.objects.link(ghost)
    region.state["message"] = "Select faces first"
    _draw_all_panels("previewing")
    region.state["message"] = ""
    bpy.data.objects.remove(ghost, do_unlink=True)
    prefs.compact_mode = False
    settings.engine = "ATLAS_Q"

    # Hiding a panel in the preferences removes it from the sidebar.
    prefs.show_debug = False
    if bpy.types.CURIOMESH_PT_debug.poll(bpy.context):
        raise AssertionError("ui prefs: show_debug False still polls true")
    prefs.show_debug = True
    prefs.show_regions = False
    if bpy.types.CURIOMESH_PT_regions.poll(bpy.context):
        raise AssertionError("ui prefs: show_regions False still polls true")
    prefs.show_regions = True

    # Reset restores every default.
    prefs.hud_scale = 1.8
    prefs.panel_tab = "Temp"
    bpy.ops.curiomesh.reset_preferences()
    if prefs.hud_scale != 1.0 or bpy.types.CURIOMESH_PT_panel.bl_category != "CurioMesh":
        raise AssertionError("ui prefs: reset_preferences did not restore defaults")
    print("PASS ui_prefs")


def make_tri_suzanne(name: str):
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.ops.mesh.primitive_monkey_add(size=2.0)
    obj = bpy.context.object
    obj.name = name
    add_material_and_uv(obj, color=(0.8, 0.5, 0.3, 1.0))
    for kind in ("SUBSURF", "TRIANGULATE"):
        modifier = obj.modifiers.new(f"CurioSmoke {kind.lower()}", kind)
        if kind == "SUBSURF":
            modifier.levels = 1
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    return obj


def _face_keys(obj) -> list:
    """Per-face key (rounded corner coordinates), in face order."""

    mesh = obj.data
    return [
        frozenset(tuple(round(c, 5) for c in mesh.vertices[v].co) for v in poly.vertices)
        for poly in mesh.polygons
    ]


def _edge_face_counts(obj):
    import numpy as np

    mesh = obj.data
    loop_edges = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("edge_index", loop_edges)
    return np.bincount(loop_edges, minlength=len(mesh.edges))


def _check_region(label, obj, before_keys, region_faces, before_counts):
    """Untouched faces unchanged, no new open edges (no T-junctions), manifold, all quads inside.

    The applied region may differ from the requested one by the seam smoothing
    and parity fix, so "untouched" means every face the patch did not remove.
    """

    from curiomesh import region

    removed = region.state["last_removed"]
    if not removed or len(removed) > 1.5 * len(region_faces) + 60:
        raise AssertionError(f"{label}: removed {len(removed)} faces for a {len(region_faces)}-face request")
    gone = set(removed)
    untouched = {key for index, key in enumerate(before_keys) if index not in gone}
    after = set(_face_keys(obj))
    missing = untouched - after
    if missing:
        raise AssertionError(f"{label}: {len(missing)} untouched faces changed")
    new_keys = after - untouched
    sizes = [len(k) for k in new_keys]
    if not sizes or any(size != 4 for size in sizes):
        raise AssertionError(f"{label}: region is not all quads ({sorted(set(sizes))})")
    counts = _edge_face_counts(obj)
    if int((counts == 1).sum()) != int((before_counts == 1).sum()):
        raise AssertionError(f"{label}: open edges {int((before_counts == 1).sum())} -> {int((counts == 1).sum())}")
    if int((counts > 2).sum()) or int((counts == 0).sum()):
        raise AssertionError(f"{label}: non-manifold or loose edges after patch")
    print(f"PASS region {label}: {len(new_keys)} quads replace {len(gone)} faces")


def _select_faces(obj, ids):
    chosen = set(ids)
    for v in obj.data.vertices:
        v.select = False
    for poly in obj.data.polygons:
        poly.select = poly.index in chosen
        if poly.select:
            for v in poly.vertices:
                obj.data.vertices[v].select = True
    for edge in obj.data.edges:
        edge.select = all(obj.data.vertices[v].select for v in edge.vertices)


def _region_faces(obj, mask_fn):
    return [p.index for p in obj.data.polygons if mask_fn(p.center)]


def run_guided_case(module):
    import numpy as np
    from curiomesh import guided

    reset_scene()
    gs = bpy.context.scene.curiomesh_guided
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    sphere = make_uv_sphere("CurioSmoke_Guided", segments=40, rings=20)
    reset_selection(sphere)
    gs.base_faces = 400
    gs.fit_iterations = 1
    # two UV islands (x < 0 and x > 0, the second moved by 10) and two materials
    sphere.data.materials.clear()
    while len(sphere.data.uv_layers):
        sphere.data.uv_layers.remove(sphere.data.uv_layers[0])
    for name in ("CurioSmoke_GuidedA", "CurioSmoke_GuidedB"):
        sphere.data.materials.append(bpy.data.materials.new(name))
    layer = sphere.data.uv_layers.new(name="SmokeUV")
    for poly in sphere.data.polygons:
        poly.material_index = int(poly.center.x > 0)
        for li in poly.loop_indices:
            v = sphere.data.vertices[sphere.data.loops[li].vertex_index].co
            layer.data[li].uv = (v.x + (10.0 if poly.center.x > 0 else 0.0), v.z)

    def mean_edges(res, centre, radius):
        co, faces = res["co"], res["faces"]
        inside, outside = [], []
        for f in faces:
            for k in range(len(f)):
                a, b = co[f[k]], co[f[(k + 1) % len(f)]]
                (inside if np.linalg.norm((a + b) / 2 - centre) < radius else outside).append(np.linalg.norm(a - b))
        return float(np.mean(inside)), float(np.mean(outside))

    # points: operator, empty, auto-suggest
    bpy.ops.curiomesh.guided_add_point(kind="DETAIL", radius=0.45, location=(0.0, 0.0, 1.0))
    bpy.ops.curiomesh.guided_add_point(kind="CALM", radius=0.4, location=(0.0, 0.0, -1.0))
    if len(sphere.curiomesh_points) != 2 or sphere.curiomesh_points[0].kind != "DETAIL":
        raise AssertionError("guided: points were not stored on the object")
    bpy.context.scene.cursor.location = (1.0, 0.0, 0.0)
    bpy.ops.curiomesh.guided_add_empty()
    if len(guided.gather(sphere, gs)) != 3:
        raise AssertionError("guided: tagged empty was not picked up as a point")
    bpy.data.objects.remove(bpy.data.objects["CM_Point"], do_unlink=True)

    # whole mesh
    gs.scope = "WHOLE"
    if not guided.run_update(bpy.context, force=True):
        raise AssertionError(f"guided: whole-mesh update failed: {guided.status()}")
    res = guided.state["res"]
    sizes = [len(f) for f in res["faces"]]
    if bpy.data.objects.get(guided.PREVIEW_NAME) is None or sizes.count(4) != len(sizes):
        raise AssertionError(f"guided: whole-mesh preview missing or not all quads ({sizes.count(4)}/{len(sizes)})")
    preview = bpy.data.objects[guided.PREVIEW_NAME]
    uv_out = preview.data.uv_layers.get("SmokeUV")
    if uv_out is None or len(preview.data.materials) != 2:
        raise AssertionError("guided: whole-mesh preview lost its UV layer or material slots")
    for poly in preview.data.polygons:
        if (poly.material_index == 1) != (poly.center.x > 0) and abs(poly.center.x) > 0.15:
            raise AssertionError(f"guided: wrong material near x={poly.center.x:.2f}")
        us = [uv_out.data[li].uv.x for li in poly.loop_indices]
        if max(us) - min(us) > 1.5:
            raise AssertionError("guided: a face stretches across two UV islands")
        for li in poly.loop_indices:
            co_, uv_ = preview.data.vertices[preview.data.loops[li].vertex_index].co, uv_out.data[li].uv
            if abs(co_.x) < 0.4:  # a quad across the UV seam clamps its far corner to its own island
                continue
            if abs(uv_.y - co_.z) > 0.1 or abs((uv_.x - 10.0 if uv_.x > 5.0 else uv_.x) - co_.x) > 0.1:
                raise AssertionError(f"guided: UV {tuple(uv_)} does not match the source at {tuple(co_)}")
    inside, outside = mean_edges(res, np.array([0.0, 0.0, 1.0]), 0.4)
    if inside > 0.8 * outside:
        raise AssertionError(f"guided: Detail point did not make smaller quads ({inside:.3f} vs {outside:.3f})")
    print(f"PASS guided whole: {len(sizes)} faces, edge inside {inside:.3f} vs outside {outside:.3f}")

    # points only: the seam stays, the mesh stays closed
    gs.scope = "POINTS"
    if not guided.run_update(bpy.context, force=True):
        raise AssertionError(f"guided: points-only update failed: {guided.status()}")
    res = guided.state["res"]
    edge_count = {}
    for f in res["faces"]:
        for k in range(len(f)):
            key = tuple(sorted((f[k], f[(k + 1) % len(f)])))
            edge_count[key] = edge_count.get(key, 0) + 1
    if any(c != 2 for c in edge_count.values()):
        raise AssertionError("guided: points-only result is not a closed manifold (seam broken)")
    if res["new"].sum() == 0 or res["new"].all():
        raise AssertionError("guided: points-only should change part of the mesh")
    print(f"PASS guided points-only: {int(res['new'].sum())} new of {len(res['faces'])} faces, seam closed")

    # points only around a limb: the region is a ring with two seam loops, and all of it is rebuilt
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, depth=4.0, radius=0.5)
    limb = bpy.context.object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=3)
    bpy.ops.object.mode_set(mode="OBJECT")
    guided.add_point(limb, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), "DETAIL", 1.15, 1.0)
    gs.base_faces = 300
    if not guided.run_update(bpy.context, force=True):
        raise AssertionError(f"guided: ring region update failed: {guided.status()}")
    res = guided.state["res"]
    if "skipped" in guided.status() or not res["new"].any() or res["removed"] is None:
        raise AssertionError(f"guided: ring region was skipped: {guided.status()}")
    wrap = np.arctan2(*(res["co"][np.concatenate([list(f) for f, n in zip(res["faces"], res["new"]) if n])][:, :2].T))
    if wrap.max() - wrap.min() < 5.5:
        raise AssertionError("guided: only part of the ring around the limb was rebuilt")
    edge_count = {}
    for f in res["faces"]:
        for k in range(len(f)):
            key = tuple(sorted((f[k], f[(k + 1) % len(f)])))
            edge_count[key] = edge_count.get(key, 0) + 1
    if any(c != 2 for c in edge_count.values()):
        raise AssertionError("guided: ring region result is not closed")
    print(f"PASS guided ring: {int(res['new'].sum())} new faces all the way around the limb")
    bpy.data.objects.remove(limb, do_unlink=True)
    guided.remove_preview()
    gs.base_faces = 400
    reset_selection(sphere)
    if not guided.run_update(bpy.context, force=True):
        raise AssertionError(f"guided: points-only update failed after the ring case: {guided.status()}")

    # lock: select some preview faces, update keeps them
    def _same_faces(keys, res):
        now = np.asarray([res["co"][list(f)].mean(axis=0) for f, lk in zip(res["faces"], res["locked"]) if lk])
        return len(now) == len(keys) and all(np.linalg.norm(now - k, axis=1).min() < 1e-4 for k in keys)

    preview = bpy.data.objects[guided.PREVIEW_NAME]
    cent = np.asarray([p.center for p in preview.data.polygons])
    pick = np.flatnonzero(cent[:, 0] > 0.5)
    keys = cent[pick].copy()
    for p in preview.data.polygons:
        p.select = p.index in set(pick.tolist())
    if "FINISHED" not in bpy.ops.curiomesh.guided_lock():
        raise AssertionError(f"guided: lock failed: {guided.status()}")
    res = guided.state["res"]
    if not _same_faces(keys, res):
        raise AssertionError("guided: locked faces changed in the update")
    sphere.curiomesh_points[0].strength = 2.5
    if not guided.run_update(bpy.context, force=True):
        raise AssertionError(f"guided: update with locks failed: {guided.status()}")
    res = guided.state["res"]
    if not _same_faces(keys, res):
        raise AssertionError("guided: locked faces moved when a point changed")
    print(f"PASS guided lock: {len(keys)} faces frozen through an update")
    bpy.ops.curiomesh.guided_unlock()

    # auto-suggest, next spot, apply
    sphere.curiomesh_points.clear()
    bpy.ops.curiomesh.guided_suggest()
    if not len(sphere.curiomesh_points):
        raise AssertionError("guided: auto-suggest placed nothing")
    gs.scope = "WHOLE"
    bpy.ops.curiomesh.guided_update()
    before = len(bpy.data.objects)
    if "FINISHED" not in bpy.ops.curiomesh.guided_apply():
        raise AssertionError("guided: apply failed")
    out = bpy.data.objects.get("CurioSmoke_Guided_Retopo")
    if out is None or len(bpy.data.objects) != before or bpy.data.objects.get(guided.PREVIEW_NAME) is not None:
        raise AssertionError("guided: apply did not leave exactly the _Retopo object")
    print(f"PASS guided apply: {len(out.data.polygons)} faces in {out.name}")


def run_region_case(module):
    import numpy as np
    from curiomesh import region

    reset_scene()
    rs = bpy.context.scene.curiomesh_region
    rs.strategy = "FLOW"
    rs.fit_iterations = 2

    # 1. Edit-mode selection on a subdivided, triangulated Suzanne, via preview -> cancel -> preview -> apply.
    suzanne = make_tri_suzanne("CurioSmoke_RegionSel")
    reset_selection(suzanne)
    region_ids = _region_faces(suzanne, lambda c: c.z > 0.35 and c.y < -0.2)
    _select_faces(suzanne, region_ids)
    keys, counts = _face_keys(suzanne), _edge_face_counts(suzanne)
    bpy.ops.object.mode_set(mode="EDIT")
    rs.source = "SELECTION"
    if "FINISHED" not in bpy.ops.curiomesh.region_preview():
        raise AssertionError(f"region: preview failed: {region.status()}")
    preview = bpy.data.objects.get(region.PREVIEW_NAME)
    if preview is None or suzanne.visible_get() or region.state["draw"] is None:
        raise AssertionError("region: preview object, hidden source or seam overlay missing")
    bpy.ops.curiomesh.region_cancel()
    if bpy.data.objects.get(region.PREVIEW_NAME) is not None or not suzanne.visible_get():
        raise AssertionError("region: cancel left the preview or kept the source hidden")
    if set(_face_keys(suzanne)) != set(keys):
        raise AssertionError("region: cancel changed the source")
    reset_selection(suzanne)
    bpy.ops.curiomesh.region_preview()
    if "FINISHED" not in bpy.ops.curiomesh.region_apply():
        raise AssertionError(f"region: apply failed: {region.status()}")
    if bpy.data.objects.get(region.PREVIEW_NAME) is not None:
        raise AssertionError("region: apply left the preview behind")
    _check_region("selection", suzanne, keys, region_ids, counts)
    for strategy in ("GRID",):
        rs.strategy = strategy
        grid_source = make_tri_suzanne(f"CurioSmoke_Region{strategy}")
        reset_selection(grid_source)
        ids = _region_faces(grid_source, lambda c: c.z < -0.3 and abs(c.x) < 0.5)
        _select_faces(grid_source, ids)
        keys, counts = _face_keys(grid_source), _edge_face_counts(grid_source)
        if "FINISHED" not in bpy.ops.curiomesh.region_apply():
            raise AssertionError(f"region: {strategy} apply failed: {region.status()}")
        _check_region(strategy.lower(), grid_source, keys, ids, counts)
    rs.strategy = "FLOW"

    # 2. Box cage (into a copy) and 3. a deformed lattice cage.
    rs.apply_mode = "COPY"
    source = make_tri_suzanne("CurioSmoke_RegionBox")
    reset_selection(source)
    bpy.context.scene.cursor.location = (0.0, -0.6, 0.6)
    bpy.ops.curiomesh.region_add_cage(kind="BOX")
    cage = rs.volumes[-1].obj
    if cage is None or cage.type != "EMPTY" or rs.source != "VOLUMES":
        raise AssertionError("region: box cage was not added")
    mask, _message = region.build_mask(bpy.context, source, rs)
    ids = np.flatnonzero(mask).tolist()
    keys, counts = _face_keys(source), _edge_face_counts(source)
    reset_selection(source)
    if "FINISHED" not in bpy.ops.curiomesh.region_apply():
        raise AssertionError(f"region: box apply failed: {region.status()}")
    copy = bpy.data.objects.get("CurioSmoke_RegionBox_Region")
    if copy is None or _face_keys(source) != keys:
        raise AssertionError("region: copy mode changed the source or made no copy")
    _check_region("box_cage_copy", copy, keys, ids, counts)
    rs.apply_mode = "SOURCE"
    source.hide_set(False)
    rs.volumes.clear()

    source = make_tri_suzanne("CurioSmoke_RegionLattice")
    reset_selection(source)
    bpy.context.scene.cursor.location = (0.0, -0.5, -0.4)
    bpy.ops.curiomesh.region_add_cage(kind="LATTICE")
    lattice = rs.volumes[-1].obj
    lattice.data.points[0].co_deform.x -= 0.3  # bend one corner: the hull is no longer a box
    mask, _message = region.build_mask(bpy.context, source, rs)
    ids = np.flatnonzero(mask).tolist()
    if not ids:
        raise AssertionError("region: lattice cage selected nothing")
    keys, counts = _face_keys(source), _edge_face_counts(source)
    reset_selection(source)
    if "FINISHED" not in bpy.ops.curiomesh.region_apply():
        raise AssertionError(f"region: lattice apply failed: {region.status()}")
    _check_region("lattice_cage", source, keys, ids, counts)
    rs.volumes.clear()

    # 4. Two volumes: Union a big box, Subtract a box over one side (leaves an L-shaped disk).
    source = make_tri_suzanne("CurioSmoke_RegionBool")
    reset_selection(source)
    bpy.context.scene.cursor.location = (0.0, -0.6, 0.5)
    bpy.ops.curiomesh.region_add_cage(kind="BOX")
    big = rs.volumes[-1].obj
    big.scale = (0.45, 0.6, 0.35)
    bpy.context.scene.cursor.location = (0.35, -0.6, 0.75)
    bpy.ops.curiomesh.region_add_cage(kind="BOX")
    small = rs.volumes[-1].obj
    small.scale = (0.2, 0.6, 0.2)
    rs.volumes[-1].operation = "SUBTRACT"
    bpy.context.view_layer.update()
    mask, _message = region.build_mask(bpy.context, source, rs)
    points = np.asarray([(source.matrix_world @ p.center)[:] for p in source.data.polygons])
    union = region.volume_mask(big, points, bpy.context.evaluated_depsgraph_get())
    cut = region.volume_mask(small, points, bpy.context.evaluated_depsgraph_get())
    if not np.array_equal(mask, union & ~cut) or not cut[union].any():
        raise AssertionError(
            f"region: Union/Subtract mask wrong: mask {int(mask.sum())} union {int(union.sum())} "
            f"cut {int(cut.sum())} overlap {int((union & cut).sum())} diff {int((mask != (union & ~cut)).sum())}"
        )
    ids = np.flatnonzero(mask).tolist()
    keys, counts = _face_keys(source), _edge_face_counts(source)
    reset_selection(source)
    if "FINISHED" not in bpy.ops.curiomesh.region_apply():
        raise AssertionError(f"region: boolean apply failed: {region.status()}")
    _check_region("union_subtract", source, keys, ids, counts)
    rs.volumes.clear()

    # 5. Live preview follows a moved cage.
    source = make_tri_suzanne("CurioSmoke_RegionLive")
    reset_selection(source)
    bpy.context.scene.cursor.location = (0.0, -0.6, 0.6)
    bpy.ops.curiomesh.region_add_cage(kind="BOX")
    cage = rs.volumes[-1].obj
    reset_selection(source)
    if "FINISHED" not in bpy.ops.curiomesh.region_preview():
        raise AssertionError(f"region: live preview failed: {region.status()}")
    first = len(bpy.data.objects[region.PREVIEW_NAME].data.polygons)
    cage.scale = (cage.scale[0] * 1.6, cage.scale[1], cage.scale[2] * 1.6)
    bpy.context.view_layer.update()
    old = region.DEBOUNCE_SECONDS
    region.DEBOUNCE_SECONDS = 0.0
    try:
        region._poll()
        region._poll()
    finally:
        region.DEBOUNCE_SECONDS = old
    if len(bpy.data.objects[region.PREVIEW_NAME].data.polygons) == first:
        raise AssertionError("region: live preview did not follow the cage")
    print(f"PASS region live_preview: {first} -> {len(bpy.data.objects[region.PREVIEW_NAME].data.polygons)} faces")
    bpy.ops.curiomesh.region_cancel()
    rs.volumes.clear()

    # 6. Fill Loop on a hole, then 7. Fit to Surface.
    reset_scene()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=13, y_subdivisions=13, size=2.0)
    plane = bpy.context.object
    plane.name = "CurioSmoke_FillLoop"
    for v in plane.data.vertices:
        v.co.z = 0.3 * (1.0 - v.co.x * v.co.x) * (1.0 - v.co.y * v.co.y)
    open_before = int((_edge_face_counts(plane) == 1).sum())
    hole = set(_region_faces(plane, lambda c: abs(c.x) < 0.45 and abs(c.y) < 0.3))
    bm = bmesh.new()
    bm.from_mesh(plane.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index in hole], context="FACES_ONLY")
    bm.to_mesh(plane.data)
    bm.free()
    counts = _edge_face_counts(plane)
    verts = plane.data.vertices
    for poly in plane.data.polygons:
        poly.select = False
    for v in verts:
        v.select = False
    for edge in plane.data.edges:
        inner = all(abs(verts[v].co.x) < 0.99 and abs(verts[v].co.y) < 0.99 for v in edge.vertices)
        edge.select = bool(counts[edge.index] == 1 and inner)
        if edge.select:
            for v in edge.vertices:
                verts[v].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    if "FINISHED" not in bpy.ops.curiomesh.fill_loop():
        raise AssertionError("region: fill loop failed")
    bpy.ops.object.mode_set(mode="OBJECT")
    counts = _edge_face_counts(plane)
    sizes = {len(p.vertices) for p in plane.data.polygons}
    if int((counts == 1).sum()) != open_before or int((counts > 2).sum()) or sizes != {4}:
        raise AssertionError(f"region: fill loop left holes/non-manifold/non-quads {sizes}")
    print(f"PASS region fill_loop: {len(plane.data.polygons)} faces, open edges {open_before}")

    sphere = make_uv_sphere("CurioSmoke_FitTarget", segments=24, rings=12)
    reset_selection(sphere)
    bpy.ops.object.duplicate()
    noisy = bpy.context.object
    rng = np.random.default_rng(3)
    for v in noisy.data.vertices:
        v.co += __import__("mathutils").Vector(rng.normal(0.0, 0.03, 3).tolist())
    before = np.asarray([abs(v.co.length - 1.0) for v in noisy.data.vertices])
    bpy.ops.object.select_all(action="DESELECT")
    sphere.select_set(True)
    noisy.select_set(True)
    bpy.context.view_layer.objects.active = noisy
    if "FINISHED" not in bpy.ops.curiomesh.fit_to_surface(iterations=3):
        raise AssertionError("region: fit to surface failed")
    after = np.asarray([abs(v.co.length - 1.0) for v in noisy.data.vertices])
    if not after.max() < 0.02 < before.max():
        raise AssertionError(f"region: fit did not reach the surface ({before.max():.3f} -> {after.max():.3f})")
    print(f"PASS region fit_to_surface: max off-surface {before.max():.3f} -> {after.max():.4f}")

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    print("PASS region")


def main():
    module = load_addon()
    reset_scene()
    run_transformed_surface_error_case()
    run_bounded_fidelity_reference_case()
    run_large_retry_proxy_order_case()
    run_pairclosure_skips_flow_proxy_case()
    run_pairclosure_unlock_topology_gate_case()
    run_projection_rollback_case()
    run_quality_gate_case()
    run_atlas_pragmatic_gates_case()
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
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
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

    run_region_case(module)
    run_guided_case(module)
    run_ui_features_case(module)
    run_ui_prefs_case(module)
    module.unregister()
    print("CurioMesh Blender smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
