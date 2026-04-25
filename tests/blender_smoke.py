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


def make_open_grid(name: str, size=8):
    verts = []
    faces = []
    for y in range(size + 1):
        for x in range(size + 1):
            verts.append((x / size - 0.5, y / size - 0.5, 0.08 * ((x + y) % 2),))
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


def make_bad_normals_sphere(name: str):
    obj = make_uv_sphere(name, segments=28, rings=14)
    obj.data.flip_normals()
    return obj


def run_remesh_case(
    name: str,
    obj,
    kwargs,
    *,
    min_quad_ratio=0.55,
    require_uv=True,
    max_face_error=None,
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
    if len(output.data.materials) == 0:
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


def main():
    module = load_addon()
    reset_scene()

    sphere = make_uv_sphere("CurioSmoke_Sphere")
    out = run_remesh_case(
        "sphere_new",
        sphere,
        {
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
            "target_faces": 90,
            "quality": "DRAFT",
            "output_mode": "NEW",
            "texture_mode": "PROJECT",
            "preserve_boundary": True,
            "seed": 3,
        },
        min_quad_ratio=0.50,
    )

    bad_normals = make_bad_normals_sphere("CurioSmoke_BadNormals")
    run_remesh_case(
        "bad_normals_repair",
        bad_normals,
        {
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
    run_remesh_case(
        "bake_color",
        bake_source,
        {
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

    module.unregister()
    print("CurioMesh Blender smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
