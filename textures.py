# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

from collections.abc import Iterable

import bpy
import numpy as np

from .triadq.uvs import uv_overlap

BAKE_LAYER = "CurioMesh_Bake"
MOST_OVERLAP = 0.01  # share of texels a UV map may reuse and still hold a bake


def ensure_target_uvs(obj: bpy.types.Object, method: str = "SMART", new_layer: str = "") -> None:
    """Unwrap a mesh with no UVs. With new_layer, always unwrap, into a new layer that becomes the active one."""
    if obj.type != "MESH" or (obj.data.uv_layers and not new_layer):
        return
    if new_layer:
        layer = obj.data.uv_layers.new(name=new_layer)
        obj.data.uv_layers.active = layer
        layer.active_render = True

    previous_active = bpy.context.view_layer.objects.active
    previous_selection = list(bpy.context.selected_objects)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    try:
        for selected in bpy.context.selected_objects:
            selected.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if method == "UNWRAP":
            bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.002)
        else:
            bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.02)
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
        for selected in bpy.context.selected_objects:
            selected.select_set(False)
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = previous_active


def create_bake_image(name: str, resolution: int) -> bpy.types.Image:
    image = bpy.data.images.new(
        name=name,
        width=max(64, int(resolution)),
        height=max(64, int(resolution)),
        alpha=True,
        float_buffer=False,
    )
    image.generated_color = (0.0, 0.0, 0.0, 1.0)
    return image


def _uv_layer_can_hold_bake(mesh: bpy.types.Mesh) -> bool:
    """Every texel must belong to one place on the surface, or the bake paints over itself."""
    layer = mesh.uv_layers.active
    if layer is None:
        return False
    mesh.calc_loop_triangles()
    loops = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("loops", loops)
    uv = np.empty(len(mesh.loops) * 2, dtype=np.float64)
    layer.data.foreach_get("uv", uv)
    return uv_overlap(uv.reshape(-1, 2)[loops].reshape(-1, 3, 2)) <= MOST_OVERLAP


def _principled(material: bpy.types.Material):
    return next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)


def _connect_normal(material: bpy.types.Material, image: bpy.types.Image, uv_layer: str) -> None:
    """Send a tangent-space normal map into the material's shader, read through the given UV layer."""
    principled = _principled(material)
    if principled is None:
        return
    nodes, links = material.node_tree.nodes, material.node_tree.links
    tex = next(node for node in nodes if node.type == "TEX_IMAGE" and node.image == image)
    normal = nodes.new("ShaderNodeNormalMap")
    normal.uv_map = uv_layer
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_layer
    links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], principled.inputs["Normal"])


def _baked_material(name: str, color: bpy.types.Image, normal: bpy.types.Image, uv_layer: str) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = normal
    _connect_normal(material, normal, uv_layer)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = color
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_layer
    material.node_tree.links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
    material.node_tree.links.new(tex.outputs["Color"], _principled(material).inputs["Base Color"])
    return material


def bake_detail(
    source_obj: bpy.types.Object,
    target_obj: bpy.types.Object,
    *,
    with_color: bool,
    resolution: int,
    cage: float,
    margin: int = 8,
    new_uvs: bool = False,
) -> list[bpy.types.Image]:
    """Bake the source's detail onto the target. The source and its materials are only read.

    With color, the target ends up with one new material holding both maps. Without it, it keeps its own materials,
    as private copies, with the normal map added to each. A new UV layer is made when the kept one overlaps itself.
    """
    mesh = target_obj.data
    if new_uvs or not _uv_layer_can_hold_bake(mesh):
        ensure_target_uvs(target_obj, new_layer=BAKE_LAYER)
    uv_layer = mesh.uv_layers.active.name
    keep_render = next((layer.name for layer in mesh.uv_layers if layer.active_render), uv_layer)
    mesh.uv_layers.active.active_render = True
    images = bake_selected_to_active(
        source_obj, target_obj, maps=("NORMAL", "COLOR") if with_color else ("NORMAL",),
        resolution=resolution, margin=margin, cage=cage, ray_limit=3.0 * cage,
    )
    normal = images[0]
    if with_color:
        private = [m for m in mesh.materials if m is not None]
        mesh.materials.clear()
        mesh.materials.append(_baked_material(f"{target_obj.name}_Baked", images[1], normal, uv_layer))
        mesh.polygons.foreach_set("material_index", np.zeros(len(mesh.polygons), dtype=np.int32))
        mesh.update()
        for material in private:
            bpy.data.materials.remove(material)
        for layer in [layer for layer in mesh.uv_layers if layer.name != uv_layer]:  # game engines read the first UV map
            mesh.uv_layers.remove(layer)
    else:
        mesh.uv_layers[keep_render].active_render = True  # the kept materials read their own textures through this one
        for material in mesh.materials:
            if material is not None:
                _connect_normal(material, normal, uv_layer)
    return images


def assign_image_to_material(obj: bpy.types.Object, image: bpy.types.Image, label: str) -> None:
    mesh = obj.data
    if not mesh.materials:
        mesh.materials.append(bpy.data.materials.new(name=f"{obj.name}_Material"))

    # Baking must not mutate a material shared by the immutable source object.
    for index, material in enumerate(list(mesh.materials)):
        if material is not None and material.users > 1:
            mesh.materials[index] = material.copy()

    for material in mesh.materials:
        if material is None:
            continue
        material.use_nodes = True
        nodes = material.node_tree.nodes
        tex = next((node for node in nodes if node.type == "TEX_IMAGE" and node.label == label), None)
        if tex is None:
            tex = nodes.new("ShaderNodeTexImage")
            tex.label = label
            tex.name = label
        tex.image = image
        nodes.active = tex


def connect_baked_image(obj: bpy.types.Object, image: bpy.types.Image, label: str) -> None:
    if label != "CurioMesh_BaseColor":
        return
    for material in obj.data.materials:
        if material is None or not material.use_nodes:
            continue
        nodes = material.node_tree.nodes
        tex = next(
            (
                node
                for node in nodes
                if node.type == "TEX_IMAGE" and node.label == label and node.image == image
            ),
            None,
        )
        principled = nodes.get("Principled BSDF")
        if tex is None or principled is None or "Base Color" not in principled.inputs:
            continue
        color_input = principled.inputs["Base Color"]
        for link in list(color_input.links):
            material.node_tree.links.remove(link)
        material.node_tree.links.new(tex.outputs["Color"], color_input)


def _bake_jobs(maps: Iterable[str]) -> list[tuple[str, str, bool]]:
    jobs: list[tuple[str, str, bool]] = []
    for map_name in maps:
        pass_name = str(map_name).upper()
        if pass_name == "COLOR":
            jobs.append(("DIFFUSE", "CurioMesh_BaseColor", True))
        elif pass_name in {"ROUGHNESS", "METALLIC", "NORMAL", "EMIT"}:
            jobs.append((pass_name, f"CurioMesh_{pass_name.title()}", False))
    return jobs


def _execute_bake(bake_type: str) -> set[str]:
    return bpy.ops.object.bake(type=bake_type)


def _snapshot_bake_settings(scene: bpy.types.Scene) -> list[tuple[object, str, object]]:
    bake = scene.render.bake
    return [
        (scene.render, "engine", scene.render.engine),
        (scene.cycles, "samples", scene.cycles.samples),
        (bake, "use_selected_to_active", bake.use_selected_to_active),
        (bake, "cage_extrusion", bake.cage_extrusion),
        (bake, "max_ray_distance", bake.max_ray_distance),
        (bake, "margin", bake.margin),
        (bake, "use_pass_direct", bake.use_pass_direct),
        (bake, "use_pass_indirect", bake.use_pass_indirect),
        (bake, "use_pass_color", bake.use_pass_color),
    ]


def _restore_properties(snapshot: list[tuple[object, str, object]]) -> None:
    for owner, name, value in snapshot:
        setattr(owner, name, value)


def _make_target_materials_private(
    target_obj: bpy.types.Object,
    created_materials: list[bpy.types.Material],
) -> list[bpy.types.Material | None]:
    """Isolate all bake node edits so a failed bake can be rolled back exactly."""

    mesh = target_obj.data
    original_materials = list(mesh.materials)
    if not original_materials:
        material = bpy.data.materials.new(name=f"{target_obj.name}_Material")
        created_materials.append(material)
        mesh.materials.append(material)
        return original_materials

    for index, material in enumerate(original_materials):
        if material is None:
            private_material = bpy.data.materials.new(name=f"{target_obj.name}_Material_{index + 1}")
        else:
            private_material = material.copy()
        created_materials.append(private_material)
        mesh.materials[index] = private_material
    return original_materials


def _rollback_bake_data(
    target_obj: bpy.types.Object,
    original_materials: list[bpy.types.Material | None],
    created_materials: list[bpy.types.Material],
    created_images: list[bpy.types.Image],
) -> list[str]:
    errors: list[str] = []
    mesh = target_obj.data

    try:
        for index, material in enumerate(original_materials):
            mesh.materials[index] = material
        while len(mesh.materials) > len(original_materials):
            mesh.materials.pop(index=len(mesh.materials) - 1)
    except Exception as exc:
        errors.append(f"material slots: {exc}")

    for material in reversed(created_materials):
        try:
            if bpy.data.materials.get(material.name) is material:
                bpy.data.materials.remove(material, do_unlink=True)
        except Exception as exc:
            errors.append(f"material {getattr(material, 'name', '<removed>')}: {exc}")

    for image in reversed(created_images):
        try:
            if bpy.data.images.get(image.name) is image:
                bpy.data.images.remove(image, do_unlink=True)
        except Exception as exc:
            errors.append(f"image {getattr(image, 'name', '<removed>')}: {exc}")
    return errors


def bake_selected_to_active(
    source_obj: bpy.types.Object,
    target_obj: bpy.types.Object,
    *,
    maps: Iterable[str],
    resolution: int = 2048,
    margin: int = 8,
    uv_method: str = "SMART",
    cage: float = 0.04,
    ray_limit: float = 0.0,
) -> list[bpy.types.Image]:
    jobs = _bake_jobs(maps)
    if not jobs:
        return []

    ensure_target_uvs(target_obj, uv_method)
    baked: list[bpy.types.Image] = []
    created_images: list[bpy.types.Image] = []
    created_materials: list[bpy.types.Material] = []
    original_materials: list[bpy.types.Material | None] = list(target_obj.data.materials)

    previous_active = bpy.context.view_layer.objects.active
    previous_selection = list(bpy.context.selected_objects)
    scene = bpy.context.scene
    bake_settings = _snapshot_bake_settings(scene)
    visibility_state = {}

    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    try:
        for obj in (source_obj, target_obj):
            visibility_state[obj.name] = (
                obj.hide_select,
                obj.hide_viewport,
                obj.hide_render,
                obj.hide_get(),
            )
            obj.hide_select = False
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_set(False)

        for selected in bpy.context.selected_objects:
            selected.select_set(False)
        source_obj.select_set(True)
        target_obj.select_set(True)
        bpy.context.view_layer.objects.active = target_obj

        scene.render.engine = "CYCLES"
        # A selected-to-active diffuse color bake has no lighting pass and does
        # not benefit from inheriting a potentially huge production render
        # sample count. Keep the asset pipeline deterministic and bounded.
        scene.cycles.samples = 16
        scene.render.bake.use_selected_to_active = True
        scene.render.bake.cage_extrusion = cage
        scene.render.bake.max_ray_distance = ray_limit
        scene.render.bake.margin = max(0, int(margin))
        _make_target_materials_private(target_obj, created_materials)

        for bake_type, label, is_color in jobs:
            if is_color:
                scene.render.bake.use_pass_direct = False
                scene.render.bake.use_pass_indirect = False
                scene.render.bake.use_pass_color = True

            image = create_bake_image(f"{target_obj.name}_{label}", int(resolution))
            created_images.append(image)
            if bake_type == "NORMAL":  # must be set before baking, and is what game engines expect
                image.colorspace_settings.name = "Non-Color"
                image.generated_color = (0.5, 0.5, 1.0, 1.0)
            assign_image_to_material(target_obj, image, label)
            result = _execute_bake(bake_type)
            if "FINISHED" not in result:
                raise RuntimeError(f"{bake_type} bake did not finish: {sorted(result)}")
            connect_baked_image(target_obj, image, label)
            try:
                image.pack()
            except Exception:
                pass
            baked.append(image)
    except Exception as bake_error:
        cleanup_errors = _rollback_bake_data(
            target_obj,
            original_materials,
            created_materials,
            created_images,
        )
        if cleanup_errors:
            details = "; ".join(cleanup_errors)
            raise RuntimeError(f"Bake failed and rollback was incomplete ({details})") from bake_error
        raise
    finally:
        _restore_properties(bake_settings)
        for obj in (source_obj, target_obj):
            state = visibility_state.get(obj.name)
            if state is None or obj.name not in bpy.data.objects:
                continue
            obj.hide_select, obj.hide_viewport, obj.hide_render, hidden = state
            obj.hide_set(hidden)
        for selected in bpy.context.selected_objects:
            selected.select_set(False)
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = previous_active

    return baked
