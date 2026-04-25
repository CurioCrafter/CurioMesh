from __future__ import annotations

from collections.abc import Iterable

import bpy


def ensure_target_uvs(obj: bpy.types.Object, method: str = "SMART") -> None:
    if obj.type != "MESH" or obj.data.uv_layers:
        return

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


def assign_image_to_material(obj: bpy.types.Object, image: bpy.types.Image, label: str) -> None:
    mesh = obj.data
    if not mesh.materials:
        mesh.materials.append(bpy.data.materials.new(name=f"{obj.name}_Material"))

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


def bake_selected_to_active(
    source_obj: bpy.types.Object,
    target_obj: bpy.types.Object,
    *,
    maps: Iterable[str],
    resolution: int = 2048,
    margin: int = 8,
    uv_method: str = "SMART",
) -> list[bpy.types.Image]:
    ensure_target_uvs(target_obj, uv_method)
    baked: list[bpy.types.Image] = []

    previous_active = bpy.context.view_layer.objects.active
    previous_selection = list(bpy.context.selected_objects)
    scene = bpy.context.scene
    previous_engine = scene.render.engine
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
        scene.cycles.samples = max(16, int(scene.cycles.samples))
        scene.render.bake.use_selected_to_active = True
        scene.render.bake.cage_extrusion = 0.04
        scene.render.bake.margin = max(0, int(margin))

        for map_name in maps:
            pass_name = str(map_name).upper()
            if pass_name == "COLOR":
                bake_type = "DIFFUSE"
                label = "CurioMesh_BaseColor"
                scene.render.bake.use_pass_direct = False
                scene.render.bake.use_pass_indirect = False
                scene.render.bake.use_pass_color = True
            elif pass_name in {"ROUGHNESS", "METALLIC", "NORMAL", "EMIT"}:
                bake_type = pass_name
                label = f"CurioMesh_{pass_name.title()}"
            else:
                continue

            image = create_bake_image(f"{target_obj.name}_{label}", int(resolution))
            assign_image_to_material(target_obj, image, label)
            bpy.ops.object.bake(type=bake_type)
            try:
                image.pack()
            except Exception:
                pass
            baked.append(image)
    finally:
        scene.render.engine = previous_engine
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
