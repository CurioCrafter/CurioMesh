# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

from collections.abc import Iterable

import bmesh
import bpy
import numpy as np

from . import sliced
from .triadq import tasks, texsets
from .triadq import uvbake
from .triadq.uvs import is_see_through, uv_overlap

BAKE_LAYER = "CurioMesh_Bake"
MIN_UV_FILL = 0.4  # share of the UV square a new unwrap must use before its island gaps are accepted
MOST_OVERLAP = 0.01  # share of texels a UV map may reuse and still hold a bake


def ensure_target_uvs_steps(obj: bpy.types.Object, method: str = "SMART", new_layer: str = "", island_margin: float = 0.02):
    """Unwrap a mesh with no UVs. With new_layer, always unwrap, into a new layer that becomes the active one. Timer steps
    (`sliced`): a long unwrap yields between short pieces; Stop falls back to Smart Project."""
    yield from ()
    if (yield from uvbake.ensure_target_uvs_steps(obj, method, new_layer, island_margin, sliced.stopped, lambda f: sliced.tick("Making the bake UVs", f))):
        return  # CurioMesh's own unwrap made the layer (or there was nothing to do)
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
    except (RuntimeError, ReferenceError):
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
            bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=island_margin)
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except (RuntimeError, ReferenceError):
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


def _uv_fill(mesh: bpy.types.Mesh) -> float:
    """The share of the UV square the active layer's triangles cover. Padding between many small islands eats the rest."""
    mesh.calc_loop_triangles()
    loops = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("loops", loops)
    uv = np.empty(len(mesh.loops) * 2, dtype=np.float64)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    a, b, c = uv.reshape(-1, 2)[loops].reshape(-1, 3, 2).transpose(1, 0, 2)
    return float(np.abs(np.cross(b - a, c - a)).sum() / 2)


def _harden_uv_seams(mesh: bpy.types.Mesh) -> int:
    """Make every edge where the active UV map splits a hard edge: the normal map can't hide a shading seam there."""
    count = len(mesh.loops)
    verts, edges = np.empty(count, dtype=np.int64), np.empty(count, dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", verts)
    mesh.loops.foreach_get("edge_index", edges)
    uv = np.empty(count * 2, dtype=np.float64)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    starts = np.empty(len(mesh.polygons), dtype=np.int64)
    sizes = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", sizes)
    ahead = np.arange(1, count + 1)
    ahead[starts + sizes - 1] = starts  # the corner after each corner, around its face
    seen: dict[int, int] = {}
    split = set()
    for corner, edge in enumerate(edges.tolist()):
        other = seen.setdefault(edge, corner)
        if other == corner:
            continue
        # The two faces run along the edge in opposite directions, so each corner's vertex is the other's next one.
        near = uv[corner] if verts[corner] == verts[ahead[other]] else uv[ahead[corner]]
        far = uv[ahead[corner]] if verts[corner] == verts[ahead[other]] else uv[corner]
        if np.abs(near - uv[ahead[other]]).max() > 1e-5 or np.abs(far - uv[other]).max() > 1e-5:
            split.add(edge)
    for edge in split:
        mesh.edges[edge].use_edge_sharp = True
    return len(split)


def _keep_triangles(obj: bpy.types.Object) -> None:
    """Fix the triangles now: the bake and the exporters read them through this, so a game can't pick other diagonals."""
    modifier = obj.modifiers.new("CurioMesh Bake Triangles", "TRIANGULATE")
    modifier.quad_method = "SHORTEST_DIAGONAL"
    modifier.min_vertices = 4


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


def _keep_see_through(mesh: bpy.types.Mesh, old: np.ndarray, glass: dict, uv_layer: str) -> None:
    """Point the faces (`old` slot each) at the baked material, slot 0, but those of `glass` (old slot: material) at their own.

    Their textures, if any, need their old UVs, and the bake's layer is the one that stays.
    """
    remap = np.zeros(max(int(old.max(initial=0)), *glass, 0) + 1, dtype=np.int32)
    for slot, (index, material) in enumerate(glass.items(), start=1):
        mesh.materials.append(material)
        remap[index] = slot
    mesh.polygons.foreach_set("material_index", remap[old])
    kept = next((layer for layer in mesh.uv_layers if layer.name != uv_layer), None)
    if not glass or kept is None:
        return
    starts, sizes = np.empty(len(old), dtype=np.int64), np.empty(len(old), dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", sizes)
    corners = np.zeros(len(mesh.loops), dtype=bool)
    order = np.argsort(starts)
    corners[np.flatnonzero(np.repeat(remap[old][order] > 0, sizes[order]))] = True
    theirs, ours = np.empty(len(mesh.loops) * 2), np.empty(len(mesh.loops) * 2)
    kept.data.foreach_get("uv", theirs)
    mesh.uv_layers[uv_layer].data.foreach_get("uv", ours)
    both = np.repeat(corners, 2)
    ours[both] = theirs[both]
    mesh.uv_layers[uv_layer].data.foreach_set("uv", ours)


def bake_detail(source_obj: bpy.types.Object, target_obj: bpy.types.Object, **kw) -> list[bpy.types.Image]:
    return sliced.finish(bake_detail_steps(source_obj, target_obj, **kw))


def bake_detail_steps(
    source_obj: bpy.types.Object,
    target_obj: bpy.types.Object,
    *,
    with_color: bool,
    resolution: int,
    cage: float,
    new_uvs: bool = False,
):
    """`bake_detail` as timer steps (`sliced`): the unwrap runs in short pieces, the bake itself is Blender's.

    Bake the source's detail onto the target. The source and its materials are only read.

    With color, the target ends up with one new material holding both maps, and see-through materials stay as they are:
    a bake has no way to hold glass. Without color, it keeps its own materials, as private copies, with the normal map
    added to each. A new UV layer is made when the kept one overlaps itself.
    """
    mesh = target_obj.data
    glass = {i: m for i, m in enumerate(mesh.materials) if with_color and is_see_through(m)}
    margin = uvbake.gap_px(resolution)
    if new_uvs or not _uv_layer_can_hold_bake(mesh):
        while True:
            yield from ensure_target_uvs_steps(target_obj, new_layer=BAKE_LAYER, island_margin=2.0 * margin / resolution)
            if margin <= 2 or _uv_fill(mesh) >= MIN_UV_FILL:
                break
            # Thousands of small islands: the gaps between them would eat the texture, and every face gets a few texels.
            mesh.uv_layers.remove(mesh.uv_layers.active)  # the one just made: on a second bake it is named BAKE_LAYER.001
            margin //= 2
    uv_layer = mesh.uv_layers.active.name
    keep_render = next((layer.name for layer in mesh.uv_layers if layer.active_render), uv_layer)
    mesh.uv_layers.active.active_render = True
    _harden_uv_seams(mesh)
    _keep_triangles(target_obj)
    images = yield from bake_selected_to_active_steps(
        source_obj, target_obj, maps=("NORMAL", "COLOR") if with_color else ("NORMAL",),
        resolution=resolution, margin=margin, cage=cage, ray_limit=3.0 * cage,
    )
    uvbake.after_bake(target_obj, images)
    normal = images[0]
    if with_color:
        private = [m for m in mesh.materials if m is not None]
        slots = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("material_index", slots)
        mesh.materials.clear()  # this sets every face's slot to 0
        mesh.materials.append(_baked_material(f"{target_obj.name}_Baked", images[1], normal, uv_layer))
        _keep_see_through(mesh, slots, glass, uv_layer)
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


def bake_texture_sets_steps(
    source_obj: bpy.types.Object,
    target_obj: bpy.types.Object,
    *,
    count: int,
    labels,
    resolution: int,
    cage: float,
):
    """Timer steps (`sliced`): each set's unwrap runs in short pieces, as in `bake_detail_steps`.

    Bake Normal + Color onto `count` sets of maps, each set its own material and its own 0-1 UV square.

    Whole parts (`labels`, per face, or None) go together where they fit, in sets of about equal area. Each set is cut
    off into its own object, baked, and the sets are joined back into the target. Faces on either side of a set's border
    no longer share corners, as on any UV seam.
    """
    mesh = target_obj.data
    n = len(mesh.polygons)
    area, centres = np.empty(n), np.empty(n * 3)
    mesh.polygons.foreach_get("area", area)
    mesh.polygons.foreach_get("center", centres)
    sets = texsets.texture_sets(np.zeros(n, np.int32) if labels is None else labels, area, centres.reshape(-1, 3), count)
    if sets.max() == 0:
        return (yield from bake_detail_steps(source_obj, target_obj, with_color=True, resolution=resolution, cage=cage, new_uvs=True))
    images, parts = [], []
    try:
        for k in range(int(sets.max()) + 1):
            part = target_obj.copy()
            part.data = mesh.copy()
            part.name = part.data.name = f"{target_obj.name}_Set{k + 1}"
            for collection in target_obj.users_collection:
                collection.objects.link(part)
            parts.append(part)
            bm = bmesh.new()
            bm.from_mesh(part.data)
            bm.faces.ensure_lookup_table()
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if sets[f.index] != k], context="FACES")
            bm.to_mesh(part.data)
            bm.free()
            images += yield from bake_detail_steps(source_obj, part, with_color=True, resolution=resolution, cage=cage, new_uvs=True)
        for part in parts:  # the joined mesh gets one, with the same triangles
            for modifier in [m for m in part.modifiers if m.name == "CurioMesh Bake Triangles"]:
                part.modifiers.remove(modifier)
        # Into the target's own mesh, emptied first: whoever holds that mesh keeps holding the result.
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.delete(bm, geom=list(bm.verts), context="VERTS")
        bm.to_mesh(mesh)
        bm.free()
        mesh.materials.clear()
        while mesh.uv_layers:
            mesh.uv_layers.remove(mesh.uv_layers[0])
        with bpy.context.temp_override(active_object=target_obj, selected_editable_objects=[target_obj, *parts], object=target_obj):
            bpy.ops.object.join()
        parts = []  # joined away
        if mesh.materials and mesh.materials[0] is None:  # the emptied target's own slot: no face uses it
            slots = np.empty(len(mesh.polygons), np.int32)
            mesh.polygons.foreach_get("material_index", slots)
            mesh.materials.pop(index=0)
            mesh.polygons.foreach_set("material_index", np.maximum(slots - 1, 0))
        _keep_triangles(target_obj)
        return images
    except BaseException:  # also a closed generator: the set copies never stay behind
        for part in parts:
            if part.name in bpy.data.objects:
                data = part.data
                bpy.data.objects.remove(part)
                if data.users == 0:
                    bpy.data.meshes.remove(data)
        for image in images:
            bpy.data.images.remove(image)
        raise


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
    with tasks.busy_cursor():  # Blender's own bake: one call, Blender is busy until it is done
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


def bake_selected_to_active_steps(
    source_obj: bpy.types.Object,
    target_obj: bpy.types.Object,
    *,
    maps: Iterable[str],
    resolution: int = 2048,
    margin: int = 8,
    uv_method: str = "CURIO",
    cage: float = 0.04,
    ray_limit: float = 0.0,
):
    """Bake `maps` from the source onto the target, as timer steps (`sliced`): the unwrap of a mesh without UVs runs in short pieces
    (Stop: Smart Project makes them); the bake itself is Blender's, in one piece."""
    jobs = _bake_jobs(maps)
    if not jobs:
        return []

    yield from ensure_target_uvs_steps(target_obj, uv_method)
    sliced.tick("Baking")
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
    except (RuntimeError, ReferenceError):
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
            except (RuntimeError, ReferenceError):
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
