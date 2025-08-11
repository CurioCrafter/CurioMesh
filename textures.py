import bpy
from typing import Iterable, Set


def ensure_uvs(obj: bpy.types.Object, method: str = "SMART") -> None:
    if obj.type != 'MESH':
        return
    mesh = obj.data
    if not mesh.uv_layers:
        # Enter edit mode ops context
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            if method == 'UNWRAP':
                bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
            else:
                bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')


def create_bake_image(name: str, res: int) -> bpy.types.Image:
    img = bpy.data.images.new(name=name, width=res, height=res, alpha=True, float_buffer=False)
    return img


def assign_image_to_material(target_obj: bpy.types.Object, image: bpy.types.Image, color_attr: str) -> None:
    me = target_obj.data
    if not me.materials:
        me.materials.append(bpy.data.materials.new(name=f"{target_obj.name}_Mat"))
    for mat in me.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        # Find or create an Image Texture node named per color_attr
        tex = next((n for n in nodes if n.type == 'TEX_IMAGE' and (n.label == color_attr or n.name == color_attr)), None)
        if tex is None:
            tex = nodes.new('ShaderNodeTexImage')
            tex.label = color_attr
            tex.name = color_attr
        tex.image = image
        # Ensure it's the active image for baking
        nodes.active = tex


def bake_maps(source_obj: bpy.types.Object,
              target_obj: bpy.types.Object,
              maps: Set[str],
              res: int = 2048,
              margin: int = 8,
              uv_method: str = 'SMART') -> None:
    # Ensure consistent transforms by temporarily duplicating source and aligning transforms
    # Apply scale on both to minimize projection errors
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = target_obj
    try:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass
    try:
        bpy.context.view_layer.objects.active = source_obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass

    # Ensure both have UVs
    ensure_uvs(target_obj, method=uv_method)

    # Select source then target; active = target
    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    # Bake settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.bake_type = 'DIFFUSE'
    scene.cycles.samples = max(1, scene.cycles.samples)
    scene.render.bake.margin = margin
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = 0.05

    def do_bake(pass_name: str, bake_type: str, use_direct: bool = True, use_indirect: bool = True, use_color: bool = True):
        img = create_bake_image(f"{target_obj.name}_{pass_name}", res)
        assign_image_to_material(target_obj, img, pass_name)
        if bake_type == 'DIFFUSE':
            scene.cycles.bake_type = 'DIFFUSE'
            scene.render.bake.use_pass_direct = use_direct
            scene.render.bake.use_pass_indirect = use_indirect
            scene.render.bake.use_pass_color = use_color
        else:
            scene.cycles.bake_type = bake_type
        bpy.ops.object.bake(type=bake_type)

    if 'COLOR' in maps:
        do_bake('BaseColor', 'DIFFUSE', use_direct=False, use_indirect=False, use_color=True)
    if 'ROUGHNESS' in maps:
        do_bake('Roughness', 'ROUGHNESS')
    if 'METALLIC' in maps:
        do_bake('Metallic', 'METALLIC')
    if 'NORMAL' in maps:
        do_bake('Normal', 'NORMAL')
    if 'EMIT' in maps:
        do_bake('Emission', 'EMIT')


