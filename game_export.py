"""Export for Game Engine: the selected meshes and their baked textures, set up the way each engine reads them.

Tested in Unreal 5.7 and Godot 4.7 by lighting a baked result from eight sides next to its source. Blender, Godot and
Unity read normal maps green-up (OpenGL); Unreal reads them green-down (DirectX), so its copy gets the green channel
flipped. Your images in Blender are never changed. A baked image lives only inside the .blend until saved, so FBX
exports would reach the engine without textures: the maps are written as PNG files next to the mesh.
"""

from __future__ import annotations

import os

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper

ENGINES = (
    ("UNREAL", "Unreal", "FBX and PNG maps. The normal map's green channel is flipped, as Unreal reads it (DirectX)"),
    ("UNITY", "Unity", "FBX and PNG maps, normal map as baked (OpenGL), scaled and turned for Unity"),
    ("GODOT", "Godot", "One .glb with the textures inside, normal map as baked (OpenGL)"),
)


def is_normal_map(material, image) -> bool:
    """The image feeds a Normal Map node in this material."""
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image == image:
            return any(link.to_node.type == "NORMAL_MAP" for out in node.outputs for link in out.links)
    return False


def flip_green(pixels: np.ndarray) -> np.ndarray:
    """RGBA rows, values 0..1: the same normals with the tangent's Y the other way (OpenGL <-> DirectX)."""
    out = pixels.copy()
    out[..., 1] = 1.0 - out[..., 1]
    return out


def write_png(image, path: str, flip: bool) -> None:
    """Write `image` to `path` without touching it: a copy holds the pixels, so its file path and data stay as they are."""
    w, h = image.size
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    if flip:
        px = flip_green(px)
    copy = bpy.data.images.new("CurioMesh_Export", w, h, alpha=True, float_buffer=image.is_float)
    try:
        copy.colorspace_settings.name = image.colorspace_settings.name
        copy.pixels.foreach_set(px.ravel())
        copy.filepath_raw, copy.file_format = path, "PNG"
        copy.save()
    finally:
        bpy.data.images.remove(copy)


def _maps(objects):
    """(image, is normal map) for every image the objects' materials use, once each."""
    seen = {}
    for obj in objects:
        for material in obj.data.materials:
            if material is None or material.node_tree is None:
                continue
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None and node.image.size[0] > 0:
                    seen.setdefault(node.image.name, (node.image, is_normal_map(material, node.image)))
    return list(seen.values())


class CURIOMESH_OT_export_game(bpy.types.Operator, ExportHelper):
    bl_idname = "curiomesh.export_game"
    bl_label = "Export for Game Engine"
    bl_description = "Write the selected meshes and their baked textures for Unreal, Unity or Godot, with the normal map in the form that engine reads"
    bl_options = {"REGISTER"}

    filename_ext = ".fbx"
    filter_glob: bpy.props.StringProperty(default="*.fbx;*.glb", options={"HIDDEN"})
    engine: bpy.props.EnumProperty(name="Engine", items=ENGINES, default="UNREAL")

    @classmethod
    def poll(cls, context):
        return any(o.type == "MESH" for o in context.selected_objects)

    def execute(self, context):
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        objects = [o for o in context.selected_objects if o.type == "MESH"]
        stem = os.path.splitext(self.filepath)[0]
        if self.engine == "GODOT":
            path = stem + ".glb"
            bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_format="GLB", export_tangents=True, export_apply=True)
            self.report({"INFO"}, f"Wrote {os.path.basename(path)} with its textures inside")
            return {"FINISHED"}
        path = stem + ".fbx"
        turn = {"axis_forward": "-Z", "axis_up": "Y", "bake_space_transform": True} if self.engine == "UNITY" else {}
        bpy.ops.export_scene.fbx(
            filepath=path, use_selection=True, use_mesh_modifiers=True, mesh_smooth_type="FACE", use_tspace=True,
            apply_scale_options="FBX_SCALE_ALL", **turn,
        )
        written = []
        for image, normal in _maps(objects):
            name = f"{os.path.basename(stem)}_{bpy.path.clean_name(image.name)}.png"
            write_png(image, os.path.join(os.path.dirname(path), name), flip=normal and self.engine == "UNREAL")
            written.append(name)
        flipped = " The normal map is flipped for Unreal: import it as it is." if self.engine == "UNREAL" and any(n for _, n in _maps(objects)) else ""
        self.report({"INFO"}, f"Wrote {os.path.basename(path)} and {len(written)} textures next to it.{flipped}")
        return {"FINISHED"}


def _menu(self, context):
    self.layout.operator(CURIOMESH_OT_export_game.bl_idname, text="CurioMesh Game Asset (.fbx/.glb)")


def register():
    bpy.utils.register_class(CURIOMESH_OT_export_game)
    bpy.types.TOPBAR_MT_file_export.append(_menu)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(_menu)
    bpy.utils.unregister_class(CURIOMESH_OT_export_game)
