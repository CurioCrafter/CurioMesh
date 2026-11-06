# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Export for Game Engine: the selected meshes and their baked textures, set up the way each engine reads them.

Unreal, Unity and Godot were tested here (Unreal 5.7, Unity 6, Godot 4.7) by lighting a baked result from eight sides
next to its source. The other presets follow their engines' own documentation and are not tested here. Normal maps come
in two kinds: green up (OpenGL: Blender, Unity, Godot, Roblox, glTF) and green down (DirectX: Unreal, CryEngine, O3DE).

Your objects and images are never changed. Textures are written as PNG files next to the mesh (or inside a .glb), from
copies: flipped for DirectX engines and shrunk to the engine's largest size. Meshes over an engine's triangle limit are
written as several pieces under it, cut along texture sets and parts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import bmesh
import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper

from .triadq import texsets


@dataclass(frozen=True)
class Engine:
    label: str
    words: str
    form: str  # FBX, GLB or OBJ
    directx: bool = False
    tris: int = 0  # most triangles per mesh, 0 for no limit
    texture: int = 0  # largest texture side, 0 for no limit
    unity_axes: bool = False
    scale: str = "FBX_SCALE_ALL"


ENGINES = {
    "UNREAL": Engine("Unreal", "FBX and PNG maps. Normal map green flipped, as Unreal reads it (DirectX). Tested in Unreal 5.7", "FBX", directx=True),
    "UNITY": Engine("Unity", "FBX turned and scaled for Unity, PNG maps, normal map as baked (OpenGL). Tested in Unity 6", "FBX", unity_axes=True),
    "GODOT": Engine("Godot", "One .glb with the textures inside, normal map as baked (OpenGL). Tested in Godot 4.7", "GLB"),
    "ROBLOX": Engine("Roblox", "FBX at FBX unit scale with tangents, PNG maps of at most 1024 pixels, normal map as baked (OpenGL). Meshes over 20,000 triangles are split. From Roblox's documentation, not tested here", "FBX", tris=20000, texture=1024, scale="FBX_SCALE_UNITS"),
    "GLTF": Engine("Web and glTF", "One .glb for three.js, Babylon.js, Bevy, Sketchfab and other glTF readers. glTF normal maps are green up (OpenGL), as baked", "GLB"),
    "DIRECTX": Engine("CryEngine, O3DE", "FBX and PNG maps with the normal map green flipped (DirectX). From these engines' documentation, not tested here", "FBX", directx=True),
    "CUSTOM": Engine("Custom", "Choose the file type, normal map kind, triangle limit and texture size yourself", "FBX"),
}

FORMS = (("FBX", "FBX", "FBX with PNG maps next to it"), ("GLB", "glTF (.glb)", "One file with the textures inside"), ("OBJ", "OBJ", "OBJ and MTL with PNG maps next to it"))
NORMALS = (("OPENGL", "OpenGL (green up)", "Blender, Unity, Godot, Roblox, glTF"), ("DIRECTX", "DirectX (green down)", "Unreal, CryEngine, O3DE"))
SIZES = (("0", "As baked", "Keep the baked size"), *((str(s), str(s), f"At most {s} pixels") for s in (512, 1024, 2048, 4096)))


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


def write_png(image, path: str, flip: bool, largest: int = 0) -> None:
    """Write `image` to `path` without touching it: a copy holds the pixels, flipped and shrunk if asked."""
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
        if largest and max(w, h) > largest:
            copy.scale(max(1, w * largest // max(w, h)), max(1, h * largest // max(w, h)))
        copy.filepath_raw, copy.file_format = path, "PNG"
        copy.save()
    finally:
        bpy.data.images.remove(copy)


def _nodes(objects):
    """(material, image node) for every image the objects' materials use."""
    for material in {m for o in objects for m in o.data.materials if m is not None and m.node_tree is not None}:
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None and node.image.size[0] > 0:
                yield material, node


def _pieces(context, objects, limit: int):
    """Objects to write: each one at most `limit` triangles, cut along materials (texture sets) and then by place.

    Returns (objects, made): `made` are new objects holding the pieces, to delete after writing.
    """
    out, made = [], []
    depsgraph = context.evaluated_depsgraph_get()
    for obj in objects:
        mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), depsgraph=depsgraph)  # with its modifiers: the triangles the engine gets
        n = len(mesh.polygons)
        sides, slots, centres = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n * 3)
        mesh.polygons.foreach_get("loop_total", sides)
        mesh.polygons.foreach_get("material_index", slots)
        mesh.polygons.foreach_get("center", centres)
        piece = texsets.split_under(slots, sides - 2, centres.reshape(-1, 3), limit)
        if piece.max() == 0:
            bpy.data.meshes.remove(mesh)
            out.append(obj)
            continue
        for k in range(int(piece.max()) + 1):
            part = mesh.copy()
            bm = bmesh.new()
            bm.from_mesh(part)
            bm.faces.ensure_lookup_table()
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if piece[f.index] != k], context="FACES")
            bm.to_mesh(part)
            bm.free()
            new = bpy.data.objects.new(f"{obj.name}_{k + 1:02d}", part)
            new.matrix_world = obj.matrix_world.copy()
            context.scene.collection.objects.link(new)
            made.append(new)
            out.append(new)
        bpy.data.meshes.remove(mesh)
    return out, made


class CURIOMESH_OT_export_game(bpy.types.Operator, ExportHelper):
    bl_idname = "curiomesh.export_game"
    bl_label = "Export for Game Engine"
    bl_description = "Write the selected meshes and their baked textures for a game engine, with the normal map, scale, texture size and triangle limit that engine needs"
    bl_options = {"REGISTER"}

    filename_ext = ".fbx"
    filter_glob: bpy.props.StringProperty(default="*.fbx;*.glb;*.obj", options={"HIDDEN"})
    engine: bpy.props.EnumProperty(name="Engine", description="Which game engine the files are for. It sets the file type, normal map kind, scale and size limits", items=[(k, e.label, e.words) for k, e in ENGINES.items()], default="UNREAL")
    form: bpy.props.EnumProperty(name="File", description="The file type to write. Only used by Custom", items=FORMS, default="FBX")
    normals: bpy.props.EnumProperty(name="Normal Map", description="Which way the normal map's green channel points: OpenGL (Blender, Unity, Godot) or DirectX (Unreal). Only used by Custom", items=NORMALS, default="OPENGL")
    tris: bpy.props.IntProperty(name="Most Triangles", description="Split meshes into pieces of at most this many triangles. 0 keeps them whole", default=0, min=0)
    texture: bpy.props.EnumProperty(name="Largest Texture", description="Shrinks textures so none is larger than this. Only used by Custom", items=SIZES, default="0")

    @classmethod
    def poll(cls, context):
        return any(o.type == "MESH" for o in context.selected_objects)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "engine")
        engine = ENGINES[self.engine]
        if self.engine == "CUSTOM":
            for name in ("form", "normals", "tris", "texture"):
                layout.prop(self, name)
        else:
            col = layout.column(align=True)
            col.scale_y = 0.8
            for line in engine.words.split(". "):
                col.label(text=line.rstrip("."))

    def _setup(self) -> Engine:
        engine = ENGINES[self.engine]
        if self.engine != "CUSTOM":
            return engine
        return Engine("Custom", "", self.form, directx=self.normals == "DIRECTX", tris=self.tris, texture=int(self.texture))

    def execute(self, context):
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        engine = self._setup()
        chosen = [o for o in context.selected_objects if o.type == "MESH"]
        active = context.view_layer.objects.active
        stem = os.path.splitext(self.filepath)[0]
        folder, base = os.path.dirname(stem), os.path.basename(stem)
        objects, made = _pieces(context, chosen, engine.tris) if engine.tris else (chosen, [])
        swapped, loaded, written = [], [], []
        try:
            # Write each map once, as the engine wants it, and point the materials at the written files while exporting.
            files = {}
            for material, node in list(_nodes(objects)):
                image = node.image
                if image.name not in files:
                    name = f"{base}_{bpy.path.clean_name(image.name)}.png"
                    path = os.path.join(folder, name)
                    write_png(image, path, flip=engine.directx and is_normal_map(material, image), largest=engine.texture)
                    new = bpy.data.images.load(path)
                    new.colorspace_settings.name = image.colorspace_settings.name
                    files[image.name] = new
                    loaded.append(new)
                    written.append(name)
                swapped.append((node, image))
                node.image = files[image.name]
            for o in context.selected_objects:
                o.select_set(False)
            for o in objects:
                o.select_set(True)
            context.view_layer.objects.active = objects[0]
            path = stem + "." + engine.form.lower()
            if engine.form == "GLB":
                bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_format="GLB", export_tangents=True, export_apply=True)
                for name in written:  # inside the .glb already
                    os.remove(os.path.join(folder, name))
                written = []
            elif engine.form == "OBJ":
                bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True, apply_modifiers=True, export_triangulated_mesh=True, path_mode="STRIP")
            else:
                turn = {"axis_forward": "-Z", "axis_up": "Y", "bake_space_transform": True} if engine.unity_axes else {}
                bpy.ops.export_scene.fbx(
                    filepath=path, use_selection=True, use_mesh_modifiers=True, mesh_smooth_type="FACE", use_tspace=True,
                    apply_scale_options=engine.scale, path_mode="STRIP", **turn,
                )
        finally:
            for node, image in swapped:
                node.image = image
            for image in loaded:
                bpy.data.images.remove(image)
            for o in made:
                data = o.data
                bpy.data.objects.remove(o)
                bpy.data.meshes.remove(data)
            for o in context.selected_objects:
                o.select_set(False)
            for o in chosen:
                o.select_set(True)
            context.view_layer.objects.active = active
        pieces = f" as {len(objects)} pieces under {engine.tris:,} triangles" if made else ""
        maps = f" and {len(written)} textures" if written else ""
        flipped = " (normal map flipped for DirectX)" if engine.directx and written else ""
        self.report({"INFO"}, f"Wrote {os.path.basename(path)}{pieces}{maps}{flipped}")
        return {"FINISHED"}


def _menu(self, context):
    self.layout.operator(CURIOMESH_OT_export_game.bl_idname, text="CurioMesh Game Asset (.fbx/.glb/.obj)")


def register():
    bpy.utils.register_class(CURIOMESH_OT_export_game)
    bpy.types.TOPBAR_MT_file_export.append(_menu)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(_menu)
    bpy.utils.unregister_class(CURIOMESH_OT_export_game)
