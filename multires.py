# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Multires Detail: a clean, low cage keeps its faces at level 0, and the source's shape comes back as Multires levels.

Only Blender's own tools write the levels: Multires Subdivide, then Reshape from a copy moved onto the source. The source
is only read.
"""

from __future__ import annotations

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from .triadq import detail

MODIFIER = "Multires Detail"
HARD_DEG = 50.0


def _vertices(mesh) -> np.ndarray:
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _vertex_normals(mesh) -> np.ndarray:
    normals = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertex_normals.foreach_get("vector", normals)
    return normals.reshape(-1, 3)


def _quads(mesh) -> np.ndarray:
    """Every polygon of a level above 0 is a quad: Multires splits any face into them."""
    corners = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    return corners.reshape(-1, 4)


def level_mesh(context, obj, modifier, level: int):
    """A new mesh of this object at this level."""
    modifier.levels = level
    context.view_layer.update()
    return bpy.data.meshes.new_from_object(obj.evaluated_get(context.evaluated_depsgraph_get()))


def source_tree(source, matrix=None) -> BVHTree:
    """The source's own mesh without its modifiers, in its object space, or moved by `matrix` into another object's."""
    mesh = source.data
    points = _vertices(mesh)
    if matrix is not None:
        m = np.asarray(matrix, dtype=np.float64)
        points = points @ m[:3, :3].T + m[:3, 3]
    return BVHTree.FromPolygons(points.tolist(), [tuple(p.vertices) for p in mesh.polygons], all_triangles=False, epsilon=0.0)


def hard_edges(mesh, degrees: float = HARD_DEG) -> list[tuple[int, int]]:
    """The cage's hard edges, to crease: without them the levels round every corner before the detail comes back."""
    corners = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    sizes = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_total", sizes)
    return detail.hard_edges(_vertices(mesh), corners, sizes, degrees)


def _base_of_quads(mesh, level: int) -> np.ndarray:
    """Per quad of a level, the polygon of level 0 it came from. An n-gon makes n quads, then each makes four per level."""
    sizes = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_total", sizes)
    return np.repeat(np.arange(len(sizes)), sizes * 4 ** (level - 1))


def add_detail(context, obj, tree: BVHTree, levels: int, flat_faces=None, simple: bool = False, on_level=None) -> dict:
    """Give `obj` a Multires modifier with `levels` levels that follow the surface in `tree`, in `obj`'s object space.

    `flat_faces` masks the faces of level 0 that keep their own shape at every level. Returns a report: per level the
    faces, the vertices that found no surface and the quads the guards had to leave folded. `on_level(level, modifier)`
    is called after each level is finished.
    """
    own_tree = None
    if flat_faces is not None:
        own_tree = BVHTree.FromPolygons(_vertices(obj.data).tolist(), [tuple(p.vertices) for p in obj.data.polygons], all_triangles=False, epsilon=0.0)
    modifier = obj.modifiers.new(MODIFIER, "MULTIRES")
    copy = bpy.data.objects.new(obj.name + "_DetailStep", bpy.data.meshes.new(obj.name + "_DetailStep"))
    context.scene.collection.objects.link(copy)
    view = context.view_layer
    selected, active = list(context.selected_objects), view.objects.active
    report = []
    try:
        for other in selected:
            other.select_set(False)
        obj.select_set(True)
        copy.select_set(True)
        view.objects.active = obj
        for level in range(1, levels + 1):
            bpy.ops.object.multires_subdivide(modifier=modifier.name, mode="SIMPLE" if simple else "CATMULL_CLARK")
            modifier.sculpt_levels = level
            mesh = level_mesh(context, obj, modifier, level)
            points = _vertices(mesh)
            quads = _quads(mesh)
            edge = float(np.linalg.norm(points[quads[:, 0]] - points[quads[:, 1]], axis=1).mean())
            normals = _vertex_normals(mesh)
            reach = detail.REACH_EDGES * edge
            targets, found = detail.look(points, normals, reach, tree.ray_cast, tree.find_nearest)
            if own_tree is not None:
                keeps = np.ones(len(points), dtype=bool)
                keeps[quads[~flat_faces[_base_of_quads(obj.data, level)]]] = False  # a seam vertex follows the source
                own, own_found = detail.look(points[keeps], normals[keeps], reach, own_tree.ray_cast, own_tree.find_nearest)
                targets[keeps], found[keeps] = own, own_found
            moved, lost, folds = detail.settle(points, targets, found, quads)
            mesh.vertices.foreach_set("co", moved.ravel())
            mesh.update()
            old = copy.data
            copy.data = mesh
            bpy.data.meshes.remove(old)
            view.update()
            if "FINISHED" not in bpy.ops.object.multires_reshape(modifier=modifier.name):
                raise RuntimeError(f"Multires Reshape refused level {level}")
            report.append({"level": level, "faces": len(quads), "lost": lost, "folded": folds})
            if on_level is not None:
                on_level(level, modifier)
    finally:
        mesh = copy.data
        bpy.data.objects.remove(copy)
        bpy.data.meshes.remove(mesh)
        obj.select_set(False)
        for other in selected:
            if other.name in bpy.data.objects:
                other.select_set(True)
        view.objects.active = active
    modifier.levels = modifier.sculpt_levels = modifier.render_levels = levels
    # Anything else on the object (the bake's Triangulate) must not change the mesh before Multires reads it.
    obj.modifiers.move(obj.modifiers.find(modifier.name), 0)
    return {"levels": report}


def add_to_result(context, obj, tree: BVHTree, levels: int, flat_faces=None) -> str:
    """add_detail for a finished object. Failing leaves the quads as they were. Returns text for the report."""
    try:
        report = add_detail(context, obj, tree, levels, flat_faces)
    except (RuntimeError, ValueError) as exc:
        modifier = obj.modifiers.get(MODIFIER)
        if modifier is not None:
            obj.modifiers.remove(modifier)
        return f". Multires Detail failed: {exc}"
    return f", Multires Detail up to {report['levels'][-1]['faces']:,} faces at level {levels}"
