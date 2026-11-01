# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Run by a separate background Blender: QuadriFlow on each piece of a job folder.

QuadriFlow can't be interrupted and blocks the Blender that runs it, so Clean Rows gives it another Blender, which keeps
the user's responsive and can be killed if a piece takes too long. Reads in_<k>.npz, writes out_<k>.npz. A piece that
QuadriFlow refuses gets an out file with no faces.
"""

import os
import sys

import bpy
import numpy as np

INPUT_PER_QUAD = 3  # QuadriFlow's input needs at most this many triangles per quad asked for


def _solve(points, polys, target):
    mesh = bpy.data.meshes.new("Rows")
    mesh.from_pydata(points.tolist(), [], polys)
    mesh.update()
    obj = bpy.data.objects.new("Rows", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        triangles = sum(len(p) - 2 for p in polys)
        if triangles > INPUT_PER_QUAD * target:
            obj.modifiers.new("Reduce", "DECIMATE").ratio = INPUT_PER_QUAD * target / triangles
            bpy.ops.object.modifier_apply(modifier="Reduce")
        try:
            done = bpy.ops.object.quadriflow_remesh(mode="FACES", target_faces=int(target), use_preserve_sharp=True, use_preserve_boundary=True, seed=0)
        except RuntimeError:
            return None
        if "FINISHED" not in done or not len(obj.data.polygons):
            return None
        co = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", co)
        return co.reshape(-1, 3), [tuple(p.vertices) for p in obj.data.polygons]
    finally:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(data)


def main():
    folder, first, last = sys.argv[sys.argv.index("--") + 1 :]
    for k in range(int(first), int(last)):
        path = os.path.join(folder, f"in_{k}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        flat, polys, at = d["flat"].tolist(), [], 0
        for size in d["lens"].tolist():
            polys.append(tuple(flat[at : at + size]))
            at += size
        result = _solve(d["co"], polys, int(d["target"]))
        temporary = os.path.join(folder, f"out_{k}.tmp.npz")
        if result is None:
            np.savez(temporary, co=np.zeros((0, 3)), lens=np.zeros(0, dtype=np.int64), flat=np.zeros(0, dtype=np.int64))
        else:
            np.savez(temporary, co=result[0], lens=np.asarray([len(f) for f in result[1]], dtype=np.int64), flat=np.asarray([v for f in result[1] for v in f], dtype=np.int64))
        os.replace(temporary, os.path.join(folder, f"out_{k}.npz"))


main()
