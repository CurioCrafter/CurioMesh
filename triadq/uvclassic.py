# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Classic Remesh UV handling (package A1c): the per-side texture transfer.

Our own design and code. `preserve_textures_steps` replaces the island-blind DataTransfer with K1's side-aware transfer
(`uvseams`, SPEC 4.8), in short steps. No `bpy` import: only the mesh objects handed in are touched.
"""

from __future__ import annotations

import numpy as np

from . import uvseams


# ---------------------------------------------------------------- texture transfer (bridge._preserve_textures_steps)
def read_uv(mesh, name):
    flat = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    mesh.uv_layers.get(name).data.foreach_get("uv", flat)
    return flat.reshape(-1, 2).astype(np.float64)


def write_uv(mesh, uv, name):
    layer = mesh.uv_layers.get(name) or mesh.uv_layers.new(name=name)
    layer.data.foreach_set("uv", np.ascontiguousarray(uv, dtype=np.float32).reshape(-1))


def _polys(obj):
    """(points in world space, indptr, flat vertex ids, material index per face) of an object's mesh."""

    mesh = obj.data
    points = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", points)
    points = points.reshape(-1, 3)
    matrix = getattr(obj, "matrix_world", None)
    if matrix is not None:
        m = np.array([[matrix[r][c] for c in range(4)] for r in range(4)], float)
        points = points @ m[:3, :3].T + m[:3, 3]
    total = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_total", total)
    flat = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", flat)
    material = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("material_index", material)
    indptr = np.concatenate(([0], np.cumsum(total))).astype(np.int64)
    return points, indptr, flat.astype(np.int64), material


# The fold fix of SPEC 4.8 step 4 (harmonic fill of folded faces and their ring) is off in the Classic path: measured on the
# nine bench models it moves the UVs of faces that are only straddling a seam, which raised texture drift p99 from
# 0.8-2 px to 7-44 px (barrier, lion, bust, hydrant) while it removed at most half of a few dozen flipped faces. Drift is
# the Classic gate; Spot-fix style repair belongs to the tools that have a preview.
FIX_FOLDS = False


def preserve_textures_steps(source, target, config, stopped=lambda: False):
    """Per-side transfer of every UV layer, as a generator for a timer: the transfer yields between short steps. Returns
    "TRANSFERRED", or None to let the old DataTransfer/BVH code run (not a transfer mode, no source UVs, the transfer gave
    nothing usable, or `stopped()` turned True)."""

    if str(config.texture_mode or "PROJECT").upper() not in {"TRANSFER", "PROJECT"}:
        return None
    layers = [layer.name for layer in source.data.uv_layers]
    if not layers:
        return None
    try:
        sp, si, sf, material = _polys(source)
        rp, ri, rf, _ = _polys(target)
        active = source.data.uv_layers.active.name if source.data.uv_layers.active else layers[0]
        layers.sort(key=lambda n: n != active)  # the active layer first: it makes the lookup the others reuse
        first = read_uv(source.data, layers[0])
        stage = uvseams.Transfer(sp, si, sf, first.ravel(), rp, ri, rf, material=material, fix_folds=FIX_FOLDS)
        while not stage.done:
            if stopped():
                stage.stop()
                return None
            stage.step()
            yield
        out = stage.result
        if not out.get("done", True):
            return None
        results = {layers[0]: out["uv"]}
        for name in layers[1:]:
            got: dict = {}
            yield from uvseams.layer_steps(si, sf, read_uv(source.data, name).ravel(), rp, ri, rf, out, fix_folds=FIX_FOLDS, got=got)
            results[name] = got["uv"]
        for name, uv in results.items():
            uv = np.asarray(uv, float).reshape(-1, 2)
            bad = ~np.isfinite(uv).all(axis=1)
            if bad.mean() > 0.5:
                return None
            uv[bad] = 0.0  # corners the lookup could not reach; the legacy check below still judges the layer
            write_uv(target.data, uv, name)
        target.data.uv_layers.active = target.data.uv_layers.get(active)
    except Exception:
        return None
    return "TRANSFERRED"
