# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Read and write a mesh's UV layers as plain arrays: one (corner count, 2) array per layer, corners in mesh order.

Only the mesh object is touched (no `bpy` import), so tests can pass a stand-in. Applying a whole candidate (vertex
moves, face splits) is added here by package A1.
"""

from __future__ import annotations

import numpy as np


def read_uv(mesh, name: str | None = None) -> np.ndarray | None:
    """UV of every corner as float64 (corners, 2), from layer `name` or the active one; None when there is no such layer."""

    layer = mesh.uv_layers.get(name) if name else mesh.uv_layers.active
    if layer is None:
        return None
    flat = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    layer.data.foreach_get("uv", flat)
    return flat.reshape(-1, 2).astype(np.float64)


def write_uv(mesh, uv, name: str | None = None, create: bool = True) -> str | None:
    """Store `uv` (corners, 2) in layer `name` (default: the active one). A missing layer is made when `create`.

    Returns the layer's name, or None when there was no layer and none was made. The shape must match the corners.
    """

    values = np.ascontiguousarray(uv, dtype=np.float32)
    if values.shape != (len(mesh.loops), 2):
        raise ValueError(f"uv has shape {values.shape}, the mesh has {len(mesh.loops)} corners")
    layer = mesh.uv_layers.get(name) if name else mesh.uv_layers.active
    if layer is None:
        if not create:
            return None
        layer = mesh.uv_layers.new(name=name or "UVMap")
    layer.data.foreach_set("uv", values.reshape(-1))
    return layer.name


def geometry(mesh):
    """(points (n, 3), CSR indptr, flat, UV of the active layer (corners, 2), material per face) of a mesh, as arrays."""

    n = len(mesh.vertices)
    points = np.empty(n * 3, dtype=np.float64)
    if n:
        mesh.vertices.foreach_get("co", points)
    sizes = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_total", sizes)
    flat = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", flat)
    material = np.zeros(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("material_index", material)
    uv = read_uv(mesh)
    if uv is None:
        uv = np.zeros((len(mesh.loops), 2))
    return points.reshape(-1, 3), np.r_[0, np.cumsum(sizes)].astype(np.int64), flat, uv, material


