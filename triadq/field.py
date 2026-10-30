# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Curvature-aware 4-RoSy cross field via Ginzburg-Landau relaxation.

Own implementation from the papers: Viertel & Osting 2019 (complex
u = exp(4i theta) relaxed with a double-well penalty instead of hard
normalisation) and Rusinkiewicz 2004 (per-face shape operator from
vertex-normal differences). Drop-in for ``engine._estimate_field``: same
inputs, one unit direction per face.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .topology import _turbo


def _unit_rows(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = length > 1e-12
    return np.where(safe, vectors / np.where(safe, length, 1.0), fallback)


def _frames(mesh: Any, adj: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-face tangent frame (e1 along side 0, e2 = n x e1), sides and side angles, as in the engine."""

    normals = adj.normals
    corners = mesh.vertices[adj.faces]
    sides = np.roll(corners, -1, axis=1) - corners
    x_axis = np.broadcast_to(np.array([1.0, 0.0, 0.0]), normals.shape)
    e1 = sides[:, 0] - normals * np.einsum("ij,ij->i", sides[:, 0], normals)[:, None]
    e1 = _unit_rows(e1, x_axis)
    e2 = np.cross(normals, e1)
    side_angle = np.arctan2(np.einsum("fcj,fj->fc", sides, e2), np.einsum("fcj,fj->fc", sides, e1))
    return e1, e2, sides, side_angle


def _curvature_cross(
    mesh: Any, adj: Any, e1: np.ndarray, e2: np.ndarray, sides: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face principal direction as exp(4i theta) and anisotropy |k1-k2| / (|k1|+|k2|+eps)."""

    faces = adj.faces
    vn = np.zeros((len(mesh.vertices), 3))
    weighted = adj.normals * adj.areas[:, None]
    for corner in range(3):
        np.add.at(vn, faces[:, corner], weighted)
    vn = _unit_rows(vn, np.zeros_like(vn))
    nrm = vn[faces]
    dn = np.roll(nrm, -1, axis=1) - nrm  # normal change along side c
    u = np.einsum("fcj,fj->fc", sides, e1)
    v = np.einsum("fcj,fj->fc", sides, e2)
    du = np.einsum("fcj,fj->fc", dn, e1)
    dv = np.einsum("fcj,fj->fc", dn, e2)
    # Least squares for II = [[e, f], [f, g]] with II (u, v) = (du, dv) on the three sides.
    zero = np.zeros_like(u)
    design = np.concatenate((np.stack((u, v, zero), -1), np.stack((zero, u, v), -1)), axis=1)  # (F, 6, 3)
    rhs = np.concatenate((du, dv), axis=1)
    gram = np.einsum("fki,fkj->fij", design, design)
    ridge = 1e-9 * (np.trace(gram, axis1=1, axis2=2) / 3.0 + 1e-30)
    gram = gram + ridge[:, None, None] * np.eye(3)
    e, f, g = np.linalg.solve(gram, np.einsum("fki,fk->fi", design, rhs)[..., None])[..., 0].T
    theta = 0.5 * np.arctan2(2.0 * f, e - g)
    radius = np.hypot(0.5 * (e - g), f)
    k1, k2 = 0.5 * (e + g) + radius, 0.5 * (e + g) - radius
    aniso = np.abs(k1 - k2) / (np.abs(k1) + np.abs(k2) + 1e-9)
    return np.exp(4j * theta), np.where(adj.areas > 0, aniso, 0.0)


def estimate_field_gl(
    mesh: Any,
    adj: Any,
    feature_mask: np.ndarray,
    smooth_iters: int,
    curvature_weight: float = 0.5,
    report: dict | None = None,
) -> np.ndarray:
    """One unit cross direction per face, shape (F, 3). ``report`` (dict) receives singularity counts."""

    faces = adj.faces
    face_count = len(faces)
    e1, e2, sides, side_angle = _frames(mesh, adj)
    lengths = np.linalg.norm(sides, axis=2)
    is_feature = feature_mask[adj.face_edges]
    priority = lengths + is_feature * (float(lengths.max(initial=0.0)) + 1.0)
    pick = np.argmax(priority, axis=1)
    initial = np.exp(4j * side_angle[np.arange(face_count), pick])
    pinned = is_feature.any(axis=1)

    curv, aniso = _curvature_cross(mesh, adj, e1, e2, sides)
    pull = np.where(pinned, 0.0, float(curvature_weight) * aniso)  # soft pin strength
    cross = np.where(pinned, initial, np.where(pull > 0.2, curv, 0.05 * initial))

    link = (adj.counts == 2) & ~feature_mask
    half_a, half_b = adj.edge_halves[link].T
    face_a, corner_a = np.divmod(half_a, 3)
    face_b, corner_b = np.divmod(half_b, 3)
    transport = np.exp(4j * (side_angle[face_a, corner_a] - side_angle[face_b, corner_b]))
    idx = np.concatenate((face_a, face_b))
    degree = np.bincount(idx, minlength=face_count)
    mu = 1.0
    denom = degree + mu + pull
    pin_term = pull * curv
    dt = 0.4

    # ponytail: fixed-count relaxation (engine's 24 * smooth_iters budget); add a residual stop if needed.
    steps = 24 * max(0, int(smooth_iters))
    fast = (
        _turbo.cross_field_gl(cross, face_a, face_b, transport, pinned, denom, pin_term, steps, mu, dt)
        if _turbo is not None
        else None
    )
    for step in range(0 if fast is not None else steps):
        val = np.concatenate((cross[face_b] * transport, cross[face_a] * np.conj(transport)))
        acc = np.bincount(idx, weights=val.real, minlength=face_count) + 1j * np.bincount(
            idx, weights=val.imag, minlength=face_count
        )
        relaxed = (acc + mu * cross + pin_term) / denom
        if step >= steps // 2:  # diffuse first (kills seed noise), then sharpen with the double well
            relaxed = relaxed + dt * (1.0 - np.abs(relaxed) ** 2) * relaxed
        cross = np.where(pinned, cross, relaxed)
    if fast is not None:
        cross = fast

    magnitude = np.abs(cross)
    cross = np.where(magnitude > 1e-12, cross / np.maximum(magnitude, 1e-12), initial)
    if report is not None:
        report.update(field_singularities(mesh, adj, cross, side_angle))
    theta = np.angle(cross) / 4.0
    return np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2


def field_singularities(
    mesh: Any, adj: Any, cross: np.ndarray, side_angle: np.ndarray | None = None
) -> dict[str, Any]:
    """Vertex indices from the holonomy of ``cross`` (exp(4i theta) per face) around each one-ring.

    index = (sum of wrapped matching jumps around the ring + angle defect) / 2 pi.
    ``quarters`` is the integer array of 4 * index (+1 == index 1/4); boundary and
    non-manifold vertices stay 0.
    """

    faces = adj.faces
    vertex_count = len(mesh.vertices)
    if side_angle is None:
        side_angle = _frames(mesh, adj)[3]
    interior = np.flatnonzero(adj.counts == 2)
    half_a, half_b = adj.edge_halves[interior].T
    face_a, corner_a = np.divmod(half_a, 3)
    face_b, corner_b = np.divmod(half_b, 3)
    transport = np.exp(4j * (side_angle[face_a, corner_a] - side_angle[face_b, corner_b]))
    jump = np.angle(cross[face_b] * transport * np.conj(cross[face_a])) / 4.0  # fa -> fb
    tail = faces[face_a, corner_a]
    head = faces[face_a, (corner_a + 1) % 3]
    total = np.bincount(head, weights=jump, minlength=vertex_count) - np.bincount(
        tail, weights=jump, minlength=vertex_count
    )
    corners = mesh.vertices[faces]
    angle_sum = np.zeros(vertex_count)
    for c in range(3):
        a = corners[:, (c + 1) % 3] - corners[:, c]
        b = corners[:, (c + 2) % 3] - corners[:, c]
        cos = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
        np.add.at(angle_sum, faces[:, c], np.arccos(np.clip(cos, -1.0, 1.0)))
    index = (total + (2.0 * np.pi - angle_sum)) / (2.0 * np.pi)
    quarters = np.rint(4.0 * index).astype(np.int64)
    skip = np.ones(vertex_count, dtype=bool)
    skip[faces.reshape(-1)] = False
    skip[adj.edges[adj.counts != 2].reshape(-1)] = True
    quarters[skip] = 0
    return {
        "quarters": quarters,
        "positive": int(np.count_nonzero(quarters > 0)),
        "negative": int(np.count_nonzero(quarters < 0)),
        "total_index": float(quarters.sum()) / 4.0,
    }
