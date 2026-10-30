# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import triadq.engine as engine  # noqa: E402
from triadq.field import estimate_field_gl, field_singularities  # noqa: E402


def grid_mesh(n: int, m: int, surface) -> tuple[np.ndarray, np.ndarray]:
    uu, vv = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, m + 1), indexing="ij")
    points = surface(uu, vv).reshape(-1, 3)
    idx = np.arange((n + 1) * (m + 1)).reshape(n + 1, m + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return points, faces


def cube(n: int) -> tuple[np.ndarray, np.ndarray]:
    parts, offset = [], 0
    for axis in range(3):
        for side in (0.0, 1.0):
            def surface(u, v, axis=axis, side=side):
                cols = [None] * 3
                cols[axis], cols[(axis + 1) % 3], cols[(axis + 2) % 3] = np.full_like(u, side), u, v
                return np.stack(cols, -1)

            p, f = grid_mesh(n, n, surface)
            parts.append((p, f[:, ::-1] if side == 0.0 else f, offset))
            offset += len(p)
    points = np.concatenate([p for p, _, _ in parts])
    faces = np.concatenate([f + o for _, f, o in parts])
    _, first, inverse = np.unique(np.round(points * 1e6).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    return points[first], inverse.reshape(-1)[faces]


def tri_mesh(points: np.ndarray, faces: np.ndarray) -> engine._TriMesh:
    return engine._TriMesh(
        vertices=points,
        faces=[tuple(f) for f in faces.tolist()],
        face_materials=["d"] * len(faces),
        face_uvs=None,
        uvs=None,
        name="t",
    )


def run(mesh: engine._TriMesh, features: np.ndarray | None = None, iters: int = 4, **kw):
    adj = engine._adjacency(mesh)
    mask = np.zeros(len(adj.edges), bool) if features is None else features(adj)
    report: dict = {}
    return adj, estimate_field_gl(mesh, adj, mask, iters, report=report, **kw), report, mask


def alignment(adj, mesh, field, target) -> np.ndarray:
    """|cos(4 * angle)| between field and target per face (1 == aligned modulo 90 degrees)."""
    n = adj.normals
    t = target - n * np.einsum("ij,ij->i", target, n)[:, None]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    cos = np.clip(np.einsum("ij,ij->i", field, t), -1, 1)
    return np.abs(np.cos(4 * np.arccos(cos)))


class GinzburgLandauFieldTests(unittest.TestCase):
    def test_cylinder_aligns_with_axis_or_circumference(self) -> None:
        def surface(u, v):
            a = 2 * np.pi * u
            return np.stack([np.cos(a), np.sin(a), 3.0 * v], -1)

        points, faces = grid_mesh(48, 24, surface)
        # weld the seam so the surface is a closed tube
        points = points.reshape(49, 25, 3)[:-1].reshape(-1, 3)
        idx = np.arange(48 * 25).reshape(48, 25)
        idx = np.concatenate([idx, idx[:1]])
        a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        mesh = tri_mesh(points, faces)
        adj, field, _, _ = run(mesh)
        axis = np.broadcast_to(np.array([0.0, 0.0, 1.0]), field.shape)
        score = alignment(adj, mesh, field, axis)
        self.assertGreater(float(np.median(score)), 0.95)
        self.assertGreater(float(np.mean(score > 0.8)), 0.9)

    def test_cube_singularities_sit_at_corners(self) -> None:
        points, faces = cube(6)
        mesh = tri_mesh(points, faces)
        _, _, report, _ = run(mesh)
        quarters = report["quarters"]
        self.assertAlmostEqual(report["total_index"], 2.0)  # Euler characteristic of a sphere
        corners = np.all(np.isin(np.round(points, 6), (0.0, 1.0)), axis=1)
        self.assertEqual(int(corners.sum()), 8)
        self.assertTrue(np.all(quarters[corners] == 1), quarters[corners])
        self.assertEqual(int(np.count_nonzero(quarters[~corners])), 0)

    def test_vortex_sign_convention(self) -> None:
        points, faces = grid_mesh(20, 20, lambda u, v: np.stack([u - 0.5, v - 0.5, 0 * u], -1))
        mesh = tri_mesh(points, faces)
        adj = engine._adjacency(mesh)
        e1, e2, _, side_angle = (
            __import__("triadq.field", fromlist=["_frames"])._frames(mesh, adj)
        )
        centre = points[faces].mean(axis=1)
        phi = np.arctan2(centre[:, 1], centre[:, 0]) / 4.0  # winding +1/4 around origin
        world = np.cos(phi)[:, None] * np.array([1, 0, 0]) + np.sin(phi)[:, None] * np.array([0, 1, 0])
        theta = np.arctan2(np.einsum("ij,ij->i", world, e2), np.einsum("ij,ij->i", world, e1))
        report = field_singularities(mesh, adj, np.exp(4j * theta), side_angle)
        self.assertEqual(report["quarters"].max(), 1)
        self.assertEqual(int(np.count_nonzero(report["quarters"])), 1)

    def test_noisy_plane_at_least_as_aligned_as_old_field(self) -> None:
        rng = np.random.default_rng(3)
        points, faces = grid_mesh(30, 30, lambda u, v: np.stack([u, v, 0 * u], -1))
        points = points + np.concatenate([rng.normal(0, 0.004, (len(points), 2)), np.zeros((len(points), 1))], 1)
        mesh = tri_mesh(points, faces)
        adj = engine._adjacency(mesh)
        mask = np.zeros(len(adj.edges), bool)
        old = engine._estimate_field(mesh, adj, mask, 4)
        new = estimate_field_gl(mesh, adj, mask, 4)

        def coherence(field: np.ndarray) -> float:  # 1.0 == every face on one common cross
            return float(abs(np.exp(4j * np.arctan2(field[:, 1], field[:, 0])).mean()))

        self.assertGreater(coherence(new), 0.99)
        self.assertGreaterEqual(coherence(new), coherence(old) - 0.005)

    def test_deterministic(self) -> None:
        points, faces = cube(4)
        mesh = tri_mesh(points, faces)
        _, first, _, _ = run(mesh)
        _, second, _, _ = run(tri_mesh(points.copy(), faces.copy()))
        np.testing.assert_array_equal(first, second)

    def test_engine_field_mode_curvature(self) -> None:
        from triadq import MeshData

        def bump(u, v):
            return np.stack([u, v, 0.3 * np.sin(np.pi * u) * np.sin(np.pi * v)], -1)

        points, faces = grid_mesh(10, 10, bump)
        mesh = MeshData(vertices=points, faces=[tuple(f) for f in faces.tolist()])
        outputs = []
        for mode in ("SMOOTH", "CURVATURE"):
            options = engine.RemeshOptions(target_faces=100, seed_count=2, field_mode=mode)
            out, report = engine.remesh_mesh(mesh, options)
            self.assertTrue(report.success, report.message)
            self.assertGreater(report.quads, 0)
            outputs.append(sorted(out.faces))
        self.assertEqual(engine.RemeshOptions().field_mode, "SMOOTH")


if __name__ == "__main__":
    unittest.main()
