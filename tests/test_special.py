# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import special  # noqa: E402
from triadq.types import drain  # noqa: E402


def _tube(rings=40, sides=12, radius=0.02, bend=0.5):
    """A long bent tube, capped by nothing: an open wire."""
    t = np.linspace(0.0, 1.0, rings)
    path = np.column_stack([t, bend * t * t, np.zeros(rings)])
    angle = 2.0 * np.pi * np.arange(sides) / sides
    points = np.vstack([p + radius * np.column_stack([np.zeros(sides), np.cos(angle), np.sin(angle)]) for p in path])
    polys = [(i * sides + j, i * sides + (j + 1) % sides, (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j) for i in range(rings - 1) for j in range(sides)]
    return points, polys


def _cloth(n=30):
    u, v = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n), indexing="ij")
    points = np.column_stack([u.ravel(), v.ravel(), 0.05 * np.sin(12.0 * u.ravel())])
    polys = [(i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1) for i in range(n - 1) for j in range(n - 1)]
    return points, polys


class SpecialPartsTest(unittest.TestCase):
    def test_wire_becomes_a_few_sided_tube(self):
        points, polys = _tube()
        fit = drain(special.fit_tube_steps(points, polys))
        self.assertIsNotNone(fit)
        out, quads, _kinds = special.build_tube(fit, 0.3, 0.5)
        self.assertLess(len(quads), len(polys) / 2)
        self.assertTrue(all(len(q) == 4 for q in quads))
        self.assertLess(np.abs(out[:, 0].max() - points[:, 0].max()), 0.06)

    def test_stack_of_fins_is_swept_from_its_own_outline(self):
        comb = np.asarray([(x, y) for k in range(4) for x, y in ((k, 0.0), (k + 0.4, 0.0), (k + 0.4, 1.0), (k + 0.6, 1.0), (k + 0.6, 0.0))] + [(4.0, 0.0), (4.0, -0.5), (0.0, -0.5)])
        rings = 6
        points = np.vstack([np.column_stack([comb, np.full(len(comb), z)]) for z in np.linspace(0.0, 2.0, rings)])
        n = len(comb)
        polys = [(i * n + j, i * n + (j + 1) % n, (i + 1) * n + (j + 1) % n, (i + 1) * n + j) for i in range(rings - 1) for j in range(n)]
        tris = special._clip_ears(comb)
        self.assertEqual(len(tris), n - 2)
        polys += [tuple(reversed(t)) for t in tris] + [tuple((rings - 1) * n + v for v in t) for t in tris]
        builds = [special.build_tube(fit, 0.01, 0.1) for fit in drain(special.fit_prism_steps(points, polys))]
        self.assertTrue(any(len(faces) < len(polys) and np.abs(np.ptp(out, axis=0) - np.ptp(points, axis=0)).max() < 0.02 for out, faces, _kinds in builds))

    def test_processed_vertices_keep_the_part_they_came_from(self):
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        known = np.vstack([square, square + 5.0])
        known_faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
        points = np.vstack([known, [[0.5, 0.5, 0.0]]])  # closing added a vertex in the middle of the first
        faces = [(0, 1, 8), (1, 2, 8), (2, 3, 8), (3, 0, 8), (4, 5, 6, 7)]
        self.assertEqual(special.vertex_owners(points, faces, known, known_faces, np.array([7, 9])).tolist(), [7, 7, 7, 7, 9, 9, 9, 9, 7])

    def test_cloth_rows_follow_the_folds(self):
        points, polys = _cloth()
        out, quads, _kinds = special.build_sheet(drain(special.fit_sheet_steps(points, polys)), 60)
        self.assertLess(len(quads), 80)
        # the waves run across x, so the lines are spent on x and the long sides of the quads lie along y
        sides = np.asarray([out[q[1]] - out[q[0]] for q in quads] + [out[q[3]] - out[q[0]] for q in quads])
        along = np.abs(sides[:, 1]) > np.abs(sides[:, 0])
        self.assertGreater(np.linalg.norm(sides[along], axis=1).mean(), np.linalg.norm(sides[~along], axis=1).mean())
        self.assertLess(np.abs(np.ptp(out, axis=0)[:2] - 1.0).max(), 0.05)

    def test_copy_found_turned_or_mirrored_in_any_vertex_order(self):
        points, _polys = _tube(rings=10, sides=8)
        points = points * np.array([1.0, 1.0, 0.5]) + np.random.default_rng(1).normal(0.0, 1e-5, points.shape)
        angle = 0.7
        turn = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        shuffled = np.random.default_rng(2).permutation(points @ turn.T + np.array([5.0, 1.0, 0.0]))
        move = special.match_copy(points, shuffled)
        self.assertIsNotNone(move)
        self.assertLess(special._nearest(special.moved(points, move), shuffled).max(), 0.01)
        self.assertIsNotNone(special.match_copy(points, shuffled * np.array([-1.0, 1.0, 1.0])))
        self.assertIsNone(special.match_copy(points, shuffled * np.array([1.3, 1.0, 1.0])))

    def test_nearly_symmetric_copy_is_placed_mirrored_not_turned(self):
        # A part almost symmetric across its own plane also fits a turn onto its mirror image, within tolerance, with
        # the faces then landing a little off: the placement that fits exactly must win.
        rng = np.random.default_rng(3)
        half = rng.uniform(-1.0, 1.0, (120, 3)) * np.array([1.0, 2.0, 0.3])
        points = np.vstack((half, half * np.array([1.0, 1.0, -1.0])))
        points[:20, 2] += 0.002  # bumps on one side only, under the tolerance
        mirrored = rng.permutation(points * np.array([-1.0, 1.0, 1.0]) + np.array([4.0, 0.0, 0.0]))
        move = special.match_copy(points, mirrored)
        self.assertIsNotNone(move)
        self.assertLess(special._nearest(special.moved(points, move), mirrored).max(), 1e-9)


if __name__ == "__main__":
    unittest.main()
