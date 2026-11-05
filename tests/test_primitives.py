# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import primitives as P  # noqa: E402


def turn(seed):
    q, _r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q


def on_box(half, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-1.0, 1.0, (n, 3))
    k = rng.integers(0, 3, n)
    p[np.arange(n), k] = rng.choice((-1.0, 1.0), n)
    return p * half


def on_cylinder(radius, length, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), rng.uniform(-length / 2, length / 2, n)])


def on_sphere(radius, n=4000, seed=0):
    p = np.random.default_rng(seed).normal(size=(n, 3))
    return radius * p / np.linalg.norm(p, axis=1, keepdims=True)


class FitTest(unittest.TestCase):
    def test_box_size_and_axes(self):
        rot, shift = turn(1), np.array([3.0, -2.0, 1.0])
        fit = P.fit_box(on_box(np.array([2.0, 1.0, 0.5])) @ rot + shift)
        np.testing.assert_allclose(fit["half"], [2.0, 1.0, 0.5], atol=0.02)
        np.testing.assert_allclose(fit["center"], shift, atol=0.02)
        self.assertGreater(abs(fit["axes"][0] @ rot[0]), 0.999)  # the long side found
        self.assertGreater(abs(fit["axes"][2] @ rot[2]), 0.999)

    def test_cube_is_found_turned(self):
        rot = turn(2)
        fit = P.fit_box(on_box(np.ones(3)) @ rot)  # no principal axes on a cube: the volume search finds the frame
        self.assertLess(float(np.prod(fit["half"])), 1.03)

    def test_cylinder_radius_length_axis(self):
        rot = turn(3)
        fit = P.fit_cylinder(on_cylinder(0.3, 2.0) @ rot)
        self.assertAlmostEqual(fit["half"][0], 0.3, delta=0.01)
        self.assertAlmostEqual(fit["half"][2], 1.0, delta=0.01)
        self.assertGreater(abs(fit["axes"][2] @ rot[2]), 0.999)

    def test_flat_cylinder_axis(self):
        rot = turn(4)  # a coin: the axis is the short side, not the long one
        fit = P.fit_cylinder(on_cylinder(1.0, 0.2) @ rot)
        self.assertGreater(abs(fit["axes"][2] @ rot[2]), 0.999)
        self.assertAlmostEqual(fit["half"][0], 1.0, delta=0.02)

    def test_sphere_radius(self):
        fit = P.fit_ellipsoid(on_sphere(0.7) + 5.0)
        np.testing.assert_allclose(fit["half"], 0.7, atol=0.02)
        np.testing.assert_allclose(fit["center"], 5.0, atol=0.02)

    def test_built_meshes_are_closed_outward_quads(self):
        for fit in P.fits(on_cylinder(0.5, 1.0)):
            points, faces = P.build(fit, 12)
            self.assertTrue(all(len(f) == 4 for f in faces), fit["kind"])
            edges = {}
            for f in faces:
                for a, b in zip(f, f[1:] + f[:1]):
                    edges[(a, b)] = edges.get((a, b), 0) + 1
            self.assertTrue(all((b, a) in edges for a, b in edges), fit["kind"])  # closed and wound one way

    def test_view_miss(self):
        hit = np.zeros((4, 4), bool)
        hit[:2] = True
        depth = np.zeros((4, 4))
        same = [(hit, None, depth, None)]
        self.assertEqual(P.view_miss(same, same, 0.1), 0.0)
        moved = [(hit, None, depth + np.eye(4), None)]
        self.assertAlmostEqual(P.view_miss(same, moved, 0.1), 2 / 8)


if __name__ == "__main__":
    unittest.main()
