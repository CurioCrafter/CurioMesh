# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import hardsurf, parts  # noqa: E402


def prism(outline, height=1.0, rows=3):
    """An extruded 2D outline (CCW), sides cut into `rows` bands, caps as triangle fans from the centre (a messy source)."""
    n = len(outline)
    co = [(x, y, height * r / rows) for r in range(rows + 1) for x, y in outline]
    faces = []
    for r in range(rows):
        for i in range(n):
            a, b = r * n + i, r * n + (i + 1) % n
            faces += [(a, b, b + n), (a, b + n, a + n)]
    for z, base, flip in ((0.0, 0, True), (height, rows * n, False)):
        c = len(co)
        co.append((float(np.mean([p[0] for p in outline])), float(np.mean([p[1] for p in outline])), z))
        for i in range(n):
            f = (c, base + i, base + (i + 1) % n)
            faces.append(f[::-1] if flip else f)
    return np.asarray(co, dtype=float), faces


def plate_with_hole(size=4.0, hole=0.5, sides=16, thick=0.2):
    """A square plate with a round hole, its top and bottom as one band of triangles from the edge to the hole."""
    per = sides // 4
    square = []
    for k in range(4):
        a = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)][k], dtype=float) * size / 2
        b = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)][(k + 1) % 4], dtype=float) * size / 2
        square += [tuple(a + (b - a) * j / per) for j in range(per)]
    start = math.atan2(square[0][1], square[0][0])
    ring = [(hole * math.cos(start + 2 * math.pi * k / sides), hole * math.sin(start + 2 * math.pi * k / sides)) for k in range(sides)]
    co, faces = [], []
    for z in (0.0, thick):
        o = len(co)
        co += [(x, y, z) for x, y in square] + [(x, y, z) for x, y in ring]
        for k in range(sides):
            a, b, c, d = o + k, o + (k + 1) % sides, o + sides + (k + 1) % sides, o + sides + k
            tri = [(a, b, c), (a, c, d)]
            faces += [t[::-1] for t in tri] if z == 0.0 else tri
    n = 2 * sides
    for loop, flip in ((list(range(sides)), False), (list(range(sides, 2 * sides)), True)):
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            tri = [(a, b, b + n), (a, b + n, a + n)]
            faces += [t[::-1] for t in tri] if flip else tri
    return np.asarray(co, dtype=float), faces


def quality(co, quads):
    """(edges used more than twice, open edges, folded quads)."""
    use = {}
    for q in quads:
        for k in range(4):
            e = tuple(sorted((q[k], q[(k + 1) % 4])))
            use[e] = use.get(e, 0) + 1
    p = np.asarray(co)[np.asarray(quads)]
    n1 = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n2 = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])
    folded = int(((n1 * n2).sum(axis=1) < 0).sum())
    return sum(v > 2 for v in use.values()), sum(v == 1 for v in use.values()), folded


class CleanPanels(unittest.TestCase):
    def test_box_is_six_quads(self):
        co, faces = prism([(0, 0), (2, 0), (2, 1), (0, 1)], 1.0, rows=4)
        out = hardsurf.panel_rebuild(co, faces, 0.0)
        self.assertIsNotNone(out)
        self.assertEqual(len(out[1]), 6)
        self.assertEqual(quality(out[0], out[1]), (0, 0, 0))

    def test_box_grid_follows_edges(self):
        co, faces = prism([(0, 0), (4, 0), (4, 1), (0, 1)], 1.0, rows=4)
        out = hardsurf.panel_rebuild(co, faces, 1.0 / 8.0)
        self.assertIsNotNone(out)
        self.assertEqual(quality(out[0], out[1]), (0, 0, 0))
        # every edge runs along x, y or z: a grid along the panel's own edges
        p = out[0][np.asarray(out[1])]
        d = np.abs(np.roll(p, -1, axis=1) - p)
        along = (d > 1e-9).sum(axis=2)
        self.assertTrue((along == 1).all())

    def test_hexagon_caps_have_no_fan(self):
        outline = [(math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)) for k in range(6)]
        co, faces = prism(outline, 2.0)
        out = hardsurf.panel_rebuild(co, faces, 0.0)
        self.assertIsNotNone(out)
        self.assertEqual(quality(out[0], out[1]), (0, 0, 0))
        self.assertTrue(all(len(f) == 4 for f in out[1]))
        self.assertLessEqual(len(out[1]), 6 + 2 * 2)

    def test_odd_prism_is_balanced(self):
        outline = [(math.cos(k * 2 * math.pi / 5), math.sin(k * 2 * math.pi / 5)) for k in range(5)]
        co, faces = prism(outline, 2.0)
        out = hardsurf.panel_rebuild(co, faces, 0.0)
        self.assertIsNotNone(out)
        self.assertEqual(quality(out[0], out[1]), (0, 0, 0))

    def test_l_shape_notch(self):
        co, faces = prism([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)], 0.5)
        out = hardsurf.panel_rebuild(co, faces, 0.0)
        self.assertIsNotNone(out)
        self.assertEqual(quality(out[0], out[1]), (0, 0, 0))

    def test_plate_with_hole(self):
        co, faces = plate_with_hole()
        for step in hardsurf.PANEL_STEPS:
            out = hardsurf.panel_rebuild(co, faces, step)
            self.assertIsNotNone(out, step)
            self.assertEqual(quality(out[0], out[1]), (0, 0, 0))

    def test_round_part_is_not_panels(self):
        sphere_co = []
        for i in range(1, 12):
            for j in range(16):
                t, p = math.pi * i / 12, 2 * math.pi * j / 16
                sphere_co.append((math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)))
        faces = [(i * 16 + j, i * 16 + (j + 1) % 16, (i + 1) * 16 + (j + 1) % 16) for i in range(10) for j in range(16)]
        self.assertIsNone(hardsurf.panel_layout(np.asarray(sphere_co), faces))


class EvenCylinders(unittest.TestCase):
    def test_noisy_tube_is_found(self):
        rng = np.random.default_rng(1)
        sides, rings = 40, 30
        co = np.array([(np.cos(2 * np.pi * i / sides), np.sin(2 * np.pi * i / sides), 4.0 * r / rings) for r in range(rings + 1) for i in range(sides)])
        co += rng.normal(0, 0.01, co.shape)
        faces = []
        for r in range(rings):
            for i in range(sides):
                a, b = r * sides + i, r * sides + (i + 1) % sides
                faces += [(a, b, b + sides), (a, b + sides, a + sides)]
        fit = hardsurf.fit_turned(co, faces)
        self.assertIsNotNone(fit)
        built = parts.build_revolve(fit, 16, 0.01)
        self.assertIsNotNone(built)
        self.assertTrue(all(len(f) == 4 for f in built[1]))
        radius = np.hypot(*(built[0] @ fit["frame"] - fit["center"])[:, :2].T)
        self.assertLess(float(np.abs(radius - radius.mean()).max()), 0.1)

    def test_tree_path_follows_a_chain(self):
        edges = [(1.0, 0, 1), (1.0, 1, 2), (1.0, 2, 3), (5.0, 0, 3), (0.5, 1, 4)]
        self.assertIn(hardsurf._tree_path(5, edges), ([0, 1, 2, 3], [3, 2, 1, 0]))


if __name__ == "__main__":
    unittest.main()
