# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import unittest
from types import SimpleNamespace

import numpy as np

from triadq.uvs import chart_codes, is_see_through, simplify_borders, uv_overlap


class UvHelpersTest(unittest.TestCase):
    def test_chart_codes_round_trip(self):
        materials, islands = np.array([0, 0, 1, 0]), np.array([5, 7, 5, 5])
        codes, by_material, by_island = chart_codes(materials, islands)
        self.assertEqual(by_material[codes].tolist(), materials.tolist())
        self.assertEqual(by_island[codes].tolist(), islands.tolist())
        self.assertEqual(codes[0], codes[3])
        self.assertEqual(len(set(codes.tolist())), 3)

    def test_straight_border_loses_its_middle_but_stays_a_border(self):
        # Two rows of four quads, split by a straight border along y = 1, with a bend at x = 2 when asked.
        for bend, expected in ((0.0, 2), (0.5, 3)):
            points = np.array([[x, y, bend * (y == 1) * (1 - abs(x - 2) / 2)] for y in range(3) for x in range(5)], dtype=float)
            polys = [(y * 5 + x, y * 5 + x + 1, (y + 1) * 5 + x + 1, (y + 1) * 5 + x) for y in range(2) for x in range(4)]
            kinds = ["a"] * 4 + ["b"] * 4
            out_points, out_polys, out_kinds, free, _used = simplify_borders(points, polys, kinds, 0.01)
            self.assertLess(len(out_points), len(points) + 1)
            on_border = set(range(len(out_points))) - set(free)
            self.assertEqual(len(on_border), expected)
            self.assertTrue(all(abs(out_points[v][1] - 1) < 1e-9 for v in on_border))
            shared = {v for p, k in zip(out_polys, out_kinds) if k == "a" for v in p} & {v for p, k in zip(out_polys, out_kinds) if k == "b" for v in p}
            self.assertEqual(shared, on_border)

    def test_fixed_vertex_on_a_border_stays(self):
        # A region's border vertex sits on the straight UV border: it must survive, or the neighbour region gets a gap.
        points = np.array([[x, y, 0.0] for y in range(3) for x in range(5)], dtype=float)
        polys = [(y * 5 + x, y * 5 + x + 1, (y + 1) * 5 + x + 1, (y + 1) * 5 + x) for y in range(2) for x in range(4)]
        out_points, _polys, _kinds, free, _used = simplify_borders(points, polys, ["a"] * 4 + ["b"] * 4, 0.01, fixed=[6])
        self.assertIn([1.0, 1.0, 0.0], out_points.tolist())
        self.assertEqual(len(set(range(len(out_points))) - set(free)), 3)

    def test_overlap(self):
        left = [[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]]
        right = [[1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
        self.assertEqual(uv_overlap(np.array([left, right])), 0.0)
        self.assertGreater(uv_overlap(np.array([left, left])), 0.9)
        self.assertEqual(uv_overlap(np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]])), 1.0)


def _plug(value=0.0, linked=False):
    return SimpleNamespace(default_value=value, is_linked=linked)


def _material(*nodes, **extra):
    return SimpleNamespace(use_nodes=True, node_tree=SimpleNamespace(nodes=list(nodes)), **extra)


class SeeThroughTests(unittest.TestCase):
    def test_opaque_shader_can_be_baked(self):
        shader = SimpleNamespace(type="BSDF_PRINCIPLED", inputs={"Alpha": _plug(1.0), "Transmission Weight": _plug(0.0)})
        self.assertFalse(is_see_through(_material(shader, blend_method="OPAQUE")))
        self.assertFalse(is_see_through(None))
        # glTF opaque materials come in with the legacy blend_method HASHED
        self.assertFalse(is_see_through(_material(shader, blend_method="HASHED", surface_render_method="DITHERED")))
        self.assertTrue(is_see_through(_material(shader, blend_method="OPAQUE", surface_render_method="BLENDED")))

    def test_glass_is_kept_out_of_the_bake(self):
        glass = SimpleNamespace(type="BSDF_GLASS", inputs={})
        faded = SimpleNamespace(type="BSDF_PRINCIPLED", inputs={"Alpha": _plug(0.4)})
        clear = SimpleNamespace(type="BSDF_PRINCIPLED", inputs={"Transmission": _plug(1.0)})
        for node in (glass, faded, clear):
            self.assertTrue(is_see_through(_material(node)))
        self.assertTrue(is_see_through(_material(blend_method="BLEND")))


if __name__ == "__main__":
    unittest.main()
