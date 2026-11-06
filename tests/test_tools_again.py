# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Look First for the tools that already existed: part swaps and Detail Markers (pure parts)."""

import unittest

import numpy as np

from triadq import tidytools, trial


def _strip(n: int, x0: float = 0.0):
    points = np.array([(x0 + x, y, 0.0) for y in range(2) for x in range(n + 1)], dtype=np.float64)
    faces = [(x, x + 1, n + 1 + x + 1, n + 1 + x) for x in range(n)]
    return points, faces


class SwapPartsTest(unittest.TestCase):
    def _swap(self):
        """Part 0 is two quads, part 1 two quads; part 1 is swapped for four smaller ones."""
        co, faces = _strip(4)
        old_piece = np.array([0, 0, 1, 1])
        small_co, small_faces = _strip(4)
        small_co[:, 0] = 2.0 + small_co[:, 0] * 0.5
        new_faces = faces[:2] + [tuple(v + len(co) for v in f) for f in small_faces]
        new_co = np.concatenate((co, small_co))
        new_piece = np.array([0, 0, 1, 1, 1, 1])
        return co, faces, old_piece, new_co, new_faces, new_piece

    def test_parent_points_into_the_same_part(self):
        _co, _faces, old_piece, _nco, _nf, new_piece = self._swap()
        parent = tidytools.swap_parent(old_piece, new_piece)
        self.assertEqual(parent.tolist(), [0, 0, 2, 2, 2, 2])
        self.assertEqual(tidytools.swap_parent(np.array([0]), np.array([0, 3])).tolist(), [0, -1])
        self.assertEqual(len(tidytools.swap_parent(old_piece, np.zeros(0))), 0)

    def test_candidate_is_the_swapped_result_and_only_the_part_is_ghosted(self):
        co, faces, old_piece, new_co, new_faces, new_piece = self._swap()
        base = trial.make_base(co, faces, scope=old_piece == 1)
        tool = type("Now", (tidytools.SwapParts,), {"name": "Smaller Quads", "swap": (new_co, new_faces, new_piece, old_piece, (1,))})()
        self.assertEqual(tool.check(base), "")
        cand = trial.diff(base, trial.finish(tool.run(base, {})))
        self.assertTrue(cand.exact)
        self.assertEqual(len(cand.faces), 6)
        ghosted = [i for i in range(len(cand.faces)) if base.scope[cand.parent[i]]]
        self.assertEqual(ghosted, [2, 3, 4, 5])
        self.assertIn("2 -> 4 faces", cand.note)
        trial.finish(trial.measure(base, cand, ("faces", "poles")))
        self.assertEqual(cand.numbers["faces"], (4, 6))

    def test_nothing_to_swap_is_refused(self):
        co, faces = _strip(2)
        base = trial.make_base(co, faces)
        self.assertTrue(tidytools.SwapParts().check(base))


class MarkerTest(unittest.TestCase):
    def test_more_detail_asks_for_smaller_quads_and_less_for_bigger(self):
        self.assertAlmostEqual(tidytools.marker_faces_factor(0.0), 1.0)
        self.assertAlmostEqual(tidytools.marker_faces_factor(1.0), 4.0)  # 0.98 keeps about four times the faces
        self.assertAlmostEqual(tidytools.marker_scale(1.0), 0.5)
        self.assertAlmostEqual(tidytools.marker_scale(1.0, less=True), 2.0)
        self.assertLess(tidytools.marker_scale(1.0, density=4.0), tidytools.marker_scale(1.0))  # a stronger Strength asks for more

    def test_extra_faces_grow_with_reach_and_strength_and_go_negative_for_less(self):
        a = tidytools.marker_extra_faces(1.0, 0.1, 1.0)
        self.assertGreater(a, 0)
        self.assertGreater(tidytools.marker_extra_faces(2.0, 0.1, 1.0), a)
        self.assertGreater(a, tidytools.marker_extra_faces(1.0, 0.1, 0.3))
        self.assertLess(tidytools.marker_extra_faces(1.0, 0.1, 1.0, less=True), 0)
        self.assertEqual(tidytools.marker_extra_faces(0.0, 0.1, 1.0), 0)

    def test_sample_grid_fills_the_circle_with_squares_of_the_size(self):
        points, quads = tidytools.sample_grid(1.0, 0.25)
        self.assertTrue(len(quads))
        side = np.linalg.norm(points[quads[:, 1]] - points[quads[:, 0]], axis=1)
        self.assertTrue(np.allclose(side, 0.25))
        self.assertLessEqual(float(np.linalg.norm(points[quads].mean(axis=1), axis=1).max()), 1.0)
        tiny_points, tiny = tidytools.sample_grid(1.0, 1e-6)  # capped for drawing, never millions of quads
        self.assertLessEqual(len(tiny), tidytools.GRID_MOST**2)
        self.assertEqual(len(tidytools.sample_grid(0.0, 0.1)[1]), 0)


def _cube():
    co = np.array([(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=np.float64)
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return co, faces


class HardEdgesTest(unittest.TestCase):
    def test_cube_edges_are_hard_below_ninety_and_not_above(self):
        co, faces = _cube()
        edges, angles = tidytools.edge_angles(co, faces)
        self.assertEqual(len(edges), 12)
        self.assertTrue(np.allclose(angles, 90.0))
        base = trial.make_base(co, faces)
        tool = tidytools.HardEdges()
        cand = trial.finish(tool.run(base, {"angle": tidytools.value_of(40.0)}))
        self.assertEqual(len(cand.lines), 12)
        self.assertTrue(np.array_equal(cand.co, co))  # nothing moves
        self.assertIn("12 hard edges at 40", cand.note)
        self.assertEqual(len(trial.finish(tool.run(base, {"angle": 1.0})).lines), 0)

    def test_flat_strip_has_none_and_the_dial_reads_in_degrees(self):
        co, faces = _strip(3)
        self.assertEqual(len(tidytools.edge_angles(co, faces)[0]), 2)  # border edges are left out
        self.assertAlmostEqual(tidytools.angle_of(tidytools.value_of(40.0)), 40.0)
        title = trial.readout(None, tidytools.HardEdges(), {"angle": tidytools.value_of(40.0)}, "ready")[0]
        self.assertIn("Angle 40 degrees", title)
        self.assertIn("Strength 0.60", trial.readout(None, tidytools.SmoothCorners(), {"strength": 0.6}, "ready")[0])


if __name__ == "__main__":
    unittest.main()
