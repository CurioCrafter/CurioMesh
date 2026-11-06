# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.flowtool import creases, find_tool, fit_normals, flow_remesh  # noqa: E402


def box_tris():
    points = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    sides = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return points, np.array([t for a, b, c, d in sides for t in ((a, b, c), (a, c, d))])


class FlowToolTests(unittest.TestCase):
    def test_missing_tool_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(find_tool(empty))
            self.assertIsNone(find_tool(None))
            self.assertIsNone(flow_remesh(empty, np.zeros((3, 3)), np.array([[0, 1, 2]]), 10, None, None, None, None))

    def test_box_edges_are_creases_listed_from_both_faces_and_short_lines_are_dropped(self):
        points, tris = box_tris()
        found = creases(points, tris)
        self.assertEqual(len(found), 12 * 2)
        self.assertEqual(creases(points, tris, length_share=2.0), [])

    def test_a_tilted_face_turns_toward_the_source_normal(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.3], [0.0, 1.0, 0.3]])
        quads = np.array([[0, 1, 2, 3]])
        moved = fit_normals(points, quads, np.array([[0.5, 0.5, 0.15]]), np.array([[0.0, 0.0, 1.0]]), np.ones(1), lambda spots, samples: np.zeros(len(samples), dtype=np.int64))
        self.assertLess(np.ptp(moved[:, 2]), 0.3)
        np.testing.assert_allclose(moved[:, :2], points[:, :2])


if __name__ == "__main__":
    unittest.main()
