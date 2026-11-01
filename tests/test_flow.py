# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from triadq.flow import snap_new_vertices, snap_rows
from triadq.patch import quad_quality
from triadq.types import MeshData


def _grid(n: int):
    co = np.array([[x, y, 0.0] for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
    at = lambda x, y: y * (n + 1) + x  # noqa: E731
    return co, [(at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)) for y in range(n) for x in range(n)]


def _flat(point):
    return (np.array([point[0], point[1], 0.0]), None, 0, abs(point[2]))


class FlowTest(unittest.TestCase):
    def test_snap_moves_only_added_vertices(self):
        co, faces = _grid(2)
        co[4, 2] = 0.1  # the middle vertex was added by closure, and sits off the surface
        co[0, 2] = 0.1  # a corner the coarse mesh already had stays where it is
        coarse = np.delete(co, 4, axis=0)
        out = snap_new_vertices(MeshData(co, faces), coarse, _flat)
        self.assertEqual(out.vertices[4, 2], 0.0)
        self.assertEqual(out.vertices[0, 2], 0.1)

    def test_rows_open_a_fold_and_drop_debris(self):
        co, faces = _grid(4)
        co[12] = co[12] + [1.6, 1.6, 0.0]  # pushed past its neighbours: folds the quads around it
        loose = len(co)
        co = np.vstack((co, np.array([[9.0, 9.0, 0.0]] * 4)))  # a zero-size quad nothing else touches
        faces = faces + [(loose, loose + 1, loose + 2, loose + 3)]
        self.assertLess(quad_quality(co[np.asarray(faces)]).min(), 0.0)
        moved, kept = snap_rows(co, faces, _flat)
        self.assertEqual(len(kept), len(faces) - 1)
        self.assertEqual(len(moved), loose)
        self.assertGreater(quad_quality(moved[np.asarray(kept)]).min(), 0.0)


if __name__ == "__main__":
    unittest.main()
