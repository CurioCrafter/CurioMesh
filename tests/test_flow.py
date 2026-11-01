# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import numpy as np

from triadq.flow import faces_that_gain, snap_new_vertices, snap_rows, split_faces, unfold
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

    def test_split_marked_stays_all_quads_without_t_junctions(self):
        co, faces = _grid(5)
        out = split_faces(MeshData(co, faces), {12, 13, 7})
        quads = np.asarray(out.faces)
        self.assertEqual(quads.shape[1], 4)
        self.assertGreater(quad_quality(out.vertices[quads]).min(), 0.0)
        area = sum(np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[1])) / 2.0 for p in out.vertices[quads])
        self.assertAlmostEqual(area, 25.0)
        # every side shared by two faces or on the outer border: a T-junction would leave a side used once inside
        edges = {}
        for f in out.faces:
            for k in range(4):
                key = tuple(sorted((f[k], f[(k + 1) % 4])))
                edges[key] = edges.get(key, 0) + 1
        inner_once = [e for e, n in edges.items() if n == 1 and not all(_on_border(out.vertices[v], 5) for v in e)]
        self.assertEqual(inner_once, [])

    def test_only_faces_off_the_source_gain(self):
        co, faces = _grid(2)
        co[4, 2] = 0.5  # the middle rises, so the middles of the sides reaching it leave the flat source
        co[0, 2] = 0.0
        found = faces_that_gain(MeshData(co, faces), _flat, 0.05)
        self.assertEqual(found, {0, 1, 2, 3})
        self.assertEqual(faces_that_gain(MeshData(co, faces), _flat, 2.0), set())

    def test_stuck_fold_closes_along_its_short_diagonal(self):
        co, faces = _grid(4)
        a, b, c, d = faces[5]
        co[[a, c]] = co[[c, a]]  # a crossed quad
        points, kept, keep = unfold(co, faces, _flat, rounds=0)
        self.assertLess(len(kept), len(faces))
        self.assertEqual(len(keep), len(kept))
        self.assertGreater(quad_quality(points[np.asarray(kept)]).min(), 0.0)


def _on_border(point, n):
    return point[0] in (0.0, n) or point[1] in (0.0, n)


if __name__ == "__main__":
    unittest.main()
