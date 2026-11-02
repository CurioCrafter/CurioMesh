# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.flatquad import _close_pieces  # noqa: E402
from triadq.quadreduce import QuadReducer, join_loose_triangles  # noqa: E402
from triadq.topology import topology_summary  # noqa: E402
from triadq.types import MeshData  # noqa: E402


def box(cells: int) -> MeshData:
    index, points, faces = {}, [], []

    def vertex(p):
        return index.setdefault(p, (len(points), points.append(p))[0])

    for axis in range(3):
        u, w = (axis + 1) % 3, (axis + 2) % 3
        for side in (0, 1):
            for i in range(cells):
                for j in range(cells):
                    corners = []
                    for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        p = [0.0, 0.0, 0.0]
                        p[axis], p[u], p[w] = float(side), a / cells, b / cells
                        corners.append(vertex(tuple(p)))
                    faces.append(tuple(corners if side else corners[::-1]))
    return MeshData(np.asarray(points), faces, ["0"] * len(faces))


class ReducerTests(unittest.TestCase):
    def test_a_box_loses_quads_but_stays_a_closed_all_quad_box(self):
        mesh = box(10)
        reducer = QuadReducer(mesh)
        for _ in reducer.remove(300):
            pass
        result = reducer.result()
        summary = topology_summary(result)
        self.assertLess(summary.faces, len(mesh.faces) - 150)
        self.assertTrue(summary.all_quads)
        self.assertEqual(summary.component_signature, topology_summary(mesh).component_signature)
        self.assertEqual(np.ptp(result.vertices, axis=0).tolist(), [1.0, 1.0, 1.0])

    def test_restore_goes_back_exactly(self):
        reducer = QuadReducer(box(6))
        saved = reducer.save()
        faces = reducer.result().faces
        for _ in reducer.remove(40):
            pass
        reducer.restore(saved)
        self.assertEqual(reducer.result().faces, faces)

    def test_border_corners_stay_put(self):
        size = 7
        points = [(float(i), float(j), 0.0) for i in range(size) for j in range(size)]
        faces = [(i * size + j, (i + 1) * size + j, (i + 1) * size + j + 1, i * size + j + 1) for i in range(size - 1) for j in range(size - 1)]
        reducer = QuadReducer(MeshData(np.asarray(points), faces, ["0"] * len(faces)))
        for _ in reducer.remove(20):
            pass
        result = reducer.result()
        self.assertLess(len(result.faces), len(faces))
        self.assertTrue(topology_summary(result).all_quads)
        rim = {p for p in points if p[0] in (0.0, size - 1.0) or p[1] in (0.0, size - 1.0)}
        self.assertTrue(rim <= {tuple(p) for p in result.vertices.tolist()})


class JoinTests(unittest.TestCase):
    def strip(self, quads: int) -> MeshData:
        points = [(-1.0, 0.5, 0.0)]
        for i in range(quads + 1):
            points += [(float(i), 0.0, 0.0), (float(i), 1.0, 0.0)]
        points.append((quads + 1.0, 0.5, 0.0))
        at = lambda i, top: 1 + 2 * i + top  # noqa: E731
        faces = [(0, at(0, 0), at(0, 1))]
        faces += [(at(i, 0), at(i + 1, 0), at(i + 1, 1), at(i, 1)) for i in range(quads)]
        faces.append((len(points) - 1, at(quads, 1), at(quads, 0)))
        return MeshData(np.asarray(points), faces, ["0"] * len(faces))

    def test_two_triangles_join_along_a_path_with_two_quads_per_quad(self):
        for quads, expected in ((1, 2), (2, 4), (4, 8)):
            result, joined = join_loose_triangles(self.strip(quads))
            self.assertEqual(joined, 2)
            summary = topology_summary(result)
            self.assertTrue(summary.all_quads)
            self.assertEqual(summary.faces, expected)

    def test_joining_never_puts_three_faces_on_an_edge(self):
        rng = np.random.default_rng(3)
        size = 14
        points = [(float(i), float(j), float(rng.random()) * 0.2) for i in range(size) for j in range(size)]
        faces = []
        for i in range(size - 1):
            for j in range(size - 1):
                a, b, c, d = i * size + j, (i + 1) * size + j, (i + 1) * size + j + 1, i * size + j + 1
                faces += [(a, b, c), (a, c, d)] if rng.random() < 0.35 else [(a, b, c, d)]
        result, joined = join_loose_triangles(MeshData(np.asarray(points), faces, ["0"] * len(faces)))
        self.assertGreater(joined, 0)
        self.assertEqual(topology_summary(result).non_manifold_edges, 0)

    def test_a_piece_with_a_bad_edge_still_comes_out_all_quads(self):
        points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0), (0.5, -1.0, 0.0), (0.5, 0.0, 1.0), (0.5, 0.0, -1.0)]
        faces = [(0, 1, 2), (1, 0, 3), (0, 1, 4), (1, 0, 5)]  # four faces on one edge
        closed = _close_pieces(MeshData(np.asarray(points), faces, ["0"] * 4))[0]
        self.assertTrue(all(len(face) == 4 for face in closed.faces))

    def test_a_lone_triangle_is_left_for_closure(self):
        mesh = MeshData(np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]), [(0, 1, 2)], ["0"])
        result, joined = join_loose_triangles(mesh)
        self.assertEqual(joined, 0)
        self.assertEqual(result.faces, mesh.faces)


if __name__ == "__main__":
    unittest.main()
