# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.chords import ChordReducer, PictureJudge, _closed_walk, _grid_cells  # noqa: E402
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


def turned_edge(mesh: MeshData) -> MeshData:
    """Spin one edge in the middle of a side: two quads share a different diagonal of their hexagon, which makes four poles."""

    faces = [list(f) for f in mesh.faces]
    middle = np.asarray(mesh.vertices).mean(axis=0)
    seen: dict[tuple[int, int], int] = {}
    for f, face in enumerate(faces):
        for a, b in zip(face, (*face[1:], face[0])):
            other = seen.get((b, a))
            seen[(a, b)] = f
            if other is None:
                continue
            spots = np.asarray(mesh.vertices)[[a, b]]
            if np.ptp(spots[:, 0]) == 0.0 and spots[0, 0] == 1.0 and np.linalg.norm(spots.mean(axis=0)[1:] - middle[1:]) < 0.25:
                one, two = faces[other], faces[f]  # other runs a -> b, f runs b -> a
                i, k = one.index(a), two.index(b)
                ring = [one[(i + 1) % 4], one[(i + 2) % 4], one[(i + 3) % 4], two[(k + 1) % 4], two[(k + 2) % 4], two[(k + 3) % 4]]
                faces[other], faces[f] = ring[1:5], [ring[4], ring[5], ring[0], ring[1]]
                return MeshData(mesh.vertices, [tuple(x) for x in faces], mesh.face_materials)
    raise AssertionError("no edge found")


class WalkTests(unittest.TestCase):
    def test_a_square_wish_closes_at_no_cost_and_encloses_a_grid(self):
        wish = []
        for k in range(8):
            wish.append((0.0, 9.0, 9.0) if k % 2 == 0 else (9.0, 0.0, 9.0))  # left at every other corner: a 2 by 2 square
        price, points, _turns = _closed_walk(wish, 100.0)
        self.assertEqual(price, 0.0)
        self.assertEqual(len(_grid_cells(points)), 4)

    def test_a_walk_that_cannot_close_under_the_limit_is_refused(self):
        self.assertIsNone(_closed_walk([(9.0, 0.0, 9.0)] * 8, 1.0))  # straight on for ever


def energy(reducer):
    return sum(reducer._faces_at_cost(v, len(u)) for v, u in enumerate(reducer.users) if u)


class RedrawTests(unittest.TestCase):
    def test_poles_go_and_the_mesh_stays_a_closed_all_quad_box(self):
        mesh = turned_edge(box(6))
        reducer = ChordReducer(mesh)
        before = energy(reducer)
        self.assertGreater(reducer.redraw_all(), 0)
        self.assertLess(energy(reducer), before)
        summary = topology_summary(reducer.result())
        self.assertEqual((summary.boundary_edges, summary.non_manifold_edges, summary.euler_characteristic), (0, 0, 2))
        self.assertEqual(summary.quads, summary.faces)


    def test_freezing_after_a_restore_skips_corners_that_are_gone(self):
        reducer = ChordReducer(turned_edge(box(6)))
        saved = reducer.save()
        known = len(reducer.users)
        reducer.redraw_all()
        self.assertGreater(len(reducer.users), known)
        reducer.restore(saved)
        reducer.freeze([0, known + 1], renamed=True)
        self.assertTrue(reducer.locked[0])


    def test_a_part_rolled_back_is_as_it_was_and_the_other_part_keeps_its_progress(self):
        one, two = turned_edge(box(6)), turned_edge(box(6))
        count = len(one.vertices)
        faces = list(one.faces) + [tuple(v + count for v in f) for f in two.faces]
        reducer = ChordReducer(MeshData(np.vstack((one.vertices, np.asarray(two.vertices) + 5.0)), faces, ["0"] * len(faces)))
        saved, messy = reducer.save(), energy(reducer)
        reducer.redraw_all()
        tidy = energy(reducer)
        self.assertLess(tidy, messy)
        reducer.roll_back_parts(saved, [0])
        self.assertTrue(tidy < energy(reducer) < messy)
        self.assertEqual(reducer.alive, sum(1 for f in reducer.faces if f is not None))
        self.assertEqual(topology_summary(reducer.result()).non_manifold_edges, 0)
        self.assertEqual(topology_summary(reducer.result()).boundary_edges, 0)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        size = 64
        hit = np.ones((size, size), dtype=bool)
        normal = np.zeros((size, size, 3))
        normal[:, :, 2] = 1.0
        depth = np.full((size, size), 2.0)  # a flat wall at z = 0 seen from +z, in a picture of half size 1
        empty = (np.zeros((size, size), dtype=bool), np.zeros((size, size, 3)), np.zeros((size, size)), np.zeros((size, size), dtype=np.int64))
        views = [empty, empty, empty, empty, (hit, normal, depth, np.zeros((size, size), dtype=np.int64)), empty]
        self.judge = PictureJudge(views, (0.0, 0.0, 0.0), 1.0, size, 20.0)

    def test_a_quad_on_the_wall_is_right_and_a_tilted_one_is_wrong(self):
        flat = ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0))
        tilted = ((-0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, -0.5))
        self.assertEqual(self.judge.wrong(flat), 0)
        self.assertGreater(self.judge.wrong(tilted), 100)


if __name__ == "__main__":
    unittest.main()
