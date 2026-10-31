# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import areas, patch  # noqa: E402


def _sheet(n=40):
    vertices = np.array([(x / n, y / n, 0.0) for y in range(n + 1) for x in range(n + 1)])
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    return vertices, faces


def _jobs(cent, items):
    groups = areas.group_areas(items)
    return [areas.Job(m, items[g[0]].target, items[g[0]].strategy, g) for g, m in zip(groups, areas.area_masks(cent, items, groups))]


class AreaTests(unittest.TestCase):
    def test_same_settings_merge_and_newest_owns_overlap(self):
        a = areas.Area(np.array([0.0, 0, 0]), 1.0, 0.1)
        b = areas.Area(np.array([1.5, 0, 0]), 1.0, 0.1)
        c = areas.Area(np.array([1.5, 0, 0]), 0.5, 0.2)
        far = areas.Area(np.array([9.0, 0, 0]), 1.0, 0.1)
        groups = areas.group_areas([a, b, c, far])
        self.assertEqual(groups, [[0, 1], [2], [3]])
        masks = areas.area_masks(np.array([[1.5, 0, 0], [0.0, 0, 0]]), [a, b, c, far], groups)
        self.assertEqual([m.tolist() for m in masks], [[False, True], [True, False], [False, False]])

    def test_each_area_gets_its_density_and_only_changes_redo(self):
        vertices, faces = _sheet()
        cent = vertices[np.asarray(faces)].mean(axis=1)
        dense = areas.Area(np.array([0.25, 0.25, 0.0]), 0.18, 0.015)
        coarse = areas.Area(np.array([0.75, 0.75, 0.0]), 0.18, 0.06)
        runner = areas.AreaRunner(vertices, faces)
        self.addCleanup(runner.stop)
        runner.start(_jobs(cent, [dense, coarse]))
        self.assertTrue(runner.wait(120))
        first = {r.quads: r for r in runner.output}
        self.assertEqual(len(first), 2)
        self.assertGreater(max(first), 2 * min(first))
        self.assertFalse(np.intersect1d(*[r.removed_faces for r in runner.output]).size)
        merged, merged_faces, _new = patch.apply_patches(vertices, faces, runner.output)
        self.assertGreater(len(merged), 0)
        edges = {}
        for face in merged_faces:
            for u, v in zip(face, face[1:] + face[:1]):
                edges[(min(u, v), max(u, v))] = edges.get((min(u, v), max(u, v)), 0) + 1
        self.assertLessEqual(max(edges.values()), 2)
        self.assertEqual(sum(c == 1 for c in edges.values()), 160)  # only the sheet's own border is open

        kept = next(r for r in runner.output if r.quads == min(first))
        dense.target = 0.03
        runner.start(_jobs(cent, [dense, coarse]))
        self.assertTrue(runner.wait(120))
        self.assertIs(next(r for r in runner.output if r.removed_faces.tobytes() == kept.removed_faces.tobytes()), kept)
        self.assertEqual(len(runner.cache), 3)

    def test_a_new_plan_replaces_a_running_one(self):
        vertices, faces = _sheet()
        cent = vertices[np.asarray(faces)].mean(axis=1)
        runner = areas.AreaRunner(vertices, faces)
        self.addCleanup(runner.stop)
        runner.start(_jobs(cent, [areas.Area(np.array([0.5, 0.5, 0.0]), 0.3, 0.01)]))
        runner.start([])
        self.assertTrue(runner.wait(120))
        self.assertEqual(runner.output, [])


if __name__ == "__main__":
    unittest.main()
