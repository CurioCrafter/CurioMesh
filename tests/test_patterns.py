# SPDX-License-Identifier: GPL-3.0-or-later
"""Topology score, grid refill and the small quad moves."""

import json
import pathlib
import random
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triadq import patterns  # noqa: E402
from triadq.refill import polish, refill  # noqa: E402
from triadq.tidy import tidy  # noqa: E402
from triadq.topology import topology_summary  # noqa: E402
from triadq.types import MeshData  # noqa: E402


def grid(n: int):
    points = [(float(x), float(y), 0.0) for y in range(n + 1) for x in range(n + 1)]
    at = lambda x, y: y * (n + 1) + x  # noqa: E731
    return np.asarray(points), [(at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)) for y in range(n) for x in range(n)]


def turned(n: int, seed: int, count: int):
    """A grid with some edges rotated, which scatters poles the way a pairing does."""

    points, faces = grid(n)
    faces = [list(f) for f in faces]
    rng = random.Random(seed)
    done = 0
    while done < count:
        k = rng.randrange(len(faces))
        if k % n == n - 1:
            continue
        a, b = faces[k], faces[k + 1]
        if a[1] != b[0] or a[2] != b[3]:
            continue
        faces[k] = [a[3], a[0], b[0], b[1]]
        faces[k + 1] = [b[1], b[2], b[3], a[3]]
        done += 1
    return points, [tuple(f) for f in faces]


class ScoreTest(unittest.TestCase):
    def test_database_holds_no_geometry(self):
        database = patterns.load_database()
        text = json.dumps(database)
        self.assertLess(len(text), 200_000)
        self.assertNotIn("vertices", text)
        self.assertTrue(database["sources"])
        self.assertFalse([s for s in database["sources"] if "Poly Haven" in s["name"]])

    def test_clean_grid_beats_scattered_poles(self):
        clean = patterns.score(patterns.measure(*grid(20)))["score"]
        messy = patterns.score(patterns.measure(*turned(20, 1, 60)))["score"]
        self.assertGreater(clean, 90.0)
        self.assertLess(messy, clean - 10.0)

    def test_loops_run_across_a_clean_grid(self):
        row = patterns.measure(*grid(10))
        self.assertEqual(row["loop_median"], 10)
        self.assertEqual(row["short_loops"], 0.0)
        self.assertEqual(row["interior_poles"], 0.0)


class FixTest(unittest.TestCase):
    def test_moves_never_add_poles(self):
        points, faces = turned(8, 3, 6)
        before = patterns.measure(points, faces)["interior_poles"]
        out, _counts = tidy(MeshData(points, faces))
        after = patterns.measure(out.vertices, out.faces)["interior_poles"]
        self.assertLessEqual(after, before)
        self.assertEqual(topology_summary(out).non_manifold_edges, 0)
        self.assertTrue(all(len(f) == 4 for f in out.faces))

    def test_refill_keeps_the_border_and_the_quads(self):
        points, faces = turned(12, 5, 24)
        mesh = MeshData(points, faces)
        out, counts = polish(mesh)
        self.assertGreater(counts["patches"], 0)
        self.assertTrue(all(len(f) == 4 for f in out.faces))
        self.assertEqual(topology_summary(out).non_manifold_edges, 0)
        area = sum(0.5 * np.linalg.norm(np.cross(out.vertices[f[2]] - out.vertices[f[0]], out.vertices[f[3]] - out.vertices[f[1]])) for f in out.faces)
        self.assertAlmostEqual(area, 144.0, places=6)  # still the same flat sheet, no folds or holes
        self.assertLess(patterns.measure(out.vertices, out.faces)["interior_poles"], patterns.measure(points, faces)["interior_poles"])

    def test_creases_stop_a_patch(self):
        points, faces = turned(12, 7, 24)
        points = points.copy()
        points[:, 2] = np.where(points[:, 0] > 6.0, (points[:, 0] - 6.0) * 2.0, 0.0)  # a sharp bend down the middle
        out, _counts = refill(MeshData(points, faces))
        on_crease = np.isclose(points[:, 0], 6.0).sum()
        self.assertEqual(int(np.isclose(out.vertices[:, 0], 6.0).sum()) >= on_crease - 1, True)


if __name__ == "__main__":
    unittest.main()
