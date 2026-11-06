# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import issues  # noqa: E402


def grid(n=10):
    verts = [(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)]
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    return np.array(verts), faces


class IssuesTest(unittest.TestCase):
    def test_spots_and_compare(self):
        verts, faces = grid(10)
        flags = {"inward": np.zeros(len(faces), bool), "sunk": np.zeros(len(faces), bool)}
        flags["inward"][[0, 1, 11]] = True  # one cluster in the corner
        flags["sunk"][99] = True  # a lone face at the far corner
        spots = issues.find_spots(verts, faces, flags, part_of_face=np.arange(len(faces)) % 2)
        self.assertEqual(len(spots), 2)
        self.assertEqual(spots[0]["kinds"], {"inward": 3})
        self.assertEqual(sorted(spots[0]["faces"]), [0, 1, 11])
        self.assertGreater(spots[0]["score"], spots[1]["score"])
        old = issues.report(spots, {})
        flags["inward"][:] = False  # the corner is fixed, a new one appears in the middle
        flags["inward"][45] = True
        new = issues.report(issues.find_spots(verts, faces, flags), {})
        diff = issues.compare(old, new)
        self.assertEqual((len(diff["fixed"]), len(diff["still"]), len(diff["new"])), (1, 1, 1))
        self.assertEqual(diff["totals"]["inward"], {"old": 3, "new": 1})

    def test_topology_flags(self):
        verts, faces = grid(4)
        verts[12] = (2.0, 2.0, 0.0)  # interior vertex unchanged: a regular grid has no poles, slivers or concave quads
        flat = issues.topology_flags(verts, faces)
        self.assertFalse(any(v.any() for v in flat.values()))
        verts[12] = (2.9, 2.9, 0.0)  # pulled into a corner: the quad on the far side goes concave or thin
        bent = issues.topology_flags(verts, faces)
        self.assertTrue(bent["sliver"].any() or bent["concave"].any() or bent["pinched"].any())


if __name__ == "__main__":
    unittest.main()
