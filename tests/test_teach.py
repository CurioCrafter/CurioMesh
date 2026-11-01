# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teach  # noqa: E402


def _grid(n=6):
    co = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=float)
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    return co, faces


def _fixed(co, faces, centre=(3, 3)):
    """The 2 x 2 block round a vertex redone as three quads (the vertex goes): what an artist might do by hand."""
    n = 6
    c = centre[1] * (n + 1) + centre[0]
    ring = [c - n - 2, c - n - 1, c - n, c + 1, c + n + 2, c + n + 1, c + n, c - 1]  # counter-clockwise from the lower left
    block = [i for i, f in enumerate(faces) if c in f]
    keep = [f for i, f in enumerate(faces) if i not in block]
    r = ring
    return co, keep + [(r[0], r[1], r[2], r[3]), (r[3], r[4], r[5], r[6]), (r[3], r[6], r[7], r[0])]


class TeachTests(unittest.TestCase):
    def test_a_fix_is_found_again_and_put_in(self):
        co, faces = _grid()
        co_b, faces_b = _fixed(co, faces)
        gone, new = teach.changed(co, faces, co_b, faces_b, 6.0)
        fix = teach.make_fix(co, faces, gone, co_b, faces_b, new, 6.0)
        self.assertIsInstance(fix, dict, fix)
        self.assertEqual((len(fix["before"]), len(fix["after"])), (4, 3))
        found = teach.find(fix, co, faces)
        self.assertEqual(len(found), 25)  # every vertex with a full 2 x 2 block round it
        out_co, out_faces, _kept = teach.put_in(co, faces, fix, found[0])
        self.assertEqual((len(out_faces), len(out_co)), (len(faces) - 1, len(co) - 1))
        use = Counter((f[k], f[(k + 1) % 4]) for f in out_faces for k in range(4))
        self.assertTrue(max(use.values()) == 1 and all(n >= 1 for n in use.values()), "the result is not a clean quad mesh")
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "fixes.json")
            teach.save(path, [fix])
            self.assertEqual(teach.load(path)[0]["id"], fix["id"])


if __name__ == "__main__":
    unittest.main()
