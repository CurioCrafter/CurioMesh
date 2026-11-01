# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import guides  # noqa: E402
import loops  # noqa: E402


def _plane(n=30):
    co = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=float)
    tris = []
    for y in range(n):
        for x in range(n):
            a, b, c, d = y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x
            tris += [(a, b, c), (a, c, d)]
    return co, np.asarray(tris), np.zeros(len(tris), dtype=int)


def _nearest(p):
    return np.array([p[0], p[1], 0.0]), np.array([0.0, 0.0, 1.0])


class EdgeLoopTests(unittest.TestCase):
    def test_strips_go_in_clean_and_loops_are_even(self):
        co, tris, kind = _plane()
        t = np.linspace(0, 2 * np.pi, 40)
        ring = np.stack([15 + 6 * np.cos(t), 15 + 4 * np.sin(t), 0 * t], axis=1)
        line = np.stack([np.linspace(4, 26, 30), np.full(30, 5.0), np.zeros(30)], axis=1)
        out, tr, _kind, keys, built, skipped = loops.insert(co, tris, kind, [(ring, 3, "LOOP"), (line, 5, "BEND")], 1.0, _nearest)
        self.assertEqual((len(built), skipped), (2, []))
        use = Counter((a, b) for t3 in tr.tolist() for a, b in ((t3[0], t3[1]), (t3[1], t3[2]), (t3[2], t3[0])))
        self.assertFalse([e for e, n in use.items() if n > 1], "a directed edge is used twice: triangles overlap")
        self.assertTrue(keys and all(a < len(out) and b < len(out) for a, b in keys))
        rows = built[1][1]
        gaps = [guides._to_line(rows[k][5:6], rows[k - 1])[0] for k in range(1, len(rows))]
        self.assertTrue(np.allclose(gaps, 1.0, rtol=0.05), f"the loops of a line are not evenly spaced: {gaps}")

    def test_region_goes_back_beside_the_rest(self):
        co, tris, _kind = _plane(8)
        faces = [tuple(t) for t in tris.tolist()]
        mask = np.array([co[list(f)].mean(axis=0)[0] < 4 for f in faces])
        mine, rest = loops.split_region(co, faces, np.zeros(len(faces), dtype=int), mask)
        joined, jf, _k, new, seam = loops.join_outside(mine[0], mine[1], [str(m) for m in mine[2]], rest)
        self.assertEqual((len(jf), int(new.sum()), seam[0] > 0), (len(faces), int(mask.sum()), True))
        self.assertEqual(len(joined), len(co))


if __name__ == "__main__":
    unittest.main()
