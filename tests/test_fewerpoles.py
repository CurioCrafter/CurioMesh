# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.chords import ChordReducer  # noqa: E402
from triadq.fewerpoles import PoleCleaner  # noqa: E402
from triadq.types import MeshData, drain  # noqa: E402


def energy(r) -> int:
    return sum(r._faces_at_cost(v, len(u)) for v, u in enumerate(r.users) if u)


def split_grid(n: int = 6):
    """A flat n x n grid whose middle corner is split in two along the x axis: a quad with 3s on one diagonal and 5s on
    the other, the pair a diagonal collapse takes away. Returns (mesh, the two 3s)."""
    at = {(i, j): i * (n + 1) + j for i in range(n + 1) for j in range(n + 1)}
    co = [(float(i), float(j), 0.0) for i in range(n + 1) for j in range(n + 1)]
    faces = [[at[i, j], at[i + 1, j], at[i + 1, j + 1], at[i, j + 1]] for i in range(n) for j in range(n)]
    m = n // 2
    v = at[m, m]
    c = len(co)
    co[v] = (m, m + 0.3, 0.0)  # v stays above as `a`, `c` goes below
    co.append((m, m - 0.3, 0.0))
    for f in faces:
        if v in f and sum(co[u][1] for u in f) / 4 < m:
            f[f.index(v)] = c
    faces.append([at[m - 1, m], c, at[m + 1, m], v])
    return MeshData(np.asarray(co), [tuple(f) for f in faces], ["0"] * len(faces)), (v, c)


class PoleCleanerTests(unittest.TestCase):
    def test_a_three_three_pair_is_collapsed_away(self):
        mesh, _pair = split_grid()
        reducer = ChordReducer(mesh)
        self.assertEqual(energy(reducer), 4)
        drain(PoleCleaner(reducer).run_steps())
        self.assertEqual(energy(reducer), 0)
        self.assertEqual(reducer.alive, len(mesh.faces) - 1)
        result = reducer.result()
        self.assertTrue(all(len(f) == 4 for f in result.faces))

    def test_locked_corners_hold(self):
        mesh, pair = split_grid()
        reducer = ChordReducer(mesh)
        for v in reducer._around(pair[0]) | reducer._around(pair[1]) | set(pair):
            reducer.locked[v] = True
        before = [list(f) for f in reducer.faces]
        drain(PoleCleaner(reducer).run_steps())
        self.assertEqual([list(f) for f in reducer.faces], before)


if __name__ == "__main__":
    unittest.main()
