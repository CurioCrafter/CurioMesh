# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _addon  # noqa: E402
from triadq import patterns, rowfill  # noqa: E402
from triadq import segment as S  # noqa: E402
from triadq.types import drain  # noqa: E402

F = _addon.load("flowstroke")


def _sheet(n=24, seed=0):
    """A gently curved sheet of quads with its points shaken about in the surface: messy rows, regular counts."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(n + 1, dtype=float), np.arange(n + 1, dtype=float), indexing="ij")
    p = np.stack((x, y, np.zeros_like(x)), axis=-1).reshape(-1, 3)
    p[:, :2] += rng.uniform(-0.3, 0.3, (len(p), 2)) * ((p[:, :2] > 0) & (p[:, :2] < n)).all(axis=1)[:, None]
    p[:, 2] = 0.02 * ((p[:, 0] - n / 2) ** 2)
    faces = [(i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1) for i in range(n) for j in range(n)]
    return p, faces


class RedrawInRowsTest(unittest.TestCase):
    def test_area_becomes_rows_tied_on(self):
        co, faces = _sheet()
        middle = np.asarray([co[list(f)].mean(axis=0) for f in faces])
        ids = np.flatnonzero(np.linalg.norm(middle[:, :2] - 12.0, axis=1) < 8.0)
        out_co, out_faces, _mat, new, area, folded = drain(F.rows_steps(co, faces, np.zeros(len(faces), dtype=np.int64), ids))
        self.assertTrue(all(len(f) == 4 for f in out_faces))
        self.assertEqual(S.open_edges(out_faces), S.open_edges(faces))  # tied on all round, the border's edges kept
        self.assertEqual(folded, 0)
        self.assertLess(abs(int(new.sum()) - len(area)), 0.3 * len(area))  # the same size of quads
        topo = patterns.Topology(out_co, out_faces)
        inner = np.unique([v for f, n in zip(out_faces, new) if n for v in f])
        old = {v for f, n in zip(out_faces, new) if not n for v in f}
        inner = np.asarray([v for v in inner if v not in old])
        self.assertLess(topo.poles[inner].mean(), 0.15)

    def test_rows_follow_the_long_side(self):
        co = np.array([[x, y, 0.0] for x in np.linspace(0, 4, 41) for y in np.linspace(0, 1, 11)])
        tris = []
        for i in range(40):
            for j in range(10):
                a, b, c, d = i * 11 + j, (i + 1) * 11 + j, (i + 1) * 11 + j + 1, i * 11 + j + 1
                tris += [(a, b, c), (a, c, d)]
        tris = np.asarray(tris)
        border = np.asarray(S.boundary_loops([tuple(t) for t in tris.tolist()], range(len(tris)))[0][0])
        uv = rowfill._flat(co, tris, border)
        theta = rowfill.direction(co, tris, uv)
        along = uv[border].max(axis=0) - uv[border].min(axis=0)
        long_axis = np.arctan2(along[1], along[0]) if along[1] > along[0] else 0.0
        self.assertLess(abs(np.sin(2.0 * (theta - long_axis))), 0.2)  # mod 90 degrees

    def test_tidy_area_keeps_one_clean_piece(self):
        _co, faces = _sheet(12)
        mask = np.zeros(len(faces), dtype=bool)
        mask[30:90] = True
        mask[140] = True  # a crumb far away
        out = rowfill.tidy_area(faces, mask)
        self.assertFalse(out[140])
        self.assertEqual(len(S.boundary_loops(faces, np.flatnonzero(out))[1]), 0)  # no corner where it touches itself


if __name__ == "__main__":
    unittest.main()
