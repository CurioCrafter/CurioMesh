# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_flowstroke import grid  # noqa: E402
from test_organize import ball  # noqa: E402
from triadq import loopscore as L  # noqa: E402
from triadq import organize as O  # noqa: E402
from triadq import strips as S  # noqa: E402
from triadq.patterns import Topology  # noqa: E402


def lines_of(co, faces, strip, stay=None):
    stay = O.pinned(co, faces) if stay is None else stay
    return S.rung_lines(strip, L._continuations(Topology(co, faces)), len(co), stay), stay


class StripsTest(unittest.TestCase):
    def test_grid_and_ball_strips(self):
        co, faces = grid(6)
        st = S.Strips(co, faces)
        self.assertEqual(len(np.unique(st.labels)), 12)  # 6 rows and 6 columns of quads
        strip = st.walk(st.element(14, 0))
        self.assertEqual((len(strip["faces"]), strip["closed"], len(strip["a"])), (6, False, 7))
        self.assertIn(14, strip["faces"])
        co, faces = ball(4)
        st = S.Strips(co, faces)
        self.assertFalse(st.walk(0)["closed"])  # a corner quad: its strips stop at the cube's corner poles
        rings = [w for w in (st.walk(e) for e in range(2 * len(st.quads))) if w["closed"]]
        self.assertTrue(rings)
        self.assertEqual({len(w["faces"]) for w in rings}, {16})  # round the cube away from its corners: 4 sides of 4
        ends, labels = st.segments()
        self.assertEqual(len(ends), 2 * len(labels))

    def test_slide_straighten_even_stay_on_the_lines(self):
        co, faces = grid(10, wobble=0.15, seed=3)
        co[:, 2] = 0.0
        st = S.Strips(co, faces)
        strip = st.walk(st.element(45, 0))  # a column through the middle
        lines, stay = lines_of(co, faces, strip)
        before = S.stats(co, strip)
        for kind, amount, check in (("straighten", 1.0, "turn"), ("even", 1.0, "width")):
            new = S.finish(co, S.move(co, lines, S.targets(co, lines, kind, amount), stay), faces)
            self.assertLess(S.stats(new, strip)[check], 0.6 * before[check], kind)
            self.assertTrue(np.allclose(new[stay], co[stay]))
        slid = S.finish(co, S.move(co, lines, S.targets(co, lines, "slide", 0.5), stay), faces)
        moved = np.flatnonzero(np.linalg.norm(slid - co, axis=1) > 1e-9)
        self.assertTrue(set(strip["a"][1:-1]) <= set(moved.tolist()))
        self.assertFalse(stay[moved].any())

    def test_cut_and_collapse_keep_quads(self):
        co, faces = grid(6)
        st = S.Strips(co, faces)
        strip = st.walk(st.element(14, 0))
        co2, faces2, parent, (face, edge) = S.cut(co, faces, [strip])
        self.assertEqual(len(faces2), 42)
        self.assertTrue(all(len(f) == 4 for f in faces2))
        self.assertEqual(len(parent), 42)
        st2 = S.Strips(co2, faces2)
        self.assertEqual(len(np.unique(st2.labels)), 13)
        new = st2.walk(st2.element(face, edge))
        self.assertEqual(len(new["faces"]), 6)
        co3, faces3, parent3 = S.collapse(co2, faces2, [new], O.pinned(co2, faces2))
        self.assertEqual((len(faces3), len(co3)), (36, 49))
        self.assertTrue(all(len(f) == 4 for f in faces3))
        self.assertEqual(len(parent3), 36)

    def test_move_vertex_eases_its_neighbours(self):
        co, faces = grid(6)
        stay = O.pinned(co, faces)
        new = S.move_vertex(co, faces, 24, co[24] + (0.6, 0.0, 0.0), stay)
        self.assertTrue(np.allclose(new[24], (3.6, 3.0, 0.0)))
        self.assertGreater(new[25, 0], co[25, 0])  # pulled along
        self.assertTrue(np.allclose(new[stay], co[stay]))

    def test_mirror_partners(self):
        co, faces = grid(6)
        partner = S.partners(co, 0, 3.0)
        self.assertTrue((partner >= 0).all())
        self.assertTrue(np.allclose(co[partner, 0], 6.0 - co[:, 0]))
        st = S.Strips(co, faces)
        strip = st.walk(st.element(1, 0))  # the column x = 1..2
        other = st.walk(S.mirror_element(st, strip, partner))
        self.assertTrue(np.allclose(sorted(co[other["a"] + other["b"], 0].tolist())[0], 4.0))


if __name__ == "__main__":
    unittest.main()
