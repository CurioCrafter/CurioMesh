# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import organize as O  # noqa: E402
from triadq.types import drain  # noqa: E402


def ball(n: int, jitter: float = 0.0):
    """A cube of n x n quads a side blown up to the unit sphere, its vertices shaken along the surface."""
    points, index, faces = [], {}, []

    def at(p):
        key = tuple(np.round(p, 6))
        if key not in index:
            index[key] = len(points)
            points.append(p)
        return index[key]

    for axis in range(3):
        for sign in (-1.0, 1.0):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for i in range(n):
                for j in range(n):
                    corners = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis], p[u], p[v] = sign, -1 + 2 * (i + di) / n, -1 + 2 * (j + dj) / n
                        corners.append(at(p))
                    faces.append(tuple(corners if sign > 0 else corners[::-1]))
    co = np.asarray(points, dtype=np.float64)
    co /= np.linalg.norm(co, axis=1, keepdims=True)
    rng = np.random.default_rng(1)
    co += jitter * rng.normal(size=co.shape) / n
    return co / np.linalg.norm(co, axis=1, keepdims=True), faces


def skew(co, faces) -> float:
    q = np.asarray(faces)
    p = co[q]
    out, back = np.roll(p, -1, axis=1) - p, np.roll(p, 1, axis=1) - p
    cos = (out * back).sum(axis=2) / (np.linalg.norm(out, axis=2) * np.linalg.norm(back, axis=2))
    return float(np.abs(np.degrees(np.arccos(np.clip(cos, -1, 1))) - 90.0).mean())


def onto_ball(points):
    return points / np.linalg.norm(points, axis=1, keepdims=True)


class OrganizeTest(unittest.TestCase):
    def test_settle_evens_quads_and_stays_on_the_surface(self):
        co, faces = ball(8, jitter=0.25)
        for method in ("settle", "smooth"):
            last = drain(O.settle_steps(co, faces, onto_ball, method=method))[-1][1]
            self.assertLess(skew(last, faces), 0.7 * skew(co, faces), method)
            self.assertLess(float(np.abs(np.linalg.norm(last, axis=1) - 1.0).max()), 1e-9)

    def test_snapshots_and_pins(self):
        co, faces = ball(6, jitter=0.2)
        stay = np.zeros(len(co), dtype=bool)
        stay[:5] = True
        shots = drain(O.settle_steps(co, faces, onto_ball, stay=stay, snapshots=(10, 20)))
        self.assertEqual([s for s, _p in shots], [10, 20, O.SETTLE_STEPS])
        self.assertTrue(np.array_equal(shots[-1][1][:5], co[:5]))

    def test_mirror_plane_vertices_stay_in_it(self):
        co, faces = ball(6)
        half, kept, _ids = O.half_of(co, faces, 0, 0.0, 1)
        on = O.on_mirror(half, 0, 0.0)
        self.assertTrue(on.any())
        self.assertFalse(O.pinned(half, kept, plane=(0, 0.0))[on].all())
        last = drain(O.settle_steps(half, kept, None, plane=(0, 0.0)))[-1][1]
        self.assertTrue(np.all(last[on, 0] == 0.0))

    def test_clean_poles_keeps_keys_and_pins(self):
        co, faces = ball(5, jitter=0.1)
        keys = ["top" if co[list(f)].mean(axis=0)[2] > 0 else "low" for f in faces]
        stay = np.zeros(len(co), dtype=bool)
        stay[0] = True
        new_co, new_faces, new_keys, held = drain(O.clean_pole_steps(co, faces, keys, stay))
        self.assertEqual(len(new_faces), len(new_keys))
        self.assertLessEqual(O.poles(new_co, new_faces), O.poles(co, faces))
        self.assertEqual(int(held.sum()), 1)
        self.assertEqual(set(new_keys), {"top", "low"})

    def test_pick_and_meter(self):
        levels = {0.0: {"view": 5.0}, 0.5: {"view": 5.5}, 1.0: {"view": 7.0}}
        self.assertEqual(O.pick(levels, 1.0), 1.0)
        self.assertEqual(O.pick(levels, 1.0, cap=1.0), 0.5)
        self.assertEqual(O.pick(levels, 0.4), 0.0)
        text = O.meter({"tidy": 38, "view": 6.2, "poles": 412}, {"tidy": 71, "view": 6.9, "poles": 240})
        self.assertEqual(text, "Tidiness 38 -> 71, view changed 6.2% -> 6.9%, poles 412 -> 240")
        self.assertTrue(O.meter({"tidy": 1, "poles": 2, "faces": 1000}, {"tidy": 2, "poles": 1, "faces": 950}).endswith("faces 1,000 -> 950"))


if __name__ == "__main__":
    unittest.main()
