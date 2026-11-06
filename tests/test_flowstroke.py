# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _addon  # noqa: E402

F = _addon.load("flowstroke")
from triadq import loopscore as L  # noqa: E402


def _arc(degrees, rx=1.0, ry=1.0, start=0.0, n=120, wobble=0.0, seed=0, tilt=0.0):
    rng = np.random.default_rng(seed)
    a = np.radians(start + np.linspace(0.0, degrees, n))
    r = 1.0 + wobble * np.convolve(rng.normal(size=n + 8), np.ones(9) / 3.0, mode="valid")[:n]
    p = np.stack((rx * r * np.cos(a), ry * r * np.sin(a), np.zeros(n)), axis=1)
    if tilt:
        c, s = np.cos(tilt), np.sin(tilt)
        p = p @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]).T
    return p


def _zigzag(n=9, amp=0.15, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2, n)
    y = amp * (-1) ** np.arange(n) * (1 + 0.2 * rng.random(n))
    return np.stack((x, y, np.zeros(n)), axis=1)


def _figure_eight(n=200, seed=0):
    t = np.linspace(0, 2 * np.pi, n)
    rng = np.random.default_rng(seed)
    return np.stack((np.sin(t), np.sin(t) * np.cos(t), 0.01 * rng.normal(size=n)), axis=1)


def _ridge(points):
    """A roof ridge along x at y = 0: on it within 0.05, running along x."""
    on = np.abs(points[:, 1]) < 0.05
    return on, np.tile((1.0, 0.0, 0.0), (len(points), 1))


def strokes():
    """(name, points, crease, expected kind)"""
    out = []
    for seed in range(3):
        out.append((f"circle {seed}", _arc(360, seed=seed, start=37 * seed), None, "RING"))
        out.append((f"wobbly circle {seed}", _arc(360, wobble=0.08, seed=seed), None, "RING"))
        out.append((f"very wobbly circle {seed}", _arc(355, wobble=0.15, seed=seed), None, "RING"))
        out.append((f"overshoot circle {seed}", _arc(385, wobble=0.05, seed=seed), None, "RING"))
        out.append((f"ellipse 2:1 {seed}", _arc(360, 1.0, 0.5, seed=seed, wobble=0.04), None, "RING"))
        out.append((f"ellipse 3:1 tilted {seed}", _arc(360, 1.5, 0.5, seed=seed, tilt=0.6), None, "RING"))
        for gap in (5, 10, 15):
            out.append((f"gap {gap}% {seed}", _arc(360 * (1 - gap / 100), wobble=0.04, seed=seed), None, "RING"))
        for deg in (60, 120, 180, 240):
            out.append((f"open arc {deg} {seed}", _arc(deg, wobble=0.03, seed=seed), None, "FLOW"))
        out.append((f"zig-zag {seed}", _zigzag(seed=seed), None, "FLOW"))
        out.append((f"figure-eight {seed}", _figure_eight(seed=seed), None, "FLOW"))
        rng = np.random.default_rng(seed)
        x = np.linspace(-1, 1, 60)
        along = np.stack((x, 0.01 * rng.normal(size=60), np.zeros(60)), axis=1)
        out.append((f"along crease {seed}", along, _ridge, "EDGE"))
        across = np.stack((0.01 * rng.normal(size=60), x, np.zeros(60)), axis=1)
        out.append((f"across crease {seed}", across, _ridge, "FLOW"))
        out.append((f"straight line {seed}", np.stack((x, 0.3 + 0.02 * rng.normal(size=60), np.zeros(60)), axis=1), _ridge, "FLOW"))
        out.append((f"ring on a crease mesh {seed}", _arc(360, 0.5, 0.5, wobble=0.05, seed=seed) + (0, 0.7, 0), _ridge, "RING"))
    return out


class RecogniseTest(unittest.TestCase):
    def test_accuracy_on_synthetic_strokes(self):
        rows = strokes()
        wrong = [(name, want, F.recognise(p, crease=c)) for name, p, c, want in rows if F.recognise(p, crease=c) != want]
        accuracy = 1.0 - len(wrong) / len(rows)
        print(f"\nrecogniser: {len(rows) - len(wrong)}/{len(rows)} right ({100 * accuracy:.0f}%)", wrong)
        self.assertGreaterEqual(accuracy, 0.95, wrong)

    def test_fit_ring_is_closed_and_smooth(self):
        p = _arc(350, 1.0, 0.6, wobble=0.1, seed=4)
        ring = F.fit_ring(p, 48)
        self.assertEqual(len(ring), 49)
        self.assertTrue(np.allclose(ring[0], ring[-1]))
        self.assertEqual(F.recognise(ring), "RING")
        r = np.linalg.norm(ring[:, :2] / (1.0, 0.6), axis=1)
        self.assertLess(float(r.std()), 0.1)

    def test_shapes(self):
        e = F.ellipse((0, 0, 0), (0, 0, 1), (1, 0, 0), 2.0, 1.0, 32)
        self.assertAlmostEqual(float(np.ptp(e[:, 0])), 4.0, places=6)
        self.assertEqual(F.recognise(e), "RING")
        eye = F.template("EYE", (0, 0, 0), (0, 0, 1), (1, 0, 0), 1.0)
        self.assertEqual([k for k, _l in eye], ["RING"])
        nose = F.template("NOSE_MOUTH", (0, 0, 0), (0, 0, 1), (1, 0, 0), 1.0)
        self.assertEqual(len(nose), 2)
        self.assertTrue(np.allclose(nose[0][1][:, 0], -nose[1][1][:, 0]))  # mirrored pair
        rec = F.record("RING", e)
        self.assertTrue(rec["closed"])
        self.assertTrue(np.allclose(np.linalg.norm(rec["directions"], axis=1), 1.0))


def grid(n, wobble=0.0, seed=0):
    rng = np.random.default_rng(seed)
    co = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
    inner = (co[:, 0] > 0) & (co[:, 0] < n) & (co[:, 1] > 0) & (co[:, 1] < n)
    co[inner, :2] += wobble * rng.normal(size=(int(inner.sum()), 2))
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    return co, faces


class RedoTest(unittest.TestCase):
    def test_ring_on_a_sheet_makes_a_closed_loop_with_clean_seams(self):
        co, faces = grid(24)
        ring = F.ellipse((12, 12, 0), (0, 0, 1), (1, 0, 0), 4.0, 2.6, 48)
        found = F.areas(co, faces, [F.record("RING", ring)])
        self.assertEqual(len(found), 1)
        ids, inside = found[0]
        flat = lambda p: np.asarray(p) * (1, 1, 0)  # noqa: E731
        made = F.redo(co, faces, ids, inside, flat)
        co2, faces2, _mat, new = F.splice(co, faces, np.zeros(len(faces), dtype=np.int32), [(ids, made)])
        self.assertTrue(all(len(f) == 4 for f in faces2))
        from triadq import segment as S

        self.assertEqual(S.open_edges(faces2), S.open_edges(faces))  # no new holes: the seam closed
        loops, closed, _label = L.walk(L.Topology(co2, faces2))
        rings = [vs for vs, c in zip(loops, closed) if c and len(vs) >= 12]
        self.assertTrue(rings, "a closed edge loop round the drawn ring")


if __name__ == "__main__":
    unittest.main()
