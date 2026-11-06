# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_flowstroke import grid  # noqa: E402
from test_organize import ball  # noqa: E402
from triadq import rings as RG  # noqa: E402
from triadq.patterns import Topology  # noqa: E402


def edge_uses(faces):
    uses = {}
    for f in faces:
        for k in range(len(f)):
            key = tuple(sorted((f[k], f[k - 1])))
            uses[key] = uses.get(key, 0) + 1
    return uses


def checked(test, co, faces, out):
    """All quads, the same open edges as before, no edge used more than twice, new faces wound like the old ones."""
    test.assertEqual({len(f) for f in out["faces"]}, {4})
    before, after = edge_uses(faces), edge_uses(out["faces"])
    test.assertLessEqual(max(after.values()), 2)
    test.assertEqual(sum(u == 1 for u in after.values()), sum(u == 1 for u in before.values()))
    return Topology(out["co"], out["faces"])


class RingsTest(unittest.TestCase):
    def sheet(self):
        co, faces = grid(24, wobble=0.15, seed=3)
        r = np.hypot(co[:, 0] - 12, (co[:, 1] - 12) * 1.6)
        co[:, 2] = 0.8 * np.exp(-(r / 2.5) ** 2)  # a bump to ring round
        return co, faces

    def test_rings_round_a_bump_on_a_sheet(self):
        co, faces = self.sheet()
        out = RG.build(co, faces, (12, 12, 0.8), (0, 0, 1), (1, 0, 0), 2.5, 1.5)
        topo = checked(self, co, faces, out)
        self.assertTrue(2 <= out["rings"] <= 4)
        a, b = out["grid"]
        self.assertEqual(out["count"], 2 * (a + b))
        self.assertLessEqual(out["corners"], RG.FANS)
        self.assertEqual(len(out["new"]), a * b + out["count"] * (out["rings"] - 1) + len(out["band"]))
        made = np.arange(len(out["co"]) - out["made"], len(out["co"]))
        self.assertEqual(int(topo.poles[made].sum()), 4 + out["corners"])  # the grid's corners and the band's
        rings = Topology(out["co"], out["faces"])
        self.assertEqual(rings.non_manifold_edges, 0)

    def test_faces_past_the_band_stay(self):
        co, faces = self.sheet()
        out = RG.build(co, faces, (12, 12, 0.8), (0, 0, 1), (1, 0, 0), 2.0, 1.2, band=1.5)
        reach = 2.0 + 2.5 * out["size"] + 1.5  # a quad more: a rim point may move to open a fold beside it
        far = [f for f in faces if np.hypot(*(co[list(f)].mean(axis=0)[:2] - 12)) > reach]
        kept = {tuple(np.round(out["co"][list(f)], 9).ravel()) for f in out["faces"]}
        self.assertTrue(all(tuple(np.round(co[list(f)], 9).ravel()) in kept for f in far))
        self.assertTrue(np.allclose(out["co"][:, 2].max(), co[:, 2].max(), atol=0.2))

    def test_watertight_on_a_ball_and_every_point_projected(self):
        co, faces = ball(10)
        sphere = lambda p: p / np.linalg.norm(p, axis=1, keepdims=True)  # noqa: E731
        centre = sphere(np.array([[0.3, 0.2, 1.0]]))[0]
        out = RG.build(co, faces, centre, centre, (1, 0, 0), 0.25, 0.18, project=sphere)
        checked(self, co, faces, out)
        p = out["co"]
        self.assertTrue(all(np.cross(p[f[2]] - p[f[0]], p[f[3]] - p[f[1]]) @ p[list(f)].mean(axis=0) > 0 for f in out["faces"]))
        self.assertTrue(all(u == 2 for u in edge_uses(out["faces"]).values()))
        self.assertTrue(np.allclose(np.linalg.norm(out["co"], axis=1), 1.0))
        self.assertEqual(len(out["co"]) - len(out["faces"]), len(co) - len(faces))  # still a sphere: V - F = 2 for quads

    def test_too_narrow_a_band_is_refused(self):
        co, faces = self.sheet()
        with self.assertRaises(ValueError):
            RG.build(co, faces, (12, 12, 0.8), (0, 0, 1), (1, 0, 0), 2.0, 1.2, band=0.1)

    def test_find_rim_reads_a_wide_bump(self):
        co, faces = grid(60)
        co = co * 0.25  # quads 0.25 wide: a source finer than the result
        x, y = co[:, 0] - 7.5, co[:, 1] - 7.5
        co[:, 2] = 0.6 * np.exp(-((x / 1.0) ** 2 + (y / 0.5) ** 2))
        tris = np.asarray([t for f in faces for t in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))])
        rx, ry = RG.find_rim(co, tris, (7.5, 7.5, 0.6), (0, 0, 1), (1, 0, 0), 0.6)
        self.assertGreater(rx, ry)
        self.assertTrue(0.6 < rx < 1.6 and 0.3 < ry < 0.9, (rx, ry))
        self.assertIsNone(RG.find_rim(co * (1, 1, 0), tris, (7.5, 7.5, 0.0), (0, 0, 1), (1, 0, 0), 0.6))

    def test_stitch_puts_corner_quads_where_the_border_turns(self):
        theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        square = np.concatenate([np.stack((np.linspace(-1, 1, 3, endpoint=False), np.full(3, -1.0)), 1) @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) for a in np.arange(4) * np.pi / 2])
        phi = np.arctan2(square[:, 1], square[:, 0])
        corner = np.where(np.arange(12) % 3 == 0, 0.0, 1.0)  # only the square's corners are cheap
        k0, kinds = RG._stitch(theta, phi, corner)
        self.assertEqual(kinds.count(1), 2)
        self.assertEqual(sum((1, 2, 0)[k] for k in kinds), 12)
        self.assertEqual(sum((1, 0, 2)[k] for k in kinds), 8)


if __name__ == "__main__":
    unittest.main()
