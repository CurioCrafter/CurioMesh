# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
import unittest

import numpy as np

from triadq import hardtidy, trial


def _fold(n: int = 6, rows: int = 3):
    """Two flat grids meeting at a right-angle crease along the x axis: z = 0 for y >= 0, and y = 0 for z >= 0."""
    points, index = [], {}
    for x in range(n + 1):
        for y in range(rows + 1):
            index[(x, y, 0)] = len(points)
            points.append((x, y, 0.0))
        for z in range(1, rows + 1):
            index[(x, 0, z)] = len(points)
            points.append((x, 0.0, z))
    faces = []
    for x in range(n):
        for y in range(rows):
            faces.append((index[(x, y, 0)], index[(x + 1, y, 0)], index[(x + 1, y + 1, 0)], index[(x, y + 1, 0)]))
        for z in range(rows):
            a, b = index[(x, 0, z)], index[(x + 1, 0, z)]
            c, d = index[(x + 1, 0, z + 1)], index[(x, 0, z + 1)]
            faces.append((d, c, b, a))
    crease = [index[(x, 0, 0)] for x in range(n + 1)]
    return np.array(points, dtype=np.float64), faces, crease


def _fold_surface(points):
    """Nearest points on the fold."""
    p = np.asarray(points, dtype=np.float64)
    a = np.stack((p[:, 0], np.maximum(p[:, 1], 0.0), np.zeros(len(p))), axis=1)
    b = np.stack((p[:, 0], np.zeros(len(p)), np.maximum(p[:, 2], 0.0)), axis=1)
    use_a = np.linalg.norm(p - a, axis=1) <= np.linalg.norm(p - b, axis=1)
    return np.where(use_a[:, None], a, b)


def _flat(points):
    out = np.array(points, dtype=np.float64)
    out[:, 2] = 0.0
    return out


class TrueUpEdgesTest(unittest.TestCase):
    def test_wavy_crease_goes_back_on_the_line(self):
        points, faces, crease = _fold()
        inner = crease[1:-1]
        points[inner, 1] += np.array([0.15, -0.0, 0.12, 0.0, 0.18])  # off the crease, still on the flat side
        points[inner, 0] += np.array([0.2, -0.1, 0.15, 0.0, -0.2])  # and unevenly spaced along it
        base = trial.make_base(points, faces, project=_fold_surface)
        tool = hardtidy.TrueUpEdges()
        self.assertEqual(tool.check(base), "")
        cand = trial.finish(tool.run(base, {"strength": 1.0}))
        self.assertFalse(cand.refusal)
        self.assertTrue(np.allclose(cand.co[inner, 1:], 0.0, atol=1e-6))  # on the crease
        self.assertTrue(np.allclose(np.diff(cand.co[crease, 0]), 1.0, atol=1e-6))  # evenly spaced
        self.assertTrue(np.allclose(cand.co[[crease[0], crease[-1]]], points[[crease[0], crease[-1]]]))  # ends stay
        self.assertIn("1 edge trued, 1 straight", cand.note)
        self.assertIn("-> 0.0%", cand.numbers["wave"][0])
        on = _fold_surface(cand.co)
        self.assertTrue(np.allclose(on, cand.co, atol=1e-9))  # every corner on the original

    def test_half_strength_moves_half_way(self):
        points, faces, crease = _fold()
        points[crease[3], 1] = 0.2
        base = trial.make_base(points, faces, project=_fold_surface)
        half = trial.finish(hardtidy.TrueUpEdges().run(base, {"strength": 0.5}))
        self.assertAlmostEqual(half.co[crease[3], 1], 0.1, places=6)

    def test_no_hard_edges_refuses(self):
        points, faces, _ = _fold()
        flat = [f for f in faces if np.allclose(points[list(f), 2], 0.0)]
        base = trial.make_base(points, flat, project=_fold_surface)
        self.assertEqual(hardtidy.TrueUpEdges().check(base), hardtidy.NO_HARD_EDGES)
        empty = trial.make_base(np.zeros((0, 3)), [], project=None)
        self.assertEqual(hardtidy.TrueUpEdges().check(empty), hardtidy.NO_HARD_EDGES)

    def test_chains_end_where_three_hard_edges_meet(self):
        points, faces, crease = _fold()
        chains = hardtidy.hard_chains(points, faces)
        self.assertEqual(len(chains), 1)
        self.assertEqual(sorted(chains[0][0]), sorted(crease))


class EdgesTest(unittest.TestCase):
    def test_edge_table_matches_a_plain_count(self):
        points, faces, _ = _fold()
        faces = faces + [(0, 1, 2)]  # a triangle on top: three faces on some edges
        ed = hardtidy.Edges(faces, len(points))
        users: dict = {}
        for i, f in enumerate(faces):
            for k in range(len(f)):
                a, b = sorted((f[k], f[(k + 1) % len(f)]))
                users.setdefault((a, b), []).append(i)
        self.assertEqual(len(ed.ab), len(users))
        for e, (a, b) in enumerate(ed.ab.tolist()):
            fs = users[(a, b)]
            self.assertEqual(ed.count[e], len(fs))
            self.assertEqual(ed.f0[e], fs[0])
            self.assertEqual(ed.f1[e], fs[1] if len(fs) > 1 else -1)
        self.assertEqual(ed.find([1, 5], [0, 999]).tolist(), [ed.find([0], [1])[0], -1])


class PressPanelsFlatTest(unittest.TestCase):
    def test_bumpy_plate_goes_flat(self):
        n = 6
        points = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
        faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
        rng = np.random.default_rng(1)
        inner = [v for v in range(len(points)) if 0 < points[v, 0] < n and 0 < points[v, 1] < n]
        points[inner, 2] += rng.uniform(-0.08, 0.08, len(inner))
        base = trial.make_base(points, faces, project=_flat)
        cand = trial.finish(hardtidy.PressPanelsFlat().run(base, {"strength": 1.0}))
        self.assertFalse(cand.refusal)
        self.assertTrue(np.allclose(cand.co[:, 2], 0.0, atol=1e-9))
        self.assertTrue(np.allclose(cand.co[:, :2], points[:, :2]))  # pressed straight down, not slid
        self.assertEqual(cand.numbers["panels"][0], "1 panel")
        self.assertIn("-> 0.0%", cand.numbers["bump"][0])

    def test_shared_border_lands_on_both_planes(self):
        points, faces, crease = _fold()
        rng = np.random.default_rng(2)
        moving = [v for v in range(len(points)) if 0 < points[v, 0] < 6]
        points[moving] += rng.uniform(-0.04, 0.04, (len(moving), 3))
        base = trial.make_base(points, faces, project=_fold_surface)
        cand = trial.finish(hardtidy.PressPanelsFlat().run(base, {"strength": 1.0}))
        self.assertEqual(cand.numbers["panels"][0], "2 panels")
        inner = crease[1:-1]
        self.assertLess(np.abs(cand.co[inner, 1:]).max(), 2e-3)  # on the crease line, give or take the small pull

    def test_curved_original_has_no_panels(self):
        ring, rows = 12, 4
        t = np.linspace(0.0, 2.0 * np.pi, ring, endpoint=False)
        points = np.array([(np.cos(a), np.sin(a), z * 0.5) for z in range(rows + 1) for a in t])
        faces = [(z * ring + k, z * ring + (k + 1) % ring, (z + 1) * ring + (k + 1) % ring, (z + 1) * ring + k) for z in range(rows) for k in range(ring)]

        def cylinder(p):
            p = np.asarray(p, dtype=np.float64)
            r = np.linalg.norm(p[:, :2], axis=1, keepdims=True)
            return np.concatenate((p[:, :2] / np.maximum(r, 1e-12), p[:, 2:]), axis=1)

        base = trial.make_base(points, faces, project=cylinder)
        cand = trial.finish(hardtidy.PressPanelsFlat().run(base, {"strength": 1.0}))
        self.assertEqual(cand.refusal, hardtidy.NO_PANELS)

    def test_empty_refuses(self):
        empty = trial.make_base(np.zeros((0, 3)), [], project=None)
        self.assertEqual(hardtidy.PressPanelsFlat().check(empty), hardtidy.NO_PANELS)


# --------------------------------------------------------------------------- Auto Fix: spots, own numbers, propose


def _spots(tool, base):
    return trial.finish(tool.spots(base))


def _propose(tool, base, spots=None, settings=None):
    from triadq import autofix

    spots = _spots(tool, base) if spots is None else spots
    return spots, trial.finish(autofix.propose(tool, base, settings or {}, spots=spots))


def _wavy_fold(mirror=None):
    points, faces, crease = _fold()
    inner = crease[1:-1]
    points[inner, 1] += 0.5 * np.array([0.15, -0.0, 0.12, 0.0, 0.18])  # half the wave of the test above: the first row of quads follows it
    points[inner, 0] += 0.5 * np.array([0.2, -0.1, 0.15, 0.0, -0.2])
    return points, faces, crease, trial.make_base(points, faces, project=_fold_surface, mirror=mirror)


class TrueUpEdgesSpotsTest(unittest.TestCase):
    def test_fields(self):
        tool = hardtidy.TrueUpEdges()
        self.assertEqual([k[0] for k in tool.kinds], ["wavy", "off"])
        self.assertTrue(tool.auto)
        self.assertFalse(tool.changes_faces)
        self.assertGreater(tool.least, 0.0)

    def test_wavy_chain_is_a_spot_and_fixing_it_drops_the_own_number(self):
        points, faces, crease, base = _wavy_fold()
        tool = hardtidy.TrueUpEdges()
        spots, p = _propose(tool, base)
        self.assertEqual([sp.kind for sp in spots], ["wavy"])
        self.assertGreater(spots[0].bad, 0.02)  # edges off the crease, on average
        self.assertEqual(p.reason, "")
        self.assertEqual(p.fixed, spots)
        after = tool.own_number(base, p.cand.co, base.faces, spots[0])
        self.assertLess(after, 1e-6)
        self.assertGreater(tool.own_number(base, base.co, base.faces, spots[0]), 0.02)
        self.assertTrue(np.allclose(p.cand.co[crease[1:-1], 1:], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(p.cand.co[[crease[0], crease[-1]]], points[[crease[0], crease[-1]]]))

    def test_straight_creases_make_a_wavy_spot_and_bent_ones_do_not_line_up(self):
        points, faces, crease = _fold()
        points[crease[1:-1], 1] += 0.15
        base = trial.make_base(points, faces, project=_fold_surface)
        self.assertEqual([sp.kind for sp in _spots(hardtidy.TrueUpEdges(), base)], ["wavy"])  # its creases line up: a wave
        bent = np.array([(0, 0, 0), (1, 0.5, 0), (2, 0, 0)], dtype=np.float64)
        self.assertIsNone(hardtidy._straight(bent, 2.0, False))
        self.assertIsNotNone(hardtidy._straight(bent * (1, 0, 0), 2.0, False))

    def test_a_chain_already_on_its_crease_is_no_spot(self):
        points, faces, _crease = _fold()
        base = trial.make_base(points, faces, project=_fold_surface)
        self.assertEqual(_spots(hardtidy.TrueUpEdges(), base), [])

    def test_a_spot_with_no_hard_edge_in_it_comes_back_in_left(self):
        points, faces, crease, base = _wavy_fold()
        tool = hardtidy.TrueUpEdges()
        real = _spots(tool, base)[0]
        flat = [i for i, f in enumerate(faces) if np.allclose(points[list(f), 2], 0.0) and points[list(f), 1].min() >= 2.0]
        fake = trial.Spot("off", np.array(flat), points[list(faces[flat[0]])].mean(axis=0), 0.5)
        _s, p = _propose(tool, base, [real, fake])
        self.assertEqual(p.fixed, [real])
        self.assertEqual([why for sp, why in p.left if sp is fake], ["no better"])

    def test_mirrored_base_and_nothing_free(self):
        points, faces, crease, base = _wavy_fold(mirror=(0, 0.0))
        spots, p = _propose(hardtidy.TrueUpEdges(), base)
        self.assertEqual(len(spots), 1)
        self.assertEqual(p.reason, "")
        on = np.isclose(points[:, 0], 0.0)
        self.assertTrue(np.allclose(p.cand.co[on, 0], 0.0))  # the plane's corners stay in it
        nothing = trial.make_base(points, faces, scope=np.zeros(len(faces), dtype=bool), project=_fold_surface)
        self.assertEqual(_spots(hardtidy.TrueUpEdges(), nothing), [])
        self.assertEqual(_spots(hardtidy.TrueUpEdges(), trial.make_base(np.zeros((0, 3)), [])), [])


def _plate(n: int = 6, bump: float = 0.08):
    points = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    rng = np.random.default_rng(1)
    inner = [v for v in range(len(points)) if 0 < points[v, 0] < n and 0 < points[v, 1] < n]
    points[inner, 2] += rng.uniform(-bump, bump, len(inner))
    return points, faces


class PressPanelsFlatSpotsTest(unittest.TestCase):
    def test_fields(self):
        tool = hardtidy.PressPanelsFlat()
        self.assertEqual([k[0] for k in tool.kinds], ["bumpy"])
        self.assertTrue(tool.auto)
        self.assertGreater(tool.least, 0.0)

    def test_bumpy_plate_is_a_spot_and_goes_flat(self):
        points, faces = _plate()
        base = trial.make_base(points, faces, project=_flat)
        tool = hardtidy.PressPanelsFlat()
        spots, p = _propose(tool, base)
        self.assertEqual([sp.kind for sp in spots], ["bumpy"])
        self.assertEqual(p.reason, "")
        self.assertEqual(p.fixed, spots)
        self.assertGreater(tool.own_number(base, base.co, base.faces, spots[0]), 0.01)
        self.assertLess(tool.own_number(base, p.cand.co, base.faces, spots[0]), 1e-9)
        self.assertTrue(np.allclose(p.cand.co[:, 2], 0.0, atol=1e-9))

    def test_flat_plate_and_curved_original_have_no_spot(self):
        points, faces = _plate(bump=0.0)
        self.assertEqual(_spots(hardtidy.PressPanelsFlat(), trial.make_base(points, faces, project=_flat)), [])
        points, faces = _plate()
        self.assertEqual(_spots(hardtidy.PressPanelsFlat(), trial.make_base(points, faces, project=lambda p: np.asarray(p) + 0.0)), [])  # your original is as bumpy as the result: no panel

    def test_a_spot_that_is_already_flat_comes_back_in_left(self):
        points, faces = _plate()
        base = trial.make_base(points, faces, project=_flat)
        tool = hardtidy.PressPanelsFlat()
        real = _spots(tool, base)[0]
        fake = trial.Spot("bumpy", np.array([0, 1]), points[list(faces[0])].mean(axis=0), 0.5)
        _s, p = _propose(tool, base, [real, fake])
        self.assertIn(fake, [sp for sp, why in p.left])
        self.assertEqual([why for sp, why in p.left if sp is fake], ["no better"])

    def test_two_panels_that_touch_are_one_spot_and_the_mirror_plane_holds(self):
        points, faces, crease = _fold()
        rng = np.random.default_rng(2)
        moving = [v for v in range(len(points)) if 0 < points[v, 0] < 6]
        points[moving] += rng.uniform(-0.04, 0.04, (len(moving), 3))
        base = trial.make_base(points, faces, project=_fold_surface, mirror=(0, 0.0))
        spots, p = _propose(hardtidy.PressPanelsFlat(), base)
        self.assertEqual(len(spots), 1)  # they share the crease corners
        self.assertEqual(p.reason, "")
        on = np.isclose(points[:, 0], 0.0)
        self.assertTrue(np.allclose(p.cand.co[on, 0], 0.0))

    def test_empty_and_nothing_free(self):
        self.assertEqual(_spots(hardtidy.PressPanelsFlat(), trial.make_base(np.zeros((0, 3)), [])), [])
        points, faces = _plate()
        nothing = trial.make_base(points, faces, scope=np.zeros(len(faces), dtype=bool), project=_flat)
        self.assertEqual(_spots(hardtidy.PressPanelsFlat(), nothing), [])


if __name__ == "__main__":
    unittest.main()
