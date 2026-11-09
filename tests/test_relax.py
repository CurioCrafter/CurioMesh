# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_organize import ball, onto_ball, skew  # noqa: E402
from triadq import relax as R  # noqa: E402
from triadq import trial as T  # noqa: E402


def grid(n: int, size: float = 2.0):
    """An n x n grid of quads on the plane z = 0, from -size/2 to size/2."""
    s = np.linspace(-size / 2, size / 2, n + 1)
    co = np.array([(x, y, 0.0) for y in s for x in s])
    faces = [(j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i) for j in range(n) for i in range(n)]
    return co, faces


def box(n: int, jitter: float = 0.0):
    """The surface of the cube [-1, 1]^3, n x n quads a side, corners shaken along the faces and the edges."""
    co, faces = ball(n)
    co = co / np.abs(co).max(axis=1, keepdims=True)
    rng = np.random.default_rng(3)
    shake = jitter * rng.normal(size=co.shape) / n
    flat = np.abs(np.abs(co) - 1.0) < 1e-9  # the coordinates pinned to a face of the cube
    shake[flat] = 0.0
    return np.clip(co + shake, -1.0, 1.0), faces


def onto_box(points):
    p = np.clip(np.asarray(points, dtype=np.float64), -1.0, 1.0)
    inside = (np.abs(p) < 1.0).all(axis=1)
    k = np.argmax(np.abs(p[inside]), axis=1)
    rows = np.flatnonzero(inside)
    p[rows, k] = np.sign(p[rows, k] + 1e-30)
    return p


def ridge(x, y):
    return 0.35 * np.exp(-((x / 0.18) ** 2))


def onto_ridge(points):
    p = np.asarray(points, dtype=np.float64).copy()
    p[:, 2] = ridge(p[:, 0], p[:, 1])
    return p


def ridge_normals(points):
    x = np.asarray(points, dtype=np.float64)[:, 0]
    dz = 0.35 * np.exp(-((x / 0.18) ** 2)) * (-2.0 * x / 0.18**2)
    n = np.stack((-dz, np.zeros_like(dz), np.ones_like(dz)), axis=1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def run(tool, base, **settings):
    return T.diff(base, T.finish(tool.run(base, dict(tool.defaults, **settings))))


def edge_spread(co, faces) -> float:
    e = T.edges_of(faces)
    length = np.linalg.norm(co[e[:, 0]] - co[e[:, 1]], axis=1)
    return float(length.std() / length.mean())


class SourceRelaxTest(unittest.TestCase):
    def test_evens_out_and_stays_on_the_surface(self):
        co, faces = ball(8, jitter=0.3)
        base = T.make_base(co, faces, project=onto_ball)
        tool = R.SourceRelax()
        self.assertEqual(tool.check(base), "")
        cand = run(tool, base, strength=1.0)
        self.assertLess(edge_spread(cand.co, faces), 0.8 * edge_spread(co, faces))
        np.testing.assert_allclose(np.linalg.norm(cand.co, axis=1), 1.0, atol=1e-9)  # on the source, nothing shrank
        self.assertTrue(cand.exact)
        self.assertGreater(cand.moved.max(), 0.0)

    def test_yields_and_blend_ends(self):
        co, faces = ball(6, jitter=0.3)
        base = T.make_base(co, faces, project=onto_ball)
        work = R.SourceRelax().run(base, {"strength": 1.0})
        self.assertGreater(sum(1 for _ in iter(lambda: next(work, None), None)), 1)
        full = run(R.SourceRelax(), base, strength=1.0)
        np.testing.assert_allclose(T.blend(base, full, 0.0).co, co)
        np.testing.assert_allclose(T.blend(base, full, 1.0).co, full.co)

    def test_scope_edge_and_zero_strength(self):
        co, faces = ball(6, jitter=0.3)
        scope = np.zeros(len(faces), dtype=bool)
        scope[: len(faces) // 6] = True  # one side of the cube
        base = T.make_base(co, faces, scope=scope, project=onto_ball)
        cand = run(R.SourceRelax(), base, strength=1.0)
        np.testing.assert_array_equal(cand.co[~base.free], co[~base.free])
        self.assertTrue(np.abs(cand.co[base.free] - co[base.free]).max() > 0)
        np.testing.assert_array_equal(run(R.SourceRelax(), base, strength=0.0).co, co)

    def test_keep_going_settles(self):
        co, faces = ball(6, jitter=0.3)
        base = T.make_base(co, faces, project=onto_ball)
        once = run(R.SourceRelax(), base, strength=1.0)
        more = run(R.SourceRelax(), base, strength=1.0, keep_going=True)
        self.assertLessEqual(edge_spread(more.co, faces), edge_spread(once.co, faces) + 1e-9)
        self.assertIn("until it settled", more.note)

    def test_nothing_to_move(self):
        co, faces = grid(1)  # one quad: every corner is on the border
        self.assertEqual(R.SourceRelax().check(T.make_base(co, faces)), R.NOTHING_FREE)
        self.assertEqual(R.SquareUp().check(T.make_base(co, faces)), R.NO_SLANT)
        empty = T.make_base(np.zeros((0, 3)), [])
        for tool in (R.SourceRelax(), R.KeepLinesRelax(), R.SquareUp(), R.RelaxTowardDetail()):
            self.assertNotEqual(tool.check(empty), "")


class SquareUpTest(unittest.TestCase):
    def test_squares_corners_and_keeps_sizes(self):
        co, faces = ball(8, jitter=0.35)
        base = T.make_base(co, faces, project=onto_ball)
        tool = R.SquareUp()
        cand = run(tool, base, strength=1.0)
        self.assertLess(skew(cand.co, faces), skew(co, faces))
        quads = np.asarray(faces)
        before, after = R._areas(co, quads), R._areas(cand.co, quads)
        self.assertLess(abs(after.sum() / before.sum() - 1.0), 0.05)
        words, sign = cand.numbers["square"]
        self.assertTrue(words.startswith("square corners"))
        self.assertEqual(sign, 1)

    def test_square_pull_leaves_rectangles(self):
        co, faces = grid(3)
        move, _at = R.square_pull(co * np.array([2.0, 1.0, 1.0]), np.asarray(faces))
        self.assertLess(np.abs(move).max(), 1e-12)

    def test_refuses_square_grid(self):
        co, faces = grid(4)
        self.assertEqual(R.SquareUp().check(T.make_base(co, faces)), R.NO_SLANT)


class KeepLinesRelaxTest(unittest.TestCase):
    def test_hard_edges_slide_along_themselves(self):
        co, faces = box(6, jitter=0.6)
        base = T.make_base(co, faces, project=onto_box)
        tool = R.KeepLinesRelax()
        lines, _nb, slide, _free = tool._plan(base, tool.defaults)
        self.assertGreater(len(lines), 0)
        self.assertTrue(slide.any())
        cand = run(tool, base, strength=1.0, outline=False)
        on_edge = (np.abs(np.abs(co) - 1.0) < 1e-9).sum(axis=1) >= 2
        corner = (np.abs(np.abs(co) - 1.0) < 1e-9).sum(axis=1) == 3
        stays = (np.abs(np.abs(cand.co[on_edge]) - 1.0) < 1e-6).sum(axis=1) >= 2
        self.assertTrue(stays.all())  # corners on a cube edge are still on it
        np.testing.assert_array_equal(cand.co[corner], co[corner])
        self.assertTrue(np.abs(cand.co[on_edge & ~corner] - co[on_edge & ~corner]).max() > 1e-4)  # and they slid
        self.assertIsNotNone(cand.lines)
        self.assertTrue(cand.numbers["lines"][0].startswith("lines kept"))

    def test_no_lines_works_like_source_relax(self):
        co, faces = ball(6, jitter=0.3)
        base = T.make_base(co, faces, project=onto_ball)
        off = {k: False for k, _w in R.LINE_KINDS}
        cand = run(R.KeepLinesRelax(), base, strength=1.0, **off)
        same = run(R.SourceRelax(), base, strength=1.0)
        np.testing.assert_allclose(cand.co, same.co)
        self.assertEqual(cand.note, R.NO_LINES)

    def test_outline_counts_each_line_of_sight_once(self):
        co, faces = ball(8)
        co = co / np.linalg.norm(co, axis=1, keepdims=True)
        # a smooth ball: only where two outlines cross, not every edge any single view flips (264 of 768 before)
        self.assertLess(len(R.outline_edges(co, faces)), 0.05 * len(T.edges_of(faces)))

    def test_part_borders(self):
        co, faces = grid(6)
        piece = np.array([0 if co[list(f)].mean(axis=0)[0] < 0 else 1 for f in faces])
        base = T.make_base(co, faces, labels={"piece": piece})
        lines = R.kept_lines(base, {"parts": True})
        self.assertEqual(len(lines), 6)


class RelaxTowardDetailTest(unittest.TestCase):
    def test_smaller_quads_where_it_bends(self):
        co, faces = grid(16)
        co = onto_ridge(co)
        base = T.make_base(co, faces, project=onto_ridge, facing=ridge_normals)
        tool = R.RelaxTowardDetail()
        self.assertEqual(tool.check(base), "")
        cand = run(tool, base, mix=1.0)
        quads = np.asarray(faces)
        middle = np.abs(co[quads].mean(axis=1)[:, 0]) < 0.15
        before, after = R._areas(co, quads), R._areas(cand.co, quads)
        self.assertLess(after[middle].mean(), 0.95 * before[middle].mean())  # smaller on the ridge
        self.assertGreater(after[~middle].mean(), before[~middle].mean())  # bigger on the flat
        self.assertIsNotNone(cand.tint)
        self.assertTrue(cand.numbers["size"][0].startswith("size range"))
        np.testing.assert_allclose(cand.co[:, 2], ridge(cand.co[:, 0], cand.co[:, 1]))

    def test_refuses_flat(self):
        co, faces = grid(6)
        up = lambda p: np.tile([0.0, 0.0, 1.0], (len(p), 1))  # noqa: E731
        self.assertEqual(R.RelaxTowardDetail().check(T.make_base(co, faces, facing=up)), R.EVEN_BENDS)


class NumberTextTest(unittest.TestCase):
    def test_worded_numbers_pass_through(self):
        self.assertEqual(T.number_text("square", ("square corners 61% -> 78%", 1)), ("square corners 61% -> 78%", 1))


# --------------------------------------------------------------------------- Auto Fix spots (AUTO_FIX.md 6.2)

from triadq import autofix  # noqa: E402


def flat(p):
    q = np.asarray(p, dtype=np.float64).copy()
    q[:, 2] = 0.0
    return q


def planted(n=10, at=(5, 5), by=(0.07, 0.05), **kw):
    """A flat n x n grid with one interior corner pushed sideways: one slanted, lumpy patch."""
    co, faces = grid(n)
    co[at[1] * (n + 1) + at[0], :2] += by
    return T.make_base(co, faces, project=flat, **kw), co, faces


def spots_of(tool, base):
    return T.finish(tool.spots(base))


def propose(tool, base, spots):
    return T.finish(autofix.propose(tool, base, {}, spots=spots))


class AutoFixSpotsTest(unittest.TestCase):
    def check_tool(self, tool, base):
        """The planted problem is found, fixed with its own number lower, and a clean spot added by hand is put back."""
        found = spots_of(tool, base)
        self.assertEqual(len(found), 1)
        sp = found[0]
        self.assertEqual(sp.kind, tool.kinds[0][0])
        self.assertGreater(sp.bad, 0.0)
        quiet = next(f for f in range(len(base.faces)) if base.co[list(base.faces[f])].mean(axis=0)[0] < -0.6 and base.co[list(base.faces[f])].mean(axis=0)[1] < -0.6)
        ring = np.array([f for f in range(len(base.faces)) if len(set(base.faces[f]) & set(base.faces[quiet])) > 0])
        clean = T.Spot(tool.kinds[0][0], ring, base.co[list(base.faces[quiet])].mean(axis=0), 0.5)
        p = propose(tool, base, [sp, clean])
        self.assertEqual([id(s) for s in p.fixed], [id(sp)])
        self.assertEqual([(id(s), why) for s, why in p.left], [(id(clean), "no better")])
        self.assertLess(tool.own_number(base, p.cand.co, base.faces, sp), sp.bad - tool.least)
        np.testing.assert_array_equal(p.cand.co[_corners(base.faces, ring)], base.co[_corners(base.faces, ring)])  # put back
        self.assertEqual(p.after.bad, 0)

    def test_square_up(self):
        base, _co, _f = planted()
        self.check_tool(R.SquareUp(), base)

    def test_source_relax_and_keep_lines_split(self):
        base, _co, _f = planted()
        self.check_tool(R.SourceRelax(), base)
        self.assertEqual(spots_of(R.KeepLinesRelax(), base), [])  # no kept lines on a flat grid
        cube, faces = box(6, jitter=0.6)
        b = T.make_base(cube, faces, project=onto_box)
        keep, plain = spots_of(R.KeepLinesRelax(), b), spots_of(R.SourceRelax(), b)
        self.assertGreater(len(keep), 0)
        self.assertEqual(set(np.concatenate([s.faces for s in keep]).tolist()) & set(np.concatenate([s.faces for s in plain] or [[]]).tolist()), set())
        p = propose(R.KeepLinesRelax(), b, keep)
        self.assertGreater(len(p.fixed), 0)
        for sp in p.fixed:
            self.assertLess(R.KeepLinesRelax().own_number(b, p.cand.co, faces, sp), sp.bad)

    def test_relax_toward_detail(self):
        co, faces = grid(16)
        co = onto_ridge(co)
        base = T.make_base(co, faces, project=onto_ridge, facing=ridge_normals)
        tool = R.RelaxTowardDetail()
        found = spots_of(tool, base)
        self.assertGreater(len(found), 0)
        self.assertEqual(found[0].kind, "sizes")
        p = propose(tool, base, found)
        self.assertGreater(len(p.fixed), 0)
        for sp in p.fixed:
            self.assertLess(tool.own_number(base, p.cand.co, faces, sp), sp.bad - tool.least)
        flatbase = T.make_base(*grid(6), facing=lambda q: np.tile([0.0, 0.0, 1.0], (len(q), 1)))
        self.assertEqual(spots_of(tool, flatbase), [])  # equally curved everywhere: nothing to find

    def test_spots_never_share_a_corner_and_have_room(self):
        co, faces = grid(14)
        for at in ((3, 3), (4, 4), (10, 10)):  # the first two touch and become one patch
            co[at[1] * 15 + at[0], :2] += (0.06, 0.04)
        base = T.make_base(co, faces, project=flat)
        found = spots_of(R.SquareUp(), base)
        self.assertEqual(len(found), 2)
        corners = [set(_corners(faces, s.faces).tolist()) for s in found]
        self.assertEqual(corners[0] & corners[1], set())

    def test_nothing_to_find(self):
        tools = (R.SourceRelax(), R.KeepLinesRelax(), R.SquareUp(), R.RelaxTowardDetail())
        empty = T.make_base(np.zeros((0, 3)), [])
        one = T.make_base(*grid(1))  # every corner on the border: nothing free
        clean = T.make_base(*grid(6))
        for tool in tools:
            for base in (empty, one, clean):
                self.assertEqual(spots_of(tool, base), [], (tool.name, len(base.faces)))
            p = propose(tool, clean, [])  # a spot list with no spots
            self.assertEqual((p.fixed, p.reason), ([], "no better"))

    def test_scope_with_nothing_free(self):
        base, co, faces = planted()
        scope = np.zeros(len(faces), dtype=bool)
        scope[0] = True  # one corner quad: its corners are all on the scope edge
        small = T.make_base(co, faces, scope=scope, project=flat)
        for tool in (R.SourceRelax(), R.SquareUp()):
            self.assertEqual(spots_of(tool, small), [])

    def test_mirror_plane_holds(self):
        base, co, faces = planted(at=(7, 5), mirror=(0, 0.0))
        tool = R.SquareUp()
        found = spots_of(tool, base)
        self.assertEqual(len(found), 1)
        p = propose(tool, base, found)
        self.assertEqual(len(p.fixed), 1)
        plane = np.abs(co[:, 0]) < 1e-9
        np.testing.assert_array_equal(p.cand.co[plane], co[plane])

    def test_every_spot_wrapper_runs(self):
        base, _co, _f = planted()
        tool = autofix.every(R.SquareUp)()
        cand = T.finish(tool.run(base, dict(tool.defaults)))
        self.assertIsNone(cand.refusal or None)
        self.assertGreater(np.abs(cand.co - base.co).max(), 0.0)

    def test_plan_picks_them_up(self):
        got = {c for c, _k in autofix.plan({"shape"})}
        self.assertTrue({R.SquareUp, R.RelaxTowardDetail, R.SourceRelax, R.KeepLinesRelax} <= got)


def _corners(faces, ids):
    return np.unique([v for f in np.asarray(ids).tolist() for v in faces[f]])


if __name__ == "__main__":
    unittest.main()
