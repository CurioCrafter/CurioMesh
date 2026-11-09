# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Preview Tidy's lines and Seal the Middle on small hand-made meshes."""

import unittest

import numpy as np

from triadq import tidytools as T
from triadq import trial


def _grid(n: int, size: float = 1.0):
    points = np.array([(x * size, y * size, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    return points, faces


def _flat(points):
    out = np.array(points, dtype=np.float64)
    out[:, 2] = 0.0
    return out


def _lines(base):
    return trial.finish(T.PreviewTidy().lines(base))


def _rows(base):
    return {r["key"]: r for r in _lines(base)}


def _apply(row):
    co, faces, parent = row["fix"]
    return np.asarray(co), faces, parent


class PreviewTidyTest(unittest.TestCase):
    def test_clean_grid_is_refused(self):
        points, faces = _grid(4)
        cand = trial.finish(T.PreviewTidy().run(trial.make_base(points, faces, project=_flat), {"row": ""}))
        self.assertEqual(cand.refusal, T.NOTHING_TO_TIDY)

    def test_doublet_found_and_joined(self):
        points, faces = _grid(4)
        # split quad 5 (corners 6, 7, 12, 11) with an extra corner in its middle: two quads sharing two edges
        points = np.vstack((points, [(1.5, 1.5, 0.0)]))
        mid = len(points) - 1
        faces = [f for i, f in enumerate(faces) if i != 5] + [(6, 7, mid, 11), (7, 12, 11, mid)]
        rows = _rows(trial.make_base(points, faces, project=_flat))
        self.assertIn("doublets", rows)
        self.assertEqual(rows["doublets"]["count"], 1)
        co, new, parent = _apply(rows["doublets"])
        self.assertEqual(len(new), len(faces) - 1)
        self.assertEqual(len(co), len(points) - 1)  # the corner went
        self.assertEqual(len(parent), len(new))
        self.assertTrue(all(len(f) == 4 for f in new))

    def test_triangle_pairs_make_quads_and_respect_kinds(self):
        points, faces = _grid(3)
        q = faces[4]
        faces = [f for i, f in enumerate(faces) if i != 4] + [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
        base = trial.make_base(points, faces, project=_flat)
        rows = _rows(base)
        self.assertEqual(rows["pairs"]["count"], 1)
        co, new, _parent = _apply(rows["pairs"])
        self.assertEqual(len(new), 9)
        self.assertTrue(all(len(f) == 4 for f in new))
        kinds = np.zeros(len(faces), dtype=np.int64)
        kinds[-1] = 1  # a material border between the two triangles
        self.assertNotIn("pairs", _rows(trial.make_base(points, faces, project=_flat, kinds=kinds)))

    def test_inward_face_turned_round(self):
        points, faces = _grid(3)
        faces[4] = tuple(reversed(faces[4]))
        rows = _rows(trial.make_base(points, faces, project=_flat))
        self.assertEqual(rows["inward"]["count"], 1)
        _co, new, _parent = _apply(rows["inward"])
        self.assertEqual(tuple(new[4]), tuple(reversed(faces[4])))
        self.assertNotIn("inward", _rows(trial.make_base(points, new, project=_flat)))

    def test_sliver_opened_up_and_stays_on_the_surface(self):
        points, faces = _grid(4)
        points[12, 0] = 2.93  # squeeze the column of quads left of corner 12 into needles
        points[7, 0] = 2.93
        rows = _rows(trial.make_base(points, faces, project=_flat))
        self.assertIn("slivers", rows)
        co = rows["slivers"]["fix"][0]
        self.assertTrue(np.allclose(co[:, 2], 0.0))
        self.assertLess(_rows(trial.make_base(co, faces, project=_flat)).get("slivers", {"count": 0})["count"], rows["slivers"]["count"])

    def test_picking_a_row_returns_its_fix_and_the_rows(self):
        points, faces = _grid(3)
        faces[4] = tuple(reversed(faces[4]))
        base = trial.make_base(points, faces, project=_flat)
        report = trial.finish(T.PreviewTidy().run(base, {"row": ""}))
        self.assertIsNone(report.co)
        self.assertEqual([r["key"] for r in report.rows], ["inward"])
        fix = trial.finish(T.PreviewTidy().run(base, {"row": "inward"}))
        self.assertIsNotNone(fix.faces)
        self.assertIs(fix.rows, report.rows)  # worked out once per result
        title, numbers, third = trial.readout(report, T.PreviewTidy(), {}, "ready")
        self.assertEqual(numbers, [])
        self.assertTrue(third)

    def test_empty_and_tiny_meshes(self):
        empty = trial.make_base(np.zeros((0, 3)), [])
        self.assertEqual(trial.finish(T.PreviewTidy().run(empty, {})).refusal, T.NOTHING_TO_TIDY)
        tri = trial.make_base(np.eye(3), [(0, 1, 2)])
        self.assertEqual(trial.finish(T.PreviewTidy().run(tri, {})).refusal, T.NOTHING_TO_TIDY)
        self.assertEqual(T.SealMiddle().check(tri), T.NOT_MIRRORED)


class SealMiddleTest(unittest.TestCase):
    def _mirrored(self, wobble=0.03, double=False):
        """A 4x2 grid across x = 2 (the plane), its middle column a little off the plane."""
        points, faces = _grid(4)
        points = points[(points[:, 1] <= 2.0)]
        faces = [f for f in faces if max(f) < len(points)]
        middle = np.flatnonzero(np.isclose(points[:, 0], 2.0))
        points[middle, 0] += wobble * np.array([1.0, -1.0, 0.5])[: len(middle)]
        if double:  # one middle corner doubled: the right-hand faces use a copy of it
            v = int(middle[1])
            points = np.vstack((points, points[v] + (0.02, 0.0, 0.0)))
            copy = len(points) - 1
            faces = [tuple(copy if (c == v and np.mean(points[list(f), 0]) > 2.0) else c for c in f) for f in faces]
        return points, faces

    def test_refuses_without_a_plane(self):
        points, faces = self._mirrored()
        base = trial.make_base(points, faces, project=_flat)
        self.assertEqual(T.SealMiddle().check(base), T.NOT_MIRRORED)

    def test_puts_the_middle_on_the_plane(self):
        points, faces = self._mirrored()
        base = trial.make_base(points, faces, project=_flat, pick={"plane": (0, 2.0)})
        self.assertEqual(T.SealMiddle().check(base), "")
        cand = trial.finish(T.SealMiddle().run(base, {}))
        self.assertFalse(cand.refusal)
        middle = np.flatnonzero(np.abs(points[:, 0] - 2.0) < 0.1)
        self.assertTrue(np.allclose(cand.co[middle, 0], 2.0))
        self.assertTrue(np.allclose(cand.co[:, 2], 0.0))
        self.assertIn("3 seam vertices put on the plane, 0 merged, 0 gaps left", cand.note)
        self.assertGreater(len(cand.lines), 0)
        again = trial.make_base(cand.co, cand.faces, project=_flat, pick={"plane": (0, 2.0)})
        self.assertEqual(trial.finish(T.SealMiddle().run(again, {})).refusal, T.SEALED)  # Again: nothing left

    def test_a_pair_either_side_of_the_middle_stays(self):
        points, faces = _grid(4)  # corners at x = 0..4; the plane at x = 2.05 puts x = 2 and 2.1 either side
        points[np.isclose(points[:, 0], 3.0), 0] = 2.1
        base = trial.make_base(points, faces, project=_flat, pick={"plane": (0, 2.05)})
        cand = trial.finish(T.SealMiddle().run(base, {}))
        self.assertEqual(cand.refusal, T.SEALED)  # each has its twin: nothing is squashed onto the plane

    def test_merges_doubled_corners_and_closes_the_gap(self):
        points, faces = self._mirrored(wobble=0.0, double=True)
        base = trial.make_base(points, faces, project=_flat, mirror=(0, 2.0))
        before = T._open_corners(faces, len(points)).sum()
        cand = trial.finish(T.SealMiddle().run(base, {}))
        self.assertIn("1 merged", cand.note)
        self.assertEqual(len(cand.co), len(points) - 1)
        self.assertLess(T._open_corners(cand.faces, len(cand.co)).sum(), before)
        self.assertEqual(len(cand.parent), len(cand.faces))


# --------------------------------------------------------------------------- Auto Fix: spots, own numbers, propose


def _plant(n: int = 8):
    """An n x n grid with a doublet (face 9), a triangle pair (face 14), a face facing in (face 49) and two sliver quads
    (faces 45 and 53, by squeezing corners (5, 5) and (5, 6)), far apart."""
    points, faces = _grid(n)
    w = n + 1
    q = faces[9]  # a corner in the middle of face 9, two quads sharing two edges
    points = np.vstack((points, [(1.5, 1.5, 0.0)]))
    mid = len(points) - 1
    out = list(faces)
    out[9] = (q[0], q[1], mid, q[3])
    out.append((q[1], q[2], q[3], mid))
    t = faces[14]  # a quad cut into two triangles
    out[14] = (t[0], t[1], t[2])
    out.append((t[0], t[2], t[3]))
    out[49] = tuple(reversed(faces[49]))
    points[5 * w + 5, 0] = 5.93
    points[6 * w + 5, 0] = 5.93
    return points, out


def _tool():
    return T.PreviewTidy()


def _spots(tool, base):
    return trial.finish(tool.spots(base))


def _propose(tool, base, spots=None):
    from triadq import autofix

    spots = _spots(tool, base) if spots is None else spots
    return spots, trial.finish(autofix.propose(tool, base, {}, spots=spots))


class PreviewTidySpotsTest(unittest.TestCase):
    def test_every_line_has_a_kind_and_the_tool_fills_the_auto_fields(self):
        tool = _tool()
        self.assertEqual([k[0] for k in tool.kinds], [k[0] for k in T.TIDY_LINES])
        self.assertTrue(tool.auto and tool.changes_faces)
        self.assertGreater(tool.least, 0.0)
        self.assertIn("steps", tool.defaults)

    def test_spots_finds_each_planted_problem(self):
        points, faces = _plant()
        base = trial.make_base(points, faces, project=_flat)
        spots = _spots(_tool(), base)
        self.assertEqual(sorted(sp.kind for sp in spots), ["doublets", "inward", "pairs", "slivers"])
        by = {sp.kind: sp for sp in spots}
        self.assertEqual(sorted(by["doublets"].faces.tolist()), [9, 64])
        self.assertEqual(sorted(by["pairs"].faces.tolist()), [14, 65])
        self.assertEqual(by["inward"].faces.tolist(), [49])
        self.assertTrue({45, 53} <= set(by["slivers"].faces.tolist()))  # the two slivers are one spot, with the faces round them
        for sp in spots:
            self.assertEqual(sp.pick["row"], sp.kind)
            self.assertEqual(sp.at.shape, (3,))
            self.assertGreater(_tool().own_number(base, base.co, base.faces, sp), 0.0)  # something of its kind is there

    def test_propose_fixes_all_of_them_and_each_own_number_drops(self):
        points, faces = _plant()
        base = trial.make_base(points, faces, project=_flat)
        tool = _tool()
        spots, p = _propose(tool, base)
        self.assertEqual(p.reason, "")
        self.assertEqual(len(p.fixed), 4)
        self.assertEqual(p.left, [])
        self.assertEqual(len(p.cand.faces), 64)  # the doublet and the triangle pair are one quad each again
        after = trial.make_base(p.cand.co, p.cand.faces, project=_flat)
        again = _spots(tool, after)
        self.assertNotIn("doublets", [sp.kind for sp in again])
        self.assertNotIn("pairs", [sp.kind for sp in again])
        self.assertNotIn("inward", [sp.kind for sp in again])
        self.assertGreaterEqual(p.before.bad, 1)  # the face facing in counts as a bad one ...
        self.assertEqual(p.after.bad, 0)  # ... and turning it round repairs it
        self.assertGreaterEqual(p.after.tidy, p.before.tidy)

    def test_sliver_spot_that_cannot_move_comes_back_in_left(self):
        points, faces = _plant()
        stay = np.zeros(len(points), dtype=bool)
        stay[[50, 51, 59, 60, 68, 69]] = True  # every corner of the two slivers is held
        base = trial.make_base(points, faces, project=_flat, stay=stay)
        spots, p = _propose(_tool(), base)
        left = {sp.kind: why for sp, why in p.left}
        self.assertIn("slivers", left)
        self.assertIn(left["slivers"], ("held", "no better"))
        self.assertEqual(len(p.fixed), 3)

    def test_the_sliver_steps_are_a_setting_with_no_top(self):
        points, faces = _plant()
        base = trial.make_base(points, faces, project=_flat)
        few = trial.finish(_tool().run(base, {"row": "slivers", "only": True, "steps": 1}))
        base2 = trial.make_base(points, faces, project=_flat)
        many = trial.finish(_tool().run(base2, {"row": "slivers", "only": True, "steps": 400}))
        self.assertLessEqual(np.abs(few.co - points).sum(), np.abs(many.co - points).sum())
        self.assertGreater(np.abs(many.co - points).sum(), 0.0)

    def test_mirrored_base_keeps_the_plane(self):
        points, faces = _plant()
        base = trial.make_base(points, faces, project=_flat, mirror=(0, 4.0))
        spots, p = _propose(_tool(), base)
        self.assertEqual(p.reason, "")
        on = np.isclose(points[:, 0], 4.0)
        self.assertTrue(np.allclose(p.cand.co[on[: len(p.cand.co)]], points[: len(p.cand.co)][on[: len(p.cand.co)]]))

    def test_empty_nothing_free_and_clean_meshes_have_no_spots(self):
        tool = _tool()
        self.assertEqual(_spots(tool, trial.make_base(np.zeros((0, 3)), [])), [])
        points, faces = _plant()
        self.assertEqual(_spots(tool, trial.make_base(points, faces, scope=np.zeros(len(faces), dtype=bool))), [])
        clean, grid = _grid(4)
        self.assertEqual(_spots(tool, trial.make_base(clean, grid, project=_flat)), [])
        self.assertEqual(tool.own_number(trial.make_base(clean, grid), clean, grid, trial.Spot("pairs", np.zeros(0, dtype=np.int64), np.zeros(3), 0.0)), 0.0)


class SealMiddleSpotTest(unittest.TestCase):
    def _base(self, **kw):
        return SealMiddleTest()._mirrored(**kw)

    def test_the_seam_is_one_spot_and_fixing_it_leaves_nothing_of_the_kind(self):
        points, faces = self._base()
        base = trial.make_base(points, faces, project=_flat, pick={"plane": (0, 2.0)})
        tool = T.SealMiddle()
        spots, p = _propose(tool, base)
        self.assertEqual([sp.kind for sp in spots], ["seam"])
        self.assertEqual(p.reason, "")
        self.assertEqual(len(p.fixed), 1)
        self.assertGreater(tool.own_number(base, base.co, base.faces, spots[0]), 0.0)
        self.assertEqual(tool.own_number(base, p.cand.co, p.cand.faces, trial.Spot("seam", np.arange(len(p.cand.faces)), np.zeros(3), 0.0, spots[0].pick)), 0.0)
        middle = np.flatnonzero(np.abs(points[:, 0] - 2.0) < 0.1)
        self.assertTrue(np.allclose(p.cand.co[middle, 0], 2.0))
        self.assertEqual(_spots(tool, trial.make_base(p.cand.co, p.cand.faces, project=_flat, pick={"plane": (0, 2.0)})), [])  # nothing left

    def test_doubled_corner_in_a_mirrored_base(self):
        points, faces = self._base(wobble=0.0, double=True)
        base = trial.make_base(points, faces, project=_flat, mirror=(0, 2.0))
        spots, p = _propose(T.SealMiddle(), base)
        self.assertEqual(len(spots), 1)
        self.assertEqual(len(p.fixed), 1)
        self.assertEqual(len(p.cand.co), len(points) - 1)

    def test_a_spot_with_no_seam_in_it_comes_back_in_left(self):
        points, faces = self._base()
        base = trial.make_base(points, faces, project=_flat, pick={"plane": (0, 2.0)})
        tool = T.SealMiddle()
        real = _spots(tool, base)[0]
        far = [i for i, f in enumerate(faces) if np.abs(points[list(f), 0] - 2.0).min() > 0.9]
        fake = trial.Spot("seam", np.array(far), points[list(faces[far[0]])].mean(axis=0), 0.5, real.pick)
        _spots_, p = _propose(tool, base, [real, fake])
        self.assertEqual([why for sp, why in p.left if sp is fake], ["held"])
        self.assertEqual(p.fixed, [real])

    def test_no_plane_no_spots_and_a_tidy_seam_has_none(self):
        points, faces = self._base()
        self.assertEqual(_spots(T.SealMiddle(), trial.make_base(points, faces, project=_flat)), [])
        points, faces = self._base(wobble=0.0)
        self.assertEqual(_spots(T.SealMiddle(), trial.make_base(points, faces, project=_flat, mirror=(0, 2.0))), [])
        self.assertEqual(_spots(T.SealMiddle(), trial.make_base(np.zeros((0, 3)), [], mirror=(0, 0.0))), [])


if __name__ == "__main__":
    unittest.main()
