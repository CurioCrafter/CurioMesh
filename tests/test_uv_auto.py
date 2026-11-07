# SPDX-License-Identifier: GPL-3.0-or-later
"""UV in Auto (A4): Seam Snap through Auto Fix's propose, the job hooks (attach, cut, remap, fix) on a small plane."""

from __future__ import annotations

import types
import unittest

import numpy as np

import uv_parity_common as common  # noqa: F401  (puts the add-on folder on the path)

from triadq import autofix, trial as T, uvauto, uvkeep
from triadq.types import MeshData
from test_uv_keep import grid, two_island_source


def faces_of(ip, fl):
    return [tuple(fl[ip[i]:ip[i + 1]].tolist()) for i in range(len(ip) - 1)]


def setup():
    sa = two_island_source()
    plan = uvkeep.make_plan(sa, 25.0, uvkeep.Settings(mode="light", allowance=1.0))
    p, ip, fl = grid(10)  # edges at 0.1 steps, moved so a column sits about 0.03 from the seam at x = 0.5, a little uneven
    p[:, 0] += 0.03
    p[:, :2] += np.random.default_rng(1).uniform(-0.01, 0.01, (len(p), 2))
    return sa, plan, p, faces_of(ip, fl)


class Snap(unittest.TestCase):
    def test_spot_found_and_fixed_through_propose(self):
        sa, plan, p, faces = setup()
        xyz = uvauto.uvsnap.chain_xyz(plan.seams, sa.points)
        trial = uvauto.SeamSnap(plan.seams, xyz, use_fast=False)
        base = T.make_base(p, faces)
        spots = T.finish(trial.spots(base))
        self.assertEqual(len(spots), 1)
        self.assertGreater(spots[0].bad, uvauto.LEAST)
        prop = T.finish(autofix.propose(trial, base, {}, spots))
        self.assertEqual(prop.reason, "", prop.left)
        self.assertEqual(len(prop.fixed), 1)
        after = trial.own_number(base, prop.cand.co, faces, spots[0])
        self.assertLess(after, 0.5 * spots[0].bad)
        x = np.asarray(prop.cand.co)[:, 0]
        self.assertTrue(np.sum(np.abs(x - 0.5) < 1e-9) >= 4)  # the column near the seam sits on it

    def test_no_move_allowed_is_left_as_it_was(self):
        sa, plan, p, faces = setup()
        trial = uvauto.SeamSnap(plan.seams, uvauto.uvsnap.chain_xyz(plan.seams, sa.points), use_fast=False)
        base = T.make_base(p, faces)
        spots = T.finish(trial.spots(base))
        prop = T.finish(autofix.propose(trial, base, {"strength": 0.0}, spots))
        self.assertEqual(prop.reason, "no better")
        np.testing.assert_array_equal(np.asarray(prop.cand.co), p)

    def test_empty_and_no_seams(self):
        uvauto.LAST.clear()
        trial = uvauto.SeamSnap()
        base = T.make_base(np.zeros((0, 3)), [])
        self.assertEqual(T.finish(trial.spots(base)), [])
        self.assertTrue(trial.check(base))


def fake_job(sa, faces=None, co=None, loss=1.0):
    job = types.SimpleNamespace(mirror=None, outside=None, guide_lines=[], strips=None, methods={}, method="AUTO", context=None,
                                faces=list(faces_of(sa.indptr, sa.flat)) if faces is None else faces, co=sa.points if co is None else co,
                                detail=5.0, _stopped=False, guide_edges=(), instances=None, stage="", looks=[])

    def look(points, out_faces, *view):
        job.looks.append(len(points))
        return (loss, 0.0, 0.0)
        yield

    job._look = look
    job._view = (np.zeros(3), 1.0, None, None, 0.01)
    return job


class Job(unittest.TestCase):
    def tearDown(self):
        uvkeep.OVERRIDE.clear()
        uvauto.LAST.clear()

    def test_attach_only_in_light(self):
        sa, plan, _p, _f = setup()
        uvkeep.OVERRIDE.update(mode="off")
        self.assertIsNone(uvauto.attach(fake_job(sa)))
        uvkeep.OVERRIDE.update(mode="light")
        a = uvauto.attach(fake_job(sa))
        self.assertIsNotNone(a)
        self.assertEqual(len(a.lines), len(a.soft))
        job = fake_job(sa)
        job.mirror = (0, 0.0, 1.0, 1, 0)
        self.assertIsNone(uvauto.attach(job))

    def test_remap_finds_the_cut_edges_by_position(self):
        sa, plan, p, faces = setup()
        uvkeep.OVERRIDE.update(mode="light")
        job = fake_job(sa)
        job.uv = uvauto.attach(job)
        job.uv.edges = p[np.array([[0, 1], [1, 2]])]
        order = np.arange(len(p))[::-1]  # the coarse mesh renumbered
        coarse = MeshData(p[order], [tuple(int(np.flatnonzero(order == v)[0]) for v in f) for f in faces], ["0"] * len(faces))
        got = uvauto.remap(job, coarse)
        want = {tuple(sorted((int(np.flatnonzero(order == a)[0]), int(np.flatnonzero(order == b)[0])))) for a, b in ((0, 1), (1, 2))}
        self.assertEqual(set(got), want)
        job.uv = None
        self.assertEqual(uvauto.remap(job, coarse), ())

    def test_fix_inside_the_job_keeps_within_the_look(self):
        sa, plan, p, faces = setup()
        uvkeep.OVERRIDE.update(mode="light")
        job = fake_job(sa)
        job.uv = uvauto.attach(job)
        job.uv_budget = 2.0
        out, looks = T.finish(uvauto.fix(job, p, faces, (1.0, 0.0, 0.0)))
        self.assertFalse(np.array_equal(out, p))
        self.assertEqual(job.uv.report["fixed"], 1)
        job.uv_budget = 0.5  # the look before is already over: nothing kept
        out, looks = T.finish(uvauto.fix(job, p, faces, (1.0, 0.0, 0.0)))
        np.testing.assert_array_equal(out, p)
        self.assertIn(job.uv.report["reason"], ("look", "no better"))  # halved until nothing moves


class AutoFixPass(unittest.TestCase):
    def test_uv_pass_is_planned_and_gone_when_stripped(self):
        import importlib.util
        from pathlib import Path

        names = [cls.__name__ for cls, _k in autofix.plan(set(autofix.GROUPS))]
        self.assertIn("SeamSnap", names)  # the UV row runs with the other passes (it finds spots only in Light)
        self.assertEqual(names[-1], "PreviewTidy")  # slivers stay last
        text = Path(autofix.__file__).read_text(encoding="utf-8")
        mark = "main-only" + ": uv"
        stripped = "\n".join(line for line in text.split("\n") if not line.rstrip().endswith(mark))
        self.assertNotIn(mark, stripped)
        spec = importlib.util.spec_from_loader("triadq._autofix_site", loader=None)
        site = importlib.util.module_from_spec(spec)
        site.__package__ = "triadq"
        import sys

        sys.modules[spec.name] = site  # dataclasses look their module up
        try:
            exec(compile(stripped, "autofix_site.py", "exec"), site.__dict__)
        finally:
            sys.modules.pop(spec.name, None)
        site_names = [cls.__name__ for cls, _k in site.plan(set(site.GROUPS))]
        self.assertNotIn("SeamSnap", site_names)
        self.assertEqual(site_names, [n for n in names if n != "SeamSnap"])

    def test_seam_snap_made_by_the_plan_uses_the_last_search(self):
        sa, plan, p, faces = setup()
        uvauto.LAST.clear()
        self.assertEqual(T.finish(uvauto.SeamSnap().spots(T.make_base(p, faces))), [])  # no search followed seams: no spots
        uvauto.LAST.update(seams=plan.seams, xyz=uvauto.uvsnap.chain_xyz(plan.seams, sa.points))
        try:
            self.assertEqual(len(T.finish(uvauto.SeamSnap(use_fast=False).spots(T.make_base(p, faces)))), 1)
        finally:
            uvauto.LAST.clear()


if __name__ == "__main__":
    unittest.main()
