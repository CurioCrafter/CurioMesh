# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Auto Fix core (bench/tools_design/AUTO_FIX.md 6.2): toy tools on a 6x6 grid with two bent spots."""
import unittest

import numpy as np

from triadq import autofix, trial
from triadq.trial import Candidate, Spot, Trial

N = 6


def _grid(n: int = N):
    points = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    return points, faces


def _v(x, y, n=N):
    return y * (n + 1) + x


def _around(x, y, n=N):
    """The four faces round corner (x, y)."""
    return np.array([(y - 1 + j) * n + (x - 1 + i) for j in (0, 1) for i in (0, 1)])


def _flat(points):
    out = np.array(points, dtype=np.float64)
    out[:, 2] = 0.0
    return out


def _inner(faces, ids):
    """Corners whose every face is in `ids`."""
    inside = set(np.asarray(ids).tolist())
    count, hit = {}, {}
    for f, face in enumerate(faces):
        for v in face:
            count[v] = count.get(v, 0) + 1
            hit[v] = hit.get(v, 0) + (f in inside)
    return [v for v in count if hit[v] == count[v]]


def _deviation(co, faces, corners) -> float:
    near = {}
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            near.setdefault(a, set()).add(b)
            near.setdefault(b, set()).add(a)
    return float(sum(np.linalg.norm(co[v] - co[list(near[v])].mean(axis=0)) for v in corners))


class Even(Trial):
    """Toy positions tool: each free corner to the middle of its neighbours."""

    name = "Even"
    defaults = {"strength": 1.0}
    stages = ("move", "look")
    kinds = (("bent", "bent corner", "bent corners"),)
    own = "bent"
    least = 1e-6

    def spots(self, base):
        out = []
        for v in np.flatnonzero(np.linalg.norm(base.co[:, :2] - np.round(base.co[:, :2]), axis=1) > 1e-3).tolist():
            x, y = np.round(base.co[v, :2]).astype(int)
            ids = np.array([f for f, face in enumerate(base.faces) if v in face])
            out.append(Spot("bent", ids, base.co[v].copy(), _deviation(base.co, base.faces, [v])))
        yield 1.0
        return out

    def own_number(self, base, co, faces, spot):
        return _deviation(co, faces, _inner(faces, spot.faces))

    def run(self, base, settings, rough=False):
        co = base.co.copy()
        free = np.flatnonzero(base.free)
        near = {}
        for f in np.flatnonzero(base.scope).tolist():
            face = base.faces[f]
            for a, b in zip(face, face[1:] + face[:1]):
                near.setdefault(a, set()).add(b)
                near.setdefault(b, set()).add(a)
        for v in free.tolist():
            co[v] = co[list(near[v])].mean(axis=0)
        yield 0.5
        s = float(settings["strength"])
        return Candidate(base.co + s * (co - base.co), strength=s, exact=not rough)


class Flip(Even):
    """Toy that turns its faces over: every result has bad faces."""

    def run(self, base, settings, rough=False):
        co = base.co.copy()
        for v in np.flatnonzero(base.free).tolist():
            co[v] = co[v] + np.array([3.0, 3.0, 0.0])  # far past its neighbours: the faces round it turn over
        yield 0.5
        return Candidate(co)


class Merge(Trial):
    """Toy faces tool: a triangle pair in scope becomes one quad again, unless its halves are of two kinds."""

    name = "Merge"
    handles = ()
    defaults = {}
    blendable = False
    stages = ("faces", "poles")
    kinds = (("pair", "triangle pair", "triangle pairs"),)
    own = "triangles"
    changes_faces = True

    def spots(self, base):
        tris = [f for f, face in enumerate(base.faces) if len(face) == 3]
        out, used = [], set()
        for f in tris:
            if f in used:
                continue
            g = next((g for g in tris if g != f and g not in used and len(set(base.faces[f]) & set(base.faces[g])) == 2), None)
            if g is not None:
                used |= {f, g}
                out.append(Spot("pair", np.array([f, g]), base.co[list(base.faces[f])].mean(axis=0), 2.0))
        yield 1.0
        return out

    def own_number(self, base, co, faces, spot):
        return float(sum(len(faces[f]) == 3 for f in spot.faces))

    def run(self, base, settings, rough=False):
        faces, parent = list(base.faces), list(range(len(base.faces)))
        tris = [f for f in np.flatnonzero(base.scope).tolist() if len(faces[f]) == 3]
        yield 0.5
        if len(tris) == 2:
            f, g = tris
            a, b, c = faces[f]
            d = next(v for v in faces[g] if v not in faces[f])
            if base.kinds is None or base.kinds[f] == base.kinds[g]:  # never joins faces of two kinds
                quad = (a, b, c, d) if {a, c} <= set(faces[g]) else (b, c, a, d) if {b, a} <= set(faces[g]) else (c, a, b, d)
                faces = [face for k, face in enumerate(faces) if k != g]
                parent = [k for k in parent if k != g]
                faces[f if f < g else f - 1] = quad
        return Candidate(base.co.copy(), faces=faces, parent=np.array(parent))


def _bent(stay_b=True, mirror=None):
    """6x6 grid, corner (1, 1) and corner (4, 4) bent; (4, 4) held so it cannot be fixed."""
    co, faces = _grid()
    co[_v(1, 1)] += (0.3, 0.2, 0.0)
    co[_v(4, 4)] += (-0.25, 0.3, 0.0)
    stay = np.zeros(len(co), dtype=bool)
    if stay_b:
        stay[_v(4, 4)] = True
    return trial.make_base(co, faces, stay=stay, project=_flat, mirror=mirror)


def _split(co, faces, quads):
    """Each quad in `quads` cut in two triangles."""
    out = []
    for k, face in enumerate(faces):
        if k in quads:
            a, b, c, d = face
            out += [(a, b, c), (a, c, d)]
        else:
            out.append(face)
    return out


class ProposeTest(unittest.TestCase):
    def test_spot_fields_and_trial_defaults(self):
        t = Trial()
        self.assertEqual((t.kinds, t.auto, t.own, t.least, t.changes_faces), ((), True, "", 0.0, False))
        base = _bent()
        self.assertEqual(trial.finish(t.spots(base)), [])
        sp = Spot("x", np.array([0]), np.zeros(3), 2.5)
        self.assertEqual(sp.pick, {})
        self.assertEqual(t.own_number(base, base.co, base.faces, sp), 2.5)

    def test_one_spot_fixed_one_put_back(self):
        base = _bent()
        tool = Even()
        spots = trial.finish(tool.spots(base))
        self.assertEqual(len(spots), 2)
        p = trial.finish(autofix.propose(tool, base, {}, spots=spots))
        self.assertEqual(p.reason, "")
        self.assertEqual([sp.kind for sp in p.fixed], ["bent"])
        self.assertEqual(len(p.left), 1)
        self.assertEqual(p.left[0][1], "no better")
        self.assertTrue(np.allclose(p.cand.co[_v(1, 1)], (1.0, 1.0, 0.0)))  # fixed
        self.assertTrue(np.allclose(p.cand.co[_v(4, 4)], base.co[_v(4, 4)]))  # put back
        untouched = [v for v in range(len(base.co)) if v != _v(1, 1)]
        self.assertTrue(np.allclose(p.cand.co[untouched], base.co[untouched]))  # the spots' edges were held
        self.assertEqual(len(p.cand.found), 2)
        self.assertEqual(len(p.cand.held), 1)
        self.assertIn("1 spot", p.words)
        self.assertIsNotNone(p.after.tidy)
        self.assertIsNotNone(p.after.loops)
        self.assertEqual(p.after.bad, 0)

    def test_manual_path_is_the_tool_on_its_scope(self):
        base = _bent(stay_b=False)
        p = trial.finish(autofix.propose(Even(), base, {}))
        self.assertFalse(p.judged)
        self.assertEqual(p.reason, "")
        direct = trial.finish(Even().run(base, {"strength": 1.0}))
        self.assertTrue(np.allclose(p.cand.co, direct.co))  # same maths as the tool alone

    def test_mirrored_base_holds_the_plane(self):
        base = _bent(stay_b=False, mirror=(0, 4.0 - 0.25))  # corner (4, 4) sits on the mirror plane
        tool = Even()
        p = trial.finish(autofix.propose(tool, base, {}, spots=trial.finish(tool.spots(base))))
        self.assertEqual(len(p.fixed), 1)
        self.assertTrue(np.allclose(p.cand.co[_v(4, 4)], base.co[_v(4, 4)]))

    def test_faces_changing_with_rebase(self):
        co, faces = _grid()
        faces = _split(co, faces, {7, 28})
        kinds = np.zeros(len(faces), dtype=np.int64)
        kinds[30] = 1  # the second pair's halves are of two kinds: it cannot merge
        base = trial.make_base(co, faces, kinds=kinds)
        tool = Merge()
        spots = trial.finish(tool.spots(base))
        self.assertEqual(len(spots), 2)
        p = trial.finish(autofix.propose(tool, base, {}, spots=spots))
        self.assertEqual(p.reason, "")
        self.assertEqual(len(p.fixed), 1)
        self.assertEqual(p.left[0][1], "no better")
        self.assertEqual(len(p.cand.faces), len(faces) - 1)
        self.assertEqual(sum(len(f) == 3 for f in p.cand.faces), 2)
        self.assertEqual(len(p.cand.parent), len(p.cand.faces))
        self.assertEqual(p.after.bad, 0)

    def test_rebase_maps_scope_and_labels(self):
        co, faces = _grid(2)
        scope = np.array([True, False, True, True])
        base = trial.make_base(co, faces, scope=scope, labels={"piece": np.array([0, 1, 2, 3])}, kinds=np.array([5, 6, 7, 8]))
        new = trial.rebase(base, Candidate(co, faces=faces[1:], parent=np.array([1, 2, 3])))
        self.assertEqual(new.scope.tolist(), [False, True, True])
        self.assertEqual(new.labels["piece"].tolist(), [1, 2, 3])
        self.assertEqual(new.kinds.tolist(), [6, 7, 8])
        same = trial.rebase(base, Candidate(co + 0.1))
        self.assertTrue(np.allclose(same.co, co + 0.1))
        self.assertEqual(same.scope.tolist(), scope.tolist())


class VerdictTest(unittest.TestCase):
    def test_look_over_budget_halves_until_it_fits(self):
        base = _bent(stay_b=False)
        tool = Even()

        def look(co, faces):
            yield
            return float(100.0 * np.linalg.norm(co - base.co, axis=1).max())

        spots = trial.finish(tool.spots(base))
        p = trial.finish(autofix.propose(tool, base, {}, spots=spots, look=look, budget=lambda before: before + 20.0))
        self.assertEqual(p.reason, "")
        self.assertIsNotNone(p.held)
        self.assertLess(p.held, 1.0)
        self.assertLessEqual(p.after.look, 20.0 + 1e-6)

    def test_look_that_cannot_fit(self):
        base = _bent(stay_b=False)
        tool = Even()

        def look(co, faces):
            yield
            return 0.0 if np.allclose(co, base.co) else 50.0

        p = trial.finish(autofix.propose(tool, base, {}, spots=trial.finish(tool.spots(base)), look=look, budget=1.0))
        self.assertEqual(p.reason, "look")

    def test_new_bad_face_is_put_back(self):
        base = _bent(stay_b=False)
        tool = Flip()
        p = trial.finish(autofix.propose(tool, base, {}, spots=trial.finish(tool.spots(base))))
        self.assertEqual(p.reason, "no better")
        self.assertTrue(all(why in ("bad faces", "no better") for _sp, why in p.left))
        self.assertTrue(np.allclose(p.cand.co, base.co))
        manual = trial.finish(autofix.propose(tool, base, {}))
        self.assertEqual(autofix.verdict(tool, manual), "bad faces")

    def test_none_better(self):
        base = _bent()
        base.stay[_v(1, 1)] = True
        base.cache.clear()
        tool = Even()
        p = trial.finish(autofix.propose(tool, base, {}, spots=trial.finish(tool.spots(base))))
        self.assertEqual(p.reason, "no better")
        self.assertEqual(p.fixed, [])

    def test_worse_numbers(self):
        p = autofix.Proposal(Candidate(np.zeros((1, 3))), before=autofix.Scores(tidy=60, loops=50, poles=4, bad=0), after=autofix.Scores(tidy=58, loops=50, poles=4, bad=0))
        self.assertEqual(autofix.verdict(Trial(), p), "worse")
        p.after.tidy, p.after.poles = 60, 5
        self.assertEqual(autofix.verdict(Trial(), p), "worse")
        p.after.poles = 3
        self.assertEqual(autofix.verdict(Trial(), p), "")


class EdgeCaseTest(unittest.TestCase):
    def test_empty_mesh(self):
        base = trial.make_base(np.zeros((0, 3)), [])
        p = trial.finish(autofix.propose(Even(), base, {}, spots=[]))
        self.assertEqual(p.reason, "no better")
        cand = trial.finish(autofix.every(Even)().run(base, {}))
        self.assertTrue(cand.refusal)
        events = list(autofix.run(lambda: base, [(Even, None)]))
        self.assertEqual(events[-1].kind, "done")
        self.assertEqual(events[-1].words, autofix.NOTHING)

    def test_no_spots(self):
        co, faces = _grid()
        base = trial.make_base(co, faces)
        p = trial.finish(autofix.propose(autofix.EverySpot(Even()), base, {}))
        self.assertTrue(p.cand.refusal.startswith("No bent corners"))

    def test_scope_with_nothing_free(self):
        base = _bent(stay_b=False)
        spots = trial.finish(Even().spots(base))
        base.scope[:] = False
        base.cache.clear()
        p = trial.finish(autofix.propose(Even(), base, {}, spots=spots))
        self.assertEqual(p.reason, "no better")
        self.assertTrue(all(why == "held" for _sp, why in p.left))


class RunTest(unittest.TestCase):
    def _driver(self, base):
        state = {"base": base}

        def make():
            return state["base"]

        return state, make

    def test_rounds_until_nothing_helps(self):
        state, make = self._driver(_bent())
        kinds = []
        for event in autofix.run(make, [(Even, None)]):
            kinds.append(event.kind)
            if event.kind == "kept":
                state["base"] = trial.rebase(state["base"], event.proposal.cand)
        self.assertEqual(kinds.count("kept"), 1)
        self.assertIn("left", kinds)  # round 2: only the held spot is left, and nothing gets better
        self.assertEqual(kinds[-1], "done")
        self.assertTrue(np.allclose(state["base"].co[_v(1, 1)], (1.0, 1.0, 0.0)))

    def test_stop_between_passes_and_go_on(self):
        state, make = self._driver(_bent())
        events = list(autofix.run(make, [(Even, None), (Even, None)], stopped=lambda: "Time Limits"))
        self.assertEqual([e.kind for e in events], ["stopped"])
        self.assertEqual(events[0].step, 0)
        events = list(autofix.run(make, [(Even, None), (Even, None)], start=1))
        self.assertEqual(events[-1].kind, "done")
        self.assertTrue(any(e.kind == "kept" and e.step == 1 for e in events))

    def test_go_on_after_a_round_that_kept(self):
        """A stop cut a round that had kept a pass: the passes after it find nothing, yet a next round still runs."""
        state, make = self._driver(_bent())
        events = list(autofix.run(make, [(Even, None)], start=1, rnd=3, carried=True))
        self.assertEqual([(e.kind, e.round) for e in events if e.kind != "working"][0][1], 4)
        events = list(autofix.run(make, [(Even, None)], start=1, rnd=3))
        self.assertEqual([e.kind for e in events], ["done"])

    def test_a_round_that_only_trades_numbers_ends(self):
        S = autofix.Scores
        self.assertFalse(autofix._round_gained(S(40.0, 20.0, 100, 0), S(40.2, 20.1, 100, 0)))
        self.assertTrue(autofix._round_gained(S(40.0, 20.0, 100, 0), S(40.0, 20.0, 99, 0)))
        self.assertTrue(autofix._round_gained(S(40.0, 20.0, 100, 0), S(40.6, 20.0, 100, 0)))
        self.assertTrue(autofix._round_gained(S(), S()))  # nothing measured: rounds go on as before

    def test_every_spot_wrapper(self):
        base = _bent()
        cls = autofix.every(Even)
        tool = cls()
        self.assertIn(("keep_going", "Keep going"), tool.toggles)
        self.assertFalse(tool.auto)
        cand = trial.finish(tool.run(base, dict(tool.defaults)))
        self.assertTrue(np.allclose(cand.co[_v(1, 1)], (1.0, 1.0, 0.0)))
        p = trial.finish(autofix.propose(tool, base, dict(tool.defaults)))
        self.assertEqual(len(p.fixed), 1)

    def test_plan_follows_the_switches(self):
        self.assertEqual(autofix.plan(set()), [])
        for cls, _kinds in autofix.plan(set(autofix.GROUPS)):
            self.assertTrue(cls.auto)
            self.assertIsNot(cls.spots, Trial.spots)

    def test_bad_stage_counts_turned_faces(self):
        base = _bent(stay_b=False)
        cand = trial.finish(Flip().run(base, {}))
        trial.finish(trial.measure(base, cand, ("bad",)))
        self.assertGreater(cand.numbers["bad"][1], cand.numbers["bad"][0])


if __name__ == "__main__":
    unittest.main()
