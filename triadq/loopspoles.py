# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Three Tidy Tools for strips and poles, as `trial.Trial` subclasses (plain NumPy, no bpy).

- Even the Band: the picked strip and the strips beside it, out to the next pole or border on each side, get the same
  width. Every rung of the strip is carried straight on across its neighbours (`strips.rung_lines`) and the points on
  that line are spaced evenly by arc length, easing in over the first and last strip so the band's edge does not kink.
  Points only slide along those lines, so no quad turns or changes neighbours.
- Cancel Pole Pair: pick a pole, then one of the poles of the other kind near it (a 3 for a 5, a 5 for a 3); edges are
  spun inside their two quads (Tarini et al. 2010's edge rotation), walking the pair together the way Tidy Up's Fewer
  Poles walks them (Peng et al. 2011 move singularity pairs), until both have four edges. Every spin is relaxed and
  checked like Tidy Up's (`fewerpoles.PoleCleaner`); when no walk within reach works, nothing changes. Same faces.
- Open Up Poles: the spokes of each pole are laid in its tangent plane, turned to even angles of 360/k degrees and equal
  lengths (an angle-scaled polar map of its one-ring, as in Schmidt, Grimm and Wyvill 2006), and put back on your original mesh; the rings round it ease the
  change out. Repeats while the narrowest angle still opens (Keep going), so the dial's 1 is as open as it gets.
"""

from __future__ import annotations

import math

import numpy as np

from . import strips as S
from .organize import MOST_DRIFT
from .tidytools import NOTHING_FREE
from .trial import Base, Candidate, Trial, edges_of

NO_POLES = "No poles in this area."
NO_PARTNER = "No pole within reach can cancel with this one."
BAD_WALK = "The walk to that pole would make a bad quad, so nothing changed. Try the next one."
NO_NEIGHBOURS = "This strip has no neighbours to even out with before a pole or border."
MIRRORED = "Cancel Pole Pair does not work on a mirrored result yet: its two halves would stop matching."
CHUNK = 48  # poles worked on between two yields (a few milliseconds)
STEP = 0.5  # share of the full opening taken per round, and most move per round in edges
OPEN_GAIN = 0.01  # degrees: a round of Open Up Poles that opens the narrowest angles (mean) less than this ends it


def _onto(base: Base, points: np.ndarray) -> np.ndarray:
    """Points back on your original mesh, the same side of it (organize._onto): a nearest point further than an edge
    away is another part or the far side of a thin one, and the point stays."""
    if base.project is None or not len(points):
        return points
    from .organize import _onto as onto

    return onto(base.project, np.asarray(points, dtype=np.float64), np.full(len(points), 0.5 * base.edge))


def _mirror(base: Base, co, side: float = 1.0):
    """strips.finish's mirror argument for a mirrored result, else None."""
    if base.mirror is None:
        return None
    axis, offset = base.mirror
    return (axis, offset, S.partners(co, axis, offset))


def corner_faces(faces, count: int):
    """(order, starts): the faces at corner v are order[starts[v]:starts[v + 1]]."""
    fv = np.fromiter((v for f in faces for v in f), dtype=np.int64)
    fi = np.repeat(np.arange(len(faces)), [len(f) for f in faces])
    order = np.argsort(fv, kind="stable")
    starts = np.searchsorted(fv[order], np.arange(count + 1))
    return fi[order], starts


def rings_of(faces, count: int, seeds, rings: int, allowed=None) -> tuple[np.ndarray, np.ndarray]:
    """(faces within `rings` face rings of the seed corners, ring number of every corner reached, -1 elsewhere).
    `allowed`: bool per face; others are not crossed."""
    order, starts = corner_faces(faces, count)
    ring = np.full(count, -1, dtype=np.int64)
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    ring[seeds] = 0
    taken = np.zeros(len(faces), dtype=bool)
    frontier = seeds
    for r in range(1, rings + 1):
        found = np.unique(np.concatenate([order[starts[v] : starts[v + 1]] for v in frontier.tolist()] or [np.zeros(0, np.int64)]))
        if allowed is not None:
            found = found[allowed[found]]
        found = found[~taken[found]]
        if not len(found):
            break
        taken[found] = True
        corners = np.unique(np.concatenate([np.asarray(faces[f]) for f in found.tolist()]))
        corners = corners[ring[corners] < 0]
        ring[corners] = r
        frontier = corners
    return np.flatnonzero(taken), ring


# --------------------------------------------------------------------------- Even the Band


class EvenTheBand(Trial):
    name = "Even the Band"
    group = "LOOPS"
    handles = ("strength",)
    defaults = {"strength": 1.0, "strip": -1}
    blendable = True
    stages = ("spread", "move", "look")
    picks = ("strip",)
    asks = {"strip": "click a strip"}

    def check(self, base: Base) -> str:
        return "" if base.free.any() else NOTHING_FREE

    def _strips(self, base: Base) -> S.Strips:
        if "strips" not in base.cache:
            base.cache["strips"] = S.Strips(base.co, base.faces)
        return base.cache["strips"]

    def pick_at(self, base: Base, settings: dict, kind: str, face: int, point) -> int:
        """The strip running along the quad edge nearest the click."""
        if face < 0 or face >= len(base.faces) or len(base.faces[face]) != 4 or not base.scope[face]:
            return -1
        p = base.co[list(base.faces[face])]
        q = np.roll(p, -1, axis=0)
        at = np.asarray(point, dtype=np.float64)
        t = np.clip(((at - p) * (q - p)).sum(axis=1) / np.maximum(((q - p) ** 2).sum(axis=1), 1e-30), 0.0, 1.0)
        k = int(np.argmin(np.linalg.norm(p + t[:, None] * (q - p) - at, axis=1)))
        return self._strips(base).element(face, k + 1)

    def _lines(self, base: Base, element: int):
        key = ("band", element)
        if key not in base.cache:
            strip = self._strips(base).walk(element)
            if "go" not in base.cache:
                from .loopscore import _continuations

                base.cache["go"] = _continuations(base.topo)
            lines = S.rung_lines(strip, base.cache["go"], len(base.co), base.stay, reach=len(base.co))
            base.cache[key] = (strip, [(ids, ia) for ids, ia in lines if len(ids) >= 3 and not base.stay[ids[1:-1]].all()])
        return base.cache[key]

    def picked_lines(self, base: Base, settings: dict) -> np.ndarray:
        element = int(settings.get("strip", -1))
        if element < 0:
            return np.zeros((0, 2, 3))
        strip, lines = self._lines(base, element)
        out = []
        for side in (strip["a"], strip["b"]):
            p = base.co[side]
            if strip["closed"]:
                p = np.vstack((p, p[:1]))
            out.append(np.stack((p[:-1], p[1:]), axis=1))
        if lines:  # the band's two outer edges: the ends of the rung lines
            for end in (0, -1):
                p = base.co[[ids[end] for ids, _ia in lines]]
                if len(p) > 1:
                    out.append(np.stack((p[:-1], p[1:]), axis=1))
        return np.concatenate(out) if out else np.zeros((0, 2, 3))

    def run(self, base: Base, settings: dict, rough: bool = False):
        element = int(settings.get("strip", -1))
        strength = float(settings.get("strength", 1.0))
        strip, lines = self._lines(base, element)
        yield 0.3
        if not lines:
            return Candidate(None, refusal=NO_NEIGHBOURS)
        co = base.co
        total, hits = np.zeros_like(co), np.zeros(len(co))
        for ids, _ia in lines:
            p = co[ids]
            s = S._arc(p)
            m = len(ids)
            even = np.linspace(s[0], s[-1], m)
            if m > 3:  # ease in over the first and last strip, so the band's edge is not kinked
                even[1] = 0.5 * (even[1] + s[1])
                even[-2] = 0.5 * (even[-2] + s[-2])
            at = s + strength * (even - s)
            inner = ids[1:-1]
            np.add.at(total, inner, S._along(p, s, at[1:-1]))
            np.add.at(hits, inner, 1.0)
        yield 0.6
        new = co.copy()
        moved = (hits > 0) & base.free
        new[moved] = total[moved] / hits[moved, None]
        side = float(np.sign(np.mean(co[strip["a"] + strip["b"], base.mirror[0]]) - base.mirror[1]) or 1.0) if base.mirror else 1.0
        new = S.finish(co, new, base.faces, lambda x: _onto(base, x), _mirror(base, co), side)
        new[base.stay] = co[base.stay]
        a, b = spread(co, lines), spread(new, lines)
        cand = Candidate(new, strength=strength, exact=True, note=f"{len(lines)} rungs across {_strips_across(lines)} strips")
        cand.numbers["spread"] = (f"width spread {100 * a:.0f}% -> {100 * b:.0f}%", 1 if b < a - 0.005 else -1 if b > a + 0.005 else 0)
        return cand


def spread(co, lines) -> float:
    """How unequal the strips of a band are: per rung line, the spread of its step lengths over their mean, averaged."""
    out = []
    for ids, _ia in lines:
        step = np.linalg.norm(np.diff(np.asarray(co)[ids], axis=0), axis=1)
        if len(step) > 1 and step.mean() > 0:
            out.append(step.std() / step.mean())
    return float(np.mean(out)) if out else 0.0


def _strips_across(lines) -> int:
    return int(np.median([len(ids) - 1 for ids, _ia in lines])) if lines else 0


# --------------------------------------------------------------------------- Open Up Poles


def _fan(faces, co, v, order, starts):
    """The corners round `v` in turn (one spoke per quad), or None when they don't close round it (a border)."""
    after = {}
    for f in order[starts[v] : starts[v + 1]].tolist():
        face = faces[f]
        if len(face) != 4:
            return None
        i = face.index(v)
        after[face[(i + 1) % 4]] = face[(i - 1) % 4]
    if not after:
        return None
    start = next(iter(after))
    out, cur = [start], after[start]
    while cur != start:
        if cur not in after or len(out) > len(after):
            return None
        out.append(cur)
        cur = after[cur]
    return out if len(out) == len(after) else None


def _narrowest(co, v, fan) -> float:
    d = co[fan] - co[v]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    cos = (d * np.roll(d, -1, axis=0)).sum(axis=1)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).min())


def _mean_narrowest(co, fans) -> float:
    """The narrowest angle at each pole, averaged over the poles."""
    return float(np.mean([_narrowest(co, v, fan) for v, fan in fans]))


class OpenUpPoles(Trial):
    name = "Open Up Poles"
    group = "POLES"
    handles = ("strength",)
    defaults = {"strength": 0.8, "rings": 2}
    blendable = True
    stages = ("narrow", "move", "look")

    def _poles(self, base: Base):
        if "fans" not in base.cache:
            order, starts = corner_faces(base.faces, len(base.co))
            topo = base.topo
            fans = []
            for v in np.flatnonzero(topo.poles & ~topo.border & base.free).tolist():
                fan = _fan(base.faces, base.co, v, order, starts)
                if fan is not None and len(fan) >= 3 and all(base.scope[f] for f in order[starts[v] : starts[v + 1]].tolist()):
                    fans.append((v, fan))
            base.cache["fans"] = fans
        return base.cache["fans"]

    def check(self, base: Base) -> str:
        return "" if self._poles(base) else NO_POLES

    def _round(self, base: Base, co: np.ndarray, fans, rings: int, edges):
        """Generator: one opening of every fan, returning the move of every corner (spokes, then the rings round them eased)."""
        move, hits = np.zeros_like(co), np.zeros(len(co))
        for i, (v, fan) in enumerate(fans):
            if i % CHUNK == CHUNK - 1:
                yield
            k = len(fan)
            d = co[fan] - co[v]
            n = np.cross(d, np.roll(d, -1, axis=0)).sum(axis=0)
            n /= max(float(np.linalg.norm(n)), 1e-30)
            t = d - np.outer(d @ n, n)
            e1 = t[0] / max(float(np.linalg.norm(t[0])), 1e-30)
            e2 = np.cross(n, e1)
            theta = np.arctan2(t @ e2, t @ e1)
            gaps = np.mod(np.diff(np.append(theta, theta[0])), 2.0 * math.pi)
            if abs(gaps.sum() - 2.0 * math.pi) > 0.5:  # the fan folds over itself in its plane: leave it
                continue
            theta = theta[0] + np.concatenate(([0.0], np.cumsum(gaps[:-1]))) * (2.0 * math.pi / gaps.sum())
            want = 2.0 * math.pi * np.arange(k) / k
            turn = math.atan2(np.sin(theta - want).mean(), np.cos(theta - want).mean())  # keep the fan's own turn
            length = float(np.linalg.norm(d, axis=1).mean())
            goal = co[v] + length * (np.outer(np.cos(want + turn), e1) + np.outer(np.sin(want + turn), e2))
            np.add.at(move, fan, goal - co[fan])
            np.add.at(hits, fan, 1.0)
        got = hits > 0
        move[got] /= hits[got, None]
        ring = np.where(got, 1, -1)
        for v, _fan_ in fans:
            ring[v] = 0
        a, b = edges[:, 0], edges[:, 1]
        for r in range(2, rings + 1):  # each further ring takes half the move of the ring inside it
            inner_a, inner_b = ring[a] == r - 1, ring[b] == r - 1
            out_a, out_b = ring[a] < 0, ring[b] < 0
            total, count = np.zeros_like(co), np.zeros(len(co))
            for src, dst, ok in ((a, b, inner_a & out_b), (b, a, inner_b & out_a)):
                np.add.at(total, dst[ok], move[src[ok]])
                np.add.at(count, dst[ok], 1.0)
            fresh = count > 0
            if not fresh.any():
                break
            move[fresh] = 0.5 * total[fresh] / count[fresh, None]
            ring[fresh] = r
        move[~base.free] = 0.0
        return move

    def run(self, base: Base, settings: dict, rough: bool = False):
        strength = float(settings.get("strength", self.defaults["strength"]))
        rings = max(1, int(settings.get("rings", self.defaults["rings"])))
        fans = self._poles(base)
        edges = edges_of(base.faces)
        co = base.co.copy()
        mirror = _mirror(base, base.co)
        project = lambda x: _onto(base, x)  # noqa: E731
        fan_of = [np.asarray([v] + fan) for v, fan in fans]
        cur = np.empty(len(fans))
        for i, (v, fan) in enumerate(fans):
            cur[i] = _narrowest(co, v, fan)
            if i % CHUNK == CHUNK - 1:
                yield
        before = last = float(cur.mean())
        while True:  # Keep going: rounds go on while the poles still open up (Esc stops between rounds)
            move = yield from self._round(base, co, fans, rings, edges)
            new = S.capped(co, co + STEP * move, base.faces, STEP)
            yield
            new = S.capped(base.co, new, base.faces, MOST_DRIFT)
            yield
            new = S.finish(co, new, base.faces, project, mirror)
            yield
            new[base.stay] = base.co[base.stay]
            # Neighbouring poles pull the same corners different ways: a pole whose fan got narrower keeps its fan.
            nn = np.empty(len(fans))
            for i, (v, fan) in enumerate(fans):
                nn[i] = _narrowest(new, v, fan)
                if i % CHUNK == CHUNK - 1:
                    yield
            worse = np.flatnonzero(nn < cur - 0.1)
            if len(worse):
                back = np.concatenate([fan_of[i] for i in worse.tolist()])
                new[back] = co[back]
                nn[worse] = cur[worse]
            now = float(nn.mean())
            yield min(0.95, 0.3 + 0.05 * (now - before))
            if now < last + OPEN_GAIN:
                if now > last:
                    co, last, cur = new, now, nn
                break
            co, last, cur = new, now, nn
            if rough:
                break
        if strength < 1.0:  # the dial: part of the way to the fully opened fans, back on your original mesh
            co = base.co + strength * (co - base.co)
            moved = np.flatnonzero(np.any(co != base.co, axis=1))
            co[moved] = project(co[moved])
            last = _mean_narrowest(co, fans)
        cand = Candidate(co, strength=strength, exact=not rough, note=f"{len(fans)} poles")
        cand.numbers["narrow"] = (f"narrowest angle {before:.0f} -> {last:.0f} degrees" + (" (mean)" if len(fans) > 1 else ""), 1 if last > before + 0.5 else -1 if last < before - 0.5 else 0)
        return cand


# --------------------------------------------------------------------------- Cancel Pole Pair


class CancelPolePair(Trial):
    name = "Cancel Pole Pair"
    group = "POLES"
    handles = ()
    defaults = {"pole": -1, "partner": -1, "reach": 6, "steps": 6, "tries": 400}
    blendable = False
    layers = ("scope", "ghost", "arrows", "poles", "held")
    stages = ("poles", "faces", "look")
    picks = ("pole", "partner")
    asks = {"pole": "click a pole", "partner": "click one of the numbered poles"}

    def _signs(self, base: Base) -> np.ndarray:
        """Per corner: -1 fewer than four edges, 1 more, 0 four or held (borders, hard edges, the scope's edge)."""
        if "signs" not in base.cache:
            topo = base.topo
            sign = np.sign(topo.code.astype(np.int64) - 4)
            sign[~(topo.poles & ~topo.border & base.free)] = 0
            base.cache["signs"] = sign
        return base.cache["signs"]

    def check(self, base: Base) -> str:
        if base.mirror is not None:
            return MIRRORED
        return "" if (self._signs(base) != 0).any() else NO_POLES

    def _partners(self, base: Base, pole: int) -> np.ndarray:
        """Poles of the other kind within reach of `pole`, nearest first."""
        key = ("partners", pole)
        if key not in base.cache:
            sign = self._signs(base)
            _faces, ring = rings_of(base.faces, len(base.co), [pole], int(base.cache.get("reach", 6)), base.scope)
            near = np.flatnonzero((ring > 0) & (sign == -sign[pole]))
            far = np.linalg.norm(base.co[near] - base.co[pole], axis=1)
            base.cache[key] = near[np.lexsort((far, ring[near]))]
        return base.cache[key]

    def targets(self, base: Base, settings: dict, kind: str):
        base.cache["reach"] = max(1, int(settings.get("reach", self.defaults["reach"])))
        if kind == "pole":
            ids = np.flatnonzero(self._signs(base) != 0)
            return ids, [str(int(base.topo.code[v])) for v in ids.tolist()]
        pole = int(settings.get("pole", -1))
        ids = self._partners(base, pole) if pole >= 0 else np.zeros(0, np.int64)
        return ids, [str(i + 1) for i in range(len(ids))]

    def picked_lines(self, base: Base, settings: dict) -> np.ndarray:
        out = []
        for kind in self.picks:
            v = int(settings.get(kind, -1))
            if v < 0:
                continue
            ends = np.unique(np.asarray([u for f in base.faces if v in f for u in (f[f.index(v) - 1], f[(f.index(v) + 1) % len(f)])]))
            out += [(base.co[v], base.co[u]) for u in ends.tolist()]
        return np.asarray(out, dtype=np.float64).reshape(-1, 2, 3)

    def run(self, base: Base, settings: dict, rough: bool = False):
        from .chords import ChordReducer
        from .fewerpoles import PoleCleaner
        from .types import MeshData

        v, w = int(settings["pole"]), int(settings["partner"])
        reach = max(1, int(settings.get("reach", self.defaults["reach"])))
        sign = self._signs(base)
        if v < 0 or w < 0 or sign[v] == 0 or sign[w] != -sign[v]:
            return Candidate(None, refusal=NO_PARTNER)
        # The walk works on a patch round the two: its edge stays where it is, so nothing outside can change.
        patch, _ring = rings_of(base.faces, len(base.co), [v, w], reach + 2, base.scope)
        used = np.unique(np.concatenate([np.asarray(base.faces[f]) for f in patch.tolist()]))
        local = np.full(len(base.co), -1, dtype=np.int64)
        local[used] = np.arange(len(used))
        faces = [tuple(int(local[u]) for u in base.faces[f]) for f in patch.tolist()]
        yield 0.05
        r = ChordReducer(MeshData(base.co[used], faces, [0] * len(faces)))
        edge = np.ones(len(used), dtype=bool)
        # A corner of the patch is free only if every face round it is in the patch.
        order, starts = corner_faces(base.faces, len(base.co))
        in_patch = np.zeros(len(base.faces), dtype=bool)
        in_patch[patch] = True
        for i, u in enumerate(used.tolist()):
            edge[i] = not in_patch[order[starts[u] : starts[u + 1]]].all()
        for i in np.flatnonzero(edge | base.stay[used]).tolist():
            r.locked[i] = True
        cleaner = PoleCleaner(r)
        yield 0.1
        lv, lw = int(local[v]), int(local[w])
        budget = [max(1, int(settings.get("tries", self.defaults["tries"])))]
        found = False
        for depth in range(1, max(1, int(settings.get("steps", self.defaults["steps"]))) + 1):
            found = yield from _walk(cleaner, {lv, lw}, lv, lw, depth, 0, set(), budget)
            if found or budget[0] <= 0:
                break
        if not found:
            return Candidate(None, refusal=BAD_WALK)
        co = base.co.copy()
        pts = np.asarray(r.points, dtype=np.float64)[: len(used)]
        moved = np.flatnonzero(np.any(np.abs(pts - base.co[used]) > 0.0, axis=1))
        co[used[moved]] = _onto(base, pts[moved])
        new = list(base.faces)
        for i, f in enumerate(patch.tolist()):
            new[f] = tuple(int(used[u]) for u in r.faces[i])
        yield 0.95
        return Candidate(co, faces=new, parent=np.arange(len(new)), strength=1.0, exact=True, note=f"walk of {cleaner.walked or 1} step{'s' if cleaner.walked > 1 else ''}")


def _spins_at(r, u: int):
    """Every edge spin that changes the edges at `u`: (energy change, f, g, first, second, ends)."""
    seen = set()
    for other in list(r._around(u)):
        for o in r._turnings(u, other):
            key = (o[1], o[2], tuple(o[3]))
            if key not in seen:
                seen.add(key)
                yield o
    for f in list(r.users[u]):
        face = r.faces[f]
        for k in range(len(face)):
            a, b = face[k], face[(k + 1) % len(face)]
            if u in (a, b):
                continue
            for o in r._turnings(a, b):
                key = (o[1], o[2], tuple(o[3]))
                if u in o[5][2:] and key not in seen:
                    seen.add(key)
                    yield o


def _walk(cleaner, active: set, v: int, w: int, depth: int, debt: int, seen: set, budget: list):
    """Generator: spins round the poles in play, `depth` deep, the pole energy never above where it started, until `v`
    and `w` both have four edges and the energy went down. Spins of a walk that ends elsewhere are undone."""
    r = cleaner.r
    goal = [0.5 * (r.points[v][k] + r.points[w][k]) for k in range(3)]
    options = []
    for u in active:
        if len(r.users[u]) == 4:
            continue
        for o in _spins_at(r, u):
            if any(r.locked[e] for e in o[5]):
                continue
            near = min(math.dist(r.points[e], goal) for e in o[5])
            options.append((o[0], near, o))
    options.sort(key=lambda x: (x[0], x[1]))
    for change, _near, o in options:
        if debt + change > 0:
            break
        key = (o[1], o[2], tuple(o[3]))
        if key in seen:
            continue
        seen.add(key)
        budget[0] -= 1
        if budget[0] < 0:
            return False
        edit = cleaner._apply("spin", o, max(-change, 1))
        yield
        if edit is None:
            continue
        if len(r.users[v]) == 4 and len(r.users[w]) == 4 and debt + change < 0:
            cleaner.walked += 1
            return True
        if depth > 1:
            got = yield from _walk(cleaner, active | set(o[5]), v, w, depth - 1, debt + change, seen, budget)
            if got:
                cleaner.walked += 1
                return True
        edit.undo()
    return False


__all__ = ["EvenTheBand", "OpenUpPoles", "CancelPolePair", "spread", "rings_of", "corner_faces"]
