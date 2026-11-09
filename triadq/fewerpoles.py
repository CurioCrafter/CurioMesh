# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Fewer poles on a finished quad mesh, at about the same faces: the "Fewer poles" part of Tidy Up.

Poles mostly come in pairs: a corner with three edges next to one with five is where a row of quads ends, and two such
pairs facing each other cancel. Three local changes do the work, each one only kept when the pole energy (the sum of
(edges - 4)^2 over the corners) goes down:

- spinning an edge inside its two quads (Tarini et al. 2010's edge rotation): its two ends lose an edge, the two
  corners it turns to gain one, so a 5-5 edge turned toward two 3s takes four poles away;
- collapsing a quad across a diagonal whose ends are both 3s: the merged corner has four edges, and the other two
  corners lose one each (Tarini et al. 2010's diagonal collapse), one quad fewer;
- walking a 3-5 pair along its row with spins that leave the energy as it is, a few steps at most, until it meets a
  pole it cancels with (Peng et al. 2011 move singularity pairs the same way); a walk that finds nothing is undone.

Before a change is checked, the corners whose edges changed are relaxed a few times toward their neighbours in their
own tangent plane, so a change is judged by the quads it leaves after a settle, not by the crooked ones it makes at
first (Tidy Up's settle puts every corner back on your mesh afterwards).
A change is kept only if no quad gets worse than `QUALITY` or turns past `MOST_TURNED`, no corner moves further
than `MOST_MOVE` of its edge length, and (with a camera judge) the view gets no worse than the pole trade allows.
Locked corners (hard edges, borders, mirror seam, guide curves) and anything across a material or part never move.

Works on a `chords.ChordReducer`, so it shares its faces, locks and judge. Plain Python.
"""

from __future__ import annotations

import math
from itertools import count

from .chords import TRADE
from .quadreduce import MOST_TURNED, _dot, _normal, _quality

QUALITY = 0.02  # no quad worse than this after the relax, on `patch.quad_quality`'s scale (the settle afterwards evens them out)
RELAX_STEPS = 3  # neighbour-average steps on the touched corners before a change is judged
MOST_MOVE = 0.75  # a corner may move this share of its mean edge length from where it started
WALK = 6  # spins in one pair walk
WALK_TRIES = 40  # spins tried in all from one pole before giving up
MOST_DEBT = 0  # how far the pole energy may rise part way through a walk
ROUNDS = 0  # 0: rounds go on until one finds nothing (each one that finds something takes pole energy away, so it ends)


class _Edit:
    """One change to the reducer that can be undone: faces set, faces dropped, corners moved."""

    def __init__(self, r):
        self.r = r
        self.faces: dict[int, list[int] | None] = {}
        self.points: dict[int, tuple] = {}
        self.dropped = 0

    def set_face(self, f: int, corners) -> None:
        r = self.r
        self.faces.setdefault(f, r.faces[f])
        old = r.faces[f]
        if old is not None:
            for v in old:
                r.users[v].discard(f)
        r.faces[f] = None if corners is None else list(corners)
        if corners is None:
            r.alive -= 1
            self.dropped += 1
        else:
            for v in corners:
                r.users[v].add(f)

    def move(self, v: int, p) -> None:
        self.points.setdefault(v, self.r.points[v])
        self.r.points[v] = tuple(p)

    def undo(self) -> None:
        r = self.r
        for v, p in self.points.items():
            r.points[v] = p
        for f, old in self.faces.items():
            now = r.faces[f]
            if now is not None:
                for v in now:
                    r.users[v].discard(f)
            r.faces[f] = old
            if old is not None:
                for v in old:
                    r.users[v].add(f)
        r.alive += self.dropped
        self.faces, self.points, self.dropped = {}, {}, 0


class PoleCleaner:
    """`run()` takes poles away from a `ChordReducer` in place."""

    def __init__(self, reducer):
        self.r = reducer
        self.start = list(reducer.points)
        self.gained = 0
        self.collapsed = 0
        self.walked = 0

    # Pieces

    def _cost(self, v: int, count: int) -> int:
        return self.r._faces_at_cost(v, count)

    def _free(self, v: int) -> bool:
        r = self.r
        return not r.locked[v] and not r.border[v]

    def _reach(self, v: int) -> float:
        r, p = self.r, self.r.points[v]
        near = r._around(v)
        if not near:
            return 0.0
        return MOST_MOVE * sum(math.dist(p, r.points[u]) for u in near) / len(near)

    def _relax(self, edit: _Edit, corners) -> bool:
        """Even out the corners around a change, in their tangent planes. False if one would have to move too far."""

        r = self.r
        free = [v for v in corners if v < len(r.users) and r.users[v] and self._free(v)]
        if not free:
            return True
        limit = {v: self._reach(v) for v in free}
        home = {v: self.start[v] if v < len(self.start) else r.points[v] for v in free}
        for _ in range(RELAX_STEPS):
            spots = []
            for v in free:
                near = r._around(v)
                p = r.points[v]
                mid = [sum(r.points[u][k] for u in near) / len(near) for k in range(3)]
                n = self._normal_at(v)
                step = [0.5 * (mid[k] - p[k]) for k in range(3)]
                h = _dot(step, n)
                spots.append([p[k] + step[k] - h * n[k] for k in range(3)])
            for v, p in zip(free, spots):
                edit.move(v, p)
        for v in free:
            if math.dist(r.points[v], home[v]) > limit[v]:
                return False
        return True

    def _normal_at(self, v: int):
        r = self.r
        total = [0.0, 0.0, 0.0]
        for f in r.users[v]:
            n = _normal([r.points[u] for u in r.faces[f]])
            total = [total[k] + n[k] for k in range(3)]
        size = math.sqrt(_dot(total, total)) or 1.0
        return [x / size for x in total]

    def _fits(self, edit: _Edit, before: dict, gain: int) -> bool:
        """The faces round a change after relaxing: none folded, turned or caved in, and the view within the trade."""

        r = self.r
        touched = set(edit.faces)
        for v in edit.points:
            touched |= r.users[v]
        old_normal = [0.0, 0.0, 0.0]
        for f, corners in before.items():
            if corners is not None and len(corners) == 4:
                n = _normal(corners)
                old_normal = [old_normal[k] + n[k] for k in range(3)]
        size = math.sqrt(_dot(old_normal, old_normal)) or 1.0
        old_normal = [x / size for x in old_normal]
        for f in touched:
            corners = r.faces[f]
            if corners is None or len(corners) != 4:
                continue
            spots = [r.points[v] for v in corners]
            if f in before and before[f] is not None and f not in edit.faces:
                was = before[f]
                if _quality(spots) < min(_quality(was), QUALITY) or _dot(_normal(spots), _normal(was)) < MOST_TURNED:
                    return False
            elif _quality(spots) < QUALITY or _dot(_normal(spots), old_normal) < MOST_TURNED:
                return False
        if r.judge is not None:
            old = sum(r._wrong(c) for c in before.values() if c is not None and len(c) == 4)
            new = sum(r._wrong([r.points[v] for v in r.faces[f]]) for f in touched if r.faces[f] is not None and len(r.faces[f]) == 4)
            if new > old + TRADE * gain * r._unit_area:
                return False
        return True

    def _shapes(self, faces) -> dict:
        r = self.r
        return {f: None if r.faces[f] is None else [r.points[v] for v in r.faces[f]] for f in faces}

    def _around_faces(self, corners) -> set[int]:
        r = self.r
        out = set()
        for v in corners:
            out |= r.users[v]
        return out

    # Changes

    def _spins(self, v: int):
        """(energy change, f, g, first, second, ends) of every spin of an edge at `v`."""

        r = self.r
        for other in list(r._around(v)):
            yield from r._turnings(v, other)

    def _collapses(self, v: int):
        """(energy change, quad, keep, drop, b, d) of every diagonal collapse with `v` on the diagonal."""

        r = self.r
        if not self._free(v):
            return
        for q in list(r.users[v]):
            face = r.faces[q]
            if len(face) != 4:
                continue
            at = face.index(v)
            a, b, c, d = face[at], face[(at + 1) % 4], face[(at + 2) % 4], face[(at + 3) % 4]
            if not self._free(c) or r.users[a] & r.users[c] != {q}:
                continue
            if r.border[b] or r.border[d] or len(r.users[b]) < 4 or len(r.users[d]) < 4:
                continue
            if r._around(a) & r._around(c) != {b, d}:
                continue
            if len({r.materials[f] for f in r.users[a] | r.users[c]}) != 1:
                continue
            na, nb, nc, nd = (len(r.users[x]) for x in (a, b, c, d))
            if na + nc - 2 > 6:
                continue
            change = self._cost(a, na + nc - 2) - self._cost(a, na) - self._cost(c, nc) + self._cost(b, nb - 1) - self._cost(b, nb) + self._cost(d, nd - 1) - self._cost(d, nd)
            yield change, q, a, c, b, d

    def _spin(self, edit: _Edit, f: int, g: int, first, second) -> set[int]:
        old = set(zip(self.r.faces[f], self.r.faces[f][1:] + self.r.faces[f][:1])) | set(zip(self.r.faces[g], self.r.faces[g][1:] + self.r.faces[g][:1]))
        edit.set_face(f, first)
        edit.set_face(g, second)
        new = set(zip(first, list(first[1:]) + [first[0]])) | set(zip(second, list(second[1:]) + [second[0]]))
        return {v for e in old ^ new for v in e}  # the ends of the edge that went and of the one that came

    def _collapse(self, edit: _Edit, q: int, keep: int, drop: int) -> set[int]:
        r = self.r
        corners = set(r.faces[q])
        others = r.users[drop] - {q}
        middle = tuple(0.5 * (r.points[keep][k] + r.points[drop][k]) for k in range(3))
        edit.set_face(q, None)
        for f in others:
            edit.set_face(f, [keep if u == drop else u for u in r.faces[f]])
        edit.move(keep, middle)
        corners.discard(drop)
        return corners

    def _keep(self, edit: _Edit, corners, gain: int, before) -> bool:
        if self._relax(edit, corners) and self._fits(edit, before, gain):
            return True
        edit.undo()
        return False

    def _steps(self, v: int, collapse: bool):
        """(energy change, kind, option, poles it may move to) of the changes at `v`, cheapest first."""

        out = [(o[0], "spin", o, o[5]) for o in self._spins(v)]
        if collapse:
            out += [(o[0], "collapse", o, (o[2], o[4], o[5])) for o in self._collapses(v)]
        out.sort(key=lambda o: o[0])
        return out

    def _apply(self, kind: str, o, gain: int):
        """Make one change, relaxed and checked. The edit (to undo it), or None if it didn't fit."""

        r = self.r
        edit = _Edit(r)
        if kind == "spin":
            change, f, g, first, second, _ends = o
            before = self._shapes(self._around_faces(set(r.faces[f]) | set(r.faces[g])))
            corners = self._spin(edit, f, g, first, second)
        else:
            change, q, a, c, _b, _d = o
            keep, drop = (a, c) if a < c else (c, a)
            before = self._shapes(self._around_faces(set(r.faces[q]) | r._around(a) | r._around(c)))
            corners = self._collapse(edit, q, keep, drop)
        if not self._keep(edit, corners, gain, before):
            return None
        if kind == "collapse":
            r.quadrics[keep] = [x + y for x, y in zip(r.quadrics[keep], r.quadrics[drop])]
            r.stamps[keep] += 1
            r.stamps[drop] += 1
            self.collapsed += 1
        return edit

    def _walk(self, v: int, steps: int, seen: set, collapse: bool, debt: int = 0) -> int:
        """Changes round `v`, `steps` deep, letting the energy rise by up to `MOST_DEBT` on the way, until the walk as a
        whole lowers it. A walk that doesn't is undone; returns the energy taken away."""

        r = self.r
        for change, kind, o, ends in self._steps(v, collapse):
            if debt + change > MOST_DEBT:
                break
            key = (kind, o[1], tuple(o[3]) if kind == "spin" else o[2])
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > WALK_TRIES:
                return 0
            edit = self._apply(kind, o, max(-change, 1))
            if edit is None:
                continue
            if debt + change < 0:
                return -(debt + change)
            if steps > 1:
                for w in ends:
                    if r.users[w] and len(r.users[w]) != 4 and not r.border[w]:
                        got = self._walk(w, steps - 1, seen, collapse, debt + change)
                        if got:
                            return got
            if kind == "collapse":
                self.collapsed -= 1
            edit.undo()
        return 0

    # Driver

    def run_steps(self, rounds: int = ROUNDS, walk: int = WALK, collapse: bool = True):
        """Generator: take poles away until a round finds nothing, a slice per pole tried. Returns the pole energy taken
        away."""

        r = self.r
        for _ in range(rounds) if rounds else count():
            gained = 0
            poles = [v for v in range(len(r.users)) if r.users[v] and len(r.users[v]) != 4 and not r.border[v]]
            for depth in (1, walk):
                for v in poles:
                    if r.users[v] and len(r.users[v]) != 4 and not r.border[v]:
                        got = self._walk(v, depth, set(), collapse)
                        yield
                        if got:
                            self.walked += depth > 1
                            gained += got
                if depth >= walk:
                    break
            self.gained += gained
            if not gained:
                break
        return self.gained


__all__ = ["PoleCleaner"]
