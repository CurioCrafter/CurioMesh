# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Face loops: the strips of quads of a result, and small edits to one strip.

A strip is what you get walking from a quad across one edge to the quad on the other side, then across that quad's
opposite edge, and so on (Blender's face loop select does the same walk) until the walk comes back to its first quad
(closed) or reaches a border, a triangle, an n-gon, or an edge with a pole at either end (open). Stopping at poles keeps
a strip to one run of rows: on a messy result a walk that went on past them winds over the whole mesh. Every quad lies on two strips, one per pair of opposite
edges; an "element" is one of those: `row * 2 + d`, the quad in row `row` of `Strips.quads` crossed through its edges
d and d + 2.

Along a strip, each crossed edge is a rung (a_i, b_i): the a's make one side of the strip, the b's the other. Each rung
carries on straight across the neighbouring quads on both sides (`rung_lines`), and the edits move the strip along those
lines only, so no quad turns or changes neighbours:

- `slide`: both sides of the strip move toward one neighbour;
- `straighten`: the middle of the strip is smoothed along its length (umbrella smoothing of a polyline, Taubin 1995),
  and each rung follows the smoothed middle across the strip;
- `even`: every rung gets the strip's median width.

The neighbours within `REACH` rungs take up the change in proportion to their own widths. `cut` adds a strip by
splitting the strip down its middle (a loop cut), `collapse` takes it away by merging its two sides; both refuse when
a face next to the strip would stop being a quad. Plain NumPy.
"""

from __future__ import annotations

import numpy as np

from .patch import _labels

REACH = 3  # neighbouring quads on each side of the strip that take up a move
NARROWEST = 0.2  # a quad along a rung line keeps at least this share of the line's mean step
LONG = 4  # quads: shorter strips are not drawn in Loop Flow
TURN_BACK = 0.3  # a quad whose normal turns past this cosine during an edit puts its corners back


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


class Strips:
    """Every strip of a mesh. `labels[element]`: the strip it lies on; `sizes[label]`: elements on that strip."""

    def __init__(self, co, faces):
        self.co = np.asarray(co, dtype=np.float64)
        self.faces = faces
        self.face_of = np.fromiter((i for i, f in enumerate(faces) if len(f) == 4), dtype=np.int64)
        self.quads = np.asarray([faces[i] for i in self.face_of], dtype=np.int64).reshape(-1, 4)
        self.row_of = {int(f): r for r, f in enumerate(self.face_of.tolist())}
        n, q = len(self.co), self.quads
        a, b = q.reshape(-1), np.roll(q, -1, axis=1).reshape(-1)
        # Edge use over every face: an edge with a third face, or only one, ends a strip.
        fa = np.fromiter((v for f in faces for v in f), dtype=np.int64)
        fb = np.fromiter((f[(k + 1) % len(f)] for f in faces for k in range(len(f))), dtype=np.int64)
        used, counts = np.unique(np.minimum(fa, fb) * n + np.maximum(fa, fb), return_counts=True)
        self.twin = np.full(len(a), -1, dtype=np.int64)
        self.full_twin = self.twin
        if len(a):
            key = a * n + b
            order = np.argsort(key)
            at = np.minimum(np.searchsorted(key[order], b * n + a), len(key) - 1)
            twin = np.where(key[order][at] == b * n + a, order[at], -1)
            two = counts[np.searchsorted(used, np.minimum(a, b) * n + np.maximum(a, b))] == 2
            self.twin = np.where(two, twin, -1)  # half-edge row * 4 + k -> the half-edge across it on the next quad
            from .patterns import Topology

            poles = Topology(self.co, faces).poles
            self.full_twin = self.twin.copy()  # the walk past poles, for adding and taking away whole strips
            self.twin[poles[a] | poles[b]] = -1  # the rows turn at a pole: the strip stops at an edge touching one
        h = np.flatnonzero(self.twin >= 0)
        here = (h // 4) * 2 + (h % 4) % 2
        there = (self.twin[h] // 4) * 2 + (self.twin[h] % 4) % 2
        self.labels = _labels(2 * len(q), here, there)
        self.sizes = np.bincount(self.labels, minlength=2 * len(q))

    def element(self, face: int, edge: int) -> int:
        """The element of quad `face` whose strip crosses its edge `edge` (corner edge -> edge + 1); -1 for no quad."""
        row = self.row_of.get(int(face))
        return -1 if row is None else row * 2 + edge % 2

    def walk(self, element: int, past_poles: bool = False) -> dict:
        """The strip through `element`, in order: {"rows", "entry" (crossed edge per quad), "faces", "a", "b" (its two
        sides, one rung each, the last rung dropped when closed), "closed", "crossing" (it would run over itself)}. `past_poles`:
        the whole strip, on past poles to a border or back to its start (a strip can only be split or joined whole)."""
        row0, d = divmod(int(element), 2)
        q, twin = self.quads, (self.full_twin if past_poles else self.twin)

        seen, crossing = {row0}, []

        def go(row, entry):
            seq = []
            while True:
                t = twin[row * 4 + (entry + 2) % 4]
                if t < 0:
                    return seq, False
                row, entry = divmod(int(t), 4)
                if row == row0 and entry % 2 == d:
                    return seq, True
                if row in seen:  # it would run over itself: stop short of that
                    crossing.append(True)
                    return seq, False
                seen.add(row)
                seq.append((row, entry))

        ahead, closed = go(row0, d)
        behind = []
        if not closed:
            back, _ = go(row0, (d + 2) % 4)
            behind = [(r, (k + 2) % 4) for r, k in reversed(back)]
        steps = behind + [(row0, d)] + ahead
        rows = [r for r, _k in steps]
        entry = [k for _r, k in steps]
        c = q[rows]
        k = np.asarray(entry)
        i = np.arange(len(rows))
        a, b = c[i, k].tolist(), c[i, (k + 1) % 4].tolist()
        if not closed:
            a.append(int(c[-1, (k[-1] + 3) % 4]))
            b.append(int(c[-1, (k[-1] + 2) % 4]))
        return {"rows": rows, "entry": entry, "faces": self.face_of[rows].tolist(), "a": a, "b": b, "closed": closed, "crossing": bool(crossing)}

    def segments(self, least: int = LONG):
        """(segment ends 2 per element, strip label per segment): each quad crossed by a line from the middle of one
        edge to the middle of the opposite one, for the strips of `least` quads or more."""
        p = self.co[self.quads]
        mid = 0.5 * (p + np.roll(p, -1, axis=1))
        keep = np.flatnonzero(self.sizes[self.labels] >= least)
        row, d = keep // 2, keep % 2
        ends = np.stack((mid[row, d], mid[row, d + 2]), axis=1).reshape(-1, 3)
        return ends, self.labels[keep]


def stats(co, strip: dict) -> dict:
    """{"faces", "closed", "turn" (mean degrees the middle of the strip turns per quad, across the surface), "width"
    (how much the width varies, as a share of the mean)}."""
    co = np.asarray(co, dtype=np.float64)
    a, b = co[strip["a"]], co[strip["b"]]
    mid = 0.5 * (a + b)
    across = b - a
    width = np.linalg.norm(across, axis=1)
    turn = 0.0
    if strip["closed"]:
        mid = np.vstack((mid[-1:], mid, mid[:1]))
        across = np.vstack((across[-1:], across, across[:1]))
    if len(mid) >= 3:
        d1, d2 = mid[1:-1] - mid[:-2], mid[2:] - mid[1:-1]
        n = _unit(np.cross(d1 + d2, across[1:-1]))  # the surface normal at the rung
        d1 -= (d1 * n).sum(axis=1, keepdims=True) * n
        d2 -= (d2 * n).sum(axis=1, keepdims=True) * n
        turn = float(np.degrees(np.abs(np.arctan2((np.cross(d1, d2) * n).sum(axis=1), (d1 * d2).sum(axis=1)))).mean())
    return {"faces": len(strip["rows"]), "closed": bool(strip["closed"]), "turn": turn, "width": float(width.std() / max(width.mean(), 1e-30))}


def describe(numbers: dict) -> str:
    shape = "closed loop" if numbers["closed"] else "open strip"
    return f"{numbers['faces']} faces, {shape}, turns {numbers['turn']:.1f} deg per quad, width varies {100.0 * numbers['width']:.0f}%"


# --------------------------------------------------------------------------- moving a strip along its rungs


def rung_lines(strip: dict, go: dict, count: int, stay: np.ndarray, reach: int = REACH) -> list[tuple[np.ndarray, int]]:
    """Per rung, (vertex ids along the rung carried straight on across `reach` quads each side, index of a). `go`:
    `loopscore._continuations` of the mesh. A line stops at a pole, a border or a pinned vertex; its two ends never move."""
    out = []
    for a, b in zip(strip["a"], strip["b"]):
        sides = []
        for here, there in ((a, b), (b, a)):
            ext, prev, cur = [], there, here
            if not stay[here]:
                for _ in range(reach):
                    nxt = go.get(prev * count + cur)
                    if nxt is None or nxt in ext or nxt in (a, b):
                        break
                    ext.append(nxt)
                    if stay[nxt]:
                        break
                    prev, cur = cur, nxt
            sides.append(ext)
        out.append((np.asarray(sides[0][::-1] + [a, b] + sides[1], dtype=np.int64), len(sides[0])))
    return out


def _arc(points: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def _along(points: np.ndarray, s: np.ndarray, at: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(at, s, points[:, k]) for k in range(3)], axis=1)


def targets(co, lines, kind: str, amount: float, closed: bool = False) -> list[tuple[float, float]]:
    """Per rung line, where its a and b go, as distances along the line: `kind` "slide" (`amount` -1..1 of the next
    quad's width toward the a side or the b side), "straighten" or "even" (`amount` 0..1). `closed`: the strip is a loop."""
    co = np.asarray(co, dtype=np.float64)
    arcs = [_arc(co[ids]) for ids, _ia in lines]
    sa = np.asarray([s[ia] for s, (_ids, ia) in zip(arcs, lines)])
    sb = np.asarray([s[ia + 1] for s, (_ids, ia) in zip(arcs, lines)])
    if kind == "slide":
        shift = []
        for s, (_ids, ia) in zip(arcs, lines):
            if amount > 0:
                shift.append(amount * (s[ia + 2] - s[ia + 1]) if ia + 2 < len(s) else 0.0)
            else:
                shift.append(amount * (s[ia] - s[ia - 1]) if ia > 0 else 0.0)
        shift = np.asarray(shift)
        return list(zip((sa + shift).tolist(), (sb + shift).tolist()))
    if kind == "even":
        centre = 0.5 * (sa + sb)
        half = 0.5 * ((1.0 - amount) * (sb - sa) + amount * np.median(sb - sa))
        return list(zip((centre - half).tolist(), (centre + half).tolist()))
    if kind == "straighten":
        a = np.asarray([co[ids[ia]] for ids, ia in lines]).reshape(-1, 3)
        b = np.asarray([co[ids[ia + 1]] for ids, ia in lines]).reshape(-1, 3)
        mid, across = 0.5 * (a + b), _unit(b - a)
        # How far each rung can shift along its line (the same room `move` leaves); a rung with a pinned end can't.
        room = np.zeros((len(lines), 2))
        way = across.copy()  # how the middle of a rung moves per unit of shift: a and b each along their own line
        for k, ((ids, ia), s) in enumerate(zip(lines, arcs)):
            m = len(ids)
            if ia > 0 and ia + 2 < m and s[-1] > 1e-30:
                step = NARROWEST * s[-1] / (m - 1)
                room[k] = (s[0] + step * ia - s[ia], s[-1] - step * (m - 2 - ia) - s[ia + 1])
                p = co[ids]
                way[k] = 0.25 * (2.0 * across[k] + _unit(p[ia + 2] - p[ia + 1]) + _unit(p[ia] - p[ia - 1]))
        if not closed and len(room):
            room[0] = room[-1] = 0.0
        # Pulling the middle of the strip straight with each rung kept inside its room (each point toward the line
        # through its two neighbours, as in Organize's loop pull):
        # wiggles go as far as the quads beside them allow, and a rung that can't move holds the curve.
        shift = np.zeros(len(lines))
        for _ in range(int(round(40 * amount))):
            p = mid + shift[:, None] * way
            before, after = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
            u = _unit(after - before)
            line = before + ((p - before) * u).sum(axis=1, keepdims=True) * u  # onto the line through its neighbours: straight, spacing left alone
            even = 0.5 * p + 0.5 * line
            shift = np.clip(((even - mid) * way).sum(axis=1) / np.maximum((way * way).sum(axis=1), 1e-30), room[:, 0], room[:, 1])
        return list(zip((sa + shift).tolist(), (sb + shift).tolist()))
    raise ValueError(kind)


def move(co, lines, goals, stay: np.ndarray, spread: bool = False) -> np.ndarray:
    """The points with each rung line's a and b at its goal and the vertices between them and the line's ends
    stretched in proportion (`spread`: evenly instead). Only along the lines as they were, so nothing leaves them."""
    co = np.asarray(co, dtype=np.float64)
    total, hits = np.zeros_like(co), np.zeros(len(co))
    for (ids, ia), (ga, gb) in zip(lines, goals):
        p = co[ids]
        s = _arc(p)
        m, ib = len(ids), ia + 1
        if m < 3 or s[-1] <= 1e-30:
            continue
        step = NARROWEST * s[-1] / (m - 1)
        lo, hi = s[0] + step * ia, s[-1] - step * (m - 1 - ib)
        ga = s[0] if ia == 0 else min(max(ga, lo), hi - step)
        gb = s[-1] if ib == m - 1 else min(max(gb, ga + step), hi)
        if ia == 0:
            gb = max(gb, ga + step)
        if spread:
            new = np.linspace(s[0], s[-1], m)
        else:
            new = s.copy()
            if ia > 0:
                new[: ia + 1] = s[0] + (s[: ia + 1] - s[0]) * (ga - s[0]) / max(s[ia] - s[0], 1e-30)
            if ib < m - 1:
                new[ib:] = s[-1] - (s[-1] - s[ib:]) * (s[-1] - gb) / max(s[-1] - s[ib], 1e-30)
            new[ia], new[ib] = ga, gb
        inner = ids[1:-1]
        np.add.at(total, inner, _along(p, s, new[1:-1]))
        np.add.at(hits, inner, 1.0)
    out = co.copy()
    moved = (hits > 0) & ~stay
    out[moved] = total[moved] / hits[moved, None]
    return out


def partners(co, axis: int, offset: float) -> np.ndarray:
    """Per vertex, its mirror image across the plane (itself on the plane), or -1."""
    co = np.asarray(co, dtype=np.float64)
    size = float(np.linalg.norm(np.ptp(co, axis=0))) or 1.0
    # rounded-key lookup; a vertex sitting on a rounding boundary finds no partner (it is then left alone)
    key = lambda p: map(tuple, np.round(p / (1e-5 * size)).astype(np.int64).tolist())  # noqa: E731
    index = {k: i for i, k in enumerate(key(co))}
    flip = co.copy()
    flip[:, axis] = 2.0 * offset - flip[:, axis]
    return np.asarray([index.get(k, -1) for k in key(flip)], dtype=np.int64)


def finish(old, new, faces, project=None, mirror=None, side: float = 1.0) -> np.ndarray:
    """`new` made whole: mirrored from the `side` of a (axis, offset, partners) mirror, put on the surface by
    `project`, and with the corners of any quad that flipped put back."""
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64).copy()
    if mirror is not None:
        axis, offset, partner = mirror
        rel = old[:, axis] - offset
        size = float(np.linalg.norm(np.ptp(old, axis=0))) or 1.0
        on = np.abs(rel) <= 1e-6 * size
        lead = np.flatnonzero((partner >= 0) & ~on & (np.sign(rel) == np.sign(side)))
        new[partner[lead]] = new[lead]
        new[partner[lead], axis] = 2.0 * offset - new[lead, axis]
        new[on, axis] = offset
    moved = np.flatnonzero(np.any(np.abs(new - old) > 0.0, axis=1))
    if project is not None and len(moved):
        new[moved] = project(new[moved])
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    if len(moved) and len(quads):
        touched = quads[np.isin(quads, moved).any(axis=1)]
        for _ in range(3):
            n0 = _quad_normals(old, touched)
            n1 = _quad_normals(new, touched)
            bad = (n0 * n1).sum(axis=1) < TURN_BACK
            if not bad.any():
                break
            back = np.unique(touched[bad])
            new[back] = old[back]
    return new


def move_vertex(co, faces, vertex: int, goal, stay: np.ndarray, rings: int = 2, rounds: int = 6) -> np.ndarray:
    """`vertex` (a pole, say) put at `goal`, and the free vertices within `rings` edges of it eased toward the middle of
    their neighbours so the quads round it follow (plain umbrella smoothing, Taubin 1995). Nothing else moves."""
    co = np.asarray(co, dtype=np.float64)
    near: dict[int, set] = {}
    for f in faces:
        for k in range(len(f)):
            near.setdefault(f[k], set()).update((f[k - 1], f[(k + 1) % len(f)]))
    ring, seen = {vertex}, {vertex}
    for _ in range(rings):
        ring = {u for v in ring for u in near.get(v, ())} - seen
        seen |= ring
    free = [v for v in seen if v != vertex and not stay[v]]
    out = co.copy()
    out[vertex] = goal
    for _ in range(rounds):
        mid = np.asarray([out[list(near[v])].mean(axis=0) for v in free]).reshape(-1, 3)
        out[free] = 0.5 * out[free] + 0.5 * mid
    return out


def bent(co, faces, degrees: float = 25.0) -> np.ndarray:
    """Vertices where the surface bends: a quad round it turns more than `degrees` from its normal. Moving these
    along the surface loses the shape they hold (a brow, a lip), so the whole-result fix leaves them."""
    co = np.asarray(co, dtype=np.float64)
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    n = _quad_normals(co, quads)
    vn = np.zeros_like(co)
    np.add.at(vn, quads.reshape(-1), np.repeat(n, 4, axis=0))
    vn = _unit(vn)
    turn = np.zeros(len(co))
    cos = (n[:, None, :] * vn[quads]).sum(axis=2)
    np.maximum.at(turn, quads.reshape(-1), np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).reshape(-1))
    return turn > degrees


def capped(old, new, faces, share: float = 1.0) -> np.ndarray:
    """`new` with no vertex further than `share` of its mean edge length from where it was in `old`."""
    old, new = np.asarray(old, dtype=np.float64), np.asarray(new, dtype=np.float64)
    a = np.fromiter((v for f in faces for v in f), dtype=np.int64)
    b = np.fromiter((f[(k + 1) % len(f)] for f in faces for k in range(len(f))), dtype=np.int64)
    length = np.linalg.norm(old[a] - old[b], axis=1)
    mean = (np.bincount(a, length, len(old)) + np.bincount(b, length, len(old))) / np.maximum(np.bincount(a, minlength=len(old)) + np.bincount(b, minlength=len(old)), 1)
    drift = new - old
    far = np.linalg.norm(drift, axis=1)
    return old + drift * np.minimum(1.0, share * mean / np.maximum(far, 1e-30))[:, None]


def _quad_normals(co, quads):
    p = co[quads]
    return _unit(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]))


# --------------------------------------------------------------------------- adding and taking away strips


def cut(co, faces, strips, project=None):
    """A strip added beside each of `strips` by splitting it down its middle. Returns (co, faces, parent face of each
    new face, (face, edge) of a quad on the new strip crossing that edge). Raises ValueError when a face beside it would
    stop being a quad (or a triangle that becomes one)."""
    co = np.asarray(co, dtype=np.float64)
    split, rungs = {}, set()
    for strip in strips:
        if strip["crossing"]:
            raise ValueError("this strip winds back over itself (common on a messy result: Tidy Up first)")
        for f, k in zip(strip["faces"], strip["entry"]):
            if f in split:
                raise ValueError("this strip runs over itself")
            split[f] = k
        ends = list(zip(strip["a"], strip["b"]))
        rungs |= {(min(x, y), max(x, y)) for x, y in ends}
    pairs = sorted(rungs)
    mids = 0.5 * (co[[x for x, _y in pairs]] + co[[y for _x, y in pairs]])
    if project is not None and len(mids):
        mids = project(mids)
    mid_of = {e: len(co) + i for i, e in enumerate(pairs)}
    new_faces, parent, start = [], [], None
    extra = []
    for i, f in enumerate(faces):
        ring = []
        for k in range(len(f)):
            u, v = f[k], f[(k + 1) % len(f)]
            ring.append(u)
            m = mid_of.get((min(u, v), max(u, v)))
            if m is not None:
                ring.append(m)
        if i in split:
            at = [j for j, v in enumerate(ring) if v >= len(co)]
            if len(ring) != 6 or len(at) != 2 or at[1] - at[0] != 3:
                raise ValueError("a quad on this strip is not a plain quad")
            p = at[0]
            one = tuple(ring[(p + j) % 6] for j in range(4))
            two = tuple(ring[(p + 3 + j) % 6] for j in range(4))
            new_faces.append(one)
            parent.append(i)
            extra.append((two, i))
            if start is None:
                start = len(faces) + len(extra) - 1
        else:
            if len(ring) != len(f) and (len(f) == 4 or len(ring) > 4):
                raise ValueError("a face at the end of this strip would stop being a quad")
            new_faces.append(tuple(ring))
            parent.append(i)
    for two, i in extra:
        new_faces.append(two)
        parent.append(i)
    return np.vstack((co, mids)), new_faces, np.asarray(parent, dtype=np.int64), (start, 0)


def collapse(co, faces, strips, stay=None):
    """Each of `strips` taken away by merging its two sides. Returns (co, faces, parent face of each face left).
    Raises ValueError where that would leave a face that is not a quad, or move two pinned vertices onto each other."""
    co = np.asarray(co, dtype=np.float64).copy()
    stay = np.zeros(len(co), dtype=bool) if stay is None else stay
    gone, into = set(), {}
    for strip in strips:
        if strip["crossing"]:
            raise ValueError("this strip runs over itself")
        gone |= set(strip["faces"])
        last = len(strip["a"]) - 1
        for i, (a, b) in enumerate(zip(strip["a"], strip["b"])):
            if a in into or b in into or a == b:
                raise ValueError("this strip touches itself")
            end = not strip["closed"] and i in (0, last)  # a border rung: its middle stays on the border
            if stay[a] and stay[b] and end:
                co[a] = 0.5 * (co[a] + co[b])
                into[b] = a
                continue
            if stay[a] and stay[b]:
                raise ValueError("both sides of this strip are held (a hard edge or a border)")
            keep, drop = (b, a) if stay[b] else (a, b)
            co[keep] = co[keep] if stay[keep] else 0.5 * (co[a] + co[b])
            into[drop] = keep
    new_faces, parent, seen = [], [], set()
    for i, f in enumerate(faces):
        if i in gone:
            continue
        g = tuple(into.get(v, v) for v in f)
        if len(set(g)) != len(g):
            raise ValueError("a face at the end of this strip would fold away")
        key = tuple(sorted(g))
        if key in seen:
            raise ValueError("two faces beside this strip would land on each other")
        seen.add(key)
        new_faces.append(g)
        parent.append(i)
    used = np.unique(np.fromiter((v for f in new_faces for v in f), dtype=np.int64))
    remap = np.full(len(co), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return co[used], [tuple(int(remap[v]) for v in f) for f in new_faces], np.asarray(parent, dtype=np.int64)


def mirror_element(strips: Strips, strip: dict, partner: np.ndarray) -> int:
    """The element of the mirror image of `strip` (-1 when it has none, or it is its own image)."""
    a, b = partner[strip["a"][0]], partner[strip["b"][0]]
    if a < 0 or b < 0:
        return -1
    face = set(partner[strips.quads[strip["rows"][0]]].tolist())
    for r in np.flatnonzero(np.isin(strips.quads, list(face)).all(axis=1)).tolist():
        c = strips.quads[r].tolist()
        for k in range(4):
            if {c[k], c[(k + 1) % 4]} == {int(a), int(b)}:
                e = r * 2 + k % 2
                return -1 if strips.labels[e] == strips.labels[strip["rows"][0] * 2 + strip["entry"][0] % 2] else e
    return -1


__all__ = ["Strips", "bent", "capped", "collapse", "cut", "describe", "finish", "mirror_element", "move", "move_vertex", "partners", "rung_lines", "stats", "targets"]
