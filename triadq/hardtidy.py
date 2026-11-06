# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Hard-surface Tidy Tools (plain NumPy): True Up Edges and Press Panels Flat, as `trial.Trial` subclasses.

True Up Edges. The hard edges of the result (two faces folding past `organize.SHARP_DEG`) are walked into chains that
end where three or more hard edges meet. Each chain is matched to the crease of your original mesh it follows: the
faces on each side of it are sampled on your original mesh and a plane is fitted to each side near every corner (least
squares), so the crease there is where the two planes meet. A chain whose creases line up (principal axis fit, residual
under `STRAIGHT` of its length) is laid on that one line; a curved one is smoothed along itself with Taubin's
shrink-free two-step filter (Taubin 1995), put back on its creases and spaced evenly by arc length. The corners beside
a chain follow half of its move along their rungs, so the first row of quads keeps its shape. Everything ends on your
original mesh.

Press Panels Flat. Flat areas of the result are grown the way Clean Panels grows them (`flatquad._grow_regions`),
loosely, since the result is the bumpy one; each is kept as a panel only when your original mesh under it is flat
(a least-squares plane through its points there, the panel test of `hardsurf`). A corner inside one panel goes onto its
plane; a corner shared by two or three panels goes to the point closest to all of their planes and a little to where it
was (a 3x3 least-squares solve), so shared borders become exact lines and corners exact points.

Both hold the edge of the scope, open borders and the mirror plane, never move a corner further than `DRIFT` of an
edge (a red X "drift" instead), and put back the corners of any face that would turn past `organize.WORST_TURN`.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .organize import SHARP_DEG, WORST_TURN
from .trial import Base, Candidate, Trial

NO_HARD_EDGES = "No hard edges in this area."
NO_PANELS = "No flat panels found in this area."
SLICE = 0.004  # seconds of work between two yields
PROJECT_CHUNK = 2048  # points sent to the source per call
WINDOW = 2  # chain edges each side of a corner whose side samples fit its crease planes
CREASE_DEG = 15.0  # the two side planes must meet at more than this for a crease to be there
CREASE_REACH = 0.5  # of an edge: a crease further than this from the chain is not the one it follows
ON_SOURCE = 0.05  # of an edge: where the two side planes meet must be this close to your original mesh, else the
# sides are one smooth curve (their tangent planes meet off the surface) and there is no crease to follow
STRAIGHT = 0.01  # of the chain's length: residual of a principal-axis fit under which a chain is laid on one line
TAUBIN = (0.5, -0.53, 3)  # lambda, mu, rounds (Taubin 1995)
FOLLOW = 0.5  # share of a chain corner's move the corners beside it take
PANEL_DEG = 10.0  # result faces whose normals differ by less than this can share a panel (looser than Clean Panels)
PANEL_GAP = 0.25  # of an edge: how far a result corner may sit off a growing panel's plane
FLAT = 0.02  # of an edge: your original mesh under a panel must sit this close to its plane (95th percentile)
PANEL_FACES = 2  # fewer faces than this is not a panel
PULL = 0.02  # how much a shared corner also stays near where it was
DRIFT = 1.0  # of an edge: no corner moves further than this in all


# --------------------------------------------------------------------------- shared


def _ticker():
    """A function that is True once SLICE seconds have passed since it last was."""
    mark = [time.perf_counter() + SLICE]

    def due() -> bool:
        now = time.perf_counter()
        if now >= mark[0]:
            mark[0] = now + SLICE
            return True
        return False

    return due


def _project(base: Base, points: np.ndarray):
    """Generator: `base.project` in chunks, yielding between them; returns the points on your original mesh."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if base.project is None or not len(points):
        return points.copy()
    out = np.empty_like(points)
    for k in range(0, len(points), PROJECT_CHUNK):
        out[k : k + PROJECT_CHUNK] = base.project(points[k : k + PROJECT_CHUNK])
        yield
    return out


def _edge_users(faces) -> dict:
    out: dict[tuple[int, int], list[int]] = {}
    for i, f in enumerate(faces):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            out.setdefault((a, b) if a < b else (b, a), []).append(i)
    return out


def _newell(co, faces) -> np.ndarray:
    """Newell's normal per face, length twice the area (faces grouped by size, so it is one NumPy call per size)."""
    out = np.zeros((len(faces), 3))
    by_size: dict[int, list[int]] = {}
    for i, f in enumerate(faces):
        by_size.setdefault(len(f), []).append(i)
    for _size, ids in by_size.items():
        p = co[np.asarray([faces[i] for i in ids], dtype=np.int64)]
        out[ids] = np.cross(p, np.roll(p, -1, axis=1)).sum(axis=1)
    return out


def face_normals(co, faces) -> np.ndarray:
    """Unit normal per face (Newell's method)."""
    n = _newell(np.asarray(co, dtype=np.float64), faces)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.zeros_like(n), where=length > 1e-300)


def movable(base: Base, users: dict | None = None) -> np.ndarray:
    """Corners these tools may move. Unlike `Base.free` this includes hard-edge corners and corners of triangles and
    n-gons (they are what the tools straighten), but never the edge of the scope or an open border."""
    users = _edge_users(base.faces) if users is None else users
    inside = np.zeros(len(base.co), dtype=bool)
    outside = np.zeros(len(base.co), dtype=bool)
    for i, f in enumerate(base.faces):
        (inside if base.scope[i] else outside)[list(f)] = True
    border = np.zeros(len(base.co), dtype=bool)
    plane = _on_plane(base)
    for (a, b), fs in users.items():
        if len(fs) != 2 and not (plane[a] and plane[b]):
            border[a] = border[b] = True
    return inside & ~outside & ~border


def _on_plane(base: Base) -> np.ndarray:
    if base.mirror is None or not len(base.co):
        return np.zeros(len(base.co), dtype=bool)
    axis, offset = base.mirror
    size = float(np.linalg.norm(np.ptp(base.co, axis=0))) or 1.0
    return np.abs(base.co[:, axis] - offset) <= 1e-6 * size


def _keep_plane(base: Base, co: np.ndarray) -> None:
    """Corners on the mirror plane slide in it only."""
    if base.mirror is not None:
        on = _on_plane(base)
        co[on, base.mirror[0]] = base.co[on, base.mirror[0]]


def _guard(base: Base, co: np.ndarray, moved: np.ndarray, held: list) -> np.ndarray:
    """Drift and turn guards: a corner moved more than DRIFT of an edge goes back ("drift"); the corners of a
    face whose normal turned past WORST_TURN go back ("turn")."""
    far = np.linalg.norm(co - base.co, axis=1) > DRIFT * base.edge
    for v in np.flatnonzero(far & moved).tolist():
        held.append((base.co[v].copy(), "drift"))
    co[far] = base.co[far]
    touched = [i for i, f in enumerate(base.faces) if moved[list(f)].any()]
    if not touched:
        return co
    sub = [base.faces[i] for i in touched]
    start = face_normals(base.co, sub)
    for _round in range(4):
        now = face_normals(co, sub)
        bad = (now * start).sum(axis=1) < WORST_TURN
        if not bad.any():
            break
        for k in np.flatnonzero(bad).tolist():
            f = list(sub[k])
            if not np.allclose(co[f], base.co[f]):
                held.append((base.co[f].mean(axis=0), "turn"))
            co[f] = base.co[f]
    return co


def _fit_plane(points: np.ndarray):
    """(unit normal, offset, residuals) of the least-squares plane through the points."""
    centre = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - centre, full_matrices=False)
    n = vt[-1]
    d = float(n @ centre)
    return n, d, np.abs(points @ n - d)


def _fit_plane_trimmed(points: np.ndarray):
    """As `_fit_plane`, fitted again without the points far off the first fit (border samples that landed on the
    panel next door)."""
    n, d, res = _fit_plane(points)
    keep = res <= max(3.0 * float(np.median(res)), 1e-12)
    if 3 <= keep.sum() < len(points):
        n, d, _ = _fit_plane(points[keep])
        res = np.abs(points @ n - d)
    return n, d, res, keep


# --------------------------------------------------------------------------- True Up Edges


def hard_chains(co, faces, scope=None, users=None, sharp_deg: float = SHARP_DEG):
    """Chains of hard edges with a face of the scope on them: [(corners, closed)], ending where a chain meets a corner
    with one or three or more hard edges."""
    co = np.asarray(co, dtype=np.float64)
    users = _edge_users(faces) if users is None else users
    scope = np.ones(len(faces), dtype=bool) if scope is None else scope
    normals = face_normals(co, faces)
    limit = math.cos(math.radians(sharp_deg))
    hard = [(a, b) for (a, b), fs in users.items() if len(fs) == 2 and (scope[fs[0]] or scope[fs[1]]) and float(normals[fs[0]] @ normals[fs[1]]) < limit]
    around: dict[int, list[int]] = {}
    for a, b in hard:
        around.setdefault(a, []).append(b)
        around.setdefault(b, []).append(a)
    seen: set = set()
    chains = []

    def walk(start, nxt):
        path = [start, nxt]
        seen.add((min(start, nxt), max(start, nxt)))
        while len(around[path[-1]]) == 2 and path[-1] != start:
            a, b = around[path[-1]]
            step = b if a == path[-2] else a
            key = (min(path[-1], step), max(path[-1], step))
            if key in seen:
                break
            seen.add(key)
            path.append(step)
        return path

    for v in sorted(around):
        if len(around[v]) != 2:
            for w in around[v]:
                if (min(v, w), max(v, w)) not in seen:
                    chains.append((walk(v, w), False))
    for v in sorted(around):
        for w in around[v]:
            if (min(v, w), max(v, w)) not in seen:
                path = walk(v, w)
                closed = path[-1] == path[0]
                chains.append((path[:-1] if closed else path, closed))
    return [(p, c) for p, c in chains if len(p) >= (3 if c else 2)]


def _side_samples(co, faces, users, path, closed):
    """Per chain edge: the points of the faces on its left and on its right (face middle and corners off the chain)."""
    on = set(path)
    n = len(path) if closed else len(path) - 1
    left, right = [], []
    for k in range(n):
        a, b = path[k], path[(k + 1) % len(path)]
        fs = users.get((min(a, b), max(a, b)), [])
        pl, pr = [], []
        for i in fs:
            f = list(faces[i])
            j = f.index(a)
            pts = [co[f].mean(axis=0)] + [co[v] for v in f if v not in on]
            (pl if f[(j + 1) % len(f)] == b else pr).extend(pts)
        left.append(pl)
        right.append(pr)
    return left, right


def _line_of(nl, dl, nr, dr):
    """A point and unit direction of the line where two planes meet, or None when they hardly meet."""
    d = np.cross(nl, nr)
    s = float(np.linalg.norm(d))
    if s < math.sin(math.radians(CREASE_DEG)):
        return None
    d /= s
    m = np.stack((nl, nr))
    p = np.linalg.lstsq(m, np.array([dl, dr]), rcond=None)[0]
    return p, d


def _on_line(x, line):
    p, d = line
    return p + float((x - p) @ d) * d


def _even(points: np.ndarray, closed: bool) -> np.ndarray:
    """The same polyline with its corners spaced evenly by arc length (ends kept)."""
    ring = np.vstack((points, points[:1])) if closed else points
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    at = np.concatenate(([0.0], np.cumsum(seg)))
    if at[-1] <= 1e-300:
        return points.copy()
    count = len(points)
    want = np.linspace(0.0, at[-1], count + 1 if closed else count)[: count]
    out = np.stack([np.interp(want, at, ring[:, k]) for k in range(3)], axis=1)
    if not closed:
        out[-1] = points[-1]
    return out


def _taubin(points: np.ndarray, closed: bool, fixed: np.ndarray) -> np.ndarray:
    lam, mu, rounds = TAUBIN
    x = points.copy()
    for _r in range(rounds):
        for f in (lam, mu):
            mid = 0.5 * (np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0))
            step = f * (mid - x)
            if not closed:
                step[0] = step[-1] = 0.0
            step[fixed] = 0.0
            x = x + step
    return x


class TrueUpEdges(Trial):
    name = "True Up Edges"
    group = "HARD"
    handles = ("strength",)
    words = ("0", "1")
    defaults = {"strength": 1.0}
    blendable = True
    layers = ("scope", "ghost", "arrows", "held", "kept")
    stages = ("edges", "wave", "look")

    def check(self, base: Base) -> str:
        if not len(base.faces) or not base.scope.any():
            return NO_HARD_EDGES
        return "" if hard_chains(base.co, base.faces, base.scope) else NO_HARD_EDGES

    def run(self, base: Base, settings: dict, rough: bool = False):
        strength = float(settings.get("strength", self.defaults["strength"]))
        users = _edge_users(base.faces)
        free = movable(base, users)
        chains = hard_chains(base.co, base.faces, base.scope, users)
        due = _ticker()
        co = base.co.copy()
        target = base.co.copy()
        on_chain = np.zeros(len(co), dtype=bool)
        held: list = []
        trued = straight = 0
        waves_before: list = []  # per trued chain: largest distance off its crease, of its length
        waves_after: list = []
        trued_jobs = []
        jobs = []  # (corners, closed, crease line per corner, chain length)
        for ci, (path, closed) in enumerate(chains):
            idx = np.asarray(path, dtype=np.int64)
            on_chain[idx] = True
            pts = co[idx]
            ring = np.vstack((pts, pts[:1])) if closed else pts
            length = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum()) or base.edge
            left, right = _side_samples(base.co, base.faces, users, path, closed)
            sl = [np.asarray(p).reshape(-1, 3) for p in left]
            sr = [np.asarray(p).reshape(-1, 3) for p in right]
            flat_l = np.concatenate(sl) if sl else np.zeros((0, 3))
            flat_r = np.concatenate(sr) if sr else np.zeros((0, 3))
            ql = yield from _project(base, flat_l)
            qr = yield from _project(base, flat_r)
            cut_l = np.cumsum([0] + [len(p) for p in sl])
            cut_r = np.cumsum([0] + [len(p) for p in sr])
            n_edges = len(sl)
            lines = []
            for k in range(len(path)):
                lo, hi = k - WINDOW, k + WINDOW  # chain edges k-WINDOW .. k+WINDOW-1 touch corner k's window
                if closed:
                    ids = [(e % n_edges) for e in range(lo, hi)]
                else:
                    ids = list(range(max(lo, 0), min(hi, n_edges)))
                a = np.concatenate([ql[cut_l[e] : cut_l[e + 1]] for e in ids]) if ids else np.zeros((0, 3))
                b = np.concatenate([qr[cut_r[e] : cut_r[e + 1]] for e in ids]) if ids else np.zeros((0, 3))
                if len(a) < 3 or len(b) < 3:
                    lines.append(None)
                    continue
                nl, dl, _r, _k = _fit_plane_trimmed(a)
                nr, dr, _r, _k = _fit_plane_trimmed(b)
                lines.append(_line_of(nl, dl, nr, dr))
            jobs.append((idx, closed, lines, length))
            if due():
                yield 0.6 * (ci + 1) / max(len(chains), 1)
        for ci, (idx, closed, lines, length) in enumerate(jobs):
            pts = co[idx]
            ends = np.zeros(len(idx), dtype=bool)
            if not closed:
                ends[0] = ends[-1] = True
            fixed = ends | ~free[idx]
            crease = pts.copy()
            missing = np.zeros(len(idx), dtype=bool)
            for k, line in enumerate(lines):
                if line is None:
                    missing[k] = True
                    continue
                c = _on_line(pts[k], line)
                if float(np.linalg.norm(c - pts[k])) > CREASE_REACH * base.edge:
                    missing[k] = True
                    continue
                crease[k] = c
            found = np.flatnonzero(~missing)
            if len(found):
                on = yield from _project(base, crease[found])
                off = np.linalg.norm(on - crease[found], axis=1) > ON_SOURCE * base.edge
                missing[found[off]] = True
            inner = ~ends
            if (missing & inner).sum() > 0.5 * max(int(inner.sum()), 1) or not (inner & ~fixed).any():
                if (~missing & inner).any():  # a crease along part of it only
                    held.append((pts[len(pts) // 2].copy(), "no crease under it"))
                continue
            waves_before.append(float(np.linalg.norm(crease - pts, axis=1)[~missing].max(initial=0.0)) / length)
            fixed = fixed | missing
            # straight: one principal-axis line through every crease point
            centre = crease.mean(axis=0)
            _u, _s, vt = np.linalg.svd(crease - centre, full_matrices=False)
            axis = vt[0]
            off = crease - centre - np.outer((crease - centre) @ axis, axis)
            if not closed and float(np.linalg.norm(off, axis=1).max()) < STRAIGHT * length:
                new = centre + np.outer((pts - centre) @ axis, axis)
                new[fixed] = pts[fixed]
                new = _even(new, False)
                straight += 1
            else:
                new = _taubin(crease, closed, fixed)
                for k, line in enumerate(lines):
                    if line is not None and not fixed[k]:
                        new[k] = _on_line(new[k], line)
                new = _even(new, closed)
            new[fixed] = pts[fixed]
            target[idx] = new
            trued += 1
            trued_jobs.append((idx, lines, length, ~missing))
            if due():
                yield 0.6 + 0.2 * (ci + 1) / max(len(jobs), 1)
        moving = on_chain & free & np.any(np.abs(target - base.co) > 0.0, axis=1)
        co[moving] = base.co[moving] + strength * (target[moving] - base.co[moving])
        # the corners beside a chain follow along their rungs
        push = np.zeros_like(co)
        count = np.zeros(len(co))
        for (a, b), _fs in users.items():
            for x, y in ((a, b), (b, a)):
                if moving[x] and not on_chain[y] and free[y]:
                    push[y] += co[x] - base.co[x]
                    count[y] += 1
        follow = count > 0
        co[follow] = base.co[follow] + FOLLOW * push[follow] / count[follow, None]
        changed = moving | follow
        if changed.any():
            placed = yield from _project(base, co[changed])
            co[changed] = placed
        _keep_plane(base, co)
        co = _guard(base, co, changed, held)
        for idx, lines, length, has in trued_jobs:
            worst = 0.0
            for k, v in enumerate(idx.tolist()):
                if has[k]:
                    p, d = lines[k]
                    off = co[v] - p
                    worst = max(worst, float(np.linalg.norm(off - (off @ d) * d)) / length)
            waves_after.append(worst)
        wave_before = float(np.mean(waves_before)) if waves_before else 0.0
        wave_after = float(np.mean(waves_after)) if waves_after else 0.0
        kept = np.concatenate([np.stack((base.co[i[:-1]], base.co[i[1:]]), axis=1) for i, _c, _l, _n in jobs if len(i) > 1] or [np.zeros((0, 2, 3))])
        cand = Candidate(co, strength=strength, exact=True, held=held)
        cand.lines = kept
        if not trued:
            cand.refusal = NO_HARD_EDGES if not held else "No hard edges here follow a crease of your original mesh."
            return cand
        cand.note = f"{trued} edge{'s' if trued != 1 else ''} trued, {straight} straight"
        cand.numbers["edges"] = (cand.note, 0)
        cand.numbers["wave"] = (f"wave {100 * wave_before:.1f}% -> {100 * wave_after:.1f}%", 1 if wave_after < wave_before - 1e-4 else 0)
        return cand


# --------------------------------------------------------------------------- Press Panels Flat


def find_panels(co, faces, scope, edge: float, users: dict | None = None):
    """Generator: panel id per face (-1: none), grown loosely on the result over the faces in scope, largest faces
    seeding first, as Clean Panels grows them (`flatquad._grow_regions`). The bumpy result has no exact plane, so a
    face joins when it faces the panel's mean way within PANEL_DEG and its middle is within PANEL_GAP of an edge of the
    panel's mean plane (both kept up to date as faces join, without refits); it yields between faces."""
    co = np.asarray(co, dtype=np.float64)
    users = _edge_users(faces) if users is None else users
    raw = _newell(co, faces)
    areas = np.linalg.norm(raw, axis=1)
    normals = np.divide(raw, areas[:, None], out=np.zeros_like(raw), where=areas[:, None] > 1e-300)
    middles = np.stack([co[list(f)].mean(axis=0) for f in faces]) if faces else np.zeros((0, 3))
    neighbours: list[list[int]] = [[] for _ in faces]
    for fs in users.values():
        if len(fs) == 2:
            neighbours[fs[0]].append(fs[1])
            neighbours[fs[1]].append(fs[0])
    limit, gap = math.cos(math.radians(PANEL_DEG)), PANEL_GAP * edge
    labels = np.full(len(faces), -1, dtype=np.int64)
    due = _ticker()
    count = 0
    for seed in np.argsort(-areas, kind="stable").tolist():
        if labels[seed] >= 0 or not scope[seed]:
            continue
        labels[seed] = count
        weight, way, spot = areas[seed], raw[seed].copy(), middles[seed] * areas[seed]
        frontier = list(neighbours[seed])
        while frontier:
            f = frontier.pop()
            if labels[f] >= 0 or not scope[f]:
                continue
            n = way / (np.linalg.norm(way) or 1.0)
            if float(normals[f] @ n) < limit or abs(float((middles[f] - spot / weight) @ n)) > gap:
                continue
            labels[f] = count
            weight += areas[f]
            way += raw[f]
            spot += middles[f] * areas[f]
            frontier.extend(neighbours[f])
            if due():
                yield
        count += 1
        if due():
            yield
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    small = ids[counts < PANEL_FACES]
    labels[np.isin(labels, small)] = -1
    return labels


class PressPanelsFlat(Trial):
    name = "Press Panels Flat"
    group = "HARD"
    handles = ("strength",)
    words = ("0", "1")
    defaults = {"strength": 1.0}
    blendable = True
    layers = ("scope", "ghost", "wash", "arrows", "held", "kept")
    stages = ("bump", "panels", "look")

    def check(self, base: Base) -> str:
        if not len(base.faces) or not base.scope.any():
            return NO_PANELS
        return ""

    def run(self, base: Base, settings: dict, rough: bool = False):
        strength = float(settings.get("strength", self.defaults["strength"]))
        users = _edge_users(base.faces)
        free = movable(base, users)
        labels = yield from find_panels(base.co, base.faces, base.scope, base.edge, users)
        yield 0.1
        due = _ticker()
        planes = {}  # label -> (normal, offset, size, corners that go onto it)
        order = np.argsort(labels, kind="stable")
        starts = np.searchsorted(labels[order], np.arange(int(labels.max(initial=-1)) + 2))
        found = np.unique(labels[labels >= 0]).tolist()
        around: list[set] = [set() for _ in base.co]
        for j, r in enumerate(found):
            members = order[starts[r] : starts[r + 1]]
            corners = np.asarray(sorted({v for i in members.tolist() for v in base.faces[i]}), dtype=np.int64)
            mids = np.stack([base.co[list(base.faces[i])].mean(axis=0) for i in members.tolist()])
            under = yield from _project(base, np.vstack((base.co[corners], mids)))
            n, d, res, keep = _fit_plane_trimmed(under)
            if keep.mean() < 0.8 or float(np.percentile(res[keep], 95)) > FLAT * base.edge:
                labels[members] = -1  # your original is not flat here
                continue
            # a corner goes onto this plane only when your original under it lies on the plane
            on = corners[res[: len(corners)] <= 3.0 * FLAT * base.edge]
            if not len(on):
                labels[members] = -1
                continue
            size = float(np.linalg.norm(np.ptp(base.co[corners], axis=0))) or base.edge
            planes[r] = (n, d, size, on)
            for v in on.tolist():
                around[v].add(r)
            if due():
                yield 0.1 + 0.6 * (j + 1) / max(len(found), 1)
        cand = Candidate(base.co.copy(), strength=strength)
        if not planes:
            cand.refusal = NO_PANELS
            return cand

        def bump(points):  # largest distance off its panel's plane of a corner that may move, of the panel's size
            worst = 0.0
            for n, d, size, on in planes.values():
                on = on[free[on]]
                if len(on):
                    worst = max(worst, float(np.abs(points[on] @ n - d).max()) / size)
            return worst

        bump_before = bump(base.co)
        co = base.co.copy()
        moved = np.zeros(len(co), dtype=bool)
        held: list = []
        for v in np.flatnonzero(free).tolist():
            near = around[v]
            if not near:
                continue
            x0 = base.co[v]
            if len(near) == 1:
                n, d, _s, _on = planes[next(iter(near))]
                goal = x0 - (float(n @ x0) - d) * n
            else:
                a = PULL * np.eye(3)
                rhs = PULL * x0
                for r in near:
                    n, d, _s, _on = planes[r]
                    a += np.outer(n, n)
                    rhs = rhs + n * d
                goal = np.linalg.solve(a, rhs)
            co[v] = x0 + strength * (goal - x0)
            moved[v] = True
            if due():
                yield 0.7 + 0.25 * v / max(len(co), 1)
        _keep_plane(base, co)
        co = _guard(base, co, moved, held)
        bump_after = bump(co)
        shared = [(a, b) for (a, b), fs in users.items() if len(fs) == 2 and labels[fs[0]] >= 0 and labels[fs[1]] >= 0 and labels[fs[0]] != labels[fs[1]]]
        cand.co = co
        cand.held = held
        cand.lines = np.stack((base.co[[a for a, _b in shared]], base.co[[b for _a, b in shared]]), axis=1) if shared else np.zeros((0, 2, 3))
        count = len(planes)
        cand.note = f"{count} panel{'s' if count != 1 else ''} pressed flat"
        cand.numbers["panels"] = (f"{count} panel{'s' if count != 1 else ''}", 0)
        cand.numbers["bump"] = (f"bump {100 * bump_before:.1f}% -> {100 * bump_after:.1f}%", 1 if bump_after < bump_before - 1e-4 else 0)
        return cand
