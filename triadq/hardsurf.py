# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Hard surface: rebuilt versions of a part, made the way an artist lays out a machined piece. The camera judges each
one like any other size of the part, so a wrong guess costs time and never quality.

Clean Panels, for a part made of flat faces (boxes, planks, plates, brackets, bevel strips). Flat regions are grown as
Flat Quads grows them. Their borders are cut into chains where three regions meet; each chain is simplified once and
shared by both sides, so every panel border is an exact edge loop and the result stays watertight. Each chain gets a
segment count from a target edge length, and the counts are balanced so the opposite sides of a four-cornered panel
match and every panel border is even (a quad mesh needs that). A convex panel is then one transfinite grid (a Coons
patch, Coons 1967): four corners, opposite sides equal, so a rectangle is an a x b grid along its own edges and a round
cap is a grid with four corner poles instead of a fan. A panel with holes or a notch is cut into triangles (constrained
Delaunay with a lattice inside, lined up with the panel's longest edge), paired, and split once (each triangle into
three quads, each quad into four, the Catmull-Clark split), so it is all quads too; its chains get even counts for that.
A chamfer or bevel strip is a long four-cornered panel, so it comes out as one clean row.

Even Cylinders, for a turned part whose rings touch in many places (a detailed casting) or that has no clean rings at
all (a scan). `parts.fit_revolve` walks its rings breadth first and can wander across such a part; here the profile is
the longest path through the minimum spanning tree of the rings, the usual way to read a curve back from samples
(the EMST curve idea, Figueiredo and Miranda Gomes 1995), and a scan is binned on a coarser grid. The part is then
swept by `parts.build_revolve`: a constant segment count around that is a multiple of four, straight rows along it and
grid caps.
"""

from __future__ import annotations

import math

import numpy as np

from . import parts
from .flatquad import _Unfit, _bridge_holes, _ear_clip, _face_frames, _grow_regions
from .patch import _key, _orient, _Planar

PANEL_DEG = 4.0  # faces whose normals differ by less than this can share a panel
PANEL_TOL = 0.001  # how far from its panel's plane a vertex may be, as a share of the part's diagonal
CHAIN_TOL = 0.001  # how far a border chain may straighten, as a share of the diagonal
MOST_REGIONS = 150  # a part with more panels than this is too busy to lay out by panels
REGION_SHARE = 0.3  # ... and one with more panels than this share of its faces is curved, not made of panels
CORNER_DEG = 25.0  # a turn of the outline past this is a corner of the panel
CONVEX_SIN = 0.05  # an outline turning the wrong way by less than this (sine) still counts as convex
PANEL_STEPS = (0.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0)  # target edge length as a share of the diagonal; 0: as few faces as can be
MOST_QUADS = 20_000
PANEL_MOST = 0.6  # a layout with more quads than this share of the part's triangles is not offered
BALANCE_ROUNDS = 400
SMOOTH_ROUNDS = 12

# How many faces a rebuilt layout counts as, per face, when the search weighs faces against change (a reduced size
# counts 1, a paired quad QUAD_WEIGHT): a clean layout that holds the view wins over a reduced one with up to about
# twice the faces, as an artist would choose.
WEIGHTS = {"panels": 0.5, "cylinders": 0.6}

TURN_CELLS = (0.004, 0.015)  # ring grouping: a clean part first, then a scan (share of the diagonal)
TURN_COVER = 0.9  # share of vertices that must lie near the profile for the part to count as turned
TURN_NEAR = 2.0  # a vertex this many cells from the profile is on it
HOLLOW_APART = 2.0  # a height band of a scan has two walls when their radii lie this many spreads apart
HOLLOW_SHARE = 0.7  # ...and a scan is a hollow tube when this share of its bands do
ROUND_VISIBLE = 0.05  # a round part this big against the whole model keeps enough faces to stay round
ROUND_TRIANGLES = 1.6  # triangles per quad of its sweep it keeps at least (pairing makes about two triangles one quad)
TURN_SUPPORT = 0.95  # share of the swept surface that must have the part near it
TURN_SPREAD = 0.3  # each sixteenth of the turn needs at least this share of an even spread of vertices
SCAN_FACES = 2000  # the coarse (scan) grid is only tried on parts with at least this many faces


# --------------------------------------------------------------------------- Clean Panels


def _dp3(points: np.ndarray, tol: float) -> list[int]:
    """Indices kept when a 3D polyline is simplified so no dropped point is farther than `tol` (Douglas and Peucker 1973)."""

    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        span = points[b] - points[a]
        length2 = float(span @ span)
        inner = points[a + 1 : b] - points[a]
        if length2 < 1e-30:
            dist = np.linalg.norm(inner, axis=1)
        else:
            t = np.clip(inner @ span / length2, 0.0, 1.0)
            dist = np.linalg.norm(inner - t[:, None] * span, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep.add(a + 1 + k)
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return sorted(keep)


def _chains(polys, labels):
    """Border chains between panels: list of (vertex path, region a, region b); b is -1 on an open border."""

    users: dict[tuple[int, int], list[int]] = {}
    for f, p in enumerate(polys):
        for i in range(len(p)):
            users.setdefault(_key(p[i - 1], p[i]), []).append(f)
    border: dict[tuple[int, int], tuple[int, int]] = {}
    for e, fs in users.items():
        if len(fs) > 2:
            return None
        a = int(labels[fs[0]])
        b = int(labels[fs[1]]) if len(fs) == 2 else -1
        if a != b:
            border[e] = (min(a, b), max(a, b)) if b >= 0 else (a, -1)
    at: dict[int, list[tuple[int, int]]] = {}
    for e in border:
        at.setdefault(e[0], []).append(e)
        at.setdefault(e[1], []).append(e)
    breaks = {v for v, es in at.items() if len(es) != 2 or border[es[0]] != border[es[1]]}
    seen: set[tuple[int, int]] = set()
    out = []

    def walk(start, first):
        path, e = [start], first
        while True:
            seen.add(e)
            v = e[0] if e[1] == path[-1] else e[1]
            path.append(v)
            if v in breaks or v == start:
                return path
            nxt = [x for x in at[v] if x not in seen]
            if not nxt:
                return path
            e = nxt[0]

    for v in sorted(breaks):
        for e in at[v]:
            if e not in seen:
                out.append((walk(v, e), *border[e]))
    for e in list(border):
        if e not in seen:  # a closed loop with no corner on it: cut it in two
            loop = walk(e[0], e)
            if len(loop) < 4:
                return None
            half = len(loop) // 2
            out.append((loop[: half + 1], *border[e]))
            out.append((loop[half:], *border[e]))
    return out


def _loops_of(region: int, chains) -> list[list[tuple[int, bool]]] | None:
    """The region's border as loops of (chain, forwards)."""

    ends: dict[int, list[tuple[int, bool]]] = {}
    mine = [c for c, (path, a, b) in enumerate(chains) if region in (a, b)]
    for c in mine:
        path = chains[c][0]
        ends.setdefault(path[0], []).append((c, True))
        ends.setdefault(path[-1], []).append((c, False))
    if any(len(v) != 2 for v in ends.values()):
        return None  # the panel touches itself at a point
    used: set[int] = set()
    loops = []
    for c in mine:
        if c in used:
            continue
        loop, cur, fwd = [], c, True
        while cur not in used:
            used.add(cur)
            loop.append((cur, fwd))
            path = chains[cur][0]
            end = path[-1] if fwd else path[0]
            other = [x for x in ends[end] if x != (cur, not fwd)]
            if not other:
                return None
            cur, fwd = other[0]
        loops.append(loop)
    return loops


def _turns(poly2: np.ndarray) -> np.ndarray:
    """Signed sine of the turn at each vertex of a closed 2D polygon (positive turns left)."""

    a = poly2 - np.roll(poly2, 1, axis=0)
    b = np.roll(poly2, -1, axis=0) - poly2
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return cross / np.maximum(la * lb, 1e-300)


def _angles(poly2: np.ndarray) -> np.ndarray:
    """Turn angle in degrees at each vertex of a closed 2D polygon (0 = straight on)."""

    a = poly2 - np.roll(poly2, 1, axis=0)
    b = np.roll(poly2, -1, axis=0) - poly2
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cos = (a * b).sum(axis=1) / np.maximum(la * lb, 1e-300)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _coons(ring: np.ndarray, corners: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """Transfinite grid inside a closed 2D ring. `corners`: four ring indices, sides a, b, a, b long.

    Returns (points of the grid as (row, column) -> index into ring or new, new points) as a full index grid."""

    n = len(ring)
    c0, c1, c2, c3 = corners
    a, b = (c1 - c0) % n, (c2 - c1) % n
    if a < 1 or b < 1 or (c3 - c2) % n != a or (c0 - c3) % n != b:
        return None
    at = np.full((a + 1, b + 1), -1, dtype=np.int64)
    for i in range(a + 1):
        at[i, 0] = (c0 + i) % n
        at[a - i, b] = (c2 + i) % n
    for j in range(b + 1):
        at[a, j] = (c1 + j) % n
        at[0, b - j] = (c3 + j) % n
    grid = np.zeros((a + 1, b + 1, 2))
    for i in range(a + 1):
        for j in range(b + 1):
            if at[i, j] >= 0:
                grid[i, j] = ring[at[i, j]]
    s = np.linspace(0.0, 1.0, a + 1)[:, None, None]
    t = np.linspace(0.0, 1.0, b + 1)[None, :, None]
    bottom, top = grid[:, :1], grid[:, b:]
    left, right = grid[:1, :], grid[a:, :]
    corner = (1 - s) * (1 - t) * grid[0, 0] + s * (1 - t) * grid[a, 0] + (1 - s) * t * grid[0, b] + s * t * grid[a, b]
    inner = (1 - t) * bottom + t * top + (1 - s) * left + s * right - corner
    grid[1:a, 1:b] = inner[1:a, 1:b]
    for _ in range(SMOOTH_ROUNDS if a > 1 and b > 1 else 0):  # evens a curved outline's grid; a straight one stays as it is
        grid[1:a, 1:b] = 0.25 * (grid[:-2, 1:b] + grid[2:, 1:b] + grid[1:a, :-2] + grid[1:a, 2:])
    return grid, at


def _best_corners(ring: np.ndarray, natural: list[int]) -> list[int] | None:
    """Four ring indices making sides a, b, a, b, on the sharpest turns of the outline."""

    n = len(ring)
    if n < 4 or n % 2:
        return None
    if len(natural) == 4:
        sides = [(natural[(k + 1) % 4] - natural[k]) % n for k in range(4)]
        if sides[0] == sides[2] and sides[1] == sides[3]:
            return list(natural)
    sharp = _angles(ring)
    half = n // 2
    best, pick = -1e30, None
    starts = natural or range(n)
    for i0 in starts:
        for a in range(1, half):
            ids = [i0 % n, (i0 + a) % n, (i0 + half) % n, (i0 + half + a) % n]
            score = float(sharp[ids].sum()) - 0.5 * abs(a - (half - a))  # sharp corners first, then square
            if score > best:
                best, pick = score, ids
    return pick


def _valid(points2: np.ndarray, quads) -> bool:
    """Every quad turns left (counter-clockwise) with a real area."""

    if not len(quads):
        return False
    q = points2[np.asarray(quads)]
    x, y = q[:, :, 0], q[:, :, 1]
    area = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    scale = float(np.ptp(points2, axis=0).max()) or 1.0
    if (area <= 1e-10 * scale * scale).any():
        return False
    turns = (np.roll(x, -1, axis=1) - x) * (np.roll(y, -2, axis=1) - np.roll(y, -1, axis=1)) - (np.roll(y, -1, axis=1) - y) * (np.roll(x, -2, axis=1) - np.roll(x, -1, axis=1))
    return bool((turns > -1e-9 * scale * scale).all())


def _pair_triangles(p2: np.ndarray, tris: np.ndarray) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Greedy pairing of CCW triangles into convex quads, squarest first. Returns (quads, triangles left)."""

    users: dict[tuple[int, int], list[int]] = {}
    for t, tri in enumerate(tris.tolist()):
        for k in range(3):
            users.setdefault(_key(tri[k], tri[(k + 1) % 3]), []).append(t)
    cand = []
    for (u, w), ts in users.items():
        if len(ts) != 2:
            continue
        t0, t1 = ts
        a = [x for x in tris[t0] if x not in (u, w)][0]
        b = [x for x in tris[t1] if x not in (u, w)][0]
        tri = tris[t0].tolist()
        k = tri.index(a)
        quad = (a, tri[(k + 1) % 3], b, tri[(k + 2) % 3])
        q = p2[list(quad)]
        e = np.roll(q, -1, axis=0) - q
        cross = e[:, 0] * np.roll(e, -1, axis=0)[:, 1] - e[:, 1] * np.roll(e, -1, axis=0)[:, 0]
        if (cross <= 0.0).any():
            continue
        cos = -(e * np.roll(e, -1, axis=0)).sum(axis=1) / np.maximum(np.linalg.norm(e, axis=1) * np.linalg.norm(np.roll(e, -1, axis=0), axis=1), 1e-300)
        worst = float(np.degrees(np.abs(np.arccos(np.clip(cos, -1, 1)) - math.pi / 2)).max())
        if worst < 60.0:
            cand.append((worst, t0, t1, quad))
    cand.sort(key=lambda c: c[0])
    used = np.zeros(len(tris), dtype=bool)
    quads = []
    for _w, t0, t1, quad in cand:
        if not used[t0] and not used[t1]:
            used[t0] = used[t1] = True
            quads.append(quad)
    return quads, [tuple(t) for t in tris[~used].tolist()]


def _fallback(p2: np.ndarray, loops: list[list[int]], spacing: float):
    """Quads for a panel with holes or a notch: the base outline (every other sample), constrained Delaunay with a lattice
    inside, paired, then split once. Returns (new 2D points, quads over [p2 indices, new...]) or None."""

    if any(len(loop) % 2 or len(loop) < 6 for loop in loops):
        return None
    base = [loop[::2] for loop in loops]
    mid = {}
    for loop in loops:  # the sample between two base vertices is that base edge's middle
        for k in range(0, len(loop), 2):
            mid[_key(loop[k], loop[(k + 2) % len(loop)])] = loop[k + 1]
    ids = sorted({v for loop in base for v in loop})
    local = {v: k for k, v in enumerate(ids)}
    uv = p2[ids]
    rings = [[local[v] for v in loop] for loop in base]
    areas = []
    for ring in rings:
        q = uv[ring]
        areas.append(0.5 * float((q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]).sum()))
    outer_at = int(np.argmax(np.abs(areas)))
    outer = rings[outer_at] if areas[outer_at] > 0 else rings[outer_at][::-1]
    holes = [r if a < 0 else r[::-1] for k, (r, a) in enumerate(zip(rings, areas)) if k != outer_at]
    try:
        tris = _ear_clip(uv, _bridge_holes(uv, outer, holes))
    except _Unfit:
        return None
    planar = _Planar(uv, tris, fixed=len(ids))
    planar.flips()
    if spacing > 0.0:  # a lattice inside, lined up with the panel (u is along its longest edge)
        low, high = uv.min(axis=0), uv.max(axis=0)
        xs = np.arange(low[0] + spacing, high[0], spacing)
        ys = np.arange(low[1] + spacing, high[1], spacing)
        if len(xs) * len(ys) <= 4 * MOST_QUADS:
            from .patch import _inside_polygon, _polygon_distance

            grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
            if len(grid):
                inside = _inside_polygon(grid, uv[outer])
                for hole in holes:
                    inside &= ~_inside_polygon(grid, uv[hole])
                grid = grid[inside]
                if len(grid):
                    far = np.ones(len(grid), dtype=bool)
                    for ring in [outer] + holes:
                        far &= _polygon_distance(grid, uv[ring]) > 0.6 * spacing
                    grid = grid[far]
                for q in grid:
                    at = planar.walk(q, len(planar.t) - 1)
                    a, b, c = (planar.p[k] for k in planar.t[at])
                    if min(_orient(a, b, q), _orient(b, c, q), _orient(c, a, q)) <= 0.0:
                        continue
                    fresh = planar.insert(at, q)
                    planar.flips([_key(t[k], t[(k + 1) % 3]) for t in (planar.t[i] for i in fresh) for k in range(3)])
    pts, tri = planar.arrays()
    quads, left = _pair_triangles(pts, tri)
    # back to the caller's numbering: base vertices are p2 indices, lattice points are new
    n0 = len(p2)
    new: list[np.ndarray] = []
    name = {k: v for k, v in enumerate(ids)}
    for k in range(len(ids), len(pts)):
        name[k] = n0 + len(new)
        new.append(pts[k])
    edge_mid: dict[tuple[int, int], int] = {}

    def middle(x, y):
        key = _key(name[x], name[y])
        if key in mid:
            return mid[key]
        if key not in edge_mid:
            edge_mid[key] = n0 + len(new)
            new.append(0.5 * (pts[x] + pts[y]))
        return edge_mid[key]

    out = []
    for face in list(quads) + list(left):
        centre = n0 + len(new)
        new.append(pts[list(face)].mean(axis=0))
        m = len(face)
        for k in range(m):
            out.append((name[face[k]], middle(face[k], face[(k + 1) % m]), centre, middle(face[k - 1], face[k])))
    return (np.asarray(new).reshape(-1, 2), out)


def _rings(coords: np.ndarray, frame, outer: list[int], inner: list[int], h: float):
    """Rings of quads from a convex hole out to the convex outline round it (both the same count). Returns (new 3D
    points, quads over the given ids and -1-k for new point k) or None."""

    origin, _n, u, v = frame
    if len(outer) != len(inner) or len(outer) < 3:
        return None

    def flat(ids):
        d = coords[ids] - origin
        q = np.stack((d @ u, d @ v), axis=1)
        area = 0.5 * float((q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]).sum())
        return (ids, q) if area > 0 else (ids[::-1], q[::-1])

    outer, po = flat(outer)
    inner, pi = flat(inner)
    n = len(outer)
    shift = min(range(n), key=lambda s: float(np.linalg.norm(po - np.roll(pi, -s, axis=0), axis=1).sum()))
    inner, pi = inner[shift:] + inner[:shift], np.roll(pi, -shift, axis=0)
    gap = float(np.linalg.norm(po - pi, axis=1).mean())
    layers = 1 if h <= 0.0 else max(1, int(round(gap / h)))
    rows = [list(outer)]
    flats = [po]
    new = []
    for layer in range(1, layers):
        t = layer / layers
        q = (1 - t) * po + t * pi
        rows.append([-1 - (len(new) + k) for k in range(n)])
        flats.append(q)
        new += [origin + x * u + y * v for x, y in q]
    rows.append(list(inner))
    flats.append(pi)
    faces, cells = [], []
    for a in range(layers):
        for k in range(n):
            faces.append((rows[a][k], rows[a][(k + 1) % n], rows[a + 1][(k + 1) % n], rows[a + 1][k]))
            cells.append((a * n + k, a * n + (k + 1) % n, (a + 1) * n + (k + 1) % n, (a + 1) * n + k))
    if not _valid(np.vstack(flats), cells):
        return None
    return new, faces


def panel_layout(points: np.ndarray, polys) -> dict | None:
    """Panels, their border chains and loops, worked out once per part. None if the part is not made of flat panels."""

    points = np.asarray(points, dtype=np.float64)
    polys = [tuple(int(v) for v in p) for p in polys]
    if len(polys) < 4:
        return None
    diag = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diag <= 0.0:
        return None
    normals, areas = _face_frames(points, polys)
    labels = _grow_regions(points, polys, normals, areas, np.zeros(len(polys), dtype=np.int64), PANEL_TOL * diag, math.cos(math.radians(PANEL_DEG)))
    regions = int(labels.max()) + 1
    if regions > MOST_REGIONS or regions > REGION_SHARE * len(polys) + 6:
        return None
    chains = _chains(polys, labels)
    if not chains:
        return None
    simple = [[path[k] for k in _dp3(points[path], CHAIN_TOL * diag)] for path, _a, _b in chains]  # once: both panels use it
    loops, frames = {}, {}
    for r in range(regions):
        found = _loops_of(r, chains)
        if not found:
            return None
        loops[r] = found
        members = labels == r
        n = (normals[members] * areas[members, None]).sum(axis=0)
        if np.linalg.norm(n) < 1e-30:
            return None
        n /= np.linalg.norm(n)
        verts = sorted({v for c, _f in found[0] for v in simple[c]})
        longest, u = -1.0, None
        for loop in found:
            for c, _f in loop:
                d = points[simple[c][1:]] - points[simple[c][:-1]]
                d = d - np.outer(d @ n, n)
                lengths = np.linalg.norm(d, axis=1)
                k = int(np.argmax(lengths))
                if lengths[k] > longest:
                    longest, u = float(lengths[k]), d[k] / max(lengths[k], 1e-300)
        if u is None or longest <= 0.0:
            return None
        frames[r] = (points[verts].mean(axis=0), n, u, np.cross(n, u))

    def outline(loop):
        """The loop's simplified vertices in order, as (vertex, chain, the chain segment leaving it)."""
        out = []
        for c, fwd in loop:
            last = len(simple[c]) - 1
            out += [(simple[c][k], c, k) for k in range(last)] if fwd else [(simple[c][k], c, k - 1) for k in range(last, 0, -1)]
        return out

    def convex(r, loop):
        """(segments, |area|) of a loop if it is convex, else None."""
        ring = outline(loop)
        if len(ring) < 3:
            return None
        origin, _n, u, v = frames[r]
        d = points[[x for x, _c, _j in ring]] - origin
        p2 = np.stack((d @ u, d @ v), axis=1)
        area = 0.5 * float((p2[:, 0] * np.roll(p2[:, 1], -1) - np.roll(p2[:, 0], -1) * p2[:, 1]).sum())
        if not (np.sign(area) * _turns(p2) >= -CONVEX_SIN).all():
            return None
        return [(c, j) for _x, c, j in ring], abs(area), p2

    # convex single-loop panels are grids, a convex panel round one convex hole is rings of quads round the hole;
    # the rest are cut into triangles and split
    coons, fourway, annulus = {}, {}, {}
    for r in range(regions):
        if len(loops[r]) == 2:
            both = [convex(r, loop) for loop in loops[r]]
            if all(both):
                outer = 0 if both[0][1] >= both[1][1] else 1
                annulus[r] = (outer, both[outer][0], both[1 - outer][0])
            continue
        if len(loops[r]) != 1:
            continue
        found = convex(r, loops[r][0])
        if found is not None:
            segs, _area, p2 = found
            coons[r] = segs
            corners = [k for k, a in enumerate(_angles(p2)) if a > CORNER_DEG]
            if len(corners) == 4:
                ends = corners[1:] + [corners[0] + len(segs)]
                fourway[r] = [[segs[i % len(segs)] for i in range(corners[k], ends[k])] for k in range(4)]
    return {"points": points, "diag": diag, "chains": chains, "simple": simple, "loops": loops, "frames": frames, "coons": coons, "fourway": fourway, "annulus": annulus, "regions": regions}


def panel_rebuild(points: np.ndarray, polys, step: float, kind: str = "0", layout: dict | None = None):
    """The part laid out as clean panels with a target edge of `step` x its diagonal (0: fewest faces). None if the part
    is not made of flat panels or the layout can't be made. Returns (points, quads, kinds)."""

    lay = panel_layout(points, polys) if layout is None else layout
    if lay is None:
        return None
    points, diag, chains, simple, loops, frames = lay["points"], lay["diag"], lay["chains"], lay["simple"], lay["loops"], lay["frames"]
    coons, fourway, annulus, regions = lay["coons"], lay["fourway"], lay["annulus"], lay["regions"]
    others = {r for r in range(regions) if r not in coons and r not in annulus}
    locked = [chains[c][1] in others or chains[c][2] in others for c in range(len(chains))]  # split once later: even counts
    h = step * diag
    seg_len = [np.linalg.norm(np.diff(points[simple[c]], axis=0), axis=1) for c in range(len(chains))]
    count = []
    for c in range(len(chains)):
        m = np.ones(len(seg_len[c]), dtype=np.int64) if h <= 0.0 else np.maximum(np.rint(seg_len[c] / h), 1).astype(np.int64)
        if locked[c]:
            m = np.maximum(2 * np.ceil(m / 2.0), 2).astype(np.int64)
        count.append(m)
    if sum(int(m.sum()) for m in count) > MOST_QUADS:
        return None

    def total(segs):
        return sum(int(count[c][j]) for c, j in segs)

    def bump(segs, by):
        """Add `by` segments, one at a time to the longest-per-segment part (two at a time where it must stay even)."""
        while by > 0:
            c, j = max(segs, key=lambda s: seg_len[s[0]][s[1]] / count[s[0]][s[1]])
            step = 2 if locked[c] else 1
            count[c][j] += step
            by -= step

    for _ in range(BALANCE_ROUNDS):
        changed = False
        for sd in fourway.values():  # opposite sides of a four-cornered panel get the same count
            for x, y in ((0, 2), (1, 3)):
                sx, sy = total(sd[x]), total(sd[y])
                if sx != sy:
                    bump(sd[x] if sx < sy else sd[y], abs(sx - sy))
                    changed = True
        for _outer, sx_segs, sy_segs in annulus.values():  # the hole and the outline round it: one ring of quads
            sx, sy = total(sx_segs), total(sy_segs)
            if sx != sy:
                bump(sx_segs if sx < sy else sy_segs, abs(sx - sy))
                changed = True
        if changed:
            continue
        odd = [r for r, segs in coons.items() if total(segs) % 2]  # and every grid panel an even border
        if not odd:
            break
        r = odd[0]
        free = [(c, j) for c, j in coons[r] if not locked[c]]
        if not free:
            return None
        partner = [s for s in free if (chains[s[0]][1] if chains[s[0]][2] == r else chains[s[0]][2]) in odd[1:] + [-1]]
        bump(partner or free, 1)
    else:
        return None
    if sum(int(m.sum()) for m in count) > MOST_QUADS:
        return None
    # the samples along every chain, shared by both panels; corners keep their source vertex
    out_points: list[np.ndarray] = []
    corner_id: dict[int, int] = {}

    def vertex(v):
        if v not in corner_id:
            corner_id[v] = len(out_points)
            out_points.append(points[v])
        return corner_id[v]

    samples = []
    for c in range(len(chains)):
        dp = simple[c]
        ids = [vertex(dp[0])]
        for j in range(len(dp) - 1):
            a, b = points[dp[j]], points[dp[j + 1]]
            m = int(count[c][j])
            for k in range(1, m):
                ids.append(len(out_points))
                out_points.append(a + (b - a) * (k / m))
            if j + 1 == len(dp) - 1:
                ids.append(vertex(dp[-1]))
            else:
                ids.append(len(out_points))
                out_points.append(points[dp[j + 1]])
        samples.append(ids)
    quads: list[tuple[int, ...]] = []
    for r in range(regions):
        origin, _n, u, v = frames[r]
        rings = []
        for loop in loops[r]:
            ring = []
            for c, fwd in loop:
                ring += (samples[c] if fwd else samples[c][::-1])[:-1]
            rings.append(ring)
        coords = np.asarray(out_points)
        if r in annulus:
            outer = rings[annulus[r][0]]
            inner = rings[1 - annulus[r][0]]
            made = _rings(coords, frames[r], outer, inner, h)
            if made is None:
                return None
            new, faces = made
            names = {k: k for k in outer + inner}
            for k, q in enumerate(new):
                names[-1 - k] = len(out_points)
                out_points.append(q)
            quads += [tuple(names[x] for x in f) for f in faces]
        elif r in coons:
            ring = rings[0]
            d = coords[ring] - origin
            p2 = np.stack((d @ u, d @ v), axis=1)
            area = 0.5 * float((p2[:, 0] * np.roll(p2[:, 1], -1) - np.roll(p2[:, 0], -1) * p2[:, 1]).sum())
            if area < 0:
                ring, p2 = ring[::-1], p2[::-1]
            natural = [k for k, a in enumerate(_angles(p2)) if a > CORNER_DEG]
            corners = _best_corners(p2, natural)
            made = None if corners is None else _coons(p2, corners)
            if made is None:
                return None
            grid, at = made
            a, b = at.shape[0] - 1, at.shape[1] - 1
            cells = [(i, j) for i in range(a) for j in range(b)]
            if not _valid(grid.reshape(-1, 2), [(i * (b + 1) + j, (i + 1) * (b + 1) + j, (i + 1) * (b + 1) + j + 1, i * (b + 1) + j + 1) for i, j in cells]):
                return None
            ids = np.empty_like(at)
            for i in range(a + 1):
                for j in range(b + 1):
                    if at[i, j] >= 0:
                        ids[i, j] = ring[at[i, j]]
                    else:
                        ids[i, j] = len(out_points)
                        out_points.append(origin + grid[i, j, 0] * u + grid[i, j, 1] * v)
            quads += [(int(ids[i, j]), int(ids[i + 1, j]), int(ids[i + 1, j + 1]), int(ids[i, j + 1])) for i, j in cells]
        else:
            every = sorted({x for ring in rings for x in ring})
            local = {x: k for k, x in enumerate(every)}
            d = coords[every] - origin
            p2 = np.stack((d @ u, d @ v), axis=1)
            made = _fallback(p2, [[local[x] for x in ring] for ring in rings], 0.0 if h <= 0.0 else 2.0 * h)
            if made is None:
                return None
            new, faces = made
            if not _valid(np.vstack([p2, new]) if len(new) else p2, faces):
                return None
            names = every + list(range(len(out_points), len(out_points) + len(new)))
            out_points += [origin + q[0] * u + q[1] * v for q in new]
            quads += [tuple(names[k] for k in f) for f in faces]
        if len(quads) > MOST_QUADS:
            return None
    users: dict[tuple[int, int], int] = {}
    for q in quads:
        for k in range(4):
            e = _key(q[k], q[(k + 1) % 4])
            users[e] = users.get(e, 0) + 1
    if any(x > 2 for x in users.values()) or len(set(map(frozenset, quads))) != len(quads):
        return None
    return np.asarray(out_points), quads, [kind] * len(quads)


# --------------------------------------------------------------------------- Even Cylinders


def _tree_path(count: int, edges: list[tuple[float, int, int]]) -> list[int]:
    """Longest path (by length) through the minimum spanning tree of a graph (Kruskal)."""

    parent = list(range(count))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    near: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for w, a, b in sorted(edges):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
            near[a].append((b, w))
            near[b].append((a, w))

    def far(start):
        dist, back, stack = {start: 0.0}, {start: -1}, [start]
        while stack:
            x = stack.pop()
            for y, w in near[x]:
                if y not in dist:
                    dist[y] = dist[x] + w
                    back[y] = x
                    stack.append(y)
        end = max(dist, key=dist.get)
        return end, back

    used = [k for k in range(count) if near[k]]
    if not used:
        return []
    biggest = max(used, key=lambda k: len(near[k]))
    one, _ = far(biggest)
    two, back = far(one)
    path = [two]
    while back[path[-1]] != -1:
        path.append(back[path[-1]])
    return path


def _near_line(q: np.ndarray, line: np.ndarray, closed: bool) -> np.ndarray:
    """Distance from each 2D point to a polyline."""

    a = line if closed else line[:-1]
    b = np.roll(line, -1, axis=0) if closed else line[1:]
    best = np.full(len(q), np.inf)
    for start in range(0, len(a), 256):
        aa, bb = a[start : start + 256], b[start : start + 256]
        ab = bb - aa
        t = np.clip(np.einsum("qkj,kj->qk", q[:, None, :] - aa[None], ab) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
        d = np.linalg.norm(q[:, None, :] - (aa[None] + t[..., None] * ab[None]), axis=2).min(axis=1)
        best = np.minimum(best, d)
    return best


def _frame(axis: np.ndarray, points: np.ndarray):
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    frame = np.stack([u, np.cross(axis, u), axis], axis=1)
    local = points @ frame
    center = (local.min(axis=0) + local.max(axis=0)) / 2.0
    return frame, center, local - center


def cover(fit: dict, points: np.ndarray) -> float:
    """Share of the part's vertices that lie on the fitted profile (within a few ring widths)."""

    local = np.asarray(points, dtype=np.float64) @ fit["frame"] - fit["center"]
    rz = np.stack([np.hypot(local[:, 0], local[:, 1]), local[:, 2]], axis=1)
    near = fit.get("cell", parts.CLUSTER) * fit["diag"] * TURN_NEAR
    line = np.asarray(fit["profile"], dtype=np.float64)
    if len(line) < 2:
        return 0.0
    return float((_near_line(rz, line, fit["closed"]) <= near).mean())


def _hollow_profile(rz: np.ndarray, bins: int = 32) -> np.ndarray | None:
    """The closed profile of a thin-walled tube (outer wall up, inner wall down), read per height band where the radii
    fall in two clear groups. None unless most bands show both walls."""

    z = rz[:, 1]
    edges = np.linspace(z.min(), z.max(), bins + 1)
    band = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, bins - 1)
    outer, inner, walls, used = [], [], 0, 0
    for k in range(bins):
        r = np.sort(rz[band == k, 0])
        if len(r) < 20:
            continue
        used += 1
        gaps = np.diff(r)
        cut = None
        best = 0.0
        for at in np.argsort(-gaps)[:5].tolist():  # the widest gaps first: the space between the walls
            lo, hi = r[: at + 1], r[at + 1 :]
            if min(len(lo), len(hi)) < 0.2 * len(r):
                continue
            score = (hi.mean() - lo.mean()) / max(lo.std() + hi.std(), 1e-30)
            if score > best:
                best, cut = score, at
        if cut is None or best < HOLLOW_APART:
            continue
        walls += 1
        mid = 0.5 * (edges[k] + edges[k + 1])
        outer.append((float(np.median(r[cut + 1 :])), mid))
        inner.append((float(np.median(r[: cut + 1])), mid))
    if used < 8 or walls < HOLLOW_SHARE * used:
        return None
    outer, inner = np.asarray(outer), np.asarray(inner)
    outer[0, 1] = inner[0, 1] = z.min()  # the end bands reach the rims, not just their middles
    outer[-1, 1] = inner[-1, 1] = z.max()
    return np.vstack([outer, inner[::-1]])


def _supported(fit: dict, cloud: np.ndarray, cell: float) -> float:
    """Share of the swept surface (profile and flat end caps) that has the part's own vertices or face middles near it,
    in (radius, height). A sweep across empty space (a cap on an open end, a profile through a gap) scores low."""

    line = np.asarray(fit["profile"], dtype=np.float64)
    seg = np.roll(line, -1, axis=0) - line if fit["closed"] else line[1:] - line[:-1]
    start = line if fit["closed"] else line[:-1]
    probe = [start + seg * t for t in (0.25, 0.5, 0.75)]
    if not fit["closed"]:
        for slot, end in ((0, line[0]), (1, line[-1])):
            if end[0] > 0.0 and not fit["open_end"][slot]:  # a flat cap will close this end
                probe.append(np.asarray([[end[0] * t, end[1]] for t in (0.25, 0.5, 0.75)]))
    probe = np.vstack(probe)
    tree = parts._turbo.point_tree(np.column_stack([cloud, np.zeros(len(cloud))])) if parts._turbo is not None else None  # None without the native core
    if tree is not None:
        gap = np.asarray(tree.nearest(np.column_stack([probe, np.zeros(len(probe))]))[1])
    else:
        gap = np.asarray([float(np.min(np.hypot(*(cloud - q).T))) for q in probe])
    return float((gap <= 3.0 * cell).mean())


def fit_turned(points: np.ndarray, polys) -> dict | None:
    """A turned part's axis and profile, read as a curve through its rings (see the module text). None if it isn't one."""

    points = np.asarray(points, dtype=np.float64)
    diag = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diag <= 0.0 or len(points) < 12:
        return None
    edges = set()
    for p in polys:
        for i in range(len(p)):
            edges.add(_key(int(p[i - 1]), int(p[i])))
    pairs = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)
    rim = parts._boundary_vertices(polys)
    centres = np.asarray([p[:3] for p in polys], dtype=np.int64)  # three corners give a face's middle near enough
    best = None
    for cell_share in TURN_CELLS:
        if cell_share != TURN_CELLS[0] and len(polys) < SCAN_FACES:
            break
        cell = cell_share * diag
        for axis in parts._axes(points):
            frame, center, local = _frame(axis, points)
            angle = np.arctan2(local[:, 1], local[:, 0])
            rz = np.stack([np.hypot(local[:, 0], local[:, 1]), local[:, 2]], axis=1)
            spread = np.bincount(((angle + math.pi) / (2 * math.pi) * 16).astype(np.int64).clip(0, 15), minlength=16)
            off_axis = rz[:, 0] > 2 * cell
            if off_axis.sum() < 8 or spread.min() < TURN_SPREAD * off_axis.sum() / 16.0:
                continue  # not turned all the way round this axis
            ids = parts._cluster(rz, cell)
            count = int(ids.max()) + 1
            mean = np.stack([np.bincount(ids, rz[:, k], count) for k in range(2)], axis=1) / np.bincount(ids, None, count)[:, None]
            a, b = ids[pairs[:, 0]], ids[pairs[:, 1]]
            keep = a != b
            link = {}
            for x, y in zip(a[keep].tolist(), b[keep].tolist()):
                link[_key(x, y)] = float(np.hypot(*(mean[x] - mean[y])))
            path = _tree_path(count, [(w, x, y) for (x, y), w in link.items()])
            if len(path) < 2:
                continue
            closed = _key(path[0], path[-1]) in link and len(path) > 3 and min(mean[path[0], 0], mean[path[-1], 0]) > 2 * cell
            profile = mean[path].copy()
            hollow = _hollow_profile(rz) if cell_share != TURN_CELLS[0] else None
            if hollow is not None:  # a scanned tube: its noisy walls touch in the ring graph, so read them apart
                profile, closed = hollow, True
            elif cell_share != TURN_CELLS[0] and len(profile) >= 5:  # a scan: settle the noise along the profile
                smooth = profile.copy()
                smooth[1:-1] = (profile[:-2] + 2 * profile[1:-1] + profile[2:]) / 4.0
                profile = smooth
            if not closed:
                for end in (0, -1):
                    if profile[end, 0] <= 2 * cell:
                        profile[end, 0] = 0.0
            fit = {"frame": frame, "center": center, "profile": profile, "closed": closed, "diag": diag, "cell": cell_share}
            share = float((_near_line(rz, profile, closed) <= TURN_NEAR * cell).mean())
            if best is None or share > best[0]:
                big = int(np.argmax(np.bincount(ids)))
                members = np.flatnonzero(ids == big)
                open_end = [False, False]
                if not closed and rim:
                    rim_ids = np.asarray(sorted(rim), dtype=np.int64)
                    for slot, k in enumerate((path[0], path[-1])):
                        open_end[slot] = bool((np.hypot(*(rz[rim_ids] - mean[k]).T) <= 3 * cell).any())
                fit.update(open_end=open_end, phase=float(math.atan2(local[members[0], 1], local[members[0], 0])), spin=int(len(members)), cover=share)
                middles = np.stack([np.hypot(*(local[centres, :2].mean(axis=1)).T), local[centres, 2].mean(axis=1)], axis=1)
                if _supported(fit, np.vstack([rz, middles]), cell) >= TURN_SUPPORT:
                    best = (share, fit)
        if best is not None and best[0] >= TURN_COVER:
            break
    if best is None or best[0] < TURN_COVER:
        return None
    return best[1]


def cylinder_rebuilds(points: np.ndarray, polys, kind: str = "0"):
    """Generator: the part swept from a fitted profile, fewest faces first, when `parts.rebuilds` would miss or misread it."""

    fit = fit_turned(points, polys)
    if fit is None:
        return
    old = parts.fit_revolve(points, polys)
    if old is not None:
        old = dict(old, cell=parts.CLUSTER)
        if cover(old, points) >= fit["cover"] - 0.02:
            return  # the ring walk read this part as well: its sweeps are offered already
    yield from parts.sweeps(fit, kind, 0.5 * fit["cell"])


def round_floor(points: np.ndarray, polys, model_diag: float) -> int:
    """Fewest triangles a reduced size of this part may keep when it is a visibly round part of the model (its radius at
    least ROUND_VISIBLE of the model's size): about what its 16-step sweep would take, so a pipe or barrel never ends up
    as a 6- or 12-sided prism however little of the view that changes. 0 when it isn't such a part."""

    fit = fit_turned(points, polys)
    if fit is None:
        old = parts.fit_revolve(points, polys)
        if old is None or cover(dict(old, cell=parts.CLUSTER), points) < TURN_COVER:
            return 0
        fit = dict(old, cell=parts.CLUSTER)
    radius = float(np.max(np.asarray(fit["profile"])[:, 0]))
    if radius < ROUND_VISIBLE * model_diag or parts.min_spin(fit) < parts.ROUND_SPIN:
        return 0
    built = parts.build_revolve(fit, parts.ROUND_SPIN, parts.REVOLVES[0][1], "0")
    return 0 if built is None else int(ROUND_TRIANGLES * len(built[1]))


def panel_rebuilds(points: np.ndarray, polys, kind: str = "0"):
    """Generator: Clean Panels layouts of the part, fewest faces first; nothing if it isn't made of flat panels."""

    layout = panel_layout(points, polys)
    if layout is None:
        return
    sizes = set()
    most = max(6.0, PANEL_MOST * sum(len(p) - 2 for p in polys))
    for step in PANEL_STEPS:
        out = panel_rebuild(points, polys, step, kind, layout)
        if out is not None and len(out[1]) > most:
            return  # more quads than reducing the part would leave: the camera would never pick it
        if out is not None and len(out[1]) not in sizes:
            sizes.add(len(out[1]))
            yield out


def rebuilds(points: np.ndarray, polys, kind: str = "0", panels: bool = True, cylinders: bool = True):
    """Generator of hard-surface rebuilds of a part for the camera to judge, as (points, quads, kinds, family)."""

    if cylinders:
        for out in cylinder_rebuilds(points, polys, kind):
            yield None if out is None else (*out, "cylinders")
    if panels:
        for out in panel_rebuilds(points, polys, kind):
            yield (*out, "panels")
