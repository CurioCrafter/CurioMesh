# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Edge loops drawn by hand: a stroke becomes a strip of regular quads that is cut into the mesh about to be paired.

Pairing triangles into quads leaves poles on a thin guide chain, so an edge loop beside it stops after two or three
edges. Here the loop is built, not found: `grid` lays `count` parallel loops evenly along the stroke, the triangles
under that band are removed, and the band goes in as quads (two triangles each, every grid edge a hard edge, so
only the diagonal can be paired). The gap between the band and the rest is closed with a strip of triangles, so the
poles end up outside the drawn loops. Everything is NumPy and stays free of Blender.
"""

from __future__ import annotations

import numpy as np

from . import guides
from .triadq import patch

MARGIN = 0.75  # triangles this many gaps beyond the outer loop are cut away too; the gap they leave is zipped shut


def room(stroke: np.ndarray, count: int, gap: float) -> float:
    """The gap this stroke can take: a ring round a small thing (an eye) needs loops close enough to fit inside it."""
    if len(stroke) > 3 and float(np.linalg.norm(stroke[0] - stroke[-1])) < 1.5 * gap:
        inner = float(np.linalg.norm(stroke - stroke.mean(axis=0), axis=1).min())
        return min(gap, inner / (count / 2.0 + 1.2))
    return gap


def grid(piece: np.ndarray, count: int, gap: float, nearest):
    """(loops, closed): `count` parallel loops as one (count, columns, 3) array, columns `gap` apart along the stroke.

    A stroke whose two ends are within one and a half gaps of each other is a closed ring (the first column is not
    repeated). None when the stroke is too short for a strip.
    """
    closed = len(piece) > 3 and float(np.linalg.norm(piece[0] - piece[-1])) < 1.5 * gap
    if closed:
        piece = np.vstack((piece, piece[:1]))
    groups = guides.parallel([piece], count, gap, nearest)
    if not groups:
        return None
    loops = groups[0]
    centre = loops[len(loops) // 2]
    along = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))))
    columns = int(round(along[-1] / gap))
    if columns < (4 if closed else 1) or any(len(x) != len(centre) for x in loops):
        return None
    at = np.linspace(0.0, along[-1], columns, endpoint=False) if closed else np.linspace(0.0, along[-1], columns + 1)
    out = np.empty((count, len(at), 3))
    for k, loop in enumerate(loops):
        pts = np.stack([np.interp(at, along, loop[:, c]) for c in range(3)], axis=1)
        out[k] = np.asarray([nearest(p)[0] for p in pts.tolist()], dtype=np.float64)
    return out, closed


def _outline(rows: int, cols: int, closed: bool, base: int) -> list[list[int]]:
    """Vertex ids around the grid: one loop for a strip, two for a ring."""

    def at(r, c):
        return base + r * cols + c

    if closed:
        return [[at(0, c) for c in range(cols)], [at(rows - 1, c) for c in range(cols)]]
    return [
        [at(0, c) for c in range(cols)]
        + [at(r, cols - 1) for r in range(1, rows)]
        + [at(rows - 1, c) for c in range(cols - 2, -1, -1)]
        + [at(r, 0) for r in range(rows - 2, 0, -1)]
    ]


def _turn(xyz: np.ndarray, loop, normal: np.ndarray) -> float:
    p = xyz[loop]
    return float((np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0) * normal).sum())


def _fatness(t: np.ndarray) -> float:
    """Area over the squared longest side: large for a fat triangle, near 0 for a sliver."""
    area = 0.5 * float(np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])))
    return area / max(float(max(np.linalg.norm(t[k] - t[(k + 1) % 3]) for k in range(3))) ** 2, 1e-30)


def _zip(xyz: np.ndarray, P: list[int], Q: list[int], mode: int) -> list[tuple[int, int, int]]:
    """Triangles that close the gap between two loops that run the same way. Mode 0 takes the shorter diagonal, 1 walks both in step by distance along them, 2 takes the fatter triangle."""
    a, b = len(P), len(Q)
    first = int(np.argmin(np.linalg.norm(xyz[P] - xyz[Q[0]], axis=1)))
    P = P[first:] + P[:first]

    def share(loop):
        p = xyz[loop + loop[:1]]
        at = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))))
        return at / max(at[-1], 1e-30)

    sp, sq = share(P), share(Q)
    out, i, j = [], 0, 0
    while i < a or j < b:
        pi, pn, qj, qn = P[i % a], P[(i + 1) % a], Q[j % b], Q[(j + 1) % b]
        if i >= a:
            turn_q = True
        elif j >= b:
            turn_q = False
        elif mode == 1:
            turn_q = bool(sq[j + 1] < sp[i + 1])
        elif mode == 2:
            turn_q = _fatness(xyz[[pi, qn, qj]]) > _fatness(xyz[[pi, pn, qj]])
        else:
            turn_q = bool(np.linalg.norm(xyz[pi] - xyz[qn]) < np.linalg.norm(xyz[pn] - xyz[qj]))
        out.append((pi, qn, qj) if turn_q else (pi, pn, qj))
        j += int(turn_q)
        i += int(not turn_q)
    return out


def _clean(near: np.ndarray, tris: np.ndarray, alive: np.ndarray, bodies: int):
    """Grow the cut-away set until its outline is clean: no islands of left-over triangles inside it, no pinched vertex.

    `bodies` is how many pieces should stay (one outside for a line, an inside and an outside for a ring).
    Returns (near, outline loops) or None.
    """
    near = near.copy()
    fa, fb = patch.face_adjacency(tris.tolist())
    for _ in range(1):  # a triangle with two sides on the cut is a spike of the outline: take it too
        both = np.concatenate((fa[near[fb]], fb[near[fa]]))
        near |= (np.bincount(both, minlength=len(tris)) >= 2) & alive
    for _ in range(8):
        rest = np.flatnonzero(alive & ~near)
        if not len(rest):
            return None
        fa, fb = patch.face_adjacency(tris[rest].tolist())
        label = patch._labels(len(rest), fa, fb)
        _kinds, inverse, count = np.unique(label, return_inverse=True, return_counts=True)
        island = ~np.isin(inverse.reshape(-1), np.argsort(-count)[:bodies])
        if island.any():
            near[rest[island]] = True
            continue
        holes, pinched = patch.boundary_loops(tris.tolist(), np.flatnonzero(near))
        if not pinched:
            return near, holes
        near |= np.isin(tris, pinched).any(axis=1) & alive
    return None


def _folded(xyz, tris, faced, where, gap, sign: float) -> np.ndarray:
    """Which of the triangles turn against the way the cut-away triangles faced (`sign` is 1, or -1 when they all turn backwards)."""
    p = xyz[np.asarray(tris)]
    turned = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    wanted = faced[np.linalg.norm(p.mean(axis=1)[:, None, :] - where[None], axis=2).argmin(axis=1)]
    size = np.linalg.norm(turned, axis=1)
    cosine = sign * np.einsum("ij,ij->i", turned, wanted) / np.maximum(size, 1e-30)
    return (cosine < -0.1) & (size > 1e-3 * gap * gap)


def _flip_folds(xyz, tris: list, first: int, faced, where, gap, sign: float) -> list:
    """Flip diagonals among the zip triangles (those from index `first` on) while that turns folded ones around."""
    tris = [tuple(t) for t in tris]
    for _ in range(4 * len(tris)):
        bad = np.flatnonzero(_folded(xyz, tris, faced, where, gap, sign))
        bad = bad[bad >= first]
        if not len(bad):
            break
        half = {}
        for i in range(first, len(tris)):
            a, b, c = tris[i]
            for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
                half[(u, v)] = (i, w)
        for i in bad.tolist():
            a, b, c = tris[i]
            done = False
            for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
                other = half.get((v, u))
                if other is None:
                    continue
                j, d = other
                trial = list(tris)
                trial[i], trial[j] = (w, u, d), (w, d, v)
                now = _folded(xyz, [tris[i], tris[j]], faced, where, gap, sign).sum()
                after = _folded(xyz, [trial[i], trial[j]], faced, where, gap, sign).sum()
                if after < now:
                    tris = trial
                    done = True
                    break
            if done:
                break
        else:
            break
    return tris


def _fill(co, tris, alive, fresh, stroke, G, closed, gap, margin, nearest):
    """Cut a strip's band out of the triangles and close the gap round it: (cut-away mask, new triangles) or a reason."""
    half = (G.shape[0] - 1) / 2.0 * gap
    away = guides._to_line(co, stroke)
    near = (away[tris].min(axis=1) < half + 0.5 * gap) | (guides._to_line(co[tris].mean(axis=1), stroke) < half + margin * gap)  # no vertex that stays may lie under the band
    if not near.any() or (near & (~alive | fresh)).any():
        return "overlaps another loop or the mesh border"
    cleaned = _clean(near, tris, alive, 2 if closed else 1)
    if cleaned is not None and closed and len(cleaned[1]) == 1:
        return "ring too small for its loops: draw it bigger or lower Loop Gap"
    if cleaned is None or len(cleaned[1]) != (2 if closed else 1):
        return "outline is not one clean loop"
    near, holes = cleaned
    rows, cols = G.shape[:2]
    base = len(co)
    xyz = np.vstack((co, G.reshape(-1, 3)))
    rings = _outline(rows, cols, closed, base)
    gone = co[tris[near]]
    faced = np.cross(gone[:, 1] - gone[:, 0], gone[:, 2] - gone[:, 0])
    faced /= np.maximum(np.linalg.norm(faced, axis=1, keepdims=True), 1e-30)
    where = gone.mean(axis=1)
    quad = []
    for r in range(rows - 1):
        for c in range(cols if closed else cols - 1):
            a, b = base + r * cols + c, base + r * cols + (c + 1) % cols
            d, e = base + (r + 1) * cols + c, base + (r + 1) * cols + (c + 1) % cols
            quad += [(a, b, e), (a, e, d)]
    grid_xyz = xyz
    for mode in (0, 2, 1):  # a zip that folds is no use: try another way of walking
        xyz = grid_xyz
        new, used = [], set()
        for loop in holes:
            mine = xyz[loop]
            pick = min((r for r in range(len(rings)) if r not in used), key=lambda r: float(np.linalg.norm(mine[:, None, :] - xyz[rings[r]][None], axis=2).min(axis=1).mean()))
            used.add(pick)
            ring = rings[pick]
            n = np.asarray(nearest(mine.mean(axis=0))[1], dtype=np.float64)
            if _turn(xyz, loop, n) * _turn(xyz, ring, n) < 0:
                ring = ring[::-1]
            new += _zip(xyz, list(loop), list(ring), mode)
        every = quad + new
        p = xyz[np.asarray(every)]
        turned = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        wanted = faced[np.linalg.norm(p.mean(axis=1)[:, None, :] - where[None], axis=2).argmin(axis=1)]  # the way the triangles that were cut away faced
        sign = -1.0 if (np.einsum("ij,ij->i", turned, wanted) < 0.0).mean() > 0.5 else 1.0
        if _folded(xyz, every, faced, where, gap, sign).any():
            every = _flip_folds(xyz, every, len(quad), faced, where, gap, sign)
        if not _folded(xyz, every, faced, where, gap, sign).any():  # no triangle may turn against the others
            return near, xyz, [(t[0], t[2], t[1]) for t in every] if sign < 0 else every
    return "quads fold on the surface"


def insert(co: np.ndarray, tris: np.ndarray, kind: np.ndarray, items, base_gap: float, nearest, keys=()):
    """Cut the strips into a triangle mesh.

    `items` is a list of (stroke, loop count, tag). Returns (co, tris, kind, edge keys, built, skipped): `built` lists
    (tag, loops) for each strip that went in, every loop a polyline; `skipped` lists why each strip that could not go in
    did not. `nearest(p)` gives (closest surface point, normal). `keys` are hard edges (lo, hi) the caller already
    has; they come back renumbered with the strips' own.
    """
    co = np.asarray(co, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    kind = np.asarray(kind, dtype=np.int64)
    alive = np.ones(len(tris), dtype=bool)
    fresh = np.zeros(len(tris), dtype=bool)  # triangles of strips already in
    keys = {tuple(k) for k in keys}
    built, skipped = [], []
    for stroke, count, tag in items:
        gap = room(stroke, count, base_gap)
        made = grid(stroke, count, gap, nearest)
        if made is None:
            skipped.append("too short")
            continue
        G, closed = made
        for margin in (MARGIN, MARGIN + 0.6, MARGIN + 1.2):  # a rougher cut may need a wider band round it to close without folds
            done = _fill(co, tris, alive, fresh, stroke, G, closed, gap, margin, nearest)
            if not isinstance(done, str) or not done.startswith("quads fold"):
                break
        if isinstance(done, str):
            skipped.append(done)
            continue
        near, xyz, fixed = done
        rows, cols = G.shape[:2]
        base = len(co)
        for r in range(rows):
            for c in range(cols if closed else cols - 1):
                keys.add(tuple(sorted((base + r * cols + c, base + r * cols + (c + 1) % cols))))
        for r in range(rows - 1):
            for c in range(cols):
                keys.add(tuple(sorted((base + r * cols + c, base + (r + 1) * cols + c))))
        style = int(np.bincount(kind[np.flatnonzero(near)]).argmax())
        co = xyz
        alive[near] = False
        alive = np.concatenate((alive, np.ones(len(fixed), dtype=bool)))
        fresh = np.concatenate((fresh, np.ones(len(fixed), dtype=bool)))
        tris = np.vstack((tris, np.asarray(fixed, dtype=np.int64)))
        kind = np.concatenate((kind, np.full(len(fixed), style, dtype=np.int64)))
        built.append((tag, [np.vstack((g, g[:1])) if closed else g for g in G]))
    tris, kind = tris[alive], kind[alive]
    used = np.unique(tris)
    remap = np.full(len(co), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    keys = {(int(remap[a]), int(remap[b])) for a, b in keys if remap[a] >= 0 and remap[b] >= 0}
    return co[used], remap[tris], kind, keys, built, skipped


def split_region(co, faces, materials, region):
    """((co, faces, materials) of the faces in `region`, the same for the rest), each with its own compact vertex numbering."""
    region = np.asarray(region, dtype=bool)
    out = []
    for take in (region, ~region):
        ids = np.flatnonzero(take)
        used = np.unique(np.fromiter((v for i in ids.tolist() for v in faces[i]), dtype=np.int64))
        remap = np.full(len(co), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        out.append((np.asarray(co)[used], [tuple(int(remap[v]) for v in faces[i]) for i in ids.tolist()], np.asarray(materials)[ids]))
    return out[0], out[1]


def join_outside(co, faces, kinds, outside):
    """The result for a region put back beside the faces that were left alone, with its seam snapped onto theirs.

    Returns (co, faces, kinds, new, seam): `new` marks the region's faces, `seam` is (snapped, region outline vertices).
    A seam vertex joins only when a vertex of the rest lies within 40% of an edge of it: where the two sides differ
    (they always do somewhat) a small gap or a T-junction stays, and `seam` tells how many matched.
    """
    oco, ofaces, omat = outside
    co = np.asarray(co, dtype=np.float64)

    def rim(fs):
        count = {}
        for f in fs:
            for k in range(len(f)):
                e = (min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)]))
                count[e] = count.get(e, 0) + 1
        return sorted({v for e, c in count.items() if c == 1 for v in e})

    mine, theirs = np.asarray(rim(faces), dtype=np.int64), np.asarray(rim(ofaces), dtype=np.int64)
    snap = {}
    if len(mine) and len(theirs):
        edge = float(np.median([np.linalg.norm(co[f[0]] - co[f[1]]) for f in faces])) or 1.0
        near = np.linalg.norm(co[mine][:, None, :] - oco[theirs][None], axis=2)
        pick, dist = near.argmin(axis=1), near.min(axis=1)
        taken = set()
        for k in np.argsort(dist):
            if dist[k] < 0.4 * edge and int(theirs[pick[k]]) not in taken:
                taken.add(int(theirs[pick[k]]))
                snap[int(mine[k])] = int(theirs[pick[k]])
    shift = len(oco)
    joined = [tuple(snap[v] if v in snap else v + shift for v in f) for f in faces]
    all_co = np.vstack((oco, co))
    all_faces = list(ofaces) + joined
    used = np.unique(np.fromiter((v for f in all_faces for v in f), dtype=np.int64))
    remap = np.full(len(all_co), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    new = np.concatenate((np.zeros(len(ofaces), dtype=bool), np.ones(len(joined), dtype=bool)))
    return all_co[used], [tuple(int(remap[v]) for v in f) for f in all_faces], list(omat) + list(kinds), new, (len(snap), len(mine))
