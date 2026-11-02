# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Quads in rows, built directly from a direction field and a lattice over the surface, with cells that shrink at detail.

Own implementation of the idea in Jakob, Tarini, Panozzo and Sorkine-Hornung, "Instant Field-Aligned Meshes" (2015): at
every point of a finely split copy of the surface, keep a cross direction and the position of a grid laid along it, and
smooth both from coarse to fine. Where this differs:

- Extraction: points whose nearest grid corner is the same form a region, and each triangle touching three regions
  becomes one output triangle. This always gives a surface, and a grid's cells come out as right triangles in pairs.
- Cell sizes follow a conformal scale solved from the field: where a grid would have to end rows to follow it, a slightly
  different size lets it carry on instead. Sweeps that trust neighbours whose grid already agrees gather what row ends
  remain into a few places.
- Each output vertex is fitted to the planes of the piece of source nearest to it, so a ridge inside a cell is kept.
- Two triangles at the ends of one row become quads by sliding that row, which adds no poles.

Plain NumPy. The only randomness is seeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .patch import _labels, quad_quality
from .topology import _turbo

SHARP_DEG = 60.0
CREASE_DEG = 20.0  # faces this far apart may be a feature line
CREASE_LENGTH = 0.1  # of the piece's size a line must reach before the grid is laid along it
SEGMENT_DETAIL = 0.5  # how finely a piece is cut into limbs, each with its own cell size
ROW_WIDTH = 4.0  # at least this many cells across a limb or panel
SEGMENT_TURN = 1.0  # radians the surface may turn across a cell, read per limb, so rows are whole inside each
SPLIT_SHARE = 0.4  # no edge of the split copy longer than this share of the local cell size
LINE_CELL, LINE_REACH = 0.5, 4.0  # cells along a drawn line, as a share of even ones, and the cells it takes to open out to even
SMALLEST = 0.5  # no cell under this share of the size that even cells would have
SMOOTH_REACH = 0.5
ORIENT_SWEEPS = 8
POSITION_SWEEPS = 8
ROBUST_SWEEPS = 4
ROBUST = 0.05  # gaps far above this share of a cell are left to be defects rather than spread over their neighbours
SAME_CORNER = 0.35
RELAX_ROUNDS = 8
RELAX_RATE = 0.5
MOST_REACH = 2.0  # cells
SETTLED = 0.02  # of a cell: a grid corner this close to its neighbours' is left alone
FINISH_ROUNDS = 4
MOST_TARGET = 30_000  # quads on one piece: beyond this a run takes many minutes in plain NumPy
MOST_SOURCE = 2_000_000  # triangles
CREASE_SMOOTH = 4
CORNER_COS = 0.5  # edges of a line this far from its direction make a corner
FAR_SIDE = 25.0  # how much further away a point facing the other way counts
SMALLEST_TARGET = 24
COUNT_SHARE = 0.88  # output quads per cell the sizes allow for: merged regions and row ends take the rest
GUARD_QUALITY = 0.1
UNTANGLE_BELOW = 0.05
WORST_CLOSING = 0.99
CONFIDENCE = 1.0
FINAL_SWEEPS = 16
FINEST_SWEEPS = 6
CONFORMAL = 1.0
MOST_LOG = 0.5
CAMERA_PULL = 1.0
MOST_FIT = 0.75  # share of a cell a vertex may move while fitting
SQUARE_ENOUGH = 0.6  # cosine of the worst corner a first-round pair may have
FIT_STRONG = 0.05  # a plane direction counts if this strong against the strongest: flat moves only along the normal


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def _edges(tris: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(ends a, ends b, edge per triangle side (T, 3), uses per edge), with a < b."""

    first, second = tris.reshape(-1), np.roll(tris, -1, axis=1).reshape(-1)
    low, high = np.minimum(first, second), np.maximum(first, second)
    keys, edge_of, uses = np.unique(low * count + high, return_inverse=True, return_counts=True)
    return keys // count, keys % count, edge_of.reshape(-1, 3), uses


def triangulate(faces) -> tuple[np.ndarray, np.ndarray]:
    """(triangles, source face of each), fans from each face's first corner."""

    tris, owner = [], []
    for index, face in enumerate(faces):
        for k in range(1, len(face) - 1):
            tris.append((face[0], face[k], face[k + 1]))
            owner.append(index)
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3), np.asarray(owner, dtype=np.int64)


def _normals_areas(points: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unit face normals, face areas, unit vertex normals weighted by area)."""

    cross = np.cross(points[tris[:, 1]] - points[tris[:, 0]], points[tris[:, 2]] - points[tris[:, 0]])
    twice = np.linalg.norm(cross, axis=1)
    vertex = np.stack([np.bincount(tris.reshape(-1), np.repeat(cross[:, k], 3), len(points)) for k in range(3)], axis=1)
    return cross / np.maximum(twice, 1e-30)[:, None], 0.5 * twice, _unit(vertex)


def _smooth(values: np.ndarray, a: np.ndarray, b: np.ndarray, rounds: int) -> np.ndarray:
    count = len(values)
    degree = np.bincount(a, minlength=count) + np.bincount(b, minlength=count)
    for _ in range(rounds):
        total = np.bincount(a, values[b], count) + np.bincount(b, values[a], count)
        values = (values + total) / (1.0 + degree)
    return values


def edge_turn(tris: np.ndarray, normals: np.ndarray, edge_of: np.ndarray, edge_count: int) -> np.ndarray:
    """Radians between the first two faces on each edge, 0 on a border."""

    order = np.argsort(edge_of.reshape(-1), kind="stable")
    ids = edge_of.reshape(-1)[order]
    same = np.flatnonzero(ids[:-1] == ids[1:])
    out = np.zeros(edge_count)
    cosine = (normals[order[same] // 3] * normals[order[same + 1] // 3]).sum(axis=1)
    out[ids[same]] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return out


def size_limits(points: np.ndarray, tris: np.ndarray, even: float, most_turn: float) -> np.ndarray:
    """Per vertex, the largest cell that keeps the surface's turn across it to `most_turn` radians, read at about an even cell's scale."""

    normals, areas, _vertex = _normals_areas(points, tris)
    a, b, edge_of, _uses = _edges(tris, len(points))
    turn = edge_turn(tris, normals, edge_of, len(a))
    length = np.linalg.norm(points[a] - points[b], axis=1)
    turn = np.minimum(turn, np.radians(SHARP_DEG))  # a crease is held by a row along it, not by very small cells
    bend = np.bincount(a, turn * length, len(points)) + np.bincount(b, turn * length, len(points))
    area = np.bincount(tris.reshape(-1), np.repeat(areas / 3.0, 3), len(points))
    rounds = int(np.clip(SMOOTH_REACH * (even / max(float(np.median(length)), 1e-30)) ** 2, 2, 40))
    curvature = _smooth(bend, a, b, rounds) / np.maximum(_smooth(area, a, b, rounds), 1e-30)  # radians per unit length
    return most_turn / np.maximum(curvature, 1e-12)


def segment_limits(points: np.ndarray, tris: np.ndarray, even: float, most_turn: float) -> np.ndarray:
    """Per vertex, its limb's largest cell: thin or bent limbs get small cells, and each limb's cells are all alike, so rows
    stay whole inside a limb and only end at its seam. Cut by `segment`; a part it can't cut is one limb."""

    from . import segment  # it uses this file for Animation rows

    bends = size_limits(points, tris, even, most_turn)
    try:
        faces = [tuple(t) for t in tris.tolist()]
        labels = segment.dense(segment.find(segment.build(points, faces), faces, SEGMENT_DETAIL))
    except ValueError:
        labels = np.zeros(len(tris), dtype=np.int64)
    vertex = np.zeros(len(points), dtype=np.int64)
    vertex[tris.reshape(-1)] = np.repeat(labels, 3)
    out = np.empty(len(points))
    order = np.argsort(vertex, kind="stable")
    for ids in np.split(order, np.flatnonzero(vertex[order][1:] != vertex[order][:-1]) + 1):
        here = points[ids] - points[ids].mean(axis=0)
        axes = np.linalg.eigh(here.T @ here)[1]
        width = float(np.ptp(here @ axes[:, 1]))  # across a tube or the shorter way along a panel
        out[ids] = min(float(np.median(bends[ids])), max(width, 1e-12) / ROW_WIDTH)
    return out


def fit_sizes(limits: np.ndarray, area: np.ndarray, target: float, shape: np.ndarray | None = None) -> np.ndarray:
    """Cell sizes covering `area` in about `target` cells: `shape`, capped by `limits`, all scaled together.

    `limits` count in units of the even cell size, so the bends of the surface decide how the cells are shared out and
    the target decides how many there are, whatever the object's scale.
    """

    shape = np.ones_like(limits) if shape is None else shape
    even = float(np.sqrt(max(area.sum(), 1e-30) / max(target, 1.0)))
    wanted = np.maximum(np.minimum(shape, limits / even), SMALLEST)
    low, high = np.log(even) - 6.0, np.log(even) + 6.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        sizes = np.exp(middle) * wanted
        if (area / (sizes * sizes)).sum() > target:
            low = middle
        else:
            high = middle
    return sizes


def split_long_edges(points: np.ndarray, tris: np.ndarray, sizes: np.ndarray, share: float = SPLIT_SHARE, carried: np.ndarray | None = None):
    """Split edges longer than `share` of their cell size, keeping the surface exactly. Returns (points, tris, sizes, carried)."""

    carried = np.zeros(len(points)) if carried is None else carried

    for _ in range(12):
        a, b, edge_of, _uses = _edges(tris, len(points))
        long = np.linalg.norm(points[a] - points[b], axis=1) > share * np.minimum(sizes[a], sizes[b])
        if not long.any():
            break
        new = np.full(len(a), -1, dtype=np.int64)
        new[long] = len(points) + np.arange(int(long.sum()))
        points = np.vstack((points, 0.5 * (points[a[long]] + points[b[long]])))
        sizes = np.concatenate((sizes, 0.5 * (sizes[a[long]] + sizes[b[long]])))
        carried = np.concatenate((carried, 0.5 * (carried[a[long]] + carried[b[long]])))
        marks = new[edge_of]
        count = (marks >= 0).sum(axis=1)
        parts = [tris[count == 0]]
        for many in (1, 2):  # turn each triangle so its marks start at side 0
            ids = np.flatnonzero(count == many)
            if not len(ids):
                continue
            mark, tri = marks[ids], tris[ids]
            turn = np.argmax(np.stack([(np.roll(mark, -k, axis=1) >= 0)[:, : many].all(axis=1) for k in range(3)], axis=1), axis=1)
            rows = np.arange(len(ids))[:, None]
            at = (turn[:, None] + np.arange(3)) % 3
            x, y, z = tri[rows, at].T
            m0, m1, _m2 = mark[rows, at].T
            if many == 1:
                parts += [np.stack((x, m0, z), axis=1), np.stack((m0, y, z), axis=1)]
            else:
                parts += [np.stack((m0, y, m1), axis=1), np.stack((x, m0, m1), axis=1), np.stack((x, m1, z), axis=1)]
        ids = np.flatnonzero(count == 3)
        if len(ids):
            x, y, z = tris[ids].T
            m0, m1, m2 = marks[ids].T
            parts += [np.stack(c, axis=1) for c in ((x, m0, m2), (m0, y, m1), (m2, m1, z), (m0, m1, m2))]
        tris = np.vstack(parts)
    return points, tris, sizes, carried


@dataclass
class _Level:
    points: np.ndarray
    normals: np.ndarray
    sizes: np.ndarray
    a: np.ndarray  # undirected edges, each once
    b: np.ndarray
    along: np.ndarray  # per vertex: a crease's direction, or zero
    pinned: np.ndarray  # per vertex: 0 free, 1 on a crease, 2 a corner
    parent: np.ndarray | None = None  # into the next coarser level
    q: np.ndarray | None = None
    o: np.ndarray | None = None
    misfit: np.ndarray | None = None


def long_creases(points: np.ndarray, a: np.ndarray, b: np.ndarray, turn: np.ndarray, degrees: float | None = None, share: float | None = None) -> np.ndarray:
    """Per edge: on a line of sharp edges, unbroken by crossings, reaching `share` of the object's size. Short ones are noise, and each breaks the rows around it."""

    sharp = np.flatnonzero(turn >= np.radians(CREASE_DEG if degrees is None else degrees))
    out = np.zeros(len(a), dtype=bool)
    if not len(sharp):
        return out
    ends = np.concatenate((a[sharp], b[sharp]))
    ids = np.concatenate((sharp, sharp))
    order = np.argsort(ends, kind="stable")
    ends, ids = ends[order], ids[order]
    plain = np.bincount(ends, minlength=len(points))[ends] == 2  # a vertex with two sharp edges joins them; more is a crossing
    link = np.flatnonzero((ends[:-1] == ends[1:]) & plain[:-1])
    lines = _labels(len(a), ids[link], ids[link + 1])
    lengths = np.linalg.norm(points[a] - points[b], axis=1)
    total = np.bincount(lines[sharp], lengths[sharp], len(a))
    out[sharp] = total[lines[sharp]] >= (CREASE_LENGTH if share is None else share) * float(np.ptp(points, axis=0).max())
    return out


def _features(points: np.ndarray, tris: np.ndarray, normals: np.ndarray, a, b, edge_of, uses) -> tuple[np.ndarray, np.ndarray]:
    """(direction along a crease or zero, 0 free / 1 on a crease or border / 2 a corner) per vertex."""

    sharp = long_creases(points, a, b, edge_turn(tris, normals, edge_of, len(a))) | (uses != 2)
    ends = np.concatenate((a[sharp], b[sharp]))
    count = np.bincount(ends, minlength=len(points))
    pinned = np.where(count == 0, 0, np.where(count <= 2, 1, 2))
    # Direction along the line: a mesh's zigzag averaged out over a few edges, each edge signed to follow its neighbours.
    ea, eb = a[sharp], b[sharp]
    direction = _unit(points[eb] - points[ea])
    along = np.zeros_like(points)
    seen = np.zeros(len(points), dtype=bool)
    for _ in range(CREASE_SMOOTH):
        total = np.zeros_like(points)
        for ends_here, sign in ((ea, 1.0), (eb, -1.0)):  # edge a -> b leaves a and enters b; through a vertex, one of each is straight
            first_time = ~seen[ends_here]
            flip = np.where(((along[ends_here] * direction).sum(axis=1) < 0.0) | (first_time & (sign < 0.0)), -1.0, 1.0)
            np.add.at(total, ends_here, flip[:, None] * direction)
        seen[ea] = True
        seen[eb] = True
        total[pinned == 2] = 0.0
        # share along the line
        spread = total.copy()
        same = np.where((total[eb] * total[ea]).sum(axis=1) >= 0.0, 0.5, -0.5)
        for k in range(3):
            spread[:, k] += np.bincount(ea, total[eb, k] * same, len(points)) + np.bincount(eb, total[ea, k] * same, len(points))
        along = _unit(spread)
    # A line that turns hard at a vertex has a corner there: its two edges point well apart.
    neighbour_dot = np.ones(len(points))
    np.minimum.at(neighbour_dot, ea, np.abs((direction * along[ea]).sum(axis=1)))
    np.minimum.at(neighbour_dot, eb, np.abs((direction * along[eb]).sum(axis=1)))
    pinned[(pinned == 1) & (neighbour_dot < CORNER_COS)] = 2
    along[pinned != 1] = 0.0
    return along, pinned


def _coarsen(level: _Level, rng: np.random.Generator) -> _Level | None:
    """Match neighbours and merge them; None when that stops shrinking the graph."""

    n = len(level.points)
    a, b = level.a, level.b
    length = np.linalg.norm(level.points[a] - level.points[b], axis=1)
    score = (0.5 + 0.5 * (level.normals[a] * level.normals[b]).sum(axis=1)) / np.maximum(length, 1e-30)
    score = score * (1.0 + 0.1 * rng.random(len(a)))
    mate = np.full(n, -1, dtype=np.int64)
    for _ in range(4):
        free = np.flatnonzero((mate[a] < 0) & (mate[b] < 0) & (level.pinned[a] + level.pinned[b] < 3))
        if not len(free):
            break
        free = free[np.argsort(score[free], kind="stable")]
        best = np.full(n, -1, dtype=np.int64)
        best[np.stack((a[free], b[free]), axis=1).reshape(-1)] = np.repeat(free, 2)  # the last write, the best score, wins
        wins = free[(best[a[free]] == free) & (best[b[free]] == free)]
        mate[a[wins]] = b[wins]
        mate[b[wins]] = a[wins]
    keep = np.flatnonzero((mate < 0) | (np.arange(n) < mate))
    if len(keep) >= 0.9 * n or len(keep) < 4:
        return None
    parent = np.zeros(n, dtype=np.int64)
    parent[keep] = np.arange(len(keep))
    paired = mate >= 0
    parent[paired] = parent[np.minimum(np.flatnonzero(paired), mate[paired])]
    level.parent = parent
    count = len(keep)
    weight = level.sizes**0
    total = np.bincount(parent, weight, count)
    # A crease must stay put: a merged point sits on its pinned child.
    rank = (level.pinned + 1.0) * 100.0 ** level.pinned.astype(float)
    bucket = np.bincount(parent, rank, count)
    share = rank / bucket[parent]
    points = np.stack([np.bincount(parent, level.points[:, k] * share, count) for k in range(3)], axis=1)
    normals = _unit(np.stack([np.bincount(parent, level.normals[:, k] * weight, count) for k in range(3)], axis=1))
    sizes = np.bincount(parent, level.sizes * weight, count) / total
    pinned = np.zeros(count, dtype=np.int64)
    np.maximum.at(pinned, parent, level.pinned)
    along = np.zeros((count, 3))
    keep_child = level.pinned == pinned[parent]
    along[parent[keep_child]] = level.along[keep_child]
    along[pinned != 1] = 0.0
    low, high = np.minimum(parent[a], parent[b]), np.maximum(parent[a], parent[b])
    keys = np.unique((low * count + high)[low != high])
    return _Level(points, normals, sizes, keys // count, keys % count, along, pinned)


def _hierarchy(points, tris, sizes, rng, forced=None) -> list[_Level]:
    normals, _areas, vertex = _normals_areas(points, tris)
    a, b, edge_of, uses = _edges(tris, len(points))
    along, pinned = _features(points, tris, normals, a, b, edge_of, uses)
    if forced is not None:  # lines drawn by hand count as creases
        on = forced[1] > 0
        along[on], pinned[on] = forced[0][on], forced[1][on]
    levels = [_Level(points, vertex, sizes, a, b, along, pinned)]
    while len(levels[-1].points) > 64:
        coarser = _coarsen(levels[-1], rng)
        if coarser is None:
            break
        levels.append(coarser)
    return levels


def _tangent(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - (v * n).sum(axis=1, keepdims=True) * n


def _native(name: str):
    module = _turbo.native() if _turbo is not None else None
    return getattr(module, name, None)


def _flat(values: np.ndarray, dtype=np.float64) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=dtype).reshape(-1)


def _orient_sweep(level: _Level, rng: np.random.Generator) -> None:
    q, n = level.q, level.normals
    colors = rng.integers(0, 3, len(q))
    fast = _native("field_orient_sweep")
    if fast is not None:  # the same sums in the same order, on all cores
        out = _flat(q).copy()
        fast(out, _flat(n), _flat(level.pinned, np.int64), _flat(level.a, np.int64), _flat(level.b, np.int64), _flat(colors, np.int64))
        q[:] = out.reshape(-1, 3)
        return
    head = np.concatenate((level.a, level.b))
    tail = np.concatenate((level.b, level.a))
    for color in range(3):
        use = colors[head] == color
        i, j = head[use], tail[use]
        wanted = _best_match(q[i], n[i], q[j], n[j])[0]
        total = np.stack([np.bincount(i, wanted[:, k], len(q)) for k in range(3)], axis=1)
        moving = (colors == color) & (level.pinned != 1) & (np.linalg.norm(total, axis=1) > 1e-12)
        q[moving] = _unit(_tangent(total[moving], n[moving]))


def _nearest_grid(o, q, t, s, target):
    """The corner of the grid (corner o, sides q and t, size s) nearest to `target`."""

    d = target - o
    return o + q * np.rint((d * q).sum(axis=1, keepdims=True) / s) * s + t * np.rint((d * t).sum(axis=1, keepdims=True) / s) * s


def _place(level: _Level, o: np.ndarray, ids: np.ndarray | None = None) -> np.ndarray:
    """Put each grid corner on its own tangent plane and crease, then back to the corner nearest its point."""

    ids = np.arange(len(o)) if ids is None else ids
    p, n, q = level.points[ids], level.normals[ids], level.q[ids]
    t = np.cross(n, q)
    s = level.sizes[ids][:, None]
    out = o - ((o - p) * n).sum(axis=1, keepdims=True) * n
    on = level.pinned[ids] == 1
    if on.any():  # the grid line along the crease runs through this point
        out[on] -= t[on] * ((out[on] - p[on]) * t[on]).sum(axis=1, keepdims=True)
    corner = level.pinned[ids] == 2
    out[corner] = p[corner]
    return _nearest_grid(out, q, t, s, p)


def _position_sweep(level: _Level, rng: np.random.Generator, robust: bool = False) -> None:
    o, q, n, p = level.o, level.q, level.normals, level.points
    colors = rng.integers(0, 3, len(o))
    fast = _native("field_position_sweep")
    if fast is not None:  # the same sums in the same order, on all cores
        out, misfit = _flat(o).copy(), np.zeros(0) if level.misfit is None else _flat(level.misfit).copy()
        fast(_flat(p), _flat(n), _flat(q), _flat(level.sizes), _flat(level.pinned, np.int64), out, misfit, _flat(level.a, np.int64), _flat(level.b, np.int64), _flat(colors, np.int64), bool(robust), float(SETTLED), float(ROBUST), bool(CONFIDENCE))
        o[:] = out.reshape(-1, 3)
        if level.misfit is not None:
            level.misfit[:] = misfit
        return
    head = np.concatenate((level.a, level.b))
    tail = np.concatenate((level.b, level.a))
    settled = None
    if robust and level.misfit is not None:  # where a grid agrees with all around it, a sweep has nothing to do
        unsettled = level.misfit > SETTLED
        unsettled[tail[unsettled[head]]] = True
        settled = ~unsettled[head]
    for color in range(3):
        use = colors[head] == color
        if settled is not None:
            use &= ~settled
        i, j = head[use], tail[use]
        if not len(i):
            continue
        middle = 0.5 * (p[i] + p[j])
        mine = _nearest_grid(o[i], q[i], np.cross(n[i], q[i]), level.sizes[i][:, None], middle)
        theirs = _nearest_grid(o[j], q[j], np.cross(n[j], q[j]), level.sizes[j][:, None], mine)
        gap = theirs - mine
        relative = np.linalg.norm(gap, axis=1) / np.minimum(level.sizes[i], level.sizes[j])
        trust = 1.0 / (ROBUST + relative) if robust else np.ones(len(i))
        if robust and CONFIDENCE:  # listen most to neighbours whose own grid agrees with everyone around them
            heard = np.bincount(i, minlength=len(o))
            level.misfit = np.where(heard > 0, np.bincount(i, relative, len(o)) / np.maximum(heard, 1), level.misfit)
            trust = trust / (ROBUST + level.misfit[j])
        total = np.stack([np.bincount(i, gap[:, k] * trust, len(o)) for k in range(3)], axis=1)
        uses = np.bincount(i, trust, len(o))
        ids = np.flatnonzero((colors == color) & (uses > 0))
        o[ids] = _place(level, o[ids] + total[ids] / uses[ids][:, None], ids)


def solve_orientation(levels: list[_Level], rng: np.random.Generator, sweeps: int | None = None) -> None:
    sweeps = int(ORIENT_SWEEPS if sweeps is None else sweeps)
    coarsest = levels[-1]
    start = rng.normal(size=coarsest.points.shape)
    coarsest.q = _unit(_tangent(start, coarsest.normals))
    for index in range(len(levels) - 1, -1, -1):
        level = levels[index]
        if index < len(levels) - 1:
            level.q = _unit(_tangent(levels[index + 1].q[level.parent], level.normals))
        on = level.pinned == 1
        level.q[on] = _unit(_tangent(level.along[on], level.normals[on]))
        for _ in range(sweeps if index else max(2, sweeps // 2)):
            _orient_sweep(level, rng)


def solve_positions(levels: list[_Level], rng: np.random.Generator, sweeps: int | None = None, stop=None) -> None:
    sweeps = int(POSITION_SWEEPS if sweeps is None else sweeps)
    coarsest = levels[-1]
    coarsest.o = _place(coarsest, coarsest.points.copy())
    for index in range(len(levels) - 1, -1, -1):
        level = levels[index]
        if index < len(levels) - 1:
            level.o = _place(level, levels[index + 1].o[level.parent])
        # Most of the work is on the second-finest level, at half the cost; the finest only takes up the rounding.
        for _ in range(max(2, sweeps // 2) if index == 0 else sweeps):
            _position_sweep(level, rng)
        level.misfit = np.ones(len(level.o))
        for _ in range(int(FINEST_SWEEPS if index == 0 else FINAL_SWEEPS if index == 1 else ROBUST_SWEEPS)):
            if stop is not None:
                stop()
            _position_sweep(level, rng, True)


def set_sizes(levels: list[_Level], sizes: np.ndarray) -> None:
    """Change the cell sizes on an existing hierarchy, for another solve of the positions."""

    levels[0].sizes = sizes
    for finer, coarser in zip(levels[:-1], levels[1:]):
        count = len(coarser.points)
        coarser.sizes = np.bincount(finer.parent, finer.sizes, count) / np.bincount(finer.parent, minlength=count)


def _best_match(qi, ni, qj, nj):
    """(direction, normal-turned direction) of the cross at j, chosen to line up with the cross at i."""

    ti, tj = np.cross(ni, qi), np.cross(nj, qj)
    dots = np.stack(((qi * qj).sum(1), (qi * tj).sum(1), (ti * qj).sum(1), (ti * tj).sum(1)), axis=1)
    best = np.argmax(np.abs(dots), axis=1)
    sign = np.sign(dots[np.arange(len(qi)), best])[:, None]
    other = np.where((best % 2 == 0)[:, None], qj, tj) * sign
    q = np.where((best < 2)[:, None], other, np.cross(other, ni))
    return q, np.cross(nj, q)


def conformal_scale(points: np.ndarray, tris: np.ndarray, q: np.ndarray, normals: np.ndarray, rounds: int = 200) -> np.ndarray:
    """Log of the cell size that lets a grid follow the field without row ends: the grid's edges need no curl.

    With du = exp(-phi) q and dv = exp(-phi) t closed on each triangle, the gradient of phi is known there from the
    curl of q and of t; this finds the phi that fits those gradients best. Triangles around a field singularity say nothing.
    """

    n = len(points)
    x, y, z = tris.T
    face_n, area, _ = _normals_areas(points, tris)
    qy, ty = _best_match(q[x], normals[x], q[y], normals[y])
    qz, tz = _best_match(q[x], normals[x], q[z], normals[z])
    qx, tx = q[x], np.cross(normals[x], q[x])
    px, py, pz = points[x], points[y], points[z]

    def curl(ax, ay, az):
        return 0.5 * (((ax + ay) * (py - px)).sum(1) + ((ay + az) * (pz - py)).sum(1) + ((az + ax) * (px - pz)).sum(1))

    # Same cross around the triangle? y-to-z must agree with what x gave them.
    agree = (qy * qz).sum(1) + (ty * tz).sum(1) > 1.0
    safe = np.maximum(area, 1e-30)
    gradient = (curl(tx, ty, tz) / safe)[:, None] * _unit(_tangent(qx, face_n)) - (curl(qx, qy, qz) / safe)[:, None] * _unit(_tangent(tx, face_n))
    gradient[~agree | (area <= 0.0)] = 0.0
    limit = 1.0 / np.maximum(np.sqrt(safe), 1e-30)
    gradient *= np.minimum(1.0, limit / np.maximum(np.linalg.norm(gradient, axis=1), 1e-30))[:, None]
    # Cotangent Laplacian: stiffness = cot of the opposite corner / 2 on each side.
    corners = points[tris]
    weights_a, weights_b, weights_w, rhs = [], [], [], np.zeros(n)
    for k in range(3):
        i, j, o = tris[:, k], tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
        u, v = corners[:, k] - corners[:, (k + 2) % 3], corners[:, (k + 1) % 3] - corners[:, (k + 2) % 3]
        cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-30)
        weights_a.append(i)
        weights_b.append(j)
        weights_w.append(0.5 * np.clip(cot, 0.0, 20.0))
        # grad of the corner opposite side (i, j): n x (j - i) / (2A); integrated over the triangle gives A * g . grad
        grad = np.cross(face_n, corners[:, (k + 1) % 3] - corners[:, k]) * 0.5
        rhs += np.bincount(o, (gradient * grad).sum(1), n)
    ea, eb, ew = np.concatenate(weights_a), np.concatenate(weights_b), np.concatenate(weights_w)
    diagonal = np.bincount(ea, ew, n) + np.bincount(eb, ew, n) + 1e-9

    def stiffness(v):
        d = ew * (v[ea] - v[eb])
        return np.bincount(ea, d, n) - np.bincount(eb, d, n)

    rhs -= rhs.mean()
    phi = np.zeros(n)
    residual = rhs.copy()
    z_ = residual / diagonal
    direction = z_.copy()
    rz = residual @ z_
    for _ in range(rounds):
        if rz <= 1e-12 * max((rhs @ (rhs / diagonal)), 1e-30):
            break
        applied = stiffness(direction)
        step = rz / max(direction @ applied, 1e-30)
        phi += step * direction
        residual -= step * applied
        z_ = residual / diagonal
        next_rz = residual @ z_
        direction = z_ + (next_rz / rz) * direction
        rz = next_rz
    return phi - phi.mean()


def regions(points: np.ndarray, tris: np.ndarray, level: _Level) -> np.ndarray:
    """Per point of the split copy, its region: connected points whose nearest grid corner is the same."""

    a, b = level.a, level.b
    same = np.linalg.norm(level.o[a] - level.o[b], axis=1) < SAME_CORNER * np.minimum(level.sizes[a], level.sizes[b])
    return np.unique(_labels(len(points), a[same], b[same]), return_inverse=True)[1]


def _split_rings(points: np.ndarray, tris: np.ndarray, labels: np.ndarray, a, b, cells: np.ndarray) -> np.ndarray:
    """Cut every region that isn't a disk in two across its long side, so it can become one vertex."""

    count = int(labels.max()) + 1
    inner = labels[a] == labels[b]
    whole = (labels[tris[:, 0]] == labels[tris[:, 1]]) & (labels[tris[:, 1]] == labels[tris[:, 2]])
    euler = np.bincount(labels, minlength=count) - np.bincount(labels[a[inner]], minlength=count) + np.bincount(labels[tris[whole, 0]], minlength=count)
    sizes = np.bincount(labels, minlength=count)
    bad = (euler != 1) & (sizes >= 3)
    # A grid can slide along a closed line and call the whole line one corner. No region reaches this far from its middle.
    middle = np.stack([np.bincount(labels, points[:, k], count) for k in range(3)], axis=1) / np.maximum(sizes, 1)[:, None]
    far = np.zeros(count)
    np.maximum.at(far, labels, np.linalg.norm(points - middle[labels], axis=1) / cells)
    bad |= (far > MOST_REACH) & (sizes >= 3)
    dual = dual_triangles(tris, labels)
    if len(dual):  # two regions may share one stretch of border only, or three faces would meet on an edge
        x, y, _e, uses = _edges(dual, count)
        crowded = uses > 2
        bad[x[crowded]] |= sizes[x[crowded]] >= 3
        bad[y[crowded]] |= sizes[y[crowded]] >= 3
    bad = np.flatnonzero(bad)
    if not len(bad):
        return labels
    side = np.zeros(len(points), dtype=np.int64)
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(labels[order][1:] != labels[order][:-1]) + 1)
    for label in bad.tolist():
        ids = groups[label]
        here = points[ids] - points[ids].mean(axis=0)
        axis = np.linalg.eigh(here.T @ here)[1][:, 2]
        side[ids] = (here @ axis) > 0.0
    keep = (labels[a] == labels[b]) & (side[a] == side[b])
    return np.unique(_labels(len(points), a[keep], b[keep]), return_inverse=True)[1]


def _merge_small(tris: np.ndarray, labels: np.ndarray, a, b) -> tuple[np.ndarray, bool]:
    """Regions in fewer than three output triangles can't be vertices: each goes to the neighbour it shares most with."""

    count = int(labels.max()) + 1
    t = labels[tris]
    three = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 2] != t[:, 0])
    junctions = np.bincount(t[three].reshape(-1), minlength=count)
    small = junctions < 3
    if not small.any():
        return labels, False
    across = labels[a] != labels[b]
    la, lb = labels[a[across]], labels[b[across]]
    pairs = np.concatenate((la * count + lb, lb * count + la))
    keys, uses = np.unique(pairs, return_counts=True)
    here, there = keys // count, keys % count
    # Largest shared border wins; never merge two small regions in the same round unless both agree it's the only way.
    order = np.argsort(uses, kind="stable")
    target = np.arange(count)
    target[here[order]] = there[order]
    target[~small] = np.arange(count)[~small]
    # Chains and swaps: follow to the end, and let a swap settle on the lower label.
    for _ in range(8):
        target = target[target]
    return np.unique(target[labels], return_inverse=True)[1], True


def dual_triangles(tris: np.ndarray, labels: np.ndarray) -> np.ndarray:
    t = labels[tris]
    three = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 2] != t[:, 0])
    return t[three]


def _clean_regions(points: np.ndarray, tris: np.ndarray, labels: np.ndarray, cells: np.ndarray) -> np.ndarray:
    a, b, _edge_of, _uses = _edges(tris, len(points))
    for _ in range(12):
        before = int(labels.max())
        labels, merged = _merge_small(tris, labels, a, b)
        labels = _split_rings(points, tris, labels, a, b, cells)
        if not merged and int(labels.max()) == before:
            break
    return labels


def fit_regions(points: np.ndarray, tris: np.ndarray, labels: np.ndarray, start: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Each vertex to where it best fits the planes of its own region. Weak directions stay, so rows aren't pulled sideways."""

    count = len(start)
    normals, areas, _vertex = _normals_areas(points, tris)
    planes = np.zeros((count, 3, 3))
    pull = np.zeros((count, 3))
    middle = points[tris].mean(axis=1)
    weight = areas[:, None, None] * normals[:, :, None] * normals[:, None, :]
    for k in range(3):
        owner = labels[tris[:, k]]
        np.add.at(planes, owner, weight / 3.0)
        np.add.at(pull, owner, (areas / 3.0 * (normals * (middle - start[owner])).sum(axis=1))[:, None] * normals)
    values, vectors = np.linalg.eigh(planes)
    strong = values > FIT_STRONG * np.maximum(values[:, 2:], 1e-30)
    along = np.where(strong, np.einsum("nij,ni->nj", vectors, pull) / np.maximum(values, 1e-30), 0.0)
    step = np.einsum("nij,nj->ni", vectors, along)
    step *= np.minimum(1.0, MOST_FIT * sizes / np.maximum(np.linalg.norm(step, axis=1), 1e-30))[:, None]
    return start + step


@dataclass
class FieldMesh:
    points: np.ndarray
    tris: np.ndarray  # output triangles, into points
    fine_points: np.ndarray
    fine_tris: np.ndarray
    labels: np.ndarray  # per fine point, its output vertex
    sizes: np.ndarray  # per fine point
    creased: np.ndarray | None = None  # per fine point: on a long crease or a border
    report: dict = field(default_factory=dict)


class FieldMesher:
    """Holds the split copy and its direction field, so positions can be solved again for other cell sizes.

    `turn`, in radians, is how far the surface may turn across a cell, read per limb.
    """

    def __init__(self, vertices, tris, target: float, turn: float | None = None, seed: int = 0, cancelled=None, lines=None):
        self.cancelled = cancelled
        self.rng = np.random.default_rng(seed)
        self.source = np.asarray(vertices, dtype=np.float64)
        self.source_tris = np.asarray(tris, dtype=np.int64)
        _normals, areas, _vertex = _normals_areas(self.source, self.source_tris)
        area = np.bincount(self.source_tris.reshape(-1), np.repeat(areas / 3.0, 3), len(self.source))
        even = float(np.sqrt(max(areas.sum(), 1e-30) / max(target, 1.0)))
        limits = segment_limits(self.source, self.source_tris, even, SEGMENT_TURN if turn is None else turn)
        self.target = float(target)
        sizes = fit_sizes(limits, area, self.target)
        self.fine_points, self.fine_tris, self.fine_sizes, self.limits = split_long_edges(self.source, self.source_tris, sizes, carried=limits)
        _fine_normals, fine_areas, _ = _normals_areas(self.fine_points, self.fine_tris)
        self.fine_area = np.bincount(self.fine_tris.reshape(-1), np.repeat(fine_areas / 3.0, 3), len(self.fine_points))
        forced = None
        if lines is not None:  # `lines(points, tris)` gives (points moved onto the lines, direction along them, 1 on them)
            self.fine_points, along, pinned = lines(self.fine_points, self.fine_tris)
            forced = (along, pinned)
            on = self.fine_points[pinned > 0]
            if len(on):  # rows close together along a drawn line, opening out over a few cells
                near = np.full(len(self.fine_points), np.inf)
                for at in range(0, len(self.fine_points), 4096):
                    gap = self.fine_points[at : at + 4096, None, :] - on[None]
                    near[at : at + 4096] = np.sqrt((gap * gap).sum(axis=2).min(axis=1))
                self.limits = np.minimum(self.limits, even * (LINE_CELL + near / (LINE_REACH * even)))
        self.levels = _hierarchy(self.fine_points, self.fine_tris, self.fine_sizes, self.rng, forced)
        solve_orientation(self.levels, self.rng)
        self.shape = None
        if CONFORMAL:
            phi = conformal_scale(self.fine_points, self.fine_tris, self.levels[0].q, self.levels[0].normals)
            # Each limb averages its own size: on a long shape phi alone gives one end far bigger cells than the other.
            _groups, group = np.unique(np.round(np.log(np.maximum(self.limits, 1e-30)), 3), return_inverse=True)
            phi -= (np.bincount(group, phi * self.fine_area) / np.maximum(np.bincount(group, self.fine_area), 1e-30))[group]
            self.shape = np.exp(np.clip(CONFORMAL * phi, -MOST_LOG, MOST_LOG))

    def solve(self, target: float | None = None) -> FieldMesh:
        """Lay the grid, once: finding the exact count by solving again cost half the time for a few percent of faces."""

        self._stop_check()
        wanted = self.target if target is None else target
        self.fine_sizes = fit_sizes(self.limits, self.fine_area, wanted / COUNT_SHARE, self.shape)
        set_sizes(self.levels, self.fine_sizes)
        solve_positions(self.levels, self.rng, stop=self._stop_check)
        return self.extract()

    def _stop_check(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise ValueError("Stopped.")

    def extract(self) -> FieldMesh:
        finest = self.levels[0]
        labels = _clean_regions(self.fine_points, self.fine_tris, regions(self.fine_points, self.fine_tris, finest), finest.sizes)
        count = int(labels.max()) + 1
        weight = self.fine_area
        total = np.maximum(np.bincount(labels, weight, count), 1e-30)
        start = np.stack([np.bincount(labels, finest.o[:, k] * weight, count) for k in range(3)], axis=1) / total[:, None]
        cell = np.bincount(labels, finest.sizes * weight, count) / total
        points = fit_regions(self.fine_points, self.fine_tris, labels, start, cell)
        tris = dual_triangles(self.fine_tris, labels)
        return FieldMesh(points, tris, self.fine_points, self.fine_tris, labels, finest.sizes, finest.pinned > 0, {"regions": count, "triangles": len(tris), "fine_points": len(self.fine_points), "levels": len(self.levels)})




def pair_triangles(points: np.ndarray, tris: np.ndarray) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Greedy pairing across the edge that makes the squarest quad: on a grid that's a cell's diagonal. Returns (quads, unpaired)."""

    a, b, edge_of, uses = _edges(tris, len(points))
    order = np.argsort(edge_of.reshape(-1), kind="stable")
    ids = edge_of.reshape(-1)[order]
    same = np.flatnonzero((ids[:-1] == ids[1:]) & (uses[ids[:-1]] == 2))
    side_x, side_y = order[same], order[same + 1]
    tx, ty = side_x // 3, side_y // 3
    kx, ky = side_x % 3, side_y % 3
    # quad corners: x's far corner, the edge's start, y's far corner, the edge's end
    p0 = tris[tx, (kx + 2) % 3]
    p1 = tris[tx, kx]
    p2 = tris[ty, (ky + 2) % 3]
    p3 = tris[tx, (kx + 1) % 3]
    quad = np.stack((p0, p1, p2, p3), axis=1)
    fair = tris[ty, ky] == p3  # opposite winding, so the pair shares the edge properly
    corners = points[quad]
    out = _unit(np.roll(corners, -1, axis=1) - corners)
    back = _unit(np.roll(corners, 1, axis=1) - corners)
    cosine = (out * back).sum(axis=2)
    error = np.abs(cosine).max(axis=1)  # 0 for right angles
    order = np.argsort(error, kind="stable")
    used = np.zeros(len(tris), dtype=bool)
    chosen = []
    for k in order[fair[order]].tolist():
        if error[k] >= SQUARE_ENOUGH:
            break
        if not used[tx[k]] and not used[ty[k]]:
            used[tx[k]] = used[ty[k]] = True
            chosen.append(k)
    return [tuple(row) for row in quad[chosen].tolist()], np.flatnonzero(~used)


def _corner_error(points: np.ndarray, quad) -> float:
    corners = points[list(quad)]
    out = _unit(np.roll(corners, -1, axis=0) - corners)
    back = _unit(np.roll(corners, 1, axis=0) - corners)
    return float(np.abs((out * back).sum(axis=1)).max())


def _walk_chord(faces, edge_faces, start: int, k: int, most: int):
    """From triangle `start`, across its side k, straight through quads to the next triangle. (faces, apex, chains, end) or None."""

    tri = faces[start]
    p, q, apex = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
    left, right, seen = [p], [q], [start]
    while len(seen) <= most:
        users = edge_faces.get((min(p, q), max(p, q)), ())
        if len(users) != 2:
            return None
        here = users[0] if users[1] == seen[-1] else users[1]
        if here in seen:
            return None
        face = faces[here]
        at = face.index(q)
        if face[(at + 1) % len(face)] != p:
            return None  # winding flips here
        seen.append(here)
        if len(face) == 3:
            return seen, apex, left, right, face[(at + 2) % 3]
        if len(face) != 4:
            return None
        p, q = face[(at + 2) % 4], face[(at + 3) % 4]
        left.append(p)
        right.append(q)
    return None


def _chord_options(apex, left, right, end):
    n = len(left) - 1
    a = [(apex, left[0], right[1] if n else end, right[0])]
    b = [(apex, left[0], left[1] if n else end, right[0])]
    for i in range(1, n + 1):
        a.append((left[i - 1], left[i], right[i + 1] if i < n else end, right[i]))
        b.append((right[i - 1], left[i], left[i + 1] if i < n else end, right[i]))
    return a, b


def pair_along_rows(points: np.ndarray, faces: list[tuple[int, ...]], most: int = 48, worst: float = 0.97) -> list[tuple[int, ...]]:
    """Two triangles at the ends of one row of quads become quads by sliding that row one step.

    Inside the row every vertex keeps its count of edges, and the two it costs one each are usually the poles beside
    the triangles. So row ends cancel in pairs without new poles, and relaxing afterwards spreads the shear along the row.
    """

    faces = [tuple(f) for f in faces]
    for _ in range(8):
        edge_faces: dict[tuple[int, int], list[int]] = {}
        neighbours: dict[int, set[int]] = {}
        for index, face in enumerate(faces):
            for k in range(len(face)):
                x, y = face[k], face[(k + 1) % len(face)]
                edge_faces.setdefault((min(x, y), max(x, y)), []).append(index)
                neighbours.setdefault(x, set()).add(y)
                neighbours.setdefault(y, set()).add(x)
        chords = []
        for start in (i for i, f in enumerate(faces) if len(f) == 3):
            for k in range(3):
                found = _walk_chord(faces, edge_faces, start, k, most)
                if found is not None and start < found[0][-1]:
                    chords.append((len(found[0]), start, found))
        chords.sort(key=lambda c: (c[0], c[1]))
        taken: set[int] = set()
        added: list[tuple[int, ...]] = []
        for _length, _start, (ids, apex, left, right, end) in chords:
            if taken.intersection(ids):
                continue
            chain = [apex, end, *left, *right]
            if len(set(chain)) != len(chain):
                continue
            best, best_cost = None, None
            options = _chord_options(apex, left, right, end)
            for which, option in enumerate(options):
                # the new edges must not exist already
                rungs = [(q[2], q[3]) if which == 0 else (q[3], q[2]) for q in option[:-1]]
                if any((min(x, y), max(x, y)) in edge_faces for x, y in rungs):
                    continue
                error = max(_corner_error(points, q) for q in option)
                if error >= worst:
                    continue
                losers = (right[0], left[-1]) if which == 0 else (left[0], right[-1])
                cost = sum(abs(len(neighbours[v]) - 1 - 4) - abs(len(neighbours[v]) - 4) for v in losers) + error
                if best is None or cost < best_cost:
                    best, best_cost = option, cost
            if best is None:
                continue
            taken.update(ids)
            added += best
        if not added:
            break
        faces = [f for i, f in enumerate(faces) if i not in taken] + added
    return faces


def _face_area_normals(points, faces, count: int) -> np.ndarray:
    out = np.zeros((count, 3))
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    if len(quads):
        cross = np.cross(points[quads[:, 2]] - points[quads[:, 0]], points[quads[:, 3]] - points[quads[:, 1]])
        for k in range(3):
            out[:, k] = np.bincount(quads.reshape(-1), np.repeat(cross[:, k], 4), count)
    return _unit(out)


def onto_creases(points: np.ndarray, fine_points: np.ndarray, labels: np.ndarray, creased: np.ndarray) -> np.ndarray:
    """A vertex whose region holds part of a crease goes to the nearest crease point there, so the ridge is an edge of the mesh."""

    ids = np.flatnonzero(creased)
    if not len(ids):
        return points
    points = points.copy()
    away = np.linalg.norm(fine_points[ids] - points[labels[ids]], axis=1)
    order = np.argsort(-away)  # the last write, the nearest, wins
    points[labels[ids[order]]] = fine_points[ids[order]]
    return points


def relax(mesh: FieldMesh, points: np.ndarray, faces, rounds: int | None = None) -> np.ndarray:
    """Even the rows out along the surface, each round fitted back to each vertex's own region of the source."""

    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    flat = np.fromiter((v for f in faces for v in f), dtype=np.int64, count=int(sizes.sum()))
    ends = np.cumsum(sizes)
    following = np.arange(len(flat)) + 1
    following[ends - 1] = ends - sizes
    first = np.concatenate((flat, flat[following]))
    second = np.concatenate((flat[following], flat))
    points = points.copy()
    count = len(points)
    known = len(mesh.points)
    cell = np.bincount(mesh.labels, mesh.sizes, known) / np.maximum(np.bincount(mesh.labels, minlength=known), 1)
    uses = np.maximum(np.bincount(first, minlength=count), 1)
    for _ in range(int(RELAX_ROUNDS if rounds is None else rounds)):
        target = np.stack([np.bincount(first, points[second, k], count) for k in range(3)], axis=1) / uses[:, None]
        points += RELAX_RATE * _tangent(target - points, _face_area_normals(points, faces, count))
        points[:known] = fit_regions(mesh.fine_points, mesh.fine_tris, mesh.labels, points[:known], cell)
        points[:known] = onto_creases(points[:known], mesh.fine_points, mesh.labels, mesh.creased)
    return points


def untangle(points: np.ndarray, faces, rounds: int = 12) -> np.ndarray:
    """Move only the corners of folded or nearly folded quads toward the middle of their neighbours, along the surface."""

    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    points = points.copy()
    if not len(quads):
        return points
    count = len(points)
    first = np.concatenate([quads[:, k] for k in range(4)] + [quads[:, (k + 1) % 4] for k in range(4)])
    second = np.concatenate([quads[:, (k + 1) % 4] for k in range(4)] + [quads[:, k] for k in range(4)])
    uses = np.maximum(np.bincount(first, minlength=count), 1)
    for _ in range(rounds):
        bad = quads[quad_quality(points[quads]) < UNTANGLE_BELOW]
        if not len(bad):
            break
        ids = np.unique(bad)
        target = np.stack([np.bincount(first, points[second, k], count) for k in range(3)], axis=1) / uses[:, None]
        points[ids] += 0.7 * _tangent(target - points, _face_area_normals(points, faces, count))[ids]
    return points


def _edge_map(faces) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(faces):
        for k in range(len(face)):
            x, y = face[k], face[(k + 1) % len(face)]
            out.setdefault((min(x, y), max(x, y)), []).append(index)
    return out


def _join(first, second, x: int, y: int) -> tuple[int, int, int, int] | None:
    """Two triangles, `first` holding x -> y and `second` y -> x, as one quad."""

    a = next(v for v in first if v != x and v != y)
    b = next(v for v in second if v != x and v != y)
    return None if a == b else (x, b, y, a)


def _turned(face, x: int, y: int) -> tuple[int, ...]:
    """This face's corners turned to start on its side x -> y."""

    at = face.index(x)
    return tuple(face[(at + k) % len(face)] for k in range(len(face)))


def pair_by_paths(points: np.ndarray, faces: list[tuple[int, ...]], reach: int = 300) -> list[tuple[int, ...]]:
    """Pair each triangle left with the nearest other one along any path of quads, splitting and re-joining those quads."""

    faces = [tuple(f) for f in faces]
    alive = {i: f for i, f in enumerate(faces)}
    edges = _edge_map(faces)
    triangles = [i for i, f in alive.items() if len(f) == 3]
    fresh = len(faces)
    for start in triangles:
        if start not in alive:
            continue
        came = {start: None}
        queue = [start]
        end = None
        for here in queue:
            if len(queue) > reach or end is not None:
                break
            face = alive[here]
            for k in range(len(face)):
                x, y = face[k], face[(k + 1) % len(face)]
                users = edges.get((min(x, y), max(x, y)), ())
                if len(users) != 2:
                    continue
                other = users[0] if users[1] == here else users[1]
                if other in came or other not in alive or x not in alive[other] or y not in alive[other]:
                    continue
                turned = alive[other].index(y)
                if alive[other][(turned + 1) % len(alive[other])] != x:
                    continue  # winding flips here
                came[other] = (here, x, y)
                queue.append(other)
                if len(alive[other]) == 3:
                    path = [other]
                    while came[path[-1]] is not None:
                        path.append(came[path[-1]][0])
                    path.reverse()
                    built = _rejoin(points, alive, came, path)
                    if built is not None:
                        end = other
                        break
        if end is None:
            continue
        for index in path:
            face = alive.pop(index)
            for k in range(len(face)):
                x, y = face[k], face[(k + 1) % len(face)]
                edges[(min(x, y), max(x, y))].remove(index)
        for quad in built:
            alive[fresh] = quad
            for k in range(4):
                x, y = quad[k], quad[(k + 1) % 4]
                edges.setdefault((min(x, y), max(x, y)), []).append(fresh)
            fresh += 1
    return list(alive.values())


def _rejoin(points, alive, came, path) -> list[tuple[int, ...]] | None:
    built: list[tuple[int, ...]] = []
    current = alive[path[0]]
    for index in path[1:]:
        _before, x, y = came[index]  # x -> y in the face before, y -> x in this one
        face = alive[index]
        if len(face) == 3:
            quad = _join(current, face, x, y)
            if quad is None:
                return None
            built.append(quad)
            break
        c = _turned(face, y, x)  # c[0] = y, c[1] = x
        following = path[path.index(index) + 1]
        _here, out_x, out_y = came[following]
        at = c.index(out_x)  # the side leaving runs c[at] -> c[at + 1]
        options = []
        if at == 2:
            options = [((c[0], c[1], c[2]), (c[0], c[2], c[3])), ((c[0], c[1], c[3]), (c[1], c[2], c[3]))]
        elif at == 1:
            options = [((c[0], c[1], c[3]), (c[1], c[2], c[3]))]
        else:
            options = [((c[0], c[1], c[2]), (c[0], c[2], c[3]))]
        best = None
        for inner, outer in options:
            quad = _join(current, inner, x, y)
            if quad is None:
                continue
            error = _corner_error(points, quad)
            if best is None or error < best[0]:
                best = (error, quad, outer)
        if best is None:
            return None
        built.append(best[1])
        current = best[2]
    return built


def _dissolve_doublets(faces) -> tuple[list[tuple[int, ...]], int]:
    """A corner in only two faces adds nothing: those two become one."""

    around: dict[int, list[int]] = {}
    for index, face in enumerate(faces):
        for v in face:
            around.setdefault(v, []).append(index)
    used: set[int] = set()
    added: list[tuple[int, ...]] = []
    for v, ids in around.items():
        if len(ids) != 2 or used.intersection(ids):
            continue
        first, second = (_turned(faces[i], v, v) for i in ids)
        if first[1] != second[-1] or first[-1] != second[1]:
            continue  # not two faces sharing both edges at this corner
        merged = first[1:] + second[2:-1]
        if len(set(merged)) != len(merged) or len(merged) < 3:
            continue
        used.update(ids)
        added.append(merged)
    return [f for i, f in enumerate(faces) if i not in used] + added, len(added)


def _merge_folded(points: np.ndarray, faces) -> tuple[list[tuple[int, ...]], int]:
    """A folded quad goes by merging two opposite corners, which leaves every other face as it was."""

    quads = np.asarray([i for i, f in enumerate(faces) if len(f) == 4], dtype=np.int64)
    if not len(quads):
        return list(faces), 0
    corners = np.asarray([faces[i] for i in quads.tolist()], dtype=np.int64)
    bad = quads[quad_quality(points[corners]) <= 0.0]
    neighbours: dict[int, set[int]] = {}
    for face in faces:
        for k in range(len(face)):
            x, y = face[k], face[(k + 1) % len(face)]
            neighbours.setdefault(x, set()).add(y)
            neighbours.setdefault(y, set()).add(x)
    rename: dict[int, int] = {}
    touched: set[int] = set()
    gone: set[int] = set()
    for index in bad.tolist():
        a, b, c, d = faces[index]
        if touched.intersection((a, b, c, d)):
            continue
        for x, y, side in ((a, c, (b, d)), (b, d, (a, c))) if np.linalg.norm(points[a] - points[c]) <= np.linalg.norm(points[b] - points[d]) else ((b, d, (a, c)), (a, c, (b, d))):
            if (neighbours[x] & neighbours[y]) == set(side) and len(neighbours[side[0]]) > 2 and len(neighbours[side[1]]) > 2:
                rename[y] = x
                points[x] = 0.5 * (points[x] + points[y])
                touched.update((a, b, c, d))
                touched.update(neighbours[x] | neighbours[y])
                gone.add(index)
                break
    out = []
    for index, face in enumerate(faces):
        if index not in gone:
            out.append(tuple(rename.get(v, v) for v in face))
    return out, len(gone)


def clean(points: np.ndarray, faces) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    points = points.copy()
    for _ in range(6):
        faces, doublets = _dissolve_doublets(faces)
        faces, folded = _merge_folded(points, faces)
        if not doublets and not folded:
            break
    return points, faces


def quads_from(mesh: FieldMesh) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """All the steps from one solved field to quads."""

    quads, left = pair_triangles(mesh.points, mesh.tris)
    faces = quads + [tuple(row) for row in mesh.tris[left].tolist()]
    faces = pair_along_rows(mesh.points, faces)
    points = relax(mesh, mesh.points, faces)
    faces = pair_along_rows(points, faces, worst=WORST_CLOSING)  # relaxed, more rows pass the shape test
    faces = pair_by_paths(points, faces)
    points = relax(mesh, points, faces, 4)
    points, faces = clean(points, faces)
    for _ in range(FINISH_ROUNDS):  # cleaning can leave a triangle, and pairing a fold
        faces = pair_by_paths(points, faces, 4000)
        points = relax(mesh, points, faces, 2)
        points, faces = clean(points, faces)
        if all(len(f) == 4 for f in faces):
            break
    points = untangle(points, faces)
    points = refit(mesh.fine_points, mesh.fine_tris, points, faces, creased=mesh.creased)[0]
    for _ in range(FINISH_ROUNDS):
        points, faces = clean(points, faces)
        points = untangle(points, faces, 24)
        if not _any_folded(points, faces):
            break
    points = _even_out_folds(points, faces)
    for _ in range(FINISH_ROUNDS * 2):
        faces = pair_by_paths(points, faces, 4000)
        if all(len(f) == 4 for f in faces):
            break
        faces = _walk_triangles(points, faces)
    points = _even_out_folds(points, faces)
    points, faces = _open_crowded(points, faces)
    used, flat = np.unique(np.fromiter((v for f in faces for v in f), dtype=np.int64), return_inverse=True)
    out, at = [], 0
    for face in faces:
        out.append(tuple(flat[at : at + len(face)].tolist()))
        at += len(face)
    return points[used], out


def nearest_points(points: np.ndarray, asks: np.ndarray, cell: float, normals: np.ndarray, ask_normals: np.ndarray) -> np.ndarray:
    """For each ask, the nearest of `points` that faces its way, so a thin wall's far side is never picked: looks in the grid cells around it, or all if those are empty."""

    low = points.min(axis=0)
    size = max(float(cell), float(np.ptp(points, axis=0).max()) / 512.0, 1e-30)
    grid = np.floor((points - low) / size).astype(np.int64)
    span = int(grid.max()) + 3
    keys = ((grid[:, 0] + 1) * span + grid[:, 1] + 1) * span + grid[:, 2] + 1
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    around = np.array([(x * span + y) * span + z for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)], dtype=np.int64)
    cells = np.floor((asks - low) / size).astype(np.int64)
    centre = ((cells[:, 0] + 1) * span + cells[:, 1] + 1) * span + cells[:, 2] + 1
    fast = _native("field_nearest_points")
    if fast is not None and len(points):
        return fast(_flat(points), _flat(normals), _flat(sorted_keys, np.int64), _flat(order, np.int64), _flat(centre, np.int64), around, _flat(asks), _flat(ask_normals), float(FAR_SIDE))
    out = np.zeros(len(asks), dtype=np.int64)
    for index, (ask, key, facing) in enumerate(zip(asks, centre.tolist(), ask_normals)):
        wanted = key + around
        lefts, rights = np.searchsorted(sorted_keys, wanted), np.searchsorted(sorted_keys, wanted, side="right")
        found = np.concatenate([order[x:y] for x, y in zip(lefts.tolist(), rights.tolist()) if y > x] or [order])
        away = ((points[found] - ask) ** 2).sum(axis=1) * np.where(normals[found] @ facing < 0.0, FAR_SIDE, 1.0)
        out[index] = found[np.argmin(away)]
    return out


def graph_reach(count: int, a: np.ndarray, b: np.ndarray, length: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per graph vertex, (the index of the seed nearest along edges, how far that is)."""

    distance = np.full(count, np.inf)
    owner = np.full(count, -1, dtype=np.int64)
    owner[seeds] = np.arange(len(seeds))
    distance[seeds] = 0.0
    tail = np.concatenate((a, b))
    head = np.concatenate((b, a))
    span = np.concatenate((length, length))
    for _ in range(4096):
        offer = distance[tail] + span
        better = offer < distance[head]
        if not better.any():
            break
        ids = np.flatnonzero(better)
        ids = ids[np.argsort(-offer[ids], kind="stable")]  # the last write, the smallest offer, wins
        distance[head[ids]] = offer[ids]
        owner[head[ids]] = owner[tail[ids]]
    return owner, distance


def refit(fine_points: np.ndarray, fine_tris: np.ndarray, points: np.ndarray, faces, rounds: int = 3, creased: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Give each vertex the piece of the source nearest to it, then fit it to the planes there. Returns (points, region per fine point)."""

    a, b, _e, _u = _edges(fine_tris, len(fine_points))
    length = np.linalg.norm(fine_points[a] - fine_points[b], axis=1)
    side = float(np.median(np.linalg.norm(points[np.asarray([f[0] for f in faces])] - points[np.asarray([f[1] for f in faces])], axis=1)))
    labels = np.zeros(len(fine_points), dtype=np.int64)
    points = points.copy()
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    fine_normals = _normals_areas(fine_points, fine_tris)[2]
    for _ in range(rounds):
        seeds = nearest_points(fine_points, points, side, fine_normals, _face_area_normals(points, faces, len(points)))
        unique_seeds, first = np.unique(seeds, return_index=True)
        owner = graph_reach(len(fine_points), a, b, length, unique_seeds)[0]
        labels = first[owner]
        known = np.bincount(labels, minlength=len(points)) > 0
        cell = np.full(len(points), side)
        moved = fit_regions(fine_points, fine_tris, labels, points, cell)
        moved[~known] = points[~known]
        if creased is not None:
            moved = onto_creases(moved, fine_points, labels, creased)
        before = quad_quality(points[quads])
        for _try in range(8):  # any vertex on a quad the move made bad goes back
            bad = quads[quad_quality(moved[quads]) < np.minimum(before, GUARD_QUALITY)]
            if not len(bad):
                break
            ids = np.unique(bad)
            moved[ids] = points[ids]
        points = moved
    return points, labels




def field_quads(vertices, faces, target: float, turn: float | None = None, seed: int = 0, cancelled=None, lines=None) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Quads in rows on one loose piece, about `target` of them. May leave a few triangles or folds: see `is_clean`.

    Best on large, smooth or simply shaped pieces. Busy machined parts lose far more shape than the usual reduction and
    pairing, so a caller should judge the result and keep the better one.
    Raises ValueError with a message for a person on a piece too small for a grid or too large to finish in reasonable time.
    """

    if target < SMALLEST_TARGET:
        raise ValueError("This part is too small for rows of quads.")
    if target > MOST_TARGET:
        raise ValueError(f"Rows of quads stop at {MOST_TARGET:,} quads on one part, and this one asks for {int(target):,}.")
    tris = triangulate(faces)[0]
    if len(tris) > MOST_SOURCE:
        raise ValueError(f"Rows of quads stop at {MOST_SOURCE:,} triangles on one part, and this one has {len(tris):,}.")
    mesh = FieldMesher(vertices, tris, target, turn, seed, cancelled, lines).solve()
    points, quads = quads_from(mesh)
    if len(quads) < SMALLEST_TARGET // 2:
        raise ValueError("Rows of quads fell apart on this part.")
    return points, quads


def _open_crowded(points: np.ndarray, faces) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """An edge with three faces, which merges can rarely leave: those faces come out, and quads go over the hole."""

    from . import segment  # it uses this file for Animation rows

    if not faces or not segment._crowded_edges(faces):
        return points, faces
    before = {v for loop in segment.boundary_loops(faces, range(len(faces)))[0] for v in loop}
    kept = segment._drop_crowded(faces)
    holes = [loop for loop in segment.boundary_loops(kept, range(len(kept)))[0] if not before.intersection(loop)]
    return segment._fill_holes(points, kept, holes)


def _walk_triangles(points: np.ndarray, faces) -> list[tuple[int, ...]]:
    """A triangle no path reaches moves over: its shortest side closes up, and the quad beyond it becomes the triangle."""

    faces = [tuple(f) for f in faces]
    edges = _edge_map(faces)
    neighbours: dict[int, set[int]] = {}
    for x, y in edges:
        neighbours.setdefault(x, set()).add(y)
        neighbours.setdefault(y, set()).add(x)
    gone: set[int] = set()
    rename: dict[int, int] = {}
    touched: set[int] = set()
    for index, face in enumerate(faces):
        if len(face) != 3 or touched.intersection(face):
            continue
        sides = sorted(((float(np.linalg.norm(points[face[k]] - points[face[(k + 1) % 3]])), face[k], face[(k + 1) % 3]) for k in range(3)))
        for _length, x, y in sides:
            users = edges[(min(x, y), max(x, y))]
            if len(users) != 2:
                continue
            other = faces[users[0] if users[1] == index else users[1]]
            common = neighbours[x] & neighbours[y]
            if common == {v for v in (*face, *other) if v not in (x, y) and (v in neighbours[x] and v in neighbours[y])} and len(common) <= 2 and len(other) in (3, 4):
                rename[y] = x
                gone.add(index)
                touched.update(neighbours[x] | neighbours[y] | {x, y})
                break
    out = []
    for index, face in enumerate(faces):
        if index in gone:
            continue
        renamed = tuple(rename.get(v, v) for v in face)
        unique = tuple(dict.fromkeys(renamed))
        if len(unique) >= 3:
            out.append(unique)
    return out


def _even_out_folds(points: np.ndarray, faces, rounds: int = 40) -> np.ndarray:
    """Last resort for what's still folded: each corner to the middle of its neighbours, in all directions, spreading outward until none is. Costs a little shape there."""

    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    if not len(quads):
        return points
    points = points.copy()
    count = len(points)
    first = np.concatenate([quads[:, k] for k in range(4)] + [quads[:, (k + 1) % 4] for k in range(4)])
    second = np.concatenate([quads[:, (k + 1) % 4] for k in range(4)] + [quads[:, k] for k in range(4)])
    uses = np.maximum(np.bincount(first, minlength=count), 1)
    reach = None
    for _ in range(rounds):
        bad = quad_quality(points[quads]) <= 0.0
        if not bad.any():
            break
        ids = np.unique(quads[bad])
        if reach is None:
            reach = np.zeros(count, dtype=bool)
        reach[ids] = True
        moving = np.flatnonzero(reach)
        target = np.stack([np.bincount(first, points[second, k], count) for k in range(3)], axis=1) / uses[:, None]
        points[moving] += 0.5 * (target[moving] - points[moving])
    return points


def _any_folded(points: np.ndarray, faces) -> bool:
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    return bool((quad_quality(points[quads]) <= 0.0).any())


def is_clean(points: np.ndarray, faces) -> bool:
    """All quads, none folded, and no edge with more than two faces: what a Quads result promises."""

    if not faces or any(len(f) != 4 for f in faces):
        return False
    quads = np.asarray(faces, dtype=np.int64)
    if (quad_quality(points[quads]) <= 0.0).any():
        return False
    return not (_edge_counts(quads, len(points)) > 2).any()


def _edge_counts(quads: np.ndarray, count: int) -> np.ndarray:
    first, second = quads.reshape(-1), np.roll(quads, -1, axis=1).reshape(-1)
    return np.unique(np.minimum(first, second) * count + np.maximum(first, second), return_counts=True)[1]
