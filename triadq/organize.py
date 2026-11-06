# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Organize: a tidier layout for a finished quad result, with the same faces, trading a little detail for it.

Two steps, both on the result as it is (any style, any size):

- `clean_poles`: the patch redrawing and edge spinning of `chords.ChordReducer` (Tarini et al. 2010's edge rotation),
  judged by the camera when a judge is given. Every new face keeps the key (material, part) of the patch it replaced.
- `settle`: the quads are pulled into shape like a damped cloth held on the surface. Every quad pulls its corners toward
  the nearest rectangle of its own size (shape matching, Mueller et al. 2005), every regular vertex pulls toward the
  straight line through its two opposite neighbours (so loops run straight), and a pull toward the nearest point of
  your mesh holds the result on it. The pulls are averaged per vertex and applied a few dozen times with momentum and
  damping (position-based dynamics, Mueller et al. 2007). Hard edges, borders and corners of triangles stay pinned;
  vertices on a mirror plane slide in the plane only.

`method="straighten"` is the last step of the slider: each corner with four quads round it moves onto the fair curve
through the loops that cross there (wiggles out, steady bends kept) and to the middle of its neighbours along them
(even widths, evenly stacked loops), while poles keep the plain pull. Same guards.

`method="smooth"` swaps the rectangle and loop pulls for the plain one (each vertex toward its neighbours' middle) in the
same damped, guarded loop. On five test models the two changed the view the same within 0.2 points, and the plain pull
left a few fewer slivers, so the Organize slider uses it; the rectangle pull is kept for the benchmark. Plain NumPy.
"""

from __future__ import annotations

import numpy as np

from .chords import ChordReducer
from .fewerpoles import PoleCleaner
from .types import MeshData

SHARP_DEG = 35.0  # a vertex on an edge folding more than this is a hard edge and stays where it is
SETTLE_STEPS = 40  # steps at full strength
STIFF = 0.35  # share of the averaged pull applied per step
MOMENTUM = 0.45  # share of the last step carried on (damped)
LINE_WEIGHT = 0.6  # the straight-loop pull against the rectangle pull
ATTRACT = 0.7  # share of the way back to the surface per step; the snapshots go all the way
SQUARE_PULL = 1.0  # how far a long quad's sides are pulled toward equal, at full strength
WORST_TURN = 0.5  # a quad whose normal turned past this cosine since the start puts its corners back
BEND_DEG = 30.0  # a vertex whose quads turn this far from its normal holds detail and doesn't move; flatter ones move more
FOLD_COS = float(np.cos(np.radians(40.0)))  # a quad folding more than this along a diagonal is bent
MOST_MOVE = 0.25  # of a vertex's mean edge length, per step
MOST_DRIFT = 0.5  # of a vertex's mean edge length, in all: settling evens quads out where they are, it doesn't move them away
STRAIGHT_STEPS = 40  # steps of the loop straightening that follows the settle (its own drift budget, as above)
CYCLES = 3  # rounds of pole cleaning in `fewer_poles` that always run; more follow while a round still takes poles away
CYCLE_STEPS = 20  # settle steps between two rounds
MIN_GAIN = 0.5  # points of Tidiness or Loop Quality a pass must add for Keep Tidying to go on (or one pole fewer)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def _quads(faces) -> tuple[np.ndarray, np.ndarray]:
    ids = np.fromiter((i for i, f in enumerate(faces) if len(f) == 4), dtype=np.int64)
    return np.asarray([faces[i] for i in ids], dtype=np.int64).reshape(-1, 4), ids


def _edges(faces):
    """{(low, high): [faces]} of every edge."""
    out: dict[tuple[int, int], list[int]] = {}
    for i, f in enumerate(faces):
        for k in range(len(f)):
            a, b = f[k - 1], f[k]
            out.setdefault((a, b) if a < b else (b, a), []).append(i)
    return out


def _normals(co: np.ndarray, quads: np.ndarray) -> np.ndarray:
    p = co[quads]
    return _unit(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]))


def pinned(co, faces, sharp_deg: float = SHARP_DEG, plane=None) -> np.ndarray:
    """Vertices that must not move: on a border (unless on the mirror plane), on a hard edge, or on a face that isn't a quad."""
    co = np.asarray(co, dtype=np.float64)
    out = np.zeros(len(co), dtype=bool)
    for f in faces:
        if len(f) != 4:
            out[list(f)] = True
    normal = {}
    for i, f in enumerate(faces):
        p = co[list(f)]
        n = np.cross(p[2] - p[0], p[-1] - p[1]) if len(f) >= 4 else np.cross(p[1] - p[0], p[2] - p[0])
        normal[i] = n / max(float(np.linalg.norm(n)), 1e-30)
    on_plane = np.zeros(len(co), dtype=bool) if plane is None else on_mirror(co, *plane)
    limit = np.cos(np.radians(sharp_deg))
    for (a, b), users in _edges(faces).items():
        if len(users) != 2:
            if not (on_plane[a] and on_plane[b]):
                out[a] = out[b] = True
        elif float(normal[users[0]] @ normal[users[1]]) < limit:
            out[a] = out[b] = True
    return out


def on_mirror(co, axis: int, offset: float) -> np.ndarray:
    co = np.asarray(co, dtype=np.float64)
    size = float(np.linalg.norm(np.ptp(co, axis=0))) or 1.0
    return np.abs(co[:, axis] - offset) <= 1e-6 * size


def _line_pairs(faces, count: int, stay: np.ndarray) -> np.ndarray:
    """(vertex, one neighbour, the opposite neighbour) for every loop through a free vertex with four quads around it."""
    around: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    for f in faces:
        if len(f) != 4:
            continue
        for k in range(4):
            around[f[k]].append((f[k - 1], f[(k + 1) % 4]))
    rows = []
    for v, sides in enumerate(around):
        if stay[v] or len(sides) != 4:
            continue
        near = {u for pair in sides for u in pair}
        if len(near) != 4:
            continue
        beside = {frozenset(pair) for pair in sides}
        rest = sorted(near)
        for u in rest:
            opposite = [w for w in rest if w != u and frozenset((u, w)) not in beside]
            if len(opposite) == 1 and u < opposite[0]:
                rows.append((v, u, opposite[0]))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def _vertex_normals(co: np.ndarray, quads: np.ndarray) -> np.ndarray:
    p = co[quads]
    n = np.zeros_like(co)
    np.add.at(n, quads.reshape(-1), np.repeat(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]), 4, axis=0))
    return _unit(n)


def _bent(co: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Quads that are concave or fold along a diagonal past FOLD_COS."""
    p = co[quads]
    n = _unit(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]))
    corner = np.cross(np.roll(p, -1, axis=1) - p, np.roll(p, 1, axis=1) - p)
    concave = ((corner * n[:, None, :]).sum(axis=2) <= 0.0).any(axis=1)
    out = concave
    for a, b, c, d in ((0, 1, 2, 3), (1, 2, 3, 0)):
        one = _unit(np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]))
        two = _unit(np.cross(p[:, c] - p[:, a], p[:, d] - p[:, a]))
        out = out | ((one * two).sum(axis=1) < FOLD_COS)
    return out


def _rectangle_pull(co: np.ndarray, quads: np.ndarray, square: float) -> tuple[np.ndarray, np.ndarray]:
    """Per quad corner: the move toward the best-fitting rectangle of the quad's own size, in its plane."""
    p = co[quads]
    centre = p.mean(axis=1)
    n = _unit(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]))
    along = 0.5 * (p[:, 1] + p[:, 2] - p[:, 0] - p[:, 3])
    t1 = _unit(along - (along * n).sum(axis=1, keepdims=True) * n)
    t2 = np.cross(n, t1)
    rel = p - centre[:, None, :]
    z = (rel * t1[:, None, :]).sum(axis=2) + 1j * (rel * t2[:, None, :]).sum(axis=2)
    side = np.linalg.norm(np.roll(p, -1, axis=1) - p, axis=2)
    a, b = 0.5 * (side[:, 0] + side[:, 2]), 0.5 * (side[:, 1] + side[:, 3])
    mean = np.sqrt(np.maximum(a * b, 1e-30))
    a, b = mean * (a / mean) ** (1.0 - square), mean * (b / mean) ** (1.0 - square)
    w = 0.5 * np.stack((-a - 1j * b, a - 1j * b, a + 1j * b, -a + 1j * b), axis=1)
    turn = (z * np.conj(w)).sum(axis=1)  # the best rotation of the rectangle onto the corners (2D Procrustes)
    turn = turn / np.maximum(np.abs(turn), 1e-30)
    goal = turn[:, None] * w
    move = (goal.real - z.real)[:, :, None] * t1[:, None, :] + (goal.imag - z.imag)[:, :, None] * t2[:, None, :]
    return move.reshape(-1, 3), quads.reshape(-1)


def _onto(project, points: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """The nearest points of the surface, except where that is further than a step could have taken a vertex off it:
    there the nearest point is another part or the other side of a thin one, and the vertex stays."""
    near = project(points)
    far = np.linalg.norm(near - points, axis=1) > 2.0 * reach
    near[far] = points[far]
    return near


def _line_pull(co: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Per row of `lines`: the move of the vertex onto the straight line through its two opposite neighbours."""
    v, a, c = co[lines[:, 0]], co[lines[:, 1]], co[lines[:, 2]]
    d = _unit(c - a)
    return a + ((v - a) * d).sum(axis=1, keepdims=True) * d - v


def _neighbour_pull(co: np.ndarray, faces) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray([f[k] for f in faces for k in range(len(f))], dtype=np.int64)
    b = np.asarray([f[(k + 1) % len(f)] for f in faces for k in range(len(f))], dtype=np.int64)
    return np.concatenate((co[b] - co[a], co[a] - co[b])), np.concatenate((a, b))


def _fair_rows(faces, count: int, stay: np.ndarray) -> np.ndarray:
    """(vertex, one neighbour, opposite neighbour, the next one beyond each or -1) for every loop through a free vertex
    with four quads round it: the five points of a loop that `_fair_pull` reads."""
    lines = _line_pairs(faces, count, stay)
    if not len(lines):
        return np.zeros((0, 5), dtype=np.int64)
    from .loopscore import _continuations
    from .patterns import Topology

    go = _continuations(Topology(np.zeros((count, 3)), faces))
    v, u, w = lines[:, 0].tolist(), lines[:, 1].tolist(), lines[:, 2].tolist()
    uu = [go.get(a * count + b, -1) for a, b in zip(v, u)]
    ww = [go.get(a * count + b, -1) for a, b in zip(v, w)]
    return np.column_stack((lines, np.asarray(uu, dtype=np.int64), np.asarray(ww, dtype=np.int64)))


def _fair_pull(co: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per row of `_fair_rows`: across the loop, the move onto the fair curve through its five points (the point where
    the fourth difference along the loop is zero: wiggles go, a steady bend such as a forehead's arch stays); along the
    loop, the move to the middle of the two neighbours (even quad widths along it, so even gaps between the loops
    stacked across it). Loop fairing in the spirit of discrete fairing (Kobbelt 1997), on our own loop rows."""
    v, u, w = co[rows[:, 0]], co[rows[:, 1]], co[rows[:, 2]]
    middle = 0.5 * (u + w)
    both = (rows[:, 3] >= 0) & (rows[:, 4] >= 0)
    fair = middle.copy()
    uu, ww = co[rows[both, 3]], co[rows[both, 4]]
    fair[both] = (4.0 * (u[both] + w[both]) - (uu + ww)) / 6.0
    d = _unit(w - u)
    across = fair - v
    across -= (across * d).sum(axis=1, keepdims=True) * d
    return across + ((middle - v) * d).sum(axis=1, keepdims=True) * d


def settle(co, faces, project=None, stay=None, plane=None, strength: float = 1.0, snapshots=(), method: str = "settle", steps: int | None = None):
    """Pull the quads into shape on the surface. `project(points)` gives the nearest points of your mesh. `stay`: vertices
    that don't move (default `pinned`). `plane`: (axis, offset) of a mirror, whose vertices slide in it. Returns a list of
    (step, points) at each step in `snapshots`, on the surface, plus the last one."""
    co = np.asarray(co, dtype=np.float64)
    if stay is None:
        stay = pinned(co, faces, plane=plane)
    free = ~stay
    steps = int(round(SETTLE_STEPS * strength)) if steps is None else steps
    quads, _ids = _quads(faces)
    lines = _line_pairs(faces, len(co), stay)
    rows = _fair_rows(faces, len(co), stay) if method == "straighten" else None
    on_plane = None if plane is None else on_mirror(co, *plane) & free
    start = _normals(co, quads)
    was_bent = _bent(co, quads)
    square = SQUARE_PULL * strength
    # Detail sits where the quads turn: a vertex there moves less, one on a flat area freely.
    vn = _vertex_normals(co, quads)
    bend = np.zeros(len(co))
    np.maximum.at(bend, quads.reshape(-1), np.repeat(np.degrees(np.arccos(np.clip((start[:, None, :] * vn[quads]).sum(axis=2), -1.0, 1.0))).max(axis=1), 4))
    weight = np.clip(1.0 - bend / BEND_DEG, 0.0, 1.0)[:, None]
    a, b = quads.reshape(-1), np.roll(quads, -1, axis=1).reshape(-1)
    length = np.linalg.norm(co[a] - co[b], axis=1)
    reach = np.bincount(a, length, len(co)) + np.bincount(b, length, len(co))
    mean = reach / np.maximum(np.bincount(a, minlength=len(co)) + np.bincount(b, minlength=len(co)), 1)
    reach, drift_limit = MOST_MOVE * mean, MOST_DRIFT * mean
    x, last = co.copy(), co.copy()
    out = []
    wanted = set(int(s) for s in snapshots)

    def kept_shape(new, old):
        """`new` with the corners of quads it flips, folds or caves in put back to `old`."""
        bad = ((_normals(new, quads) * start).sum(axis=1) < WORST_TURN) | (_bent(new, quads) & ~was_bent)
        if bad.any():
            back = np.unique(quads[bad])
            new[back] = old[back]
        return new

    def landed(points):
        shot = points.copy()
        if project is not None:
            shot[free] = _onto(project, shot[free], reach[free])
        if on_plane is not None:
            shot[on_plane, plane[0]] = plane[1]
        return kept_shape(shot, points)  # onto the surface, except where that would bend a quad

    if 0 in wanted:
        out.append((0, landed(x)))
    for step in range(1, steps + 1):
        pull, count = np.zeros_like(x), np.zeros(len(x))
        if method == "settle":
            move, at = _rectangle_pull(x, quads, square)
            np.add.at(pull, at, move)
            np.add.at(count, at, 1.0)
            pull /= np.maximum(count, 1.0)[:, None]
            if len(lines):
                line, extra = np.zeros_like(x), np.zeros(len(x))
                np.add.at(line, lines[:, 0], _line_pull(x, lines))
                np.add.at(extra, lines[:, 0], 1.0)
                pull += LINE_WEIGHT * line / np.maximum(extra, 1.0)[:, None]
        else:
            move, at = _neighbour_pull(x, faces)
            np.add.at(pull, at, move)
            np.add.at(count, at, 1.0)
            pull /= np.maximum(count, 1.0)[:, None]
            if rows is not None and len(rows):  # loop corners follow their loops; poles keep the plain pull
                fair, extra = np.zeros_like(x), np.zeros(len(x))
                np.add.at(fair, rows[:, 0], _fair_pull(x, rows))
                np.add.at(extra, rows[:, 0], 1.0)
                on_rows = extra > 0
                pull[on_rows] = fair[on_rows] / extra[on_rows, None]
        n = _vertex_normals(x, quads)
        move = weight * (STIFF * pull + MOMENTUM * (x - last))
        move -= (move * n).sum(axis=1, keepdims=True) * n  # along the surface: the pull back onto it does the rest
        size = np.linalg.norm(move, axis=1)
        move *= np.minimum(1.0, reach / np.maximum(size, 1e-30))[:, None]
        new = x + move
        drift = new - co
        far = np.linalg.norm(drift, axis=1)
        new = co + drift * np.minimum(1.0, drift_limit / np.maximum(far, 1e-30))[:, None]
        new[stay] = co[stay]
        if project is not None:
            new[free] += ATTRACT * (_onto(project, new[free], reach[free]) - new[free])
        if on_plane is not None:
            new[on_plane, plane[0]] = plane[1]
        last, x = x, kept_shape(new, x)  # a quad flipping, folding or caving in: its corners stay where they were this step
        if step in wanted:
            out.append((step, landed(x)))
    out.append((steps, landed(x)))
    return out


def clean_poles(co, faces, keys, stay=None, judge=None, depth: int = 2, redraw: bool = True, pairs: bool = True):
    """Fewer poles: messy patches around poles redrawn as plain grids (`ChordReducer.redraw_all`), then edges spun to
    walk the poles left together (`turn_poles`), then with `pairs` pole pairs walked until they cancel
    (`fewerpoles.PoleCleaner`). `keys` per face: faces with different keys (another material or part)
    are never mixed, and every new face takes the key of the patch it replaced. `stay`: vertices that don't move.
    Returns (co, faces, keys, stay) of the cleaned mesh, with unused vertices dropped."""
    co = np.asarray(co, dtype=np.float64)
    reducer = ChordReducer(MeshData(co, [tuple(f) for f in faces], list(keys)), judge)
    stay = np.zeros(len(co), dtype=bool) if stay is None else np.asarray(stay, dtype=bool)
    for v in np.flatnonzero(stay).tolist():
        reducer.locked[v] = True  # not `freeze`: that stops the whole connected part
    if redraw:  # without the judge: its check of a new grid skips the fold test, and the redrawn grids came out bent
        reducer.judge = None
        reducer.redraw_all(rounds=len(reducer.users))  # until a round takes nothing away; it stops itself sooner
        reducer.judge = judge
    reducer.turn_poles(depth=depth)
    if pairs:
        PoleCleaner(reducer).run()
    alive = [i for i, f in enumerate(reducer.faces) if f is not None]
    used = np.unique(np.fromiter((v for i in alive for v in reducer.faces[i]), dtype=np.int64))
    remap = np.full(len(reducer.points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = np.asarray(reducer.points, dtype=np.float64)[used]
    held = np.concatenate((stay, np.zeros(len(reducer.points) - len(stay), dtype=bool)))[used]
    return points, [tuple(int(remap[v]) for v in reducer.faces[i]) for i in alive], [reducer.materials[i] for i in alive], held


def fewer_poles(co, faces, keys, stay=None, judge=None, project=None, plane=None, cycles: int = CYCLES, redraw: bool = True, stopped=None):
    """`clean_poles` again and again with a short settle in between: once the quads are even, more patches fit a grid and
    more pole pairs can walk. At least `cycles` rounds, then on while a round still takes poles away. Same arguments and
    result as `clean_poles`; `project` and `plane` as in `settle`. `stopped()` True ends it after the round running."""
    co = np.asarray(co, dtype=np.float64)
    stay = np.zeros(len(co), dtype=bool) if stay is None else np.asarray(stay, dtype=bool)
    before, k = poles(co, faces), 0
    while True:
        co, faces, keys, stay = clean_poles(co, faces, keys, stay, judge, redraw=redraw)
        k += 1
        now = poles(co, faces)
        if (k >= cycles and now >= before) or (stopped is not None and stopped()):
            return co, faces, keys, stay
        before = now
        stay = stay | pinned(co, faces, plane=plane)
        co = settle(co, faces, project, stay, plane, method="smooth", steps=CYCLE_STEPS)[-1][1]


def poles(co, faces) -> int:
    """Corners an artist would call poles (see `patterns.Topology`)."""
    from .patterns import Topology

    return int(Topology(co, faces).poles.sum())


def half_of(co, faces, axis: int, offset: float, keep: int):
    """The faces of a mirrored result on the kept side: (co, faces, ids of those faces in `faces`)."""
    co = np.asarray(co, dtype=np.float64)
    middle = np.asarray([co[list(f)].mean(axis=0)[axis] for f in faces])
    ids = np.flatnonzero(keep * (middle - offset) > 0.0)
    kept = [faces[i] for i in ids]
    used = np.unique(np.fromiter((v for f in kept for v in f), dtype=np.int64))
    remap = np.full(len(co), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return co[used], [tuple(int(remap[v]) for v in f) for f in kept], ids


def meter(before: dict, after: dict) -> str:
    """The line under the Organize slider: each number from strength 0 to the one shown."""
    loops = ""
    if before.get("loops") and after.get("loops"):
        from .loopscore import meter as loop_meter

        loops = ". " + loop_meter(before["loops"], after["loops"])
    view = "" if before.get("view") is None or after.get("view") is None else f", view changed {before['view']:.1f}% -> {after['view']:.1f}%"
    faces = "" if before.get("faces") is None or after.get("faces") is None else f", faces {before['faces']:,} -> {after['faces']:,}"
    return f"Tidiness {before['tidy']:.0f} -> {after['tidy']:.0f}{view}, poles {before['poles']:,} -> {after['poles']:,}{faces}{loops}"


def pick(levels: dict, wanted: float, cap: float | None = None, base: float | None = None):
    """The ready strength to show for the slider at `wanted`: the nearest at or below it whose view change stays within
    `cap` points of `base` (default: strength 0's; with Tidy Again, the result before the first pass). None: no cap.
    Strength 0 is always ready."""
    if base is None:
        base = levels.get(0.0, {}).get("view")
    fine = [s for s, row in levels.items() if s <= wanted + 1e-9 and (cap is None or base is None or row.get("view") is None or row["view"] <= base + cap + 1e-9)]
    return max(fine, default=0.0)


def gained(before: dict, after: dict, least: float = MIN_GAIN) -> bool:
    """Whether a pass made the mesh tidier: fewer poles, or Tidiness or Loop Quality up by `least` points, and none of the
    three worse. Keep Tidying stops when no strength of a pass does this."""
    ups = [before["poles"] - after["poles"], after["tidy"] - before["tidy"]]
    margins = [1, least]
    a, b = (before.get("loops") or {}).get("quality"), (after.get("loops") or {}).get("quality")
    if a is not None and b is not None:
        ups.append(b - a)
        margins.append(least)
    return any(u >= m for u, m in zip(ups, margins)) and all(u > -m for u, m in zip(ups, margins))


__all__ = ["clean_poles", "fewer_poles", "gained", "half_of", "meter", "pick", "pinned", "poles", "settle"]
