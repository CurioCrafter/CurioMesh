# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Special parts: wires and stacks of fins as sweeps, cloth as rows of long quads, and copies of one piece recognised so it's only worked out once.

A tube's stations come from a scalar field along it: the walk from one end for an open piece, the angle around the middle
for a closed loop (a chain link). Each station has a centre and a radius; the path is simplified and swept again with few
sides. A sheet is spread over a square; grid lines bunch up where the surface turns, so quads lie long along the folds.
Pure NumPy. The caller checks the result against the real piece, so a wrong guess costs time and never quality.
"""

from __future__ import annotations

import numpy as np

from .parts import orient_like  # noqa: F401 (the same vote for these rebuilds)
from .patch import quad_quality
from .types import FlatFaces, drain

SLENDER = 0.2  # a tube's mean radius is at most this share of its length
TUBES = ((0.5, 0.5), (0.25, 0.25), (0.12, 0.25), (0.12, 0.1), (0.06, 0.1), (0.03, 0.05))  # (section, path) tolerances as shares of the radius, fewest faces first
PRISM_SHARE = 0.8  # share of a straight sweep's surface that faces straight sideways or along it
PRISM_STATIONS = 24  # cuts along a straight sweep
PRISM_INSET = 0.002  # first and last cut this share of its length in from the ends, so they don't cut along the caps
PRISMS = ((0.12, 0.2), (0.06, 0.1), (0.03, 0.1), (0.015, 0.05), (0.007, 0.03))  # (section, path) tolerances, fewest faces first
MOST_MISSED = 0.7  # share of a tube's stations that must have a clean cross-section
SAMPLES = 6000  # points over a piece's surface to find its stations from
RASTER = 256  # cells per side of the look-up from a sheet's flattened square back to its triangles
FINE = 64  # cells per side of the first look at how a sheet folds
PLAIN = 0.5  # how much plain length counts against folds when placing a sheet's lines
MOST_STRETCH = 6.0  # how much longer a sheet's quads may run along its folds than across
BEND = 0.8  # how far a sheet's lines may bend from straight towards the folds near them
BLUR = 2  # fine cells to either side a sheet's row looks at
SHEETS = (0.03, 0.06, 0.12, 0.25, 0.5)  # shares of a sheet's triangles to try as quads
COPY_TOLERANCE = 0.002  # copies' vertices agree to about this share of the piece's size
MOST_TRIES = 48  # placements to try before giving up on two pieces being copies
COPY_EXACT = 0.05  # of the tolerance: a placement this close is the copy, so no other is tried
COPY_MORE = 8  # placements still tried after one fits, for one that fits closer


SOLVE_ROUNDS = 64  # conjugate gradient rounds per slice in `_solve_steps`
SECTIONS_PER_SLICE = 4  # cross-sections cut per slice
RASTER_TRIANGLES = 2_000  # triangles laid on a sheet's raster per slice


def _sides(polys) -> np.ndarray:
    """Each face's sides as (corner, the corner before it), face after face, by the arrays alone."""
    flat = FlatFaces.of(polys)
    if not len(flat.flat):
        return np.zeros((0, 2), dtype=np.int64)
    before = np.arange(len(flat.flat)) - 1
    before[flat.indptr[:-1]] = flat.indptr[1:] - 1  # a face's first corner follows its last
    return np.column_stack([flat.flat, flat.flat[before]])


def _edges(polys) -> np.ndarray:
    return _edge_uses(polys)[0]


def _edge_uses(polys) -> tuple[np.ndarray, np.ndarray]:
    """(the edges as sorted vertex pairs in order, how many faces use each), by one sort of one number per side."""
    pairs = np.sort(_sides(polys), axis=1)
    span = int(pairs.max(initial=0)) + 1
    keys, uses = np.unique(pairs[:, 0] * span + pairs[:, 1], return_counts=True)
    return np.column_stack([keys // span, keys % span]).reshape(-1, 2), uses


def _solve(points: np.ndarray, edges: np.ndarray, known: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Values spread smoothly over the mesh from the `known` vertices, by conjugate gradients with inverse edge lengths.

    All weights are positive, so nothing folds over when this flattens a sheet (Tutte), and a tube's values make clean rings.
    """
    return drain(_solve_steps(points, edges, known, values))


def _solve_steps(points: np.ndarray, edges: np.ndarray, known: np.ndarray, values: np.ndarray):
    """Generator: `_solve`, yielding every SOLVE_ROUNDS rounds."""

    n = len(points)
    weight = 1.0 / np.maximum(np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1), 1e-30)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    both = np.concatenate([weight, weight])
    total = np.maximum(np.bincount(row, both, n), 1e-30)
    free = ~known

    def laplace(x: np.ndarray) -> np.ndarray:
        return total * x - np.bincount(row, both * x[col], n)

    fixed = values * known
    x = np.zeros(n)
    r = -laplace(fixed) * free
    z = r / total
    d = z.copy()
    rz = float(r @ z)
    for done in range(min(3 * n, 2000)):
        if rz <= 1e-24:
            break
        if done % SOLVE_ROUNDS == SOLVE_ROUNDS - 1:
            yield
        q = laplace(d) * free
        step = rz / max(float(d @ q), 1e-300)
        x += step * d
        r -= step * q
        z = r / total
        new = float(r @ z)
        d = z + (new / rz) * d
        rz = new
    return fixed + x


def _surface_samples(points: np.ndarray, polys, field: np.ndarray, count: int = SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Points spread evenly over the surface, each with its field value, so long thin faces count for what they cover."""

    tris = _triangles(polys)
    corner = points[tris]
    area = np.linalg.norm(np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]), axis=1)
    many = np.ceil(area / max(float(area.sum()), 1e-30) * count).astype(np.int64)
    random = np.random.default_rng(len(points))  # the same piece always gives the same samples
    owner = np.repeat(np.arange(len(tris)), many)
    w = random.random((len(owner), 2))
    flip = w.sum(axis=1) > 1.0
    w[flip] = 1.0 - w[flip]
    weights = np.column_stack([1.0 - w.sum(axis=1), w])
    at = (corner[owner] * weights[:, :, None]).sum(axis=1)
    return at, (field[tris[owner]] * weights).sum(axis=1)


def _frames(centres: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tangent and two sideways directions at each station, carried along the path without twisting."""

    n = len(centres)
    if closed:
        tangents = np.roll(centres, -1, axis=0) - np.roll(centres, 1, axis=0)
    else:
        tangents = np.gradient(centres, axis=0)
    tangents = tangents / np.maximum(np.linalg.norm(tangents, axis=1), 1e-30)[:, None]
    u = np.cross(tangents[0], np.eye(3)[int(np.argmin(np.abs(tangents[0])))])
    u /= np.linalg.norm(u)
    us = [u]
    for i in range(1, n):
        u = us[-1] - tangents[i] * float(us[-1] @ tangents[i])
        u = u / max(float(np.linalg.norm(u)), 1e-30)
        us.append(u)
    us = np.asarray(us)
    vs = np.cross(tangents, us)
    if closed:  # share out what's left of the turn when the frame comes back to the start
        last = us[-1] - tangents[0] * float(us[-1] @ tangents[0])
        twist = float(np.arctan2(last @ vs[0], last @ us[0]))
        turn = -twist * np.arange(n) / n
        us, vs = np.cos(turn)[:, None] * us + np.sin(turn)[:, None] * vs, -np.sin(turn)[:, None] * us + np.cos(turn)[:, None] * vs
    return tangents, us, vs


def _triangles(polys) -> np.ndarray:
    """Each face as a fan of triangles from its first corner, face after face, by the arrays alone."""
    flat = FlatFaces.of(polys)
    fans = np.maximum(np.diff(flat.indptr) - 2, 0)
    first = np.repeat(flat.indptr[:-1], fans)
    step = np.arange(int(fans.sum())) - np.repeat(np.cumsum(fans) - fans, fans) + 1
    return np.column_stack([flat.flat[first], flat.flat[first + step], flat.flat[first + step + 1]]).reshape(-1, 3)


def _sections(points, tris, stations, largest: bool = False):
    """Generator: `_section` at each (centre, tangent, u, v), a slice per SECTIONS_PER_SLICE of them."""
    out = []
    for c, t, u, v in stations:
        out.append(_section(points, tris, c, t, u, v, largest))
        if len(out) % SECTIONS_PER_SLICE == 0:
            yield
    return out


def _section(points: np.ndarray, tris: np.ndarray, centre: np.ndarray, tangent: np.ndarray, u: np.ndarray, v: np.ndarray, largest: bool = False) -> np.ndarray | None:
    """The outline where a plane across the path cuts the piece, nearest the path (or the `largest`), in (u, v). None if it doesn't close."""

    height = (points - centre) @ tangent
    above = height[tris] > 0.0
    crossing = np.flatnonzero(above.any(axis=1) & ~above.all(axis=1))
    # each face the plane crosses joins the two edges it crosses
    links: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tri in tris[crossing].tolist():
        cut = [(min(a, b), max(a, b)) for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])) if (height[a] > 0.0) != (height[b] > 0.0)]
        links.setdefault(cut[0], []).append(cut[1])
        links.setdefault(cut[1], []).append(cut[0])
    best = None
    left = set(links)
    while left:
        start = left.pop()
        loop, here, back = [start], start, None
        while True:
            onward = [e for e in links[here] if e != back]
            if not onward or onward[0] == start:
                closes = bool(onward)
                break
            back, here = here, onward[0]
            if here not in left:
                closes = False
                break
            left.discard(here)
            loop.append(here)
        if not closes or len(loop) < 3:
            continue
        rows = np.asarray(loop)
        t = (height[rows[:, 0]] / (height[rows[:, 0]] - height[rows[:, 1]]))[:, None]
        where = points[rows[:, 0]] + t * (points[rows[:, 1]] - points[rows[:, 0]]) - centre
        flat = np.column_stack([where @ u, where @ v])
        near = -float(np.ptp(flat, axis=0).sum()) if largest else float(np.linalg.norm(flat.mean(axis=0)))
        if best is None or near < best[0]:
            best = (near, flat)
    return None if best is None else best[1]


def _evened(centres: np.ndarray, closed: bool) -> np.ndarray:
    """Jitter in the path would add up to a twist in the frames, so even it out."""

    for _ in range(4):
        smooth = 0.25 * np.roll(centres, 1, axis=0) + 0.5 * centres + 0.25 * np.roll(centres, -1, axis=0)
        if not closed:
            smooth[[0, -1]] = centres[[0, -1]]
        centres = smooth
    return centres


def _steady(values: np.ndarray, closed: bool) -> np.ndarray:
    """A running median over a few stations: sampling jitter shouldn't show up as lumps."""

    n = len(values)
    at = np.arange(n)[:, None] + np.arange(-3, 4)
    at = at % n if closed else np.clip(at, 0, n - 1)
    return np.median(values[at], axis=1)


def fit_tube(points: np.ndarray, polys) -> dict | None:
    """The path and cross-sections of a slender piece. None if it isn't a tube."""
    return drain(fit_tube_steps(points, polys))


def fit_tube_steps(points: np.ndarray, polys):
    """Generator: `fit_tube`, a slice per few cross-sections."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 12:
        return None
    centred = points - points.mean(axis=0)
    spread = np.sqrt(np.maximum(np.linalg.eigvalsh(centred.T @ centred / len(points)), 0.0))
    edges, uses = _edge_uses(polys)
    rim = np.unique(edges[uses == 1])  # `_rim_vertices`, from the same sort
    yield
    closed = len(points) - len(edges) + len(polys) == 0 and not len(rim)
    if closed:
        frame = np.linalg.eigh(centred.T @ centred)[1]
        field = np.arctan2(centred @ frame[:, 1], centred @ frame[:, 2])
        if float(np.linalg.norm(centred @ frame[:, 1:], axis=1).min()) < 0.4 * float(spread[2]):
            return None  # the middle of a loop has to be empty
    else:
        far = np.linalg.norm(centred, axis=1)
        one = int(np.argmax(far))
        two = int(np.argmax(np.linalg.norm(points - points[one], axis=1)))
        known = np.zeros(len(points), dtype=bool)
        known[[one, two]] = True
        values = np.zeros(len(points))
        values[two] = 1.0
        field = yield from _solve_steps(points, edges, known, values)
    yield
    samples, values = _surface_samples(points, polys, field)
    yield
    order = np.argsort(values)
    slices = np.array_split(order, 32)
    centres = np.asarray([samples[k].mean(axis=0) for k in slices])
    length = float(np.linalg.norm(np.diff(np.vstack([centres, centres[:1]]) if closed else centres, axis=0), axis=1).sum())
    rough = float(np.mean([np.linalg.norm(samples[k] - samples[k].mean(axis=0), axis=1).mean() for k in slices]))
    if length <= 0.0 or rough > SLENDER * length:
        return None
    slices = np.array_split(order, int(np.clip(length / (0.5 * rough), 8, 128)))
    centres = _evened(np.asarray([samples[k].mean(axis=0) for k in slices]), closed)
    tris = _triangles(polys)
    for _ in range(2):  # the real outlines say where the middle is; then cut again square to the better path
        tangents, us, vs = _frames(centres, closed)
        outlines = yield from _sections(points, tris, zip(centres, tangents, us, vs))
        good = np.asarray([o is not None for o in outlines])
        if good.mean() < MOST_MISSED:
            return None
        lows = np.asarray([o.min(axis=0) if o is not None else (0.0, 0.0) for o in outlines])
        highs = np.asarray([o.max(axis=0) if o is not None else (0.0, 0.0) for o in outlines])
        at = np.arange(len(centres))
        middles = np.stack([np.interp(at, at[good], 0.5 * (lows + highs)[good, k]) for k in range(2)], axis=1)
        centres = _evened(centres + middles[:, 0, None] * us + middles[:, 1, None] * vs, closed)
    half = 0.5 * (highs - lows)
    wide = _steady(np.interp(at, at[good], half[good, 0]), closed)
    deep = _steady(np.interp(at, at[good], half[good, 1]), closed)
    # the section comes from where the piece is most typical, and everywhere else scales from that
    far = np.abs(wide / max(float(np.median(wide)), 1e-30) - 1.0) + np.abs(deep / max(float(np.median(deep)), 1e-30) - 1.0) + 1e6 * ~good
    k = int(np.argmin(far))
    section = (outlines[k] - 0.5 * (lows[k] + highs[k])) / np.maximum(half[k], 1e-30)
    ends = (False, False)
    if len(rim) and not closed:
        ends = (bool((field[rim] < 0.5).any()), bool((field[rim] >= 0.5).any()))
    return {"centres": centres, "us": us, "vs": vs, "wide": wide, "deep": deep, "closed": closed, "open_ends": ends, "section": section, "size": float(np.median(wide) + np.median(deep)) / 2.0}


def fit_prisms(points: np.ndarray, polys) -> list[dict]:
    """The piece as the same outline along a straight line (a stack of fins, a gear, a plate), for each line that may do."""
    return drain(fit_prism_steps(points, polys))


def fit_prism_steps(points: np.ndarray, polys):
    """Generator: `fit_prisms`, a slice per few cross-sections."""

    points = np.asarray(points, dtype=np.float64)
    if len(polys) < 24:
        return []
    tris = _triangles(polys)
    corner = points[tris]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    area = np.linalg.norm(normal, axis=1)
    normal = normal / np.maximum(area, 1e-30)[:, None]
    centred = points - points.mean(axis=0)
    # along the right line nearly all the surface faces straight out sideways or straight along it
    axes = np.linalg.eigh(centred.T @ centred)[1].T
    shares = [float(area[(np.abs(normal @ a) < 0.1) | (np.abs(normal @ a) > 0.98)].sum()) / max(float(area.sum()), 1e-30) for a in axes]
    fits = []
    for k in range(3):
        if shares[k] >= PRISM_SHARE:
            yield
            fits.append((yield from _fit_prism(points, polys, tris, axes[k], axes[(k + 1) % 3])))
    return [fit for fit in fits if fit is not None]


def _fit_prism(points: np.ndarray, polys, tris: np.ndarray, axis: np.ndarray, u: np.ndarray):
    """Generator: the prism along this axis, or None."""
    v = np.cross(axis, u)
    height = points @ axis
    low, high = float(height.min()), float(height.max())
    if high <= low:
        return None
    middle = points.mean(axis=0) - axis * float(points.mean(axis=0) @ axis)
    inset = PRISM_INSET * (high - low)
    levels = np.linspace(low + inset, high - inset, PRISM_STATIONS)
    centres = middle + levels[:, None] * axis
    outlines = yield from _sections(points, tris, ((c, axis, u, v) for c in centres), True)
    good = np.asarray([o is not None for o in outlines])
    if good.mean() < MOST_MISSED:
        return None
    half = np.asarray([0.5 * np.ptp(o, axis=0) if o is not None else (0.0, 0.0) for o in outlines])
    mid = np.asarray([0.5 * (o.min(axis=0) + o.max(axis=0)) if o is not None else (0.0, 0.0) for o in outlines])
    at = np.arange(len(centres))
    wide = np.interp(at, at[good], half[good, 0])
    deep = np.interp(at, at[good], half[good, 1])
    centres = centres + np.interp(at, at[good], mid[good, 0])[:, None] * u + np.interp(at, at[good], mid[good, 1])[:, None] * v
    # the ends reach the piece's real ends
    centres[0] -= axis * inset
    centres[-1] += axis * inset
    far = np.abs(wide / max(float(np.median(wide)), 1e-30) - 1.0) + np.abs(deep / max(float(np.median(deep)), 1e-30) - 1.0) + 1e6 * ~good
    k = int(np.argmin(far))
    section = (outlines[k] - mid[k]) / np.maximum(half[k], 1e-30)
    rim = _rim_vertices(polys)
    ends = (bool(len(rim) and (height[rim] < 0.5 * (low + high)).any()), bool(len(rim) and (height[rim] >= 0.5 * (low + high)).any()))
    return {"centres": centres, "us": np.tile(u, (len(at), 1)), "vs": np.tile(v, (len(at), 1)), "wide": wide, "deep": deep, "closed": False, "open_ends": ends, "section": section, "size": float(np.median(wide) + np.median(deep)) / 2.0}


def _rim_vertices(polys) -> np.ndarray:
    edges, count = _edge_uses(polys)
    return np.unique(edges[count == 1])


def _simplify(path: np.ndarray, tol: float, closed: bool) -> np.ndarray:
    """Indices of the path's stations kept so no dropped one is farther than `tol` (path holds position and radius)."""

    n = len(path)
    if closed:
        far = int(np.argmax(np.linalg.norm(path - path[0], axis=1)))
        stack = [(0, far), (far, n)]
        keep = {0, far}
    else:
        stack = [(0, n - 1)]
        keep = {0, n - 1}
    looped = np.vstack([path, path[:1]]) if closed else path
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        d = looped[b] - looped[a]
        mid = looped[a + 1 : b] - looped[a]
        sq = float(d @ d)
        along = (mid @ d) / sq if sq > 0 else np.zeros(len(mid))
        dist = np.linalg.norm(mid - along[:, None] * d, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep.add(a + 1 + k)
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return np.asarray(sorted(keep), dtype=np.int64)


def _cap(ring: list[int], shape: np.ndarray) -> list[tuple[int, ...]]:
    """A flat end of a ring: a strip of quads if it's convex, otherwise triangles that the pairing makes into quads."""

    n = len(ring)
    ahead = np.roll(shape, -1, axis=0) - shape
    turns = ahead[:, 0] * np.roll(ahead, -1, axis=0)[:, 1] - ahead[:, 1] * np.roll(ahead, -1, axis=0)[:, 0]
    if n % 2 == 0 and ((turns >= 0.0).all() or (turns <= 0.0).all()):
        return [(ring[i], ring[i + 1], ring[n - i - 2], ring[(n - i - 1) % n]) for i in range(n // 2 - 1)]
    return [(ring[a], ring[b], ring[c]) for a, b, c in _clip_ears(shape)]


def _clip_ears(shape: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangles filling a simple polygon, keeping its winding."""

    left = list(range(len(shape)))
    x, y = shape[:, 0], shape[:, 1]
    area = float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())
    sign = 1.0 if area >= 0.0 else -1.0

    def cross(a: int, b: int, c: int) -> float:
        return float((x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a]))

    out = []
    while len(left) > 3:
        n = len(left)
        best, best_size = None, np.inf  # the smallest good ear first keeps the rest roomy
        for i in range(n):
            a, b, c = left[i - 1], left[i], left[(i + 1) % n]
            size = sign * cross(a, b, c)
            if size <= 0.0 or size >= best_size:
                continue
            others = [k for k in left if k not in (a, b, c)]
            if not any(sign * cross(a, b, k) >= 0.0 and sign * cross(b, c, k) >= 0.0 and sign * cross(c, a, k) >= 0.0 for k in others):
                best, best_size = i, size
        if best is None:  # a polygon that crosses itself: close it however it goes
            best = 0
        out.append((left[best - 1], left[best], left[(best + 1) % n]))
        left.pop(best)
    out.append((left[0], left[1], left[2]))
    return out


def build_tube(fit: dict, section_tol: float, path_tol: float, kind: str = "0") -> tuple[np.ndarray, list[tuple[int, ...]], list[str]] | None:
    """The fitted path as a sweep of the piece's own cross-section, simplified, with rows of long quads."""

    centres, closed, size = fit["centres"], fit["closed"], fit["size"]
    wide, deep = fit["wide"], fit["deep"]
    keep = _simplify(np.column_stack([centres, size * wide, size * deep]), path_tol * size, closed)
    if len(keep) < (3 if closed else 2):
        return None
    section = fit["section"]
    shape = section[_simplify(section, section_tol, True)]
    if len(shape) < 3:
        return None
    if len(shape) % 2:  # an odd ring's end leaves a triangle over after pairing
        k = int(np.argmax(np.linalg.norm(np.roll(shape, -1, axis=0) - shape, axis=1)))
        shape = np.insert(shape, k + 1, 0.5 * (shape[k] + shape[(k + 1) % len(shape)]), axis=0)
    sides = len(shape)
    n = len(keep)
    points = np.zeros((n * sides, 3))
    for i, k in enumerate(keep.tolist()):
        ring = (wide[k] * shape[:, 0])[:, None] * fit["us"][k] + (deep[k] * shape[:, 1])[:, None] * fit["vs"][k]
        points[i * sides : (i + 1) * sides] = centres[k] + ring
    polys: list[tuple[int, ...]] = []
    for i in range(n if closed else n - 1):
        a, b = i * sides, (i + 1) % n * sides
        polys += [(a + j, a + (j + 1) % sides, b + (j + 1) % sides, b + j) for j in range(sides)]
    if not closed:
        if not fit["open_ends"][0]:
            polys += [tuple(reversed(f)) for f in _cap(list(range(sides)), shape)]
        if not fit["open_ends"][1]:
            polys += _cap(list(range((n - 1) * sides, n * sides)), shape)
    return points, polys, [kind] * len(polys)


def _rim_loop(polys) -> list[int] | None:
    """The one boundary loop of a disc, in order. None if there's more than one or it isn't simple."""

    count: dict[tuple[int, int], int] = {}
    for p in polys:
        for i in range(len(p)):
            e = (min(p[i], p[i - 1]), max(p[i], p[i - 1]))
            count[e] = count.get(e, 0) + 1
    near: dict[int, list[int]] = {}
    for (a, b), c in count.items():
        if c == 1:
            near.setdefault(a, []).append(b)
            near.setdefault(b, []).append(a)
    if not near or any(len(n) != 2 for n in near.values()):
        return None
    start = next(iter(near))
    loop, prev = [start], -1
    while True:
        nxt = near[loop[-1]][0] if near[loop[-1]][0] != prev else near[loop[-1]][1]
        if nxt == start:
            break
        prev = loop[-1]
        loop.append(nxt)
        if len(loop) > len(near):
            return None
    return loop if len(loop) == len(near) else None


def _corners(points: np.ndarray, loop: list[int]) -> list[int]:
    """Four places on the rim (positions in the loop) that span it best: the corners of the sheet."""

    rim = points[loop]
    n = len(rim)
    centre = rim.mean(axis=0)
    frame = np.linalg.eigh((rim - centre).T @ (rim - centre))[1]
    flat = (rim - centre) @ frame[:, 1:]
    a, b = flat[:, 1] / max(float(np.ptp(flat[:, 1])), 1e-30), flat[:, 0] / max(float(np.ptp(flat[:, 0])), 1e-30)
    picks = {int(np.argmax(a + b)), int(np.argmax(a - b)), int(np.argmin(a + b)), int(np.argmin(a - b))}
    return sorted(picks) if len(picks) == 4 else [0, n // 4, n // 2, 3 * n // 4]


def _flatten(points: np.ndarray, polys, loop: list[int], corners: list[int]):
    """Generator: (u, v) in the unit square, the rim along its sides by length, the inside spread smoothly."""

    n = len(points)
    uv = np.zeros((n, 2))
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rim = np.asarray(loop)
    for side in range(4):
        a, b = corners[side], corners[(side + 1) % 4]
        at = rim[np.arange(a, (b if b > a else b + len(rim)) + 1) % len(rim)]
        steps = np.linalg.norm(np.diff(points[at], axis=0), axis=1)
        t = np.concatenate(([0.0], np.cumsum(steps))) / max(float(steps.sum()), 1e-30)
        uv[at] = square[side] + t[:, None] * (square[(side + 1) % 4] - square[side])
    known = np.zeros(n, dtype=bool)
    known[rim] = True
    edges = _edges(polys)
    yield
    first = yield from _solve_steps(points, edges, known, uv[:, 0])
    second = yield from _solve_steps(points, edges, known, uv[:, 1])
    return np.stack([first, second], axis=1)


def _weights(corners: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Barycentric weights of points in their triangles (a negative one means outside)."""

    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    v0, v1, v2 = b - a, c - a, at - a
    d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    d = np.where(np.abs(d) < 1e-30, 1e-30, d)
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / d
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / d
    return np.stack([1.0 - w1 - w2, w1, w2], axis=1)


def fit_sheet(points: np.ndarray, polys) -> dict | None:
    """A disc with one rim, spread over a square so rows can follow it. None if it isn't one."""
    return drain(fit_sheet_steps(points, polys))


def fit_sheet_steps(points: np.ndarray, polys):
    """Generator: `fit_sheet`, in slices."""

    points = np.asarray(points, dtype=np.float64)
    if len(polys) < 48 or len(points) - len(_edges(polys)) + len(polys) != 1:
        return None
    yield
    loop = _rim_loop(polys)
    if loop is None:
        return None
    yield
    uv = yield from _flatten(points, polys, loop, _corners(points, loop))
    tris = _triangles(polys)
    # which triangle lies over each cell of a fine raster of the square, its centre decided by the best weights
    corner = uv[tris] * (RASTER - 1)
    low = np.floor(corner.min(axis=1)).astype(np.int64)
    high = np.ceil(corner.max(axis=1)).astype(np.int64)
    under = np.zeros((RASTER, RASTER), dtype=np.int64)
    best = np.full((RASTER, RASTER), -np.inf)
    for t in range(len(tris)):
        if t % RASTER_TRIANGLES == RASTER_TRIANGLES - 1:
            yield
        xs, ys = np.meshgrid(np.arange(low[t, 0], high[t, 0] + 1), np.arange(low[t, 1], high[t, 1] + 1), indexing="ij")
        at = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
        score = _weights(np.repeat(corner[t][None], len(at), axis=0), at).min(axis=1)
        better = score > best[xs.ravel(), ys.ravel()]
        best[xs.ravel()[better], ys.ravel()[better]] = score[better]
        under[xs.ravel()[better], ys.ravel()[better]] = t
    return {"uv": uv, "tris": tris, "under": under, "points": points}


def _sample(fit: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Points of the source at grid lines `u` by `v`, shape (len(u), len(v), 3)."""

    gu, gv = np.meshgrid(u, v, indexing="ij")
    return _sample_at(fit, gu.ravel(), gv.ravel()).reshape(len(u), len(v), 3)


def _sample_at(fit: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Points of the source at these places of the square."""

    at = np.stack([u, v], axis=1)
    cell = np.clip(np.rint(at * (RASTER - 1)).astype(np.int64), 0, RASTER - 1)
    tri = fit["tris"][fit["under"][cell[:, 0], cell[:, 1]]]
    w = np.clip(_weights(fit["uv"][tri], at), 0.0, 1.0)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
    return (fit["points"][tri] * w[:, :, None]).sum(axis=1)


def _lines(need: np.ndarray, count: int) -> np.ndarray:
    """`count` + 1 positions in 0..1 so each gap holds the same amount of `need` (given per fine step)."""

    total = np.concatenate(([0.0], np.cumsum(need)))
    return np.interp(np.linspace(0.0, total[-1], count + 1), total, np.linspace(0.0, 1.0, len(total)))


def build_sheet(fit: dict, faces: int, kind: str = "0") -> tuple[np.ndarray, list[tuple[int, ...]], list[str]]:
    """About `faces` quads in rows: lines bunch up where the surface turns across them, so quads lie long along the folds."""

    fine = np.linspace(0.0, 1.0, FINE + 1)
    grid = _sample(fit, fine, fine)
    a = np.cross(grid[1:, :-1] - grid[:-1, :-1], grid[:-1, 1:] - grid[:-1, :-1])
    normal = a / np.maximum(np.linalg.norm(a, axis=2, keepdims=True), 1e-30)
    turn_u = np.linalg.norm(normal[1:, :] - normal[:-1, :], axis=2).sum(axis=1)
    turn_v = np.linalg.norm(normal[:, 1:] - normal[:, :-1], axis=2).sum(axis=0)
    long_u = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean(axis=1)
    long_v = np.linalg.norm(np.diff(grid, axis=1), axis=2).mean(axis=0)
    # some of the lines go by plain length, or a flat stretch would get none and hang wrong
    need_u = np.concatenate(([turn_u[0]], turn_u)) / max(float(turn_u.sum()), 1e-30) + PLAIN * long_u / max(float(long_u.sum()), 1e-30)
    need_v = np.concatenate(([turn_v[0]], turn_v)) / max(float(turn_v.sum()), 1e-30) + PLAIN * long_v / max(float(long_v.sum()), 1e-30)
    ratio = min(max((float(turn_u.sum()) + 1e-9) / (float(turn_v.sum()) + 1e-9), 1.0 / MOST_STRETCH), MOST_STRETCH)
    nu = max(2, int(round(np.sqrt(faces * ratio))))
    nv = max(2, int(round(faces / nu)))
    u, v = _lines(need_u, nu), _lines(need_v, nv)
    # each row and column then places its own nodes, so lines bend to follow a fold that runs slantwise
    across_u = np.linalg.norm(normal[1:, :] - normal[:-1, :], axis=2)
    across_v = np.linalg.norm(normal[:, 1:] - normal[:, :-1], axis=2)
    gu, gv = np.meshgrid(u, v, indexing="ij")
    for j in range(nv + 1):
        at = min(int(v[j] * (FINE - 1)), FINE - 1)
        rows = across_u[:, max(at - BLUR, 0) : at + BLUR + 1].mean(axis=1)
        gu[:, j] = (1.0 - BEND) * u + BEND * _lines(np.concatenate(([rows[0]], rows)) / max(float(rows.sum()), 1e-30) + PLAIN * long_u / max(float(long_u.sum()), 1e-30), nu)
    for i in range(nu + 1):
        at = min(int(u[i] * (FINE - 1)), FINE - 1)
        cols = across_v[max(at - BLUR, 0) : at + BLUR + 1, :].mean(axis=0)
        gv[i, :] = (1.0 - BEND) * v + BEND * _lines(np.concatenate(([cols[0]], cols)) / max(float(cols.sum()), 1e-30) + PLAIN * long_v / max(float(long_v.sum()), 1e-30), nv)
    points = _sample_at(fit, gu.ravel(), gv.ravel())
    w = nv + 1
    polys = [(i * w + j, (i + 1) * w + j, (i + 1) * w + j + 1, i * w + j + 1) for i in range(nu) for j in range(nv)]
    return points, polys, [kind] * len(polys)


def _nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each of `a` to the nearest of `b`."""

    out = np.empty(len(a))
    for i in range(0, len(a), 256):
        out[i : i + 256] = np.sqrt(((a[i : i + 256, None, :] - b[None]) ** 2).sum(axis=2).min(axis=1))
    return out


def match_copy(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    """How piece `a` moves rigidly (or mirrored) onto piece `b`: (rotation, a's centre, b's centre, mirrored). None if it can't.

    Vertex order differs between copies, so two vertices of `a` far from its centre and from each other are looked for in `b`.
    """

    centre_a, centre_b = a.mean(axis=0), b.mean(axis=0)
    p, q = a - centre_a, b - centre_b
    size = float(np.ptp(a, axis=0).max())
    tol = COPY_TOLERANCE * size
    radius_a, radius_b = np.linalg.norm(p, axis=1), np.linalg.norm(q, axis=1)
    if abs(float(np.sort(radius_a)[-1] - np.sort(radius_b)[-1])) > 2.0 * tol or np.abs(np.sort(radius_a) - np.sort(radius_b)).mean() > 2.0 * tol:
        return None
    one = int(np.argmax(radius_a))
    axis = p[one] / max(float(radius_a[one]), 1e-30)
    off_line = np.linalg.norm(p - (p @ axis)[:, None] * axis, axis=1)
    two = int(np.argmax(off_line))
    side = np.cross(axis, p[two])
    three = int(np.argmax(np.abs(p @ side) / max(float(np.linalg.norm(side)), 1e-30)))
    frame_a = _frame(p[one], p[two])
    if frame_a is None:
        return None
    sample = np.linspace(0, len(p) - 1, min(len(p), 48)).astype(int)
    tried = 0
    best = None  # (mean distance of the sample, placement): a nearly symmetric part also fits turned the wrong way, within tolerance
    apart = float(np.linalg.norm(p[one] - p[two]))
    for i in np.flatnonzero(np.abs(radius_b - radius_a[one]) <= 2.0 * tol)[:64].tolist():
        for j in np.flatnonzero((np.abs(radius_b - radius_a[two]) <= 2.0 * tol) & (np.abs(np.linalg.norm(q - q[i], axis=1) - apart) <= 2.0 * tol))[:64].tolist():
            frame_b = _frame(q[i], q[j])
            if frame_b is None:
                continue
            for mirror in (False, True):
                flip = np.diag([1.0, 1.0, -1.0 if mirror else 1.0])
                turn = frame_b @ flip @ frame_a.T
                if float(_nearest((p[three] @ turn.T)[None], q)[0]) > 2.0 * tol:
                    continue
                tried += 1
                if tried > MOST_TRIES or (best is not None and tried > best[2] + COPY_MORE):
                    return None if best is None else best[1]
                moved_a = p @ turn.T
                near = _nearest(moved_a[sample], q)
                if near.max() > 2.0 * tol or (best is not None and near.mean() >= best[0]):
                    continue
                if _nearest(moved_a, q).max() <= 2.0 * tol:
                    best = (float(near.mean()), (turn, centre_a, centre_b, mirror), tried if best is None else best[2])
                    if best[0] <= COPY_EXACT * tol:
                        return best[1]
    return None if best is None else best[1]


def _frame(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    x = a / max(float(np.linalg.norm(a)), 1e-30)
    z = np.cross(x, b)
    length = float(np.linalg.norm(z))
    if length <= 1e-12 * max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30):
        return None
    z /= length
    return np.stack([x, np.cross(z, x), z], axis=1)


def vertex_owners(points: np.ndarray, faces, known_points: np.ndarray, known_faces, known_parts: np.ndarray) -> np.ndarray:
    """Which part each vertex of a processed mesh belongs to, from the parts of the mesh it was made from.

    A connected stretch of faces goes by what most of its vertices were; a new vertex (from closing a stretch) follows it.
    """

    size = max(float(np.ptp(known_points, axis=0).max()), 1e-30) if len(known_points) else 1.0
    known_owner = np.zeros(len(known_points), dtype=np.int64)
    for f, part in zip(known_faces, np.asarray(known_parts).tolist()):
        known_owner[list(f)] = part
    table = {tuple(key): part for key, part in zip(np.rint(known_points / (1e-9 * size)).astype(np.int64).tolist(), known_owner.tolist())}
    keys = np.rint(points / (1e-9 * size)).astype(np.int64).tolist()
    parent = list(range(len(points)))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for f in faces:
        first = find(f[0])
        for v in f[1:]:
            parent[find(v)] = first
    votes: dict[int, dict[int, int]] = {}
    for v, key in enumerate(keys):
        part = table.get(tuple(key))
        if part is not None:
            count = votes.setdefault(find(v), {})
            count[part] = count.get(part, 0) + 1
    owner = np.zeros(len(points), dtype=np.int64)
    for v in range(len(points)):
        count = votes.get(find(v))
        if count:
            owner[v] = max(count, key=count.get)
        else:
            owner[v] = known_owner[int(np.argmin(((known_points - points[v]) ** 2).sum(axis=1)))]
    return owner


def moved(points: np.ndarray, move: tuple) -> np.ndarray:
    turn, centre_a, centre_b, _mirror = move
    return (points - centre_a) @ turn.T + centre_b


def wound(polys, move: tuple) -> list[tuple[int, ...]]:
    """A mirrored copy's faces have to be wound the other way to keep facing out."""

    return [tuple(reversed(p)) for p in polys] if move[3] else list(polys)


def _unfolded(out):
    """The rebuilt piece, or None if any of its quads folds over."""

    if out is None:
        return None
    quads = np.asarray([f for f in out[1] if len(f) == 4], dtype=np.int64)
    return None if len(quads) and (quad_quality(out[0][quads]) <= 0.0).any() else out


def fit_steps(points: np.ndarray, polys):
    """Generator: (tube, prisms, sheet) as `rebuilds` would fit them, in slices, to hand to `rebuilds` as `fits`."""
    tube = yield from fit_tube_steps(points, polys)
    yield
    prisms = yield from fit_prism_steps(points, polys)
    yield
    sheet = yield from fit_sheet_steps(points, polys)
    return tube, prisms, sheet


def rebuilds(points: np.ndarray, polys, kind: str = "0", fits=None):
    """Generator of kinds of rebuild that may suit a piece. Each is a generator of (points, quads, kinds) or None, most faces first.

    `fits` is what `fit_steps` returned for this piece; without it each fit is made as its turn comes."""

    fit = fits[0] if fits is not None else fit_tube(points, polys)
    if fit is not None:
        yield (_unfolded(build_tube(fit, section, path, kind)) for section, path in reversed(TUBES))
    for fit in fits[1] if fits is not None else fit_prisms(points, polys):
        yield (_unfolded(build_tube(fit, section, path, kind)) for section, path in reversed(PRISMS))
    fit = fits[2] if fits is not None else fit_sheet(points, polys)
    if fit is not None:
        yield (_unfolded(build_sheet(fit, max(16, int(share * len(polys))), kind)) for share in reversed(SHEETS))
