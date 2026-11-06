# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Ring Round This: clean rings of quads round an eye, a mouth or a hole, built straight into the result.

The faces in a disk round the opening are replaced; the disk's border vertices and every face outside it stay as they
are. Inside out, the new faces are:

- the opening: a small a x b grid of quads filling the rim (a discrete Coons patch, Coons 1967): the lids or the lips;
- the rings: closed strips of quads round the rim, each with the same count of quads, so every one is a face loop;
- the band: one strip from the outermost ring to the faces left round it. Where the outside has more edges than the
  ring, a band quad takes two outside edges and none of the ring's (a corner quad); those few spots are the band's
  poles, and they go where the old border turns most. Which outside vertex each ring vertex joins is the cheapest path
  through a table of pairings, the way two outlines are joined into a strip (Fuchs, Kedem and Uselton 1977).

The layout is made flat, in the plane the opening faces, laid back onto the old faces there and then handed to
`project`, so every point lands on the source. The rim comes from the caller (an oval the user dragged) or `find_rim`
reads it off the source: where the surface bends most going out from the middle. Plain NumPy.
"""

from __future__ import annotations

import math

import numpy as np

from .patch import boundary_loops, coons_grid, face_adjacency, mask_components
from .rowfill import untangle

BAND = 2.5  # quad widths round the rim whose faces are redone, by default (the user drags this)
FANS = 6  # most corner quads (poles) in the band: past this the rings get more quads instead
SIDEWAYS_DEG = 85.0  # faces turned further than this from the way the opening faces stay (the side of a nose)
TIGHTEST = 0.3  # quad widths: the rim's quads are kept at least this long along the rim


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _areas(co: np.ndarray, faces) -> np.ndarray:
    """Area vector of every face (its normal times twice its area, by the shoelace sum)."""
    out = np.zeros((len(faces), 3))
    for k, f in enumerate(faces):
        p = co[list(f)]
        out[k] = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
    return out


def _vertex_faces(faces, count: int) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(count)]
    for k, f in enumerate(faces):
        for v in f:
            out[v].append(k)
    return out


def _region(faces, mask: np.ndarray, adjacency, vertex_faces) -> tuple[np.ndarray, list[int]]:
    """`mask` grown into a disk with one even border loop: holes filled, pinched vertices and odd counts mended."""
    mask = mask.copy()
    for _ in range(12):
        rest = mask_components(faces, ~mask, adjacency)
        for hole in rest[1:]:  # everything but the outside
            mask[hole] = True
        loops, pinched = boundary_loops(faces, np.flatnonzero(mask))
        if pinched:
            for v in pinched:
                mask[vertex_faces[v]] = True
            continue
        if len(loops) != 1:
            raise ValueError("the faces round it do not make one patch")
        loop = loops[0]
        if len(loop) % 2 == 0:
            return mask, loop
        tris = [f for v in loop for f in vertex_faces[v] if len(faces[f]) == 3]
        for f in tris:
            trial = mask.copy()
            trial[f] = not trial[f]
            got, pin = boundary_loops(faces, np.flatnonzero(trial))
            if not pin and len(got) == 1 and len(got[0]) % 2 == 0 and len(mask_components(faces, ~trial, adjacency)) == 1:
                return trial, got[0]
        raise ValueError("its border has an odd number of edges")
    raise ValueError("the faces round it touch themselves")


def _ray_out(q: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """How far out from the origin, along each direction `theta`, a ray first meets the closed polygon `q`."""
    a, b = q, np.roll(q, -1, axis=0)
    d = np.stack((np.cos(theta), np.sin(theta)), axis=1)[:, None, :]
    e = (b - a)[None]
    den = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    safe = np.where(np.abs(den) < 1e-12, np.nan, den)
    t = (a[None, :, 0] * e[..., 1] - a[None, :, 1] * e[..., 0]) / safe
    s = (a[None, :, 0] * d[..., 1] - a[None, :, 1] * d[..., 0]) / safe
    t = np.where((t > 1e-9) & (s >= 0.0) & (s < 1.0), t, np.inf)
    return t.min(axis=1)


def _inside(q: np.ndarray) -> bool:
    """Is the origin inside the closed polygon `q` (winding number)?"""
    ang = np.arctan2(q[:, 1], q[:, 0])
    return abs(float(_wrap(np.roll(ang, -1) - ang).sum())) > np.pi


def _rim_angles(a: int, b: int) -> np.ndarray:
    """Angles round the rim (squashed to a circle) of the 2(a+b) ends of an a x b grid's border, counter-clockwise from
    its first corner; each side spans a quarter, so its corners sit on the diagonals."""
    sides = []
    for k, n in enumerate((a, b, a, b)):
        start = -0.75 * np.pi + 0.5 * np.pi * k
        sides.append(start + 0.5 * np.pi * np.arange(n) / n)
    return np.concatenate(sides)


def _stitch(theta: np.ndarray, phi: np.ndarray, corner_cost: np.ndarray, shape=None):
    """Join a ring's vertices (angles `theta`, rising) to the border's (angles `phi`, in order round it) with quads.

    A quad steps one along both (a plain quad), two along the border (a corner quad, cost `corner_cost` of its middle
    border vertex) or two along the ring; `shape(border vertex, ring vertex, kind)` adds the cost of the quad's own
    shape. The pairing with the least squared angle mismatch over all its spokes wins:
    a cheapest path through the table of (border step, ring step) pairs (Fuchs, Kedem and Uselton 1977 join two
    outlines with triangles the same way). Returns (border start, the kind of each quad in order) or None."""
    m, n = len(theta), len(phi)
    if (n + m) % 2:
        return None
    th = np.append(theta, theta[0] + 2.0 * np.pi)
    step = _wrap(np.roll(phi, -1) - phi)
    best = None
    first = np.argsort(np.abs(_wrap(phi - theta[0])))[: min(n, 6)]
    for k0 in first.tolist():
        ph = theta[0] + float(_wrap(phi[k0] - theta[0])) + np.concatenate(([0.0], np.cumsum(np.roll(step, -k0))))
        spoke = (ph[:, None] - th[None, :]) ** 2
        cost = np.full((n + 1, m + 1, 3), np.inf)
        back = np.zeros((n + 1, m + 1, 3, 3), dtype=np.int64)
        cost[0, 0, 0] = spoke[0, 0]
        for total in range(n + m + 1):
            for k in range(max(0, total - m), min(n, total) + 1):
                i = total - k
                for last in range(3):
                    here = cost[k, i, last]
                    if not np.isfinite(here):
                        continue
                    for kind, (dk, di) in enumerate(((1, 1), (2, 0), (0, 2))):
                        k2, i2 = k + dk, i + di
                        if k2 > n or i2 > m:
                            continue
                        c = here + spoke[k2, i2]
                        if kind == 1:
                            c += corner_cost[(k0 + k + 1) % n]
                        elif kind == 2:
                            c += 1.0
                        if shape is not None:
                            c += shape((k0 + k) % n, i % m, kind)
                        if kind and last:
                            c += 0.5  # two irregular quads side by side
                        if c < cost[k2, i2, kind]:
                            cost[k2, i2, kind] = c
                            back[k2, i2, kind] = (k, i, last)
        end = int(np.argmin(cost[n, m]))
        if np.isfinite(cost[n, m, end]) and (best is None or cost[n, m, end] < best[0]):
            path, at = [], (n, m, end)
            while at[:2] != (0, 0):
                path.append(at[2])
                at = tuple(int(x) for x in back[at])
            best = (float(cost[n, m, end]), k0, path[::-1])
    return None if best is None else (best[1], best[2])


def _signed_areas(p: np.ndarray, quads: np.ndarray) -> np.ndarray:
    q = p[quads]
    return 0.5 * (q[..., 0] * np.roll(q[..., 1], -1, axis=1) - np.roll(q[..., 0], -1, axis=1) * q[..., 1]).sum(axis=1)


def _bent(p: np.ndarray, quads: np.ndarray, sign: float) -> np.ndarray:
    """Quads with a corner bent the wrong way (darts) or folded over, in the flat layout."""
    q = p[quads]
    a, b = np.roll(q, 1, axis=1) - q, np.roll(q, -1, axis=1) - q
    turn = b[..., 0] * a[..., 1] - b[..., 1] * a[..., 0]
    return (sign * turn <= 0).any(axis=1)


def _quad_normals(p: np.ndarray, quads: np.ndarray) -> np.ndarray:
    n = np.cross(p[quads[:, 2]] - p[quads[:, 0]], p[quads[:, 3]] - p[quads[:, 1]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def _relax(xy: np.ndarray, plane: np.ndarray, faces, free: np.ndarray, base: int, rounds: int = 30) -> np.ndarray:
    """The outer rings eased toward the middle of their neighbours (umbrella smoothing, Taubin 1995), so the band's
    corner quads open up instead of lying flat; the rim and the old border stay. Stops before more quads bend the wrong way."""
    if not len(free):
        return xy
    quads = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([quads[:, [k, (k + 1) % 4]] for k in range(4)])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    p = np.vstack((plane[:base], xy))
    sign = np.sign(_signed_areas(p, quads).sum())
    folded = int(_bent(p, quads, sign).sum())
    moving = base + free
    for _ in range(rounds):
        total = np.zeros_like(p)
        count = np.zeros(len(p))
        np.add.at(total, edges[:, 0], p[edges[:, 1]])
        np.add.at(total, edges[:, 1], p[edges[:, 0]])
        np.add.at(count, edges.ravel(), 1.0)
        trial = p.copy()
        trial[moving] += 0.5 * (total[moving] / count[moving, None] - p[moving])
        now = int(_bent(trial, quads, sign).sum())
        if now > folded:
            break
        p, folded = trial, now
    free_set = set(moving.tolist())
    for _ in range(3):  # a quad still bent: its free corners stepped toward the middle of the other three
        bent = np.flatnonzero(_bent(p, quads, sign))
        if not len(bent):
            break
        for f in bent.tolist():
            for c in range(4):
                v = int(quads[f, c])
                if v not in free_set:
                    continue
                goal = p[np.delete(quads[f], c)].mean(axis=0)
                for t in (0.25, 0.5, 0.75, 1.0):
                    trial = p.copy()
                    trial[v] += t * (goal - p[v])
                    now = int(_bent(trial, quads, sign).sum())
                    if now < folded:
                        p, folded = trial, now
                        break
    return p[base:]


def _lift(xy: np.ndarray, plane: np.ndarray, co: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Flat points back onto the old faces: each in the old triangle it falls in (or nearest to), by its barycentric
    weights there."""
    a, b, c = plane[tris[:, 0]], plane[tris[:, 1]], plane[tris[:, 2]]
    v0, v1 = b - a, c - a
    v2 = xy[:, None, :] - a[None]
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    good = np.abs(den) > 1e-18 * np.maximum(d00 * d11, 1e-300)
    den = np.where(good, den, 1.0)
    l1 = (d11 * d20 - d01 * d21) / den
    l2 = (d00 * d21 - d01 * d20) / den
    w = np.stack((1.0 - l1 - l2, l1, l2), axis=-1)
    score = np.where(good, w.min(axis=-1), -np.inf)
    pick = np.argmax(score, axis=1)
    w = np.clip(w[np.arange(len(xy)), pick], 0.0, None)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
    t = tris[pick]
    return (w[:, :, None] * co[t]).sum(axis=1)


def facing(co, faces, centre, radius, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    """The way the surface within `radius` of `centre` faces: the sum of its faces' area vectors, made unit length."""
    co = np.asarray(co, dtype=np.float64)
    cent = np.asarray([co[list(f)].mean(axis=0) for f in faces])
    near = np.flatnonzero(np.linalg.norm(cent - centre, axis=1) <= radius)
    n = _areas(co, [faces[k] for k in near]).sum(axis=0) if len(near) else np.asarray(fallback, dtype=np.float64)
    if np.linalg.norm(n) < 1e-30:
        n = np.asarray(fallback, dtype=np.float64)
    return n / max(float(np.linalg.norm(n)), 1e-30)


def _frame(co, faces, centre, normal, right, radius):
    n = facing(co, faces, centre, radius, normal)
    u = np.asarray(right, dtype=np.float64)
    u = u - (u @ n) * n
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(n, (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0))
    u = u / np.linalg.norm(u)
    return n, u, np.cross(n, u)


def quad_size(co, faces, centre, radius) -> float:
    """Median edge length of the faces within `radius` of `centre` (all faces when none is that near)."""
    co = np.asarray(co, dtype=np.float64)
    cent = np.asarray([co[list(f)].mean(axis=0) for f in faces])
    near = np.flatnonzero(np.linalg.norm(cent - centre, axis=1) <= radius)
    pick = near if len(near) >= 4 else np.arange(len(faces))
    lengths = [np.linalg.norm(co[f[k]] - co[f[k - 1]]) for i in pick for f in (faces[i],) for k in range(len(f))]
    return float(np.median(lengths))


def find_rim(src_co, src_tris, centre, normal, right, size: float):
    """(width, height) of the opening round `centre` on the source, or None where nothing stands out there.

    A smooth bowl (z = a quadratic in x and y, by least squares) is fitted to the source 2.5 to 5 quad widths `size`
    out; what is left over in the middle is the eye's or the lips' bulge (or a hole's dip). Going out in each of 8
    directions, the rim is where that bulge falls to half its height; an oval is fitted through those 8 points."""
    co = np.asarray(src_co, dtype=np.float64)
    pts = np.vstack((co, co[np.asarray(src_tris)].mean(axis=1)))
    n = np.asarray(normal, dtype=np.float64) / max(float(np.linalg.norm(normal)), 1e-30)
    u = np.asarray(right, dtype=np.float64) - (np.asarray(right) @ n) * n
    u /= max(float(np.linalg.norm(u)), 1e-30)
    v = np.cross(n, u)
    d = pts - centre
    d = d[np.linalg.norm(d, axis=1) < 5.0 * size] / size
    x, y, z = d @ u, d @ v, d @ n
    r = np.hypot(x, y)
    outer = r > 2.5
    if outer.sum() < 12 or (r < 1.0).sum() < 3:
        return None
    a = np.stack((x * x, x * y, y * y, x, y, np.ones_like(x)), axis=1)
    fit, *_ = np.linalg.lstsq(a[outer], z[outer], rcond=None)
    rest = z - a @ fit
    top = float(rest[r < 1.0].mean())
    if abs(top) < 0.15:
        return None
    rest = rest * np.sign(top) / abs(top)  # 1 in the middle
    sector = ((np.arctan2(y, x) + np.pi) / (2 * np.pi) * 8).astype(np.int64) % 8
    radii, angles = [], []
    for s in range(8):
        mine = sector == s
        bins = np.minimum((r[mine] * 2).astype(np.int64), 8)
        total = np.bincount(bins, weights=rest[mine], minlength=9)[:8]
        count = np.bincount(bins, minlength=9)[:8]
        prof = total / np.maximum(count, 1)
        at = 4.0
        for k in range(1, 8):
            if count[k] and prof[k] < 0.5:
                was = prof[k - 1] if count[k - 1] else 1.0
                at = 0.5 * (k - 0.5 + (was - 0.5) / max(was - prof[k], 1e-9)) if was > 0.5 else 0.5 * k
                break
        radii.append(at)
        angles.append(-np.pi + (s + 0.5) * np.pi / 4)
    radii, angles = np.asarray(radii), np.asarray(angles)
    m = np.stack((np.cos(angles) ** 2, np.sin(angles) ** 2), axis=1)
    fit, *_ = np.linalg.lstsq(m, 1.0 / radii**2, rcond=None)
    wide, tall = sorted((float(np.clip(size / math.sqrt(f), 0.8 * size, 3.0 * size)) for f in np.maximum(fit, 1.0 / 9.0)), reverse=True)
    if wide <= 1.0 * size:  # under a quad across: a bump in the middle, not the opening's rim
        return None
    return wide, max(tall, 0.4 * wide)  # a click is read as an opening at least as wide as tall: draw it out for others


def build(co, faces, centre, normal, right, rx: float, ry: float, band: float = BAND, rings: int | None = None, project=None) -> dict:
    """Rings of quads round an oval rim (`rx` along `right`, `ry` across it) centred on `centre`, a point on the result.

    The faces within `band` quad widths outside the rim, joined to the clicked spot and facing the same way, are
    replaced. Returns {"co", "faces", "parent" (old face of each face, the nearest old one for a new face), "new" (ids
    of the new faces), "band" (ids of the band's faces), "rings", "count" (quads per ring), "grid" (a, b), "corners"
    (corner quads in the band), "size" (quad width), "replaced" (old faces taken out), "made" (new vertices, the last
    ones)}. Raises ValueError, in words a user can act on."""
    co = np.asarray(co, dtype=np.float64)
    faces = [tuple(int(v) for v in f) for f in faces]
    centre = np.asarray(centre, dtype=np.float64)
    reach = max(rx, ry)
    h = quad_size(co, faces, centre, 3.0 * reach)
    width = band * h
    areas = _areas(co, faces)
    cent = np.asarray([co[list(f)].mean(axis=0) for f in faces])
    n, u, v = _frame(co, faces, centre, normal, right, 1.5 * reach)
    d = cent - centre
    x, y, z = d @ u, d @ v, d @ n
    facing = areas @ n > math.cos(math.radians(SIDEWAYS_DEG)) * np.linalg.norm(areas, axis=1)
    near = ((x / (rx + width)) ** 2 + (y / (ry + width)) ** 2 <= 1.0) & (np.abs(z) <= 2.0 * (reach + width))
    adjacency = face_adjacency(faces)
    seed = int(np.argmin(np.where(near, np.linalg.norm(d, axis=1), np.inf)))
    if not near[seed]:
        raise ValueError("there are no faces there")
    picked = next(g for g in mask_components(faces, near & facing | (np.arange(len(faces)) == seed), adjacency) if seed in g)
    mask = np.zeros(len(faces), dtype=bool)
    mask[picked] = True
    fa, fb = adjacency
    for _ in range(4):  # a rounder patch: fill notches, drop faces that hang on by one edge
        shared = np.bincount(fa[mask[fb]], minlength=len(faces)) + np.bincount(fb[mask[fa]], minlength=len(faces))
        mask |= ~mask & facing & (shared >= 2)
        mask &= ~((shared <= 1) & (np.arange(len(faces)) != seed))
    vertex_faces = _vertex_faces(faces, len(co))
    mask, loop = _region(faces, mask, adjacency, vertex_faces)
    region = np.flatnonzero(mask)

    plane = np.stack(((co - centre) @ u, (co - centre) @ v), axis=1)
    border = plane[loop]
    signed = float((border[:, 0] * np.roll(border[:, 1], -1) - np.roll(border[:, 0], -1) * border[:, 1]).sum())
    flip = signed < 0.0
    if flip:
        loop = loop[::-1]
        border = border[::-1]
    squash = np.array((rx, ry))
    q = border / squash
    if not _inside(q):
        raise ValueError("the faces round it do not go all the way round its middle")

    probe = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    out = _ray_out(q, probe)
    oval = np.hypot(rx * np.cos(probe), ry * np.sin(probe))
    gap = (out - 1.0) * oval
    if not np.isfinite(out).all() or gap.min() < 0.5 * h:
        raise ValueError("the rim reaches the faces round it; drag the rim smaller or the band wider")
    count_rings = int(np.clip(round(float(np.median(gap)) / h), 2, 4)) if rings is None else int(rings)

    def ring_at(theta, j):
        far = _ray_out(q, theta)
        smooth = far.copy()
        for _ in range(3):
            smooth = 0.25 * np.roll(smooth, 1) + 0.5 * smooth + 0.25 * np.roll(smooth, -1)
        far = np.minimum(far, smooth)
        rho = 1.0 + j / (count_rings + 0.5) * (far - 1.0)  # the band gets a layer and a half: room for its corners
        return np.stack((rx * rho * np.cos(theta), ry * rho * np.sin(theta)), axis=1)

    def length(points):
        return float(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum())

    nb = len(loop)
    first, last = length(ring_at(probe, 0)), length(ring_at(probe, count_rings - 1))
    m = max(8, 2 * round(0.25 * (first + last) / h))
    m = max(m, nb - 2 * FANS)  # few poles in the band...
    m = min(m, max(8, 2 * int(first / (TIGHTEST * h) / 2)), nb + 2 * FANS)  # ...but no needle-thin quads at the rim
    half = m // 2
    gb = int(np.clip(round(half * ry / (rx + ry)), 1, half - 1))
    ga = half - gb
    theta = _rim_angles(ga, gb)
    ring_xy = [ring_at(theta, j) for j in range(count_rings)]

    phi = np.arctan2(q[:, 1], q[:, 0])
    e1, e2 = np.roll(border, 1, axis=0) - border, np.roll(border, -1, axis=0) - border
    inner = np.degrees(np.arctan2(e2[:, 0] * e1[:, 1] - e2[:, 1] * e1[:, 0], (e1 * e2).sum(axis=1)) % (2 * np.pi))
    corner_cost = 0.3 * np.clip((inner - 100.0) / 80.0, 0.0, 3.0) + 0.02
    outer_xy = ring_xy[-1]

    def shape(k, i, kind):  # a quad bent the wrong way (a dart) costs as much as a badly placed corner quad
        b = border[[(k + j) % nb for j in range(3)]]
        r = outer_xy[[(i + j) % m for j in range(3)]]
        quad = (b[0], b[1], r[1], r[0]) if kind == 0 else (b[0], b[1], b[2], r[0]) if kind == 1 else (b[0], r[2], r[1], r[0])
        return 10.0 if _bent(np.asarray(quad), np.arange(4)[None], 1.0)[0] else 0.0

    joined = _stitch(theta, phi, corner_cost, shape)
    if joined is None:
        raise ValueError("the rings can't be joined to the faces round them")
    k0, path = joined

    # new vertices: the grid's inside, then the rings
    grid_xy, ids = coons_grid(ring_xy[0], (0, ga, ga + gb, 2 * ga + gb))
    inside = ids < 0
    base = len(co)
    flat = [grid_xy[inside]]
    node = np.where(inside, -1, ids)
    node[inside] = base + np.arange(int(inside.sum()))
    ring_ids = []
    start = base + int(inside.sum())
    for j in range(count_rings):
        ring_ids.append(start + np.arange(m))
        flat.append(ring_xy[j])
        start += m
    node[~inside] = ring_ids[0][ids[~inside]]
    new_faces = []
    for i in range(ga):
        for j in range(gb):
            new_faces.append((node[i, j], node[i + 1, j], node[i + 1, j + 1], node[i, j + 1]))
    for j in range(count_rings - 1):
        r0, r1 = ring_ids[j], ring_ids[j + 1]
        for k in range(m):
            new_faces.append((r1[k], r1[(k + 1) % m], r0[(k + 1) % m], r0[k]))
    first_band = len(new_faces)
    outer = ring_ids[-1]
    rim = [loop[(k0 + k) % nb] for k in range(nb + 2)]
    k, i = 0, 0
    corners = 0
    for kind in path:
        if kind == 0:
            new_faces.append((rim[k], rim[k + 1], outer[(i + 1) % m], outer[i % m]))
        elif kind == 1:
            new_faces.append((rim[k], rim[k + 1], rim[k + 2], outer[i % m]))
            corners += 1
        else:
            new_faces.append((rim[k], outer[(i + 2) % m], outer[(i + 1) % m], outer[i % m]))
            corners += 1
        k, i = k + (1, 2, 0)[kind], i + (1, 0, 2)[kind]
    new_faces = [tuple(int(x) for x in (f[::-1] if flip else f)) for f in new_faces]

    xy = _relax(np.vstack(flat), plane, new_faces, np.concatenate(ring_ids[1:]) - base if count_rings > 1 else np.zeros(0, np.int64), base)
    tris = np.asarray([(f[0], f[t], f[t + 1]) for r in region for f in (faces[r],) for t in range(1, len(f) - 1)], dtype=np.int64)
    lifted = _lift(xy, plane, co, tris)
    points = lifted if project is None else np.asarray(project(lifted), dtype=np.float64)
    quads = np.asarray(new_faces, dtype=np.int64)
    flat_normals = _quad_normals(np.vstack((co, lifted)), quads)
    for _ in range(3):  # a quad the surface snap turned over gets its new corners back on the old faces
        turned = (_quad_normals(np.vstack((co, points)), quads) * flat_normals).sum(axis=1) < 0.3
        back = np.unique(quads[turned])
        back = back[back >= base] - base
        if not len(back):
            break
        points[back] = lifted[back]

    kept = np.flatnonzero(~mask)
    out_faces = [faces[f] for f in kept.tolist()] + new_faces
    fresh = np.arange(len(co) + len(points)) >= base
    every = untangle(np.vstack((co, points)), out_faces, fresh, project)
    used = np.zeros(len(every), dtype=bool)
    for f in out_faces:
        used[list(f)] = True
    remap = np.cumsum(used) - 1
    region_xy = np.asarray([plane[list(faces[r])].mean(axis=0) for r in region])
    new_xy = np.asarray([np.vstack((plane, xy))[list(f)].mean(axis=0) for f in new_faces])
    nearest = region[np.argmin(((new_xy[:, None, :] - region_xy[None]) ** 2).sum(-1), axis=1)]
    count_kept = len(kept)
    return {
        "co": every[used],
        "faces": [tuple(int(remap[v]) for v in f) for f in out_faces],
        "parent": np.concatenate((kept, nearest)).astype(np.int64),
        "new": list(range(count_kept, len(out_faces))),
        "band": list(range(count_kept + first_band, len(out_faces))),
        "rings": count_rings,
        "count": m,
        "grid": (ga, gb),
        "corners": corners,
        "size": h,
        "replaced": len(region),
        "made": int(used[len(co):].sum()),
    }
