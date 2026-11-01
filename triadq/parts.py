# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Part rebuilds: a loose piece that is a surface of revolution (pipe, bolt, ring, link) is rebuilt by hand-style rules.

The piece's rings are found by grouping vertices with the same (radius, height) around an axis. The profile through the
rings is simplified and swept again with a few angular steps, so a pipe becomes long quads with clean rings. Pure NumPy.
The caller checks the result against the real piece, so a wrong guess costs time and never quality.
"""

from __future__ import annotations

import math

import numpy as np

CLUSTER = 0.004  # vertices closer than this (share of the piece's diagonal) in (radius, height) are one ring
MIN_SPIN = 8  # a ring has at least this many vertices to count as turned around the axis
REVOLVES = ((8, 0.010), (12, 0.006), (16, 0.004), (24, 0.002), (32, 0.001))  # (steps around, profile tolerance), fewest faces first
BEAM_STATIONS = (1, 2, 4)
BOX_SHARE = 0.7  # share of surface area that must face along the box axes for a closed piece to count as a box
FITS = 0.75  # share of vertices that must sit on a ring (or the axis) for the piece to count as turned


def _axes(points: np.ndarray) -> list[np.ndarray]:
    centred = points - points.mean(axis=0)
    found = list(np.linalg.eigh(centred.T @ centred)[1].T) + list(np.eye(3))
    out: list[np.ndarray] = []
    for a in found:
        if all(abs(float(a @ b)) < 0.999 for b in out):
            out.append(a)
    return out


def _cluster(values: np.ndarray, tol: float) -> np.ndarray:
    """Leader clustering of 2D points; returns a cluster id per point."""

    cells: dict[tuple[int, int], list[int]] = {}
    leaders: list[np.ndarray] = []
    ids = np.zeros(len(values), dtype=np.int64)
    for i, p in enumerate(values):
        cx, cy = int(p[0] // tol), int(p[1] // tol)
        hit = -1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in cells.get((cx + dx, cy + dy), ()):
                    if hit < 0 and float(np.hypot(*(leaders[k] - p))) <= tol:
                        hit = k
        if hit < 0:
            hit = len(leaders)
            leaders.append(p)
            cells.setdefault((cx, cy), []).append(hit)
        ids[i] = hit
    return ids


def _boundary_vertices(polys) -> set[int]:
    count: dict[tuple[int, int], int] = {}
    for p in polys:
        for i in range(len(p)):
            e = (min(p[i], p[i - 1]), max(p[i], p[i - 1]))
            count[e] = count.get(e, 0) + 1
    return {v for e, c in count.items() if c == 1 for v in e}


def _rdp(profile: np.ndarray, tol: float) -> list[int]:
    """Indices kept when a path is simplified so that no dropped point is farther than `tol`."""

    keep = [0, len(profile) - 1]
    stack = [(0, len(profile) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        d = profile[b] - profile[a]
        length = float(np.hypot(*d))
        mid = profile[a + 1 : b] - profile[a]
        dist = np.abs(mid[:, 0] * d[1] - mid[:, 1] * d[0]) / length if length > 1e-30 else np.hypot(mid[:, 0], mid[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep.append(a + 1 + k)
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return sorted(set(keep))


def fit_revolve(points: np.ndarray, polys) -> dict | None:
    """Find an axis around which the piece is turned, and its profile. None if the piece isn't such a part."""

    points = np.asarray(points, dtype=np.float64)
    diag = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diag <= 0.0 or len(points) < 12:
        return None
    tol = CLUSTER * diag
    best = None
    for axis in _axes(points):
        helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        frame = np.stack([u, np.cross(axis, u), axis], axis=1)
        local = points @ frame
        center = (local.min(axis=0) + local.max(axis=0)) / 2.0
        local = local - center
        rz = np.stack([np.hypot(local[:, 0], local[:, 1]), local[:, 2]], axis=1)
        ids = _cluster(rz, tol)
        size = np.bincount(ids)
        on_axis = rz[:, 0] <= tol
        share = float(((size[ids] >= MIN_SPIN) | on_axis).mean())
        if best is None or share > best[0]:
            best = (share, frame, center, local, rz, ids)
    share, frame, center, local, rz, ids = best
    if share < FITS:
        return None
    count = int(ids.max()) + 1
    mean = np.stack([np.bincount(ids, rz[:, k], count) for k in range(2)], axis=1) / np.bincount(ids, None, count)[:, None]
    # rings that really turn (or sit on the axis) are the profile; anything else (a lug, a bolt) is left out
    valid = (np.bincount(ids) >= MIN_SPIN) | (mean[:, 0] <= tol)
    near = [set() for _ in range(count)]
    for p in polys:
        for i in range(len(p)):
            a, b = int(ids[p[i]]), int(ids[p[i - 1]])
            if a != b and valid[a] and valid[b]:
                near[a].add(b)
                near[b].add(a)
    nodes = [k for k in range(count) if valid[k] and near[k]]
    if len(nodes) < 2:
        return None

    def walk(start):
        """Breadth first: (farthest ring, parents) from `start`."""
        parent, queue = {start: -1}, [start]
        for k in queue:
            for m in near[k]:
                if m not in parent:
                    parent[m] = k
                    queue.append(m)
        return queue[-1], parent

    closed = all(len(near[k]) == 2 for k in nodes) and len(walk(nodes[0])[1]) == len(nodes)
    if closed:
        order, prev = [nodes[0]], -1
        while len(order) < len(nodes):
            nxt = [m for m in near[order[-1]] if m != prev and m not in order]
            if not nxt:
                return None
            prev = order[-1]
            order.append(nxt[0])
    else:
        far, _ = walk(nodes[0])
        end, parent = walk(far)
        order = [end]
        while parent[order[-1]] != -1:
            order.append(parent[order[-1]])
    profile = mean[order]
    sizes = np.bincount(ids)[order]
    # which end is open: an end with boundary edges is a rim, otherwise the piece is capped there
    rim = _boundary_vertices(polys)
    open_end = [False, False]
    if not closed:
        for slot, k in enumerate((order[0], order[-1])):
            open_end[slot] = any(ids[v] == k for v in rim)
    # the ring with the most vertices gives the angular phase and count
    big = int(order[int(np.argmax(sizes))])
    members = np.flatnonzero(ids == big)
    phase = float(math.atan2(local[members[0], 1], local[members[0], 0]))
    return {"frame": frame, "center": center, "profile": profile, "closed": closed, "open_end": open_end, "phase": phase, "spin": int(len(members)), "diag": diag}


def _simplify(profile: np.ndarray, closed: bool, tol: float) -> np.ndarray:
    if closed:
        start = int(np.argmin(profile[:, 1]))
        far = int(np.argmax(np.hypot(*(profile - profile[start]).T)))
        a, b = sorted((start, far))
        one = profile[a : b + 1]
        two = np.vstack([profile[b:], profile[: a + 1]])
        keep_a = [one[i] for i in _rdp(one, tol)]
        keep_b = [two[i] for i in _rdp(two, tol)]
        return np.asarray(keep_a[:-1] + keep_b[:-1])
    return profile[_rdp(profile, tol)]


def _cap(rim: list[int], radius: float, z_rim: float, z_pole: float, phase: float, points: list, polys: list) -> None:
    """Fill a ring of 4k vertices with a k by k grid of quads."""

    n = len(rim)
    k = n // 4
    # square boundary order: bottom row, right column, top row, left column
    cell = {}
    for j in range(n):
        if j <= k:
            cell[(j, 0)] = j
        elif j <= 2 * k:
            cell[(k, j - k)] = j
        elif j <= 3 * k:
            cell[(3 * k - j, k)] = j
        else:
            cell[(0, n - j)] = j
    ring = np.asarray([points[v][:2] for v in rim])
    plane = {}
    for (a, b), j in cell.items():
        plane[(a, b)] = ring[j]
    for a in range(1, k):
        for b in range(1, k):
            s, t = a / k, b / k
            plane[(a, b)] = (
                (1 - t) * plane[(a, 0)] + t * plane[(a, k)] + (1 - s) * plane[(0, b)] + s * plane[(k, b)]
                - ((1 - s) * (1 - t) * plane[(0, 0)] + s * (1 - t) * plane[(k, 0)] + (1 - s) * t * plane[(0, k)] + s * t * plane[(k, k)])
            )
    for _ in range(12):
        for a in range(1, k):
            for b in range(1, k):
                plane[(a, b)] = 0.25 * (plane[(a - 1, b)] + plane[(a + 1, b)] + plane[(a, b - 1)] + plane[(a, b + 1)])
    at = {}
    for (a, b), j in cell.items():
        at[(a, b)] = rim[j]
    for a in range(1, k):
        for b in range(1, k):
            rho = float(np.hypot(*plane[(a, b)])) / max(radius, 1e-30)
            at[(a, b)] = len(points)
            points.append((float(plane[(a, b)][0]), float(plane[(a, b)][1]), z_rim + (z_pole - z_rim) * (1.0 - min(rho, 1.0) ** 2)))
    for a in range(k):
        for b in range(k):
            polys.append((at[(a, b)], at[(a + 1, b)], at[(a + 1, b + 1)], at[(a, b + 1)]))


def build_revolve(fit: dict, spin: int, tol_share: float, kind: str = "0") -> tuple[np.ndarray, list[tuple[int, ...]], list[str]] | None:
    """The fitted part as quads with `spin` steps around (a multiple of 4), rings at the simplified profile."""

    diag = fit["diag"]
    profile = _simplify(fit["profile"], fit["closed"], tol_share * diag)
    closed = fit["closed"]
    if len(profile) < (3 if closed else 2):
        return None
    tol = CLUSTER * diag
    # profile points on the axis are poles; every other point is a ring
    local: list[tuple[float, float, float]] = []
    rings: list[list[int] | None] = []
    # radius that makes the polygon's area match the circle's
    scale = math.sqrt((2.0 * math.pi / spin) / math.sin(2.0 * math.pi / spin))
    for r, z in profile:
        if r <= 2 * tol:
            rings.append(None)
            continue
        ids = []
        for j in range(spin):
            t = fit["phase"] + 2.0 * math.pi * j / spin
            ids.append(len(local))
            local.append((r * scale * math.cos(t), r * scale * math.sin(t), z))
        rings.append(ids)
    polys: list[tuple[int, ...]] = []
    for i in range(len(profile) - (0 if closed else 1)):
        a, b = rings[i], rings[(i + 1) % len(profile)]
        if a is None or b is None:
            continue
        for j in range(spin):
            polys.append((a[j], a[(j + 1) % spin], b[(j + 1) % spin], b[j]))
    # caps: a pole at an end, or a flat close of a rim the source had closed
    if not closed:
        for slot, at in ((0, 0), (1, len(profile) - 1)):
            ring = rings[at]
            nxt = rings[at + 1 if slot == 0 else at - 1]
            if ring is None:
                if nxt is None:
                    return None
                z_pole = float(profile[at][1])
                z_rim = local[nxt[0]][2]
                _cap(nxt, float(profile[at + 1 if slot == 0 else at - 1][0]), z_rim, z_pole, fit["phase"], local, polys)
            elif not fit["open_end"][slot]:
                z = local[ring[0]][2]
                _cap(ring, float(profile[at][0]), z, z, fit["phase"], local, polys)
    if not polys:
        return None
    world = (np.asarray(local) + fit["center"]) @ fit["frame"].T
    return world, _consistent(polys), [kind] * len(polys)


def _consistent(polys) -> list[tuple[int, ...]]:
    """Wind neighbouring faces the same way (flood from the first face)."""

    polys = [tuple(p) for p in polys]
    sides: dict[tuple[int, int], list[int]] = {}
    for f, p in enumerate(polys):
        for i in range(len(p)):
            sides.setdefault((min(p[i], p[i - 1]), max(p[i], p[i - 1])), []).append(f)
    done = [False] * len(polys)
    for start in range(len(polys)):
        if done[start]:
            continue
        done[start] = True
        stack = [start]
        while stack:
            f = stack.pop()
            p = polys[f]
            for i in range(len(p)):
                for g in sides[(min(p[i], p[i - 1]), max(p[i], p[i - 1]))]:
                    if g == f or done[g]:
                        continue
                    q = polys[g]
                    if any(q[j - 1] == p[i - 1] and q[j] == p[i] for j in range(len(q))):
                        polys[g] = q[::-1]
                    done[g] = True
                    stack.append(g)
    return polys


def orient_like(points: np.ndarray, polys, source_points: np.ndarray, source_polys) -> list[tuple[int, ...]]:
    """Flip the new faces together if most of them face against the nearest source faces."""

    def frames(p, f):
        c = np.asarray([p[list(x)].mean(axis=0) for x in f])
        n = np.asarray([np.cross(p[x[2]] - p[x[0]], p[x[1 if len(x) == 3 else 3]] - p[x[0]]) for x in f])
        return c, n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]

    c0, n0 = frames(source_points, source_polys)
    c1, n1 = frames(points, polys)
    pick = np.linspace(0, len(c1) - 1, min(len(c1), 64)).astype(int)
    near = np.argmin(((c1[pick][:, None, :] - c0[None]) ** 2).sum(axis=2), axis=1)
    vote = float((np.sum(n1[pick] * n0[near], axis=1)).sum())
    return [tuple(reversed(p)) for p in polys] if vote < 0 else list(polys)


def _area_normals(points: np.ndarray, polys) -> tuple[np.ndarray, np.ndarray]:
    """(unit normal, area) of every polygon, by Newell's method."""

    n = np.zeros((len(polys), 3))
    for k, p in enumerate(polys):
        q = points[list(p)]
        n[k] = np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0)
    length = np.linalg.norm(n, axis=1)
    return n / np.maximum(length, 1e-30)[:, None], 0.5 * length


def fit_box(points: np.ndarray, polys) -> dict | None:
    """The oriented bounding box along the piece's principal axes, and the share of surface area facing along them."""

    points = np.asarray(points, dtype=np.float64)
    normals, area = _area_normals(points, polys)
    total = float(area.sum())
    if total <= 0.0:
        return None
    centred = points - points.mean(axis=0)
    frame = np.linalg.eigh(centred.T @ centred)[1][:, ::-1]
    if np.linalg.det(frame) < 0:
        frame[:, 2] *= -1.0
    along = np.abs(normals @ frame).max(axis=1) > math.cos(math.radians(12.0))
    local = points @ frame
    return {"frame": frame, "low": local.min(axis=0), "high": local.max(axis=0), "share": float(area[along].sum() / total), "local": local}


def build_box(fit: dict, divisions: tuple[int, int, int], kind: str = "0") -> tuple[np.ndarray, list[tuple[int, ...]], list[str]]:
    """The fitted box as a grid of quads on each side, `divisions` cells along each axis."""

    at: dict[tuple[int, int, int], int] = {}
    local: list[np.ndarray] = []
    d = divisions
    span = fit["high"] - fit["low"]

    def vertex(i, j, k):
        key = (i, j, k)
        if key not in at:
            at[key] = len(local)
            local.append(fit["low"] + span * np.asarray([i / d[0], j / d[1], k / d[2]]))
        return at[key]

    polys: list[tuple[int, ...]] = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for side in (0, d[axis]):
            for a in range(d[u]):
                for b in range(d[v]):
                    cell = []
                    for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        idx = [0, 0, 0]
                        idx[axis], idx[u], idx[v] = side, a + da, b + db
                        cell.append(vertex(*idx))
                    polys.append(tuple(cell))
    return np.asarray(local) @ fit["frame"].T, _consistent(polys), [kind] * len(polys)


def build_beam(fit: dict, stations: int, kind: str = "0") -> tuple[np.ndarray, list[tuple[int, ...]], list[str]] | None:
    """A box whose rectangle is fitted again at each of `stations` + 1 places along the long axis, so taper and bow follow."""

    local, low, high = fit["local"], fit["low"], fit["high"]
    length = high[0] - low[0]
    if length <= 0.0 or stations < 1:
        return None
    t = low[0] + length * np.arange(stations + 1) / stations
    reach = 0.5 * length / stations
    rings = []
    for at in t:
        near = local[np.abs(local[:, 0] - at) <= reach]
        if len(near) < 4:
            return None
        lo, hi = near[:, 1:].min(axis=0), near[:, 1:].max(axis=0)
        rings.append([(at, lo[0], lo[1]), (at, hi[0], lo[1]), (at, hi[0], hi[1]), (at, lo[0], hi[1])])
    polys: list[tuple[int, ...]] = [(0, 1, 2, 3), tuple(range(4 * stations + 3, 4 * stations - 1, -1))]
    for i in range(stations):
        for j in range(4):
            polys.append((4 * i + j, 4 * i + (j + 1) % 4, 4 * (i + 1) + (j + 1) % 4, 4 * (i + 1) + j))
    return np.asarray([v for r in rings for v in r]) @ fit["frame"].T, _consistent(polys), [kind] * len(polys)


def rebuilds(points: np.ndarray, polys, kind: str = "0"):
    """Generator of rebuilt versions of a piece, fewest faces first, as (points, quads, kinds). None if it isn't a known part."""

    fit = fit_revolve(points, polys)
    if fit is not None:
        for spin, tolerance in REVOLVES:
            yield build_revolve(fit, spin, tolerance, kind)
        return
    if _boundary_vertices(polys):
        return
    box = fit_box(points, polys)
    if box is not None and box["share"] >= BOX_SHARE:
        for stations in BEAM_STATIONS:
            yield build_beam(box, stations, kind)
