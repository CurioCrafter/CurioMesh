# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Plain-NumPy code about one part of a mesh: what kind it is, cutting it out of a result and putting a new one in,
and rebuilding a turned part (pipe, bolt, ring, link) by hand-style rules.

A turned part's rings are found by grouping vertices with the same (radius, height) around an axis. The profile through
the rings is simplified and swept again with a few angular steps, so a pipe becomes long quads with clean rings. The
caller checks the result against the real piece, so a wrong guess costs time and never quality.
"""

from __future__ import annotations

import math

import numpy as np

from .types import FlatFaces, drain


KINDS = ("TUBE", "FLAT", "ROUGH")
KIND_WORDS = {"TUBE": "Tube", "FLAT": "Flat", "ROUGH": "Rough"}


def classify(points: np.ndarray, faces) -> str:
    """TUBE (long and round), FLAT (a thin sheet) or ROUGH (anything else), from the spread of the part's vertices.

    A rough guide for picking parts in bulk, not a measurement: `s1 >= s2 >= s3` are the spreads along the three axes
    of the vertex cloud. Under ten vertices there is nothing to tell, so it is ROUGH.
    """

    used = np.unique(np.fromiter((v for f in faces for v in f), dtype=np.int64))
    if len(used) < 10:
        return "ROUGH"
    cloud = np.asarray(points, dtype=np.float64)[used]
    spread = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov((cloud - cloud.mean(axis=0)).T)), 0.0))[::-1]
    s1, s2, s3 = (float(x) for x in spread)
    if s1 <= 1e-12:
        return "ROUGH"
    if s3 < 0.08 * s1 and s3 < 0.4 * s2:
        return "FLAT"
    if s2 < 0.45 * s1 and s3 > 0.5 * s2:
        return "TUBE"
    return "ROUGH"


def cut(co: np.ndarray, faces, mat, face_piece: np.ndarray, piece: int):
    """One part of a result on its own: (points, faces, materials) with its own vertex numbers."""

    ids = np.flatnonzero(np.asarray(face_piece) == piece)
    used, flat = np.unique(np.fromiter((v for i in ids.tolist() for v in faces[i]), dtype=np.int64), return_inverse=True)
    out, at = [], 0
    for i in ids.tolist():
        size = len(faces[i])
        out.append(tuple(int(v) for v in flat[at : at + size]))
        at += size
    return np.asarray(co)[used], out, np.asarray(mat)[ids]


def splice(co: np.ndarray, faces, mat, face_piece: np.ndarray, builds: dict):
    """The result with every part in `builds` ({piece: (points, faces, materials)}) replaced.

    Returns (points, faces, materials, face_piece). Vertices nobody uses any more are dropped.
    """

    face_piece = np.asarray(face_piece)
    keep = ~np.isin(face_piece, list(builds))
    all_faces = [f for f, k in zip(faces, keep) if k]
    all_mat = list(np.asarray(mat)[keep])
    all_piece = list(face_piece[keep])
    pieces = [co]
    offset = len(co)
    for piece, (points, new_faces, new_mat) in builds.items():
        pieces.append(np.asarray(points, dtype=np.float64))
        all_faces += [tuple(int(v) + offset for v in f) for f in new_faces]
        all_mat += [int(m) if str(m).lstrip("-").isdigit() else 0 for m in new_mat]
        all_piece += [piece] * len(new_faces)
        offset += len(points)
    every = np.vstack(pieces)
    used, flat = np.unique(np.fromiter((v for f in all_faces for v in f), dtype=np.int64), return_inverse=True)
    out, at = [], 0
    for f in all_faces:
        out.append(tuple(int(v) for v in flat[at : at + len(f)]))
        at += len(f)
    return every[used], out, np.asarray(all_mat, dtype=np.int32), np.asarray(all_piece, dtype=np.int64)


def worst_first(changes, count: int | None = None) -> list[int]:
    """Part numbers, the part whose view changed most first. Parts with no number (None) come last."""

    order = sorted(range(len(changes)), key=lambda i: (changes[i] is None, -(changes[i] or 0.0), i))
    return order if count is None else order[:count]


# viridis: readable with every common kind of colour blindness, and in grey
VIRIDIS = ((0.267, 0.005, 0.329), (0.229, 0.322, 0.546), (0.128, 0.567, 0.551), (0.369, 0.789, 0.383), (0.993, 0.906, 0.144))


def heat_colour(value: float | None, top: float) -> tuple[float, float, float]:
    """Dark purple for no change through teal to yellow at `top` percent or more (viridis). Grey when unknown."""

    if value is None:
        return 0.55, 0.55, 0.55
    t = min(max(value / max(top, 1e-9), 0.0), 1.0) * (len(VIRIDIS) - 1)
    k = min(int(t), len(VIRIDIS) - 2)
    a, b, f = VIRIDIS[k], VIRIDIS[k + 1], t - k
    return tuple(x + (y - x) * f for x, y in zip(a, b))


CLUSTER = 0.004  # vertices closer than this (share of the piece's diagonal) in (radius, height) are one ring
MIN_SPIN = 8  # a ring has at least this many vertices to count as turned around the axis
REVOLVES = ((8, 0.010), (12, 0.006), (16, 0.004), (24, 0.002), (32, 0.001))  # (steps around, profile tolerance), fewest faces first
FITS = 0.75  # share of vertices that must sit on a ring (or the axis) for the piece to count as turned


def _axes(points: np.ndarray) -> list[np.ndarray]:
    centred = points - points.mean(axis=0)
    found = list(np.linalg.eigh(centred.T @ centred)[1].T) + list(np.eye(3))
    out: list[np.ndarray] = []
    for a in found:
        if all(abs(float(a @ b)) < 0.999 for b in out):
            out.append(a)
    return out


CLUSTER_ROWS = 4_000  # points clustered per slice in `_cluster_steps`


def _cluster(values: np.ndarray, tol: float) -> np.ndarray:
    """Leader clustering of 2D points; returns a cluster id per point."""
    return drain(_cluster_steps(values, tol))


def _cluster_steps(values: np.ndarray, tol: float):
    """Generator: `_cluster`, yielding every CLUSTER_ROWS points."""

    cells: dict[tuple[int, int], list[int]] = {}
    leaders: list[np.ndarray] = []
    ids = np.zeros(len(values), dtype=np.int64)
    for i, p in enumerate(values):
        if i % CLUSTER_ROWS == CLUSTER_ROWS - 1:
            yield
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
    return drain(fit_revolve_steps(points, polys))


def fit_revolve_steps(points: np.ndarray, polys):
    """Generator: `fit_revolve`, in slices of a few thousand points or faces."""

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
        ids = yield from _cluster_steps(rz, tol)
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
    for f, p in enumerate(polys):
        if f % CLUSTER_ROWS == CLUSTER_ROWS - 1:
            yield
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


def facing(points: np.ndarray, polys, source_points: np.ndarray, source_polys) -> float:
    """Above zero if these faces mostly face the way the nearest source faces do, below if they face against them."""

    def frames(p, f):
        p = np.asarray(p, dtype=np.float64)
        c, n = np.zeros((len(f), 3)), np.zeros((len(f), 3))
        flat = f if isinstance(f, FlatFaces) else None
        sizes = np.diff(flat.indptr) if flat is not None else np.fromiter(map(len, f), dtype=np.int64, count=len(f))
        for size in np.unique(sizes).tolist():
            at = np.flatnonzero(sizes == size)
            rows = flat.flat[flat.indptr[at][:, None] + np.arange(size)] if flat is not None else np.asarray([f[k] for k in at.tolist()], dtype=np.int64)
            c[at] = p[rows].mean(axis=1)
            n[at] = np.cross(p[rows[:, 2]] - p[rows[:, 0]], p[rows[:, -1]] - p[rows[:, 1]])  # a quad's diagonals; on a triangle its own normal too
        return c, n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]

    c0, n0 = frames(source_points, source_polys)
    c1, n1 = frames(points, polys)
    pick = np.linspace(0, len(c1) - 1, min(len(c1), 64)).astype(int)
    near = np.asarray([np.argmin(((c1[k] - c0) ** 2).sum(axis=1)) for k in pick.tolist()])  # one by one: all at once wants gigabytes on a big scan
    return float((np.sum(n1[pick] * n0[near], axis=1)).sum())


def orient_like(points: np.ndarray, polys, source_points: np.ndarray, source_polys) -> list[tuple[int, ...]]:
    """Flip the new faces together if most of them face against the nearest source faces."""

    return [tuple(reversed(p)) for p in polys] if facing(points, polys, source_points, source_polys) < 0 else list(polys)


ROUND_SHARE = 0.1  # a turned part whose radius is at least this share of its diagonal is visibly round (a pipe, a barrel, a wheel, a bolt)...
ROUND_SPIN = 16  # ...and gets at least this many steps around, so it still reads as round
WALL_KEEP = 0.7  # a closed profile (a hollow tube, a ring) simplified to less than this share of its area has lost a wall


def min_spin(fit: dict) -> int:
    """Fewest steps around for this turned part: a mostly round one never becomes an 8- or 12-sided prism."""

    radius = float(np.max(np.asarray(fit["profile"])[:, 0])) if len(fit["profile"]) else 0.0
    return ROUND_SPIN if radius >= ROUND_SHARE * fit["diag"] else 0


def profile_holds(fit: dict, tol_share: float) -> bool:
    """False when simplifying a closed profile at this tolerance would fold its inner and outer walls onto each other."""

    if not fit["closed"]:
        return True
    profile = np.asarray(fit["profile"], dtype=np.float64)

    def area(p):
        return 0.5 * abs(float((p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]).sum()))

    full = area(profile)
    simple = _simplify(profile, True, tol_share * fit["diag"])
    return full > 0.0 and len(simple) >= 4 and area(simple) >= WALL_KEEP * full


def sweeps(fit: dict, kind: str = "0", floor: float = 0.0):
    """The turned part at each step count of REVOLVES that keeps it round and its walls apart, fewest faces first.
    `floor`: the smallest profile tolerance, as a share of the diagonal."""

    least = min_spin(fit)
    for spin, tolerance in REVOLVES:
        tolerance = max(tolerance, floor)
        if spin >= least and profile_holds(fit, tolerance):
            yield build_revolve(fit, spin, tolerance, kind)


def rebuilds(points: np.ndarray, polys, kind: str = "0", fit=...):
    """Generator of rebuilt versions of a piece, fewest faces first, as (points, quads, kinds); nothing if it isn't a known part.

    `fit` is what `fit_revolve_steps` returned for this piece; without it, it is made here."""

    if fit is ...:
        fit = fit_revolve(points, polys)
    if fit is not None:
        yield from sweeps(fit, kind)
