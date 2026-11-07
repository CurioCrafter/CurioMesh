# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Seam kernel twin (package K1): source islands, side regions, seam chains, classes, Hard triage, per-side transfer
and the fold fix (SPEC 4.1, 4.2, 4.8).

NumPy only, no bpy. The reference lives in the kernel sources; this twin follows it step for step (same order of
work, same ties, same float operations) so a parity test can compare the two within 1e-6. Long work runs as a
sliceable stage (`Transfer.step()` does about 4 ms) or, when the adapter installed a faster callable, as a background
job the stage polls.

Our own design and code. Textbook pieces: closest point on a triangle (Ericson, "Real-Time Collision Detection",
2005, 5.1.5); a harmonic fill by Gauss-Seidel sweeps (convex-combination fill after Tutte 1963 and Floater 1997, here
with inverse-length weights). The side rules are ours.

Data: `extract` returns a dict with the same keys as the faster callable (weld, island, region, edge_a, edge_b,
edge_chain, edge_face, corner, run_indptr, run_vertex, uv_a, uv_b, gap_px, class, chain_island, chain_material,
chain_length, island_tile, n_welded, n_islands, n_regions, texture_px); pairs are flattened (2 values per row).
"""

from __future__ import annotations

from collections import deque
import math
import time

import numpy as np

# Constants table: same names and values as the top of the matching .rs file (tests/test_uv_scaffold.py compares).
UV_TOL_PX = 0.01
TEXTURE_PX_DEFAULT = 1024.0
IGNORE_PX = 1.0
SHIFT_TOL_PX = 1.0
HARD_ACROSS_EDGES = 3.0
COST_A = 1.0
COST_B = 4.0
HARD_BUDGET_SHARE = 0.5
WALK_REACH = 2.0
WALK_UV_SLACK = 2.0
SLIDE_EDGE = 0.02
CENTRE_WEIGHT = 0.5
DEGENERATE_SHARE = 1e-6  # a fan triangle with 2 x area <= this x its summed squared edges has no area (no say in the vote)
EDGE_ON_COS = 1e-4  # normals this close to 90 deg are edge-on to rounding: no say in the winding vote
F32_TURN = 8.0 / 16777216.0  # cosine noise per unit rounding scale (largest coordinate / shortest edge): 8 float32 ulps
WIND_BALL_EDGE = 0.25  # the winding vote's reach: nearest distance plus this share of a fan triangle's shortest edge
REF_ALIGN_COS = 0.5  # cos 60 deg: a reference face turned further from the result face is swapped (see _read_face)
STOP_TEXEL_SHARE = 0.02
STALL_WINDOW = 50
STALL_DROP = 0.01
UNTANGLE_STALL = 3

IGNORE, SHIFT, SOFT, HARD = 0, 1, 2, 3
SLICE_S = 0.004  # one step() of a stage does about this much work

# Optional faster callables, set by the one adapter (uv_fast.py); the NumPy code runs when a name is missing.
FAST: dict = {}


def set_fast(**callables) -> None:
    """Install (or, with None, remove) faster callables by name."""

    for key, fn in callables.items():
        if fn is None:
            FAST.pop(key, None)
        else:
            FAST[key] = fn


# ---------------------------------------------------------------- small helpers (scalar, same order as the reference)
def _whole(x: float) -> float:
    """Round half away from zero."""

    return math.copysign(math.floor(abs(x) + 0.5), x) if x == x else x


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _dist(a, b):
    d = _sub(a, b)
    return math.sqrt(_dot(d, d))


def _quotient(a, b):
    return 0.0 if b == 0.0 else a / b


def closest_bary(p, a, b, c):
    """Closest point on triangle abc to p as barycentric weights (Ericson 2005; zero divisors give 0)."""

    ab, ac, ap = _sub(b, a), _sub(c, a), _sub(p, a)
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return (1.0, 0.0, 0.0)
    bp = _sub(p, b)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return (0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = _quotient(d1, d1 - d3)
        return (1.0 - v, v, 0.0)
    cp = _sub(p, c)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return (0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = _quotient(d2, d2 - d6)
        return (1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = _quotient(d4 - d3, (d4 - d3) + (d5 - d6))
        return (0.0, 1.0 - w, w)
    under = va + vb + vc
    v, w = _quotient(vb, under), _quotient(vc, under)
    return (1.0 - v - w, v, w)


def _closest_bary_many(p, a, b, c):
    """`closest_bary` for one point against many triangles (a, b, c: (m, 3)); returns (m, 3) weights."""

    m = len(a)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab[:, 0] * ap[:, 0] + ab[:, 1] * ap[:, 1]) + ab[:, 2] * ap[:, 2], (ac[:, 0] * ap[:, 0] + ac[:, 1] * ap[:, 1]) + ac[:, 2] * ap[:, 2]
    bp = p - b
    d3, d4 = (ab[:, 0] * bp[:, 0] + ab[:, 1] * bp[:, 1]) + ab[:, 2] * bp[:, 2], (ac[:, 0] * bp[:, 0] + ac[:, 1] * bp[:, 1]) + ac[:, 2] * bp[:, 2]
    cp = p - c
    d5, d6 = (ab[:, 0] * cp[:, 0] + ab[:, 1] * cp[:, 1]) + ab[:, 2] * cp[:, 2], (ac[:, 0] * cp[:, 0] + ac[:, 1] * cp[:, 1]) + ac[:, 2] * cp[:, 2]
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    w = np.empty((m, 3))
    left = np.ones(m, bool)

    def take(mask, w0, w1, w2):
        nonlocal left
        mask = mask & left
        w[mask, 0], w[mask, 1], w[mask, 2] = w0[mask], w1[mask], w2[mask]
        left = left & ~mask

    def q(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(y == 0.0, 0.0, x / np.where(y == 0.0, 1.0, y))

    one, zero = np.ones(m), np.zeros(m)
    take((d1 <= 0) & (d2 <= 0), one, zero, zero)
    take((d3 >= 0) & (d4 <= d3), zero, one, zero)
    v = q(d1, d1 - d3)
    take((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1.0 - v, v, zero)
    take((d6 >= 0) & (d5 <= d6), zero, zero, one)
    t = q(d2, d2 - d6)
    take((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1.0 - t, zero, t)
    t = q(d4 - d3, (d4 - d3) + (d5 - d6))
    take((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), zero, 1.0 - t, t)
    under = (va + vb) + vc
    v, t = q(vb, under), q(vc, under)
    take(left, (1.0 - v) - t, v, t)
    return w


def _closest_point_many(p, a, b, c):
    """The closest point of each triangle to p, the way the face tree computes it (for the global lookup)."""

    m = len(a)
    ab, ac, ap = b - a, c - a, p - a
    dot = lambda x, y: (x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]) + x[:, 2] * y[:, 2]  # noqa: E731
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    out = np.empty((m, 3))
    left = np.ones(m, bool)

    def take(mask, pts):
        nonlocal left
        mask = mask & left
        out[mask] = pts[mask]
        left = left & ~mask

    with np.errstate(divide="ignore", invalid="ignore"):
        take((d1 <= 0) & (d2 <= 0), a)
        take((d3 >= 0) & (d4 <= d3), b)
        take((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / (d1 - d3))[:, None] * ab)
        take((d6 >= 0) & (d5 <= d6), c)
        take((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / (d2 - d6))[:, None] * ac)
        bc = c - b
        take((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None] * bc)
        under = (va + vb) + vc
        take(under == 0.0, a)
        v, w = vb / under, vc / under
        take(left, (a + ab * v[:, None]) + ac * w[:, None])
    return out


def _plane_bary(p, a, b, c):
    e1, e2, d = _sub(b, a), _sub(c, a), _sub(p, a)
    g11, g12, g22 = _dot(e1, e1), _dot(e1, e2), _dot(e2, e2)
    det = g11 * g22 - g12 * g12
    if det <= 1e-30 * max(g11 * g22, 1e-300):
        return None
    r1, r2 = _dot(d, e1), _dot(d, e2)
    s = (g22 * r1 - g12 * r2) / det
    t = (g11 * r2 - g12 * r1) / det
    return (1.0 - s - t, s, t)


def _mix(w, a, b, c):
    return tuple(w[0] * a[k] + w[1] * b[k] + w[2] * c[k] for k in range(3))


def _newell(P, face):
    n0 = n1 = n2 = 0.0
    m = len(face)
    for i in range(m):
        a, b = P[face[i]], P[face[(i + 1) % m]]
        n0 += (a[1] - b[1]) * (a[2] + b[2])
        n1 += (a[2] - b[2]) * (a[0] + b[0])
        n2 += (a[0] - b[0]) * (a[1] + b[1])
    return (n0, n1, n2)


def _min_labels(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected components of n items joined by pairs (a[k], b[k]): each item's label = the lowest item in its group."""

    label = np.arange(n, dtype=np.int64)
    if len(a) == 0:
        return label
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    while True:
        low = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]
        while True:
            jump = new[new]
            if np.array_equal(jump, new):
                break
            new = jump
        if np.array_equal(new, label):
            return label
        label = new


def _first_order(keys: np.ndarray) -> tuple:
    """Ids by first appearance of each key: (id per item, count)."""

    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[inverse.ravel()], len(first)


def _weld(points: np.ndarray) -> tuple:
    p = np.ascontiguousarray(points.reshape(-1, 3) + 0.0)
    if len(p) == 0:
        return np.zeros(0, np.int64), 0
    keys = p.view(np.dtype((np.void, 24))).ravel()
    return _first_order(keys)


def _next_corners(indptr: np.ndarray) -> np.ndarray:
    n = int(indptr[-1])
    nxt = np.arange(1, n + 1, dtype=np.int64)
    ends = indptr[1:][indptr[1:] > indptr[:-1]]
    starts = indptr[:-1][indptr[1:] > indptr[:-1]]
    nxt[ends - 1] = starts
    return nxt


def _face_of_corner(indptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))


def _edges(wflat: np.ndarray, nxt: np.ndarray) -> tuple:
    """Undirected edges in order of first corner: (a, b, start, occ) with occ[start[e]:start[e+1]] the corners (in
    corner order) that use edge e."""

    frm, to = wflat, wflat[nxt]
    lo, hi = np.minimum(frm, to), np.maximum(frm, to)
    corner = np.arange(len(wflat), dtype=np.int64)
    order = np.lexsort((corner, hi, lo))
    slo, shi = lo[order], hi[order]
    new = np.ones(len(order), bool)
    new[1:] = (slo[1:] != slo[:-1]) | (shi[1:] != shi[:-1])
    gstart = np.flatnonzero(new)
    gend = np.r_[gstart[1:], len(order)]
    first = order[gstart]
    rank = np.argsort(first, kind="stable")
    a, b = slo[gstart][rank], shi[gstart][rank]
    sizes = (gend - gstart)[rank]
    start = np.r_[0, np.cumsum(sizes)]
    occ = np.concatenate([order[gstart[g]:gend[g]] for g in rank]) if len(rank) else np.zeros(0, np.int64)
    return a, b, start, occ


# ---------------------------------------------------------------- extract, classify, regions
def extract(points, indptr, flat, corner_uv, material=None, repeat=(), texture_px: float = TEXTURE_PX_DEFAULT, uv_tol_px: float = UV_TOL_PX,
            cut_here=(), use_fast: bool = True) -> dict:
    """Islands, seam edges, chains, classes and side regions of a source mesh. See the module docstring for the keys."""

    fast = FAST.get("extract") if use_fast else None
    if fast is not None:
        job = fast(points, indptr, flat, corner_uv, () if material is None else material, repeat, texture_px, uv_tol_px, cut_here)
        if job is not None:
            job._thread.join()
            return job.result()
    seams = {}
    for _ in _extract_steps(points, indptr, flat, corner_uv, material, repeat, texture_px, uv_tol_px, cut_here, seams):
        pass
    return seams


def _extract_steps(points, indptr, flat, corner_uv, material, repeat, texture_px, uv_tol_px, cut_here, seams: dict):
    """Generator behind `extract`: fills `seams`, yielding between pieces of work."""

    P = np.asarray(points, np.float64).reshape(-1, 3)
    indptr, flat = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
    uv = np.asarray(corner_uv, np.float64).reshape(-1, 2)
    nf = len(indptr) - 1
    mat = np.zeros(nf, np.int64) if material is None or len(material) == 0 else np.asarray(material, np.int64)
    tex = float(texture_px) if texture_px > 0 else TEXTURE_PX_DEFAULT
    tol = max(float(uv_tol_px), 0.0) / tex
    weld, n_welded = _weld(P)
    wflat = weld[flat]
    nxt = _next_corners(indptr)
    cface = _face_of_corner(indptr)
    ea, eb, estart, occ = _edges(wflat, nxt)
    ne_all = len(ea)

    def uv_at(h, w):
        return uv[h] if wflat[h] == w else uv[nxt[h]]

    def same(x, y):
        d0, d1 = x[0] - y[0], x[1] - y[1]
        return math.sqrt(d0 * d0 + d1 * d1) <= tol

    def joined(h, g, a, b):
        return mat[cface[h]] == mat[cface[g]] and same(uv_at(h, a), uv_at(g, a)) and same(uv_at(h, b), uv_at(g, b))

    # Vectorised for the common case (two uses), then the rest one by one.
    counts = np.diff(estart)
    two = np.flatnonzero(counts == 2)
    h2, g2 = occ[estart[two]], occ[estart[two] + 1]
    a2, b2 = ea[two], eb[two]
    diff_face = cface[h2] != cface[g2]

    def at_many(h, w):
        return np.where((wflat[h] == w)[:, None], uv[h], uv[nxt[h]])

    def same_many(x, y):
        d = x - y
        return np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) <= tol

    ok2 = diff_face & (mat[cface[h2]] == mat[cface[g2]]) & same_many(at_many(h2, a2), at_many(g2, a2)) & same_many(at_many(h2, b2), at_many(g2, b2))
    join_a, join_b = [cface[h2[ok2]]], [cface[g2[ok2]]]
    is_seam = np.zeros(ne_all, bool)
    is_seam[two[diff_face & ~ok2]] = True
    odd = np.zeros(n_welded, bool)
    other = np.flatnonzero((counts != 2) | np.isin(np.arange(ne_all), two[~diff_face]))
    yield
    for count, e in enumerate(other):
        if count % 256 == 255:
            yield
        uses = occ[estart[e]:estart[e + 1]]
        a, b = ea[e], eb[e]
        odd[a] = odd[b] = True
        for k in range(1, len(uses)):
            h, g = uses[k - 1], uses[k]
            if cface[h] != cface[g] and joined(h, g, a, b):
                join_a.append(np.array([cface[h]]))
                join_b.append(np.array([cface[g]]))
    root = _min_labels(nf, np.concatenate(join_a), np.concatenate(join_b))
    island, n_islands = _first_order(root)

    seam_edges = np.flatnonzero(is_seam)
    ne = len(seam_edges)
    sa, sb = ea[seam_edges], eb[seam_edges]
    valence = np.bincount(np.r_[sa, sb], minlength=n_welded).astype(np.int64)
    yield
    adj = [[] for _ in range(n_welded)]
    for k in range(ne):
        adj[sa[k]].append((int(sb[k]), k))
        adj[sb[k]].append((int(sa[k]), k))
    for lst in adj:
        lst.sort()
    su = occ[estart[seam_edges]]
    sg = occ[estart[seam_edges] + 1]

    def key_of(k):
        f, g = cface[su[k]], cface[sg[k]]
        i0, i1 = island[f], island[g]
        m0, m1 = mat[f], mat[g]
        return (min(i0, i1), max(i0, i1), min(m0, m1), max(m0, m1))

    corner = odd | ((valence != 2) & (valence > 0))
    for v in np.flatnonzero((valence == 2) & ~corner):
        if key_of(adj[v][0][1]) != key_of(adj[v][1][1]):
            corner[v] = True

    visited = np.zeros(ne, bool)
    runs = []

    def walk(start, first):
        run, used = [start], []
        cur, k = start, first
        while True:
            visited[k] = True
            used.append(k)
            nxt_v = int(sb[k]) if sa[k] == cur else int(sa[k])
            run.append(nxt_v)
            if corner[nxt_v] or nxt_v == start:
                break
            found = next((j for _, j in adj[nxt_v] if j != k and not visited[j]), None)
            if found is None:
                break
            cur, k = nxt_v, found
        return run, used

    for v in np.flatnonzero(corner):
        for _, k in adj[v]:
            if not visited[k]:
                runs.append(walk(int(v), k))
                yield
    for v in range(n_welded):
        for _, k in adj[v]:
            if not visited[k]:
                runs.append(walk(v, k))
                yield

    wpos = np.zeros((n_welded, 3))
    wpos[weld] = P  # the last point of each welded id, as the reference
    edge_face = np.zeros((ne, 2), np.int64)
    edge_chain = np.full(ne, -1, np.int64)
    run_indptr, run_vertex, uv_a, uv_b, gap_px = [0], [], [], [], []
    chain_island, chain_material, chain_length = [], [], []
    for c, (run, used) in enumerate(runs):
        yield
        side_a = None
        faces_ab = []
        for j, k in enumerate(used):
            h, g = int(su[k]), int(sg[k])
            frm, to = run[j], run[j + 1]

            def along(x):
                return wflat[x] == frm and wflat[nxt[x]] == to

            fh, fg = cface[h], cface[g]
            ah, ag = along(h), along(g)
            if ah and not ag:
                pick_h = True
            elif ag and not ah:
                pick_h = False
            elif side_a is not None and island[fh] != island[fg]:
                pick_h = island[fh] == side_a
            else:
                pick_h = fh <= fg
            ha, hb = (h, g) if pick_h else (g, h)
            if side_a is None:
                side_a = island[cface[ha]]
            faces_ab.append((ha, hb))
            edge_chain[k] = c
            edge_face[k] = (cface[ha], cface[hb])
        length = 0.0
        for j, w in enumerate(run):
            ha, hb = faces_ab[min(j, len(used) - 1)]
            x, y = uv_at(ha, w), uv_at(hb, w)
            run_vertex.append(w)
            uv_a.extend((float(x[0]), float(x[1])))
            uv_b.extend((float(y[0]), float(y[1])))
            d0, d1 = x[0] - y[0], x[1] - y[1]
            gap_px.append(math.sqrt(d0 * d0 + d1 * d1) * tex)
            if j > 0:
                length += _dist(wpos[run[j - 1]], wpos[w])
        run_indptr.append(len(run_vertex))
        fa, fb = edge_face[used[0]]
        chain_island.append((island[fa], island[fb]))
        chain_material.append((mat[fa], mat[fb]))
        chain_length.append(length)
    lo = np.full((n_islands, 2), np.inf)
    hi = np.full((n_islands, 2), -np.inf)
    ci = island[cface]
    np.minimum.at(lo, ci, uv)
    np.maximum.at(hi, ci, uv)
    tile = np.floor(0.5 * (lo + hi)).astype(np.int64) if n_islands else np.zeros((0, 2), np.int64)
    seams.update(
        texture_px=tex, weld=weld, n_welded=n_welded, island=island, n_islands=n_islands,
        edge_a=sa.astype(np.int64), edge_b=sb.astype(np.int64), edge_chain=edge_chain, edge_face=edge_face.ravel(), corner=corner,
        run_indptr=np.array(run_indptr, np.int64), run_vertex=np.array(run_vertex, np.int64),
        uv_a=np.array(uv_a, np.float64), uv_b=np.array(uv_b, np.float64), gap_px=np.array(gap_px, np.float64),
        chain_island=np.array(chain_island, np.int64).ravel(), chain_material=np.array(chain_material, np.int64).ravel(),
        chain_length=np.array(chain_length, np.float64), island_tile=tile.ravel(),
    )
    classify(seams, repeat, cut_here)


def classify(seams: dict, repeat=(), cut_here=()) -> None:
    """Chain classes (SPEC 4.1) and the side regions they give, in place. `cut_here` lists chains the user made Hard."""

    tex = seams["texture_px"]
    repeat = [bool(x) for x in repeat]
    cut = {int(c) for c in cut_here}

    def repeats(m):
        return m < 0 or m >= len(repeat) or repeat[m]

    ptr, ua, ub, gap = seams["run_indptr"], seams["uv_a"], seams["uv_b"], seams["gap_px"]
    isl, mats, tile = seams["chain_island"].reshape(-1, 2), seams["chain_material"].reshape(-1, 2), seams["island_tile"].reshape(-1, 2)
    out = []
    for c in range(len(ptr) - 1):
        s, e = ptr[c], ptr[c + 1]
        ma, mb = mats[c]
        ia, ib = isl[c]
        g = max([0.0] + [float(x) for x in gap[s:e]])
        if ma != mb or c in cut:
            cls = HARD
        elif g <= IGNORE_PX:
            cls = IGNORE
        else:
            n = float(e - s)
            du = np.cumsum(ub[2 * s:2 * e:2] - ua[2 * s:2 * e:2])[-1] / n
            dv = np.cumsum(ub[2 * s + 1:2 * e:2] - ua[2 * s + 1:2 * e:2])[-1] / n
            shift = (_whole(float(du)), _whole(float(dv)))
            fits = True
            for j in range(s, e):
                x = ub[2 * j] - ua[2 * j] - shift[0]
                y = ub[2 * j + 1] - ua[2 * j + 1] - shift[1]
                if math.sqrt(x * x + y * y) * tex > SHIFT_TOL_PX:
                    fits = False
                    break
            other_tile = tuple(tile[ia]) != tuple(tile[ib])
            if fits and shift != (0.0, 0.0) and repeats(ma):
                cls = SHIFT
            elif other_tile and not repeats(ma):
                cls = HARD
            else:
                cls = SOFT
        out.append(cls)
    seams["class"] = np.array(out, np.int64)
    regions(seams)


def regions(seams: dict) -> None:
    """Side regions (islands joined across Ignore and Shift chains), in place."""

    isl = seams["chain_island"].reshape(-1, 2)
    keep = (seams["class"] == IGNORE) | (seams["class"] == SHIFT)
    root = _min_labels(seams["n_islands"], isl[keep, 0], isl[keep, 1])
    seams["region"], seams["n_regions"] = _first_order(root[seams["island"]]) if len(seams["island"]) else (np.zeros(0, np.int64), 0)


def set_classes(seams: dict, classes) -> None:
    seams["class"] = np.asarray(classes, np.int64).copy()
    regions(seams)


def join_tolerance(seams: dict) -> np.ndarray:
    """Per side region: half the smallest median gap of its self-seams (UV units), infinite without one."""

    join = np.full(seams["n_regions"], np.inf)
    roi = np.zeros(seams["n_islands"], np.int64)
    roi[seams["island"]] = seams["region"]
    isl, ptr = seams["chain_island"].reshape(-1, 2), seams["run_indptr"]
    for c in range(len(ptr) - 1):
        ra, rb = roi[isl[c, 0]], roi[isl[c, 1]]
        if ra != rb or seams["class"][c] < SOFT:
            continue
        gaps = np.sort(seams["gap_px"][ptr[c]:ptr[c + 1]])
        tol = max(0.5 * float(gaps[len(gaps) // 2]), IGNORE_PX) / seams["texture_px"]
        join[ra] = min(join[ra], tol)
    return join


# ---------------------------------------------------------------- triage
def triage(seams: dict, points, indptr, flat, f_target: float, allowance: float, seen_face=None, cost_a: float = COST_A, cost_b: float = COST_B) -> dict:
    """Pre-search Hard choice (SPEC 4.2 step 1), in place on `seams` (Soft chains of chosen islands become Hard)."""

    P = np.asarray(points, np.float64).reshape(-1, 3)
    indptr, flat = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
    ni, nf = seams["n_islands"], len(indptr) - 1
    island = seams["island"]
    area, seen_area, perim = np.zeros(ni), np.zeros(ni), np.zeros(ni)
    lo, hi = np.full((ni, 3), np.inf), np.full((ni, 3), -np.inf)
    total = 0.0
    for f in range(nf):
        face = flat[indptr[f]:indptr[f + 1]]
        i = island[f]
        n = _newell(P, face)
        a = 0.5 * math.sqrt(_dot(n, n))
        area[i] += a
        seen_area[i] += a * (1.0 if seen_face is None or len(seen_face) == 0 else float(seen_face[f]))
        total += a
        lo[i] = np.minimum(lo[i], P[face].min(0))
        hi[i] = np.maximum(hi[i], P[face].max(0))
    weld = seams["weld"]
    wflat = weld[flat]
    nxt = _next_corners(indptr)
    cface = _face_of_corner(indptr)
    ea, eb, estart, occ = _edges(wflat, nxt)
    wpos = np.zeros((seams["n_welded"], 3))
    wpos[weld] = P
    for e in np.flatnonzero(np.diff(estart) == 1):
        perim[island[cface[occ[estart[e]]]]] += _dist(wpos[ea[e]], wpos[eb[e]])
    ef = seams["edge_face"].reshape(-1, 2)
    for k in range(len(seams["edge_a"])):
        ln = _dist(wpos[seams["edge_a"][k]], wpos[seams["edge_b"][k]])
        for f in ef[k]:
            perim[island[f]] += ln
    gap, soft = np.zeros(ni), np.zeros(ni, bool)
    ptr, isl = seams["run_indptr"], seams["chain_island"].reshape(-1, 2)
    for c in range(len(ptr) - 1):
        if seams["class"][c] == SOFT:
            g = max([0.0] + [float(x) for x in seams["gap_px"][ptr[c]:ptr[c + 1]]])
            for i in isl[c]:
                gap[i] = max(gap[i], g)
                soft[i] = True
    edge_est = math.sqrt(total / f_target) if f_target > 0 else 0.0
    ranked = []
    for i in range(ni):
        if not soft[i] or perim[i] <= 0.0 or edge_est <= 0.0:
            continue
        diag = _dist(lo[i], hi[i])
        across = min(diag, 4.0 * area[i] / perim[i])
        if across / edge_est < HARD_ACROSS_EDGES:
            seen = seen_area[i] / area[i] if area[i] > 0 else 1.0
            ranked.append((area[i] * gap[i] * seen, i, cost_a * perim[i] / edge_est + cost_b))
    ranked.sort(key=lambda r: (-r[0], r[1]))
    budget = HARD_BUDGET_SHARE * max(allowance, 0.0) * max(f_target, 0.0)
    out = dict(hard_islands=[], hard_cost=[], left_islands=[], left_cost=[], spent=0.0, budget=budget, edge_est=edge_est)
    hard = np.zeros(ni, bool)
    for _, i, cost in ranked:
        if out["spent"] + cost <= budget:
            out["spent"] += cost
            hard[i] = True
            out["hard_islands"].append(i)
            out["hard_cost"].append(cost)
        else:
            out["left_islands"].append(i)
            out["left_cost"].append(cost)
    cls = seams["class"]
    for c in range(len(cls)):
        if cls[c] == SOFT and (hard[isl[c, 0]] or hard[isl[c, 1]]):
            cls[c] = HARD
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in out.items()}


# ---------------------------------------------------------------- lookup (per-side transfer)
class _Grid:
    """Uniform grid over triangles for the global nearest-face lookup (single-precision corners, like the face tree;
    ties go to the lowest face)."""

    def __init__(self, a, b, c, owner):
        self.a, self.b, self.c, self.owner = a, b, c, owner
        lo = np.minimum(np.minimum(a, b), c)
        hi = np.maximum(np.maximum(a, b), c)
        self.origin = lo.min(0) if len(a) else np.zeros(3)
        ext = (hi - lo).max(1) if len(a) else np.ones(1)
        span = float((hi.max(0) - self.origin).max()) if len(a) else 1.0
        self.h = max(float(np.median(ext)) * 2.0, span * 1e-6, 1e-12)
        il = np.floor((lo - self.origin) / self.h).astype(np.int64)
        ih = np.floor((hi - self.origin) / self.h).astype(np.int64)
        cells, tris = [], []
        n = ih - il + 1
        for dx in range(int(n[:, 0].max()) if len(a) else 0):
            for dy in range(int(n[:, 1].max())):
                for dz in range(int(n[:, 2].max())):
                    m = (dx < n[:, 0]) & (dy < n[:, 1]) & (dz < n[:, 2])
                    t = np.flatnonzero(m)
                    cells.append(il[t] + (dx, dy, dz))
                    tris.append(t)
        self.cells = {}
        self._pending = (np.concatenate(cells), np.concatenate(tris)) if cells else None
        self.lo_cell = il.min(0) if len(a) else np.zeros(3, np.int64)
        self.hi_cell = ih.max(0) if len(a) else np.zeros(3, np.int64)

    def fill(self):
        """Generator: puts the triangles into their cells (yields every few thousand)."""

        if self._pending is None:
            return
        cells, tris = self._pending
        self._pending = None
        order = np.lexsort((tris, cells[:, 2], cells[:, 1], cells[:, 0]))
        cells, tris = cells[order].tolist(), tris[order].tolist()
        for k, (t, cell) in enumerate(zip(tris, cells)):
            self.cells.setdefault(tuple(cell), []).append(t)
            if k % 4096 == 4095:
                yield

    def nearest_many(self, qs) -> list:
        """`nearest` for several points: one pass over the cells next to all of them, then the ring search only for
        the points that pass needs more."""

        if not self.cells:
            return [-1] * len(qs)
        qs = np.asarray(qs, np.float64)
        qc = np.floor((qs - self.origin) / self.h).astype(np.int64)
        found = set()
        for cell in {tuple(x) for x in qc.tolist()}:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        lst = self.cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz))
                        if lst:
                            found.update(lst)
        out = [-1] * len(qs)
        if not found:
            return [self.nearest(q) for q in qs]
        t = np.array(sorted(found), np.int64)
        m, k = len(qs), len(t)
        p = _closest_point_many(np.repeat(qs, k, 0), np.tile(self.a[t], (m, 1)), np.tile(self.b[t], (m, 1)), np.tile(self.c[t], (m, 1)))
        d = p - np.repeat(qs, k, 0)
        d2 = ((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]).reshape(m, k)
        owner = self.owner[t]
        for i in range(m):
            best = d2[i].min()
            if math.sqrt(best) < self.h:  # nothing outside the searched cells can be closer
                out[i] = int(owner[d2[i] == best].min())
            else:
                out[i] = self.nearest(qs[i])
        return out

    def nearest(self, q) -> int:
        """Face of the nearest triangle to q (lowest face on ties), or -1 when empty."""

        if not self.cells:
            return -1
        qc = np.floor((np.asarray(q) - self.origin) / self.h).astype(np.int64)
        best_d, best_f = math.inf, -1
        r = 0
        reach = int(max(np.max(np.abs(qc - self.lo_cell)), np.max(np.abs(qc - self.hi_cell)))) + 1
        while r <= reach:
            found = []
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != r:
                            continue
                        lst = self.cells.get((qc[0] + dx, qc[1] + dy, qc[2] + dz))
                        if lst:
                            found.extend(lst)
            if found:
                t = np.unique(np.array(found, np.int64))
                p = _closest_point_many(np.asarray(q, np.float64), self.a[t], self.b[t], self.c[t])
                d = p - q
                d2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
                for k in np.flatnonzero(d2 <= best_d):
                    if d2[k] < best_d or (d2[k] == best_d and self.owner[t[k]] < best_f):
                        best_d, best_f = float(d2[k]), int(self.owner[t[k]])
            if best_f >= 0 and math.sqrt(best_d) < r * self.h:
                break
            r += 1
        return best_f


def _cut(P, face):
    """Triangles of a polygon the way the face tree cuts it (3 and 4 corners; a fan above that)."""

    n = len(face)
    if n == 3:
        return [(0, 1, 2)]
    if n == 4:
        a, b, c, d = (P[v] for v in face)

        def cross(x, y):
            return (x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0])

        facing = cross(_sub(c, a), _sub(d, b))
        inward = _dot(cross(_sub(a, d), _sub(b, a)), facing) < 0.0 or _dot(cross(_sub(c, b), _sub(d, c)), facing) < 0.0
        return [(0, 1, 2), (2, 3, 0)] if inward else [(3, 0, 1), (1, 2, 3)]
    return [(0, k, k + 1) for k in range(1, n - 1)]


class _Source:
    def __init__(self, P, indptr, flat, seams, uv=None):
        self.P, self.indptr, self.flat, self.seams = P, indptr, flat, seams
        self.uv = None if uv is None or len(uv) == 0 else np.asarray(uv, np.float64).reshape(-1, 2)

    def build(self):
        """Generator: the fan triangles, centres, normals, crossings and the global lookup grid."""

        P, indptr, flat, seams = self.P, self.indptr, self.flat, self.seams
        nf = len(indptr) - 1
        tri, tri_ptr = [], [0]
        for f in range(nf):
            s, e = int(indptr[f]), int(indptr[f + 1])
            tri.extend((s, k, k + 1) for k in range(s + 1, e - 1))
            tri_ptr.append(len(tri))
            if f % 2048 == 2047:
                yield
        self.tri = np.array(tri, np.int64).reshape(-1, 3)
        self.tri_ptr = np.array(tri_ptr, np.int64)
        self.centre, self.normal = [], []
        Pl = P.tolist()
        for f in range(nf):
            face = flat[indptr[f]:indptr[f + 1]].tolist()
            c0 = c1 = c2 = 0.0
            for v in face:
                c0 += Pl[v][0] / len(face)
                c1 += Pl[v][1] / len(face)
                c2 += Pl[v][2] / len(face)
            self.centre.append((c0, c1, c2))
            self.normal.append(_newell(Pl, face))
            if f % 1024 == 1023:
                yield
        self.cp = P[flat]  # corner points
        if self.uv is not None:  # per face: UV centre and 3D radius, for the walk's UV check
            self.uv_centre, self.radius = [], []
            Ul = self.uv.tolist()
            for f in range(nf):
                s, e = int(indptr[f]), int(indptr[f + 1])
                u0 = u1 = 0.0
                for h in range(s, e):
                    u0 += Ul[h][0] / (e - s)
                    u1 += Ul[h][1] / (e - s)
                self.uv_centre.append((u0, u1))
                c = self.centre[f]
                r = 0.0
                for v in flat[s:e].tolist():
                    r = max(r, _dist(Pl[v], c))
                self.radius.append(r)
        weld = seams["weld"]
        wflat = weld[flat]
        # faces at each welded vertex, in face order (the winding vote's ball grows through them)
        self.wflat = wflat
        order = np.argsort(wflat, kind="stable")
        self.vf = _face_of_corner(indptr)[order]
        self.vf_ptr = np.concatenate([[0], np.cumsum(np.bincount(wflat, minlength=int(seams["n_welded"])))])
        nxt = _next_corners(indptr)
        cface = _face_of_corner(indptr)
        ea, eb, estart, occ = _edges(wflat, nxt)
        cls_of = {}
        cls = seams["class"]
        for k in range(len(seams["edge_a"])):
            a, b = int(seams["edge_a"][k]), int(seams["edge_b"][k])
            cls_of[(min(a, b), max(a, b))] = int(cls[seams["edge_chain"][k]])
        self.across = np.full(len(flat), -1, np.int64)
        self.crossing = np.full(len(flat), 2, np.int64)
        two = np.flatnonzero(np.diff(estart) == 2)
        yield
        for count, e in enumerate(two):
            if count % 2048 == 2047:
                yield
            h, g = int(occ[estart[e]]), int(occ[estart[e] + 1])
            if cface[h] == cface[g]:
                continue
            c = cls_of.get((int(ea[e]), int(eb[e])))
            kind = 0 if c is None or c == IGNORE else (1 if c == SHIFT else 2)
            self.across[h], self.across[g] = cface[g], cface[h]
            self.crossing[h] = self.crossing[g] = kind
        # Global lookup: the tree's triangles, single-precision corners.
        P32 = P.astype(np.float32).astype(np.float64)
        ga, gb, gc, go = [], [], [], []
        for f in range(nf):
            face = flat[indptr[f]:indptr[f + 1]]
            for t in _cut(P, face):
                ga.append(P32[face[t[0]]])
                gb.append(P32[face[t[1]]])
                gc.append(P32[face[t[2]]])
                go.append(f)
            if f % 1024 == 1023:
                yield
        self.grid = _Grid(np.array(ga).reshape(-1, 3), np.array(gb).reshape(-1, 3), np.array(gc).reshape(-1, 3), np.array(go, np.int64))
        yield
        yield from self.grid.fill()
        self.tri_a, self.tri_b, self.tri_c = self.cp[self.tri[:, 0]], self.cp[self.tri[:, 1]], self.cp[self.tri[:, 2]]

    def on_each(self, face_lists, qs):
        """`on_faces` for several points at once (point i against `face_lists[i]`), in one vectorised pass."""

        tl, owner_index, row = [], [], []
        for i, faces in enumerate(face_lists):
            faces = np.asarray(faces, np.int64)
            first, count = self.tri_ptr[faces], self.tri_ptr[faces + 1] - self.tri_ptr[faces]
            total = int(count.sum())
            before = np.cumsum(count) - count
            tl.append(np.arange(total) - np.repeat(before, count) + np.repeat(first, count))
            owner_index.append(np.repeat(np.arange(len(faces)), count))
            row.append(np.full(total, i))
        t, which, row = np.concatenate(tl), np.concatenate(owner_index), np.concatenate(row)
        q = np.asarray(qs, np.float64)[row]
        pa, pb, pc = self.tri_a[t], self.tri_b[t], self.tri_c[t]
        w = _closest_bary_many(q, pa, pb, pc)
        pt = (w[:, 0:1] * pa + w[:, 1:2] * pb) + w[:, 2:3] * pc
        d = pt - q
        d2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
        out = []
        starts = np.searchsorted(row, np.arange(len(face_lists) + 1))
        for i in range(len(face_lists)):
            a, b = starts[i], starts[i + 1]
            k = a + int(np.argmin(d2[a:b]))
            out.append((float(d2[k]), int(t[k]), tuple(float(x) for x in w[k]), int(which[k])))
        return out

    def on_faces(self, faces, q):
        """Closest point over the fan triangles of `faces` (in order): (squared distance, triangle, weights, index in
        `faces`), first minimum wins."""

        lens = self.tri_ptr[np.asarray(faces) + 1] - self.tri_ptr[np.asarray(faces)]
        t = np.concatenate([np.arange(self.tri_ptr[f], self.tri_ptr[f + 1]) for f in faces])
        w = _closest_bary_many(np.asarray(q, np.float64), self.tri_a[t], self.tri_b[t], self.tri_c[t])
        pa, pb, pc = self.tri_a[t], self.tri_b[t], self.tri_c[t]
        pt = (w[:, 0:1] * pa + w[:, 1:2] * pb) + w[:, 2:3] * pc
        d = pt - q
        d2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
        k = int(np.argmin(d2))
        which = int(np.searchsorted(np.cumsum(lens), k, side="right"))
        return float(d2[k]), int(t[k]), tuple(float(x) for x in w[k]), which


def _walk(src: _Source, rf, rt, rw, c, reach):
    """The source faces a result face reads from: from the reference face `rf` (hit at triangle `rt`, weights `rw`)
    across free and Shift edges only, within `reach` of the face centre `c`, passing the walk's UV check (with a layer).
    Breadth-first, each with whether a Shift chain was crossed on the way."""

    # The walk's UV check: a face whose UV offset from the reference UV, taken back to 3D through the reference
    # triangle's map, is much longer than its 3D distance lies on the far side of a seam reached the long way round (a
    # thin tube or plate inside one island); it is not walked.
    uv_reach = None
    if src.uv is not None:
        tri = src.tri[rt]
        p0, p1, p2 = (tuple(src.cp[tri[k]]) for k in range(3))
        q0, q1, q2 = (tuple(src.uv[tri[k]]) for k in range(3))
        u_ref = (rw[0] * q0[0] + rw[1] * q1[0] + rw[2] * q2[0], rw[0] * q0[1] + rw[1] * q1[1] + rw[2] * q2[1])
        p_ref = _mix(rw, p0, p1, p2)
        inverse = _uv_to_3d(p0, p1, p2, q0, q1, q2)
        if inverse is not None:
            uv_reach = (u_ref, p_ref, inverse)
    order = [(rf, False)]
    seen = {rf}
    queue = deque([0])
    indptr = src.indptr
    while queue:
        f, shifted = order[queue.popleft()]
        for h in range(int(indptr[f]), int(indptr[f + 1])):
            g = int(src.across[h])
            if g < 0 or src.crossing[h] == 2 or g in seen or _dist(src.centre[g], c) > reach:
                continue
            sh = shifted or src.crossing[h] == 1
            if uv_reach is not None and not _uv_near(src, g, sh, uv_reach):
                continue
            seen.add(g)
            order.append((g, sh))
            queue.append(len(order) - 1)
    return order


def _sign(x: float) -> float:
    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else (x if x != x else 0.0))


def _uv_area(uv, indptr, f) -> float:
    """Twice the signed UV area of source face f (shoelace over its corners; `uv` is 2 per source corner)."""

    s, e = int(indptr[f]), int(indptr[f + 1])
    area = 0.0
    for i in range(s, e):
        j = s if i + 1 == e else i + 1
        area += float(uv[i][0]) * float(uv[j][1]) - float(uv[j][0]) * float(uv[i][1])
    return area


def _rounding_scale(pts) -> float:
    """How far single-precision rounding of a face's corners (`pts`) can turn its normal: largest coordinate over
    shortest edge."""

    big, short = 0.0, math.inf
    for i, p in enumerate(pts):
        big = max(big, abs(p[0]), abs(p[1]), abs(p[2]))
        short = min(short, _dist(p, pts[(i + 1) % len(pts)]))
    return big / short if short > 0.0 else math.inf


def _facing(a, b, scale) -> float:
    """-1, 0 or 1: whether normals a and b face the same way, 0 when edge-on to rounding (cosine at most EDGE_ON_COS or
    F32_TURN times the two faces' rounding scales)."""

    d = _dot(a, b)
    return 0.0 if abs(d) <= max(EDGE_ON_COS, F32_TURN * scale) * math.sqrt(_dot(a, a)) * math.sqrt(_dot(b, b)) else _sign(d)


def _has_area(src, g) -> bool:
    """True when source face g has area (twice-area above DEGENERATE_SHARE of its summed squared edges)."""

    s, e = int(src.indptr[g]), int(src.indptr[g + 1])
    pts = [tuple(src.P[v]) for v in src.flat[s:e]]
    e2 = 0.0
    for i in range(len(pts)):
        d = _sub(pts[(i + 1) % len(pts)], pts[i])
        e2 += _dot(d, d)
    n = src.normal[g]
    return math.sqrt(_dot(n, n)) > DEGENERATE_SHARE * e2


def _fan_orient(src, Pr, face, rf, fallback) -> float:
    """A result face's orientation sign against its reference face `rf`, from the source under each of its fan
    triangles. Each triangle (0, i, i+1) gathers the source faces within reach of its centre (the nearest one's distance
    plus WIND_BALL_EDGE of its shortest edge, joined through shared vertices) and the windings they would show a texture with: a
    face's UV winding times whether it faces the triangle, relative to the reference's UV winding (no UV area: both
    allowed; edge-on or no 3D area: neither). The face takes the winding every triangle allows; both throughout: `fallback`. No common
    winding (folded in 3D, or across a crease with one half leaning into a wall): 0 (uvkeep cuts it into its fan
    triangles); no area at all (DEGENERATE_SHARE): 0 too, so the fold fix leaves its reads alone."""

    ref_wind = 1.0 if src.uv is None else _sign(_uv_area(src.uv, src.indptr, rf))
    if ref_wind == 0.0:
        return fallback
    tris = [[face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1)]
    qs = []
    for t in tris:
        a, b, cc = Pr[t[0]], Pr[t[1]], Pr[t[2]]
        qs.append(((a[0] + b[0] + cc[0]) / 3.0, (a[1] + b[1] + cc[1]) / 3.0, (a[2] + b[2] + cc[2]) / 3.0))
    near = src.grid.nearest_many(qs) if qs else []
    common = 3  # bit 0: +1 allowed, bit 1: -1 allowed
    said = False
    for t, q, g0 in zip(tris, qs, near):
        a, b, cc = Pr[t[0]], Pr[t[1]], Pr[t[2]]
        nt = _newell(Pr, t)
        ab, bc, ca = _sub(b, a), _sub(cc, b), _sub(a, cc)
        if math.sqrt(_dot(nt, nt)) <= DEGENERATE_SHARE * (_dot(ab, ab) + _dot(bc, bc) + _dot(ca, ca)):
            continue  # a zero-area triangle: its normal, and so the winding it shows, is rounding
        if g0 < 0:
            continue
        shortest = min(_dist(a, b), _dist(b, cc), _dist(cc, a))
        scale_t = _rounding_scale([a, b, cc])
        reach = math.sqrt(src.on_faces([g0], q)[0]) + WIND_BALL_EDGE * shortest
        ball, seen, out, allowed = [g0], {g0}, set(), 0
        frontier = [g0]
        while frontier:  # the ball grows a ring at a time: one batched distance pass per ring (the set is what counts)
            cands = []
            for f in frontier:
                for h in range(int(src.indptr[f]), int(src.indptr[f + 1])):
                    v = int(src.wflat[h])
                    for g in src.vf[src.vf_ptr[v]:src.vf_ptr[v + 1]].tolist():
                        if g not in seen and g not in out:
                            out.add(g)
                            cands.append(g)
            frontier = []
            if cands:
                for g, hit in zip(cands, src.on_each([[g] for g in cands], [q] * len(cands))):
                    if math.sqrt(hit[0]) <= reach:
                        out.discard(g)
                        seen.add(g)
                        frontier.append(g)
                        ball.append(g)
        for f in ball:
            if _has_area(src, f):
                wind = 1.0 if src.uv is None else _sign(_uv_area(src.uv, src.indptr, f))
                w = _facing(nt, src.normal[f], scale_t + _rounding_scale([tuple(src.P[v]) for v in src.flat[int(src.indptr[f]):int(src.indptr[f + 1])]])) * wind * ref_wind
                allowed |= 3 if wind == 0.0 else (1 if w > 0.0 else (2 if w < 0.0 else 0))
        if allowed:
            common &= allowed
            said = True
    if not said:
        return 0.0
    return {1: 1.0, 2: -1.0, 0: 0.0}.get(common, fallback)


def _read_face(src: _Source, region, Pr, face):
    n = len(face)
    corners = [tuple(Pr[v]) for v in face]
    c0 = c1 = c2 = 0.0
    for p in corners:
        c0 += p[0] / n
        c1 += p[1] / n
        c2 += p[2] / n
    c = (c0, c1, c2)
    diameter, edge = 0.0, 0.0
    for i in range(n):
        edge += _dist(corners[i], corners[(i + 1) % n]) / n
        for j in range(i + 1, n):
            diameter = max(diameter, _dist(corners[i], corners[j]))
    samples = corners + [c]
    near = [max(f, 0) for f in src.grid.nearest_many(samples)]
    hits = [(f, math.sqrt(d2), t, w) for f, (d2, t, w, _) in zip(near, src.on_each([[f] for f in near], samples))]
    weight = [(1.0 - CENTRE_WEIGHT) / n] * n + [CENTRE_WEIGHT]
    votes = []
    for s, hit in enumerate(hits):
        r = region[hit[0]]
        for x in votes:
            if x[0] == r:
                x[1] += weight[s]
                break
        else:
            votes.append([r, weight[s]])
    win = votes[0]
    for v in votes[1:]:
        if v[1] > win[1] or (v[1] == win[1] and v[0] < win[0]):
            win = v
    ref = n if region[hits[n][0]] == win[0] else next((s for s in range(n) if region[hits[s][0]] == win[0]), n)
    rf, rd, rt, rw = hits[ref]
    reg = win[0]
    # A reference turned away from the face (REF_ALIGN_COS): a coarse face across a crease often has its centre nearest
    # a wall turned away from it, whose winding would make the fold fix fold it. The nearest source face within the
    # walk's reach turned at most 60 deg from it, found across every edge (seams too: the vote may have picked the
    # wall's side on a tie of the corners on the crease), becomes the reference and gives the face its side region when
    # it is nearer the centre.
    nr = _newell(Pr, face)
    na = math.sqrt(_dot(nr, nr))

    def aligned(g):
        ng = src.normal[g]
        nb = math.sqrt(_dot(ng, ng))
        return na > 0.0 and nb > 0.0 and _dot(nr, ng) >= REF_ALIGN_COS * na * nb
    if not aligned(rf):
        reach = WALK_REACH * diameter + rd
        near, seen = [], set()
        for f in [rf] + [h[0] for h in hits]:
            if f not in seen:
                seen.add(f)
                near.append(f)
        at = 0
        while at < len(near):
            f = near[at]
            at += 1
            for h in range(int(src.indptr[f]), int(src.indptr[f + 1])):
                g = int(src.across[h])
                if g >= 0 and g not in seen and _dist(src.centre[g], c) <= reach:
                    seen.add(g)
                    near.append(g)
        cands = [g for g in near if aligned(g)]
        # Only a face nearer the centre than the reference: a reference that is the nearest face is the surface the
        # face lies on, however it is turned (a thin part's far side seen from a coarse face).
        if cands:
            d2, t2, w2, which = src.on_faces(cands, c)
            if d2 < src.on_faces([rf], c)[0]:
                rf, rd, rt, rw = cands[which], math.sqrt(d2), t2, w2
                reg = int(region[rf])
    order = _walk(src, rf, rt, rw, c, WALK_REACH * diameter + rd)
    reached = [f for f, _ in order]
    out = dict(ids=[], w=[], shift=[], slid=[], ref_ids=[int(x) for x in src.tri[rt]], ref_w=list(rw), ref_face=rf,
               orient=_fan_orient(src, Pr, face, rf, -1.0 if _dot(nr, src.normal[rf]) < 0.0 else 1.0), region=reg)
    for i, (q, (d2, t, w, which)) in enumerate(zip(corners, src.on_each([reached] * n, corners))):
        tri = src.tri[t]
        slid = math.sqrt(d2) > hits[i][1] + SLIDE_EDGE * edge
        if slid:
            x = _plane_bary(q, tuple(src.cp[tri[0]]), tuple(src.cp[tri[1]]), tuple(src.cp[tri[2]]))
            if x is not None:
                w = x
        out["ids"].extend(int(x) for x in tri)
        out["w"].extend(w)
        out["shift"].append(order[which][1])
        out["slid"].append(slid)
    return out


def _uv_to_3d(p0, p1, p2, q0, q1, q2):
    """The inverse of a triangle's 3D-to-UV map as (k00, k01, k10, k11): UV offsets to 3D lengths in the triangle's
    plane. None when either side has no area."""

    e1, e2 = _sub(p1, p0), _sub(p2, p0)
    l1 = math.sqrt(_dot(e1, e1))
    if l1 <= 0.0:
        return None
    x = (e1[0] / l1, e1[1] / l1, e1[2] / l1)
    bx = _dot(e2, x)
    y3 = (e2[0] - bx * x[0], e2[1] - bx * x[1], e2[2] - bx * x[2])
    by = math.sqrt(_dot(y3, y3))
    if by <= 0.0:
        return None
    du1, dv1, du2, dv2 = q1[0] - q0[0], q1[1] - q0[1], q2[0] - q0[0], q2[1] - q0[1]
    # J = [du1 du2; dv1 dv2] * inverse([l1 bx; 0 by])
    j00, j10 = du1 / l1, dv1 / l1
    j01, j11 = (du2 - j00 * bx) / by, (dv2 - j10 * bx) / by
    det = j00 * j11 - j01 * j10
    if det == 0.0:
        return None
    return (j11 / det, -j01 / det, -j10 / det, j00 / det)


def _uv_near(src, g, shifted, uv_reach) -> bool:
    u_ref, p_ref, k = uv_reach
    u = src.uv_centre[g]
    du, dv = u[0] - u_ref[0], u[1] - u_ref[1]
    if shifted:
        du, dv = du + _whole(-du), dv + _whole(-dv)
    lx, ly = k[0] * du + k[1] * dv, k[2] * du + k[3] * dv
    return math.sqrt(lx * lx + ly * ly) <= WALK_UV_SLACK * (_dist(src.centre[g], p_ref) + src.radius[g])


def _lookup_steps(seams, src_points, src_indptr, src_flat, res_points, res_indptr, res_flat, look: dict, uv=None):
    """Generator: fills `look` face by face (yields after each face). `uv` (the active layer, per source corner) turns
    on the walk's UV check."""

    P = np.asarray(src_points, np.float64).reshape(-1, 3)
    si, sf = np.asarray(src_indptr, np.int64), np.asarray(src_flat, np.int64)
    Pr = np.asarray(res_points, np.float64).reshape(-1, 3).tolist()
    ri, rf = np.asarray(res_indptr, np.int64), np.asarray(res_flat, np.int64)
    src = _Source(P, si, sf, seams, uv)
    yield from src.build()
    region = seams["region"]
    for f in range(len(ri) - 1):
        s, e = int(ri[f]), int(ri[f + 1])
        if len(si) <= 1 or e - s < 3:
            continue
        r = _read_face(src, region, Pr, rf[s:e].tolist())
        look["ids"][3 * s:3 * e] = r["ids"]
        look["w"][3 * s:3 * e] = r["w"]
        look["shift"][s:e] = r["shift"]
        look["slid"][s:e] = r["slid"]
        look["ref_ids"][3 * f:3 * f + 3] = r["ref_ids"]
        look["ref_w"][3 * f:3 * f + 3] = r["ref_w"]
        look["ref_face"][f] = r["ref_face"]
        look["orient"][f] = r["orient"]
        look["region"][f] = r["region"]
        yield


def _empty_lookup(seams, res_indptr, n_corners):
    nf = len(res_indptr) - 1
    return dict(ids=np.zeros(3 * n_corners, np.int64), w=np.zeros(3 * n_corners), shift=np.zeros(n_corners, bool), slid=np.zeros(n_corners, bool),
                ref_ids=np.zeros(3 * nf, np.int64), ref_w=np.zeros(3 * nf), ref_face=np.full(nf, -1, np.int64), orient=np.ones(nf),
                region=np.full(nf, -1, np.int64), join=join_tolerance(seams), done=False)


def evaluate(look: dict, src_indptr, src_flat, layer, res_indptr) -> tuple:
    """UVs per result corner for one source layer (2 per source corner) and the expected winding per result face."""

    si = np.asarray(src_indptr, np.int64)
    L = np.asarray(layer, np.float64).ravel()
    ri = np.asarray(res_indptr, np.int64)
    nf = len(ri) - 1
    uv = np.full(2 * int(ri[-1]), np.nan)
    expected = np.zeros(nf)
    ids, w, ref_ids, ref_w = look["ids"], look["w"], look["ref_ids"], look["ref_w"]

    def at(i3, w3):
        u = v = 0.0
        for k in range(3):
            u += w3[k] * L[2 * i3[k]]
            v += w3[k] * L[2 * i3[k] + 1]
        return u, v

    for f in range(nf):
        rf = look["ref_face"][f]
        if rf < 0:
            continue
        r = at(ref_ids[3 * f:3 * f + 3], ref_w[3 * f:3 * f + 3])
        for h in range(int(ri[f]), int(ri[f + 1])):
            x = at(ids[3 * h:3 * h + 3], w[3 * h:3 * h + 3])
            if look["shift"][h]:
                x = (x[0] + _whole(r[0] - x[0]), x[1] + _whole(r[1] - x[1]))
            uv[2 * h], uv[2 * h + 1] = x
        s, m = int(si[rf]), int(si[rf + 1] - si[rf])
        area = 0.0
        for i in range(m):
            j = (i + 1) % m
            area += L[2 * (s + i)] * L[2 * (s + j) + 1] - L[2 * (s + j)] * L[2 * (s + i) + 1]
        expected[f] = look["orient"][f] if area > 0 else (-look["orient"][f] if area < 0 else 0.0)
    return uv, expected


# ---------------------------------------------------------------- fold fix
def folded(res_indptr, uv, expected, f) -> bool:
    if expected[f] == 0.0:
        return False
    s, n = int(res_indptr[f]), int(res_indptr[f + 1] - res_indptr[f])
    if uv[2 * s] != uv[2 * s]:
        return False
    a = (uv[2 * s], uv[2 * s + 1])
    for i in range(1, n - 1):
        b = (uv[2 * (s + i)], uv[2 * (s + i) + 1])
        c = (uv[2 * (s + i + 1)], uv[2 * (s + i + 1) + 1])
        area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if area * expected[f] <= 0.0:
            return True
    return False


def _uv_gap(uv, h, g):
    return max(abs(uv[2 * h] - uv[2 * g]), abs(uv[2 * h + 1] - uv[2 * g + 1]))


def _fold_fix_steps(res_points, res_indptr, res_flat, uv, region, expected, join, texture_px, report: dict):
    """Generator twin of the kernel's fold fix: untangling, the ring fills for what is left, untangling again. Changes
    `uv` (a NumPy array) in place, fills `report`."""

    ri = [int(x) for x in res_indptr]
    nf = len(ri) - 1
    ex = [float(x) for x in expected]
    before = sum(folded(ri, uv, ex, f) for f in range(nf))
    report.update(before=before, after=before, rings=0, sweeps=0, stuck=0, stalled=False, stopped=False, untangled=0)
    if before == 0:
        report["after"] = 0
        return
    moves = [0]
    yield from _untangle_steps(res_indptr, res_flat, uv, region, ex, join, moves)
    yield from _ring_fill_steps(res_points, res_indptr, res_flat, uv, region, expected, join, texture_px, report)
    if report["after"] > 0:
        yield from _untangle_steps(res_indptr, res_flat, uv, region, ex, join, moves)
        report["after"] = sum(folded(ri, uv, ex, f) for f in range(nf))
    report["before"] = before
    report["untangled"] = moves[0]


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])


def _chebyshev(g, h):
    """Largest t with g_l . x + h_l >= t for every row (unit rows): (x, t) by trying every triple as the active set."""

    n = len(g)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                m = [[g[i][0], g[i][1], -1.0], [g[j][0], g[j][1], -1.0], [g[k][0], g[k][1], -1.0]]
                det = _det3(m)
                if abs(det) < 1e-12:
                    continue
                r = [-h[i], -h[j], -h[k]]
                sol = []
                for c in range(3):
                    a = [row[:] for row in m]
                    for row in range(3):
                        a[row][c] = r[row]
                    sol.append(_det3(a) / det)
                x0, x1, t = sol
                tol = 1e-12 * (1.0 + abs(t))
                if all(g[l][0] * x0 + g[l][1] * x1 + h[l] >= t - tol for l in range(n)) and (best is None or t > best[1]):
                    best = ((x0, x1), t)
    return best


def _untangle_steps(res_indptr, res_flat, uv, region, expected, join, moves: list):
    """Generator twin of the kernel's untangling (moves[0] counts the vertex moves)."""

    ri = [int(x) for x in res_indptr]
    rf = [int(x) for x in res_flat]
    nf = len(ri) - 1
    region = [int(x) for x in region]
    join = [float(x) for x in join]
    n_points = max(rf) + 1 if rf else 0
    vc = [[] for _ in range(n_points)]
    for h, v in enumerate(rf):
        vc[v].append(h)
    face_of = [0] * len(rf)
    for f in range(nf):
        for h in range(ri[f], ri[f + 1]):
            face_of[h] = f
    edge_use = {}
    for f in range(nf):
        face = rf[ri[f]:ri[f + 1]]
        for i in range(len(face)):
            a, b = face[i], face[(i + 1) % len(face)]
            key = (min(a, b), max(a, b))
            edge_use[key] = edge_use.get(key, 0) + 1

    def join_of(r):
        return join[r] if 0 <= r < len(join) else math.inf

    best, stall = None, 0
    while True:
        bad = [f for f in range(nf) if folded(ri, uv, expected, f)]
        if not bad:
            break
        if best is None or len(bad) < best:
            best, stall = len(bad), 0
        else:
            stall += 1
            if stall >= UNTANGLE_STALL:
                break
        verts = sorted({rf[h] for f in bad for h in range(ri[f], ri[f + 1])})
        moved = 0
        for count, v in enumerate(verts):
            if count % 256 == 255:
                yield
            alls = vc[v]
            wedges = []  # the vertex's corners grouped by side region and shared UV (within join)
            for h in alls:
                r = region[face_of[h]]
                if r < 0:
                    continue
                w = next((w for w in wedges if region[face_of[w[0]]] == r and _uv_gap(uv, h, w[0]) <= join_of(r)), None)
                if w is None:
                    wedges.append([h])
                else:
                    w.append(h)

            def closed(h):
                f = face_of[h]
                s, e = ri[f], ri[f + 1]
                n, i = e - s, h - s
                me = rf[h]
                return all(edge_use.get((min(me, o), max(me, o))) == 2 for o in (rf[s + (i + n - 1) % n], rf[s + (i + 1) % n]))

            whole = len(wedges) == 1 and len(wedges[0]) == len(alls) and all(closed(h) for h in alls)
            for cs in wedges:
                if not whole and not any(folded(ri, uv, expected, face_of[h]) for h in cs):
                    continue
                g, hh = [], []
                shortest = math.inf
                for h in cs:
                    f = face_of[h]
                    s, e = ri[f], ri[f + 1]
                    n, i = e - s, h - s
                    for o in (s + (i + n - 1) % n, s + (i + 1) % n):
                        shortest = min(shortest, math.sqrt((float(uv[2 * o]) - float(uv[2 * h])) ** 2 + (float(uv[2 * o + 1]) - float(uv[2 * h + 1])) ** 2))
                    if expected[f] == 0.0:
                        continue
                    for k in range(1, n - 1):
                        tri = (0, k, k + 1)
                        if i not in tri:
                            continue
                        p = tri.index(i)
                        a, b = s + tri[(p + 1) % 3], s + tri[(p + 2) % 3]
                        a0, a1, b0, b1 = float(uv[2 * a]), float(uv[2 * a + 1]), float(uv[2 * b]), float(uv[2 * b + 1])
                        d0, d1 = a0 - b0, a1 - b1
                        g0, g1, h0 = expected[f] * d1, expected[f] * -d0, expected[f] * (a0 * b1 - a1 * b0)
                        norm = math.sqrt(g0 * g0 + g1 * g1)
                        if norm < 1e-15:
                            continue
                        g.append((g0 / norm, g1 / norm))
                        hh.append(h0 / norm)
                x = (float(uv[2 * cs[0]]), float(uv[2 * cs[0] + 1]))
                m0 = [g[l][0] * x[0] + g[l][1] * x[1] + hh[l] for l in range(len(g))]
                if not whole:
                    if not (0.0 < shortest < math.inf) or not g:
                        continue
                    g += [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
                    hh += [shortest - x[0], shortest + x[0], shortest - x[1], shortest + x[1]]
                elif len(g) < 3:
                    continue
                worst = min(m0)
                sol = _chebyshev(g, hh)
                if sol is None:
                    continue
                xs, t = sol
                if t <= max(worst, 0.0):
                    continue
                goal = 0.5 * t
                step = 0.0
                for l in range(len(m0)):
                    m1 = g[l][0] * xs[0] + g[l][1] * xs[1] + hh[l]
                    if m0[l] < goal and m1 > m0[l]:
                        step = max(step, (goal - m0[l]) / (m1 - m0[l]))
                step = min(step, 1.0)
                xn = (x[0] + step * (xs[0] - x[0]), x[1] + step * (xs[1] - x[1]))
                for h in cs:
                    uv[2 * h], uv[2 * h + 1] = xn
                moved += 1
        moves[0] += moved
        yield
        if moved == 0:
            break


def _ring_fill_steps(res_points, res_indptr, res_flat, uv, region, expected, join, texture_px, report: dict):
    """Generator twin of the kernel's ring fills (the fold fix's second layer)."""

    Pr = np.asarray(res_points, np.float64).reshape(-1, 3).tolist()
    ri = [int(x) for x in res_indptr]
    rf = [int(x) for x in res_flat]
    nf = len(ri) - 1
    region = [int(x) for x in region]
    expected = [float(x) for x in expected]
    bad = [folded(ri, uv, expected, f) for f in range(nf)]
    report.update(before=sum(bad), after=0, rings=0, sweeps=0, stuck=0, stalled=False, stopped=False)
    if report["before"] == 0:
        return
    tex = texture_px if texture_px > 0 else TEXTURE_PX_DEFAULT
    original = uv.copy()
    vf = [[] for _ in range(len(Pr))]
    for f in range(nf):
        for v in rf[ri[f]:ri[f + 1]]:
            vf[v].append(f)
    join = [float(x) for x in join]

    def join_of(r):
        return join[r] if 0 <= r < len(join) else math.inf

    def ring(sel):
        out = list(sel)
        for f in range(nf):
            if sel[f]:
                for v in rf[ri[f]:ri[f + 1]]:
                    for g in vf[v]:
                        if region[g] == region[f] and region[g] >= 0:
                            out[g] = True
        return out

    open_edge = {}
    for f in range(nf):
        face = rf[ri[f]:ri[f + 1]]
        for i in range(len(face)):
            a, b = face[i], face[(i + 1) % len(face)]
            key = (min(a, b), max(a, b))
            open_edge[key] = open_edge.get(key, 0) + 1

    def fixed(f, i):
        face = rf[ri[f]:ri[f + 1]]
        v, h = face[i], ri[f] + i
        prev, nxt = face[(i + len(face) - 1) % len(face)], face[(i + 1) % len(face)]

        def on_open(a, b):
            return open_edge.get((min(a, b), max(a, b)), 1) == 1

        if on_open(prev, v) or on_open(v, nxt):
            return True
        for g in vf[v]:
            if region[g] != region[f]:
                return True
            if any(rf[c] == v and _uv_gap(uv, c, h) > join_of(region[f]) for c in range(ri[g], ri[g + 1])):
                return True
        return False

    stuck = []
    for f in range(nf):
        st = False
        if bad[f]:
            s, n = ri[f], ri[f + 1] - ri[f]
            q = lambda i: (uv[2 * (s + i)], uv[2 * (s + i) + 1])  # noqa: E731
            for i in range(1, n - 1):
                a, b, c = q(0), q(i), q(i + 1)
                area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
                if area * expected[f] <= 0.0 and fixed(f, 0) and fixed(f, i) and fixed(f, i + 1):
                    st = True
                    break
        stuck.append(st)
    report["stuck"] = sum(stuck)
    if report["stuck"] == report["before"]:
        report["after"] = report["before"]
        return
    active = ring([bad[f] and not stuck[f] for f in range(nf)])
    touched = list(active)
    rings = 1
    while True:
        report["rings"] = max(report["rings"], rings)
        fill = {}
        yield from _harmonic_fill_steps(Pr, ri, rf, uv, region, active, vf, join_of, tex, fill)
        report["sweeps"] += fill["sweeps"]
        report["stalled"] = report["stalled"] or fill["stalled"]
        spot = _spots(ri, rf, region, active, vf)
        still = [False] * nf
        for f in range(nf):
            if active[f] and not stuck[f] and folded(ri, uv, expected, f):
                still[spot[f]] = True
        keep = [active[f] and still[spot[f]] for f in range(nf)]
        if not any(keep):
            break
        grown = ring(keep)
        if grown == keep:
            break
        touched = [t or g for t, g in zip(touched, grown)]
        active = grown
        rings += 1
    spot = _spots(ri, rf, region, touched, vf)
    was, now = [0] * nf, [0] * nf
    for f in range(nf):
        if touched[f]:
            was[spot[f]] += int(bad[f])
            now[spot[f]] += int(folded(ri, uv, expected, f))
    for f in range(nf):
        if touched[f] and now[spot[f]] >= was[spot[f]]:
            uv[2 * ri[f]:2 * ri[f + 1]] = original[2 * ri[f]:2 * ri[f + 1]]
    report["after"] = sum(folded(ri, uv, expected, f) for f in range(nf))


def _spots(ri, rf, region, active, vf):
    nf = len(ri) - 1
    a, b = [], []
    for f in range(nf):
        if active[f]:
            for v in rf[ri[f]:ri[f + 1]]:
                for g in vf[v]:
                    if active[g] and region[g] == region[f]:
                        a.append(f)
                        b.append(g)
    return _min_labels(nf, np.array(a, np.int64), np.array(b, np.int64)).tolist()


def _harmonic_fill_steps(Pr, ri, rf, uv, region, active, vf, join_of, tex, out: dict):
    nf = len(ri) - 1
    groups = {}
    value, vertex, free, first = [], [], [], []
    corner_node = {}
    for f in range(nf):
        if not active[f]:
            continue
        for h in range(ri[f], ri[f + 1]):
            v = rf[h]
            tol = join_of(region[f])
            lst = groups.setdefault((v, region[f]), [])
            node = next((n for n in lst if _uv_gap(uv, first[n], h) <= tol), None)
            if node is None:
                value.append([float(uv[2 * h]), float(uv[2 * h + 1])])
                vertex.append(v)
                first.append(h)
                pinned = False
                for g in vf[v]:
                    if not active[g] and region[g] == region[f]:
                        c = next((c for c in range(ri[g], ri[g + 1]) if rf[c] == v), None)
                        if c is not None and _uv_gap(uv, c, h) <= tol:
                            pinned = True
                            break
                free.append(not pinned)
                lst.append(len(value) - 1)
                node = len(value) - 1
            corner_node[h] = node
    pairs = []
    for f in range(nf):
        if active[f]:
            s, e = ri[f], ri[f + 1]
            for h in range(s, e):
                g = s if h + 1 == e else h + 1
                a, b = corner_node[h], corner_node[g]
                if a != b:
                    pairs.append((min(a, b), max(a, b)))
    pairs.sort()
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j] == pairs[i]:
            j += 1
        if j - i == 1:
            free[pairs[i][0]] = free[pairs[i][1]] = False
        i = j
    uniq = []
    for p in pairs:
        if not uniq or uniq[-1] != p:
            uniq.append(p)
    nbr = [[] for _ in value]
    for a, b in uniq:
        w = 1.0 / max(_dist(Pr[vertex[a]], Pr[vertex[b]]), 1e-12)
        nbr[a].append((b, w))
        nbr[b].append((a, w))
    goal = STOP_TEXEL_SHARE / tex
    history, stalled = [], False
    order = [k for k in range(len(value)) if free[k]]
    while True:
        most = 0.0
        for count, k in enumerate(order):
            su = sv = sw = 0.0
            for j, w in nbr[k]:
                su += w * value[j][0]
                sv += w * value[j][1]
                sw += w
            if sw > 0.0:
                nu, nv = su / sw, sv / sw
                most = max(most, max(abs(nu - value[k][0]), abs(nv - value[k][1])))
                value[k][0], value[k][1] = nu, nv
            if count % 512 == 511:
                yield
        history.append(most)
        yield
        if most <= goal:
            break
        it = len(history)
        if it > STALL_WINDOW and most > (1.0 - STALL_DROP) * history[it - 1 - STALL_WINDOW]:
            stalled = True
            break
    for h, node in corner_node.items():
        if free[node]:
            uv[2 * h], uv[2 * h + 1] = value[node]
    out.update(sweeps=len(history), stalled=stalled)


def fold_fix(res_points, res_indptr, res_flat, uv, region, expected, join=(), texture_px: float = TEXTURE_PX_DEFAULT) -> dict:
    """Fold fix in one call (changes `uv` in place); returns the report."""

    report = {}
    for _ in _fold_fix_steps(res_points, res_indptr, res_flat, uv, region, expected, join, texture_px, report):
        pass
    return report


def reread_folded(src_points, src_flat, layer_uv, look: dict, res_points, res_indptr, res_flat, uv, expected) -> int:
    """Twin of the kernel's last fold-fix layer: a face that still folds reads its corners again through its reference
    triangle's own (affine) map; kept only when that unfolds it. Changes `uv` in place; returns the faces read again."""

    if src_points is None or len(src_points) == 0:
        return 0
    SP = np.asarray(src_points, np.float64).reshape(-1, 3)
    sf = np.asarray(src_flat, np.int64)
    L = np.asarray(layer_uv, np.float64).ravel()
    RP = np.asarray(res_points, np.float64).reshape(-1, 3)
    ri, rf = np.asarray(res_indptr, np.int64), np.asarray(res_flat, np.int64)
    done = 0
    for f in range(len(ri) - 1):
        if look["ref_face"][f] < 0 or not folded(ri, uv, expected, f):
            continue
        c = [int(x) for x in look["ref_ids"][3 * f:3 * f + 3]]
        p = [tuple(SP[sf[h]]) for h in c]
        q = [(L[2 * h], L[2 * h + 1]) for h in c]
        e1, e2 = _sub(p[1], p[0]), _sub(p[2], p[0])
        g11, g12, g22 = _dot(e1, e1), _dot(e1, e2), _dot(e2, e2)
        det = g11 * g22 - g12 * g12
        if not abs(det) > 1e-30:
            continue
        s, e = int(ri[f]), int(ri[f + 1])
        old = uv[2 * s:2 * e].copy()
        for h in range(s, e):
            x = _sub(tuple(RP[rf[h]]), p[0])
            r1, r2 = _dot(x, e1), _dot(x, e2)
            b1, b2 = (g22 * r1 - g12 * r2) / det, (g11 * r2 - g12 * r1) / det
            for k in range(2):
                uv[2 * h + k] = q[0][k] + b1 * (q[1][k] - q[0][k]) + b2 * (q[2][k] - q[0][k])
        if folded(ri, uv, expected, f):
            uv[2 * s:2 * e] = old
        else:
            done += 1
    return done


# ---------------------------------------------------------------- the sliced stage
class Transfer:
    """Per-side transfer as a sliced stage: `step()` does about 4 ms of work (or polls the faster job), `done` turns
    True at the end, `result` then holds the same dict the faster callable returns (uv, expected, lookup arrays, fold
    report). `stop()` ends it early with what was read (`result["done"]` False)."""

    def __init__(self, src_points, src_indptr, src_flat, corner_uv, res_points, res_indptr, res_flat, material=None, repeat=(),
                 texture_px: float = TEXTURE_PX_DEFAULT, uv_tol_px: float = UV_TOL_PX, classes=(), fix_folds: bool = True, use_fast: bool = True):
        self.args = (src_points, src_indptr, src_flat, corner_uv, res_points, res_indptr, res_flat)
        self.material, self.repeat, self.texture_px, self.uv_tol_px = material, repeat, texture_px, uv_tol_px
        self.classes, self.fix_folds = classes, fix_folds
        self.done, self.result, self.numbers = False, None, {"estimate": True}
        self._job = None
        self._stopped = False
        fast = FAST.get("transfer") if use_fast else None
        if fast is not None:
            self._job = fast(src_points, src_indptr, src_flat, corner_uv, res_points, res_indptr, res_flat, () if material is None else material,
                             repeat, texture_px, uv_tol_px, classes, fix_folds)
        self._steps = None if self._job is not None else self._run()

    def stop(self) -> None:
        self._stopped = True
        if self._job is not None:
            self._job.stop()

    def step(self) -> None:
        if self.done:
            return
        if self._job is not None:
            if self._job.done():
                self.result = self._job.result()
                self._finish()
            return
        end = time.perf_counter() + SLICE_S
        while time.perf_counter() < end:
            try:
                next(self._steps)
            except StopIteration:
                self._finish()
                return

    def _finish(self) -> None:
        self.done = True
        r = self.result
        self.numbers = dict(estimate=False, done=bool(r["done"]), slid=int(np.sum(r["slid"])), folded_before=r["folded_before"],
                            folded_after=r["folded_after"], fold_stuck=r["fold_stuck"])

    def _run(self):
        sp, si, sf, suv, rp, ri, rf = self.args
        seams = {}
        yield from _extract_steps(sp, si, sf, suv, self.material, self.repeat, self.texture_px, self.uv_tol_px, (), seams)
        if len(self.classes):
            set_classes(seams, self.classes)
        yield
        ri = np.asarray(ri, np.int64)
        n_corners = int(ri[-1])
        look = _empty_lookup(seams, ri, n_corners)
        for _ in _lookup_steps(seams, sp, si, sf, rp, ri, rf, look, suv):
            if self._stopped:
                break
            yield
        look["done"] = not self._stopped
        uv, expected = evaluate(look, si, sf, np.asarray(suv, np.float64).ravel(), ri)
        report = dict(before=0, after=0, rings=0, sweeps=0, stuck=0, stalled=False, stopped=False, untangled=0, reread=0)
        if self.fix_folds and look["done"]:
            for _ in _fold_fix_steps(rp, ri, rf, uv, look["region"], expected, look["join"], seams["texture_px"], report):
                if self._stopped:
                    report["stopped"] = True
                    break
                yield
            if report["after"] > 0 and not self._stopped:
                report["reread"] = reread_folded(sp, sf, suv, look, rp, ri, rf, uv, expected)
                report["after"] -= report["reread"]
        self.result = dict(uv=uv, expected=expected, **look, **{f"fold_{k}" if k not in ("before", "after") else f"folded_{k}": v for k, v in report.items()})


def transfer(src_points, src_indptr, src_flat, corner_uv, res_points, res_indptr, res_flat, material=None, repeat=(),
             texture_px: float = TEXTURE_PX_DEFAULT, uv_tol_px: float = UV_TOL_PX, classes=(), fix_folds: bool = True, use_fast: bool = True) -> dict:
    """Per-side transfer run to the end (see `Transfer`)."""

    stage = Transfer(src_points, src_indptr, src_flat, corner_uv, res_points, res_indptr, res_flat, material, repeat, texture_px, uv_tol_px,
                     classes, fix_folds, use_fast)
    if stage._job is not None:
        stage._job._thread.join()
    while not stage.done:
        stage.step()
    return stage.result


def layer(src_indptr, src_flat, layer_uv, res_points, res_indptr, res_flat, look: dict, texture_px: float = TEXTURE_PX_DEFAULT,
          fix_folds: bool = True, use_fast: bool = True, src_points=()) -> tuple:
    """Another UV layer through a transfer's lookup: (uv, expected, fold report). `src_points` (empty =
    none) turns on the fold fix's last layer (`reread_folded`)."""

    fast = FAST.get("layer") if use_fast else None
    if fast is not None:
        job = fast(src_indptr, src_flat, layer_uv, res_points, res_indptr, res_flat, look, texture_px, fix_folds, src_points)
        if job is not None:
            job._thread.join()
            uv, expected, fold = job.result()
            return uv, expected, fold
    uv, expected = evaluate(look, src_indptr, src_flat, layer_uv, res_indptr)
    report = dict(before=0, after=0, rings=0, sweeps=0, stuck=0, stalled=False, stopped=False, untangled=0, reread=0)
    if fix_folds:
        for _ in _fold_fix_steps(res_points, res_indptr, res_flat, uv, look["region"], expected, look["join"], texture_px, report):
            pass
        if report["after"] > 0:
            report["reread"] = reread_folded(src_points, src_flat, layer_uv, look, res_points, res_indptr, res_flat, uv, expected)
            report["after"] -= report["reread"]
    return uv, expected, {f"fold_{k}" if k not in ("before", "after") else f"folded_{k}": v for k, v in report.items()}
