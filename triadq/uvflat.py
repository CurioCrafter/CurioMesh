# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Flatten and solver twin (package K5).

NumPy only, no bpy. The reference is the kernel source `uvflat.rs`; this twin covers the exact grid layout, quad ARAP
(local/global, Liu, Zhang, Xu, Gotsman, Gortler 2008) with Straight rows Off / Soft / On, the injective step (Smith,
Schaefer 2015), Tutte's start (Tutte 1963; Floater 1997) and the chart stretch. The global step is CG with a Jacobi
preconditioner (Hestenes, Stiefel 1952), matrix products by `np.bincount`, the same pin and stop rule as the kernel.

`Flatten` is sliceable: `step()` does about 4 ms of work and returns progress; its numbers say "estimate" until it
has converged. When the adapter installed a faster `flatten` callable, the stage hands the chart to it (a job that
never blocks) and only polls it.
"""

from __future__ import annotations

import time

import numpy as np

# Constants table: same names and values as the top of the matching .rs file (tests/test_uv_scaffold.py compares).
TEXTURE_PX = 1024.0
STOP_TEXEL_SHARE = 0.02
STALL_WINDOW = 50
STALL_DROP = 0.01
RATE_WINDOW = 10
ARAP_CHANGE = 0.01
STEP_SHARE = 0.9
PULL_SHARE = 0.001
SOFT_START = 0.1
SOFT_GROW = 4.0
BEND_P95_DEG = 2.0
STRETCH_QUANTILE = 0.99
FACTOR_MAX_BYTES = 2.0e9
ND_LEAF = 64
STATS_LEN = 17
# Twin only: systems with at most this many rows (pinned included) are solved directly (dense, like the kernel's direct
# factor) instead of by CG; a merge band has ~20. Well under a millisecond per solve at this size.
DIRECT_MAX = 64  # dense solve only this small: above about 100 rows NumPy's threaded solve can take 10+ ms on a busy PC (one slice)

ROWS = {"off": 0, "soft": 1, "on": 2, "auto": 3}
STARTS = {"given": 0, "pull": 1, "free": 2}
STAT_COLUMNS = ("chart", "vertices", "faces", "path", "rows", "steps", "energy", "flips", "stretch", "bend", "factor_ms", "solve_ms", "total_ms", "stopped", "nnz_l", "start", "stalled")

# Optional faster callables, set by the one adapter (uv_fast.py); the NumPy code runs when a name is missing.
FAST: dict = {}


def set_fast(**callables) -> None:
    """Install (or, with None, remove) faster callables by name."""

    for key, fn in callables.items():
        if fn is None:
            FAST.pop(key, None)
        else:
            FAST[key] = fn


# ---------------------------------------------------------------------------------------------------------------------
# Chart geometry
# ---------------------------------------------------------------------------------------------------------------------


def _faces(indptr, flat):
    indptr, flat = np.asarray(indptr, dtype=np.int64), np.asarray(flat, dtype=np.int64)
    return [flat[indptr[f]:indptr[f + 1]] for f in range(len(indptr) - 1)]


class Elements:
    """ARAP terms (one rotation per face) and the test triangles (quads: both diagonals) of a chart."""

    def __init__(self, points, indptr, flat):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        indptr, flat = np.asarray(indptr, dtype=np.int64), np.asarray(flat, dtype=np.int64)
        sizes = np.diff(indptr)
        self.nfaces = len(sizes)
        a, b, w, e, elem = [], [], [], [], []
        tv, tr, tf = [], [], []
        for k in np.unique(sizes):
            k = int(k)
            ids = np.flatnonzero(sizes == k)
            if k < 3 or len(ids) == 0:
                continue
            corner = indptr[ids][:, None] + np.arange(k)
            verts = flat[corner]
            q = _flat_polygons(points[verts])
            nxt = np.roll(np.arange(k), -1)
            side = q[:, nxt] - q
            length = np.linalg.norm(side, axis=2)
            if k == 3:
                opp = (np.arange(3) + 2) % 3
                u = q - q[:, opp]
                v = q[:, nxt] - q[:, opp]
                area2 = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
                with np.errstate(divide="ignore", invalid="ignore"):
                    weight = np.where(area2 < 1e-300, 0.0, np.maximum(0.5 * (u * v).sum(axis=2) / area2, 0.0))
            else:
                prev = np.roll(np.arange(k), 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    weight = np.where(length < 1e-300, 0.0, 0.25 * (length[:, prev] + length[:, nxt]) / length)
            a.append(verts.ravel())
            b.append(verts[:, nxt].ravel())
            w.append(weight.ravel())
            e.append(side.reshape(-1, 2))
            elem.append(np.repeat(ids, k))
            tris = [[0, 1, 2]] if k == 3 else ([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]] if k == 4 else [[0, i, i + 1] for i in range(1, k - 1)])
            for t in tris:
                tv.append(verts[:, t])
                tr.append(q[:, t])
                tf.append(ids)
        def cat(parts, shape, dtype):
            return np.concatenate(parts) if parts else np.zeros(shape, dtype=dtype)

        self.a, self.b = cat(a, (0,), np.int64), cat(b, (0,), np.int64)
        self.w, self.e = cat(w, (0,), np.float64), cat(e, (0, 2), np.float64)
        self.elem = cat(elem, (0,), np.int64)
        self.tri_v = cat(tv, (0, 3), np.int64)
        self.tri_r = cat(tr, (0, 3, 2), np.float64)
        self.tri_face = cat(tf, (0,), np.int64)


def _flat_polygons(p: np.ndarray) -> np.ndarray:
    """(m, k, 3) corners -> (m, k, 2) in each polygon's own plane (Newell normal, first side along x)."""

    nxt = np.roll(np.arange(p.shape[1]), -1)
    a, b = p, p[:, nxt]
    normal = np.stack([((a[..., 1] - b[..., 1]) * (a[..., 2] + b[..., 2])).sum(1), ((a[..., 2] - b[..., 2]) * (a[..., 0] + b[..., 0])).sum(1), ((a[..., 0] - b[..., 0]) * (a[..., 1] + b[..., 1])).sum(1)], axis=1)
    centre = p.mean(axis=1)
    ln = np.linalg.norm(normal, axis=1)
    first = p[:, 1] - p[:, 0]
    bad = (ln < 1e-300) | (np.linalg.norm(first, axis=1) < 1e-300)
    normal = normal / np.where(bad, 1.0, ln)[:, None]
    t = first - normal * (first * normal).sum(1, keepdims=True)
    e1 = t / np.maximum(np.linalg.norm(t, axis=1), 1e-300)[:, None]
    e2 = np.cross(normal, e1)
    rel = p - centre[:, None]
    out = np.stack([(rel * e1[:, None]).sum(2), (rel * e2[:, None]).sum(2)], axis=2)
    out[bad] = 0.0
    return out


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def count_flips(indptr, flat, uv) -> int:
    """Faces whose UV polygon has a non-positive signed area."""

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    indptr, flat = np.asarray(indptr, dtype=np.int64), np.asarray(flat, dtype=np.int64)
    sizes = np.diff(indptr)
    face = np.repeat(np.arange(len(sizes)), sizes)
    nxt = indptr[face] + (np.arange(len(flat)) - indptr[face] + 1) % np.maximum(sizes[face], 1)
    area = np.bincount(face, weights=_cross2(uv[flat], uv[flat[nxt]]), minlength=len(sizes))
    return int(np.count_nonzero(~(area > 0.0) & (sizes >= 3)))


def _quantile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    at = min(int(round(q * (len(values) - 1))), len(values) - 1)
    return float(np.partition(values, at)[at])


def _stretch_of(el: Elements, uv: np.ndarray) -> tuple[float, np.ndarray]:
    uv = uv.reshape(-1, 2)
    r = el.tri_r
    e1, e2 = r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]
    det = _cross2(e1, e2)
    p = uv[el.tri_v]
    a3, a2 = 0.5 * np.abs(det).sum(), 0.5 * np.abs(_cross2(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])).sum()
    per = np.zeros(el.nfaces)
    if not (a3 > 0 and a2 > 0):
        per[:] = np.inf
        return float("inf"), per
    s = np.sqrt(a3 / a2)
    f1, f2 = s * (p[:, 1] - p[:, 0]), s * (p[:, 2] - p[:, 0])
    ok = np.abs(det) >= 1e-300
    d = np.where(ok, det, 1.0)
    inv = np.stack([np.stack([e2[:, 1] / d, -e2[:, 0] / d], 1), np.stack([-e1[:, 1] / d, e1[:, 0] / d], 1)], 1)
    ja = f1[:, 0] * inv[:, 0, 0] + f2[:, 0] * inv[:, 1, 0]
    jb = f1[:, 0] * inv[:, 0, 1] + f2[:, 0] * inv[:, 1, 1]
    jc = f1[:, 1] * inv[:, 0, 0] + f2[:, 1] * inv[:, 1, 0]
    jd = f1[:, 1] * inv[:, 0, 1] + f2[:, 1] * inv[:, 1, 1]
    qq = np.hypot((ja + jd) / 2, (jc - jb) / 2)
    rr = np.hypot((ja - jd) / 2, (jc + jb) / 2)
    s1, s2 = qq + rr, np.abs(qq - rr)
    with np.errstate(divide="ignore"):
        value = np.where((_cross2(f1, f2) * det <= 0) | (s2 < 1e-300), np.inf, np.maximum(s1, 1.0 / np.maximum(s2, 1e-300)))
    np.maximum.at(per, el.tri_face[ok], value[ok])
    return _quantile(per, STRETCH_QUANTILE), per


def stretch(points, indptr, flat, uv) -> tuple[float, np.ndarray]:
    """(stretch at `STRETCH_QUANTILE` over faces, per face): UVs scaled to the surface area, per face the worst test
    triangle's max(s1, 1/s2) (singular values, Sander, Snyder, Gortler, Hoppe 2001). 1 = isometric."""

    return _stretch_of(Elements(points, indptr, flat), np.asarray(uv, dtype=np.float64))


def _injective_step(el: Elements, uv: np.ndarray, d: np.ndarray) -> float:
    """`STEP_SHARE` of the largest step along `d` that keeps every positive test triangle positive (at most 1)."""

    uv, d = uv.reshape(-1, 2), d.reshape(-1, 2)
    p, q = uv[el.tri_v], d[el.tri_v]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    f1, f2 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    c = _cross2(e1, e2)
    keep = c > 0
    a, b, c = _cross2(f1, f2)[keep], (_cross2(e1, f2) + _cross2(f1, e2))[keep], c[keep]
    if len(c) == 0:
        return 1.0
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.abs(c))
    linear = np.abs(a) <= 1e-14 * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lin = np.where(b < 0, -c / np.where(b < 0, b, -1.0), np.inf)
        disc = b * b - 4 * a * c
        root = np.sqrt(np.maximum(disc, 0.0))
        qv = -0.5 * (b + np.where(b >= 0, 1.0, -1.0) * root)
        r1 = qv / np.where(linear, 1.0, a)
        r2 = np.where(qv != 0, c / np.where(qv != 0, qv, 1.0), np.inf)
        r1 = np.where(r1 > 0, r1, np.inf)
        r2 = np.where(r2 > 0, r2, np.inf)
        t_quad = np.where(disc < 0, np.inf, np.minimum(r1, r2))
    t = np.where(linear, t_lin, t_quad)
    return float(min(STEP_SHARE * t.min(), 1.0))


# ---------------------------------------------------------------------------------------------------------------------
# Exact grid layout and loops
# ---------------------------------------------------------------------------------------------------------------------


def grid_layout(points, indptr, flat):
    """Exact grid UVs of a pole-free quad chart: (uv (n, 2), col, row), NaN / -1 for unused vertices; None when the
    faces are not a grid (a non-quad, a pole or a closed ring inside, two vertices on one grid point)."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    faces = _faces(indptr, flat)
    if any(len(f) != 4 for f in faces):
        return None
    edge_faces: dict = {}
    for f, poly in enumerate(faces):
        for i in range(4):
            x, y = int(poly[i]), int(poly[(i + 1) % 4])
            edge_faces.setdefault((min(x, y), max(x, y)), []).append(f)
    ij: dict = {}
    if not faces:
        return np.full((n, 2), np.nan), np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    for c, key in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        ij[int(faces[0][c])] = key
    done = np.zeros(len(faces), dtype=bool)
    done[0] = True
    queue, head = [0], 0
    while head < len(queue):
        f = queue[head]
        head += 1
        poly = faces[f]
        for i in range(4):
            x, y = int(poly[i]), int(poly[(i + 1) % 4])
            for g in edge_faces[(min(x, y), max(x, y))]:
                if done[g]:
                    continue
                gp = [int(v) for v in faces[g]]
                at = next((c for c in range(4) if gp[c] == y and gp[(c + 1) % 4] == x), None)
                if at is None:
                    return None
                p, q = ij[y], ij[x]
                step = (q[0] - p[0], q[1] - p[1])
                rot = (-step[1], step[0])
                for v, want in ((gp[(at + 2) % 4], (q[0] + rot[0], q[1] + rot[1])), (gp[(at + 3) % 4], (p[0] + rot[0], p[1] + rot[1]))):
                    have = ij.setdefault(v, want)
                    if have != want:
                        return None
                done[g] = True
                queue.append(g)
    if not done.all() or len(set(ij.values())) != len(ij):
        return None
    keys = np.array(list(ij.values()), dtype=np.int64)
    i0, j0 = keys.min(axis=0)
    ni, nj = keys.max(axis=0) - (i0, j0)
    wsum, wcnt, hsum, hcnt = np.zeros(ni), np.zeros(ni), np.zeros(nj), np.zeros(nj)
    for (x, y) in edge_faces:
        pa, pb = ij[x], ij[y]
        length = float(np.linalg.norm(points[x] - points[y]))
        if pa[1] == pb[1]:
            k = min(pa[0], pb[0]) - i0
            wsum[k] += length
            wcnt[k] += 1
        else:
            k = min(pa[1], pb[1]) - j0
            hsum[k] += length
            hcnt[k] += 1
    us = np.concatenate([[0.0], np.cumsum(np.where(wcnt > 0, wsum / np.maximum(wcnt, 1), 0.0))])
    vs = np.concatenate([[0.0], np.cumsum(np.where(hcnt > 0, hsum / np.maximum(hcnt, 1), 0.0))])
    uv, col, row = np.full((n, 2), np.nan), np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    for v, (i, j) in ij.items():
        col[v], row[v] = i - i0, j - j0
        uv[v] = us[i - i0], vs[j - j0]
    return uv, col, row


def loop_triples(indptr, flat, n: int) -> np.ndarray:
    """(a, v, b) where a - v - b continues one quad row through v (interior vertices with four quads, border vertices
    with two quads); poles end rows. Label-free."""

    fans: list = [[] for _ in range(n)]
    for poly in _faces(indptr, flat):
        k = len(poly)
        for c in range(k):
            fans[int(poly[c])].append((int(poly[c - 1]), int(poly[(c + 1) % k]), k == 4))
    out = []
    for v, fan in enumerate(fans):
        if not fan or not all(x[2] for x in fan):
            continue
        nb = sorted({w for x in fan for w in x[:2]})
        def shares(p, q):
            return any((x[0] == p and x[1] == q) or (x[0] == q and x[1] == p) for x in fan)

        if len(fan) == 4 and len(nb) == 4:
            for i in range(4):
                for j in range(i + 1, 4):
                    if not shares(nb[i], nb[j]):
                        out.append((nb[i], v, nb[j]))
        elif len(fan) == 2 and len(nb) == 3:
            inner = next((w for w in nb if all(x[0] == w or x[1] == w for x in fan)), None)
            if inner is not None:
                ends = [w for w in nb if w != inner]
                out.append((ends[0], v, ends[1]))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def bend_p95(triples: np.ndarray, uv) -> float:
    """95th percentile of the turning angle (degrees) along the triples."""

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(triples) == 0:
        return float("nan")
    x = uv[triples[:, 1]] - uv[triples[:, 0]]
    y = uv[triples[:, 2]] - uv[triples[:, 1]]
    return _quantile(np.degrees(np.abs(np.arctan2(_cross2(x, y), (x * y).sum(1)))), 0.95)


def labels_fit(indptr, flat, col, row) -> bool:
    """Straight rows On is possible only when every face is a quad whose sides alternate "same column" / "same row"."""

    for poly in _faces(indptr, flat):
        if len(poly) != 4:
            return False
        s = [(col[poly[i]] == col[poly[(i + 1) % 4]], row[poly[i]] == row[poly[(i + 1) % 4]]) for i in range(4)]
        if not (all(x[0] != x[1] for x in s) and s[0][0] != s[1][0] and s[0] == s[2] and s[1] == s[3]):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------------
# One SPD system with pinned unknowns, CG with a Jacobi preconditioner
# ---------------------------------------------------------------------------------------------------------------------


class _System:
    """Symmetric matrix as (row, col, value) entries (duplicates add up), pinned rows and columns replaced by the
    identity; products by `np.bincount`."""

    def __init__(self, m: int, rows, cols, vals, pins):
        rows, cols, vals = np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float64)
        rows, cols, vals = np.concatenate([rows, np.arange(m)]), np.concatenate([cols, np.arange(m)]), np.concatenate([vals, np.full(m, 1e-12)])
        pinned = np.zeros(m, dtype=bool)
        pinned[np.asarray(pins, np.int64)] = True
        moved = ~pinned[rows] & pinned[cols]
        self.move_r, self.move_c, self.move_v = rows[moved], cols[moved], vals[moved]
        keep = ~pinned[rows] & ~pinned[cols]
        self.r = np.concatenate([rows[keep], np.flatnonzero(pinned)])
        self.c = np.concatenate([cols[keep], np.flatnonzero(pinned)])
        self.v = np.concatenate([vals[keep], np.ones(int(pinned.sum()))])
        self.m, self.pinned = m, pinned
        self.diag = np.bincount(self.r[self.r == self.c], weights=self.v[self.r == self.c], minlength=m)
        self.dense = None
        if m <= DIRECT_MAX:  # the dense matrix holds every row, pinned ones too
            self.dense = np.zeros((m, m))
            np.add.at(self.dense, (self.r, self.c), self.v)

    def mul(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.r, weights=self.v * x[self.c], minlength=self.m)

    def rhs(self, b: np.ndarray, pin_values: np.ndarray) -> np.ndarray:
        """`b` with the pinned rows set to their values and the pinned columns moved over."""

        b = b.copy()
        b[self.pinned] = pin_values[self.pinned]
        b -= np.bincount(self.move_r, weights=self.move_v * b[self.move_c], minlength=self.m)
        return b

    def cg(self, b: np.ndarray, x: np.ndarray, tol: float, info: dict, stopped=lambda: False):
        """Jacobi-preconditioned CG from `x` (changed in place), a generator yielding after every iteration. Same stop
        rule as the kernel: the moves still to come (this step's largest move times rho/(1-rho), rho the residual's mean
        shrink per iteration over `RATE_WINDOW` iterations) shift no unknown by more than `tol` (twice in a row), or the
        residual stalls (counted in info["stalled"]). No iteration cap. A small system (`DIRECT_MAX`) is solved
        directly instead, without yielding."""

        if self.dense is not None:
            x[:] = np.linalg.solve(self.dense, b)
            return
        r = b - self.mul(x)
        z = r / self.diag
        p = z.copy()
        rz = float(r @ z)
        window = float(np.linalg.norm(r))
        recent = [window] * RATE_WINDOW
        small = iterations = 0
        while rz > 0.0:
            if stopped():
                info["stopped"] = True
                return
            ap = self.mul(p)
            pap = float(p @ ap)
            if pap <= 0:
                return
            alpha = rz / pap
            x += alpha * p
            r -= alpha * ap
            iterations += 1
            info["iterations"] = info.get("iterations", 0) + 1
            now = float(np.linalg.norm(r))
            then = recent[iterations % RATE_WINDOW]
            recent[iterations % RATE_WINDOW] = now
            rho = min((now / then) ** (1.0 / min(iterations, RATE_WINDOW)), 1 - 1e-12) if then > 0 else 0.0
            if float(np.abs(alpha * p).max()) * rho / (1 - rho) <= tol:
                small += 1
                if small >= 2:
                    return
            else:
                small = 0
            if iterations % STALL_WINDOW == 0:
                if now > (1 - STALL_DROP) * window:
                    info["stalled"] = info.get("stalled", 0) + 1
                    return
                window = now
            z = r / self.diag
            rz_new = float(r @ z)
            p = z + (rz_new / rz) * p
            rz = rz_new
            yield


# ---------------------------------------------------------------------------------------------------------------------
# Flatten one chart (sliceable)
# ---------------------------------------------------------------------------------------------------------------------


def _border_loop(indptr, flat, n: int) -> list:
    sides = set()
    for poly in _faces(indptr, flat):
        k = len(poly)
        for i in range(k):
            sides.add((int(poly[i]), int(poly[(i + 1) % k])))
    nxt = {a: b for (a, b) in sides if (b, a) not in sides}
    seen, loops = set(), []
    for s in sorted(nxt):
        if s in seen:
            continue
        loop, v = [], s
        while v is not None and v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt.get(v)
        loops.append(loop)
    loops.sort(key=len, reverse=True)
    return loops[0] if loops else []


class Flatten:
    """Flatten one chart (section 6.5 of the spec): start, quad ARAP with Straight rows, injective steps.

    points (n, 3), faces in CSR over 0..n-1, `start` (n, 2) or None, `col` / `row` per vertex for On. rows: "off",
    "soft", "on", "auto"; start_mode: "given" (Tutte when it flips or is missing), "free" (Tutte here; the faster
    callable uses LSCM), "pull" (as "given" here). `steps` = 0 runs until the energy changes by less than
    `ARAP_CHANGE`. `step()` works about `budget` seconds and returns progress in 0..1; `done`, `uv`, `numbers`.
    """

    def __init__(self, points, indptr, flat, start=None, col=None, row=None, rows: str = "auto", start_mode: str = "given",
                 steps: int = 0, stretch_allowed: float = 0.10, texture_px: float = TEXTURE_PX, budget: float = 0.004, use_fast: bool = True):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.indptr, self.flat = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
        self.n = len(self.points)
        self.start = None if start is None else np.asarray(start, dtype=np.float64).reshape(-1, 2)
        self.col = None if col is None else np.asarray(col, np.int64)
        self.row = None if row is None else np.asarray(row, np.int64)
        self.rows, self.start_mode, self.steps = rows, start_mode, int(steps)
        self.stretch_allowed, self.texture_px, self.budget = float(stretch_allowed), float(texture_px), float(budget)
        self.done, self.uv, self._stop = False, None, False
        self.numbers: dict = {"estimate": True}
        self._job = None
        self._fast = FAST.get("flatten") if use_fast else None
        self._work = self._run()

    def stop(self) -> None:
        """End early; the flip-free state reached so far is kept (Keep going resumes from it)."""

        self._stop = True
        if self._job is not None:
            self._job.stop()

    def step(self) -> float:
        if self.done:
            return 1.0
        if self._fast is not None:
            return self._step_fast()
        until = time.perf_counter() + self.budget
        progress = 0.0
        try:
            while time.perf_counter() < until:
                value = next(self._work)
                progress = progress if value is None else value
        except StopIteration:
            self.done = True
            return 1.0
        return progress

    # Faster callable: one job for the chart, polled.
    def _step_fast(self) -> float:
        if self._job is None:
            corners = len(self.flat)
            start = np.full(2 * corners, np.nan) if self.start is None else self.start[self.flat].ravel()
            col = np.zeros(0, np.int64) if self.col is None else self.col[self.flat]
            row = np.zeros(0, np.int64) if self.row is None else self.row[self.flat]
            self._job = self._fast(self.points.ravel(), self.indptr, self.flat, np.zeros(len(self.indptr) - 1, np.int64), start, col, row,
                                   ROWS[self.rows], STARTS[self.start_mode], self.stretch_allowed, self.steps, 0.0, self.texture_px)
            if self._job is None:  # the callable declined: NumPy path
                self._fast = None
                return 0.0
            if self._stop:
                self._job.stop()
        if not self._job.done():
            return 0.5
        corner_uv, stats = self._job.result()
        uv = np.full((self.n, 2), np.nan)
        uv[self.flat] = np.asarray(corner_uv).reshape(-1, 2)
        self.uv = uv
        row = np.asarray(stats, dtype=np.float64)[:STATS_LEN]
        self.numbers = dict(zip(STAT_COLUMNS, row.tolist()))
        self.numbers["rows"] = ("off", "soft", "on")[int(row[4])] if np.isfinite(row[4]) else "off"
        self.numbers["estimate"] = bool(row[13])
        self.done = True
        return 1.0

    # NumPy path: a generator, sliced by step().
    def _run(self):
        n = self.n
        yield 0.0
        el = Elements(self.points, self.indptr, self.flat)
        yield 0.02
        given = self.start
        if given is not None and given.shape == (n, 2) and np.isfinite(given).all() and self.start_mode != "free" and count_flips(self.indptr, self.flat, given) == 0:
            uv, kind = given.copy(), "given"
        else:
            uv = yield from self._tutte()
            kind = "tutte"
        triples = loop_triples(self.indptr, self.flat, n)
        span = float(np.linalg.norm(np.nanmax(uv, axis=0) - np.nanmin(uv, axis=0))) or 1e-300
        tol = STOP_TEXEL_SHARE * span / (self.texture_px if self.texture_px > 0 else TEXTURE_PX)
        labelled = (self.col is not None and self.row is not None and len(self.col) == n and len(self.row) == n
                    and (self.col >= 0).all() and (self.row >= 0).all() and labels_fit(self.indptr, self.flat, self.col, self.row))
        info: dict = {}
        used = None
        if self.rows in ("on", "auto") and labelled:
            trial = uv.copy()
            energy = yield from self._arap(el, trial, triples, 0.0, True, tol, info)
            if self.rows == "on" or _stretch_of(el, trial)[0] <= 1.0 + self.stretch_allowed or self._stop:
                uv, used = trial, "on"
        if used is None:
            used = "off" if self.rows == "off" or len(triples) == 0 else "soft"
            if used == "off":
                energy = yield from self._arap(el, uv, triples, 0.0, False, tol, info)
            else:
                d = np.bincount(el.a, weights=el.w, minlength=n) + np.bincount(el.b, weights=el.w, minlength=n)
                w = SOFT_START * float(d.mean())
                base, best = uv.copy(), None
                while True:
                    trial, trial_info = base.copy(), {}
                    e = yield from self._arap(el, trial, triples, w, False, tol, trial_info)
                    bend = bend_p95(triples, trial)
                    improved = best is None or bend < (1 - STALL_DROP) * best[0]
                    if improved:
                        best = (bend, trial, e, trial_info)
                    if bend <= BEND_P95_DEG or not improved or self._stop:
                        break
                    w *= SOFT_GROW
                _, soft_uv, soft_energy, soft_info = best
                # Auto refuses Soft rows that break Stretch allowed, as it does for On: Off is the fallback.
                if self.rows == "auto" and not self._stop and _stretch_of(el, soft_uv)[0] > 1.0 + self.stretch_allowed:
                    used, info = "off", {}
                    energy = yield from self._arap(el, uv, triples, 0.0, False, tol, info)
                else:
                    uv, energy, info = soft_uv, soft_energy, soft_info
        self.uv = uv
        self.numbers = {
            "rows": used, "start": kind, "steps": info.get("steps", 0), "energy": energy,
            "flips": count_flips(self.indptr, self.flat, uv), "stretch": _stretch_of(el, uv)[0],
            "bend": bend_p95(triples, uv), "stalled": info.get("stalled", 0), "stopped": bool(self._stop),
            "estimate": bool(self._stop or info.get("stalled", 0)),
        }

    def _tutte(self):
        n = self.n
        border = _border_loop(self.indptr, self.flat, n)
        if not border:
            raise ValueError("chart has no border (closed): it needs a cut")
        a, b = [], []
        for poly in _faces(self.indptr, self.flat):
            k = len(poly)
            a.extend(int(poly[i]) for i in range(k))
            b.extend(int(poly[(i + 1) % k]) for i in range(k))
        a, b = np.array(a), np.array(b)
        half = np.full(len(a), 0.5)
        rows = np.concatenate([a, b, a, b])
        cols = np.concatenate([a, b, b, a])
        vals = np.concatenate([half, half, -half, -half])
        border = np.array(border)
        p = self.points[border]
        lens = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        total = max(float(lens.sum()), 1e-300)
        theta = 2 * np.pi * np.concatenate([[0.0], np.cumsum(lens)[:-1]]) / total
        radius = total / (2 * np.pi)
        uv = np.zeros((n, 2))
        uv[border] = radius * np.stack([np.cos(theta), np.sin(theta)], 1)
        system = _System(n, rows, cols, vals, border)
        info: dict = {}
        for axis in range(2):
            x = uv[:, axis].copy()
            yield from system.cg(system.rhs(np.zeros(n), uv[:, axis]), x, radius * 1e-9, info, lambda: self._stop)
            uv[:, axis] = x
        return uv

    def _arap(self, el: Elements, uv: np.ndarray, triples, soft: float, on: bool, tol: float, info: dict):
        """Quad ARAP on `uv` in place (one vertex pinned: the translation null space); returns the energy."""

        n = self.n
        if on:
            maps = [np.unique(self.col, return_inverse=True)[1].ravel(), np.unique(self.row, return_inverse=True)[1].ravel()]
        else:
            maps = [np.arange(n), np.arange(n)]
        systems = []
        for axis in range(2 if on else 1):
            mp = maps[axis]
            m = int(mp.max()) + 1 if n else 0
            A, B, W = mp[el.a], mp[el.b], el.w
            rows, cols, vals = [A, B, A, B], [A, B, B, A], [W, W, -W, -W]
            if soft > 0 and len(triples):
                ids = mp[triples]
                coef = np.array([1.0, -2.0, 1.0])
                for i in range(3):
                    for j in range(3):
                        rows.append(ids[:, i])
                        cols.append(ids[:, j])
                        vals.append(np.full(len(ids), soft * coef[i] * coef[j]))
            systems.append((mp, m, _System(m, np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), [mp[0]])))
            yield 0.05
        if not on:
            systems.append(systems[0])
        zero = 1e-20 * float((el.w * (el.e ** 2).sum(1)).sum())
        nel = el.nfaces
        steps, last, energy = 0, np.inf, 0.0
        while True:
            du = uv[el.b] - uv[el.a]
            m00 = np.bincount(el.elem, weights=el.w * du[:, 0] * el.e[:, 0], minlength=nel)
            m01 = np.bincount(el.elem, weights=el.w * du[:, 0] * el.e[:, 1], minlength=nel)
            m10 = np.bincount(el.elem, weights=el.w * du[:, 1] * el.e[:, 0], minlength=nel)
            m11 = np.bincount(el.elem, weights=el.w * du[:, 1] * el.e[:, 1], minlength=nel)
            # cos and sin of the rotation straight from its vector (the kernel's form); zero gives the identity.
            rx, ry = m00 + m11, m10 - m01
            h = np.sqrt(rx * rx + ry * ry)
            zero_h = h == 0.0
            h = np.where(zero_h, 1.0, h)
            c, s = np.where(zero_h, 1.0, rx / h)[el.elem], np.where(zero_h, 0.0, ry / h)[el.elem]
            re = np.stack([c * el.e[:, 0] - s * el.e[:, 1], s * el.e[:, 0] + c * el.e[:, 1]], 1)
            energy = float((el.w * ((du - re) ** 2).sum(1)).sum())
            converged = steps >= self.steps if self.steps > 0 else (steps > 0 and abs(last - energy) <= ARAP_CHANGE * max(last, 1e-300))
            if converged or energy <= zero or self._stop:
                break
            last = energy
            new = uv.copy()
            for axis in range(2):
                mp, m, system = systems[axis]
                b = np.bincount(mp[el.b], weights=el.w * re[:, axis], minlength=m) - np.bincount(mp[el.a], weights=el.w * re[:, axis], minlength=m)
                x = np.bincount(mp, weights=uv[:, axis], minlength=m) / np.maximum(np.bincount(mp, minlength=m), 1)
                yield from system.cg(system.rhs(b, x), x, tol, info, lambda: self._stop)
                new[:, axis] = x[mp]
            if self._stop:
                break
            t = _injective_step(el, uv, new - uv)
            uv += t * (new - uv)
            steps += 1
            info["steps"] = steps
            yield (steps / self.steps) if self.steps > 0 else steps / (steps + 5.0)
        info["steps"] = steps
        return energy


def flatten_chart(points, indptr, flat, start=None, **options) -> tuple[np.ndarray, dict]:
    """Run `Flatten` to the end (tests and scripts): (uv (n, 2), numbers)."""

    stage = Flatten(points, indptr, flat, start, **options)
    while not stage.done:
        stage.step()
    return stage.uv, stage.numbers
