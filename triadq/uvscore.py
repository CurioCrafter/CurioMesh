# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""UV Check twin (package K3): every UV number and the verdict. NumPy only, no bpy.

Same steps as the compiled kernel `uvscore.rs` (SPEC 9.1); a parity test compares the two. Our own code. Cited
ideas: Sander, Snyder, Gortler, Hoppe 2001 (L2 stretch from the Jacobian of the 3D to UV map); closest point on a
triangle is textbook geometry (Ericson 2005); the pixel-centre fill rule with a shared-edge tie break is the usual
rasteriser rule (Pineda 1988). `uv_verdict()` turns the numbers into GOOD / CHECK / BAD (SPEC 9.2).

`check()` does the work in one call (about 4 ms slices are not needed here: it is a few NumPy passes; a caller that
must stay responsive runs it from its stage and polls). When a faster callable is installed (`set_fast`) it is used.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

# Constants table: same names and values as the top of the matching .rs file (tests/test_uv_scaffold.py compares).
TEXTURE_PX = 1024.0
UV_TOL_PX = 0.01
OWNER_UV_PX = 2.0
SEEN_MIN = 0.5
DRIFT_EDGE_SLACK = 0.25
EPS_DIAG = 1e-4
EPS_EDGE = 0.05
WELD = 1e-6
ROW_LINE_TEXEL = 0.25
RASTER_MAX_PX = 8192.0
GRID_EDGES = 2.0
GRID_MAX_DIM = 4096.0
# A nearest-distance probe starts at this share of the cell edge (a speed setting only: the answer is exact either way).
PROBE_START = 0.0625
STRETCH_BAD = 4.0
DRIFT_BAD_PX = 8.0
AREA_MIN = 1e-14

NAMES = [
    "faces", "tris", "seen_faces_pct", "straddle_faces", "straddle_pct", "drift_p50", "drift_p99", "drift_over8_pct",
    "seam_fit_max", "seam_fit_p99", "extra_faces", "split_faces", "stretch_med", "stretch_worst1", "stretch_over4_pct",
    "stretch_worst1_seen", "angle_p90", "flipped_faces", "flipped_seen_faces", "flipped_vs_source", "overlap_pct",
    "space_used_pct", "pixel_spread", "seams_in_sight", "cut_per_sqrt_area", "straight_rows_pct", "row_bend_p95",
    "islands", "island_min_px",
]
_I = {name: i for i, name in enumerate(NAMES)}

# Verdict thresholds (SPEC 9.2), one block. "good" / "check" are the upper (or, for the `min` rules, lower) edges; the
# Keep rules that are relative to the source add the delta.
UV_VERDICT: dict = {
    "across": {"good": 3.0, "check": 10.0},  # % of faces across seams (Keep)
    "drift": {"good": 2.0, "check": 8.0},  # texture drift p99, px (Keep)
    "flipped_per_1000": {"check": 1.0},  # on seen faces; GOOD is none (Keep: no more than the source)
    "overlap_pct": {"check": 0.5},  # unintended; GOOD is none (Keep: no more than the source)
    "stretch_med": {"good": 1.10, "check": 1.30},  # New
    "stretch_worst1": {"good": 1.6, "check": 2.5},  # New
    "stretch_delta": {"med": 0.10, "worst1": 0.5},  # Keep: GOOD while <= source + delta
    "space": {"good": 70.0, "check": 55.0},  # New, lower is worse
    "spread": {"good": 1.5, "check": 2.5},  # New
    "spread_delta": 0.5,  # Keep: GOOD while <= source + delta
    "straight_rows": {"good": 85.0, "check": 60.0},  # New, On charts, lower is worse
    "in_sight": {"good": 0.35, "check": 0.55},  # New, share of cut px seen
    "seam_step": {"good": 2.0, "check": 6.0},  # after a bake, p95 normal degrees
}

# Optional faster callables, set by the one adapter (uv_fast.py); the NumPy code runs when a name is missing.
FAST: dict = {}


def set_fast(**callables) -> None:
    """Install (or, with None, remove) faster callables by name."""

    for key, fn in callables.items():
        if fn is None:
            FAST.pop(key, None)
        else:
            FAST[key] = fn


Mesh = namedtuple("Mesh", "points indptr flat uv material")
Mesh.__new__.__defaults__ = (None,)
Check = namedtuple("Check", "numbers straddle flipped")

_BARY7 = np.array([[1 / 3, 1 / 3, 1 / 3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])


def mesh(points, indptr, flat, uv, material=None) -> Mesh:
    """Arrays as the kernel takes them: points (n, 3), CSR faces, UV per corner (c, 2), material per face or None."""

    return Mesh(np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3), np.asarray(indptr, dtype=np.int64),
                np.asarray(flat, dtype=np.int64), np.ascontiguousarray(uv, dtype=np.float64).reshape(-1, 2),
                None if material is None else np.asarray(material, dtype=np.int64))


# ---------------------------------------------------------------- helpers


def _fan(m: Mesh):
    """Triangle corner ids (nt, 3) and face of each triangle: a fan from corner 0 of every face."""

    sizes = np.diff(m.indptr)
    corners, faces = [], []
    for k in np.unique(sizes):
        if k < 3:
            continue
        fs = np.flatnonzero(sizes == k)
        base = m.indptr[fs]
        for j in range(1, k - 1):
            corners.append(np.stack([base, base + j, base + j + 1], 1))
            faces.append(fs)
    if not corners:
        return np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    corners, faces = np.concatenate(corners), np.concatenate(faces)
    order = np.lexsort((np.arange(len(faces)), faces))
    return corners[order], faces[order]


def _material(m: Mesh, faces):
    return np.zeros(len(faces), np.int64) if m.material is None or len(m.material) == 0 else m.material[faces]


def _weld(points):
    if len(points) == 0:
        return np.zeros(0, np.int64)
    ext = float(np.ptp(points, axis=0).max())
    q = ext * WELD if ext > 0 else 1.0
    key = np.round(points / q).astype(np.int64)
    return np.unique(key, axis=0, return_inverse=True)[1].ravel()


def _percentile(values, q):
    if len(values) == 0:
        return float("nan")
    v = np.sort(values)
    pos = q / 100.0 * (len(v) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (pos - lo))


def _wquantile(values, weights, q):
    if len(values) == 0:
        return float("nan")
    order = np.lexsort((np.arange(len(values)), values))
    cum = np.cumsum(weights[order])
    k = int(np.searchsorted(cum / cum[-1], q, side="left"))
    return float(values[order[min(k, len(order) - 1)]])


def _union_labels(n, a, b):
    """Connected component label (0..) of n items joined by the pairs (a, b)."""

    lab = np.arange(n)
    if len(a):
        while True:
            la, lb = lab[a], lab[b]
            low = np.minimum(la, lb)
            new = lab.copy()
            np.minimum.at(new, la, low)
            np.minimum.at(new, lb, low)
            new = new[new]
            if np.array_equal(new, lab):
                break
            lab = new
    return np.unique(lab, return_inverse=True)[1].ravel(), lab


def _islands(m: Mesh, tc, tf, tex):
    """Island label per triangle, island count, cut edges as (half-edge triangle ids, sides, group starts)."""

    nt = len(tc)
    tol = UV_TOL_PX / tex
    w = _weld(m.points)
    vert = w[m.flat[tc]]  # (nt, 3) welded vertices
    uvc = m.uv[tc]  # (nt, 3, 2)
    ha, hb = vert, np.roll(vert, -1, axis=1)
    keep = (ha != hb).ravel()
    hid = np.arange(nt * 3)[keep]
    lo = np.minimum(ha, hb).ravel()[keep]
    hi = np.maximum(ha, hb).ravel()[keep]
    order = np.lexsort((hid, hi, lo))
    hid, lo, hi = hid[order], lo[order], hi[order]
    new = np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])]
    start = np.flatnonzero(new)
    end = np.r_[start[1:], len(hid)]
    size = end - start
    tri, side = hid // 3, hid % 3
    uv0 = uvc[tri, side]
    uv1 = uvc[tri, (side + 1) % 3]
    v0 = vert[tri, side]
    mat = _material(m, tf)[tri]
    ua, ub, joined_pairs, cut_groups = [], [], [], []

    def pair(x, y):
        same_dir = v0[x] == v0[y]
        close = lambda p, q: (np.abs(p - q) <= tol).all(axis=-1)
        same_uv = np.where(same_dir, close(uv0[x], uv0[y]) & close(uv1[x], uv1[y]), close(uv0[x], uv1[y]) & close(uv1[x], uv0[y]))
        return same_uv & (mat[x] == mat[y])

    two = np.flatnonzero(size == 2)
    ok2 = pair(start[two], start[two] + 1)
    ua.append(tri[start[two]][ok2])
    ub.append(tri[start[two] + 1][ok2])
    cut_groups.append(two[~ok2])
    for g in np.flatnonzero(size > 2):  # non-manifold edges: every pair
        idx = np.arange(start[g], end[g])
        all_joined = True
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if pair(np.array([idx[i]]), np.array([idx[j]]))[0]:
                    ua.append(np.array([tri[idx[i]]]))
                    ub.append(np.array([tri[idx[j]]]))
                else:
                    all_joined = False
        if not all_joined:
            cut_groups.append(np.array([g]))
    label, _ = _union_labels(nt, np.concatenate(ua), np.concatenate(ub))
    cut = np.sort(np.concatenate(cut_groups)) if cut_groups else np.zeros(0, np.int64)
    cut_tri = [tri[start[g]:end[g]] for g in cut]
    cut_side = [side[start[g]:end[g]] for g in cut]
    return label, int(label.max()) + 1 if nt else 0, cut_tri, cut_side


def _closest(p, a, b, c):
    """Closest point on triangle abc to p, all (k, 3): barycentric (k, 3) and distance (Ericson 2005, last put wins)."""

    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    den = va + vb + vc
    safe = np.where(np.abs(den) < 1e-300, 1.0, den)
    v, w = vb / safe, vc / safe
    u = 1 - v - w
    u = np.where(np.abs(den) < 1e-300, 1.0, u)
    v = np.where(np.abs(den) < 1e-300, 0.0, v)
    w = np.where(np.abs(den) < 1e-300, 0.0, w)

    def put(mask, uu, vv, ww):
        nonlocal u, v, w
        u, v, w = np.where(mask, uu, u), np.where(mask, vv, v), np.where(mask, ww, w)

    z, o = np.zeros_like(d1), np.ones_like(d1)

    def div(x, y):
        return x / np.where(y == 0, 1.0, y)

    t = div(d4 - d3, (d4 - d3) + (d5 - d6))
    put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), z, 1 - t, t)
    t = div(d2, d2 - d6)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t, z, t)
    put((d6 >= 0) & (d5 <= d6), z, z, o)
    t = div(d1, d1 - d3)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t, t, z)
    put((d3 >= 0) & (d4 <= d3), z, o, z)
    put((d1 <= 0) & (d2 <= 0), o, z, z)
    q = u[:, None] * a + v[:, None] * b + w[:, None] * c
    return np.stack([u, v, w], 1), np.linalg.norm(p - q, axis=1)


class _Grid:
    """Uniform grid over triangles; `ball` finds every triangle within a radius of each query (exact)."""

    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c
        n = len(a)
        allp = np.concatenate([a, b, c]) if n else np.zeros((1, 3))
        self.lo = allp.min(0)
        hi = allp.max(0)
        ext = float((hi - self.lo).max())
        edge = np.sort(np.linalg.norm(b - a, axis=1)) if n else np.zeros(1)
        median = float(edge[len(edge) // 2] if len(edge) % 2 else (edge[len(edge) // 2 - 1] + edge[len(edge) // 2]) / 2) if n else 0.0
        self.h = max(GRID_EDGES * median, ext / GRID_MAX_DIM, 1e-12)
        self.dim = np.floor((hi - self.lo) / self.h).astype(np.int64) + 1
        tlo = self.cell(np.minimum(np.minimum(a, b), c))
        thi = self.cell(np.maximum(np.maximum(a, b), c))
        span = thi - tlo + 1
        count = span.prod(1)
        tri = np.repeat(np.arange(n), count)
        k = np.arange(len(tri)) - np.repeat(np.cumsum(count) - count, count)
        sx, sy = span[tri, 0], span[tri, 1]
        cell = np.stack([k % sx, (k // sx) % sy, k // (sx * sy)], 1) + tlo[tri]
        ids = self._id(cell)
        order = np.argsort(ids, kind="stable")
        self.ids, self.tris = ids[order], tri[order]
        self.tlo = tlo

    def cell(self, p):
        return np.clip(np.floor((p - self.lo) / self.h).astype(np.int64), 0, self.dim - 1)

    def _id(self, cell):
        return (cell[:, 2] * self.dim[1] + cell[:, 1]) * self.dim[0] + cell[:, 0]

    def _tris_in(self, cl, cu):
        """Triangles in the cells of the box cl..cu (deduplicated)."""

        xs, ys, zs = (np.arange(cl[k], cu[k] + 1) for k in range(3))
        cells = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
        ids = self._id(cells)
        lo = np.searchsorted(self.ids, ids, side="left")
        hi = np.searchsorted(self.ids, ids, side="right")
        if not len(lo):
            return np.zeros(0, np.int64)
        picks = np.concatenate([self.tris[x:y] for x, y in zip(lo, hi)]) if (hi > lo).any() else np.zeros(0, np.int64)
        return np.unique(picks)

    def ball(self, q, r, chunk=40000):
        """Pairs (query, triangle, barycentric, distance) with distance <= r[query]."""

        out_q, out_t, out_b, out_d = [], [], [], []
        if len(self.a) == 0 or len(q) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0)
        cells = self._id(self.cell(q))
        order = np.argsort(cells, kind="stable")
        sc = cells[order]
        bounds = np.r_[0, np.flatnonzero(sc[1:] != sc[:-1]) + 1, len(sc)]
        for g in range(len(bounds) - 1):
            idx = order[bounds[g]:bounds[g + 1]]
            rmax = float(r[idx].max())
            cl = self.cell(q[idx].min(0) - rmax)
            cu = self.cell(q[idx].max(0) + rmax)
            cand = self._tris_in(cl, cu)
            if not len(cand):
                continue
            for s in range(0, len(idx), max(1, chunk // len(cand))):
                part = idx[s:s + max(1, chunk // len(cand))]
                qq = np.repeat(q[part], len(cand), 0)
                tt = np.tile(cand, len(part))
                bary, dist = _closest(qq, self.a[tt], self.b[tt], self.c[tt])
                keep = dist <= np.repeat(r[part], len(cand))
                out_q.append(np.repeat(part, len(cand))[keep])
                out_t.append(tt[keep])
                out_b.append(bary[keep])
                out_d.append(dist[keep])
        if not out_q:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0)
        return np.concatenate(out_q), np.concatenate(out_t), np.concatenate(out_b), np.concatenate(out_d)

    def nearest(self, q):
        """Exact distance to the nearest triangle for every query."""

        best = np.full(len(q), np.inf)
        todo = np.arange(len(q))
        r = self.h
        while len(todo):
            qi, _, _, d = self.ball(q[todo], np.full(len(todo), r))
            found = np.full(len(todo), np.inf)
            np.minimum.at(found, qi, d)
            done = found <= r
            best[todo[done]] = found[done]
            todo = todo[~done]
            r *= 2.0
            if r > 1e30:
                break
        return best


# ---------------------------------------------------------------- raster


def _fill(p, size):
    """Texel centres inside each triangle p (k, 3, 2) in pixel units: (triangle index, texel index) pairs."""

    p = p.copy()
    area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    good = (area != 0) & np.isfinite(area)
    swap = good & (area < 0)
    tmp = p[swap, 1].copy()
    p[swap, 1] = p[swap, 2]
    p[swap, 2] = tmp
    lim = float(size)
    x0, x1 = p[:, :, 0].min(1), p[:, :, 0].max(1)
    y0, y1 = p[:, :, 1].min(1), p[:, :, 1].max(1)
    ix0 = np.floor(x0 - 0.5).clip(0).astype(np.int64)
    ix1 = np.ceil(np.minimum(x1 + 0.5, lim)).astype(np.int64)
    iy0 = np.floor(y0 - 0.5).clip(0).astype(np.int64)
    iy1 = np.ceil(np.minimum(y1 + 0.5, lim)).astype(np.int64)
    w = np.where(good, np.maximum(ix1 - ix0, 0), 0)
    h = np.where(good, np.maximum(iy1 - iy0, 0), 0)
    count = w * h
    tri = np.repeat(np.arange(len(p)), count)
    k = np.arange(len(tri)) - np.repeat(np.cumsum(count) - count, count)
    x = ix0[tri] + k % np.maximum(w[tri], 1)
    y = iy0[tri] + k // np.maximum(w[tri], 1)
    cx, cy = x + 0.5, y + 0.5
    inside = np.ones(len(tri), bool)
    for e in range(3):
        a, b = p[tri, e], p[tri, (e + 1) % 3]
        a_first = (a[:, 0] < b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] <= b[:, 1]))
        lo = np.where(a_first[:, None], a, b)
        hi = np.where(a_first[:, None], b, a)
        s = np.where(a_first, 1.0, -1.0)
        val = s * ((hi[:, 0] - lo[:, 0]) * (cy - lo[:, 1]) - (hi[:, 1] - lo[:, 1]) * (cx - lo[:, 0]))
        owns = (b[:, 1] > a[:, 1]) | ((b[:, 1] == a[:, 1]) & (b[:, 0] < a[:, 0]))
        inside &= (val > 0) | ((val == 0) & owns)
    return tri[inside], (y * size + x)[inside]


def _raster(m, tc, tf, ru, a3, allow, tex):
    size = int(np.clip(np.round(tex), 16, RASTER_MAX_PX))
    mat = _material(m, tf)
    cen = (ru[:, 0] + ru[:, 1] + ru[:, 2]) / 3.0
    tile = np.floor(cen).astype(np.int64)
    key = np.stack([mat, tile[:, 0], tile[:, 1]], 1)
    pages, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    allowed = np.zeros(len(tf), bool) if allow is None or len(allow) == 0 else allow[tf].astype(bool)
    covered = over = used = weight = 0.0
    for pg in range(len(pages)):
        sel = np.flatnonzero(inv == pg)
        count = np.zeros(size * size, np.int64)
        flag = np.zeros(size * size, bool)
        p = (ru[sel] - pages[pg, 1:3].astype(np.float64)) * float(size)
        for part in range(0, len(sel), 20000):
            sl = slice(part, part + 20000)
            tri, texel = _fill(p[sl], size)
            normal = ~allowed[sel][sl][tri]
            count += np.bincount(texel[normal], minlength=size * size)
            flag[texel[~normal]] = True
        c = np.minimum(count, 2)
        covered += float((c > 0).sum())
        over += float((c >= 2).sum())
        used += float(a3[sel].sum()) * float(((c > 0) | flag).sum()) / (size * size)
        weight += float(a3[sel].sum())
    if not len(pages):
        return float("nan"), float("nan")
    return (100.0 * over / covered if covered > 0 else 0.0), (100.0 * used / weight if weight > 0 else 0.0)


# ---------------------------------------------------------------- the check


def _tri_stats(q, t):
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    a2s = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    a3 = 0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
    return a2s, a3


def _seam_fit(src_seg, src_mat, res_seg, res_mat, tex):
    """Distance (px) of each result seam sample to the nearest source seam segment of the same material."""

    samples = np.concatenate([res_seg[:, :2], (res_seg[:, :2] + res_seg[:, 2:]) / 2.0, res_seg[:, 2:]])
    smat = np.tile(res_mat, 3)
    out = np.zeros(len(samples))
    for mt in np.unique(smat):
        segs = src_seg[src_mat == mt]
        if not len(segs):
            continue
        idx = np.flatnonzero(smat == mt)
        d = segs[:, 2:] - segs[:, :2]
        l2 = (d * d).sum(1)
        step = max(1, 4000000 // len(segs))
        for s in range(0, len(idx), step):
            p = samples[idx[s:s + step]][:, None, :]
            w = p - segs[None, :, :2]
            t = np.where(l2 > 0, np.clip((w * d).sum(-1) / np.where(l2 > 0, l2, 1.0), 0.0, 1.0), 0.0)
            best = np.sqrt(((w - t[..., None] * d) ** 2).sum(-1)).min(1)
            out[idx[s:s + step]] = best * tex
    return out


def _edges(m: Mesh):
    """Unique undirected edges in first-occurrence order: (a, b, corner lists per edge as first corner, count)."""

    faces = np.repeat(np.arange(len(m.indptr) - 1), np.diff(m.indptr))
    nxt = np.arange(len(m.flat)) + 1
    nxt[m.indptr[1:] - 1] = m.indptr[:-1]
    a, b = m.flat, m.flat[nxt]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = np.stack([lo, hi], 1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return lo[first][order], hi[first][order], rank[inv], faces


def check(res: Mesh, src: "Mesh | None" = None, *, seen_face=None, seen_edge_px=None, allow_overlap=None, col=None, row=None,
          on_chart=None, density=None, src_region=None, faces_off: int = 0, split=None, texture_px: float = TEXTURE_PX,
          use_fast: bool = True) -> Check:
    """UV Check numbers of `res` (and, with `src`, of the result against its source), keyed by `NAMES`."""

    tex = float(texture_px) if texture_px and texture_px > 0 else TEXTURE_PX
    fast = FAST.get("check") if use_fast else None
    if fast is not None:
        def f(a, dt=np.float64):
            return np.zeros(0, dt) if a is None else np.asarray(a, dtype=dt).ravel()

        extra = (src.points, src.indptr, src.flat, src.uv, f(src.material, np.int64)) if src is not None else ((), (), (), (), ())
        job = fast(res.points, res.indptr, res.flat, res.uv, f(res.material, np.int64), *extra, src_region=f(src_region, np.int64),
                   seen_face=f(seen_face), seen_edge_px=f(seen_edge_px), allow_overlap=f(allow_overlap, bool), col=f(col, np.int64),
                   row=f(row, np.int64), on_chart=f(on_chart, bool), density=f(density), faces_off=faces_off, split=f(split, bool),
                   texture_px=tex)
        if job is not None:
            while not job.done():
                pass
            numbers, straddle, flipped = job.result()
            return Check(np.asarray(numbers), np.asarray(straddle), np.asarray(flipped))
    return _check(res, src, seen_face, seen_edge_px, allow_overlap, col, row, on_chart, density, src_region, faces_off, split, tex)


def _check(res, src, seen_face, seen_edge_px, allow_overlap, col, row, on_chart, density, src_region, faces_off, split, tex):
    num = np.full(len(NAMES), np.nan)
    nf = len(res.indptr) - 1
    tc, tf = _fan(res)
    nt = len(tc)
    num[_I["faces"]], num[_I["tris"]] = nf, nt
    seen_face = None if seen_face is None or len(seen_face) == 0 else np.asarray(seen_face, dtype=np.float64)
    seen = np.ones(nf, bool) if seen_face is None else seen_face >= SEEN_MIN
    num[_I["seen_faces_pct"]] = 100.0 if nf == 0 else 100.0 * seen.mean()
    rq = res.points[res.flat[tc]]
    ru = res.uv[tc]
    a2s, a3 = _tri_stats(rq, ru)
    a2 = np.abs(a2s)
    label, nisl, cut_tri, cut_side = _islands(res, tc, tf, tex)
    straddle = np.zeros(nf, bool)
    flipped_tri = np.zeros(nt, bool)
    vs_source = False

    if src is not None:
        sc, sf = _fan(src)
        slabel, _, scut_tri, scut_side = _islands(src, sc, sf, tex)
        region = slabel if src_region is None or len(src_region) != len(src.indptr) - 1 else np.maximum(np.asarray(src_region), 0)[sf]
        sq = src.points[src.flat[sc]]
        su = src.uv[sc]
        grid = _Grid(sq[:, 0], sq[:, 1], sq[:, 2])
        diag = float(np.linalg.norm(np.ptp(src.points, axis=0)))
        edge = np.sort(np.linalg.norm(sq[:, 1] - sq[:, 0], axis=1))
        n = len(edge)
        median = float(edge[n // 2] if n % 2 else (edge[n // 2 - 1] + edge[n // 2]) / 2) if n else 0.0
        eps = max(EPS_DIAG * diag, EPS_EDGE * median)
        # owner sets per result corner
        cpts = res.points[res.flat]
        d0 = grid.nearest(cpts)
        qi, ti, bary, _ = grid.ball(cpts, d0 + eps)
        reg = region[ti]
        suv = np.einsum("kj,kjc->kc", bary, su[ti])
        pairs = np.unique(np.stack([qi, reg], 1), axis=0) if len(qi) else np.zeros((0, 2), np.int64)
        many = np.bincount(pairs[:, 0], minlength=len(cpts)) > 1
        close = np.linalg.norm(suv - res.uv[qi], axis=1) * tex <= OWNER_UV_PX
        use = ~many[qi] | close
        pairs = np.unique(np.stack([qi[use], reg[use]], 1), axis=0) if len(qi) else pairs
        corner_face = np.repeat(np.arange(nf), np.diff(res.indptr))
        fr = np.stack([corner_face[pairs[:, 0]], pairs[:, 1]], 1)
        uniq, cnt = np.unique(fr, axis=0, return_counts=True) if len(fr) else (np.zeros((0, 2), np.int64), np.zeros(0, np.int64))
        full = cnt == np.diff(res.indptr)[uniq[:, 0]] if len(uniq) else np.zeros(0, bool)
        ok_face = np.zeros(nf, bool)
        ok_face[uniq[full, 0]] = True
        straddle = ~ok_face
        num[_I["straddle_faces"]] = straddle.sum()
        num[_I["straddle_pct"]] = 0.0 if nf == 0 else 100.0 * straddle.mean()
        # drift and flips: 7 samples per result triangle
        sp = np.einsum("ijk,lj->ilk", rq, _BARY7).reshape(-1, 3)
        suv_r = np.einsum("ijk,lj->ilk", ru, _BARY7).reshape(-1, 2)
        sd0 = grid.nearest(sp)
        face_err = sd0.reshape(-1, 7).sum(1) / 7.0
        slack = np.maximum(face_err, DRIFT_EDGE_SLACK * np.linalg.norm(rq[:, 1] - rq[:, 0], axis=1))
        rad = sd0 + np.repeat(slack, 7)
        qi, ti, bary, _ = grid.ball(sp, rad)
        uv_c = np.einsum("kj,kjc->kc", bary, su[ti])
        duv = np.linalg.norm(uv_c - suv_r[qi], axis=1) * tex
        drift = np.full(len(sp), np.inf)
        np.minimum.at(drift, qi, duv)
        drift_all = np.sort(drift)
        num[_I["drift_p50"]] = _percentile(drift_all, 50)
        num[_I["drift_p99"]] = _percentile(drift_all, 99)
        num[_I["drift_over8_pct"]] = 0.0 if not len(drift) else 100.0 * (drift > DRIFT_BAD_PX).mean()
        s2s, _ = _tri_stats(sq, su)
        sn = np.cross(sq[:, 1] - sq[:, 0], sq[:, 2] - sq[:, 0])
        nr = np.cross(rq[:, 1] - rq[:, 0], rq[:, 2] - rq[:, 0])
        centre = (qi % 7) == 0
        qt = qi[centre] // 7
        sgn = np.sign(s2s[ti[centre]]) * np.sign((sn[ti[centre]] * nr[qt]).sum(-1))
        match = (sgn == np.sign(a2s[qt])) | (np.abs(s2s[ti[centre]]) < AREA_MIN)
        matched = np.zeros(nt, bool)
        matched[qt[match]] = True
        flipped_tri = ~matched & (np.abs(a2s) >= AREA_MIN)
        vs_source = True
        # seam fit
        def segments(m, ctri, cside, tcorner, tmat):
            if not ctri:
                return np.zeros((0, 4)), np.zeros(0, np.int64)
            t = np.concatenate(ctri)
            s = np.concatenate(cside)
            p, q = m.uv[tcorner[t, s]], m.uv[tcorner[t, (s + 1) % 3]]
            return np.concatenate([p, q], 1), tmat[t]

        ss, smat = segments(src, scut_tri, scut_side, sc, _material(src, sf))
        rs, rmat = segments(res, cut_tri, cut_side, tc, _material(res, tf))
        if len(ss) and len(rs):
            fit = _seam_fit(ss, smat, rs, rmat, tex)
            num[_I["seam_fit_max"]] = fit.max()
            num[_I["seam_fit_p99"]] = _percentile(fit, 99)
        else:
            num[_I["seam_fit_max"]] = num[_I["seam_fit_p99"]] = 0.0

    # stretch and angle per result chart
    a2i = np.bincount(label, weights=a2, minlength=nisl)
    a3i = np.bincount(label, weights=a3, minlength=nisl)
    sgn_i = np.bincount(label, weights=a2s, minlength=nisl)
    ok = (a2 > AREA_MIN) & (a3 > 0)
    if ok.any():
        q, t = rq[ok], ru[ok]
        d = 2 * a2s[ok][:, None]
        s_, t_ = t[:, :, 0], t[:, :, 1]
        ss_ = (q[:, 0] * (t_[:, 1] - t_[:, 2])[:, None] + q[:, 1] * (t_[:, 2] - t_[:, 0])[:, None] + q[:, 2] * (t_[:, 0] - t_[:, 1])[:, None]) / d
        st_ = (q[:, 0] * (s_[:, 2] - s_[:, 1])[:, None] + q[:, 1] * (s_[:, 0] - s_[:, 2])[:, None] + q[:, 2] * (s_[:, 1] - s_[:, 0])[:, None]) / d
        ga, gb, gc = (ss_ * ss_).sum(1), (ss_ * st_).sum(1), (st_ * st_).sum(1)
        l2sq = 0.5 * (ga + gc)
        lab = label[ok]
        tc_ = np.sqrt(l2sq * (a2i / np.maximum(a3i, 1e-30))[lab])
        w = a3[ok]
        num[_I["stretch_med"]] = _wquantile(tc_, w, 0.5)
        num[_I["stretch_worst1"]] = _wquantile(tc_, w, 0.99)
        num[_I["stretch_over4_pct"]] = 100.0 * w[tc_ > STRETCH_BAD].sum() / w.sum()
        r_ = np.sqrt(((ga - gc) / 2) ** 2 + gb ** 2)
        l1, l2 = (ga + gc) / 2 + r_, np.maximum((ga + gc) / 2 - r_, 1e-30)
        num[_I["angle_p90"]] = _wquantile(np.sqrt(l1 / l2), w, 0.9)
        sw = np.where(seen[tf[ok]], w, 0.0)
        if (sw > 0).any():
            num[_I["stretch_worst1_seen"]] = _wquantile(tc_, sw, 0.99)
        if src is None:
            flipped_tri[ok] = np.sign(a2s[ok]) != np.sign(sgn_i[lab])
    num[_I["flipped_vs_source"]] = 1.0 if vs_source else 0.0
    flipped = np.zeros(nf, bool)
    flipped[tf[flipped_tri]] = True
    num[_I["flipped_faces"]] = flipped.sum()
    num[_I["flipped_seen_faces"]] = (flipped & seen).sum()

    # islands and pixel spread
    num[_I["islands"]] = nisl
    px = np.sqrt(a2i[a2i > 0]) * tex
    num[_I["island_min_px"]] = px.min() if len(px) else np.nan
    dens = np.ones(nf) if density is None or len(density) == 0 else np.asarray(density, dtype=np.float64)
    dsum = np.bincount(label, weights=dens[tf] * a3, minlength=nisl)
    good = (a3i > 0) & (a2i > 0)
    if good.any():
        v = np.sqrt(a2i[good] / a3i[good]) * tex / np.maximum(dsum[good] / a3i[good], 1e-30)
        num[_I["pixel_spread"]] = _wquantile(v, a3i[good], 0.95) / max(_wquantile(v, a3i[good], 0.05), 1e-30)

    # overlap and space used
    if nt:
        num[_I["overlap_pct"]], num[_I["space_used_pct"]] = _raster(res, tc, tf, ru, a3, allow_overlap, tex)

    # cut length and seams in sight
    total_area = float(a3.sum())
    cut_len = 0.0
    for ctri, cside in zip(cut_tri, cut_side):
        t, s = ctri[0], cside[0]
        cut_len += float(np.linalg.norm(res.points[res.flat[tc[t, s]]] - res.points[res.flat[tc[t, (s + 1) % 3]]]))
    if total_area > 0:
        num[_I["cut_per_sqrt_area"]] = cut_len / np.sqrt(total_area)
    ea, eb, eid, efaces = _edges(res)
    if seen_edge_px is not None and len(seen_edge_px) == len(ea) and len(ea) > 0:
        px_e = np.asarray(seen_edge_px, dtype=np.float64)
        len3 = np.linalg.norm(res.points[ea] - res.points[eb], axis=1)
        if seen_face is None:
            frac = np.ones(len(ea))
        else:
            frac = np.bincount(eid, weights=seen_face[efaces], minlength=len(ea)) / np.bincount(eid, minlength=len(ea))
        want = float((len3 * frac).sum())
        scale = px_e.sum() / want if want > 0 else float("nan")
        lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ea, eb))}
        seen_cut = all_cut = 0.0
        for ctri, cside in zip(cut_tri, cut_side):
            t, s = ctri[0], cside[0]
            va, vb = int(res.flat[tc[t, s]]), int(res.flat[tc[t, (s + 1) % 3]])
            i = lookup.get((min(va, vb), max(va, vb)))
            if i is not None:
                seen_cut += px_e[i]
                all_cut += len3[i] * scale
        if all_cut > 0:
            num[_I["seams_in_sight"]] = seen_cut / all_cut

    if faces_off and faces_off > 0:
        num[_I["extra_faces"]] = nf - faces_off
    if split is not None and len(split):
        num[_I["split_faces"]] = int(np.count_nonzero(split))
    _rows(res, tc, tf, label, col, row, on_chart, tex, num)
    return Check(num, straddle, flipped)


def _rows(res, tc, tf, label, col, row, on_chart, tex, num):
    nc = len(res.flat)
    if col is None or row is None or len(col) != nc or len(row) != nc:
        return
    col, row = np.asarray(col), np.asarray(row)
    face_isl = np.zeros(len(res.indptr) - 1, np.int64)
    face_isl[tf] = label
    corner_face = np.repeat(np.arange(len(res.indptr) - 1), np.diff(res.indptr))
    on = np.zeros(len(res.indptr) - 1, bool) if on_chart is None or len(on_chart) == 0 else np.asarray(on_chart, dtype=bool)
    sel = np.flatnonzero((row >= 0) & (col >= 0))
    groups: dict = {}
    for c in sel:
        groups.setdefault((int(face_isl[corner_face[c]]), int(row[c])), []).append((int(col[c]), c))
    on_rows = on_straight = 0
    bends = []
    for key in sorted(groups):
        pts = sorted(groups[key], key=lambda x: x[0])  # stable: first corner of a column wins, as in the kernel
        seen_cols, keep = set(), []
        for cc, c in pts:
            if cc not in seen_cols:
                seen_cols.add(cc)
                keep.append(c)
        if len(keep) < 3:
            continue
        uv = res.uv[keep]
        is_on = bool(on[corner_face[keep[0]]])
        a, b = uv[0], uv[-1]
        d = b - a
        ln = float(np.hypot(*d))
        w = uv - a
        off = np.abs(w[:, 0] * d[1] - w[:, 1] * d[0]) / ln if ln > 0 else np.hypot(w[:, 0], w[:, 1])
        if is_on:
            on_rows += 1
            on_straight += int(off.max() * tex <= ROW_LINE_TEXEL)
        else:
            u, v = uv[1:-1] - uv[:-2], uv[2:] - uv[1:-1]
            lu, lv = np.hypot(u[:, 0], u[:, 1]), np.hypot(v[:, 0], v[:, 1])
            okb = (lu > 0) & (lv > 0)
            cosv = np.clip((u * v).sum(1) / np.where(okb, lu * lv, 1.0), -1.0, 1.0)
            bends.append(float(np.degrees(np.arccos(cosv[okb])).max()) if okb.any() else 0.0)
    if on_rows:
        num[_I["straight_rows_pct"]] = 100.0 * on_straight / on_rows
    if bends:
        num[_I["row_bend_p95"]] = _percentile(np.array(bends), 95)


def as_dict(numbers) -> dict:
    return {name: float(numbers[i]) for i, name in enumerate(NAMES)}


# ---------------------------------------------------------------- verdict


def _rule(value, good, check, higher_is_worse=True):
    """0 GOOD, 1 CHECK, 2 BAD for one number; NaN is no problem."""

    if value is None or value != value:
        return 0
    if higher_is_worse:
        return 0 if value <= good else 1 if value <= check else 2
    return 0 if value >= good else 1 if value >= check else 2


def uv_verdict(numbers, mode: str = "new", source=None, seam_step=None) -> tuple:
    """(level, title, reasons): level GOOD / CHECK / BAD (the words of verdict.py). `numbers` and `source` are arrays in
    `NAMES` order or dicts. mode "keep" keeps the source's UVs (Faces across seams, Texture drift, relative rules against
    `source`); "new" makes new UVs. Problems found only on unseen faces never raise the level above CHECK."""

    n = as_dict(numbers) if not isinstance(numbers, dict) else numbers
    s = None if source is None else (as_dict(source) if not isinstance(source, dict) else source)
    c = UV_VERDICT
    keep = mode == "keep"
    found = []  # (severity, words, only_unseen)

    def add(sev, words, unseen=False):
        if sev:
            found.append((sev, words, unseen))

    faces = max(n["faces"], 1)
    if keep:
        add(_rule(n["straddle_pct"], c["across"]["good"], c["across"]["check"]), f"{n['straddle_pct']:.1f} % of faces cross a seam")
        add(_rule(n["drift_p99"], c["drift"]["good"], c["drift"]["check"]), f"texture drifts {n['drift_p99']:.1f} px")
    # flipped: counted on seen faces; the same count on all faces is capped at CHECK
    per_seen = 1000.0 * n["flipped_seen_faces"] / faces
    per_all = 1000.0 * n["flipped_faces"] / faces
    base = 0.0
    if keep and s is not None:
        base = 1000.0 * max(s["flipped_seen_faces"], 0) / max(s["faces"], 1)
    if per_seen > base:
        add(1 if per_seen <= c["flipped_per_1000"]["check"] else 2, f"{int(n['flipped_seen_faces'])} flipped faces in sight")
    elif per_all > base:
        add(1, f"{int(n['flipped_faces'])} flipped faces out of sight", True)
    ov = n["overlap_pct"]
    ov_base = s["overlap_pct"] if keep and s is not None and s["overlap_pct"] == s["overlap_pct"] else 0.0
    if ov == ov and ov > ov_base:
        add(1 if ov <= c["overlap_pct"]["check"] else 2, f"{ov:.2f} % of texels used twice")
    d = c["stretch_delta"]
    for key, label in (("stretch_med", "median stretch"), ("stretch_worst1", "worst 1 % stretch")):
        good, check = c[key]["good"], c[key]["check"]
        if keep and s is not None and s[key] == s[key]:
            good, check = s[key] + d["med" if key == "stretch_med" else "worst1"], max(check, s[key] + d["med" if key == "stretch_med" else "worst1"])
        value = n[key]
        sev = _rule(value, good, check)
        if key == "stretch_worst1" and sev == 2 and n["stretch_worst1_seen"] == n["stretch_worst1_seen"]:
            seen_sev = _rule(n["stretch_worst1_seen"], good, check)
            if seen_sev < sev:
                add(1, f"{label} {value:.2f}x out of sight", True)
                continue
        add(sev, f"{label} {value:.2f}x")
    if not keep:
        add(_rule(n["space_used_pct"], c["space"]["good"], c["space"]["check"], False), f"only {n['space_used_pct']:.0f} % of the page used")
        add(_rule(n["seams_in_sight"], c["in_sight"]["good"], c["in_sight"]["check"]), f"{100 * n['seams_in_sight']:.0f} % of seams in sight" if n["seams_in_sight"] == n["seams_in_sight"] else "")
        add(_rule(n["straight_rows_pct"], c["straight_rows"]["good"], c["straight_rows"]["check"], False), f"only {n['straight_rows_pct']:.0f} % of rows straight" if n["straight_rows_pct"] == n["straight_rows_pct"] else "")
    spread_good, spread_check = c["spread"]["good"], c["spread"]["check"]
    if keep and s is not None and s["pixel_spread"] == s["pixel_spread"]:
        spread_good = s["pixel_spread"] + c["spread_delta"]
        spread_check = max(spread_check, spread_good)
    add(_rule(n["pixel_spread"], spread_good, spread_check), f"pixel size varies {n['pixel_spread']:.1f}x")
    if seam_step is not None:
        add(_rule(seam_step, c["seam_step"]["good"], c["seam_step"]["check"]), f"seams step {seam_step:.1f} degrees after the bake")
    level = 0
    words = ""
    for sev, text, unseen in found:
        sev = min(sev, 1) if unseen else sev
        if sev > level or (sev == level and not words):
            level, words = sev, text
    reasons = [text for _, text, _ in found]
    if level == 0:
        return "GOOD", "Good for baking" if not keep else "Good: UVs kept", reasons
    return ("CHECK", f"Check: {words}", reasons) if level == 1 else ("BAD", f"Bad: {words}", reasons)
