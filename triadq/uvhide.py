# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Seams out of sight (R7): once the charts of a level are made, every border between two charts slides onto nearby
edges where it shows less, by giving faces from one of the two charts to the other. The NumPy twin of the compiled
`uvhide` kernel: same constants, same steps, same order. Our own code, written from:

- Where a seam goes between two fixed regions: the cheapest set of edges that separates them is a minimum cut of the
  face graph (Boykov, Jolly 2001, "Interactive graph cuts for optimal boundary and region segmentation"; Kwatra,
  Schoedl, Essa, Turk, Bobick 2003, "Graphcut textures"). Maximum flow by blocking flows on a layered graph (Dinic
  1970); the cut is the side the source still reaches in the residual graph (Ford, Fulkerson 1956), the same for every
  maximum flow.
- What an edge costs: the merge tree's cut cost (visibility-weighted seams: Sheffer, Hart 2002, "Seamster").
- Charts stay disks (Euler characteristic over corner wedges, one piece); a chart left with holes gets slits, each the
  cheapest path of inner edges between two border loops (Dijkstra 1959).
- Charts keep their stretch, measured as the UV Check does (Sander, Snyder, Gortler, Hoppe 2001 L2), and the surface
  stretched past `L2_WORST` stays within `OVER_SHARE` of the mesh; one-to-one layouts (flip-free disk with a border
  that does not cross itself, Floater 2003); boxes by rotating calipers (Toussaint 1983) on a monotone-chain hull
  (Andrew 1979); rigid fits by Kabsch's 2D case.
- Ours: the band of rings around the border that may change sides, the pairs worked in rounds, the replay at every
  level (a face that changed sides follows the chart of its anchor).

One call per level (`apply`, from `ChartTree._level`); the moves are made once per tree on its full level and kept.
Pure Python loops: fine for the tests and small meshes; the compiled kernel is the fast path.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

from . import uvflat
from .uvchart import FREE_FACES, _components, _Dsu

# Constants table (the compiled kernel holds the same names and values).
RINGS = (32, 16, 8, 4, 2, 1)
CAP_SCALE = 1024.0
L2_MEDIAN = 1.08
L2_WORST = 1.3
OVER_SHARE = 0.005
FREE_RINGS = 2
PAGE_SLACK = 0.1
FULL_STEPS = 10
FAN_MARGIN = 0.01
KEEP_TREES = 8

NONE = -1
INF = math.inf


class Hide:
    __slots__ = ("anchor", "changed", "start", "slit", "moved")

    def __init__(self, anchor, changed, start, slit, moved):
        self.anchor, self.changed, self.start, self.slit, self.moved = anchor, changed, start, slit, moved


def of(tree) -> Hide:
    """The moves of `tree`, made on its first level and kept on the tree."""

    h = getattr(tree, "_hide", None)
    if h is None:
        h = _drive(moves(tree))
        tree._hide = h
    return h


def _drive(gen):
    """Run a sliced generator to its end; its return value."""

    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


def _flat(pts, ip, fl, start, **options):
    """`uvflat.Flatten` driven in its own slices: its UVs flat, or None when the chart cannot be laid out."""

    stage = uvflat.Flatten(pts, ip, fl, start, use_fast=False, budget=0.001, **options)  # short steps: this runs inside a caller's slice
    try:
        while not stage.done:
            stage.step()
            yield
    except ValueError:
        return None
    return np.asarray(stage.uv, np.float64).reshape(-1)


def apply(tree, chart, cut):
    """Slide the borders of a level's charts: (chart per face, cut per edge, charts, charts that are not disks), or
    None when no move was made."""

    return _drive(apply_steps(tree, chart, cut))


class _Arrays:
    """A tree's topology as arrays, made once per tree (the moves and the replay of the moves at every level)."""

    def _build(self, tree):
        """Generator: the arrays, a few at a time."""

        t = tree.topo
        ip = self.ip = np.asarray(t.indptr, np.int64)
        self.flat = np.asarray(t.flat, np.int64)
        yield
        sizes = np.diff(ip)
        self.cface = np.repeat(np.arange(t.nf), sizes)
        nxt = self.cnext = np.arange(len(self.flat)) + 1
        last = ip[1:] - 1
        nxt[last[sizes > 0]] = ip[:-1][sizes > 0]
        self.cedge = np.asarray(t.cedge, np.int64)
        yield
        use = np.asarray(t.euse, np.int64).reshape(-1, 2)
        yield
        h0, h1 = self.h0_all, self.h1_all = use[:, 0], use[:, 1]
        self.inner_mask = (h1 >= 0) & ~np.asarray(t.wall, bool)
        self.af0 = self.cface[h0]  # the two faces of every edge (-1: none)
        self.af1 = np.where(h1 >= 0, self.cface[np.maximum(h1, 0)], -1)
        self.ea_all, self.eb_all = np.asarray(t.ea, np.int64), np.asarray(t.eb, np.int64)
        yield
        self.inner = np.flatnonzero(self.inner_mask)
        h0, h1 = h0[self.inner], h1[self.inner]
        self.f0, self.f1 = self.cface[h0], self.cface[h1]
        ea, eb = self.ea_all[self.inner], self.eb_all[self.inner]
        # the corner pairs each inner edge glues (one per end): corner of h0 at v, corner of h1 at v
        self.ca = np.concatenate([np.where(self.flat[h0] == v, h0, nxt[h0]) for v in (ea, eb)])
        self.cb = np.concatenate([np.where(self.flat[h1] == v, h1, nxt[h1]) for v in (ea, eb)])
        self.fixed_all = np.asarray(tree.fixed, bool)
        self.fixed = self.fixed_all[self.inner]


def _arrays_steps(tree):
    """Generator: the tree's `_Arrays` (made once and kept on the tree)."""

    got = getattr(tree, "_hide_arrays", None)
    if got is None:
        got = _Arrays()
        yield from got._build(tree)
        tree._hide_arrays = got
    return got


def _arrays(tree):
    return _drive(_arrays_steps(tree))


def _parts_and_chi(t, z, new_cut):
    """Charts (numbered by their first face) of the faces joined across the uncut inner edges, and the Euler
    characteristic of each (edges + faces - corners + wedges), on arrays."""

    glue = ~new_cut[z.inner]
    lab = _components(t.nf, z.f0[glue], z.f1[glue])
    roots, ch = np.unique(lab, return_inverse=True)
    n = len(roots)
    g2 = np.concatenate([glue, glue])
    wedge = _components(len(z.flat), z.ca[g2], z.cb[g2])
    own = wedge == np.arange(len(z.flat))
    chi = (np.bincount(ch[z.f0[glue]], minlength=n) + np.bincount(ch, minlength=n)
           - np.bincount(ch[z.cface], minlength=n) + np.bincount(ch[z.cface[own]], minlength=n))
    return ch, n, chi


def apply_steps(tree, chart, cut):
    """`apply` as a generator, yielding between short pieces."""

    h = of(tree)
    if h.moved == 0:
        return None
    t = tree.topo
    z = _arrays(tree)
    yield
    raw = np.asarray(chart, np.int64)
    cut0 = np.asarray(cut, bool)
    anchor = np.asarray(h.anchor, np.int64)
    lab = np.where(anchor == NONE, raw, raw[np.maximum(anchor, 0)])
    e, f0, f1 = z.inner, z.f0, z.f1
    new_cut = cut0.copy()
    new_cut[e] = z.fixed | (cut0[e] & (raw[f0] == raw[f1])) | np.asarray(h.slit, bool)[e] | (lab[f0] != lab[f1])
    yield
    ch, n, chi = _parts_and_chi(t, z, new_cut)
    yield
    if (chi != 1).any():
        # below the level the moves were made on, a chart can close round a hole: there it keeps this level's own
        # borders as well (still inside the charts above)
        bad = (chi[ch[f0]] != 1) | (chi[ch[f1]] != 1)
        new_cut[e[bad & (raw[f0] != raw[f1])]] = True
        ch, n, chi = _parts_and_chi(t, z, new_cut)
        yield
    not_disk = int((chi != 1).sum())
    return ch.astype(np.int64), new_cut, n, not_disk


class _Mesh:
    def __init__(self, tree, z, fedge, w, free, aw, afree, fe_ptr, fe_edge):
        self.tree, self.t, self.fedge, self.w, self.free = tree, tree.topo, fedge, w, free
        self.points = np.asarray(tree.points, np.float64).reshape(-1, 3)
        self.f0, self.f1 = z.af0.tolist(), z.af1.tolist()  # the two faces of each edge (asked for inner edges only)
        # the same as arrays, for the per-chart work done on whole arrays (`_lay`, `_cuts`)
        self.ip, self.fl, self.cedge, self.cnext = z.ip, z.flat, z.cedge, z.cnext
        self.h0, self.h1, self.ea, self.eb = z.h0_all, z.h1_all, z.ea_all, z.eb_all
        self.af0, self.af1, self.inner = z.af0, z.af1, z.inner_mask
        self.afree, self.aw = afree, aw  # kept equal to `free` and `w` (slits are added to both)
        self.fe_ptr, self.fe_edge = fe_ptr, fe_edge  # each face's inner edges, in corner order
        self.alabel = None  # the chart of each face, as an array (set by `moves`, kept equal to its list)
        self.slit_at = np.zeros(len(self.ea), bool)  # scratch for `_lay` (left all False)

    def other(self, e, f):
        a = self.f0[e]
        return self.f1[e] if a == f else a


class _Laid:
    __slots__ = ("corners", "vert", "uv", "q", "bx", "mesh", "sides", "grid", "slit", "slit_cost")

    def __init__(self, corners, vert, uv, q, mesh, sides, grid, slit, slit_cost):
        self.corners, self.vert, self.uv, self.q, self.mesh, self.sides, self.grid = corners, vert, uv, q, mesh, sides, grid
        self.slit, self.slit_cost = slit, slit_cost
        self.bx = _min_box(uv, _border_points(sides))

    def confirm(self, allowed, limit) -> bool:
        """Lay the chart out the way the level's flatten will (from this layout) and keep it when still one-to-one
        and within `limit`."""

        if self.grid:
            return True
        yield
        pts, ip, fl = self.mesh
        uv = yield from _flat(pts, ip, fl, self.uv.reshape(-1, 2), rows="auto", start_mode="given", stretch_allowed=allowed)
        if uv is None:
            return False
        yield
        if not np.isfinite(uv).all() or uvflat.count_flips(ip, fl, uv) != 0 or not _fans_ok(pts, ip, fl, uv, FAN_MARGIN):
            return False
        yield
        if _border_crosses(uv, self.sides):
            return False
        yield
        q = _l2(pts, ip, fl, uv)
        if not _within(q, limit):
            return False
        yield
        self.bx, self.uv, self.q = _min_box(uv, _border_points(self.sides)), uv, q
        return True


def _fmax(a, b):
    """max that skips a NaN (as the kernel's float max)."""

    if a != a:
        return b
    if b != b:
        return a
    return a if a >= b else b


def _border_points(sides):
    """The vertices on a chart's border sides: the hull of a one-to-one layout is the hull of its border."""

    return sorted({v for side in sides for v in side})


def _min_box(uv, only=None):
    """Smallest-area box around points (x, y pairs; `only`: just these points, e.g. a laid-out chart's border):
    (area, longer side)."""

    ids = range(len(uv) // 2) if only is None else only
    p = sorted(set((float(uv[2 * i]), float(uv[2 * i + 1])) for i in ids))
    if len(p) < 3:
        return (0.0, 0.0)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    h: list = []
    for seq in (p, p[::-1]):
        start = len(h)
        for x in seq:
            while len(h) >= start + 2 and cross(h[-2], h[-1], x) <= 0.0:
                h.pop()
            h.append(x)
        h.pop()
    hx, hy = np.array([x[0] for x in h]), np.array([x[1] for x in h])
    best = (INF, 0.0)
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        ln = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        if not ln > 0.0:
            continue
        d = ((b[0] - a[0]) / ln, (b[1] - a[1]) / ln)
        r0, r1 = hx - a[0], hy - a[1]
        s = r0 * d[0] + r1 * d[1]
        lo, hi = float(s.min()), float(s.max())
        top = max(0.0, float(np.abs(r0 * d[1] - r1 * d[0]).max()))
        area = (hi - lo) * top
        if area < best[0]:
            best = (area, _fmax(hi - lo, top))
    return best


def _fan_tris(pts, ip, fl):
    """The fan triangles (first corner, i, i + 1) of every face, in face order: three index arrays."""

    ip, fl = np.asarray(ip, np.int64), np.asarray(fl, np.int64)
    sizes = np.diff(ip)
    nt = np.maximum(sizes - 2, 0)
    face = np.repeat(np.arange(len(sizes)), nt)
    k = np.arange(len(face)) - np.repeat(np.cumsum(nt) - nt, nt) + 1  # 1 .. size - 2 within each face
    first = ip[:-1][face]
    return fl[first], fl[first + k], fl[first + k + 1]


def _tri_geometry(pts, ip, fl):
    """Per fan triangle: corners (p0, p1, p2), first side e1 = x1 - x0, second e2 = x2 - x0, and the surface area,
    each number made with the same operations in the same order as one triangle at a time."""

    p0, p1, p2 = _fan_tris(pts, ip, fl)
    x0, x1, x2 = pts[p0], pts[p1], pts[p2]
    e1, e2 = x1 - x0, x2 - x0
    c0 = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
    c1 = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]
    c2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    area = 0.5 * np.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
    return p0, p1, p2, e1, e2, area


def _seq_sum(x) -> float:
    """Left-to-right sum (as a running total in a loop), not NumPy's pairwise one."""

    return float(np.add.accumulate(x)[-1]) if len(x) else 0.0


def _fans_ok(pts, ip, fl, uv, margin) -> bool:
    uv = np.asarray(uv, np.float64)
    p0, p1, p2, _, _, a3 = _tri_geometry(pts, ip, fl)
    u0, v0, u1, v1, u2, v2 = uv[2 * p0], uv[2 * p0 + 1], uv[2 * p1], uv[2 * p1 + 1], uv[2 * p2], uv[2 * p2 + 1]
    a2 = 0.5 * ((u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0))
    if not (a2 > 0.0).all():
        return False
    s2, s3 = _seq_sum(a2), _seq_sum(a3)
    scale = s2 / s3 if s3 > 0.0 else 0.0
    return bool((a2 >= margin * scale * a3).all())


def _within(q, limit) -> bool:
    return q[0] <= limit[0] and q[1] <= limit[1]


def _l2(pts, ip, fl, uv):
    """Sander et al. L2 of the map from the layout to the surface per fan triangle, the layout's area scaled to the
    surface's: (area-weighted median, worst 1 %, surface area above `L2_WORST`)."""

    uv = np.asarray(uv, np.float64)
    p0, p1, p2, e1, e2, area = _tri_geometry(pts, ip, fl)
    l1 = np.sqrt(e1[:, 0] * e1[:, 0] + e1[:, 1] * e1[:, 1] + e1[:, 2] * e1[:, 2])
    use = (l1 > 0.0) & (area > 0.0)
    p0, p1, p2, e1, e2, area, l1 = p0[use], p1[use], p2[use], e1[use], e2[use], area[use], l1[use]
    bx = (e1[:, 0] * e2[:, 0] + e1[:, 1] * e2[:, 1] + e1[:, 2] * e2[:, 2]) / l1
    by = 2.0 * area / l1
    d10, d11 = uv[2 * p1] - uv[2 * p0], uv[2 * p1 + 1] - uv[2 * p0 + 1]
    d20, d21 = uv[2 * p2] - uv[2 * p0], uv[2 * p2 + 1] - uv[2 * p0 + 1]
    a3 = _seq_sum(area)
    det = d10 * d21 - d11 * d20
    a2 = _seq_sum(0.5 * np.abs(det))
    if not (a3 > 0.0 and a2 > 0.0):
        return (INF, INF, INF)
    s = math.sqrt(a2 / a3)
    ok = det > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(ok, det, 1.0)
        i00, i01, i10, i11 = d21 / dd, -d20 / dd, -d11 / dd, d10 / dd
        m00, m01 = l1 * i00 + bx * i10, l1 * i01 + bx * i11
        m10, m11 = by * i10, by * i11
        vals = np.where(ok, s * np.sqrt((m00 * m00 + m01 * m01 + m10 * m10 + m11 * m11) / 2.0), INF)
    over = _seq_sum(area[vals > L2_WORST])
    return (_wq(vals, area, 0.5), _wq(vals, area, 0.99), over)


def _wq(values, w, q):
    if len(values) == 0:
        return math.nan
    w = np.asarray(w, np.float64)
    order = np.argsort(values, kind="stable")
    acc = np.add.accumulate(w[order])
    hit = np.flatnonzero(acc >= q * _seq_sum(w))  # the total in face order, the running sum in value order
    return float(values[order[hit[0]]]) if len(hit) else float(values[order[-1]])


def _align(pairs):
    """Rigid 2D fit q = R p + t by least squares: ((cos, sin), t)."""

    k = float(len(pairs))
    cp, cq = [0.0, 0.0], [0.0, 0.0]
    for p, q in pairs:
        for r in range(2):
            cp[r] += p[r] / k
            cq[r] += q[r] / k
    sxx = sxy = 0.0
    for p, q in pairs:
        a, b = (p[0] - cp[0], p[1] - cp[1]), (q[0] - cq[0], q[1] - cq[1])
        sxx += a[0] * b[0] + a[1] * b[1]
        sxy += a[0] * b[1] - a[1] * b[0]
    r = math.hypot(sxx, sxy)
    c = (sxx / r, sxy / r) if r > 0.0 else (1.0, 0.0)
    return c, (cq[0] - (c[0] * cp[0] - c[1] * cp[1]), cq[1] - (c[1] * cp[0] + c[0] * cp[1]))


def _border_crosses(uv, sides) -> bool:
    """Whether two border sides that share no vertex cross (touching does not count), through a grid of cells: the
    sides are put in every cell their box covers and the pairs that meet in a cell are tested, all on arrays."""

    if len(sides) < 4:
        return False
    uv = np.asarray(uv, np.float64)
    sd = np.asarray(sides, np.int64).reshape(-1, 2)
    a, b = sd[:, 0], sd[:, 1]
    ax, ay, bx, by = uv[2 * a], uv[2 * a + 1], uv[2 * b], uv[2 * b + 1]
    lo0, lo1 = min(float(ax.min()), float(bx.min())), min(float(ay.min()), float(by.min()))
    cell = max(_seq_sum(np.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by))) / len(sd), 1e-12)
    x0 = np.floor((np.minimum(ax, bx) - lo0) / cell).astype(np.int64)
    x1 = np.floor((np.maximum(ax, bx) - lo0) / cell).astype(np.int64)
    y0 = np.floor((np.minimum(ay, by) - lo1) / cell).astype(np.int64)
    y1 = np.floor((np.maximum(ay, by) - lo1) / cell).astype(np.int64)
    ny = y1 - y0 + 1
    count = (x1 - x0 + 1) * ny
    seg = np.repeat(np.arange(len(sd)), count)
    k = np.arange(len(seg)) - np.repeat(np.cumsum(count) - count, count)
    cx, cy = x0[seg] + k // ny[seg], y0[seg] + k % ny[seg]
    key = cx * (int(cy.max()) + 1) + cy
    order = np.argsort(key, kind="stable")
    key, seg = key[order], seg[order]
    end = np.searchsorted(key, key, side="right")  # one past the last entry of each entry's cell
    after = end - np.arange(len(key)) - 1  # entries after it in its cell
    if not after.any():
        return False
    i = np.repeat(np.arange(len(key)), after)
    j = i + 1 + (np.arange(len(i)) - np.repeat(np.cumsum(after) - after, after))
    p, q = seg[i], seg[j]
    pa, pb, pc, pd = a[p], b[p], a[q], b[q]
    ok = (pa != pc) & (pa != pd) & (pb != pc) & (pb != pd)
    p, q = p[ok], q[ok]
    ax_, ay_, bx_, by_ = ax[p], ay[p], bx[p], by[p]
    cx_, cy_, dx_, dy_ = ax[q], ay[q], bx[q], by[q]

    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    hit = (orient(ax_, ay_, bx_, by_, cx_, cy_) * orient(ax_, ay_, bx_, by_, dx_, dy_) < 0.0) &           (orient(cx_, cy_, dx_, dy_, ax_, ay_) * orient(cx_, cy_, dx_, dy_, bx_, by_) < 0.0)
    return bool(hit.any())


def _band(ip, fl, n, seeds, k):
    """The vertices within `k` rings (by shared faces) of `seeds`."""

    ip, fl = np.asarray(ip, np.int64), np.asarray(fl, np.int64)
    face = np.repeat(np.arange(len(ip) - 1), np.diff(ip))
    inside = np.zeros(n, bool)
    inside[np.asarray(list(seeds), np.int64)] = True
    for _ in range(k):
        touch = np.zeros(len(ip) - 1, bool)
        touch[face[inside[fl]]] = True
        inside[fl[touch[face]]] = True
    return inside.tolist()


def _quick_stretch(pts, ip, fl, uv, free) -> bool:
    """ARAP (rows off) on the `free` vertices with the rest held, until the energy settles; `uv` in place."""

    n = len(pts)
    el = uvflat.Elements(pts, ip, fl)
    pins = [v for v in range(n) if not free[v]] or [0]
    A, B, W = el.a, el.b, el.w
    system = uvflat._System(n, np.concatenate([A, B, A, B]), np.concatenate([A, B, B, A]),
                            np.concatenate([W, W, -W, -W]), pins)
    x2 = np.asarray(uv, np.float64).reshape(-1, 2)
    span = max(float(np.abs(x2).max()) if x2.size else 0.0, 1e-300)
    tol = uvflat.STOP_TEXEL_SHARE * span / uvflat.TEXTURE_PX
    zero = 1e-20 * float((el.w * (el.e ** 2).sum(1)).sum())
    nel = el.nfaces
    info: dict = {}
    steps, last = 0, INF
    while True:
        du = x2[el.b] - x2[el.a]
        m00 = np.bincount(el.elem, weights=el.w * du[:, 0] * el.e[:, 0], minlength=nel)
        m01 = np.bincount(el.elem, weights=el.w * du[:, 0] * el.e[:, 1], minlength=nel)
        m10 = np.bincount(el.elem, weights=el.w * du[:, 1] * el.e[:, 0], minlength=nel)
        m11 = np.bincount(el.elem, weights=el.w * du[:, 1] * el.e[:, 1], minlength=nel)
        rx, ry = m00 + m11, m10 - m01
        hh = np.sqrt(rx * rx + ry * ry)
        zh = hh == 0.0
        hh = np.where(zh, 1.0, hh)
        c, s = np.where(zh, 1.0, rx / hh)[el.elem], np.where(zh, 0.0, ry / hh)[el.elem]
        re = np.stack([c * el.e[:, 0] - s * el.e[:, 1], s * el.e[:, 0] + c * el.e[:, 1]], 1)
        energy = float((el.w * ((du - re) ** 2).sum(1)).sum())
        if (steps > 0 and abs(last - energy) <= uvflat.ARAP_CHANGE * max(last, 1e-300)) or energy <= zero:
            break
        last = energy
        new = x2.copy()
        for axis in range(2):
            b = np.bincount(el.b, weights=el.w * re[:, axis], minlength=n) - np.bincount(el.a, weights=el.w * re[:, axis], minlength=n)
            x = x2[:, axis].copy()
            yield from system.cg(system.rhs(b, x), x, tol, info)
            new[:, axis] = x
        tstep = uvflat._injective_step(el, x2, new - x2)
        x2 += tstep * (new - x2)
        steps += 1
        yield
    uv[:] = x2.reshape(-1)


def _bridge(m, slot, vert, nv, side_a, side_b, ge, chi):
    """For a chart that is a disk with holes (chi = 2 - loops): the cheapest path of inner edges from the border loop
    of the lowest vertex to another loop, through no other border vertex; None for any other shape. Generator.
    vert: the chart vertex of each corner (array); side_a, side_b: its border sides; ge: its glued edges."""

    # border loops by union-find (label hooking would need a round per step along a long loop)
    d = _Dsu(nv)
    for i, (x, y) in enumerate(zip(side_a.tolist(), side_b.tolist())):
        if i & 2047 == 2047:
            yield
        d.union(x, y)
    loops_np = np.asarray([d.find(v) for v in range(nv)], np.int64)
    on_np = np.zeros(nv, bool)
    on_np[side_a] = on_np[side_b] = True
    b = len(np.unique(loops_np[on_np]))
    if b < 2 or chi != 2 - b:
        return None
    yield
    es = np.unique(ge)  # the glued edges, in order
    h0 = m.h0[es]
    va = vert[slot[np.where(m.fl[h0] == m.ea[es], h0, m.cnext[h0])]]
    vb = vert[slot[np.where(m.fl[h0] == m.eb[es], h0, m.cnext[h0])]]
    node, to, edge = np.concatenate([va, vb]), np.concatenate([vb, va]), np.concatenate([es, es])
    order = np.lexsort((edge, node))  # per vertex, its edges in order (as appended one edge at a time)
    to_l, edge_l = to[order].tolist(), edge[order].tolist()
    at = np.searchsorted(node[order], np.arange(nv + 1)).tolist()
    adj = [list(zip(to_l[at[v] : at[v + 1]], edge_l[at[v] : at[v + 1]])) for v in range(nv)]
    on, loop_of = on_np.tolist(), loops_np.tolist()
    yield
    start = next((v for v in range(nv) if on[v]), None)
    if start is None:
        return None
    home = loop_of[start]
    dist = [None] * nv
    prev = [(NONE, NONE)] * nv
    heap: list = []
    for v in range(nv):
        if on[v] and loop_of[v] == home:
            dist[v] = 0
            heapq.heappush(heap, (0, v))
    pops = 0
    while heap:
        pops += 1
        if pops & 2047 == 0:
            yield
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        if on[v] and loop_of[v] != home:
            path, x = [], v
            while prev[x][0] != NONE:
                path.append(prev[x][1])
                x = prev[x][0]
            return path
        if on[v] and d > 0:
            continue
        for o, e in adj[v]:
            if on[o] and loop_of[o] == home:
                continue
            nd = d + m.w[e]
            if dist[o] is None or nd < dist[o]:
                dist[o] = nd
                prev[o] = (v, e)
                heapq.heappush(heap, (nd, o))
    return None


def _lay(m, faces, lab, cur, before, touched, limit):
    """A chart as its own mesh when it is one disk (slits added for holes), laid out: from its exact grid, else from
    the current layouts with what changed sides refitted, else again from scratch; None when it cannot be."""

    t, tree = m.t, m.tree
    F = np.asarray(faces, np.int64)
    nfc = len(F)
    sizes = m.ip[F + 1] - m.ip[F]
    nc = int(sizes.sum())
    offs = np.cumsum(sizes) - sizes
    corners = np.repeat(m.ip[F] - offs, sizes) + np.arange(nc)  # the chart's corners, face by face
    owner = np.repeat(np.arange(nfc), sizes)  # local face of each corner
    slot = np.full(len(m.fl), -1, np.int64)
    slot[corners] = np.arange(nc)
    fidx = np.full(t.nf, -1, np.int64)
    fidx[F] = np.arange(nfc)
    ce = m.cedge[corners]
    fo = F[owner]
    other = np.where(m.af0[ce] == fo, m.af1[ce], m.af0[ce])
    cand = m.inner[ce] & (other > fo) & ~m.afree[ce] & (fidx[np.maximum(other, 0)] >= 0)
    ip = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    nxt_local = np.arange(1, nc + 1)  # the next corner in its face, in the chart's numbering
    ends = ip[1:] - 1
    nxt_local[ends[sizes > 0]] = ip[:-1][sizes > 0]
    yield
    slit: list = []
    try:
        while True:
            glue = cand & ~m.slit_at[ce] if slit else cand
            ge = ce[glue]
            glued = len(ge)
            inside = set(ge.tolist())
            h0, h1 = m.h0[ge], m.h1[ge]
            ca, cb = [], []
            for v in (m.ea[ge], m.eb[ge]):
                ca.append(slot[np.where(m.fl[h0] == v, h0, m.cnext[h0])])
                cb.append(slot[np.where(m.fl[h1] == v, h1, m.cnext[h1])])
            fpart = _components(nfc, owner[glue], fidx[other[glue]])
            yield
            if nfc and (fpart != 0).any():
                return None
            part = _components(nc, np.concatenate(ca), np.concatenate(cb))
            roots = np.flatnonzero(part == np.arange(nc))  # the first corner of each vertex, in corner order
            vert = np.searchsorted(roots, part)
            nv = len(roots)
            chi = nv - (nc - glued) + nfc
            yield
            if chi == 1:
                break
            on_border = ~np.isin(ce, ge)
            path = yield from _bridge(m, slot, vert, nv, vert[on_border], vert[nxt_local[on_border]], ge, chi)
            if path is None:
                return None
            slit.extend(path)
            m.slit_at[path] = True
    finally:
        m.slit_at[slit] = False
    slit_cost = sum(m.w[e] for e in slit)
    pts = m.points[m.fl[corners[roots]]]
    fl = vert
    border = np.ones(nc, bool)
    if inside:
        border = ~np.isin(ce, np.fromiter(inside, np.int64, len(inside)))
    sides = list(zip(vert[border].tolist(), vert[nxt_local[border]].tolist()))
    yield

    def good(uv):  # generator: each check in its own piece
        yield
        if not np.isfinite(uv).all() or uvflat.count_flips(ip, fl, uv) != 0 or not _fans_ok(pts, ip, fl, uv, FAN_MARGIN):
            return None
        yield
        if _border_crosses(uv, sides):
            return None
        yield
        q = _l2(pts, ip, fl, uv)
        return q if math.isfinite(q[1]) else None

    def fin(uv, q, grid):  # generator: the box in its own piece
        yield
        return _Laid(corners, vert, uv, q, (pts, ip, fl), sides, grid, list(slit), slit_cost)

    yield
    g = uvflat.grid_layout(pts, ip, fl)
    yield
    if g is not None:
        uv = yield from _flat(pts, ip, fl, g[0], col=g[1], row=g[2], rows="auto", start_mode="given", stretch_allowed=tree.build_stretch)
        if uv is None:
            return None
        q = yield from good(uv)
        if q is None or not _within(q, limit):
            return None
        return (yield from fin(uv, q, True))
    came_f = before[F] != lab
    came = came_f[owner]
    cu, cv = cur[2 * corners], cur[2 * corners + 1]
    sb = [np.bincount(vert[came], weights=x[came], minlength=nv) for x in (cu, cv)] + [np.bincount(vert[came], minlength=nv).astype(np.float64)]
    sa = [np.bincount(vert[~came], weights=x[~came], minlength=nv) for x in (cu, cv)] + [np.bincount(vert[~came], minlength=nv).astype(np.float64)]
    both = np.flatnonzero((sa[2] > 0.0) & (sb[2] > 0.0))
    pairs = list(zip(zip((sb[0][both] / sb[2][both]).tolist(), (sb[1][both] / sb[2][both]).tolist()),
                     zip((sa[0][both] / sa[2][both]).tolist(), (sa[1][both] / sa[2][both]).tolist())))
    any_came = bool(came_f.any())
    if not any_came or len(pairs) >= 2:
        c, tr = _align(pairs) if any_came else ((1.0, 0.0), (0.0, 0.0))
        pu, pv = cu.copy(), cv.copy()
        pu[came] = c[0] * cu[came] - c[1] * cv[came] + tr[0]
        pv[came] = c[1] * cu[came] + c[0] * cv[came] + tr[1]
        cnt = np.bincount(vert, minlength=nv).astype(np.float64)
        uv = np.empty(2 * nv)
        uv[0::2] = np.bincount(vert, weights=pu, minlength=nv) / cnt
        uv[1::2] = np.bincount(vert, weights=pv, minlength=nv) / cnt
        if np.isfinite(uv).all() and uvflat.count_flips(ip, fl, uv) == 0:
            seeds = vert[np.isin(m.fl[corners], np.fromiter(touched, np.int64, len(touched)))].tolist() if touched else []
            if seeds:
                yield from _quick_stretch(pts, ip, fl, uv, _band(ip, fl, nv, seeds, FREE_RINGS))
            q = yield from good(uv)
            if q is not None and _within(q, limit):
                return (yield from fin(uv, q, False))
    # else the whole chart laid out again from scratch
    uv = yield from _flat(pts, ip, fl, None, rows="off", start_mode="free", steps=FULL_STEPS)
    if uv is None:
        return None
    q = yield from good(uv)
    if q is None:
        return None
    return (yield from fin(uv, q, False))


class _Flow:
    """Max flow by blocking flows (Dinic 1970), undirected, whole-number capacities."""

    def __init__(self, n):
        self.head, self.to, self.nxt, self.cap = [NONE] * n, [], [], []

    @classmethod
    def from_arcs(cls, n, x, y, c):
        """The graph `add(x[k], y[k], c[k])` for k in order would make, built on arrays: arc 2k goes x -> y, 2k + 1
        back; each arc's next is the arc added before it from the same node."""

        g = cls.__new__(cls)
        q = len(x)
        src, to = np.empty(2 * q, np.int64), np.empty(2 * q, np.int64)
        src[0::2], src[1::2] = x, y
        to[0::2], to[1::2] = y, x
        order = np.argsort(src, kind="stable")
        ss = src[order]
        same = np.zeros(2 * q, bool)
        same[1:] = ss[1:] == ss[:-1]
        nxt = np.full(2 * q, NONE, np.int64)
        nxt[order[same]] = order[np.flatnonzero(same) - 1]
        head = np.full(n, NONE, np.int64)
        last = np.ones(2 * q, bool)
        last[:-1] = ~same[1:]
        head[ss[last]] = order[last]
        g.head, g.to, g.nxt, g.cap = head.tolist(), to.tolist(), nxt.tolist(), np.repeat(np.asarray(c, np.int64), 2).tolist()
        return g

    def add(self, a, b, c):
        to, head = self.to, self.head
        k = len(to)
        to += (b, a)
        self.cap += (c, c)
        self.nxt += (head[a], head[b])
        head[a], head[b] = k, k + 1

    def _levels(self, s):
        """Generator: breadth-first levels from s over arcs with room left."""

        head, to, nxt, cap = self.head, self.to, self.nxt, self.cap
        lv = [NONE] * len(head)
        lv[s] = 0
        q = deque([s])
        pop, push = q.popleft, q.append
        pops = 0
        while q:
            pops += 1
            if pops & 1023 == 0:
                yield
            x = pop()
            up = lv[x] + 1
            a = head[x]
            while a != NONE:
                y = to[a]
                if cap[a] > 0 and lv[y] == NONE:
                    lv[y] = up
                    push(y)
                a = nxt[a]
        return lv

    def run(self, s, t):
        to, nxt, cap = self.to, self.nxt, self.cap
        total = 0
        while True:
            yield
            lv = yield from self._levels(s)
            if lv[t] == NONE:
                return total, [x != NONE for x in lv]
            it = list(self.head)
            moves = 0
            while True:
                path, x = [], s
                while True:
                    moves += 1
                    if moves & 4095 == 0:
                        yield
                    if x == t:
                        found = True
                        break
                    a = it[x]
                    want = lv[x] + 1
                    while a != NONE:
                        if cap[a] > 0 and lv[to[a]] == want:
                            break
                        a = nxt[a]
                    it[x] = a
                    if a == NONE:
                        if not path:
                            found = False
                            break
                        back = path.pop()
                        x = to[back ^ 1]
                        it[x] = nxt[it[x]]
                    else:
                        path.append(a)
                        x = to[a]
                if not found:
                    break
                push = min(cap[a] for a in path) if path else 0
                for a in path:
                    cap[a] -= push
                    cap[a ^ 1] += push
                total += push
                yield


class _Cut:
    __slots__ = ("gain", "swap", "fa", "fb")

    def __init__(self, gain, swap, fa, fb):
        self.gain, self.swap, self.fa, self.fb = gain, swap, fa, fb


def _cuts(m, label, members, a, b):
    """The slides of the border between charts a and b that lower the cut cost, one per band width, widest first.
    `label`: the chart of each face (array)."""

    found: list = []
    ma, mb = members[a], members[b]
    union = np.asarray(ma + mb, np.int64)
    n, na = len(union), len(ma)
    idx = np.full(len(label), -1, np.int64)
    idx[union] = np.arange(n)
    # each face's inner edges, face by face in the pair's order: pi (index in the pair), pj (other face's index or -1)
    cnt = m.fe_ptr[union + 1] - m.fe_ptr[union]
    at = np.repeat(m.fe_ptr[union] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    E = m.fe_edge[at]
    pi = np.repeat(np.arange(n), cnt)
    FI = union[pi]
    po = np.where(m.af0[E] == FI, m.af1[E], m.af0[E])
    pj, FR, W, LO = idx[po], m.afree[E], m.aw[E], label[po]
    start = np.concatenate([[0], np.cumsum(cnt)])
    sel = (pi < na) & (LO == b) & ~FR
    old = int(W[sel].sum())
    if old == 0:
        return found
    for x, y in ((a, b), (b, a)):
        if len(members[x]) <= FREE_FACES:
            swap = [(f, y) for f in members[x]]
            fa, fb = ([], union.tolist()) if x == a else (union.tolist(), [])
            found.append(_Cut(old, swap, fa, fb))
    yield
    big = 1 << 40  # "not reached": outside every ring
    dist = np.full(n, big, np.int64)
    front = np.unique(np.concatenate([pi[sel], pj[sel]]))
    dist[front] = 0
    level = 0
    while len(front):  # rings by breadth-first levels over every inner edge of the pair
        k = cnt[front]
        ent = np.repeat(start[front] - (np.cumsum(k) - k), k) + np.arange(int(k.sum()))
        nb = pj[ent]
        nb = nb[nb >= 0]
        nb = np.unique(nb[dist[nb] == big])
        level += 1
        dist[nb] = level
        front = nb
    yield
    ok = (pj >= 0) & ~FR
    pjs = np.maximum(pj, 0)
    last_band = None
    for r in RINGS:
        inb = dist < r
        nb = int(inb.sum())
        if nb == last_band:
            continue
        last_band = nb
        node = np.full(n, NONE, np.int64)
        node[inb] = np.arange(nb) + 2
        v = ok & inb[pi]
        jn = node[pjs]
        inner = v & (jn != NONE) & (FI < po)
        src = v & (jn == NONE) & (LO == a)
        snk = v & (jn == NONE) & (LO != a)
        if not (src.any() and snk.any()):
            continue
        arc = inner | src | snk
        ni = node[pi]
        g = _Flow.from_arcs(nb + 2, np.where(inner, ni, np.where(src, 0, 1))[arc], np.where(inner, jn, ni)[arc], W[arc])
        yield
        flow, reach = yield from g.run(0, 1)
        if flow >= old:
            break
        reach = np.asarray(reach, bool)
        new = np.where(node != NONE, np.where(reach[np.maximum(node, 0)], a, b), label[union])
        yield
        for c in (a, b):
            mine = new == c
            e2 = ok & mine[pi] & mine[pjs]
            part = _components(n, pi[e2], pj[e2])
            seed = mine & (node == NONE)
            keep = mine & np.isin(part, part[seed])
            new[mine & ~keep] = b if c == a else a
            yield
        cost = int(W[ok & (new[pi] == a) & (new[pjs] == b)].sum())
        yield
        if cost >= old:
            continue
        on_a = new == a
        fa, fb = union[on_a].tolist(), union[~on_a].tolist()
        moved = np.flatnonzero(new != label[union])
        swap = list(zip(union[moved].tolist(), new[moved].tolist()))
        if not fa or not fb or not swap or (found and found[-1].swap == swap and found[-1].fa and found[-1].fb):
            continue
        found.append(_Cut(old - cost, swap, fa, fb))
    return found


class _Prop:
    __slots__ = ("a", "b", "swap", "la", "lb", "grow")

    def __init__(self, a, b, swap, la, lb, grow):
        self.a, self.b, self.swap, self.la, self.lb, self.grow = a, b, swap, la, lb, grow


def _evaluate(m, label, q, bx, cur, room, a, b, lst):
    """The first of a pair's cuts whose two charts stay disks, lie flat one-to-one, keep their stretch and boxes."""

    def known(x):
        return x if math.isfinite(x) else 0.0

    qa, qb = q[a], q[b]
    limit = (max(known(qa[0]), known(qb[0]), L2_MEDIAN), max(known(qa[1]), known(qb[1]), L2_WORST))
    t = m.t
    for cut in lst:
        touched = {t.flat[h] for f, _ in cut.swap for h in t.face_corners(f)}

        def side(faces, c):
            if not faces:
                return (None,)
            lay = yield from _lay(m, faces, c, cur, label, touched, limit)
            if lay is None or not _within(lay.q, limit):
                return None
            ok = yield from lay.confirm(m.tree.build_stretch, limit)
            return (lay,) if ok else None

        # the smaller side first: when it breaks a rule the larger one is never laid out (both must pass)
        order = ((cut.fa, a), (cut.fb, b)) if len(cut.fa) <= len(cut.fb) else ((cut.fb, b), (cut.fa, a))
        got = {}
        for faces, c in order:
            got[c] = yield from side(faces, c)
            if got[c] is None:
                break
        ra, rb = got.get(a), got.get(b)
        if ra is None or rb is None:
            continue
        laid = [x for x in (ra[0], rb[0]) if x is not None]
        over = 0.0
        for x in laid:
            over += x.q[2]
        grow = over - (qa[2] + qb[2]) if math.isfinite(qa[2]) and math.isfinite(qb[2]) else 0.0
        if grow > 0.0 and grow > room:
            continue
        oa, ob = bx[a], bx[b]
        long = 0.0
        for x in laid:
            long = _fmax(long, x.bx[1])
        if math.isfinite(oa[1]) and math.isfinite(ob[1]) and long > _fmax(oa[1], ob[1]) * (1.0 + PAGE_SLACK):
            continue
        if cut.gain - sum(x.slit_cost for x in laid) <= 0:
            continue
        return _Prop(a, b, cut.swap, ra[0], rb[0], grow)
    return None


def moves(tree):
    """Generator: the moves of a tree, made on its level at the build stretch (returns a `Hide`)."""

    t = tree.topo
    raw_chart, raw_cut, raw_numbers = yield from tree._level_steps(100.0 * tree.build_stretch, hide=False)
    yield
    z = yield from _arrays_steps(tree)
    nf, ne = t.nf, t.ne
    raw_np = np.asarray(raw_chart, np.int64)
    raw = raw_np.tolist()
    inn = z.inner_mask
    # an inner edge is free (costs nothing to cut) when fixed or a slit inside one chart; the rest cost their weight
    slit = np.asarray(raw_cut, bool) & inn & (raw_np[z.af0] == raw_np[np.maximum(z.af1, 0)])
    afree = ~inn | z.fixed_all | slit
    aw = np.where(afree, 0, np.round(np.asarray(tree.cost, np.float64) * CAP_SCALE).astype(np.int64) + 1)
    yield
    on = inn[z.cedge]
    fe_edge = z.cedge[on]
    fe_ptr = np.concatenate([[0], np.cumsum(np.bincount(z.cface[on], minlength=nf))]).astype(np.int64)
    fe_list, ptr = fe_edge.tolist(), fe_ptr.tolist()
    fedge: list = []
    for f0 in range(0, nf, 2048):
        fedge += [fe_list[ptr[f] : ptr[f + 1]] for f in range(f0, min(nf, f0 + 2048))]
        yield
    m = _Mesh(tree, z, fedge, aw.tolist(), afree.tolist(), aw, afree, fe_ptr, fe_edge)
    slit_e = [False] * ne
    label = list(raw)
    m.alabel = raw_np.copy()
    nc = int(raw_numbers[0])
    by = np.argsort(raw_np, kind="stable")
    cut_at = np.searchsorted(raw_np[by], np.arange(nc + 1))
    members: list = [by[cut_at[c] : cut_at[c + 1]].tolist() for c in range(nc)]
    yield
    _, sm_start = yield from tree._smoothing_steps()
    cur = np.array(sm_start if sm_start is not None else tree.start_uv, np.float64).reshape(-1)
    changed = [False] * nf
    q, bx = [], []
    for c in range(nc):
        res = ((INF, INF, INF), (math.nan, math.nan))
        if members[c]:
            lay = yield from _lay(m, members[c], c, cur, m.alabel, set(), (INF, INF))
            if lay is not None and (yield from lay.confirm(tree.build_stretch, (INF, INF))):
                res = (lay.q, lay.bx)
        q.append(res[0])
        bx.append(res[1])
    over = 0.0
    for x in q:
        if math.isfinite(x[2]):
            over += x[2]
    p0, _, _, _, _, tri_area = _tri_geometry(m.points, m.ip, m.fl)
    tri_face = np.repeat(np.arange(nf), np.maximum(np.diff(m.ip) - 2, 0))
    area = _seq_sum(np.bincount(tri_face, weights=tri_area, minlength=nf))  # per face, then over faces, as running sums
    yield
    budget = OVER_SHARE * area
    dirty = [True] * nc
    while True:
        live = m.inner & ~m.afree
        x, y = m.alabel[m.af0[live]], m.alabel[m.af1[live]]
        dmask = np.asarray(dirty, bool)
        pick = (x != y) & (dmask[x] | dmask[y])
        keys = np.unique(np.minimum(x, y)[pick] * nc + np.maximum(x, y)[pick])
        yield
        if len(keys) == 0:
            break
        ranked = []
        for a, b in zip((keys // nc).tolist(), (keys % nc).tolist()):
            c = yield from _cuts(m, m.alabel, members, a, b)
            if c:
                ranked.append((max(x.gain for x in c), a, b, c))
        ranked.sort(key=lambda x: (-x[0], x[1], x[2]))
        dirty = [False] * nc
        any_ = False
        left = [(a, b, c) for _, a, b, c in ranked]
        while True:
            busy = list(dirty)
            batch = []
            for i, slot_ in enumerate(left):
                if slot_ is None:
                    continue
                a, b, _ = slot_
                if dirty[a] or dirty[b]:
                    left[i] = None
                    continue
                if busy[a] or busy[b]:
                    continue
                busy[a] = busy[b] = True
                batch.append(slot_)
                left[i] = None
            if not batch:
                break
            room = budget - over
            props = []
            for a, b, c in batch:
                props.append((yield from _evaluate(m, m.alabel, q, bx, cur, room, a, b, c)))
            for p in props:
                if p is None:
                    continue
                if p.grow > 0.0 and over + p.grow > budget:
                    continue
                over += p.grow
                dirty[p.a] = dirty[p.b] = True
                any_ = True
                for f, s in p.swap:
                    label[f] = s
                    m.alabel[f] = s
                for c, lay in ((p.a, p.la), (p.b, p.lb)):
                    if lay is None:
                        continue
                    for e in lay.slit:
                        slit_e[e] = True
                        m.free[e] = True
                        m.afree[e] = True
                    q[c], bx[c] = lay.q, lay.bx
                    cur[2 * lay.corners] = lay.uv[2 * lay.vert]
                    cur[2 * lay.corners + 1] = lay.uv[2 * lay.vert + 1]
                for c in (p.a, p.b):
                    members[c] = [f for f in members[c] if label[f] == c]
                for f, s in p.swap:
                    members[s].append(f)
                for c in (p.a, p.b):
                    members[c].sort()
                    for f in members[c]:
                        changed[f] = True
        if not any_:
            break
    anchor = [NONE] * nf
    queue: deque = deque()
    moved = 0
    for f in range(nf):
        if label[f] == raw[f]:
            anchor[f] = f
            queue.append(f)
        else:
            moved += 1
    yield
    pops = 0
    while queue:
        pops += 1
        if pops & 2047 == 0:
            yield
        f = queue.popleft()
        for e in m.fedge[f]:
            if m.free[e]:
                continue
            o = m.other(e, f)
            if anchor[o] == NONE and label[o] == label[f]:
                anchor[o] = anchor[f]
                queue.append(o)
    yield
    for f in range(nf):
        if anchor[f] == f:
            anchor[f] = NONE
    yield
    start = np.where(np.repeat(np.asarray(changed, bool)[np.asarray(t.cface, np.int64)], 2), cur, math.nan).tolist()
    return Hide(anchor, changed, start, slit_e, moved)
