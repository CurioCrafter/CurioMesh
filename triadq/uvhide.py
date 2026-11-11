# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Seams out of sight (R7): once the charts of a level are made, every border between two charts slides onto nearby
edges where it shows less, by giving faces from one of the two charts to the other. Our own code, written from:

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
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

from . import uvflat
from .uvchart import FREE_FACES, _Dsu

RINGS = (32, 16, 8, 4, 2, 1)
CAP_SCALE = 1024.0
L2_MEDIAN = 1.08
L2_WORST = 1.3
OVER_SHARE = 0.005
FREE_RINGS = 2
PAGE_SLACK = 0.1
FULL_STEPS = 10
FAN_MARGIN = 0.01

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

    stage = uvflat.Flatten(pts, ip, fl, start, **options)
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


def apply_steps(tree, chart, cut):
    """`apply` as a generator, yielding between short pieces."""

    h = of(tree)
    if h.moved == 0:
        return None
    t = tree.topo
    raw = [int(x) for x in chart]
    cut0 = [bool(x) for x in cut]
    lab = [raw[f] if h.anchor[f] == NONE else raw[h.anchor[f]] for f in range(t.nf)]
    fixed = tree.fixed
    new_cut = list(cut0)
    yield
    for e in range(t.ne):
        if e & 4095 == 4095:
            yield
        fs = _inner(t, e)
        if fs is not None:
            f0, f1 = fs
            slit = cut0[e] and raw[f0] == raw[f1]
            new_cut[e] = bool(fixed[e]) or slit or h.slit[e] or lab[f0] != lab[f1]
    ch, n = yield from _pieces_steps(t, new_cut)
    chi = yield from _chi_all_steps(t, new_cut, ch, n)
    if any(c != 1 for c in chi):
        # below the level the moves were made on, a chart can close round a hole: there it keeps this level's own
        # borders as well (still inside the charts above)
        for e in range(t.ne):
            if e & 4095 == 4095:
                yield
            fs = _inner(t, e)
            if fs is not None:
                f0, f1 = fs
                bad = chi[ch[f0]] != 1 or chi[ch[f1]] != 1
                if bad and raw[f0] != raw[f1]:
                    new_cut[e] = True
        ch, n = yield from _pieces_steps(t, new_cut)
        chi = yield from _chi_all_steps(t, new_cut, ch, n)
    not_disk = sum(1 for c in chi if c != 1)
    return np.array(ch, np.int64), np.array(new_cut, bool), n, not_disk


def _pieces(t, cut):
    return _drive(_pieces_steps(t, cut))


def _pieces_steps(t, cut):
    d = _Dsu(t.nf)
    for e in range(t.ne):
        if e & 4095 == 4095:
            yield
        fs = _inner(t, e)
        if fs is not None and not cut[e]:
            d.union(fs[0], fs[1])
    ids: dict = {}
    chart = [ids.setdefault(d.find(f), len(ids)) for f in range(t.nf)]
    return chart, len(ids)


def _inner(t, e):
    h0, h1 = t.euse[e]
    if h1 < 0 or t.wall[e]:
        return None
    return t.cface[h0], t.cface[h1]


def _chi_all_steps(t, cut, chart, n):
    d = _Dsu(len(t.flat))
    chi = [0] * n
    for e in range(t.ne):
        if e & 4095 == 4095:
            yield
        if _inner(t, e) is not None and not cut[e]:
            h0, h1 = t.euse[e]
            for v in (t.ea[e], t.eb[e]):
                d.union(t.at(h0, v), t.at(h1, v))
            chi[chart[t.cface[h0]]] += 1
    for f in range(t.nf):
        if f & 4095 == 4095:
            yield
        c = chart[f]
        chi[c] += 1
        for h in t.face_corners(f):
            chi[c] -= 1
            if d.find(h) == h:
                chi[c] += 1
    return chi


# ---------------------------------------------------------------------------------------------------------------------
# The moves on the full level
# ---------------------------------------------------------------------------------------------------------------------


class _Mesh:
    def __init__(self, tree, fedge, w, free):
        self.tree, self.t, self.fedge, self.w, self.free = tree, tree.topo, fedge, w, free
        self.points = np.asarray(tree.points, np.float64).reshape(-1, 3)

    def other(self, e, f):
        a, b = _inner(self.t, e)
        return b if a == f else a


class _Laid:
    __slots__ = ("corners", "vert", "uv", "q", "bx", "mesh", "sides", "grid", "slit", "slit_cost")

    def __init__(self, corners, vert, uv, q, mesh, sides, grid, slit, slit_cost):
        self.corners, self.vert, self.uv, self.q, self.mesh, self.sides, self.grid = corners, vert, uv, q, mesh, sides, grid
        self.slit, self.slit_cost = slit, slit_cost
        self.bx = _min_box(uv)

    def confirm(self, allowed, limit) -> bool:
        """Lay the chart out the way the level's flatten will (from this layout) and keep it when still one-to-one
        and within `limit`."""

        if self.grid:
            return True
        pts, ip, fl = self.mesh
        uv = yield from _flat(pts, ip, fl, self.uv.reshape(-1, 2), rows="auto", start_mode="given", stretch_allowed=allowed)
        if uv is None:
            return False
        yield
        if not np.isfinite(uv).all() or uvflat.count_flips(ip, fl, uv) != 0 or not _fans_ok(pts, ip, fl, uv, FAN_MARGIN):
            return False
        if _border_crosses(uv, self.sides):
            return False
        q = _l2(pts, ip, fl, uv)
        if not _within(q, limit):
            return False
        self.bx, self.uv, self.q = _min_box(uv), uv, q
        return True


def _face_area(pts, t, f):
    r = t.face_corners(f)
    x0 = pts[t.flat[r.start]]
    a = 0.0
    for h in range(r.start + 1, r.stop - 1):
        x1, x2 = pts[t.flat[h]], pts[t.flat[h + 1]]
        a += 0.5 * _norm(_cross3(x1 - x0, x2 - x0))
    return a


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(c):
    return math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])


def _fmax(a, b):
    """max that skips a NaN."""

    if a != a:
        return b
    if b != b:
        return a
    return a if a >= b else b


def _min_box(uv):
    """Smallest-area box around points (x, y pairs): (area, longer side)."""

    p = sorted(set((float(uv[2 * i]), float(uv[2 * i + 1])) for i in range(len(uv) // 2)))
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
    best = (INF, 0.0)
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        ln = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        if not ln > 0.0:
            continue
        d = ((b[0] - a[0]) / ln, (b[1] - a[1]) / ln)
        lo, hi, top = INF, -INF, 0.0
        for x in h:
            r = (x[0] - a[0], x[1] - a[1])
            s = r[0] * d[0] + r[1] * d[1]
            lo, hi = min(lo, s), max(hi, s)
            top = _fmax(top, abs(r[0] * d[1] - r[1] * d[0]))
        area = (hi - lo) * top
        if area < best[0]:
            best = (area, _fmax(hi - lo, top))
    return best


def _fan_tris(pts, ip, fl):
    for f in range(len(ip) - 1):
        poly = fl[ip[f]:ip[f + 1]]
        for i in range(1, len(poly) - 1):
            yield int(poly[0]), int(poly[i]), int(poly[i + 1])


def _fans_ok(pts, ip, fl, uv, margin) -> bool:
    tris = []
    for p0, p1, p2 in _fan_tris(pts, ip, fl):
        a, b, c = (uv[2 * p0], uv[2 * p0 + 1]), (uv[2 * p1], uv[2 * p1 + 1]), (uv[2 * p2], uv[2 * p2 + 1])
        a2 = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if not a2 > 0.0:
            return False
        tris.append((a2, 0.5 * _norm(_cross3(pts[p1] - pts[p0], pts[p2] - pts[p0]))))
    s2 = s3 = 0.0
    for x, y in tris:
        s2 += x
        s3 += y
    scale = s2 / s3 if s3 > 0.0 else 0.0
    return all(a2 >= margin * scale * a3 for a2, a3 in tris)


def _within(q, limit) -> bool:
    return q[0] <= limit[0] and q[1] <= limit[1]


def _l2(pts, ip, fl, uv):
    """Sander et al. L2 of the map from the layout to the surface per fan triangle, the layout's area scaled to the
    surface's: (area-weighted median, worst 1 %, surface area above `L2_WORST`)."""

    tris = []
    a3 = a2 = 0.0
    for p0, p1, p2 in _fan_tris(pts, ip, fl):
        x0, x1, x2 = pts[p0], pts[p1], pts[p2]
        e1, e2 = x1 - x0, x2 - x0
        l1 = math.sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2])
        area = 0.5 * _norm(_cross3(e1, e2))
        if not (l1 > 0.0 and area > 0.0):
            continue
        bx, by = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / l1, 2.0 * area / l1
        u0, u1, u2 = (uv[2 * p0], uv[2 * p0 + 1]), (uv[2 * p1], uv[2 * p1 + 1]), (uv[2 * p2], uv[2 * p2 + 1])
        d1, d2 = (u1[0] - u0[0], u1[1] - u0[1]), (u2[0] - u0[0], u2[1] - u0[1])
        a3 += area
        a2 += 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
        tris.append((l1, bx, by, area, d1, d2))
    if not (a3 > 0.0 and a2 > 0.0):
        return (INF, INF, INF)
    s = math.sqrt(a2 / a3)
    vals, w = [], []
    for l1, bx, by, area, d1, d2 in tris:
        det = d1[0] * d2[1] - d1[1] * d2[0]
        if det > 0.0:
            inv = ((d2[1] / det, -d2[0] / det), (-d1[1] / det, d1[0] / det))
            m = ((l1 * inv[0][0] + bx * inv[1][0], l1 * inv[0][1] + bx * inv[1][1]), (by * inv[1][0], by * inv[1][1]))
            v = s * math.sqrt((m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[1][0] * m[1][0] + m[1][1] * m[1][1]) / 2.0)
        else:
            v = INF
        vals.append(v)
        w.append(area)
    over = 0.0
    for v, a in zip(vals, w):
        if v > L2_WORST:
            over += a
    return (_wq(vals, w, 0.5), _wq(vals, w, 0.99), over)


def _wq(values, w, q):
    order = sorted(range(len(values)), key=lambda i: values[i])
    total = 0.0
    for x in w:
        total += x
    acc = 0.0
    for i in order:
        acc += w[i]
        if acc >= q * total:
            return values[i]
    return values[order[-1]] if order else math.nan


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
    """Whether two border sides that share no vertex cross (touching does not count), through a grid of cells."""

    if len(sides) < 4:
        return False

    def p(v):
        return (uv[2 * v], uv[2 * v + 1])

    lo = [INF, INF]
    ln = 0.0
    for a, b in sides:
        x, y = p(a), p(b)
        for r in range(2):
            lo[r] = min(lo[r], x[r], y[r])
        ln += math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    cell = max(ln / len(sides), 1e-12)

    def cells(x, r):
        return math.floor((x - lo[r]) / cell)

    grid: dict = {}
    for i, (a, b) in enumerate(sides):
        x, y = p(a), p(b)
        for cx in range(cells(min(x[0], y[0]), 0), cells(max(x[0], y[0]), 0) + 1):
            for cy in range(cells(min(x[1], y[1]), 1), cells(max(x[1], y[1]), 1) + 1):
                grid.setdefault((cx, cy), []).append(i)

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for lst in grid.values():
        for n, i in enumerate(lst):
            a, b = sides[i]
            for j in lst[n + 1:]:
                c, d = sides[j]
                if a == c or a == d or b == c or b == d:
                    continue
                pa, pb, pc, pd = p(a), p(b), p(c), p(d)
                if orient(pa, pb, pc) * orient(pa, pb, pd) < 0.0 and orient(pc, pd, pa) * orient(pc, pd, pb) < 0.0:
                    return True
    return False


def _band(ip, fl, n, seeds, k):
    """The vertices within `k` rings (by shared faces) of `seeds`."""

    vf: list = [[] for _ in range(n)]
    for f in range(len(ip) - 1):
        for v in fl[ip[f]:ip[f + 1]]:
            vf[int(v)].append(f)
    inside = [False] * n
    front = list(seeds)
    for s in seeds:
        inside[s] = True
    for _ in range(k):
        nxt = []
        for v in front:
            for f in vf[v]:
                for w in fl[ip[f]:ip[f + 1]]:
                    w = int(w)
                    if not inside[w]:
                        inside[w] = True
                        nxt.append(w)
        front = nxt
    return inside


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


def _bridge(m, faces, slot, vert, nv, inside, chi):
    """For a chart that is a disk with holes (chi = 2 - loops): the cheapest path of inner edges from the border loop
    of the lowest vertex to another loop, through no other border vertex; None for any other shape."""

    t = m.t
    loops = _Dsu(nv)
    on = [False] * nv
    k = 0
    for f in faces:
        r = t.face_corners(f)
        first = k
        for h in r:
            nk = first if h + 1 == r.stop else k + 1
            if t.cedge[h] not in inside:
                loops.union(vert[k], vert[nk])
                on[vert[k]] = on[vert[nk]] = True
            k += 1
    roots = {loops.find(v) for v in range(nv) if on[v]}
    b = len(roots)
    if b < 2 or chi != 2 - b:
        return None
    adj: list = [[] for _ in range(nv)]
    for e in sorted(inside):
        h0 = t.euse[e][0]
        va = vert[slot[t.at(h0, t.ea[e])]]
        vb = vert[slot[t.at(h0, t.eb[e])]]
        adj[va].append((vb, e))
        adj[vb].append((va, e))
    start = next((v for v in range(nv) if on[v]), None)
    if start is None:
        return None
    home = loops.find(start)
    dist = [None] * nv
    prev = [(NONE, NONE)] * nv
    heap: list = []
    for v in range(nv):
        if on[v] and loops.find(v) == home:
            dist[v] = 0
            heapq.heappush(heap, (0, v))
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        if on[v] and loops.find(v) != home:
            path, x = [], v
            while prev[x][0] != NONE:
                path.append(prev[x][1])
                x = prev[x][0]
            return path
        if on[v] and d > 0:
            continue
        for o, e in adj[v]:
            if on[o] and loops.find(o) == home:
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
    slot: dict = {}
    corners: list = []
    for f in faces:
        for h in t.face_corners(f):
            slot[h] = len(corners)
            corners.append(h)
    fidx = {f: i for i, f in enumerate(faces)}
    slit: list = []
    while True:
        cd, fd = _Dsu(len(corners)), _Dsu(len(faces))
        glued = 0
        inside: set = set()
        for f in faces:
            for e in m.fedge[f]:
                o = m.other(e, f)
                if o > f and not m.free[e] and o in fidx and e not in slit:
                    glued += 1
                    inside.add(e)
                    fd.union(fidx[f], fidx[o])
                    h0, h1 = t.euse[e]
                    for v in (t.ea[e], t.eb[e]):
                        cd.union(slot[t.at(h0, v)], slot[t.at(h1, v)])
        yield
        r0 = fd.find(0)
        if any(fd.find(i) != r0 for i in range(1, len(faces))):
            return None
        vid: dict = {}
        vert, pts = [], []
        for k, h in enumerate(corners):
            r = cd.find(k)
            if r not in vid:
                vid[r] = len(vid)
                pts.append(m.points[t.flat[h]])
            vert.append(vid[r])
        chi = len(vid) - (len(corners) - glued) + len(faces)
        if chi == 1:
            break
        path = _bridge(m, faces, slot, vert, len(vid), inside, chi)
        if path is None:
            return None
        slit.extend(path)
    slit_cost = sum(m.w[e] for e in slit)
    pts = np.array(pts, np.float64).reshape(-1, 3)
    ip, fl = [0], []
    k = 0
    for f in faces:
        for _ in t.face_corners(f):
            fl.append(vert[k])
            k += 1
        ip.append(len(fl))
    ip, fl = np.array(ip, np.int64), np.array(fl, np.int64)
    nv = len(vid)
    sides = []
    k = 0
    for f in faces:
        r = t.face_corners(f)
        first = k
        for h in r:
            nk = first if h + 1 == r.stop else k + 1
            if t.cedge[h] not in inside:
                sides.append((vert[k], vert[nk]))
            k += 1

    def good(uv):
        if not np.isfinite(uv).all() or uvflat.count_flips(ip, fl, uv) != 0 or not _fans_ok(pts, ip, fl, uv, FAN_MARGIN):
            return None
        if _border_crosses(uv, sides):
            return None
        q = _l2(pts, ip, fl, uv)
        return q if math.isfinite(q[1]) else None

    def fin(uv, q, grid):
        return _Laid(corners, vert, uv, q, (pts, ip, fl), sides, grid, list(slit), slit_cost)

    yield
    g = uvflat.grid_layout(pts, ip, fl)
    yield
    if g is not None:
        uv = yield from _flat(pts, ip, fl, g[0], col=g[1], row=g[2], rows="auto", start_mode="given", stretch_allowed=tree.build_stretch)
        if uv is None:
            return None
        q = good(uv)
        if q is None or not _within(q, limit):
            return None
        return fin(uv, q, True)
    came = [before[f] != lab for f in faces]
    sa, sb = [[0.0, 0.0, 0.0] for _ in range(nv)], [[0.0, 0.0, 0.0] for _ in range(nv)]
    k = 0
    for i, f in enumerate(faces):
        for h in t.face_corners(f):
            s = sb[vert[k]] if came[i] else sa[vert[k]]
            s[0] += cur[2 * h]
            s[1] += cur[2 * h + 1]
            s[2] += 1.0
            k += 1
    pairs = [((sb[v][0] / sb[v][2], sb[v][1] / sb[v][2]), (sa[v][0] / sa[v][2], sa[v][1] / sa[v][2])) for v in range(nv) if sa[v][2] > 0.0 and sb[v][2] > 0.0]
    any_came = any(came)
    if not any_came or len(pairs) >= 2:
        c, tr = _align(pairs) if any_came else ((1.0, 0.0), (0.0, 0.0))
        uv = [0.0] * (2 * nv)
        cnt = [0.0] * nv
        k = 0
        for i, f in enumerate(faces):
            for h in t.face_corners(f):
                p = (cur[2 * h], cur[2 * h + 1])
                if came[i]:
                    p = (c[0] * p[0] - c[1] * p[1] + tr[0], c[1] * p[0] + c[0] * p[1] + tr[1])
                v = vert[k]
                uv[2 * v] += p[0]
                uv[2 * v + 1] += p[1]
                cnt[v] += 1.0
                k += 1
        for v in range(nv):
            uv[2 * v] /= cnt[v]
            uv[2 * v + 1] /= cnt[v]
        uv = np.array(uv, np.float64)
        if np.isfinite(uv).all() and uvflat.count_flips(ip, fl, uv) == 0:
            seeds = [vert[k] for k in range(len(corners)) if t.flat[corners[k]] in touched]
            if seeds:
                yield from _quick_stretch(pts, ip, fl, uv, _band(ip, fl, nv, seeds, FREE_RINGS))
            q = good(uv)
            if q is not None and _within(q, limit):
                return fin(uv, q, False)
    # else the whole chart laid out again from scratch
    uv = yield from _flat(pts, ip, fl, None, rows="off", start_mode="free", steps=FULL_STEPS)
    if uv is None:
        return None
    q = good(uv)
    if q is None:
        return None
    return fin(uv, q, False)


class _Flow:
    """Max flow by blocking flows (Dinic 1970), undirected, whole-number capacities."""

    def __init__(self, n):
        self.head, self.to, self.nxt, self.cap = [NONE] * n, [], [], []

    def add(self, a, b, c):
        for x, y in ((a, b), (b, a)):
            self.to.append(y)
            self.cap.append(c)
            self.nxt.append(self.head[x])
            self.head[x] = len(self.to) - 1

    def _levels(self, s):
        lv = [NONE] * len(self.head)
        lv[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            a = self.head[x]
            while a != NONE:
                y = self.to[a]
                if self.cap[a] > 0 and lv[y] == NONE:
                    lv[y] = lv[x] + 1
                    q.append(y)
                a = self.nxt[a]
        return lv

    def run(self, s, t):
        total = 0
        while True:
            yield
            lv = self._levels(s)
            if lv[t] == NONE:
                return total, [x != NONE for x in lv]
            it = list(self.head)
            while True:
                path, x = [], s
                while True:
                    if x == t:
                        found = True
                        break
                    a = it[x]
                    while a != NONE:
                        if self.cap[a] > 0 and lv[self.to[a]] == lv[x] + 1:
                            break
                        a = self.nxt[a]
                    it[x] = a
                    if a == NONE:
                        if not path:
                            found = False
                            break
                        back = path.pop()
                        x = self.to[back ^ 1]
                        it[x] = self.nxt[it[x]]
                    else:
                        path.append(a)
                        x = self.to[a]
                if not found:
                    break
                push = min(self.cap[a] for a in path) if path else 0
                for a in path:
                    self.cap[a] -= push
                    self.cap[a ^ 1] += push
                total += push
                yield


class _Cut:
    __slots__ = ("gain", "swap", "fa", "fb")

    def __init__(self, gain, swap, fa, fb):
        self.gain, self.swap, self.fa, self.fb = gain, swap, fa, fb


def _cuts(m, label, members, a, b):
    """The slides of the border between charts a and b that lower the cut cost, one per band width, widest first."""

    found: list = []
    union = members[a] + members[b]
    idx = {f: i for i, f in enumerate(union)}
    n = len(union)
    old = 0
    dist = [None] * n
    queue: deque = deque()
    for f in members[a]:
        for e in m.fedge[f]:
            o = m.other(e, f)
            if label[o] == b and not m.free[e]:
                old += m.w[e]
                for x in (f, o):
                    i = idx[x]
                    if dist[i] is None:
                        dist[i] = 0
                        queue.append(i)
    if old == 0:
        return found
    for x, y in ((a, b), (b, a)):
        if len(members[x]) <= FREE_FACES:
            swap = [(f, y) for f in members[x]]
            fa, fb = ([], list(union)) if x == a else (list(union), [])
            found.append(_Cut(old, swap, fa, fb))
    yield
    while queue:
        i = queue.popleft()
        for e in m.fedge[union[i]]:
            j = idx.get(m.other(e, union[i]))
            if j is not None and dist[j] is None:
                dist[j] = dist[i] + 1
                queue.append(j)
    last_band = None
    for r in RINGS:
        band = [i for i in range(n) if dist[i] is not None and dist[i] < r]
        if len(band) == last_band:
            continue
        last_band = len(band)
        node = [NONE] * n
        for k, i in enumerate(band):
            node[i] = k + 2
        g = _Flow(len(band) + 2)
        to_s = to_t = False
        for i in band:
            f = union[i]
            for e in m.fedge[f]:
                o = m.other(e, f)
                j = idx.get(o)
                if j is None or m.free[e]:
                    continue
                w = m.w[e]
                if node[j] != NONE:
                    if f < o:
                        g.add(node[i], node[j], w)
                elif label[o] == a:
                    g.add(0, node[i], w)
                    to_s = True
                else:
                    g.add(1, node[i], w)
                    to_t = True
        if not (to_s and to_t):
            continue
        flow, reach = yield from g.run(0, 1)
        if flow >= old:
            break
        new = [(a if reach[node[i]] else b) if node[i] != NONE else label[union[i]] for i in range(n)]
        for c in (a, b):
            keep = [False] * n
            q2 = deque(i for i in range(n) if node[i] == NONE and new[i] == c)
            for i in q2:
                keep[i] = True
            while q2:
                i = q2.popleft()
                for e in m.fedge[union[i]]:
                    if m.free[e]:
                        continue
                    j = idx.get(m.other(e, union[i]))
                    if j is not None and not keep[j] and new[j] == c:
                        keep[j] = True
                        q2.append(j)
            o = b if c == a else a
            for i in range(n):
                if new[i] == c and not keep[i]:
                    new[i] = o
        cost = 0
        for i in range(n):
            if new[i] != a:
                continue
            for e in m.fedge[union[i]]:
                j = idx.get(m.other(e, union[i]))
                if j is not None and new[j] == b and not m.free[e]:
                    cost += m.w[e]
        yield
        if cost >= old:
            continue
        fa, fb, swap = [], [], []
        for i in range(n):
            f = union[i]
            (fa if new[i] == a else fb).append(f)
            if new[i] != label[f]:
                swap.append((f, new[i]))
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
    raw = [int(x) for x in raw_chart]
    rcut = [bool(x) for x in raw_cut]
    nf, ne = t.nf, t.ne
    yield
    fedge = []
    for f in range(nf):
        if f & 2047 == 2047:
            yield
        fedge.append([t.cedge[h] for h in t.face_corners(f) if _inner(t, t.cedge[h]) is not None])
    free, w = [True] * ne, [0] * ne
    for e in range(ne):
        if e & 4095 == 4095:
            yield
        fs = _inner(t, e)
        if fs is not None:
            slit = rcut[e] and raw[fs[0]] == raw[fs[1]]
            free[e] = bool(tree.fixed[e]) or slit
            if not free[e]:
                w[e] = int(round(float(tree.cost[e]) * CAP_SCALE)) + 1
    m = _Mesh(tree, fedge, w, free)
    slit_e = [False] * ne
    label = list(raw)
    nc = int(raw_numbers[0])
    members: list = [[] for _ in range(nc)]
    for f in range(nf):
        members[label[f]].append(f)
    yield
    _, sm_start = yield from tree._smoothing_steps()
    cur = list(np.asarray(sm_start if sm_start is not None else tree.start_uv, np.float64).reshape(-1))
    changed = [False] * nf
    q, bx = [], []
    for c in range(nc):
        res = ((INF, INF, INF), (math.nan, math.nan))
        if members[c]:
            lay = yield from _lay(m, members[c], c, cur, label, set(), (INF, INF))
            if lay is not None and (yield from lay.confirm(tree.build_stretch, (INF, INF))):
                res = (lay.q, lay.bx)
        q.append(res[0])
        bx.append(res[1])
    over = 0.0
    for x in q:
        if math.isfinite(x[2]):
            over += x[2]
    area = 0.0
    for f in range(nf):
        if f & 2047 == 2047:
            yield
        area += _face_area(m.points, t, f)
    budget = OVER_SHARE * area
    dirty = [True] * nc
    while True:
        pairs = set()
        for e in range(ne):
            if e & 4095 == 4095:
                yield
            if m.free[e]:
                continue
            fs = _inner(t, e)
            if fs is not None:
                x, y = label[fs[0]], label[fs[1]]
                if x != y and (dirty[x] or dirty[y]):
                    pairs.add((min(x, y), max(x, y)))
        if not pairs:
            break
        ranked = []
        for a, b in sorted(pairs):
            c = yield from _cuts(m, label, members, a, b)
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
                props.append((yield from _evaluate(m, label, q, bx, cur, room, a, b, c)))
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
                for c, lay in ((p.a, p.la), (p.b, p.lb)):
                    if lay is None:
                        continue
                    for e in lay.slit:
                        slit_e[e] = True
                        m.free[e] = True
                    q[c], bx[c] = lay.q, lay.bx
                    for k, h in enumerate(lay.corners):
                        v = lay.vert[k]
                        cur[2 * h] = float(lay.uv[2 * v])
                        cur[2 * h + 1] = float(lay.uv[2 * v + 1])
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
    while queue:
        f = queue.popleft()
        for e in m.fedge[f]:
            if m.free[e]:
                continue
            o = m.other(e, f)
            if anchor[o] == NONE and label[o] == label[f]:
                anchor[o] = anchor[f]
                queue.append(o)
    for f in range(nf):
        if anchor[f] == f:
            anchor[f] = NONE
    start = [cur[i] if changed[t.cface[i // 2]] else math.nan for i in range(2 * len(t.flat))]
    return Hide(anchor, changed, start, slit_e, moved)
