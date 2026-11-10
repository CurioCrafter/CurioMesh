# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Row Charts twin (package K6): patches, merge tree, levels.

NumPy only, no bpy. The reference is the kernel source `uvchart.rs`; this twin covers the quad topology view, irregular
points (forced-line turning vertices included), traces that all advance one edge per step and stop at a border, a
forced line or another trace (motorcycle graphs: Eppstein, Goodrich, Kim, Tamstorf 2008), patches, the Euler check
(annulus: the cheapest straight crossing row; otherwise tree-cotree after Eppstein 2003), segments with their cut cost
(`length_px`, or `seen_px + length_px / L` with seams out of sight; idea after Sheffer, Hart 2002), the merge tree
(greedy pair merges after Garland, Willmott, Heckbert 2001; grid unions measured exactly from the glued grid) and
`level(x)`, which applies every recorded merge whose stretch fits and skips the others.

Compact charts (R4): the queue ranks a union by its irregularity, outline^2 / (4 pi area), the outline weighted by
what a cut there costs (its length, less on a fold, more where it is seen): the shape term of hierarchical face
clustering (Garland, Willmott, Heckbert 2001) and the compactness terms of chart growing (D-Charts: Julius, Kraevoy,
Sheffer 2005; Cohen-Steiner, Alliez, Desbrun 2004), as a merge order. A union more irregular than `SHAPE_MAX` (plain
lengths) and than both its parts is refused (ours).

Thin tubes (R5, `_tube_cuts`): a part that wraps round and is a long strip of surface (springs, chain links, rope, wire)
is cut across into pieces and each piece once along its length on its least-seen side, started from Tutte's embedding
on a convex outline of the strip (Tutte 1963, Floater 1997) and kept when the UV Check's own stretch allows it. A union
refused for its shape is still made when its gap ring outweighs its box and its box with the ring takes less page than
its parts' (`_min_box`: rotating calipers, Toussaint 1983). Ours, as the kernel.

Fewer, smoother charts (R6): a last merge round where every chart, smallest first, joins a neighbour glued along all
it shares or hinged on its costliest shared segment (small compact charts free of the shape rule), and an outline
smoothing of the full tree (`_smooth_full`): a patch moves to the chart it lies against when that lowers the cut's
outline weight, both charts stay disks, no cone closes and its fitted start crosses no border (boundary smoothing of
chart growing after Sander, Wood, Gortler, Snyder, Hoppe 2003 and Julius, Kraevoy, Sheffer 2005; our rules). Every level
replays the moves (`_level`), so lower levels' charts stay inside higher ones. Ours, as the kernel.

A union is refused when its layout would lie over itself (a flip-free disk layout is one-to-one exactly when its
border does not cross itself: Floater 2003, "One-to-one piecewise linear mappings over triangulations"). The merge
tree runs in rounds of rising stretch (`round_caps`), like the kernel.

Subset: a pole union (no grid) is measured here by flattening the whole union (start: the two layouts fitted onto
each other) instead of the band-only solve, so on meshes with poles its merge order can differ; its numbers say
"estimate". Keep going's cut moves have no twin: they run through the faster callable only.

`Charts` is sliceable: `step()` does about 4 ms of work and returns progress. When the adapter installed a faster
`build` callable, the stage hands the mesh to it (a job that never blocks) and only polls it; `tree` then is the
faster tree, with the same `level(x)` and `numbers()`.
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from . import uvflat

# Constants table: same names and values as the top of the matching .rs file (tests/test_uv_scaffold.py compares).
HIDDEN_L = 20.0
STRETCH_ALLOWED_PCT = 10.0
BUILD_STRETCH = 1.0
ROUND_FACTOR = 2.0
ROUND_LOWEST = 0.01
BAND_RINGS = 2
PX_PER_DIAGONAL = 1024.0
GRID_SLACK = 0.01
FOLD_DISCOUNT = 0.9
SEEN_WEIGHT = 4.0
SHAPE_MAX = 4.0
GAP_PX = 8.0
TUBE_ASPECT = 3.0
PIECE_ASPECT = 6.0
TUBE_L2_MEDIAN = 1.08
TUBE_L2_BAD = 1.3
TUBE_BAD_SHARE = 0.002
FREE_FACES = 64
# R10big: a mesh piece of more faces than REGION_MIN_FACES runs each merge round region by region (`_regions`), then
# across the region borders (the kernel runs the regions apart on worker threads; the charts are the same); its regions
# are halved until each has at most REGION_FACES faces.
REGION_MIN_FACES = 16384
REGION_FACES = 8192
REGION_NONE = 1 << 32  # outside a piece large enough (sorts after every region, as the kernel's u32::MAX)
SMOOTH_DEFECT_DEG = 10.0
SMOOTH_AREA = 1.5
NUMBERS_LEN = 22

NUMBER_COLUMNS = ("patches", "irregular", "turning", "traces", "euler_cut_patches", "euler_cut_edges", "not_disk_patches", "segments",
                  "merges", "grid_merges", "pole_merges", "quick_calls", "exact_evals", "build_ms", "stopped", "grid_patches", "dropped",
                  "top_stretch", "forced_edges", "px_per_unit", "lay_over", "tube_faces")
LEVEL_COLUMNS = ("charts", "not_disk", "merges_used", "segments_glued", "above_build", "ms")

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
# Quad topology view
# ---------------------------------------------------------------------------------------------------------------------


class Topo:
    """Corners, edges in first-use order (the order of `seen_edge_px`), edges and faces per vertex."""

    def __init__(self, indptr, flat, nv: int):
        self.indptr = [int(x) for x in indptr]
        self.flat = [int(v) for v in flat]
        self.nv, self.nf = nv, len(self.indptr) - 1
        self.cface = [0] * len(self.flat)
        for f in range(self.nf):
            for h in range(self.indptr[f], self.indptr[f + 1]):
                self.cface[h] = f
        ids: dict = {}
        uses: list = []
        self.cedge = [0] * len(self.flat)
        for h, v in enumerate(self.flat):
            w = self.flat[self.next(h)]
            key = (v, w) if v <= w else (w, v)
            e = ids.get(key)
            if e is None:
                e = ids[key] = len(uses)
                uses.append([])
            uses[e].append(h)
            self.cedge[h] = e
        self.ea = [k[0] for k in ids]
        self.eb = [k[1] for k in ids]
        self.euse = [(u[0], u[1] if len(u) > 1 else -1) for u in uses]
        self.wall = [len(u) != 2 or self.cface[u[0]] == self.cface[u[1]] for u in uses]
        self.vedge: list = [[] for _ in range(nv)]
        for e in range(len(self.ea)):
            self.vedge[self.ea[e]].append(e)
            self.vedge[self.eb[e]].append(e)
        self.vface: list = [[] for _ in range(nv)]
        for h, v in enumerate(self.flat):
            self.vface[v].append(self.cface[h])
        self.border_v = [False] * nv
        for e, w in enumerate(self.wall):
            if w:
                self.border_v[self.ea[e]] = self.border_v[self.eb[e]] = True

    @property
    def ne(self) -> int:
        return len(self.ea)

    def next(self, h: int) -> int:
        f = self.cface[h]
        return self.indptr[f] if h + 1 == self.indptr[f + 1] else h + 1

    def at(self, h: int, v: int) -> int:
        return h if self.flat[h] == v else self.next(h)

    def faces_of(self, e: int) -> tuple:
        a, b = self.euse[e]
        return self.cface[a], (self.cface[b] if b >= 0 else -1)

    def other(self, e: int, v: int) -> int:
        return self.eb[e] if self.ea[e] == v else self.ea[e]

    def face_corners(self, f: int) -> range:
        return range(self.indptr[f], self.indptr[f + 1])

    def straight_next(self, v: int, e: int):
        at = self.vedge[v]
        if len(at) != 4 or self.border_v[v]:
            return None
        fe = self.faces_of(e)
        found = None
        for x in at:
            if x == e:
                continue
            if any(f >= 0 and f in fe for f in self.faces_of(x)):
                continue
            if found is not None:
                return None
            found = x
        return found


class _Dsu:
    """Union-find whose root is the smallest id (the kernel's `topo::Dsu`)."""

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra < rb:
            self.p[rb] = ra
        elif rb < ra:
            self.p[ra] = rb


def _drive(gen):
    """Run a sliced generator to its end; its return value."""

    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


class _Undo:
    """Union-find with undo (union by size, no path compression) over corners: the wedges of a cut state."""

    def __init__(self, n: int):
        self.parent, self.size, self.log = list(range(n)), [1] * n, []

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            x = p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.log.append(rb)
        return True

    def mark(self) -> int:
        return len(self.log)

    def undo(self, mark: int) -> None:
        while len(self.log) > mark:
            rb = self.log.pop()
            ra = self.parent[rb]
            self.size[ra] -= self.size[rb]
            self.parent[rb] = rb

    def commit(self) -> None:
        self.log.clear()


def _glue(t: Topo, dsu: _Undo, edges) -> tuple:
    m = 0
    for e in edges:
        h0, h1 = t.euse[e]
        for v in (t.ea[e], t.eb[e]):
            m += dsu.union(t.at(h0, v), t.at(h1, v))
    return len(edges), m


def _glue_disk(t: Topo, dsu: _Undo, segments, order, chi: int) -> tuple:
    pending, glued, cost = list(order), [], 0.0
    while True:
        left, before = [], len(glued)
        for s in pending:
            mk = dsu.mark()
            g, m = _glue(t, dsu, segments[s].edges)
            if chi + g - m == 1:
                chi = 1
                glued.append(s)
                cost += segments[s].cost
            else:
                dsu.undo(mk)
                left.append(s)
        pending = left
        if len(glued) == before or not pending:
            break
    return glued, cost


def _wedges(t: Topo, cut) -> list:
    return _drive(_wedges_steps(t, cut))


def _wedges_steps(t: Topo, cut):
    d = _Dsu(len(t.flat))
    for e in range(t.ne):
        if e & 4095 == 4095:
            yield
        if not cut[e] and not t.wall[e]:
            h0, h1 = t.euse[e]
            for v in (t.ea[e], t.eb[e]):
                d.union(t.at(h0, v), t.at(h1, v))
    return [d.find(c) for c in range(len(t.flat))]


def _chi_per(t: Topo, cut, patch, np_: int) -> list:
    return _drive(_chi_per_steps(t, cut, patch, np_))


def _chi_per_steps(t: Topo, cut, patch, np_: int):
    wr = yield from _wedges_steps(t, cut)
    yield
    chi, first = [0] * np_, set()
    for f in range(t.nf):
        if f & 4095 == 4095:
            yield
        p = patch[f]
        chi[p] += 1
        for h in t.face_corners(f):
            chi[p] -= 1
            if wr[h] not in first:
                first.add(wr[h])
                chi[p] += 1
    for e in range(t.ne):
        if e & 8191 == 8191:
            yield
        if not cut[e] and not t.wall[e]:
            chi[patch[t.cface[t.euse[e][0]]]] += 1
    return chi


class _Local:
    """A chart as its own mesh: vertices are wedges, so a cut seam inside the chart is open."""

    def __init__(self, t: Topo, points: np.ndarray, faces, root):
        ids: dict = {}
        verts, ip, fl, corner = [], [0], [], []
        for f in faces:
            for h in t.face_corners(f):
                r = root(h)
                k = ids.get(r)
                if k is None:
                    k = ids[r] = len(verts)
                    verts.append(t.flat[h])
                fl.append(k)
                corner.append(h)
            ip.append(len(fl))
        self.points = points[np.array(verts, dtype=np.int64)] if verts else np.zeros((0, 3))
        self.indptr, self.flat = np.array(ip, np.int64), np.array(fl, np.int64)
        self.corner, self.faces = np.array(corner, np.int64), list(faces)

    @property
    def n(self) -> int:
        return len(self.points)

    def gather(self, corner_uv: np.ndarray) -> np.ndarray:
        s = np.zeros((self.n, 2))
        np.add.at(s, self.flat, corner_uv[self.corner])
        c = np.bincount(self.flat, minlength=self.n).astype(float)
        return s / np.maximum(c, 1.0)[:, None]

    def scatter(self, uv: np.ndarray, corner_uv: np.ndarray) -> None:
        corner_uv[self.corner] = uv[self.flat]


def _quantile(values, q: float = uvflat.STRETCH_QUANTILE) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0:
        return float("nan")
    return float(v[min(int(round(q * (len(v) - 1))), len(v) - 1)])


def _align(pairs) -> tuple:
    """Rigid 2D fit q = R p + t (rotation and translation, least squares)."""

    p = np.array([x[0] for x in pairs], float)
    q = np.array([x[1] for x in pairs], float)
    pc, qc = p.mean(0), q.mean(0)
    a, b = p - pc, q - qc
    th = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    c, s = np.cos(th), np.sin(th)
    return (c, s), (qc[0] - (c * pc[0] - s * pc[1]), qc[1] - (s * pc[0] + c * pc[1]))


def _rigid(r, uv: np.ndarray) -> np.ndarray:
    (c, s), t = r
    return np.stack([c * uv[:, 0] - s * uv[:, 1] + t[0], s * uv[:, 0] + c * uv[:, 1] + t[1]], 1)


def round_caps(build: float) -> list:
    """Stretch caps of the merge rounds (as stretch - 1), rising, the last one `build` (kernel: `round_caps`)."""

    x = STRETCH_ALLOWED_PCT / 100.0
    while x / ROUND_FACTOR >= ROUND_LOWEST:
        x /= ROUND_FACTOR
    caps = []
    while x < build:
        caps.append(x)
        x *= ROUND_FACTOR
    caps.append(build)
    return caps


def _sides_cross(p: np.ndarray, q: np.ndarray, v: np.ndarray) -> bool:
    """Do two border sides (p[i] to q[i], wedge ids v[i]) cross, each one's ends strictly on both sides of the other?
    Sides sharing a wedge touch and are not tested. Broad phase: a cell grid no finer than the longest side."""

    m = len(p)
    if m < 2:
        return False
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        return True  # a side without a position: the layout failed
    lo, hi = np.minimum(p, q), np.maximum(p, q)
    org = lo.min(0)
    span = float((hi.max(0) - org).max())
    cell = max(float((hi - lo).max()), span / max(np.sqrt(m), 1.0), 1e-300)
    c0, c1 = np.floor((lo - org) / cell).astype(np.int64), np.floor((hi - org) / cell).astype(np.int64)
    nx = int(c1[:, 0].max()) + 1
    ids, cells = [], []
    for dx in (0, 1):  # a side spans at most two cells per axis
        for dy in (0, 1):
            x, y = c0[:, 0] + dx, c0[:, 1] + dy
            ok = (x <= c1[:, 0]) & (y <= c1[:, 1])
            ids.append(np.nonzero(ok)[0])
            cells.append(y[ok] * nx + x[ok])
    ids, cells = np.concatenate(ids), np.concatenate(cells)
    order = np.argsort(cells, kind="stable")
    ids, cells = ids[order], cells[order]
    starts = np.flatnonzero(np.r_[True, cells[1:] != cells[:-1]])
    ends = np.r_[starts[1:], len(cells)]
    ii, jj = [], []
    for a, b in zip(starts, ends):
        if b - a > 1:
            i, j = np.triu_indices(b - a, 1)
            ii.append(ids[a + i])
            jj.append(ids[a + j])
    if not ii:
        return False
    i, j = np.concatenate(ii), np.concatenate(jj)
    share = (v[i, 0] == v[j, 0]) | (v[i, 0] == v[j, 1]) | (v[i, 1] == v[j, 0]) | (v[i, 1] == v[j, 1])
    i, j = i[~share], j[~share]

    def orient(a, b, c):
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    tol = 1e-7 * np.linalg.norm(q[i] - p[i], axis=1) * np.linalg.norm(q[j] - p[j], axis=1)

    def apart(x, y):
        return ((x > tol) & (y < -tol)) | ((x < -tol) & (y > tol))

    hit = apart(orient(p[i], q[i], p[j]), orient(p[i], q[i], q[j])) & apart(orient(p[j], q[j], p[i]), orient(p[j], q[j], q[i]))
    return bool(hit.any())


def _lies_over(local, uv: np.ndarray, before: np.ndarray) -> bool:
    """Would a union laid out as `uv` (per local vertex) lie over itself? More faces turned over than `before` (per
    corner, the two layouts placed as the union starts), or two border sides cross."""

    ip, fl = local.indptr, local.flat
    if len(fl) == 0:
        return False
    sizes = np.diff(ip)
    nxt = np.arange(1, len(fl) + 1)
    nxt[ip[1:] - 1] = ip[:-1]
    now = np.asarray(uv, float)[fl]

    def turned(c):
        cr = c[:, 0] * c[nxt, 1] - c[nxt, 0] * c[:, 1]
        area = np.add.reduceat(cr, ip[:-1])
        return int(((sizes >= 3) & ~(area > 0)).sum())

    if turned(now) > turned(np.asarray(before, float)):
        return True
    a, b = fl, fl[nxt]
    key = np.minimum(a, b) * (local.n + 1) + np.maximum(a, b)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    open_ = cnt[inv.reshape(-1)] == 1
    return _sides_cross(now[open_], now[nxt][open_], np.stack([a[open_], b[open_]], 1))


def _key(added: float, cost: float, irregular: float) -> tuple:
    """Rank of a merge (R4): most compact union first, then cost removed; a union that cannot be measured last."""

    if not np.isfinite(added):
        return (2, 0.0, -cost)
    return (0, irregular, -cost)


def _outline_w(length_px: float, seen_px: float, cos: float, out_of_sight: bool) -> float:
    return length_px * (1.0 - FOLD_DISCOUNT * (1.0 - max(cos, 0.0))) + (SEEN_WEIGHT * seen_px if out_of_sight else 0.0)


def _irregularity(outline: float, area: float) -> float:
    """outline^2 / (4 pi area); a disk is 1."""

    return outline * outline / (4.0 * math.pi * area) if area > 0.0 else float("inf")


def _newell(points, t: Topo, f: int) -> list:
    n = [0.0, 0.0, 0.0]
    for h in t.face_corners(f):
        p, q = points[t.flat[h]], points[t.flat[t.next(h)]]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return n


def _face_area(points, t: Topo, f: int) -> float:
    n = _newell(points, t, f)
    return 0.5 * math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])


def _fold_cos(points, t: Topo) -> list:
    """Per edge: cosine of the angle between its two faces' normals (1 on a border edge)."""

    normals = []
    for f in range(t.nf):
        n = _newell(points, t, f)
        ln = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        normals.append([n[0] / ln, n[1] / ln, n[2] / ln] if ln > 0.0 else [0.0, 0.0, 0.0])
    out = []
    for e in range(t.ne):
        h0, h1 = t.euse[e]
        if h1 < 0:
            out.append(1.0)
            continue
        a, b = normals[t.cface[h0]], normals[t.cface[h1]]
        out.append(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
    return out


def _total(values) -> float:
    """Plain left-to-right sum (as the kernel; `sum()` of floats compensates)."""

    acc = 0.0
    for x in values:
        acc += x
    return acc


class Segment:
    __slots__ = ("edges", "p", "q", "cost", "seen", "weight", "length")

    def __init__(self, edges, p, q, cost, seen, weight=0.0, length=0.0):
        self.edges, self.p, self.q, self.cost, self.seen = edges, p, q, cost, seen
        self.weight, self.length = weight, length


def _regions(t, points, members) -> list:
    """R10big: regions of the large mesh pieces, per patch (REGION_NONE elsewhere), as the kernel's `Tree::regions`: a
    piece (faces joined across edges) of more than REGION_MIN_FACES faces is cut in two by face count across the longest
    side of the box of its patches' centres, and each half again while it has more than REGION_FACES (recursive coordinate bisection,
    Berger and Bokhari 1987); regions numbered in that order, pieces by their first patch."""

    d = _Dsu(t.nf)
    for h0, h1 in t.euse:
        if h1 >= 0:
            d.union(t.cface[h0], t.cface[h1])
    piece_of: dict = {}
    pieces: list = []
    for p, m in enumerate(members):
        if not len(m):
            continue
        k = piece_of.setdefault(d.find(m[0]), len(pieces))
        if k == len(pieces):
            pieces.append([0, []])
        pieces[k][0] += len(m)
        pieces[k][1].append(p)
    region = [REGION_NONE] * len(members)
    nxt = 0
    for n, ps in pieces:
        if n <= REGION_MIN_FACES:
            continue
        centre = {}
        for p in ps:
            c, k = [0.0, 0.0, 0.0], 0
            for f in members[p]:
                for h in t.face_corners(f):
                    v = t.flat[h]
                    for x in range(3):
                        c[x] += points[v][x]
                    k += 1
            centre[p] = [x / max(k, 1) for x in c]
        out, todo = [], [ps]
        while todo:
            st = todo.pop()
            total = sum(len(members[p]) for p in st)
            if total <= REGION_FACES or len(st) < 2:
                out.append(st)
                continue
            lo = [min(centre[p][x] for p in st) for x in range(3)]
            hi = [max(centre[p][x] for p in st) for x in range(3)]
            axis = max(range(3), key=lambda x: (hi[x] - lo[x], -x))
            st = sorted(st, key=lambda p: (centre[p][axis], p))
            acc, cut = 0, len(st) - 1
            for i, p in enumerate(st):
                acc += len(members[p])
                if 2 * acc >= total:
                    cut = i + 1
                    break
            cut = min(max(cut, 1), len(st) - 1)
            todo.append(st[cut:])  # the lower half first
            todo.append(st[:cut])
        for st in out:
            for p in st:
                region[p] = nxt
            nxt += 1
    return region


def _members(root) -> list:
    out: list = [[] for _ in range(len(root))]
    for p, r in enumerate(root):
        out[r].append(p)
    return out


class ChartTree:
    """Patches, segments and recorded merges of one quad mesh; `level(x)` gives the charts at Stretch allowed x."""

    def __init__(self):
        self.merges: list = []  # (segment ids, stretch, cost, pole)
        self._numbers = np.zeros(NUMBERS_LEN)
        self._cache: dict = {}
        self._smooth = None  # (moves, start layout) of the outline smoothing, made on the first level

    def numbers(self) -> np.ndarray:
        return self._numbers.copy()

    def patches(self) -> np.ndarray:
        return np.array(self.patch, np.int64)

    def level(self, x: float, hide: bool = True):
        """(chart per face, cut per edge, numbers in `LEVEL_COLUMNS`), cached per (x, hide). `hide` False: the charts
        before their borders slide out of sight (the first result is shown from these while the moves are made)."""

        return _drive(self.level_steps(x, hide))

    def level_steps(self, x: float, hide: bool = True):
        """`level` as a generator, yielding between short pieces."""

        key = (float(x), bool(hide))
        if key not in self._cache:
            self._cache[key] = yield from self._level_steps(*key)
        chart, cut, numbers = self._cache[key]
        return chart.copy(), cut.copy(), numbers.copy()

    def _level(self, x: float, hide: bool = True):
        return _drive(self._level_steps(x, hide))

    def _level_steps(self, x: float, hide: bool = True):
        """`_level` as a generator, yielding between short pieces (the hidden-parts moves run it in slices)."""

        started = time.perf_counter()
        t = self.topo
        root, glued, chi, used, segs_glued = yield from self._replay_steps(1.0 + x / 100.0)
        moves, _ = yield from self._smoothing_steps()
        if moves:
            pfaces, psegs = self._smooth_ctx()
            yield
            members = _members(root)
            for p, q in moves:
                yield  # one move a step: each checks its chart is still a disk
                if root[p] != root[q]:
                    self._move_patch(pfaces, psegs, p, root[q], root, glued, members, [])
        cut = np.asarray(self.cut_base, bool) & ~glued
        yield
        if moves:  # a chart that gave a patch away can fall apart below the full tree: one chart per piece
            d = _Dsu(t.nf)
            for e in range(t.ne):
                if e & 4095 == 4095:
                    yield
                if not cut[e]:
                    f0, f1 = t.faces_of(e)
                    d.union(f0, f1)
            ids: dict = {}
            chart = np.array([ids.setdefault(d.find(f), len(ids)) for f in range(t.nf)], np.int64)
            yield
            not_disk = sum(1 for c in (yield from _chi_per_steps(t, cut.tolist(), chart.tolist(), len(ids))) if c != 1)
        else:
            ids = {}
            chart = np.array([ids.setdefault(root[self.patch[f]], len(ids)) for f in range(t.nf)], np.int64)
            not_disk = sum(1 for r in ids if chi[r] != 1)
        from . import uvhide  # R7hide: borders slid out of sight (imports this module)

        slid = (yield from uvhide.apply_steps(self, chart, cut)) if hide else None
        chart, cut, n_charts, not_disk = slid if slid is not None else (chart, cut, len(ids), not_disk)
        numbers = np.array([n_charts, not_disk, used, segs_glued, float(x / 100.0 > self.build_stretch), (time.perf_counter() - started) * 1e3])
        return chart, cut, numbers

    def _replay(self, thr: float):
        """The recorded merges with stretch <= thr, in order: (root per patch, glued per edge, chi per root, merges used,
        segments glued) (kernel: `replay`)."""

        return _drive(self._replay_steps(thr))

    def _replay_steps(self, thr: float):
        t = self.topo
        dsu = _Undo(len(t.flat))
        for e in range(t.ne):
            if e & 2047 == 2047:
                yield
            if not self.cut_base[e]:
                _glue(t, dsu, [e])
        dsu.commit()
        yield
        pd = _Dsu(self.n_patches)
        chi = list(self.patch_chi)
        glued = np.zeros(t.ne, bool)
        used = segs_glued = 0
        for i, (segs, stretch, _, _) in enumerate(self.merges):
            if i & 255 == 255:
                yield
            if not stretch <= thr:
                continue
            used += 1
            for s in segs:
                sg = self.segments[s]
                rp, rq = pd.find(sg.p), pd.find(sg.q)
                chi0 = chi[rp] + chi[rq] if rp != rq else chi[rp]
                mk = dsu.mark()
                g, m = _glue(t, dsu, sg.edges)
                if chi0 + g - m == 1:
                    dsu.commit()
                    pd.union(rp, rq)
                    chi[pd.find(rp)] = 1
                    glued[sg.edges] = True
                    segs_glued += 1
                else:
                    dsu.undo(mk)
        return [pd.find(p) for p in range(self.n_patches)], glued, chi, used, segs_glued

    def _smoothing(self):
        """(moves, start layout) of the outline smoothing of the full tree, made once (kernel: `smoothing`)."""

        return _drive(self._smoothing_steps())

    def _smoothing_steps(self):
        if self._smooth is None:
            root, glued, chi, _, _ = yield from self._replay_steps(float("inf"))
            self._smooth = yield from self._smooth_full_steps(root, glued, chi)
        return self._smooth

    def _smooth_ctx(self):
        t = self.topo
        pfaces: list = [[] for _ in range(self.n_patches)]
        for f in range(t.nf):
            pfaces[self.patch[f]].append(f)
        psegs: list = [[] for _ in range(self.n_patches)]
        for s, sg in enumerate(self.segments):
            if sg.p != sg.q:
                psegs[sg.p].append(s)
                psegs[sg.q].append(s)
        return pfaces, psegs

    def _set_seg(self, glued, s: int, on: bool, undo: list) -> None:
        edges = self.segments[s].edges
        if bool(glued[edges[0]]) != on:
            undo.append((s, not on))
            glued[edges] = on

    def _move_patch(self, pfaces, psegs, p: int, b: int, root, glued, members, undo):
        """Move patch p into chart b (kernel: `move_patch`): (weight glued to b, is b with p one disk, wedge find)."""

        a = root[p]
        to_b = []
        for s in psegs[p]:
            sg = self.segments[s]
            c = root[sg.q if sg.p == p else sg.p]
            if c == a:
                self._set_seg(glued, s, False, undo)
            elif c == b:
                to_b.append(s)
        root[p] = b
        members[a].remove(p)
        members[b].append(p)
        for s in to_b:
            self._set_seg(glued, s, True, undo)
        ok, wedge = self._is_disk(members[b], pfaces, root, glued, b)
        if ok:
            return _total(self.segments[s].weight for s in to_b), True, wedge
        if not to_b:
            return 0.0, False, wedge
        best = max(to_b, key=lambda s: (self.segments[s].weight, -s))
        for s in to_b:
            self._set_seg(glued, s, s == best, undo)
        ok, wedge = self._is_disk(members[b], pfaces, root, glued, b)
        return self.segments[best].weight, ok, wedge

    def _is_disk(self, patches, pfaces, root, glued, r: int):
        """Are these patches' faces (root r) one disk under the glue? (bool, wedge find over their corners)."""

        t = self.topo
        cpar: dict = {}
        fpar: dict = {}

        def find(par, x):
            while par.get(x, x) != x:
                par[x] = par.get(par[x], par[x])
                x = par[x]
            return x

        def join(par, a, b):
            ra, rb = find(par, a), find(par, b)
            if ra == rb:
                return False
            par[max(ra, rb)] = min(ra, rb)
            return True

        chi = comps = sum(len(pfaces[p]) for p in patches)
        for p in patches:
            for f in pfaces[p]:
                for h in t.face_corners(f):
                    e = t.cedge[h]
                    h0, h1 = t.euse[e]
                    if h != h0 or h1 < 0 or t.wall[e] or (self.cut_base[e] and not glued[e]):
                        continue
                    g = t.cface[h1]
                    if root[self.patch[g]] != r:
                        continue
                    chi += 1
                    for v in (t.ea[e], t.eb[e]):
                        chi -= int(join(cpar, t.at(h0, v), t.at(h1, v)))
                    comps -= int(join(fpar, f, g))
        return comps == 1 and chi == 1, (lambda x: find(cpar, x))

    def _smooth_full(self, root, glued, chi):
        """Smoother chart outlines on the full tree (kernel: `smooth`): (moves, start layout)."""

        return _drive(self._smooth_full_steps(root, glued, chi))

    def _smooth_full_steps(self, root, glued, chi):
        t = self.topo
        pts = np.asarray(self.points, float)
        pfaces, psegs = self._smooth_ctx()
        members = _members(root)
        defect_max = math.radians(SMOOTH_DEFECT_DEG)
        angle_sum = None  # corner angles summed per vertex, made when a move first needs them
        start = np.asarray(self.start_uv, float).reshape(-1, 2).copy()
        moves: list = []
        while True:
            before = len(moves)
            for p in range(self.n_patches):
                yield  # one patch a step: a move is checked against its chart's outline and the corners around it
                a = root[p]
                if chi[a] != 1:
                    continue
                wa = 0.0
                wb: list = []  # [chart, weight, a patch of it]
                for s in psegs[p]:
                    sg = self.segments[s]
                    q = sg.q if sg.p == p else sg.p
                    c = root[q]
                    if c == a:
                        if glued[sg.edges[0]]:
                            wa += sg.weight
                    else:
                        for x in wb:
                            if x[0] == c:
                                x[1] += sg.weight
                                break
                        else:
                            wb.append([c, sg.weight, q])
                if not wb:
                    continue
                b, w, q = max(wb, key=lambda x: (x[1], -x[0]))
                if chi[b] != 1 or not w - wa > 1e-9 * (w + wa):
                    continue
                undo: list = []
                got, disk, wedge = self._move_patch(pfaces, psegs, p, b, root, glued, members, undo)
                ok = disk and got == w and got - wa > 1e-9 * (got + wa)
                if ok and angle_sum is None:
                    ip = np.asarray(t.indptr)
                    fl = np.asarray(t.flat)
                    h = np.arange(len(fl))
                    f = np.asarray(t.cface)
                    nxt = np.where(h + 1 == ip[f + 1], ip[f], h + 1)
                    prv = np.where(h == ip[f], ip[f + 1] - 1, h - 1)
                    x, y = pts[fl[prv]] - pts[fl], pts[fl[nxt]] - pts[fl]
                    ang = np.arctan2(np.linalg.norm(np.cross(x, y), axis=1), np.einsum("ij,ij->i", x, y))
                    angle_sum = np.bincount(fl, weights=ang, minlength=t.nv).tolist()
                if ok:
                    for s, was in undo:
                        if was:
                            continue
                        for e in self.segments[s].edges:
                            for v in (t.ea[e], t.eb[e]):
                                inside = (not t.border_v[v] and all(root[self.patch[f]] == b for f in t.vface[v])
                                          and all(not self.cut_base[x] or glued[x] for x in t.vedge[v]))
                                if inside and abs(2.0 * math.pi - angle_sum[v]) > defect_max:
                                    ok = False
                old = None
                if ok:
                    pairs, snap = [], {}
                    for s in psegs[p]:
                        sg = self.segments[s]
                        o = sg.q if sg.p == p else sg.p
                        if root[o] != b or not glued[sg.edges[0]]:
                            continue
                        for e in sg.edges:
                            h0, h1 = t.euse[e]
                            hp, hb = (h0, h1) if self.patch[t.cface[h0]] == p else (h1, h0)
                            for v in (t.ea[e], t.eb[e]):
                                cp, cb = t.at(hp, v), t.at(hb, v)
                                pairs.append((start[cp].copy(), start[cb].copy()))
                                snap[v] = start[cb].copy()
                    rigid = _align(pairs)
                    corners = np.array([h for f in pfaces[p] for h in t.face_corners(f)], np.int64)
                    old = start[corners].copy()
                    start[corners] = _rigid(rigid, start[corners])
                    for h in corners:
                        got_v = snap.get(t.flat[h])
                        if got_v is not None:
                            start[h] = got_v
                    for f in pfaces[p]:
                        r = list(t.face_corners(f))
                        n = len(r)
                        c = start[r]
                        ratio = 0.5 * float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])) / max(_face_area(self.points, t, f), 1e-300)
                        turns = [(c[i, 0] - c[i - 1, 0]) * (c[(i + 1) % n, 1] - c[i - 1, 1]) - (c[(i + 1) % n, 0] - c[i - 1, 0]) * (c[i, 1] - c[i - 1, 1]) for i in range(n)]
                        if not (ratio <= SMOOTH_AREA and ratio * SMOOTH_AREA >= 1.0 and all(x > 0.0 for x in turns)):
                            ok = False
                            break
                    if ok:
                        sp, sq, sv = [], [], []
                        for m in members[b]:
                            for f in pfaces[m]:
                                for h in t.face_corners(f):
                                    e = t.cedge[h]
                                    h0, h1 = t.euse[e]
                                    o = h1 if h == h0 else h0
                                    if o >= 0 and not t.wall[e] and not (self.cut_base[e] and not glued[e]) and root[self.patch[t.cface[o]]] == b:
                                        continue
                                    nx = t.next(h)
                                    sp.append(start[h])
                                    sq.append(start[nx])
                                    sv.append((wedge(h), wedge(nx)))
                        ok = not _sides_cross(np.array(sp), np.array(sq), np.array(sv))
                if ok and members[a]:
                    ok = self._is_disk(members[a], pfaces, root, glued, a)[0]
                if ok:
                    moves.append((p, q))
                else:
                    if old is not None:
                        start[corners] = old
                    for s, was in reversed(undo):
                        glued[self.segments[s].edges] = was
                    root[p] = a
                    members[b].remove(p)
                    members[a].append(p)
            if len(moves) == before:
                break
        return moves, (start if moves else None)


# ---------------------------------------------------------------------------------------------------------------------
# The sliced build
# ---------------------------------------------------------------------------------------------------------------------


class Charts:
    """Build the Row Charts tree (spec 6.1-6.4) in slices: `step()` works about `budget` seconds and returns progress
    in 0..1; `done`, `tree`. Inputs as the kernel: `part_label` per face (materials folded in by the caller), `forced`
    vertex pairs, `seen_edge_px` per edge in first-use order."""

    def __init__(self, points, indptr, flat, part_label=None, forced=None, seen_edge_px=None, px_per_unit: float = 0.0, out_of_sight: bool = False,
                 hidden_l: float = 0.0, build_stretch: float = 0.0, budget: float = 0.004, use_fast: bool = True, merge: bool = True):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.indptr, self.flat = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
        self.part_label = None if part_label is None or len(part_label) == 0 else np.asarray(part_label, np.int64)
        self.forced = None if forced is None or len(forced) == 0 else np.asarray(forced, np.int64).reshape(-1, 2)
        self.seen_edge_px = None if seen_edge_px is None or len(seen_edge_px) == 0 else np.asarray(seen_edge_px, np.float64)
        self.px_per_unit, self.out_of_sight, self.hidden_l = float(px_per_unit), bool(out_of_sight), float(hidden_l)
        self.build_stretch = float(build_stretch) if build_stretch > 0 else BUILD_STRETCH
        self.budget, self.done, self.tree, self._stop = float(budget), False, None, False
        self.merge = bool(merge)  # False: patches and segments only (no layouts, no merges)
        self._job = None
        self._fast = FAST.get("build") if use_fast else None
        self._work = self._run()

    def stop(self) -> None:
        """End early; the tree reached so far is kept (every level of it is valid)."""

        self._stop = True
        if self._job is not None:
            self._job.stop()

    def step(self) -> float:
        if self.done:
            return 1.0
        if self._fast is not None:
            if self._job is None:
                self._job = self._fast(self.points.ravel(), self.indptr, self.flat, self.part_label, None if self.forced is None else self.forced.ravel(),
                                       self.seen_edge_px, self.px_per_unit, self.out_of_sight, self.hidden_l, self.build_stretch)
                if self._job is None:
                    self._fast = None
                    return 0.0
                if self._stop:
                    self._job.stop()
            if not self._job.done():
                return 0.5
            self.tree = self._job.result()
            self.done = True
            return 1.0
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

    # -----------------------------------------------------------------------------------------------------------------
    def _run(self):
        started = time.perf_counter()
        points = self.points
        nv, nf = len(points), len(self.indptr) - 1
        if self.part_label is not None and len(self.part_label) != nf:
            raise ValueError("part_label must hold one label per face")
        t = Topo(self.indptr, self.flat, nv)
        yield 0.02
        ne = t.ne
        if self.seen_edge_px is not None and len(self.seen_edge_px) != ne:
            raise ValueError(f"seen_edge_px must hold one number per edge ({ne})")
        tree = ChartTree()
        num = tree._numbers
        if self.px_per_unit > 0:
            ppu = self.px_per_unit
        else:
            diag = float(np.linalg.norm(points.max(0) - points.min(0))) if nv else 0.0
            ppu = PX_PER_DIAGONAL / diag if diag > 0 and np.isfinite(diag) else 1.0
        num[19] = ppu
        L = self.hidden_l if self.hidden_l > 0 else HIDDEN_L
        ea, eb = np.array(t.ea, np.int64), np.array(t.eb, np.int64)
        length_px = np.sqrt(((points[ea] - points[eb]) ** 2).sum(1)) * ppu if ne else np.zeros(0)
        seen = np.maximum(self.seen_edge_px, 0.0) if self.seen_edge_px is not None else length_px.copy()
        cost = seen + length_px / L if self.out_of_sight else length_px.copy()
        cost_l, seen_l = cost.tolist(), seen.tolist()
        fold = _fold_cos(points.tolist(), t)
        length_l = length_px.tolist()
        outline_l = [_outline_w(length_l[e], seen_l[e], fold[e], self.out_of_sight) for e in range(ne)]
        # Forced lines.
        forced_line = [False] * ne
        if self.forced is not None:
            lookup = {(t.ea[e], t.eb[e]): e for e in range(ne)}
            for a, b in self.forced.tolist():
                if a < 0 or b < 0:
                    continue
                e = lookup.get((min(a, b), max(a, b)))
                if e is not None:
                    forced_line[e] = not t.wall[e]
        if self.part_label is not None:
            lab = self.part_label.tolist()
            for e in range(ne):
                if not t.wall[e]:
                    f0, f1 = t.faces_of(e)
                    if lab[f0] != lab[f1]:
                        forced_line[e] = True
        fixed = [t.wall[e] or forced_line[e] for e in range(ne)]
        num[18] = sum(forced_line)
        yield 0.05
        # Irregular points (6.1).
        irregular = [False] * nv
        for f in range(nf):
            r = t.face_corners(f)
            if len(r) != 4:
                for h in r:
                    irregular[t.flat[h]] = True
        fcount = [0] * nv
        for e in range(ne):
            if forced_line[e]:
                fcount[t.ea[e]] += 1
                fcount[t.eb[e]] += 1
        turning = [False] * nv
        for v in range(nv):
            at = t.vedge[v]
            nonmanifold = any(t.wall[e] and t.euse[e][1] >= 0 for e in at)
            if (len(at) != 3 if t.border_v[v] else len(at) != 4) or nonmanifold:
                irregular[v] = True
            k = fcount[v]
            if k == 0:
                turns = False
            elif k == 2:
                if t.border_v[v] or irregular[v]:
                    turns = False
                else:
                    pair = [e for e in at if forced_line[e]]
                    turns = t.straight_next(v, pair[0]) != pair[1]
            else:
                turns = not t.border_v[v]
            if turns and not irregular[v]:
                turning[v] = irregular[v] = True
        num[1], num[2] = sum(irregular), sum(turning)
        yield 0.08
        # Traces (6.2.2).
        blocked = [t.border_v[v] or irregular[v] or fcount[v] > 0 for v in range(nv)]
        claimed = [False] * nv
        traced = [False] * ne
        active = []
        for v in range(nv):
            if irregular[v]:
                claimed[v] = True
                active.extend((v, e) for e in t.vedge[v] if not fixed[e])
        num[3] = len(active)
        while active:
            hits: dict = {}
            for v, e in active:
                w = t.other(e, v)
                hits[w] = hits.get(w, 0) + 1
            nxt = []
            for v, e in active:
                traced[e] = True
                w = t.other(e, v)
                if blocked[w] or claimed[w] or hits[w] > 1:
                    continue
                n2 = t.straight_next(w, e)
                if n2 is not None and not fixed[n2]:
                    nxt.append((w, n2))
            for v, e in active:
                claimed[t.other(e, v)] = True
            active = nxt
            yield 0.1
        sep = [(traced[e] or forced_line[e]) and not t.wall[e] for e in range(ne)]
        # Thin tubes (R5): a part that is a long strip of surface is laid out as strip pieces.
        tubes = yield from _tube_cuts(t, points, cost_l, forced_line, self._flatten)
        tube_uv: dict = {}
        for faces_, cut_, cuv in tubes:
            for f in faces_:
                for h in t.face_corners(f):
                    sep[t.cedge[h]] = False
            for e in cut_:
                sep[e] = True
            tube_uv.update(cuv)
        num[21] = sum(len(x[0]) for x in tubes)
        cut_base = [sep[e] or t.wall[e] for e in range(ne)]
        # Patches and the Euler check (6.2.3).
        d = _Dsu(nf)
        for e in range(ne):
            if not cut_base[e]:
                f0, f1 = t.faces_of(e)
                d.union(f0, f1)
        ids: dict = {}
        patch = [ids.setdefault(d.find(f), len(ids)) for f in range(nf)]
        n_patches = len(ids)
        members: list = [[] for _ in range(n_patches)]
        for f in range(nf):
            members[patch[f]].append(f)
        yield 0.15
        euler_cut = [False] * ne
        cut_patches = 0
        for rnd in range(2):
            chi = _chi_per(t, cut_base, patch, n_patches)
            bad = [p for p in range(n_patches) if chi[p] != 1]
            if not bad:
                break
            for p in bad:
                path = _annulus_row(t, cut_base, patch, p, members[p], cost_l) if rnd == 0 else None
                if path is None:
                    path = _tree_cotree(t, cut_base, patch, p, members[p], cost_l)
                if path and rnd == 0:
                    cut_patches += 1
                for e in path:
                    sep[e] = euler_cut[e] = True
            cut_base = [sep[e] or t.wall[e] for e in range(ne)]
            yield 0.2
        yield 0.18
        patch_chi = _chi_per(t, cut_base, patch, n_patches)
        num[0], num[4], num[5], num[6] = n_patches, cut_patches, sum(euler_cut), sum(1 for c in patch_chi if c != 1)
        # Segments.
        gluable = [sep[e] and not fixed[e] for e in range(ne)]
        cut_deg = [0] * nv
        for e in range(ne):
            if cut_base[e]:
                cut_deg[t.ea[e]] += 1
                cut_deg[t.eb[e]] += 1

        def pair_of(e):
            f0, f1 = t.faces_of(e)
            a, b = patch[f0], patch[f1]
            return (min(a, b), max(a, b))

        seg_of = [-1] * ne
        segments = []
        for e0 in range(ne):
            if not gluable[e0] or seg_of[e0] >= 0:
                continue
            sid, pr = len(segments), pair_of(e0)
            lst = [e0]
            seg_of[e0] = sid
            for end in (t.ea[e0], t.eb[e0]):
                v, e = end, e0
                while not (irregular[v] or cut_deg[v] != 2):
                    x = next((x for x in t.vedge[v] if x != e and cut_base[x]), None)
                    if x is not None and gluable[x] and seg_of[x] < 0 and pair_of(x) == pr:
                        seg_of[x] = sid
                        lst.append(x)
                        v, e = t.other(x, v), x
                    else:
                        break
            segments.append(Segment(lst, pr[0], pr[1], sum(cost_l[x] for x in lst), sum(seen_l[x] for x in lst),
                                    _total(outline_l[x] for x in lst), _total(length_l[x] for x in lst)))
        num[7] = len(segments)
        yield 0.22
        tree.topo, tree.points, tree.cost, tree.seen = t, points, cost, seen
        tree.outline, tree.length_px = outline_l, length_l
        tree.irregular, tree.turning, tree.sep, tree.fixed, tree.cut_base = irregular, turning, sep, fixed, cut_base
        tree.patch, tree.n_patches, tree.patch_chi, tree.segments, tree.build_stretch = patch, n_patches, patch_chi, segments, self.build_stretch
        self.tree = tree
        if self.merge:
            yield from self._merge_tree(tree, members, tube_uv)
        num[13] = (time.perf_counter() - started) * 1e3
        num[14] = float(self._stop)

    def _face_stretch(self, tree, local: _Local, uv) -> list:
        if uv is None or not np.isfinite(uv).all():
            return [float("inf")] * len(local.faces)
        return [float(x) for x in uvflat.stretch(local.points, local.indptr, local.flat, uv)[1]]

    def _flatten(self, local: _Local, start):
        """ARAP (rows off) driven in its own slices; None when the chart cannot be laid out."""

        stage = uvflat.Flatten(local.points, local.indptr, local.flat, start, rows="off", budget=self.budget, use_fast=False)
        try:
            while not stage.done:
                stage.step()
                yield 0.5
        except ValueError:
            return None
        return stage.uv

    def _patch_layout(self, tree, local: _Local):
        g = uvflat.grid_layout(local.points, local.indptr, local.flat)
        q_grid = None if g is None else _quantile(self._face_stretch(tree, local, g[0]))
        if g is not None and q_grid <= 1.0 + GRID_SLACK:
            return True, g[0]
        arap = yield from self._flatten(local, None if g is None else g[0])
        if g is not None and arap is not None:
            q_arap = _quantile(self._face_stretch(tree, local, arap))
            return (True, g[0]) if q_grid <= q_arap * (1.0 + GRID_SLACK) else (False, arap)
        if g is not None:
            return True, g[0]
        return False, (arap if arap is not None else np.full((local.n, 2), np.nan))

    def _merge_tree(self, tree, members, tube_uv=None):
        t = tree.topo
        np_ = tree.n_patches
        dsu = _Undo(len(t.flat))
        for e in range(t.ne):
            if not tree.cut_base[e]:
                _glue(t, dsu, [e])
        dsu.commit()
        corner_uv = np.full((len(t.flat), 2), np.nan)
        face_str = [float("inf")] * t.nf
        owner = list(tree.patch)
        cpar = list(range(np_))
        faces = [list(m) for m in members]
        segs: list = [[] for _ in range(np_)]
        stretch, grid = [float("inf")] * np_, [False] * np_
        chi, ver = list(tree.patch_chi), [0] * np_
        glued_seg = [False] * len(tree.segments)
        ppu = float(tree._numbers[19])
        ppu2 = ppu * ppu
        pts_l = tree.points.tolist()
        area, weight, length = [0.0] * np_, [0.0] * np_, [0.0] * np_
        for p in range(np_):
            for f in members[p]:
                area[p] += _face_area(pts_l, t, f) * ppu2
                for h in t.face_corners(f):
                    e = t.cedge[h]
                    if tree.cut_base[e]:
                        weight[p] += tree.outline[e]
                        length[p] += tree.length_px[e]
        hull: list = [[] for _ in range(np_)]
        for p in range(np_):
            local = _Local(t, tree.points, members[p], dsu.find)
            if tube_uv and len(local.corner) and int(local.corner[0]) in tube_uv:
                g, uv = False, np.full((local.n, 2), np.nan)
                uv[local.flat] = np.array([tube_uv[int(h)] for h in local.corner])
            else:
                g, uv = yield from self._patch_layout(tree, local)
            local.scatter(uv, corner_uv)
            hull[p] = _hull_of(local.corner.tolist(), corner_uv[local.corner])
            per = self._face_stretch(tree, local, uv)
            for f, x in zip(local.faces, per):
                face_str[f] = x
            stretch[p], grid[p] = _quantile(per), g
            yield 0.3
        tree._numbers[15] = sum(grid)
        for s, sg in enumerate(tree.segments):
            segs[sg.p].append(s)
            if sg.q != sg.p:
                segs[sg.q].append(s)

        def find(p):
            r = p
            while cpar[r] != r:
                r = cpar[r]
            while cpar[p] != r:
                cpar[p], p = r, cpar[p]
            return r

        def ordered(lst):
            lst = sorted(set(lst))
            return sorted(lst, key=lambda s: (-tree.segments[s].cost, s))

        def neighbours(r):
            by: dict = {}
            for s in segs[r]:
                if glued_seg[s]:
                    continue
                sg = tree.segments[s]
                rp, rq = find(sg.p), find(sg.q)
                o = rq if rp == r else rp
                if o != r:
                    by.setdefault(o, []).append(s)
            return sorted((o, ordered(lst)) for o, lst in by.items())

        def between(a, b):
            x, y = (a, b) if len(segs[a]) <= len(segs[b]) else (b, a)
            out = []
            for s in segs[x]:
                if glued_seg[s]:
                    continue
                sg = tree.segments[s]
                rp, rq = find(sg.p), find(sg.q)
                if (rp, rq) in ((x, y), (y, x)):
                    out.append(s)
            return ordered(out)

        def estimate(lst):
            diff = total = c = 0.0
            for s in lst:
                sg = tree.segments[s]
                c += sg.cost
                for e in sg.edges:
                    ln = []
                    for h in t.euse[e]:
                        h0, h1 = t.at(h, t.ea[e]), t.at(h, t.eb[e])
                        ln.append(float(np.sqrt((corner_uv[h0, 0] - corner_uv[h1, 0]) ** 2 + (corner_uv[h0, 1] - corner_uv[h1, 1]) ** 2)))
                    diff += abs(ln[0] - ln[1])
                    total += max(ln[0], ln[1])
            return (diff / total if total > 0 and np.isfinite(diff) else float("inf")), c

        def shape(a, b, lst):
            """Irregularity of the union by outline weight; by plain length, and the larger of its parts'."""

            ar = area[a] + area[b]
            w = _total(tree.segments[s].weight for s in lst)
            ln = _total(tree.segments[s].length for s in lst)
            by_weight = _irregularity(max(weight[a] + weight[b] - 2.0 * w, 0.0), ar)
            plain = _irregularity(max(length[a] + length[b] - 2.0 * ln, 0.0), ar)
            return by_weight, plain, max(_irregularity(length[a], area[a]), _irregularity(length[b], area[b]))

        def saves_page(a, b, ev):
            """A chart whose gap ring outweighs its box: does the union's box with its ring take less page than the
            parts' (the kernel's `saves_page`)?"""

            pts = np.asarray(ev["uv"], float)
            wu, hu = _min_box(pts)
            wu, hu = wu * ppu, hu * ppu
            if wu * hu > 2.0 * GAP_PX * (wu + hu):
                return False

            def page(x):
                w, h = _min_box(x)
                return (w * ppu + GAP_PX) * (h * ppu + GAP_PX)

            return page(pts) <= page(corner_uv[hull[a]]) + page(corner_uv[hull[b]])

        heap: list = []
        cur = [heap]  # the queue being run (a region's, R10big, or the whole one)
        seq = [0]
        region = _regions(t, tree.points.tolist(), members)
        regional = any(r != REGION_NONE for r in region)
        defer = [None]  # the region being run: a pair across its border is noted in `deferred`
        deferred: list = []
        refused: dict = {}  # pairs whose union lay over itself: (smaller chart, its version then)
        last_st: dict = {}  # last measured stretch per pair
        parked: list = []  # candidates measured above the round's cap, waiting for the round they fit
        caps = round_caps(tree.build_stretch)
        cap = [caps[0]]

        def push_list(r, o, lst):
            a, b = min(r, o), max(r, o)
            ref = refused.get((a, b))
            if ref is not None and ver[ref[0]] == ref[1]:
                return
            mis, c = estimate(lst)
            key = _key(mis, c, shape(a, b, lst)[0])
            seq[0] += 1
            st = last_st.get((a, b))
            if st is not None and st > 1.0 + cap[0]:
                parked.append(((key, a, b, seq[0], ver[a], ver[b], True), st))
            else:
                heapq.heappush(cur[0], (key, a, b, seq[0], ver[a], ver[b], False))

        def push_for(r):
            if chi[r] != 1:
                return
            for o, lst in neighbours(r):
                if chi[o] != 1:
                    continue
                if defer[0] is not None and (region[o] != defer[0] or crosses(r) or crosses(o)):
                    # across a region border, or a chart on one: queued after the regions' part
                    deferred.append((min(r, o), max(r, o)))
                    continue
                push_list(r, o, lst)

        def crosses(r):
            """Chart r (a root) touches a disk chart of another region (the kernel's `crosses_region`)."""

            return any(chi[o] == 1 and region[o] != region[r] for o, _ in neighbours(r))

        def push_pair(a, b):
            if chi[a] != 1 or chi[b] != 1:
                return
            for o, lst in neighbours(a):
                if o == b:
                    push_list(a, b, lst)
                    break

        def evaluate(a, b, lst):
            mark = dsu.mark()
            glued, c = _glue_disk(t, dsu, tree.segments, lst, 2)
            if not glued:
                dsu.undo(mark)
                return None
            yield 0.4
            union = faces[a] + faces[b]
            local = _Local(t, tree.points, union, dsu.find)
            if grid[a] and grid[b]:
                g = uvflat.grid_layout(local.points, local.indptr, local.flat)
                if g is not None:
                    per = self._face_stretch(tree, local, g[0])
                    q = _quantile(per)
                    if q <= 1.0 + GRID_SLACK or q <= max(stretch[a], stretch[b]) + GRID_SLACK:
                        dsu.undo(mark)
                        return dict(glued=glued, cost=c, stretch=q, pole=False, local=local, uv=g[0], per=per, before=corner_uv[local.corner])
            # Pole union (subset): B fitted onto A along the glue, then the whole union flattened.
            pairs = []
            for s in glued:
                for e in tree.segments[s].edges:
                    h0, h1 = t.euse[e]
                    ha, hb = (h0, h1) if owner[t.cface[h0]] == a else (h1, h0)
                    for v in (t.ea[e], t.eb[e]):
                        pairs.append((corner_uv[t.at(hb, v)], corner_uv[t.at(ha, v)]))
            rigid = _align(pairs)
            moved = corner_uv.copy()
            hb_corners = [h for f in faces[b] for h in t.face_corners(f)]
            moved[hb_corners] = _rigid(rigid, corner_uv[hb_corners])
            start = local.gather(moved)
            uv = yield from self._flatten(local, start)
            per = self._face_stretch(tree, local, uv)
            dsu.undo(mark)
            return dict(glued=glued, cost=c, stretch=_quantile(per), pole=True, local=local, uv=uv, per=per, before=moved[local.corner])

        for p in range(np_):
            push_for(p)
            yield 0.35
        exact_evals = dropped = grid_merges = pole_merges = lay_over = 0
        top = 1.0
        def apply_ev(a, b, ev):
            """Glue an evaluated union for good and take its layout; returns the surviving chart."""

            nonlocal top, grid_merges, pole_merges
            ar = area[a] + area[b]
            wt = max(weight[a] + weight[b] - 2.0 * _total(tree.segments[s].weight for s in ev["glued"]), 0.0)
            ln = max(length[a] + length[b] - 2.0 * _total(tree.segments[s].length for s in ev["glued"]), 0.0)
            for s in ev["glued"]:
                _glue(t, dsu, tree.segments[s].edges)
                glued_seg[s] = True
            dsu.commit()
            ev["local"].scatter(ev["uv"], corner_uv)
            for f, x in zip(ev["local"].faces, ev["per"]):
                face_str[f] = x
            ids_ = hull[a] + hull[b] + ev["local"].corner.tolist()
            new_hull = _hull_of(ids_, corner_uv[ids_])
            r, o = (a, b) if len(faces[a]) >= len(faces[b]) else (b, a)
            hull[r], hull[o] = new_hull, []
            cpar[o] = r
            for f in faces[o]:
                owner[f] = r
            faces[r].extend(faces[o])
            faces[o] = []
            segs[r] = [s for s in segs[r] + segs[o] if not glued_seg[s]]
            segs[o] = []
            stretch[r], grid[r], chi[r] = ev["stretch"], not ev["pole"], 1
            area[r], weight[r], length[r] = ar, wt, ln
            ver[a] += 1
            ver[b] += 1
            top = max(top, ev["stretch"])
            if ev["pole"]:
                pole_merges += 1
            else:
                grid_merges += 1
            tree.merges.append((ev["glued"], ev["stretch"], ev["cost"], ev["pole"]))
            return r

        cached = None

        def run_heap(heap):
            """One merge round over the queue `heap` (the kernel's `Build::merge_round`)."""

            nonlocal cached, exact_evals, dropped, lay_over
            cur[0] = heap
            while heap:
                if self._stop:
                    break
                key, a, b, _, va, vb, exact = heapq.heappop(heap)
                if find(a) != a or find(b) != b or ver[a] != va or ver[b] != vb:
                    continue
                if cached is not None and cached[0] == (a, b, va, vb):
                    ev, cached = cached[1], None
                else:
                    exact_evals += 1
                    ev = yield from evaluate(a, b, between(a, b))
                    yield 0.5
                    if ev is None:
                        continue
                added = ev["stretch"] - max(stretch[a], stretch[b])
                irr, plain, parts = shape(a, b, ev["glued"])
                if not exact:
                    k2 = _key(added, ev["cost"], irr)
                    seq[0] += 1
                    heapq.heappush(heap, (k2, a, b, seq[0], va, vb, True))
                    cached = ((a, b, va, vb), ev)
                    continue
                if not ev["stretch"] <= 1.0 + tree.build_stretch:
                    dropped += 1
                    continue
                last_st[(a, b)] = ev["stretch"]
                if ev["stretch"] > 1.0 + cap[0]:
                    cached = ((a, b, va, vb), ev)
                    parked.append(((key, a, b, _, va, vb, True), ev["stretch"]))
                    continue
                if plain > SHAPE_MAX and plain > parts and not saves_page(a, b, ev):
                    # Too irregular: not tried again until the smaller chart changes (as a union that lies over).
                    small = a if len(faces[a]) <= len(faces[b]) else b
                    refused[(a, b)] = (small, ver[small])
                    continue
                if _lies_over(ev["local"], ev["uv"], ev["before"]):
                    lay_over += 1
                    # Not tried again until the smaller chart changes (as the kernel).
                    small = a if len(faces[a]) <= len(faces[b]) else b
                    refused[(a, b)] = (small, ver[small])
                    continue
                push_for(apply_ev(a, b, ev))

        for ri, cap_now in enumerate(caps):
            cap[0] = cap_now
            if ri > 0:
                waiting, parked[:] = parked[:], []
                for c_, st in waiting:
                    if find(c_[1]) != c_[1] or find(c_[2]) != c_[2] or ver[c_[1]] != c_[4] or ver[c_[2]] != c_[5]:
                        continue
                    if st <= 1.0 + cap_now:
                        heapq.heappush(heap, c_)
                    else:
                        parked.append((c_, st))
            if regional:
                # R10big: the round region by region (the kernel runs them apart), then across the region borders.
                by: dict = {}
                for c_ in heap:
                    ka, kb = region[c_[1]], region[c_[2]]
                    if ka != kb or crosses(c_[1]) or crosses(c_[2]):
                        deferred.append((c_[1], c_[2]))
                    else:
                        by.setdefault(ka, []).append(c_)
                heap.clear()
                for k in sorted(by):
                    hp = by[k]
                    heapq.heapify(hp)
                    defer[0] = k
                    yield from run_heap(hp)
                defer[0] = None
                pairs = sorted({(min(x, y), max(x, y)) for x, y in ((find(x), find(y)) for x, y in deferred) if x != y})
                deferred.clear()
                cur[0] = heap
                for a, b in pairs:
                    push_pair(a, b)
            yield from run_heap(heap)
        # R6: a last round in which every chart, smallest first, tries to join a neighbour, glued along all it shares or
        # else hinged on the costliest shared segment; small compact charts join free of the shape rule (kernel: the
        # last round in `merge_tree`). Passes repeat until one joins nothing.
        tried: dict = {}
        while not self._stop:
            order = sorted((p for p in range(np_) if cpar[p] == p and chi[p] == 1), key=lambda p: (len(faces[p]), p))
            joined = 0
            for p in order:
                if self._stop:
                    break
                if find(p) != p:
                    continue
                nb = [(o, _total(tree.segments[s].length for s in lst)) for o, lst in neighbours(p) if chi[o] == 1]
                nb.sort(key=lambda x: (-x[1], x[0]))
                for o, _ in nb:
                    vers = (ver[p], ver[o])
                    if tried.get((p, o)) == vers:
                        continue
                    tried[(p, o)] = vers
                    lst = between(o, p)
                    take = None
                    for hinge in (False, True):
                        exact_evals += 1
                        ev = yield from evaluate(o, p, lst[:1] if hinge else lst)
                        if ev is None:
                            break
                        free = len(faces[p]) <= FREE_FACES and _irregularity(length[p], area[p]) <= SHAPE_MAX
                        shaped = free
                        if not shaped:
                            _, plain, parts = shape(o, p, ev["glued"])
                            shaped = not (plain > SHAPE_MAX and plain > parts) or saves_page(o, p, ev)
                        if shaped and ev["stretch"] <= 1.0 + tree.build_stretch and not _lies_over(ev["local"], ev["uv"], ev["before"]):
                            take = ev
                            break
                        if len(ev["glued"]) <= 1:
                            break
                    if take is not None:
                        apply_ev(o, p, take)
                        joined += 1
                        break
            if not joined:
                break
        n = tree._numbers
        n[8], n[9], n[10], n[11], n[12], n[16], n[17] = len(tree.merges), grid_merges, pole_merges, 0, exact_evals, dropped, top
        n[20] = lay_over
        tree.start_uv = corner_uv
        # R7hide's slides of the borders out of sight are made later, in slices (`uvhide.moves`, from New UVs), or on the
        # first level asked for with `hide` (R8late)


def build(points, indptr, flat, **options) -> ChartTree:
    """Run `Charts` to the end (tests and scripts)."""

    stage = Charts(points, indptr, flat, **options)
    while not stage.done:
        stage.step()
    return stage.tree


def chart_meshes(tree, cut, chart):
    """Per chart of a cut state: (chart id, its faces, points (n, 3), indptr, flat, global corner of each local corner),
    with vertices split along every cut inside the chart. The meshes `uvflat.Flatten` lays out."""

    t = tree.topo
    wr = _wedges(t, cut)
    members: dict = {}
    for f, c in enumerate(np.asarray(chart).tolist()):
        members.setdefault(c, []).append(f)
    for c in sorted(members):
        local = _Local(t, tree.points, members[c], lambda h: wr[h])
        yield c, members[c], local.points, local.indptr, local.flat, local.corner


def _annulus_row(t: Topo, cut, patch, p: int, members, cost):
    loops = _patch_loops(t, cut, members)
    ids = sorted(set(loops.values()))
    if len(ids) != 2:
        return None

    def inside(e):
        f0, f1 = t.faces_of(e)
        return not cut[e] and f1 >= 0 and patch[f0] == p and patch[f1] == p

    best = None
    for v in sorted(v for v, loop in loops.items() if loop == ids[0]):
        inward = [e for e in t.vedge[v] if inside(e)]
        if len(inward) != 1:
            continue
        cur, e = v, inward[0]
        path, c = [], 0.0
        while True:
            path.append(e)
            c += cost[e]
            w = t.other(e, cur)
            if w in loops:
                if loops[w] == ids[1] and (best is None or c < best[0]):
                    best = (c, list(path))
                break
            x = t.straight_next(w, e)
            if x is not None and inside(x) and len(path) <= len(members):
                cur, e = w, x
            else:
                break
    return None if best is None else best[1]


def _patch_loops(t: Topo, cut, members) -> dict:
    ids: dict = {}
    pairs = []
    for f in members:
        for h in t.face_corners(f):
            if cut[t.cedge[h]]:
                a, b = t.flat[h], t.flat[t.next(h)]
                for v in (a, b):
                    ids.setdefault(v, len(ids))
                pairs.append((ids[a], ids[b]))
    d = _Dsu(len(ids))
    for a, b in pairs:
        d.union(a, b)
    return {v: d.find(k) for v, k in ids.items()}


def _tree_cotree(t: Topo, cut, patch, p: int, members, cost) -> list:
    inner = set()
    for f in members:
        for h in t.face_corners(f):
            e = t.cedge[h]
            f0, f1 = t.faces_of(e)
            if not cut[e] and not t.wall[e] and f0 == f and patch[f1] == p:
                inner.add(e)
    inner = sorted(sorted(inner), key=lambda e: (-cost[e], e))
    local = {f: k for k, f in enumerate(members)}
    d = _Dsu(len(members))
    cutset = []
    for e in inner:
        f0, f1 = t.faces_of(e)
        a, b = local[f0], local[f1]
        if d.find(a) == d.find(b):
            cutset.append(e)
        else:
            d.union(a, b)
    on = {e: True for e in cutset}
    deg: dict = {}
    for f in members:
        for h in t.face_corners(f):
            if cut[t.cedge[h]]:
                deg[t.flat[h]] = deg.get(t.flat[h], 0) + 2
    for e in cutset:
        for v in (t.ea[e], t.eb[e]):
            deg[v] = deg.get(v, 0) + 1
    queue = [v for v, k in deg.items() if k == 1]
    while queue:
        v = queue.pop()
        if deg[v] != 1:
            continue
        e = next((e for e in t.vedge[v] if on.get(e)), None)
        if e is None:
            continue
        on[e] = False
        for w in (t.ea[e], t.eb[e]):
            deg[w] -= 1
            if deg[w] == 1:
                queue.append(w)
    return sorted(e for e, x in on.items() if x)


# ---------------------------------------------------------------------------------------------------------------------
# Thin tubes (R5) and chart boxes on the page
# ---------------------------------------------------------------------------------------------------------------------


def _dijkstra(t: Topo, sources, weight, inside) -> dict:
    """Shortest paths over the edges with `inside`, from the given sources: vertex -> (distance, edge taken). Ties pop
    the larger vertex first, as the kernel's heap."""

    dist = {s: (0.0, -1) for s in sources}
    heap = [(0.0, -s) for s in sources]
    heapq.heapify(heap)
    while heap:
        d, nv = heapq.heappop(heap)
        v = -nv
        if d > dist[v][0]:
            continue
        for e in t.vedge[v]:
            if not inside(e):
                continue
            w = t.other(e, v)
            nd = d + max(weight(e), 0.0)
            old = dist.get(w)
            if old is None or nd < old[0]:
                dist[w] = (nd, e)
                heapq.heappush(heap, (nd, -w))
    return dist


def _farthest(dist: dict):
    """(vertex, distance) of the largest distance, the smaller vertex on a tie; None when empty."""

    if not dist:
        return None
    v = min(dist, key=lambda x: (-dist[x][0], x))
    return v, dist[v][0]


def _path_back(t: Topo, dist: dict, v: int) -> tuple:
    path = []
    while True:
        e = dist[v][1]
        if e < 0:
            break
        path.append(e)
        v = t.other(e, v)
    return path, v


def _weighted_quantile(values, w, q: float) -> float:
    order = np.argsort(np.asarray(values, float), kind="stable")
    acc, total = 0.0, float(np.sum(w))
    for i in order:
        acc += w[i]
        if acc >= q * total:
            return float(values[i])
    return float(values[order[-1]]) if len(order) else float("nan")


def _l2_stretch(local: _Local, uv: np.ndarray) -> tuple:
    """The UV Check's stretch of one chart (SPEC 9.1, as the kernel's `l2_stretch`): per fan triangle, the root mean square
    of the singular values of the map from the triangle to the UV, the UVs scaled to the surface area; returns the
    median weighted by surface area and the surface area above `TUBE_L2_BAD`."""

    tris = []
    for f in range(len(local.indptr) - 1):
        lo, hi = int(local.indptr[f]), int(local.indptr[f + 1])
        for i in range(lo + 1, hi - 1):
            tris.append((local.flat[lo], local.flat[i], local.flat[i + 1]))
    if not tris:
        return float("inf"), 0.0
    tri = np.array(tris, np.int64)
    x0, x1, x2 = local.points[tri[:, 0]], local.points[tri[:, 1]], local.points[tri[:, 2]]
    e1, e2 = x1 - x0, x2 - x0
    l1 = np.linalg.norm(e1, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    ok = (l1 > 0) & (area > 0)
    tri, e1, e2, l1, area = tri[ok], e1[ok], e2[ok], l1[ok], area[ok]
    bx = (e1 * e2).sum(1) / l1
    by = 2.0 * area / l1
    u = np.asarray(uv, float)
    d1, d2 = u[tri[:, 1]] - u[tri[:, 0]], u[tri[:, 2]] - u[tri[:, 0]]
    a3, a2 = float(area.sum()), float((0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])).sum())
    if not (a3 > 0 and a2 > 0):
        return float("inf"), 0.0
    s = math.sqrt(a3 / a2)
    det = l1 * by  # e1 = (l1, 0), e2 = (bx, by)
    inv = np.stack([np.stack([by / det, -bx / det], 1), np.stack([np.zeros_like(det), l1 / det], 1)], 1)
    j00 = d1[:, 0] * inv[:, 0, 0] + d2[:, 0] * inv[:, 1, 0]
    j01 = d1[:, 0] * inv[:, 0, 1] + d2[:, 0] * inv[:, 1, 1]
    j10 = d1[:, 1] * inv[:, 0, 0] + d2[:, 1] * inv[:, 1, 0]
    j11 = d1[:, 1] * inv[:, 0, 1] + d2[:, 1] * inv[:, 1, 1]
    vals = s * np.sqrt((j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11) / 2.0)
    bad = float(area[~(vals <= TUBE_L2_BAD)].sum())
    return _weighted_quantile(vals, area, 0.5), bad


def _strip_start(t: Topo, local: _Local, ends, area: float):
    """Start layout of a strip chart (as the kernel's `strip_start`): Tutte's embedding (Tutte 1963, Floater 1997) with
    the border on a convex outline as long as the cut and as wide as the loops, a point (with a bevel) at an end that is
    a point. None when the border does not fit that shape."""

    n = local.n
    loop = uvflat._border_loop(local.indptr, local.flat, n)
    m = len(loop)
    if m < 3:
        return None
    glob = np.empty(n, np.int64)
    glob[local.flat] = np.array([t.flat[h] for h in local.corner], np.int64)
    p = local.points[np.array(loop)]
    seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    at = np.concatenate([[0.0], np.cumsum(seg)]).tolist()
    total = at[m]
    if not total > 0:
        return None
    oa = [i for i in range(m) if glob[loop[i]] == ends[0]]
    ob = [i for i in range(m) if glob[loop[i]] == ends[1]]
    if not oa or not ob or len(oa) > 2 or len(ob) > 2 or ends[0] == ends[1]:
        return None

    def fwd(i, j):
        return (at[j] - at[i]) % total

    def pick(o, other):
        if len(o) == 1:
            return o[0], o[0]
        inside = any(fwd(o[0], k) < fwd(o[0], o[1]) for k in other)
        return (o[0], o[1]) if not inside else (o[1], o[0])

    a1, a2 = pick(oa, ob)
    b1, b2 = pick(ob, oa)
    ra, rb = fwd(a1, a2), fwd(b1, b2)
    bottom, top = fwd(a2, b1), fwd(b2, a1)
    long_ = 0.5 * (bottom + top)
    if len(oa) == 2 and len(ob) == 2:
        wide = 0.5 * (ra + rb)
    elif len(oa) == 2:
        wide = ra
    elif len(ob) == 2:
        wide = rb
    else:
        wide = area / max(long_, 1e-300)
    if not (long_ > 0 and wide > 0):
        return None
    bevel = min(0.5 * wide, 0.25 * long_)
    ha = 0.0 if len(oa) == 2 else bevel
    hb = 0.0 if len(ob) == 2 else bevel
    left = [(0.0, wide), (0.0, 0.0)] if len(oa) == 2 else [(0.0, 0.5 * wide)]
    right = [(long_, 0.0), (long_, wide)] if len(ob) == 2 else [(long_, 0.5 * wide)]
    lower = [left[-1]] + ([(ha, 0.0)] if ha > 0 else []) + ([(long_ - hb, 0.0)] if hb > 0 else []) + [right[0]]
    upper = [right[-1]] + ([(long_ - hb, wide)] if hb > 0 else []) + ([(ha, wide)] if ha > 0 else []) + [left[0]]

    def along(line, x):
        lens = [math.hypot(line[k + 1][0] - line[k][0], line[k + 1][1] - line[k][1]) for k in range(len(line) - 1)]
        tot = sum(lens)
        if len(line) == 1 or not tot > 0:
            return line[0]
        want = min(max(x, 0.0), 1.0) * tot
        for k, ln in enumerate(lens):
            if want <= ln or k + 1 == len(lens):
                r = min(want / ln, 1.0) if ln > 0 else 0.0
                return (line[k][0] + r * (line[k + 1][0] - line[k][0]), line[k][1] + r * (line[k + 1][1] - line[k][1]))
            want -= ln
        return line[-1]

    runs = [(a1, ra, left), (a2, bottom, lower), (b1, rb, right), (b2, top, upper)]
    uv = np.zeros((n, 2))
    for i in range(m):
        cands = [r for r in runs if r[1] > 0 or r[0] == i]
        start, ln, line = min(cands, key=lambda r: fwd(r[0], i))  # the first of equal ones, as the kernel
        uv[loop[i]] = along(line, fwd(start, i) / ln if ln > 0 else 0.0)
    scale = max(long_, wide)
    a, b = [], []
    for poly in uvflat._faces(local.indptr, local.flat):
        k = len(poly)
        a.extend(int(poly[i]) for i in range(k))
        b.extend(int(poly[(i + 1) % k]) for i in range(k))
    a, b = np.array(a), np.array(b)
    half = np.full(len(a), 0.5)
    system = uvflat._System(n, np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]), np.concatenate([half, half, -half, -half]), loop)
    info: dict = {}
    for axis in range(2):
        x = uv[:, axis].copy()
        for _ in system.cg(system.rhs(np.zeros(n), uv[:, axis]), x, scale * 1e-9, info):
            pass
        uv[:, axis] = x
    return uv if np.isfinite(uv).all() else None


def _tube_cuts(t: Topo, points: np.ndarray, cost, forced_line, flatten):
    """Thin tubes (R5, as the kernel's `tube_cuts`): parts whose surface is a long strip laid out as strip pieces.
    `flatten(local, start)` is a generator returning the ARAP layout. Generator; returns a list of (faces, cut edges,
    {corner: uv})."""

    ne, nf = t.ne, t.nf
    d = _Dsu(nf)
    for e in range(ne):
        if not t.wall[e] and not forced_line[e]:
            f0, f1 = t.faces_of(e)
            d.union(f0, f1)
    ids: dict = {}
    comp = [ids.setdefault(d.find(f), len(ids)) for f in range(nf)]
    nc = len(ids)
    members: list = [[] for _ in range(nc)]
    bad = [False] * nc
    for f in range(nf):
        members[comp[f]].append(f)
        for h in t.face_corners(f):
            bad[comp[f]] |= forced_line[t.cedge[h]]

    part_chi = _chi_per(t, t.wall, comp, nc)  # a tube wraps round: a part that is already a disk is left to the merge tree
    pts_l = points.tolist()

    def len3(e):  # summed in the kernel's order
        p, q = pts_l[t.ea[e]], pts_l[t.eb[e]]
        return math.sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]))

    cands = []
    for c in range(nc):
        if bad[c] or part_chi[c] == 1 or len(members[c]) < 4:
            continue
        faces = members[c]

        def in_part(e, c=c):
            f0, f1 = t.faces_of(e)
            return comp[f0] == c and (f1 < 0 or comp[f1] == c)

        area = _total(_face_area(pts_l, t, f) for f in faces)
        if not area > 0:
            continue
        v0 = t.flat[t.indptr[faces[0]]]
        u = _farthest(_dijkstra(t, [v0], len3, in_part))[0]
        su = _dijkstra(t, [u], len3, in_part)
        diameter = _farthest(su)[1]
        if not diameter > 0:
            continue
        sf = {f: _total(su[t.flat[h]][0] if t.flat[h] in su else 0.0 for h in t.face_corners(f)) / len(t.face_corners(f)) for f in faces}

        def pieces_of(band, faces=faces):
            loc = {f: k for k, f in enumerate(faces)}
            dd = _Dsu(len(faces))
            for f in faces:
                for h in t.face_corners(f):
                    e = t.cedge[h]
                    if t.wall[e]:
                        continue
                    f0, f1 = t.faces_of(e)
                    if band(f0) == band(f1):
                        dd.union(loc[f0], loc[f1])
            return {f: dd.find(k) for f, k in loc.items()}

        def mid(f, sf=sf, diameter=diameter):
            return 1 if 0.4 * diameter <= sf[f] <= 0.6 * diameter else 0

        mp = pieces_of(mid)
        halves = {mp[f] for f in faces if mid(f) == 1}
        closed = not any(t.wall[t.cedge[h]] for f in faces for h in t.face_corners(f))
        length = 2.0 * diameter if closed and len(halves) >= 2 else diameter
        if not length * length / area >= TUBE_ASPECT:
            continue
        width = PIECE_ASPECT * (area / length)
        bins = max(int(math.floor(diameter / width + 0.5)), 1)

        def bin_(f, sf=sf, diameter=diameter, bins=bins):
            return min(max(int(math.floor(sf[f] / diameter * bins)), 0), bins - 1)

        piece = pieces_of(bin_)
        groups: dict = {}
        for f in faces:
            groups.setdefault(piece[f], []).append(f)
        rings = []
        for f in faces:
            for h in t.face_corners(f):
                e = t.cedge[h]
                if not t.wall[e]:
                    f0, f1 = t.faces_of(e)
                    if f0 == f and piece[f0] != piece[f1]:
                        rings.append(e)
        is_ring = set(rings)
        out, ok = [], True
        for k in sorted(groups):
            pf = groups[k]

            def inside(e, k=k, piece=piece):
                f0, f1 = t.faces_of(e)
                return piece.get(f0) == k and (f1 < 0 or piece.get(f1) == k)

            border = [t.cedge[h] for f in pf for h in t.face_corners(f) if t.wall[t.cedge[h]] or t.cedge[h] in is_ring]
            ld = _Dsu(t.nv)
            for e in border:
                ld.union(t.ea[e], t.eb[e])
            root = {v: ld.find(v) for e in border for v in (t.ea[e], t.eb[e])}
            roots = sorted(set(root.values()))

            def on(r, root=root):
                return sorted(v for v, x in root.items() if x == r)

            def inner(e, inside=inside, is_ring=is_ring):
                return inside(e) and not t.wall[e] and e not in is_ring

            if len(roots) == 0:
                lp = _dijkstra(t, [u], len3, inside)
                frm, to = [u], [_farthest(lp)[0]]
            elif len(roots) == 1:
                rim = on(roots[0])
                far = _farthest(_dijkstra(t, rim, len3, inner))
                if far is None:
                    ok = False
                    break
                frm, to = [far[0]], rim
            elif len(roots) == 2:
                frm, to = on(roots[0]), on(roots[1])
            else:
                ok = False
                break
            dist = _dijkstra(t, frm, lambda e: cost[e], inner)
            reach = [(dist[v][0], v) for v in to if v in dist]
            if not reach:
                ok = False
                break
            end = min(reach)[1]
            path, start = _path_back(t, dist, end)
            out.append((pf, path, (start, end)))
        if ok:
            cands.append((out, rings))
        yield
    if not cands:
        return []
    cut = list(t.wall)
    label = [-1] * nf
    np_ = 0
    for pieces, rings in cands:
        for e in rings:
            cut[e] = True
        for pf, path, _ in pieces:
            for e in path:
                cut[e] = True
            for f in pf:
                label[f] = np_
            np_ += 1
    lab = [x if x >= 0 else np_ for x in label]
    chi = _chi_per(t, cut, lab, np_ + 1)
    wr = _wedges(t, cut)
    laid = []
    for pieces, _ in cands:
        for pf, path, ends in pieces:
            got = None
            if chi[label[pf[0]]] == 1:
                local = _Local(t, points, pf, lambda h: wr[h])
                area = _total(_face_area(pts_l, t, f) for f in pf)
                start = _strip_start(t, local, ends, area)
                uv = None if start is None else (yield from flatten(local, start))
                if uv is not None and np.isfinite(uv).all():
                    med, badarea = _l2_stretch(local, uv)
                    if med <= TUBE_L2_MEDIAN and uvflat.count_flips(local.indptr, local.flat, uv) == 0:
                        cuv = {int(h): uv[local.flat[k]] for k, h in enumerate(local.corner)}
                        sides = [(cuv[h], cuv[t.next(h)], (wr[h], wr[t.next(h)])) for f in pf for h in t.face_corners(f) if cut[t.cedge[h]]]
                        p = np.array([s[0] for s in sides]).reshape(-1, 2)
                        q = np.array([s[1] for s in sides]).reshape(-1, 2)
                        v = np.array([s[2] for s in sides], np.int64).reshape(-1, 2)
                        if not _sides_cross(p, q, v):
                            got = (cuv, badarea, area)
            laid.append(got)
            yield
    total = _total(_face_area(pts_l, t, f) for f in range(nf))
    order = sorted((j for j in range(len(laid)) if laid[j] is not None), key=lambda j: (laid[j][1] / max(laid[j][2], 1e-300), j))
    keep = [False] * len(laid)
    spent = 0.0
    for j in order:
        if spent + laid[j][1] <= TUBE_BAD_SHARE * total:
            spent += laid[j][1]
            keep[j] = True
    tubes, j = [], 0
    for pieces, rings in cands:
        faces, cutl, cuv = [], [], {}
        kept = set()
        for pf, path, _ in pieces:
            if keep[j]:
                cuv.update(laid[j][0])
                kept.update(pf)
                faces.extend(pf)
                cutl.extend(path)
            j += 1
        if faces:
            cutl.extend(e for e in rings if any(f in kept for f in t.faces_of(e)))
            tubes.append((faces, cutl, cuv))
    return tubes


def _hull_of(ids, pts) -> list:
    """Ids on the convex hull of the points (Andrew's monotone chain), as the kernel's `hull_of`."""

    items = [(i, (float(p[0]), float(p[1]))) for i, p in zip(ids, pts) if np.isfinite(p[0]) and np.isfinite(p[1])]
    items.sort(key=lambda x: (x[1][0], x[1][1]))
    dedup = []
    for it in items:
        if not dedup or dedup[-1][1] != it[1]:
            dedup.append(it)
    if len(dedup) < 3:
        return [x[0] for x in dedup]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    out: list = []
    for seq in (dedup, dedup[::-1]):
        start = len(out)
        for q in seq:
            while len(out) >= start + 2 and cross(out[-2][1], out[-1][1], q[1]) <= 0.0:
                out.pop()
            out.append(q)
        out.pop()
    return [x[0] for x in out]


def _min_box(pts) -> tuple:
    """Sides (w, h) of the smallest-area box around the points, one side along a hull edge (Toussaint 1983)."""

    pts = np.asarray(pts, float).reshape(-1, 2)
    hull = pts[_hull_of(range(len(pts)), pts)] if len(pts) else pts
    if len(hull) < 3:
        if len(hull) == 0:
            return 0.0, 0.0
        span = hull.max(0) - hull.min(0)
        return float(span[0]), float(span[1])
    best = (float("inf"), 0.0, 0.0)
    for i in range(len(hull)):
        d = hull[(i + 1) % len(hull)] - hull[i]
        ln = math.hypot(d[0], d[1])
        if ln <= 0:
            continue
        ux, uy = d[0] / ln, d[1] / ln
        x = hull[:, 0] * ux + hull[:, 1] * uy
        y = -hull[:, 0] * uy + hull[:, 1] * ux
        w, h = float(x.max() - x.min()), float(y.max() - y.min())
        if w * h < best[0]:
            best = (w * h, w, h)
    return best[1], best[2]
