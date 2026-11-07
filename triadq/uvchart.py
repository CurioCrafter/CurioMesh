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

Subset: a pole union (no grid) is measured here by flattening the whole union (start: the two layouts fitted onto
each other) instead of the band-only solve, so on meshes with poles its merge order can differ; its numbers say
"estimate". Keep going's cut moves have no twin: they run through the faster callable only.

`Charts` is sliceable: `step()` does about 4 ms of work and returns progress. When the adapter installed a faster
`build` callable, the stage hands the mesh to it (a job that never blocks) and only polls it; `tree` then is the
faster tree, with the same `level(x)` and `numbers()`.
"""

from __future__ import annotations

import heapq
import time

import numpy as np

from . import uvflat

# Constants table: same names and values as the top of the matching .rs file (tests/test_uv_scaffold.py compares).
HIDDEN_L = 20.0
STRETCH_ALLOWED_PCT = 10.0
BUILD_STRETCH = 1.0
BAND_RINGS = 2
PX_PER_DIAGONAL = 1024.0
GRID_SLACK = 0.01
SAME_STRETCH = 1e-9
NUMBERS_LEN = 20

NUMBER_COLUMNS = ("patches", "irregular", "turning", "traces", "euler_cut_patches", "euler_cut_edges", "not_disk_patches", "segments",
                  "merges", "grid_merges", "pole_merges", "quick_calls", "exact_evals", "build_ms", "stopped", "grid_patches", "dropped",
                  "top_stretch", "forced_edges", "px_per_unit")
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
    d = _Dsu(len(t.flat))
    for e in range(t.ne):
        if not cut[e] and not t.wall[e]:
            h0, h1 = t.euse[e]
            for v in (t.ea[e], t.eb[e]):
                d.union(t.at(h0, v), t.at(h1, v))
    return [d.find(c) for c in range(len(t.flat))]


def _chi_per(t: Topo, cut, patch, np_: int) -> list:
    wr = _wedges(t, cut)
    chi, first = [0] * np_, set()
    for f in range(t.nf):
        p = patch[f]
        chi[p] += 1
        for h in t.face_corners(f):
            chi[p] -= 1
            if wr[h] not in first:
                first.add(wr[h])
                chi[p] += 1
    for e in range(t.ne):
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


def _key(added: float, cost: float) -> tuple:
    if not np.isfinite(added):
        return (2, 0.0, -cost)
    if added <= SAME_STRETCH:
        return (0, 0.0 if abs(added) <= SAME_STRETCH else added, -cost)
    return (1, -(cost / added), 0.0)


class Segment:
    __slots__ = ("edges", "p", "q", "cost", "seen")

    def __init__(self, edges, p, q, cost, seen):
        self.edges, self.p, self.q, self.cost, self.seen = edges, p, q, cost, seen


class ChartTree:
    """Patches, segments and recorded merges of one quad mesh; `level(x)` gives the charts at Stretch allowed x."""

    def __init__(self):
        self.merges: list = []  # (segment ids, stretch, cost, pole)
        self._numbers = np.zeros(NUMBERS_LEN)
        self._cache: dict = {}

    def numbers(self) -> np.ndarray:
        return self._numbers.copy()

    def patches(self) -> np.ndarray:
        return np.array(self.patch, np.int64)

    def level(self, x: float):
        """(chart per face, cut per edge, numbers in `LEVEL_COLUMNS`), cached per x."""

        x = float(x)
        if x not in self._cache:
            self._cache[x] = self._level(x)
        chart, cut, numbers = self._cache[x]
        return chart.copy(), cut.copy(), numbers.copy()

    def _level(self, x: float):
        started = time.perf_counter()
        t = self.topo
        thr = 1.0 + x / 100.0
        dsu = _Undo(len(t.flat))
        for e in range(t.ne):
            if not self.cut_base[e]:
                _glue(t, dsu, [e])
        dsu.commit()
        pd = _Dsu(self.n_patches)
        chi = list(self.patch_chi)
        glued = np.zeros(t.ne, bool)
        used = segs_glued = 0
        for segs, stretch, _, _ in self.merges:
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
        cut = np.asarray(self.cut_base, bool) & ~glued
        ids: dict = {}
        chart = np.array([ids.setdefault(pd.find(self.patch[f]), len(ids)) for f in range(t.nf)], np.int64)
        not_disk = sum(1 for r in ids if chi[r] != 1)
        numbers = np.array([len(ids), not_disk, used, segs_glued, float(x / 100.0 > self.build_stretch), (time.perf_counter() - started) * 1e3])
        return chart, cut, numbers


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
            segments.append(Segment(lst, pr[0], pr[1], sum(cost_l[x] for x in lst), sum(seen_l[x] for x in lst)))
        num[7] = len(segments)
        yield 0.22
        tree.topo, tree.points, tree.cost, tree.seen = t, points, cost, seen
        tree.irregular, tree.turning, tree.sep, tree.fixed, tree.cut_base = irregular, turning, sep, fixed, cut_base
        tree.patch, tree.n_patches, tree.patch_chi, tree.segments, tree.build_stretch = patch, n_patches, patch_chi, segments, self.build_stretch
        self.tree = tree
        if self.merge:
            yield from self._merge_tree(tree, members)
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

    def _merge_tree(self, tree, members):
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
        last: dict = {}
        for p in range(np_):
            local = _Local(t, tree.points, members[p], dsu.find)
            g, uv = yield from self._patch_layout(tree, local)
            local.scatter(uv, corner_uv)
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

        heap: list = []
        seq = [0]

        def push_for(r):
            if chi[r] != 1:
                return
            for o, lst in neighbours(r):
                if chi[o] != 1:
                    continue
                a, b = min(r, o), max(r, o)
                key = last.get((a, b))
                if key is None:
                    mis, c = estimate(lst)
                    key = _key(mis, c)
                seq[0] += 1
                heapq.heappush(heap, (key, a, b, seq[0], ver[a], ver[b], False))

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
                        return dict(glued=glued, cost=c, stretch=q, pole=False, local=local, uv=g[0], per=per)
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
            return dict(glued=glued, cost=c, stretch=_quantile(per), pole=True, local=local, uv=uv, per=per)

        for p in range(np_):
            push_for(p)
            yield 0.35
        exact_evals = dropped = grid_merges = pole_merges = 0
        top = 1.0
        cached = None
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
            if not exact:
                k2 = _key(added, ev["cost"])
                last[(a, b)] = k2
                seq[0] += 1
                heapq.heappush(heap, (k2, a, b, seq[0], va, vb, True))
                cached = ((a, b, va, vb), ev)
                continue
            if not ev["stretch"] <= 1.0 + tree.build_stretch:
                dropped += 1
                continue
            for s in ev["glued"]:
                _glue(t, dsu, tree.segments[s].edges)
                glued_seg[s] = True
            dsu.commit()
            ev["local"].scatter(ev["uv"], corner_uv)
            for f, x in zip(ev["local"].faces, ev["per"]):
                face_str[f] = x
            r, o = (a, b) if len(faces[a]) >= len(faces[b]) else (b, a)
            cpar[o] = r
            for f in faces[o]:
                owner[f] = r
            faces[r].extend(faces[o])
            faces[o] = []
            segs[r] = [s for s in segs[r] + segs[o] if not glued_seg[s]]
            segs[o] = []
            stretch[r], grid[r], chi[r] = ev["stretch"], not ev["pole"], 1
            ver[a] += 1
            ver[b] += 1
            top = max(top, ev["stretch"])
            if ev["pole"]:
                pole_merges += 1
            else:
                grid_merges += 1
            tree.merges.append((ev["glued"], ev["stretch"], ev["cost"], ev["pole"]))
            push_for(r)
        n = tree._numbers
        n[8], n[9], n[10], n[11], n[12], n[16], n[17] = len(tree.merges), grid_merges, pole_merges, 0, exact_evals, dropped, top
        tree.start_uv = corner_uv


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
    for v in sorted(v for v, l in loops.items() if l == ids[0]):
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
