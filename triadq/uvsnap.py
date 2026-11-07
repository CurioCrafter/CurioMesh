# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Seam snap and split-leftovers twin (package K1b, SPEC 4.5 and 4.7).

NumPy only, no bpy. The reference lives in the kernel sources; this twin follows it step for step (same order of
work, same ties, same float operations) so a parity test can compare the two within 1e-6. Long work runs as a
sliceable stage (`Snap.step()` / `Split.step()` do about 2 ms) or, when the adapter installed a faster callable, as a
background job the stage polls.

Our own design and code. Textbook pieces: Dijkstra's shortest paths (Dijkstra 1959) inside a corridor around the
chain; closest point on a segment and a segment crossing a plane (Ericson, "Real-Time Collision Detection", 2005,
5.1.2 and 5.3.1); Newell's polygon normal (Newell 1972). The corridor rule, the staircase test, the move guard, the
per-edge crossings with vertex reuse, the per-side UVs from the chain polylines, fan and successive cuts and the
damage-ordered face budget are ours. No strip or row of faces is ever added along an island border: only single faces
the remesh and the snap could not fit are cut, once per chain, along the source polyline.

Inputs: a result mesh (points, indptr, flat), the seam dict of `uvseams.extract` (chains, classes, gaps) and the welded
source positions `xyz` (`chain_xyz`). `snap` moves vertices onto Soft chains (never flipping a face, never past
`max_move`; refused vertices are held); `split` cuts the faces still crossed and hands back a new mesh, the seam vertex
tags and per-corner UVs on the seams (`with_seam_uvs` lays them over a per-side transfer).
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from .topology import _quad_points_valid
from .uvseams import _edges, closest_bary

# Constants table: same names and values as the top of the matching .rs file (tests/test_uv_scaffold.py compares).
SNAP_REACH = 1.0
SNAP_COST_K = 4.0
SNAP_MEAN_DIST = 0.35
SNAP_TURN_EVERY = 5.0
SNAP_TURN_COS = 0.7071067811865476
ON_EDGE_SHARE = 1e-6
RELAX_WEIGHT = 0.5
RELAX_RINGS = 1
SPLIT_GAP_PX = 1.0
CROSS_REACH = 0.5
REUSE_EDGE = 0.1
NEARER_SHARE = 0.5
WARP_COSINE = 0.5

SKIPPED, ACCEPTED, REFUSED, NO_PATH, ON_EDGES, SHORT, PARTLY_HELD = 0, 1, 2, 3, 4, 5, 6
KEPT, SPLIT, LEFT, CANNOT, BAKE = 0, 1, 2, 3, 4
SOFT = 2
CONTACT = 4  # a part contact (`part_contacts`): the snap fits it like a Soft chain, the split leaves it alone
SLICE_S = 0.002  # one step() of a stage does about this much work (short: a loaded PC can stretch a slice several times)

# Optional faster callables, set by the one adapter (uv_fast.py); the NumPy code runs when a name is missing.
FAST: dict = {}


def set_fast(**callables) -> None:
    """Install (or, with None, remove) faster callables by name."""

    for key, fn in callables.items():
        if fn is None:
            FAST.pop(key, None)
        else:
            FAST[key] = fn


# ---------------------------------------------------------------- inputs and outputs


def chain_xyz(seams: dict, points) -> np.ndarray:
    """Welded source positions (3 per welded vertex) for the chains' `run_vertex` ids."""

    points = np.asarray(points, np.float64).reshape(-1, 3)
    xyz = np.zeros((int(seams["n_welded"]), 3))
    xyz[np.asarray(seams["weld"], np.int64)] = points
    extra = seams.get("contact_xyz")  # welded ids after the source's: the part contacts' vertices (`part_contacts`)
    if extra is not None and len(extra):
        xyz[len(xyz) - len(extra):] = extra
    return xyz.ravel()


def vertex_edge(points, indptr, flat) -> np.ndarray:
    """Mean length of the edges at each vertex (the snap's local edge); the mean edge where a vertex has none."""

    return _Mesh(points, indptr, flat).vedge.copy()


def with_seam_uvs(uv, corner_uv) -> np.ndarray:
    """A transfer's per-corner UVs with the seam corners replaced by the split's chain-polyline UVs (NaN = keep)."""

    out = np.array(uv, np.float64).ravel()
    seam = np.asarray(corner_uv, np.float64).ravel()
    keep = ~np.isnan(seam)
    out[keep] = seam[keep]
    return out


def _parts(fl, ip, weld, nw):
    """Per source face: its part (faces joined across shared welded edges; union-find)."""

    nf = len(ip) - 1
    counts = np.diff(ip)
    face_of = np.repeat(np.arange(nf), counts)
    nxt = np.arange(len(fl)) + 1
    last = ip[1:] - 1
    nxt[last[counts > 0]] = ip[:-1][counts > 0]
    wa, wb = weld[fl], weld[fl[nxt]]
    key = np.minimum(wa, wb) * nw + np.maximum(wa, wb)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    lab = np.arange(nf)

    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]
            x = lab[x]
        return x

    fo = face_of[order]
    for i in np.flatnonzero(ks[1:] == ks[:-1]).tolist():
        x, y = find(int(fo[i])), find(int(fo[i + 1]))
        if x != y:
            lab[max(x, y)] = min(x, y)
    return np.array([find(f) for f in range(nf)], np.int64)


def _walk_runs(adj):
    """Polylines through a graph of degree <= 2 mostly: runs between ends / branch points, then closed loops (first
    node repeated at the end)."""

    seen = set()
    runs = []
    for v0 in sorted(adj):
        if len(adj[v0]) == 2:
            continue
        for w0 in sorted(adj[v0]):
            e = (min(v0, w0), max(v0, w0))
            if e in seen:
                continue
            seen.add(e)
            run, prev, cur = [v0], v0, w0
            while True:
                run.append(cur)
                if len(adj[cur]) != 2:
                    break
                nxt = next(x for x in adj[cur] if x != prev)
                e = (min(cur, nxt), max(cur, nxt))
                if e in seen:
                    break
                seen.add(e)
                prev, cur = cur, nxt
            runs.append(run)
    for v0 in sorted(adj):
        for w0 in sorted(adj[v0]):
            e = (min(v0, w0), max(v0, w0))
            if e in seen:
                continue
            seen.add(e)
            run, prev, cur = [v0], v0, w0
            while cur != v0:
                run.append(cur)
                nxt = next(x for x in adj[cur] if x != prev)
                e = (min(cur, nxt), max(cur, nxt))
                if e in seen:
                    break
                seen.add(e)
                prev, cur = cur, nxt
            run.append(cur)
            runs.append(run)
    return [r for r in runs if len(r) >= 2]


def part_contacts(seams: dict, points, indptr, flat, corner_uv, texture_px: float = 1024.0) -> dict:
    """The seam dict with **part contacts** added as chains of class `CONTACT`: the curves where two unconnected pieces of the source
    cut through each other (a bolt through a plate, a cap pushed into a body).

    Such pieces share no edge, so the UV seams have no chain there, yet a result face across the crease takes its
    texture from two places. The snap fits result edges onto the curve like a Soft chain (zero faces); the split leaves
    it alone. Side A is the lower part's UV there, side B the other part's.
    New chain vertices get welded ids after the source's (`chain_xyz` reads their places from `contact_xyz`).

    Our own wiring of textbook pieces: candidate triangle pairs from a uniform grid, segment-triangle crossings
    (Moller, Trumbore 1997, "Fast, minimum storage ray-triangle intersection"), each crossing keyed by (welded edge,
    other triangle) so the segments of neighbouring triangle pairs join exactly into polylines."""

    P = np.asarray(points, np.float64).reshape(-1, 3)
    ip = np.asarray(indptr, np.int64)
    fl = np.asarray(flat, np.int64)
    cuv = np.asarray(corner_uv, np.float64).reshape(-1, 2)
    weld = np.asarray(seams["weld"], np.int64)
    nw = int(seams["n_welded"])
    out = dict(seams)
    nf = len(ip) - 1
    if nf <= 1:
        return out
    part = _parts(fl, ip, weld, nw)
    if part.min() == part.max():
        return out
    # fan triangles: corner ids (into flat / corner_uv) and their face's part
    counts = np.diff(ip)
    tc = []
    for k in np.unique(counts[counts >= 3]).tolist():
        fs = np.flatnonzero(counts == k)
        for j in range(1, k - 1):
            tc.append(np.stack([ip[fs], ip[fs] + j, ip[fs] + j + 1], 1))
    if not tc:
        return out
    TC = np.concatenate(tc)
    tpart = part[np.searchsorted(ip, TC[:, 0], side="right") - 1]
    V = P[fl[TC]]  # (nt, 3, 3)
    nt = len(TC)
    # candidate pairs: triangles of different parts sharing a grid cell
    elen = np.linalg.norm(V[:, 1] - V[:, 0], axis=1)
    cell = 2.0 * float(np.median(elen[elen > 0])) if np.any(elen > 0) else 1.0
    lo = np.floor(V.min(1) / cell).astype(np.int64)
    hi = np.floor(V.max(1) / cell).astype(np.int64)
    span = hi - lo + 1
    ncell = span.prod(1)
    tid = np.repeat(np.arange(nt), ncell)
    k = np.arange(len(tid)) - np.repeat(np.cumsum(ncell) - ncell, ncell)
    sx, sy = span[tid, 0], span[tid, 1]
    cxyz = np.stack([lo[tid, 0] + k % sx, lo[tid, 1] + (k // sx) % sy, lo[tid, 2] + k // (sx * sy)], 1)
    ckey = np.unique(cxyz, axis=0, return_inverse=True)[1].ravel()
    order = np.lexsort((tid, ckey))
    ck, ct = ckey[order], tid[order]
    starts = np.flatnonzero(np.r_[True, ck[1:] != ck[:-1]])
    ends = np.r_[starts[1:], len(ck)]
    mixed = np.minimum.reduceat(tpart[ct], starts) != np.maximum.reduceat(tpart[ct], starts)
    pa, pb = [], []
    for s0, e0 in zip(starts[mixed].tolist(), ends[mixed].tolist()):
        g = ct[s0:e0]
        x, y = np.meshgrid(g, g, indexing="ij")
        m = (x < y) & (tpart[x] != tpart[y])
        pa.append(x[m])
        pb.append(y[m])
    if not pa:
        return out
    pairs = np.unique(np.stack([np.concatenate(pa), np.concatenate(pb)], 1), axis=0)
    # crossings: each edge of one triangle against the other triangle, both ways
    node, npos, nuv, by_pair = {}, [], [], {}
    for one, other in ((0, 1), (1, 0)):
        A, B = pairs[:, one], pairs[:, other]
        b0, b1, b2 = V[B, 0], V[B, 1], V[B, 2]
        e1, e2 = b1 - b0, b2 - b0
        for j in range(3):
            c0, c1 = TC[A, j], TC[A, (j + 1) % 3]
            o = P[fl[c0]]
            dvec = P[fl[c1]] - o
            pv = np.cross(dvec, e2)
            det = np.einsum("ij,ij->i", e1, pv)
            with np.errstate(all="ignore"):
                inv = 1.0 / det
                tv = o - b0
                u = np.einsum("ij,ij->i", tv, pv) * inv
                qv = np.cross(tv, e1)
                v = np.einsum("ij,ij->i", dvec, qv) * inv
                t = np.einsum("ij,ij->i", e2, qv) * inv
                ok = (np.abs(det) > 1e-300) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
            for i in np.flatnonzero(ok).tolist():
                wa, wb = int(weld[fl[c0[i]]]), int(weld[fl[c1[i]]])
                bb = int(B[i])
                key = (min(wa, wb), max(wa, wb), bb)
                if key not in node:
                    ti = float(t[i])
                    own = cuv[c0[i]] + ti * (cuv[c1[i]] - cuv[c0[i]])
                    oth = (1.0 - u[i] - v[i]) * cuv[TC[bb, 0]] + u[i] * cuv[TC[bb, 1]] + v[i] * cuv[TC[bb, 2]]
                    node[key] = len(npos)
                    npos.append(o[i] + ti * dvec[i])
                    nuv.append((own, oth) if tpart[A[i]] < tpart[bb] else (oth, own))
                by_pair.setdefault(i, set()).add(node[key])
    adj = {}
    for ns in by_pair.values():
        ns = sorted(ns)
        if len(ns) < 2:
            continue
        if len(ns) > 2:  # touching at a vertex or along an edge: the two farthest crossings
            q = np.array([npos[x] for x in ns])
            x, y = np.unravel_index(np.linalg.norm(q[:, None] - q[None], axis=2).argmax(), (len(ns), len(ns)))
            ns = [ns[x], ns[y]]
        adj.setdefault(ns[0], set()).add(ns[1])
        adj.setdefault(ns[1], set()).add(ns[0])
    runs = _walk_runs(adj)
    if not runs:
        return out
    rv, ua, ub, gp, ri = [], [], [], [], []
    for run in runs:
        for x in run:
            rv.append(nw + x)
            ua.append(nuv[x][0])
            ub.append(nuv[x][1])
            gp.append(float(np.linalg.norm(nuv[x][0] - nuv[x][1])) * texture_px)
        ri.append(len(rv))
    old_ip = np.asarray(seams["run_indptr"], np.int64)
    cls = np.asarray(seams["class"])
    out["run_indptr"] = np.r_[old_ip, np.asarray(ri, np.int64) + int(old_ip[-1])]
    out["run_vertex"] = np.r_[np.asarray(seams["run_vertex"], np.int64), np.asarray(rv, np.int64)]
    out["uv_a"] = np.r_[np.asarray(seams["uv_a"], np.float64).ravel(), np.asarray(ua, np.float64).ravel()]
    out["uv_b"] = np.r_[np.asarray(seams["uv_b"], np.float64).ravel(), np.asarray(ub, np.float64).ravel()]
    out["gap_px"] = np.r_[np.asarray(seams["gap_px"], np.float64), np.asarray(gp, np.float64)]
    out["class"] = np.r_[cls, np.full(len(runs), CONTACT, cls.dtype)]
    out["n_welded"] = nw + len(npos)
    out["contact_xyz"] = np.asarray(npos, np.float64).reshape(-1, 3)
    out["part_contacts"] = len(runs)
    out["part_contact_first"] = len(cls)
    return out


# ---------------------------------------------------------------- small helpers (scalar, same order as the reference)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(_dot(a, a))


def _lerp(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))


def _unit(a):
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 0.0 else (0.0, 0.0, 0.0)


def _newell(pos):
    n = (0.0, 0.0, 0.0)
    k = len(pos)
    for i in range(k):
        n = _add(n, _cross(pos[i], pos[(i + 1) % k]))
    return n


def _tolerance(pos):
    size = 0.0
    for c in range(3):
        lo = hi = pos[0][c]
        for q in pos[1:]:
            lo, hi = min(lo, q[c]), max(hi, q[c])
        size += (hi - lo) * (hi - lo)
    return max(size * 1e-14, 1e-20)


def _face_ok(pos) -> bool:
    if len(pos) < 3 or not all(math.isfinite(x) for p in pos for x in p):
        return False
    eps = _tolerance(pos)
    if len(pos) == 4:
        return bool(_quad_points_valid(np.array(pos, np.float64)[None], eps)[0])
    n = _newell(pos)
    if not _norm(n) > eps:
        return False
    p0 = pos[0]
    return all(_dot(_cross(_sub(pos[k], p0), _sub(pos[k + 1], p0)), n) > 0.0 for k in range(1, len(pos) - 1))


def _bbox(pts, pad):
    lo, hi = list(pts[0]), list(pts[0])
    for p in pts:
        for c in range(3):
            lo[c], hi[c] = min(lo[c], p[c]), max(hi[c], p[c])
    return (lo[0] - pad, lo[1] - pad, lo[2] - pad), (hi[0] + pad, hi[1] + pad, hi[2] + pad)


class _Mesh:
    """Result topology: edges (first-corner order), neighbours, faces per vertex, mean edge per vertex."""

    def __init__(self, points, indptr, flat, lazy=False):
        """`lazy`: build later through `build()`, a generator that yields between short pieces (the sliced stages)."""
        if not lazy:
            for _ in self.build(points, indptr, flat):
                pass

    def build(self, points, indptr, flat):
        P = np.asarray(points, np.float64).reshape(-1, 3)
        self.indptr = ip = np.asarray(indptr, np.int64)
        self.flat = fl = np.asarray(flat, np.int64)
        n = len(P)
        self.n, self.nf = n, len(ip) - 1
        counts = np.diff(ip)
        face_of = np.repeat(np.arange(self.nf), counts)
        nxt = np.arange(len(fl)) + 1
        last = ip[1:] - 1
        nxt[last[counts > 0]] = ip[:-1][counts > 0]
        a, b, start, occ = _edges(fl, nxt)
        self.ea, self.eb = a.astype(np.int64), b.astype(np.int64)
        d = P[self.eb] - P[self.ea]
        length = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        total = 0.0
        for x in length.tolist():
            total += x
        ne = len(self.ea)
        self.mean_edge = total / ne if ne else 1.0
        self.max_edge = float(length.max()) if ne else 0.0
        s = np.zeros(n)
        cnt = np.zeros(n, np.int64)
        np.add.at(s, np.stack([self.ea, self.eb], 1).ravel(), np.repeat(length, 2))
        np.add.at(cnt, np.stack([self.ea, self.eb], 1).ravel(), 1)
        self.vedge = np.where(cnt > 0, s / np.maximum(cnt, 1), self.mean_edge)
        self.efaces = []
        yield
        for e in range(ne):
            fs = face_of[occ[start[e]:start[e + 1]]].tolist()
            self.efaces.append([f for i, f in enumerate(fs) if i == 0 or fs[i - 1] != f])
            if e & 255 == 255:
                yield
        nb = [[] for _ in range(n)]
        for i, (x, y) in enumerate(zip(self.ea.tolist(), self.eb.tolist())):
            if x != y:
                nb[x].append(y)
                nb[y].append(x)
            if i & 1023 == 1023:
                yield
        self.nbr = [sorted(v) for v in nb]
        yield
        vf = [[] for _ in range(n)]
        for f in range(self.nf):
            for v in fl[ip[f]:ip[f + 1]].tolist():
                vf[v].append(f)
            if f & 511 == 511:
                yield
        self.vf = vf

    def face(self, f):
        return self.flat[self.indptr[f]:self.indptr[f + 1]].tolist()


class _Chain:
    def __init__(self, seams, xyz, c):
        a, b = int(seams["run_indptr"][c]), int(seams["run_indptr"][c + 1])
        ids = np.asarray(seams["run_vertex"][a:b], np.int64)
        X = np.asarray(xyz, np.float64).reshape(-1, 3)
        self.P = X[ids]
        self.pts = [tuple(p) for p in self.P.tolist()]
        ua = np.asarray(seams["uv_a"], np.float64).reshape(-1, 2)[a:b]
        ub = np.asarray(seams["uv_b"], np.float64).reshape(-1, 2)[a:b]
        self.uv_a, self.uv_b = [tuple(x) for x in ua.tolist()], [tuple(x) for x in ub.tolist()]
        self.gap = np.asarray(seams["gap_px"], np.float64)[a:b].tolist()
        self.closed = len(ids) > 2 and ids[0] == ids[-1]
        self.start_id, self.end_id = int(ids[0]), int(ids[-1])

    def closest(self, lo, hi, q):
        best = (math.inf, float(lo))
        for j in range(lo, hi):
            a, b = self.pts[j], self.pts[j + 1]
            ab = _sub(b, a)
            l2 = _dot(ab, ab)
            t = _dot(_sub(q, a), ab) / l2 if l2 > 0.0 else 0.0
            t = min(max(t, 0.0), 1.0)
            d = _sub(q, _lerp(a, b, t))
            d2 = _dot(d, d)
            if d2 < best[0]:
                best = (d2, j + t)
        return best

    def closest_many(self, lo, hi, Q):
        """`closest` for many points at once (same operations, elementwise)."""

        best = np.full(len(Q), np.inf)
        prm = np.full(len(Q), float(lo))
        for j in range(lo, hi):
            a, b = self.P[j], self.P[j + 1]
            ab = b - a
            l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]
            qa = Q - a
            t = (qa[:, 0] * ab[0] + qa[:, 1] * ab[1] + qa[:, 2] * ab[2]) / l2 if l2 > 0.0 else np.zeros(len(Q))
            t = np.minimum(np.maximum(t, 0.0), 1.0)
            d = Q - np.stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])], 1)
            d2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
            better = d2 < best
            best[better] = d2[better]
            prm[better] = j + t[better]
        return best, prm

    def seg(self, param):
        last = max(len(self.pts) - 2, 0)
        j = min(int(math.floor(max(param, 0.0))), last)
        return j, param - j

    def at(self, param):
        if len(self.pts) < 2:
            return self.pts[0]
        j, t = self.seg(param)
        return _lerp(self.pts[j], self.pts[j + 1], t)

    def tangent(self, param):
        if len(self.pts) < 2:
            return (0.0, 0.0, 0.0)
        j, _ = self.seg(param)
        return _unit(_sub(self.pts[j + 1], self.pts[j]))

    def uv(self, param, side_b):
        uv = self.uv_b if side_b else self.uv_a
        if len(uv) < 2:
            return uv[0]
        j, t = self.seg(param)
        return (uv[j][0] + t * (uv[j + 1][0] - uv[j][0]), uv[j][1] + t * (uv[j + 1][1] - uv[j][1]))

    def gap_at(self, param):
        if len(self.gap) < 2:
            return self.gap[0]
        j, t = self.seg(param)
        return self.gap[j] + t * (self.gap[j + 1] - self.gap[j])

    def max_gap(self):
        return max([0.0] + self.gap)

    def end(self):
        return float(len(self.pts) - 1)


def _chains(seams, xyz):
    return [_Chain(seams, xyz, c) for c in range(len(seams["class"]))]


def _face_pos(pos, mesh, f):
    return [pos[v] for v in mesh.face(f)]


def _move_ok(pos, mesh, v, to) -> bool:
    before = []
    for f in mesh.vf[v]:
        p = _face_pos(pos, mesh, f)
        before.append((_face_ok(p), _newell(p)))
    frm = pos[v]
    pos[v] = to
    ok = True
    for f, (was, n0) in zip(mesh.vf[v], before):
        p = _face_pos(pos, mesh, f)
        if not ((_face_ok(p) or not was) and _dot(_newell(p), n0) > 0.0):
            ok = False
            break
    pos[v] = frm
    return ok


def _nearest(P, q) -> int:
    d = P - np.asarray(q)
    d2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
    return int(np.argmin(d2))


# ---------------------------------------------------------------- snap (SPEC 4.5)


def _snap_steps(points, indptr, flat, seams, xyz, max_move, vchain_in, vparam_in, rings, out, stopped):
    """Generator behind `snap`: fills `out`, yielding after each chain and each relaxed vertex batch."""

    mesh = _Mesh(points, indptr, flat)
    n = mesh.n
    P = np.asarray(points, np.float64).reshape(-1, 3)
    orig = [tuple(p) for p in P.tolist()]
    pos = list(orig)
    chs = _chains(seams, xyz)
    nc = len(chs)
    cls = np.asarray(seams["class"], np.int64)
    mm = np.asarray(max_move, np.float64).ravel()
    limit = (lambda v: float(mm[v])) if len(mm) == n else (lambda v: math.inf)
    vc = np.asarray(vchain_in, np.int64).ravel()
    vp = np.asarray(vparam_in, np.float64).ravel()
    out.update(held=np.zeros(n, bool), snapped=np.zeros(n, bool), vchain=vc.copy() if len(vc) == n else np.full(n, -1, np.int64),
               vparam=vp.copy() if len(vp) == n else np.zeros(n), state=np.zeros(nc, np.int64), mean_dist=np.zeros(nc),
               turns=np.zeros(nc, np.int64), path_edges=np.zeros(nc, np.int64), moved=0, relaxed=0, done=True)
    held, snapped, vchain, vparam = out["held"], out["snapped"], out["vchain"], out["vparam"]
    vedge = mesh.vedge
    max_vedge = float(vedge.max()) if n else 0.0
    yield
    for c in range(nc):
        if stopped():
            out["done"] = False
            break
        ch = chs[c]
        if cls[c] not in (SOFT, CONTACT) or len(ch.pts) < 2 or n == 0:
            continue
        Pc = np.array(pos)
        near = [_nearest(Pc, p) for p in ch.pts]
        on = all(_norm(_sub(pos[v], p)) <= ON_EDGE_SHARE * vedge[v] for v, p in zip(near, ch.pts)) and all(
            x == y or y in mesh.nbr[x] for x, y in zip(near[:-1], near[1:]))
        if on:
            for i, v in enumerate(near):
                if vchain[v] < 0:
                    vchain[v], vparam[v] = c, float(i)
            out["state"][c] = ON_EDGES
            yield
            continue
        last = len(ch.pts) - 1
        halves = [(0, last // 2), (last // 2, last)] if ch.closed and last >= 2 else [(0, last)]
        states = []
        for j0, j1 in halves:
            a, b = ch.pts[j0], ch.pts[j1]
            s, t = _nearest(Pc, a), _nearest(Pc, b)
            if s == t:
                states.append(SHORT)
                continue
            lo, hi = _bbox(ch.pts[j0:j1 + 1], SNAP_REACH * max_vedge)
            inbox = np.flatnonzero(np.all((Pc >= lo) & (Pc <= hi), axis=1))
            free = (vchain[inbox] < 0) | (vchain[inbox] == c) | (inbox == s) | (inbox == t)
            cand = inbox[free]
            d2, prm = ch.closest_many(j0, j1, Pc[cand])
            dd = np.sqrt(d2)
            keep = (dd <= SNAP_REACH * vedge[cand]) | (cand == s) | (cand == t)
            dist = {int(v): (float(x), float(y)) for v, x, y in zip(cand[keep], dd[keep], prm[keep])}
            for v in (s, t):
                if v not in dist:
                    x, y = ch.closest(j0, j1, pos[v])
                    dist[v] = (math.sqrt(x), y)
            best = {s: (0.0, -1)}
            heap = [(0.0, s)]
            while heap:
                cost, u = heapq.heappop(heap)
                if cost > best[u][0]:
                    continue
                if u == t:
                    break
                du = dist[u][0] / vedge[u]
                for w in mesh.nbr[u]:
                    if w not in dist:
                        continue
                    dw = dist[w][0]
                    step = cost + _norm(_sub(pos[w], pos[u])) * (1.0 + SNAP_COST_K * (du + dw / vedge[w]) / 2.0)
                    if w not in best or step < best[w][0]:
                        best[w] = (step, u)
                        heapq.heappush(heap, (step, w))
            if t not in best:
                states.append(NO_PATH)
                continue
            path = [t]
            while path[-1] != s:
                path.append(best[path[-1]][1])
            path.reverse()
            total = 0.0
            for v in path:
                total += dist[v][0] / vedge[v]
            mean = total / len(path)
            turns = 0
            for u, w in zip(path[:-1], path[1:]):
                pu, pw = pos[u], pos[w]
                _, pm = ch.closest(j0, j1, _scale(_add(pu, pw), 0.5))
                along = _dot(_sub(pw, pu), ch.tangent(pm))
                if along < SNAP_TURN_COS * _norm(_sub(pw, pu)):
                    turns += 1
            ne = len(path) - 1
            out["mean_dist"][c] = max(out["mean_dist"][c], mean)
            out["turns"][c] += turns
            out["path_edges"][c] += ne
            if not mean < SNAP_MEAN_DIST or turns * SNAP_TURN_EVERY > ne:
                states.append(REFUSED)
                continue
            any_held = False
            for i, v in enumerate(path):
                if vchain[v] >= 0:
                    continue
                pv = float(j0) if i == 0 else float(j1) if i == len(path) - 1 else dist[v][1]
                to = ch.at(pv)
                if _norm(_sub(to, orig[v])) > limit(v) or not _move_ok(pos, mesh, v, to):
                    held[v] = True
                    any_held = True
                    continue
                pos[v] = to
                held[v] = False
                snapped[v] = True
                vchain[v], vparam[v] = c, pv
                out["moved"] += 1
            Pc = np.array(pos)
            states.append(PARTLY_HELD if any_held else ACCEPTED)
        if all(x == ACCEPTED for x in states):
            out["state"][c] = ACCEPTED
        elif any(x in (ACCEPTED, PARTLY_HELD) for x in states):
            out["state"][c] = PARTLY_HELD
        else:
            out["state"][c] = states[0]
        yield
    ring = [v for v in range(n) if snapped[v]]
    seen = snapped.copy()
    for _ in range(int(rings)):
        if not out["done"] or stopped():
            out["done"] = False
            break
        nxt = []
        for u in ring:
            for w in mesh.nbr[u]:
                if not seen[w]:
                    seen[w] = True
                    nxt.append(w)
        nxt.sort()
        for v in nxt:
            if held[v] or vchain[v] >= 0 or not mesh.nbr[v] or not limit(v) > 0.0:
                continue
            m = (0.0, 0.0, 0.0)
            for w in mesh.nbr[v]:
                m = _add(m, pos[w])
            m = _scale(m, 1.0 / len(mesh.nbr[v]))
            nv = (0.0, 0.0, 0.0)
            for f in mesh.vf[v]:
                nv = _add(nv, _unit(_newell(_face_pos(pos, mesh, f))))
            nv = _unit(nv)
            d = _sub(m, pos[v])
            dt = _sub(d, _scale(nv, _dot(d, nv)))
            to = _add(pos[v], _scale(dt, RELAX_WEIGHT))
            if _norm(_sub(to, orig[v])) <= limit(v) and _move_ok(pos, mesh, v, to):
                pos[v] = to
                out["relaxed"] += 1
            yield
        ring = nxt
    out["points"] = np.array(pos, np.float64).ravel()


def snap(points, indptr, flat, seams: dict, xyz, max_move=(), vchain=(), vparam=(), rings: int = RELAX_RINGS, use_fast: bool = True) -> dict:
    """Seam snap in one call (the faster callable when installed and `use_fast`, else the twin)."""

    stage = Snap(points, indptr, flat, seams, xyz, max_move, vchain, vparam, rings, use_fast=use_fast)
    while not stage.done:
        stage.step()
    return stage.result


# ---------------------------------------------------------------- split leftovers (SPEC 4.7)

# Nodes: (0, vertex, 0) an existing vertex, (1, edge, chain) a crossing on an edge, (2, corner id, 0) a seam corner
# inside a face. Tuple order = the reference's node order.
_V, _E, _C = 0, 1, 2


def _inside(fp, x) -> bool:
    n = _unit(_newell(fp))
    k = len(fp)
    c = (0.0, 0.0, 0.0)
    perim = 0.0
    for i in range(k):
        c = _add(c, fp[i])
        perim += _norm(_sub(fp[(i + 1) % k], fp[i]))
    c = _scale(c, 1.0 / k)
    if abs(_dot(_sub(x, c), n)) > CROSS_REACH * perim / k:
        return False
    return all(_dot(_sub(x, fp[i]), _cross(n, _sub(fp[(i + 1) % k], fp[i]))) >= 0.0 for i in range(k))


def _face_dist(fp, x):
    best = math.inf
    for k in range(1, len(fp) - 1):
        a, b, c = fp[0], fp[k], fp[k + 1]
        w = closest_bary(x, a, b, c)
        p = (w[0] * a[0] + w[1] * b[0] + w[2] * c[0], w[0] * a[1] + w[1] * b[1] + w[2] * c[1], w[0] * a[2] + w[1] * b[2] + w[2] * c[2])
        best = min(best, _norm(_sub(x, p)))
    return best


def _to_quads(poly, at, normal):
    out = []
    cur = list(poly)
    while len(cur) > 4:
        m = len(cur)
        best = None
        for i in range(m):
            q = [cur[i], cur[(i + 1) % m], cur[(i + 2) % m], cur[(i + 3) % m]]
            p = [at[x] for x in q]
            if not _face_ok(p) or _dot(_newell(p), normal) <= 0.0:
                continue
            score = math.inf
            for k in range(4):
                e1, e2 = _sub(p[(k + 1) % 4], p[k]), _sub(p[(k + 3) % 4], p[k])
                score = min(score, _dot(_cross(e1, e2), normal) / (_norm(e1) * _norm(e2)))
            if best is None or score > best[0]:
                best = (score, i)
        if best is not None:
            i = best[1]
            out.append([cur[i], cur[(i + 1) % m], cur[(i + 2) % m], cur[(i + 3) % m]])
            r1, r2 = (i + 1) % m, (i + 2) % m
            cur = [x for k, x in enumerate(cur) if k != r1 and k != r2]
        else:
            ear = 0
            for i in range(m):
                p = [at[cur[i]], at[cur[(i + 1) % m]], at[cur[(i + 2) % m]]]
                if _dot(_cross(_sub(p[1], p[0]), _sub(p[2], p[0])), normal) > 0.0:
                    ear = i
                    break
            out.append([cur[ear], cur[(ear + 1) % m], cur[(ear + 2) % m]])
            del cur[(ear + 1) % m]
    out.append(cur)
    return out


def _node_edges(node, fv, fe):
    k = len(fv)
    if node[0] == _V:
        out = []
        for i in range(k):
            if fv[i] == node[1]:
                out += [fe[i], fe[(i + k - 1) % k]]
        return out
    if node[0] == _E:
        return [node[1]]
    return []


class _Dsu:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _split_steps(points, indptr, flat, seams, xyz, vchain, vparam, seen_face, budget, out, stopped):
    """Generator behind `split`: fills `out`, yielding between pieces of work."""

    mesh = _Mesh(points, indptr, flat, lazy=True)
    yield from mesh.build(points, indptr, flat)
    n, nf = mesh.n, mesh.nf
    P = np.asarray(points, np.float64).reshape(-1, 3)
    pos = [tuple(p) for p in P.tolist()]
    chs = _chains(seams, xyz)
    yield
    nc = len(chs)
    cls = np.asarray(seams["class"], np.int64)
    vc = np.asarray(vchain, np.int64).ravel()
    vp = np.asarray(vparam, np.float64).ravel()
    tag_c = vc.tolist() if len(vc) == n else [-1] * n
    tag_p = vp.tolist() if len(vp) == n else [0.0] * n
    seen_face = np.asarray(seen_face, np.float64).ravel()
    active = [bool(cls[c] == SOFT and len(chs[c].pts) >= 2 and chs[c].max_gap() >= SPLIT_GAP_PX) for c in range(nc)]
    stop_hit = False
    edge_of = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(mesh.ea.tolist(), mesh.eb.tolist()))}

    def side_edge(a, b):
        return edge_of[(min(a, b), max(a, b))]

    fnormal = []
    for f in range(nf):
        fnormal.append(_unit(_newell(_face_pos(pos, mesh, f))))
        if f & 127 == 127:
            yield
    enormal = []
    for e in range(len(mesh.ea)):
        s = (0.0, 0.0, 0.0)
        for f in mesh.efaces[e]:
            s = _add(s, fnormal[f])
        enormal.append(s)
        if e & 255 == 255:
            yield
    EN = np.array(enormal, np.float64).reshape(-1, 3)
    PA, PB = P[mesh.ea], P[mesh.eb]
    # Candidates: an edge's box padded by CROSS_REACH of its length, a face's box padded by the inside() plane tolerance.
    dE = PB - PA
    pad = (CROSS_REACH * np.sqrt(dE[:, 0] * dE[:, 0] + dE[:, 1] * dE[:, 1] + dE[:, 2] * dE[:, 2]))[:, None]
    ELO, EHI = np.minimum(PA, PB) - pad, np.maximum(PA, PB) + pad
    FLO, FHI = np.zeros((nf, 3)), np.zeros((nf, 3))
    for f in range(nf):
        fp = _face_pos(pos, mesh, f)
        k = len(fp)
        perim = 0.0
        for i in range(k):
            perim += _norm(_sub(fp[(i + 1) % k], fp[i]))
        FLO[f], FHI[f] = _bbox(fp, CROSS_REACH * perim / k)
        if f & 127 == 127:
            yield
    corner_map = {}
    for c, ch in enumerate(chs):
        if not ch.closed:
            corner_map.setdefault(ch.start_id, []).append(c)
            if ch.end_id != ch.start_id:
                corner_map.setdefault(ch.end_id, []).append(c)
    yield

    def near_edges(lo, hi):
        return np.flatnonzero(np.all((ELO <= np.asarray(hi)) & (EHI >= np.asarray(lo)), axis=1))

    def faces_at(x):
        x = np.asarray(x)
        return np.flatnonzero(np.all((FLO <= x) & (FHI >= x), axis=1)).tolist()

    # 1. Crossings per (edge, chain), once.
    crosses = {}
    pq = PB - PA
    m_all = np.stack([pq[:, 1] * EN[:, 2] - pq[:, 2] * EN[:, 1], pq[:, 2] * EN[:, 0] - pq[:, 0] * EN[:, 2], pq[:, 0] * EN[:, 1] - pq[:, 1] * EN[:, 0]], 1)
    l2_all = pq[:, 0] * pq[:, 0] + pq[:, 1] * pq[:, 1] + pq[:, 2] * pq[:, 2]
    tags = np.asarray(tag_c, np.int64)
    for c in range(nc):
        if not active[c]:
            continue
        if stopped():
            stop_hit = True
            break
        ch = chs[c]
        lo, hi = _bbox(ch.pts, 0.0)
        sub_e = near_edges(lo, hi)
        sub_e = sub_e[~((tags[mesh.ea[sub_e]] == c) & (tags[mesh.eb[sub_e]] == c))]
        best = np.full(len(sub_e), np.inf)
        bs, bp = np.zeros(len(sub_e)), np.zeros(len(sub_e))
        for j in range(len(ch.pts) - 1):
            pa, pb = ch.P[j], ch.P[j + 1]
            slo, shi = _bbox([ch.pts[j], ch.pts[j + 1]], 0.0)
            k = np.flatnonzero(np.all((ELO[sub_e] <= np.asarray(shi)) & (EHI[sub_e] >= np.asarray(slo)), axis=1))
            if not len(k):
                continue
            e = sub_e[k]
            p, m = PA[e], m_all[e]
            a0, a1 = pa - p, pb - p
            d0 = a0[:, 0] * m[:, 0] + a0[:, 1] * m[:, 1] + a0[:, 2] * m[:, 2]
            d1 = a1[:, 0] * m[:, 0] + a1[:, 1] * m[:, 1] + a1[:, 2] * m[:, 2]
            hit = (d0 > 0.0) != (d1 > 0.0)
            with np.errstate(all="ignore"):
                lam = d0 / (d0 - d1)
                x = np.stack([pa[0] + lam * (pb[0] - pa[0]), pa[1] + lam * (pb[1] - pa[1]), pa[2] + lam * (pb[2] - pa[2])], 1)
                l2 = l2_all[e]
                q = pq[e]
                xp = x - p
                s = (xp[:, 0] * q[:, 0] + xp[:, 1] * q[:, 1] + xp[:, 2] * q[:, 2]) / l2
                y = np.stack([p[:, 0] + s * (PB[e][:, 0] - p[:, 0]), p[:, 1] + s * (PB[e][:, 1] - p[:, 1]), p[:, 2] + s * (PB[e][:, 2] - p[:, 2])], 1)
                g = x - y
                gapd = np.sqrt(g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1] + g[:, 2] * g[:, 2])
                ok = hit & (l2 > 0.0) & (s >= 0.0) & (s <= 1.0) & ~(gapd > CROSS_REACH * np.sqrt(l2)) & (gapd < best[k])
            for i in np.flatnonzero(ok).tolist():  # the far side of a thin part: another face is clearly nearer
                ei, xi = int(e[i]), tuple(x[i].tolist())
                own = min(_face_dist(_face_pos(pos, mesh, f), xi) for f in mesh.efaces[ei])
                if any(f not in mesh.efaces[ei] and _face_dist(_face_pos(pos, mesh, f), xi) < NEARER_SHARE * own for f in faces_at(xi)):
                    ok[i] = False
            kk = k[ok]
            best[kk], bs[kk], bp[kk] = gapd[ok], s[ok], j + lam[ok]
            yield
        for i in np.flatnonzero(np.isfinite(best)).tolist():
            e, s = int(sub_e[i]), float(bs[i])
            reuse = int(mesh.ea[e]) if s < REUSE_EDGE else int(mesh.eb[e]) if s > 1.0 - REUSE_EDGE else None
            crosses[(e, c)] = (float(best[i]), (s, float(bp[i]), reuse))
    cross_keys = sorted(crosses)
    yield

    # 2. Points per face.
    fpts = {}

    def push(f, c, node, prm):
        lst = fpts.setdefault(f, [])
        if not any(cc == c and nn == node for cc, nn, _ in lst):
            lst.append((c, node, prm))

    end_tagged = [[False, False] for _ in range(nc)]
    for v in range(n):
        if v & 1023 == 1023:
            yield
        if tag_c[v] < 0:
            continue
        c, prm = tag_c[v], tag_p[v]
        on = [(c, prm)]
        if not chs[c].closed and (prm == 0.0 or prm == chs[c].end()):
            cid = chs[c].start_id if prm == 0.0 else chs[c].end_id
            on += [(cc, 0.0 if chs[cc].start_id == cid else chs[cc].end()) for cc in corner_map[cid] if cc != c]
        for cc, pp in on:
            if pp == 0.0:
                end_tagged[cc][0] = True
            if pp == chs[cc].end():
                end_tagged[cc][1] = True
            if active[cc]:
                for f in mesh.vf[v]:
                    push(f, cc, (_V, v, 0), pp)
    Pv = np.array(pos, np.float64).reshape(-1, 3)
    reuse_tags = []
    yield
    for i_, (e, c) in enumerate(cross_keys):
        if i_ & 127 == 127:
            yield
        s, prm, reuse = crosses[(e, c)][1]
        if reuse is not None:
            reuse_tags.append((reuse, c, prm))
        node = (_V, reuse, 0) if reuse is not None else (_E, e, c)
        for f in (mesh.vf[reuse] if reuse is not None else mesh.efaces[e]):
            push(f, c, node, prm)
    corner_at = {}
    for c in range(nc):
        if not active[c] or chs[c].closed:
            continue
        ch = chs[c]
        yield
        for k, (cid, prm) in enumerate(((ch.start_id, 0.0), (ch.end_id, ch.end()))):
            x = ch.at(prm)
            if end_tagged[c][k]:
                continue
            near = _nearest(Pv, x)
            if _norm(_sub(pos[near], x)) < REUSE_EDGE * mesh.vedge[near]:
                reuse_tags.append((near, c, prm))
                for g in mesh.vf[near]:
                    push(g, c, (_V, near, 0), prm)
                continue
            f = next((f for f in faces_at(x) if _inside(_face_pos(pos, mesh, f), x)), None)
            if f is None:
                continue
            fv = mesh.face(f)
            kk = len(fv)
            on_edge = None
            for i in range(kk):
                e = side_edge(fv[i], fv[(i + 1) % kk])
                p, q = pos[mesh.ea[e]], pos[mesh.eb[e]]
                pqv = _sub(q, p)
                l2 = _dot(pqv, pqv)
                if not l2 > 0.0:
                    continue
                s = min(max(_dot(_sub(x, p), pqv) / l2, 0.0), 1.0)
                d = _norm(_sub(x, _lerp(p, q, s)))
                if d < REUSE_EDGE * math.sqrt(l2) and (on_edge is None or d < on_edge[0]):
                    on_edge = (d, e, s)
            if on_edge is not None:
                _, e, s = on_edge
                if (e, c) not in crosses:
                    crosses[(e, c)] = (0.0, (s, prm, None))
                    for g in mesh.efaces[e]:
                        push(g, c, (_E, e, c), prm)
            else:
                corner_at.setdefault(cid, (f, x))
                if corner_at[cid][0] == f:
                    push(f, c, (_C, cid, 0), prm)
    yield

    # 3. Plans per face.
    face_list = sorted(fpts)
    plans = {}
    for i_, f in enumerate(face_list):
        if i_ & 15 == 15:
            yield
        fv = mesh.face(f)
        k = len(fv)
        fe = [side_edge(fv[i], fv[(i + 1) % k]) for i in range(k)]
        fp = _face_pos(pos, mesh, f)
        lst = sorted(fpts[f], key=lambda p: (p[0], p[2], p[1]))
        plan = {"chords": [], "fan": None, "cannot": False, "whole": False}
        corners = []
        i = 0
        while i < len(lst):
            c = lst[i][0]
            j = i
            while j < len(lst) and lst[j][0] == c:
                j += 1
            pts = lst[i:j]
            ch = chs[c]
            pairs = [(q, q + 1, 0.5 * (pts[q][2] + pts[q + 1][2])) for q in range(len(pts) - 1)]
            if ch.closed and len(pts) >= 2:
                mid = 0.5 * (pts[-1][2] + pts[0][2] + ch.end())
                if mid > ch.end():
                    mid -= ch.end()
                pairs.append((len(pts) - 1, 0, mid))
            explained = [False] * len(pts)
            for x, y, mid in pairs:
                na, nb = pts[x][1], pts[y][1]
                if na == nb:
                    continue
                ea, eb = _node_edges(na, fv, fe), _node_edges(nb, fv, fe)
                if any(e in eb for e in ea):
                    explained[x] = explained[y] = True
                    continue
                if not _inside(fp, ch.at(mid)):
                    continue
                explained[x] = explained[y] = True
                if any((a == na and b == nb) or (a == nb and b == na) for a, b, _, _, _ in plan["chords"]):
                    continue
                plan["chords"].append((na, nb, c, pts[x][2], pts[y][2]))
            for p in pts:
                if p[1][0] == _C and p[1] not in corners:
                    corners.append(p[1])
            if any(not ok and p[1][0] == _E for p, ok in zip(pts, explained)):
                plan["cannot"] = True
            i = j
        if corners:
            cn = corners[0]
            spokes = [(b, c, pb) if a == cn else (a, c, pa) for a, b, c, pa, pb in plan["chords"] if a == cn or b == cn]
            has_edge = any(p[1][0] == _E for p in lst)
            if len(corners) == 1 and len(spokes) >= 2:
                plan["fan"] = (cn, spokes)
                plan["chords"] = [ch_ for ch_ in plan["chords"] if ch_[0] != cn and ch_[1] != cn]
            elif not plan["chords"] and not has_edge:
                plan["whole"] = True
            else:
                plan["cannot"] = True
        if plan["cannot"]:
            plan["chords"], plan["fan"] = [], None
        plans[f] = plan
    yield

    # 4. Used crossing vertices and the runs of faces split together.
    used = set()
    for f in face_list:
        p = plans[f]
        for a, b, _, _, _ in p["chords"]:
            used.add(a)
            used.add(b)
        if p["fan"] is not None:
            used.add(p["fan"][0])
            used.update(x[0] for x in p["fan"][1])
    used = sorted(used)
    used_set = set(used)
    yield
    dsu = _Dsu(nf)
    for node in used:
        if node[0] == _E:
            fs = mesh.efaces[node[1]]
            for x, y in zip(fs[:-1], fs[1:]):
                dsu.union(x, y)
    touched = {f for f in face_list if plans[f]["chords"] or plans[f]["fan"] is not None}
    for node in used:
        if node[0] == _E:
            touched.update(mesh.efaces[node[1]])
    touched = sorted(touched)
    yield

    at = list(pos)
    node_id = {}
    new_tag = []
    for node in used:
        if node[0] == _V:
            node_id[node] = node[1]
        elif node[0] == _E:
            e, c = node[1], node[2]
            s = crosses[(e, c)][1][0]
            node_id[node] = len(at)
            at.append(_lerp(pos[mesh.ea[e]], pos[mesh.eb[e]], s))
            new_tag.append((c, crosses[(e, c)][1][1]))
    for node in used:
        if node[0] == _C:
            cid = node[1]
            f, x = corner_at[cid]
            fp = _face_pos(pos, mesh, f)
            nrm = fnormal[f]
            cen = (0.0, 0.0, 0.0)
            for p in fp:
                cen = _add(cen, p)
            cen = _scale(cen, 1.0 / len(fp))
            node_id[node] = len(at)
            at.append(_sub(x, _scale(nrm, _dot(_sub(x, cen), nrm))))
            c = next(c for c in corner_map[cid] if active[c])
            new_tag.append((c, 0.0 if chs[c].start_id == cid else chs[c].end()))
    yield
    cross_by_edge = {}
    for (e, c) in sorted(crosses):
        cross_by_edge.setdefault(e, []).append(c)

    def pieces_of(f):
        fv = mesh.face(f)
        k = len(fv)
        ring = []
        for i in range(k):
            ring.append(fv[i])
            a, b = fv[i], fv[(i + 1) % k]
            e = side_edge(a, b)
            on = []
            for c in cross_by_edge.get(e, ()):
                s, _, reuse = crosses[(e, c)][1]
                node = (_E, e, c)
                if reuse is None and node in used_set:
                    on.append((s if mesh.ea[e] == a else 1.0 - s, node_id[node]))
            on.sort()
            ring += [x[1] for x in on]
        plan = plans.get(f)
        pieces = [list(ring)]
        if plan is not None:
            if plan["fan"] is not None:
                cn, spokes = plan["fan"]
                cid = node_id[cn]
                ends = sorted((ring.index(node_id[sp[0]]), node_id[sp[0]]) for sp in spokes)
                pieces = []
                for q in range(len(ends)):
                    s0, e0 = ends[q][0], ends[(q + 1) % len(ends)][0]
                    piece = [cid]
                    i = s0
                    while True:
                        piece.append(ring[i])
                        if i == e0:
                            break
                        i = (i + 1) % len(ring)
                    pieces.append(piece)
            for a, b, _, _, _ in plan["chords"]:
                ia, ib = node_id[a], node_id[b]
                w = next((w for w, pc in enumerate(pieces) if ia in pc and ib in pc), None)
                if w is None:
                    continue
                pc = pieces.pop(w)
                x, y = pc.index(ia), pc.index(ib)
                x, y = min(x, y), max(x, y)
                if y - x == 1 or (x == 0 and y == len(pc) - 1):
                    pieces.append(pc)
                    continue
                pieces.append(pc[x:y + 1])
                pieces.append(pc[y:] + pc[:x + 1])
        res = []
        for pc in pieces:
            res += _to_quads(pc, at, fnormal[f])
        return res

    # 5. Budget.
    runs = {}
    for f in touched:
        runs.setdefault(dsu.find(f), []).append(f)
    run_list = []
    piece_cache = {}
    for r in sorted(runs):
        members = runs[r]
        damage, cost = 0.0, 0
        for i_, f in enumerate(members):
            if i_ & 7 == 7:
                yield
            pc = pieces_of(f)
            cost += len(pc) - 1
            p = plans.get(f)
            if p is not None:
                g = 0.0
                for _, _, c, pa, pb in p["chords"]:
                    g = max(g, chs[c].gap_at(pa), chs[c].gap_at(pb))
                if p["fan"] is not None:
                    cid = p["fan"][0][1]
                    for c in corner_map[cid]:
                        if active[c]:
                            g = max(g, chs[c].max_gap())
                area = 0.5 * _norm(_newell(_face_pos(pos, mesh, f)))
                seen = float(seen_face[f]) if len(seen_face) == nf else 1.0
                damage += area * seen * g
            piece_cache[f] = pc
        run_list.append((damage, members[0], members, cost))
        yield
    run_list.sort(key=lambda r: (-r[0], r[1]))
    wanted = 0.0
    for r in run_list:
        wanted += float(r[3])
    chosen = [False] * nf
    spent = 0.0
    for _, _, members, cost in run_list:
        if budget < 0.0 or spent + float(cost) <= budget:
            spent += float(cost)
            for f in members:
                chosen[f] = True

    state = [KEPT] * nf
    for f in face_list:
        p = plans[f]
        if p["whole"]:
            state[f] = BAKE
        elif p["cannot"]:
            state[f] = CANNOT
        elif p["chords"] or p["fan"] is not None:
            state[f] = SPLIT if chosen[f] else LEFT
    chain_touched = [False] * nc
    for f in face_list:
        for p in fpts[f]:
            chain_touched[p[0]] = True
    for c in range(nc):
        if not active[c] or not chs[c].closed or chain_touched[c]:
            continue
        ch = chs[c]
        cen = (0.0, 0.0, 0.0)
        for p in ch.pts[:-1]:
            cen = _add(cen, p)
        cen = _scale(cen, 1.0 / (len(ch.pts) - 1))
        f = next((f for f in faces_at(cen) if _inside(_face_pos(pos, mesh, f), cen)), None)
        if f is not None:
            state[f] = BAKE
        yield
    yield

    # 6. Output mesh.
    keep_node = [True] * n + [False] * (len(at) - n)
    for f in range(nf):
        if chosen[f]:
            for pc in piece_cache[f]:
                for v in pc:
                    keep_node[v] = True
    remap = [-1] * len(at)
    out_pts = []
    for i, k in enumerate(keep_node):
        if k:
            remap[i] = len(out_pts)
            out_pts.append(at[i])
    for i in range(n, len(at)):
        if keep_node[i]:
            c, prm = new_tag[i - n]
            tag_c.append(c)
            tag_p.append(prm)
    for v, c, prm in reuse_tags:
        if tag_c[v] < 0:
            tag_c[v], tag_p[v] = c, prm
    for f in face_list:
        if not chosen[f]:
            continue
        p = plans[f]
        ends = [x for a, b, c, pa, pb in p["chords"] for x in ((a, c, pa), (b, c, pb))]
        if p["fan"] is not None:
            ends += list(p["fan"][1])
        for node, c, prm in ends:
            if node[0] == _V and tag_c[node[1]] < 0:
                tag_c[node[1]], tag_p[node[1]] = c, prm
    yield
    polys = []
    for f in range(nf):
        if f & 1023 == 1023:
            yield
        if chosen[f]:
            for pc in piece_cache[f]:
                polys.append([[remap[v] for v in pc], f, True])
        else:
            polys.append([mesh.face(f), f, False])
    seam_edge = set()
    for f in face_list:
        if not chosen[f]:
            continue
        p = plans[f]
        for a, b, _, _, _ in p["chords"]:
            x, y = remap[node_id[a]], remap[node_id[b]]
            seam_edge.add((min(x, y), max(x, y)))
        if p["fan"] is not None:
            x = remap[node_id[p["fan"][0]]]
            for sp in p["fan"][1]:
                y = remap[node_id[sp[0]]]
                seam_edge.add((min(x, y), max(x, y)))
    by_edge = {}
    for i, (pv, _, frm) in enumerate(polys):
        if frm and len(pv) == 3:
            for k in range(3):
                a, b = pv[k], pv[(k + 1) % 3]
                by_edge.setdefault((min(a, b), max(a, b)), []).append(i)
    gone = [False] * len(polys)
    yield
    for i in range(len(polys)):
        if i & 255 == 255:
            yield
        if gone[i] or not polys[i][2] or len(polys[i][0]) != 3:
            continue
        tri = list(polys[i][0])
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            key = (min(a, b), max(a, b))
            if key in seam_edge:
                continue
            j = next((j for j in by_edge[key] if j != i and not gone[j] and len(polys[j][0]) == 3), None)
            if j is None:
                continue
            other = polys[j][0]
            apex = next(v for v in other if v != a and v != b)
            opp = tri[(k + 2) % 3]
            quad = [a, apex, b, opp]
            p = [out_pts[v] for v in quad]
            want = _add(_newell([out_pts[v] for v in tri]), _newell([out_pts[v] for v in other]))
            if _face_ok(p) and _dot(_newell(p), want) > 0.0:
                polys[i][0] = quad
                gone[j] = True
                break
    ind, fl, origin = [0], [], []
    triangles_left = 0
    for i, (pv, f, frm) in enumerate(polys):
        if gone[i]:
            continue
        if frm and len(pv) == 3:
            triangles_left += 1
        fl += pv
        ind.append(len(fl))
        origin.append(f)
    yield

    # 7. Per-side UVs at seam vertices (side A = left of the chain direction).
    corner_uv = np.full(2 * len(fl), np.nan)
    corner_side = np.full(len(fl), -1, np.int64)
    for f in range(len(ind) - 1):
        s, e = ind[f], ind[f + 1]
        fp = [out_pts[v] for v in fl[s:e]]
        nrm = _newell(fp)
        cen = (0.0, 0.0, 0.0)
        for p in fp:
            cen = _add(cen, p)
        cen = _scale(cen, 1.0 / len(fp))
        for h in range(s, e):
            v = fl[h]
            if tag_c[v] < 0:
                continue
            c, prm = tag_c[v], tag_p[v]
            x = out_pts[v]
            if not chs[c].closed and (prm == 0.0 or prm == chs[c].end()):
                cid = chs[c].start_id if prm == 0.0 else chs[c].end_id
                best = (math.inf, c, prm)
                for cc in corner_map[cid]:
                    p0 = 0.0 if chs[cc].start_id == cid else chs[cc].end()
                    d2, _ = chs[cc].closest(0, len(chs[cc].pts) - 1, cen)
                    if d2 < best[0]:
                        best = (d2, cc, p0)
                c, prm = best[1], best[2]
            ch = chs[c]
            side_b = _dot(_cross(ch.tangent(prm), _sub(cen, x)), nrm) <= 0.0
            u = ch.uv(prm, side_b)
            corner_uv[2 * h], corner_uv[2 * h + 1] = u
            corner_side[h] = int(side_b)
        if f & 63 == 63:
            yield
    st = np.array(state, np.int64)
    out.update(done=not stop_hit, points=np.array(out_pts, np.float64).ravel(), indptr=np.array(ind, np.int64), flat=np.array(fl, np.int64),
               origin=np.array(origin, np.int64), face_state=st, vchain=np.array(tag_c, np.int64), vparam=np.array(tag_p, np.float64),
               corner_uv=corner_uv, corner_side=corner_side, faces_split=int(np.sum(st == SPLIT)), extra_faces=len(ind) - 1 - nf,
               triangles_left=triangles_left, faces_left=int(np.sum(st == LEFT)), faces_cannot=int(np.sum(st == CANNOT)),
               faces_bake=int(np.sum(st == BAKE)), spent=spent, wanted=wanted)


def split(points, indptr, flat, seams: dict, xyz, vchain=(), vparam=(), seen_face=(), budget: float = -1.0, use_fast: bool = True) -> dict:
    """Split leftovers in one call (the faster callable when installed and `use_fast`, else the twin)."""

    stage = Split(points, indptr, flat, seams, xyz, vchain, vparam, seen_face, budget, use_fast=use_fast)
    while not stage.done:
        stage.step()
    return stage.result


# ---------------------------------------------------------------- sliced stages


class _Stage:
    """`step()` does about 2 ms of work (or polls the faster job); `done` turns True at the end and `result` holds the
    same dict the faster callable returns. `stop()` ends it early (`result["done"]` False)."""

    def __init__(self, job, steps):
        self.done, self.result, self.numbers = False, None, {"estimate": True}
        self._stopped = False
        self._job = job
        self._out = {}
        self._steps = None if job is not None else steps

    def _is_stopped(self) -> bool:
        return self._stopped

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
                self.result = self._out
                self._finish()
                return

    def _finish(self) -> None:
        self.done = True


class Snap(_Stage):
    """Seam snap as a sliced stage (numbers: moved, held, accepted / refused chains)."""

    def __init__(self, points, indptr, flat, seams: dict, xyz, max_move=(), vchain=(), vparam=(), rings: int = RELAX_RINGS, use_fast: bool = True):
        fast = FAST.get("snap") if use_fast else None
        job = fast(points, indptr, flat, seams, xyz, max_move, vchain, vparam, rings) if fast is not None else None
        super().__init__(job, None)
        if job is None:
            self._steps = _snap_steps(points, indptr, flat, seams, xyz, max_move, vchain, vparam, rings, self._out, self._is_stopped)

    def _finish(self) -> None:
        self.done = True
        r = self.result
        st = np.asarray(r["state"])
        self.numbers = dict(estimate=False, done=bool(r["done"]), moved=int(r["moved"]), held=int(np.sum(r["held"])),
                            accepted=int(np.sum((st == ACCEPTED) | (st == PARTLY_HELD))), refused=int(np.sum(st == REFUSED)))


class Split(_Stage):
    """Split leftovers as a sliced stage (numbers: faces split, triangles left, faces left for baking)."""

    def __init__(self, points, indptr, flat, seams: dict, xyz, vchain=(), vparam=(), seen_face=(), budget: float = -1.0, use_fast: bool = True):
        fast = FAST.get("split") if use_fast else None
        job = fast(points, indptr, flat, seams, xyz, vchain, vparam, seen_face, budget) if fast is not None else None
        super().__init__(job, None)
        if job is None:
            self._steps = _split_steps(points, indptr, flat, seams, xyz, vchain, vparam, seen_face, float(budget), self._out, self._is_stopped)

    def _finish(self) -> None:
        self.done = True
        r = self.result
        self.numbers = dict(estimate=False, done=bool(r["done"]), faces_split=int(r["faces_split"]), extra_faces=int(r["extra_faces"]),
                            triangles_left=int(r["triangles_left"]), faces_left=int(r["faces_left"]),
                            faces_for_baking=int(r["faces_cannot"]) + int(r["faces_bake"]))
