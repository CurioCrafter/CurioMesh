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
tags and per-corner UVs on the seams (`with_seam_uvs` lays them over a per-side transfer). `part_switches` adds
chains where the nearest source piece changes between unconnected pieces, which have no UV seam between them; call it
on the snapped mesh and hand its seam dict (and `chain_xyz` of it) to `split` only.
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from .topology import _quad_points_valid
from .uvseams import _closest_bary_many, _edges, closest_bary

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
IGNORE, SOFT = 0, 2
OWNER_EPS_DIAG, OWNER_EPS_EDGE = 1e-4, 0.05  # the UV Check's owner ball (SPEC 9.1): nearest + max(this x diagonal, this x median source edge)
SWITCH_STEP = 0.5  # part_switches samples each result edge at most this many source edges apart, so thin strips are found
SWITCH_BY = "side"  # part_switches: "side" = where the nearest side region changes, "part" = only between unconnected pieces
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
    extra = seams.get("extra_xyz")  # welded ids after the source's: the `part_switches` chain vertices
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


def _parts_steps(fl, ip, weld, nw, out: dict):
    """Generator: `out["part"]` = per source face its part (faces joined across shared welded edges; union-find)."""

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

    fo = face_of[order].tolist()
    yield
    for n, i in enumerate(np.flatnonzero(ks[1:] == ks[:-1]).tolist()):
        x, y = find(fo[i]), find(fo[i + 1])
        if x != y:
            lab[max(x, y)] = min(x, y)
        if n % 4096 == 4095:
            yield
    part = np.empty(nf, np.int64)
    for f in range(nf):
        part[f] = find(f)
        if f % 8192 == 8191:
            yield
    out["part"] = part


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


def _uv_at(src_P, src_ip, src_fl, src_uv, face, at):
    """Source UV at closest points `at` (N, 3) on source polygons `face` (the fan triangle nearest to each point)."""

    out = np.zeros((len(face), 2))
    best = np.full(len(face), np.inf)
    counts = src_ip[face + 1] - src_ip[face]
    for j in range(1, int(counts.max()) - 1 if len(face) else 0):
        i = np.flatnonzero(counts > j + 1)
        c0 = src_ip[face[i]]
        c1, c2 = c0 + j, c0 + j + 1
        a, b, c = src_P[src_fl[c0]], src_P[src_fl[c1]], src_P[src_fl[c2]]
        w = _closest_bary_many(at[i], a, b, c)  # elementwise: one point per triangle
        q = w[:, :1] * a + w[:, 1:2] * b + w[:, 2:] * c
        d = np.sqrt(((q - at[i]) ** 2).sum(1))
        better = d < best[i]
        k = i[better]
        best[k] = d[better]
        out[k] = (w[better, :1] * src_uv[c0[better]] + w[better, 1:2] * src_uv[c1[better]]) + w[better, 2:] * src_uv[c2[better]]
    return out


def part_switches(seams: dict, src_points, src_indptr, src_flat, src_uv, points, indptr, flat, nearest, texture_px: float = 1024.0,
                  steps: int = 16, by: str = SWITCH_BY, only=None) -> dict:
    """`part_switches_steps` in one call: the seam dict with the part switches added."""

    out = {}
    for _ in part_switches_steps(seams, src_points, src_indptr, src_flat, src_uv, points, indptr, flat, nearest, out, texture_px, steps, by, only):
        pass
    return out["seams"]


def part_switches_steps(seams: dict, src_points, src_indptr, src_flat, src_uv, points, indptr, flat, nearest, out: dict,
                        texture_px: float = 1024.0, steps: int = 16, by: str = SWITCH_BY, only=None):
    """Generator (yields between short pieces; `out["seams"]` at the end): the seam dict with **part switches** added as
    Soft chains: on the result mesh, the lines where the nearest source piece changes between two unconnected pieces
    (a bolt on a plate, a cap pushed into a body).

    Unconnected pieces share no edge, so the UV seams have no chain there, yet a result face spanning both takes texture
    from two places (on the bench models most faces left across seams were such faces). Each result edge whose two
    ends are nearest different pieces gets the point where the nearest piece changes (bisection along the edge, `steps`
    halvings), and the points of a face's two such edges are joined, so the chains run on the result surface itself
    and the split cuts where the texture changes owner. uv_a / uv_b are the two pieces' UVs there (not oriented: they
    only give the gap, which orders the split's budget; the corners keep the per-side transfer's UVs). New chain
    vertices get welded ids after the source's (`chain_xyz` reads them from `extra_xyz`). `nearest(points) -> (closest
    (N, 3), polygon, distance)` is the caller's lookup on the source polygons (the add-on's face tree). The source's
    parts are kept in `seams["face_part"]` for the next call. Our own rule; bisection is textbook."""

    sP = np.asarray(src_points, np.float64).reshape(-1, 3)
    sip, sfl = np.asarray(src_indptr, np.int64), np.asarray(src_flat, np.int64)
    suv = np.asarray(src_uv, np.float64).reshape(-1, 2)
    P = np.asarray(points, np.float64).reshape(-1, 3)
    res = dict(seams)
    out["seams"] = res
    mesh = _Mesh(P, indptr, flat, lazy=True)
    yield from mesh.build(P, indptr, flat)
    if mesh.nf == 0 or len(sip) < 3:
        return
    if by == "side" and "region" not in seams:
        by = "part"  # a seam dict without side regions: only the unconnected pieces are known
    if by == "side":  # every change of side region (UV island joined across Ignore/Shift chains), parts included
        part = np.asarray(seams["region"], np.int64)
    else:
        if "face_part" not in seams:
            got = {}
            yield from _parts_steps(sfl, sip, np.asarray(seams["weld"], np.int64), int(seams["n_welded"]), got)
            seams["face_part"] = got["part"]
        part = np.asarray(seams["face_part"], np.int64)
    if part.min() == part.max():
        return
    vpart = np.empty(len(P), np.int64)
    for i in range(0, len(P), 4096):
        vpart[i:i + 4096] = part[np.asarray(nearest(P[i:i + 4096])[1], np.int64)]
        yield
    # Every switch along an edge, not only one per edge: a strip of another side thinner than the edge (a trim band, a
    # rim) has two. Each edge is sampled at steps of at most SWITCH_STEP source edges and every step whose two ends
    # are nearest different sides is bisected.
    sizes = np.diff(sip)
    s_next = np.arange(len(sfl)) + 1
    s_next[sip[1:][sizes > 0] - 1] = sip[:-1][sizes > 0]
    src_edge = float(np.median(np.linalg.norm(sP[sfl[s_next]] - sP[sfl], axis=1))) if len(sfl) else 1.0
    PA, PB = P[mesh.ea], P[mesh.eb]
    elen = np.linalg.norm(PB - PA, axis=1)
    pieces = np.maximum(1, np.ceil(elen / max(SWITCH_STEP * src_edge, 1e-30))).astype(np.int64)
    inner = pieces - 1  # samples strictly inside each edge
    s_edge = np.repeat(np.arange(len(elen)), inner)
    s_t = (np.arange(len(s_edge)) - np.repeat(np.cumsum(inner) - inner, inner) + 1) / np.repeat(pieces, inner).astype(np.float64)
    s_lab = np.empty(len(s_edge), np.int64)
    s_at = PA[s_edge] + s_t[:, None] * (PB - PA)[s_edge]
    for i in range(0, len(s_edge), 4096):
        s_lab[i:i + 4096] = part[np.asarray(nearest(s_at[i:i + 4096])[1], np.int64)]
        yield
    # samples per edge in order: t = 0 (end a), the inner ones, t = 1 (end b)
    starts = np.cumsum(pieces + 1) - (pieces + 1)
    n_all = int((pieces + 1).sum())
    all_e = np.repeat(np.arange(len(elen)), pieces + 1)
    all_t = np.zeros(n_all)
    all_l = np.empty(n_all, np.int64)
    all_l[starts] = vpart[mesh.ea]
    all_l[starts + pieces] = vpart[mesh.eb]
    all_t[starts + pieces] = 1.0
    inner_pos = np.repeat(starts, inner) + (np.arange(len(s_edge)) - np.repeat(np.cumsum(inner) - inner, inner) + 1)
    all_l[inner_pos], all_t[inner_pos] = s_lab, s_t
    step = np.flatnonzero((all_l[:-1] != all_l[1:]) & (all_e[:-1] == all_e[1:]))
    if not len(step):
        return
    cross = all_e[step]  # edge of each switch (an edge may hold several, in order along it)
    pa, pb = PA[cross], PB[cross]
    first = all_l[step]
    lo, hi = all_t[step].copy(), all_t[step + 1].copy()
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        same = part[np.asarray(nearest(pa + mid[:, None] * (pb - pa))[1], np.int64)] == first
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        yield
    tx = 0.5 * (lo + hi)
    x = pa + tx[:, None] * (pb - pa)
    at_a, fa, _ = nearest(pa + lo[:, None] * (pb - pa))
    at_b, fb, _ = nearest(pa + hi[:, None] * (pb - pa))
    fa, fb = np.asarray(fa, np.int64), np.asarray(fb, np.int64)
    yield
    uva = _uv_at(sP, sip, sfl, suv, fa, np.asarray(at_a).reshape(-1, 3))
    yield
    uvb = _uv_at(sP, sip, sfl, suv, fb, np.asarray(at_b).reshape(-1, 3))
    yield
    pair = [tuple(r) for r in np.sort(np.stack([part[fa], part[fb]], 1), 1).tolist()]
    on_edge = {}
    for k, e in enumerate(cross.tolist()):
        on_edge.setdefault(e, []).append(k)
    edge_of = {(x_, y_): e for e, (x_, y_) in enumerate(zip(mesh.ea.tolist(), mesh.eb.tolist()))}
    # Per face: the switches on its edges with the same side pair are joined when that pair shows exactly twice; the
    # rest (three sides meeting inside the face, or a pair showing 4 times) are joined to one new point inside the face
    # (the mean of theirs), which the split cuts as a fan.
    adj = {}
    face_cut = {}  # face -> (switch pairs joined across it, (junction, its switches) or None)
    junction = []  # (position, gap px)
    junction_face = []
    nx = len(x)
    gap = np.sqrt(((uva - uvb) ** 2).sum(1)) * texture_px
    for f in range(mesh.nf):
        if f % 1024 == 1023:
            yield
        if only is not None and not only[f]:
            continue
        fv = mesh.face(f)
        n = len(fv)
        ks = []
        for i in range(n):
            e = edge_of.get((min(fv[i], fv[(i + 1) % n]), max(fv[i], fv[(i + 1) % n])))
            ks.extend(on_edge.get(e, ()))
        if len(ks) < 2:
            continue
        by_pair = {}
        for k in ks:
            by_pair.setdefault(pair[k], []).append(k)
        loose = []
        links = []
        for got_ in by_pair.values():
            if len(got_) == 2:
                links.append((got_[0], got_[1]))
            elif len(by_pair) == 1 and len(got_) % 2 == 0:
                # two sides only, crossing back and forth (a thin strip): in order round the face the sides alternate,
                # so consecutive switches pair up (1-2, 3-4, ...) and each line cuts off one stretch
                ring = []
                for i in range(n):
                    a_, b_ = fv[i], fv[(i + 1) % n]
                    e = edge_of.get((min(a_, b_), max(a_, b_)))
                    fwd = int(mesh.ea[e]) == a_ if e is not None else True
                    ring += sorted(on_edge.get(e, ()), key=lambda k: tx[k] if fwd else -tx[k])
                m = len(ring)
                best = None
                for off in (0, 1):  # either pairing splits the sides; a pair on one edge cuts nothing, so take the other
                    prs = [(ring[(i + off) % m], ring[(i + 1 + off) % m]) for i in range(0, m, 2)]
                    key = (sum(cross[p] == cross[q] for p, q in prs), sum(float(np.linalg.norm(x[p] - x[q])) for p, q in prs))
                    if best is None or key < best[0]:
                        best = (key, prs)
                links += best[1]
            else:
                loose.extend(got_)
        if len(loose) == 2:  # two lone switches: one line between them
            links.append((loose[0], loose[1]))
            loose = []
        for k1, k2 in links:
            adj.setdefault(k1, set()).add(k2)
            adj.setdefault(k2, set()).add(k1)
        fan = None
        if len(loose) >= 3:
            j = nx + len(junction)
            junction.append((x[loose].mean(0), float(gap[loose].max())))
            junction_face.append(f)
            for k in loose:
                adj.setdefault(k, set()).add(j)
                adj.setdefault(j, set()).add(k)
            fan = (j, loose)
        if links or fan:
            face_cut[f] = (links, fan)
    runs = _walk_runs(adj)
    if not runs:
        return
    if junction:  # where the sides meet inside each junction face, not the mean of its switches
        got = {}
        yield from _meeting_points_steps(P, mesh, junction_face, vpart, part, nearest, got)
        for i, q in enumerate(got["at"]):
            if q is not None:
                junction[i] = (q, junction[i][1])
        x = np.r_[x, np.array([q for q, _ in junction])]
        gap = np.r_[gap, np.array([g for _, g in junction])]
        uva = np.r_[uva, np.zeros((len(junction), 2))]
        uvb = np.r_[uvb, np.zeros((len(junction), 2))]
        for i, (_, g) in enumerate(junction):  # a junction's UVs only carry its gap (the split orders its budget by it)
            uvb[nx + i, 0] = g / texture_px
    base = int(seams["n_welded"])
    order = np.concatenate([np.asarray(r, np.int64) for r in runs])
    old_ip = np.asarray(seams["run_indptr"], np.int64)
    cls = np.asarray(seams["class"])
    old_extra = seams.get("extra_xyz")
    res["run_indptr"] = np.r_[old_ip, np.cumsum([len(r) for r in runs]).astype(np.int64) + int(old_ip[-1])]
    res["run_vertex"] = np.r_[np.asarray(seams["run_vertex"], np.int64), base + order]
    res["uv_a"] = np.r_[np.asarray(seams["uv_a"], np.float64).ravel(), uva[order].ravel()]
    res["uv_b"] = np.r_[np.asarray(seams["uv_b"], np.float64).ravel(), uvb[order].ravel()]
    res["gap_px"] = np.r_[np.asarray(seams["gap_px"], np.float64), gap[order]]
    new_cls = cls.copy()
    if by == "side":  # the switches stand for every Soft chain here: the split cuts where the owner changes on the result
        new_cls[new_cls == SOFT] = IGNORE
    res["class"] = np.r_[new_cls, np.full(len(runs), SOFT, cls.dtype)]
    res["n_welded"] = base + len(x)
    res["extra_xyz"] = x if old_extra is None else np.r_[np.asarray(old_extra, np.float64).reshape(-1, 3), x]
    res["part_switches"] = len(runs)
    res["part_switch_first"] = len(cls)
    if by == "side":
        res["switch_plan"] = _switch_plan(runs, face_cut, cross, tx, x, nx, base, len(cls), mesh)



def face_sides(points, indptr, flat, label, on_cut) -> np.ndarray:
    """Per result face the side it lies on: the label (nearest source side region per vertex) its corners agree on,
    leaving out corners on a cut (`on_cut` per vertex); -1 when they disagree or every corner is on a cut."""

    ip, fl = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
    lab, cut = np.asarray(label, np.int64), np.asarray(on_cut, bool)
    nf = len(ip) - 1
    out = np.full(nf, -1, np.int64)
    cl = np.where(cut[fl], -1, lab[fl])
    face_of = np.repeat(np.arange(nf), np.diff(ip))
    hi = np.full(nf, -1, np.int64)
    np.maximum.at(hi, face_of, cl)
    lo = np.full(nf, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(lo, face_of, np.where(cl < 0, np.iinfo(np.int64).max, cl))
    ok = (hi >= 0) & (lo == hi)
    out[ok] = hi[ok]
    return out


def _ids_w(sP, sip, sfl, face, at):
    """(source corner ids (N, 3), barycentric weights (N, 3)) of the closest points `at` on polygons `face` (their
    nearest fan triangle), the lookup's own form."""

    n = len(face)
    ids, w = np.zeros((n, 3), np.int64), np.zeros((n, 3))
    best = np.full(n, np.inf)
    counts = sip[face + 1] - sip[face]
    for j in range(1, int(counts.max()) - 1 if n else 0):
        i = np.flatnonzero(counts > j + 1)
        c0 = sip[face[i]]
        a, b, c = sP[sfl[c0]], sP[sfl[c0 + j]], sP[sfl[c0 + j + 1]]
        ww = _closest_bary_many(at[i], a, b, c)
        q = ww[:, :1] * a + ww[:, 1:2] * b + ww[:, 2:] * c
        d = np.sqrt(((q - at[i]) ** 2).sum(1))
        better = d < best[i]
        k = i[better]
        best[k] = d[better]
        ids[k] = np.stack([c0, c0 + j, c0 + j + 1], 1)[better]
        w[k] = ww[better]
    return ids, w


def _wind_of(sP, sip, sfl, src_uv, faces) -> np.ndarray:
    """Sign of the UV area of source polygons `faces` (shoelace over their corners; 0 = no UV area)."""

    uv = np.asarray(src_uv, np.float64).reshape(-1, 2)
    out = np.zeros(len(faces))
    for i, h in enumerate(np.asarray(faces, np.int64).tolist()):
        u = uv[sip[h]:sip[h + 1]]
        out[i] = np.sign(np.sum(u[:, 0] * np.roll(u[:, 1], -1) - np.roll(u[:, 0], -1) * u[:, 1]))
    return out


def _src_normal(sP, sip, sfl, faces) -> np.ndarray:
    """Newell normals (not unit) of source polygons `faces`."""

    out = np.zeros((len(faces), 3))
    for i, h in enumerate(np.asarray(faces, np.int64).tolist()):
        c = sP[sfl[sip[h]:sip[h + 1]]]
        out[i] = np.cross(c, np.roll(c, -1, 0)).sum(0)
    return out


class RegionTrees:
    """Closest-point lookups over the source polygons of one side region at a time, built when first asked (the
    native face tree over that region's polygons; `tree_of(points, FlatFaces)`). `eps` is the UV Check's owner-ball
    reach (SPEC 9.1) for this source."""

    def __init__(self, src_points, src_indptr, src_flat, src_region, tree_of):
        self.sP = np.asarray(src_points, np.float64).reshape(-1, 3)
        self.sip, self.sfl = np.asarray(src_indptr, np.int64), np.asarray(src_flat, np.int64)
        self.reg = np.asarray(src_region, np.int64)
        self.tree_of = tree_of
        self.order = np.argsort(self.reg, kind="stable")
        self.starts = np.searchsorted(self.reg[self.order], np.arange(int(self.reg.max()) + 2 if len(self.reg) else 1))
        self.trees = {}
        sizes = np.diff(self.sip)
        nxt = np.arange(len(self.sfl)) + 1
        nxt[self.sip[1:][sizes > 0] - 1] = self.sip[:-1][sizes > 0]
        diag = float(np.linalg.norm(self.sP.max(0) - self.sP.min(0))) if len(self.sP) else 0.0
        edge = float(np.median(np.linalg.norm(self.sP[self.sfl[nxt]] - self.sP[self.sfl], axis=1))) if len(self.sfl) else 0.0
        self.eps = max(OWNER_EPS_DIAG * diag, OWNER_EPS_EDGE * edge)

    def nearest(self, r, q):
        """(closest points, source polygon ids, distances) on side `r`; None when the side has no polygons."""

        from .types import FlatFaces

        if r not in self.trees:
            own = self.order[self.starts[r]:self.starts[r + 1]] if 0 <= r < len(self.starts) - 1 else np.zeros(0, np.int64)
            tree = None
            if len(own):
                sub_ip = np.r_[0, np.cumsum(self.sip[own + 1] - self.sip[own])].astype(np.int64)
                sub_fl = np.concatenate([self.sfl[self.sip[g]:self.sip[g + 1]] for g in own.tolist()])
                tree = self.tree_of(self.sP, FlatFaces(sub_ip, sub_fl))
            self.trees[r] = (own, tree)
        own, tree = self.trees[r]
        if tree is None:
            return None
        at, g, d = tree.nearest(np.asarray(q, np.float64).reshape(-1, 3))
        return np.asarray(at, np.float64).reshape(-1, 3), own[np.asarray(g, np.int64)], np.asarray(d, np.float64)


def ball_sides_steps(look_region, points, indptr, flat, label, d0, trees: RegionTrees, out: dict):
    """Generator: out["sides"] = per result face a side region whose owner ball (the UV Check's rule, SPEC 9.1: within
    the corner's nearest distance `d0` plus the ball's reach) holds every corner of it, -1 when none does. Tried in
    order: the lookup's own side, then the sides its corners lie nearest (`label`). A face whose corners all lie nearest
    the lookup's side keeps it without a test. Our own rule."""

    ip, fl = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
    P = np.asarray(points, np.float64).reshape(-1, 3)
    lab, d0 = np.asarray(label, np.int64), np.asarray(d0, np.float64)
    lr = np.asarray(look_region, np.int64)
    nf = len(ip) - 1
    sizes = np.diff(ip)
    face_of = np.repeat(np.arange(nf), sizes)
    agree = np.bincount(face_of, lab[fl] == lr[face_of], nf) == sizes
    sides = np.where(agree, lr, -1)
    cand = {}
    for f in np.flatnonzero(~agree & (sizes > 0)).tolist():
        cs = [int(lr[f])] if lr[f] >= 0 else []
        for v in fl[ip[f]:ip[f + 1]].tolist():
            if int(lab[v]) not in cs:
                cs.append(int(lab[v]))
        cand[f] = cs
    yield
    by_r = {}
    for f, cs in cand.items():
        for r in cs:
            by_r.setdefault(r, []).append(f)
    ok = {}
    for r, fs in by_r.items():
        corners = np.concatenate([np.arange(ip[f], ip[f + 1]) for f in fs])
        got = trees.nearest(r, P[fl[corners]])
        if got is None:
            continue
        inside = got[2] <= d0[fl[corners]] + trees.eps
        k = 0
        for f in fs:
            n = int(sizes[f])
            if inside[k:k + n].all():
                ok.setdefault(f, set()).add(r)
            k += n
        yield
    for f, cs in cand.items():
        good = ok.get(f, ())
        sides[f] = next((r for r in cs if r in good), -1)
    out["sides"] = sides


def side_reread_steps(look: dict, sides, src_points, src_indptr, src_flat, src_region, points, indptr, flat, tree_of, out: dict, trees=None,
                      src_uv=None):
    """Generator: re-reads, in a per-side lookup `look` (`uvseams` form), the faces whose side region the lookup picked
    against `sides` (from `ball_sides_steps` or `face_sides`): every corner and the reference from the nearest point of
    that side. The lookup votes with the face centre, which on a face cut along a side switch, or on a coarse face
    beside a crease, can lie nearest the other side; the corners then read the wrong side's texture. `tree_of(points,
    FlatFaces) -> tree with .nearest` builds a lookup over some source polygons (the native face tree); `trees` (a
    RegionTrees) reuses the lookups already built. With `src_uv` (the active layer, per source corner) a face is left as
    the lookup read it when its new reads would wind two ways (a crease where the side's texture turns over): the fold
    fix and the halving judge those. Fills out["reread"] (faces changed). Our own rule."""

    sP = np.asarray(src_points, np.float64).reshape(-1, 3)
    sip, sfl = np.asarray(src_indptr, np.int64), np.asarray(src_flat, np.int64)
    P = np.asarray(points, np.float64).reshape(-1, 3)
    ip, fl = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
    sides = np.asarray(sides, np.int64)
    todo = np.flatnonzero((sides >= 0) & (sides != np.asarray(look["region"], np.int64)) & (np.asarray(look["ref_face"]) >= 0))
    out["reread"] = 0
    if not len(todo):
        return
    trees = trees or RegionTrees(sP, sip, sfl, src_region, tree_of)
    done = 0
    for r in np.unique(sides[todo]).tolist():
        fs = todo[sides[todo] == r]
        corner = np.concatenate([np.arange(ip[f], ip[f + 1]) for f in fs.tolist()])
        q = P[fl[corner]]
        cen = np.array([P[fl[ip[f]:ip[f + 1]]].mean(0) for f in fs.tolist()])
        got = trees.nearest(r, np.r_[q, cen])
        if got is None:
            continue
        at, g, _d = got
        nq = len(q)
        if src_uv is not None:  # keep only faces whose reads all show the texture the same way round
            wind = _wind_of(sP, sip, sfl, src_uv, g)
            nf_r = np.array([np.cross(P[fl[ip[f]:ip[f + 1]]], np.roll(P[fl[ip[f]:ip[f + 1]]], -1, 0)).sum(0) for f in fs.tolist()])
            sizes_r = np.diff(ip)[fs]
            face_n = np.repeat(nf_r, sizes_r, axis=0)
            sn = _src_normal(sP, sip, sfl, g)
            show = wind * np.where((np.r_[face_n, nf_r] * sn).sum(1) >= 0.0, 1.0, -1.0)
            cshow, rshow = show[:nq], show[nq:]
            ok_f = np.array([bool(np.all(cshow[k0:k0 + n_] == rshow[i])) and rshow[i] != 0.0
                             for i, (k0, n_) in enumerate(zip(np.r_[0, np.cumsum(sizes_r)[:-1]].tolist(), sizes_r.tolist()))])
            if not ok_f.all():
                keep_c = np.repeat(ok_f, sizes_r)
                fs, corner, q = fs[ok_f], corner[keep_c], q[keep_c]
                at, g = np.r_[at[:nq][keep_c], at[nq:][ok_f]], np.r_[g[:nq][keep_c], g[nq:][ok_f]]
                nq = len(q)
                if not len(fs):
                    continue
        ids, w = _ids_w(sP, sip, sfl, g, at)
        look["ids"].reshape(-1, 3)[corner] = ids[:nq]
        look["w"].reshape(-1, 3)[corner] = w[:nq]
        look["shift"][corner] = False
        look["slid"][corner] = False
        look["ref_ids"].reshape(-1, 3)[fs] = ids[nq:]
        look["ref_w"].reshape(-1, 3)[fs] = w[nq:]
        rf = g[nq:]
        look["ref_face"][fs] = rf
        look["region"][fs] = r
        # orientation against the reference (its UV winding is applied per layer): turned when it faces away
        for f, h in zip(fs.tolist(), rf.tolist()):
            cs = sfl[sip[h]:sip[h + 1]]
            pf = P[fl[ip[f]:ip[f + 1]]]
            ns = np.cross(sP[cs], sP[np.roll(cs, -1)]).sum(0)
            nr = np.cross(pf, np.roll(pf, -1, 0)).sum(0)
            look["orient"][f] = 1.0 if float(ns @ nr) >= 0.0 else -1.0
        done += len(fs)
        yield
    out["reread"] = done

MEET_LEVELS = 14  # halvings of the triangle that holds the meeting point of three sides (a face / 2^14 is well below a source edge)


def _meeting_points_steps(P, mesh, faces, vlabel, part, nearest, out: dict, levels: int = MEET_LEVELS):
    """Generator: per face in `faces`, the point where three side regions meet on it (out["at"], None when no fan
    triangle of the face shows three sides at its corners). The fan triangle with three different nearest sides at its
    corners holds the meeting point (the sides split the plane like a Sperner labelling); it is cut into four at its
    edge midpoints, `levels` times, keeping a quarter that still shows three sides. Our own use of the textbook
    subdivision."""

    n = len(faces)
    tri = np.zeros((n, 3, 3))
    lab = np.full((n, 3), -1, np.int64)
    ok = np.zeros(n, bool)
    for i, f in enumerate(faces):
        fv = mesh.face(f)
        best = -1
        for k in range(1, len(fv) - 1):
            t = (fv[0], fv[k], fv[k + 1])
            d = len({int(vlabel[v]) for v in t})
            if d > best:
                best = d
                tri[i] = P[list(t)]
                lab[i] = [int(vlabel[v]) for v in t]
        ok[i] = best >= 3
    yield
    idx = np.flatnonzero(ok)
    for _ in range(levels if len(idx) else 0):
        T, L = tri[idx], lab[idx]
        mids = np.stack([(T[:, 0] + T[:, 1]) / 2, (T[:, 1] + T[:, 2]) / 2, (T[:, 2] + T[:, 0]) / 2], 1)  # ab, bc, ca
        ml = part[np.asarray(nearest(mids.reshape(-1, 3))[1], np.int64)].reshape(-1, 3)
        kids_p = np.stack([np.stack([T[:, 0], mids[:, 0], mids[:, 2]], 1), np.stack([mids[:, 0], T[:, 1], mids[:, 1]], 1),
                           np.stack([mids[:, 2], mids[:, 1], T[:, 2]], 1), np.stack([mids[:, 0], mids[:, 1], mids[:, 2]], 1)], 1)
        kids_l = np.stack([np.stack([L[:, 0], ml[:, 0], ml[:, 2]], 1), np.stack([ml[:, 0], L[:, 1], ml[:, 1]], 1),
                           np.stack([ml[:, 2], ml[:, 1], L[:, 2]], 1), np.stack([ml[:, 0], ml[:, 1], ml[:, 2]], 1)], 1)
        three = (kids_l[:, :, 0] != kids_l[:, :, 1]) & (kids_l[:, :, 1] != kids_l[:, :, 2]) & (kids_l[:, :, 0] != kids_l[:, :, 2])
        has = three.any(1)
        pick = np.argmax(three, 1)
        r = np.arange(len(idx))
        tri[idx[has]] = kids_p[r[has], pick[has]]
        lab[idx[has]] = kids_l[r[has], pick[has]]
        idx = idx[has]  # a triangle with no three-sided quarter stops where it is
        yield
    out["at"] = [tri[i].mean(0) if ok[i] else None for i in range(n)]




SWITCH_SNAP_AGAIN = True  # a vertex may slide again in a later pass; passes stop when one mends no more faces than the last


def switch_snap_steps(points, indptr, flat, label_of, max_move, only, out: dict, passes: int | None = None, steps: int = 16):
    """Generator (zero faces, SPEC 4.5's idea on the side lines): moves result vertices onto the line where the nearest
    source side changes, so a face whose corners lie on two sides gets one side every corner touches. Per face to mend
    (`only`, per face; None = every face whose corners' sides differ) the side most of its corners lie on is kept; each
    other corner moves along its edge towards a kept-side neighbour of the face to the first point where the side
    changes, when that change is straight into the kept side, the move is within `max_move` (per vertex; the caller's
    camera check or share of an edge) and no face around the vertex flips or folds (the snap's own guard). Smallest
    moves first; a vertex moves once (it then touches both sides); passes repeat while one moves something (a face
    with two corners over the line needs both), `passes` None = no other end. `label_of(points (N, 3)) -> side per point` is the caller's lookup.
    Fills out["points"] (flat), out["moved"] (vertex ids). Our own rule; bisection is textbook."""

    P = np.asarray(points, np.float64).reshape(-1, 3).copy()
    ip, fl = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
    mesh = _Mesh(P, ip, fl, lazy=True)
    yield from mesh.build(P, ip, fl)
    limit = np.asarray(max_move, np.float64).ravel()
    moved_all = []
    nf = len(ip) - 1
    pass_no = 0
    last_mixed = None
    moved_once = set()  # a vertex moved onto a side line stays there: it touches both sides, so every pass ends
    while passes is None or pass_no < passes:
        pass_no += 1
        lab = np.asarray(label_of(P), np.int64)
        yield
        face_of = np.repeat(np.arange(nf), np.diff(ip))
        wild = np.zeros(len(P), bool)  # a vertex slid onto a line touches both sides
        wild[np.asarray(moved_all, np.int64)] = True
        cl = np.where(wild[fl], -1, lab[fl])
        hi = np.full(nf, -1, np.int64)
        np.maximum.at(hi, face_of, cl)
        lo = np.full(nf, np.iinfo(np.int64).max, np.int64)
        np.minimum.at(lo, face_of, np.where(cl < 0, np.iinfo(np.int64).max, cl))
        mixed = (hi >= 0) & (lo < hi)
        if only is not None:
            mixed &= np.asarray(only, bool)
        n_mixed = int(mixed.sum())
        if SWITCH_SNAP_AGAIN and pass_no > 1 and n_mixed >= last_mixed:  # keep going only while a pass mends faces
            P = P_before
            moved_all = moved_before
            break
        last_mixed, P_before, moved_before = n_mixed, P.copy(), list(moved_all)
        cand = {}  # (v, w) -> kept side
        for f in range(nf):
            if only is not None and not only[f]:
                continue
            fv = fl[ip[f]:ip[f + 1]]
            fl_ = lab[fv]
            if (fl_ == fl_[0]).all():
                continue
            sides, counts = np.unique(fl_, return_counts=True)
            keep = int(sides[np.argmax(counts)])
            n = len(fv)
            for i in range(n):
                if fl_[i] == keep:
                    continue
                for j in ((i - 1) % n, (i + 1) % n):
                    if fl_[j] == keep:
                        cand[(int(fv[i]), int(fv[j]))] = keep
            if f % 2048 == 2047:
                yield
        if not cand:
            break
        keys = list(cand)
        vv = np.array([k[0] for k in keys], np.int64)
        ww = np.array([k[1] for k in keys], np.int64)
        want = np.array([cand[k] for k in keys], np.int64)
        first = lab[vv]
        pa, pb = P[vv], P[ww]
        lo, hi = np.zeros(len(keys)), np.ones(len(keys))
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            same = np.asarray(label_of(pa + mid[:, None] * (pb - pa)), np.int64) == first
            lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
            yield
        after = np.asarray(label_of(pa + hi[:, None] * (pb - pa)), np.int64)
        to = pa + (0.5 * (lo + hi))[:, None] * (pb - pa)
        dist = np.linalg.norm(to - pa, axis=1)
        good = (after == want) & (dist <= limit[vv])
        pos = [tuple(p) for p in P.tolist()]
        done = set()
        for c, k in enumerate(np.flatnonzero(good)[np.argsort(dist[good], kind="stable")].tolist()):
            v = int(vv[k])
            if v in done or (v in moved_once and not SWITCH_SNAP_AGAIN):
                continue
            t = tuple(to[k].tolist())
            if _move_ok(pos, mesh, v, t):
                pos[v] = t
                done.add(v)
            if c % 256 == 255:
                yield
        if not done:
            break
        P = np.array(pos, np.float64)
        moved_once |= done
        moved_all += sorted(done)
    out["points"] = P.ravel()
    out["moved"] = np.array(sorted(set(moved_all)), np.int64)


SPLIT_ORDER = "damage"  # the twin's budget order for chain cuts: "damage" (SPEC 4.7) or "per_face" (damage per face spent)
SWITCH_REUSE = 0.02  # a switch this close to an edge's end (share of the edge) uses that vertex instead of a new one


def _switch_plan(runs, face_cut, cross, tx, x, nx, base, first_c, mesh) -> dict:
    """The split's tables for the switches (`_split_steps` reads them instead of searching crossings): per face the
    cuts between its switches (chords) and the fan to a junction, as the split's nodes (vertex, edge point, corner)."""

    where = {}
    seg = {}
    for ri, r in enumerate(runs):
        c = first_c + ri
        for idx, k in enumerate(r):
            where.setdefault(k, (c, float(idx)))
        for idx in range(len(r) - 1):
            seg[(r[idx], r[idx + 1])] = (c, float(idx), float(idx + 1))
            seg[(r[idx + 1], r[idx])] = (c, float(idx + 1), float(idx))
    crosses, reuse_tags, corner_at = {}, [], {}
    node_of = {}
    for k, (c0, prm) in where.items():
        if k >= nx:
            node_of[k] = (_C, base + k, 0)
            continue
        e, t = int(cross[k]), float(tx[k])
        reuse = int(mesh.ea[e]) if t < SWITCH_REUSE else int(mesh.eb[e]) if t > 1.0 - SWITCH_REUSE else None
        if reuse is None:  # keyed by the point (-1 - k), not the chain: one line may cross an edge twice
            crosses[(e, -1 - k)] = (0.0, (t, prm, None, c0))
            node_of[k] = (_E, e, -1 - k)
        else:
            reuse_tags.append((reuse, c0, prm))
            node_of[k] = (_V, reuse, 0)
    faces = {}
    for f, (links, fan) in face_cut.items():
        chords = []
        for k1, k2 in links:
            if (k1, k2) not in seg:
                continue
            c, p1, p2 = seg[(k1, k2)]
            if node_of[k1] != node_of[k2]:
                chords.append((node_of[k1], node_of[k2], c, p1, p2))
        fn = None
        if fan is not None:
            j, ks = fan
            corner_at[base + j] = (f, tuple(float(v) for v in x[j]))
            spokes = []
            for k in ks:
                if (j, k) in seg and all(node_of[k] != sp[0] for sp in spokes):
                    c, _pj, pk = seg[(j, k)]
                    spokes.append((node_of[k], c, pk))
            fn = (node_of[j], spokes)
        faces[f] = (chords, fn)
    return dict(crosses=crosses, faces=faces, corner_at=corner_at, reuse_tags=reuse_tags)


def _plan_tables(plan_in: dict, active) -> tuple:
    """(crosses, fpts, plans, corner_at, reuse_tags) of `_split_steps` from a switch plan, keeping active chains only."""

    plans, fpts = {}, {}
    for f, (chords, fan) in plan_in["faces"].items():
        ch = [x for x in chords if active[x[2]]]
        fn = None
        if fan is not None:
            sp = [x for x in fan[1] if active[x[1]]]
            if len(sp) >= 2:
                fn = (fan[0], sp)
        if not ch and fn is None:
            continue
        plans[f] = {"chords": ch, "fan": fn, "cannot": False, "whole": False}
        pts = [(c, a, pa) for a, b, c, pa, pb in ch] + [(c, b, pb) for a, b, c, pa, pb in ch]
        if fn is not None:
            pts += [(c, nd, pr) for nd, c, pr in fn[1]] + [(fn[1][0][1], fn[0], 0.0)]
        fpts[f] = pts
    return dict(plan_in["crosses"]), fpts, plans, dict(plan_in["corner_at"]), list(plan_in["reuse_tags"])


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
        if cls[c] != SOFT or len(ch.pts) < 2 or n == 0:
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

    plan_in = seams.get("switch_plan")
    if plan_in is not None:  # the switches' own cuts (part_switches_steps): no crossing search, the plans are exact
        crosses, fpts, plans, corner_at, reuse_tags = _plan_tables(plan_in, active)
        for f_, pl in plans.items():  # a junction whose lines all come back to it (loops) is no open chain's end
            if pl["fan"] is not None and pl["fan"][0][0] == _C:
                lst_ = corner_map.setdefault(pl["fan"][0][1], [])
                lst_.extend(c_ for _n, c_, _p in pl["fan"][1] if c_ not in lst_)
        face_list = sorted(plans)
        yield
    else:
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
            cr_ = crosses[(e, c)][1]
            new_tag.append((cr_[3] if len(cr_) > 3 else c, cr_[1]))  # a switch point keys its own id, its chain rides along
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
                s, _, reuse = crosses[(e, c)][1][:3]
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
    if plan_in is not None or SPLIT_ORDER == "per_face":  # most damage per face spent first, so a small allowance mends the most
        run_list.sort(key=lambda r: (-r[0] / max(r[3], 1), r[1]))
    else:
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
        fast = FAST.get("split") if use_fast and "switch_plan" not in seams else None  # the switches' exact plans run here
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
