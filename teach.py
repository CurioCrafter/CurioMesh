# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Teach by example: remember how an artist fixed one patch of quads, and offer the same fix where the same patch shows up.

A fix is a small rewrite rule. Its left side is the faces the artist replaced ("before"), its right side the faces
they put there ("after"); the two share one outline loop. Both are stored without coordinates in the world: vertex
positions are relative to the outline's centre and scaled by its edge length. Finding the rule again is a walk
across the quads (a patch of a quad mesh is fixed by one face and which way it is turned), so a "before" pattern
matches exactly where the quads around it are connected the same way. Where it matches, the new faces are put in by
a similarity fit of the outline onto the one found, then projected onto the surface. Everything is plain Python
and NumPy, and nothing leaves the machine: the library is one JSON file the user can read, export and import.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time

import numpy as np

TOL = 1e-6  # of the model's size: vertices closer than this are the same vertex when two meshes are compared
SHAPE_LIMIT = 0.35  # how far a place's shape may be from the shape that was fixed (each of three numbers)


def _keys(co: np.ndarray, size: float) -> list[tuple]:
    return [tuple(r) for r in np.round(np.asarray(co) / max(size * TOL, 1e-300)).astype(np.int64).tolist()]


def _sig(face, keys) -> tuple:
    ring = [keys[v] for v in face]
    k = ring.index(min(ring))
    return tuple(ring[k:] + ring[:k])


def changed(co_a, faces_a, co_b, faces_b, size: float):
    """(faces of A that are gone in B, faces of B that are new), found by where their corners are."""
    ka, kb = _keys(co_a, size), _keys(co_b, size)
    sa, sb = [_sig(f, ka) for f in faces_a], [_sig(f, kb) for f in faces_b]
    in_a, in_b = set(sa), set(sb)
    return [i for i, s in enumerate(sa) if s not in in_b], [i for i, s in enumerate(sb) if s not in in_a]


def _components(faces, ids) -> list[list[int]]:
    ids = list(ids)
    edge: dict = {}
    for i in ids:
        f = faces[i]
        for k in range(len(f)):
            edge.setdefault((min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)])), []).append(i)
    seen, out = set(), []
    for start in ids:
        if start in seen:
            continue
        stack, part = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            part.append(i)
            f = faces[i]
            for k in range(len(f)):
                for j in edge[(min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)]))]:
                    if j not in seen:
                        seen.add(j)
                        stack.append(j)
        out.append(part)
    return out


def outline(faces, ids):
    """The outline of a face set as one loop of vertices in the faces' winding, or None when it is not one clean loop."""
    half = {(f[k], f[(k + 1) % len(f)]) for f in (faces[i] for i in ids) for k in range(len(f))}
    after: dict = {}
    for a, b in half:
        if (b, a) not in half:
            if a in after:
                return None
            after[a] = b
    if not after:
        return None
    start = next(iter(after))
    loop, v = [start], after[start]
    while v != start:
        if v not in after or len(loop) > len(after):
            return None
        loop.append(v)
        v = after[v]
    return loop if len(loop) == len(after) else None


def similarity(src: np.ndarray, dst: np.ndarray):
    """Scale, rotation and shift that carry `src` points onto `dst` points, and the rms that is left over."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - ms, dst - md
    u, s, vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    flip = np.diag([1.0, 1.0, d])
    rot = u @ flip @ vt
    scale = float((s * np.diag(flip)).sum() / max((xs * xs).sum() / len(src), 1e-30))
    shift = md - scale * rot @ ms
    left = np.sqrt(np.mean(np.sum((scale * xs @ rot.T - xd) ** 2, axis=1)))
    return scale, rot, shift, float(left)


def describe(points: np.ndarray, faces) -> list[float]:
    """Three numbers for the shape of a patch: how flat, how long, how much it turns (all without size)."""
    w = np.sort(np.linalg.eigvalsh(np.cov((points - points.mean(axis=0)).T)))[::-1]
    normals = []
    for f in faces:
        p = points[list(f)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        normals.append(n / max(np.linalg.norm(n), 1e-30))
    normals = np.asarray(normals)
    mean = normals.mean(axis=0)
    mean /= max(np.linalg.norm(mean), 1e-30)
    return [float(w[2] / max(w[0], 1e-30)), float(w[1] / max(w[0], 1e-30)), float(np.mean(1.0 - normals @ mean))]


def make_fix(co_a, faces_a, gone, co_b, faces_b, new, size: float):
    """The rule for one fix as a dict, or a string saying why this is not one."""
    parts = _components(faces_a, gone)
    if len(parts) != 1:
        return "More than one place changed: remember one fix at a time"
    gone = parts[0]
    loop = outline(faces_a, gone)
    if loop is None:
        return "The fixed area has to be one patch with one outline (no holes, no pinches)"
    ka, kb = _keys(co_a, size), _keys(co_b, size)
    in_b = {k: i for i, k in enumerate(kb)}
    if any(ka[v] not in in_b for v in loop):
        return "The outline of the patch was moved or cut: change only the inside"
    ring_b = [in_b[ka[v]] for v in loop]
    if outline(faces_b, new) is None or set(outline(faces_b, new)) != set(ring_b):
        return "The new faces must close on the same outline as the old ones"
    local_a = {v: i for i, v in enumerate(loop)}
    inner_a = sorted({v for i in gone for v in faces_a[i]} - set(loop))
    for v in inner_a:
        local_a[v] = len(local_a)
    local_b = {v: i for i, v in enumerate(ring_b)}
    inner_b = sorted({v for i in new for v in faces_b[i]} - set(ring_b))
    for v in inner_b:
        local_b[v] = len(local_b)
    ring = np.asarray(co_a)[loop]
    centre = ring.mean(axis=0)
    scale = float(np.mean(np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1))) or 1.0

    def put(points):
        return ((np.asarray(points) - centre) / scale).round(6).tolist()

    before = [tuple(local_a[v] for v in faces_a[i]) for i in gone]
    after = [tuple(local_b[v] for v in faces_b[i]) for i in new]
    first = np.asarray(co_a)[sorted(local_a, key=local_a.get)]
    return {
        "id": f"fix-{time.strftime('%Y%m%d-%H%M%S')}",
        "ring": len(loop),
        "ring_points": put(ring),
        "before": before,
        "before_inner": put(first[len(loop):]),
        "after": after,
        "after_inner": put(np.asarray(co_b)[inner_b]),
        "shape": describe(first, before),
        "uses": 0,
    }


# --------------------------------------------------------------------------- finding it again


def find(fix: dict, co, faces, limit: float = SHAPE_LIMIT, near=None, reach: float = 0.0):
    """Places in a quad mesh where the 'before' of `fix` is, best first: list of dicts with faces, vertex map, score.

    `near` (a point) with `reach` keeps only places whose outline centre is within `reach` of it.
    """
    co = np.asarray(co, dtype=np.float64)
    before = [tuple(f) for f in fix["before"]]
    ring_n = int(fix["ring"])
    inner_n = len(fix["before_inner"])
    dedge = {(f[k], f[(k + 1) % len(f)]): (i, k) for i, f in enumerate(faces) for k in range(len(f))}
    sedge = {(f[k], f[(k + 1) % len(f)]): (i, k) for i, f in enumerate(before) for k in range(len(f))}
    around: dict = {}
    for i, f in enumerate(faces):
        for v in f:
            around.setdefault(v, []).append(i)
    inside = {v: sum(1 for f in before if v in f) for v in range(ring_n, ring_n + inner_n)}
    pivot = max(inside, key=lambda v: abs(inside[v] - 4)) if inside else None
    roots = []
    if pivot is not None and inside[pivot] != 4:
        sf = next(i for i, f in enumerate(before) if pivot in f)
        cp = before[sf].index(pivot)
        for w, fl in around.items():
            if len(fl) == inside[pivot]:
                for t in fl:
                    if len(faces[t]) == len(before[sf]) and w in faces[t]:
                        roots.append((sf, t, (faces[t].index(w) - cp) % len(before[sf])))
    else:
        roots = [(0, t, o) for t, f in enumerate(faces) if len(f) == len(before[0]) for o in range(len(f))]
    ring_pts = np.asarray(fix["ring_points"])
    found, taken = [], set()
    for sf, t, o in roots:
        got = _walk(before, sedge, faces, dedge, sf, t, o)
        if got is None:
            continue
        fmap, vmap = got
        if any(len(around[vmap[v]]) != n for v, n in inside.items()):
            continue
        target = [f for f, _o in fmap.values()]
        key = frozenset(target)
        if key in taken:
            continue
        ring = co[[vmap[i] for i in range(ring_n)]]
        scale, rot, shift, left = similarity(ring_pts, ring)
        gap = float(np.mean(np.linalg.norm(np.roll(ring_pts, -1, axis=0) - ring_pts, axis=1))) * scale
        distance = max(abs(a - b) for a, b in zip(describe(co[[vmap[i] for i in range(ring_n + inner_n)]], before), fix["shape"]))
        if distance > limit or left > 0.5 * gap:
            continue
        centre = ring.mean(axis=0)
        if near is not None and np.linalg.norm(centre - np.asarray(near)) > reach:
            continue
        taken.add(key)
        found.append({"faces": sorted(target), "vmap": {int(k): int(v) for k, v in vmap.items()}, "score": float(distance + left / max(gap, 1e-30)), "centre": centre.tolist(), "fit": (scale, rot, shift)})
    return sorted(found, key=lambda m: m["score"])


def _walk(before, sedge, faces, dedge, root, tface, offset):
    """Lay the 'before' faces over the mesh starting from one face, or None where the quads around it are connected differently."""
    fmap, vmap, queue, used = {root: (tface, offset)}, {}, [root], {tface}
    while queue:
        s = queue.pop()
        t, o = fmap[s]
        sv, tv = before[s], faces[t]
        m = len(sv)
        if len(tv) != m:
            return None
        for i in range(m):
            if vmap.setdefault(sv[i], tv[(o + i) % m]) != tv[(o + i) % m]:
                return None
        for i in range(m):
            other = sedge.get((sv[(i + 1) % m], sv[i]))
            if other is None:
                continue
            g, j = other
            hit = dedge.get((tv[(o + i + 1) % m], tv[(o + i) % m]))
            if hit is None or len(faces[hit[0]]) != len(before[g]):
                return None
            u, p = hit
            want = (u, (p - j) % len(before[g]))
            if g in fmap:
                if fmap[g] != want:
                    return None
            elif u in used:
                return None
            else:
                fmap[g] = want
                used.add(u)
                queue.append(g)
    if len(set(vmap.values())) != len(vmap):
        return None
    return fmap, vmap


def put_in(co, faces, fix: dict, match: dict, project=None):
    """Replace the matched faces by the fix's 'after'. Returns (co, faces, kept) with `kept` a boolean over the old faces
    that survive (the new faces come last); vertices nothing uses any more are dropped."""
    co = np.asarray(co, dtype=np.float64)
    scale, rot, shift = match["fit"]
    inner = np.asarray(fix["after_inner"], dtype=np.float64).reshape(-1, 3)
    placed = (scale * inner @ rot.T + shift) if len(inner) else inner
    if project is not None and len(placed):
        placed = np.asarray([project(p) for p in placed.tolist()], dtype=np.float64)
    base = len(co)
    ring_n = int(fix["ring"])
    vmap = match["vmap"]
    kept = np.ones(len(faces), dtype=bool)
    kept[match["faces"]] = False
    added = [tuple(vmap[v] if v < ring_n else base + v - ring_n for v in f) for f in fix["after"]]
    all_co = np.vstack((co, placed))
    all_faces = [f for i, f in enumerate(faces) if kept[i]] + added
    used = np.unique(np.fromiter((v for f in all_faces for v in f), dtype=np.int64))
    renumber = np.full(len(all_co), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))
    return all_co[used], [tuple(int(renumber[v]) for v in f) for f in all_faces], kept


# --------------------------------------------------------------------------- the library file


MOST_BYTES = 32_000_000  # a library file bigger than this is not read
MOST_FIXES = 2000
MOST_VERTICES = 20_000  # in one fix
MOST_FACES = 20_000
LONGEST_NAME = 200


def _points(value) -> bool:
    """A list of finite (x, y, z)."""
    return isinstance(value, list) and len(value) <= MOST_VERTICES and all(
        isinstance(p, list) and len(p) == 3 and all(type(x) in (int, float) and math.isfinite(x) for x in p) for p in value
    )


def _faces_in(value, count: int) -> bool:
    """A list of faces of 3 or more corners, each a different vertex below `count`."""
    return isinstance(value, list) and 0 < len(value) <= MOST_FACES and all(
        isinstance(f, list) and 3 <= len(f) <= 64 and len(set(f)) == len(f) and all(type(v) is int and 0 <= v < count for v in f) for f in value
    )


def valid_fix(fix) -> bool:
    """Whether a fix read from a file is safe to use: a library can come from anybody, so nothing in it is taken on trust."""
    if not isinstance(fix, dict):
        return False
    ring = fix.get("ring")
    if type(ring) is not int or not 3 <= ring <= MOST_VERTICES:
        return False
    if not isinstance(fix.get("id"), str) or len(fix["id"]) > LONGEST_NAME or type(fix.get("uses", 0)) is not int:
        return False
    if not (_points(fix.get("ring_points")) and _points(fix.get("before_inner", [])) and _points(fix.get("after_inner", []))) or len(fix["ring_points"]) != ring:
        return False
    shape = fix.get("shape", [])
    if not isinstance(shape, list) or len(shape) > 16 or not all(type(x) in (int, float) and math.isfinite(x) for x in shape):
        return False
    return _faces_in(fix.get("before"), ring + len(fix.get("before_inner", []))) and _faces_in(fix.get("after"), ring + len(fix.get("after_inner", [])))


def load(path: str) -> list[dict]:
    """The valid fixes in a file. Anything else in it is left out; a file that can't be read gives none."""
    try:
        if os.path.getsize(path) > MOST_BYTES:
            return []
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        fixes = data.get("fixes", [])
        return [f for f in fixes[:MOST_FIXES] if valid_fix(f)] if isinstance(fixes, list) else []
    except (OSError, ValueError, AttributeError, RecursionError):
        return []


def save(path: str, fixes: list[dict]) -> None:
    """Write through a unique file beside it, so nobody else's file is followed and a crash leaves the old one whole."""
    descriptor, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), prefix=".fixes-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "note": "Quad fixes remembered by CurioMesh. Local only; share by exporting.", "fixes": fixes}, handle)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def merge(old: list[dict], new: list[dict]) -> list[dict]:
    have = {f["id"] for f in old}
    return old + [f for f in new if f["id"] not in have]
