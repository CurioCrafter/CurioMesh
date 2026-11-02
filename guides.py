# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Guide curves: strokes the quads must follow, like an edge loop round an eye or along a panel line.

Deformation lines are guide curves that become several parallel loops (`parallel`): the rows of quads a joint needs
to bend without pinching. `loop_edges` finds those loops again in the result and `bend_stats` measures them.

A stroke becomes a chain of mesh edges on the coarse triangle mesh that is about to be paired: the triangles near
it are split until their edges are short, the cheapest path of edges that hugs the stroke is found, and the
vertices on it are pulled onto the stroke. Those edges are then handed to the pairing as feature edges, which no
quad may cross and which the cross field runs along. Everything here is NumPy except `gather`.
"""

from __future__ import annotations

import heapq

import numpy as np

REFINE_PASSES = 6
LONG = 1.5  # triangles near a stroke are split until their edges are under this many stroke spacings
LEG = 6  # samples between the waypoints a path must pass
HUG = 8.0  # how much an edge costs extra per stroke spacing it lies away from the stroke
PULL = 0.45  # a vertex moves onto the stroke by at most this share of its shortest edge: no triangle flips


# --------------------------------------------------------------------------- reading strokes (Blender)


def lines_from_edges(co: np.ndarray, edges) -> list[np.ndarray]:
    """Polylines from a vertex array and its edges, in the order the edges come (splines list their points in order)."""
    out, run = [], []
    for a, b in edges:
        if not run or run[-1] != a:
            if len(run) > 1:
                out.append(co[run])
            run = [a]
        run.append(b)
    if len(run) > 1:
        out.append(co[run])
    return out


def _gp_points(obj):
    """Polylines of a Grease Pencil object (the old and the new data model), in the object's own space."""
    data = obj.data
    lines = []
    try:
        for layer in data.layers:
            frames = getattr(layer, "frames", [])
            for frame in frames:
                drawing = getattr(frame, "drawing", None)
                strokes = drawing.strokes if drawing is not None else frame.strokes
                for stroke in strokes:
                    pts = [tuple(getattr(p, "position", None) or p.co) for p in stroke.points]
                    if len(pts) > 1:
                        lines.append(np.asarray(pts, dtype=np.float64))
    except (AttributeError, RuntimeError):
        return []
    return lines


def is_guide_object(obj) -> bool:
    """A curve, a Grease Pencil drawing, or a mesh that has edges but no faces."""
    if obj is None:
        return False
    if obj.type in {"CURVE", "GPENCIL", "GREASEPENCIL"}:
        return True
    return obj.type == "MESH" and len(obj.data.edges) > 0 and len(obj.data.polygons) == 0


def gather(objects, source, depsgraph) -> list[np.ndarray]:
    """Every stroke of `objects`, as polylines in the source mesh's object space."""
    inverse = source.matrix_world.inverted()
    lines = []
    for obj in objects:
        if obj is None or not is_guide_object(obj):
            continue
        matrix = inverse @ obj.matrix_world
        m = np.asarray(matrix, dtype=np.float64)
        if obj.type in {"GPENCIL", "GREASEPENCIL"}:
            found = _gp_points(obj)
        else:
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            if mesh is None:  # not in this view layer, or a curve with nothing to draw
                evaluated.to_mesh_clear()
                continue
            try:
                flat = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
                mesh.vertices.foreach_get("co", flat)
                pairs = np.empty(len(mesh.edges) * 2, dtype=np.int64)
                mesh.edges.foreach_get("vertices", pairs)
                found = lines_from_edges(flat.reshape(-1, 3), pairs.reshape(-1, 2).tolist())
            finally:
                evaluated.to_mesh_clear()
        lines += [line @ m[:3, :3].T + m[:3, 3] for line in found]
    return lines


# --------------------------------------------------------------------------- strokes on the surface


def resample(line: np.ndarray, step: float) -> np.ndarray:
    """Points along a polyline, `step` apart (the last point is kept)."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    at = np.concatenate(([0.0], np.cumsum(seg)))
    if at[-1] <= 1e-12:
        return line[:1]
    t = np.append(np.arange(0.0, at[-1], max(step, 1e-12)), at[-1])
    return np.stack([np.interp(t, at, line[:, k]) for k in range(3)], axis=1)


def on_surface(lines, nearest, step: float) -> list[np.ndarray]:
    """The strokes laid onto the surface: `nearest(p)` gives the closest surface point to p.

    A stroke that leaves the surface and comes back would be joined by a line through the air, so it is cut where
    its surface points jump apart.
    """
    out = []
    for line in lines:
        pts = resample(line, step)
        if len(pts) < 2:
            continue
        laid = np.asarray([nearest(p) for p in pts.tolist()], dtype=np.float64)
        cut = np.flatnonzero(np.linalg.norm(np.diff(laid, axis=0), axis=1) > 6.0 * step) + 1
        out += [part for part in np.split(laid, cut) if len(part) > 1]
    return out


def reflect_to_side(lines, axis: int, offset: float, keep: int) -> list[np.ndarray]:
    """Points drawn on the side of the plane that is not solved are reflected onto the side that is."""
    out = []
    for line in lines:
        line = line.copy()
        wrong = keep * (line[:, axis] - offset) < 0
        line[wrong, axis] = 2.0 * offset - line[wrong, axis]
        out.append(line)
    return out


def _nearest_sample(points: np.ndarray, samples: np.ndarray, block: int = 4096):
    """(distance, index) of the closest sample to each point."""
    dist = np.empty(len(points))
    index = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), block):
        d = np.linalg.norm(points[start : start + block, None, :] - samples[None, :, :], axis=2)
        index[start : start + block] = d.argmin(axis=1)
        dist[start : start + block] = d.min(axis=1)
    return dist, index


# --------------------------------------------------------------------------- the mesh the stroke is cut into


def _edge_use(tris: np.ndarray):
    a = tris.ravel()
    b = tris[:, [1, 2, 0]].ravel()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * (int(tris.max()) + 1) + hi
    uniq, count = np.unique(key, return_counts=True)
    n = int(tris.max()) + 1
    return uniq // n, uniq % n, count


def _split(co: np.ndarray, tris: np.ndarray, mat: np.ndarray, keys: set, project):
    """Split the given edges at their midpoints; the triangles beside them are split to match."""
    lo, hi, count = _edge_use(tris)
    border = set(zip(lo[count == 1].tolist(), hi[count == 1].tolist()))
    mid, new_pts = {}, []
    for k in sorted(keys):
        p = 0.5 * (co[k[0]] + co[k[1]])
        if project is not None and k not in border:
            p = np.asarray(project(p), dtype=np.float64)
        mid[k] = len(co) + len(new_pts)
        new_pts.append(p)
    out_tris, out_mat = [], []
    for t, m in zip(tris.tolist(), mat.tolist()):
        a, b, c = t
        flags = [((min(a, b), max(a, b)) in mid), ((min(b, c), max(b, c)) in mid), ((min(c, a), max(c, a)) in mid)]
        if not any(flags):
            out_tris.append(t)
            out_mat.append(m)
            continue
        verts, vm = [a, b, c], [mid.get((min(a, b), max(a, b))), mid.get((min(b, c), max(b, c))), mid.get((min(c, a), max(c, a)))]
        # turn the triangle so that the split edges come first: edge k runs verts[k] -> verts[k+1]
        n = sum(flags)
        if n == 1:
            k = flags.index(True)
        elif n == 2:
            k = (flags.index(False) + 1) % 3  # the edge after the unsplit one
        else:
            k = 0
        p0, p1, p2 = verts[k], verts[(k + 1) % 3], verts[(k + 2) % 3]
        m01, m12, m20 = vm[k], vm[(k + 1) % 3], vm[(k + 2) % 3]
        if n == 1:
            new = [(p0, m01, p2), (m01, p1, p2)]
        elif n == 2:  # edges p0p1 and p1p2 are split
            new = [(m01, p1, m12), (p0, m01, p2), (m01, m12, p2)]
        else:
            new = [(p0, m01, m20), (m01, p1, m12), (m20, m12, p2), (m01, m12, m20)]
        out_tris += new
        out_mat += [m] * len(new)
    return np.vstack((co, np.asarray(new_pts))), np.asarray(out_tris, dtype=np.int64), np.asarray(out_mat, dtype=np.int64)


def refine(co, tris, mat, samples: np.ndarray, spacing: float, project=None):
    """Split the triangles near the stroke until their longest edge is under LONG * spacing. Returns (co, tris, mat)."""
    co, tris, mat = np.asarray(co, dtype=np.float64), np.asarray(tris, dtype=np.int64), np.asarray(mat, dtype=np.int64)
    for _ in range(REFINE_PASSES):
        corners = co[tris]
        sides = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2)
        longest = sides.max(axis=1)
        dist, _i = _nearest_sample(corners.mean(axis=1), samples)
        want = (dist < longest) & (longest > LONG * spacing)
        if not want.any():
            break
        edge = sides.argmax(axis=1)[want]
        t = tris[want]
        a, b = t[np.arange(len(t)), edge], t[np.arange(len(t)), (edge + 1) % 3]
        keys = {(int(x), int(y)) for x, y in zip(np.minimum(a, b), np.maximum(a, b))}
        co, tris, mat = _split(co, tris, mat, keys, project)
    return co, tris, mat


# --------------------------------------------------------------------------- the chain of edges


def _path(adj, cost, start: int, goal: int, banned: set):
    """Cheapest path of vertices from start to goal over the edges in `adj` (a dict), or None."""
    best = {start: 0.0}
    came = {}
    heap = [(0.0, start)]
    while heap:
        c, u = heapq.heappop(heap)
        if u == goal:
            path = [u]
            while path[-1] != start:
                path.append(came[path[-1]])
            return path[::-1]
        if c > best.get(u, np.inf):
            continue
        for w in adj.get(u, ()):
            key = (u, w) if u < w else (w, u)
            step = c + cost[key] * (40.0 if key in banned else 1.0)
            if step < best.get(w, np.inf):
                best[w] = step
                came[w] = u
                heapq.heappush(heap, (step, w))
    return None


def chain(co: np.ndarray, tris: np.ndarray, lines: list[np.ndarray], spacing: float, edges=None):
    """Vertex chains that hug the strokes. Returns (chains, edge keys): chains are lists of vertex ids, keys are (lo, hi). `edges`, as (lo, hi), are walked instead of the triangles' own: a quad mesh's, say."""
    lo, hi = edges if edges is not None else _edge_use(tris)[:2]
    length = np.linalg.norm(co[hi] - co[lo], axis=1)
    mid = 0.5 * (co[lo] + co[hi])
    chains, keys = [], set()
    for line in lines:
        samples = resample(line, spacing / 2.0)
        if len(samples) < 2:
            continue
        away, _i = _nearest_sample(mid, samples)
        near = away < 3.0 * spacing
        cost = {}
        adj: dict[int, list[int]] = {}
        for a, b, w, d in zip(lo[near].tolist(), hi[near].tolist(), length[near].tolist(), away[near].tolist()):
            cost[a, b] = w * (1.0 + HUG * d / spacing)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        if not adj:
            continue
        verts = np.fromiter(adj, dtype=np.int64)
        _d, nearest = _nearest_sample(samples, co[verts])
        stops = list(range(0, len(samples), LEG))
        if stops[-1] != len(samples) - 1:
            stops.append(len(samples) - 1)
        waypoints = [int(verts[nearest[i]]) for i in stops]
        banned: set = set()
        run = [waypoints[0]]
        for u, w in zip(waypoints[:-1], waypoints[1:]):
            if u == w:
                continue
            found = _path(adj, cost, u, w, banned)
            if found is None:
                continue
            for x, y in zip(found[:-1], found[1:]):
                banned.add((x, y) if x < y else (y, x))
            run += found[1:]
        if len(run) > 1:
            chains.append(run)
            keys |= {(min(x, y), max(x, y)) for x, y in zip(run[:-1], run[1:]) if x != y}
    return chains, keys


def pull(co: np.ndarray, tris: np.ndarray, chains, lines, spacing: float, protect: np.ndarray | None = None) -> np.ndarray:
    """Move the vertices of the chains onto the nearest point of the strokes, a little at a time at most."""
    co = co.copy()
    lo, hi, count = _edge_use(tris)
    shortest = np.full(len(co), np.inf)
    length = np.linalg.norm(co[hi] - co[lo], axis=1)
    np.minimum.at(shortest, lo, length)
    np.minimum.at(shortest, hi, length)
    border = np.zeros(len(co), dtype=bool)
    border[lo[count == 1]] = border[hi[count == 1]] = True
    if protect is not None:
        border |= protect
    samples = np.vstack([resample(line, spacing / 4.0) for line in lines])
    used = np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in chains]))
    used = used[~border[used]]
    dist, index = _nearest_sample(co[used], samples)
    step = np.minimum(1.0, PULL * shortest[used] / np.maximum(dist, 1e-30))
    co[used] += (samples[index] - co[used]) * step[:, None]
    return co


# --------------------------------------------------------------------------- how closely the result follows


def follow(lines, co: np.ndarray, faces, spacing: float):
    """(mean, worst) distance from points along the strokes to the nearest edge of the result."""
    seen = set()
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            seen.add((a, b) if a < b else (b, a))
    e = np.asarray(sorted(seen), dtype=np.int64)
    p, q = co[e[:, 0]], co[e[:, 1]]
    d = q - p
    dd = np.maximum((d * d).sum(axis=1), 1e-30)
    worst, total, n = 0.0, 0.0, 0
    for line in lines:
        pts = resample(line, spacing)
        best = np.full(len(pts), np.inf)
        for start in range(0, len(pts), 256):
            x = pts[start : start + 256, None, :]
            t = np.clip(((x - p[None]) * d[None]).sum(axis=2) / dd[None], 0.0, 1.0)
            best[start : start + 256] = np.linalg.norm(p[None] + t[..., None] * d[None] - x, axis=2).min(axis=1)
        worst = max(worst, float(best.max()))
        total += float(best.sum())
        n += len(pts)
    return total / max(n, 1), worst


# --------------------------------------------------------------------------- deformation lines: loops across a joint


def parallel(lines, count: int, gap: float, nearest):
    """`count` parallel loops for each stroke, `gap` apart and centred on it, all laid on the surface.

    `nearest(p)` gives (closest surface point, surface normal). Returns one list of loops per stroke piece, the
    middle loop being the stroke itself.
    """
    groups = []
    for piece in on_surface(lines, lambda p: nearest(p)[0], gap / 2.0):
        if len(piece) < 3:
            continue
        normal = np.asarray([nearest(p)[1] for p in piece.tolist()], dtype=np.float64)
        tangent = np.gradient(piece, axis=0)
        kernel = np.ones(5) / 5.0  # a stroke of straight pieces has jumpy tangents: smooth them so the loops stay parallel
        tangent = np.stack([np.convolve(np.pad(tangent[:, k], 2, mode="edge"), kernel, mode="valid") for k in range(3)], axis=1)
        side = np.cross(normal, tangent)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-30)
        loops = []
        for k in range(count):
            offset = (k - (count - 1) / 2.0) * gap
            if abs(offset) < 1e-12:
                loops.append(piece)
            else:
                loops.append(np.asarray([nearest(p)[0] for p in (piece + offset * side).tolist()], dtype=np.float64))
        groups.append(loops)
    return groups


def _edge_array(faces) -> np.ndarray:
    seen = set()
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            seen.add((a, b) if a < b else (b, a))
    return np.asarray(sorted(seen), dtype=np.int64).reshape(-1, 2)


def _to_line(points: np.ndarray, line: np.ndarray, block: int = 512) -> np.ndarray:
    """Distance from each point to a polyline."""
    a, b = line[:-1], line[1:]
    d = b - a
    dd = np.maximum((d * d).sum(axis=1), 1e-30)
    out = np.empty(len(points))
    for start in range(0, len(points), block):
        x = points[start : start + block, None, :]
        t = np.clip(((x - a[None]) * d[None]).sum(axis=2) / dd[None], 0.0, 1.0)
        out[start : start + block] = np.linalg.norm(a[None] + t[..., None] * d[None] - x, axis=2).min(axis=1)
    return out


def loop_edges(lines, co: np.ndarray, faces, tol: float) -> list[tuple[int, int]]:
    """Edges of the result that run along one of the lines: both ends are within `tol` of it. Sorted (lo, hi) pairs."""
    edges = _edge_array(faces)
    if not len(edges):
        return []
    hit = np.zeros(len(edges), dtype=bool)
    for line in lines:
        near = _to_line(co, line) < tol if len(line) > 1 else np.zeros(len(co), dtype=bool)
        hit |= near[edges[:, 0]] & near[edges[:, 1]]
    return [(int(a), int(b)) for a, b in edges[hit]]


def bend_stats(groups, co: np.ndarray, faces, gap: float) -> dict:
    """How well the result carries the loops: loops found, poles on them, and how even their spacing is."""
    edges = _edge_array(faces)
    valence = np.bincount(edges.ravel(), minlength=len(co))
    tol = 0.3 * gap
    found = expected = 0
    on_loops: set[int] = set()
    gaps = []
    for loops in groups:
        previous = None
        for loop in loops:
            expected += 1
            mine = loop_edges([loop], co, faces, tol)
            length = float(np.linalg.norm(co[[b for _a, b in mine]] - co[[a for a, _b in mine]], axis=1).sum()) if mine else 0.0
            if length >= 0.7 * float(np.linalg.norm(np.diff(loop, axis=0), axis=1).sum()):
                found += 1
            ids = sorted({v for e in mine for v in e})
            on_loops.update(ids)
            if previous is not None and ids:
                gaps.append(_to_line(co[ids], previous))
            previous = loop
    poles = sum(1 for v in on_loops if valence[v] != 4)
    spread = np.concatenate(gaps) if gaps else np.zeros(1)
    return {"loops": expected, "found": found, "verts": len(on_loops), "poles": poles, "gap": float(spread.mean() / max(gap, 1e-30)), "uneven": float(spread.std() / max(spread.mean(), 1e-30))}
