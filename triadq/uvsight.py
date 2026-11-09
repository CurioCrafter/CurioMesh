# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Visibility twin (package K4): which faces and edges are seen, and how many pixels of each edge show.

NumPy only, no bpy. Same method as the kernel source: orthographic depth pictures from many directions, seen face =
weighted hits over expected hits, edge pixels from a scan of the right and lower neighbour of every pixel (different
faces at near-equal depth = inside, a depth jump or an empty neighbour = outline). Stages are sliceable: `Sight.step()`
does about 4 ms of work and says whether it is finished. Pictures come from, in this order: a finished set given by the
caller, the optional callable `FAST["draw_job"]` (starts a job that draws them), or the caller's own `pictures`
function (the six-view generator the add-on already has; extra views turn the points so that the view looks along an
axis).
"""

from __future__ import annotations

import numpy as np

# Constants table: same names and values as the top of the matching .rs file (tests/test_uv_scaffold.py compares).
VIEWS_DEFAULT = 6
PIXELS_MIN = 512
PIXELS_PER_EDGE = 4.0
DEPTH_PIXELS = 2.0
SUBPIXEL_HITS = 1.0
SMOOTH_ROUNDS = 2
UNDERSIDE_WEIGHT = 0.3
FRAME_PAD = 1.02
MAX_PICTURE_BYTES = 4294967296

CELLS_PER_STEP = 1 << 16  # about 4 ms of NumPy per step

# Optional faster callables, set by the one adapter (uv_fast.py); the NumPy code runs when a name is missing.
FAST: dict = {}


def set_fast(**callables) -> None:
    """Install (or, with None, remove) faster callables by name."""

    for key, fn in callables.items():
        if fn is None:
            FAST.pop(key, None)
        else:
            FAST[key] = fn


# --------------------------------------------------------------------------- views


def view_dirs(count: int) -> np.ndarray:
    """First `count` directions toward the camera: 6 axes, 8 corners, 12 edge middles, then a golden-angle spiral."""

    out = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    if count > 6:
        for i in range(8):
            out.append(tuple(-1.0 if (i >> bit) & 1 else 1.0 for bit in range(3)))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            for sa in (1.0, -1.0):
                for sb in (1.0, -1.0):
                    d = [0.0, 0.0, 0.0]
                    d[a], d[b] = sa, sb
                    out.append(tuple(d))
    dirs = np.array(out, dtype=np.float64).reshape(-1, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    if count > 26:
        extra = count - 26
        golden = np.pi * (3.0 - 5.0**0.5)
        k = np.arange(extra, dtype=np.float64)
        z = 1.0 - 2.0 * (k + 0.5) / extra
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        spiral = np.stack((r * np.cos(golden * k), r * np.sin(golden * k), z), axis=1)
        dirs = np.vstack((dirs, spiral))
    return dirs[:count]


def view_weights(dirs: np.ndarray, all_sides: bool = True) -> np.ndarray:
    """All 1; with `all_sides` off the view from straight below gets `UNDERSIDE_WEIGHT`."""

    w = np.ones(len(dirs))
    if not all_sides:
        w[dirs[:, 2] < -0.99] = UNDERSIDE_WEIGHT
    return w


def basis(d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image axes u, v and the direction d toward the camera, right-handed (same choice as the kernel)."""

    d = np.asarray(d, dtype=np.float64)
    d = d / np.linalg.norm(d)
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(d)))] = 1.0
    u = np.cross(helper, d)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u), d


def frame(points: np.ndarray, flat: np.ndarray) -> tuple[np.ndarray, float]:
    """Bounding-box center and the radius of the sphere around it that holds every used vertex."""

    if len(flat) == 0:
        return np.zeros(3), 1.0
    used = points[np.unique(flat)]
    center = (used.min(axis=0) + used.max(axis=0)) / 2.0
    return center, max(float(np.sqrt(((used - center) ** 2).sum(axis=1).max())), 1e-12)


def default_pixels(points: np.ndarray, indptr: np.ndarray, flat: np.ndarray) -> int:
    """About `PIXELS_PER_EDGE` across the median edge, at least `PIXELS_MIN`. No upper cap."""

    if len(flat) == 0:
        return PIXELS_MIN
    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    lengths = np.linalg.norm(points[flat] - points[flat[nxt]], axis=1)
    median = float(np.sort(lengths)[len(lengths) // 2])
    if median <= 0.0:
        return PIXELS_MIN
    _, radius = frame(points, flat)
    return max(int(np.ceil(PIXELS_PER_EDGE * 2.0 * radius / median)), PIXELS_MIN)


def csr(polys) -> tuple[np.ndarray, np.ndarray]:
    """(indptr, flat) from a list of vertex tuples, or a pair already in that form."""

    if isinstance(polys, tuple) and len(polys) == 2 and isinstance(polys[0], np.ndarray):
        return polys[0].astype(np.int64, copy=False), polys[1].astype(np.int64, copy=False)
    sizes = np.fromiter((len(p) for p in polys), dtype=np.int64, count=len(polys))
    indptr = np.zeros(len(polys) + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    flat = np.fromiter((v for p in polys for v in p), dtype=np.int64, count=int(indptr[-1]))
    return indptr, flat


# --------------------------------------------------------------------------- pictures


class Pictures:
    """Pictures of one mesh from many directions: `face[k]` (pixels, pixels) ints with -1 for empty, `depth[k]` float32."""

    def __init__(self, dirs, weights, pixels, center, half, face=None, depth=None):
        self.dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.pixels, self.center, self.half = int(pixels), np.asarray(center, dtype=np.float64), float(half)
        self.face = [np.asarray(f, dtype=np.int64) for f in face] if face is not None else []
        self.depth = [np.asarray(d, dtype=np.float32) for d in depth] if depth is not None else []

    @property
    def drawn(self) -> int:
        return len(self.face)


def _turned_pictures(pictures_fn, points, polys, dirs, center, half, pixels):
    """Generator: draw each direction with the six-view function by turning the points so the view looks down +z."""

    out_face, out_depth = [], []
    for d in dirs:
        u, v, d = basis(d)
        turned = (points - center) @ np.stack((u, v, d), axis=1)  # columns: u, v, d coordinates
        found = pictures_fn(turned, polys, np.zeros(3), half, pixels)
        if hasattr(found, "__next__"):  # a generator that yields progress and returns the pictures
            while True:
                try:
                    next(found)
                    yield
                except StopIteration as done:
                    found = done.value
                    break
        hit, _, depth, face = found[4]  # the +z picture: camera at +z looking down, pixel (row, col) = (u, v)
        face = np.where(np.asarray(hit).reshape(pixels, pixels), np.asarray(face).reshape(pixels, pixels), -1)
        out_face.append(face.astype(np.int64))
        out_depth.append(np.asarray(depth, dtype=np.float32).reshape(pixels, pixels))
        yield
    return out_face, out_depth


# --------------------------------------------------------------------------- reading the pictures


class Sight:
    """Sliceable: `step()` does about 4 ms and returns True when finished; then `seen_face`, `inside_px`, `outline_px`.

    `pictures` may be a finished `Pictures`; otherwise they are drawn first (see the module note). Reading again on
    the same `Pictures` (`again()`) draws nothing.
    """

    def __init__(self, points, polys, views=VIEWS_DEFAULT, pixels=0, all_sides=True, pictures=None, draw=None):
        self.points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        self.indptr, self.flat = csr(polys)
        self.polys = polys
        self.draw = draw  # the caller's six-view function, used when nothing faster is installed
        if pictures is None:
            dirs = view_dirs(views)
            center, radius = frame(self.points, self.flat)
            pixels = pixels or default_pixels(self.points, self.indptr, self.flat)
            if len(dirs) * pixels * pixels * 8 > MAX_PICTURE_BYTES:
                raise MemoryError("these views at this size do not fit in memory")
            pictures = Pictures(dirs, view_weights(dirs, all_sides), pixels, center, radius * FRAME_PAD)
            self._to_draw = True
        else:
            self._to_draw = pictures.drawn < len(pictures.dirs)
        self.pictures = pictures
        self.renders = 0
        self.seen_face = self.inside_px = self.outline_px = None
        self._gen = self._run()
        self.done = False

    def again(self, weights=None) -> None:
        """Read the held pictures once more (new weights allowed); nothing is drawn."""

        if weights is not None:
            self.pictures.weights = np.asarray(weights, dtype=np.float64)
        self._gen, self.done = self._run(), False

    def step(self) -> bool:
        if not self.done:
            try:
                next(self._gen)
            except StopIteration:
                self.done = True
        return self.done

    def run(self) -> "Sight":
        while not self.step():
            pass
        return self

    # -- stages
    def _run(self):
        pic = self.pictures
        if self._to_draw or pic.drawn < len(pic.dirs):
            todo = pic.dirs[pic.drawn :]
            job = FAST["draw_job"](self.points, (self.indptr, self.flat), len(pic.dirs), pic.pixels, pic.weights) if "draw_job" in FAST and pic.drawn == 0 else None
            if job is not None:
                while not job.done():
                    yield
                got = job.result()
                if hasattr(got, "__next__"):  # copied a slice at a time
                    got = yield from got
                pic.face, pic.depth = list(got[0]), list(got[1])
                self.renders += len(pic.face)
            else:
                if self.draw is None:
                    from .. import autoretopo  # the add-on's six-view pictures; read only, never edited

                    self.draw = autoretopo.pictures
                polys = [self.flat[a:b].tolist() for a, b in zip(self.indptr[:-1].tolist(), self.indptr[1:].tolist())]  # polygons, not the (indptr, flat) pair
                drawn = yield from _turned_pictures(self.draw, self.points, polys, todo, pic.center, pic.half, pic.pixels)
                pic.face += drawn[0]
                pic.depth += drawn[1]
                self.renders += len(drawn[0])
            self._to_draw = False
        yield from self._read()

    def _read(self):
        pic, pts = self.pictures, self.points
        indptr, flat = self.indptr, self.flat
        nf, n = len(indptr) - 1, pic.pixels
        sizes = np.diff(indptr)
        owner = np.repeat(np.arange(nf), sizes)
        nxt = np.arange(1, len(flat) + 1)
        nxt[indptr[1:] - 1] = indptr[:-1]
        # Face areas and normals (Newell).
        cr = np.cross(pts[flat], pts[flat[nxt]])
        m = np.zeros((nf, 3))
        np.add.at(m, owner, cr)
        size = np.linalg.norm(m, axis=1)
        area = size / 2.0
        normal = np.divide(m, size[:, None], out=np.zeros_like(m), where=size[:, None] > 0)
        # Edges: first-occurrence order like the kernel (sorted keys, ordered by first corner).
        lo, hi = np.minimum(flat, flat[nxt]), np.maximum(flat, flat[nxt])
        keys = lo.astype(np.uint64) << np.uint64(32) | hi.astype(np.uint64)
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        corner_edge = rank[inverse.reshape(-1)]
        ne = len(order)
        ea, eb = lo[first[order]], hi[first[order]]
        edge_key = keys[first[order]]
        sorted_edge = np.argsort(edge_key)
        # Face pair -> first edge shared (edges in order, uses in corner order).
        pair_keys, pair_edges = [], []
        by_edge = np.argsort(corner_edge, kind="stable")
        starts = np.searchsorted(corner_edge[by_edge], np.arange(ne + 1))
        for e in range(ne):
            uses = owner[by_edge[starts[e] : starts[e + 1]]]
            for i in range(len(uses)):
                for j in range(i + 1, len(uses)):
                    if uses[i] != uses[j]:
                        pair_keys.append((min(uses[i], uses[j]) << 32) | max(uses[i], uses[j]))
                        pair_edges.append(e)
        pair_keys = np.array(pair_keys, dtype=np.uint64)
        pair_edges = np.array(pair_edges, dtype=np.int64)
        if len(pair_keys):
            _, keep = np.unique(pair_keys, return_index=True)  # first occurrence wins
            pair_keys, pair_edges = pair_keys[keep], pair_edges[keep]
        pixel = 2.0 * pic.half / pic.pixels
        scale = pic.pixels / (2.0 * pic.half)
        tol = np.float32(DEPTH_PIXELS * pixel)
        total_w = max(float(pic.weights[: pic.drawn].sum()), 1e-300)
        got, expected = np.zeros(nf), np.zeros(nf)
        inside, outline = np.zeros(ne), np.zeros(ne)
        for k in range(pic.drawn):
            u, v, d = basis(pic.dirs[k])
            q = pts - pic.center
            proj = np.stack(((q @ u + pic.half) * scale - 0.5, (q @ v + pic.half) * scale - 0.5, 2.0 * pic.half - q @ d), axis=1)
            facing = normal @ d
            front = facing > 1e-9
            F, D = pic.face[k], pic.depth[k]
            got += pic.weights[k] * np.bincount(F[F >= 0][front[F[F >= 0]]], minlength=nf)
            expected += pic.weights[k] * np.where(front, area * facing / (pixel * pixel), 0.0)
            dx, dy = proj[eb, 0] - proj[ea, 0], proj[eb, 1] - proj[ea, 1]
            length = np.hypot(dx, dy)
            factor = np.divide(length, np.abs(dx) + np.abs(dy), out=np.zeros(ne), where=length > 1e-9)
            for r0 in range(0, n, max(1, CELLS_PER_STEP // n)):
                r1 = min(n, r0 + max(1, CELLS_PER_STEP // n))
                self._scan(F, D, r0, r1, front, proj, scale, tol, pair_keys, pair_edges, factor, inside, outline, pic.weights[k] / total_w, owner, flat, indptr, sizes, edge_key, sorted_edge)
                yield
        seen = np.where(expected > 0, np.clip(got / np.where(expected > 0, expected, 1.0), 0.0, 1.0), 0.0)
        # Faces too small to be hit reliably borrow the mean of their edge neighbours.
        small = expected < SUBPIXEL_HITS * total_w / max(pic.drawn, 1)
        neighbours: list[list[int]] = [[] for _ in range(nf)]
        for e in range(ne):
            uses = owner[by_edge[starts[e] : starts[e + 1]]]
            for a in uses:
                if small[a]:
                    neighbours[a].extend(int(b) for b in uses if b != a)
        for _ in range(SMOOTH_ROUNDS):
            before = seen.copy()
            for f in np.nonzero(small)[0]:
                if neighbours[f]:
                    seen[f] = sum(before[g] for g in neighbours[f]) / len(neighbours[f])
        self.seen_face, self.inside_px, self.outline_px = seen, inside, outline
        yield

    @staticmethod
    def _scan(F, D, r0, r1, front, proj, scale, tol, pair_keys, pair_edges, factor, inside, outline, w, owner, flat, indptr, sizes, edge_key, sorted_edge):
        n = F.shape[0]
        rows = np.arange(r0, r1)
        for kind in (0, 1):  # 0: next row (iu + 1), 1: next column (iv + 1); the kernel reads (iu + 1, iv) then (iu, iv + 1)
            A = F[r0:r1]
            DA = D[r0:r1]
            if kind == 0:
                if r1 < n:
                    B, DB = F[r0 + 1 : r1 + 1], D[r0 + 1 : r1 + 1]
                else:
                    B = np.vstack((F[r0 + 1 : r1], np.full((1, n), -1)))
                    DB = np.vstack((D[r0 + 1 : r1], np.zeros((1, n), dtype=np.float32)))
            else:
                B = np.hstack((F[r0:r1, 1:], np.full((r1 - r0, 1), -1)))
                DB = np.hstack((D[r0:r1, 1:], np.zeros((r1 - r0, 1), dtype=np.float32)))
            ri, ci = np.nonzero((A >= 0) & (B != A))
            if len(ri) == 0:
                continue
            a, b = A[ri, ci], B[ri, ci]
            da, db = DA[ri, ci], DB[ri, ci]
            nbv = b >= 0
            near = nbv & (np.abs(da - db) <= tol)
            both = near & front[a] & front[np.where(nbv, b, 0)]
            if both.any():
                fa, fb = a[both].astype(np.uint64), b[both].astype(np.uint64)
                pk = (np.minimum(fa, fb) << np.uint64(32)) | np.maximum(fa, fb)
                at = np.searchsorted(pair_keys, pk)
                ok = (at < len(pair_keys)) & (pair_keys[np.minimum(at, max(len(pair_keys) - 1, 0))] == pk) if len(pair_keys) else np.zeros(len(pk), bool)
                np.add.at(inside, pair_edges[at[ok]], w * factor[pair_edges[at[ok]]])
            rest = ~both
            if not rest.any():
                continue
            a, b, da, db, ri, ci, nbv = a[rest], b[rest], da[rest], db[rest], ri[rest], ci[rest], nbv[rest]
            mine = ~nbv | (da <= db)
            who = np.where(mine, a, b)
            cu = np.where(mine, rows[ri], rows[ri] + (kind == 0))
            cv = np.where(mine, ci, ci + (kind == 1))
            cd = np.where(mine, da, db).astype(np.float64)
            qx, qy, qz = cu.astype(np.float64), cv.astype(np.float64), cd * scale
            best = np.full(len(who), np.inf)
            pick = np.full(len(who), -1, dtype=np.int64)
            for c in range(int(sizes[who].max())):
                valid = c < sizes[who]
                s = indptr[who] + c
                s2 = np.where(c + 1 < sizes[who], s + 1, indptr[who])
                va, vb = flat[np.where(valid, s, 0)], flat[np.where(valid, s2, 0)]
                pa, pb = proj[va].copy(), proj[vb].copy()
                pa[:, 2] *= scale
                pb[:, 2] *= scale
                ab = pb - pa
                aq = np.stack((qx, qy, qz), axis=1) - pa
                len2 = (ab * ab).sum(axis=1)
                t = np.clip(np.divide((aq * ab).sum(axis=1), len2, out=np.zeros_like(len2), where=len2 > 0), 0.0, 1.0)
                r = aq - ab * t[:, None]
                d2 = (r * r).sum(axis=1)
                lo, hi = np.minimum(va, vb).astype(np.uint64), np.maximum(va, vb).astype(np.uint64)
                ek = (lo << np.uint64(32)) | hi
                at = np.searchsorted(edge_key[sorted_edge], ek)
                better = valid & (d2 < best)
                best = np.where(better, d2, best)
                pick = np.where(better, sorted_edge[np.minimum(at, len(sorted_edge) - 1)], pick)
            got = pick >= 0
            np.add.at(outline, pick[got], w * factor[pick[got]])
