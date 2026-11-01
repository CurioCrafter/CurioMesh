# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Auto Segments: you say how much of the view may change, and it finds the fewest faces that hold it.

Each loose piece is reduced to several sizes and looked at from six sides, with one fixed pixel size for the whole
object, so a bolt is judged by how big it looks on the object, not blown up to fill the picture. Faces then go where
they remove the most visible change (`budget.allocate`), with a limit no single piece may pass. The pieces are paired
into quads, the whole result is looked at, and the sharing is redone until it lands on what was asked for.

Blender's Decimate has to run on the main thread, so the search is a generator that runs in short slices from a timer,
which keeps Blender usable. Only the pairing, which is plain NumPy, goes to a worker thread.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import threading
import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .triadq import patch
from .triadq.flatquad import _face_frames
from .triadq.flow import faces_that_gain, seek_features, snap_new_vertices, snap_rows, split_faces
from .triadq.budget import Level, allocate
from .triadq.flatquad import pair_and_close
from .triadq.types import MeshData

VIEW_PIXELS = 2048  # the whole object's size in the camera's pixels: sets the smallest detail that counts
MAX_PIXELS = 192  # a side of one picture, so one huge piece can't take minutes per look
KEEP_TRIANGLES = 24  # pieces this small stay as they are: there's nothing to gain, and thousands of them to look at
LEVELS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # shares of a piece's triangles to try
PIECE_LIMIT = 3.0  # no piece may change by more than this many times the amount asked for
GOOD_ENOUGH = 0.25  # stop trying bigger sizes of a piece once its change is this share of the amount asked for
ROUNDS = 4  # tries at landing on the amount asked for
ROWS_SMALLEST = 32  # Clean Rows: a piece due fewer quads than this is paired instead
ROWS_START = 0.04  # Clean Rows: first try, as a share of the source's triangles
ROWS_MOST = 0.8  # Clean Rows: most faces, as a share of the source's triangles
ROWS_FAR = 1.0  # Clean Rows: a change this many times the amount asked for gets a denser layout, not local density
ROWS_DENSER = 0  # Clean Rows: most rounds of density where the camera finds a difference
ROWS_FIX_SHARE = 0.6  # Clean Rows: share of the wrong pixels on a quad that splitting it is assumed to fix
ROWS_SECONDS = 30.0  # Clean Rows: QuadriFlow's time limit per piece, plus...
ROWS_SECONDS_PER_QUAD = 0.01  # ...this much for each quad asked for
FLAT_SNAP_PIXELS = 0.25  # Balanced: a face whose new vertices would move less than this many pixels stays whole
NARROW_TRIES = 2  # extra tries with a new size in between, when the sizes on offer land well short of it
TURNED_DEG = 20.0
MOST_RAYS = 20_000_000  # one look at every piece, so a model of thousands of parts still finishes in minutes
RAYS_PER_YIELD = 2000
SHOW_EVERY = 1.0  # seconds between preview updates


def pictures(points: np.ndarray, polys, center: np.ndarray, half: float, pixels: int):
    """Generator: six exact orthographic pictures, one ray per pixel. Returns [(hit, normal, depth, face seen)] when exhausted."""

    tree = BVHTree.FromPolygons(points.tolist(), [list(p) for p in polys], all_triangles=False, epsilon=0.0)
    grid = (((np.arange(pixels) + 0.5) / pixels * 2.0 - 1.0) * half).tolist()
    out = []
    since_yield = 0
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (1.0, -1.0):
            hit = np.zeros((pixels, pixels), dtype=bool)
            normal = np.zeros((pixels, pixels, 3))
            depth = np.zeros((pixels, pixels))
            seen = np.full((pixels, pixels), -1, dtype=np.int64)
            direction = [0.0, 0.0, 0.0]
            direction[axis] = -sign
            direction = Vector(direction)
            origin = [0.0, 0.0, 0.0]
            origin[axis] = float(center[axis]) + sign * 2.0 * half
            for i, x in enumerate(grid):
                origin[u] = float(center[u]) + x
                for j, y in enumerate(grid):
                    origin[v] = float(center[v]) + y
                    found = tree.ray_cast(Vector(origin), direction)
                    if found[0] is not None:
                        hit[i, j] = True
                        normal[i, j] = found[1]
                        depth[i, j] = found[3]
                        seen[i, j] = found[2]
                since_yield += pixels
                if since_yield >= RAYS_PER_YIELD:
                    since_yield = 0
                    yield
            out.append((hit, normal, depth, seen))
    return out


def picture_counts(reference, other, pixel: float) -> tuple[int, int]:
    """(pixels a viewer would see as different in outline, direction or depth, pixels showing the object)."""

    wrong, total = 0, 0
    limit = math.cos(math.radians(TURNED_DEG))
    for (hit_a, normal_a, depth_a, _seen_a), (hit_b, normal_b, depth_b, _seen_b) in zip(reference, other):
        both = hit_a & hit_b
        turned = np.abs((normal_a[both] * normal_b[both]).sum(axis=1)) < limit  # winding isn't shape
        moved = np.abs(depth_a[both] - depth_b[both]) > 2.0 * pixel
        wrong += int((turned | moved).sum()) + int((hit_a ^ hit_b).sum())
        total += int((hit_a | hit_b).sum())
    return wrong, total


def wrong_pixels_per_face(reference, other, pixel: float, face_count: int) -> np.ndarray:
    """How many pixels the viewer would see as different, on each face of `other`. A pixel it misses entirely isn't counted."""

    wrong = np.zeros(face_count, dtype=np.int64)
    limit = math.cos(math.radians(TURNED_DEG))
    for (hit_a, normal_a, depth_a, _seen_a), (hit_b, normal_b, depth_b, seen_b) in zip(reference, other):
        both = hit_a & hit_b
        bad = np.abs((normal_a * normal_b).sum(axis=2)) < limit
        bad |= np.abs(depth_a - depth_b) > 2.0 * pixel
        rows = seen_b[(both & bad) | (hit_b & ~hit_a)]
        np.add.at(wrong, rows[rows >= 0], 1)
    return wrong


def _decimate(context, points, polys, kinds, ratio) -> tuple[np.ndarray, list[tuple[int, ...]], list[str]]:
    mesh = bpy.data.meshes.new("CurioMesh_Piece")
    mesh.from_pydata(points.tolist(), [], polys)
    mesh.polygons.foreach_set("material_index", np.asarray([int(k) for k in kinds], dtype=np.int32))
    mesh.update()
    obj = bpy.data.objects.new("CurioMesh_Piece", mesh)
    context.scene.collection.objects.link(obj)
    try:
        obj.modifiers.new("Reduce", "DECIMATE").ratio = ratio
        evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
        reduced = evaluated.to_mesh()
        co = np.empty(len(reduced.vertices) * 3, dtype=np.float64)
        reduced.vertices.foreach_get("co", co)
        faces = [tuple(p.vertices) for p in reduced.polygons]
        materials = [str(p.material_index) for p in reduced.polygons]
        evaluated.to_mesh_clear()
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
    return co.reshape(-1, 3), faces, materials


def _picture_sizes(sizes: np.ndarray) -> np.ndarray:
    """Pixels per side for pieces of these sizes (the whole object is 1.0), coarser for a model with very many parts."""

    view, most = float(VIEW_PIXELS), float(MAX_PIXELS)
    while True:
        pixels = np.clip(np.ceil(sizes * view), 8, most).astype(np.int64)
        if 6 * int((pixels * pixels).sum()) <= MOST_RAYS or most <= 16:
            return pixels
        view *= 0.85
        most *= 0.85


class AutoJob:
    """One Auto Segments run. Call `step` from a timer until `busy` is False; `output` holds the latest result to show."""

    def __init__(self, context, co, faces, materials, detail: float, style: str = "DETAIL"):
        self.context = context
        self.co = np.asarray(co, dtype=np.float64)
        self.faces = list(faces)
        self.materials = [str(int(m)) for m in materials]
        self.detail = float(detail)
        self.style = style
        self._tree = None
        self._worker: subprocess.Popen | None = None
        self._refused = 0
        self.busy = True
        self.current = None
        self.output = None
        self.output_id = 0
        self.warnings: list[str] = []
        self.report: dict | None = None
        self.progress = (0, 0)
        self.stage = "Looking at the pieces..."
        self.started = time.perf_counter()
        self.shown_at = 0.0
        self._stopped = False
        self._pair_thread: threading.Thread | None = None
        self._paired = None
        self.pieces: list[np.ndarray] = []
        self.sizes: list[list[tuple]] = []  # per piece, smallest first: (points, polys, materials)
        self.levels: list[list[Level]] = []
        self.seen: list[float] = []
        self.picked: list[int] = []
        self._search = self._run_rows() if style == "ROWS" else self._run()

    def stop(self) -> None:
        self._stopped = True
        self.busy = False
        self._kill_worker()

    def step(self, seconds: float = 0.04) -> None:
        """Do the next slice of work on this (the main) thread."""
        if not self.busy:
            return
        ends = time.perf_counter() + seconds
        try:
            while time.perf_counter() < ends:
                next(self._search)
        except StopIteration:
            self.busy = False
        except Exception as exc:  # a timer has nobody above it to report to
            self.warnings = [f"Auto Segments failed: {exc}"]
            self.busy = False

    def wait(self, timeout: float | None = None) -> bool:
        ends = None if timeout is None else time.perf_counter() + timeout
        while self.busy and (ends is None or time.perf_counter() < ends):
            self.step(1.0)
            if self._pair_thread is not None and self._pair_thread.is_alive():
                self._pair_thread.join(0.05)
        return not self.busy

    def _piece_arrays(self, ids: np.ndarray):
        used, flat = np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64), return_inverse=True)
        polys, at = [], 0
        for f in ids.tolist():
            size = len(self.faces[f])
            polys.append(tuple(int(v) for v in flat[at : at + size]))
            at += size
        return self.co[used], polys, [self.materials[f] for f in ids.tolist()]

    def _measure_piece(self, ids: np.ndarray, pixels: int):
        """Generator: this piece at each size, smallest first, with how much each changes it."""
        points, polys, kinds = self._piece_arrays(ids)
        whole = (points, polys, kinds)
        triangles = sum(len(p) - 2 for p in polys)
        if triangles <= KEEP_TRIANGLES:
            self.sizes.append([whole])
            self.levels.append([Level(len(polys), 0.0)])
            self.seen.append(0.0)
            return
        low, high = points.min(axis=0), points.max(axis=0)
        center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
        pixel = 2.0 * half / pixels
        reference = yield from pictures(points, polys, center, half, pixels)
        sizes, levels, seen = [], [], 0.0
        for share in LEVELS:
            if share * triangles < 12.0:
                continue
            yield
            reduced = _decimate(self.context, points, polys, kinds, share)
            view = yield from pictures(reduced[0], reduced[1], center, half, pixels)
            wrong, shown = picture_counts(reference, view, pixel)
            area = pixel * pixel
            seen = max(seen, shown * area)
            sizes.append(reduced)
            levels.append(Level(len(reduced[1]), wrong * area))
            if wrong <= GOOD_ENOUGH * self.detail / 100.0 * shown:
                break
        sizes.append(whole)
        levels.append(Level(len(polys), 0.0))
        self.sizes.append(sizes)
        self.levels.append(levels)
        self.seen.append(seen)

    def _combined(self) -> tuple[np.ndarray, list[tuple[int, ...]], list[str], np.ndarray]:
        """Every piece at its picked size, and any not measured yet as it is."""
        all_points, all_faces, all_kinds, parts = [], [], [], []
        offset = 0
        for index, ids in enumerate(self.pieces):
            if index < len(self.sizes):
                points, polys, kinds = self.sizes[index][self.picked[index] if index < len(self.picked) else 0]
            else:
                points, polys, kinds = self._piece_arrays(ids)
            all_points.append(points)
            all_faces += [tuple(v + offset for v in p) for p in polys]
            all_kinds += kinds
            parts += [index] * len(polys)
            offset += len(points)
        return np.vstack(all_points), all_faces, all_kinds, np.asarray(parts, dtype=np.int64)

    def _show(self, final=None) -> None:
        if final is None:
            co, faces, kinds, parts = self._combined()
            count = len(faces)
            new = np.ones(count, bool)
        else:
            co, faces, kinds, parts = final
            count = len(faces)
            new = np.ones(count, bool)
        self.output = {"co": co, "faces": faces, "new": new, "locked": np.zeros(count, bool), "mat": np.asarray([int(k) if str(k).isdigit() else 0 for k in kinds], dtype=np.int32), "parts": parts}
        self.output_id += 1
        self.shown_at = time.perf_counter()

    def _pair(self, co, faces, kinds) -> None:
        try:
            coarse = MeshData(co, faces, kinds)
            if self._tree is None:
                self._tree = BVHTree.FromPolygons(self.co.tolist(), [list(f) for f in self.faces], all_triangles=False, epsilon=0.0)
            find = self._tree.find_nearest
            if self.style == "BALANCED":
                paired = pair_and_close(coarse, seed_count=2, force_quads=False)[0]
                span = float(np.ptp(self.co, axis=0).max()) * 1.1 / MAX_PIXELS
                result = split_faces(paired, faces_that_gain(paired, find, FLAT_SNAP_PIXELS * span))
            else:
                result = pair_and_close(coarse, seed_count=2)[0]
            result = snap_new_vertices(result, coarse.vertices, find)
            self._paired = result
        except Exception as exc:  # a thread has nobody above it to report to
            self.warnings = [f"Auto Segments failed: {exc}"]

    def _attempt(self, reference, center, half, pixel, number: int):
        """Generator: pair the picked sizes and look at the result. Returns its loss, or None if stopped or failed."""
        self.stage = f"Pairing into quads, try {number}..."
        co, faces, kinds, _parts = self._combined()
        self._paired = None
        self._pair_thread = threading.Thread(target=self._pair, args=(co, faces, kinds), daemon=True, name="CurioMeshPair")
        self._pair_thread.start()
        while self._pair_thread.is_alive():
            if self._stopped:
                return None
            yield
        if self._paired is None:
            return None
        result = self._paired
        points = np.asarray(result.vertices, dtype=np.float64)
        out_faces = [tuple(face) for face in result.faces]
        self.stage = f"Checking the result, try {number}..."
        view = yield from pictures(points, out_faces, center, half, MAX_PIXELS)
        wrong, shown = picture_counts(reference, view, pixel)
        loss = 100.0 * wrong / max(shown, 1)
        fits = loss <= 1.1 * self.detail
        # Prefer anything that fits, then the fewest faces. With nothing fitting yet, the least change.
        rank = (not fits, len(out_faces) if fits else loss)
        if self._best is None or rank < self._best[0]:
            self._best = (rank, points, out_faces, list(result.face_materials), loss, list(self.picked))
        return loss

    def _narrow(self) -> bool:
        """Add a size halfway to the next smaller one on the biggest piece, and pick it. False if there's no need or no room."""
        if self._best is None or self._best[0][0] or self._best[4] >= 0.9 * self.detail:
            return False
        picked = self._best[5]
        faces = [self.levels[i][k].faces if len(self.levels[i]) > 1 else 0 for i, k in enumerate(picked)]
        piece = int(np.argmax(faces))
        at = picked[piece]
        if faces[piece] < KEEP_TRIANGLES:
            return False
        high = self.levels[piece][at]
        low = self.levels[piece][at - 1] if at > 0 else Level(high.faces // 2, 2.0 * high.loss)  # the smallest size may still be too big
        if high.faces < 1.2 * low.faces:
            return False
        points, polys, kinds = self.sizes[piece][-1]
        triangles = sum(len(p) - 2 for p in polys)
        reduced = _decimate(self.context, points, polys, kinds, math.sqrt(high.faces * low.faces) / triangles)
        self.sizes[piece].insert(at, reduced)
        self.levels[piece].insert(at, Level(len(reduced[1]), 0.5 * (high.loss + low.loss)))
        picked[piece] = at + 1  # the best so far moved up one
        self.picked = list(picked)
        self.picked[piece] = at
        return True

    def _run(self):
        fa, fb = patch.face_adjacency(self.faces)
        shell = np.unique(patch._labels(len(self.faces), fa, fb), return_inverse=True)[1]
        order = np.argsort(shell, kind="stable")
        starts = np.concatenate(([0], np.flatnonzero(shell[order][1:] != shell[order][:-1]) + 1, [len(order)]))
        self.pieces = sorted((order[a:b] for a, b in zip(starts[:-1], starts[1:])), key=len, reverse=True)
        extent = float(np.ptp(self.co, axis=0).max()) or 1.0
        sizes = np.asarray([2.0 * 0.55 * float(np.ptp(self.co[np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))], axis=0).max()) for ids in self.pieces]) / extent
        picture_sizes = _picture_sizes(sizes).tolist()
        self._show()
        for index, ids in enumerate(self.pieces):
            self.progress = (index, len(self.pieces))
            self.stage = f"Looking at piece {index + 1} of {len(self.pieces)}"
            yield from self._measure_piece(ids, picture_sizes[index])
            yield
            if self._stopped:
                return
            if time.perf_counter() - self.shown_at > SHOW_EVERY:
                self._show()
        self.progress = (len(self.pieces), len(self.pieces))
        low, high = self.co.min(axis=0), self.co.max(axis=0)
        center, half = (low + high) / 2.0, extent * 0.55
        pixel = 2.0 * half / MAX_PIXELS
        reference = yield from pictures(self.co, self.faces, center, half, MAX_PIXELS)
        total_seen = sum(self.seen)
        asked = self.detail / 100.0
        allowed = asked * total_seen
        tried: set[tuple[int, ...]] = set()
        self._best = None
        for round_index in range(ROUNDS):
            self.picked = allocate(self.levels, allowed, PIECE_LIMIT * asked, self.seen)
            if tuple(self.picked) in tried:
                break
            tried.add(tuple(self.picked))
            loss = yield from self._attempt(reference, center, half, pixel, round_index + 1)
            if loss is None:
                return
            if abs(loss - self.detail) <= 0.1 * self.detail:
                break
            allowed *= min(max(self.detail / max(loss, 0.01), 0.3), 4.0)
        for round_index in range(NARROW_TRIES):  # sizes double, so a big piece can land far short: try a size in between
            if not self._narrow():
                break
            yield
            loss = yield from self._attempt(reference, center, half, pixel, ROUNDS + round_index + 1)
            if loss is None:
                return
        _rank, points, out_faces, materials, loss, _picked = self._best
        out_fa, out_fb = patch.face_adjacency(out_faces)
        parts = np.unique(patch._labels(len(out_faces), out_fa, out_fb), return_inverse=True)[1]
        quads = sum(1 for face in out_faces if len(face) == 4)
        self.report = {"faces": len(out_faces), "quads": quads, "tris": sum(1 for face in out_faces if len(face) == 3), "pieces": len(self.pieces), "loss": loss, "seconds": time.perf_counter() - self.started}
        self._show((points, out_faces, materials, parts))

    def _kill_worker(self) -> None:
        if self._worker is not None:
            self._worker.kill()
            self._worker.wait()
            self._worker = None

    def _rows_once(self, areas, target: int):
        """Generator: QuadriFlow on every piece big enough, in a second Blender, snapped to the source. Returns (points, faces, materials)."""

        total = float(areas.sum()) or 1.0
        folder = tempfile.mkdtemp(prefix="curiomesh_rows_")
        try:
            sizes = np.asarray([len(ids) for ids in self.pieces], dtype=np.float64)
            wants = (target * 0.5 * (areas / total + sizes / sizes.sum())).astype(np.int64)  # a small part with many faces is usually detail
            jobs = [k for k, want in enumerate(wants.tolist()) if want >= ROWS_SMALLEST]
            for k in jobs:
                points, polys, _kinds = self._piece_arrays(self.pieces[k])
                np.savez(os.path.join(folder, f"in_{k}.npz"), co=points, target=wants[k], lens=np.asarray([len(p) for p in polys], dtype=np.int64), flat=np.asarray([v for p in polys for v in p], dtype=np.int64))
                yield
            at = 0
            while at < len(jobs):
                self._worker = subprocess.Popen(
                    [bpy.app.binary_path, "--background", "--factory-startup", "--python", os.path.join(os.path.dirname(__file__), "rows_worker.py"), "--", folder, str(jobs[at]), str(jobs[-1] + 1)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                since = time.perf_counter()
                while at < len(jobs):
                    k = jobs[at]
                    self.stage = f"Laying rows on piece {at + 1} of {len(jobs)}"
                    self.progress = (at, len(jobs))
                    if os.path.exists(os.path.join(folder, f"out_{k}.npz")):
                        at += 1
                        since = time.perf_counter()
                        continue
                    if self._stopped:
                        self._kill_worker()
                        return None
                    if time.perf_counter() - since > ROWS_SECONDS + ROWS_SECONDS_PER_QUAD * float(wants[k]) or self._worker.poll() is not None:
                        if self._worker.poll() is None or not os.path.exists(os.path.join(folder, f"out_{k}.npz")):
                            print(f"CURIOROWS piece {k} gave up, {len(self.pieces[k])} faces, asked {wants[k]}", flush=True)
                            self._kill_worker()
                            self._refused += 1
                            at += 1
                            break
                    time.sleep(0.005)
                    yield
            self._kill_worker()
            all_points, all_faces, all_kinds = [], [], []
            offset = 0
            leftovers = []
            for k, ids in enumerate(self.pieces):
                points, polys, kinds = self._piece_arrays(ids)
                path = os.path.join(folder, f"out_{k}.npz")
                rows = np.load(path) if os.path.exists(path) else None
                if rows is None or not len(rows["lens"]):
                    triangles = sum(len(p) - 2 for p in polys)
                    want = max(KEEP_TRIANGLES, 2 * int(wants[k]))
                    if triangles > want:
                        points, polys, kinds = _decimate(self.context, points, polys, kinds, want / triangles)
                    leftovers.append((points, polys, kinds))
                    yield
                    continue
                flat, faces, cursor = rows["flat"].tolist(), [], 0
                for size in rows["lens"].tolist():
                    faces.append(tuple(flat[cursor : cursor + size]))
                    cursor += size
                tree = BVHTree.FromPolygons(points.tolist(), [list(p) for p in polys], all_triangles=False, epsilon=0.0)
                triangles = np.asarray([(q[0], q[i], q[i + 1]) for q in polys for i in range(1, len(q) - 1)], dtype=np.int64)

                def sharpen(moved, points=points, triangles=triangles):
                    grid = KDTree(len(moved))
                    for i, co in enumerate(moved.tolist()):
                        grid.insert(co, i)
                    grid.balance()
                    return seek_features(moved, points, triangles, lambda at: grid.find(at)[1])

                moved, quads = yield from self._in_thread(lambda: snap_rows(rows["co"], faces, tree.find_nearest, sharpen=sharpen))
                all_points.append(moved)
                all_faces += [tuple(v + offset for v in q) for q in quads]
                centers = np.asarray([moved[list(q)].mean(axis=0) for q in quads]).tolist()
                all_kinds += [kinds[tree.find_nearest(c)[2]] for c in centers]
                offset += len(moved)
                yield
        finally:
            self._kill_worker()
            shutil.rmtree(folder, ignore_errors=True)
        if leftovers:
            pts, fcs, knd, cursor = [], [], [], 0
            for points, polys, kinds in leftovers:
                pts.append(points)
                fcs += [tuple(v + cursor for v in p) for p in polys]
                knd += kinds
                cursor += len(points)
            closed = pair_and_close(MeshData(np.vstack(pts), fcs, knd), seed_count=1)[0]
            all_points.append(np.asarray(closed.vertices))
            all_faces += [tuple(v + offset for v in f) for f in closed.faces]
            all_kinds += list(closed.face_materials)
        return np.vstack(all_points), all_faces, all_kinds

    def _run_rows(self):
        """Clean Rows: order first. Lands near the amount asked for by a denser layout, then by more density where the camera sees a difference."""

        fa, fb = patch.face_adjacency(self.faces)
        shell = np.unique(patch._labels(len(self.faces), fa, fb), return_inverse=True)[1]
        order = np.argsort(shell, kind="stable")
        starts = np.concatenate(([0], np.flatnonzero(shell[order][1:] != shell[order][:-1]) + 1, [len(order)]))
        self.pieces = sorted((order[a:b] for a, b in zip(starts[:-1], starts[1:])), key=len, reverse=True)
        face_areas = _face_frames(self.co, self.faces)[1]
        areas = np.asarray([face_areas[ids].sum() for ids in self.pieces])
        self._show()
        extent = float(np.ptp(self.co, axis=0).max()) or 1.0
        low, high = self.co.min(axis=0), self.co.max(axis=0)
        center, half = (low + high) / 2.0, extent * 0.55
        pixel = 2.0 * half / MAX_PIXELS
        reference = yield from pictures(self.co, self.faces, center, half, MAX_PIXELS)
        source = sum(len(f) - 2 for f in self.faces)
        most = int(ROWS_MOST * source)
        target = min(max(int(ROWS_START * source), 256), most)
        tree = BVHTree.FromPolygons(self.co.tolist(), [list(f) for f in self.faces], all_triangles=False, epsilon=0.0)
        best = None

        def judge(points, out_faces, kinds, loss):
            nonlocal best
            fits = loss <= 1.1 * self.detail
            rank = (not fits, len(out_faces) if fits else loss)  # as in Keep Detail: fits first, then the fewest faces
            if best is None or rank < best[0]:
                best = (rank, points, out_faces, kinds, loss)

        for _round in range(3):  # a layout near enough that local density can finish the job
            built = yield from self._rows_once(areas, target)
            if built is None:
                return
            points, out_faces, kinds = built
            self.stage = "Checking the result..."
            view = yield from pictures(points, out_faces, center, half, MAX_PIXELS)
            wrong, shown = picture_counts(reference, view, pixel)
            loss = 100.0 * wrong / max(shown, 1)
            print(f"CURIOTRY layout {target} faces {len(out_faces)} loss {loss:.2f}", flush=True)
            judge(points, out_faces, kinds, loss)
            aim = 1.0 if loss < 0.9 * self.detail else ROWS_FAR if loss > ROWS_FAR * self.detail else None
            if aim is None:
                break
            wanted = min(max(int(target * (loss / (aim * self.detail)) ** 2), 64), most)  # change falls about as the square root of the faces
            if wanted == target:
                break
            target = wanted
        self.stage = "Adding density where it differs..."
        for _round in range(ROWS_DENSER):
            if loss <= 1.1 * self.detail or len(out_faces) >= most:
                break
            per_face = wrong_pixels_per_face(reference, view, pixel, len(out_faces))
            order_by = np.argsort(-per_face, kind="stable")
            to_fix = (loss - self.detail) / 100.0 * shown / ROWS_FIX_SHARE
            count = int(np.searchsorted(np.cumsum(per_face[order_by]), to_fix)) + 1
            marked = set(order_by[:count][per_face[order_by[:count]] > 0].tolist())
            if not marked:
                break
            mesh = yield from self._in_thread(lambda: split_faces(MeshData(points, out_faces, kinds), marked))
            if mesh is None:
                break
            mesh = yield from self._in_thread(lambda: snap_new_vertices(mesh, points, tree.find_nearest))
            points, out_faces, kinds = np.asarray(mesh.vertices), [tuple(f) for f in mesh.faces], list(mesh.face_materials)
            view = yield from pictures(points, out_faces, center, half, MAX_PIXELS)
            wrong, shown = picture_counts(reference, view, pixel)
            loss = 100.0 * wrong / max(shown, 1)
            print(f"CURIOTRY denser marked {len(marked)} faces {len(out_faces)} loss {loss:.2f}", flush=True)
            judge(points, out_faces, kinds, loss)
        _rank, points, out_faces, kinds, loss = best
        if self._refused:
            self.warnings = [f"QuadriFlow gave up on {self._refused} pieces. They were paired instead."]
        out_fa, out_fb = patch.face_adjacency(out_faces)
        parts = np.unique(patch._labels(len(out_faces), out_fa, out_fb), return_inverse=True)[1]
        quads = sum(1 for face in out_faces if len(face) == 4)
        self.report = {"faces": len(out_faces), "quads": quads, "tris": len(out_faces) - quads, "pieces": len(self.pieces), "loss": loss, "seconds": time.perf_counter() - self.started}
        self._show((points, out_faces, kinds, parts))

    def _in_thread(self, work):
        """Generator: run plain NumPy work on a worker thread, so the timer's slices stay short. Returns its result, or None if it failed."""

        box = []

        def run():
            try:
                box.append(work())
            except ValueError as exc:
                self.warnings = [f"Clean Rows: {exc}"]

        thread = threading.Thread(target=run, daemon=True, name="CurioMeshRows")
        thread.start()
        while thread.is_alive():
            if self._stopped:
                return None
            yield
        return box[0] if box else None

    def status(self) -> str:
        if self.report is None:
            return self.stage
        r = self.report
        return f"{r['faces']:,} faces, {r['tris']} triangles, {r['pieces']:,} pieces | Camera: {r['loss']:.1f}% of the view differs. {r['seconds']:.0f} s"
