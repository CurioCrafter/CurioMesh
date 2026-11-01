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
import threading
import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .triadq import patch
from .triadq.budget import Level, allocate, best_levels, hottest, vertex_heat
from .triadq.flatquad import pair_and_close
from .triadq.shading import blended_normals, corner_normals, corner_table
from .triadq.types import MeshData

VIEW_PIXELS = 2048  # the whole object's size in the camera's pixels: sets the smallest detail that counts
MAX_PIXELS = 192  # a side of one picture, so one huge piece can't take minutes per look
KEEP_TRIANGLES = 24  # pieces this small stay as they are: there's nothing to gain, and thousands of them to look at
LEVELS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # shares of a piece's triangles to try
PIECE_LIMIT = 3.0  # no piece may change by more than this many times the amount asked for
GOOD_ENOUGH = 0.25  # stop trying bigger sizes of a piece once its change is this share of the amount asked for
FINER_TRIANGLES = 200  # pieces smaller than this have too few faces for a size between two others to matter
LOCKED_SHARE = 0.2  # vertices held in the first pass, as a share of those the piece ends with
HELD_SIZE = 1.5  # the first pass stops at this many times the faces asked for
ROUNDS = 4  # tries at landing on the amount asked for
TURNED_DEG = 20.0
SOFT_DEG = 10.0  # shallow folds in cloth only turn the shading this much
MOST_RAYS = 20_000_000  # one look at every piece, so a model of thousands of parts still finishes in minutes
RAYS_PER_YIELD = 2000
SHOW_EVERY = 1.0  # seconds between preview updates


def pictures(points: np.ndarray, polys, center: np.ndarray, half: float, pixels: int):
    """Generator: six exact orthographic pictures, one ray per pixel. Returns [(hit, normal, depth, face)] when exhausted."""

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
            face = np.zeros((pixels, pixels), dtype=np.int32)
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
                        face[i, j] = found[2]
                        depth[i, j] = found[3]
                since_yield += pixels
                if since_yield >= RAYS_PER_YIELD:
                    since_yield = 0
                    yield
            out.append((hit, normal, depth, face))
    return out


def picture_counts(reference, other, pixel: float) -> tuple[int, int]:
    """(pixels a viewer would see as different in outline, direction or depth, pixels showing the object)."""

    wrong, total = 0, 0
    limit = math.cos(math.radians(TURNED_DEG))
    for (hit_a, normal_a, depth_a, _), (hit_b, normal_b, depth_b, _) in zip(reference, other):
        both = hit_a & hit_b
        turned = np.abs((normal_a[both] * normal_b[both]).sum(axis=1)) < limit  # winding isn't shape
        moved = np.abs(depth_a[both] - depth_b[both]) > 2.0 * pixel
        wrong += int((turned | moved).sum()) + int((hit_a ^ hit_b).sum())
        total += int((hit_a | hit_b).sum())
    return wrong, total


def shaded(picture_set, points: np.ndarray, polys, center: np.ndarray, half: float, pixels: int) -> list[np.ndarray]:
    """Each picture's normals as smooth shading shows them, blended from the vertices at each hit."""

    normals = corner_normals(points, polys)
    table = corner_table(polys)
    grid = ((np.arange(pixels) + 0.5) / pixels * 2.0 - 1.0) * half
    out = []
    views = iter(picture_set)
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (1.0, -1.0):
            hit, _normal, depth, face = next(views)
            i, j = np.nonzero(hit)
            at = np.empty((len(i), 3))
            at[:, u] = center[u] + grid[i]
            at[:, v] = center[v] + grid[j]
            at[:, axis] = center[axis] + sign * (2.0 * half - depth[i, j])
            blended = np.zeros((pixels, pixels, 3))
            blended[i, j] = blended_normals(points, table, normals, face[i, j], at)
            out.append(blended)
    return out


def shading_counts(reference, other, shaded_a, shaded_b, degrees: float = TURNED_DEG) -> tuple[int, int]:
    """(pixels showing both whose smooth-shaded normal turned by more than `degrees`, pixels showing both)."""

    limit = math.cos(math.radians(degrees))
    wrong, total = 0, 0
    for (hit_a, *_), (hit_b, *_), normal_a, normal_b in zip(reference, other, shaded_a, shaded_b):
        both = hit_a & hit_b
        wrong += int((np.abs((normal_a[both] * normal_b[both]).sum(axis=1)) < limit).sum())
        total += int(both.sum())
    return wrong, total


def wrong_points(reference, other, center: np.ndarray, half: float, pixels: int, pixel: float) -> np.ndarray:
    """Where on the object the pixels `picture_counts` calls wrong are, as seen in each of the two pictures."""

    limit = math.cos(math.radians(TURNED_DEG))
    grid = ((np.arange(pixels) + 0.5) / pixels * 2.0 - 1.0) * half
    found = []
    views = iter(zip(reference, other))
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (1.0, -1.0):
            (hit_a, normal_a, depth_a, _), (hit_b, normal_b, depth_b, _) = next(views)
            turned = np.abs((normal_a * normal_b).sum(axis=2)) < limit
            moved = np.abs(depth_a - depth_b) > 2.0 * pixel
            wrong = (hit_a & hit_b & (turned | moved)) | (hit_a ^ hit_b)
            for hit, depth in ((hit_a, depth_a), (hit_b, depth_b)):
                i, j = np.nonzero(wrong & hit)
                at = np.empty((len(i), 3))
                at[:, u] = center[u] + grid[i]
                at[:, v] = center[v] + grid[j]
                at[:, axis] = center[axis] + sign * (2.0 * half - depth[i, j])
                found.append(at)
    return np.vstack(found)


def _decimate(context, points, polys, kinds, ratio, keep=None) -> tuple[np.ndarray, list[tuple[int, ...]], list[str]]:
    """Reduce to `ratio`. Vertices in the mask `keep` go last: Decimate's group is a lock, at any strength."""

    mesh = bpy.data.meshes.new("CurioMesh_Piece")
    mesh.from_pydata(points.tolist(), [], polys)
    mesh.polygons.foreach_set("material_index", np.asarray([int(k) for k in kinds], dtype=np.int32))
    mesh.update()
    obj = bpy.data.objects.new("CurioMesh_Piece", mesh)
    context.scene.collection.objects.link(obj)
    try:
        reduce = obj.modifiers.new("Reduce", "DECIMATE")
        reduce.ratio = ratio
        if keep is not None:
            free = obj.vertex_groups.new(name="Free")  # weight 1 collapses, anything outside the group counts as 0
            free.add(np.flatnonzero(~keep).tolist(), 1.0, "REPLACE")
            reduce.vertex_group = free.name
            reduce.vertex_group_factor = 1000.0
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

    def __init__(self, context, co, faces, materials, detail: float):
        self.context = context
        self.co = np.asarray(co, dtype=np.float64)
        self.faces = list(faces)
        self.materials = [str(int(m)) for m in materials]
        self.detail = float(detail)
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
        self._search = self._run()

    def stop(self) -> None:
        self._stopped = True
        self.busy = False

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
        sizes, levels, seen, spots = [], [], 0.0, []
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
            spots.append(wrong_points(reference, view, center, half, pixels, pixel))
            if wrong <= GOOD_ENOUGH * self.detail / 100.0 * shown:
                break
        if triangles >= FINER_TRIANGLES and seen > 0.0:
            asked = self.detail / 100.0
            uppers = {next((k for k, level in enumerate(levels) if level.loss <= limit * seen), 0) for limit in (asked, PIECE_LIMIT * asked)} - {0}
            for upper in sorted(uppers):
                # A size halfway across the jump, so a piece doesn't get twice the faces to clear a limit.
                yield
                faces = math.sqrt(levels[upper - 1].faces * levels[upper].faces)
                reduced = _decimate(self.context, points, polys, kinds, faces / len(polys))
                view = yield from pictures(reduced[0], reduced[1], center, half, pixels)
                sizes.append(reduced)
                levels.append(Level(len(reduced[1]), picture_counts(reference, view, pixel)[0] * area))
                spots.append(wrong_points(reference, view, center, half, pixels, pixel))
                # Density where the camera saw it go wrong: hold those spots while the rest goes further, then let go.
                for k in (upper - 1, upper, len(levels) - 1):
                    yield
                    keep = hottest(vertex_heat(points, spots[k], 3.0 * pixel), LOCKED_SHARE * levels[k].faces / 2.0)
                    held = _decimate(self.context, points, polys, kinds, HELD_SIZE * levels[k].faces / triangles, keep)
                    reduced = _decimate(self.context, held[0], held[1], held[2], levels[k].faces / max(len(held[1]), 1))
                    view = yield from pictures(reduced[0], reduced[1], center, half, pixels)
                    sizes.append(reduced)
                    levels.append(Level(len(reduced[1]), picture_counts(reference, view, pixel)[0] * area))
                    spots.append(spots[k])
            kept = best_levels(levels)
            sizes, levels = [sizes[k] for k in kept], [levels[k] for k in kept]
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
            self._paired = pair_and_close(MeshData(co, faces, kinds), seed_count=2)[0]
        except Exception as exc:  # a thread has nobody above it to report to
            self.warnings = [f"Auto Segments failed: {exc}"]

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
        reference_shaded = shaded(reference, self.co, self.faces, center, half, MAX_PIXELS)
        total_seen = sum(self.seen)
        asked = self.detail / 100.0
        allowed = asked * total_seen
        best = None
        tried: set[tuple[int, ...]] = set()
        for round_index in range(ROUNDS):
            self.picked = allocate(self.levels, allowed, PIECE_LIMIT * asked, self.seen)
            if tuple(self.picked) in tried:
                break
            tried.add(tuple(self.picked))
            self.stage = f"Pairing into quads, try {round_index + 1}..."
            co, faces, kinds, _parts = self._combined()
            self._paired = None
            self._pair_thread = threading.Thread(target=self._pair, args=(co, faces, kinds), daemon=True, name="CurioMeshPair")
            self._pair_thread.start()
            while self._pair_thread.is_alive():
                if self._stopped:
                    return
                yield
            if self._paired is None:
                return
            result = self._paired
            points = np.asarray(result.vertices, dtype=np.float64)
            out_faces = [tuple(face) for face in result.faces]
            self.stage = f"Checking the result, try {round_index + 1}..."
            view = yield from pictures(points, out_faces, center, half, MAX_PIXELS)
            wrong, shown = picture_counts(reference, view, pixel)
            loss = 100.0 * wrong / max(shown, 1)
            view_shaded = shaded(view, points, out_faces, center, half, MAX_PIXELS)
            bad_shade, both = shading_counts(reference, view, reference_shaded, view_shaded)
            shading = 100.0 * bad_shade / max(both, 1)
            soft = 100.0 * shading_counts(reference, view, reference_shaded, view_shaded, SOFT_DEG)[0] / max(both, 1)
            fits = loss <= 1.1 * self.detail
            # Prefer anything that fits, then the fewest faces. With nothing fitting yet, the least change.
            rank = (not fits, len(out_faces) if fits else loss)
            if best is None or rank < best[0]:
                best = (rank, points, out_faces, list(result.face_materials), loss, shading, soft)
            if abs(loss - self.detail) <= 0.1 * self.detail:
                break
            allowed *= min(max(self.detail / max(loss, 0.01), 0.3), 4.0)
        _rank, points, out_faces, materials, loss, shading, soft = best
        out_fa, out_fb = patch.face_adjacency(out_faces)
        parts = np.unique(patch._labels(len(out_faces), out_fa, out_fb), return_inverse=True)[1]
        quads = sum(1 for face in out_faces if len(face) == 4)
        self.report = {"faces": len(out_faces), "quads": quads, "tris": sum(1 for face in out_faces if len(face) == 3), "pieces": len(self.pieces), "loss": loss, "shading": shading, "shading_soft": soft, "seconds": time.perf_counter() - self.started}
        self._show((points, out_faces, materials, parts))

    def status(self) -> str:
        if self.report is None:
            return self.stage
        r = self.report
        return f"{r['faces']:,} faces, {r['tris']} triangles, {r['pieces']:,} pieces | Camera: {r['loss']:.1f}% of the view differs in shape, {r['shading']:.1f}% in smooth shading. {r['seconds']:.0f} s"
