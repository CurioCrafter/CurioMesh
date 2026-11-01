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
from .triadq.budget import Level, allocate
from .triadq.flatquad import _face_frames, pair_and_close
from .triadq.smoothgrid import GridReport, crease_areas, grid_areas, smooth_areas
from .triadq.types import MeshData

VIEW_PIXELS = 2048  # the whole object's size in the camera's pixels: sets the smallest detail that counts
MAX_PIXELS = 192  # a side of one picture, so one huge piece can't take minutes per look
KEEP_TRIANGLES = 24  # pieces this small stay as they are: there's nothing to gain, and thousands of them to look at
LEVELS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # shares of a piece's triangles to try
PIECE_LIMIT = 3.0  # no piece may change by more than this many times the amount asked for
GOOD_ENOUGH = 0.25  # stop trying bigger sizes of a piece once its change is this share of the amount asked for
ROUNDS = 4  # tries at landing on the amount asked for
TURNED_DEG = 20.0
MOST_RAYS = 20_000_000  # one look at every piece, so a model of thousands of parts still finishes in minutes
RAYS_PER_YIELD = 2000
SHOW_EVERY = 1.0  # seconds between preview updates


def pictures(points: np.ndarray, polys, center: np.ndarray, half: float, pixels: int):
    """Generator: six exact orthographic pictures, one ray per pixel. Returns [(hit, normal, depth)] when exhausted."""

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
                since_yield += pixels
                if since_yield >= RAYS_PER_YIELD:
                    since_yield = 0
                    yield
            out.append((hit, normal, depth))
    return out


def picture_counts(reference, other, pixel: float) -> tuple[int, int]:
    """(pixels a viewer would see as different in outline, direction or depth, pixels showing the object)."""

    wrong, total = 0, 0
    limit = math.cos(math.radians(TURNED_DEG))
    for (hit_a, normal_a, depth_a), (hit_b, normal_b, depth_b) in zip(reference, other):
        both = hit_a & hit_b
        turned = np.abs((normal_a[both] * normal_b[both]).sum(axis=1)) < limit  # winding isn't shape
        moved = np.abs(depth_a[both] - depth_b[both]) > 2.0 * pixel
        wrong += int((turned | moved).sum()) + int((hit_a ^ hit_b).sum())
        total += int((hit_a | hit_b).sum())
    return wrong, total


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


def smooth_grids(co, faces, coarse_points, coarse_faces, coarse_kinds, labels=None):
    """The reduced mesh with its smooth round areas rebuilt as rows of quads, using BVH lookups on the source."""

    if labels is None:
        domes = smooth_areas(co, faces)
        labels = (domes, crease_areas(co, faces, domes >= 0))
    coarse = MeshData(coarse_points, coarse_faces, coarse_kinds)
    report = GridReport()
    if int(max(area.max(initial=-1) for area in labels)) < 0:
        return coarse, report
    points = co.tolist()
    whole = BVHTree.FromPolygons(points, [list(f) for f in faces], all_triangles=False, epsilon=0.0)

    def nearest(query):
        found = [whole.find_nearest(Vector(p)) for p in query.tolist()]
        return np.asarray([row[2] for row in found], dtype=np.int64), np.asarray([row[0] for row in found], dtype=np.float64).reshape(-1, 3)

    def project_for(area):
        ids = np.flatnonzero(labels[1] == area).tolist()
        tree = BVHTree.FromPolygons(points, [list(faces[f]) for f in ids], all_triangles=False, epsilon=0.0)
        return lambda query: np.asarray([tree.find_nearest(Vector(p))[0] for p in query.tolist()], dtype=np.float64).reshape(-1, 3)

    extent = float(np.ptp(co, axis=0).max())
    normals = _face_frames(co, faces)[0]
    coarse, report = grid_areas(coarse, labels[0], nearest, extent=extent, normals=normals)
    coarse, later = grid_areas(coarse, labels[1], nearest, project_for=project_for, extent=extent, normals=normals)
    report.built += later.built
    report.faces_in += later.faces_in
    report.quads_out += later.quads_out
    report.loops += later.loops
    for reason, count in later.rejected.items():
        report.rejected[reason] = report.rejected.get(reason, 0) + count
    return coarse, report


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
        self._labels = None
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
            if self._labels is None:
                domes = smooth_areas(self.co, self.faces)
                self._labels = (domes, crease_areas(self.co, self.faces, domes >= 0))
            coarse = smooth_grids(self.co, self.faces, co, faces, kinds, self._labels)[0]
            self._paired = pair_and_close(coarse, seed_count=2, keep_quads=True)[0]
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
            fits = loss <= 1.1 * self.detail
            # Prefer anything that fits, then the fewest faces. With nothing fitting yet, the least change.
            rank = (not fits, len(out_faces) if fits else loss)
            if best is None or rank < best[0]:
                best = (rank, points, out_faces, list(result.face_materials), loss)
            if abs(loss - self.detail) <= 0.1 * self.detail:
                break
            allowed *= min(max(self.detail / max(loss, 0.01), 0.3), 4.0)
        _rank, points, out_faces, materials, loss = best
        out_fa, out_fb = patch.face_adjacency(out_faces)
        parts = np.unique(patch._labels(len(out_faces), out_fa, out_fb), return_inverse=True)[1]
        quads = sum(1 for face in out_faces if len(face) == 4)
        self.report = {"faces": len(out_faces), "quads": quads, "tris": sum(1 for face in out_faces if len(face) == 3), "pieces": len(self.pieces), "loss": loss, "seconds": time.perf_counter() - self.started}
        self._show((points, out_faces, materials, parts))

    def status(self) -> str:
        if self.report is None:
            return self.stage
        r = self.report
        return f"{r['faces']:,} faces, {r['tris']} triangles, {r['pieces']:,} pieces | Camera: {r['loss']:.1f}% of the view differs. {r['seconds']:.0f} s"
