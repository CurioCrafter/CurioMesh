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
from .triadq.flatquad import _face_frames
from .triadq.flow import snap_new_vertices, snap_rows
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
ROWS_INPUT = 3  # Clean Rows: QuadriFlow's input needs at most this many triangles per quad asked for
ROWS_SMALLEST = 32  # Clean Rows: a piece due fewer quads than this is paired instead
ROWS_START = 0.08  # Clean Rows: first try, as a share of the source's triangles
ROWS_MOST = 0.8  # Clean Rows: most faces, as a share of the source's triangles
ROWS_ROUNDS = 3
NARROW_TRIES = 2  # extra tries with a new size in between, when the sizes on offer land well short of it
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


def _quadriflow(context, points, polys, target: int):
    """(points, quads) from QuadriFlow, or None where it refuses the piece. Runs on a smaller copy when that is plenty."""

    triangles = sum(len(p) - 2 for p in polys)
    if triangles > ROWS_INPUT * target:
        points, polys, _kinds = _decimate(context, points, polys, ["0"] * len(polys), ROWS_INPUT * target / triangles)
    mesh = bpy.data.meshes.new("CurioMesh_Rows")
    mesh.from_pydata(points.tolist(), [], polys)
    mesh.update()
    obj = bpy.data.objects.new("CurioMesh_Rows", mesh)
    context.scene.collection.objects.link(obj)
    layer = context.view_layer
    before = layer.objects.active
    try:
        layer.objects.active = obj
        obj.select_set(True)
        try:
            done = bpy.ops.object.quadriflow_remesh(mode="FACES", target_faces=int(target), use_preserve_sharp=True, use_preserve_boundary=True, seed=0)
        except RuntimeError:
            return None
        if "FINISHED" not in done or not len(obj.data.polygons):
            return None
        co = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", co)
        return co.reshape(-1, 3), [tuple(p.vertices) for p in obj.data.polygons]
    finally:
        layer.objects.active = before
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(data)


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
            result = pair_and_close(coarse, seed_count=2, split_all=self.style == "BALANCED")[0]
            if self._tree is None:
                self._tree = BVHTree.FromPolygons(self.co.tolist(), [list(f) for f in self.faces], all_triangles=False, epsilon=0.0)
            result = snap_new_vertices(result, coarse.vertices, self._tree.find_nearest)
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

    def _rows_once(self, areas, target: int):
        """Generator: every piece as field-aligned rows of one quad size, onto the source. Returns (points, faces, materials)."""

        total = float(areas.sum()) or 1.0
        all_points, all_faces, all_kinds = [], [], []
        offset = 0
        leftovers = []
        for index, ids in enumerate(self.pieces):
            self.stage = f"Laying rows on piece {index + 1} of {len(self.pieces)}"
            self.progress = (index, len(self.pieces))
            yield
            if self._stopped:
                return None
            points, polys, kinds = self._piece_arrays(ids)
            want = int(target * float(areas[index]) / total)
            rows = _quadriflow(self.context, points, polys, want) if want >= ROWS_SMALLEST else None
            if rows is None:  # too small for rows, or QuadriFlow refused it
                triangles = sum(len(p) - 2 for p in polys)
                if triangles > max(KEEP_TRIANGLES, 2 * want):
                    points, polys, kinds = _decimate(self.context, points, polys, kinds, max(KEEP_TRIANGLES, 2 * want) / triangles)
                leftovers.append((points, polys, kinds))
                continue
            yield
            tree = BVHTree.FromPolygons(points.tolist(), [list(p) for p in polys], all_triangles=False, epsilon=0.0)
            moved, quads = snap_rows(rows[0], rows[1], tree.find_nearest)
            all_points.append(moved)
            all_faces += [tuple(v + offset for v in q) for q in quads]
            centers = moved[np.asarray(quads)].mean(axis=1).tolist()
            all_kinds += [kinds[tree.find_nearest(c)[2]] for c in centers]
            offset += len(moved)
        if leftovers:
            pts, fcs, knd, at = [], [], [], 0
            for points, polys, kinds in leftovers:
                pts.append(points)
                fcs += [tuple(v + at for v in p) for p in polys]
                knd += kinds
                at += len(points)
            closed = pair_and_close(MeshData(np.vstack(pts), fcs, knd), seed_count=1)[0]
            all_points.append(np.asarray(closed.vertices))
            all_faces += [tuple(v + offset for v in f) for f in closed.faces]
            all_kinds += list(closed.face_materials)
        return np.vstack(all_points), all_faces, all_kinds

    def _run_rows(self):
        """Clean Rows: order first. Looks at the result at most three times, so it reports the change it got and may miss the amount asked for."""

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
        target = max(int(ROWS_START * source), 256)
        best = None
        for _round in range(ROWS_ROUNDS):
            built = yield from self._rows_once(areas, target)
            if built is None:
                return
            points, out_faces, kinds = built
            self.stage = "Checking the result..."
            view = yield from pictures(points, out_faces, center, half, MAX_PIXELS)
            wrong, shown = picture_counts(reference, view, pixel)
            loss = 100.0 * wrong / max(shown, 1)
            fits = loss <= 1.1 * self.detail
            rank = (not fits, len(out_faces) if fits else loss)  # as in Keep Detail: fits first, then the fewest faces
            if best is None or rank < best[0]:
                best = (rank, points, out_faces, kinds, loss)
            if abs(loss - self.detail) <= 0.1 * self.detail:
                break
            # Change falls about as the square root of the faces.
            wanted = min(max(int(target * (loss / self.detail) ** 2), 64), int(ROWS_MOST * source))
            if wanted == target:
                break
            target = wanted
        _rank, points, out_faces, kinds, loss = best
        out_fa, out_fb = patch.face_adjacency(out_faces)
        parts = np.unique(patch._labels(len(out_faces), out_fa, out_fb), return_inverse=True)[1]
        quads = sum(1 for face in out_faces if len(face) == 4)
        self.report = {"faces": len(out_faces), "quads": quads, "tris": len(out_faces) - quads, "pieces": len(self.pieces), "loss": loss, "seconds": time.perf_counter() - self.started}
        self._show((points, out_faces, kinds, parts))

    def status(self) -> str:
        if self.report is None:
            return self.stage
        r = self.report
        return f"{r['faces']:,} faces, {r['tris']} triangles, {r['pieces']:,} pieces | Camera: {r['loss']:.1f}% of the view differs. {r['seconds']:.0f} s"
