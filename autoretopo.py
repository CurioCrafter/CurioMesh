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
import shutil
import tempfile
import threading
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .triadq import patch
from .triadq.budget import Level, allocate, allocate_to_faces
from .triadq.flatquad import pair_and_close
from . import guides, hardware, symmetry
from .triadq.types import Cancelled, MeshData

VIEW_PIXELS = 2048  # the whole object's size in the camera's pixels: sets the smallest detail that counts
MAX_PIXELS = 192  # a side of one picture, so one huge piece can't take minutes per look
KEEP_TRIANGLES = 24  # pieces this small stay as they are: there's nothing to gain, and thousands of them to look at
MEASURE_SHARE = 0.6  # of the progress bar, spent looking at the pieces
LEVELS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # shares of a piece's triangles to try
PIECE_LIMIT = 3.0  # no piece may change by more than this many times the amount asked for
GOOD_ENOUGH = 0.25  # stop trying bigger sizes of a piece once its change is this share of the amount asked for
ROUNDS = 4  # tries at landing on the amount asked for
CLOSE_ENOUGH = 0.10  # Reduce To: a face count this near the one asked for is a hit
PAIRED_SHARE = 0.6  # first guess of faces after pairing per face before it, corrected after each try
TURNED_DEG = 20.0
MOST_RAYS = 20_000_000  # one look at every piece, so a model of thousands of parts still finishes in minutes
RAYS_PER_YIELD = 2000


def pictures(points: np.ndarray, polys, center: np.ndarray, half: float, pixels: int, tree: BVHTree | None = None):
    """Generator: six exact orthographic pictures, one ray per pixel. Returns [(hit, normal, depth)] when exhausted.

    `tree` is the BVH of this very mesh when the caller already has one (building it is a fifth of a picture set).
    """

    if tree is None:
        tree = BVHTree.FromPolygons(points.tolist(), [list(p) for p in polys], all_triangles=False, epsilon=0.0)
    grid = (((np.arange(pixels) + 0.5) / pixels * 2.0 - 1.0) * half).tolist()
    out = []
    since_yield = 0
    cast = tree.ray_cast
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (1.0, -1.0):
            direction = [0.0, 0.0, 0.0]
            direction[axis] = -sign
            direction = Vector(direction)
            origin = Vector((0.0, 0.0, 0.0))
            origin[axis] = float(center[axis]) + sign * 2.0 * half
            cu, cv = float(center[u]), float(center[v])
            at, normals, depths = [], [], []  # only the hits are kept: most of a picture is often empty
            for i, x in enumerate(grid):
                origin[u] = cu + x
                base = i * pixels
                for j, y in enumerate(grid):
                    origin[v] = cv + y
                    found = cast(origin, direction)
                    if found[0] is not None:
                        at.append(base + j)
                        normals.append(found[1][:])
                        depths.append(found[3])
                since_yield += pixels
                if since_yield >= RAYS_PER_YIELD:
                    since_yield = 0
                    yield
            hit = np.zeros(pixels * pixels, dtype=bool)
            normal = np.zeros((pixels * pixels, 3))
            depth = np.zeros(pixels * pixels)
            if at:
                hit[at] = True
                normal[at] = normals
                depth[at] = depths
            out.append((hit.reshape(pixels, pixels), normal.reshape(pixels, pixels, 3), depth.reshape(pixels, pixels)))
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


DENSITY_STEPS = 24  # painted weights are rounded to this many levels, so a vertex group takes a few calls, not one per vertex


def _decimate(context, points, polys, kinds, ratio, drop=None, lock=None) -> tuple[np.ndarray, list[tuple[int, ...]], list[str]]:
    mesh = bpy.data.meshes.new("CurioMesh_Piece")
    mesh.from_pydata(points.tolist(), [], polys)
    mesh.polygons.foreach_set("material_index", np.asarray([int(k) for k in kinds], dtype=np.int32))
    mesh.update()
    obj = bpy.data.objects.new("CurioMesh_Piece", mesh)
    context.scene.collection.objects.link(obj)
    try:
        modifier = obj.modifiers.new("Reduce", "DECIMATE")
        modifier.ratio = ratio
        painted = drop is not None and float(drop.max(initial=0.0)) > 0.0
        if painted or (lock is not None and lock.any()):
            group = obj.vertex_groups.new(name="CurioMesh_Density")
            group.add(list(range(len(points))), 1.0, "REPLACE")  # a vertex outside the group is never collapsed
            if painted:
                level = np.round(drop / 0.5 * DENSITY_STEPS).astype(np.int64)
                for step in np.unique(level[level > 0]).tolist():
                    group.add(np.flatnonzero(level == step).tolist(), 1.0 - step * 0.5 / DENSITY_STEPS, "REPLACE")
            if lock is not None and lock.any():  # weight 0: Decimate leaves these vertices exactly where they are (chunk borders)
                group.add(np.flatnonzero(lock).tolist(), 0.0, "REPLACE")
            modifier.vertex_group = group.name
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


def _native_tree(context, co: np.ndarray, flat: np.ndarray, sizes: np.ndarray) -> BVHTree | None:
    """The BVH of a big mesh built inside Blender from arrays, with no Python lists of faces. None if this Blender won't take it."""
    mesh = obj = None
    try:
        starts = np.cumsum(sizes) - sizes
        counts = sizes - 2  # fan-triangulate like BVHTree.FromPolygons does, so a quad is cut the same way and the pictures match
        face = np.repeat(np.arange(len(sizes)), counts)
        local = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        base = starts[face]
        flat = np.stack((flat[base], flat[base + local + 1], flat[base + local + 2]), axis=1).ravel()
        sizes = np.full(len(face), 3, dtype=np.int64)
        mesh = bpy.data.meshes.new("CurioMesh_Tree")
        mesh.vertices.add(len(co))
        mesh.vertices.foreach_set("co", np.ascontiguousarray(co, dtype=np.float32).ravel())
        mesh.loops.add(len(flat))
        mesh.loops.foreach_set("vertex_index", flat.astype(np.int32))
        mesh.polygons.add(len(sizes))
        mesh.polygons.foreach_set("loop_start", (np.cumsum(sizes) - sizes).astype(np.int32))
        try:
            mesh.polygons.foreach_set("loop_total", sizes.astype(np.int32))  # read-only from Blender 4.0, where the starts are enough
        except Exception:
            pass
        mesh.update()
        if len(mesh.polygons) != len(sizes) or mesh.polygons[len(sizes) - 1].loop_total != int(sizes[-1]) or mesh.polygons[0].loop_total != int(sizes[0]):
            return None
        obj = bpy.data.objects.new("CurioMesh_Tree", mesh)
        context.scene.collection.objects.link(obj)
        return BVHTree.FromObject(obj, context.evaluated_depsgraph_get(), epsilon=0.0)
    except Exception:
        return None
    finally:
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None:
            bpy.data.meshes.remove(mesh)


def _picture_sizes(sizes: np.ndarray, most_rays: int = MOST_RAYS, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    """Pixels per side for pieces of these sizes (the whole object is 1.0), coarser for a model with very many parts."""

    view, most = float(VIEW_PIXELS), float(max_pixels)
    while True:
        pixels = np.clip(np.ceil(sizes * view), 8, most).astype(np.int64)
        if 6 * int((pixels * pixels).sum()) <= most_rays or most <= 16:
            return pixels
        view *= 0.85
        most *= 0.85


class AutoJob:
    """One Auto Segments run. Call `step` from a timer until `busy` is False; `output` holds the latest result to show."""

    def __init__(self, context, co, faces, materials, detail: float, target_faces: int = 0, cache: dict | None = None, mirror: tuple | None = None, note: str = "", guide_lines: list | None = None, project=None, drop: np.ndarray | None = None, settings: hardware.Settings | None = None, checkpoint=None, keep_progress: bool = False):
        self.context = context
        self.co = np.asarray(co, dtype=np.float64)
        self.faces = list(faces)
        self.materials = [str(int(m)) for m in materials]
        self.detail = float(detail)
        self.cache = {} if cache is None else cache  # what was measured on this very mesh before: starting again with another amount skips it
        self.mirror = mirror  # (axis, plane offset, size, faces of the whole mesh, side solved): co/faces are one half, the result is mirrored
        self.note = note
        self.guide_lines = guide_lines or []  # strokes laid on the surface; the quads are made to follow them
        self.project = project  # the nearest point on the original surface, for the points a stroke adds to the mesh
        self.guide_edges: tuple = ()
        self.drop = None if drop is None or not float(np.max(drop, initial=0.0)) > 0.0 else np.asarray(drop, dtype=np.float64)  # per vertex, from the painted density
        self.cfg = settings or hardware.EFFORTS["NORMAL"]  # how hard to work: slice length, rays, chunk size
        self.max_pixels = self.cfg.max_pixels
        self.checkpoint = checkpoint  # folder where finished parts are saved so a stopped job can resume (None = don't save)
        self.keep_progress = keep_progress
        self.resumed = 0  # parts loaded from the saved progress instead of measured
        self.names: list[str] = []  # where each part is, in words
        self.locked: np.ndarray | None = None  # vertices on a chunk border: never moved, so neighbours join without a gap
        self.chunks = 0  # how many regions a huge piece was cut into
        self.losses: list[float] = []  # the view change measured after each try
        self.first: list = []  # per piece, its smallest size (what the preview shows before sizes are picked)
        self.files: list = []  # per piece, the file that holds its sizes when they are kept on disk, not in memory
        self._tmp: Path | None = None  # where sizes go when there is no checkpoint folder
        self.mem: dict[str, int] = {"start": round(hardware.peak_memory_mb())}  # the most memory used by the end of each stage, MB
        self.target_faces = int(target_faces)  # 0 = keep the detail asked for; above 0 = land on this many faces
        self.fraction = 0.0  # 0 to 1 of the whole job, for the progress bar
        self.tuned: dict[int, tuple] = {}  # piece -> a size made to measure, to land on a face count
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
        self.timings: dict[str, float] = {}  # seconds spent per kind of work, for the report
        self._search = self._run()

    def seconds_left(self) -> float | None:
        """A rough guess from the pace so far; None until there is enough to go on."""
        if not self.busy or self.fraction < 0.05:
            return None
        spent = time.perf_counter() - self.started
        return spent * (1.0 - self.fraction) / self.fraction

    def stop(self) -> None:
        self._stopped = True
        self.busy = False
        self._drop_tmp()

    def _drop_tmp(self) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def _mark(self, stage: str) -> None:
        self.mem[stage] = round(hardware.peak_memory_mb())

    def _size(self, index: int, level: int):
        """(points, polys, kinds) of a piece at a picked size: from memory, from its file, or rebuilt from the mesh for the whole piece."""
        sizes = self.sizes[index]
        if sizes is None:
            loaded = hardware.load_piece(self.files[index])
            if loaded is None:
                raise RuntimeError("a saved part went missing")
            sizes = loaded[1] + [None]
        return sizes[level] if sizes[level] is not None else self._piece_arrays(self.pieces[index])

    def _finish_piece(self, key, sizes: list, levels: list, seen: float, file, fresh: bool) -> None:
        """Keep a measured piece. `sizes` ends with None, the whole piece, which is rebuilt from the mesh when needed.

        A piece of a cut mesh goes to disk and only its numbers stay in memory.
        """
        self.levels.append(levels)
        self.seen.append(seen)
        self.first.append(sizes[0])
        pairs = [(lv.faces, lv.loss) for lv in levels]
        if self.chunks:
            if file is None:
                if self._tmp is None:
                    self._tmp = Path(tempfile.mkdtemp(prefix="curiomesh_"))
                file = hardware.piece_file(self._tmp, repr(key[2]).encode() + key[1])
            if fresh:
                hardware.save_piece(file, self.detail, sizes[:-1], pairs, seen)
            self.sizes.append(None)
            self.files.append(file)
            return
        self.sizes.append(sizes)
        self.files.append(None)
        self.cache[key] = (self.detail, sizes, levels, seen)
        if file is not None and fresh:
            hardware.save_piece(file, self.detail, sizes[:-1], pairs, seen)

    def step(self, seconds: float | None = None) -> None:
        """Do the next slice of work on this (the main) thread."""
        if not self.busy:
            return
        seconds = self.cfg.slice_s if seconds is None else seconds
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

    def _timed(self, name: str, work):
        """Generator: run `work` (a generator) and add the time it really runs to `self.timings[name]`."""
        while True:
            started = time.perf_counter()
            try:
                step = next(work)
            except StopIteration as done:
                self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - started
                return done.value
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - started
            yield step

    def _piece_drop(self, ids: np.ndarray):
        """The painted density of the vertices of a piece, in the order `_piece_arrays` lists them (or None)."""
        return self._piece_vertex(ids, self.drop)

    def _piece_lock(self, ids: np.ndarray):
        """Which vertices of a piece sit on a chunk border (or None when nothing was cut)."""
        return self._piece_vertex(ids, self.locked)

    def _piece_vertex(self, ids: np.ndarray, per_vertex):
        if per_vertex is None:
            return None
        return per_vertex[np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))]

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
            self.files.append(None)
            self.first.append(whole)
            self.levels.append([Level(len(polys), 0.0)])
            self.seen.append(0.0)
            return
        key = ("piece", ids.tobytes(), None if self.drop is None else hash(self.drop.tobytes()))
        file = None
        if self.checkpoint is not None:
            file = hardware.piece_file(self.checkpoint, repr(key[2]).encode() + key[1])
            saved = hardware.load_piece(file) if file.exists() else None
            if saved is not None and saved[0] <= self.detail:  # a stopped run measured this part already
                self.resumed += 1
                self._finish_piece(key, saved[1] + [None], [Level(f, w) for f, w in saved[2]], saved[3], file, False)
                return
        known = self.cache.get(key)
        if known is not None and known[0] <= self.detail:  # measured with the same or a stricter amount (more sizes tried): nothing to redo
            self.sizes.append(known[1])
            self.files.append(None)
            self.first.append(known[1][0])
            self.levels.append(known[2])
            self.seen.append(known[3])
            return
        low, high = points.min(axis=0), points.max(axis=0)
        center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
        pixel = 2.0 * half / pixels
        reference = yield from self._reference(center, half, pixels, points, polys)
        sizes, levels, seen = [], [], 0.0
        for share in LEVELS:
            if share * triangles < 12.0:
                continue
            yield
            started = time.perf_counter()
            reduced = _decimate(self.context, points, polys, kinds, share, self._piece_drop(ids), self._piece_lock(ids))
            self.timings["decimate"] = self.timings.get("decimate", 0.0) + time.perf_counter() - started
            view = yield from self._timed("pictures", pictures(reduced[0], reduced[1], center, half, pixels))
            wrong, shown = picture_counts(reference, view, pixel)
            area = pixel * pixel
            seen = max(seen, shown * area)
            sizes.append(reduced)
            levels.append(Level(len(reduced[1]), wrong * area))
            if wrong <= GOOD_ENOUGH * self.detail / 100.0 * shown:
                break
        sizes.append(None)  # the whole piece is not kept: it is rebuilt from the mesh
        levels.append(Level(len(polys), 0.0))
        self._finish_piece(key, sizes, levels, seen, file, True)

    def _reference(self, center, half, pixels, points, polys, tree=None):
        """Generator: the six pictures of the original at this view. A lone piece is the whole object, so they are made once."""
        key = ("view", center.tobytes(), half, pixels, len(polys), hash(points.tobytes()))
        if key in self.cache:
            return self.cache[key]
        views = yield from self._timed("pictures", pictures(points, polys, center, half, pixels, tree))
        if not self.chunks:  # a cut mesh has a set per region: they are used once, so they are not kept
            self.cache[key] = views
        return views

    def _combined(self, join: bool = False, only_done: bool = False) -> tuple[np.ndarray, list[tuple[int, ...]], list[str], np.ndarray]:
        """Every piece at its picked size, and any not measured yet as it is (or left out, with `only_done`).

        `join` welds the chunk borders, which were never moved, so the regions of a cut piece are one surface again.
        """
        all_points, all_faces, all_kinds, parts = [], [], [], []
        offset = 0
        for index, ids in enumerate(self.pieces):
            if only_done and index >= len(self.sizes):
                continue
            if index in self.tuned:
                points, polys, kinds = self.tuned[index]
            elif index < len(self.sizes):
                if index < len(self.picked):
                    points, polys, kinds = self._size(index, self.picked[index])
                else:
                    points, polys, kinds = self.first[index] if self.first[index] is not None else self._piece_arrays(ids)
            else:
                points, polys, kinds = self._piece_arrays(ids)
            all_points.append(points)
            all_faces += [tuple(v + offset for v in p) for p in polys]
            all_kinds += kinds
            parts += [index] * len(polys)
            offset += len(points)
        if not all_points:
            return np.zeros((0, 3)), [], [], np.zeros(0, dtype=np.int64)
        points = np.vstack(all_points)
        if join and self.locked is not None:
            points, all_faces = hardware.weld(points, all_faces)
        return points, all_faces, all_kinds, np.asarray(parts, dtype=np.int64)

    def _show(self, final=None) -> None:
        if final is None:
            co, faces, kinds, parts = self._combined(only_done=self.cfg.only_done or bool(self.chunks))
            if not faces:
                return
        else:
            co, faces, kinds, parts = final
        if self.mirror is not None:
            axis, offset, size = self.mirror[:3]
            co, faces, kinds, _snapped = symmetry.mirror_result(co, faces, kinds, axis, offset, size)
            if final is None:
                parts = np.concatenate((parts, parts + int(parts.max(initial=-1)) + 1))
            else:
                fa, fb = patch.face_adjacency(faces)
                parts = np.unique(patch._labels(len(faces), fa, fb), return_inverse=True)[1]
        count = len(faces)
        new = np.ones(count, bool)
        self.output = {"co": co, "faces": faces, "new": new, "locked": np.zeros(count, bool), "mat": np.asarray([int(k) if str(k).isdigit() else 0 for k in kinds], dtype=np.int32), "parts": parts}
        self.output_id += 1
        self.shown_at = time.perf_counter()

    def _guided(self, co, faces, kinds):
        """The mesh about to be paired, cut along the strokes: (co, faces, kinds, guide edges)."""
        started = time.perf_counter()
        tris, owner = [], []
        for i, face in enumerate(faces):
            for k in range(1, len(face) - 1):
                tris.append((face[0], face[k], face[k + 1]))
                owner.append(i)
        tris = np.asarray(tris, dtype=np.int64)
        edge = np.linalg.norm(co[tris[:, 0]] - co[tris[:, 1]], axis=1)
        spacing = float(np.median(edge))
        samples = np.vstack([guides.resample(line, spacing / 2.0) for line in self.guide_lines])
        kind = np.asarray([int(k) if str(k).isdigit() else 0 for k in kinds], dtype=np.int64)[owner]
        co, tris, kind = guides.refine(co, tris, kind, samples, spacing, self.project)
        chains, keys = guides.chain(co, tris, self.guide_lines, spacing)
        if chains:
            co = guides.pull(co, tris, chains, self.guide_lines, spacing)
        self.timings["guides"] = self.timings.get("guides", 0.0) + time.perf_counter() - started
        return co, [tuple(t) for t in tris.tolist()], [str(k) for k in kind.tolist()], tuple(sorted(keys))

    def _pair(self, co, faces, kinds) -> None:
        started = time.perf_counter()
        try:
            self._paired = pair_and_close(MeshData(co, faces, kinds), seed_count=2, cancelled=lambda: self._stopped, guide_edges=self.guide_edges)[0]
            self.timings["pairing"] = self.timings.get("pairing", 0.0) + time.perf_counter() - started
        except Cancelled:  # Stop was pressed: the thread ends at its next stopping point
            pass
        except Exception as exc:  # a thread has nobody above it to report to
            self.warnings = [f"Auto Segments failed: {exc}"]

    def _run(self):
        fa, fb = patch.face_adjacency(self.faces)
        shell = np.unique(patch._labels(len(self.faces), fa, fb), return_inverse=True)[1]
        order = np.argsort(shell, kind="stable")
        starts = np.concatenate(([0], np.flatnonzero(shell[order][1:] != shell[order][:-1]) + 1, [len(order)]))
        shells = [order[a:b] for a, b in zip(starts[:-1], starts[1:])]
        low, high = self.co.min(axis=0), self.co.max(axis=0)
        sizes_f = np.fromiter((len(f) for f in self.faces), dtype=np.int64, count=len(self.faces))
        flat = np.fromiter((v for f in self.faces for v in f), dtype=np.int64, count=int(sizes_f.sum()))
        centers = np.add.reduceat(self.co[flat], np.concatenate(([0], np.cumsum(sizes_f)[:-1])), axis=0) / sizes_f[:, None]
        cut = []  # a piece with more faces than the machine should hold at once is cut into regions
        for ids in shells:
            if len(ids) > self.cfg.chunk_faces:
                regions = hardware.split_chunks(centers[ids], self.cfg.chunk_faces)
                cut += [ids[r] for r in regions]
                self.chunks += len(regions)
            else:
                cut.append(ids)
        cut.sort(key=len, reverse=True)
        self.pieces = cut
        self.names = [hardware.region_name(centers[ids].mean(axis=0), low, high) for ids in cut]
        if self.chunks:
            label = np.zeros(len(self.faces), dtype=np.int64)
            for number, ids in enumerate(cut):
                label[ids] = number
            self.locked = hardware.border_vertices(self.faces, label, len(self.co))
        extent = float(np.ptp(self.co, axis=0).max()) or 1.0
        sizes = np.asarray([2.0 * 0.55 * float(np.ptp(self.co[np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))], axis=0).max()) for ids in self.pieces]) / extent
        picture_sizes = _picture_sizes(sizes, self.cfg.max_rays, self.max_pixels).tolist()
        weights = np.asarray([sum(len(self.faces[f]) - 2 for f in ids.tolist()) for ids in self.pieces], dtype=np.float64)
        done_share = np.concatenate(([0.0], np.cumsum(weights))) / max(float(weights.sum()), 1.0)
        self._show()
        for index, ids in enumerate(self.pieces):
            self.progress = (index, len(self.pieces))
            self.fraction = MEASURE_SHARE * float(done_share[index])
            self.stage = f"Part {index + 1} of {len(self.pieces)}: {self.names[index]}" if self.chunks or len(self.pieces) < 40 else f"Looking at piece {index + 1} of {len(self.pieces)}"
            yield from self._measure_piece(ids, picture_sizes[index])
            yield
            if self._stopped:
                return
            if time.perf_counter() - self.shown_at > self.cfg.show_every:
                self._show()
        self.progress = (len(self.pieces), len(self.pieces))
        self.fraction = MEASURE_SHARE
        self.stage = "Looking at the whole object"
        center, half = (low + high) / 2.0, extent * 0.55
        pixel = 2.0 * half / self.max_pixels
        self._mark("measured")
        tree = _native_tree(self.context, self.co, flat, sizes_f) if self.chunks else None
        reference = yield from self._reference(center, half, self.max_pixels, self.co, self.faces, tree)
        del tree
        total_seen = sum(self.seen)
        asked = self.detail / 100.0
        allowed = asked * total_seen
        reducing = self.target_faces > 0
        paired_share = PAIRED_SHARE
        best = None
        tried: set[tuple[int, ...]] = set()
        for round_index in range(ROUNDS):
            self.fraction = MEASURE_SHARE + (1.0 - MEASURE_SHARE) * round_index / ROUNDS
            self.tuned = {}
            if reducing:
                want = self.target_faces / paired_share  # faces before pairing that should give the faces asked for
                self.picked = allocate_to_faces(self.levels, want, 1.0, self.seen)
                self._tune(want)
            else:
                self.picked = allocate(self.levels, allowed, PIECE_LIMIT * asked, self.seen)
            if tuple(self.picked) in tried and not self.tuned:
                break
            tried.add(tuple(self.picked))
            self.stage = f"Pairing into quads, try {round_index + 1}"
            co, faces, kinds, _parts = self._combined(join=True)
            if self.guide_lines:
                self.stage = f"Cutting the guide curves in, try {round_index + 1}"
                yield
                co, faces, kinds, self.guide_edges = self._guided(co, faces, kinds)
            before_pairing = len(faces)
            self._paired = None
            self._pair_thread = threading.Thread(target=self._pair, args=(co, faces, kinds), daemon=True, name="CurioMeshPair")
            self._pair_thread.start()
            while self._pair_thread.is_alive():
                if self._stopped:
                    return
                self._pair_thread.join(0.02)  # sleep instead of spinning: a busy main thread slows the pairing thread (shared GIL)
                yield
            if self._paired is None:
                return
            result = self._paired
            self._mark("paired")
            points = np.asarray(result.vertices, dtype=np.float64)
            out_faces = [tuple(face) for face in result.faces]
            self.fraction += 0.5 * (1.0 - MEASURE_SHARE) / ROUNDS
            if reducing:
                miss = abs(len(out_faces) - self.target_faces) / self.target_faces
                if best is None or miss < best[0]:
                    best = (miss, points, out_faces, list(result.face_materials), None)
                if miss <= CLOSE_ENOUGH:
                    break
                paired_share = len(out_faces) / max(before_pairing, 1)
                continue
            self.stage = f"Checking the result, try {round_index + 1}"
            loss = yield from self._look(points, out_faces, center, half, reference, pixel)
            self.losses.append(round(loss, 2))
            fits = loss <= 1.1 * self.detail
            # Prefer anything that fits, then the fewest faces. With nothing fitting yet, the least change.
            rank = (not fits, len(out_faces) if fits else loss)
            if best is None or rank < best[0]:
                best = (rank, points, out_faces, list(result.face_materials), loss)
            if abs(loss - self.detail) <= 0.1 * self.detail:
                break
            allowed *= min(max(self.detail / max(loss, 0.01), 0.3), 4.0)
        _rank, points, out_faces, materials, loss = best
        if loss is None:  # Reduce To looks at the one it keeps, so the view change can still be judged
            self.stage = "Checking the result"
            self.fraction = 0.95
            loss = yield from self._look(points, out_faces, center, half, reference, pixel)
        out_fa, out_fb = patch.face_adjacency(out_faces)
        parts = np.unique(patch._labels(len(out_faces), out_fa, out_fb), return_inverse=True)[1]
        self.fraction = 1.0
        self._show((points, out_faces, materials, parts))
        shown = self.output["faces"]
        follows = None
        if self.guide_lines:  # how far, along each stroke, the nearest edge of the result is: mean and worst, as a share of the size
            size = float(np.linalg.norm(np.ptp(self.co, axis=0))) or 1.0
            mean, worst = guides.follow(self.guide_lines, self.output["co"], shown, size / 400.0)
            follows = {"strokes": len(self.guide_lines), "mean": mean / size, "worst": worst / size}
        whole = self.mirror[3] if self.mirror is not None else len(self.faces)
        quads = sum(1 for face in shown if len(face) == 4)
        self.report = {"before": whole, "faces": len(shown), "quads": quads, "tris": sum(1 for face in shown if len(face) == 3), "pieces": len(self.pieces), "loss": loss, "asked": self.target_faces * (2 if self.mirror is not None else 1), "seconds": time.perf_counter() - self.started, "timings": {k: round(v, 1) for k, v in self.timings.items()}, "guides": follows, "effort": self.cfg.name, "chunks": self.chunks, "resumed": self.resumed, "peak_mb": hardware.peak_memory_mb(), "mem": self.mem, "losses": self.losses}
        if self.checkpoint is not None and not self.keep_progress:
            shutil.rmtree(self.checkpoint, ignore_errors=True)  # finished: nothing left to resume
        self._drop_tmp()

    def _look(self, points, out_faces, center, half, reference, pixel):
        """Generator: the percent of the view that differs from the original. Returns it when exhausted."""
        view = yield from self._timed("pictures", pictures(points, out_faces, center, half, self.max_pixels))
        wrong, shown = picture_counts(reference, view, pixel)
        return 100.0 * wrong / max(shown, 1)

    def _tune(self, want: float) -> None:
        """Reduce To: the sizes come in steps of two, so remake the biggest picked piece to land closer on `want`."""
        total = sum(levels[k].faces for levels, k in zip(self.levels, self.picked))
        miss = want - total
        if abs(miss) <= 0.03 * want:
            return
        index = max(range(len(self.picked)), key=lambda i: self.levels[i][self.picked[i]].faces)
        points, polys, kinds = self._piece_arrays(self.pieces[index])
        triangles = sum(len(p) - 2 for p in polys)
        goal = self.levels[index][self.picked[index]].faces + miss
        if triangles <= KEEP_TRIANGLES or goal < 12 or goal >= triangles:
            return
        self.tuned[index] = _decimate(self.context, points, polys, kinds, goal / triangles, self._piece_drop(self.pieces[index]))

    def status(self) -> str:
        extra = f" | {self.note}" if self.note else ""
        if self.report is None:
            return self.stage + extra
        r = self.report
        return f"{r['faces']:,} faces, {r['tris']} triangles, {r['pieces']:,} pieces | Camera: {r['loss']:.1f}% of the view differs. {r['seconds']:.0f} s{extra}"
