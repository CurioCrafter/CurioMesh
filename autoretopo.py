# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Auto Segments: you say how much of the view may change, and it finds the fewest faces that hold it.

Each loose piece is reduced to several sizes and looked at from six sides, with one fixed pixel size for the whole
object, so a bolt is judged by how big it looks on the object, not blown up to fill the picture. Faces then go where
they remove the most visible change (`budget.allocate`), with a limit no single piece may pass. The pieces are paired
into quads, the whole result is looked at, and the sharing is redone, with extra sizes for the pieces that hold most of
the faces, until it comes near what was asked for. Quads are then taken away (`quadreduce`) until it lands on it.

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

from .triadq import patch, views
from .triadq.budget import Level, allocate
from .triadq.flatquad import pair_and_close
from .triadq.quadreduce import QuadReducer
from .triadq.types import MeshData

VIEW_PIXELS = 2048  # the whole object's size in the camera's pixels: sets the smallest detail that counts
MAX_PIXELS = 192  # a side of one picture, so one huge piece can't take minutes per look
KEEP_TRIANGLES = 24  # pieces this small stay as they are: there's nothing to gain, and thousands of them to look at
LEVELS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # shares of a piece's triangles to try
PIECE_LIMIT = 3.0  # no piece may change by more than this many times the amount asked for
KEEP_SHARE = 0.25  # a piece with over this share of the faces is the body of the model, and always stays
GOOD_ENOUGH = 0.25  # stop trying bigger sizes of a piece once its change is this share of the amount asked for
ROUNDS = 4  # tries at landing on the amount asked for
MOST_RAYS = 20_000_000  # one look at every piece, so a model of thousands of parts still finishes in minutes
RAYS_PER_YIELD = 2000
REDUCE_STEP = 0.16  # share of the quads to try taking away between looks; halved on going too far, down to REDUCE_FINEST
REDUCE_FINEST = 0.015
MOST_LOOKS = 40
AIM = 0.9  # share of the amount asked for that a start should reach
ROOM = 0.5  # a start this close to the amount asked for is good enough: taking quads away uses up the rest
PART_PIXELS = 100  # a part showing fewer pixels than this over the six views can't be judged by itself
REFINE_PIECES = 8
REFINE_COVER = 0.8  # share of the faces the extra sizes need to cover
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
            face = np.zeros((pixels, pixels), dtype=np.int64)
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
                        face[i, j] = found[2]
                since_yield += pixels
                if since_yield >= RAYS_PER_YIELD:
                    since_yield = 0
                    yield
            out.append((hit, normal, depth, face))
    return out


def parts_changed_too_much(reference, view, pixel: float, labels: np.ndarray, limit: float, rough: bool = False) -> np.ndarray:
    """The labels of connected parts of the viewed mesh where over `limit` of what's shown has changed.

    A big flat part can hide a ruined small one in the total. Parts showing too few pixels to judge are left out.
    """

    count = int(labels.max()) + 1
    wrong, shown = np.zeros(count), np.zeros(count)
    for (hit_a, normal_a, depth_a, _face_a), (hit_b, normal_b, depth_b, face_b) in zip(reference, view):
        shown += np.bincount(labels[face_b[hit_b]], minlength=count)
        changed = views.surface_changed(hit_a, normal_a, depth_a, hit_b, normal_b, depth_b, pixel, rough)
        wrong += np.bincount(labels[face_b[changed]], minlength=count)
    return np.flatnonzero((shown >= PART_PIXELS) & (wrong > limit * shown))


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

    def __init__(self, context, co, faces, materials, detail: float, rough: bool = False, drop_hidden: bool = False):
        self.context = context
        self.co = np.asarray(co, dtype=np.float64)
        self.faces = list(faces)
        self.materials = [str(int(m)) for m in materials]
        self.detail = float(detail)
        self.rough = bool(rough)  # judge rough surface as texture (`views`), and let triangles walk away for fewer faces
        self.drop_hidden = bool(drop_hidden)
        self.dropped: list[np.ndarray] = []
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
        self.shares: list[list[float]] = []  # per piece, the share of its triangles at each size
        self.picture_sizes: list[int] = []
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

    def _drop_hidden_pieces(self):
        """Generator: leave out the loose pieces no camera outside can see. A piece with a big share of the faces always stays."""
        tree = BVHTree.FromPolygons(self.co.tolist(), [list(f) for f in self.faces], all_triangles=False, epsilon=0.0)
        labels = np.zeros(len(self.faces), dtype=np.int64)
        for index, ids in enumerate(self.pieces):
            labels[ids] = index
        labels = labels.tolist()
        pixel = float(np.ptp(self.co, axis=0).max()) / VIEW_PIXELS
        reach = 4.0 * float(np.linalg.norm(np.ptp(self.co, axis=0)))

        def cast(origin, direction):
            return tree.ray_cast(Vector(origin), Vector(direction))[2]

        kept = []
        for index, ids in enumerate(self.pieces):
            self.stage = f"Is piece {index + 1} of {len(self.pieces)} in sight?"
            if len(ids) > KEEP_SHARE * len(self.faces):
                kept.append(ids)
                continue
            points = self.co[np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))]
            if views.pixels_seen(cast, labels, points.min(axis=0), points.max(axis=0), pixel, index, reach, enough=1):
                kept.append(ids)
            else:
                self.dropped.append(ids)
            yield
        self.pieces = kept

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
            self.shares.append([1.0])
            self.seen.append(0.0)
            return
        low, high = points.min(axis=0), points.max(axis=0)
        center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
        pixel = 2.0 * half / pixels
        reference = yield from pictures(points, polys, center, half, pixels)
        sizes, levels, shares, seen = [], [], [], 0.0
        for share in LEVELS:
            if share * triangles < 12.0:
                continue
            yield
            reduced = _decimate(self.context, points, polys, kinds, share)
            view = yield from pictures(reduced[0], reduced[1], center, half, pixels)
            wrong, shown = views.picture_counts(reference, view, pixel, self.rough)
            area = pixel * pixel
            seen = max(seen, shown * area)
            sizes.append(reduced)
            levels.append(Level(len(reduced[1]), wrong * area))
            shares.append(share)
            if wrong <= GOOD_ENOUGH * self.detail / 100.0 * shown:
                break
        sizes.append(whole)
        levels.append(Level(len(polys), 0.0))
        shares.append(1.0)
        self.sizes.append(sizes)
        self.levels.append(levels)
        self.shares.append(shares)
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
            self._paired = pair_and_close(MeshData(co, faces, kinds), seed_count=2, join_triangles=True, merge_triangles=self.rough)[0]
        except Exception as exc:  # a thread has nobody above it to report to
            self.warnings = [f"Auto Segments failed: {exc}"]

    def _background(self, work):
        """Generator: run this on a worker thread, so a very large mesh can't hold up Blender. Returns its result."""
        out = []
        thread = threading.Thread(target=lambda: out.append(work()), daemon=True, name="CurioMeshReduce")
        thread.start()
        while thread.is_alive():
            thread.join(0.02)
            yield
        return out[0]

    def _refine(self, finer: bool):
        """Generator: a size halfway between two tried ones, for the pieces holding most of the faces.

        Sizes double from one to the next, which is a huge step on a big piece: it would land well under what was asked
        for, or over it. So only the sizes next to the current pick, and only for the pieces where that matters.
        """
        counts = [self.levels[i][k].faces for i, k in enumerate(self.picked)]
        total = sum(counts)
        spent = 0
        for index in sorted(range(len(counts)), key=lambda i: -counts[i])[:REFINE_PIECES]:
            if spent > REFINE_COVER * total:
                break
            spent += counts[index]
            at = self.picked[index]
            other = at + 1 if finer else at - 1
            shares = self.shares[index]
            if other < 0 or other >= len(shares):
                continue
            share = math.sqrt(shares[at] * shares[other])
            whole = self.sizes[index][-1]
            triangles = sum(len(f) - 2 for f in whole[1])
            if abs(share - shares[at]) * triangles < 8 or abs(share - shares[other]) * triangles < 8:
                continue
            pixels = self.picture_sizes[index]
            low, high = whole[0].min(axis=0), whole[0].max(axis=0)
            center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
            pixel = 2.0 * half / pixels
            self.stage = f"Trying one more size of piece {index + 1}"
            reference = yield from pictures(whole[0], whole[1], center, half, pixels)
            yield
            reduced = _decimate(self.context, whole[0], whole[1], whole[2], share)
            view = yield from pictures(reduced[0], reduced[1], center, half, pixels)
            wrong, _shown = views.picture_counts(reference, view, pixel, self.rough)
            where = max(at, other)
            self.sizes[index].insert(where, reduced)
            self.levels[index].insert(where, Level(len(reduced[1]), wrong * pixel * pixel))
            shares.insert(where, share)

    def _reduce(self, result, loss, reference, center, half, pixel):
        """Generator: take quads away until the view has changed by the amount asked for. Returns (mesh, loss)."""
        if loss > self.detail:
            return result, loss
        reducer = yield from self._background(lambda: QuadReducer(result))
        limit = PIECE_LIMIT * self.detail / 100.0

        def over_limit(mesh, view):
            fa, fb = patch.face_adjacency(mesh.faces)
            labels = np.unique(patch._labels(len(mesh.faces), fa, fb), return_inverse=True)[1]
            parts = parts_changed_too_much(reference, view, pixel, labels, limit, self.rough)
            kept = np.isin(labels, parts)
            return sorted({v for k in np.flatnonzero(kept).tolist() for v in mesh.faces[k]})

        view = yield from pictures(np.asarray(result.vertices), result.faces, center, half, MAX_PIXELS)
        reducer.freeze(over_limit(result, view))  # whatever is already at its limit stays as it is
        saved = yield from self._background(reducer.save)
        step = REDUCE_STEP
        looks = 0
        while step >= REDUCE_FINEST and looks < MOST_LOOKS:
            looks += 1
            before = reducer.alive
            yield from reducer.remove(max(1, int(step * before)))
            if reducer.alive == before:
                if not reducer.refused:
                    break
                reducer.retry_refused()
                yield from reducer.remove(max(1, int(step * before)))
                if reducer.alive == before:
                    break
            attempt = yield from self._background(reducer.result)
            self.stage = f"Taking away quads: {len(attempt.faces):,}"
            view = yield from pictures(np.asarray(attempt.vertices), attempt.faces, center, half, MAX_PIXELS)
            wrong, shown = views.picture_counts(reference, view, pixel, self.rough)
            now = 100.0 * wrong / max(shown, 1)
            ruined = over_limit(attempt, view)
            if now <= self.detail and not ruined:
                result, loss = attempt, now
                saved = yield from self._background(reducer.save)
                continue
            names = reducer.original_ids(ruined)
            yield from self._background(lambda: reducer.restore(saved))
            if ruined:  # those parts stop here, and the rest can go on
                reducer.freeze(names, renamed=True)
            else:
                step *= 0.5
        return result, loss

    def _run(self):
        fa, fb = patch.face_adjacency(self.faces)
        shell = np.unique(patch._labels(len(self.faces), fa, fb), return_inverse=True)[1]
        order = np.argsort(shell, kind="stable")
        starts = np.concatenate(([0], np.flatnonzero(shell[order][1:] != shell[order][:-1]) + 1, [len(order)]))
        self.pieces = sorted((order[a:b] for a, b in zip(starts[:-1], starts[1:])), key=len, reverse=True)
        if self.drop_hidden:
            yield from self._drop_hidden_pieces()
        extent = float(np.ptp(self.co, axis=0).max()) or 1.0
        sizes = np.asarray([2.0 * 0.55 * float(np.ptp(self.co[np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))], axis=0).max()) for ids in self.pieces]) / extent
        picture_sizes = self.picture_sizes = _picture_sizes(sizes).tolist()
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
            pick = tuple(self.levels[i][k].faces for i, k in enumerate(self.picked))
            if pick in tried:
                break
            tried.add(pick)
            self.stage = f"Pairing into quads, try {round_index + 1}..."
            co, faces, kinds, _parts = self._combined()
            self._paired = None
            self._pair_thread = threading.Thread(target=self._pair, args=(co, faces, kinds), daemon=True, name="CurioMeshPair")
            self._pair_thread.start()
            while self._pair_thread.is_alive():
                if self._stopped:
                    return
                self._pair_thread.join(0.02)  # let it have the interpreter: spinning here slows it several times over
                yield
            if self._paired is None:
                return
            result = self._paired
            points = np.asarray(result.vertices, dtype=np.float64)
            out_faces = [tuple(face) for face in result.faces]
            self.stage = f"Checking the result, try {round_index + 1}..."
            view = yield from pictures(points, out_faces, center, half, MAX_PIXELS)
            wrong, shown = views.picture_counts(reference, view, pixel, self.rough)
            unreduced = 100.0 * wrong / max(shown, 1)
            # Prefer anything at or under the amount asked for, then the fewest faces. With nothing fitting, the least change.
            over = unreduced > self.detail
            rank = (unreduced > 1.1 * self.detail, over, len(result.faces) if unreduced <= 1.1 * self.detail else unreduced)
            if best is None or rank < best[0]:
                best = (rank, result, unreduced)
            if ROOM * self.detail <= unreduced <= self.detail:
                break
            if round_index + 1 < ROUNDS:
                yield from self._refine(over)
            # What the sizes say they lose differs from what the whole result loses, so go by the last pick.
            planned = sum(self.levels[i][k].loss for i, k in enumerate(self.picked))
            allowed = max(planned, 0.01 * asked * total_seen) * min(max(AIM * self.detail / max(unreduced, 0.01), 0.3), 4.0)
        result, loss = yield from self._reduce(best[1], best[2], reference, center, half, pixel)
        points = np.asarray(result.vertices, dtype=np.float64)
        out_faces = [tuple(face) for face in result.faces]
        materials = list(result.face_materials)
        out_fa, out_fb = patch.face_adjacency(out_faces)
        parts = np.unique(patch._labels(len(out_faces), out_fa, out_fb), return_inverse=True)[1]
        quads = sum(1 for face in out_faces if len(face) == 4)
        self.report = {"faces": len(out_faces), "quads": quads, "tris": sum(1 for face in out_faces if len(face) == 3), "pieces": len(self.pieces), "loss": loss, "seconds": time.perf_counter() - self.started, "dropped": len(self.dropped), "dropped_faces": sum(len(ids) for ids in self.dropped)}
        self._show((points, out_faces, materials, parts))

    def status(self) -> str:
        if self.report is None:
            return self.stage
        r = self.report
        hidden = f" Left out {r['dropped']:,} hidden pieces ({r['dropped_faces']:,} faces)." if r["dropped"] else ""
        return f"{r['faces']:,} faces, {r['tris']} triangles, {r['pieces']:,} pieces | Camera: {r['loss']:.1f}% of the view differs. {r['seconds']:.0f} s.{hidden}"
