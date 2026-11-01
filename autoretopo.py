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
from mathutils import Vector, geometry
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .triadq import checks, patch
from .triadq.budget import Level, allocate, allocate_to_faces
from .triadq.flatquad import flat_quad_remesh, flatten_regions, pair_and_close
from .triadq.types import MeshData

VIEW_PIXELS = 2048  # the whole object's size in the camera's pixels: sets the smallest detail that counts
MAX_PIXELS = 192  # a side of one picture, so one huge piece can't take minutes per look
KEEP_TRIANGLES = 24  # pieces this small stay as they are: there's nothing to gain, and thousands of them to look at
MEASURE_SHARE = 0.6  # of the progress bar, spent looking at the pieces
LEVELS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # shares of a piece's triangles to try
PIECE_LIMIT = 3.0  # no piece may change by more than this many times the amount asked for
GOOD_ENOUGH = 0.25  # stop trying bigger sizes of a piece once its change is this share of the amount asked for
ROUNDS = 4  # tries at landing on the amount asked for
CLOSE_ENOUGH = 0.10  # Reduce To: a face count this near the one asked for is a hit
PAIRED_SHARE = 0.6  # first guess of faces after pairing per face before it, corrected after each try
TURNED_DEG = 20.0
MOST_RAYS = 20_000_000  # one look at every piece, so a model of thousands of parts still finishes in minutes
RAYS_PER_YIELD = 2000
SHOW_EVERY = 1.0  # seconds between preview updates
FIX_TRIES = 3  # repair tries on a part that got new problems: one size up each time, the last try the source's own faces
MOST_REPAIRED = 30  # parts remade per try, worst first, so a model of thousands of parts still finishes
SOURCE_BACK = 3000  # faces: only a part this small is put back as the source's own faces; a bigger one is left and selected
INLINE = ("QUADRIFLOW", "GRID", "FLOW")  # methods that must run on the main thread


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


def triangulate(points, faces, kinds):
    """Every face as triangles. A quad splits on its shorter diagonal, a larger face is tessellated."""
    out, mats = [], []
    for face, kind in zip(faces, kinds):
        n = len(face)
        if n == 3:
            tris = [face]
        elif n == 4:
            a, b, c, d = (points[v] for v in face)
            tris = [(face[0], face[1], face[2]), (face[0], face[2], face[3])] if np.linalg.norm(a - c) <= np.linalg.norm(b - d) else [(face[0], face[1], face[3]), (face[1], face[2], face[3])]
        else:
            try:
                tris = [tuple(face[i] for i in t) for t in geometry.tessellate_polygon([[Vector(points[v]) for v in face]])]
            except Exception:
                tris = [(face[0], face[i], face[i + 1]) for i in range(1, n - 1)]
        out += tris
        mats += [kind] * len(tris)
    return points, out, mats


def make_pure(co, faces, kinds, method, output):
    """The faces for one group of pieces by NumPy alone (runs on a worker thread): (points, faces, materials)."""
    mesh = MeshData(co, faces, kinds)
    if method == "FLAT":
        if output == "TRIS":
            out = flatten_regions(mesh)[0]
        else:
            out = flat_quad_remesh(mesh, force_quads=output == "QUADS")[0]
    elif output == "TRIS" or method == "TRIS":
        out = mesh
    else:
        out = pair_and_close(mesh, seed_count=4 if method == "PAIRING" else 2, force_quads=output == "QUADS")[0]
    points, out_faces, mats = np.asarray(out.vertices, dtype=np.float64), [tuple(f) for f in out.faces], list(out.face_materials)
    if output == "TRIS" or method == "TRIS":
        points, out_faces, mats = triangulate(points, out_faces, mats)
    return points, out_faces, mats


def _quadriflow(context, points, polys, kinds, goal):
    mesh = bpy.data.meshes.new("CurioMesh_Piece")
    mesh.from_pydata(points.tolist(), [], polys)
    mesh.update()
    obj = bpy.data.objects.new("CurioMesh_Piece", mesh)
    context.scene.collection.objects.link(obj)
    try:
        for other in context.view_layer.objects:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if "FINISHED" not in bpy.ops.object.quadriflow_remesh(mode="FACES", target_faces=max(4, int(goal)), seed=0) or not len(obj.data.polygons):
            raise ValueError("QuadriFlow made nothing")
        done = obj.data
        co = np.empty(len(done.vertices) * 3, dtype=np.float64)
        done.vertices.foreach_get("co", co)
        return co.reshape(-1, 3), [tuple(p.vertices) for p in done.polygons], [kinds[0]] * len(done.polygons)
    finally:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(data)


def _fill(points, polys, kinds, strategy, goal):
    """Grid or Flow fill over a whole part: only works when the part is one open surface with a single edge loop."""
    tris = [t for p in polys for t in ((p[0], p[i], p[i + 1]) for i in range(1, len(p) - 1))]
    t = np.asarray(tris)
    area = 0.5 * float(np.linalg.norm(np.cross(points[t[:, 1]] - points[t[:, 0]], points[t[:, 2]] - points[t[:, 0]]), axis=1).sum())
    result = patch.remesh_region(points, polys, np.arange(len(polys)), strategy=strategy, target_length=math.sqrt(area / max(goal, 1.0)), max_faces=max(200, int(goal * 2)))
    if not result.ok:
        raise ValueError(result.message or "the part is not one open surface")
    out = np.vstack([points, result.points])
    return out, [tuple(f) for f in result.faces], [kinds[0]] * len(result.faces)


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

    def __init__(self, context, co, faces, materials, detail: float, target_faces: int = 0, method: str = "AUTO", output: str = "QUADS", methods: dict | None = None):
        self.context = context
        self.co = np.asarray(co, dtype=np.float64)
        self.faces = list(faces)
        self.materials = [str(int(m)) for m in materials]
        self.detail = float(detail)
        self.target_faces = int(target_faces)  # 0 = keep the detail asked for; above 0 = land on this many faces
        self.method = method  # for every part without its own in `methods`
        self.output_kind = output  # TRIS, QUADS or MIXED
        self.methods: dict[int, str] = dict(methods or {})  # piece -> method
        self.notes: list[str] = []
        self.face_piece = np.zeros(0, dtype=np.int64)
        self.flag = np.zeros(0, dtype=bool)
        self._kd = None
        self._kd_owner = None
        self._src_check = None
        self._view = None
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
            if index in self.tuned:
                points, polys, kinds = self.tuned[index]
            elif index < len(self.sizes):
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
        else:
            co, faces, kinds, parts = final
        count = len(faces)
        self.output = {"co": co, "faces": faces, "new": np.ones(count, bool), "locked": np.zeros(count, bool), "mat": np.asarray([int(k) if str(k).isdigit() else 0 for k in kinds], dtype=np.int32), "parts": parts}
        if final is not None:
            self.output["piece"] = self.face_piece  # the source piece each face came from
            self.output["flag"] = self.flag  # faces still broken after the repair tries
        self.output_id += 1
        self.shown_at = time.perf_counter()

    def _pair(self, co, faces, kinds, method) -> None:
        try:
            self._paired = make_pure(co, faces, kinds, method, self.output_kind)
        except Exception as exc:  # a thread has nobody above it to report to
            self.warnings = [f"Auto Segments failed: {exc}"]

    def _build(self, co, faces, kinds, method, goal=0):
        """Generator: faces for these pieces by `method` in the chosen output. Returns (points, faces, materials) or None."""
        if method in INLINE:
            try:
                made = _quadriflow(self.context, co, faces, kinds, goal) if method == "QUADRIFLOW" else _fill(co, faces, kinds, method, goal)
                return triangulate(*made) if self.output_kind == "TRIS" else made
            except Exception as exc:
                self.notes.append(f"{method.title()} did not work on a part ({str(exc)[:60]}): Auto was used there")
                method = "AUTO"
        self._paired = None
        self._pair_thread = threading.Thread(target=self._pair, args=(co, faces, kinds, method), daemon=True, name="CurioMeshPair")
        self._pair_thread.start()
        while self._pair_thread.is_alive():
            if self._stopped:
                return None
            yield
        return self._paired

    def _method(self, piece: int) -> str:
        return self.methods.get(piece, self.method)

    def _vote(self, points, faces) -> np.ndarray:
        """The source piece of each result face: by the nearest source vertices of each connected part of the result."""
        if self._kd is None:
            owner = np.full(len(self.co), -1, dtype=np.int64)
            for index, ids in enumerate(self.pieces):
                for f in ids.tolist():
                    owner[list(self.faces[f])] = index
            used = np.flatnonzero(owner >= 0)
            kd = KDTree(len(used))
            for slot, v in enumerate(used.tolist()):
                kd.insert(Vector(self.co[v]), slot)
            kd.balance()
            self._kd, self._kd_owner = kd, owner[used]
        fa, fb = patch.face_adjacency(faces)
        labels = patch._labels(len(faces), fa, fb)
        order = np.argsort(labels, kind="stable")
        starts = np.concatenate(([0], np.flatnonzero(labels[order][1:] != labels[order][:-1]) + 1, [len(order)]))
        piece = np.zeros(len(faces), dtype=np.int64)
        for a, b in zip(starts[:-1], starts[1:]):
            ids = order[a:b]
            sample = ids[np.linspace(0, len(ids) - 1, min(len(ids), 24)).astype(np.int64)]
            votes = [int(self._kd_owner[self._kd.find(Vector(points[faces[f][0]]))[1]]) for f in sample.tolist()]
            piece[ids] = np.bincount(votes).argmax()
        return piece

    @staticmethod
    def _splice(points, faces, mats, piece, builds):
        """Drop every face of the pieces in `builds` ({piece: (points, faces, materials)}) and put the new ones in."""
        gone = np.fromiter(builds, dtype=np.int64)
        keep = ~np.isin(piece, gone)
        pieces_points, offset = [points], len(points)
        faces = [f for f, k in zip(faces, keep.tolist()) if k]
        mats = [m for m, k in zip(mats, keep.tolist()) if k]
        new_piece = []
        for p, (pts, fs, ms) in builds.items():
            pieces_points.append(pts)
            faces += [tuple(v + offset for v in f) for f in fs]
            mats += list(ms)
            new_piece += [p] * len(fs)
            offset += len(pts)
        piece = np.concatenate([piece[keep], np.asarray(new_piece, dtype=np.int64)])
        every = np.vstack(pieces_points)
        used, flat = np.unique(np.fromiter((v for f in faces for v in f), dtype=np.int64), return_inverse=True)
        out, at = [], 0
        for f in faces:
            out.append(tuple(int(x) for x in flat[at : at + len(f)]))
            at += len(f)
        return every[used], out, mats, piece

    def _remake(self, bad, points, faces, mats, piece, to_source):
        """Generator: remake these pieces, each one size up (or from the source's own faces), and splice them in."""
        builds = {}
        for p in bad:
            self.tuned.pop(p, None)
            sizes = self.sizes[p]
            if not to_source and self.picked[p] < len(sizes) - 1:
                self.picked[p] += 1
                co, polys, kinds = sizes[self.picked[p]]
                made = yield from self._build(co, polys, kinds, self._method(p), len(polys) // 2)
                if made is None:
                    return points, faces, mats, piece
            else:
                co, polys, kinds = sizes[-1]
                if len(polys) > SOURCE_BACK:
                    continue
                made = triangulate(co, polys, kinds) if self.output_kind == "TRIS" else (co, list(polys), list(kinds))
            builds[p] = made
        return self._splice(points, faces, mats, piece, builds)

    def _check(self, points, faces, mats, piece, loss):
        """Generator: measure the result against the source, repair the parts that got new problems, say what happened."""
        self.stage = "Checking for broken topology"
        yield
        if self._src_check is None:
            source_piece = np.zeros(len(self.faces), dtype=np.int64)
            for index, ids in enumerate(self.pieces):
                source_piece[ids] = index
            self._src_check = checks.inspect(self.co, self.faces, source_piece)
            yield
        before = self._src_check
        after = checks.inspect(points, faces, piece)
        fixed = 0
        keys = ("nm_edges", "nm_verts", "flipped", "dups", "holes", "selfx")
        first = {k: getattr(after, k) for k in keys}
        for attempt in range(FIX_TRIES):
            bad = checks.new_spots(before, after)[:MOST_REPAIRED]
            if not bad or self._stopped:
                break
            self.stage = f"Repairing {len(bad)} part{'s' if len(bad) != 1 else ''}, try {attempt + 1}"
            yield
            points, faces, mats, piece = yield from self._remake(bad, points, faces, mats, piece, attempt == FIX_TRIES - 1)
            after = checks.inspect(points, faces, piece)
            fixed += len(bad) - len(checks.new_spots(before, after))
            loss = None
        left = checks.new_spots(before, after)
        flag = np.zeros(len(faces), bool)
        for p in left:
            here = piece == p
            mask = here & (after.broken | after.crossing)
            flag |= mask if mask.any() else here
        if loss is None:
            center, half, reference, pixel = self._view
            loss = yield from self._look(points, faces, center, half, reference, pixel)
        self.face_piece, self.flag = piece, flag
        check = {"first": first, "text": checks.describe(before, after, max(fixed, 0), len(left)), "result": {k: getattr(after, k) for k in keys}, "source": {k: getattr(before, k) for k in keys}, "fixed": max(fixed, 0), "unfixed": len(left)}
        return points, faces, mats, loss, check

    def _conclude(self, points, faces, mats, loss, check):
        out_fa, out_fb = patch.face_adjacency(faces)
        parts = np.unique(patch._labels(len(faces), out_fa, out_fb), return_inverse=True)[1]
        self.report = {"before": len(self.faces), "faces": len(faces), "quads": sum(1 for f in faces if len(f) == 4), "tris": sum(1 for f in faces if len(f) == 3), "pieces": len(self.pieces), "loss": loss, "asked": self.target_faces, "seconds": time.perf_counter() - self.started, "check": check, "notes": list(dict.fromkeys(self.notes))}
        self.fraction = 1.0
        self._show((points, faces, mats, parts))

    def redo_pieces(self, pieces) -> None:
        """Remake these source pieces by their own method (after `methods` changed) and check the whole result again."""
        self._stopped = False
        self.busy = True
        self.fraction = 0.5
        self.notes = []
        self._search = self._redo(list(pieces))

    def _redo(self, pieces):
        out = self.output
        self.stage = "Remaking the part"
        yield
        builds = {}
        for p in pieces:
            self.tuned.pop(p, None)
            co, polys, kinds = self.sizes[p][self.picked[p]]
            made = yield from self._build(co, polys, kinds, self._method(p), len(polys) // 2)
            if made is None:
                return
            builds[p] = made
        points, faces, mats, piece = self._splice(out["co"], list(out["faces"]), [str(m) for m in out["mat"]], np.asarray(out["piece"]), builds)
        points, faces, mats, loss, check = yield from self._check(points, faces, mats, piece, None)
        self._conclude(points, faces, mats, loss, check)

    def _run(self):
        fa, fb = patch.face_adjacency(self.faces)
        shell = np.unique(patch._labels(len(self.faces), fa, fb), return_inverse=True)[1]
        order = np.argsort(shell, kind="stable")
        starts = np.concatenate(([0], np.flatnonzero(shell[order][1:] != shell[order][:-1]) + 1, [len(order)]))
        self.pieces = sorted((order[a:b] for a, b in zip(starts[:-1], starts[1:])), key=len, reverse=True)
        extent = float(np.ptp(self.co, axis=0).max()) or 1.0
        sizes = np.asarray([2.0 * 0.55 * float(np.ptp(self.co[np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))], axis=0).max()) for ids in self.pieces]) / extent
        picture_sizes = _picture_sizes(sizes).tolist()
        weights = np.asarray([sum(len(self.faces[f]) - 2 for f in ids.tolist()) for ids in self.pieces], dtype=np.float64)
        done_share = np.concatenate(([0.0], np.cumsum(weights))) / max(float(weights.sum()), 1.0)
        self._show()
        for index, ids in enumerate(self.pieces):
            self.progress = (index, len(self.pieces))
            self.fraction = MEASURE_SHARE * float(done_share[index])
            self.stage = f"Looking at piece {index + 1} of {len(self.pieces)}"
            yield from self._measure_piece(ids, picture_sizes[index])
            yield
            if self._stopped:
                return
            if time.perf_counter() - self.shown_at > SHOW_EVERY:
                self._show()
        self.progress = (len(self.pieces), len(self.pieces))
        self.fraction = MEASURE_SHARE
        self.stage = "Looking at the whole object"
        low, high = self.co.min(axis=0), self.co.max(axis=0)
        center, half = (low + high) / 2.0, extent * 0.55
        pixel = 2.0 * half / MAX_PIXELS
        reference = yield from pictures(self.co, self.faces, center, half, MAX_PIXELS)
        self._view = (center, half, reference, pixel)
        total_seen = sum(self.seen)
        asked = self.detail / 100.0
        allowed = asked * total_seen
        reducing = self.target_faces > 0
        paired_share = 1.0 if self.output_kind == "TRIS" else PAIRED_SHARE
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
            self.stage = f"Making the faces, try {round_index + 1}"
            co, faces, kinds, _parts = self._combined()
            before_pairing = len(faces)
            result = yield from self._make_all(co, faces, kinds)
            if result is None:
                return
            points, out_faces, out_mats = result
            self.fraction += 0.5 * (1.0 - MEASURE_SHARE) / ROUNDS
            picked, tuned = list(self.picked), dict(self.tuned)
            if reducing:
                miss = abs(len(out_faces) - self.target_faces) / self.target_faces
                if best is None or miss < best[0]:
                    best = (miss, points, out_faces, out_mats, None, picked, tuned)
                if miss <= CLOSE_ENOUGH:
                    break
                paired_share = len(out_faces) / max(before_pairing, 1)
                continue
            self.stage = f"Checking the view, try {round_index + 1}"
            loss = yield from self._look(points, out_faces, center, half, reference, pixel)
            fits = loss <= 1.1 * self.detail
            # Prefer anything that fits, then the fewest faces. With nothing fitting yet, the least change.
            rank = (not fits, len(out_faces) if fits else loss)
            if best is None or rank < best[0]:
                best = (rank, points, out_faces, out_mats, loss, picked, tuned)
            if abs(loss - self.detail) <= 0.1 * self.detail:
                break
            allowed *= min(max(self.detail / max(loss, 0.01), 0.3), 4.0)
        _rank, points, out_faces, materials, loss, self.picked, self.tuned = best
        if loss is None:  # Reduce To looks at the one it keeps, so the view change can still be judged
            self.stage = "Checking the view"
            self.fraction = 0.95
            loss = yield from self._look(points, out_faces, center, half, reference, pixel)
        piece = self._vote(points, out_faces)
        points, out_faces, materials, loss, check = yield from self._check(points, out_faces, materials, piece, loss)
        self._conclude(points, out_faces, materials, loss, check)

    def _make_all(self, co, faces, kinds):
        """Generator: the faces for the whole object. Pieces on the usual method go in one call; others one by one."""
        _co, _faces, _kinds, parts = self._combined()
        special = {i for i, m in self.methods.items() if m != self.method} | ({i for i in range(len(self.pieces))} if self.method in INLINE else set())
        if not special:
            return (yield from self._build(co, faces, kinds, self.method))
        results = []
        rest = [i for i in range(len(self.pieces)) if i not in special]
        groups = [(self.method, rest)] if rest else []
        groups += [(self._method(i), [i]) for i in sorted(special)]
        for method, members in groups:
            take = np.flatnonzero(np.isin(parts, members))
            if not len(take):
                continue
            used, flat = np.unique(np.fromiter((v for f in take.tolist() for v in faces[f]), dtype=np.int64), return_inverse=True)
            polys, at = [], 0
            for f in take.tolist():
                polys.append(tuple(int(x) for x in flat[at : at + len(faces[f])]))
                at += len(faces[f])
            made = yield from self._build(co[used], polys, [kinds[f] for f in take.tolist()], method, len(polys) // 2)
            if made is None:
                return None
            results.append(made)
        points, out, mats, offset = [], [], [], 0
        for pts, fs, ms in results:
            points.append(pts)
            out += [tuple(v + offset for v in f) for f in fs]
            mats += list(ms)
            offset += len(pts)
        return np.vstack(points), out, mats

    def _look(self, points, out_faces, center, half, reference, pixel):
        """Generator: the percent of the view that differs from the original. Returns it when exhausted."""
        view = yield from pictures(points, out_faces, center, half, MAX_PIXELS)
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
        self.tuned[index] = _decimate(self.context, points, polys, kinds, goal / triangles)

    def status(self) -> str:
        if self.report is None:
            return self.stage
        r = self.report
        text = f"{r['faces']:,} faces, {r['tris']} triangles, {r['pieces']:,} pieces | Camera: {r['loss']:.1f}% of the view differs. {r['seconds']:.0f} s"
        return text + (f" | {r['check']['text']}" if r.get("check") else "") + "".join(f" | {n}" for n in r.get("notes", []))
