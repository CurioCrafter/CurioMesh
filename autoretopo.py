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

from .triadq import checks, parts, patch, patterns
from .triadq.budget import Level, allocate, allocate_to_faces, best_levels, hottest, vertex_heat
from .triadq.fit import fit_vertices, vertex_parts
from .triadq.flatquad import _face_frames, flat_quad_remesh, flatten_regions, pair_and_close
from .triadq.refill import LOOSE, MIDDLE, STRICT, polish
from .triadq.shading import blended_normals, corner_normals, corner_table
from .triadq.smoothgrid import GridReport, crease_areas, grid_areas, smooth_areas
from . import guides, symmetry
from .triadq.types import Cancelled, MeshData
from .triadq.uvs import simplify_borders

VIEW_PIXELS = 2048  # the whole object's size in the camera's pixels: sets the smallest detail that counts
MAX_PIXELS = 192  # a side of one picture, so one huge piece can't take minutes per look
KEEP_TRIANGLES = 24  # pieces this small stay as they are: there's nothing to gain, and thousands of them to look at
MEASURE_SHARE = 0.6  # of the progress bar, spent looking at the pieces
LEVELS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # shares of a piece's triangles to try
PIECE_LIMIT = 3.0  # no piece may change by more than this many times the amount asked for
REBUILD_MARGIN = 1.5  # a rebuilt part's measured change counts this many times, as it looks better alone than in the whole view
QUAD_WEIGHT = 0.9  # a rebuilt quad counts as this many faces of a triangle level (those end up about 1.1 quads each)
GOOD_ENOUGH = 0.25  # stop trying bigger sizes of a piece once its change is this share of the amount asked for
FINER_TRIANGLES = 200  # pieces smaller than this have too few faces for a size between two others to matter
LOCKED_SHARE = 0.2  # vertices held in the first pass, as a share of those the piece ends with
HELD_SIZE = 1.5  # the first pass stops at this many times the faces asked for
ROUNDS = 4  # tries at landing on the amount asked for
CLOSE_ENOUGH = 0.10  # Reduce To: a face count this near the one asked for is a hit
PAIRED_SHARE = 0.6  # first guess of faces after pairing per face before it, corrected after each try
TURNED_DEG = 20.0
BORDER_TOLERANCE = 0.05  # how far a border between kinds may straighten, as a share of the level's edge length
SOFT_DEG = 10.0  # shallow folds in cloth only turn the shading this much
MOST_RAYS = 20_000_000  # one look at every piece, so a model of thousands of parts still finishes in minutes
RAYS_PER_YIELD = 2000
OTHER_VERTICES = 8  # how far down a sample's nearest vertices to look for one of its own loose part
SHOW_EVERY = 1.0  # seconds between preview updates
POLISH_LEVELS = (LOOSE, MIDDLE, STRICT)  # looser tidies more and moves the surface more; the camera decides
FIX_TRIES = 3  # repair tries on a part that got new problems: one size up each time, the last try the source's own faces
MOST_REPAIRED = 30  # parts remade per try, worst first, so a model of thousands of parts still finishes
SOURCE_BACK = 3000  # faces: only a part this small is put back as the source's own faces; a bigger one is left and selected
INLINE = ("QUADRIFLOW", "GRID", "FLOW")  # methods that must run on the main thread


def pictures(points: np.ndarray, polys, center: np.ndarray, half: float, pixels: int, tree: BVHTree | None = None):
    """Generator: six exact orthographic pictures, one ray per pixel. Returns [(hit, normal, depth, face)] when exhausted.

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
            at, normals, hit_faces, depths = [], [], [], []  # only the hits are kept: most of a picture is often empty
            for i, x in enumerate(grid):
                origin[u] = cu + x
                base = i * pixels
                for j, y in enumerate(grid):
                    origin[v] = cv + y
                    found = cast(origin, direction)
                    if found[0] is not None:
                        at.append(base + j)
                        normals.append(found[1][:])
                        hit_faces.append(found[2])
                        depths.append(found[3])
                since_yield += pixels
                if since_yield >= RAYS_PER_YIELD:
                    since_yield = 0
                    yield
            hit = np.zeros(pixels * pixels, dtype=bool)
            normal = np.zeros((pixels * pixels, 3))
            depth = np.zeros(pixels * pixels)
            face = np.zeros(pixels * pixels, dtype=np.int32)
            if at:
                hit[at] = True
                normal[at] = normals
                depth[at] = depths
                face[at] = hit_faces
            out.append((hit.reshape(pixels, pixels), normal.reshape(pixels, pixels, 3), depth.reshape(pixels, pixels), face.reshape(pixels, pixels)))
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


def make_pure(mesh: MeshData, method, output, **pairing):
    """The faces for one group of pieces by NumPy alone (runs on a worker thread): (points, faces, materials)."""
    if method == "FLAT":
        if output == "TRIS":
            out = flatten_regions(mesh)[0]
        else:
            out = flat_quad_remesh(mesh, force_quads=output == "QUADS")[0]
    elif output == "TRIS" or method == "TRIS":
        out = mesh
    else:
        out = pair_and_close(mesh, seed_count=4 if method == "PAIRING" else 2, force_quads=output == "QUADS", **pairing)[0]
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


DENSITY_STEPS = 24  # painted weights are rounded to this many levels, so a vertex group takes a few calls, not one per vertex


def _decimate(context, points, polys, kinds, ratio, drop=None, keep=None) -> tuple[np.ndarray, list[tuple[int, ...]], list[str]]:
    """Reduce to `ratio`. `drop` is the painted density per vertex; vertices in the mask `keep` are locked, as are UV borders."""

    free = None if keep is None else np.flatnonzero(~keep)
    if len(set(kinds)) > 1:  # borders stay where they are, so no face ends up across two UV islands
        triangles = sum(len(p) - 2 for p in polys)
        wanted = max(ratio * triangles, 1.0)
        triangle = float(_face_areas(points, polys).sum()) / wanted
        points, polys, kinds, unbound, used = simplify_borders(points, polys, kinds, BORDER_TOLERANCE * math.sqrt(triangle))
        drop = None if drop is None else drop[used]
        unbound = np.asarray(unbound, dtype=np.int64)
        free = unbound if keep is None else np.intersect1d(unbound, np.flatnonzero(~keep[used]))
        ratio = min(wanted / max(sum(len(p) - 2 for p in polys), 1), 1.0)
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
        if free is not None or painted:  # Decimate never collapses a vertex outside the group or of weight 0
            group = obj.vertex_groups.new(name="CurioMesh_Density")
            group.add(list(range(len(points))) if free is None else free.tolist(), 1.0, "REPLACE")
            modifier.vertex_group = group.name
        if painted:
            level = np.round(drop / 0.5 * DENSITY_STEPS).astype(np.int64)
            for step in np.unique(level[level > 0]).tolist():
                at = np.flatnonzero(level == step)
                group.add((at if free is None else np.intersect1d(at, free)).tolist(), 1.0 - step * 0.5 / DENSITY_STEPS, "REPLACE")
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


def _face_areas(points: np.ndarray, polys) -> np.ndarray:
    fan = np.asarray([(p[0], p[k], p[k + 1]) for p in polys for k in range(1, len(p) - 1)], dtype=np.int64)
    owner = np.repeat(np.arange(len(polys)), [len(p) - 2 for p in polys])
    tri = 0.5 * np.linalg.norm(np.cross(points[fan[:, 1]] - points[fan[:, 0]], points[fan[:, 2]] - points[fan[:, 0]]), axis=1)
    return np.bincount(owner, weights=tri, minlength=len(polys))


def _overlaps(points: np.ndarray, faces) -> int:
    """Pairs of faces in one piece that pass through each other. Separate parts often overlap on purpose."""

    tree = BVHTree.FromPolygons(points.tolist(), [list(f) for f in faces], all_triangles=False, epsilon=0.0)
    fa, fb = patch.face_adjacency(faces)
    piece = patch._labels(len(faces), fa, fb).tolist()
    ids = [set(f) for f in faces]
    return sum(1 for a, b in tree.overlap(tree) if a < b and piece[a] == piece[b] and not ids[a] & ids[b])


def _folded(points: np.ndarray, faces) -> int:
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    return int((patch.quad_quality(points[quads]) <= 0.0).sum())


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

    def __init__(self, context, co, faces, materials, detail: float, target_faces: int = 0, cache: dict | None = None, mirror: tuple | None = None, note: str = "", guide_lines: list | None = None, project=None, drop: np.ndarray | None = None, tidy: bool = False, method: str = "AUTO", output: str = "QUADS", methods: dict | None = None):
        self.tidy = tidy  # try to tidy the finished quads, kept only if the score rises and the camera allows it
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
        self.topology: float | None = None  # 0-100 against hand-made meshes: see triadq/patterns.py
        self.progress = (0, 0)
        self.stage = "Looking at the pieces..."
        self.started = time.perf_counter()
        self.shown_at = 0.0
        self._stopped = False
        self._pair_thread: threading.Thread | None = None
        self._paired = None
        self._labels = None
        self._shell = None
        self._samples = None
        self.pieces: list[np.ndarray] = []
        self.sizes: list[list[tuple]] = []  # per piece, smallest first: (points, polys, materials)
        self.levels: list[list[Level]] = []
        self.ready: list[list[bool]] = []  # per piece and size: already all quads, so it skips pairing
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
        if self.drop is None:
            return None
        used = np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))
        return self.drop[used]

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
            self.ready.append([False])
            self.seen.append(0.0)
            return
        key = ("piece", ids.tobytes(), None if self.drop is None else hash(self.drop.tobytes()), bool(self.guide_lines))
        known = self.cache.get(key)
        if known is not None and known[0] <= self.detail:  # measured with the same or a stricter amount (more sizes tried): nothing to redo
            self.sizes.append(known[1])
            self.levels.append(known[2])
            self.ready.append(known[4])
            self.seen.append(known[3])
            return
        low, high = points.min(axis=0), points.max(axis=0)
        center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
        pixel = 2.0 * half / pixels
        reference = yield from self._reference(center, half, pixels, points, polys)
        turned_sizes, turned_levels = [], []
        sizes, levels, seen, spots = [], [], 0.0, []
        enough = GOOD_ENOUGH * self.detail / 100.0
        if self.output_kind != "TRIS" and self._method(len(self.sizes)) == "AUTO" and not self.guide_lines and self.drop is None and len(set(kinds)) == 1:  # a rebuilt part would ignore strokes, paint and UV islands
            for out in parts.rebuilds(points, polys, max(set(kinds), key=kinds.count)):
                yield
                if out is None:
                    continue
                rebuilt = (out[0], parts.orient_like(out[0], out[1], points, polys), out[2])
                view = yield from self._timed("pictures", pictures(rebuilt[0], rebuilt[1], center, half, pixels))
                wrong, shown = picture_counts(reference, view, pixel)
                seen = max(seen, shown * pixel * pixel)
                turned_sizes.append(rebuilt)
                turned_levels.append(Level(len(rebuilt[1]) * QUAD_WEIGHT, REBUILD_MARGIN * wrong * pixel * pixel))
                if wrong <= enough * shown:
                    break
        for share in LEVELS:
            if share * triangles < 12.0:
                continue
            yield
            started = time.perf_counter()
            reduced = _decimate(self.context, points, polys, kinds, share, self._piece_drop(ids))
            self.timings["decimate"] = self.timings.get("decimate", 0.0) + time.perf_counter() - started
            view = yield from self._timed("pictures", pictures(reduced[0], reduced[1], center, half, pixels))
            wrong, shown = picture_counts(reference, view, pixel)
            area = pixel * pixel
            seen = max(seen, shown * area)
            sizes.append(reduced)
            levels.append(Level(len(reduced[1]), wrong * area))
            spots.append(wrong_points(reference, view, center, half, pixels, pixel))
            if wrong <= enough * shown:
                break
        if triangles >= FINER_TRIANGLES and seen > 0.0 and self.drop is None:  # painted vertices would not survive the first pass
            asked = self.detail / 100.0
            uppers = {next((k for k, level in enumerate(levels) if level.loss <= limit * seen), 0) for limit in (asked, PIECE_LIMIT * asked)} - {0}
            for upper in sorted(uppers):
                # A size halfway across the jump, so a piece doesn't get twice the faces to clear a limit.
                yield
                faces = math.sqrt(levels[upper - 1].faces * levels[upper].faces)
                reduced = _decimate(self.context, points, polys, kinds, faces / len(polys))
                view = yield from self._timed("pictures", pictures(reduced[0], reduced[1], center, half, pixels))
                sizes.append(reduced)
                levels.append(Level(len(reduced[1]), picture_counts(reference, view, pixel)[0] * area))
                spots.append(wrong_points(reference, view, center, half, pixels, pixel))
                # Density where the camera saw it go wrong: hold those spots while the rest goes further, then let go.
                for k in (upper - 1, upper, len(levels) - 1):
                    yield
                    keep = hottest(vertex_heat(points, spots[k], 3.0 * pixel), LOCKED_SHARE * levels[k].faces / 2.0)
                    held = _decimate(self.context, points, polys, kinds, HELD_SIZE * levels[k].faces / triangles, keep=keep)
                    reduced = _decimate(self.context, held[0], held[1], held[2], levels[k].faces / max(len(held[1]), 1))
                    view = yield from self._timed("pictures", pictures(reduced[0], reduced[1], center, half, pixels))
                    sizes.append(reduced)
                    levels.append(Level(len(reduced[1]), picture_counts(reference, view, pixel)[0] * area))
                    spots.append(spots[k])
            kept = best_levels(levels)
            sizes, levels = [sizes[k] for k in kept], [levels[k] for k in kept]
        ready = [True] * len(turned_sizes) + [False] * (len(sizes) + 1)
        sizes = turned_sizes + sizes + [whole]
        levels = turned_levels + levels + [Level(len(polys), 0.0)]
        order = sorted(range(len(sizes)), key=lambda k: levels[k].faces)
        sizes, levels, ready = [sizes[k] for k in order], [levels[k] for k in order], [ready[k] for k in order]
        self.sizes.append(sizes)
        self.levels.append(levels)
        self.ready.append(ready)
        self.seen.append(seen)
        self.cache[key] = (self.detail, sizes, levels, seen, ready)

    def _reference(self, center, half, pixels, points, polys):
        """Generator: the six pictures of the original at this view. A lone piece is the whole object, so they are made once."""
        key = ("view", center.tobytes(), half, pixels, len(polys), hash(points.tobytes()))
        if key not in self.cache:
            self.cache[key] = yield from self._timed("pictures", pictures(points, polys, center, half, pixels))
        return self.cache[key]

    def _combined(self, rebuilt: bool | None = None) -> tuple[np.ndarray, list[tuple[int, ...]], list[str], np.ndarray]:
        """Every piece at its picked size, and any not measured yet as it is. `rebuilt` True or False keeps only those."""
        all_points, all_faces, all_kinds, owner = [], [], [], []
        offset = 0
        for index, ids in enumerate(self.pieces):
            if index in self.tuned:
                if rebuilt:
                    continue
                points, polys, kinds = self.tuned[index]
            elif index < len(self.sizes):
                at = self.picked[index] if index < len(self.picked) else 0
                if rebuilt is not None and self.ready[index][at] != rebuilt:
                    continue
                points, polys, kinds = self.sizes[index][at]
            elif rebuilt:
                continue
            else:
                points, polys, kinds = self._piece_arrays(ids)
            all_points.append(points)
            all_faces += [tuple(v + offset for v in p) for p in polys]
            all_kinds += kinds
            owner += [index] * len(polys)
            offset += len(points)
        return np.vstack(all_points) if all_points else np.zeros((0, 3)), all_faces, all_kinds, np.asarray(owner, dtype=np.int64)

    def _show(self, final=None) -> None:
        if final is None:
            co, faces, kinds, parts = self._combined()
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
        if final is not None and self.mirror is None:
            self.output["piece"] = self.face_piece  # the source piece each face came from
            self.output["flag"] = self.flag  # faces still broken after the repair tries
        self.output_id += 1
        self.shown_at = time.perf_counter()

    def _source_samples(self):
        """Each source triangle as a point: (positions, normals, areas, loose parts, the source's tree). Made once."""
        if self._samples is None:
            sizes = np.fromiter((len(f) for f in self.faces), dtype=np.int64, count=len(self.faces))
            triangles = np.asarray([(f[0], f[i], f[i + 1]) for f in self.faces for i in range(1, len(f) - 1)], dtype=np.int64)
            parts = np.repeat(self._shell, sizes - 2)
            a, b, c = self.co[triangles[:, 0]], self.co[triangles[:, 1]], self.co[triangles[:, 2]]
            cross = np.cross(b - a, c - a)
            twice = np.linalg.norm(cross, axis=1)
            real = twice > 0.0
            tree = BVHTree.FromPolygons(self.co.tolist(), [list(f) for f in self.faces], all_triangles=False, epsilon=0.0)
            self._samples = (((a + b + c) / 3.0)[real], cross[real] / twice[real][:, None], twice[real] / 2.0, parts[real], tree)
        return self._samples

    def _fit(self, mesh: MeshData) -> MeshData:
        """Fit the vertices to their share of the source, part by part, and keep it where the part really got closer.

        Idea from cav3dw3ll3r: each source sample goes to its nearest vertex, so a vertex's share grows with its quads.
        """
        quads = np.asarray([f for f in mesh.faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
        if not len(quads):
            return mesh
        points = mesh.vertices
        position, normal, area, sample_part, source_tree = self._source_samples()
        tree = KDTree(len(points))
        for index, point in enumerate(points.tolist()):
            tree.insert(point, index)
        tree.balance()
        source_parts = int(sample_part.max()) + 1
        if source_parts == 1:
            vertex_part, source_of = vertex_parts(points, mesh.faces, sample_part, None)
            owner = np.fromiter((tree.find(p)[1] for p in position.tolist()), dtype=np.int64, count=len(position))
        else:
            # A result part came from one source part, and samples of another part, however near, say nothing about it.
            sample_tree = KDTree(len(position))
            for index, point in enumerate(position.tolist()):
                sample_tree.insert(point, index)
            sample_tree.balance()
            vertex_part, source_of = vertex_parts(points, mesh.faces, sample_part, lambda p: sample_tree.find(p)[1])
            vertex_source = source_of[vertex_part]
            owner = np.full(len(position), -1, dtype=np.int64)
            for index, (point, part) in enumerate(zip(position.tolist(), sample_part.tolist())):
                for _found, vertex, _away in tree.find_n(point, OTHER_VERTICES):
                    if vertex_source[vertex] == part:
                        owner[index] = vertex
                        break
        count = len(source_of)
        moved = fit_vertices(points, quads, position, normal, area, owner)

        def gaps(where: np.ndarray) -> np.ndarray:
            """Per result part, the mean distance from its source to it plus from it to its source."""
            result_tree = BVHTree.FromPolygons(where.tolist(), [list(f) for f in mesh.faces], all_triangles=False, epsilon=0.0)
            to_result = np.fromiter((result_tree.find_nearest(p)[3] for p in position.tolist()), dtype=np.float64, count=len(position))
            to_source = np.fromiter((source_tree.find_nearest(p)[3] for p in where.tolist()), dtype=np.float64, count=len(where))
            weighted = np.bincount(sample_part, area * to_result, minlength=source_parts) / np.maximum(np.bincount(sample_part, area, minlength=source_parts), 1e-30)
            back = np.bincount(vertex_part, to_source, minlength=count) / np.maximum(np.bincount(vertex_part, minlength=count), 1)
            return weighted[source_of] + back

        worse = gaps(moved) >= gaps(points)
        moved[worse[vertex_part]] = points[worse[vertex_part]]
        return MeshData(vertices=moved, faces=mesh.faces, face_materials=mesh.face_materials, name=mesh.name)

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

    def _pair(self, co, faces, kinds, method) -> None:
        started = time.perf_counter()
        try:
            coarse = MeshData(co, faces, kinds)
            # Rows of quads would ignore the strokes and the paint, and they are what Auto means.
            if method == "AUTO" and self.output_kind != "TRIS" and not self.guide_lines and self.drop is None:
                if self._labels is None:
                    domes = smooth_areas(self.co, self.faces)
                    self._labels = (domes, crease_areas(self.co, self.faces, domes >= 0))
                coarse = smooth_grids(self.co, self.faces, co, faces, kinds, self._labels)[0]
            made = make_pure(coarse, method, self.output_kind, cancelled=lambda: self._stopped, guide_edges=self.guide_edges, keep_quads=True)
            # Strokes and a mirror seam must stay where they are.
            if self.output_kind != "TRIS" and method != "TRIS" and not self.guide_lines and self.mirror is None and not self._stopped:
                made = (self._fit(MeshData(*made)).vertices, made[1], made[2])
            self._paired = made
            self.timings["pairing"] = self.timings.get("pairing", 0.0) + time.perf_counter() - started
        except Cancelled:  # Stop was pressed: the thread ends at its next stopping point
            pass
        except Exception as exc:  # a thread has nobody above it to report to
            self.warnings = [f"Auto Segments failed: {exc}"]

    def _score(self, points, faces) -> None:
        try:
            self.topology = patterns.score(patterns.measure(points, faces))["score"]
        except Exception as exc:  # a thread has nobody above it to report to; the score is only a report
            self.warnings = [f"No tidiness score: {exc}"]

    def _polish(self, points, faces, materials, level) -> None:
        """Leaves the tidied mesh in `_paired` only if the score says it is tidier and nothing got broken."""
        try:
            before = patterns.measure(points, faces)
            tidied = polish(MeshData(points, faces, materials), level)[0]
            after = patterns.measure(tidied.vertices, tidied.faces)
            if patterns.score(after)["score"] > patterns.score(before)["score"] and after["non_manifold_edges"] <= before["non_manifold_edges"] and _folded(tidied.vertices, tidied.faces) <= _folded(np.asarray(points), faces):
                self._paired = tidied
        except Exception as exc:  # a thread has nobody above it to report to
            self.warnings = [f"Tidying failed, kept the quads as they were: {exc}"]

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
            self._pair_thread.join(0.02)  # sleep instead of spinning: a busy main thread slows the pairing thread (shared GIL)
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
                if self.ready[p][self.picked[p]]:  # a rebuilt part is already quads
                    made = (co, list(polys), list(kinds))
                else:
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

    def _check(self, points, faces, mats, piece, view):
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
            view = None
        left = checks.new_spots(before, after)
        flag = np.zeros(len(faces), bool)
        for p in left:
            here = piece == p
            mask = here & (after.broken | after.crossing)
            flag |= mask if mask.any() else here
        if view is None:
            view = yield from self._look(points, faces, *self._view)
        self.face_piece, self.flag = piece, flag
        check = {"first": first, "text": checks.describe(before, after, max(fixed, 0), len(left)), "result": {k: getattr(after, k) for k in keys}, "source": {k: getattr(before, k) for k in keys}, "fixed": max(fixed, 0), "unfixed": len(left)}
        return points, faces, mats, view, check

    def _run(self):
        fa, fb = patch.face_adjacency(self.faces)
        shell = np.unique(patch._labels(len(self.faces), fa, fb), return_inverse=True)[1]
        self._shell = shell
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
        reference = yield from self._reference(center, half, MAX_PIXELS, self.co, self.faces)
        reference_shaded = shaded(reference, self.co, self.faces, center, half, MAX_PIXELS)
        self._view = (center, half, reference, reference_shaded, pixel)
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
            co, faces, kinds, _parts = self._combined(False)
            rco, rfaces, rkinds, _owner = self._combined(True)  # rebuilt parts are already quads
            if self.guide_lines:
                self.stage = f"Cutting the guide curves in, try {round_index + 1}"
                yield
                co, faces, kinds, self.guide_edges = self._guided(co, faces, kinds)
            before_pairing = len(faces) + len(rfaces)
            result = (np.zeros((0, 3)), [], [])
            if faces:
                result = yield from self._make_all(co, faces, kinds)
                if result is None:
                    return
            points, out_faces, out_kinds = result
            out_faces += [tuple(v + len(points) for v in f) for f in rfaces]
            out_kinds += rkinds
            points = np.vstack([points, rco])
            self.fraction += 0.5 * (1.0 - MEASURE_SHARE) / ROUNDS
            picked, tuned = list(self.picked), dict(self.tuned)
            if reducing:
                miss = abs(len(out_faces) - self.target_faces) / self.target_faces
                if best is None or miss < best[0]:
                    best = (miss, points, out_faces, out_kinds, None, None, None, picked, tuned)
                if miss <= CLOSE_ENOUGH:
                    break
                paired_share = len(out_faces) / max(before_pairing, 1)
                continue
            self.stage = f"Checking the result, try {round_index + 1}"
            loss, shading, soft = yield from self._look(points, out_faces, center, half, reference, reference_shaded, pixel)
            fits = loss <= 1.1 * self.detail
            # Prefer anything that fits, then the fewest faces. With nothing fitting yet, the least change.
            rank = (not fits, len(out_faces) if fits else loss)
            if best is None or rank < best[0]:
                best = (rank, points, out_faces, out_kinds, loss, shading, soft, picked, tuned)
            if abs(loss - self.detail) <= 0.1 * self.detail:
                break
            allowed *= min(max(self.detail / max(loss, 0.01), 0.3), 4.0)
        _rank, points, out_faces, materials, loss, shading, soft, self.picked, self.tuned = best
        if loss is None:  # Reduce To looks at the one it keeps, so the view change can still be judged
            self.stage = "Checking the view"
            self.fraction = 0.95
            loss, shading, soft = yield from self._look(points, out_faces, center, half, reference, reference_shaded, pixel)
        for level in POLISH_LEVELS if self.tidy and self.output_kind != "TRIS" else ():  # an option: it is slow and the loss goes up a little
            self.stage = "Tidying the quads..."
            self._paired = None
            self._pair_thread = threading.Thread(target=self._polish, args=(points, out_faces, materials, level), daemon=True, name="CurioMeshPolish")
            self._pair_thread.start()
            while self._pair_thread.is_alive():
                if self._stopped:
                    return
                self._pair_thread.join(0.02)  # this is many small Python steps: give it the interpreter
                yield
            if self._paired is None:
                continue
            tidied = self._paired
            tidied_points = np.asarray(tidied.vertices, dtype=np.float64)
            tidied_faces = [tuple(face) for face in tidied.faces]
            tidied_loss, tidied_shading, tidied_soft = yield from self._look(tidied_points, tidied_faces, center, half, reference, reference_shaded, pixel)
            # Passing through itself is worse than any loose quad, and the camera can't see it.
            if tidied_loss <= max(1.1 * self.detail, loss) and _overlaps(tidied_points, tidied_faces) <= _overlaps(points, out_faces):
                points, out_faces, materials, loss, shading, soft = tidied_points, tidied_faces, list(tidied.face_materials), tidied_loss, tidied_shading, tidied_soft
                break
        piece = self._vote(points, out_faces)
        points, out_faces, materials, (loss, shading, soft), check = yield from self._check(points, out_faces, materials, piece, (loss, shading, soft))
        self.stage = "Scoring the quads"
        if any(len(face) == 4 for face in out_faces):
            self._pair_thread = threading.Thread(target=self._score, args=(points, out_faces), daemon=True, name="CurioMeshScore")
            self._pair_thread.start()
            while self._pair_thread.is_alive():
                if self._stopped:
                    return
                self._pair_thread.join(0.02)
                yield
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
        self.report = {"before": whole, "faces": len(shown), "quads": quads, "tris": sum(1 for face in shown if len(face) == 3), "pieces": len(self.pieces), "loss": loss, "shading": shading, "shading_soft": soft, "topology": self.topology, "asked": self.target_faces * (2 if self.mirror is not None else 1), "seconds": time.perf_counter() - self.started, "timings": {k: round(v, 1) for k, v in self.timings.items()}, "check": check, "notes": list(dict.fromkeys(self.notes)), "guides": follows}

    def _make_all(self, co, faces, kinds):
        """Generator: the faces for the whole object. Pieces on the usual method go in one call; others one by one."""
        parts = self._combined(False)[3]
        special = {i for i, m in self.methods.items() if m != self.method} | ({i for i in range(len(self.pieces))} if self.method in INLINE else set())
        if not special or self.guide_lines:  # with strokes the faces have changed: one call
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

    def _look(self, points, out_faces, center, half, reference, reference_shaded, pixel):
        """Generator: percent of the view that differs in shape, and in smooth shading past 20 and 10 degrees, when exhausted."""
        view = yield from self._timed("pictures", pictures(points, out_faces, center, half, MAX_PIXELS))
        wrong, shown = picture_counts(reference, view, pixel)
        view_shaded = shaded(view, points, out_faces, center, half, MAX_PIXELS)
        bad_shade, seen_both = shading_counts(reference, view, reference_shaded, view_shaded)
        soft = shading_counts(reference, view, reference_shaded, view_shaded, SOFT_DEG)[0]
        return 100.0 * wrong / max(shown, 1), 100.0 * bad_shade / max(seen_both, 1), 100.0 * soft / max(seen_both, 1)

    def _tune(self, want: float) -> None:
        """Reduce To: the sizes come in steps of two, so remake the biggest picked piece to land closer on `want`."""
        total = sum(levels[k].faces for levels, k in zip(self.levels, self.picked))
        miss = want - total
        if abs(miss) <= 0.03 * want:
            return
        made = [i for i in range(len(self.picked)) if not self.ready[i][self.picked[i]]]  # leave a rebuilt part alone while another can take it
        index = max(made or range(len(self.picked)), key=lambda i: self.levels[i][self.picked[i]].faces)
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
        shade = "" if r.get("shading") is None else f", {r['shading']:.1f}% in smooth shading"
        tidy = "" if r.get("topology") is None else f"Tidiness: {r['topology']:.0f}/100. "
        text = f"{r['faces']:,} faces, {r['tris']} triangles, {r['pieces']:,} pieces | Camera: {r['loss']:.1f}% of the view differs in shape{shade}. {tidy}{r['seconds']:.0f} s{extra}"
        return text + (f" | {r['check']['text']}" if r.get("check") else "") + "".join(f" | {n}" for n in r.get("notes", []))
