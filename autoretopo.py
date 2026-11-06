# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Auto Segments: you say how much of the view may change, and it finds the fewest faces that hold it.

Each loose piece is reduced to several sizes and looked at from six sides, with one fixed pixel size for the whole
object, so a bolt is judged by how big it looks on the object, not blown up to fill the picture. Faces then go where
they remove the most visible change (`budget.allocate`), with a limit no single piece may pass. The pieces are paired
into quads, the whole result is looked at, and the sharing is redone until it lands on what was asked for.

Everything runs on Blender's main thread. The search is a generator that runs in short slices from a timer, so Blender
stays usable between them; a single slow NumPy step (such as the pairing) runs as one slice.
"""

from __future__ import annotations

import atexit
import dataclasses
from itertools import chain
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector, geometry
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .triadq import tasks
from .triadq import bakescore, cases, checks, detail, hardsurf, issues, parts, patch, patterns, redo, repair, special, views
from .triadq.chords import ChordReducer, PictureJudge
from .triadq.budget import Level, allocate, allocate_to_faces, best_levels, hottest, next_ask, vertex_heat
from .triadq.fit import fit_vertices, vertex_parts
from .triadq.flatquad import FlatQuadOptions, _close_pieces, _face_frames, flat_quad_remesh, flatten_regions, pair_and_close
from .triadq.flow import faces_that_gain, seek_features, snap_new_vertices, snap_rows, split_faces, unfold
from .triadq.quadreduce import quad_reducer
from .triadq.refill import LOOSE, MIDDLE, STRICT, polish
from .triadq.shading import blended_normals, corner_normals, corner_table
from .triadq.fieldmesh import MOST_TARGET, field_quads, is_clean
from .triadq.smoothgrid import GridOptions, GridReport, crease_areas, grid_areas, smooth_areas
from . import guides, hardware, loops, parallel, symmetry
from .triadq.types import Cancelled, FlatFaces, MeshData
from .triadq.uvs import simplify_borders

VIEW_PIXELS = 2048  # the whole object's size in the camera's pixels: sets the smallest detail that counts
MAX_PIXELS = 192  # a side of one picture, so one huge piece can't take minutes per look
KEEP_TRIANGLES = 24  # pieces this small stay as they are: there's nothing to gain, and thousands of them to look at
MEASURE_SHARE = 0.6  # of the progress bar, spent looking at the pieces
LEVELS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # shares of a piece's triangles to try
BAKED_LEVELS = (0.0025, 0.005, 0.01) + LEVELS  # a normal map takes over the bumps, so far smaller sizes have a chance
BAKED_MOST_FLAT = 0.06  # Judge After Baking is only for scans: at most this share of flat edges, as the Auto button tells a scan
PIECE_LIMIT = 3.0  # no piece may change by more than this many times the amount asked for
HOPELESS = 1.5  # a rebuild this many times over the limit for a piece won't get picked, so smaller ones aren't tried
KEEP_SHARE = 0.25  # a piece with over this share of the faces is the body of the model, and always stays
REBUILD_MARGIN = 1.5  # a rebuilt part's measured change counts this many times, as it looks better alone than in the whole view
FIELD_CASES = ("flat plate", "thin sheet", "box")  # part cases Auto offers rows to
FIELD_SMALLEST = 1000  # triangles: a part with fewer isn't worth laying a grid on while Auto looks for the best
FIELD_MOST = 8000  # quads Auto may ask of it in one look, as it does this for every large part
FIELD_SECONDS = 60.0  # Auto stops offering rows after spending this long on them in one run
FIELD_START = 0.1  # share of a part's triangles, as quads, that the Field Rows method tries first
FIELD_FIRST = 1500.0  # and at most this many: looking costs far more the more quads there are
FIELD_LANDED = 0.75  # of the change asked for: near enough to stop looking
FIELD_TRIES = 4
FIELD_AIM = 0.9
QUAD_WEIGHT = 0.9  # a rebuilt quad counts as this many faces of a triangle level (those end up about 1.1 quads each)
GOOD_ENOUGH = 0.25  # stop trying bigger sizes of a piece once its change is this share of the amount asked for
FINER_TRIANGLES = 200  # pieces smaller than this have too few faces for a size between two others to matter
LOCKED_SHARE = 0.2  # vertices held in the first pass, as a share of those the piece ends with
HELD_SIZE = 1.5  # the first pass stops at this many times the faces asked for
ROUNDS = 4  # tries at landing on the amount asked for
ROWS_SMALLEST = 32  # Even Rows: a piece due fewer quads than this is paired instead
ROWS_START = 0.04  # Even Rows: first try, as a share of the source's triangles
ROWS_MOST = 0.8  # Even Rows: most faces, as a share of the source's triangles
ROWS_SECONDS = 30.0  # Even Rows: QuadriFlow's time limit per piece, plus...
ROWS_SECONDS_PER_QUAD = 0.01  # ...this much for each quad asked for
EVEN_FLAT_UP_TO = 150_000  # faces before pairing: past this, flat areas keep the long triangles reduction left (Python, about 1 s per 10k)
EVEN_FLAT = FlatQuadOptions(tolerance=0.0002, angle_deg=8.0, refine=True, min_faces=4, min_saving=-3.0)  # points added inside flat areas, so no long thin quads cross them
FLAT_SNAP_SHARE = 0.015  # Balanced: a face whose new vertices would move less than this share of its size stays whole
ROWS_TIME = 600.0  # Even Rows: a denser layout is only asked for if it should finish within this many seconds
ROWS_SECONDS_ESTIMATE = 0.003  # Even Rows: rough QuadriFlow time per quad, measured on a 74,000 quad rock
ROWS_SHARP_INPUT = 6  # Even Rows: source triangles per quad needed before vertices seek creases
ROWS_BAD = 0.05  # Even Rows: a piece where repair leaves more than this many triangles per quad gets paired instead
ROWS_ROUNDS = 3  # Even Rows: layouts tried, at most
CLOSE_ENOUGH = 0.10  # Reduce To: a face count this near the one asked for is a hit
PAIRED_SHARE = 0.6  # first guess of faces after pairing per face before it, corrected after each try
TURNED_DEG = 20.0
BORDER_TOLERANCE = 0.05  # how far a border between kinds may straighten, as a share of the level's edge length
SOFT_DEG = 10.0  # shallow folds in cloth only turn the shading this much
MOST_RAYS = 20_000_000  # one look at every piece, so a model of thousands of parts still finishes in minutes
RAYS_PER_YIELD = 2000
OTHER_VERTICES = 8  # how far down a sample's nearest vertices to look for one of its own loose part
REDUCE_STEP = 0.16  # share of the quads to try taking away between looks; halved on going too far, down to REDUCE_FINEST
REDUCE_FINEST = 0.015
MOST_LOOKS = 40
CHORDS_FINEST, CHORDS_LOOKS = 0.004, 60  # with Redraw Messy Quads, which can land closer
NEARLY = 0.7  # when a part passes its limit, parts this close to theirs stop too
CHORDS_UP_TO = 120_000  # quads: the redrawing reducer up to here, the plain quad reducer past it
AIM = 0.9  # share of the amount asked for that a start should reach
ROOM = 0.5  # a start this close to the amount asked for is good enough: taking quads away uses up the rest
CHORDS_START = (0.4, 0.5, 0.25)  # with Redraw Messy Quads: start share, AIM and ROOM. Starting well inside what was asked for leaves the redraw room
CHORDS_SECONDS = 240.0  # time for redrawing and judging per run; plain collapses go on after it
CHORDS_MESSY = 0.3  # a part is redrawn when more than this share of its corners are poles
TURN_UP_TO = 150_000  # quads: edges are spun at poles up to here (plain Python, about 3 s per 15,000 quads)
TURN_LOSS = 1.0  # points of view change spinning edges may add, staying within what Keep Detail counts as a fit
HARD = {"panels": False, "cylinders": False}  # hard-surface rebuilds when the caller says nothing (hard_surface.job_flags)
TWEAK = {"rows": 0.5, "flat": 0.5, "poles": 1}  # the Tweak settings as they are by default: what the constants above give (poles: 1 on, 0 off)


def grid_options(rows: float) -> GridOptions | None:
    """Rows on Round Areas: 0 builds none, 0.5 the defaults, 1 finds round areas with half the curvature and lets them
    add twice the faces."""
    if rows <= 0.0:
        return None
    f = 2.0 ** (2.0 * min(rows, 1.0) - 1.0)
    base = GridOptions()
    return dataclasses.replace(base, min_total_deg=base.min_total_deg / f, min_size=base.min_size / f, max_added=base.max_added * f)


def flat_options(flat: float) -> FlatQuadOptions | None:
    """Flat Area Size: left adds more points inside flat areas (they grow from the edges more slowly, and regions with
    less thin triangles get them), 0.5 is EVEN_FLAT, all the way right adds none."""
    if flat >= 1.0:
        return None
    f = 2.0 ** (2.0 * max(flat, 0.0) - 1.0)
    return dataclasses.replace(EVEN_FLAT, grading=EVEN_FLAT.grading * f * f, thin=EVEN_FLAT.thin * f)
PART_PIXELS = 100  # a part showing fewer pixels than this over the six views can't be judged by itself
REFINE_PIECES = 8
REFINE_COVER = 0.8  # share of the faces the extra sizes need to cover
USE_CASES = True  # let the table of part cases say which size of a part to look at first
POLISH_LEVELS = (LOOSE, MIDDLE, STRICT)  # looser tidies more and moves the surface more; the camera decides
FIX_TRIES = 3  # repair tries on a part that got new problems: one size up each time, the last try the source's own faces
MOST_REPAIRED = 30  # parts remade per try, worst first, so a model of thousands of parts still finishes
REPAIR_MOST_FACES = 200_000  # a part bigger than this is left and its spots selected: remaking it costs minutes for a few faces
UNTANGLE_SECONDS = 60.0  # safety net only: turning folded faces back ends after its rounds (a big scan takes a few seconds)
REPAIR_SECONDS = 30.0  # time for clearing crossings per run; what's left is selected on Apply (the crane spent 215 s for 71 spots)
REPAIR_GROWTH = 1.03  # a repair may not add more than this share of faces to the whole result
REDO_SECONDS = 60.0  # Redo Rough Spots: time per run; the spots not reached stay as they are
REDO_ROUNDS = 2  # the second round goes back to spots still rough, in a region one ring wider
REDO_SPOTS = 200  # spots looked at per round, worst first
REDO_RINGS = 2  # faces round a spot taken into its region, in rings of neighbours
REDO_MOST_FACES = 600  # a region never gets bigger than this
REDO_SHARES = (0.5, 1.0, 0.25)  # per region: how far its inner corners are put back towards the source, each tried
SOURCE_BACK = 3000  # faces: only a part this small is put back as the source's own faces; a bigger one is left and selected
BLENDER_OPERATOR_METHODS = ("QUADRIFLOW", "GRID", "FLOW")  # these call Blender operators, so each part is built on its own
STEP = 0.8  # each size made in steps is this share of the size before it


def pictures(points: np.ndarray, polys, center: np.ndarray, half: float, pixels: int, tree: BVHTree | None = None, baked_from: BVHTree | None = None):
    """Generator: six exact orthographic pictures, one ray per pixel. Returns [(hit, normal, depth, face)] when exhausted.

    `tree` is the BVH of this very mesh when the caller already has one (building it is a fifth of a picture set).
    With `baked_from`, each normal is the source's nearest, what a bake paints there.
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
                        near = found if baked_from is None else baked_from.find_nearest(found[0])
                        normals.append((found[1] if near[1] is None else near[1])[:])
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


def picture_counts(reference, other, pixel: float, baked: bool = False) -> tuple[int, int]:
    """(pixels a viewer would see as different in outline, direction or depth, pixels showing the object)."""

    if baked:
        return bakescore.picture_counts(reference, other, pixel)
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


def make_pure(mesh: MeshData, method, output, notes=None, **pairing):
    """The faces for one group of pieces by NumPy alone, no Blender data: (points, faces, materials)."""
    if method == "FLAT":
        if output == "TRIS":
            out = flatten_regions(mesh)[0]
        else:
            out = flat_quad_remesh(mesh, force_quads=output == "QUADS")[0]
    elif output == "TRIS" or method == "TRIS":
        out = mesh
    else:
        out, solver = pair_and_close(mesh, seed_count=4 if method == "PAIRING" else 2, force_quads=output == "QUADS", **pairing)
        if solver.forced_pieces and notes is not None:
            notes.append(f"{solver.forced_pieces} part(s) only closed by the last-resort split: up to 3-4x the faces and poor quads there, so check them")
    points, out_faces, mats = np.asarray(out.vertices, dtype=np.float64), [tuple(p) for p in out.faces], list(out.face_materials)
    if output == "TRIS" or method == "TRIS":
        points, out_faces, mats = triangulate(points, out_faces, mats)
    return points, out_faces, mats


def _quadriflow(context, points, polys, kinds, goal):
    context = bpy.context  # the operator's context, kept by the job, is gone by now
    mesh = bpy.data.meshes.new("CurioMesh_Piece")
    mesh.from_pydata(points.tolist(), [], polys)
    mesh.update()
    obj = bpy.data.objects.new("CurioMesh_Piece", mesh)
    context.scene.collection.objects.link(obj)
    layer = context.view_layer
    was_selected, was_active = list(layer.objects.selected), layer.objects.active
    try:
        for other in was_selected:
            other.select_set(False)
        obj.select_set(True)
        layer.objects.active = obj
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
        for other in was_selected:
            other.select_set(True)
        layer.objects.active = was_active


def _even_border(points, polys):
    """(points, polys) with one more vertex on the longest border edge when the part's one border loop is odd.

    Grid and Flow need an even seam. Here the seam is the part's own open border, with no face on the other side,
    so a midpoint there changes nothing else; how decimation left the border decided it before (it failed at every
    Detail on a test sheet)."""
    loops, pinched = patch.boundary_loops(polys, np.arange(len(polys)))
    if pinched or len(loops) != 1 or len(loops[0]) % 2 == 0:
        return points, polys
    loop = loops[0]
    edges = list(zip(loop, loop[1:] + loop[:1]))
    a, b = max(edges, key=lambda e: float(np.linalg.norm(points[e[0]] - points[e[1]])))
    mid = len(points)
    out = []
    for poly in polys:
        poly = list(poly)
        for k in range(len(poly)):
            if {poly[k], poly[(k + 1) % len(poly)]} == {a, b}:
                poly.insert(k + 1, mid)
                break
        out.append(tuple(poly))
    return np.vstack([points, 0.5 * (points[a] + points[b])]), out


def _fill(points, polys, kinds, strategy, goal):
    """Grid or Flow fill over a whole part: only works when the part is one open surface with a single edge loop."""
    points, polys = _even_border(np.asarray(points, dtype=np.float64), polys)
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


def parts_changed_too_much(reference, view, pixel: float, labels: np.ndarray, limit: float, rough: bool = False, baked: bool = False) -> np.ndarray:
    """The labels of connected parts of the viewed mesh where over `limit` of what's shown has changed.

    A big flat part can hide a ruined small one in the total. Parts showing too few pixels to judge are left out.
    """

    count = int(labels.max()) + 1
    wrong, shown = np.zeros(count), np.zeros(count)
    for (hit_a, normal_a, depth_a, _face_a), (hit_b, normal_b, depth_b, face_b) in zip(reference, view):
        shown += np.bincount(labels[face_b[hit_b]], minlength=count)
        if baked:
            changed = bakescore.surface_changed(hit_a, normal_a, depth_a, hit_b, normal_b, depth_b, pixel) & hit_a & hit_b
        else:
            changed = views.surface_changed(hit_a, normal_a, depth_a, hit_b, normal_b, depth_b, pixel, rough)
        wrong += np.bincount(labels[face_b[changed]], minlength=count)
    return np.flatnonzero((shown >= PART_PIXELS) & (wrong > limit * shown))


WAITING = "waiting"  # yielded while only helper Blenders are working: the timer slice ends early, without sleeping
ROWS_NEEDS_HELPERS = "Even Rows runs Blender's QuadriFlow on whole pieces, which takes a long time and can't be stopped, so it runs in a background Blender and yours stays usable. Turn on Background Blender Processes in the CurioMesh preferences to use it"
KEEP_HERE = 0.5  # with helpers, a part of this share of the job's triangles or more is measured in the job itself
PAIR_CHUNK_FACES = 200_000  # Light only: above 0, pairing works per region of this many faces: less memory on a huge mesh, and a few more triangles at the seams
HUGE_FROM = 60_000  # a piece of this many triangles keeps no sizes of its own and is measured in the job itself
SPLIT_FACES = 2 * HUGE_FROM  # with helpers, a piece of more faces is cut into regions of half to all of this
DENSITY_STEPS = 24  # painted weights are rounded to this many levels, so a vertex group takes a few calls, not one per vertex


def _decimate(context, points, polys, kinds, ratio, drop=None, keep=None) -> tuple[np.ndarray, list[tuple[int, ...]], list[str]]:
    """Reduce to `ratio`. `drop` is the painted density per vertex; vertices in the mask `keep` are locked, as are UV borders."""

    free = None if keep is None else np.flatnonzero(~keep)
    if len(set(kinds)) > 1:  # borders stay where they are, so no face ends up across two UV islands
        triangles = sum(len(p) - 2 for p in polys)
        wanted = max(ratio * triangles, 1.0)
        triangle = float(_face_areas(points, polys).sum()) / wanted
        points, polys, kinds, unbound, used = simplify_borders(points, polys, kinds, BORDER_TOLERANCE * math.sqrt(triangle), () if keep is None else np.flatnonzero(keep).tolist())
        drop = None if drop is None else drop[used]
        unbound = np.asarray(unbound, dtype=np.int64)
        free = unbound if keep is None else np.intersect1d(unbound, np.flatnonzero(~keep[used]))
        ratio = min(wanted / max(sum(len(p) - 2 for p in polys), 1), 1.0)
    mesh = bpy.data.meshes.new("CurioMesh_Piece")
    mesh.from_pydata(points.tolist(), [], polys)
    mesh.polygons.foreach_set("material_index", np.asarray([int(k) for k in kinds], dtype=np.int32))
    mesh.update()
    painted = drop is not None and float(drop.max(initial=0.0)) > 0.0
    obj = bpy.data.objects.new("CurioMesh_Piece", mesh)
    context.scene.collection.objects.link(obj)
    try:
        modifier = obj.modifiers.new("Reduce", "DECIMATE")
        modifier.ratio = ratio
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


def _triangle_count(polys) -> int:
    return len(polys.flat) - 2 * len(polys) if isinstance(polys, FlatFaces) else sum(len(p) - 2 for p in polys)


def _same_points(points: np.ndarray, these: np.ndarray) -> np.ndarray:
    """True for each of `points` exactly at one of `these`."""
    seen = set(map(tuple, these.tolist()))
    return np.fromiter((p in seen for p in map(tuple, points.tolist())), dtype=bool, count=len(points))


def _huge_piece(polys, kinds, drop) -> bool:
    return len(polys) >= HUGE_FROM and len(set(kinds)) == 1 and (drop is None or not float(drop.max(initial=0.0)) > 0.0) and _triangle_count(polys) == len(polys)


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


def _point_tree(points: np.ndarray):
    """Blender's tree of points."""

    tree = KDTree(len(points))
    for index, point in enumerate(points.tolist()):
        tree.insert(point, index)
    tree.balance()
    return tree


def _nearest_points(tree, queries: np.ndarray) -> np.ndarray:
    return np.fromiter((tree.find(p)[1] for p in queries.tolist()), dtype=np.int64, count=len(queries))


def _nearest_of_kind(tree, queries: np.ndarray, query_kind: np.ndarray, point_kind: np.ndarray, count: int) -> np.ndarray:
    """For each query, the closest of its `count` nearest points of the same kind, or -1."""

    owner = np.full(len(queries), -1, dtype=np.int64)
    for index, (point, kind) in enumerate(zip(queries.tolist(), query_kind.tolist())):
        for _found, vertex, _away in tree.find_n(point, count):
            if point_kind[vertex] == kind:
                owner[index] = vertex
                break
    return owner


def _face_tree(points: np.ndarray, faces):
    """Blender's tree of faces."""

    return BVHTree.FromPolygons(points.tolist(), [list(f) for f in faces], all_triangles=False, epsilon=0.0)


def _nearest_faces(tree, queries: np.ndarray) -> np.ndarray:
    return np.fromiter((tree.find_nearest(p)[2] for p in queries.tolist()), dtype=np.int64, count=len(queries))


def _face_distances(tree, queries: np.ndarray) -> np.ndarray:
    return np.fromiter((tree.find_nearest(p)[3] for p in queries.tolist()), dtype=np.float64, count=len(queries))


def _picture_sizes(sizes: np.ndarray, most_rays: int = MOST_RAYS, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    """Pixels per side for pieces of these sizes (the whole object is 1.0), coarser for a model with very many parts."""

    view, most = float(VIEW_PIXELS), float(max_pixels)
    while True:
        pixels = np.clip(np.ceil(sizes * view), 8, most).astype(np.int64)
        if 6 * int((pixels * pixels).sum()) <= most_rays or most <= 16:
            return pixels
        view *= 0.85
        most *= 0.85


def smooth_grids(co, faces, coarse_points, coarse_faces, coarse_kinds, labels=None, options: GridOptions | None = None):
    """The reduced mesh with its smooth round areas rebuilt as rows of quads, using BVH lookups on the source."""

    if labels is None:
        domes = smooth_areas(co, faces, options=options)
        labels = (domes, crease_areas(co, faces, domes >= 0, options))
    coarse = MeshData(coarse_points, coarse_faces, coarse_kinds)
    report = GridReport()
    if int(max(area.max(initial=-1) for area in labels)) < 0:
        return coarse, report
    whole = _face_tree(co, faces)

    def nearest(query):
        if not isinstance(whole, BVHTree):
            at, found, _away = whole.nearest(query)
            return found, at
        found = [whole.find_nearest(Vector(p)) for p in query.tolist()]
        return np.asarray([row[2] for row in found], dtype=np.int64), np.asarray([row[0] for row in found], dtype=np.float64).reshape(-1, 3)

    def project_for(area):
        tree = _face_tree(co, [faces[f] for f in np.flatnonzero(labels[1] == area).tolist()])
        if not isinstance(tree, BVHTree):
            return lambda query: tree.nearest(query)[0]
        return lambda query: np.asarray([tree.find_nearest(Vector(p))[0] for p in query.tolist()], dtype=np.float64).reshape(-1, 3)

    extent = float(np.ptp(co, axis=0).max())
    normals = _face_frames(co, faces)[0]
    coarse, report = grid_areas(coarse, labels[0], nearest, options, extent=extent, normals=normals)
    coarse, later = grid_areas(coarse, labels[1], nearest, options, project_for=project_for, extent=extent, normals=normals)
    report.built += later.built
    report.faces_in += later.faces_in
    report.quads_out += later.quads_out
    report.loops += later.loops
    for reason, count in later.rejected.items():
        report.rejected[reason] = report.rejected.get(reason, 0) + count
    return coarse, report


class _SizesOnDisk:
    """The sizes of one part of a cut mesh: read from its file when asked for, the last being the whole part, from the mesh."""

    def __init__(self, job, index: int, file, count: int):
        self.job, self.index, self.file, self.count = job, index, file, count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, at: int):
        at = at + self.count if at < 0 else at
        if at == self.count - 1:
            return self.job._piece_arrays(self.job.pieces[self.index])
        loaded = hardware.load_piece(self.file)
        if loaded is None:
            raise RuntimeError("a saved part went missing")
        return loaded[1][at]


class AutoJob:
    """One Auto Segments run. Call `step` from a timer until `busy` is False; `output` holds the latest result to show."""

    def __init__(self, context, co, faces, materials, detail: float, target_faces: int = 0, cache: dict | None = None, mirror: tuple | None = None, note: str = "", guide_lines: list | None = None, project=None, drop: np.ndarray | None = None, tidy: bool = False, method: str = "AUTO", output: str = "QUADS", methods: dict | None = None, strips=None, outside=None, style: str = "DETAIL", settings=None, checkpoint=None, keep_progress: bool = False, rough: bool = False, drop_hidden: bool = False, baked: bool = False, chords: bool = False, parallel: bool = False, helpers: bool = False, redo_rough: bool = False, tweak: dict | None = None, hard: dict | None = None):
        self.hard = dict(HARD if hard is None else hard)  # Clean Panels and Even Cylinders: more rebuilt sizes per part
        self.redo_rough = bool(redo_rough)  # Redo Rough Spots: regions round the spots still bad after the repairs are made again, finer
        self.redo_numbers: tuple[int, int] | None = None  # (spots redone, bad faces before); the bad faces after are counted at the end
        self.redo_counts = {"regions": 0, "no better": 0, "kept": 0}  # what happened to the regions tried, for the bench
        self.parallel = bool(parallel)  # measure parts in helper processes when the job and the computer suit it (parallel.py)
        self.helpers = bool(helpers)  # may start background Blenders at all (the add-on preference); False: Use Several Cores and Even Rows are off
        self._pool = None
        self._copy_plan: dict | None = None  # with helpers: part -> (the part it copies, how to move onto it) or None, found ahead of time
        self._plan_reps: list = []
        self._plan_kinds: dict[int, str] = {}
        self._todo: list[int] = []  # planned parts for the helpers, in order, not handed out yet
        self._noted: set[int] = set()
        self._done_share = None
        self._ahead = 0  # parts before this one are planned
        self._work = 0  # triangles in the parts to measure
        self.spot_copies = True  # a helper measures only parts already known to be no copy
        self.chords = bool(chords)  # Redraw Messy Quads: messy parts redrawn as grids, collapses priced by the pixels they change
        self.rough = bool(rough)  # judge rough surface as texture (`views`), and let triangles walk away for fewer faces
        self.drop_hidden = bool(drop_hidden)
        self.dropped: list[np.ndarray] = []
        self.baked = bool(baked)  # judge the result as it will look with a normal map baked from the source
        self._bake_tree = None
        self.style = style  # DETAIL, BALANCED or ROWS
        self._tree = None
        self._worker: subprocess.Popen | None = None
        self._refused = 0
        self.collapse = style == "DETAIL" and method == "AUTO" and output == "QUADS" and not guide_lines and not strips and mirror is None and (drop is None or not float(np.max(drop, initial=0.0)) > 0.0)  # take quads away after pairing: only where nothing else asks for the layout
        self.tidy = tidy  # try to tidy the finished quads, kept only if the score rises and the camera allows it
        self.context = context
        self.co = np.asarray(co, dtype=np.float64)
        self.faces = faces if isinstance(faces, FlatFaces) else list(faces)
        self.materials = [str(int(m)) for m in materials]
        self.detail = float(detail)
        self.cache = {} if cache is None else cache  # what was measured on this very mesh before: starting again with another amount skips it
        self.tweak = dict(TWEAK, **(tweak or {}))
        self.seam_report = None  # with a mirror: the Issue Finder's counts along the seam of the final result
        self.mirror = mirror  # (axis, plane offset, size, faces of the whole mesh, side solved): co/faces are one half, the result is mirrored
        self.note = note
        self.guide_lines = guide_lines or []  # strokes laid on the surface; the quads are made to follow them
        self.project = project  # the nearest point on the original surface, for the points a stroke adds to the mesh
        self.guide_edges: tuple = ()
        self.strips = strips  # ([(stroke, loops, tag)], gap or 0, nearest(p) -> (point, normal)): drawn edge loops and deformation lines
        self.bend_groups: list = []  # (tag, loops) of each strip that went in
        self.skipped = 0
        self.outside = outside  # (co, faces, materials) left alone when only a region is redone; the result is put back beside it
        self.seam = (0, 0)
        self.why: list[str] = []
        self.gap = 0.0
        self.drop = None if drop is None or not float(np.max(drop, initial=0.0)) > 0.0 else np.asarray(drop, dtype=np.float64)  # per vertex, from the painted density
        self.cfg = settings or hardware.EFFORTS["NORMAL"]  # how hard to work: slice length, rays, chunk size
        self.max_pixels = self.cfg.max_pixels
        self.checkpoint = checkpoint  # folder where finished parts are saved so a stopped job can resume (None = don't save)
        self.keep_progress = keep_progress
        self.resumed = 0  # parts loaded from the saved progress instead of measured
        self.names: list[str] = []  # where each part is, in words
        self.locked: np.ndarray | None = None  # vertices on a chunk border: never moved, so neighbours join without a gap
        self.chunks = 0  # how many regions a huge piece was cut into
        self.split = 0  # with helpers: how many regions the big pieces were cut into, so the helpers share them
        self.losses: list[float] = []  # the view change measured after each try
        self.first: list = []  # per piece, its smallest size (what the preview shows before sizes are picked)
        self.files: list = []  # per piece, the file that holds its sizes when they are kept on disk, not in memory
        self._tmp: Path | None = None  # where sizes go when there is no checkpoint folder
        self.mem: dict[str, int] = {"start": round(hardware.peak_memory_mb())}  # the most memory used by the end of each stage, MB
        self.target_faces = int(target_faces)  # 0 = keep the detail asked for; above 0 = land on this many faces
        self.method = method  # for every part without its own in `methods`
        self.output_kind = output  # TRIS, QUADS or MIXED
        self.methods: dict[int, str] = dict(methods or {})  # piece -> method
        self.notes: list[str] = []
        self.face_piece = np.zeros(0, dtype=np.int64)
        self.flag = np.zeros(0, dtype=bool)
        self.bad = np.zeros(0, dtype=bool)  # faces the defect check flags in the result, for the red overlay
        self.issues = None  # (spots, report) of the Issue Finder, once the run is done
        self.found = None  # checks.Defects of the final result
        self._defects = None  # (points, faces, checks.Defects) of the last defect check, so the end of the run does not repeat it
        self._kd = None
        self._unmeasured: dict = {}  # piece -> its arrays, as `_piece_arrays` gives them, until the piece is measured
        self._kd_owner = None
        self._src_check = None
        self._src_middles = None  # each source face's middle, to find the source round a redone region
        self._known = None  # what the source crosses itself
        self._src_pieces = None  # pieces the source was checked in
        self.repair_fills = True  # off for the size ladder's rungs: one fill can hold Blender for seconds
        self._piece_trees: dict[int, BVHTree] = {}
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
        self._paired = None
        self._labels = None
        self._shell = None
        self._samples = None
        self.pieces: list[np.ndarray] = []
        self.picture_sizes: list[int] = []
        self.sizes: list[list[tuple]] = []  # per piece, smallest first: (points, polys, materials)
        self.levels: list[list[Level]] = []
        self.field_spent = 0.0
        self.ready: list[list[bool]] = []  # per piece and size: already all quads, so it skips pairing
        self.seen: list[float] = []
        self.twins: dict[int, tuple[int, tuple]] = {}  # a piece that's a copy: the piece it copies, and how to move onto it
        self.reps: list[tuple[int, np.ndarray, list]] = []  # pieces worked out so far, to spot copies of them
        self.instances: dict | None = None  # the result as one mesh per repeated piece with where each goes, and one for the rest
        self.picked: list[int] = []
        self.timings: dict[str, float] = {}  # seconds spent per kind of work, for the report
        self.looks = 0  # sizes of pieces the camera looked at
        self.part_cases: dict[int, tuple] = {}  # piece -> (case, sizes skipped, sizes looked at), for `explain_piece`
        self.table = cases.load_table() if USE_CASES else None
        self._search = self._run_rows() if style == "ROWS" else self._run()

    def seconds_left(self) -> float | None:
        """A rough guess from the pace so far; None until there is enough to go on."""
        if not self.busy or self.fraction < 0.05:
            return None
        spent = time.perf_counter() - self.started
        return spent * (1.0 - self.fraction) / self.fraction

    def stop(self) -> None:
        self._stopped = True
        self.busy = False
        self._kill_worker()
        self._close_pool()
        self._search.close()  # runs what cleans up after it, such as the Even Rows folder
        self._drop_tmp()

    def _drop_tmp(self) -> None:
        if self._tmp is not None:
            atexit.unregister(self._drop_tmp)
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def _mark(self, stage: str) -> None:
        self.mem[stage] = round(hardware.peak_memory_mb())

    def _finish_piece(self, key, sizes: list, levels: list, ready: list, seen: float, file, fresh: bool) -> None:
        """Keep a measured piece, whose sizes end with the whole piece. A piece of a cut mesh goes to disk and only its numbers stay in memory."""
        self.levels.append(levels)
        self.ready.append(ready)
        self.seen.append(seen)
        self.first.append(sizes[0])
        pairs = [(lv.faces, lv.loss) for lv in levels]
        if self.chunks:
            if file is None:
                if self._tmp is None:
                    self._tmp = hardware.job_folder("curiomesh_")
                    atexit.register(self._drop_tmp)  # Blender quit in the middle
                file = hardware.piece_file(self._tmp, repr(key[2]).encode() + key[1])
            if fresh:
                hardware.save_piece(file, self.detail, sizes[:-1], pairs[:-1], seen)
            self.sizes.append(_SizesOnDisk(self, len(self.sizes), file, len(sizes)))
            self.files.append(file)
            return
        self.sizes.append(sizes)
        self.files.append(None)
        self.cache[key] = (self.detail, sizes, levels, seen, ready)
        if file is not None and fresh:
            hardware.save_piece(file, self.detail, sizes[:-1], pairs[:-1], seen)

    def step(self, seconds: float | None = None) -> None:
        """Do the next slice of work (called from a timer on the main thread)."""
        if not self.busy:
            return
        seconds = self.cfg.slice_s if seconds is None else seconds
        ends = time.perf_counter() + seconds
        try:
            while time.perf_counter() < ends:
                if next(self._search) is WAITING:
                    break
        except StopIteration:
            self.busy = False
        except Exception as exc:  # a timer has nobody above it to report to
            self.warnings = [f"Auto Segments failed: {exc}"]
            self.busy = False

    def wait(self, timeout: float | None = None) -> bool:
        """Run to the end in one call, for a script in background mode (an exported run_job.py). The panel never calls it."""
        ends = None if timeout is None else time.perf_counter() + timeout
        while self.busy and (ends is None or time.perf_counter() < ends):
            began = time.perf_counter()
            self.step(1.0)
            if self.busy and time.perf_counter() - began < 0.05:
                time.sleep(0.05)  # only helper Blenders are working: don't spin a core (never in a timer, see step)
        return not self.busy

    def _reduced(self, points, polys, kinds, ratio, drop=None, keep=None, one_off: bool = False):
        """Generator: this piece reduced to `ratio` by `_decimate`, which has to run on the main thread."""
        started = time.perf_counter()
        with tasks.busy_cursor():
            out = _decimate(self.context, points, polys, kinds, ratio, drop, keep)
        yield from ()
        self.timings["decimate"] = self.timings.get("decimate", 0.0) + time.perf_counter() - started
        return out

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

    def _cut(self, ids: np.ndarray) -> bool:
        """A region of a cut piece, whose border must not move. When a piece was cut for memory, every part counts as one."""
        return bool(self.chunks) or (self.locked is not None and bool(self._piece_lock(ids).any()))

    def _piece_vertex(self, ids: np.ndarray, per_vertex):
        if per_vertex is None:
            return None
        return per_vertex[np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))]

    def _piece_arrays(self, ids: np.ndarray):
        chosen = ids.tolist()
        sizes = np.fromiter((len(self.faces[f]) for f in chosen), dtype=np.int64, count=len(chosen)) if not isinstance(self.faces, FlatFaces) else np.diff(self.faces.indptr)[ids]
        used, flat = np.unique(np.fromiter(chain.from_iterable(self.faces[f] for f in chosen), dtype=np.int64, count=int(sizes.sum())), return_inverse=True)
        ends = np.zeros(len(chosen) + 1, dtype=np.int64)
        np.cumsum(sizes, out=ends[1:])
        polys = FlatFaces(ends, flat.reshape(-1))
        return self.co[used], polys, [self.materials[f] for f in ids.tolist()]

    def _copy_of(self, index: int, points: np.ndarray, polys, kinds) -> bool:
        """If this piece is a copy of one already worked out, take that one's sizes, moved over. True if it was."""
        if self._copy_plan is not None:
            found = self._copy_plan.pop(index, None)
        else:
            found = self._find_copy(self.reps, points, polys, kinds, lambda rep: self.sizes[rep][-1][2][0])
        if found is not None:
            rep, move = found
            self.twins[index] = (rep, move)
            sizes = [(special.moved(p, move), special.wound(f, move), k) for p, f, k in self.sizes[rep][:-1]]
            self.sizes.append(sizes + [(points, polys, kinds)])
            self.files.append(None)
            self.first.append(sizes[0] if sizes else (points, polys, kinds))
            self.levels.append(list(self.levels[rep]))
            self.ready.append(list(self.ready[rep]))
            self.seen.append(self.seen[rep])
            if rep in self.part_cases:
                self.part_cases[index] = self.part_cases[rep]
            return True
        return False

    @staticmethod
    def _find_copy(reps, points, polys, kinds, kind_of):
        """(the part in `reps` this one copies, how to move onto it), or None."""
        if len(set(kinds)) > 1:
            return None  # faces of copies don't correspond one to one, so materials could land on the wrong ones
        for rep, rep_points, rep_polys in reps:
            if len(rep_polys) != len(polys) or len(rep_points) != len(points) or kind_of(rep) != kinds[0]:
                continue
            move = special.match_copy(rep_points, points)
            if move is not None:
                if parts.facing(special.moved(rep_points, move), special.wound(rep_polys, move), points, polys) < 0:
                    move = (*move[:3], not move[3])  # a symmetric part also matches mirrored: its own faces say which way it faces
                return rep, move
        return None

    def _piece_key(self, ids: np.ndarray) -> tuple:
        # Reduce To needs the small sizes a kept-detail search skips; sizes judged another way are no use
        return ("piece", ids.tobytes(), (None if self.drop is None else hash(self.drop.tobytes()), bool(self.guide_lines or self.strips), self.rough, self.baked, self.target_faces > 0))

    def _plan_next(self) -> bool:
        """Look at the next part not yet planned: is it a copy, and must it be measured? True if a helper should measure it."""
        index, ids = self._ahead, self.pieces[self._ahead]
        self._ahead += 1
        points, polys, kinds = self._piece_arrays(ids)
        if _triangle_count(polys) <= KEEP_TRIANGLES:
            return False
        if not self._cut(ids) and not self.guide_lines and not self.strips and self.drop is None:
            found = self._find_copy(self._plan_reps, points, polys, kinds, lambda rep: self._plan_kinds[rep])
            self._copy_plan[index] = found
            if found is not None:
                return False
            self._plan_reps.append((index, points, polys))
            self._plan_kinds[index] = kinds[0]
        if _triangle_count(polys) >= KEEP_HERE * self._work:
            return False  # most of the job: measured here, while the helpers make its sizes and measure the rest
        key = self._piece_key(ids)
        if self.checkpoint is not None and hardware.piece_file(self.checkpoint, repr(key[2]).encode() + key[1]).exists():
            return False
        known = self.cache.get(key)
        return known is None or known[0] > self.detail

    def _measure_piece(self, ids: np.ndarray, pixels: int):
        """Generator: this piece at each size, smallest first, with how much each changes it."""
        points, polys, kinds = yield from self._slice(lambda: self._piece_arrays(ids))
        whole = (points, polys, kinds)
        triangles = _triangle_count(polys)
        cut = self._cut(ids)
        lock = self._piece_lock(ids)
        if triangles <= KEEP_TRIANGLES:
            self.sizes.append([whole])
            self.files.append(None)
            self.first.append(whole)
            self.levels.append([Level(len(polys), 0.0)])
            self.ready.append([False])
            self.seen.append(0.0)
            return
        if self.spot_copies and not cut and not self.guide_lines and not self.strips and self.drop is None:
            found = yield from self._slice(lambda: self._copy_of(len(self.sizes), points, polys, kinds))
            if found:
                return
            if self._copy_plan is None:
                self.reps.append((len(self.sizes), points, polys))
        key = self._piece_key(ids)
        file = None
        if self.checkpoint is not None:
            file = hardware.piece_file(self.checkpoint, repr(key[2]).encode() + key[1])
        index = len(self.sizes)
        if self._pool is not None and (index in self._todo or self._pool.waiting_for(index)):
            result = yield from self._from_pool(index, index)
            if result is not None:  # measured by a helper, exactly as it would have been here
                parallel.apply_result(self, index, result)
                self._finish_piece(key, result["sizes"], result["levels"], result["ready"], result["seen"], file, not result.get("saved"))
                return
        if file is not None:
            saved = hardware.load_piece(file) if file.exists() else None
            if saved is not None and saved[0] <= self.detail:  # a stopped run measured this part already
                self.resumed += 1
                ready = [all(len(p) == 4 for p in polys_at) for _co, polys_at, _kinds in saved[1]] + [False]
                self._finish_piece(key, saved[1] + [whole], [Level(f, w) for f, w in saved[2]] + [Level(len(polys), 0.0)], ready, saved[3], file, False)
                return
        known = self.cache.get(key)
        if self._pool is not None and (known is None or known[0] > self.detail) and not _huge_piece(polys, kinds, self._piece_drop(ids)):
            # Measured here (the biggest part): the helpers make its first sizes meanwhile, each a Decimate of seconds.
            self._todo[:0] = [("size", index, share) for share in (BAKED_LEVELS if self.baked else LEVELS) if share * triangles >= 12.0]
        if known is not None and known[0] <= self.detail:  # measured with the same or a stricter amount (more sizes tried): nothing to redo
            self.sizes.append(known[1])
            self.files.append(None)
            self.first.append(known[1][0])
            self.levels.append(known[2])
            self.ready.append(known[4])
            self.seen.append(known[3])
            return
        low, high = points.min(axis=0), points.max(axis=0)
        center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
        pixel = 2.0 * half / pixels
        reference = yield from self._reference(center, half, pixels, points, polys)
        turned_sizes, turned_levels = [], []
        seen = 0.0
        enough = GOOD_ENOUGH * self.detail / 100.0
        if not cut and self.output_kind != "TRIS" and self._method(len(self.sizes)) == "AUTO" and not self.guide_lines and not self.strips and self.drop is None and len(set(kinds)) == 1:  # a rebuilt part would ignore strokes, paint and UV islands
            kind = max(set(kinds), key=kinds.count)
            tries = chain(parts.rebuilds(points, polys, kind), hardsurf.rebuilds(points, polys, kind, **self.hard))
            while True:
                out = yield from self._slice(lambda: next(tries, False))
                if out is False:
                    break
                yield
                if out is None:
                    continue
                turned = yield from self._slice(lambda: parts.orient_like(out[0], out[1], points, polys))
                rebuilt = (out[0], turned, out[2])
                view = yield from self._timed("pictures", self._pictures(rebuilt[0], rebuilt[1], center, half, pixels))
                wrong, shown = self._counts(reference, view, pixel)
                seen = max(seen, shown * pixel * pixel)
                turned_sizes.append(rebuilt)
                weight = hardsurf.WEIGHTS.get(out[3], QUAD_WEIGHT) if len(out) > 3 else QUAD_WEIGHT  # clean panels and even cylinders count as fewer faces
                turned_levels.append(Level(len(rebuilt[1]) * weight, REBUILD_MARGIN * wrong * pixel * pixel))
                if wrong <= enough * shown:
                    break
            yield  # the last try above can end with a long fit: the special shapes start in a slice of their own
            hopeless = HOPELESS * PIECE_LIMIT * self.detail / 100.0
            for family in special.rebuilds(points, polys, max(set(kinds), key=kinds.count)):  # wires, stacks of fins, cloth
                for out in family:  # most faces first: once one is far off, the smaller ones won't do either
                    yield
                    if out is None:
                        continue
                    rebuilt = (out[0], special.orient_like(out[0], out[1], points, polys), out[2])
                    view = yield from self._timed("pictures", self._pictures(rebuilt[0], rebuilt[1], center, half, pixels))
                    wrong, shown = self._counts(reference, view, pixel)
                    if REBUILD_MARGIN * wrong > hopeless * shown:
                        break
                    seen = max(seen, shown * pixel * pixel)
                    turned_sizes.append(rebuilt)
                    turned_levels.append(Level(len(rebuilt[1]) * QUAD_WEIGHT, REBUILD_MARGIN * wrong * pixel * pixel))
        method_here = self._method(len(self.sizes))
        plain = self.output_kind != "TRIS" and not self.guide_lines and not self.strips and self.mirror is None and self.drop is None and not cut and len(set(kinds)) == 1
        if method_here == "FIELD" and plain:
            want = max(min(FIELD_START * triangles, FIELD_FIRST), 24.0)
            asked = self.detail / 100.0
            for _try in range(FIELD_TRIES):  # change falls about as faces ** -0.63, so each look says where to go next
                want = min(want, float(MOST_TARGET))
                self.stage = f"Laying rows of quads on piece {len(self.sizes) + 1}: about {int(want):,} quads"
                out = yield from self._field_size(points, polys, kinds, want, center, half, pixels, reference, pixel, 1.0)
                if self._stopped:
                    return
                if out is None:
                    want *= 1.6  # refused: limbs thinner than a cell are the usual reason, and more cells fit them
                    continue
                seen = max(seen, out[2])
                turned_sizes.append(out[0])
                turned_levels.append(out[1])
                if FIELD_LANDED * asked <= out[3] <= asked or want >= MOST_TARGET:
                    break
                want = len(out[0][1]) * min(max((max(out[3], 1e-4) / (FIELD_AIM * asked)) ** (1.0 / 0.63), 0.25), 4.0)
            if turned_sizes:
                # Rows were asked for: the whole source is never picked in their place, even when they miss the amount.
                sizes = [*turned_sizes, whole]
                levels = [*turned_levels, Level(len(polys), 1e30)]
                ready = [True] * len(turned_sizes) + [False]
                order = sorted(range(len(sizes)), key=lambda k: levels[k].faces)
                sizes, levels, ready = [sizes[k] for k in order], [levels[k] for k in order], [ready[k] for k in order]
                if out is None or out[3] > asked:
                    self.notes.append(f"Field Rows kept its rows on a part that changes more than the Detail asked for ({100.0 * min(level.loss for level in turned_levels) / max(seen, 1e-30):.1f}% of it)")
                self._finish_piece(key, sizes, levels, ready, seen, file, True)
                return
            self.notes.append("Rows of quads did not work on a part: Auto was used there")
        if self.baked:
            self._bake_tree = BVHTree.FromPolygons(points.tolist(), [list(p) for p in polys], all_triangles=False, epsilon=0.0)
        shares = [share for share in (BAKED_LEVELS if self.baked else LEVELS) if share * triangles >= 12.0]
        floor = PIECE_LIMIT * self.detail / 100.0
        first, case = 0, None
        if self.table is not None:
            case = yield from self._slice(lambda: cases.classify(cases.features(points, polys, float(np.ptp(self.co, axis=0).max()) or 1.0)))
            if not self.target_faces:  # Reduce To may want any small size to land on its count
                first = cases.first_size(case, shares, floor, self.table)
        area = pixel * pixel
        looked = {}

        def look(k: int):
            """Generator: this size measured. Returns the share of its view that changed."""
            yield
            reduced = yield from self._sized(index, shares[k], points, polys, kinds, self._piece_drop(ids), lock)
            view = yield from self._timed("pictures", self._pictures(reduced[0], reduced[1], center, half, pixels))
            wrong, shown = self._counts(reference, view, pixel)
            self.looks += 1
            looked[k] = (reduced, Level(len(reduced[1]), wrong * area), wrong_points(reference, view, center, half, pixels, pixel), shown * area)
            return wrong / max(shown, 1)

        at = first
        start_change = change = yield from look(at)
        while at > 0 and change <= floor / 2.0:  # room to spare: the table was too careful, so the camera sends the search back down
            at -= 1
            change = yield from look(at)
        at, change = first, start_change
        while change > enough and at + 1 < len(shares):
            at += 1
            change = yield from look(at)
        order = sorted(looked)
        sizes, levels, spots = [looked[k][0] for k in order], [looked[k][1] for k in order], [looked[k][2] for k in order]
        seen = max([seen] + [looked[k][3] for k in order])
        if case is not None:
            self.part_cases[len(self.sizes)] = (case, order[0], len(order))
        if triangles >= FINER_TRIANGLES and seen > 0.0 and not self.chunks:
            painted = self.drop is not None  # Detail Markers: only the halfway sizes, which keep the weights; the held and stepped ones would lose them
            asked = self.detail / 100.0
            uppers = {next((k for k, level in enumerate(levels) if level.loss <= limit * seen), 0) for limit in (asked, PIECE_LIMIT * asked)} - {0}
            for upper in sorted(uppers):
                # A size halfway across the jump, so a piece doesn't get twice the faces to clear a limit. With markers,
                # which skip the held and stepped sizes below, three across it and one across the next jump up.
                between = [(upper - 1, upper, t) for t in ((0.25, 0.5, 0.75) if painted else (0.5,))]
                if painted and upper + 1 < len(order):
                    between.append((upper, upper + 1, 0.5))
                for a, b, t in between:
                    yield
                    faces = levels[a].faces ** (1.0 - t) * levels[b].faces ** t
                    reduced = yield from self._reduced(points, polys, kinds, faces / len(polys), self._piece_drop(ids), keep=lock)
                    view = yield from self._timed("pictures", self._pictures(reduced[0], reduced[1], center, half, pixels))
                    sizes.append(reduced)
                    levels.append(Level(len(reduced[1]), self._counts(reference, view, pixel)[0] * area))
                    spots.append(wrong_points(reference, view, center, half, pixels, pixel))
                # Density where the camera saw it go wrong: hold those spots while the rest goes further, then let go.
                for k in () if painted else (upper - 1, upper, len(levels) - 1):
                    yield
                    keep = yield from self._slice(lambda: hottest(vertex_heat(points, spots[k], 3.0 * pixel), LOCKED_SHARE * levels[k].faces / 2.0))
                    held = yield from self._reduced(points, polys, kinds, HELD_SIZE * levels[k].faces / triangles, keep=keep if lock is None else keep | lock, one_off=True)
                    border = None if lock is None else (yield from self._slice(lambda: _same_points(held[0], points[lock])))  # a region's border never moved
                    reduced = yield from self._reduced(held[0], held[1], held[2], levels[k].faces / max(len(held[1]), 1), keep=border, one_off=True)
                    view = yield from self._timed("pictures", self._pictures(reduced[0], reduced[1], center, half, pixels))
                    sizes.append(reduced)
                    levels.append(Level(len(reduced[1]), self._counts(reference, view, pixel)[0] * area))
                    spots.append(spots[k])
            # Reduced a little at a time, Decimate forgets the errors it has built up and leaves a closer surface: 15-20% fewer
            # triangles on skin and rock, about even on plates. So these go in beside the other sizes, and the camera chooses.
            reduced = sizes[len(order) - 1] if triangles < HUGE_FROM and not painted else None  # a huge piece keeps no sizes
            while reduced is not None and len(reduced[1]) * STEP >= 12.0:
                yield
                before = len(reduced[1])
                reduced = _decimate(self.context, *reduced, STEP)
                if len(reduced[1]) >= before:  # Decimate can't take this piece further: stepping again would never end (30 GB on a ship)
                    break
                view = yield from self._timed("pictures", self._pictures(reduced[0], reduced[1], center, half, pixels))
                sizes.append(reduced)
                levels.append(Level(len(reduced[1]), self._counts(reference, view, pixel)[0] * area))
                if levels[-1].loss > floor * seen:
                    break
            kept = best_levels(levels)
            sizes, levels = [sizes[k] for k in kept], [levels[k] for k in kept]
        # Rows are offered to Auto only on flat or simple parts: the camera mostly turns them down elsewhere, and each try costs seconds.
        if method_here == "AUTO" and plain and case in FIELD_CASES and triangles >= FIELD_SMALLEST and self.field_spent < FIELD_SECONDS and seen > 0.0:
            # At about the size the camera settled on: the camera says if rows hold the shape.
            good = next((level.faces for level in levels if level.loss <= self.detail / 100.0 * seen), levels[-1].faces if levels else 0)
            if FIELD_SMALLEST / 40.0 <= good <= FIELD_MOST:
                out = yield from self._field_size(points, polys, kinds, good, center, half, pixels, reference, pixel, REBUILD_MARGIN)
                if out is not None:
                    turned_sizes.append(out[0])
                    turned_levels.append(out[1])
        if self.hard.get("cylinders") and not cut and sizes:  # Even Cylinders: a visibly round part never becomes a coarse prism
            least = yield from self._slice(lambda: hardsurf.round_floor(points, polys, float(np.linalg.norm(np.ptp(self.co, axis=0)))))
            if least and any(level.faces >= least for level in levels):
                kept = [k for k, level in enumerate(levels) if level.faces >= least]
                sizes, levels = [sizes[k] for k in kept], [levels[k] for k in kept]
        ready = [all(len(f) == 4 for f in size[1]) for size in turned_sizes] + [False] * (len(sizes) + 1)
        sizes = turned_sizes + sizes + [whole]
        levels = turned_levels + levels + [Level(len(polys), 0.0)]
        order = sorted(range(len(sizes)), key=lambda k: levels[k].faces)
        sizes, levels, ready = [sizes[k] for k in order], [levels[k] for k in order], [ready[k] for k in order]
        self._finish_piece(key, sizes, levels, ready, seen, file, True)

    def _start_pool(self):
        """Generator: helper processes for measuring the parts, when this job and this computer suit them (see parallel.py)."""
        if not self.parallel or not self.helpers or self.guide_lines or self.strips or len(self.pieces) < 2:
            return

        def size_up():
            sizes = np.diff(self.faces.indptr) if isinstance(self.faces, FlatFaces) else np.fromiter(map(len, self.faces), dtype=np.int64, count=len(self.faces))
            triangles = [int(sizes[ids].sum()) - 2 * len(ids) for ids in self.pieces]
            total, free = hardware._memory()
            machine = {"cores": os.cpu_count() or 1, "ram_gb": total, "free_gb": free}
            work = sum(t for t in triangles if t > KEEP_TRIANGLES)
            measured = sum(t > KEEP_TRIANGLES for t in triangles)
            return hardware.worker_count(machine, self.cfg.name, len(self.faces), max(len(ids) for ids in self.pieces), work, measured, hardware.busy_cores()), machine, work

        count, _machine, self._work = yield from self._slice(size_up)
        if not count:
            return
        command, env = parallel.helper_command("parallel.worker_main", [])
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")  # the child's own copy: each helper's NumPy keeps to one core
        setup = yield from self._slice(lambda: parallel.setup_of(self))
        self._pool = parallel.Pool(command, count, setup, env, hardware.job_folder("curiomesh_parts_"))
        self._copy_plan = {}
        self.notes.append(f"Parts were measured {count} at a time, in helper processes")

    def _close_pool(self) -> None:
        if self._pool is not None:
            if self._pool.peak_mb:
                self.mem["helpers"] = round(sum(self._pool.peak_mb.values()))
            self._pool.close()
            self._pool = None
        self._todo = []

    def _pump(self, upto: int):
        """Generator: plan the parts up to `upto` and enough more to keep the helpers busy, hand tasks out, collect results."""
        while self._ahead < len(self.pieces) and (self._ahead <= upto or self._pool is not None and len(self._todo) < 2 * self._pool.alive()):
            index = self._ahead
            wanted = yield from self._slice(self._plan_next)
            if wanted and self._pool is not None:
                self._todo.append(index)
        pool = self._pool
        if pool is None:
            return
        self._feed()
        if pool.failed - self._noted:
            self._noted |= pool.failed
            self.notes.append("A helper process could not finish a task, so it was done here")
        if not pool.alive():
            self.notes.append("The helper processes stopped: the rest of the parts were measured here")
            self._close_pool()
            return
        if self.checkpoint is not None:  # saved as soon as they are back, so a stopped job keeps them
            for part, result in list(pool.ready.items()):  # helpers keep adding while a part is saved
                if isinstance(part, int) and not result.get("saved"):
                    key = self._piece_key(self.pieces[part])
                    file = hardware.piece_file(self.checkpoint, repr(key[2]).encode() + key[1])
                    yield from self._slice(lambda: hardware.save_piece(file, self.detail, result["sizes"][:-1], [(lv.faces, lv.loss) for lv in result["levels"]][:-1], result["seen"]))
                    result["saved"] = True

    def _feed(self) -> None:
        """Collect what the helpers finished and hand idle ones the next tasks. Quick: also called just before each long step."""
        pool = self._pool
        if pool is None:
            return
        pool.poll()
        done = len(self.sizes)
        for task in [t for t in pool.ready if not isinstance(t, int) and t[1] < done]:
            del pool.ready[task]  # a size of a part that is finished: not needed
        while self._todo and pool.idle():
            task = self._todo.pop(0)
            if isinstance(task, int):
                pool.give(task, f"measure {task} {self.field_spent!r}")
            elif task[1] >= done:
                pool.give(task, f"size {task[1]} {task[2]!r}")

    def _from_pool(self, task, index: int):
        """Generator: the result of a helper's task for part `index`, keeping the others busy meanwhile. None if no helper did it."""
        stage = self.stage
        while True:
            yield from self._pump(index)
            pool = self._pool
            if pool is None or task in pool.failed or not (task in self._todo or pool.waiting_for(task)):
                return None
            result = pool.take(task)
            if result is not None:
                return result
            ahead = sum(float(self._done_share[k + 1] - self._done_share[k]) for k in pool.ready if isinstance(k, int))
            self.fraction = MEASURE_SHARE * min(float(self._done_share[index]) + ahead, 1.0)
            self.stage = f"{stage} ({len(pool.busy)} at once)"
            yield WAITING  # nothing to do here but wait: the slice ends and the core is left to the helpers

    def _sized(self, index: int, ratio: float, points, polys, kinds, drop=None, keep=None):
        """Generator: `_reduced` at `ratio`, from a helper when one was asked to make it, else here."""
        if self._pool is not None:
            result = yield from self._from_pool(("size", index, ratio), index)
            if result is not None:
                return result["size"]
        return (yield from self._reduced(points, polys, kinds, ratio, drop, keep))

    def _field_size(self, points, polys, kinds, want, center, half, pixels, reference, pixel, margin):
        """Generator: rows of quads on this part at about `want` quads, looked at. (build, level, area seen, change) or None."""
        started = time.perf_counter()
        strict = self.output_kind == "QUADS"

        def work():
            try:
                made = field_quads(points, polys, want, cancelled=lambda: self._stopped)
                return made if not strict or is_clean(made[0], made[1]) else None
            except ValueError as exc:
                if "Stopped" not in str(exc) and "too small" not in str(exc):
                    self.notes.append(str(exc))
                return None

        made = yield from self._slice(work)
        self.field_spent += time.perf_counter() - started
        self.timings["field"] = self.timings.get("field", 0.0) + time.perf_counter() - started
        if made is None or self._stopped:
            return None
        kind = max(set(kinds), key=kinds.count)
        build = (made[0], parts.orient_like(made[0], made[1], points, polys), [kind] * len(made[1]))
        view = yield from self._timed("pictures", self._pictures(build[0], build[1], center, half, pixels))
        wrong, shown = self._counts(reference, view, pixel)
        area = pixel * pixel
        return build, Level(len(build[1]) * QUAD_WEIGHT, margin * wrong * area), shown * area, wrong / max(shown, 1)

    def _reference(self, center, half, pixels, points, polys, tree=None):
        """Generator: the six pictures of the original at this view. A lone piece is the whole object, so they are made once."""
        key = ("view", center.tobytes(), half, pixels, len(polys), hash(points.tobytes()))
        if key not in self.cache:
            self.cache[key] = yield from self._timed("pictures", self._pictures(points, polys, center, half, pixels))
        return self.cache[key]

    def _combined(self, rebuilt: bool | None = None, join: bool = False, only_done: bool = False, firsts: bool = False) -> tuple[np.ndarray, list[tuple[int, ...]], list[str], np.ndarray]:
        """Every piece at its picked size, and any not measured yet as it is. `rebuilt` True or False keeps only those."""
        all_points, all_faces, all_kinds, owner, seams = [], [], [], [], []
        offset = 0
        join = join and self.locked is not None
        for index, ids in enumerate(self.pieces):
            if only_done and index >= len(self.sizes):
                continue
            if firsts and index in self.twins:
                continue
            if index in self.tuned:
                if rebuilt:
                    continue
                points, polys, kinds = self.tuned[index]
            elif index < len(self.sizes):
                at = self.picked[index] if index < len(self.picked) else 0
                if rebuilt is not None and self.ready[index][at] != rebuilt:
                    continue
                points, polys, kinds = self.sizes[index][at] if index < len(self.picked) else self.first[index]
            elif rebuilt:
                continue
            else:
                if index not in self._unmeasured:  # the preview asks for these again and again while the job runs
                    self._unmeasured[index] = self._piece_arrays(ids)
                points, polys, kinds = self._unmeasured[index]
            if index < len(self.sizes):
                self._unmeasured.pop(index, None)
            all_points.append(points)
            if join:
                seams.append(np.full(len(points), self._cut(ids)))
            all_faces += [tuple(v + offset for v in p) for p in polys]
            all_kinds += kinds
            owner += [index] * len(polys)
            offset += len(points)
        if not all_points:
            return np.zeros((0, 3)), [], [], np.zeros(0, dtype=np.int64)
        points = np.vstack(all_points)
        if join:  # the chunk borders were never moved, so a cut piece is one surface again
            points, all_faces = hardware.weld(points, all_faces, np.concatenate(seams))
        return points, all_faces, all_kinds, np.asarray(owner, dtype=np.int64)

    def _show(self, final=None) -> None:
        if final is None:
            co, faces, kinds, parts = self._combined(only_done=self.cfg.only_done or bool(self.chunks))
            if not faces:
                return
        else:
            co, faces, kinds, parts = final
        if self.mirror is not None:
            axis, offset, size = self.mirror[:3]
            co, faces, kinds, _snapped, kept = symmetry.mirror_result(co, faces, kinds, axis, offset, size, self.mirror[4])
            if final is not None:
                self.seam_report = symmetry.seam_check(co, faces, axis, offset, size)
                self.notes.append(symmetry.seam_words(self.seam_report))
            if final is None:
                parts = np.asarray(parts)[kept]
                parts = np.concatenate((parts, parts + int(parts.max(initial=-1)) + 1))
            else:
                fa, fb = patch.face_adjacency(faces)
                parts = np.unique(patch._labels(len(faces), fa, fb), return_inverse=True)[1]
        new = None
        if self.outside is not None:
            co, faces, kinds, new, self.seam = loops.join_outside(co, faces, kinds, self.outside)
            fa, fb = patch.face_adjacency(faces)
            parts = np.unique(patch._labels(len(faces), fa, fb), return_inverse=True)[1]
        count = len(faces)
        new = np.ones(count, bool) if new is None else new
        codes = {k: int(k) if str(k).isdigit() else 0 for k in set(kinds)}
        self.output = {"instances": self.instances if final is not None else None, "co": co, "faces": faces, "new": new, "locked": np.zeros(count, bool), "mat": np.fromiter(map(codes.__getitem__, kinds), dtype=np.int32, count=len(kinds)), "parts": parts}
        if final is not None and self.mirror is None:
            self.output["piece"] = self.face_piece  # the source piece each face came from
            self.output["flag"] = self.flag  # faces still broken after the repair tries
            self.output["bad"] = self.bad  # faces the defect check flags (red in the segment view)
        if final is not None:  # before output_id moves on: the panel copies the output as soon as it does
            self._find_issues(faces)
        self.output_id += 1
        self.shown_at = time.perf_counter()

    def _source_samples(self):
        """Each source triangle as a point: (positions, normals, areas, loose parts, the source's tree). Made once."""
        if self._samples is None:
            flat, starts, sizes = corner_table(self.faces)
            first = np.repeat(starts, sizes - 2)
            step = np.arange(len(first)) - np.repeat(np.cumsum(sizes - 2) - (sizes - 2), sizes - 2)  # 0, 1.. within each face
            parts = np.repeat(self._shell, sizes - 2)
            a, b, c = self.co[flat[first]], self.co[flat[first + step + 1]], self.co[flat[first + step + 2]]
            cross = np.cross(b - a, c - a)
            twice = np.linalg.norm(cross, axis=1)
            real = twice > 0.0
            self._samples = (((a + b + c) / 3.0)[real], cross[real] / twice[real][:, None], twice[real] / 2.0, parts[real], _face_tree(self.co, self.faces))
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
        tree = _point_tree(points)
        source_parts = int(sample_part.max()) + 1
        if source_parts == 1:
            vertex_part, source_of = vertex_parts(points, mesh.faces, sample_part, None)
            owner = _nearest_points(tree, position)
        else:
            # A result part came from one source part, and samples of another part, however near, say nothing about it.
            sample_tree = _point_tree(position)
            vertex_part, source_of = vertex_parts(points, mesh.faces, sample_part, lambda at: _nearest_points(sample_tree, at))
            owner = _nearest_of_kind(tree, position, sample_part, source_of[vertex_part], OTHER_VERTICES)
        count = len(source_of)
        moved = fit_vertices(points, quads, position, normal, area, owner)

        def gaps(where: np.ndarray) -> np.ndarray:
            """Per result part, the mean distance from its source to it plus from it to its source."""
            to_result = _face_distances(_face_tree(where, mesh.faces), position)
            to_source = _face_distances(source_tree, where)
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
        kind = np.asarray([int(k) if str(k).isdigit() else 0 for k in kinds], dtype=np.int64)[owner]
        parts = [(self.guide_lines, spacing)] if self.guide_lines else []
        for lines, step in parts:
            samples = np.vstack([guides.resample(line, step / 2.0) for line in lines])
            co, tris, kind = guides.refine(co, tris, kind, samples, step, self.project)
        keys: set = set()
        for lines, step in parts:
            chains, found = guides.chain(co, tris, lines, step)
            if chains:
                co = guides.pull(co, tris, chains, lines, step)
            keys |= found
        if self.strips:  # drawn edge loops and deformation lines go in as built strips of quads
            items, gap, nearest = self.strips
            self.gap = gap if gap > 0 else 0.7 * spacing  # the pairing then splits every face once more, so the result is about as fine as the rest
            fine = min(loops.room(line, count, self.gap) for line, count, _tag in items)  # a ring round a small thing needs finer triangles
            samples = np.vstack([guides.resample(line, fine / 4.0) for line, _count, _tag in items])
            co, tris, kind = guides.refine(co, tris, kind, samples, fine / 2.5, self.project)  # edges under 0.6 of a gap near the strokes, so a small ring still has triangles inside it
            co, tris, kind, keys, self.bend_groups, skipped = loops.insert(co, tris, kind, items, self.gap, nearest, keys)
            self.skipped = len(skipped)
            self.why = sorted(set(skipped))
            if skipped:
                self.note = f"{len(skipped)} loop(s) left out: " + ", ".join(self.why)
        self.timings["guides"] = self.timings.get("guides", 0.0) + time.perf_counter() - started
        return co, [tuple(t) for t in tris.tolist()], [str(k) for k in kind.tolist()], tuple(sorted(keys))

    def _with_copies(self, mesh: MeshData, owner: np.ndarray) -> tuple[MeshData, np.ndarray]:
        """The worked-out pieces with every copy added, exactly alike, and which worked-out vertex each vertex came from."""
        points = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        back = np.arange(len(points))
        if not self.twins:
            return mesh, back
        of_piece: dict[int, list[int]] = {}
        for k, face in enumerate(mesh.faces):
            of_piece.setdefault(int(owner[face[0]]), []).append(k)
        all_points, all_back = [points], [back]
        faces = [tuple(face) for face in mesh.faces]
        kinds = list(mesh.face_materials)
        offset = len(points)
        for rep, move in self.twins.values():
            ids = of_piece.get(rep, [])
            if not ids:
                continue
            used, inverse = np.unique(np.fromiter((v for k in ids for v in mesh.faces[k]), dtype=np.int64), return_inverse=True)
            local, at = [], 0
            for k in ids:
                size = len(mesh.faces[k])
                local.append(tuple(int(v) + offset for v in inverse[at : at + size]))
                at += size
                kinds.append(mesh.face_materials[k])
            faces += special.wound(local, move)
            all_points.append(special.moved(points[used], move))
            all_back.append(used)
            offset += len(used)
        return MeshData(np.vstack(all_points), faces, kinds), np.concatenate(all_back)

    def _instances(self, mesh: MeshData, owner: np.ndarray) -> dict:
        """The result as one mesh per repeated piece with where each goes, and one mesh for everything else."""
        points = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        repeated: dict[int, list] = {}
        for rep, move in self.twins.values():
            repeated.setdefault(rep, []).append(move)
        of_piece: dict[int, list[int]] = {}
        for k, face in enumerate(mesh.faces):
            part = int(owner[face[0]])
            of_piece.setdefault(part if part in repeated else -1, []).append(k)

        def part_mesh(ids):
            used, inverse = np.unique(np.fromiter((v for k in ids for v in mesh.faces[k]), dtype=np.int64), return_inverse=True)
            faces, at = [], 0
            for k in ids:
                size = len(mesh.faces[k])
                faces.append(tuple(int(v) for v in inverse[at : at + size]))
                at += size
            return {"co": points[used], "faces": faces, "mat": np.asarray([int(mesh.face_materials[k]) if str(mesh.face_materials[k]).isdigit() else 0 for k in ids], dtype=np.int32)}

        groups = []
        for rep, moves in repeated.items():
            if rep not in of_piece:
                continue
            group = part_mesh(of_piece[rep])
            centre = moves[0][1]
            place = np.eye(4)
            place[:3, 3] = centre
            matrices = [place]
            for turn, _from, to, _mirror in moves:
                matrix = np.eye(4)
                matrix[:3, :3] = turn
                matrix[:3, 3] = to
                matrices.append(matrix)
            group.update(centre=centre, matrices=matrices)
            groups.append(group)
        rest = part_mesh(of_piece[-1]) if -1 in of_piece else {"co": np.zeros((0, 3)), "faces": [], "mat": np.zeros(0, dtype=np.int32)}
        return {"rest": rest, "groups": groups}

    def _balanced(self, coarse: MeshData):
        """Balanced: leftover triangles are kept out of the closing, and only faces that gain shape are split."""
        if self._tree is None:
            self._tree = _face_tree(self.co, self.faces)
        find = self._tree.find_nearest if isinstance(self._tree, BVHTree) else self._tree
        paired = pair_and_close(coarse, seed_count=2, force_quads=False, keep_quads=True, cancelled=lambda: self._stopped)[0]
        result = snap_new_vertices(split_faces(paired, faces_that_gain(paired, find, FLAT_SNAP_SHARE)), coarse.vertices, find)
        points, out_faces, keep = unfold(result.vertices, result.faces, find)
        return np.asarray(points, dtype=np.float64), [tuple(f) for f in out_faces], [result.face_materials[k] for k in keep]

    def _pair(self, co, faces, kinds, method) -> None:
        started = time.perf_counter()
        try:
            coarse = MeshData(co, faces, kinds)
            # Rows of quads would ignore the strokes and the paint, and they are what Auto means.
            if method == "AUTO" and self.output_kind != "TRIS" and not self.guide_lines and not self.strips and self.drop is None and not self.chunks:
                grid, flat = grid_options(self.tweak["rows"]), flat_options(self.tweak["flat"])
                if grid is not None:
                    if self._labels is None:
                        domes = smooth_areas(self.co, self.faces, options=grid)
                        self._labels = (domes, crease_areas(self.co, self.faces, domes >= 0, grid))
                    coarse = smooth_grids(self.co, self.faces, co, faces, kinds, self._labels, grid)[0]
                if flat is not None and len(coarse.faces) <= EVEN_FLAT_UP_TO:
                    coarse = flatten_regions(coarse, flat)[0]
            if self.style == "BALANCED" and method == "AUTO" and self.output_kind == "QUADS":  # quads split only where that gains shape
                self._paired = self._balanced(coarse)
                self.timings["pairing"] = self.timings.get("pairing", 0.0) + time.perf_counter() - started
                return
            made = make_pure(coarse, method, self.output_kind, self.notes, cancelled=lambda: self._stopped, guide_edges=self.guide_edges, keep_quads=True, merge_triangles=self.rough and method == "AUTO", join_triangles=self.collapse and method == "AUTO", chunk_faces=PAIR_CHUNK_FACES if self.cfg.name == "Light" else 0)
            # Strokes and a mirror seam must stay where they are.
            if self.output_kind != "TRIS" and method != "TRIS" and not self.guide_lines and not self.strips and self.mirror is None and not self._stopped:
                made = (self._fit(MeshData(*made)).vertices, made[1], made[2])
            self._paired = made
            self.timings["pairing"] = self.timings.get("pairing", 0.0) + time.perf_counter() - started
        except Cancelled:  # Stop was pressed: the step ends at its next stopping point
            pass
        except Exception as exc:  # shown as the job's warning
            self.warnings = [f"Auto Segments failed: {exc}"]

    def _turn_poles(self, mesh: MeshData, owner: np.ndarray) -> MeshData:
        """The same corners with edges spun where that leaves fewer poles (Tarini et al. 2010), or `mesh` if none was. Rebuilt parts stay as they are."""
        reducer = ChordReducer(mesh)
        rebuilt = [i for i, k in enumerate(self.picked) if self.ready[i][k]]
        reducer.freeze(np.flatnonzero(np.isin(owner, rebuilt)).tolist(), renamed=True)
        if not reducer.turn_poles():
            return mesh
        return MeshData(mesh.vertices, [tuple(face) for face in reducer.faces], list(reducer.materials))

    def _score(self, points, faces) -> None:
        try:
            self.topology = patterns.score(patterns.measure(points, faces))["score"]
        except Exception as exc:  # the score is only a report: a warning, and the job goes on
            self.warnings = [f"No tidiness score: {exc}"]

    def _polish(self, points, faces, materials, level) -> None:
        """Leaves the tidied mesh in `_paired` only if the score says it is tidier and nothing got broken."""
        try:
            before = patterns.measure(points, faces)
            tidied = polish(MeshData(points, faces, materials), level)[0]
            after = patterns.measure(tidied.vertices, tidied.faces)
            if patterns.score(after)["score"] > patterns.score(before)["score"] and after["non_manifold_edges"] <= before["non_manifold_edges"] and _folded(tidied.vertices, tidied.faces) <= _folded(np.asarray(points), faces):
                self._paired = tidied
        except Exception as exc:  # shown as the job's warning; the untidied quads stay
            self.warnings = [f"Tidying failed, kept the quads as they were: {exc}"]

    def _slice(self, work):
        """Generator: `work()` as a slice of its own, after handing the helpers their next tasks. Returns its result."""
        return (yield from tasks.run(work, tick=self._feed))

    def _counts(self, reference, view, pixel: float) -> tuple[int, int]:
        """(pixels that changed, pixels showing), by this job's way of judging."""
        return bakescore.picture_counts(reference, view, pixel) if self.baked else views.picture_counts(reference, view, pixel, self.rough)

    def _pictures(self, points, polys, center, half, pixels):
        """Generator: `pictures`, a few rows of rays per slice."""
        baked_from = self._bake_tree if self.baked else None
        return (yield from pictures(points, polys, center, half, pixels, baked_from=baked_from))

    def _allocate(self, allowed: float, piece_share: float) -> list[int]:
        """A size per piece. Copies stay alike: a piece and its copies are chosen as one, costing and losing what they all do."""
        many = [1] * len(self.levels)
        for rep, _move in self.twins.values():
            many[rep] += 1
        firsts = [i for i in range(len(self.levels)) if i not in self.twins]
        grouped = [[Level(level.faces * many[i], level.loss * many[i]) for level in self.levels[i]] for i in firsts]
        chosen = dict(zip(firsts, allocate(grouped, allowed, piece_share, [self.seen[i] * many[i] for i in firsts])))
        return [chosen[self.twins[i][0] if i in self.twins else i] for i in range(len(self.levels))]

    def _refine(self, finer: bool, kept: list[int]):
        """Generator: a size halfway between two tried ones, for the pieces holding most of the faces.

        Sizes double from one to the next, which is a huge step on a big piece: it would land well under what was asked
        for, or over it. So only the sizes next to the current pick, and only for the pieces where that matters.
        """
        if self.chunks:
            return
        counts = [self.levels[i][k].faces for i, k in enumerate(self.picked)]
        total = sum(counts)
        spent = 0
        for index in [i for i in sorted(range(len(counts)), key=lambda i: -counts[i]) if i not in self.twins][:REFINE_PIECES]:  # `kept`, the pick of the best try so far, stays right
            if spent > REFINE_COVER * total:
                break
            spent += counts[index] * (1 + sum(rep == index for rep, _move in self.twins.values()))
            at = self.picked[index]
            other = at + 1 if finer else at - 1
            whole = self.sizes[index][-1]
            triangles = _triangle_count(whole[1])
            if other < 0 or other >= len(self.sizes[index]) or self.ready[index][at] or self.ready[index][other]:
                continue
            first, second = len(self.sizes[index][at][1]) / triangles, len(self.sizes[index][other][1]) / triangles
            share = math.sqrt(first * second)
            if abs(share - first) * triangles < 8 or abs(share - second) * triangles < 8:
                continue
            pixels = self.picture_sizes[index]
            low, high = whole[0].min(axis=0), whole[0].max(axis=0)
            center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
            pixel = 2.0 * half / pixels
            self.stage = f"Trying one more size of piece {index + 1}"
            reference = yield from self._reference(center, half, pixels, whole[0], whole[1])
            yield
            reduced = yield from self._reduced(whole[0], whole[1], whole[2], share, keep=self._piece_lock(self.pieces[index]))
            view = yield from self._timed("pictures", self._pictures(reduced[0], reduced[1], center, half, pixels))
            wrong, _shown = self._counts(reference, view, pixel)
            where = max(at, other)
            self.sizes[index].insert(where, reduced)
            self.levels[index].insert(where, Level(len(reduced[1]), wrong * pixel * pixel))
            self.ready[index].insert(where, False)
            if where <= kept[index]:
                kept[index] += 1
            for twin, (rep, move) in self.twins.items():  # its copies get the same size
                if rep == index:
                    self.sizes[twin].insert(where, (special.moved(reduced[0], move), special.wound(reduced[1], move), reduced[2]))
                    self.levels[twin].insert(where, self.levels[index][where])
                    self.ready[twin].insert(where, False)
                    if where <= kept[twin]:
                        kept[twin] += 1

    def _reduce(self, result, owner, loss, reference, center, half, pixel, take: bool = True):
        """Generator: take quads away until the view has changed by the amount asked for. Returns (mesh, its vertices' pieces, loss).

        Quads only come off the worked-out pieces, so copies stay exact; each look is at them all. With `take` False,
        messy parts are only redrawn (Redraw Messy Quads), and nothing else changes.
        """
        if loss > self.detail or (not take and not self.chords):
            return result, owner, loss
        # Up to CHORDS_UP_TO quads, the reducer that redraws messy patches as grids and prices collapses by the
        # pixels they change; past that, the plain one, which only counts distance.
        chords = self.chords and len(result.faces) <= CHORDS_UP_TO
        judge = PictureJudge(reference, center, half, self.max_pixels, TURNED_DEG) if chords else None
        reducer = yield from self._slice(lambda: ChordReducer(result, judge) if chords else quad_reducer(result))
        if chords:
            chords_began = time.perf_counter()
            reducer.deadline = chords_began + CHORDS_SECONDS
            reducer.flag_messy(CHORDS_MESSY)
            reducer.project = self.project
        start_owner = owner
        # A rebuilt part is as few rows as it can hold. Its long flat quads look cheap to take away, but each one shows.
        rebuilt = [i for i, k in enumerate(self.picked) if self.ready[i][k]]
        reducer.freeze(np.flatnonzero(np.isin(owner, rebuilt)).tolist())
        limit = PIECE_LIMIT * self.detail / 100.0

        def over_limit(mesh, view, share=1.0):
            fa, fb = patch.face_adjacency(mesh.faces)
            labels = np.unique(patch._labels(len(mesh.faces), fa, fb), return_inverse=True)[1]
            parts = parts_changed_too_much(reference, view, pixel, labels, share * limit, self.baked)
            kept = np.isin(labels, parts)
            return np.asarray(sorted({v for k in np.flatnonzero(kept).tolist() for v in mesh.faces[k]}), dtype=np.int64)

        full, back = yield from self._slice(lambda: self._with_copies(result, owner))
        view = yield from self._timed("pictures", self._pictures(np.asarray(full.vertices), full.faces, center, half, self.max_pixels))
        limited = yield from self._slice(lambda: over_limit(full, view))
        reducer.freeze(sorted(set(back[limited].tolist())))  # whatever is at its limit stays as it is
        already = set(reducer.original_ids(sorted(set(back[limited].tolist()))))  # numbered as the reducer started: parts over before are not to blame

        def look(saved):
            """Generator: (mesh, its vertices' pieces, change, corners of parts past their limit) as the reducer stands.

            With the chord reducer, parts past their limit go back to `saved` on their own and are looked at again, so one
            bad bolt does not cost every other part its progress (with hundreds of parts the looks ran out at half the target).
            """
            attempt, attempt_owner, now, ruined = yield from one_look()
            if ruined and chords:
                reducer.roll_back_parts(saved, reducer.original_ids(ruined))
                attempt, attempt_owner, now, ruined = yield from one_look()
            return attempt, attempt_owner, now, ruined

        def one_look():
            attempt = yield from self._slice(reducer.result)
            ids = np.asarray(reducer.original_ids(range(len(attempt.vertices))), dtype=np.int64)
            attempt_owner = start_owner[np.asarray(reducer.home, dtype=np.int64)[ids]] if chords else start_owner[ids]
            full, back = yield from self._slice(lambda: self._with_copies(attempt, attempt_owner))
            self.stage = f"Taking away quads: {len(full.faces):,}"
            view = yield from self._timed("pictures", self._pictures(np.asarray(full.vertices), full.faces, center, half, self.max_pixels))
            wrong, shown = self._counts(reference, view, pixel)
            ruined = yield from self._slice(lambda: over_limit(full, view))
            if len(ruined):  # those close behind would cost a look each, and with hundreds of parts the looks run out
                ruined = yield from self._slice(lambda: over_limit(full, view, NEARLY))
            found = sorted(set(back[ruined].tolist()))
            ruined = [v for v, name in zip(found, reducer.original_ids(found)) if name not in already]  # parts over before, and frozen, are not to blame
            return attempt, attempt_owner, 100.0 * wrong / max(shown, 1), ruined
        saved = yield from self._slice(reducer.save)
        if chords:
            self.stage = "Redrawing messy patches as grids..."
            yield from self._slice(reducer.tidy)
            attempt, attempt_owner, now, ruined = yield from look(saved)
            if now <= self.detail and not ruined:
                result, owner, loss = attempt, attempt_owner, now
                saved = yield from self._slice(reducer.save)
            else:
                yield from self._slice(lambda: reducer.restore(saved))
        step = REDUCE_STEP
        looks = 0
        finest, most = (CHORDS_FINEST, CHORDS_LOOKS) if chords else (REDUCE_FINEST, MOST_LOOKS)
        if not take:  # redraw only: the faces were chosen by something else
            most = 0
        while step >= finest and looks < most:
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
            attempt, attempt_owner, now, ruined = yield from look(saved)
            if now <= self.detail and not ruined:
                result, owner, loss = attempt, attempt_owner, now
                saved = yield from self._slice(reducer.save)
                continue
            names = reducer.original_ids(ruined)
            yield from self._slice(lambda: reducer.restore(saved))
            if ruined:  # those parts stop here, and the rest can go on
                reducer.freeze(names, renamed=True)
            else:
                step *= 0.5
        if chords:
            self.notes.append(f"Redraw Messy Quads was on: {time.perf_counter() - chords_began:.0f} s of this run; tidier loops cost more faces (7-50% more in tests)")
        return result, owner, loss

    def _build(self, co, faces, kinds, method, goal=0):
        """Generator: faces for these pieces by `method` in the chosen output. Returns (points, faces, materials) or None."""
        if method == "FIELD":  # a part it could do skips this: it is here because it was refused or isn't plain enough
            method = "AUTO"
        if method in BLENDER_OPERATOR_METHODS:
            try:
                made = _quadriflow(self.context, co, faces, kinds, goal) if method == "QUADRIFLOW" else _fill(co, faces, kinds, method, goal)
                return triangulate(*made) if self.output_kind == "TRIS" else made
            except Exception as exc:
                self.notes.append(f"{method.title()} did not work on a part ({str(exc)[:60]}): Auto was used there")
                method = "AUTO"
        self._paired = None
        yield from tasks.run(lambda: self._pair(co, faces, kinds, method), stopped=lambda: self._stopped)
        return None if self._stopped else self._paired

    def _method(self, piece: int) -> str:
        return self.methods.get(piece, self.method)

    def _vote(self, points, faces) -> np.ndarray:
        """The source piece of each result face: by the nearest source vertices of each connected part of the result."""
        if len(self.pieces) == 1:
            return np.zeros(len(faces), dtype=np.int64)
        if self._kd is None:
            owner = np.full(len(self.co), -1, dtype=np.int64)
            for index, ids in enumerate(self.pieces):
                owner[list(chain.from_iterable(self.faces[f] for f in ids.tolist()))] = index
            used = np.flatnonzero(owner >= 0)
            kd = KDTree(len(used))
            coordinates = self.co[used].tolist()
            # In random order: a welded source's vertices come sorted, and Blender's balance took minutes on them, holding up everything.
            for slot in np.random.default_rng(0).permutation(len(used)).tolist():
                kd.insert(coordinates[slot], slot)
            del coordinates
            kd.balance()
            self._kd, self._kd_owner = kd, owner[used]
        fa, fb = patch.face_adjacency(faces)
        labels = patch._labels(len(faces), fa, fb)
        order = np.argsort(labels, kind="stable")
        starts = np.concatenate(([0], np.flatnonzero(labels[order][1:] != labels[order][:-1]) + 1, [len(order)]))
        piece = np.zeros(len(faces), dtype=np.int64)
        samples = [order[a:b][np.linspace(0, b - a - 1, min(b - a, 24)).astype(np.int64)] for a, b in zip(starts[:-1], starts[1:])]
        corners = points[[faces[f][0] for f in np.concatenate(samples).tolist()]]
        votes = np.split(self._kd_owner[_nearest_points(self._kd, corners)], np.cumsum([len(s) for s in samples])[:-1])
        for a, b, vote in zip(starts[:-1], starts[1:], votes):
            piece[order[a:b]] = np.bincount(vote).argmax()
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
                if len(polys) > SOURCE_BACK or self.output_kind == "QUADS":  # the source's own faces are often triangles
                    continue
                made = triangulate(co, polys, kinds) if self.output_kind == "TRIS" else (co, list(polys), list(kinds))
            builds[p] = made
        return (yield from self._slice(lambda: self._splice(points, faces, mats, piece, builds)))

    def _inspected(self, points, faces, piece):
        """Generator: all the numbers for one mesh, the crossing test and the other checks in slices."""
        began = time.perf_counter()
        pairs = yield from checks.crossing_slices(points, faces)
        self._feed()
        out = yield from checks.inspect_steps(points, faces, piece, pairs=pairs)
        if self._known is not None:
            out.new_cross, new = repair.new_by_piece(self._known, pairs, points, faces, piece)
            out.crossing[:] = False
            out.crossing[new.ravel()] = True
        self.timings["inspect"] = self.timings.get("inspect", 0.0) + time.perf_counter() - began
        return out

    def _near_source(self, part: int, where: np.ndarray):
        """(closest points, normals) on one piece of the source. Each piece's tree is made once, when it is needed."""
        part = -1 if self.locked is not None else part  # a cut piece came back as one surface
        tree = self._piece_trees.get(part)
        if tree is None:
            points, polys, _kinds = (self.co, self.faces, None) if part < 0 else self._piece_arrays(self.pieces[part])
            tree = self._piece_trees[part] = BVHTree.FromPolygons(points.tolist(), [list(f) for f in polys], epsilon=0.0)
        hits = [tree.find_nearest(p) for p in np.asarray(where).tolist()]
        return np.asarray([h[0] for h in hits]), np.asarray([h[1] for h in hits])

    def _fix_locally(self, points, faces, mats, piece, view, after):
        """Generator: clear new crossings spot by spot. Kept only if nothing got broken and the camera still agrees."""
        same = (points, faces, mats, piece, view, after, 0)
        if not after.new_cross or self.mirror is not None or self.outside is not None or self.guide_lines or self.strips:  # those need their seams and strokes left where they are
            return same
        self.stage = "Repairing crossings"
        yield
        began = time.perf_counter()
        yield  # one spot per slice: a spot's fills can take seconds, the whole repair up to REPAIR_SECONDS
        fixed = yield from repair.fix_crossings_steps(points, faces, mats, piece, after.pairs, self._known, self._near_source, self.output_kind == "TRIS", growth=0.03 if self.repair_fills else 0.0, stopped=lambda: self._stopped or time.perf_counter() > self._repair_until)
        numbers = fixed[4]
        self.timings["fix crossings"] = self.timings.get("fix crossings", 0.0) + time.perf_counter() - began
        if not numbers["pushed"] and not numbers["filled"]:
            return same
        now = yield from self._inspected(fixed[0], fixed[1], fixed[3])
        if any(getattr(now, k) > getattr(after, k) for k in ("nm_edges", "nm_verts", "flipped", "dups", "holes", "selfx", "folded")):
            return same
        seen = yield from self._look(fixed[0], fixed[1], *self._view)
        if seen[0] > max(1.1 * self.detail, 1.02 * view[0]):
            return same
        return fixed[0], fixed[1], fixed[2], fixed[3], seen, now, numbers["pushed"] + numbers["filled"]

    def _untangle(self, points, faces, piece, view, after):
        """Generator: corners of faces turned inward or folded over moved until they face out. Kept only if nothing got
        broken and the camera still agrees. (points, faces, view, after, corners moved)."""
        same = (points, faces, view, after, 0)
        if self.mirror is not None or self.outside is not None or self.guide_lines or self.strips:  # seams and strokes stay put
            return same
        self.stage = "Turning folded faces back"
        yield
        began = time.perf_counter()
        found = yield from self._slice(lambda: checks.defects(points, faces, self.co, self.faces, pairs=after.pairs))
        self._defects = (points, faces, found)
        bad = found.inward | found.folded_over
        if not bad.any():
            return same
        moved, count = yield from repair.untangle_steps(points, faces, piece, bad, self._near_source, stopped=lambda: self._stopped or time.perf_counter() > began + UNTANGLE_SECONDS)  # one group of corners per slice
        self.timings["untangle"] = self.timings.get("untangle", 0.0) + time.perf_counter() - began
        if not count:
            return same
        now = yield from self._inspected(moved, faces, piece)
        if any(getattr(now, k) > getattr(after, k) for k in ("nm_edges", "nm_verts", "flipped", "dups", "holes", "selfx", "folded")):
            return same
        left = yield from self._slice(lambda: checks.defects(moved, faces, self.co, self.faces, pairs=now.pairs))
        if (left.inward | left.folded_over).sum() >= bad.sum():
            return same
        kept = list(faces)  # a new list: copies placed as instances no longer match
        self._defects = (moved, kept, left)
        seen = yield from self._look(moved, faces, *self._view)
        if seen[0] > max(1.1 * self.detail, 1.02 * view[0]):
            return same
        return moved, kept, seen, now, count

    def _serious(self, points, faces, after):
        """Generator: per face, turned inward, folded over or crossing (the Issue Finder's serious kinds). Kept for reuse."""
        found = yield from self._slice(lambda: checks.defects(points, faces, self.co, self.faces, pairs=after.pairs))
        self._defects = (points, faces, found)
        return found.inward | found.folded_over | found.crossing

    def _affordable(self, bad: list[int], start_faces: int) -> list[int]:
        """As many of these parts as can go one size up inside REPAIR_GROWTH, so one big part does not stop the small ones."""
        room = (REPAIR_GROWTH - 1.0) * start_faces
        out = []
        for p in sorted(bad, key=lambda p: len(self.sizes[p][self.picked[p]][1])):
            if self.picked[p] + 1 < len(self.sizes[p]):
                room -= max(0, len(self.sizes[p][self.picked[p] + 1][1]) - len(self.sizes[p][self.picked[p]][1]))
            if room < 0:
                break
            out.append(p)
        return out

    def _check(self, points, faces, mats, piece, view):
        """Generator: measure the result against the source, repair the parts that got new problems, say what happened."""
        began = time.perf_counter()
        self.stage = "Checking for broken topology"
        yield
        if self._src_check is None or self._src_pieces != len(self.pieces):  # the size ladder hands its rungs the search's own
            self._src_pieces = len(self.pieces)
            source_piece = np.zeros(len(self.faces), dtype=np.int64)
            for index, ids in enumerate(self.pieces):
                source_piece[ids] = index
            self._src_check = yield from self._inspected(self.co, self.faces, source_piece)
            self._known = repair.Known(self.co, self.faces, source_piece, self._src_check.pairs)
        before = self._src_check
        after = yield from self._inspected(points, faces, piece)
        keys = ("nm_edges", "nm_verts", "flipped", "dups", "holes", "selfx", "folded")
        first = {k: getattr(after, k) for k in keys}
        self._repair_until = time.perf_counter() + REPAIR_SECONDS  # from here: checking the source first must not use it up
        start_faces = len(faces)
        points, faces, view, after, _turned = yield from self._untangle(points, faces, piece, view, after)
        # Parts one size up first, then the crossings left filled spot by spot. A size up costs a pairing and a look, a
        # filled spot about a second, and the fills used to eat the whole time before any part was remade: the chess
        # knight's crossings took 30 s of fills and stayed, one size up cleared them. Parts still turned inward or
        # folded over after untangling go too: their size is too coarse for how thin they are.
        fixed, stuck, serious = 0, set(), None
        for attempt in range(0 if self.chunks else FIX_TRIES):  # a cut piece's parts are one surface: they can't be remade alone
            spots = checks.new_spots(before, after)
            bent = set(np.unique(piece[after.bent]).tolist()) if len(after.bent) == len(piece) else set()
            spots = sorted(spots, key=lambda p: p not in bent)  # folded quads first: small thin parts, cheap to remake a size up
            if serious is None:
                serious = yield from self._serious(points, faces, after)
            spots += [p for p in checks.worst_parts(piece, serious) if p not in stuck and p not in spots]
            bad = self._affordable([p for p in spots[:MOST_REPAIRED] if len(self.pieces[p]) <= REPAIR_MOST_FACES and not self._cut(self.pieces[p])], start_faces)
            if not bad or self._stopped or time.perf_counter() > self._repair_until:
                break
            self.stage = f"Repairing {len(bad)} part{'s' if len(bad) != 1 else ''}, try {attempt + 1}"
            yield
            picked, tuned = list(self.picked), dict(self.tuned)
            # The source's own faces are often triangles, so that last try isn't for an all-quads result.
            tried = yield from self._remake(bad, points, faces, mats, piece, attempt == FIX_TRIES - 1 and self.output_kind != "QUADS")
            seen = yield from self._look(tried[0], tried[1], *self._view)
            if seen[0] > max(1.1 * self.detail, 1.02 * view[0]) or len(tried[1]) > REPAIR_GROWTH * start_faces:  # worse than what it repairs
                self.picked, self.tuned = picked, tuned
                break
            now = yield from self._inspected(tried[0], tried[1], tried[3])
            better = yield from self._serious(tried[0], tried[1], now)
            if better.sum() > serious.sum():  # a size up that leaves more broken faces is no repair
                self.picked, self.tuned = picked, tuned
                stuck |= set(bad)
                continue
            stuck |= {p for p in bad if better[tried[3] == p].sum() >= serious[piece == p].sum()}  # one size up didn't help: try others next
            fixed += max(0, len(checks.new_spots(before, after)) - len(checks.new_spots(before, now)))
            points, faces, mats, piece = tried
            view, after, serious = seen, now, better
        # The fills have what the remakes left of the time. More does not help: at 55 to 120 s instead of 30 they cleared
        # no more parts on four of six test models, as the spots left are ones a fill can't clear.
        points, faces, mats, piece, view, after, spots = yield from self._fix_locally(points, faces, mats, piece, view, after)
        fixed += spots
        points, faces, mats, piece, view, after = yield from self._redo_rough(points, faces, mats, piece, view, after)
        left = checks.new_spots(before, after)
        flag = np.zeros(len(faces), bool)
        for p in left:
            here = piece == p
            mask = here & (after.broken | after.crossing)
            flag |= mask if mask.any() else here
        self.face_piece, self.flag = piece, flag
        seen_before = self._defects
        if seen_before is None or seen_before[0] is not points or seen_before[1] is not faces:  # the repairs changed it, or it was never checked
            found = yield from self._slice(lambda: checks.defects(points, faces, self.co, self.faces, pairs=after.pairs))
        else:
            found = seen_before[2]
        self.bad = found.any()
        self.found = found
        if self.redo_numbers is not None:
            spots, bad_before = self.redo_numbers
            self.notes.append(f"Redo Rough Spots redid {spots} spot{'s' if spots != 1 else ''}: {max(bad_before - int(self.bad.sum()), 0):,} bad face(s) fewer")
        self.timings["check and repair"] = self.timings.get("check and repair", 0.0) + time.perf_counter() - began
        if left and time.perf_counter() > self._repair_until:
            self.notes.append(f"Repair stopped at its {REPAIR_SECONDS:.0f} s limit: {len(left)} part(s) with crossings are selected on Apply")
        check = {"first": first, "text": checks.describe(before, after, max(fixed, 0), len(left), int(found.overlap.sum())), "result": {k: getattr(after, k) for k in keys}, "source": {k: getattr(before, k) for k in keys}, "fixed": max(fixed, 0), "unfixed": len(left), "bad": int(self.bad.sum())}
        return points, faces, mats, view, check

    def _redo_rough(self, points, faces, mats, piece, view, after):
        """Generator: the spots still bad after the repairs, each in a small region whose outline stays, laid out again:
        its inner corners put back towards the source, part or all of the way. A region's redo is kept if it leaves fewer bad
        faces there; a round is kept if nothing broke and the camera still agrees. Returns (points, faces, mats, piece, view, after)."""
        same = (points, faces, mats, piece, view, after)
        if not self.redo_rough or self.output_kind == "TRIS" or self.mirror is not None or self.outside is not None or self.guide_lines or self.strips:
            return same
        began = time.perf_counter()
        until = began + REDO_SECONDS
        keys = ("nm_edges", "nm_verts", "flipped", "dups", "holes", "selfx", "folded")
        redone, bad_before = 0, None
        for round_index in range(REDO_ROUNDS):
            if self._stopped or time.perf_counter() > until:
                break
            self.stage = f"Redoing rough spots, round {round_index + 1}"
            yield
            found = yield from self._slice(lambda: checks.defects(points, faces, self.co, self.faces, pairs=after.pairs))
            bad = found.any()
            if bad_before is None:
                bad_before = int(bad.sum())
            if not bad.any():
                break
            spots = issues.find_spots(points, faces, {k: getattr(found, k) for k in checks.Defects.BROKEN}, piece, REDO_SPOTS)
            rings = REDO_RINGS + round_index
            moved, regions = yield from self._slice(lambda: self._redo_spots(points, faces, piece, spots, rings, until))
            if not regions:
                break
            now = yield from self._inspected(moved, faces, piece)
            if any(getattr(now, k) > getattr(after, k) for k in keys):
                # A moved corner can pass through a face outside its region's window: those regions go back, the rest stay.
                touched = np.zeros(len(points), bool)
                for f in np.flatnonzero(now.crossing | now.bent if len(now.bent) == len(faces) else now.crossing).tolist():
                    touched[list(faces[f])] = True
                kept = [ids for ids in regions if not touched[ids].any()]
                back = np.array(points, dtype=np.float64)
                for ids in kept:
                    back[ids] = moved[ids]
                self.redo_counts["undone"] = self.redo_counts.get("undone", 0) + len(regions) - len(kept)
                moved, regions = back, kept
                if kept:
                    now = yield from self._inspected(moved, faces, piece)
                if not kept or any(getattr(now, k) > getattr(after, k) for k in keys):
                    self.redo_counts["broke"] = {k: (getattr(after, k), getattr(now, k)) for k in keys if getattr(now, k) != getattr(after, k)}
                    break
            seen = yield from self._look(moved, faces, *self._view)
            if seen[0] > max(1.1 * self.detail, 1.02 * view[0]):
                self.redo_counts["view"] = (view[0], seen[0])
                break
            points, view, after = moved, seen, now
            faces = list(faces)  # a new list: copies placed as instances no longer match
            redone += len(regions)
        self.timings["redo rough spots"] = time.perf_counter() - began
        if redone:
            self.redo_numbers = (redone, bad_before)
            self._defects = None
        return points, faces, mats, piece, view, after

    def _redo_spots(self, points, faces, piece, spots, rings: int, until: float):
        """(points, the corners moved in each region redone): each spot's region with its inner corners laid out the way
        that leaves the fewest bad faces round it, where that is fewer than now. Plain NumPy and BVH queries."""
        if self._src_middles is None:
            sizes = np.fromiter((len(f) for f in self.faces), dtype=np.int64, count=len(self.faces))
            flat = np.fromiter(chain.from_iterable(self.faces), dtype=np.int64, count=int(sizes.sum()))
            self._src_middles = np.add.reduceat(self.co[flat], np.concatenate(([0], np.cumsum(sizes)[:-1])), axis=0) / sizes[:, None]
        taken = np.zeros(len(faces), bool)
        users = redo.corner_users(faces)
        points = np.array(points, dtype=np.float64)
        regions = []
        for spot in spots:
            if self._stopped or time.perf_counter() > until:
                break
            region = redo.carve(faces, spot["faces"], rings, REDO_MOST_FACES, taken, users)
            if not len(region):
                continue
            taken[region] = True
            window = redo.carve(faces, region, 2, 50 * REDO_MOST_FACES, users=users)
            used = np.unique(np.fromiter((v for f in window.tolist() for v in faces[f]), dtype=np.int64))
            local = {int(v): k for k, v in enumerate(used.tolist())}
            near = [tuple(local[v] for v in faces[f]) for f in window.tolist()]
            mine = np.flatnonzero(np.isin(window, region))
            free = redo.free_corners(near, mine, len(used))
            if not free.any():
                continue
            self.redo_counts["regions"] += 1
            co = points[used]
            region_faces = [near[i] for i in mine.tolist()]
            edge = float(np.mean([np.linalg.norm(co[f[k]] - co[f[k - 1]]) for f in region_faces for k in range(len(f))]))
            on = self._near_source(int(np.bincount(piece[window[mine]]).argmax()), co[free])[0]
            close = np.linalg.norm(on - co[free], axis=1) < edge
            best, best_co = self._bad_here(co, near), None
            for share in REDO_SHARES:
                tried = co.copy()
                tried[free] = redo.toward(co[free], on, close, share)
                bad = self._bad_here(tried, near)
                if redo.better(best, bad):
                    best, best_co = bad, tried
            if best_co is None:
                self.redo_counts["no better"] += 1
                continue
            points[used[free]] = best_co[free]
            regions.append(used[free])
            self.redo_counts["kept"] += 1
        return points, regions

    def _bad_here(self, points, faces) -> tuple[int, int]:
        """(bad faces, bent quads) of this patch of the result, against the source round it. Sunk is not asked: that needs the whole source."""
        low, high = points.min(axis=0), points.max(axis=0)
        margin = 0.5 * float((high - low).max())
        ids = np.flatnonzero(((self._src_middles >= low - margin) & (self._src_middles <= high + margin)).all(axis=1))
        if not len(ids):
            return 0, 0
        src_co, src_faces, _kinds = self._piece_arrays(ids)
        found = checks.defects(points, faces, src_co, src_faces)
        return int(sum(getattr(found, k) for k in checks.Defects.BROKEN if k != "sunk").sum()), int(found.bent.sum())

    def _kill_worker(self) -> None:
        if self._worker is not None:
            try:
                self._worker.stdin.close()
            except (OSError, AttributeError):
                pass
            self._worker.kill()
            self._worker.wait()
            self._worker = None

    def _rows_once(self, areas, target: int):
        """Generator: QuadriFlow on every piece big enough, in a second Blender, snapped to the source. Returns (points, faces, materials)."""

        total = float(areas.sum()) or 1.0
        folder = str(hardware.job_folder("curiomesh_rows_"))

        def clean_up():  # also run if Blender quits in the middle
            self._kill_worker()
            shutil.rmtree(folder, ignore_errors=True)

        atexit.register(clean_up)
        try:
            sizes = np.asarray([len(ids) for ids in self.pieces], dtype=np.float64)
            wants = (target * 0.5 * (areas / total + sizes / sizes.sum())).astype(np.int64)  # a small part with many faces is usually detail
            jobs = [k for k, want in enumerate(wants.tolist()) if want >= ROWS_SMALLEST]
            for k in jobs:
                points, polys, _kinds = self._piece_arrays(self.pieces[k])
                np.savez(os.path.join(folder, f"in_{k}.npz"), co=points, target=wants[k], lens=np.asarray([len(p) for p in polys], dtype=np.int64), flat=np.asarray([v for p in polys for v in p], dtype=np.int64))
                yield
            at, made = 0, 0
            while at < len(jobs):
                command, env = parallel.helper_command("rows_worker.main", [folder])
                self._worker = subprocess.Popen(  # handed one piece at a time on stdin: it ends when that closes, even if this Blender crashes
                    command, env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                since, sent = time.perf_counter(), None
                while at < len(jobs):
                    k = jobs[at]
                    if sent != k:
                        try:
                            self._worker.stdin.write(b"%d\n" % k)
                            self._worker.stdin.flush()
                        except OSError:  # it has exited: seen just below
                            pass
                        sent = k
                    self.stage = f"Laying rows on piece {at + 1} of {len(jobs)}"
                    self.progress = (at, len(jobs))
                    if os.path.exists(os.path.join(folder, f"out_{k}.npz")):
                        at += 1
                        made += 1
                        since = time.perf_counter()
                        continue
                    if self._stopped:
                        self._kill_worker()
                        return None
                    if time.perf_counter() - since > ROWS_SECONDS + ROWS_SECONDS_PER_QUAD * float(wants[k]) or self._worker.poll() is not None:
                        if self._worker.poll() is None or not os.path.exists(os.path.join(folder, f"out_{k}.npz")):
                            exited = self._worker.poll() is not None
                            self._kill_worker()
                            at += 1
                            if exited and not made:  # it could not even start: no use starting it again for every piece
                                self.notes.append("Even Rows could not start a background Blender: the pieces were reduced with Decimate instead")
                                at = len(jobs)
                            else:
                                self._refused += 1
                            break
                    yield WAITING
            self._kill_worker()
            all_points, all_faces, all_kinds = [], [], []
            offset = 0
            leftovers = []
            for k, ids in enumerate(self.pieces):
                points, polys, kinds = self._piece_arrays(ids)
                path = os.path.join(folder, f"out_{k}.npz")
                rows = None
                if os.path.exists(path):
                    with hardware.load_npz(path) as saved:  # an open file couldn't be deleted with the folder
                        rows = {name: saved[name] for name in saved.files}
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
                tree = _face_tree(points, polys)
                find = tree.find_nearest if isinstance(tree, BVHTree) else tree
                triangles = np.asarray([(q[0], q[i], q[i + 1]) for q in polys for i in range(1, len(q) - 1)], dtype=np.int64)

                def sharpen(moved, points=points, triangles=triangles):
                    grid = KDTree(len(moved))
                    for i, co in enumerate(moved.tolist()):
                        grid.insert(co, i)
                    grid.balance()
                    return seek_features(moved, points, triangles, lambda at: _nearest_points(grid, at))

                # With few source triangles per quad, their planes say more about noise than about shape.
                wanted_sharp = len(triangles) >= ROWS_SHARP_INPUT * len(faces)
                done = yield from self._step(lambda: snap_rows(rows["co"], faces, find, sharpen=sharpen if wanted_sharp else None))
                if self._stopped:
                    return None
                moved, quads = done if done is not None else (None, [])
                # Repair had to give up on too many quads: QuadriFlow doesn't suit this piece.
                if done is None or sum(1 for q in quads if len(q) != 4) > ROWS_BAD * len(faces):
                    triangles_in = sum(len(p) - 2 for p in polys)
                    want = max(KEEP_TRIANGLES, 2 * int(wants[k]))
                    if triangles_in > want:
                        points, polys, kinds = _decimate(self.context, points, polys, kinds, want / triangles_in)
                    leftovers.append((points, polys, kinds))
                    self._refused += 1
                    continue
                all_points.append(moved)
                all_faces += [tuple(v + offset for v in q) for q in quads]
                centers = np.asarray([moved[list(q)].mean(axis=0) for q in quads])
                all_kinds += [kinds[f] for f in (_nearest_faces(tree, centers).tolist())]
                offset += len(moved)
                yield
        finally:
            atexit.unregister(clean_up)
            clean_up()
        if leftovers:
            pts, fcs, knd, cursor = [], [], [], 0
            for points, polys, kinds in leftovers:
                pts.append(points)
                fcs += [tuple(v + cursor for v in p) for p in polys]
                knd += kinds
                cursor += len(points)
            closed = yield from self._step(lambda: pair_and_close(MeshData(np.vstack(pts), fcs, knd), seed_count=1)[0])
            if closed is None:
                return None
            all_points.append(np.asarray(closed.vertices))
            all_faces += [tuple(v + offset for v in f) for f in closed.faces]
            all_kinds += list(closed.face_materials)
        return np.vstack(all_points), all_faces, all_kinds

    def _run_rows(self):
        """Even Rows: order first. Lands near the amount asked for by the density of the layout, and reports what it got when time or faces run out."""
        if not self.helpers:
            self.warnings = [ROWS_NEEDS_HELPERS]
            return

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
        pixel = 2.0 * half / self.max_pixels
        reference = yield from pictures(self.co, self.faces, center, half, self.max_pixels)
        source = sum(len(f) - 2 for f in self.faces)
        most = int(ROWS_MOST * source)
        target = self.target_faces or min(max(int(ROWS_START * source), 256), most)
        best = None
        reference_shaded = shaded(reference, self.co, self.faces, center, half, self.max_pixels)
        self._view = (center, half, reference, reference_shaded, pixel)

        def judge(points, out_faces, kinds, loss):
            nonlocal best
            if self.target_faces:  # Reduce To: the nearest to the amount asked for
                rank = (abs(len(out_faces) - self.target_faces),)
            else:
                fits = loss <= 1.1 * self.detail
                rank = (not fits, len(out_faces) if fits else loss)  # as in Keep Detail: fits first, then the fewest faces
            if best is None or rank < best[0]:
                best = (rank, points, out_faces, kinds, loss)

        tries = []  # Reduce To: (layout asked for, faces it gave)
        for _round in range(ROWS_ROUNDS):
            built = yield from self._rows_once(areas, target)
            if built is None:
                return
            points, out_faces, kinds = built
            if self.target_faces:
                # QuadriFlow lands near what it is asked for, but the paired small parts and their closure don't scale with it.
                judge(points, out_faces, kinds, None)
                got = len(out_faces)
                if abs(got - self.target_faces) <= CLOSE_ENOUGH * self.target_faces:
                    break
                tries.append((target, got))
                wanted = next_ask(tries, self.target_faces, 64, most)
                if wanted == target:
                    break
                target = wanted
                continue
            self.stage = "Checking the result..."
            loss, _shading, _soft = yield from self._look(points, out_faces, *self._view)
            judge(points, out_faces, kinds, loss)
            if abs(loss - self.detail) <= 0.1 * self.detail:
                break
            # Change falls about as the square root of the faces. A denser layout also has to fit in the time allowed.
            room = (ROWS_TIME - (time.perf_counter() - self.started)) / ROWS_SECONDS_ESTIMATE
            wanted = min(max(int(target * (loss / self.detail) ** 2), 64), most, int(room))
            if wanted <= target and loss > self.detail or abs(wanted - target) < 0.05 * target:
                break
            target = wanted
        _rank, points, out_faces, kinds, loss = best
        if self._refused:
            self.notes.append(f"QuadriFlow gave up on {self._refused} parts: they were paired instead")
        if self.output_kind == "QUADS" and any(len(face) != 4 for face in out_faces):  # stuck quads came back as triangles: Quads means all quads
            mesh = MeshData(points, out_faces, kinds)
            closed = yield from self._step(lambda: _close_pieces(mesh))
            if closed is not None:
                points, out_faces, kinds = np.asarray(closed[0].vertices, dtype=np.float64), [tuple(f) for f in closed[0].faces], list(closed[0].face_materials)
                if closed[4]:
                    self.notes.append(f"{closed[4]} part(s) only closed by the last-resort split: up to 3-4x the faces and poor quads there, so check them")
        elif self.output_kind == "TRIS":
            points, out_faces, kinds = triangulate(points, out_faces, kinds)
        loss, shading, soft = yield from self._look(points, out_faces, *self._view)
        self.face_piece, self.flag = self._vote(points, out_faces), np.zeros(len(out_faces), dtype=bool)
        if any(len(face) == 4 for face in out_faces):
            yield from self._step(lambda: self._score(points, out_faces))
        out_fa, out_fb = patch.face_adjacency(out_faces)
        parts = np.unique(patch._labels(len(out_faces), out_fa, out_fb), return_inverse=True)[1]
        self.fraction = 1.0
        self._show((points, out_faces, kinds, parts))
        shown = self.output["faces"]
        whole = self.mirror[3] if self.mirror is not None else len(self.faces)
        quads = sum(1 for face in shown if len(face) == 4)
        self.report = {"before": whole, "faces": len(shown), "quads": quads, "tris": sum(1 for face in shown if len(face) == 3), "pieces": len(self.pieces), "looks": self.looks, "loss": loss, "shading": shading, "shading_soft": soft, "topology": self.topology, "asked": self.target_faces * (2 if self.mirror is not None else 1), "seconds": time.perf_counter() - self.started, "timings": {k: round(v, 1) for k, v in self.timings.items()}, "guides": None, "bend": None, "check": None, "notes": list(dict.fromkeys(self.notes))}

    def _step(self, work):
        """Generator: plain NumPy work as a slice of its own. Returns its result, or None if it failed."""

        try:
            return (yield from tasks.run(work, stopped=lambda: self._stopped))
        except Exception as exc:
            self.warnings = [f"Even Rows: {exc}"]
            return None

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
            if len(ids) > KEEP_SHARE * len(self.faces) or (self.locked is not None and self._piece_lock(ids).any()):  # the body, or a region of a cut piece
                kept.append(ids)
                continue
            points = self.co[np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))]
            if views.pixels_seen(cast, labels, points.min(axis=0), points.max(axis=0), pixel, index, reach, enough=1):
                kept.append(ids)
            else:
                self.dropped.append(ids)
            yield
        if self.dropped:
            at = {ids.tobytes(): k for k, ids in enumerate(self.pieces)}
            order = [at[ids.tobytes()] for ids in kept]
            self.names = [self.names[k] for k in order]
            self.picture_sizes = [self.picture_sizes[k] for k in order]
            self.notes.append(f"Left out {len(self.dropped):,} hidden loose pieces ({sum(len(ids) for ids in self.dropped):,} faces). No camera outside could see them: anything inside a closed space goes too")
        self.pieces = kept

    def _baked_suits(self) -> bool:
        """Judge After Baking damages hard-surface models and its own score misses that, so it only runs on scans: almost
        no flat edges, as the Auto button tells a scan."""
        tris = np.asarray([(f[0], f[k], f[k + 1]) for f in self.faces for k in range(1, len(f) - 1)], dtype=np.int64)
        flat = detail.mesh_facts(self.co, tris)["flat"]
        suits = flat < BAKED_MOST_FLAT
        if not suits:
            self.notes.append(f"Judge After Baking was not used: {100.0 * flat:.0f}% of the edges are flat, so this isn't a scan, and it damages hard surfaces")
        return suits

    def _split_pieces(self) -> np.ndarray:
        """Find the loose pieces and how big a picture each gets. Returns each piece's share of the work done before it."""
        fa, fb = patch.face_adjacency(self.faces)
        shell = np.unique(patch._labels(len(self.faces), fa, fb), return_inverse=True)[1]
        self._shell = shell
        order = np.argsort(shell, kind="stable")
        starts = np.concatenate(([0], np.flatnonzero(shell[order][1:] != shell[order][:-1]) + 1, [len(order)]))
        shells = [order[a:b] for a, b in zip(starts[:-1], starts[1:])]
        low, high = self.co.min(axis=0), self.co.max(axis=0)
        sizes_f = np.fromiter((len(f) for f in self.faces), dtype=np.int64, count=len(self.faces))
        flat = np.fromiter(chain.from_iterable(self.faces), dtype=np.int64, count=int(sizes_f.sum()))
        centers = np.add.reduceat(self.co[flat], np.concatenate(([0], np.cumsum(sizes_f)[:-1])), axis=0) / sizes_f[:, None]
        cut = []  # a piece with more faces than the machine should hold at once is cut into regions
        label = np.full(len(self.faces), -1, dtype=np.int64)  # region number; loose pieces left whole share -1, so one touching another at a corner isn't locked
        # With helpers a big piece is cut too, so they share it: in one piece it would be measured by one core alone.
        limit = min(self.cfg.chunk_faces, SPLIT_FACES) if self.parallel and not self.guide_lines and not self.strips else self.cfg.chunk_faces
        for ids in shells:
            if len(ids) > limit:
                regions = hardware.split_chunks(centers[ids], limit)
                for r in regions:
                    label[ids[r]] = len(cut)
                    cut.append(ids[r])
                if len(ids) > self.cfg.chunk_faces:
                    self.chunks += len(regions)
                else:
                    self.split += len(regions)
            else:
                cut.append(ids)
        cut.sort(key=len, reverse=True)
        self.pieces = cut
        self.names = [hardware.region_name(centers[ids].mean(axis=0), low, high) for ids in cut]
        if self.chunks or self.split:
            self.locked = hardware.border_vertices(self.faces, label, len(self.co))
        extent = float(np.ptp(self.co, axis=0).max()) or 1.0
        sizes = np.asarray([2.0 * 0.55 * float(np.ptp(self.co[np.unique(np.fromiter((v for f in ids.tolist() for v in self.faces[f]), dtype=np.int64))], axis=0).max()) for ids in self.pieces]) / extent
        self.picture_sizes = _picture_sizes(sizes, self.cfg.max_rays, self.max_pixels).tolist()
        weights = np.asarray([sum(len(self.faces[f]) - 2 for f in ids.tolist()) for ids in self.pieces], dtype=np.float64)
        return np.concatenate(([0.0], np.cumsum(weights))) / max(float(weights.sum()), 1.0)

    def _run(self):
        done_share = yield from self._slice(self._split_pieces)
        if self.drop_hidden:
            yield from self._drop_hidden_pieces()
            if self.dropped:
                done_share = np.linspace(0.0, 1.0, len(self.pieces) + 1)
        if self.baked:
            self.baked = yield from self._slice(self._baked_suits)
        picture_sizes = self.picture_sizes
        extent = float(np.ptp(self.co, axis=0).max()) or 1.0
        yield from self._slice(self._show)
        measuring = time.perf_counter()
        self._done_share = done_share
        yield from self._start_pool()
        try:
            for index, ids in enumerate(self.pieces):
                self.progress = (index, len(self.pieces))
                self.fraction = MEASURE_SHARE * float(done_share[index])
                self.stage = f"Part {index + 1} of {len(self.pieces)}: {self.names[index]}" if self.chunks or len(self.pieces) < 40 else f"Looking at piece {index + 1} of {len(self.pieces)}"
                if self._copy_plan is not None:
                    yield from self._pump(index)
                yield from self._measure_piece(ids, picture_sizes[index])
                yield
                if self._stopped:
                    return
                if time.perf_counter() - self.shown_at > self.cfg.show_every:
                    yield from self._slice(self._show)
        finally:
            self._close_pool()  # also when the job is stopped or fails
        self.timings["measure"] = time.perf_counter() - measuring
        self.progress = (len(self.pieces), len(self.pieces))
        self.fraction = MEASURE_SHARE
        self.stage = "Looking at the whole object"
        if self.baked:
            self._bake_tree = BVHTree.FromPolygons(self.co.tolist(), [list(f) for f in self.faces], all_triangles=False, epsilon=0.0)
        low, high = self.co.min(axis=0), self.co.max(axis=0)
        center, half = (low + high) / 2.0, extent * 0.55
        pixel = 2.0 * half / self.max_pixels
        self._mark("measured")
        reference = yield from self._reference(center, half, self.max_pixels, self.co, self.faces)
        reference_shaded = yield from self._slice(lambda: shaded(reference, self.co, self.faces, center, half, self.max_pixels))
        self._view = (center, half, reference, reference_shaded, pixel)
        total_seen = sum(self.seen)
        asked = self.detail / 100.0
        start_share, aim, room = CHORDS_START if self.chords and self.collapse else (1.0, AIM, ROOM)
        allowed = start_share * asked * total_seen
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
                for twin, (rep, _move) in self.twins.items():  # copies stay alike
                    self.picked[twin] = self.picked[rep]
                yield from self._tune(want)
            else:
                self.picked = self._allocate(allowed, PIECE_LIMIT * asked)
            pick = tuple(self.levels[i][k].faces for i, k in enumerate(self.picked))
            if pick in tried and not self.tuned:
                break
            tried.add(pick)
            self.stage = f"Making the faces, try {round_index + 1}"
            co, faces, kinds, from_part = yield from self._slice(lambda: self._combined(False, join=True, firsts=True))
            rco, rfaces, rkinds, rebuilt_part = yield from self._slice(lambda: self._combined(True, firsts=True))  # rebuilt parts are already quads
            input_co, input_faces = co, faces
            if self.guide_lines or self.strips:
                self.stage = f"Cutting the guide curves in, try {round_index + 1}"
                yield
                co, faces, kinds, self.guide_edges = self._guided(co, faces, kinds)
            copied = sum(len(self.sizes[i][self.picked[i]][1]) for i in self.twins) if self.twins else 0
            before_pairing = len(faces) + len(rfaces) + copied
            result = (np.zeros((0, 3)), [], [])
            if faces:
                result = yield from self._make_all(co, faces, kinds)
                if result is None:
                    return
            points, out_faces, out_kinds = result
            owner = np.full(len(points), -1, dtype=np.int64)  # unknown: part 0 being rebuilt must not freeze every paired corner
            if self.twins and out_faces:
                owner = yield from self._slice(lambda: special.vertex_owners(points, out_faces, input_co, input_faces, from_part))
            rebuilt_owner = np.zeros(len(rco), dtype=np.int64)
            for f, part in zip(rfaces, rebuilt_part.tolist()):
                rebuilt_owner[list(f)] = part
            if rfaces:
                out_faces += [tuple(v + len(points) for v in f) for f in rfaces]
            out_kinds += rkinds
            points = np.vstack([points, rco])
            owner = np.concatenate([owner, rebuilt_owner])
            worked = (MeshData(points, out_faces, out_kinds), owner)
            if self.twins:
                full, _back = yield from self._slice(lambda: self._with_copies(worked[0], owner))
                points, out_faces, out_kinds = np.asarray(full.vertices, dtype=np.float64), [tuple(f) for f in full.faces], list(full.face_materials)
            self.fraction += 0.5 * (1.0 - MEASURE_SHARE) / ROUNDS
            picked, tuned = list(self.picked), dict(self.tuned)
            if reducing:
                miss = abs(len(out_faces) - self.target_faces) / self.target_faces
                if best is None or miss < best[0]:
                    best = (miss, points, out_faces, out_kinds, None, None, None, picked, tuned, worked)
                if miss <= CLOSE_ENOUGH:
                    break
                paired_share = len(out_faces) / max(before_pairing, 1)
                continue
            self.stage = f"Checking the result, try {round_index + 1}"
            loss, shading, soft = yield from self._look(points, out_faces, center, half, reference, reference_shaded, pixel)
            fits = loss <= 1.1 * self.detail
            # Prefer anything that fits, then the fewest faces. With nothing fitting yet, the least change.
            over = loss > self.detail
            rank = (not fits, over and self.collapse, len(out_faces) if fits else loss)
            if best is None or rank < best[0]:
                best = (rank, points, out_faces, out_kinds, loss, shading, soft, picked, tuned, worked)
            if self.collapse:
                if room * self.detail <= loss <= self.detail:
                    break
                if round_index + 1 < ROUNDS:
                    yield from self._refine(over, best[7])
                # What the sizes say they lose differs from what the whole result loses, so go by the last pick.
                planned = sum(self.levels[i][k].loss for i, k in enumerate(self.picked))
                allowed = max(planned, 0.01 * asked * total_seen) * min(max(aim * self.detail / max(loss, 0.01), 0.3), 4.0)
            else:
                if abs(loss - self.detail) <= 0.1 * self.detail:
                    break
                allowed *= min(max(self.detail / max(loss, 0.01), 0.3), 4.0)
        _rank, points, out_faces, materials, loss, shading, soft, self.picked, self.tuned, (worked, owner) = best
        if loss is None:  # Reduce To looks at the one it keeps, so the view change can still be judged
            self.stage = "Checking the view"
            self.fraction = 0.95
            loss, shading, soft = yield from self._look(points, out_faces, center, half, reference, reference_shaded, pixel)
        elif self.collapse or (self.chords and self.output_kind == "QUADS"):
            before = worked
            worked, owner, _loss = yield from self._reduce(worked, owner, loss, reference, center, half, pixel, take=self.collapse)
            if worked is not before:
                full, _back = yield from self._slice(lambda: self._with_copies(worked, owner))
                points, materials = np.asarray(full.vertices, dtype=np.float64), list(full.face_materials)
                out_faces = yield from self._slice(lambda: [tuple(face) for face in full.faces])
                loss, shading, soft = yield from self._look(points, out_faces, center, half, reference, reference_shaded, pixel)
            if self.chords and worked is not before:
                # Redrawn and relaxed corners sit on the planes they were given, not on the source, and smooth shading
                # shows that: fit them to the source again and keep it if the shading gets closer within what was asked for.
                self.stage = "Fitting the redrawn quads to the source"
                fitted = yield from self._slice(lambda: self._fit(worked))
                full, _back = yield from self._slice(lambda: self._with_copies(fitted, owner))
                fitted_points = np.asarray(full.vertices, dtype=np.float64)
                fit_loss, fit_shading, fit_soft = yield from self._look(fitted_points, out_faces, center, half, reference, reference_shaded, pixel)
                if fit_loss <= max(self.detail, loss) and fit_shading is not None and (shading is None or fit_shading < shading):
                    worked, points, loss, shading, soft = fitted, fitted_points, fit_loss, fit_shading, fit_soft
        if self.tweak["poles"] > 0 and self.style == "DETAIL" and self.method == "AUTO" and self.output_kind != "TRIS" and not self.chords and not self.guide_lines and not self.strips and len(worked.faces) <= TURN_UP_TO:
            # Rows of quads break wherever a corner has more or fewer than four. Spinning an edge moves no corner, so the shape stays.
            # Balanced splits every quad, which leaves few poles: spinning there gained under a point.
            self.stage = "Spinning edges at poles"
            turned = yield from self._slice(lambda: self._turn_poles(worked, owner))
            if turned is not worked:
                full, _back = yield from self._slice(lambda: self._with_copies(turned, owner))
                turned_points, turned_faces = np.asarray(full.vertices, dtype=np.float64), [tuple(face) for face in full.faces]
                turned_loss, turned_shading, turned_soft = yield from self._look(turned_points, turned_faces, center, half, reference, reference_shaded, pixel)
                if turned_loss <= min(loss + TURN_LOSS, max(loss, 1.1 * self.detail)):
                    worked, points, out_faces, materials = turned, turned_points, turned_faces, list(full.face_materials)
                    loss, shading, soft = turned_loss, turned_shading, turned_soft
        plain_faces = out_faces
        if self.twins and self.mirror is None:
            self.instances = yield from self._slice(lambda: self._instances(worked, owner))
        for level in POLISH_LEVELS if self.tidy and self.output_kind != "TRIS" else ():  # an option: it is slow and the loss goes up a little
            self.stage = "Tidying the quads..."
            self._paired = None
            yield from tasks.run(lambda: self._polish(points, out_faces, materials, level), stopped=lambda: self._stopped)
            if self._stopped:
                return
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
        piece = yield from self._slice(lambda: self._vote(points, out_faces))
        points, out_faces, materials, (loss, shading, soft), check = yield from self._check(points, out_faces, materials, piece, (loss, shading, soft))
        if out_faces is not plain_faces:  # tidied or repaired: copies are no longer alike
            self.instances = None
        self.stage = "Scoring the quads"
        if any(len(face) == 4 for face in out_faces):
            yield from tasks.run(lambda: self._score(points, out_faces), stopped=lambda: self._stopped)
            if self._stopped:
                return
        def last_look():
            out_fa, out_fb = patch.face_adjacency(out_faces)
            self._show((points, out_faces, materials, np.unique(patch._labels(len(out_faces), out_fa, out_fb), return_inverse=True)[1]))

        self.fraction = 1.0
        yield from self._slice(last_look)
        shown = self.output["faces"]
        follows = None
        if self.guide_lines:  # how far, along each stroke, the nearest edge of the result is: mean and worst, as a share of the size
            size = float(np.linalg.norm(np.ptp(self.co, axis=0))) or 1.0
            mean, worst = guides.follow(self.guide_lines, self.output["co"], shown, size / 400.0)
            follows = {"strokes": len(self.guide_lines), "mean": mean / size, "worst": worst / size}
        whole = self.mirror[3] if self.mirror is not None else len(self.faces)
        quads = sum(1 for face in shown if len(face) == 4)
        bend = guides.bend_stats([group for _tag, group in self.bend_groups], self.output["co"], shown, self.gap) if self.bend_groups else None
        self.report = {"before": whole, "faces": len(shown), "quads": quads, "tris": sum(1 for face in shown if len(face) == 3), "pieces": len(self.pieces), "looks": self.looks, "loss": loss, "shading": shading, "shading_soft": soft, "topology": self.topology, "asked": self.target_faces * (2 if self.mirror is not None else 1), "seconds": time.perf_counter() - self.started, "timings": {k: round(v, 1) for k, v in self.timings.items()}, "bend": bend, "dropped": len(self.dropped), "dropped_faces": sum(len(ids) for ids in self.dropped), "baked": self.baked, "copies": len(self.twins), "instanced": self.instances is not None, "check": check, "notes": list(dict.fromkeys(self.notes)), "chunks": self.chunks, "split": self.split, "resumed": self.resumed, "memory_mb": dict(self.mem), "guides": follows}
        if self.checkpoint is not None and not self.keep_progress:
            shutil.rmtree(self.checkpoint, ignore_errors=True)  # finished: nothing left to resume
        self._drop_tmp()

    def _find_issues(self, shown) -> None:
        """Group the faces the checks flag into ranked spots (triadq.issues), reusing the defect masks the check cached. A mirrored or joined result has other faces than were checked: then only the topology kinds are looked for."""
        began = time.perf_counter()
        masks = {}
        if self.found is not None and len(self.found.inward) == len(shown):
            masks = {k: getattr(self.found, k) for k in checks.Defects.KINDS}
        piece = self.output.get("piece")  # the parts Fix Parts lists, so "part 3" there is part 3 here
        parts = piece if piece is not None and len(piece) == len(shown) else self.output["parts"]
        spots, record = issues.analyse(self.output["co"], shown, masks, parts, {"source_faces": len(self.faces)})
        self.issues = spots, record
        self.output["issues"] = record
        self.output["issue_spots"] = spots
        self.timings["issues"] = time.perf_counter() - began

    def _make_all(self, co, faces, kinds):
        """Generator: the faces for the whole object. Pieces on the usual method go in one call; others one by one."""
        parts = (yield from self._slice(lambda: self._combined(False, firsts=True)))[3]
        special = {i for i, m in self.methods.items() if m != self.method} | ({i for i in range(len(self.pieces))} if self.method in BLENDER_OPERATOR_METHODS else set())
        if not special or self.guide_lines or self.strips:  # with strokes the faces have changed: one call
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
        view = yield from self._timed("pictures", self._pictures(points, out_faces, center, half, self.max_pixels))
        wrong, shown = self._counts(reference, view, pixel)
        view_shaded = yield from self._slice(lambda: shaded(view, points, out_faces, center, half, self.max_pixels))
        bad_shade, seen_both = shading_counts(reference, view, reference_shaded, view_shaded)
        soft = shading_counts(reference, view, reference_shaded, view_shaded, SOFT_DEG)[0]
        return 100.0 * wrong / max(shown, 1), 100.0 * bad_shade / max(seen_both, 1), 100.0 * soft / max(seen_both, 1)

    def _tune(self, want: float):
        """Generator. Reduce To: the sizes come in steps of two, so remake the biggest picked piece to land closer on `want`."""
        total = sum(levels[k].faces for levels, k in zip(self.levels, self.picked))
        miss = want - total
        if abs(miss) <= 0.03 * want:
            return
        made = [i for i in range(len(self.picked)) if not self.ready[i][self.picked[i]] and i not in self.twins]  # leave a rebuilt part alone while another can take it
        index = max(made or range(len(self.picked)), key=lambda i: self.levels[i][self.picked[i]].faces)
        points, polys, kinds = yield from self._slice(lambda: self._piece_arrays(self.pieces[index]))
        triangles = _triangle_count(polys)
        goal = self.levels[index][self.picked[index]].faces + miss
        if triangles <= KEEP_TRIANGLES or goal < 12 or goal >= triangles:
            return
        self.tuned[index] = yield from self._reduced(points, polys, kinds, goal / triangles, self._piece_drop(self.pieces[index]), self._piece_lock(self.pieces[index]))

    def explain_piece(self, index: int) -> str:
        """In plain words, for the segment view: what this part was taken for and what was kept. Empty before that."""
        if self.report is None or index >= len(self.picked):
            return ""
        if len(self.levels[index]) == 1:
            return "Too small to reduce, so it was kept as it was."
        if index not in self.part_cases:
            return ""
        case, skipped, looked = self.part_cases[index]
        if index in self.tuned:
            return cases.explain(case, len(self.tuned[index][1]), None, skipped, looked)
        kept = self.levels[index][self.picked[index]]
        whole = self.picked[index] == len(self.levels[index]) - 1
        return cases.explain(case, kept.faces, kept.loss / max(self.seen[index], 1e-30), skipped, looked, whole, False)

    def status(self) -> str:
        extra = f" | {self.note}" if self.note else ""
        if self.report is None:
            return self.stage + extra
        r = self.report
        shade = "" if r.get("shading") is None else f", {r['shading']:.1f}% in smooth shading"
        tidy = "" if r.get("topology") is None else f"Tidiness: {r['topology']:.0f}/100. "
        text = f"{r['faces']:,} faces, {r['tris']} triangles, {r['pieces']:,} pieces | Camera: {r['loss']:.1f}% of the view differs in shape{' once baked' if r.get('baked') else ''}{shade}. {tidy}{r['seconds']:.0f} s{extra}"
        bad = r.get("check") and r["check"].get("bad")
        text += f" | {bad:,} bad face{'s' if bad != 1 else ''} (red under Show Bad Faces; after Apply, Fix Selection repairs most of them)" if bad else ""
        return text + (f" | {r['check']['text']}" if r.get("check") else "") + "".join(f" | {n}" for n in r.get("notes", []))

