# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Blockout and Hulls: each loose part of the mesh becomes a few low-poly solids.

Blockout fits a box, a cylinder, an ellipsoid and a small convex hull to every part (`triadq.primitives`) and keeps
the one the camera finds closest from the part's own six views. A part no single solid fits is cut at its creases by
the segment finder (`triadq.segment`) and each segment gets its own solid. Hulls does the same with reduced convex
hulls only. A part that still looks wrong keeps a reduced copy of itself, so no part comes out wrecked.

Runs in slices from the panel's timer like `autoretopo.AutoJob`, and gives the same `output` and `report`, so the
preview, Apply, Undo, Bake Detail and Make LOD Set treat it like any other Use.
"""

from __future__ import annotations

import math
import time

import bmesh
import bpy
import numpy as np

from . import autoretopo, hardware
from .triadq import patch, segment, views
from .triadq import primitives as P
from .triadq.types import MeshData

PART_PIXELS = 48  # a side of each picture of one part, at least
MOST_PIXELS = 160  # and at most: a thin part gets more, so its thinnest side still spans THIN_PIXELS
THIN_PIXELS = 8.0
DEPTH_SHARE = 0.06  # a part's surface may move this share of its picture's width before a pixel counts as wrong
FITS = 0.35  # a solid that gets more of a part's pixels wrong than this (outline, or bulk moved) doesn't fit it
HULL_HANDICAP = 0.03  # Blockout: a hull must beat the plain solids by this much, as they read more clearly
SPLIT_DETAIL = 0.4  # Segment Detail used to cut a part no single solid fits
SMALLEST_SEGMENT = 0.01  # of a part's faces: a segment smaller than this gets no solid of its own
REDUCE_RATIOS = (0.02, 0.05, 0.1, 0.2, 0.4)  # shares of a part's triangles tried when it keeps a reduced copy of itself
REDUCED_MISS = 0.1  # the first reduced copy this close to the part is kept
SMALL_PART = 0.15  # of the model's size: smaller parts get half the sides


def _hull(points: np.ndarray, budget: int):
    """The convex hull of the points with about `budget` faces, triangles joined into quads: (points, faces) or None."""
    pts = np.asarray(points, dtype=np.float64)
    for _try in range(4):
        bm = bmesh.new()
        try:
            verts = [bm.verts.new(p) for p in pts.tolist()]
            try:
                out = bmesh.ops.convex_hull(bm, input=verts, use_existing_faces=False)
            except (RuntimeError, ValueError):
                return None
            gone = list(dict.fromkeys(g for g in out["geom_interior"] + out["geom_unused"] if isinstance(g, bmesh.types.BMVert)))
            if gone:
                bmesh.ops.delete(bm, geom=gone, context="VERTS")
            if len(bm.faces) < 4:
                return None
            if len(bm.faces) <= 1.5 * budget or len(bm.verts) <= 6:
                bmesh.ops.join_triangles(bm, faces=list(bm.faces), angle_face_threshold=math.radians(40.0), angle_shape_threshold=math.radians(40.0))
                bm.verts.index_update()
                co = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
                faces = [tuple(v.index for v in f.verts) for f in bm.faces]
                return co, P.outward(co, faces, co.mean(axis=0))
            hull_points = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
        finally:
            bm.free()
        pts = hull_points[P.spread_points(hull_points, max(5, int(0.75 * budget) + 2))]
    return None


def _local(co, faces, ids):
    """The faces `ids` as their own mesh: (points, faces with new numbers)."""
    picked = [faces[f] for f in ids.tolist()]
    used, flat = np.unique(np.fromiter((v for f in picked for v in f), dtype=np.int64), return_inverse=True)
    polys, at = [], 0
    for f in picked:
        polys.append(tuple(int(x) for x in flat[at : at + len(f)]))
        at += len(f)
    return co[used], polys


def _joined(meshes):
    points, faces, offset = [], [], 0
    for pts, fs in meshes:
        points.append(pts)
        faces += [tuple(v + offset for v in f) for f in fs]
        offset += len(pts)
    return (np.vstack(points) if points else np.zeros((0, 3))), faces


def split_result(res: dict) -> list[dict]:
    """A result cut into one result per source part (Each Part Its Own Object)."""
    piece = np.asarray(res["piece"])
    out = []
    for part in np.unique(piece).tolist():
        ids = np.flatnonzero(piece == part)
        co, faces = _local(np.asarray(res["co"]), res["faces"], ids)
        out.append({"co": co, "faces": faces, "mat": np.asarray(res["mat"])[ids], "new": np.ones(len(ids), bool)})
    return out


class BlockoutJob:
    """One Blockout or Hulls run. Call `step` from a timer until `busy` is False; `output` holds the latest result."""

    def __init__(self, co, faces, materials, hull: bool = False, sides: int = 12, settings=None):
        self.co = np.asarray(co, dtype=np.float64)
        self.faces = faces
        self.materials = np.asarray(materials, dtype=np.int64)
        self.hull = bool(hull)
        self.sides = max(4, int(sides))
        self.cfg = settings or hardware.EFFORTS["NORMAL"]
        self.busy = True
        self.output = None
        self.output_id = 0
        self.report: dict | None = None
        self.warnings: list[str] = []
        self.note = ""
        self.progress = (0, 0)
        self.fraction = 0.0
        self.stage = "Finding the parts..."
        self.started = time.perf_counter()
        self.shown_at = 0.0
        self.made: list[tuple] = []  # per source part: (points, faces, materials, kind)
        self.kinds: dict[str, int] = {}
        self.kept: list[str] = []  # where the parts are that kept a reduced copy of themselves
        self.misses: list[float | None] = []  # per source part: share of its pixels its solid gets wrong
        self.pieces: list[np.ndarray] = []  # per source part: its faces, for the Fix Parts list
        self.mirror = ()  # not None: Fix Parts offers no redo, as solids have no sizes to pick from (segments.mirrored)
        self.chunks = 0
        self.style = "SOLIDS"
        self._search = self._run()

    # The same driving methods as AutoJob, so the panel runs either.
    def step(self, seconds: float | None = None) -> None:
        if not self.busy:
            return
        ends = time.perf_counter() + (self.cfg.slice_s if seconds is None else seconds)
        try:
            while time.perf_counter() < ends:
                next(self._search)
        except StopIteration:
            self.busy = False
        except Exception as exc:  # a timer has nobody above it to report to
            self.warnings = [f"Blockout failed: {exc}"]
            self.busy = False

    def wait(self, timeout: float | None = None) -> bool:
        ends = None if timeout is None else time.perf_counter() + timeout
        while self.busy and (ends is None or time.perf_counter() < ends):
            self.step(0.2)
        return not self.busy

    def stop(self) -> None:
        self.busy = False
        self._search.close()

    def seconds_left(self) -> float | None:
        if not self.busy or self.fraction < 0.05:
            return None
        spent = time.perf_counter() - self.started
        return spent * (1.0 - self.fraction) / self.fraction

    def status(self) -> str:
        if self.report is None:
            return self.stage
        r = self.report
        made = ", ".join(f"{k}: {n}" for k, n in sorted(r["kinds"].items(), key=lambda kv: -kv[1]))
        text = f"{r['faces']:,} faces from {r['pieces']:,} parts ({made}) | Camera: {r['loss']:.1f}% of the view differs in shape, {r['outline']:.1f}% in outline and bulk"
        if self.kept:
            text += f" | Kept as a reduced copy, no solid fits: {', '.join(dict.fromkeys(self.kept))}"
        return text

    # What the Fix Parts list asks of a job.
    def _piece_arrays(self, ids):
        points, polys = _local(self.co, self.faces, ids)
        return points, polys, [str(int(m)) for m in self.materials[ids].tolist()]

    def explain_piece(self, index: int) -> str:
        if index >= len(self.made):
            return ""
        miss = self.misses[index]
        return f"Made as: {self.made[index][3]}" + ("" if miss is None else f", {100.0 * miss:.0f}% of its outline and bulk differ from the part")

    # The work.
    def _pictures(self, points, faces, center, half, pixels):
        return (yield from autoretopo.pictures(points, faces, center, half, pixels))

    def _candidates(self, points, budget: int, sides: int, box: dict):
        """(kind, points, faces) of every solid tried on these points."""
        out = []
        if not self.hull:
            for fit in P.fits(points, box):
                out.append((fit["kind"], *P.build(fit, sides)))
        hull = _hull(points, budget)
        if hull is not None:
            out.append(("hull", *hull))
        return out

    def _best(self, candidates, reference, frame):
        """Generator: (miss, kind, points, faces) of the candidate the camera likes best."""
        best = None
        center, half, pixels, tol = frame
        for kind, pts, fs in candidates:
            view = yield from self._pictures(pts, fs, center, half, pixels)
            miss = P.view_miss(reference, view, tol)
            score = miss + (HULL_HANDICAP if kind == "hull" and not self.hull else 0.0)
            if best is None or score < best[0]:
                best = (score, miss, kind, pts, fs)
        return None if best is None else best[1:]

    @staticmethod
    def _frame(points, box):
        """(centre, half width, pixels, depth tolerance) of the six pictures of some points, whose box is `box`."""
        low, high = points.min(axis=0), points.max(axis=0)
        half = max(float((high - low).max()) * 0.7, 1e-9)
        pixels = int(np.clip(math.ceil(THIN_PIXELS * half / max(float(box["half"].min()), 1e-12)), PART_PIXELS, MOST_PIXELS))
        return (low + high) / 2.0, half, pixels, DEPTH_SHARE * 2.0 * half

    def _split(self, points, polys, reference, frame, budget, sides):
        """Generator: the part cut at its creases, a solid per segment: (miss of them all, points, faces) or None."""
        if len(polys) < segment.MIN_FACES:
            return None
        self.stage += ", cutting it at its creases"
        labels = segment.find(segment.build(points, polys), polys, SPLIT_DETAIL)
        counts = np.bincount(labels)
        if len(counts) < 2:
            return None
        pieces = []
        for label in np.flatnonzero(counts >= max(SMALLEST_SEGMENT * len(polys), 4)).tolist():
            pts, fs = _local(points, polys, np.flatnonzero(labels == label))
            box = P.fit_box(pts)
            s_frame = self._frame(pts, box)
            s_reference = yield from self._pictures(pts, fs, *s_frame[:3])
            found = yield from self._best(self._candidates(pts, max(8, budget // 2), sides, box), s_reference, s_frame)
            if found is not None:
                pieces.append((found[2], found[3]))
            yield
        if not pieces:
            return None
        pts, fs = _joined(pieces)
        view = yield from self._pictures(pts, fs, *frame[:3])
        return P.view_miss(reference, view, frame[3]), pts, fs

    def _reduced(self, points, polys, kinds, reference, frame):
        """Generator: the part's own faces, reduced as far as still looks like it: (points, faces, materials)."""
        triangles = sum(len(f) - 2 for f in polys)
        made = (points, polys, kinds)
        for ratio in REDUCE_RATIOS:
            if ratio * triangles < 12:
                continue
            pts, fs, ms = autoretopo._decimate(bpy.context, points, polys, kinds, ratio)
            pts, fs, ms = autoretopo.make_pure(MeshData(pts, fs, ms), "AUTO", "MIXED")
            view = yield from self._pictures(pts, fs, *frame[:3])
            made = (pts, fs, ms)
            if P.view_miss(reference, view, frame[3]) <= REDUCED_MISS:
                break
        return made

    def _run(self):
        fa, fb = patch.face_adjacency(self.faces)
        shell = np.unique(patch._labels(len(self.faces), fa, fb), return_inverse=True)[1].reshape(-1)
        order = np.argsort(shell, kind="stable")
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(shell[order])) + 1, [len(order)]))
        parts = sorted((order[a:b] for a, b in zip(bounds[:-1], bounds[1:])), key=len, reverse=True)
        self.pieces = parts
        low, high = self.co.min(axis=0), self.co.max(axis=0)
        extent = float((high - low).max()) or 1.0
        for index, ids in enumerate(parts):
            self.progress = (index, len(parts))
            self.fraction = 0.9 * index / len(parts)
            points, polys = _local(self.co, self.faces, ids)
            name = hardware.region_name(points.mean(axis=0), low, high)
            self.stage = f"Part {index + 1} of {len(parts)}: {name}"
            share = float(np.ptp(points, axis=0).max()) / extent
            sides = self.sides if share >= SMALL_PART else max(min(6, self.sides), self.sides // 2)  # never more than a big part gets
            budget = max(8, int(round(self.sides * 4 * math.sqrt(share)))) if self.hull else 2 * sides
            box = P.fit_box(points)
            frame = self._frame(points, box)
            reference = yield from self._pictures(points, polys, *frame[:3])
            found = yield from self._best(self._candidates(points, budget, sides, box), reference, frame)
            mat = int(np.bincount(self.materials[ids]).argmax())
            if found is None or found[0] > FITS:
                cut = yield from self._split(points, polys, reference, frame, budget, sides)
                if cut is not None and (found is None or cut[0] < found[0]):
                    found = (cut[0], "cut at creases", cut[1], cut[2])
            if found is None or found[0] > FITS:
                kinds = [str(int(m)) for m in self.materials[ids].tolist()]
                pts, fs, ms = yield from self._reduced(points, polys, kinds, reference, frame)
                self.made.append((pts, fs, [int(m) for m in ms], "reduced copy"))
                self.misses.append(None)
                self.kept.append(name)
            else:
                self.made.append((found[2], found[3], [mat] * len(found[3]), found[1]))
                self.misses.append(found[0])
            yield
            if time.perf_counter() - self.shown_at > self.cfg.show_every:
                self._show()
        self._show()
        yield from self._finish(parts)

    def _show(self) -> None:
        if not self.made:
            return
        points, faces = _joined([(pts, fs) for pts, fs, _ms, _kind in self.made])
        mats = np.fromiter((m for _p, _f, ms, _k in self.made for m in ms), dtype=np.int32, count=len(faces))
        piece = np.repeat(np.arange(len(self.made)), [len(fs) for _p, fs, _m, _k in self.made])
        fa, fb = patch.face_adjacency(faces)
        loose = np.unique(patch._labels(len(faces), fa, fb), return_inverse=True)[1].reshape(-1)
        count = len(faces)
        self.output = {"instances": None, "co": points, "faces": faces, "new": np.ones(count, bool), "locked": np.zeros(count, bool), "mat": mats, "parts": loose, "piece": piece}
        self.output_id += 1
        self.shown_at = time.perf_counter()

    def _finish(self, parts):
        self.stage = "Looking at the whole object"
        self.fraction = 0.9
        low, high = self.co.min(axis=0), self.co.max(axis=0)
        center, half = (low + high) / 2.0, float((high - low).max()) * 0.55 or 1.0
        pixels = self.cfg.max_pixels
        pixel = 2.0 * half / pixels
        reference = yield from self._pictures(self.co, self.faces, center, half, pixels)
        view = yield from self._pictures(self.output["co"], self.output["faces"], center, half, pixels)
        wrong, shown = views.picture_counts(reference, view, pixel)
        out_faces = self.output["faces"]
        kinds: dict[str, int] = {}
        for _p, _f, _m, kind in self.made:
            kinds[kind] = kinds.get(kind, 0) + 1
        self.report = {
            "before": len(self.faces), "faces": len(out_faces), "quads": sum(1 for f in out_faces if len(f) == 4), "tris": sum(1 for f in out_faces if len(f) == 3),
            "pieces": len(parts), "loss": 100.0 * wrong / max(shown, 1), "outline": 100.0 * P.view_miss(reference, view, DEPTH_SHARE * 2.0 * half),
            "shading": None, "topology": None, "asked": 0, "seconds": time.perf_counter() - self.started, "kinds": kinds, "kept": list(self.kept), "check": None, "notes": [],
        }
        self.fraction = 1.0
        self.output_id += 1  # the panel copies the finished result once it sees a new id
