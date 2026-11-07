# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Segments inside a connected mesh: find them, change them by hand, give each its own Use.

Once the search is done, `FindJob` splits every big loose piece along its creases and necks, on your mesh and on the
result. A segment can be merged, split with a drawn line, painted, cut out with a shape and renamed. All of that is saved
on the object: a face attribute holds the segment of each face, a custom property the names and Uses.

`GoalJob` remakes one segment for its Use and leaves the others alone. The seam is never moved: every remade part reuses
the seam's vertices at exactly the same place, so `segments.splice` can weld them back together, and a redo that would
open an edge is refused. That is also why any try of any segment fits with any try of its neighbours.

The numbers shown per segment are the camera check against the same faces of your mesh and the topology score. An
Animation segment is ranked by its topology score, since that is what it is for.
"""

from __future__ import annotations

import json
import time

import bpy
import numpy as np
from mathutils import Vector

from . import autoretopo, flowstroke, guides, loops, overlay, parallel, region
from . import segments as view
from .triadq import parts as P
from .triadq import patch, tasks
from .triadq import segment as S
from .triadq.flatquad import flat_quad_remesh
from .triadq.types import MeshData

ATTRIBUTE = "curiomesh_segment"
SAVED = "curiomesh_segments"
MOST_MEASURED = 60  # segments looked at by the camera after finding them: a model of thousands of parts would take minutes
LOW_POLY_EDGE = 2.2  # Low Poly's quads against the sizes the others use at the same detail
SMALLEST_RESULT = 60  # a found segment with fewer faces in the result joins a neighbour: too few to be worth remaking
WORSE_TIMES, WORSE_BY = 1.5, 4.0  # Reduce may change the view this many times, and this many points, more than the part had before
MATCH_SHARE = 0.6  # a segment's surface on the result may be this share of its surface on your mesh, or its inverse
LOOSE_TIMES, LOOSE_BY = 3.0, 15.0  # the other Uses change more on purpose, but never this much: past it the part has lost its shape
CLOSED_SHARE, STROKE_SHARE = 1.3, 0.8  # triangles to reduce to, per quad wanted, with closure and with drawn lines
HISTORY = 20  # hand edits Undo Edit can go back through
FIELD_MOST = 300_000  # faces of your mesh above which a part is too slow for the field mesher
SHARPEN_DROP = 8.0  # topology score a sheet may give up for following the camera's misses
GOAL_ITEMS = [
    ("KEEP", "Keep as Is", "Your original faces, unchanged except for the row along the seam, which joins the neighbours", "FREEZE", 0),
    ("REDUCE", "Reduce", "Fewer faces where the surface allows it: quads that keep the shape, with no regard for rows", "MOD_DECIM", 1),
    ("LOWPOLY", "Low Poly", "Far fewer faces. The seam keeps its vertices, so go lower on the whole mesh first for a coarser seam", "MESH_ICOSPHERE", 2),
    ("ANIMATION", "Animation", "A clean quad sheet: even rows and few poles, so it bends well. Judged mainly on its topology score", "ARMATURE_DATA", 3),
    ("SUBD", "Subdivision", "All quads of an even size that follow the surface, ready for a Subdivision modifier", "MOD_SUBSURF", 4),
    ("TRIS", "Tris", "Reduced triangles, not paired. About twice the faces of the same shape in quads", "MESH_DATA", 5),
]
GOAL_NAMES = {item[0]: item[1] for item in GOAL_ITEMS}


def _g():
    return view._g()


# --------------------------------------------------------------------------- saved with the object


def save() -> None:
    """Write the segments on the object, where they survive saving the .blend."""
    guided = _g()
    source, src = guided.state_source(), guided.state["src"]
    labels = view.seg.get("src")
    if source is None or src is None or labels is None or len(src.get("keep", ())) != len(labels):
        return
    mesh = source.data
    values = np.zeros(len(mesh.polygons), dtype=np.int32)
    values[src["keep"]] = labels
    attribute = mesh.attributes.get(ATTRIBUTE)
    if attribute is None or attribute.domain != "FACE" or attribute.data_type != "INT":
        if attribute is not None:
            mesh.attributes.remove(attribute)
        attribute = mesh.attributes.new(ATTRIBUTE, "INT", "FACE")
    attribute.data.foreach_set("value", values)
    source[SAVED] = json.dumps({
        "names": [r["name"] for r in view.seg["parts"]],
        "goals": [r["goal"] for r in view.seg["parts"]],
        "details": [r["goal_detail"] for r in view.seg["parts"]],
        "methods": [r.get("choice", "AUTO") for r in view.seg["parts"]],
        "multires": [r.get("multires", True) for r in view.seg["parts"]],
        "locked": [bool(r.get("locked")) for r in view.seg["parts"]],
        "sel": sorted(view.seg["sel"]),
    })


def load(source, src, count: int):
    """(labels per face, what was saved) from the object, or (None, None) if it has none that fit."""
    mesh = source.data
    attribute = mesh.attributes.get(ATTRIBUTE)
    text = source.get(SAVED)
    if attribute is None or attribute.domain != "FACE" or attribute.data_type != "INT" or not text or len(src.get("keep", ())) != count:
        return None, None
    values = np.zeros(len(mesh.polygons), dtype=np.int32)
    attribute.data.foreach_get("value", values)
    try:
        data = json.loads(text)
    except ValueError:
        return None, None
    labels = values[src["keep"]].astype(np.int64)
    if len(data.get("names", ())) != int(labels.max()) + 1:
        return None, None
    return labels, data


# --------------------------------------------------------------------------- the table


def _loose_of(job) -> np.ndarray:
    if not hasattr(job, "_face_loose"):
        job._face_loose = np.zeros(len(job.faces), dtype=np.int64)
        for p, ids in enumerate(job.pieces):
            job._face_loose[ids] = p
    return job._face_loose


def _record(job, label: int, ids: np.ndarray, count: int, saved=None) -> dict:
    points, polys, _kinds = job._piece_arrays(ids)
    loose = int(_loose_of(job)[ids[0]])
    whole = len(ids) == len(job.pieces[loose])
    change = None
    if whole:
        try:
            level = job.levels[loose][job.picked[loose]]
            change = 100.0 * level.loss / job.seen[loose] if job.seen[loose] > 0 else 0.0
        except (IndexError, AttributeError):
            pass
    saved = saved or {}
    pick = lambda key, default: saved[key][label] if label < len(saved.get(key, ())) else default  # noqa: E731
    return {
        "id": label, "name": pick("names", f"Part {label + 1}"), "kind": P.classify(points, polys), "before": len(ids), "after": count,
        "change": change, "topology": None, "method": "AUTO", "goal": pick("goals", "REDUCE"), "goal_detail": pick("details", 0.5), "multires": pick("multires", True),
        "loose": loose, "whole": whole, "source": ids, "choice": pick("methods", "AUTO"), "locked": bool(pick("locked", False)),
        "tries": [{"method": "AUTO", "after": count, "change": change, "topology": None, "detail": None, "build": None}], "cur": 0,
    }


def _groups(labels: np.ndarray, count: int) -> list[np.ndarray]:
    order = np.argsort(labels, kind="stable")
    return np.split(order, np.searchsorted(labels[order], np.arange(1, count)))


def install(base: dict, src_labels: np.ndarray, out: dict, saved=None) -> None:
    """These segments become the table, and `out` the result on screen."""
    job = base["job"]
    count = int(src_labels.max()) + 1
    made = np.bincount(np.asarray(out["piece"]), minlength=count)
    records = [_record(job, k, ids, int(made[k]), saved) for k, ids in enumerate(_groups(src_labels, count))]
    base["out"] = out
    base["id"] = time.perf_counter_ns()
    base.pop("area", None)
    job._face_piece = src_labels
    view.seg.update(base=base["id"], piece=np.asarray(out["piece"]), parts=records, src=src_labels, ref={}, batch_key=None, tris=None, hover=None, surface=True, live=None, live_count=None)
    view.seg["sel"] = {p for p in (saved or {}).get("sel", []) if p < count}
    view.refresh_list()
    _g().state["res"] = out
    _g().state["draw_key"] = None
    overlay.tag_redraw()


def _thickness(bvh, size: float):
    """Local thickness, shot inwards from the middle of a patch (Shapira's shape diameter, one ray per patch)."""

    def measure(points, normals):
        out = np.zeros(len(points))
        for k, (p, n) in enumerate(zip(points.tolist(), normals.tolist())):
            start = Vector((p[0] - n[0] * 1e-4 * size, p[1] - n[1] * 1e-4 * size, p[2] - n[2] * 1e-4 * size))
            hit = bvh.ray_cast(start, Vector((-n[0], -n[1], -n[2])), size * 4.0)
            out[k] = hit[3] if hit[0] is not None else 0.0
        return out

    return measure


def _drop_labels(labels, gone, fa, fb, avoid=()) -> np.ndarray:
    """Faces of labels that can't stay join the neighbour they touch most, never one in `avoid` (a locked segment keeps its faces)."""
    labels = np.asarray(labels).copy()
    avoid = np.asarray(sorted(avoid), dtype=np.int64)
    for value in gone:
        ids = labels == value
        crossing = ids[fa] ^ ids[fb]
        if not crossing.any():
            continue
        other = np.where(ids[fa[crossing]], labels[fb[crossing]], labels[fa[crossing]])
        other = other[~np.isin(other, avoid)]
        if len(other):
            labels[ids] = np.bincount(other).argmax()
    return labels


def _tiny_segments(job, labels, piece, keep=()) -> list[int]:
    """Segments with few faces in the result, that are only part of a loose piece (a whole loose piece has no neighbour to join). Never one in `keep`."""
    loose = _loose_of(job)
    count = int(labels.max()) + 1
    made = np.bincount(piece, minlength=count)
    pieces = np.zeros(count, dtype=np.int64)
    pieces[labels] = loose
    per_loose = np.bincount(pieces, minlength=len(job.pieces))
    order = np.argsort(made)
    tiny = []
    for k in order.tolist():  # never the last segment of a loose piece
        if made[k] < SMALLEST_RESULT and per_loose[pieces[k]] > 1 and k not in keep:
            tiny.append(k)
            per_loose[pieces[k]] -= 1
    return tiny


def _extents(labels, centres, count):
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    np.minimum.at(low, labels, centres)
    np.maximum.at(high, labels, centres)
    return np.where(np.isfinite(high - low), high - low, 0.0)


def _mismatched(labels, piece, src_areas, res_areas, src_centres, res_centres, share: float = MATCH_SHARE) -> list[int]:
    """Segments whose faces on the result cover far more or far less surface than on your mesh, or reach far less or
    further along an axis: the result is too coarse there to hold them."""
    count = int(max(labels.max(), piece.max())) + 1
    on_src = np.bincount(labels, src_areas, count)
    on_res = np.bincount(piece, res_areas, count)
    ratio = on_res / np.maximum(on_src, 1e-30)
    bad = (ratio < share) | (ratio > 1.0 / share)
    src_ext, res_ext = _extents(labels, src_centres, count), _extents(piece, res_centres, count)
    real = src_ext > 0.1 * np.linalg.norm(src_ext, axis=1, keepdims=True)
    bad |= (real & ((res_ext < share * src_ext) | (res_ext > src_ext / share))).any(axis=1)
    return np.flatnonzero((on_src > 0) & bad).tolist()


_is_closed, _on_lines, _lay_edges = flowstroke.is_closed, flowstroke.on_lines, flowstroke.lay_edges  # shared with the Flow Editor


def _misses(reference, view, center, half, pixels) -> np.ndarray:
    """Where on your mesh the camera saw a difference: 3D points, from the same pictures `picture_counts` compares."""
    limit = np.cos(np.radians(autoretopo.TURNED_DEG))
    pixel = 2.0 * half / pixels
    grid = ((np.arange(pixels) + 0.5) / pixels * 2.0 - 1.0) * half
    out = []
    side = 0
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (1.0, -1.0):
            hit_a, normal_a, depth_a, _face_a = reference[side]
            hit_b, normal_b, depth_b, _face_b = view[side]
            side += 1
            both = hit_a & hit_b
            wrong = both & ((np.abs((normal_a * normal_b).sum(axis=2)) < limit) | (np.abs(depth_a - depth_b) > 2.0 * pixel))
            wrong |= hit_a & ~hit_b
            i, j = np.nonzero(wrong)
            if len(i):
                points = np.zeros((len(i), 3))
                points[:, axis] = center[axis] + sign * (2.0 * half - depth_a[i, j])
                points[:, u] = center[u] + grid[i]
                points[:, v] = center[v] + grid[j]
                out.append(points)
    return np.vstack(out) if out else np.zeros((0, 3))


def nearest_faces(src: dict, bvh, co, faces) -> np.ndarray:
    """Per result face: the face of your mesh nearest to its middle."""
    centres = S.face_frames(co, faces)[2]
    face_of_tri = src["tri_face"]
    out = np.zeros(len(faces), dtype=np.int64)
    find = bvh.find_nearest
    for k, p in enumerate(centres.tolist()):
        hit = find(Vector(p))
        if hit[0] is not None:
            out[k] = face_of_tri[hit[2]]
    return out


def transfer(src: dict, bvh, src_labels, co, faces) -> np.ndarray:
    """The segment of each result face: that of the face of your mesh nearest to it."""
    return np.asarray(src_labels)[nearest_faces(src, bvh, co, faces)]


def carried(labels, old, locked) -> dict:
    """{new segment: the locked old one it is}, for new segments made almost wholly of one locked segment's faces."""
    if old is None or not locked or len(old) != len(labels):
        return {}
    labels, old = np.asarray(labels), np.asarray(old)
    out = {}
    for k in sorted(locked):
        held = old == k
        if not held.any():
            continue
        values, counts = np.unique(labels[held], return_counts=True)
        best = int(values[np.argmax(counts)])
        if counts.max() >= 0.9 * held.sum() and (labels == best).sum() <= 1.1 * held.sum():
            out[best] = k
    return out


def _kept_records(count: int, carry: dict, records: list) -> dict:
    """What `_record` reads, for segments found again: the locked ones keep their names, Uses and lock."""
    saved = {key: [] for key in ("names", "goals", "details", "methods", "multires", "locked")}
    for k in range(count):
        old = records[carry[k]] if k in carry and carry[k] < len(records) else None
        saved["names"].append(old["name"] if old else f"Part {k + 1}")
        saved["goals"].append(old["goal"] if old else "REDUCE")
        saved["details"].append(old["goal_detail"] if old else 0.5)
        saved["methods"].append(old.get("choice", "AUTO") if old else "AUTO")
        saved["multires"].append(old.get("multires", True) if old else True)
        saved["locked"].append(old is not None)
    return saved


class FindJob(view.PartJob):
    """Find the segments on a thread (plain NumPy and read-only ray casts), show them, then measure them in slices."""

    is_find = True

    def __init__(self, context, fresh: bool = False):
        self.fresh = fresh
        super().__init__(context, [], "AUTO")
        self.stage = "Finding the segments"
        self.report = self.base["report"]  # finding them changes no face, so the search's numbers still hold

    def __getattr__(self, name):
        """Whatever else is asked of the job on screen is the search's to answer: finding segments only adds labels."""
        return getattr(self.__dict__["job"], name)

    def status(self) -> str:
        if self.warnings:
            return self.warnings[0]
        if self.busy:
            return self.stage
        return self.base.get("summary", "") or f"{len(view.seg['parts']):,} segments"

    def _work(self, box, src, bvh, source):
        job = self.job
        cancelled = lambda: self._stopped  # noqa: E731
        tree = view.seg.get("tree")
        st = box["settings"]
        labels, saved = (None, None) if self.fresh else load(source, src, len(job.faces))
        if tree is None or view.seg.get("tree_job") is not job:
            tree = S.build(job.co, job.faces, _thickness(bvh, float(np.ptp(job.co, axis=0).max()) or 1.0), cancelled=cancelled, creases=st["creases"])
            box["tree"] = tree
        elif abs(tree.creases - st["creases"]) > 1e-9:
            tree = box["tree"] = S.remerge(tree, st["creases"], cancelled)
        locked = box["locked"] if labels is None else set()
        if labels is None:
            self.fraction = 0.4
            scale = S.patch_scale(tree, box["paint"]) if box["paint"] is not None else None
            labels = S.find(tree, job.faces, st["detail"], cancelled, scale, st["largest"], st["smallest"], st["smooth"])
            labels = S.keep_locked(labels, box["old"], locked, tree.links[0], tree.links[1])
        out = self.base["out"]
        co, faces = out["co"], out["faces"]
        self.fraction, self.stage = 0.5, "Marking the segments on the result"
        near = nearest_faces(src, bvh, co, faces)
        box["res_src"] = near
        piece = labels[near]
        normals, areas, centres = S.face_frames(co, faces)
        links = S.face_links(faces)
        raw = piece
        piece = S.tidy(faces, S.even_borders(faces, piece, normals, areas, rounds=1), links)
        carry = carried(labels, box["old"], locked)
        if carry:  # a locked segment keeps its faces on the result too: the nearest-face map or tidying could take a thin one away
            held = list(carry)
            old_piece = box["old_piece"]
            if old_piece is not None and len(old_piece) == len(raw):  # the same result: its faces as they were
                back = np.full(int(max(old_piece.max(), max(carry.values()))) + 1, -1, dtype=np.int64)
                for new, before in carry.items():
                    back[before] = new
                mapped = back[old_piece]
                raw = np.where(mapped >= 0, mapped, np.where(np.isin(raw, held), -1, raw))
                gone = raw < 0  # was not locked, but the map gave it a locked segment: tidying decides
                raw[gone] = piece[gone]
            fix = np.isin(raw, held) | np.isin(piece, held)
            piece[fix] = raw[fix]
        missing = np.setdiff1d(np.unique(labels), np.unique(piece))
        box["debug"] = {"locked": sorted(locked), "carry": dict(carry), "missing": missing.tolist()[:20], "old_piece": box["old_piece"] is not None}
        if len(missing):  # too small to have a face of its own in the result
            labels = _drop_labels(labels, missing.tolist(), tree.links[0], tree.links[1], set(carry))
        values, labels = np.unique(labels, return_inverse=True)
        piece = np.searchsorted(values, piece)
        labels = labels.reshape(-1)
        if saved is None:  # segments found here, not ones you kept, painted or split
            keep = set(carried(labels, box["old"], locked))
            tiny = _tiny_segments(job, labels, piece, keep)
            box["debug"]["tiny"] = list(tiny)[:20]
            if tiny:
                labels = _drop_labels(labels, tiny, tree.links[0], tree.links[1], keep)
                piece = _drop_labels(piece, tiny, links[0], links[1], keep)
                values, labels = np.unique(labels, return_inverse=True)
                labels = labels.reshape(-1)
                piece = np.searchsorted(values, piece)
            for _round in range(4):  # a segment finer than the result's faces gets result faces that are not it: it joins a neighbour
                keep = set(carried(labels, box["old"], locked))
                off = [k for k in _mismatched(labels, piece, tree.areas, areas, tree.centroids, centres) if k not in keep]
                if not off:
                    break
                labels = _drop_labels(labels, off, tree.links[0], tree.links[1], keep)
                piece = _drop_labels(piece, off, links[0], links[1], keep)
                values, labels = np.unique(labels, return_inverse=True)
                labels = labels.reshape(-1)
                piece = np.searchsorted(values, piece)
        moved = S.settle_seams(co, faces, piece, S.seam_samples(job.co, tree.links, labels))
        project = region._projector(bvh)
        moved = S.straighten_seams(moved, faces, piece, project)
        moved = moved.astype(np.float32).astype(np.float64)  # what the preview mesh will hold, so comparing it with this later finds no false change
        if saved is None and locked:
            saved = _kept_records(int(labels.max()) + 1, carried(labels, box["old"], locked), box["records"])
        box.update(labels=labels, piece=piece, co=moved, saved=saved)

    def _run(self):
        guided = _g()
        source = guided.state_source()
        job = self.job
        if source is None:
            return
        src = guided._source_data(source)
        if len(job.faces) != len(src["faces"]):
            return  # the search ran on something other than your mesh's faces: only loose pieces can be told apart then
        bvh = guided._bvh(src)
        old = view.seg.get("src")
        old = old if old is not None and len(old) == len(job.faces) else None
        records = list(view.seg["parts"]) if old is not None else []
        settings = view.seglive.snapshot(source)
        old_piece = view.seg.get("piece") if old is not None and view.ready() else None  # ready: it is the result on screen, face for face
        box = {"old_piece": None if old_piece is None else np.asarray(old_piece).copy(), "settings": settings, "paint": view.seglive.paint_for(source, src) if settings["paint"] else None, "old": old, "records": records, "locked": {r["id"] for r in records if r.get("locked")}}

        def work():
            try:
                self._work(box, src, bvh, source)
            except Exception as exc:  # a thread has nobody above it to report to
                box["error"] = str(exc)[:100]

        yield from tasks.run(work, "CurioMeshSegments", stopped=lambda: self._stopped)
        if self._stopped:
            return
        if "tree" in box:
            view.seg.update(tree=box["tree"], tree_job=job)
        if "labels" not in box:
            if box.get("error") != "cancelled":
                self.warnings = [f"Finding segments failed: {box.get('error', 'no result')}"]
                view.seglive.clear()  # the kept segments are what holds now
            return
        old = self.base["out"]
        out = dict(old, co=box["co"], piece=box["piece"], parts=box["piece"])
        self.debug = box.get("debug")
        install(self.base, box["labels"], out, box["saved"])
        view.seg["res_src"] = (out["faces"], box["res_src"])
        self.output = out
        self.output_id += 1
        self.committed = True
        save()
        records = view.seg["parts"]
        order = sorted(range(len(records)), key=lambda p: -records[p]["after"])[:MOST_MEASURED]
        for n, p in enumerate(order):
            self.progress = (n, len(order))
            self.fraction = 0.6 + 0.4 * n / max(len(order), 1)
            self.stage = f"Checking {records[p]['name']}"
            yield
            if self._stopped:
                return
            record = records[p]
            points, faces, _mat = P.cut(out["co"], out["faces"], out["mat"], out["piece"], p)
            record["topology"] = record["tries"][0]["topology"] = S.topology_score(points, faces)
            if record["change"] is None:
                record["change"] = yield from self._change(p, points, faces)
                record["tries"][0]["change"] = record["change"]
        view.refresh_list()
        self.fraction = 1.0


def start_find(context, fresh: bool = False) -> bool:
    guided = _g()
    base = guided.state["auto_base"]
    if base is None or view.mirrored():  # a mirrored or Clean Rows result has no sizes or one-sided seams to work from
        return False
    work = guided.state["work"]
    if not parallel.find_may_start(work and work["runner"]):
        return False
    guided.stop_work()
    runner = FindJob(context, fresh)
    guided.state["work"] = {"key": base.get("key", ("", "AUTO")), "runner": runner, "cent": None, "seen": 0, "areas": []}
    if not bpy.app.timers.is_registered(guided._auto_slice):
        bpy.app.timers.register(guided._auto_slice, first_interval=0.0)
    return True


# --------------------------------------------------------------------------- remaking a segment for its Use


def in_thread(call):
    """Generator: run `call` on a thread, yielding while it works. Returns {"out": result} or {"error": text}."""
    box = {}

    def work():
        try:
            box["out"] = call()
        except Exception as exc:  # a thread has nobody above it to report to
            box["error"] = str(exc)[:100]

    yield from tasks.run(work, "CurioMeshSegment")
    return box


class GoalJob(view.PartJob):
    """Remake the chosen segments for their Use, one at a time, then look at the whole object again."""

    def __init__(self, context, pieces):
        super().__init__(context, pieces, "AUTO")
        self.opened = 0
        self.direction = 0  # Bigger Quads (-1) keeps a remake only with fewer faces, Smaller Quads (+1) only with more

    def _in_thread(self, call):
        """Generator: run `call` on a thread. Returns its result, or None after noting what went wrong."""
        box = yield from in_thread(call)
        if "out" not in box and box.get("error") != "cancelled":
            self.warnings = [box.get("error", "no result")]
        return box.get("out")

    def _strokes(self, record: dict) -> list:
        """(stroke, loops, tag) for the drawn edge loops and deformation lines that run over this segment."""
        guided = _g()
        gs = self.context.scene.curiomesh_guided
        src = guided._source_data(guided.state_source())
        bvh = guided._bvh(src)
        labels = view.seg["src"]
        if labels is None or not hasattr(gs, "loop_count"):
            return []
        found = []
        for kind, count in (("LOOP", gs.loop_count), ("BEND", gs.bend_loops)):
            for line in guided._guide_lines(self.context, gs, src, kind):
                nearest = [bvh.find_nearest(Vector(p))[2] for p in line[:: max(len(line) // 16, 1)].tolist()]
                if np.mean([labels[src["tri_face"][t]] == record["id"] for t in nearest]) > 0.5:
                    found.append((line, count, kind))
        return found

    def _cut_strokes(self, points, polys, kinds, items, nearest):
        """Drawn lines go into the reduced triangles as strips of quads, the way Auto Segments does. Returns (points, triangles, kinds, edges to keep)."""
        gs = self.context.scene.curiomesh_guided
        tris, kind = [], []
        for f, k in zip(polys, kinds):
            for c in range(1, len(f) - 1):
                tris.append((f[0], f[c], f[c + 1]))
                kind.append(int(k) if str(k).isdigit() else 0)
        tris, kind = np.asarray(tris, dtype=np.int64), np.asarray(kind, dtype=np.int64)
        spacing = float(np.median(np.linalg.norm(points[tris[:, 0]] - points[tris[:, 1]], axis=1)))
        gap = gs.loop_gap if gs.loop_gap > 0 else 0.4 * spacing  # the triangles here are coarser than the quads wanted
        fine = min(loops.room(line, count, gap) for line, count, _tag in items)
        samples = np.vstack([guides.resample(line, fine / 4.0) for line, _count, _tag in items])
        co, tris, kind = guides.refine(points, tris, kind, samples, fine / 2.5, lambda p: nearest(p)[0])
        co, tris, kind, keys, _built, skipped = loops.insert(co, tris, kind, items, gap, nearest, set())
        if skipped:
            self.note = f"{len(skipped)} drawn line(s) left out: " + ", ".join(sorted(set(skipped)))
        return co, [tuple(t) for t in tris.tolist()], [str(k) for k in kind.tolist()], tuple(sorted(keys))

    def _creases(self, record: dict, goal: str):
        """Long creases of this part on your mesh, for an Animation sheet to keep."""
        if goal != "ANIMATION":
            return None
        if "creases" not in record:
            points, polys, _kinds = self.job._piece_arrays(record["source"])
            record["creases"] = S.long_creases(points, polys)
        return record["creases"]

    def _field(self, record: dict, ids, edge: float, items, project):
        """Generator: the part in rows from the field mesher, on its seams. None when that can't be done, for the others to try."""
        out = self.base["out"]
        if S.fieldmesh is None or len(record["source"]) > FIELD_MOST:
            return None
        points, polys, kinds = self.job._piece_arrays(record["source"])
        area = float(S.face_frames(points, polys)[1].sum())
        target = area / max(edge * edge, 1e-30)
        try:
            seams = S.seam_loops(out["co"], out["faces"], ids)
        except ValueError:
            return None
        rings = [line for line, _count, tag in items if tag != "BEND" and _is_closed(line)]
        drawn = [line for line in self._expand_lines(items, edge, project) if not any(line is ring for ring in rings)]
        marks = (lambda pts, tris: _on_lines(pts, tris, drawn)) if drawn else None
        settle = (lambda co, faces, pinned: _lay_edges(co, faces, pinned, drawn, project)) if drawn else None
        most = int(np.bincount(np.asarray(out["mat"])[ids]).argmax()) if len(ids) else 0
        made = yield from self._in_thread(lambda: S.field_with_rings(points, polys, rings, seams, target, project, marks, settle))
        if made is None and rings:  # a loop that can't be a true one still gets edges along it
            drawn = self._expand_lines(items, edge, project)
            marks = lambda pts, tris: _on_lines(pts, tris, drawn)  # noqa: E731
            settle = lambda co, faces, pinned: _lay_edges(co, faces, pinned, drawn, project)  # noqa: E731
            made = yield from self._in_thread(lambda: S.field_on_seams(points, polys, seams, target, project, marks, settle))
        if made is None:
            return None
        return made[0], made[1], [most] * len(made[1])

    def _expand_lines(self, items, edge: float, project) -> list:
        """Strokes as polylines. A deformation line becomes its parallel loops, spaced like the quads."""
        if not items:
            return []
        guided = _g()
        bvh = guided._bvh(guided._source_data(guided.state_source()))
        nearest = lambda p: bvh.find_nearest(Vector(p))[:2]  # noqa: E731
        out = []
        for line, count, tag in items:
            if tag == "BEND":
                for group in guides.parallel([line], count, edge, nearest):
                    out += group
            else:
                out.append(line)
        return out

    def _build(self, goal: str, record: dict, ids, edge: float, topology, project, focus=None):
        """Generator: (points, faces, materials) of this segment for this Use, or None."""
        out = self.base["out"]
        co, faces = out["co"], out["faces"]
        cancelled = lambda: self._stopped  # noqa: E731
        most = np.bincount(np.asarray(out["mat"])[ids]).argmax() if len(ids) else 0
        items = self._strokes(record) if goal in {"ANIMATION", "SUBD"} else []
        self.field_made = False
        if goal == "ANIMATION" or items:  # rows from a direction field, with drawn lines as creases it has to follow
            made = yield from self._field(record, ids, edge, items, project)
            if made is not None:
                self.field_made = True
                return made
            self.warnings = []
        if items:  # lines you drew come first: the rows run along them, and the rest is paired around them
            bvh = _g()._bvh(_g()._source_data(_g().state_source()))
            nearest = lambda p: bvh.find_nearest(Vector(p))[:2]  # noqa: E731
            goal = "STROKES"
        if goal in {"KEEP", "REDUCE", "LOWPOLY", "TRIS", "STROKES"}:
            points, polys, kinds = self.job._piece_arrays(record["source"])
            try:
                seams = S.seam_loops(co, faces, ids)
            except ValueError as exc:
                self.warnings = [str(exc)]
                return None
            if goal != "KEEP":
                triangles = sum(len(p) - 2 for p in polys)
                area = float(S.face_frames(points, polys)[1].sum())
                # Closure splits some faces: about 0.78 quads per triangle come out, not the 0.5 of pure pairing. Drawn lines add their own.
                ratio = (STROKE_SHARE if goal == "STROKES" else CLOSED_SHARE if goal in {"REDUCE", "LOWPOLY"} else 2.0) * area / max(edge * edge, 1e-30) / max(triangles, 1)
                if ratio < 0.98:
                    yield
                    points, polys, kinds = autoretopo._decimate(self.context, points, polys, kinds, max(ratio, 1e-4))
            choice = record.get("choice", "AUTO") if goal in {"REDUCE", "LOWPOLY"} else "AUTO"
            quads = goal in {"REDUCE", "LOWPOLY", "STROKES"} and choice != "TRIS"
            seeds = 4 if choice == "PAIRING" else 2
            if choice == "FLAT":
                def flat_then_join():
                    made = flat_quad_remesh(MeshData(points, polys, kinds), force_quads=True)[0]
                    co2, faces2, origin, _ids = S.conform(np.asarray(made.vertices, dtype=np.float64), [tuple(f) for f in made.faces], seams)
                    return co2, faces2, [list(made.face_materials)[i] for i in origin]

                made = yield from self._in_thread(flat_then_join)
                if made is not None:
                    return made
                self.warnings = []
            keep = ()
            if goal == "STROKES":
                cut = yield from self._in_thread(lambda: self._cut_strokes(points, polys, kinds, items, nearest))
                if cut is not None:
                    points, polys, kinds, keep = cut
            made = yield from self._in_thread(lambda: S.pair_on_seams(points, polys, kinds, seams, quads=quads, guide_edges=keep, project=project, seeds=seeds))
            if made is not None:
                return made
            # Your mesh's border and the result's differ too much for a join here. The seam-keeping fill always fits.
            note = self.warnings[0] if self.warnings else ""
            self.warnings = []
            goal = "SUBD"
            self.note = f"{GOAL_NAMES.get(record['goal'], record['goal'])} could not join its neighbours ({note}): made quads on the seam instead"
        made = yield from self._in_thread(lambda: S.fill_part(co, faces, ids, goal, edge, project, topology, cancelled, focus=focus, creases=self._creases(record, goal)))
        if made is None:
            return None
        points, new_faces, warnings = made
        if warnings:
            self.warnings = warnings[:1]
        return points, new_faces, [most] * len(new_faces)

    def _run(self):
        guided = _g()
        source = guided.state_source()
        out = self.base["out"]
        co, faces, mat, piece = out["co"], list(out["faces"]), np.asarray(out["mat"]), view.seg["piece"]
        project = region._projector(guided._bvh(guided._source_data(source))) if source is not None else None
        topology = patch.region_topology(co, faces)
        total = max(len(self.pieces), 1)
        self.note = ""
        area_of = None
        for n, p in enumerate(self.pieces):
            record = view.seg["parts"][p]
            self.progress = (n, len(self.pieces))
            self.stage = f"Remaking {record['name']} for {GOAL_NAMES.get(record['goal'], record['goal'])}"
            self.fraction = 0.8 * n / total
            yield
            if self._stopped:
                return
            ids = np.flatnonzero(piece == p)
            current = record["tries"][record["cur"]]
            if current["build"] is None:
                current["build"] = P.cut(co, faces, mat, piece, p)
            if current["change"] is None:  # the guard below needs what it has now: a hand-made or unmeasured segment is measured first
                current["change"] = yield from self._change(p, current["build"][0], current["build"][1])
                if record["cur"] == 0 and record["change"] is None:
                    record["change"] = current["change"]
            if "edge" not in record:
                area_of = area_of if area_of is not None else S.face_frames(co, faces)[1]
                record["edge"] = float(np.sqrt(area_of[ids].sum() / max(len(ids), 1)))
            # ponytail: Detail has no end, but quads never get smaller than an eighth of the part's own (64x its faces), so a
            # long run of Smaller Quads can't build billions of faces and run out of memory; raise it if a part needs finer
            edge = record["edge"] * max(2.5 ** (1.0 - 2.0 * record["goal_detail"]), 0.125) * (LOW_POLY_EDGE if record["goal"] == "LOWPOLY" else 1.0)
            made = yield from self._build(record["goal"], record, ids, edge, topology, project)
            if self._stopped:
                return
            if made is None:
                continue
            change = yield from self._change(p, made[0], made[1])
            before = current["change"]
            if record["goal"] == "REDUCE":  # a part this small can lose its shape: more faces, up to twice
                for _again in range(2):
                    if change <= max(WORSE_TIMES * before, before + WORSE_BY):
                        break
                    edge *= 0.7
                    self.stage = f"{record['name']} changed by {change:.0f}%: giving it more faces"
                    more = yield from self._build("REDUCE", record, ids, edge, topology, project)
                    if self._stopped:
                        return
                    if more is None:
                        break
                    tried = yield from self._change(p, more[0], more[1])
                    if tried < change:
                        made, change = more, tried
            elif record["goal"] == "ANIMATION" and not self.field_made and not self._strokes(record):  # second go: more rows where the camera saw the first one go wrong
                self.stage = f"{record['name']}: more rows where the camera found a difference"
                misses = _misses(*self.last_look)
                if len(self._creases(record, "ANIMATION")):
                    misses = np.vstack((misses, self._creases(record, "ANIMATION")))
                sharper = yield from self._build("ANIMATION", record, ids, edge, topology, project, misses)
                if self._stopped:
                    return
                if sharper is not None:
                    tried = yield from self._change(p, sharper[0], sharper[1])
                    old, new = S.topology_score(made[0], made[1]) or 0.0, S.topology_score(sharper[0], sharper[1]) or 0.0
                    if tried < change and new >= old - SHARPEN_DROP:
                        made, change = sharper, tried
            times, more = (WORSE_TIMES, WORSE_BY) if record["goal"] in {"REDUCE", "TRIS"} else (LOOSE_TIMES, LOOSE_BY)
            src_points, src_polys, _kinds = self.job._piece_arrays(record["source"])
            if not S.view_kept(before, change, times, more) or not S.shape_kept(src_points, src_polys, made[0], made[1], 0.4 if record["goal"] == "LOWPOLY" else 0.7):
                self.warnings = [f"{record['name']}: {GOAL_NAMES.get(record['goal'], record['goal'])} would lose its shape ({change:.0f}% of its view changes, it had {before:.0f}%), so it stays as it was. Try more Detail"]
                continue
            if self.direction and (len(made[1]) - len(current["build"][1])) * self.direction <= 0:
                self.warnings = [f"{record['name']}: no {'bigger' if self.direction < 0 else 'smaller'} quads that keep its shape here, so it stays as it was"]
                continue
            trial = view.splice(co, faces, mat, piece, {p: (made[0], made[1], made[2])})[1]
            if S.open_edges(trial) > S.open_edges(faces):  # never hand back a mesh with a new hole, whatever went wrong above
                self.warnings = [f"{record['name']}: the new faces did not close on the seam, so it stays as it was"]
                continue
            self.new_tries[p] = {
                "method": record["goal"], "detail": round(record["goal_detail"], 3), "choice": _choice_of(record["goal"], record.get("choice", "AUTO")), "after": len(made[1]), "change": change, "topology": S.topology_score(made[0], made[1]),
                "build": (made[0], made[1], np.asarray([int(m) if str(m).lstrip("-").isdigit() else 0 for m in made[2]], dtype=np.int32)),
            }
        self.progress = (len(self.pieces), len(self.pieces))
        self.stage = "Looking at the whole object"
        self.fraction = 0.85
        if self.new_tries:
            co, faces, mat, piece = view.splice(co, faces, mat, piece, {p: t["build"] for p, t in self.new_tries.items()})
        loss = yield from self._whole_loss(co, faces)
        count = len(faces)
        self.output = {"co": co, "faces": faces, "new": np.ones(count, bool), "locked": np.zeros(count, bool), "mat": np.asarray(mat, dtype=np.int32), "parts": piece, "piece": piece}
        self.output_id += 1
        self.report = dict(self.base["report"], faces=count, quads=sum(1 for f in faces if len(f) == 4), tris=sum(1 for f in faces if len(f) == 3), loss=loss, seconds=time.perf_counter() - self.started)
        if self.note and not self.warnings:
            self.warnings = [self.note]
        self.fraction = 1.0


def start_goal(context, pieces, name: str = "Redo Part") -> bool:
    from . import history

    guided = _g()
    if guided.state["auto_base"] is None or not pieces:
        return False
    history.before(context)  # the result before the redo is the step Ctrl+Z goes back to
    guided.stop_work()
    runner = GoalJob(context, pieces)
    runner.step_name = name  # the History entry when it finishes
    guided.state["work"] = {"key": guided.state["auto_base"].get("key", ("", "AUTO")), "runner": runner, "cent": None, "seen": 0, "areas": []}
    guided.state["message"] = runner.status()
    if not bpy.app.timers.is_registered(guided._auto_slice):
        bpy.app.timers.register(guided._auto_slice, first_interval=0.0)
    return True


# --------------------------------------------------------------------------- changing segments by hand


def _fresh_record(job, label, ids, count, like=None) -> dict:
    record = _record(job, label, ids, count)
    if like is not None:
        record.update(goal=like["goal"], goal_detail=like["goal_detail"], choice=like.get("choice", "AUTO"))
    return record


def relabel(src_labels, piece, touched, lock=(), store: bool = True) -> str:
    """Hand edits go in: `src_labels` and `piece` hold the new segments, and old segments in `touched` lose their tries.
    Segments in `lock` (by their label before renumbering) are locked, so a size change keeps them. Returns a problem, or ''."""
    guided = _g()
    base = guided.state["auto_base"]
    out, job = base["out"], base["job"]
    links = S.face_links(out["faces"])
    piece = S.one_piece_each(piece, links[0], links[1], smallest=1)
    src_links = view.seg["tree"].links if view.seg.get("tree") is not None else S.face_links(job.faces)
    src_labels = S.one_piece_each(src_labels, src_links[0], src_links[1], smallest=1)
    only_source = np.setdiff1d(np.unique(src_labels), np.unique(piece))
    # A loose piece the result left out has no faces there and no neighbour to join: it was like that before this edit.
    only_source = np.setdiff1d(only_source, np.setdiff1d(np.unique(view.seg["src"]), np.unique(view.seg["piece"])))
    if len(only_source):
        return "Too small for the result: it has no faces of its own at this density"
    only_result = np.setdiff1d(np.unique(piece), np.unique(src_labels))
    if len(only_result):
        return "Too small for your mesh: it has no faces of its own there"
    values = np.unique(src_labels)
    old = {r["id"]: r for r in view.seg["parts"]}
    new_src, new_piece = np.searchsorted(values, src_labels), np.searchsorted(values, piece)
    made = np.bincount(new_piece, minlength=len(values))
    records = []
    for k, ids in enumerate(_groups(new_src, len(values))):
        label = int(values[k])
        record = old.get(label)
        if record is None or label in touched:
            record = _fresh_record(job, k, ids, int(made[k]), record)
            if label in old:
                record["name"] = old[label]["name"]
                record["locked"] = old[label].get("locked", False)
        else:
            record["id"] = k
        if label in lock:
            record["locked"] = True
        records.append(record)
    selected = {int(np.searchsorted(values, p)) for p in view.seg["sel"] if p in values}
    if store:  # Undo Edit goes back to this
        history = view.seg.setdefault("history", [])
        history.append((base["out"], view.seg["piece"], view.seg["parts"], view.seg["src"], set(view.seg["sel"])))
        del history[:-HISTORY]
    out = dict(out, piece=new_piece, parts=new_piece)
    base.update(out=out, id=time.perf_counter_ns())
    job._face_piece = new_src
    view.seg.update(base=base["id"], piece=new_piece, parts=records, src=new_src, sel=selected, ref={}, batch_key=None, tris=None, hover=None, surface=True, live=None, live_count=None)
    guided.state["res"] = out
    view.refresh_list()
    if store:
        save()
    overlay.tag_redraw()
    return ""


def undo_edit() -> str:
    """Put the segments back as they were before the last hand edit."""
    guided = _g()
    base = guided.state["auto_base"]
    history = view.seg.get("history") or []
    while history and (base is None or history[-1][0]["faces"] is not base["out"]["faces"]):
        history.pop()  # made for a result that has been remade since
    if not history:
        return "No hand edit to undo"
    out, piece, parts, src, sel = history.pop()
    if out["co"] is not base["out"]["co"]:  # a find since then laid the seams again: show the points Apply will write
        guided._build_preview(bpy.context, guided.state_source(), out, uvs=False)
        guided.state["draw_key"] = None
    base.update(out=out, id=time.perf_counter_ns())
    base["job"]._face_piece = src
    for k, record in enumerate(parts):
        record["id"] = k
    view.seg.update(base=base["id"], piece=piece, parts=parts, src=src, sel=sel, ref={}, batch_key=None, tris=None, hover=None, live=None, live_count=None)
    guided.state["res"] = out
    view.refresh_list()
    save()
    overlay.tag_redraw()
    return ""


def merge_selected() -> str:
    selected = sorted(view.seg["sel"])
    if len(selected) < 2:
        return "Select two or more segments"
    first = selected[0]
    src, piece = view.seg["src"].copy(), view.seg["piece"].copy()
    src[np.isin(src, selected)] = first
    piece[np.isin(piece, selected)] = first
    view.seg["sel"] = {first}
    return relabel(src, piece, set(selected), lock={first})


def new_label() -> int:
    return len(view.seg["parts"])


def split(segment: int, source_side: np.ndarray, result_side: np.ndarray) -> str:
    """Faces of `segment` on the True side of the masks become a new segment."""
    src, piece = view.seg["src"].copy(), view.seg["piece"].copy()
    fresh = new_label()
    moved_src = (src == segment) & source_side
    moved_res = (piece == segment) & result_side
    if not moved_src.any() or moved_src.sum() == (src == segment).sum() or not moved_res.any() or moved_res.sum() == (piece == segment).sum():
        return "That line does not cross the segment"
    src[moved_src] = fresh
    piece[moved_res] = fresh
    view.seg["sel"] = {segment}
    return relabel(src, piece, {segment}, lock={segment, fresh})


def claim(source_mask: np.ndarray, result_mask: np.ndarray) -> str:
    """Faces in the masks become one new segment, taken from whatever held them."""
    if not source_mask.any() or not result_mask.any():
        return "That covers no faces"
    src, piece = view.seg["src"].copy(), view.seg["piece"].copy()
    fresh = new_label()
    touched = set(np.unique(piece[result_mask]).tolist()) | set(np.unique(src[source_mask]).tolist())
    old_src, old_piece = view.seg["src"], view.seg["piece"]
    src[source_mask] = fresh
    piece[result_mask] = fresh
    problem = relabel(src, _follow(src, old_src, old_piece, piece), touched, lock={fresh})
    if not problem:
        view.select([len(view.seg["parts"]) - 1])
    return problem


def _follow(src, old_src, old_piece, result_piece):
    """The result's segments after a brush: where the face of your mesh nearest a result face changed segment, the
    result face follows it, so the segment covers the same surface on both. Without that map, the brushed result."""
    mapping = view.seglive.res_map()
    if mapping is None:
        return result_piece
    piece = old_piece.copy()
    moved = src[mapping] != old_src[mapping]
    piece[moved] = src[mapping][moved]
    return piece


def grow(segment: int, source_mask: np.ndarray, result_mask: np.ndarray) -> str:
    """Faces in the masks join `segment`, taken from whatever held them."""
    old_src, old_piece = view.seg["src"], view.seg["piece"]
    src, piece = old_src.copy(), old_piece.copy()
    touched = {segment} | set(np.unique(src[source_mask]).tolist()) | set(np.unique(piece[result_mask]).tolist())
    src[source_mask] = segment
    piece[result_mask] = segment
    return relabel(src, _follow(src, old_src, old_piece, piece), touched, lock={segment})


def shrink(segment: int, source_mask: np.ndarray, result_mask: np.ndarray) -> str:
    """Faces of `segment` in the masks go to the neighbours beside them."""
    base = _g().state["auto_base"]
    tree = view.seg.get("tree")
    src_links = tree.links if tree is not None else S.face_links(base["job"].faces)
    res_links = S.face_links(base["out"]["faces"])
    old_src, old_piece = view.seg["src"], view.seg["piece"]
    src = S.shrink_into(old_src, source_mask, segment, src_links[0], src_links[1])
    piece = S.shrink_into(old_piece, result_mask, segment, res_links[0], res_links[1])
    if np.array_equal(src, old_src) and np.array_equal(piece, old_piece):
        return "Nothing to give away there: paint over the segment's border"
    piece = _follow(src, old_src, old_piece, piece)
    touched = {segment} | set(np.unique(src[src != old_src]).tolist()) | set(np.unique(piece[piece != old_piece]).tolist())
    return relabel(src, piece, touched, lock={segment})


def halve(segment: int) -> str:
    """Split a segment where the merge tree last joined it: its own natural border, no line needed."""
    tree = view.seg.get("tree")
    if tree is None:
        return "Find the segments first"
    side = S.split_in_two(tree, view.seg["src"], segment)
    mapping = view.seglive.res_map()
    if side is None or mapping is None:
        return "This segment has no smaller parts inside: draw a line with Split"
    return split(segment, side, side[mapping])


def set_locked(pieces, on: bool) -> None:
    for p in pieces:
        view.seg["parts"][p]["locked"] = on
    save()
    view.seg["batch_key"] = None
    overlay.tag_redraw()


def reload_saved() -> None:
    """After an undo or redo: the segments saved on the mesh at that step come back."""
    guided = _g()
    source, base = guided.state_source(), guided.state["auto_base"]
    if source is None or base is None or not view.ready() or view.seg.get("src") is None:
        return
    view.seglive.clear()
    src = guided._source_data(source)
    labels, data = load(source, src, len(base["job"].faces))
    if labels is None:
        return
    current = view.seg["src"]
    locks = [bool(x) for x in data.get("locked", [])]
    if len(current) == len(labels) and np.array_equal(labels, current):
        if locks != [bool(r.get("locked")) for r in view.seg["parts"]]:
            for record, on in zip(view.seg["parts"], locks):
                record["locked"] = on
            overlay.tag_redraw()
        return
    mapping = view.seglive.res_map()
    if mapping is None:
        return
    differ = labels != current if len(current) == len(labels) else np.ones(len(labels), dtype=bool)
    touched = set(np.unique(labels[differ]).tolist()) | (set(np.unique(current[differ]).tolist()) if len(current) == len(labels) else set())
    if relabel(labels, labels[mapping], touched, store=False):
        return
    for k, record in enumerate(view.seg["parts"]):
        if k < len(data.get("names", ())):
            record["name"] = data["names"][k]
        if k < len(locks):
            record["locked"] = locks[k]


def _centres():
    guided = _g()
    base = guided.state["auto_base"]
    job, out = base["job"], base["out"]
    return S.face_frames(job.co, job.faces)[2], S.face_frames(out["co"], out["faces"])[2]


class _SegmentOp:
    bl_options = {"INTERNAL"}

    def _done(self, problem: str):
        if problem:
            self.report({"WARNING"}, problem)
            _g().state["message"] = problem
            return {"CANCELLED"}
        return {"FINISHED"}


class CURIOMESH_OT_seg_merge(_SegmentOp, bpy.types.Operator):
    bl_idname = "curiomesh.seg_merge"
    bl_label = "Merge"
    bl_description = "Makes the selected parts one (Shift+click two while reviewing, then M). The merged part is locked, so size changes keep it"
    bl_options = {"INTERNAL", "UNDO"}

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and len(view.seg["sel"]) > 1

    def execute(self, context):
        return self._done(merge_selected())


class CURIOMESH_OT_seg_halve(_SegmentOp, bpy.types.Operator):
    bl_idname = "curiomesh.seg_halve"
    bl_label = "Split in Two"
    bl_description = "Splits the selected part in two along its most natural border. Both halves are locked"
    bl_options = {"INTERNAL", "UNDO"}

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and len(view.seg["sel"]) == 1 and view.seg.get("tree") is not None

    def execute(self, context):
        return self._done(halve(next(iter(view.seg["sel"]))))


class CURIOMESH_OT_seg_undo(_SegmentOp, bpy.types.Operator):
    bl_idname = "curiomesh.seg_undo"
    bl_label = "Undo Edit"
    bl_description = "Puts the parts back as they were before the last merge, split, grow, shrink or paint (Ctrl+Z while reviewing)"

    @classmethod
    def poll(cls, context):
        return view.ready() and bool(view.seg.get("history"))

    def execute(self, context):
        return self._done(undo_edit())


class CURIOMESH_OT_seg_lock(bpy.types.Operator):
    bl_idname = "curiomesh.seg_lock"
    bl_label = "Lock"
    bl_description = "Locks the selected parts, so size changes keep them face for face. Parts you merge, split or paint are locked for you"
    bl_options = {"INTERNAL", "UNDO"}

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and bool(view.seg["sel"])

    def execute(self, context):
        selected = sorted(view.seg["sel"])
        set_locked(selected, not all(view.seg["parts"][p].get("locked") for p in selected))
        return {"FINISHED"}


class CURIOMESH_OT_seg_rename(bpy.types.Operator):
    bl_idname = "curiomesh.seg_rename"
    bl_label = "Rename Part"
    bl_description = "Gives the selected part a name of your own"
    bl_options = {"INTERNAL", "UNDO"}

    name: bpy.props.StringProperty(name="Name", description="The new name of the part", default="")

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and len(view.seg["sel"]) == 1

    def invoke(self, context, event):
        self.name = view.seg["parts"][next(iter(view.seg["sel"]))]["name"]
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        if self.name.strip():
            view.seg["parts"][next(iter(view.seg["sel"]))]["name"] = self.name.strip()
            save()
        return {"FINISHED"}


class CURIOMESH_OT_seg_from_shapes(_SegmentOp, bpy.types.Operator):
    bl_idname = "curiomesh.seg_from_shapes"
    bl_label = "Make a Part"
    bl_description = "The faces inside the shapes in the list become a new part. Union, Subtract and Intersect combine the shapes"

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and len(context.scene.curiomesh_guided.bool_shapes) > 0

    def execute(self, context):
        guided = _g()
        source = guided.state_source()
        shapes = [(s.operation, s.obj) for s in context.scene.curiomesh_guided.bool_shapes if s.obj is not None and s.obj != source]
        if not shapes:
            return self._done("Add a shape over a part first")
        src_centres, res_centres = _centres()
        depsgraph = context.evaluated_depsgraph_get()
        src_mask = patch.combine_masks(len(src_centres), [(op, region.volume_mask(shape, guided._world(source.matrix_world, src_centres), depsgraph)) for op, shape in shapes])
        res_mask = guided._shape_mask(context, source, guided.state["auto_base"], shapes)
        return self._done(claim(src_mask, res_mask))


def _screen(context, region_, matrix_world, points):
    """Points in an object's own space on the screen, in pixels."""
    rv3d = context.region_data
    transform = np.asarray(rv3d.perspective_matrix @ matrix_world, dtype=np.float64)
    clip = points @ transform[:3, :3].T + transform[:3, 3]
    w = points @ transform[3, :3] + transform[3, 3]
    ndc = clip[:, :2] / np.where(np.abs(w) < 1e-12, 1e-12, w)[:, None]
    return (ndc + 1.0) * 0.5 * np.array([region_.width, region_.height])


def _hit_object(context, region_, event):
    """(object hit, location in its own space, the segment there) under the mouse, or None."""
    from bpy_extras import view3d_utils

    guided = _g()
    point = (event.mouse_x - region_.x, event.mouse_y - region_.y)
    origin = view3d_utils.region_2d_to_origin_3d(region_, context.region_data, point)
    direction = view3d_utils.region_2d_to_vector_3d(region_, context.region_data, point)
    best = None
    for obj in (bpy.data.objects.get(guided.PREVIEW_NAME), guided.state_source()):
        if obj is None:
            continue
        inverse = obj.matrix_world.inverted()
        hit, location, _n, index = obj.ray_cast(inverse @ origin, (inverse.to_3x3() @ direction).normalized())
        if hit:
            distance = (obj.matrix_world @ location - origin).length
            if best is None or distance < best[0]:
                best = (distance, obj, location, index)
    if best is None:
        return None
    return best[1], np.asarray(best[2], dtype=np.float64), view._part_at(context, region_, event)


class CURIOMESH_OT_seg_split(bpy.types.Operator):
    bl_idname = "curiomesh.seg_split"
    bl_label = "Split"
    bl_description = "Draw a line across a part to cut it in two, through to the back. Both halves are locked; Esc cancels"
    bl_options = {"INTERNAL", "UNDO"}

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and context.area is not None and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        self.stroke, self.object, self.target, self.handle = [], None, None, None
        self.handle = bpy.types.SpaceView3D.draw_handler_add(_draw_stroke, (self,), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set("Split: press on the segment and draw across it, release to cut. Esc cancels")
        return {"RUNNING_MODAL"}

    def _end(self, context):
        if self.handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, "WINDOW")
            self.handle = None
        context.workspace.status_text_set(None)
        overlay.tag_redraw()

    def cancel(self, context):
        self._end(context)

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._end(context)
            return {"CANCELLED"}
        region_ = view._region_under_mouse(context, event)
        if region_ is None:
            return {"PASS_THROUGH"}
        point = (event.mouse_x - region_.x, event.mouse_y - region_.y)
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = _hit_object(context, region_, event)
            selected = next(iter(view.seg["sel"])) if len(view.seg["sel"]) == 1 else None
            self.target = selected if selected is not None else (hit[2] if hit else None)
            self.object = hit[0] if hit else (_g().state_source() if selected is not None else None)
            if self.target is None or self.object is None:
                return {"RUNNING_MODAL"}
            self.stroke = [point]
            return {"RUNNING_MODAL"}
        if event.type == "MOUSEMOVE" and self.stroke:
            self.stroke.append(point)
            overlay.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE" and self.stroke:
            self.stroke.append(point)
            src_centres, res_centres = _centres()
            matrix = self.object.matrix_world
            problem = "Draw a longer line"
            if len(self.stroke) >= 2:
                problem = split(self.target, S.side_of_line(_screen(context, region_, matrix, src_centres), self.stroke), S.side_of_line(_screen(context, region_, matrix, res_centres), self.stroke))
            if problem:
                self.report({"WARNING"}, problem)
            self._end(context)
            return {"FINISHED"} if not problem else {"CANCELLED"}
        return {"PASS_THROUGH"}


class CURIOMESH_OT_seg_paint(bpy.types.Operator):
    bl_idname = "curiomesh.seg_paint"
    bl_label = "Paint"
    bl_description = "Paint a new part on the mesh with the left mouse button. [ and ] change the brush; Enter ends, Esc cancels"
    bl_options = {"INTERNAL", "UNDO"}

    mode: bpy.props.EnumProperty(description="New paints a new part. Grow and Shrink move the border of the selected part", items=[
        ("NEW", "New", "Paint a new segment"),
        ("GROW", "Grow", "Brush over the border of the selected segment to move it outwards. Ctrl paints the other way"),
        ("SHRINK", "Shrink", "Brush over the selected segment to give its faces to the neighbours beside them. Ctrl paints the other way"),
    ], default="NEW", options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        if properties.mode == "GROW":
            return "Grow the selected segment: brush over its border to move it outwards. Ctrl shrinks. [ ] brush size, Enter or right-click ends, Esc cancels"
        if properties.mode == "SHRINK":
            return "Shrink the selected segment: brush over it to give its faces to the neighbours. Ctrl grows. [ ] brush size, Enter or right-click ends, Esc cancels"
        return cls.bl_description

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and context.area is not None and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        self.target = None
        if self.mode != "NEW":
            if len(view.seg["sel"]) != 1:
                self.report({"WARNING"}, "Select one segment first")
                return {"CANCELLED"}
            self.target = next(iter(view.seg["sel"]))
            self.src_old = view.seg["src"]
        self.src_centres, self.res_centres = _centres()
        self.src_mask = np.zeros(len(self.src_centres), dtype=bool)
        self.res_mask = np.zeros(len(self.res_centres), dtype=bool)
        self.saved = view.seg["piece"]
        self.src_take = np.zeros(len(self.src_centres), dtype=bool)
        self.res_take = np.zeros(len(self.res_centres), dtype=bool)
        self.fresh = new_label()
        self.size = float(np.ptp(self.src_centres, axis=0).max()) * 0.04
        self.painting = False
        self.shrinking = self.mode == "SHRINK"
        context.window_manager.modal_handler_add(self)
        what = {"NEW": "Paint a segment", "GROW": "Grow the segment (Ctrl: shrink)", "SHRINK": "Shrink the segment (Ctrl: grow)"}[self.mode]
        context.workspace.status_text_set(f"{what}: left mouse paints   [ ] brush size   Enter or right-click ends   Esc cancels")
        return {"RUNNING_MODAL"}

    def _dab(self, context, region_, event):
        hit = _hit_object(context, region_, event)
        if hit is None:
            return
        src_dab = np.linalg.norm(self.src_centres - hit[1], axis=1) < self.size
        res_dab = np.linalg.norm(self.res_centres - hit[1], axis=1) < self.size
        if self.target is not None and self.shrinking != bool(event.ctrl):  # a shrink stroke only takes the segment's own faces
            src_dab &= self.src_old == self.target
            res_dab &= self.saved == self.target
            self.src_take |= src_dab
            self.res_take |= res_dab
            self.src_mask &= ~src_dab
            self.res_mask &= ~res_dab
        else:
            self.src_mask |= src_dab
            self.res_mask |= res_dab
            if self.target is not None:
                self.src_take &= ~src_dab
                self.res_take &= ~res_dab
        piece = self.saved.copy()
        if self.target is None:
            piece[self.res_mask] = self.fresh
        else:
            piece[self.res_mask] = self.target
            piece[self.res_take] = self.fresh  # given away: shown in the paint colour until it ends
        # Show the dabs in the selection colour: painted faces look like one part until it is made one.
        view.seg["piece"] = piece
        view.seg["painted"] = self.fresh
        view.seg["tris"], view.seg["batch_key"] = None, None
        overlay.tag_redraw()

    def _end(self, context):
        view.seg["piece"] = self.saved
        view.seg["painted"] = None
        view.seg["tris"], view.seg["batch_key"] = None, None
        context.workspace.status_text_set(None)
        overlay.tag_redraw()

    def cancel(self, context):
        self._end(context)

    def modal(self, context, event):
        if event.type == "ESC":
            self._end(context)
            return {"CANCELLED"}
        if event.type in {"RET", "NUMPAD_ENTER", "RIGHTMOUSE"} and event.value == "PRESS":
            self._end(context)
            if self.target is None:
                problem = claim(self.src_mask, self.res_mask)
            else:
                problem = ""
                if self.src_mask.any() or self.res_mask.any():
                    problem = grow(self.target, self.src_mask, self.res_mask)
                if not problem and (self.src_take.any() or self.res_take.any()):
                    target = next(iter(view.seg["sel"]), self.target)  # grown first: the numbers may have moved
                    problem = shrink(target, self.src_take, self.res_take)
            if problem:
                self.report({"WARNING"}, problem)
            return {"FINISHED"}
        if event.type in {"LEFT_BRACKET", "RIGHT_BRACKET"} and event.value == "PRESS":
            self.size *= 0.8 if event.type == "LEFT_BRACKET" else 1.25
            return {"RUNNING_MODAL"}
        region_ = view._region_under_mouse(context, event)
        if region_ is None:
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE":
            self.painting = event.value == "PRESS"
            if self.painting:
                self._dab(context, region_, event)
            return {"RUNNING_MODAL"}
        if event.type == "MOUSEMOVE" and self.painting:
            self._dab(context, region_, event)
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


def _draw_stroke(op) -> None:
    try:
        if len(op.stroke) < 2:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        shader.bind()
        shader.uniform_float("color", (1.0, 0.85, 0.1, 1.0))
        gpu.state.line_width_set(2.0)
        batch_for_shader(shader, "LINE_STRIP", {"pos": [(float(x), float(y), 0.0) for x, y in op.stroke]}).draw(shader)
        gpu.state.line_width_set(1.0)
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


class CURIOMESH_OT_seg_goal(bpy.types.Operator):
    bl_idname = "curiomesh.seg_goal"
    bl_label = "Redo"
    bl_description = "Remakes the selected parts for this Use; only they change and their borders stay. Every result stays in the part's tries"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return view.ready() and bool(view.seg["sel"]) and _g().progress_info() is None

    def execute(self, context):
        gs = context.scene.curiomesh_guided
        todo = []
        detail = round(gs.seg_goal_detail, 3)
        for p in sorted(view.seg["sel"]):
            record = view.seg["parts"][p]
            record.update(goal=gs.seg_goal, goal_detail=gs.seg_goal_detail, choice=gs.seg_method)
            tried = next((i for i, t in enumerate(record["tries"]) if t["method"] == gs.seg_goal and t.get("detail") == detail and t.get("choice") == _choice_of(gs.seg_goal, gs.seg_method)), None)
            if tried is None:
                todo.append(p)
            elif record["cur"] != tried:
                view.use_try(context, p, tried)
        save()
        # A whole loose piece at the search's own size can keep what the search measured and only change method.
        keeps_size = abs(gs.seg_goal_detail - 0.5) < 1e-3 and gs.seg_goal == "REDUCE" and all(view.seg["parts"][p].get("whole") for p in todo)
        if todo and keeps_size and gs.seg_method != "AUTO":
            view.start_redo(context, todo, gs.seg_method)
        elif todo:
            start_goal(context, todo)
        return {"FINISHED"}


def _choice_of(goal: str, method: str):
    return method if goal in {"REDUCE", "LOWPOLY"} else None


def on_selected(gs) -> None:
    """The Use, Method and Detail on the panel follow the part you click."""
    if len(view.seg["sel"]) == 1:
        record = view.seg["parts"][next(iter(view.seg["sel"]))]
        gs["seg_goal"] = next(item[4] for item in GOAL_ITEMS if item[0] == record["goal"])
        gs["seg_goal_detail"] = record["goal_detail"]
        method = next((item[4] for item in view.METHOD_ITEMS if item[0] == record.get("choice", "AUTO")), None)
        if method is not None:
            gs["seg_method"] = method


def faces_without_detail(res):
    """Per result face: True where its segment has Multires Detail off. None when every face gets it."""
    records = view.seg["parts"]
    piece = view.seg.get("piece")
    if view.seg.get("src") is None or piece is None or len(piece) != len(res["faces"]) or all(r.get("multires", True) for r in records):
        return None
    return ~np.asarray([r.get("multires", True) for r in records], dtype=bool)[np.asarray(piece)]


class CURIOMESH_OT_seg_multires(bpy.types.Operator):
    bl_idname = "curiomesh.seg_multires"
    bl_label = "Multires Detail"
    bl_description = "On Apply, the selected parts follow your mesh in the Multires levels. Off, they keep their own shape at every level"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and bool(view.seg["sel"])

    def execute(self, context):
        records = [view.seg["parts"][p] for p in view.seg["sel"]]
        on = not all(r.get("multires", True) for r in records)
        for r in records:
            r["multires"] = on
        save()
        return {"FINISHED"}


def draw_finder(layout) -> None:
    """How the mesh is cut into parts (the Finder popover of Fix)."""
    source = _g().state_source()
    if view.seg["src"] is None or source is None:
        layout.label(text="Parts are not ready yet", icon="INFO")
        return
    view.seglive.draw(layout, source)


def draw_edit(layout) -> None:
    """The hand tools for parts: merge, split, lock, grow, shrink, paint, rename, undo (the Edit Parts popover of Fix)."""
    col = layout.column(align=True)
    row = col.row(align=True)  # two by two: four buttons in one row were cut to "Ren..." in a narrow sidebar
    row.operator("curiomesh.seg_merge", icon="AUTOMERGE_ON")
    row.operator("curiomesh.seg_split", icon="SCULPTMODE_HLT")
    row = col.row(align=True)
    row.operator("curiomesh.seg_halve", icon="MOD_EDGESPLIT")
    selected = view.seg["sel"]
    on = bool(selected) and all(view.seg["parts"][p].get("locked") for p in selected if p < len(view.seg["parts"]))
    row.operator("curiomesh.seg_lock", text="Locked" if on else "Lock", icon="LOCKED" if on else "UNLOCKED", depress=on)
    row = col.row(align=True)
    row.operator("curiomesh.seg_paint", text="Grow", icon="ADD").mode = "GROW"
    row.operator("curiomesh.seg_paint", text="Shrink", icon="REMOVE").mode = "SHRINK"
    row = col.row(align=True)
    row.operator("curiomesh.seg_paint", text="Paint New", icon="BRUSH_DATA").mode = "NEW"
    row.operator("curiomesh.seg_rename", text="Rename", icon="GREASEPENCIL")
    col.operator("curiomesh.seg_undo", icon="LOOP_BACK")


def draw_selected(layout, gs) -> None:
    """The selected part, here and in the panel that opens at the click: numbers, what to change, the tries, and on to the next."""
    guided = _g()
    row = layout.row(align=True)
    row.operator("curiomesh.seg_walk", text="Previous", icon="TRIA_LEFT").step = -1
    row.operator("curiomesh.seg_walk", text="Next", icon="TRIA_RIGHT").step = 1
    row.operator("curiomesh.seg_frame", text="", icon="VIEWZOOM")
    selected = sorted(view.seg["sel"])
    if not selected:
        layout.label(text="Click a part, or press Next: worst first", icon="INFO")
        return
    box = layout.box()
    if len(selected) == 1:
        for i, line in enumerate(view.info_lines(selected[0])):
            box.label(text=line, icon="OBJECT_DATA" if i == 0 else "BLANK1")
        words = guided.state["auto_base"]["job"].explain_piece(selected[0])
        if words:
            overlay.wrap_label(box, words, icon="INFO", width=32, limit=6)
    else:
        parts = [view.seg["parts"][p] for p in selected]
        known = [r["change"] for r in parts if r["change"] is not None]
        box.label(text=f"{len(selected)} parts, {sum(r['after'] for r in parts):,} faces now", icon="OBJECT_DATA")
        if known:
            box.label(text=f"Least alike looks {view.alike(max(known))}", icon="BLANK1")
    if view.mirrored():
        overlay.wrap_label(box, "Redo is off on this result (Mirror, Clean Rows, a very large piece or an imported job)", icon="INFO", width=32, limit=3)
        return
    col = box.column(align=True)
    col.prop(gs, "seg_goal", text="Use")
    method = col.row(align=True)
    method.enabled = gs.seg_goal in {"REDUCE", "LOWPOLY"}
    method.prop(gs, "seg_method", text="Method")
    col.prop(gs, "seg_goal_detail", text="Detail", slider=True)
    if gs.multires_on:
        on = all(view.seg["parts"][p].get("multires", True) for p in view.seg["sel"])
        col.operator("curiomesh.seg_multires", text="Multires Detail: follows your mesh" if on else "Multires Detail: off here", icon="MOD_MULTIRES", depress=on)
    row = box.row(align=True)
    row.scale_y = 1.3
    row.operator("curiomesh.seg_goal", text=f"Redo {len(selected)}" if len(selected) > 1 else "Redo", icon="FILE_REFRESH")
    row = box.row(align=True)
    row.operator("curiomesh.part_size", text="Bigger Quads", icon="REMOVE").bigger = True
    row.operator("curiomesh.part_size", text="Smaller Quads", icon="ADD").bigger = False
    if guided.progress_info() is not None:
        guided.draw_progress(box, *guided.progress_info())
        box.operator("curiomesh.guided_stop_work", icon="CANCEL")
    if len(selected) == 1:
        record = view.seg["parts"][selected[0]]
        if len(record["tries"]) > 1:
            box.label(text="Tried (click to go back):")
            col = box.column(align=True)
            for i, t in enumerate(record["tries"]):
                change = "?" if t["change"] is None else view.alike(t["change"]).replace(" the same", " alike")
                score = "" if t.get("topology") is None else f", tidy {t['topology']:.0f}"
                col.operator("curiomesh.seg_try", text=f"{view.method_name(t['method'])}: {t['after']:,} faces, {change}{score}", depress=i == record["cur"]).index = i


class CURIOMESH_PT_seg_review(bpy.types.Panel):
    """Opens at the click. Registered in the header region, which Blender does not draw panels in by itself."""

    bl_label = "Segment"
    bl_idname = "CURIOMESH_PT_seg_review"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 14

    def draw(self, context):
        draw_selected(self.layout, context.scene.curiomesh_guided)


class CURIOMESH_OT_seg_walk(bpy.types.Operator):
    bl_idname = "curiomesh.seg_walk"
    bl_label = "Next Part"
    bl_description = "Go to the next or previous part, worst first, and move the view to it"
    bl_options = {"INTERNAL"}

    step: bpy.props.IntProperty(name="Step", description="1 goes to the next part, -1 to the one before", default=1)

    @classmethod
    def poll(cls, context):
        return view.ready()

    def execute(self, context):
        walk(context, self.step)
        return {"FINISHED"}


def walk(context, step: int) -> None:
    order = P.worst_first([view.judged(r) for r in view.seg["parts"]])
    current = next(iter(view.seg["sel"])) if len(view.seg["sel"]) == 1 else None
    target = S.next_in_order(order, current, step)
    if target is None:
        return
    view.select([target])
    view.frame_selected(context)
    overlay.tag_redraw()


HOVER_GAP = 1.0 / 30.0  # seconds between looks under the mouse, stretched when a look takes long


class CURIOMESH_OT_seg_review(bpy.types.Operator):
    bl_idname = "curiomesh.seg_review"
    bl_label = "Review Parts"
    bl_description = "Walks the parts in the viewport: hover to see one, click for its options, arrow keys go worst first. Esc leaves and keeps everything"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return view.ready() and context.area is not None and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        if view.seg.get("reviewing"):  # the same button leaves
            view.seg["reviewing"] = False
            return {"FINISHED"}
        view.seg["reviewing"] = True
        self.next_look, self.gap = 0.0, HOVER_GAP
        if context.space_data is not None and hasattr(context.space_data, "show_region_ui"):
            context.space_data.show_region_ui = True
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set("Review Segments   Click: select   Shift+Click: add   Left, Right: worst first   M: merge   K: split in two   L: lock   G, S: grow, shrink   Ctrl+Z: undo edit   Esc: done")
        overlay.tag_redraw()
        return {"RUNNING_MODAL"}

    def _end(self, context):
        view.seg["reviewing"] = False
        view.seg["hover"] = None
        context.workspace.status_text_set(None)
        overlay.tag_redraw()

    def cancel(self, context):
        self._end(context)

    def _look(self, context, region_, event):
        """What is under the mouse, no oftener than the mesh can take: the first look on a big mesh builds its ray tree."""
        now = time.perf_counter()
        if now < self.next_look:
            return
        part = view._part_at(context, region_, event)
        spent = time.perf_counter() - now
        self.gap = max(HOVER_GAP, 4.0 * spent)
        self.next_look = time.perf_counter() + self.gap
        if part != view.seg["hover"]:
            view.seg["hover"] = part
        overlay.tag_redraw()

    def modal(self, context, event):
        if not view.seg.get("reviewing") or _g().state["auto_base"] is None:
            self._end(context)
            return {"FINISHED"}
        if event.type == "ESC" and event.value == "PRESS":
            self._end(context)
            return {"FINISHED"}
        if not view.ready():  # a redo is putting its result in: look again in a moment
            return {"PASS_THROUGH"}
        if event.type in {"RIGHT_ARROW", "LEFT_ARROW"} and event.value == "PRESS":
            walk(context, 1 if event.type == "RIGHT_ARROW" else -1)
            return {"RUNNING_MODAL"}
        if event.type == "Z" and event.value == "PRESS" and (event.ctrl or event.oskey) and not event.shift:
            if bpy.ops.curiomesh.seg_undo.poll():
                bpy.ops.curiomesh.seg_undo()
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}
        tool = {"M": "merge", "K": "halve", "L": "lock", "G": "GROW", "S": "SHRINK"}.get(event.type) if event.value == "PRESS" and not (event.ctrl or event.alt or event.oskey) else None
        if tool is not None:
            if tool in {"GROW", "SHRINK"}:
                if bpy.ops.curiomesh.seg_paint.poll():
                    bpy.ops.curiomesh.seg_paint("INVOKE_DEFAULT", mode=tool)
            else:
                op = getattr(bpy.ops.curiomesh, "seg_" + tool)
                if op.poll():
                    op()
            return {"RUNNING_MODAL"}
        region_ = view._region_under_mouse(context, event)
        if region_ is None:
            if view.seg["hover"] is not None:
                view.seg["hover"] = None
                overlay.tag_redraw()
            return {"PASS_THROUGH"}
        x, y = event.mouse_x - region_.x, event.mouse_y - region_.y
        if x > region_.width - 100 and y > region_.height - 240:  # the view gizmo in the corner
            return {"PASS_THROUGH"}
        if event.type == "MOUSEMOVE":
            view.seg["mouse"], view.seg["region"] = (x, y), region_.as_pointer()
            self._look(context, region_, event)
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            part = view._part_at(context, region_, event)
            if part is None:
                if not event.shift:
                    view.select([])
            else:
                view.select([part], extend=event.shift)
                if not event.shift:
                    try:
                        bpy.ops.wm.call_panel(name=CURIOMESH_PT_seg_review.bl_idname, keep_open=False)
                    except RuntimeError:  # no window to open it in: the sidebar shows the same
                        pass
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


def _outlines(res, piece):
    """(edge ends, labels on either side) of every border between parts, worked out once per result, on a thread so a big mesh doesn't wait."""
    key = (id(res), id(piece))
    seg = view.seg
    if seg.get("outline_key") != key:
        seg["outline_key"], seg["outline"] = key, None

        def work():
            fa, fb, u, v = S.face_links(res["faces"])
            across = piece[fa] != piece[fb]
            if seg.get("outline_key") == key:
                seg["outline"] = (u[across], v[across], piece[fa[across]], piece[fb[across]])
                overlay.tag_redraw()

        tasks.start(work, "CurioMeshOutlines")
    return seg["outline"]


def draw_highlight(guided, preview, res, piece, flat, batch_for_shader) -> None:
    """The part under the mouse in light, its outline and those of the selected parts. Small batches, so hovering costs little on a big mesh."""
    import gpu

    seg = view.seg
    hover, selected = seg["hover"], frozenset(seg["sel"])
    outline = _outlines(res, piece)
    key = (id(res), id(piece), hover, selected, tuple(round(x, 5) for row in preview.matrix_world for x in row), outline is not None)
    if seg.get("high_key") != key:
        fill, lines = None, None
        if hover is not None:
            _id, corners, owners = view._triangles(guided, res, piece)
            world = guided._world(preview.matrix_world, res["co"][corners[owners == hover]]).astype(np.float32)
            fill = batch_for_shader(flat, "TRIS", {"pos": world, "color": np.tile(np.float32((1.0, 1.0, 1.0, 0.22)), (len(world), 1))})
        if outline is not None and (hover is not None or selected):
            u, v, la, lb = outline
            lit = {*selected} | ({hover} if hover is not None else set())
            near = np.isin(la, list(lit)) | np.isin(lb, list(lit))
            if near.any():
                ends = np.stack((u[near], v[near]), axis=1).reshape(-1)
                on_hover = (la[near] == hover) | (lb[near] == hover) if hover is not None else np.zeros(int(near.sum()), dtype=bool)
                colour = np.where(np.repeat(on_hover, 2)[:, None], np.float32((1.0, 1.0, 1.0, 1.0)), np.float32((1.0, 0.6, 0.1, 1.0))).astype(np.float32)
                lines = batch_for_shader(flat, "LINES", {"pos": guided._world(preview.matrix_world, res["co"][ends]).astype(np.float32), "color": colour})
        seg["high"], seg["high_key"] = (fill, lines), key
    fill, lines = seg["high"]
    flat.bind()
    if fill is not None:
        fill.draw(flat)
    if lines is not None:
        gpu.state.line_width_set(2.5)
        lines.draw(flat)
        gpu.state.depth_test_set("NONE")  # a faint line where the outline passes behind something
        gpu.state.line_width_set(1.0)
        lines.draw(flat)
        gpu.state.depth_test_set("LESS_EQUAL")


CLASSES = (
    CURIOMESH_OT_seg_merge,
    CURIOMESH_OT_seg_halve,
    CURIOMESH_OT_seg_lock,
    CURIOMESH_OT_seg_undo,
    CURIOMESH_OT_seg_rename,
    CURIOMESH_OT_seg_from_shapes,
    CURIOMESH_OT_seg_split,
    CURIOMESH_OT_seg_paint,
    CURIOMESH_OT_seg_goal,
    CURIOMESH_OT_seg_walk,
    CURIOMESH_OT_seg_review,
    CURIOMESH_PT_seg_review,
    CURIOMESH_OT_seg_multires,
)
