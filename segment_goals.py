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
import threading
import time

import bpy
import numpy as np
from mathutils import Vector

from . import autoretopo, guides, loops, overlay, region
from . import segments as view
from .triadq import parts as P
from .triadq import patch
from .triadq import segment as S

ATTRIBUTE = "curiomesh_segment"
SAVED = "curiomesh_segments"
MOST_MEASURED = 60  # segments looked at by the camera after finding them: a model of thousands of parts would take minutes
LOW_POLY_EDGE = 2.2  # Low Poly's quads against the sizes the others use at the same detail
SMALLEST_RESULT = 60  # a found segment with fewer faces in the result joins a neighbour: too few to be worth remaking
WORSE_TIMES, WORSE_BY = 1.5, 4.0  # Reduce may change the view this many times, and this many points, more than the part had before
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


def on_detail(self, context) -> None:
    """Segment Detail: cut the merge tree at another height. Hand-made changes to the segments start over."""
    if view.ready():
        start_find(context, fresh=True)


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
        "multires": [r.get("multires", True) for r in view.seg["parts"]],
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
        "id": label, "name": pick("names", f"Segment {label + 1}"), "kind": P.classify(points, polys), "before": len(ids), "after": count,
        "change": change, "topology": None, "method": "AUTO", "goal": pick("goals", "REDUCE"), "goal_detail": pick("details", 0.5), "multires": pick("multires", True),
        "loose": loose, "whole": whole, "source": ids,
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
    view.seg.update(base=base["id"], piece=np.asarray(out["piece"]), parts=records, src=src_labels, ref={}, batch_key=None, tris=None, hover=None, surface=True)
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


def _drop_labels(labels, gone, fa, fb) -> np.ndarray:
    """Faces of labels that can't stay join the neighbour they touch most."""
    labels = np.asarray(labels).copy()
    for value in gone:
        ids = labels == value
        crossing = ids[fa] ^ ids[fb]
        if not crossing.any():
            continue
        other = np.where(ids[fa[crossing]], labels[fb[crossing]], labels[fa[crossing]])
        labels[ids] = np.bincount(other).argmax()
    return labels


def _tiny_segments(job, labels, piece) -> list[int]:
    """Segments with few faces in the result, that are only part of a loose piece (a whole loose piece has no neighbour to join)."""
    loose = _loose_of(job)
    count = int(labels.max()) + 1
    made = np.bincount(piece, minlength=count)
    pieces = np.zeros(count, dtype=np.int64)
    pieces[labels] = loose
    per_loose = np.bincount(pieces, minlength=len(job.pieces))
    order = np.argsort(made)
    tiny = []
    for k in order.tolist():  # never the last segment of a loose piece
        if made[k] < SMALLEST_RESULT and per_loose[pieces[k]] > 1:
            tiny.append(k)
            per_loose[pieces[k]] -= 1
    return tiny


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


def transfer(src: dict, bvh, src_labels, co, faces) -> np.ndarray:
    """The segment of each result face: that of the face of your mesh nearest to it."""
    centres = S.face_frames(co, faces)[2]
    face_of_tri = src["tri_face"]
    out = np.zeros(len(faces), dtype=np.int64)
    find = bvh.find_nearest
    for k, p in enumerate(centres.tolist()):
        hit = find(Vector(p))
        if hit[0] is not None:
            out[k] = src_labels[face_of_tri[hit[2]]]
    return out


class FindJob(view.PartJob):
    """Find the segments on a thread (plain NumPy and read-only ray casts), show them, then measure them in slices."""

    def __init__(self, context, fresh: bool = False):
        self.fresh = fresh
        super().__init__(context, [], "AUTO")
        self.stage = "Finding the segments"
        self.report = self.base["report"]  # finding them changes no face, so the search's numbers still hold

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
        labels, saved = (None, None) if self.fresh else load(source, src, len(job.faces))
        if tree is None or view.seg.get("tree_job") is not job:
            tree = S.build(job.co, job.faces, _thickness(bvh, float(np.ptp(job.co, axis=0).max()) or 1.0), cancelled=cancelled)
            box["tree"] = tree
        if labels is None:
            self.fraction = 0.4
            labels = S.find(tree, job.faces, box["detail"], cancelled=cancelled)
        out = self.base["out"]
        co, faces = out["co"], out["faces"]
        self.fraction, self.stage = 0.5, "Marking the segments on the result"
        piece = transfer(src, bvh, labels, co, faces)
        normals, areas, _centres = S.face_frames(co, faces)
        links = S.face_links(faces)
        piece = S.tidy(faces, S.even_borders(faces, piece, normals, areas, rounds=1), links)
        missing = np.setdiff1d(np.unique(labels), np.unique(piece))
        if len(missing):  # too small to have a face of its own in the result
            labels = _drop_labels(labels, missing.tolist(), tree.links[0], tree.links[1])
        values, labels = np.unique(labels, return_inverse=True)
        piece = np.searchsorted(values, piece)
        labels = labels.reshape(-1)
        if saved is None:  # segments found here, not ones you kept, painted or split
            tiny = _tiny_segments(job, labels, piece)
            if tiny:
                labels = _drop_labels(labels, tiny, tree.links[0], tree.links[1])
                piece = _drop_labels(piece, tiny, links[0], links[1])
                values, labels = np.unique(labels, return_inverse=True)
                labels = labels.reshape(-1)
                piece = np.searchsorted(values, piece)
        moved = S.settle_seams(co, faces, piece, S.seam_samples(job.co, tree.links, labels))
        project = region._projector(bvh)
        moved = S.straighten_seams(moved, faces, piece, project)
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
        box = {"detail": self.context.scene.curiomesh_guided.seg_detail}

        def work():
            try:
                self._work(box, src, bvh, source)
            except Exception as exc:  # a thread has nobody above it to report to
                box["error"] = str(exc)[:100]

        thread = threading.Thread(target=work, daemon=True, name="CurioMeshSegments")
        thread.start()
        while thread.is_alive():
            if self._stopped:
                return
            yield
        if "tree" in box:
            view.seg.update(tree=box["tree"], tree_job=job)
        if "labels" not in box:
            if box.get("error") != "cancelled":
                self.warnings = [f"Finding segments failed: {box.get('error', 'no result')}"]
            return
        old = self.base["out"]
        out = dict(old, co=box["co"], piece=box["piece"], parts=box["piece"])
        install(self.base, box["labels"], out, box["saved"])
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
    if base is None:
        return False
    guided.stop_work()
    runner = FindJob(context, fresh)
    guided.state["work"] = {"key": base.get("key", ("", "AUTO")), "runner": runner, "cent": None, "seen": 0, "areas": []}
    if not bpy.app.timers.is_registered(guided._auto_slice):
        bpy.app.timers.register(guided._auto_slice, first_interval=0.0)
    return True


def worth_finding(base: dict) -> bool:
    return base["job"].mirror is None and any(len(ids) >= S.MIN_FACES for ids in base["job"].pieces)  # a mirrored result has no one-sided seams


# --------------------------------------------------------------------------- remaking a segment for its Use


class GoalJob(view.PartJob):
    """Remake the chosen segments for their Use, one at a time, then look at the whole object again."""

    def __init__(self, context, pieces):
        super().__init__(context, pieces, "AUTO")
        self.opened = 0

    def _in_thread(self, call):
        """Generator: run `call` on a thread. Returns its result, or None after noting what went wrong."""
        box = {}

        def work():
            try:
                box["out"] = call()
            except Exception as exc:  # a thread has nobody above it to report to
                box["error"] = str(exc)[:100]

        thread = threading.Thread(target=work, daemon=True, name="CurioMeshSegment")
        thread.start()
        while thread.is_alive():
            yield
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
        gap = gs.loop_gap if gs.loop_gap > 0 else 0.7 * spacing
        fine = min(loops.room(line, count, gap) for line, count, _tag in items)
        samples = np.vstack([guides.resample(line, fine / 4.0) for line, _count, _tag in items])
        co, tris, kind = guides.refine(points, tris, kind, samples, fine / 2.5, lambda p: nearest(p)[0])
        co, tris, kind, keys, _built, skipped = loops.insert(co, tris, kind, items, gap, nearest, set())
        if skipped:
            self.note = f"{len(skipped)} drawn line(s) left out: " + ", ".join(sorted(set(skipped)))
        return co, [tuple(t) for t in tris.tolist()], [str(k) for k in kind.tolist()], tuple(sorted(keys))

    def _build(self, goal: str, record: dict, ids, edge: float, topology, project, focus=None):
        """Generator: (points, faces, materials) of this segment for this Use, or None."""
        out = self.base["out"]
        co, faces = out["co"], out["faces"]
        cancelled = lambda: self._stopped  # noqa: E731
        most = np.bincount(np.asarray(out["mat"])[ids]).argmax() if len(ids) else 0
        items = self._strokes(record) if goal in {"ANIMATION", "SUBD"} else []
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
                ratio = 2.0 * area / max(edge * edge, 1e-30) / max(triangles, 1)
                if ratio < 0.98:
                    yield
                    points, polys, kinds = autoretopo._decimate(self.context, points, polys, kinds, max(ratio, 1e-4))
            quads = goal in {"REDUCE", "LOWPOLY", "STROKES"}
            keep = ()
            if goal == "STROKES":
                cut = yield from self._in_thread(lambda: self._cut_strokes(points, polys, kinds, items, nearest))
                if cut is not None:
                    points, polys, kinds, keep = cut
            made = yield from self._in_thread(lambda: S.pair_on_seams(points, polys, kinds, seams, quads=quads, guide_edges=keep, project=project))
            if made is not None:
                return made
            # Your mesh's border and the result's differ too much for a join here. The seam-keeping fill always fits.
            note = self.warnings[0] if self.warnings else ""
            self.warnings = []
            goal = "SUBD"
            self.note = f"{GOAL_NAMES.get(record['goal'], record['goal'])} could not join its neighbours ({note}): made quads on the seam instead"
        made = yield from self._in_thread(lambda: S.fill_part(co, faces, ids, goal, edge, project, topology, cancelled, focus=focus))
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
            if "edge" not in record:
                area_of = area_of if area_of is not None else S.face_frames(co, faces)[1]
                record["edge"] = float(np.sqrt(area_of[ids].sum() / max(len(ids), 1)))
            edge = record["edge"] * 2.5 ** (1.0 - 2.0 * record["goal_detail"]) * (LOW_POLY_EDGE if record["goal"] == "LOWPOLY" else 1.0)
            made = yield from self._build(record["goal"], record, ids, edge, topology, project)
            if self._stopped:
                return
            if made is None:
                continue
            change = yield from self._change(p, made[0], made[1])
            before = current["change"]
            if record["goal"] == "REDUCE" and before is not None:  # a part this small can lose its shape: more faces, up to twice
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
            elif record["goal"] == "ANIMATION" and not self._strokes(record):  # second go: more rows where the camera saw the first one go wrong
                self.stage = f"{record['name']}: more rows where the camera found a difference"
                misses = _misses(*self.last_look)
                sharper = yield from self._build("ANIMATION", record, ids, edge, topology, project, misses)
                if self._stopped:
                    return
                if sharper is not None:
                    tried = yield from self._change(p, sharper[0], sharper[1])
                    old, new = S.topology_score(made[0], made[1]) or 0.0, S.topology_score(sharper[0], sharper[1]) or 0.0
                    if tried < change and new >= old - SHARPEN_DROP:
                        made, change = sharper, tried
            trial = view.splice(co, faces, mat, piece, {p: (made[0], made[1], made[2])})[1]
            if S.open_edges(trial) > S.open_edges(faces):  # never hand back a mesh with a new hole, whatever went wrong above
                self.warnings = [f"{record['name']}: the new faces did not close on the seam, so it stays as it was"]
                continue
            self.new_tries[p] = {
                "method": record["goal"], "detail": round(record["goal_detail"], 3), "after": len(made[1]), "change": change, "topology": S.topology_score(made[0], made[1]),
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


def start_goal(context, pieces) -> bool:
    guided = _g()
    if guided.state["auto_base"] is None or not pieces:
        return False
    guided.stop_work()
    runner = GoalJob(context, pieces)
    guided.state["work"] = {"key": guided.state["auto_base"].get("key", ("", "AUTO")), "runner": runner, "cent": None, "seen": 0, "areas": []}
    guided.state["message"] = runner.status()
    if not bpy.app.timers.is_registered(guided._auto_slice):
        bpy.app.timers.register(guided._auto_slice, first_interval=0.0)
    return True


# --------------------------------------------------------------------------- changing segments by hand


def _fresh_record(job, label, ids, count, like=None) -> dict:
    record = _record(job, label, ids, count)
    if like is not None:
        record.update(goal=like["goal"], goal_detail=like["goal_detail"])
    return record


def relabel(src_labels, piece, touched) -> str:
    """Hand edits go in: `src_labels` and `piece` hold the new segments, and old segments in `touched` lose their tries. Returns a problem, or ''."""
    guided = _g()
    base = guided.state["auto_base"]
    out, job = base["out"], base["job"]
    links = S.face_links(out["faces"])
    piece = S.one_piece_each(piece, links[0], links[1], smallest=1)
    src_links = view.seg["tree"].links if view.seg.get("tree") is not None else S.face_links(job.faces)
    src_labels = S.one_piece_each(src_labels, src_links[0], src_links[1], smallest=1)
    only_source = np.setdiff1d(np.unique(src_labels), np.unique(piece))
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
        else:
            record["id"] = k
        records.append(record)
    selected = {int(np.searchsorted(values, p)) for p in view.seg["sel"] if p in values}
    out = dict(out, piece=new_piece, parts=new_piece)
    base.update(out=out, id=time.perf_counter_ns())
    job._face_piece = new_src
    view.seg.update(base=base["id"], piece=new_piece, parts=records, src=new_src, sel=selected, ref={}, batch_key=None, tris=None, hover=None, surface=True)
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
    return relabel(src, piece, set(selected))


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
    return relabel(src, piece, {segment})


def claim(source_mask: np.ndarray, result_mask: np.ndarray) -> str:
    """Faces in the masks become one new segment, taken from whatever held them."""
    if not source_mask.any() or not result_mask.any():
        return "That covers no faces"
    src, piece = view.seg["src"].copy(), view.seg["piece"].copy()
    fresh = new_label()
    touched = set(np.unique(piece[result_mask]).tolist()) | set(np.unique(src[source_mask]).tolist())
    src[source_mask] = fresh
    piece[result_mask] = fresh
    problem = relabel(src, piece, touched)
    if not problem:
        view.select([len(view.seg["parts"]) - 1])
    return problem


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
    bl_description = "Make the selected segments one"

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and len(view.seg["sel"]) > 1

    def execute(self, context):
        return self._done(merge_selected())


class CURIOMESH_OT_seg_rename(bpy.types.Operator):
    bl_idname = "curiomesh.seg_rename"
    bl_label = "Rename Segment"
    bl_description = "Give the selected segment a name of your own"
    bl_options = {"INTERNAL"}

    name: bpy.props.StringProperty(name="Name", default="")

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
    bl_label = "Make a Segment"
    bl_description = "The faces inside the shapes in the list become a new segment. Union, Subtract and Intersect combine them"

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
    bl_description = "Draw a line across a segment to cut it in two, on the side you see and through to the back. Esc cancels"
    bl_options = {"INTERNAL"}

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
    bl_description = "Paint a new segment on the mesh with the left mouse button. [ and ] change the brush. Enter or right-click ends, Esc cancels"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and context.area is not None and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        self.src_centres, self.res_centres = _centres()
        self.src_mask = np.zeros(len(self.src_centres), dtype=bool)
        self.res_mask = np.zeros(len(self.res_centres), dtype=bool)
        self.saved = view.seg["piece"]
        self.fresh = new_label()
        self.size = float(np.ptp(self.src_centres, axis=0).max()) * 0.04
        self.painting = False
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set("Paint a segment: left mouse paints   [ ] brush size   Enter or right-click ends   Esc cancels")
        return {"RUNNING_MODAL"}

    def _dab(self, context, region_, event):
        hit = _hit_object(context, region_, event)
        if hit is None:
            return
        self.src_mask |= np.linalg.norm(self.src_centres - hit[1], axis=1) < self.size
        self.res_mask |= np.linalg.norm(self.res_centres - hit[1], axis=1) < self.size
        piece = self.saved.copy()
        piece[self.res_mask] = self.fresh
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
            problem = claim(self.src_mask, self.res_mask)
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
    bl_label = "Remake"
    bl_description = "Remake the selected segments for this Use and detail. Only they change, their seams stay where they are, and every result stays in the list of tries"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return view.ready() and view.seg.get("src") is not None and bool(view.seg["sel"]) and _g().progress_info() is None

    def execute(self, context):
        gs = context.scene.curiomesh_guided
        todo = []
        detail = round(gs.seg_goal_detail, 3)
        for p in sorted(view.seg["sel"]):
            record = view.seg["parts"][p]
            record.update(goal=gs.seg_goal, goal_detail=gs.seg_goal_detail)
            tried = next((i for i, t in enumerate(record["tries"]) if t["method"] == gs.seg_goal and t.get("detail") == detail), None)
            if tried is None:
                todo.append(p)
            elif record["cur"] != tried:
                view.use_try(context, p, tried)
        save()
        if todo:
            start_goal(context, todo)
        return {"FINISHED"}


def on_selected(gs) -> None:
    """The Use and detail on the panel follow the segment you click."""
    if len(view.seg["sel"]) == 1:
        record = view.seg["parts"][next(iter(view.seg["sel"]))]
        if "goal" not in record:  # a table of loose pieces, with no segments found yet
            return
        gs["seg_goal"] = next(i for i, item in enumerate(GOAL_ITEMS) if item[0] == record["goal"])
        gs["seg_goal_detail"] = record["goal_detail"]


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
    bl_description = "On Apply, the selected segments follow your mesh in the Multires levels. Off, they keep the shape of their own quads at every level"
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


def draw_tools(layout, gs) -> None:
    """Segment Detail and the hand tools, for the top of the Parts panel."""
    col = layout.column(align=True)
    col.prop(gs, "seg_detail", slider=True)
    row = col.row(align=True)
    row.operator("curiomesh.seg_merge", icon="AUTOMERGE_ON")
    row.operator("curiomesh.seg_split", icon="SCULPTMODE_HLT")
    row.operator("curiomesh.seg_paint", icon="BRUSH_DATA")
    row.operator("curiomesh.seg_rename", text="Rename", icon="GREASEPENCIL")


def draw_goal(box, gs, multi: bool) -> None:
    """Use and detail for the selected segments."""
    col = box.column(align=True)
    col.prop(gs, "seg_goal", text="Use")
    row = col.row(align=True)
    row.prop(gs, "seg_goal_detail", text="Detail", slider=True)
    row.operator("curiomesh.seg_goal", text="Remake All" if multi else "Remake", icon="FILE_REFRESH")
    if gs.multires_on:
        on = all(view.seg["parts"][p].get("multires", True) for p in view.seg["sel"])
        col.operator("curiomesh.seg_multires", text="Multires Detail: follows your mesh" if on else "Multires Detail: off here", icon="MOD_MULTIRES", depress=on)


CLASSES = (
    CURIOMESH_OT_seg_merge,
    CURIOMESH_OT_seg_rename,
    CURIOMESH_OT_seg_from_shapes,
    CURIOMESH_OT_seg_split,
    CURIOMESH_OT_seg_paint,
    CURIOMESH_OT_seg_goal,
    CURIOMESH_OT_seg_multires,
)
