# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Segment view: the whole Auto Segments result with every part in its own colour.

Review Segments (`segment_goals`) is the one way to pick a part: hover, click, walk them worst first, change its Use,
Method and Detail, watch only it redo (the same timer and progress bar as the search), flip between the tries, or select
parts in bulk by kind. Colours, a wireframe,
the original laid over the result and a heat map of each part's camera check are overlays you switch on and off.

A part is a segment: one loose piece of your mesh, or a piece of it found by `segment_goals`. Per-part choices live in `job.methods` ({piece: method}) when the job has it, so
the Method setting and this view agree on the names. Everything here works on `guided.state["auto_base"]`.
"""

from __future__ import annotations

import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

from . import autoretopo, overlay
from .triadq import checks, issues, tasks
from .triadq import parts as P
from .triadq import segment as S

from .profiles import METHOD_ITEMS
from .translations import tip, tr

METHOD_NAMES = {item[0]: item[1] for item in METHOD_ITEMS}
ORIGINAL = "ORIGINAL"  # a try that is your own faces, untouched
METHOD_NAMES.update(ORIGINAL="Original", KEEP="Keep as Is", REDUCE="Reduce", LOWPOLY="Low Poly", ANIMATION="Animation", SUBD="Subdivision", OUTSIDE="From Outside")
KIND_ITEMS = [(k, P.KIND_WORDS[k], "") for k in P.KINDS]
MAX_TRIES = 0  # results kept per part; 0 keeps every one, so nothing you made is thrown away
LIST_ROWS = 300  # parts shown in the worst-first list
SAMPLES = 200  # vertices of each source piece used to say which piece a result part came from

seg = {
    "base": None,  # id of the auto_base this table was made for
    "piece": None,  # per result face: the source piece (part) it belongs to
    "parts": [],  # one record per part
    "sel": set(),
    "hover": None,
    "mouse": (0, 0),
    "region": 0,
    "ref": {},  # part -> its pictures of the original, for the camera check
    "whole": None,  # pictures of the whole original
    "batch": None,
    "batch_key": None,
    "bad": None,  # batch of the faces the defect check flagged
    "bad_key": None,
    "tris": None,
    "lines": None,
    "lines_key": None,
    "src": None,  # per face of your mesh: its segment, once segments are found
    "surface": False,  # parts share seams, so putting one back has to weld it
    "tree": None,  # the merge tree, kept so Segment Detail is instant
    "tree_job": None,
    "painted": None,
    "live": None,  # per result face: the live preview's segment while you change the size, until the find keeps it
    "live_count": None,
    "res_src": None,  # (result faces, per result face the nearest face of your mesh)
}


def _g():
    from . import guided  # late: guided imports this module

    return guided


def reset() -> None:
    seg.update(reviewing=False, review_mode="PARTS", seg_bad_was=None, tryall=None, outline=None, outline_key=None, high_key=None, high=None, base=None, piece=None, parts=[], sel=set(), hover=None, ref={}, whole=None, batch=None, batch_key=None, bad=None, bad_key=None, tris=None, lines=None, lines_key=None, src=None, surface=False, tree=None, tree_job=None, painted=None, live=None, live_count=None, res_src=None, history=[])


def ready() -> bool:
    """A finished result is on screen and the table is for it (not for a redo inside shapes)."""
    res = _g().state["res"]
    return bool(seg["parts"]) and _g().state["auto_base"] is not None and res is not None and seg["piece"] is not None and len(seg["piece"]) == len(res["faces"])


def name_of(piece: int) -> str:
    if piece < len(seg["parts"]) and seg["parts"][piece].get("name"):
        return seg["parts"][piece]["name"]
    return f"Part {piece + 1}"


def by_topology(record: dict) -> bool:
    """An Animation part is there for its topology, so its Tidiness ranks it, not its look."""
    return record.get("goal") == "ANIMATION" and record["method"] == "ANIMATION" and record.get("topology") is not None


def judged(record: dict):
    """What says how good a part is, scaled like a percent of change."""
    return 0.2 * (100.0 - record["topology"]) if by_topology(record) else record["change"]


def splice(co, faces, mat, face_piece, builds):
    """`parts.splice`, and where parts share seams their seam vertices join again."""
    co, faces, mat, face_piece = P.splice(co, faces, mat, face_piece, builds)
    if seg["surface"]:
        co, faces = S.weld_border(co, faces)
    return co, faces, mat, face_piece


def method_name(key: str) -> str:
    return METHOD_NAMES.get(key, key.title())


# --------------------------------------------------------------------------- the table of parts


def _vote(job, out) -> np.ndarray:
    """Source piece of each result face: the piece whose vertices lie nearest to each loose part of the result."""
    rng = np.random.default_rng(1)
    points, owner = [], []
    for index, ids in enumerate(job.pieces):
        used = np.unique(np.fromiter((v for f in ids.tolist() for v in job.faces[f]), dtype=np.int64))
        if len(used) > SAMPLES:
            used = rng.choice(used, SAMPLES, replace=False)
        points.append(job.co[used])
        owner += [index] * len(used)
    points, owner = np.vstack(points), np.asarray(owner)
    tree = KDTree(len(points))
    for i, p in enumerate(points.tolist()):
        tree.insert(p, i)
    tree.balance()
    labels = np.asarray(out["parts"])
    piece = np.zeros(len(out["faces"]), dtype=np.int64)
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        used = np.unique(np.fromiter((v for i in ids[:400].tolist() for v in out["faces"][i]), dtype=np.int64))[:60]
        votes = [owner[tree.find(tuple(out["co"][v]))[1]] for v in used.tolist()]
        piece[ids] = np.bincount(votes).argmax() if votes else 0
    return piece


def _method_of(job, piece: int) -> str:
    methods = getattr(job, "methods", None)
    return (methods or {}).get(piece, getattr(job, "method", "AUTO"))


def build_table(base: dict) -> None:
    """Make the record of every part from a finished Auto Segments result."""
    job, out = base["job"], base["out"]
    piece = np.asarray(out["piece"]) if out.get("piece") is not None else _vote(job, out)
    counts = np.bincount(piece, minlength=len(job.pieces))
    records = []
    for index, ids in enumerate(job.pieces):
        points, polys, _kinds = job._piece_arrays(ids)
        change = None
        try:
            level = job.levels[index][job.picked[index]]
            change = 100.0 * level.loss / job.seen[index] if job.seen[index] > 0 else 0.0
        except (IndexError, AttributeError, ZeroDivisionError):
            pass
        method = _method_of(job, index)
        records.append({
            "id": index, "kind": P.classify(points, polys), "before": len(ids), "after": int(counts[index]) if index < len(counts) else 0,
            "change": change, "method": method, "tries": [{"method": method, "after": int(counts[index]), "change": change, "build": None}], "cur": 0,
            "name": f"Part {index + 1}", "goal": "REDUCE", "goal_detail": 0.5, "choice": "AUTO", "whole": True, "loose": index, "source": ids, "topology": None,
        })
    seg.update(base=base["id"], piece=piece, parts=records, ref={}, whole=None, batch_key=None, bad_key=None, tris=None, hover=None, src=None, surface=False)
    seg["sel"] = {p for p in seg["sel"] if p < len(records)}
    refresh_list()


def mirrored() -> bool:
    """A mirrored result holds both sides but the job only solved one, Even Rows has no sizes to pick from, a cut piece's parts are one surface, and an imported result has no job: no part redos."""
    base = _g().state["auto_base"]
    return base is not None and (base["job"] is None or base["job"].mirror is not None or base["job"].style == "ROWS" or bool(base["job"].chunks))


def ensure() -> None:
    """Make the table for the result on screen, when it is a new one."""
    base = _g().state["auto_base"]
    if base is None:
        if seg["parts"]:
            reset()
            refresh_list()
        return
    if base["job"] is None:  # an imported result has no job to make the table from: no parts, nothing to redo
        if seg["parts"]:
            reset()
            refresh_list()
        return
    if seg["base"] != base["id"]:
        build_table(base)


def refresh_list() -> None:
    """Fill the worst-first list in the scene's settings."""
    for scene in bpy.data.scenes:
        gs = getattr(scene, "curiomesh_guided", None)
        if gs is None:
            continue
        gs.seg_parts.clear()
        for index in P.worst_first([judged(r) for r in seg["parts"]], LIST_ROWS):
            item = gs.seg_parts.add()
            item.piece = index
        gs["seg_index"] = -1


def select(pieces, extend: bool = False) -> None:
    pieces = {int(p) for p in pieces if 0 <= int(p) < len(seg["parts"])}
    seg["sel"] = (seg["sel"] ^ pieces) if extend else pieces
    seg["batch_key"] = None
    segment_goals.on_selected(bpy.context.scene.curiomesh_guided)
    overlay.tag_redraw()


def on_index(self, context) -> None:
    """Clicking a row in the worst-first list selects that part."""
    if 0 <= self.seg_index < len(self.seg_parts):
        select([self.seg_parts[self.seg_index].piece])


def on_view(self, context) -> None:
    """A view switch: wireframe on the result, the original beside or over it."""
    guided = _g()
    preview = guided.preview_object()
    source = guided.state_source()
    if preview is not None and source is not None:
        preview.show_wire = preview.show_all_edges = self.seg_wire
        place(preview, source, self)
    seg["batch_key"] = None
    overlay.tag_redraw()


def place(preview, source, gs) -> None:
    """The result beside the original, or on top of it (Auto Segments only)."""
    matrix = source.matrix_world.copy()
    if gs.scope != "AUTO" or gs.seg_compare == "BESIDE":
        matrix.translation.x += max(source.dimensions.x, 0.1) * 1.25
    preview.matrix_world = matrix


def alike(change) -> str:
    """The camera check in an artist's words, the same everywhere: "looks 94% the same"."""
    return "not measured" if change is None else f"{max(0.0, 100.0 - change):.0f}% the same"


def info_lines(piece: int) -> list[str]:
    r = seg["parts"][piece]
    fewer = 100.0 * (1.0 - r["after"] / max(r["before"], 1))
    lines = [
        f"{name_of(piece)}  ({P.KIND_WORDS[r['kind']]})",
        f"{r['after']:,} faces, {fewer:.0f}% fewer",  # as in the summary; "than 11,719" was cut in the box
        f"Looks {alike(r['change'])}" if r["change"] is not None else "Look not measured",
        f"Method: {method_name(r['method'])}",
    ]
    if r.get("locked"):
        lines.append("Locked: size changes keep it")
    if r.get("topology") is not None:
        lines.insert(3, f"Tidiness {r['topology']:.0f}/100" + (" (ranks this Animation part)" if by_topology(r) else ""))
    return lines


# --------------------------------------------------------------------------- redoing parts


def kinds_back(kinds, charts) -> np.ndarray:
    """Face kinds of a job as real materials (with Keep UV Seams each kind is a material and UV island)."""
    codes = np.asarray([int(m) if str(m).isdigit() else 0 for m in kinds], dtype=np.int32)
    return codes if charts is None else charts[1][codes]


def _faces_for(job, method: str, co, polys, kinds):
    """Generator: (points, faces, materials) for one reduced part by `method`. Returns None when it fails."""
    job._stopped = False  # the search itself was stopped when this redo took over
    return (yield from job._build(co, polys, kinds, method, len(polys) // 2))


def _step(work):
    """Generator: `work()` after one yield, so the stage on screen is current. Returns its result."""
    return (yield from tasks.run(work))


class PartJob:
    """Remake some parts by a chosen method on the main thread's timer, then look at the whole object again."""

    def __init__(self, pieces, method: str):
        guided = _g()
        base = guided.state["auto_base"]
        self.base = base
        self.job = base["job"]
        self.pieces = [int(p) for p in pieces]
        self.method = method
        self.busy = True
        self.fraction = 0.0
        self.progress = (0, len(self.pieces))
        self.stage = "Getting ready"
        self.current = None
        self.output = None
        self.output_id = 0
        self.warnings: list[str] = []
        self.report = None
        self.committed = False
        self.new_tries: dict[int, dict] = {}
        self.hold = False  # Look First: keep what it makes as tries only; the result on screen stays as it is
        self.held: dict[int, int] = {}  # part -> index of the try it made, once held
        self.started = time.perf_counter()
        self._stopped = False
        self._search = self._run()

    def seconds_left(self) -> float | None:
        if not self.busy or self.fraction < 0.1:
            return None
        return (time.perf_counter() - self.started) * (1.0 - self.fraction) / self.fraction

    def stop(self) -> None:
        self._stopped = True
        self.busy = False
        try:
            self._search.close()  # a stopped search never resumes
        except ValueError:  # stopped from inside its own step
            pass

    def step(self, seconds: float = 0.04) -> None:
        if not self.busy:
            return
        ends = time.perf_counter() + seconds
        try:
            while time.perf_counter() < ends:
                next(self._search)
        except StopIteration:
            self.busy = False
        except Exception as exc:  # a timer has nobody above it to report to
            self.warnings = [f"Remaking the part failed: {exc}"]
            self.busy = False

    def status(self) -> str:
        if self.report is None:
            return self.warnings[0] if self.warnings else self.stage
        r = self.report
        return f"{r['faces']:,} faces | Camera: {r['loss']:.1f}% of the view differs" + (f" | {self.warnings[0]}" if self.warnings else "")

    def _pictures_of(self, points, faces, center, half, pixels):
        """Generator: the pictures a few rows of rays per slice (one set of a big part's took up to 0.4 s)."""
        return (yield from autoretopo.pictures(points, faces, center, half, pixels))

    def _change(self, piece: int, points, faces):
        """Generator: percent of the part's view that differs from the original part."""
        job = self.job
        pts, polys, _kinds = job._piece_arrays(seg["parts"][piece]["source"] if "source" in seg["parts"][piece] else job.pieces[piece])
        low, high = pts.min(axis=0), pts.max(axis=0)
        center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
        extent = float(np.ptp(job.co, axis=0).max()) or 1.0
        pixels = int(autoretopo._picture_sizes(np.asarray([2.0 * half / extent]))[0])
        if piece not in seg["ref"]:
            seg["ref"][piece] = yield from self._pictures_of(pts, polys, center, half, pixels)
        view = yield from self._pictures_of(points, faces, center, half, pixels)
        wrong, shown = autoretopo.picture_counts(seg["ref"][piece], view, 2.0 * half / pixels)
        self.last_look = (seg["ref"][piece], view, center, half, pixels)
        return 100.0 * wrong / max(shown, 1)

    def _whole_loss(self, co, faces):
        job = self.job
        low, high = job.co.min(axis=0), job.co.max(axis=0)
        center, half = (low + high) / 2.0, (float(np.ptp(job.co, axis=0).max()) or 1.0) * 0.55
        pixel = 2.0 * half / autoretopo.MAX_PIXELS
        if seg["whole"] is None:
            seg["whole"] = yield from self._pictures_of(job.co, job.faces, center, half, autoretopo.MAX_PIXELS)
        view = yield from self._pictures_of(co, faces, center, half, autoretopo.MAX_PIXELS)
        wrong, shown = autoretopo.picture_counts(seg["whole"], view, pixel)
        return 100.0 * wrong / max(shown, 1)

    def _run(self):
        out = self.base["out"]
        co, faces, mat, piece = out["co"], list(out["faces"]), np.asarray(out["mat"]), seg["piece"]
        job = self.job
        builds = {}
        total = max(len(self.pieces), 1)
        for n, p in enumerate(self.pieces):
            self.progress = (n, len(self.pieces))
            self.stage = f"Remaking {name_of(p)} with {method_name(self.method)}"
            self.fraction = 0.8 * n / total
            yield
            if self._stopped:
                return
            record = seg["parts"][p]
            current = record["tries"][record["cur"]]
            if current["build"] is None:
                current["build"] = P.cut(co, faces, mat, piece, p)
            loose = record.get("loose", p)
            if loose in job.tuned:
                points, polys, kinds = job.tuned[loose]
            else:
                at = job.picked[loose]
                if job.ready[loose][at]:  # a rebuilt turned part: start from the triangles nearest its size
                    at = next((k for k in range(at, len(job.sizes[loose])) if not job.ready[loose][k]), at)
                points, polys, kinds = job.sizes[loose][at]
            made = yield from _faces_for(job, self.method, points, polys, kinds)
            if made is None:
                continue
            change = yield from self._change(p, made[0], made[1])
            had = current["change"] if current["change"] is not None else (yield from self._change(p, current["build"][0], current["build"][1]))
            src_points, src_polys, _kinds = job._piece_arrays(record.get("source", job.pieces[p]))
            if not S.view_kept(had, change, segment_goals.WORSE_TIMES, segment_goals.WORSE_BY) or not S.shape_kept(src_points, src_polys, made[0], made[1]):
                self.warnings = [f"{name_of(p)}: {method_name(self.method)} would lose its shape ({change:.0f}% of its view changes, it had {had:.0f}%), so it stays as it was"]
                continue
            builds[p] = made
            self.new_tries[p] = {"method": self.method, "after": len(made[1]), "change": change, "build": (made[0], made[1], kinds_back(made[2], self.base.get("charts")))}
        self.progress = (len(self.pieces), len(self.pieces))
        self.stage = "Looking at the whole object"
        self.fraction = 0.85
        if builds:
            co, faces, mat, piece = splice(co, faces, mat, piece, {p: self.new_tries[p]["build"] for p in builds})
        loss = yield from self._whole_loss(co, faces)
        count = len(faces)
        output = {"co": co, "faces": faces, "new": np.ones(count, bool), "locked": np.zeros(count, bool), "mat": np.asarray(mat, dtype=np.int32), "parts": piece, "piece": piece}
        self.stage = "Looking for problem spots"
        found = yield from checks.defect_steps(np.asarray(co), faces, job.co, job.faces)
        output["issue_spots"], output["issues"] = issues.analyse(np.asarray(co), faces, {k: getattr(found, k) for k in checks.Defects.KINDS}, piece, {"source_faces": len(job.faces)})
        self.output = output  # issues are in before the panel can see the new output
        self.output_id += 1
        before = self.base["report"]
        self.report = dict(before, shading=None, shading_soft=None, topology=None, faces=count, quads=sum(1 for f in faces if len(f) == 4), tris=sum(1 for f in faces if len(f) == 3), loss=loss, seconds=time.perf_counter() - self.started)
        self.fraction = 1.0


def _remember_try(record: dict, new: dict) -> int:
    """Put a made result into the part's tries (replacing one made the same way). Returns its index."""
    same = next((i for i, t in enumerate(record["tries"]) if t["method"] == new["method"] and t.get("detail") == new.get("detail")), None)
    if same is None:
        record["tries"].append(new)
        same = len(record["tries"]) - 1
    else:
        record["tries"][same] = new
    if MAX_TRIES and len(record["tries"]) > MAX_TRIES:
        drop = next(i for i in range(len(record["tries"])) if i != same)
        record["tries"].pop(drop)
        same -= drop < same
    return same


def commit(runner: PartJob) -> None:
    """A part job finished: the new result becomes the one Apply writes, and the tries are remembered."""
    guided = _g()
    base = guided.state["auto_base"]
    if isinstance(runner, segment_goals.FindJob):  # it put its result in place when it had one
        runner.committed = True
        guided.state["message"] = runner.status()
        return
    runner.committed = True
    if base is None or base is not runner.base or (runner.report is None and not getattr(runner, "hold", False)):
        guided.state["message"] = runner.status()
        return
    for p, new in runner.new_tries.items():
        record = seg["parts"][p]
        same = _remember_try(record, new)
        if getattr(runner, "hold", False):  # Look First shows this try before it is kept: nothing else changes
            runner.held[p] = same
            continue
        record.update(cur=same, method=new["method"], after=new["after"], change=new["change"], topology=new.get("topology"))
        methods = getattr(runner.job, "methods", None)
        if methods is not None and record.get("whole"):
            methods[record.get("loose", p)] = new["method"]
    if getattr(runner, "hold", False):
        return
    base["out"] = dict(runner.output)
    base["report"] = runner.report
    base["id"] = time.perf_counter_ns()
    base["summary"] = runner.status()
    base.pop("area", None)
    seg["piece"] = np.asarray(runner.output["piece"])
    seg["base"] = base["id"]
    seg["batch_key"] = None
    refresh_list()
    guided.state["message"] = runner.status()
    from . import history

    if getattr(runner, "quiet", False):  # Try Every Method records one step for the whole run
        pass
    elif runner.new_tries:  # a part was remade: one Ctrl+Z step back to the result before it
        history.record(bpy.context, getattr(runner, "step_name", "Redo Part"), f"{len(base['out']['faces']):,} faces", push=True)
    else:  # only the check after a swap: that swap's step keeps the checked result
        history.refresh(bpy.context)


def start_redo(context, pieces, method: str) -> bool:
    from . import history

    guided = _g()
    if guided.state["auto_base"] is None or not pieces:
        return False
    history.before(context)  # the result before the redo is the step Ctrl+Z goes back to
    guided.stop_work()
    runner = PartJob(pieces, method)
    guided.state["work"] = {"key": guided.state["auto_base"].get("key", ("", "AUTO")), "runner": runner, "cent": None, "seen": 0, "areas": []}
    guided.state["message"] = runner.status()
    if not bpy.app.timers.is_registered(guided._auto_slice):
        bpy.app.timers.register(guided._auto_slice, first_interval=0.0)
    return True


def use_try(context, piece: int, index: int) -> bool:
    """Go back to a result you tried: shown at once, then the whole view is checked again in short timer slices."""
    return use_tries(context, {piece: index})


def _wanted(wanted: dict) -> dict:
    return {p: i for p, i in wanted.items() if 0 <= p < len(seg["parts"]) and 0 <= i < len(seg["parts"][p]["tries"]) and i != seg["parts"][p]["cur"]}


def preview_tries(wanted: dict):
    """(co, faces, mat, face_piece) of the result with these tries swapped in ({piece: try index}), or None. Nothing
    changes: Look First draws it, and `use_tries` makes the same splice when it is kept."""
    base = _g().state["auto_base"]
    wanted = _wanted(wanted)
    if base is None or not wanted:
        return None
    out = base["out"]
    builds = {}
    for piece, index in wanted.items():
        record = seg["parts"][piece]
        if record["tries"][record["cur"]]["build"] is None:  # kept so switching back is one splice
            record["tries"][record["cur"]]["build"] = P.cut(out["co"], out["faces"], out["mat"], seg["piece"], piece)
        builds[piece] = record["tries"][index]["build"]
    return splice(out["co"], list(out["faces"]), out["mat"], seg["piece"], builds)


def use_tries(context, wanted: dict, name: str = "Use This Result") -> bool:
    """`use_try` for several parts at once ({piece: try index}): one splice, one preview, one check. One Ctrl+Z step
    and one History entry called `name`."""
    from . import history

    guided = _g()
    base = guided.state["auto_base"]
    wanted = _wanted(wanted)
    made = preview_tries(wanted)
    if base is None or made is None:
        return False
    history.before(context)
    methods = getattr(base["job"], "methods", None)
    for piece, index in wanted.items():
        record = seg["parts"][piece]
        chosen = record["tries"][index]
        record.update(cur=index, method=chosen["method"], after=chosen["after"], change=chosen["change"], topology=chosen.get("topology"))
        if methods is not None and chosen["method"] != ORIGINAL:
            methods[piece] = chosen["method"]
    co, faces, mat, face_piece = made
    count = len(faces)
    new = {"co": co, "faces": faces, "new": np.ones(count, bool), "locked": np.zeros(count, bool), "mat": mat, "parts": face_piece, "piece": face_piece}
    guided.stop_work()
    base.update(out=new, id=time.perf_counter_ns())
    base.pop("area", None)
    seg["piece"], seg["base"], seg["batch_key"] = face_piece, base["id"], None
    guided.state["res"] = new
    guided._build_preview(context, guided.state_source(), new, uvs=False)
    guided.state["draw_key"] = None
    refresh_list()
    runner = PartJob([], base["job"].method if base["job"] is not None else "AUTO")  # no part to remake: only the whole-object check
    guided.state["work"] = {"key": base.get("key", ("", "AUTO")), "runner": runner, "cent": None, "seen": 0, "areas": []}
    if not bpy.app.timers.is_registered(guided._auto_slice):
        bpy.app.timers.register(guided._auto_slice, first_interval=0.0)
    history.record(context, name, f"{count:,} faces", push=True)
    overlay.tag_redraw()
    return True


def bad_parts() -> list[int]:
    """Parts holding a problem spot the checks found (inward, folded, crossing...), worst spot's part first."""
    res = _g().state["res"]
    if res is None or not seg["parts"]:
        return []
    found = []
    for spot in res.get("issue_spots") or []:
        p = int(spot.get("part", -1))
        if 0 <= p < len(seg["parts"]) and p not in found and seg["parts"][p]["method"] != ORIGINAL:
            found.append(p)
    return found


def keep_original(context, pieces, name: str = "Keep Original") -> int:
    """Put your own faces back on these parts, as one more try each (so the reduced result stays one click away). Returns how many changed."""
    wanted = original_tries(pieces)
    return len(wanted) if wanted and use_tries(context, wanted, name) else 0


def original_tries(pieces) -> dict:
    """{piece: index of its try that is your own faces}, adding that try where there is none yet."""
    base = _g().state["auto_base"]
    if base is None or base["job"] is None or mirrored():
        return {}
    job, wanted = base["job"], {}
    for p in pieces:
        record = seg["parts"][p]
        index = next((i for i, t in enumerate(record["tries"]) if t["method"] == ORIGINAL), None)
        if index is None:
            points, polys, mats = job._piece_arrays(record["source"])
            faces = [tuple(int(v) for v in polys.flat[a:b]) for a, b in zip(polys.indptr[:-1].tolist(), polys.indptr[1:].tolist())]
            mats = np.asarray(mats).astype(np.asarray(base["out"]["mat"]).dtype)
            record["tries"].append({"method": ORIGINAL, "after": len(faces), "change": 0.0, "build": (points, faces, mats)})
            index = len(record["tries"]) - 1
        wanted[p] = index
    return wanted


def original_cost(pieces) -> int:
    """Faces that putting your own faces back on these parts would add (negative: it saves faces)."""
    parts = seg["parts"]
    return int(sum(len(parts[p]["source"]) - parts[p]["after"] for p in pieces if 0 <= p < len(parts) and "source" in parts[p]))


def ask(operator, context, event, message: str):
    """An OK / Cancel box with this question (just the title on Blender before 4.1)."""
    try:
        return context.window_manager.invoke_confirm(operator, event, title=tr(operator.bl_label), message=tr(message))
    except TypeError:
        return context.window_manager.invoke_confirm(operator, event)


class CURIOMESH_OT_keep_original_bad(bpy.types.Operator):
    bl_idname = "curiomesh.keep_original_bad"
    bl_label = "Keep Original Where Bad"
    bl_description = "Puts your own faces back on every part with bad faces. Each part can be switched back from its card"
    bl_options = {"INTERNAL"}

    @classmethod
    def description(cls, context, properties):
        found = bad_parts()
        return tip(f"Puts your own faces back on the {len(found)} part{'s' if len(found) != 1 else ''} with bad faces. Adds about {original_cost(found):,} faces; each part can be switched back from its card")

    @classmethod
    def poll(cls, context):
        ensure()
        return ready() and not mirrored() and _g().progress_info() is None and bool(bad_parts())

    def invoke(self, context, event):
        found = bad_parts()
        return ask(self, context, event, f"Your own faces go back on {len(found)} part{'s' if len(found) != 1 else ''}: about {original_cost(found):,} more faces.")

    def execute(self, context):
        changed = keep_original(context, bad_parts())
        if not changed:
            self.report({"WARNING"}, tr("No part could be put back"))
            return {"CANCELLED"}
        self.report({"INFO"}, tr(f"Original faces kept on {changed} part{'s' if changed != 1 else ''}"))
        return {"FINISHED"}


# --------------------------------------------------------------------------- drawing


def _palette(count: int) -> np.ndarray:
    hue = (np.arange(count) * 0.61803398875) % 1.0  # golden steps keep neighbouring parts far apart in colour
    return 0.45 + 0.45 * np.cos(2.0 * np.pi * (hue[:, None] + np.array([0.0, 0.33, 0.67])))


def part_colours(gs, count: int, showing: bool = False):
    """RGBA per part for the current overlay switches, or None when nothing should be tinted. The part under the mouse has its own small batch: redoing this one on every hover would stall a big mesh.
    A live preview (`showing`) is always tinted, in plain colours: its segments have no numbers or selection yet."""
    if showing:
        return np.concatenate((_palette(count), np.full((count, 1), 0.6)), axis=1).astype(np.float32)
    selected = seg["sel"]
    if not (gs.seg_colors or gs.seg_heat or selected):
        return None
    if gs.seg_heat:
        changes = [judged(r) for r in seg["parts"]]
        top = max([3.0 * gs.auto_detail] + [c for c in changes if c is not None])
        rgb = np.asarray([P.heat_colour(c, top) for c in changes])
    else:
        rgb = _palette(count)
    alpha = np.full(count, 0.42 if (gs.seg_colors or gs.seg_heat) else 0.0)
    if gs.seg_heat:
        alpha[:] = 0.6
    if seg["painted"] is not None:  # the segment being painted has no number yet
        rgb = np.vstack((rgb, [[1.0, 0.85, 0.1]]))
        alpha = np.concatenate((alpha, [0.9]))
    for p in selected:
        rgb[p] = rgb[p] * 0.5 + 0.5
        alpha[p] = 0.85
    return np.concatenate((rgb, alpha[:, None]), axis=1).astype(np.float32)


def _triangles(guided, res, piece):
    if seg["tris"] is None or seg["tris"][0] is not res or seg["tris"][3] is not piece:  # not by id: a freed result's id comes back
        tris = guided._fan(res["faces"])
        owner = np.repeat(piece, [len(f) - 2 for f in res["faces"]])
        seg["tris"] = (res, tris.reshape(-1), np.repeat(owner, 3), piece)
    return seg["tris"][:3]


BAD_COLOUR = (1.0, 0.05, 0.05, 0.85)


def bad_corners(guided, res) -> np.ndarray:
    """Triangle corner ids of the faces the defect check flagged (res["bad"]), 3 per triangle; empty if none or if the result was edited since."""
    bad = res.get("bad")
    if bad is None or len(bad) != len(res["faces"]) or not bad.any():
        return np.zeros(0, dtype=np.int64)
    corners = guided._fan(res["faces"]).reshape(-1, 3)
    owner = np.repeat(np.arange(len(res["faces"])), [len(f) - 2 for f in res["faces"]])
    return corners[bad[owner]].reshape(-1)


def _draw_view() -> None:
    try:
        guided = _g()
        gs = getattr(bpy.context.scene, "curiomesh_guided", None)
        source, preview, res = guided.state_source(), guided.preview_object(), guided.state["res"]
        if gs is None or not gs.enabled or gs.scope != "AUTO" or source is None or preview is None or res is None or not ready():
            return
        piece = seg["piece"]
        if piece is None or len(piece) != len(res["faces"]):
            return
        live = seg.get("live")
        showing = live is not None and len(live) == len(res["faces"])
        if showing:
            piece = live
        import gpu
        from gpu_extras.batch import batch_for_shader

        matrix = tuple(round(x, 5) for row in preview.matrix_world for x in row)
        seg["held"] = (res, piece)  # kept alive so the ids below can't come back on a new result
        colour_key = (id(res), id(piece), matrix, gs.seg_colors, gs.seg_heat, frozenset(seg["sel"]), round(gs.auto_detail, 2), showing)
        flat = gpu.shader.from_builtin("FLAT_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set("LESS_EQUAL")
        count = seg["live_count"] if showing else len(seg["parts"])

        def colour_arrays():  # in overlay.drawn_arrays' timer, not in this draw
            colours = part_colours(gs, count, showing)
            if colours is None:
                return ()
            _id, corners, owners = _triangles(guided, res, piece)
            return (guided._world(preview.matrix_world, res["co"][corners]).astype(np.float32), colours[owners])

        if seg["batch_key"] is None:  # set to None elsewhere: the colours are built again
            seg["batch_key"] = time.perf_counter_ns()
        arrays = overlay.drawn_arrays("segments.colours", (colour_key, seg["batch_key"]), colour_arrays)
        if arrays is not None and seg.get("batch_of") is not arrays:
            seg["batch"] = batch_for_shader(flat, "TRIS", {"pos": arrays[0], "color": arrays[1]}) if arrays else None
            seg["batch_of"] = arrays
        if seg["batch"] is not None:
            flat.bind()
            seg["batch"].draw(flat)
        if gs.seg_bad:

            def bad_arrays():
                corners = bad_corners(guided, res)
                return guided._world(preview.matrix_world, res["co"][corners]).astype(np.float32) if len(corners) else None

            if seg.get("bad_key") is None:
                seg["bad_key"] = time.perf_counter_ns()
            bad = overlay.drawn_arrays("segments.bad", (id(res), matrix, seg["bad_key"]), bad_arrays)
            if seg.get("bad_of") is not bad:
                seg["bad"] = batch_for_shader(flat, "TRIS", {"pos": bad, "color": np.tile(np.float32(BAD_COLOUR), (len(bad), 1))}) if bad is not None else None
                seg["bad_of"] = bad
            if seg.get("bad") is not None:
                flat.bind()
                seg["bad"].draw(flat)
            issue_panel.draw_overlay(guided, preview, res, flat, batch_for_shader)
        if not showing:  # outlines and hover belong to the kept segments
            segment_goals.draw_highlight(guided, preview, res, piece, flat, batch_for_shader)
        if gs.seg_compare == "OVER":
            lines_key = (source.name, len(source.data.edges), tuple(round(x, 5) for row in source.matrix_world for x in row))

            def line_arrays():
                edges = np.empty(len(source.data.edges) * 2, dtype=np.int64)
                source.data.edges.foreach_get("vertices", edges)
                verts = np.empty(len(source.data.vertices) * 3, dtype=np.float64)
                source.data.vertices.foreach_get("co", verts)
                return guided._world(source.matrix_world, verts.reshape(-1, 3)[edges]).astype(np.float32)

            if seg.get("lines_key") is None:
                seg["lines_key"] = time.perf_counter_ns()
            ends = overlay.drawn_arrays("segments.lines", (lines_key, seg["lines_key"]), line_arrays)
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            if ends is not None and seg.get("lines_of") is not ends:
                seg["lines"] = batch_for_shader(shader, "LINES", {"pos": ends})
                seg["lines_of"] = ends
            if seg.get("lines") is not None:
                shader.bind()
                shader.uniform_float("color", (1.0, 1.0, 1.0, 0.45))
                gpu.state.line_width_set(1.0)
                seg["lines"].draw(shader)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
    except Exception:
        overlay.draw_failed("segments._draw_view")


def _draw_legend() -> None:
    """With Heat on: the colour bar and what its ends mean, bottom right of the viewport."""
    gs = getattr(bpy.context.scene, "curiomesh_guided", None)
    if gs is None or not gs.seg_heat or not seg["parts"]:
        return
    import blf
    import gpu
    from gpu_extras.batch import batch_for_shader

    ui = overlay._ui_scale()
    wide, high, bottom = 160 * ui, 10 * ui, 40 * ui
    left = bpy.context.region.width - overlay._sidebar_inset() - wide - 24 * ui  # bottom right: the HUD card holds the bottom left
    top = max([3.0 * gs.auto_detail] + [c for c in (judged(r) for r in seg["parts"]) if c is not None])
    steps = 16
    pos, col, idx = [], [], []
    for i in range(steps + 1):
        x = left + wide * i / steps
        c = (*P.heat_colour(top * i / steps, top), 1.0)
        pos += [(x, bottom), (x, bottom + high)]
        col += [c, c]
        if i:
            k = 2 * i
            idx += [(k - 2, k - 1, k), (k - 1, k + 1, k)]
    shader = gpu.shader.from_builtin("SMOOTH_COLOR")
    batch_for_shader(shader, "TRIS", {"pos": pos, "color": col}, indices=idx).draw(shader)
    blf.size(0, 12.0 * ui)
    blf.color(0, 1.0, 1.0, 1.0, 1.0)
    blf.position(0, left, bottom + high + 6 * ui, 0)
    blf.draw(0, tr("View change per part"))
    blf.position(0, left, bottom - 16 * ui, 0)
    blf.draw(0, tr("0%"))
    text = f"{top:.0f}%+"
    blf.position(0, left + wide - blf.dimensions(0, tr(text))[0], bottom - 16 * ui, 0)
    blf.draw(0, tr(text))


def _draw_text() -> None:
    """The numbers of the part under the mouse, next to the mouse, and the Heat legend."""
    try:
        _draw_legend()  # every 3D view: the tint shows in each
        hover = seg["hover"]
        if hover is None or not seg["parts"] or hover >= len(seg["parts"]) or bpy.context.region is None or bpy.context.region.as_pointer() != seg["region"]:
            return
        import blf

        x, y = seg["mouse"]
        lines = info_lines(hover) + ["Click: select   Arrows: next   Esc: done"]
        ui = overlay._ui_scale()
        size, step, pad = 13.0 * ui, 18.0 * ui, 8.0 * ui
        blf.size(0, size)
        wide = max(blf.dimensions(0, line)[0] for line in lines)
        edge = bpy.context.region.width - overlay._sidebar_inset()  # the N-panel is drawn over the viewport's right side
        left = x + 22 if x + 22 + wide + pad < edge else max(x - 22 - wide - pad, pad)
        y = max(y, step * len(lines) + 2 * pad)  # keep the last line above the bottom edge
        # a dark card behind the words: white on a pale part (or a light mesh) was unreadable
        import gpu

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        shader.bind()
        top, bottom = y - step + size + pad, y - step * len(lines) - pad  # first line's cap height to last line's descender
        overlay._fill(shader, overlay._rounded_rect(left - pad, bottom, wide + 2 * pad, top - bottom, 6 * ui), (0.07, 0.08, 0.11, 0.88))
        gpu.state.blend_set("NONE")
        for i, line in enumerate(lines):
            muted = i == len(lines) - 1
            blf.color(0, *((0.62, 0.65, 0.75, 1.0) if muted else (1.0, 1.0, 1.0, 1.0) if i else (1.0, 0.86, 0.45, 1.0)))
            blf.position(0, left, y - step - step * i, 0)
            blf.draw(0, line)
    except Exception:
        overlay.draw_failed("segments._draw_text")


# --------------------------------------------------------------------------- operators


def _region_under_mouse(context, event):
    area = context.area
    if area is None or area.type != "VIEW_3D":
        return None
    for region_ in area.regions:
        if region_.type == "WINDOW" and region_.x <= event.mouse_x < region_.x + region_.width and region_.y <= event.mouse_y < region_.y + region_.height:
            return region_
    return None


def _part_at(context, region_, event):
    """The part under the mouse, on the result or on your original mesh. None over nothing."""
    from bpy_extras import view3d_utils

    guided = _g()
    rv3d = context.region_data
    point = (event.mouse_x - region_.x, event.mouse_y - region_.y)
    origin = view3d_utils.region_2d_to_origin_3d(region_, rv3d, point)
    direction = view3d_utils.region_2d_to_vector_3d(region_, rv3d, point)
    best = None
    for obj, owner in ((guided.preview_object(), "result"), (guided.state_source(), "source")):
        if obj is None:
            continue
        inverse = obj.matrix_world.inverted()
        hit, location, _n, index = obj.ray_cast(inverse @ origin, (inverse.to_3x3() @ direction).normalized())
        if not hit:
            continue
        distance = (obj.matrix_world @ location - origin).length
        if best is None or distance < best[0]:
            best = (distance, owner, index)
    if best is None:
        return None
    _d, owner, index = best
    if owner == "result":
        piece = seg["piece"]
        return int(piece[index]) if piece is not None and index < len(piece) else None
    job = guided.state["auto_base"]["job"]
    if len(job.faces) != len(guided.state_source().data.polygons):
        return None  # the original was welded or cleaned, so its face numbers no longer match
    if not hasattr(job, "_face_piece"):
        job._face_piece = np.zeros(len(job.faces), dtype=np.int64)
        for p, ids in enumerate(job.pieces):
            job._face_piece[ids] = p
    return int(job._face_piece[index]) if index < len(job._face_piece) else None


class CURIOMESH_OT_seg_select(bpy.types.Operator):
    bl_idname = "curiomesh.seg_select"
    bl_label = "Select Parts"
    bl_description = "Selects all the parts, none, the ten that changed most, or every part of one kind"
    bl_options = {"INTERNAL"}

    what: bpy.props.EnumProperty(name="Which", description="Which parts to select: all, none, the ten that changed most, or every part of one kind", items=[("ALL", "All", "Every part"), ("NONE", "None", "No part"), ("WORST", "Worst 10", "The ten parts that changed most"), *KIND_ITEMS], default="ALL")

    @classmethod
    def poll(cls, context):
        return ready()

    def execute(self, context):
        parts = seg["parts"]
        if self.what == "ALL":
            select(range(len(parts)))
        elif self.what == "NONE":
            select([])
        elif self.what == "WORST":
            select(P.worst_first([r["change"] for r in parts], 10))
        else:
            select([r["id"] for r in parts if r["kind"] == self.what])
        context.scene.curiomesh_guided["seg_index"] = -1
        return {"FINISHED"}


class CURIOMESH_OT_seg_redo(bpy.types.Operator):
    bl_idname = "curiomesh.seg_redo"
    bl_label = "Redo"
    bl_description = "Remakes the selected parts with this method; the rest stays. Every result stays in the part's tries"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return ready() and not mirrored() and bool(seg["sel"]) and _g().progress_info() is None

    def execute(self, context):
        method = context.scene.curiomesh_guided.seg_method
        todo = []
        for p in sorted(seg["sel"]):
            record = seg["parts"][p]
            tried = next((i for i, t in enumerate(record["tries"]) if t["method"] == method), None)
            if tried is not None:  # already made: just go back to it
                if record["cur"] != tried:
                    use_try(context, p, tried)
            else:
                todo.append(p)
        if todo:
            start_redo(context, todo, method)
        return {"FINISHED"}


class CURIOMESH_OT_seg_try(bpy.types.Operator):
    bl_idname = "curiomesh.seg_try"
    bl_label = "Use This Result"
    bl_description = "Switch the selected part back to this result. It is shown at once; the whole-object check follows in short steps while you keep working"
    bl_options = {"INTERNAL"}

    index: bpy.props.IntProperty(name="Try", description="Which earlier result of this part to switch to; 0 is the first")

    @classmethod
    def poll(cls, context):
        return ready() and not mirrored() and len(seg["sel"]) == 1 and _g().progress_info() is None

    def execute(self, context):
        use_try(context, next(iter(seg["sel"])), self.index)
        return {"FINISHED"}


def frame_selected(context) -> bool:
    """Move the 3D view to the selected parts."""
    guided = _g()
    rv3d = getattr(context.space_data, "region_3d", None) or context.region_data
    preview, res = guided.preview_object(), guided.state["res"]
    if rv3d is None or preview is None or res is None:
        return False
    mask = np.isin(seg["piece"], list(seg["sel"]))
    used = np.unique(np.fromiter((v for i in np.flatnonzero(mask).tolist() for v in res["faces"][i]), dtype=np.int64))
    if not len(used):
        return False
    world = guided._world(preview.matrix_world, res["co"][used])
    low, high = world.min(axis=0), world.max(axis=0)
    rv3d.view_location = Vector(((low + high) / 2.0).tolist())
    rv3d.view_distance = max(float(np.linalg.norm(high - low)) * 1.6, 1e-3)
    return True


class CURIOMESH_OT_seg_frame(bpy.types.Operator):
    bl_idname = "curiomesh.seg_frame"
    bl_label = "Frame"
    bl_description = "Moves the 3D view to the selected parts"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return ready() and bool(seg["sel"])

    def execute(self, context):
        return {"FINISHED"} if frame_selected(context) else {"CANCELLED"}


# --------------------------------------------------------------------------- panel


class CURIOMESH_PG_segpart(bpy.types.PropertyGroup):
    piece: bpy.props.IntProperty()


class CURIOMESH_UL_segparts(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if item.piece >= len(seg["parts"]):
            return
        r = seg["parts"][item.piece]
        row = layout.split(factor=0.5, align=True)  # the name needs room: "Segment 12" was cut to "Seg..."
        row.label(text=tr(name_of(item.piece)), icon="LOCKED" if r.get("locked") else "CHECKBOX_HLT" if item.piece in seg["sel"] else "CHECKBOX_DEHLT")
        row = row.split(factor=0.42, align=True)  # "92%" is short; the face count needs the room ("1,65...")
        # the number the list is sorted by: the look, or Tidiness for an Animation part (a 98% look ranked low read as a mistake)
        row.label(text=tr(f"T {r['topology']:.0f}" if by_topology(r) else "?" if r["change"] is None else alike(r["change"]).replace(" the same", "")))
        row.label(text=tr(f"{r['after']:,} f"))


# --------------------------------------------------------------------------- registration

from . import issue_panel, seglive, segment_goals  # noqa: E402  (segment_goals builds on PartJob above)

PRE_CLASSES = (CURIOMESH_PG_segpart, *seglive.PRE_CLASSES)
CLASSES = (
    CURIOMESH_UL_segparts,
    CURIOMESH_OT_seg_select,
    CURIOMESH_OT_seg_redo,
    CURIOMESH_OT_seg_try,
    CURIOMESH_OT_seg_frame,
    CURIOMESH_OT_keep_original_bad,
    *segment_goals.CLASSES,
    *issue_panel.CLASSES,
    *seglive.CLASSES,
)
_handles: list = []


def register() -> None:
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), "WINDOW", "POST_VIEW"))
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_text, (), "WINDOW", "POST_PIXEL"))
    seglive.register()


def unregister() -> None:
    while _handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), "WINDOW")
        except Exception:
            pass
    seglive.unregister()
    segment_goals.stop_try_all(quiet=True)
    reset()
    issue_panel.ui.update(res=None, cur=-1, diff=None, batch=None, batch_key=None)  # its GPU batches go with the draw handler
