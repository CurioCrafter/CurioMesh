# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Quad Size and Organization: two sliders over a ladder of finished results.

When Auto Segments is done, the same search runs again in the background at other Details (a "rung" each) and Styles,
reusing what it measured, so most rungs take seconds. Each rung is a whole result that lands on its own amount of
change, with faces where the camera sees detail missing. Moving a slider puts the nearest ready rung on screen at once;
in between two Styles the loose pieces change over one at a time (`triadq.ladder.switched`). Bigger and Smaller Quads
move only the selected parts: a whole loose piece takes its faces from the next rung, a segment of a bigger piece is
remade for Reduce at a lower or higher Detail.

The work runs from its own timer in small slices, and waits while any job of yours runs. A new Start cancels it.
"""

from __future__ import annotations

import time

import bpy
import numpy as np

from . import autoretopo, overlay, segments
from .triadq import ladder as L
from .triadq import parts as P

MOST_FACES = 400_000  # faces of your mesh: past this each rung is a long search, so there is no ladder
ROWS_MOST = 100_000  # Clean Rows rungs only up to here: one took over three minutes on a 17,000 face bust
SLICE, GAP, WAIT = 0.03, 0.02, 0.25  # seconds of work per tick, between ticks, and between looks while a job of yours runs
GOAL_STEP = 0.15  # Bigger or Smaller Quads on a segment: this much of its Detail slider
STYLE_WORDS = {"DETAIL": "Keep Detail", "BALANCED": "Balanced", "ROWS": "Clean Rows"}

lad = {
    "key": None,  # the search these rungs belong to
    "base": None,  # (style, detail) you searched with
    "rungs": {},  # (style, detail) -> the finished AutoJob
    "plan": [],  # rungs still to work out, in order
    "job": None,  # the rung being worked out
    "doing": None,
    "shown": None,  # what is on screen: (left rung, right rung, pieces from the right one)
    "failed": [],
}


def _g():
    from . import guided  # late: guided imports this module

    return guided


def cancel() -> None:
    """Stop working out rungs and forget them (a new search, another mesh, or the add-on going away)."""
    if lad["job"] is not None:
        lad["job"].stop()
    lad.update(key=None, base=None, rungs={}, plan=[], job=None, doing=None, shown=None, failed=[])


def start(context) -> bool:
    """The search just finished: it is the first rung, and the others are worked out from now on."""
    guided = _g()
    cancel()
    base, inputs = guided.state["auto_base"], guided.state.get("auto_inputs")
    job = base and base["job"]
    if not isinstance(job, autoretopo.AutoJob) or inputs is None or guided.state.get("lod") is not None:
        return False
    if inputs[0] or job.outside is not None or len(job.faces) > MOST_FACES:  # Reduce To aims at a face count, not a Detail
        return False
    here = (job.style, round(job.detail, 3))
    lad.update(key=base["key"], base=here, rungs={here: job}, plan=L.plan(here[1], here[0], rows=len(job.faces) <= ROWS_MOST), shown=(here, here, ()))
    gs = context.scene.curiomesh_guided
    gs["quad_size"], gs["quad_order"] = 0.5, L.order_of(job.style)  # no update: the result on screen is this already
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=WAIT)
    return True


def active() -> bool:
    base = _g().state["auto_base"]
    return lad["key"] is not None and base is not None and base.get("key") == lad["key"]


def _drop_heavy(job) -> None:
    """A finished rung keeps its result and sizes; what it only needed while searching is made again if ever asked for."""
    job._samples = job._kd = job._tree = job._view = job._labels = job._src_check = job._known = None
    job._piece_trees, job._runs, job._unmeasured = {}, {}, {}


def work(seconds: float) -> None:
    """Work out rungs for about this long. The timer calls it in slices; a script can call it to finish them."""
    if not active():
        if lad["key"] is not None:
            cancel()
        return
    ends = time.perf_counter() + seconds
    while time.perf_counter() < ends:
        job = lad["job"]
        if job is None:
            if not lad["plan"]:
                return
            style, detail = lad["doing"] = lad["plan"].pop(0)
            guided = _g()
            source = guided.state_source()
            if source is None:
                cancel()
                return
            lad["job"] = guided.make_job(bpy.context, bpy.context.scene.curiomesh_guided, guided._source_data(source), guided.state["auto_inputs"], detail, style, quiet=True)
            continue
        job.step(min(SLICE, max(ends - time.perf_counter(), 0.005)))
        if job.busy:
            continue
        if job.report is not None and job.output is not None:
            _drop_heavy(job)
            lad["rungs"][lad["doing"]] = job
            overlay.tag_redraw()
        else:
            lad["failed"].append(lad["doing"])
        lad["job"] = lad["doing"] = None


def _tick():
    if not active():
        cancel()
        return None
    work_ = _g().state["work"]
    if work_ is not None and work_["runner"].busy:
        return WAIT  # yours first
    work(SLICE)
    return GAP if lad["job"] is not None or lad["plan"] else None


def finish(timeout: float | None = None, styles=None) -> bool:
    """Work out every rung (of these styles) now, for scripts and tests. True when none is left."""
    ends = None if timeout is None else time.perf_counter() + timeout
    if styles is not None:
        lad["plan"] = [r for r in lad["plan"] if r[0] in styles]
    while (lad["job"] is not None or lad["plan"]) and (ends is None or time.perf_counter() < ends):
        work(1.0)
    return lad["job"] is None and not lad["plan"]


# --------------------------------------------------------------------------- picking rungs


def ready(style: str) -> list[float]:
    return sorted(d for s, d in lad["rungs"] if s == style)


def _pick(style: str, wanted: float):
    found = L.nearest(ready(style), wanted)
    return None if found is None else (style, found)


def _real(out: dict) -> dict:
    """Face kinds back to the materials of your mesh (they differ with Keep UV Seams)."""
    return _g()._real_materials(out, (_g().state["auto_base"] or {}).get("charts"))


def _piece_keys(job) -> list[int]:
    """Each loose piece by its first face of your mesh, the same whichever rung made it."""
    return [int(ids.min()) for ids in job.pieces]


def choice(gs):
    """(left rung, right rung, pieces taken from the right) for the sliders, from what is ready. None before anything is."""
    wanted = L.size_detail(lad["base"][1], gs.quad_size)
    left, right, t = L.between(gs.quad_order)
    a, b = _pick(left, wanted), _pick(right, wanted)
    if a is None and b is None:
        return None
    if a is None or b is None or t <= 0.0 or t >= 1.0:
        one = (a if t < 0.5 else b) or a or b
        return one, one, ()
    job = lad["rungs"][a]
    blend = not job.chunks and job.mirror is None and lad["rungs"][b].mirror is None and lad["rungs"][b].output.get("piece") is not None and job.output.get("piece") is not None
    if not blend:  # one surface: no pieces to change over one at a time
        return (a, a, ()) if t < 0.5 else (b, b, ())
    moved = L.switched([len(ids) for ids in job.pieces], t)
    if len(moved) == len(job.pieces):
        return b, b, ()
    return a, b, tuple(moved)


def compose(a, b, moved) -> dict:
    """Rung `a` with the pieces in `moved` (numbered as `a` has them) taken from rung `b`."""
    out = lad["rungs"][a].output
    if not moved or a == b:
        return _real(out)
    other = lad["rungs"][b].output
    where = {key: k for k, key in enumerate(_piece_keys(lad["rungs"][b]))}
    keys = _piece_keys(lad["rungs"][a])
    builds = {p: P.cut(other["co"], other["faces"], other["mat"], other["piece"], where[keys[p]]) for p in moved if keys[p] in where}
    co, faces, mat, piece = P.splice(out["co"], out["faces"], out["mat"], out["piece"], builds)
    count = len(faces)
    return _real({"co": co, "faces": faces, "new": np.ones(count, bool), "locked": np.zeros(count, bool), "mat": mat, "parts": piece, "piece": piece})


class ShowJob(segments.segment_goals.FindJob):
    """A blend of two rungs: look at the whole object first (its numbers are not known yet), then find its segments."""

    def __init__(self, context, measure: bool):
        self.measure = measure
        super().__init__(context)

    def _run(self):
        if self.measure:
            self.stage = "Looking at the whole object"
            out = self.base["out"]
            loss = yield from self._whole_loss(out["co"], out["faces"])
            self.base["report"] = self.report = dict(self.base["report"], loss=loss, shading=None, shading_soft=None, topology=None)
        if not segments.mirrored():
            yield from super()._run()


def show(context) -> bool:
    """Put what the sliders ask for on screen. Instant: the rungs are finished results; only the numbers of a blend and
    the segments of the new result are worked out afterwards, in the background."""
    guided = _g()
    if not active():
        return False
    picked = choice(context.scene.curiomesh_guided)
    if picked is None or picked == lad["shown"]:
        return False
    a, b, moved = picked
    job = lad["rungs"][a]
    out = compose(a, b, moved)
    old = guided.state["auto_base"]
    report = dict(job.report, faces=len(out["faces"]), quads=sum(1 for f in out["faces"] if len(f) == 4), tris=sum(1 for f in out["faces"] if len(f) == 3))
    guided.stop_work()
    base = {"id": time.perf_counter_ns(), "sig": old["sig"], "key": lad["key"], "out": out, "charts": old.get("charts"), "report": report, "job": job, "summary": job.status()}
    guided.state["auto_base"] = base
    guided.state["res"] = out
    guided._build_preview(context, guided.state_source(), out, uvs=False)
    guided.state["draw_key"] = None
    if segments.seg.get("tree") is not None:
        segments.seg["tree_job"] = job  # every rung is of the same mesh: the segment tree holds
    segments.ensure()
    lad["shown"] = picked
    guided.state["message"] = f"{len(out['faces']):,} faces, {STYLE_WORDS.get(a[0], a[0])}" + ("" if not moved else f" with {len(moved)} piece{'s' if len(moved) != 1 else ''} {STYLE_WORDS.get(b[0], b[0])}")
    if moved or not segments.mirrored():
        runner = ShowJob(context, bool(moved))
        guided.state["work"] = {"key": lad["key"], "runner": runner, "cent": None, "seen": 0, "areas": []}
        if not bpy.app.timers.is_registered(guided._auto_slice):
            bpy.app.timers.register(guided._auto_slice, first_interval=0.0)
    overlay.tag_redraw()
    return True


def on_slider(self, context) -> None:
    show(context)


def status() -> str:
    """One line for under the sliders: what is being worked out, or nothing once every rung is ready."""
    if not active():
        return ""
    left = len(lad["plan"]) + (lad["job"] is not None)
    if not left:
        return ""
    doing = lad["doing"]
    words = f" ({STYLE_WORDS.get(doing[0], doing[0])}, {doing[1]:g}%)" if doing else ""
    return f"Working out more sizes: {len(lad['rungs'])} ready, {left} to go{words}"


def draw(layout, gs) -> None:
    """The two sliders, under the result."""
    if not active():
        return
    box = layout.box()
    col = box.column(align=True)
    col.prop(gs, "quad_size", slider=True)
    row = col.row(align=True)
    row.label(text="Smaller quads")
    row.label(text="Bigger quads")
    col = box.column(align=True)
    col.prop(gs, "quad_order", slider=True)
    row = col.row(align=True)
    row.label(text="Fewest faces")
    row.label(text="Even rows")
    picked = lad["shown"]
    if picked is not None:
        a, b, moved = picked
        words = STYLE_WORDS.get(a[0], a[0]) if not moved else f"{STYLE_WORDS.get(a[0], a[0])}, {len(moved)} piece{'s' if len(moved) != 1 else ''} {STYLE_WORDS.get(b[0], b[0])}"
        box.label(text=f"Showing: {words}, Detail {a[1]:g}%", icon="MESH_GRID")
    note = status()
    if note:
        overlay.wrap_label(box, note, icon="TIME", width=36, limit=2)


# --------------------------------------------------------------------------- per part


def _rung_of(record):
    """The rung a whole loose piece's faces come from now."""
    if record.get("rung"):
        return record["rung"]
    a, b, moved = lad["shown"]
    return b if record.get("loose") in moved else a


def part_sizes(context, pieces, bigger: bool) -> tuple[int, list[str]]:
    """Bigger or smaller quads on these parts only. Returns (parts changed or started, why others could not)."""
    seg, wanted, todo, why = segments.seg, {}, [], []
    base = _g().state["auto_base"]
    for p in pieces:
        record = seg["parts"][p]
        loose = record.get("loose", p)
        if record.get("whole") and active():
            style, detail = _rung_of(record)
            nxt = L.step(ready(style), detail, bigger)
            if nxt is None:
                why.append(f"{record['name']}: no {'bigger' if bigger else 'smaller'} size worked out yet")
                continue
            rung = (style, nxt)
            index = next((i for i, t in enumerate(record["tries"]) if t.get("rung") == rung), None)
            if index is None:
                job = lad["rungs"][rung]
                other = _real(job.output)
                where = {key: k for k, key in enumerate(_piece_keys(job))}
                key = int(base["job"].pieces[loose].min())
                if key not in where:
                    continue
                k = where[key]
                build = P.cut(other["co"], other["faces"], other["mat"], other["piece"], k)
                level = job.levels[k][job.picked[k]] if k < len(job.picked) else None
                change = 100.0 * level.loss / job.seen[k] if level is not None and job.seen[k] > 0 else None
                record["tries"].append({"method": record["method"], "detail": None, "rung": rung, "after": len(build[1]), "change": change, "topology": None, "build": build})
                index = len(record["tries"]) - 1
            record["rung"] = rung
            wanted[p] = index
        elif not segments.mirrored():
            record.update(goal="REDUCE", goal_detail=min(max(record["goal_detail"] + (-GOAL_STEP if bigger else GOAL_STEP), 0.0), 1.0))
            todo.append(p)
    changed = 0
    if wanted and segments.use_tries(context, wanted):
        changed += len(wanted)
    if todo:
        segments.segment_goals.save()
        if segments.segment_goals.start_goal(context, todo):
            _g().state["work"]["runner"].direction = -1 if bigger else 1  # it keeps only fewer (or more) faces
            changed += len(todo)
    return changed, why


class CURIOMESH_OT_part_size(bpy.types.Operator):
    bl_idname = "curiomesh.part_size"
    bl_label = "Quad Size of Selected"
    bl_description = "Bigger or smaller quads on the selected parts only; the rest stays as it is"
    bl_options = {"INTERNAL"}

    bigger: bpy.props.BoolProperty(name="Bigger", default=True, options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        if properties.bigger:
            return "Bigger quads on the selected parts only: fewer faces there, the rest stays as it is"
        return "Smaller quads on the selected parts only: more faces and more detail there, the rest stays as it is"

    @classmethod
    def poll(cls, context):
        return segments.ready() and bool(segments.seg["sel"]) and _g().progress_info() is None

    def execute(self, context):
        changed, why = part_sizes(context, sorted(segments.seg["sel"]), self.bigger)
        if why:
            self.report({"WARNING"}, why[0])
        return {"FINISHED"} if changed else {"CANCELLED"}


CLASSES = (CURIOMESH_OT_part_size,)
