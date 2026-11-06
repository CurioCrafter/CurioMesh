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
from itertools import chain

import bpy
import numpy as np

from . import autoretopo, organize, overlay, segments
from .triadq import ladder as L
from .triadq import tasks
from .triadq import parts as P

MOST_FACES = 400_000  # faces of your mesh: past this each rung is a long search, so there is no ladder
ROWS_MOST = 100_000  # Clean Rows rungs only up to here: one took over three minutes on a 17,000 face bust
SLOW = 120.0  # seconds: a search slower than this gets only the sizes next to its own, and no Clean Rows (a 79,000 triangle machine took about 6 minutes a rung)
SLICE, GAP, WAIT = 0.03, 0.02, 0.25  # seconds of work per tick, between ticks, and between looks while a job of yours runs
THROTTLE = 0.1  # while you drag a slider the preview changes at most this often
SETTLE = 0.4  # seconds after the last move: then the segments of the result are found (and a blend is looked at)
QUIET = 1.0  # the ladder's own work waits this long after a slider moved, so dragging never competes with it
MOST_MESHES = 40  # preview meshes kept, one per size or blend shown
GOAL_STEP = 0.15  # Bigger or Smaller Quads on a segment: this much of its Detail slider
STYLE_WORDS = {"DETAIL": "Fewest Faces", "BALANCED": "Balanced", "ROWS": "Even Rows"}  # the names of guided.STYLE_ITEMS

lad = {
    "key": None,  # the search these rungs belong to
    "base": None,  # (style, detail) you searched with
    "rungs": {},  # (style, detail) -> the finished AutoJob
    "plan": [],  # rungs still to work out, in order
    "job": None,  # the rung being worked out
    "doing": None,
    "shown": None,  # what is on screen: (left rung, right rung, pieces from the right one)
    "failed": [],
    "meshes": {},  # what the sliders picked -> its preview mesh, so moving back and forth only swaps meshes
    "want": None,  # what the sliders ask for; while dragging a blend, the nearer Style stands in for it
    "blend": None,  # (wanted, thread, box) while a blend is spliced off the main thread
    "settings": None,  # the panel settings the search ran with (`_settings`): each rung reads them again
}
drag = {"pending": False, "shown_at": 0.0, "moved": -1e9, "settled": True}


def _g():
    from . import guided  # late: guided imports this module

    return guided


def cancel() -> None:
    """Stop working out rungs and forget them (a new search, another mesh, or the add-on going away)."""
    organize.cancel()
    if lad["job"] is not None:
        lad["job"].stop()
    for mesh in lad["meshes"].values():
        try:
            mesh.use_fake_user = False
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        except ReferenceError:  # gone with a file load
            pass
    lad.update(key=None, base=None, rungs={}, plan=[], job=None, doing=None, shown=None, failed=[], meshes={}, want=None, blend=None, settings=None)
    drag.update(pending=False, settled=True)


def unregister() -> None:
    for timer in (_tick, _drag_tick):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)


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
    slow = (job.report or {}).get("seconds", 0.0) > SLOW
    plan = L.plan(here[1], here[0], rows=len(job.faces) <= ROWS_MOST and not slow)
    if slow:  # ponytail: a fixed cut; helpers for the ladder would make all sizes affordable, at the cost of more Blenders running
        plan = [(s, d) for s, d in plan if here[1] / 2.6 <= d <= here[1] * 2.6]
    gs = context.scene.curiomesh_guided
    lad.update(key=base["key"], base=here, rungs={here: job}, plan=plan, shown=(here, here, ()), settings=_settings(gs))
    _mesh_for((here, here, ()), compose(here, here, ()))
    gs["quad_size"], gs["quad_order"], gs["organize"] = 0.5, L.order_of(job.style), 0.0  # no update: the result on screen is this already
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=WAIT)
    return True


def _settings(gs) -> tuple:
    """What `make_job` reads from the panel besides the search's inputs. Changed after the search (Symmetrize, a Tweak
    with Auto-Update off), the rungs still to come would not match the result on screen."""
    return (gs.sym_axis, gs.sym_force, gs.sym_side, gs.sym_offset_use and round(gs.sym_offset, 6), tuple(_g().tweaks(gs).values()), gs.density_strength, gs.auto_tidy, gs.auto_method, gs.auto_output, gs.auto_rough, gs.auto_drop_hidden, gs.auto_redo_rough, gs.loop_count, gs.bend_loops, round(gs.loop_gap, 6))


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
            gs = bpy.context.scene.curiomesh_guided
            src = None if source is None else guided._source_data(source)
            if src is None or src["sig"] != lad["key"][0] or _settings(gs) != lad["settings"]:  # another mesh or other settings: no rung would match
                cancel()
                return
            lad["job"] = guided.make_job(bpy.context, gs, src, guided.state["auto_inputs"], detail, style, quiet=True)
            lad["job"].threaded = True
            continue
        job.step(min(SLICE, max(ends - time.perf_counter(), 0.005)))
        if job.busy:
            continue
        if job.report is not None and job.output is not None:
            _drop_heavy(job)
            lad["rungs"][lad["doing"]] = job
            rung = lad["doing"]
            _mesh_for((rung, rung, ()), compose(rung, rung, ()))  # made now, while nobody drags, so showing it is a swap
            overlay.tag_redraw()
            shown = lad["shown"]
            if len(_g().state["res"]["faces"]) > (job.report.get("before") or 10**9) and shown[0] != rung:  # on screen is heavier than your mesh: take the lighter size
                if preview(bpy.context):
                    settle(bpy.context)
        else:
            lad["failed"].append(lad["doing"])
        lad["job"] = lad["doing"] = None


def _tick():
    if not active():
        cancel()
        return None
    work_ = _g().state["work"]
    if (work_ is not None and work_["runner"].busy) or time.perf_counter() - drag["moved"] < QUIET:
        return WAIT  # yours first, and nothing while a slider moves
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
    """Details of the finished rungs of this style worth showing (none another one beats on faces and change)."""
    mine = [(d, j.report["faces"], j.report["loss"]) for (s, d), j in lad["rungs"].items() if s == style]
    source = min((j.report.get("before") or 10**9 for j in lad["rungs"].values()), default=10**9)  # never heavier than your mesh
    return sorted(L.useful([r for r in mine if r[0] in L.within(mine, source)]))


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
        if not segments.seg["parts"]:  # no segments found: the loose pieces are the parts
            segments.seg["base"] = None


def _make_mesh(name, out, source):
    """A preview mesh straight from arrays (no UVs): several times quicker than `from_pydata` on 30,000 faces."""
    faces = out["faces"]
    sizes = np.fromiter(map(len, faces), dtype=np.int64, count=len(faces))
    flat = np.fromiter(chain.from_iterable(faces), dtype=np.int32, count=int(sizes.sum()))
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(out["co"]))
    mesh.vertices.foreach_set("co", np.asarray(out["co"], dtype=np.float32).ravel())
    mesh.loops.add(len(flat))
    mesh.loops.foreach_set("vertex_index", flat)
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", (np.cumsum(sizes) - sizes).astype(np.int32))
    mesh.update(calc_edges=True)
    for m in source.data.materials:
        mesh.materials.append(m)
    if out.get("mat") is not None and len(out["mat"]) == len(faces):
        mesh.polygons.foreach_set("material_index", np.asarray(out["mat"], dtype=np.int32))
    return mesh


def _mesh_for(picked, out):
    """The preview mesh of what the sliders picked, made once."""
    mesh = lad["meshes"].get(picked)
    try:
        if mesh is not None and mesh.name:
            return mesh
    except ReferenceError:
        pass
    guided = _g()
    source = guided.state_source()
    if source is None:
        return None
    mesh = _make_mesh(guided.PREVIEW_NAME + "_Size", out, source)
    mesh.use_fake_user = True  # kept while another size is on screen
    lad["meshes"][picked] = mesh
    while len(lad["meshes"]) > MOST_MESHES:
        old = lad["meshes"].pop(next(iter(lad["meshes"])))
        old.use_fake_user = False
        if old.users == 0:
            bpy.data.meshes.remove(old)
    return mesh


def _put(context, picked, out) -> None:
    """`out` on screen as the result, for what the sliders picked. The parts table waits for `settle`."""
    guided = _g()
    a, b, moved = picked
    job = lad["rungs"][a]
    old = guided.state["auto_base"]
    report = dict(job.report, faces=len(out["faces"]), quads=sum(1 for f in out["faces"] if len(f) == 4), tris=sum(1 for f in out["faces"] if len(f) == 3))
    guided.stop_work()
    organize.cancel()  # its strengths were of the result this one replaces
    context.scene.curiomesh_guided["organize"] = 0.0
    kept = {k: segments.seg.get(k) for k in ("tree", "tree_job", "src")}  # what finding the segments again reuses
    segments.reset()  # the parts table is made once the slider rests, not on every step of a drag
    guided.state["auto_base"] = {"id": time.perf_counter_ns(), "sig": old["sig"], "key": lad["key"], "out": out, "charts": old.get("charts"), "report": report, "job": job, "summary": job.status()}
    segments.seg.update(kept, base=guided.state["auto_base"]["id"])
    segments.refresh_list()
    guided.state["res"] = out
    guided._build_preview(context, guided.state_source(), out, uvs=False, mesh=_mesh_for(picked, out))
    guided.state["draw_key"] = None
    lad["shown"] = picked
    guided.state["message"] = f"{len(out['faces']):,} faces, {STYLE_WORDS.get(a[0], a[0])}" + ("" if not moved else f" with {len(moved)} piece{'s' if len(moved) != 1 else ''} {STYLE_WORDS.get(b[0], b[0])}")
    overlay.tag_redraw()


def preview(context) -> bool:
    """Put what the sliders ask for on screen at once: a finished size is a mesh swap. A blend of two Styles shows the
    nearer of the two meanwhile; `settle` splices the real blend on a worker thread once the slider rests."""
    if not active():
        return False
    picked = choice(context.scene.curiomesh_guided)
    if picked is None:
        return False
    lad["want"] = picked
    a, b, moved = picked
    if moved:
        side = a if 2 * len(moved) < len(lad["rungs"][a].pieces) else b
        picked = (side, side, ())
    if picked == lad["shown"]:
        return False
    _put(context, picked, compose(*picked))
    return True


def settle(context) -> None:
    """The slider rests: splice a blend if one was asked for (on a thread), then find the segments of the result on
    screen and look at a blend, whose numbers are not known yet."""
    want = lad.get("want")
    if not active():
        return
    if want is not None and want != lad["shown"] and want[2]:
        box = {}

        def work():
            try:
                box["out"] = compose(*want)
            except Exception as exc:  # a thread has nobody above it to report to
                box["error"] = exc

        lad["blend"] = (want, tasks.start(work, "CurioMeshBlend"), box)
        return
    _finish(context)


def _finish(context) -> None:
    guided = _g()
    a, _b, moved = lad["shown"]
    if segments.seg.get("tree") is not None:
        segments.seg["tree_job"] = lad["rungs"][a]  # every rung is of the same mesh: the segment tree holds
    if moved or not segments.mirrored():  # finding the segments makes the parts table
        runner = ShowJob(context, bool(moved))
        guided.state["work"] = {"key": lad["key"], "runner": runner, "cent": None, "seen": 0, "areas": []}
        if not bpy.app.timers.is_registered(guided._auto_slice):
            bpy.app.timers.register(guided._auto_slice, first_interval=0.0)
    else:
        segments.seg["base"] = None
        segments.ensure()


def _blend_done(context) -> bool:
    """Put a blend spliced on its thread on screen. False while it is still being made."""
    want, task, box = lad["blend"]
    if not task.done():
        return False
    lad["blend"] = None
    if "out" in box and active() and lad.get("want") == want:
        _put(context, want, box["out"])
        _finish(context)
    return True


def show(context) -> bool:
    """`preview` and `settle` at once, for scripts: waits for a blend."""
    preview(context)
    settle(context)
    while lad.get("blend"):
        lad["blend"][1].join()
        _blend_done(context)
    return True


def pump(context):
    """While a slider moves: the preview at most every THROTTLE seconds, the rest once it rests. Returns when to look
    again, or None when there is nothing left to do (the timer's return value)."""
    now = time.perf_counter()
    if lad.get("blend") and (drag["pending"] or not _blend_done(context)):
        if drag["pending"]:
            lad["blend"] = None  # moved again: that blend is not wanted any more (its thread ends by itself)
        else:
            return THROTTLE / 2
    if drag["pending"] and now - drag["shown_at"] >= THROTTLE:
        drag.update(pending=False, shown_at=now, settled=False)
        preview(context)
    if not drag["settled"] and not drag["pending"] and now - drag["moved"] >= SETTLE:
        drag["settled"] = True
        settle(context)
    return None if drag["settled"] and not drag["pending"] and not lad.get("blend") else THROTTLE / 2


def _drag_tick():
    return pump(bpy.context)


def on_slider(self, context) -> None:
    drag.update(pending=True, moved=time.perf_counter())
    pump(context)
    if not bpy.app.timers.is_registered(_drag_tick):
        bpy.app.timers.register(_drag_tick, first_interval=THROTTLE / 2)


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
    if base is not None and getattr(base["job"], "mirror", None) is not None:  # no piece per face, and one side alone would break the mirror
        return 0, ["Bigger and Smaller Quads don't work on a mirrored result"]
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

    bigger: bpy.props.BoolProperty(name="Bigger", description="On: bigger quads on the selected parts (fewer faces). Off: smaller quads (more faces)", default=True, options={"SKIP_SAVE"})

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
