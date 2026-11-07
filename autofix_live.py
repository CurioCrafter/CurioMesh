# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Auto Fix after Start, the Blender half (bench/tools_design/AUTO_FIX.md 3 to 5).

`triadq/autofix.py` decides what to fix and whether to keep it; this module runs it on the main thread in timer slices
(no thread, no process: the GitHub add-on and the Blender-site build run the same code), commits each kept pass as one
Blender undo step and one History entry, and tells the user what happened: a progress box with Stop while it works, then
a readout with Show, Undo Auto Fix, and Keep going when the time limit ended it.
"""

from __future__ import annotations

import time

import bpy
import numpy as np
from bpy.app.handlers import persistent

import mathutils

from . import history, overlay, tidylive
from .triadq import autofix
from .triadq import trial as T

GAP = 0.01  # seconds between timer ticks while working: the interface stays usable
REDRAW = 0.25  # seconds between panel redraws while working
WIDTH = 36
GROUP_TEXT = {  # the seven ticks of the popover: (label, what it fixes), AUTO_FIX.md 5.1
    "leftovers": ("Leftovers", "faces facing in, two-edge corners, triangle pairs, thin quads"),
    "seam": ("Mirror seam", "puts the middle back on the mirror plane"),
    "poles": ("Poles", "removes stray pairs and short dead-end strips"),
    "patches": ("Messy patches", "redraws patches full of poles as clean rows"),
    "hard": ("Hard edges", "wavy edges, edges off the crease, bumpy flat panels"),
    "rows": ("Rows", "wiggly strips, uneven widths, bunched bands"),
    "shape": ("Quad shape", "pinched poles, slanted quads, sizes, lumpy spacing"),
}
GROUP_TIPS = {
    "patches": "Redraws small patches full of poles as clean rows of quads. Changes the faces there; off keeps the faces as the search made them",
}
TIME_WORDS = {"NORMAL": "Normal", "LONGER": "Longer", "NONE": "No Limit"}

fx = {"job": None, "report": None, "show": None, "show_handle": None, "show_op": False, "seen_kinds": {}}


# --------------------------------------------------------------------------- one shim


def _relay(work, rnd, i, trial, count):
    """`autofix._relay`, with one change: a step of the work that has no number yet (the camera pictures, the word
    "measuring") is also an event, so a timer slice ends there and Blender never waits on a whole measurement.
    ponytail: drop this once autofix._relay yields on those steps itself."""
    while True:
        try:
            progress = next(work)
        except StopIteration as done:
            return done.value
        yield autofix.Event("working", rnd, i, trial, spots=count, progress=float(progress) if isinstance(progress, (int, float)) else -1.0)


autofix._relay = _relay


# --------------------------------------------------------------------------- the settings


def _ticks(context) -> set:
    pg = context.scene.curiomesh_autofix
    return {g for g in autofix.GROUPS if getattr(pg, g)}


class CURIOMESH_PG_autofix(bpy.types.PropertyGroup):
    leftovers: bpy.props.BoolProperty(name="Leftovers", description="Fixes faces facing in, two-edge corners, triangle pairs and thin quads", default=True)
    seam: bpy.props.BoolProperty(name="Mirror seam", description="Puts the middle of a mirrored result back on the mirror plane", default=True)
    poles: bpy.props.BoolProperty(name="Poles", description="Removes stray pairs of poles and short dead-end strips", default=True)
    patches: bpy.props.BoolProperty(name="Messy patches", description=GROUP_TIPS["patches"], default=True)
    hard: bpy.props.BoolProperty(name="Hard edges", description="Straightens wavy hard edges, moves edges back onto the crease and presses bumpy flat panels flat", default=True)
    rows: bpy.props.BoolProperty(name="Rows", description="Straightens wiggly strips, evens uneven strip widths and bunched bands", default=True)
    shape: bpy.props.BoolProperty(name="Quad shape", description="Opens up pinched poles and evens slanted quads, sizes and lumpy spacing", default=True)
    seen: bpy.props.BoolProperty(name="Seen", description="The first-time note under the Auto Fix result was read", default=False, options={"HIDDEN"})
    open: bpy.props.BoolProperty(name="Details", description="Show what each fix did", default=False)


# --------------------------------------------------------------------------- look, as the camera check asks for it


class _Look:
    """The camera check for Auto Fix. A new `_looker` per pass (its newest-pictures cache is keyed by array identity),
    and it holds on to the arrays it was last asked about, so an identity can never be reused under it."""

    def __init__(self, src_co, src_faces):
        self.src = (src_co, src_faces)
        self.fresh()

    def fresh(self):
        self.inner = tidylive._looker(*self.src)
        self.keep = None

    def __call__(self, co, faces):
        self.keep = (co, faces)
        return self.inner(co, faces)

    def parts(self, co, faces, labels, limit):
        self.keep = (co, faces)
        return self.inner.parts(co, faces, labels, limit)


# --------------------------------------------------------------------------- the job


def _g():
    from . import guided

    return guided


def _words_time(seconds: float) -> str:
    return "under 1 s" if seconds < 1 else f"{seconds / 60.0:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


class AutoFixJob:
    """Auto Fix's passes, pumped in slices. `passes` is the report: one entry per pass that ran."""

    def __init__(self, context, plan, report: dict, start: int = 0, limit: float | None = None):
        guided = _g()
        gs = context.scene.curiomesh_guided
        self.plan, self.report, self.start_at = plan, report, start
        self.base_id = guided.state["auto_base"]["id"]
        self.busy, self.status, self.fraction = True, "starting", 0.0
        self.spent = 0.0
        from . import autoretopo

        autoretopo.set_time_limits(gs.time_limits)
        self.limit = float(autoretopo.AUTOFIX_SECONDS if limit is None else limit)
        self.kind = gs.time_limits
        self.gs_cap = gs.organize_cap if gs.organize_cap_use else None
        self.detail = float(getattr(guided.state["auto_base"].get("job"), "detail", 0.0) or 0.0)
        self.look = None
        self.stopped_words = ""
        self.last_draw = 0.0
        self.work = self._drive(context)

    # ---- what the passes work on

    def make_base(self) -> T.Base:
        """A Base of the result on screen now, the way a Tidy Tool's preview builds it (Whole Result)."""
        guided = _g()
        res = guided.state["res"]
        co = np.asarray(res["co"], dtype=np.float64)
        faces = res["faces"]
        src_co, src_faces = tidylive._source()
        project = tidylive._projector(src_co, src_faces) if src_co is not None else None
        facing = tidylive._facer(src_co, src_faces) if src_co is not None else None
        job = guided.state["auto_base"].get("job") if guided.state["auto_base"] else None
        mirror = getattr(job, "mirror", None)
        if self.look is not None:
            self.look.fresh()
        return T.make_base(co, faces, scope=np.ones(len(faces), dtype=bool), project=project, mirror=(mirror[0], mirror[1]) if mirror else None, facing=facing, labels=tidylive._labels(res, co), kinds=tidylive._kinds(res))

    def stopped(self) -> str:
        if self.spent >= self.limit:
            self.stopped_words = f"Auto Fix stopped after {_words_time(self.spent)} (Time Limits: {TIME_WORDS.get(self.kind, 'Normal')}). It was still improving."
            return self.stopped_words
        return ""

    # ---- the work

    def _drive(self, context):
        guided = _g()
        res = guided.state["res"]
        src_co, src_faces = tidylive._source()
        budget = None
        if src_co is not None:
            self.look = _Look(src_co, src_faces)
            self.status = "measuring the look"
            before = yield from self._pump(self.look(np.asarray(res["co"], dtype=np.float64), res["faces"]))
            budget = (before + self.gs_cap) if self.gs_cap is not None else max(1.1 * self.detail, before)
            self.report["ceiling"] = budget
        events = autofix.run(self.make_base, self.plan, {}, self.look, budget, self.stopped, self.start_at)
        for ev in events:
            if self._handle(context, ev):
                return
            yield
        self.busy = False

    def _pump(self, work):
        """Run a generator that yields nothing useful to the end, one step per slice. Returns its value."""
        while True:
            try:
                next(work)
            except StopIteration as done:
                return done.value
            yield

    def _handle(self, context, ev) -> bool:
        """One thing the passes said. True: Auto Fix is over."""
        rep = self.report
        name = ev.trial.name if ev.trial is not None else ""
        if ev.kind == "working":
            if ev.progress >= 0.0:
                self.fraction = ev.progress
                done = max(1, round(ev.progress * ev.spots)) if ev.spots else 0
                self.status = f"Auto Fix, round {ev.round}: {name}, {min(done, ev.spots)} of {ev.spots} spots" if ev.spots else f"Auto Fix, round {ev.round}: {name}"
            else:
                self.status = f"Auto Fix, round {ev.round}: {name}, measuring"
            self._redraw()
            return False
        if ev.kind in ("kept", "left"):
            p = ev.proposal
            kept = ev.kind == "kept"
            if kept:
                self._first_step(context)
                numbers = p.words.split(": ", 1)[1] if ": " in p.words else p.words
                if not tidylive.commit(context, p.cand, f"Auto Fix: {name}", numbers, push=True):
                    kept = False
                    p.reason = "held"
            if kept:
                for sp in p.fixed:
                    fx["seen_kinds"][sp.kind] = fx["seen_kinds"].get(sp.kind, 0) + 1
            rep["passes"].append({"trial": ev.trial, "name": name, "round": ev.round, "kept": kept, "proposal": p})
            rep["res"] = _g().state["res"]
            overlay.tag_redraw()
            return False
        if ev.kind == "stopped":
            rep["stop"], rep["resume"] = ev.words, ev.step
            self.finish("limit")
            return True
        rep["words"] = ev.words
        self.finish("done")
        return True

    def _first_step(self, context) -> None:
        """Before the first pass: the result as it is has a History entry to go back to ("Before Auto Fix")."""
        rep = self.report
        if rep.get("step0") is not None:
            return
        history.before(context)
        rep["step0"] = history.current(context.scene)
        entry = history.steps.at(rep["step0"])
        if entry is not None and entry.name == "Before":
            entry.name = "Before Auto Fix"

    def _redraw(self):
        now = time.perf_counter()
        if now - self.last_draw > REDRAW:
            self.last_draw = now
            overlay.tag_redraw()

    # ---- driving it

    def step(self, budget: float) -> None:
        """Work for about `budget` seconds."""
        began = time.perf_counter()
        try:
            while self.busy and time.perf_counter() - began < budget:
                if _g().state["auto_base"] is None or _g().state["auto_base"]["id"] != self.base_id:
                    self.finish("gone")  # another search replaced the result: nothing to fix here any more
                    break
                try:
                    next(self.work)
                except StopIteration:
                    break
        except Exception as exc:  # a timer must never raise at the user
            self.report["error"] = repr(exc)
            self.finish("error")
        self.spent += time.perf_counter() - began
        if self.busy and not self.report.get("passes_started"):
            self.report["passes_started"] = True

    def finish(self, how: str) -> None:
        if not self.busy and self.report.get("how"):
            return
        self.busy = False
        self.report["how"] = how
        try:
            self.work.close()
        except Exception:
            pass
        self.report["seconds"] = self.report.get("seconds", 0.0) + self.spent
        overlay.tag_redraw()


# --------------------------------------------------------------------------- start, stop, timers


def busy() -> bool:
    job = fx["job"]
    return job is not None and job.busy


def _tick():
    job = fx["job"]
    if job is None or not job.busy:
        return None
    job.step(tidylive.BUDGET)
    return GAP if job.busy else None


def _new_report(context) -> dict:
    guided = _g()
    from . import ladder

    return {"base_id": guided.state["auto_base"]["id"], "shown": ladder.lad["shown"] if ladder.active() else None, "passes": [], "words": "", "how": "", "step0": None, "res": guided.state["res"], "seconds": 0.0, "plan": []}


def start(context, plan=None, force: bool = False, limit: float | None = None) -> bool:
    """Start Auto Fix on the result on screen. The result of a search starts it when Auto Fix is on (`guided`).
    `plan`: [(Trial class, kinds)] instead of the ticked groups (tests). `force`: also in a background Blender, which
    has no timers (tests pump the job with `finish_now`)."""
    guided = _g()
    gs = context.scene.curiomesh_guided
    if busy() or guided.state["auto_base"] is None or guided.state["res"] is None:
        return False
    if not force and (not gs.auto_fix or bpy.app.background):
        return False
    stop(context, quiet=True)
    plan = autofix.plan(_ticks(context)) if plan is None else plan
    report = _new_report(context)
    report["plan"] = plan
    fx["report"], fx["show"] = report, None
    if not plan:
        report.update(words=autofix.NOTHING, how="done")
        overlay.tag_redraw()
        return True
    job = fx["job"] = AutoFixJob(context, plan, report, 0, limit)
    _arm()
    return True


def _arm() -> None:
    if not bpy.app.background and not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.3)  # after the search's own History step is in


def keep_going(context) -> bool:
    """The time limit ended Auto Fix: go on from the next pass, with the time limit again."""
    rep = fx["report"]
    if busy() or rep is None or rep.get("how") != "limit":
        return False
    rep.update(how="", stop="", words="")
    fx["job"] = AutoFixJob(context, rep["plan"], rep, rep.get("resume", 0))  # the time limit as it is now (Time Limits may have changed)
    _arm()
    overlay.tag_redraw()
    return True


def stop(context=None, quiet: bool = False) -> bool:
    """End Auto Fix now. Every pass kept so far stays, in History too."""
    job = fx["job"]
    if job is None or not job.busy:
        return False
    job.finish("user")
    if not quiet:
        job.report["words"] = "Auto Fix stopped. It kept what it had fixed so far."
    return True


def finish_now(context, seconds: float = 3600.0) -> None:
    """Run the job to its end here, for scripts and tests."""
    ends = time.perf_counter() + seconds
    while busy() and time.perf_counter() < ends:
        fx["job"].step(0.5)


def undo_all(context) -> bool:
    """Back to the result as it was before Auto Fix, in one History step. Ctrl+Z brings the fixes back."""
    rep = fx["report"]
    if rep is None or busy() or rep.get("step0") is None:
        return False
    for i, entry in enumerate(history.steps.entries):
        if entry.step == rep["step0"]:
            pg = context.scene.curiomesh_autofix
            pg.seen = True
            return history.jump(context, i)
    return False


# --------------------------------------------------------------------------- the report


def lines(rep: dict) -> list:
    """The readout's per-kind lines: [{"tool", "text", "at", "kept"}], fixed ones first, in the order they were made."""
    fixed, left = {}, {}
    for entry in rep["passes"]:
        p, trial = entry["proposal"], entry["trial"]
        if p is None:
            continue
        if entry["kept"]:
            for sp in p.fixed:
                fixed.setdefault((entry["name"], trial, sp.kind), []).append(sp.at)
            lost = [(sp, why) for sp, why in p.left]
        else:
            lost = [(sp, p.reason or "held") for sp in p.fixed] + list(p.left)
        for sp, why in lost:
            left.setdefault((entry["name"], trial, sp.kind, why), []).append(sp.at)
    out = []
    for (name, trial, kind), at in fixed.items():
        out.append({"tool": name, "text": "Fixed " + autofix.kind_words(trial, kind, len(at)), "at": at, "kept": True})
    for (name, trial, kind, why), at in left.items():
        out.append({"tool": name, "text": f"Left {autofix.kind_words(trial, kind, len(at))} as they were ({autofix.WHY.get(why, why)})", "at": at, "kept": False})
    return out


def head_words(rep: dict) -> str:
    kept = [e for e in rep["passes"] if e["kept"]]
    if not kept:
        return rep["words"] or autofix.NOTHING
    spots = sum(len(e["proposal"].fixed) for e in kept)
    first, last = kept[0]["proposal"].before, kept[-1]["proposal"].after
    parts = [f"Auto Fix: {spots} spot{'s' if spots != 1 else ''} in {len(kept)} fix{'es' if len(kept) != 1 else ''}."]
    if first.tidy is not None and last.tidy is not None:
        parts.append(f"Tidiness {first.tidy:.0f} -> {last.tidy:.0f},")
    if first.look is not None and last.look is not None:
        parts.append(f"look change {last.look - first.look:+.1f}%")
    text = " ".join(parts).rstrip(",")
    return text


def kinds_found() -> list:
    """Plain words for the kinds the ticked tools fix, most often found first (from the last report), then as listed."""
    kinds = []
    for cls, only in autofix.plan(set(autofix.GROUPS)):
        for key, _one, many in cls.kinds:
            if only is None or key in only:
                kinds.append((key, many))
    seen = fx["seen_kinds"]
    kinds.sort(key=lambda k: -seen.get(k[0], 0))
    return [many for _key, many in kinds]


def summary_words() -> str:
    kinds = kinds_found()
    if not kinds:
        return "Fixes the spots the tidy tools find"
    first = ", ".join(kinds[:2])
    more = len(kinds) - 2
    return f"Fixes {first}" + (f" and {more} more" if more > 0 else "")


# --------------------------------------------------------------------------- panel


def draw_option(layout, context, gs) -> None:
    """Auto's options: one row, a tick and the popover with the seven groups."""
    row = layout.row(align=True)
    row.prop(gs, "auto_fix", text="Auto Fix", toggle=False)
    row.popover(panel="CURIOMESH_PT_pop_autofix", text="", icon="DOWNARROW_HLT")
    if gs.auto_fix:
        sub = layout.row()
        sub.active = False
        overlay.wrap_label(sub, summary_words(), width=WIDTH, limit=2)


class CURIOMESH_PT_pop_autofix(bpy.types.Panel):
    bl_label = "Auto Fix"
    bl_idname = "CURIOMESH_PT_pop_autofix"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"  # drawn only where layout.popover() opens it
    bl_ui_units_x = 18

    def draw(self, context):
        layout, pg = self.layout, context.scene.curiomesh_autofix
        overlay.wrap_label(layout, "After the search, each tool fixes the spots it is made for. It keeps a fix only when the quads get better and the look stays within Most Detail Loss. Every fix is a step in History.", width=44, limit=5)
        layout.separator()
        for key in autofix.GROUPS:
            label, what = GROUP_TEXT[key]
            col = layout.column(align=True)
            col.prop(pg, key, text=label)
            sub = col.row()
            sub.active = False
            sub.label(text=what)


def _same_size(rep: dict) -> bool:
    """False when the size on screen is another rung of the ladder than the one Auto Fix ran on."""
    from . import ladder

    return not (ladder.active() and ladder.lad["shown"] != rep["shown"])


def draw_readout(layout, context) -> None:
    """Under the verdict: the progress box with Stop while Auto Fix works, then what it did."""
    guided = _g()
    job, rep = fx["job"], fx["report"]
    if rep is None or guided.state["auto_base"] is None or rep["base_id"] != guided.state["auto_base"]["id"]:
        return
    pg = context.scene.curiomesh_autofix
    box = layout.box()
    if job is not None and job.busy:
        box.label(text="Auto Fix is working", icon="TIME")
        guided.draw_progress(box, job.fraction, job.status)
        box.operator("curiomesh.autofix_stop", icon="CANCEL")
        sub = box.row()
        sub.active = False
        sub.label(text="Stop keeps what it fixed so far.")
        return
    if not _same_size(rep):
        overlay.wrap_label(box, "Auto Fix ran on the size before.", icon="INFO", width=WIDTH, limit=2)
        box.operator("curiomesh.autofix_run_here", icon="PLAY")
        return
    rows = lines(rep)
    head = box.column(align=True)
    overlay.wrap_label(head, head_words(rep), icon="CHECKMARK" if any(e["kept"] for e in rep["passes"]) else "INFO", width=WIDTH, limit=3)
    kept = any(e["kept"] for e in rep["passes"])
    if rep.get("stop"):
        overlay.wrap_label(box, rep["stop"], icon="TIME", width=WIDTH, limit=4)
        box.operator("curiomesh.autofix_keep_going", icon="PLAY")
    elif rep.get("error"):
        overlay.wrap_label(box, "Auto Fix hit a problem and stopped. What it fixed so far is kept.", icon="ERROR", width=WIDTH, limit=3)
    if kept:
        row = box.row(align=True)
        row.operator("curiomesh.autofix_show", text="Show", icon="VIEWZOOM").line = next((i for i, r in enumerate(rows) if r["kept"]), 0)
        row.operator("curiomesh.autofix_undo", text="Undo Auto Fix", icon="LOOP_BACK")
        if not pg.seen:
            sub = box.row()
            sub.active = False
            overlay.wrap_label(sub, "Each fix is a step in History. Ctrl+Z takes them back one by one.", width=WIDTH, limit=2)
    if rows:
        box.prop(pg, "open", text=f"{len(rows)} line{'s' if len(rows) != 1 else ''}", icon="DOWNARROW_HLT" if pg.open else "RIGHTARROW", emboss=False)
        if pg.open:
            col = box.column(align=True)
            for i, r in enumerate(rows):
                row = col.row(align=True)
                row.label(text=r["text"])
                row.operator("curiomesh.autofix_show", text="Show").line = i


# --------------------------------------------------------------------------- Show


def _show_points():
    rep, show = fx["report"], fx["show"]
    if rep is None or show is None:
        return None
    rows = lines(rep)
    if not 0 <= show["line"] < len(rows):
        return None
    return np.asarray(rows[show["line"]]["at"], dtype=np.float64).reshape(-1, 3)


def _draw_show() -> None:
    show = fx["show"]
    pts = _show_points()
    context = bpy.context
    if show is None or pts is None or context.region is None or context.region_data is None:
        return
    try:
        import gpu

        obj = tidylive._preview()
        if obj is None:
            return
        px, front = tidylive._to_screen(context.region, context.region_data, obj.matrix_world, pts)
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        shader.bind()
        gpu.state.blend_set("ALPHA")
        scale = tidylive._ui_scale()
        for i, ((x, y), ok) in enumerate(zip(px.tolist(), front.tolist())):
            if ok:
                tidylive._dashed(shader, x, y, (14.0 if i == show["spot"] else 9.0) * scale)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


def _frame(context) -> None:
    pts = _show_points()
    if pts is None or not len(pts):
        return
    obj = tidylive._preview()
    area, region = tidylive._view(context)
    if obj is None or region is None:
        return
    rv3d = region.data
    at = pts[fx["show"]["spot"] % len(pts)]
    rv3d.view_location = obj.matrix_world @ mathutils.Vector(at.tolist())
    rv3d.view_distance = min(rv3d.view_distance, max(0.3 * float(obj.dimensions.length), 1e-3))
    area.header_text_set(f"Spot {fx['show']['spot'] % len(pts) + 1} of {len(pts)}. Show again for the next, Esc to stop.")
    area.tag_redraw()


def _show_off(context=None) -> None:
    fx["show"], fx["show_op"] = None, False
    if fx["show_handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(fx["show_handle"], "WINDOW")
        except Exception:
            pass
        fx["show_handle"] = None
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.header_text_set(None)
    except Exception:
        pass
    overlay.tag_redraw()


class CURIOMESH_OT_autofix_stop(bpy.types.Operator):
    bl_idname = "curiomesh.autofix_stop"
    bl_label = "Stop"
    bl_description = "Stops Auto Fix now and keeps every fix made so far. Ctrl+Z takes them back one by one"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return busy()

    def execute(self, context):
        return {"FINISHED"} if stop(context) else {"CANCELLED"}


class CURIOMESH_OT_autofix_keep_going(bpy.types.Operator):
    bl_idname = "curiomesh.autofix_keep_going"
    bl_label = "Keep going"
    bl_description = "Carries on from the pass where Auto Fix stopped, with the same time limit again"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        rep = fx["report"]
        return not busy() and rep is not None and rep.get("how") == "limit"

    def execute(self, context):
        return {"FINISHED"} if keep_going(context) else {"CANCELLED"}


class CURIOMESH_OT_autofix_undo(bpy.types.Operator):
    bl_idname = "curiomesh.autofix_undo"
    bl_label = "Undo Auto Fix"
    bl_description = "Puts the result back as it was before Auto Fix, in one step. Ctrl+Z brings the fixes back"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        rep = fx["report"]
        return not busy() and rep is not None and rep.get("step0") is not None

    def execute(self, context):
        return {"FINISHED"} if undo_all(context) else {"CANCELLED"}


class CURIOMESH_OT_autofix_run_here(bpy.types.Operator):
    bl_idname = "curiomesh.autofix_run_here"
    bl_label = "Run on this one"
    bl_description = "Runs Auto Fix on the size on screen now. Each fix is a History step"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return not busy() and _g().state["auto_base"] is not None

    def execute(self, context):
        return {"FINISHED"} if start(context, force=True) else {"CANCELLED"}


class CURIOMESH_OT_autofix_show(bpy.types.Operator):
    bl_idname = "curiomesh.autofix_show"
    bl_label = "Show"
    bl_description = "Shows where this was found, one spot at a time. Press again for the next, Esc to stop"
    bl_options = {"INTERNAL"}

    line: bpy.props.IntProperty(name="Line", description="Which line of the Auto Fix result to show", default=0, options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        rep = fx["report"]
        rows = lines(rep) if rep is not None else []
        if 0 <= properties.line < len(rows):
            return f"Shows where this was found, one spot at a time. Fixed by {rows[properties.line]['tool']}. Press again for the next, Esc to stop"
        return cls.bl_description

    @classmethod
    def poll(cls, context):
        return fx["report"] is not None and not busy()

    def invoke(self, context, event):
        pg = context.scene.curiomesh_autofix
        pg.seen = True
        show = fx["show"]
        fx["show"] = {"line": self.line, "spot": 0 if show is None or show["line"] != self.line else show["spot"] + 1}
        pts = _show_points()
        if pts is None or not len(pts):
            fx["show"] = None
            return {"CANCELLED"}
        fx["show"]["spot"] %= len(pts)
        _frame(context)
        if fx["show_op"]:
            return {"FINISHED"}  # the modal already runs: it only needed the next spot
        fx["show_op"] = True
        fx["show_handle"] = bpy.types.SpaceView3D.draw_handler_add(_draw_show, (), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not fx["show_op"] or fx["show"] is None:
            _show_off()
            return {"CANCELLED"}
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            _show_off()
            return {"CANCELLED"}
        return {"PASS_THROUGH"}

    def cancel(self, context):
        _show_off()


# --------------------------------------------------------------------------- handlers, keys, registration


@persistent
def _curiomesh_autofix_undo(*_args):
    try:
        if busy():
            stop(quiet=True)  # Ctrl+Z while it works: stop first; the undo itself undoes the last kept pass
            fx["report"]["words"] = "Auto Fix stopped by Undo. It kept the fixes before that."
        _show_off()
    except Exception as exc:  # a handler must never stop Blender's undo
        overlay.LAST_DRAW_ERROR = f"Auto Fix: {exc!r}"


@persistent
def _curiomesh_autofix_load(*_args):
    clear()


def clear() -> None:
    job = fx["job"]
    if job is not None and job.busy:
        job.finish("user")
    fx.update(job=None, report=None)
    _show_off()
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)


HANDLERS = ((bpy.app.handlers.undo_post, _curiomesh_autofix_undo), (bpy.app.handlers.redo_post, _curiomesh_autofix_undo), (bpy.app.handlers.load_post, _curiomesh_autofix_load))
CLASSES = (CURIOMESH_PG_autofix, CURIOMESH_PT_pop_autofix, CURIOMESH_OT_autofix_stop, CURIOMESH_OT_autofix_keep_going, CURIOMESH_OT_autofix_undo, CURIOMESH_OT_autofix_run_here, CURIOMESH_OT_autofix_show)
_keys: list = []


def _drop_handlers() -> None:
    for handlers, fn in HANDLERS:
        for old in [h for h in handlers if getattr(h, "__name__", "") == fn.__name__]:
            handlers.remove(old)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_autofix = bpy.props.PointerProperty(type=CURIOMESH_PG_autofix)
    _drop_handlers()
    for handlers, fn in HANDLERS:
        handlers.append(fn)
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon if wm is not None else None
    if kc is not None:  # Esc over the 3D view stops Auto Fix; the operator's poll lets Esc through when nothing runs
        km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        _keys.append((km, km.keymap_items.new("curiomesh.autofix_stop", "ESC", "PRESS")))


def unregister() -> None:
    clear()
    _drop_handlers()
    for km, kmi in _keys:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _keys.clear()
    if hasattr(bpy.types.Scene, "curiomesh_autofix"):
        del bpy.types.Scene.curiomesh_autofix
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
