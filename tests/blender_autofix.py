# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Auto Fix after Start in background Blender (AUTO_FIX.md 6.2): Start, then per model one table row, then Ctrl+Z and
Ctrl+Shift+Z pass by pass, a History jump back, Stop mid-run, the time limit with Keep going, and the per-tool button.

    blender -b --factory-startup --python tests/blender_autofix.py -- [--addon <folder>] [--models bust,lion,crane,barrier]

Models are the owner's free test meshes (bench/assets/uvm.py). Only the models named run (default: bust). While no tool
has `spots` yet (builders C, D and E are still to land), a toy tool on the same code path stands in, and says so.
On the owner's PC start Blender only through bench/rough/lowblender.sh. Exits 1 on any failure.
"""

import importlib
import os
import sys
import time

import bpy
import numpy as np

ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ADDON = os.path.abspath(ARGV[ARGV.index("--addon") + 1]) if "--addon" in ARGV else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ARGV[ARGV.index("--models") + 1].split(",") if "--models" in ARGV else ["bust"]
NAMES = {"bust": "marble_bust_01", "lion": "lion_head", "crane": "overhead_crane", "barrier": "concrete_road_barrier"}
BUTTON = {"bust": ("relax", "SquareUp"), "barrier": ("hardtidy", "TrueUpEdges"), "crane": ("loopspoles", "EvenTheBand"), "lion": ("loopspoles", "OpenUpPoles")}
sys.path.insert(0, os.path.dirname(ADDON))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "bench", "assets")))
import uvm  # noqa: E402

pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
guided, H, TL, AL, ladder, autoretopo = pkg.guided, pkg.history, pkg.tidylive, pkg.autofix_live, pkg.ladder, pkg.autoretopo
N = pkg.__name__
autofix = importlib.import_module(N + ".triadq.autofix")
TT = importlib.import_module(N + ".triadq.trial")
TTools = importlib.import_module(N + ".triadq.tidytools")
FAILS = []


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def res_co():
    return np.asarray(guided.state["res"]["co"], dtype=np.float64).copy()


def faces_n():
    return len(guided.state["res"]["faces"])


def step():
    return H.current(bpy.context.scene)


def names():
    return [e.name for e in H.steps.entries]


# --------------------------------------------------------------------------- a stand-in tool until the real ones have spots


class Toy(TTools.SmoothCorners):
    """Even edge lengths round the corners with the most uneven ones: a positions-only tool on the Auto Fix path."""

    name = "Even Corners (toy)"
    kinds = (("lumpy", "lumpy patch", "lumpy patches"),)
    own, least = "lumpy", 1e-6
    defaults = {"strength": 0.9, "steps": 4}

    def spots(self, base):
        faces, co = base.faces, base.co
        star = [[] for _ in range(len(co))]
        for f in np.flatnonzero(base.scope).tolist():
            for v in faces[f]:
                star[v].append(f)
        free = np.flatnonzero(base.free)
        score = np.zeros(len(co))
        for v in free.tolist():
            score[v] = self._spread(co, faces, star[v])
        yield 0.5
        used, out = set(), []
        for v in free[np.argsort(-score[free])].tolist():
            corners = {c for f in star[v] for c in faces[f]}
            if score[v] <= 0 or corners & used or len(out) >= 12:  # a toy: a dozen spots is plenty to test with
                continue
            used |= corners
            out.append(TT.Spot("lumpy", np.asarray(star[v], dtype=np.int64), co[v].copy(), float(score[v])))
        return out

    @staticmethod
    def _spread(co, faces, ids):
        lengths = [np.linalg.norm(co[f[i]] - co[f[(i + 1) % len(f)]]) for f in (faces[k] for k in ids) for i in range(len(f))]
        return float(np.std(lengths) / max(np.mean(lengths), 1e-12)) if lengths else 0.0

    def own_number(self, base, co, faces, spot):
        return self._spread(co, faces, np.asarray(spot.faces).tolist())


class Toy2(Toy):
    name = "Even Corners again (toy)"


class Toy3(Toy):
    name = "Even Corners once more (toy)"


def real_plan():
    return autofix.plan(set(autofix.GROUPS))


# --------------------------------------------------------------------------- undo as the other Blender tests do it


def can_undo():
    bpy.ops.ed.undo_push(message="probe")
    bpy.ops.mesh.primitive_cube_add()
    probe = bpy.context.object.name
    can = True
    try:
        bpy.ops.ed.undo()
        can = probe not in bpy.data.objects
    except RuntimeError:
        can = False
    for o in list(bpy.data.objects):
        if o.name.startswith("Cube"):
            bpy.data.objects.remove(o, do_unlink=True)
    return can


def undo(can):
    if can:
        bpy.ops.ed.undo()
    else:
        H.settings()["step"] = H.steps.entries[H.steps.index_of(step()) - 1].step
        H._curiomesh_history_undo(bpy.context.scene)


def redo(can, to):
    if can:
        bpy.ops.ed.redo()
    else:
        H.settings()["step"] = to
        H._curiomesh_history_undo(bpy.context.scene)


def settle_work():
    while guided.state["work"] is not None:
        runner = guided.state["work"]["runner"]
        while runner.busy:
            runner.step(0.05)
        before = guided.state["work"]
        guided._collect_work(bpy.context)
        if guided.state["work"] is before:
            break


def tick_to_end(limit=3600.0):
    """Pump the job the way Blender's timer does: `_tick` until it stops asking to be called."""
    ends, t0 = time.perf_counter() + limit, time.perf_counter()
    seen = len(AL.fx["report"]["passes"]) if AL.fx["report"] else 0
    beat, said = t0, ""
    while AL.busy() and time.perf_counter() < ends:
        if AL._tick() is None:
            break
        job = AL.fx["job"]
        if job is not None and time.perf_counter() - beat > 30.0 and job.status != said:  # what a long pass is doing
            beat, said = time.perf_counter(), job.status
            print(f"STATUS {beat - t0:6.1f}s {said}", flush=True)
        passes = AL.fx["report"]["passes"] if AL.fx["report"] else []
        for e in passes[seen:]:
            p = e["proposal"]
            print(f"PASS-LOG {time.perf_counter() - t0:6.1f}s round {e['round']} {e['name']}: {'kept' if e['kept'] else 'left (' + str(p.reason) + ')'} spots {len(p.fixed)}/{len(p.fixed) + len(p.left)} tidy {p.before.tidy} -> {p.after.tidy}", flush=True)
        seen = len(passes)


class Fake:
    """A recording layout: every call works, and the operators and properties it was asked for are checked after."""

    def __init__(self, log=None):
        self.log = log if log is not None else {"ops": set(), "props": [], "text": []}
        self.enabled = self.active = self.alert = True
        self.scale_y = self.scale_x = 1.0
        self.alignment = ""

    def operator(self, idname, text="", **kw):
        self.log["ops"].add(idname)
        self.log["text"].append(text)
        return type("Op", (), {})()

    def prop(self, data, name, text="", **kw):
        self.log["props"].append((data, name))
        return None

    def label(self, text="", **kw):
        self.log["text"].append(text)

    def popover(self, panel="", text="", **kw):
        self.log["text"].append(panel)

    def progress(self, **kw):
        self.log["text"].append(kw.get("text", ""))

    def separator(self, **kw):
        pass

    def __getattr__(self, name):  # box, column, row, grid_flow, ...
        return lambda *a, **k: self


def draw_smoke(what):
    """Draw the Auto Fix pieces against the recording layout: no draw error, every operator and property exists."""
    ctx, gs = bpy.context, bpy.context.scene.curiomesh_guided
    log = {"ops": set(), "props": [], "text": []}
    try:
        AL.draw_readout(Fake(log), ctx)
        AL.draw_option(Fake(log), ctx, gs)
        panel = type("P", (), {"layout": Fake(log)})()
        AL.CURIOMESH_PT_pop_autofix.draw(panel, ctx)
    except Exception as exc:
        ok(False, f"{what}: drawing raised {exc!r}")
        return log
    missing = [o for o in log["ops"] if getattr(bpy.ops.curiomesh, o.split(".")[1], None) is None]
    bad = [(d, n) for d, n in log["props"] if n not in d.bl_rna.properties]
    ok(not missing and not bad, f"{what}: the readout, option row and popover draw; missing {missing} {bad}")
    return log


def kept_passes(rep):
    return [e for e in rep["passes"] if e["kept"]]


def run_model(key):
    print(f"--- {key}", flush=True)
    t0 = time.perf_counter()
    src = uvm.load_model(NAMES[key])
    uvm.run_auto(guided, src, 6.0)
    settle_work()
    ok(guided.state["res"] is not None, f"{key}: a result ({faces_n()} faces, {time.perf_counter() - t0:.0f} s)")
    gs = bpy.context.scene.curiomesh_guided
    ok(gs.auto_fix is True, "Auto Fix is on by default")
    ok(not AL.busy(), "a background Blender has no timer, so the search result does not start a job by itself")
    can = can_undo()
    bpy.context.view_layer.objects.active = bpy.data.objects.get("Model") or bpy.context.view_layer.objects.active
    H.steps.clear()
    H.settings()["step"] = 0
    bpy.ops.ed.undo_push(message="test start")

    real = real_plan()
    toy = not real
    plan = real or [(Toy, None), (Toy2, None)]
    if toy:
        print("INFO no tool has spots on this branch yet: a toy tool stands in on the same path", flush=True)
    co0, faces0 = res_co(), faces_n()
    rec0 = len(H.steps.entries)

    # --- Start, pumped as the timer does
    ok(AL.start(bpy.context, plan=plan, force=True), "Auto Fix starts")
    ok(AL.busy() and TL.busy(), "while it works, tidylive.busy() is True (the ladder and Tidy Up wait)")
    ladder.finish(timeout=0.01)  # must return at once while Auto Fix works, not spin
    began = time.perf_counter()
    tick_to_end()
    rep = AL.fx["report"]
    limits = 0
    while rep["how"] == "limit":  # Time Limits Normal hit on a big model: the stop line, then Keep going
        limits += 1
        ok("Time Limits: Normal" in rep["stop"] and "still improving" in rep["stop"], f"the time limit says so: {rep['stop']}")
        if not AL.keep_going(bpy.context):  # Keep going clears the stop line, so it is read first
            break
        tick_to_end()
        rep = AL.fx["report"]
    seconds = time.perf_counter() - began
    ok(not AL.busy() and rep["how"] == "done", f"it ends by itself: {rep['how']}, {rep['words']} (time limit hit {limits} times)")
    kept = kept_passes(rep)
    ok(all(e["proposal"].after.bad <= e["proposal"].before.bad for e in kept), "bad faces never higher after a kept pass")
    first, last = (kept[0]["proposal"].before, kept[-1]["proposal"].after) if kept else (autofix.Scores(), autofix.Scores())

    def pair(a, b, fmt="{:.1f}"):
        return "-" if a is None or b is None else f"{fmt.format(a)} -> {fmt.format(b)}"

    look = "-" if first.look is None or last.look is None else f"{last.look - first.look:+.2f}% (ceiling {rep.get('ceiling', float('nan')):.2f}%, ends at {last.look:.2f}%)"
    print(f"ROW {key}: Tidiness {pair(first.tidy, last.tidy)}, Loop Quality {pair(first.loops, last.loops)}, poles {pair(first.poles, last.poles, '{:.0f}')}, bad faces {pair(first.bad, last.bad, '{:.0f}')}, look change {look}, faces {faces_n()}, passes kept {len(kept)} of {len(rep['passes'])}, {seconds:.1f} s", flush=True)
    log = draw_smoke("after Start")
    print("INFO readout:", [t for t in log["text"] if t][:8], flush=True)
    if rep.get("ceiling") is not None and last.look is not None:
        ok(last.look <= rep["ceiling"] + 1e-6, "the look stays inside the budget")
    fixes = [n for n in names()[rec0:] if n.startswith("Auto Fix: ")]
    ok(len(fixes) == len(kept), f"one History entry per kept pass: {len(fixes)} of {len(kept)}")
    ok(any(n.startswith("Before") or n == "New result" or n.startswith("Back to") for n in names()) or not kept, f"the result before has an entry: {names()[:3]}")
    co_end = res_co()
    if toy:
        ok(len(kept) >= 1, "the toy pass was kept (the path works end to end)")
    if not kept:
        print(f"INFO {key}: nothing kept, the undo checks need a pass", flush=True)
        return
    # --- Ctrl+Z one pass back, Ctrl+Shift+Z forward
    top = step()
    state_before_last = H.steps.entries[H.steps.index_of(top) - 1].state["res"]["co"]
    undo(can)
    ok(np.allclose(res_co(), state_before_last), "Ctrl+Z puts back the result before the last kept pass")
    ok(len(kept) == 1 or not np.allclose(res_co(), co_end) or np.allclose(state_before_last, co_end), "one pass back, not all")
    redo(can, top)
    ok(np.allclose(res_co(), co_end), "Ctrl+Shift+Z brings the pass back (same points)")
    # --- Undo Auto Fix: one jump to the result before
    n = len(H.steps.entries)
    ok(AL.undo_all(bpy.context), "Undo Auto Fix")
    ok(faces_n() == faces0 and np.allclose(res_co(), co0), "the Auto result is back exactly")
    ok(len(H.steps.entries) == n + 1, "the jump is one new step")
    undo(can)
    ok(np.allclose(res_co(), co_end), "Ctrl+Z after Undo Auto Fix brings every pass back")
    AL.undo_all(bpy.context)

    # --- Stop mid-run keeps the last kept pass and leaves nothing behind
    timers, handles = bpy.app.timers.is_registered(AL._tick), len(TL.live["handles"])
    ok(AL.start(bpy.context, plan=plan, force=True), "a second run starts")
    job = AL.fx["job"]
    draw_smoke("while working")
    ends = time.perf_counter() + 600
    while AL.busy() and not kept_passes(AL.fx["report"]) and time.perf_counter() < ends:
        job.step(0.05)
    if AL.busy():
        after_first = res_co()
        passes = len(AL.fx["report"]["passes"])
        ok(AL.stop(bpy.context), "Stop mid-run")
        ok(not AL.busy() and not TL.busy(), "stopped: nothing is busy")
        ok(np.allclose(res_co(), after_first) and kept_passes(AL.fx["report"]), "the result is the last kept pass")
        ok(len(AL.fx["report"]["passes"]) == passes, "no pass was added after Stop")
    else:
        print("INFO the run ended before Stop could be tested on this model", flush=True)
    ok(bpy.app.timers.is_registered(AL._tick) == timers and len(TL.live["handles"]) == handles and AL.fx["show_handle"] is None, "no timer and no draw handler left")
    AL.undo_all(bpy.context)

    # --- the time limit: the stop line, then Keep going continues
    autoretopo.AUTOFIX_SECONDS = 0.01
    ok(AL.start(bpy.context, plan=plan, force=True), "a run with a tiny time limit starts")
    tick_to_end()
    rep = AL.fx["report"]
    ok(rep["how"] == "limit" and "Time Limits: Normal" in rep["stop"] and "still improving" in rep["stop"], f"the stop line: {rep.get('stop')}")
    before_passes = len(rep["passes"])
    autoretopo.AUTOFIX_SECONDS = 1e9
    ok(AL.keep_going(bpy.context), "Keep going")
    tick_to_end()
    ok(len(AL.fx["report"]["passes"]) > before_passes and AL.fx["report"]["how"] == "done", "it carried on from the next pass and ended by itself")
    autoretopo.AUTOFIX_SECONDS = 180.0
    AL.undo_all(bpy.context)

    # --- the per-tool button: Enter (commit), Ctrl+Z
    module, name = BUTTON[key]
    cls = getattr(importlib.import_module(f"{N}.triadq.{module}"), name, None)
    tool_op = "curiomesh.tool_" + {"SquareUp": "square_up", "TrueUpEdges": "true_up_edges", "EvenTheBand": "even_band", "OpenUpPoles": "open_poles"}[name]
    real_button = cls is not None and cls.auto and cls.spots is not TT.Trial.spots
    rec = len(H.steps.entries)
    before = res_co()
    if real_button:
        r = bpy.ops.curiomesh.tool_every("EXEC_DEFAULT", tool=tool_op)
        ok(r == {"FINISHED"} or r == {"CANCELLED"}, f"{name}: the per-tool button ran: {r}")
        kept_one = r == {"FINISHED"}
    else:
        s = TL.Session(bpy.context, autofix.every(Toy)(), {}, "WHOLE")
        s.finish()
        ok(not s.error and s.cand is not None, f"{name} has no spots yet: the toy through EverySpot ({s.cand.refusal if s.cand is not None and s.cand.refusal else 'ready'})")
        kept_one = s.ready_to_keep() and TL.commit(bpy.context, s.cand, s.trial.name, "", push=True)
    if kept_one:
        ok(len(H.steps.entries) == rec + 1, "Enter on the button is one History entry")
        undo(can)
        ok(np.allclose(res_co(), before), "Ctrl+Z takes the per-tool run back")
    AL.fx.update(report=None)


for key in MODELS:
    run_model(key)

print("AUTOFIX OK" if not FAILS else f"AUTOFIX FAILED: {FAILS}", flush=True)
sys.exit(0 if not FAILS else 1)
