# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: Fix Bad Spots and Try Every Method
"""Fix Bad Spots marking, the pinned card, Keep Original on marked parts, and Try Every Method as one Ctrl+Z step,
in background Blender on a subdivided monkey.

    blender -b --factory-startup --python tests/blender_review_spots.py -- --addon <folder of the add-on, a valid module name>

On the owner's PC start Blender only through bench/rough/lowblender.sh. Exits 1 on any failure.
"""

import importlib
import os
import sys

import bpy

ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ADDON = os.path.abspath(ARGV[ARGV.index("--addon") + 1]) if "--addon" in ARGV else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ADDON))
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
guided, H, segments = pkg.guided, pkg.history, pkg.segments
SG, IP = segments.segment_goals, segments.issue_panel
FAILS = []
ICONS = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


class Op:
    pass


class Rec:
    """A UILayout stand-in that checks icons, properties and operators and records the words."""

    def __init__(self, log):
        self.log = log

    def _sub(self, *a, **k):
        return Rec(self.log)

    row = column = box = split = _sub

    def panel(self, idname, default_closed=False):
        return Rec(self.log), Rec(self.log)

    def label(self, text="", icon="NONE", **k):
        assert icon in ICONS, icon
        self.log.append(text)

    def separator(self, *a, **k):
        pass

    def progress(self, *a, **k):
        pass

    def prop(self, data, name, text=None, icon="NONE", **k):
        assert name in data.bl_rna.properties, name

    def operator(self, idname, text="", icon="NONE", **k):
        assert icon in ICONS, icon
        group, _, op = idname.partition(".")
        getattr(getattr(bpy.ops, group), op).get_rna_type()
        self.log.append(text)
        return Op()


def run_jobs(limit=600):
    """What the add-on's timers do, by hand: step the running job, collect it, tick Try Every Method."""
    for _ in range(limit):
        work = guided.state["work"]
        if work is not None:
            runner = work["runner"]
            while runner.busy:
                runner.step(0.05)
            guided._collect_work(bpy.context)
        if segments.seg.get("tryall") is None:
            return True
        SG._tick()
    return False


for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_monkey_add()
mod = bpy.context.object.modifiers.new("sub", "SUBSURF")
mod.levels = 2
bpy.ops.object.modifier_apply(modifier=mod.name)
gs = bpy.context.scene.curiomesh_guided
gs.scope, gs.auto_update = "AUTO", False
ok(bpy.ops.curiomesh.guided_auto_start() == {"FINISHED"}, "Auto starts")
for _ in range(60):
    work = guided.state["work"]
    if work is None:
        break
    runner = work["runner"]
    while runner.busy:
        runner.step(0.05)
    guided._collect_work(bpy.context)
    if guided.state["work"] is work or guided.state["work"] is None or not guided.state["work"]["runner"].busy:
        break
segments.ensure()
ok(segments.ready() and len(segments.seg["parts"]) >= 1, f"{len(segments.seg['parts'])} parts")

# the card is a panel class the panel module registers: it registers next to its parent
bpy.utils.register_class(SG.CURIOMESH_PT_seg_card)
ok(hasattr(bpy.types, "CURIOMESH_PT_seg_card"), "CURIOMESH_PT_seg_card registers under CURIOMESH_PT_guided")

# a bad spot on part 0, so there is something to walk and mark
res = guided.state["res"]
faces0 = [int(i) for i in range(len(res["faces"])) if int(segments.seg["piece"][i]) == 0][:3]
res["issue_spots"] = [{"faces": faces0, "center": [0.0, 0.0, 0.0], "radius": 0.5, "part": 0, "kinds": {"folded_over": len(faces0)}}]
IP.ui["res"] = None
ok(len(IP.spots()) == 1 and segments.bad_parts() == [0], "one bad spot on part 0")

# the mode: bad faces shown for the session, then put back
gs.seg_bad = False
segments.seg["reviewing"] = True
SG._set_mode(bpy.context, "SPOTS")
ok(segments.seg["review_mode"] == "SPOTS" and gs.seg_bad is True and IP.ui["cur"] == 0, "Fix Bad Spots shows bad faces and goes to the worst spot")
IP.mark("ALL")
ok(segments.seg["sel"] == {0}, "A marks every bad part")
IP.mark("CURRENT")
ok(segments.seg["sel"] == set(), "marking the current spot again unmarks it")
IP.mark("CURRENT")
IP.focus_part(0)
log = []
SG.draw_card(Rec(log), bpy.context)
ok(any("1 marked" in t for t in log) and "Redo Marked" in log, f"the card draws the marked parts and Redo Marked: {log[:6]}")
SG._set_mode(bpy.context, "PARTS")
ok(gs.seg_bad is False, "leaving puts Show Bad Faces back")
SG._set_mode(bpy.context, "SPOTS")

# fixing the marked parts: Keep Original is a plan, a preview-less execute here, one History step
plan = IP.original_plan([0], 0)
ok(plan["tries"] and plan["name"] == "Keep Original Here", "Keep Original plan for the marked part")
plan = IP.redo_plan([0], 0)
ok("sizes" in plan and plan["name"] == "Redo This Spot", "Redo plan for the marked part")
segments.select([0])
steps0 = len(H.steps.entries)
ok(bpy.ops.curiomesh.issue_keep_original(marked=True) == {"FINISHED"}, "Keep Original on Marked runs")
ok(segments.seg["parts"][0]["method"] == segments.ORIGINAL, "the part shows your own faces")
ok(len(H.steps.entries) - steps0 == 1, f"one History step ({len(H.steps.entries) - steps0})")
segments.seg["reviewing"] = False
SG._set_mode(bpy.context, "PARTS")

# Try Every Method: one job at a time from a timer, duplicates folded, one History step, the part back as it was
guided.stop_work()
segments.select([0])
record = segments.seg["parts"][0]
names0, was = [e.name for e in H.steps.entries], record["cur"]
ways = len(SG.try_queue(record))
ok(ways >= len(segments.METHOD_ITEMS), f"{ways} ways queued, every method among them")
ok(bpy.ops.curiomesh.seg_try_all() == {"FINISHED"}, "Try Every Method starts")
ok(run_jobs(), "every way finished")
added = [e.name for e in H.steps.entries if e.name not in names0]
ok(added.count("Try Every Method") == 1 and set(added) <= {"Try Every Method", "Before"}, f"one Ctrl+Z step for the whole run: {added}")
ok(record["cur"] == was, "the part is back as it was")
shown = SG.tried_order(record)
ok(len(record["tries"]) > 1 and len(shown) <= len(record["tries"]), f"{len(record['tries'])} tries, {len(shown)} listed, {segments.seg['tryall_note']!r}")
ok(len(SG.try_queue(record)) < ways, "a second press only runs ways not tried yet")
log = []
SG.draw_card(Rec(log), bpy.context)
ok(bool(log), "the card draws with the tries")

bpy.utils.unregister_class(SG.CURIOMESH_PT_seg_card)
pkg.unregister()
print("FAILED" if FAILS else "ALL PASS", FAILS, flush=True)
sys.exit(1 if FAILS else 0)
