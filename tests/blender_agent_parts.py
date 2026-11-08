# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Agent part commands in background Blender: Auto on a mesh of three loose pieces, parts.worst 2, parts.redo for Low Poly,
parts.use_try back to try 0 (the face count returns to the start), issues.list, and a few cheap calls of the other
commands. Each change must make exactly one History step. Makes its own mesh, so it needs no models.

    blender -b --factory-startup --python tests/blender_agent_parts.py -- [--addon <folder of the add-on>]

On the owner's PC start Blender only through bench/rough/lowblender.sh. Exits 1 on any failure.
"""

import importlib
import os
import sys
import tempfile

import bpy

ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ADDON = os.path.abspath(ARGV[ARGV.index("--addon") + 1]) if "--addon" in ARGV else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ADDON))
pkg = importlib.import_module(os.path.basename(ADDON))
pkg.register()
guided, H, segments = pkg.guided, pkg.history, pkg.segments
A = importlib.import_module(pkg.__name__ + ".agentops_parts")
FAILS = []


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def cmd(_command, /, **args):
    """Run a command the way agentcmd does: the reply is the result, or ('ERR', code, message)."""
    try:
        return A.COMMANDS[_command]["fn"](bpy.context, **args)
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "agent_code", None)
        if code is None:
            raise
        return ("ERR", code, str(exc))


def failed(reply, code):
    return isinstance(reply, tuple) and reply[0] == "ERR" and reply[1] == code


def steps() -> int:
    """History entries made so far, not counting the 'Before' entry History adds for the state it starts from."""
    return len([e for e in H.steps.entries if e.name != "Before"])


def drain():
    guided.finish_work(bpy.context)
    guided._collect_work(bpy.context)
    while guided.progress_info() is not None:
        guided.finish_work(bpy.context)


def faces() -> int:
    return len(guided.state["res"]["faces"])


# --- a source of three loose pieces
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_monkey_add()
bpy.ops.object.modifier_add(type="SUBSURF")
bpy.context.object.modifiers[0].levels = 2
bpy.ops.object.modifier_apply(modifier="Subdivision")
bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=(4, 0, 0))
bpy.ops.mesh.primitive_cube_add(location=(-4, 0, 0))
bpy.ops.object.modifier_add(type="SUBSURF")
bpy.context.object.modifiers[0].levels = 4
bpy.ops.object.modifier_apply(modifier="Subdivision")
for o in bpy.data.objects:
    o.select_set(True)
bpy.context.view_layer.objects.active = bpy.data.objects[0]
bpy.ops.object.join()
src = bpy.context.object
src.name = "Mix"
gs = bpy.context.scene.curiomesh_guided
gs.scope, gs.auto_update = "AUTO", False

# --- before there is a result
ok(failed(cmd("parts.list"), "NO_RESULT"), "parts.list without a result is NO_RESULT")
ok(failed(cmd("issues.list"), "NO_RESULT"), "issues.list without a result is NO_RESULT")

r = bpy.ops.curiomesh.guided_auto_start()
ok(r == {"FINISHED"}, f"Start: {r}")
drain()
ok(guided.state["res"] is not None, f"a result: {guided.state['message']}")
ok(segments.seg.get("src") is not None, "the parts are found")
ok(segments.ready() and not segments.mirrored(), "the parts table is ready and redos are allowed")

# --- the list
listing = cmd("parts.list")
n = len(listing["parts"])
ok(n >= 2, f"parts.list: {n} parts, worst first {listing['worst_first']}")
ok(sorted(listing["worst_first"]) == list(range(n)), "worst_first holds every part once")
start_faces = faces()
print("INFO start faces", start_faces, flush=True)

# --- worst 2 and a Low Poly redo
picked = cmd("parts.worst", count=2)["selected"]
ok(len(picked) == min(2, n) and set(picked) == set(segments.seg["sel"]), f"parts.worst 2 marks {picked}")
ok(failed(cmd("parts.select", parts=[999]), "NOT_FOUND"), "a part that is not there is NOT_FOUND")
ok(failed(cmd("parts.redo", parts=picked, use="GAME"), "BAD_ARGS"), "an unknown Use is BAD_ARGS")

before = steps()
reply = cmd("parts.redo", parts=picked, use="LOW_POLY")
ok(isinstance(reply, dict) and reply["running"] and reply["started_in_background"], f"parts.redo starts a background redo: {reply}")
ok(failed(cmd("parts.rename", part=picked[0], name="x"), "BUSY"), "an edit while the redo runs is BUSY")
drain()
ok(steps() == before + 1, f"the redo is one History step ({before} -> {steps()}: {[e.name for e in H.steps.entries][-3:]})")
low_faces = faces()
ok(low_faces != start_faces, f"Low Poly changed the face count: {start_faces} -> {low_faces}")
tries = cmd("parts.tries", part=picked[0])["tries"]
ok(len(tries) >= 2 and tries[-1]["current"] and tries[-1]["method"] != tries[0]["method"], f"parts.tries lists the new try as the one shown: {[(t['method'], t['faces']) for t in tries]}")

# --- back to try 0, part by part
for p in picked:
    before = steps()
    reply = cmd("parts.use_try", part=p, index=0)
    ok(isinstance(reply, dict) and reply["changed"], f"parts.use_try {p} -> 0: {reply}")
    ok(steps() == before + 1, f"use_try {p} is one History step")
    drain()
ok(faces() == start_faces, f"face count is back at the start: {faces()} vs {start_faces}")
ok(cmd("parts.use_try", part=picked[0], index=0)["changed"] is False, "use_try on the try already shown changes nothing")
ok(failed(cmd("parts.use_try", part=picked[0], index=99), "NOT_FOUND"), "a try that is not there is NOT_FOUND")

# --- issues
found = cmd("issues.list")
ok(isinstance(found, dict) and found["count"] == len(found["spots"]), f"issues.list: {found['count']} spots, totals {found['totals']}")
path = os.path.join(tempfile.mkdtemp(), "issues.json")
saved = cmd("issues.save", path=path)
ok(isinstance(saved, dict) and os.path.exists(path), f"issues.save wrote {path}")
diff = cmd("issues.compare", path=path)
ok(isinstance(diff, dict) and diff["fixed"] == 0 and diff["new"] == 0 and diff["still"] == found["count"], f"issues.compare with itself: {diff['fixed']} fixed, {diff['still']} still, {diff['new']} new")
ok(failed(cmd("issues.compare", path=path + ".missing"), "FILE"), "a missing report is FILE")
ok(failed(cmd("issues.redo", index=10_000), "NOT_FOUND"), "a spot that is not there is NOT_FOUND")
if found["count"]:
    before = steps()
    reply = cmd("issues.keep_original", index=0)
    ok(isinstance(reply, dict) or failed(reply, "REFUSED"), f"issues.keep_original: {reply}")
    if isinstance(reply, dict):
        ok(steps() == before + 1, "issues.keep_original is one History step")
        drain()
        back = cmd("issues.put_back")
        ok(isinstance(back, dict), f"issues.put_back: {back}")
        drain()
    reply = cmd("issues.redo", index=0)
    ok(isinstance(reply, dict), f"issues.redo: {reply}")
    drain()
else:
    print("INFO no bad spots on this mesh: issues.redo / keep_original not exercised", flush=True)

# --- part edits (they renumber, so last)
before = steps()
reply = cmd("parts.rename", part=0, name="Head")
ok(isinstance(reply, dict) and segments.seg["parts"][0]["name"] == "Head" and steps() == before + 1, f"parts.rename: {reply}")
before = steps()
reply = cmd("parts.lock", parts=[0], on=True)
ok(isinstance(reply, dict) and segments.seg["parts"][0]["locked"] and steps() == before + 1, f"parts.lock: {reply}")
count = len(segments.seg["parts"])
before = steps()
reply = cmd("parts.merge", parts=[0, 1])
ok(isinstance(reply, dict) and len(segments.seg["parts"]) == count - 1 and steps() == before + 1, f"parts.merge: {reply}")
ok(failed(cmd("parts.merge", parts=[0]), "BAD_ARGS"), "merging one part is BAD_ARGS")
count, before = len(segments.seg["parts"]), steps()
reply = cmd("parts.halve", part=0)
ok((isinstance(reply, dict) and len(segments.seg["parts"]) == count + 1 and steps() == before + 1) or failed(reply, "REFUSED"), f"parts.halve: {reply}")
count, before = len(segments.seg["parts"]), steps()
reply = cmd("parts.from_faces", faces=list(range(min(40, faces()))))
ok((isinstance(reply, dict) and steps() == before + 1) or failed(reply, "REFUSED"), f"parts.from_faces: {reply}")
ok(segments.ready() and len(segments.seg["piece"]) == faces(), "the parts table still fits the result")

# --- Fix Selection on a plain mesh, and the preview is refused
bpy.ops.mesh.primitive_cube_add(location=(0, 8, 0))
cube = bpy.context.object
cube.name = "PlainCube"
found_bad = cmd("fix.find", object="PlainCube")
ok(isinstance(found_bad, dict) and found_bad["faces"] == 6, f"fix.find on a cube: {found_bad}")
ok(failed(cmd("fix.find", object="PlainCube", faces=[99]), "NOT_FOUND"), "a face that is not there is NOT_FOUND")
before = steps()
fixed = cmd("fix.run", how="ALL", faces=[0, 1], object="PlainCube")
ok(isinstance(fixed, dict) and steps() == before + 1 and cube.mode == "OBJECT", f"fix.run is one History step and leaves Edit Mode: {fixed}")
ok(failed(cmd("fix.run", how="NOPE"), "BAD_ARGS"), "an unknown fix is BAD_ARGS")
preview = bpy.data.objects.get(guided.PREVIEW_NAME)
if preview is not None:
    ok(failed(cmd("fix.find", object=preview.name), "REFUSED"), "the preview object is refused")

print("FAILS:", FAILS, flush=True)
sys.exit(1 if FAILS else 0)
