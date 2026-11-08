# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Run with (on the owner's PC only through bench/rough/lowblender.sh):
    blender --background --factory-startup --python tests/blender_agent_run.py

The commands of agentops_run, called directly: a bumpy ball, pick the Game use, Start, wait, read the numbers, change
Detail, tidy through the Tidy Up slider, then History: list, jump back one step and check the faces match that step. Exits 1 on any failure.
"""

import importlib
from pathlib import Path
import sys
import traceback

import bmesh
import bpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
addon = importlib.import_module(REPO.name)
addon.register()
R = importlib.import_module(REPO.name + ".agentops_run")
guided = importlib.import_module(REPO.name + ".guided")
FAILS = []


def ok(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def call(name, **args):
    return R.COMMANDS[name]["fn"](bpy.context, **args)


def error_of(name, **args):
    try:
        call(name, **args)
    except R.AgentError as exc:
        return exc
    return None


def faces_now():
    return len(guided.state["res"]["faces"])


try:
    mesh_bm = bmesh.new()
    bmesh.ops.create_icosphere(mesh_bm, subdivisions=4, radius=1.0)
    for v in mesh_bm.verts:
        v.co *= 1.0 + 0.04 * ((v.co.x * 9.0) % 1.0)
    mesh = bpy.data.meshes.new("Ball")
    mesh_bm.to_mesh(mesh)
    mesh_bm.free()
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("Ball", mesh))

    ok(error_of("scene.select", object="Nope").agent_code == "NO_OBJECT", "unknown mesh gives NO_OBJECT")
    ok(call("scene.select", object="Ball")["faces"] == len(mesh.polygons), "scene.select")
    ok(call("scene.objects")["objects"][0]["name"] == "Ball", "scene.objects")
    ok(error_of("result.numbers").agent_code == "NO_RESULT", "no result yet gives NO_RESULT")
    ok(call("job.status")["running"] is False, "idle status")

    groups = call("settings.groups")["groups"]
    ok({"guided", "tools", "prefs", "part"} <= {g["group"] for g in groups}, "settings.groups")
    detail = next(s for s in call("settings.list", group="guided")["settings"] if s["name"] == "auto_detail")
    ok(detail["min"] == 0.1 and detail["max"] == 80.0 and detail["help"], f"auto_detail range from RNA: {detail['min']}..{detail['max']}")
    ok(error_of("settings.set", group="guided", values={"auto_detail": 500}).agent_code == "BAD_ARGS", "out of range refused")
    ok(error_of("settings.set", group="guided", values={"no_such": 1}).agent_code == "BAD_ARGS", "unknown name refused")
    ok(error_of("settings.set", group="guided", values={"auto_profile": "NOPE"}).agent_code == "BAD_ARGS", "bad choice refused")
    for group in ("tools", "autofix", "fix", "classic", "region", "part"):
        ok(len(call("settings.list", group=group)["settings"]) > 0, f"settings.list {group}")

    ok(call("use.pick", use="GAME")["use"] == "GAME", "use.pick GAME")
    r = call("auto.start")
    ok(r["running"], "auto.start returns running")
    ok(error_of("use.pick", use="SCAN").agent_code == "BUSY", "edit while running gives BUSY")
    st = call("job.wait", seconds=900)
    ok(not st["running"] and not st["timed_out"], f"job.wait finished: {st['message']}")
    num = call("result.numbers")
    ok(num["faces"] > 50 and num["quads"] > 0 and "level" in num["verdict"], f"result.numbers {num['faces']} faces, {num['quad_pct']}% quads, view {num['view_change_pct']}%, {num['verdict']['title']}")
    ok(isinstance(call("result.report")["issues"]["totals"], dict), "result.report")
    first_faces = faces_now()
    ok(first_faces == num["faces"], "numbers match the preview")
    rows = call("history.list")
    ok(len(rows["steps"]) >= 1 and rows["steps"][-1]["name"] == "New result" and rows["steps"][-1]["faces"] == first_faces, "history has New result with its faces")

    ch = call("settings.set", group="guided", values={"auto_detail": 8.0})["changed"]["auto_detail"]
    ok(ch["new"] == 8.0, f"settings.set auto_detail {ch['old']} -> {ch['new']}")
    call("settings.set", group="guided", values={"organize": 1.0})  # the Tidy Up slider: its update starts the first pass
    st = call("job.wait", seconds=900)
    ok(not st["running"], "tidy pass finished")
    tl = call("tidy.levels")
    ok(len(tl["levels"]) >= 2, f"tidy.levels {[(lv['strength'], lv.get('tidy')) for lv in tl['levels']]}")
    rows = call("history.list")
    ok(len(rows["steps"]) >= 2, f"history steps {[(r['name'], r['faces']) for r in rows['steps']]}")

    # jump back one step and check the faces are those of that step
    back = rows["steps"][rows["current"]["index"] - 1]
    out = call("history.jump", step=back["step"])
    ok(back["faces"] is not None and faces_now() == back["faces"], f"history.jump back to '{back['name']}': {faces_now()} faces == {back['faces']}")
    ok(out["faces"] == faces_now(), "jump reply faces")
    ok(error_of("history.jump", step=999).agent_code == "NOT_FOUND", "unknown step gives NOT_FOUND")

    # undo and redo without a window
    before = call("history.list")
    u = call("history.undo")
    ok(u["current"]["index"] == before["current"]["index"] - 1 and faces_now() == before["steps"][u["current"]["index"]]["faces"], f"headless undo -> '{u['current']['name']}' {faces_now()} faces")
    rd = call("history.redo")
    ok(rd["current"]["index"] == before["current"]["index"] and faces_now() == before["steps"][rd["current"]["index"]]["faces"], f"headless redo -> '{rd['current']['name']}' {faces_now()} faces")

    ok(len(call("diag.log")["lines"]) >= 1, "diag.log has lines")
    ok(call("result.discard")["discarded"], "result.discard")
except Exception:
    traceback.print_exc()
    FAILS.append("exception")

print("FAILED: " + "; ".join(FAILS) if FAILS else "ALL PASSED", flush=True)
sys.exit(1 if FAILS else 0)
