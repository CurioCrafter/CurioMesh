# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Write a ready prompt for the user's AI agent. `build(info)` is pure text; `gather(context)` reads the open scene."""

try:
    import bpy
except ImportError:  # unit tests run without Blender
    bpy = None

GOALS = (
    ("GAME", "Game asset", "A clean, light mesh for a game engine"),
    ("FIX", "Fix bad spots", "Find the bad spots in a result and fix them"),
    ("LODS", "LODs", "Make a set of lighter copies"),
    ("SCAN", "Clean scan", "Turn a noisy scan into clean quads"),
    ("OTHER", "Other", "Say it in your own words"),
)
GOAL_WORDS = {key: label for key, label, _tip in GOALS}

SAFETY = "Never change objects I did not name. Ask before raising Most Detail Loss."

# The README "Starter prompt" is the base; the test keeps the command names real.
CONNECT = "Connect, then run session.info and commands.list so you know every command."
KEEP = "Keep it if it is better and run it again while it keeps improving; if it is worse, history.jump back to the step you noted."
STEPS = {
    "GAME": [
        "Pick the Use with use.pick GAME, read settings.list, set what my notes ask for, and run auto.start. Wait for it with job.wait.",
        "Read result.numbers and result.report. Note the History step.",
        f"Improve it with the tools, one at a time: autofix.run, tools.run, loops.fix_all. After each, read the numbers again. {KEEP}",
        "If a few parts are still bad, find them with parts.worst or issues.list and fix them with issues.redo or the parts exchange (parts.export, then parts.import).",
        "Stop when nothing improves. Tell me the before and after numbers, what you kept, and what is still not good. Export with export.game only if I asked for it.",
    ],
    "FIX": [
        "Look at the result that is already there: result.numbers, result.report and issues.list. Note the History step.",
        "Find the worst parts with parts.worst and look at each one with parts.list.",
        f"Fix them one at a time with issues.redo (or parts.redo with smaller quads). After each, read result.numbers again. {KEEP}",
        "If a part is still bad, build better topology for it with the parts exchange (parts.export, then parts.import). You can mark several at once.",
        "Stop when nothing improves. Tell me the before and after numbers, what you kept, and what is still not good.",
    ],
    "LODS": [
        "Pick the Use with use.pick GAME and run auto.start. Wait for it with job.wait, then read result.numbers.",
        "Make the set with lod.make and wait with job.wait. Tell me the face count of each level.",
        "If a level has bad spots, fix them with issues.list and issues.redo, then check the set again.",
        "Stop when each level looks right at its distance. Export with export.game only if I asked for it.",
    ],
    "SCAN": [
        "Pick the Use with use.pick SCAN, read settings.list, set what my notes ask for, and run auto.start. Wait for it with job.wait.",
        "Read result.numbers and result.report. Note the History step.",
        f"Improve it with the tools, one at a time: tidy.keep_going, autofix.run, loops.fix_all. After each, read the numbers again. {KEEP}",
        "If a few parts are still bad, fix them with issues.redo or the parts exchange.",
        "Stop when nothing improves. Tell me the before and after numbers, what you kept, and what is still not good.",
    ],
    "OTHER": [
        "Pick the Use that fits (use.pick), read settings.list, set what my notes ask for, and run auto.start. Wait for it with job.wait.",
        "Read result.numbers and result.report. Note the History step.",
        f"Improve it with the tools, one at a time. After each, read the numbers again. {KEEP}",
        "If a few parts are still bad, mark them with parts.worst or issues.list and fix them with the parts exchange (parts.export, then parts.import).",
        "Stop when nothing improves. Tell me the before and after numbers, what you kept, and what is still not good. Export only if I asked for it.",
    ],
}


def _route(info: dict) -> str:
    folder = info.get("folder")
    if info.get("route") == "mailbox" and folder:
        return (
            f"Connection: my Blender is open and CurioMesh is listening on the mailbox folder {folder}. "
            f'Start the MCP server with curiomesh_mcp.py --folder "{folder}", or write inbox/<id>.json there and read outbox/<id>.json.'
        )
    return (
        "Connection: there is no live link, so use a headless Blender: run curiomesh_mcp.py with no folder, or "
        "blender -b --factory-startup --python cm_do.py -- --commands job.json --out replies.jsonl. "
        "(To work in my open Blender instead, I can turn on Let an Agent Drive CurioMesh in the Pro tab, This Computer.)"
    )


def build(info: dict) -> str:
    """The prompt text. `info` keys (all optional): object, file, faces, use, changed (list of "name=value"),
    numbers (result.numbers dict), worst (list of text), issues (text), goal, note, route ("mailbox" or "headless"), folder."""
    goal = info.get("goal") if info.get("goal") in STEPS else "OTHER"
    model = info.get("object") or "<the object's name in my open Blender>"
    if info.get("file"):
        model += f" (file {info['file']})"
    if info.get("faces"):
        model += f", {info['faces']:,} faces"
    wants = GOAL_WORDS[goal]
    if info.get("note"):
        wants += f": {info['note']}"
    lines = ["Use the curiomesh-agent skill to retopologise a model with CurioMesh.", "", f"Model: {model}", f"What it is for: {wants}"]
    state = []
    if info.get("use"):
        state.append(f"Use: {info['use']}")
    if info.get("changed"):
        state.append("Settings I changed: " + ", ".join(info["changed"]))
    nums = info.get("numbers") or {}
    bits = [f"{k.replace('_', ' ')} {nums[k]}" for k in ("faces", "quad_pct", "view_change_pct") if nums.get(k) is not None]
    if (nums.get("verdict") or {}).get("title"):
        bits.append("verdict: " + nums["verdict"]["title"])
    if bits:
        state.append("Result so far: " + ", ".join(bits))
    if info.get("worst"):
        state.append("Worst parts: " + "; ".join(info["worst"]))
    if info.get("issues"):
        state.append("Bad spots: " + info["issues"])
    if state:
        lines += ["What is already done:"] + [f"- {s}" for s in state]
    lines += ["", _route(info), ""]
    lines += [f"{i}. {s}" for i, s in enumerate([CONNECT] + STEPS[goal], 1)]
    lines += [SAFETY, "Run commands.list first: it is the full set of commands."]
    return "\n".join(lines)


# ----- reading the open scene (needs Blender; every read is guarded so a missing piece just leaves a line out)


def _changed(group) -> list[str]:
    out = []
    for prop in group.bl_rna.properties:
        if prop.identifier == "rna_type" or prop.type in {"POINTER", "COLLECTION"} or prop.is_readonly or prop.is_array:
            continue
        try:
            value = getattr(group, prop.identifier)
            if not (prop.type == "ENUM" and prop.is_enum_flag) and value != prop.default:
                out.append(f"{prop.identifier}={value}")
        except Exception:
            continue
    return out


def gather(context) -> dict:
    info = {}
    wm = context.window_manager
    info["goal"] = getattr(wm, "curiomesh_prompt_goal", "GAME")
    if info["goal"] == "OTHER":
        info["note"] = (getattr(wm, "curiomesh_prompt_note", "") or "").strip()
    obj = context.active_object
    if obj is not None:
        info["object"] = obj.name
        if obj.type == "MESH":
            info["faces"] = len(obj.data.polygons)
    if bpy.data.filepath:
        info["file"] = bpy.data.filepath.replace("\\", "/").rsplit("/", 1)[-1]
    gs = getattr(context.scene, "curiomesh_guided", None)
    if gs is not None:
        info["use"] = gs.auto_profile
        info["changed"] = _changed(gs)[:20]
    try:
        from . import agentops_run

        info["numbers"] = agentops_run.result_numbers(context)
    except Exception:
        pass  # no result yet
    try:
        from . import issue_panel, segments

        info["worst"] = [f"part {p}" for p in segments.bad_parts()[:3]]
        kinds = {}
        for spot in issue_panel.spots():
            for k, c in spot.get("kinds", {}).items():
                kinds[k] = kinds.get(k, 0) + c
        if kinds:
            info["issues"] = issue_panel.summary(kinds)
    except Exception:
        pass
    info["route"] = "headless"
    try:
        from . import agent_mailbox

        folder = agent_mailbox.default_folder()
        if wm.curiomesh_agent_on and folder is not None:
            info["route"], info["folder"] = "mailbox", str(folder)
    except Exception:
        pass
    return info


if bpy is not None:

    class CURIOMESH_OT_write_prompt(bpy.types.Operator):
        bl_idname = "curiomesh.write_prompt"
        bl_label = "Write a Prompt for My AI"
        bl_description = "Copy a ready prompt for your AI agent, with your model and settings filled in. Nothing is sent anywhere"
        bl_options = {"INTERNAL"}

        def execute(self, context):
            prompt = build(gather(context))
            context.window_manager.clipboard = prompt
            block = bpy.data.texts.get("CurioMesh AI prompt") or bpy.data.texts.new("CurioMesh AI prompt")
            block.clear()
            block.write(prompt)
            self.report({"INFO"}, "Prompt copied. It is also in the text block 'CurioMesh AI prompt'.")
            return {"FINISHED"}

    CLASSES = (CURIOMESH_OT_write_prompt,)
else:
    CLASSES = ()


def draw(layout, context) -> None:
    """The body of the Pro "Write a Prompt for My AI" panel."""
    from . import help as help_mod

    wm = context.window_manager
    row = layout.row(align=True)
    row.prop(wm, "curiomesh_prompt_goal", text="Goal")
    help_mod.help_button(row, "prompt")
    if wm.curiomesh_prompt_goal == "OTHER":
        layout.prop(wm, "curiomesh_prompt_note", text="")
    layout.operator("curiomesh.write_prompt", icon="COPYDOWN")


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.curiomesh_prompt_goal = bpy.props.EnumProperty(
        name="Goal", items=GOALS, default="GAME", description="What you want your AI agent to do"
    )
    bpy.types.WindowManager.curiomesh_prompt_note = bpy.props.StringProperty(
        name="Your goal", description="Say what you want in your own words", default=""
    )


def unregister():
    for name in ("curiomesh_prompt_goal", "curiomesh_prompt_note"):
        if hasattr(bpy.types.WindowManager, name):
            delattr(bpy.types.WindowManager, name)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
