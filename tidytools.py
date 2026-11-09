# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The Tools section of the main tab and one small operator per Tidy Tool.

Tools: "Work on" (Selection, Part, Whole Result), the first-time button, five group popovers (Relax, Loops & Strips,
Poles, Cleanup, Hard Surface; a group with no tool in this build is not drawn), and the Last line with Again and Undo.
While a tool is open the section shows only that tool with Apply and Cancel. History, below it, lists every step.
How to add a tool: see the developer note at the top of `tidylive.py`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import bpy
import numpy as np

from . import autofix_live, history, icons, overlay, tidylive
from .triadq import autofix
from .triadq import relax as RX
from .triadq import loopspoles as LP
from .triadq import hardtidy as HT
from .triadq import tidytools as T
from .triadq import trial as TT
from .translations import tip, tr

WIDTH = 36


@dataclass(frozen=True)
class Tool:
    group: str
    idname: str
    label: str
    icon: str  # our icon name (icons/), with `stock` when it did not load
    stock: str
    developer: bool = False  # shown only with Blender's Developer Extras on (examples, not for everyday use)


# (key, label, our icon, stock icon, tooltip). One colour per group: tidylive.GROUP_COLOURS.
GROUPS = (
    ("RELAX", "Relax", "tt_relax", "MOD_SMOOTH", "Tools that even out the quads by sliding corners on your original mesh. Each one keeps something different; hover a tool to see what"),
    ("LOOPS", "Loops & Strips", "tt_loops", "FORCE_CURVE", "Pick a strip of quads and move, straighten, widen or rebuild it. Also rings round eyes and holes"),
    ("POLES", "Poles", "tt_poles", "PARTICLE_POINT", "Tools for the points where three or five edges meet. Remove pairs, move them, or open up the quads round them"),
    ("CLEANUP", "Cleanup", "tt_cleanup", "BRUSH_DATA", "Find and fix leftover triangles, thin quads and a messy mirror seam. Each shows what it found first"),
    ("HARD", "Hard Surface", "tt_hard", "MESH_CUBE", "Tools for hard edges, flat panels, round parts and shading on hard-surface results. They compare with your original mesh, not with the low-poly one"),
)
TOOLS: list[Tool] = [
    Tool("RELAX", "curiomesh.tool_source_relax", "Source Relax", "tt_source_relax", "MOD_SMOOTH"),
    Tool("RELAX", "curiomesh.tool_keep_lines_relax", "Keep Lines Relax", "tt_keep_lines", "MOD_EDGESPLIT"),
    Tool("RELAX", "curiomesh.tool_square_up", "Square Up", "tt_square_up", "MESH_PLANE"),
    Tool("RELAX", "curiomesh.tool_relax_toward_detail", "Relax Toward Detail", "tt_toward_detail", "MOD_REMESH"),
    Tool("RELAX", "curiomesh.tool_smooth_corners", "Smooth Corners", "tt_relax", "MOD_SMOOTH", developer=True),
    Tool("LOOPS", "curiomesh.tool_even_band", "Even the Band", "tt_even_band", "ALIGN_JUSTIFY"),
    Tool("POLES", "curiomesh.tool_cancel_pole_pair", "Cancel Pole Pair", "tt_cancel_pair", "UNLINKED"),
    Tool("POLES", "curiomesh.tool_open_poles", "Open Up Poles", "tt_open_poles", "PROP_ON"),
    Tool("CLEANUP", "curiomesh.tool_preview_tidy", "Preview Tidy", "tt_preview_tidy", "VIEWZOOM"),
    Tool("CLEANUP", "curiomesh.tool_seal_middle", "Seal the Middle", "mirror", "MOD_MIRROR"),
    Tool("HARD", "curiomesh.tool_true_up_edges", "True Up Edges", "tt_true_up_edges", "MOD_EDGESPLIT"),
    Tool("HARD", "curiomesh.tool_press_panels_flat", "Press Panels Flat", "tt_press_panels_flat", "MESH_PLANE"),
    Tool("HARD", "curiomesh.tool_hard_edges", "Hard Edges", "tt_true_edges", "MOD_EDGESPLIT"),
]
FIRST_RUN = "curiomesh.tool_source_relax"  # the tool "Even out the roughest patch" opens


def _guided():
    from . import guided

    return guided


def result_ready() -> bool:
    g = _guided()
    return g.state["auto_base"] is not None and g.state["res"] is not None


def _main_tab(context) -> bool:
    prefs = overlay.get_prefs()
    return bool(getattr(prefs, "show_guided", True)) and context.scene.curiomesh_guided.scope == "AUTO"


def shown_tools(context, group: str | None = None) -> list[Tool]:
    dev = bool(getattr(context.preferences.view, "show_developer_ui", False))
    return [t for t in TOOLS if (group is None or t.group == group) and (dev or not t.developer) and _op(t.idname) is not None]


def _op(idname: str):
    """The registered operator for an idname, or None."""
    module, _, name = idname.partition(".")
    op = getattr(getattr(bpy.ops, module, None), name, None)
    try:
        return op if op is not None and op.get_rna_type() is not None else None
    except (KeyError, AttributeError):
        return None


def _again_op(idname: str):
    """The operator Again may reopen: only this add-on's own, since last_tool is saved in the .blend and a file
    from someone else could name any operator."""
    return _op(idname) if idname and idname.startswith("curiomesh.") and idname != "curiomesh.tool_again" else None


def why_not(context) -> str:
    """The reason a tool cannot open on the Work On scope now, or ""."""
    g = _guided()
    if not result_ready():
        return "Make a result first with Start. Tools work on the result before you apply it."
    if g.progress_info("AUTO") is not None:
        return "Wait for the search to finish or press Stop."
    if autofix_live.busy():
        return "Auto Fix is working. Wait for it, or press Stop to keep what it fixed so far."
    ts = history.settings(context.scene)
    scope = ts.tool_scope if ts is not None else "WHOLE"
    if scope == "SELECTION":
        obj = tidylive._preview()
        faces = 0
        if obj is not None:
            try:
                faces = obj.data.count_selected_items()[2] if obj.mode == "EDIT" else obj.data.total_face_sel
            except (AttributeError, RuntimeError):
                faces = obj.data.total_face_sel
        if not faces:
            return "Nothing selected: select faces of the result in Edit Mode, or set Work On to Part or Whole Result."
    if scope == "PART":
        from . import segments

        if not segments.seg.get("sel"):
            return "Pick a part first: click one in the Fix list or in the viewport."
    return ""


class _Tool(tidylive.LiveTool):
    @classmethod
    def poll(cls, context):
        why = why_not(context)
        if why:
            cls.poll_message_set(why)
        return not why


def trial_of(idname: str):
    """The Trial class of a tool's operator, or None."""
    cls = getattr(bpy.types, idname.partition(".")[0].upper() + "_OT_" + idname.partition(".")[2], None)
    return getattr(cls, "trial", None)


def finds_spots(idname: str) -> bool:
    """True when this tool fixes spots by itself (it has `spots`, and is not manual only)."""
    trial = trial_of(idname)
    return trial is not None and getattr(trial, "auto", False) and trial.spots is not TT.Trial.spots


def _every_class(idname: str):
    """The tool's Trial wrapped in `EverySpot`, with the tool's idname among its settings so Again can open it."""
    inner = autofix.every(trial_of(idname))

    class Every(inner):
        def __init__(self, trial=None):
            super().__init__(trial)
            self.defaults = dict(self.defaults, tool=idname)

    Every.__name__ = inner.__name__
    return Every


class CURIOMESH_OT_tool_every(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_every"
    bl_label = "Fix Every Spot Like This"
    bl_description = "Finds every spot this tool is made for in Work On and shows every fix first. Enter keeps the ones that got better"
    bl_options = {"REGISTER", "UNDO"}

    tool: bpy.props.StringProperty(name="Tool", description="Which tool to run on every spot it finds", default="", options={"SKIP_SAVE"})
    keep_going: bpy.props.BoolProperty(name="Keep Going", description="Runs again on the kept result, and keeps each run that makes something better, until a run makes nothing better. Esc stops", default=False)

    @classmethod
    def description(cls, context, properties):
        trial = trial_of(properties.tool) if properties.tool else None
        many = trial.kinds[0][2] if trial is not None and trial.kinds else "spots"
        return tip(f"Fix every spot like this: finds every {many} in Work On and shows every fix first. Enter keeps the ones that got better")

    @classmethod
    def poll(cls, context):
        why = why_not(context)
        if why:
            cls.poll_message_set(why)
        return not why

    def _bind(self) -> bool:
        if not finds_spots(self.tool):
            self.report({"WARNING"}, tr("This tool has no spots to find. Open it by itself instead."))
            return False
        self.trial = _every_class(self.tool)
        return True

    def invoke(self, context, event):
        return super().invoke(context, event) if self._bind() else {"CANCELLED"}

    def execute(self, context):
        return super().execute(context) if self._bind() else {"CANCELLED"}


class PartSwap(tidylive.LiveTool):
    """Look First for the buttons that give parts other faces (Bigger and Smaller Quads, Redo This Spot, Keep Original
    Here, Put Reduced Back). The new faces already exist as a part's try, so the preview shows exactly what Enter keeps;
    Enter makes it one Ctrl+Z step and one History entry, Esc changes nothing. A size that is not worked out yet is
    redone in the background as before, and that redo is its own Ctrl+Z step when it finishes.

    A subclass writes `plan(context)` -> {"name", "tries": {part: try index}, "todo": [parts to redo], "why": ""} and
    `keep(context, plan)` -> bool, and may set `again` -> (operator idname, settings) for the Last line's Again."""

    def plan(self, context) -> dict:
        raise NotImplementedError

    def keep(self, context, plan: dict) -> bool:
        raise NotImplementedError

    def again(self, plan: dict):
        return None

    def _scope(self, context) -> str:
        return "PART"

    def invoke(self, context, event):
        plan = self.plan(context)
        if plan["why"] and not plan["tries"] and not plan["todo"]:
            self.report({"WARNING"}, tr(plan["why"]))
            return {"CANCELLED"}
        self._building = None
        if tidylive._view(context)[1] is None:
            return self.execute(context)  # no 3D view to draw the preview in
        if plan["todo"]:  # sizes not worked out yet: build them as tries first (nothing changes), then show them
            return self._build(context, plan)
        return self._show(context, event, plan, modal=False)

    def _build(self, context, plan):
        from . import ladder

        runner = ladder.build_sizes(context, plan["todo"], bool(plan.get("bigger")), plan["name"])
        if runner is None:
            return self.execute(context)
        self._building, self._plan = runner, plan
        self._wait = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, tr(f"{plan['name']}: building the new faces. The preview opens when they are ready; Esc cancels"))
        return {"RUNNING_MODAL"}

    def _stop_waiting(self, context) -> None:
        if getattr(self, "_wait", None) is not None:
            context.window_manager.event_timer_remove(self._wait)
            self._wait = None

    def modal(self, context, event):
        runner = getattr(self, "_building", None)
        if runner is None:
            return super().modal(context, event)
        if event.type == "ESC" and event.value == "PRESS":
            return self._abandon(context)
        if event.type != "TIMER" or runner.busy:
            return {"PASS_THROUGH"}
        from . import ladder

        self._building = None
        self._stop_waiting(context)
        built = ladder.finish_build(runner)
        plan = dict(self._plan)
        if not built:
            self.report({"WARNING"}, tr((runner.warnings[0] if runner.warnings else "") or f"{plan['name']}: nothing changed"))
            return {"CANCELLED"}
        plan.update(todo=[], tries={**plan["tries"], **{p: i for p, (i, _r) in built.items()}}, sizes={**plan.get("sizes", {}), **built})
        if runner.warnings:
            self.report({"WARNING"}, tr(runner.warnings[0]))
        shown = self._show(context, event, plan, modal=True)
        return {"RUNNING_MODAL"} if shown is None else shown

    def _abandon(self, context):
        from . import ladder

        runner, self._building = self._building, None
        self._stop_waiting(context)
        _guided().stop_work()
        ladder.finish_build(runner)  # the parts get their old numbers back; the result was never touched
        self.report({"INFO"}, tr(f"{self._plan['name']}: cancelled, nothing changed"))
        return {"CANCELLED"}

    def cancel(self, context):
        if getattr(self, "_building", None) is not None:
            self._abandon(context)
        else:
            super().cancel(context)

    def _show(self, context, event, plan, modal: bool):
        """Open the preview of `plan` (its tries all exist). modal: already running as a modal, so only open it. Returns
        None when open and modal, else the result to give back."""
        from . import segments

        made = segments.preview_tries(plan["tries"])
        if made is None:
            if modal:
                self.report({"WARNING"}, tr(f"{plan['name']}: nothing to show"))
                return {"CANCELLED"}
            return self.execute(context)
        self._plan = plan
        segments.select(sorted(plan["tries"]))
        old = np.asarray(segments.seg["piece"]).copy()
        self.trial = type("SwapNow", (T.SwapParts,), {"name": plan["name"], "swap": (made[0], made[1], made[3], old, tuple(plan["tries"]))})
        if not modal:
            return super().invoke(context, event)
        return self._open(context, event)

    def _keep(self, context):
        s = self._s
        plan, numbers = self._plan, tidylive.numbers_text(s.cand, s.trial, s.cap)
        tidylive.close("")  # the preview goes first, so it does not close itself when the result changes under it
        kept = self.keep(context, plan)
        self._finish(context)
        if not kept:
            self.report({"WARNING"}, tr(f"{plan['name']}: nothing to keep"))
            return {"CANCELLED"}
        self._remember(context, plan, numbers)
        return {"FINISHED"}

    def _remember(self, context, plan: dict, numbers: str) -> None:
        again = self.again(plan)
        if again is not None:
            tidylive._remember(context, again[0], plan["name"], numbers, again[1])

    def execute(self, context):
        plan = self.plan(context)
        if not self.keep(context, plan):
            self.report({"WARNING"}, tr(plan["why"] or f"{plan['name']}: nothing changed"))
            return {"CANCELLED"}
        if plan["why"]:
            self.report({"WARNING"}, tr(plan["why"]))
        self._remember(context, plan, "")
        return {"FINISHED"}


# --------------------------------------------------------------------------- tools


class CURIOMESH_OT_tool_smooth_corners(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_smooth_corners"
    bl_label = "Smooth Corners"
    bl_description = "Slides each corner toward the middle of its neighbours and back onto your original mesh. A small example of a Look First tool"
    bl_options = {"REGISTER", "UNDO"}
    trial = T.SmoothCorners

    strength: bpy.props.FloatProperty(name="Strength", description="How far each corner slides toward the middle of its neighbours. 0 leaves the result as it is", default=0.6, min=0.0, max=1.0, subtype="FACTOR")
    steps: bpy.props.IntProperty(name="Steps", description="How many small slides in a row. More evens out a wider area; higher can take longer", default=6, min=1, soft_max=50)


# --------------------------------------------------------------------------- Relax (triadq/relax.py)

KEEP_GOING = "Repeat the tool from where it ended until a run hardly moves anything. Esc stops it any time"
STEPS = "How many small slides in a row. More evens out a wider area; higher can take longer"


def _roughest_part():
    """The part whose quads are furthest from square and even on average, or None when the result has no parts."""
    from . import segments
    from .triadq import relax as RX

    res = _guided().state["res"]
    piece = segments.seg.get("piece")
    if res is None or piece is None or len(piece) != len(res["faces"]):
        return None
    piece = np.asarray(piece)
    co = np.asarray(res["co"], dtype=np.float64)
    worst, part = -1.0, None
    for p in np.unique(piece).tolist():
        quads = np.asarray([res["faces"][i] for i in np.flatnonzero(piece == p).tolist() if len(res["faces"][i]) == 4], dtype=np.int64).reshape(-1, 4)
        if len(quads) < 8:
            continue
        # off-square share plus size spread stands in for per-part Tidiness, which costs a pattern pass per part
        rough = (100.0 - RX.square_share(co, quads)) / 100.0 + RX.size_range(co, quads)
        if rough > worst:
            worst, part = rough, p
    return part


class CURIOMESH_OT_tool_source_relax(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_source_relax"
    bl_label = "Source Relax"
    bl_description = "Evens out the spacing of the corners by sliding them on your original mesh, so nothing shrinks. Use it after hand edits or on lumpy patches"
    bl_options = {"REGISTER", "UNDO"}
    trial = RX.SourceRelax

    strength: bpy.props.FloatProperty(name="Strength", description="How far the corners even out. 0 leaves the result as it is", default=0.6, min=0.0, max=1.0, subtype="FACTOR")
    steps: bpy.props.IntProperty(name="Steps", description=STEPS, default=8, min=1, soft_max=50)
    keep_going: bpy.props.BoolProperty(name="Keep Going", description=KEEP_GOING, default=False)
    first: bpy.props.BoolProperty(name="First Time", description="Open on the roughest part of the result, framed in the view", default=False, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        if properties.first:
            return tip("Opens Source Relax on the roughest part of your result and shows the change before anything is kept. Enter keeps it, Esc leaves everything as it was")
        return tip(cls.bl_description)

    def _scope(self, context) -> str:
        if not self.first:
            return super()._scope(context)
        from . import segments

        part = _roughest_part()
        if part is None:
            return "WHOLE"
        segments.seg["sel"] = {int(part)}
        segments.frame_selected(context)
        return "PART"

    def invoke(self, context, event):
        out = super().invoke(context, event)
        if self.first and "RUNNING_MODAL" in out:
            tidylive.live["session"].area.header_text_set(tr("Drag the orange ring to see more or less. Enter keeps it, Esc cancels, B shows before/after."))
        return out

    def _keep(self, context):
        out = super()._keep(context)
        ts = history.settings(context.scene)
        if self.first and "FINISHED" in out and ts is not None:  # the first win, named once
            numbers = ts.last_numbers.split(", ")[0] if ts.last_numbers else "Done"
            tidylive.live["note"] = f"Kept. {numbers} on this part. Ctrl+Z takes it back. Every tool here works the same way: look, drag, Enter."
        return out


class CURIOMESH_OT_tool_keep_lines_relax(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_keep_lines_relax"
    bl_label = "Keep Lines Relax"
    bl_description = "Evens out the corners while hard edges, the outline, UV seams and part borders stay put; corners on them only slide along them. Use it on hard-surface parts and faces with seams"
    bl_options = {"REGISTER", "UNDO"}
    trial = RX.KeepLinesRelax

    strength: bpy.props.FloatProperty(name="Strength", description="How far the corners even out. 0 leaves the result as it is", default=0.6, min=0.0, max=1.0, subtype="FACTOR")
    steps: bpy.props.IntProperty(name="Steps", description=STEPS, default=8, min=1, soft_max=50)
    hard: bpy.props.BoolProperty(name="Hard Edges", description="Corners on edges that fold sharply only slide along them. Off: they stay where they are, as in Source Relax", default=True)
    outline: bpy.props.BoolProperty(name="Outline", description="Corners on the outline seen from several sides only slide along it. Off: they relax like the rest", default=True)
    seams: bpy.props.BoolProperty(name="UV Seams", description="Corners on the edges between UV islands only slide along them. Off: they relax like the rest", default=True)
    parts: bpy.props.BoolProperty(name="Part Borders", description="Corners on the borders between parts only slide along them. Off: they relax like the rest", default=True)
    guides: bpy.props.BoolProperty(name="Guides", description="Corners on your guide curves and drawn loops only slide along them. Off: they relax like the rest", default=True)
    keep_going: bpy.props.BoolProperty(name="Keep Going", description=KEEP_GOING, default=False)


class CURIOMESH_OT_tool_square_up(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_square_up"
    bl_label = "Square Up"
    bl_description = "Turns slanted quads toward square corners and keeps each quad's size and place in its rows. Use it where quads look sheared on smooth surfaces"
    bl_options = {"REGISTER", "UNDO"}
    trial = RX.SquareUp

    strength: bpy.props.FloatProperty(name="Strength", description="How far the corners turn toward 90 degrees. 0 leaves the result as it is", default=0.7, min=0.0, max=1.0, subtype="FACTOR")
    steps: bpy.props.IntProperty(name="Steps", description=STEPS, default=12, min=1, soft_max=60)
    keep_going: bpy.props.BoolProperty(name="Keep Going", description=KEEP_GOING, default=False)


class CURIOMESH_OT_tool_relax_toward_detail(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_relax_toward_detail"
    bl_label = "Relax Toward Detail"
    bl_description = "Moves corners so quads get smaller where your mesh bends and bigger where it is flat, with the same number of faces. Use it when flat areas have too many small quads"
    bl_options = {"REGISTER", "UNDO"}
    trial = RX.RelaxTowardDetail

    mix: bpy.props.FloatProperty(name="Flat - Detail", description="Toward Detail, quad sizes follow the bends of your mesh more strongly. Toward Flat, sizes stay nearly even", default=0.5, min=0.0, max=1.0, subtype="FACTOR")
    steps: bpy.props.IntProperty(name="Steps", description=STEPS, default=12, min=1, soft_max=60)
    keep_going: bpy.props.BoolProperty(name="Keep Going", description=KEEP_GOING, default=False)


PICK = {"SKIP_SAVE"}  # a pick is for this run only: the next time the tool opens, you click again


class CURIOMESH_OT_tool_even_band(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_even_band"
    bl_label = "Even the Band"
    bl_description = "Gives the picked strip and its neighbours, out to the next pole or border, the same width. Use it on a patch whose strips are bunched together"
    bl_options = {"REGISTER", "UNDO"}
    trial = LP.EvenTheBand

    strength: bpy.props.FloatProperty(name="Strength", description="How far the strips go toward the same width. 1 makes them equal, 0 leaves them as they are", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    strip: bpy.props.IntProperty(name="Strip", description="The strip you clicked, as a number. -1 asks you to click one in the viewport", default=-1, min=-1, options=PICK)


class CURIOMESH_OT_tool_cancel_pole_pair(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_cancel_pole_pair"
    bl_label = "Cancel Pole Pair"
    bl_description = "Click a pole, then one of its lit-up partners, and both poles go away with the same face count. Use it on stray poles in places you care about"
    bl_options = {"REGISTER", "UNDO"}
    trial = LP.CancelPolePair

    pole: bpy.props.IntProperty(name="Pole", description="The pole you clicked, as a corner number. -1 asks you to click one in the viewport", default=-1, min=-1, options=PICK)
    partner: bpy.props.IntProperty(name="Partner", description="The partner you clicked, as a corner number. -1 asks you to click one of the numbered poles", default=-1, min=-1, options=PICK)
    reach: bpy.props.IntProperty(name="Reach", description="How many rings of quads round the pole are searched for partners. Higher finds partners further away and can take longer", default=6, min=1, soft_max=20, max=100000)
    steps: bpy.props.IntProperty(name="Steps", description="Most edge spins in one walk between the two poles. Higher reaches further and can take longer", default=6, min=1, soft_max=12, max=100000)
    tries: bpy.props.IntProperty(name="Tries", description="How many spins the walk may try before it says it found no way. Higher can take longer", default=400, min=1, soft_max=5000, max=100000000)


class CURIOMESH_OT_tool_open_poles(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_open_poles"
    bl_label = "Open Up Poles"
    bl_description = "Spreads the edges round each pole to even angles, so the thin quads round it open up. Use it where poles cause dents in the shading"
    bl_options = {"REGISTER", "UNDO"}
    trial = LP.OpenUpPoles

    strength: bpy.props.FloatProperty(name="Strength", description="How far the edges round each pole turn toward even angles. 1 goes as far as it can, 0 leaves them as they are", default=0.8, min=0.0, max=1.0, subtype="FACTOR")
    rings: bpy.props.IntProperty(name="Rings", description="How many rings of quads round each pole share the change. More spreads it wider and moves more corners", default=2, min=1, soft_max=3, max=100000)


class CURIOMESH_OT_tool_preview_tidy(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_preview_tidy"
    bl_label = "Preview Tidy"
    bl_description = "Lists the problems found in this area, each with a count, Show and Fix These. Nothing changes until you fix a line"
    bl_options = {"REGISTER"}  # no UNDO flag: each Fix These pushes its own Undo step while the list stays open
    trial = T.PreviewTidy

    row: bpy.props.StringProperty(
        name="Line",
        description="Which line to fix when run from a script: corners, doublets, pairs, slivers or inward. Empty only opens the list",
        default="",
        options={"SKIP_SAVE"},
    )
    steps: bpy.props.IntProperty(name="Sliver Steps", description="How many small steps each thin quad's corner takes toward its neighbours. Higher can take longer", default=T.SLIVER_STEPS, min=1, soft_max=60)


def _mirror_plane(context) -> dict:
    """{"plane": (axis, offset)} found on the result when it was not made mirrored, else {}."""
    from mathutils.bvhtree import BVHTree

    from . import symmetry

    g = _guided()
    job = (g.state["auto_base"] or {}).get("job")
    res = g.state["res"]
    if getattr(job, "mirror", None) or res is None or not len(res["faces"]):
        return {}  # mirrored already: the Session reads the plane from the job
    co = np.asarray(res["co"], dtype=np.float64)
    tree = BVHTree.FromPolygons(co.tolist(), [list(f) for f in res["faces"]], all_triangles=False, epsilon=0.0)
    best = symmetry.find_plane(co, tree)[0]
    return {"plane": (best.axis, best.offset)} if best.ok() else {}


class CURIOMESH_OT_tool_seal_middle(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_seal_middle"
    bl_label = "Seal the Middle"
    bl_description = "Puts every vertex near the mirror plane onto it and merges the doubled ones. Use it when the middle of a mirrored model has gaps or overlaps"
    bl_options = {"REGISTER", "UNDO"}
    trial = T.SealMiddle

    def _pick(self, context) -> dict:
        return _mirror_plane(context)


class CURIOMESH_OT_tool_true_up_edges(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_true_up_edges"
    bl_label = "True Up Edges"
    bl_description = "Finds the hard edges and lays each one on the crease of your original mesh, straight ones straight and curved ones smooth. Use it on wavy hard edges"
    bl_options = {"REGISTER", "UNDO"}
    trial = HT.TrueUpEdges

    strength: bpy.props.FloatProperty(name="Strength", description="How far each hard edge goes onto the crease of your original mesh. 1 lays it right on the crease, 0 leaves it as it is", default=1.0, min=0.0, max=1.0, subtype="FACTOR")


class CURIOMESH_OT_tool_press_panels_flat(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_press_panels_flat"
    bl_label = "Press Panels Flat"
    bl_description = "Finds the flat panels of your original mesh and lays each one exactly flat, with clean shared edges and corners. Use it on hard-surface parts that look bumpy"
    bl_options = {"REGISTER", "UNDO"}
    trial = HT.PressPanelsFlat

    strength: bpy.props.FloatProperty(name="Strength", description="How far each corner goes onto its panel's plane. 1 makes the panels exactly flat, 0 leaves them as they are", default=1.0, min=0.0, max=1.0, subtype="FACTOR")


class CURIOMESH_OT_tool_hard_edges(_Tool, bpy.types.Operator):
    bl_idname = "curiomesh.tool_hard_edges"
    bl_label = "Hard Edges"
    bl_description = "Shows in pink which edges Apply will make hard, while you drag the angle. Enter keeps the angle for Apply; nothing on the result moves"
    bl_options = {"REGISTER", "UNDO"}
    trial = T.HardEdges

    degrees: bpy.props.FloatProperty(name="Angle", description="Faces meeting at more than this angle make a hard edge. Lower finds softer edges too", default=40.0, min=5.0, max=180.0, precision=0)

    @classmethod
    def poll(cls, context):
        why = "" if result_ready() else "Make a result first with Start. Tools work on the result before you apply it."
        if not why and _guided().progress_info("AUTO") is not None:
            why = "Wait for the search to finish or press Stop."
        if why:  # Work On does not matter: Apply treats every edge
            cls.poll_message_set(why)
        return not why

    def _settings(self) -> dict:
        return {"angle": T.value_of(self.degrees)}

    def _scope(self, context) -> str:
        return "WHOLE"  # Apply treats every edge of the result

    def invoke(self, context, event):
        self.degrees = context.scene.curiomesh_guided.hs_angle  # opens on the angle Apply would use now
        return super().invoke(context, event)

    def _set(self, context, degrees: float) -> None:
        gs = context.scene.curiomesh_guided
        gs.hs_angle = degrees
        if gs.hs_edges == "NONE":
            gs.hs_edges = "MARK"  # keeping an angle means wanting hard edges on Apply

    def _keep(self, context):
        s = self._s
        degrees = float(round(T.angle_of(s.settings["angle"])))
        note = s.cand.note if s.cand is not None else ""
        tidylive.close("")
        self._finish(context)
        self.degrees = degrees  # what F9 shows and redoes
        self._set(context, degrees)
        tidylive._remember(context, self.bl_idname, "Hard Edges", note, {})
        return {"FINISHED"}

    def execute(self, context):
        self._set(context, self.degrees)
        tidylive._remember(context, self.bl_idname, "Hard Edges", f"angle {self.degrees:.0f} degrees", {})
        return {"FINISHED"}


# --------------------------------------------------------------------------- Detail Markers, Look First
# The ball is placed by clicking the mesh. Inside its reach ring a sample grid of quads at the size it asks for is drawn
# on the surface, and it follows the dial (strength) and the ring's edge (reach) live. Enter adds the ball (one Ctrl+Z
# step, and the search it starts fills that step in); Esc adds nothing. The operator is `guided_add_marker`; these are
# its invoke, modal and clean-up, kept here with the other Look First drawing.

MARKER_HINT = "Detail Marker: click your mesh to place it, drag the ring's edge for reach and the small ring for strength. Enter keeps it, Esc cancels."
MARKER_MISS = "Click on your mesh or the result to place the marker."
marker_live = {"state": None, "handles": []}


def _mat(obj) -> np.ndarray:
    return np.asarray(obj.matrix_world, dtype=np.float64)


def _marker_hit(m, x: float, y: float):
    """(world point, world normal) of your mesh or the result under a region pixel, nearest first, or None."""
    from bpy_extras import view3d_utils

    guided = _guided()
    region = m["region"]
    origin = view3d_utils.region_2d_to_origin_3d(region, region.data, (x, y))
    direction = view3d_utils.region_2d_to_vector_3d(region, region.data, (x, y))
    best = None
    for obj in (bpy.data.objects.get(m["src"]), guided.preview_object()):
        if obj is None or obj.type != "MESH":
            continue
        hit = guided.surface_hit(obj, origin, direction)
        if hit is None:
            continue
        mat = _mat(obj)
        point = mat[:3, :3] @ hit[0] + mat[:3, 3]
        normal = np.linalg.inv(mat[:3, :3]).T @ hit[1]
        far = float(np.linalg.norm(point - np.asarray(origin)))
        if best is None or far < best[0]:
            best = (far, point, normal / (np.linalg.norm(normal) or 1.0))
    return None if best is None else (best[1], best[2])


def _marker_onto(m, points: np.ndarray) -> np.ndarray:
    """World points moved onto your mesh (nearest point), so the ring and the grid lie on the surface."""
    from mathutils import Vector

    guided = _guided()
    src = bpy.data.objects.get(m["src"])
    if src is None:
        return points
    tree = guided._bvh(guided._source_data(src))
    mat = _mat(src)
    inv = np.linalg.inv(mat)
    local = points @ inv[:3, :3].T + inv[:3, 3]
    out = np.empty_like(local)
    for i, p in enumerate(local.tolist()):
        found = tree.find_nearest(Vector(p))[0]
        out[i] = found if found is not None else p
    return out @ mat[:3, :3].T + mat[:3, 3]


def _marker_where(m):
    return (m["at"], m["nrm"]) if m["at"] is not None else m["hover"] if m["hover"] is not None else (None, None)


def _marker_numbers(m) -> tuple[float, float, int]:
    """(quad side now, side the ball asks for at its middle, about how many faces it adds)."""
    now = m["edge"] or m["reach"] / 4.0
    scale = T.marker_scale(m["strength"], m["density"], m["less"])
    extra = T.marker_extra_faces(m["reach"], now, m["strength"], m["density"], m["less"]) if m["edge"] else 0
    return now, now * scale, extra


def _marker_batches(m):
    import gpu
    from gpu_extras.batch import batch_for_shader

    centre, normal = _marker_where(m)
    if centre is None:
        return None
    key = (tuple(np.round(centre, 6)), round(m["reach"], 6), round(m["strength"], 3))
    if key == m["key"]:
        return m["batches"]
    side = np.cross(normal, (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0))
    side /= np.linalg.norm(side)
    up = np.cross(normal, side)
    _now, size, _extra = _marker_numbers(m)
    lift = normal * 0.02 * min(size, m["reach"])
    uv, quads = T.sample_grid(m["reach"], size)
    flat = gpu.shader.from_builtin("FLAT_COLOR")
    out = {}
    if len(quads):
        points = _marker_onto(m, centre + uv[:, :1] * side + uv[:, 1:] * up) + lift
        edges = TT.edges_of([tuple(q) for q in quads.tolist()])
        out["grid"] = tidylive._lines(flat, batch_for_shader, points[edges.ravel()].astype(np.float32), tidylive.TEAL)
    turn = np.linspace(0.0, 2.0 * np.pi, 65)
    ring = _marker_onto(m, centre + m["reach"] * (np.cos(turn)[:, None] * side + np.sin(turn)[:, None] * up)) + lift
    out["ring"] = tidylive._lines(flat, batch_for_shader, np.repeat(ring, 2, axis=0)[1:-1].astype(np.float32), tidylive.ORANGE)
    m["key"], m["batches"] = key, out
    return out


def _marker_draw_view() -> None:
    m = marker_live["state"]
    if m is None:
        return
    try:
        import gpu

        out = _marker_batches(m)
        if not out:
            return
        flat = gpu.shader.from_builtin("FLAT_COLOR")
        halo = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        for name, width in (("grid", 1.5), ("ring", 2.5)):
            if name not in out:
                continue
            gpu.state.line_width_set(width + 2.0)  # dark halo: readable on a light surface too
            halo.bind()
            halo.uniform_float("color", tidylive.HALO)
            out[name].draw(halo)
            gpu.state.line_width_set(width)
            flat.bind()
            out[name].draw(flat)
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
    except Exception:
        overlay.draw_failed("tidytools._marker_draw_view")


def _marker_screen(m, point):
    px, front = tidylive._to_screen(m["region"], m["region"].data, np.eye(4), point)
    return (float(px[0][0]), float(px[0][1])) if front[0] else None


def _marker_ring_px(m):
    """(centre pixel, reach ring radius in pixels) or None."""
    centre, normal = _marker_where(m)
    if centre is None:
        return None
    mid = _marker_screen(m, centre)
    if mid is None:
        return None
    region = m["region"]
    view = np.asarray(region.data.view_matrix, dtype=np.float64)[:3, :3]
    across = view.T @ np.array((1.0, 0.0, 0.0))  # the screen's sideways direction: the ring's size as seen
    edge = _marker_screen(m, centre + m["reach"] * across)
    return mid, (math.hypot(edge[0] - mid[0], edge[1] - mid[1]) if edge is not None else 0.0)


def _marker_draw_pixel() -> None:
    m = marker_live["state"]
    context = bpy.context
    if m is None or context.region is None or context.region.as_pointer() != m["region_ptr"]:
        return
    try:
        import blf
        import gpu

        scale = tidylive._ui_scale()
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        shader.bind()
        gpu.state.blend_set("ALPHA")
        seen = _marker_ring_px(m)
        dial = seen[0] if seen is not None else None
        if dial is not None and m["at"] is not None:
            x, y = dial
            r = tidylive.DIAL_PX * scale
            start, sweep = math.radians(225.0), -math.radians(270.0)
            tidylive._poly(shader, tidylive._circle(x, y, r, 48), (*tidylive.ORANGE[:3], 0.45), 1.5)
            tidylive._poly(shader, tidylive._circle(x, y, r, max(2, int(48 * m["strength"])), start, sweep * m["strength"]), tidylive.ORANGE, 4.0)
        w, h = tidylive.CARD_W * scale, tidylive.CARD_H * scale
        region = context.region
        x, y = (dial[0] + 24 * scale, dial[1] - 24 * scale - h) if dial is not None else (16 * scale, 16 * scale)
        right = region.width - overlay._sidebar_inset() - 8 * scale
        x, y = max(8 * scale, min(x, right - w)), max(8 * scale, min(y, region.height - h - 8 * scale))
        overlay._fill(shader, overlay._rounded_rect(x, y, w, h, 6 * scale), tidylive.CARD_BG)
        overlay._fill(shader, overlay._rounded_rect(x, y, 5 * scale, h, 2 * scale), tidylive.GROUP_COLOURS["LOOPS"])
        size, pad = max(11, int(11 * scale)), 12 * scale
        _now, _new, extra = _marker_numbers(m)
        ask = T.marker_scale(m["strength"], m["density"], m["less"])
        reach = f"reach {m['reach'] / m['edge']:.0f} quads" if m["edge"] else f"reach {m['reach']:.3g}"
        overlay._text(x + pad, y + h - 18 * scale, size, (1, 1, 1, 1), f"{'Less' if m['less'] else 'More'} Detail   Strength {m['strength']:.2f}   {reach}")
        if m["note"]:
            overlay._text(x + pad, y + h - 38 * scale, size, tidylive.HELD, m["note"][:70])
        else:
            words = f"quads here about {(ask if m['less'] else 1.0 / ask):.1f}x {'bigger' if m['less'] else 'smaller'}"
            if m["edge"]:
                words += f"   about {extra:+,} faces"
            overlay._text(x + pad, y + h - 38 * scale, size, (0.85, 0.87, 0.9, 1.0), words if m["at"] is not None else "Click your mesh to place it")
        chips, cx, cy = {}, x + pad, y + 8 * scale
        for name, text, lit in (("apply", "Apply  Enter", tidylive.TEAL), ("cancel", "Cancel  Esc", tidylive.SLATE)):
            blf.size(0, size)
            cw = blf.dimensions(0, tr(text))[0] + 14 * scale
            overlay._fill(shader, overlay._rounded_rect(cx, cy, cw, 20 * scale, 4 * scale), (*lit[:3], 0.55))
            overlay._text(cx + 7 * scale, cy + 6 * scale, size, (1, 1, 1, 1), text)
            chips[name] = (cx, cy, cx + cw, cy + 20 * scale)
            cx += cw + 8 * scale
        m["chips"] = chips
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")
    except Exception:
        overlay.draw_failed("tidytools._marker_draw_pixel")


def _marker_handlers_off() -> None:
    while marker_live["handles"]:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(marker_live["handles"].pop(), "WINDOW")
        except Exception:
            pass


def marker_invoke(op, context, event):
    guided = _guided()
    area, region = tidylive._view(context)
    src = guided._source_mesh(context)
    if region is None or src is None or event is None:
        return op.execute(context)  # no view to show it in: the ball goes to the 3D cursor, as before
    tidylive.close("")  # one preview at a time
    marker_finish(None, context)
    res = guided.state["res"]
    scale = float(np.mean(np.asarray(src.matrix_world.to_scale())))
    edge = TT.mean_edge(np.asarray(res["co"], dtype=np.float64), res["faces"]) * scale if res is not None and res["faces"] and guided.state_source() is src else None
    marker_live["state"] = {
        "src": src.name, "area": area, "region": region, "region_ptr": region.as_pointer(), "less": bool(op.less),
        "strength": float(op.strength), "reach": float(op.reach) if op.reach > 0.0 else 0.1 * max(src.dimensions),
        "density": float(context.scene.curiomesh_guided.density_strength), "edge": edge, "at": None, "nrm": None,
        "hover": None, "drag": None, "note": "", "chips": {}, "key": None, "batches": None,
    }
    marker_live["handles"] = [bpy.types.SpaceView3D.draw_handler_add(_marker_draw_view, (), "WINDOW", "POST_VIEW"), bpy.types.SpaceView3D.draw_handler_add(_marker_draw_pixel, (), "WINDOW", "POST_PIXEL")]
    area.header_text_set(tr(MARKER_HINT))
    context.window_manager.modal_handler_add(op)
    area.tag_redraw()
    return {"RUNNING_MODAL"}


def marker_finish(op, context) -> None:
    m = marker_live["state"]
    marker_live["state"] = None
    _marker_handlers_off()
    if m is not None:
        try:
            m["area"].header_text_set(None)
            m["area"].tag_redraw()
        except Exception:
            pass


def marker_modal(op, context, event):
    m = marker_live["state"]
    if m is None:
        return {"CANCELLED"}
    region = m["region"]
    x, y = event.mouse_x - region.x, event.mouse_y - region.y
    inside = 0 <= x < region.width and 0 <= y < region.height
    scale = tidylive._ui_scale()
    if m["drag"] is not None:
        kind, start, value = m["drag"]
        if event.type == "MOUSEMOVE":
            if kind == "dial":
                fine = 0.1 if event.shift else 1.0
                m["strength"] = min(max(value + (x - start[0]) / (tidylive.DRAG_PX * scale) * fine, 0.05), 1.0)
            else:
                seen = _marker_ring_px(m)
                if seen is not None and start > 1.0:
                    m["reach"] = max(value * math.hypot(x - seen[0][0], y - seen[0][1]) / start, 1e-6)
            m["note"] = ""
            m["area"].tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            m["drag"] = None
            return {"RUNNING_MODAL"}
    if event.value == "PRESS" and event.type in {"RET", "NUMPAD_ENTER"}:
        return _marker_keep(op, context)
    if event.value == "PRESS" and event.type in {"ESC", "RIGHTMOUSE"}:
        marker_finish(op, context)
        return {"CANCELLED"}
    if not inside or event.type in tidylive.NAVIGATION or event.type.startswith(("NDOF", "NUMPAD")):
        return {"PASS_THROUGH"}
    if event.type == "MOUSEMOVE":
        if m["at"] is None:
            m["hover"] = _marker_hit(m, x, y)
            m["area"].tag_redraw()
        return {"PASS_THROUGH"}
    if event.type == "LEFTMOUSE" and event.value == "PRESS":
        for name, (x0, y0, x1, y1) in m["chips"].items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                if name == "apply":
                    return _marker_keep(op, context)
                marker_finish(op, context)
                return {"CANCELLED"}
        seen = _marker_ring_px(m) if m["at"] is not None else None
        if seen is not None:
            (cx, cy), ring = seen
            off = math.hypot(x - cx, y - cy)
            dial = tidylive.DIAL_PX * scale
            if abs(off - dial) < 0.45 * dial + 6:
                m["drag"] = ("dial", (x, y), m["strength"])
                return {"RUNNING_MODAL"}
            if ring > 1.0 and abs(off - ring) < 8 * scale:
                m["drag"] = ("reach", off, m["reach"])
                return {"RUNNING_MODAL"}
        hit = _marker_hit(m, x, y)
        if hit is None:
            m["note"] = MARKER_MISS
        else:
            m["at"], m["nrm"], m["note"] = hit[0], hit[1], ""
        m["area"].tag_redraw()
        return {"RUNNING_MODAL"}
    if event.type.startswith(("TIMER", "WINDOW", "EVT")):
        return {"PASS_THROUGH"}
    return {"RUNNING_MODAL"}


def _marker_keep(op, context):
    m = marker_live["state"]
    if m["at"] is None:
        m["note"] = MARKER_MISS
        m["area"].tag_redraw()
        return {"RUNNING_MODAL"}
    op.place, op.placed, op.reach, op.strength = m["at"].tolist(), True, m["reach"], m["strength"]
    _now, _new, extra = _marker_numbers(m)
    marker_finish(op, context)
    done = op.execute(context)
    if "FINISHED" in done:
        tidylive._remember(context, "curiomesh.guided_add_marker", "Detail Marker", f"about {extra:+,} faces" if m["edge"] else "", {"less": bool(op.less), "strength": op.strength, "reach": op.reach})
    return done


# --------------------------------------------------------------------------- panel operators


class CURIOMESH_OT_tool_again(bpy.types.Operator):
    bl_idname = "curiomesh.tool_again"
    bl_label = "Again"
    bl_description = "Opens the last tool again with the settings you kept last time. It starts from the result as it is now"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        ts = history.settings(context.scene)
        return ts is not None and _again_op(ts.last_tool) is not None and not why_not(context)

    def invoke(self, context, event):
        ts = history.settings(context.scene)
        op = _again_op(ts.last_tool)
        if op is None:
            return {"CANCELLED"}
        try:
            kept = json.loads(ts.last_settings or "{}")
        except ValueError:
            kept = {}
        if not isinstance(kept, dict):
            kept = {}
        names = {p.identifier for p in op.get_rna_type().properties}
        try:
            return op("INVOKE_DEFAULT", **{k: v for k, v in kept.items() if k in names})
        except (TypeError, ValueError):  # a stored value of the wrong type: open with the tool's own defaults
            return op("INVOKE_DEFAULT")

    def execute(self, context):
        return self.invoke(context, None)


class CURIOMESH_OT_tool_finish(bpy.types.Operator):
    bl_idname = "curiomesh.tool_finish"
    bl_label = "Finish Tool"
    bl_description = "Keeps the change you see. One Ctrl+Z takes it back"
    bl_options = {"INTERNAL"}

    keep: bpy.props.BoolProperty(name="Keep", description="Keep the change (on) or close the tool and change nothing (off)", default=True, options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        return tip(cls.bl_description if properties.keep else "Closes the tool and changes nothing. Same as Esc or right-click")

    @classmethod
    def poll(cls, context):
        return tidylive.busy()

    def execute(self, context):
        s = tidylive.live["session"]
        if self.keep:
            s.apply()  # the open tool keeps it on its next tick, once the exact result is there
        else:
            tidylive.close("")
        return {"FINISHED"}


class CURIOMESH_OT_tool_intro(bpy.types.Operator):
    bl_idname = "curiomesh.tool_intro"
    bl_label = "Tools Intro"
    bl_description = "Hides this button. The ? button brings it back"
    bl_options = {"INTERNAL"}

    show: bpy.props.BoolProperty(name="Show", description="Show the first-time button again (on) or hide it (off)", default=False, options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        return tip("Shows the first-time button and what tools do again" if properties.show else cls.bl_description)

    def execute(self, context):
        prefs = overlay.get_prefs()
        if hasattr(prefs, "tools_tried"):
            prefs.tools_tried = not self.show
        return {"FINISHED"}


# --------------------------------------------------------------------------- panels


class _Sub:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"
    bl_parent_id = "CURIOMESH_PT_guided"


def _first_run(layout, context) -> bool:
    """The first-time intro and big button, until a tool was kept once or Hide was pressed. False when not shown."""
    prefs = overlay.get_prefs()
    if not FIRST_RUN or _op(FIRST_RUN) is None or getattr(prefs, "tools_tried", True):
        return False
    sub = layout.column(align=True)
    sub.active = False
    overlay.wrap_label(sub, "Tools change one area of the result. You see the change on the mesh first; nothing changes until you press Enter.", width=WIDTH, limit=3)
    row = layout.row()
    row.scale_y = 1.8
    row.operator(FIRST_RUN, text="Even out the roughest patch", **icons.kw("tt_look_first", "HIDE_OFF")).first = True
    row = layout.row()
    row.alignment = "RIGHT"
    row.operator("curiomesh.tool_intro", text="Hide", emboss=False).show = False
    return True


def draw_tools(layout, context) -> None:
    s = tidylive.live["session"]
    if s is not None:  # a tool is open: only it, with Apply and Cancel for people who never look at the card
        box = layout.box()
        box.label(text=s.trial.name, **icons.kw("tt_look_first", "HIDE_OFF"))
        sub = box.column(align=True)
        sub.active = False
        sub.label(text="Look First: Enter keeps it, Esc cancels")
        row = box.row(align=True)
        row.scale_y = 1.6
        row.operator("curiomesh.tool_finish", text="Apply", icon="CHECKMARK").keep = True
        row.operator("curiomesh.tool_finish", text="Cancel", icon="CANCEL").keep = False
        return
    ts = history.settings(context.scene)
    if ts is not None:
        layout.row(align=True).prop(ts, "tool_scope", expand=True)
    _first_run(layout, context)
    groups = [g for g in GROUPS if shown_tools(context, g[0])]
    if groups:
        grid = layout.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
        grid.scale_y = 1.4
        for key, label, icon, stock, _tip in groups:
            grid.popover(panel=f"CURIOMESH_PT_tools_{key.lower()}", text=tr(label), **icons.kw(icon, stock))
    if ts is not None and ts.last_tool:
        row = layout.row(align=True)
        sub = row.row(align=True)
        sub.active = False
        sub.label(text=tr(f"Last: {ts.last_name}" + (f"  {ts.last_numbers}" if ts.last_numbers else "")))
        row.operator("curiomesh.tool_again", text="Again", icon="FILE_REFRESH")
        row.operator("ed.undo", text="", icon="LOOP_BACK")
    if tidylive.live["note"]:
        overlay.wrap_label(layout, tidylive.live["note"], icon="INFO", width=WIDTH, limit=2)


class CURIOMESH_PT_guided_tools(_Sub, bpy.types.Panel):
    bl_label = "Tools"
    bl_idname = "CURIOMESH_PT_guided_tools"

    @classmethod
    def poll(cls, context):
        return _main_tab(context) and (tidylive.busy() or (result_ready() and bool(shown_tools(context))))

    def draw_header(self, context):
        self.layout.label(text="", **icons.kw("tt_look_first", "MOD_SMOOTH"))

    def draw_header_preset(self, context):
        if FIRST_RUN and getattr(overlay.get_prefs(), "tools_tried", False):
            self.layout.operator("curiomesh.tool_intro", text="", icon="QUESTION", emboss=False).show = True

    def draw(self, context):
        draw_tools(self.layout, context)


class CURIOMESH_PT_guided_history(_Sub, bpy.types.Panel):
    bl_label = "History"
    bl_idname = "CURIOMESH_PT_guided_history"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _main_tab(context) and (result_ready() or bool(history.steps.entries))

    def draw_header(self, context):
        self.layout.label(text="", icon="RECOVER_LAST")

    def draw(self, context):
        history.draw(self.layout, context)


class _GroupPopover:
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"  # drawn only where layout.popover() opens it
    bl_ui_units_x = 13
    GROUP = ""

    def draw(self, context):
        col = self.layout.column(align=True)
        for tool in shown_tools(context, self.GROUP):
            if not finds_spots(tool.idname):  # manual only: the row is just the tool
                col.operator(tool.idname, text=tr(tool.label), **icons.kw(tool.icon, tool.stock))
                continue
            row = col.row(align=True)
            row.operator(tool.idname, text=tr(tool.label), **icons.kw(tool.icon, tool.stock))
            row.operator("curiomesh.tool_every", text="", **icons.kw("tt_fix_every", "SHADERFX")).tool = tool.idname


def _popover(key, label, tip):
    return type(f"CURIOMESH_PT_tools_{key.lower()}", (_GroupPopover, bpy.types.Panel), {"bl_idname": f"CURIOMESH_PT_tools_{key.lower()}", "bl_label": label, "bl_description": tip, "GROUP": key})


POPOVERS = tuple(_popover(key, label, tip) for key, label, _icon, _stock, tip in GROUPS)
PANELS = (CURIOMESH_PT_guided_tools, CURIOMESH_PT_guided_history)  # registered by panel.py, in the main tab's order
CLASSES = (CURIOMESH_OT_tool_source_relax, CURIOMESH_OT_tool_keep_lines_relax, CURIOMESH_OT_tool_square_up, CURIOMESH_OT_tool_relax_toward_detail, CURIOMESH_OT_tool_smooth_corners, CURIOMESH_OT_tool_even_band, CURIOMESH_OT_tool_cancel_pole_pair, CURIOMESH_OT_tool_open_poles, CURIOMESH_OT_tool_preview_tidy, CURIOMESH_OT_tool_seal_middle, CURIOMESH_OT_tool_true_up_edges, CURIOMESH_OT_tool_press_panels_flat, CURIOMESH_OT_tool_hard_edges, CURIOMESH_OT_tool_every, CURIOMESH_OT_tool_again, CURIOMESH_OT_tool_finish, CURIOMESH_OT_tool_intro, *POPOVERS)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    tidylive.register()


def unregister() -> None:
    marker_finish(None, None)
    tidylive.unregister()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
