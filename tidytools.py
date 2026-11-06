# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The Tools section of the main tab and one small operator per Tidy Tool (bench/tools_design/INTERACTION.md 3).

Tools: "Work on" (Selection, Part, Whole Result), the first-time button, five group popovers (Relax, Loops & Strips,
Poles, Cleanup, Hard Surface; a group with no tool in this build is not drawn), and the Last line with Again and Undo.
While a tool is open the section shows only that tool with Apply and Cancel. History, below it, lists every step.
How to add a tool: see the developer note at the top of `tidylive.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import bpy
import numpy as np

from . import history, icons, overlay, tidylive
from .triadq import relax as RX
from .triadq import tidytools as T

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


def why_not(context) -> str:
    """The reason a tool cannot open on the Work On scope now, or ""."""
    g = _guided()
    if not result_ready():
        return "Make a result first with Start. Tools work on the result before you apply it."
    if g.progress_info("AUTO") is not None:
        return "Wait for the search to finish or press Stop."
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
        # ponytail: off-square share plus size spread stands in for per-part Tidiness, which costs a pattern pass per part
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
            return "Opens Source Relax on the roughest part of your result and shows the change before anything is kept. Enter keeps it, Esc leaves everything as it was"
        return cls.bl_description

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
            tidylive.live["session"].area.header_text_set("Drag the orange ring to see more or less. Enter keeps it, Esc cancels, B shows before/after.")
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


# --------------------------------------------------------------------------- panel operators


class CURIOMESH_OT_tool_again(bpy.types.Operator):
    bl_idname = "curiomesh.tool_again"
    bl_label = "Again"
    bl_description = "Opens the last tool again with the settings you kept last time. It starts from the result as it is now"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        ts = history.settings(context.scene)
        return ts is not None and bool(ts.last_tool) and _op(ts.last_tool) is not None and not why_not(context)

    def invoke(self, context, event):
        ts = history.settings(context.scene)
        try:
            kept = json.loads(ts.last_settings or "{}")
        except ValueError:
            kept = {}
        op = _op(ts.last_tool)
        names = {p.identifier for p in op.get_rna_type().properties}
        return op("INVOKE_DEFAULT", **{k: v for k, v in kept.items() if k in names})

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
        return cls.bl_description if properties.keep else "Closes the tool and changes nothing. Same as Esc or right-click"

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
        return "Shows the first-time button and what tools do again" if properties.show else cls.bl_description

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
            grid.popover(panel=f"CURIOMESH_PT_tools_{key.lower()}", text=label, **icons.kw(icon, stock))
    if ts is not None and ts.last_tool:
        row = layout.row(align=True)
        sub = row.row(align=True)
        sub.active = False
        sub.label(text=f"Last: {ts.last_name}" + (f"  {ts.last_numbers}" if ts.last_numbers else ""))
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
            col.operator(tool.idname, text=tool.label, **icons.kw(tool.icon, tool.stock))


def _popover(key, label, tip):
    return type(f"CURIOMESH_PT_tools_{key.lower()}", (_GroupPopover, bpy.types.Panel), {"bl_idname": f"CURIOMESH_PT_tools_{key.lower()}", "bl_label": label, "bl_description": tip, "GROUP": key})


POPOVERS = tuple(_popover(key, label, tip) for key, label, _icon, _stock, tip in GROUPS)
PANELS = (CURIOMESH_PT_guided_tools, CURIOMESH_PT_guided_history)  # registered by panel.py, in the main tab's order
CLASSES = (CURIOMESH_OT_tool_source_relax, CURIOMESH_OT_tool_keep_lines_relax, CURIOMESH_OT_tool_square_up, CURIOMESH_OT_tool_relax_toward_detail, CURIOMESH_OT_tool_smooth_corners, CURIOMESH_OT_tool_again, CURIOMESH_OT_tool_finish, CURIOMESH_OT_tool_intro, *POPOVERS)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    tidylive.register()


def unregister() -> None:
    tidylive.unregister()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
