# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The CurioMesh sidebar: every panel and popover of the main tab and the Pro tab (bench/uidesign/DESIGN.md).

Main tab, top to bottom: CurioMesh (pick, What is it for?, Start, verdict), Adjust, Fix, Finish. Each sub-panel shows
only when it can do something. Everything else is one click away in a popover, or in the CurioMesh Pro tab.

Adding a control: find the section function of its area (draw_setup, draw_options, draw_shape_flow, draw_adjust,
draw_fix, draw_finish, or a Pro panel) and add it there; nothing else needs to change. The Flow Editor plugs into
draw_shape_flow through FLOW_EDITOR_OP."""

from __future__ import annotations

import bpy

from . import guided as G
from . import autofix_live, faceloops, flowedit, hard_surface, hardware, icons, issue_panel, ladder, organize, overlay, profiles, segment_goals, segments, symmetry, tidylive, tidytools
from . import verdict as V
from . import uv_panel  # main-only: uv
from . import help as helpmod, promptgen  # main-only

FLOW_EDITOR_OP = ("CURIOMESH_OT_flow_edit", "curiomesh.flow_edit")  # drawn as [Draw on the mesh] once registered
WIDTH = 36  # characters per wrapped line in the sidebar
STATUS_ICONS = {"GOOD": ("status_good", "CHECKMARK"), "CHECK": ("status_check", "ERROR"), "BAD": ("status_bad", "CANCEL")}
# (Use key, tile text, our icon, stock icon when ours did not load). AUTO is not a Use: it lets Start pick one.
TILES = (
    ("AUTO", "Auto", "use_auto", "SHADERFX"),
    ("GAME", "Game", "use_game", "MESH_CUBE"),
    ("LOW_POLY", "Low Poly", "use_lowpoly", "MOD_DECIM"),
    ("ANIMATION", "Animation", "use_animation", "ARMATURE_DATA"),
    ("SUBDIV", "Subdiv", "use_subdiv", "MOD_SUBSURF"),
    ("SCAN", "Scan", "use_scan", "OUTLINER_OB_POINTCLOUD"),
    ("SCULPT", "Sculpt", "use_sculpt", "SCULPTMODE_HLT"),
    ("BLOCKOUT", "Blockout", "use_blockout", "CUBE"),
)
USE_NAMES = {key: label for key, label, _tip in profiles.PROFILE_ITEMS}
HELP = {  # one line under a section title, hidden by Compact Panels (DESIGN.md 2.6)
    "empty": "Click a mesh in the 3D view. CurioMesh makes a clean, lighter copy; yours is not changed until you press Apply.",
    "adjust": "Move a slider to see another version right away. Nothing is lost until you Apply.",
    "fix": "Go through the spots that need a look, one at a time, and fix each one here.",
    "finish": "Write the result into your scene, then bake or export it if you need to.",
    "pro": "Every setting and the classic tools. You never need these for a good result.",
}


def _prefs():
    return overlay.get_prefs()


def _compact() -> bool:
    return bool(getattr(_prefs(), "compact_mode", False))


def _help(layout, key: str) -> None:
    if not _compact():
        row = layout.column(align=True)
        row.active = False  # grey
        overlay.wrap_label(row, HELP[key], width=WIDTH, limit=3)


def _ends(layout, left: str, right: str) -> None:
    """The words at both ends of a slider, small and grey."""
    row = layout.row(align=True)
    row.active = False
    row.label(text=left)
    sub = row.row(align=True)
    sub.alignment = "RIGHT"
    sub.label(text=right)


def _pro_toggle(layout, text: str = "Pro") -> None:
    prefs = _prefs()
    if isinstance(prefs, bpy.types.AddonPreferences):
        layout.prop(prefs, "pro_mode", text=text, toggle=True, **icons.kw("pro", "PREFERENCES"))


def result_ready() -> bool:
    return G.state["auto_base"] is not None and G.state["res"] is not None


def _auto_tab(context) -> bool:
    return bool(_prefs().show_guided) and context.scene.curiomesh_guided.scope == "AUTO"


def setup_words(gs) -> str:
    """One grey line for the settings a job runs with: 'Animation, Keep Most (1.5%), Mirror X'."""
    words = ["Auto" if gs.use_auto else USE_NAMES.get(gs.auto_profile, gs.auto_profile)]
    if gs.auto_profile not in profiles.BLOCKOUT_USES:
        words.append(gs.bl_rna.properties["detail_pick"].enum_items[gs.detail_pick].name)
    if gs.sym_axis != "OFF":
        words.append(f"Mirror {gs.sym_axis.title()}")
    return ", ".join(words)


# --------------------------------------------------------------------------- operators


class CURIOMESH_OT_pick_use(bpy.types.Operator):
    bl_idname = "curiomesh.pick_use"
    bl_label = "What Is It For"
    bl_description = "Pick what you will do with the result"
    bl_options = {"INTERNAL", "UNDO"}

    use: bpy.props.StringProperty(name="Use", description="The Use to pick, or AUTO to let Start pick one", default="GAME", options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        if properties.use == "AUTO":
            return "Start looks at your mesh and picks the Use for you, then says what it picked and why"
        return next((tip for key, _label, tip in profiles.PROFILE_ITEMS if key == properties.use), cls.bl_description)

    def execute(self, context):
        gs = context.scene.curiomesh_guided
        if self.use == "AUTO":
            gs.use_auto = True
        elif self.use in USE_NAMES:
            gs.use_auto = False
            gs.auto_why = ""
            gs.auto_profile = self.use
        else:
            return {"CANCELLED"}
        overlay.tag_redraw()
        return {"FINISHED"}


# --------------------------------------------------------------------------- main panel sections


# --------------------------------------------------------------------------- dashboard button and step strip

STEPS = ("Use", "Auto", "Review", "Fix", "Finish")
STEP_HINTS = (
    "Click a mesh in the 3D view",
    "Step 1 of 5: pick what it is for, then press Start",
    "Step 2 of 5: press Start",
    "Step 3 of 5: check the result, then fix spots or Apply",
    "Step 4 of 5: fix the bad spots, then Apply",
    "Step 5 of 5: applied. Export it or pick another mesh",
)


def current_step(context) -> int:
    """0 = no mesh, 1 Use, 2 Auto (picked or working), 3 Review, 4 Fix (bad spots), 5 Finish (applied)."""
    gs = context.scene.curiomesh_guided
    if G.state["applied"] and not result_ready():
        return 5
    if G._source_mesh(context) is None:
        return 0
    if G.progress_info("AUTO"):
        return 2
    if result_ready():
        return 4 if serious_spots(G.state["res"]) else 3
    return 1 if gs.use_auto else 2


def draw_steps(layout, context) -> None:
    """One line: where you are in the five steps, then what to press next."""
    step = current_step(context)
    row = layout.row(align=True)
    for i, name in enumerate(STEPS, 1):
        cell = row.row(align=True)
        cell.active = i == step
        cell.label(text=name, icon="CHECKMARK" if i < step else ("RADIOBUT_ON" if i == step else "BLANK1"))
    if not _compact():
        sub = layout.row()
        sub.active = False
        sub.label(text=STEP_HINTS[step])


def draw_dashboard_button(layout, context) -> None:
    on = bool(context.scene.curiomesh_settings.hud_enabled)
    row = layout.row(align=True)
    row.operator("curiomesh.hud_toggle", text="Hide Dashboard" if on else "Show Dashboard", icon="OVERLAY", depress=on)
    helpmod.help_button(row, "dashboard")


def _slot(layout, text: str, rows: int, icon: str = "NONE", width: int = WIDTH) -> None:
    """A message in exactly `rows` label rows (blank rows when there is nothing to say), so what sits below never moves."""
    import textwrap

    lines = textwrap.wrap(str(text), width=width) if text else []
    if len(lines) > rows:
        lines = lines[:rows]
        lines[-1] = lines[-1][: width - 3] + "..."
    for i in range(rows):
        shown = i < len(lines)
        layout.label(text=lines[i] if shown else "", icon=icon if i == 0 and shown else "BLANK1")


def draw_review_button(layout, scale: float = 1.5, count: int | None = None) -> None:
    """Fix Bad Spots: walk the bad spots worst first, mark parts in the 3D view, redo them.
    With a count it is the main action: big, red while there are bad spots, and in the same place (greyed out) when there are none."""
    row = layout.row()
    row.scale_y = scale
    reviewing = bool(segments.seg.get("reviewing")) and segments.seg.get("review_mode") == "SPOTS"
    text = "Stop (Esc)" if reviewing else (f"Fix Bad Spots ({count})" if count else "Fix Bad Spots")
    if count is not None and not reviewing:
        row.alert = count > 0
        row.enabled = count > 0
    row.operator("curiomesh.seg_review", text=text, icon="VIEWZOOM", depress=reviewing).mode = "SPOTS"


def draw_empty(layout, context) -> None:
    box = layout.box()
    active = context.object
    if active is not None and active.type != "MESH":
        box.label(text=f"{active.name} is a {active.type.title()}, not a mesh", icon="ERROR")
        overlay.wrap_label(box, "Convert it first: Object menu, Convert, Mesh.", icon="BLANK1", width=WIDTH, limit=2)
        return
    row = box.row()
    row.label(text="Pick a mesh to start", **icons.kw("step_pick", "RESTRICT_SELECT_OFF"))
    helpmod.help_button(row, "empty")
    _help(box, "empty")
    row = box.row(align=True)
    row.active = False
    for name, text, stock in (("step_pick", "Pick", "RESTRICT_SELECT_OFF"), ("step_start", "Start", "PLAY"), ("step_adjust", "Adjust", "OPTIONS"), ("step_finish", "Finish", "CHECKMARK")):
        row.label(text=text, **icons.kw(name, stock))


def draw_mesh_line(layout, obj) -> None:
    row = layout.split(factor=0.62, align=True)  # the name gets the room: "marble_bus..." was cut at half width
    row.label(text=obj.name, icon="MESH_DATA")
    row.label(text=f"{len(obj.data.polygons):,} faces")


def draw_use_tiles(layout, gs) -> None:
    """What is it for? Eight picture tiles, the picked one pressed in, then one line about it."""
    row = layout.row()
    row.label(text="" if _compact() else "What is it for?")
    helpmod.help_button(row, "use")
    grid = layout.grid_flow(row_major=True, columns=4, even_columns=True, even_rows=True, align=True)
    for key, text, name, stock in TILES:
        picked = gs.use_auto if key == "AUTO" else (not gs.use_auto and gs.auto_profile == key)
        col = grid.column(align=True)
        found = icons.icon(name)
        if found:
            col.template_icon(icon_value=found, scale=2.0)
            col.operator("curiomesh.pick_use", text=text, depress=picked).use = key
        else:
            col.operator("curiomesh.pick_use", text=text, icon=stock, depress=picked).use = key
    if gs.use_auto:
        words = gs.auto_why or "Auto looks at your mesh when you press Start and picks the Use for it, then says why."
    else:
        words = next((tip for key, _label, tip in profiles.PROFILE_ITEMS if key == gs.auto_profile), "")
    row = layout.row(align=True)
    col = row.column(align=True)
    col.active = False
    if not _compact():
        overlay.wrap_label(col, words, width=WIDTH - 4, limit=3)
    row.popover(panel="CURIOMESH_PT_pop_uses", text="", icon="DOWNARROW_HLT")


def draw_detail(layout, gs, faces: int) -> None:
    col = layout.column(align=True)
    col.use_property_split = True
    col.use_property_decorate = False
    if gs.auto_profile in profiles.BLOCKOUT_USES:  # no Detail: each part's solid is the camera's pick, and the sides set the rest
        col.prop(gs, "blockout_sides")
        col.prop(gs, "blockout_separate", toggle=True, icon="OUTLINER_OB_GROUP_INSTANCE")
        return
    col.prop(gs, "detail_pick")
    pick = gs.detail_pick
    if pick == "CUSTOM":
        col.prop(gs, "auto_detail", text="View Change")
    elif pick in {"PERCENT", "COUNT"}:
        col.prop(gs, "auto_reduce_percent" if pick == "PERCENT" else "auto_reduce_count")
        sub = col.row()
        sub.active = False
        sub.label(text=f"Aiming for about {G.auto_target(gs, faces):,} faces")
    row = col.row(align=True)
    row.prop(gs, "sym_axis", **icons.kw("mirror", "MOD_MIRROR"))
    row.popover(panel="CURIOMESH_PT_pop_mirror", text="", icon="PREFERENCES")
    helpmod.help_button(row, "mirror")


def draw_start(layout, gs, faces: int, again: bool) -> None:
    """The one primary button before a result. With Auto picked it runs the Auto choice first."""
    if G.state["machine"] is not None and not again:
        level, text, _settings = G.job_rating(gs, faces, G.state["src"]["sig"] if G.state["src"] is not None else None)
        sub = layout.column(align=True)
        sub.active = level == "OK"
        overlay.wrap_label(sub, text, icon="TIME" if level == "OK" else "ERROR", width=WIDTH, limit=3)
    outer = layout.row(align=True)
    row = outer.row()
    row.scale_y = 1.4 if again else 2.1
    row.enabled = G.progress_info() is None
    row.operator("curiomesh.guided_auto_all" if gs.use_auto else "curiomesh.guided_auto_start", text="Start again" if again else "Start", **icons.kw("step_start", "PLAY"))
    helpmod.help_button(outer, "start")
    if not again and not _compact():
        sub = layout.row()
        sub.active = False
        sub.label(text="Nothing changes until you press Apply", icon="LOCKED")


def draw_setup(layout, context, gs, obj, again: bool = False) -> None:
    """Pick, Detail, Mirror, the two popovers and Start: everything before a search."""
    faces = len(obj.data.polygons)
    draw_use_tiles(layout, gs)
    draw_detail(layout, gs, faces)
    row = layout.row(align=True)
    row.popover(panel="CURIOMESH_PT_pop_flow", text="Shape the flow...", **icons.kw("tool_pen", "GREASEPENCIL"))
    helpmod.help_button(row, "shape_flow")
    row.popover(panel="CURIOMESH_PT_pop_options", text="Options...", icon="PREFERENCES")
    helpmod.help_button(row, "options")
    draw_start(layout, gs, faces, again)


def draw_working(layout, gs, searching) -> None:
    sub = layout.row()
    sub.active = False
    sub.label(text=setup_words(gs), icon="BLANK1")
    box = layout.box()
    row = box.row()
    row.label(text="Working: you can keep using Blender", **icons.kw("status_working", "TIME"))
    helpmod.help_button(row, "working")
    G.draw_progress(box, *searching)
    box.operator("curiomesh.guided_stop_work", icon="CANCEL")


def serious_spots(res) -> list:
    return [s for s in res.get("issue_spots") or [] if any(k in V.SERIOUS for k in s.get("kinds", {}))]


def draw_verdict(layout, gs) -> None:
    """Good / check / bad in plain words, the numbers under it in grey, then the one next action."""
    base, res = G.state["auto_base"], G.state["res"]
    report, after = base["report"], len(res["faces"])
    asked = gs.auto_detail if gs.auto_aim == "KEEP" and gs.auto_profile not in profiles.BLOCKOUT_USES and not gs.multires_on else None
    bad = serious_spots(res)
    use = USE_NAMES.get(gs.auto_profile, "")
    level, title = V.verdict(use if gs.auto_profile != "CUSTOM" else "", after, report["loss"], asked, V.serious_count((res.get("issues") or {}).get("totals")), len(bad), report.get("before"))
    box = layout.box()
    head = box.row()
    head.alert = level == "BAD"
    head.label(text=title, **icons.kw(*STATUS_ICONS[level]))
    helpmod.help_button(head, "verdict")
    col = box.column(align=True)
    col.active = False
    fewer = 100.0 * (1.0 - after / max(report["before"], 1))
    later = " (before fixes)" if gs.bool_on and len(gs.bool_shapes) else ""
    stats = [(f"{after:,} faces ({fewer:.0f}% fewer), looks {max(0.0, 100.0 - report['loss']):.0f}% the same{later}", "NONE")]
    if report.get("topology") is not None:
        stats.append((f"Tidiness {report['topology']:.0f}: {V.tidiness_words(report['topology'])}", "NONE"))
    if report.get("shading") is not None and not _compact():
        stats.append((f"Shading {max(0.0, 100.0 - report['shading']):.0f}% the same", "NONE"))
    if report.get("asked") and asked is None:
        stats.append((f"You asked for about {report['asked']:,}", "NONE"))
        short = G.short_of_target(report["asked"], after)
        if short:
            import textwrap

            stats += [(line, "ERROR" if i == 0 else "BLANK1") for i, line in enumerate(textwrap.wrap(short, width=WIDTH)[:3])]
    follows = report.get("guides")
    if follows:
        stats.append((f"Guides: {follows['strokes']} strokes, nearest edge {100.0 * follows['mean']:.2f}% of the size away", "NONE"))
    bend = report.get("bend")
    if bend:
        stats.append((f"Joint lines: {bend['found']} of {bend['loops']} loops found, gap {100.0 * bend['gap']:.0f}% of asked", "NONE"))
    for text, icon in stats + [("", "NONE")] * (3 - len(stats)):  # at least three rows, so the spots box below stays put
        col.label(text=text, icon=icon)
    box2 = layout.box()  # always drawn: the same rows with or without bad spots
    line = box2.row()
    if bad:
        line.alert = level == "BAD"
        line.label(text=f"{len(bad)} spot{'s' if len(bad) != 1 else ''} with bad faces", **icons.kw("view_bad", "ERROR"))
    else:
        line.active = False
        line.label(text="No bad spots", **icons.kw("status_good", "CHECKMARK"))
    draw_review_button(box2, 1.5, len(bad))
    sub = box2.row()
    sub.active = False
    sub.label(text="Mark parts to fix, or try the sliders below" if bad else "Next: try the sliders below, then Apply", icon="FORWARD")


def draw_status(layout, gs, searching, done) -> None:
    message = G.status()
    show = bool(message) and not searching and not (done and message == G.state["auto_base"]["summary"])  # while searching the bar already says it
    if done:  # a result is shown: this box is always drawn at the same height, so the Adjust sliders below never jump
        import textwrap

        box = layout.box()
        if autofix_live.busy():
            autofix_live.draw_working(box)
            return
        problem = show and G._is_problem(message)
        box.alert = problem
        text = " | ".join(message.split(" | ")[:2]) if show else ""
        _slot(box, text, max(3, min(8, len(textwrap.wrap(text, width=32)))) if problem else 3, "ERROR" if problem else "INFO", 32)
        return
    if show:
        box = layout.box()
        box.alert = G._is_problem(message)
        for i, chunk in enumerate(message.split(" | ")[:2]):
            overlay.wrap_label(box, chunk, icon=("ERROR" if box.alert else "INFO") if i == 0 else "BLANK1", width=32, limit=4)


def draw_classic_notice(layout, gs) -> None:
    box = layout.box()
    words = gs.bl_rna.properties["scope"].enum_items[gs.scope].name
    box.label(text=f"This file uses Classic Guided ({words})", icon="INFO")
    row = box.row(align=True)
    _pro_toggle(row, "Open in Pro")
    row.prop_enum(gs, "scope", "AUTO", text="Use the main path")


class CURIOMESH_PT_guided(bpy.types.Panel):
    bl_label = "CurioMesh"
    bl_idname = "CURIOMESH_PT_guided"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"

    @classmethod
    def poll(cls, context):
        return bool(_prefs().show_guided)

    def draw_header(self, context):
        self.layout.label(text="", **icons.kw("logo", "PIVOT_ACTIVE"))

    def draw_header_preset(self, context):
        _pro_toggle(self.layout)

    def draw(self, context):
        layout, gs = self.layout, context.scene.curiomesh_guided
        draw_dashboard_button(layout, context)
        if gs.scope != "AUTO":
            draw_classic_notice(layout, gs)
            return
        draw_steps(layout, context)
        obj = G._source_mesh(context)
        if obj is None:
            draw_empty(layout, context)
            return
        draw_mesh_line(layout, obj)
        if not len(obj.data.polygons):
            box = layout.box()
            box.label(text="This mesh has no faces to reduce", icon="ERROR")
            overlay.wrap_label(box, "Pick a mesh with faces, or add some in Edit Mode.", icon="BLANK1", width=WIDTH, limit=2)
            return
        searching = G.progress_info("AUTO")
        done = result_ready()
        if searching:
            draw_working(layout, gs, searching)
        elif done:
            draw_verdict(layout, gs)
            autofix_live.draw_readout(layout, context)
            body = G.fold(layout, gs, "ui_setup_open", f"Settings: {setup_words(gs)}")
            if body is not None:
                draw_setup(body, context, gs, obj, again=True)
        else:
            draw_setup(layout, context, gs, obj)
        draw_status(layout, gs, searching, done)


# --------------------------------------------------------------------------- popovers of the main panel


class _Popover:
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"  # drawn only where layout.popover() opens it
    bl_ui_units_x = 14


def draw_shape_flow(layout, context, gs) -> None:
    """Everything you draw or paint on the mesh. The Flow Editor's button goes first once it is registered."""
    obj = G._source_mesh(context)
    col = layout.column(align=True)
    col.enabled = G.progress_info("AUTO") is None
    if hasattr(bpy.types, FLOW_EDITOR_OP[0]) and getattr(overlay.get_prefs(), "flow_editor", False):  # experimental for 0.9.8
        row = col.row()
        row.scale_y = 1.4
        row.operator(FLOW_EDITOR_OP[1], text="Draw on the mesh", **icons.kw("tool_pen", "GREASEPENCIL"))
        flowedit.draw_tools(col, gs)
        col.separator()
    if G.REGION_READY:
        col.prop(gs, "region_use", text="Selected Faces Only", toggle=True, icon="FACESEL")
    col.prop(gs, "loop_use", text="Edge Loops", toggle=True, **icons.kw("tool_ring", "MESH_CIRCLE"))
    if gs.loop_use:
        row = col.row(align=True)
        row.operator("curiomesh.guided_draw_guide", text="Draw Loops", icon="GREASEPENCIL").kind = "LOOP"
        row.operator("curiomesh.guided_add_guide", text="Add Selected", icon="ADD").kind = "LOOP"
        row = col.row(align=True)
        row.prop(gs, "loop_count")
        row.prop(gs, "loop_gap")
    col.prop(gs, "guide_use", text="Flow Lines", toggle=True, **icons.kw("tool_pen", "CURVE_DATA"))
    if gs.guide_use:
        row = col.row(align=True)
        row.operator("curiomesh.guided_draw_guide", text="Draw Line", icon="GREASEPENCIL").kind = "GUIDE"
        row.operator("curiomesh.guided_add_guide", text="Add Selected", icon="ADD").kind = "GUIDE"
    col.prop(gs, "bend_use", text="Joint Lines", toggle=True, **icons.kw("tool_bend", "MOD_SIMPLEDEFORM"))
    if gs.bend_use:
        row = col.row(align=True)
        row.operator("curiomesh.guided_draw_guide", text="Draw Joint", icon="GREASEPENCIL").kind = "BEND"
        row.operator("curiomesh.guided_add_guide", text="Add Selected", icon="ADD").kind = "BEND"
        row = col.row(align=True)
        row.prop(gs, "bend_loops")
        row.prop(gs, "loop_gap")
        row = col.row(align=True)
        row.prop(gs, "bend_crease", toggle=True)
        row.prop(gs, "bend_sharp", toggle=True)
    if len(gs.guide_curves):
        kinds = [g.kind for g in gs.guide_curves]
        col.label(text=f"{kinds.count('GUIDE')} flow lines, {kinds.count('LOOP')} rings, {kinds.count('BEND')} joint lines")
        row = col.row()
        row.template_list("CURIOMESH_UL_guided_guides", "", gs, "guide_curves", gs, "guide_index", rows=3)
        side = row.column(align=True)
        side.operator("curiomesh.guided_remove_guide", text="", icon="REMOVE")
        side.operator("curiomesh.guided_draw_guide", text="", icon="GREASEPENCIL").replace = True
        side.operator("curiomesh.guided_edit_guide", text="", icon="EDITMODE_HLT")
    elif not _compact():
        overlay.wrap_label(col, "Optional: draw where rows and loops should run, such as a ring round an eye or a line across a joint.", icon="INFO", width=WIDTH, limit=3)
    col.separator()
    col.prop(gs, "density_use", text="Detail Markers", toggle=True, **icons.kw("tool_more", "BRUSH_DATA"))
    if gs.density_use:
        row = col.row(align=True)
        row.operator("curiomesh.guided_paint_density", text="More Here", **icons.kw("tool_more", "WPAINT_HLT")).less = False
        row.operator("curiomesh.guided_paint_density", text="Less Here", **icons.kw("tool_less", "WPAINT_HLT")).less = True
        row = col.row(align=True)
        row.operator("curiomesh.guided_add_marker", text="Ball +", icon="ADD").less = False
        row.operator("curiomesh.guided_add_marker", text="Ball -", icon="REMOVE").less = True
        row.operator("curiomesh.guided_clear_density", text="", icon="X")
        col.prop(gs, "density_strength")
        balls = G.markers(context.scene)
        if balls:
            col.label(text=f"{sum(1 for b in balls if b[G.MARKER_KEY] > 0)} More and {sum(1 for b in balls if b[G.MARKER_KEY] < 0)} Less Detail balls", icon="SPHERE")
        elif obj is not None and obj.vertex_groups.get(gs.density_group) is None and obj.vertex_groups.get(gs.less_group) is None:
            overlay.wrap_label(col, "Paint More or Less Here, or put the 3D cursor on the mesh and add a ball; scale it to change its reach.", icon="INFO", width=WIDTH, limit=3)


class CURIOMESH_PT_pop_flow(_Popover, bpy.types.Panel):
    bl_label = "Shape the Flow"
    bl_idname = "CURIOMESH_PT_pop_flow"

    def draw(self, context):
        draw_shape_flow(self.layout, context, context.scene.curiomesh_guided)


def draw_options(layout, context, gs) -> None:
    """How the result is made, in three small groups. The rest is in the Pro tab."""
    obj = G._source_mesh(context)
    col = layout.column(align=True)
    col.enabled = G.progress_info("AUTO") is None
    col.label(text="Tidiness", **icons.kw("tidy", "MESH_GRID"))
    col.row(align=True).prop(gs, "auto_style", expand=True)
    col.row(align=True).prop(gs, "cleanup", expand=True)
    autofix_live.draw_option(col, context, gs)
    factor = hardware.style_factor(gs.auto_style, gs.auto_method)
    if factor > 1:
        col.label(text=f"Slow: about {factor:g}x Fewest Faces, or more", icon="TIME")
    if gs.auto_profile in profiles.MULTIRES_USES or gs.multires_on or (obj is not None and obj.data.uv_layers and gs.bake_detail != "NORMAL_COLOR"):
        col.separator()
        col.label(text="Detail", icon="MOD_MULTIRES")
    if gs.auto_profile in profiles.MULTIRES_USES or gs.multires_on:
        row = col.row(align=True)
        row.prop(gs, "multires_on", toggle=True, icon="MOD_MULTIRES")
        sub = row.row(align=True)
        sub.active = gs.multires_on
        sub.prop(gs, "multires_levels")
        if gs.multires_on and not _compact():
            cage = len(G.state["res"]["faces"]) if result_ready() else gs.auto_reduce_count
            overlay.wrap_label(col, profiles.multires_words(cage, gs.multires_levels), icon="INFO", width=WIDTH, limit=4)
            col.label(text="Slow: also builds the levels on Apply. Bake in Finish", icon="TIME")
    if obj is not None and obj.data.uv_layers and gs.bake_detail != "NORMAL_COLOR":
        col.prop(gs, "keep_uvs", toggle=True, icon="UV")
    col.separator()
    col.label(text="Hard Surface", icon="MOD_BEVEL")
    hard_surface.draw(col, gs)
    col.separator()
    col.label(text="Output", icon="MESH_DATA")
    col.row(align=True).prop(gs, "auto_output", expand=True)
    col.separator()
    _pro_toggle(col, "More in the Pro tab")


class CURIOMESH_PT_pop_options(_Popover, bpy.types.Panel):
    bl_label = "Options"
    bl_idname = "CURIOMESH_PT_pop_options"

    def draw(self, context):
        draw_options(self.layout, context, context.scene.curiomesh_guided)


class CURIOMESH_PT_pop_mirror(_Popover, bpy.types.Panel):
    bl_label = "Mirror"
    bl_idname = "CURIOMESH_PT_pop_mirror"

    def draw(self, context):
        gs = context.scene.curiomesh_guided
        col = self.layout.column(align=True)
        row = col.row(align=True)
        row.prop(gs, "sym_force", toggle=True, icon="MOD_MIRROR")
        row.prop(gs, "sym_side", text="")
        row = col.row(align=True)
        row.prop(gs, "sym_offset_use", toggle=True)
        sub = row.row(align=True)
        sub.active = gs.sym_offset_use
        sub.prop(gs, "sym_offset", text="")
        if G.state.get("sym_note"):
            overlay.wrap_label(col, G.state["sym_note"], icon="INFO", width=WIDTH, limit=2)
        base = G.state["auto_base"] if result_ready() else None
        seam = getattr(base and base.get("job"), "seam_report", None)
        if seam is not None:
            overlay.wrap_label(col, symmetry.seam_words(seam), icon="CHECKMARK" if not any(seam[k] for k in ("open", "non_manifold", "across", "duplicate")) else "ERROR", width=WIDTH, limit=2)


class CURIOMESH_PT_pop_uses(_Popover, bpy.types.Panel):
    bl_label = "More Uses"
    bl_idname = "CURIOMESH_PT_pop_uses"

    def draw(self, context):
        gs = context.scene.curiomesh_guided
        col = self.layout.column(align=True)
        col.operator("curiomesh.pick_use", text="Hard Surface", depress=not gs.use_auto and gs.auto_profile == "HARD", icon="MOD_BEVEL").use = "HARD"
        col.operator("curiomesh.pick_use", text="Hulls", depress=not gs.use_auto and gs.auto_profile == "HULL", **icons.kw("use_hull", "MESH_ICOSPHERE")).use = "HULL"
        if gs.auto_profile == "CUSTOM":
            col.label(text="Custom: your own settings from the Pro tab", icon="INFO")


# --------------------------------------------------------------------------- Adjust


class _Sub:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"
    bl_parent_id = "CURIOMESH_PT_guided"
    help_topic = ""

    def draw_header_preset(self, context):
        if self.help_topic:
            helpmod.help_button(self.layout, self.help_topic)


def part_edits() -> int:
    """Parts changed by hand (redone, switched or locked), which a whole-result slider would replace."""
    return sum(1 for r in segments.seg["parts"] if r.get("cur", 0) != 0 or r.get("locked"))


def draw_adjust(layout, gs) -> None:
    _help(layout, "adjust")
    edits = part_edits() if segments.ready() else 0
    if autofix_live.busy():  # the sliders would swap the result under the fixes
        note, icon = "Auto Fix is working. Wait or press Stop.", "INFO"
    elif tidylive.busy():  # the sliders would change the result under the open preview
        note, icon = "Finish or cancel the open tool.", "INFO"
    elif edits and ladder.active():
        note, icon = f"Sliders replace your {edits} part edit{'s' if edits != 1 else ''}", "ERROR"
    else:
        note, icon = "", "NONE"
    _slot(layout, note, 1, icon)  # always one row: the sliders under it do not move when a note comes and goes
    if autofix_live.busy() or tidylive.busy():
        layout = layout.column()
        layout.enabled = False
    if ladder.active():
        col = layout.column(align=True)
        row = col.row()
        row.label(text="Quad Size", **icons.kw("size", "FULLSCREEN_EXIT"))
        row.label(text=f"{len(G.state['res']['faces']):,} faces")
        col.prop(gs, "quad_size", text="", slider=True)
        _ends(col, "smaller quads", "bigger quads")
        col = layout.column(align=True)
        row = col.row()
        row.label(text="Rows", **icons.kw("style_rows", "MESH_GRID"))
        shown = ladder.lad["shown"]
        if shown is not None:
            a, b, moved = shown
            row.label(text=ladder.STYLE_WORDS.get(a[0], a[0]) if not moved else f"{len(moved)} pieces {ladder.STYLE_WORDS.get(b[0], b[0])}")
        col.prop(gs, "quad_order", text="", slider=True)
        _ends(col, "fewest faces", "even rows")
        _slot(layout, ladder.status(), 2, "TIME")
    col = layout.column(align=True)
    col.label(text="Tidy Up", **icons.kw("tidy", "MESH_GRID"))
    col.prop(gs, "organize", text="", slider=True)
    _ends(col, "as found", "full pass")
    organize.draw_again(col, gs)
    grid = col.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
    for prop in ("tidy_poles", "tidy_rows", "tidy_sizes", "tidy_square"):
        grid.prop(gs, prop, toggle=True)
    col.prop(gs, "tidy_selected", toggle=True, icon="RESTRICT_SELECT_OFF")
    if organize.active():
        sub = layout.column(align=True)
        sub.active = False
        _slot(sub, organize.meter_text(gs), 3)
        _slot(layout, organize.status(), 2, "TIME")
    faceloops.draw(layout, gs)
    draw_view_strip(layout, gs)
    layout.popover(panel="CURIOMESH_PT_pop_finetune", text="Fine-tune...", icon="PREFERENCES")


def draw_view_strip(layout, gs) -> None:
    """Five icon-only view switches and where your own mesh sits."""
    row = layout.row(align=True)
    for prop, name, stock in (("seg_colors", "view_parts", "COLOR"), ("seg_heat", "view_change", "SEQ_HISTOGRAM"), ("seg_wire", "view_wire", "SHADING_WIRE"), ("seg_bad", "view_bad", "ERROR"), ("organize_flow", "view_flow", "FORCE_CURVE")):
        row.prop(gs, prop, text="", toggle=True, **icons.kw(name, stock))
    row.prop(gs, "seg_compare", text="")
    if gs.seg_heat and not _compact():
        sub = layout.row()
        sub.active = False
        sub.label(text="Purple: not changed, yellow: changed most")


class CURIOMESH_PT_guided_adjust(_Sub, bpy.types.Panel):
    bl_label = "Adjust"
    bl_idname = "CURIOMESH_PT_guided_adjust"
    help_topic = "adjust"

    @classmethod
    def poll(cls, context):
        return _auto_tab(context) and result_ready() and G.progress_info("AUTO") is None

    def draw_header(self, context):
        self.layout.label(text="", **icons.kw("step_adjust", "OPTIONS"))

    def draw(self, context):
        draw_adjust(self.layout, context.scene.curiomesh_guided)


class CURIOMESH_PT_pop_finetune(_Popover, bpy.types.Panel):
    bl_label = "Fine-tune"
    bl_idname = "CURIOMESH_PT_pop_finetune"

    def draw(self, context):
        gs = context.scene.curiomesh_guided
        col = self.layout.column(align=True)
        col.enabled = G.progress_info() is None
        if gs.auto_style == "ROWS" or gs.auto_method != "AUTO":
            overlay.wrap_label(col, "These three work with Fewest Faces and Balanced, Method Auto", icon="INFO", width=WIDTH, limit=2)
        else:
            col.prop(gs, "tweak_rows", slider=True)
            col.prop(gs, "tweak_flat", slider=True)
            if gs.auto_style == "DETAIL":
                col.prop(gs, "tweak_poles", toggle=True)
            if gs.density_use or gs.guide_use or gs.loop_use or gs.bend_use:
                overlay.wrap_label(col, "Detail markers and drawn lines turn the first two off", icon="INFO", width=WIDTH, limit=2)
        col.separator()
        row = col.row(align=True)
        row.prop(gs, "organize_cap_use", toggle=True)
        sub = row.row(align=True)
        sub.active = gs.organize_cap_use
        sub.prop(gs, "organize_cap", text="")


# --------------------------------------------------------------------------- Fix


def _is_original(part: int) -> bool:
    parts = segments.seg["parts"]
    return 0 <= part < len(parts) and parts[part]["method"] == segments.ORIGINAL


def draw_spots_line(layout) -> None:
    """How many bad spots there are, and the way back for parts that show your own faces."""
    found = issue_panel.spots()
    row = layout.row()
    if found:
        row.alert = any(any(k in V.SERIOUS for k in s.get("kinds", {})) for s in found)
        row.label(text=f"{len(found)} spot{'s' if len(found) != 1 else ''} with bad faces", **icons.kw("status_bad", "ERROR"))
    else:
        row.label(text="No bad spots found", **icons.kw("status_good", "CHECKMARK"))
    kept = issue_panel.original_parts()
    if kept:
        row = layout.row(align=True)
        row.label(text=f"{len(kept)} part{'s' if len(kept) != 1 else ''} show your own faces", **icons.kw("keep_original", "LOOP_BACK"))
        row.operator("curiomesh.issue_put_back", text="Put reduced back", icon="LOOP_FORWARDS").part = -1


def draw_part_card(layout, gs) -> None:
    """The selected part: quad size, its Use and Redo, and its tries. Everything else is in its click popover."""
    selected = sorted(segments.seg["sel"])
    if not selected:
        sub = layout.row()
        sub.active = False
        sub.label(text="Click a part in the list or in the viewport", icon="INFO")
        return
    box = layout.box()
    if len(selected) == 1:
        r = segments.seg["parts"][selected[0]]
        box.label(text=f"{segments.name_of(selected[0])}: {r['after']:,} faces, {segments.method_name(r['method'])}", icon="OBJECT_DATA")
    else:
        box.label(text=f"{len(selected)} parts, {sum(segments.seg['parts'][p]['after'] for p in selected):,} faces", icon="OBJECT_DATA")
    row = box.row(align=True)  # main-only
    row.operator("curiomesh.parts_export", text="Send to Agent", icon="EXPORT")  # main-only
    row.operator("curiomesh.parts_import", text="Bring Back", icon="IMPORT")  # main-only
    row.operator("curiomesh.parts_report", text="", icon="FILE_TEXT")  # main-only
    sub = box.row()
    sub.active = False
    sub.label(text="Redo, quad size and tries are in Review Parts below", icon="INFO")


def draw_fix(layout, context, gs) -> None:
    segments.ensure()
    if not segments.ready():
        layout.label(text="Parts appear when the search is done", icon="INFO")
        return
    _help(layout, "fix")
    draw_review_button(layout, 1.5, len(issue_panel.spots()))
    draw_spots_line(layout)
    layout.label(text="Parts", **icons.kw("view_parts", "MOD_EXPLODE"))
    row = layout.row()
    row.template_list("CURIOMESH_UL_segparts", "", gs, "seg_parts", gs, "seg_index", rows=4)
    col = row.column(align=True)
    col.operator("curiomesh.seg_select", text="", icon="SORTSIZE").what = "WORST"
    col.operator("curiomesh.seg_frame", text="", icon="VIEWZOOM")
    draw_part_card(layout, gs)
    row = layout.row(align=True)
    row.popover(panel="CURIOMESH_PT_pop_partedit", text="Edit parts...", **icons.kw("view_parts", "AUTOMERGE_ON"))
    row.popover(panel="CURIOMESH_PT_pop_finder", text="Finder...", icon="VIEWZOOM")


class CURIOMESH_PT_guided_parts(_Sub, bpy.types.Panel):
    bl_label = "Fix"
    bl_idname = "CURIOMESH_PT_guided_parts"
    help_topic = "fix"

    @classmethod
    def poll(cls, context):
        return _auto_tab(context) and result_ready()

    def draw_header(self, context):
        self.layout.label(text="", **icons.kw("step_fix", "TOOL_SETTINGS"))

    def draw(self, context):
        draw_fix(self.layout, context, context.scene.curiomesh_guided)


class CURIOMESH_PT_pop_partedit(_Popover, bpy.types.Panel):
    bl_label = "Edit Parts"
    bl_idname = "CURIOMESH_PT_pop_partedit"

    def draw(self, context):
        segment_goals.draw_edit(self.layout)
        sub = self.layout.row()
        sub.active = False
        sub.label(text=f"{len(segments.seg['parts']):,} parts, {len(segments.seg['sel'])} selected")
        row = self.layout.row(align=True)
        for what, text in (("ALL", "All"), ("TUBE", "Tubes"), ("FLAT", "Flat"), ("ROUGH", "Rough"), ("NONE", "None")):
            row.operator("curiomesh.seg_select", text=text).what = what


class CURIOMESH_PT_pop_finder(_Popover, bpy.types.Panel):
    bl_label = "Part Finder"
    bl_idname = "CURIOMESH_PT_pop_finder"

    def draw(self, context):
        segment_goals.draw_finder(self.layout)


# --------------------------------------------------------------------------- Finish


def draw_finish(layout, context, gs) -> None:
    done = result_ready()
    busy = G.progress_info() is not None
    if not done and G.state["applied"]:
        overlay.wrap_label(layout, f"Applied: {G.state['applied']}", icon="CHECKMARK", width=30, limit=4)
        layout.operator("curiomesh.export_game", text="Export for game...", **icons.kw("export", "EXPORT"))
        uv_panel.draw_row(layout, context)  # main-only: uv
        G._draw_thanks(layout)
        return
    _help(layout, "finish")
    row = layout.row()
    row.scale_y = 2.0
    row.enabled = done and not busy
    row.operator("curiomesh.guided_apply", text="Apply", **icons.kw("step_finish", "CHECKMARK"))
    layout.row(align=True).prop(gs, "apply_mode", expand=True)
    row = layout.row(align=True)
    row.prop(gs, "bake_detail", text="Bake", **icons.kw("bake", "IMAGE_DATA"))
    row.popover(panel="CURIOMESH_PT_pop_bake", text="", icon="PREFERENCES")
    helpmod.help_button(row, "bake")
    uv_panel.draw_row(layout, context)  # main-only: uv
    row = layout.row(align=True)
    row.popover(panel="CURIOMESH_PT_pop_lod", text="LOD set...", **icons.kw("lod", "MOD_DECIM"))
    helpmod.help_button(row, "lod")
    if done and not busy:
        layout.operator("curiomesh.guided_stop", text="Discard result", icon="X")


class CURIOMESH_PT_guided_apply(_Sub, bpy.types.Panel):
    bl_label = "Finish"
    bl_idname = "CURIOMESH_PT_guided_apply"
    help_topic = "finish"

    @classmethod
    def poll(cls, context):
        return _auto_tab(context) and (G._source_mesh(context) is not None or bool(G.state["applied"]))

    def draw_header(self, context):
        self.layout.label(text="", **icons.kw("step_finish", "CHECKMARK"))

    def draw(self, context):
        draw_finish(self.layout, context, context.scene.curiomesh_guided)


class CURIOMESH_PT_better(_Sub, bpy.types.Panel):
    """Help Make It Better: mark spots in the 3D view, then share them (or a fix) on GitHub."""
    bl_label = "Help Make It Better"
    bl_idname = "CURIOMESH_PT_better"
    bl_options = {"DEFAULT_CLOSED"}
    help_topic = "better"

    @classmethod
    def poll(cls, context):
        return _auto_tab(context)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Share spots CurioMesh gets wrong, or your fix.")
        if segments.ready():
            draw_review_button(layout, 1.0)
            n = len(segments.seg["sel"])
            layout.label(text=f"{n} part(s) marked" if n else "Click bad spots in the 3D view to mark them", icon="RESTRICT_SELECT_OFF")
            row = layout.row(align=True)
            row.operator("curiomesh.send_problem", text="Send as a Problem", icon="ERROR").kind = "PROBLEM"
            row.operator("curiomesh.send_problem", text="Share a Fix", icon="CHECKMARK").kind = "FIX"
        else:
            layout.label(text="Run Auto first to pick spots.", icon="INFO")
        row = layout.row(align=True)
        row.operator("wm.url_open", text="Report a Bug", icon="URL").url = G.problem_url()
        row.operator("wm.url_open", text="How to Help", icon="HELP").url = G.REPO_URL + "/blob/main/CONTRIBUTING.md"


class CURIOMESH_PT_pop_bake(_Popover, bpy.types.Panel):
    bl_label = "Bake"
    bl_idname = "CURIOMESH_PT_pop_bake"

    def draw(self, context):
        layout, gs = self.layout, context.scene.curiomesh_guided
        obj = G._source_mesh(context)
        col = layout.column(align=True)
        col.prop(gs, "bake_detail", text="")
        sub = col.column(align=True)
        sub.active = gs.bake_detail != "OFF"
        sub.prop(gs, "bake_size")
        if gs.bake_detail == "NORMAL_COLOR":
            col.prop(gs, "bake_sets")
        elif obj is not None and obj.data.uv_layers:
            overlay.wrap_label(col, "For textured game assets, Normal + Color is recommended", icon="INFO", width=WIDTH, limit=2)
        G._bake_tag(col, gs)


class CURIOMESH_PT_pop_lod(_Popover, bpy.types.Panel):
    bl_label = "LOD Set"
    bl_idname = "CURIOMESH_PT_pop_lod"

    def draw(self, context):
        layout, gs = self.layout, context.scene.curiomesh_guided
        col = layout.column(align=True)  # one per line: three in a row were cut in a narrow sidebar
        col.prop(gs, "lod0_percent")
        col.prop(gs, "lod1_percent")
        col.prop(gs, "lod2_percent")
        col.operator("curiomesh.guided_make_lods", icon="MOD_DECIM")
        obj = G._source_mesh(context)
        if G.state["machine"] is not None and obj is not None:
            _, _, _, seconds, bake_s = G.job_cost(gs, len(obj.data.polygons), G.state["src"]["sig"] if G.state["src"] is not None else None, profiles.LOD_LEVELS)
            G._tag(col, hardware.cost_tag(seconds + bake_s))
        col.label(text="Searches once, then makes 3 sizes", icon="INFO")
        for line in G.state["lod_report"].split(" | ") if G.state["lod_report"] else ():
            overlay.wrap_label(col, line, icon="CHECKMARK", width=WIDTH, limit=2)


# --------------------------------------------------------------------------- Pro tab


class _Pro:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh Pro"
    bl_options = {"DEFAULT_CLOSED"}
    help_topic = ""

    def draw_header_preset(self, context):
        if self.help_topic:
            helpmod.help_button(self.layout, self.help_topic)

    @classmethod
    def poll(cls, context):
        return bool(getattr(_prefs(), "pro_mode", False))


class CURIOMESH_PT_pro_engine(_Pro, bpy.types.Panel):
    bl_label = "Engine"
    bl_idname = "CURIOMESH_PT_pro_engine"
    help_topic = "engine"
    bl_order = 0

    def draw_header(self, context):
        self.layout.label(text="", **icons.kw("pro", "PREFERENCES"))

    def draw(self, context):
        layout, gs = self.layout, context.scene.curiomesh_guided
        _help(layout, "pro")
        col = layout.column(align=True)
        col.enabled = G.progress_info("AUTO") is None
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(gs, "auto_profile")
        col.prop(gs, "auto_style")
        col.prop(gs, "auto_method")
        factor = hardware.style_factor(gs.auto_style, gs.auto_method)
        if factor > 1:
            col.label(text=f"Slow: about {factor:g}x Fewest Faces, or more", icon="TIME")
        col.prop(gs, "auto_output")
        col.prop(gs, "auto_aim")
        col = layout.column(align=True)
        col.enabled = G.progress_info("AUTO") is None
        col.label(text="Cleanup, one by one")
        col.prop(gs, "auto_redo_rough", toggle=True, icon="MOD_SMOOTH")
        col.prop(gs, "auto_fix", toggle=True, icon="SHADERFX")
        col.prop(gs, "auto_chords", toggle=True, icon="MOD_REMESH")
        col.prop(gs, "auto_tidy", toggle=True, icon="MESH_GRID")
        col.prop(gs, "time_limits", text="Time")
        col.label(text="What is kept")
        col.prop(gs, "keep_uvs", toggle=True, icon="UV")
        col.prop(gs, "auto_instances", toggle=True, icon="LINKED")
        col.prop(gs, "auto_drop_hidden", toggle=True, icon="HIDE_ON")
        col.prop(gs, "auto_rough", toggle=True, icon="TEXTURE")
        if gs.auto_profile == "LOW_POLY":
            col.prop(gs, "auto_baked", toggle=True, icon="EXPERIMENTAL")
        row = col.row(align=True)
        row.prop(gs, "multires_on", toggle=True, icon="MOD_MULTIRES")
        row.prop(gs, "multires_levels")
        if G.REGION_READY:
            col.prop(gs, "region_use", toggle=True, icon="FACESEL")
        col.label(text="Fine-tune")
        col.prop(gs, "tweak_rows", slider=True)
        col.prop(gs, "tweak_flat", slider=True)
        col.prop(gs, "tweak_poles", toggle=True)


def draw_computer(layout, gs) -> None:
    """The machine in plain words, how hard to work, and handing the job to another computer."""
    machine = G.state["machine"]
    searching = G.progress_info("AUTO") is not None
    col = layout.column(align=True)
    if machine is None:
        col.label(text="Not checked yet", icon="INFO")
    else:
        for line in hardware.describe(machine, G.turbo_words(G.turbo.status())):
            overlay.wrap_label(col, line, icon="BLANK1", width=WIDTH, limit=2)
        if hardware.weak_pc(machine):
            overlay.wrap_label(col, hardware.weak_pc(machine), icon="ERROR", width=WIDTH, limit=5)
    sub = layout.column()
    sub.enabled = not searching
    sub.prop(gs, "auto_effort")
    sub.prop(gs, "auto_gpu", toggle=True, icon="SHADING_RENDERED")
    if machine is not None and gs.auto_effort == "AUTO":
        sub.label(text=f"Auto picked {hardware.settings_for(gs.auto_effort, machine).name}", icon="BLANK1")
    sub.prop(gs, "auto_parallel", toggle=True, icon="MOD_ARRAY")
    sub.operator("curiomesh.guided_check_pc", icon="FILE_REFRESH")
    row = sub.row(align=True)
    row.operator("curiomesh.guided_export_job", text="Export Job", icon="EXPORT")
    row.operator("curiomesh.guided_import_result", text="Import Result", icon="IMPORT")
    row = sub.row(align=True)  # main-only
    row.prop(bpy.context.window_manager, "curiomesh_agent_on", toggle=True, icon="CONSOLE")  # main-only
    helpmod.help_button(row, "agent")  # main-only


class CURIOMESH_PT_pro_computer(_Pro, bpy.types.Panel):
    bl_label = "This Computer"
    bl_idname = "CURIOMESH_PT_pro_computer"
    help_topic = "computer"
    bl_order = 1

    def draw_header(self, context):
        self.layout.label(text="", **icons.kw("computer", "DESKTOP"))

    def draw(self, context):
        draw_computer(self.layout, context.scene.curiomesh_guided)


class CURIOMESH_PT_pro_prompt(_Pro, bpy.types.Panel):
    bl_label = "Write a Prompt for My AI"
    bl_idname = "CURIOMESH_PT_pro_prompt"
    help_topic = "prompt"
    bl_order = 2

    def draw_header(self, context):
        self.layout.label(text="", icon="COPYDOWN")

    def draw(self, context):
        promptgen.draw(self.layout, context)


class CURIOMESH_PT_guided_shapes(_Pro, bpy.types.Panel):
    bl_label = "Inside Shapes"
    bl_idname = "CURIOMESH_PT_guided_shapes"
    help_topic = "shapes"
    bl_order = 3

    def draw_header(self, context):
        self.layout.label(icon="CUBE")

    def draw(self, context):
        layout, gs = self.layout, context.scene.curiomesh_guided
        done = result_ready()
        redoing = G.progress_info("BOOL")
        col = layout.column()
        if not done:
            col.label(text="Needs a finished result", icon="INFO")
            return
        col.label(text="Put a shape over a part to redo it live:")
        row = col.row(align=True)
        row.operator("curiomesh.guided_add_shape", text="Box", icon="CUBE").kind = "BOX"
        row.operator("curiomesh.guided_add_shape", text="Sphere", icon="MESH_UVSPHERE").kind = "SPHERE"
        row.operator("curiomesh.guided_add_shape", text="Selected", icon="RESTRICT_SELECT_OFF").kind = "SELECTED"
        if len(gs.bool_shapes):
            row = col.row()
            row.template_list("CURIOMESH_UL_guided_shapes", "", gs, "bool_shapes", gs, "bool_index", rows=2)
            row.operator("curiomesh.guided_remove_shape", text="", icon="REMOVE")
            col.prop(gs, "bool_on", toggle=True, icon="HIDE_OFF" if gs.bool_on else "HIDE_ON")
            row = col.row(align=True)
            row.label(text="Organized")
            row.prop(gs, "bool_detail", text="")
            row.label(text="Detailed")
            col.row(align=True).prop(gs, "strategy", expand=True)
            if redoing:
                G.draw_progress(col, *redoing)
                col.operator("curiomesh.guided_stop_work", icon="CANCEL")
            col.operator("curiomesh.seg_from_shapes", icon="MOD_BOOLEAN")
        else:
            overlay.wrap_label(col, "Union, Subtract and Intersect combine them", width=WIDTH, limit=2)


def draw_teach(layout, obj) -> None:
    """Fix a spot by hand, remember it, and find the same spot next time."""
    col = layout.column(align=True)
    overlay.wrap_label(col, "Fix a spot by hand, then:", width=WIDTH, limit=2)
    row = col.row(align=True)
    row.operator("curiomesh.guided_edit_preview", text="Edit Preview", icon="EDITMODE_HLT")
    row.operator("curiomesh.teach_remember", text="Remember", icon="BOOKMARKS")
    known = len(G._library())
    col.label(text=f"{known} fix{'' if known == 1 else 'es'} remembered on this PC")
    if G.state["fixes_in_memory"]:
        overlay.wrap_label(col, "Blender's config folder can't be written, so these last only until you close Blender. Export them to keep them.", icon="ERROR", width=WIDTH, limit=4)
    if G.state["found"]:
        col.label(text="You've fixed something like this before:", icon="INFO")
        col.operator("curiomesh.teach_apply", text=f"Apply ({len(G.state['found'])} places)", icon="CHECKMARK")
    else:
        col.operator("curiomesh.teach_find", icon="VIEWZOOM")
    row = col.row(align=True)
    row.operator("curiomesh.teach_export", icon="EXPORT")
    row.operator("curiomesh.teach_import", icon="IMPORT")
    kept = len(G._kept(obj)) if obj is not None else 0
    if kept:
        col.operator("curiomesh.teach_forget_kept", text=f"Forget {kept} Kept Fix{'' if kept == 1 else 'es'}", icon="X")


class CURIOMESH_PT_pro_teach(_Pro, bpy.types.Panel):
    bl_label = "Show It How"
    bl_idname = "CURIOMESH_PT_pro_teach"
    help_topic = "teach"
    bl_order = 4

    def draw_header(self, context):
        self.layout.label(text="", icon="BOOKMARKS")

    def draw(self, context):
        draw_teach(self.layout, G._source_mesh(context))


class CURIOMESH_PT_pro_reports(_Pro, bpy.types.Panel):
    bl_label = "Issue Reports"
    bl_idname = "CURIOMESH_PT_pro_reports"
    help_topic = "reports"
    bl_order = 5

    def draw_header(self, context):
        self.layout.label(text="", icon="FILE_TICK")

    def draw(self, context):
        issue_panel.draw_reports(self.layout)


class CURIOMESH_PT_pro_classic(_Pro, bpy.types.Panel):
    """The point tools from before Auto Segments: Whole Mesh, Points Only and Work Areas."""

    bl_label = "Classic Guided"
    bl_idname = "CURIOMESH_PT_pro_classic"
    help_topic = "classic"
    bl_order = 6

    def draw_header(self, context):
        self.layout.label(text="", icon="PIVOT_ACTIVE")

    def draw(self, context):
        layout = self.layout
        gs = context.scene.curiomesh_guided
        layout.prop(gs, "scope", text="")
        if gs.scope == "AUTO":
            overlay.wrap_label(layout, "Pick Whole Mesh, Points Only or Work Areas to use the point tools. Auto Segments is the CurioMesh tab.", icon="INFO", width=WIDTH, limit=3)
            return
        draw_classic(layout, context, gs)


def draw_classic(layout, context, gs) -> None:
    compact = _compact()
    obj = G._source_mesh(context)
    previewing = G.state["res"] is not None and G.preview_object() is not None
    if obj is None:
        layout.box().label(text="Select a mesh to begin", icon="INFO")
    layout.label(text="1. Choose a preset & place points", icon="PIVOT_ACTIVE")
    layout.prop(gs, "preset", text="")
    row = layout.row(align=True)
    row.scale_y = 1.4
    row.operator("curiomesh.guided_place", text="Place Points", icon="MOUSE_LMB")
    row.operator("curiomesh.guided_suggest", text="Auto-suggest", icon="LIGHT")
    row = layout.row()
    row.scale_y = 1.4
    row.operator("curiomesh.guided_work", icon="BRUSH_DATA")
    if obj is not None:
        row = layout.row()
        row.template_list("CURIOMESH_UL_guided_points", "", obj, "curiomesh_points", obj, "curiomesh_point_index", rows=3)
        col = row.column(align=True)
        col.operator("curiomesh.guided_add_point", text="", icon="ADD")
        col.operator("curiomesh.guided_remove_point", text="", icon="REMOVE")
        col.operator("curiomesh.guided_clear_points", text="", icon="TRASH")
        col.operator("curiomesh.guided_add_empty", text="", icon="EMPTY_AXIS")
        if 0 <= obj.curiomesh_point_index < len(obj.curiomesh_points):
            item = obj.curiomesh_points[obj.curiomesh_point_index]
            col = layout.column(align=True)
            col.prop(item, "kind", text="")
            col.prop(item, "fill", text="")
            col.prop(item, "radius")
            col.prop(item, "strength")
            col.prop(item, "co", text="Location")
        elif not len(obj.curiomesh_points) and not compact:
            layout.label(text="Detail = smaller quads, Calm = bigger,", icon="INFO")
            layout.label(text="Loop = rings around, Sharp = keep edges")
    col = layout.column(align=True)
    col.prop(gs, "new_kind", text="Next")
    col.prop(gs, "new_radius", text="Radius")
    col.prop(gs, "new_strength", text="Strength")

    layout.label(text="2. Watch the live preview", icon="HIDE_OFF")
    layout.prop(gs, "enabled", toggle=True, text="Live Preview & Cage", icon="HIDE_OFF" if gs.enabled else "HIDE_ON")
    layout.row(align=True).prop(gs, "strategy", expand=True)
    layout.prop(gs, "base_faces")
    row = layout.row(align=True)
    row.scale_y = 1.3
    row.operator("curiomesh.guided_update", text="Update", icon="FILE_REFRESH")
    row.prop(gs, "auto_update", text="Auto", toggle=True)
    working = G.progress_info()
    if working:
        G.draw_progress(layout, *working)
        layout.operator("curiomesh.guided_stop_work", icon="CANCEL")
    message = G.status()
    if message:
        box = layout.box()
        box.alert = G._is_problem(message)
        for i, chunk in enumerate(message.split(" | ")[:2]):
            overlay.wrap_label(box, chunk, icon=("ERROR" if box.alert else "INFO") if i == 0 else "BLANK1", width=32, limit=4)
    if G.state["spot"] is not None:
        box = layout.box()
        box.label(text="Next spot to fix", icon="ZOOM_ALL")
        overlay.wrap_label(box, G.state["spot"]["text"], width=WIDTH, limit=2)
        box.operator("curiomesh.guided_frame", icon="VIEWZOOM")

    layout.label(text="3. Lock, redraw, apply", icon="CHECKMARK")
    col = layout.column(align=True)
    col.enabled = previewing
    col.operator("curiomesh.guided_edit_preview", icon="EDITMODE_HLT")
    row = col.row(align=True)
    row.operator("curiomesh.guided_lock", icon="LOCKED")
    row.operator("curiomesh.guided_redraw_strip", icon="MOD_SOLIDIFY")
    if G.state["lock"] is not None:
        layout.operator("curiomesh.guided_unlock", icon="UNLOCKED", text=f"Unlock All ({int(G.state['lock']['locked'].sum()):,})")
    row = layout.row(align=True)
    row.scale_y = 1.6 if previewing else 1.2
    sub = row.row(align=True)
    sub.enabled = previewing
    sub.operator("curiomesh.guided_apply", icon="CHECKMARK")
    row.operator("curiomesh.guided_stop", icon="X")
    layout.prop(gs, "apply_mode", text="")
    layout.prop(gs, "bake_detail", text="Bake")
    if gs.bake_detail != "OFF":
        layout.prop(gs, "bake_size", text="")
    if gs.bake_detail == "NORMAL_COLOR":
        layout.prop(gs, "bake_sets")
    G._bake_tag(layout, gs)
    if not previewing and not compact:
        overlay.wrap_label(layout, "Nothing changes until you press Apply.", icon="LOCKED", width=WIDTH, limit=2)


class CURIOMESH_PT_guided_advanced(_Pro, bpy.types.Panel):
    bl_label = "Advanced"
    bl_idname = "CURIOMESH_PT_guided_advanced"
    help_topic = "advanced"
    bl_parent_id = "CURIOMESH_PT_pro_classic"

    def draw_header(self, context):
        self.layout.label(icon="PREFERENCES")

    def draw(self, context):
        layout = self.layout
        gs = context.scene.curiomesh_guided
        layout.prop(gs, "show_cage")
        layout.prop(gs, "feature_angle")
        layout.prop(gs, "fit_iterations")
        layout.prop(gs, "pure_quads")
        layout.prop(gs, "suggest_count")


# --------------------------------------------------------------------------- registration

tidytools.CURIOMESH_PT_guided_history.draw_header_preset = lambda self, context: helpmod.help_button(self.layout, "history")  # History has no "?" of its own

MAIN_PANELS = (CURIOMESH_PT_guided, CURIOMESH_PT_guided_adjust, *tidytools.PANELS, CURIOMESH_PT_guided_parts, *segment_goals.PANELS, CURIOMESH_PT_guided_apply, CURIOMESH_PT_better)
PRO_PANELS = (
    CURIOMESH_PT_pro_engine,
    CURIOMESH_PT_pro_computer,
    CURIOMESH_PT_pro_prompt,
    CURIOMESH_PT_guided_shapes,
    CURIOMESH_PT_pro_teach,
    CURIOMESH_PT_pro_reports,
    CURIOMESH_PT_pro_classic,
    CURIOMESH_PT_guided_advanced,
)
POPOVERS = (
    CURIOMESH_PT_pop_flow,
    CURIOMESH_PT_pop_options,
    CURIOMESH_PT_pop_mirror,
    CURIOMESH_PT_pop_uses,
    CURIOMESH_PT_pop_finetune,
    CURIOMESH_PT_pop_partedit,
    CURIOMESH_PT_pop_finder,
    CURIOMESH_PT_pop_bake,
    CURIOMESH_PT_pop_lod,
)
CLASSES = (CURIOMESH_OT_pick_use, *POPOVERS, *MAIN_PANELS, *PRO_PANELS)


def register() -> None:
    icons.load()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    autofix_live.register()


def unregister() -> None:
    autofix_live.unregister()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    icons.free()
