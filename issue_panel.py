# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""The Issues box of the Fix Parts panel: the worst spots of a finished result (faces turned inward, folded, crossing,
slivers, pinched faces, poles...), a way to jump to each, a saved record, and a comparison with an older record.

The spots come from `triadq.issues`, made at the end of every Auto Segments run (`res["issue_spots"]`)."""

from __future__ import annotations

import json
import os
import time

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Vector

from . import overlay
from .triadq import issues

SPOT_COLOUR = (1.0, 0.45, 0.0, 0.6)
CURRENT_COLOUR = (1.0, 1.0, 0.0, 0.95)
KIND_WORDS = {
    "inward": "turned inward", "sunk": "sunk into your mesh", "crossing": "crossing", "folded_over": "folded over",
    "bent": "bent", "flat": "flat", "duplicate": "duplicate", "overlap": "overlapping",
    "sliver": "slivers", "pinched": "pinched", "pole": "poles", "concave": "concave",
    "touching": "touching parts",
}

ui = {"res": None, "cur": -1, "diff": None, "batch": None, "batch_key": None}


def _g():
    from . import guided

    return guided


def spots() -> list:
    res = _g().state["res"]
    if res is None:
        return []
    if ui["res"] is not res:  # a new result: start the walk over
        ui.update(res=res, cur=-1, diff=None, batch=None, batch_key=None)
    return res.get("issue_spots") or []


def summary(kinds: dict) -> str:
    return ", ".join(f"{c:,} {KIND_WORDS.get(k, k)}" for k, c in sorted(kinds.items(), key=lambda kv: -issues.WEIGHTS.get(kv[0], 0) * kv[1])[:3])


def _frame(context, spot) -> bool:
    guided = _g()
    rv3d = getattr(context.space_data, "region_3d", None) or context.region_data
    preview = guided.preview_object()
    if rv3d is None or preview is None:
        return False
    centre = guided._world(preview.matrix_world, np.asarray([spot["center"]]))[0]
    scale = float(np.linalg.norm(np.asarray(preview.matrix_world.to_scale())) / 3 ** 0.5)
    rv3d.view_location = Vector(centre.tolist())
    rv3d.view_distance = max(spot["radius"] * scale * 4.0, 1e-3)
    return True


def _select(context, spot) -> bool:
    import bmesh

    preview = _g().preview_object()
    if preview is None:
        return False
    if preview.mode != "EDIT":  # leaves any other mode first, and makes the preview (unselectable otherwise) active, faces mode
        bpy.ops.curiomesh.guided_edit_preview()
    if preview.mode != "EDIT":
        return False
    bm = bmesh.from_edit_mesh(preview.data)
    bm.select_mode = {"FACE"}
    bm.faces.ensure_lookup_table()
    for f in bm.faces:
        f.select_set(False)
    for i in spot["faces"]:
        if i < len(bm.faces):
            bm.faces[i].select_set(True)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(preview.data)
    return True


def _go(context, index: int) -> None:
    found = spots()
    if not found:
        return
    ui["cur"] = index % len(found)
    _frame(context, found[ui["cur"]])
    overlay.tag_redraw()


class _IssueOp:
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return bool(spots())


class CURIOMESH_OT_issue_view(_IssueOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_view"
    bl_label = "View"
    bl_description = "Moves the 3D view to this bad spot and highlights it"

    index: bpy.props.IntProperty(name="Spot", description="Which bad spot, 0 is the worst")

    def execute(self, context):
        _go(context, self.index)
        return {"FINISHED"}


class CURIOMESH_OT_issue_select(_IssueOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_select"
    bl_label = "Select"
    bl_description = "Select the faces of this problem spot in Edit Mode"

    index: bpy.props.IntProperty(name="Spot", description="Which bad spot, 0 is the worst")

    def execute(self, context):
        found = spots()
        if not 0 <= self.index < len(found):
            return {"CANCELLED"}
        ui["cur"] = self.index
        if not _select(context, found[self.index]):
            self.report({"WARNING"}, "The preview is gone: run Auto again")
            return {"CANCELLED"}
        overlay.tag_redraw()
        return {"FINISHED"}


class CURIOMESH_OT_issue_step(_IssueOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_step"
    bl_label = "Next Issue"
    bl_description = "Goes to the next (or previous) bad spot, worst first, and moves the view to it"

    step: bpy.props.IntProperty(name="Step", description="1 goes to the next bad spot, -1 to the one before", default=1)

    def execute(self, context):
        _go(context, ui["cur"] + self.step if ui["cur"] >= 0 else (0 if self.step > 0 else -1))
        return {"FINISHED"}


def spot_part(index: int) -> int:
    """The part a spot sits on, or -1."""
    found = spots()
    return int(found[index].get("part", -1)) if 0 <= index < len(found) else -1


class _SpotPartOp(_IssueOp):
    index: bpy.props.IntProperty(name="Spot", description="Which bad spot, 0 is the worst")

    @classmethod
    def poll(cls, context):
        from . import segments

        segments.ensure()
        return bool(spots()) and segments.ready() and not segments.mirrored() and _g().progress_info() is None

    def _part(self):
        from . import segments

        p = spot_part(self.index)
        return p if 0 <= p < len(segments.seg["parts"]) else None


class CURIOMESH_OT_issue_redo(_SpotPartOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_redo"
    bl_label = "Redo This Spot"
    bl_description = "Remakes the part under this spot with smaller quads, which usually clears folded or crossing faces. The old version stays in the part's tries"

    @classmethod
    def description(cls, context, properties):
        from . import segments

        p = spot_part(properties.index)
        if not 0 <= p < len(segments.seg["parts"]):
            return cls.bl_description
        return f"Remakes {segments.name_of(p)} ({segments.seg['parts'][p]['after']:,} faces now) with smaller quads, which usually clears folded or crossing faces. The old version stays in its tries"

    def execute(self, context):
        from . import ladder, segments

        p = self._part()
        if p is None:
            self.report({"WARNING"}, "This spot is not on a part that can be redone")
            return {"CANCELLED"}
        ui["cur"] = self.index
        segments.select([p])
        changed, why = ladder.part_sizes(context, [p], bigger=False)
        if why:
            self.report({"WARNING"}, why[0])
        overlay.tag_redraw()
        return {"FINISHED"} if changed else {"CANCELLED"}


class CURIOMESH_OT_issue_keep_original(_SpotPartOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_keep_original"
    bl_label = "Keep Original Here"
    bl_description = "Puts your own faces back on the part under this spot. It asks first; Put Reduced Back undoes it"

    @classmethod
    def description(cls, context, properties):
        from . import segments

        p = spot_part(properties.index)
        if not 0 <= p < len(segments.seg["parts"]):
            return cls.bl_description
        return f"Puts your own faces back on {segments.name_of(p)}: about {segments.original_cost([p]):,} more faces. Put Reduced Back undoes it"

    def invoke(self, context, event):
        from . import segments

        p = self._part()
        if p is None:
            return {"CANCELLED"}
        return segments.ask(self, context, event, f"Your own faces go back on {segments.name_of(p)}: about {segments.original_cost([p]):,} more faces.")

    def execute(self, context):
        from . import segments

        p = self._part()
        if p is None or not segments.keep_original(context, [p]):
            self.report({"WARNING"}, "This part could not be put back")
            return {"CANCELLED"}
        ui["cur"] = self.index
        return {"FINISHED"}


def original_parts() -> list[int]:
    """Parts that show your own faces now (Keep Original), which Put Reduced Back can switch back."""
    from . import segments

    return [p for p, r in enumerate(segments.seg["parts"]) if r["method"] == segments.ORIGINAL]


class CURIOMESH_OT_issue_put_back(bpy.types.Operator):
    bl_idname = "curiomesh.issue_put_back"
    bl_label = "Put Reduced Back"
    bl_description = "Switches the parts that got your own faces back to the reduced faces they had before Keep Original"
    bl_options = {"INTERNAL"}

    part: bpy.props.IntProperty(name="Part", description="Which part to switch back; -1 for every part showing your own faces", default=-1)

    @classmethod
    def poll(cls, context):
        from . import segments

        segments.ensure()
        return segments.ready() and _g().progress_info() is None and bool(original_parts())

    def execute(self, context):
        from . import segments

        wanted = {}
        for p in original_parts() if self.part < 0 else [self.part]:
            tries = segments.seg["parts"][p]["tries"]
            index = next((i for i in range(len(tries) - 1, -1, -1) if tries[i]["method"] != segments.ORIGINAL), None)
            if index is not None:
                wanted[p] = index
        if not wanted or not segments.use_tries(context, wanted):
            return {"CANCELLED"}
        return {"FINISHED"}


def _record(context) -> dict:
    res = _g().state["res"]
    rec = dict(res["issues"])
    rec["meta"] = dict(rec.get("meta", {}), source=_g().state["source"], blend=os.path.basename(bpy.data.filepath))
    return rec


class CURIOMESH_OT_issue_save(_IssueOp, bpy.types.Operator, ExportHelper):
    bl_idname = "curiomesh.issue_save"
    bl_label = "Save Issue Report"
    bl_description = "Saves the list of bad spots to a file you pick, so a later run can be compared with it"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        name = f"curiomesh_issues_{bpy.path.clean_name(_g().state['source'] or 'mesh')}.json"  # an object name can hold : or /
        # The file browser always opens; it starts next to the .blend when there is one.
        self.filepath = os.path.join(os.path.dirname(bpy.data.filepath), name) if bpy.data.filepath else name
        return super().invoke(context, event)

    def execute(self, context):
        try:
            with open(self.filepath, "w", encoding="utf-8") as fh:
                json.dump(_record(context), fh, indent=1)
        except OSError as exc:
            self.report({"ERROR"}, f"Could not save the issue report: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Issue report saved: {self.filepath}")
        return {"FINISHED"}


class CURIOMESH_OT_issue_compare(_IssueOp, bpy.types.Operator, ImportHelper):
    bl_idname = "curiomesh.issue_compare"
    bl_label = "Compare With Report..."
    bl_description = "Loads an older report and shows which of its bad spots are fixed, still there, or new"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        try:
            with open(self.filepath, encoding="utf-8") as fh:
                old = json.load(fh)
            ui["diff"] = issues.compare(old, _record(context))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:  # not a report, or not one of ours
            self.report({"ERROR"}, f"Could not read that report: {exc}")
            return {"CANCELLED"}
        overlay.tag_redraw()
        return {"FINISHED"}


def draw_reports(layout) -> None:
    """Save the spot list and compare a later run with it (Advanced > Issue Reports), and the link to report a problem."""
    layout.operator("wm.url_open", text="Report a Problem (opens browser)", icon="ERROR").url = _g().problem_url()
    found = spots()
    res = _g().state["res"]
    if res is None or res.get("issues") is None:
        layout.label(text="Reports need a finished result", icon="INFO")
        return
    layout.label(text=f"{len(found)} spot{'s' if len(found) != 1 else ''} with bad faces" if found else "No bad spots found", icon="ERROR" if found else "CHECKMARK")
    for kind, count in sorted(res["issues"]["totals"].items(), key=lambda kv: -issues.WEIGHTS.get(kv[0], 0)):
        layout.label(text=f"{count:,} {KIND_WORDS.get(kind, kind)}", icon="BLANK1")
    row = layout.row(align=True)
    row.operator("curiomesh.issue_save", icon="FILE_TICK")
    row.operator("curiomesh.issue_compare", icon="FILE_REFRESH")
    diff = ui["diff"]
    if diff is not None:
        layout.label(text=f"Since that report: {len(diff['fixed'])} fixed, {len(diff['still'])} still there, {len(diff['new'])} new", icon="INFO")
        for kind, t in diff["totals"].items():
            layout.label(text=f"   {KIND_WORDS.get(kind, kind)}: {t['old']:,} before, {t['new']:,} now")


def draw_overlay(guided, preview, res, flat, batch_for_shader) -> None:
    """All spots in orange, the current one in yellow, when Show Bad Faces is on. Called from the segment view's draw."""
    found = res.get("issue_spots")
    if not found or ui["res"] is not res:
        return
    matrix = tuple(round(x, 5) for row in preview.matrix_world for x in row)
    if ui["batch_key"] is None:  # set to None elsewhere: built again
        ui["batch_key"] = time.perf_counter_ns()
    cur = ui["cur"]

    def arrays():  # in overlay.drawn_arrays' timer, not in the draw
        out = []
        tris = guided._fan(res["faces"]).reshape(-1, 3)
        owner = np.repeat(np.arange(len(res["faces"])), [len(f) - 2 for f in res["faces"]])
        for colour, picked in ((SPOT_COLOUR, [s for i, s in enumerate(found) if i != cur]), (CURRENT_COLOUR, [found[cur]] if 0 <= cur < len(found) else [])):
            mask = np.zeros(len(res["faces"]), bool)
            for s in picked:
                mask[s["faces"]] = True
            corners = tris[mask[owner]].reshape(-1)
            if len(corners):
                out.append((guided._world(preview.matrix_world, res["co"][corners]).astype(np.float32), np.tile(np.float32(colour), (len(corners), 1))))
        return out

    built = overlay.drawn_arrays("issues.spots", (id(res), matrix, cur, ui["batch_key"]), arrays)
    if built is None:
        return
    if ui.get("batch_of") is not built:
        ui["batch"] = [batch_for_shader(flat, "TRIS", {"pos": pos, "color": col}) for pos, col in built]
        ui["batch_of"] = built
    for batch in ui["batch"] or ():
        flat.bind()
        batch.draw(flat)


CLASSES = (
    CURIOMESH_OT_issue_view,
    CURIOMESH_OT_issue_select,
    CURIOMESH_OT_issue_step,
    CURIOMESH_OT_issue_redo,
    CURIOMESH_OT_issue_keep_original,
    CURIOMESH_OT_issue_put_back,
    CURIOMESH_OT_issue_save,
    CURIOMESH_OT_issue_compare,
)
