# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""Bring Back: parts made outside CurioMesh come back into the result (bench/tools_design/AGENT_TOOLS.md section 4).

`partload` reads and checks the folder. A background job (`CheckJob`, a `segments.PartJob`, so it shares the part
redo's pictures, progress bar and Stop) matches each part's border, stitches it into a copy, and checks for holes, lost
shape, its camera view and the whole object's loss against Most Detail Loss. Every part that passes becomes a try
called "From Outside"; the others are refused with the reason and the rest still come in. Keeping them all is one call
to `segments.use_tries`: one splice, one Ctrl+Z step and one History entry. In a window the Look First preview
(`tidytools.PartSwap`) shows them first.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np

from . import partload, segments, tidytools
from .triadq import checks, issues
from .triadq import parts as P
from .triadq import segment as S

NAME = "From Outside"
METHOD = "OUTSIDE"


class NotReady(partload.PartError):
    agent_code = "NOT_READY"
    default_hint = "Run Auto (without Mirror or Even Rows) and wait for it to finish, then bring the parts back"


class NoResult(partload.PartError):
    agent_code = "NO_RESULT"
    default_hint = "Run Auto first (auto.start), then export and bring back parts"


class NeedsWindow(partload.PartError):
    agent_code = "NEEDS_WINDOW"
    default_hint = "Leave look_first off: the parts are then checked and kept straight away"


class ViewChanged(partload.PartError):
    agent_code = "VIEW_CHANGED"
    default_hint = "Raise Most Detail Loss to allow it, or follow part_<id>_source.obj more closely"


pending = {"job": None, "plan": None, "reply": None}


def _g():
    return segments._g()


def check_ready() -> None:
    """NO_RESULT or NOT_READY (with the reason) unless the result on screen allows part swaps."""
    guided = _g()
    segments.ensure()
    base = guided.state["auto_base"]
    if base is None or guided.state["res"] is None:
        raise NoResult("There is no Auto result on screen")
    job = base["job"]
    why = (
        "an imported job result has no parts to swap" if job is None
        else "Mirror is on: the result holds both sides of one solved half" if job.mirror is not None
        else "Even Rows has no part sizes to swap" if job.style == "ROWS"
        else "this result was made in cut pieces, which are one surface" if job.chunks
        else "the part table is not ready for the result on screen; wait for its check to finish" if not segments.ready()
        else ""
    )
    if why:
        raise NotReady(f"Parts cannot be brought back: {why}", details={"reason": why})


def _folder(path) -> Path:
    """The picked folder, or the parts folder above a picked `returned` folder."""
    folder = Path(bpy.path.abspath(str(path))) if path else Path()
    if not (folder / "parts.json").is_file() and (folder.parent / "parts.json").is_file():
        folder = folder.parent
    return folder


def _window_view():
    """(window, area, region) of a 3D view, or None (headless)."""
    if bpy.app.background:
        return None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            region = next((r for r in area.regions if r.type == "WINDOW"), None) if area.type == "VIEW_3D" else None
            if region is not None:
                return window, area, region
    return None


class CheckJob(segments.PartJob):
    """Checks every returned part on the part redo's timer: border, holes, shape, view, Most Detail Loss, bad spots.
    It changes nothing; `finish` turns the parts that passed into tries."""

    def __init__(self, context, folder: Path, manifest: dict, look: bool):
        super().__init__(context, [], METHOD)
        self.folder, self.manifest, self.look = folder, manifest, look
        self.base_id = self.base["id"]
        gs = getattr(context.scene, "curiomesh_guided", None)
        self.cap = gs.organize_cap if gs is not None and gs.organize_cap_use else None
        self.made: dict[int, dict] = {}  # part -> the try it becomes
        self.refused: dict[int, dict] = {}  # part -> its error
        self.whole = None  # (loss before, loss with every passing part) when Most Detail Loss is on
        self.step_name = NAME

    def _one(self, p: int, co, faces, mat, piece, size, accepted):
        """Generator: the build of part `p` if it passes every check, else it raises the refusal."""
        job = self.job
        record = segments.seg["parts"][p]
        current = record["tries"][record["cur"]]
        ids = np.flatnonzero(piece == p)
        try:
            border = S.seam_loops(co, faces, ids)
        except ValueError:
            raise NotReady(f"{segments.name_of(p)} touches itself at a vertex, so its border cannot be matched", details={"part": p}) from None
        got = partload.returned(self.folder, p, int(mat[ids[0]]) if len(ids) else 0)
        if got is None:
            raise partload.PartError(f"returned/part_{p}.obj is not there", "Write the new faces for this part to returned/part_<id>.obj", {"part": p})
        points, new_faces = partload.match_border(got[0], got[1], border, size, part=p)
        build = (points, new_faces, np.asarray(got[2], dtype=np.int32))
        spliced = segments.splice(co, faces, mat, piece, {**accepted, p: build})
        partload.check_no_hole(faces, spliced[1], p)
        src_points, src_polys, _kinds = job._piece_arrays(record["source"] if "source" in record else job.pieces[p])  # a found segment is not one of the job's pieces
        partload.check_shape(src_points, src_polys, points, new_faces, p)
        change = yield from self._change(p, points, new_faces)
        if current["change"] is not None:
            had = current["change"]
        else:
            if current["build"] is None:
                current["build"] = P.cut(co, faces, mat, piece, p)
            had = yield from self._change(p, current["build"][0], current["build"][1])
        if not S.view_kept(had, change, segments.segment_goals.WORSE_TIMES, segments.segment_goals.WORSE_BY):
            raise ViewChanged(f"{segments.name_of(p)}: {change:.1f}% of its view changes, it had {had:.1f}%", details={"part": p, "view_change": change, "had": had})
        if self.cap is not None:
            if self.whole is None:
                self.whole = [(yield from self._whole_loss(co, faces)), None]
            loss = yield from self._whole_loss(spliced[0], spliced[1])
            if loss > self.whole[0] + self.cap:
                raise ViewChanged(
                    f"{segments.name_of(p)}: the whole object would change {loss:.1f}%, Most Detail Loss allows {self.whole[0] + self.cap:.1f}%",
                    details={"part": p, "whole_loss": loss, "allowed": self.whole[0] + self.cap, "most_detail_loss": self.cap},
                )
            self.whole[1] = loss
        return build, change

    def _run(self):
        out = self.base["out"]
        co, faces, mat, piece = np.asarray(out["co"], dtype=np.float64), list(out["faces"]), np.asarray(out["mat"]), np.asarray(segments.seg["piece"])
        size = float(np.ptp(self.job.co, axis=0).max()) or 1.0
        entries = [e for e in self.manifest["parts"]]
        accepted, changes = {}, {}
        for n, entry in enumerate(entries):
            p = entry["id"]
            self.progress = (n, len(entries))
            self.stage = f"Checking {segments.name_of(p)} from outside ({n + 1} of {len(entries)})"
            self.fraction = 0.85 * n / max(len(entries), 1)
            yield
            if self._stopped:
                return
            try:
                accepted[p], changes[p] = yield from self._one(p, co, faces, mat, piece, size, accepted)
            except partload.PartError as exc:
                self.refused[p] = partload.error_reply(exc)
        self.progress = (len(entries), len(entries))
        if accepted:
            self.stage = "Looking for problem spots in the new faces"
            self.fraction = 0.9
            new_co, new_faces, _mat, new_piece = segments.splice(co, faces, mat, piece, accepted)
            found = yield from segments._threaded(lambda: checks.defects(np.asarray(new_co), new_faces, self.job.co, self.job.faces))
            spots, _summary = issues.analyse(np.asarray(new_co), new_faces, {k: getattr(found, k) for k in checks.Defects.KINDS}, new_piece, {"source_faces": len(self.job.faces)})
            for p, build in accepted.items():
                tidy = yield from segments._threaded(lambda b=build: S.topology_score(b[0], b[1]))
                self.made[p] = {"method": METHOD, "detail": None, "after": len(build[1]), "change": changes[p], "topology": tidy, "issues": sum(1 for s in spots if int(s.get("part", -1)) == p), "build": build}
        self.fraction = 1.0
        self.stage = f"{len(self.made)} part(s) passed the checks, {len(self.refused)} refused"


def start(context, folder, look: bool) -> CheckJob:
    """Read and check the folder, stop running work, and start the check job. Raises the coded refusals."""
    guided = _g()
    check_ready()
    folder = _folder(folder)
    manifest = partload.load_manifest(folder)
    base, piece = guided.state["auto_base"], np.asarray(segments.seg["piece"])
    co, faces = np.asarray(base["out"]["co"], dtype=np.float64), base["out"]["faces"]
    count = len(segments.seg["parts"])
    border = []
    for entry in manifest["parts"]:
        if 0 <= entry["id"] < count:
            try:
                border += S.seam_loops(co, faces, np.flatnonzero(piece == entry["id"]))
            except ValueError:  # refused per part by the check job
                pass
    partload.check_result(manifest, len(faces), np.vstack(border) if border else np.zeros((0, 3)), range(count))
    guided.stop_work()
    job = CheckJob(context, folder, manifest, look)
    guided.state["work"] = {"key": base.get("key", ("", "AUTO")), "runner": job, "cent": None, "seen": 0, "areas": []}
    guided.state["message"] = job.status()
    pending.update(job=job, plan=None, reply=None)
    if not bpy.app.background:
        if not bpy.app.timers.is_registered(guided._auto_slice):
            bpy.app.timers.register(guided._auto_slice, first_interval=0.0)
        if not bpy.app.timers.is_registered(_watch):
            bpy.app.timers.register(_watch, first_interval=0.2)
    return job


def _reply(job: CheckJob, wanted: dict) -> dict:
    parts = segments.seg["parts"]
    kept = []
    for p, index in wanted.items():
        t, r = parts[p]["tries"][index], parts[p]
        old = r["tries"][r["cur"]]
        kept.append({"part": p, "name": segments.name_of(p), "faces_before": old["after"], "faces": t["after"], "view_change_before": old["change"], "view_change": t["change"], "tidiness": t["topology"], "bad_spots": t["issues"], "try": index})
    refused = [{"part": p, "name": segments.name_of(p) if p < len(parts) else f"Part {p + 1}", "error": e} for p, e in job.refused.items()]
    reply = {"folder": str(job.folder), "kept": kept, "refused": refused}
    if job.whole is not None:
        reply["whole_view_change"] = {"before": job.whole[0], "after": job.whole[1], "most_detail_loss": job.cap}
    return reply


def finish(context, job: CheckJob) -> dict:
    """The check is done: the passing parts become tries, then Look First opens (in a window, when asked) or they are
    kept at once. Returns the reply."""
    guided = _g()
    pending["job"] = None
    base = guided.state["auto_base"]
    if job._stopped:
        raise NotReady("The check was stopped before it finished; nothing changed", "Bring the parts back again")
    if job.warnings:  # the check itself failed: a bug, reported as INTERNAL
        raise RuntimeError(job.warnings[0])
    if base is not job.base or base["id"] != job.base_id:
        raise partload.MismatchError("The result changed while the parts were checked; nothing changed", "Bring the parts back again")
    wanted = {}
    for p, made in job.made.items():
        segments.seg["parts"][p]["tries"].append(made)
        wanted[p] = len(segments.seg["parts"][p]["tries"]) - 1
    reply = _reply(job, wanted)
    why = "; ".join(r["error"]["message"] for r in reply["refused"])
    pending.update(plan={"name": NAME, "tries": wanted, "todo": [], "why": why}, reply=reply)
    view = _window_view() if job.look else None
    if wanted and view is not None:
        window, area, region = view
        with context.temp_override(window=window, area=area, region=region, screen=window.screen):
            bpy.ops.curiomesh.parts_import("INVOKE_DEFAULT", checked=True)
        reply["look_first"] = True
    elif wanted:
        reply["kept_now"] = segments.use_tries(context, wanted, NAME)
    guided.state["message"] = f"{NAME}: {len(wanted)} part(s) in" + (f", {len(reply['refused'])} refused ({why})" if reply["refused"] else "")
    return reply


def _watch():
    """Timer: when the check job ends, finish it."""
    job = pending["job"]
    if job is None:
        return None
    if job.busy:
        return 0.2
    try:
        finish(bpy.context, job)
    except partload.PartError as exc:
        _g().state["message"] = f"{exc} ({exc.hint})"
    except Exception as exc:  # a timer has nobody above it to report to
        _g().state["message"] = f"Bringing the parts back failed: {exc}"
    return None


class CURIOMESH_OT_parts_import(tidytools.PartSwap, bpy.types.Operator):
    bl_idname = "curiomesh.parts_import"
    bl_label = "Bring Back"
    bl_description = "Pick a part folder an agent or another program filled in. Its parts are stitched in at their borders, checked with the camera within Most Detail Loss and shown before anything is kept. Ctrl+Z takes them all back out"
    bl_options = {"INTERNAL"}

    directory: bpy.props.StringProperty(name="Folder", description="The part folder (the one with parts.json)", subtype="DIR_PATH")
    filter_folder: bpy.props.BoolProperty(default=True, options={"HIDDEN", "SKIP_SAVE"})
    look_first: bpy.props.BoolProperty(name="Look First", description="Show the returned parts before they are kept. Off: they are kept as soon as they pass the checks", default=True)
    checked: bpy.props.BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        try:
            check_ready()
        except partload.PartError as exc:
            cls.poll_message_set(str(exc))
            return False
        if _g().progress_info() is not None:
            cls.poll_message_set("Wait for the work running now, or stop it")
            return False
        return True

    def plan(self, context) -> dict:
        return pending["plan"] or {"name": NAME, "tries": {}, "todo": [], "why": "Nothing was checked yet"}

    def keep(self, context, plan: dict) -> bool:
        return bool(plan["tries"]) and segments.use_tries(context, plan["tries"], NAME)

    def invoke(self, context, event):
        if self.checked:
            return super().invoke(context, event)
        if not self.directory:
            context.window_manager.fileselect_add(self)
            return {"RUNNING_MODAL"}
        return self.execute(context)

    def execute(self, context):
        if self.checked:
            return super().execute(context)
        try:
            start(context, self.directory, self.look_first)
        except partload.PartError as exc:
            self.report({"WARNING"}, f"{exc}. {exc.hint}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Checking the returned parts")
        return {"FINISHED"}


# --------------------------------------------------------------------------- agent commands


def _object(context, name) -> None:
    source = _g().state_source()
    if name and (source is None or source.name != name):
        raise NotReady(f"The result on screen is not for {name!r}", "Select that object and run Auto on it first", {"object": name, "on_screen": source.name if source else None})


def cmd_import(context, folder, look_first=False, object=None):  # noqa: A002 - every command takes `object`
    _object(context, object)
    if look_first and _window_view() is None:
        raise NeedsWindow("Look First needs a live window to show the parts in")
    job = start(context, folder, look_first)
    if bpy.app.background:
        job.wait()  # headless: run the check to the end here
        return finish(context, job)
    return {"running": True, "folder": str(job.folder), "parts": [e["id"] for e in job.manifest["parts"]], "hint": "Call parts.import_result when job.status says it is done"}


def cmd_result(context, object=None):  # noqa: A002
    job = pending["job"]
    if job is not None:
        return {"running": job.busy, "words": job.status(), "percent": round(100.0 * job.fraction, 1)}
    if pending["reply"] is None:
        raise partload.PartError("No parts were brought back yet", "Call parts.import first", {})
    return pending["reply"]


_FOLDER = {"type": "string", "description": "The part folder (the one holding parts.json and returned/)"}
_OBJECT = {"type": "string", "description": "Object name; the result on screen must be for it"}
COMMANDS = {
    "parts.import": {
        "fn": cmd_import, "edits": True,
        "help": "Bring returned parts back: border matched, stitched, checked with the camera within Most Detail Loss, kept as one History step. Parts that fail are refused one by one and the rest still come in.",
        "args": {"type": "object", "required": ["folder"], "properties": {"folder": _FOLDER, "look_first": {"type": "boolean", "description": "Show them in Look First in the window before they are kept"}, "object": _OBJECT}},
    },
    "parts.import_result": {
        "fn": cmd_result, "edits": False,
        "help": "The reply of the last parts.import (kept and refused parts), or its progress while it is still checking.",
        "args": {"type": "object", "properties": {"object": _OBJECT}},
    },
}

CLASSES = (CURIOMESH_OT_parts_import,)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy.app.timers.is_registered(_watch):
        bpy.app.timers.unregister(_watch)
    pending.update(job=None, plan=None, reply=None)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
