# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Batch remesh: run the normal pipeline on every selected mesh and write one report."""

from __future__ import annotations

from dataclasses import asdict
import datetime
import json
from pathlib import Path
import tempfile
import time

import bpy

from . import overlay
from .bridge import run_curiomesh
from .operators import apply_result_to_settings, config_from_owner


class CURIOMESH_PG_batch_item(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Object")
    output: bpy.props.StringProperty(name="Output")
    faces: bpy.props.IntProperty(name="Faces")
    quad_ratio: bpy.props.FloatProperty(name="Quad Ratio", precision=3, subtype="FACTOR")
    tris: bpy.props.IntProperty(name="Triangles")
    poles: bpy.props.IntProperty(name="Poles")
    ms: bpy.props.FloatProperty(name="Time ms", precision=0)
    ok: bpy.props.BoolProperty(name="Success")
    error: bpy.props.StringProperty(name="Error")


def batch_targets(context) -> list[str]:
    return [
        o.name
        for o in context.selected_objects
        if o.type == "MESH" and o.name != overlay.PREVIEW_NAME
    ]


def report_row(name: str, result, elapsed_ms: float, error: str = "") -> dict:
    """One report row; `result` is a RemeshResult or None when the object could not be run."""
    ok = bool(result is not None and result.success and not error)
    row = {
        "object": name,
        "output": "",
        "faces": 0,
        "quad_ratio": 0.0,
        "tris": 0,
        "poles": 0,
        "time_ms": round(elapsed_ms, 1),
        "success": ok,
        "error": error or ("" if ok else (result.message if result is not None else "not run")),
    }
    if result is not None:
        d = result.diagnostics
        row.update(
            output=result.output_object.name if result.output_object is not None else "",
            faces=int(d.faces),
            quad_ratio=round(float(d.quad_ratio), 4),
            tris=int(d.tris),
            poles=int(d.extraordinary),
        )
    return row


def report_path() -> Path:
    base = Path(bpy.data.filepath).parent if bpy.data.filepath else Path(tempfile.gettempdir())
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"curiomesh_batch_{stamp}.json"


class BatchRunner:
    def __init__(self, context):
        self.settings = context.scene.curiomesh_settings
        self.config = config_from_owner(self.settings)
        self.names = batch_targets(context)
        self.rows: list[dict] = []
        self.index = 0
        self.started = time.perf_counter()
        context.window_manager.progress_begin(0, max(1, len(self.names)))
        self.settings.batch_results.clear()

    @property
    def done(self) -> bool:
        return self.index >= len(self.names)

    def step(self, context) -> None:
        name = self.names[self.index]
        self.settings.batch_status = f"Batch {self.index + 1}/{len(self.names)}: {name}"
        started = time.perf_counter()
        result, error = None, ""
        try:
            obj = bpy.data.objects.get(name)
            if obj is None:
                error = "object no longer exists"
            else:
                result = run_curiomesh(obj, self.config)
                if result.success:
                    apply_result_to_settings(self.settings, result)
        except Exception as exc:  # one bad mesh must not stop the batch
            error = f"{type(exc).__name__}: {exc}"
        self.rows.append(report_row(name, result, (time.perf_counter() - started) * 1000.0, error))
        self.index += 1
        context.window_manager.progress_update(self.index)
        overlay.tag_redraw()

    def finish(self, context) -> Path | None:
        context.window_manager.progress_end()
        ok = sum(1 for r in self.rows if r["success"])
        self.settings.batch_status = f"Batch done: {ok} ok, {len(self.rows) - ok} failed"
        for row in self.rows:
            item = self.settings.batch_results.add()
            item.name, item.output = row["object"], row["output"]
            item.faces, item.quad_ratio, item.tris = row["faces"], row["quad_ratio"], row["tris"]
            item.poles, item.ms, item.ok, item.error = row["poles"], row["time_ms"], row["success"], row["error"]
        path = report_path()
        payload = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "blend_file": bpy.data.filepath,
            "total_seconds": round(time.perf_counter() - self.started, 2),
            "config": asdict(self.config),
            "objects": self.rows,
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.settings.batch_report_path = str(path)
        except OSError:
            path = None
            self.settings.batch_report_path = ""
        overlay.tag_redraw()
        return path


class CURIOMESH_OT_batch_remesh(bpy.types.Operator):
    bl_idname = "curiomesh.batch_remesh"
    bl_label = "Batch Remesh Selected"
    bl_description = "Remesh every selected mesh with the current settings and save one JSON report"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(batch_targets(context))

    def _report_end(self, runner, path):
        ok = sum(1 for r in runner.rows if r["success"])
        level = {"INFO"} if ok == len(runner.rows) else {"WARNING"}
        self.report(level, f"Batch: {ok}/{len(runner.rows)} succeeded" + (f", report {path}" if path else ""))

    def execute(self, context):
        runner = BatchRunner(context)
        while not runner.done:
            runner.step(context)
        self._report_end(runner, runner.finish(context))
        return {"FINISHED"}

    def invoke(self, context, event):
        if bpy.app.background or context.window is None:
            return self.execute(context)
        self._runner = BatchRunner(context)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        cancel = event.type == "ESC"
        if event.type == "TIMER" or cancel:
            if not cancel and not self._runner.done:
                self._runner.step(context)
            if cancel or self._runner.done:
                context.window_manager.event_timer_remove(self._timer)
                self._report_end(self._runner, self._runner.finish(context))
                return {"CANCELLED" if cancel else "FINISHED"}
        return {"RUNNING_MODAL"}


CLASSES = (CURIOMESH_PG_batch_item, CURIOMESH_OT_batch_remesh)
