# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Long Apply work in short timer slices, so Blender never freezes (no threads, no depsgraph handlers).

An operator writes its body as a generator, `steps(context)`, that yields between small pieces of work and returns the
operator's result set. `SlicedOp` runs it: from a button (invoke) as a modal operator that does one slice per timer
event, with the job shown on the dashboard and the status bar, Esc asking it to stop; from a
script, F9 or background Blender to the end at once, as before. The operator keeps its UNDO flag, and Blender pushes
the one undo step when the modal ends, so one Ctrl+Z covers the result and its UVs.
"""

from __future__ import annotations

import time

SLICE_S = 0.05  # work per timer slice; a single step is a few ms, so no slice runs much over this
GAP_S = 0.01  # timer interval: Blender draws and reads input between slices
NAVIGATION = {
    "MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "MOUSEMOVE", "INBETWEEN_MOUSEMOVE", "TRACKPADPAN", "TRACKPADZOOM",
    "MOUSEROTATE", "MOUSESMARTZOOM", "NDOF_MOTION", "WINDOW_DEACTIVATE", "TIMER_REPORT",
}


class Job:
    """The sliced work running now, read by the dashboard (`guided.progress_info`, `overlay.job_info`) and Stop."""

    def __init__(self, stage: str):
        self.stage, self.fraction, self.started = stage, 0.0, time.perf_counter()
        self.stopped, self.busy, self.progress = False, True, (0, 0)
        self.longest = 0.0  # the longest slice so far, in seconds

    def stop(self) -> None:
        self.stopped = True

    def seconds_left(self):
        return None


JOB: Job | None = None


def stopped() -> bool:
    """True once Esc or Stop asked the running job to stop. The steps then fall back and end."""
    return JOB is not None and JOB.stopped


def tick(stage: str, fraction: float | None = None) -> None:
    """What the job is doing now, for the dashboard and the status bar."""
    if JOB is not None:
        JOB.stage = stage
        JOB.fraction = JOB.fraction if fraction is None else min(1.0, max(0.0, float(fraction)))


def finish(steps):
    """Run a step generator to the end and return its value (scripts, F9, background Blender: as before)."""
    while True:
        try:
            next(steps)
        except StopIteration as end:
            return end.value


def run_slice(steps, budget: float = SLICE_S):
    """One slice of a step generator: (True, its value) once it ended, else (False, None)."""
    end = time.perf_counter() + budget
    try:
        while time.perf_counter() < end:
            next(steps)
    except StopIteration as done:
        return True, done.value
    return False, None



class SlicedOp:
    """Mixin for an operator whose `steps(context)` generator does the work (see the module text)."""

    job_name = "Working"

    def invoke(self, context, event):
        self._invoked = True
        return self.execute(context)

    def execute(self, context):
        import bpy  # late: the cores' unit tests load this module without Blender

        steps = self.steps(context)
        if not getattr(self, "_invoked", False) or bpy.app.background or context.window is None or self.is_repeat():
            return finish(steps)
        global JOB
        JOB = Job(self.job_name)
        done, value = self._slice(context, steps)
        if done:
            return value
        self._steps = steps
        wm = context.window_manager
        self._timer = wm.event_timer_add(GAP_S, window=context.window)
        wm.modal_handler_add(self)
        from . import overlay

        overlay.ensure_ticker()
        return {"RUNNING_MODAL"}

    def _slice(self, context, steps):
        global JOB
        started = time.perf_counter()
        try:
            done, value = run_slice(steps)
        except BaseException:
            self._end(context)
            raise
        if JOB is not None:
            JOB.longest = max(JOB.longest, time.perf_counter() - started)
        if done:
            self._end(context)
        else:
            self._status(context)
        return done, value

    def _status(self, context) -> None:
        from .translations import tr

        if JOB is None or context.workspace is None:
            return
        words = [tr(JOB.stage)]
        if JOB.fraction:
            words.append(f"{100 * JOB.fraction:.0f}%")
        words.append(tr("Stopping...") if JOB.stopped else tr("Esc stops"))
        context.workspace.status_text_set("  ".join(words))
        from . import overlay

        overlay.tag_redraw()

    def _end(self, context) -> None:
        global JOB
        JOB = None
        timer = getattr(self, "_timer", None)
        if timer is not None:
            context.window_manager.event_timer_remove(timer)
            self._timer = None
        if context.workspace is not None:
            context.workspace.status_text_set(None)
        from . import overlay

        overlay.tag_redraw()

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            if JOB is not None:
                JOB.stop()
            return {"RUNNING_MODAL"}
        if event.type == "TIMER":  # ours or another timer: either way one more slice
            done, value = self._slice(context, self._steps)
            return value if done else {"RUNNING_MODAL"}
        if event.type in NAVIGATION:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}  # nothing else reaches Blender until the result is written

    def cancel(self, context):
        """Blender ends the modal (window closed, file loaded): the steps' own clean-up runs."""
        self._steps.close()
        self._end(context)
