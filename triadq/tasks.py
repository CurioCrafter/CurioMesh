# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""One slow step inside a job's generator.

Jobs run as generators, one short slice per `bpy.app.timers` call on the main thread. `run` yields once, so the stage
name on screen is current before the step starts, then calls `work()` right there, on the main thread, and returns its
result. While it runs, the windows show the wait cursor. Nothing here starts a thread or a process.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def busy_cursor():
    """The wait cursor in every Blender window while a step that can't be cut into slices runs. Nothing headless."""
    try:
        import bpy

        windows = list(bpy.context.window_manager.windows)
    except (ImportError, AttributeError):  # no bpy (unit tests)
        windows = []
    for window in windows:
        window.cursor_modal_set("WAIT")
    try:
        yield
    finally:
        for window in windows:
            window.cursor_modal_restore()


def run(work, stopped=lambda: False, tick=None):
    """Generator: `work()` after one yield, under the wait cursor. Returns its result, or None once `stopped()`."""
    if tick is not None:
        tick()
    yield
    if stopped():
        return None
    with busy_cursor():
        return work()
