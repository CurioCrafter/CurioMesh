# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""One slow step inside a job's generator.

Jobs run as generators, one short slice per `bpy.app.timers` call on the main thread. `run` yields once, so the stage
name on screen is current before the step starts, then calls `work()` right there, on the main thread, and returns its
result. Nothing here starts a thread or a process.
"""

from __future__ import annotations


def run(work, stopped=lambda: False, tick=None):
    """Generator: `work()` after one yield. Returns its result, or None once `stopped()`."""
    if tick is not None:
        tick()
    yield
    if stopped():
        return None
    return work()
