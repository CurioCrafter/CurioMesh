# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The one place where slow work is handed off the caller's loop.

Every job uses `start` (a handle it polls), `run` (a generator that yields until the result is there), `each` (one
call per item) and `local` (a per-runner stopping flag). This build starts no threads: each task runs to its end on
the caller's (main) thread, between the timer slices of the job that asked for it. Very big work goes to helper
Blender processes instead (see parallel.py).
"""

from __future__ import annotations

from types import SimpleNamespace


class Task:
    """`work()`, run now. `wait(seconds)` is True once it ended; `result()` returns its value or raises its error."""

    def __init__(self, work, name: str = "CurioMeshTask"):
        self._out = self._failed = None
        try:
            self._out = work()
        except Exception as exc:  # raised again by `result` on the caller's side
            self._failed = exc

    def wait(self, seconds: float | None = 0.0) -> bool:
        return True

    def done(self) -> bool:
        return True

    def result(self):
        if self._failed is not None:
            raise self._failed
        return self._out


def start(work, name: str = "CurioMeshTask") -> Task:
    return Task(work, name)


def run(work, name: str = "CurioMeshTask", stopped=lambda: False, tick=None):
    """Generator: `work()` after one yield (so the stage on screen is current). Returns its result, or None once `stopped()`."""
    if tick is not None:
        tick()
    yield
    if stopped():
        return None
    return Task(work, name).result()


def each(call, items, workers: int = 1):
    """`call(item)` for each item, in order."""
    yield from map(call, items)


def local():
    """A namespace for one runner's stopping flag."""
    return SimpleNamespace()
