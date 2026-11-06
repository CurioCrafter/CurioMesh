# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The one place where slow work is handed off the caller's loop.

Every job uses `start` (a handle it polls), `run` (a generator that yields until the result is there), `each` (one
call per item) and `local` (a per-runner stopping flag). Nothing else in the add-on starts threads. This build runs
each task on a worker thread, so Blender stays usable; a build may swap this file for one that runs them inline.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor


class Task:
    """`work()` started now. `wait(seconds)` is True once it ended; `result()` returns its value or raises its error."""

    def __init__(self, work, name: str = "CurioMeshTask"):
        self._out: list = []
        self._failed: list = []
        self._thread = threading.Thread(target=self._go, args=(work,), daemon=True, name=name)
        self._thread.start()

    def _go(self, work) -> None:
        try:
            self._out.append(work())
        except BaseException as exc:  # raised again by `result` on the caller's side
            self._failed.append(exc)

    def wait(self, seconds: float | None = 0.0) -> bool:
        if seconds is None or seconds > 0:
            self._thread.join(seconds)
        return not self._thread.is_alive()

    def done(self) -> bool:
        return self.wait(0.0)

    def result(self):
        if self._failed:
            raise self._failed[0]
        return self._out[0] if self._out else None


def start(work, name: str = "CurioMeshTask") -> Task:
    return Task(work, name)


def run(work, name: str = "CurioMeshTask", stopped=lambda: False, tick=None):
    """Generator: `work()` as a task, yielding while it runs. Returns its result, or None once `stopped()`."""
    task = Task(work, name)
    while not task.wait(0.02):  # sleep, don't spin: a busy caller slows the worker (one interpreter lock)
        if stopped():
            return None
        if tick is not None:
            tick()
        yield
    return task.result()


def each(call, items, workers: int = 1):
    """`call(item)` for each item, in order, as they are ready: on up to `workers` threads at once."""
    if workers <= 1:
        yield from map(call, items)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="CurioMeshPair") as pool:
        yield from pool.map(call, items)


def local():
    """A namespace each thread sees its own copy of."""
    return threading.local()
