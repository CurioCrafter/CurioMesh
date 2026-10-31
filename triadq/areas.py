# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Work areas: parts of a mesh remeshed one at a time on a worker thread, each at its own density and fill style.

Nothing here touches Blender data, so the thread is safe. Finished pieces are cached by their faces and settings:
changing one area leaves the others alone, and going back to an earlier setting is instant.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import patch

CACHE_PIECES = 256


@dataclass
class Area:
    center: np.ndarray
    radius: float
    target: float  # edge length
    strategy: str = "FLOW"


@dataclass
class Job:
    mask: np.ndarray
    target: float
    strategy: str
    tag: Any = None


def group_areas(areas: list[Area]) -> list[list[int]]:
    """Overlapping areas with the same settings act as one (a painted stroke). Oldest group first."""
    parent = list(range(len(areas)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(areas):
        for j in range(i):
            b = areas[j]
            if a.target == b.target and a.strategy == b.strategy and np.linalg.norm(a.center - b.center) < a.radius + b.radius:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(areas)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[-1])


def area_masks(centroids: np.ndarray, areas: list[Area], groups: list[list[int]]) -> list[np.ndarray]:
    """Face mask of each group. Where groups overlap, the newest one owns the face."""
    owner = np.full(len(centroids), -1)
    for k, group in enumerate(groups):
        for i in group:
            owner[np.linalg.norm(centroids - areas[i].center, axis=1) < areas[i].radius] = k
    return [owner == k for k in range(len(groups))]


class AreaRunner:
    """Runs one plan of jobs at a time; a new plan stops the old one right away.

    `fit(result)` settles a finished piece on the surface; `merge(results)` builds what `output` shows;
    `finish(output)` runs once the whole plan is done.
    `output_id` changes each time a new `output` is ready, so the main thread can pick it up from a timer.
    """

    def __init__(self, vertices, faces, frozen=None, *, fit=None, merge=None, finish=None, max_faces: int = 40_000):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = faces
        self.frozen = np.zeros(len(faces), dtype=bool) if frozen is None else np.asarray(frozen, dtype=bool)
        self.fit, self.merge, self.finish, self.max_faces = fit, merge, finish, max_faces
        self.cache: dict[tuple, patch.PatchResult] = {}
        self.current: Any = None  # tag of the job being worked on
        self.progress = (0, 0)  # pieces done, pieces to do
        self.output: Any = None
        self.output_id = 0
        self.warnings: list[str] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._plan: list[Job] = []
        self._plan_id = 0
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._topology = None

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def start(self, jobs: list[Job]) -> None:
        with self._lock:
            self._plan = list(jobs)
            self._plan_id += 1
            self._idle.clear()
            self._wake.set()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="CurioMeshAreas")
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._plan_id += 1
            self._wake.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def _stale(self, plan_id: int) -> bool:
        return self._stopped or plan_id != self._plan_id

    def _loop(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                if self._stopped:
                    self._idle.set()
                    return
                self._wake.clear()
                plan, plan_id = self._plan, self._plan_id
            try:
                self._run(plan, plan_id)
            except Exception as exc:  # a thread has nobody above it to report to
                self.warnings = [f"Area failed: {exc}"]
            with self._lock:
                if plan_id == self._plan_id:
                    self.current = None
                    self._idle.set()

    def _publish(self, pieces: list[tuple], plan_id: int) -> None:
        results = [self.cache[key] for key, _job, _comp in pieces if key in self.cache and self.cache[key].ok]
        output = self.merge(results) if self.merge is not None else results
        if not self._stale(plan_id):
            self.output = output
            self.output_id += 1

    def _run(self, plan: list[Job], plan_id: int) -> None:
        if self._topology is None:
            self._topology = patch.region_topology(self.vertices, self.faces)
        taken = self.frozen.copy()
        pieces: list[tuple] = []
        warnings: list[str] = []
        for job in plan:  # quick: decide every piece first, so the areas already done stay on screen
            others = taken.copy()
            for other in plan:
                if other is not job:
                    others |= other.mask
            comps, found = patch.prepare_regions(self.vertices, self.faces, job.mask, min_island=4, max_hole=4, smooth=2, split_rings=True, frozen=others, topology=self._topology)
            warnings += found
            for comp in comps:
                taken[comp] = True
                pieces.append(((comp.tobytes(), job.target, job.strategy), job, comp))
            if self._stale(plan_id):
                return
        missing = [p for p in pieces if p[0] not in self.cache]
        self.progress = (0, len(missing))
        self.warnings = warnings
        self._publish(pieces, plan_id)
        for done, (key, job, comp) in enumerate(missing):
            if self._stale(plan_id):
                return
            self.current, self.progress = job.tag, (done, len(missing))
            result = patch.remesh_region(self.vertices, self.faces, comp, target_length=job.target, strategy=job.strategy, falloff=2.5 * job.target, max_faces=self.max_faces, cancelled=lambda: self._stale(plan_id))
            if self._stale(plan_id):  # a half-done piece from an old plan is worth nothing
                return
            if result.ok and self.fit is not None:
                self.fit(result)
            if not result.ok:
                warnings.append(f"Area ({len(comp)} faces) skipped: {result.message}")
            self.cache[key] = result
            while len(self.cache) > CACHE_PIECES:
                self.cache.pop(next(iter(self.cache)))
            self._publish(pieces, plan_id)
        self.progress = (len(missing), len(missing))
        if self.finish is not None:
            self.finish(self.output)
