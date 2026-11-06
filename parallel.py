# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Measure the parts of one Auto Segments job in several background Blenders at once.

Python holds one core however many threads it has, so the job hands parts to helper processes (`parts_worker.py`):
each loads the job once, then measures the parts it is given with the very same `AutoJob._measure_piece` and writes
what it found to a file. The job takes the results back in part order, so it ends up exactly where the serial path
would. `Pool` starts, feeds and kills the helpers; `measure_in_worker` and `apply_result` are the two ends of one part.

Files go in a private temporary folder that only this job and its own helpers write, and they are pickles: the sizes
must come back exactly as they were made (NumPy types, face lists), and nothing outside that folder is ever read.
No `bpy`, so the tests run outside Blender.
"""

from __future__ import annotations

import atexit
import os
import pickle
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path


PACKAGE = __name__.rpartition(".")[0]  # the add-on's package: an extension's is longer in the user's Blender than in a helper


def _write(path: Path, value) -> None:
    """Pickle `value` to `path` so a reader never sees half a file. The writer's package name goes first."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(PACKAGE, handle)
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


class _Reader(pickle.Unpickler):
    """Only NumPy, plain Python values and the add-on's own types, with the writer's package name turned into ours."""

    def __init__(self, handle, theirs: str):
        super().__init__(handle)
        self.theirs = theirs

    def find_class(self, module, name):
        if self.theirs and (module == self.theirs or module.startswith(self.theirs + ".")):
            module = (PACKAGE + module[len(self.theirs) :]).lstrip(".")
        elif not (module.split(".")[0] in ("numpy", "builtins", "collections", "copyreg", "_codecs") or (not self.theirs and not PACKAGE)):
            raise pickle.UnpicklingError(f"{module}.{name} is not something a helper sends")
        return super().find_class(module, name)


def _read(path: Path):
    with open(path, "rb") as handle:
        theirs = _Reader(handle, "").load()
        if not isinstance(theirs, str):
            raise pickle.UnpicklingError("not a helper's file")
        return _Reader(handle, theirs).load()


class Pool:
    """Helper processes that do tasks (measure a part, reduce a part to a size). Hand them out with `give`, call `poll`
    often, take results with `take`. A task is any key the job picks, such as a part number.

    `command` starts one helper; the folder is added as its last argument. A helper reads tasks from its standard
    input, one a line, and quits when that closes, so a helper never outlives the Blender that started it.
    """

    def __init__(self, command: list[str], count: int, setup: dict, env: dict | None = None):
        self.folder = Path(tempfile.mkdtemp(prefix="curiomesh_parts_"))
        self.workers: list[subprocess.Popen] = []
        self.busy: dict[int, object] = {}  # helper -> the task it is doing
        self.handed: dict = {}  # task -> the helper doing it
        self.ready: dict = {}  # task -> its result, back from a helper but not yet taken
        self.failed: set = set()  # tasks a helper didn't finish: the job does those itself
        self._names: dict = {}  # task -> the name of its result file
        self.peak_mb: dict[int, float] = {}  # helper -> the most memory it reported using
        self.lost = 0  # helpers that stopped by themselves
        atexit.register(self.close)  # Blender quit in the middle
        _write(self.folder / "job.pkl", setup)
        for _ in range(count):
            log = open(self.folder / f"log_{len(self.workers)}.txt", "wb")
            try:
                self.workers.append(subprocess.Popen(command + [str(self.folder)], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
            finally:
                log.close()  # the helper has its own handle now

    def idle(self) -> list[int]:
        return [w for w, proc in enumerate(self.workers) if proc is not None and w not in self.busy]

    def alive(self) -> int:
        return sum(proc is not None for proc in self.workers)

    def give(self, task, words: str) -> bool:
        """Hand `task`, told to the helper as `words`, to an idle helper. False if none is idle."""
        name = self._names.setdefault(task, f"t{len(self._names)}")
        for worker in self.idle():
            try:
                self.workers[worker].stdin.write(f"{name} {words}\n".encode())
                self.workers[worker].stdin.flush()
            except OSError:
                self._lose(worker)
                continue
            self.busy[worker] = task
            self.handed[task] = worker
            return True
        return False

    def poll(self) -> None:
        """Collect finished tasks and notice helpers that died; their tasks go to `failed`."""
        for worker, part in list(self.busy.items()):
            path = self.folder / f"{self._names[part]}.pkl"
            if path.exists():
                try:
                    result = _read(path)
                except PermissionError:  # Windows: a virus scanner can hold a new file for a moment
                    continue
                try:
                    path.unlink()
                except OSError:
                    pass  # read already; the folder goes at the end
                del self.busy[worker]
                self.peak_mb[worker] = max(self.peak_mb.get(worker, 0.0), float(result.get("peak_mb", 0.0)))
                if "error" in result:
                    self.failed.add(part)
                    self.handed.pop(part, None)
                else:
                    self.ready[part] = result
            elif self.workers[worker].poll() is not None:
                self._lose(worker)

    def _lose(self, worker: int) -> None:
        part = self.busy.pop(worker, None)
        if part is not None:
            self.failed.add(part)
            self.handed.pop(part, None)
        proc = self.workers[worker]
        if proc is not None:
            _kill(proc)
            self.workers[worker] = None
            self.lost += 1

    def waiting_for(self, part) -> bool:
        """True while task `part` is with a helper or back and not yet taken."""
        return part in self.handed and part not in self.failed

    def take(self, part) -> dict | None:
        """The result of task `part` once it is back, else None (still busy, or failed: see `failed`)."""
        result = self.ready.pop(part, None)
        if result is not None:
            self.handed.pop(part, None)
        return result

    def close(self) -> None:
        """Kill every helper and remove the folder. Safe to call twice."""
        for worker, proc in enumerate(self.workers):
            if proc is not None:
                _kill(proc)
                self.workers[worker] = None
        self.busy.clear()
        atexit.unregister(self.close)
        shutil.rmtree(self.folder, ignore_errors=True)


def _kill(proc: subprocess.Popen) -> None:
    try:
        proc.stdin.close()
    except OSError:
        pass
    proc.kill()
    try:
        proc.wait(5.0)
    except subprocess.TimeoutExpired:
        pass


# --------------------------------------------------------------------------- one part, both ends


def setup_of(job) -> dict:
    """What a helper needs to rebuild this job: the mesh, the settings, and the parts as `_split_pieces` cut them."""
    import numpy as np

    from .triadq.types import FlatFaces

    return {
        "co": job.co, "faces": FlatFaces.of(job.faces), "materials": np.asarray([int(m) for m in job.materials], dtype=np.int32),
        "detail": job.detail, "target_faces": job.target_faces, "mirror": job.mirror, "drop": job.drop, "method": job.method, "output": job.output_kind,
        "methods": dict(job.methods), "style": job.style, "settings": job.cfg, "rough": job.rough, "baked": job.baked, "hard": job.hard,
        "pieces": job.pieces, "locked": job.locked, "chunks": job.chunks, "names": job.names, "picture_sizes": job.picture_sizes,
    }


def worker_job(job_class, context, setup: dict):
    """The job a helper measures parts of: built like the one in the user's Blender, already cut into its parts."""
    job = job_class(context, setup["co"], setup["faces"], setup["materials"], setup["detail"], setup["target_faces"], mirror=setup["mirror"], drop=setup["drop"], method=setup["method"], output=setup["output"], methods=setup["methods"], style=setup["style"], settings=setup["settings"], rough=setup["rough"], baked=setup["baked"], hard=setup.get("hard"))
    for name in ("pieces", "locked", "chunks", "names", "picture_sizes"):
        setattr(job, name, setup[name])
    job.spot_copies = False  # the job that sent the part already knows it is no copy
    return job


def measure_in_worker(job, part: int, field_spent: float, peak_mb=lambda: 0.0) -> dict:
    """Measure one part in a helper's job. Everything the part changed in the job comes back, to be applied in order."""
    captured: dict = {}

    def finish(key, sizes, levels, ready, seen, file, fresh):
        captured.update(sizes=sizes, levels=levels, ready=ready, seen=seen)

    job._finish_piece = finish
    job.sizes = [None] * part  # the part's number is how many came before it: its method and its case go by it
    job.field_spent = field_spent
    looks, spent, notes, timings = job.looks, job.field_spent, len(job.notes), dict(job.timings)
    try:
        for _ in job._measure_piece(job.pieces[part], job.picture_sizes[part]):
            pass
        if not captured:
            raise RuntimeError("the part was not finished")
    except Exception:
        return {"error": traceback.format_exc(), "peak_mb": peak_mb()}
    captured.update(
        looks=job.looks - looks, field=job.field_spent - spent, notes=job.notes[notes:], case=job.part_cases.pop(part, None), peak_mb=peak_mb(),
        timings={k: v - timings.get(k, 0.0) for k, v in job.timings.items() if v != timings.get(k, 0.0)},
    )
    del job.notes[notes:]
    return captured


def reduce_in_worker(job, part: int, ratio: float, peak_mb=lambda: 0.0) -> dict:
    """One size of a part, made in a helper exactly as `AutoJob._reduced` makes it in the job."""
    try:
        ids = job.pieces[part]
        points, polys, kinds = job._piece_arrays(ids)
        work = job._reduced(points, polys, kinds, ratio, job._piece_drop(ids), job._piece_lock(ids))
        while True:
            try:
                next(work)
            except StopIteration as done:
                return {"size": done.value, "peak_mb": peak_mb()}
    except Exception:
        return {"error": traceback.format_exc(), "peak_mb": peak_mb()}


def apply_result(job, part: int, result: dict) -> None:
    """What measuring `part` in a helper changed, done to this job as if it had measured the part itself."""
    job.looks += result["looks"]
    job.field_spent += result["field"]
    job.notes += result["notes"]
    for name, seconds in result["timings"].items():
        job.timings[name] = job.timings.get(name, 0.0) + seconds
    if result["case"] is not None:
        job.part_cases[part] = result["case"]


def worker_main(argv: list[str]) -> None:
    """The loop of a helper process (see parts_worker.py): rebuild the job, then measure each part it is handed."""
    import importlib
    import queue
    import sys
    import threading

    import bpy

    folder = Path(argv[argv.index("--") + 1])
    autoretopo = importlib.import_module(PACKAGE + ".autoretopo")
    hardware = importlib.import_module(PACKAGE + ".hardware")
    job = worker_job(autoretopo.AutoJob, bpy.context, _read(folder / "job.pkl"))
    tasks: queue.Queue = queue.Queue()

    def read():
        for line in sys.stdin:
            tasks.put(line)
        os._exit(0)  # the job stopped, or the Blender that started this one is gone

    threading.Thread(target=read, daemon=True).start()
    while True:
        name, kind, part, number = tasks.get().split()
        started = time.perf_counter()
        if kind == "measure":
            result = measure_in_worker(job, int(part), float(number), hardware.peak_memory_mb)
        else:
            result = reduce_in_worker(job, int(part), float(number), hardware.peak_memory_mb)
        print(f"{kind} {part} {number}: {time.perf_counter() - started:.1f} s", "failed" if "error" in result else "", flush=True)
        if "error" in result:
            print(result["error"], flush=True)
        _write(folder / f"{name}.pkl", result)


def find_may_start(runner) -> bool:
    """Finding the segments may start when nothing runs or only an older find does (that one is stopped first).
    It never takes over a running search, redo or Use remake: they would be stopped for a job that only adds labels."""
    return runner is None or not runner.busy or getattr(runner, "is_find", False)


class Debounce:
    """The newest request wins. `poke` asks for a run; `due` says when the asking has been quiet for `wait` seconds, or
    has gone on for `most` seconds (a long drag still shows something); `take` starts one and hands it a ticket;
    `stale(ticket)` tells a running one that a newer request came in."""

    def __init__(self, wait: float, most: float | None = None):
        self.wait = wait
        self.most = most
        self.asked = None  # time of the last request not yet taken
        self.first = None  # time of the first request not yet taken
        self.serial = 0

    def poke(self, now: float) -> None:
        if self.asked is None:
            self.first = now
        self.asked = now
        self.serial += 1

    def due(self, now: float) -> bool:
        if self.asked is None:
            return False
        return now - self.asked >= self.wait or (self.most is not None and now - self.first >= self.most)

    def take(self) -> int:
        self.asked = self.first = None
        return self.serial

    def stale(self, ticket: int) -> bool:
        return ticket != self.serial
