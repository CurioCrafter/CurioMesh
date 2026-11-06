# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Measure the parts of one Auto Segments job in several background Blenders at once.

The add-on's own Python runs on Blender's main thread only, so a big job can hand parts to helper processes
(`helper_command`), but only once the user has turned on Background Blender Processes in the preferences (`allowed`):
each loads the job once, then measures the parts it is given with the very same `AutoJob._measure_piece` and writes
what it found to a file. The job takes the results back in part order, so it ends up exactly where the serial path
would. `Pool` starts, feeds and kills the helpers; `measure_in_worker` and `apply_result` are the two ends of one part.

Files go in a private temporary folder that only this job and its own helpers write. Each is one NumPy .npz (read
with allow_pickle=False): the arrays as entries, everything else as JSON with tags that bring back tuples, sets,
NumPy scalars and a short list of the add-on's own types exactly as they were sent, so the sizes come back as made.
No `bpy`, so the tests run outside Blender.
"""

from __future__ import annotations

import atexit
import dataclasses
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


PACKAGE = __name__.rpartition(".")[0]  # the add-on's package: an extension's is longer in the user's Blender than in a helper
# The add-on's own types a helper file may hold: name -> module, relative to the package. Nothing else is rebuilt.
TYPES = {"Settings": "hardware", "Level": "triadq.budget", "FlatFaces": "triadq.types"}


def _type(name: str):
    if name not in TYPES:
        raise ValueError(f"{name} is not something a helper sends")
    return getattr(importlib.import_module((PACKAGE + "." + TYPES[name]).lstrip(".")), name)


def _encode(value, arrays: list):
    """A JSON-able tree for `value`; its arrays are appended to `arrays`. Raises TypeError for anything else."""
    if value is None or isinstance(value, (bool, int, float, str)) and not isinstance(value, np.generic):
        return value
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("an object array can't go to a helper")
        arrays.append(array)
        return {"$a": len(arrays) - 1, "s": isinstance(value, np.generic)}
    if isinstance(value, list):
        return [_encode(v, arrays) for v in value]
    if isinstance(value, tuple):
        return {"$t": [_encode(v, arrays) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"$set": [_encode(v, arrays) for v in value], "f": isinstance(value, frozenset)}
    if isinstance(value, dict):
        return {"$d": [[_encode(k, arrays), _encode(v, arrays)] for k, v in value.items()]}
    name = type(value).__name__
    if name in TYPES and type(value) is _type(name):
        if name == "FlatFaces":
            return {"$c": name, "f": {"indptr": _encode(value.indptr, arrays), "flat": _encode(value.flat, arrays)}}
        return {"$c": name, "f": {f.name: _encode(getattr(value, f.name), arrays) for f in dataclasses.fields(value)}}
    raise TypeError(f"{type(value).__module__}.{name} can't go to a helper")


def _decode(tree, arrays):
    if isinstance(tree, list):
        return [_decode(v, arrays) for v in tree]
    if not isinstance(tree, dict):
        return tree
    if "$a" in tree:
        array = arrays[f"a{int(tree['$a'])}"]
        return array[()] if tree["s"] else array
    if "$t" in tree:
        return tuple(_decode(v, arrays) for v in tree["$t"])
    if "$set" in tree:
        items = [_decode(v, arrays) for v in tree["$set"]]
        return frozenset(items) if tree["f"] else set(items)
    if "$d" in tree:
        return {_decode(k, arrays): _decode(v, arrays) for k, v in tree["$d"]}
    if "$c" in tree:
        return _type(tree["$c"])(**{k: _decode(v, arrays) for k, v in tree["f"].items()})
    raise ValueError("not a helper's file")


def _write(path: Path, value) -> None:
    """Write `value` to `path` so a reader never sees half a file."""
    arrays: list = []
    tree = json.dumps(_encode(value, arrays))
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as handle:
        np.savez(handle, tree=np.asarray(tree), **{f"a{k}": a for k, a in enumerate(arrays)})
    os.replace(tmp, path)


def _read(path: Path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return _decode(json.loads(str(arrays.pop("tree"))), arrays)


def allowed() -> bool:
    """The user's choice in the add-on preferences: may CurioMesh start background Blenders? Off until turned on."""
    from . import overlay

    return bool(getattr(overlay.get_prefs(), "allow_helpers", False))


def helper_command(entry: str, args: list[str], binary: str | None = None) -> tuple[list[str], dict]:
    """(command, environment) for a background Blender that runs `<the add-on>.<entry>(sys.argv)` with `args` after "--".

    `binary` is Blender itself (bpy.app.binary_path) and nothing else, started without a shell. The child enables the
    add-on by its own package name (`bl_ext.<repo>.curiomesh`) with `addon_utils.enable`, the way Blender loads it, and
    then imports the module; nothing edits Python's import path. If the add-on can't be enabled there, the child exits
    and the job does that work itself.
    """
    if binary is None:
        import bpy

        binary = bpy.app.binary_path
    module, _, function = entry.rpartition(".")
    package = PACKAGE
    code = (
        "import addon_utils, importlib, sys; "
        f"addon_utils.enable({package!r}, default_set=False); "
        f"importlib.import_module({package + '.' + module!r}).{function}(sys.argv)"
    )
    return [binary, "--background", "--factory-startup", "--python-expr", code, "--", *args], dict(os.environ)


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
        _write(self.folder / "job.npz", setup)
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
            path = self.folder / f"{self._names[part]}.npz"
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
    """The loop of a helper process (see `helper_command`): rebuild the job, then measure each part it is handed."""
    import bpy

    folder = Path(argv[argv.index("--") + 1])
    autoretopo = importlib.import_module(PACKAGE + ".autoretopo")
    hardware = importlib.import_module(PACKAGE + ".hardware")
    job = worker_job(autoretopo.AutoJob, bpy.context, _read(folder / "job.npz"))
    while True:
        line = sys.stdin.readline()
        if not line:
            os._exit(0)  # the job stopped, or the Blender that started this one is gone
        name, kind, part, number = line.split()
        started = time.perf_counter()
        if kind == "measure":
            result = measure_in_worker(job, int(part), float(number), hardware.peak_memory_mb)
        else:
            result = reduce_in_worker(job, int(part), float(number), hardware.peak_memory_mb)
        print(f"{kind} {part} {number}: {time.perf_counter() - started:.1f} s", "failed" if "error" in result else "", flush=True)
        if "error" in result:
            print(result["error"], flush=True)
        _write(folder / f"{name}.npz", result)


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
