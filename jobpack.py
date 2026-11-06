# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Move an Auto Segments job to another computer and bring the result back.

`export_job` writes a folder: the mesh, the settings and `run_job.py`, which runs the very same `AutoJob` of the
CurioMesh installed in the Blender that runs it, headless (`blender --background --python run_job.py`) and writes `result.npz`. `load_result` reads
that file back as the dict the preview uses. Nothing here talks to a network, an account or a cloud service: you
move the folder yourself. No `bpy` import, so the tests run outside Blender.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RUN_JOB = '''# Runs one CurioMesh Auto Segments job without a window. Made by the CurioMesh add-on.
#   blender --background --python run_job.py [-- --effort FULL]
# Stopped or crashed? Run the same command again: parts already finished are in the curiomesh_resume folder and are kept.
import importlib
import json
import sys
from pathlib import Path

import addon_utils
import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
job = json.loads((HERE / "job.json").read_text())
if "--effort" in args:
    job["effort"] = args[args.index("--effort") + 1].upper()

# CurioMesh must be installed in this Blender (it need not be switched on): find it and switch it on for this run.
found = [m.__name__ for m in addon_utils.modules() if m.__name__.rpartition(".")[2].lower() == "curiomesh" or getattr(m, "bl_info", {}).get("name") == "CurioMesh"]
if not found:
    raise SystemExit("CurioMesh is not installed in this Blender: install the add-on first, then run this again")
addon_utils.enable(found[0], default_set=False)
autoretopo, hardware, jobpack = (importlib.import_module(f"{found[0]}.{name}") for name in ("autoretopo", "hardware", "jobpack"))

data = np.load(HERE / "mesh.npz", allow_pickle=False)
loops, starts, totals = data["loops"].tolist(), data["start"].tolist(), data["total"].tolist()
faces = [tuple(loops[a : a + b]) for a, b in zip(starts, totals)]
mirror = tuple(job["mirror"]) if job.get("mirror") else None  # (axis, plane offset, size, faces of the whole mesh, side solved)
lines, project = [], None
if (HERE / "guides.npz").exists():
    with np.load(HERE / "guides.npz", allow_pickle=False) as guide_file:
        lines = [guide_file[f"l{i}"] for i in range(int(guide_file["n"]))]
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    surface = BVHTree.FromPolygons(data["co"].tolist(), [list(f) for f in faces])
    project = lambda p: surface.find_nearest(Vector(p))[0]  # noqa: E731
drop = data["drop"] if "drop" in data.files else None
machine = hardware.detect()
print("Machine:", "; ".join(hardware.describe(machine)), flush=True)
settings = hardware.settings_for(job["effort"], machine)
print(f"Working at {settings.name}: {len(faces):,} faces", flush=True)
runner = autoretopo.AutoJob(
    bpy.context, data["co"], faces, data["mat"], job["detail"], job["target_faces"], mirror=mirror, guide_lines=lines, project=project, drop=drop,
    settings=settings, checkpoint=hardware.checkpoint_dir(data["co"], len(faces), str(HERE)), keep_progress=True, parallel=True,
)
while runner.busy:
    runner.wait(10.0)
    print(f"{runner.fraction * 100:5.1f}%  {runner.stage}", flush=True)
if runner.warnings or runner.report is None:
    print("FAILED:", runner.warnings or "stopped before the end", flush=True)
    raise SystemExit(1)
jobpack.save_result(HERE / "result.npz", runner.output, runner.report, job["source_faces"], job["source_verts"])
print("Done:", jobpack.summary(runner.report), flush=True)
print("Result written to", HERE / "result.npz", flush=True)
'''

README = """CurioMesh job: {faces:,} faces, {name}
=======================================

This folder is a CurioMesh Auto Segments job. It runs on any computer that has Blender with the CurioMesh add-on
installed (the same version as yours is best; it does not have to be switched on).

Run it
  1. Copy this whole folder to the computer that will do the work.
  2. In a terminal inside the folder run:
         blender --background --python run_job.py
     Add  -- --effort FULL  (or LIGHT, NORMAL) to choose how hard it works. It is set to {effort} now.
     If "blender" is not found, use the full path to the Blender program.
  3. It prints its progress. When it says "Done" the file result.npz is in this folder.
  4. Copy result.npz back to your own computer.
  5. In Blender, in the CurioMesh panel, press "Import Job Result", pick result.npz, then press Apply.

Stopped, or the machine crashed?
  Run the same command again. Finished parts are saved in the curiomesh_resume folder and are not done twice.

Good to know
  - The job uses the same engine as the button in Blender, so the result is the same kind of result.
  - Mirror, guide curves and painted density are carried in the folder (mesh.npz, guides.npz, job.json).
  - Today the work runs on the CPU. A machine with many CPU cores and plenty of memory finishes soonest; a graphics
    card does not speed it up yet. See docs/CLOUD_WORKFLOW.md in the CurioMesh repository.
  - Keep the mesh private if it is private: nothing in this folder sends anything anywhere.
  - run_job.py is a Python program. Only run a job folder you exported yourself or got from someone you trust,
    the same as any script. Importing a result.npz back runs nothing.
"""


def job_folder(directory, name: str) -> Path:
    """<directory>/<name>_curiomesh_job, with the name cleaned. Raises ValueError if it would land outside `directory`."""
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", name).lstrip(".") or "mesh"
    base = Path(directory).resolve()
    folder = (base / f"{clean}_curiomesh_job").resolve()
    if folder.parent != base:
        raise ValueError("That folder name would leave the folder you picked")
    return folder


def export_job(folder, co, faces, mat, detail: float, target_faces: int, effort: str, addon_dir=None, mirror=None, lines=(), drop=None, source=None) -> Path:
    """Write the job folder. `co`/`faces`/`mat` are what the search runs on (the solved half when `mirror` is set).

    `mirror` is (axis, offset, size, faces of the whole mesh, side solved); `lines` the guide strokes already on that half;
    `drop` the painted density per vertex; `source` the (faces, vertices) of the whole mesh the result must fit. Returns the folder.
    `addon_dir` is no longer used (the folder holds no copy of the add-on).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    extra = {} if drop is None else {"drop": np.asarray(drop, dtype=np.float64)}
    np.savez(folder / "mesh.npz", co=np.asarray(co, dtype=np.float64), loops=np.fromiter((v for f in faces for v in f), dtype=np.int64), start=np.cumsum(sizes) - sizes, total=sizes, mat=np.asarray(mat, dtype=np.int32), **extra)
    if len(lines):
        np.savez(folder / "guides.npz", n=np.int64(len(lines)), **{f"l{i}": np.asarray(line, dtype=np.float64) for i, line in enumerate(lines)})
    source = source or (len(faces), len(co))
    settings = {"detail": float(detail), "target_faces": int(target_faces), "effort": effort, "faces": len(faces), "verts": len(co), "source_faces": int(source[0]), "source_verts": int(source[1])}
    settings["mirror"] = None if mirror is None else [int(mirror[0]), float(mirror[1]), float(mirror[2]), int(mirror[3]), int(mirror[4])]
    (folder / "job.json").write_text(json.dumps(settings, indent=2))
    (folder / "run_job.py").write_text(RUN_JOB)
    (folder / "README.txt").write_text(README.format(faces=len(faces), name=folder.name, effort=effort))
    return folder


def summary(report: dict) -> str:
    return f"{report['faces']:,} faces ({report['quads']:,} quads), the view changed by {report['loss']:.1f}%, {report['seconds']:.0f} s"


def save_result(path, output: dict, report: dict, source_faces: int, source_verts: int) -> None:
    faces = output["faces"]
    np.savez(
        path,
        co=np.asarray(output["co"], dtype=np.float64),
        lens=np.fromiter((len(f) for f in faces), dtype=np.int32, count=len(faces)),
        flat=np.fromiter((v for f in faces for v in f), dtype=np.int64),
        mat=np.asarray(output["mat"], dtype=np.int32),
        parts=np.asarray(output["parts"], dtype=np.int64),
        source=np.asarray([source_faces, source_verts], dtype=np.int64),
        report=np.asarray(json.dumps({k: v for k, v in report.items() if k != "guides"})),
    )


def load_result(path, source_faces: int, source_verts: int):
    """(result dict like the preview's, report). Raises ValueError when it was made from a different mesh."""
    try:
        from .hardware import load_npz
    except ImportError:  # loaded as a plain module (tests)
        from hardware import load_npz

    with load_npz(path) as data:
        made_from = data["source"].tolist()
        if made_from != [source_faces, source_verts]:
            raise ValueError(f"This result was made from a mesh with {made_from[0]:,} faces; the selected object has {source_faces:,}")
        co, lens, flat, mat, parts = data["co"], data["lens"], data["flat"], data["mat"], data["parts"]
        report = json.loads(str(data["report"]))
    # the file may come from anywhere: check every array before a face is built from it
    count = len(lens)
    if any(a.dtype.kind not in "iu" for a in (lens, flat, mat, parts)) or co.dtype.kind != "f" or mat.ndim != 1 or parts.ndim != 1:
        raise ValueError("This result file is damaged (wrong kinds of numbers)")
    if co.ndim != 2 or co.shape[1] != 3 or not np.isfinite(co).all() or lens.ndim != 1 or count == 0 or lens.min() < 3 or lens.max() > 64:
        raise ValueError("This result file is damaged (bad points or faces)")
    if int(lens.sum()) != len(flat) or flat.min() < 0 or flat.max() >= len(co) or len(mat) != count or len(parts) != count:
        raise ValueError("This result file is damaged (indices out of range)")
    if not isinstance(report, dict) or not all(isinstance(report.get(k), (int, float)) for k in ("faces", "quads", "loss", "seconds")):
        raise ValueError("This result file is damaged (bad report)")
    ends = np.cumsum(lens)
    faces = [tuple(int(v) for v in flat[a:b]) for a, b in zip(ends - lens, ends)]
    result = {"co": co, "faces": faces, "new": np.ones(count, bool), "locked": np.zeros(count, bool), "mat": mat, "parts": parts}
    return result, report
