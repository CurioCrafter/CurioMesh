# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Move an Auto Segments job to another computer and bring the result back.

`export_job` writes a folder: the mesh, the settings, a copy of the engine and `run_job.py`, which runs the very
same `AutoJob` headless (`blender --background --python run_job.py`) and writes `result.npz`. `load_result` reads
that file back as the dict the preview uses. Nothing here talks to a network, an account or a cloud service: you
move the folder yourself. No `bpy` import, so the tests run outside Blender.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

ENGINE = "curiomesh_engine"

RUN_JOB = '''# Runs one CurioMesh Auto Segments job without a window. Made by the CurioMesh add-on.
#   blender --background --python run_job.py [-- --effort FULL]
# Stopped or crashed? Run the same command again: parts already finished are in the progress folder and are kept.
import json
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
job = json.loads((HERE / "job.json").read_text())
if "--effort" in args:
    job["effort"] = args[args.index("--effort") + 1].upper()

from curiomesh_engine import autoretopo, hardware, jobpack  # noqa: E402

data = np.load(HERE / "mesh.npz", allow_pickle=False)
loops, starts, totals = data["loops"].tolist(), data["start"].tolist(), data["total"].tolist()
faces = [tuple(loops[a : a + b]) for a, b in zip(starts, totals)]
machine = hardware.detect()
print("Machine:", "; ".join(hardware.describe(machine)), flush=True)
settings = hardware.settings_for(job["effort"], machine)
print(f"Working at {settings.name}: {len(faces):,} faces", flush=True)
runner = autoretopo.AutoJob(
    bpy.context, data["co"], faces, data["mat"], job["detail"], job["target_faces"],
    settings=settings, checkpoint=HERE / "progress", keep_progress=True,
)
while runner.busy:
    runner.wait(10.0)
    print(f"{runner.fraction * 100:5.1f}%  {runner.stage}", flush=True)
if runner.warnings or runner.report is None:
    print("FAILED:", runner.warnings or "stopped before the end", flush=True)
    raise SystemExit(1)
jobpack.save_result(HERE / "result.npz", runner.output, runner.report, len(faces), len(data["co"]))
print("Done:", jobpack.summary(runner.report), flush=True)
print("Result written to", HERE / "result.npz", flush=True)
'''

README = """CurioMesh job: {faces:,} faces, {name}
=======================================

This folder is a CurioMesh Auto Segments job. It runs on any computer that has Blender (3.6 or newer) and does not
need the CurioMesh add-on installed: a copy of the engine is inside ({engine}/).

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
  Run the same command again. Finished parts are saved in the progress folder and are not done twice.

Good to know
  - The job uses the same engine as the button in Blender, so the result is the same kind of result.
  - It solves the whole mesh: guide curves, painted density and Mirror are not part of an exported job.
  - Today the work runs on the CPU. A machine with many CPU cores and plenty of memory finishes soonest; a graphics
    card does not speed it up yet. See docs/CLOUD_WORKFLOW.md in the CurioMesh repository.
  - Keep the mesh private if it is private: nothing in this folder sends anything anywhere.
"""


def job_folder(directory, name: str) -> Path:
    """<directory>/<name>_curiomesh_job, with the name cleaned. Raises ValueError if it would land outside `directory`."""
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", name).lstrip(".") or "mesh"
    base = Path(directory).resolve()
    folder = (base / f"{clean}_curiomesh_job").resolve()
    if folder.parent != base:
        raise ValueError("That folder name would leave the folder you picked")
    return folder


def _copy_engine(source: Path, target: Path) -> None:
    """The add-on's Python files (not tests, docs or compiled binaries) as a package next to run_job.py."""
    for name in [*source.glob("*.py"), *source.glob("triadq/*.py")]:
        destination = target / name.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)


def export_job(folder, co, faces, mat, detail: float, target_faces: int, effort: str, addon_dir) -> Path:
    """Write the job folder. `co`/`faces`/`mat` are the (welded) source mesh; returns the folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    np.savez(folder / "mesh.npz", co=np.asarray(co, dtype=np.float64), loops=np.fromiter((v for f in faces for v in f), dtype=np.int64), start=np.cumsum(sizes) - sizes, total=sizes, mat=np.asarray(mat, dtype=np.int32))
    (folder / "job.json").write_text(json.dumps({"detail": float(detail), "target_faces": int(target_faces), "effort": effort, "faces": len(faces), "verts": len(co)}, indent=2))
    _copy_engine(Path(addon_dir), folder / ENGINE)
    (folder / "run_job.py").write_text(RUN_JOB)
    (folder / "README.txt").write_text(README.format(faces=len(faces), name=folder.name, engine=ENGINE, effort=effort))
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
    with np.load(path, allow_pickle=False) as data:
        made_from = data["source"].tolist()
        if made_from != [source_faces, source_verts]:
            raise ValueError(f"This result was made from a mesh with {made_from[0]:,} faces; the selected object has {source_faces:,}")
        co, lens, flat, mat, parts = data["co"], data["lens"], data["flat"], data["mat"], data["parts"]
        report = json.loads(str(data["report"]))
    # the file may come from anywhere: check every array before a face is built from it
    count = len(lens)
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
