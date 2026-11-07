# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""What this computer can do, and how to cut a job down to fit it.

`detect` reads the cores and memory (on Windows the memory is assumed) and never fails: whatever it cannot read is left
out. `settings_for` turns an effort (Light, Normal, Full, or Auto) into the knobs `autoretopo.AutoJob` uses.
`estimate` and `rate` turn face and part counts into a time, a memory figure and one plain sentence. `split_chunks`
cuts a huge piece into compact regions, and the checkpoint helpers save finished parts so a stopped job can resume.

No imports from the rest of the add-on and no `bpy` at import time: helper Blenders import it too.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EFFORT_ITEMS = [
    ("AUTO", "Auto", "Pick Light, Normal or Full from this computer's cores and free memory", "AUTO", 0),
    ("LIGHT", "Light", "Work in small steps and leave most of the computer free: short slices with long gaps, fewer camera rays, smaller pieces held in memory. Slower, and the view check is coarser", "TIME", 1),
    ("NORMAL", "Normal", "The usual balance between speed and keeping Blender responsive", "PREFERENCES", 2),
    ("FULL", "Full", "Use as much of the computer as it takes to finish soonest. Blender is sluggish while it runs", "FAST_FORWARD", 3),
]
BYTES_PER_FACE = 1500  # working memory per face of one chunk (Python lists, a decimated copy, the BVH), measured


@dataclass(frozen=True)
class Settings:
    """The knobs of one job. Everything an effort changes is here."""

    name: str
    slice_s: float  # seconds of work per timer call
    gap_s: float  # seconds Blender gets back between slices
    max_rays: int  # budget of camera rays for one look at every piece
    max_pixels: int  # a side of one picture
    chunk_faces: int  # a piece with more faces than this is cut into regions
    show_every: float  # seconds between preview updates
    only_done: bool  # the preview shows the parts that are finished, not the unmeasured rest at full size


EFFORTS = {
    "LIGHT": Settings("Light", 0.015, 0.06, 5_000_000, 128, 40_000, 3.0, True),
    "NORMAL": Settings("Normal", 0.04, 0.01, 20_000_000, 192, 150_000, 1.0, False),
    "FULL": Settings("Full", 0.12, 0.002, 30_000_000, 192, 400_000, 1.0, False),
}


# --------------------------------------------------------------------------- the machine


GUESS_RAM_GB = 16.0  # Windows: Python can't read the memory without calling into the system, so a modest guess


def _memory() -> tuple[float | None, float | None]:
    """(total GB, free GB) of RAM from what Python can read; on Windows a guess of `GUESS_RAM_GB`, half of it free."""
    try:
        if sys.platform.startswith("linux"):
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                info[key] = float(rest.split()[0]) / 2**20  # kB -> GB
            return info["MemTotal"], info.get("MemAvailable", info["MemFree"])
        if sys.platform == "darwin":
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
            return total, total * 0.5  # no cheap exact figure: assume half is free
    except (OSError, ValueError, KeyError, IndexError):
        return None, None
    return GUESS_RAM_GB, GUESS_RAM_GB * 0.5


def detect() -> dict:
    """Read the computer. Never raises; unknown things are left out."""
    total, free = _memory()
    machine = {"os": {"win32": "Windows", "darwin": "macOS"}.get(sys.platform, "Linux" if sys.platform.startswith("linux") else sys.platform), "cores": os.cpu_count() or 1, "ram_gb": total, "free_gb": free}
    if sys.platform == "win32":
        machine["ram_guessed"] = True
    return machine


def describe(machine: dict) -> list[str]:
    """The machine in plain words, one short line each."""
    lines = [f"{machine['os']}, {machine['cores']} cores"]
    if machine.get("ram_guessed"):
        lines.append(f"Memory: not read on Windows, assumed {machine['ram_gb']:.0f} GB")
    elif machine.get("ram_gb"):
        lines.append(f"Memory: {machine['ram_gb']:.0f} GB, {machine['free_gb']:.0f} GB free")
    return lines


def benchmark() -> float:
    """Camera rays per second on this computer, from a 0.1 s test on a small sphere."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    n, m = 40, 80
    theta = np.linspace(0.05, np.pi - 0.05, n)[:, None]
    phi = np.linspace(0, 2 * np.pi, m, endpoint=False)[None, :]
    pts = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) + 0 * phi], axis=-1).reshape(-1, 3)
    quads = [(i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j) for i in range(n - 1) for j in range(m)]
    tree = BVHTree.FromPolygons(pts.tolist(), quads, all_triangles=False)
    cast, direction = tree.ray_cast, Vector((0.0, 0.0, -1.0))
    started, count = time.perf_counter(), 0
    grid = np.linspace(-0.9, 0.9, 150).tolist()
    while time.perf_counter() - started < 0.12:
        for x in grid:
            for y in grid[::5]:
                cast(Vector((x, y, 3.0)), direction)
                count += 1
    return count / (time.perf_counter() - started)


def peak_memory_mb() -> float:
    """The most memory this process has used so far, in MB (0 where it can't be read)."""
    try:
        if sys.platform == "win32":
            return 0.0  # Python has no reading for this on Windows without calling into the system
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024.0
    except Exception:
        return 0.0


def pick_effort(machine: dict) -> str:
    """Auto: the effort this computer can carry. Few cores or little free memory means Light."""
    free, cores = machine.get("free_gb"), machine["cores"]
    if (free is not None and free < 6.0) or cores <= 2:
        return "LIGHT"
    if (free is None or free >= 16.0) and cores >= 8:
        return "FULL"
    return "NORMAL"


def settings_for(effort: str, machine: dict | None = None) -> Settings:
    """The knobs for an effort. A chunk is also kept small enough for the free memory (a fifth of it)."""
    machine = machine or {"cores": os.cpu_count() or 1, "free_gb": None}
    key = pick_effort(machine) if effort == "AUTO" else effort
    base = EFFORTS[key]
    chunk = base.chunk_faces
    if machine.get("free_gb"):
        chunk = min(chunk, max(int(machine["free_gb"] * 2**30 * 0.2 / BYTES_PER_FACE), 20_000))
    return Settings(base.name, base.slice_s, base.gap_s, base.max_rays, base.max_pixels, chunk, base.show_every, base.only_done)


def busy_cores() -> float:
    """About how many cores other programs keep busy right now: the load average (0 on Windows, where Python can't read it)."""
    try:
        return float(os.getloadavg()[0])
    except Exception:
        return 0.0


WORKER_MB = 400  # one background Blender with the add-on loaded and nothing else, measured
WORKER_BYTES_PER_JOB_FACE = 400  # each helper holds the whole mesh (points, faces, one material string a face), measured
RAM_CEILING = 0.85  # helpers never take the computer past this share of its memory
PARALLEL_FROM = 40_000  # triangles in the parts left to measure: below this, starting helpers costs about what it saves (28k: 2 s of 83)


def worker_mb(job_faces: int, biggest_part_faces: int) -> float:
    """Memory one helper process is expected to need, in MB: Blender, the whole mesh, and the working copy of its biggest part."""
    return WORKER_MB + (job_faces * WORKER_BYTES_PER_JOB_FACE + biggest_part_faces * BYTES_PER_FACE) / 2**20


def worker_count(machine: dict, effort: str, job_faces: int, biggest_part_faces: int, work_triangles: int, parts: int, busy: float = 0.0, most: int = 0) -> int:
    """How many helper processes measure parts at once: 0 means the job stays on its own.

    From the free cores (Light takes a quarter of them and at most two, Normal half, Full all but the one Blender
    runs on) and the free memory: each helper's need is estimated first, and together they never take the computer
    past `RAM_CEILING` of its memory. One helper alone gains nothing, so that is 0 as well. `most` caps it (0 = no cap).
    """
    if work_triangles < PARALLEL_FROM or parts < 2:
        return 0
    free_cores = max(int((machine.get("cores") or 1) - 1 - busy), 0)
    count = {"Light": min(free_cores // 4, 2), "Normal": free_cores // 2}.get(effort, free_cores)
    total, free = machine.get("ram_gb"), machine.get("free_gb")
    if total and free is not None:
        room_mb = (free - (1.0 - RAM_CEILING) * total) * 1024.0
        count = min(count, int(room_mb // worker_mb(job_faces, biggest_part_faces)))
    count = min(count, parts, most or count)
    return count if count >= 2 else 0


# --------------------------------------------------------------------------- how big, how long


SECONDS_PER_FACE = 0.0008  # decimating, pairing and Python bookkeeping per face; fitted to timed runs of a 1.5M-face scan
DEFAULT_RAYS_PER_S = 150_000.0


def estimate(faces: int, parts: int | None, settings: Settings, rays_per_s: float = 0.0) -> dict:
    """Rough seconds and peak memory (GB) for a job. `parts` is the number of loose pieces (None = unknown)."""
    parts = max(parts if parts is not None else faces // 5000, 1)
    units = parts + faces // max(settings.chunk_faces, 1)  # pieces plus the extra regions huge ones are cut into
    pixels = min(settings.max_pixels, 77)
    rays = min(units * 6 * pixels * pixels * 5, settings.max_rays * 5) + 6 * settings.max_pixels**2 * 4
    seconds = rays / (rays_per_s or DEFAULT_RAYS_PER_S) + faces * SECONDS_PER_FACE
    seconds *= 1.0 + settings.gap_s / settings.slice_s * 0.35  # the gaps between slices leave the computer idle part of the time
    memory = (min(faces, settings.chunk_faces) * BYTES_PER_FACE + faces * 3700) / 2**30  # 3.7 KB a face held for the whole job: 5.5 GB peak measured on 1.5M faces
    return {"seconds": seconds, "gb": memory, "units": units}


def clock(seconds: float) -> str:
    minutes = int(round(seconds / 60.0))
    if seconds < 90:
        return f"about {max(int(round(seconds)), 1)} seconds"
    if minutes < 90:
        return f"about {minutes} minutes"
    return f"about {seconds / 3600.0:.1f} hours"


SLOW_SPREAD = 4.0  # timed runs took 1.3x to 4x the estimate with Auto; Even Rows, Field Rows and QuadriFlow far longer
SLOW_METHODS_SPREAD = 20.0


def span(seconds: float, slow: bool = False) -> str:
    """The estimate as a range: from it to a few times it (many times with the slow methods)."""
    low, high = clock(seconds), clock(seconds * (SLOW_METHODS_SPREAD if slow else SLOW_SPREAD))
    return low if low == high else f"{low} to {high.removeprefix('about ')}"


def rate(machine: dict, est: dict, settings: Settings, slow: bool = False) -> tuple[str, str]:
    """(level, sentence) for the panel. Level is OK, LARGE or TOO_BIG. `slow`: Even Rows, Field Rows or QuadriFlow."""
    free = machine.get("free_gb")
    level = "OK"
    if est["seconds"] > 600 or (free and est["gb"] > 0.25 * free):
        level = "LARGE"
    if est["seconds"] > 4 * 3600 or (free and est["gb"] > 0.7 * free):
        level = "TOO_BIG"
    when = span(est["seconds"], slow)
    steps = ", working in small steps" if settings.name == "Light" or level != "OK" else ""
    if level == "OK":
        return level, f"This mesh suits this PC: {when}{steps}"
    if level == "LARGE":
        return level, f"This mesh is large for this PC: {when}{steps}"
    return level, f"This mesh is too big for this PC to do comfortably: {when}. Export it and run it on a stronger machine"


# --------------------------------------------------------------------------- what the heavy options cost


# Bakes run in Cycles on the main thread and block Blender. Fitted on two timed CPU bakes (Normal + Color, 4 render
# threads, new UVs, 16 samples) of an 86,000-face free model onto a 3,000-face copy: 512 px took 0.66 s and 1024 px 2.51 s,
# i.e. 1.18 s per megapixel per map and next to no fixed part. The source's face count is not modelled (a denser source
# makes each ray dearer); the unwrap of 3,000 faces is inside the fixed part.
BAKE_FIXED_S = 0.04
BAKE_S_PER_MPIX_MAP = 1.18
BAKE_FIT_THREADS = 4  # more cores bake faster, up to BAKE_MOST_THREADS (limitation: assumes bake time falls linearly with cores)
BAKE_MOST_THREADS = 16
BAKE_MAPS = {"OFF": 0, "NORMAL": 1, "NORMAL_COLOR": 2}
# Not fitted: how much longer than Keep Detail each style runs. Even Rows (QuadriFlow, capped at 10 minutes a try) and the
# field methods are the ones the 20x spread above is for.
STYLE_FACTOR = {"DETAIL": 1.0, "BALANCED": 1.0, "ROWS": 2.0}
METHOD_FACTOR = {"FIELD": 2.0, "QUADRIFLOW": 2.0}
CONFIRM_FROM_S = 600.0  # ask before a job expected to run longer than this
CONFIRM_BAKE_S = 120.0  # ...or an Apply whose bake alone runs longer


def bake_seconds(size: int, bake: str, sets: int = 1, machine: dict | None = None) -> float:
    """Seconds one Apply spends baking (0 when Off). Blender is frozen for all of it."""
    maps = BAKE_MAPS.get(bake, 0)
    if not maps:
        return 0.0
    threads = min(max((machine or {}).get("cores") or BAKE_FIT_THREADS, 1), BAKE_MOST_THREADS)
    megapixels = (int(size) / 1024.0) ** 2 * 1.048576 * maps * max(int(sets) if bake == "NORMAL_COLOR" else 1, 1)
    return (BAKE_FIXED_S + BAKE_S_PER_MPIX_MAP * megapixels) * BAKE_FIT_THREADS / threads


def style_factor(style: str, method: str = "") -> float:
    return max(STYLE_FACTOR.get(style, 1.0), METHOD_FACTOR.get(method, 1.0))


def cost_tag(seconds: float, runs: int = 1) -> tuple[str, bool]:
    """(short line, heavy?) for a label next to an option. Under 15 seconds there is nothing to say."""
    if seconds < 15:
        return "", False
    heavy = seconds * runs >= 180
    each = f" each, {runs} runs" if runs > 1 else ""
    return f"{'Heavy' if heavy else 'Slow'}: {clock(seconds)}{each}", heavy


def with_bake(text: str, bake_s: float) -> str:
    """The panel's time sentence, with the bake added when there is one."""
    return text if bake_s < 15 else f"{text}, plus {clock(bake_s)} of baking that blocks Blender"


def needs_confirm(level: str, seconds: float, bake_s: float = 0.0, applying: bool = False) -> bool:
    """Ask first? Starts: a LARGE or TOO_BIG rating or a long estimate. Apply: only a long bake."""
    if applying:
        return bake_s > CONFIRM_BAKE_S
    return level != "OK" or seconds + bake_s > CONFIRM_FROM_S


def cheaper_ways(style: str, bake: str, size: int, effort: str, lods: bool = False) -> list[str]:
    """What the user could change first, only the ones that still help."""
    ways = []
    if effort != "LIGHT":
        ways.append("Effort: Light (gentler on this PC, slower)")
    if style == "ROWS":
        ways.append("Style: Fewest Faces instead of Even Rows")
    if bake != "OFF" and int(size) > 1024:
        ways.append(f"A smaller bake: {int(size) // 2} px takes a quarter of the time")
    ways.append("Export Job (This Computer) to run it on a stronger machine")
    return ways


def weak_pc(machine: dict | None) -> str:
    """One plain sentence when this computer is weak (4 cores or fewer, or 8 GB or less), else an empty string.

    Only cores and memory count: bakes use the scene's Cycles device."""
    if not machine:
        return ""
    small = []
    if (machine.get("cores") or 99) <= 4:
        small.append(f"{machine['cores']} cores")
    if (machine.get("ram_gb") or 99) <= 8:
        small.append(f"{machine['ram_gb']:.0f} GB of memory")
    if not small:
        return ""
    return f"This PC is on the small side ({', '.join(small)}). Big meshes and bakes will be slow. Effort Light, a smaller bake, Keep Detail and Export Job (to another computer) all help."


# --------------------------------------------------------------------------- chunks


def split_chunks(centers: np.ndarray, limit: int) -> list[np.ndarray]:
    """Cut face centres into compact regions of at most `limit` faces, by halving along the longest side."""
    out, todo = [], [np.arange(len(centers))]
    while todo:
        ids = todo.pop()
        if len(ids) <= limit:
            out.append(ids)
            continue
        c = centers[ids]
        axis = int(np.argmax(np.ptp(c, axis=0)))
        order = np.argsort(c[:, axis], kind="stable")
        half = len(ids) // 2
        todo += [ids[order[half:]], ids[order[:half]]]
    return out[::-1] if len(out) > 1 else out


def border_vertices(faces, label: np.ndarray, vertex_count: int) -> np.ndarray:
    """True for the vertices used by faces of more than one label (a chunk border)."""
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    flat = np.fromiter((v for f in faces for v in f), dtype=np.int64, count=int(sizes.sum()))
    owner = np.repeat(label, sizes)
    low = np.full(vertex_count, np.iinfo(np.int64).max)
    high = np.full(vertex_count, -1)
    np.minimum.at(low, flat, owner)
    np.maximum.at(high, flat, owner)
    return low != high


def _tuples(sizes: np.ndarray, flat: np.ndarray) -> list:
    """Faces as a list of tuples from their sizes and corners, a row of NumPy at a time for each size."""
    out: list = [None] * len(sizes)
    starts = np.cumsum(sizes) - sizes
    for size in np.unique(sizes).tolist():
        ids = np.flatnonzero(sizes == size)
        rows = map(tuple, flat[starts[ids][:, None] + np.arange(size)].tolist())
        for at, row in zip(ids.tolist(), rows):
            out[at] = row
    return out


def weld(points: np.ndarray, faces: list, only: np.ndarray | None = None) -> tuple[np.ndarray, list]:
    """Merge vertices at exactly the same place (the chunk borders, which were kept unmoved). `only`: just these may merge."""
    if only is None:
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
    else:  # the rest keep their own vertex even where another sits on it
        some = np.flatnonzero(only)
        _same, first, back = np.unique(points[some], axis=0, return_index=True, return_inverse=True)
        target = np.arange(len(points))
        target[some] = some[first[back.reshape(-1)]]
        kept, inverse = np.unique(target, return_inverse=True)
        unique = points[kept]
    sizes = np.fromiter(map(len, faces), dtype=np.int64, count=len(faces))
    flat = np.fromiter((v for f in faces for v in f), dtype=np.int64, count=int(sizes.sum()))
    return unique, _tuples(sizes, inverse.reshape(-1)[flat])


def region_name(center, low, high) -> str:
    """Where a part is, in words: 'upper left front', or 'middle'."""
    words = (("left", "right"), ("front", "back"), ("lower", "upper"))
    parts = []
    for axis in (2, 0, 1):
        span = high[axis] - low[axis]
        t = (center[axis] - low[axis]) / span if span > 0 else 0.5
        if t < 0.34:
            parts.append(words[axis][0])
        elif t > 0.66:
            parts.append(words[axis][1])
    return " ".join(parts) or "middle"


# --------------------------------------------------------------------------- saving progress


def _private_dir(folder: Path) -> bool:
    """Make `folder` (0700) and check it is a real directory of ours, not a symlink or someone else's."""
    try:
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        if sys.platform != "win32":
            info = os.lstat(folder)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                return False
            if info.st_mode & 0o077:
                os.chmod(folder, 0o700)
        return folder.is_dir() and not folder.is_symlink()
    except OSError:
        return False


def user_folder(sub: str = "", create: bool = True) -> Path | None:
    """The extension's own folder for files it keeps (Blender's extension user folder), None when there is none."""
    try:
        import bpy

        return Path(bpy.utils.extension_path_user(__package__ or "", path=sub, create=create))
    except Exception:  # the folder can't be made (read-only disk, no permission)
        return None


STALE_JOB_S = 12 * 3600  # a job folder nothing has touched for this long was left by a Blender that quit or crashed
STALE_RESUME_S = 3 * 24 * 3600  # saved progress of a stopped job is kept this long, then cleared when the next job starts


def _clear_old(base: Path | None, age: float) -> int:
    """Delete the folders in `base` nothing has touched for `age` seconds (all of them for age 0). Returns the bytes freed."""
    freed = 0
    if base is None or not base.is_dir():
        return 0
    for old in base.iterdir():
        try:
            if old.is_dir() and not old.is_symlink() and time.time() - old.stat().st_mtime >= age:
                freed += sum(f.stat().st_size for f in old.rglob("*") if f.is_file())
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return freed


def clear_saved_progress() -> int:
    """The preferences' Clear Saved Progress button: deletes every resume point. Returns the bytes freed."""
    return _clear_old(user_folder("curiomesh_resume", create=False), 0.0)


def job_folder(prefix: str) -> Path:
    """A new folder for one job's working files, in the extension's user folder ("jobs"). Clears folders left there by
    a Blender that quit mid-job. Raises RuntimeError when that folder can't be made: nothing goes to system temp."""
    base = user_folder("jobs")
    if base is None:
        raise RuntimeError("CurioMesh can't make its folder in Blender's extension user folder for this job's files")
    _clear_old(base, STALE_JOB_S)  # Age-based: another Blender may be running a job in this folder right now.
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def checkpoint_dir(co: np.ndarray, face_count: int, root: str | None = None) -> Path | None:
    """The folder that holds the saved parts of the job on this very mesh: private to this user, or None if none is safe."""
    if root:
        folder = Path(root) / "curiomesh_resume"
    else:
        base = user_folder()
        if base is None:
            return None
        folder = base / "curiomesh_resume"
        _clear_old(folder, STALE_RESUME_S)  # progress of jobs stopped days ago: kept no longer
    if folder.exists() and (folder.is_symlink() or not folder.is_dir()):
        return None  # no checkpoints rather than files in a place someone else could plant
    digest = hashlib.sha1(np.ascontiguousarray(co).tobytes() + str(face_count).encode()).hexdigest()[:16]
    return folder / digest  # made by the first save_piece, so a job stopped before any part leaves no empty folder


def piece_file(folder: Path, key: bytes) -> Path:
    return folder / (hashlib.sha1(key).hexdigest()[:20] + ".npz")


def save_piece(path: Path, detail: float, sizes: list, levels: list[tuple[int, float]], seen: float) -> None:
    """Write one measured part. `sizes` are (points, polys, kinds) triples, `levels` (faces, loss) pairs."""
    data = {"detail": np.float64(detail), "seen": np.float64(seen), "n": np.int64(len(sizes)), "levels": np.asarray(levels, dtype=np.float64).reshape(-1, 2)}
    for k, (points, polys, kinds) in enumerate(sizes):
        data[f"p{k}"] = np.asarray(points, dtype=np.float64)
        data[f"l{k}"] = np.fromiter((len(p) for p in polys), dtype=np.int32, count=len(polys))
        data[f"f{k}"] = np.fromiter((v for p in polys for v in p), dtype=np.int64)
        data[f"m{k}"] = np.asarray([int(m) for m in kinds], dtype=np.int32)
    if not (_private_dir(path.parent.parent) and _private_dir(path.parent)):  # the checkpoint folders are made here, on the first part
        raise OSError(f"{path.parent} is not a private folder of this user")
    descriptor, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")  # a name nobody could have planted
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **data)
        os.replace(tmp, path)  # a crash while writing never leaves a half file under the real name
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


MOST_NPZ_BYTES = 2**30  # unpacked size of any .npz read back (results, saved parts): a small zip can unpack to far more
MOST_NPZ_ENTRIES = 100_000


def load_npz(path):
    """np.load for a .npz that may come from anywhere: no pickles, and refused before reading if it unpacks too big."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as packed:
            infos = packed.infolist()
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{Path(path).name} is not a CurioMesh .npz file") from exc
    if len(infos) > MOST_NPZ_ENTRIES or sum(info.file_size for info in infos) > MOST_NPZ_BYTES:
        raise ValueError(f"{Path(path).name} unpacks to more than {MOST_NPZ_BYTES // 2**20:,} MB or {MOST_NPZ_ENTRIES:,} arrays: not read")
    return np.load(path, allow_pickle=False)


def load_piece(path: Path):
    """(detail, sizes, levels, seen) of a saved part, or None if it is missing or unreadable."""
    try:
        with load_npz(path) as data:
            sizes = []
            for k in range(int(data["n"])):
                lens, flat = data[f"l{k}"], data[f"f{k}"]
                points = data[f"p{k}"]
                if points.ndim != 2 or points.shape[1] != 3 or lens.min(initial=3) < 3 or int(lens.sum()) != len(flat) or (len(flat) and (flat.min() < 0 or flat.max() >= len(points))):
                    return None  # not a file we wrote
                polys = _tuples(np.asarray(lens, dtype=np.int64), np.asarray(flat, dtype=np.int64))
                sizes.append((data[f"p{k}"], polys, [str(int(m)) for m in data[f"m{k}"]]))
            return float(data["detail"]), sizes, [(int(f), float(w)) for f, w in data["levels"]], float(data["seen"])
    except Exception:
        return None
