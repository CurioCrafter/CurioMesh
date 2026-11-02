# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""What this computer can do, and how to cut a job down to fit it.

`detect` reads the machine (cores, memory, graphics card, Cycles devices) and never fails: whatever it cannot read
is left out. `settings_for` turns an effort (Light, Normal, Full, or Auto) into the knobs `autoretopo.AutoJob` uses.
`estimate` and `rate` turn face and part counts into a time, a memory figure and one plain sentence. `split_chunks`
cuts a huge piece into compact regions, and the checkpoint helpers save finished parts so a stopped job can resume.

No imports from the rest of the add-on and no `bpy` at import time, so the tests run outside Blender.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
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
    slice_s: float  # seconds of work per timer call on the main thread
    gap_s: float  # seconds Blender gets back between slices
    max_rays: int  # budget of camera rays for one look at every piece
    max_pixels: int  # a side of one picture
    chunk_faces: int  # a piece with more faces than this is cut into regions
    threads: int  # Turbo threads (0 = all of them)
    show_every: float  # seconds between preview updates
    only_done: bool  # the preview shows the parts that are finished, not the unmeasured rest at full size


EFFORTS = {
    "LIGHT": Settings("Light", 0.015, 0.06, 5_000_000, 128, 40_000, 1, 3.0, True),
    "NORMAL": Settings("Normal", 0.04, 0.01, 20_000_000, 192, 150_000, 0, 1.0, False),
    "FULL": Settings("Full", 0.12, 0.002, 30_000_000, 192, 400_000, 0, 1.0, False),
}


# --------------------------------------------------------------------------- the machine


def _memory() -> tuple[float | None, float | None]:
    """(total GB, free GB) of RAM, or None where it can't be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("total", "free", "ptotal", "pfree", "vtotal", "vfree", "ext")]

            status = Status()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 2**30, status.free / 2**30
        if sys.platform.startswith("linux"):
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                info[key] = float(rest.split()[0]) / 2**20  # kB -> GB
            return info["MemTotal"], info.get("MemAvailable", info["MemFree"])
        if sys.platform == "darwin":
            total = float(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3).stdout) / 2**30
            return total, total * 0.5  # no cheap exact figure: assume half is free
    except Exception:
        pass
    return None, None


def _on_path(name: str) -> str | None:
    """`name` from the absolute folders on PATH only, never the current folder.

    shutil.which on Python 3.11 and older (Blender 3.6 - 4.x) looks in the current folder first on Windows, and that
    can be the folder of a .blend somebody sent, so a planted nvidia-smi.exe there would run. Only .exe on Windows.
    """
    suffix = ".exe" if sys.platform == "win32" else ""
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        folder = folder.strip().strip('"')
        if not folder or not os.path.isabs(folder):
            continue
        candidate = os.path.join(folder, name + suffix)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _gpu() -> dict:
    """The NVIDIA card as nvidia-smi sees it. Empty when there is none or the tool is missing."""
    exe = _on_path("nvidia-smi")
    if exe is None:
        return {}
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout
        name, total, used = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"gpu": name, "vram_gb": float(total) / 1024.0, "vram_free_gb": (float(total) - float(used)) / 1024.0, "cuda": True}
    except Exception:
        return {}


def _cycles_devices() -> list[str]:
    """Names of the non-CPU devices Blender's Cycles can see."""
    try:
        import bpy

        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.refresh_devices() if hasattr(prefs, "refresh_devices") else None
        return sorted({f"{d.name} ({d.type})" for d in prefs.devices if d.type != "CPU"})
    except Exception:
        return []


def detect() -> dict:
    """Read the computer. Never raises; unknown things are left out."""
    total, free = _memory()
    machine = {"os": {"win32": "Windows", "darwin": "macOS"}.get(sys.platform, "Linux" if sys.platform.startswith("linux") else sys.platform), "cores": os.cpu_count() or 1, "ram_gb": total, "free_gb": free}
    machine.update(_gpu())
    machine["cycles"] = _cycles_devices()
    return machine


def describe(machine: dict, turbo: str = "") -> list[str]:
    """The machine in plain words, one short line each."""
    lines = [f"{machine['os']}, {machine['cores']} cores"]
    if machine.get("ram_gb"):
        lines.append(f"Memory: {machine['ram_gb']:.0f} GB, {machine['free_gb']:.0f} GB free")
    if machine.get("gpu"):
        lines.append(f"Graphics: {machine['gpu']}, {machine['vram_gb']:.0f} GB, CUDA available")
    elif machine.get("cycles"):
        lines.append("Graphics: " + ", ".join(machine["cycles"][:2]))
    else:
        lines.append("Graphics: no supported card found")
    if turbo:
        lines.append(turbo)
    return lines


def benchmark() -> float:
    """Camera rays per second on this computer, from a 0.1 s test on a small sphere. 0 outside Blender."""
    try:
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
    except Exception:
        return 0.0


def peak_memory_mb() -> float:
    """The most memory this process has used so far, in MB (0 where it can't be read)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD), ("peak", ctypes.c_size_t)] + [(n, ctypes.c_size_t) for n in ("ws", "pp", "ppp", "np", "npp", "pf", "ppf")]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return counters.peak / 2**20
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
    threads = base.threads if key == "LIGHT" else (max(machine["cores"] // 2, 1) if key == "NORMAL" else 0)
    chunk = base.chunk_faces
    if machine.get("free_gb"):
        chunk = min(chunk, max(int(machine["free_gb"] * 2**30 * 0.2 / BYTES_PER_FACE), 20_000))
    return Settings(base.name, base.slice_s, base.gap_s, base.max_rays, base.max_pixels, chunk, threads, base.show_every, base.only_done)


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


SLOW_SPREAD = 4.0  # timed runs took 1.3x to 4x the estimate with Auto; Clean Rows, Field Rows and QuadriFlow far longer
SLOW_METHODS_SPREAD = 20.0


def span(seconds: float, slow: bool = False) -> str:
    """The estimate as a range: from it to a few times it (many times with the slow methods)."""
    low, high = clock(seconds), clock(seconds * (SLOW_METHODS_SPREAD if slow else SLOW_SPREAD))
    return low if low == high else f"{low} to {high.removeprefix('about ')}"


def rate(machine: dict, est: dict, settings: Settings, slow: bool = False) -> tuple[str, str]:
    """(level, sentence) for the panel. Level is OK, LARGE or TOO_BIG. `slow`: Clean Rows, Field Rows or QuadriFlow."""
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


def weld(points: np.ndarray, faces: list) -> tuple[np.ndarray, list]:
    """Merge vertices at exactly the same place (the chunk borders, which were kept unmoved)."""
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return unique, [tuple(int(inverse[v]) for v in f) for f in faces]


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


def checkpoint_dir(co: np.ndarray, face_count: int, root: str | None = None) -> Path | None:
    """The folder that holds the saved parts of the job on this very mesh: private to this user, or None if none is safe."""
    if root:
        base = Path(root)
    else:
        try:
            import bpy

            base = Path(bpy.utils.user_resource("CONFIG"))
        except Exception:
            base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
    digest = hashlib.sha1(np.ascontiguousarray(co).tobytes() + str(face_count).encode()).hexdigest()[:16]
    folder = base / "curiomesh_resume"
    if not (_private_dir(folder) and _private_dir(folder / digest)):
        return None  # no checkpoints rather than files in a place someone else could plant
    return folder / digest


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
                ends = np.cumsum(lens)
                polys = [tuple(int(v) for v in flat[a:b]) for a, b in zip(ends - lens, ends)]
                sizes.append((data[f"p{k}"], polys, [str(int(m)) for m in data[f"m{k}"]]))
            return float(data["detail"]), sizes, [(int(f), float(w)) for f, w in data["levels"]], float(data["seen"])
    except Exception:
        return None
