# SPDX-License-Identifier: GPL-3.0-or-later
"""CurioMesh Turbo: optional native (Rust) kernels with pure-Python fallbacks.

The compiled module ships as ``_native/<platform tag>/_curiomesh_turbo.*`` next
to this file, one stable-ABI (abi3, Python >= 3.10) binary per OS/arch, so a
single add-on zip serves every Blender 3.6 - 5.x on every platform. Loading
never raises: a missing, blocked or broken binary is logged once and every
wrapper returns ``None``, which tells the caller to run its Python path.

Set ``CURIOMESH_TURBO=0`` (or call ``set_enabled(False)``) to force Python.
The graphics card draws the camera pictures of big meshes (``pictures``); ``CURIOMESH_GPU=0`` or ``set_gpu(False)``
keeps them on the processor, and so does any card error.
Wrappers take and return exactly what their Python counterparts do.
"""

from __future__ import annotations

import gc
import importlib.machinery
import importlib.util
from itertools import chain
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np


_log = logging.getLogger(__name__)
MODULE_NAME = "_curiomesh_turbo"
NATIVE_DIR = Path(__file__).resolve().parent / "_native"

_module: Any = None
_tried = False
_error = ""
_enabled = True
_gpu_wanted = True
_gpu_broken = ""  # why the card was given up on, once it was
_gpu_unsaid = False  # a reason nobody has been told yet
_gpu_drawn = 0  # pictures the card has drawn so far
GPU_MIN_TRIANGLES = 150_000  # measured on an RTX 3080 Ti: below this the card is level with the processor or slower


def platform_tag() -> str:
    """Folder name for this interpreter: win_amd64, linux_x86_64, macos_arm64, ..."""

    if sys.platform == "win32":  # platform.machine() queries WMI on Python 3.12+ (~1 s)
        machine = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "")
    else:
        machine = os.uname().machine
    machine = machine.lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    if sys.platform == "win32":
        return {"x86_64": "win_amd64", "arm64": "win_arm64"}.get(machine, f"win_{machine}")
    if sys.platform == "darwin":
        return f"macos_{machine}"
    return f"linux_{'aarch64' if machine == 'arm64' else machine}"


def _load() -> Any:
    global _module, _tried, _error
    if _tried:
        return _module
    _tried = True
    folder = NATIVE_DIR / platform_tag()
    try:
        candidates = sorted(p for p in folder.glob(f"{MODULE_NAME}*") if p.suffix in {".pyd", ".so"})
        if not candidates:
            raise ImportError(f"no native build for {platform_tag()} in {folder}")
        # The add-on can be loaded under several package names; reuse one copy.
        module = sys.modules.get(MODULE_NAME)
        if module is None:
            loader = importlib.machinery.ExtensionFileLoader(MODULE_NAME, str(candidates[0]))
            spec = importlib.util.spec_from_file_location(MODULE_NAME, candidates[0], loader=loader)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            sys.modules[MODULE_NAME] = module
        _module = module
    except Exception as exc:  # ImportError, OSError (blocked/quarantined DLL), anything else
        _error = f"{type(exc).__name__}: {exc}"
        _log.info("CurioMesh Turbo unavailable, using Python: %s", _error)
    return _module


def set_enabled(flag: bool) -> None:
    """Preference hook: False forces the Python paths (same as CURIOMESH_TURBO=0)."""

    global _enabled
    _enabled = bool(flag)


def set_gpu(flag: bool) -> None:
    """Preference hook: False keeps the pictures on the processor."""

    global _gpu_wanted
    _gpu_wanted = bool(flag)


def _gpu_allowed() -> bool:
    return _gpu_wanted and not _gpu_broken and os.environ.get("CURIOMESH_GPU", "1").strip().lower() not in {"0", "false", "off", "no"}


def gpu_name() -> str | None:
    """The card that would draw pictures ("NVIDIA GeForce RTX 3080 Ti (Vulkan)"), or None. The first call opens it (about a second)."""

    module = native()
    if module is None or not hasattr(module, "gpu_info") or not _gpu_allowed():
        return None
    try:
        ok, text = module.gpu_info()
    except Exception as exc:
        ok, text = False, str(exc)
    if not ok:
        _give_up_on_gpu(text, say=False)  # no card is not news: only a card that fails mid-job is
        return None
    return str(text)


def _give_up_on_gpu(why: str, say: bool = True) -> None:
    global _gpu_broken, _gpu_unsaid
    if not _gpu_broken:
        _gpu_broken, _gpu_unsaid = why or "unknown error", say
        _log.info("CurioMesh graphics card not used: %s", _gpu_broken)


def gpu_drawn() -> int:
    return _gpu_drawn


def gpu_note() -> str | None:
    """One sentence, once, saying the card failed and the processor did the work; None when there is nothing new to say."""

    global _gpu_unsaid
    if not _gpu_unsaid:
        return None
    _gpu_unsaid = False
    return f"The graphics card could not draw the camera pictures ({_gpu_broken[:80]}): the processor did"


def native() -> Any:
    """The native module, or None when disabled or not loadable."""

    if not _enabled or os.environ.get("CURIOMESH_TURBO", "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    return _load()


def limit_threads(count: int) -> bool:
    """Ask Turbo to use `count` threads (0 = all). Only works before the native module first loads; True if it took."""

    if _tried and _module is not None:
        return False
    if count > 0:
        os.environ["RAYON_NUM_THREADS"] = str(count)
    else:
        os.environ.pop("RAYON_NUM_THREADS", None)
    return True


def available() -> bool:
    return native() is not None


def thread_count() -> int:
    """How many threads the native core may use: 1 without it, or on Light."""

    module = native()
    return 1 if module is None else int(module.threads)


def version() -> str | None:
    module = native()
    return None if module is None else str(module.__version__)


def status() -> str:
    """One line for the UI / logs."""

    module = native()
    if module is not None:
        return f"Turbo {module.__version__} ({platform_tag()}, {module.threads} threads)"
    if not _enabled or os.environ.get("CURIOMESH_TURBO", "1").strip().lower() in {"0", "false", "off", "no"}:
        return "Turbo disabled (Python)"
    return f"Turbo unavailable (Python): {_error}"


# --------------------------------------------------------------------------- helpers


def _topology():
    try:
        from .triadq import topology
    except ImportError:
        from triadq import topology
    return topology


def _csr(faces) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(faces, "indptr"):  # types.FlatFaces
        return faces.indptr, faces.flat
    sizes = np.fromiter(map(len, faces), dtype=np.int64, count=len(faces))
    indptr = np.zeros(len(faces) + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    flat = np.fromiter(chain.from_iterable(faces), dtype=np.int64, count=int(indptr[-1]))
    return indptr, flat


def _edge_array(edges) -> np.ndarray:
    return np.asarray(sorted(tuple(e) for e in edges), dtype=np.int64).reshape(-1)


# --------------------------------------------------------------------------- topology


def topology_counts(faces) -> dict | None:
    """Combinatorial half of `topology_summary` as a dict (no geometry)."""

    module = native()
    if module is None or not faces:
        return None
    indptr, flat = _csr(faces)
    if flat.min() < 0:
        return None
    return module.topology_counts(indptr, flat)


def topology_summary(mesh):
    """Native `triadq.topology.topology_summary`; None means "use Python"."""

    module = native()
    if module is None or not mesh.faces:
        return None
    T = _topology()
    indptr, flat = _csr(mesh.faces)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    if flat.min() < 0 or flat.max() >= len(vertices):
        return None  # let Python produce its own error / semantics
    counts = module.topology_counts(indptr, flat)
    epsilon = T._span_epsilon(vertices)
    degenerate = module.degenerate_faces(indptr, flat, vertices.reshape(-1), epsilon)
    sizes = np.diff(indptr)
    quad_starts = indptr[:-1][sizes == 4]
    invalid = 0
    if len(quad_starts):
        quads = flat[quad_starts[:, None] + np.arange(4)]
        invalid = int(np.count_nonzero(~T._quads_valid(vertices, quads, epsilon)))
    detail = tuple(tuple(item) for item in counts["component_detail"])
    signature = tuple(sorted((item[0], item[1]) for item in detail))
    c = counts
    return T.TopologySummary(
        used_vertices=c["used_vertices"],
        edges=c["edges"],
        faces=c["faces"],
        quads=c["quads"],
        boundary_edges=c["boundary_edges"],
        boundary_components=c["boundary_components"],
        connected_components=len(detail),
        component_euler=tuple(item[0] for item in signature),
        component_signature=signature,
        component_detail=detail,
        non_manifold_edges=c["non_manifold_edges"],
        non_manifold_vertices=c["non_manifold_vertices"],
        branched_boundary_vertices=c["branched_boundary_vertices"],
        inconsistent_edges=c["inconsistent_edges"],
        duplicate_faces=c["duplicate_faces"],
        euler_characteristic=c["euler_characteristic"],
        extraordinary_vertices=c["extraordinary_vertices"],
        defect_charge=c["defect_charge"],
        degenerate_faces=degenerate,
        invalid_quads=invalid,
        all_quads=(
            c["quads"] == c["faces"]
            and degenerate == 0
            and c["non_manifold_edges"] == 0
            and c["non_manifold_vertices"] == 0
            and c["branched_boundary_vertices"] == 0
            and c["inconsistent_edges"] == 0
            and c["duplicate_faces"] == 0
            and invalid == 0
        ),
    )


def non_manifold_vertex_count(faces) -> int | None:
    """`topology._non_manifold_vertex_count` (the occurrence table is rebuilt natively)."""

    counts = topology_counts(faces)
    return None if counts is None else counts["non_manifold_vertices"]


def edge_occurrences(faces) -> dict | None:
    """`topology._edge_occurrences`: same keys, insertion order and (face, a, b) lists."""

    module = native()
    if module is None:
        return None
    indptr, flat = _csr(faces)
    edges, start, face, a, b = module.edge_occurrences(indptr, flat)
    # Millions of small tuples would trigger repeated cyclic-GC passes; none can form cycles.
    paused = gc.isenabled()
    gc.disable()
    try:
        uses = list(zip(face.tolist(), a.tolist(), b.tolist()))
        bounds = start.tolist()
        keys = map(tuple, edges.reshape(-1, 2).tolist())
        return {key: uses[bounds[k] : bounds[k + 1]] for k, key in enumerate(keys)}
    finally:
        if paused:
            gc.enable()


def quad_only_closure(mesh, *, name: str | None = None, frozen_midpoints: dict | None = None):
    """Native `topology.quad_only_closure` (same output, same ValueErrors)."""

    module = native()
    if module is None:
        return None
    T = _topology()
    indptr, flat = _csr(mesh.faces)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    frozen = np.asarray(
        [(int(a), int(b), int(m)) for (a, b), m in (frozen_midpoints or {}).items()], dtype=np.int64
    ).reshape(-1)
    uv_args = {}
    if mesh.uvs is not None:
        uv_args["uvs"] = np.ascontiguousarray(mesh.uvs, dtype=np.float64).reshape(-1)
        if mesh.face_uvs is not None:
            uv_args["uv_indptr"], uv_args["uv_flat"] = _csr(mesh.face_uvs)
    out_vertices, quads, uvs, face_uvs = module.quad_closure(indptr, flat, vertices.reshape(-1), frozen, **uv_args)
    sizes = np.diff(indptr).tolist()
    materials = mesh.face_materials
    # MeshData would re-tuple millions of faces in __post_init__; the lists below
    # already have its normalized form, so they are assigned after construction.
    result = T.MeshData(
        vertices=out_vertices.reshape(-1, 3),
        faces=[],
        uvs=None if uvs is None else uvs.reshape(-1, 2),
        name=name or f"{mesh.name}_quad_closure",
    )
    result.faces = list(map(tuple, quads.reshape(-1, 4).tolist()))
    result.face_materials = [
        materials[index] if index < len(materials) else "default"
        for index, size in enumerate(sizes)
        for _ in range(size)
    ]
    result.face_uvs = None if face_uvs is None else list(map(tuple, face_uvs.reshape(-1, 4).tolist()))
    summary = topology_summary(result) or T.topology_summary(result)
    if not summary.all_quads:
        raise ValueError(
            "Quad closure failed validation: "
            f"quads={summary.quads}/{summary.faces}, degenerate={summary.degenerate_faces}, "
            f"invalid_quads={summary.invalid_quads}, non_manifold_edges={summary.non_manifold_edges}, "
            f"non_manifold_vertices={summary.non_manifold_vertices}, "
            f"branched_boundary_vertices={summary.branched_boundary_vertices}, "
            f"inconsistent_edges={summary.inconsistent_edges}, duplicate_faces={summary.duplicate_faces}."
        )
    return result


def parity_path_edges(mesh, frozen=frozenset()) -> set | None:
    """Native `topology._parity_path_edges` (same set, same ValueErrors)."""

    module = native()
    if module is None or not mesh.faces:
        return None
    indptr, flat = _csr(mesh.faces)
    edges = module.parity_path_edges(indptr, flat, _edge_array(frozen))
    return set(map(tuple, edges.reshape(-1, 2).tolist()))


def quad_points_valid(points, epsilon, warp_cosine: float) -> np.ndarray | None:
    """Native `topology._quad_points_valid` for (M, 4, 3) quads, one tolerance per quad."""

    module = native()
    if module is None or not hasattr(module, "quad_points_valid"):
        return None
    return module.quad_points_valid(np.ascontiguousarray(points, dtype=np.float64).reshape(-1), np.ascontiguousarray(epsilon, dtype=np.float64), float(warp_cosine))


def parity_template_validity(faces, vertices, base, selected_edges, midpoints, warp_cosine: float) -> list | None:
    """Native `topology._parity_template_validity`: per face, the validity of each template in the order they're tried."""

    module = native()
    if module is None or not hasattr(module, "parity_templates"):
        return None
    indptr, flat = _csr(faces)
    order = sorted(selected_edges)
    edges = np.asarray(order, dtype=np.int64).reshape(-1)
    mids = np.asarray([midpoints[edge] for edge in order], dtype=np.int64)
    counts, valid = module.parity_templates(
        indptr, flat, edges, mids,
        np.ascontiguousarray(vertices, dtype=np.float64).reshape(-1), np.ascontiguousarray(base, dtype=np.float64).reshape(-1), float(warp_cosine),
    )
    valid = valid.tolist()
    ends = np.cumsum(counts).tolist()
    return [valid[end - count : end] for count, end in zip(counts.tolist(), ends)]


def _flat_mesh(mesh):
    """(indptr, flat, vertices) of a mesh the native closure can take, or None for one with NaNs: Python judges those."""

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    if not np.isfinite(vertices).all():
        return None
    indptr, flat = _csr(mesh.faces)
    if len(flat) and (flat.min() < 0 or flat.max() >= len(vertices)):
        return None
    return indptr, flat, vertices.reshape(-1)


def _built(mesh, vertices, indptr, flat, owner):
    """MeshData with `mesh`'s name and each face's material from the input face `owner` names (no re-tupling of the faces)."""

    T = _topology()
    result = T.MeshData(vertices=vertices.reshape(-1, 3), faces=[], name=mesh.name)
    result.faces = T.FlatFaces(indptr, flat)
    materials = mesh.face_materials
    result.face_materials = list(map(materials.__getitem__, owner.tolist()))
    return result


def make_closable(mesh):
    """Native `flatquad.make_closable`: (mesh, vertices added), or None for "use Python"."""

    module = native()
    if module is None or not mesh.faces or not hasattr(module, "make_closable"):
        return None
    parts = _flat_mesh(mesh)
    if parts is None:
        return None
    span = float(np.linalg.norm(np.ptp(parts[2].reshape(-1, 3), axis=0))) if len(parts[2]) else 1.0
    vertices, indptr, flat, kept, added = module.make_closable(*parts, span)
    return _built(mesh, vertices, indptr, flat, kept), int(added)


def close_pieces(paired, warp_cosine: float):
    """Native `flatquad._close_pieces`, pieces side by side: (mesh, four counts), or None for "use Python" (and its errors)."""

    module = native()
    if module is None or not paired.faces or not hasattr(module, "close_pieces"):
        return None
    parts = _flat_mesh(paired)
    if parts is None:
        return None
    try:
        vertices, indptr, flat, owner, counts = module.close_pieces(*parts, float(warp_cosine))
    except InterruptedError:  # the job was stopped
        raise _cancelled() from None
    except ValueError:
        return None
    return _built(paired, vertices, indptr, flat, owner), tuple(int(c) for c in counts)


def join_loose_triangles(mesh, farthest: int):
    """Native `quadreduce.join_loose_triangles`: (mesh, triangles joined), or None for "use Python"."""

    module = native()
    if module is None or not mesh.faces or not hasattr(module, "join_loose_triangles"):
        return None
    parts = _flat_mesh(mesh)
    if parts is None:
        return None
    codes: dict = {}
    material = np.fromiter((codes.setdefault(m, len(codes)) for m in mesh.face_materials), dtype=np.int64, count=len(mesh.faces))
    vertices, indptr, flat, owner, joined = module.join_loose_triangles(*parts, material, int(farthest))
    return _built(mesh, vertices, indptr, flat, owner), int(joined)


class QuadReducer:
    """Native `quadreduce.QuadReducer`: the same methods, and `remove` runs off the main thread."""

    def __init__(self, module, mesh, parts, material, dissolve_limit: float, most_turned: float):
        self._core = module.QuadReducer(*parts, material, float(dissolve_limit), float(most_turned))
        self._mesh = mesh

    @property
    def alive(self) -> int:
        return self._core.alive

    @property
    def refused(self) -> int:
        return self._core.refused

    def remove(self, count: int, every: int = 64):
        """Generator: take away up to `count` quads, cheapest first, yielding so a timer can keep Blender live."""

        thread = threading.Thread(target=self._core.remove, args=(int(count),), daemon=True, name="CurioMeshReduce")
        thread.start()
        while thread.is_alive():
            thread.join(0.02)
            yield

    def retry_refused(self) -> None:
        self._core.retry_refused()

    def original_ids(self, vertices) -> list:
        return self._core.original_ids(np.asarray(vertices, dtype=np.int64).reshape(-1)).tolist()

    def freeze(self, vertices, renamed: bool = False) -> None:
        ids = vertices if renamed else self.original_ids(vertices)
        self._core.freeze(np.asarray(ids, dtype=np.int64).reshape(-1))

    def save(self):
        return self._core.save()

    def restore(self, saved) -> None:
        self._core.restore(saved)

    def result(self):
        vertices, indptr, flat, owner = self._core.result()
        return _built(self._mesh, vertices, indptr, flat, owner)


def quad_reducer(mesh, dissolve_limit: float, most_turned: float):
    """A native `QuadReducer`, or None for "use Python"."""

    module = native()
    if module is None or not mesh.faces or not hasattr(module, "QuadReducer"):
        return None
    parts = _flat_mesh(mesh)
    if parts is None or int(np.diff(parts[0]).min()) < 3:
        return None
    codes: dict = {}
    material = np.fromiter((codes.setdefault(m, len(codes)) for m in mesh.face_materials), dtype=np.int64, count=len(mesh.faces))
    return QuadReducer(module, mesh, parts, material, dissolve_limit, most_turned)


# --------------------------------------------------------------------------- matching


def pair_seeds(face_count: int, face_a, face_b, quads, scores, minimum, seeds) -> list | None:
    """Native `engine._pair_triangles` for each seed, side by side: rows of (quads (M, 4), leftover triangles, material face per face, rescued).

    `minimum` is the score a candidate must beat, None to take them all. None for NaN scores: Python's sort decides those.
    """

    module = native()
    if module is None or not hasattr(module, "pair_seeds") or np.isnan(scores).any():
        return None
    rows = _uninterrupted(
        module.pair_seeds,
        int(face_count),
        np.ascontiguousarray(face_a, dtype=np.int64),
        np.ascontiguousarray(face_b, dtype=np.int64),
        np.ascontiguousarray(quads, dtype=np.int64).reshape(-1),
        np.ascontiguousarray(scores, dtype=np.float64),
        minimum,
        [int(seed) for seed in seeds],
    )
    return [(found.reshape(-1, 4), tris, owner, int(rescued)) for found, tris, owner, rescued in rows]


def augment_matching(mate: dict, allowed: dict, order: list) -> int | None:
    """Native `engine._augment_matching`: grows `mate` in place, returns the rescued count."""

    module = native()
    if module is None:
        return None
    ids = list(chain(allowed, order, mate))
    for linked in allowed.values():
        ids.extend(item[1] for item in linked)
    count = max(ids, default=-1) + 1
    if not count or min(ids) < 0:
        return 0 if not count else None
    sizes = np.zeros(count, dtype=np.int64)
    for face, linked in allowed.items():
        sizes[face] = len(linked)
    indptr = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    for face, linked in allowed.items():
        indices[indptr[face] : indptr[face + 1]] = [item[1] for item in linked]
    mate_array = np.full(count, -1, dtype=np.int64)
    if mate:
        mate_array[np.fromiter(mate.keys(), np.int64)] = np.fromiter(mate.values(), np.int64)
    before = mate_array.copy()
    rescued = _uninterrupted(module.augment_matching, indptr, indices, np.asarray(order, dtype=np.int64), mate_array)
    for face in np.flatnonzero(mate_array != before).tolist():
        mate[face] = int(mate_array[face])
    return int(rescued)


# --------------------------------------------------------------------------- lookups


_stop: Any = None


def stop_native() -> None:
    """Calls that take the stop flag and are running now end early, with `Cancelled`. Later calls are not affected."""

    global _stop
    if _stop is not None:
        _stop.set()
        _stop = _module.StopFlag()
    if _module is not None and hasattr(_module, "stop_running"):
        _module.stop_running()


def _cancelled():
    try:
        from .triadq.types import Cancelled
    except ImportError:
        from triadq.types import Cancelled
    return Cancelled()


def _uninterrupted(call, *args):
    """`call(*args)` for the kernels a stop ends by itself (see `stop_native`), turning that into `Cancelled`."""

    try:
        return call(*args)
    except InterruptedError:
        raise _cancelled() from None


def _stoppable(call, *args):
    """`call(*args, stop flag)`, turning a stop into the `Cancelled` the jobs already handle."""

    global _stop
    if _stop is None:
        _stop = _module.StopFlag()
    try:
        return call(*args, _stop)
    except InterruptedError:
        raise _cancelled() from None


def _query(points) -> np.ndarray:
    return np.ascontiguousarray(points, dtype=np.float64).reshape(-1)


class PointTree:
    """Nearest points for a whole array at once, on all cores. Exact ties go to the lowest index."""

    def __init__(self, module, points):
        self._core = module.PointLookup(_query(points))

    def nearest(self, queries) -> tuple[np.ndarray, np.ndarray]:
        """(index, distance) of the nearest point to each query."""

        return _stoppable(self._core.nearest, _query(queries))

    def first_of_kind(self, queries, query_kind, point_kind, count: int) -> np.ndarray:
        """For each query, the closest of its `count` nearest points of the query's own kind, or -1."""

        return _stoppable(self._core.first_of_kind, _query(queries), np.ascontiguousarray(query_kind, dtype=np.int64), np.ascontiguousarray(point_kind, dtype=np.int64), int(count))


class FaceTree:
    """Blender's BVHTree.FromPolygons for arrays of queries, on all cores: polygons cut the same way, single precision."""

    def __init__(self, module, parts):
        self._core = module.FaceLookup(*parts)

    def nearest(self, queries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(closest point (N, 3), polygon, distance) for each query."""

        at, face, away = _stoppable(self._core.nearest, _query(queries))
        return at.reshape(-1, 3), face, away

    def crossing_pairs(self) -> np.ndarray:
        """Polygon pairs (M, 2), lower first and sorted, that pass through each other and share no vertex."""

        return _stoppable(self._core.crossing_pairs).reshape(-1, 2)


def point_tree(points) -> PointTree | None:
    """A `PointTree`, or None for "use Blender's KDTree"."""

    module = native()
    return None if module is None or not hasattr(module, "PointLookup") else PointTree(module, points)


def face_tree(points, faces) -> FaceTree | None:
    """A `FaceTree`, or None for "use Blender's BVHTree"."""

    module = native()
    if module is None or not hasattr(module, "FaceLookup"):
        return None
    indptr, flat = _csr(faces)
    vertices = _query(points)
    if len(flat) and (flat.min() < 0 or flat.max() * 3 >= len(vertices)):
        return None
    return FaceTree(module, (vertices, indptr, flat))


def shifted_faces(polys, offset: int) -> list:
    """`[tuple(v + offset for v in p) for p in polys]`, natively when it can be."""

    module = native()
    if module is None or not hasattr(module, "shifted_faces"):
        return [tuple(v + offset for v in p) for p in polys]
    if hasattr(polys, "indptr"):  # types.FlatFaces
        return module.shifted_faces(None, int(offset), polys.indptr, polys.flat)
    return module.shifted_faces(polys, int(offset))


def mesh_checks(faces, piece):
    """Native `checks.edge_flags` and `checks.hole_counts`: (broken per face, non-manifold edges, flipped edges, duplicate faces, holes, holes per piece), or None."""

    module = native()
    if module is None or not hasattr(module, "mesh_checks"):
        return None
    indptr, flat = _csr(faces)
    if len(flat) and flat.min() < 0:
        return None
    broken, bad_edges, flipped, dups, holes, pieces, counts = module.mesh_checks(indptr, flat, np.ascontiguousarray(piece, dtype=np.int64))
    return broken, bad_edges, flipped, dups, holes, dict(zip(pieces.tolist(), counts.tolist()))


# --------------------------------------------------------------------------- geometry


def reducer(points, polys, keep=None, least: int = 0):
    """A piece of triangles taken all the way down by the native reducer (`.level(faces)` gives any size), or None for "use Decimate"."""

    module = native()
    if module is None or not hasattr(module, "Reducer"):
        return None
    points = np.ascontiguousarray(points, dtype=np.float64)
    tris = _csr(polys)[1]
    locked = np.zeros(0 if keep is None else len(points), dtype=bool) if keep is None else np.ascontiguousarray(keep, dtype=bool)
    return module.Reducer(points.reshape(-1), tris, locked, int(least))


def winding_number(points, triangles) -> np.ndarray | None:
    """Native `patch.winding_number` (parallel over points)."""

    module = native()
    if module is None:
        return None
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.ascontiguousarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if not len(points) or not len(triangles):
        return np.zeros(len(points))
    return module.winding_number(points.reshape(-1), triangles.reshape(-1))


def inside_volume(points, triangles) -> np.ndarray | None:
    """Native `patch.inside_volume` (same bounding-box cull, native winding number)."""

    module = native()
    if module is None:
        return None
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    inside = np.zeros(len(points), dtype=bool)
    if not len(triangles):
        return inside
    low, high = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
    pad = 1e-9 * float(np.linalg.norm(high - low))
    near = np.all((points >= low - pad) & (points <= high + pad), axis=1)
    inside[near] = np.abs(winding_number(points[near], triangles)) > 0.5
    return inside


def pictures(points, polys, center, half: float, pixels: int) -> list | None:
    """Native `autoretopo.pictures`: the same six orthographic pictures from a depth buffer, no rays or BVH.

    None without the native module, or for a mesh with polygons over four sides (the rays handle those).
    A mesh of `GPU_MIN_TRIANGLES` or more is drawn on the graphics card when there is one (see `rust gpu.rs` for how
    its answer can differ); any card error falls back to the processor for good and sets `gpu_note`.
    """

    global _gpu_drawn
    module = native()
    if module is None or not hasattr(module, "pictures"):  # an older binary
        return None
    indptr, flat = _csr(polys)
    sizes = np.diff(indptr)
    if len(sizes) and int(sizes.max()) > 4:
        return None
    points = np.ascontiguousarray(points, dtype=np.float64)
    tris = flat[indptr[:-1][sizes == 3][:, None] + np.arange(3)]
    quads = flat[indptr[:-1][sizes == 4][:, None] + np.arange(4)]
    a, b, c = np.moveaxis(points[tris], 1, 0)
    normals = np.cross(b - a, c - a)
    if len(quads):
        # Blender's BVH cuts a quad's first corner off (the diagonal from corner 1 to 3), or, when corner 0 or 2 turns
        # inward seen along the quad's normal, the other diagonal. The same split keeps the two paths' pictures equal.
        a, b, c, d = np.moveaxis(points[quads], 1, 0)
        facing = np.cross(c - a, d - b)
        inward = ((np.cross(a - d, b - a) * facing).sum(axis=1) < 0.0) | ((np.cross(c - b, d - c) * facing).sum(axis=1) < 0.0)
        quads[inward] = quads[inward][:, [1, 2, 3, 0]]
        tris = np.vstack((tris, quads[:, [3, 0, 1]], quads[:, [1, 2, 3]]))
        # And like the BVH, both halves report the whole quad's normal, not their own: on a bent quad that differs a lot.
        normals = np.vstack((normals, facing, facing))
    args = (points.reshape(-1), tris.reshape(-1), normals.reshape(-1), [float(x) for x in center], float(half), int(pixels))
    out = None
    if len(tris) >= GPU_MIN_TRIANGLES and hasattr(module, "gpu_info") and _gpu_allowed():
        try:
            out = module.pictures(*args, True)
            _gpu_drawn += 1
        except Exception as exc:  # no card, a driver error, a mesh too big for its memory: the processor does it
            _give_up_on_gpu(str(exc))
    hit, normal, depth, tri = out if out is not None else module.pictures(*args)
    # Back from triangles to polygons, as the BVH reports them: the triangles came first, then both halves of each quad.
    four = np.flatnonzero(sizes == 4)
    owner = np.concatenate((np.flatnonzero(sizes == 3), four, four))
    face = owner[tri].astype(np.int32) if len(owner) else tri.astype(np.int32)
    shape = (6, pixels, pixels)
    hit, normal, depth, face = hit.reshape(shape), normal.reshape((*shape, 3)), depth.reshape(shape), face.reshape(shape)
    return [(hit[i], normal[i], depth[i], face[i]) for i in range(6)]


def leader_clusters(values, tol: float) -> np.ndarray | None:
    """Native `parts._cluster`."""

    module = native()
    if module is None or not hasattr(module, "leader_clusters"):
        return None
    return module.leader_clusters(np.ascontiguousarray(values, dtype=np.float64).reshape(-1), float(tol))


def cross_field_jacobi(cross, face_a, face_b, transport, pinned, degree, steps: int) -> np.ndarray | None:
    """The smoothing loop of `engine._estimate_field`; returns the relaxed complex field."""

    module = native()
    if module is None:
        return None
    out = np.array(cross, dtype=np.complex128)
    module.cross_field_jacobi(
        out,
        np.ascontiguousarray(face_a, dtype=np.int64),
        np.ascontiguousarray(face_b, dtype=np.int64),
        np.ascontiguousarray(transport, dtype=np.complex128),
        np.ascontiguousarray(pinned, dtype=bool),
        np.ascontiguousarray(degree, dtype=np.float64),
        int(steps),
    )
    return out


def cross_field_gl(
    cross, face_a, face_b, transport, pinned, denom, pin_term, steps: int, mu: float = 1.0, dt: float = 0.4
) -> np.ndarray | None:
    """The relaxation loop of `field.estimate_field_gl`; returns the relaxed complex field."""

    module = native()
    if module is None:
        return None
    out = np.array(cross, dtype=np.complex128)
    module.cross_field_gl(
        out,
        np.ascontiguousarray(face_a, dtype=np.int64),
        np.ascontiguousarray(face_b, dtype=np.int64),
        np.ascontiguousarray(transport, dtype=np.complex128),
        np.ascontiguousarray(pinned, dtype=bool),
        np.ascontiguousarray(denom, dtype=np.float64),
        np.ascontiguousarray(pin_term, dtype=np.complex128),
        float(mu),
        float(dt),
        int(steps),
    )
    return out


# main-only: uv begin common
class UvJob:
    """A long UV kernel running on a worker thread: `done()`, `result()`, `stop()`. Creating one never blocks the caller.

    `call(stop)` is the one-call kernel (it gets the StopFlag, or None). A stopped kernel ends with InterruptedError;
    `result()` then raises it too. Any other error is raised from `result()` as well.
    """

    def __init__(self, call, stop=None, name: str = "CurioMeshUV"):
        self._stop, self._out, self._error = stop, None, None
        self._thread = threading.Thread(target=self._run, args=(call,), daemon=True, name=name)
        self._thread.start()

    def _run(self, call) -> None:
        try:
            self._out = call(self._stop)
        except BaseException as exc:  # handed to result(); a worker must never die silently
            self._error = exc

    def done(self) -> bool:
        return not self._thread.is_alive()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()

    def result(self):
        if not self.done():
            raise RuntimeError("the kernel is still running")
        if self._error is not None:
            raise self._error
        return self._out


def uv_job(function: str, *args, name: str = "CurioMeshUV") -> "UvJob | None":
    """Start native `function(*args, stop)` on a worker thread, or None for "use the NumPy path"."""

    module = native()
    if module is None or not hasattr(module, function) or not hasattr(module, "StopFlag"):
        return None
    fn, stop = getattr(module, function), module.StopFlag()
    return UvJob(lambda flag: fn(*args, flag), stop=stop, name=name)


def _uv_probe(function: str) -> "int | None":
    module = native()
    if module is None or not hasattr(module, function):
        return None
    return int(getattr(module, function)())
# main-only: uv end common

# main-only: uv begin seam
def uvseam_probe() -> "int | None":
    """1 when the compiled `uvseam` kernel is there and its compile probe passes; None for "use NumPy"."""

    return _uv_probe("uvseam_probe")
# main-only: uv end seam

# main-only: uv begin snap
def uvsnap_probe() -> "int | None":
    """1 when the compiled `uvsnap` kernel is there and its compile probe passes; None for "use NumPy"."""

    return _uv_probe("uvsnap_probe")
# main-only: uv end snap

# main-only: uv begin qem
def uvqem_probe() -> "int | None":
    """1 when the compiled `uvqem` kernel is there and its compile probe passes; None for "use NumPy"."""

    return _uv_probe("uvqem_probe")
# main-only: uv end qem

# main-only: uv begin score
def uvscore_probe() -> "int | None":
    """1 when the compiled `uvscore` kernel is there and its compile probe passes; None for "use NumPy"."""

    return _uv_probe("uvscore_probe")
# main-only: uv end score

# main-only: uv begin sight
def uvsight_probe() -> "int | None":
    """1 when the compiled `uvsight` kernel is there and its compile probe passes; None for "use NumPy"."""

    return _uv_probe("uvsight_probe")
# main-only: uv end sight

# main-only: uv begin flat
def uvflat_probe() -> "int | None":
    """1 when the compiled `uvflat` kernel is there and its compile probe passes; None for "use NumPy"."""

    return _uv_probe("uvflat_probe")


UV_FACTOR_RAM_SHARE = 0.25  # one direct factor may use this share of the free memory; above it the iterative path runs


def _uv_free_ram() -> float:
    """Free physical memory in bytes, or 0 when unknown (the kernel then uses its own constant)."""

    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ("total", "avail", "page_total", "page_avail", "virt_total", "virt_avail", "ext_avail")
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.avail)
            return 0.0
        return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0.0


def _uv_factor_bytes(max_factor_bytes: float) -> float:
    return float(max_factor_bytes) if max_factor_bytes > 0 else UV_FACTOR_RAM_SHARE * _uv_free_ram()


def _uv_f64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())


def _uv_i64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.int64).ravel())


def uvflat_flatten(points, indptr, flat, chart, start, col, row, rows: int = 3, start_mode: int = 0, stretch_allowed: float = 0.10,
                   steps: int = 0, max_factor_bytes: float = 0.0, texture_px: float = 1024.0) -> "UvJob | None":
    """Flatten every chart in one worker-thread job (never blocks): `result()` = (UVs per corner (2C,), stats rows).

    chart per face (< 0 skips), start per corner (2C, NaN = none) or empty, col / row per corner (-1 = none) or empty.
    rows 0 Off / 1 Soft / 2 On / 3 Auto; start_mode 0 given / 1 pull / 2 free. None: use the NumPy twin.
    """

    return uv_job("uvflat_flatten", _uv_f64(points), _uv_i64(indptr), _uv_i64(flat), _uv_i64(chart), _uv_f64(start), _uv_i64(col), _uv_i64(row),
                  int(rows), int(start_mode), float(stretch_allowed), int(steps), _uv_factor_bytes(max_factor_bytes), float(texture_px),
                  name="CurioMeshUVFlatten")


def uvflat_grid_layout(points, indptr, flat) -> "UvJob | None":
    """Exact grid UVs of the given quad faces as a job: `result()` = (uv (2N,), col, row) or None when not a grid."""

    return uv_job("uvflat_grid_layout", _uv_f64(points), _uv_i64(indptr), _uv_i64(flat), name="CurioMeshUVGrid")


def uvflat_quick_stretch(points, indptr, flat, uv, free, max_factor_bytes: float = 0.0) -> "UvJob | None":
    """Band stretch of a merge candidate as a job: `result()` = (band stretch, chart stretch, uv (2N,))."""

    free = np.ascontiguousarray(np.asarray(free, dtype=bool).ravel())
    return uv_job("uvflat_quick_stretch", _uv_f64(points), _uv_i64(indptr), _uv_i64(flat), _uv_f64(uv), free, _uv_factor_bytes(max_factor_bytes), name="CurioMeshUVBand")


def uvflat_stretch(points, indptr, flat, uv) -> "UvJob | None":
    """Chart stretch as a job: `result()` = (stretch quantile, per face)."""

    return uv_job("uvflat_stretch", _uv_f64(points), _uv_i64(indptr), _uv_i64(flat), _uv_f64(uv), name="CurioMeshUVStretch")
# main-only: uv end flat

# main-only: uv begin chart
def uvchart_probe() -> "int | None":
    """1 when the compiled `uvchart` kernel is there and its compile probe passes; None for "use NumPy"."""

    return _uv_probe("uvchart_probe")
# main-only: uv end chart

# main-only: uv begin pack
def uvpack_probe() -> "int | None":
    """1 when the compiled `uvpack` kernel is there and its compile probe passes; None for "use NumPy"."""

    return _uv_probe("uvpack_probe")
# main-only: uv end pack
