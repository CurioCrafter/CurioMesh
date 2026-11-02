# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Clean-row quads from the separate QuadWild-BiMDF programs, told which creases to keep.

QuadWild gives flowing rows but rounds what it isn't told about. Handing it the source's long creases, including
soft and concave ones around eyes and lips, as feature lines keeps them and costs few irregular vertices. Short
creases are left out: each one breaks the rows around it. Then fit.py pulls the vertices back to the source.

The programs are not part of this add-on. Point `folder` at an unpacked release; if they are missing or fail,
flow_remesh returns None and the caller carries on with its other candidates.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from triadq.fit import fit_vertices

CREASE_DEG = 20.0  # faces this far apart may be a feature line. Lower adds noise, higher misses soft forms
CREASE_LENGTH = 0.1  # share of the object's size a line must reach. Shorter ones break rows for little detail
TRIES = 3
CLOSE_ENOUGH = 0.06  # share of the asked face count
TIMEOUT = 1800.0
FIT_ROUNDS = 3
NORMAL_ROUNDS = 12
NORMAL_STEP = 0.5


def find_tool(folder) -> Path | None:
    """The folder if it holds both programs and their configs, else None."""

    if not folder:
        return None
    path = Path(folder)
    for suffix in ("", ".exe"):
        names = (path / f"quadwild{suffix}", path / f"quad_from_patches{suffix}")
        if all(name.is_file() for name in names) and (path / "config" / "main_config" / "flow.txt").is_file():
            return path
    return None


def creases(points: np.ndarray, tris: np.ndarray, degrees: float = CREASE_DEG, length_share: float = CREASE_LENGTH) -> list[tuple[int, int, int]]:
    """Each face side on a long crease as (kind, triangle, side), as QuadWild's .sharp lists them. Kind 0 is convex."""

    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    normal = np.cross(points[tris[:, 1]] - points[tris[:, 0]], points[tris[:, 2]] - points[tris[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    sides = defaultdict(list)
    for t, (a, b, c) in enumerate(tris.tolist()):
        for s, (u, v) in enumerate(((a, b), (b, c), (c, a))):
            sides[(u, v) if u < v else (v, u)].append((t, s))
    limit = math.cos(math.radians(degrees))
    sharp = {}
    for edge, found in sides.items():
        if len(found) == 2 and normal[found[0][0]] @ normal[found[1][0]] <= limit:
            third = tris[found[1][0]][(found[1][1] + 2) % 3]
            sharp[edge] = 0 if (points[third] - points[edge[0]]) @ normal[found[0][0]] < 0.0 else 1
    at = defaultdict(list)
    for edge in sharp:
        at[edge[0]].append(edge)
        at[edge[1]].append(edge)
    size = float(np.ptp(points, axis=0).max())
    done = set()
    out = []
    for start in sharp:
        if start in done:
            continue
        done.add(start)
        line = [start]
        for end in start:  # walk both ways, stopping where lines cross
            v, edge = end, start
            while len(at[v]) == 2:
                following = at[v][0] if at[v][1] == edge else at[v][1]
                if following in done:
                    break
                done.add(following)
                line.append(following)
                v = following[0] if following[1] == v else following[1]
                edge = following
        if sum(np.linalg.norm(points[a] - points[b]) for a, b in line) >= length_share * size:
            out += [(sharp[edge], t, s) for edge in line for t, s in sides[edge]]
    return out


def _config(source: Path, target: Path, **values) -> Path:
    text = source.read_text()
    for key, value in values.items():
        text = re.sub(rf"^{key} .*$", f"{key} {value}", text, flags=re.M)
    target.write_text(text)
    return target


def _read_quads(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    with open(path) as lines:
        for line in lines:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])
    if not faces or any(len(f) != 4 for f in faces):
        raise ValueError("not all quads")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def run_once(tool: Path, points, tris, lines, scale: float, remesh: bool = True, timeout: float = TIMEOUT) -> tuple[np.ndarray, np.ndarray] | None:
    """(points, quads), or None if the programs made nothing usable.

    Their return codes mean nothing, since quad_from_patches crashes on exit after writing, so the files decide.
    They must run from their own folder to find their configs.
    """

    with tempfile.TemporaryDirectory(prefix="curiomesh_flow", ignore_cleanup_errors=True) as folder:
        work = Path(folder)
        mesh = work / "in.obj"
        with open(mesh, "w") as out:
            out.write("".join(f"v {x!r} {y!r} {z!r}\n" for x, y, z in np.asarray(points, dtype=np.float64).tolist()))
            out.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in np.asarray(tris).tolist()))
        sharp = work / "in.sharp"
        sharp.write_text(f"{len(lines)}\n" + "".join(f"{k},{t},{s}\n" for k, t, s in lines))
        prep = _config(tool / "config" / "prep_config" / "basic_setup_Organic.txt", work / "prep.txt", do_remesh=int(remesh), sharp_feature_thr=-1, scaleFact=scale)
        main = _config(tool / "config" / "main_config" / "flow.txt", work / "main.txt", scaleFact=scale)
        suffix = ".exe" if (tool / "quadwild.exe").is_file() else ""
        try:
            subprocess.run([str(tool / f"quadwild{suffix}"), str(mesh), "2", str(prep), str(sharp)], cwd=tool, timeout=timeout, capture_output=True, check=False)
            patches = next(work.glob("*_p0.obj"), None)
            if patches is None:
                return None
            subprocess.run([str(tool / f"quad_from_patches{suffix}"), str(patches), "1", str(main)], cwd=tool, timeout=timeout, capture_output=True, check=False)
            result = next(work.glob("*quadrangulation_smooth.obj"), None)
            return _read_quads(result) if result else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None


def fit_normals(points, quads, sample_pos, sample_nrm, sample_area, nearest) -> np.ndarray:
    """One step turning each face toward the mean source normal under it (the vertex update of Sun et al. 2007).

    The camera mostly reads a wrong surface direction, not a wrong place, and the plane fit doesn't aim at that.
    """

    corners = points[quads]
    middle = corners.mean(axis=1)
    own = np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
    spots = np.concatenate([middle] + [(middle + corners[:, k]) / 2.0 for k in range(4)])
    face = nearest(spots, sample_pos) % len(quads)
    facing = (sample_nrm * own[face]).sum(axis=1) > 0.0  # the far side of a thin wall has no say
    wanted = np.zeros_like(middle)
    np.add.at(wanted, face[facing], sample_area[facing][:, None] * sample_nrm[facing])
    length = np.linalg.norm(wanted, axis=1, keepdims=True)
    wanted = np.where(length > 0.0, wanted / np.maximum(length, 1e-30), own / np.maximum(np.linalg.norm(own, axis=1, keepdims=True), 1e-30))
    move = np.zeros_like(points)
    count = np.zeros(len(points))
    for k in range(4):
        np.add.at(move, quads[:, k], wanted * (wanted * (middle - corners[:, k])).sum(axis=1, keepdims=True))
        np.add.at(count, quads[:, k], 1.0)
    return points + NORMAL_STEP * move / np.maximum(count, 1.0)[:, None]


def flow_remesh(folder, points, tris, target: int, sample_pos, sample_nrm, sample_area, nearest, remesh: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    """(points, quads) near `target` faces and fitted to the source, or None if the tool is missing or failed.

    `nearest(points, sample_pos)` gives each sample's nearest point, so Blender and tests can each bring their own tree.
    """

    tool = find_tool(folder)
    if tool is None:
        return None
    lines = creases(points, tris)
    scale = 0.8 * math.sqrt(len(tris) / max(target, 1))
    best = None
    for _ in range(TRIES):
        found = run_once(tool, points, tris, lines, round(scale, 3), remesh)
        if found is None:
            break
        if best is None or abs(len(found[1]) - target) < abs(len(best[1]) - target):
            best = found
        if abs(len(found[1]) - target) <= CLOSE_ENOUGH * target:
            break
        scale *= (len(found[1]) / target) ** 0.45  # features put a floor under the count, so don't overshoot
    if best is None:
        return None
    where, quads = best
    for _ in range(FIT_ROUNDS):
        where = fit_vertices(where, quads, sample_pos, sample_nrm, sample_area, nearest(where, sample_pos))
    for round_index in range(NORMAL_ROUNDS):
        where = fit_normals(where, quads, sample_pos, sample_nrm, sample_area, nearest)
        if round_index % 3 == 2:
            where = fit_vertices(where, quads, sample_pos, sample_nrm, sample_area, nearest(where, sample_pos))
    return where, quads
