# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""Read a part folder that comes back from outside (bench/tools_design/AGENT_TOOLS.md sections 3 and 4) and check it.

The files can come from anywhere (an agent's script, another 3D program), so every number is checked before a face is
built from it, in the same defensive style as `jobpack.load_result`. Plain NumPy, no `bpy`: the tests run outside
Blender. Every refusal is an exception with `agent_code`, `hint` and `details`, which the agent layer turns into a reply.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .triadq import segment as S
    from .triadq.patch import boundary_loops
except ImportError:  # loaded as a plain module (tests)
    from triadq import segment as S
    from triadq.patch import boundary_loops

FORMAT = "curiomesh-parts"
READS = (1, 0)  # the newest format version this reader knows
MADE_BY = "CurioMesh 0.9.9"
MAX_CORNERS = 64
TOLERANCE = 1e-6  # share of the object's size within which a returned border point counts as the stored one


class PartError(Exception):
    """A part file that cannot be used. `agent_code` names the kind (AGENT_TOOLS.md section 5)."""

    agent_code = "FILE"
    default_hint = "Check the files in the part folder; parts.json and README.txt say what each one holds"

    def __init__(self, message: str, hint: str = "", details: dict | None = None):
        super().__init__(message)
        self.hint = hint or self.default_hint
        self.details = details or {}


class VersionError(PartError):
    agent_code = "VERSION"
    default_hint = "Update CurioMesh to the version that wrote these files, or export the parts again from this one"


class MismatchError(PartError):
    agent_code = "MISMATCH"
    default_hint = "The result changed since these parts were exported: go back to that step in History, or export the parts again"


class BorderMoved(PartError):
    agent_code = "BORDER_MOVED"
    default_hint = "Keep every border vertex where parts.json puts it; only the inside may change"


class HoleError(PartError):
    agent_code = "HOLE"
    default_hint = "Every face along the border has to stay: fill the gap so the part closes against its neighbours"


class ShapeLost(PartError):
    agent_code = "SHAPE_LOST"
    default_hint = "Follow the surface in part_<id>_source.obj; the returned part has to cover about the same area and size"


# --------------------------------------------------------------------------- files


def read_obj(path, default_mat: int = 0):
    """(points float64 (n, 3), faces [tuple], materials int32 per face) from an OBJ. Only `v`, `f` and `usemtl m<index>`
    matter; other lines are ignored. Faces before any `usemtl` get `default_mat`. Raises PartError on anything broken."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PartError(f"{path.name} cannot be read: {exc}", details={"file": path.name}) from None
    points, faces, mats, mat = [], [], [], int(default_mat)
    for number, raw in enumerate(text.splitlines(), 1):
        words = raw.split("#", 1)[0].split()
        if not words:
            continue
        where = {"file": path.name, "line": number}
        if words[0] == "v":
            try:
                point = [float(w) for w in words[1:4]]
            except ValueError:
                point = []
            if len(point) != 3 or not all(np.isfinite(point)):
                raise PartError(f"{path.name} line {number}: a vertex needs three finite numbers", details=where)
            points.append(point)
        elif words[0] == "usemtl":
            name = words[1] if len(words) > 1 else ""
            if not (name.startswith("m") and name[1:].isdigit()):
                raise PartError(f"{path.name} line {number}: materials are written as usemtl m<index>, not {name!r}", details=where)
            mat = int(name[1:])
        elif words[0] == "f":
            face = []
            for token in words[1:]:
                try:
                    index = int(token.split("/", 1)[0])
                except ValueError:
                    raise PartError(f"{path.name} line {number}: {token!r} is not a vertex number", details=where) from None
                index = len(points) + index if index < 0 else index - 1  # OBJ counts from 1; negative counts back
                if not 0 <= index < len(points):
                    raise PartError(f"{path.name} line {number}: vertex {token} is not there", details=dict(where, vertices=len(points)))
                face.append(index)
            if not 3 <= len(face) <= MAX_CORNERS:
                raise PartError(f"{path.name} line {number}: a face has {len(face)} corners; 3 to {MAX_CORNERS} are allowed", details=dict(where, corners=len(face)))
            if len(set(face)) != len(face):
                raise PartError(f"{path.name} line {number}: a face uses the same corner twice", details=where)
            faces.append(tuple(face))
            mats.append(mat)
    if not faces:
        raise PartError(f"{path.name} has no faces", details={"file": path.name})
    return np.asarray(points, dtype=np.float64).reshape(-1, 3), faces, np.asarray(mats, dtype=np.int32)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_manifest(folder) -> dict:
    """parts.json, checked: format, version, and every part's id, border loops and border points."""
    path = Path(folder) / "parts.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PartError(f"parts.json cannot be read: {exc}", "Pick the folder that holds parts.json", {"folder": str(folder)}) from None
    except ValueError as exc:
        raise PartError(f"parts.json is not valid JSON: {exc}", details={"folder": str(folder)}) from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise PartError(f"parts.json is not a CurioMesh part file (format should be {FORMAT!r})", details={"format": manifest.get("format") if isinstance(manifest, dict) else None})
    version = manifest.get("version")
    if not (isinstance(version, list) and len(version) == 2 and all(_is_int(v) for v in version)):
        raise PartError("parts.json has no version [major, minor]", details={"version": version})
    if version[0] != READS[0]:
        raise VersionError(
            f"These parts are format {version[0]}.{version[1]}; {MADE_BY} reads format {READS[0]}.x",
            f"Use the CurioMesh that wrote them ({manifest.get('made_by', 'a newer one')})" if version[0] > READS[0] else "",
            {"version": version, "reads": list(READS), "made_by": manifest.get("made_by")},
        )
    parts = manifest.get("parts")
    if not isinstance(parts, list):
        raise PartError("parts.json has no list of parts")
    seen = set()
    for entry in parts:
        if not isinstance(entry, dict) or not _is_int(entry.get("id")) or entry["id"] < 0 or entry["id"] in seen:
            raise PartError("parts.json: every part needs its own whole-number id", details={"part": entry.get("id") if isinstance(entry, dict) else None})
        seen.add(entry["id"])
        entry["loops"] = border_of(entry)
    return manifest


def border_of(entry: dict) -> list[np.ndarray]:
    """A manifest part's border as loops of points, split by the lengths of its `border` loops (a loop that repeats its
    first vertex at the end is closed; the repeat has no point of its own)."""
    part = entry.get("id")
    loops, flat = entry.get("border"), entry.get("border_points")
    if not isinstance(loops, list) or not isinstance(flat, list) or not all(isinstance(loop, list) and all(_is_int(v) for v in loop) for loop in loops):
        raise PartError(f"parts.json: part {part} has no border loops", details={"part": part})
    try:
        points = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
    except (ValueError, TypeError):
        raise PartError(f"parts.json: part {part} has broken border points", details={"part": part}) from None
    if not np.isfinite(points).all():
        raise PartError(f"parts.json: part {part} has border points that are not finite numbers", details={"part": part})
    sizes = [len(loop) - 1 if len(loop) > 1 and loop[0] == loop[-1] else len(loop) for loop in loops]
    if sum(sizes) != len(points) or any(s < 3 for s in sizes):
        raise PartError(f"parts.json: part {part} has {len(points)} border points for loops of {sizes} points", details={"part": part, "points": len(points), "loops": sizes})
    ends = np.cumsum(sizes)
    return [points[a:b] for a, b in zip(ends - sizes, ends)]


def result_key(faces_count: int, border_points) -> str:
    """The result's short key: its face count and its border points in any order, as float32 (what the files hold).
    It must stay the same as the exporter's (`partpack.result_key`)."""
    pts = np.asarray(border_points, dtype=np.float32).reshape(-1, 3)
    if len(pts):
        pts = pts[np.lexsort(pts.T[::-1])]
    return hashlib.sha1(str(int(faces_count)).encode() + pts.tobytes()).hexdigest()[:8]


def check_result(manifest: dict, faces_count: int, border_points, part_ids) -> None:
    """MISMATCH unless the parts were exported from the result on screen: every part id is in its table and the key
    (face count plus the parts' border points now) is the one stored."""
    made = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    details = {"made_from_faces": made.get("result_faces"), "faces_now": int(faces_count), "made_at_step": manifest.get("step")}
    unknown = sorted(e["id"] for e in manifest["parts"] if e["id"] not in set(part_ids))
    if unknown:
        raise MismatchError(f"Parts {unknown} are not in the result on screen", details=dict(details, unknown=unknown))
    key = result_key(faces_count, border_points)
    if made.get("result_key") != key:
        step = f" (step {manifest['step']})" if manifest.get("step") is not None else ""
        raise MismatchError(f"These parts were exported from a different result{step}: {made.get('result_faces')} faces then, {faces_count} now", details=dict(details, key=made.get("result_key"), key_now=key))


def returned(folder, part: int, default_mat: int = 0):
    """(points, faces, materials) of returned/part_<id>.obj, or None when there is no such file."""
    path = Path(folder) / "returned" / f"part_{int(part)}.obj"
    return read_obj(path, default_mat) if path.is_file() else None


# --------------------------------------------------------------------------- checks


def _label(part) -> str:
    """The name the panel shows for part id `part` (ids count from 0, names from 1)."""
    return f"Part {part + 1}" if isinstance(part, int) else "The part"


def _tolerance(size: float, border: np.ndarray) -> float:
    # the files hold float32: far from the origin its rounding is bigger than a share of the size
    reach = float(np.abs(border).max()) if len(border) else 0.0
    return max(TOLERANCE * float(size), 4.0 * float(np.finfo(np.float32).eps) * reach)


def match_border(points, faces, border_points, size: float, part=None):
    """Find the returned part's open-edge loops and match each stored border point to exactly one of their vertices,
    within 1e-6 of the object's `size`. Returns (points with the border snapped to the stored values, faces); when every
    loop runs the other way round the faces are turned over, so they face the same side as their neighbours.

    BORDER_MOVED when a point is missing, matched twice, an extra vertex lies on the border, or a loop visits the
    points in another cyclic order. `details` names the loop, the point and how far off it is."""
    points = np.array(points, dtype=np.float64).reshape(-1, 3)
    stored = [np.asarray(loop, dtype=np.float64).reshape(-1, 3) for loop in border_points]
    every = np.vstack(stored) if stored else np.zeros((0, 3))
    tol = _tolerance(size, every)
    loops, pinched = boundary_loops(faces, range(len(faces)))
    if pinched:
        raise BorderMoved(f"{_label(part)}: the returned border touches itself at {len(pinched)} vertex(es)", details={"part": part, "points": points[pinched[:5]].tolist()})
    if not stored:
        if loops:
            raise BorderMoved(f"{_label(part)} had no border, the returned one has {len(loops)} open loop(s)", details={"part": part, "loops": len(loops)})
        return points, list(faces)
    on_border = np.asarray([v for loop in loops for v in loop], dtype=np.int64)
    loop_of = {v: (k, i) for k, loop in enumerate(loops) for i, v in enumerate(loop)}
    if not len(on_border):
        raise BorderMoved(f"{_label(part)}: the returned part has no open border; its border has to stay where it was", details={"part": part, "loop": 0, "point": 0})
    # ponytail: brute-force distances, border x border; a k-d tree if parts with tens of thousands of border points show up
    matched, owner = [], {}
    for k, loop in enumerate(stored):
        found = []
        for i, p in enumerate(loop):
            dist = np.linalg.norm(points[on_border] - p, axis=1)
            near = np.flatnonzero(dist <= tol)
            where = {"part": part, "loop": k, "point": i, "at": p.tolist()}
            if len(near) != 1:
                if not len(near):
                    raise BorderMoved(f"{_label(part)}: border point {i} of loop {k + 1} moved {float(dist.min()):.3g}", details=dict(where, off=float(dist.min())))
                raise BorderMoved(f"{_label(part)}: border point {i} of loop {k + 1} is there {len(near)} times", details=dict(where, off=0.0, copies=int(len(near))))
            v = int(on_border[near[0]])
            if v in owner:
                raise BorderMoved(f"{_label(part)}: border points {owner[v][1]} and {i} land on the same returned vertex", details=dict(where, off=0.0))
            owner[v] = (k, i)
            found.append(v)
        matched.append(found)
    extra = [v for v in on_border.tolist() if v not in owner]
    if extra:
        raise BorderMoved(f"{_label(part)}: {len(extra)} returned vertex(es) on the border were not in the exported border", details={"part": part, "loop": loop_of[extra[0]][0], "point": None, "extra": points[extra[:5]].tolist(), "off": None})
    turned = []
    for k, found in enumerate(matched):
        ring = loops[loop_of[found[0]][0]]
        if len(ring) != len(found):
            raise BorderMoved(f"{_label(part)}: loop {k + 1} is split or joined with another loop", details={"part": part, "loop": k, "point": 0, "off": 0.0})
        start = ring.index(found[0])
        forward = [ring[(start + i) % len(ring)] for i in range(len(ring))]
        backward = [ring[(start - i) % len(ring)] for i in range(len(ring))]
        if found not in (forward, backward):
            bad = next(i for i, (a, b) in enumerate(zip(found, forward if found[1] == forward[1] else backward)) if a != b)
            raise BorderMoved(f"{_label(part)}: loop {k + 1} visits its points in another order from point {bad}", details={"part": part, "loop": k, "point": bad, "off": 0.0})
        turned.append(found != forward)
    if any(turned) and not all(turned):
        raise BorderMoved(f"{_label(part)}: some border loops run the other way round, so some faces are turned over", details={"part": part, "loop": turned.index(True), "point": 0, "off": 0.0})
    for found, loop in zip(matched, stored):
        points[found] = loop  # exactly the stored values, so the seams weld
    faces = [tuple(reversed(f)) for f in faces] if all(turned) else list(faces)
    return points, faces


def check_no_hole(old_faces, new_faces, part=None) -> None:
    """HOLE when the result has more open edges after the part was spliced in and welded than before."""
    before, after = S.open_edges(old_faces), S.open_edges(new_faces)
    if after > before:
        raise HoleError(f"{_label(part)}: {after - before} new open edge(s) after stitching", details={"part": part, "open_before": before, "open_after": after})


def check_shape(src_points, src_faces, points, faces, part=None) -> None:
    """SHAPE_LOST when the returned part lost the source part's shape (`segment.shape_kept`), with the area and size shares."""
    if S.shape_kept(src_points, src_faces, points, faces):
        return
    area = float(S.face_frames(src_points, src_faces)[1].sum())
    made = float(S.face_frames(points, faces)[1].sum())
    extent, made_extent = np.ptp(np.asarray(src_points, dtype=np.float64), axis=0), np.ptp(np.asarray(points, dtype=np.float64), axis=0)
    real = extent > 0.01 * float(np.linalg.norm(extent))
    share = float((made_extent[real] / extent[real]).min()) if real.any() else 1.0
    raise ShapeLost(f"{_label(part)}: the returned faces cover {100 * made / max(area, 1e-30):.0f}% of the area, and {100 * share:.0f}% of the size along its shortest side", details={"part": part, "area_share": made / max(area, 1e-30), "extent_share": share})


def error_reply(exc: Exception) -> dict:
    """{code, message, hint, details} of a refusal, for the per-part list of an import."""
    return {"code": getattr(exc, "agent_code", "INTERNAL"), "message": str(exc), "hint": getattr(exc, "hint", ""), "details": getattr(exc, "details", {})}
