# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Flow Editor strokes: what a drawn stroke is, clean shapes made from it, and the part of a result redone to follow it.

A stroke is one of three kinds:

- FLOW: the rows should run along it (it steers the direction of the quads, no edge has to sit on it);
- RING: a closed loop that becomes a true edge loop, with rows of the same count on both sides (eye, mouth, hole, bolt);
- EDGE: a line that must become a run of edges (a crease, a panel line).

`recognise` tells them apart from the drawing alone, `tidy` turns a hand-drawn stroke into a clean one, `ellipse` and
`template` make clicked shapes. `areas` picks the faces of a result around the strokes, and `redo` remakes them on
their seams with the existing field mesher (`triadq.segment.field_with_rings`), so the rest of the result is untouched.
`record` gives each stroke as surface points, unit directions and its kind: plain data any other solver can read as
a constraint. Plain NumPy.
"""

from __future__ import annotations

import numpy as np

from . import guides
from .triadq import parts as P
from .triadq import patch
from .triadq import segment as S

KINDS = ("FLOW", "RING", "EDGE")
NAMES = {"FLOW": "Flow", "RING": "Ring", "EDGE": "Edge"}
SAMPLES = 96  # points a stroke is resampled to before it is judged
CLOSED_GAP = 0.2  # end-to-end gap, as a share of the stroke's length, under which it can be a ring
RING_TURN = 290.0  # degrees a ring has to sweep round its middle (a figure-eight goes both ways and nets ~0)
ON_CREASE = 0.7  # share of the stroke that has to run along a sharp crease for it to be an edge
ALONG = 0.8  # |cos| between the stroke and the crease for "along"
HARMONICS = 4  # the cleaned ring keeps the shape up to this many waves round: an ellipse or an egg, not the wobble
LAY_ROUNDS = 3  # steps pulling a run of edges onto a drawn line
REACH = {"FLOW": 2.5, "EDGE": 2.5, "RING": 6.0}  # a ring needs room for its own rows each side and a gap  # faces this many quad widths from a stroke are redone


# --------------------------------------------------------------------------- reading a stroke


def resample(points, count: int, closed: bool = False) -> np.ndarray:
    """`count` points evenly spaced along the polyline (round the loop when `closed`)."""
    p = np.asarray(points, dtype=np.float64)
    if closed:
        p = np.vstack((p, p[:1]))
    at = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))))
    if at[-1] <= 1e-12:
        return np.repeat(p[:1], count, axis=0)
    t = np.linspace(0.0, at[-1], count, endpoint=not closed)
    return np.stack([np.interp(t, at, p[:, k]) for k in range(3)], axis=1)


def _frame(points):
    """(centre, normal, u, v) of the plane that best fits the points."""
    centre = points.mean(axis=0)
    _s, _v, vt = np.linalg.svd(points - centre, full_matrices=False)
    return centre, vt[2], vt[0], vt[1]


def turning(points, normal) -> float:
    """Signed degrees the open polyline turns, seen along `normal`."""
    d = np.diff(points, axis=0)
    d -= (d @ normal)[:, None] * normal
    a, b = d[:-1], d[1:]
    return float(np.degrees(np.arctan2(np.cross(a, b) @ normal, (a * b).sum(axis=1))).sum())


def winding(points, normal) -> float:
    """Signed degrees the polyline sweeps round its own middle, seen along `normal` (steady against a shaky hand)."""
    c = points.mean(axis=0)
    u = points[1] - points[0] if len(points) > 1 else np.array([1.0, 0.0, 0.0])
    u = u - (u @ normal) * normal
    if np.linalg.norm(u) < 1e-12:
        u = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rel = points - c
    a = np.unwrap(np.arctan2(rel @ v, rel @ u))
    return float(np.degrees(a[-1] - a[0]))


def recognise(points, normals=None, crease=None) -> str:
    """"RING", "FLOW" or "EDGE" for a drawn stroke. `normals`: the surface normal under each point (else the plane
    of the stroke is used). `crease(points)` -> (on a sharp crease per point, the crease's direction per point), or None."""
    raw = np.asarray(points, dtype=np.float64)
    if len(raw) < 3:
        return "FLOW"
    p = resample(raw, SAMPLES)
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    if length <= 1e-12:
        return "FLOW"
    if crease is not None:
        on, along = crease(p)
        d = np.gradient(p, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
        follows = np.asarray(on, dtype=bool) & (np.abs((d * np.asarray(along)).sum(axis=1)) > ALONG)
        if follows.mean() >= ON_CREASE:
            return "EDGE"
    gap = float(np.linalg.norm(p[0] - p[-1])) / length
    if gap < CLOSED_GAP:
        normal = _frame(p)[1]
        if normals is not None:
            n = np.asarray(normals, dtype=np.float64).mean(axis=0)
            if np.linalg.norm(n) > 1e-9 and abs(float(n @ normal)) > 0.3:  # a ring round a bump lies in its own plane; keep that, facing out
                normal = normal * np.sign(float(n @ normal))
        sweep, turn = winding(p, normal), turning(p, normal)
        if abs(sweep) >= RING_TURN and sweep * turn > 0 and abs(turn) >= 0.5 * RING_TURN:  # a figure-eight's heading turns both ways
            return "RING"
    return "FLOW"


def _smooth(p, rounds: int, closed: bool) -> np.ndarray:
    p = p.copy()
    for _ in range(rounds):
        if closed:
            p = 0.25 * np.roll(p, 1, axis=0) + 0.5 * p + 0.25 * np.roll(p, -1, axis=0)
        else:
            p[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    return p


def fit_ring(points, count: int = 64) -> np.ndarray:
    """A clean closed curve through a hand-drawn loop: the loop as waves round it, keeping the first few (an ellipse
    or an egg survives, the hand's wobble goes). Ends where it starts (the first point is repeated)."""
    p = np.asarray(points, dtype=np.float64)
    if np.linalg.norm(p[0] - p[-1]) > 1e-12:
        pass  # the gap is closed by the resampling round the loop
    p = resample(p, 256, closed=True)
    spectrum = np.fft.rfft(p, axis=0)
    spectrum[HARMONICS + 1 :] = 0.0
    smooth = np.fft.irfft(spectrum, n=len(p), axis=0)
    out = resample(smooth, count, closed=True)
    return np.vstack((out, out[:1]))


def tidy(kind: str, points, nearest=None, count: int | None = None) -> np.ndarray:
    """A drawn stroke as the clean line its kind wants, laid on the surface by `nearest(points)` (else left where it is)."""
    p = np.asarray(points, dtype=np.float64)
    if kind == "RING":
        out = fit_ring(p, count or 64)
    else:
        n = count or max(16, min(200, len(p)))
        out = _smooth(resample(p, n), 6 if kind == "FLOW" else 2, False)
    return out if nearest is None else np.asarray(nearest(out), dtype=np.float64)


def ellipse(centre, normal, right, rx: float, ry: float, count: int = 48) -> np.ndarray:
    """A closed ellipse in the plane at `centre` facing `normal`, `rx` along `right` (first point repeated)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-30)
    u = np.asarray(right, dtype=np.float64) - (np.asarray(right) @ n) * n
    u = u / max(np.linalg.norm(u), 1e-30)
    v = np.cross(n, u)
    a = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    out = np.asarray(centre, dtype=np.float64) + rx * np.cos(a)[:, None] * u + ry * np.sin(a)[:, None] * v
    return np.vstack((out, out[:1]))


def template(shape: str, centre, normal, right, size: float) -> list[tuple[str, np.ndarray]]:
    """[(kind, line)] for a clicked shape `size` wide: an eye ring (an almond, wider than tall), a mouth ring, or the two
    lines from beside the nose wings round the mouth corners to below the lip (click the middle of the mouth)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-30)
    u = np.asarray(right, dtype=np.float64) - (np.asarray(right) @ n) * n
    u = u / max(np.linalg.norm(u), 1e-30)
    v = np.cross(n, u)
    c = np.asarray(centre, dtype=np.float64)
    if shape == "EYE":
        return [("RING", ellipse(c, n, u, 0.5 * size, 0.24 * size))]
    if shape == "MOUTH":
        return [("RING", ellipse(c, n, u, 0.5 * size, 0.28 * size))]
    # In mouth widths: from beside the nose wing, out round the corner, in below the lower lip.
    path = np.asarray([(0.32, 0.75), (0.45, 0.45), (0.6, 0.1), (0.62, -0.2), (0.52, -0.5), (0.35, -0.7)])
    path = np.vstack([np.interp(np.linspace(0, len(path) - 1, 24), np.arange(len(path)), path[:, k]) for k in range(2)]).T
    out = []
    for side in (1.0, -1.0):
        line = c + size * (side * path[:, :1] * u + path[:, 1:] * v)
        out.append(("FLOW", line))
    return out


def record(kind: str, points, normals=None) -> dict:
    """A stroke as plain data for any solver: surface points, unit direction per point, its kind and whether it closes."""
    p = np.asarray(points, dtype=np.float64)
    closed = kind == "RING" or (len(p) > 3 and np.linalg.norm(p[0] - p[-1]) < 1e-9)
    d = np.gradient(p, axis=0)
    if closed and len(p) > 3:
        d[0] = d[-1] = p[1] - p[-2]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    return {"kind": kind, "points": p, "directions": d, "normals": None if normals is None else np.asarray(normals, dtype=np.float64), "closed": bool(closed)}


# --------------------------------------------------------------------------- strokes into edges (shared with segment redo)


def is_closed(line) -> bool:
    """A stroke that ends where it began."""
    line = np.asarray(line, dtype=np.float64)
    return len(line) >= 4 and float(np.linalg.norm(line[0] - line[-1])) < 0.03 * float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def on_lines(points, tris, lines):
    """For the field mesher: chains of edges hugging the drawn lines, moved onto them, each vertex told which way its line runs."""
    spacing = float(np.median(np.linalg.norm(points[tris[:, 0]] - points[tris[:, 1]], axis=1)))
    chains, _keys = guides.chain(points, tris, lines, spacing)
    moved = guides.pull(points, tris, chains, lines, spacing) if chains else points
    along = np.zeros_like(moved)
    pinned = np.zeros(len(moved), dtype=np.int64)
    for chain in chains:
        ids = np.asarray(chain, dtype=np.int64)
        if len(ids) < 3:
            continue
        closed = ids[0] == ids[-1]
        ids = ids[:-1] if closed else ids
        step = np.roll(moved[ids], -1, axis=0) - np.roll(moved[ids], 1, axis=0)
        if not closed:
            step[0], step[-1] = moved[ids[1]] - moved[ids[0]], moved[ids[-1]] - moved[ids[-2]]
        along[ids] = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-30)
        pinned[ids] = 1
    return moved, along, pinned


def lay_edges(co, faces, pinned, lines, project):
    """The drawn lines become runs of the result's own edges: the cheapest path of edges along each is pulled onto it, and the faces beside it settle."""
    edges = patch._unique_edges(faces)
    cell = float(np.median(np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)))
    chains, _keys = guides.chain(co, None, lines, cell, edges=(edges[:, 0], edges[:, 1]))
    if not chains:
        return co
    fan = np.asarray([(f[0], f[k], f[k + 1]) for f in faces for k in range(1, len(f) - 1)], dtype=np.int64)
    held = pinned.copy()
    held[np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in chains]))] = True
    for _ in range(LAY_ROUNDS):
        co = guides.pull(co, fan, chains, lines, cell, pinned)
        co = patch.fit_to_surface(co, faces, held, project, 2, 0.5, guard=False)
    return patch.fit_to_surface(co, faces, held, project, 1, 0.5)


# --------------------------------------------------------------------------- the part of a result redone


def _centres(co, faces) -> np.ndarray:
    return np.asarray([co[list(f)].mean(axis=0) for f in faces])


def edge_near(co, faces, line, centres=None) -> float:
    """The quad width of the result where a stroke runs."""
    centres = _centres(co, faces) if centres is None else centres
    size = float(np.linalg.norm(np.ptp(co, axis=0))) or 1.0
    d = guides._to_line(centres, np.asarray(line, dtype=np.float64))
    near = np.argsort(d)[: max(8, min(64, len(d)))]
    return float(np.mean([np.linalg.norm(co[faces[i][0]] - co[faces[i][2 % len(faces[i])]]) for i in near])) / np.sqrt(2.0) or 0.01 * size


def _inside_ring(centres, ring) -> np.ndarray:
    """Face centres inside a closed stroke and near its plane (a ring round an eye takes the eye, not the back of the head)."""
    c, n, u, v = _frame(ring)
    radius = float(np.linalg.norm(ring - c, axis=1).max())
    rel = centres - c
    flat = np.stack((rel @ u, rel @ v), axis=1)
    poly = np.stack(((ring - c) @ u, (ring - c) @ v), axis=1)
    near = (np.abs(rel @ n) < 0.6 * radius) & (np.linalg.norm(rel, axis=1) < radius)
    out = np.zeros(len(centres), dtype=bool)
    idx = np.flatnonzero(near)
    if len(idx):
        out[idx] = patch._inside_polygon(flat[idx], poly)
    return out


def areas(co, faces, strokes, centres=None, adjacency=None) -> list[tuple[np.ndarray, list]]:
    """[(face ids, strokes in it)]: the faces each stroke reaches, joined where they touch, cleaned into solid patches."""
    co = np.asarray(co, dtype=np.float64)
    centres = _centres(co, faces) if centres is None else centres
    adjacency = adjacency if adjacency is not None else patch.face_adjacency(faces)
    masks = []
    for s in strokes:
        line = np.asarray(s["points"], dtype=np.float64)
        width = edge_near(co, faces, line, centres)
        mask = guides._to_line(centres, line) < REACH[s["kind"]] * width
        if s["kind"] == "RING":
            mask |= _inside_ring(centres, line)
        masks.append(mask)
    if not masks:
        return []
    whole = patch.clean_mask(faces, np.any(masks, axis=0), min_island=4, max_hole=64, adjacency=adjacency)
    out = []
    for ids in patch.mask_components(faces, whole, adjacency):
        inside = np.zeros(len(faces), dtype=bool)
        inside[ids] = True
        out.append((np.sort(ids), [s for s, m in zip(strokes, masks) if (m & inside).any()]))
    return out


def piece(co, faces, ids, project=None):
    """The patch of the result as a finer triangle mesh on your surface (every quad cut in four, then in two), for the
    field mesher to work in. Returns (points, triangles)."""
    co = np.asarray(co, dtype=np.float64)
    pts = [p for p in co]
    index: dict = {}

    def mid(*vs):
        key = tuple(sorted(vs))
        if key not in index:
            index[key] = len(pts)
            pts.append(co[list(vs)].mean(axis=0))
        return index[key]

    used = {v for i in ids for v in faces[i]}
    remap = {v: k for k, v in enumerate(sorted(used))}
    tris = []
    for i in ids:
        f = faces[i]
        m = mid(*f)
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            e = mid(a, b)
            tris += [(a, e, m), (e, b, m)]
    all_used = sorted({v for t in tris for v in t})
    order = {v: k for k, v in enumerate(all_used)}
    points = np.asarray([pts[v] for v in all_used], dtype=np.float64)
    if project is not None:
        points = np.asarray(project(points), dtype=np.float64)
    del remap
    return points, [tuple(order[v] for v in t) for t in tris]


def redo(co, faces, ids, strokes, project=None, target: int | None = None):
    """The faces `ids` of a result remade to follow `strokes` on the same seams. Returns (points, faces). Raises
    ValueError where the field mesher can't."""
    seams = S.seam_loops(co, faces, ids)
    points, tris = piece(co, faces, ids, project)
    target = len(ids) if target is None else target
    rings = [np.asarray(s["points"], dtype=np.float64) for s in strokes if s["kind"] == "RING"]
    lines = [np.asarray(s["points"], dtype=np.float64) for s in strokes if s["kind"] != "RING"]
    edges = [np.asarray(s["points"], dtype=np.float64) for s in strokes if s["kind"] == "EDGE"]
    marks = (lambda pts, t: on_lines(pts, t, lines)) if lines else None
    settle = (lambda c, f, pinned: lay_edges(c, f, pinned, edges, project)) if edges and project is not None else None
    try:
        return S.field_with_rings(points, tris, rings, seams, target, project, marks, settle)
    except ValueError:
        if not rings:
            raise
    lines = lines + rings  # a loop that can't be a true one still steers the rows
    marks = lambda pts, t: on_lines(pts, t, lines)  # noqa: E731
    return S.field_on_seams(points, tris, seams, target, project, marks, settle)


def splice(co, faces, mat, builds: list[tuple[np.ndarray, tuple]]):
    """The result with each (face ids, (points, faces)) put in, seams welded. Returns (co, faces, mat, new face mask)."""
    label = np.zeros(len(faces), dtype=np.int64)
    parts = {}
    for k, (ids, (points, made)) in enumerate(builds, start=1):
        label[ids] = k
        most = int(np.bincount(np.asarray(mat)[ids]).argmax()) if len(ids) else 0
        parts[k] = (points, made, [most] * len(made))
    co2, faces2, mat2, piece2 = P.splice(co, faces, mat, label, parts)
    co2, faces2 = S.weld_border(co2, faces2)
    return co2, faces2, mat2, piece2 > 0


__all__ = ["KINDS", "areas", "ellipse", "fit_ring", "recognise", "record", "redo", "splice", "template", "tidy"]
