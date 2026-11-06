# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Redraw in Rows: a marked area of a result laid out again as straight rows of quads, at the quads' own size.

The area is flattened (the flat map of `segment.even_embedding`, its border the border's outline on its best plane,
made convex and spaced by length on the surface; spread out where the old quads were small, so the rows keep that
detail), and a grid on the map comes out as quads on the surface. One direction is chosen for the
whole area: a cross field that is constant on the map, which is the smoothest field there is, turned to best fit

- the surface's bends: per point, the bending of the edges around it summed as a curvature tensor (Cohen-Steiner and
  Morvan 2003, "Restricted Delaunay triangulations and normal cycle"), which also holds any hard edge inside;
- the drawn lines inside the area, which count far more;
- the area's long side, a little, so a plain area gets rows along its length.

The fit is a least-squares average of the four-fold directions in the power form u = exp(4i theta) (Knoppel, Crane,
Pinkall and Schroder 2013, "Globally Optimal Direction Fields": with the smoothness weight taken to its limit, the
optimum on a flat map is this constant field). The grid's cells are whole inside the area, so it has no poles there;
they go to the one row that ties the grid to the untouched faces round it (`band`: the seam's own vertices, so the
border keeps its edge count, each tie chosen by dynamic programming so the grid's corners and the border's turns get
the edges they lack). Plain NumPy.
"""

from __future__ import annotations

import numpy as np

from . import fieldmesh
from .patch import (
    _inside_polygon,
    _Locator,
    _polygon_distance,
    boundary_loops,
    circle_boundary,
    clean_mask,
    face_adjacency,
    mask_components,
    quad_quality,
)

MARGIN = 0.55  # cells: grid corners stay this far inside the border, leaving room for the row that ties it on
LINE_WEIGHT = 4.0  # drawn lines against all the surface's bends
LONG_WEIGHT = 0.15  # the area's long side against all the surface's bends
TENSOR_SMOOTH = 3  # rounds averaging each point's bending tensor with its neighbours'
DETAIL_RANGE = 3.0  # quads at most this much finer or coarser than the area's usual where the old ones were
CONVEX_ROUNDS = 400  # rounds pulling in the corners of the flat map's border that turn back
BAND_REACH = 30  # seam vertices a tie may run ahead of or behind the even spread
POLE_COST = 1.0  # a grid corner off four edges, against one bad quad
SEAM_POLE_COST = 0.3  # the same on the seam, where the faces outside decide much of it
SHAPE_COST = 2.0  # per unit of a tie quad's quality under 1
TIDY_ROUNDS = 3  # rounds smoothing the marked area's border
TIDY_HOLE = 64  # faces: a smaller gap in the marked area is filled, a bigger one stays a hole the rows go round
FOLD = -0.866  # neighbours more than 150 degrees apart lie folded over each other
SEAM_ROUNDS = 3  # rounds easing the area's border along itself
SEAM_MOST = 0.35  # of the shorter border edge beside it: the most a border point moves
POOR, BAD_COST = 0.1, 40.0  # a tie quad this poor (folded, or a corner near flat) costs as much as many poles


def _detail(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per point, how much finer than usual the quads were there: the piece's own triangles are the old quads cut up,
    so their size is the old quads' size. Spreads the map where quads were small, so the rows keep that detail."""
    area = fieldmesh._normals_areas(points, tris)[1]
    count = np.maximum(np.bincount(tris.reshape(-1), minlength=len(points)), 1)
    size = np.sqrt(np.maximum(np.bincount(tris.reshape(-1), np.repeat(area, 3), len(points)) / count, 1e-30))
    return np.clip(float(np.median(size)) / size, 1.0 / DETAIL_RANGE, DETAIL_RANGE)


def _convex(outline: np.ndarray, rounds: int = CONVEX_ROUNDS) -> np.ndarray:
    """The outline with each corner that turns back in pulled toward its neighbours' middle until none does: a flat
    map with a convex border can't fold (Tutte 1963, Floater 1997)."""
    p = outline.copy()
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    for _ in range(rounds):
        before, after = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
        turn = (p[:, 0] - before[:, 0]) * (after[:, 1] - p[:, 1]) - (p[:, 1] - before[:, 1]) * (after[:, 0] - p[:, 0])
        back = turn * area < 0.0
        if not back.any():
            break
        p[back] = 0.5 * (p[back] + 0.5 * (before[back] + after[back]))
    return p


def _along(outline: np.ndarray, border: np.ndarray) -> np.ndarray:
    """The border's points moved round the outline so each gets the share of it that its edges have of the border's
    length on the surface: a side seen at a slant gets its full length back instead of a squeezed one."""
    def run(p):
        step = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        return np.concatenate(([0.0], np.cumsum(step))), step.sum()

    have, total = run(outline)
    want, length = run(border)
    at = want[:-1] / max(length, 1e-30) * total
    closed = np.vstack((outline, outline[:1]))
    return np.stack([np.interp(at, have, closed[:, k]) for k in range(2)], axis=1)


def _flat(points: np.ndarray, tris: np.ndarray, border: np.ndarray) -> np.ndarray:
    """The even flat map, its border the border's own outline on the area's best plane made convex; a circle where that
    still folds."""
    from . import segment

    weight = _detail(points, tris)
    centred = points - points.mean(axis=0)
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    outline = _along(_convex(centred[border] @ axes[:2].T), points[border])
    for loop_uv in (outline, circle_boundary(points[border]) * float(np.ptp(outline, axis=0).max()) * 0.5):
        uv = segment.even_embedding(points, tris, border, loop_uv, weight, graded=True)
        q = uv[tris]
        signed = (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1])
        if min((signed > 0).sum(), (signed < 0).sum()) == 0:
            return uv
    raise ValueError("This area can't be laid flat to draw rows on it")


def _jacobians(points, tris, uv) -> np.ndarray:
    """Per point, the 2x3 map from a direction on the surface to one on the flat map (its triangles' maps averaged)."""
    e = np.stack((points[tris[:, 1]] - points[tris[:, 0]], points[tris[:, 2]] - points[tris[:, 0]]), axis=2)  # (T, 3, 2)
    f = np.stack((uv[tris[:, 1]] - uv[tris[:, 0]], uv[tris[:, 2]] - uv[tris[:, 0]]), axis=2)  # (T, 2, 2)
    gram = np.einsum("tki,tkj->tij", e, e)
    gram += np.eye(2) * 1e-30
    per_tri = np.einsum("tij,tjk,tlk->til", f, np.linalg.inv(gram), e)  # F (E^T E)^-1 E^T
    out = np.zeros((len(points), 2, 3))
    np.add.at(out, tris.reshape(-1), np.repeat(per_tri, 3, axis=0))
    return out / np.maximum(np.bincount(tris.reshape(-1), minlength=len(points)), 1)[:, None, None]


def _power(vectors: np.ndarray) -> complex:
    """Sum of the four-fold power form of 2D directions, each counting by its length."""
    angle = np.arctan2(vectors[:, 1], vectors[:, 0])
    return complex(np.sum(np.linalg.norm(vectors, axis=1) * np.exp(4j * angle)))


def direction(points, tris, uv, along=None, on_line=None) -> float:
    """The angle on the flat map the rows run at (mod 90 degrees): bends, drawn lines and the long side, see above."""
    a, b, edge_of, uses = fieldmesh._edges(tris, len(points))
    normals = fieldmesh._normals_areas(points, tris)[0]
    bend = fieldmesh.edge_turn(tris, normals, edge_of, len(a)) * np.linalg.norm(points[a] - points[b], axis=1)
    d = uv[b] - uv[a]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    parts = (d[:, 0] * d[:, 0], d[:, 0] * d[:, 1], d[:, 1] * d[:, 1])
    tensor = np.stack([np.bincount(a, bend * p, len(points)) + np.bincount(b, bend * p, len(points)) for p in parts], axis=1)
    tensor = np.stack([fieldmesh._smooth(tensor[:, k], a, b, TENSOR_SMOOTH) for k in range(3)], axis=1)
    # (t00 - t11) + 2i t01 is the tensor's spread times exp(2i phi); squared and over the spread, the four-fold vote.
    two = (tensor[:, 0] - tensor[:, 2]) + 2j * tensor[:, 1]
    votes = np.where(np.abs(two) > 0, two * two / np.maximum(np.abs(two), 1e-300), 0)
    area = np.bincount(tris.reshape(-1), np.repeat(fieldmesh._normals_areas(points, tris)[1], 3), len(points))
    total = complex(np.sum(area * votes))
    scale = float(np.sum(area * np.abs(votes))) or 1.0
    if along is not None and on_line is not None and np.any(on_line):
        ids = np.flatnonzero(on_line)
        mapped = np.einsum("pij,pj->pi", _jacobians(points, tris, uv)[ids], along[ids])
        line = _power(mapped)
        total += LINE_WEIGHT * scale * line / max(float(np.linalg.norm(mapped, axis=1).sum()), 1e-30)
    centred = uv - uv.mean(axis=0)
    w, v = np.linalg.eigh(centred.T @ centred)
    long_side = v[:, 1] * (1.0 - w[0] / max(w[1], 1e-30))
    total += LONG_WEIGHT * scale * _power(long_side[None])
    return float(np.angle(total)) / 4.0 if abs(total) > 0 else 0.0


def _parts(mask: np.ndarray) -> np.ndarray:
    """Labels of the side-by-side connected pieces of a 2D mask (-1 off it)."""
    from .patch import _labels

    ids = np.flatnonzero(mask.ravel())
    out = np.full(mask.size, -1, dtype=np.int64)
    if len(ids):
        at = np.full(mask.size, -1, dtype=np.int64)
        at[ids] = np.arange(len(ids))
        w = mask.shape[1]
        right = ids[(ids % w < w - 1) & (at[np.minimum(ids + 1, mask.size - 1)] >= 0)]
        down = ids[(ids + w < mask.size) & (at[np.minimum(ids + w, mask.size - 1)] >= 0)]
        out[ids] = _labels(len(ids), np.concatenate((at[right], at[down])), np.concatenate((at[right + 1], at[down + w])))
    return out.reshape(mask.shape)


def _cells(inside: np.ndarray, in_hole: np.ndarray | None = None) -> np.ndarray:
    """Grid cells (i, j) whose four corners are inside, as one piece with no cell touching another only at a corner.
    Gaps in the middle are filled, except round a hole of the area (`in_hole`: grid corners inside one)."""
    cell = inside[:-1, :-1] & inside[1:, :-1] & inside[1:, 1:] & inside[:-1, 1:]
    for _ in range(64):
        # A corner where only two diagonal cells meet would pinch the border: the cell with fewer neighbours goes.
        count = np.zeros(cell.shape, dtype=np.int64)
        count[1:] += cell[:-1]
        count[:-1] += cell[1:]
        count[:, 1:] += cell[:, :-1]
        count[:, :-1] += cell[:, 1:]
        spike = cell & (count <= 1)
        a, b, c, d = cell[:-1, :-1], cell[1:, :-1], cell[:-1, 1:], cell[1:, 1:]
        pinch = np.zeros_like(cell)
        pinch[:-1, :-1] |= a & d & ~b & ~c
        pinch[1:, :-1] |= b & c & ~a & ~d
        if not spike.any() and not pinch.any():
            break
        cell &= ~spike & ~pinch
    labels = _parts(cell)
    if not cell.any():
        return cell
    out = labels == np.bincount(labels[cell]).argmax()
    gaps = _parts(~out)
    edge = np.unique(np.concatenate((gaps[0], gaps[-1], gaps[:, 0], gaps[:, -1])))
    keep_open = set(edge.tolist())
    if in_hole is not None:
        hole_cell = in_hole[:-1, :-1] | in_hole[1:, :-1] | in_hole[1:, 1:] | in_hole[:-1, 1:]
        keep_open |= set(np.unique(gaps[hole_cell]).tolist())
    return out | ((gaps >= 0) & ~np.isin(gaps, list(keep_open)))


def _loop_count(cell: np.ndarray) -> int:
    """Border loops of a set of grid cells."""
    i, j = np.nonzero(cell)
    h = cell.shape[1] + 1
    quads = [tuple(f) for f in np.stack((i * h + j, (i + 1) * h + j, (i + 1) * h + j + 1, i * h + j + 1), axis=1).tolist()]
    return len(boundary_loops(quads, range(len(quads)))[0])


def grid_quads(vertices, faces, target: float, turn: float | None = None, seed: int = 0, cancelled=None, lines=None):
    """A grid of about `target` quads over a piece (a disk, or a disk with holes in it), along `direction`, its corners
    on the piece: (points, quads). Same call as `fieldmesh.field_quads`, which does the work where the piece touches
    itself. Each hole is closed by a fan for the flat map only, and the grid keeps out of it."""
    points = np.asarray(vertices, dtype=np.float64)
    tris = fieldmesh.triangulate(faces)[0]
    loops, pinched = boundary_loops([tuple(t) for t in tris.tolist()], range(len(tris)))
    if pinched or not loops:
        return fieldmesh.field_quads(points, faces, target, turn, seed, cancelled, lines)
    along = on_line = None
    if lines is not None:
        points, along, on_line = lines(points, tris)
    loops = sorted(loops, key=lambda loop: -float(np.linalg.norm(points[loop] - points[np.roll(loop, 1)], axis=1).sum()))
    border, holes = np.asarray(loops[0], dtype=np.int64), [np.asarray(h, dtype=np.int64) for h in loops[1:]]
    count, real = len(points), len(tris)
    if holes:  # a fan over each hole, for the flat map
        points = np.vstack([points] + [points[h].mean(axis=0, keepdims=True) for h in holes])
        tris = np.vstack([tris] + [np.stack((np.roll(h, -1), h, np.full(len(h), count + k)), axis=1) for k, h in enumerate(holes)])
    uv = _flat(points, tris, border)
    theta = direction(points, tris, uv, along, on_line)
    turn_by = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    flat = uv @ turn_by.T  # the rows run along the map's axes
    q = flat[tris[:real]]
    area = 0.5 * np.abs((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1])).sum()
    step = float(np.sqrt(area / max(target, 1.0)))
    low, high = flat.min(axis=0), flat.max(axis=0)
    counts = np.ceil((high - low) / step).astype(int) + 1
    start = (low + high) / 2.0 - (counts - 1) * step / 2.0
    gx, gy = np.meshgrid(start[0] + step * np.arange(counts[0]), start[1] + step * np.arange(counts[1]), indexing="ij")
    nodes = np.stack((gx.ravel(), gy.ravel()), axis=1)
    inside = _inside_polygon(nodes, flat[border])
    in_hole = np.zeros(len(nodes), dtype=bool)
    for hole in holes:
        in_hole |= _inside_polygon(nodes, flat[hole])
    for hole in holes:  # a hole smaller than a cell still takes the cells it touches out
        at = np.clip(np.floor((flat[hole] - start) / step).astype(np.int64), 0, counts - 2)
        for di in (0, 1):
            for dj in (0, 1):
                in_hole[(at[:, 0] + di) * counts[1] + at[:, 1] + dj] = True
    inside &= ~in_hole
    for loop in [border] + holes:
        if inside.any():
            inside[inside] = _polygon_distance(nodes[inside], flat[loop]) >= MARGIN * step
    locator = _Locator(flat, tris)
    cell = _cells(inside.reshape(counts), in_hole.reshape(counts) if holes else None)
    if cell.sum() < 4:
        raise ValueError("This area is too small for rows of quads")
    if holes and _loop_count(cell) != 1 + len(holes):
        raise ValueError("two holes in the area are too close for rows between them")
    i, j = np.nonzero(cell)
    h = counts[1]
    quads = np.stack((i * h + j, (i + 1) * h + j, (i + 1) * h + j + 1, i * h + j + 1), axis=1)
    used, corners = np.unique(quads, return_inverse=True)
    quads = corners.reshape(-1, 4)
    tri, weights = locator.locate(nodes[used])
    co = np.einsum("ij,ijk->ik", weights, points[tris[tri]])
    facing = np.cross(co[quads[:, 2]] - co[quads[:, 0]], co[quads[:, 3]] - co[quads[:, 1]])
    up = fieldmesh._normals_areas(points, tris)[0][tri]
    if float((facing * up[quads[:, 0]]).sum()) < 0.0:
        quads = quads[:, ::-1]
    return co, [tuple(int(v) for v in f) for f in quads]


def folded(points, faces) -> np.ndarray:
    """Per face: concave or bent past 60 degrees, or folded over a neighbour (normals more than 150 degrees apart)."""
    points = np.asarray(points, dtype=np.float64)
    out = np.zeros(len(faces), dtype=bool)
    quads = np.flatnonzero([len(f) == 4 for f in faces])
    if len(quads):
        out[quads] = quad_quality(points[np.asarray([faces[i] for i in quads.tolist()], dtype=np.int64)]) <= 0.0
    normal = np.asarray([np.cross(points[f[2]] - points[f[0]], points[f[-1]] - points[f[1]]) for f in faces]).reshape(-1, 3)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    fa, fb = face_adjacency(faces)
    over = (normal[fa] * normal[fb]).sum(axis=1) < FOLD
    out[fa[over]] = True
    out[fb[over]] = True
    return out


def ease_seam(co, faces, ids, project=None, rounds: int = SEAM_ROUNDS) -> np.ndarray:
    """The points with the area's border eased along itself, each border point a little toward the middle of its two
    border neighbours and back onto the surface, so the row tying the new quads on doesn't fold over a zigzag. A point
    goes back where an untouched face beside it would fold."""
    co = np.asarray(co, dtype=np.float64)
    loops, pinched = boundary_loops(faces, ids)
    if pinched:
        return co
    out = co.copy()
    for loop in loops:
        loop = np.asarray(loop, dtype=np.int64)
        if len(loop) < 5:
            continue
        step = np.linalg.norm(co[loop] - co[np.roll(loop, 1)], axis=1)
        most = SEAM_MOST * np.minimum(step, np.roll(step, -1))
        p = co[loop]
        for _ in range(rounds):
            p = p + 0.5 * (0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) - p)
            if project is not None:
                p = np.asarray(project(p), dtype=np.float64)
            move = p - co[loop]
            length = np.linalg.norm(move, axis=1)
            p = co[loop] + move * np.minimum(1.0, most / np.maximum(length, 1e-30))[:, None]
        out[loop] = p
    outside = np.setdiff1d(np.arange(len(faces)), np.asarray(ids, dtype=np.int64))
    for _ in range(4):  # an untouched face that would fold puts its points back
        worse = folded(out, [faces[i] for i in outside]) & ~folded(co, [faces[i] for i in outside])
        if not worse.any():
            break
        back = np.unique([v for i in outside[worse] for v in faces[i]])
        out[back] = co[back]
    return out


def tidy_area(faces, mask, rounds: int = TIDY_ROUNDS) -> np.ndarray:
    """The marked faces with a smoother border: a face with at most one marked neighbour leaves, an unmarked face with
    at most one unmarked neighbour joins (a majority filter), small gaps and crumbs go, and the faces round a corner
    where the area touches itself join. A ragged border would leave the tie row folded quads it can't avoid. A bigger
    hole stays: the rows go round it."""
    mask = np.asarray(mask, dtype=bool).copy()
    fa, fb = face_adjacency(faces)
    sides = np.bincount(fa, minlength=len(mask)) + np.bincount(fb, minlength=len(mask))
    users: dict = {}
    for i, f in enumerate(faces):
        for v in f:
            users.setdefault(v, []).append(i)
    for _ in range(rounds):
        marked = np.bincount(fa, mask[fb], len(mask)) + np.bincount(fb, mask[fa], len(mask))
        grow = ~mask & (marked >= sides - 1) & (marked >= 2)
        shrink = mask & (marked <= 1)
        mask = (mask | grow) & ~shrink
    for _ in range(16):
        mask = clean_mask(faces, mask, min_island=4, max_hole=TIDY_HOLE, adjacency=(fa, fb))
        parts = mask_components(faces, mask, (fa, fb))
        if len(parts) > 1:
            mask[:] = False
            mask[parts[0]] = True
        _loops, pinched = boundary_loops(faces, np.flatnonzero(mask))
        if not pinched:
            break
        for v in pinched:
            mask[users[v]] = True
    return mask


def band(points, loop, seam, next_id: int = 0, faces=()):
    """One row of faces tying the grid's border `loop` to the `seam` (vertex ids into `points`, both running the part's
    way round), with no new vertices: (faces, []), as `segment._bridge` returns.

    A face takes one edge from each side, two edges of the grid and one seam vertex (it fills a corner where the grid's
    border turns in), or two edges of the seam and one grid corner (a corner where it turns out). The choice is the
    cheapest walk round both, by dynamic programming over (grid vertex, seam vertex, ties at each so far): a grid
    vertex wants as many ties as it lacks edges for four, a seam vertex one per right angle its corner spans past the
    first, and each face counts by its shape. A
    triangle is taken only when the two counts differ by an odd number."""
    loop, seam = np.asarray(loop, dtype=np.int64), np.asarray(seam, dtype=np.int64)
    m, n = len(loop), len(seam)
    if m < 3 or n < 3:
        raise ValueError("the rows' border is too short to tie on")
    grid = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    at_grid = np.bincount(grid.reshape(-1), minlength=len(points))
    want_l = np.clip(3 - at_grid[loop], 0, 2)
    seam = np.roll(seam, -int(np.argmin(np.linalg.norm(points[seam] - points[loop[0]], axis=1))))
    # A seam vertex wants a tie per right angle past the first that the area's corner there spans: none at a sharp
    # corner, one on a straight run, two where the area's border turns back in. Measured round the nearest grid normal.
    up = np.zeros((len(points), 3))
    np.add.at(up, grid.reshape(-1), np.repeat(np.cross(points[grid[:, 2]] - points[grid[:, 0]], points[grid[:, 3]] - points[grid[:, 1]]), 4, axis=0))
    normal = up[loop[np.argmin(np.linalg.norm(points[seam][:, None] - points[loop][None], axis=2), axis=1)]]
    e1, e2 = points[seam] - points[np.roll(seam, 1)], points[np.roll(seam, -1)] - points[seam]
    turn = np.arctan2((np.cross(e1, e2) * normal).sum(axis=1) / np.maximum(np.linalg.norm(normal, axis=1), 1e-30), (e1 * e2).sum(axis=1))
    want_s = np.clip(np.round((np.pi - turn) / (0.5 * np.pi)) - 1, 0, 2).astype(np.int64).tolist()
    reach = BAND_REACH + abs(m - n) // 2
    ii = np.arange(m + 1)[:, None]
    jj = np.clip(np.round(ii * n / m).astype(np.int64) + np.arange(-reach, reach + 1)[None], 0, n)  # (m+1, w)

    def shape(*corners):  # the walk's faces run against the surface (they are turned round when made): one that doesn't is upside down
        c = [points[k] for k in np.broadcast_arrays(*corners)]
        q = quad_quality(np.stack(c, axis=-2))
        upside = (np.cross(c[2] - c[0], c[3] - c[1]) * up[np.broadcast_to(corners[0], q.shape)]).sum(axis=-1) > 0.0
        return SHAPE_COST * (1.0 - np.clip(q, -1.0, 1.0)) + np.where((q < POOR) | upside, BAD_COST, 0.0)

    cost_r = shape(loop[ii % m], loop[(ii + 1) % m], seam[(jj + 1) % n], seam[jj % n]).tolist()
    cost_a = shape(loop[ii % m], loop[(ii + 1) % m], loop[(ii + 2) % m], seam[jj % n]).tolist()
    cost_b = shape(loop[ii % m], seam[(jj + 2) % n], seam[(jj + 1) % n], seam[jj % n]).tolist()
    tri = SHAPE_COST * 1.5
    odd = (m - n) % 2 == 1
    col = {}
    for i in range(m + 1):
        for k, j in enumerate(jj[i].tolist()):
            col.setdefault((i, j), k)
    want = want_l.tolist()

    def pen_l(i, c):
        return POLE_COST * (c - want[i % m]) ** 2

    def pen_s(j, c):
        return SEAM_POLE_COST * (c - want_s[j % n]) ** 2

    best = {(0, 0, 1, 1, 0): (0.0, None, None)}
    levels: dict = {0: [(0, 0, 1, 1, 0)]}
    for level in range(m + n + 1):
        for state in levels.pop(level, []):
            i, j, ci, cj, t = state
            k = col.get((i, j))
            if k is None:
                continue
            base = best[state][0]
            moves = []
            if i < m and j < n:
                moves.append(((i + 1, j + 1, 1, 1, t), pen_l(i, ci) + pen_s(j, cj) + cost_r[i][k], (loop[i % m], loop[(i + 1) % m], seam[(j + 1) % n], seam[j % n])))
            if i + 2 <= m:
                moves.append(((i + 2, j, 1, min(cj + 1, 3), t), pen_l(i, ci) + pen_l(i + 1, 0) + cost_a[i][k], (loop[i % m], loop[(i + 1) % m], loop[(i + 2) % m], seam[j % n])))
            if j + 2 <= n:
                moves.append(((i, j + 2, min(ci + 1, 3), 1, t), pen_s(j, cj) + pen_s(j + 1, 0) + cost_b[i][k], (loop[i % m], seam[(j + 2) % n], seam[(j + 1) % n], seam[j % n])))
            if odd and not t:
                if i < m:
                    moves.append(((i + 1, j, 1, min(cj + 1, 3), 1), pen_l(i, ci) + tri, (loop[i % m], loop[(i + 1) % m], seam[j % n])))
                if j < n:
                    moves.append(((i, j + 1, min(ci + 1, 3), 1, 1), pen_s(j, cj) + tri, (loop[i % m], seam[(j + 1) % n], seam[j % n])))
            for nxt, cost, face in moves:
                if (nxt[0], nxt[1]) not in col:
                    continue
                total = base + cost
                old = best.get(nxt)
                if old is None:
                    levels.setdefault(nxt[0] + nxt[1], []).append(nxt)
                if old is None or total < old[0]:
                    best[nxt] = (total, state, face)
    ends = [k for k in best if k[0] == m and k[1] == n]
    if not ends:
        raise ValueError("the rows' border and the seam are too far apart to tie together")
    state = min(ends, key=lambda k: best[k][0])
    made = []
    while best[state][1] is not None:
        made.append(tuple(int(v) for v in best[state][2][::-1]))  # the loops run the part's way; the tie faces run back
        state = best[state][1]
    return made[::-1], []


__all__ = ["band", "direction", "ease_seam", "folded", "grid_quads", "tidy_area"]
