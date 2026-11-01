# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Smooth Grids: rounded areas come out as clean rows of quads.

Reducing and pairing triangles leaves a dome with scattered, slanted quads. This pass finds smooth doubly curved
areas on the source (how the normals spread tells a dome from a cylinder or a plate, sign changes tell it from rock), finds the same areas on the reduced mesh, and rebuilds each disk with one ring of quads along its border
and a four-sided grid inside: the classic way to fill a circle, four poles, every border vertex kept so neighbours
need no change. New vertices land on the source. The caller's camera check judges the whole result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .flatquad import _edge_faces, _face_frames
from .patch import (
    _Locator,
    _corner_angle_sums,
    _half_edges,
    _euler,
    _labels,
    boundary_loops,
    circle_boundary,
    clean_mask,
    coons_grid,
    face_adjacency,
    fit_to_surface,
    mask_components,
    quad_quality,
    tutte_embedding,
)
from .types import MeshData


@dataclass(slots=True)
class GridOptions:
    sharp_deg: float = 35.0  # an edge bent more than this ends a smooth area
    min_spread: float = 2e-4  # smallest eigenvalue of the normals around a face, as a share: 0 on a cylinder
    min_total_deg: float = 20.0  # total curvature an area needs, so nearly flat noise stays as it is
    min_agreement: float = 0.7  # edge length bending one way, minus the other way, as a share: rock fails this
    min_source_faces: int = 20
    min_border: int = 8  # border vertices a reduced area needs before a grid fits inside
    relax_rounds: int = 8
    spike_deg: float = 100.0  # a border corner sharper than this gets tidied away
    plain_quality: float = 0.25  # a grid without its ring must be this good, or the ring goes on
    max_added: float = 0.03  # all grids together may add this share of the mesh's quads
    max_error: float = 1.5  # a grid may stray from the source this many times what the reduced triangles did
    max_sag: float = 0.02  # or this share of the area's size, which no camera at this density would notice


@dataclass(slots=True)
class GridReport:
    built: int = 0
    faces_in: int = 0
    quads_out: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    loops: list[int] = field(default_factory=list)


def smooth_areas(vertices: np.ndarray, faces, materials=None, options: GridOptions | None = None) -> np.ndarray:
    """Area id per source face, -1 where the surface isn't a smooth doubly curved area."""

    options = options or GridOptions()
    count = len(vertices)
    normals, areas = _face_frames(vertices, faces)
    face_of, a, b = _half_edges(faces)
    low, high = np.minimum(a, b), np.maximum(a, b)
    key = low * (count + 1) + high
    order = np.argsort(key, kind="stable")
    uniq, start, uses = np.unique(key[order], return_index=True, return_counts=True)
    two = uses == 2
    fa, fb = face_of[order][start[two]], face_of[order][start[two] + 1]
    va, vb = low[order][start[two]], high[order][start[two]]
    centre = np.stack([np.bincount(face_of, vertices[a, axis], len(faces)) for axis in range(3)], axis=1)
    centre /= np.bincount(face_of, minlength=len(faces))[:, None]
    bend = np.arctan2(np.linalg.norm(np.cross(normals[fa], normals[fb]), axis=1), np.einsum("ij,ij->i", normals[fa], normals[fb]))
    convex = np.einsum("ij,ij->i", centre[fb] - centre[fa], normals[fa] - normals[fb]) >= 0.0
    signed = np.where(convex, bend, -bend)
    lengths = np.linalg.norm(vertices[va] - vertices[vb], axis=1)
    sharp = bend > math.radians(options.sharp_deg)
    if materials is not None:
        kinds = np.unique(np.asarray(materials), return_inverse=True)[1]
        sharp |= kinds[fa] != kinds[fb]

    triangles = np.asarray([(face[0], face[k], face[k + 1]) for face in faces for k in range(1, len(face) - 1)], dtype=np.int64).reshape(-1, 3)
    defect = 2.0 * math.pi - np.radians(_corner_angle_sums(vertices, triangles))
    soft = ~sharp

    # Normals around a face: one direction on a plate, a great circle on a cylinder, a patch of the sphere on a dome.
    # A neighbour counts by its size relative to the face, squared, so a small dome face at the end of a long cylinder face barely counts.
    tensor = areas[:, None, None] * np.einsum("ij,ik->ijk", normals, normals)
    for here, there in ((fa[soft], fb[soft]), (fb[soft], fa[soft])):
        weight = np.minimum(areas[here], areas[there]) ** 2 / np.maximum(areas[here], 1e-30)
        np.add.at(tensor, here, weight[:, None, None] * np.einsum("ij,ik->ijk", normals[there], normals[there]))
    spread = np.linalg.eigvalsh(tensor)
    mask = spread[:, 0] > options.min_spread * np.maximum(spread.sum(axis=1), 1e-30)
    support = np.zeros(len(faces))
    np.add.at(support, fa[soft], mask[fb[soft]])
    np.add.at(support, fb[soft], mask[fa[soft]])
    mask |= support >= 2  # close notches in the outline

    keep = soft & mask[fa] & mask[fb]
    labels = _labels(len(faces), fa[keep], fb[keep])
    labels[~mask] = -1
    out = np.full(len(faces), -1, dtype=np.int64)
    ids, inverse, members = np.unique(labels, return_inverse=True, return_counts=True)
    total_curvature = np.zeros(len(ids))
    faces_at = np.maximum(np.bincount(a, minlength=count), 1)
    np.add.at(total_curvature, inverse[face_of], (np.maximum(defect, 0.0) / faces_at)[a])  # a vertex shares out its defect
    pull = np.zeros(len(ids))
    span = np.zeros(len(ids))
    inner = keep
    np.add.at(pull, inverse[fa[inner]], (signed * lengths)[inner])
    np.add.at(span, inverse[fa[inner]], (bend * lengths)[inner])
    next_id = 0
    for row, label in enumerate(ids.tolist()):
        if label < 0 or members[row] < options.min_source_faces:
            continue
        if total_curvature[row] < math.radians(options.min_total_deg):
            continue
        if abs(pull[row]) < options.min_agreement * max(span[row], 1e-30):
            continue
        out[inverse == row] = next_id
        next_id += 1
    return out


def _tidy_border(points: np.ndarray, faces: list, mask: np.ndarray, free: np.ndarray, links: list[list[int]], sharp_deg: float, rounds: int = 6) -> np.ndarray:
    """Take off spikes and fill notches: a triangle with two border edges meeting at a sharp angle changes sides.

    Reduced triangles don't follow an area's outline exactly, and a zigzag border folds the quads next to it.
    `free` marks the faces no other area wants.
    """

    limit = math.cos(math.radians(sharp_deg))
    mask = mask.copy()
    for _ in range(rounds):
        candidates = set(np.flatnonzero(mask).tolist())
        candidates.update(n for f in list(candidates) for n in links[f])
        flips = []
        for face in sorted(candidates):
            polygon = faces[face]
            if len(polygon) != 3:
                continue
            inside = [n for n in links[face] if mask[n]]
            if mask[face] and len(inside) == 1:
                apex = next(v for v in polygon if v not in faces[inside[0]])
            elif not mask[face] and free[face] and len(inside) == 2:
                common = set(faces[inside[0]]) & set(faces[inside[1]]) & set(polygon)
                if len(common) != 1:
                    continue
                apex = common.pop()
            else:
                continue
            at = polygon.index(apex)
            u = points[polygon[(at + 1) % 3]] - points[apex]
            w = points[polygon[(at + 2) % 3]] - points[apex]
            if float(u @ w) > limit * float(np.linalg.norm(u) * np.linalg.norm(w)):
                flips.append(face)
        if not flips:
            break
        mask[flips] = ~mask[flips]
    return mask


def _split_border(vertices: list, faces: list, area_list: list, materials: list, area: int, loop: list[int], project) -> list[int] | None:
    """Split the longest border edge that can be split, so the border gets one vertex more and the face outside stays conforming."""

    users = _edge_faces(faces)
    points = np.asarray([vertices[v] for v in loop])
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    for k in np.argsort(-lengths).tolist():
        v, w = loop[k], loop[(k + 1) % len(loop)]
        here = users[(v, w) if v < w else (w, v)]
        inside = [f for f in here if area_list[f] == area]
        outside = [f for f in here if area_list[f] != area]
        if len(inside) != 1 or len(outside) > 1 or any(area_list[f] >= 0 for f in outside):
            continue  # a neighbour's border must not change once it has been counted
        if any(len(faces[f]) != 3 for f in inside + outside):
            continue
        new = len(vertices)
        vertices.append(project(0.5 * (vertices[v] + vertices[w])[None])[0])
        for face in inside + outside:
            tri = faces[face]
            at = next(i for i in range(3) if {tri[i], tri[(i + 1) % 3]} == {v, w})
            p, q, r = tri[at], tri[(at + 1) % 3], tri[(at + 2) % 3]
            faces[face] = (p, new, r)
            faces.append((new, q, r))
            area_list.append(area_list[face])
            materials.append(materials[face])
        return loop[: k + 1] + [new] + loop[k + 1 :]
    return None


def cap_fill(loop_xyz: np.ndarray, tris: np.ndarray, points: np.ndarray, border: np.ndarray, project, ring: bool = True, relax_rounds: int = 8):
    """A four-sided grid over a disk, with a ring of quads around it if `ring`: (xyz of loop + new vertices, quads).

    `points`, `tris` are the disk as it stands, with `border` its loop in the same order as `loop_xyz`. The disk is
    flattened onto a circle (Tutte), so the grid can't fold however the disk curves, and the grid's nodes are lifted
    back through it, then relaxed on the surface. Without the ring, the grid's corners flatten to 180 degrees on a
    round border; with it, they become four poles just inside.
    """

    count = len(loop_xyz)
    circle = circle_boundary(loop_xyz)
    uv = tutte_embedding(len(points), tris, border, circle, points)
    first = count // 4
    second = count // 2 - first
    steps = np.linalg.norm(np.roll(loop_xyz, -1, axis=0) - loop_xyz, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))

    def run(start: int, length: int) -> float:
        start %= count
        end = start + length
        return cumulative[end % count] + cumulative[-1] * (end // count) - cumulative[start]

    # Corners go where the four sides come out closest to equal.
    best, offset = math.inf, 0
    for start in range(first + second):
        spread = float(np.ptp([run(start, first), run(start + first, second), run(start + first + second, first), run(start + 2 * first + second, second)]))
        if spread < best:
            best, offset = spread, start
    corners = (offset, (offset + first) % count, (offset + first + second) % count, (offset + 2 * first + second) % count)
    inner = circle * (1.0 - 1.0 / (0.5 * first + 1.5)) if ring else circle
    nodes, ids = coons_grid(inner, corners)
    free = ids < 0
    base = count if ring else 0
    ids[~free] += base
    ids[free] = count + base + np.arange(int(free.sum()))
    new_uv = np.concatenate((inner, nodes[free])) if ring else nodes[free]
    quads = [(k, (k + 1) % count, count + (k + 1) % count, count + k) for k in range(count)] if ring else []
    for i in range(ids.shape[0] - 1):
        for j in range(ids.shape[1] - 1):
            quads.append((int(ids[i, j]), int(ids[i + 1, j]), int(ids[i + 1, j + 1]), int(ids[i, j + 1])))
    if not len(new_uv):
        return loop_xyz, quads
    tri, weights = _Locator(uv, tris).locate(new_uv)
    xyz = np.vstack((loop_xyz, project(np.einsum("ij,ijk->ik", weights, points[tris[tri]]))))
    pinned = np.zeros(len(xyz), dtype=bool)
    pinned[:count] = True
    return fit_to_surface(xyz, quads, pinned, project, iterations=relax_rounds), quads


def grid_areas(coarse: MeshData, labels: np.ndarray, nearest, options: GridOptions | None = None) -> tuple[MeshData, GridReport]:
    """Rebuild the reduced mesh's smooth areas as grids. Everything else passes through untouched.

    `labels` comes from `smooth_areas`; `nearest(points)` gives the closest source face and point per point.
    """

    options = options or GridOptions()
    report = GridReport()
    faces = [tuple(face) for face in coarse.faces]
    materials = list(coarse.face_materials)
    if not faces or int(labels.max(initial=-1)) < 0:
        return coarse, report
    vertices = [row for row in np.asarray(coarse.vertices, dtype=np.float64)]
    face_of, a, _b = _half_edges(faces)
    points = np.asarray(coarse.vertices, dtype=np.float64)
    centres = np.stack([np.bincount(face_of, points[a, axis], len(faces)) for axis in range(3)], axis=1)
    centres /= np.bincount(face_of, minlength=len(faces))[:, None]
    area_of = labels[nearest(centres)[0]]
    adjacency = face_adjacency(faces)

    def reject(reason: str) -> None:
        report.rejected[reason] = report.rejected.get(reason, 0) + 1

    links: list[list[int]] = [[] for _ in faces]
    for first, second in zip(*(side.tolist() for side in adjacency)):
        links[first].append(second)
        links[second].append(first)

    # One disk per area, small holes filled, the rest given back.
    final = np.full(len(faces), -1, dtype=np.int64)
    for area in np.unique(area_of[area_of >= 0]).tolist():
        mask = clean_mask(faces, area_of == area, min_island=1, max_hole=4, adjacency=adjacency)
        mask = _tidy_border(points, faces, mask, area_of < 0, links, options.spike_deg)
        parts = mask_components(faces, mask, adjacency)
        if not parts:
            continue
        loops, pinched = boundary_loops(faces, parts[0])
        if pinched or len(loops) != 1 or _euler(faces, parts[0]) != 1:
            reject("not a disk")
        elif len({materials[f] for f in parts[0].tolist()}) > 1:
            reject("mixed materials")
        elif len(loops[0]) < options.min_border:
            reject("border too short")
        else:
            final[parts[0]] = area
    area_list = final.tolist()

    def project(query: np.ndarray) -> np.ndarray:
        return nearest(query)[1]

    built = []
    for area in np.unique(final[final >= 0]).tolist():
        ids = [f for f, label in enumerate(area_list) if label == area]
        seam = boundary_loops(faces, ids)[0][0]
        if len(seam) % 2:  # a grid needs an even border
            seam = _split_border(vertices, faces, area_list, materials, area, seam, project)
            if seam is None:
                reject("odd border")
                continue
        ids = [f for f, label in enumerate(area_list) if label == area]
        used = sorted({v for f in ids for v in faces[f]})
        local = {v: k for k, v in enumerate(used)}
        here = np.asarray([vertices[v] for v in used])
        tris = np.asarray([(local[faces[f][0]], local[faces[f][k]], local[faces[f][k + 1]]) for f in ids for k in range(1, len(faces[f]) - 1)], dtype=np.int64)
        limit = max(options.max_error * float(np.linalg.norm(here[tris].mean(axis=1) - nearest(here[tris].mean(axis=1))[1], axis=1).max()), options.max_sag * float(np.linalg.norm(np.ptp(here, axis=0))))

        def fault(xyz, quads, floor: float = 0.0) -> str | None:
            quad_array = np.asarray(quads, dtype=np.int64)
            if quad_quality(xyz[quad_array]).min() <= floor:
                return "folded"
            centre = xyz[quad_array].mean(axis=1)
            if float(np.linalg.norm(centre - nearest(centre)[1], axis=1).max()) > limit + 1e-9:
                return "strays from the source"
            return None

        border = np.asarray([local[v] for v in seam], dtype=np.int64)
        xyz, quads = cap_fill(here[border], tris, here, border, project, False, options.relax_rounds)
        problem = fault(xyz, quads, options.plain_quality)
        if problem:
            xyz, quads = cap_fill(here[border], tris, here, border, project, True, options.relax_rounds)
            problem = fault(xyz, quads)
        if problem:
            reject(problem)
            continue
        built.append((area, materials[ids[0]], seam, xyz, quads, len(ids), len(quads) - len(tris) / 2.0, float(np.linalg.norm(np.ptp(here, axis=0)))))
    # Extra quads go to the biggest areas first, and only up to a share of the mesh.
    budget = options.max_added * len(faces) / 2.0
    kept = []
    for row in sorted(built, key=lambda row: -row[7]):
        if row[6] > 0.0 and row[6] > budget:
            reject("too many quads")
            continue
        budget -= max(row[6], 0.0)
        kept.append(row[:6])
    built = kept
    # Faces added while evening out a border stay triangles if their area then fails.
    gone = {area for area, *_rest in built}
    out_points = list(vertices)
    out_faces = [face for face, label in zip(faces, area_list) if label not in gone]
    out_materials = [material for material, label in zip(materials, area_list) if label not in gone]
    for _area, material, loop, xyz, quads, removed in built:
        offset = len(out_points) - len(loop)
        table = lambda v, offset=offset, loop=loop: loop[v] if v < len(loop) else v + offset  # noqa: E731
        out_points.extend(xyz[len(loop) :])
        out_faces.extend(tuple(table(v) for v in quad) for quad in quads)
        out_materials.extend([material] * len(quads))
        report.built += 1
        report.faces_in += removed
        report.quads_out += len(quads)
        report.loops.append(len(loop))
    if not built:
        return coarse, report
    stacked = np.asarray(out_points, dtype=np.float64)
    alive = np.zeros(len(stacked), dtype=bool)
    for face in out_faces:
        alive[list(face)] = True
    remap = np.cumsum(alive) - 1
    return MeshData(vertices=stacked[alive], faces=[tuple(remap[list(face)].tolist()) for face in out_faces], face_materials=out_materials, name=coarse.name), report


__all__ = ["GridOptions", "GridReport", "cap_fill", "grid_areas", "smooth_areas"]
