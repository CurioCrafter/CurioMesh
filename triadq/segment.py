# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Surface segments: split a connected mesh into parts a person would name, and join remade parts back on their seams.

Part boundaries run along concave creases (Hoffman and Singh's minima rule; Shamir's 2008 survey) and between thick and
thin parts (Shapira's shape diameter). Plain watershed over-segments (Mangan and Whitaker), so this keeps a merge tree:
small patches of faces, at a scale well above the noise, are merged cheapest first, and Segment Detail only picks how far
down the tree to cut. A marker watershed on the real faces then settles each boundary on the crease nearby.

Plain NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .engine import RemeshOptions, remesh_mesh
from .flatquad import pair_and_close
from .patch import (
    _arc_parameters,
    _half_edges,
    _labels,
    _Locator,
    _triangulate_component,
    _unique_edges,
    boundary_loops,
    circle_boundary,
    coons_grid,
    fit_to_surface,
    prepare_regions,
    quad_quality,
    remesh_region,
    tutte_embedding,
)
from .topology import parity_path_quad_closure
from .types import MeshData

try:  # the scorer and the grids are features of their own
    from . import patterns
except ImportError:
    patterns = None
try:
    from . import smoothgrid
except ImportError:
    smoothgrid = None

CELLS = 44  # patches across a loose piece's longest side: coarse enough to see the shape and not the scan noise
MIN_FACES = 200  # a loose piece with fewer faces stays one segment
CONVEX = 0.3  # a convex ridge counts this much against a concave crease of the same angle
BASE = 0.12  # what any border costs, so a smooth surface can still split at necks when asked for much detail
USUAL = 0.8  # share of the bending around a border that it takes for granted
CURVE_STEP = 0.8  # weight of flat meeting curved
NECK = 3.5  # border length over the root of the smaller side's area, where a border stops counting as a neck
THICK_STEP = 0.5  # weight of thick meeting thin, on log thickness
LEAST_CUT = 0.02  # the merge cost Segment Detail reaches at 1
MOST_CUT = 0.3  # and at 0
SMALLEST = 0.004  # of a loose piece's area: a segment smaller than this is noise, at any detail
SNAP_LEVELS = 16
SHEET_RATIO = 1.6  # how fast a row of an Animation sheet may gain or lose vertices
RING_SAMPLES = 128
RING_LEVELS = 96
RING_CHANGE = 0.18  # a row stays the same length until the surface wants this much more or less
RING_GRID = 12  # vertices round the last row at which a grid can finish a disk
SHEET_DETAIL = 0.5  # how much more of the sheet's rows a bumpy area gets, on a smooth one's 1
SHEET_EVEN_ROUNDS = 10  # rounds of evening out its flat map
SHEET_RELAX = 12  # passes settling a sheet on the surface
SHEET_GOOD = 70.0  # a sheet with this topology score needs no flow fill to compare it with


def face_links(faces):
    """Edge-adjacent face pairs with the edge's two vertices: (fa, fb, u, v)."""

    face_of, a, b = _half_edges(faces)
    if not len(a):
        empty = np.zeros(0, np.int64)
        return empty, empty, empty, empty
    span = int(max(a.max(), b.max())) + 1
    low, high = np.minimum(a, b), np.maximum(a, b)
    key = low * span + high
    order = np.argsort(key, kind="stable")
    key, face_of, low, high = key[order], face_of[order], low[order], high[order]
    first = np.concatenate(([True], key[1:] != key[:-1]))
    head = np.maximum.accumulate(np.where(first, np.arange(len(key)), 0))
    linked = ~first & (face_of != face_of[head])
    return face_of[head][linked], face_of[linked], low[linked], high[linked]


def face_frames(co, faces):
    """(unit normals, areas, centroids) of polygons of any size."""

    co = np.asarray(co, dtype=np.float64)
    face_of, a, b = _half_edges(faces)
    count = len(faces)
    cross = np.cross(co[a], co[b])
    vector = np.stack([np.bincount(face_of, cross[:, k], count) for k in range(3)], axis=1) * 0.5
    area = np.linalg.norm(vector, axis=1)
    size = np.maximum(np.bincount(face_of, minlength=count), 1)
    centroid = np.stack([np.bincount(face_of, co[a, k], count) for k in range(3)], axis=1) / size[:, None]
    return vector / np.maximum(area, 1e-30)[:, None], area, centroid


@dataclass(slots=True)
class Tree:
    """What cutting at any detail needs."""

    leaf: np.ndarray  # per face: its patch
    shell: np.ndarray  # per face: its loose piece
    merges: np.ndarray  # (kept patch, absorbed patch) in the order they merged
    costs: np.ndarray  # never falling, one per merge
    leaf_area: np.ndarray
    links: tuple  # face_links
    normals: np.ndarray
    centroids: np.ndarray
    areas: np.ndarray


def _smooth_normals(normals, areas, fa, fb, rounds: int):
    out = normals * areas[:, None]
    for _ in range(rounds):
        total = out.copy()
        for k in range(3):
            total[:, k] += np.bincount(fa, out[fb, k], len(out)) + np.bincount(fb, out[fa, k], len(out))
        out = total / np.maximum(np.linalg.norm(total, axis=1), 1e-30)[:, None] * areas[:, None]
    return out / np.maximum(np.linalg.norm(out, axis=1), 1e-30)[:, None]


def _patches(links, normals, centroids, shell, cells: int) -> np.ndarray:
    """Patches: connected faces sharing a cell of a grid and a rough facing direction. One sweep, even on millions of faces."""

    fa, fb = links[0], links[1]
    count = len(normals)
    low = np.full((shell.max() + 1, 3), np.inf)
    high = np.full((shell.max() + 1, 3), -np.inf)
    np.minimum.at(low, shell, centroids)
    np.maximum.at(high, shell, centroids)
    size = np.maximum((high - low).max(axis=1), 1e-30) / cells
    cell = np.floor((centroids - low[shell]) / size[shell][:, None]).astype(np.int64)
    axis = np.argmax(np.abs(normals), axis=1)
    facing = axis * 2 + (normals[np.arange(count), axis] > 0)
    key = cell[:, 0] + (cells + 1) * (cell[:, 1] + (cells + 1) * (cell[:, 2] + (cells + 1) * facing))
    same = (key[fa] == key[fb]) & (shell[fa] == shell[fb])
    return np.unique(_labels(count, fa[same], fb[same]), return_inverse=True)[1].reshape(-1)


def _merge_all(leaf, shell, links, co, normals, areas, centroids, log_thick, cancelled=None):
    """Merge cheapest first until every loose piece is one patch. Returns (merges, costs, leaf area)."""

    fa, fb, u, v = links
    count = int(leaf.max()) + 1 if len(leaf) else 0
    leaf_area = np.bincount(leaf, areas, count)
    weight = np.maximum(leaf_area, 1e-30)[:, None]
    n = np.stack([np.bincount(leaf, normals[:, k] * areas, count) for k in range(3)], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    c = np.stack([np.bincount(leaf, centroids[:, k] * areas, count) for k in range(3)], axis=1) / weight
    shell_of = np.zeros(count, dtype=np.int64)
    shell_of[leaf] = shell
    shell_area = np.bincount(shell, areas)
    across = leaf[fa] != leaf[fb]
    a, b = leaf[fa[across]], leaf[fb[across]]
    a, b = np.minimum(a, b), np.maximum(a, b)
    lengths = np.linalg.norm(co[u[across]] - co[v[across]], axis=1)
    pair, inverse = np.unique(a * count + b, return_inverse=True)
    pair_len = np.bincount(inverse.reshape(-1), lengths)
    pa, pb = pair // max(count, 1), pair % max(count, 1)
    angle = np.arccos(np.clip((n[pa] * n[pb]).sum(axis=1), -1.0, 1.0))
    concave = ((c[pb] - c[pa]) * (n[pa] - n[pb])).sum(axis=1) > 0.0
    ends = np.concatenate((pa, pb))
    # Per patch: how bent its surroundings are. Tells a flat plate from the round rim next to it.
    turned = np.bincount(ends, np.tile(angle * pair_len, 2), count) / np.maximum(np.bincount(ends, np.tile(pair_len, 2), count), 1e-30)
    near: list[dict] = [{} for _ in range(count)]
    for x, y, length, ang, dent in zip(pa.tolist(), pb.tolist(), pair_len.tolist(), angle.tolist(), concave.tolist()):
        row = [length, length * ang if dent else 0.0, 0.0 if dent else length * ang]
        near[x][y] = row
        near[y][x] = row
    area = leaf_area.tolist()
    curve = (turned * leaf_area).tolist()
    thick = (log_thick * leaf_area).tolist()
    whole = shell_area[shell_of].tolist()
    version = [0] * count

    def cost(i, j):
        row = near[i][j]
        ai, aj = max(area[i], 1e-30), max(area[j], 1e-30)
        inside_i, inside_j = curve[i] / ai, curve[j] / aj
        # A crease has to bend more than the surface around it does: on rock or hair every border bends.
        bend = max((row[1] + CONVEX * row[2]) / max(row[0], 1e-30) - USUAL * min(inside_i, inside_j), 0.0)
        step = CURVE_STEP * abs(inside_i - inside_j) + THICK_STEP * abs(thick[i] / ai - thick[j] / aj)
        small = min(ai, aj)
        # A short border for the size of the smaller side is a neck, however smooth. A patch on open surface has about 3.5.
        neck = 1.05 - min(row[0] / (NECK * small**0.5), 1.0)
        return (BASE + bend + step) * (small / max(whole[i], 1e-30)) ** 0.5 * neck

    heap = [(cost(x, y), x, y, 0, 0) for x, y in zip(pa.tolist(), pb.tolist())]
    heapq.heapify(heap)
    merges, costs = [], []
    top = 0.0
    while heap:
        price, i, j, vi, vj = heapq.heappop(heap)
        if vi != version[i] or vj != version[j] or j not in near[i]:
            continue
        if len(merges) % 512 == 0 and cancelled is not None and cancelled():
            raise ValueError("cancelled")
        top = max(top, price)
        if len(near[i]) < len(near[j]):
            i, j = j, i
        merges.append((i, j))
        costs.append(top)
        near[i].pop(j)
        near[j].pop(i)
        area[i] += area[j]
        curve[i] += curve[j]
        thick[i] += thick[j]
        version[i] += 1
        for k, row in near[j].items():
            del near[k][j]
            mine = near[i].get(k)
            if mine is None:
                near[i][k] = near[k][i] = row
            else:
                near[i][k] = near[k][i] = [mine[0] + row[0], mine[1] + row[1], mine[2] + row[2]]
        near[j] = {}
        for k in near[i]:
            heapq.heappush(heap, (cost(i, k), i, k, version[i], version[k]))
    return np.asarray(merges, dtype=np.int64).reshape(-1, 2), np.asarray(costs, dtype=np.float64), leaf_area


def build(co, faces, thickness=None, cells: int = CELLS, cancelled=None) -> Tree:
    """The merge tree of a mesh. `thickness(points, normals)` is the local thickness (0 = unknown), or None to go without."""

    co = np.asarray(co, dtype=np.float64)
    links = face_links(faces)
    fa, fb = links[0], links[1]
    normals, areas, centroids = face_frames(co, faces)
    shell = np.unique(_labels(len(faces), fa, fb), return_inverse=True)[1].reshape(-1)
    big = np.bincount(shell)[shell] >= MIN_FACES
    leaf = _patches(links, _smooth_normals(normals, areas, fa, fb, 2), centroids, shell, cells)
    # A small loose piece is one patch from the start: nothing to find, and a model can have thousands of them.
    leaf = np.where(big, leaf, leaf.max() + 1 + shell)
    leaf = np.unique(leaf, return_inverse=True)[1].reshape(-1)
    count = int(leaf.max()) + 1
    log_thick = np.zeros(count)
    if thickness is not None:
        weight = np.maximum(np.bincount(leaf, areas, count), 1e-30)
        n = np.stack([np.bincount(leaf, normals[:, k] * areas, count) for k in range(3)], axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
        c = np.stack([np.bincount(leaf, centroids[:, k] * areas, count) for k in range(3)], axis=1) / weight[:, None]
        measured = np.asarray(thickness(c, n), dtype=np.float64)
        known = measured > 0.0
        if known.any():
            log_thick[known] = np.log(measured[known])
            log_thick[~known] = float(np.median(log_thick[known]))
    merges, costs, leaf_area = _merge_all(leaf, shell, links, co, normals, areas, centroids, log_thick, cancelled)
    return Tree(leaf, shell, merges, costs, leaf_area, links, normals, centroids, areas)


def _find(parent, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _flatten(parent: np.ndarray) -> np.ndarray:
    while True:
        jumped = parent[parent]
        if np.array_equal(jumped, parent):
            return parent
        parent = jumped


def cut(tree: Tree, detail: float) -> np.ndarray:
    """Per patch: its segment at this Segment Detail (0 to 1). Instant, so the slider can be live."""

    t = min(max(float(detail), 0.0), 1.0)
    limit = MOST_CUT * (LEAST_CUT / MOST_CUT) ** t
    parent = np.arange(len(tree.leaf_area))
    done = int(np.searchsorted(tree.costs, limit, side="right"))
    parent[tree.merges[:done, 1][::-1]] = tree.merges[:done, 0][::-1]
    parent = _flatten(parent)
    shell_of = np.zeros(len(parent), dtype=np.int64)
    shell_of[tree.leaf] = tree.shell
    whole = np.bincount(shell_of, tree.leaf_area)
    area = np.bincount(parent, tree.leaf_area, len(parent))
    # A segment under the smallest size joins whichever neighbour the tree merges it with first.
    for a, b in tree.merges[done:].tolist():
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb and min(area[ra], area[rb]) < SMALLEST * whole[shell_of[ra]]:
            parent[rb] = ra
            area[ra] += area[rb]
    return _flatten(parent)


def snap(tree: Tree, face_labels: np.ndarray, rounds: int = 6, cancelled=None) -> np.ndarray:
    """Marker watershed on the real faces: a band around each boundary floods back in from both sides, creases last."""

    fa, fb = tree.links[0], tree.links[1]
    labels = np.asarray(face_labels, dtype=np.int64).copy()
    count = len(labels)
    band = np.zeros(count, dtype=bool)
    differ = labels[fa] != labels[fb]
    band[fa[differ]] = True
    band[fb[differ]] = True
    if not band.any():
        return labels
    for _ in range(rounds):
        grown = band.copy()
        grown[fa[band[fb]]] = True
        grown[fb[band[fa]]] = True
        band = grown
    # A segment with no face left outside the band keeps its own faces as the marker.
    present = np.bincount(labels[~band], minlength=int(labels.max()) + 1) > 0
    band &= present[labels]
    smooth = _smooth_normals(tree.normals, tree.areas, fa, fb, 1)
    angle = np.arccos(np.clip((smooth[fa] * smooth[fb]).sum(axis=1), -1.0, 1.0))
    concave = ((tree.centroids[fb] - tree.centroids[fa]) * (smooth[fa] - smooth[fb])).sum(axis=1) > 0.0
    strength = np.where(concave, angle, CONVEX * angle)
    touching = band[fa] | band[fb]
    scale = max(float(np.percentile(strength[touching], 98)), 1e-6) if touching.any() else 1.0
    level = np.minimum((strength / scale * SNAP_LEVELS).astype(np.int64), SNAP_LEVELS)
    both = np.concatenate((fa, fb))
    order = np.argsort(both, kind="stable")
    other = np.concatenate((fb, fa))[order].tolist()
    levels = np.concatenate((level, level))[order].tolist()
    start = np.searchsorted(both[order], np.arange(count + 1)).tolist()
    labels[band] = -1
    out = labels.tolist()
    heap: list = []
    serial = 0
    for f in np.unique(np.concatenate((fa[band[fb] & ~band[fa]], fb[band[fa] & ~band[fb]]))).tolist():
        for k in range(start[f], start[f + 1]):
            g = other[k]
            if out[g] < 0:
                heap.append((levels[k], serial, g, out[f]))
                serial += 1
    heapq.heapify(heap)
    popped = 0
    while heap:
        _level, _serial, f, label = heapq.heappop(heap)
        if out[f] >= 0:
            continue
        out[f] = label
        popped += 1
        if popped % 4096 == 0 and cancelled is not None and cancelled():
            raise ValueError("cancelled")
        for k in range(start[f], start[f + 1]):
            g = other[k]
            if out[g] < 0:
                heapq.heappush(heap, (levels[k], serial, g, label))
                serial += 1
    result = np.asarray(out, dtype=np.int64)
    missing = result < 0  # a band cut off from every marker keeps what it had
    result[missing] = np.asarray(face_labels, dtype=np.int64)[missing]
    return result


def one_piece_each(labels, fa, fb, smallest: int = 4) -> np.ndarray:
    """Keep each label's largest connected piece. Its other pieces, and pieces under `smallest` faces, join a neighbour."""

    labels = np.asarray(labels, dtype=np.int64).copy()
    for _ in range(8):
        same = labels[fa] == labels[fb]
        piece = np.unique(_labels(len(labels), fa[same], fb[same]), return_inverse=True)[1].reshape(-1)
        size = np.bincount(piece)
        label_of = np.zeros(len(size), dtype=np.int64)
        label_of[piece] = labels
        winner = np.zeros(int(labels.max()) + 1, dtype=np.int64)
        by_size = np.argsort(size, kind="stable")
        winner[label_of[by_size]] = by_size  # the biggest piece of each label is written last
        extra = (piece != winner[labels]) | ((size[piece] < smallest) & (size[winner[labels]] >= smallest))
        crossing = ~same & (extra[fa] ^ extra[fb]) if extra.any() else np.zeros(0, dtype=bool)
        if not crossing.any():
            return labels
        fx = np.where(extra[fa[crossing]], fa[crossing], fb[crossing])
        fy = np.where(extra[fa[crossing]], fb[crossing], fa[crossing])
        pairs, counts = np.unique(np.stack((piece[fx], labels[fy]), axis=1), axis=0, return_counts=True)
        choice = np.full(len(size), -1, dtype=np.int64)
        for k in np.argsort(counts, kind="stable").tolist():  # the label each piece touches most is written last
            choice[pairs[k, 0]] = pairs[k, 1]
        moved = choice[piece] >= 0
        labels[moved] = choice[piece][moved]
    return labels


def tidy(faces, labels, links, rounds: int = 3, smallest: int = 4) -> np.ndarray:
    """Straighten jagged borders and give every segment one connected piece."""

    fa, fb = links[0], links[1]
    labels = np.asarray(labels, dtype=np.int64).copy()
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    for _ in range(rounds):  # a face with all edges but one on other segments changes sides
        differ = labels[fa] != labels[fb]
        ends = np.concatenate((fa[differ], fb[differ]))
        others = np.concatenate((labels[fb[differ]], labels[fa[differ]]))
        outside = np.bincount(ends, minlength=len(labels))
        spike = outside[ends] >= sizes[ends] - 1
        if not spike.any():
            break
        labels[ends[spike]] = others[spike]
    return one_piece_each(labels, fa, fb, smallest)


def dense(labels) -> np.ndarray:
    """Labels renumbered 0..n-1, biggest first."""

    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    rank = np.empty(len(unique), dtype=np.int64)
    rank[np.argsort(-counts, kind="stable")] = np.arange(len(unique))
    return rank[inverse.reshape(-1)]


def find(tree: Tree, faces, detail: float, cancelled=None) -> np.ndarray:
    """Per face: its segment at this detail, borders settled on the creases."""

    parent = cut(tree, detail)
    labels = snap(tree, parent[tree.leaf], cancelled=cancelled)
    labels = even_borders(faces, labels, _smooth_normals(tree.normals, tree.areas, tree.links[0], tree.links[1], 1), tree.areas)
    return dense(tidy(faces, labels, tree.links))


def even_borders(faces, labels, normals, areas, rounds: int = 2, turned: float = 0.5) -> np.ndarray:
    """Round off the wiggles a border has where no crease holds it. Each face near a border votes with the faces around its vertices; a face turned away has no vote, so creases stay."""

    labels = np.asarray(labels, dtype=np.int64).copy()
    face_of, a, _b = _half_edges(faces)
    if not len(a):
        return labels
    order = np.argsort(a, kind="stable")
    around, corners = face_of[order], a[order]
    start = np.searchsorted(corners, np.arange(int(corners.max()) + 2))
    corner_start = np.searchsorted(face_of, np.arange(len(faces) + 1))
    for _ in range(rounds):
        low = np.full(len(start) - 1, np.iinfo(np.int64).max)
        high = np.full(len(start) - 1, -1)
        np.minimum.at(low, corners, labels[around])
        np.maximum.at(high, corners, labels[around])
        near = np.unique(face_of[(low != high)[a]])
        changes = []
        for f in near.tolist():
            votes: dict[int, float] = {}
            for v in a[corner_start[f] : corner_start[f + 1]].tolist():
                for g in around[start[v] : start[v + 1]].tolist():
                    if float(normals[f] @ normals[g]) > turned:
                        votes[int(labels[g])] = votes.get(int(labels[g]), 0.0) + float(areas[g])
            if votes:
                best = max(votes, key=votes.get)
                if best != labels[f]:
                    changes.append((f, best))
        if not changes:
            break
        for f, best in changes:
            labels[f] = best
    return labels


def open_edges(faces) -> int:
    """How many edges have a face on one side only. A T-junction or a hole adds some."""

    _face, a, b = _half_edges(faces)
    if not len(a):
        return 0
    span = int(max(a.max(), b.max())) + 1
    counts = np.unique(np.minimum(a, b) * span + np.maximum(a, b), return_counts=True)[1]
    return int((counts == 1).sum())


def weld_border(points, faces):
    """Join vertices at exactly the same place on open edges: the seam of a part put back. Returns (points, faces); unused vertices are dropped."""

    points = np.asarray(points, dtype=np.float64)
    _face, a, b = _half_edges(faces)
    span = len(points)
    key = np.minimum(a, b) * span + np.maximum(a, b)
    _u, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    single = counts[inverse.reshape(-1)] == 1
    border = np.unique(np.concatenate((a[single], b[single])))
    target = np.arange(span)
    if len(border):
        _same, first, back = np.unique(points[border], axis=0, return_index=True, return_inverse=True)
        target[border] = border[first[back.reshape(-1)]]
    used, compact = np.unique(target[a], return_inverse=True)
    compact = compact.reshape(-1)
    out, at = [], 0
    for f in faces:
        out.append(tuple(int(v) for v in compact[at : at + len(f)]))
        at += len(f)
    return points[used], out


def side_of_line(points2d, stroke) -> np.ndarray:
    """True for points on the left of a drawn line. Its first and last pieces run on for ever, so the line cuts the whole view."""

    points2d = np.asarray(points2d, dtype=np.float64)
    stroke = np.asarray(stroke, dtype=np.float64)
    start, step = stroke[:-1], np.diff(stroke, axis=0)
    long = np.maximum((step * step).sum(axis=1), 1e-30)
    low = np.where(np.arange(len(step)) == 0, -np.inf, 0.0)
    high = np.where(np.arange(len(step)) == len(step) - 1, np.inf, 1.0)
    out = np.zeros(len(points2d), dtype=bool)
    for at in range(0, len(points2d), 4096):
        chunk = points2d[at : at + 4096, None, :] - start[None]
        t = np.clip((chunk * step[None]).sum(axis=2) / long[None], low, high)
        gap = chunk - t[..., None] * step[None]
        nearest = np.argmin((gap * gap).sum(axis=2), axis=1)
        rows = np.arange(len(nearest))
        out[at : at + 4096] = step[nearest, 0] * chunk[rows, nearest, 1] - step[nearest, 1] * chunk[rows, nearest, 0] > 0.0
    return out


def _nearest_on_cloud(points, cloud, block: int = 512):
    """(index, distance) of the nearest cloud point to each point."""

    index = np.zeros(len(points), dtype=np.int64)
    dist = np.zeros(len(points))
    for at in range(0, len(points), block):
        gap = points[at : at + block, None, :] - cloud[None]
        square = (gap * gap).sum(axis=2)
        index[at : at + block] = np.argmin(square, axis=1)
        dist[at : at + block] = np.sqrt(square[np.arange(square.shape[0]), index[at : at + block]])
    return index, dist


def seam_samples(co, links, labels, most: int = 30_000) -> np.ndarray:
    """Points along the borders between segments of a mesh."""

    fa, fb, u, v = links
    across = labels[fa] != labels[fb]
    a, b = co[u[across]], co[v[across]]
    if not len(a):
        return np.zeros((0, 3))
    points = np.vstack((a, b, 0.5 * (a + b)))
    if len(points) > most:
        points = points[:: len(points) // most + 1]
    return points


def settle_seams(co, faces, labels, samples, reach: float = 1.0):
    """Move the result's seam vertices onto the borders found on your mesh, where that is within `reach` edges and folds no quad."""

    co = np.asarray(co, dtype=np.float64)
    fa, fb, u, v = face_links(faces)
    across = labels[fa] != labels[fb]
    seam = np.unique(np.concatenate((u[across], v[across])))
    if not len(seam) or not len(samples):
        return co
    lengths = np.linalg.norm(co[u] - co[v], axis=1)
    ends = np.concatenate((u, v))
    edge = np.bincount(ends, np.tile(lengths, 2), len(co)) / np.maximum(np.bincount(ends, minlength=len(co)), 1)
    index, dist = _nearest_on_cloud(co[seam], samples)
    move = dist < reach * edge[seam]
    moved = co.copy()
    moved[seam[move]] = samples[index[move]]
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    if len(quads):  # any quad this spoils goes back, with all its vertices
        before, after = quad_quality(co[quads]), quad_quality(moved[quads])
        for _ in range(8):
            bad = (after < 0.05) & (after < before)
            if not bad.any():
                break
            moved[quads[bad].ravel()] = co[quads[bad].ravel()]
            after = quad_quality(moved[quads])
    return moved


# --------------------------------------------------------------------------- joining a remade part to its seam


def _arc(points):
    steps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)[:-1])) / max(float(steps.sum()), 1e-30)


def _match_loop(loop_xyz, seam_xyz):
    """Where along the seam each vertex of a loop lands, never going back. Returns (seam rotation, loop reversed?, positions)."""

    def run(a):
        start = int(np.argmin(np.linalg.norm(seam_xyz - a[0], axis=1)))
        b = np.roll(seam_xyz, -start, axis=0)
        ta, tb = _arc(a), _arc(b)
        at = np.zeros(len(a), dtype=np.int64)
        for k in range(1, len(a)):
            near = np.flatnonzero(np.abs(tb - ta[k]) < 0.15)
            if not len(near):
                near = np.asarray([min(int(np.searchsorted(tb, ta[k])), len(b) - 1)])
            at[k] = near[np.argmin(np.linalg.norm(b[near] - a[k], axis=1))]
        at = np.maximum.accumulate(at)
        return float(np.linalg.norm(b[at] - a, axis=1).mean()), start, at

    forward = run(loop_xyz)
    backward = run(loop_xyz[::-1])
    return (forward[1], False, forward[2]) if forward[0] <= backward[0] else (backward[1], True, backward[2])


def conform(points, faces, seams):
    """Make a remade part's open edges exactly the seams it has to join.

    `seams` are ordered loops of 3D points. Border vertices of the part merge onto the seam vertex nearest along the loop,
    and border edges too long for the seam get the seam vertices in between (the face behind them becomes a fan). Inside
    faces stay as they are. Returns (points, faces, source face per new face, vertex ids per seam). Raises ValueError if the
    borders do not match the seams or the join would not be a clean surface.
    """

    points = np.asarray(points, dtype=np.float64)
    loops, pinched = boundary_loops(faces, range(len(faces)))
    if pinched:
        raise ValueError("the part touches itself at a vertex")
    if len(loops) != len(seams):
        raise ValueError(f"{len(loops)} borders against {len(seams)} seams")
    if not seams:
        return points, [tuple(f) for f in faces], list(range(len(faces))), []
    left = list(range(len(loops)))
    paired = {}
    for s in sorted(range(len(seams)), key=lambda i: -len(seams[i])):  # each seam takes the border running along it
        seam_xyz = np.asarray(seams[s], dtype=np.float64)
        paired[s] = min((float(_nearest_on_cloud(points[loops[k]], seam_xyz)[1].mean()), k) for k in left)[1]
        left.remove(paired[s])
    table = [points]
    offset = len(points)
    target: dict[int, int] = {}
    extras: dict[tuple[int, int], list[int]] = {}
    seam_ids = []
    for s, seam_xyz in enumerate(seams):
        seam_xyz = np.asarray(seam_xyz, dtype=np.float64)
        if len(seam_xyz) < 3:
            raise ValueError("a seam needs three vertices")
        loop = loops[paired[s]]
        start, flipped, at = _match_loop(points[loop], seam_xyz)
        ids = offset + np.roll(np.arange(len(seam_xyz)), -start)
        order = loop[::-1] if flipped else loop
        for k, v in enumerate(order):
            target[v] = int(ids[at[k]])
            after = order[(k + 1) % len(order)]
            stop = int(at[k + 1]) if k + 1 < len(order) else len(seam_xyz)
            between = [int(ids[j]) for j in range(int(at[k]) + 1, stop)]
            if between:
                extras[(v, after)] = between
                extras[(after, v)] = between[::-1]
        table.append(seam_xyz)
        seam_ids.append(offset + np.arange(len(seam_xyz)))
        offset += len(seam_xyz)
    all_points = np.vstack(table)
    out: list = []
    origin = []
    seen: dict[tuple, int] = {}
    for index, face in enumerate(faces):
        polygon = []
        split = False
        for k, v in enumerate(face):
            polygon.append(target.get(v, v))
            more = extras.get((v, face[(k + 1) % len(face)]))
            if more:
                polygon += more
                split = True
        clean = [p for k, p in enumerate(polygon) if p != polygon[k - 1]]
        if len(clean) < 3:
            continue
        if split or len(clean) > 4:
            hub = next((k for k, p in enumerate(clean) if p < len(points)), 0)
            spun = clean[hub:] + clean[:hub]
            new = [(spun[0], spun[k], spun[k + 1]) for k in range(1, len(spun) - 1)]
        else:
            new = [tuple(clean)]
        for f in new:
            key = tuple(sorted(f))
            if key in seen:  # two faces collapsed onto each other: neither is surface any more
                out[seen[key]] = None
                continue
            seen[key] = len(out)
            out.append(f)
            origin.append(index)
    keep = [i for i, f in enumerate(out) if f is not None]
    out, origin = [out[i] for i in keep], [origin[i] for i in keep]
    uses: dict[tuple[int, int], int] = {}
    for f in out:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            key = (a, b) if a < b else (b, a)
            uses[key] = uses.get(key, 0) + 1
    if any(n > 2 for n in uses.values()):
        raise ValueError("the join would fold over itself")
    wanted = set()
    for ids in seam_ids:
        for a, b in zip(ids.tolist(), np.roll(ids, -1).tolist()):
            wanted.add((a, b) if a < b else (b, a))
    if {key for key, n in uses.items() if n == 1} != wanted:
        raise ValueError("the join left a gap")
    final_points, final = _local_mesh(all_points, out)
    used = np.unique(np.asarray([v for f in out for v in f], dtype=np.int64))
    return final_points, final, origin, [np.searchsorted(used, ids) for ids in seam_ids]


def pair_on_seams(points, faces, kinds, seams, *, quads: bool = True):
    """A reduced part joined to its seams: all quads when ParityPath can close it, every second seam vertex taking the others as frozen midpoints.

    Returns (points, faces, materials). Falls back to pairs and triangles on the full seam, and gives triangles with `quads` off.
    """

    pairing = RemeshOptions(target_faces=max(1, len(faces) // 2), seed_count=2, force_quads=False)
    if not seams and quads:
        made = pair_and_close(MeshData(points, faces, kinds), seed_count=2)[0]
        return np.asarray(made.vertices, dtype=np.float64), [tuple(f) for f in made.faces], list(made.face_materials)
    if quads and all(len(s) % 2 == 0 and len(s) >= 6 for s in seams):
        try:
            p, f, origin, ids = conform(points, faces, [np.asarray(s, dtype=np.float64)[::2] for s in seams])
            at = len(p)
            extra, frozen = [], {}
            for seam, loop in zip(seams, ids):
                for k, middle in enumerate(np.asarray(seam, dtype=np.float64)[1::2]):
                    a, b = int(loop[k]), int(loop[(k + 1) % len(loop)])
                    frozen[(a, b) if a < b else (b, a)] = at
                    extra.append(middle)
                    at += 1
            paired = remesh_mesh(MeshData(np.vstack((p, extra)), f, [kinds[i] for i in origin]), pairing)[0]
            closed = parity_path_quad_closure(paired, frozen_midpoints=frozen)
            return np.asarray(closed.vertices, dtype=np.float64), [tuple(x) for x in closed.faces], list(closed.face_materials)
        except ValueError:
            pass
    p, f, origin, _ids = conform(points, faces, seams)
    mats = [kinds[i] for i in origin]
    if quads:
        paired = remesh_mesh(MeshData(p, f, mats), pairing)[0]
        return np.asarray(paired.vertices, dtype=np.float64), [tuple(x) for x in paired.faces], list(paired.face_materials)
    return p, f, mats


# --------------------------------------------------------------------------- filling a part on its seams


def topology_score(points, faces) -> float | None:
    """0-100: how tidy the quads are against hand-made meshes. None with nothing to judge, or without the scorer."""

    if patterns is None or not any(len(f) == 4 for f in faces):
        return None
    try:
        return float(patterns.score(patterns.measure(points, faces))["score"])
    except (OSError, ValueError, KeyError, IndexError):  # OSError: the scorer's data file isn't there
        return None


def _local_mesh(co, faces):
    used, compact = np.unique(np.asarray([v for f in faces for v in f], dtype=np.int64), return_inverse=True)
    compact = compact.reshape(-1)
    out, at = [], 0
    for f in faces:
        out.append(tuple(int(v) for v in compact[at : at + len(f)]))
        at += len(f)
    return np.asarray(co)[used], out


def even_embedding(points, tris, border, circle, rounds: int = SHEET_EVEN_ROUNDS) -> np.ndarray:
    """A flat map of a disk where equal areas stay about equal, so a grid laid on it comes out even on the surface.

    A plain Tutte map squeezes the far end of a dome into the middle. Edge weights are lowered where the map stretches
    (Yoshizawa, Belyaev and Seidel 2004), keeping them symmetric and positive, so the map stays one-to-one. The least stretched map found wins.
    """

    edges = _unique_edges(tris)
    i, j = edges[:, 0], edges[:, 1]
    weights = 1.0 / np.maximum(np.linalg.norm(points[i] - points[j], axis=1), 1e-30)
    q = points[tris]
    area3 = np.maximum(0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1), 1e-30)
    normals = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-30)[:, None]
    fa, fb, _u, _v = face_links([tuple(t) for t in tris.tolist()])
    bend = np.arccos(np.clip((normals[fa] * normals[fb]).sum(axis=1), -1.0, 1.0))
    bent = np.bincount(np.concatenate((fa, fb)), np.tile(bend, 2), len(tris)) / 3.0
    bumpy = np.bincount(tris.ravel(), np.repeat(bent, 3), len(points)) / np.maximum(np.bincount(tris.ravel(), minlength=len(points)), 1)
    importance = 1.0 + SHEET_DETAIL * np.minimum(bumpy / max(float(np.median(bumpy)), 1e-6), 4.0)
    best, best_stretch, uv = None, np.inf, None
    for k in range(rounds):
        uv = tutte_embedding(len(points), tris, border, circle, points, weights, uv, 1e-7 if k < rounds - 1 else 1e-9)
        s, t = uv[tris][:, :, 0], uv[tris][:, :, 1]
        double = (s[:, 1] - s[:, 0]) * (t[:, 2] - t[:, 0]) - (s[:, 2] - s[:, 0]) * (t[:, 1] - t[:, 0])
        double = np.where(np.abs(double) < 1e-14, 1e-14, double)[:, None]
        ds = (q[:, 0] * (t[:, 1] - t[:, 2])[:, None] + q[:, 1] * (t[:, 2] - t[:, 0])[:, None] + q[:, 2] * (t[:, 0] - t[:, 1])[:, None]) / double
        dt = (q[:, 0] * (s[:, 2] - s[:, 1])[:, None] + q[:, 1] * (s[:, 0] - s[:, 2])[:, None] + q[:, 2] * (s[:, 1] - s[:, 0])[:, None]) / double
        gamma = 0.5 * ((ds * ds).sum(axis=1) + (dt * dt).sum(axis=1))
        stretch = float(np.sqrt((area3 * gamma).sum() / area3.sum()))
        if stretch < best_stretch:
            best, best_stretch = uv, stretch
        corner = np.repeat(area3, 3)
        weight_sum = np.bincount(tris.ravel(), corner, len(points))
        vertex = np.sqrt(np.bincount(tris.ravel(), corner * np.repeat(gamma, 3), len(points)) / np.maximum(weight_sum, 1e-30))
        vertex = np.maximum(vertex * importance / np.median(vertex * importance), 1e-3)
        weights = weights / np.sqrt(vertex[i] * vertex[j])
        weights /= float(np.median(weights))
    return best


def _ring_plan(uv, tris, points, edge: float, seam_count: int):
    """How rows should run inside a flattened disk: (radius of each row, vertices on it). Rows are `edge` apart on the surface and have quads about `edge` long.

    A row keeps its count until the surface asks for clearly more or fewer, so most rows run clean all the way round.
    """

    locator = _Locator(uv, tris)
    angles = np.linspace(0.0, 2.0 * np.pi, RING_SAMPLES, endpoint=False)
    radii = np.linspace(1.0, 0.02, RING_LEVELS)
    flat = np.stack((np.outer(radii, np.cos(angles)).ravel(), np.outer(radii, np.sin(angles)).ravel()), axis=1)
    tri, weights = locator.locate(flat)
    rings = np.einsum("ij,ijk->ik", weights, points[tris[tri]]).reshape(len(radii), len(angles), 3)
    around = np.linalg.norm(np.roll(rings, -1, axis=1) - rings, axis=2).sum(axis=1)
    inwards = np.concatenate(([0.0], np.cumsum(np.linalg.norm(rings[1:] - rings[:-1], axis=2).mean(axis=1))))
    rows = int(inwards[-1] / max(edge, 1e-30))
    out_radii, counts = [], []
    current = seam_count
    for k in range(1, rows + 1):
        radius = float(np.interp(k * edge, inwards, radii))
        wanted = float(np.interp(k * edge, inwards, around)) / max(edge, 1e-30)
        if abs(wanted - current) > RING_CHANGE * current:
            current = int(np.clip(wanted, current / SHEET_RATIO, current * SHEET_RATIO))
            current = max(8, current - current % 2)
        if radius < 0.03 or (current <= 2.0 * RING_GRID and k > 1 and wanted < current):
            break
        out_radii.append(radius)
        counts.append(current)
    if not counts:  # too small for a row: one row just inside the seam, and the grid
        out_radii, counts = [0.7], [max(8, seam_count)]
    counts[-1] = max(8, 4 * max(2, int(round(counts[-1] / 4.0))))
    if len(counts) > 1 and (counts[-2] - counts[-1]) % 2:
        counts[-1] += 2
    return out_radii, counts


def _sheet(co, faces, comp, edge, project):
    """Rows round a disk with a small grid at the far end, the even sheet an animator lays by hand: (points, quads, seam loop) or None.

    Four poles sit where the rows meet the grid, and a row gains or loses vertices through smoothgrid's strips only where the surface widens or narrows.
    """

    loops, pinched = boundary_loops(faces, comp)
    if smoothgrid is None or pinched or len(loops) != 1 or len(loops[0]) % 2 or len(loops[0]) < 8:
        return None
    vertex_ids, tris, _source, _uv = _triangulate_component(faces, comp)
    points = co[vertex_ids]
    loop = np.asarray(loops[0], dtype=np.int64)
    border = np.searchsorted(vertex_ids, loop)
    loop_xyz = points[border]
    count = len(border)
    circle = circle_boundary(loop_xyz)
    uv = even_embedding(points, tris, border, circle)
    ring_radii, ring_counts = _ring_plan(uv, tris, points, edge, count)
    counts = [count] + ring_counts
    rows = len(ring_counts)
    table = [np.arange(count)]
    phases = [_arc_parameters(loop_xyz)]
    at = count
    new_uv = []
    for r in range(rows):
        angle = 2.0 * np.pi * np.arange(counts[r + 1]) / counts[r + 1]
        new_uv.append(ring_radii[r] * np.stack((np.cos(angle), np.sin(angle)), axis=1))
        table.append(at + np.arange(counts[r + 1]))
        phases.append(np.arange(counts[r + 1]) / counts[r + 1])
        at += counts[r + 1]
    quads, middles = [], []
    for r in range(rows):
        if counts[r] >= counts[r + 1]:
            strip, added = smoothgrid._strip(table[r], table[r + 1], at + len(middles), phases[r], phases[r + 1])
        else:  # the row inside has more vertices: the same strip seen from inside, wound the other way
            strip, added = smoothgrid._strip(table[r + 1], table[r], at + len(middles), phases[r + 1], phases[r])
            strip = [q[::-1] for q in strip]
        quads += strip
        middles += added
    if middles:
        known = np.concatenate([circle] + new_uv)
        pairs = np.asarray(middles, dtype=np.int64)
        new_uv = new_uv + [0.5 * (known[pairs[:, 0]] + known[pairs[:, 1]])]
        at += len(middles)
    first = counts[-1] // 4
    nodes, ids = coons_grid(new_uv[rows - 1], (0, first, 2 * first, 3 * first))
    free = ids < 0
    ids[~free] = table[-1][ids[~free]]
    ids[free] = at + np.arange(int(free.sum()))
    new_uv.append(nodes[free])
    for i in range(ids.shape[0] - 1):
        for j in range(ids.shape[1] - 1):
            quads.append((int(ids[i, j]), int(ids[i + 1, j]), int(ids[i + 1, j + 1]), int(ids[i, j + 1])))
    new_uv = np.concatenate(new_uv)
    tri, weights = _Locator(uv, tris).locate(new_uv)
    xyz = np.vstack((loop_xyz, project(np.einsum("ij,ijk->ik", weights, points[tris[tri]]))))
    pinned = np.zeros(len(xyz), dtype=bool)
    pinned[:count] = True
    settled = fit_to_surface(xyz, quads, pinned, project, iterations=SHEET_RELAX, guard=False)
    if (quad_quality(settled[np.asarray(quads, dtype=np.int64)]) <= 0.0).any():  # the repair is slow, so only when something folded
        settled = fit_to_surface(settled, quads, pinned, project, iterations=1)
    return settled, quads, loop


def fill_part(co, faces, ids, goal: str, edge: float, project=None, topology=None, cancelled=None, passes: int = 3):
    """Rebuild a part of the result on its own seams. Returns (points, faces, warnings); seam vertices stay exactly where they were.

    Animation tries a row-and-grid sheet and a flow fill on each disk and keeps the better topology score. Subdivision is the flow fill.
    """

    co = np.asarray(co, dtype=np.float64)
    mask = np.zeros(len(faces), dtype=bool)
    mask[ids] = True
    comps, warnings = prepare_regions(co, faces, mask, min_island=1, max_hole=0, smooth=0, split_rings=True, frozen=~mask, topology=topology)
    done = np.zeros(len(faces), dtype=bool)
    all_points, out_faces = [co], []
    offset = len(co)
    for comp in comps:
        if cancelled is not None and cancelled():
            raise ValueError("cancelled")
        options = []
        if goal == "ANIMATION":
            grid = _sheet(co, faces, comp, edge, project or (lambda p: p))
            if grid is not None:
                xyz, quads, loop = grid
                new = xyz[len(loop) :]
                remap = np.concatenate((loop, offset + np.arange(len(new))))
                made = [tuple(int(remap[v]) for v in q) for q in quads]
                options.append((topology_score(*_local_mesh(np.vstack((co, new)), made)) or 0.0, new, made))
        good = options and options[0][0] >= SHEET_GOOD
        result = None if good else remesh_region(co, faces, comp, target_length=edge, strategy="FLOW", falloff=2.5 * edge, max_faces=40_000, cancelled=cancelled)
        if good:
            pass
        elif result.ok:
            pts = result.points
            if project is not None and passes > 0 and len(pts):
                seam = sorted({v for face in result.faces for v in face if v < result.base})
                local = {v: k for k, v in enumerate(seam)}
                both = np.vstack((co[seam], pts))
                plain = [tuple(local[v] if v < result.base else len(seam) + v - result.base for v in face) for face in result.faces]
                pts = fit_to_surface(both, plain, np.arange(len(both)) < len(seam), project, passes, 0.5)[len(seam) :]
            made = [tuple(v if v < result.base else offset + v - result.base for v in face) for face in result.faces]
            options.append((topology_score(*_local_mesh(np.vstack((co, pts)), made)) or 0.0, pts, made))
        elif not options:
            warnings.append(f"A piece of {len(comp)} faces stayed as it was: {result.message}")
        if options:
            _score, new, made = max(options, key=lambda row: row[0])
            all_points.append(new)
            out_faces += made
            offset += len(new)
            done[comp] = True
    out_faces += [faces[i] for i in np.flatnonzero(mask & ~done).tolist()]
    points, final = _local_mesh(np.vstack(all_points), out_faces)
    return points, final, warnings


def seam_loops(co, faces, ids) -> list[np.ndarray]:
    """A part's open borders in the result, as ordered loops of points. Raises ValueError where it touches itself."""

    loops, pinched = boundary_loops(faces, ids)
    if pinched:
        raise ValueError("the part touches itself at a vertex")
    return [np.asarray(co)[loop] for loop in loops]
