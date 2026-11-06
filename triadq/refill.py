# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Replace patches full of poles with one plain grid, keeping every border vertex so neighbours still fit.

Any disk of quads has an even border of 2(a + b) edges, so an a-by-b grid always fits if its four corners are picked
around the border. Picking them where the outside already has an extra edge leaves the border vertices regular.
Filling a patch from a pattern chosen by its border is the idea of Takayama et al. 2014 and Marcias et al. 2015.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .tidy import _corner, _len, _shape, _sub, tidy
from .types import MeshData


@dataclass(slots=True)
class RefillOptions:
    passes: int = 2
    min_poles: int = 2  # corners of a seed quad that must be poles: most quads with one are not worth a try
    tries_per_quad: float = 0.5  # patches looked at per pass, worst first, so a huge mesh stays quick
    sizes: tuple[int, ...] = (6, 12, 24, 48)  # quads in a patch to try replacing
    sharp_deg: float = 30.0  # a patch stops at edges this sharp, so creases stay
    fit_share: float = 0.015  # a grid may leave the old surface by this share of the patch's size
    turn_deg: float = 25.0  # a new quad may differ this much from the old surface under it
    fold_deg: float = 25.0
    min_corner_deg: float = 25.0
    max_corner_deg: float = 165.0
    most_aspect: float = 5.0
    most_growth: float = 1.1  # a grid may use this many times the quads it replaces
    tries: int = 2  # corner choices to test against the surface, per patch


def _nearest_triangle(q: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point, (distance to the nearest triangle, its index). Points times triangles, so keep both small."""

    ab, ac = b - a, c - a
    ap = q[:, None] - a
    d1, d2 = (ab * ap).sum(axis=2), (ac * ap).sum(axis=2)
    d00, d01, d11 = (ab * ab).sum(axis=1), (ab * ac).sum(axis=1), (ac * ac).sum(axis=1)
    denominator = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    v = np.clip((d11 * d1 - d01 * d2) / denominator, 0.0, 1.0)
    w = np.clip((d00 * d2 - d01 * d1) / denominator, 0.0, 1.0)
    total = np.maximum(v + w, 1.0)
    v, w = v / total, w / total  # Limitation: clamped, not the true closest point; only ever overestimates a distance
    nearest = a + v[..., None] * ab + w[..., None] * ac
    distance = np.linalg.norm(q[:, None] - nearest, axis=2)
    at = np.argmin(distance, axis=1)
    return distance[np.arange(len(q)), at], at


def grid_points(sides: list[np.ndarray]) -> np.ndarray:
    """Transfinite interpolation of four polylines running around a patch: (a + 1, b + 1, 3) points."""

    bottom, right, top, left = sides[0], sides[1], sides[2][::-1], sides[3][::-1]
    a = len(bottom) - 1

    def along(line: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(line[1:] - line[:-1], axis=1)
        total = float(lengths.sum())
        return np.concatenate(([0.0], np.cumsum(lengths) / total)) if total > 1e-30 else np.linspace(0.0, 1.0, len(line))

    u = 0.5 * (along(bottom) + along(top))[:, None, None]
    v = 0.5 * (along(left) + along(right))[None, :, None]
    return (
        (1.0 - v) * bottom[:, None] + v * top[:, None] + (1.0 - u) * left[None] + u * right[None]
        - (1.0 - u) * (1.0 - v) * bottom[0] - u * (1.0 - v) * bottom[a] - (1.0 - u) * v * top[0] - u * v * top[a]
    )


class _Pass:
    """One sweep over the mesh. Patches never overlap or touch within a sweep, so the counts stay true."""

    def __init__(self, mesh: MeshData, options: RefillOptions, cost):
        self.options, self.cost = options, cost
        self.points, self.faces, self.kinds = mesh.vertices, mesh.faces, mesh.face_materials
        self.edge_faces: dict[tuple[int, int], list[int]] = {}
        self.users: list[list[int]] = [[] for _ in range(len(self.points))]
        for index, face in enumerate(self.faces):
            for k, a in enumerate(face):
                b = face[(k + 1) % len(face)]
                self.edge_faces.setdefault((a, b) if a < b else (b, a), []).append(index)
                self.users[a].append(index)
        self.valence = np.zeros(len(self.points), dtype=np.int64)
        self.border = np.zeros(len(self.points), dtype=bool)
        for (a, b), who in self.edge_faces.items():
            self.valence[a] += 1
            self.valence[b] += 1
            if len(who) != 2:
                self.border[a] = self.border[b] = True
        self.is_quad = np.asarray([len(f) == 4 for f in self.faces], dtype=bool)
        quads = np.asarray([f if len(f) == 4 else (f[0], f[1], f[2], f[2]) for f in self.faces], dtype=np.int64).reshape(-1, 4)
        normals = np.cross(self.points[quads[:, 2]] - self.points[quads[:, 0]], self.points[quads[:, 3]] - self.points[quads[:, 1]])
        self.normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        self.badness = ((self.valence != 4) & ~self.border)[quads].sum(axis=1) * self.is_quad
        self.claimed = np.zeros(len(self.faces), dtype=bool)
        self.cos_sharp = math.cos(math.radians(options.sharp_deg))
        self.cos_turn = math.cos(math.radians(options.turn_deg))
        self.limits = (math.cos(math.radians(options.min_corner_deg)), math.cos(math.radians(options.max_corner_deg)), math.cos(math.radians(options.fold_deg)))

    def grow(self, seed: int) -> list[list[int]]:
        """The seed's patch at each size, stopping at sharp edges, other materials and claimed faces."""

        faces, wanted = self.faces, list(self.options.sizes)
        region, seen, frontier, out = [seed], {seed}, [seed], []
        while frontier and wanted:
            following = []
            for f in frontier:
                face = faces[f]
                for k, a in enumerate(face):
                    b = face[(k + 1) % 4]
                    who = self.edge_faces[(a, b) if a < b else (b, a)]
                    if len(who) != 2:
                        continue
                    g = who[0] if who[1] == f else who[1]
                    if g in seen or self.claimed[g] or not self.is_quad[g] or self.kinds[g] != self.kinds[seed]:
                        continue
                    if float(self.normals[f] @ self.normals[g]) < self.cos_sharp:
                        continue
                    seen.add(g)
                    following.append(g)
                    region.append(g)
            frontier = following
            while wanted and len(region) >= wanted[0]:
                out.append(list(region))
                wanted.pop(0)
        if wanted and len(region) >= 4 and (not out or len(region) > len(out[-1])):
            out.append(list(region))
        return out

    def loop_of(self, region: list[int]) -> list[int] | None:
        """The border of a patch as one simple loop, or None unless the patch is a disk."""

        faces = self.faces
        halves: set[tuple[int, int]] = set()
        for f in region:
            face = faces[f]
            for k, a in enumerate(face):
                pair = (a, face[(k + 1) % 4])
                if pair in halves:
                    return None
                halves.add(pair)
        following: dict[int, int] = {}
        for a, b in halves:
            if (b, a) not in halves:
                if a in following:
                    return None  # pinched
                following[a] = b
        if len(following) < 6 or len(following) % 2:
            return None
        start = next(iter(following))
        loop, at = [start], following[start]
        while at != start:
            loop.append(at)
            if at not in following or len(loop) > len(following):
                return None
            at = following[at]
        if len(loop) != len(following):
            return None
        vertices = {v for f in region for v in faces[f]}
        edges = {(a, b) if a < b else (b, a) for a, b in halves}
        return loop if len(vertices) - len(edges) + len(region) == 1 else None

    def attempt(self, region: list[int]):
        """(cost saved, walk of the border from the first corner, a, grid), or None."""

        loop = self.loop_of(region)
        if loop is None:
            return None
        faces, users, valence, border, cost, options = self.faces, self.users, self.valence, self.border, self.cost, self.options
        inside = {v for f in region for v in faces[f]} - set(loop)
        if any(border[v] for v in inside):
            return None
        members = set(region)
        m, half = len(loop), len(loop) // 2
        before = sum(cost(int(valence[v]), False, int(valence[v])) for v in inside)
        side_cost, corner_extra = [], []
        for v in loop:
            within = sum(1 for f in users[v] if f in members)
            outside = len(users[v]) - within
            kept = int(valence[v]) - (within - 1)  # the edges running into the patch go away
            on_border = bool(border[v])
            before += cost(int(valence[v]), on_border, len(users[v]))
            as_side = cost(kept + 1, on_border, outside + 2)
            as_corner = cost(kept, on_border, outside + 1) if kept >= (2 if on_border else 3) else 1e6
            side_cost.append(as_side)
            corner_extra.append(as_corner - as_side)
        base = sum(side_cost)
        ring = self.points[loop]
        out = np.roll(ring, -1, axis=0) - ring
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)
        turn = np.degrees(np.arccos(np.clip(-(out * np.roll(out, 1, axis=0)).sum(axis=1), -1.0, 1.0)))
        off_square = (np.abs(turn - 90.0) / 90.0).tolist()  # a good grid corner is a real corner
        choices = []
        for start in range(half):
            for a in range(1, half):
                if a * (half - a) > options.most_growth * len(region):
                    continue
                corners = (start, (start + a) % m, (start + half) % m, (start + half + a) % m)
                total = base + sum(corner_extra[c] for c in corners)
                if total < before - 0.5:
                    choices.append((total + 0.5 * sum(off_square[c] for c in corners), total, start, a))
        if not choices:
            return None
        choices.sort()
        old = self.points[np.asarray([faces[f] for f in region], dtype=np.int64)]
        tri_a = np.concatenate((old[:, 0], old[:, 0]))
        tri_b = np.concatenate((old[:, 1], old[:, 2]))
        tri_c = np.concatenate((old[:, 2], old[:, 3]))
        tri_n = np.cross(tri_b - tri_a, tri_c - tri_a)
        size = math.sqrt(max(0.5 * float(np.linalg.norm(tri_n, axis=1).sum()), 1e-30))
        tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-30)
        for _rank, total, start, a in choices[: options.tries]:
            b = half - a
            walk = [loop[(start + k) % m] for k in range(m + 1)]
            p = self.points
            grid = grid_points([p[walk[: a + 1]], p[walk[a : a + b + 1]], p[walk[a + b : 2 * a + b + 1]], p[walk[2 * a + b :]]])
            centres = 0.25 * (grid[:-1, :-1] + grid[1:, :-1] + grid[1:, 1:] + grid[:-1, 1:]).reshape(-1, 3)
            inner = grid[1:-1, 1:-1].reshape(-1, 3)
            if len(inner) and float(_nearest_triangle(inner, tri_a, tri_b, tri_c)[0].max()) > options.fit_share * size:
                continue
            distance, at = _nearest_triangle(centres, tri_a, tri_b, tri_c)
            if float(distance.max()) > options.fit_share * size:
                continue
            if self._grid_fine(grid, tri_n[at], a, b):
                return before - total, walk, a, grid
        return None

    def _grid_fine(self, grid, under, a: int, b: int) -> bool:
        for i in range(a):
            for j in range(b):
                corners = (tuple(grid[i, j]), tuple(grid[i + 1, j]), tuple(grid[i + 1, j + 1]), tuple(grid[i, j + 1]))
                n = _shape(*corners, *self.limits)
                if n is None:
                    return False
                below = under[i * b + j]
                if n[0] * below[0] + n[1] * below[1] + n[2] * below[2] < self.cos_turn:
                    return False
                across = _len(_sub(corners[1], corners[0])) + _len(_sub(corners[2], corners[3]))
                along = _len(_sub(corners[3], corners[0])) + _len(_sub(corners[2], corners[1]))
                if max(across, along) > self.options.most_aspect * max(min(across, along), 1e-30):
                    return False
        return True

    def run(self, counts: dict) -> MeshData | None:
        replaced = []
        new_points: list[np.ndarray] = []
        next_vertex = len(self.points)
        left = max(int(self.options.tries_per_quad * len(self.faces)), 500)
        for seed in np.argsort(-self.badness, kind="stable").tolist():
            if self.badness[seed] < self.options.min_poles:
                break
            if left <= 0:
                break
            if self.claimed[seed]:
                continue
            best = None
            for region in self.grow(seed):
                left -= 1
                found = self.attempt(region)
                if found is not None and (best is None or found[0] > best[0][0]):
                    best = (found, region)
            if best is None:
                continue
            (_saved, walk, a, grid), region = best
            b = (len(walk) - 1) // 2 - a
            table = np.full((a + 1, b + 1), -1, dtype=np.int64)
            border_cells = [(i, 0) for i in range(a)] + [(a, j) for j in range(b)] + [(i, b) for i in range(a, 0, -1)] + [(0, j) for j in range(b, 0, -1)]
            for (i, j), v in zip(border_cells, walk):
                table[i, j] = v
            for i in range(1, a):
                for j in range(1, b):
                    table[i, j] = next_vertex
                    next_vertex += 1
                    new_points.append(grid[i, j])
            fresh = [(int(table[i, j]), int(table[i + 1, j]), int(table[i + 1, j + 1]), int(table[i, j + 1])) for i in range(a) for j in range(b)]
            replaced.append((region, fresh, self.kinds[region[0]]))
            counts["patches"] += 1
            counts["quads_in"] += len(region)
            counts["quads_out"] += len(fresh)
            for v in {v for f in region for v in self.faces[f]}:  # everything touching it changed valence: next pass
                self.claimed[self.users[v]] = True
        if not replaced:
            return None
        gone = np.zeros(len(self.faces), dtype=bool)
        out_faces, out_kinds = [], []
        for region, fresh, kind in replaced:
            gone[region] = True
            out_faces += fresh
            out_kinds += [kind] * len(fresh)
        for f in np.flatnonzero(~gone).tolist():
            out_faces.append(self.faces[f])
            out_kinds.append(self.kinds[f])
        stacked = np.vstack([self.points] + [p[None] for p in new_points]) if new_points else self.points
        used = sorted({v for f in out_faces for v in f})
        remap = np.full(len(stacked), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return MeshData(vertices=stacked[used], faces=[tuple(remap[list(f)].tolist()) for f in out_faces], face_materials=out_kinds)


def refill(mesh: MeshData, options: RefillOptions | None = None, corner_cost=None) -> tuple[MeshData, dict]:
    """(mesh with messy patches replaced by grids, how many). Faces that aren't quads and creases stay."""

    options = options or RefillOptions()
    counts = {"patches": 0, "quads_in": 0, "quads_out": 0}
    current = mesh
    for _pass in range(options.passes):
        result = _Pass(current, options, corner_cost or _corner).run(counts)
        if result is None:
            break
        result.name = mesh.name
        current = result
    return current, counts


# Strictest last: Auto Segments tries each until the camera accepts the result.
LOOSE = RefillOptions(fit_share=0.08, turn_deg=45.0, min_corner_deg=10.0, max_corner_deg=175.0, fold_deg=29.0, most_aspect=12.0, most_growth=1.5)
MIDDLE = RefillOptions(fit_share=0.04, turn_deg=35.0, min_corner_deg=15.0, max_corner_deg=172.0, fold_deg=29.0, most_aspect=8.0)
STRICT = RefillOptions()


def polish(mesh: MeshData, options: RefillOptions = STRICT, corner_cost=None) -> tuple[MeshData, dict]:
    """Grids over the worst patches, then the small moves over what is left."""

    filled, counts = refill(mesh, options, corner_cost)
    moved, moves = tidy(filled, corner_cost=corner_cost)
    return moved, {**counts, **moves}


__all__ = ["LOOSE", "MIDDLE", "RefillOptions", "STRICT", "grid_points", "polish", "refill"]
