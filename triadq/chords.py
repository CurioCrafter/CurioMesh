# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Fewer quads and tidier ones at the same time: redraw messy patches as plain grids, then collapse by what the camera sees.

Walk once around a patch of quads: a corner that has k faces outside the patch wants 4 - k inside, which means a turn of
k - 2 quarter turns. Laid out on squared paper, a walk that closes without crossing itself encloses a plain grid with no
poles at all, joined to the outside without touching it, and every boundary corner that got its wish is regular as well.
`_closed_walk` finds the walk that disappoints the fewest corners, and `PictureJudge` says what the grid does to the view.

Removing whole strips (Daniels et al. 2008) isn't here on purpose: measured on these meshes there are about twenty strips
in all, most of them one huge strip crossing itself thousands of times, so almost nothing can be taken away that way.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .quadreduce import MIN_QUALITY, MOST_TURNED, QuadReducer, _cross, _dot, _normal, _quality, _sub
from .topology import _turbo
from .types import MeshData

SMOOTH_COS = math.cos(math.radians(45.0))  # a corner with faces further apart than this sits on an edge of the shape
LONGEST_WALK = 44
SMALL_BOX = 64  # pixels: up to this, draw a triangle in plain Python
STEPS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_SHIFTS = (
    ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
    ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
)  # per heading, the slices of the grid a step lands in and comes from
TURNS = (1, 0, -1)  # left, straight on, right: 1, 2 or 3 faces inside the patch
NEW_QUALITY = 0.05
NEW_TURNED = math.cos(math.radians(20.0))
MOST_LEFT = 1.5  # a redrawn patch may leave the old surface by this many of the reducer's `dissolve_limit`
QUARTER = math.pi / 2.0
SHAPE_PRICE = 0.25  # between equally tidy walks, take the one whose turns match the patch's real corners
WALK_SLACK = 6  # a walk may disappoint the corners this much past what each wants: more costs far more time than it finds
WRONG_WEIGHT = 1.0  # a quad's area of wrong view counts as this many of `dissolve_limit` when choosing the next collapse
BEHIND_PIXELS = 6.0  # a quad this far behind the picture is taken to be hidden by something in front of it
RELAX_PASSES = 2
RELAX_COS = math.cos(math.radians(15.0))  # corners only relax where their faces are this close to flat
REDRAW_KEEP = 0.0  # a redrawn patch keeps at least this share of its quads
TRADE = 0.03  # a redraw may add this many quads' area of wrong view per unit of pole energy it takes away
POLE_WEIGHT = 0.2  # one unit of pole energy counts as this many of `dissolve_limit` when choosing the next collapse
REDRAW_EVERY = 0.05  # share of the quads to take away between redraws


def _closed_walk(costs: list[tuple[float, float, float]], limit: float) -> tuple[float, list[tuple[int, int]], list[int]] | None:
    """The cheapest closed walk with one unit step per corner. `costs[i]` is corner i's price per turn in TURNS.

    Returns (price, points, turns) with points[i] where corner i lands, or None if nothing under `limit` closes without crossing.
    """

    count = len(costs)
    if sum(min(row) for row in costs) > limit:
        return None
    module = _turbo.native() if _turbo is not None else None
    if module is not None and hasattr(module, "closed_walk"):
        found = module.closed_walk([tuple(row) for row in costs], float(limit))
        return None if found is None else _laid_out(found[0], found[1])
    middle = count // 2 + 1  # a walk that has to come back never gets further away than this
    width = 2 * middle + 1
    grid = np.full((4, width, width), np.inf)
    grid[0, middle, middle] = 0.0
    history = []
    for row in costs:
        history.append(grid)
        following = np.full((4, width, width), np.inf)
        for heading in range(4):
            source = grid[heading]
            for turn, price in zip(TURNS, row):
                new_heading = (heading + turn) & 3
                into, out = _SHIFTS[new_heading]
                target = following[new_heading]
                target[into] = np.minimum(target[into], source[out] + price)
        grid = following
    price = float(grid[0, middle, middle])
    if price > limit:
        return None
    turns = [0] * count
    heading, x, y, value = 0, middle, middle, price
    for i in range(count - 1, -1, -1):
        dx, dy = STEPS[heading]
        px, py = x - dx, y - dy
        for t, turn in enumerate(TURNS):
            before = (heading - turn) & 3
            if 0 <= px < width and 0 <= py < width and abs(history[i][before, px, py] + costs[i][t] - value) < 1e-9:
                turns[i] = t
                heading, x, y, value = before, px, py, float(history[i][before, px, py])
                break
        else:
            return None
    return _laid_out(price, turns)


def _laid_out(price: float, turns: list[int]) -> tuple[float, list[tuple[int, int]], list[int]] | None:
    """The walk's points from its turns, or None if it crosses itself or goes round the wrong way."""

    points = []
    x, y, heading = 0, 0, 0
    area = 0
    seen = set()
    for t in turns:
        if (x, y) in seen:
            return None
        seen.add((x, y))
        points.append((x, y))
        heading = (heading + TURNS[t]) & 3
        dx, dy = STEPS[heading]
        area += x * dy - y * dx
        x, y = x + dx, y + dy
    return (price, points, turns) if area > 0 else None


def _grid_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The unit squares inside a closed walk of unit steps that doesn't cross itself."""

    crossings: dict[int, list[int]] = {}
    for (x, y), (u, w) in zip(points, (*points[1:], points[0])):
        if x == u:
            crossings.setdefault(min(y, w), []).append(x)
    cells = []
    for y, row in crossings.items():
        row.sort()
        for a, b in zip(row[0::2], row[1::2]):
            cells += [(x, y) for x in range(a, b)]
    return cells


class PictureJudge:
    """How many pixels of the source's six pictures a quad would get wrong, by the same rules as the camera check.

    The quad is drawn into the pictures it faces, so this is the real measure at the real pixels and not an estimate from
    a few sample points, which a search then only learns to fool. A pixel where the quad lies well behind the picture is
    taken to be hidden by something else.
    """

    def __init__(self, reference, center, half: float, pixels: int, turned_deg: float):
        self.views = reference
        self.center = [float(x) for x in center]
        self.half = float(half)
        self.pixels = pixels
        self.pixel = 2.0 * self.half / pixels
        self.cosine = math.cos(math.radians(turned_deg))
        self.total = sum(int(view[0].sum()) for view in reference)
        module = _turbo.native() if _turbo is not None else None
        self._core = None
        if module is not None and hasattr(module, "PictureJudgeCore"):
            stack = [np.ascontiguousarray(np.stack([view[k] for view in reference]), dtype=t).reshape(-1) for k, t in ((0, bool), (1, np.float64), (2, np.float64))]
            self._core = module.PictureJudgeCore(*stack, self.center, self.half, pixels, self.cosine, BEHIND_PIXELS)
            self.wrong = self._core.wrong  # the same count, about a hundred times sooner
        else:
            self.lists = [(view[0].tolist(), view[1].tolist(), view[2].tolist()) for view in reference]

    def wrong(self, corners) -> float:
        return self._triangle(corners[0], corners[1], corners[2]) + self._triangle(corners[0], corners[2], corners[3])

    def _triangle(self, a, b, c) -> int:
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        normal = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        length = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
        if length < 1e-30:
            return 0
        normal = (normal[0] / length, normal[1] / length, normal[2] / length)
        total = 0
        scale = 1.0 / self.pixel
        half, center = self.half, self.center
        for axis in range(3):
            facing = normal[axis]
            if abs(facing) < 1e-6:
                continue
            sign = 1.0 if facing > 0.0 else -1.0
            i, j = (axis + 1) % 3, (axis + 2) % 3
            ax, ay = (a[i] - center[i] + half) * scale - 0.5, (a[j] - center[j] + half) * scale - 0.5
            bx, by = (b[i] - center[i] + half) * scale - 0.5, (b[j] - center[j] + half) * scale - 0.5
            cx, cy = (c[i] - center[i] + half) * scale - 0.5, (c[j] - center[j] + half) * scale - 0.5
            low_x, high_x = max(math.ceil(min(ax, bx, cx)), 0), min(math.floor(max(ax, bx, cx)), self.pixels - 1)
            low_y, high_y = max(math.ceil(min(ay, by, cy)), 0), min(math.floor(max(ay, by, cy)), self.pixels - 1)
            if low_x > high_x or low_y > high_y:
                continue
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-12:
                continue
            view = 2 * axis + (0 if sign > 0.0 else 1)
            if (high_x - low_x + 1) * (high_y - low_y + 1) <= SMALL_BOX:
                total += self._few_pixels(view, low_x, high_x, low_y, high_y, (ax, ay, bx, by, cx, cy, area), (a[axis], b[axis] - a[axis], c[axis] - a[axis]), normal, sign, center[axis] + 0.0)
                continue
            xs = np.arange(low_x, high_x + 1)[:, None]
            ys = np.arange(low_y, high_y + 1)[None, :]
            w1 = ((xs - ax) * (cy - ay) - (ys - ay) * (cx - ax)) / area
            w2 = ((bx - ax) * (ys - ay) - (by - ay) * (xs - ax)) / area
            inside = (w1 >= 0.0) & (w2 >= 0.0) & (w1 + w2 <= 1.0)
            if not inside.any():
                continue
            hit, normals, depth, _face = self.views[view]
            box = (slice(low_x, high_x + 1), slice(low_y, high_y + 1))
            here = a[axis] + w1[inside] * (b[axis] - a[axis]) + w2[inside] * (c[axis] - a[axis])
            mine = sign * (center[axis] - here) + 2.0 * half
            found = hit[box][inside]
            gap = mine - depth[box][inside]
            shown = normals[box][inside]
            turned = np.abs(shown[:, 0] * normal[0] + shown[:, 1] * normal[1] + shown[:, 2] * normal[2]) < self.cosine
            far = 2.0 * self.pixel
            behind = gap > BEHIND_PIXELS * self.pixel
            total += int(np.count_nonzero(~found | (~behind & (turned | (np.abs(gap) > far)))))
        return total


    def _few_pixels(self, view, low_x, high_x, low_y, high_y, flat, heights, normal, sign, middle) -> int:
        """The same test as `_triangle`'s, one pixel at a time, which beats NumPy on the few pixels of a small triangle."""

        ax, ay, bx, by, cx, cy, area = flat
        start, rise_b, rise_c = heights
        hit, normals, depth = self.lists[view]
        half, cosine = self.half, self.cosine
        far, hidden = 2.0 * self.pixel, BEHIND_PIXELS * self.pixel
        nx, ny, nz = normal
        wrong = 0
        for x in range(low_x, high_x + 1):
            row_hit, row_normal, row_depth = hit[x], normals[x], depth[x]
            for y in range(low_y, high_y + 1):
                w1 = ((x - ax) * (cy - ay) - (y - ay) * (cx - ax)) / area
                w2 = ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / area
                if w1 < 0.0 or w2 < 0.0 or w1 + w2 > 1.0:
                    continue
                if not row_hit[y]:
                    wrong += 1
                    continue
                gap = sign * (middle - (start + w1 * rise_b + w2 * rise_c)) + 2.0 * half - row_depth[y]
                if gap > hidden:
                    continue
                m = row_normal[y]
                if abs(m[0] * nx + m[1] * ny + m[2] * nz) < cosine or abs(gap) > far:
                    wrong += 1
        return wrong


class ChordReducer(QuadReducer):
    """`QuadReducer` that also redraws messy patches as grids, spins edges at poles, and prefers collapses that cancel poles."""

    def __init__(self, mesh: MeshData, judge=None):
        """`judge.wrong(corners)`, if given, is how much of this quad a viewer would see as wrong."""

        self.judge = judge
        self._refused_patches: set[tuple] = set()
        self._seen_stamps: list[int] = []
        self._turned: set[int] = set()
        self.project = None  # if set, the nearest point on the source, where relaxed corners go back to
        self.deadline = math.inf  # `time.perf_counter()` past which redrawing and judging stop and plain collapses go on
        self.messy: set[int] | None = None  # corners of the parts worth redrawing, None for all
        self.home: list[int] = list(range(len(mesh.vertices)))  # per corner, a corner of the mesh it was made from
        self._judged: dict[tuple, float] = {}
        self._unit_area = judge.total / max(len(mesh.faces), 1) if judge is not None else 1.0  # pixels per quad
        super().__init__(mesh)

    def _wrong(self, corners) -> float:
        """What the judge says of this quad, remembered by its corners."""

        key = tuple(corners)
        known = self._judged.get(key)
        if known is None:
            if len(self._judged) > 2_000_000:
                self._judged.clear()
            known = self._judged[key] = self.judge.wrong(key)
        return known

    def _extra_price(self, index: int, corner: int, spot) -> float:
        """The change in pole energy this collapse would make, and in what a viewer sees, priced as distance."""

        face = self.faces[index]
        a, b, c, d = face[corner], face[(corner + 1) % 4], face[(corner + 2) % 4], face[(corner + 3) % 4]
        users, cost = self.users, self._faces_at_cost
        na, nb, nc, nd = len(users[a]), len(users[b]), len(users[c]), len(users[d])
        change = cost(a, na + nc - 2) - cost(a, na) - cost(c, nc) + cost(b, nb - 1) - cost(b, nb) + cost(d, nd - 1) - cost(d, nd)
        price = POLE_WEIGHT * change
        if self.judge is not None and time.perf_counter() < self.deadline:
            points, faces = self.points, self.faces
            lost = -self._wrong([points[v] for v in face])
            for f in (users[a] | users[c]) - {index}:
                corners = faces[f]
                if len(corners) == 4:
                    lost += self._wrong([spot if v == a or v == c else points[v] for v in corners]) - self._wrong([points[v] for v in corners])
            price += WRONG_WEIGHT * lost / self._unit_area
        return self.dissolve_limit * price

    def remove(self, count: int, every: int = 64):
        """Generator: as in `QuadReducer`, stopping now and then to redraw what the collapses have messed up."""

        goal = self.alive - count
        while self.alive > goal:
            before = self.alive
            yield from super().remove(min(self.alive - goal, max(16, int(REDRAW_EVERY * self.alive))), every)
            if time.perf_counter() < self.deadline:
                near = self._seeds(self._near_changes())
                self.redraw_all(rounds=1, seeds=near)
                near = self._seeds(self._near_changes(near))
                self.turn_poles(seeds=near)
                self.relax(RELAX_PASSES, seeds=near)
            yield
            if self.alive >= before:
                break

    def freeze(self, vertices, renamed: bool = False) -> None:
        """As in `QuadReducer`. Corners a redraw made since the last `save` are gone after a `restore`, so they're left out."""

        known = len(self.users)
        super().freeze([v for v in (vertices if renamed else self.original_ids(vertices)) if v < known], renamed=True)

    def roll_back_parts(self, saved, vertices) -> None:
        """Put the connected parts holding these vertices back as they were in `saved` (from `save`), and stop them there."""

        old_points, old_faces, old_users, old_quadrics = saved[:4]
        faces, users, points = self.faces, self.users, self.points

        def part_of(starts, every_face, every_user):
            seen, stack, found = set(starts), list(starts), set()
            while stack:
                for f in every_user[stack.pop()]:
                    if f not in found:
                        found.add(f)
                        for w in every_face[f]:
                            if w not in seen:
                                seen.add(w)
                                stack.append(w)
            return found, seen

        now, _ = part_of([v for v in vertices if users[v]], faces, users)
        back, corners = part_of([v for v in vertices if v < len(old_users) and old_users[v]], old_faces, old_users)
        for f in now:
            for v in faces[f]:
                users[v].discard(f)
            faces[f] = None
        for f in back:
            faces[f] = list(old_faces[f])
        self.alive += len(back) - len(now)
        for v in corners:
            points[v], users[v], self.quadrics[v] = old_points[v], set(old_users[v]), old_quadrics[v]
            self.stamps[v] += 1
            self.locked[v] = True

    def _faces_at_cost(self, vertex: int, faces: int) -> int:
        if self.border[vertex]:
            return (faces - 2) ** 2 if faces > 2 else 0
        return (faces - 4) ** 2

    def energy(self) -> int:
        return sum(self._faces_at_cost(v, len(u)) for v, u in enumerate(self.users) if u)

    def _is_smooth(self, vertex: int) -> bool:
        """No sharp edge of the shape runs through this corner, and it's free to go."""

        if self.locked[vertex] or self.border[vertex]:
            return False
        normals = [_normal([self.points[v] for v in self.faces[f]]) for f in self.users[vertex]]
        return all(_dot(a, b) >= SMOOTH_COS for k, a in enumerate(normals) for b in normals[:k])

    def _patch(self, seed: int, radius: int) -> set[int] | None:
        """The quads within `radius` rings of this corner, stopping at sharp edges and material changes."""

        faces, users, materials = self.faces, self.users, self.materials
        if not self._is_smooth(seed):
            return None
        kind = materials[next(iter(users[seed]))]
        region: set[int] = set()
        frontier = {seed}
        seen = {seed}
        for _ring in range(radius):
            following = set()
            for v in frontier:
                if not self._is_smooth(v):
                    continue
                for f in users[v]:
                    if f not in region and materials[f] == kind and len(faces[f]) == 4:
                        region.add(f)
                        following.update(faces[f])
            frontier = following - seen
            seen |= following
        return region

    def _angle_inside(self, vertex: int, region: set[int]) -> float:
        """Radians the patch's quads take up at this corner."""

        total = 0.0
        points = self.points
        here = points[vertex]
        for f in self.users[vertex]:
            if f in region:
                face = self.faces[f]
                at = face.index(vertex)
                one, two = _sub(points[face[(at + 1) % 4]], here), _sub(points[face[at - 1]], here)
                total += math.atan2(math.sqrt(_dot(n := _cross(one, two), n)), _dot(one, two))
        return total

    def _outline(self, region: set[int]) -> list[int] | None:
        """The patch's one boundary, in the faces' own direction. None unless the patch is a plain disk."""

        following: dict[int, int] = {}
        halves = set()
        faces = self.faces
        for f in region:
            face = faces[f]
            halves.update(zip(face, (*face[1:], face[0])))
        for a, b in halves:
            if (b, a) not in halves:
                if a in following:
                    return None  # two parts of the patch touch only at this corner
                following[a] = b
        if not following:
            return None
        start = next(iter(following))
        ring = [start]
        v = following[start]
        while v != start:
            ring.append(v)
            v = following.get(v)
            if v is None or len(ring) > len(following):
                return None
        if len(ring) != len(following):
            return None
        return ring

    def redraw(self, seed: int, radius: int, slack: int = 0) -> int:
        """Try to redraw the patch around a corner as a grid. Returns how much the pole energy fell, 0 if nothing changed."""

        faces, users = self.faces, self.users
        region = self._patch(seed, radius)
        if not region or len(region) < 3:
            return 0
        ring = self._outline(region)
        if ring is None or len(ring) > LONGEST_WALK:
            return 0
        inside: dict[int, int] = {}
        for f in region:
            for v in faces[f]:
                inside[v] = inside.get(v, 0) + 1
        on_ring = set(ring)
        loose = [v for v in inside if v not in on_ring]
        if any(self.locked[v] for v in loose):
            return 0
        # A disk: V - E + F == 1 with one boundary. Anything else has a hole or a handle in it.
        edge_count = (4 * len(region) + len(ring)) // 2
        if len(inside) - edge_count + len(region) != 1:
            return 0
        before = sum(self._faces_at_cost(v, len(users[v])) for v in inside)
        if before == 0:
            return 0
        memo_key = (tuple(ring), tuple(len(users[v]) for v in ring), before, len(region))
        if memo_key in self._refused_patches:
            return 0
        self._refused_patches.add(memo_key)
        costs, wishes = [], []
        for v in ring:
            outside = len(users[v]) - inside[v]
            turn_angle = self._angle_inside(v, region) / QUARTER
            row = [self._faces_at_cost(v, outside + n) for n in (1, 2, 3)]
            wishes.append(row)
            costs.append(tuple(c + SHAPE_PRICE * (turn_angle - n) ** 2 for c, n in zip(row, (1, 2, 3))))
        walk = _closed_walk(costs, min(before - 0.5, sum(min(row) for row in costs) + WALK_SLACK))
        if walk is None:
            return 0
        _price, spots, turns = walk
        after = sum(row[t] for row, t in zip(wishes, turns))
        if after >= before:
            return 0
        cells = _grid_cells(spots)
        if not cells or len(cells) > len(region) + slack or len(cells) < REDRAW_KEEP * len(region):
            return 0
        made = self._place_grid(region, ring, spots, cells, loose, before - after)
        return before - after if made else 0

    def _place_grid(self, region: set[int], ring: list[int], spots, cells, loose: list[int], gain: int) -> bool:
        """Work out the grid on the surface, and swap it in if it fits the old patch."""

        faces, points = self.faces, self.points
        where: dict[tuple[int, int], tuple[float, float, float]] = {spot: points[v] for spot, v in zip(spots, ring)}
        middle: list[tuple[int, int]] = []
        for x, y in cells:
            for spot in ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)):
                if spot not in where:
                    where[spot] = None
                    middle.append(spot)
        center = tuple(sum(points[v][k] for v in ring) / len(ring) for k in range(3))
        for spot in middle:
            where[spot] = center
        old_quads = []
        for f in region:
            corners = [points[v] for v in faces[f]]
            mid = tuple(sum(c[k] for c in corners) / 4.0 for k in range(3))
            old_quads.append((mid, _normal(corners)))
        for sweep in range(24 if middle else 0):
            for x, y in middle:
                near = [where[s] for s in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if s in where]
                where[(x, y)] = tuple(sum(n[k] for n in near) / len(near) for k in range(3))

        def nearest(p):
            return min(old_quads, key=lambda o: (o[0][0] - p[0]) ** 2 + (o[0][1] - p[1]) ** 2 + (o[0][2] - p[2]) ** 2)

        judge = self.judge
        for spot in middle:
            p = where[spot]
            mid, n = nearest(p)
            h = _dot(_sub(p, mid), n)
            where[spot] = (p[0] - h * n[0], p[1] - h * n[1], p[2] - h * n[2])
        new_normals = []
        new_wrong = 0.0
        for x, y in cells:
            quad = [where[(x, y)], where[(x + 1, y)], where[(x + 1, y + 1)], where[(x, y + 1)]]
            if _quality(quad) < NEW_QUALITY:
                return False
            if judge is not None:
                new_wrong += self._wrong(quad)
                continue
            n = _normal(quad)
            mid = tuple(sum(c[k] for c in quad) / 4.0 for k in range(3))
            if _dot(n, nearest(mid)[1]) < NEW_TURNED:
                return False
            new_normals.append((mid, n))
        if judge is not None:
            old_wrong = sum(self._wrong([points[v] for v in faces[f]]) for f in region)
            if new_wrong > old_wrong + TRADE * gain * self._unit_area:
                return False
        limit = MOST_LEFT * self.dissolve_limit
        for v in loose if judge is None else ():
            p = points[v]
            mid, n = min(new_normals, key=lambda o: (o[0][0] - p[0]) ** 2 + (o[0][1] - p[1]) ** 2 + (o[0][2] - p[2]) ** 2)
            if abs(_dot(_sub(p, mid), n)) > limit:
                return False
        # It fits: swap it in.
        material = self.materials[next(iter(region))]
        names = dict(zip(spots, ring))
        spare = {}
        for spot in middle:
            names[spot] = self._new_corner(where[spot], ring[0])
            spare[names[spot]] = where[spot]
        everyone = list(spare) or ring
        for v in loose:
            p = points[v]
            home = min(everyone, key=lambda u: (points[u][0] - p[0]) ** 2 + (points[u][1] - p[1]) ** 2 + (points[u][2] - p[2]) ** 2)
            self.quadrics[home] = [a + b for a, b in zip(self.quadrics[home], self.quadrics[v])]
        for f in region:
            for v in faces[f]:
                self.users[v].discard(f)
            faces[f] = None
            self.alive -= 1
        for v in ring:
            self.stamps[v] += 1
        for v in loose:
            self.stamps[v] += 1
        fresh = []
        for x, y in cells:
            face = [names[(x, y)], names[(x + 1, y)], names[(x + 1, y + 1)], names[(x, y + 1)]]
            index = len(faces)
            faces.append(face)
            if index < len(self.materials):
                self.materials[index] = material
            else:
                self.materials.append(material)
            for v in face:
                self.users[v].add(index)
            self.alive += 1
            fresh.append(index)
        for index in {f for v in ring for f in self.users[v]} | set(fresh):
            if len(faces[index]) == 4:
                self._push(index, 0, self.heap)
                self._push(index, 1, self.heap)
        return True

    def _new_corner(self, position, home: int) -> int:
        vertex = len(self.points)
        if vertex < len(self.home):
            self.home[vertex] = self.home[home]
        else:
            self.home.append(self.home[home])
        self.points.append(position)
        self.users.append(set())
        self.quadrics.append([0.0] * 11)
        self.stamps.append(0)
        if vertex < len(self.locked):
            self.locked[vertex] = False
            self.border[vertex] = False
        else:
            self.locked.append(False)
            self.border.append(False)
        return vertex

    def _near_changes(self, also=()) -> set[int]:
        """The corners within two rings of anything changed since the last call."""

        stamps, seen, users, faces = self.stamps, self._seen_stamps, self.users, self.faces
        changed = {v for v, stamp in enumerate(stamps) if v >= len(seen) or seen[v] != stamp}
        changed |= self._turned
        self._turned = set()
        self._seen_stamps = list(stamps)
        ring = {v for v in changed if v < len(users) and users[v]}
        out = set(ring)
        for _ in range(2):
            ring = {w for v in ring for f in users[v] for w in faces[f]} - out
            out |= ring
        return out | set(also)

    def redraw_all(self, radii=(1, 2, 3), rounds: int = 3, seeds=None) -> int:
        """Redraw patches around poles, worst first, until that stops paying. Returns the energy taken away."""

        total = 0
        for _round in range(rounds):
            gained = 0
            pool = range(len(self.users)) if seeds is None else seeds
            order = sorted((v for v in pool if self.users[v] and not self.locked[v] and len(self.users[v]) != 4), key=lambda v: -abs(len(self.users[v]) - 4))
            for v in order:
                if time.perf_counter() > self.deadline:
                    return total + gained
                if not self.users[v] or len(self.users[v]) == 4:
                    continue
                for radius in radii:
                    gain = self.redraw(v, radius)
                    if gain:
                        gained += gain
                        break
            total += gained
            if not gained:
                break
        return total

    def _turnings(self, a: int, b: int):
        """Every way to spin edge a-b within its two quads (the other two diagonals of their hexagon), with the change in pole energy."""

        shared = self.users[a] & self.users[b]
        if len(shared) != 2 or self.locked[a] or self.locked[b]:
            return
        f, g = shared
        one, two = self.faces[f], self.faces[g]
        if len(one) != 4 or len(two) != 4 or self.materials[f] != self.materials[g]:
            return
        at = one.index(a)
        if one[(at + 1) % 4] != b:
            f, g, one, two = g, f, two, one
            at = one.index(a)
            if one[(at + 1) % 4] != b:
                return
        k = two.index(b)
        if two[(k + 1) % 4] != a:
            return
        ring = [b, one[(at + 2) % 4], one[(at + 3) % 4], a, two[(k + 2) % 4], two[(k + 3) % 4]]
        if len(set(ring)) != 6:
            return
        users, cost = self.users, self._faces_at_cost
        for s in (1, 2):
            x, y = ring[s], ring[s + 3]
            if y in self._around(x):
                continue
            change = 0
            for v, by in ((a, -1), (b, -1), (x, 1), (y, 1)):
                change += cost(v, len(users[v]) + by) - cost(v, len(users[v]))
            yield change, f, g, [ring[(s + t) % 6] for t in range(4)], [ring[(s + 3 + t) % 6] for t in range(4)], (a, b, x, y)

    def _turn_fits(self, f: int, g: int, first: list[int], second: list[int], gain: int) -> bool:
        points, faces = self.points, self.faces
        new_one, new_two = [points[v] for v in first], [points[v] for v in second]
        old_one, old_two = [points[v] for v in faces[f]], [points[v] for v in faces[g]]
        if _quality(new_one) < NEW_QUALITY or _quality(new_two) < NEW_QUALITY:
            return False
        if _dot(_normal(new_one), _normal(old_one)) < MOST_TURNED or _dot(_normal(new_two), _normal(old_two)) < MOST_TURNED:
            return False
        if self.judge is None:
            return True
        return self._wrong(new_one) + self._wrong(new_two) <= self._wrong(old_one) + self._wrong(old_two) + TRADE * gain * self._unit_area

    def _set_face(self, f: int, corners: list[int]) -> None:
        for v in self.faces[f]:
            self.users[v].discard(f)
        self.faces[f] = corners
        for v in corners:
            self.users[v].add(f)

    def _turn_from(self, vertex: int, depth: int) -> int:
        """Spin an edge at this corner if that lowers the pole energy. With `depth`, also walk a pole pair along first, undoing the walk if it finds nothing."""

        for other in list(self._around(vertex)):
            for change, f, g, first, second, ends in list(self._turnings(vertex, other)):
                if change > 0 or not self._turn_fits(f, g, first, second, max(-change, 0)):
                    continue
                old_one, old_two = self.faces[f], self.faces[g]
                self._set_face(f, first)
                self._set_face(g, second)
                if change < 0:
                    self._push_turned(f, g)
                    return -change
                if depth > 0:
                    for w in ends:
                        if len(self.users[w]) != 4:
                            gain = self._turn_from(w, depth - 1)
                            if gain:
                                self._push_turned(f, g)
                                return gain
                self._set_face(f, old_one)
                self._set_face(g, old_two)
        return 0

    def _push_turned(self, f: int, g: int) -> None:
        for index in (f, g):
            self._turned.update(self.faces[index])
            self._push(index, 0, self.heap)
            self._push(index, 1, self.heap)

    def turn_poles(self, depth: int = 1, seeds=None) -> int:
        """Spin edges around poles (Tarini et al. 2010's edge rotation): cheap, and it picks up what redrawing leaves. Returns the energy taken away."""

        total = 0
        for v in range(len(self.users)) if seeds is None else seeds:
            around = self.users[v]
            while around and len(around) != 4 and not self.border[v]:
                gain = self._turn_from(v, depth)
                if not gain:
                    break
                total += gain
        return total

    def _seeds(self, near):
        return near if self.messy is None else near & self.messy

    def flag_messy(self, share: float) -> int:
        """Redraw only connected parts where more than `share` of the corners are poles. Returns how many parts that is."""

        users, faces = self.users, self.faces
        seen: set[int] = set()
        self.messy = set()
        flagged = 0
        for start, around in enumerate(users):
            if not around or start in seen:
                continue
            part, stack = {start}, [start]
            while stack:
                for f in users[stack.pop()]:
                    for w in faces[f]:
                        if w not in part:
                            part.add(w)
                            stack.append(w)
            seen |= part
            if sum(1 for v in part if len(users[v]) != 4) > share * len(part):
                self.messy |= part
                flagged += 1
        return flagged

    def tidy(self) -> None:
        seeds = None if self.messy is None else sorted(self.messy)
        self.redraw_all(seeds=seeds)
        if time.perf_counter() < self.deadline:
            self.turn_poles(seeds=seeds)
            self.relax(RELAX_PASSES, seeds=seeds)

    # Smoothing

    def relax(self, passes: int = 2, seeds=None) -> None:
        """Even the quads out without leaving the surface: each corner goes toward its neighbours' middle as far as its own planes allow.

        The planes are the corner's quadric, so a corner on a flat slides freely, on a crease only along it, at a tip not at all.
        """

        points, faces, users, quadrics = self.points, self.faces, self.users, self.quadrics
        for _ in range(passes):
            for v in range(len(users)) if seeds is None else seeds:
                around = users[v]
                if not around or self.locked[v]:
                    continue
                normals = [_normal([points[u] for u in faces[g]]) for g in around]
                if any(_dot(n, m) < RELAX_COS for k, n in enumerate(normals) for m in normals[:k]):
                    continue  # on a bend: sliding a corner there moves the smooth shading more than it evens the quads
                q = quadrics[v]
                mass = q[10]
                if mass <= 1e-30:
                    continue
                near = self._around(v)
                count = len(near)
                t = [sum(points[u][k] for u in near) / count for k in range(3)]
                p = points[v]
                size = sum((points[u][0] - p[0]) ** 2 + (points[u][1] - p[1]) ** 2 + (points[u][2] - p[2]) ** 2 for u in near) / count
                if size <= 1e-30:
                    continue
                # (A + w I) x = w t - b, with A and b scaled so that w = 0.02 is a weak pull toward the middle.
                w = 0.02 * mass
                a, b, c, d, e, f = q[0] + w, q[1], q[2], q[4] + w, q[5], q[7] + w
                r0, r1, r2 = w * t[0] - q[3], w * t[1] - q[6], w * t[2] - q[8]
                m0, m1, m2 = d * f - e * e, c * e - b * f, b * e - c * d
                det = a * m0 + b * m1 + c * m2
                if abs(det) < 1e-30:
                    continue
                x = ((r0 * m0 + r1 * m1 + r2 * m2) / det, (r0 * m1 + r1 * (a * f - c * c) + r2 * (b * c - a * e)) / det, (r0 * m2 + r1 * (b * c - a * e) + r2 * (a * d - b * b)) / det)
                ok = True
                for g in around:
                    before = [points[u] for u in faces[g]]
                    now = [x if u == v else points[u] for u in faces[g]]
                    if _quality(now) < min(_quality(before), MIN_QUALITY) or _dot(_normal(now), _normal(before)) < MOST_TURNED:
                        ok = False
                        break
                if self.project is not None:  # back onto the source: on the planes alone, smooth shading drifts
                    x = tuple(self.project(x))
                    ok = all(_quality([x if u == v else points[u] for u in faces[g]]) >= min(_quality([points[u] for u in faces[g]]), MIN_QUALITY) for g in around)
                if ok and self.judge is not None:
                    old = sum(self._wrong([points[u] for u in faces[g]]) for g in around)
                    ok = sum(self._wrong([x if u == v else points[u] for u in faces[g]]) for g in around) <= old
                if ok:
                    points[v] = x


__all__ = ["ChordReducer", "PictureJudge"]
