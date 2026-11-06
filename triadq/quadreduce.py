# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Fewer quads on an all-quad mesh: remove the quads whose loss changes the surface least.

Closing leftover triangles spends about seven quads on each one, so a closed mesh often has twice the quads its shape
needs. Collapsing a quad across a diagonal (merge two opposite corners) takes one quad away and leaves every other face
a quad, the same operation a person does by hand with Merge. Which quad goes next is decided by the area-weighted
squared distance to the original planes (Garland and Heckbert's quadrics) through the merged corner. A corner left in
only two quads adds nothing to the shape, so those two become one.

Plain Python: each removal touches about ten faces, and a removal is checked before anything changes.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from .types import MeshData

MIN_QUALITY = 0.15  # no new quad worse than this, on `patch.quad_quality`'s scale (> 0 means convex and warped < 60 degrees)
MOST_TURNED = math.cos(math.radians(25.0))  # no face may turn further than this in one removal
MOST_FACES_AT_CORNER = 8
JOIN_QUALITY = 0.1  # the worst quad allowed when joining leftover triangles
DISSOLVE_SHARE = 0.002  # a corner may leave the surface by this share of the object's size when its two quads merge


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(a):
    length = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if length < 1e-30:
        return (0.0, 0.0, 0.0)
    return a[0] / length, a[1] / length, a[2] / length


def _normal(p):
    return _unit(_cross(_sub(p[2], p[0]), _sub(p[3], p[1])))


def _quality(p) -> float:
    """`patch.quad_quality` for one quad, without NumPy's cost per call."""

    normal = _normal(p)
    worst = 1.0
    for k in range(4):
        out = _sub(p[(k + 1) % 4], p[k])
        back = _sub(p[k - 1], p[k])
        size = math.sqrt(_dot(out, out) * _dot(back, back))
        if size < 1e-30:
            return -1.0
        worst = min(worst, _dot(_cross(out, back), normal) / size)
    if worst <= 0.0:
        return worst
    for x, y, z, w in ((0, 1, 2, 3), (1, 2, 3, 0)):
        fold = _dot(_unit(_cross(_sub(p[y], p[x]), _sub(p[z], p[x]))), _unit(_cross(_sub(p[z], p[x]), _sub(p[w], p[x]))))
        worst = min(worst, (fold - 0.5) * 2.0)
    return worst


def _plane_quadrics(points, faces) -> list[list[float]]:
    """Each corner's area-weighted squared distance to the planes of its faces."""

    out = [[0.0] * 11 for _ in points]
    for face in faces:
        for k in range(1, len(face) - 1):
            a, b, c = points[face[0]], points[face[k]], points[face[k + 1]]
            n = _cross(_sub(b, a), _sub(c, a))
            twice_area = math.sqrt(_dot(n, n))
            if twice_area < 1e-30:
                continue
            x, y, z = n[0] / twice_area, n[1] / twice_area, n[2] / twice_area
            d = -(x * a[0] + y * a[1] + z * a[2])
            row = (x * x, x * y, x * z, x * d, y * y, y * z, y * d, z * z, z * d, d * d, 1.0)
            for v in (face[0], face[k], face[k + 1]):
                q = out[v]
                for i in range(11):
                    q[i] += row[i] * twice_area
    return out


class QuadReducer:
    """Removes quads cheapest first. `remove(count)` is a generator; `save`/`restore` allow stepping back after a look."""

    def __init__(self, mesh: MeshData):
        self.materials = list(mesh.face_materials)
        self.name = mesh.name
        self.points = [tuple(p) for p in np.asarray(mesh.vertices, dtype=np.float64).tolist()]
        self.faces: list[list[int] | None] = [list(f) for f in mesh.faces]
        self.alive = len(self.faces)
        self._result_ids = list(range(len(self.points)))
        self.users: list[set[int]] = [set() for _ in self.points]
        edge_use: dict[tuple[int, int], int] = {}
        material_of: list[set[str]] = [set() for _ in self.points]
        self.locked = [False] * len(self.points)
        for index, face in enumerate(self.faces):
            for k, v in enumerate(face):
                self.users[v].add(index)
                material_of[v].add(self.materials[index])
                w = face[(k + 1) % len(face)]
                key = (v, w) if v < w else (w, v)
                edge_use[key] = edge_use.get(key, 0) + 1
            if len(face) != 4:
                for v in face:
                    self.locked[v] = True
        self.border = [False] * len(self.points)
        for (v, w), count in edge_use.items():
            if count != 2:
                self.border[v] = self.border[w] = True
        for v in range(len(self.points)):
            if self.border[v] or len(material_of[v]) > 1:
                self.locked[v] = True
        self.quadrics = _plane_quadrics(self.points, self.faces)
        used = np.asarray(mesh.vertices, dtype=np.float64)
        self.dissolve_limit = DISSOLVE_SHARE * float(np.linalg.norm(np.ptp(used, axis=0))) if len(used) else 0.0
        self.stamps = [0] * len(self.points)
        self.heap: list[tuple] = []
        self.refused: list[tuple] = []
        for index in range(len(self.faces)):
            if len(self.faces[index]) == 4:
                self._push(index, 0)
                self._push(index, 1)
        heapq.heapify(self.heap)

    @staticmethod
    def _error(q: list[float], p) -> float:
        x, y, z = p
        return x * (q[0] * x + 2.0 * (q[1] * y + q[2] * z + q[3])) + y * (q[4] * y + 2.0 * (q[5] * z + q[6])) + z * (q[7] * z + 2.0 * q[8]) + q[9]

    def _push(self, index: int, corner: int, into=None) -> None:
        """Queue this quad's collapse across the diagonal from `corner`, at its cheapest place."""

        a, c = self.faces[index][corner], self.faces[index][corner + 2]
        if self.locked[a] and self.locked[c]:
            return
        pa, pc = self.points[a], self.points[c]
        q = [x + y for x, y in zip(self.quadrics[a], self.quadrics[c])]
        if self.locked[a]:
            options = ((pa, 0),)
        elif self.locked[c]:
            options = ((pc, 1),)
        else:
            middle = ((pa[0] + pc[0]) * 0.5, (pa[1] + pc[1]) * 0.5, (pa[2] + pc[2]) * 0.5)
            options = ((pa, 0), (pc, 1), (middle, 2))
        error, place = min(((self._error(q, p), k) for p, k in options), key=lambda item: item[0])
        spot = next(p for p, k in options if k == place)
        side = _sub(pa, pc)
        # In a flat area every error is zero: the shortest diagonal then goes first, which keeps the sizes even.
        key = math.sqrt(max(error, 0.0) / max(q[10], 1e-30)) + 1e-4 * math.sqrt(_dot(side, side)) + self._extra_price(index, corner, spot)
        entry = (key, index, corner, a, c, self.stamps[a], self.stamps[c], place)
        if into is None:
            self.heap.append(entry)
        else:
            heapq.heappush(into, entry)

    def _extra_price(self, index: int, corner: int, spot) -> float:
        return 0.0

    def _try(self, q: int, corner: int, place: int) -> bool:
        """One collapse: everything is checked first, then applied. False leaves the mesh as it was."""

        faces, users, points = self.faces, self.users, self.points
        face = faces[q]
        a, b, c, d = face[corner], face[(corner + 1) % 4], face[(corner + 2) % 4], face[(corner + 3) % 4]
        ua, uc = users[a], users[c]
        if (ua & uc) != {q} or len(ua) + len(uc) - 2 > MOST_FACES_AT_CORNER:
            return False
        keep, drop = (c, a) if place == 1 else (a, c)
        if (self._around(a) & self._around(c)) != {b, d}:
            return False
        pa, pc = points[a], points[c]
        position = pa if place == 0 else pc if place == 1 else ((pa[0] + pc[0]) * 0.5, (pa[1] + pc[1]) * 0.5, (pa[2] + pc[2]) * 0.5)

        def renamed(corners):
            return [keep if v == drop else v for v in corners]

        moved = {f: renamed(faces[f]) for f in (ua | uc) - {q}}
        merged_with: dict[int, tuple[int, int]] = {}  # kept face -> (face merged into it, corner that left)
        for side in (b, d):
            count = len(users[side])
            if self.border[side]:
                if count < 2:
                    return False
            elif count < 3:
                return False
            elif count == 3:  # it would be left in two quads: they become one
                first, second = sorted(f for f in users[side] if f != q)  # sorted: a set's order is not a fixed one
                if self.locked[side] or first not in moved or second not in moved or len(faces[first]) != 4 or len(faces[second]) != 4:
                    return False
                if first in merged_with or second in merged_with:  # both sides share a quad, which can't merge twice in one removal
                    return False
                one, two = renamed(faces[first]), renamed(faces[second])
                one, two = one[one.index(side):] + one[: one.index(side)], two[two.index(side):] + two[: two.index(side)]
                if one[1] != two[3] or one[3] != two[1] or one[2] == two[2]:
                    return False
                other = one[3] if one[1] == keep else one[1]
                if not self.border[other] and len(users[other]) <= 3:
                    return False
                moved[first] = [one[1], one[2], one[3], two[2]]
                del moved[second]
                merged_with[first] = (second, side)

        if not self.border[keep] and sum(1 for corners in moved.values() if keep in corners) < 3:
            return False
        seen = set()
        for f, corners in moved.items():
            size = len(corners)
            if len(set(corners)) != size:
                return False
            shape = frozenset(corners)
            if shape in seen:
                return False
            seen.add(shape)
            if size != 4:
                continue
            new = [position if v == keep else points[v] for v in corners]
            if _quality(new) < MIN_QUALITY:
                return False
            turned = _normal(new)
            if _dot(turned, _normal([points[v] for v in faces[f]])) < MOST_TURNED:
                return False
            if f in merged_with:
                second, side = merged_with[f]
                if _dot(turned, _normal([points[v] for v in faces[second]])) < MOST_TURNED:
                    return False
                if abs(_dot(turned, _sub(points[side], new[0]))) > self.dissolve_limit:
                    return False

        # Everything holds: apply it.
        total = [x + y for x, y in zip(self.quadrics[a], self.quadrics[c])]
        for f in (q, *(second for second, _side in merged_with.values())):
            for v in faces[f]:
                users[v].discard(f)
            faces[f] = None
            self.alive -= 1
        for f, corners in moved.items():
            for v in faces[f]:
                users[v].discard(f)
            faces[f] = corners
            for v in corners:
                users[v].add(f)
        for _second, side in merged_with.values():
            total = [x + y for x, y in zip(self.quadrics[side], total)]
        self.quadrics[keep] = total
        points[keep] = position
        self.stamps[keep] += 1
        self.stamps[drop] += 1
        for f in users[keep]:
            if len(faces[f]) == 4:
                self._push(f, faces[f].index(keep) % 2, self.heap)
        for f in merged_with:
            self._push(f, 0, self.heap)
            self._push(f, 1, self.heap)
        return True

    def _around(self, vertex: int) -> set[int]:
        """The vertices joined to this one by an edge."""

        around = set()
        for f in self.users[vertex]:
            face = self.faces[f]
            at = face.index(vertex)
            around.add(face[at - 1])
            around.add(face[(at + 1) % len(face)])
        return around

    def remove(self, count: int, every: int = 64):
        """Generator: take away up to `count` quads, cheapest first. Yields now and then so a timer can keep Blender live."""

        goal = self.alive - count
        heap = self.heap
        since = 0
        while heap and self.alive > goal:
            entry = heapq.heappop(heap)
            _key, q, corner, a, c, stamp_a, stamp_c, place = entry
            face = self.faces[q]
            since += 1
            if since >= every:
                since = 0
                yield
            if face is None or face[corner] != a or face[corner + 2] != c or self.stamps[a] != stamp_a or self.stamps[c] != stamp_c:
                continue
            if self.locked[a] and self.locked[c]:
                continue
            # On a flat grid only the middle works, though every place has the same error.
            others = () if self.locked[a] or self.locked[c] else tuple(k for k in (2, 0, 1) if k != place)
            if not any(self._try(q, corner, k) for k in (place, *others)):
                self.refused.append(entry)

    def freeze(self, vertices, renamed: bool = False) -> None:
        """Nothing more comes off the connected parts holding these vertices (numbered as in `result()`, or as here if `renamed`)."""

        stack = list(vertices) if renamed else self.original_ids(vertices)
        seen = set(stack)
        while stack:
            vertex = stack.pop()
            self.locked[vertex] = True
            for f in self.users[vertex]:
                for other in self.faces[f]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)

    def original_ids(self, vertices) -> list[int]:
        """Vertices numbered as in the last `result()`, in this reducer's own numbering."""

        return [self._result_ids[v] for v in vertices]

    def retry_refused(self) -> None:
        """What was refused may fit now that its surroundings have changed."""

        self.heap.extend(self.refused)
        self.refused = []
        heapq.heapify(self.heap)

    def save(self):
        return (
            list(self.points),
            [None if f is None else list(f) for f in self.faces],
            [set(u) for u in self.users],
            list(self.quadrics),  # quadrics are replaced, never changed in place
            list(self.stamps),
            list(self.heap),
            list(self.refused),
            self.alive,
        )

    def restore(self, saved) -> None:
        self.points, faces, users, self.quadrics, self.stamps, self.heap, self.refused, self.alive = saved
        self.points, self.quadrics, self.stamps, self.heap, self.refused = list(self.points), list(self.quadrics), list(self.stamps), list(self.heap), list(self.refused)
        self.faces = [None if f is None else list(f) for f in faces]
        self.users = [set(u) for u in users]

    def result(self) -> MeshData:
        kept = [k for k, face in enumerate(self.faces) if face is not None]
        used = sorted({v for k in kept for v in self.faces[k]})
        self._result_ids = used
        table = {v: k for k, v in enumerate(used)}
        return MeshData(
            vertices=np.asarray([self.points[v] for v in used], dtype=np.float64).reshape(-1, 3),
            faces=[tuple(table[v] for v in self.faces[k]) for k in kept],
            face_materials=[self.materials[k] for k in kept],
            name=self.name,
        )


def quad_reducer(mesh: MeshData):
    """A reducer for this mesh."""

    return QuadReducer(mesh)


def _compensated_sum(values) -> float:
    """Python 3.12's float `sum` (Neumaier), spelled out so 3.11 gives the same last digit."""

    high = low = 0.0
    for x in values:
        t = high + x
        low += (high - t) + x if abs(high) >= abs(x) else (x - t) + high
        high = t
    return high + low if low and math.isfinite(low) else high


def _facing(p):
    """The way a triangle or a quad faces (a quad's diagonals; a triangle's last corner stands in for the fourth)."""
    return _unit(_cross(_sub(p[2], p[0]), _sub(p[-1], p[1])))


def _cut_hexagon(hexagon: list[int], points, free_chord, facings=()) -> tuple[list[tuple[int, ...]], list[int]] | None:
    """The best two quads for a six-sided polygon, or three around a new middle corner. (quads, middle's id or []).

    Each quad must also face within about 70 degrees of every face in `facings`, the faces the polygon was made of: across
    a sharp edge one would fold over its neighbours, so the triangle there is left for closure instead.
    """

    corners = [points[v] for v in hexagon]
    total = (0.0, 0.0, 0.0)
    for k in range(1, 5):
        n = _cross(_sub(corners[k], corners[0]), _sub(corners[k + 1], corners[0]))
        total = (total[0] + n[0], total[1] + n[1], total[2] + n[2])
    normal = _unit(total)

    def worst(quads) -> float:
        low = 1.0
        for quad in quads:
            spots = [points[v] for v in quad]
            facing = _normal(spots)
            if _dot(facing, normal) <= 0.3 or any(_dot(facing, other) <= 0.3 for other in facings):
                return -1.0
            low = min(low, _quality(spots))
        return low

    best, best_worst = None, JOIN_QUALITY
    for k in range(3):
        if not free_chord(hexagon[k], hexagon[k + 3]):
            continue
        quads = [tuple(hexagon[k:k + 4]), tuple(hexagon[k + 3:] + hexagon[:k + 1])]
        value = worst(quads)
        if value >= best_worst:
            best, best_worst = quads, value
    if best is not None:
        return best, []
    middle = len(points)
    points.append(tuple(_compensated_sum(c[k] for c in corners) / 6.0 for k in range(3)))
    for k in (0, 1):
        ring = hexagon[k:] + hexagon[:k]
        quads = [(middle, ring[0], ring[1], ring[2]), (middle, ring[2], ring[3], ring[4]), (middle, ring[4], ring[5], ring[0])]
        value = worst(quads)
        if value >= best_worst:
            best, best_worst = quads, value
    if best is None:
        points.pop()
        return None
    return best, [middle]


def join_loose_triangles(mesh: MeshData, farthest: int = 8) -> tuple[MeshData, int]:
    """Join leftover triangles two at a time along a path of quads. Returns (mesh, triangles joined).

    Closure splits all three edges of a leftover triangle, and each split runs on through the mesh, which costs about
    seven quads per triangle. A triangle only needs one extra corner, though: a triangle and its neighbouring quad have
    five sides, and one split on the far side of the quad makes six, which is two quads. Each quad further along the
    path has two splits and becomes two quads. So a path of k quads costs k - 2 more faces in all, and nothing changes
    outside it. What can't be joined this way is left for closure.
    """

    points = [tuple(p) for p in np.asarray(mesh.vertices, dtype=np.float64).tolist()]
    faces: list[tuple[int, ...] | None] = list(mesh.faces)
    materials = list(mesh.face_materials)
    users: dict[tuple[int, int], list[int]] = {}

    def edges(face):
        return (((a, b) if a < b else (b, a)) for a, b in zip(face, (*face[1:], face[0])))

    for index, face in enumerate(faces):
        for edge in edges(face):
            users.setdefault(edge, []).append(index)
    loose = {k for k, face in enumerate(faces) if len(face) == 3}

    def merged(first: list[int], face: tuple[int, ...]) -> list[int] | None:
        """The outline of a polygon and a face that shares exactly one edge with it."""
        at = {v: k for k, v in enumerate(first)}
        for k, v in enumerate(face):
            u = face[(k + 1) % len(face)]  # the face runs v -> u, so the outline must run u -> v
            if u in at and first[(at[u] + 1) % len(first)] == v:
                rest = [face[(k + 2 + m) % len(face)] for m in range(len(face) - 2)]
                if any(w in at for w in rest):
                    return None
                return first[: at[u] + 1] + rest + first[at[u] + 1 :]
        return None

    def with_split(outline: list[int], edge: tuple[int, int], middle: int) -> list[int] | None:
        for k, v in enumerate(outline):
            u = outline[(k + 1) % len(outline)]
            if ((v, u) if v < u else (u, v)) == edge:
                return outline[: k + 1] + [middle] + outline[k + 1 :]
        return None

    def join(path: list[int]) -> bool:
        """Replace these faces (triangle, quads, triangle) by quads, if every one comes out valid."""
        count = len(path)
        shared = []
        for one, two in zip(path, path[1:]):
            common = set(edges(faces[one])) & set(edges(faces[two]))
            if len(common) != 1:
                return False
            shared.append(common.pop())
        region = set(path)
        merged_away = {shared[0], shared[-1]}
        if count == 3:
            merged_away = set(shared)

        def free_chord(x: int, y: int) -> bool:
            edge = (x, y) if x < y else (y, x)
            return edge not in users or edge in merged_away

        size = len(points)
        middles = {}
        for edge in shared[1:-1]:
            middles[edge] = len(points)
            pa, pb = points[edge[0]], points[edge[1]]
            points.append(((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5))
        outlines = []
        made_of = [[path[0], path[1], path[2]]] if count == 3 else [[path[0], path[1]], *([path[k]] for k in range(2, count - 2)), [path[-2], path[-1]]]
        if count == 3:
            outlines.append(merged(merged(list(faces[path[0]]), faces[path[1]]) or [], faces[path[2]]))
        else:
            first = merged(list(faces[path[0]]), faces[path[1]])
            outlines.append(first and with_split(first, shared[1], middles[shared[1]]))
            for k in range(2, count - 2):
                middle = with_split(list(faces[path[k]]), shared[k - 1], middles[shared[k - 1]])
                outlines.append(middle and with_split(middle, shared[k], middles[shared[k]]))
            last = merged(list(faces[path[-2]]), faces[path[-1]])
            outlines.append(last and with_split(last, shared[-2], middles[shared[-2]]))
        made = []
        for outline, ids in zip(outlines, made_of):
            facings = [_facing([points[v] for v in faces[f]]) for f in ids]
            cut = _cut_hexagon(outline, points, free_chord, facings) if outline and len(outline) == 6 else None
            if cut is None:
                del points[size:]
                return False
            made += cut[0]
        # Faces of one path can touch away from it, so the new quads may not add a third face to any edge.
        uses: dict[tuple[int, int], int] = {}
        for quad in made:
            for e in edges(quad):
                uses[e] = uses.get(e, 0) + 1
        if any(n + len(set(users.get(e, ())) - region) > 2 for e, n in uses.items()):
            del points[size:]
            return False
        for f in path:
            for e in edges(faces[f]):
                row = users[e]
                row.remove(f)
                if not row:
                    del users[e]
            faces[f] = None
        material = materials[path[0]]
        for quad in made:
            faces.append(quad)
            materials.append(material)
            for e in edges(quad):
                users.setdefault(e, []).append(len(faces) - 1)
        loose.discard(path[0])
        loose.discard(path[-1])
        return True

    def search(start: int, reach: int) -> None:
        """Try the other triangles within `reach` quads, nearest first."""
        parents = {start: -1}
        level = [start]
        for _depth in range(reach + 1):
            following = []
            for f in level:
                for edge in edges(faces[f]):
                    for other in users[edge]:
                        if other in parents or materials[other] != materials[start]:
                            continue
                        parents[other] = f
                        if other in loose:
                            if f == start:  # next to each other: pairing already found that this can't be one quad
                                continue
                            path = [other]
                            while path[-1] != start:
                                path.append(parents[path[-1]])
                            if join(path):
                                return
                        else:
                            following.append(other)
            level = following

    reach = 0
    while reach < farthest:
        reach = max(1, reach * 2)
        for start in sorted(loose):
            if start in loose:
                search(start, reach)
        if not loose:
            break
    kept = [k for k, face in enumerate(faces) if face is not None]
    joined = sum(1 for face in mesh.faces if len(face) == 3) - len(loose)
    return MeshData(vertices=np.asarray(points, dtype=np.float64), faces=[faces[k] for k in kept], face_materials=[materials[k] for k in kept], name=mesh.name), joined


__all__ = ["QuadReducer", "join_loose_triangles", "quad_reducer"]


def collapse_loose_triangles(mesh: MeshData, farthest: int = 8) -> tuple[MeshData, int]:
    """Get rid of leftover triangles by merging edges along a path of quads. Returns (mesh, triangles gone).

    Merging an edge a triangle shares with a quad removes the triangle and leaves the quad a triangle: one face fewer,
    and the triangle has moved a step. When it reaches another triangle, both are gone. So where `join_loose_triangles`
    pays k - 2 faces for a path of k quads, this gains k + 2, for moving a few corners by about an edge's length. Every
    merge is checked first. A triangle that gets stuck is left where it is for joining and closure.
    """

    points = [tuple(p) for p in np.asarray(mesh.vertices, dtype=np.float64).tolist()]
    faces: list[list[int] | None] = [list(f) for f in mesh.faces]
    materials = mesh.face_materials
    users: list[set[int]] = [set() for _ in points]
    edge_use: dict[tuple[int, int], int] = {}
    material_of: list[set[str]] = [set() for _ in points]
    for index, face in enumerate(faces):
        for k, v in enumerate(face):
            users[v].add(index)
            material_of[v].add(materials[index])
            w = face[(k + 1) % len(face)]
            key = (v, w) if v < w else (w, v)
            edge_use[key] = edge_use.get(key, 0) + 1
    locked = [len(m) > 1 for m in material_of]
    for (v, w), count in edge_use.items():
        if count != 2:
            locked[v] = locked[w] = True
    quadrics = _plane_quadrics(points, faces)
    loose = {k for k, face in enumerate(faces) if len(face) == 3}
    start_count = len(loose)

    def around(vertex: int) -> set[int]:
        out = set()
        for f in users[vertex]:
            face = faces[f]
            at = face.index(vertex)
            out.add(face[at - 1])
            out.add(face[(at + 1) % len(face)])
        return out

    def facing(spots) -> tuple[float, float, float]:
        return _normal(spots) if len(spots) == 4 else _unit(_cross(_sub(spots[1], spots[0]), _sub(spots[2], spots[0])))

    def still_good(corners: list[int], old: list[int], position, keep: int, drop: int) -> bool:
        new = [position if v == keep else points[v] for v in corners]
        before = [points[v] for v in old]
        if len(new) == 4 and _quality(new) < min(JOIN_QUALITY, _quality(before)):
            return False
        return _dot(facing(new), facing(before)) >= MOST_TURNED

    def collapse(tri: int, other: int) -> bool:
        """Merge the edge between this triangle and the next face. False leaves everything as it was."""
        face, next_face = faces[tri], faces[other]
        common = set(face) & set(next_face)
        if len(common) != 2 or materials[tri] != materials[other]:
            return False
        a, b = sorted(common)
        at = next_face.index(a)
        if b not in (next_face[at - 1], next_face[(at + 1) % len(next_face)]):
            return False
        if (users[a] & users[b]) != {tri, other} or (locked[a] and locked[b]):
            return False
        tips = {v for v in face if v not in common} | ({v for v in next_face if v not in common} if len(next_face) == 3 else set())
        if (around(a) & around(b)) != tips or any(len(users[v]) < 3 for v in tips):
            return False
        pa, pb = points[a], points[b]
        total = [x + y for x, y in zip(quadrics[a], quadrics[b])]
        if locked[a]:
            options = [(pa, a, b)]
        elif locked[b]:
            options = [(pb, b, a)]
        else:
            middle = ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5)
            options = sorted([(middle, a, b), (pa, a, b), (pb, b, a)], key=lambda item: QuadReducer._error(total, item[0]))
        for position, keep, drop in options:
            moved = {}
            shapes = set()
            for f in (users[a] | users[b]) - {tri}:
                if f == other and len(next_face) == 3:
                    continue
                corners = list(dict.fromkeys(keep if v == drop else v for v in faces[f]))
                if f != other and len(corners) != len(faces[f]):
                    break
                shape = frozenset(corners)
                if shape in shapes or not still_good(corners, faces[f], position, keep, drop):
                    break
                shapes.add(shape)
                moved[f] = corners
            else:
                break
        else:
            return False
        for f in (tri,) if len(next_face) == 4 else (tri, other):
            for v in faces[f]:
                users[v].discard(f)
            faces[f] = None
            loose.discard(f)
        for f, corners in moved.items():
            for v in faces[f]:
                users[v].discard(f)
            faces[f] = corners
            for v in corners:
                users[v].add(f)
            if len(corners) == 3:
                loose.add(f)
        quadrics[keep] = total
        points[keep] = position
        return True

    def search(start: int, reach: int) -> None:
        """Walk this triangle to the nearest other triangle within `reach` quads."""
        parents = {start: -1}
        level = [start]
        for _depth in range(reach + 1):
            following = []
            for f in level:
                for other in {g for v in faces[f] for g in users[v]}:
                    if other in parents or len(set(faces[f]) & set(faces[other])) != 2:
                        continue
                    parents[other] = f
                    if other not in loose:
                        following.append(other)
                        continue
                    path = [other]
                    while path[-1] != start:
                        path.append(parents[path[-1]])
                    path.reverse()
                    for k, (at, step) in enumerate(zip(path, path[1:])):
                        if not collapse(at, step):
                            if k:
                                return  # it has moved: the next round starts from where it is
                            break
                    else:
                        return
            level = following

    reach = 0
    while reach < farthest:
        reach = max(1, reach * 2)
        for start in sorted(loose):
            if start in loose:
                search(start, reach)
        if not loose:
            break
    kept = [k for k, face in enumerate(faces) if face is not None]
    used = sorted({v for k in kept for v in faces[k]})
    table = {v: k for k, v in enumerate(used)}
    result = MeshData(
        vertices=np.asarray([points[v] for v in used], dtype=np.float64).reshape(-1, 3),
        faces=[tuple(table[v] for v in faces[k]) for k in kept],
        face_materials=[materials[k] for k in kept],
        name=mesh.name,
    )
    return result, start_count - len(loose)


__all__ = ["QuadReducer", "collapse_loose_triangles", "join_loose_triangles"]
