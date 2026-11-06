# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Local repair of faces that pass through each other, without touching the rest of the part.

Each spot gets the cheapest thing that clears it: its corners back on the source, then a small push along the source's
normal, then its faces filled again with smaller quads and a welded seam. A change is kept only if the spot has fewer
new crossings afterwards and no quad got folded. A crossing the source already had is left alone.
"""

from __future__ import annotations

import numpy as np

from . import checks, patch

PUSHES = (0.05, 0.12, 0.25, 0.5)  # a push, as a share of the spot's edge length
FILLS = ((1.0, 0), (0.5, 0), (0.5, 1), (0.25, 1), (0.25, 2))  # (edge length as a share of the spot's, rings of faces around it)
ROUNDS = 3
NEAR_SOURCE = 5.0  # a crossing this many face sizes from one the source has is the source's own
MOST_FILLS = 300  # tries per repair: each costs a fair part of a second
MOST_FILLED = 80  # faces in one fill: a bigger one is not local
FEW_PAIRS = 4
SMOOTHS = (1, 3, 8)  # rounds of evening out the corners of a spot


class Known:
    """What the source already crosses, to tell a new crossing from an old one."""

    def __init__(self, co, faces, piece, pairs):
        co = np.asarray(co, dtype=np.float64)
        piece = np.asarray(piece, dtype=np.int64)
        two = piece[pairs] if len(pairs) else np.zeros((0, 2), np.int64)
        self.others = {tuple(p) for p in np.sort(two[two[:, 0] != two[:, 1]], axis=1).tolist()}
        self.inside: dict[int, np.ndarray] = {}  # piece -> where the source crosses itself
        inner = pairs[two[:, 0] == two[:, 1]] if len(pairs) else pairs
        for p in np.unique(piece[inner[:, 0]]).tolist() if len(inner) else ():
            ids = np.unique(inner[piece[inner[:, 0]] == p])
            self.inside[p] = np.asarray([co[list(faces[f])].mean(axis=0) for f in ids.tolist()])


def new_crossing(known: Known, piece, points, faces, a: int, b: int) -> bool:
    """Did the source not have this crossing? Pieces that crossed still do, and a knot in the source stays a knot."""
    pa, pb = int(piece[a]), int(piece[b])
    if pa != pb:
        return (min(pa, pb), max(pa, pb)) not in known.others
    old = known.inside.get(pa)
    if old is None:
        return True
    here = points[list(faces[a])]
    return float(np.min(np.linalg.norm(old - here.mean(axis=0), axis=1))) > NEAR_SOURCE * float(np.ptp(here, axis=0).max())


class _Work:
    """The mesh while it is being repaired: faces may die and be added, numbers stay put until `finished`."""

    def __init__(self, points, faces, mats, piece, known: Known, near, tris_only: bool):
        self.points = np.array(points, dtype=np.float64)
        self.faces = [tuple(f) for f in faces]
        self.mats = list(mats)
        self.piece = np.array(piece, dtype=np.int64)
        self.known, self.near, self.tris_only = known, near, tris_only
        self.alive = np.ones(len(self.faces), bool)
        self.pad = checks.corner_table(self.faces)
        self.low, self.high = checks.face_boxes(self.points, self.pad)
        self.users: dict[int, set[int]] = {}
        for f, face in enumerate(self.faces):
            for v in face:
                self.users.setdefault(v, set()).add(f)
        self.added = 0
        self.pushed = 0
        self.filled = 0
        self.fresh: set[int] = set()  # the faces the last accepted fill made
        self.fills = 0

    def is_new(self, a: int, b: int) -> bool:
        return new_crossing(self.known, self.piece, self.points, self.faces, a, b)

    def new_pairs(self, ids) -> list[tuple[int, int]]:
        """New crossings that involve these faces, measured now."""
        ids = np.asarray(sorted(ids), dtype=np.int64)
        ids = ids[self.alive[ids]]
        if not len(ids):
            return []
        near = np.flatnonzero(self.alive & ((self.low <= self.high[ids].max(axis=0)) & (self.high >= self.low[ids].min(axis=0))).all(axis=1))
        hits = checks.tree_of(self.points, self.faces, ids).overlap(checks.tree_of(self.points, self.faces, near))
        if not hits:
            return []
        hits = np.asarray(hits, dtype=np.int64)
        pairs = np.column_stack((ids[hits[:, 0]], near[hits[:, 1]]))
        pairs = np.unique(np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1), axis=0)
        return [(a, b) for a, b in checks.drop_shared(self.pad, pairs).tolist() if self.is_new(a, b)]

    def _folds(self, ids) -> set[int]:
        quads = [f for f in ids if len(self.faces[f]) == 4]
        if not quads:
            return set()
        quality = patch.quad_quality(self.points[np.asarray([self.faces[f] for f in quads], dtype=np.int64)])
        return {f for f, q in zip(quads, quality.tolist()) if q <= 0.0}

    def _rebox(self, ids) -> None:
        ids = np.asarray(sorted(ids), dtype=np.int64)
        self.low[ids], self.high[ids] = checks.face_boxes(self.points, self.pad[ids])

    def attempt_move(self, verts: np.ndarray, moved: np.ndarray, spot: set[int]) -> bool:
        """Move these corners if it leaves fewer new crossings around them and no new folded quad."""
        touched = {f for v in verts.tolist() for f in self.users[v]}
        before, folded = self.new_pairs(touched | spot), self._folds(touched)
        saved = self.points[verts].copy()
        self.points[verts] = moved
        self._rebox(touched)
        if not self._folds(touched) - folded and len(self.new_pairs(touched | spot)) < len(before):
            return True
        self.points[verts] = saved
        self._rebox(touched)
        return False

    def attempt_collapse(self, f: int, spot: set[int]) -> bool:
        """Take this quad away by merging two opposite corners in the middle, if the others stay proper quads."""
        face = self.faces[f]
        if len(face) != 4:
            return False
        for k in (0, 1):
            a, c = face[k], face[k + 2]
            region = sorted(self.users[a] | self.users[c])
            if len({self.mats[g] for g in region}) > 1 or len({int(self.piece[g]) for g in region}) > 1 or any(len(self.faces[g]) != 4 for g in region):
                continue
            links = [{w for g in self.users[v] for w in self._beside(self.faces[g], v)} for v in (a, c)]
            closed = all(sum(v in (self.faces[g][i], self.faces[g][(i + 1) % 4]) for g in self.users[v] for i in range(4)) == 2 * len(link) for v, link in zip((a, c), links))
            if not closed or links[0] & links[1] != {face[k + 1], face[(k + 3) % 4]}:
                continue
            mid = len(self.points)
            made = [tuple(mid if v in (a, c) else v for v in self.faces[g]) for g in region if g != f]
            spot_now = spot - {f}
            if self._swap(region, ((self.points[a] + self.points[c]) / 2.0)[None], made, self.mats[f], int(self.piece[f]), spot_now):
                return True
        return False

    def _evened(self, verts: np.ndarray, rounds: int) -> np.ndarray:
        """Where these corners go when each moves to the middle of its neighbours, the corners around them held still.

        A knot in a tight fold opens up this way, with no change in the faces."""
        where = {v: k for k, v in enumerate(verts.tolist())}
        links = [sorted({w for f in self.users[v] for w in self._beside(self.faces[f], v)}) for v in verts.tolist()]
        moved = self.points[verts].copy()
        for _ in range(rounds):
            moved = np.asarray([np.mean([moved[where[w]] if w in where else self.points[w] for w in near], axis=0) for near in links])
        return moved

    @staticmethod
    def _beside(face, v) -> tuple[int, int]:
        at = face.index(v)
        return face[at - 1], face[(at + 1) % len(face)]

    def _ring(self, ids: set[int], rings: int, kind: str, part: int) -> set[int]:
        """These faces and `rings` rings around them, of one material and one piece."""
        out = set(ids)
        for _ in range(rings):
            out |= {g for f in out for v in self.faces[f] for g in self.users[v] if self.mats[g] == kind and self.piece[g] == part}
        return out

    def _sheets(self, ids: set[int]) -> list[set[int]]:
        """These faces split into groups that touch: the two surfaces of a crossing are filled one at a time."""
        left, out = set(ids), []
        while left:
            stack = [left.pop()]
            sheet = set(stack)
            while stack:
                near = {g for v in self.faces[stack.pop()] for g in self.users[v]} & left
                left -= near
                sheet |= near
                stack += near
            out.append(sheet)
        return out

    def _disks(self, ids: set[int]) -> list[list[int]]:
        """These faces as disks with even seams, cut where they aren't one. Works on the faces around them only."""
        area = sorted(self._ring(ids, 3, self.mats[next(iter(ids))], int(self.piece[next(iter(ids))])))
        local = {f: k for k, f in enumerate(area)}
        faces = [self.faces[f] for f in area]
        pieces, _ = patch._disk_pieces(self.points, faces, np.asarray([local[f] for f in ids]), patch._Incidence(faces), patch.face_adjacency(faces), set(), 1)
        return [[area[k] for k in piece.tolist()] for piece in pieces]

    def attempt_fill(self, ids: set[int], share: float, spot: set[int]) -> bool:
        """Fill these faces again at this edge length, seam welded, if that leaves fewer new crossings."""
        done = False
        for region in self._disks(ids):
            self.fills += 1
            done = (len(region) <= MOST_FILLED and self._fill_disk(region, share, spot)) or done
        return done

    def _fill_disk(self, region: list[int], share: float, spot: set[int]) -> bool:
        loops, _ = patch.boundary_loops(self.faces, region)
        if len(loops) != 1:
            return False
        kind, part = self.mats[region[0]], int(self.piece[region[0]])
        edges = [np.linalg.norm(np.diff(self.points[list(self.faces[f]) + [self.faces[f][0]]], axis=0), axis=1).mean() for f in region]
        result = patch.fill_loop(self.points, loops[0], target_length=share * float(np.mean(edges)), strategy="FLOW", max_faces=MOST_FILLED * 8)
        if not result.ok:
            return False
        made = result.faces
        if self.tris_only or self.pad.shape[1] == 3:
            made = [t for f in made for t in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))]
        points = result.points
        if len(points):
            base = result.base
            seam = np.asarray(loops[0])
            where = {v: k for k, v in enumerate(loops[0])}
            local = [tuple(where[v] if v < base else len(seam) + v - base for v in f) for f in result.faces]
            stacked = np.vstack((self.points[seam], points))
            pinned = np.zeros(len(stacked), bool)
            pinned[: len(seam)] = True
            stacked = patch.fit_to_surface(stacked, local, pinned, lambda q: self.near(part, q)[0], 3, 0.5)
            if self._swap(region, stacked[len(seam) :], made, kind, part, spot):
                return True
        return self._swap(region, points, made, kind, part, spot)  # plain, without the source: a knot too tight to follow

    def _swap(self, region: list[int], points, made, kind: str, part: int, spot: set[int]) -> bool:
        around = {g for f in region for v in self.faces[f] for g in self.users[v]} | spot
        before = self.new_pairs(around)
        base = len(self.points)
        count = len(self.faces)
        self.points = np.vstack((self.points, points))
        self.faces += made
        self.mats += [kind] * len(made)
        self.piece = np.concatenate((self.piece, np.full(len(made), part, np.int64)))
        width = self.pad.shape[1]
        self.pad = np.vstack((self.pad, np.asarray([tuple(f) + (f[0],) * (width - len(f)) for f in made], dtype=np.int64)))
        low, high = checks.face_boxes(self.points, self.pad[count:])
        self.low, self.high = np.vstack((self.low, low)), np.vstack((self.high, high))
        self.alive = np.concatenate((self.alive, np.ones(len(made), bool)))
        self.alive[region] = False
        fresh = set(range(count, len(self.faces)))
        for v in range(base, len(self.points)):
            self.users[v] = set()
        for f in region:
            for v in self.faces[f]:
                self.users[v].discard(f)
        for f in fresh:
            for v in self.faces[f]:
                self.users[v].add(f)
        if not self._folds(fresh) and len(self.new_pairs(fresh | around)) < len(before):
            self.added += len(made) - len(region)
            self.fresh |= fresh
            return True
        self.alive[region] = True
        self.alive[count:] = False
        for f in fresh:
            for v in self.faces[f]:
                self.users[v].discard(f)
        for f in region:
            for v in self.faces[f]:
                self.users[v].add(f)
        return False

    def fix_spot(self, pairs: list[tuple[int, int]], most_added: int):
        """Generator, yielding before each try: the cheap things on one spot, then the fills. Stops as soon as it is clear."""
        pairs = [pair for pair in pairs if self.alive[pair[0]] and self.alive[pair[1]]]  # an earlier spot may have remade these
        if not pairs:
            return
        spot = {f for pair in pairs for f in pair}

        def live() -> list[tuple[int, int]]:
            return self.new_pairs({f for f in spot if self.alive[f]})

        users = np.bincount(np.asarray(pairs).ravel())
        hubs = [{int(f)} for f in np.argsort(-users)[:3] if users[f] > 1]  # a face that crosses many others may be all that's wrong
        sides = [spot, {a for a, _ in pairs}, {b for _, b in pairs}, *hubs]

        def verts_of(ids) -> np.ndarray:
            return np.asarray(sorted({v for f in ids for v in self.faces[f]}), dtype=np.int64)

        size = float(np.mean([np.linalg.norm(np.diff(self.points[list(self.faces[f]) + [self.faces[f][0]]], axis=0), axis=1).mean() for f in spot]))
        for side in sides:
            verts = verts_of(side)
            spots = np.asarray([self.near(int(self.piece[next(iter(self.users[v]))]), self.points[[v]])[0][0] for v in verts.tolist()])
            yield  # one try per slice: a timer runs the repair in short steps
            if self.attempt_move(verts, spots, spot) and not live():
                self.pushed += 1
                return
        for rounds in SMOOTHS:
            verts = verts_of(spot)
            yield  # one try per slice: a timer runs the repair in short steps
            if self.attempt_move(verts, self._evened(verts, rounds), spot) and not live():
                self.pushed += 1
                return
        for share in PUSHES:
            for side in sides:
                verts = verts_of(side)
                normals = np.asarray([self.near(int(self.piece[next(iter(self.users[v]))]), self.points[[v]])[1][0] for v in verts.tolist()])
                for sign in (-1.0, 1.0):
                    yield  # one try per slice: a timer runs the repair in short steps
                    if self.attempt_move(verts, self.points[verts] + sign * share * size * normals, spot) and not live():
                        self.pushed += 1
                        return
        if len(pairs) <= FEW_PAIRS:  # one corner in the wrong place: try each by itself, further than the group went
            for v in verts_of(spot).tolist():
                one = np.asarray([v])
                closest, normal = self.near(int(self.piece[next(iter(self.users[v]))]), self.points[one])
                for target in [self._evened(one, 1), closest] + [self.points[one] + sign * share * size * normal for share in (*PUSHES, 1.0) for sign in (-1.0, 1.0)]:
                    yield  # one try per slice: a timer runs the repair in short steps
                    if self.attempt_move(one, target, spot) and not live():
                        self.pushed += 1
                        return
        for f in sorted(spot, key=lambda f: -int(users[f]) if f < len(users) else 0):  # the worst first: one face fewer often clears a knot
            yield  # one try per slice: a timer runs the repair in short steps
            if self.alive[f] and self.attempt_collapse(f, spot):
                spot = {g for g in spot if self.alive[g]} | self.fresh
                self.fresh = set()
                if not live():
                    self.pushed += 1
                    return
        for share, rings in FILLS:
            if self.added >= most_added or self.fills >= MOST_FILLS:
                return
            for kind, part in sorted({(self.mats[f], int(self.piece[f])) for f in spot if self.alive[f]}):
                ids = {f for f in spot if self.alive[f] and self.mats[f] == kind and self.piece[f] == part}
                for sheet in self._sheets(self._ring(ids, rings, kind, part)):
                    self.fresh = set()
                    yield  # one try per slice: a timer runs the repair in short steps
                    try:
                        done = self.attempt_fill(sheet, share, spot)
                    except (ValueError, np.linalg.LinAlgError):  # a fill can't always close a tangled seam: the spot stays as it was
                        done = False
                    if done:
                        spot = {f for f in spot if self.alive[f]} | self.fresh
                        if not live():
                            self.filled += 1
                            return

    def finished(self):
        keep = np.flatnonzero(self.alive).tolist()
        used, flat = np.unique(np.fromiter((v for f in keep for v in self.faces[f]), dtype=np.int64), return_inverse=True)
        faces, at = [], 0
        for f in keep:
            size = len(self.faces[f])
            faces.append(tuple(flat[at : at + size].tolist()))
            at += size
        return self.points[used], faces, [self.mats[f] for f in keep], self.piece[keep]


def spots_of(pairs) -> list[list[tuple[int, int]]]:
    """Crossing pairs grouped: pairs that share a face are one spot."""
    parent: dict[int, int] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups: dict[int, list[tuple[int, int]]] = {}
    for a, b in pairs:
        groups.setdefault(find(a), []).append((a, b))
    return list(groups.values())


def new_by_piece(known: Known, pairs, points, faces, piece) -> tuple[dict[int, int], np.ndarray]:
    """(new crossings per piece, the pairs). A piece passing through another counts once for each, however many faces meet."""
    new = np.asarray([(a, b) for a, b in np.asarray(pairs).tolist() if new_crossing(known, piece, points, faces, a, b)], dtype=np.int64).reshape(-1, 2)
    two = np.asarray(piece)[new]
    same = two[:, 0] == two[:, 1]
    counts = dict(zip(*(x.tolist() for x in np.unique(two[same, 0], return_counts=True))))
    for a, b in np.unique(np.sort(two[~same], axis=1), axis=0).tolist():
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    return counts, new


def untangle(*args, **kwargs):
    """`untangle_steps` run to its end in one call."""
    return _drain(untangle_steps(*args, **kwargs))


def _drain(steps):
    try:
        while True:
            next(steps)
    except StopIteration as done:
        return done.value


def fix_crossings_steps(points, faces, mats, piece, pairs, known: Known, near, tris_only: bool = False, growth: float = 0.03, stopped=lambda: False):
    """Generator: repair the new crossings among `pairs`, yielding after each spot so a timer can run it in short
    slices. `near(piece, points)` gives (closest, normal) on that source piece.

    Returns (points, faces, mats, piece, numbers). A filled spot may add faces, up to `growth` of them. Plain work with
    no Blender data.
    """
    work = _Work(points, faces, mats, piece, known, near, tris_only)
    most_added = int(growth * len(work.faces))
    todo = [(a, b) for a, b in np.asarray(pairs).tolist() if work.is_new(a, b)]
    found = len(todo)
    for _round in range(ROUNDS):
        spots = sorted(spots_of(todo), key=len)  # small ones first: they take least and are most likely to clear
        for spot in spots:
            if stopped():
                break
            yield from work.fix_spot(spot, most_added)
        todo = [p for spot in spots for p in work.new_pairs({f for pair in spot for f in pair if work.alive[f]})]
        if not todo or stopped():
            break
    return (*work.finished(), {"new": found, "pushed": work.pushed, "filled": work.filled, "added": work.added})


def untangle_steps(points, faces, piece, bad, near, rounds: int = ROUNDS, stopped=lambda: False):
    """Generator, yielding after each group of corners: move the corners of faces turned against the source or folded over a neighbour until they face out again.

    `bad` marks those faces. Each of their corners tries the middle of its neighbours and its own nearest source point,
    both put on the source, and keeps the one that leaves fewer faces around it turned against the source or folded
    over a neighbour, if no quad there gets bent. Faces are not changed. Returns (points, corners moved).

    The corners go in groups that share no face, so a whole group is judged at once, with the same result as one by one;
    `rounds` passes over every corner is the whole budget, which makes the result the same on any computer. `stopped` is
    only a safety net.
    """
    points = np.array(points, dtype=np.float64)
    faces = [tuple(int(v) for v in f) for f in faces]
    piece = np.asarray(piece, dtype=np.int64)
    todo = sorted({v for f in np.flatnonzero(bad).tolist() for v in faces[f]})
    if not todo:
        return points, 0
    users: dict[int, list[int]] = {v: [] for v in todo}
    for f, face in enumerate(faces):
        for v in face:
            if v in users:
                users[v].append(f)
    width = max(len(f) for f in faces)
    pad = np.asarray([f + (f[-1],) * (width - len(f)) for f in faces], dtype=np.int64)  # a repeated corner adds nothing to a normal
    sizes = np.asarray([len(f) for f in faces], dtype=np.float64)
    real = np.arange(width) < sizes[:, None]
    out_of = np.zeros((len(faces), 3))  # the source's way out at a face's middle, asked once per position
    fresh = np.zeros(len(faces), dtype=bool)

    # Groups of corners that share no face, in waves: a corner goes one wave after the latest lower-numbered corner it
    # shares a face with. Judging the waves in turn is the same as judging the corners one by one in number order.
    group: dict[int, int] = {}
    for v in todo:
        group[v] = 1 + max((group[w] for f in users[v] for w in faces[f] if w in group), default=-1)
    batches = []
    for c in range(max(group.values()) + 1):
        yield
        vs = [v for v in todo if group[v] == c]
        around, owner, fold_a, fold_b, ring_w, ring_o = [], [], [], [], [], []
        for i, v in enumerate(vs):
            first = len(around)
            ids = users[v]
            around += ids
            owner += [i] * len(ids)
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    if len(set(faces[ids[a]]) & set(faces[ids[b]])) >= 2:
                        fold_a.append(first + a)
                        fold_b.append(first + b)
            ring = sorted({w for f in ids for w in faces[f]} - {v})
            ring_w += ring
            ring_o += [i] * len(ring)
        around_arr = np.asarray(around, dtype=np.int64)
        quad = sizes[around_arr] == 4
        batches.append(
            dict(
                vs=np.asarray(vs, dtype=np.int64),
                faces=around_arr,
                owner=np.asarray(owner, dtype=np.int64),
                quad=quad,
                quad_owner=np.asarray(owner, dtype=np.int64)[quad],
                fold=(np.asarray(fold_a, dtype=np.int64), np.asarray(fold_b, dtype=np.int64)),
                ring=(np.asarray(ring_w, dtype=np.int64), np.asarray(ring_o, dtype=np.int64)),
                part=piece[[users[v][0] for v in vs]],
            )
        )

    def on_source(parts, where):
        out = np.empty_like(where)
        for p in np.unique(parts).tolist():
            these = parts == p
            out[these] = np.asarray(near(p, where[these])[0]).reshape(-1, 3)
        return out

    def outward(ids, moving):
        """The source's way out at these faces' middles: asked again for the faces round a corner that is trying a place
        (`moving`), from what was asked before for the rest."""
        ask = moving | ~fresh[ids]
        out = out_of[ids]
        if ask.any():
            these_ids = ids[ask]
            middle = (points[pad[these_ids]] * real[these_ids][..., None]).sum(axis=1) / sizes[these_ids][:, None]
            normals = np.empty_like(middle)
            for p in np.unique(piece[these_ids]).tolist():
                these = piece[these_ids] == p
                normals[these] = np.asarray(near(p, middle[these])[1]).reshape(-1, 3)
            out[ask] = normals
            keep = these_ids[~moving[ask]]
            out_of[keep] = normals[~moving[ask]]
            fresh[keep] = True
        return out

    def scores(b, trying=None) -> np.ndarray:
        """Per corner of batch b: faces around it turned against the source, plus folds on their edges; infinity if a quad
        there bends."""
        ids, owner, count = b["faces"], b["owner"], len(b["vs"])
        corners = points[pad[ids]]
        following = np.roll(corners, -1, axis=1)
        normal = np.cross(corners, following).sum(axis=1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
        moving = np.zeros(len(ids), dtype=bool) if trying is None else trying[owner]
        turned = np.einsum("ij,ij->i", normal, outward(ids, moving)) < checks.FACING
        a, f = b["fold"]
        folded = np.einsum("ij,ij->i", normal[a], normal[f]) < checks.FOLD
        value = np.bincount(owner[turned], minlength=count) + np.bincount(owner[a[folded]], minlength=count)
        value = value.astype(np.float64)
        if b["quad"].any():
            bent = patch.quad_quality(points[pad[ids[b["quad"]]]]) <= 0.0
            value[np.unique(b["quad_owner"][bent])] = np.inf
        return value

    moved = 0
    for _round in range(rounds):
        changed = False
        for b in batches:
            if stopped():
                return points, moved
            vs = b["vs"]
            best = scores(b)
            go = best > 0
            if not go.any():
                continue
            ring_w, ring_o = b["ring"]
            middle = np.stack([np.bincount(ring_o, weights=points[ring_w, k], minlength=len(vs)) for k in range(3)], axis=1)
            middle /= np.maximum(np.bincount(ring_o, minlength=len(vs)), 1)[:, None]
            here = points[vs].copy()
            pick = np.full(len(vs), -1)
            tries = []
            for start in (middle, here):
                at = here.copy()
                at[go] = on_source(b["part"][go], start[go])
                tries.append(at)
            for t, at in enumerate(tries):
                points[vs] = at
                value = scores(b, go)
                better = go & (value < best)
                best = np.where(better, value, best)
                pick[better] = t
            points[vs] = here
            for t, at in enumerate(tries):
                points[vs[pick == t]] = at[pick == t]
            took = pick >= 0
            if took.any():
                fresh[b["faces"][took[b["owner"]]]] = False  # their middles moved
                moved += int(took.sum())
                changed = True
            yield
        if not changed:
            break
    return points, moved
