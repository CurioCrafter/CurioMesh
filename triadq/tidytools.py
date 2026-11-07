# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The maths of each Tidy Tool, as a `trial.Trial` subclass (plain NumPy, no bpy).

Smooth Corners is the small example every tool follows: it slides each free corner part of the way toward the middle
of its neighbours (umbrella smoothing, Taubin 1995 without the shrink step, as the projection undoes the shrink) and
puts it back on your original mesh after each step, so nothing shrinks and every corner stays on the source surface.

Cleanup: Preview Tidy lists what it finds (two-edge corners and doublets, Tarini et al. 2010; triangle pairs that make
good quads; slivers; faces turned the other way) with each line's fix worked out up front. Seal the Middle puts the
corners near the mirror plane onto it and merges doubled ones.
"""

from __future__ import annotations

import numpy as np

from .trial import Base, Candidate, Trial, edges_of, unfold

NOTHING_FREE = "Nothing can move here: every corner is held (borders, hard edges or the mirror plane)."


class SmoothCorners(Trial):
    name = "Smooth Corners"
    group = "RELAX"
    handles = ("strength",)
    words = ("0", "1")
    defaults = {"strength": 0.6, "steps": 6}
    blendable = True
    stages = ("poles", "move", "look")

    def check(self, base: Base) -> str:
        return "" if base.free.any() else NOTHING_FREE

    def run(self, base: Base, settings: dict, rough: bool = False):
        strength = float(settings.get("strength", self.defaults["strength"]))
        steps = int(settings.get("steps", self.defaults["steps"]))
        steps = max(1, steps // 4) if rough else max(1, steps)
        free = base.free
        co = base.co.copy()
        edges = edges_of(base.faces)
        count = np.bincount(edges.ravel(), minlength=len(co)).astype(np.float64)
        count[count == 0] = 1.0
        for step in range(steps):
            middle = np.stack([np.bincount(edges[:, 0], co[edges[:, 1], k], len(co)) + np.bincount(edges[:, 1], co[edges[:, 0], k], len(co)) for k in range(3)], axis=1) / count[:, None]
            co[free] += 0.5 * strength * (middle[free] - co[free])
            yield (step + 0.5) / steps
            if base.project is not None:
                co[free] = base.project(co[free])
            yield (step + 1.0) / steps
        return Candidate(unfold(base, co), strength=strength, exact=not rough)


# --------------------------------------------------------------------------- shared by the Cleanup tools


def _half_edges(faces):
    """(face of each corner, corner, next corner) over every face, as arrays."""
    from .organize import half_edges

    return half_edges(faces)


def _edge_users(a, b, count):
    """(edge key of each half-edge, how many half-edges share that edge)."""
    key = np.minimum(a, b) * (count + 1) + np.maximum(a, b)
    _u, inverse, users = np.unique(key, return_inverse=True, return_counts=True)
    return key, users[inverse.reshape(-1)]


def _open_corners(faces, count) -> np.ndarray:
    """Corners on an open edge (used by one face): the outline, which the Cleanup tools never change."""
    _f, a, b = _half_edges(faces)
    out = np.zeros(count, dtype=bool)
    if len(a):
        _key, users = _edge_users(a, b, count)
        out[a[users == 1]] = True
        out[b[users == 1]] = True
    return out


def _outside_scope(base) -> np.ndarray:
    """Corners used by a face outside the scope: the scope's edge, which never moves."""
    out = np.zeros(len(base.co), dtype=bool)
    for f in np.flatnonzero(~base.scope).tolist():
        out[list(base.faces[f])] = True
    return out


def _on_plane(base) -> np.ndarray:
    if base.mirror is None:
        return np.zeros(len(base.co), dtype=bool)
    axis, offset = base.mirror
    size = float(np.ptp(base.co, axis=0).max()) if len(base.co) else 1.0
    return np.abs(base.co[:, axis] - offset) < 1e-6 * (size or 1.0)


def _compact(co, faces, parent):
    """Drop corners no face uses any more. Returns (co, faces, parent)."""
    used = np.zeros(len(co), dtype=bool)
    for f in faces:
        used[list(f)] = True
    remap = np.cumsum(used) - 1
    return np.asarray(co)[used], [tuple(int(remap[v]) for v in f) for f in faces], np.asarray(parent, dtype=np.int64)


def _middles(co, faces, ids) -> np.ndarray:
    return np.asarray([co[list(faces[i])].mean(axis=0) for i in ids], dtype=np.float64).reshape(-1, 3)


def _quads_ok(co, quads) -> np.ndarray:
    """Convex, not crossed and not folded much (the shared quad test of `topology.py`)."""
    from .topology import _quad_points_valid

    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    if not len(quads):
        return np.zeros(0, dtype=bool)
    distinct = np.all(np.diff(np.sort(quads, axis=1), axis=1) != 0, axis=1)
    span = float(np.ptp(co, axis=0).max()) if len(co) else 1.0
    return distinct & _quad_points_valid(co[quads], 1e-12 * max(span, 1e-12) ** 2)


def _plural(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


# --------------------------------------------------------------------------- Preview Tidy: what each line finds and fixes


def two_edge_corners(base, quads: bool):
    """Corners inside the scope with only two edges, so their two faces fold onto each other. `quads`: only those
    between two quads (Tarini et al. 2010 call such a pair a doublet), else only the others (a triangle or n-gon on one
    side). Fix: the corner goes and its two faces become one. Returns (corners found, (co, faces, parent) or None,
    held), a face joined at most once per run (Again takes the rest)."""
    co, faces = base.co, base.faces
    n = len(co)
    fid, a, b = _half_edges(faces)
    if not len(a):
        return [], None, []
    key, _users = _edge_users(a, b, n)
    _k, first = np.unique(key, return_index=True)
    edge_count = np.bincount(a[first], minlength=n) + np.bincount(b[first], minlength=n)
    face_count = np.bincount(a, minlength=n)
    keep_out = _open_corners(faces, n) | _outside_scope(base) | _on_plane(base)
    found = np.flatnonzero((edge_count == 2) & (face_count == 2) & ~keep_out)
    at: dict[int, list[int]] = {}
    want = set(found.tolist())
    for f, v in zip(fid.tolist(), a.tolist()):
        if v in want:
            at.setdefault(v, []).append(f)
    corners = [v for v in found.tolist() if (len(faces[at[v][0]]) == 4 and len(faces[at[v][1]]) == 4) == quads]
    out_faces = list(faces)
    gone = np.zeros(len(faces), dtype=bool)
    held, touched, joined = [], set(), 0
    for v in corners:
        f, g = at[v]
        if f in touched or g in touched:
            continue
        F, G = list(faces[f]), list(faces[g])
        k, j = F.index(v), G.index(v)
        before, after = F[k - 1], F[(k + 1) % len(F)]
        if G[(j + 1) % len(G)] != before or G[j - 1] != after:
            held.append((co[v], "its two faces turn opposite ways"))
            continue
        if base.kinds is not None and base.kinds[f] != base.kinds[g]:
            held.append((co[v], "on a material or UV border"))
            continue
        part = [F[(k + 1 + i) % len(F)] for i in range(len(F) - 1)]  # after .. before, round F
        jb = G.index(before)
        middle = [G[(jb + 1 + i) % len(G)] for i in range(len(G) - 3)]  # strictly between before and after, round G
        merged = part + middle
        if len(merged) < 3 or len(set(merged)) != len(merged) or (len(merged) == 4 and not _quads_ok(co, [merged])[0]):
            held.append((co[v], "the joined face would be a bad shape"))
            continue
        out_faces[f] = tuple(merged)
        gone[g] = True
        touched.update((f, g))
        joined += 1
    if not joined:
        return corners, None, held
    keep = np.flatnonzero(~gone)
    return corners, _compact(co, [out_faces[i] for i in keep.tolist()], keep), held


def triangle_pairs(base):
    """Pairs of triangles in the scope that share an edge and make a good quad together, squarest quad first, each
    triangle in one pair at most (a greedy matching). Never across a material or UV border or along a mirror plane, and
    never a folded quad. Returns (pairs (k, 2), spots (k, 3), (co, faces, parent) or None)."""
    co, faces = base.co, base.faces
    nothing = (np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3)), None)
    tris = [i for i in np.flatnonzero(base.scope).tolist() if len(faces[i]) == 3]
    if len(tris) < 2:
        return nothing
    plane = _on_plane(base)
    by_edge: dict = {}
    for t in tris:
        f = faces[t]
        for k in range(3):
            p, q = f[k], f[(k + 1) % 3]
            by_edge.setdefault((min(p, q), max(p, q)), []).append((t, p, q, f[(k + 2) % 3]))
    pairs, quads = [], []
    for (p, q), users in by_edge.items():
        if len(users) != 2 or (plane[p] and plane[q]):
            continue
        (t, a, b, c), (u, a2, b2, d) = users
        if a2 != b or b2 != a:
            continue  # the two turn opposite ways
        if base.kinds is not None and base.kinds[t] != base.kinds[u]:
            continue
        pairs.append((t, u))
        quads.append((a, d, b, c))
    if not pairs:
        return nothing
    pairs, quads = np.asarray(pairs, dtype=np.int64), np.asarray(quads, dtype=np.int64)
    good = _quads_ok(co, quads)
    pairs, quads = pairs[good], quads[good]
    if not len(pairs):
        return nothing
    p = co[quads]
    u, w = np.roll(p, -1, axis=1) - p, np.roll(p, 1, axis=1) - p
    cos = np.abs((u * w).sum(axis=2)) / np.maximum(np.linalg.norm(u, axis=2) * np.linalg.norm(w, axis=2), 1e-300)
    used: set[int] = set()
    chosen = []
    for i in np.argsort(cos.max(axis=1), kind="stable").tolist():  # squarest first
        t, s = int(pairs[i, 0]), int(pairs[i, 1])
        if t in used or s in used:
            continue
        used.update((t, s))
        chosen.append(i)
    pairs, quads = pairs[chosen], quads[chosen]
    out_faces = list(faces)
    gone = np.zeros(len(faces), dtype=bool)
    for (t, s), quad in zip(pairs.tolist(), quads.tolist()):
        out_faces[t] = tuple(quad)
        gone[s] = True
    keep = np.flatnonzero(~gone)
    return pairs, co[quads].mean(axis=1), (co, [out_faces[i] for i in keep.tolist()], keep)


def slivers(base, steps: int = 8):
    """Thin needle faces in the scope (the Issue Finder's sliver test) and a gentler place for their corners: each free
    corner of a sliver slides toward the middle of its neighbours and back onto your original mesh, a few small steps.
    A face round them that would turn concave or thin puts its corners back. Returns (sliver faces, new co or None,
    held)."""
    from .issues import topology_flags

    co, faces = base.co, base.faces
    found = np.flatnonzero(topology_flags(co, faces)["sliver"] & base.scope)
    if not len(found):
        return found, None, []
    move = np.zeros(len(co), dtype=bool)
    for f in found.tolist():
        move[list(faces[f])] = True
    move &= base.free
    stuck = np.asarray([not move[list(faces[f])].any() for f in found.tolist()], dtype=bool)
    held = [(p, "every corner is held") for p in _middles(co, faces, found[stuck])]
    if not move.any():
        return found, None, held
    edges = edges_of(faces)
    count = np.maximum(np.bincount(edges.ravel(), minlength=len(co)), 1).astype(np.float64)
    out = co.copy()
    for _ in range(steps):
        middle = np.stack([np.bincount(edges[:, 0], out[edges[:, 1], k], len(co)) + np.bincount(edges[:, 1], out[edges[:, 0], k], len(co)) for k in range(3)], axis=1) / count[:, None]
        out[move] += 0.5 * (middle[move] - out[move])
        if base.project is not None:
            out[move] = base.project(out[move])
    near = np.flatnonzero(np.fromiter((move[list(f)].any() for f in faces), dtype=bool, count=len(faces)))
    sub = [faces[i] for i in near.tolist()]
    old, new = topology_flags(co, sub), topology_flags(out, sub)
    worse = (new["concave"] & ~old["concave"]) | (new["sliver"] & ~old["sliver"])
    for i in np.flatnonzero(worse).tolist():
        ids = list(sub[i])
        out[ids] = co[ids]
    out = unfold(base, out)
    still = topology_flags(out, faces)["sliver"][found] & ~stuck
    held += [(p, "could not open it up") for p in _middles(out, faces, found[still])]
    if np.allclose(out, co):
        return found, None, held
    return found, out, held


def facing_in(base):
    """Faces in the scope that turn the other way from most of their neighbours (their shared edges run the same way
    instead of opposite ways). Fix: turn them round. Returns (faces, (co, faces, parent) or None)."""
    faces = base.faces
    fid, a, b = _half_edges(faces)
    if not len(a):
        return np.zeros(0, dtype=np.int64), None
    key, users = _edge_users(a, b, len(base.co))
    two = users == 2
    order = np.argsort(key[two], kind="stable")
    f2, a2 = fid[two][order], a[two][order]
    same = a2[0::2] == a2[1::2]  # both faces walk the edge from the same corner: they disagree
    bad, good = np.zeros(len(faces)), np.zeros(len(faces))
    for side in (f2[0::2], f2[1::2]):
        np.add.at(bad, side[same], 1)
        np.add.at(good, side[~same], 1)
    found = np.flatnonzero((bad > good) & base.scope)
    if not len(found):
        return found, None
    out = list(faces)
    for f in found.tolist():
        out[f] = tuple(reversed(faces[f]))
    return found, (base.co, out, np.arange(len(faces)))


NOTHING_TO_TIDY = "Nothing to tidy here: no two-edge corners, doublets, triangle pairs, slivers or inward faces."
TIDY_LINES = (  # (key, one, many) in the order the report lists them
    ("corners", "corner with two edges", "corners with two edges"),
    ("doublets", "doublet", "doublets"),
    ("pairs", "triangle pair that makes a quad", "triangle pairs that make quads"),
    ("slivers", "sliver", "slivers"),
    ("inward", "face facing in", "faces facing in"),
)


class PreviewTidy(Trial):
    """A findings report: one line per kind of problem actually found in the scope, each with a count, its look cost,
    Show and Fix These. Each line's fix is worked out up front and kept on the base, so picking a line shows its ghost
    at once; the Session measures each line's look cost after."""

    name = "Preview Tidy"
    group = "CLEANUP"
    handles = ()
    defaults = {"row": ""}
    blendable = False
    stay_open = True  # after Enter the list refreshes for the next line
    layers = ("scope", "ghost", "wash", "arrows", "poles", "held", "found")
    stages = ("faces", "poles", "look")

    def lines(self, base):
        """Generator: the report's lines (dicts), worked out once per result."""
        if "tidy_lines" in base.cache:
            return base.cache["tidy_lines"]
        found: dict = {}
        for key, quads in (("corners", False), ("doublets", True)):
            corners, fix, held = two_edge_corners(base, quads)
            if corners:
                found[key] = (len(corners), base.co[corners], fix, held)
            yield 0.2 if quads else 0.1
        pairs, spots, fix = triangle_pairs(base)
        if len(pairs):
            found["pairs"] = (len(pairs), spots, fix, [])
        yield 0.4
        thin, moved, held = slivers(base)
        if len(thin):
            found["slivers"] = (len(thin), _middles(base.co, base.faces, thin), None if moved is None else (moved, None, None), held)
        yield 0.7
        inward, fix = facing_in(base)
        if len(inward):
            found["inward"] = (len(inward), _middles(base.co, base.faces, inward), fix, [])
        yield 0.9
        rows = []
        for key, one, many in TIDY_LINES:
            if key in found:
                count, spots, fix, held = found[key]
                rows.append({"key": key, "words": _plural(count, one, many), "count": count, "spots": np.asarray(spots, dtype=np.float64).reshape(-1, 3), "fix": fix, "held": list(held), "cost": None})
        base.cache["tidy_lines"] = rows
        return rows

    def run(self, base: Base, settings: dict, rough: bool = False):
        rows = yield from self.lines(base)
        if not rows:
            return Candidate(None, refusal=NOTHING_TO_TIDY)
        row = next((r for r in rows if r["key"] == settings.get("row", "")), None)
        if row is None:
            return Candidate(None, rows=rows, note="Show lights up where; Fix These shows the fix. Nothing changes until Enter.")
        if row["fix"] is None:
            return Candidate(None, rows=rows, held=list(row["held"]), found=row["spots"], note=f"The {row['words'].split(' ', 1)[1]} here cannot be fixed without bad faces.")
        co, faces, parent = row["fix"]
        return Candidate(np.asarray(co, dtype=np.float64), faces=None if faces is None else list(faces), parent=parent, held=list(row["held"]), found=row["spots"], rows=rows, note=f"Fix These: {row['words']}")


# --------------------------------------------------------------------------- Seal the Middle

NOT_MIRRORED = "This result is not mirrored, and no mirror plane was found on it."
SEALED = "Nothing to seal: every vertex near the mirror plane is on it and none are doubled."
REACH = 0.2  # of an edge: corners this near the plane go onto it
GAP_REACH = 0.6  # of an edge: corners on an open edge this near the plane go onto it too (a gap along the middle)
MERGE = 0.1  # of an edge: corners on the plane closer than this become one
TWIN = 0.3  # of an edge: a corner with another this near its mirror image is one of a pair, not a stray seam corner


def plane_of(base):
    """(axis, offset) of the mirror plane: the result's own, else the one found on it (`pick["plane"]`), else None."""
    if base.mirror is not None:
        return base.mirror
    plane = base.pick.get("plane")
    return None if plane is None else (int(plane[0]), float(plane[1]))


def _has_twin(co, ids, axis, offset, edge, tiny, chunk=256):
    """Generator: bool per corner of `ids`, True when another corner sits at its mirror image (within TWIN of an edge).
    Those are a real pair either side of the middle, not a seam corner that strayed, so they stay where they are."""
    out = np.zeros(len(co), dtype=bool)
    if not len(ids):
        return out
    d = np.abs(co[:, axis] - offset)
    others = np.flatnonzero((d > tiny) & (d <= GAP_REACH * edge + TWIN * edge))
    mirrored = co[ids].copy()
    mirrored[:, axis] = 2.0 * offset - mirrored[:, axis]
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        far = np.linalg.norm(mirrored[start : start + chunk, None, :] - co[others][None, :, :], axis=2)
        far[part[:, None] == others[None, :]] = np.inf
        out[part] = far.min(axis=1, initial=np.inf) < TWIN * edge
        yield
    return out


def _merge_close(co, ids, reach):
    """Union-find roots for corners `ids` (all on the plane) closer than `reach`: sorted along one axis, so each corner
    is only compared with the few after it."""
    root = {int(v): int(v) for v in ids}
    if len(ids) < 2:
        return root

    def top(v):
        while root[v] != v:
            root[v] = root[root[v]]
            v = root[v]
        return v

    p = co[ids]
    axis = int(np.argmax(np.ptp(p, axis=0)))
    order = np.argsort(p[:, axis], kind="stable")
    ids, p = np.asarray(ids)[order], p[order]
    for i in range(len(ids)):
        j = i + 1
        while j < len(ids) and p[j, axis] - p[i, axis] < reach:
            if np.linalg.norm(p[j] - p[i]) < reach:
                a, b = top(int(ids[i])), top(int(ids[j]))
                if a != b:
                    root[max(a, b)] = min(a, b)
            j += 1
    return {v: top(v) for v in root}


def _grid_patch(co, axis, offset, near, edge, lines=6):
    """Pink grid lines on the mirror plane over the seam: (k, 2, 3) segments."""
    if not near.any():
        return np.zeros((0, 2, 3))
    u, w = [k for k in range(3) if k != axis]
    lo, hi = co[near].min(axis=0) - edge, co[near].max(axis=0) + edge
    out = []
    for t in np.linspace(0.0, 1.0, lines):
        for along, across in ((u, w), (w, u)):
            a = np.zeros(3)
            a[axis] = offset
            a[across] = lo[across] + t * (hi[across] - lo[across])
            b = a.copy()
            a[along], b[along] = lo[along], hi[along]
            out.append((a, b))
    return np.asarray(out, dtype=np.float64)


class SealMiddle(Trial):
    """Puts the corners of the scope near the mirror plane exactly onto it, merges doubled ones, and reports the faces
    still crossing the plane and the gaps still open along it. Exact, so no handle."""

    name = "Seal the Middle"
    group = "CLEANUP"
    handles = ()
    defaults: dict = {}
    blendable = False
    layers = ("scope", "ghost", "arrows", "held", "found", "kept")
    stages = ("faces", "look")

    def check(self, base: Base) -> str:
        return "" if plane_of(base) is not None else NOT_MIRRORED

    def run(self, base: Base, settings: dict, rough: bool = False):
        axis, offset = plane_of(base)
        co, faces, edge = base.co, base.faces, base.edge
        size = float(np.ptp(co, axis=0).max()) if len(co) else 1.0
        tiny = 1e-6 * (size or 1.0)
        mine = np.zeros(len(co), dtype=bool)
        for f in np.flatnonzero(base.scope).tolist():
            mine[list(faces[f])] = True
        mine &= ~_outside_scope(base)
        d = np.abs(co[:, axis] - offset)
        open_ = _open_corners(faces, len(co))
        go = mine & (d > tiny) & ((d <= REACH * edge) | (open_ & (d <= GAP_REACH * edge)))
        go &= ~(yield from _has_twin(co, np.flatnonzero(go), axis, offset, edge, tiny))
        out = co.copy()
        if go.any():
            out[go, axis] = offset
            if base.project is not None:  # back onto your original mesh, then exactly onto the plane again
                out[go] = base.project(out[go])
                out[go, axis] = offset
        yield 0.4
        on = mine & (np.abs(out[:, axis] - offset) <= tiny)
        root = _merge_close(out, np.flatnonzero(on), MERGE * edge)
        merged = sum(1 for v, r in root.items() if v != r)
        yield 0.7
        remap = np.arange(len(co))
        for v, r in root.items():
            remap[v] = r
        new_faces, parent, seen = [], [], set()
        for i, f in enumerate(faces):
            g = [int(remap[v]) for v in f]
            g = [v for k, v in enumerate(g) if v != g[k - 1]]
            if len(g) < 3 or len(set(g)) != len(g):
                continue  # squeezed to nothing by a merge
            key = tuple(sorted(g))
            if key in seen:
                continue  # the same face twice
            seen.add(key)
            new_faces.append(tuple(g))
            parent.append(i)
        moved = int(go.sum())
        if not moved and not merged:
            return Candidate(None, refusal=SEALED)
        links = [(out[v], out[r]) for v, r in root.items() if v != r]
        arrows = np.asarray([(co[v], out[v]) for v in np.flatnonzero(go)] + links, dtype=np.float64).reshape(-1, 2, 3)
        out, new_faces, parent = _compact(out, new_faces, parent)
        side = out[:, axis] - offset
        held = []
        for f in new_faces:
            s = side[list(f)]
            if (s > tiny).any() and (s < -tiny).any():
                held.append((out[list(f)].mean(axis=0), "crosses the middle"))
        crossing = len(held)
        both = (side > tiny).any() and (side < -tiny).any()  # a whole model: an open edge on the plane is a gap
        gaps = 0
        if both:
            _f, a, b = _half_edges(new_faces)
            _key, users = _edge_users(a, b, len(out))
            flat = np.abs(side) <= tiny
            for p, q in zip(a[users == 1].tolist(), b[users == 1].tolist()):
                if flat[p] and flat[q]:
                    held.append(((out[p] + out[q]) / 2.0, "gap"))
                    gaps += 1
        note = f"{moved:,} seam vert{'ex' if moved == 1 else 'ices'} put on the plane, {merged:,} merged, {_plural(gaps, 'gap', 'gaps')} left"
        if crossing:
            note += f", {_plural(crossing, 'face crosses', 'faces cross')} it"
        return Candidate(out, faces=new_faces, parent=parent, held=held, found=co[go] if moved else None, kept=_grid_patch(out, axis, offset, np.abs(side) <= tiny, edge), arrows=arrows, note=note)


# --------------------------------------------------------------------------- Look First for tools that already existed


def swap_parent(old_piece, new_piece) -> np.ndarray:
    """For each face of a result with some parts swapped, a face of the result before it in the same part (the first
    one), so the preview can tell the swapped parts' new faces from the rest. -1 where the part had no face before."""
    old_piece, new_piece = np.asarray(old_piece, dtype=np.int64), np.asarray(new_piece, dtype=np.int64)
    if not len(new_piece):
        return np.zeros(0, dtype=np.int64)
    top = int(max(old_piece.max(initial=-1), new_piece.max(initial=-1))) + 1
    first = np.full(top, -1, dtype=np.int64)
    order = np.arange(len(old_piece), dtype=np.int64)[::-1]
    first[old_piece[order]] = order  # the last write wins: the first face of each part
    return first[new_piece]


class SwapParts(Trial):
    """Some parts of the result with other faces: a ready Quad Size, your own faces, or the reduced ones again. The
    faces are worked out before the tool opens (they already exist as a part's try); this only shows them."""

    name = "Swap Parts"
    group = "LOOPS"
    handles = ()
    defaults: dict = {}
    blendable = False
    layers = ("scope", "ghost", "poles", "held")
    stages = ("faces", "poles", "look")
    swap = None  # (co, faces, new piece per face, old piece per face, parts swapped): set on a copy of the class

    def check(self, base: Base) -> str:
        if self.swap is None:
            return "Nothing to swap here: these parts already show these faces."
        return ""

    def run(self, base: Base, settings: dict, rough: bool = False):
        co, faces, new_piece, old_piece, parts = self.swap
        parent = swap_parent(old_piece, new_piece)
        yield 0.5
        inside = np.isin(np.asarray(new_piece), list(parts))
        parent = np.where(parent >= 0, parent, 0)
        before, after = int(base.scope.sum()), int(inside.sum())
        word = "parts" if len(parts) != 1 else "part"
        return Candidate(np.asarray(co, dtype=np.float64), faces=list(faces), parent=parent, strength=1.0, note=f"this {word} {before:,} -> {after:,} faces")


# Detail Markers. Decimate weight drop -> about how many times the faces it keeps (the numbers `budget.density_drop`
# is written from); a marker at strength s drops the weight by 0.02 * Strength * s at its middle.
MARKER_DROP = ((0.0, 1.0), (0.01, 3.0), (0.02, 4.0), (0.1, 6.0))
GRID_MOST = 40  # sample quads across the ring at most: a drawing cap so a tiny size never draws millions of lines


def marker_faces_factor(strength: float, density: float = 1.0) -> float:
    """About how many times the faces a marker asks for at its middle (More Detail; Less Detail asks for as many
    times fewer). An estimate: the search still decides."""
    drop = min(0.02 * max(float(density), 0.0) * abs(float(strength)), 0.5)
    x, y = zip(*MARKER_DROP)
    return float(np.interp(drop, x, y))


def marker_scale(strength: float, density: float = 1.0, less: bool = False) -> float:
    """How many times longer a quad's side gets at the marker's middle: under 1 for More Detail, over 1 for Less."""
    side = float(np.sqrt(marker_faces_factor(strength, density)))
    return side if less else 1.0 / side


def marker_extra_faces(radius: float, edge: float, strength: float, density: float = 1.0, less: bool = False) -> int:
    """About how many faces the marker adds (negative: takes away) on a flat disc of `radius` covered by quads of side
    `edge`. The ask falls off as the ball's weight does (1 in the middle, 0 at the edge: (1 - d^2)^2), ring by ring."""
    if radius <= 0.0 or edge <= 0.0:
        return 0
    rings = np.linspace(0.0, 1.0, 33)
    mid = 0.5 * (rings[:-1] + rings[1:])
    area = np.pi * radius**2 * (rings[1:] ** 2 - rings[:-1] ** 2)
    weight = (1.0 - mid**2) ** 2
    factor = np.array([marker_faces_factor(strength * w, density) for w in weight])
    now = area / edge**2
    after = now / factor if less else now * factor
    return int(round(float((after - now).sum())))


def sample_grid(radius: float, size: float):
    """A square grid of quads of side `size` inside a circle of `radius`, in its own plane: ((k, 2) points, (m, 4)
    quads). The middle of the circle is a grid corner. At most GRID_MOST quads across (bigger squares past that)."""
    if radius <= 0.0 or size <= 0.0:
        return np.zeros((0, 2)), np.zeros((0, 4), dtype=np.int64)
    size = max(size, 2.0 * radius / GRID_MOST)
    n = int(np.ceil(radius / size))
    ticks = np.arange(-n, n + 1) * size
    u, v = np.meshgrid(ticks, ticks, indexing="xy")
    points = np.stack((u.ravel(), v.ravel()), axis=1)
    side = 2 * n + 1
    i, j = np.meshgrid(np.arange(2 * n), np.arange(2 * n), indexing="xy")
    a = (j * side + i).ravel()
    quads = np.stack((a, a + 1, a + side + 1, a + side), axis=1)
    centre = points[quads].mean(axis=1)
    keep = (centre**2).sum(axis=1) <= radius**2
    quads = quads[keep]
    used = np.unique(quads)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[quads]


# Hard Edges on Apply: which edges of the result Apply will mark sharp (or hold with loops), shown before Apply.
ANGLE_LOW, ANGLE_HIGH = 5.0, 95.0  # degrees at the dial's two ends (the setting itself goes to 180 in the panel)


def edge_angles(co, faces) -> tuple[np.ndarray, np.ndarray]:
    """((e, 2) edges used by exactly two faces, (e,) degrees between those two faces' normals). Edges on a border or
    shared by more than two faces are left out, as Apply leaves them out."""
    co = np.asarray(co, dtype=np.float64)
    if not faces:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
    normals = np.zeros((len(faces), 3))
    by_size: dict[int, list] = {}
    for i, f in enumerate(faces):
        by_size.setdefault(len(f), []).append(i)
    rows, pairs = [], []
    for _size, which in by_size.items():  # one NumPy pass per face size (quads, triangles...)
        which = np.asarray(which, dtype=np.int64)
        ids = np.asarray([faces[i] for i in which.tolist()], dtype=np.int64)
        p = co[ids]
        normals[which] = np.cross(p, np.roll(p, -1, axis=1)).sum(axis=1)  # Newell's normal
        nxt = np.roll(ids, -1, axis=1)
        pairs.append(np.stack((np.minimum(ids, nxt).ravel(), np.maximum(ids, nxt).ravel()), axis=1))
        rows.append(np.repeat(which, ids.shape[1]))
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)
    pairs, rows = np.concatenate(pairs), np.concatenate(rows)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, rows = pairs[order], rows[order]
    found, start, count = np.unique(pairs, axis=0, return_index=True, return_counts=True)
    two = count == 2
    a, b = rows[start[two]], rows[start[two] + 1]
    cos = np.clip((normals[a] * normals[b]).sum(axis=1), -1.0, 1.0)
    return found[two], np.degrees(np.arccos(cos))


def angle_of(value: float) -> float:
    """Degrees for a dial value 0..1."""
    return ANGLE_LOW + (ANGLE_HIGH - ANGLE_LOW) * min(max(float(value), 0.0), 1.0)


def value_of(angle: float) -> float:
    """The dial value 0..1 for an angle in degrees."""
    return min(max((float(angle) - ANGLE_LOW) / (ANGLE_HIGH - ANGLE_LOW), 0.0), 1.0)


class HardEdges(Trial):
    """Every edge of the result whose two faces meet at more than the angle, drawn pink: what Apply marks sharp (or
    holds with loops). Nothing moves; Enter keeps the angle for Apply."""

    name = "Hard Edges"
    group = "HARD"
    handles = ("angle",)
    words = (f"{ANGLE_LOW:.0f}", f"{ANGLE_HIGH:.0f}")
    defaults = {"angle": value_of(40.0)}
    blendable = False
    layers = ("scope", "kept")
    stages = ()

    def value_text(self, value: float) -> str:
        return f"Angle {angle_of(value):.0f} degrees"

    def check(self, base: Base) -> str:
        return "" if len(base.faces) > 1 else "No hard edges in this area."

    def run(self, base: Base, settings: dict, rough: bool = False):
        if "edge_angles" not in base.cache:
            base.cache["edge_angles"] = edge_angles(base.co, base.faces)
            yield 0.8
        edges, angles = base.cache["edge_angles"]
        angle = angle_of(settings.get("angle", self.defaults["angle"]))
        hard = edges[angles > angle]
        word = "edge" if len(hard) == 1 else "edges"
        return Candidate(base.co.copy(), strength=float(settings.get("angle", 0.0)), lines=base.co[hard], note=f"{len(hard):,} hard {word} at {angle:.0f} degrees, in pink")
