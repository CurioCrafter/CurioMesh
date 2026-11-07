# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The four Relax tools of Tidy Tools, as Look First trials (plain NumPy, no bpy).

All four slide corners in their own tangent plane and put them back on your original mesh after every step (the
same-side rule of `organize._onto`), so nothing shrinks and every corner stays on the source surface. They differ in
where each corner is pulled:

- Source Relax: to the area-weighted middle of the quads round it (tangential Laplacian smoothing as used in isotropic
  remeshing, Botsch and Kobbelt 2004; the Laplacian of Desbrun, Meyer, Schroeder and Barr 1999).
- Keep Lines Relax: as Source Relax, except that corners on a kept line (hard edges, the outline, UV seams, part
  borders, guides) only slide along that line, toward the middle of their two line neighbours (a 1D relax by arc
  length, the feature-edge rule of Botsch and Kobbelt 2004); corners where lines meet stay put.
- Square Up: each quad pulls its corners toward a rectangle with its own two side lengths, set along its own two row
  directions; only the corner angles change (a shape-matching pull, Mueller et al. 2005, in damped steps,
  Mueller et al. 2007).
- Relax Toward Detail: to the middle of the quads round it weighted by area times a density read from how much your
  original bends under each quad (centroidal relaxation with a density, Du, Faber and Gunzburger 1999; used for
  remeshing by Alliez et al. 2003). Quads get smaller where it bends and bigger where it is flat, with the same faces.

Every step is guarded as Tidy Up's settle is: a corner moves at most `MOST_MOVE` of its edges per step and
`MOST_DRIFT` in a run, and a quad that would turn past `WORST_TURN` or fold puts its corners back. Keep going repeats
runs from where the last one ended until a run moves no corner by more than `SETTLED` of an edge.
"""

from __future__ import annotations

import numpy as np

from .organize import MOST_DRIFT, MOST_MOVE, SHARP_DEG, WORST_TURN, _bent, _normals, _onto, _quads, _unit, _vertex_normals, half_edges, pinned
from .tidytools import NOTHING_FREE
from .trial import Base, Candidate, Trial, edges_of

STEP = 0.5  # share of the way to the pull's goal per step, at strength 1
SQUARE_DEG = 10.0  # a quad corner this close to 90 degrees counts as square
SETTLED = 0.005  # of an edge: Keep going stops when a whole run moved no corner further than this
DETAIL_GAIN = 2.0  # at Detail = 1, how much denser quads get on the most-bent areas than on flat ones (as a density)
EVEN_CURVES = 1.15  # Relax Toward Detail refuses when the densest and the thinnest tenth differ less than this
CREASE_REACH = 0.1  # of an edge: how far a corner on a kept line may leave the line it started on
LINE_KINDS = (("hard", "Hard Edges"), ("outline", "Outline"), ("seams", "UV Seams"), ("parts", "Part Borders"), ("guides", "Guides"))
NO_SLANT = "No slanted quads here: every corner is already within 10 degrees of square."
EVEN_BENDS = "Your mesh is about equally curved everywhere here, so sizes would not change."
NO_LINES = "No lines to keep here: this works like Source Relax."
BAD_FACES = "Kept as it was: this would add {n} bad faces (folded or facing in)."


def _scope_quads(base: Base) -> np.ndarray:
    quads, ids = _quads(base.faces)
    return quads[base.scope[ids]] if len(ids) else quads


def _mean_edge(co: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Per corner: the mean length of its quad edges (1 where it has none)."""
    a, b = quads.reshape(-1), np.roll(quads, -1, axis=1).reshape(-1)
    length = np.linalg.norm(co[a] - co[b], axis=1)
    total = np.bincount(a, length, len(co)) + np.bincount(b, length, len(co))
    count = np.bincount(a, minlength=len(co)) + np.bincount(b, minlength=len(co))
    return np.where(count > 0, total / np.maximum(count, 1), 1.0)


def _areas(co: np.ndarray, quads: np.ndarray) -> np.ndarray:
    p = co[quads]
    return 0.5 * np.linalg.norm(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]), axis=1)


def _weighted_middle(co: np.ndarray, quads: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Per corner: the middle of the quads round it, each counted by `weight` (corners with no quad stay)."""
    centre = co[quads].mean(axis=1)
    flat = quads.reshape(-1)
    w = np.repeat(weight, 4)
    total = np.stack([np.bincount(flat, w * np.repeat(centre[:, k], 4), len(co)) for k in range(3)], axis=1)
    sums = np.bincount(flat, w, len(co))
    out = co.copy()
    has = sums > 1e-30
    out[has] = total[has] / sums[has, None]
    return out


def bad_faces(co: np.ndarray, quads: np.ndarray, start: np.ndarray) -> int:
    """Quads facing the other way from where they started, or concave/folded."""
    if not len(quads):
        return 0
    return int((((_normals(co, quads) * start).sum(axis=1) < 0.0) | _bent(co, quads)).sum())


def square_share(co: np.ndarray, quads: np.ndarray) -> float:
    """Percent of quad corners within `SQUARE_DEG` of 90 degrees."""
    if not len(quads):
        return 100.0
    p = co[quads]
    out, back = np.roll(p, -1, axis=1) - p, np.roll(p, 1, axis=1) - p
    cos = (out * back).sum(axis=2) / np.maximum(np.linalg.norm(out, axis=2) * np.linalg.norm(back, axis=2), 1e-30)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return float(100.0 * (np.abs(angle - 90.0) <= SQUARE_DEG).mean())


def size_range(co: np.ndarray, quads: np.ndarray) -> float:
    """How much bigger the biggest tenth of quads is than the smallest tenth, by side length."""
    if len(quads) < 2:
        return 1.0
    side = np.sqrt(_areas(co, quads))
    low, high = np.percentile(side, (10.0, 90.0))
    return float(high / max(low, 1e-30))


def square_pull(co: np.ndarray, quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per quad corner: the move to a rectangle with the quad's own two side lengths, centred where the quad is and laid
    along the quad's own two row directions (the average of the "along" direction and the "across" one turned a
    quarter), so only the corner angles change. Returns (moves, corner of each move)."""
    p = co[quads]
    centre = p.mean(axis=1)
    n = _unit(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]))
    along = _unit(0.5 * (p[:, 1] + p[:, 2] - p[:, 0] - p[:, 3]))
    across = _unit(0.5 * (p[:, 2] + p[:, 3] - p[:, 0] - p[:, 1]))
    t1 = along - np.cross(n, across)  # both estimates of the along axis; the right angle sits between them
    t1 = _unit(t1 - (t1 * n).sum(axis=1, keepdims=True) * n)
    t2 = np.cross(n, t1)
    side = np.linalg.norm(np.roll(p, -1, axis=1) - p, axis=2)
    a, b = 0.5 * (side[:, 0] + side[:, 2]), 0.5 * (side[:, 1] + side[:, 3])
    sx, sy = np.array([-0.5, 0.5, 0.5, -0.5]), np.array([-0.5, -0.5, 0.5, 0.5])
    goal = centre[:, None, :] + (a[:, None] * sx)[:, :, None] * t1[:, None, :] + (b[:, None] * sy)[:, :, None] * t2[:, None, :]
    return (goal - p).reshape(-1, 3), quads.reshape(-1)


def _edge_faces(faces: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(edges (e, 2), first face, second face) of every edge two faces share."""
    if not faces:
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 2), dtype=np.int64), empty, empty
    size = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    a = np.fromiter((v for f in faces for v in f), dtype=np.int64, count=int(size.sum()))
    owner = np.repeat(np.arange(len(faces)), size)
    nxt = np.arange(len(a)) + 1
    nxt[np.cumsum(size) - 1] = np.cumsum(size) - size  # the last corner of a face wraps to its first
    b = a[nxt]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * (int(a.max()) + 1) + hi
    order = np.argsort(key, kind="stable")
    key, owner = key[order], owner[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    count = np.diff(np.r_[starts, len(key)])
    two = starts[count == 2]
    return np.stack((lo[order][two], hi[order][two]), axis=1), owner[two], owner[two + 1]


def _face_normals(co: np.ndarray, faces: list) -> np.ndarray:
    """Unit normals of the faces (Newell's normal, any corner count)."""
    out = np.zeros((len(faces), 3))
    by_size: dict = {}
    for i, f in enumerate(faces):
        by_size.setdefault(len(f), []).append(i)
    for ids in by_size.values():
        q = co[np.asarray([faces[i] for i in ids], dtype=np.int64)]
        out[ids] = np.cross(q, np.roll(q, -1, axis=1)).sum(axis=1)
    return _unit(out)


def outline_edges(co: np.ndarray, faces: list, views: int = 2) -> np.ndarray:
    """Edges on the outline (one face toward the camera, the other away) seen along at least `views` of the camera's
    lines of sight (a side and its opposite are one line of sight)."""
    from .views import SIDES

    edges, f0, f1 = _edge_faces(faces)
    if not len(edges):
        return edges
    normal = _face_normals(co, faces)
    look = np.asarray(SIDES, dtype=np.float64)
    # a flip seen from one side is also seen from the opposite side: count each line of sight once
    first = look[np.arange(len(look)), np.argmax(look != 0.0, axis=1)]
    look = _unit(look[first > 0.0])
    count = ((normal[f0] @ look.T > 0.0) != (normal[f1] @ look.T > 0.0)).sum(axis=1)
    return edges[count >= views]


def hard_edges(co: np.ndarray, faces: list, sharp_deg: float = SHARP_DEG) -> np.ndarray:
    """Edges whose two faces fold more than `sharp_deg` (the rule `organize.pinned` holds corners by)."""
    edges, f0, f1 = _edge_faces(faces)
    normal = _face_normals(co, faces)
    return edges[(normal[f0] * normal[f1]).sum(axis=1) < np.cos(np.radians(sharp_deg))]


def label_edges(faces: list, label) -> np.ndarray:
    """Edges between two faces with different labels (UV islands, parts)."""
    label = np.asarray(label)
    edges, f0, f1 = _edge_faces(faces)
    return edges[label[f0] != label[f1]]


def kept_lines(base: Base, kinds: dict) -> np.ndarray:
    """(k, 2) edges of the scope that are kept lines of the ticked kinds."""
    faces = base.faces
    labels = getattr(base, "labels", None) or {}
    found = []
    if kinds.get("hard"):
        found.append(hard_edges(base.co, faces))
    if kinds.get("outline"):
        found.append(outline_edges(base.co, faces))
    for key, name in (("seams", "chart"), ("parts", "piece")):
        if kinds.get(key) and labels.get(name) is not None and len(labels[name]) == len(faces):
            found.append(label_edges(faces, labels[name]))
    if kinds.get("guides") and labels.get("guide") is not None and len(labels["guide"]) == len(base.co):
        on = np.asarray(labels["guide"], dtype=bool)
        every = edges_of(faces)
        found.append(every[on[every[:, 0]] & on[every[:, 1]]])
    if not found:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.unique(np.sort(np.concatenate(found), axis=1), axis=0)
    inside = edges_of([faces[f] for f in np.flatnonzero(base.scope).tolist()])
    if not len(inside) or not len(edges):
        return np.zeros((0, 2), dtype=np.int64)
    n = len(base.co) + 1
    return edges[np.isin(edges[:, 0] * n + edges[:, 1], inside[:, 0] * n + inside[:, 1])]


BEND_CHUNK = 2048  # quads whose original normals are read per slice (Blender's own lookup is one call per point)


def bend_steps(base: Base):
    """Generator: reads the bend under each quad of the scope into `base.cache["bend"]`, a chunk per step, so a big
    result never stalls the preview while it is read."""
    if "bend" in base.cache:
        return
    quads = _scope_quads(base)
    co = base.co
    if not len(quads):
        base.cache["bend"] = np.zeros(len(co))
        return
    p = co[quads]
    samples = np.concatenate((p, p.mean(axis=1, keepdims=True), 0.5 * (p + np.roll(p, -1, axis=1))), axis=1)  # (k, 9, 3)
    facing = getattr(base, "facing", None)
    if facing is not None:
        normals = np.zeros((len(quads), 9, 3))
        for i in range(0, len(quads), BEND_CHUNK):
            normals[i : i + BEND_CHUNK] = np.asarray(facing(samples[i : i + BEND_CHUNK].reshape(-1, 3)), dtype=np.float64).reshape(-1, 9, 3)
            yield
    else:  # no original to read: the result's own corner normals stand in
        normals = _vertex_normals(co, quads)[quads]
    spread = np.arccos(np.clip(np.linalg.norm(_unit(normals).mean(axis=1), axis=1), 0.0, 1.0))
    # a quad's share: its bend for its size, so small quads on a curve count as much as big ones on it
    per_quad = spread / np.maximum(np.sqrt(_areas(co, quads)), 1e-30)
    high = float(np.percentile(per_quad, 95.0)) or 1.0
    per_quad = np.clip(per_quad / high, 0.0, 1.0)
    total = np.bincount(quads.reshape(-1), np.repeat(per_quad, 4), len(co))
    count = np.bincount(quads.reshape(-1), minlength=len(co))
    base.cache["bend"] = np.where(count > 0, total / np.maximum(count, 1), 0.0)


def detail_density(base: Base, mix: float) -> np.ndarray:
    """Per corner: how densely quads should sit there, 1 on the flattest areas, up to 1 + DETAIL_GAIN * mix where your
    original bends most. Read once from the spread of your original's normals under each quad of the scope."""
    for _ in bend_steps(base):
        pass
    return 1.0 + DETAIL_GAIN * float(mix) * base.cache["bend"]


class _Relax(Trial):
    """The shared guarded loop. A subclass writes `goal(x, quads, settings, base)`: where each corner is pulled."""

    group = "RELAX"
    handles = ("strength",)
    words = ("0", "1")
    blendable = True
    steps = 8

    def check(self, base: Base) -> str:
        return "" if self._free(base, dict(self.defaults)).any() else NOTHING_FREE

    def _free(self, base: Base, settings: dict) -> np.ndarray:
        return base.free

    def _lines(self, base: Base, settings: dict):
        """(line neighbours (n, 2) or None, corners that slide along a line)."""
        return None, np.zeros(len(base.co), dtype=bool)

    def goal(self, x, quads, settings, base):
        raise NotImplementedError

    def _strength(self, settings: dict) -> float:
        return float(settings.get("strength", self.defaults.get("strength", 1.0)))

    def run(self, base: Base, settings: dict, rough: bool = False):
        settings = dict(self.defaults, **settings)
        quads = _scope_quads(base)
        free = self._free(base, settings)
        nb, slide = self._lines(base, settings)
        strength = self._strength(settings)
        steps = int(settings.get("steps", self.steps))
        steps = max(1, steps // 4) if rough else max(1, steps)
        keep_going = bool(settings.get("keep_going", False)) and not rough
        co0 = base.co
        x = co0.copy()
        start = _normals(co0, quads) if len(quads) else np.zeros((0, 3))
        was_bent = _bent(co0, quads) if len(quads) else np.zeros(0, dtype=bool)
        bad_before = bad_faces(co0, quads, start)
        mean = _mean_edge(co0, quads) if len(quads) else np.ones(len(co0))
        reach = MOST_MOVE * mean
        held = np.zeros(len(co0), dtype=bool)
        moving = free | slide
        if strength <= 0.0 or not moving.any() or not len(quads):
            return self._finish(base, x, quads, settings, strength, rough, held)
        line_from = co0.copy()
        rounds = 0
        while True:
            origin = x.copy()
            drift_limit = MOST_DRIFT * _mean_edge(origin, quads)
            last = x.copy()
            for step in range(steps):
                target = self.goal(x, quads, settings, base)
                move = STEP * strength * (target - x)
                n = _vertex_normals(x, quads)
                move -= (move * n).sum(axis=1, keepdims=True) * n  # in the tangent plane: nothing can shrink
                if nb is not None and slide.any():  # a corner on a kept line slides along it only
                    s = np.flatnonzero(slide)
                    mid = 0.5 * (x[nb[s, 0]] + x[nb[s, 1]])
                    t = _unit(x[nb[s, 1]] - x[nb[s, 0]])
                    move[s] = STEP * strength * ((mid - x[s]) * t).sum(axis=1, keepdims=True) * t
                size = np.linalg.norm(move, axis=1)
                move *= np.minimum(1.0, reach / np.maximum(size, 1e-30))[:, None]
                new = x + move
                new[~moving] = x[~moving]
                drift = new - origin
                far = np.linalg.norm(drift, axis=1)
                clamp = far > drift_limit
                new[clamp] = origin[clamp] + drift[clamp] * (drift_limit[clamp] / far[clamp])[:, None]
                if base.project is not None:
                    m = np.flatnonzero(moving)
                    new[m] = _onto(base.project, new[m], reach[m])
                if nb is not None and slide.any():  # off the line it started on by more than a tenth of an edge: back
                    s = np.flatnonzero(slide)
                    off = np.minimum(_to_segment(new[s], line_from[s], line_from[nb[s, 0]]), _to_segment(new[s], line_from[s], line_from[nb[s, 1]]))
                    out = s[off > CREASE_REACH * mean[s]]
                    new[out] = x[out]
                bad = ((_normals(new, quads) * start).sum(axis=1) < WORST_TURN) | (_bent(new, quads) & ~was_bent)
                turned = np.zeros(len(co0), dtype=bool)
                if bad.any():
                    turned[np.unique(quads[bad])] = True
                    turned &= moving
                    new[turned] = x[turned]
                held = (clamp & moving) | turned
                x = new
                yield (step + 1.0) / steps if not keep_going else min(0.99, (step + 1.0) / steps)
            rounds += 1
            if not keep_going or float((np.linalg.norm(x - origin, axis=1) / mean).max()) <= SETTLED or np.array_equal(x, last):
                break
        while True:  # a corner put back by a guard can bend the quad next to it: those corners go home too
            worse = ((_normals(x, quads) * start).sum(axis=1) < 0.0) | (_bent(x, quads) & ~was_bent)
            home = np.zeros(len(co0), dtype=bool)
            if worse.any():
                home[np.unique(quads[worse])] = True
            home &= moving & np.any(x != co0, axis=1)
            if not home.any():
                break
            x = x.copy()
            x[home] = co0[home]
            held |= home
        cand = self._finish(base, x, quads, settings, strength, rough, held)
        after = bad_faces(cand.co, quads, start)
        if after > bad_before:
            return Candidate(None, refusal=BAD_FACES.format(n=after - bad_before))
        if keep_going:
            cand.note = (cand.note + "  " if cand.note else "") + f"{rounds} run{'s' if rounds != 1 else ''} until it settled"
        return cand

    def _finish(self, base, x, quads, settings, strength, rough, held) -> Candidate:
        cand = Candidate(x, strength=strength, exact=not rough)
        cand.held = [(x[i], "drift") for i in np.flatnonzero(held)[:300].tolist()]
        return cand


def _to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    t = np.clip(((p - a) * d).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-30), 0.0, 1.0)
    return np.linalg.norm(a + t[:, None] * d - p, axis=1)


class SourceRelax(_Relax):
    name = "Source Relax"
    defaults = {"strength": 0.6, "steps": 8, "keep_going": False}
    stages = ("tidy", "look", "move")

    def goal(self, x, quads, settings, base):
        return _weighted_middle(x, quads, _areas(x, quads))


class KeepLinesRelax(_Relax):
    name = "Keep Lines Relax"
    defaults = {"strength": 0.6, "steps": 8, "hard": True, "outline": True, "seams": True, "parts": True, "guides": True, "keep_going": False}
    toggles = LINE_KINDS
    layers = ("scope", "ghost", "wash", "arrows", "poles", "held", "kept")
    stages = ("tidy", "look", "lines")

    def _plan(self, base: Base, settings: dict):
        key = ("keep_lines",) + tuple(bool(settings.get(k)) for k, _w in LINE_KINDS)
        if key in base.cache:
            return base.cache[key]
        lines = kept_lines(base, {k: settings.get(k) for k, _w in LINE_KINDS})
        n = len(base.co)
        degree = np.bincount(lines.reshape(-1), minlength=n) if len(lines) else np.zeros(n, dtype=np.int64)
        # held for a reason other than a hard edge: a border, a triangle or n-gon, the scope's edge, the mirror plane
        half = half_edges(base.faces)
        other = pinned(base.co, base.faces, sharp_deg=180.0, plane=base.mirror, half=half) if base.faces else np.zeros(n, dtype=bool)
        fid, corner, _next = half
        used = np.zeros(n, dtype=bool)
        used[corner[base.scope[fid]]] = True
        other[corner[~base.scope[fid]]] = True
        if base.mirror is not None:
            axis, offset = base.mirror
            size = float(np.ptp(base.co, axis=0).max()) if n else 1.0
            other |= np.abs(base.co[:, axis] - offset) < 1e-6 * (size or 1.0)
        slide = (degree == 2) & used & ~other
        nb = np.full((n, 2), -1, dtype=np.int64)
        if len(lines):
            ends = np.concatenate((lines, lines[:, ::-1]))
            order = np.argsort(ends[:, 0], kind="stable")
            ends = ends[order]
            first = np.searchsorted(ends[:, 0], np.arange(n))
            s = np.flatnonzero(slide)
            nb[s, 0] = ends[first[s], 1]
            nb[s, 1] = ends[first[s] + 1, 1]
        free = base.free & (degree == 0)
        base.cache[key] = (lines, nb, slide, free)
        return base.cache[key]

    def _free(self, base, settings):
        return self._plan(base, settings)[3]

    def check(self, base: Base) -> str:
        _lines, _nb, slide, free = self._plan(base, dict(self.defaults))
        return "" if (slide | free).any() else NOTHING_FREE

    def _lines(self, base, settings):
        _lines, nb, slide, _free = self._plan(base, settings)
        return nb, slide

    def goal(self, x, quads, settings, base):
        return _weighted_middle(x, quads, _areas(x, quads))

    def _finish(self, base, x, quads, settings, strength, rough, held) -> Candidate:
        cand = super()._finish(base, x, quads, settings, strength, rough, held)
        lines = self._plan(base, settings)[0]
        cand.lines = lines
        cand.numbers["lines"] = (f"lines kept {len(lines):,}", 0)
        if not len(lines):
            cand.note = NO_LINES
        return cand


class SquareUp(_Relax):
    name = "Square Up"
    defaults = {"strength": 0.7, "steps": 12, "keep_going": False}
    layers = ("scope", "ghost", "wash", "arrows", "held")
    stages = ("square", "look", "tidy")

    def check(self, base: Base) -> str:
        quads = _scope_quads(base)
        if not len(quads) or square_share(base.co, quads) >= 100.0:
            return NO_SLANT
        return super().check(base)

    def goal(self, x, quads, settings, base):
        move, at = square_pull(x, quads)
        pull = np.stack([np.bincount(at, move[:, k], len(x)) for k in range(3)], axis=1)
        count = np.bincount(at, minlength=len(x))
        return x + pull / np.maximum(count, 1)[:, None]

    def _finish(self, base, x, quads, settings, strength, rough, held) -> Candidate:
        cand = super()._finish(base, x, quads, settings, strength, rough, held)
        a, b = square_share(base.co, quads), square_share(x, quads)
        cand.numbers["square"] = (f"square corners {a:.0f}% -> {b:.0f}%", 1 if b > a + 0.5 else -1 if b < a - 0.5 else 0)
        return cand


class RelaxTowardDetail(_Relax):
    name = "Relax Toward Detail"
    handles = ("mix",)
    words = ("Flat", "Detail")
    defaults = {"mix": 0.5, "steps": 12, "keep_going": False}
    blendable = False  # every Mix value relaxes; it changes where sizes go, not how far
    layers = ("scope", "ghost", "wash", "arrows", "held")
    stages = ("look", "size", "tidy")
    legend = "blue grew  orange shrank"

    def prepare(self, base: Base):
        yield from bend_steps(base)

    def check(self, base: Base) -> str:
        quads = _scope_quads(base)
        if not len(quads):
            return NOTHING_FREE
        density = detail_density(base, 1.0)
        used = np.unique(quads)
        low, high = np.percentile(density[used], (10.0, 90.0))
        if high / max(low, 1e-30) < EVEN_CURVES:
            return EVEN_BENDS
        return super().check(base)

    def _strength(self, settings: dict) -> float:
        return 1.0

    def goal(self, x, quads, settings, base):
        density = detail_density(base, float(settings.get("mix", 0.5)))
        return _weighted_middle(x, quads, _areas(x, quads) * density[quads].mean(axis=1) ** 2)

    def _finish(self, base, x, quads, settings, strength, rough, held) -> Candidate:
        cand = super()._finish(base, x, quads, settings, strength, rough, held)
        cand.strength = float(settings.get("mix", 0.5))
        if len(quads):
            grew = np.log(np.maximum(_areas(x, quads), 1e-30) / np.maximum(_areas(base.co, quads), 1e-30))
            total = np.bincount(quads.reshape(-1), np.repeat(grew, 4), len(x))
            count = np.bincount(quads.reshape(-1), minlength=len(x))
            tint = np.where(count > 0, total / np.maximum(count, 1), 0.0)
            top = float(np.abs(tint).max())
            cand.tint = tint / top if top > 1e-9 else tint
            a, b = size_range(base.co, quads), size_range(x, quads)
            cand.numbers["size"] = (f"size range 1 : {a:.1f} -> 1 : {b:.1f}", 0)
        return cand
