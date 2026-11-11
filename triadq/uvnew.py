# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""New UVs for the add-on (package A2): the whole pipeline of SPEC section 6 as sliced stages over the K4-K7 cores.

NumPy only, no bpy. Stages, each a state object with `step()` of about 4 ms that the caller (Look First's timer, a
script loop, a test) keeps calling: 1. what the camera sees (`uvsight.Sight`, only when it matters: seams out of
sight or hidden parts get fewer pixels); 2. patches and the merge tree (`uvchart.Charts`); 3. charts at Stretch allowed
and their flat layouts (quad ARAP, Straight rows, chart by chart); 4. per chart: Hidden-parts scale, box by rotating calipers (Toussaint 1983) or the rows
axis for On charts; 5. pages: MaxRects boxes at the largest scale that fits (`uvpack.FillPage`, Jylanki 2010), whole
texels (Carr, Hoberock, Crane, Hart 2006) and the gap of SPEC 8.2; 6. corner UVs written back in mesh order.

`again()` reruns from the stage that a changed setting touches and keeps
the pictures and the merge tree, so a Stretch allowed or Gap change costs the later stages only.
"""

from __future__ import annotations

from math import log2
import time

import numpy as np

from . import uvchart, uvflat, uvpack, uvscore, uvsight

TEXTURE_PX = 1024
CLEAN_DOWN_TO_PX = 256  # SPEC 8.2: the gap keeps neighbours apart down to this mip size
SLICE_SECONDS = 0.002
LEVEL_STEP = 0.5  # a chart that lies over itself is redone at this share of the Stretch allowed it had
STRETCH_CHUNK = 8000  # triangles measured per slice in the stretch check
CELLS_PER_TRI = 8  # grid cells per triangle (on average) in the overlap test's spatial hash
CHECK_PAIRS = 200  # triangle pairs tested in one go (a slice)
WRITE_CHUNK = 20000  # corners written per slice
QUICK_FACES = 6000  # a mesh with this many faces or more gets a first layout before the repairs (Settings.quick_first)
TRI_CHUNK = 4000  # triangles turned into grid entries per slice
FACE_CHUNK = 6000  # faces whose triangles are built and measured in one slice
STRIPE_ENTRIES = 4000  # grid entries sorted together (a stripe)
_H1, _H2, _H3 = np.int64(73856093), np.int64(19349663), np.int64(83492791)  # hash of (chart, cell) (Teschner et al. 2003)
OVERLAP_EPS = 1e-4  # triangles overlapping by less than this share of a median triangle's side count as touching
REDO_FACES = 64  # charts this small are laid out again on their own when their layout stretches too much
REDO_MEDIAN = 1.25  # ... or when more than half their area stretches more than this (L2 stretch)
ATTACH_FACES = 16  # charts this small are hinged onto a neighbour when that overlaps nothing (a setting; 0 = never)
ATTACH_TOL = 0.05  # border layouts that agree within this share of an edge are glued along the whole shared border
PY_PACK_SECONDS_PER_BOX = 3e-5  # the Python packer's time for one box in one pack (turns Time per pass into a number of packs)
PY_MIN_PACKS = 8  # the Python packer always runs at least this many packs


def mip_level(texture_px: float, clean_px: float = CLEAN_DOWN_TO_PX) -> int:
    """k of the gap rule: how many halvings from the texture size down to `clean_px` (never below 0)."""

    return max(0, int(round(log2(max(float(texture_px), 1.0) / max(float(clean_px), 1.0)))))


def gap_texels(texture_px: float, clean_px: float = CLEAN_DOWN_TO_PX) -> int:
    """SPEC 8.2: 2^(k+1) texels (16 at 2048, 8 at 1024)."""

    return 1 << (mip_level(texture_px, clean_px) + 1)


class Settings:
    """The New UVs popover values (SPEC 10.1), in the units the card shows."""

    def __init__(self, **kw):
        self.stretch_allowed = 120.0  # percent (SPEC 6.4); tuned on 13 models: worst 1 % stretch <= 1.26
        self.rows = "auto"  # auto / on / soft / off
        self.out_of_sight = True
        self.hidden_pct = 50.0  # px per metre of hidden parts, percent of seen ones
        self.views = uvsight.VIEWS_DEFAULT
        self.all_sides = True
        self.texture_px = TEXTURE_PX
        self.clean_px = CLEAN_DOWN_TO_PX
        self.time_per_pass = 2.0  # seconds' worth of search the packer may do before it shows what it has (counted in packs, so the layout is the same every time)
        self.density = None  # px per mesh unit; None = Fill the page
        self.quick_first = True  # big meshes: show a first layout before small charts are laid out again or joined to a neighbour, then the full one
        self.hidden_l = 0.0  # 0 = the cores' default L
        for key, value in kw.items():
            if not hasattr(self, key):
                raise TypeError(f"unknown New UVs setting {key!r}")
            setattr(self, key, value)

    def copy(self, **kw) -> "Settings":
        out = Settings()
        out.__dict__.update(self.__dict__)
        for key, value in kw.items():
            if not hasattr(out, key):
                raise TypeError(f"unknown New UVs setting {key!r}")
            setattr(out, key, value)
        return out


class Result:
    """What a run hands back. `uv` is per mesh corner, on the 0-1 square (one page)."""

    def __init__(self):
        self.uv = None
        self.chart = None  # per face
        self.charts = 0
        self.on_chart = None  # per face: the chart is a Straight rows On chart
        self.col = self.row = None  # per corner, on On charts (else -1)
        self.density = None  # per face: the intended Hidden-parts factor
        self.seen_face = self.seen_edge_px = None
        self.scale = 0.0  # texels per mesh unit
        self.used = 0.0  # share of the page the boxes take
        self.messages: list = []
        self.stats = None
        self.timing: dict = {}
        self.overlap_pct = None
        self.stretch_allowed = None


# ---------------------------------------------------------------------------------------------------------------------
# Small geometry helpers (vectorised over the whole mesh)
# ---------------------------------------------------------------------------------------------------------------------


def face_areas(points, indptr, flat) -> np.ndarray:
    """3D area of every face (Newell)."""

    nf = len(indptr) - 1
    owner = np.repeat(np.arange(nf), np.diff(indptr))
    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    cr = np.cross(points[flat], points[flat[nxt]])
    m = np.zeros((nf, 3))
    for k in range(3):
        m[:, k] = np.bincount(owner, weights=cr[:, k], minlength=nf)
    return np.linalg.norm(m, axis=1) / 2.0


def _drive(gen):
    """Run a generator to its end and return what it returns (the sliced helpers below, outside a slicing caller)."""

    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


def _stable_order(keys) -> np.ndarray:
    """`np.argsort(keys, kind="stable")` for non-negative integer keys in linear time: an LSD radix sort over 16-bit digits (numpy sorts
    16-bit integers by radix; a stable sort of 64-bit ones takes several milliseconds on a large mesh, which is a slice)."""

    keys = np.asarray(keys)
    if len(keys) < 4096 or int(keys.min()) < 0:
        return np.argsort(keys, kind="stable")
    top = int(keys.max())
    order = np.argsort((keys & 0xFFFF).astype(np.uint16), kind="stable")
    shift = 16
    while top >> shift:
        order = order[np.argsort(((keys[order] >> shift) & 0xFFFF).astype(np.uint16), kind="stable")]
        shift += 16
    return order


def _next_corner(indptr, flat) -> np.ndarray:
    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    return nxt


def _first_occurrence(keys):
    """Generator, returns for every key the index of its first occurrence. Keys are sorted stripe by stripe
    (`STRIPE_ENTRIES` at a time), so no slice sorts a whole mesh."""

    n = len(keys)
    stripes = max(1, -(-n // STRIPE_ENTRIES))
    which = (keys % np.uint64(stripes)).astype(np.int16 if stripes < 32000 else np.int64)
    order = np.argsort(which, kind="stable")  # small integers: a radix sort, and each stripe keeps the element order
    bounds = np.searchsorted(which[order], np.arange(stripes + 1))
    first_of = np.empty(n, np.int64)
    yield
    for p in range(stripes):
        idx = order[bounds[p] : bounds[p + 1]]
        if len(idx):
            _, first, inverse = np.unique(keys[idx], return_index=True, return_inverse=True)
            first_of[idx] = idx[first][inverse.reshape(-1)]
        if p % 4 == 3:
            yield
    return first_of


def _edge_table(indptr, flat):
    """Generator, returns (edge id of every corner's edge to the next corner, edge count, first corner of that edge per
    corner), edges numbered by first use."""

    nxt = _next_corner(indptr, flat)
    lo, hi = np.minimum(flat, flat[nxt]), np.maximum(flat, flat[nxt])
    keys = lo.astype(np.uint64) << np.uint64(32) | hi.astype(np.uint64)
    yield
    first_of = yield from _first_occurrence(keys)
    n = len(keys)
    rank = np.cumsum(first_of == np.arange(n)) - 1
    yield
    return rank[first_of], (int(rank[-1]) + 1 if n else 0), first_of


def edge_ids(indptr, flat) -> tuple:
    """(edge id of every corner's edge to the next corner, edge count), edges numbered by first use (the order of `seen_edge_px`)."""

    ids, ne, _ = _drive(_edge_table(indptr, flat))
    return ids, ne


def _edge_pairs(ids, ne, first_of):
    """Generator, returns (edges used by exactly two corners, the first and the second corner of each), in edge order."""

    n = len(ids)
    count = np.bincount(ids, minlength=ne)
    yield
    firsts = np.flatnonzero(first_of == np.arange(n))  # corner of each edge's first use
    second = np.full(ne, -1, np.int64)
    later = np.flatnonzero(first_of != np.arange(n))
    second[ids[later]] = later  # for an edge used twice there is exactly one later corner
    yield
    two = np.flatnonzero(count == 2)
    return two, firsts[two], second[two]


def _cut_edges(indptr, flat, uv, tol: float = 1e-6, table=None):
    """Generator behind `cut_edges`; `table` is a finished `_edge_table` result to reuse."""

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    ids, ne, first_of = table if table is not None else (yield from _edge_table(indptr, flat))
    nxt = _next_corner(indptr, flat)
    cut = np.zeros(ne, bool)
    two, a, b = yield from _edge_pairs(ids, ne, first_of)
    if len(two):
        same = flat[a] == flat[b]  # faces that disagree on the winding run the edge the same way
        ua0, ua1 = uv[a], uv[nxt[a]]
        yield
        ub, ub_next = uv[b], uv[nxt[b]]
        yield
        ub0 = np.where(same[:, None], ub, ub_next)
        ub1 = np.where(same[:, None], ub_next, ub)
        yield
        d0 = np.abs(ua0 - ub0)
        yield
        d1 = np.abs(ua1 - ub1)
        cut[two] = (np.maximum(d0[:, 0], d0[:, 1]) > tol) | (np.maximum(d1[:, 0], d1[:, 1]) > tol)
    return cut


def cut_edges(indptr, flat, uv, tol: float = 1e-6) -> np.ndarray:
    """Per edge (first-use order): True where the two faces meeting there do not share the edge's UVs (a UV seam)."""

    return _drive(_cut_edges(indptr, flat, uv, tol))


OCTAGON_FROM = 256  # R9first: charts with more corners than this drop the points inside their extreme octagon first (cheaper than the hull loop)
_OCT2 = np.array(((1, 0), (0, 1), (1, 1), (1, -1)), float).T  # the octagon's four directions, one per column


def _octagon(p: np.ndarray) -> np.ndarray:
    """Akl-Toussaint: drop the points inside the octagon of the extreme points (they are not on the hull)."""

    proj = p @ _OCT2
    ext = np.unique(np.concatenate((np.argmin(proj, axis=0), np.argmax(proj, axis=0))))
    o = p[ext]
    o = o[np.argsort(np.arctan2(o[:, 1] - o[:, 1].mean(), o[:, 0] - o[:, 0].mean()))]
    edge = np.roll(o, -1, axis=0) - o
    inside = ((edge[:, 0:1] * (p[:, 1][None] - o[:, 1:2]) - edge[:, 1:2] * (p[:, 0][None] - o[:, 0:1])) > 0).all(axis=0)
    if len(o) >= 3 and inside.any():
        return np.concatenate([p[~inside], o])
    return p


def _hull(p: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points (Andrew's monotone chain), counter-clockwise, no repeated end point."""

    if len(p) > 512:
        p = _octagon(p)
    pts = np.unique(np.round(p, 12), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))].tolist()
    return np.array(_chain(pts))


def _chain(pts: list) -> list:
    """Andrew's monotone chain over x-then-y sorted, distinct points (lists): the hull counter-clockwise from the first."""

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (q[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return lower[:-1] + upper[:-1]


def _calipers(h: np.ndarray) -> float:
    """The angle (radians) that turns a hull (counter-clockwise, at least 3 points) into its smallest axis-aligned box."""

    d = np.concatenate((h[1:], h[:1])) - h
    ang = np.arctan2(d[:, 1], d[:, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    x = h[None, :, 0] * c[:, None] - h[None, :, 1] * s[:, None]
    y = h[None, :, 0] * s[:, None] + h[None, :, 1] * c[:, None]
    area = (x.max(axis=1) - x.min(axis=1)) * (y.max(axis=1) - y.min(axis=1))
    return float(-ang[int(np.argmin(area))])


def min_area_angles(uv: np.ndarray, order: np.ndarray, bounds: np.ndarray, which, budget: float = 0.004):
    """Generator, returns the smallest-box angle (`_calipers` on the hull) of each group in `which` (group c is `uv[order[bounds[c]:bounds[c + 1]]]`;
    0.0 for fewer than 3 distinct points). R9first: the same hulls and angles without the per-chart `np.unique` and
    array round trips (most charts are small, so those fixed costs were most of the orient stage). Yields every
    `budget` seconds."""

    out = np.zeros(len(which))
    until = time.perf_counter() + budget
    for i, c in enumerate(which):
        if time.perf_counter() > until:
            yield
            until = time.perf_counter() + budget
        a, b = bounds[c], bounds[c + 1]
        if b - a <= 2:
            continue
        q = uv[order[a:b]]
        if b - a > OCTAGON_FROM:  # the octagon filter of `_hull` (Akl-Toussaint): the same hull from fewer points
            q = _octagon(q)
        q = np.round(q, 12)
        q = q[np.lexsort((q[:, 1], q[:, 0]))]
        keep = np.ones(len(q), bool)
        keep[1:] = (q[1:, 0] != q[:-1, 0]) | (q[1:, 1] != q[:-1, 1])
        q = q[keep]
        if len(q) <= 2:
            continue
        h = _chain(q.tolist())
        if len(h) >= 3:
            out[i] = _calipers(np.array(h))
    return out


def rows_angle(uv: np.ndarray, col: np.ndarray, row: np.ndarray, indptr, flat) -> float | None:
    """Angle that lays the u rows of a labelled grid chart along the u axis (mean over its u edges, modulo 90 degrees)."""

    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    a, b = flat, flat[nxt]
    along = (row[a] == row[b]) & (col[a] != col[b]) & (row[a] >= 0)
    if not along.any():
        return None
    d = uv[b][along] - uv[a][along]
    t = np.arctan2(d[:, 1], d[:, 0])
    mean = np.arctan2(np.sin(4 * t).sum(), np.cos(4 * t).sum()) / 4.0  # a row has no sign, and v rows differ by 90 degrees
    return float(-mean)


# ---------------------------------------------------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------------------------------------------------


class NewUVs:
    """Make New UVs for a quad (or any polygon) mesh in slices.

    points (n, 3); indptr / flat: faces in CSR form; material per face or None (a material border is a forced cut);
    forced: vertex pairs that must be cut (hard edges for a normal bake). `step()` returns progress 0..1, `done`,
    `result` (a `Result`). `again(**settings)` starts a run that keeps this run's pictures and tree.

    A preview does not wait for `done`: `first_ready` turns True as soon as `result` holds a complete layout (no overlaps, no
    flips), `revision` counts the layouts put in `result` (on a big mesh the first one comes before small charts are laid out
    again or joined to a neighbour; a later one has its borders slid out of sight), so a preview redraws when
    `revision` changes. `run()` makes only the last one.
    """

    def __init__(self, points, indptr, flat, material=None, forced=None, settings: Settings | None = None, budget: float = SLICE_SECONDS,
                 _keep: dict | None = None):
        self.points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.flat = np.asarray(flat, dtype=np.int64)
        self.material = None if material is None else np.asarray(material, dtype=np.int64)
        self.forced = None if forced is None or len(forced) == 0 else np.asarray(forced, dtype=np.int64).reshape(-1, 2)
        self.settings = settings or Settings()
        self.budget = float(budget)
        self.done, self.result, self.error = False, None, None
        self.progress, self.stage_name = 0.0, "start"
        self.sight = None
        self.tree = None
        self.messages: list = []
        self._stop = False
        self._whole = False  # `run()` was asked for before anything ran: no first layout
        self.revision = 0  # counts the layouts put in `result`: the first one, the full one, the one out of sight; a preview redraws when it changes
        self.first_ready = False  # True once `result` holds a complete layout
        self._active = None  # the sliced core stage being polled
        self._hidden = True  # R8late: the levels used are the ones whose borders slid out of sight (False for the first result)
        self._keep = _keep or {}
        self._work = self._run()

    # -- driving ------------------------------------------------------------------------------------------------------
    def stop(self) -> None:
        """End early: the stage that runs stops with what it has; the later stages still run so the result is valid."""

        self._stop = True
        if self._active is not None and hasattr(self._active, "stop"):
            self._active.stop()

    def step(self) -> float:
        if self.done:
            return 1.0
        until = time.perf_counter() + self.budget
        try:
            while time.perf_counter() < until:
                next(self._work)
        except StopIteration:
            self.done = True
            self.progress = 1.0
        except Exception as exc:  # the caller falls back (Smart Project) and shows the message
            self.error, self.done = exc, True
        return self.progress

    def run(self) -> Result:
        """Run to the end (scripts and tests). No first layout is made for a preview: it would only repeat work."""

        if self.stage_name == "start":
            self._whole = True
        while not self.done:
            self.step()
        if self.error is not None:
            raise self.error
        return self.result

    def again(self, **changes) -> "NewUVs":
        """A new run with changed settings that reuses the pictures (always) and the merge tree (unless a setting it
        was built with changed)."""

        new = self.settings.copy(**changes)
        same = lambda *names: all(getattr(new, k) == getattr(self.settings, k) for k in names)  # noqa: E731
        keep = {"sight": self.sight if self.sight is not None and self.sight.done and same("views", "all_sides") else None}
        if same("out_of_sight", "hidden_l", "views", "all_sides") and self.tree is not None and new.stretch_allowed / 100.0 <= getattr(self.tree, "build_stretch", uvchart.BUILD_STRETCH):
            keep["tree"] = self.tree
        return NewUVs(self.points, self.indptr, self.flat, self.material, self.forced, new, self.budget, _keep=keep)

    def _poll(self, stage):
        """Step a core stage until it reports finished."""

        self._active = stage
        while True:
            stage.step()
            if getattr(stage, "done", False) is True:
                break
            yield
        self._active = None

    def _tick(self, name: str, progress: float):
        self.stage_name, self.progress = name, progress

    # -- stages -------------------------------------------------------------------------------------------------------
    def _run(self):
        s, pts, indptr, flat = self.settings, self.points, self.indptr, self.flat
        nf = len(indptr) - 1
        if nf == 0 or len(pts) == 0:
            raise ValueError("The mesh has no faces to unwrap.")
        res = Result()
        t0 = time.perf_counter()
        area = np.empty(nf)
        for i in range(0, nf, 4000):
            j = min(nf, i + 4000)
            area[i:j] = face_areas(pts, indptr[i : j + 1] - indptr[i], flat[indptr[i] : indptr[j]])
            yield
        if not np.isfinite(area).all() or area.sum() <= 0:
            raise ValueError("The mesh has no surface to unwrap.")
        yield

        # 1. what the camera sees
        needs_sight = s.out_of_sight or s.hidden_pct < 100.0
        seen_face, inside = np.ones(nf), None
        if needs_sight:
            self._tick("seen", 0.02)
            sight = self._keep.get("sight")
            if sight is None:
                sight = uvsight.Sight(pts, (indptr, flat), views=int(s.views), all_sides=s.all_sides)
            self.sight = sight
            if not sight.done:
                yield from self._poll_sight(sight)
            seen_face, inside = np.asarray(sight.seen_face, dtype=np.float64), np.asarray(sight.inside_px, dtype=np.float64)
        res.seen_face, res.seen_edge_px = seen_face, inside
        res.timing["seen_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()

        # 2. patches and the merge tree
        tree = self._keep.get("tree")
        if tree is None:
            self._tick("charts", 0.2)
            stage = uvchart.Charts(pts, indptr, flat, part_label=self.material, forced=self.forced, seen_edge_px=inside if s.out_of_sight else None,
                                   out_of_sight=bool(s.out_of_sight and inside is not None), hidden_l=s.hidden_l,
                                   build_stretch=max(uvchart.BUILD_STRETCH, float(s.stretch_allowed) / 100.0), budget=self.budget)
            self._active = stage
            while not stage.done:
                stage.step()
                yield
            self._active = None
            tree = stage.tree
        self.tree = tree
        res.timing["charts_s"] = time.perf_counter() - t1

        # 3-6 on the charts as the merges left them, then (R8late) the same on the charts whose borders slid out of sight
        # (R7hide's moves, made as a later step in slices while the first result is shown), swapped in when it is ready.
        # A tree whose moves are made (`again`), and `run()`,
        # go to the hidden charts at once.
        self._hidden = self._whole or self._hide_ready(tree)
        if self._whole and hasattr(tree, "topo") and getattr(tree, "_hide", None) is None:
            yield from self._make_hide(tree)  # in slices: the first hidden level would make the same moves in one call
        yield from self._from_charts(tree, area, seen_face, inside, res.timing, t0, show=True)
        if not self._hidden and not self._stop:
            shown = self.result
            th = time.perf_counter()
            self._tick("hide", 0.95)
            moved = yield from self._make_hide(tree)
            hide_s = time.perf_counter() - th
            if moved and not self._stop:
                self._hidden = True
                head = {k: shown.timing[k] for k in ("seen_s", "charts_s") if k in shown.timing}
                late = yield from self._from_charts(tree, area, seen_face, inside, head, t0, show=False)
                if not self._stop:
                    late.timing["first_s"] = shown.timing.get("first_s")
                    late.timing["hide_s"] = hide_s
                    late.timing["pack_s"] = late.timing["total_s"] - sum(late.timing.get(k, 0.0) for k in ("flatten_s", "charts_s", "seen_s", "hide_s"))
                    self.result = late
                    self.revision += 1
        self._tick("done", 1.0)
        yield

    def _hide_ready(self, tree) -> bool:
        """The tree's borders have slid out of sight already (its levels with `hide` cost only the replay)."""

        return getattr(tree, "_hide", None) is not None

    def _make_hide(self, tree):
        """Generator: make the tree's moves in slices; returns the faces moved (0 when stopped)."""

        from . import uvhide

        gen = uvhide.moves(tree)
        while True:
            if self._stop:
                return 0
            try:
                next(gen)
            except StopIteration as done:
                tree._hide = done.value
                return int(done.value.moved)
            yield
    def _from_charts(self, tree, area, seen_face, inside, head, t0, show):
        """Generator: the layout of the tree's charts (hidden or not, `self._hidden`), from the flatten to the page;
        returns its `Result`. `show`: each layout goes into `self.result` as it is made (the first result);
        else the caller swaps the finished one in."""

        s = self.settings
        nf = len(self.indptr) - 1
        res = Result()
        res.timing = dict(head)
        t2 = time.perf_counter()
        x_now = float(s.stretch_allowed)
        # 3. charts at Stretch allowed, flattened
        self._tick("flatten", 0.5)
        corner_uv0, chart0, rows0, _ = yield from self._flatten(tree, x_now)
        if self._stop:
            res.messages.append("Stopped early: the layout is what had been reached.")
        res.timing["flatten_s"] = time.perf_counter() - t2
        head, head_messages = dict(res.timing), list(res.messages)

        # 3b-6. The layout in one pass, or in two on a big mesh: a first one (no small chart is yet laid out again or joined to a
        # neighbour: that takes a second or more), shown at once, then the full one.
        passes = ("first", "full") if show and s.quick_first and not self._whole and nf >= QUICK_FACES and not self._stop else ("full",)
        first_total = None
        for kind in passes:
            if kind == "full" and len(passes) == 2 and self._stop:
                break  # Stop: the first layout stays
            res = Result()
            res.timing, res.messages = dict(head), list(head_messages)
            res.seen_face, res.seen_edge_px = seen_face, inside
            t3 = time.perf_counter()
            if kind == "first" or len(passes) == 1:
                chart1, corner_uv1, rows1, back = yield from self._split_overlapping(tree, chart0, corner_uv0, rows0, x_now)
            chart, corner_uv, chart_rows = chart1, corner_uv1, rows1
            res.timing["split_s"] = time.perf_counter() - t3
            t3a = time.perf_counter()
            self._tick("repair", 0.62)
            if kind == "full":
                if back:
                    res.messages.append(f"{back} chart{'s' if back != 1 else ''} lay over themselves and were redone at a lower Stretch allowed.")
                corner_uv, chart_rows, fixed = yield from self._redo_stretched(chart, corner_uv, chart_rows, x_now)
                if fixed:
                    res.messages.append(f"{fixed} small chart{'s' if fixed != 1 else ''} laid out again with less stretch.")
                res.timing["redo_s"] = time.perf_counter() - t3a
                t3a = time.perf_counter()
                self._tick("attach", 0.65)
                chart, corner_uv, chart_rows = yield from self._attach_small(chart, corner_uv, chart_rows)
                res.timing["attach_s"] = time.perf_counter() - t3a
            else:
                corner_uv, chart_rows, _ = yield from self._redo_stretched(chart, corner_uv, chart_rows, x_now, only_lying=True)
            yield from self._layout(res, chart, corner_uv, chart_rows, area, seen_face, x_now)
            res.timing["total_s"] = time.perf_counter() - t0
            first_total = res.timing["total_s"] if first_total is None else first_total
            res.timing["first_s"] = first_total
            res.messages = list(self.messages) + res.messages
            if show:
                self.result = res
                self.revision += 1
                self.first_ready = True
            self._tick("first" if kind == "first" else "pack", 0.9)
            yield

        return res

    def _layout(self, res, chart, corner_uv, chart_rows, area, seen_face, x_now):
        """Generator: the corner UVs of charts laid out in the plane, on the page as boxes (stages 4 to 6 of the pipeline).
        Fills `res`."""

        s, indptr, flat = self.settings, self.indptr, self.flat
        nf = len(indptr) - 1
        t3a = time.perf_counter()
        nchart = int(chart.max()) + 1 if len(chart) else 0
        res.chart, res.charts = chart, nchart
        res.stats = chart_rows
        corner_face = np.repeat(np.arange(nf), np.diff(indptr))
        yield
        corner_chart = chart[corner_face]
        yield

        # 4. per chart: grid labels for On charts, the rows axis or the calipers box, the Hidden-parts scale
        self._tick("orient", 0.7)
        on = chart_rows == 2
        res.on_chart = on[chart]
        col = np.full(len(flat), -1, np.int64)
        row = np.full(len(flat), -1, np.int64)
        order = _stable_order(corner_chart)
        yield
        bounds = np.searchsorted(corner_chart[order], np.arange(nchart + 1))
        face_order = _stable_order(chart)
        yield
        fbounds = np.searchsorted(chart[face_order], np.arange(nchart + 1))
        angle = np.zeros(nchart)
        boxed = []
        until = time.perf_counter() + self.budget
        for c in range(nchart):
            if time.perf_counter() > until:
                yield
                until = time.perf_counter() + self.budget
            if on[c]:
                got = self._labels(c, face_order[fbounds[c] : fbounds[c + 1]], corner_uv)
                if got is not None:
                    col[got[0]], row[got[0]] = got[1], got[2]
                    if got[4] is not None:
                        angle[c] = got[4]
                        continue
                res.on_chart[face_order[fbounds[c] : fbounds[c + 1]]] = False  # no clean labels: not counted as a straight-rows chart
            boxed.append(c)
        if boxed:  # the other charts: their smallest box by rotating calipers, all at once
            angle[boxed] = yield from min_area_angles(corner_uv, order, bounds, boxed, self.budget)
        res.col, res.row = col, row
        yield
        ca, sa = np.cos(angle)[corner_chart], np.sin(angle)[corner_chart]
        ru = corner_uv[:, 0] * ca - corner_uv[:, 1] * sa
        yield
        rv = corner_uv[:, 0] * sa + corner_uv[:, 1] * ca
        yield
        # Isotropic fit to 3D area per chart (the solver works in surface units already; this only removes rounding),
        # then the Hidden-parts factor: scale ~ sum(area * max(floor, seen)) / sum(area), no square root (SPEC 6.7).
        floor_ = min(max(s.hidden_pct, 0.0), 100.0) / 100.0
        a3 = np.bincount(chart, weights=area, minlength=nchart)
        a2 = _uv_area(corner_uv, indptr, flat, chart, nchart)
        yield
        fit = np.where(a2 > 0, np.sqrt(np.divide(a3, a2, out=np.ones(nchart), where=a2 > 0)), 1.0)
        weighted = np.bincount(chart, weights=area * np.maximum(floor_, seen_face), minlength=nchart)
        factor = np.where(a3 > 0, weighted / np.where(a3 > 0, a3, 1.0), 1.0)
        mult = (fit * factor)[corner_chart]
        ru, rv = ru * mult, rv * mult
        yield
        starts = bounds[:-1]  # every chart has corners: the sorted corners of a chart are one run
        lo_u, hi_u = np.minimum.reduceat(ru[order], starts), np.maximum.reduceat(ru[order], starts)
        yield
        lo_v, hi_v = np.minimum.reduceat(rv[order], starts), np.maximum.reduceat(rv[order], starts)
        width, height = np.maximum(hi_u - lo_u, 1e-12), np.maximum(hi_v - lo_v, 1e-12)
        res.density = factor[chart]
        yield

        res.timing["orient_s"] = time.perf_counter() - t3a
        t3a = time.perf_counter()
        # 5. page
        self._tick("pack", 0.8)
        k = mip_level(s.texture_px, s.clean_px)
        page = (int(s.texture_px), int(s.texture_px))
        weight = np.maximum(np.bincount(chart, weights=area * seen_face, minlength=nchart), 1e-12)
        fill = None
        if s.density:
            places = uvpack.pack(width.tolist(), height.tolist(), [True] * nchart, uvpack.make_order(width.tolist(), height.tolist(), weight.tolist(), 0), 0,
                                 float(s.density), page, k)
            if places is not None:
                fill = dict(scale=float(s.density), places=places, used=uvpack.used(width.tolist(), height.tolist(), float(s.density), page))
            else:
                res.messages.append("Pixel density does not fit the page: the largest density that fits was used.")
        if fill is None:
            for gap_k in (k, -1):
                # stops after a number of packs worked out from Time per pass: the same layout every time
                packs = max(PY_MIN_PACKS, int(float(s.time_per_pass) / (max(nchart, 1) * PY_PACK_SECONDS_PER_BOX)))
                stage = uvpack.FillPage(width.tolist(), height.tolist(), [True] * nchart, weight.tolist(), page, gap_k, max_packs=packs)
                yield from self._poll(stage)
                fill = stage.result
                if fill is not None:
                    break
                res.messages.append("Too many small islands for the gap at this texture size: the gap was dropped.")
            if fill is not None and fill.get("budget_hit"):
                res.messages.append(f"Stopped after {s.time_per_pass:g} s with the layout still improving. Keep going continues, or raise Time per pass.")
        if fill is None:
            raise ValueError("The charts do not fit on the page, even without a gap.")
        res.scale, res.used = float(fill["scale"]), float(fill["used"])
        res.stretch_allowed = x_now
        res.timing["fill_s"] = time.perf_counter() - t3a
        lu, lv = ru - lo_u[corner_chart], rv - lo_v[corner_chart]
        yield

        # 6. corner UVs on the 0-1 square (the box layout)
        t4 = time.perf_counter()
        px, py, turned = (np.array([p[i] for p in fill["places"]]) for i in range(3))
        turn = turned.astype(bool)[corner_chart]
        wsc = (width * res.scale)[corner_chart]
        out = np.empty_like(corner_uv)
        for i in range(0, len(out), WRITE_CHUNK):
            sl = slice(i, i + WRITE_CHUNK)
            du, dv = lu[sl] * res.scale, lv[sl] * res.scale
            uu = np.where(turn[sl], dv, du)  # a quarter turn (u, v) -> (v, w - u) keeps the winding
            vv = np.where(turn[sl], wsc[sl] - du, dv)
            out[sl, 0] = (px[corner_chart[sl]] + uu) / page[0]
            out[sl, 1] = (py[corner_chart[sl]] + vv) / page[1]
            yield
        res.uv = out
        res.timing["write_s"] = time.perf_counter() - t4

    def _redo_stretched(self, chart, corner_uv, rows, x, only_lying=False):
        """A small chart (at most `REDO_FACES` faces) whose layout stretches more than Stretch allowed somewhere, or more
        than `REDO_MEDIAN` on half its area, is laid out again on its own (`uvflat.Flatten`, free start, rows off: Tutte
        then quad ARAP with the injective step) and kept when its worst stretch drops and nothing overlaps. Such charts
        come from patches whose grid or band layout did not fit them (thin planks, slivers). Returns (corner UV, rows per
        chart, number laid out again)."""

        if only_lying and not self._unresolved:
            return corner_uv, rows, 0
        indptr, flat = self.indptr, self.flat
        nf = len(indptr) - 1
        size = np.bincount(chart, minlength=len(rows))
        tri = _fan(indptr, np.arange(nf))
        yield
        tface = np.repeat(np.arange(nf), np.maximum(np.diff(indptr) - 2, 0))
        yield
        q3, a3, a2 = np.empty((len(tri), 3, 3)), np.empty(len(tri)), np.empty(len(tri))
        for i in range(0, len(tri), STRETCH_CHUNK):  # the triangles' 3D and UV areas a chunk per slice
            sl = slice(i, i + STRETCH_CHUNK)
            q = q3[sl] = self.points[flat[tri[sl]]]
            a3[sl] = 0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
            a2[sl] = 0.5 * np.abs(_tri_area2(corner_uv[tri[sl]]))
            yield
        tch0 = chart[tface]
        yield
        ratio = np.bincount(tch0, weights=a2, minlength=len(rows)) / np.maximum(np.bincount(tch0, weights=a3, minlength=len(rows)), 1e-30)
        L = np.empty(len(tri))
        for i in range(0, len(tri), STRETCH_CHUNK):
            yield
            sl = slice(i, i + STRETCH_CHUNK)
            L[sl] = _l2_stretch(q3[sl], corner_uv[tri[sl]], chart[tface[sl]], len(rows), ratio)
        yield
        tch = chart[tface]
        worst = np.zeros(len(rows))
        Lf = np.where(np.isfinite(L), L, np.inf)
        for i in range(0, len(tch), STRETCH_CHUNK):
            yield
            np.maximum.at(worst, tch[i : i + STRETCH_CHUNK], Lf[i : i + STRETCH_CHUNK])
        yield
        over = np.bincount(tch, weights=a3 * (L > REDO_MEDIAN), minlength=len(rows)) > 0.5 * np.bincount(tch, weights=a3, minlength=len(rows))
        lying = np.zeros(len(rows), bool)
        lying[list(getattr(self, "_unresolved", ()))] = True  # still lying over itself after the guard: any clean layout wins
        worst[lying] = np.inf
        todo = np.flatnonzero((size > 0) & (lying if only_lying else (((size <= REDO_FACES) & ((worst > 1.0 + x / 100.0) | over)) | lying)))
        if not len(todo):
            return corner_uv, rows, 0
        corner_uv, rows = corner_uv.copy(), np.asarray(rows).copy()
        order = _stable_order(chart)
        bounds = np.searchsorted(chart[order], np.arange(len(rows) + 1))
        fixed = 0
        for c in todo:
            faces = order[bounds[c] : bounds[c + 1]]
            corners = self._sub(faces)[0]
            key = np.column_stack((flat[corners], np.round(corner_uv[corners] * 1e9).astype(np.int64)))
            _, local = np.unique(key, axis=0, return_inverse=True)
            local = local.reshape(-1)
            lpts = np.zeros((int(local.max()) + 1, 3))
            lpts[local] = self.points[flat[corners]]
            lptr = np.concatenate([[0], np.cumsum(np.diff(indptr)[faces])])
            lt = _fan(lptr, np.arange(len(faces)))  # local corner order is the order of `corners`
            best, best_w = None, worst[c]
            # 1. the chart's own best-fit plane (a small chart is nearly flat); 2. a free flatten (Tutte, then ARAP)
            centred = lpts - lpts.mean(axis=0)
            axes = np.linalg.svd(centred, full_matrices=False)[2]
            tries = [(centred @ axes[0], centred @ axes[1])]
            stage = uvflat.Flatten(lpts, lptr, local, None, rows="off", start_mode="free", stretch_allowed=x / 100.0, texture_px=float(self.settings.texture_px),
                                   budget=self.budget)
            yield from self._poll(stage)
            if stage.uv is not None and np.isfinite(stage.uv).all():
                tries.append(tuple(np.asarray(stage.uv, dtype=np.float64).T))
            for u, v in tries:
                new = np.column_stack((u, v))[local]
                ar = _tri_area2(new[lt])
                if ar.sum() < 0:
                    new[:, 1] *= -1.0  # the projection may look from behind
                    ar = -ar
                if not np.all(ar > 0):
                    continue
                got = _l2_stretch(lpts[local[lt]], new[lt], np.zeros(len(lt), np.int64), 1)
                if not (np.isfinite(got).all() and got.max() < best_w):
                    continue
                hit = False
                for found in overlap_pairs(new[lt], OVERLAP_EPS * float(np.median(ar)) ** 0.5):
                    if found:
                        hit = True
                        break
                    yield  # R9first: a slice per chunk of pairs (`any()` ran them all in one slice: 8-24 ms)
                if hit:
                    continue
                best, best_w = new, got.max()
            if best is None:
                continue
            corner_uv[corners] = best
            rows[c] = 0
            fixed += 1
        return corner_uv, rows, fixed

    def _edges(self):
        """Generator, returns `_edge_table` of this mesh (ids, edge count, first corner of each corner's edge), worked out once."""

        if getattr(self, "_edge_cache", None) is None:
            self._edge_cache = yield from _edge_table(self.indptr, self.flat)
        return self._edge_cache

    def _attach_small(self, chart, corner_uv, rows):
        """Small charts hinged onto a neighbour (ours): a chart of at most `ATTACH_FACES` faces is turned and moved onto
        the neighbour it shares the longest border with (not a Straight rows On chart), its shared corners snapped to the
        neighbour's, so the two become one island. Every shared edge is glued when the two layouts agree along the border
        (within `ATTACH_TOL` of an edge), else only the longest shared edge. Kept only when no triangle of the small chart
        then overlaps the neighbour and none flips (exact triangle tests, `_tris_overlap`). Each one saves a chart's gap
        ring and box on the page. Returns (chart per face, corner UV, rows per chart), charts renumbered."""

        indptr, flat, pts = self.indptr, self.flat, self.points
        nf = len(indptr) - 1
        size = np.bincount(chart, minlength=len(rows))
        small = np.flatnonzero((size > 0) & (size <= ATTACH_FACES))
        if len(small) == 0:
            return chart, corner_uv, rows
        chart, corner_uv = chart.copy(), corner_uv.copy()
        owner = np.repeat(np.arange(nf), np.diff(indptr))
        nxt = np.arange(1, len(flat) + 1)
        nxt[indptr[1:] - 1] = indptr[:-1]
        yield
        ids, ne, first_of = yield from self._edges()
        two, A, B = yield from _edge_pairs(ids, ne, first_of)
        yield
        same = flat[A] == flat[B]
        B0 = np.where(same, B, nxt[B])  # the other face's corner at A's start vertex
        B1 = np.where(same, nxt[B], B)  # ... at A's end vertex
        yield
        length = np.empty(len(A))
        for i in range(0, len(A), WRITE_CHUNK):  # edge lengths a chunk per slice
            sl = slice(i, i + WRITE_CHUNK)
            length[sl] = np.linalg.norm(pts[flat[nxt[A[sl]]]] - pts[flat[A[sl]]], axis=1)
            yield
        # both directions: small-side corners (a0, a1), other-side corners (b0, b1)
        sa0, sa1 = np.concatenate([A, B0]), np.concatenate([nxt[A], B1])
        sb0, sb1 = np.concatenate([B0, A]), np.concatenate([B1, nxt[A]])
        yield
        slen = np.concatenate([length, length])
        sface, oface = owner[sa0], owner[sb0]
        yield
        half = len(A)  # the shared edges of a face: first those where it is the first corner's face, then the others (the order of one stable sort of `sface`)
        by1 = _stable_order(sface[:half])
        yield
        by2 = _stable_order(sface[half:])
        fstart1 = np.searchsorted(sface[:half][by1], np.arange(nf + 1))
        yield
        fstart2 = np.searchsorted(sface[half:][by2], np.arange(nf + 1))
        face_order = _stable_order(chart)
        yield
        cbounds = np.searchsorted(chart[face_order], np.arange(len(rows) + 1))
        signed = np.empty(nf)
        for i in range(0, nf, FACE_CHUNK):  # a few thousand faces at a time
            signed[i : i + FACE_CHUNK] = _signed_areas(corner_uv, indptr, np.arange(i, min(nf, i + FACE_CHUNK)))
            yield
        csign = np.sign(np.bincount(chart, weights=signed, minlength=len(rows)))
        yield
        added = {}  # host -> face arrays joined to it so far
        moved = 0
        until = time.perf_counter() + self.budget
        for c in small:
            if time.perf_counter() > until:
                yield
                until = time.perf_counter() + self.budget
            faces = face_order[cbounds[c] : cbounds[c + 1]]
            if not len(faces) or np.any(chart[faces] != c):
                continue
            k = np.concatenate([by1[fstart1[f] : fstart1[f + 1]] for f in faces] + [half + by2[fstart2[f] : fstart2[f + 1]] for f in faces])
            host = chart[oface[k]]
            ok = (host != c) & (rows[host] != 2) & (size[host] > ATTACH_FACES)
            k, host = k[ok], host[ok]
            if not len(k):
                continue
            h = int(np.argmax(np.bincount(host, weights=slen[k])))
            k = k[host == h]
            corners = self._sub(faces)[0]
            mirror = np.array([-1.0, 1.0]) if csign[c] != csign[h] else np.ones(2)  # wound the other way: mirrored to match
            own = corner_uv[corners] * mirror
            p = np.concatenate([corner_uv[sa0[k]], corner_uv[sa1[k]]]) * mirror
            q = np.concatenate([corner_uv[sb0[k]], corner_uv[sb1[k]]])
            r_, t_ = _fit_rigid(p, q)
            mean_len = float(slen[k].mean()) or 1.0
            if np.abs(p @ r_.T + t_ - q).max() > ATTACH_TOL * mean_len:
                k = k[[int(np.argmax(slen[k]))]]
                p = np.array([corner_uv[sa0[k[0]]], corner_uv[sa1[k[0]]]]) * mirror
                q = np.array([corner_uv[sb0[k[0]]], corner_uv[sb1[k[0]]]])
                r_, t_ = _fit_rigid(p, q)
            new = own @ r_.T + t_
            snap = {}
            for a0, a1, b0, b1 in zip(sa0[k], sa1[k], sb0[k], sb1[k]):
                snap[int(flat[a0])] = corner_uv[b0]
                snap[int(flat[a1])] = corner_uv[b1]
            for i, v in enumerate(flat[corners]):
                got = snap.get(int(v))
                if got is not None:
                    new[i] = got
            after = _tri_area2(new[_fan(indptr, faces, local=True)])  # per triangle, as the UV Check counts flips
            if csign[h] == 0 or np.any(np.sign(after) != csign[h]):
                continue
            hfaces = np.concatenate([face_order[cbounds[h] : cbounds[h + 1]]] + added.get(h, []))
            if (yield from _tris_overlap(new, faces, indptr, corner_uv, hfaces, 1e-6 * mean_len)):
                continue
            corner_uv[corners] = new
            chart[faces] = h
            added.setdefault(h, []).append(faces)
            moved += 1
        if moved:
            self.messages.append(f"{moved} small chart{'s' if moved != 1 else ''} joined a neighbour.")
        present = np.bincount(chart) > 0  # charts renumbered without a sort
        used, renum = np.flatnonzero(present), (np.cumsum(present) - 1)[chart]
        return renum.reshape(-1).astype(np.int64), corner_uv, np.asarray(rows)[used]

    def _charts_lying(self, chart, faces, corner_uv, rows=None):
        """Generator, returns the (sorted) chart ids among `faces` (a face index array) that overlap themselves or have a
        face wound against the rest, all charts at once (`overlapping_charts`). Charts whose `rows` entry is 2 (On charts:
        exact grids) are skipped."""

        if rows is not None:
            faces = faces[rows[chart[faces]] != 2]
        yield
        if not len(faces):
            return np.zeros(0, np.int64)
        ids, local = np.unique(chart[faces], return_inverse=True)
        local = local.reshape(-1)
        yield
        nt = np.maximum(np.diff(self.indptr)[faces] - 2, 0)
        parts = []
        for i in range(0, len(faces), FACE_CHUNK):  # the triangles' corners a few thousand faces at a time
            parts.append(corner_uv[_fan(self.indptr, faces[i : i + FACE_CHUNK])])
            yield
        t, tchart = np.concatenate(parts), np.repeat(local, nt)
        yield
        flags = yield from overlapping_charts(t, tchart, len(ids))
        return ids[flags]

    def _split_overlapping(self, tree, chart, corner_uv, rows, x):
        """A chart the solver kept free of flips can still lie over itself (a cap with a pole of valence 5 or more).
        Every chart is checked alone (all charts together, one grid per chart); one that overlaps is replaced by the
        charts of a lower Stretch allowed (half, quarter, ... when they lie inside it) and finally by its grid patches,
        whose exact layout cannot overlap. Returns (chart, corner UV, rows per chart, number of charts sent back)."""

        nf = len(self.indptr) - 1
        nchart = int(chart.max()) + 1
        chart, corner_uv = chart.copy(), corner_uv.copy()
        self._unresolved = set()
        rows_arr = np.asarray(rows, dtype=np.int64)
        rows = {c: int(rows_arr[c]) for c in range(nchart)}
        next_id = nchart

        def grouped(labels, count):
            order = _stable_order(labels)
            return order, np.searchsorted(labels[order], np.arange(count + 1))

        ladder = []  # the lower levels in order
        lower = x
        while lower > 0:
            lower = lower * LEVEL_STEP if lower >= 1 else 0.0
            ladder.append(lower)
        pending = yield from self._charts_lying(chart, np.arange(nf), corner_uv, rows_arr)  # On charts are exact grids
        pending = [int(c) for c in pending]
        back = len(pending)
        patch = np.asarray(tree.patches() if callable(getattr(tree, "patches", None)) else tree.patch, dtype=np.int64)
        order, start = grouped(chart, next_id)
        faces_of = {c: order[start[c] : start[c + 1]] for c in pending}  # R9first: the faces of each pending chart, kept up to date (no regrouping per rung)
        rung = 0
        while pending and rung < len(ladder):
            x = ladder[rung]
            rung += 1
            luv, lchart, lrows, _ = yield from self._flatten(tree, x)
            yield
            lorder, lstart = grouped(lchart, int(lchart.max()) + 1)
            still, born = [], []
            yield
            until = time.perf_counter() + self.budget
            for c in pending:
                if time.perf_counter() > until:
                    yield
                    until = time.perf_counter() + self.budget
                faces = faces_of[c]
                inside = np.unique(lchart[faces])
                lfaces = [lorder[lstart[i] : lstart[i + 1]] for i in inside]
                if not all(np.all(chart[f] == c) for f in lfaces):  # not nested: try a lower level
                    still.append(c)
                    continue
                corners = self._corners(faces)
                corner_uv[corners] = luv[corners]
                rows.pop(c, None)
                del faces_of[c]
                for i, f in zip(inside, lfaces):
                    chart[f] = next_id
                    rows[next_id] = int(lrows[int(i)])
                    if lrows[int(i)] != 2:
                        born.append(next_id)
                        faces_of[next_id] = np.sort(f)
                    next_id += 1
            if born:  # the new charts are checked together
                faces = np.sort(np.concatenate([faces_of[b] for b in born]))
                yield
                still += [int(c) for c in (yield from self._charts_lying(chart, faces, corner_uv))]
            pending = still
            keep = set(pending)
            for c in [c for c in faces_of if c not in keep]:
                del faces_of[c]  # born charts that lie flat: done
            yield
        until = time.perf_counter() + self.budget
        for c in pending:  # last resort: grid patches
            faces = faces_of[c]
            placed = []
            for p in np.unique(patch[faces]):
                if time.perf_counter() > until:  # a grid layout per patch: a slice holds as many as fit its time
                    yield
                    until = time.perf_counter() + self.budget
                pf = faces[patch[faces] == p]
                pc, pl, pp, ppts = self._sub(pf)
                grid = uvflat.grid_layout(ppts, pp, pl)
                if grid is None:
                    placed = None
                    break
                placed.append((pf, pc, grid[0][pl]))
            if placed is None:
                self._unresolved.add(int(c))  # a patch that is not a clean grid: laid out again on its own (`_redo_stretched`)
                continue
            rows.pop(c, None)
            for pf, pc, puv in placed:
                chart[pf] = next_id
                corner_uv[pc] = puv
                rows[next_id] = 2
                next_id += 1
        yield
        present = np.bincount(chart) > 0  # charts renumbered without a sort
        used, renum = np.flatnonzero(present), (np.cumsum(present) - 1)[chart]
        self._unresolved = {int(i) for i in np.searchsorted(used, sorted(self._unresolved))}
        return renum.reshape(-1).astype(np.int64), corner_uv, np.array([rows[int(u)] for u in used], dtype=np.int64), back

    def _corners(self, faces):
        """Mesh corner indices of a set of faces, in face order."""

        counts = np.diff(self.indptr)[faces]
        return np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts) + np.repeat(self.indptr[faces], counts)

    def _sub(self, faces):
        """(corner indices, local vertex per corner, local indptr, local points) of a set of faces."""

        indptr, flat = self.indptr, self.flat
        counts = np.diff(indptr)[faces]
        starts = indptr[faces]
        corners = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts) + np.repeat(starts, counts)
        uniq, local = np.unique(flat[corners], return_inverse=True)
        return corners, local.reshape(-1), np.concatenate([[0], np.cumsum(counts)]), self.points[uniq]

    def _poll_sight(self, sight):
        while not sight.done:
            self._active = None
            sight.step()
            yield
        self._active = None

    def _flatten(self, tree, x):
        """(corner UV in mesh order, chart per face, rows per chart 0/1/2, stretch per face) at the current Stretch allowed."""

        s = self.settings
        nf = len(self.indptr) - 1
        # chart by chart
        hide = bool(getattr(self, "_hidden", True))
        if hasattr(tree, "level_steps"):
            chart, cut, _ = yield from tree.level_steps(float(x), hide=hide)
        else:
            chart, cut, _ = tree.level(float(x), hide=hide)
        chart = np.asarray(chart, dtype=np.int64)
        corner_uv = np.zeros((len(self.flat), 2))
        rows = np.zeros(int(chart.max()) + 1 if len(chart) else 0, np.int64)
        ratio = np.ones(nf)
        mode = s.rows
        for c, faces, lpts, lptr, lflat, corner in uvchart.chart_meshes(tree, cut, chart):
            grid = uvflat.grid_layout(lpts, lptr, lflat)
            start, col, row = (None, None, None) if grid is None else grid
            use = mode if mode != "auto" else ("on" if grid is not None and start is not None else "soft")
            if grid is None and use == "on":
                use = "soft"
            stage = uvflat.Flatten(lpts, lptr, lflat, start, col=col, row=row, rows=use, start_mode="given" if start is not None else "free",
                                   stretch_allowed=float(x) / 100.0, texture_px=float(s.texture_px), budget=self.budget)
            self._active = stage
            while not stage.done:
                stage.step()
                yield
            self._active = None
            got = np.asarray(stage.uv, dtype=np.float64)
            corner_uv[np.asarray(corner, dtype=np.int64)] = got[lflat]
            rows[c] = {"off": 0, "soft": 1, "on": 2}.get(stage.numbers.get("rows", use), 1)
        return corner_uv, chart, rows, ratio

    def _labels(self, c, faces, corner_uv):
        """Grid labels of an On chart from the chart's own faces: (corner indices, col, row, (indptr, flat), angle) or None."""

        indptr, flat = self.indptr, self.flat
        sizes = np.diff(indptr)[faces]
        starts = indptr[faces]
        corners = np.concatenate([np.arange(a, a + n) for a, n in zip(starts, sizes)]) if len(faces) else np.zeros(0, np.int64)
        v = flat[corners]
        key = np.column_stack((v, np.round(corner_uv[corners] * 1e9).astype(np.int64)))
        _, local = np.unique(key, axis=0, return_inverse=True)
        local = local.reshape(-1)
        lptr = np.concatenate([[0], np.cumsum(sizes)])
        pts = np.zeros((int(local.max()) + 1, 3))
        pts[local] = self.points[v]
        grid = uvflat.grid_layout(pts, lptr, local)
        if grid is None:
            return None
        _, lcol, lrow = grid
        ccol, crow = lcol[local], lrow[local]
        luv = np.zeros((int(local.max()) + 1, 2))
        luv[local] = corner_uv[corners]
        angle = rows_angle(luv, lcol, lrow, lptr, local)
        return corners, ccol, crow, (lptr, local), angle

    # -- numbers ------------------------------------------------------------------------------------------------------
    def check(self, result: Result | None = None, texture_px: float | None = None):
        """UV Check numbers (`uvscore.Check`) of a finished run."""

        r = result or self.result
        m = uvscore.mesh(self.points, self.indptr, self.flat, r.uv, self.material)
        return uvscore.check(m, seen_face=r.seen_face if self.sight is not None else None,
                             seen_edge_px=r.seen_edge_px, col=r.col, row=r.row, on_chart=r.on_chart, density=r.density,
                             texture_px=texture_px or self.settings.texture_px)


def _fit_rigid(p: np.ndarray, q: np.ndarray) -> tuple:
    """Rotation R (2x2, no mirror) and shift t with R p + t closest to q (least squares, Kabsch in 2D)."""

    pc, qc = p.mean(axis=0), q.mean(axis=0)
    h = (p - pc).T @ (q - qc)
    ang = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    r = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return r, qc - pc @ r.T


def _fan(indptr, faces, local: bool = False) -> np.ndarray:
    """Fan triangles of `faces` as corner indices: mesh corners, or (local) positions in the faces' own corner order."""

    faces = np.asarray(faces, dtype=np.int64)
    n = (indptr[faces + 1] - indptr[faces]).astype(np.int64)
    base = (np.cumsum(n) - n) if local else indptr[faces].astype(np.int64)
    nt = np.maximum(n - 2, 0)
    b = np.repeat(base, nt)
    i = np.arange(int(nt.sum())) - np.repeat(np.cumsum(nt) - nt, nt) + 1
    return np.stack([b, b + i, b + i + 1], 1) if len(b) else np.zeros((0, 3), np.int64)


def _signed_areas(uv, indptr, faces, local: bool = False) -> np.ndarray:
    """Twice the signed UV area of every face of `faces` (uv per mesh corner, or per local corner when `local`)."""

    t = _fan(indptr, faces, local)
    tri = _tri_area2(uv[t])
    n = np.maximum(indptr[np.asarray(faces) + 1] - indptr[np.asarray(faces)] - 2, 0)
    return np.add.reduceat(tri, np.concatenate([[0], np.cumsum(n)[:-1]])) if len(tri) else tri


def _tris_overlap(new, faces, indptr, corner_uv, hfaces, eps):
    """Generator, returns True when a triangle of the moved small chart (`new`, per local corner of `faces`) overlaps a
    triangle of `hfaces` by more than `eps` (separating axes; shared edges and touching corners do not count)."""

    t1 = new[_fan(indptr, faces, local=True)]
    lo, hi = t1.reshape(-1, 2).min(axis=0) - eps, t1.reshape(-1, 2).max(axis=0) + eps
    t2 = corner_uv[_fan(indptr, hfaces)]
    t2lo, t2hi = _tri_box(t2)
    near = (t2hi >= lo).all(axis=1) & (t2lo <= hi).all(axis=1)
    t2 = t2[near]
    if not len(t2) or not len(t1):
        return False
    i, j = np.meshgrid(np.arange(len(t1)), np.arange(len(t2)), indexing="ij")
    i, j = i.ravel(), j.ravel()
    for a in range(0, len(i), CHECK_PAIRS):
        if _sat_overlap(t1[i[a : a + CHECK_PAIRS]], t2[j[a : a + CHECK_PAIRS]], eps).any():
            return True
        yield
    return False


def _l2_stretch(q: np.ndarray, t: np.ndarray, chart: np.ndarray, nchart: int, ratio=None) -> np.ndarray:
    """Per triangle (3D corners q (n, 3, 3), UV t (n, 3, 2)): the L2 stretch of Sander, Snyder, Gortler, Hoppe 2001,
    each chart scaled to its 3D area (the UV Check's stretch); inf for a triangle with no UV area."""

    a2s = 0.5 * _tri_area2(t)
    if ratio is None:  # UV area / 3D area per chart
        a3 = 0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
        ratio = np.bincount(chart, weights=np.abs(a2s), minlength=nchart) / np.maximum(np.bincount(chart, weights=a3, minlength=nchart), 1e-30)
    s_, t_ = t[:, :, 0], t[:, :, 1]
    d = np.where(a2s == 0, np.nan, 2 * a2s)[:, None]
    ss = (q[:, 0] * (t_[:, 1] - t_[:, 2])[:, None] + q[:, 1] * (t_[:, 2] - t_[:, 0])[:, None] + q[:, 2] * (t_[:, 0] - t_[:, 1])[:, None]) / d
    st = (q[:, 0] * (s_[:, 2] - s_[:, 1])[:, None] + q[:, 1] * (s_[:, 0] - s_[:, 2])[:, None] + q[:, 2] * (s_[:, 1] - s_[:, 0])[:, None]) / d
    out = np.sqrt(0.5 * ((ss * ss).sum(1) + (st * st).sum(1)) * ratio[chart])
    return np.where(np.isfinite(out), out, np.inf)


def _tri_box(t: np.ndarray) -> tuple:
    """(low corner, high corner) of the box of every triangle (n, 3, 2); `t.min(axis=1)` is slow on a short axis."""

    return np.minimum(np.minimum(t[:, 0], t[:, 1]), t[:, 2]), np.maximum(np.maximum(t[:, 0], t[:, 1]), t[:, 2])


def _tri_area2(t: np.ndarray) -> np.ndarray:
    """Twice the signed area of triangles (n, 3, 2)."""

    return (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])


def _sat_overlap(P: np.ndarray, Q: np.ndarray, eps) -> np.ndarray:
    """Per pair of triangles (n, 3, 2): True where their insides overlap by more than `eps` (a number, or one per pair;
    separating axes: two triangles that only touch, or share an edge or a corner, have a separating line along one of
    their six sides). All six axes at once."""

    d = np.concatenate([np.roll(P, -1, axis=1) - P, np.roll(Q, -1, axis=1) - Q], axis=1)  # (n, 6, 2) the six sides
    ax = np.stack([-d[..., 1], d[..., 0]], axis=-1)
    ax /= np.maximum(np.sqrt((ax * ax).sum(axis=-1, keepdims=True)), 1e-300)
    pp, qq = np.einsum("nkd,nad->nak", P, ax), np.einsum("nkd,nad->nak", Q, ax)  # (n, 6, 3) corners on each axis
    eps = np.asarray(eps, dtype=np.float64)
    if eps.ndim:
        eps = eps[:, None]
    return ((pp.max(axis=2) > qq.min(axis=2) + eps) & (qq.max(axis=2) > pp.min(axis=2) + eps)).all(axis=1)


def overlap_pairs(t: np.ndarray, eps: float, chunk: int = CHECK_PAIRS):
    """Generator over the triangle pairs of one chart (n, 3, 2) whose boxes meet, `chunk` pairs at a time: yields True as
    soon as two triangles overlap, else False after each chunk. Pairs come from a uniform grid of about one triangle
    per cell (spatial hashing), so the work grows with the triangles, not their square."""

    n = len(t)
    if n < 2:
        return
    lo, hi = _tri_box(t)
    cell = max(float(np.median(hi - lo)), 1e-300)
    yield False
    ext = hi - lo
    while True:  # a few huge triangles must not fill thousands of cells each: grow the cell until about CELLS_PER_TRI per triangle
        cnt = (np.floor(ext[:, 0] / cell) + 2) * (np.floor(ext[:, 1] / cell) + 2)
        if cnt.sum() <= CELLS_PER_TRI * n:
            break
        cell *= 1.5
    c0 = np.floor((lo - lo.min(axis=0)) / cell).astype(np.int64)
    c1 = np.floor((hi - lo.min(axis=0)) / cell).astype(np.int64)
    span = c1 - c0 + 1
    cnt = span[:, 0] * span[:, 1]
    tri = np.repeat(np.arange(n), cnt)
    k = np.arange(len(tri)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    cx = c0[tri, 0] + k % span[tri, 0]
    cy = c0[tri, 1] + k // span[tri, 0]
    key = cx * (int(c1[:, 1].max()) + 2) + cy
    order = np.lexsort((tri, key))
    key, tri = key[order], tri[order]
    pairs = []
    for d in range(1, len(key)):
        same = np.flatnonzero(key[d:] == key[:-d])
        if not len(same):
            break
        pairs.append(np.stack([tri[same], tri[same + d]], 1))
        if d % 8 == 0:
            yield False
    if not pairs:
        return
    yield False
    pr = np.sort(np.concatenate(pairs), axis=1)
    code = np.unique(pr[:, 0] * n + pr[:, 1])
    yield False
    pr = np.stack([code // n, code % n], 1)
    box = (hi[pr[:, 0]] >= lo[pr[:, 1]]).all(axis=1) & (hi[pr[:, 1]] >= lo[pr[:, 0]]).all(axis=1)
    pr = pr[box]
    for i in range(0, len(pr), chunk):
        q = pr[i : i + chunk]
        yield bool(_sat_overlap(t[q[:, 0]], t[q[:, 1]], eps).any())


def overlapping_charts(t: np.ndarray, tchart: np.ndarray, nchart: int, chunk: int = CHECK_PAIRS):
    """Generator, returns a bool per chart: True where two triangles of the chart overlap (by more than `OVERLAP_EPS` of
    the chart's mean triangle side) or the chart has a face wound against the rest or a face with no area. t is
    (n, 3, 2), `tchart` the chart of every triangle. Every chart gets its own uniform grid (spatial hashing, cells of
    about one triangle), all charts in one pass. Nothing here works on more than a few thousand triangles or grid
    entries at a time (entries are split into stripes by key and sorted stripe by stripe), so a slice stays short on any
    mesh. A pair is tested once, in the cell that holds the corner of its boxes' intersection, so no pair list needs
    de-duplicating."""

    n = len(t)
    flags = np.zeros(nchart, bool)
    if n == 0:
        return flags
    area = np.empty(n)
    pos, neg, zero = (np.zeros(nchart) for _ in range(3))
    for i in range(0, n, TRI_CHUNK * 4):
        sl = slice(i, i + TRI_CHUNK * 4)
        a, c = _tri_area2(t[sl]), tchart[sl]
        area[sl] = a
        pos += np.bincount(c, weights=a > 0, minlength=nchart)
        neg += np.bincount(c, weights=a < 0, minlength=nchart)
        zero += np.bincount(c, weights=a == 0, minlength=nchart)
        yield False
    flags |= ((pos > 0) & (neg > 0)) | (zero > 0)
    keep = np.flatnonzero(~flags[tchart])
    if len(keep) < 2:
        return flags
    yield False
    t, tc, area = t[keep], tchart[keep], np.abs(area[keep])
    n = len(t)
    yield False
    count = np.maximum(np.bincount(tc, minlength=nchart), 1)
    eps_c = OVERLAP_EPS * np.sqrt(np.bincount(tc, weights=area, minlength=nchart) / count)
    lo, hi = _tri_box(t)
    ext = hi - lo
    yield False
    cell_c = np.maximum(np.bincount(tc, weights=ext.mean(axis=1), minlength=nchart) / count, 1e-9 * max(float(np.abs(t).max()), 1e-300))
    while True:  # a few huge triangles must not fill thousands of cells each: grow a chart's cell until about CELLS_PER_TRI per triangle
        cell = cell_c[tc]
        cnt = (np.floor(ext[:, 0] / cell) + 2) * (np.floor(ext[:, 1] / cell) + 2)
        yield False
        bad = np.bincount(tc, weights=cnt, minlength=nchart) > CELLS_PER_TRI * count
        if not bad.any():
            break
        cell_c[bad] *= 1.5
        yield False
    stripes = max(1, -(-int(cnt.sum()) // STRIPE_ENTRIES))
    parts = [[] for _ in range(stripes)]
    for i in range(0, n, TRI_CHUNK):
        j = min(n, i + TRI_CHUNK)
        cell = cell_c[tc[i:j]][:, None]
        c0 = np.floor(lo[i:j] / cell).astype(np.int64)
        span = np.floor(hi[i:j] / cell).astype(np.int64) - c0 + 1
        per = span[:, 0] * span[:, 1]
        own = np.repeat(np.arange(j - i), per)
        k = np.arange(len(own)) - np.repeat(np.cumsum(per) - per, per)
        cx = c0[own, 0] + k % span[own, 0]
        cy = c0[own, 1] + k // span[own, 0]
        key = tc[i + own] * _H1 + cx * _H2 + cy * _H3  # int64 wrap-around is the hash; a collision only adds a pair that is filtered below
        which = key % stripes
        order = np.argsort(which, kind="stable")
        bounds = np.searchsorted(which[order], np.arange(stripes + 1))
        key, tri_, cx, cy = key[order], (own + i)[order], cx[order], cy[order]
        for p in range(stripes):
            b0, b1 = bounds[p], bounds[p + 1]
            if b1 > b0:
                parts[p].append((key[b0:b1], tri_[b0:b1], cx[b0:b1], cy[b0:b1]))
        yield False
    queue, queued = [], 0

    def flush():  # the queued pairs, `chunk` at a time
        nonlocal queue, queued
        if queue:
            pa, pb = np.concatenate([q[0] for q in queue]), np.concatenate([q[1] for q in queue])
            queue, queued = [], 0
            for i in range(0, len(pa), chunk):
                qa, qb = pa[i : i + chunk], pb[i : i + chunk]
                live = ~flags[tc[qa]]
                qa, qb = qa[live], qb[live]
                if len(qa):
                    flags[tc[qa[_sat_overlap(t[qa], t[qb], eps_c[tc[qa]])]]] = True
                yield False

    for part in parts:
        if not part:
            continue
        key, tri_, cx, cy = (np.concatenate([q[m] for q in part]) for m in range(4))
        order = np.argsort(key, kind="stable")
        key, tri_, cx, cy = key[order], tri_[order], cx[order], cy[order]
        yield False
        for d in range(1, len(key)):
            same = np.flatnonzero(key[d:] == key[:-d])
            if not len(same):
                break
            a, b = tri_[same], tri_[same + d]
            ok = (tc[a] == tc[b]) & (a != b) & (cx[same] == cx[same + d]) & (cy[same] == cy[same + d])
            ok &= (hi[a] >= lo[b]).all(axis=1) & (hi[b] >= lo[a]).all(axis=1)
            a, b, same = a[ok], b[ok], same[ok]
            if len(a):  # the pair belongs to the cell that holds the corner of its boxes' intersection
                ref = np.floor(np.maximum(lo[a], lo[b]) / cell_c[tc[a]][:, None]).astype(np.int64)
                ok = (ref[:, 0] == cx[same]) & (ref[:, 1] == cy[same])
                a, b = a[ok], b[ok]
            if len(a):
                queue.append((a, b))
                queued += len(a)
            if queued >= chunk:
                yield from flush()
            if d % 2 == 0:
                yield False
        yield from flush()
        yield False
    return flags


def _uv_area(uv, indptr, flat, chart, nchart) -> np.ndarray:
    """UV area per chart (shoelace per face)."""

    nf = len(indptr) - 1
    owner = np.repeat(np.arange(nf), np.diff(indptr))
    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    cross = uv[:, 0] * uv[nxt, 1] - uv[nxt, 0] * uv[:, 1]
    per_face = np.abs(np.bincount(owner, weights=cross, minlength=nf)) / 2.0
    return np.bincount(chart, weights=per_face, minlength=nchart)


