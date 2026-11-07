# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""New UVs for the add-on (package A2): the whole pipeline of SPEC section 6 as sliced stages over the K4-K7 cores.

NumPy only, no bpy. Stages, each a state object with `step()` of about 4 ms that the caller (Look First's timer, a
script loop, a test) keeps calling: 1. what the camera sees (`uvsight.Sight`, only when it matters: seams out of
sight or hidden parts get fewer pixels); 2. patches and the merge tree (`uvchart.Charts`); 3. charts at Stretch allowed
and their flat layouts (quad ARAP, Straight rows; the faster `uvs` callable when the adapter installed one, else the
NumPy twin chart by chart); 4. per chart: Hidden-parts scale, box by rotating calipers (Toussaint 1983) or the rows
axis for On charts; 5. pages: MaxRects boxes at the largest scale that fits (`uvpack.FillPage`, Jylanki 2010), whole
texels (Carr, Hoberock, Crane, Hart 2006) and the gap of SPEC 8.2; 6. corner UVs written back in mesh order.

Nothing here calls compiled code: the faster callables come in through the cores' `FAST` tables (`uv_fast.install()`),
which is also how `uvsight` gets its pictures. `again()` reruns from the stage that a changed setting touches and keeps
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
ROW_TOL_DEG = 1e-6
WAIT = "wait"  # a stage yields this while a faster job runs: step() returns at once instead of spinning
SELF_CHECK_FACES = 4  # charts smaller than this are not checked for lying over themselves (a pole of valence 5 needs 5 faces)
CHECK_BATCH = 48  # charts checked in one call, each in its own tile
RASTER_PX = 128  # tile size of the check that runs without the faster callable
SELF_CHECK_PX = 256.0  # a checked chart is drawn this big (its longest side is SELF_CHECK_FILL of it)
SELF_CHECK_FILL = 0.9
NUMPY_CHECK_FACES = 30000  # without the faster check the guard is skipped above this size
ROWS_INDEX = {"off": 0, "soft": 1, "on": 2, "auto": 3}


def mip_level(texture_px: float, clean_px: float = CLEAN_DOWN_TO_PX) -> int:
    """k of the gap rule: how many halvings from the texture size down to `clean_px` (never below 0)."""

    return max(0, int(round(log2(max(float(texture_px), 1.0) / max(float(clean_px), 1.0)))))


def gap_texels(texture_px: float, clean_px: float = CLEAN_DOWN_TO_PX) -> int:
    """SPEC 8.2: 2^(k+1) texels (16 at 2048, 8 at 1024)."""

    return 1 << (mip_level(texture_px, clean_px) + 1)


class Settings:
    """The New UVs popover values (SPEC 10.1), in the units the card shows."""

    def __init__(self, **kw):
        self.stretch_allowed = 10.0  # percent; P0's start value (SPEC 6.4), P1 tunes. Sweeps 10-100 on 13 models: charts within 25 %
        self.rows = "auto"  # auto / on / soft / off
        self.out_of_sight = True
        self.hidden_pct = 50.0  # px per metre of hidden parts, percent of seen ones
        self.views = uvsight.VIEWS_DEFAULT
        self.all_sides = True
        self.texture_px = TEXTURE_PX
        self.clean_px = CLEAN_DOWN_TO_PX
        self.time_per_pass = 2.0  # seconds one search (Fill the page) may run before it shows what it has
        self.density = None  # px per mesh unit; None = Fill the page
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


def edge_ids(indptr, flat) -> tuple:
    """(edge id of every corner's edge to the next corner, edge count), edges numbered by first use (the order of `seen_edge_px`)."""

    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    lo, hi = np.minimum(flat, flat[nxt]), np.maximum(flat, flat[nxt])
    keys = lo.astype(np.uint64) << np.uint64(32) | hi.astype(np.uint64)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return rank[inverse.reshape(-1)], len(order)


def cut_edges(indptr, flat, uv, tol: float = 1e-6) -> np.ndarray:
    """Per edge (first-use order): True where the two faces meeting there do not share the edge's UVs (a UV seam)."""

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    ids, ne = edge_ids(indptr, flat)
    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    cut = np.zeros(ne, bool)
    count = np.bincount(ids, minlength=ne)
    order = np.argsort(ids, kind="stable")
    first = np.concatenate([[0], np.cumsum(count)])[:-1]
    two = np.flatnonzero(count == 2)
    if len(two):
        a, b = order[first[two]], order[first[two] + 1]  # the two corners that use the edge
        same = flat[a] == flat[b]  # faces that disagree on the winding run the edge the same way
        ua0, ua1 = uv[a], uv[nxt[a]]
        ub0 = np.where(same[:, None], uv[b], uv[nxt[b]])
        ub1 = np.where(same[:, None], uv[nxt[b]], uv[b])
        cut[two] = (np.abs(ua0 - ub0).max(axis=1) > tol) | (np.abs(ua1 - ub1).max(axis=1) > tol)
    return cut


def _hull(p: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points (Andrew's monotone chain), counter-clockwise, no repeated end point."""

    pts = np.unique(np.round(p, 12), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))].tolist()

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (q[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def min_area_angle(p: np.ndarray) -> float:
    """Angle (radians) that turns `p` into its smallest axis-aligned box: rotating calipers over the hull's edges."""

    h = _hull(p)
    if len(h) < 3:
        return 0.0
    d = np.roll(h, -1, axis=0) - h
    ang = np.arctan2(d[:, 1], d[:, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    x = h[None, :, 0] * c[:, None] - h[None, :, 1] * s[:, None]
    y = h[None, :, 0] * s[:, None] + h[None, :, 1] * c[:, None]
    area = (x.max(axis=1) - x.min(axis=1)) * (y.max(axis=1) - y.min(axis=1))
    return float(-ang[int(np.argmin(area))])


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
        self.waiting = False
        self._active = None  # the sliced core stage being polled
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
        self.waiting = False
        try:
            while time.perf_counter() < until:
                if next(self._work) == WAIT:
                    self.waiting = True  # a job runs: the caller should not spin on step()
                    break
        except StopIteration:
            self.done = True
            self.progress = 1.0
        except Exception as exc:  # the caller falls back (Smart Project) and shows the message
            self.error, self.done = exc, True
        return self.progress

    def run(self) -> Result:
        """Run to the end (scripts and tests)."""

        while not self.done:
            self.step()
            if self.waiting:
                time.sleep(0.0005)
        if self.error is not None:
            raise self.error
        return self.result

    def again(self, **changes) -> "NewUVs":
        """A new run with changed settings that reuses the pictures (always) and the merge tree (unless a setting it
        was built with changed)."""

        new = self.settings.copy(**changes)
        same = lambda *names: all(getattr(new, k) == getattr(self.settings, k) for k in names)  # noqa: E731
        keep = {"sight": self.sight if self.sight is not None and self.sight.done and same("views", "all_sides") else None}
        if same("out_of_sight", "hidden_l", "views", "all_sides") and self.tree is not None:
            keep["tree"] = self.tree
        return NewUVs(self.points, self.indptr, self.flat, self.material, self.forced, new, self.budget, _keep=keep)

    def _poll(self, stage):
        """Step a core stage until it reports finished (a faster job makes it yield WAIT)."""

        self._active = stage
        while True:
            stage.step()
            if getattr(stage, "done", False) is True:
                break
            yield WAIT if getattr(stage, "_job", None) is not None else None
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
        area = face_areas(pts, indptr, flat)
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
                                   out_of_sight=bool(s.out_of_sight and inside is not None), hidden_l=s.hidden_l, budget=self.budget)
            self._active = stage
            while not stage.done:
                value = stage.step()
                yield WAIT if value == 0.5 else None
            self._active = None
            tree = stage.tree
        self.tree = tree
        res.timing["charts_s"] = time.perf_counter() - t1
        t2 = time.perf_counter()

        x_now = float(s.stretch_allowed)
        # 3. charts at Stretch allowed, flattened
        self._tick("flatten", 0.5)
        corner_uv, chart, chart_rows, ratio = yield from self._flatten(tree, x_now)
        if self._stop:
            res.messages.append("Stopped early: the layout is what had been reached.")
        res.timing["flatten_s"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        chart, corner_uv, chart_rows, back = yield from self._split_overlapping(tree, chart, corner_uv, chart_rows, x_now)
        if back:
            res.messages.append(f"{back} chart{'s' if back != 1 else ''} lay over themselves and were redone at a lower Stretch allowed.")
        nchart = int(chart.max()) + 1 if len(chart) else 0
        res.chart, res.charts = chart, nchart
        res.stats = chart_rows
        corner_face = np.repeat(np.arange(nf), np.diff(indptr))
        corner_chart = chart[corner_face]
        yield

        # 4. per chart: grid labels for On charts, the rows axis or the calipers box, the Hidden-parts scale
        self._tick("orient", 0.7)
        on = chart_rows == 2
        res.on_chart = on[chart]
        col = np.full(len(flat), -1, np.int64)
        row = np.full(len(flat), -1, np.int64)
        order = np.argsort(corner_chart, kind="stable")
        bounds = np.searchsorted(corner_chart[order], np.arange(nchart + 1))
        face_order = np.argsort(chart, kind="stable")
        fbounds = np.searchsorted(chart[face_order], np.arange(nchart + 1))
        angle = np.zeros(nchart)
        for c in range(nchart):
            cc = order[bounds[c] : bounds[c + 1]]
            if on[c]:
                got = self._labels(c, face_order[fbounds[c] : fbounds[c + 1]], corner_uv)
                if got is not None:
                    col[got[0]], row[got[0]] = got[1], got[2]
                    if got[4] is not None:
                        angle[c] = got[4]
                        continue
                res.on_chart[face_order[fbounds[c] : fbounds[c + 1]]] = False  # no clean labels: not counted as a straight-rows chart
            angle[c] = min_area_angle(corner_uv[cc]) if len(cc) > 2 else 0.0
            if c % 16 == 15:
                yield
        res.col, res.row = col, row
        ca, sa = np.cos(angle)[corner_chart], np.sin(angle)[corner_chart]
        ru = corner_uv[:, 0] * ca - corner_uv[:, 1] * sa
        rv = corner_uv[:, 0] * sa + corner_uv[:, 1] * ca
        # Isotropic fit to 3D area per chart (the solver works in surface units already; this only removes rounding),
        # then the Hidden-parts factor: scale ~ sum(area * max(floor, seen)) / sum(area), no square root (SPEC 6.7).
        floor_ = min(max(s.hidden_pct, 0.0), 100.0) / 100.0
        a3 = np.bincount(chart, weights=area, minlength=nchart)
        a2 = _uv_area(corner_uv, indptr, flat, chart, nchart)
        fit = np.where(a2 > 0, np.sqrt(np.divide(a3, a2, out=np.ones(nchart), where=a2 > 0)), 1.0)
        weighted = np.bincount(chart, weights=area * np.maximum(floor_, seen_face), minlength=nchart)
        factor = np.where(a3 > 0, weighted / np.where(a3 > 0, a3, 1.0), 1.0)
        mult = (fit * factor)[corner_chart]
        ru, rv = ru * mult, rv * mult
        lo_u = np.full(nchart, np.inf)
        lo_v = np.full(nchart, np.inf)
        hi_u = np.full(nchart, -np.inf)
        hi_v = np.full(nchart, -np.inf)
        np.minimum.at(lo_u, corner_chart, ru)
        np.minimum.at(lo_v, corner_chart, rv)
        np.maximum.at(hi_u, corner_chart, ru)
        np.maximum.at(hi_v, corner_chart, rv)
        width, height = np.maximum(hi_u - lo_u, 1e-12), np.maximum(hi_v - lo_v, 1e-12)
        res.density = factor[chart]
        yield

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
                stage = uvpack.FillPage(width.tolist(), height.tolist(), [True] * nchart, weight.tolist(), page, gap_k, budget=float(s.time_per_pass))
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
        res.timing["pack_s"] = time.perf_counter() - t3
        t4 = time.perf_counter()

        # 6. corner UVs on the 0-1 square
        px, py, turned = (np.array([p[i] for p in fill["places"]]) for i in range(3))
        sc = res.scale
        du = (ru - lo_u[corner_chart]) * sc
        dv = (rv - lo_v[corner_chart]) * sc
        wsc = (width * sc)[corner_chart]
        turn = turned.astype(bool)[corner_chart]
        uu = np.where(turn, dv, du)  # a quarter turn (u, v) -> (v, w - u) keeps the winding
        vv = np.where(turn, wsc - du, dv)
        out = np.empty_like(corner_uv)
        out[:, 0] = (px[corner_chart] + uu) / page[0]
        out[:, 1] = (py[corner_chart] + vv) / page[1]
        res.uv = out
        res.timing["write_s"] = time.perf_counter() - t4
        res.timing["total_s"] = time.perf_counter() - t0
        res.messages = list(self.messages) + res.messages
        self.result = res
        self._tick("done", 1.0)
        yield

    def _lies_over_itself(self, groups, corner_uv):
        """Generator: True when any chart of `groups` (face index arrays) overlaps itself or has a flipped face. The charts
        are drawn at a small size, each in its own integer tile, so one check answers for the whole group."""

        side = int(np.ceil(np.sqrt(len(groups))))
        pieces, vbase, points, ptr, flat_, uvs = [], 0, [], [0], [], []
        for i, faces in enumerate(groups):
            corners, local, lptr, pts = self._sub(faces)
            uv = corner_uv[corners]
            span = float((uv.max(axis=0) - uv.min(axis=0)).max())
            if not span > 0:
                continue
            uv = (uv - uv.min(axis=0)) * (SELF_CHECK_FILL / span) + np.array([i % side, i // side])
            points.append(pts)
            flat_.append(local + vbase)
            ptr.extend((lptr[1:] + ptr[-1]).tolist())
            uvs.append(uv)
            vbase += len(pts)
        if not points:
            return False
        m = uvscore.mesh(np.vstack(points), np.array(ptr), np.concatenate(flat_), np.vstack(uvs))
        fast = uvscore.FAST.get("check")
        z, zi, zb = np.zeros(0), np.zeros(0, np.int64), np.zeros(0, bool)
        job = None
        if fast is not None:
            job = fast(m.points, m.indptr, m.flat, m.uv, zi, (), (), (), (), (), src_region=zi, seen_face=z, seen_edge_px=z, allow_overlap=zb, col=zi,
                       row=zi, on_chart=zb, density=z, faces_off=0, split=zb, texture_px=SELF_CHECK_PX)
        if job is not None:
            while not job.done():
                yield WAIT
            numbers = np.asarray(job.result()[0])
        else:
            return (yield from self._raster_overlap(m, side))
        return bool(numbers[uvscore._I["overlap_pct"]] > 0.0 or numbers[uvscore._I["flipped_faces"]] > 0)

    def _raster_overlap(self, m, side):
        """The same question without the faster check, in slices: pixel centres covered by two triangles, and triangles
        wound against their tile's majority. Coarser than `uvscore.check` (128 px per tile, centres only)."""

        res = RASTER_PX
        sizes = np.diff(m.indptr)
        fan = np.concatenate([np.arange(1, n - 1) for n in sizes]) if len(sizes) else np.zeros(0, np.int64)
        face = np.repeat(np.arange(len(sizes)), np.maximum(sizes - 2, 0))
        base = m.indptr[face]
        a, b, c = (m.uv[base] * res), (m.uv[base + fan] * res), (m.uv[base + fan + 1] * res)
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        tile = (np.floor(a[:, 1] / res) * side + np.floor(a[:, 0] / res)).astype(np.int64)
        ntile = side * side
        pos = np.bincount(tile, weights=(area > 0).astype(float), minlength=ntile)
        neg = np.bincount(tile, weights=(area < 0).astype(float), minlength=ntile)
        if (np.minimum(pos, neg) > 0).any():
            return True  # a triangle wound against its chart
        image = np.zeros((side * res + 1, side * res + 1), np.int16)
        until = time.perf_counter() + self.budget
        for i in range(len(area)):
            if not area[i]:
                continue
            tri = np.array([a[i], b[i], c[i]])
            lo, hi = np.floor(tri.min(axis=0)).astype(int), np.ceil(tri.max(axis=0)).astype(int)
            xs, ys = np.meshgrid(np.arange(lo[0], hi[0]) + 0.5, np.arange(lo[1], hi[1]) + 0.5)
            w0 = ((b[i, 0] - xs) * (c[i, 1] - ys) - (b[i, 1] - ys) * (c[i, 0] - xs)) / area[i]
            w1 = ((c[i, 0] - xs) * (a[i, 1] - ys) - (c[i, 1] - ys) * (a[i, 0] - xs)) / area[i]
            inside = (w0 > 1e-9) & (w1 > 1e-9) & (w0 + w1 < 1 - 1e-9)
            image[lo[1] : hi[1], lo[0] : hi[0]] += inside
            if time.perf_counter() > until:
                yield
                until = time.perf_counter() + self.budget
        return bool((image >= 2).any())

    def _split_overlapping(self, tree, chart, corner_uv, rows, x):
        """A chart the solver kept free of flips can still lie over itself (a cap with a pole of valence 5 or more).
        Each chart of at least `SELF_CHECK_FACES` faces is checked alone; one that overlaps is replaced by the charts of
        a lower Stretch allowed (half, quarter, ... when they lie inside it) and finally by its grid patches, whose exact
        layout cannot overlap. Returns (chart, corner UV, rows per chart, number of charts sent back)."""

        indptr, nf = self.indptr, len(self.indptr) - 1
        nchart = int(chart.max()) + 1
        fast = uvscore.FAST.get("check")
        if fast is None and nf > NUMPY_CHECK_FACES:
            return chart, corner_uv, rows, 0
        chart, corner_uv = chart.copy(), corner_uv.copy()
        rows = {c: int(rows[c]) for c in range(nchart)}
        next_id = nchart

        def faces_of_each():
            order = np.argsort(chart, kind="stable")
            ids, start = np.unique(chart[order], return_index=True)
            return {int(i): order[a:b] for i, a, b in zip(ids, start, list(start[1:]) + [len(order)])}

        pending = []
        by = faces_of_each()
        todo = [c for c, faces in by.items() if len(faces) >= SELF_CHECK_FACES and rows.get(c) != 2]  # On charts are exact grids
        for i in range(0, len(todo), CHECK_BATCH):
            batch = todo[i : i + CHECK_BATCH]
            if (yield from self._lies_over_itself([by[c] for c in batch], corner_uv)):
                for c in batch:
                    if len(batch) == 1 or (yield from self._lies_over_itself([by[c]], corner_uv)):
                        pending.append(c)
        back = len(pending)
        patch = np.asarray(tree.patches() if callable(getattr(tree, "patches", None)) else tree.patch, dtype=np.int64)
        while pending and x > 0:
            x = x / 2 if x >= 1 else 0.0
            luv, lchart, lrows, _ = yield from self._flatten(tree, x)
            lby = {}
            order = np.argsort(lchart, kind="stable")
            ids, start = np.unique(lchart[order], return_index=True)
            lby = {int(i): order[a:b] for i, a, b in zip(ids, start, list(start[1:]) + [len(order)])}
            still = []
            for c in pending:
                faces = np.flatnonzero(chart == c)
                inside = np.unique(lchart[faces])
                if not all(np.all(chart[lby[int(i)]] == c) for i in inside):  # not nested: try a lower level
                    still.append(c)
                    continue
                corners = np.concatenate([np.arange(indptr[f], indptr[f + 1]) for f in faces])
                corner_uv[corners] = luv[corners]
                rows.pop(c, None)
                for i in inside:
                    chart[lby[int(i)]] = next_id
                    rows[next_id] = int(lrows[int(i)])
                    if len(lby[int(i)]) >= SELF_CHECK_FACES and lrows[int(i)] != 2 and (yield from self._lies_over_itself([lby[int(i)]], corner_uv)):
                        still.append(next_id)
                    next_id += 1
            pending = still
        for c in pending:  # last resort: grid patches
            faces = np.flatnonzero(chart == c)
            placed = []
            for n_done, p in enumerate(np.unique(patch[faces])):
                pf = faces[patch[faces] == p]
                pc, pl, pp, ppts = self._sub(pf)
                grid = uvflat.grid_layout(ppts, pp, pl)
                if grid is None:
                    placed = None
                    break
                placed.append((pf, pc, grid[0][pl]))
                if n_done % 16 == 15:
                    yield
            if placed is None:
                continue  # a patch that is not a clean grid (a ring): the chart stays as the solver left it
            rows.pop(c, None)
            for pf, pc, puv in placed:
                chart[pf] = next_id
                corner_uv[pc] = puv
                rows[next_id] = 2
                next_id += 1
        used, renum = np.unique(chart, return_inverse=True)
        return renum.reshape(-1).astype(np.int64), corner_uv, np.array([rows[int(u)] for u in used], dtype=np.int64), back

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
        fast = uvchart.FAST.get("uvs")
        if fast is not None and not hasattr(tree, "topo"):
            job = fast(tree, float(x), rows=ROWS_INDEX.get(s.rows, 3), stretch_allowed=float(x) / 100.0,
                       texture_px=float(s.texture_px))
            if job is not None:
                self._active = job
                while not job.done():
                    yield WAIT
                self._active = None
                uv, chart, stats, ratio = job.result()
                stats = np.asarray(stats, dtype=np.float64).reshape(-1, uvflat.STATS_LEN)
                chart = np.asarray(chart, dtype=np.int64)
                rows = np.zeros(int(chart.max()) + 1 if len(chart) else 0, np.int64)
                for r in stats:
                    if 0 <= int(r[0]) < len(rows) and np.isfinite(r[4]):
                        rows[int(r[0])] = int(r[4])
                return np.asarray(uv, dtype=np.float64).reshape(-1, 2), chart, rows, np.asarray(ratio, dtype=np.float64)
        # NumPy twin: chart by chart.
        chart, cut, _ = tree.level(float(x))
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
                                   stretch_allowed=float(x) / 100.0, texture_px=float(s.texture_px), budget=self.budget, use_fast=False)
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


def _uv_area(uv, indptr, flat, chart, nchart) -> np.ndarray:
    """UV area per chart (shoelace per face)."""

    nf = len(indptr) - 1
    owner = np.repeat(np.arange(nf), np.diff(indptr))
    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    cross = uv[:, 0] * uv[nxt, 1] - uv[nxt, 0] * uv[:, 1]
    per_face = np.abs(np.bincount(owner, weights=cross, minlength=nf)) / 2.0
    return np.bincount(chart, weights=per_face, minlength=nchart)


def seen_cut_px(indptr, flat, uv, seen_edge_px) -> tuple:
    """(seen cut px, total cut edges) of a UV layout: the inside pixels summed over the edges where the UVs split."""

    cut = cut_edges(indptr, flat, uv)
    if seen_edge_px is None:
        return 0.0, int(cut.sum())
    return float(np.asarray(seen_edge_px)[cut].sum()), int(cut.sum())
