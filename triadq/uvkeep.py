# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Keep source UVs in the add-on (package A1, SPEC 4.2, 4.6): which source faces stay one kind before the search, and
the stage after it (snap, split leftovers, per-side transfer) that makes the result's UVs sit where the source's did.

NumPy only, no bpy. The seams, snap, split and transfer cores do the work (`uvseams`, `uvsnap`); this file chooses and
orders them. Long work is a sliced stage (`FitStage.step()` does about 4 ms, or polls a background job the adapter
installed), driven by whoever owns the timer.

Our own design and code. The Hard/Soft choice and its order, the reconciliation of Hard chains that share a kind, and
the face budget for splits are written in SPEC 4.2 and 4.6; cost estimates follow the model fitted by the bench.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math

import threading

import numpy as np

from . import uvseams, uvsnap
from .types import FlatFaces
from .uvs import MOST_CHARTS, chart_codes

# Constants table (starting values; the bench tunes them, nothing here is a limit).
DEFAULT_ALLOWANCE = 0.15  # extra faces for seams, as a share of the faces with Seam Fit off
MOVE_SHARE = 0.5  # the snap may move a vertex this share of its local edge, unless the caller gives a view check
FALLBACK_TARGET_SHARE = 0.015  # faces the result is expected to have, as a share of the source's, when the aim is "keep detail"
TEXTURE_PX = 1024.0
CACHED_CANDIDATES = 8  # finished fits kept so Apply can write exactly what was previewed
VIEW_PIXELS = 192  # a side of one source picture for the snap's camera check (higher can take longer)
VIEW_TURNED_DEG = 20.0  # a drawn face turned further than this from the picture's normal counts as wrong (the camera check's rule)
VIEW_BEHIND_PIXELS = 6.0  # a face this far behind the picture is hidden by something in front of it
DEFAULT_DETAIL = 6.0  # Most Detail Loss in percent when the settings give none

HARD, SOFT = uvseams.HARD, uvseams.SOFT

# Set by a test harness (or the panel's property group, later) before the search; every key is optional.
OVERRIDE: dict = {}


@dataclass
class Settings:
    mode: str = "off"  # "light", "exact" or "off"
    allowance: float | None = DEFAULT_ALLOWANCE  # None = "any": split everything
    split: bool = True
    snap: bool = True
    move_share: float = MOVE_SHARE
    texture_px: float = TEXTURE_PX
    repeat: tuple = ()
    cut_here: tuple = ()
    view_check: bool = True  # the camera check holds snap moves that would spend more than Most Detail Loss allows
    detail: float = DEFAULT_DETAIL  # Most Detail Loss, percent of the source's picture pixels
    view_pixels: int = VIEW_PIXELS


def settings_of(gs) -> Settings:
    """Settings from the guided settings `gs` plus OVERRIDE. Until the UVs row exists: Keep UV Seams on = Exact, off = Off."""

    ov = OVERRIDE
    mode = ov.get("mode") or ("exact" if getattr(gs, "keep_uvs", False) else "off")
    s = Settings(mode=mode)
    detail = getattr(gs, "auto_detail", None)
    if isinstance(detail, (int, float)) and detail > 0:
        s.detail = float(detail)
    for key in ("allowance", "split", "snap", "move_share", "texture_px", "repeat", "cut_here", "view_check", "detail", "view_pixels"):
        if key in ov:
            setattr(s, key, ov[key])
    return s


def mode_of(gs) -> str:
    return settings_of(gs).mode


# ---------------------------------------------------------------- the source as arrays


@dataclass
class Source:
    """The welded source as the cores take it: positions, CSR faces, material per face, one UV array per layer (corner order)."""

    points: np.ndarray
    indptr: np.ndarray
    flat: np.ndarray
    material: np.ndarray
    layers: list  # [(name, (corners, 2) float64)]
    active: int
    key: tuple


def source_of(source, src) -> Source | None:
    """The source mesh object's welded arrays (`src` is guided's welded copy), or None when it has no UV layer."""

    from .. import uv_layers

    mesh = source.data
    names = uv_layers.layer_names(mesh)
    if not names:
        return None
    loops = FlatFaces.of(src["loops"])
    faces = FlatFaces.of(src["faces"])
    layers = []
    for name in names:
        uv = uv_layers.read_uv(mesh, name)
        layers.append((name, uv[loops.flat]))
    active = max(0, mesh.uv_layers.active_index)
    key = (src.get("sig"), len(faces.flat), hashlib.sha1(layers[active][1].astype(np.float32).tobytes()).hexdigest())
    return Source(np.ascontiguousarray(src["co"], np.float64).reshape(-1, 3), faces.indptr, faces.flat, np.asarray(src["mat"], np.int64), layers, active, key)


# ---------------------------------------------------------------- before the search: the kinds


@dataclass
class Plan:
    """What the pre-search choice decided: final chain classes, the kind of every source face, and what to tell the user."""

    seams: dict
    kind: np.ndarray  # per source face
    codes: tuple | None  # (code per face, material per code, kind per code), None = too many
    hard_spent: float  # estimated extra faces the Hard borders cost
    f_target: float
    edge_est: float
    report: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)


_PLANS: OrderedDict = OrderedDict()
_LATEST: dict = {}  # source key -> the plan the last search used


def _triaged(sa: Source, f_target: float, allowance: float, settings: Settings) -> tuple:
    """(seams with classes, triage numbers). One call to the faster kernel when installed, else extract + the twin's triage."""

    uv = sa.layers[sa.active][1]
    fast = uvseams.FAST.get("triage")
    if fast is not None:
        job = fast(sa.points, sa.indptr, sa.flat, uv, f_target, allowance, sa.material, settings.repeat, settings.texture_px, uvseams.UV_TOL_PX, settings.cut_here)
        if job is not None:
            job._thread.join()
            seams = job.result()
            tri = {k: seams[k] for k in ("hard_islands", "hard_cost", "left_islands", "left_cost", "spent", "budget", "edge_est")}
            return seams, tri
    seams = uvseams.extract(sa.points, sa.indptr, sa.flat, uv, sa.material, settings.repeat, settings.texture_px, uvseams.UV_TOL_PX, settings.cut_here)
    tri = uvseams.triage(seams, sa.points, sa.indptr, sa.flat, f_target, allowance)
    return seams, tri


def _kinds(seams: dict) -> np.ndarray:
    """Kind of each island: islands joined across every chain that is not Hard."""

    isl = seams["chain_island"].reshape(-1, 2)
    join = seams["class"] != HARD
    root = uvseams._min_labels(seams["n_islands"], isl[join, 0], isl[join, 1])
    return uvseams._first_order(root)[0]


def _risk(seams: dict) -> np.ndarray:
    ptr, gap = seams["run_indptr"], seams["gap_px"]
    worst = np.array([max([0.0] + [float(x) for x in gap[ptr[c]:ptr[c + 1]]]) for c in range(len(ptr) - 1)])
    return seams["chain_length"] * worst


def _reconcile(seams: dict, edge_est: float, spent: float, budget: float, rep: dict) -> float:
    """Hard chains whose two sides ended up in one kind (they share an island pair with a non-Hard chain, or are a
    self-seam, or the kinds meet by another route) cannot be borders. The pair's other chains join them while the Hard
    estimate still fits the budget, else the Hard chain turns Soft. Both outcomes are counted, never silent."""

    cls = seams["class"]
    isl = seams["chain_island"].reshape(-1, 2)
    mats = seams["chain_material"].reshape(-1, 2)
    length = seams["chain_length"]
    fixed = set()
    kept = demoted = 0
    while True:
        kind = _kinds(seams)
        bad = [c for c in range(len(cls)) if cls[c] == HARD and mats[c, 0] == mats[c, 1] and kind[isl[c, 0]] == kind[isl[c, 1]] and c not in fixed]
        if not bad:
            break
        for c in bad:
            if cls[c] != HARD:
                continue
            a, b = isl[c]
            others = [d for d in range(len(cls)) if d != c and cls[d] != HARD and d not in fixed and {int(isl[d, 0]), int(isl[d, 1])} == {int(a), int(b)}] if a != b else []
            cost = sum(length[d] for d in others) / edge_est if edge_est > 0 else math.inf
            if others and spent + cost <= budget:
                for d in others:
                    cls[d] = HARD
                spent += cost
                kept += 1
            else:
                cls[c] = SOFT
                fixed.add(c)
                demoted += 1
    rep["hard_conflicts_kept"], rep["hard_conflicts_soft"] = kept, demoted
    seams["class"] = cls
    uvseams.regions(seams)
    return spent


def make_plan(sa: Source, f_target: float, settings: Settings) -> Plan:
    allowance = settings.allowance if settings.allowance is not None else DEFAULT_ALLOWANCE  # "any" splits without limit; Hard still spends its default share
    key = (sa.key, round(float(f_target), 3), settings.allowance, settings.repeat, settings.cut_here, settings.texture_px)
    hit = _PLANS.get(key)
    if hit is not None:
        _LATEST[sa.key] = hit
        return hit
    seams, tri = _triaged(sa, f_target, allowance, settings)
    rep = dict(chains=int(len(seams["class"])), hard=int(np.sum(seams["class"] == HARD)), soft=int(np.sum(seams["class"] == SOFT)),
               hard_islands=int(len(tri["hard_islands"])), islands_not_covered=int(len(tri["left_islands"])))
    edge_est = float(tri["edge_est"])
    spent = _reconcile(seams, edge_est, float(tri["spent"]), float(tri["budget"]), rep)
    messages = []
    if rep["hard_conflicts_kept"] or rep["hard_conflicts_soft"]:
        n = rep["hard_conflicts_kept"] + rep["hard_conflicts_soft"]
        done = "those seams were kept as borders too" if rep["hard_conflicts_kept"] and not rep["hard_conflicts_soft"] else "they are fitted softly"
        messages.append(f"{n} seams marked to stay borders share islands with other seams; {done}.")
    if rep["islands_not_covered"]:
        messages.append(f"{rep['islands_not_covered']} source islands are smaller than one face and can't keep their own UVs. Bake those spots keeps their look.")
    kind_of_island = _kinds(seams)
    kind = kind_of_island[seams["island"]] if len(seams["island"]) else np.zeros(0, np.int64)
    codes = chart_codes(sa.material, kind)
    if codes is None:  # more kinds than Blender can number: turn the least damaging Hard chains Soft until it fits
        isl = seams["chain_island"].reshape(-1, 2)
        mats = seams["chain_material"].reshape(-1, 2)
        cand = [c for c in np.argsort(_risk(seams), kind="stable") if seams["class"][c] == HARD and mats[c, 0] == mats[c, 1]]
        orig = seams["class"].copy()
        lo, hi = 0, len(cand)
        while lo < hi:  # fewest chains turned Soft that makes the codes fit
            mid = (lo + hi) // 2
            seams["class"] = orig.copy()
            seams["class"][cand[:mid]] = SOFT
            k = _kinds(seams)[seams["island"]]
            if chart_codes(sa.material, k) is None:
                lo = mid + 1
            else:
                hi = mid
        seams["class"] = orig
        seams["class"][cand[:lo]] = SOFT
        uvseams.regions(seams)
        kind = _kinds(seams)[seams["island"]]
        codes = chart_codes(sa.material, kind)
        rep["too_many_seams"] = lo
        messages.append(f"Too many seams to keep as borders ({lo}); the rest are fitted softly.")
    rep["hard_spent_faces"] = round(float(spent), 1)
    plan = Plan(seams, kind, codes, float(spent), float(f_target), edge_est, rep, messages)
    _PLANS[key] = plan
    _LATEST[sa.key] = plan
    while len(_PLANS) > 4:
        _PLANS.popitem(last=False)
    return plan


def target_faces(gs, source_faces: int, target: int | None = None) -> float:
    """Faces the search is expected to land on (SPEC 4.2): the asked-for count, else a share of the source's."""

    if target:
        return float(target)
    return max(12.0, FALLBACK_TARGET_SHARE * source_faces)


def kinds_for_remesh(source, src, gs, target: int | None = None):
    """Today's `(code per face, material per code, island per code)` tuple for the search, or None (no UVs, or too many).
    The kinds are the connected source faces across every non-Hard edge (SPEC 4.6). The plan is kept for `fit_stage`."""

    sa = source_of(source, src)
    if sa is None:
        return None
    s = settings_of(gs)
    plan = make_plan(sa, target_faces(gs, len(sa.indptr) - 1, target), s)
    return plan.codes


def plan_for(source, src, gs, f_target: float | None = None, target: int | None = None):
    """(Source, Plan) the search used. `f_target` None = the same estimate `kinds_for_remesh` would use."""

    sa = source_of(source, src)
    if sa is None:
        return None, None
    s = settings_of(gs)
    if not f_target and not target and sa.key in _LATEST:
        return sa, _LATEST[sa.key]  # the plan the search ran with
    f = f_target if f_target else target_faces(gs, len(sa.indptr) - 1, target)
    return sa, make_plan(sa, f, s)


# ---------------------------------------------------------------- after the search: snap, split, per-side transfer


@dataclass
class Candidate:
    """A finished fit: the new mesh, a UV per corner for every source layer, and the face each new face came from."""

    points: np.ndarray  # (n, 3)
    indptr: np.ndarray
    flat: np.ndarray
    origin: np.ndarray  # per face: the result face it was cut from (arange when nothing was split)
    material: np.ndarray  # per face
    layers: list  # [(name, (corners, 2))]
    active: int
    numbers: dict
    messages: list
    key: str = ""
    result_key: str = ""

    @property
    def extra_faces(self) -> int:
        return int(self.numbers.get("extra_faces", 0))


def candidate_hash(points, indptr, flat, material, layers) -> str:
    """Hash of what ends up in a mesh: float32 positions and UVs (as stored), faces, materials, layer names in order."""

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(points, np.float32).tobytes())
    h.update(np.ascontiguousarray(indptr, np.int64).tobytes())
    h.update(np.ascontiguousarray(flat, np.int64).tobytes())
    h.update(np.ascontiguousarray(material, np.int32).tobytes())
    for name, uv in layers:
        h.update(name.encode("utf-8") + b"\0")
        h.update(np.ascontiguousarray(uv, np.float32).tobytes())
    return h.hexdigest()


def result_key(points, indptr, flat) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(points, np.float32).tobytes())
    h.update(np.ascontiguousarray(indptr, np.int64).tobytes())
    h.update(np.ascontiguousarray(flat, np.int64).tobytes())
    return h.hexdigest()


CANDIDATES: OrderedDict = OrderedDict()
_JUDGES: OrderedDict = OrderedDict()


def view_judge(sa: Source, pixels: int = VIEW_PIXELS):
    """The camera check's judge over the source's six pictures (native pictures, the chords PictureJudge), or None
    without the native pictures. Kept per source."""

    key = (sa.key, int(pixels))
    if key in _JUDGES:
        return _JUDGES[key]
    from .chords import PictureJudge
    from .topology import _turbo as turbo

    if turbo is None:
        return None

    pts = np.ascontiguousarray(sa.points, np.float64).reshape(-1, 3)
    ip, fl = sa.indptr, sa.flat
    tris = [(int(fl[a]), int(fl[k]), int(fl[k + 1])) for a, b in zip(ip[:-1], ip[1:]) for k in range(a + 1, b - 1)]
    if not len(tris) or not len(pts):
        return None
    low, high = pts.min(0), pts.max(0)
    center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
    pictures = turbo.pictures(pts, tris, center, half, int(pixels))
    judge = None if pictures is None else PictureJudge(pictures, center, half, int(pixels), VIEW_TURNED_DEG)
    _JUDGES[key] = judge
    while len(_JUDGES) > 2:
        _JUDGES.popitem(last=False)
    return judge


def _face_wrong(judge, P, ip, fl, f) -> float:
    s, e = int(ip[f]), int(ip[f + 1])
    c = [P[int(v)].tolist() for v in fl[s:e]]
    if len(c) == 4:
        return judge.wrong(c)
    return sum(judge.wrong([c[0], c[k], c[k + 1], c[k + 1]]) for k in range(1, len(c) - 1))


def view_limit(judge, P0, P1, ip, fl, edge_move, detail: float, out: dict):
    """Generator: per-vertex allowed move for the snap from the camera check. `P1` is a snap with the plain limit
    `edge_move`; each moved vertex is priced by the wrong pixels its faces gain there (shared out between the moved
    vertices of a face), and moves are kept cheapest first while what they add stays within Most Detail Loss
    (`detail` percent) of the pixels the touched faces show (their area's share of the pictures). Only the change is
    priced: the remesh's own loss was judged by Auto's camera check already, and these pictures are coarser. A move
    that costs nothing or wins pixels back is always kept; the rest get 0 (held; split takes over). Fills `out` with
    limit, held, room, spent."""

    P0 = np.asarray(P0, np.float64).reshape(-1, 3)
    P1 = np.asarray(P1, np.float64).reshape(-1, 3)
    ip, fl = np.asarray(ip, np.int64), np.asarray(fl, np.int64)
    nf = len(ip) - 1
    moved = np.any(P1 != P0, axis=1)
    face_of = np.repeat(np.arange(nf), np.diff(ip))
    touched = np.zeros(nf, bool)
    touched[face_of[moved[fl]]] = True
    nxt = np.arange(len(fl)) + 1
    nxt[ip[1:] - 1] = ip[:-1]
    area = np.linalg.norm(np.add.reduceat(np.cross(P0[fl], P0[fl[nxt]]), ip[:-1], axis=0), axis=1) if nf else np.zeros(0)  # Newell
    delta = {}
    for count, f in enumerate(np.flatnonzero(touched).tolist()):
        delta[f] = _face_wrong(judge, P1, ip, fl, f) - _face_wrong(judge, P0, ip, fl, f)
        if count % 32 == 31:
            yield
    cost = np.zeros(len(P0))
    for f, d in delta.items():
        vs = [int(v) for v in fl[ip[f]:ip[f + 1]] if moved[v]]
        for v in vs:
            cost[v] += d / len(vs)
    shown = judge.total * float(area[touched].sum()) / max(float(area.sum()), 1e-30)
    room = detail / 100.0 * shown
    limit = np.asarray(edge_move, np.float64).copy()
    spent, held = 0.0, 0
    for v in sorted(np.flatnonzero(moved).tolist(), key=lambda v: (cost[v], v)):
        if cost[v] <= 0.0 or spent + cost[v] <= room:
            spent += max(cost[v], 0.0)
        else:
            limit[v] = 0.0
            held += 1
    out.update(limit=limit, held=held, room=room, spent=spent)


def remember(candidate: Candidate) -> None:
    CANDIDATES[candidate.result_key] = candidate
    while len(CANDIDATES) > CACHED_CANDIDATES:
        CANDIDATES.popitem(last=False)


class FitStage:
    """Snap, split leftovers and per-side transfer as one sliced stage. `step()` does about 4 ms of work; `done` turns
    True at the end and `candidate` holds the fit. `stop()` ends early (`candidate` None). Again: pass the previous
    candidate's mesh as the result and its `extra_faces` as `extra_so_far`."""

    def __init__(self, sa: Source, plan: Plan, points, indptr, flat, material, settings: Settings, extra_so_far: int = 0, only_transfer: bool = False,
                 use_fast: bool = True, view_limit=None):
        self.sa, self.plan, self.settings = sa, plan, settings
        self.points = np.ascontiguousarray(points, np.float64).reshape(-1, 3)
        self.indptr = np.asarray(indptr, np.int64)
        self.flat = np.asarray(flat, np.int64)
        self.material = None if material is None else np.asarray(material, np.int64)
        self.extra_so_far, self.only_transfer, self.use_fast = extra_so_far, only_transfer, use_fast
        self.view_limit = view_limit  # optional per-vertex allowed move from a camera check; the share of an edge otherwise
        self.done, self.candidate, self.numbers = False, None, {"estimate": True}
        self._stopped = False
        self._now = None
        self._steps = self._run()

    def stop(self) -> None:
        self._stopped = True
        if self._now is not None:
            self._now.stop()

    def step(self) -> None:
        if self.done:
            return
        try:
            next(self._steps)
        except StopIteration:
            self.done = True

    def _lookup(self, uv, P, ip, fl):
        """Generator: the per-side lookup of the active layer `uv` on result (P, ip, fl), sliced; returns the lookup dict,
        or None when stopped."""

        sa, s = self.sa, self.settings
        tr = uvseams.Transfer(sa.points, sa.indptr, sa.flat, uv, P, ip, fl, sa.material, s.repeat, s.texture_px, uvseams.UV_TOL_PX, self.plan.seams["class"], False, self.use_fast)
        if tr._job is not None:
            self._now = tr
        while not tr.done:
            tr.step()
            yield
            if self._stopped:
                tr.stop()
                return None
        self._now = None
        return tr.result

    def _drive(self, stage):
        self._now = stage
        while not stage.done:
            stage.step()
            yield
        self._now = None

    def _run(self):
        sa, plan, s = self.sa, self.plan, self.settings
        seams = plan.seams
        nf = len(self.indptr) - 1
        numbers = dict(estimate=False, faces=nf, extra_faces=0, moved=0, held=0, faces_split=0, triangles_left=0, faces_left=0, faces_for_baking=0,
                       soft_chains=int(np.sum(seams["class"] == SOFT)))
        P, ip, fl = self.points.ravel(), self.indptr, self.flat
        origin = np.arange(nf, dtype=np.int64)
        xyz = uvsnap.chain_xyz(seams, sa.points)
        vchain = vparam = ()
        soft = numbers["soft_chains"] > 0 and not self.only_transfer
        if soft and s.snap:
            move = self.view_limit
            snap = None
            if move is None:
                move = s.move_share * uvsnap.vertex_edge(P, ip, fl)
                judge = None
                if s.view_check and self.use_fast:  # the pictures are native only; drawn on a thread while the first pass runs
                    got = {}
                    worker = threading.Thread(target=lambda: got.update(judge=view_judge(sa, s.view_pixels)), name="CurioMeshUVView", daemon=True)
                    worker.start()
                    first = uvsnap.Snap(P, ip, fl, seams, xyz, max_move=move, use_fast=self.use_fast)
                    yield from self._drive(first)
                    while worker.is_alive():
                        yield
                    if self._stopped:
                        return
                    judge = got.get("judge")
                    snap = None if judge is not None else first
                if judge is not None:  # the first pass had the plain limit; priced by the camera check, then the real pass
                    view = {}
                    yield from view_limit(judge, P, first.result["points"], ip, fl, move, s.detail, view)
                    numbers.update(view_held=int(view["held"]), view_room_px=round(float(view["room"]), 1), view_spent_px=round(float(view["spent"]), 1))
                    if view["held"]:
                        move = view["limit"]
                    else:
                        snap = first
            if snap is None:
                snap = uvsnap.Snap(P, ip, fl, seams, xyz, max_move=move, use_fast=self.use_fast)
                yield from self._drive(snap)
            if self._stopped:
                return
            r = snap.result
            P, vchain, vparam = np.asarray(r["points"], np.float64), r["vchain"], r["vparam"]
            numbers.update(moved=int(r["moved"]), held=int(np.sum(r["held"])))
        budget = -1.0
        if soft and s.split and s.allowance is not None:
            off = nf - plan.hard_spent - self.extra_so_far
            budget = max(0.0, s.allowance * off - plan.hard_spent - self.extra_so_far)
        # per-side transfer: the active layer's lookup first (folds fixed after the split's seam UVs are laid over it),
        # the other layers through the same lookup. A face whose fan triangles want opposite windings (orient 0: folded
        # in 3D, or across a crease with one half leaning into a wall) shows the texture mirrored on part of it under any
        # one winding; it is cut into its fan triangles, which can each wind their own way, and read again. So is a face
        # the fold fix leaves folded, and a face with a zero-area fan triangle (closed by `close_flat`). The cut faces come
        # out of the extra-face allowance: when they do not fit beside the split, the split runs again with less.
        name, uv = sa.layers[sa.active]
        P0, ip0, fl0 = P, ip, fl
        cut_seams, cut_xyz = seams, xyz
        if soft and s.split and self.use_fast:  # part switches: cut faces spanning two unconnected source pieces
            from .topology import _turbo as turbo
            if turbo is not None:
                tree = turbo.face_tree(sa.points, FlatFaces(sa.indptr, sa.flat))
                got = {}
                yield from uvsnap.part_switches_steps(seams, sa.points, sa.indptr, sa.flat, uv, P, ip, fl, tree.nearest, got)
                if self._stopped:
                    return
                cut_seams = got["seams"]
                cut_xyz = uvsnap.chain_xyz(cut_seams, sa.points)
                numbers.update(part_switches=int(cut_seams.get("part_switches", 0)))
        reserve = 0
        while True:
            P, ip, fl, origin = P0, ip0, fl0, np.arange(nf, dtype=np.int64)
            if soft and s.split:
                give = budget if budget < 0 else max(0.0, budget - reserve)
                cut = uvsnap.Split(P, ip, fl, cut_seams, cut_xyz, vchain, vparam, (), give, use_fast=self.use_fast)
                yield from self._drive(cut)
                if self._stopped:
                    return
                r = cut.result
                P, ip, fl, origin = np.asarray(r["points"], np.float64), np.asarray(r["indptr"], np.int64), np.asarray(r["flat"], np.int64), np.asarray(r["origin"], np.int64)
                numbers.update(extra_faces=int(r["extra_faces"]), faces_split=int(r["faces_split"]), triangles_left=int(r["triangles_left"]),
                               faces_left=int(r["faces_left"]), faces_for_baking=int(r["faces_cannot"]) + int(r["faces_bake"]), split_done=bool(r["done"]),
                               split_budget=give)
            look = yield from self._lookup(uv, P, ip, fl)
            if look is None:
                return
            ready = {}
            yield from _layer_steps(sa, uv, P, ip, fl, look, s.texture_px, self.use_fast, ready)
            if self._stopped:
                return
            halve = halve_mask(P, ip, fl, look["orient"]) | still_folded(ip, ready["uv"], ready["expected"])
            cost = int((np.diff(ip)[halve] - 3).sum())
            if soft and s.split and budget >= 0 and numbers["extra_faces"] + cost > budget and reserve < cost:
                reserve = cost  # room for the cut faces: split again with less
                continue
            break
        if halve.any():
            ready = None
            ip, fl, origin = fan_split(ip, fl, origin, halve)
            numbers.update(extra_faces=numbers["extra_faces"] + cost, faces_halved=int(halve.sum()), halve_reserve=reserve)
            look = yield from self._lookup(uv, P, ip, fl)
            if look is None:
                return
        out = {}
        for i, (nm, luv) in enumerate(sa.layers):
            got = ready if i == sa.active and ready is not None else {}
            if not got:
                yield from _layer_steps(sa, luv, P, ip, fl, look, s.texture_px, self.use_fast, got)
            if self._stopped:
                return
            out[i] = close_flat(P, ip, fl, got["uv"], s.texture_px)
            if i == sa.active:
                f = got["fold"]
                numbers.update(folded_before=int(f["folded_before"]), folded_after=int(f["folded_after"]), fold_stuck=int(f["fold_stuck"]),
                               fold_untangled=int(f.get("fold_untangled", 0)), fold_reread=int(f.get("fold_reread", 0)))
        layers = [(nm, out[i]) for i, (nm, _u) in enumerate(sa.layers)]
        mat = (self.material if self.material is not None else np.zeros(nf, np.int64))[origin]
        pts = P.reshape(-1, 3)
        numbers["faces"] = len(ip) - 1
        msgs = list(plan.messages)
        if numbers["faces_left"]:
            msgs.append(f"{numbers['faces_left']} faces on seams not fitted: the extra-face allowance is used up. Raise it, or bake those spots.")
        c = Candidate(pts, ip, fl, origin, mat, layers, sa.active, numbers, msgs)
        c.key = candidate_hash(pts, ip, fl, mat, layers)
        c.result_key = result_key(self.points, self.indptr, self.flat)
        self.candidate, self.numbers = c, numbers
        if not self.only_transfer:
            remember(c)


def fans_with_area(points, indptr, flat) -> tuple:
    """Per result face: how many of its fan triangles have area (twice-area above uvseams.DEGENERATE_SHARE of their
    summed squared edges), and how many fan triangles it has."""

    P = np.asarray(points, np.float64).reshape(-1, 3)
    ip, fl = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
    sizes = np.diff(ip)
    fan = np.maximum(sizes - 2, 0)
    tri_face = np.repeat(np.arange(len(sizes)), fan)
    k = np.arange(len(tri_face)) - np.repeat(np.cumsum(fan) - fan, fan) + 1
    s = ip[tri_face]
    a, b, c = P[fl[s]], P[fl[s + k]], P[fl[s + k + 1]]
    two_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    edges2 = ((b - a) ** 2).sum(1) + ((c - b) ** 2).sum(1) + ((a - c) ** 2).sum(1)
    return np.bincount(tri_face, two_area > uvseams.DEGENERATE_SHARE * edges2, len(sizes)).astype(np.int64), fan


def halve_mask(points, indptr, flat, orient) -> np.ndarray:
    """Per result face of 4 or more corners with area: True when the lookup gave it no common winding (orient 0: its fan
    triangles want opposite windings), or when some of its fan triangles have no area (a corner on the line between two
    others: that sliver's winding is rounding, so it is cut off and closed by `close_flat`)."""

    good, fan = fans_with_area(points, indptr, flat)
    return (fan > 1) & (good > 0) & ((np.asarray(orient) == 0) | (good < fan))


def still_folded(indptr, uv, expected) -> np.ndarray:
    """Per result face of 4 or more corners: True when a fan triangle winds against its expected winding (or has no UV
    area) after the fold fix."""

    ip = np.asarray(indptr, np.int64)
    q = np.asarray(uv, np.float64).reshape(-1, 2)
    x = np.asarray(expected, np.float64)
    sizes = np.diff(ip)
    fan = np.maximum(sizes - 2, 0)
    tri_face = np.repeat(np.arange(len(sizes)), fan)
    k = np.arange(len(tri_face)) - np.repeat(np.cumsum(fan) - fan, fan) + 1
    s = ip[tri_face]
    a, b = q[s + k] - q[s], q[s + k + 1] - q[s]
    bad = (a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]) * x[tri_face] <= 0.0
    return (fan > 1) & (x != 0.0) & (np.bincount(tri_face, bad, len(sizes)) > 0)


SLIVER_UV_PX2 = 1e-3  # a fan triangle with twice its UV area below this many texels squared shows nothing (and its winding is rounding)


def close_flat(points, indptr, flat, uv, texture_px: float = 1024.0) -> np.ndarray:
    """`uv` (2 per corner) with the slivers closed. Every corner of a face with no 3D area goes to the mean of its UVs; an
    end fan triangle (the first or last of its face) with next to no UV area (SLIVER_UV_PX2) gets its own corner onto
    the nearer of its two neighbours in the face. Such a triangle shows no texture, but a single-precision rounding of
    its UVs, or its rounding-noise normal, would give it a winding against the source's."""

    ip = np.asarray(indptr, np.int64)
    out = np.asarray(uv, np.float64).reshape(-1, 2).copy()
    good, fan = fans_with_area(points, indptr, flat)
    for f in np.flatnonzero(good == 0):
        out[ip[f]:ip[f + 1]] = out[ip[f]:ip[f + 1]].mean(0)
    tiny = SLIVER_UV_PX2 / (float(texture_px) ** 2)
    s, e = ip[:-1], ip[1:]
    for own, a, c in ((s + 1, s, s + 2), (e - 1, s, e - 2)):  # first fan triangle (0, 1, 2), last (0, n-2, n-1)
        u0, u1, u2 = out[a], out[own], out[c]
        area = np.abs((u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1]) - (u2[:, 0] - u0[:, 0]) * (u1[:, 1] - u0[:, 1]))
        for f in np.flatnonzero((fan > 0) & (good > 0) & (area <= tiny)):
            near = a[f] if np.sum((out[a[f]] - out[own[f]]) ** 2) <= np.sum((out[c[f]] - out[own[f]]) ** 2) else c[f]
            out[own[f]] = out[near]
    return out


def fan_split(indptr, flat, origin, mask) -> tuple:
    """Cuts every face with mask True into its fan triangles (0, i, i+1); `origin` (source face per result face) follows."""

    ip, fl, org, mask = np.asarray(indptr, np.int64), np.asarray(flat, np.int64), np.asarray(origin, np.int64), np.asarray(mask, bool)
    sizes = np.diff(ip)
    pieces = np.where(mask, sizes - 2, 1)
    old = np.repeat(np.arange(len(sizes)), pieces)  # old face of each new face
    j = np.arange(len(old)) - np.repeat(np.cumsum(pieces) - pieces, pieces)  # which fan triangle
    cut = mask[old]
    new_sizes = np.where(cut, 3, sizes[old])
    new_ip = np.concatenate([[0], np.cumsum(new_sizes)]).astype(np.int64)
    f_of = np.repeat(old, new_sizes)
    r = np.arange(int(new_ip[-1])) - np.repeat(new_ip[:-1], new_sizes)
    corner = ip[f_of] + np.where(cut[np.repeat(np.arange(len(old)), new_sizes)], np.where(r == 0, 0, np.repeat(j, new_sizes) + r), r)
    return new_ip, fl[corner], org[old]


def _layer_steps(sa: Source, luv, P, ip, fl, look: dict, texture_px: float, use_fast: bool, got: dict):
    """Generator: one UV layer through the lookup with the fold fix (`uvseams.layer`), polling the faster job when there is one so no slice blocks. Fills got[uv], got[expected], got[fold]."""

    fast = uvseams.FAST.get("layer") if use_fast else None
    job = fast(sa.indptr, sa.flat, luv, P, ip, fl, look, texture_px, True, sa.points) if fast is not None else None
    if job is not None:
        while not job.done():
            yield
        u, e, fold = job.result()
    else:
        u, e, fold = uvseams.layer(sa.indptr, sa.flat, luv, P, ip, fl, look, texture_px, True, False, sa.points)
        yield
    got.update(uv=u, expected=e, fold=fold)


def fit_stage(source, src, res, gs, only_transfer: bool = False, use_fast: bool = True):
    """The stage for a result `res` (guided's dict: co, faces, mat), or None when the source has no UVs or Seam Fit is not Light."""

    s = settings_of(gs)
    if s.mode != "light":
        return None
    sa, plan = plan_for(source, src, gs, f_target=None)
    if sa is None:
        return None
    faces = FlatFaces.of(res["faces"])
    mat = res.get("mat")
    return FitStage(sa, plan, res["co"], faces.indptr, faces.flat, mat, s, only_transfer=only_transfer, use_fast=use_fast)


def run(stage: FitStage):
    """Run a stage to the end (tests, the Apply path without a finished preview)."""

    while not stage.done:
        stage.step()
    return stage.candidate


def again(candidate: Candidate, sa: Source, plan: Plan, settings: Settings, use_fast: bool = True) -> FitStage:
    """Another pass from a finished fit: its mesh is the result, its extra faces already count against the allowance."""

    return FitStage(sa, plan, candidate.points, candidate.indptr, candidate.flat, candidate.material, settings, extra_so_far=candidate.extra_faces, use_fast=use_fast)
