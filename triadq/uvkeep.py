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

from . import uvscore, uvseams, uvsnap
from .types import FlatFaces
from .uvs import chart_codes

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

# Optional faster callables, set by the one adapter (uv_fast.py); the NumPy / Blender code runs when a name is missing.
#   face_tree(points, FlatFaces) -> object with nearest(queries) -> (closest points, polygon, distance), or None
#   pictures(points, triangles, center, half, pixels) -> the six pictures of `autoretopo.pictures`, or None
#   check(...) -> a job (done(), result() = (numbers, straddling faces, flipped faces), stop()) for the UV Check, or None
FAST: dict = {}


def set_fast(**callables) -> None:
    """Install (or, with None, remove) faster callables by name."""

    for key, fn in callables.items():
        if fn is None:
            FAST.pop(key, None)
        else:
            FAST[key] = fn


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
    switch_by: str = "part"  # the split cuts the Soft chains plus where the nearest unconnected "part" changes, or every "side" switch (costs more faces per mended face under a small allowance, R5keep)
    reread: bool = True  # faces whose corners all lie within one side's owner ball read that side
    switch_snap: bool = True  # corners just over a side line move onto it (zero faces) before any split
    ball_snap: bool = True  # corners of faces still across a seam move onto the side the face's other corners hold (zero faces), before and after the split


def settings_of(gs) -> Settings:
    """Settings from the guided settings `gs` plus OVERRIDE. Until the UVs row exists: Keep UV Seams on = Exact, off = Off."""

    ov = OVERRIDE
    mode = ov.get("mode") or ("exact" if getattr(gs, "keep_uvs", False) else "off")
    s = Settings(mode=mode)
    detail = getattr(gs, "auto_detail", None)
    if isinstance(detail, (int, float)) and detail > 0:
        s.detail = float(detail)
    for key in ("allowance", "split", "snap", "move_share", "texture_px", "repeat", "cut_here", "view_check", "detail", "view_pixels", "switch_by", "reread", "switch_snap", "ball_snap"):
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
    """Kind of each island: islands joined across every chain that is not Hard, then those groups merged wherever no
    Hard chain runs between them (a greedy colouring of the groups by their Hard neighbours). Only a Hard chain keeps
    the search apart; loose pieces and islands with no border between them stay one kind, as with Seam Fit off (kinds
    per loose piece cost the search 8 % more faces on spinning_wheel_01 with no Hard chain at all, R5keep)."""

    isl = seams["chain_island"].reshape(-1, 2)
    hard = seams["class"] == HARD
    root = uvseams._min_labels(seams["n_islands"], isl[~hard, 0], isl[~hard, 1])
    group, n = uvseams._first_order(root)
    nbr = [set() for _ in range(int(n))]
    for a, b in zip(group[isl[hard, 0]].tolist(), group[isl[hard, 1]].tolist()):
        if a != b:
            nbr[a].add(b)
            nbr[b].add(a)
    colour = np.full(int(n), -1, np.int64)
    for g in range(int(n)):
        taken = {int(colour[h]) for h in nbr[g] if colour[h] >= 0}
        c = 0
        while c in taken:
            c += 1
        colour[g] = c
    return colour[group]


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


def _tris(sa: Source) -> list:
    ip, fl = sa.indptr, sa.flat
    return [(int(fl[a]), int(fl[k]), int(fl[k + 1])) for a, b in zip(ip[:-1], ip[1:]) for k in range(a + 1, b - 1)]


def _frame(sa: Source) -> tuple:
    pts = np.ascontiguousarray(sa.points, np.float64).reshape(-1, 3)
    low, high = pts.min(0), pts.max(0)
    return pts, (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)


def view_judge(sa: Source, pixels: int = VIEW_PIXELS):
    """The camera check's judge over the source's six pictures (the chords PictureJudge) in one call, or None when no
    faster `pictures` callable is installed (then `view_judge_steps` draws them in slices). Kept per source."""

    key = (sa.key, int(pixels))
    if key in _JUDGES:
        return _JUDGES[key]
    fast = FAST.get("pictures")
    if fast is None or not len(sa.points):
        return None
    tris = _tris(sa)
    if not tris:
        return None
    pts, center, half = _frame(sa)
    return _keep_judge(key, fast(pts, tris, center, half, int(pixels)), center, half, pixels)


def _keep_judge(key, pictures, center, half, pixels):
    from .chords import PictureJudge

    judge = None if pictures is None else PictureJudge(pictures, center, half, int(pixels), VIEW_TURNED_DEG)
    _JUDGES[key] = judge
    while len(_JUDGES) > 2:
        _JUDGES.popitem(last=False)
    return judge


def view_judge_steps(sa: Source, pixels: int, out: dict, use_fast: bool = True):
    """Generator: out["judge"] = the camera check's judge. With a faster `pictures` callable it is drawn on a worker
    thread (one call, the caller keeps slicing); without it the add-on's own ray pictures (`autoretopo.pictures`, a
    generator) are drawn a slice at a time on the caller's thread. None outside Blender."""

    key = (sa.key, int(pixels))
    if key in _JUDGES:
        out["judge"] = _JUDGES[key]
        return
    if use_fast and FAST.get("pictures") is not None:
        worker = threading.Thread(target=lambda: out.update(judge=view_judge(sa, pixels)), name="CurioMeshUVView", daemon=True)
        worker.start()
        while worker.is_alive():
            yield
        return
    out["judge"] = None
    try:
        from .. import autoretopo  # the add-on's six pictures by rays; read only, never edited
    except ImportError:  # outside Blender
        return
    if not len(sa.points):
        return
    pts, center, half = _frame(sa)
    polys = [sa.flat[a:b].tolist() for a, b in zip(sa.indptr[:-1].tolist(), sa.indptr[1:].tolist())]
    yield
    pictures = yield from autoretopo.pictures(pts, polys, center, half, int(pixels))
    out["judge"] = _keep_judge(key, pictures, center, half, pixels)


def _together(*gens):
    """Generator: steps every generator in turn until all are exhausted."""

    live = list(gens)
    while live:
        for g in list(live):
            try:
                next(g)
            except StopIteration:
                live.remove(g)
        yield


class _PolyTree:
    """Closest points on polygons for arrays of queries, the faster face tree's form: (closest points, polygon,
    distance). Blender's BVHTree (polygons cut the same way, single precision) when there is one; else a NumPy scan
    over the fan triangles (small meshes, tests)."""

    def __init__(self, points, faces):
        P = np.asarray(points, np.float64).reshape(-1, 3)
        ip, fl = np.asarray(faces.indptr, np.int64), np.asarray(faces.flat, np.int64)
        try:
            from mathutils.bvhtree import BVHTree
        except ImportError:
            BVHTree = None
        if BVHTree is not None:
            polys = [fl[a:b].tolist() for a, b in zip(ip[:-1].tolist(), ip[1:].tolist())]
            self._bvh = BVHTree.FromPolygons(P.tolist(), polys, all_triangles=False, epsilon=0.0)
            return
        self._bvh = None
        tri, owner = [], []
        for f in range(len(ip) - 1):
            for k in range(int(ip[f]) + 1, int(ip[f + 1]) - 1):
                tri.append((fl[ip[f]], fl[k], fl[k + 1]))
                owner.append(f)
        t = np.array(tri, np.int64).reshape(-1, 3)
        self.a, self.b, self.c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
        self.owner = np.array(owner, np.int64)
        self.grid = uvscore._Grid(self.a, self.b, self.c) if len(t) else None

    def nearest(self, queries):
        q = np.asarray(queries, np.float64).reshape(-1, 3)
        at, face, away = np.full((len(q), 3), np.nan), np.full(len(q), -1, np.int64), np.full(len(q), np.inf)
        if self._bvh is not None:
            find = self._bvh.find_nearest
            for i, x in enumerate(q.tolist()):
                co, _n, f, d = find(x)
                if co is not None:
                    at[i], face[i], away[i] = co, f, d
            return at, face, away
        if self.grid is None or not len(q):
            return at, face, away
        d = self.grid.nearest(q)  # exact nearest distance, then the triangles at that distance (lowest polygon on ties)
        qi, ti, bary, dist = self.grid.ball(q, d * (1.0 + 1e-9) + 1e-300)
        order = np.lexsort((self.owner[ti], dist, qi))
        qi, ti, bary, dist = qi[order], ti[order], bary[order], dist[order]
        first = np.r_[True, qi[1:] != qi[:-1]]
        qi, ti, bary = qi[first], ti[first], bary[first]
        at[qi] = bary[:, :1] * self.a[ti] + bary[:, 1:2] * self.b[ti] + bary[:, 2:] * self.c[ti]
        face[qi], away[qi] = self.owner[ti], dist[first]
        return at, face, away


def face_tree(points, faces, use_fast: bool = True):
    """The closest-point tree over polygons (`faces` a FlatFaces): the faster one when installed, else `_PolyTree`."""

    fast = FAST.get("face_tree") if use_fast else None
    tree = fast(points, faces) if fast is not None else None
    return tree if tree is not None else _PolyTree(points, faces)


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


class SeamCheck:
    """The UV Check's straddling and flipped faces of a result against one source layer, by the Check's own rules
    (`uvscore._check`, SPEC 9.1: owner sets per corner; the winding at each fan triangle's centre against the source
    triangles within its reach) and the same numbers, for the NumPy path of the fit. The source's half (fan, islands,
    triangle grid) is built once and kept; every query runs a few hundred points a slice. Only what the fit reads is
    measured: no drift, stretch or overlap."""

    CHUNK = 32  # queries per slice (256 took 0.3-0.4 s on the 61k-face barrier with big result faces)

    def __init__(self, sa: Source, src_uv, texture_px: float):
        self.sa, self.tex = sa, float(texture_px) if texture_px and texture_px > 0 else uvscore.TEXTURE_PX
        self.src = uvscore.mesh(sa.points, sa.indptr, sa.flat, src_uv, sa.material)
        self._layer = src_uv  # held, so the cache key (its id) stays this layer's
        self.ready = False

    def _build(self):
        src = self.src
        sc, sf = uvscore._fan(src)
        yield
        self.region = uvscore._islands(src, sc, sf, self.tex)[0]
        yield
        sq = src.points[src.flat[sc]]
        self.su = src.uv[sc]
        self.grid = uvscore._Grid(sq[:, 0], sq[:, 1], sq[:, 2])
        yield
        diag = float(np.linalg.norm(np.ptp(src.points, axis=0)))
        edge = np.sort(np.linalg.norm(sq[:, 1] - sq[:, 0], axis=1))
        n = len(edge)
        median = float(edge[n // 2] if n % 2 else (edge[n // 2 - 1] + edge[n // 2]) / 2) if n else 0.0
        self.eps = max(uvscore.EPS_DIAG * diag, uvscore.EPS_EDGE * median)
        self.s2s, _ = uvscore._tri_stats(sq, self.su)
        self.sn = np.cross(sq[:, 1] - sq[:, 0], sq[:, 2] - sq[:, 0])
        self.ready = True

    def _nearest(self, q, stopped):
        out = np.empty(len(q))
        for a in range(0, len(q), self.CHUNK):
            out[a:a + self.CHUNK] = self.grid.nearest(q[a:a + self.CHUNK])
            yield
            if stopped():
                return None
        return out

    def _ball(self, q, r, stopped):
        parts = []
        for a in range(0, len(q), self.CHUNK):
            qi, ti, bary, _d = self.grid.ball(q[a:a + self.CHUNK], r[a:a + self.CHUNK])
            parts.append((qi + a, ti, bary))
            yield
            if stopped():
                return None
        if not parts:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3))
        return tuple(np.concatenate([x[k] for x in parts]) for k in range(3))

    def steps(self, points, indptr, flat, uv, stopped=lambda: False):
        """Generator: (straddling, flipped) per face of the result (corner UVs `uv`); None when `stopped()`."""

        if not self.ready:
            yield from self._build()
        res = uvscore.mesh(points, indptr, flat, uv)
        nf = len(res.indptr) - 1
        tc, tf = uvscore._fan(res)
        rq, ru = res.points[res.flat[tc]], res.uv[tc]
        a2s, _a3 = uvscore._tri_stats(rq, ru)
        tex = self.tex
        # owner sets per result corner
        cpts = res.points[res.flat]
        d0 = yield from self._nearest(cpts, stopped)
        if d0 is None:
            return None
        got = yield from self._ball(cpts, d0 + self.eps, stopped)
        if got is None:
            return None
        qi, ti, bary = got
        reg = self.region[ti]
        suv = np.einsum("kj,kjc->kc", bary, self.su[ti])
        pairs = np.unique(np.stack([qi, reg], 1), axis=0) if len(qi) else np.zeros((0, 2), np.int64)
        many = np.bincount(pairs[:, 0], minlength=len(cpts)) > 1
        close = np.linalg.norm(suv - res.uv[qi], axis=1) * tex <= uvscore.OWNER_UV_PX
        use = ~many[qi] | close
        pairs = np.unique(np.stack([qi[use], reg[use]], 1), axis=0) if len(qi) else pairs
        corner_face = np.repeat(np.arange(nf), np.diff(res.indptr))
        fr = np.stack([corner_face[pairs[:, 0]], pairs[:, 1]], 1)
        uniq, cnt = np.unique(fr, axis=0, return_counts=True) if len(fr) else (np.zeros((0, 2), np.int64), np.zeros(0, np.int64))
        full = cnt == np.diff(res.indptr)[uniq[:, 0]] if len(uniq) else np.zeros(0, bool)
        ok_face = np.zeros(nf, bool)
        ok_face[uniq[full, 0]] = True
        yield
        # flips: the reach of each fan triangle from its 7 samples, the winding at its centre
        sp = np.einsum("ijk,lj->ilk", rq, uvscore._BARY7).reshape(-1, 3)
        sd0 = yield from self._nearest(sp, stopped)
        if sd0 is None:
            return None
        face_err = sd0.reshape(-1, 7).sum(1) / 7.0
        slack = np.maximum(face_err, uvscore.DRIFT_EDGE_SLACK * np.linalg.norm(rq[:, 1] - rq[:, 0], axis=1))
        centre = np.arange(0, len(sp), 7)
        got = yield from self._ball(sp[centre], sd0[centre] + slack, stopped)
        if got is None:
            return None
        qt, ti, _b = got
        nr = np.cross(rq[:, 1] - rq[:, 0], rq[:, 2] - rq[:, 0])
        sgn = np.sign(self.s2s[ti]) * np.sign((self.sn[ti] * nr[qt]).sum(-1))
        match = (sgn == np.sign(a2s[qt])) | (np.abs(self.s2s[ti]) < uvscore.AREA_MIN)
        matched = np.zeros(len(tc), bool)
        matched[qt[match]] = True
        flipped = np.zeros(nf, bool)
        flipped[tf[~matched & (np.abs(a2s) >= uvscore.AREA_MIN)]] = True
        return ~ok_face, flipped


_CHECKS: OrderedDict = OrderedDict()


def seam_check(sa: Source, src_uv, texture_px: float) -> SeamCheck:
    """The kept `SeamCheck` for this source layer and texture size."""

    key = (sa.key, id(src_uv), float(texture_px))
    hit = _CHECKS.get(key)
    if hit is None:
        hit = _CHECKS[key] = SeamCheck(sa, src_uv, texture_px)
        while len(_CHECKS) > 2:
            _CHECKS.popitem(last=False)
    return hit


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
        self._trees = None  # closest points per source side, built as the re-read needs them
        self._tree = None  # closest points over the whole source
        self._judge = None  # the camera check's judge, once the snap has drawn it
        self._last_flipped = None  # the UV Check's flipped faces from the last `_straddling`
        self._ball_moved = {}  # vertices each ball pass moved
        self._flip_corners = np.zeros(0, np.int64)  # vertices of faces the last check left turned over
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

    def _straddling(self, uv, P, ip, fl, origin=None, tree=None):
        """Generator: per result face True when the UV Check (its owner-set rule) calls it across a seam after a plain
        per-side transfer (with `tree`, the global face tree: after the owner-ball re-read too, as the fit reads it);
        the side switches cut only those. None when stopped."""

        sa, s = self.sa, self.settings
        look = yield from self._lookup(uv, P, ip, fl)
        if look is None:
            return None
        got = {}
        if tree is not None:  # the fit's own counts stay the fit's
            kept = (getattr(self, "numbers_reread", 0), getattr(self, "numbers_reread_undone", 0))
            yield from self._reread(look, tree, P, ip, fl, 0, ())
            yield from self._layer_checked(uv, P, ip, fl, look, got, origin)
            self._before = None
            self.numbers_reread, self.numbers_reread_undone = kept
        else:
            yield from _layer_steps(sa, uv, P, ip, fl, look, s.texture_px, self.use_fast, got)
        res = yield from self._check(P, ip, fl, got["uv"], origin, uv)
        if res is None:
            return None
        self._last_flipped = res[1]
        return res[0]

    def _check(self, P, ip, fl, uv, origin, src_uv):
        """Generator: the UV Check's (straddling, flipped) per face of the mesh (P, ip, fl) with corner UVs `uv`,
        against the source layer `src_uv`; the faster job is polled when installed, else the NumPy Check runs between
        two slices. None when stopped."""

        sa, s = self.sa, self.settings
        mat = self._face_material(ip, origin)
        fast = FAST.get("check") if self.use_fast else None
        job = fast(P, ip, fl, uv, mat, sa.points, sa.indptr, sa.flat, src_uv, sa.material, texture_px=s.texture_px) if fast is not None else None
        if job is not None:
            while not job.done():
                yield
                if self._stopped:
                    job.stop()
                    return None
            res = job.result()
        else:
            res = yield from seam_check(sa, src_uv, s.texture_px).steps(P, ip, fl, uv, lambda: self._stopped)
            if res is None:
                return None
            res = (None,) + res
        return np.asarray(res[1]).astype(bool), np.asarray(res[2]).astype(bool)

    def _tree_of(self, points, faces):
        return face_tree(points, faces, self.use_fast)

    def _source_tree(self):
        """The face tree over the whole source, built once per stage."""

        if self._tree is None:
            self._tree = self._tree_of(self.sa.points, FlatFaces(self.sa.indptr, self.sa.flat))
        return self._tree

    def _reread(self, look, tree, P, ip, fl, n_before, vchain):
        """Generator: a face whose corners all lie within one side's owner ball reads that side (`uvsnap.ball_sides_steps`,
        `uvsnap.side_reread_steps`): the lookup's vote by the face centre can pick a side some corners are far from."""

        if tree is None or not self.settings.reread:
            return
        sa = self.sa
        region = np.asarray(self.plan.seams["region"], np.int64)
        pts = np.asarray(P, np.float64).reshape(-1, 3)
        _at, near_f, d0 = tree.nearest(pts)
        label = region[np.asarray(near_f, np.int64)]
        if self._trees is None:
            self._trees = uvsnap.RegionTrees(sa.points, sa.indptr, sa.flat, region, self._tree_of)
        got = {}
        self._before = {k: np.array(look[k], copy=True) for k in ("ids", "w", "shift", "slid", "ref_ids", "ref_w", "ref_face", "orient", "region")}
        yield from uvsnap.ball_sides_steps(look["region"], pts, ip, fl, label, d0, self._trees, got)
        yield from uvsnap.side_reread_steps(look, got["sides"], sa.points, sa.indptr, sa.flat, region, pts, ip, fl, None, got, trees=self._trees,
                                            src_uv=sa.layers[sa.active][1])
        self.numbers_reread = int(got.get("reread", 0))

    def _face_material(self, ip, origin):
        """Material per face of the current mesh (the UV Check matches source faces of the same material), () when unknown."""

        if self.material is None or len(self.material) != len(self.indptr) - 1:
            return ()
        if origin is None:
            return self.material if len(ip) - 1 == len(self.material) else ()
        return self.material[np.asarray(origin, np.int64)] if len(origin) == len(ip) - 1 else ()

    def _layer_checked(self, uv, P, ip, fl, look, got, origin=None):
        """Generator: the active layer through the lookup; then the UV Check's flip test, and every re-read face that
        shows the texture turned over gets the lookup's own read back (and the layer is read again). The re-read only
        moves texture that was across a seam; it never turns any over."""

        sa, s = self.sa, self.settings
        yield from _layer_steps(sa, uv, P, ip, fl, look, s.texture_px, self.use_fast, got)
        before = getattr(self, "_before", None)
        if before is None or len(before["region"]) != len(ip) - 1:
            return
        changed = np.asarray(look["region"]) != before["region"]
        if not changed.any():
            return
        res = yield from self._check(P, ip, fl, got["uv"], origin, uv)
        if res is None:
            return
        bad = changed & res[1]
        if not bad.any():
            return
        ipa = np.asarray(ip, np.int64)
        corner = np.concatenate([np.arange(ipa[f], ipa[f + 1]) for f in np.flatnonzero(bad).tolist()])
        for k in ("ids", "w"):
            look[k].reshape(-1, 3)[corner] = before[k].reshape(-1, 3)[corner]
        for k in ("shift", "slid"):
            look[k][corner] = before[k][corner]
        for k in ("ref_ids", "ref_w"):
            look[k].reshape(-1, 3)[bad] = before[k].reshape(-1, 3)[bad]
        for k in ("ref_face", "orient", "region"):
            look[k][bad] = before[k][bad]
        self.numbers_reread = max(0, int(getattr(self, "numbers_reread", 0)) - int(bad.sum()))
        self.numbers_reread_undone = int(getattr(self, "numbers_reread_undone", 0)) + int(bad.sum())
        got.clear()
        yield from _layer_steps(sa, uv, P, ip, fl, look, s.texture_px, self.use_fast, got)

    def _ball_pass(self, uv, P, ip, fl, origin, only, lim, tree, numbers, tag, full=False):
        """Generator: `uvsnap.ball_snap_steps` on the faces still across a seam (`only`, the UV Check's), then the Check
        again: a vertex whose move leaves one of its faces across a seam that was not, or turns one's texture over, goes
        back. Passes repeat while one mends faces (Keep going: it stops by itself). Returns (points, only)."""

        if tree is None or only is None or not self.settings.ball_snap:
            return P, only
        sa, s = self.sa, self.settings
        region = np.asarray(self.plan.seams["region"], np.int64)
        if self._trees is None:
            self._trees = uvsnap.RegionTrees(sa.points, sa.indptr, sa.flat, region, self._tree_of)
        ipa = np.asarray(ip, np.int64)
        face_of = np.repeat(np.arange(len(ipa) - 1), np.diff(ipa))
        moved_total = undone_total = passes = 0
        self._ball_start = np.asarray(P, np.float64).copy()
        refused = np.zeros(0, np.int64)  # moves taken back stay back, so a later pass tries the faces' other ways
        mesh = uvsnap._Mesh(P, ip, fl, lazy=True)  # the topology stays; only points move
        yield from mesh.build(P, ip, fl)
        while True:
            pts = np.asarray(P, np.float64).reshape(-1, 3)
            _at, near_f, d0 = tree.nearest(pts)
            label = region[np.asarray(near_f, np.int64)]
            limit = np.asarray(lim, np.float64).ravel()
            if len(limit) < len(pts):  # vertices the split added: the plain share of their edge
                limit = np.r_[limit, s.move_share * uvsnap.vertex_edge(P, ip, fl)[len(limit):]]
            limit = limit.copy()
            limit[refused] = 0.0
            got = {}
            yield from uvsnap.ball_snap_steps(P, ip, fl, only, self._trees, label, d0, limit, got, mesh)
            if self._stopped:
                return P, only
            if self._judge is not None and len(got["moved"]):  # the camera check prices these moves too (SPEC 4.10)
                view = {}
                yield from view_limit(self._judge, P, got["points"], ip, fl, limit, s.detail, view)
                if view["held"]:
                    got = {}
                    yield from uvsnap.ball_snap_steps(P, ip, fl, only, self._trees, label, d0, view["limit"], got, mesh)
                    if self._stopped:
                        return P, only
            if not len(got["moved"]):
                break
            was_flip = self._last_flipped
            Q = np.asarray(got["points"], np.float64)
            now = yield from self._straddling(uv, Q, ip, fl, origin, tree if full else None)
            if self._stopped or now is None:
                return P, only
            worse = now & ~only
            turned = np.zeros(len(only), bool)
            if was_flip is not None and self._last_flipped is not None:
                turned = self._last_flipped & ~was_flip
            worse |= turned
            undone = 0
            if worse.any():  # take back each move that turned a face over or broke as many faces round it as it mended, then read again
                R = Q.reshape(-1, 3).copy()
                cand = np.intersect1d(np.unique(fl[worse[face_of]]), got["moved"])
                corner_v = fl[ipa[0]:ipa[-1]]
                mend = np.bincount(corner_v, (only & ~now)[face_of], len(pts))
                broke = np.bincount(corner_v, (now & ~only)[face_of], len(pts))
                flip_at = np.bincount(corner_v, turned[face_of], len(pts))
                back = cand[(flip_at[cand] > 0) | (broke[cand] >= mend[cand])]
                R[back] = pts[back]
                Q = R.ravel()
                undone = len(back)
                refused = np.union1d(refused, back)
                now = yield from self._straddling(uv, Q, ip, fl, origin, tree if full else None)
                if self._stopped or now is None:
                    return P, only
            gain = int(only.sum()) - int(now.sum())
            if gain <= 0:
                self._last_flipped = was_flip
                if undone and len(got["moved"]) > undone:  # some moves were good but the ones taken back cost as much: try without them
                    continue
                break
            P, only = Q, now
            moved_total += len(got["moved"]) - undone
            undone_total += undone
            passes += 1
        numbers.update({tag + "_moved": moved_total, tag + "_undone": undone_total, tag + "_passes": passes})
        start = np.asarray(self._ball_start, np.float64).reshape(-1, 3)
        now_p = np.asarray(P, np.float64).reshape(-1, 3)
        self._ball_moved[tag] = np.flatnonzero(np.any(now_p[:len(start)] != start, axis=1))
        return P, only

    def _last_check(self, P, ip, fl, origin, out, numbers):
        """Generator: the UV Check on the finished fit, as computed and at single precision. A speck (under
        FLIP_CLOSE_PX2 texels) it calls flipped, or one standing edge-on to the source polygons nearest it (`edge_on`:
        its winding against them is rounding), has its UVs closed to a streak on every layer (`streak`, corners move at
        most STREAK_MOVE_PX), so no rounding can mirror it; a flipped speck that cannot streak gets one UV, its
        centre's. The card's "left for baking" counts only faces across a seam that show at least one texel
        (`texel_area`), the faces Bake those spots bakes; the split's own counts are kept beside them."""

        sa, s = self.sa, self.settings
        a = sa.active
        numbers.update(faces_left_split=numbers["faces_left"], faces_for_baking_split=numbers["faces_for_baking"])
        straddle = flipped = None
        # as computed, and as the mesh stores it (single precision): a face edge-on to the source faces it reads has a
        # winding that rounding alone can turn, so it counts as flipped when either reading calls it so
        for pts, luv in ((P, out[a]), (np.asarray(P, np.float32).astype(np.float64), np.asarray(out[a], np.float32).astype(np.float64))):
            res = yield from self._check(pts, ip, fl, luv, origin, sa.layers[a][1])
            if res is None:
                return
            f_ = res[1]
            if straddle is None:
                straddle, flipped = res[0], f_
            else:
                flipped = flipped | f_
        shows, _density = texel_area(P, ip, fl, out[a], s.texture_px)
        ipa = np.asarray(ip, np.int64)
        small = np.flatnonzero(np.abs(_signed_uv_px(ipa, out[a], s.texture_px)) < FLIP_CLOSE_PX2)
        risky = flipped.copy()
        if len(small):  # a speck edge-on to the source faces nearest it: the Check's winding test on it is rounding
            yield
            risky[small[edge_on(P, ipa, fl, small, sa, self.use_fast, self._source_tree())]] = True
        shut = np.zeros(len(risky), bool)
        for f in np.flatnonzero(risky).tolist():
            c = slice(ipa[f], ipa[f + 1])
            if uv_area_px(out[a][c], s.texture_px) >= FLIP_CLOSE_PX2:
                continue
            got = streak(out[a][c])
            if got is None or np.max(np.linalg.norm(got - out[a][c], axis=1)) * s.texture_px > STREAK_MOVE_PX:
                if not flipped[f]:
                    continue  # an edge-on speck the Check reads the right way round keeps its texture
                got = None
            shut[f] = True
            for i in out:
                out[i][c] = streak(out[i][c]) if got is not None else out[i][c].mean(0)
        left = int(np.sum(straddle & (shows >= 1.0)))
        self._flip_corners = np.unique(np.concatenate([np.asarray(fl, np.int64)[ipa[f]:ipa[f + 1]] for f in np.flatnonzero(flipped & ~shut).tolist()] or [np.zeros(0, np.int64)]))
        numbers.update(flipped_closed=int(shut.sum()), flipped_left=int(np.sum(flipped & ~shut)), faces_under_texel=int(np.sum(straddle & (shows < 1.0))),
                       faces_left=min(int(numbers["faces_left"]), left))
        numbers["faces_for_baking"] = left - numbers["faces_left"]

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
                if s.view_check:  # the pictures are drawn while the first pass runs (on a worker only for the faster pictures)
                    got = {}
                    first = uvsnap.Snap(P, ip, fl, seams, xyz, max_move=move, use_fast=self.use_fast)
                    for _ in _together(self._drive(first), view_judge_steps(sa, s.view_pixels, got, self.use_fast)):
                        yield
                        if self._stopped:
                            return
                    judge = got.get("judge")
                    self._judge = judge
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
            move_used = move
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
        tree = None
        if soft and s.split:  # side switches: cut faces where the nearest source side changes
            yield
            tree = self._source_tree()
            yield
            only = None
            if s.switch_by == "side" or s.switch_snap:
                only = yield from self._straddling(uv, P, ip, fl)
                if self._stopped:
                    return
            if s.switch_snap and only is not None:  # zero faces first: corners over a side line move onto it
                region = np.asarray(seams["region"], np.int64)
                lim = np.asarray(move_used, np.float64) if s.snap else s.move_share * uvsnap.vertex_edge(P, ip, fl)
                side_of = lambda q: region[np.asarray(tree.nearest(np.asarray(q).reshape(-1, 3))[1], np.int64)]  # noqa: E731
                got = {}
                yield from uvsnap.switch_snap_steps(P, ip, fl, side_of, lim, only, got)
                if self._stopped:
                    return
                if self._judge is not None and len(got["moved"]):  # the camera check prices these moves too (SPEC 4.10)
                    view = {}
                    yield from view_limit(self._judge, P, got["points"], ip, fl, lim, s.detail, view)
                    numbers.update(side_view_held=int(view["held"]))
                    if view["held"]:
                        got = {}
                        yield from uvsnap.switch_snap_steps(P, ip, fl, side_of, view["limit"], only, got)
                        if self._stopped:
                            return
                before_slide = np.asarray(P, np.float64).reshape(-1, 3)
                P = np.asarray(got["points"], np.float64)
                numbers.update(side_moved=int(len(got["moved"])))
                if len(got["moved"]):  # a slide that turns a face's texture over is taken back (the UV Check's flip test)
                    was = self._last_flipped
                    only = yield from self._straddling(uv, P, ip, fl)
                    if self._stopped:
                        return
                    if only is not None and was is not None:
                        now = self._last_flipped & ~was
                        if now.any():
                            Q = P.reshape(-1, 3).copy()
                            face_of = np.repeat(np.arange(len(ip) - 1), np.diff(ip))
                            vs = np.unique(fl[now[face_of]])
                            Q[vs] = before_slide[vs]
                            P = Q.ravel()
                            numbers.update(side_slides_undone=int(len(vs)))
                            only = yield from self._straddling(uv, P, ip, fl)
                            if self._stopped:
                                return
                P0 = P
            ball_lim = np.asarray(move_used, np.float64) if s.snap else s.move_share * uvsnap.vertex_edge(P, ip, fl)
            if s.ball_snap:  # zero faces: corners of faces still across a seam move onto a side the others hold
                if only is None:
                    only = yield from self._straddling(uv, P, ip, fl)
                P, only = yield from self._ball_pass(uv, P, ip, fl, None, only, ball_lim, tree, numbers, "ball")
                if self._stopped:
                    return
                P0 = P
            if s.switch_by != "side":
                only = None
            got = {}
            yield from uvsnap.part_switches_steps(seams, sa.points, sa.indptr, sa.flat, uv, P, ip, fl, tree.nearest, got, by=s.switch_by, only=only)
            if self._stopped:
                return
            cut_seams = got["seams"]
            cut_xyz = uvsnap.chain_xyz(cut_seams, sa.points)
            numbers.update(part_switches=int(cut_seams.get("part_switches", 0)), straddling_before=int(np.sum(only)) if only is not None else -1)
        n_before = len(P0) // 3 if P0.ndim == 1 else len(P0)
        hold = np.zeros(0, np.int64)  # vertices the snap after the split leaves where they are (their moves turned a face over)
        while True:
            reserve = 0
            while True:
                P, ip, fl, origin = P0, ip0, fl0, np.arange(nf, dtype=np.int64)
                r_vchain = np.asarray(vchain, np.int64) if len(vchain) else np.full(n_before, -1, np.int64)
                if soft and s.split:
                    give = budget if budget < 0 else max(0.0, budget - reserve)
                    cut = uvsnap.Split(P, ip, fl, cut_seams, cut_xyz, vchain, vparam, (), give, use_fast=self.use_fast)
                    yield from self._drive(cut)
                    if self._stopped:
                        return
                    r = cut.result
                    P, ip, fl, origin = np.asarray(r["points"], np.float64), np.asarray(r["indptr"], np.int64), np.asarray(r["flat"], np.int64), np.asarray(r["origin"], np.int64)
                    r_vchain = np.asarray(r["vchain"], np.int64)
                    numbers.update(extra_faces=int(r["extra_faces"]), faces_split=int(r["faces_split"]), triangles_left=int(r["triangles_left"]),
                                   faces_left=int(r["faces_left"]), faces_for_baking=int(r["faces_cannot"]) + int(r["faces_bake"]), split_done=bool(r["done"]),
                                   split_budget=give)
                    if s.ball_snap and tree is not None:  # and on what the split left across a seam
                        left = yield from self._straddling(uv, P, ip, fl, origin, tree)
                        if self._stopped:
                            return
                        lim_after = np.asarray(ball_lim, np.float64).ravel()
                        lim_after = np.r_[lim_after, s.move_share * uvsnap.vertex_edge(P, ip, fl)[len(lim_after):]]
                        lim_after[hold[hold < len(lim_after)]] = 0.0
                        P, _left = yield from self._ball_pass(uv, P, ip, fl, origin, left, lim_after, tree, numbers, "ball_after", full=True)
                        if self._stopped:
                            return
                look = yield from self._lookup(uv, P, ip, fl)
                if look is None:
                    return
                yield from self._reread(look, tree, P, ip, fl, n_before, r_vchain)
                ready = {}
                yield from self._layer_checked(uv, P, ip, fl, look, ready, origin)
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
                yield from self._reread(look, tree, P, ip, fl, n_before, r_vchain)
            out = {}
            for i, (nm, luv) in enumerate(sa.layers):
                got = ready if i == sa.active and ready is not None else {}
                if not got:
                    if i == sa.active:
                        yield from self._layer_checked(luv, P, ip, fl, look, got, origin)
                    else:
                        yield from _layer_steps(sa, luv, P, ip, fl, look, s.texture_px, self.use_fast, got)
                if self._stopped:
                    return
                out[i] = close_flat(P, ip, fl, got["uv"], s.texture_px)
                if i == sa.active:
                    f = got["fold"]
                    numbers.update(folded_before=int(f["folded_before"]), folded_after=int(f["folded_after"]), fold_stuck=int(f["fold_stuck"]),
                                   fold_untangled=int(f.get("fold_untangled", 0)), fold_reread=int(f.get("fold_reread", 0)))
            yield from self._last_check(P, ip, fl, origin, out, numbers)
            if self._stopped:
                return
            back = np.setdiff1d(np.intersect1d(self._flip_corners, self._ball_moved.get("ball_after", np.zeros(0, np.int64))), hold)
            if not len(back):
                break
            hold = np.r_[hold, back]  # a face the fit turned over: the moves after the split at its corners are taken back, and the fit runs again
            numbers["ball_after_held"] = int(len(hold))
        numbers["side_reread"] = int(getattr(self, "numbers_reread", 0))
        numbers["side_reread_undone"] = int(getattr(self, "numbers_reread_undone", 0))
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


FLIP_CLOSE_PX2 = 4.0  # a face showing fewer texels than this (a 2 x 2 block) is a speck: flipped or edge-on, its UVs close to a streak


def uv_area_px(uv, texture_px: float = TEXTURE_PX) -> float:
    """Texels one polygon's UVs (corner order) cover (shoelace)."""

    u = np.asarray(uv, np.float64).reshape(-1, 2) * float(texture_px)
    return 0.5 * abs(float(np.sum(u[:, 0] * np.roll(u[:, 1], -1) - np.roll(u[:, 0], -1) * u[:, 1])))


STREAK_MOVE_PX = 1.0  # a speck's corners move at most this many texels when its UVs are closed to a streak


def streak(uv):
    """A polygon's UVs (corner order) closed to a line along their long way: a triangle's shortest edge, or a quad's
    two short opposite edges, shrink to their midpoints. Every fan triangle (0, k, k+1) then has two equal corners, so
    no rounding gives it a winding. None for other polygons."""

    u = np.asarray(uv, np.float64).reshape(-1, 2).copy()
    n = len(u)
    if n == 3:
        e = [float(np.linalg.norm(u[(i + 1) % 3] - u[i])) for i in range(3)]
        i = int(np.argmin(e))
        j = (i + 1) % 3
        u[i] = u[j] = (u[i] + u[j]) / 2.0
        return u
    if n == 4:
        pairs = ((0, 1), (2, 3)) if np.linalg.norm(u[1] - u[0]) + np.linalg.norm(u[3] - u[2]) <= np.linalg.norm(u[2] - u[1]) + np.linalg.norm(u[0] - u[3]) else ((1, 2), (3, 0))
        for i, j in pairs:
            u[i] = u[j] = (u[i] + u[j]) / 2.0
        return u
    return None


EDGE_ON_COS = 0.05  # a fan triangle this close to perpendicular (|cos| of the normals) to the source polygon nearest its centre is edge-on to it


def _signed_uv_px(indptr, uv, texture_px: float) -> np.ndarray:
    ip = np.asarray(indptr, np.int64)
    u = np.asarray(uv, np.float64).reshape(-1, 2) * float(texture_px)
    nf = len(ip) - 1
    owner = np.repeat(np.arange(nf), np.diff(ip))
    nxt = np.arange(1, len(u) + 1)
    nxt[ip[1:] - 1] = ip[:-1]
    return np.bincount(owner, u[:, 0] * u[nxt, 1] - u[nxt, 0] * u[:, 1], nf) / 2.0


def edge_on(points, indptr, flat, faces, sa: "Source", use_fast: bool = True, tree=None) -> np.ndarray:
    """Per face in `faces`: True when every fan triangle of it stands edge-on (EDGE_ON_COS) to the source polygon nearest
    its centre (`tree`: the source's face tree, built when not given). Such a face reads the texture at a grazing angle,
    and the sign of its winding against those source faces is down to rounding."""

    faces = np.asarray(faces, np.int64)
    if not len(faces):
        return np.zeros(len(faces), bool)
    P = np.asarray(points, np.float64).reshape(-1, 3)
    ip, fl = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
    sizes = np.diff(ip)[faces]
    fan = np.maximum(sizes - 2, 0)
    tf = np.repeat(np.arange(len(faces)), fan)
    k = np.arange(len(tf)) - np.repeat(np.cumsum(fan) - fan, fan) + 1
    s0 = ip[faces][tf]
    a, b, c = P[fl[s0]], P[fl[s0 + k]], P[fl[s0 + k + 1]]
    n = np.cross(b - a, c - a)
    if tree is None:
        tree = face_tree(sa.points, FlatFaces(sa.indptr, sa.flat), use_fast)
    _at, g, _d = tree.nearest((a + b + c) / 3.0)
    g = np.asarray(g, np.int64)
    sP = np.asarray(sa.points, np.float64).reshape(-1, 3)
    sip, sfl = np.asarray(sa.indptr, np.int64), np.asarray(sa.flat, np.int64)
    s_own = np.repeat(np.arange(len(sip) - 1), np.diff(sip))
    s_nxt = np.arange(1, len(sfl) + 1)
    s_nxt[sip[1:] - 1] = sip[:-1]
    cr = np.cross(sP[sfl], sP[sfl[s_nxt]])
    sn = np.stack([np.bincount(s_own, cr[:, j], len(sip) - 1) for j in range(3)], 1)[g]  # Newell normals of the nearest polygons
    cos = np.abs((n * sn).sum(1)) / np.maximum(np.linalg.norm(n, axis=1) * np.linalg.norm(sn, axis=1), 1e-300)
    return np.bincount(tf, cos >= EDGE_ON_COS, len(faces)) == 0


def texel_area(points, indptr, flat, uv, texture_px: float = TEXTURE_PX) -> tuple:
    """(texels per face at the layout's own density, density): 3D area x density^2, density = the median over faces of
    sqrt(UV texels / 3D area), the texels per mesh unit of the layout. A face below 1 shows no texture of its own."""

    P = np.asarray(points, np.float64).reshape(-1, 3)
    ip, fl = np.asarray(indptr, np.int64), np.asarray(flat, np.int64)
    nf = len(ip) - 1
    if nf == 0:
        return np.zeros(0), float(texture_px)
    owner = np.repeat(np.arange(nf), np.diff(ip))
    nxt = np.arange(1, len(fl) + 1)
    nxt[ip[1:] - 1] = ip[:-1]
    cr = np.cross(P[fl], P[fl[nxt]])
    a3 = np.linalg.norm(np.stack([np.bincount(owner, cr[:, k], nf) for k in range(3)], 1), axis=1) / 2.0
    u = np.asarray(uv, np.float64).reshape(-1, 2) * float(texture_px)
    a2 = np.abs(np.bincount(owner, u[:, 0] * u[nxt, 1] - u[nxt, 0] * u[:, 1], nf)) / 2.0
    ok = (a3 > 0) & (a2 > 0)
    dens = float(np.median(np.sqrt(a2[ok] / a3[ok]))) if ok.any() else float(texture_px)
    return a3 * dens * dens, dens


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
        u, e, fold = yield from uvseams.layer_steps(sa.indptr, sa.flat, luv, P, ip, fl, look, texture_px, True, sa.points)
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
