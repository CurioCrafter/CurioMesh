# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""UV seams inside Auto retopology (package A4, SPEC 4.6 Phase B). Main-only file.

When UVs keep the source's (Seam Fit: Light), the search itself follows the Soft seams instead of leaving them all to
the fit after it:
- `attach(job)`: the job finds the plan its kinds came from (`uvkeep`), and keeps the Soft chains as 3D lines.
- `cut(job, ...)`: the Soft chains go into the coarse mesh the way drawn guide curves do (the job's own `_guided`:
  refine near the line, a chain of edges that hugs it, pulled onto it), and their edges become the pairing's guide
  edges, so no quad is made across them. Nothing else of the job changes: collapse, rebuilt parts and repairs still run.
- `remap(job, coarse)`: those guide edges found again by position when the rows-on-round-areas or flat-area steps
  renumber the coarse mesh before pairing.
- `fix(job, ...)`: after the job's repairs, Seam Snap runs on every seam still off the edges through Auto Fix's
  `propose` (the same Trial, spots, guards and accept rule as every Tidy Tool), within Most Detail Loss or Auto's own
  test for its polish. The split leftovers and per-side transfer stay where A1 put them (after the search, inside the
  extra-face allowance).

NumPy only, no bpy. Our own design and code (SPEC 4.4-4.6). The guide-edge mechanism is the add-on's own (guides.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys

import numpy as np

from . import autofix, trial as T, uvkeep, uvsnap, uvseams, uvtrial

# Constants table (starting values, nothing here is a limit).
NEAR = 1.0  # a seam's spot is the faces with a corner within this many local edges of it (the snap's corridor)
LEAST = 0.05  # the smallest drop of a spot's own number (seam area off the edges, in quads) that counts as better
STEP = 0.25  # seams are sampled every this many local edges to measure how far they lie from the edges
FOLLOW = False  # cut the Soft seams in during the search. Off by default: in the add-on's pairing every seam edge left
# triangles that became quads, measured +60 % faces from the search on concrete_road_barrier, +34 % marble_bust_01,
# +194 % lion_head (A4 bench, 2026-11-07). uvkeep.OVERRIDE["follow"] (or a later setting) turns it on.
REFINE = True  # split the triangles near a seam finer before laying it on edges (as drawn guide curves are)
DEFAULT_STRENGTH = 1.0  # share of uvkeep.MOVE_SHARE of the local edge a corner may move; Auto Fix halves it for the look


LAST: dict = {}  # the seams the last search followed: Seam Snap made by Auto Fix (no arguments) works on them


@dataclass
class Auto:
    """What a job keeps about the seams it follows."""

    plan: object
    seams: dict
    xyz: np.ndarray  # welded source positions for the chains (`uvsnap.chain_xyz`)
    soft: np.ndarray  # Soft chain ids
    lines: list  # [(k, 3)] the Soft chains as 3D polylines
    edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3)))  # (k, 2, 3) the cut's guide edges, by position
    report: dict = field(default_factory=dict)


def chain_lines(seams: dict, xyz, chains, keep_empty: bool = False) -> list:
    """The chains as (k, 3) polylines; chains with fewer than two points are left out (or None with `keep_empty`)."""

    X = np.asarray(xyz, np.float64).reshape(-1, 3)
    ptr, run = np.asarray(seams["run_indptr"], np.int64), np.asarray(seams["run_vertex"], np.int64)
    out = []
    for c in np.asarray(chains, np.int64).tolist():
        line = X[run[ptr[c]:ptr[c + 1]]]
        if len(line) >= 2:
            out.append(line)
        elif keep_empty:
            out.append(None)
    return out


def resample(line, step: float) -> np.ndarray:
    """Points along the polyline about `step` apart, both ends included."""

    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    at = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(np.ceil(at[-1] / max(step, 1e-30))) + 1)
    t = np.linspace(0.0, at[-1], n)
    return np.stack([np.interp(t, at, line[:, k]) for k in range(3)], axis=1)


def _gs(job):
    try:
        return job.context.scene.curiomesh_guided
    except Exception:  # no Blender scene (tests): uvkeep.OVERRIDE decides
        return None


def attach(job):
    """The seams this job follows, or None: Seam Fit is not Light, the source has no Soft seams, or the job works on a
    mirrored half, a region or drawn strokes (those keep their own lines; the fit after the search still runs)."""

    LAST.clear()
    if job.mirror is not None or job.outside is not None or job.guide_lines or job.strips:
        return None
    if uvkeep.mode_of(_gs(job)) != "light":
        return None
    nf, npt = len(job.faces), len(job.co)
    for plan in reversed(list(uvkeep._LATEST.values())):
        s = plan.seams
        if len(s.get("island", ())) == nf and len(s.get("weld", ())) == npt:
            break
    else:
        return None
    soft = np.flatnonzero(np.asarray(s["class"]) == uvseams.SOFT)
    xyz = uvsnap.chain_xyz(s, job.co)
    lines = chain_lines(s, xyz, soft)
    if not lines:
        return None
    LAST.update(seams=s, xyz=xyz)
    return Auto(plan, s, xyz, soft, lines, report=dict(soft=len(lines)))


def cut(job, co, faces, kinds):
    """The coarse mesh with the Soft seams laid on its edges (the guide-curve steps of `guides`: the chain of edges
    that hugs each seam, pulled onto it) and those edges handed to the pairing as guide edges, or the input when not
    following. The triangles are not split finer first (`REFINE`): every edge a seam adds costs quads in the pairing."""

    a = getattr(job, "uv", None)
    if a is None or job.guide_lines or job.strips or job.methods or job.method != "AUTO" or not len(faces):
        return co, faces, kinds
    if not uvkeep.OVERRIDE.get("follow", FOLLOW):
        return co, faces, kinds
    if REFINE:
        job.guide_lines = a.lines
        try:
            co, faces, kinds, job.guide_edges = job._guided(co, faces, kinds)
        finally:
            job.guide_lines = []
    else:
        guides = sys.modules[type(job).__module__].guides  # the job's own guide-curve module
        tris, owner = [], []
        for i, face in enumerate(faces):
            for k in range(1, len(face) - 1):
                tris.append((face[0], face[k], face[k + 1]))
                owner.append(i)
        tris = np.asarray(tris, np.int64)
        co = np.asarray(co, np.float64)
        spacing = float(np.median(np.linalg.norm(co[tris[:, 0]] - co[tris[:, 1]], axis=1)))
        chains, keys = guides.chain(co, tris, a.lines, spacing)
        if chains:
            co = guides.pull(co, tris, chains, a.lines, spacing)
        faces, kinds = [tuple(t) for t in tris.tolist()], [kinds[o] for o in owner]
        job.guide_edges = tuple(sorted(keys))
    pairs = np.asarray(job.guide_edges, np.int64).reshape(-1, 2)
    a.edges = np.asarray(co, np.float64)[pairs] if len(pairs) else np.zeros((0, 2, 3))
    a.report["guide_edges"] = int(len(pairs))
    return co, faces, kinds


def _keys(points, tol):
    return [tuple(k) for k in np.round(np.asarray(points, np.float64) / tol).astype(np.int64).tolist()]


def remap(job, coarse):
    """The cut's guide edges as corner pairs of `coarse` (found by position), or the job's own when not following."""

    a = getattr(job, "uv", None)
    if a is None or not len(a.edges):
        return job.guide_edges
    pts = np.asarray(coarse.vertices, np.float64)
    tol = 1e-6 * (float(np.ptp(pts, axis=0).max()) if len(pts) else 1.0) or 1e-9
    where = {k: i for i, k in enumerate(_keys(pts, tol))}
    ends = _keys(a.edges.reshape(-1, 3), tol)
    out = set()
    for x, y in zip(ends[0::2], ends[1::2]):
        i, j = where.get(x), where.get(y)
        if i is not None and j is not None and i != j:
            out.add((min(i, j), max(i, j)))
    a.report["guide_edges_kept"] = len(out)
    return tuple(sorted(out))


# ---------------------------------------------------------------- Seam Snap as a Trial (Auto Fix's spot pattern)


@dataclass
class SeamSpot(T.Spot):
    chains: tuple = ()  # the Soft chains this spot holds (touching chains are one spot)
    samples: np.ndarray | None = None  # (k, 3) points along those chains, `STEP` edges apart


def _flat(faces):
    sizes = np.fromiter((len(f) for f in faces), np.int64, len(faces))
    indptr = np.concatenate([[0], np.cumsum(sizes)])
    flat = np.fromiter((v for f in faces for v in f), np.int64, int(indptr[-1]))
    return indptr, flat


def _to_segments(q, a, b):
    """Distance of each point of `q` (m, 3) to the nearest of the segments a[i]-b[i] (closest point on a segment:
    Ericson 2005, 5.1.2). The segments are binned in a grid first (`uvtrial._to_segments`: the same distances, each point
    measured against the segments near it instead of all of them)."""

    return uvtrial._to_segments(q, a, b)


def _segment_distance(q, line):
    """Distance of each point of `q` (m, 3) to the polyline `line` (k, 3)."""

    return _to_segments(q, line[:-1], line[1:])


class SeamSnap(T.Trial):
    """Moves result corners onto the Soft UV seams the edges do not follow yet (`uvsnap.Snap`, SPEC 4.5), never
    flipping a face and never past the allowed move. Positions only. Its own number per spot: the seam's area off the
    edges of the spot, in quads (how far its samples lie from the nearest edge, summed along it)."""

    name = "Seam Snap"
    group = "CLEANUP"
    handles = ("strength",)
    words = ("0", "1")
    defaults = {"strength": DEFAULT_STRENGTH}
    blendable = True  # positions only, each corner moves along the surface toward its seam
    stages = ("move", "look")
    kinds = (("seam", "seam off the edges", "seams off the edges"),)
    own = "seam_off"
    least = LEAST
    changes_faces = False

    def __init__(self, seams: dict | None = None, xyz=None, use_fast: bool = True):
        if seams is None:  # made by Auto Fix's plan: the seams of the last search, if it followed any
            seams, xyz = LAST.get("seams"), LAST.get("xyz")
        self.seams, self.xyz, self.use_fast = seams, xyz, use_fast
        self._active = None  # Soft chains that have a spot; the run works on those only

    def check(self, base: T.Base) -> str:
        if self.seams is None or not np.any(np.asarray(self.seams["class"]) == uvseams.SOFT):
            return "No UV seams to fit on this result."
        return ""

    def _only(self, chains):
        s = dict(self.seams)
        cls = np.asarray(s["class"]).copy()
        keep = np.zeros(len(cls), bool)
        keep[np.asarray(list(chains), np.int64)] = True
        cls[(cls == uvseams.SOFT) & ~keep] = uvseams.IGNORE
        s["class"] = cls
        return s

    def _snap(self, co, faces, seams, max_move):
        ip, fl = _flat(faces)
        stage = uvsnap.Snap(np.asarray(co, np.float64).ravel(), ip, fl, seams, self.xyz, max_move=max_move, use_fast=self.use_fast)
        while not stage.done:
            stage.step()
            yield
        return stage.result

    def spots(self, base: T.Base):
        if self.check(base):
            return []
        co = np.asarray(base.co, np.float64)
        reach, step = NEAR * base.edge, STEP * base.edge
        ip, fl = _flat(base.faces)
        face_of = np.repeat(np.arange(len(base.faces)), np.diff(ip))
        free = base.scope[face_of]
        soft = np.flatnonzero(np.asarray(self.seams["class"]) == uvseams.SOFT)
        found = []  # (chain, corner ids near it, its samples)
        for c, line in zip(soft.tolist(), chain_lines(self.seams, self.xyz, soft, keep_empty=True)):
            if line is None:
                continue
            lo, hi = line.min(0) - reach, line.max(0) + reach
            cand = np.flatnonzero(np.all((co >= lo) & (co <= hi), axis=1))
            if len(cand):
                near = cand[_segment_distance(co[cand], line) <= reach]
                if len(near):
                    found.append((c, near, resample(line, step)))
            yield
        if not found:
            return []
        # chains whose corridors share a corner are one spot (spots never share a corner)
        owner = np.full(len(co), -1, np.int64)
        root = list(range(len(found)))

        def top(i):
            while root[i] != i:
                root[i] = root[root[i]]
                i = root[i]
            return i

        for i, (_c, near, _s) in enumerate(found):
            for j in np.unique(owner[near][owner[near] >= 0]).tolist():
                root[top(j)] = top(i)
            owner[near] = i
        groups: dict = {}
        for i in range(len(found)):
            groups.setdefault(top(i), []).append(i)
        on_corner = np.zeros(len(co), bool)
        out = []
        for members in groups.values():
            on_corner[:] = False
            for i in members:
                on_corner[found[i][1]] = True
            fids = np.unique(face_of[on_corner[fl] & free])
            if not len(fids):
                continue
            sp = SeamSpot("seam", fids, co[np.flatnonzero(on_corner)].mean(0), 0.0, chains=tuple(sorted(found[i][0] for i in members)),
                          samples=np.vstack([found[i][2] for i in members]))
            sp.bad = self.own_number(base, co, base.faces, sp)
            if sp.bad > LEAST:
                out.append(sp)
            yield
        self._active = sorted(c for sp in out for c in sp.chains)
        return out

    def own_number(self, base: T.Base, co, faces, spot) -> float:
        """The seam's area off the edges of the spot's faces, in quads: lower is better, 0 when it runs along edges."""

        samples = getattr(spot, "samples", None)
        if samples is None or not len(samples) or not len(spot.faces):
            return float(spot.bad)
        e = T.edges_of([faces[f] for f in np.asarray(spot.faces, np.int64).tolist()])
        co = np.asarray(co, np.float64)
        d = _to_segments(samples, co[e[:, 0]], co[e[:, 1]])
        return float(d.sum() * STEP / base.edge)

    def run(self, base: T.Base, settings: dict, rough: bool = False):
        if self.check(base):
            return T.Candidate(None, refusal=self.check(base))
        co = np.asarray(base.co, np.float64)
        ip, fl = _flat(base.faces)
        face_of = np.repeat(np.arange(len(base.faces)), np.diff(ip))
        inside = np.zeros(len(co), bool)
        inside[fl[base.scope[face_of]]] = True
        outside = np.zeros(len(co), bool)
        outside[fl[~base.scope[face_of]]] = True
        free = inside & ~outside  # the scope's edge stays, as for every tool
        share = float(settings.get("strength", DEFAULT_STRENGTH)) * uvkeep.MOVE_SHARE
        max_move = np.where(free, share * uvsnap.vertex_edge(co.ravel(), ip, fl), 0.0)
        seams = self.seams if self._active is None else self._only(self._active)
        r = yield from self._snap(co, base.faces, seams, max_move)
        moved = np.asarray(r["points"], np.float64).reshape(-1, 3)
        cand = T.Candidate(moved, strength=float(settings.get("strength", DEFAULT_STRENGTH)), exact=not rough)
        cand.note = f"{int(r['moved'])} corners onto seams, {int(np.sum(r['held']))} held"
        return cand


# ---------------------------------------------------------------- inside the job


def fix(job, points, faces, looks):
    """Generator -> (points, (loss, shading, soft)): Seam Snap through Auto Fix's `propose` on every seam spot, kept by
    `autofix.verdict` within the look budget (Most Detail Loss, or Auto's polish test: 1.1 x Detail or the look
    before). Nothing changes when it would not be kept."""

    a = getattr(job, "uv", None)
    if a is None or getattr(job, "_stopped", False) or not len(faces) or getattr(job, "_view", None) is None:
        return points, looks
    center, half, reference, reference_shaded, pixel = job._view
    last = {"looks": looks}

    def look(co, fs):
        got = yield from job._look(np.asarray(co, np.float64), [tuple(f) for f in fs], center, half, reference, reference_shaded, pixel)
        last["looks"] = got
        return got[0]

    base = T.make_base(points, faces)
    base.cache["look"] = looks[0]
    trial = SeamSnap(a.seams, a.xyz)
    job.stage = "Finding seams off the edges"
    spots = yield from trial.spots(base)
    a.report.update(spots=len(spots))
    if not spots:
        return points, looks
    job.stage = f"Moving edges onto {len(spots)} UV seams"
    budget = getattr(job, "uv_budget", None)
    if budget is None:
        budget = lambda before: max(1.1 * job.detail, before)  # Auto's own test for its polish
    p = yield from autofix.propose(trial, base, {}, spots, look, budget)
    a.report.update(fixed=len(p.fixed), left=len(p.left), reason=p.reason, words=p.words, strength=p.held)
    if p.reason:
        return points, looks
    if getattr(job, "instances", None) is not None:
        job.instances = None  # copies are no longer alike
    return np.asarray(p.cand.co, np.float64), last["looks"]
