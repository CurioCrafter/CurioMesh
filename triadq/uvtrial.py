# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Look First Trials for UVs (package A3, SPEC 10.1): Seam Fit, New, New page layout, Cut here / Join here and Bake
those spots.

Seam Fit changes the result's faces, so it is a `trial.Trial` and opens in Look First like every Tidy Tool; Apply
keeps it as one Undo step and one History entry, and the result's Apply writes exactly the fit that was shown
(`remember_applied`: `uv_hooks.after_apply` finds it by the result's hash). The other four change only the UVs of a
mesh that has them. They are `UVTrial`s on a `UVBase`, with the same shape: `check`, a sliced `run` that returns a
`UVCand`, `measure` for the card, Again from the candidate, Keep going while a pass gains.

NumPy only, no bpy. Every `run` yields after a few milliseconds of work; long core stages are polled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
import time

import numpy as np

from . import trial as T
from . import uvkeep, uvnew, uvpack, uvscore, uvseams, uvsnap
from .types import FlatFaces

# What the panel hands Seam Fit before it opens: a callable returning dict(source, plan, settings, material, extra) or None.
FIT: dict = {"inputs": None}
APPLIED_EXTRA: dict = {}  # result key -> extra faces Seam Fit already spent on it (Again counts them against the allowance)

SEAM_STEP = 0.25  # Soft chains are sampled every this many edges for Seam Fit's own number
OWN_CHUNK = 128  # seam samples measured per slice
_OWN: dict = {}  # own numbers already worked out (keyed by the mesh's fingerprint and the faces)
SEAM_LEAST = 0.05  # the smallest drop of that number (in quads) that counts as better: a tolerance, not a cap
NO_UVS = "Your mesh has no UVs to keep. New will make some."
NOT_LIGHT = "Seam Fit runs when UVs are set to keep the source's (Seam Fit: Light)."
CHECK_PX = 1024.0
TIME_WORDS = "Stopped after {t:g} s with {what} still improving. Keep going continues, or raise Time per pass."


def _faces_of(indptr, flat) -> list:
    return list(FlatFaces(indptr, flat))


def _edge_keys(faces, n: int) -> np.ndarray:
    f = FlatFaces.of(faces)
    ip, fl = f.indptr, f.flat
    nxt = np.arange(1, len(fl) + 1)
    nxt[ip[1:] - 1] = ip[:-1]
    lo, hi = np.minimum(fl, fl[nxt]), np.maximum(fl, fl[nxt])
    return np.unique(lo * n + hi)


def new_edges(old_faces, new_faces, n: int) -> np.ndarray:
    """(k, 2) corner pairs of edges of `new_faces` that `old_faces` did not have: the split lines. `n`: corner count."""
    new = _edge_keys(new_faces, n)
    new = new[~np.isin(new, _edge_keys(old_faces, n))]
    return np.stack([new // n, new % n], axis=1)


def allowance_of(value: float):
    """The dial's value as the allowance: 0..1 is 0..100 % extra faces, the end of the dial is "any" (split everything)."""
    return None if value >= 1.0 - 1e-9 else max(0.0, float(value))


class SeamFit(T.Trial):
    """Snap, split faces on seams and per-side transfer on the result (`uvkeep.FitStage`), previewed before Apply."""

    name = "Seam Fit"
    group = "CLEANUP"
    handles = ("faces",)
    words = ("0%", "any")
    defaults = {"faces": uvkeep.DEFAULT_ALLOWANCE, "split": True, "keep_going": False}
    blendable = False
    changes_faces = True
    splits_faces = True  # split pieces are expected: Auto Fix counts bad faces per parent and reads poles, Tidiness and Loop Quality off them
    layers = ("scope", "ghost", "wash", "arrows", "held")
    stages = ("split", "faces", "left")
    toggles = (("split", "Split faces on seams"), ("keep_going", "Keep going"))
    legend = ""
    # Auto Fix (AUTO_FIX.md 2.1): one spot holds every face the fit changes; its own number is the seams' length off
    # the spot's edges, in quads (0 when every seam runs along edges)
    auto = True
    kinds = (("across", "face across a UV seam", "faces across UV seams"),)
    own = "seam_off"
    least = SEAM_LEAST

    def __init__(self, inputs: dict | None = None):
        self._inputs = inputs  # dict(source, plan, settings, material); None: the panel's provider (FIT["inputs"])

    def inputs(self):
        if self._inputs is not None:
            return self._inputs
        get = FIT.get("inputs")
        return get() if get is not None else None

    def value_text(self, value: float) -> str:
        a = allowance_of(value)
        return "Extra faces any" if a is None else f"Extra faces {100 * a:.0f}%"

    def check(self, base: T.Base) -> str:
        got = self.inputs()
        if got is None:
            return NO_UVS
        if got.get("refusal"):
            return got["refusal"]
        mat = got.get("material")
        if mat is not None and len(mat) != len(base.faces):
            return "The result changed since Seam Fit was set up. Open it again."
        return ""

    def _edge(self, base: T.Base) -> float:
        """One length scale for every own number of this trial: the first result it saw (spots and run work on sub-scopes
        whose mean edge differs, and a split shortens edges; the before and after numbers must use the same quad)."""
        if getattr(self, "_scale", None) is None:
            self._scale = float(base.edge)
        return self._scale

    def _samples(self, base: T.Base):
        """(k, 3) points along the Soft chains, SEAM_STEP quads apart, worked out once per inputs."""
        got = self.inputs()
        key = (id(got["plan"]), self._edge(base))
        if getattr(self, "_sample_key", None) != key:
            seams = got["plan"].seams
            xyz = uvsnap.chain_xyz(seams, got["source"].points).reshape(-1, 3)
            ptr, run = np.asarray(seams["run_indptr"], np.int64), np.asarray(seams["run_vertex"], np.int64)
            out = []
            for c in np.flatnonzero(np.asarray(seams["class"]) == uvseams.SOFT).tolist():
                line = xyz[run[ptr[c]:ptr[c + 1]]]
                if len(line) < 2:
                    continue
                seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
                at = np.concatenate([[0.0], np.cumsum(seg)])
                t = np.linspace(0.0, at[-1], max(2, int(np.ceil(at[-1] / (SEAM_STEP * key[1]))) + 1))
                out.append(np.stack([np.interp(t, at, line[:, k]) for k in range(3)], axis=1))
            self._sample_key, self._sample_pts = key, (np.vstack(out) if out else np.zeros((0, 3)))
        return self._sample_pts

    def own_steps(self, base: T.Base, co, faces, ids):
        """Generator: the own number of faces `ids` of (co, faces), worked out in slices of OWN_CHUNK samples and kept,
        so `own_number` (which Auto Fix calls in one go) only looks it up."""
        ids = np.asarray(ids, np.int64)
        co = np.asarray(co, np.float64)
        key = (co.shape, float(co.sum()), len(faces), hashlib.sha1(ids.tobytes()).hexdigest(), self._edge(base))
        if key in _OWN:
            return _OWN[key]
        edge = self._edge(base)
        value = 0.0
        if len(ids):
            sub = [faces[f] for f in ids.tolist()]
            corners = co[np.unique(np.fromiter((v for f in sub for v in f), np.int64))]
            q = self._samples(base)
            lo, hi = corners.min(0) - edge, corners.max(0) + edge
            q = q[np.all((q >= lo) & (q <= hi), axis=1)]
            if len(q):
                e = T.edges_of(sub)
                a, b = co[e[:, 0]], co[e[:, 1]]
                yield
                for k in range(0, len(q), OWN_CHUNK):
                    value += float(_to_segments(q[k:k + OWN_CHUNK], a, b, near_only=True).sum())
                    yield
                value *= SEAM_STEP / edge
        _OWN[key] = value
        while len(_OWN) > 32:
            _OWN.pop(next(iter(_OWN)))
        return value

    def own_number(self, base: T.Base, co, faces, spot) -> float:
        """Seam length off the edges of the spot's faces, in quads: the samples within one quad of the spot's corners,
        each counted by how far it lies from the nearest edge, up to about a quad (lower is better)."""
        if not len(spot.faces):
            return float(spot.bad)
        return T.finish(self.own_steps(base, co, faces, spot.faces))

    def spots(self, base: T.Base):
        """One spot: every face the fit at any allowance changes (moved corners, split faces)."""
        if self.check(base):
            return []
        allowance = self.inputs()["settings"].allowance
        faces = 1.0 if allowance is None else min(1.0, float(allowance))  # the popover's Extra faces for seams
        cand = yield from self.run(base, dict(self.defaults, faces=faces))
        if cand.refusal or cand.co is None:
            return []
        moved = np.zeros(len(base.co), bool)
        n = min(len(base.co), len(cand.co))
        moved[:n] = np.linalg.norm(np.asarray(cand.co[:n]) - base.co[:n], axis=1) > 1e-9 * base.edge
        f = FlatFaces.of(base.faces)
        owner = np.repeat(np.arange(len(base.faces)), np.diff(f.indptr))
        changed = np.zeros(len(base.faces), bool)
        changed[owner[moved[f.flat]]] = True
        changed |= np.bincount(np.asarray(cand.parent, np.int64), minlength=len(base.faces)) > 1
        ids = np.flatnonzero(changed & base.scope)
        if not len(ids):
            return []
        at = base.co[np.unique(f.flat[np.isin(owner, ids)])].mean(axis=0)
        sp = T.Spot("across", ids, at, 0.0, pick={"faces": faces})  # its run uses the same allowance
        sp.bad = yield from self.own_steps(base, base.co, base.faces, ids)
        return [sp] if sp.bad > self.least else []

    def run(self, base: T.Base, settings: dict, rough: bool = False):
        got = self.inputs()
        sa, plan = got["source"], got["plan"]
        s = replace(got["settings"], mode="light", allowance=allowance_of(float(settings.get("faces", uvkeep.DEFAULT_ALLOWANCE))), split=bool(settings.get("split", True)))
        faces = FlatFaces.of(base.faces)
        key = uvkeep.result_key(base.co, faces.indptr, faces.flat)
        extra = int(got.get("extra", APPLIED_EXTRA.get(key, 0)))
        stage = uvkeep.FitStage(sa, plan, base.co, faces.indptr, faces.flat, got.get("material"), s, extra_so_far=extra, file=False)  # only an applied fit is filed
        passes, best = 0, None
        try:
            while True:
                t = 0
                while not stage.done:
                    stage.step()
                    t += 1
                    yield min(0.95, 0.02 * t / (1 + passes))
                c = stage.candidate
                if c is None:
                    break
                left = int(c.numbers.get("faces_left", 0)) + int(c.numbers.get("faces_for_baking", 0))
                if best is not None and left >= best[1]:
                    break  # this pass made nothing better: keep the last one (Keep going stops by itself)
                best, passes = (c, left), passes + 1
                if not settings.get("keep_going") or left == 0:
                    break
                stage = uvkeep.again(c, sa, plan, s, file=False)
        finally:
            stage.stop()
        if best is None:
            return T.Candidate(None, refusal="Seam Fit stopped before it finished.")
        c = best[0]
        n = c.numbers
        yield 0.96
        new_faces = _faces_of(c.indptr, c.flat)
        cand = T.Candidate(np.asarray(c.points, np.float64), faces=new_faces, parent=np.asarray(c.origin, np.int64), exact=True)
        yield 0.97
        cand.lines = new_edges(base.faces, FlatFaces(c.indptr, c.flat), len(c.points))
        yield 0.98
        cand.fit = c
        cand.fit_extra = extra + int(n.get("extra_faces", 0))
        split, left = int(n.get("faces_split", 0)), int(n.get("faces_left", 0)) + int(n.get("faces_for_baking", 0))
        moved = int(n.get("moved", 0)) + int(n.get("side_moved", 0))  # side_moved: the side-line snap's moves
        cand.numbers["split"] = (f"{split} faces split, {moved} points moved", 1 if split or moved else 0)
        cand.numbers["left"] = (f"{left} left for baking" if left else "every seam fitted", -1 if left else 1)
        notes = list(c.messages)
        if int(n.get("held", 0)):
            notes.append(f"{int(n['held'])} points held by the camera check")
        if passes > 1:
            notes.append(f"{passes} passes")
        cand.note = "; ".join(notes)
        remember_applied(cand)  # filed now, so whichever path keeps it (the modal, F9, a script) is written by Apply
        if not base.scope.all():  # a spot of Auto Fix: its number after the fit, worked out here in slices
            yield from self.own_steps(base, cand.co, new_faces, np.flatnonzero(np.isin(cand.parent, np.flatnonzero(base.scope))))
        return cand


def _point_seg(q, a, b):
    """Distance of each point q[i] to the segment a[i]-b[i] (rows paired)."""
    ab = b - a
    t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-30), 0.0, 1.0)
    return np.linalg.norm(q - (a + t[:, None] * ab), axis=1)


def _to_segments(q, a, b, near_only: bool = False):
    """Distance of each point of `q` (m, 3) to the nearest segment a[i]-b[i]. Segments are binned in a grid of cells as
    big as the longest one; a point checks the 27 cells round it, which is exact whenever the nearest segment is
    within half a cell (any segment outside lies further). Points with none that near are checked against all, or,
    with `near_only`, count as one cell away (a seam's own number only needs "off by about a quad or more")."""
    q, a, b = np.asarray(q, np.float64), np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not len(q) or not len(a):
        return np.full(len(q), np.inf)
    h = max(float(np.linalg.norm(b - a, axis=1).max()), 1e-12)
    origin = np.minimum(a.min(0), q.min(0))
    cell = np.floor(((a + b) / 2 - origin) / h).astype(np.int64)
    span = cell.max(0) + 3
    key = (cell[:, 0] * span[1] + cell[:, 1]) * span[2] + cell[:, 2]
    order = np.argsort(key, kind="stable")
    skey = key[order]
    qc = np.floor((q - origin) / h).astype(np.int64)
    best = np.full(len(q), np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = qc + (dx, dy, dz)
                k = (c[:, 0] * span[1] + c[:, 1]) * span[2] + c[:, 2]
                lo, hi = np.searchsorted(skey, k), np.searchsorted(skey, k, side="right")
                n = hi - lo
                if not n.any():
                    continue
                pi = np.repeat(np.arange(len(q)), n)
                si = order[np.repeat(lo, n) + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))]
                np.minimum.at(best, pi, _point_seg(q[pi], a[si], b[si]))
    far = np.flatnonzero(best > h / 2)
    if near_only:
        best[far] = np.minimum(best[far], h)
    elif len(far):
        best[far] = _to_segments_all(q[far], a, b)
    return best


def _to_segments_all(q, a, b):
    """Distance of each point of `q` (m, 3) to the nearest segment a[i]-b[i] (closest point on a segment), in blocks."""
    ab = b - a
    l2 = np.maximum((ab * ab).sum(1), 1e-30)
    best = np.full(len(q), np.inf)
    for s in range(0, len(a), 256):
        A, AB = a[None, s:s + 256], ab[None, s:s + 256]
        t = np.clip(((q[:, None, :] - A) * AB).sum(2) / l2[None, s:s + 256], 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(q[:, None, :] - (A + t[..., None] * AB), axis=2).min(1))
    return best


def remember_applied(cand) -> str:
    """After Seam Fit's Apply the result IS the fitted mesh: file the fit under that result's key, so the result's own
    Apply (`uv_hooks.after_apply`) writes exactly this candidate. Returns the key."""
    c = getattr(cand, "fit", None)
    if c is None:
        return ""
    key = uvkeep.result_key(c.points, c.indptr, c.flat)
    uvkeep.remember(replace(c, result_key=key), pin=True)
    APPLIED_EXTRA[key] = int(getattr(cand, "fit_extra", 0))
    return key


# --------------------------------------------------------------------------- UV-only Trials


@dataclass
class UVBase:
    """A mesh with UVs as a UV Trial sees it. Built once per preview."""

    points: np.ndarray
    indptr: np.ndarray
    flat: np.ndarray
    material: np.ndarray | None
    uv: np.ndarray | None  # (corners, 2) of the active layer, or None
    texture_px: float = CHECK_PX
    source: object = None  # uvscore Mesh of your original mesh (Keep numbers, Bake those spots)
    forced: np.ndarray | None = None  # vertex pairs that stay cut (hard edges for a normal bake)
    cache: dict = field(default_factory=dict)

    @property
    def faces(self) -> int:
        return len(self.indptr) - 1

    @property
    def corner_face(self) -> np.ndarray:
        if "corner_face" not in self.cache:
            self.cache["corner_face"] = np.repeat(np.arange(self.faces), np.diff(self.indptr))
        return self.cache["corner_face"]


def make_uv_base(points, indptr, flat, material=None, uv=None, texture_px=CHECK_PX, source=None, forced=None) -> UVBase:
    p = np.ascontiguousarray(points, np.float64).reshape(-1, 3)
    m = None if material is None else np.asarray(material, np.int64)
    u = None if uv is None else np.asarray(uv, np.float64).reshape(-1, 2)
    return UVBase(p, np.asarray(indptr, np.int64), np.asarray(flat, np.int64), m, u, float(texture_px or CHECK_PX), source, forced)


@dataclass
class UVCand:
    uv: np.ndarray | None
    exact: bool = True
    refusal: str = ""
    note: str = ""
    numbers: dict = field(default_factory=dict)  # uvscore numbers of the candidate (card names)
    before: dict = field(default_factory=dict)  # the same of the mesh as it is
    verdict: tuple = ()  # (level, title, reasons)
    chart: np.ndarray | None = None  # per face
    found: np.ndarray | None = None  # (k, 3) spots: the dashed rings
    faces: np.ndarray | None = None  # faces the candidate changed (Bake those spots)
    budget_hit: bool = False
    gain: float = 0.0  # what Keep going compares, higher is better
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return uv_hash(self.uv)


def uv_hash(uv) -> str:
    """Hash of UVs as a mesh stores them (float32), so preview and Apply compare like for like."""
    return "" if uv is None else hashlib.sha256(np.ascontiguousarray(uv, np.float32).tobytes()).hexdigest()


def _check(base: UVBase, uv, keep: bool, **kw):
    """Generator: UV Check numbers of `uv` on the base mesh, as a dict, in slices."""
    res = uvscore.mesh(base.points, base.indptr, base.flat, uv, base.material)
    src = base.source if keep else None
    tex = float(base.texture_px)
    got = yield from uvscore.check_steps(res, src, texture_px=tex, **{k: v for k, v in kw.items() if v is not None})
    return uvscore.as_dict(got.numbers), np.asarray(got.straddle)


class UVTrial:
    """One UV tool's maths. `run(base, settings, prev)` is a generator returning a `UVCand`; `prev` is the candidate to
    start from (Again, Keep going) or None."""

    name = ""
    handle = ""  # the dial's setting (0..1), "" for none
    words = ("0", "1")
    defaults: dict = {}
    mode = "new"  # the verdict's mode: "new" or "keep"
    picks = False  # True: clicks on edges change `settings["cuts"]`
    card = (("space_used_pct", "space used", "{:.0f} %", 1), ("stretch_med", "stretch", "{:.2f}", -1), ("islands", "islands", "{:.0f}", -1))

    def check(self, base: UVBase) -> str:
        return "" if base.faces else "The mesh has no faces."

    def value_text(self, settings: dict) -> str:
        return ""

    def run(self, base: UVBase, settings: dict, prev: UVCand | None = None):
        raise NotImplementedError
        yield

    def gained(self, before: UVCand, after: UVCand) -> bool:
        return after.gain > before.gain + 1e-9


def measure(trial: UVTrial, base: UVBase, cand: UVCand):
    """Generator: the card's numbers before and after, and the verdict."""
    if cand.refusal or cand.uv is None:
        return cand
    keep = trial.mode == "keep" and base.source is not None
    kw = cand.extra.get("check", {})
    if "numbers" not in base.cache and base.uv is not None:
        base.cache["numbers"], base.cache["straddle"] = yield from _check(base, base.uv, keep)
    cand.before = dict(base.cache.get("numbers", {}))
    cand.numbers, _straddle = yield from _check(base, cand.uv, keep, **kw)
    src = None
    if keep:
        src_numbers = base.cache.get("source_numbers")
        if src_numbers is None and base.source is not None:
            s = base.source
            sb = make_uv_base(s.points, s.indptr, s.flat, s.material, s.uv, base.texture_px)
            src_numbers, _ = yield from _check(sb, s.uv, False)
            base.cache["source_numbers"] = src_numbers
        src = src_numbers
    cand.verdict = uvscore.uv_verdict(cand.numbers, trial.mode, source=src)
    return cand


def card_numbers(trial: UVTrial, cand: UVCand | None) -> list:
    """[(words, sign)] for the card: before -> after with a better/worse sign."""
    out = []
    for key, words, fmt, better in trial.card:
        if cand is None or key not in cand.numbers:
            out.append(("...", 0))
            continue
        b, a = cand.before.get(key), cand.numbers[key]
        if a != a:  # not measured (NaN)
            continue
        if b is None or b != b:
            out.append((f"{words} {fmt.format(a)}", 0))
            continue
        sign = 0 if abs(a - b) < 1e-6 else (1 if (a - b) * better > 0 else -1)
        out.append((f"{words} {fmt.format(b)} -> {fmt.format(a)}", sign))
    return out


def _same_pairs(a, b) -> bool:
    a = np.zeros((0, 2), np.int64) if a is None else np.asarray(a, np.int64).reshape(-1, 2)
    b = np.zeros((0, 2), np.int64) if b is None else np.asarray(b, np.int64).reshape(-1, 2)
    return a.shape == b.shape and np.array_equal(a, b)


def _poll_new(run):
    try:
        while not run.done:
            yield run.step()
    finally:
        if not run.done:  # dropped unfinished (a drag, Again, Esc, Cancel): its work stops too
            run.stop()
    if run.error is not None:
        return None, f"New UVs could not be made ({run.error})."
    return run.result, ""


def _new_cand(run, result) -> UVCand:
    r = result
    cand = UVCand(np.asarray(r.uv, np.float64), chart=r.chart, note="; ".join(r.messages))
    cand.budget_hit = any(m.startswith("Stopped after") for m in r.messages)
    cand.gain = float(r.used)
    cand.extra = {"run": run, "result": r, "check": dict(seen_face=r.seen_face if run.sight is not None else None, seen_edge_px=r.seen_edge_px,
                                                         col=r.col, row=r.row, on_chart=r.on_chart, density=r.density)}
    return cand


NEW_SLICE = 0.0005  # seconds New's own loop runs per step: the caller's slice decides the rest
STRETCH_DIAL = 25.0  # percent at the dial's end (the popover's soft max); the popover itself has no hard max


class NewUVs(UVTrial):
    """New UVs (SPEC 6) on the mesh, at the dial's Stretch allowed. Again keeps the camera pictures and the charts."""

    name = "New"
    handle = "stretch"
    words = ("1%", "25%")
    defaults = {"stretch": 0.4, "new": None}

    def top(self, settings) -> float:
        s = settings.get("new")
        return max(STRETCH_DIAL, float(getattr(s, "stretch_allowed", 0.0)) if s is not None else 0.0)

    def stretch(self, settings) -> float:
        return max(1.0, float(settings.get(self.handle, 0.4)) * self.top(settings))

    def value_text(self, settings: dict) -> str:
        return f"Stretch allowed {self.stretch(settings):.0f}%"

    def gained(self, before: UVCand, after: UVCand) -> bool:
        """Keep going gains while the space used on the page, measured on the UVs as they are now, rises."""
        a, b = before.numbers.get("space_used_pct"), after.numbers.get("space_used_pct")
        if a is None or b is None:
            return super().gained(before, after)
        return b > a + 1e-9

    def run(self, base, settings, prev=None):
        s = (settings.get("new") or uvnew.Settings()).copy(stretch_allowed=self.stretch(settings), texture_px=base.texture_px)
        if prev is not None and prev.uv is not None and settings.get("keep_going"):
            return (yield from _more_nesting(prev))  # Keep going: the charts stay
        forced = base.forced
        cuts = settings.get("cuts") or ()
        if cuts:
            add = np.asarray(cuts, np.int64).reshape(-1, 2)
            forced = add if forced is None else np.concatenate([np.asarray(forced, np.int64).reshape(-1, 2), add])
        old = (prev or UVCand(None)).extra.get("run") or base.cache.get("new_run")
        if old is not None and old.done and old.error is None and _same_pairs(old.forced, forced):
            run = old.again(**{k: v for k, v in s.__dict__.items()})
        else:
            keep = {"sight": old.sight} if old is not None and old.sight is not None and old.sight.done else None
            run = uvnew.NewUVs(base.points, base.indptr, base.flat, base.material, forced, s, budget=NEW_SLICE, _keep=keep)
        result, error = yield from _poll_new(run)
        if error:
            return UVCand(None, refusal=error)
        base.cache["new_run"] = run
        return _new_cand(run, result)


def islands(indptr, flat, uv) -> tuple:
    """(island per face, count): faces joined across edges whose two sides share the edge's UVs."""
    nf = len(indptr) - 1
    ids, ne = uvnew.edge_ids(indptr, flat)
    cut = uvnew.cut_edges(indptr, flat, uv)
    owner = np.repeat(np.arange(nf), np.diff(indptr))
    order = np.argsort(ids, kind="stable")
    count = np.bincount(ids, minlength=ne)
    first = np.concatenate([[0], np.cumsum(count)])[:-1]
    two = np.flatnonzero((count == 2) & ~cut)
    a, b = owner[order[first[two]]], owner[order[first[two] + 1]]
    label = np.arange(nf)
    while True:  # label spreading: each pass joins along every uncut edge; ends when nothing changes
        low = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    _, out = np.unique(label, return_inverse=True)
    return out.reshape(-1), int(out.max()) + 1 if nf else 0


def _layout(uv, corner_island, fill, page, lo) -> np.ndarray:
    px = np.array([p[0] for p in fill["places"]], np.float64)
    py = np.array([p[1] for p in fill["places"]], np.float64)
    turned = np.array([p[2] for p in fill["places"]], bool)
    sc = float(fill["scale"])
    du = (uv[:, 0] - lo[corner_island, 0]) * sc
    dv = (uv[:, 1] - lo[corner_island, 1]) * sc
    w = fill["w"][corner_island] * sc
    t = turned[corner_island]
    uu = np.where(t, dv, du)
    vv = np.where(t, w - du, dv)
    out = np.empty_like(uv)
    out[:, 0] = (px[corner_island] + uu) / page[0]
    out[:, 1] = (py[corner_island] + vv) / page[1]
    return out


def _pack(base: UVBase, uv, k: int, budget: float, resume=None, passes: int = 0):
    """Generator: pack the islands of `uv` as they are (shape and relative size kept) on one page. Returns (uv, fill)."""
    lab, n = islands(base.indptr, base.flat, uv)
    yield
    ci = lab[base.corner_face]
    lo = np.full((n, 2), np.inf)
    hi = np.full((n, 2), -np.inf)
    np.minimum.at(lo, ci, uv)
    np.maximum.at(hi, ci, uv)
    w, h = np.maximum(hi[:, 0] - lo[:, 0], 1e-9), np.maximum(hi[:, 1] - lo[:, 1], 1e-9)
    area = np.bincount(lab, weights=uvnew.face_areas(base.points, base.indptr, base.flat), minlength=n)
    page = (int(base.texture_px), int(base.texture_px))
    r = resume or {}
    stage = uvpack.FillPage(w.tolist(), h.tolist(), [True] * n, np.maximum(area, 1e-12).tolist(), page, k, order=r.get("order"),
                            policy=int(r.get("policy", 0)), scale=float(r.get("scale", 0.0)), passes=passes, budget=budget)
    try:
        while not stage.done:
            stage.step()
            yield
    finally:
        if not stage.done:  # dropped unfinished: the page search stops too
            stage.stop()
    fill = stage.result
    if fill is None:
        return None, None
    fill = dict(fill, w=w)
    return _layout(np.asarray(uv, np.float64), ci, fill, page, lo), fill


def _more_nesting(cand):
    """Keep going for New: the charts and their box layout are as full as this run makes them, so it ends at once with
    the same candidate and says so (Again with another Stretch allowed or New page layout's Keep going pack again)."""
    yield
    if "Keep going found no fuller page." not in cand.note:
        cand.note = (cand.note + "; " if cand.note else "") + "Keep going found no fuller page."
    return cand


CLEAN = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4)  # "clean down to" sizes the Gap dial steps through


class PageLayout(UVTrial):
    """New page layout: the islands as they are, packed again; Keep going tries other orders and keeps the best."""

    name = "New page layout"
    handle = "gap"
    words = ("small gap", "big gap")
    defaults = {"gap": 0.4, "time": 2.0, "keep_going": False}
    card = (("space_used_pct", "space used", "{:.0f} %", 1), ("islands", "islands", "{:.0f}", 0), ("overlap_pct", "used twice", "{:.2f} %", -1))

    def clean_px(self, settings, texture_px) -> int:
        sizes = [c for c in CLEAN if c <= texture_px] or [4]
        i = min(len(sizes) - 1, int(round(float(settings.get(self.handle, 0.4)) * (len(sizes) - 1))))
        return sizes[i]

    def value_text(self, settings: dict) -> str:
        return f"clean down to {self.clean_px(settings, settings.get('texture_px', CHECK_PX))} px"

    def check(self, base):
        if base.uv is None:
            return "This mesh has no UVs to lay out. New makes some."
        return super().check(base)

    def run(self, base, settings, prev=None):
        clean = self.clean_px(dict(settings, texture_px=base.texture_px), base.texture_px)
        k = uvnew.mip_level(base.texture_px, clean)
        start = prev.uv if prev is not None and prev.uv is not None else base.uv
        resume = prev.extra.get("fill") if prev is not None and settings.get("keep_going") else None
        uv, fill = yield from _pack(base, start, k, float(settings.get("time", 2.0)), resume, passes=-1 if settings.get("keep_going") else 0)
        if uv is None:
            return UVCand(None, refusal="The islands do not fit on the page with this gap. Pick a smaller gap.")
        cand = UVCand(uv, gain=float(fill["used"]), budget_hit=bool(fill.get("budget_hit")), extra={"fill": fill})
        gap = uvnew.gap_texels(base.texture_px, clean)
        cand.note = f"gap {gap} px, clean down to {clean} px ({k} level{'s' if k != 1 else ''})"
        if cand.budget_hit:
            cand.note += ". " + TIME_WORDS.format(t=float(settings.get("time", 2.0)), what="the layout")
        if prev is not None and prev.gain >= cand.gain - 1e-9 and settings.get("keep_going"):
            prev.note = "Keep going found no fuller page; the best layout is kept."
            return prev
        return cand


class CutHere(NewUVs):
    """Cut here / Join here: click an edge to cut it (click it again to join it back), then New UVs run with the cuts."""

    name = "Cut here"
    picks = True
    defaults = {"stretch": 0.4, "new": None, "cuts": ()}

    def check(self, base):
        return super().check(base)


def toggle_cut(cuts: tuple, a: int, b: int) -> tuple:
    """Cut here / Join here: the edge (a, b) is cut when it was not, joined when it was."""
    e = (min(a, b), max(a, b))
    return tuple(c for c in cuts if tuple(c) != e) if e in {tuple(c) for c in cuts} else tuple(cuts) + (e,)


# --------------------------------------------------------------------------- Bake those spots


def straddlers(base: UVBase):
    """Generator: per face, True when it straddles a seam of your original mesh (no single side holds all its corners)."""
    if "straddle" not in base.cache:
        base.cache["numbers"], base.cache["straddle"] = yield from _check(base, base.uv, True)
    s = np.asarray(base.cache["straddle"])
    return s.astype(bool) if len(s) == base.faces else np.zeros(base.faces, bool)


def _groups(indptr, flat, faces) -> list:
    """Connected groups of `faces` (sharing an edge)."""
    if not len(faces):
        return []
    sub_ptr = np.r_[0, np.cumsum(np.diff(indptr)[faces])]
    sub_flat = np.concatenate([flat[indptr[f]:indptr[f + 1]] for f in faces])
    lab, n = islands(sub_ptr, sub_flat, np.zeros((len(sub_flat), 2)))
    return [np.asarray(faces)[lab == g] for g in range(n)]


def _flatten_group(points, indptr, flat, group) -> tuple:
    """(corner ids, (k, 2) UVs in mesh units) of a small group: projected on its own best plane."""
    corners = np.concatenate([np.arange(indptr[f], indptr[f + 1]) for f in group])
    p = points[flat[corners]]
    c = p - p.mean(axis=0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    uv = c @ vt[:2].T
    normal = T.face_normals(points, [tuple(flat[indptr[f]:indptr[f + 1]]) for f in group]).sum(axis=0)
    if np.dot(np.cross(vt[0], vt[1]), normal) < 0:  # keep the winding the faces have
        uv[:, 1] = -uv[:, 1]
    return corners, uv - uv.min(axis=0)


def _fan(indptr, faces) -> np.ndarray:
    """(k, 3) corner ids of the fan triangles of `faces`."""
    faces = np.asarray(faces, np.int64)
    sizes = np.diff(indptr)[faces]
    n = np.maximum(sizes - 2, 0)
    start = np.repeat(indptr[faces], n)
    j = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n) + 1
    return np.stack([start, start + j, start + j + 1], axis=1)


def _density(base: UVBase) -> float:
    """Texels per mesh unit of the layout as it is (median over faces)."""
    a3 = uvnew.face_areas(base.points, base.indptr, base.flat)
    owner = base.corner_face
    uv = base.uv * base.texture_px
    nxt = np.arange(1, len(base.flat) + 1)
    nxt[base.indptr[1:] - 1] = base.indptr[:-1]
    a2 = np.abs(np.bincount(owner, weights=uv[:, 0] * uv[nxt, 1] - uv[nxt, 0] * uv[:, 1], minlength=base.faces)) / 2.0
    ok = (a3 > 0) & (a2 > 0)
    return float(np.median(np.sqrt(a2[ok] / a3[ok]))) if ok.any() else base.texture_px


def shows_texture(base: UVBase) -> np.ndarray:
    """Per face: True when it takes at least one texel. A face under one texel shows no texture: its own island would
    only take page, so Bake those spots leaves it."""
    if "shows" not in base.cache:
        area = uvnew.face_areas(base.points, base.indptr, base.flat)
        dens = _density(base)
        base.cache["shows"] = area * dens * dens >= 1.0
    return base.cache["shows"]


def _pack_into(page, k, seed, tris, tri_chart, charts):
    """Generator: `uvpack.pack_into` in slices. [(ok, dx, dy, quarter)] per chart."""
    return (yield from uvpack.pack_into_steps(page, k, seed, tris, tri_chart, charts, turn_ok=[True] * charts))


class BakeSpotsUV(UVTrial):
    """Bake those spots: the faces that still straddle a seam each get a small new island in the page's free space, and
    the bake fills just those islands from your original mesh. The rest of the texture is untouched."""

    name = "Bake those spots"
    handle = "share"
    words = ("worst", "all")
    mode = "keep"
    defaults = {"share": 1.0, "keep_going": False}
    card = (("straddle_pct", "across seams", "{:.1f} %", -1), ("space_used_pct", "space used", "{:.0f} %", 1))

    def value_text(self, settings):
        return f"faces {100 * float(settings.get(self.handle, 1.0)):.0f}%"

    def check(self, base):
        if base.uv is None:
            return "This mesh has no UVs. New makes some, then bake."
        if base.source is None:
            return "Bake those spots needs your original mesh: make this result from it with Start first."
        return super().check(base)

    def run(self, base, settings, prev=None):
        straddle = yield from straddlers(base)
        done = prev.extra.get("baked", np.zeros(0, np.int64)) if prev is not None and settings.get("keep_going") else np.zeros(0, np.int64)
        todo = np.setdiff1d(np.flatnonzero(straddle), done)
        area = uvnew.face_areas(base.points, base.indptr, base.flat)
        dens = _density(base)
        todo = todo[shows_texture(base)[todo]]
        if not len(todo):
            return UVCand(None, refusal="No faces cross a seam where it shows: nothing to bake.")
        todo = todo[np.argsort(-area[todo], kind="stable")]  # most damage first: the biggest faces across a seam
        take = todo[: max(1, int(math.ceil(float(settings.get(self.handle, 1.0)) * len(todo))))]
        uv0 = prev.uv if prev is not None and settings.get("keep_going") and prev.uv is not None else base.uv
        groups = _groups(base.indptr, base.flat, take)
        yield
        tris, tri_chart, placed_corners = [], [], []
        where = np.full(len(base.flat), -1, np.int64)  # corner -> row of its group's local UVs
        for g, group in enumerate(groups):
            corners, local = _flatten_group(base.points, base.indptr, base.flat, group)
            local = local * dens
            placed_corners.append((corners, local))
            where[corners] = np.arange(len(corners))
            fan = _fan(base.indptr, group)  # triangles in texels, for the packer
            tris += local[where[fan]].reshape(-1).tolist()
            tri_chart += [g] * len(fan)
            if g % 32 == 31:
                yield
        keep = np.ones(base.faces, bool)
        keep[take] = False
        seed = (uv0 * base.texture_px)[_fan(base.indptr, np.flatnonzero(keep))].reshape(-1).tolist()
        yield
        k = uvnew.mip_level(base.texture_px)
        page = (int(base.texture_px), int(base.texture_px))
        got = yield from _pack_into(page, k, seed, tris, tri_chart, len(groups))
        uv = np.array(uv0, np.float64)
        baked, missed = [], 0
        for (corners, local), group, (ok, dx, dy, q) in zip(placed_corners, groups, got):
            if not ok:
                missed += len(group)
                continue
            x, y = local[:, 0].copy(), local[:, 1].copy()
            for _ in range(q):  # (x, y) -> (-y, x), as pack_into reports it
                x, y = -y, x
            uv[corners, 0] = (x + dx) / page[0]
            uv[corners, 1] = (y + dy) / page[1]
            baked.extend(group.tolist())
        if not baked:
            return UVCand(None, refusal="The page has no free space left for new islands. Use New page layout or a bigger texture.")
        baked = np.asarray(sorted(set(baked) | set(done.tolist())), np.int64)
        cand = UVCand(uv, faces=np.asarray(sorted(set(baked.tolist()) - set(done.tolist())), np.int64), gain=float(len(baked)))
        cand.extra = {"baked": baked}
        centre = np.array([base.points[base.flat[base.indptr[f]:base.indptr[f + 1]]].mean(axis=0) for f in take[:2000]])
        cand.found = centre
        left = len(todo) - len(cand.faces)
        cand.note = f"{len(cand.faces)} faces get their own island and are baked" + (f"; {left} left" if left else "") + (f"; {missed} did not fit on the page" if missed else "")
        return cand


TRIALS = {"new": NewUVs, "layout": PageLayout, "cut": CutHere, "bake": BakeSpotsUV}


_STEP = {}  # generator id -> its longest step so far (s), so a slice stops before a step would overrun it


def drive(work, budget: float = 0.004):
    """Run a generator for at most about `budget` seconds: it stops before the next step when the longest step so far
    would not fit any more (one step always runs). Returns (done, value)."""
    start = time.perf_counter()
    end = start + budget
    longest = _STEP.get(id(work), 0.0)
    now = start
    while True:
        try:
            next(work)
        except StopIteration as fin:
            _STEP.pop(id(work), None)
            return True, fin.value
        t = time.perf_counter()
        longest = max(longest, t - now)
        now = t
        if now + longest >= end:
            _STEP[id(work)] = longest
            if len(_STEP) > 64:  # generators dropped without finishing: forget the oldest
                _STEP.pop(next(iter(_STEP)))
            return False, None


def finish(work):
    """Run a generator to the end (F9, scripts, tests)."""
    return T.finish(work)
