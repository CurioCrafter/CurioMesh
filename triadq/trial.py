# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Look First, the pure half: what a Tidy Tool works on, what it proposes, and the numbers the preview card shows.

A tool is a `Trial` subclass. `run(base, settings)` is a generator that yields a progress 0..1 after a few milliseconds
of work and returns a `Candidate`: new corner positions (or new faces), never an edited mesh. The viewport side
(`tidylive.py`) pumps that generator in short slices, draws the candidate as a ghost and applies it only on Enter.
`diff` fills in what moved, `blend` makes an instant stand-in while a handle is dragged, `measure` adds the card's
numbers cheapest first, and `readout` turns them into the card's lines. No bpy, no globals: the same generator run to
the end (`finish`) is what F9 and scripts get.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

ARROWS = 300  # most move arrows drawn, biggest first
ARROW_LEAST = 0.15  # of an edge: smaller moves get no arrow


@dataclass
class Base:
    """The result as it is, and what a tool may change. Built once per preview and shared by every candidate."""

    co: np.ndarray  # (n, 3) corners of the result, object space
    faces: list  # faces of the result (tuples of corner numbers)
    scope: np.ndarray  # bool per face: what the tool may change (Work On)
    stay: np.ndarray  # bool per corner: held where it is (borders, hard edges, the scope's edge, the mirror plane)
    project: Callable | None  # (k, 3) points -> nearest points on your original mesh; None keeps them as they are
    edge: float  # mean edge length in scope
    mirror: tuple | None = None  # (axis, offset) when the result is mirrored
    pick: dict = field(default_factory=dict)  # what the user picked: {"strip": ...} / {"pole": v} / {}
    cache: dict = field(default_factory=dict)  # numbers of the result as it is, worked out once
    facing: Callable | None = None  # (k, 3) points -> unit normals of your original at the nearest points, or None
    labels: dict = field(default_factory=dict)  # per-face "chart" (UV island) and "piece" (part), per-corner "guide"
    kinds: np.ndarray | None = None  # per face: material and UV island together; tools never join faces of two kinds

    @property
    def topo(self):
        if "topo" not in self.cache:
            from .patterns import Topology

            self.cache["topo"] = Topology(self.co, self.faces)
        return self.cache["topo"]

    @property
    def free(self) -> np.ndarray:
        """Corners a tool may move: used by a face in scope and not held."""
        if "free" not in self.cache:
            used = np.zeros(len(self.co), dtype=bool)
            for f in np.flatnonzero(self.scope).tolist():
                used[list(self.faces[f])] = True
            self.cache["free"] = used & ~self.stay
        return self.cache["free"]


def make_base(co, faces, scope=None, stay=None, project=None, mirror=None, pick=None, facing=None, labels=None, kinds=None) -> Base:
    """A Base with the usual holds: corners on a border, a hard edge or a non-quad (`organize.pinned`), on the mirror
    plane, and every corner a face outside the scope uses, so the edge of the scope never moves."""
    return finish(make_base_steps(co, faces, scope, stay, project, mirror, pick, facing, labels, kinds))


def make_base_steps(co, faces, scope=None, stay=None, project=None, mirror=None, pick=None, facing=None, labels=None, kinds=None):
    """`make_base` as a generator: yields (nothing) between its pieces, so no slice runs long. Returns the Base."""
    from .organize import half_edges, pinned

    yield
    co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
    faces = _int_faces(faces)
    scope = np.ones(len(faces), dtype=bool) if scope is None else np.asarray(scope, dtype=bool)
    yield
    half = half_edges(faces)
    yield
    held = pinned(co, faces, plane=mirror, half=half) if len(faces) else np.zeros(len(co), dtype=bool)
    yield
    if stay is not None:
        held |= np.asarray(stay, dtype=bool)
    fid, a, b = half
    if not scope.all():
        held[a[~scope[fid]]] = True
    if mirror is not None:
        axis, offset = mirror
        size = float(np.ptp(co, axis=0).max()) if len(co) else 1.0
        held |= np.abs(co[:, axis] - offset) < 1e-6 * (size or 1.0)
    inside = scope[fid]
    length = float(np.linalg.norm(co[a[inside]] - co[b[inside]], axis=1).mean()) if inside.any() else 1.0
    edge = length if length > 0.0 else 1.0
    kinds = None if kinds is None or len(kinds) != len(faces) else np.asarray(kinds)
    return Base(co, faces, scope, held, project, edge, mirror, dict(pick or {}), facing=facing, labels=dict(labels or {}), kinds=kinds)


def unfold(base: Base, co: np.ndarray) -> np.ndarray:
    """Corners of faces that turned over go back where they were, pass by pass, until no face is turned over."""
    from .organize import half_edges

    fid, corner, _nxt = half_edges(base.faces)
    before = face_normals(base.co, base.faces)
    while True:  # each pass puts at least one face's corners back, so it ends
        flipped = (before * face_normals(co, base.faces)).sum(axis=1) < 0.0
        back = np.unique(corner[flipped[fid]])
        back = back[(co[back] != base.co[back]).any(axis=1)]
        if not len(back):
            return co
        co[back] = base.co[back]


def _int_faces(faces) -> list:
    """Faces as tuples of plain ints, read in one pass rather than corner by corner."""
    from itertools import accumulate, chain

    sizes = list(map(len, faces))
    flat = np.fromiter(chain.from_iterable(faces), dtype=np.int64, count=sum(sizes)).tolist()
    ends = list(accumulate(sizes))
    return [tuple(flat[e - n : e]) for e, n in zip(ends, sizes)]


def mean_edge(co, faces) -> float:
    if not faces:
        return 1.0
    from .organize import half_edges

    _f, a, b = half_edges(faces)
    length = float(np.linalg.norm(co[a] - co[b], axis=1).mean())
    return length if length > 0.0 else 1.0


def edges_of(faces) -> np.ndarray:
    """(e, 2) unique edges of the faces, smaller corner first."""
    if not faces:
        return np.zeros((0, 2), dtype=np.int64)
    from .organize import half_edges

    _f, a, b = half_edges(faces)
    return np.unique(np.sort(np.stack((a, b), axis=1), axis=1), axis=0)


@dataclass
class Candidate:
    """What a tool proposes. Drawn by the preview; kept only on Apply, and only when `exact`."""

    co: np.ndarray | None  # new positions of the same corners, or the corners of `faces`
    faces: list | None = None  # new faces when the topology changes, else None
    parent: np.ndarray | None = None  # old face of each new face (per-face data follows it)
    strength: float = 1.0
    exact: bool = True  # False: a quick or blended stand-in, never applied
    moved: np.ndarray | None = None  # per corner: distance moved / edge length
    arrows: np.ndarray | None = None  # (k, 2, 3) from -> to, biggest moves first
    poles_gone: np.ndarray | None = None  # (g, 3)
    poles_new: np.ndarray | None = None  # (h, 3)
    held: list = field(default_factory=list)  # [(point, why)]
    numbers: dict = field(default_factory=dict)  # filled by `measure`: {"poles": (a, b), "faces": (a, b), ...}
    note: str = ""
    refusal: str = ""
    found: np.ndarray | None = None  # (k, 3) spots the tool found to work on: the dashed Found rings
    tint: np.ndarray | None = None  # per corner -1..1, drawn instead of the move wash: blue grew, orange shrank
    rows: list = field(default_factory=list)  # a report's lines (Preview Tidy): {"key", "words", "spots", "cost", ...}
    uv: object = None  # a tool's UV payload (new UVs): carried through `autofix.propose` untouched, for `Trial.after_commit`
    uv_cand: object = None  # the tool's own candidate that made `uv` (what its `after_commit` bakes from)
    lines: np.ndarray | None = None  # the pink Kept lines layer: (k, 2, 3) float points, or (k, 2) int corner pairs of the candidate (they follow its corners)


@dataclass
class Spot:
    """One place a tool can fix by itself (Auto Fix, "Fix every spot like this"): bench/tools_design/AUTO_FIX.md 2.1."""

    kind: str  # a key of the tool's `kinds`: "slanted"
    faces: np.ndarray  # face ids of the region the fix may change (its edge is held, like Work On's)
    at: np.ndarray  # (3,) where Show frames and the dial opens
    bad: float  # the tool's own number here, lower is better: ranks the spots, worst first
    pick: dict = field(default_factory=dict)  # settings a picking tool needs for this spot: {"strip": e}, {"row": "pairs"}


class Trial:
    """One Tidy Tool's maths. Subclass it, set the class fields, write `check` and `run`."""

    name = ""
    group = "RELAX"  # RELAX, LOOPS, POLES, CLEANUP, HARD
    handles: tuple = ("strength",)  # the first one is the dial; () for none
    words = ("0", "1")  # the dial's two end words
    defaults: dict = {"strength": 0.6}
    blendable = True  # positions only, and strength s about lerp(before, full, s): the dial is instant
    layers = ("scope", "ghost", "wash", "arrows", "poles", "held")
    stages = ("poles", "move", "look")  # card numbers, most important first (up to three are shown)
    toggles: tuple = ()  # ((setting, words), ...): on/off chips on the card's fourth row
    legend = ""  # the fourth row's words when the tool tints by something other than how far corners moved
    picks: tuple = ()  # what you click before it can run, in order ("pole", "partner"); each is a setting, -1 until clicked
    asks: dict = {}  # pick -> what the header and the card ask for: {"pole": "click a pole"}
    stay_open = False  # True: after Enter the tool opens again on the new result (one Undo step per Enter)
    # Auto Fix (AUTO_FIX.md 2.1): a tool that fills these in also fixes its spots by itself
    kinds: tuple = ()  # ((key, one, many), ...): ("slanted", "patch of slanted quads", "patches of slanted quads")
    auto: bool = True  # False: manual only (it needs the user's intent; there is no defect to find)
    own: str = ""  # this tool's own measure, lower is better ("square", "wave", "bump", ...); "" until it has `own_number`
    least: float = 0.0  # the smallest drop of the own number that counts as better: a tolerance against noise, not a cap
    changes_faces: bool = False  # True when `run` returns new faces (with `parent`)
    splits_faces: bool = False  # True: the tool splits faces (Seam Fit): the faces made by a split are expected, see `bad_by_parent`

    def spots(self, base: Base):
        """Generator: yield a progress 0..1, return the list of `Spot`s in `base.scope` this tool would fix. Touching
        spots are one spot, so two never share a corner. Never cut to a count."""
        return []
        yield 0.0

    def own_number(self, base: Base, co, faces, spot: Spot) -> float:
        """The own number on `spot.faces` of the mesh (co, faces), lower is better. `base` is the mesh before the fix
        (its cache, project and labels); `spot.faces` are ids into `faces`."""
        return float(spot.bad)

    def prepare(self, base: Base):
        """Generator: slow reads a tool needs before `check`, in small steps (the Session yields between them)."""
        yield from ()

    def check(self, base: Base) -> str:
        """"" when the tool can run here, else the message the card shows instead."""
        return ""

    def waiting(self, settings: dict) -> str:
        """The first pick not made yet, or ""."""
        return next((k for k in self.picks if int(settings.get(k, -1)) < 0), "")

    def targets(self, base: Base, settings: dict, kind: str):
        """(corner ids you can click for pick `kind`, a short label each). None: click anywhere and `pick_at` decides."""
        return None

    def pick_at(self, base: Base, settings: dict, kind: str, face: int, point) -> int:
        """The pick for a click on `face` at `point` (object space), or -1 for nothing there."""
        return -1

    def picked_lines(self, base: Base, settings: dict) -> np.ndarray:
        """(k, 2, 3) segments outlining what the picks in `settings` pick (drawn orange; white for a hover)."""
        return np.zeros((0, 2, 3))

    def value_text(self, value: float) -> str:
        """The dial's value on the card: "Strength 0.60". A tool whose dial means something else says it its way."""
        return f"{self.handles[0].title()} {value:.2f}" if self.handles else ""

    def run(self, base: Base, settings: dict, rough: bool = False):
        """Generator: yield a progress 0..1 about every few milliseconds of work, return a Candidate."""
        raise NotImplementedError
        yield 0.0

    def after_commit(self, obj, cand: Candidate) -> None:
        """Called once by `tidylive.commit` after the candidate's geometry is written to the applied mesh `obj`: a tool
        that carries more than geometry (`cand.uv`) writes it here (bake). Never called for a refused or unapplied one."""


def run_after_commit(trial, obj, cand) -> bool:
    """`trial.after_commit(obj, cand)` once, for a kept candidate on an applied mesh `obj` (None: nothing applied yet,
    so nothing to bake). A refused or stand-in candidate never gets here. True when the hook was called."""
    if trial is None or obj is None or cand is None or cand.refusal or cand.co is None or not cand.exact:
        return False
    trial.after_commit(obj, cand)
    return True


def rebase(base: Base, cand: Candidate) -> Base:
    """A Base for the candidate's mesh, with what the tool did not change carried over: project, facing, mirror, the
    per-face labels and kinds (through `parent`), the scope mapped through `parent`, and the corners' extra holds when
    the corners are the same ones. For a tool that changed the faces, before the next spot is worked on."""
    co = base.co if cand.co is None else np.asarray(cand.co, dtype=np.float64)
    if cand.faces is None:
        new = Base(co.copy(), base.faces, base.scope.copy(), base.stay.copy(), base.project, base.edge, base.mirror, facing=base.facing, labels=dict(base.labels), kinds=base.kinds)
        return new
    faces = list(cand.faces)
    parent = None if cand.parent is None else np.asarray(cand.parent, dtype=np.int64)
    ok = parent is not None and len(parent) == len(faces) and len(base.faces) > 0
    take = np.clip(parent, 0, max(len(base.faces) - 1, 0)) if ok else None

    def per_face(values):
        values = np.asarray(values)
        return values[take] if ok and len(values) == len(base.faces) else None

    scope = per_face(base.scope)
    if scope is not None:
        scope = scope & (parent >= 0)
    labels = {}
    for key, values in base.labels.items():
        if values is None:
            continue
        mapped = per_face(values) if len(values) == len(base.faces) else (values if len(values) == len(co) else None)
        if mapped is not None:
            labels[key] = mapped
    kinds = None if base.kinds is None else per_face(base.kinds)
    stay = base.stay if len(base.stay) == len(co) else None
    return make_base(co, faces, scope=scope, stay=stay, project=base.project, mirror=base.mirror, facing=base.facing, labels=labels, kinds=kinds)


def finish(work):
    """Run a generator to the end and return its value (F9, scripts and tests)."""
    while True:
        try:
            next(work)
        except StopIteration as done:
            return done.value


def _pole_points(co, faces) -> np.ndarray:
    from .patterns import Topology

    topo = Topology(co, faces)
    return topo.points[np.flatnonzero(topo.poles)]


def diff(base: Base, cand: Candidate) -> Candidate:
    """Fill in how far each corner moved, the biggest moves as arrows, and the poles that went or came."""
    return finish(diff_steps(base, cand))


def diff_steps(base: Base, cand: Candidate):
    """`diff` as a generator: yields (nothing) between the poles before and the poles after. Returns the candidate."""
    if cand.refusal or cand.co is None:
        return cand
    co = np.asarray(cand.co, dtype=np.float64)
    if cand.faces is None:  # same corners: moves are per corner; poles cannot change without new faces
        far = np.linalg.norm(co - base.co, axis=1) / base.edge
        cand.moved = far
        big = np.flatnonzero(far > ARROW_LEAST)
        big = big[np.argsort(-far[big])][:ARROWS]
        cand.arrows = np.stack((base.co[big], co[big]), axis=1) if len(big) else np.zeros((0, 2, 3))
        cand.poles_gone = cand.poles_gone if cand.poles_gone is not None else np.zeros((0, 3))
        cand.poles_new = cand.poles_new if cand.poles_new is not None else np.zeros((0, 3))
        return cand
    if "pole_points" not in base.cache:
        base.cache["pole_points"] = _pole_points(base.co, base.faces)
        yield
    before, after = base.cache["pole_points"], _pole_points(co, cand.faces)
    yield
    tol = 1e-4 * base.edge

    def keys(points):
        return [tuple(k) for k in np.round(points / tol).astype(np.int64).tolist()]

    kb, ka = keys(before), keys(after)
    sb, sa = set(kb), set(ka)
    cand.poles_gone = before[np.array([k not in sa for k in kb], dtype=bool)].reshape(-1, 3)
    cand.poles_new = after[np.array([k not in sb for k in ka], dtype=bool)].reshape(-1, 3)
    if len(co) == len(base.co):
        cand.moved = np.linalg.norm(co - base.co, axis=1) / base.edge
    else:
        cand.moved = np.zeros(len(co))
    if cand.arrows is None:  # a tool may draw its own (Seal the Middle's moves onto the plane and merges)
        cand.arrows = np.zeros((0, 2, 3))
    return cand


def blend(base: Base, full: Candidate, s: float) -> Candidate:
    """An instant stand-in at strength `s`: each corner `s` of the way to where the full-strength run put it."""
    co = base.co + float(s) * (np.asarray(full.co) - base.co)
    return diff(base, Candidate(co, strength=float(s), exact=False, held=list(full.held), note=full.note, lines=full.lines))


def _tidiness_steps(co, faces):
    from . import patterns

    return float(patterns.score((yield from patterns.measure_steps(co, faces)))["score"])


def _tidiness(co, faces) -> float:
    return finish(_tidiness_steps(co, faces))


def face_normals(co, faces) -> np.ndarray:
    """(f, 3) Newell normals (not unit), from the half-edges."""
    from .organize import half_edges

    if not len(faces):
        return np.zeros((0, 3))
    fid, corner, nxt = half_edges(faces)
    co = np.asarray(co, dtype=np.float64)
    c = np.cross(co[corner], co[nxt])
    return np.stack([np.bincount(fid, c[:, k], len(faces)) for k in range(3)], axis=1)


BAD_CHUNK = 1000  # faces checked between two yields of `bad_faces_steps` (about 3 ms)


DEGENERATE = 1e-6  # a piece of a split with less than this share of its parent's area has no direction: it is not "turned over"


def bad_faces(base: Base, co, faces=None, parent=None, splits=False) -> np.ndarray:
    """Per face of (co, faces): turned over against the face of `base` it came from (the same face, or `parent`), or a
    quad folded or concave. `faces=None`: base's faces. The one bad-face test of the Tidy Tools and Auto Fix.
    `splits` (`Trial.splits_faces`): a split can leave a zero-area piece (the cut runs along an edge) whose sign is
    noise; it is not counted as turned over. A real turned or folded face still is."""
    return finish(bad_faces_steps(base, co, faces, parent, splits))


def bad_faces_steps(base: Base, co, faces=None, parent=None, splits=False):
    """`bad_faces` as a generator: yields (nothing) every `BAD_CHUNK` faces, so no slice runs long. Same answer."""
    from .checks import folded_quads

    faces = base.faces if faces is None else faces
    if not len(faces):
        return np.zeros(0, dtype=bool)
    if "normals" not in base.cache:
        base.cache["normals"] = face_normals(base.co, base.faces)
        yield
    ref = base.cache["normals"]
    if faces is not base.faces:
        ok = parent is not None and len(parent) == len(faces) and len(ref)
        ref = ref[np.clip(np.asarray(parent, dtype=np.int64), 0, len(ref) - 1)] if ok else None
    co = np.asarray(co, dtype=np.float64)
    out = np.zeros(len(faces), dtype=bool)
    for s in range(0, len(faces), BAD_CHUNK):
        part = faces[s : s + BAD_CHUNK]
        out[s : s + len(part)] = folded_quads(co, part)
        if ref is not None and len(ref) == len(faces):
            n, r = face_normals(co, part), ref[s : s + len(part)]
            turned = (n * r).sum(axis=1) < 0.0
            if splits:
                turned &= np.linalg.norm(n, axis=1) > DEGENERATE * np.linalg.norm(r, axis=1)
            out[s : s + len(part)] |= turned
        if s + BAD_CHUNK < len(faces):
            yield
    return out


def bad_by_parent(bad: np.ndarray, parent) -> int:
    """Bad faces counted per face they came from: a face split in two that is bad counts once, a good one that gave a
    bad piece counts once (Seam Fit's splits: `Trial.splits_faces`)."""
    bad = np.asarray(bad, dtype=bool)
    if parent is None or len(parent) != len(bad):
        return int(bad.sum())
    parent = np.asarray(parent, dtype=np.int64)
    return int(len(np.unique(parent[bad & (parent >= 0)])) + (bad & (parent < 0)).sum())


def _loops_steps(co, faces, plane):
    from .loopscore import loop_quality_steps

    return float((yield from loop_quality_steps(co, faces, plane))["quality"]) if len(faces) else None


def _loops(co, faces, plane) -> float | None:
    return finish(_loops_steps(co, faces, plane))


def _unsplit(base: Base, cand: Candidate):
    """(base faces, candidate faces) without the faces a split touched, for a tool that splits faces. Both lose the same
    places, so a pole or a Loop Quality point gained or lost at the split edges themselves counts for neither side."""
    parent = np.asarray(cand.parent, dtype=np.int64)
    kids = np.bincount(parent[parent >= 0], minlength=len(base.faces))
    cut = kids > 1
    before = [f for f, c in zip(base.faces, cut.tolist()) if not c]
    after = [f for f, p in zip(cand.faces, parent.tolist()) if p < 0 or not cut[p]]
    return before, after


def measure(base: Base, cand: Candidate, stages=("poles", "move", "look"), look=None, splits=False):
    """Generator: fill `cand.numbers` cheapest first, yielding between stages. `look(co, faces)` is a generator
    returning the look change in percent against your original mesh (the camera check), or None to skip it.
    Auto Fix's stages: "tidy", "loops" (Loop Quality), "bad" (bad faces, `bad_faces`). `splits`: the tool splits faces
    (`Trial.splits_faces`): bad faces count per face they came from, and poles, Tidiness and Loop Quality are read
    without the faces a split touched."""
    if cand.refusal or cand.co is None:
        return cand
    faces = base.faces if cand.faces is None else cand.faces
    cut = splits and cand.faces is not None and cand.parent is not None and len(cand.parent) == len(faces)
    b_faces, a_faces = _unsplit(base, cand) if cut else (base.faces, faces)
    tag = "~" if cut else ""  # a base number worked out without the split faces is kept apart from the whole one
    co_after = np.asarray(cand.co, dtype=np.float64)
    if "poles" in stages:
        if "poles" + tag not in base.cache:
            base.cache["poles" + tag] = int(len(_pole_points(base.co, b_faces))) if cut else int(base.topo.poles.sum())
        after = base.cache["poles" + tag] if cand.faces is None else int(len(_pole_points(cand.co, a_faces)))
        cand.numbers["poles"] = (base.cache["poles" + tag], after)
        yield
    if "faces" in stages:
        cand.numbers["faces"] = (len(base.faces), len(faces))
    if "move" in stages and cand.moved is not None:
        cand.numbers["move"] = float(cand.moved.max()) if len(cand.moved) else 0.0
    if "tidy" in stages:
        if "tidy" + tag not in base.cache:
            base.cache["tidy" + tag] = yield from _tidiness_steps(base.co, b_faces)
            yield
        cand.numbers["tidy"] = (base.cache["tidy" + tag], (yield from _tidiness_steps(cand.co, a_faces)))
        yield
    if "loops" in stages:
        if "loops" + tag not in base.cache:
            base.cache["loops" + tag] = yield from _loops_steps(base.co, b_faces, base.mirror)
            yield
        cand.numbers["loops"] = (base.cache["loops" + tag], (yield from _loops_steps(co_after, a_faces, base.mirror)))
        yield
    if "bad" in stages:
        if "bad" not in base.cache:
            base.cache["bad"] = int((yield from bad_faces_steps(base, base.co)).sum())
        bad = yield from bad_faces_steps(base, co_after, faces if cand.faces is not None else None, cand.parent, splits)
        cand.numbers["bad"] = (base.cache["bad"], bad_by_parent(bad, cand.parent) if cut else int(bad.sum()))
        yield
    if "look" in stages and look is not None:
        if "look" not in base.cache:
            base.cache["look"] = yield from look(base.co, base.faces)
        after = yield from look(co_after, faces)
        cand.numbers["look"] = (base.cache["look"], after)
    return cand


def number_text(key: str, value, cap: float | None = None) -> tuple[str, int]:
    """(words, sign) of one card number: sign 1 better (up triangle), -1 worse (down triangle), 0 the same."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        return value  # a tool's own number, already in words: ("narrowest angle 28 -> 61 degrees", 1)
    if key == "poles":
        a, b = value
        if b < a:
            return f"{a - b} pole{'s' if a - b != 1 else ''} gone", 1
        if b > a:
            return f"{b - a} new pole{'s' if b - a != 1 else ''}", -1
        return f"poles {a:,}", 0
    if key == "faces":
        a, b = value
        return (f"faces {a:,}", 0) if a == b else (f"faces {a:,} -> {b:,}", 0)
    if key == "tidy":
        a, b = value
        return f"Tidiness {a:.0f} -> {b:.0f}", 1 if b > a + 0.5 else -1 if b < a - 0.5 else 0
    if key == "look":
        a, b = value
        change = b - a
        worse = change > 0.5 * (cap if cap is not None else 1.0)
        return f"look change {change:+.1f}%", -1 if worse else 1 if change < -0.05 else 0
    if key == "move":
        return f"most move {value:.2f} quad", 0
    if key == "loops":
        a, b = value
        if a is None or b is None:
            return "Loop Quality -", 0
        return f"Loop Quality {a:.0f} -> {b:.0f}", 1 if b > a + 0.5 else -1 if b < a - 0.5 else 0
    if key == "bad":
        a, b = value
        if b > a:
            return f"{b - a} new bad face{'s' if b - a != 1 else ''}", -1
        return (f"{a - b} bad face{'s' if a - b != 1 else ''} gone", 1) if b < a else ("no new bad faces", 0)
    return f"{key} {value}", 0


def readout(cand: Candidate | None, trial: Trial, settings: dict, status: str, cap: float | None = None):
    """The card's lines: (title line, [(number words, sign)] up to three, third line)."""
    handle = trial.handles[0] if trial.handles else ""
    value = f"   {trial.value_text(float(settings.get(handle, 0.0)))}" if handle else ""
    if cand is not None and cand.refusal:
        return f"{trial.name}", [], cand.refusal
    title = f"{trial.name}{value}   {status}"
    if cand is not None and cand.co is None:  # a report with nothing picked yet: no numbers, its own line
        return title, [], cand.note
    numbers = []
    for key in trial.stages:
        if len(numbers) == 3:
            break
        if cand is None or key not in cand.numbers:
            numbers.append(("...", 0))
        else:
            numbers.append(number_text(key, cand.numbers[key], cap))
    third = cand.note if cand is not None else ""
    return title, numbers, third
