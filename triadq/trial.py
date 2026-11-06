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


def make_base(co, faces, scope=None, stay=None, project=None, mirror=None, pick=None) -> Base:
    """A Base with the usual holds: corners on a border, a hard edge or a non-quad (`organize.pinned`), on the mirror
    plane, and every corner a face outside the scope uses, so the edge of the scope never moves."""
    from .organize import pinned

    co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
    faces = [tuple(int(v) for v in f) for f in faces]
    scope = np.ones(len(faces), dtype=bool) if scope is None else np.asarray(scope, dtype=bool)
    held = pinned(co, faces, plane=mirror) if len(faces) else np.zeros(len(co), dtype=bool)
    if stay is not None:
        held |= np.asarray(stay, dtype=bool)
    for f in np.flatnonzero(~scope).tolist():
        held[list(faces[f])] = True
    if mirror is not None:
        axis, offset = mirror
        size = float(np.ptp(co, axis=0).max()) if len(co) else 1.0
        held |= np.abs(co[:, axis] - offset) < 1e-6 * (size or 1.0)
    return Base(co, faces, scope, held, project, mean_edge(co, [faces[f] for f in np.flatnonzero(scope).tolist()]), mirror, dict(pick or {}))


def mean_edge(co, faces) -> float:
    if not faces:
        return 1.0
    a = np.concatenate([np.asarray(f) for f in faces])
    b = np.concatenate([np.roll(np.asarray(f), -1) for f in faces])
    length = float(np.linalg.norm(co[a] - co[b], axis=1).mean())
    return length if length > 0.0 else 1.0


def edges_of(faces) -> np.ndarray:
    """(e, 2) unique edges of the faces, smaller corner first."""
    if not faces:
        return np.zeros((0, 2), dtype=np.int64)
    a = np.concatenate([np.asarray(f, dtype=np.int64) for f in faces])
    b = np.concatenate([np.roll(np.asarray(f, dtype=np.int64), -1) for f in faces])
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
    lines: np.ndarray | None = None  # (k, 2, 3) lines the tool keeps by meaning (hard edges, panel borders): the pink layer


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

    def check(self, base: Base) -> str:
        """"" when the tool can run here, else the message the card shows instead."""
        return ""

    def run(self, base: Base, settings: dict, rough: bool = False):
        """Generator: yield a progress 0..1 about every few milliseconds of work, return a Candidate."""
        raise NotImplementedError
        yield 0.0


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
    before, after = base.cache["pole_points"], _pole_points(co, cand.faces)
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
    cand.arrows = np.zeros((0, 2, 3))
    return cand


def blend(base: Base, full: Candidate, s: float) -> Candidate:
    """An instant stand-in at strength `s`: each corner `s` of the way to where the full-strength run put it."""
    co = base.co + float(s) * (np.asarray(full.co) - base.co)
    return diff(base, Candidate(co, strength=float(s), exact=False, held=list(full.held), note=full.note, lines=full.lines))


def _tidiness(co, faces) -> float:
    from . import patterns

    return float(patterns.score(patterns.measure(co, faces))["score"])


def measure(base: Base, cand: Candidate, stages=("poles", "move", "look"), look=None):
    """Generator: fill `cand.numbers` cheapest first, yielding between stages. `look(co, faces)` is a generator
    returning the look change in percent against your original mesh (the camera check), or None to skip it."""
    if cand.refusal or cand.co is None:
        return cand
    faces = base.faces if cand.faces is None else cand.faces
    if "poles" in stages:
        if "poles" not in base.cache:
            base.cache["poles"] = int(base.topo.poles.sum())
        after = base.cache["poles"] if cand.faces is None else int(len(_pole_points(cand.co, faces)))
        cand.numbers["poles"] = (base.cache["poles"], after)
        yield
    if "faces" in stages:
        cand.numbers["faces"] = (len(base.faces), len(faces))
    if "move" in stages and cand.moved is not None:
        cand.numbers["move"] = float(cand.moved.max()) if len(cand.moved) else 0.0
    if "tidy" in stages:
        if "tidy" not in base.cache:
            base.cache["tidy"] = _tidiness(base.co, base.faces)
            yield
        cand.numbers["tidy"] = (base.cache["tidy"], _tidiness(cand.co, faces))
        yield
    if "look" in stages and look is not None:
        if "look" not in base.cache:
            base.cache["look"] = yield from look(base.co, base.faces)
        after = yield from look(np.asarray(cand.co, dtype=np.float64), faces)
        cand.numbers["look"] = (base.cache["look"], after)
    return cand


def number_text(key: str, value, cap: float | None = None) -> tuple[str, int]:
    """(words, sign) of one card number: sign 1 better (up triangle), -1 worse (down triangle), 0 the same. A tool's
    own number is stored already worded, as (words, sign)."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        return value
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
    return f"{key} {value}", 0


def readout(cand: Candidate | None, trial: Trial, settings: dict, status: str, cap: float | None = None):
    """The card's lines: (title line, [(number words, sign)] up to three, third line)."""
    handle = trial.handles[0] if trial.handles else ""
    value = f"   {handle.title()} {settings.get(handle, 0.0):.2f}" if handle else ""
    if cand is not None and cand.refusal:
        return f"{trial.name}", [], cand.refusal
    title = f"{trial.name}{value}   {status}"
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
