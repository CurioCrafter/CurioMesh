# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Auto Fix, the pure half: every Tidy Tool run by itself on the spots it is made for.

One function, `propose`, runs a tool and returns the guarded result with its numbers. The Look First preview (no
spots: the tool on Work On, as before), the per-tool "Fix every spot like this" button (`EverySpot`) and Auto Fix after
Start (`run`) all call it, so the maths, the guards and the numbers are the same everywhere. `verdict` is the one
accept rule. No bpy, no threads: generators the caller pumps in short slices.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import numpy as np

from . import trial as T
from .organize import MIN_GAIN

WHY = {  # the reasons of AUTO_FIX.md 2.3, in the readout's words
    "no better": "no better",
    "bad faces": "would add bad faces",
    "worse": "the quads would get worse",
    "look": "the look would change too much",
    "held": "could not be worked on",
}
EXTRA = ("poles", "tidy", "loops", "bad")  # measured on top of the tool's own stages: the accept rule needs them
STRENGTH_LEAST = 0.02  # a handle below this moves nothing visible: halving ends there (nothing left to try, not a cap)

FINISHED = "Auto Fix finished: another round found nothing it could make better."
NOTHING = "Auto Fix found nothing to fix."


@dataclass
class Scores:
    """One result, measured. None: not measured (a quick preview, no look check)."""

    tidy: float | None = None
    loops: float | None = None
    poles: int | None = None
    bad: int | None = None
    faces: int | None = None
    look: float | None = None


@dataclass
class Proposal:
    cand: T.Candidate  # the guarded result (positions, or new faces + parent); exact unless rough
    fixed: list = field(default_factory=list)  # [Spot] whose own number got better by `least`
    left: list = field(default_factory=list)  # [(Spot, why)] put back; why is a key of WHY
    before: Scores = field(default_factory=Scores)
    after: Scores = field(default_factory=Scores)
    words: str = ""  # the meter line: "Square Up: 12 spots, Tidiness 56 -> 60, look change +0.1%"
    reason: str = ""  # verdict: "" kept, else a key of WHY
    held: float | None = None  # the handle value it was held back to for the look, or None
    judged: bool = False  # True: spots were given, so rule 1 (some spot better) applies


def _scores(numbers: dict, side: int) -> Scores:
    def get(key):
        value = numbers.get(key)
        return value[side] if isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], str) else None

    return Scores(get("tidy"), get("loops"), get("poles"), get("bad"), get("faces"), get("look"))


def _ceiling(budget, before_look):
    """The most look change allowed: `budget` is a number (percent), a function of the look before, or None."""
    if budget is None:
        return None
    if callable(budget):
        return None if before_look is None else budget(before_look)
    return float(budget)


def _fits(p: Proposal, budget) -> bool:
    top = _ceiling(budget, p.before.look)
    return top is None or p.after.look is None or p.after.look <= top + 1e-9


def verdict(trial, proposal: Proposal, budget=None) -> str:
    """"" when the proposal is kept, else the reason (a key of WHY). The same rule for a manual Apply's warning, the
    per-tool button and Auto Fix (AUTO_FIX.md 2.3)."""
    b, a = proposal.before, proposal.after
    if proposal.cand is None or proposal.cand.refusal or proposal.cand.co is None:
        return "held"
    if proposal.judged and not proposal.fixed:
        return "no better"
    if a.bad is not None and b.bad is not None and a.bad > b.bad:
        return "bad faces"
    if a.tidy is not None and b.tidy is not None and a.tidy < b.tidy - MIN_GAIN:
        return "worse"
    if a.loops is not None and b.loops is not None and a.loops < b.loops - MIN_GAIN:
        return "worse"
    if a.poles is not None and b.poles is not None and a.poles > b.poles:
        return "worse"
    return "" if _fits(proposal, budget) else "look"


def _better(before: float, after: float, least: float) -> bool:
    gain = before - after
    return gain > 0.0 and gain >= least


def _corners(faces, ids) -> np.ndarray:
    return np.unique(np.fromiter((v for f in np.asarray(ids).tolist() for v in faces[f]), dtype=np.int64))


def _mask(n: int, ids) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[np.asarray(ids, dtype=np.int64)] = True
    return out


def _run(trial, base: T.Base, settings: dict, rough: bool):
    work = trial.run(base, settings, rough)
    while True:
        try:
            progress = next(work)
        except StopIteration as done:
            return done.value
        yield progress


def _sub(base: T.Base, scope: np.ndarray, co=None, pick=None) -> T.Base:
    """`base` with another scope inside the same holds (the union's edge is held, and spots never share a corner)."""
    return replace(base, co=base.co if co is None else co, scope=scope, cache={}, pick=dict(pick or {}))


def _positions(trial, base: T.Base, settings: dict, spots: list, rough: bool):
    """Positions-only tools: one run on the union of the spots (or one per spot for spots with picks), then each spot
    judged on its own faces; a spot not better goes back to where it was (safe: its edge was held)."""
    union = np.zeros(len(base.faces), dtype=bool)
    for sp in spots:
        union[np.asarray(sp.faces, dtype=np.int64)] = True
    union &= base.scope
    work = T.make_base(base.co, base.faces, scope=union, stay=base.stay, project=base.project, mirror=base.mirror, facing=base.facing, labels=base.labels, kinds=base.kinds)
    co, last, left = base.co.copy(), None, []
    picked = any(sp.pick for sp in spots)
    groups = [[sp] for sp in spots] if picked else [spots]
    for k, group in enumerate(groups):
        scope = (_mask(len(base.faces), group[0].faces) & union) if picked else union
        here = _sub(work, scope, co, group[0].pick if picked else base.pick)
        yield from trial.prepare(here)
        refusal = trial.check(here) if scope.any() else "nothing free"
        if refusal:
            left += [(sp, "held") for sp in group]
            continue
        cand = yield from _run(trial, here, dict(settings, **(group[0].pick if picked else {})), rough)
        if cand is None or cand.refusal or cand.co is None:
            left += [(sp, "held") for sp in group]
            continue
        co, last = np.asarray(cand.co, dtype=np.float64).copy(), cand
        yield (k + 1) / len(groups)
    out = T.Candidate(co, strength=float(settings.get(trial.handles[0], 1.0)) if trial.handles else 1.0, exact=not rough)
    if last is not None:
        out.lines, out.note, out.held = last.lines, last.note, list(last.held)
    gone = {id(sp) for sp, _ in left}
    bad_after = yield from T.bad_faces_steps(base, co)
    bad_before = yield from T.bad_faces_steps(base, base.co)
    fixed = []
    for sp in spots:
        if id(sp) in gone:
            continue
        ids = np.asarray(sp.faces, dtype=np.int64)
        a, b = trial.own_number(base, base.co, base.faces, sp), trial.own_number(base, co, base.faces, sp)
        why = "" if _better(a, b, trial.least) else "no better"
        if not why and bad_after[ids].sum() > bad_before[ids].sum():
            why = "bad faces"
        if why:
            back = _corners(base.faces, ids)
            co[back] = base.co[back]
            left.append((sp, why))
        else:
            fixed.append(sp)
    return out, fixed, left


def _faces(trial, base: T.Base, settings: dict, spots: list, rough: bool):
    """Tools that change faces: spots worst first, each on the mesh as the last one left it (`rebase`); a spot that is
    not better is skipped and the mesh stays as it was before it."""
    cur, origin, changed = base, np.arange(len(base.faces)), False
    fixed, left, last = [], [], None
    for k, sp in enumerate(spots):
        here = np.isin(origin, np.asarray(sp.faces, dtype=np.int64))
        scope = here & cur.scope
        if not scope.any():
            left.append((sp, "held"))
            continue
        work = T.make_base(cur.co, cur.faces, scope=scope, stay=cur.stay, project=cur.project, mirror=cur.mirror, pick=sp.pick, facing=cur.facing, labels=cur.labels, kinds=cur.kinds)
        yield from trial.prepare(work)
        refusal = trial.check(work)
        cand = None if refusal else (yield from _run(trial, work, dict(settings, **sp.pick), rough))
        if cand is None or cand.refusal or cand.co is None:
            left.append((sp, "held"))
            continue
        faces = cur.faces if cand.faces is None else cand.faces
        parent = np.arange(len(faces)) if cand.faces is None else (np.asarray(cand.parent, dtype=np.int64) if cand.parent is not None else None)
        if parent is None or len(parent) != len(faces):
            left.append((sp, "held"))  # without `parent` the spot cannot be followed onto the new faces
            continue
        new_origin = np.where(parent >= 0, origin[np.clip(parent, 0, max(len(origin) - 1, 0))], -1)
        now = replace(sp, faces=np.flatnonzero(here))
        then = replace(sp, faces=np.flatnonzero(np.isin(new_origin, np.asarray(sp.faces, dtype=np.int64))))
        co = np.asarray(cand.co, dtype=np.float64)
        a, b = trial.own_number(base, cur.co, cur.faces, now), trial.own_number(base, co, faces, then)
        why = "" if _better(a, b, trial.least) else "no better"
        if not why:
            before = yield from T.bad_faces_steps(cur, cur.co)
            after = yield from T.bad_faces_steps(cur, co, faces if cand.faces is not None else None, parent, getattr(trial, "splits_faces", False))
            bad_before = int(before[now.faces].sum())
            if getattr(trial, "splits_faces", False):
                bad_after = T.bad_by_parent(after[then.faces], parent[then.faces])
            else:
                bad_after = int(after[then.faces].sum())
            why = "bad faces" if bad_after > bad_before else ""
        if why:
            left.append((sp, why))
            continue
        changed |= cand.faces is not None
        cur, origin, last = T.rebase(cur, T.Candidate(co, faces=None if cand.faces is None else list(faces), parent=parent)), new_origin, cand
        fixed.append(sp)
        yield (k + 1) / len(spots)
    out = T.Candidate(cur.co.copy(), faces=list(cur.faces) if changed else None, parent=origin.copy() if changed else None, exact=not rough)
    if last is not None:
        out.note, out.lines = last.note, last.lines if not changed else None
    return out, fixed, left


def _build(trial, base, settings, spots, rough):
    if spots is None:
        cand = yield from _run(trial, base, settings, rough)
        return cand, [], []
    if not spots:
        return T.Candidate(base.co.copy(), exact=not rough), [], []
    if trial.changes_faces:
        return (yield from _faces(trial, base, settings, spots, rough))
    return (yield from _positions(trial, base, settings, spots, rough))


def _words(trial, p: Proposal, cap=None) -> str:
    parts = []
    if p.judged:
        n = len(p.fixed)
        parts.append(f"{n} spot{'s' if n != 1 else ''}")
    for key, a, b in (("tidy", p.before.tidy, p.after.tidy), ("look", p.before.look, p.after.look)):
        if a is not None and b is not None:
            parts.append(T.number_text(key, (a, b), cap)[0])
    return f"{trial.name}: " + ", ".join(parts) if parts else trial.name


def _pieces(base: T.Base, cand: T.Candidate):
    piece = base.labels.get("piece")
    if piece is None or len(piece) != len(base.faces):
        return None
    piece = np.asarray(piece, dtype=np.int64)
    if cand.faces is None:
        return piece
    return None if cand.parent is None else piece[np.clip(np.asarray(cand.parent, dtype=np.int64), 0, len(piece) - 1)]


def propose(trial, base: T.Base, settings: dict, spots=None, look=None, budget=None, rough=False, shown=None):
    """Generator -> Proposal. spots=None: the manual path, the tool on base.scope with base.pick (Look First).
    spots=[...]: the tool on each spot's region. Same maths, guards and numbers either way.

    `look(co, faces)`: generator returning the look change in percent (the camera check), or None. `look.parts(co,
    faces, labels, limit)`, when present: generator returning the labels of parts that changed more than `limit`.
    `budget`: the most look change allowed, a number, a function of the look before, or None. `shown(cand)`: called
    with each candidate as soon as it exists, so a preview can draw it while it is measured.
    Yields a progress 0..1 while the tool runs, the word "measuring" before the numbers."""
    if isinstance(trial, EverySpot):
        trial = trial.trial
        if spots is None:
            spots = yield from trial.spots(base)
            if not spots:
                return Proposal(T.Candidate(None, refusal=f"No {_many(trial)} in Work On."), judged=True, reason="no better")
    settings = dict(trial.defaults, **settings)
    judged = spots is not None
    spots = None if spots is None else sorted(spots, key=lambda sp: -float(sp.bad))
    stages = tuple(trial.stages) + (() if rough else tuple(k for k in EXTRA if k not in trial.stages))
    handle = trial.handles[0] if trial.handles else ""
    halve = bool(handle) and (trial.blendable or not judged)
    dropped: set = set()  # spots put back because their part changed too much
    live, settled, last = spots, [], None  # a rebuild runs only the spots still kept; `settled` keeps the others' reasons
    held = None
    while True:
        cand, fixed, left = yield from _build(trial, base, settings, live, rough)
        if spots is not None:
            settled, left = settled + left, settled + left + [(sp, "look") for sp in spots if id(sp) in dropped]
        if cand is None or cand.refusal or cand.co is None:
            p = Proposal(cand, fixed, left, judged=judged, held=held)
            p.reason = verdict(trial, p, budget) if judged else ""
            return p
        cand = yield from T.diff_steps(base, cand)
        if shown is not None:
            shown(cand)
        yield "measuring"
        yield from T.measure(base, cand, stages, look, splits=getattr(trial, "splits_faces", False))
        p = Proposal(cand, fixed, left, _scores(cand.numbers, 0), _scores(cand.numbers, 1), judged=judged, held=held)
        if _fits(p, budget) or (judged and not fixed):
            break
        top = _ceiling(budget, p.before.look)
        parts = getattr(look, "parts", None)
        pieces = _pieces(base, cand) if judged and parts is not None else None
        # each drop step must lower the look change: one that does not gains nothing, and dropping ends there
        if pieces is not None and (last is None or p.after.look < last - 1e-9):
            changed = yield from parts(np.asarray(cand.co, dtype=np.float64), base.faces if cand.faces is None else cand.faces, pieces, max(top, 0.0) / 100.0)
            changed = set(np.asarray(changed, dtype=np.int64).tolist())
            base_piece = np.asarray(base.labels["piece"], dtype=np.int64)
            drop = {id(sp) for sp in fixed if changed & set(base_piece[np.asarray(sp.faces, dtype=np.int64)].tolist())}
            if drop and len(drop) < len(fixed):
                dropped |= drop
                live, last = [sp for sp in fixed if id(sp) not in drop], p.after.look
                continue
        if halve and float(settings[handle]) >= STRENGTH_LEAST:
            settings[handle] = held = round(float(settings[handle]) * 0.5, 2)  # halving; a search between would land closer
            dropped, live, settled, last = set(), spots, [], None
            continue
        break
    p.held = held
    if judged:
        found = [sp.at for sp in spots]
        cand.found = np.asarray(found, dtype=np.float64).reshape(-1, 3)
        cand.held = list(cand.held) + [(np.asarray(sp.at, dtype=np.float64), WHY.get(why, why)) for sp, why in p.left]
    p.reason = verdict(trial, p, budget)
    p.words = _words(trial, p)
    return p


def _many(trial) -> str:
    return trial.kinds[0][2] if trial.kinds else "spots this tool fixes"


def kind_words(trial, kind: str, count: int) -> str:
    """"3 patches of slanted quads", from the tool's `kinds`."""
    for key, one, many in trial.kinds:
        if key == kind:
            return f"{count} {one if count == 1 else many}"
    return f"{count} spot{'s' if count != 1 else ''}"


# --------------------------------------------------------------------------- the per-tool button


class EverySpot(T.Trial):
    """"Fix every spot like this": the tool on every spot it finds in Work On, through `propose`. Look First opens on
    it like on any tool. Keep going re-runs it on the kept result while it still finds something to make better."""

    inner = None  # the wrapped Trial class, for `every(cls)`
    blendable = False  # spots put back differ per strength: no instant blend
    picks: tuple = ()
    asks: dict = {}
    auto = False  # never a pass of its own in Auto Fix

    def __init__(self, trial=None):
        t = trial if trial is not None else self.inner
        self.trial = t() if isinstance(t, type) else t
        t = self.trial
        self.name = f"{t.name}, every spot"
        self.group, self.handles, self.words, self.legend = t.group, t.handles, t.words, t.legend
        self.defaults = dict(t.defaults, keep_going=False)
        self.layers = tuple(dict.fromkeys(tuple(t.layers) + ("found", "held")))
        self.stages = t.stages
        self.toggles = tuple(t.toggles) + (("keep_going", "Keep going"),)
        self.kinds, self.own, self.least, self.changes_faces, self.splits_faces = t.kinds, t.own, t.least, t.changes_faces, t.splits_faces

    def value_text(self, value: float) -> str:
        return self.trial.value_text(value)

    def prepare(self, base):
        yield from self.trial.prepare(base)

    def check(self, base) -> str:
        return self.trial.check(base)

    def spots(self, base):
        return (yield from self.trial.spots(base))

    def own_number(self, base, co, faces, spot) -> float:
        return self.trial.own_number(base, co, faces, spot)

    def run(self, base, settings, rough=False):
        p = yield from propose(self, base, settings, rough=rough)
        if p.reason and p.cand is not None and not p.cand.refusal:
            return T.Candidate(None, refusal=refusal_words(self.trial, p))
        return p.cand


def every(cls) -> type:
    """An EverySpot class for an operator's `trial` field: `trial = autofix.every(SquareUp)`."""
    return type(f"Every{cls.__name__}", (EverySpot,), {"inner": cls})


def refusal_words(trial, p: Proposal) -> str:
    """What the card says when nothing of a per-tool run is kept."""
    n = len(p.left) or len(p.fixed)
    why = WHY.get(p.reason, p.reason)
    return f"Nothing kept: {n} spot{'s' if n != 1 else ''} left as they were ({why})."


# --------------------------------------------------------------------------- Auto Fix after Start

PASSES = (  # AUTO_FIX.md 3.2: (group switch, module, class, kinds or None for all)
    ("leftovers", "tidytools", "PreviewTidy", ("inward", "corners", "doublets", "pairs")),
    ("seam", "tidytools", "SealMiddle", None),
    ("poles", "loopspoles", "CancelPolePair", None),
    ("poles", "stripfix", "RemoveDeadEnd", None),
    ("patches", "stripfix", "RedrawInRows", None),
    ("hard", "hardtidy", "TrueUpEdges", None),
    ("hard", "hardtidy", "PressPanelsFlat", None),
    ("hard", "tidytools", "HardEdges", None),
    ("rows", "stripfix", "Straighten", None),
    ("rows", "stripfix", "EvenWidth", None),
    ("rows", "loopspoles", "EvenTheBand", None),
    ("shape", "loopspoles", "OpenUpPoles", None),
    ("shape", "relax", "SquareUp", None),
    ("shape", "relax", "RelaxTowardDetail", None),
    ("shape", "relax", "SourceRelax", None),
    ("shape", "relax", "KeepLinesRelax", None),
    ("leftovers", "tidytools", "PreviewTidy", ("slivers",)),
)
GROUPS = ("leftovers", "seam", "poles", "patches", "hard", "rows", "shape")


def plan(on) -> list:
    """The pass order of AUTO_FIX.md 3.2 for the groups switched on: [(Trial class, kinds or None)]. A tool that is
    manual only or cannot find spots yet is left out; it joins the day it can."""
    import importlib

    out = []
    for group, module, name, kinds in PASSES:
        if group not in on:
            continue
        cls = getattr(importlib.import_module(f"{__package__}.{module}"), name)
        if cls.auto and cls.spots is not T.Trial.spots:
            out.append((cls, kinds))
    return out


@dataclass
class Event:
    """One thing Auto Fix tells its driver. kind: "working" (progress), "kept" (commit `proposal.cand` now, one History
    entry), "left" (nothing changed, `proposal.reason` says why), "stopped" (`stopped()` said so; `step` is the pass to
    continue from), "done" (no round left that helps)."""

    kind: str
    round: int = 1
    step: int = 0  # index into the plan
    trial: object = None
    proposal: Proposal | None = None
    spots: int = 0
    progress: float = 0.0
    words: str = ""


def _key(base: T.Base) -> str:
    co = np.round(np.asarray(base.co, dtype=np.float64) / (base.edge or 1.0), 5)
    h = hashlib.blake2b(co.tobytes(), digest_size=16)
    h.update(repr(base.faces).encode())
    return h.hexdigest()


def run(make_base, plan, settings=None, look=None, budget=None, stopped=None, start: int = 0, rnd: int = 1, carried: bool = False):
    """Generator of `Event`s: Auto Fix's rounds (AUTO_FIX.md 3). `make_base()` gives a Base of the result as it is now;
    it is called before each pass, so a driver that commits on "kept" before pumping on is what the next pass sees.
    `settings`: {Trial class name: settings}. `stopped()`: "" to go on, or words for why it stops between passes (the
    time limit). `start`: the pass to begin the first round at (Keep going after a stop). Rounds repeat while one keeps
    a pass and the round made the result better as a whole (`_round_gained`); no round count. A round that ends on a
    mesh an earlier round ended on also ends it (the tools went round in a circle). `rnd` and `carried` resume a round a stop cut short: its number, and whether its passes before the
    stop kept something (then another round follows, as it would have)."""
    settings = settings or {}
    seen, found, first = set(), carried, True
    while True:
        kept, gain, carried = carried, carried, False
        first_before = None
        for i, (cls, kinds) in enumerate(plan):
            if first and i < start:
                continue
            why = stopped() if stopped is not None else ""
            if why:
                yield Event("stopped", rnd, i, words=why)
                return
            trial = cls()
            base = make_base()
            spots = yield from _relay(trial.spots(base), rnd, i, trial, 0)
            spots = [sp for sp in spots if kinds is None or sp.kind in kinds]
            if not spots:
                continue
            found = True
            s = dict(settings.get(cls.__name__, {}))
            p = yield from _relay(propose(trial, base, s, spots, look, budget), rnd, i, trial, len(spots))
            yield Event("left" if p.reason else "kept", rnd, i, trial, p, len(spots), 1.0, p.words)
            first_before = first_before or p.before
            if not p.reason:
                kept = True
                gain = gain or _round_gained(first_before, p.after)
        first = False
        if kept and gain:  # passes that only trade their own numbers back and forth do not start another round
            key = _key(make_base())
            if key not in seen:
                seen.add(key)
                rnd += 1
                continue
        yield Event("done", rnd, len(plan), words=FINISHED if found else NOTHING)
        return


def _round_gained(b: Scores, a: Scores) -> bool:
    """Whether the round so far made the result better as a whole: a pole or a bad face fewer, or Tidiness or Loop
    Quality up by `MIN_GAIN` (`organize.gained`'s measure). Rounds go on while this holds. Nothing measured: True."""

    def up(x, y, m):
        return x is not None and y is not None and y - x >= m

    pairs = ((b.poles, a.poles), (b.bad, a.bad), (b.tidy, a.tidy), (b.loops, a.loops))
    if all(x is None or y is None for x, y in pairs):
        return True
    return up(a.poles, b.poles, 1) or up(a.bad, b.bad, 1) or up(b.tidy, a.tidy, MIN_GAIN) or up(b.loops, a.loops, MIN_GAIN)


def _relay(work, rnd, i, trial, count):
    """Pump a generator, turning its progress into "working" events. A step with no number yet (camera pictures, the
    word "measuring") is an event too (progress -1), so a timer slice can end there."""
    while True:
        try:
            progress = next(work)
        except StopIteration as done:
            return done.value
        yield Event("working", rnd, i, trial, spots=count, progress=float(progress) if isinstance(progress, (int, float)) else -1.0)
