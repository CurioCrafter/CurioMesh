# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Face Loops' fixes as Tidy Tools (`trial.Trial`, plain NumPy, no bpy): AUTO_FIX.md 2.5, the faceloops2 row.

- Straighten and Even Width move one strip along its rung lines (`strips.targets` / `strips.move`), as Face Loops'
  buttons always did. Their spots are the wiggly strips (turn over WIGGLY degrees a quad) and the strips of uneven
  width (spread over UNEVEN); Fix Strips is both at once, the maths of Fix Every Strip.
- Remove Dead End takes away a short open strip that runs between two poles (`strips.collapse`, the whole strip and
  its mirror image), so the two poles at its ends go.
- Redraw in Rows lays the faces of an area out again as rows (`flowstroke.rows`, with `rowfill`); with no view it
  follows the bends and the long side of the area (`rowfill.direction` with no `across`). Its spots are the messy
  patches (`messy_patches`): parts with more than twice the result's pole density and lower Tidiness than the result.

`faceloops.py` keeps the operators and calls these; Auto Fix and "Fix every spot like this" call them through
`autofix.propose`. Same code everywhere.
"""

from __future__ import annotations

import numpy as np

from . import strips as S
from .loopspoles import CHUNK, EvenTheBand, _onto, face_poles, join, lead_side, region
from .tidytools import NOTHING_FREE
from .trial import Base, Candidate, Spot, Trial, _tidiness

WIGGLY = 4.0  # degrees per quad: a strip that turns more is wiggly
UNEVEN = 0.15  # a strip whose width varies more than this share of its mean is uneven
SHORT = 3  # quads: an open strip this short between two poles is a dead end
MESSY = 2.0  # a part with more than this many times the result's pole density (and lower Tidiness) is messy
SMALLEST = 8  # faces: rows need at least a few quads each way
KIND = {"straighten": "wiggly", "even": "uneven"}
PICK_FIRST = "Pick a strip first."


def _flowstroke():
    """The add-on's `flowstroke` (Redraw in Rows' maths lives beside the add-on, not in the triadq package), or None."""
    try:
        from .. import flowstroke  # inside the add-on
    except ImportError:
        try:
            import flowstroke  # the tests, run from the repo
        except ImportError:
            return None
    return flowstroke


def _mirror(base: Base):
    """`strips.finish`'s mirror argument, the partners worked out once per Base."""
    if base.mirror is None:
        return None
    if "partners" not in base.cache:
        base.cache["partners"] = S.partners(base.co, *base.mirror)
    return (base.mirror[0], base.mirror[1], base.cache["partners"])


def _side(base: Base, co, strip) -> float:
    if base.mirror is None:
        return 1.0
    return 1.0 if float(np.mean(co[strip["a"] + strip["b"], base.mirror[0]]) - base.mirror[1]) >= 0 else -1.0


class _StripTools(Trial):
    """Shared by the strip tools: the strips and loop continuations of the mesh, worked out once per mesh."""

    group = "LOOPS"

    def _strips(self, base: Base) -> S.Strips:
        # every Base of one propose shares the faces list: one Strips for them all (topology only)
        held = getattr(self, "_held", None)
        if held is None or held[0] is not base.faces:
            from .loopscore import _continuations
            from .patterns import Topology

            st = S.Strips(base.co, base.faces)
            self._held = held = (base.faces, st, _continuations(Topology(base.co, base.faces)))
        return held[1]

    def _go(self, base: Base) -> dict:
        self._strips(base)
        return self._held[2]

    def _first(self, base: Base) -> dict:
        """{strip label: its first element}."""
        st = self._strips(base)
        return {int(lab): int(e) for e, lab in reversed(list(enumerate(st.labels.tolist())))}


# --------------------------------------------------------------------------- Straighten, Even Width


class _StripMove(_StripTools):
    handles = ("amount",)
    defaults = {"amount": 1.0, "strip": -1}
    blendable = True
    stages = ("strips", "loops", "look")
    picks = ("strip",)
    asks = {"strip": "click a strip"}
    least = 0.01  # of the limit (WIGGLY degrees, UNEVEN width spread)
    fixes: tuple = ()  # "straighten", "even": what this tool does to a strip that needs it, first that fits

    pick_at = EvenTheBand.pick_at

    def value_text(self, value: float) -> str:
        return f"Amount {value:.2f}"

    def check(self, base: Base) -> str:
        return "" if base.free.any() else NOTHING_FREE

    def _need(self, co, strip):
        """(fix, how bad as a share of its limit) for the first of `fixes` the strip needs, or None."""
        st = S.stats(co, strip)
        for fix in self.fixes:
            if fix == "straighten" and st["turn"] > WIGGLY:
                return fix, st["turn"] / WIGGLY
            if fix == "even" and st["width"] > UNEVEN:
                return fix, st["width"] / UNEVEN
        return None

    def _lines(self, base: Base, strip):
        return S.rung_lines(strip, self._go(base), len(base.co), base.stay)

    def spots(self, base: Base):
        """One spot per strip that needs a fix (touching spots joined): the faces round the corners it would move."""
        st = self._strips(base)
        first = self._first(base)
        out = []
        for i, (label, element) in enumerate(first.items()):
            if i % CHUNK == CHUNK - 1:
                yield i / max(len(first), 1)
            if st.sizes[label] < 2:
                continue
            strip = st.walk(element)
            if not all(base.scope[f] for f in strip["faces"]) or not lead_side(base, base.co[strip["a"] + strip["b"]]):
                continue
            need = self._need(base.co, strip)
            if need is None:
                continue
            lines = self._lines(base, strip)
            moved = np.unique(np.concatenate([ids[1:-1] for ids, _ia in lines] or [np.zeros(0, np.int64)]))
            moved = moved[base.free[moved]]
            faces = region(base, moved) if len(moved) else moved
            if len(faces):
                out.append(Spot(KIND[need[0]], faces, base.co[strip["a"] + strip["b"]].mean(axis=0), need[1], {"strips": [element], "fix": [need[0]]}))
        return join(out, base.faces)

    def _score(self, co, strip, fix) -> float:
        st = S.stats(co, strip)
        return st["turn"] / WIGGLY if fix == "straighten" else st["width"] / UNEVEN

    def own_number(self, base: Base, co, faces, spot) -> float:
        """Wiggle (or width spread) of the spot's strips as a share of its limit, mean: lower is better."""
        st = self._strips(base)
        pairs = list(zip(spot.pick.get("strips", []), spot.pick.get("fix", [])))
        return float(np.mean([self._score(np.asarray(co), st.walk(int(e)), fix) for e, fix in pairs])) if pairs else 0.0

    def run(self, base: Base, settings: dict, rough: bool = False):
        elements = [int(e) for e in settings.get("strips") or []] or [int(settings.get("strip", -1))]
        fixes = list(settings.get("fix") or []) or [self.fixes[0]] * len(elements)
        amount = float(settings.get("amount", 1.0))
        st = self._strips(base)
        if any(e < 0 or e >= len(st.labels) for e in elements):
            return Candidate(None, refusal=PICK_FIRST)
        co = base.co.copy()
        walks = [st.walk(e) for e in elements]
        before = [self._score(base.co, w, fix) for w, fix in zip(walks, fixes)]
        order = np.argsort(-np.asarray(before)) if len(walks) > 1 else [0]
        for k, i in enumerate(order):  # several strips (a joined spot): worst first, each from where the last left it
            strip, fix = walks[i], fixes[i]
            if len(walks) > 1 and self._score(co, strip, fix) <= 1.0:
                continue  # a neighbour's fix already took this one's wiggle out
            lines = self._lines(base, strip)
            if lines:
                goals = S.targets(co, lines, fix, amount, strip["closed"])
                new = S.move(co, lines, goals, base.stay)
                co = S.finish(co, new, base.faces, lambda x: _onto(base, x), _mirror(base), _side(base, co, strip))
                co[base.stay] = base.co[base.stay]
            yield (k + 1) / len(walks)
        after = [self._score(co, w, fix) for w, fix in zip(walks, fixes)]
        a, b = float(np.mean(before)), float(np.mean(after))
        cand = Candidate(co, strength=amount, exact=True, note=f"{len(elements)} strip{'s' if len(elements) != 1 else ''}")
        sign = 1 if b < a - 0.01 else -1 if b > a + 0.01 else 0
        if set(fixes) == {"straighten"}:
            words = f"wiggle {a * WIGGLY:.1f} -> {b * WIGGLY:.1f} degrees a quad"
        elif set(fixes) == {"even"}:
            words = f"width varies {100 * a * UNEVEN:.0f}% -> {100 * b * UNEVEN:.0f}%"
        else:
            words = f"strips off {100 * a:.0f}% -> {100 * b:.0f}% of clean"
        cand.numbers["strips"] = (words, sign)
        return cand


class Straighten(_StripMove):
    name = "Straighten"
    fixes = ("straighten",)
    kinds = (("wiggly", "wiggly strip", "wiggly strips"),)
    own = "turn"


class EvenWidth(_StripMove):
    name = "Even Width"
    fixes = ("even",)
    kinds = (("uneven", "strip of uneven width", "strips of uneven width"),)
    own = "width"


class FixStrips(_StripMove):
    """Straighten where a strip is wiggly, else Even Width where it is uneven: Fix Every Strip's maths (as an EverySpot
    run). Auto Fix runs Straighten and Even Width as passes of their own, so this one is not a pass."""

    name = "Fix Strips"
    fixes = ("straighten", "even")
    kinds = Straighten.kinds + EvenWidth.kinds
    own = "strips"
    auto = False


# --------------------------------------------------------------------------- Remove Dead End


class RemoveDeadEnd(_StripTools):
    name = "Remove Dead End"
    handles = ()
    defaults = {"strip": -1}
    blendable = False
    layers = ("scope", "ghost", "arrows", "poles", "held")
    stages = ("poles", "faces", "look")
    picks = ("strip",)
    asks = {"strip": "click a strip"}
    kinds = (("dead end", "short dead-end strip", "short dead-end strips"),)
    own = "poles"
    least = 1.0  # one pole fewer
    changes_faces = True

    pick_at = EvenTheBand.pick_at

    def check(self, base: Base) -> str:
        return "" if base.free.any() else NOTHING_FREE

    def _group(self, base: Base, element: int) -> list:
        """The whole strip through `element` (on past poles), and its mirror image on a mirrored result."""
        st = self._strips(base)
        strip = st.walk(element, past_poles=True)
        group = [strip]
        mirror = _mirror(base)
        if mirror is not None:
            other = S.mirror_element(st, strip, mirror[2])
            if other >= 0:
                group.append(st.walk(other, past_poles=True))
        return group

    def _dead_ends(self, base: Base):
        """Generator -> [(element, middle point, quads)]: open strips of SHORT quads or fewer with a pole at both ends,
        whose whole strip is open and inside the scope. One per whole strip."""
        st = self._strips(base)
        poles = base.topo.poles & ~base.topo.border
        # a dead end lies wholly inside the scope, so its first element does too: on a spot's small scope this skips
        # every other strip of the mesh (Auto Fix runs this once per spot)
        inside = base.scope[st.face_of[np.arange(2 * len(st.face_of)) // 2]] if len(st.face_of) else np.zeros(0, dtype=bool)
        first = {label: element for label, element in self._first(base).items() if inside[element]}
        seen, out = set(), []
        for i, (label, element) in enumerate(first.items()):
            if i % CHUNK == CHUNK - 1:
                yield i / max(len(first), 1)
            if st.sizes[label] > SHORT:
                continue
            short = st.walk(element)
            if short["closed"] or len(short["rows"]) > SHORT:
                continue
            if not (poles[[short["a"][0], short["b"][0]]].any() and poles[[short["a"][-1], short["b"][-1]]].any()):
                continue
            full = st.walk(element, past_poles=True)
            key = min(full["faces"])
            if full["closed"] or full["crossing"] or key in seen or not all(base.scope[f] for f in full["faces"]):
                continue
            seen.add(key)
            middle = base.co[short["a"] + short["b"]].mean(axis=0)
            out.append((element, middle, len(short["rows"])))
        return out

    def spots(self, base: Base):
        """One spot per dead end: the faces round the whole strip (and its mirror image)."""
        out = []
        for element, middle, quads in (yield from self._dead_ends(base)):
            if not lead_side(base, middle[None]):
                continue
            group = self._group(base, element)
            corners = np.unique([v for strip in group for v in strip["a"] + strip["b"]])
            faces = region(base, corners)
            if len(faces):
                out.append(Spot("dead end", faces, middle, 1.0 / quads, {"at": tuple(float(x) for x in middle)}))
        return out

    def own_number(self, base: Base, co, faces, spot) -> float:
        return float(face_poles(faces, co, spot.faces))

    def run(self, base: Base, settings: dict, rough: bool = False):
        element = int(settings.get("strip", -1))
        if element < 0 and settings.get("at") is not None:  # a spot: the dead end in the scope nearest where it was found
            found = yield from self._dead_ends(base)
            if found:
                at = np.asarray(settings["at"], dtype=np.float64)
                element = min(found, key=lambda x: float(np.linalg.norm(x[1] - at)))[0]
        if element < 0 or element >= len(self._strips(base).labels):
            return Candidate(None, refusal=PICK_FIRST)
        group = self._group(base, element)
        yield 0.5
        try:
            co, faces, parent = S.collapse(base.co, base.faces, group, base.stay)
        except ValueError as exc:
            return Candidate(None, refusal=f"Can't remove this strip: {exc}")
        n = sum(len(strip["faces"]) for strip in group)
        return Candidate(co, faces=faces, parent=parent, exact=True, note=f"{n} quad{'s' if n != 1 else ''} taken away")


# --------------------------------------------------------------------------- Redraw in Rows


def _sub(co, faces, ids):
    """(points, faces) of the faces `ids` alone, numbered from 0."""
    sub = [faces[f] for f in np.asarray(ids).tolist()]
    used = np.unique(np.fromiter((v for f in sub for v in f), dtype=np.int64))
    at = np.full(len(co), -1, dtype=np.int64)
    at[used] = np.arange(len(used))
    return np.asarray(co, dtype=np.float64)[used], [tuple(int(at[v]) for v in f) for f in sub]


def messy_patches(base: Base):
    """Generator -> [Spot]: the parts of the scope (the result's parts when it has them, else its loose pieces) with
    more than MESSY times the scope's pole density and a Tidiness no higher than the whole result's. Most poles first."""
    from .patch import mask_components

    scope = np.asarray(base.scope, dtype=bool)
    if not scope.any():
        return []
    piece = base.labels.get("piece")
    if piece is not None and len(piece) == len(base.faces):
        piece = np.asarray(piece)
        parts = [np.flatnonzero(scope & (piece == p)) for p in np.unique(piece[scope]).tolist()]
    else:
        parts = mask_components(base.faces, scope)
    topo = base.topo
    poles = topo.poles & ~topo.border
    every = np.unique(np.fromiter((v for f in np.flatnonzero(scope).tolist() for v in base.faces[f]), dtype=np.int64))
    density = float(poles[every].mean()) if len(every) else 0.0
    if "tidy" not in base.cache:
        base.cache["tidy"] = _tidiness(base.co, base.faces)
    yield 0.1
    out = []
    for i, ids in enumerate(parts):
        yield 0.1 + 0.9 * i / max(len(parts), 1)
        if len(ids) < SMALLEST:
            continue
        corners = np.unique(np.fromiter((v for f in ids.tolist() for v in base.faces[f]), dtype=np.int64))
        count = int(poles[corners].sum())
        if count < 2 or float(poles[corners].mean()) <= MESSY * density:
            continue
        if _tidiness(*_sub(base.co, base.faces, ids)) > base.cache["tidy"]:
            continue
        out.append(Spot("messy", ids, base.co[corners].mean(axis=0), float(count)))
    return sorted(out, key=lambda sp: -sp.bad)


class RedrawInRows(Trial):
    name = "Redraw in Rows"
    group = "LOOPS"
    handles = ()
    defaults: dict = {}
    blendable = False
    layers = ("scope", "ghost", "poles", "held")
    stages = ("poles", "loops", "look")
    kinds = (("messy", "messy patch", "messy patches"),)
    own = "poles"
    least = 1.0  # one pole fewer (Loop Quality only breaks a tie)
    changes_faces = True

    def check(self, base: Base) -> str:
        if not base.scope.any():
            return NOTHING_FREE
        return "" if _flowstroke() is not None else "Redraw in Rows needs the full add-on."

    def spots(self, base: Base):
        return (yield from messy_patches(base))

    def own_number(self, base: Base, co, faces, spot) -> float:
        """Poles in the patch, then Loop Quality there (100 - it, a hundredth of a pole per point)."""
        from .loopscore import loop_quality

        ids = np.asarray(spot.faces, dtype=np.int64)
        if not len(ids):
            return 0.0
        quality = float(loop_quality(*_sub(co, faces, ids))["quality"])
        return face_poles(faces, co, ids) + 0.01 * (100.0 - quality)

    def run(self, base: Base, settings: dict, rough: bool = False):
        """The faces `settings["faces"]` (default: the scope) as rows. `strokes`: Flow Editor strokes that steer them,
        `across`: the view's sideways direction (None: the bends and the long side of the area decide)."""
        from .segment import open_edges

        F = _flowstroke()
        ids = np.asarray(settings.get("faces") if settings.get("faces") is not None else np.flatnonzero(base.scope), dtype=np.int64)
        if len(ids) < SMALLEST:
            return Candidate(None, refusal="Mark a bigger area: rows need at least a few quads each way.")
        yield 0.1
        mat = np.zeros(len(base.faces), dtype=np.int32)
        try:
            co, faces, _mat, new, area, bad = F.rows(base.co, base.faces, mat, ids, settings.get("strokes") or (), base.project, settings.get("across"))
        except ValueError as exc:
            return Candidate(None, refusal=f"Can't redraw this area in rows: {exc}")
        yield 0.8
        if not base.scope[area].all():
            return Candidate(None, refusal="The rows would reach past the area you work on, so nothing changed.")
        if open_edges(faces) > open_edges(base.faces):
            return Candidate(None, refusal="The rows did not close on the faces round them, so the result stays as it was.")
        old = np.asarray([base.co[list(base.faces[i])].mean(axis=0) for i in area])
        kept = np.setdiff1d(np.arange(len(base.faces)), area)
        born = np.asarray([co[list(f)].mean(axis=0) for f, n in zip(faces, new) if n]).reshape(-1, 3)
        parent = np.empty(len(faces), dtype=np.int64)
        parent[~new] = kept
        parent[new] = area[np.argmin(np.linalg.norm(born[:, None] - old[None], axis=2), axis=1)] if len(born) else 0
        note = f"{len(area)} faces -> {int(new.sum())}" + (f", {bad} folded faces to check" if bad else "")
        return Candidate(np.asarray(co, dtype=np.float64), faces=[tuple(f) for f in faces], parent=parent, exact=True, note=note)


__all__ = ["Straighten", "EvenWidth", "FixStrips", "RemoveDeadEnd", "RedrawInRows", "messy_patches", "WIGGLY", "UNEVEN", "SHORT"]
