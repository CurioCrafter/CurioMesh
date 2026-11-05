# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The size ladder behind the Quad Size and Organization sliders: plain arithmetic, no Blender.

A rung is one whole result of the search, made at another Detail (how much of the view may change) and Style. The
search lands each one on its own Detail and spends the faces where the camera sees they are missing, so every rung
already gives each area the detail it needs. The sliders only pick rungs; in between two Styles, loose pieces change
over one at a time, smallest first.
"""

from __future__ import annotations

import math

FACTORS = (0.25, 0.4, 0.63, 1.0, 1.59, 2.52, 4.0)  # of the Detail you searched with: smallest quads first
ROWS_FACTORS = (1.0, 2.52)  # Clean Rows takes minutes a rung, so only these
STYLES = ("DETAIL", "BALANCED", "ROWS")  # left to right on the Organization bar
STOPS = (0.0, 0.5, 1.0)
LOWEST, HIGHEST = 0.1, 80.0  # what the Detail setting allows


def details(base: float, factors=FACTORS) -> list[float]:
    """The Detail of every rung, rounded so the same rung is always the same key."""
    return sorted({round(min(max(base * f, LOWEST), HIGHEST), 3) for f in factors})


def plan(base: float, style: str, rows: bool = True) -> list[tuple[str, float]]:
    """(style, detail) to work out, in order. Coarser sizes of your Style first (they reuse what was measured, so they are
    quick), then the finest one (it measures again, and the ones between then reuse that), then the other Styles."""
    here = round(min(max(base, LOWEST), HIGHEST), 3)
    own = [d for d in details(base) if d != here]
    coarser, finer = [d for d in own if d > here], [d for d in own if d < here]
    order = [(style, d) for d in coarser + finer]
    for other in STYLES:
        if other == style or (other == "ROWS" and not rows):
            continue
        sizes = details(base, ROWS_FACTORS if other == "ROWS" else FACTORS)
        order += [(other, d) for d in sorted(sizes, key=lambda d: (abs(math.log(d / here)), -d))]
    return order


def useful(rungs) -> list:
    """Keys of the rungs worth showing, from (key, faces, change) each: none that another rung beats on both faces and
    change. A higher Detail can land on more faces (seen on a many-part machine); skipping it keeps the slider's face
    count going one way."""
    rungs = list(rungs)
    return [k for k, f, c in rungs if not any((f2 <= f and c2 <= c) and (f2 < f or c2 < c) for _k2, f2, c2 in rungs)]


def size_detail(base: float, size: float) -> float:
    """Quad Size 0..1 as a Detail: 0 the smallest quads, 0.5 the size you searched with, 1 the biggest."""
    size = min(max(size, 0.0), 1.0)
    return base * FACTORS[-1] ** (2.0 * size - 1.0)


def size_of(base: float, detail: float) -> float:
    """The Quad Size slider position of a Detail (the inverse of `size_detail`)."""
    return min(max(0.5 + 0.5 * math.log(detail / base) / math.log(FACTORS[-1]), 0.0), 1.0)


def nearest(ready, wanted: float):
    """The ready Detail closest to `wanted`, by ratio (2 and 8 are as far from 4). None when nothing is ready."""
    ready = list(ready)
    if not ready:
        return None
    return min(ready, key=lambda d: (abs(math.log(d / wanted)), d))


def step(ready, current: float, bigger: bool):
    """The next ready Detail up (bigger quads) or down (smaller quads) from `current`, or None at the end."""
    side = [d for d in ready if (d > current * 1.001 if bigger else d < current / 1.001)]
    if not side:
        return None
    return min(side) if bigger else max(side)


def between(order: float) -> tuple[str, str, float]:
    """The Organization bar as (style on the left, style on the right, 0..1 of the way between them)."""
    order = min(max(order, 0.0), 1.0)
    for k in range(len(STOPS) - 1):
        if order <= STOPS[k + 1] or k == len(STOPS) - 2:
            return STYLES[k], STYLES[k + 1], (order - STOPS[k]) / (STOPS[k + 1] - STOPS[k])
    raise AssertionError("unreachable")


def order_of(style: str) -> float:
    return STOPS[STYLES.index(style)] if style in STYLES else 0.0


def switched(sizes, t: float) -> list[int]:
    """Pieces that take the right-hand Style when the bar is `t` of the way across. Smallest first, each when the bar
    passes the middle of its share of the faces: moving the bar on only ever adds pieces, and one piece flips at 0.5."""
    sizes = [max(float(s), 0.0) for s in sizes]
    total = sum(sizes)
    if total <= 0.0 or t <= 0.0:
        return []
    out, before = [], 0.0
    for k in sorted(range(len(sizes)), key=lambda k: (sizes[k], k)):
        if t >= (before + 0.5 * sizes[k]) / total:
            out.append(k)
        before += sizes[k]
    return sorted(out)
