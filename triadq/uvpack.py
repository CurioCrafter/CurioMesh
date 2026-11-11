# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Island packer (package K7).

NumPy-free pure Python, no bpy. One algorithm for boxes and for charts packed into an existing layout:

- MaxRects best-short-side-fit (Jylanki 2010, "A thousand ways to pack the bin"), 90 degree steps only; a second placement
  rule, bottom-left (Baker, Coffman and Rivest 1980): the place whose box ends lowest, then leftmost. A policy is an
  orientation policy plus a rule (`policy % ORIENTS`, `policy // ORIENTS`); both rules are tried and the better kept.
- The gap rule: a mip-k texel is a block of G = 2^k texels. Two neighbours are mip-clean when the blocks they touch, each
  grown by one block, do not meet. A box *aligned* on a block edge needs G of gap on each side, any other (an outline, a
  chart of an existing layout) 2G - 1: aligned + aligned = 2G, aligned + loose = 3G - 1, loose + loose = 4G - 2. The box
  packer always aligns; the loose gap is for outlines and for charts already in a layout.
- Irregular charts: conservative row intervals per texel row, dilated by the gap in both axes (Minkowski sum with a square),
  bottom-left free-interval search (Chazelle 1983); score: least new page height, then least wasted interval length.
- Any angle (with `FREE_TURNS`): the first charts of the order also turn by even angle steps and by the angles that lay
  their longest convex hull edges (Andrew 1979) along an axis; the rotation is rational in t = tan(angle / 2), so no trig and
  the same bits on every machine. The first chart's every way is tried as the start (a beam on the first chart; the nofit view
  of Bennell and Oliveira 2008, "The geometry of nesting problems: a tutorial"), the first start that places all wins.
- Fill the page: the largest uniform scale at which everything fits, by a bracket that only tightens (growth steps, then
  bisection), so the best feasible layout is kept and nothing oscillates. Keep going: other orders and orientation
  policies, then adjacent swaps kept only when the page then fits a larger scale.

`FillPage` is sliceable: `step()` does about 3 ms of work and returns progress.
"""

from __future__ import annotations

from math import ceil, floor, inf, sqrt
import time

START_FILL = 0.5
GROW = 1.25
SCALE_TOL = 0.00001
MIN_GAIN = 0.0001
ORDERS = 5
ORIENTS = 4
POLICIES = 8
MAX_K = 20
EFFORT_STEP = 32  # `pack_into_steps` takes its effort in the high bits of k (k + EFFORT_STEP * effort, k from -1 to MAX_K)
ORDER_MODES = (0, 1, 2, 3)  # pack_into_steps orders: owned area, owned bounding box, longer owned side, long side of the chart
FREE_TURNS = 16777216  # `pack_into_steps` turns charts by any angle when k carries this flag (turn_ok then means: also by other angles)
TURN_STEPS = 65536  # an angle is j, t = j / TURN_STEPS = tan(angle / 2) in [0, 1); see `turn_of`
FREE_SPLIT = 6  # angles per turned chart at effort 0: this many even steps of t ...
HULL_EDGES = 4  # ... plus the angles laying this many longest hull edges along an axis; each effort doubles both
MAX_EFFORT = 13  # highest effort pack_into_steps reads
BEAM_SHARE = 8  # the first-chart beam runs when that chart owns at least 1 / (BEAM_SHARE << effort) of the page

SLICE_SECONDS = 0.003  # one step() of a stage works about this long
YIELD_EVERY = 1  # a pack gives control back after this many boxes

def snap(x: float) -> int:
    """Texel size of a box side: whole texels, at least one (the tiny tolerance keeps exact products exact)."""

    return max(int(ceil(x - 1e-9)), 1)


def block(k: int) -> int:
    """G = 2^k texels; 0 (no gap) for a negative level."""

    return 0 if k < 0 else 1 << min(k, MAX_K)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------- MaxRects

def _gap_axis(f0: int, flen: int, length: int, g: int):
    """One axis of a placement in the span [f0, f0 + flen): (footprint start, footprint end, content start) or None."""

    cs = f0 if g == 0 else _ceil_div(f0 + g, g) * g
    end = cs + length + g
    return (cs - g, end, cs) if end <= f0 + flen else None


class MaxRects:
    def __init__(self, w: int, h: int, k: int, bottom_left: bool = False):
        self.g = block(k)
        self.bottom_left = bottom_left
        self.free = [(0, 0, w, h)]

    def insert(self, w: int, h: int, can_turn: bool):
        """Place a box of w x h texels. Returns (content x, content y, turned) or None."""

        best = None
        for f in self.free:
            for turned in (False, True):
                if turned and (not can_turn or w == h):
                    continue
                bw, bh = (h, w) if turned else (w, h)
                ax = _gap_axis(f[0], f[2] - f[0], bw, self.g)
                ay = _gap_axis(f[1], f[3] - f[1], bh, self.g)
                if ax is None or ay is None:
                    continue
                rx, ry = f[2] - f[0] - (ax[1] - f[0]), f[3] - f[1] - (ay[1] - f[1])
                score = (ay[1], ax[1], 0, 0) if self.bottom_left else (min(rx, ry), max(rx, ry), ay[1], ax[1])
                if best is None or score < best[0]:
                    best = (score, (ax[0], ay[0], ax[1], ay[1]), (ax[2], ay[2]), turned)
        if best is None:
            return None
        _, p, (cx, cy), turned = best
        self._split(p)
        return cx, cy, turned

    def prune(self, side: int) -> None:
        """Forget free rectangles in which no box of `side` texels or more can sit (the layout stays the same, only faster)."""

        need = side + 2 * self.g
        self.free = [f for f in self.free if f[2] - f[0] >= need and f[3] - f[1] >= need]

    def _split(self, p):
        keep, news = [], []
        for f in self.free:
            if p[2] <= f[0] or p[0] >= f[2] or p[3] <= f[1] or p[1] >= f[3]:
                keep.append(f)
                continue
            if p[0] > f[0]:
                news.append((f[0], f[1], p[0], f[3]))
            if p[2] < f[2]:
                news.append((p[2], f[1], f[2], f[3]))
            if p[1] > f[1]:
                news.append((f[0], f[1], f[2], p[1]))
            if p[3] < f[3]:
                news.append((f[0], p[3], f[2], f[3]))

        def inside(a, b):
            return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]

        added = []
        for i, n in enumerate(news):
            covered = any(inside(n, r) for r in keep) or any(j != i and inside(n, m) and (m != n or j < i) for j, m in enumerate(news))
            if not covered:
                added.append(n)
        self.free = keep + added


# ---------------------------------------------------------------- order, policies, one pack

def make_order(w, h, weight, mode: int) -> list[int]:
    """Order of a mode: 0 long side then area, 1 area, 2 area times weight, 3 height then width, 4 width then height."""

    def key(i):
        a, b = w[i], h[i]
        if mode == 0:
            return max(a, b), a * b
        if mode == 1:
            return a * b, max(a, b)
        if mode == 2:
            return a * b * weight[i], a * b
        if mode == 3:
            return b, a
        return a, b

    return sorted(range(len(w)), key=lambda i: (-key(i)[0], -key(i)[1], i))


def _pack_gen(w, h, turn_ok, order, pol: int, s: float, page, k: int, clock=None):
    """Generator form of one pack: yields now and then, returns [(x, y, turned)] or None. `clock` counts the packs (see `_Clock`)."""

    if clock is not None:
        clock.packs += 1

    mr = MaxRects(page[0], page[1], k, pol >= ORIENTS)
    pol %= ORIENTS
    out = [(0, 0, False)] * len(w)
    rest = [0] * (len(order) + 1)  # shortest side among the boxes still to place, from each position on
    rest[len(order)] = None
    for count in range(len(order) - 1, -1, -1):
        i = order[count]
        side = min(snap(w[i] * s), snap(h[i] * s))
        rest[count] = side if rest[count + 1] is None else min(side, rest[count + 1])
    for count, i in enumerate(order):
        bw, bh = snap(w[i] * s), snap(h[i] * s)
        ok = bool(turn_ok[i])
        pre = False
        if ok and ((pol == 1 and bh > bw) or (pol == 2 and bw > bh)):
            bw, bh = bh, bw
            pre = True
        got = mr.insert(bw, bh, ok and pol == 0)
        if got is None:
            return None
        out[i] = (got[0], got[1], pre != got[2])
        if rest[count + 1] is not None:
            mr.prune(rest[count + 1])
        if count % YIELD_EVERY == YIELD_EVERY - 1:
            yield
    return out


def _run(gen):
    """Drive a generator to its end and return its value."""

    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


def pack(w, h, turn_ok, order, pol: int, s: float, page, k: int):
    """Pack every box at scale s in `order`. Returns [(x, y, turned)] per box, or None when one does not fit."""

    return _run(_pack_gen(w, h, turn_ok, order, pol, s, page, k))


def used(w, h, s: float, page) -> float:
    return sum(snap(a * s) * snap(b * s) for a, b in zip(w, h)) / (page[0] * page[1])


# ---------------------------------------------------------------- fill the page and keep going

class _Clock:
    """Time budget (seconds, 0 = none) and/or a work budget (`max_packs` packs, 0 = none): the work budget gives the same layout every time."""

    def __init__(self, budget: float, stop, max_packs: int = 0):
        self.start, self.budget, self.stop, self.max_packs, self.packs = time.perf_counter(), budget, stop, max_packs, 0

    def stopped(self) -> bool:
        return bool(self.stop is not None and self.stop())

    def over(self) -> bool:
        return (self.budget > 0.0 and time.perf_counter() - self.start > self.budget) or (self.max_packs > 0 and self.packs >= self.max_packs)

    def halt(self) -> bool:
        return self.stopped() or self.over()


def _search(w, h, turn_ok, order, pol, page, k, hint, clock):
    """Best feasible scale for one order and policy: (scale, places) or None. A generator."""

    total = sum(a * b for a, b in zip(w, h))
    if not w or total <= 0.0:
        return None
    s = hint if hint > 0.0 else sqrt(START_FILL * page[0] * page[1] / total)
    lo, hi, best = 0.0, inf, None
    while True:  # bracket
        p = yield from _pack_gen(w, h, turn_ok, order, pol, s, page, k, clock)
        if p is not None:
            lo, best = s, p
            if hi != inf or clock.halt():
                break
            s *= GROW
        else:
            hi = s
            if best is not None or clock.halt():
                break
            if all(snap(a * s) == 1 and snap(b * s) == 1 for a, b in zip(w, h)):
                return None
            s /= GROW
    if best is None:
        return None
    while hi != inf and (hi - lo) > SCALE_TOL * lo and not clock.halt():  # lo is feasible, hi never: the bracket only tightens
        mid = 0.5 * (lo + hi)
        p = yield from _pack_gen(w, h, turn_ok, order, pol, mid, page, k, clock)
        if p is not None:
            lo, best = mid, p
        else:
            hi = mid
    return lo, best


def _fill_gen(w, h, turn_ok, weight, page, k, order, policy, scale, passes, clock):
    """Fill one page (generator). Returns a dict (scale, places, order, policy, used, passes, budget_hit, stopped), or None when nothing fits."""

    n = len(w)
    inc = None  # (scale, places, order, pol)

    def consider(order_, pol, hint):
        nonlocal inc
        floor_ = inc[0] * (1.0 + MIN_GAIN) if inc else 0.0
        if floor_ > 0.0:
            probe = yield from _pack_gen(w, h, turn_ok, order_, pol, floor_, page, k, clock)
            if probe is None:
                return
        found = yield from _search(w, h, turn_ok, order_, pol, page, k, max(hint, floor_), clock)
        if found and (inc is None or found[0] > inc[0]):
            inc = (found[0], found[1], list(order_), pol)

    if order is not None and len(order) == n:
        yield from consider(list(order), min(policy, POLICIES - 1), scale)
    if inc is None:
        for mode in range(ORDERS):
            base = make_order(w, h, weight, mode)
            for pol in range(POLICIES):
                if clock.halt():
                    break
                yield from consider(base, pol, 0.0)
    if inc is None:
        return None
    s, places, order_, pol = inc
    done = 0
    while (passes < 0 or done < passes) and not clock.halt() and n > 1:
        before = s
        for i in range(n - 1):
            if clock.halt():
                break
            order_[i], order_[i + 1] = order_[i + 1], order_[i]
            target = s * (1.0 + MIN_GAIN)
            ok = yield from _pack_gen(w, h, turn_ok, order_, pol, target, page, k, clock)
            found = None
            if ok is not None:
                found = yield from _search(w, h, turn_ok, order_, pol, page, k, target, clock)
            if found and found[0] > s:
                s, places = found
            else:
                order_[i], order_[i + 1] = order_[i + 1], order_[i]
        done += 1
        if s < before * (1.0 + MIN_GAIN):
            break
    return dict(scale=s, places=places, order=order_, policy=pol, used=used(w, h, s, page), passes=done,
                budget_hit=clock.over(), stopped=clock.stopped())


def fill_page(w, h, turn_ok=None, weight=None, page=(1024, 1024), k: int = 1, order=None, policy: int = 0, scale: float = 0.0,
              passes: int = 0, budget: float = 0.0, stop=None):
    """Fill one page with boxes (unit sizes w x h): the largest scale at which they fit, plus Keep going (`passes` = swap passes
    after the variants: 0 none, negative until a pass gains nothing). `k` < 0 means no gap. Returns a dict (scale, places [(x, y, turned)], order, policy, used, passes,
    budget_hit, stopped) or None when nothing fits. Passing a result's order, policy and scale back resumes it."""

    w, h = [float(a) for a in w], [float(a) for a in h]
    n = len(w)
    turn_ok = [True] * n if turn_ok is None else [bool(a) for a in turn_ok]
    weight = [1.0] * n if weight is None else [float(a) for a in weight]
    return _run(_fill_gen(w, h, turn_ok, weight, tuple(page), int(k), order, int(policy), float(scale), int(passes), _Clock(float(budget), stop)))


class FillPage:
    """Sliceable Fill the page. `step()` works about 3 ms and returns progress 0..1 (1 when done); `result` is the dict of `fill_page`."""

    def __init__(self, w, h, turn_ok=None, weight=None, page=(1024, 1024), k: int = 1, order=None, policy: int = 0, scale: float = 0.0,
                 passes: int = 0, budget: float = 0.0, max_packs: int = 0):
        self.result, self.done, self._stopped = None, False, False
        n = len(w)
        turn_ok = [True] * n if turn_ok is None else turn_ok
        weight = [1.0] * n if weight is None else weight
        self._args = (w, h, turn_ok, weight, tuple(page), int(k), order, int(policy), float(scale), int(passes), float(budget))
        w_, h_ = [float(a) for a in w], [float(a) for a in h]
        self._gen = _fill_gen(w_, h_, [bool(a) for a in turn_ok], [float(a) for a in weight], tuple(page), int(k), order, int(policy), float(scale),
                              int(passes), _Clock(float(budget), lambda: self._stopped, int(max_packs)))

    def stop(self) -> None:
        self._stopped = True

    def step(self) -> float:
        if self.done:
            return 1.0
        end = time.perf_counter() + SLICE_SECONDS
        try:
            while time.perf_counter() < end:
                next(self._gen)
        except StopIteration as fin:
            self.result, self.done = fin.value, True
        return 1.0 if self.done else 0.0


# ---------------------------------------------------------------- irregular charts: row intervals

def _merge(v):
    out = []
    for a, b in sorted(v):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _strip_range(t, y0: float, y1: float):
    poly = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
    for keep_above, edge in ((True, y0), (False, y1)):
        def inside(p, keep_above=keep_above, edge=edge):
            return p[1] >= edge if keep_above else p[1] <= edge

        out = []
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            if inside(p):
                out.append(p)
            if inside(p) != inside(q):
                u = (edge - p[1]) / (q[1] - p[1])
                out.append((p[0] + u * (q[0] - p[0]), edge))
        poly = out
        if not poly:
            return None
    xs = [p[0] for p in poly]
    return min(xs), max(xs)


def raster(tris, y_origin: int, rows: int):
    """Conservative raster of triangles (6 numbers each): every texel a triangle touches, as merged intervals per row."""

    return _drain(raster_steps(tris, y_origin, rows))


def _drain(steps):
    """Run a generator to the end and return its value."""

    while True:
        try:
            next(steps)
        except StopIteration as end:
            return end.value


RASTER_TRIS_PER_STEP = 512  # triangles `raster_steps` rasterises between two yields
FIND_ROWS_PER_STEP = 32  # block rows `Grid.find_steps` tries between two yields


def raster_steps(tris, y_origin: int, rows: int):
    """Generator behind `raster`: yields every RASTER_TRIS_PER_STEP triangles, returns the rows."""

    acc = [[] for _ in range(rows)]
    for i in range(0, len(tris) - 5, 6):
        if i and i % (6 * RASTER_TRIS_PER_STEP) == 0:
            yield
        t = tris[i:i + 6]
        ylo, yhi = min(t[1], t[3], t[5]), max(t[1], t[3], t[5])
        r0 = int(floor(ylo))
        r1 = max(int(ceil(yhi)), r0 + 1)
        for r in range(max(r0, y_origin), min(r1, y_origin + rows)):
            rng = _strip_range(t, float(r), r + 1.0)
            if rng is not None:
                a = int(floor(rng[0]))
                acc[r - y_origin].append((a, max(int(ceil(rng[1])), a + 1)))
    return [_merge(v) for v in acc]


def dilate(rows, r: int):
    """Grow row intervals by r texels in both axes; the result has len(rows) + 2r rows and its corner is r up and left (x shifted by +r)."""

    h = len(rows)
    wide = [[(a, b + 2 * r) for a, b in v] for v in rows]
    return [_merge([iv for j in range(max(0, y - 2 * r), min(h - 1, y) + 1) for iv in wide[j]]) for y in range(h + 2 * r)]


class Grid:
    """The page as a grid of blocks of 2^k texels, one bit per block (a Python int per row). A chart owns the blocks it touches
    plus a ring of one block around them; owned blocks never overlap (the mip-clean gap rule). No gap: a block is a texel."""

    def __init__(self, nx: int, ny: int):
        self.nx, self.ny = max(nx, 1), max(ny, 1)
        self.rows = [0] * self.ny
        self.used = 0

    def height(self) -> int:
        return self.used

    def last_set(self, y: int, a: int, b: int):
        """Highest set bit in [a, b) of row y, or None."""

        if a >= b:
            return None
        m = (self.rows[y] >> a) & ((1 << (b - a)) - 1)
        return None if m == 0 else a + m.bit_length() - 1

    def next_free(self, y: int, x: int) -> int:
        """First free bit at or after x (nx when the rest of the row is taken)."""

        if x >= self.nx:
            return self.nx
        m = (~self.rows[y]) >> x
        return min(x + (m & -m).bit_length() - 1, self.nx)

    def set(self, y: int, a: int, b: int) -> None:
        a, b = max(a, 0), min(b, self.nx)
        if y < 0 or y >= self.ny or a >= b:
            return
        self.rows[y] |= ((1 << (b - a)) - 1) << a
        self.used = max(self.used, y + 1)

    def add(self, shape, x: int, y: int) -> None:
        for r, v in enumerate(shape[1]):
            for a, b in v:
                self.set(y + r, x + a, x + b)

    def _leftmost(self, shape, y: int):
        w, rows = shape[0], shape[1]
        x = 0
        while True:
            if x + w > self.nx:
                return None
            moved = False
            for r, v in enumerate(rows):
                for a, b in v:
                    p = self.last_set(y + r, x + a, x + b)
                    if p is not None:
                        x = self.next_free(y + r, p + 1) - a
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                return x

    def _waste(self, shape, x: int, y: int) -> int:
        total = 0
        for r, v in enumerate(shape[1]):
            if v:
                left = x + v[0][0]
                p = self.last_set(y + r, 0, left)
                total += left - (0 if p is None else p + 1)
        return total

    def find(self, shape, top_max=None):
        """Best spot for a shape (w, rows, area): inside the used height first (least waste), else the lowest row; only spots
        whose page height stays at most `top_max` (one above it cannot beat a spot already found for another way)."""

        return _drain(self.find_steps(shape, top_max))

    def find_steps(self, shape, top_max=None):
        """Generator behind `find`: yields every FIND_ROWS_PER_STEP rows tried, returns the spot."""

        h, used_h = len(shape[1]), self.used
        if h > self.ny or shape[0] > self.nx:
            return None
        best = None
        y = 0
        while y + h <= self.ny and (top_max is None or max(y + h, used_h) <= top_max):
            if y and y % FIND_ROWS_PER_STEP == 0:
                yield
            x = self._leftmost(shape, y)
            if x is not None:
                top = max(y + h, used_h)
                score = (top, self._waste(shape, x, y), y)
                if best is None or score < best[0]:
                    best = (score, x, y)
                if top > used_h:
                    break
            y += 1
        return None if best is None else (best[1], best[2], best[0][1])


def owned_shape(core, g: int, sx: int, sy: int, ring: int):
    """Owned blocks (w, rows, area) of a chart with texel rows `core` whose corner sits (sx, sy) texels into a block of g texels:
    the blocks it touches grown by `ring` blocks all round."""

    nrows = max((len(core) - 1 + sy) // g + 1, 1)
    acc = [[] for _ in range(nrows)]
    for y, v in enumerate(core):
        for a, b in v:
            acc[(y + sy) // g].append(((a + sx) // g, (b - 1 + sx) // g + 1))
    rows = dilate([_merge(v) for v in acc], ring)
    return max((b for v in rows for _, b in v), default=0), rows, sum(b - a for v in rows for a, b in v)


def shifts(g: int):
    """Corner offsets inside a block that are tried for a chart (at most 16, spread evenly)."""

    lst = list(range(0, g, max(g // 4, 1)))
    return [(sx, sy) for sy in lst for sx in lst]


def turn_tris(tris, quarter: int):
    out = []
    for i in range(0, len(tris) - 1, 2):
        x, y = tris[i], tris[i + 1]
        out += [(x, y), (-y, x), (-x, -y), (y, -x)][quarter % 4]
    return out


def turn_of(j):
    """(cos, sin) of angle step j (int or NumPy array): t = j / TURN_STEPS, ((1 - t^2) / (1 + t^2), 2t / (1 + t^2))."""

    t = j / float(TURN_STEPS)
    d = 1.0 + t * t
    return (1.0 - t * t) / d, (2.0 * t) / d


def spin_tris(tris, j: int):
    """Points turned by angle step j ((x, y) -> (c x - s y, s x + c y)); j = 0 leaves them."""

    if j == 0:
        return list(tris)
    c, s = turn_of(j)
    out = []
    for i in range(0, len(tris) - 1, 2):
        x, y = tris[i], tris[i + 1]
        out += [c * x - s * y, s * x + c * y]
    return out


def hull(pts):
    """Convex hull of x, y pairs, counter-clockwise (Andrew 1979, monotone chain)."""

    p = sorted(set((pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)))
    if len(p) < 3:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    h = []
    for seq in (p, p[::-1]):
        start = len(h)
        for q in seq:
            while len(h) >= start + 2 and cross(h[-2], h[-1], q) <= 0.0:
                h.pop()
            h.append(q)
        h.pop()
    return h


def _round_half_away(x: float) -> int:
    """Round halves away from zero."""

    return int(floor(x + 0.5)) if x >= 0 else -int(floor(-x + 0.5))


def free_angles(tris, effort: int):
    """Angle steps for a chart (0 first): `FREE_SPLIT << effort` even steps of t and the angles laying its
    `HULL_EDGES << effort` longest hull edges along an axis (t = dx / (len + dy), the half-angle formula). Sorted, no repeats."""

    n = min(FREE_SPLIT << effort, TURN_STEPS)
    out = {i * TURN_STEPS // n for i in range(n)}
    h = hull(tris)
    edges = []
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx * dx + dy * dy > 0.0:
            edges.append((dx * dx + dy * dy, dx, dy))
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))
    for l2, dx, dy in edges[:HULL_EDGES << effort]:
        while not (dx > 0.0 and dy >= 0.0):
            dx, dy = dy, -dx
        t = dx / (sqrt(l2) + dy)
        out.add(_round_half_away(t * TURN_STEPS) % TURN_STEPS)
    return sorted(out)


PLACE_EVERY = 1  # charts `pack_into_steps` places between two yields (each find also yields every FIND_ROWS_PER_STEP rows)


def pack_into_steps(page, k: int, seed, tris, tri_chart, charts: int, as_box=None, turn_ok=None, stop=None):
    """Generator: yields between short pieces (every few raster rows, every way of a chart worked out, every chart placed)
    and returns [(ok, dx, dy, quarter turns)] per chart. Pack charts (`tris` in texels, 6 numbers per triangle, `tri_chart`
    per triangle) into a page that may hold `seed` triangles.

    The page is a grid of blocks of 2^k texels (no gap, k < 0: texels); a chart owns the blocks it touches plus one ring of
    blocks around them and owned blocks never overlap: the mip-clean gap rule (neighbours keep two free blocks between the
    blocks they touch) without wasting a texel beyond it. Each chart is tried turned by every quarter, at the corner offset
    inside a block that owns the fewest blocks; `as_box` charts reserve their bounding box, the rest follow their outline,
    so a small chart can sit in a big one's bay. Order: most blocks owned first (other orders with more effort). The same input gives the same layout (no clock).

    Returns [(ok, dx, dy, quarter)] per chart: turn the chart's points `quarter` x 90 degrees ((x, y) -> (-y, x)), then add (dx, dy).
    """

    free = k >= FREE_TURNS - 1
    if free:
        k -= FREE_TURNS
    k, effort = (k + 1) % EFFORT_STEP - 1, min(max((k + 1) // EFFORT_STEP, 0), MAX_EFFORT)
    orders = min(effort, len(ORDER_MODES) - 1)
    g, ring = (1, 0) if k < 0 else (block(k), 1)
    base = Grid(page[0] // g, page[1] // g)
    if len(seed):
        ys = [seed[i] for i in range(1, len(seed), 2)]
        y0 = int(floor(min(ys)))
        rows = max(int(ceil(max(ys))) + 1 - y0, 1)
        acc = {}
        for y, v in enumerate((yield from raster_steps(seed, y0, rows))):
            for a, b in v:
                acc.setdefault((y0 + y) // g, []).append((a // g, (b - 1) // g + 1))
        for j, v in acc.items():
            for a, b in _merge(v):
                for dj in range(-ring, ring + 1):
                    base.set(j + dj, a - ring, b + ring)
    per = [[] for _ in range(charts)]
    for t, c in enumerate(tri_chart):
        if 0 <= c < charts:
            per[c] += list(tris[6 * t:6 * t + 6])
    as_box = [False] * charts if as_box is None else as_box
    turn_ok = [True] * charts if turn_ok is None else turn_ok

    def extent(t):
        xs, ys = t[0::2], t[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    def key(c):
        e = extent(per[c])
        w, h = e[2] - e[0], e[3] - e[1]
        return max(w, h), w * h

    order = sorted((c for c in range(charts) if per[c]), key=lambda c: (-key(c)[0], -key(c)[1], c))
    offsets = shifts(g)
    few = [(sx, sy) for sx, sy in offsets if sx % max(g // 2, 1) == 0 and sy % max(g // 2, 1) == 0]
    spun = set()
    if free:  # the first `2 << effort` charts of the order (the biggest) may turn by any angle; Keep going adds more
        spun = set([c for c in order if turn_ok[c] and not as_box[c]][:2 << effort])
    ways = [[] for _ in range(charts)]  # every way a chart can sit (angle, quarter, best corner offset); it does not depend on the page
    for c in order:
        spins = free_angles(per[c], effort) if c in spun else [0]
        quarters = (0, 1, 2, 3) if (turn_ok[c] or free) else (0,)
        for j in spins:
            for q in quarters:
                t = turn_tris(spin_tris(per[c], j), q)
                e = extent(t)
                fx, fy = int(floor(e[0])), int(floor(e[1]))
                rows = max(int(ceil(e[3])), fy + 1) - fy
                core = raster([v - (fx if i % 2 == 0 else fy) for i, v in enumerate(t)], 0, rows)
                if as_box[c]:
                    cw = max((b for v in core for _, b in v), default=1)
                    core = [[(0, cw)] for _ in range(rows)]
                pick = None
                for sx, sy in (few if j != 0 else offsets):
                    shape = owned_shape(core, g, sx, sy, ring)
                    if pick is None or shape[2] < pick[0][2]:
                        pick = (shape, sx, sy)
                ways[c].append((pick[0], pick[1], pick[2], fx, fy, q + 4 * j))
                yield

    def place(seq, root=None, quarters=False):
        """Place the charts in `seq` (`root` = (chart, way index) forces that way); returns (result per chart, the charts that
        found no room in that order). A forced start ends at its first chart without room (it only answers yes or no)."""

        occ = Grid(page[0] // g, page[1] // g)
        occ.rows, occ.used = list(base.rows), base.used
        out = [(False, 0, 0, 0)] * charts
        failed = []
        for n_, c in enumerate(seq):
            if n_ % PLACE_EVERY == 0:
                yield
            if stop is not None and stop():
                failed.append(c)
                continue
            best = None
            for wi, way in enumerate(ways[c]):
                if (root is not None and root[0] == c and root[1] != wi) or (quarters and way[5] >= 4):
                    continue
                found = yield from occ.find_steps(way[0], None if best is None else best[0][0])
                if found is not None:
                    x, y, waste = found
                    score = (max(y + len(way[0][1]), occ.height()), waste, y, way[5])
                    if best is None or score < best[0]:
                        best = (score, way, x, y)
            if best is None:
                failed.append(c)
                if root is not None:
                    failed += [o for o in seq if not out[o][0] and o != c]
                    break
                continue
            _, (shape, sx, sy, fx, fy, q), x, y = best
            occ.add(shape, x, y)
            out[c] = (True, (x + ring) * g + sx - fx, (y + ring) * g + sy - fy, q)
        return out, failed

    def solve(seq, root=None, quarters=False):
        """One order with repair rounds (not for a forced start): charts that found no room go first in the next round; go on
        while a round leaves fewer of them."""

        out, failed = yield from place(seq, root, quarters)
        while root is None and failed and not (stop is not None and stop()):
            lost = set(failed)
            nxt = failed + [c for c in seq if c not in lost]
            out2, failed2 = yield from place(nxt, root, quarters)
            if len(failed2) >= len(failed):
                break
            seq, out, failed = nxt, out2, failed2
        return out, failed

    own = lambda c: min(w[0][2] for w in ways[c])  # noqa: E731
    longer = lambda c: min(max(len(w[0][1]), w[0][0]) for w in ways[c])  # noqa: E731
    bbox = lambda c: min(len(w[0][1]) * w[0][0] for w in ways[c])  # noqa: E731
    best = None
    for mode in ORDER_MODES[:1 + orders]:
        if mode == 0:
            seq = sorted(order, key=lambda c: (-own(c), c))
        elif mode == 1:
            seq = sorted(order, key=lambda c: (-bbox(c), c))
        elif mode == 2:
            seq = sorted(order, key=lambda c: (-longer(c), -own(c), c))
        else:
            seq = list(order)
        for quarters in ((True, False) if free else (False,)):  # quarter turns first (the layout without free turns), then angles
            out, failed = yield from solve(seq, None, quarters)
            if not failed:
                return out
            if best is None or len(failed) < best[1]:
                best = (out, len(failed))
        if free and seq and len(ways[seq[0]]) > 4 and own(seq[0]) * (BEAM_SHARE << effort) >= base.nx * base.ny:  # every way of the first chart as the start; the first that places all wins
            for wi in range(len(ways[seq[0]])):
                out, failed = yield from solve(seq, (seq[0], wi))
                if not failed:
                    return out
        if stop is not None and stop():
            break
    return best[0] if best else [(False, 0, 0, 0)] * charts
