# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Island packer twin (package K7).

NumPy-free pure Python, no bpy. The reference is the kernel source `uvpack.rs`; this twin covers the same algorithm for
boxes and for charts packed into an existing layout, step for step, so the two give the same layouts:

- MaxRects best-short-side-fit (Jylanki 2010, "A thousand ways to pack the bin"), 90 degree steps only.
- The gap rule: a mip-k texel is a block of G = 2^k texels. Two neighbours are mip-clean when the blocks they touch, each
  grown by one block, do not meet. A box *aligned* on a block edge needs G of gap on each side, any other (an outline, a
  chart of an existing layout) 2G - 1: aligned + aligned = 2G, aligned + loose = 3G - 1, loose + loose = 4G - 2. The box
  packer always aligns; the loose gap is for outlines and for charts already in a layout.
- Irregular charts: conservative row intervals per texel row, dilated by the gap in both axes (Minkowski sum with a square),
  bottom-left free-interval search (Chazelle 1983); score: least new page height, then least wasted interval length.
- Fill the page: the largest uniform scale at which everything fits, by a bracket that only tightens (growth steps, then
  bisection), so the best feasible layout is kept and nothing oscillates. Keep going: other orders and orientation
  policies, then adjacent swaps kept only when the page then fits a larger scale.

`FillPage` is sliceable: `step()` does about 3 ms of work and returns progress. When the adapter installed a faster
`fill_page` callable, the stage hands the page to it (a job that never blocks) and only polls it.
"""

from __future__ import annotations

from math import ceil, floor, inf, sqrt
import time

# Constants table: same names and values as the top of the matching .rs file (tests/test_uv_scaffold.py compares).
START_FILL = 0.5
GROW = 1.25
SCALE_TOL = 0.00001
MIN_GAIN = 0.0001
ORDERS = 5
POLICIES = 4
MAX_K = 20

SLICE_SECONDS = 0.003  # one step() of a stage works about this long
YIELD_EVERY = 1  # a pack gives control back after this many boxes

# Optional faster callables, set by the one adapter (uv_fast.py); the Python code runs when a name is missing.
FAST: dict = {}


def set_fast(**callables) -> None:
    """Install (or, with None, remove) faster callables by name."""

    for key, fn in callables.items():
        if fn is None:
            FAST.pop(key, None)
        else:
            FAST[key] = fn


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
    def __init__(self, w: int, h: int, k: int):
        self.g = block(k)
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
                score = (min(rx, ry), max(rx, ry), ay[1], ax[1])
                if best is None or score < best[0]:
                    best = (score, (ax[0], ay[0], ax[1], ay[1]), (ax[2], ay[2]), turned)
        if best is None:
            return None
        _, p, (cx, cy), turned = best
        self._split(p)
        return cx, cy, turned

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


def _pack_gen(w, h, turn_ok, order, pol: int, s: float, page, k: int):
    """Generator form of one pack: yields now and then, returns [(x, y, turned)] or None."""

    mr = MaxRects(page[0], page[1], k)
    out = [(0, 0, False)] * len(w)
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
    def __init__(self, budget: float, stop):
        self.start, self.budget, self.stop = time.perf_counter(), budget, stop

    def stopped(self) -> bool:
        return bool(self.stop is not None and self.stop())

    def over(self) -> bool:
        return self.budget > 0.0 and time.perf_counter() - self.start > self.budget

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
        p = yield from _pack_gen(w, h, turn_ok, order, pol, s, page, k)
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
        p = yield from _pack_gen(w, h, turn_ok, order, pol, mid, page, k)
        if p is not None:
            lo, best = mid, p
        else:
            hi = mid
    return lo, best


def _fill_gen(w, h, turn_ok, weight, page, k, order, policy, scale, passes, clock):
    """Fill one page (generator). Returns a dict like the faster call's result, or None when nothing fits."""

    n = len(w)
    inc = None  # (scale, places, order, pol)

    def consider(order_, pol, hint):
        nonlocal inc
        floor_ = inc[0] * (1.0 + MIN_GAIN) if inc else 0.0
        if floor_ > 0.0:
            probe = yield from _pack_gen(w, h, turn_ok, order_, pol, floor_, page, k)
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
            ok = yield from _pack_gen(w, h, turn_ok, order_, pol, target, page, k)
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
                 passes: int = 0, budget: float = 0.0, use_fast: bool = True):
        self.result, self.done, self._job, self._stopped = None, False, None, False
        n = len(w)
        turn_ok = [True] * n if turn_ok is None else turn_ok
        weight = [1.0] * n if weight is None else weight
        fast = FAST.get("fill_page") if use_fast else None
        self._args = (w, h, turn_ok, weight, tuple(page), int(k), order, int(policy), float(scale), int(passes), float(budget))
        if fast is not None:
            self._job = fast(w, h, turn_ok, weight, int(page[0]), int(page[1]), int(k), order, int(policy), float(scale), int(passes), float(budget))
        if self._job is None:
            w_, h_ = [float(a) for a in w], [float(a) for a in h]
            self._gen = _fill_gen(w_, h_, [bool(a) for a in turn_ok], [float(a) for a in weight], tuple(page), int(k), order, int(policy), float(scale),
                                  int(passes), _Clock(float(budget), lambda: self._stopped))

    def stop(self) -> None:
        self._stopped = True
        if self._job is not None:
            self._job.stop()

    def step(self) -> float:
        if self.done:
            return 1.0
        if self._job is not None:
            if self._job.done():
                info, x, y, t, order = self._job.result()
                if info[6]:
                    self.result = dict(scale=float(info[0]), places=[(int(a), int(b), bool(c)) for a, b, c in zip(x, y, t)], order=[int(i) for i in order],
                                       policy=int(info[2]), used=float(info[1]), passes=int(info[3]), budget_hit=bool(info[4]), stopped=bool(info[5]))
                self.done = True
            return 1.0 if self.done else 0.0
        end = time.perf_counter() + SLICE_SECONDS
        try:
            while time.perf_counter() < end:
                next(self._gen)
        except StopIteration as fin:
            self.result, self.done = fin.value, True
        return 1.0 if self.done else 0.0


def pages_for(labels, area, centres, count: int):
    """Which texture page each face goes to: whole parts together, pages of about equal area (calls `texsets.texture_sets`)."""

    from . import texsets

    return texsets.texture_sets(labels, area, centres, count)


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

    acc = [[] for _ in range(rows)]
    for i in range(0, len(tris) - 5, 6):
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


class Page:
    """The page's occupied texels per row."""

    def __init__(self, w: int, h: int):
        self.w, self.rows = w, [[] for _ in range(h)]

    def add(self, y0: int, rows, dx: int) -> None:
        for j, v in enumerate(rows):
            y = y0 + j
            if 0 <= y < len(self.rows):
                self.rows[y] = _merge(self.rows[y] + [(a + dx, b + dx) for a, b in v])

    def height(self) -> int:
        for y in range(len(self.rows) - 1, -1, -1):
            if self.rows[y]:
                return y + 1
        return 0

    def _leftmost(self, sw: int, srows, y: int, step: int):
        x = 0
        while True:
            if x + sw > self.w:
                return None
            moved = False
            for r, v in enumerate(srows):
                for a, b in v:
                    for c, d in self.rows[y + r]:
                        if c < x + b and d > x + a:
                            x = _ceil_div(d - a, step) * step
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    break
            if not moved:
                return x

    def _waste(self, srows, x: int, y: int) -> int:
        total = 0
        for r, v in enumerate(srows):
            if v:
                left = x + v[0][0]
                total += left - max([d for _, d in self.rows[y + r] if d <= left], default=0)
        return total

    def find(self, sw: int, srows, step: int):
        h, used_h = len(srows), self.height()
        if h > len(self.rows) or sw > self.w:
            return None
        best = None
        y = 0
        while y + h <= len(self.rows):
            x = self._leftmost(sw, srows, y, step)
            if x is not None:
                top = max(y + h, used_h)
                score = (top, self._waste(srows, x, y), y)
                if best is None or score < best[0]:
                    best = (score, x, y)
                if top > used_h:
                    break
            y += step
        return None if best is None else (best[1], best[2], best[0][1])


def turn_tris(tris, quarter: int):
    out = []
    for i in range(0, len(tris) - 1, 2):
        x, y = tris[i], tris[i + 1]
        out += [(x, y), (-y, x), (-x, -y), (y, -x)][quarter % 4]
    return out


def pack_into(page, k: int, seed, tris, tri_chart, charts: int, as_box=None, turn_ok=None, stop=None):
    """Pack charts (`tris` in texels, 6 numbers per triangle, `tri_chart` per triangle) into a page that may hold `seed` triangles.

    Returns [(ok, dx, dy, quarter)] per chart: turn the chart's points `quarter` x 90 degrees ((x, y) -> (-y, x)), then add (dx, dy).
    """

    g = block(k)
    loose, tight = (0, 0) if g == 0 else (2 * g - 1, g)
    occ = Page(page[0], page[1])
    if len(seed):
        ys = [seed[i] for i in range(1, len(seed), 2)]
        y0 = int(floor(min(ys)))
        rows = max(int(ceil(max(ys))) + 1 - y0, 1)
        occ.add(y0 - loose, dilate(raster(seed, y0, rows), loose), -loose)
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
    out = [(False, 0, 0, 0)] * charts
    for c in order:
        if stop is not None and stop():
            break
        best = None
        for q in ((0, 1, 2, 3) if turn_ok[c] else (0,)):
            t = turn_tris(per[c], q)
            e = extent(t)
            fx, fy = int(floor(e[0])), int(floor(e[1]))
            rows = max(int(ceil(e[3])), fy + 1) - fy
            core = raster([v - (fx if i % 2 == 0 else fy) for i, v in enumerate(t)], 0, rows)
            cw = max((b for v in core for _, b in v), default=1)
            boxed = bool(as_box[c])
            if boxed:
                core = [[(0, cw)] for _ in range(rows)]
            variants = [(loose, 1, core, cw)]
            if boxed and g > 0:
                variants.insert(0, (tight, g, core, cw))
            for r, step, srows, w in variants:
                dil = dilate(srows, r)
                found = occ.find(w + 2 * r, dil, step)
                if found is not None:
                    x, y, waste = found
                    score = (max(y + len(dil), occ.height()), waste, y, q)
                    if best is None or score < best[0]:
                        best = (score, dil, x, y, r, q, fx, fy)
        if best is not None:
            _, dil, x, y, r, q, fx, fy = best
            occ.add(y, dil, x)
            out[c] = (True, x + r - fx, y + r - fy, q)
    return out
