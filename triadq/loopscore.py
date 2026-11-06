# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Loop Quality: how straight, even and clean the edge loops of a quad mesh are, the way an artist reads them.

A loop is a chain of edges walked straight on through every corner with four quads around it (as Blender's loop
select does) until it reaches a pole, a border or its own start. Every loop is scored for:

- straight: the turn at each loop corner, measured in the surface (the tangent plane of the corner), so a row that
  follows a curved surface but runs straight across it counts as straight;
- smooth: zig-zags, i.e. the turn flipping side from one corner to the next;
- spacing: how evenly two neighbouring loops sit apart along their whole length (the rungs between them);
- width: how even the quads are along a loop;
- square: corner angles near 90 degrees and sides near equal;
- poles: how many there are and whether they bunch up;
- flow: loops that die after a few quads (stubs) and loops that cross themselves (spirals).

`score` puts these on 0-100 each and a weighted "Loop Quality". The rules and targets are our own reading of artist
practice (bench notes); no other tool's code or numbers are used. Plain NumPy.
"""

from __future__ import annotations

import numpy as np

from .patterns import Topology

TURN_WORST = 25.0  # mean loop turn (degrees) that scores 0 for straightness
FLIP_MIN = 3.0  # degrees: smaller turns are noise, not a zig-zag
SPACING_WORST = 0.5  # coefficient of variation of the gap between two loops that scores 0
WIDTH_WORST = 0.5  # coefficient of variation of edge lengths along a loop that scores 0
SKEW_WORST = 45.0  # mean corner angle off 90 degrees that scores 0
POLE_WORST = 0.3  # share of vertices that are poles that scores 0
CLUSTER = 2  # edges: poles this close to another pole are bunched up
STUB = 6  # edges: an open loop shorter than this is a stub
SKEWED = 25.0  # degrees: a quad corner further off 90 than this is slanted
WEIGHTS = {"straight": 0.25, "smooth": 0.1, "spacing": 0.2, "width": 0.15, "square": 0.1, "poles": 0.1, "flow": 0.1}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def _continuations(topo: Topology) -> dict[int, int]:
    """{a * n + b: c}: walking a loop from a to b, it carries on to c (b has four quads round it and no border)."""
    q, n = topo.quads, len(topo.points)
    a, b = q.reshape(-1), np.roll(q, -1, axis=1).reshape(-1)
    nxt = np.arange(len(a))
    nxt[3::4] -= 4
    nxt += 1  # the next half-edge round the same quad
    key = a * n + b
    order = np.argsort(key)
    at = np.minimum(np.searchsorted(key[order], b * n + a), len(key) - 1)
    twin = np.where(key[order][at] == b * n + a, order[at], -1)
    step = twin[nxt]  # the other side of the edge after h
    ok = (step >= 0) & (topo.code[b] == 4) & ~topo.border[b]
    h = np.flatnonzero(ok)
    c = b[nxt[step[h]]]
    out = dict(zip((a[h] * n + b[h]).tolist(), c.tolist()))
    # A straight border is a loop too: at a border corner with two quads it carries on along the border.
    rim = np.flatnonzero(topo.border_edges)
    ends: dict[int, list[int]] = {}
    for x, y in zip(topo.edge_a[rim].tolist(), topo.edge_b[rim].tolist()):
        ends.setdefault(x, []).append(y)
        ends.setdefault(y, []).append(x)
    for v, near in ends.items():
        if len(near) == 2 and topo.faces_at[v] == 2:
            out[near[0] * n + v], out[near[1] * n + v] = near[1], near[0]
    return out


def walk(topo: Topology) -> tuple[list[list[int]], list[bool], np.ndarray]:
    """(loops as vertex lists, closed per loop, loop id per edge of `topo.edges`). A closed loop lists its start once."""
    n = len(topo.points)
    go = _continuations(topo)
    edge_id = {int(k): i for i, k in enumerate(topo.edges.tolist())}
    label = np.full(len(topo.edges), -1, dtype=np.int64)
    loops, closed = [], []
    for e in range(len(topo.edges)):
        if label[e] >= 0:
            continue
        a0, b0 = int(topo.edge_a[e]), int(topo.edge_b[e])
        ahead = [a0, b0]
        ring = False
        while True:
            c = go.get(ahead[-2] * n + ahead[-1])
            if c is None:
                break
            if ahead[-1] == a0 and c == b0:  # back on the first edge
                ring = True
                ahead.pop()
                break
            ahead.append(c)
            if len(ahead) > 4 * len(topo.edges):  # cannot happen on a valid mesh; a guard all the same
                break
        if not ring:
            behind = [b0, a0]
            while True:
                c = go.get(behind[-2] * n + behind[-1])
                if c is None or (behind[-1] == b0 and c == a0):
                    break
                behind.append(c)
                if len(behind) > 4 * len(topo.edges):
                    break
            ahead = behind[:1:-1] + ahead
        k = len(loops)
        verts = ahead + ahead[:1] if ring else ahead
        for u, v in zip(verts[:-1], verts[1:]):
            label[edge_id[min(u, v) * n + max(u, v)]] = k
        loops.append(ahead)
        closed.append(ring)
    return loops, closed, label


def _pole_gaps(topo: Topology) -> np.ndarray:
    """Per pole, the number of edges to the nearest other pole (capped at 9)."""
    poles = np.flatnonzero(topo.poles)
    if len(poles) < 2:
        return np.full(len(poles), 9, dtype=np.int64)
    n = len(topo.points)
    dist = np.full(n, 99, dtype=np.int64)
    owner = np.full(n, -1, dtype=np.int64)
    dist[poles], owner[poles] = 0, np.arange(len(poles))
    ea, eb = topo.edge_a, topo.edge_b
    for _ in range(5):  # nearest-pole regions grown a ring at a time
        for s, t in ((ea, eb), (eb, ea)):
            better = dist[s] + 1 < dist[t]
            if better.any():
                order = np.argsort(dist[s][better], kind="stable")[::-1]  # the nearest wins when written last
                tt, ss = t[better][order], s[better][order]
                dist[tt], owner[tt] = dist[ss] + 1, owner[ss]
    gap = np.full(len(poles), 9, dtype=np.int64)
    meet = (owner[ea] != owner[eb]) & (owner[ea] >= 0) & (owner[eb] >= 0)
    d = np.minimum(dist[ea[meet]] + dist[eb[meet]] + 1, 9)
    np.minimum.at(gap, owner[ea[meet]], d)
    np.minimum.at(gap, owner[eb[meet]], d)
    return gap


def symmetry(points: np.ndarray, valence: np.ndarray, used: np.ndarray, length: float, plane=None) -> float:
    """Share of vertices with a mirror partner (same valence) within 0.3 mean edges, across `plane` (axis, offset) or
    the best of the three axis planes through the middle of the box."""
    p = points[used]
    if len(p) < 4:
        return 1.0
    val = valence[used]
    if len(p) > 20000:  # a fixed sample is plenty for a share
        pick = np.random.default_rng(0).choice(len(p), 20000, replace=False)
    else:
        pick = np.arange(len(p))
    r = 0.3 * length
    cells: dict[tuple, list[int]] = {}
    for i, c in enumerate(np.floor(p / r).astype(np.int64).tolist()):
        cells.setdefault(tuple(c), []).append(i)
    middle = 0.5 * (p.min(axis=0) + p.max(axis=0))
    planes = [plane] if plane is not None else [(k, float(middle[k])) for k in range(3)]
    best = 0.0
    for axis, offset in planes:
        m = p[pick].copy()
        m[:, axis] = 2.0 * offset - m[:, axis]
        hits = 0
        for i, (x, cell) in enumerate(zip(m, np.floor(m / r).astype(np.int64).tolist())):
            want = val[pick[i]]
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                            if val[j] == want and float(((p[j] - x) ** 2).sum()) <= r * r:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            hits += found
        best = max(best, hits / len(pick))
    return best


def measure(co, faces, plane=None, with_symmetry: bool = True) -> dict:
    """The loop numbers of a mesh (see the module text). `plane`: (axis, offset) of a known mirror."""
    topo = Topology(co, faces)
    out = {"loops": 0}
    if not len(topo.quads):
        return out
    p, vn = topo.points, topo.vertex_normal
    loops, closed, label = walk(topo)
    length = np.linalg.norm(p[topo.edge_a] - p[topo.edge_b], axis=1)
    mean_edge = float(length.mean())
    # Turn at each loop corner, in the tangent plane of the corner.
    prev, mid, nxt, owner, seq = [], [], [], [], []
    for k, (vs, ring) in enumerate(zip(loops, closed)):
        if ring and len(vs) >= 3:
            prev += vs[-1:] + vs[:-1]
            mid += vs
            nxt += vs[1:] + vs[:1]
            owner += [k] * len(vs)
            seq += list(range(len(vs)))
        elif len(vs) >= 3:
            prev += vs[:-2]
            mid += vs[1:-1]
            nxt += vs[2:]
            owner += [k] * (len(vs) - 2)
            seq += list(range(len(vs) - 2))
    prev, mid, nxt, owner = (np.asarray(x, dtype=np.int64) for x in (prev, mid, nxt, owner))
    n = vn[mid]
    d1, d2 = p[mid] - p[prev], p[nxt] - p[mid]
    d1 -= (d1 * n).sum(axis=1, keepdims=True) * n
    d2 -= (d2 * n).sum(axis=1, keepdims=True) * n
    turn = np.degrees(np.arctan2((np.cross(d1, d2) * n).sum(axis=1), (d1 * d2).sum(axis=1)))
    mag = np.abs(turn)
    # Zig-zag: the turn flips side between two corners next to each other on the same loop.
    same = owner[1:] == owner[:-1]
    both = same & (mag[1:] > FLIP_MIN) & (mag[:-1] > FLIP_MIN)
    flips = int((both & (np.sign(turn[1:]) != np.sign(turn[:-1]))).sum())
    flip_rate = flips / max(int(same.sum()), 1)
    # Width along loops: the edge lengths of each loop.
    widths, jumps, pairs = [], 0, 0
    for k, (vs, ring) in enumerate(zip(loops, closed)):
        chain = vs + vs[:1] if ring else vs
        if len(chain) < 5:
            continue
        e = np.linalg.norm(p[chain[1:]] - p[chain[:-1]], axis=1)
        widths.append((float(e.std() / max(e.mean(), 1e-30)), len(e)))
        ratio = np.maximum(e[1:], e[:-1]) / np.maximum(np.minimum(e[1:], e[:-1]), 1e-30)
        jumps += int((ratio > 1.5).sum())
        pairs += len(ratio)
    width_cv = _weighted_median(widths)
    # Spacing between neighbouring loops: the rungs of the quads between them.
    q_edges = topo.edge_of.reshape(-1, 4)
    lab, ln = label[q_edges], length[q_edges]
    side_a = np.concatenate((lab[:, 0], lab[:, 1]))
    side_b = np.concatenate((lab[:, 2], lab[:, 3]))
    rung1 = np.concatenate((ln[:, 1], ln[:, 0]))
    rung2 = np.concatenate((ln[:, 3], ln[:, 2]))
    keep = side_a != side_b
    lo, hi = np.minimum(side_a, side_b)[keep], np.maximum(side_a, side_b)[keep]
    rungs = np.concatenate((rung1[keep], rung2[keep]))
    keys, inv = np.unique(np.concatenate((lo, lo)) * (len(loops) + 1) + np.concatenate((hi, hi)), return_inverse=True)
    cnt = np.bincount(inv, minlength=len(keys))
    s1 = np.bincount(inv, rungs, minlength=len(keys))
    s2 = np.bincount(inv, rungs * rungs, minlength=len(keys))
    mean = s1 / np.maximum(cnt, 1)
    cv = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mean * mean, 0.0)) / np.maximum(mean, 1e-30)
    long = cnt >= 6  # three quads or more between the two loops
    spacing_cv = _weighted_median(list(zip(cv[long].tolist(), cnt[long].tolist())))
    # Quad shape.
    qp = p[topo.quads]
    out_e, back_e = np.roll(qp, -1, axis=1) - qp, np.roll(qp, 1, axis=1) - qp
    cos = (out_e * back_e).sum(axis=2) / np.maximum(np.linalg.norm(out_e, axis=2) * np.linalg.norm(back_e, axis=2), 1e-30)
    off = np.abs(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) - 90.0)
    side = np.linalg.norm(out_e, axis=2)
    s_a, s_b = side[:, 0] + side[:, 2], side[:, 1] + side[:, 3]
    aspect = np.maximum(s_a, s_b) / np.maximum(np.minimum(s_a, s_b), 1e-30)
    # Poles and loop ends.
    gaps = _pole_gaps(topo)
    sizes = np.asarray([len(vs) if ring else len(vs) - 1 for vs, ring in zip(loops, closed)])
    is_open = ~np.asarray(closed, dtype=bool)
    spirals = sum(1 for vs in loops if len(set(vs)) < len(vs))
    used = int(topo.used.sum())
    per_loop = np.bincount(owner, mag, minlength=len(loops)) / np.maximum(np.bincount(owner, minlength=len(loops)), 1)
    counted = np.flatnonzero(np.bincount(owner, minlength=len(loops)) >= 3)
    worst = counted[np.argsort(-per_loop[counted])][: max(1, len(counted) // 20)] if len(counted) else counted
    out.update(
        loops=len(loops),
        closed=int(sum(closed)),
        turn_mean=float(mag.mean()) if len(mag) else 0.0,
        turn_p95=float(np.percentile(mag, 95)) if len(mag) else 0.0,
        turn_over20=float((mag > 20.0).mean()) if len(mag) else 0.0,
        zigzag=float(flip_rate),
        spacing_cv=spacing_cv,
        width_cv=width_cv,
        width_jumps=jumps / max(pairs, 1),
        aspect_median=float(np.median(aspect)),
        aspect_over2=float((aspect > 2.0).mean()),
        skew_mean=float(off.mean()),
        skewed=float((off.max(axis=1) > SKEWED).mean()),
        poles=int(topo.poles.sum()),
        pole_share=float(topo.poles.sum()) / max(used, 1),
        pole_gap=float(gaps.mean()) if len(gaps) else 9.0,
        pole_bunched=float((gaps <= CLUSTER).mean()) if len(gaps) else 0.0,
        stubs=float(((sizes < STUB) & is_open).sum()) / max(len(loops), 1),
        spirals=spirals,
        mean_loop=float(sizes.mean()),
        worst=worst.tolist(),
    )
    if with_symmetry:
        out["symmetry"] = symmetry(p, topo.valence, topo.used, mean_edge, plane)
    return out


def _weighted_median(rows) -> float:
    if not rows:
        return 0.0
    vals = np.asarray([r[0] for r in rows])
    w = np.asarray([r[1] for r in rows], dtype=np.float64)
    order = np.argsort(vals)
    c = np.cumsum(w[order])
    return float(vals[order][np.searchsorted(c, 0.5 * c[-1])])


def score(m: dict) -> dict:
    """0-100 per part and the weighted "quality" from `measure`'s numbers."""
    if not m.get("loops"):
        return {"quality": 0.0, "parts": {}}
    clip = lambda x: 100.0 * min(1.0, max(0.0, x))  # noqa: E731
    parts = {
        "straight": clip(1.0 - m["turn_mean"] / TURN_WORST),
        "smooth": clip(1.0 - 2.0 * m["zigzag"]),
        "spacing": clip(1.0 - m["spacing_cv"] / SPACING_WORST),
        "width": clip(1.0 - m["width_cv"] / WIDTH_WORST),
        "square": clip(1.0 - m["skew_mean"] / SKEW_WORST),
        "poles": clip((1.0 - m["pole_share"] / POLE_WORST) * (1.0 - 0.5 * m["pole_bunched"])),
        "flow": clip(1.0 - m["stubs"] - 0.02 * m["spirals"]),
    }
    quality = sum(WEIGHTS[k] * v for k, v in parts.items()) / sum(WEIGHTS.values())
    return {"quality": round(quality, 1), "parts": {k: round(v, 1) for k, v in parts.items()}}


def loop_quality(co, faces, plane=None, with_symmetry: bool = False) -> dict:
    """`measure` and `score` in one: {"quality", "parts", "numbers"}."""
    m = measure(co, faces, plane, with_symmetry)
    return dict(score(m), numbers=m)


def meter(before: dict, after: dict) -> str:
    """'Loop Quality 41 -> 63 (straight 40 -> 70, even spacing 52 -> 80)': straight rows and evenly stacked rows are
    what artists look at first."""
    a, b = before.get("parts", {}), after.get("parts", {})
    detail = ", ".join(f"{label} {a[k]:.0f} -> {b[k]:.0f}" for k, label in (("straight", "straight"), ("spacing", "even spacing")) if k in a and k in b)
    return f"Loop Quality {before['quality']:.0f} -> {after['quality']:.0f}" + (f" ({detail})" if detail else "")


__all__ = ["loop_quality", "measure", "meter", "score", "symmetry", "walk"]
