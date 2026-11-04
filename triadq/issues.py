# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Issue Finder: group the faces the checks flag into spots, rank the worst, and compare two runs.

`topology_flags` adds the kinds that are about tight or untidy topology (slivers, pinched faces, poles, concave quads) to
the defect kinds `checks.defects` already finds per face. `find_spots` joins flagged faces that touch or lie close together
into spots with a score, worst first. `report` makes that JSON-able and `compare` says which spots of an older report are
fixed, still there, or new. Pure numpy: no Blender needed."""

from __future__ import annotations

import math

import numpy as np

DEFECT_KINDS = ("inward", "sunk", "crossing", "folded_over", "bent", "flat", "duplicate", "overlap")
TOPOLOGY_KINDS = ("sliver", "pinched", "pole", "concave")

SLIVER_ANGLE = 10.0  # degrees: a face with a corner sharper than this is a sliver
SLIVER_ASPECT = 8.0  # longest side over shortest side
PINCHED_SHARE = 0.15  # a face with less area than this share of the median of its neighbours is pinched
POLE_HIGH = 7  # an interior vertex of quads with this many edges or more is a pole
POLE_HIGH_TRIS = 9  # the same where triangles meet (six is normal there)
POLE_LOW = 2  # an interior vertex with this many edges or fewer
JOIN_EDGES = 1.5  # flagged faces closer than this many average edge lengths belong to one spot
MIN_RADIUS_EDGES = 1.5  # a spot is never smaller than this many average edge lengths, so a re-run's spot still matches it
SERIOUS = 3.0  # kinds weighing this much or more are defects; lighter ones are untidiness
WEIGHTS = {  # how bad one kind is; a spot's score adds weight * (1 + log2(faces of that kind)) over its kinds
    "inward": 10.0, "crossing": 10.0, "folded_over": 10.0,
    "sunk": 5.0, "overlap": 5.0, "duplicate": 4.0,
    "bent": 3.0, "flat": 3.0,
    "concave": 2.0, "sliver": 1.0, "pinched": 1.0, "pole": 0.5,
}


def _table(faces):
    """(padded corners, corner counts): short faces repeat their first corner."""
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    width = int(sizes.max(initial=3))
    flat = np.fromiter((v for f in faces for v in f), dtype=np.int64, count=int(sizes.sum()))
    pad = np.zeros((len(faces), width), np.int64)
    col = np.arange(width)
    live = col[None, :] < sizes[:, None]
    pad[live] = flat
    pad = np.where(live, pad, pad[:, :1])
    return pad, sizes


def _neighbours(n: np.ndarray, width: int):
    col = np.arange(width)[None, :]
    return np.where(col + 1 < n[:, None], col + 1, 0), np.where(col >= 1, col - 1, n[:, None] - 1), col < n[:, None]


def components(count: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected-component label (smallest member) for `count` nodes joined by the pairs a-b."""
    label = np.arange(count)
    if not len(a):
        return label
    while True:
        low = np.minimum(label[a], label[b])
        before = label.copy()
        np.minimum.at(label, a, low)
        np.minimum.at(label, b, low)
        while True:  # jump to the label's own label until it settles
            jumped = label[label]
            if (jumped == label).all():
                break
            label = jumped
        if (label == before).all():
            return label


def topology_flags(verts, faces) -> dict[str, np.ndarray]:
    """{kind: bool per face} for sliver, pinched, pole and concave faces."""
    count = len(faces)
    out = {k: np.zeros(count, bool) for k in TOPOLOGY_KINDS}
    if not count:
        return out
    co = np.asarray(verts, dtype=np.float64)
    pad, n = _table(faces)
    width = pad.shape[1]
    nxt, prv, live = _neighbours(n, width)
    rows = np.arange(count)[:, None]
    p = co[pad]
    to_next = p[rows, nxt] - p
    to_prev = p[rows, prv] - p
    side = np.linalg.norm(to_next, axis=2)
    cross = np.cross(to_next, to_prev)
    cos = (to_next * to_prev).sum(axis=2) / np.maximum(np.linalg.norm(to_next, axis=2) * np.linalg.norm(to_prev, axis=2), 1e-300)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    normal = 0.5 * np.where(live[:, :, None], np.cross(p, p[rows, nxt]), 0.0).sum(axis=1)
    area = np.linalg.norm(normal, axis=1)
    unit = normal / np.maximum(area, 1e-300)[:, None]
    shortest = np.where(live, side, np.inf).min(axis=1)
    longest = np.where(live, side, 0.0).max(axis=1)
    out["sliver"] = (np.where(live, angle, 180.0).min(axis=1) < SLIVER_ANGLE) | (longest > SLIVER_ASPECT * np.maximum(shortest, 1e-300))
    out["concave"] = ((n[:, None] > 3) & live & ((cross * unit[:, None, :]).sum(axis=2) < 0.0)).any(axis=1)
    # edges: who shares each
    a, b = pad, np.take_along_axis(pad, nxt, axis=1)
    real = live & (a != b)
    face = np.broadcast_to(rows, a.shape)[real]
    lo, hi = np.minimum(a, b)[real], np.maximum(a, b)[real]
    key = lo * (len(co) + 1) + hi
    order = np.argsort(key, kind="stable")
    key, face, lo, hi = key[order], face[order], lo[order], hi[order]
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    users = np.diff(np.r_[start, len(key)])
    two = start[users == 2]
    fa, fb = face[two], face[two + 1]
    # pinched: area against the median of the neighbours across edges
    around = np.full((count, width), np.inf)
    fill = np.zeros(count, np.int64)
    x, y = np.r_[fa, fb], np.r_[fb, fa]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if len(x):
        first = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
        slot = np.arange(len(x)) - np.repeat(first, np.diff(np.r_[first, len(x)]))
        keep = slot < width
        around[x[keep], slot[keep]] = area[y[keep]]
        np.add.at(fill, x[keep], 1)
    around.sort(axis=1)
    has = fill > 0
    mid = np.where(has, around[rows[:, 0], np.maximum(fill - 1, 0) // 2] + around[rows[:, 0], fill // 2], np.inf) / 2.0
    out["pinched"] = has & (area < PINCHED_SHARE * mid)
    # poles: interior vertices (every edge has two faces) with unusual valence
    valence = np.bincount(np.r_[lo[start], hi[start]], minlength=len(co))
    edge_users = np.repeat(users, users)  # users of the edge of each sorted corner
    open_edge = edge_users != 2
    on_open = np.zeros(len(co), bool)
    on_open[lo[open_edge]] = on_open[hi[open_edge]] = True
    has_tri = np.zeros(len(co), bool)
    has_tri[pad[(n == 3)].ravel()] = True
    pole = ~on_open & ((valence <= POLE_LOW) & (valence > 0) | (valence >= np.where(has_tri, POLE_HIGH_TRIS, POLE_HIGH)))
    out["pole"] = (pole[pad] & live).any(axis=1)
    return out


def find_spots(verts, faces, flags: dict[str, np.ndarray], part_of_face=None, max_spots: int = 200) -> list[dict]:
    """Group flagged faces into spots, worst first. See the module text for what a spot holds."""
    count = len(faces)
    flags = {k: np.asarray(v, bool) for k, v in flags.items() if k in WEIGHTS and len(v) == count}
    hit = np.zeros(count, bool)
    for v in flags.values():
        hit |= v
    ids = np.flatnonzero(hit)
    if not len(ids):
        return []
    co = np.asarray(verts, dtype=np.float64)
    pad, n = _table(faces)
    live = np.arange(pad.shape[1])[None, :] < n[:, None]
    middle = (co[pad] * live[:, :, None]).sum(axis=1) / n[:, None]
    length = float(np.linalg.norm(co[pad] - co[np.roll(pad, -1, axis=1)], axis=2)[live].mean()) or 1.0
    # one node per flagged face; joined by a shared vertex, or by lying in touching voxels
    sub = pad[ids]
    sub_live = live[ids]
    corner = np.where(sub_live, sub, -1).ravel()
    owner = np.repeat(np.arange(len(ids)), pad.shape[1])
    keep = corner >= 0
    corner, owner = corner[keep], owner[keep]
    order = np.argsort(corner, kind="stable")
    corner, owner = corner[order], owner[order]
    same = corner[1:] == corner[:-1]
    ja, jb = [owner[:-1][same]], [owner[1:][same]]
    cell = JOIN_EDGES * length
    vox = np.floor(middle[ids] / cell).astype(np.int64)
    vox -= vox.min(axis=0)
    span = vox.max(axis=0) + 3
    vkey = ((vox[:, 0] + 1) * span[1] + vox[:, 1] + 1) * span[2] + vox[:, 2] + 1
    order = np.argsort(vkey, kind="stable")
    sk = vkey[order]
    first = np.r_[True, sk[1:] != sk[:-1]]
    ja.append(order[:-1][~first[1:]])
    jb.append(order[1:][~first[1:]])
    rep, rkey = order[first], sk[first]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                want = rkey + (dx * span[1] + dy) * span[2] + dz
                at = np.minimum(np.searchsorted(rkey, want), len(rkey) - 1)
                found = rkey[at] == want
                ja.append(rep[found])
                jb.append(rep[at[found]])
    # a face with a real defect never joins one that is only untidy, so widespread slivers don't swallow the defects
    serious = np.zeros(count, bool)
    for k, v in flags.items():
        if WEIGHTS[k] >= SERIOUS:
            serious |= v
    ja, jb = np.concatenate(ja), np.concatenate(jb)
    same_class = serious[ids][ja] == serious[ids][jb]
    label = components(len(ids), ja[same_class], jb[same_class])
    groups = np.unique(label, return_inverse=True)[1]
    parts = None if part_of_face is None else np.asarray(part_of_face)
    spots = []
    for g in range(int(groups.max()) + 1):
        members = ids[groups == g]
        kinds = {k: int(v[members].sum()) for k, v in flags.items() if v[members].any()}
        score = sum(WEIGHTS[k] * (1.0 + math.log2(c)) for k, c in kinds.items())
        used = co[pad[members][live[members]]]
        centre = middle[members].mean(axis=0)
        radius = max(float(np.linalg.norm(used - centre, axis=1).max()), MIN_RADIUS_EDGES * length)
        part = -1
        if parts is not None:
            vals, cnt = np.unique(parts[members], return_counts=True)
            part = int(vals[cnt.argmax()])
        spots.append({"kinds": kinds, "faces": members.tolist(), "center": [round(float(x), 6) for x in centre], "radius": round(radius, 6), "part": part, "score": round(score, 3)})
    spots.sort(key=lambda s: -s["score"])
    spots = spots[:max_spots]
    for i, s in enumerate(spots):
        s["id"] = i
    return spots


def report(spots: list[dict], meta: dict | None = None) -> dict:
    """A JSON-able record of the spots (face numbers dropped but their count kept) and the totals per kind."""
    totals: dict[str, int] = {}
    slim = []
    for s in spots:
        for k, c in s["kinds"].items():
            totals[k] = totals.get(k, 0) + c
        slim.append({**{k: v for k, v in s.items() if k != "faces"}, "face_count": len(s["faces"])})
    return {"version": 1, "meta": dict(meta or {}), "totals": totals, "spots": slim}


def compare(old: dict, new: dict) -> dict:
    """Match spots by position. Returns {"fixed": old spots with nothing left there, "still": new spots that were already
    there, "new": new spots that were not, "totals": {kind: {"old", "new"}}}. One old spot matches at most one new spot."""
    olds, news = old.get("spots", []), new.get("spots", [])
    taken: set[int] = set()
    still, fresh = [], []
    for s in news:
        best, near = None, None
        for j, o in enumerate(olds):
            if j in taken:
                continue
            d = math.dist(s["center"], o["center"])
            if d <= max(s["radius"], o["radius"]) and (near is None or d < near):
                best, near = j, d
        if best is None:
            fresh.append(s)
        else:
            taken.add(best)
            still.append(s)
    fixed = [o for j, o in enumerate(olds) if j not in taken]
    kinds = set(old.get("totals", {})) | set(new.get("totals", {}))
    totals = {k: {"old": old.get("totals", {}).get(k, 0), "new": new.get("totals", {}).get(k, 0)} for k in sorted(kinds)}
    return {"fixed": fixed, "still": still, "new": fresh, "totals": totals}


def analyse(verts, faces, defect_flags: dict[str, np.ndarray] | None = None, part_of_face=None, meta: dict | None = None, max_spots: int = 200):
    """(spots, report) for a result: the given defect masks plus the topology kinds."""
    flags = dict(defect_flags or {})
    flags.update(topology_flags(verts, faces))
    spots = find_spots(verts, faces, flags, part_of_face, max_spots)
    return spots, report(spots, dict(meta or {}, faces=len(faces)))
