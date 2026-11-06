# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Measure quad topology the way an artist would look at it, against numbers taken from hand-made meshes.

`measure` needs only the mesh. `score` compares those numbers with `patterns.json`, which holds statistics from
licensed hand-made meshes and no geometry (sources in docs/PATTERN_SOURCES.md). Plain NumPy.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .patch import _labels

CLASSES = ("flat", "cylinder", "dome", "saddle", "sharp", "small")
SHARP_DEG = 40.0
FLAT_TURN = 0.1  # radians turned across one quad: below this a quad counts as flat
CYLINDER_RATIO = 0.3  # the weaker curvature is below this share of the stronger one
SMALL_PART = 200  # quads: a loose piece this small is a bolt or strap, built with its own rules
VALENCE_TOP = 8  # valences at or above this share a bin


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


class Topology:
    """The quads of a mesh with what every measure below needs. Triangles and n-gons are only counted."""

    def __init__(self, vertices, faces):
        self.points = np.asarray(vertices, dtype=np.float64)
        self.quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
        self.tris = sum(1 for f in faces if len(f) == 3)
        self.ngons = len(faces) - len(self.quads) - self.tris
        q, n = self.quads, len(self.points)
        self.a, self.b = q.reshape(-1), np.roll(q, -1, axis=1).reshape(-1)  # half-edges, 4 per quad
        low, high = np.minimum(self.a, self.b), np.maximum(self.a, self.b)
        self.edges, self.edge_of, counts = np.unique(low * n + high, return_inverse=True, return_counts=True)
        self.edge_of = self.edge_of.reshape(-1)
        self.edge_uses = counts
        self.edge_a, self.edge_b = self.edges // n, self.edges % n
        self.non_manifold_edges = int((counts > 2).sum())
        self.border_edges = counts == 1
        self.faces_at = np.bincount(q.reshape(-1), minlength=n)
        self.valence = np.bincount(self.edge_a, minlength=n) + np.bincount(self.edge_b, minlength=n)
        self.border = np.zeros(n, dtype=bool)
        self.border[self.edge_a[self.border_edges]] = True
        self.border[self.edge_b[self.border_edges]] = True
        # What a vertex looks like with its border filled in by more of the same: a straight border is regular.
        self.code = np.minimum(np.where(self.border, np.where(self.faces_at <= 2, 4, self.faces_at + 2), self.valence), VALENCE_TOP)
        self.used = self.faces_at > 0
        self.poles = self.used & (self.code != 4)
        self._frames()

    def _frames(self) -> None:
        p, q = self.points, self.quads
        diagonal = np.cross(p[q[:, 2]] - p[q[:, 0]], p[q[:, 3]] - p[q[:, 1]])
        self.area = 0.5 * np.linalg.norm(diagonal, axis=1)
        self.normal = _unit(diagonal)
        self.size = np.sqrt(np.maximum(self.area, 1e-30))
        vn = np.zeros_like(p)
        np.add.at(vn, q.reshape(-1), np.repeat(diagonal, 4, axis=0))
        self.vertex_normal = _unit(vn)

    def shape(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per quad: (signed turn across one quad along each principal direction, sorted; flow error in degrees)."""

        p, q, vn, n = self.points, self.quads, self.vertex_normal, self.normal
        e1 = 0.5 * (p[q[:, 1]] + p[q[:, 2]] - p[q[:, 0]] - p[q[:, 3]])
        e2 = 0.5 * (p[q[:, 2]] + p[q[:, 3]] - p[q[:, 0]] - p[q[:, 1]])
        d1 = 0.5 * (vn[q[:, 1]] + vn[q[:, 2]] - vn[q[:, 0]] - vn[q[:, 3]])
        d2 = 0.5 * (vn[q[:, 2]] + vn[q[:, 3]] - vn[q[:, 0]] - vn[q[:, 1]])
        t1 = _unit(e1 - (e1 * n).sum(axis=1, keepdims=True) * n)
        t2 = np.cross(n, t1)
        # S maps a tangent step to the change in normal: S E = D, in the frame (t1, t2).
        a, b = (e1 * t1).sum(axis=1), (e2 * t1).sum(axis=1)
        c, d = (e1 * t2).sum(axis=1), (e2 * t2).sum(axis=1)
        det = a * d - b * c
        safe = np.where(np.abs(det) > 1e-30, det, np.inf)
        g, h = (d1 * t1).sum(axis=1), (d2 * t1).sum(axis=1)
        k, m = (d1 * t2).sum(axis=1), (d2 * t2).sum(axis=1)
        s11, s12 = (g * d - h * c) / safe, (h * a - g * b) / safe
        s21, s22 = (k * d - m * c) / safe, (m * a - k * b) / safe
        off = 0.5 * (s12 + s21)
        mean, half = 0.5 * (s11 + s22), np.hypot(0.5 * (s11 - s22), off)
        k1, k2 = (mean + half) * self.size, (mean - half) * self.size
        # Angle from t1 (an edge) to the nearest principal direction, folded into 0..45.
        angle = 0.5 * np.arctan2(2.0 * off, s11 - s22)
        error = np.degrees(np.abs(np.arctan2(np.sin(2.0 * angle), np.cos(2.0 * angle)))) / 2.0
        return k1, k2, np.minimum(error, 90.0 - error)

    def dihedral(self) -> np.ndarray:
        """Degrees between the two faces at each edge; 0 where an edge has no second face."""

        order = np.argsort(self.edge_of, kind="stable")
        ids = self.edge_of[order]
        same = np.flatnonzero((ids[:-1] == ids[1:]) & (self.edge_uses[ids[:-1]] == 2))
        out = np.zeros(len(self.edges))
        quad = order // 4
        cosine = (self.normal[quad[same]] * self.normal[quad[same + 1]]).sum(axis=1)
        out[ids[same]] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return out

    def corner_angles(self) -> np.ndarray:
        p, q = self.points, self.quads
        here = p[q]
        out = _unit(np.roll(here, -1, axis=1) - here)
        back = _unit(np.roll(here, 1, axis=1) - here)
        return np.degrees(np.arccos(np.clip((out * back).sum(axis=2), -1.0, 1.0)))

    def classes(self, k1, k2, dihedral) -> np.ndarray:
        """Index into CLASSES per quad: small part first, then a sharp edge, then the curvature."""

        strong = np.maximum(np.abs(k1), np.abs(k2))
        weak = np.minimum(np.abs(k1), np.abs(k2))
        out = np.full(len(self.quads), CLASSES.index("dome"))
        out[(k1 * k2 < 0.0) & (weak > FLAT_TURN * 0.5)] = CLASSES.index("saddle")
        out[weak < CYLINDER_RATIO * strong] = CLASSES.index("cylinder")
        out[strong < FLAT_TURN] = CLASSES.index("flat")
        sharp_edge = dihedral >= SHARP_DEG
        out[sharp_edge[self.edge_of.reshape(-1, 4)].any(axis=1)] = CLASSES.index("sharp")
        fa, fb = self._adjacency()
        labels = _labels(len(self.quads), fa, fb)
        out[np.bincount(labels)[labels] <= SMALL_PART] = CLASSES.index("small")
        self.part = labels
        return out

    def _adjacency(self) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.edge_of, kind="stable")
        ids = self.edge_of[order]
        same = np.flatnonzero(ids[:-1] == ids[1:])
        return order[same] // 4, order[same + 1] // 4

    def loops(self) -> np.ndarray:
        """Edges in the loop each edge belongs to. A loop runs straight through regular vertices."""

        labels = self.loop_labels()
        return np.bincount(labels)[labels]

    def loop_labels(self) -> np.ndarray:
        """Per edge, a number for the loop it belongs to (the same for every edge of one loop)."""

        key = self.a * len(self.points) + self.b
        order = np.argsort(key)
        keys = key[order]
        following = np.arange(len(key))
        following[3::4] -= 4
        following += 1  # next half-edge around the same quad
        twin_key = self.b * len(self.points) + self.a
        at = np.searchsorted(keys, twin_key)
        at[at >= len(keys)] = 0
        twin = np.where(keys[at] == twin_key, order[at], -1)
        step = twin[following]  # h -> the other side of the edge after h
        has = (step >= 0) & (self.code[self.b] == 4) & ~self.border[self.b]
        joined = np.flatnonzero(has)
        after = following[step[joined]]
        first, second = self.edge_of[joined], self.edge_of[after]
        # A straight border is a loop too: its two edges meet at a vertex with two faces.
        rim = np.flatnonzero(self.border_edges)
        ends = np.concatenate((self.edge_a[rim], self.edge_b[rim]))
        ids = np.concatenate((rim, rim))
        order = np.argsort(ends, kind="stable")
        ends, ids = ends[order], ids[order]
        meet = np.flatnonzero((ends[:-1] == ends[1:]) & (self.faces_at[ends[:-1]] == 2))
        return _labels(len(self.edges), np.concatenate((first, ids[meet])), np.concatenate((second, ids[meet + 1])))

    def quad_codes(self) -> np.ndarray:
        """The four corner codes per quad as one number, turned and mirrored into one spelling."""

        codes = self.code[self.quads]
        spin = np.stack([np.roll(codes, s, axis=1) for s in range(4)] + [np.roll(codes[:, ::-1], s, axis=1) for s in range(4)], axis=1)
        keys = (spin * np.array([1000, 100, 10, 1])).sum(axis=2).min(axis=1)
        return keys


def _shares(values: np.ndarray, top: int) -> list[float]:
    counts = np.bincount(np.minimum(values, top), minlength=top + 1)[1:]
    return (counts / max(int(counts.sum()), 1)).round(5).tolist()


def _class_row(topo: Topology, chosen, angles, aspect, flow, anisotropy, short, neighbour_ratio, codes) -> dict:
    quads = topo.quads[chosen]
    ids = np.unique(quads)
    interior = ids[~topo.border[ids]]
    weight = anisotropy[chosen]
    pole = topo.poles[ids]
    return {
        "quads": int(chosen.sum()),
        "valence": _shares(topo.code[ids], VALENCE_TOP),
        "poles": round(float(pole.mean()), 4) if len(ids) else 0.0,
        "interior_poles": round(float(topo.poles[interior].mean()), 4) if len(interior) else 0.0,
        "skew": round(float(np.abs(angles[chosen] - 90.0).mean()), 2),
        "slanted": round(float((np.abs(angles[chosen] - 90.0) > 30.0).any(axis=1).mean()), 4),
        "aspect": round(float(np.median(aspect[chosen])), 3),
        "flow": round(float((flow[chosen] * weight).sum() / max(float(weight.sum()), 1e-30)), 2),
        "short_loops": round(float(short[chosen].mean()), 4),
        "size_jump": round(float(np.median(neighbour_ratio[chosen])), 3),
        "codes": dict(zip(*[x.tolist() for x in np.unique(codes[chosen], return_counts=True)])),
    }


def measure(vertices, faces) -> dict:
    """Numbers for one mesh, overall (`all`) and per local shape class."""

    topo = Topology(vertices, faces)
    if not len(topo.quads):
        return {"quads": 0, "tris": topo.tris, "ngons": topo.ngons, "classes": {}}
    k1, k2, flow = topo.shape()
    dihedral = topo.dihedral()
    labels = topo.classes(k1, k2, dihedral)
    angles = topo.corner_angles()
    p, q = topo.points, topo.quads
    lengths = np.linalg.norm(p[np.roll(q, -1, axis=1)] - p[q], axis=2)
    across, along = lengths[:, 0] + lengths[:, 2], lengths[:, 1] + lengths[:, 3]
    aspect = np.maximum(across, along) / np.maximum(np.minimum(across, along), 1e-30)
    anisotropy = np.abs(k1 - k2)
    anisotropy = np.where(anisotropy > 0.05, anisotropy, 0.0)
    loop_size = topo.loops()
    per_quad_edges = topo.edge_of.reshape(-1, 4)
    short = (loop_size[per_quad_edges] <= 3).mean(axis=1)
    fa, fb = topo._adjacency()
    jump = np.abs(np.log(np.maximum(topo.area[fa], 1e-30) / np.maximum(topo.area[fb], 1e-30)))
    total = np.bincount(fa, jump, len(q)) + np.bincount(fb, jump, len(q))
    uses = np.bincount(fa, minlength=len(q)) + np.bincount(fb, minlength=len(q))
    neighbour_ratio = total / np.maximum(uses, 1)
    codes = topo.quad_codes()
    args = (topo, angles, aspect, flow, anisotropy, short, neighbour_ratio, codes)
    row = _class_row(args[0], np.ones(len(q), dtype=bool), *args[1:])
    classes = {name: _class_row(args[0], labels == i, *args[1:]) for i, name in enumerate(CLASSES) if (labels == i).any()}
    # Whether quads shrink where the surface turns. Only curved quads: flat ones say nothing either way.
    curved = np.maximum(np.abs(k1), np.abs(k2)) / topo.size
    mask = (curved > 0.0) & (labels != CLASSES.index("flat")) & (topo.area > 0.0)
    slope = 0.0
    if int(mask.sum()) > 16 and float(np.ptp(np.log(curved[mask]))) > 0.0:
        slope = float(np.polyfit(np.log(curved[mask]), np.log(topo.size[mask]), 1)[0])
    used = topo.used
    row.update(
        quads=len(q), tris=topo.tris, ngons=topo.ngons, parts=len(np.unique(topo.part)), non_manifold_edges=topo.non_manifold_edges,
        border_vertices=round(float(topo.border[used].mean()), 4), density_slope=round(slope, 3),
        pole_near1=_pole_near(topo, 1), pole_near2=_pole_near(topo, 2),
        loop_median=int(np.median(loop_size)),
        irregular_share_of_vertices=round(float((topo.faces_at[used] != 4).mean()), 3),
    )
    row["classes"] = classes
    return row


def _pole_near(topo: Topology, reach: int) -> float:
    """Share of poles with another pole within this many edges. Scattered meshes have most of them in clusters."""

    poles = np.flatnonzero(topo.poles)
    if not len(poles):
        return 0.0
    a, b = topo.edge_a, topo.edge_b
    hits = np.zeros(len(topo.points), dtype=np.int64)
    hits[poles] = 1
    # Number of walks reaching each vertex minus the pole's own return walks is enough for reach 1 and 2.
    near = np.zeros(len(topo.points), dtype=bool)
    frontier = hits.astype(bool)
    start_a, start_b = frontier[a], frontier[b]
    seen_near = np.bincount(a[start_b], minlength=len(near)) + np.bincount(b[start_a], minlength=len(near))
    near |= seen_near > 0
    if reach >= 2:
        # A vertex touching two different poles links them, or a pole touches a vertex touching another pole.
        link = np.flatnonzero(seen_near >= 2)
        is_link = np.zeros(len(near), dtype=bool)
        is_link[link] = True
        near[a[is_link[b] & frontier[a]]] = True
        near[b[is_link[a] & frontier[b]]] = True
    return round(float(near[poles].mean()), 4)


# Lower is better for each, weighted by how much artists notice it. Medians come from the references.
SCORED = {"interior_poles": 3.0, "slanted": 2.0, "skew": 1.0, "short_loops": 3.0, "flow": 1.0, "size_jump": 1.0}
# Values no hand-made mesh comes near: where a feature's desirability reaches zero.
WORST = {"interior_poles": 0.6, "slanted": 1.0, "skew": 45.0, "short_loops": 0.9, "flow": 30.0, "size_jump": 1.0}
LIKELIHOOD_WEIGHT = 3.0
LIKELIHOOD_WORST = 6.0  # nats per quad worse than the references


def load_database(path: str | None = None) -> dict:
    with open(path or os.path.join(os.path.dirname(__file__), "patterns.json"), encoding="utf-8") as stream:
        return json.load(stream)


def corner_costs(database: dict | None = None, use: str | None = None):
    """A `corner_cost` for tidy and refill: how much rarer a valence is than 4 in the references, in nats."""

    database = database or load_database()
    known = database["uses"][use] if use else database["classes"]
    shares = np.zeros(VALENCE_TOP)
    for row in known.values():
        shares += row["quads"] * np.asarray(row["valence"])
    shares = (shares + 1.0) / float(shares.sum() + VALENCE_TOP)
    table = np.maximum(np.log(shares[3] / shares), 0.0).tolist()  # shares[k] is valence k + 1

    def cost(valence: int, border: bool, faces: int) -> float:
        if border:
            valence = 4 if faces <= 2 else faces + 2
        return table[min(max(valence, 1), VALENCE_TOP) - 1]

    return cost


def _desire(value: float, good: float, worst: float) -> float:
    return float(np.clip(1.0 - (value - good) / max(worst - good, 1e-9), 0.0, 1.0))


def code_cost(row: dict, reference: dict) -> float:
    """Nats per quad to spell this class's corner patterns with the references' frequencies, smoothed for unseen ones."""

    table = reference.get("codes", {})
    seen = sum(table.values()) or 1
    bins = max(len(table), 1) + 64
    cost, total = 0.0, 0
    for code, count in row["codes"].items():
        share = (table.get(str(code), 0) + 1.0) / (seen + bins)
        cost -= count * float(np.log(share))
        total += count
    return cost / max(total, 1)


def score(measured: dict, database: dict | None = None) -> dict:
    """0-100, 100 meaning as tidy as the hand-made references, class by class, plus which parts cost the most."""

    database = database or load_database()
    known = database["classes"]
    parts: dict[str, float] = {}
    likely = 0.0
    total = max(sum(r["quads"] for r in measured["classes"].values()), 1)
    for name, row in measured["classes"].items():
        reference = known.get(name) or known["flat"]
        share = row["quads"] / total
        for feature in SCORED:
            parts[feature] = parts.get(feature, 0.0) + share * _desire(row[feature], reference[feature], WORST[feature])
        likely += share * _desire(code_cost(row, reference) - reference["code_cost"], 0.0, LIKELIHOOD_WORST)
    weight = sum(SCORED.values()) + LIKELIHOOD_WEIGHT
    value = (sum(SCORED[k] * v for k, v in parts.items()) + LIKELIHOOD_WEIGHT * likely) / weight
    parts = {k: round(v, 3) for k, v in parts.items()}
    parts["patterns"] = round(likely, 3)
    return {"score": round(100.0 * value, 1), "parts": parts}


__all__ = ["CLASSES", "Topology", "code_cost", "corner_costs", "load_database", "measure", "score"]
