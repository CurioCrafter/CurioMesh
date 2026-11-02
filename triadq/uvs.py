# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Plain NumPy helpers for keeping UVs good: which faces may share a quad, and whether a UV map can hold a bake."""

from __future__ import annotations

import numpy as np

MOST_CHARTS = 30000  # Blender keeps a face's material index in 16 bits while reducing


def chart_codes(materials: np.ndarray, islands: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """One small number per (material, UV island) for each face, and the tables back. None if there are too many."""

    pairs, codes = np.unique(np.column_stack((materials, islands)), axis=0, return_inverse=True)
    if len(pairs) > MOST_CHARTS:
        return None
    return codes.reshape(-1), pairs[:, 0], pairs[:, 1]


def _straighten(points: np.ndarray, chain: list[int], tolerance: float) -> list[int]:
    """Positions in the chain worth keeping: the first, the last, and whatever strays further than tolerance between."""

    keep = {0, len(chain) - 1}
    spans = [(0, len(chain) - 1)]
    while spans:
        a, b = spans.pop()
        if b - a < 2:
            continue
        line = points[chain[b]] - points[chain[a]]
        rel = points[chain[a + 1 : b]] - points[chain[a]]
        off = np.linalg.norm(rel - np.outer(rel @ line / max(float(line @ line), 1e-30), line), axis=1)
        far = int(off.argmax())
        if off[far] > tolerance:
            mid = a + 1 + far
            keep.add(mid)
            spans += [(a, mid), (mid, b)]
    return sorted(keep)


def simplify_borders(points: np.ndarray, polys, kinds, tolerance: float):
    """Straighten the borders between kinds of face, along themselves, so each side still shares the same vertices.

    Returns (points, polys, kinds, free, used): free lists the vertices not on a border, which are the ones allowed to
    move; used lists the input vertices that are left, in their new order.
    """

    first: dict[tuple[int, int], object] = {}
    seen: dict[int, object] = {}
    corner = set()
    links: dict[int, set[int]] = {}
    for poly, kind in zip(polys, kinds):
        for at, v in enumerate(poly):
            w = poly[(at + 1) % len(poly)]
            edge = (v, w) if v < w else (w, v)
            if first.setdefault(edge, kind) != kind:
                links.setdefault(v, set()).add(w)
                links.setdefault(w, set()).add(v)
            if seen.setdefault(v, kind) != kind:
                corner.add(v)
    stops = {v for v in corner if len(links.get(v, ())) != 2} | {v for v, near in links.items() if len(near) != 2}
    target = np.arange(len(points))
    kept = set(stops)
    done: set[int] = set()

    def walk(start: int, step: int) -> None:
        chain = [start, step]
        while chain[-1] not in stops:
            done.add(chain[-1])
            ahead = [n for n in links[chain[-1]] if n != chain[-2]]
            chain.append(ahead[0])
        places = _straighten(points, chain, tolerance)
        kept.update(chain[p] for p in places)
        for a, b in zip(places[:-1], places[1:]):
            middle = (a + b) // 2
            target[chain[a + 1 : middle + 1]] = chain[a]
            target[chain[middle + 1 : b]] = chain[b]

    for start in sorted(stops):
        for step in sorted(links.get(start, ())):
            if step not in done:
                walk(start, step)
    for start in sorted(links):  # closed loops have no ends: start at one vertex and the one furthest from it
        if start in done or start in stops:
            continue
        loop = [start]
        while True:
            ahead = [n for n in links[loop[-1]] if n not in loop[-2:]]
            if not ahead or ahead[0] == start:
                break
            loop.append(ahead[0])
        far = loop[int(np.linalg.norm(points[loop] - points[start], axis=1).argmax())]
        stops |= {start, far}
        kept |= {start, far}
        for step in sorted(links[start]):
            if step not in done:
                walk(start, step)

    out_polys, out_kinds, faces_seen = [], [], set()
    for poly, kind in zip(polys, kinds):
        mapped = [int(target[v]) for v in poly]
        mapped = [v for at, v in enumerate(mapped) if v != mapped[at - 1]]
        key = tuple(sorted(mapped))
        if len(mapped) >= 3 and len(set(mapped)) == len(mapped) and key not in faces_seen:
            faces_seen.add(key)
            out_polys.append(mapped)
            out_kinds.append(kind)
    used, inverse = np.unique(np.fromiter((v for p in out_polys for v in p), dtype=np.int64), return_inverse=True)
    out, at = [], 0
    for p in out_polys:
        out.append(tuple(int(v) for v in inverse[at : at + len(p)]))
        at += len(p)
    locked = np.isin(used, np.fromiter(kept | corner, dtype=np.int64))
    return points[used], out, out_kinds, np.flatnonzero(~locked).tolist(), used


def uv_overlap(uv_tris: np.ndarray, pixels: int = 256) -> float:
    """Share of the used texels that more than one triangle lands on. 1.0 when the map leaves the 0-1 square."""

    if not len(uv_tris):
        return 1.0
    if uv_tris.min() < -1e-4 or uv_tris.max() > 1.0 + 1e-4:
        return 1.0
    hits = np.zeros((pixels, pixels), dtype=np.int32)
    points = uv_tris * pixels - 0.5  # texel centres fall on whole numbers
    low = np.clip(np.ceil(points.min(axis=1)).astype(np.int64), 0, pixels - 1)
    high = np.clip(np.floor(points.max(axis=1)).astype(np.int64), 0, pixels - 1)
    edge1, edge2 = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    det = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    for t in np.flatnonzero(np.abs(det) > 1e-12).tolist():
        x, y = np.meshgrid(np.arange(low[t, 0], high[t, 0] + 1), np.arange(low[t, 1], high[t, 1] + 1), indexing="ij")
        dx, dy = x - points[t, 0, 0], y - points[t, 0, 1]
        a = (dx * edge2[t, 1] - dy * edge2[t, 0]) / det[t]
        b = (edge1[t, 0] * dy - edge1[t, 1] * dx) / det[t]
        inside = (a > 1e-6) & (b > 1e-6) & (a + b < 1.0 - 1e-6)
        hits[x[inside], y[inside]] += 1
    used = int((hits > 0).sum())
    return float((hits > 1).sum()) / used if used else 1.0


def is_see_through(material) -> bool:
    """A material a color bake would ruin: glass nodes, blended transparency, or a Principled shader with transmission or alpha."""

    if material is None:
        return False
    if getattr(material, "blend_method", "OPAQUE") in {"BLEND", "HASHED"} or getattr(material, "surface_render_method", "DITHERED") == "BLENDED":
        return True
    tree = material.node_tree if material.use_nodes else None
    for node in tree.nodes if tree is not None else ():
        if node.type in {"BSDF_GLASS", "BSDF_TRANSPARENT", "BSDF_REFRACTION"}:
            return True
        if node.type == "BSDF_PRINCIPLED":
            for name, opaque in (("Alpha", 1.0), ("Transmission Weight", 0.0), ("Transmission", 0.0)):  # the name changed in Blender 4.0
                plug = node.inputs.get(name)
                if plug is not None and (plug.is_linked or plug.default_value != opaque):
                    return True
    return False
