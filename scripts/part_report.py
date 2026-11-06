# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Look at every loose part of a retopo result, compared with the source, and write down what the camera saw.

blender --background --factory-startup --python scripts/part_report.py -- source.npz result.npz out_folder [worst_count]

Both .npz files hold `co`, and faces as `lens` + `flat` or as `start` + `total` + `loops`. Writes parts.csv, every part
with the worst first (most faces per pixel it shows on the object), and parts.png: for the worst parts, the source on
the left of each pair and the result on the right, in that order. CPU rays only, nothing is rendered.
"""

import csv
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triadq.views import pixels_seen  # noqa: E402

VIEW_PIXELS = 2048  # the whole object's size in the camera's pixels
TILE = 112
BIGGEST_SHARE = 0.25  # a part with over this share of the faces is the main body, not a small part: no tile for it


def load(path):
    d = np.load(path)
    co = d["co"]
    if "lens" in d:
        lens, flat = d["lens"], d["flat"]
        offsets = np.concatenate(([0], np.cumsum(lens)))
    else:
        lens, flat, offsets = d["total"], d["loops"], d["start"]
    return co, [flat[a : a + b].tolist() for a, b in zip(offsets.tolist(), lens.tolist())]


def parts_of(count, faces):
    """Label of each face's connected part (faces sharing a vertex count as connected)."""
    parent = list(range(count))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for face in faces:
        first = root(face[0])
        for v in face[1:]:
            parent[root(v)] = first
    labels = np.asarray([root(face[0]) for face in faces], dtype=np.int64)
    return np.unique(labels, return_inverse=True)[1]


def tile(points, polys, low, high):
    """One grey view of a part from a three-quarter direction, face edges drawn dark, with the same framing for any mesh."""
    out = np.full((TILE, TILE), 0.18)
    if not polys:
        return out
    tree = BVHTree.FromPolygons(points.tolist(), polys, all_triangles=False, epsilon=0.0)
    middle = (low + high) / 2.0
    half = max(float(np.linalg.norm(high - low)) * 0.55, 1e-9)
    view = np.array([0.55, -0.7, 0.45])
    view /= np.linalg.norm(view)
    right = np.cross(view, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    light = (view + up * 0.5 + right * 0.3) / np.linalg.norm(view + up * 0.5 + right * 0.3)
    ids = np.full((TILE, TILE), -1, dtype=np.int64)
    direction = Vector((-view).tolist())
    for i in range(TILE):
        for j in range(TILE):
            x, y = ((j + 0.5) / TILE * 2 - 1) * half, (1 - (i + 0.5) / TILE * 2) * half
            found = tree.ray_cast(Vector((middle + right * x + up * y + view * 4 * half).tolist()), direction)
            if found[0] is not None:
                ids[i, j] = found[2]
                out[i, j] = 0.3 + 0.65 * abs(float(np.dot(found[1], light)))
    edge = np.zeros((TILE, TILE), dtype=bool)
    edge[1:] |= (ids[1:] != ids[:-1]) & (ids[1:] >= 0) & (ids[:-1] >= 0)
    edge[:, 1:] |= (ids[:, 1:] != ids[:, :-1]) & (ids[:, 1:] >= 0) & (ids[:, :-1] >= 0)
    out[edge] *= 0.45
    return out


def sub_mesh(co, faces, chosen):
    used = np.unique(np.fromiter((v for f in chosen for v in faces[f]), dtype=np.int64))
    table = {int(v): k for k, v in enumerate(used.tolist())}
    return co[used], [[table[v] for v in faces[f]] for f in chosen]


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    source_co, source_faces = load(args[0])
    result_co, result_faces = load(args[1])
    out = os.path.abspath(args[2])
    worst_count = int(args[3]) if len(args) > 3 else 50
    os.makedirs(out, exist_ok=True)
    pixel = float(np.ptp(source_co, axis=0).max()) / VIEW_PIXELS
    reach = 4.0 * float(np.linalg.norm(np.ptp(source_co, axis=0)))
    source_labels = parts_of(len(source_co), source_faces)
    result_labels = parts_of(len(result_co), result_faces)
    source_tree = BVHTree.FromPolygons(source_co.tolist(), source_faces, all_triangles=False, epsilon=0.0)
    result_tree = BVHTree.FromPolygons(result_co.tolist(), result_faces, all_triangles=False, epsilon=0.0)
    source_cast = lambda origin, direction: source_tree.ray_cast(Vector(origin), Vector(direction))[2]  # noqa: E731
    result_cast = lambda origin, direction: result_tree.ray_cast(Vector(origin), Vector(direction))[2]  # noqa: E731
    source_list, result_list = source_labels.tolist(), result_labels.tolist()
    members = [np.flatnonzero(source_labels == k).tolist() for k in range(int(source_labels.max()) + 1)]
    result_members = [np.flatnonzero(result_labels == k).tolist() for k in range(int(result_labels.max()) + 1)]
    # Which result parts stand for each source part: by whichever source face each part's first corner is closest to.
    matched = [[] for _ in members]
    for k, chosen in enumerate(result_members):
        found = source_tree.find_nearest(Vector(result_co[result_faces[chosen[0]][0]].tolist()))
        matched[source_labels[found[2]]].append(k)
    rows = []
    for k, chosen in enumerate(members):
        points, polys = sub_mesh(source_co, source_faces, chosen)
        low, high = points.min(axis=0), points.max(axis=0)
        big = len(chosen) > BIGGEST_SHARE * len(source_faces)
        seen = -1 if big else pixels_seen(source_cast, source_list, low, high, pixel, k, reach)
        out_faces = sum(len(result_members[m]) for m in matched[k])
        seen_after = -1
        if not big:
            seen_after = sum(pixels_seen(result_cast, result_list, low, high, pixel, m, reach) for m in matched[k])
        rows.append({
            "part": k, "source_faces": len(chosen), "result_faces": out_faces, "size_pixels": round(float((high - low).max()) / pixel, 1),
            "pixels_seen": seen, "pixels_seen_after": seen_after,
            "change": round(abs(seen - seen_after) / max(seen, 1), 3) if not big else "",
            "faces_per_100_pixels": round(100.0 * out_faces / max(seen, 1), 1) if not big else 0.0,
        })
    rows.sort(key=lambda r: -r["faces_per_100_pixels"])
    with open(os.path.join(out, "parts.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    worst = [r for r in rows if r["pixels_seen"] >= 0][:worst_count]
    across = 5
    down = math.ceil(len(worst) / across)
    gap = 6
    sheet = np.full((down * (TILE + gap), across * (2 * TILE + 3 * gap), 3), 0.08)
    for n, row in enumerate(worst):
        points, polys = sub_mesh(source_co, source_faces, members[row["part"]])
        low, high = points.min(axis=0), points.max(axis=0)
        chosen = [f for m in matched[row["part"]] for f in result_members[m]]
        after = tile(*sub_mesh(result_co, result_faces, chosen), low, high) if chosen else np.full((TILE, TILE), 0.18)
        y, x = (n // across) * (TILE + gap), (n % across) * (2 * TILE + 3 * gap)
        sheet[y : y + TILE, x : x + TILE] = tile(points, polys, low, high)[..., None]
        sheet[y : y + TILE, x + TILE + 1 : x + 2 * TILE + 1] = after[..., None]
    height, width = sheet.shape[:2]
    image = bpy.data.images.new("Parts", width, height, alpha=False)
    pixels = np.ones((height, width, 4), dtype=np.float32)
    pixels[..., :3] = sheet[::-1]
    image.pixels.foreach_set(pixels.ravel())
    image.filepath_raw = os.path.join(out, "parts.png")
    image.file_format = "PNG"
    image.save()
    print("PARTS", len(rows), "worst", [(r["part"], r["faces_per_100_pixels"]) for r in worst[:5]], flush=True)


main()
