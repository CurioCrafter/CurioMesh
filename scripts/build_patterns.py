#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rebuild triadq/patterns.json from the reference meshes listed in a references.json.

Only entries named in TRUSTED and allowed in derived form are read, and only statistics are written, never geometry.
    python scripts/build_patterns.py <folder with references.json>
    python scripts/build_patterns.py <folder> --cc0     # CC0 / public domain sources only -> triadq/patterns_cc0.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from triadq import patterns  # noqa: E402

MIN_QUADS = 16
MOST_CODES = 64  # kept per class in the data file
MESH_WEIGHT = 2000  # quads a mesh may count for, so one big body doesn't outvote many small parts
FEATURES = ("valence", "poles", "interior_poles", "skew", "slanted", "aspect", "flow", "short_loops", "size_jump")
# Proven hand-made work only. Poly Haven props stay out: nobody knows if a remesher made them.
TRUSTED = {
    "Blender Studio Human Base Meshes v1.4.1": {"use": "character", "files": [], "min_quads": 0.0},
    "MakeHuman base mesh (base.obj)": {"use": "character", "files": ["base.obj"], "min_quads": 0.0},
    "Keenan Crane - Spot": {"use": "character", "files": ["spot_control_mesh.obj"], "min_quads": 0.0},
    "Keenan Crane - Bob": {"use": "character", "files": ["bob_controlmesh.obj"], "min_quads": 0.0},
    "Keenan Crane - Blub": {"use": "character", "files": ["blub_control_mesh.obj"], "min_quads": 0.0},
    "Keenan Crane - Jerry the Ogre (control mesh only)": {"use": "character", "files": ["controlmesh.obj"], "min_quads": 0.0},
    "Blender Studio Sprite (Sprite Fright)": {"use": "character", "files": [], "min_quads": 0.0},
    "Blender Studio Rain": {"use": "character", "files": [], "min_quads": 0.0},
    "Blender Studio Pack Bot (Charge)": {"use": "hard_surface", "files": [], "min_quads": 0.0},
    "Blender Studio Security Bot (Charge)": {"use": "hard_surface", "files": [], "min_quads": 0.95},  # its clean parts only
}


def load_obj(path: str):
    points, faces = [], []
    skip = False
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                points.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("g "):
                words = line.split()
                skip = len(words) > 1 and words[1].startswith(("helper-", "joint-"))  # rigging boxes in the MakeHuman file
            elif line.startswith("f ") and not skip:
                ids = [int(token.split("/")[0]) for token in line.split()[1:]]
                faces.append(tuple(i - 1 if i > 0 else len(points) + i for i in ids))
    co = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    unique, inverse = np.unique(np.round(co, 7), axis=0, return_inverse=True)  # OBJ splits seams: weld so valence is real
    inverse = inverse.reshape(-1).tolist()
    return unique, [tuple(inverse[v] for v in f) for f in faces]


def mesh_files(item: dict) -> list[str]:
    """Control meshes only for the Crane models: the others in those folders are subdivided or remeshed."""

    plan = TRUSTED[item["name"]]
    if plan["files"]:
        return [os.path.join(item["folder"], name) for name in plan["files"]]
    return sorted(glob.glob(os.path.join(item["folder"], "meshes", "*.obj")))


def is_cc0(item: dict) -> bool:
    return str(item.get("license", "")).startswith(("CC0", "Public domain"))


def measure_all(folder: str, cc0_only: bool = False) -> tuple[dict, list]:
    with open(os.path.join(folder, "references.json"), encoding="utf-8") as stream:
        book = json.load(stream)
    rows, used = {}, []
    for item in book["items"]:
        if item["name"] not in TRUSTED or not str(item.get("redistribution_derived", "")).lower().startswith("yes"):
            continue
        if cc0_only and not is_cc0(item):
            continue
        plan = TRUSTED[item["name"]]
        seen = set()
        count = 0
        for path in mesh_files(item):
            co, faces = load_obj(path)
            row = patterns.measure(co, faces)
            if row.get("quads", 0) < MIN_QUADS or row["quads"] < plan["min_quads"] * len(faces):
                continue
            # The same mesh under two names, or its mirror image, says nothing new.
            twin = (row["quads"], row["interior_poles"], row["short_loops"], row["loop_median"])
            if twin in seen:
                continue
            seen.add(twin)
            row["source"], row["use"] = item["name"], plan["use"]
            rows[item["name"] + "/" + os.path.basename(path)[:-4]] = row
            count += 1
        used.append({k: item.get(k) for k in ("name", "license", "license_url", "attribution", "source_url")} | {"meshes": count, "use": plan["use"]})
    return rows, used


def combine(rows: list[dict]) -> dict:
    """One class: the median over meshes, and pooled corner patterns with big meshes capped."""

    out = {"meshes": len(rows), "quads": int(sum(r["quads"] for r in rows))}
    for feature in FEATURES:
        values = np.asarray([r[feature] for r in rows], dtype=np.float64)
        out[feature] = np.round(np.median(values, axis=0), 4).tolist()
    pooled: dict[str, float] = {}
    for row in rows:
        scale = min(row["quads"], MESH_WEIGHT) / max(row["quads"], 1)
        for code, count in row["codes"].items():
            pooled[str(code)] = pooled.get(str(code), 0.0) + count * scale
    top = sorted(pooled.items(), key=lambda kv: -kv[1])[:MOST_CODES]
    out["codes"] = {code: max(int(round(count)), 1) for code, count in top}
    out["code_cost"] = round(float(np.median([patterns.code_cost(r, out) for r in rows])), 3)
    return out


def _classes(rows: list[dict]) -> dict:
    out = {}
    for name in patterns.CLASSES:
        members = [row["classes"][name] for row in rows if row["classes"].get(name, {}).get("quads", 0) >= 12]
        if members:
            out[name] = combine(members)
    return out


def build(rows: dict, used: list) -> dict:
    uses = {use: _classes([r for r in rows.values() if r["use"] == use]) for use in ("character", "hard_surface")}
    return {"version": 1, "sources": used, "classes": _classes(list(rows.values())), "uses": uses}


if __name__ == "__main__":
    cc0 = "--cc0" in sys.argv
    all_rows, sources = measure_all(sys.argv[1], cc0)
    database = build(all_rows, sources)
    target = os.path.join(ROOT, "triadq", "patterns_cc0.json" if cc0 else "patterns.json")
    with open(target, "w", encoding="utf-8") as stream:
        json.dump(database, stream, indent=0, sort_keys=True)
    print(f"{len(all_rows)} meshes, {os.path.getsize(target)} bytes -> {target}")
