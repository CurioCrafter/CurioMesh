# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Rebuild triadq/cases_table.json: measure every part of some models at every size, then sum it up per case.

1. Once per model, in Blender (the folder holding this repo is imported as a package, like the add-on itself):
   blender --background --factory-startup --python scripts/build_case_table.py -- measure rows.jsonl model.npz [parts.py]
   The .npz has `co`, `loops`, `start` and `total`, the arrays of a Blender mesh. The optional parts.py is a module with
   `rebuilds(points, polys)`, whose results get measured too, as the method "revolve".
2. Then with plain Python, on the models the table should learn from:
   python scripts/build_case_table.py table triadq/cases_table.json rows1.jsonl rows2.jsonl ...
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time



def _drain(generator):
    try:
        while True:
            next(generator)
    except StopIteration as done:
        return done.value


def _regular(faces) -> float:
    """Share of the vertices with four faces around them."""
    count: dict[int, int] = {}
    for face in faces:
        for vertex in face:
            count[vertex] = count.get(vertex, 0) + 1
    return sum(1 for value in count.values() if value == 4) / max(len(count), 1)


def measure(out_path: str, model_path: str, parts_path: str | None) -> None:
    import importlib
    import importlib.util

    import bpy
    import numpy as np

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(repo))
    package = os.path.basename(repo)
    auto = importlib.import_module(package + ".autoretopo")
    cases = importlib.import_module(package + ".triadq.cases")
    flatquad = importlib.import_module(package + ".triadq.flatquad")
    types = importlib.import_module(package + ".triadq.types")
    patch = importlib.import_module(package + ".triadq.patch")
    rebuilds = None
    if parts_path:
        spec = importlib.util.spec_from_file_location("curiomesh_parts", parts_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rebuilds = module

    data = np.load(model_path)
    loops, starts, totals = data["loops"].tolist(), data["start"].tolist(), data["total"].tolist()
    faces = [tuple(loops[a : a + b]) for a, b in zip(starts, totals)]
    co = data["co"].astype(np.float64)
    auto.USE_CASES = False  # this is what builds the table
    job = auto.AutoJob(bpy.context, co, faces, [0] * len(faces), 6.0)
    fa, fb = patch.face_adjacency(faces)
    shell = np.unique(patch._labels(len(faces), fa, fb), return_inverse=True)[1]
    order = np.argsort(shell, kind="stable")
    cuts = np.concatenate(([0], np.flatnonzero(shell[order][1:] != shell[order][:-1]) + 1, [len(order)]))
    pieces = sorted((order[a:b] for a, b in zip(cuts[:-1], cuts[1:])), key=len, reverse=True)
    extent = float(np.ptp(co, axis=0).max()) or 1.0
    sizes = np.asarray([2.0 * 0.55 * float(np.ptp(co[np.unique(np.concatenate([faces[f] for f in ids.tolist()]))], axis=0).max()) for ids in pieces]) / extent
    pixels_for = auto._picture_sizes(sizes).tolist()
    model = os.path.splitext(os.path.basename(model_path))[0]
    with open(out_path, "w", encoding="utf-8") as out:
        for index, ids in enumerate(pieces):
            points, polys, kinds = job._piece_arrays(ids)
            triangles = sum(len(p) - 2 for p in polys)
            if triangles <= auto.KEEP_TRIANGLES:
                continue
            started = time.perf_counter()
            found = cases.features(points, polys, extent)
            row = {"model": model, "piece": index, "pixels": pixels_for[index], "polys": len(polys), "features": found, "case": cases.classify(found), "feature_s": time.perf_counter() - started, "tries": []}
            low, high = points.min(axis=0), points.max(axis=0)
            center, half = (low + high) / 2.0, max(float((high - low).max()) * 0.55, 1e-12)
            pixel = 2.0 * half / pixels_for[index]
            started = time.perf_counter()
            reference = _drain(auto.pictures(points, polys, center, half, pixels_for[index]))
            row["reference_s"] = time.perf_counter() - started

            def look(method: str, share: float, mesh, reduce_s: float, pair: bool) -> None:
                started = time.perf_counter()
                wrong, shown = auto.picture_counts(reference, _drain(auto.pictures(mesh[0], mesh[1], center, half, pixels_for[index])), pixel)
                try_row = {"method": method, "share": share, "faces": len(mesh[1]), "wrong": wrong, "shown": shown, "reduce_s": reduce_s, "look_s": time.perf_counter() - started}
                if pair and len(mesh[1]) <= 30000:
                    paired = flatquad.pair_and_close(types.MeshData(mesh[0], mesh[1], mesh[2]), seed_count=2)[0]
                    try_row["quads"] = len(paired.faces)
                    try_row["regular"] = _regular(paired.faces)
                elif not pair:
                    try_row["quads"] = len(mesh[1])
                    try_row["regular"] = _regular(mesh[1])
                row["tries"].append(try_row)

            for share in auto.LEVELS:
                if share * triangles < 12.0:
                    continue
                started = time.perf_counter()
                mesh = auto._decimate(bpy.context, points, polys, kinds, share)
                look("decimate", share, mesh, time.perf_counter() - started, True)
            if rebuilds is not None:
                started = time.perf_counter()
                for built in rebuilds.rebuilds(points, polys, "0"):
                    if built is not None:
                        mesh = (built[0], rebuilds.orient_like(built[0], built[1], points, polys), built[2])
                        look("revolve", len(mesh[1]) / triangles, mesh, time.perf_counter() - started, False)
                    started = time.perf_counter()
            out.write(json.dumps(row) + "\n")
            out.flush()


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        return values[0], values[0], values[0]
    low, middle, high = statistics.quantiles(values, n=4)
    return low, middle, high


def summarize(rows: list[dict]) -> dict:
    """Per case and share of faces: the loss range, faces, quads and regularity, and how often a rebuild beat Decimate."""
    groups: dict[str, list[dict]] = {"all": rows}
    for row in rows:
        groups.setdefault(row["case"], []).append(row)
    out = {}
    for case, members in groups.items():
        by_share: dict[float, list[dict]] = {}
        wins = turned = 0
        for row in members:
            kept = [t for t in row["tries"] if t["method"] == "decimate"]
            for t in kept:
                by_share.setdefault(t["share"], []).append(t)
            built = [t for t in row["tries"] if t["method"] == "revolve"]
            if built:
                turned += 1
                wins += _rebuild_wins(built, kept)
        shares = {}
        for share, tries in sorted(by_share.items()):
            paired = [t for t in tries if "quads" in t]
            shares[str(share)] = {
                "parts": len(tries),
                "loss": [round(v, 5) for v in _quantiles([t["wrong"] / max(t["shown"], 1) for t in tries])],
                "faces": round(statistics.median(t["faces"] for t in tries)),
                "quads_per_face": round(statistics.median(t["quads"] / max(t["faces"], 1) for t in paired), 3) if paired else None,
                "regular": round(statistics.median(t["regular"] for t in paired), 3) if paired else None,
            }
        out[case] = {"parts": len(members), "shares": shares, "revolve": {"fits": turned, "wins": wins}}
    return out


def _rebuild_wins(built: list[dict], kept: list[dict], loss: float = 0.06) -> bool:
    """A rebuild wins when it reaches the loss with fewer faces than Decimate (after pairing) does."""
    best = min((t["quads"] for t in built if t["wrong"] <= loss * t["shown"]), default=None)
    if best is None:
        return False
    other = min((t.get("quads", t["faces"]) for t in kept if t["wrong"] <= loss * t["shown"]), default=None)
    return other is None or best < other


def main(args: list[str]) -> None:
    if args[0] == "measure":
        measure(args[1], args[2], args[3] if len(args) > 3 else None)
        return
    rows = []
    for path in args[2:]:
        with open(path, encoding="utf-8") as stream:
            rows += [json.loads(line) for line in stream if line.strip()]
    table = {"about": "Camera results per kind of part. Rebuild with scripts/build_case_table.py. loss: low, typical and high share of the view changed.", "models": sorted({row["model"] for row in rows}), "cases": summarize(rows)}
    with open(args[1], "w", encoding="utf-8") as out:
        json.dump(table, out, indent=0)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
