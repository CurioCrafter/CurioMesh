# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""partpack: two grid parts written, read back as plain files (no bpy)."""

import json
import os
import re
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import partpack  # noqa: E402


def grid(n, shift, seed):
    pts = np.array([[x + shift, y, 0.1 * np.sin(x * 1.3 + y + seed)] for y in range(n + 1) for x in range(n + 1)], dtype=np.float64) / 7.0
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    ring = [x for x in range(n)] + [n + y * (n + 1) for y in range(n)] + [(n + 1) * n + x for x in range(n, 0, -1)] + [y * (n + 1) for y in range(n, 0, -1)]
    return pts, faces, ring + [ring[0]]


def read_obj(path):
    v, f, mats, m = [], [], [], 0
    for line in open(path, encoding="utf-8"):
        t = line.split()
        if t[0] == "v":
            v.append([float(x) for x in t[1:]])
        elif t[0] == "usemtl":
            m = int(t[1][1:])
        elif t[0] == "f":
            f.append([int(x) - 1 for x in t[1:]])
            mats.append(m)
    return np.array(v), f, mats


def main():
    parts = []
    for i, (n, sh) in enumerate([(3, 0.0), (4, 9.0)]):
        pts, faces, loop = grid(n, sh, i)
        parts.append({"id": i + 5, "name": f"Part {i + 6}", "kind": "panel", "numbers": {"faces": len(faces), "view_change_pct": 6.1, "tidiness": 71.0},
                      "issues": ["pole"], "points": pts, "faces": faces, "mats": [i] * len(faces), "border": [loop],
                      "source": (pts, faces), "around": (pts + 5, faces)})
    key = partpack.result_key(100, parts[0]["points"][:4])
    assert key == partpack.result_key(100, parts[0]["points"][:4][::-1]) and len(key) == 8
    assert key != partpack.result_key(101, parts[0]["points"][:4])
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "Statue-parts-1")
        partpack.write_parts(folder, "Statue", parts, {"source_faces": 10, "source_verts": 8, "result_faces": 25, "result_key": key, "made_by": "CurioMesh 0.9.9"}, "hello")
        assert os.path.isdir(os.path.join(folder, "returned")) and not os.listdir(os.path.join(folder, "returned"))
        man = json.load(open(os.path.join(folder, "parts.json")))
        assert man["format"] == "curiomesh-parts" and man["version"] == [1, 0] and man["note"] == "hello" and man["object"] == "Statue"
        assert set(man) >= {"format", "version", "made_by", "object", "result", "units", "rules", "parts", "note"}
        assert set(man["result"]) == {"source_faces", "source_verts", "result_faces", "result_key"}
        for entry, p in zip(man["parts"], parts):
            assert set(entry) == {"id", "name", "kind", "faces", "view_change_pct", "tidiness", "issues", "border", "border_points", "materials"}
            assert entry["border"] == [p["border"][0]] and entry["border"][0][0] == entry["border"][0][-1]
            assert len(entry["border_points"]) == len(p["border"][0]) - 1
            f32 = p["points"].astype(np.float32)
            v, f, mats = read_obj(os.path.join(folder, f"part_{p['id']}.obj"))
            assert np.array_equal(v.astype(np.float32), f32), "OBJ floats round-trip"
            assert f == [list(x) for x in p["faces"]] and set(mats) == set(entry["materials"])
            assert np.array_equal(np.array(entry["border_points"], dtype=np.float32), f32[p["border"][0][:-1]])
            for suffix in ("_source", "_around"):
                assert os.path.isfile(os.path.join(folder, f"part_{p['id']}{suffix}.obj"))
        assert re.search(r"usemtl m0", open(os.path.join(folder, "part_5.obj")).read())
        assert os.path.isfile(os.path.join(folder, "README.txt"))
        z = os.path.join(tmp, "r.zip")
        partpack.write_report_zip(z, folder, "# report\n", {"a": 1})
        names = zipfile.ZipFile(z).namelist()
        for want in ("report.md", "settings.json", "Statue-parts-1/parts.json", "Statue-parts-1/part_5.obj", "Statue-parts-1/part_6_around.obj", "Statue-parts-1/README.txt", "Statue-parts-1/returned/"):
            assert want in names, (want, names)
    print("test_partpack ok")


def test_partpack():
    main()


if __name__ == "__main__":
    main()
