# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Blender, background: Auto Segments on a mesh with and without drawn edge loops, and how well the loops came out.

    blender --background --factory-startup --python scripts/loops_test.py -- MODEL.npz DETAIL [STROKES.json] [OUT.npz]

STROKES.json is a list of {"closed": bool, "points": [[x, y, z], ...]} in the mesh's own space (they are put on the
surface). MODEL.npz holds co, start, total, loops. Set CURIOMESH_PATTERNS to a folder that has triadq/patterns.py
to also print the topology score. Prints one RESULT line of JSON per run.
"""

import importlib
import json
import os
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(REPO))
addon = importlib.import_module(os.path.basename(REPO))
addon.register()
guided = importlib.import_module(os.path.basename(REPO) + ".guided")
guides = importlib.import_module(os.path.basename(REPO) + ".guides")
ARGS = sys.argv[sys.argv.index("--") + 1 :]


def load(path):
    d = np.load(path)
    loops, starts, totals = d["loops"].tolist(), d["start"].tolist(), d["total"].tolist()
    return d["co"].tolist(), [tuple(loops[a : a + b]) for a, b in zip(starts, totals)]


def scored(co, faces):
    root = os.environ.get("CURIOMESH_PATTERNS")
    if not root:
        return {}
    sys.path.insert(0, root)
    from triadq import patterns

    measured = patterns.measure(np.asarray(co), faces)
    return {"score": patterns.score(measured)["score"], "loop_median": measured["loop_median"], "irregular": measured["irregular_share_of_vertices"]}


def runs(edges):
    """Edge chains: groups of edges that touch. (edges in the group, every vertex has two edges in the group)"""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for a, b in edges:
        groups.setdefault(find(a), []).append((a, b))
    out = []
    for group in groups.values():
        degree = {}
        for a, b in group:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        out.append((len(group), all(v == 2 for v in degree.values())))
    return max(out) if out else (0, False)


def loop_sizes(v, f, line, tol):
    """Topological edge loops (they run straight through valence-4 vertices) under a stroke: (median, longest) size in edges."""
    root = os.environ.get("CURIOMESH_PATTERNS")
    if not root or any(len(x) != 4 for x in f):
        return 0, 0
    sys.path.insert(0, root)
    from triadq import patterns

    topo = patterns.Topology(v, f)
    size = topo.loops()
    near = guides._to_line(v, line) < tol
    pick = near[topo.edge_a] & near[topo.edge_b]
    return (int(np.median(size[pick])), int(size[pick].max())) if pick.any() else (0, 0)


def run(co, faces, strokes, detail, out):
    for old in list(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    mesh = bpy.data.meshes.new("Model")
    mesh.from_pydata(co, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Model", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    gs = bpy.context.scene.curiomesh_guided
    gs.guide_curves.clear()
    gs.scope, gs.auto_detail, gs.auto_update, gs.guide_use, gs.loop_use, gs.bend_use = "AUTO", detail, False, True, True, True
    guided.state["source"] = obj.name
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    tree = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    drawn = []
    for i, stroke in enumerate(strokes):
        if "ring" in stroke:  # a ring round the model, put on the surface from outside: [centre x, centre y, z, radius, points]
            cx, cy, z, r, n = stroke["ring"]
            points = np.asarray([[cx + r * np.cos(a), cy + r * np.sin(a), z] for a in np.linspace(0.0, 2 * np.pi, int(n) + 1)])
        else:
            points = np.asarray(stroke["points"], dtype=np.float64)
        curve = bpy.data.curves.new(f"CM_Guide{i}", "CURVE")
        curve.dimensions = "3D"
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for p, c in zip(spline.points, points):
            if "ring" in stroke:
                start = Vector((c[0], c[1], c[2]))
                hit = tree.ray_cast(start, Vector((stroke["ring"][0], stroke["ring"][1], c[2])) - start)[0]
            else:
                hit = tree.ray_cast(Vector((c[0], c[1] + 1.0, c[2])), Vector((0.0, -1.0, 0.0)))[0]
            hit = hit if hit is not None else tree.find_nearest(Vector(c))[0]
            p.co = (hit.x, hit.y, hit.z, 1.0)
        o = bpy.data.objects.new(f"CM_Guide{i}", curve)
        bpy.context.scene.collection.objects.link(o)
        item = gs.guide_curves.add()
        item.obj, item.kind = o, stroke.get("kind", "LOOP")
        drawn.append(o)
    if os.environ.get("CURIOMESH_REGION"):  # redo only a box round the face
        src = guided._source_data(obj)
        c = np.asarray([np.mean([src["co"][v] for v in f], axis=0) for f in src["faces"]])
        guided.state["region"] = (c[:, 0] > -0.1) & (c[:, 0] < 0.06) & (c[:, 2] > 0.26) & (c[:, 2] < 0.43) & (c[:, 1] > 0.0)
        gs.region_use = True
    started = time.time()
    guided.run_update(bpy.context, force=True)
    job = guided.state["work"]["runner"]
    while job.busy:
        job.step(0.2)
    res = job.output
    v = np.asarray(res["co"])
    f = res["faces"]
    edges = np.concatenate([[(min(a, b), max(a, b)) for a, b in zip(x, x[1:] + x[:1])] for x in f])
    size = float(np.median(np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)))
    per = []
    for line in guided._guide_lines(bpy.context, gs, guided._source_data(obj), "LOOP") + guided._guide_lines(bpy.context, gs, guided._source_data(obj), "BEND"):
        mine = guides.loop_edges([line], v, f, 0.35 * size)
        n, closed = runs(mine)
        mean, worst = guides.follow([line], v, f, size / 4.0)
        length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        median, longest = loop_sizes(v, f, line, 0.45 * size)
        per.append({"loop_median": median, "loop_longest": longest, "stroke_edges": round(length / size, 1), "loop_edges": n, "closed_loop": closed, "mean_away": round(mean / size, 3), "worst_away": round(worst / size, 3)})
    row = {"seam": list(job.seam), "new_faces": int(res["new"].sum()), "strips_built": len(job.bend_groups), "strips_skipped": job.skipped, "why": job.why, "gap": round(job.gap, 5), "faces": len(f), "quads": sum(1 for x in f if len(x) == 4), "median_edge": round(size, 5), "seconds": round(time.time() - started), "strokes": per, **scored(v.tolist(), f)}
    print("RESULT " + json.dumps(row), flush=True)
    if out:
        np.savez(out, co=v, lens=np.array([len(x) for x in f]), flat=np.array([x for y in f for x in y]))
    job.stop()
    return row


if os.environ.get("CURIOMESH_DUMP"):  # keep what the strips were cut into, to study a failure outside Blender
    loops = importlib.import_module(os.path.basename(REPO) + ".loops")
    real = loops.insert

    def dump(co, tris, kind, items, gap, nearest, keys=()):
        np.savez(os.environ["CURIOMESH_DUMP"], co=co, tris=tris, kind=kind, gap=gap, strokes=np.concatenate([i[0] for i in items]), lens=[len(i[0]) for i in items], counts=[i[1] for i in items])
        return real(co, tris, kind, items, gap, nearest, keys)

    loops.insert = dump
model, detail = ARGS[0], float(ARGS[1])
strokes = json.load(open(ARGS[2])) if len(ARGS) > 2 and ARGS[2] != "-" else []
co, faces = load(model)
run(co, faces, strokes, detail, ARGS[3] if len(ARGS) > 3 else "")
