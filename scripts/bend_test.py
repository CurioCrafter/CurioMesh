# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""How much a bend hurts a retopo result, with and without a deformation line (plain Python and NumPy, no Blender).

    python scripts/bend_test.py SOURCE.npz RESULT.npz [RESULT.npz ...] --axis 0 --at 0.27 --width 0.02 --degrees 50

The mesh is bent like a simple hinge: points across `at` (along the axis of the second coordinate pair) are turned about
`axis` through `at`, by an angle that grows from 0 to `degrees` over `width` either side of it. For every result it prints
how many quads in the bend zone lose their shape (scaled Jacobian under 0.2, where they had more before), how much
their worst corner moves, and how much the volume changes next to the source bent the same way.
SOURCE.npz holds co, start, total, loops; each RESULT.npz holds co, lens, flat (what loops_test.py saves).
"""

import json
import sys

import numpy as np


def bend(co, axis, at, width, degrees, up=2):
    """Turn the points above `at` (along coordinate `up`) about the coordinate `axis` through the line, smoothly over 2 * width."""
    s = (co[:, up] - at) / max(width, 1e-12)
    ramp = np.clip((s + 1.0) / 2.0, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    angle = np.radians(degrees) * ramp
    a, b = [k for k in range(3) if k != axis][::1]
    out = co.copy()
    centre = np.zeros(3)
    centre[up] = at
    rel = co - centre
    other = a if a != up else b
    c, s_ = np.cos(angle), np.sin(angle)
    out[:, up] = centre[up] + rel[:, up] * c - rel[:, other] * s_
    out[:, other] = rel[:, other] * c + rel[:, up] * s_
    return out


def jacobian(co, quads):
    """Worst scaled Jacobian of the four corners of each quad: 1 is a square, 0 or less is collapsed or folded."""
    p = co[quads]
    worst = np.full(len(quads), 9.0)
    for k in range(4):
        e1, e2 = p[:, (k + 1) % 4] - p[:, k], p[:, (k - 1) % 4] - p[:, k]
        n = np.cross(e1, e2)
        worst = np.minimum(worst, np.linalg.norm(n, axis=1) / np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-30))
    return worst


def volume(co, faces):
    total = 0.0
    for f in faces:
        for k in range(1, len(f) - 1):
            total += float(np.dot(co[f[0]], np.cross(co[f[k]], co[f[k + 1]]))) / 6.0
    return total


def load_source(path):
    d = np.load(path)
    loops, starts, totals = d["loops"].tolist(), d["start"].tolist(), d["total"].tolist()
    return d["co"], [tuple(loops[a : a + b]) for a, b in zip(starts, totals)]


def load_result(path):
    d = np.load(path)
    flat, lens, out, k = d["flat"].tolist(), d["lens"].tolist(), [], 0
    for n in lens:
        out.append(tuple(flat[k : k + n]))
        k += n
    return d["co"], out


def measure(co, faces, source, args):
    bent = bend(co, *args)
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64)
    zone = np.abs(co[quads].mean(axis=1)[:, 2] - args[1]) < 2.0 * args[2]
    before, after = jacobian(co, quads)[zone], jacobian(bent, quads)[zone]
    v0, v1 = volume(co, faces), volume(bent, faces)
    sc, sf = source
    s0, s1 = volume(sc, sf), volume(bend(sc, *args), sf)
    return {
        "zone_quads": int(zone.sum()),
        "collapsed_before": int((before < 0.2).sum()),
        "collapsed_after": int((after < 0.2).sum()),
        "collapsed_share_after": round(float((after < 0.2).mean()), 3),
        "worst_corner_drop": round(float(np.mean(before - after)), 3),
        "volume_change": round(v1 / v0 - 1.0, 4),
        "volume_change_source": round(s1 / s0 - 1.0, 4),
    }


if __name__ == "__main__":
    argv = sys.argv[1:]
    opt = {"--axis": 0, "--at": 0.27, "--width": 0.02, "--degrees": 50.0}
    paths = []
    while argv:
        a = argv.pop(0)
        if a in opt:
            opt[a] = type(opt[a])(argv.pop(0))
        else:
            paths.append(a)
    args = (int(opt["--axis"]), float(opt["--at"]), float(opt["--width"]), float(opt["--degrees"]))
    source = load_source(paths[0])
    for path in paths[1:]:
        co, faces = load_result(path)
        print(path, json.dumps(measure(co, faces, source, args)))
