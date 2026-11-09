# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""The targets of the lines in `guided.py` (package A1, SPEC 4.6, 11.1). Main-only file.

`kinds(old_charts, source, src, gs)` stands where `_uv_charts` was called: Exact gives the old output, Light the kinds of
`triadq.uvkeep`, Off nothing. `after_apply(mesh, source, res, gs)` runs right after the result mesh is made on the commit
and Apply paths (never on previews or per-part meshes): in Light it writes the finished fit that matches `res`, or, with
none, the per-side transfer alone.
"""

from __future__ import annotations

import numpy as np

from . import sliced, uv_layers
from .triadq import uvkeep
from .triadq.types import FlatFaces

SOURCE_DATA = None  # tests set this; inside Blender guided._source_data is used
LAST: dict = {}  # what the last after_apply wrote: {"hash", "fitted", "numbers", "messages"}


def kinds(old_charts, source, src, gs):
    mode = uvkeep.mode_of(gs)
    if mode == "off" or getattr(gs, "bake_detail", "") == "NORMAL_COLOR":
        return None
    if mode == "exact":
        return old_charts(source, src)
    target = 0
    try:
        from . import guided

        target = guided.auto_target(gs, len(src["faces"]))
    except Exception:  # not running inside the add-on (tests)
        pass
    return as_islands(uvkeep.kinds_for_remesh(source, src, gs, target or None), _islands_of(source, src))


def _islands_of(source, src):
    """UV island of every source face in the active layer, numbered as guided's own UV transfer numbers them, or None."""
    try:
        from . import guided

        uv = uv_layers.read_uv(source.data)
        return None if uv is None else np.asarray(guided._uv_islands(src, uv)).reshape(-1)
    except Exception:  # not inside the add-on (tests), or no layer
        return None


def as_islands(got, island):
    """The kinds with each code's third entry an island id: guided._transfer_uvs reads res["chart"] (filled from it) as
    the island of the active layer. Each code gets the island most of its faces lie on, one not given to another code
    when there is one, so codes stay apart (tools never join faces of two kinds)."""
    if got is None or island is None or len(island) != len(got[0]):
        return got
    codes, mats, _kind = got
    pairs, count = np.unique(np.stack([np.asarray(codes, np.int64), island.astype(np.int64)], axis=1), axis=0, return_counts=True)
    out = np.full(len(mats), -1, np.int64)
    taken = set()
    for c, i in pairs[np.argsort(-count, kind="stable")].tolist():
        if out[c] < 0 and i not in taken:
            out[c] = i
            taken.add(i)
    for c in np.flatnonzero(out < 0).tolist():  # every island of this code went to another one: share the biggest
        hit = pairs[pairs[:, 0] == c]
        out[c] = hit[0, 1] if len(hit) else 0
    return codes, mats, out


def _write_back(res: dict, c: uvkeep.Candidate) -> None:
    """Make `res` describe the written mesh, so the normals, edge marks and selection that follow line up with it."""

    old = FlatFaces.of(res["faces"])
    nf = len(old)
    res["co"] = c.points
    if len(c.indptr) - 1 != nf or not np.array_equal(c.flat, old.flat):
        res["faces"] = FlatFaces(c.indptr, c.flat) if isinstance(res["faces"], FlatFaces) else [tuple(c.flat[c.indptr[i]:c.indptr[i + 1]].tolist()) for i in range(len(c.indptr) - 1)]
        for key, value in list(res.items()):
            if key not in ("co", "faces") and isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == nf:
                res[key] = value[c.origin]


def _source_data(source):
    if SOURCE_DATA is not None:
        return SOURCE_DATA(source)
    from . import guided

    return guided._source_data(source)


def after_apply(mesh, source, res, gs) -> None:
    sliced.finish(after_apply_steps(mesh, source, res, gs))


def after_apply_steps(mesh, source, res, gs):
    """`after_apply` as timer steps (`sliced`): a transfer with no finished fit runs in short pieces. Stop keeps the
    UVs the mesh was made with (guided's own transfer) and says so in LAST["messages"]."""
    yield from ()
    if uvkeep.mode_of(gs) != "light":
        return
    src = _source_data(source)
    sa, plan = uvkeep.plan_for(source, src, gs)
    if sa is None:
        return
    faces = FlatFaces.of(res["faces"])
    key = uvkeep.result_key(res["co"], faces.indptr, faces.flat)
    c = uvkeep.CANDIDATES.get(key)
    fitted = c is not None
    if c is None:  # no finished fit for this result: the per-side transfer alone, geometry untouched
        stage = uvkeep.fit_stage(source, src, res, gs, only_transfer=True)
        sliced.tick("Carrying the UVs over")
        while not stage.done:
            if sliced.stopped():
                stage.stop()
            stage.step()
            yield
        c = stage.candidate
        if c is None:  # stopped
            LAST.clear()
            LAST.update(hash="", fitted=False, numbers={}, messages=["Stopped: the UVs are a plain copy from your mesh"])
            return
        c.messages.append("Seam Fit was not run on this result. Look runs it.")
    uv_layers.apply_candidate(mesh, c.points, c.indptr, c.flat, c.material, c.layers, c.active)
    _write_back(res, c)
    LAST.clear()
    LAST.update(hash=c.key, fitted=fitted, numbers=dict(c.numbers), messages=list(c.messages))
