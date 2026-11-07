# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Look First: every Tidy Tool shows its change on the mesh before it is kept (bench/tools_design/INTERACTION.md 1).

A tool opens a `Session`: its maths (`triadq.trial.Trial`) runs in slices of about 8 ms from the operator's own modal
timer, and the candidate is drawn over the result (teal ghost edges, a purple-to-yellow wash for how far corners move,
arrows, poles that go or come, a red X where something was held back). One ring on the screen near the middle of the
view sets the strength; dragging it shows a blended stand-in at once and the exact run after the mouse rests. The card
beside the ring has up to three numbers, Apply, Cancel and After | Before. Nothing touches the result until Apply,
which is one Undo step and one History entry. No threads, no depsgraph handlers; the draw handlers and the timer live
only while a Session is open.

DEVELOPER NOTE: adding a tool in about 30 lines
----------------------------------------------
1. The maths, in `triadq/tidytools.py` (plain NumPy, unit tested):

       class MyTool(Trial):
           name = "My Tool"
           group = "RELAX"                       # RELAX, LOOPS, POLES, CLEANUP or HARD: the panel group it shows in
           defaults = {"strength": 0.6}          # every key here is also an operator property of the same name
           stages = ("poles", "move", "look")    # the card's numbers, most important first

           def check(self, base):                # "" or the refusal the card shows
               return "" if base.free.any() else NOTHING_FREE

           def run(self, base, settings, rough=False):
               co = base.co.copy()
               ...                               # move co[base.free] a little, then
               yield 0.5                         # yield every few milliseconds of work
               co[base.free] = base.project(co[base.free])   # corners stay on your original mesh
               return Candidate(co, strength=settings["strength"], exact=not rough)

2. The button, in `tidytools.py`:

       class CURIOMESH_OT_tool_my_tool(tidylive.LiveTool, bpy.types.Operator):
           bl_idname = "curiomesh.tool_my_tool"
           bl_label = "My Tool"
           bl_description = "What it does in one sentence. When to use it in a second one"
           bl_options = {"REGISTER", "UNDO"}
           trial = T.MyTool
           strength: bpy.props.FloatProperty(name="Strength", description="...", default=0.6, min=0.0, max=1.0)

   and add it to `TOOLS` there with its icon. That is all: the preview, the ring, the card, Enter / Esc / B, the
   Last line, Again, Undo and History come from here.
"""

from __future__ import annotations

import json
import math
import time

import bpy
import numpy as np

from . import history, overlay
from .triadq import trial as T

BUDGET = 0.008  # seconds of work per timer tick: half a 60 Hz frame
REST = 0.15  # seconds after the handle stops: the exact run starts
IDLE = 0.6  # seconds after the handle stops: a quick preview on a big scope is replaced by the exact one
ROUGH_FACES = 20_000  # scopes bigger than this get a quick preview first
GHOST_MOST = 200_000  # edges drawn at most (nearest to the ring win)
POLES_MOST = 5_000
ARROWS_SHOWN = 0.5  # seconds the arrows stay in After after the ring moved
DIAL_PX, DRAG_PX = 28.0, 300.0  # ring radius; mouse travel for the whole range
CARD_W, CARD_H = 340.0, 74.0
ROW_H, ROWS_W = 22.0, 430.0  # a report line on the card (Preview Tidy), and the card's width when it has lines
PINK = (1.0, 0.373, 0.635, 1.0)  # Kept lines: the mirror plane

ORANGE = (0.961, 0.651, 0.137, 1.0)  # Curio Orange: the scope edge and the ring
TEAL = (0.180, 0.769, 0.714, 1.0)  # ghost
WHITE = (1.0, 1.0, 1.0, 0.8)
GONE = (0.243, 0.812, 0.431, 1.0)
NEW = (1.0, 0.788, 0.302, 1.0)
HELD = (1.0, 0.353, 0.373, 1.0)
HALO = (0.02, 0.02, 0.03, 0.55)
GREW, SHRANK = np.array((0.231, 0.510, 0.965)), np.array((0.976, 0.502, 0.141))  # size tint: blue grew, orange shrank
ROW4_H = 22.0
BETTER, WORSE = (0.243, 0.812, 0.431, 1.0), (1.0, 0.72, 0.16, 1.0)
CARD_BG = (0.07, 0.08, 0.11, 0.9)
SLATE = (0.36, 0.40, 0.48, 1.0)
GROUP_COLOURS = {"RELAX": TEAL, "LOOPS": (0.247, 0.784, 0.910, 1.0), "POLES": NEW, "CLEANUP": (0.561, 0.639, 0.749, 1.0), "HARD": (0.608, 0.420, 1.0, 1.0)}
WASH = np.array([(0.267, 0.005, 0.329), (0.230, 0.322, 0.546), (0.128, 0.567, 0.551), (0.369, 0.789, 0.383), (0.993, 0.906, 0.144)], dtype=np.float32)  # dark purple to yellow (viridis stops)
NAVIGATION = {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "WHEELINMOUSE", "WHEELOUTMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE", "MOUSESMARTZOOM", "LEFT_ALT", "RIGHT_ALT", "LEFT_SHIFT", "RIGHT_SHIFT", "LEFT_CTRL", "RIGHT_CTRL", "HOME"}
HINT = "{name}: drag the ring for {handle}. Enter keeps it, Esc cancels, B shows before/after."
HINT_NONE = "{name}: Enter keeps it, Esc cancels, B shows before/after."
HINT_PICK = "{name}: {ask}. Enter keeps it, Esc goes back a step, B shows before/after."
PICK_PX = 24.0  # a click this close to a target dot picks it

live = {"session": None, "handles": [], "note": ""}


def _g():
    from . import guided  # late: guided imports modules that import this one

    return guided


def busy() -> bool:
    """A tool preview is open: the ladder, Tidy Up and the Adjust sliders wait."""
    return live["session"] is not None


def close(message: str) -> None:
    """Close the open preview without changing anything (an undo, a file load, another tool opening)."""
    s = live["session"]
    if s is not None:
        s.closed = True
        live["note"] = message
        live["session"] = None
        overlay.tag_redraw()


def _ui_scale() -> float:
    try:
        return float(bpy.context.preferences.system.ui_scale)
    except Exception:
        return 1.0


def _preview():
    obj = bpy.data.objects.get(_g().PREVIEW_NAME)
    return obj if obj is not None and obj.type == "MESH" else None


def scope_faces(context, res, scope: str):
    """(bool per face of the result, refusal): the faces Work On allows."""
    from . import segments

    count = len(res["faces"])
    if scope == "SELECTION":
        obj = _preview()
        if obj is None or len(obj.data.polygons) != count:
            return None, "Nothing selected: select faces of the result in Edit Mode, or set Work On to Part or Whole Result."
        if obj.mode == "EDIT":
            obj.update_from_editmode()
        picked = np.zeros(count, dtype=bool)
        obj.data.polygons.foreach_get("select", picked)
        if not picked.any():
            return None, "Nothing selected: select faces of the result in Edit Mode, or set Work On to Part or Whole Result."
        return picked, ""
    if scope == "PART":
        piece, sel = segments.seg.get("piece"), segments.seg.get("sel") or set()
        if piece is None or len(piece) != count or not sel:
            return None, "Pick a part first: click one in the Fix list or in the viewport."
        return np.isin(np.asarray(piece), sorted(sel)), ""
    return np.ones(count, dtype=bool), ""


def _source():
    """(points, faces) of your original mesh, as the search measured it."""
    guided = _g()
    base = guided.state["auto_base"]
    job = base and base.get("job")
    if job is not None and not getattr(job, "mirror", None) and getattr(job, "co", None) is not None and getattr(job, "faces", None) is not None:  # a mirrored job holds one half
        return np.asarray(job.co, dtype=np.float64), job.faces
    source = guided.state_source()
    if source is None:
        return None, None
    data = guided._source_data(source)
    return np.asarray(data["co"], dtype=np.float64), data["faces"]


def _projector(src_co, src_faces):
    from mathutils.bvhtree import BVHTree

    from . import organize, turbo

    tree = turbo.face_tree(src_co, src_faces) or BVHTree.FromPolygons(src_co.tolist(), [list(f) for f in src_faces], all_triangles=False, epsilon=0.0)
    return lambda points: organize._nearest(tree, np.asarray(points, dtype=np.float64))


def _facer(src_co, src_faces):
    """`facing(points)`: unit normals of your original at the nearest points. The lookup is built on first use, since
    only tools that read your original's bends ask for it."""
    from . import turbo

    held = {}

    def facing(points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if "find" not in held:
            tree = turbo.face_tree(src_co, src_faces)
            if tree is not None:
                normals = _face_normals(src_co, src_faces)
                held["find"] = lambda q: normals[np.asarray(tree.nearest(q)[1], dtype=np.int64)]
            else:
                from mathutils.bvhtree import BVHTree

                bvh = BVHTree.FromPolygons(src_co.tolist(), [list(f) for f in src_faces], all_triangles=False, epsilon=0.0)
                held["find"] = lambda q: np.asarray([tuple(bvh.find_nearest(x)[1] or (0.0, 0.0, 1.0)) for x in q.tolist()], dtype=np.float64).reshape(-1, 3)
        return held["find"](points)

    return facing


def _face_normals(co, faces) -> np.ndarray:
    out = np.zeros((len(faces), 3))
    by_size: dict[int, list] = {}
    for i, f in enumerate(faces):
        by_size.setdefault(len(f), []).append(i)
    for _size, ids in by_size.items():
        corners = np.asarray([faces[i] for i in ids], dtype=np.int64)
        q = co[corners]
        out[ids] = np.cross(q, np.roll(q, -1, axis=1)).sum(axis=1)  # Newell's normal
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, length, out=np.zeros_like(out), where=length > 0)


def _labels(res, co) -> dict:
    """What the kept-line tools read: UV island ("chart") and part ("piece") per face, on-a-guide ("guide") per corner."""
    from . import organize, segments

    out = {}
    if res.get("chart") is not None:
        out["chart"] = np.asarray(res["chart"])
    if segments.seg.get("piece") is not None:
        out["piece"] = np.asarray(segments.seg["piece"])
    base = _g().state["auto_base"]
    job = base.get("job") if base else None
    strokes = list(getattr(job, "guide_lines", None) or []) + [line for _tag, rows in (getattr(job, "bend_groups", None) or []) for line in rows]
    if strokes:
        out["guide"] = organize._held_by_strokes(co, (strokes, float(getattr(job, "gap", 0.0) or 0.0)))
    return out


def _looker(src_co, src_faces):
    """`look(co, faces)`: a generator returning the share of the camera pictures that differ from your original."""
    from . import autoretopo

    low, high = src_co.min(axis=0), src_co.max(axis=0)
    centre, reach = (low + high) / 2.0, (float(np.ptp(src_co, axis=0).max()) or 1.0) * 0.55
    pixels = autoretopo.MAX_PIXELS
    held = {}

    def look(co, faces):
        if "ref" not in held:
            held["ref"] = yield from autoretopo.pictures(src_co, src_faces, centre, reach, pixels)
        pictures = yield from autoretopo.pictures(np.asarray(co, dtype=np.float64), faces, centre, reach, pixels)
        wrong, shown = autoretopo.picture_counts(held["ref"], pictures, 2.0 * reach / pixels)
        return 100.0 * wrong / max(shown, 1)

    return look


def _cap(context):
    gs = getattr(context.scene, "curiomesh_guided", None)
    return gs.organize_cap if gs is not None and gs.organize_cap_use else None


# --------------------------------------------------------------------------- Session


class Session:
    """One open preview. `pump` advances its work for about `budget` seconds; `set` moves the handle; `apply` asks
    for the exact result and `ready_to_keep` says when it is there."""

    def __init__(self, context, trial: T.Trial, settings: dict, scope: str = "WHOLE", pick: dict | None = None):
        guided = _g()
        self.trial, self.settings = trial, dict(trial.defaults, **settings)
        self.handle = trial.handles[0] if trial.handles else ""
        self.res = guided.state["res"]
        self.base: T.Base | None = None
        self.full: T.Candidate | None = None  # full strength, for the instant blend while dragging
        self.cand: T.Candidate | None = None  # the newest exact (or quick) candidate
        self.shown: T.Candidate | None = None  # what is drawn: cand, or a blend while dragging
        self.status, self.error, self.closed = "getting ready", "", False
        self.before = False
        self.applying = False
        self.dragging = False
        self.drag_from = (0.0, 0.0)
        self.moved_at = -1e9
        self.anchor: int | None = None  # corner the ring sits on
        self.chips: dict = {}  # name -> (x0, y0, x1, y1) in region pixels, from the last draw
        self.batch_key, self.batches = None, None
        self.cap = _cap(context)
        self.look = None
        self.work = None
        self.ran_for = None  # the handle value the last run started with
        self.area = self.region = None
        self.region_ptr = 0
        self.hover, self.hover_lines = None, None  # (kind, value) under the mouse, and its outline
        self._tree = None  # BVH of the result, for clicks on it (built on the first one)
        self._targets = (None, None, None)  # (key, corner ids, labels) of the clickable dots
        self.pick = dict(pick or {})
        self.showing: tuple[int, int] | None = None  # (report line, spot) lit by Show
        self.message = ""  # a short word for the card ("Pick Fix These on a line first.")
        if self.res is None:
            self.error = "Make a result first with Start. Tools work on the result before you apply it."
            return
        mask, refusal = scope_faces(context, self.res, scope)
        if refusal:
            self.error = refusal
            return
        mirror = getattr(guided.state["auto_base"].get("job"), "mirror", None) if guided.state["auto_base"] else None
        self._inputs = (np.asarray(self.res["co"], dtype=np.float64), self.res["faces"], mask, (mirror[0], mirror[1]) if mirror else None)  # blockout's is ()
        self.kinds = _kinds(self.res)
        self.work = self._open()

    # ---- the work, as generators

    def _open(self):
        co, faces, mask, plane = self._inputs
        src_co, src_faces = _source()
        yield
        project = _projector(src_co, src_faces) if src_co is not None else None
        yield
        facing = _facer(src_co, src_faces) if src_co is not None else None
        labels = _labels(self.res, co) if self.trial.toggles else None
        self.base = T.make_base(co, faces, scope=mask, project=project, mirror=plane, pick=self.pick, facing=facing, labels=labels, kinds=self.kinds)
        self.look = _looker(src_co, src_faces) if src_co is not None and "look" in self.trial.stages else None
        yield
        refusal = self.trial.check(self.base)
        if refusal:
            self.cand = self.shown = T.Candidate(None, refusal=refusal)
            self.status = ""
            return
        if self.waiting():
            self.status = self.ask()
            return
        yield from self._exact(rough=self.big())

    def big(self) -> bool:
        return self.base is not None and int(self.base.scope.sum()) > ROUGH_FACES

    def _run(self, settings: dict, rough: bool):
        work = self.trial.run(self.base, settings, rough)
        while True:
            try:
                progress = next(work)
            except StopIteration as done:
                return T.diff(self.base, done.value)
            if progress is not None:
                self.status = ("quick preview " if rough else "working ") + f"{int(100 * float(progress))}%"
            yield

    def _exact(self, rough: bool):
        """The candidate at the handle's value, measured; held back while the look changes more than Most Detail Loss."""
        settings = dict(self.settings)
        self.ran_for = settings.get(self.handle)
        if self.trial.blendable and self.handle and (self.full is None or (not self.full.exact and not rough)):
            self.full = yield from self._run(dict(settings, **{self.handle: 1.0}), rough)
            if self.full.co is not None:  # a refused full-strength run has nothing to blend toward
                self.shown = T.blend(self.base, self.full, float(settings[self.handle]))
        cand = yield from self._run(settings, rough)
        self.cand = self.shown = cand
        held = None
        while True:
            self.status = "measuring look" if self.look is not None else "measuring"
            yield from T.measure(self.base, cand, self.trial.stages, self.look)
            change = cand.numbers.get("look")
            if self.cap is None or change is None or change[1] - change[0] <= self.cap or not self.handle or settings[self.handle] < 0.02:
                break
            settings[self.handle] = held = round(settings[self.handle] * 0.5, 2)  # ponytail: halving; a search between would land closer
            cand = yield from self._run(settings, rough)
            self.cand = self.shown = cand
        self.status = f"held at {held:.2f}" if held is not None else ("quick preview" if rough else "ready")
        yield from self._row_costs(cand)

    def _row_costs(self, cand):
        """A report's lines each get the look change their fix would make, worked out after everything else."""
        if self.look is None or not cand.rows or cand.co is not None:  # only while the list is up: a picked line keeps first
            return
        if "look" not in self.base.cache:
            self.base.cache["look"] = yield from self.look(self.base.co, self.base.faces)
        for row in cand.rows:
            if row.get("cost") is not None or row.get("fix") is None:
                continue
            co, faces, _parent = row["fix"]
            after = yield from self.look(np.asarray(co, dtype=np.float64), self.base.faces if faces is None else faces)
            row["cost"] = after - self.base.cache["look"]

    def set_line(self, **settings) -> None:
        """Settings without a handle changed (a report line picked): work the candidate out again now."""
        self.settings.update(settings)
        self.work, self.cand, self.message = None, None, ""
        self.status = "..."
        self.moved_at = -1e9

    def show(self, line: int):
        """Show on a report line: light up its spots, the next one each time. Returns the spot (object space) or None."""
        rows = self.shown.rows if self.shown is not None else []
        if not 0 <= line < len(rows) or not len(rows[line]["spots"]):
            return None
        spot = 0 if self.showing is None or self.showing[0] != line else (self.showing[1] + 1) % len(rows[line]["spots"])
        self.showing = (line, spot)
        return rows[line]["spots"][spot]

    def pump(self, budget: float = BUDGET) -> bool:
        """Advance the work for about `budget` seconds. True when there is something new to draw."""
        if self.closed:
            return False
        if self.res is not _g().state["res"]:
            close("The result changed, so this preview was closed. Open the tool again.")
            return True
        now = time.perf_counter()
        if self.work is None and self.base is not None and not self.dragging and not self.refused() and not self.waiting():
            rested = now - self.moved_at
            if self.stale():
                if rested >= REST or self.applying:
                    self.work = self._exact(rough=self.big() and not self.applying)
            elif not self.cand.exact and (rested >= IDLE or self.applying):
                self.work = self._exact(rough=False)
        if self.work is None:
            return False
        ends = now + budget
        before = (id(self.shown), self.status)
        while time.perf_counter() < ends:
            try:
                next(self.work)
            except StopIteration:
                self.work = None
                break
            except Exception as exc:  # a tool's maths failing must never leave a modal running
                self.work, self.error = None, f"{self.trial.name} failed: {exc}"
                break
        return (id(self.shown), self.status) != before or self.work is None

    def refused(self) -> bool:
        return self.cand is not None and bool(self.cand.refusal)

    def stale(self) -> bool:
        """No candidate yet, or the handle has moved since the last run started."""
        return self.cand is None or (bool(self.handle) and self.ran_for != self.settings[self.handle])

    def finish(self) -> None:
        """Run everything to the end now (F9, scripts): the exact candidate at the settings."""
        self.applying = True
        while self.base is None and not (self.error or self.closed) and self.work is not None:
            self.pump(1.0)
        if self.waiting():
            return
        while not (self.error or self.closed or self.refused() or self.ready_to_keep()):
            self.pump(1.0)
            if self.work is None and not (self.ready_to_keep() or self.refused() or self.stale() or (self.cand is not None and not self.cand.exact)):
                return  # nothing left that could run

    def set(self, value: float) -> None:
        """The handle moved: a blended stand-in now, the exact run once it rests."""
        if not self.handle:
            return
        value = min(max(float(value), 0.0), 1.0)
        self.settings[self.handle] = value
        self.moved_at = time.perf_counter()
        self.status = "..."
        if self.base is None:
            return  # still getting ready: the first run reads the new value
        self.work = None  # whatever was being worked out is for another value
        if self.trial.blendable and self.full is not None and not self.full.refusal:
            self.shown = T.blend(self.base, self.full, value)

    def toggle(self, key: str) -> None:
        """A chip on the card's fourth row was clicked: flip that setting and work the preview out again."""
        self.settings[key] = not bool(self.settings.get(key))
        self.ran_for, self.full, self.work = None, None, None  # stale: the exact run starts once it rests
        self.moved_at = time.perf_counter()
        self.status = "..."

    def ready_to_keep(self) -> bool:
        c = self.cand
        return not self.waiting() and c is not None and c.co is not None and not c.refusal and c.exact and self.work is None and self.shown is c and not self.stale()

    def apply(self) -> None:
        self.applying = True
        self.moved_at = -1e9

    # ---- picks: what a tool needs clicked before it can run (a strip, a pole, its partner)

    def waiting(self) -> str:
        """The pick still to be made, or ""."""
        return self.trial.waiting(self.settings) if self.trial.picks else ""

    def ask(self) -> str:
        kind = self.waiting()
        return self.trial.asks.get(kind, f"click a {kind}") if kind else ""

    def pick_kind(self) -> str:
        """What a click picks now: the pick still to be made, else the last one again (another strip, another partner)."""
        return self.waiting() or (self.trial.picks[-1] if self.trial.picks else "")

    def targets(self, kind: str):
        """(corner ids, labels) of the dots a click can pick for `kind`, or (None, None) to click anywhere."""
        key = (kind, tuple(int(self.settings.get(k, -1)) for k in self.trial.picks))
        if self._targets[0] != key and self.base is not None:
            found = self.trial.targets(self.base, self.settings, kind)
            self._targets = (key, *(found if found is not None else (None, None)))
        return self._targets[1], self._targets[2]

    def choose(self, kind: str, value: int) -> None:
        """A pick was clicked: the picks after it are cleared and the tool runs again."""
        self.settings[kind] = int(value)
        for later in self.trial.picks[self.trial.picks.index(kind) + 1 :]:
            self.settings[later] = -1
        self._again()

    def back(self) -> bool:
        """Esc with picks made: undo the last one. False when there was none."""
        made = [k for k in self.trial.picks if int(self.settings.get(k, -1)) >= 0]
        if not made:
            return False
        self.settings[made[-1]] = -1
        self._again()
        return True

    def _again(self) -> None:
        self.work, self.full, self.cand, self.shown, self.ran_for = None, None, None, None, None
        self.applying, self.moved_at, self.batch_key = False, -1e9, None
        self.hover, self.hover_lines = None, None
        self.status = self.ask() or "..."

    def ray_hit(self, region, rv3d, matrix, x, y):
        """(face, point) of the result under a screen point (object space), or None."""
        from bpy_extras import view3d_utils
        from mathutils import Matrix, Vector
        from mathutils.bvhtree import BVHTree

        if self._tree is None:
            self._tree = BVHTree.FromPolygons(self.base.co.tolist(), [list(f) for f in self.base.faces], all_triangles=False, epsilon=0.0)
        inv = Matrix(matrix).inverted()
        origin = inv @ view3d_utils.region_2d_to_origin_3d(region, rv3d, (x, y))
        ray = (inv.to_3x3() @ view3d_utils.region_2d_to_vector_3d(region, rv3d, (x, y))).normalized()
        loc, _n, face, _d = self._tree.ray_cast(Vector(origin), Vector(ray))
        return None if face is None else (int(face), np.asarray(loc, dtype=np.float64))

    def pick_under(self, region, rv3d, matrix, x, y):
        """(kind, value, corner for the ring) a click at this screen point would pick, or None."""
        kind = self.pick_kind()
        if not kind or self.base is None:
            return None
        ids, _labels = self.targets(kind)
        if ids is not None:
            if not len(ids):
                return None
            px, front = _to_screen(region, rv3d, matrix, self.base.co[ids])
            far = np.where(front, np.hypot(px[:, 0] - x, px[:, 1] - y), np.inf)
            best = int(np.argmin(far))
            return (kind, int(ids[best]), int(ids[best])) if far[best] <= PICK_PX * _ui_scale() else None
        hit = self.ray_hit(region, rv3d, matrix, x, y)
        if hit is None:
            return None
        face, point = hit
        value = self.trial.pick_at(self.base, self.settings, kind, face, point)
        if value < 0:
            return None
        corners = list(self.base.faces[face])
        return kind, int(value), int(corners[int(np.argmin(np.linalg.norm(self.base.co[corners] - point, axis=1)))])

    def set_hover(self, found) -> bool:
        """The thing under the mouse: its outline (white) for picks without dots. True when it changed."""
        key = None if found is None else found[:2]
        if key == self.hover:
            return False
        self.hover, self.hover_lines = key, None
        if key is not None and self.targets(key[0])[0] is None:
            self.hover_lines = self.trial.picked_lines(self.base, dict(self.settings, **{key[0]: key[1]}))
        return True

    # ---- drawing data

    def _batches(self, matrix):
        """GPU batches of the shown candidate, rebuilt only when it (or the object's place) changes."""
        import gpu
        from gpu_extras.batch import batch_for_shader

        guided = _g()
        picks = tuple(int(self.settings.get(k, -1)) for k in self.trial.picks)
        key = (id(self.shown), self.before, picks, self.hover, tuple(round(x, 5) for row in matrix for x in row))
        if key == self.batch_key:
            return self.batches
        base, cand = self.base, self.shown
        flat = gpu.shader.from_builtin("FLAT_COLOR")
        smooth = gpu.shader.from_builtin("SMOOTH_COLOR")
        world = lambda p: guided._world(matrix, p).astype(np.float32)  # noqa: E731
        out = {}
        scope_faces_ = [base.faces[f] for f in np.flatnonzero(base.scope).tolist()]
        border = _scope_edge(scope_faces_)
        if len(border):
            out["scope"] = _lines(flat, batch_for_shader, world(base.co[border.ravel()]), ORANGE)
        if any(v >= 0 for v in picks):  # what is picked, orange, and what the mouse is over, white
            seg = np.asarray(self.trial.picked_lines(base, self.settings), dtype=np.float64).reshape(-1, 3)
            if len(seg):
                out["picks"] = _lines(flat, batch_for_shader, world(seg + 0.02 * base.edge * _lift_of(base, seg)), ORANGE)
        if self.hover_lines is not None and len(self.hover_lines):
            seg = np.asarray(self.hover_lines, dtype=np.float64).reshape(-1, 3)
            out["hover"] = _lines(flat, batch_for_shader, world(seg + 0.02 * base.edge * _lift_of(base, seg)), (1.0, 1.0, 1.0, 0.6))
        if cand is not None and cand.co is not None and not cand.refusal:
            co = np.asarray(cand.co, dtype=np.float64)
            faces = scope_faces_ if cand.faces is None else ([cand.faces[i] for i in range(len(cand.faces)) if cand.parent is None or base.scope[cand.parent[i]]])
            edges = T.edges_of(faces)[:GHOST_MOST]
            lift = _vertex_normals(co, faces) * (0.02 * base.edge)
            out["ghost"] = _lines(flat, batch_for_shader, world((co + lift)[edges.ravel()]), TEAL)
            old = T.edges_of(scope_faces_)[:GHOST_MOST]
            out["wire"] = _lines(flat, batch_for_shader, world(base.co[old.ravel()]), (0.9, 0.9, 0.95, 0.3))
            if cand.kept is not None and len(cand.kept) and np.asarray(cand.kept).dtype.kind in "iu":
                out["kept"] = _lines(flat, batch_for_shader, world((co + 1.5 * lift)[np.asarray(cand.kept).ravel()]), PINK)
            if cand.tint is not None and len(cand.tint) == len(co):
                tris = _triangles(faces)
                if len(tris):
                    tint = _size_tint(cand.tint)
                    out["wash"] = batch_for_shader(smooth, "TRIS", {"pos": world(co + 0.5 * lift), "color": tint}, indices=tris.astype(np.int32))
            elif cand.moved is not None and len(cand.moved) == len(co):
                tris = _triangles(faces)
                if len(tris):
                    top = float(cand.moved.max())
                    tint = _wash(cand.moved / top if top > 1e-9 else np.zeros(len(co)))
                    out["wash"] = batch_for_shader(smooth, "TRIS", {"pos": world(co + 0.5 * lift), "color": tint}, indices=tris.astype(np.int32))
            if cand.arrows is not None and len(cand.arrows):
                out["arrows"] = _lines(flat, batch_for_shader, world(cand.arrows.reshape(-1, 3)), WHITE)
        if cand is not None and cand.kept is not None and len(cand.kept) and np.asarray(cand.kept).dtype.kind == "f":
            out["kept"] = _lines(flat, batch_for_shader, world(np.asarray(cand.kept, dtype=np.float64).reshape(-1, 3)), PINK)
        self.batch_key, self.batches = key, out
        return out


def _lift_of(base, points) -> np.ndarray:
    """Unit normals of the result at the corners nearest to `points` (to lift drawn lines off the surface)."""
    if "normals" not in base.cache:
        base.cache["normals"] = _vertex_normals(base.co, base.faces)
    if "lift_tree" not in base.cache:
        from mathutils.kdtree import KDTree

        tree = KDTree(len(base.co))
        for i, p in enumerate(base.co.tolist()):
            tree.insert(p, i)
        tree.balance()
        base.cache["lift_tree"] = tree
    tree = base.cache["lift_tree"]
    return base.cache["normals"][[tree.find(p)[1] for p in points.tolist()]]


def _lines(shader, batch_for_shader, points, colour):
    return batch_for_shader(shader, "LINES", {"pos": points, "color": np.tile(np.float32(colour), (len(points), 1))})


def _scope_edge(faces) -> np.ndarray:
    """Edges used by exactly one face of the scope: its outline."""
    if not faces:
        return np.zeros((0, 2), dtype=np.int64)
    a = np.concatenate([np.asarray(f, dtype=np.int64) for f in faces])
    b = np.concatenate([np.roll(np.asarray(f, dtype=np.int64), -1) for f in faces])
    pairs = np.sort(np.stack((a, b), axis=1), axis=1)
    found, count = np.unique(pairs, axis=0, return_counts=True)
    return found[count == 1]


def _vertex_normals(co, faces) -> np.ndarray:
    out = np.zeros_like(co)
    by_size: dict[int, list] = {}
    for f in faces:
        by_size.setdefault(len(f), []).append(f)
    for _size, group in by_size.items():
        ids = np.asarray(group, dtype=np.int64)
        p = co[ids]
        n = np.cross(p, np.roll(p, -1, axis=1)).sum(axis=1)  # Newell's normal
        for k in range(ids.shape[1]):
            np.add.at(out, ids[:, k], n)
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, length, out=np.zeros_like(out), where=length > 0)


def _triangles(faces) -> np.ndarray:
    tris = [(f[0], f[i], f[i + 1]) for f in faces for i in range(1, len(f) - 1)]
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _wash(t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * (len(WASH) - 1)
    i = np.minimum(t.astype(np.int64), len(WASH) - 2)
    f = (t - i)[:, None]
    rgb = WASH[i] * (1.0 - f) + WASH[i + 1] * f
    return np.concatenate((rgb, np.full((len(rgb), 1), 0.4)), axis=1).astype(np.float32)


def _size_tint(t: np.ndarray) -> np.ndarray:
    """-1..1 per corner: orange where quads shrank, pale where they stayed, blue where they grew."""
    t = np.clip(np.asarray(t, dtype=np.float64), -1.0, 1.0)[:, None]
    pale = np.array((0.92, 0.92, 0.9))
    rgb = np.where(t >= 0.0, pale + t * (GREW - pale), pale - t * (SHRANK - pale))
    return np.concatenate((rgb, np.full((len(rgb), 1), 0.45)), axis=1).astype(np.float32)


def _to_screen(region, rv3d, matrix, points) -> tuple[np.ndarray, np.ndarray]:
    """(pixels (k, 2), in front (k,)) of object-space points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(rv3d.perspective_matrix, dtype=np.float64) @ np.asarray(matrix, dtype=np.float64)
    clip = np.concatenate((points, np.ones((len(points), 1))), axis=1) @ m.T
    w = clip[:, 3]
    front = w > 1e-9
    ndc = clip[:, :2] / np.where(front, w, 1.0)[:, None]
    px = (ndc * 0.5 + 0.5) * np.array([region.width, region.height])
    return px, front


# --------------------------------------------------------------------------- drawing


def _draw_view() -> None:
    """3D layers: scope edge, old wire, ghost, wash, arrows (each line layer with a dark halo under it)."""
    s = live["session"]
    if s is None or s.base is None:
        return
    try:
        import gpu

        obj = _preview()
        if obj is None:
            return
        out = s._batches(obj.matrix_world)
        flat = gpu.shader.from_builtin("FLAT_COLOR")
        smooth = gpu.shader.from_builtin("SMOOTH_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        now = time.perf_counter()
        arrows = s.before or s.dragging or now - s.moved_at < ARROWS_SHOWN or not s.handle  # no handle: nothing to drag, so always
        layers = s.trial.layers
        if not s.before:
            if "wash" in out and "wash" in layers:
                smooth.bind()
                out["wash"].draw(smooth)
            if "wire" in out:
                gpu.state.line_width_set(1.0)
                flat.bind()
                out["wire"].draw(flat)
        for name, width in (("scope", 2.0), ("kept", 3.0), ("ghost", 2.0), ("arrows", 1.5), ("picks", 3.0), ("hover", 3.0)):
            if name not in out or (name == "ghost" and s.before) or (name == "arrows" and not arrows) or (name not in layers and name not in ("scope", "picks", "hover")):
                continue
            # the dark halo: the same lines wider, so teal, white and orange read on a light surface too
            gpu.state.line_width_set(width + 2.0)
            halo = gpu.shader.from_builtin("UNIFORM_COLOR")
            halo.bind()
            halo.uniform_float("color", HALO)
            out[name].draw(halo)
            gpu.state.line_width_set(width)
            flat.bind()
            out[name].draw(flat)
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


def _circle(x, y, r, segments=32, start=0.0, sweep=2.0 * math.pi):
    return [(x + r * math.cos(start + sweep * i / segments), y + r * math.sin(start + sweep * i / segments)) for i in range(segments + 1)]


def _poly(shader, points, colour, width, mode="LINE_STRIP"):
    import gpu

    gpu.state.line_width_set(width + 2.0)
    overlay._fill(shader, points, HALO, mode)
    gpu.state.line_width_set(width)
    overlay._fill(shader, points, colour, mode)


def _triangle(shader, x, y, size, up, colour):
    pts = [(x - size, y - size * 0.6), (x + size, y - size * 0.6), (x, y + size * 0.8)] if up else [(x - size, y + size * 0.6), (x + size, y + size * 0.6), (x, y - size * 0.8)]
    overlay._fill(shader, pts, colour, "TRIS")


def _draw_pixel() -> None:
    """Screen layers: pole marks, held X, the ring and the card."""
    s = live["session"]
    context = bpy.context
    if s is None or s.base is None or context.region is None or context.region.as_pointer() != s.region_ptr:
        return
    try:
        import gpu

        obj = _preview()
        if obj is None:
            return
        region, rv3d = context.region, context.region_data
        scale = _ui_scale()
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        shader.bind()
        gpu.state.blend_set("ALPHA")
        cand = s.shown
        if cand is not None and not cand.refusal and "poles" in s.trial.layers:
            r = 4.5 * scale
            for points, gone in ((cand.poles_gone, True), (cand.poles_new, False)):
                if points is None or not len(points) or (s.before and not gone):
                    continue
                px, front = _to_screen(region, rv3d, obj.matrix_world, points[:POLES_MOST])
                for (x, y), ok in zip(px.tolist(), front.tolist()):
                    if not ok:
                        continue
                    colour = GONE if gone else NEW
                    _poly(shader, _circle(x, y, r), colour, 1.5)  # hollow ring
                    if gone:
                        _poly(shader, [(x - r, y - r), (x + r, y + r)], colour, 1.5, "LINES")  # with a slash: goes away
                    else:
                        overlay._fill(shader, _circle(x, y, r * 0.5, 12), colour, "TRI_FAN")  # with a dot: appears
        if cand is not None and cand.held and not s.before and "held" in s.trial.layers:
            pts = np.asarray([p for p, _why in cand.held], dtype=np.float64)
            px, front = _to_screen(region, rv3d, obj.matrix_world, pts)
            r = 4.0 * scale
            for (x, y), ok in zip(px.tolist(), front.tolist()):
                if ok:
                    _poly(shader, [(x - r, y - r), (x + r, y + r), (x - r, y + r), (x + r, y - r)], HELD, 2.0, "LINES")
        _draw_targets(shader, s, region, rv3d, obj.matrix_world, scale)
        found = _found_points(s)
        if found is not None and len(found) and "found" in s.trial.layers:
            px, front = _to_screen(region, rv3d, obj.matrix_world, found[:POLES_MOST])
            lit = s.showing[1] if s.showing is not None else -1
            for i, ((x, y), ok) in enumerate(zip(px.tolist(), front.tolist())):
                if ok:
                    _dashed(shader, x, y, (14.0 if i == lit else 9.0) * scale)
        dial = _dial_point(s, region, rv3d, obj.matrix_world)
        if dial is not None and s.handle and not s.waiting():
            _draw_dial(shader, s, dial, scale)
        _draw_card(shader, s, region, dial, scale)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


def _draw_targets(shader, s, region, rv3d, matrix, scale) -> None:
    """The dots a click can pick (poles with their edge count, partners numbered nearest first) and the picked
    corners. Size and number tell them apart, never colour alone."""
    kind = s.pick_kind()
    ids, labels = s.targets(kind) if kind else (None, None)
    size = max(11, int(11 * scale))
    if ids is not None and len(ids):
        px, front = _to_screen(region, rv3d, matrix, s.base.co[ids[:POLES_MOST]])
        hovered = s.hover[1] if s.hover is not None and s.hover[0] == kind else -1
        for v, (x, y), ok, label in zip(ids.tolist(), px.tolist(), front.tolist(), labels):
            if not ok or not (0 <= x <= region.width and 0 <= y <= region.height):
                continue
            r = 4.5 * scale * (1.45 if kind == "pole" and label.isdigit() and int(label) >= 5 else 1.0)
            if v == hovered:
                r *= 1.5
            _poly(shader, _circle(x, y, r), ORANGE if v == hovered else WHITE, 1.5)
            if kind != "pole" or v == hovered:
                overlay._text(x + r + 2 * scale, y + r, size, (1, 1, 1, 0.95), label)
    for k in s.trial.picks:  # picked corners: a filled orange dot
        v = int(s.settings.get(k, -1))
        if v < 0 or s.targets(k)[0] is None:
            continue
        px, front = _to_screen(region, rv3d, matrix, s.base.co[v])
        if front[0]:
            x, y = px[0]
            overlay._fill(shader, _circle(x, y, 5.0 * scale, 16), HALO, "TRI_FAN")
            overlay._fill(shader, _circle(x, y, 3.5 * scale, 16), ORANGE, "TRI_FAN")


def _found_points(s):
    """The Found layer: the spots of the report line Show lit, else the candidate's own."""
    cand = s.shown
    if cand is None or cand.refusal:
        return None
    if s.showing is not None and s.showing[0] < len(cand.rows):
        return cand.rows[s.showing[0]]["spots"]
    return cand.found


def _dashed(shader, x, y, r, dashes=8) -> None:
    """A white dashed ring: "found here", said by shape and not by colour."""
    ring = _circle(x, y, r, dashes * 2)
    pairs = [p for i in range(0, len(ring) - 1, 2) for p in (ring[i], ring[i + 1])]
    _poly(shader, pairs, (1.0, 1.0, 1.0, 0.95), 2.0, "LINES")


def _kinds(res) -> np.ndarray | None:
    """Per face: material and UV island together, so a tool never joins faces across either border."""
    count = len(res["faces"])
    out, any_ = np.zeros(count, dtype=np.int64), False
    for key in ("mat", "chart"):
        arr = res.get(key)
        if arr is not None and len(arr) == count:
            arr = np.asarray(arr).astype(np.int64)
            out = out * (int(arr.max(initial=0)) + 1) + arr
            any_ = True
    return out if any_ else None


def _dial_point(s, region, rv3d, matrix):
    """The ring's screen point, or None when its corner is behind the view or off it."""
    if s.anchor is None:
        s.anchor = _anchor_near(s, region, rv3d, matrix, region.width / 2.0, region.height / 2.0)
    if s.anchor is None:
        return None
    px, front = _to_screen(region, rv3d, matrix, s.base.co[s.anchor])
    x, y = px[0]
    if not front[0] or not (0 <= x <= region.width and 0 <= y <= region.height):
        return None
    return float(x), float(y)


def _anchor_near(s, region, rv3d, matrix, x, y, most: float | None = None):
    """The free corner of the scope nearest to a screen point (the middle of the view when a tool opens)."""
    free = np.flatnonzero(s.base.free)
    if not len(free):
        free = np.unique(np.concatenate([np.asarray(s.base.faces[f]) for f in np.flatnonzero(s.base.scope).tolist()] or [np.zeros(0, np.int64)]))
    if not len(free):
        return None
    px, front = _to_screen(region, rv3d, matrix, s.base.co[free])
    far = np.where(front, np.hypot(px[:, 0] - x, px[:, 1] - y), np.inf)
    best = int(np.argmin(far))
    if not np.isfinite(far[best]) or (most is not None and far[best] > most):
        return None
    return int(free[best])


def _draw_dial(shader, s, dial, scale) -> None:
    x, y = dial
    r = DIAL_PX * scale
    value = float(s.settings.get(s.handle, 0.0))
    start, sweep = math.radians(225.0), -math.radians(270.0)  # from bottom-left, clockwise over the top
    _poly(shader, _circle(x, y, r, 48), (*ORANGE[:3], 0.45), 1.5)
    if value > 0.0:
        _poly(shader, _circle(x, y, r, max(2, int(48 * value)), start, sweep * value), ORANGE, 4.0)
    size = int(11 * scale)
    low, high = s.trial.words
    overlay._text(x + r * math.cos(start) - 10 * scale, y + r * math.sin(start) - 14 * scale, size, (1, 1, 1, 0.85), low)
    end = start + sweep
    overlay._text(x + r * math.cos(end) + 2 * scale, y + r * math.sin(end) - 14 * scale, size, (1, 1, 1, 0.85), high)


def card_lines(s) -> tuple:
    """The card's text: (title, [(number, sign)], third line). Also used by the open-tool box in the panel."""
    if s.error:
        return s.trial.name, [], s.error
    return T.readout(s.shown, s.trial, s.settings, s.status, s.cap)


def _chip(shader, cx, cy, text, colour, size, scale, alpha=0.55) -> float:
    """One clickable chip; returns its width."""
    import blf

    blf.size(0, size)
    cw = blf.dimensions(0, text)[0] + 14 * scale
    overlay._fill(shader, overlay._rounded_rect(cx, cy, cw, 20 * scale, 4 * scale), (*colour[:3], alpha))
    overlay._text(cx + 7 * scale, cy + 6 * scale, size, (1, 1, 1, 1 if alpha > 0.3 else 0.5), text)
    return cw


def _draw_card(shader, s, region, dial, scale) -> None:
    title, numbers, third = card_lines(s)
    refused = s.error or (s.shown is not None and s.shown.refusal)
    rows = [] if refused or s.shown is None else s.shown.rows
    words = s.message or third
    note_line = bool(not refused and numbers and words)  # the tool's own line under its numbers ("14 seam vertices ...")
    row4 = bool(s.trial.toggles or s.trial.legend)
    w, h = (ROWS_W if rows else CARD_W) * scale, (CARD_H + ROW_H * (note_line + len(rows)) + (ROW4_H if row4 else 0.0)) * scale
    if dial is None:
        x, y = 16 * scale, 16 * scale  # the ring left the view: the card parks in the bottom-left corner
    else:
        x, y = dial[0] + 24 * scale, dial[1] - 24 * scale - h
    right = region.width - overlay._sidebar_inset() - 8 * scale
    x, y = max(8 * scale, min(x, right - w)), max(8 * scale, min(y, region.height - h - 8 * scale))
    overlay._fill(shader, overlay._rounded_rect(x, y, w, h, 6 * scale), CARD_BG)
    overlay._fill(shader, overlay._rounded_rect(x, y, 5 * scale, h, 2 * scale), GROUP_COLOURS.get(s.trial.group, TEAL))
    size = max(11, int(11 * scale))
    pad = 12 * scale
    overlay._text(x + pad, y + h - 18 * scale, size, (1, 1, 1, 1), title)
    if refused:
        overlay._text(x + pad, y + h - 38 * scale, size, HELD, third[:70])
        if len(third) > 70:
            overlay._text(x + pad, y + h - 54 * scale, size, HELD, third[70:140])
    elif not numbers:
        overlay._text(x + pad, y + h - 38 * scale, size, (0.85, 0.87, 0.9, 1.0), words[:80])
    else:
        cx = x + pad
        for text, sign in numbers:
            colour = BETTER if sign > 0 else WORSE if sign < 0 else (0.85, 0.87, 0.9, 1.0)
            cx += overlay._text(cx, y + h - 38 * scale, size, colour, text) + 4 * scale
            if sign:
                _triangle(shader, cx + 4 * scale, y + h - 34 * scale, 4 * scale, sign > 0, colour)
                cx += 10 * scale
            cx += 10 * scale
        if note_line:
            overlay._text(x + pad, y + h - 60 * scale, size, (0.85, 0.87, 0.9, 1.0), words[:80])
    import blf

    chips = {}
    top = y + h - (60 + ROW_H * note_line) * scale  # the first report line sits under the numbers
    picked = s.settings.get("row", "")
    for i, row in enumerate(rows):
        ry = top - (i + 1) * ROW_H * scale + 4 * scale
        overlay._text(x + pad, ry + 6 * scale, size, (1, 1, 1, 1), row["words"])
        cost = "no fix" if row.get("fix") is None else "look ..." if row.get("cost") is None else f"look {row['cost']:+.1f}%"
        cx = x + w - pad - 150 * scale
        overlay._text(cx - 70 * scale, ry + 6 * scale, size, (0.75, 0.78, 0.85, 1.0), cost)
        shown = s.showing is not None and s.showing[0] == i
        cw = _chip(shader, cx, ry, "Show", TEAL if shown else SLATE, size, scale)
        chips[f"show:{i}"] = (cx, ry, cx + cw, ry + 20 * scale)
        cx += cw + 6 * scale
        cw = _chip(shader, cx, ry, "Fix These", TEAL if picked == row["key"] else SLATE, size, scale, 0.55 if row.get("fix") is not None else 0.2)
        chips[f"fix:{i}"] = (cx, ry, cx + cw, ry + 20 * scale)
    cx, cy = x + pad, y + (8 + (ROW4_H if row4 else 0.0)) * scale
    for name, text, lit in (("apply", "Apply  Enter", TEAL), ("cancel", "Cancel  Esc", SLATE), ("before", "Before  B" if s.before else "After  B", SLATE if s.before else TEAL)):
        cw = _chip(shader, cx, cy, text, lit, size, scale)
        chips[name] = (cx, cy, cx + cw, cy + 20 * scale)
        cx += cw + 8 * scale
    if s.trial.toggles:  # row 4: one on/off chip per setting, lit when on
        cx, cy = x + pad, y + 6 * scale
        small = max(10, int(10 * scale))
        for key, words in s.trial.toggles:
            on = bool(s.settings.get(key))
            blf.size(0, small)
            cw = blf.dimensions(0, words)[0] + 10 * scale
            overlay._fill(shader, overlay._rounded_rect(cx, cy, cw, 17 * scale, 4 * scale), (*(PINK if on else SLATE)[:3], 0.5 if on else 0.3))
            overlay._text(cx + 5 * scale, cy + 5 * scale, small, (1, 1, 1, 1.0 if on else 0.6), words)
            chips["toggle:" + key] = (cx, cy, cx + cw, cy + 17 * scale)
            cx += cw + 5 * scale
    elif s.trial.legend:
        overlay._text(x + pad, y + 10 * scale, size, (0.85, 0.87, 0.9, 1.0), s.trial.legend)
    s.chips = chips


# --------------------------------------------------------------------------- Apply


FACE_ARRAYS = ("mat", "piece", "parts", "new", "locked", "chart")


def commit(context, cand: T.Candidate, name: str, numbers: str = "") -> bool:
    """Make `cand` the result on screen and keep it as a History step (the operator's UNDO flag pushes Blender's).
    The Tidy Up passes of the result before it are let go (History still has them)."""
    from . import organize, segments

    guided = _g()
    base = guided.state["auto_base"]
    if base is None or cand is None or cand.co is None or cand.refusal or not cand.exact:
        return False
    history.before(context)
    out = base["out"]
    new = dict(out, co=np.asarray(cand.co, dtype=np.float64))
    if cand.faces is not None:
        new["faces"] = [tuple(int(v) for v in f) for f in cand.faces]
        parent = None if cand.parent is None else np.asarray(cand.parent, dtype=np.int64)
        for k in FACE_ARRAYS:
            if out.get(k) is not None and len(out[k]) == len(out["faces"]):
                new[k] = np.asarray(out[k])[parent] if parent is not None else np.zeros(len(new["faces"]), dtype=np.asarray(out[k]).dtype)
        piece = segments.seg.get("piece")
        if piece is not None and parent is not None and len(piece) == len(out["faces"]):
            segments.seg.update(piece=np.asarray(piece)[parent], batch_key=None, bad_key=None, tris=None, hover=None)
    organize.cancel()
    gs = getattr(context.scene, "curiomesh_guided", None)
    if gs is not None:
        gs["organize"] = 0.0
    base["out"] = guided.state["res"] = new
    source = guided.state_source()
    if source is not None:
        from .ladder import _make_mesh

        guided._build_preview(context, source, new, uvs=False, mesh=_make_mesh(guided.PREVIEW_NAME + "_Mesh", new, source))
    guided.state["draw_key"] = None
    history.record(context, f"Tidy: {name}", numbers)
    overlay.tag_redraw()
    return True


def numbers_text(cand: T.Candidate, trial: T.Trial, cap=None) -> str:
    return ", ".join(T.number_text(k, cand.numbers[k], cap)[0] for k in trial.stages if k in cand.numbers)


# --------------------------------------------------------------------------- the operator mixin


def _view(context):
    """(area, its main region) of the 3D view the tool works in: the one the button was pressed in, else the first."""
    areas = [context.area] if context.area is not None and context.area.type == "VIEW_3D" else []
    areas += [a for a in (context.window.screen.areas if context.window else ()) if a.type == "VIEW_3D"]
    for area in areas:
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region is not None:
            return area, region
    return None, None


def _add_handlers() -> None:
    if not live["handles"]:
        live["handles"] = [bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), "WINDOW", "POST_VIEW"), bpy.types.SpaceView3D.draw_handler_add(_draw_pixel, (), "WINDOW", "POST_PIXEL")]


def _remove_handlers() -> None:
    while live["handles"]:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(live["handles"].pop(), "WINDOW")
        except Exception:
            pass


class LiveTool:
    """Mixin for a Tidy Tool operator: set `trial` and one property per key of its `defaults`."""

    trial = None

    def _settings(self) -> dict:
        return {k: getattr(self, k) for k in self.trial.defaults if hasattr(self, k)}

    def _scope(self, context) -> str:
        ts = history.settings(context.scene)
        return ts.tool_scope if ts is not None else "WHOLE"

    def _pick(self, context) -> dict:
        """What the tool needs to know before it opens, beyond the scope (Seal the Middle: the mirror plane found)."""
        return {}

    def invoke(self, context, event):
        area, region = _view(context)
        if region is None:
            self.report({"WARNING"}, "Open a 3D view first: the preview is drawn there.")
            return {"CANCELLED"}
        close("")  # one preview at a time: opening a second cancels the first
        s = Session(context, self.trial(), self._settings(), self._scope(context), self._pick(context))
        if s.error:
            self.report({"WARNING"}, s.error)
            return {"CANCELLED"}
        obj = _preview()
        self._edit = obj is not None and obj.mode == "EDIT"
        if self._edit:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                self._edit = False
        s.area, s.region, s.region_ptr = area, region, region.as_pointer()
        live["session"], live["note"] = s, ""
        self._s = s
        _add_handlers()
        self._timer = context.window_manager.event_timer_add(1.0 / 60.0, window=context.window)
        words = HINT_PICK if s.trial.picks else HINT if s.handle else HINT_NONE
        area.header_text_set(words.format(name=s.trial.name, handle=s.handle, ask=s.trial.asks.get(s.trial.picks[0], "click on the result") if s.trial.picks else ""))
        context.window_manager.modal_handler_add(self)
        overlay.tag_redraw()
        return {"RUNNING_MODAL"}

    def _finish(self, context) -> None:
        s = self._s
        if live["session"] is s:
            live["session"] = None
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if not busy():
            _remove_handlers()
        try:
            s.area.header_text_set(None)
        except Exception:
            pass
        if getattr(self, "_edit", False):
            try:
                bpy.ops.object.mode_set(mode="EDIT")
            except RuntimeError:
                pass
        overlay.tag_redraw()

    def cancel(self, context):
        """Blender ended the modal (a file load, the area closed): as Esc."""
        close("")
        self._finish(context)

    def _keep(self, context):
        s = self._s
        for k, v in s.settings.items():
            if hasattr(self, k):
                setattr(self, k, v)  # what F9 shows and redoes
        name = s.trial.name
        numbers = numbers_text(s.cand, s.trial, s.cap)
        if not commit(context, s.cand, name, numbers):
            self.report({"WARNING"}, f"{name}: nothing to keep")
            close("")
            self._finish(context)
            return {"CANCELLED"}
        _remember(context, self.bl_idname, name, numbers, {k: v for k, v in s.settings.items() if k not in s.trial.picks})
        if s.trial.stay_open:
            return self._reopen(context, name)
        close("")
        self._finish(context)
        return {"FINISHED"}

    def _reopen(self, context, name: str):
        """A tool that stays open (Preview Tidy): the kept change gets its own Undo step now, and the tool opens again
        on the new result, its list refreshed. Esc closes it; nothing more is pushed then."""
        history._push(f"Tidy: {name}")
        old = self._s
        settings = {k: v for k, v in old.settings.items() if k != "row"}
        s = Session(context, self.trial(), settings, self._scope(context), self._pick(context))
        close("")
        if s.error:
            live["note"] = f"Kept. {s.error}"
            self._finish(context)
            return {"CANCELLED"}  # the change is kept: its Undo step was pushed above
        s.area, s.region, s.region_ptr = old.area, old.region, old.region_ptr
        live["session"], live["note"] = s, ""
        self._s = s
        _add_handlers()
        s.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _region_xy(self, event):
        r = self._s.region
        return event.mouse_x - r.x, event.mouse_y - r.y, (0 <= event.mouse_x - r.x < r.width and 0 <= event.mouse_y - r.y < r.height)

    def modal(self, context, event):
        s = self._s
        if s.closed or live["session"] is not s:
            self._finish(context)
            return {"CANCELLED"}
        if event.type == "TIMER":
            if s.pump(BUDGET):
                s.area.tag_redraw()
            if s.error:
                self.report({"WARNING"}, s.error)
                close("")
                self._finish(context)
                return {"CANCELLED"}
            if s.applying and s.ready_to_keep():
                return self._keep(context)
            if s.applying and s.cand is not None and s.cand.refusal:
                s.applying = False  # Apply does nothing on a refusal
            if s.applying and s.work is None and s.cand is not None and s.cand.co is None and not s.stale():
                s.applying = False  # a report with no line picked: nothing to keep yet
                s.message = "Pick Fix These on a line first, then Enter keeps it."
                s.area.tag_redraw()
            return {"PASS_THROUGH"}
        x, y, inside = self._region_xy(event)
        if s.dragging:
            if event.type == "MOUSEMOVE":
                fine = 0.1 if event.shift else 1.0
                value = s.drag_from[1] + (x - s.drag_from[0]) / (DRAG_PX * _ui_scale()) * fine
                if event.ctrl:
                    value = round(value * 10.0) / 10.0
                s.set(value)
                s.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                s.dragging = False
                s.moved_at = time.perf_counter()
                return {"RUNNING_MODAL"}
        if event.value == "PRESS" and event.type in {"RET", "NUMPAD_ENTER"}:
            s.apply()
            return {"RUNNING_MODAL"}
        if event.value == "PRESS" and event.type in {"ESC", "RIGHTMOUSE"} and s.back():
            s.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.value == "PRESS" and (event.type in {"ESC", "RIGHTMOUSE"} or (event.type == "Z" and (event.ctrl or event.oskey))):
            close("")
            self._finish(context)
            return {"CANCELLED"}
        if event.value == "PRESS" and event.type == "B":
            s.before = not s.before
            s.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if not inside or event.type in NAVIGATION or event.type.startswith(("NDOF", "NUMPAD")):
            return {"PASS_THROUGH"}  # the sidebar, headers and navigation keep working
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            for name, (x0, y0, x1, y1) in s.chips.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    if name == "apply":
                        s.apply()
                        return {"RUNNING_MODAL"}
                    if name == "cancel":
                        close("")
                        self._finish(context)
                        return {"CANCELLED"}
                    if name.startswith("toggle:"):
                        s.toggle(name[7:])
                        s.area.tag_redraw()
                        return {"RUNNING_MODAL"}
                    if name.startswith(("show:", "fix:")):
                        self._line(s, name)
                        return {"RUNNING_MODAL"}
                    s.before = not s.before
                    s.area.tag_redraw()
                    return {"RUNNING_MODAL"}
            obj = _preview()
            if obj is not None and s.base is not None:
                dial = _dial_point(s, s.region, s.region.data, obj.matrix_world)
                r = DIAL_PX * _ui_scale()
                if s.handle and not s.waiting() and dial is not None and abs(math.hypot(x - dial[0], y - dial[1]) - r) < 0.45 * r + 6:
                    s.dragging = True
                    s.drag_from = (x, float(s.settings.get(s.handle, 0.0)))
                    return {"RUNNING_MODAL"}
                if s.trial.picks and (s.waiting() or not s.handle or dial is None or abs(math.hypot(x - dial[0], y - dial[1]) - r) >= 0.45 * r + 6):
                    found = s.pick_under(s.region, s.region.data, obj.matrix_world, x, y)
                    if found is not None:
                        s.choose(found[0], found[1])
                        s.anchor = found[2]
                        s.area.tag_redraw()
                        return {"RUNNING_MODAL"}
                    if s.waiting():
                        return {"RUNNING_MODAL"}
                moved = _anchor_near(s, s.region, s.region.data, obj.matrix_world, x, y, most=40.0 * _ui_scale())
                if moved is not None:
                    s.anchor = moved  # clicking the result puts the ring where you are looking
                    s.area.tag_redraw()
                    return {"RUNNING_MODAL"}
            return {"RUNNING_MODAL"}
        if event.type == "MOUSEMOVE" and s.trial.picks and s.base is not None:
            obj = _preview()
            if obj is not None and s.set_hover(s.pick_under(s.region, s.region.data, obj.matrix_world, x, y)):
                s.area.tag_redraw()
        if event.type == "MOUSEMOVE" or event.type.startswith(("TIMER", "WINDOW", "EVT")):
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}  # other keys would edit the scene under the preview

    def _line(self, s, name: str) -> None:
        """A report line's chip: Show lights its spots and frames the next one; Fix These shows that line's fix."""
        kind, _, i = name.partition(":")
        i = int(i)
        rows = s.shown.rows if s.shown is not None else []
        if i >= len(rows):
            return
        if kind == "show":
            spot = s.show(i)
            obj = _preview()
            if spot is not None and obj is not None:
                from mathutils import Vector

                s.region.data.view_location = obj.matrix_world @ Vector(spot.tolist())  # frame it: the view turns round it
        elif rows[i].get("fix") is None:
            s.message = f"The {rows[i]['words'].split(' ', 1)[1]} here cannot be fixed without bad faces."
        else:
            s.showing = (i, s.showing[1]) if s.showing is not None and s.showing[0] == i else (i, -1)
            s.set_line(row=rows[i]["key"])
        s.area.tag_redraw()

    def execute(self, context):
        """No preview: run to the end and keep it (F9 redo, scripts, the agent skills)."""
        s = Session(context, self.trial(), self._settings(), self._scope(context), self._pick(context))
        if s.error:
            self.report({"WARNING"}, s.error)
            return {"CANCELLED"}
        s.finish()
        if s.waiting():
            self.report({"WARNING"}, f"{s.trial.name}: {s.ask()} first (or set it in the redo panel)")
            return {"CANCELLED"}
        if s.cand is not None and s.cand.refusal:
            self.report({"WARNING"}, s.cand.refusal)
            return {"CANCELLED"}
        if s.cand is not None and s.cand.co is None and not s.error:
            self.report({"WARNING"}, s.cand.note or f"{s.trial.name}: nothing to keep")
            return {"CANCELLED"}
        if s.error or not s.ready_to_keep():
            self.report({"WARNING"}, s.error or f"{s.trial.name} did not finish")
            return {"CANCELLED"}
        numbers = numbers_text(s.cand, s.trial, s.cap)
        if not commit(context, s.cand, s.trial.name, numbers):
            return {"CANCELLED"}
        if s.trial.stay_open:  # no UNDO flag on these operators (one step per Enter instead): push it here
            history._push(f"Tidy: {s.trial.name}")
        _remember(context, self.bl_idname, s.trial.name, numbers, {k: v for k, v in s.settings.items() if k not in s.trial.picks})
        return {"FINISHED"}


def _remember(context, idname: str, name: str, numbers: str, settings: dict) -> None:
    """The Last line, and the settings Again opens the tool with next time."""
    prefs = overlay.get_prefs()
    if not getattr(prefs, "tools_tried", True):
        prefs.tools_tried = True  # the first win is in: the first-time button retires
    ts = history.settings(context.scene)
    if ts is None:
        return
    ts.last_tool, ts.last_name, ts.last_numbers = idname, name, numbers
    ts.last_settings = json.dumps({k: v for k, v in settings.items() if isinstance(v, (int, float, bool, str)) and k != "row"})  # Again opens a report on its list


def register() -> None:
    pass


def unregister() -> None:
    close("")
    _remove_handlers()
