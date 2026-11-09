# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Fix Bad Spots R on marked parts whose sizes are not built yet: it must build them as tries, then open the preview
(Enter keeps, Esc changes nothing), never apply at once. Runs without Blender: bpy is stubbed."""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Patch:
    """What pytest's monkeypatch does, undone by the test's cleanup."""

    def __init__(self, case):
        self.case = case

    def setitem(self, mapping, key, value):
        p = mock.patch.dict(mapping, {key: value})
        p.start()
        self.case.addCleanup(p.stop)

    def setattr(self, obj, name, value):
        p = mock.patch.object(obj, name, value)
        p.start()
        self.case.addCleanup(p.stop)


def load(patch):
    bpy = mock.MagicMock(name="bpy")
    bpy.types.Operator = type("Operator", (), {})
    for n in ("Panel", "PropertyGroup", "Menu", "Header", "UIList", "AddonPreferences", "Scene", "Object", "WindowManager"):
        setattr(bpy.types, n, type(n, (), {}))
    class Stubs(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Every module of Blender's (bpy.*, gpu.*, mathutils.*, ...) is a mock that imports anything from it."""

        roots = {"gpu", "gpu_extras", "bmesh", "mathutils", "blf", "bpy_extras", "bpy_types", "aud", "idprop"}

        def find_spec(self, name, path, target=None):
            return importlib.machinery.ModuleSpec(name, self, is_package=True) if name.split(".")[0] in self.roots or name.startswith("bpy.") else None

        def create_module(self, spec):
            m = mock.MagicMock(name=spec.name)
            m.__path__, m.__spec__, m.__name__ = [], spec, spec.name
            if spec.name == "bpy_extras.io_utils":
                m.ExportHelper, m.ImportHelper = type("ExportHelper", (), {}), type("ImportHelper", (), {})
            return m

        def exec_module(self, module):
            pass

    patch.setitem(sys.modules, "bpy", bpy)
    finder = Stubs()
    sys.meta_path.insert(0, finder)
    patch.case.addCleanup(sys.meta_path.remove, finder)
    bpy.__path__, bpy.__spec__ = [], importlib.machinery.ModuleSpec("bpy", None, is_package=True)
    for n in ("bpy.types",):
        patch.setitem(sys.modules, n, bpy.types)
    spec = importlib.util.spec_from_file_location("curiomesh", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    pkg = importlib.util.module_from_spec(spec)
    patch.setitem(sys.modules, "curiomesh", pkg)
    return SimpleNamespace(tt=importlib.import_module("curiomesh.tidytools"), seg=importlib.import_module("curiomesh.segments"), ladder=importlib.import_module("curiomesh.ladder"), live=importlib.import_module("curiomesh.tidylive"))


class Swap:
    """PartSwap with its plan fixed: part 0 has a size still to build."""

    def __init__(self, cm, monkeypatch, busy_ticks=2):
        self.cm, self.log, self.ticks = cm, [], busy_ticks
        self.runner = SimpleNamespace(busy=True, warnings=[], _stopped=False)
        plan = {"name": "Redo Marked Spots", "tries": {}, "sizes": {}, "todo": [0], "why": ""}
        monkeypatch.setattr(cm.live, "_view", lambda ctx: (object(), object()))
        monkeypatch.setattr(cm.ladder, "build_sizes", lambda *a, **k: self.runner)
        monkeypatch.setattr(cm.ladder, "finish_build", lambda r: {} if r._stopped else {0: (3, None)})
        monkeypatch.setattr(cm.seg, "preview_tries", lambda tries: self.log.append(("preview", dict(tries))) or ("co", [], None, "piece"))
        monkeypatch.setattr(cm.tt, "_guided", lambda: SimpleNamespace(stop_work=lambda: self.log.append("stopped")))
        op = type("Op", (cm.tt.PartSwap,), {"plan": lambda s, c: dict(plan), "keep": lambda s, c, p: self.log.append("keep") or True, "report": lambda s, t, m: None, "execute": lambda s, c: self.log.append("execute") or {"FINISHED"}, "_open": lambda s, c, e: self.log.append("open") or None})()
        self.op = op
        self.ctx = SimpleNamespace(window_manager=mock.MagicMock(), window=object())
        monkeypatch.setattr(cm.seg, "select", lambda parts: None)
        monkeypatch.setattr(cm.seg, "seg", {"piece": np.zeros(1)})


def event(kind, value="PRESS"):
    return SimpleNamespace(type=kind, value=value)




class PartSwapPreview(unittest.TestCase):
    def setUp(self):
        self.patch = Patch(self)
        self.cm = load(self.patch)

    def swap(self):
        return Swap(self.cm, self.patch)

    def test_parts_to_build_open_a_preview_not_an_apply(self):
        s = self.swap()
        self.assertEqual(s.op.invoke(s.ctx, None), {"RUNNING_MODAL"})
        self.assertEqual(s.log, [])  # nothing applied, nothing executed, nothing shown yet: it is building
        self.assertEqual(s.op.modal(s.ctx, event("TIMER")), {"PASS_THROUGH"})  # still busy
        s.runner.busy = False
        self.assertEqual(s.op.modal(s.ctx, event("TIMER")), {"RUNNING_MODAL"})
        self.assertEqual(s.log, [("preview", {0: 3}), "open"])  # the built try is what the preview shows; nothing is kept yet
        self.assertTrue("keep" not in s.log and "execute" not in s.log)


    def test_esc_while_building_changes_nothing(self):
        s = self.swap()
        s.op.invoke(s.ctx, None)
        s.runner._stopped = True
        self.assertEqual(s.op.modal(s.ctx, event("ESC")), {"CANCELLED"})
        self.assertEqual(s.log, ["stopped"])  # the job is stopped, the parts get their old numbers back, nothing kept


    def test_held_job_leaves_the_result_and_makes_a_try(self):
        monkeypatch, seg = Patch(self), self.cm.seg
        record = {"tries": [{"method": "REDUCE", "detail": 0.5}], "cur": 0, "method": "REDUCE", "after": 10}
        monkeypatch.setattr(seg, "seg", {"parts": [record], "piece": np.zeros(1)})
        base = {"out": {"faces": [1]}, "id": 1}
        monkeypatch.setattr(seg, "_g", lambda: SimpleNamespace(state={"auto_base": base, "message": ""}))
        runner = SimpleNamespace(committed=False, base=base, report=None, hold=True, held={}, new_tries={0: {"method": "REDUCE", "detail": 0.65, "after": 20, "change": 1.0}}, status=lambda: "")
        seg.commit(runner)
        self.assertEqual(runner.held, {0: 1})
        self.assertEqual((record["cur"], record["after"], base["out"]), (0, 10, {"faces": [1]}))  # the result on screen is untouched


if __name__ == "__main__":
    unittest.main()
