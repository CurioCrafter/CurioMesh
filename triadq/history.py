# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The steps behind "you can always go back": what the add-on kept at each numbered step, for Ctrl+Z and History.

Blender's own undo brings back the scene, including a step number the add-on keeps on it. What lives outside Blender
(the results worked out in Python: Tidy Up passes, ladder rungs, tool results) is kept here under that number, so after
an undo or redo the add-on puts back the entry of the number Blender restored. Each entry holds references to the
arrays that were on screen, never copies; entries are let go oldest first only when the computer runs short of memory,
and `let_go` counts them so the panel can say so. Plain Python.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

COALESCE = 1.5  # seconds: slider moves of the same kind this close together are one step (one drag)


@dataclass
class Entry:
    step: int  # the scene's step number this entry belongs to
    name: str  # "Tidy: Again", "Quad Size", "Smooth Corners"
    numbers: str  # "Tidiness 56 -> 61, look 6.2% -> 6.4%"
    state: object  # whatever the add-on needs to put this step back (references, not copies)
    kind: str = ""  # entries of the same kind made within COALESCE seconds of each other merge (a slider drag)
    size: int = 0  # bytes of arrays this entry holds that no earlier entry held
    when: float = field(default_factory=time.monotonic)


def size_of(obj, seen: set, depth: int = 4) -> int:
    """Bytes of the NumPy arrays and face lists reachable from `obj` (dicts, lists, tuples) not in `seen` yet."""
    if depth < 0 or id(obj) in seen:
        return 0
    if isinstance(obj, np.ndarray):
        seen.add(id(obj))
        return int(obj.nbytes)
    if isinstance(obj, dict):
        seen.add(id(obj))
        return sum(size_of(v, seen, depth - 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        seen.add(id(obj))
        if obj and isinstance(obj[0], tuple) and obj[0] and isinstance(obj[0][0], (int, np.integer)):
            return 64 * len(obj)  # a face list: about a tuple of four ints each
        return sum(size_of(v, seen, depth - 1) for v in obj)
    return 0


class Steps:
    """Entries in the order they were made. `at(step)` is what Ctrl+Z needs; `entries` is the History list."""

    def __init__(self):
        self.entries: list[Entry] = []
        self.let_go = 0  # entries dropped to free memory, said in the panel
        self._seen: set = set()

    def clear(self) -> None:
        self.entries, self.let_go, self._seen = [], 0, set()

    def at(self, step: int) -> Entry | None:
        """The entry of this step number (the last one made with it)."""
        for entry in reversed(self.entries):
            if entry.step == step:
                return entry
        return None

    def index_of(self, step: int) -> int:
        """Position in `entries` of the step on screen, or -1."""
        for i in range(len(self.entries) - 1, -1, -1):
            if self.entries[i].step == step:
                return i
        return -1

    def add(self, current: int, name: str, numbers: str, state, kind: str = "", now: float | None = None) -> int:
        """Keep `state` as the step after `current` and return its number. Entries after `current` (steps you undid
        and then did something else) go, as Blender's redo does. A slider kind within COALESCE seconds of the last
        entry, still on screen, replaces it and keeps its number, so one drag is one step."""
        now = time.monotonic() if now is None else now
        self.entries = [e for e in self.entries if e.step <= current]
        last = self.entries[-1] if self.entries else None
        if kind and last is not None and last.kind == kind and last.step == current and now - last.when < COALESCE:
            last.name, last.numbers, last.state, last.when = name, numbers, state, now
            last.size += size_of(state, self._seen)
            return current
        step = current + 1
        self.entries.append(Entry(step, name, numbers, state, kind, size_of(state, self._seen), now))
        return step

    def renumber(self, kind: str, numbers: str) -> bool:
        """Fill in the numbers of the newest entry once they are known (a slider's result settles later)."""
        if self.entries and self.entries[-1].kind == kind:
            self.entries[-1].numbers = numbers
            return True
        return False

    def size(self) -> int:
        return sum(e.size for e in self.entries)

    def let_go_oldest(self, free: int) -> int:
        """Drop the oldest entries until about `free` bytes are let go, never the newest. Returns how many went."""
        gone, freed = 0, 0
        while len(self.entries) > 1 and freed < free:
            freed += self.entries.pop(0).size
            gone += 1
        self.let_go += gone
        if gone:
            self._seen = set()  # sizes are estimates; recount from the entries left
            for entry in self.entries:
                entry.size = size_of(entry.state, self._seen)
        return gone
