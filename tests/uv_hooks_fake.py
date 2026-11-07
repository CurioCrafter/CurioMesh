# SPDX-License-Identifier: GPL-3.0-or-later
"""A stand-in for the part of a Blender mesh that `uv_layers` uses, so the writer runs without bpy."""

from __future__ import annotations

import numpy as np


class _Data:
    def __init__(self, layer):
        self.layer = layer

    def foreach_get(self, name, out):
        out[:] = self.layer.uv.ravel()

    def foreach_set(self, name, values):
        self.layer.uv = np.asarray(values, np.float32).reshape(-1, 2).copy()


class _Layer:
    def __init__(self, owner, name):
        self.name, self.uv = name, np.zeros((len(owner.loops), 2), np.float32)
        self.data = _Data(self)


class _Layers(list):
    active_index = 0

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def new(self, name="UVMap"):
        self.append(_Layer(self.owner, name))
        return self[-1]

    def get(self, name):
        return next((x for x in self if x.name == name), None)

    def remove(self, layer):
        super().remove(layer)

    @property
    def active(self):
        return self[self.active_index] if self else None

    def names(self):
        return [x.name for x in self]


class _Seq:
    def __init__(self, n=0):
        self.n = n

    def __len__(self):
        return self.n


class _Verts(_Seq):
    def __init__(self, co):
        super().__init__(len(co))
        self.co = np.asarray(co, np.float32)

    def foreach_get(self, name, out):
        out[:] = self.co.ravel()


class _Loops(_Seq):
    def __init__(self, v):
        super().__init__(len(v))
        self.v = np.asarray(v, np.int64)

    def foreach_get(self, name, out):
        out[:] = self.v


class _Polys(_Seq):
    def __init__(self, sizes, mat):
        super().__init__(len(sizes))
        self.sizes, self.mat = np.asarray(sizes, np.int64), np.asarray(mat, np.int32)

    def foreach_get(self, name, out):
        out[:] = self.sizes if name == "loop_total" else self.mat

    def foreach_set(self, name, values):
        self.mat = np.asarray(values, np.int32).copy()


class FakeMesh:
    def __init__(self):
        self.uv_layers = _Layers(self)
        self.clear_geometry()

    def clear_geometry(self):
        self.vertices, self.loops, self.polygons = _Verts(np.zeros((0, 3))), _Loops([]), _Polys([], [])

    def from_pydata(self, co, edges, faces):
        self.vertices = _Verts(co)
        self.loops = _Loops([v for f in faces for v in f])
        self.polygons = _Polys([len(f) for f in faces], np.zeros(len(faces)))

    def update(self):
        pass
