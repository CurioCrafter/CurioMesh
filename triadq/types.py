# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Iterable

import numpy as np

from . import tasks


Face = tuple[int, ...]
UVFace = tuple[int, ...]


class FlatFaces(Sequence):
    """A list of faces held as two arrays: a tenth of the memory of a Python list of tuples, and it reads like one.

    Face `k` is `flat[indptr[k]:indptr[k + 1]]`. Reading one builds its tuple, so loops over a big mesh should use the arrays.
    """

    __slots__ = ("indptr", "flat")

    def __init__(self, indptr: np.ndarray, flat: np.ndarray):
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.flat = np.ascontiguousarray(flat, dtype=np.int64)

    @classmethod
    def of(cls, faces) -> FlatFaces:
        if isinstance(faces, cls):
            return faces
        sizes = np.fromiter(map(len, faces), dtype=np.int64, count=len(faces))
        indptr = np.zeros(len(faces) + 1, dtype=np.int64)
        np.cumsum(sizes, out=indptr[1:])
        return cls(indptr, np.fromiter(chain.from_iterable(faces), dtype=np.int64, count=int(indptr[-1])))

    def __len__(self) -> int:
        return len(self.indptr) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[k] for k in range(*index.indices(len(self)))]
        index = index + len(self) if index < 0 else index
        return tuple(self.flat[self.indptr[index] : self.indptr[index + 1]].tolist())

    def __iter__(self):
        for start in range(0, len(self), 65536):
            ends = self.indptr[start : start + 65537].tolist()
            values = self.flat[ends[0] : ends[-1]].tolist()
            base = ends[0]
            for a, b in zip(ends[:-1], ends[1:]):
                yield tuple(values[a - base : b - base])

    def __eq__(self, other) -> bool:
        return list(self) == list(other)


@dataclass(slots=True)
class MeshData:
    vertices: np.ndarray
    faces: list[Face]
    face_materials: list[str] = field(default_factory=list)
    uvs: np.ndarray | None = None
    face_uvs: list[UVFace] | None = None
    name: str = "mesh"

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be an (N, 3) array")
        self.faces = [tuple(int(i) for i in face) for face in self.faces if len(face) >= 3]
        if not self.face_materials:
            self.face_materials = ["default"] * len(self.faces)
        if len(self.face_materials) != len(self.faces):
            raise ValueError("face_materials length must match faces")
        if self.uvs is not None:
            self.uvs = np.asarray(self.uvs, dtype=np.float64)
            if self.uvs.ndim != 2 or self.uvs.shape[1] != 2:
                raise ValueError("uvs must be an (N, 2) array")
        if self.face_uvs is not None and len(self.face_uvs) != len(self.faces):
            raise ValueError("face_uvs length must match faces")

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    def copy_with(
        self,
        *,
        vertices: np.ndarray | None = None,
        faces: Iterable[Face] | None = None,
        face_materials: Iterable[str] | None = None,
        face_uvs: Iterable[UVFace] | None = None,
        name: str | None = None,
    ) -> MeshData:
        next_faces = list(faces) if faces is not None else list(self.faces)
        next_materials = (
            list(face_materials)
            if face_materials is not None
            else list(self.face_materials[: len(next_faces)])
        )
        if len(next_materials) < len(next_faces):
            next_materials.extend(["default"] * (len(next_faces) - len(next_materials)))
        return MeshData(
            vertices=np.array(vertices if vertices is not None else self.vertices, dtype=np.float64),
            faces=next_faces,
            face_materials=next_materials,
            uvs=None if self.uvs is None else np.array(self.uvs, dtype=np.float64),
            face_uvs=list(face_uvs) if face_uvs is not None else (None if self.face_uvs is None else list(self.face_uvs)),
            name=name or self.name,
        )


def face_edges(face: Face) -> list[tuple[int, int]]:
    return [
        tuple(sorted((face[i], face[(i + 1) % len(face)])))
        for i in range(len(face))
    ]


class Cancelled(Exception):
    """Raised inside a worker when the caller no longer wants its result."""


_cancel = tasks.local()  # each worker has its own


def cancel_with(check) -> None:
    """Make `check_cancelled()` on this thread raise `Cancelled` once `check()` is true. None switches it off."""
    _cancel.check = check


def check_cancelled() -> None:
    """A stopping point for long loops: free when nobody asked for cancelling."""
    check = getattr(_cancel, "check", None)
    if check is not None and check():
        raise Cancelled()
