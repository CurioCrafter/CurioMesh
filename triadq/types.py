from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


Face = tuple[int, ...]
UVFace = tuple[int, ...]


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
