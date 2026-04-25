from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import MeshData


def _obj_index(raw: str, count: int) -> int:
    value = int(raw)
    if value < 0:
        return count + value
    return value - 1


def read_obj(path: str | Path) -> MeshData:
    src = Path(path)
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[tuple[int, ...]] = []
    face_uvs: list[tuple[int, ...]] = []
    materials: list[str] = []
    current_material = "default"
    saw_uv_faces = False

    for raw_line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "vt" and len(parts) >= 3:
            uvs.append([float(parts[1]), float(parts[2])])
        elif parts[0] == "usemtl" and len(parts) >= 2:
            current_material = parts[1]
        elif parts[0] == "f" and len(parts) >= 4:
            face: list[int] = []
            uv_face: list[int] = []
            for token in parts[1:]:
                values = token.split("/")
                face.append(_obj_index(values[0], len(vertices)))
                if len(values) >= 2 and values[1]:
                    uv_face.append(_obj_index(values[1], len(uvs)))
                    saw_uv_faces = True
                else:
                    uv_face.append(-1)
            faces.append(tuple(face))
            face_uvs.append(tuple(uv_face))
            materials.append(current_material)

    return MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=faces,
        face_materials=materials,
        uvs=np.asarray(uvs, dtype=np.float64) if uvs else None,
        face_uvs=face_uvs if saw_uv_faces else None,
        name=src.stem,
    )


def write_obj(mesh: MeshData, path: str | Path) -> None:
    dst = Path(path)
    lines: list[str] = [
        "# CurioMesh TRIAD-Q Lite OBJ",
        f"o {mesh.name or 'triadq_mesh'}",
    ]
    for vertex in mesh.vertices:
        lines.append(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}")
    if mesh.uvs is not None:
        for uv in mesh.uvs:
            lines.append(f"vt {uv[0]:.9g} {uv[1]:.9g}")

    last_material = None
    for index, face in enumerate(mesh.faces):
        material = mesh.face_materials[index] if index < len(mesh.face_materials) else "default"
        if material != last_material:
            lines.append(f"usemtl {material}")
            last_material = material
        if mesh.face_uvs is not None and index < len(mesh.face_uvs):
            uv_face = mesh.face_uvs[index]
            tokens = []
            for corner, vertex_index in enumerate(face):
                uv_index = uv_face[corner] if corner < len(uv_face) else -1
                if uv_index >= 0:
                    tokens.append(f"{vertex_index + 1}/{uv_index + 1}")
                else:
                    tokens.append(str(vertex_index + 1))
            lines.append("f " + " ".join(tokens))
        else:
            lines.append("f " + " ".join(str(vertex_index + 1) for vertex_index in face))
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
