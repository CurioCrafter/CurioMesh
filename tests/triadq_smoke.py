from __future__ import annotations

# ruff: noqa: E402

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triadq import MeshData, RemeshOptions, read_obj, remesh_mesh, write_obj


def triangulated_grid(name: str, size: int = 6, material_split: bool = False, noise: float = 0.0) -> MeshData:
    verts = []
    for y in range(size + 1):
        for x in range(size + 1):
            z = noise * ((x + y) % 2)
            verts.append((x / size - 0.5, y / size - 0.5, z))
    faces = []
    materials = []
    uvs = []
    uv_lookup = {}
    face_uvs = []
    for y in range(size):
        for x in range(size):
            a = y * (size + 1) + x
            b = a + 1
            c = a + size + 2
            d = a + size + 1
            mat = "left" if material_split and x < size // 2 else "right" if material_split else "grid"
            for tri in ((a, b, c), (a, c, d)):
                faces.append(tri)
                materials.append(mat)
                uv_face = []
                for vi in tri:
                    vx = vi % (size + 1)
                    vy = vi // (size + 1)
                    key = (vx / size, vy / size)
                    if key not in uv_lookup:
                        uv_lookup[key] = len(uvs)
                        uvs.append(key)
                    uv_face.append(uv_lookup[key])
                face_uvs.append(tuple(uv_face))
    return MeshData(
        vertices=np.asarray(verts, dtype=float),
        faces=faces,
        face_materials=materials,
        uvs=np.asarray(uvs, dtype=float),
        face_uvs=face_uvs,
        name=name,
    )


def cube_mesh() -> MeshData:
    verts = np.asarray(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ],
        dtype=float,
    )
    quads = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    faces = []
    materials = []
    for index, quad in enumerate(quads):
        faces.append((quad[0], quad[1], quad[2]))
        faces.append((quad[0], quad[2], quad[3]))
        materials.extend([f"side_{index}", f"side_{index}"])
    return MeshData(vertices=verts, faces=faces, face_materials=materials, name="cube")


def non_manifold_stress() -> MeshData:
    verts = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
        ],
        dtype=float,
    )
    faces = [(0, 1, 2), (1, 0, 3), (0, 1, 4)]
    return MeshData(vertices=verts, faces=faces, face_materials=["a", "b", "c"], name="nonmanifold")


def assert_report(name: str, mesh: MeshData, *, min_quad_ratio: float) -> MeshData:
    out, report = remesh_mesh(
        mesh,
        RemeshOptions(
            target_faces=max(1, mesh.face_count // 2),
            mode="AUTO",
            seed_count=4,
            feature_angle_deg=35.0,
            force_quads=False,
        ),
    )
    if not report.success:
        raise AssertionError(f"{name}: remesh failed: {report.message}")
    if not out.faces:
        raise AssertionError(f"{name}: output has no faces")
    if report.quad_ratio < min_quad_ratio:
        raise AssertionError(f"{name}: quad ratio {report.quad_ratio:.3f} < {min_quad_ratio:.3f}")
    if report.face_count_error < 0:
        raise AssertionError(f"{name}: invalid target error")
    for face in out.faces:
        if any(index < 0 or index >= out.vertex_count for index in face):
            raise AssertionError(f"{name}: invalid face index {face}")
    print(
        f"PASS {name}: faces={report.output_faces} quad_ratio={report.quad_ratio:.3f} "
        f"mode={report.mode} seed={report.selected_seed}"
    )
    return out


def cli_round_trip(mesh: MeshData) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "input.obj"
        dst = root / "output.obj"
        report_path = root / "report.json"
        write_obj(mesh, src)
        cmd = [
            sys.executable,
            "-m",
            "triadq",
            str(src),
            str(dst),
            "--target-faces",
            "18",
            "--seed-count",
            "3",
            "--report",
            str(report_path),
        ]
        completed = subprocess.run(cmd, cwd=ROOT, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise AssertionError(f"CLI failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
        out = read_obj(dst)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not out.faces or not payload["success"]:
            raise AssertionError("CLI output/report invalid")
        print(f"PASS cli_round_trip: faces={len(out.faces)} quad_ratio={payload['quad_ratio']:.3f}")


def main() -> None:
    assert_report("cube", cube_mesh(), min_quad_ratio=0.90)
    assert_report("grid", triangulated_grid("grid", size=8), min_quad_ratio=0.80)
    assert_report("noisy_grid", triangulated_grid("noisy", size=8, noise=0.02), min_quad_ratio=0.60)
    assert_report("material_split_grid", triangulated_grid("split", size=8, material_split=True), min_quad_ratio=0.65)
    assert_report("non_manifold", non_manifold_stress(), min_quad_ratio=0.0)
    cli_round_trip(triangulated_grid("cli", size=5, material_split=True))


if __name__ == "__main__":
    main()
