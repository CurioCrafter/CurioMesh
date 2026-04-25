from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import RemeshOptions, remesh_mesh
from .io import read_obj, write_obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRIAD-Q Lite standalone remesher")
    parser.add_argument("input", type=Path, help="Input OBJ file")
    parser.add_argument("output", type=Path, help="Output OBJ file")
    parser.add_argument("--target-faces", type=int, default=4000)
    parser.add_argument(
        "--mode",
        default="balanced",
        choices=["auto", "balanced", "organic", "patch", "dirty", "texture"],
    )
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--feature-angle", type=float, default=35.0)
    parser.add_argument("--pure-quads", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mesh = read_obj(args.input)
    options = RemeshOptions(
        target_faces=args.target_faces,
        mode=args.mode.upper(),
        seed_count=args.seed_count,
        feature_angle_deg=args.feature_angle,
        force_quads=args.pure_quads,
    )
    result, report = remesh_mesh(mesh, options)
    write_obj(result, args.output)
    payload = report.to_dict()
    if args.report:
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
