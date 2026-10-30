# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Run the CurioMesh smoke tests on every installed Blender series, one Blender at a time.

Plain Python (run outside Blender):
    python scripts/test_blender_matrix.py [--builds E:\\BlenderBuildsWindows\\stable] [--series 4.2 5.0]
        [--report matrix.md] [--max-ram 90]

Per series: tests/blender_smoke.py, then tests/blender_package_smoke.py on that series'
per-version zip (plus the universal 3.6 add-on zip as a legacy check) inside an isolated
user profile. Prints a pass/fail table and writes it as markdown.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_release import EXTENSION_SERIES, LEGACY_SERIES, build_archives, build_per_version, release_metadata  # noqa: E402

DEFAULT_BUILDS = Path(r"E:\BlenderBuildsWindows\stable")
SKIP = re.compile(r"alpha|beta|candidate|bforartists|texture-?studio", re.I)
FOLDER = re.compile(r"^blender-(\d+)\.(\d+)\.(\d+)-(?:stable|lts)", re.I)
TIMEOUT = 1200


def ram_used_percent() -> float:
    class Status(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
            ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
            ("tpage", ctypes.c_ulonglong), ("apage", ctypes.c_ulonglong),
            ("tvirt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong),
            ("ext", ctypes.c_ulonglong),
        ]

    status = Status()
    status.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return 100.0 * (1.0 - status.avail / status.total)


def wait_for_ram(limit: float) -> None:
    while (used := ram_used_percent()) >= limit:
        print(f"  RAM {used:.0f}% >= {limit:.0f}%, waiting...", flush=True)
        time.sleep(15)


def find_builds(root: Path) -> dict[str, Path]:
    """Newest patch per 'major.minor' series -> blender.exe."""
    best: dict[str, tuple[tuple[int, ...], Path]] = {}
    for folder in root.iterdir():
        match = FOLDER.match(folder.name)
        if not folder.is_dir() or SKIP.search(folder.name) or not match or not (folder / "blender.exe").is_file():
            continue
        numbers = tuple(int(part) for part in match.groups())
        series = f"{numbers[0]}.{numbers[1]}"
        if series not in best or numbers > best[series][0]:
            best[series] = (numbers, folder / "blender.exe")
    return {series: path for series, (_, path) in sorted(best.items(), key=lambda kv: tuple(map(int, kv[0].split("."))))}


def run_blender(exe: Path, args: list[str], log: Path, env: dict[str, str], limit: float) -> tuple[bool, str]:
    wait_for_ram(limit)
    command = [str(exe), "--background", "--factory-startup", "--python-exit-code", "1", *args]
    try:
        with log.open("w", encoding="utf-8", errors="replace") as handle:
            code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        return False, "timeout"
    tail = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return code == 0, (tail[-1][:90] if tail else "")


def isolated_env(base: Path) -> dict[str, str]:
    env = dict(os.environ)
    for name, folder in (("CONFIG", "config"), ("DATAFILES", "datafiles"), ("EXTENSIONS", "extensions"), ("SCRIPTS", "scripts")):
        (base / folder).mkdir(parents=True, exist_ok=True)
        env[f"BLENDER_USER_{name}"] = str(base / folder)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--builds", type=Path, default=DEFAULT_BUILDS)
    parser.add_argument("--series", nargs="*", help="only these series, e.g. 4.2 5.0")
    parser.add_argument("--report", type=Path, default=ROOT / "dist" / "blender_matrix.md")
    parser.add_argument("--max-ram", type=float, default=90.0)
    args = parser.parse_args()

    builds = find_builds(args.builds)
    supported = set(LEGACY_SERIES + EXTENSION_SERIES)
    series_list = [s for s in builds if s in supported and (not args.series or s in args.series)]
    print("Blender builds:", {s: builds[s].parent.name for s in series_list})
    package_id, version = release_metadata(ROOT)

    work = Path(tempfile.mkdtemp(prefix="curiomesh_matrix_"))
    zips = work / "zips"
    build_archives(ROOT, zips)
    build_per_version(ROOT, zips)
    rows = []
    try:
        for series in series_list:
            exe = builds[series]
            print(f"=== Blender {series} ({exe.parent.name})", flush=True)
            env = isolated_env(work / f"profile_{series}_smoke")
            ok, note = run_blender(exe, ["--python", str(ROOT / "tests" / "blender_smoke.py")], work / f"{series}.smoke.log", env, args.max_ram)
            row = {"series": series, "build": exe.parent.name, "smoke": (ok, note)}
            checks = [("legacy", zips / f"{package_id}-{version}-blender-3.6-plus-addon.zip")]
            if series in LEGACY_SERIES:
                checks[0] = ("legacy", zips / f"{package_id}-{version}-blender-{series}-addon.zip")
            else:
                checks.append(("extension", zips / f"{package_id}-{version}-blender-{series}-extension.zip"))
            for layout, archive in checks:
                env = isolated_env(work / f"profile_{series}_{layout}")
                row[layout] = run_blender(
                    exe,
                    ["--python", str(ROOT / "tests" / "blender_package_smoke.py"), "--", "--layout", layout, str(archive)],
                    work / f"{series}.{layout}.log",
                    env,
                    args.max_ram,
                )
            rows.append(row)
            print("  ", {k: v for k, v in row.items() if k in ("smoke", "legacy", "extension")}, flush=True)

        def cell(result: tuple[bool, str] | None) -> str:
            return "n/a" if result is None else ("PASS" if result[0] else f"FAIL ({result[1]})")

        lines = [
            f"# CurioMesh {version} Blender matrix",
            "",
            "| Blender | Build | Smoke | Legacy package | Extension package |",
            "|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['series']} | {row['build']} | {cell(row['smoke'])} | {cell(row.get('legacy'))} | {cell(row.get('extension'))} |"
            )
        text = "\n".join(lines) + "\n"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print("\n" + text + f"(written to {args.report}; logs in {work})")
        failed = any(not r[0] for row in rows for r in (row["smoke"], row.get("legacy"), row.get("extension")) if r)
        return 1 if failed else 0
    finally:
        if not any(not r[0] for row in rows for r in (row["smoke"], row.get("legacy"), row.get("extension")) if r):
            shutil.rmtree(work, ignore_errors=True)  # keep logs only when something failed


if __name__ == "__main__":
    raise SystemExit(main())
