# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path, PurePosixPath
import tomllib
import zipfile


PACKAGE_NAME = "curiomesh"
# Every top-level *.py module is packed automatically (see source_files).
ROOT_FILES = {
    "blender_manifest.toml",
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
    "THIRD_PARTY_NOTICES.md",
}


def _bl_info(source: str, *, origin: str) -> dict:
    tree = ast.parse(source, filename=origin)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "bl_info" for target in statement.targets):
            continue
        bl_info = ast.literal_eval(statement.value)
        version = bl_info.get("version") if isinstance(bl_info, dict) else None
        if not isinstance(version, tuple) or not version or not all(isinstance(part, int) for part in version):
            raise ValueError(f"{origin}: bl_info.version must be a non-empty integer tuple")
        return bl_info
    raise ValueError(f"{origin}: bl_info assignment was not found")


def _bl_info_version(source: str, *, origin: str) -> str:
    return ".".join(str(part) for part in _bl_info(source, origin=origin)["version"])


def _major_minor(version: str | tuple) -> str:
    parts = version.split(".") if isinstance(version, str) else [str(p) for p in version]
    return ".".join(parts[:2])


def asset_name(package_id: str, version: str, minimum: str, layout: str) -> str:
    """layout is 'addon' (legacy zip) or 'extension'; minimum is the lowest Blender major.minor."""
    return f"{package_id}-{version}-blender-{minimum}-plus-{layout}.zip"


def _validated_metadata(manifest_source: str, init_source: str, *, origin: str) -> tuple[str, str]:
    manifest = tomllib.loads(manifest_source)
    package_id = manifest.get("id")
    version = manifest.get("version")
    if package_id != PACKAGE_NAME:
        raise ValueError(f"{origin}: manifest id must be {PACKAGE_NAME!r}, got {package_id!r}")
    if manifest.get("type") != "add-on":
        raise ValueError(f"{origin}: manifest type must be 'add-on'")
    if not isinstance(version, str) or not version:
        raise ValueError(f"{origin}: manifest version must be a non-empty string")
    bl_info_version = _bl_info_version(init_source, origin=f"{origin}/__init__.py")
    if bl_info_version != version:
        raise ValueError(
            f"{origin}: blender_manifest.toml version {version!r} does not match bl_info {bl_info_version!r}"
        )
    return package_id, version


def release_metadata(root: Path) -> tuple[str, str]:
    package_id, version = _validated_metadata(
        (root / "blender_manifest.toml").read_text(encoding="utf-8"),
        (root / "__init__.py").read_text(encoding="utf-8"),
        origin=str(root),
    )
    project_file = root / "pyproject.toml"
    if project_file.is_file():
        project_version = tomllib.loads(project_file.read_text(encoding="utf-8")).get("project", {}).get("version")
        if project_version != version:
            raise ValueError(
                f"{project_file}: project version {project_version!r} does not match add-on version {version!r}"
            )
    return package_id, version


def source_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*.py")) + [root / name for name in sorted(ROOT_FILES)]
    files.extend(sorted(path for path in (root / "triadq").rglob("*.py") if path.is_file()))
    files.append(root / "triadq" / "patterns.json")
    files.append(root / "triadq" / "cases_table.json")
    files.extend(sorted(path for path in (root / "docs").rglob("*.md") if path.is_file()))
    files.extend(sorted((root / "icons").glob("*.png")))  # the panel's own icons (icons.py)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release inputs missing: {missing}")
    return files


# Prebuilt CurioMesh Turbo binaries: <native-dir>/<platform tag>/_curiomesh_turbo.{pyd,so}
# are packed as _native/<platform tag>/... and picked at runtime by turbo.py.
NATIVE_TAGS = ("win_amd64", "win_arm64", "linux_x86_64", "linux_aarch64", "macos_x86_64", "macos_arm64")
NATIVE_SUFFIXES = (".pyd", ".so")


def native_files(native_dir: Path | None) -> list[tuple[Path, str]]:
    """(source, archive-relative path) for every prebuilt binary; empty without --native-dir."""
    if native_dir is None:
        return []
    found = []
    for path in sorted(native_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(native_dir)
        if (
            len(relative.parts) != 2
            or relative.parts[0] not in NATIVE_TAGS
            or not relative.name.startswith("_curiomesh_turbo")
            or path.suffix not in NATIVE_SUFFIXES
        ):
            raise ValueError(f"Unexpected file in native dir (want <tag>/_curiomesh_turbo.pyd|so): {relative}")
        found.append((path, f"_native/{relative.as_posix()}"))
    if not found:
        raise FileNotFoundError(f"No native binaries in {native_dir}")
    return found


LEGACY_SERIES = ("3.6", "4.0", "4.1")
EXTENSION_SERIES = ("4.2", "4.3", "4.4", "4.5", "5.0", "5.1", "5.2")


def per_version_name(package_id: str, version: str, series: str, layout: str) -> str:
    return f"{package_id}-{version}-blender-{series}-{layout}.zip"


def _next_series(series: str) -> str | None:
    """Blender series after `series` in EXTENSION_SERIES (None for the last one)."""
    i = EXTENSION_SERIES.index(series)
    return EXTENSION_SERIES[i + 1] if i + 1 < len(EXTENSION_SERIES) else None


def _series_overrides(root: Path, series: str, layout: str) -> dict[str, bytes]:
    """Rewritten manifest (extension) or bl_info minimum (legacy) for one Blender series."""
    if layout == "extension":
        text = (root / "blender_manifest.toml").read_text(encoding="utf-8")
        text = re.sub(r"(?m)^blender_version_(min|max)\s*=.*\n?", "", text)
        nxt = _next_series(series)
        limits = f'blender_version_min = "{series}.0"\n'
        if nxt:
            limits += f'blender_version_max = "{nxt}.0"\n'
        head, table, rest = text.partition("\n[")  # top-level keys must stay above the first [table]
        tail = "\n[" + rest if table else ""
        return {"blender_manifest.toml": (head.rstrip("\n") + "\n" + limits + tail).encode("utf-8")}
    major, minor = series.split(".")
    text = (root / "__init__.py").read_text(encoding="utf-8")
    text, n = re.subn(r'("blender"\s*:\s*)\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)', rf"\g<1>({major}, {minor}, 0)", text, count=1)
    if n != 1:
        raise ValueError("bl_info 'blender' minimum not found in __init__.py")
    return {"__init__.py": text.encode("utf-8")}


# extensions.blender.org build: Python only, CC0 data only, no outside programs.
# Rules read 2026-11-02: https://extensions.blender.org/terms-of-service/ (1.2, 3.6, 5.2).
LISTING_NOTE = (
    "> **Python-only build.** This is the extensions.blender.org version. The site allows only Python code, so it has\n"
    "> no Turbo core and is slower on big meshes. Its tidiness score uses CC0 reference meshes only, so the numbers sit\n"
    "> a few points higher than in the GitHub zips. The GitHub zips include the fast native core:\n"
    "> https://github.com/CurioCrafter/CurioMesh/releases/latest\n\n"
)
LISTING_NOTICES = """# Third-party notices (extensions.blender.org build)

CurioMesh is GPL-3.0-or-later. This build is Python only and contains no compiled code.

## Data

- `triadq/patterns.json` holds statistics measured on CC0 / public-domain meshes only (Blender Studio Human Base
  Meshes, MakeHuman base mesh, Keenan Crane's models). Sources are in `docs/PATTERN_SOURCES.md`.
- `triadq/cases_table.json` holds camera-check results measured on Poly Haven models (CC0 1.0,
  https://polyhaven.com). No geometry from them is included.
"""


def listing_overrides(root: Path) -> dict[str, bytes]:
    """Text files rewritten for the listing zip: CC-BY credits, Rust and QuadWild mentions removed, permissions added."""
    manifest = (root / "blender_manifest.toml").read_text(encoding="utf-8")
    manifest = "".join(line for line in manifest.splitlines(True) if "native/src" not in line)
    if "[permissions]" not in manifest:  # the main manifest declares it already; a second table is invalid TOML
        manifest = manifest.rstrip("\n") + '\n\n[permissions]\nfiles = "Save job files and exports next to the .blend"\n'
    readme = (root / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    readme = re.sub(r"- Credits for the CC-BY works.*?Security Bot Rig[^\n]*\n", "", readme, flags=re.S)
    readme = re.sub(r"- The triangle intersection test in the native core.*?THIRD_PARTY_NOTICES\.md\)\.\n", "", readme, flags=re.S)
    readme = re.sub(r" ?[^.\n]*flowtool[^.\n]*\.", "", readme)  # the QuadWild script needs an outside program
    readme = readme.replace("\n## ", "\n" + LISTING_NOTE + "## ", 1)
    sources = (root / "docs" / "PATTERN_SOURCES.md").read_text(encoding="utf-8")
    sources = "".join(line for line in sources.splitlines(True) if "CC-BY" not in line)
    return {
        "blender_manifest.toml": manifest.encode("utf-8"),
        "README.md": readme.encode("utf-8"),
        "THIRD_PARTY_NOTICES.md": LISTING_NOTICES.encode("utf-8"),
        "docs/PATTERN_SOURCES.md": sources.encode("utf-8"),
    }


def build_listing(root: Path, output_dir: Path) -> Path:
    """One zip for extensions.blender.org: no _native, CC0-only patterns.json, nothing that needs outside programs."""
    package_id, version = release_metadata(root)
    out = output_dir / f"{package_id}-{version}-blender-4.2-plus-listing.zip"
    _write_archive(root, out, prefix="", overrides=listing_overrides(root), listing=True)
    return out


def _write_archive(
    root: Path,
    output: Path,
    *,
    prefix: str,
    overrides: dict[str, bytes] | None = None,
    series: str | None = None,
    native: list[tuple[Path, str]] | None = None,
    listing: bool = False,
) -> None:
    overrides = overrides or {}
    files = source_files(root)
    if listing:
        files = [root / "triadq" / "patterns_cc0.json" if f.name == "patterns.json" else f for f in files]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if relative.name == "patterns_cc0.json":
                relative = relative.with_name("patterns.json")
            arcname = str(PurePosixPath(prefix) / relative) if prefix else str(relative)
            if str(relative) in overrides:
                archive.writestr(arcname, overrides[str(relative)])
            else:
                archive.write(path, arcname)
        for path, relative in native or ():
            archive.write(path, f"{prefix}/{relative}" if prefix else relative)
    verify_archive(output, prefix=prefix, series=series, listing=listing)


def verify_archive(path: Path, *, prefix: str, series: str | None = None, listing: bool = False) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        required_prefix = f"{prefix}/" if prefix else ""
        package_id, version = _validated_metadata(
            archive.read(f"{required_prefix}blender_manifest.toml").decode("utf-8"),
            archive.read(f"{required_prefix}__init__.py").decode("utf-8"),
            origin=str(path),
        )
    if len(entries) != len(set(entries)):
        raise ValueError(f"Archive contains duplicate entries: {path}")
    unsafe = [
        entry
        for entry in entries
        if PurePosixPath(entry).is_absolute() or ".." in PurePosixPath(entry).parts or "\\" in entry
    ]
    if unsafe:
        raise ValueError(f"Unsafe archive entries: {unsafe}")
    forbidden = ("__pycache__", ".pyc", ".pyo", ".git/", ".tools/", "dist/")
    bad = [entry for entry in entries if any(token in entry for token in forbidden)]
    if bad:
        raise ValueError(f"Forbidden archive entries: {bad}")
    required = {
        f"{required_prefix}__init__.py",
        f"{required_prefix}blender_manifest.toml",
        f"{required_prefix}triadq/engine.py",
        f"{required_prefix}triadq/topology.py",
    }
    missing = required - set(entries)
    if missing:
        raise ValueError(f"Archive is incomplete: {sorted(missing)}")
    if prefix and any(not entry.startswith(required_prefix) for entry in entries):
        raise ValueError(f"Legacy archive entries must all be rooted at {required_prefix!r}")
    layout = "addon" if prefix else "extension"
    manifest = tomllib.loads(
        zipfile.ZipFile(path).read(f"{required_prefix}blender_manifest.toml").decode("utf-8")
    )
    init_source = zipfile.ZipFile(path).read(f"{required_prefix}__init__.py").decode("utf-8")
    minimum = (
        _major_minor(manifest["blender_version_min"])
        if layout == "extension"
        else _major_minor(_bl_info(init_source, origin=str(path))["blender"])
    )
    if listing:
        banned = [e for e in entries if e.startswith(f"{required_prefix}_native") or "patterns_cc0" in e]
        text = " ".join(zipfile.ZipFile(path).read(e).decode("utf-8", "ignore") for e in entries if e.endswith((".md", ".json", ".toml")))
        if banned or any(w in text for w in ("CC-BY", "Pack Bot", "Security Bot", "Sprite", "flowtool")):
            raise ValueError(f"{path}: listing zip must hold no native code, CC-BY data or outside-program notes: {banned}")
        expected_name = f"{package_id}-{version}-blender-{minimum}-plus-listing.zip"
    elif series is not None:
        if minimum != series:
            raise ValueError(f"{path}: minimum Blender {minimum} does not match series {series}")
        expected_name = per_version_name(package_id, version, series, layout)
        if layout == "extension":
            if manifest["blender_version_min"] != f"{series}.0":
                raise ValueError(f"{path}: bad blender_version_min")
            nxt = _next_series(series)
            if manifest.get("blender_version_max") != (f"{nxt}.0" if nxt else None):
                raise ValueError(f"{path}: bad blender_version_max")
    else:
        expected_name = asset_name(package_id, version, minimum, layout)
    if path.name != expected_name:
        raise ValueError(f"Archive filename must be {expected_name!r}, got {path.name!r}")


def build_archives(root: Path, output_dir: Path, native_dir: Path | None = None) -> tuple[Path, Path]:
    package_name, version = release_metadata(root)
    manifest = tomllib.loads((root / "blender_manifest.toml").read_text(encoding="utf-8"))
    legacy_min = _major_minor(_bl_info((root / "__init__.py").read_text(encoding="utf-8"), origin=str(root))["blender"])
    legacy = output_dir / asset_name(package_name, version, legacy_min, "addon")
    extension = output_dir / asset_name(
        package_name, version, _major_minor(manifest["blender_version_min"]), "extension"
    )
    native = native_files(native_dir)
    _write_archive(root, legacy, prefix=package_name, native=native)
    _write_archive(root, extension, prefix="", native=native)
    return legacy, extension


def build_per_version(root: Path, output_dir: Path, native_dir: Path | None = None) -> list[Path]:
    package_id, version = release_metadata(root)
    native = native_files(native_dir)
    built = []
    for series in LEGACY_SERIES + EXTENSION_SERIES:
        layout = "addon" if series in LEGACY_SERIES else "extension"
        out = output_dir / per_version_name(package_id, version, series, layout)
        _write_archive(
            root,
            out,
            prefix=package_id if layout == "addon" else "",
            overrides=_series_overrides(root, series, layout),
            series=series,
            native=native,
        )
        built.append(out)
    return built


def build_skills(root: Path, output_dir: Path) -> Path:
    """The Claude/agent skills as one zip: unzip it into ~/.claude/skills (or an agent's skills folder). Not part of any add-on zip."""
    _package_id, version = release_metadata(root)
    out = output_dir / f"curiomesh-{version}-agent-skills.zip"
    files = sorted(p for p in (root / "skills").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    if not any(p.name == "SKILL.md" for p in files):
        raise FileNotFoundError("skills/ holds no SKILL.md")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(root / "skills").as_posix())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated CurioMesh release archives.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: repository dist)")
    parser.add_argument("--per-version", action="store_true", help="Also write one zip per Blender series")
    parser.add_argument(
        "--native-dir",
        type=Path,
        default=None,
        help="Folder of prebuilt CurioMesh Turbo binaries (<platform tag>/_curiomesh_turbo.pyd|so) "
        "packed into every zip; without it the zips stay pure Python",
    )
    parser.add_argument(
        "--blender-listing",
        action="store_true",
        help="Build only the extensions.blender.org zip: Python only (no _native), CC0-only patterns, no CC-BY credits",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve() if args.output else root / "dist"
    native_dir = args.native_dir.resolve() if args.native_dir else None
    if args.blender_listing:
        print(build_listing(root, output))
        return
    if native_dir and not os.environ.get("GITHUB_ACTIONS"):
        print("Test build: locally built binaries hold this computer's paths. Publish only zips from the release workflow.", file=sys.stderr)
    archives = list(build_archives(root, output, native_dir))
    archives.append(build_skills(root, output))
    if args.per_version:
        archives += build_per_version(root, output, native_dir)
    for archive in archives:
        print(archive)


if __name__ == "__main__":
    main()
