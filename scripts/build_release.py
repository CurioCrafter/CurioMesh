from __future__ import annotations

import argparse
import ast
from pathlib import Path, PurePosixPath
import tomllib
import zipfile


PACKAGE_NAME = "curiomesh"
ROOT_FILES = {
    "__init__.py",
    "atlasq.py",
    "batch.py",
    "bridge.py",
    "live.py",
    "metrics.py",
    "operators.py",
    "overlay.py",
    "textures.py",
    "ui.py",
    "blender_manifest.toml",
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
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
    files = [root / name for name in sorted(ROOT_FILES)]
    files.extend(sorted(path for path in (root / "triadq").rglob("*.py") if path.is_file()))
    files.extend(sorted(path for path in (root / "docs").rglob("*.md") if path.is_file()))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release inputs missing: {missing}")
    return files


def _write_archive(root: Path, output: Path, *, prefix: str) -> None:
    files = source_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = PurePosixPath(path.relative_to(root).as_posix())
            archive.write(path, str(PurePosixPath(prefix) / relative) if prefix else str(relative))
    verify_archive(output, prefix=prefix)


def verify_archive(path: Path, *, prefix: str) -> None:
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
    expected_name = asset_name(package_id, version, minimum, layout)
    if path.name != expected_name:
        raise ValueError(f"Archive filename must be {expected_name!r}, got {path.name!r}")


def build_archives(root: Path, output_dir: Path) -> tuple[Path, Path]:
    package_name, version = release_metadata(root)
    manifest = tomllib.loads((root / "blender_manifest.toml").read_text(encoding="utf-8"))
    legacy_min = _major_minor(_bl_info((root / "__init__.py").read_text(encoding="utf-8"), origin=str(root))["blender"])
    legacy = output_dir / asset_name(package_name, version, legacy_min, "addon")
    extension = output_dir / asset_name(
        package_name, version, _major_minor(manifest["blender_version_min"]), "extension"
    )
    _write_archive(root, legacy, prefix=package_name)
    _write_archive(root, extension, prefix="")
    return legacy, extension


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated CurioMesh release archives.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: repository dist)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve() if args.output else root / "dist"
    for archive in build_archives(root, output):
        print(archive)


if __name__ == "__main__":
    main()
