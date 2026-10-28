from __future__ import annotations

import addon_utils
import argparse
import ast
import importlib
import os
from pathlib import Path, PurePosixPath
import sys
import re
import traceback
import zipfile

import bpy


def _toml_loads(text: str) -> dict:
    """Flat manifest reader: Blender 3.6 bundles Python 3.10 (no tomllib)."""
    try:
        import tomllib
    except ModuleNotFoundError:
        return dict(re.findall(r'^(\w+)\s*=\s*"([^"]*)"', text, re.M))
    return tomllib.loads(text)


def arguments_after_separator() -> argparse.Namespace:
    if "--" not in sys.argv:
        raise RuntimeError("Pass --layout and the CurioMesh archive after --")
    parser = argparse.ArgumentParser(description="Smoke an installed CurioMesh release archive")
    parser.add_argument("--layout", choices=("legacy", "extension"), required=True)
    parser.add_argument("archive", type=Path)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _bl_info_version(source: str, origin: str) -> str:
    tree = ast.parse(source, filename=origin)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "bl_info" for target in statement.targets):
            bl_info = ast.literal_eval(statement.value)
            version = bl_info.get("version") if isinstance(bl_info, dict) else None
            if not isinstance(version, tuple) or not all(isinstance(part, int) for part in version):
                raise AssertionError(f"{origin}: invalid bl_info.version")
            return ".".join(str(part) for part in version)
    raise AssertionError(f"{origin}: bl_info was not found")


def validate_archive(archive: Path, layout: str) -> tuple[str, str]:
    prefix = "curiomesh/" if layout == "legacy" else ""
    with zipfile.ZipFile(archive) as package:
        entries = package.namelist()
        if len(entries) != len(set(entries)):
            raise AssertionError(f"{archive}: duplicate ZIP entries")
        unsafe = [
            entry
            for entry in entries
            if PurePosixPath(entry).is_absolute() or ".." in PurePosixPath(entry).parts or "\\" in entry
        ]
        if unsafe:
            raise AssertionError(f"{archive}: unsafe ZIP entries: {unsafe}")
        if layout == "legacy" and any(not entry.startswith(prefix) for entry in entries):
            raise AssertionError(f"{archive}: legacy entries must all be rooted at {prefix!r}")
        required = {f"{prefix}__init__.py", f"{prefix}blender_manifest.toml", f"{prefix}triadq/engine.py"}
        missing = required - set(entries)
        if missing:
            raise AssertionError(f"{archive}: missing entries: {sorted(missing)}")
        manifest = _toml_loads(package.read(f"{prefix}blender_manifest.toml").decode("utf-8"))
        init_source = package.read(f"{prefix}__init__.py").decode("utf-8")

    package_id = manifest.get("id")
    version = manifest.get("version")
    if package_id != "curiomesh" or manifest.get("type") != "add-on":
        raise AssertionError(f"{archive}: invalid CurioMesh add-on manifest identity")
    if not isinstance(version, str) or not version:
        raise AssertionError(f"{archive}: invalid manifest version")
    info_version = _bl_info_version(init_source, f"{archive}!/{prefix}__init__.py")
    if info_version != version:
        raise AssertionError(f"{archive}: manifest version {version!r} != bl_info version {info_version!r}")
    archive_kind = "addon" if layout == "legacy" else "extension"
    name_pattern = rf"{package_id}-{re.escape(version)}-blender-\d+\.\d+-plus-{archive_kind}\.zip"
    if not re.fullmatch(name_pattern, archive.name):
        raise AssertionError(f"{archive}: expected filename matching {name_pattern!r}")
    return package_id, version


def require_finished(result: set[str], operation: str) -> None:
    if "FINISHED" not in result:
        raise AssertionError(f"{operation} returned {sorted(result)}")


def isolated_path(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise RuntimeError(f"{variable} must point to an isolated test root")
    path = Path(raw).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_legacy(archive: Path, package_id: str) -> tuple[str, Path]:
    scripts_root = isolated_path("BLENDER_USER_SCRIPTS")
    result = bpy.ops.preferences.addon_install(filepath=str(archive), overwrite=True, target="DEFAULT")
    require_finished(result, "legacy add-on install")
    addon_utils.modules_refresh()
    result = bpy.ops.preferences.addon_enable(module=package_id)
    require_finished(result, "legacy add-on enable")
    if package_id not in bpy.context.preferences.addons:
        raise AssertionError("Legacy CurioMesh add-on is not enabled")
    module = importlib.import_module(package_id)
    installed_root = Path(module.__file__).resolve().parent
    if not installed_root.is_relative_to(scripts_root):
        raise AssertionError(f"Legacy add-on escaped isolated scripts root: {installed_root}")
    return package_id, installed_root


def _isolated_extension_repo() -> tuple[object, Path]:
    extensions_root = isolated_path("BLENDER_USER_EXTENSIONS")
    repo_root = extensions_root / "curiomesh_package_smoke"
    repo_root.mkdir(parents=True, exist_ok=True)
    repos = bpy.context.preferences.extensions.repos
    repo = next((candidate for candidate in repos if candidate.source != "SYSTEM"), None)
    if repo is None:
        result = bpy.ops.preferences.extension_repo_add(type="LOCAL")
        require_finished(result, "extension repository creation")
        repo = next((candidate for candidate in reversed(repos) if candidate.source != "SYSTEM"), None)
    if repo is None:
        raise AssertionError("Blender has no writable extension repository")
    repo.enabled = True
    repo.use_custom_directory = True
    repo.custom_directory = str(repo_root)
    if Path(repo.directory).resolve() != repo_root.resolve():
        raise AssertionError(f"Extension repository did not adopt isolated directory: {repo.directory}")
    return repo, repo_root


def install_extension(archive: Path, package_id: str) -> tuple[str, Path]:
    repo, repo_root = _isolated_extension_repo()
    result = bpy.ops.extensions.package_install_files(
        filepath=str(archive),
        repo=repo.module,
        enable_on_install=True,
    )
    require_finished(result, "extension package install")
    module_name = f"bl_ext.{repo.module}.{package_id}"
    if module_name not in bpy.context.preferences.addons:
        raise AssertionError(f"Extension installer did not enable {module_name}")
    module = importlib.import_module(module_name)
    installed_root = Path(module.__file__).resolve().parent
    if not installed_root.is_relative_to(repo_root):
        raise AssertionError(f"Extension escaped isolated repository: {installed_root}")
    return module_name, installed_root


def assert_installed_runtime(
    module_name: str,
    installed_root: Path,
    package_id: str,
    version: str,
) -> None:
    module = importlib.import_module(module_name)
    manifest_path = installed_root / "blender_manifest.toml"
    if not manifest_path.is_file():
        raise AssertionError(f"Installed package is missing its manifest: {manifest_path}")
    installed_manifest = _toml_loads(manifest_path.read_text(encoding="utf-8"))
    if installed_manifest.get("id") != package_id or installed_manifest.get("version") != version:
        raise AssertionError(
            f"Installed manifest identity/version mismatch: "
            f"id={installed_manifest.get('id')!r} version={installed_manifest.get('version')!r}"
        )
    if hasattr(module, "bl_info"):
        installed_version = ".".join(str(part) for part in module.bl_info["version"])
        if installed_version != version:
            raise AssertionError(f"Installed module version {installed_version!r} != archive version {version!r}")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    result = bpy.ops.curiomesh.analyze()
    require_finished(result, "installed analyze operator")
    settings = bpy.context.scene.curiomesh_settings
    if settings.analysis_faces != 6 or not settings.analysis_profile:
        raise AssertionError(
            f"Installed analysis did not populate expected state: faces={settings.analysis_faces} "
            f"profile={settings.analysis_profile!r}"
        )


def main() -> None:
    args = arguments_after_separator()
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    package_id, version = validate_archive(archive, args.layout)
    if args.layout == "legacy":
        module_name, installed_root = install_legacy(archive, package_id)
    else:
        module_name, installed_root = install_extension(archive, package_id)

    assert_installed_runtime(module_name, installed_root, package_id, version)
    result = bpy.ops.preferences.addon_disable(module=module_name)
    require_finished(result, f"{args.layout} add-on disable")
    if module_name in bpy.context.preferences.addons:
        raise AssertionError(f"{module_name} remained enabled after disable")
    print(
        f"CurioMesh {args.layout} installed-package smoke passed: "
        f"archive={archive.name} module={module_name} root={installed_root}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
