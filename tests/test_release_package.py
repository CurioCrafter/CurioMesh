from __future__ import annotations

import ast
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_release import build_archives, release_metadata  # noqa: E402


def bl_info_version(source: str) -> str:
    tree = ast.parse(source)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "bl_info" for target in statement.targets):
            bl_info = ast.literal_eval(statement.value)
            return ".".join(str(part) for part in bl_info["version"])
    raise AssertionError("bl_info was not found")


def archive_payload(path: Path, prefix: str) -> dict[str, bytes]:
    root = f"{prefix}/" if prefix else ""
    with zipfile.ZipFile(path) as archive:
        return {name.removeprefix(root): archive.read(name) for name in archive.namelist()}


class ReleasePackageTests(unittest.TestCase):
    def test_both_layouts_are_complete_and_clean(self) -> None:
        package_id, version = release_metadata(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            legacy, extension = build_archives(ROOT, Path(temporary))
            self.assertTrue(legacy.is_file())
            self.assertTrue(extension.is_file())
            self.assertEqual(legacy.name, f"{package_id}-{version}-addon.zip")
            self.assertEqual(extension.name, f"{package_id}-{version}-extension.zip")
            with zipfile.ZipFile(legacy) as archive:
                self.assertIn("curiomesh/__init__.py", archive.namelist())
                self.assertIn("curiomesh/triadq/topology.py", archive.namelist())
                self.assertTrue(all(name.startswith("curiomesh/") for name in archive.namelist()))
            with zipfile.ZipFile(extension) as archive:
                self.assertIn("blender_manifest.toml", archive.namelist())
                self.assertIn("triadq/topology.py", archive.namelist())
                self.assertNotIn("curiomesh/__init__.py", archive.namelist())

            legacy_payload = archive_payload(legacy, "curiomesh")
            extension_payload = archive_payload(extension, "")
            self.assertEqual(legacy_payload, extension_payload)
            for payload in (legacy_payload, extension_payload):
                manifest = tomllib.loads(payload["blender_manifest.toml"].decode("utf-8"))
                self.assertEqual(manifest["id"], package_id)
                self.assertEqual(manifest["version"], version)
                self.assertEqual(bl_info_version(payload["__init__.py"].decode("utf-8")), version)

    def test_source_version_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "blender_manifest.toml").write_text(
                'id = "curiomesh"\nversion = "1.2.3"\ntype = "add-on"\n',
                encoding="utf-8",
            )
            (root / "__init__.py").write_text('bl_info = {"version": (1, 2, 4)}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match bl_info"):
                release_metadata(root)


if __name__ == "__main__":
    unittest.main()
