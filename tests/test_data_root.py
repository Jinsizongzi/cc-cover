from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cc_cover.gui.data_root import (
    apply_data_root,
    configured_data_root,
    default_data_root,
    ensure_data_root,
    is_writable,
    resolve_data_root,
    runtime_paths,
)
from cc_cover.gui.settings import read_settings, settings_file, write_settings


class RuntimePathsTests(unittest.TestCase):
    def test_runtime_paths_follow_fixed_data_root_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=root / "bundle",
                data_root=root / "data",
            )

        self.assertEqual(paths.source_root, (root / "bundle" / "src").resolve())
        self.assertEqual(paths.data_root, (root / "data").resolve())
        self.assertEqual(paths.venv_root, (root / "data" / "venv").resolve())
        self.assertEqual(
            paths.venv_python,
            (root / "data" / "venv" / "Scripts" / "python.exe").resolve(),
        )
        self.assertEqual(paths.model_cache, (root / "data" / "model-cache").resolve())
        self.assertEqual(paths.runs_root, (root / "data" / "runs").resolve())
        self.assertEqual(paths.temp_root, (root / "data" / "temp").resolve())

    def test_default_data_root_is_app_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                default_data_root(frozen=True, app_dir=root / "exe"),
                (root / "exe").resolve(),
            )
            self.assertEqual(
                default_data_root(frozen=False, bundle_root=root / "project"),
                (root / "project").resolve(),
            )

    def test_default_data_root_uses_launched_executable_when_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(sys, "executable", str(root / "exe" / "app.exe")):
                self.assertEqual(
                    default_data_root(frozen=True),
                    (root / "exe").resolve(),
                )

    def test_ensure_data_root_creates_fixed_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=root / "bundle",
                data_root=root / "data",
            )
            ensure_data_root(paths)
            for directory in (
                paths.data_root,
                paths.venv_root,
                paths.model_cache,
                paths.runs_root,
                paths.temp_root,
            ):
                self.assertTrue(directory.is_dir())


class DataRootWritabilityTests(unittest.TestCase):
    def test_is_writable_accepts_writable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertTrue(is_writable(Path(temporary)))

    def test_is_writable_creates_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "data"
            self.assertTrue(is_writable(target))
            self.assertTrue(target.is_dir())

    def test_is_writable_rejects_path_under_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocker = root / "blocker"
            blocker.write_bytes(b"")
            self.assertFalse(is_writable(blocker / "data"))


class DataRootResolutionTests(unittest.TestCase):
    def test_configured_data_root_defaults_to_app_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(configured_data_root(root), root.resolve())

    def test_resolve_uses_writable_default_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolution = resolve_data_root(root)
            self.assertEqual(resolution.root, root.resolve())
            self.assertFalse(resolution.needs_choice)

    def test_resolve_requests_choice_when_default_unwritable_and_no_pointer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocker = root / "blocker"
            blocker.write_bytes(b"")
            default = blocker / "app"
            with patch.dict(os.environ, {"LOCALAPPDATA": str(root / "local")}):
                resolution = resolve_data_root(default)
            self.assertEqual(resolution.root, default.resolve())
            self.assertTrue(resolution.needs_choice)

    def test_resolve_uses_custom_data_root_from_default_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom = root / "custom"
            write_settings(root, {"data_root": str(custom)})
            resolution = resolve_data_root(root)
            self.assertEqual(resolution.root, custom.resolve())
            self.assertFalse(resolution.needs_choice)

    def test_resolve_uses_fallback_pointer_when_default_unwritable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocker = root / "blocker"
            blocker.write_bytes(b"")
            default = blocker / "app"
            custom = root / "custom"
            with patch.dict(os.environ, {"LOCALAPPDATA": str(root / "local")}):
                write_settings(root / "local" / "CC-Cover", {"data_root": str(custom)})
                resolution = resolve_data_root(default)
            self.assertEqual(resolution.root, custom.resolve())
            self.assertFalse(resolution.needs_choice)

    def test_resolve_flags_unavailable_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocker = root / "blocker"
            blocker.write_bytes(b"")
            missing = blocker / "missing"
            write_settings(root, {"data_root": str(missing)})
            resolution = resolve_data_root(root)
            self.assertEqual(resolution.root, missing.resolve())
            self.assertTrue(resolution.needs_choice)

    def test_resolve_ignores_relative_data_root_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"data_root": "relative/path"})
            self.assertEqual(configured_data_root(root), root.resolve())
            self.assertFalse(resolve_data_root(root).needs_choice)


class DataRootSwitchTests(unittest.TestCase):
    def test_apply_data_root_copies_settings_and_updates_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_root = root / "old"
            new_root = root / "new"
            write_settings(old_root, {"device": "cpu", "data_root": str(old_root)})

            result = apply_data_root(root, old_root, new_root)

            self.assertEqual(result, new_root.resolve())
            self.assertEqual(
                read_settings(new_root),
                {"device": "cpu", "data_root": str(new_root.resolve())},
            )
            self.assertEqual(
                read_settings(old_root),
                {"device": "cpu", "data_root": str(old_root)},
            )
            self.assertEqual(
                read_settings(root),
                {"data_root": str(new_root.resolve())},
            )

    def test_apply_data_root_from_default_retains_default_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            new_root = root / "new"
            write_settings(root, {"device": "cpu"})

            apply_data_root(root, root, new_root)

            self.assertEqual(
                read_settings(new_root),
                {"device": "cpu", "data_root": str(new_root.resolve())},
            )
            retained = read_settings(root)
            self.assertEqual(retained["device"], "cpu")
            self.assertEqual(retained["data_root"], str(new_root.resolve()))
            self.assertTrue(settings_file(root).is_file())

    def test_apply_data_root_reset_to_default_clears_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_root = root / "old"
            apply_data_root(root, root, old_root)

            result = apply_data_root(root, old_root, root)

            self.assertEqual(result, root.resolve())
            self.assertEqual(read_settings(root), {})
            self.assertEqual(configured_data_root(root), root.resolve())
            self.assertTrue(settings_file(old_root).is_file())

    def test_apply_data_root_uses_fallback_pointer_for_unwritable_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocker = root / "blocker"
            blocker.write_bytes(b"")
            default = blocker / "app"
            new_root = root / "new"
            with patch.dict(os.environ, {"LOCALAPPDATA": str(root / "local")}):
                apply_data_root(default, default, new_root)
                resolution = resolve_data_root(default)
            self.assertEqual(resolution.root, new_root.resolve())
            self.assertFalse(resolution.needs_choice)


if __name__ == "__main__":
    unittest.main()
