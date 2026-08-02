from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cc_cover.cli import create_parser
from cc_cover.gui_support import (
    GuiSettings,
    GuiOptions,
    SettingsError,
    SingleInstanceLock,
    apply_data_root,
    command_environment,
    configured_data_root,
    default_data_root,
    ensure_data_root,
    environment_check_command,
    environment_status_label,
    focus_existing_window,
    is_writable,
    load_gui_settings,
    read_settings,
    resolve_data_root,
    runtime_paths,
    scan_command,
    setup_commands,
    settings_file,
    save_gui_settings,
    transcribe_command,
    write_settings,
)


class GuiSupportTests(unittest.TestCase):
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
        self.assertEqual(
            paths.model_cache, (root / "data" / "model-cache").resolve()
        )
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

    def test_commands_always_receive_user_selected_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            selected = base / "selected"
            options = GuiOptions(
                device="cpu",
                hash_videos=True,
                ffmpeg=base / "ffmpeg.exe",
            )
            scan = scan_command(paths, selected, options)
            transcribe = transcribe_command(paths, selected, options)

        create_parser().parse_args(scan[3:])
        create_parser().parse_args(transcribe[3:])
        self.assertIn(str(selected), scan)
        self.assertIn(str(selected), transcribe)
        self.assertIn("--no-hash-videos", scan)
        self.assertNotIn("--no-hash-videos", transcribe)
        self.assertNotIn("--device", scan)
        self.assertNotIn("--ffmpeg", scan)
        self.assertIn("--device", transcribe)
        self.assertIn("--ffmpeg", transcribe)

    def test_transcribe_without_hash_protection_passes_no_hash_videos(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            selected = base / "selected"
            options = GuiOptions(
                device="cpu",
                hash_videos=False,
                ffmpeg=base / "ffmpeg.exe",
            )
            scan = scan_command(paths, selected, options)
            transcribe = transcribe_command(paths, selected, options)

        self.assertIn("--no-hash-videos", scan)
        self.assertIn("--no-hash-videos", transcribe)

    def test_commands_never_pass_removed_discovery_toggles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            selected = base / "selected"
            options = GuiOptions(device="cpu")
            scan = scan_command(paths, selected, options)
            transcribe = transcribe_command(paths, selected, options)

        for command in (scan, transcribe):
            self.assertNotIn("--include-missing", command)
            self.assertNotIn("--include-whitespace-only", command)
        create_parser().parse_args(scan[3:])
        create_parser().parse_args(transcribe[3:])

    def test_setup_commands_select_requested_torch_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            cuda = setup_commands(paths, ["python"], "cuda")
            cpu = setup_commands(paths, ["python"], "cpu")

        self.assertIn("uninstall", cuda[2])
        self.assertIn("-y", cuda[2])
        self.assertIn("torch", cuda[2])
        self.assertIn("torchaudio", cuda[2])
        self.assertIn("--force-reinstall", cuda[3])
        self.assertIn("--no-cache-dir", cuda[3])
        self.assertIn("https://download.pytorch.org/whl/cu121", cuda[3])
        self.assertIn("--force-reinstall", cpu[3])
        self.assertIn("https://download.pytorch.org/whl/cpu", cpu[3])

    def test_environment_check_requires_cuda_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            cuda = environment_check_command(paths, "cuda")
            cpu = environment_check_command(paths, "cpu")

        self.assertIn("require_cuda = True", cuda[-1])
        self.assertIn("require_cuda = False", cpu[-1])
        self.assertIn("CTranslate2 CUDA devices", cuda[-1])
        self.assertEqual(
            environment_status_label("cuda", "CUDA: True"),
            "运行环境已就绪（GPU）",
        )
        self.assertEqual(
            environment_status_label("cpu", "CUDA: False"),
            "运行环境已就绪（CPU）",
        )

    def test_subprocess_environment_includes_bundled_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            environment = command_environment(paths, {"PATH": "value"})

        self.assertEqual(environment["PATH"], "value")
        self.assertTrue(environment["PYTHONPATH"].startswith(str(paths.source_root)))
        self.assertEqual(environment["PYTHONUTF8"], "1")


class DataRootSettingsTests(unittest.TestCase):
    def test_settings_file_lives_inside_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(settings_file(root), (root / "settings.json").resolve())

    def test_read_settings_returns_empty_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(read_settings(Path(temporary)), {})

    def test_read_settings_treats_unaddressable_root_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocker = root / "blocker"
            blocker.write_bytes(b"")
            self.assertEqual(read_settings(blocker / "app"), {})

    def test_write_settings_round_trips_utf8_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"data_root": "D:/数据", "device": "cpu"})
            self.assertEqual(
                read_settings(root),
                {"data_root": "D:/数据", "device": "cpu"},
            )
            self.assertEqual(settings_file(root).is_file(), True)

    def test_read_settings_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_file(root).write_text("{not json", encoding="utf-8")
            with self.assertRaises(SettingsError):
                read_settings(root)

    def test_read_settings_rejects_non_object_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_file(root).write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(SettingsError):
                read_settings(root)


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


class GuiPreferencesTests(unittest.TestCase):
    def test_load_returns_defaults_for_missing_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = load_gui_settings(Path(temporary))

        self.assertEqual(
            settings,
            GuiSettings(
                scan_path="",
                device="auto",
                accelerator="cuda",
                ffmpeg="",
                hash_videos=True,
            ),
        )

    def test_save_then_load_round_trips_user_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = GuiSettings(
                scan_path="D:/视频",
                device="cpu",
                accelerator="cpu",
                ffmpeg="C:/ffmpeg.exe",
                hash_videos=False,
            )

            save_gui_settings(root, expected)
            loaded = load_gui_settings(root)

            self.assertEqual(loaded, expected)
            self.assertIn(
                "D:/视频", settings_file(root).read_text(encoding="utf-8")
            )

    def test_load_fills_defaults_for_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"device": "cpu", "unknown": "ignored"})

            settings = load_gui_settings(root)

        self.assertEqual(
            settings,
            GuiSettings(
                scan_path="",
                device="cpu",
                accelerator="cuda",
                ffmpeg="",
                hash_videos=True,
            ),
        )

    def test_load_falls_back_to_defaults_for_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(
                root,
                {
                    "scan_path": None,
                    "device": "tpu",
                    "accelerator": "gpu",
                    "ffmpeg": None,
                    "hash_videos": "yes",
                },
            )

            settings = load_gui_settings(root)

        self.assertEqual(
            settings,
            GuiSettings(
                scan_path="",
                device="auto",
                accelerator="cuda",
                ffmpeg="",
                hash_videos=True,
            ),
        )

    def test_save_preserves_existing_settings_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"data_root": "D:/data"})

            save_gui_settings(
                root,
                GuiSettings(
                    scan_path="C:/videos",
                    device="cuda",
                    accelerator="cuda",
                    ffmpeg="",
                    hash_videos=True,
                ),
            )

            stored = read_settings(root)
        self.assertEqual(stored["data_root"], "D:/data")
        self.assertEqual(stored["scan_path"], "C:/videos")
        self.assertEqual(stored["device"], "cuda")

    def test_load_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_file(root).write_text("{not json", encoding="utf-8")
            with self.assertRaises(SettingsError):
                load_gui_settings(root)

    def test_save_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_file(root).write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(SettingsError):
                save_gui_settings(root, GuiSettings())


class SingleInstanceLockTests(unittest.TestCase):
    def test_mutex_name_is_stable_for_same_data_root(self) -> None:
        first = SingleInstanceLock(Path("E:/CC-Cover/Data"))
        second = SingleInstanceLock(Path("e:/cc-cover/data"))
        third = SingleInstanceLock(Path("D:/other"))
        self.assertEqual(first._mutex_name(), second._mutex_name())
        self.assertNotEqual(first._mutex_name(), third._mutex_name())
        self.assertTrue(first._mutex_name().startswith("CC-Cover-"))

    def test_second_acquire_is_rejected_until_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = SingleInstanceLock(root)
            self.assertTrue(first.acquire())
            try:
                self.assertFalse(SingleInstanceLock(root).acquire())
            finally:
                first.release()
            self.assertTrue(SingleInstanceLock(root).acquire())

    def test_context_manager_releases_lock_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with SingleInstanceLock(root) as first:
                self.assertTrue(first.acquired)
                self.assertFalse(SingleInstanceLock(root).acquire())
            self.assertTrue(SingleInstanceLock(root).acquire())

    def test_second_process_is_rejected_until_first_exits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_code = (
                "import sys, time\n"
                "from pathlib import Path\n"
                "from cc_cover.gui_support import SingleInstanceLock\n"
                "lock = SingleInstanceLock(Path(sys.argv[1]))\n"
                "print(lock.acquire(), flush=True)\n"
                "time.sleep(10)\n"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", child_code, str(root)],
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                first_line = child.stdout.readline()
                self.assertEqual(first_line.strip(), "True")
                self.assertFalse(SingleInstanceLock(root).acquire())
            finally:
                child.terminate()
                child.wait(timeout=10)
                child.stdout.close()
            self.assertTrue(SingleInstanceLock(root).acquire())

    def test_focus_existing_window_never_raises(self) -> None:
        self.assertIsInstance(focus_existing_window(), bool)
        self.assertIsInstance(focus_existing_window(""), bool)


@unittest.skipIf(os.name == "nt", "Windows 使用命名互斥体，文件锁回退在非 Windows 平台验证")
class SingleInstanceLockFileFallbackTests(unittest.TestCase):
    def test_file_lock_rejects_live_second_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = SingleInstanceLock(root)
            self.assertTrue(first._acquire_file())
            try:
                self.assertFalse(SingleInstanceLock(root)._acquire_file())
            finally:
                first._release_file()
            self.assertTrue(SingleInstanceLock(root)._acquire_file())

    def test_file_lock_cleans_up_stale_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "instance.lock"
            dead = subprocess.Popen(
                [sys.executable, "-c", "pass"], shell=False
            )
            dead.wait()
            lock_path.write_text(f"{dead.pid}\n", encoding="ascii")

            self.assertTrue(SingleInstanceLock(root)._acquire_file())
            self.assertEqual(
                lock_path.read_text(encoding="ascii").strip(),
                str(os.getpid()),
            )

    def test_file_lock_keeps_corrupt_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "instance.lock"
            lock_path.write_text("not-a-pid\n", encoding="ascii")
            self.assertFalse(SingleInstanceLock(root)._acquire_file())


if __name__ == "__main__":
    unittest.main()
