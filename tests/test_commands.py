from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cc_cover.core.cli import create_parser
from cc_cover.gui.commands import command_environment, scan_command, transcribe_command
from cc_cover.gui.data_root import runtime_paths
from cc_cover.gui.settings import GuiSettings


class CommandConstructionTests(unittest.TestCase):
    def test_commands_always_receive_user_selected_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            selected = base / "selected"
            settings = GuiSettings(
                device="cpu",
                hash_videos=True,
                ffmpeg=str(base / "ffmpeg.exe"),
            )
            scan = scan_command(paths, selected, settings)
            transcribe = transcribe_command(paths, selected, settings)

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
            settings = GuiSettings(
                device="cpu",
                hash_videos=False,
                ffmpeg=str(base / "ffmpeg.exe"),
            )
            scan = scan_command(paths, selected, settings)
            transcribe = transcribe_command(paths, selected, settings)

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
            settings = GuiSettings(device="cpu")
            scan = scan_command(paths, selected, settings)
            transcribe = transcribe_command(paths, selected, settings)

        for command in (scan, transcribe):
            self.assertNotIn("--include-missing", command)
            self.assertNotIn("--include-whitespace-only", command)
        create_parser().parse_args(scan[3:])
        create_parser().parse_args(transcribe[3:])

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

    def test_subprocess_environment_sets_hf_token_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True, bundle_root=base / "bundle", data_root=base / "data"
            )
            environment = command_environment(
                paths, {"PATH": "value"}, hf_token="hf_abc123"
            )

        self.assertEqual(environment["HF_TOKEN"], "hf_abc123")

    def test_subprocess_environment_leaves_inherited_hf_token_when_blank(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True, bundle_root=base / "bundle", data_root=base / "data"
            )
            environment = command_environment(
                paths, {"PATH": "value", "HF_TOKEN": "system-level-token"}
            )

        # 空字符串（默认值）不覆盖已经存在于继承环境里的 HF_TOKEN——
        # GUI 设置项之前就能用的"系统环境变量"临时解法不受影响。
        self.assertEqual(environment["HF_TOKEN"], "system-level-token")

    def test_transcribe_command_includes_exclude_file_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "local",
            )
            selected = base / "selected"
            exclude_file = base / "excluded.json"
            settings = GuiSettings(device="cpu")
            scan = scan_command(paths, selected, settings)
            transcribe = transcribe_command(
                paths,
                selected,
                settings,
                exclude_file=exclude_file,
            )

        self.assertNotIn("--exclude", scan)
        self.assertIn("--exclude", transcribe)
        self.assertIn(str(exclude_file), transcribe)
        create_parser().parse_args(transcribe[3:])


if __name__ == "__main__":
    unittest.main()
