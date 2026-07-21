from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cc_cover.cli import create_parser
from cc_cover.gui_support import (
    GuiOptions,
    command_environment,
    runtime_paths,
    scan_command,
    setup_commands,
    transcribe_command,
)


class GuiSupportTests(unittest.TestCase):
    def test_runtime_paths_use_local_application_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=root / "bundle",
                local_app_data=root / "local",
            )

        self.assertEqual(paths.source_root, (root / "bundle" / "src").resolve())
        self.assertEqual(paths.data_root, (root / "local" / "CC-Cover").resolve())
        self.assertEqual(paths.venv_python.name, "python.exe")

    def test_commands_always_receive_user_selected_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                local_app_data=base / "local",
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

    def test_setup_commands_select_requested_torch_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                local_app_data=base / "local",
            )
            cuda = setup_commands(paths, ["python"], "cuda")
            cpu = setup_commands(paths, ["python"], "cpu")

        self.assertIn("https://download.pytorch.org/whl/cu121", cuda[2])
        self.assertIn("https://download.pytorch.org/whl/cpu", cpu[2])

    def test_subprocess_environment_includes_bundled_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                local_app_data=base / "local",
            )
            environment = command_environment(paths, {"PATH": "value"})

        self.assertEqual(environment["PATH"], "value")
        self.assertTrue(environment["PYTHONPATH"].startswith(str(paths.source_root)))
        self.assertEqual(environment["PYTHONUTF8"], "1")


if __name__ == "__main__":
    unittest.main()
