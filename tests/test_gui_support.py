from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cc_cover.cli import create_parser
from cc_cover.data_root import default_data_root, ensure_data_root, runtime_paths
from cc_cover.gui_support import (
    GuiOptions,
    command_environment,
    detect_device_command,
    device_probe_commands,
    environment_check_command,
    environment_status_label,
    nvidia_probe_command,
    parsed_device,
    parsed_nvidia_probe,
    scan_command,
    setup_commands,
    transcribe_command,
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
            options = GuiOptions(device="cpu")
            scan = scan_command(paths, selected, options)
            transcribe = transcribe_command(
                paths,
                selected,
                options,
                exclude_file=exclude_file,
            )

        self.assertNotIn("--exclude", scan)
        self.assertIn("--exclude", transcribe)
        self.assertIn(str(exclude_file), transcribe)
        create_parser().parse_args(transcribe[3:])


class DeviceDetectionTests(unittest.TestCase):
    def test_detect_command_probes_runtime_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            command = detect_device_command(paths)

        self.assertEqual(command[0], str(paths.venv_python))
        self.assertIn("torch.cuda.is_available()", command[-1])
        self.assertIn("ctranslate2.get_cuda_device_count()", command[-1])
        self.assertIn("print('cuda'", command[-1])

    def test_parsed_device_accepts_cuda_and_cpu_output(self) -> None:
        self.assertEqual(parsed_device("cuda\n"), "cuda")
        self.assertEqual(parsed_device("cpu\n"), "cpu")

    def test_parsed_device_rejects_unknown_output(self) -> None:
        self.assertIsNone(parsed_device(""))
        self.assertIsNone(parsed_device("无法检测"))

    def test_nvidia_probe_command_queries_gpu_list(self) -> None:
        command = nvidia_probe_command()

        self.assertEqual(command[0], "nvidia-smi")
        self.assertIn("--query-gpu=name", command[1:])
        self.assertIn("--format=csv,noheader", command[1:])

    def test_parsed_nvidia_probe_maps_gpu_names_to_cuda(self) -> None:
        self.assertEqual(
            parsed_nvidia_probe("NVIDIA GeForce RTX 4090\n"), "cuda"
        )
        self.assertEqual(parsed_nvidia_probe("  NVIDIA RTX A6000  \n"), "cuda")

    def test_parsed_nvidia_probe_rejects_empty_output(self) -> None:
        self.assertIsNone(parsed_nvidia_probe(""))
        self.assertIsNone(parsed_nvidia_probe("\n"))

    def test_device_probe_commands_prefer_runtime_then_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            commands = device_probe_commands(paths)

        self.assertEqual(commands[0][0], str(paths.venv_python))
        self.assertEqual(commands[1][0], "nvidia-smi")


if __name__ == "__main__":
    unittest.main()
