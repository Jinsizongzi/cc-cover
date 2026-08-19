from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cc_cover.core.cli import create_parser
from cc_cover.gui.commands import (
    ASR_DEPENDENCIES,
    NOT_INSTALLED_STATUS_LABEL,
    TORCH_VERSION,
    command_environment,
    detect_device_command,
    device_probe_commands,
    environment_check_command,
    environment_status_label,
    needs_force_reinstall_prompt,
    nvidia_probe_command,
    outdated_packages,
    parse_installed_versions,
    parsed_device,
    parsed_nvidia_probe,
    reinstall_scope,
    scan_command,
    setup_commands,
    should_play_completion_sound,
    terminate_process_tree,
    transcribe_command,
)
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
        self.assertEqual(
            environment_status_label("cpu", "CUDA: False", outdated=True),
            "运行环境已就绪（CPU）（有更新可用）",
        )
        self.assertEqual(
            environment_status_label("cuda", "CUDA: True", outdated=False),
            "运行环境已就绪（GPU）",
        )

    def test_not_installed_status_hints_at_change_data_root(self) -> None:
        self.assertIn("尚未安装", NOT_INSTALLED_STATUS_LABEL)
        self.assertIn("更改", NOT_INSTALLED_STATUS_LABEL)

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
        self.assertEqual(parsed_nvidia_probe("NVIDIA GeForce RTX 4090\n"), "cuda")
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


class TerminateProcessTreeTests(unittest.TestCase):
    def test_kills_live_child_process(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        try:
            terminate_process_tree(process)
            deadline = time.monotonic() + 5
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_finished_process_is_a_noop(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        process.wait()

        terminate_process_tree(process)

        self.assertIsNotNone(process.poll())


class CompletionSoundTests(unittest.TestCase):
    def test_plays_sound_when_run_exceeds_five_minutes(self) -> None:
        self.assertTrue(should_play_completion_sound(301.0))

    def test_no_sound_at_exactly_five_minutes(self) -> None:
        self.assertFalse(should_play_completion_sound(300.0))

    def test_no_sound_under_five_minutes(self) -> None:
        self.assertFalse(should_play_completion_sound(299.0))

    def test_no_sound_when_elapsed_unknown(self) -> None:
        self.assertFalse(should_play_completion_sound(None))


class VersionConsistencyTests(unittest.TestCase):
    ALL_MATCHING = {
        "torch": TORCH_VERSION,
        "torchaudio": TORCH_VERSION,
        "imageio-ffmpeg": "0.9.0",
        "funasr": "1.3.16",
        "modelscope": "1.38.1",
        "faster-whisper": "1.2.1",
        "ctranslate2": "4.8.1",
        "numpy": "1.27.0",
        "soundfile": "0.12.5",
    }

    def test_environment_check_command_prints_parsable_version_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True, bundle_root=base / "bundle", data_root=base / "data"
            )
            script = environment_check_command(paths, "cpu")[-1]

        self.assertIn("VERSIONS:", script)
        self.assertIn("importlib.metadata", script)

    def test_parse_installed_versions_reads_the_versions_line(self) -> None:
        output = (
            "环境检查通过\n"
            "VERSIONS: torch=2.5.1 torchaudio=2.5.1 funasr=1.3.15\n"
            "PyTorch: 2.5.1\n"
        )

        self.assertEqual(
            parse_installed_versions(output),
            {"torch": "2.5.1", "torchaudio": "2.5.1", "funasr": "1.3.15"},
        )

    def test_parse_installed_versions_missing_line_returns_empty(self) -> None:
        self.assertEqual(parse_installed_versions("没有版本行"), {})

    def test_outdated_packages_empty_when_everything_matches(self) -> None:
        self.assertEqual(outdated_packages(self.ALL_MATCHING), set())

    def test_outdated_packages_flags_exact_pin_mismatch(self) -> None:
        installed = dict(self.ALL_MATCHING, funasr="1.3.15")

        self.assertEqual(outdated_packages(installed), {"funasr"})

    def test_outdated_packages_flags_range_constraint_violation(self) -> None:
        installed = dict(self.ALL_MATCHING, numpy="2.1.0")

        self.assertEqual(outdated_packages(installed), {"numpy"})

    def test_outdated_packages_pairs_torch_and_torchaudio(self) -> None:
        installed = dict(self.ALL_MATCHING, torch="2.4.0")

        self.assertEqual(outdated_packages(installed), {"torch", "torchaudio"})

    def test_outdated_packages_treats_missing_entry_as_outdated(self) -> None:
        installed = dict(self.ALL_MATCHING)
        del installed["soundfile"]

        self.assertEqual(outdated_packages(installed), {"soundfile"})

    def test_setup_commands_outdated_none_matches_full_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True, bundle_root=base / "bundle", data_root=base / "data"
            )
            paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
            paths.venv_python.write_text("")
            full = setup_commands(paths, ["python"], "cpu")
            explicit_none = setup_commands(paths, ["python"], "cpu", outdated=None)

        self.assertEqual(full, explicit_none)
        self.assertEqual(len(full), 4)
        for spec in ASR_DEPENDENCIES:
            self.assertIn(spec, full[3])

    def test_setup_commands_outdated_subset_skips_torch_and_untouched_deps(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True, bundle_root=base / "bundle", data_root=base / "data"
            )
            paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
            paths.venv_python.write_text("")
            commands = setup_commands(paths, ["python"], "cpu", outdated={"funasr"})

        self.assertEqual(len(commands), 2)
        self.assertNotIn("uninstall", [part for cmd in commands for part in cmd])
        self.assertIn("funasr==1.3.16", commands[1])
        for spec in ASR_DEPENDENCIES:
            if not spec.startswith("funasr"):
                self.assertNotIn(spec, commands[1])

    def test_setup_commands_outdated_empty_set_reinstalls_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True, bundle_root=base / "bundle", data_root=base / "data"
            )
            paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
            paths.venv_python.write_text("")
            commands = setup_commands(paths, ["python"], "cpu", outdated=set())

        self.assertEqual(len(commands), 1)
        self.assertIn("--upgrade", commands[0])

    def test_setup_commands_outdated_torch_only_skips_asr_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True, bundle_root=base / "bundle", data_root=base / "data"
            )
            paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
            paths.venv_python.write_text("")
            commands = setup_commands(paths, ["python"], "cpu", outdated={"torch"})

        self.assertEqual(len(commands), 3)
        self.assertIn("uninstall", commands[1])
        self.assertIn("--force-reinstall", commands[2])

    def test_reinstall_scope_none_means_everything(self) -> None:
        self.assertEqual(reinstall_scope(None), (True, True))

    def test_reinstall_scope_torch_only(self) -> None:
        self.assertEqual(reinstall_scope({"torch"}), (True, False))
        self.assertEqual(reinstall_scope({"torchaudio"}), (True, False))

    def test_reinstall_scope_asr_only(self) -> None:
        self.assertEqual(reinstall_scope({"funasr"}), (False, True))

    def test_reinstall_scope_empty_means_nothing(self) -> None:
        self.assertEqual(reinstall_scope(set()), (False, False))

    def test_reinstall_scope_mixed(self) -> None:
        self.assertEqual(reinstall_scope({"torch", "numpy"}), (True, True))

    def test_needs_force_reinstall_prompt_when_nothing_outdated(self) -> None:
        self.assertTrue(needs_force_reinstall_prompt(set()))

    def test_no_force_reinstall_prompt_when_something_outdated(self) -> None:
        self.assertFalse(needs_force_reinstall_prompt({"funasr"}))
        self.assertFalse(needs_force_reinstall_prompt({"torch", "torchaudio"}))


if __name__ == "__main__":
    unittest.main()
