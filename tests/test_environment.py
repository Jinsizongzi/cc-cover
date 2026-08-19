from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cc_cover.gui.data_root import runtime_paths
from cc_cover.gui.environment import (
    ASR_DEPENDENCIES,
    NOT_INSTALLED_STATUS_LABEL,
    TORCH_VERSION,
    environment_check_command,
    environment_status_label,
    needs_force_reinstall_prompt,
    outdated_packages,
    parse_installed_versions,
    reinstall_scope,
    setup_commands,
)


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
