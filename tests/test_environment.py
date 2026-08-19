from __future__ import annotations

import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from cc_cover.gui.background import DoneOutcome, ErrorOutcome, IdleOutcome
from cc_cover.gui.data_root import runtime_paths, RuntimePaths
from cc_cover.gui.environment import (
    ASR_DEPENDENCIES,
    NOT_INSTALLED_STATUS_LABEL,
    TORCH_VERSION,
    EnvironmentController,
    environment_check_command,
    environment_status_label,
    needs_force_reinstall_prompt,
    outdated_packages,
    parse_installed_versions,
    reinstall_scope,
    setup_commands,
)
from cc_cover.gui.storage import DiskCheck


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


class _FakeTasks:
    """TaskRunner 的假实现：记录调用、返回预设输出，不碰真子进程。"""

    def __init__(self, *, output: str = "", raises: Exception | None = None) -> None:
        self.output = output
        self.raises = raises
        self.cancel_requested = False
        self.capture_calls: list[list[str]] = []
        self.streaming_calls: list[list[str]] = []

    def run_capture(self, command: list[str], *, hf_token: str = "") -> str:
        self.capture_calls.append(command)
        if self.raises is not None:
            raise self.raises
        return self.output

    def run_streaming(self, command: list[str], *, hf_token: str = "") -> str:
        self.streaming_calls.append(command)
        return self.output


class _FakeDialogs:
    """DialogHost 的假实现：记录调用、返回预设答案，不碰真 Tk。"""

    def __init__(self, *, confirm_result: bool = True) -> None:
        self.confirm_result = confirm_result
        self.errors: list[str] = []
        self.confirm_calls: list[tuple[DiskCheck, int]] = []

    def show_disk_precheck_error(self, message: str) -> None:
        self.errors.append(message)

    def confirm_low_disk_space(self, check: DiskCheck, runs_bytes: int) -> bool:
        self.confirm_calls.append((check, runs_bytes))
        return self.confirm_result


_MATCHING_VERSIONS_OUTPUT = (
    "环境检查通过\n"
    f"VERSIONS: torch={TORCH_VERSION} torchaudio={TORCH_VERSION} "
    "imageio-ffmpeg=0.9.0 funasr=1.3.16 modelscope=1.38.1 "
    "faster-whisper=1.2.1 ctranslate2=4.8.1 numpy=1.27.0 soundfile=0.12.5\n"
)


class EnvironmentControllerTests(unittest.TestCase):
    def _paths(self, temp: Path, *, with_venv: bool) -> RuntimePaths:
        base = Path(temp)
        paths = runtime_paths(
            frozen=True, bundle_root=base / "bundle", data_root=base / "data"
        )
        if with_venv:
            paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
            paths.venv_python.write_text("")
        return paths

    def _controller(
        self,
        paths: RuntimePaths,
        *,
        tasks: _FakeTasks | None = None,
        dialogs: _FakeDialogs | None = None,
        device_auto: bool = False,
        device: str = "cpu",
    ) -> EnvironmentController:
        return EnvironmentController(
            paths,
            queue.Queue(),
            tasks if tasks is not None else _FakeTasks(),
            dialogs if dialogs is not None else _FakeDialogs(),
            is_device_auto=lambda: device_auto,
            current_device=lambda: device,
            hf_token=lambda: "",
        )

    def _drain(self, controller: EnvironmentController) -> list:
        items = []
        while True:
            try:
                items.append(controller.events.get_nowait())
            except queue.Empty:
                return items

    def test_recheck_prompt_starts_cleared_and_toggles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(self._paths(Path(temporary), with_venv=True))

        self.assertFalse(controller._prompt_recheck)
        controller.request_recheck_prompt()
        self.assertTrue(controller._prompt_recheck)
        controller.clear_recheck_prompt()
        self.assertFalse(controller._prompt_recheck)

    def test_precheck_setup_true_when_disk_sufficient_no_dialog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dialogs = _FakeDialogs()
            controller = self._controller(
                self._paths(Path(temporary), with_venv=True), dialogs=dialogs
            )
            check = DiskCheck(
                target=controller.paths.data_root,
                required_bytes=10,
                free_bytes=10_000,
                sufficient=True,
            )
            with mock.patch(
                "cc_cover.gui.environment.disk_precheck", return_value=check
            ):
                result = controller.precheck_setup()

        self.assertTrue(result)
        self.assertEqual(dialogs.confirm_calls, [])
        self.assertEqual(dialogs.errors, [])

    def test_precheck_setup_asks_dialog_when_insufficient_and_returns_its_answer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dialogs = _FakeDialogs(confirm_result=False)
            controller = self._controller(
                self._paths(Path(temporary), with_venv=True), dialogs=dialogs
            )
            check = DiskCheck(
                target=controller.paths.data_root,
                required_bytes=10_000,
                free_bytes=10,
                sufficient=False,
            )
            with mock.patch(
                "cc_cover.gui.environment.disk_precheck", return_value=check
            ):
                result = controller.precheck_setup()

        self.assertFalse(result)
        self.assertEqual(len(dialogs.confirm_calls), 1)
        self.assertEqual(dialogs.confirm_calls[0][0], check)

    def test_precheck_setup_reports_error_and_returns_false_on_oserror(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dialogs = _FakeDialogs()
            controller = self._controller(
                self._paths(Path(temporary), with_venv=True), dialogs=dialogs
            )
            with mock.patch(
                "cc_cover.gui.environment.disk_precheck",
                side_effect=OSError("磁盘读取失败"),
            ):
                result = controller.precheck_setup()

        self.assertFalse(result)
        self.assertEqual(dialogs.errors, ["磁盘读取失败"])

    def test_check_worker_reports_not_installed_when_venv_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(
                self._paths(Path(temporary), with_venv=False)
            )
            controller.build_check_worker()()
            events = self._drain(controller)

        self.assertIn(
            ("environment", (False, NOT_INSTALLED_STATUS_LABEL)), events
        )
        self.assertIn(IdleOutcome("请先安装运行环境"), events)
        self.assertEqual(controller.tasks.capture_calls, [])

    def test_check_worker_reports_ready_and_clears_recheck_prompt_on_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = _FakeTasks(output=_MATCHING_VERSIONS_OUTPUT)
            controller = self._controller(
                self._paths(Path(temporary), with_venv=True), tasks=tasks
            )
            controller.request_recheck_prompt()

            controller.build_check_worker()()
            events = self._drain(controller)

        self.assertIn(
            ("environment", (True, "运行环境已就绪（CPU）")), events
        )
        self.assertIn(IdleOutcome("就绪"), events)
        self.assertFalse(controller._prompt_recheck)

    def test_check_worker_posts_device_check_failed_when_prompt_requested(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = _FakeTasks(raises=RuntimeError("boom"))
            controller = self._controller(
                self._paths(Path(temporary), with_venv=True), tasks=tasks
            )
            controller.request_recheck_prompt()

            controller.build_check_worker()()
            events = self._drain(controller)

        self.assertIn(("device_check_failed", "需要安装或修复"), events)
        self.assertFalse(controller._prompt_recheck)
        self.assertFalse(any(isinstance(e, ErrorOutcome) for e in events))

    def test_check_worker_silent_on_error_when_prompt_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = _FakeTasks(raises=RuntimeError("boom"))
            controller = self._controller(
                self._paths(Path(temporary), with_venv=True), tasks=tasks
            )

            controller.build_check_worker()()
            events = self._drain(controller)

        self.assertFalse(
            any(item[0] == "device_check_failed" for item in events if isinstance(item, tuple))
        )

    def test_setup_worker_full_install_posts_install_events_and_done(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary), with_venv=False)
            expected_commands = setup_commands(paths, [sys.executable], "cpu")
            tasks = _FakeTasks(output="环境检查通过\n")
            controller = self._controller(paths, tasks=tasks)

            with mock.patch(
                "cc_cover.gui.environment.python_candidates",
                return_value=[[sys.executable]],
            ):
                controller.build_setup_worker()()
            events = self._drain(controller)

        self.assertEqual(len(tasks.streaming_calls), len(expected_commands))
        install_start = next(item for item in events if item[0] == "install_start")
        self.assertEqual(install_start[1][1], len(expected_commands))
        done_events = [e for e in events if isinstance(e, DoneOutcome)]
        self.assertEqual(len(done_events), 1)
        self.assertEqual(done_events[0].title, "安装完成")

    def test_confirm_force_reinstall_returns_true_from_result_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(self._paths(Path(temporary), with_venv=True))

        def responder() -> None:
            kind, result_queue = controller.events.get(timeout=1)
            self.assertEqual(kind, "confirm_force_reinstall")
            result_queue.put(True)

        thread = threading.Thread(target=responder)
        thread.start()
        result = controller._confirm_force_reinstall()
        thread.join(timeout=1)

        self.assertTrue(result)

    def test_confirm_force_reinstall_returns_false_when_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = _FakeTasks()
            tasks.cancel_requested = True
            controller = self._controller(
                self._paths(Path(temporary), with_venv=True), tasks=tasks
            )

        def responder() -> None:
            _, result_queue = controller.events.get(timeout=1)
            result_queue.put(True)

        thread = threading.Thread(target=responder)
        thread.start()
        result = controller._confirm_force_reinstall()
        thread.join(timeout=1)

        self.assertFalse(result)

    def test_find_base_python_returns_first_matching_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(self._paths(Path(temporary), with_venv=True))
            with mock.patch(
                "cc_cover.gui.environment.python_candidates",
                return_value=[[sys.executable]],
            ):
                result = controller._find_base_python()

        self.assertEqual(result, [sys.executable])

    def test_find_base_python_raises_when_no_candidate_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(self._paths(Path(temporary), with_venv=True))
            with mock.patch(
                "cc_cover.gui.environment.python_candidates", return_value=[]
            ):
                with self.assertRaises(RuntimeError):
                    controller._find_base_python()

    def test_detect_and_report_skips_when_not_auto(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = _FakeTasks(output="cuda")
            controller = self._controller(
                self._paths(Path(temporary), with_venv=True),
                tasks=tasks,
                device_auto=False,
            )
            controller._detect_and_report()
            events = self._drain(controller)

        self.assertEqual(events, [])
        self.assertEqual(tasks.capture_calls, [])

    def test_detect_and_report_posts_detected_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = _FakeTasks(output="cuda")
            controller = self._controller(
                self._paths(Path(temporary), with_venv=True),
                tasks=tasks,
                device_auto=True,
            )
            controller._detect_and_report()
            events = self._drain(controller)

        self.assertIn(("device_detected", "cuda"), events)


if __name__ == "__main__":
    unittest.main()
