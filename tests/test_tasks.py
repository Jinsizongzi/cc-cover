from __future__ import annotations

import json
import queue
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cc_cover.core.models import DoneEvent
from cc_cover.gui.background import TaskCancelled
from cc_cover.gui.data_root import runtime_paths
from cc_cover.gui.tasks import (
    TaskRunner,
    should_play_completion_sound,
    terminate_process_tree,
)


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


class TaskRunnerTests(unittest.TestCase):
    def _runner(self, temp_dir: Path) -> TaskRunner:
        data_root = temp_dir / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        paths = runtime_paths(
            frozen=True, bundle_root=temp_dir / "bundle", data_root=data_root
        )
        return TaskRunner(paths, queue.Queue())

    def _completed_process(self, code: int) -> "subprocess.Popen[str]":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [sys.executable, "-c", f"import sys; sys.exit({code})"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        process.wait()
        return process

    def test_run_capture_returns_output_and_clears_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            output = runner.run_capture([sys.executable, "-c", "print('hello-world')"])

        self.assertIn("hello-world", output)
        self.assertIsNone(runner.process)

    def test_run_capture_raises_when_already_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            runner.cancel_requested = True

            with self.assertRaises(TaskCancelled):
                runner.run_capture([sys.executable, "-c", "print('unused')"])

    def test_run_streaming_pushes_lines_to_events_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            output = runner.run_streaming(
                [sys.executable, "-c", "print('line1'); print('line2')"]
            )

        self.assertIn("line1", output)
        self.assertIn("line2", output)
        log_lines: list[str] = []
        while True:
            try:
                kind, value = runner.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                log_lines.append(value)
        self.assertTrue(any("line1" in line for line in log_lines))
        self.assertTrue(any("▶" in line for line in log_lines))

    def test_cancel_terminates_live_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            runner.process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            try:
                runner.cancel()

                deadline = time.monotonic() + 5
                while runner.process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertIsNotNone(runner.process.poll())
                self.assertTrue(runner.cancel_requested)
                self.assertTrue(runner.stop_triggered)
            finally:
                if runner.process.poll() is None:
                    runner.process.kill()
                    runner.process.wait()

    def test_cancel_without_live_process_only_sets_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))

            runner.cancel()

            self.assertTrue(runner.cancel_requested)
            self.assertFalse(runner.stop_triggered)

    def test_reset_clears_cancel_and_stop_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            runner.cancel_requested = True
            runner.stop_triggered = True

            runner.reset()

            self.assertFalse(runner.cancel_requested)
            self.assertFalse(runner.stop_triggered)

    def test_finish_process_returns_output_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            process = self._completed_process(0)

            self.assertEqual(runner._finish_process(process, "输出内容"), "输出内容")

    def test_finish_process_raises_task_cancelled_when_stop_triggered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            runner.stop_triggered = True
            process = self._completed_process(0)

            with self.assertRaises(TaskCancelled):
                runner._finish_process(process, "输出内容")

    def test_finish_process_raises_task_cancelled_when_cancelled_with_nonzero_exit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            runner.cancel_requested = True
            process = self._completed_process(1)

            with self.assertRaises(TaskCancelled):
                runner._finish_process(process, "输出内容")

    def test_finish_process_raises_runtime_error_on_unexplained_nonzero_exit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            process = self._completed_process(1)

            with self.assertRaises(RuntimeError):
                runner._finish_process(process, "没有 done 事件的输出")

    def test_finish_process_tolerates_nonzero_exit_when_done_event_present(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            process = self._completed_process(1)
            done_line = json.dumps(DoneEvent(run_dir="C:/runs/run1").to_dict())

            output = runner._finish_process(process, done_line)

            self.assertEqual(output, done_line)


if __name__ == "__main__":
    unittest.main()
