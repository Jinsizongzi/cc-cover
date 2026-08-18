from __future__ import annotations

import unittest
from pathlib import Path

from cc_cover.background import (
    CancelledOutcome,
    DoneOutcome,
    ErrorOutcome,
    IdleOutcome,
    TaskCancelled,
    run_in_background,
)
from cc_cover.progress import FailureInfo


class RunInBackgroundTests(unittest.TestCase):
    def test_normal_return_calls_no_callback(self) -> None:
        calls: list[str] = []

        run_in_background(
            lambda: None,
            on_cancel=lambda exc: calls.append("cancel"),
            on_error=lambda exc: calls.append("error"),
        )

        self.assertEqual(calls, [])

    def test_task_cancelled_calls_on_cancel_with_the_same_exception(self) -> None:
        raised = TaskCancelled("任务已由用户停止。")
        received: list[TaskCancelled] = []

        def run() -> None:
            raise raised

        run_in_background(
            run,
            on_cancel=received.append,
            on_error=lambda exc: self.fail("on_error 不应被调用"),
        )

        self.assertEqual(received, [raised])

    def test_other_exception_calls_on_error_with_the_same_exception(self) -> None:
        raised = RuntimeError("安装组件超时")
        received: list[Exception] = []

        def run() -> None:
            raise raised

        run_in_background(
            run,
            on_cancel=lambda exc: self.fail("on_cancel 不应被调用"),
            on_error=received.append,
        )

        self.assertEqual(received, [raised])


class WorkerOutcomeShapeTests(unittest.TestCase):
    def test_idle_outcome_equality(self) -> None:
        self.assertEqual(IdleOutcome("就绪"), IdleOutcome("就绪"))
        self.assertNotEqual(IdleOutcome("就绪"), IdleOutcome("扫描完成"))

    def test_done_outcome_carries_optional_run_dir(self) -> None:
        without_run_dir = DoneOutcome(
            title="无需处理", message="没有需要处理的候选。", run_dir=None
        )
        with_run_dir = DoneOutcome(
            title="字幕补全完成",
            message="已完成 3 个字幕文件的生成、替换和复核。",
            run_dir=Path("/runs/2026-08-18"),
        )

        self.assertIsNone(without_run_dir.run_dir)
        self.assertEqual(with_run_dir.run_dir, Path("/runs/2026-08-18"))

    def test_cancelled_outcome_wraps_failure_info(self) -> None:
        info = FailureInfo(stage="安装运行环境", reason="用户已停止")

        self.assertEqual(CancelledOutcome(info).info, info)

    def test_error_outcome_wraps_title_and_failure_info(self) -> None:
        info = FailureInfo(stage="安装运行环境", reason="安装组件超时")
        outcome = ErrorOutcome(title="环境安装失败", info=info)

        self.assertEqual(outcome.title, "环境安装失败")
        self.assertEqual(outcome.info, info)

    def test_outcomes_have_no_kind_field_or_serialization_methods(self) -> None:
        for outcome in (
            IdleOutcome("就绪"),
            DoneOutcome(title="t", message="m", run_dir=None),
            CancelledOutcome(FailureInfo(stage="s", reason="r")),
            ErrorOutcome(title="t", info=FailureInfo(stage="s", reason="r")),
        ):
            with self.subTest(outcome=type(outcome).__name__):
                self.assertFalse(hasattr(outcome, "kind"))
                self.assertFalse(hasattr(outcome, "to_dict"))
                self.assertFalse(hasattr(outcome, "from_dict"))


if __name__ == "__main__":
    unittest.main()
