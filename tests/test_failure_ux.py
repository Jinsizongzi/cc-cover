from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cc_cover.gui_support import (
    FailureInfo,
    error_text,
    failure_info,
    first_failed_sample,
    run_is_resumable,
    terminate_process_tree,
)


class FailureInfoTests(unittest.TestCase):
    def test_parses_run_dir_with_spaces_and_last_progress(self) -> None:
        output = (
            "运行目录：C:\\Users\\me\\AppData\\Local\\CC-Cover\\runs\\20260802_010203_12345\n"
            "[funasr 44/66] F:\\LLM\\Vibe Coding\\47_skills实操.mp4\n"
            "[funasr 45/66] F:\\LLM\\Vibe Coding\\48_总结.mp4\n"
        )

        info = failure_info(output, fallback_stage="转写与写回")

        self.assertEqual(
            info.run_dir,
            Path("C:\\Users\\me\\AppData\\Local\\CC-Cover\\runs\\20260802_010203_12345"),
        )
        self.assertEqual(info.done_count, 45)
        self.assertEqual(info.total_count, 66)
        self.assertEqual(info.file, "F:\\LLM\\Vibe Coding\\48_总结.mp4")
        self.assertEqual(info.stage, "转写与写回")
        self.assertEqual(info.reason, output.strip())

    def test_parses_file_stage_and_reason_from_engine_error(self) -> None:
        output = (
            "运行目录：C:\\runs\\20260802_010203_12345\n"
            "[faster_whisper 3/10] E:\\videos\\47_skills实操.mp4\n"
            "错误：引擎字幕段无效：#20 (engine=faster-whisper, "
            "sample=CC-CANDIDATE-00047, video=E:\\videos\\47_skills实操.mp4, "
            "duration_ms=600000, start_ms=12000, end_ms=11000)\n"
        )

        info = failure_info(output, fallback_stage="转写与写回")

        self.assertEqual(info.stage, "faster-whisper 转写")
        self.assertEqual(info.file, "E:\\videos\\47_skills实操.mp4")
        self.assertIn("引擎字幕段无效", info.reason)
        self.assertEqual(info.run_dir, Path("C:\\runs\\20260802_010203_12345"))

    def test_audio_extraction_failure_names_file_and_stage(self) -> None:
        info = failure_info(
            "错误：音频提取失败：E:\\videos\\broken.mp4: invalid data",
            fallback_stage="转写与写回",
        )

        self.assertEqual(info.stage, "音频提取")
        self.assertEqual(info.file, "E:\\videos\\broken.mp4")

    def test_fallback_stage_used_when_reason_has_no_hints(self) -> None:
        info = failure_info("错误：磁盘已满", fallback_stage="写回")

        self.assertEqual(info.stage, "写回")
        self.assertEqual(info.reason, "磁盘已满")
        self.assertIsNone(info.file)
        self.assertIsNone(info.run_dir)

    def test_explicit_reason_and_run_dir_win_over_output(self) -> None:
        info = failure_info(
            "运行目录：C:\\runs\\a\n错误：旧错误",
            fallback_stage="继续中断任务",
            reason="新原因",
            run_dir=Path("C:\\runs\\b"),
        )

        self.assertEqual(info.reason, "新原因")
        self.assertEqual(info.run_dir, Path("C:\\runs\\b"))
        self.assertEqual(info.stage, "继续中断任务")

    def test_quality_gate_error_uses_quality_stage(self) -> None:
        info = failure_info(
            "错误：试样或全量质量门禁未通过，未写回课程目录",
            fallback_stage="转写与写回",
        )

        self.assertEqual(info.stage, "质量门禁")


class FirstFailedSampleTests(unittest.TestCase):
    def test_returns_first_failed_sample_from_stage_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "stage_report.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "a",
                                "video_path": "E:\\videos\\ok.mp4",
                                "passed": True,
                                "errors": [],
                            },
                            {
                                "sample_id": "b",
                                "video_path": "E:\\videos\\bad.mp4",
                                "passed": False,
                                "errors": ["文本密度异常：30.0 chars/min"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = first_failed_sample(run_dir)

        self.assertEqual(
            result, ("E:\\videos\\bad.mp4", "文本密度异常：30.0 chars/min")
        )

    def test_returns_none_without_stage_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(first_failed_sample(Path(temporary) / "missing"))

    def test_returns_none_when_report_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "stage_report.json").write_text(
                "not json", encoding="utf-8"
            )
            self.assertIsNone(first_failed_sample(run_dir))


class ResumeabilityTests(unittest.TestCase):
    def test_run_is_resumable_only_with_non_committed_manifest(self) -> None:
        self.assertFalse(run_is_resumable(None))
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.assertFalse(run_is_resumable(run_dir))
            run_dir.mkdir()
            self.assertFalse(run_is_resumable(run_dir))

            manifest = run_dir / "manifest.json"
            manifest.write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )
            self.assertTrue(run_is_resumable(run_dir))

            manifest.write_text(
                json.dumps({"status": "committed"}), encoding="utf-8"
            )
            self.assertFalse(run_is_resumable(run_dir))

            manifest.write_text("broken", encoding="utf-8")
            self.assertFalse(run_is_resumable(run_dir))


class ErrorTextTests(unittest.TestCase):
    def test_error_text_contains_all_dialog_fields(self) -> None:
        info = FailureInfo(
            stage="faster-whisper 转写",
            reason="引擎字幕段无效",
            file="E:\\videos\\bad.mp4",
            run_dir=Path("C:\\runs\\20260802_010203_12345"),
            done_count=44,
            total_count=66,
        )

        text = error_text("字幕补全失败", info)

        self.assertIn("字幕补全失败", text)
        self.assertIn("文件：E:\\videos\\bad.mp4", text)
        self.assertIn("阶段：faster-whisper 转写", text)
        self.assertIn("原因：引擎字幕段无效", text)
        self.assertIn("运行目录：C:\\runs\\20260802_010203_12345", text)
        self.assertIn("已处理 44/66 个视频，产物已暂存。", text)


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


if __name__ == "__main__":
    unittest.main()
