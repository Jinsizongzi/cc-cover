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
    detect_stage,
    error_text,
    failure_info,
    failure_info_from_command,
    first_failed_sample,
    parse_event_line,
    run_dir_from_output,
    run_is_resumable,
    stopped_message,
    terminate_process_tree,
)
from cc_cover.models import (
    DoneEvent,
    EngineStartEvent,
    ErrorEvent,
    Phase,
    ProgressEvent,
    RunDirEvent,
)


class FailureInfoTests(unittest.TestCase):
    def test_command_failure_output_feeds_dialog_fields(self) -> None:
        exc = RuntimeError(
            "运行目录：C:\\runs\\20260802_010203_12345\n"
            "[faster_whisper 44/66] E:\\videos\\47_skills实操.mp4\n"
            "错误：引擎字幕段无效：#20 (engine=faster-whisper, "
            "sample=CC-CANDIDATE-00047, video=E:\\videos\\47_skills实操.mp4, "
            "duration_ms=600000, start_ms=12000, end_ms=11000)"
        )

        info = failure_info_from_command(
            [], exc, fallback_stage="转写与写回"
        )

        self.assertEqual(info.stage, "faster-whisper 转写")
        self.assertEqual(info.file, "E:\\videos\\47_skills实操.mp4")
        self.assertEqual(
            info.run_dir, Path("C:\\runs\\20260802_010203_12345")
        )
        self.assertEqual(info.done_count, 44)
        self.assertEqual(info.total_count, 66)
        self.assertIn("引擎字幕段无效", info.reason)

    def test_command_failure_info_combines_chunks_with_exception(self) -> None:
        info = failure_info_from_command(
            ["运行目录：C:\\runs\\a\n"],
            RuntimeError("错误：安装组件超时"),
            fallback_stage="安装运行环境",
        )

        self.assertEqual(info.run_dir, Path("C:\\runs\\a"))
        self.assertEqual(info.reason, "安装组件超时")

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

    def test_scan_error_uses_scan_stage(self) -> None:
        self.assertEqual(detect_stage("扫描目录失败：权限不足", "转写与写回"), "扫描")

    def test_writeback_error_uses_writeback_stage(self) -> None:
        self.assertEqual(detect_stage("写回失败：磁盘已满", "转写与写回"), "写回")


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


class RunDirFromOutputTests(unittest.TestCase):
    def test_returns_last_run_dir_match(self) -> None:
        output = (
            "运行目录：C:\\runs\\a\n"
            "[funasr 1/2] E:\\videos\\a.mp4\n"
            "运行目录：C:\\runs\\b\n"
            "字幕已写回并复核通过：C:\\runs\\b\n"
        )

        self.assertEqual(run_dir_from_output(output), Path("C:\\runs\\b"))

    def test_returns_none_without_run_dir_line(self) -> None:
        self.assertIsNone(run_dir_from_output("没有运行目录"))
        self.assertIsNone(run_dir_from_output(""))


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


class StoppedMessageTests(unittest.TestCase):
    def test_resumable_run_mentions_staged_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )
            info = FailureInfo(
                stage="转写与写回",
                reason="用户停止",
                run_dir=run_dir,
            )

            message = stopped_message(info)

        self.assertEqual(message, "任务已停止，产物已暂存，可点击「继续中断任务」恢复。")

    def test_scan_stop_mentions_no_partial_results(self) -> None:
        info = FailureInfo(stage="扫描", reason="用户停止")

        self.assertEqual(
            stopped_message(info),
            "扫描已停止，未展示扫描结果，可重新扫描。",
        )

    def test_stop_without_artifacts_is_honest(self) -> None:
        info = FailureInfo(stage="环境检查", reason="用户停止")

        self.assertEqual(
            stopped_message(info),
            "任务已停止，未产生可恢复的运行产物。",
        )


class ParseEventLineTests(unittest.TestCase):
    def test_decodes_engine_start_event(self) -> None:
        line = json.dumps({"event": "engine_start", "engine": "funasr", "device": "cuda"})

        self.assertEqual(
            parse_event_line(line), EngineStartEvent(engine="funasr", device="cuda")
        )

    def test_decodes_progress_event(self) -> None:
        line = json.dumps(
            {
                "event": "progress",
                "engine": "faster_whisper",
                "index": 2,
                "total": 5,
                "video_path": "a.mp4",
            }
        )

        self.assertEqual(
            parse_event_line(line),
            ProgressEvent(engine="faster_whisper", index=2, total=5, video_path="a.mp4"),
        )

    def test_decodes_run_dir_event(self) -> None:
        line = json.dumps({"event": "run_dir", "path": "C:\\runs\\a"})

        self.assertEqual(parse_event_line(line), RunDirEvent(path="C:\\runs\\a"))

    def test_decodes_done_event(self) -> None:
        line = json.dumps({"event": "done", "run_dir": "C:\\runs\\a"})

        self.assertEqual(parse_event_line(line), DoneEvent(run_dir="C:\\runs\\a"))

    def test_decodes_error_event_with_optional_fields(self) -> None:
        line = json.dumps(
            {
                "event": "error",
                "phase": "writeback",
                "reason": "写回后内容不一致：a.txt",
                "video_path": "E:/videos/a.mp4",
                "sample_id": "CC-MISSING-00047",
            }
        )

        self.assertEqual(
            parse_event_line(line),
            ErrorEvent(
                phase=Phase.WRITEBACK,
                reason="写回后内容不一致：a.txt",
                video_path="E:/videos/a.mp4",
                sample_id="CC-MISSING-00047",
            ),
        )

    def test_decodes_error_event_without_optional_fields(self) -> None:
        line = json.dumps({"event": "error", "phase": "setup", "reason": "找不到 FFmpeg"})

        self.assertEqual(
            parse_event_line(line),
            ErrorEvent(phase=Phase.SETUP, reason="找不到 FFmpeg"),
        )

    def test_plain_human_text_passes_through_unchanged(self) -> None:
        line = "[funasr 1/2] E:\\videos\\a.mp4"

        self.assertEqual(parse_event_line(line), line)

    def test_valid_json_without_event_key_is_plain_text(self) -> None:
        line = json.dumps({"foo": "bar"})

        self.assertEqual(parse_event_line(line), line)

    def test_unknown_event_kind_falls_back_to_plain_text(self) -> None:
        line = json.dumps({"event": "future_kind", "value": 1})

        self.assertEqual(parse_event_line(line), line)

    def test_known_event_kind_missing_required_field_falls_back_to_plain_text(
        self,
    ) -> None:
        line = json.dumps({"event": "progress", "engine": "funasr"})

        self.assertEqual(parse_event_line(line), line)


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
