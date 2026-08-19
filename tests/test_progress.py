from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # ubuntu CI 通常未安装 python3-tk
    tk = None

from cc_cover.core.models import (
    DoneEvent,
    EngineStartEvent,
    ErrorEvent,
    Phase,
    ProgressEvent,
    RunDirEvent,
)
from cc_cover.gui.progress import (
    FailureInfo,
    InstallProgressSnapshot,
    InstallProgressTracker,
    ProgressPresenter,
    ProgressSnapshot,
    ProgressTracker,
    captured_events,
    detect_stage,
    done_event_present,
    error_text,
    failure_info,
    failure_info_from_command,
    failure_info_from_run,
    first_failed_sample,
    install_progress_text,
    last_error_event,
    last_progress_counts,
    parse_event_line,
    phase_stage_label,
    progress_text,
    run_dir_from_events,
    run_is_resumable,
    stopped_message,
)


class FailureInfoTests(unittest.TestCase):
    def test_command_failure_info_combines_chunks_with_exception(self) -> None:
        info = failure_info_from_command(
            ["开始安装运行环境。此过程可能需要较长时间。\n"],
            RuntimeError("错误：安装组件超时"),
            fallback_stage="安装运行环境",
        )

        self.assertIsNone(info.run_dir)
        self.assertEqual(
            info.reason,
            "开始安装运行环境。此过程可能需要较长时间。\n错误：安装组件超时",
        )

    def test_audio_extraction_failure_stage_still_detected_without_file(self) -> None:
        info = failure_info(
            "错误：音频提取失败：E:\\videos\\broken.mp4: invalid data",
            fallback_stage="转写与写回",
        )

        self.assertEqual(info.stage, "音频提取")
        self.assertIsNone(info.file)

    def test_fallback_stage_used_when_reason_has_no_hints(self) -> None:
        info = failure_info("错误：磁盘已满", fallback_stage="写回")

        self.assertEqual(info.stage, "写回")
        self.assertEqual(info.reason, "错误：磁盘已满")
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
            (run_dir / "stage_report.json").write_text("not json", encoding="utf-8")
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
            manifest.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            self.assertTrue(run_is_resumable(run_dir))

            manifest.write_text(json.dumps({"status": "committed"}), encoding="utf-8")
            self.assertFalse(run_is_resumable(run_dir))

            manifest.write_text("broken", encoding="utf-8")
            self.assertFalse(run_is_resumable(run_dir))


class CapturedEventsTests(unittest.TestCase):
    def test_extracts_structured_events_in_order_and_skips_plain_text(self) -> None:
        output = (
            "运行目录：C:\\runs\\a\n"
            + json.dumps(RunDirEvent(path="C:\\runs\\a").to_dict())
            + "\n"
            + json.dumps(
                ProgressEvent(
                    engine="funasr", index=1, total=2, video_path="a.mp4"
                ).to_dict()
            )
            + "\n"
        )

        self.assertEqual(
            captured_events(output),
            [
                RunDirEvent(path="C:\\runs\\a"),
                ProgressEvent(engine="funasr", index=1, total=2, video_path="a.mp4"),
            ],
        )

    def test_empty_output_yields_no_events(self) -> None:
        self.assertEqual(captured_events(""), [])


class RunDirFromEventsTests(unittest.TestCase):
    def test_returns_last_run_dir_event(self) -> None:
        output = (
            json.dumps(RunDirEvent(path="C:\\runs\\a").to_dict())
            + "\n[funasr 1/2] E:\\videos\\a.mp4\n"
            + json.dumps(RunDirEvent(path="C:\\runs\\b").to_dict())
            + "\n字幕已写回并复核通过：C:\\runs\\b\n"
        )

        self.assertEqual(run_dir_from_events(output), Path("C:\\runs\\b"))

    def test_returns_none_without_run_dir_event(self) -> None:
        self.assertIsNone(run_dir_from_events("没有运行目录"))
        self.assertIsNone(run_dir_from_events(""))


class DoneEventPresentTests(unittest.TestCase):
    def test_true_when_done_event_captured(self) -> None:
        output = (
            "[funasr 1/2] E:\\videos\\a.mp4\n"
            + json.dumps(DoneEvent(run_dir="C:\\runs\\a").to_dict())
            + "\n"
        )

        self.assertTrue(done_event_present(output))

    def test_false_without_done_event(self) -> None:
        output = json.dumps(ErrorEvent(phase=Phase.FUNASR, reason="失败").to_dict())

        self.assertFalse(done_event_present(output))
        self.assertFalse(done_event_present(""))


class LastErrorEventTests(unittest.TestCase):
    def test_returns_last_error_event(self) -> None:
        output = (
            json.dumps(ErrorEvent(phase=Phase.FUNASR, reason="第一次失败").to_dict())
            + "\n"
            + json.dumps(
                ErrorEvent(
                    phase=Phase.WRITEBACK,
                    reason="第二次失败",
                    video_path="E:/a.mp4",
                    sample_id="s1",
                ).to_dict()
            )
        )

        event = last_error_event(output)

        assert event is not None
        self.assertEqual(event.phase, Phase.WRITEBACK)
        self.assertEqual(event.reason, "第二次失败")
        self.assertEqual(event.video_path, "E:/a.mp4")
        self.assertEqual(event.sample_id, "s1")

    def test_returns_none_without_error_event(self) -> None:
        self.assertIsNone(last_error_event("普通输出，没有错误"))


class LastProgressCountsTests(unittest.TestCase):
    def test_returns_last_progress_event_counts(self) -> None:
        output = "\n".join(
            json.dumps(
                ProgressEvent(
                    engine="funasr", index=index, total=5, video_path=f"{index}.mp4"
                ).to_dict()
            )
            for index in (1, 2, 3)
        )

        self.assertEqual(last_progress_counts(output), (3, 5))

    def test_returns_none_without_progress_event(self) -> None:
        self.assertIsNone(last_progress_counts("普通输出"))


class PhaseStageLabelTests(unittest.TestCase):
    def test_known_phases_map_to_existing_stage_labels(self) -> None:
        cases = {
            Phase.AUDIO_EXTRACT: "音频提取",
            Phase.FUNASR: "FunASR 转写",
            Phase.FASTER_WHISPER: "faster-whisper 转写",
            Phase.QUALITY_GATE: "质量门禁",
            Phase.WRITEBACK: "写回",
            Phase.VERIFY: "写回",
        }
        for phase, label in cases.items():
            with self.subTest(phase=phase):
                self.assertEqual(phase_stage_label(phase, "兜底"), label)

    def test_setup_falls_back_without_a_dedicated_label(self) -> None:
        self.assertEqual(phase_stage_label(Phase.SETUP, "转写与写回"), "转写与写回")


class FailureInfoFromRunTests(unittest.TestCase):
    def test_builds_directly_from_captured_error_event(self) -> None:
        chunks = [
            json.dumps(RunDirEvent(path="C:\\runs\\a").to_dict()) + "\n",
            "[faster_whisper 3/10] E:\\videos\\47_skills实操.mp4\n",
            json.dumps(
                ProgressEvent(
                    engine="faster_whisper",
                    index=3,
                    total=10,
                    video_path="E:\\videos\\47_skills实操.mp4",
                ).to_dict()
            )
            + "\n",
            json.dumps(
                ErrorEvent(
                    phase=Phase.FASTER_WHISPER,
                    reason="引擎字幕段无效：#20",
                    video_path="E:\\videos\\47_skills实操.mp4",
                    sample_id="CC-CANDIDATE-00047",
                ).to_dict()
            )
            + "\n",
        ]

        info = failure_info_from_run(
            chunks,
            RuntimeError("任务执行失败，退出代码：1"),
            fallback_stage="转写与写回",
        )

        self.assertEqual(info.stage, "faster-whisper 转写")
        self.assertEqual(info.reason, "引擎字幕段无效：#20")
        self.assertEqual(info.file, "E:\\videos\\47_skills实操.mp4")
        self.assertEqual(info.run_dir, Path("C:\\runs\\a"))
        self.assertEqual(info.done_count, 3)
        self.assertEqual(info.total_count, 10)

    def test_explicit_run_dir_wins_over_captured_run_dir_event(self) -> None:
        chunks = [
            json.dumps(RunDirEvent(path="C:\\runs\\a").to_dict()) + "\n",
            json.dumps(ErrorEvent(phase=Phase.WRITEBACK, reason="写回失败").to_dict())
            + "\n",
        ]

        info = failure_info_from_run(
            chunks,
            RuntimeError(""),
            fallback_stage="继续中断任务",
            run_dir=Path("C:\\runs\\b"),
        )

        self.assertEqual(info.run_dir, Path("C:\\runs\\b"))

    def test_falls_back_to_plain_text_when_process_was_hard_killed(self) -> None:
        # 典型场景：用户点「停止」，CLI 进程被硬杀，来不及吐出任何结构化事件。
        chunks = ["运行目录：C:\\runs\\a\n[funasr 1/2] E:\\videos\\a.mp4\n"]

        info = failure_info_from_run(
            chunks,
            RuntimeError("任务已由用户停止。"),
            fallback_stage="转写与写回",
        )

        self.assertIsNone(info.file)
        self.assertIsNone(info.run_dir)
        self.assertIn("任务已由用户停止", info.reason)


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

        self.assertEqual(
            message, "任务已停止，产物已暂存，可点击「继续中断任务」恢复。"
        )

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
        line = json.dumps(
            {"event": "engine_start", "engine": "funasr", "device": "cuda"}
        )

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
            ProgressEvent(
                engine="faster_whisper", index=2, total=5, video_path="a.mp4"
            ),
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
        line = json.dumps(
            {"event": "error", "phase": "setup", "reason": "找不到 FFmpeg"}
        )

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


def _progress(engine: str, index: int, total: int, video_path: str) -> ProgressEvent:
    return ProgressEvent(engine=engine, index=index, total=total, video_path=video_path)


class ProgressTrackerTests(unittest.TestCase):
    def test_counts_file_only_after_both_engine_lines(self) -> None:
        tracker = ProgressTracker(total=5)
        tracker.on_event(_progress("funasr", 1, 2, "C:\\videos\\a.mp4"))
        tracker.on_event(_progress("funasr", 2, 2, "C:\\videos\\b.mp4"))
        # a 已完成双模型，b 只出现一次 funasr，仍不算完成。
        tracker.on_event(_progress("faster_whisper", 1, 2, "C:\\videos\\a.mp4"))

        snapshot = tracker.snapshot(now=100.0)

        self.assertEqual(snapshot.current, 1)
        self.assertEqual(snapshot.total, 5)
        # percent 走步数口径（3 步 / 共 10 步），不是 candidate 完成比例
        # （current/total 会是 20%）——两者故意不同，见 ProgressTracker 类文档。
        self.assertEqual(snapshot.percent, 30)

    def test_ignores_non_progress_events_and_plain_text(self) -> None:
        tracker = ProgressTracker(total=3)
        tracker.on_event(RunDirEvent(path="C:\\runs\\a"))
        tracker.on_event(EngineStartEvent(engine="funasr", device="cpu"))
        tracker.on_event(DoneEvent(run_dir="C:\\runs\\a"))
        tracker.on_event("普通人读文字，不是事件")

        self.assertEqual(tracker.snapshot(now=10.0).current, 0)

    def test_estimates_remaining_by_average_duration(self) -> None:
        tracker = ProgressTracker(total=10, started_at=0.0)
        tracker.on_event(_progress("funasr", 1, 1, "C:\\a.mp4"))
        tracker.on_event(_progress("faster_whisper", 1, 1, "C:\\a.mp4"))

        snapshot = tracker.snapshot(now=100.0)

        self.assertEqual(snapshot.elapsed_seconds, 100.0)
        # 步数口径：共 20 步（10 候选 × 2 引擎），已走 2 步，平均每步 50 秒
        # × 剩余 18 步。
        self.assertEqual(snapshot.remaining_seconds, 900.0)

    def test_remaining_estimated_once_any_engine_has_touched_a_candidate(
        self,
    ) -> None:
        """只有一个引擎碰过一个候选（远未到"完成一个"的地步）也能给出粗估。

        这是这次改动本身要解决的问题：批次内两个引擎分两轮跑，funasr 单独
        跑的那一整轮里 current（第 N 个）完全不会涨，但步数会——只要走过
        至少一步，就不该继续显示"无法估算"。
        """
        tracker = ProgressTracker(total=10)
        tracker.on_event(_progress("funasr", 1, 1, "C:\\a.mp4"))

        snapshot = tracker.snapshot(now=50.0)

        self.assertEqual(snapshot.current, 0)
        # 共 20 步，已走 1 步，剩余 19 步，每步耗时 50 秒。
        self.assertEqual(snapshot.remaining_seconds, 950.0)
        self.assertEqual(snapshot.percent, 5)

    def test_no_remaining_estimate_when_no_steps_touched_yet(self) -> None:
        tracker = ProgressTracker(total=10)

        snapshot = tracker.snapshot(now=50.0)

        self.assertIsNone(snapshot.remaining_seconds)
        self.assertEqual(snapshot.percent, 0)

    def test_no_remaining_estimate_when_all_done(self) -> None:
        tracker = ProgressTracker(total=2)
        tracker.on_event(_progress("funasr", 1, 2, "C:\\a.mp4"))
        tracker.on_event(_progress("faster_whisper", 1, 2, "C:\\a.mp4"))
        tracker.on_event(_progress("funasr", 2, 2, "C:\\b.mp4"))
        tracker.on_event(_progress("faster_whisper", 2, 2, "C:\\b.mp4"))

        snapshot = tracker.snapshot(now=30.0)

        self.assertEqual(snapshot.percent, 100)
        self.assertIsNone(snapshot.remaining_seconds)

    def test_percent_rounds_to_nearest_integer(self) -> None:
        tracker = ProgressTracker(total=3)
        tracker.on_event(_progress("funasr", 1, 1, "C:\\a.mp4"))
        tracker.on_event(_progress("faster_whisper", 1, 1, "C:\\a.mp4"))

        self.assertEqual(tracker.snapshot(now=1.0).percent, 33)

    def test_zero_total_never_divides_by_zero(self) -> None:
        tracker = ProgressTracker(total=0)

        snapshot = tracker.snapshot(now=1.0)

        self.assertEqual(snapshot.total, 0)
        self.assertEqual(snapshot.percent, 0)
        self.assertEqual(snapshot.current, 0)
        self.assertIsNone(snapshot.remaining_seconds)

    def test_negative_total_is_clamped(self) -> None:
        self.assertEqual(ProgressTracker(total=-3).total, 0)


class ProgressTextTests(unittest.TestCase):
    def test_contains_count_percent_elapsed_and_estimate(self) -> None:
        snapshot = ProgressSnapshot(
            current=5,
            total=10,
            percent=50,
            elapsed_seconds=120.0,
            remaining_seconds=300.0,
        )

        text = progress_text(snapshot)

        self.assertIn("第 5 / 共 10 个", text)
        self.assertIn("50%", text)
        self.assertIn("已用时 2 分 0 秒", text)
        self.assertIn("约剩余 5 分 0 秒", text)

    def test_omits_estimate_when_unavailable(self) -> None:
        snapshot = ProgressSnapshot(
            current=0, total=10, percent=0, elapsed_seconds=5.0, remaining_seconds=None
        )

        text = progress_text(snapshot)

        self.assertNotIn("约剩余", text)
        self.assertIn("已用时 5 秒", text)


class InstallProgressTrackerTests(unittest.TestCase):
    def test_accumulates_download_header_sizes(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        tracker.on_output("Downloading torch-2.5.1-cp312.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=10.0)

        self.assertEqual(snapshot.downloaded_bytes, 500 * 1024**2)

    def test_parses_small_units(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        tracker.on_output("Downloading six.whl (10 kB)\n")

        self.assertEqual(tracker.snapshot(now=1.0).downloaded_bytes, 10 * 1024)

    def test_multiple_headers_accumulate(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        tracker.on_output("Downloading a.whl (100.0 MB)\nDownloading b.whl (50.0 MB)\n")

        self.assertEqual(tracker.snapshot(now=5.0).downloaded_bytes, 150 * 1024**2)

    def test_ignores_non_download_lines(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        for line in (
            "Collecting six\n",
            "Installing collected packages\n",
            "Successfully installed six\n",
        ):
            tracker.on_output(line)

        self.assertEqual(tracker.snapshot(now=5.0).downloaded_bytes, 0)

    def test_downloaded_capped_at_total(self) -> None:
        tracker = InstallProgressTracker(total_bytes=100 * 1024**2, component_count=4)
        tracker.on_output("Downloading huge.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=5.0)

        self.assertEqual(snapshot.downloaded_bytes, 100 * 1024**2)

    def test_percent_reflects_completed_components(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1000 * 1024**2, component_count=4)
        self.assertEqual(tracker.snapshot(now=5.0).percent, 0)

        tracker.on_component(1)
        self.assertEqual(tracker.snapshot(now=5.0).percent, 0)

        tracker.on_component(3)
        self.assertEqual(tracker.snapshot(now=5.0).percent, 50)

    def test_speed_and_remaining_estimated_from_elapsed(self) -> None:
        tracker = InstallProgressTracker(
            total_bytes=1000 * 1024**2, component_count=4, started_at=0.0
        )
        tracker.on_output("Downloading a.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=100.0)

        self.assertAlmostEqual(snapshot.speed_bytes, 5 * 1024**2)
        self.assertAlmostEqual(snapshot.remaining_seconds, 100.0)

    def test_no_speed_estimate_when_little_time_elapsed(self) -> None:
        tracker = InstallProgressTracker(
            total_bytes=1000 * 1024**2, component_count=4, started_at=0.0
        )
        tracker.on_output("Downloading a.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=1.0)

        self.assertIsNone(snapshot.speed_bytes)
        self.assertIsNone(snapshot.remaining_seconds)

    def test_no_remaining_estimate_when_downloaded(self) -> None:
        tracker = InstallProgressTracker(total_bytes=500 * 1024**2, component_count=4)
        tracker.on_output("Downloading a.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=60.0)

        self.assertEqual(snapshot.downloaded_bytes, 500 * 1024**2)
        self.assertIsNone(snapshot.remaining_seconds)

    def test_zero_total_never_divides_by_zero(self) -> None:
        tracker = InstallProgressTracker(total_bytes=0, component_count=3)
        tracker.on_output("Downloading a.whl (10.0 MB)\n")

        snapshot = tracker.snapshot(now=5.0)

        self.assertEqual(snapshot.downloaded_bytes, 0)
        self.assertEqual(snapshot.percent, 0)
        self.assertIsNone(snapshot.speed_bytes)

    def test_component_index_tracks_worker_emissions(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=6)
        tracker.on_component(3)

        self.assertEqual(tracker.snapshot(now=1.0).component_index, 3)
        self.assertEqual(tracker.snapshot(now=1.0).component_count, 6)

    def test_parses_rich_progress_bar_frame_when_present(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        tracker.on_output("\r  45%|##### | 450.0/1000.0 MB [00:10<00:12, 45MB/s]")

        self.assertEqual(tracker.snapshot(now=5.0).downloaded_bytes, 450 * 1024**2)


class InstallProgressTextTests(unittest.TestCase):
    def test_text_contains_component_downloaded_and_remaining(self) -> None:
        snapshot = InstallProgressSnapshot(
            component_index=3,
            component_count=6,
            downloaded_bytes=300 * 1024**2,
            total_bytes=900 * 1024**2,
            percent=33,
            speed_bytes=5 * 1024**2,
            remaining_seconds=120.0,
        )

        text = install_progress_text(snapshot)

        self.assertIn("组件 3/6", text)
        self.assertIn("已下载约 300.0 MB", text)
        self.assertIn("约剩余 2 分 0 秒", text)

    def test_text_omits_remaining_when_unavailable(self) -> None:
        snapshot = InstallProgressSnapshot(
            component_index=1,
            component_count=6,
            downloaded_bytes=10 * 1024**2,
            total_bytes=900 * 1024**2,
            percent=1,
            speed_bytes=None,
            remaining_seconds=None,
        )

        text = install_progress_text(snapshot)

        self.assertNotIn("约剩余", text)
        self.assertIn("已下载约 10.0 MB", text)

    def test_text_omits_downloaded_when_zero(self) -> None:
        snapshot = InstallProgressSnapshot(
            component_index=1,
            component_count=6,
            downloaded_bytes=0,
            total_bytes=900 * 1024**2,
            percent=0,
            speed_bytes=None,
            remaining_seconds=None,
        )

        text = install_progress_text(snapshot)

        self.assertNotIn("已下载约", text)
        self.assertIn("组件 1/6", text)


@unittest.skipUnless(
    sys.platform.startswith("win") and tk is not None, "需要真实 Tk 环境"
)
class ProgressPresenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        self.progress = ttk.Progressbar(self.root, mode="indeterminate", length=260)
        self.progress_var = tk.StringVar(value="")
        self.presenter = ProgressPresenter(self.progress, self.progress_var)

    def tearDown(self) -> None:
        self.root.destroy()

    def test_start_busy_switches_to_indeterminate_mode(self) -> None:
        self.presenter.start_busy()

        self.assertEqual(str(self.progress.cget("mode")), "indeterminate")

    def test_stop_busy_leaves_var_untouched_when_nothing_tracked(self) -> None:
        self.progress_var.set("之前的文案")

        self.presenter.stop_busy()

        self.assertEqual(self.progress_var.get(), "之前的文案")

    def test_stop_busy_resets_var_and_elapsed_when_tracking_active(self) -> None:
        self.presenter.start(total=10)

        self.presenter.stop_busy()

        self.assertEqual(self.progress_var.get(), "")
        self.assertIsNone(self.presenter.elapsed())
        self.assertEqual(str(self.progress.cget("mode")), "indeterminate")

    def test_start_switches_to_determinate_and_sets_text(self) -> None:
        self.presenter.start(total=5)

        self.assertEqual(str(self.progress.cget("mode")), "determinate")
        self.assertNotEqual(self.progress_var.get(), "")

    def test_on_line_routes_to_progress_tracker_when_install_not_active(self) -> None:
        self.presenter.start(total=1)

        self.presenter.on_line("普通日志行\n")

        self.assertNotIn("组件", self.progress_var.get())

    def test_on_line_routes_to_install_tracker_when_install_active(self) -> None:
        self.presenter.start_install(total_bytes=100, component_count=2)

        self.presenter.on_line("正在下载……\n")

        self.assertIn("组件", self.progress_var.get())

    def test_on_install_component_updates_component_text(self) -> None:
        self.presenter.start_install(total_bytes=100, component_count=2)

        self.presenter.on_install_component(1, 2)

        self.assertIn("组件 1/2", self.progress_var.get())

    def test_elapsed_none_before_start(self) -> None:
        self.assertIsNone(self.presenter.elapsed())

    def test_elapsed_set_after_start(self) -> None:
        self.presenter.start(total=1)

        self.assertIsNotNone(self.presenter.elapsed())

    def test_start_install_does_not_set_elapsed_anchor(self) -> None:
        self.presenter.start_install(total_bytes=100, component_count=1)

        self.assertIsNone(self.presenter.elapsed())


if __name__ == "__main__":
    unittest.main()
