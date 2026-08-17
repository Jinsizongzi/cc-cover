from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cc_cover.gui_support import (
    ProgressSnapshot,
    ProgressTracker,
    confirmation_text,
    format_duration,
    progress_text,
    scan_confirmation_stats,
    should_play_completion_sound,
)
from cc_cover.models import DoneEvent, EngineStartEvent, ProgressEvent, RunDirEvent
from cc_cover.pipeline import CompletionStats, run_completion_stats


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
        self.assertEqual(snapshot.percent, 20)

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
        # 平均完成耗时 100 秒 × 剩余 9 个。
        self.assertEqual(snapshot.remaining_seconds, 900.0)

    def test_no_remaining_estimate_when_nothing_completed(self) -> None:
        tracker = ProgressTracker(total=10)
        tracker.on_event(_progress("funasr", 1, 1, "C:\\a.mp4"))

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


class ScanConfirmationStatsTests(unittest.TestCase):
    def test_counts_conflict_videos_as_excluded(self) -> None:
        report = {
            "candidate_count": 5,
            "conflicts": [
                {"target_path": "t1.txt", "videos": ["a.mp4", "b.mp4"]},
                {"target_path": "t2.txt", "videos": ["c.mp4"]},
            ],
        }

        candidates, excluded = scan_confirmation_stats(report)

        self.assertEqual(candidates, 5)
        self.assertEqual(excluded, 3)

    def test_prefers_explicit_excluded_count_when_present(self) -> None:
        report = {"candidate_count": 5, "excluded_count": 2, "conflicts": []}

        candidates, excluded = scan_confirmation_stats(report)

        self.assertEqual((candidates, excluded), (5, 2))

    def test_zero_excluded_without_conflicts(self) -> None:
        self.assertEqual(
            scan_confirmation_stats({"candidate_count": 4, "conflicts": []}),
            (4, 0),
        )

    def test_empty_report_defaults_to_zero(self) -> None:
        self.assertEqual(scan_confirmation_stats({}), (0, 0))


class ConfirmationTextTests(unittest.TestCase):
    def test_mentions_count_backup_and_excluded(self) -> None:
        text = confirmation_text(8, 2)

        self.assertIn("将处理 8 个视频并替换同名 TXT", text)
        self.assertIn("替换前自动备份", text)
        self.assertIn("已排除 2 个视频", text)

    def test_zero_excluded_uses_explicit_wording(self) -> None:
        text = confirmation_text(1, 0)

        self.assertIn("将处理 1 个视频", text)
        self.assertNotIn("已排除 0 个视频，本次不处理", text)


class CompletionStatsTests(unittest.TestCase):
    def test_reads_elapsed_written_and_warning_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at_utc": "2026-08-02T08:00:00+00:00",
                        "updated_at_utc": "2026-08-02T08:05:30+00:00",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "commit_report.json").write_text(
                json.dumps(
                    {
                        "committed_at_utc": "2026-08-02T08:05:30+00:00",
                        "entry_count": 7,
                        "entries": [{"sample_id": f"s{i}"} for i in range(7)],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "stage_report.json").write_text(
                json.dumps({"warning_count": 3}),
                encoding="utf-8",
            )

            stats = run_completion_stats(run_dir)

        self.assertEqual(stats.elapsed_seconds, 330.0)
        self.assertEqual(stats.written_count, 7)
        self.assertEqual(stats.warning_count, 3)

    def test_missing_artifacts_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()

            stats = run_completion_stats(run_dir)

        self.assertEqual(stats, CompletionStats(None, 0, 0))

    def test_unparseable_dates_yield_unknown_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps({"created_at_utc": "不是日期"}),
                encoding="utf-8",
            )
            (run_dir / "commit_report.json").write_text(
                json.dumps({"committed_at_utc": "2026-08-02T08:05:30+00:00"}),
                encoding="utf-8",
            )

            stats = run_completion_stats(run_dir)

        self.assertIsNone(stats.elapsed_seconds)

    def test_invalid_json_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            for name in ("manifest.json", "commit_report.json", "stage_report.json"):
                (run_dir / name).write_text("broken", encoding="utf-8")

            stats = run_completion_stats(run_dir)

        self.assertEqual(stats, CompletionStats(None, 0, 0))

    def test_commit_without_dates_yields_unknown_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "commit_report.json").write_text(
                json.dumps({"entry_count": 2, "entries": [{"s": "a"}]}),
                encoding="utf-8",
            )

            stats = run_completion_stats(run_dir)

        self.assertIsNone(stats.elapsed_seconds)
        self.assertEqual(stats.written_count, 2)

    def test_written_count_falls_back_to_entries_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "commit_report.json").write_text(
                json.dumps({"entries": [{"sample_id": "a"}, {"sample_id": "b"}]}),
                encoding="utf-8",
            )

            stats = run_completion_stats(run_dir)

        self.assertEqual(stats.written_count, 2)


class FormatDurationTests(unittest.TestCase):
    def test_plain_seconds(self) -> None:
        self.assertEqual(format_duration(0), "0 秒")
        self.assertEqual(format_duration(45), "45 秒")

    def test_minutes_and_seconds(self) -> None:
        self.assertEqual(format_duration(120), "2 分 0 秒")
        self.assertEqual(format_duration(125), "2 分 5 秒")

    def test_hours_minutes_seconds(self) -> None:
        self.assertEqual(format_duration(3725), "1 时 2 分 5 秒")

    def test_unknown_duration(self) -> None:
        self.assertEqual(format_duration(None), "未知")


class CompletionSoundTests(unittest.TestCase):
    def test_plays_sound_when_run_exceeds_five_minutes(self) -> None:
        self.assertTrue(should_play_completion_sound(301.0))

    def test_no_sound_at_exactly_five_minutes(self) -> None:
        self.assertFalse(should_play_completion_sound(300.0))

    def test_no_sound_under_five_minutes(self) -> None:
        self.assertFalse(should_play_completion_sound(299.0))

    def test_no_sound_when_elapsed_unknown(self) -> None:
        self.assertFalse(should_play_completion_sound(None))


if __name__ == "__main__":
    unittest.main()
