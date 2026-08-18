from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cc_cover.gui_support import should_play_completion_sound
from cc_cover.pipeline import CompletionStats, run_completion_stats


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
