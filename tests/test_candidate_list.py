from __future__ import annotations

import unittest

from cc_cover.gui.candidate_list import (
    confirmation_text,
    estimate_processing_seconds,
    format_column_duration,
    format_column_size,
    format_estimate,
    scan_confirmation_stats,
    selection_summary,
)


class EstimateFormattingTests(unittest.TestCase):
    def test_estimate_requires_duration_and_grows_with_work(self) -> None:
        self.assertIsNone(estimate_processing_seconds(None, 1024))
        self.assertEqual(estimate_processing_seconds(600, 0), 210)
        self.assertEqual(estimate_processing_seconds(600, 8 * 1024 * 1024), 211)
        self.assertEqual(estimate_processing_seconds(3600, 0), 1110)
        self.assertLess(
            estimate_processing_seconds(120, 0),
            estimate_processing_seconds(600, 0),
        )

    def test_format_duration_helpers(self) -> None:
        self.assertEqual(format_column_duration(None), "—")
        self.assertEqual(format_column_duration(0), "00:00")
        self.assertEqual(format_column_duration(83.45), "01:23")
        self.assertEqual(format_column_duration(3723), "1:02:03")

    def test_format_size_helpers(self) -> None:
        self.assertEqual(format_column_size(None), "—")
        self.assertEqual(format_column_size(512), "512 B")
        self.assertEqual(format_column_size(1536), "1.5 KB")
        self.assertEqual(format_column_size(1048576), "1.0 MB")
        self.assertEqual(format_column_size(5 * 1024**3), "5.0 GB")

    def test_format_estimate_helpers(self) -> None:
        self.assertEqual(format_estimate(None), "—")
        self.assertEqual(format_estimate(45), "45 秒")
        self.assertEqual(format_estimate(90), "约 2 分钟")
        self.assertEqual(format_estimate(7200), "约 2.0 小时")

    def test_selection_summary_includes_selected_and_excluded_counts(self) -> None:
        self.assertEqual(
            selection_summary(
                video_count=10,
                candidate_count=8,
                selected_count=5,
                conflict_count=2,
                protected_count=3,
            ),
            "视频 10 个 · 待补全 8 个 · 已选 5 个 · 已排除 3 个 · 冲突 2 个 · 受保护非空 TXT 3 个",
        )


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


if __name__ == "__main__":
    unittest.main()
