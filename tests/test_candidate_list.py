from __future__ import annotations

import sys
import unittest
from typing import Any
from unittest import mock

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # ubuntu CI 通常未安装 python3-tk
    tk = None

from cc_cover.gui.candidate_list import (
    CandidateListPanel,
    confirmation_text,
    estimate_processing_seconds,
    format_column_duration,
    format_column_size,
    format_estimate,
    scan_confirmation_stats,
    selection_summary,
)

_TREE_COLUMNS = ("state", "video", "target", "duration", "size", "estimate", "format")


def _sample_report() -> dict[str, Any]:
    return {
        "video_count": 3,
        "candidate_count": 2,
        "conflict_count": 1,
        "protected_nonempty_txt_count": 0,
        "candidates": [
            {
                "sample_id": "CC-CANDIDATE-00001",
                "video_path": "C:/videos/a.mp4",
                "target_path": "C:/videos/a.txt",
                "state": "missing",
                "video_duration_s": 120,
                "video_size": 1024,
            },
            {
                "sample_id": "CC-CANDIDATE-00002",
                "video_path": "C:/videos/b.mp4",
                "target_path": "C:/videos/b.txt",
                "state": "empty",
                "video_duration_s": 60,
                "video_size": 2048,
            },
        ],
        "conflicts": [
            {"target_path": "C:/videos/c.txt", "videos": ["C:/videos/c1.mp4"]},
        ],
    }


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


@unittest.skipUnless(
    sys.platform.startswith("win") and tk is not None, "需要真实 Tk 环境"
)
class CandidateListPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        self.tree = ttk.Treeview(self.root, columns=_TREE_COLUMNS, show="tree headings")
        self.select_all_var = tk.BooleanVar(value=True)
        self.changes = 0
        self.panel = CandidateListPanel(
            self.tree, self.select_all_var, on_change=self._on_change
        )

    def tearDown(self) -> None:
        self.root.destroy()

    def _on_change(self) -> None:
        self.changes += 1

    def test_load_inserts_candidate_and_conflict_rows(self) -> None:
        self.panel.load(_sample_report())

        self.assertEqual(len(self.tree.get_children()), 3)  # 2 候选 + 1 冲突视频
        self.assertEqual(len(self.panel.candidate_row_video), 2)
        self.assertEqual(len(self.panel.conflict_row_ids), 1)

    def test_load_checks_all_candidates_by_default(self) -> None:
        self.panel.load(_sample_report())

        self.assertEqual(len(self.panel.checked_paths), 2)
        for row_id in self.panel.candidate_row_video:
            self.assertEqual(self.tree.item(row_id, "text"), "☑")
        self.assertTrue(self.select_all_var.get())

    def test_load_notifies_on_change(self) -> None:
        self.panel.load(_sample_report())

        self.assertEqual(self.changes, 1)

    def test_conflict_rows_are_not_selectable_candidates(self) -> None:
        self.panel.load(_sample_report())

        conflict_row = next(iter(self.panel.conflict_row_ids))
        self.assertTrue(self.panel.is_conflict(conflict_row))
        self.assertNotIn(conflict_row, self.panel.candidate_row_video)

    def test_toggle_unchecks_and_marks_excluded(self) -> None:
        self.panel.load(_sample_report())
        row_id = next(iter(self.panel.candidate_row_video))
        video = self.panel.candidate_row_video[row_id]

        self.panel.toggle(row_id)

        self.assertNotIn(video, self.panel.checked_paths)
        self.assertEqual(self.tree.item(row_id, "text"), "☐")
        self.assertEqual(self.tree.item(row_id, "values")[0], "已排除")
        self.assertIn("excluded", self.tree.item(row_id, "tags"))
        self.assertFalse(self.select_all_var.get())
        self.assertEqual(self.changes, 2)  # load() + toggle()

    def test_toggle_twice_restores_original_state(self) -> None:
        self.panel.load(_sample_report())
        row_id = next(iter(self.panel.candidate_row_video))
        original_state = self.panel.original_state_by_row[row_id]

        self.panel.toggle(row_id)
        self.panel.toggle(row_id)

        self.assertEqual(self.tree.item(row_id, "text"), "☑")
        self.assertEqual(self.tree.item(row_id, "values")[0], original_state)
        self.assertTrue(self.select_all_var.get())

    def test_toggle_ignores_unknown_row(self) -> None:
        self.panel.load(_sample_report())
        before = set(self.panel.checked_paths)

        self.panel.toggle("not-a-real-row")

        self.assertEqual(self.panel.checked_paths, before)

    def test_toggle_all_unchecks_every_candidate(self) -> None:
        self.panel.load(_sample_report())
        self.select_all_var.set(False)

        self.panel.toggle_all()

        self.assertEqual(self.panel.checked_paths, set())
        for row_id in self.panel.candidate_row_video:
            self.assertEqual(self.tree.item(row_id, "text"), "☐")

    def test_toggle_all_rechecks_every_candidate(self) -> None:
        self.panel.load(_sample_report())
        self.select_all_var.set(False)
        self.panel.toggle_all()

        self.select_all_var.set(True)
        self.panel.toggle_all()

        self.assertEqual(
            self.panel.checked_paths, set(self.panel.candidate_row_video.values())
        )

    def test_summary_text_reflects_selection(self) -> None:
        self.panel.load(_sample_report())
        row_id = next(iter(self.panel.candidate_row_video))

        self.panel.toggle(row_id)

        self.assertIn("已选 1 个", self.panel.summary_text())

    def test_excluded_paths_lists_unchecked_videos_sorted(self) -> None:
        self.panel.load(_sample_report())
        for row_id in self.panel.candidate_row_video:
            self.panel.toggle(row_id)

        self.assertEqual(
            self.panel.excluded_paths(),
            sorted(self.panel.candidate_row_video.values()),
        )

    def test_on_click_toggles_candidate_row_on_checkbox_column(self) -> None:
        self.panel.load(_sample_report())
        row_id = next(iter(self.panel.candidate_row_video))
        video = self.panel.candidate_row_video[row_id]
        self.tree.identify_row = mock.Mock(return_value=row_id)
        self.tree.identify_column = mock.Mock(return_value="#0")

        self.panel.on_click(mock.Mock(x=5, y=5))

        self.assertNotIn(video, self.panel.checked_paths)

    def test_on_click_ignores_conflict_rows(self) -> None:
        self.panel.load(_sample_report())
        conflict_row = next(iter(self.panel.conflict_row_ids))
        self.tree.identify_row = mock.Mock(return_value=conflict_row)
        self.tree.identify_column = mock.Mock(return_value="#0")
        before = set(self.panel.checked_paths)

        self.panel.on_click(mock.Mock(x=5, y=5))

        self.assertEqual(self.panel.checked_paths, before)

    def test_on_click_ignores_non_checkbox_column(self) -> None:
        self.panel.load(_sample_report())
        row_id = next(iter(self.panel.candidate_row_video))
        self.tree.identify_row = mock.Mock(return_value=row_id)
        self.tree.identify_column = mock.Mock(return_value="#1")
        before = set(self.panel.checked_paths)

        self.panel.on_click(mock.Mock(x=5, y=5))

        self.assertEqual(self.panel.checked_paths, before)

    def test_show_context_menu_offers_restore_when_excluded(self) -> None:
        # tk_popup() 会进入本地模态事件循环、在无人交互的测试里永久阻塞，
        # 必须 mock 掉 tk.Menu 本身，不能真的构造/弹出。
        self.panel.load(_sample_report())
        row_id = next(iter(self.panel.candidate_row_video))
        self.panel.toggle(row_id)  # 先排除
        self.tree.identify_row = mock.Mock(return_value=row_id)

        with mock.patch("tkinter.Menu") as menu_cls:
            self.panel.show_context_menu(
                self.root,
                mock.Mock(x_root=100, y_root=100, y=5),
                open_in_explorer=lambda _path: None,
            )

        menu = menu_cls.return_value
        labels = [call.kwargs["label"] for call in menu.add_command.call_args_list]
        self.assertIn("恢复", labels)
        self.assertIn("打开视频所在位置", labels)
        self.assertIn("打开目标 TXT 所在位置", labels)
        menu.tk_popup.assert_called_once_with(100, 100)

    def test_show_context_menu_offers_exclude_when_selected(self) -> None:
        self.panel.load(_sample_report())
        row_id = next(iter(self.panel.candidate_row_video))
        self.tree.identify_row = mock.Mock(return_value=row_id)

        with mock.patch("tkinter.Menu") as menu_cls:
            self.panel.show_context_menu(
                self.root,
                mock.Mock(x_root=100, y_root=100, y=5),
                open_in_explorer=lambda _path: None,
            )

        labels = [
            call.kwargs["label"]
            for call in menu_cls.return_value.add_command.call_args_list
        ]
        self.assertIn("从本次处理中排除", labels)

    def test_show_context_menu_open_in_explorer_receives_video_path(self) -> None:
        self.panel.load(_sample_report())
        row_id = next(iter(self.panel.candidate_row_video))
        video = self.panel.candidate_row_video[row_id]
        self.tree.identify_row = mock.Mock(return_value=row_id)
        opened: list[str] = []

        with mock.patch("tkinter.Menu") as menu_cls:
            self.panel.show_context_menu(
                self.root,
                mock.Mock(x_root=100, y_root=100, y=5),
                open_in_explorer=opened.append,
            )

        commands = {
            call.kwargs["label"]: call.kwargs["command"]
            for call in menu_cls.return_value.add_command.call_args_list
        }
        commands["打开视频所在位置"]()
        self.assertEqual(opened, [video])

    def test_show_context_menu_does_nothing_for_unknown_row(self) -> None:
        self.tree.identify_row = mock.Mock(return_value="")

        with mock.patch("tkinter.Menu") as menu_cls:
            self.panel.show_context_menu(
                self.root, mock.Mock(y=5), open_in_explorer=lambda _path: None
            )

        menu_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
