from __future__ import annotations

import json
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # ubuntu CI 通常未安装 python3-tk
    tk = None

from cc_cover.core.pipeline import CompletionStats
from cc_cover.gui.dialogs import DialogHost, enrich_failure
from cc_cover.gui.progress import FailureInfo
from cc_cover.gui.storage import DiskCheck


def _run_dir_with_failed_sample(root: Path, *, video: str, errors: list[str]) -> Path:
    run_dir = root / "run1"
    run_dir.mkdir()
    (run_dir / "stage_report.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"passed": True, "video_path": "ok.mp4"},
                    {"passed": False, "video_path": video, "errors": errors},
                ]
            }
        ),
        encoding="utf-8",
    )
    return run_dir


class EnrichFailureTests(unittest.TestCase):
    def test_passthrough_when_run_dir_missing(self) -> None:
        info = FailureInfo(stage="转写", reason="子进程失败")

        self.assertEqual(enrich_failure(info), info)

    def test_passthrough_when_file_already_set_and_not_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _run_dir_with_failed_sample(
                Path(temporary), video="bad.mp4", errors=["格式错误"]
            )
            info = FailureInfo(
                stage="转写", reason="子进程失败", file="already.mp4", run_dir=run_dir
            )

            self.assertEqual(enrich_failure(info), info)

    def test_passthrough_when_no_failed_sample_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run1"
            run_dir.mkdir()
            info = FailureInfo(stage="质量门禁", reason="未知", run_dir=run_dir)

            self.assertEqual(enrich_failure(info), info)

    def test_quality_gate_fills_file_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _run_dir_with_failed_sample(
                Path(temporary), video="bad.mp4", errors=["文本密度异常"]
            )
            info = FailureInfo(stage="质量门禁", reason="占位", run_dir=run_dir)

            result = enrich_failure(info)

            self.assertEqual(result.file, "bad.mp4")
            self.assertEqual(result.reason, "文本密度异常")

    def test_quality_gate_keeps_existing_file_but_replaces_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _run_dir_with_failed_sample(
                Path(temporary), video="bad.mp4", errors=["文本密度异常"]
            )
            info = FailureInfo(
                stage="质量门禁", reason="占位", file="already.mp4", run_dir=run_dir
            )

            result = enrich_failure(info)

            self.assertEqual(result.file, "already.mp4")
            self.assertEqual(result.reason, "文本密度异常")

    def test_non_quality_gate_fills_file_but_keeps_original_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _run_dir_with_failed_sample(
                Path(temporary), video="bad.mp4", errors=["文本密度异常"]
            )
            info = FailureInfo(stage="转写", reason="子进程失败", run_dir=run_dir)

            result = enrich_failure(info)

            self.assertEqual(result.file, "bad.mp4")
            self.assertEqual(result.reason, "子进程失败")


@unittest.skipUnless(
    sys.platform.startswith("win") and tk is not None, "需要真实 Tk 环境"
)
class DialogHostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        notebook = ttk.Notebook(self.root)
        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="运行日志")
        self.notebook = notebook
        self.log_tab = log_tab
        self.opened_directories: list[Path] = []
        self.resumed_dirs: list[Path] = []
        self.host = DialogHost(
            self.root,
            notebook=notebook,
            log_tab=log_tab,
            runs_root=Path("C:/runs"),
            open_directory=self.opened_directories.append,
            resume_run_dir=self.resumed_dirs.append,
        )

    def tearDown(self) -> None:
        self.root.destroy()

    def _button_labels(self, widget: tk.Misc) -> dict[str, tk.Widget]:
        labels: dict[str, tk.Widget] = {}
        for child in widget.winfo_children():
            if isinstance(child, ttk.Button):
                labels[str(child.cget("text"))] = child
            labels.update(self._button_labels(child))
        return labels

    def test_result_dialog_sets_title_and_is_transient(self) -> None:
        dialog = self.host.result_dialog("测试标题")
        try:
            self.assertEqual(dialog.title(), "测试标题")
        finally:
            dialog.destroy()

    def test_confirm_dialog_confirm_button_pushes_true(self) -> None:
        result_queue: queue.Queue[bool] = queue.Queue()
        self.host.show_confirm_start_dialog(5, 2, result_queue)
        dialog = self.root.winfo_children()[-1]
        try:
            self._button_labels(dialog)["开始"].invoke()
            self.assertTrue(result_queue.get_nowait())
        finally:
            if dialog.winfo_exists():
                dialog.destroy()

    def test_confirm_dialog_cancel_button_pushes_false(self) -> None:
        result_queue: queue.Queue[bool] = queue.Queue()
        self.host.show_confirm_force_reinstall_dialog(result_queue)
        dialog = self.root.winfo_children()[-1]
        try:
            self._button_labels(dialog)["取消"].invoke()
            self.assertFalse(result_queue.get_nowait())
        finally:
            if dialog.winfo_exists():
                dialog.destroy()

    def test_show_disk_precheck_error_calls_native_showerror(self) -> None:
        with mock.patch("cc_cover.gui.dialogs.messagebox.showerror") as showerror:
            self.host.show_disk_precheck_error("磁盘读取失败")

        showerror.assert_called_once_with(
            "磁盘预检失败", "磁盘读取失败", parent=self.root
        )

    def test_confirm_low_disk_space_returns_askyesno_result(self) -> None:
        check = DiskCheck(
            target=Path("C:/data"),
            required_bytes=10,
            free_bytes=1,
            sufficient=False,
        )
        with mock.patch(
            "cc_cover.gui.dialogs.messagebox.askyesno", return_value=True
        ) as askyesno:
            result = self.host.confirm_low_disk_space(check, 0)

        self.assertTrue(result)
        askyesno.assert_called_once()
        self.assertEqual(askyesno.call_args.kwargs["parent"], self.root)

    def test_failure_dialog_shows_provided_fields_without_resume_button(self) -> None:
        info = FailureInfo(stage="转写", reason="子进程失败", file="a.mp4")
        self.host.show_failure_dialog("字幕补全失败", info)
        dialog = self.root.winfo_children()[-1]
        try:
            labels = self._button_labels(dialog)
            self.assertIn("复制错误信息", labels)
            self.assertIn("查看日志", labels)
            self.assertNotIn("继续中断任务", labels)
        finally:
            dialog.destroy()

    def test_failure_dialog_view_log_switches_to_log_tab(self) -> None:
        info = FailureInfo(stage="转写", reason="子进程失败")
        self.host.show_failure_dialog("字幕补全失败", info)
        dialog = self.root.winfo_children()[-1]
        try:
            self._button_labels(dialog)["查看日志"].invoke()
            self.assertEqual(str(self.notebook.select()), str(self.log_tab))
        finally:
            if dialog.winfo_exists():
                dialog.destroy()

    def test_done_dialog_shows_warning_button_when_warnings_present(self) -> None:
        stats = CompletionStats(
            elapsed_seconds=90.0, written_count=3, warning_count=2, failed_count=0
        )
        self.host.show_done_dialog(
            "已完成", "全部处理完成", Path("C:/runs/run1"), stats
        )
        dialog = self.root.winfo_children()[-1]
        try:
            labels = self._button_labels(dialog)
            self.assertIn("2 条（点击查看 summary.txt）", labels)
        finally:
            dialog.destroy()

    def test_done_dialog_open_run_dir_forwards_to_callback(self) -> None:
        run_dir = Path("C:/runs/run1")
        self.host.show_done_dialog("已完成", "全部处理完成", run_dir, None)
        dialog = self.root.winfo_children()[-1]
        try:
            self._button_labels(dialog)["打开本次运行目录"].invoke()
            self.assertEqual(self.opened_directories, [run_dir])
        finally:
            if dialog.winfo_exists():
                dialog.destroy()


if __name__ == "__main__":
    unittest.main()
