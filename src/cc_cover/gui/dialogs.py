from __future__ import annotations

import os
import queue
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from cc_cover.core.pipeline import CompletionStats, SUMMARY_FILENAME
from cc_cover.gui.candidate_list import confirmation_text
from cc_cover.gui.human_readable import format_duration
from cc_cover.gui.progress import (
    FailureInfo,
    error_text,
    first_failed_sample,
    run_is_resumable,
    stopped_message,
)

PANEL = "#ffffff"


def enrich_failure(info: FailureInfo) -> FailureInfo:
    """质量门禁失败时，把第一个未通过样例的视频/原因补进 FailureInfo。"""
    if info.run_dir is None:
        return info
    if info.file and info.stage != "质量门禁":
        return info
    sample = first_failed_sample(info.run_dir)
    if sample is None:
        return info
    video, reason = sample
    return replace(
        info,
        file=info.file or video,
        reason=reason if info.stage == "质量门禁" else info.reason,
    )


class DialogHost:
    """结果 / 确认 / 失败对话框的共用外壳。

    只持有 CCCoverApp 需要转发进来的少量协作方（窗口、日志页签、运行目录
    回调），不反过来创建或引用整个 CCCoverApp——对话框内容本身（该显示
    哪些字段、按钮如何排布）因此可以脱离完整窗口单独构造。
    """

    def __init__(
        self,
        master: tk.Tk,
        *,
        notebook: ttk.Notebook,
        log_tab: ttk.Frame,
        runs_root: Path,
        open_directory: Callable[[Path], None],
        resume_run_dir: Callable[[Path], None],
    ) -> None:
        self.master = master
        self.notebook = notebook
        self.log_tab = log_tab
        self.runs_root = runs_root
        self._open_directory = open_directory
        self._resume_run_dir = resume_run_dir

    def result_dialog(self, title: str) -> tk.Toplevel:
        dialog = tk.Toplevel(self.master)
        dialog.title(title)
        dialog.configure(background=PANEL)
        dialog.resizable(False, False)
        dialog.transient(self.master)
        dialog.grab_set()
        dialog.update_idletasks()
        x = self.master.winfo_rootx() + max(
            0, (self.master.winfo_width() - dialog.winfo_reqwidth()) // 2
        )
        y = self.master.winfo_rooty() + max(
            0, (self.master.winfo_height() - dialog.winfo_reqheight()) // 3
        )
        dialog.geometry(f"+{x}+{y}")
        return dialog

    def dialog_row(self, parent: ttk.Frame, row: int, label: str, value: str) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").grid(
            row=row, column=0, sticky="nw", pady=(4, 0)
        )
        ttk.Label(
            parent,
            text=value,
            style="Body.TLabel",
            wraplength=500,
            justify="left",
        ).grid(row=row, column=1, sticky="nw", pady=(4, 0), padx=(10, 0))

    def _copy_error(self, dialog: tk.Toplevel, title: str, info: FailureInfo) -> None:
        self.master.clipboard_clear()
        self.master.clipboard_append(error_text(title, info))
        dialog.destroy()

    def _view_log(self, dialog: tk.Toplevel) -> None:
        self.notebook.select(self.log_tab)
        dialog.destroy()

    def _open_run_dir(self, dialog: tk.Toplevel, info: FailureInfo) -> None:
        target = info.run_dir if info.run_dir is not None else self.runs_root
        self._open_directory(target)
        dialog.destroy()

    def _resume_from_dialog(self, dialog: tk.Toplevel, info: FailureInfo) -> None:
        run_dir = info.run_dir
        dialog.destroy()
        if run_dir is not None and run_is_resumable(run_dir):
            self._resume_run_dir(run_dir)

    def _open_summary(self, run_dir: Path) -> None:
        summary = run_dir / SUMMARY_FILENAME
        try:
            os.startfile(str(summary))
        except OSError:
            messagebox.showwarning(
                "无法打开运行摘要",
                f"未找到运行摘要：{summary}",
                parent=self.master,
            )

    def show_confirm_dialog(
        self,
        title: str,
        body_text: str,
        *,
        confirm_label: str,
        result_queue: queue.Queue[bool],
    ) -> None:
        """两按钮确认框的共用外壳：confirm_label 触发 True，取消/关闭触发 False。"""
        dialog = self.result_dialog(title)
        body = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 18, 20, 6))
        body.pack(fill="x")
        ttk.Label(
            body,
            text=body_text,
            style="Body.TLabel",
            wraplength=460,
            justify="left",
        ).pack(anchor="w")

        def confirm() -> None:
            result_queue.put(True)
            dialog.destroy()

        def cancel() -> None:
            result_queue.put(False)
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", cancel)
        actions = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 12, 20, 18))
        actions.pack(fill="x")
        ttk.Button(
            actions, text=confirm_label, style="Primary.TButton", command=confirm
        ).pack(side="right")
        ttk.Button(actions, text="取消", style="Action.TButton", command=cancel).pack(
            side="right", padx=(0, 8)
        )

    def show_confirm_start_dialog(
        self,
        candidate_count: int,
        excluded_count: int,
        result_queue: queue.Queue[bool],
    ) -> None:
        self.show_confirm_dialog(
            "确认开始",
            confirmation_text(candidate_count, excluded_count),
            confirm_label="开始",
            result_queue=result_queue,
        )

    def show_confirm_force_reinstall_dialog(
        self, result_queue: queue.Queue[bool]
    ) -> None:
        """版本比对全部匹配、没有可精简重装的目标时，询问是否强制完整重装。

        用于版本号没变但文件本身损坏（磁盘错误、杀软误隔离等）这类版本比对
        查不出来的场景——给用户一个手动逃生舱，而不是让"安装 / 修复运行环境"
        在这种情况下悄悄变成什么都不做。
        """
        self.show_confirm_dialog(
            "强制完整重装？",
            (
                "当前依赖版本均已匹配，未检测到需要更新的项。\n\n"
                "如果怀疑运行环境本身有问题（比如文件损坏），"
                "可以选择强制完整重装。"
            ),
            confirm_label="强制完整重装",
            result_queue=result_queue,
        )

    def show_failure_dialog(self, title: str, info: FailureInfo) -> None:
        dialog = self.result_dialog(title)
        body = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 18, 20, 6))
        body.pack(fill="x")
        body.columnconfigure(1, weight=1)
        row = 0
        for label, value in (
            ("文件：", info.file),
            ("阶段：", info.stage),
            ("原因：", info.reason),
        ):
            if not value:
                continue
            self.dialog_row(body, row, label, value)
            row += 1
        if info.run_dir is not None:
            self.dialog_row(body, row, "运行目录：", str(info.run_dir))
            row += 1
        if info.done_count is not None and info.total_count is not None:
            note = (
                f"已处理 {info.done_count}/{info.total_count} 个视频，"
                "全部产物已暂存，可点击「继续中断任务」恢复，已完成的文件不会重跑。"
            )
            ttk.Label(body, text=note, style="Body.TLabel", wraplength=500).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(12, 0)
            )
        actions = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 12, 20, 18))
        actions.pack(fill="x")
        ttk.Button(
            actions,
            text="复制错误信息",
            style="Action.TButton",
            command=lambda: self._copy_error(dialog, title, info),
        ).pack(side="left")
        ttk.Button(
            actions,
            text="查看日志",
            style="Action.TButton",
            command=lambda: self._view_log(dialog),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text="打开运行目录",
            style="Action.TButton",
            command=lambda: self._open_run_dir(dialog, info),
        ).pack(side="left", padx=(8, 0))
        if run_is_resumable(info.run_dir):
            ttk.Button(
                actions,
                text="继续中断任务",
                style="Action.TButton",
                command=lambda: self._resume_from_dialog(dialog, info),
            ).pack(side="right")
        ttk.Button(
            actions, text="关闭", style="Action.TButton", command=dialog.destroy
        ).pack(side="right", padx=(0, 8))

    def show_stopped_dialog(self, info: FailureInfo) -> None:
        dialog = self.result_dialog("任务已停止")
        body = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 18, 20, 6))
        body.pack(fill="x")
        body.columnconfigure(1, weight=1)
        resumable = run_is_resumable(info.run_dir)
        ttk.Label(
            body, text=stopped_message(info), style="Body.TLabel", wraplength=500
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        if info.run_dir is not None:
            self.dialog_row(body, 1, "运行目录：", str(info.run_dir))
        actions = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 12, 20, 18))
        actions.pack(fill="x")
        if resumable:
            ttk.Button(
                actions,
                text="继续中断任务",
                style="Action.TButton",
                command=lambda: self._resume_from_dialog(dialog, info),
            ).pack(side="left")
        ttk.Button(
            actions,
            text="打开运行目录",
            style="Action.TButton",
            command=lambda: self._open_run_dir(dialog, info),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            actions, text="关闭", style="Action.TButton", command=dialog.destroy
        ).pack(side="right")

    def show_done_dialog(
        self,
        title: str,
        message: str,
        run_dir: Path | None,
        stats: CompletionStats | None = None,
    ) -> None:
        dialog = self.result_dialog(title)
        body = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 18, 20, 6))
        body.pack(fill="x")
        body.columnconfigure(1, weight=1)
        ttk.Label(
            body,
            text=message,
            style="Body.TLabel",
            wraplength=500,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        if stats is not None:
            self.dialog_row(body, 1, "总耗时：", format_duration(stats.elapsed_seconds))
            self.dialog_row(
                body, 2, "写回：", f"{stats.written_count} 个视频已生成并写回"
            )
            ttk.Label(body, text="告警：", style="Body.TLabel").grid(
                row=3, column=0, sticky="nw", pady=(4, 0)
            )
            if stats.warning_count:
                ttk.Button(
                    body,
                    text=f"{stats.warning_count} 条（点击查看 summary.txt）",
                    style="Action.TButton",
                    command=lambda: self._open_summary(run_dir),
                ).grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(4, 0))
            else:
                ttk.Label(body, text="无", style="Body.TLabel").grid(
                    row=3, column=1, sticky="nw", pady=(4, 0), padx=(10, 0)
                )
            if stats.failed_count:
                ttk.Label(body, text="处理失败：", style="Body.TLabel").grid(
                    row=4, column=0, sticky="nw", pady=(4, 0)
                )
                ttk.Button(
                    body,
                    text=f"{stats.failed_count} 个候选已跳过（点击查看 summary.txt）",
                    style="Action.TButton",
                    command=lambda: self._open_summary(run_dir),
                ).grid(row=4, column=1, sticky="w", padx=(10, 0), pady=(4, 0))
        actions = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 12, 20, 18))
        actions.pack(fill="x")
        if run_dir is not None:
            ttk.Button(
                actions,
                text="打开本次运行目录",
                style="Action.TButton",
                command=lambda: self._open_directory(run_dir),
            ).pack(side="left")
        ttk.Button(
            actions, text="完成", style="Action.TButton", command=dialog.destroy
        ).pack(side="right")
