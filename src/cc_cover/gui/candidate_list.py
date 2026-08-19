from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Mapping


def scan_confirmation_stats(report: Mapping[str, Any]) -> tuple[int, int]:
    """从扫描报告取开始前确认框的统计：(待处理 N, 已排除 M)。

    已排除数优先使用显式 ``excluded_count`` 字段；否则统计冲突条目中的视频总数
    （同 stem 冲突默认不处理、不写回）。
    """
    candidates = int(report.get("candidate_count") or 0)
    if "excluded_count" in report:
        excluded = int(report.get("excluded_count") or 0)
    else:
        excluded = sum(
            len(conflict.get("videos") or [])
            for conflict in report.get("conflicts") or []
            if isinstance(conflict, dict)
        )
    return candidates, excluded


def confirmation_text(candidate_count: int, excluded_count: int) -> str:
    """开始前确认框的文案：将处理 N 个视频并替换同名 TXT，含已排除数量。"""
    lines = [f"将处理 {candidate_count} 个视频并替换同名 TXT（替换前自动备份）。"]
    if excluded_count:
        lines.append(f"已排除 {excluded_count} 个视频，本次不处理。")
    else:
        lines.append("本次无已排除的视频。")
    return "\n".join(lines)


def estimate_processing_seconds(
    duration_seconds: float | None,
    size_bytes: int | None,
) -> int | None:
    """粗估单个候选的双模型处理耗时（秒）；无时长信息时返回 None。"""
    if duration_seconds is None:
        return None
    duration = max(0.0, float(duration_seconds))
    size = max(0, int(size_bytes or 0))
    seconds = 30.0 + duration * 0.30 + size / (8 * 1024 * 1024)
    return int(math.ceil(seconds))


def format_column_duration(seconds: float | None) -> str:
    """候选列表「时长」列的 H:MM:SS 格式；与进度条的中文 format_duration 区分。"""
    if seconds is None:
        return "—"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_column_size(size_bytes: int | None) -> str:
    """候选列表「大小」列格式；与磁盘占用的 format_size 区分（None 显示 —）。"""
    if size_bytes is None:
        return "—"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return "—"


def format_estimate(seconds: int | None) -> str:
    if seconds is None:
        return "—"
    minutes = max(0, seconds) / 60.0
    if minutes < 1:
        return f"{seconds} 秒"
    if minutes < 60:
        return f"约 {int(math.ceil(minutes))} 分钟"
    return f"约 {minutes / 60.0:.1f} 小时"


def selection_summary(
    *,
    video_count: int,
    candidate_count: int,
    selected_count: int,
    conflict_count: int,
    protected_count: int,
) -> str:
    excluded = max(0, candidate_count - selected_count)
    return (
        f"视频 {video_count} 个 · 待补全 {candidate_count} 个 · "
        f"已选 {selected_count} 个 · 已排除 {excluded} 个 · "
        f"冲突 {conflict_count} 个 · 受保护非空 TXT {protected_count} 个"
    )


class CandidateListPanel:
    """候选列表屏幕的状态与交互：勾选/排除、冲突标记、右键菜单、统计文案。

    只接收构造时传入的 Treeview 与 BooleanVar，不反过来创建或持有整个
    CCCoverApp——这样候选列表的状态转换（比如"全选"要不要连带同步）可以
    脱离完整窗口单独构造、单独测试。
    """

    def __init__(
        self,
        tree: ttk.Treeview,
        select_all_var: tk.BooleanVar,
        *,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.tree = tree
        self.select_all_var = select_all_var
        self._on_change = on_change
        self.last_report: dict[str, Any] | None = None
        self.checked_paths: set[str] = set()
        self.candidate_row_video: dict[str, str] = {}
        self.original_state_by_row: dict[str, str] = {}
        self.conflict_row_ids: set[str] = set()

    def load(self, report: dict[str, Any]) -> None:
        """用一份新的扫描报告重建整张候选列表；默认全选，冲突项单独标记。"""
        self.last_report = report
        self.checked_paths.clear()
        self.candidate_row_video.clear()
        self.original_state_by_row.clear()
        self.conflict_row_ids.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for candidate in report.get("candidates", []):
            video_path = str(candidate.get("video_path", ""))
            duration = candidate.get("video_duration_s")
            size = candidate.get("video_size")
            estimate = estimate_processing_seconds(duration, size)
            row_id = self.tree.insert(
                "",
                "end",
                values=(
                    candidate.get("state", ""),
                    video_path,
                    candidate.get("target_path", ""),
                    format_column_duration(duration),
                    format_column_size(size),
                    format_estimate(estimate),
                    "MM:SS / H:MM:SS",
                ),
            )
            self.candidate_row_video[row_id] = video_path
            self.original_state_by_row[row_id] = str(candidate.get("state", ""))
            self.checked_paths.add(video_path)
            self._set_checkbox(row_id, True)
        for conflict in report.get("conflicts", []):
            for video in conflict.get("videos", []):
                row_id = self.tree.insert(
                    "",
                    "end",
                    values=(
                        "冲突",
                        video,
                        conflict.get("target_path", ""),
                        "—",
                        "—",
                        "—",
                        "—",
                    ),
                    tags=("conflict",),
                )
                self.conflict_row_ids.add(row_id)
        self._sync_select_all()
        self._notify()

    def _set_checkbox(self, row_id: str, checked: bool) -> None:
        self.tree.item(row_id, text="☑" if checked else "☐")

    def _set_row_state(self, row_id: str, state: str, tags: tuple[str, ...]) -> None:
        values = list(self.tree.item(row_id, "values"))
        values[0] = state
        self.tree.item(row_id, values=tuple(values), tags=tags)

    def toggle(self, row_id: str) -> None:
        video = self.candidate_row_video.get(row_id)
        if video is None:
            return
        self._apply_checked(row_id, video not in self.checked_paths)
        self._sync_select_all()
        self._notify()

    def toggle_all(self) -> None:
        select_all = bool(self.select_all_var.get())
        for row_id in list(self.candidate_row_video):
            self._apply_checked(row_id, select_all)
        self._notify()

    def _apply_checked(self, row_id: str, checked: bool) -> None:
        video = self.candidate_row_video[row_id]
        self._set_checkbox(row_id, checked)
        if checked:
            self.checked_paths.add(video)
            self._set_row_state(row_id, self.original_state_by_row.get(row_id, ""), ())
        else:
            self.checked_paths.discard(video)
            self._set_row_state(row_id, "已排除", ("excluded",))

    def _sync_select_all(self) -> None:
        self.select_all_var.set(
            all(
                video in self.checked_paths
                for video in self.candidate_row_video.values()
            )
        )

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def summary_text(self) -> str:
        report = self.last_report or {}
        return selection_summary(
            video_count=int(report.get("video_count", 0)),
            candidate_count=int(report.get("candidate_count", 0)),
            selected_count=len(self.checked_paths),
            conflict_count=int(report.get("conflict_count", 0)),
            protected_count=int(report.get("protected_nonempty_txt_count", 0)),
        )

    def excluded_paths(self) -> list[str]:
        return sorted(
            video
            for row_id, video in self.candidate_row_video.items()
            if video not in self.checked_paths
        )

    def is_conflict(self, row_id: str) -> bool:
        return row_id in self.conflict_row_ids

    def on_click(self, event: Any) -> None:
        row_id = self.tree.identify_row(event.y)
        if not row_id or row_id in self.conflict_row_ids:
            return
        if self.tree.identify_column(event.x) != "#0":
            return
        self.toggle(row_id)

    def show_context_menu(
        self, master: tk.Misc, event: Any, *, open_in_explorer: Callable[[str], None]
    ) -> None:
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return
        self.tree.selection_set(row_id)
        self.tree.focus(row_id)
        values = self.tree.item(row_id, "values")
        video = str(values[1]) if len(values) > 1 else ""
        target = str(values[2]) if len(values) > 2 else ""
        menu = tk.Menu(master, tearoff=0)
        if row_id in self.candidate_row_video:
            excluded = self.candidate_row_video[row_id] not in self.checked_paths
            menu.add_command(
                label="恢复" if excluded else "从本次处理中排除",
                command=lambda: self.toggle(row_id),
            )
            menu.add_separator()
        menu.add_command(
            label="打开视频所在位置",
            command=lambda: open_in_explorer(video),
        )
        menu.add_command(
            label="打开目标 TXT 所在位置",
            command=lambda: open_in_explorer(target),
        )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
