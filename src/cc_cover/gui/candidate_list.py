from __future__ import annotations

import math
from typing import Any, Mapping


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
