from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from cc_cover.core.pipeline.io import load_optional_json, write_bytes_atomic

SUMMARY_FILENAME = "summary.txt"

RUN_STATUS_LABELS = {
    "prepared": "已准备",
    "running": "运行中",
    "staged_all": "已暂存（全部候选）",
    "staged_partial": "已暂存（部分候选）",
    "committed": "已完成",
    "unknown": "未知",
}


def run_status_label(status: str) -> str:
    """运行状态的用户可读标签；未知状态原样透传。"""
    return RUN_STATUS_LABELS.get(status, status)


def _sample_line(sample: dict[str, Any]) -> str:
    return "{} {}".format(
        sample.get("sample_id", ""), sample.get("video_path", "")
    ).strip()


@dataclass(frozen=True)
class _RunFacts:
    """build_summary_text() 派生出的统计量，供下面几个小节格式化函数共享。"""

    run_id: str
    status: str
    started: Any
    ended: Any
    candidate_count: int
    excluded_count: int
    candidate_failure_count: int
    passed_count: int
    failed_count: int
    committed_count: int
    warning_sample_count: int
    warning_count: int
    verification: dict[str, Any] | None
    verify_failures: list[str]


def _summary_header_lines(facts: _RunFacts) -> list[str]:
    lines = [
        "CC-Cover 运行摘要",
        "================",
        "",
        f"运行 ID：{facts.run_id}",
        f"状态：{run_status_label(facts.status)}（{facts.status}）",
        f"开始时间：{facts.started}",
        f"结束时间：{facts.ended}",
        "",
        "统计",
        "----",
        f"候选总数：{facts.candidate_count}",
        f"已排除：{facts.excluded_count}（本次不处理）",
        f"处理失败（已跳过）：{facts.candidate_failure_count}",
        f"质量门禁通过：{facts.passed_count}",
        f"质量门禁失败：{facts.failed_count}",
        f"写回成功：{facts.committed_count}",
        f"告警：{facts.warning_sample_count} 个视频，共 {facts.warning_count} 条",
    ]
    if facts.verification is not None:
        if facts.verify_failures:
            lines.append(f"最终复核：失败（{len(facts.verify_failures)} 项）")
        else:
            lines.append(
                f"最终复核：{int(facts.verification.get('verified_count') or 0)} 项通过"
            )
    return lines


def _summary_warning_lines(warning_samples: Sequence[dict[str, Any]]) -> list[str]:
    lines = ["", "告警明细", "--------"]
    if warning_samples:
        for sample in warning_samples:
            lines.append(_sample_line(sample))
            lines.extend(f"  - {warning}" for warning in sample.get("warnings") or [])
    else:
        lines.append("（无）")
    return lines


def _summary_skipped_lines(
    candidate_failure_dicts: Sequence[dict[str, Any]],
) -> list[str]:
    lines = ["", "处理失败（已跳过）明细", "--------"]
    if candidate_failure_dicts:
        for failure in candidate_failure_dicts:
            lines.append(
                "{} {}".format(
                    failure.get("sample_id", ""), failure.get("video_path", "")
                ).strip()
            )
            lines.append(f"  - {failure.get('reason', '')}")
    else:
        lines.append("（无）")
    return lines


def _summary_failure_lines(
    failed_samples: Sequence[dict[str, Any]], verify_failures: Sequence[str]
) -> list[str]:
    lines = ["", "失败明细", "--------"]
    failure_lines: list[str] = []
    for sample in failed_samples:
        failure_lines.append(_sample_line(sample))
        failure_lines.extend(f"  - {error}" for error in sample.get("errors") or [])
    failure_lines.extend(f"  - {failure}" for failure in verify_failures)
    lines.extend(failure_lines if failure_lines else ["（无）"])
    return lines


def _summary_writeback_lines(entries: Sequence[dict[str, Any]]) -> list[str]:
    lines = ["", "写回清单", "--------"]
    if entries:
        for entry in entries:
            sample_id = entry.get("sample_id", "")
            target = entry.get("target_path", "")
            size = entry.get("target_size", "?")
            lines.append(f"{sample_id} {target}（{size} 字节）".strip())
    else:
        lines.append("（未写回）")
    return lines


def build_summary_text(run_dir: Path) -> str:
    """从运行产物生成人读摘要文本；产物缺失时以 0 / 未知降级。"""
    run_dir = run_dir.expanduser().resolve()
    manifest = load_optional_json(run_dir / "manifest.json") or {}
    stage = load_optional_json(run_dir / "stage_report.json")
    commit = load_optional_json(run_dir / "commit_report.json")
    verification = load_optional_json(run_dir / "verification.json")

    run_id = str(manifest.get("run_id") or run_dir.name)
    status = str(manifest.get("status") or "unknown")
    started = manifest.get("created_at_utc")
    ended = None
    if commit is not None:
        ended = commit.get("committed_at_utc")
    if ended is None:
        ended = manifest.get("updated_at_utc")
    if ended is None:
        ended = started

    candidates = manifest.get("candidates") or []
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    excluded_videos = manifest.get("excluded_videos") or []
    excluded_count = len(excluded_videos) if isinstance(excluded_videos, list) else 0
    discovery = manifest.get("discovery")
    if isinstance(discovery, dict):
        # 已排除 = 发现视频总数 - 候选数（同 stem 冲突、显式排除项均不进入候选）。
        video_count = int(discovery.get("video_count") or 0)
        discovered_candidates = int(discovery.get("candidate_count") or 0)
        if video_count and discovered_candidates:
            excluded_count = max(0, video_count - discovered_candidates)

    samples = (stage.get("samples") or []) if stage else []
    sample_dicts = [item for item in samples if isinstance(item, dict)]
    passed_samples = [item for item in sample_dicts if bool(item.get("passed"))]
    failed_samples = [item for item in sample_dicts if not bool(item.get("passed"))]
    warning_samples = [item for item in sample_dicts if item.get("warnings")]
    warning_count = sum(int(item.get("warning_count") or 0) for item in warning_samples)

    entries = (commit.get("entries") or []) if commit else []
    committed_count = len(entries) if isinstance(entries, list) else 0

    verify_failures: list[str] = []
    if verification is not None and not bool(verification.get("passed", True)):
        verify_failures = [str(item) for item in verification.get("failures") or []]

    candidate_failures = manifest.get("candidate_failures") or []
    candidate_failure_dicts = [
        item for item in candidate_failures if isinstance(item, dict)
    ]

    facts = _RunFacts(
        run_id=run_id,
        status=status,
        started=started,
        ended=ended,
        candidate_count=candidate_count,
        excluded_count=excluded_count,
        candidate_failure_count=len(candidate_failure_dicts),
        passed_count=len(passed_samples),
        failed_count=len(failed_samples),
        committed_count=committed_count,
        warning_sample_count=len(warning_samples),
        warning_count=warning_count,
        verification=verification,
        verify_failures=verify_failures,
    )

    lines = _summary_header_lines(facts)
    lines += _summary_warning_lines(warning_samples)
    lines += _summary_skipped_lines(candidate_failure_dicts)
    lines += _summary_failure_lines(failed_samples, verify_failures)
    lines += _summary_writeback_lines(entries)
    lines += ["", "路径", "----", f"运行目录：{run_dir}"]
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path) -> Path:
    """在运行目录写入 summary.txt（UTF-8），返回其路径。"""
    run_dir = run_dir.expanduser().resolve()
    path = run_dir / SUMMARY_FILENAME
    write_bytes_atomic(path, build_summary_text(run_dir).encode("utf-8"))
    return path


@dataclass(frozen=True)
class CompletionStats:
    """完成弹窗展示的运行结果统计。"""

    elapsed_seconds: float | None
    written_count: int
    warning_count: int
    failed_count: int = 0


def _timestamp_epoch(value: Any) -> float | None:
    """ISO 时间戳转 epoch 秒；缺失或不可解析时返回 None。"""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def run_completion_stats(run_dir: Path) -> CompletionStats:
    """从运行产物读取完成弹窗需要的统计：总耗时、写回数、告警数。"""
    run_dir = run_dir.expanduser().resolve()
    manifest = load_optional_json(run_dir / "manifest.json")
    commit = load_optional_json(run_dir / "commit_report.json")
    stage = load_optional_json(run_dir / "stage_report.json")

    started = _timestamp_epoch(manifest.get("created_at_utc") if manifest else None)
    ended_raw = None
    if commit is not None:
        ended_raw = commit.get("committed_at_utc")
    if ended_raw is None and manifest is not None:
        ended_raw = manifest.get("updated_at_utc")
    ended = _timestamp_epoch(ended_raw)
    elapsed = ended - started if started is not None and ended is not None else None

    written = 0
    if commit is not None:
        written = int(commit.get("entry_count") or 0)
        entries = commit.get("entries")
        if not written and isinstance(entries, list):
            written = len(entries)

    warnings = int(stage.get("warning_count") or 0) if stage else 0
    candidate_failures = manifest.get("candidate_failures") if manifest else None
    failed_count = (
        len(candidate_failures) if isinstance(candidate_failures, list) else 0
    )
    return CompletionStats(
        elapsed_seconds=elapsed,
        written_count=written,
        warning_count=warnings,
        failed_count=failed_count,
    )
