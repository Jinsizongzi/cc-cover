from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from cc_cover.human_readable import format_duration, format_size
from cc_cover.models import (
    ErrorEvent,
    Event,
    Phase,
    ProgressEvent,
    RunDirEvent,
    event_from_dict,
)


@dataclass(frozen=True)
class FailureInfo:
    """任务失败/停止时用于对话框的结构化信息。"""

    stage: str
    reason: str
    file: str | None = None
    run_dir: Path | None = None
    done_count: int | None = None
    total_count: int | None = None


def parse_event_line(line: str) -> Event | str:
    """判断子进程输出的一行是结构化事件还是人读文字，是事件则解码。

    判据：该行能 json.loads 成功，且顶层对象带 "event" 键；否则（包括语法
    合法但没有 "event" 键的 JSON，以及未知 event 取值）一律原样返回该行。
    """
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return line
    if not isinstance(payload, dict) or "event" not in payload:
        return line
    try:
        return event_from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return line


def detect_stage(reason: str, fallback: str) -> str:
    """纯文本兜底：没有收到结构化 error 事件时，从错误文字里粗略猜阶段。

    典型触发场景：用户点「停止」，CLI 进程被硬杀，来不及吐出任何结构化
    事件；以及 install/环境检查这类本就不产出 JSON 事件的路径。收到结构化
    error 事件时优先用 phase_stage_label()，不经过这里。
    """
    if not reason:
        return fallback
    if "质量门禁" in reason:
        return "质量门禁"
    if "音频提取失败" in reason:
        return "音频提取"
    if "engine=funasr" in reason or "FunASR" in reason:
        return "FunASR 转写"
    if "engine=faster-whisper" in reason or "faster-whisper" in reason:
        return "faster-whisper 转写"
    if any(token in reason for token in ("写回", "替换", "备份", "回滚", "复核")):
        return "写回"
    if "扫描" in reason:
        return "扫描"
    return fallback


def failure_info(
    output: str,
    *,
    fallback_stage: str,
    reason: str | None = None,
    run_dir: Path | None = None,
) -> FailureInfo:
    """纯文本兜底：没有显式 reason 就把全部输出截断当 reason。

    服务不产出结构化事件的路径（install/环境检查、扫描）。转写/继续中断
    任务改由 failure_info_from_run() 优先从结构化 error 事件构造，只在
    没有捕获到结构化事件时才落回这里。
    """
    text = output or ""
    if reason is None:
        reason = text.strip()
        if len(reason) > 1200:
            reason = reason[-1200:]
    return FailureInfo(
        stage=detect_stage(reason, fallback_stage),
        reason=reason,
        file=None,
        run_dir=run_dir,
        done_count=None,
        total_count=None,
    )


def captured_events(output: str) -> list[Event]:
    """解析已收集输出的每一行，按原有顺序返回其中成功解码的结构化事件。"""
    events: list[Event] = []
    for line in (output or "").splitlines():
        parsed = parse_event_line(line)
        if not isinstance(parsed, str):
            events.append(parsed)
    return events


def run_dir_from_events(output: str) -> Path | None:
    """从已收集输出的结构化事件里取运行目录；取最后一个 run_dir 事件。"""
    resolved: Path | None = None
    for event in captured_events(output):
        if isinstance(event, RunDirEvent):
            resolved = Path(event.path)
    return resolved


def last_error_event(output: str) -> ErrorEvent | None:
    """从已收集输出的结构化事件里取最后一个 error 事件；没有则返回 None。"""
    last: ErrorEvent | None = None
    for event in captured_events(output):
        if isinstance(event, ErrorEvent):
            last = event
    return last


def last_progress_counts(output: str) -> tuple[int, int] | None:
    """从已收集输出的结构化事件里取最后一个 progress 事件的（第几个, 共几个）。"""
    last: ProgressEvent | None = None
    for event in captured_events(output):
        if isinstance(event, ProgressEvent):
            last = event
    return (last.index, last.total) if last is not None else None


_PHASE_STAGE_LABELS: Mapping[Phase, str] = {
    Phase.AUDIO_EXTRACT: "音频提取",
    Phase.FINGERPRINT: "视频完整性校验",
    Phase.FUNASR: "FunASR 转写",
    Phase.FASTER_WHISPER: "faster-whisper 转写",
    Phase.QUALITY_GATE: "质量门禁",
    Phase.WRITEBACK: "写回",
    Phase.VERIFY: "写回",
}


def phase_stage_label(phase: Phase, fallback: str) -> str:
    """结构化 error 事件的 phase 到失败对话框「阶段」文案。

    setup 没有比 fallback 更具体的既有标签（既有 detect_stage() 也是这样
    处理无法归类的原因文字的），沿用 fallback。
    """
    return _PHASE_STAGE_LABELS.get(phase, fallback)


def failure_info_from_command(
    chunks: Sequence[str],
    exc: BaseException,
    *,
    fallback_stage: str,
    run_dir: Path | None = None,
) -> FailureInfo:
    """从已收集输出与命令异常构建失败对话框信息。"""
    output = "".join(chunks)
    message = str(exc).strip()
    if message:
        output = output.rstrip("\n") + "\n" + message
    return failure_info(output, fallback_stage=fallback_stage, run_dir=run_dir)


def failure_info_from_run(
    chunks: Sequence[str],
    exc: BaseException,
    *,
    fallback_stage: str,
    run_dir: Path | None = None,
) -> FailureInfo:
    """转写/继续中断任务失败或停止时构造失败对话框信息。

    优先用已收集输出里最后一个结构化 error 事件直接构造；没有捕获到
    结构化事件时（典型场景：用户点「停止」，CLI 进程被硬杀，来不及吐出
    任何结构化事件）落回 failure_info_from_command() 的纯文本兜底。
    """
    output = "".join(chunks)
    error_event = last_error_event(output)
    if error_event is None:
        return failure_info_from_command(
            chunks, exc, fallback_stage=fallback_stage, run_dir=run_dir
        )
    counts = last_progress_counts(output)
    return FailureInfo(
        stage=phase_stage_label(error_event.phase, fallback_stage),
        reason=error_event.reason,
        file=error_event.video_path,
        run_dir=run_dir if run_dir is not None else run_dir_from_events(output),
        done_count=counts[0] if counts else None,
        total_count=counts[1] if counts else None,
    )


def first_failed_sample(run_dir: Path | None) -> tuple[str, str] | None:
    """质量门禁失败时，从运行目录读取第一个未通过的样例。"""
    if run_dir is None:
        return None
    report = run_dir / "stage_report.json"
    if not report.is_file():
        return None
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    for sample in payload.get("samples", []):
        if not isinstance(sample, dict) or bool(sample.get("passed", True)):
            continue
        video = str(sample.get("video_path") or "")
        errors = [str(item) for item in sample.get("errors", [])]
        reason = "；".join(errors) if errors else "质量门禁未通过"
        return video, reason
    return None


def run_is_resumable(run_dir: Path | None) -> bool:
    if run_dir is None:
        return False
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and str(payload.get("status")) != "committed"


@dataclass(frozen=True)
class ProgressSnapshot:
    current: int
    total: int
    percent: int
    elapsed_seconds: float
    remaining_seconds: float | None


class ProgressTracker:
    """跟踪结构化 progress 事件，得到「第 N / 共 M 个」与耗时/粗估剩余时间。

    「算完成」与「驱动显示」是两个刻意分开的口径：

    - current（第 N 个）：一个文件两个引擎（funasr / faster_whisper）都出现
      过才计为完成，反映真实的转写完成度，语义不能退让。
    - percent / remaining_seconds：批次内两个引擎是分两轮跑完的（先把这批
      全部跑完 funasr，再全部跑一遍 faster_whisper），如果百分比/剩余时间
      也按 current 算，funasr 单独跑的那一整轮里 current 完全不会涨，
      elapsed 却持续增长，ETA 公式必然单调走高到失真。这两者改用更细的
      步数计量——总步数 = 总数 × 2，任意一个引擎碰过一个候选就算一步，不
      要求两个都碰到——这样 funasr 单轮跑的时候也会持续平滑前进。

    因此 current/total 与 percent 衡量的不是同一件事，界面上可能长期不一致
    （比如「第 2 个」但百分比已经 60%），这是预期行为，不是两个数字打架。
    """

    _ENGINES = ("funasr", "faster_whisper")

    def __init__(self, total: int, started_at: float = 0.0):
        self.total = max(0, int(total))
        self.started_at = float(started_at)
        self._engines_by_path: dict[str, set[str]] = {}

    def on_event(self, event: Event | str) -> None:
        if not isinstance(event, ProgressEvent):
            return
        self._engines_by_path.setdefault(event.video_path, set()).add(event.engine)

    def _completed_count(self) -> int:
        required = set(self._ENGINES)
        return sum(
            1
            for engines in self._engines_by_path.values()
            if required <= engines
        )

    def _steps_touched(self) -> int:
        """已产出的 (视频, 引擎) 组合数——任一引擎碰过一个视频就算一步。"""
        return sum(len(engines) for engines in self._engines_by_path.values())

    def snapshot(self, now: float | None = None) -> ProgressSnapshot:
        current = self._completed_count()
        now = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, now - self.started_at)
        total_steps = self.total * len(self._ENGINES)
        steps = self._steps_touched()
        percent = round(steps / total_steps * 100) if total_steps > 0 else 0
        remaining: float | None = None
        if steps > 0 and total_steps > steps:
            remaining = (elapsed / steps) * (total_steps - steps)
        return ProgressSnapshot(
            current=current,
            total=self.total,
            percent=percent,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
        )


def progress_text(snapshot: ProgressSnapshot) -> str:
    """进度显示文本：第 N / 共 M 个 · 总体进度 · 已用时 · 约剩余。

    「第 N 个」与「总体进度」故意分开展示、不放进同一个括号——两者口径不同
    （见 ProgressTracker 的类文档），分开写清楚，不会看起来像同一个数字算
    错了。
    """
    parts = [
        f"第 {snapshot.current} / 共 {snapshot.total} 个",
        f"总体进度 {snapshot.percent}%",
        f"已用时 {format_duration(snapshot.elapsed_seconds)}",
    ]
    if snapshot.remaining_seconds is not None:
        parts.append(f"约剩余 {format_duration(snapshot.remaining_seconds)}")
    return " · ".join(parts)


_DOWNLOAD_HEADER_PATTERN = re.compile(
    r"Downloading\s+.*?\((?P<size>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]?B)\)",
    re.IGNORECASE,
)
_BAR_FRAME_PATTERN = re.compile(
    r"(?P<current>\d+(?:\.\d+)?)/(?P<total>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]?B)",
    re.IGNORECASE,
)
_BYTE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def _bytes_from_quantity(value: str, unit: str) -> int:
    """把「数值 + 单位」组合换算为字节；单位缺失时按字节处理。"""
    factor = _BYTE_UNITS.get(str(unit).upper(), 1)
    return int(round(float(value) * factor))


@dataclass(frozen=True)
class InstallProgressSnapshot:
    """安装进度快照：当前组件、已下载估算、总量与粗估剩余。"""

    component_index: int
    component_count: int
    downloaded_bytes: int
    total_bytes: int
    percent: int
    speed_bytes: float | None
    remaining_seconds: float | None


class InstallProgressTracker:
    """解析 pip 安装输出的下载信号，跟踪「组件 N/M + 已下载约 X + 约剩余」。

    管道环境下 pip 不输出进度条，唯一可靠的下载信号是 ``Downloading <包> (<大小>)``
    头：每个工件开始下载时宣布其大小，逐项累计即近似已下载量（标注「约」）。
    若个别环境出现 rich/tqdm 风格的 ``当前/总量`` 进度帧，则取其中的当前值。
    """

    def __init__(
        self,
        total_bytes: int,
        component_count: int,
        started_at: float = 0.0,
    ):
        self.total_bytes = max(0, int(total_bytes))
        self.component_count = max(0, int(component_count))
        self.component_index = 0
        self.started_at = float(started_at)
        self._announced_bytes = 0
        self._bar_bytes = 0

    def on_component(self, index: int) -> None:
        """工作线程进入下一个组件时调用，记录组件序号（从 1 开始）。"""
        self.component_index = max(0, int(index))

    def on_output(self, chunk: str) -> None:
        """消费 pip 输出片段，累计已下载估算。"""
        if not chunk:
            return
        for match in _DOWNLOAD_HEADER_PATTERN.finditer(chunk):
            self._announced_bytes += _bytes_from_quantity(
                match.group("size"), match.group("unit")
            )
        for match in _BAR_FRAME_PATTERN.finditer(chunk):
            current = _bytes_from_quantity(
                match.group("current"), match.group("unit")
            )
            if current > self._bar_bytes:
                self._bar_bytes = current

    def snapshot(self, now: float | None = None) -> InstallProgressSnapshot:
        now = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, now - self.started_at)
        downloaded = max(self._announced_bytes, self._bar_bytes)
        if self.total_bytes:
            downloaded = min(downloaded, self.total_bytes)
        else:
            downloaded = 0
        # 进度条百分比按组件完成度计算（已完成组件数 / 总数），避免在
        # 安装收尾阶段因下载量提前到 100% 造成误导；下载细节由文案展示。
        completed = max(0, self.component_index - 1)
        percent = (
            round(completed / self.component_count * 100)
            if self.component_count
            else 0
        )
        speed: float | None = None
        remaining: float | None = None
        if elapsed >= 2.0 and downloaded > 0:
            speed = downloaded / elapsed
            if self.total_bytes > downloaded:
                remaining = (self.total_bytes - downloaded) / speed
        return InstallProgressSnapshot(
            component_index=self.component_index,
            component_count=self.component_count,
            downloaded_bytes=downloaded,
            total_bytes=self.total_bytes,
            percent=percent,
            speed_bytes=speed,
            remaining_seconds=remaining,
        )


def install_progress_text(snapshot: InstallProgressSnapshot) -> str:
    """安装进度显示文本：组件 N/M · 已下载约 X · 约剩余。"""
    parts = [
        f"组件 {snapshot.component_index}/{snapshot.component_count}"
    ]
    if snapshot.downloaded_bytes > 0:
        parts.append(f"已下载约 {format_size(snapshot.downloaded_bytes)}")
    if snapshot.remaining_seconds is not None:
        parts.append(f"约剩余 {format_duration(snapshot.remaining_seconds)}")
    return " · ".join(parts)


def stopped_message(info: FailureInfo) -> str:
    """用户主动停止任务时，日志与提示框共用的文案。"""
    if run_is_resumable(info.run_dir):
        return "任务已停止，产物已暂存，可点击「继续中断任务」恢复。"
    if info.stage == "扫描":
        return "扫描已停止，未展示扫描结果，可重新扫描。"
    return "任务已停止，未产生可恢复的运行产物。"


def error_text(title: str, info: FailureInfo) -> str:
    lines = [title]
    if info.file:
        lines.append(f"文件：{info.file}")
    lines.append(f"阶段：{info.stage}")
    lines.append(f"原因：{info.reason}")
    if info.run_dir is not None:
        lines.append(f"运行目录：{info.run_dir}")
    if info.done_count is not None and info.total_count is not None:
        lines.append(
            f"已处理 {info.done_count}/{info.total_count} 个视频，产物已暂存。"
        )
    return "\n".join(lines) + "\n"
