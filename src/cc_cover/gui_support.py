from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cc_cover.human_readable import format_duration, format_size
from cc_cover.models import (
    ErrorEvent,
    Event,
    Phase,
    ProgressEvent,
    RunDirEvent,
    event_from_dict,
)


APP_DATA_DIRECTORY = "CC-Cover"
SETTINGS_FILENAME = "settings.json"
# 数据根固定子目录：与 README/安装器卸载清单保持一致，改动需同步
# packaging/CC-Cover.iss 的 [UninstallDelete] 与 tests/test_packaging.py。
DATA_ROOT_SUBDIRECTORIES = ("venv", "model-cache", "runs", "temp")
DATA_ROOT_KEY = "data_root"
TORCH_VERSION = "2.5.1"
ASR_DEPENDENCIES = (
    "imageio-ffmpeg>=0.6,<1",
    "funasr==1.3.16",
    "modelscope==1.38.1",
    "faster-whisper==1.2.1",
    "ctranslate2==4.8.1",
    "numpy>=1.26,<2",
    "soundfile>=0.12,<1",
)

CLEANUP_WARNING_BYTES = 5 * 1024**3

# 磁盘预检与安装进度使用的字节估算（仅用于提示与粗估剩余，非精确计量）。
# torch + torchaudio 轮子：CPU 约 330MB，CUDA cu121 约 2.7GB，均留余量。
TORCH_CPU_BYTES = 400 * 1024**2
TORCH_CUDA_BYTES = 3200 * 1024**2
# 默认模型：FunASR（paraformer-large + fsmn-vad + ct-punc）约 2GB，large-v3-turbo 约 1.6GB。
FUNASR_MODELS_BYTES = 2500 * 1024**2
FAST_WHISPER_MODELS_BYTES = 1600 * 1024**2
# 其余 ASR 依赖轮子（funasr、faster-whisper、ctranslate2 等）约 250MB。
ASR_DEPENDENCIES_BYTES = 400 * 1024**2
# venv/临时目录开销与磁盘碎片余量。
INSTALL_BUFFER_BYTES = 1024**3


def install_download_bytes(device: str) -> int:
    """安装阶段 pip 实际下载量的估算（torch 轮子 + ASR 依赖，不含模型）。

    用于安装进度条的总量与剩余时间估算；与``estimate_install_required_bytes``
    不同——后者额外计入首次运行需下载的模型与余量，用于磁盘预检。
    """
    torch_bytes = TORCH_CUDA_BYTES if device == "cuda" else TORCH_CPU_BYTES
    return torch_bytes + ASR_DEPENDENCIES_BYTES


def estimate_install_required_bytes(device: str) -> int:
    """安装运行环境并下载默认模型的总需求估算，用于磁盘预检提示。"""
    return (
        install_download_bytes(device)
        + FUNASR_MODELS_BYTES
        + FAST_WHISPER_MODELS_BYTES
        + INSTALL_BUFFER_BYTES
    )


@dataclass(frozen=True)
class DiskCheck:
    """目标盘剩余空间预检结果：所需、剩余与是否充足。"""

    target: Path
    required_bytes: int
    free_bytes: int
    sufficient: bool


def disk_precheck(directory: Path, required_bytes: int) -> DiskCheck:
    """检查目标目录所在磁盘能否容纳所需字节；目标盘随数据根联动。

    目标目录尚不存在时先创建（数据根首次安装前可能尚未建立）。
    """
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    required = max(0, int(required_bytes))
    return DiskCheck(
        target=target,
        required_bytes=required,
        free_bytes=int(usage.free),
        sufficient=int(usage.free) >= required,
    )


def disk_precheck_text(check: DiskCheck, runs_bytes: int = 0) -> str:
    """磁盘预检提示文案：至少需要 N，当前剩余 M；不足时建议先清理运行目录。"""
    lines = [
        f"至少需要 {format_size(check.required_bytes)}，"
        f"目标盘当前剩余 {format_size(check.free_bytes)}。"
    ]
    if check.sufficient:
        lines.append("磁盘空间充足。")
        return "\n".join(lines)
    lines.append("磁盘空间不足。")
    if runs_bytes > 0:
        lines.append(
            f"建议先清理运行目录（当前占用 {format_size(runs_bytes)}）后再尝试。"
        )
    else:
        lines.append("建议清理该磁盘上的其他文件后再尝试。")
    return "\n".join(lines)


@dataclass(frozen=True)
class FailureInfo:
    """任务失败/停止时用于对话框的结构化信息。"""

    stage: str
    reason: str
    file: str | None = None
    run_dir: Path | None = None
    done_count: int | None = None
    total_count: int | None = None


class SettingsError(RuntimeError):
    """设置文件无效（JSON 损坏或顶层不是对象）时抛出。"""


@dataclass(frozen=True)
class RuntimePaths:
    source_root: Path
    data_root: Path
    venv_root: Path
    venv_python: Path
    model_cache: Path
    runs_root: Path
    temp_root: Path


@dataclass(frozen=True)
class DataRootResolution:
    root: Path
    needs_choice: bool


@dataclass(frozen=True)
class GuiOptions:
    device: str = "auto"
    hash_videos: bool = True
    ffmpeg: Path | None = None


DEVICE_CHOICES = ("auto", "cuda", "cpu")
GUI_DEVICE_CHOICES = ("cuda", "cpu")


@dataclass(frozen=True)
class GuiSettings:
    """GUI 用户偏好，持久化到数据根下的 settings.json。

    device 是唯一的运行设备设置；旧版 accelerator 键在读取时迁移。
    """

    scan_path: str = ""
    device: str = "auto"
    ffmpeg: str = ""
    hash_videos: bool = True


def resolve_default_device(saved: str, detected: str | None) -> str:
    """启动默认设备：已保存的明确选择优先；否则跟随检测结果，回退 CPU。"""
    if saved in GUI_DEVICE_CHOICES:
        return saved
    return detected if detected in GUI_DEVICE_CHOICES else "cpu"


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
class RunInfo:
    """运行目录清理列表中的一行：标识、状态与占用。"""

    run_id: str
    path: Path
    status: str
    size_bytes: int
    created_at_utc: str | None = None


def _manifest_status(run_dir: Path) -> tuple[str, str | None]:
    """读取运行目录 manifest 的状态与创建时间；缺失或无效时视为 unknown。"""
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return "unknown", None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return "unknown", None
    if not isinstance(payload, dict):
        return "unknown", None
    status = payload.get("status")
    created = payload.get("created_at_utc")
    return (
        str(status) if isinstance(status, str) else "unknown",
        str(created) if isinstance(created, str) else None,
    )


def directory_size(path: Path) -> int:
    """递归统计目录占用字节数；无法读取的文件跳过。"""
    total = 0
    for item in Path(path).rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def list_runs(runs_root: Path) -> list[RunInfo]:
    """枚举运行根下的运行目录，按名称排序；跳过普通文件。"""
    root = Path(runs_root).expanduser().resolve()
    runs: list[RunInfo] = []
    if not root.is_dir():
        return runs
    for child in sorted(root.iterdir(), key=lambda item: str(item.name).casefold()):
        if not child.is_dir():
            continue
        status, created_at = _manifest_status(child)
        resolved = child.resolve()
        runs.append(
            RunInfo(
                run_id=child.name,
                path=resolved,
                status=status,
                size_bytes=directory_size(resolved),
                created_at_utc=created_at,
            )
        )
    return runs


def runs_total_size(runs: Sequence[RunInfo]) -> int:
    """所有列出的运行目录总占用字节数。"""
    return sum(run.size_bytes for run in runs)


def delete_runs(runs: Sequence[RunInfo]) -> int:
    """删除选中的运行目录；跳过已不存在或不存在的路径，返回实际删除数。"""
    deleted = 0
    for run in runs:
        if not run.path.is_dir():
            continue
        shutil.rmtree(run.path)
        deleted += 1
    return deleted


def _reset_directory(directory: Path) -> None:
    """删除目录内容并重建为空目录；目录不存在时直接创建。"""
    target = Path(directory)
    if target.is_dir():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def clean_model_cache(paths: RuntimePaths) -> None:
    """删除模型缓存目录并重建；下次运行需重新下载模型。"""
    _reset_directory(paths.model_cache)


def local_data_usage(paths: RuntimePaths) -> int:
    """venv、模型缓存、运行记录与临时目录的总占用字节数。"""
    directories = (
        paths.venv_root,
        paths.model_cache,
        paths.runs_root,
        paths.temp_root,
    )
    return sum(directory_size(directory) for directory in directories)


def clear_local_data(paths: RuntimePaths) -> None:
    """删除 venv、模型缓存、运行记录与临时目录并重建。

    数据根本身与其中的 settings.json 保留（数据根指针不能丢）。
    """
    for directory in (
        paths.venv_root,
        paths.model_cache,
        paths.runs_root,
        paths.temp_root,
    ):
        _reset_directory(directory)


def model_cache_cleanup_text(size_bytes: int) -> str:
    """清理模型缓存的确认文案：删除需重新下载。"""
    return (
        f"将删除模型缓存（共 {format_size(size_bytes)}），"
        "下次运行需重新下载模型。\n\n确定继续吗？"
    )


def clear_all_data_text(usage_bytes: int) -> str:
    """清理全部本地数据的确认文案：列出删除项并显示总占用。"""
    lines = [
        f"将删除 venv、模型缓存、运行记录与临时文件"
        f"（共 {format_size(usage_bytes)}），不可恢复。",
        "",
        "删除后需重新安装运行环境，并重新下载模型。",
        "确定继续吗？",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class ProgressSnapshot:
    current: int
    total: int
    percent: int
    elapsed_seconds: float
    remaining_seconds: float | None


class ProgressTracker:
    """跟踪结构化 progress 事件，得到「第 N / 共 M 个」与耗时/粗估剩余时间。

    每个视频在双模型（funasr / faster_whisper）各产出一个 progress 事件；
    一个文件两个引擎都出现过才计为完成，百分比因此反映真实的转写完成度，
    而不是"已开始"。剩余时间按「平均耗时 × 剩余数」粗估。
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

    def snapshot(self, now: float | None = None) -> ProgressSnapshot:
        current = self._completed_count()
        now = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, now - self.started_at)
        percent = round(current / self.total * 100) if self.total > 0 else 0
        remaining: float | None = None
        if current > 0 and self.total > current:
            remaining = (elapsed / current) * (self.total - current)
        return ProgressSnapshot(
            current=current,
            total=self.total,
            percent=percent,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
        )


def progress_text(snapshot: ProgressSnapshot) -> str:
    """进度显示文本：第 N / 共 M 个（百分比）· 已用时 · 约剩余。"""
    parts = [
        f"第 {snapshot.current} / 共 {snapshot.total} 个（{snapshot.percent}%）",
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


RUN_SOUND_MIN_SECONDS = 5 * 60


def should_play_completion_sound(elapsed_seconds: float | None) -> bool:
    """运行超过 5 分钟才播放完成提示音。"""
    return elapsed_seconds is not None and elapsed_seconds > RUN_SOUND_MIN_SECONDS


def play_completion_sound() -> None:
    """播放 Windows 提示音；其他平台或失败时静默跳过。"""
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_ICONASTERISK)
    except (ImportError, OSError, RuntimeError):
        pass


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


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    """立即终止进程及其子进程树（Windows 用 taskkill，其他平台退化为 terminate）。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
            return
        except OSError:
            pass
    process.terminate()


def default_data_root(
    *,
    frozen: bool | None = None,
    bundle_root: Path | None = None,
    app_dir: Path | None = None,
) -> Path:
    """数据根默认位置：打包后为 exe 所在目录，开发模式为项目根目录。"""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if app_dir is not None:
        return Path(app_dir).expanduser().resolve()
    if is_frozen:
        return Path(sys.executable).resolve().parent
    if bundle_root is None:
        bundle_root = Path(__file__).resolve().parents[2]
    return Path(bundle_root).resolve()


def runtime_paths(
    *,
    frozen: bool | None = None,
    bundle_root: Path | None = None,
    data_root: Path | None = None,
) -> RuntimePaths:
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if bundle_root is None:
        if is_frozen:
            bundle_root = Path(str(getattr(sys, "_MEIPASS"))).resolve()
        else:
            bundle_root = Path(__file__).resolve().parents[2]
    source_root = (
        bundle_root / "src" if is_frozen else bundle_root.resolve() / "src"
    ).resolve()
    if data_root is None:
        data_root = default_data_root(frozen=is_frozen, bundle_root=bundle_root)
    resolved_data_root = Path(data_root).expanduser().resolve()
    venv_root = resolved_data_root / "venv"
    return RuntimePaths(
        source_root=source_root,
        data_root=resolved_data_root,
        venv_root=venv_root,
        venv_python=venv_root / "Scripts" / "python.exe",
        model_cache=resolved_data_root / "model-cache",
        runs_root=resolved_data_root / "runs",
        temp_root=resolved_data_root / "temp",
    )


def ensure_data_root(paths: RuntimePaths) -> None:
    """创建数据根与固定子目录：venv、model-cache、runs、temp。"""
    paths.data_root.mkdir(parents=True, exist_ok=True)
    for name in DATA_ROOT_SUBDIRECTORIES:
        (paths.data_root / name).mkdir(parents=True, exist_ok=True)


def settings_file(data_root: Path) -> Path:
    """数据根内固定位置：<data_root>/settings.json。"""
    return Path(data_root).expanduser().resolve() / SETTINGS_FILENAME


def read_settings(data_root: Path) -> dict[str, Any]:
    """读取数据根下的设置文件；文件不存在时返回空字典。"""
    path = settings_file(data_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, NotADirectoryError):
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise SettingsError(f"设置文件无效：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SettingsError(f"设置文件顶层必须是 JSON 对象：{path}")
    return value


def write_settings(data_root: Path, values: Mapping[str, Any]) -> None:
    """原子写入数据根下的设置文件（UTF-8）。"""
    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = settings_file(root)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(values), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_gui_settings(data_root: Path) -> GuiSettings:
    """读取数据根下 settings.json 中的 GUI 偏好；缺失或非法字段回退默认值。"""
    values = read_settings(data_root)
    scan_path = values.get("scan_path")
    device = values.get("device")
    accelerator = values.get("accelerator")
    if accelerator in GUI_DEVICE_CHOICES:
        # 旧版 accelerator 是“运行环境”选择（NVIDIA GPU / CPU），合并后优先迁移。
        device = accelerator
    elif device not in DEVICE_CHOICES:
        device = "auto"
    ffmpeg = values.get("ffmpeg")
    hash_videos = values.get("hash_videos")
    return GuiSettings(
        scan_path=str(scan_path) if isinstance(scan_path, str) else "",
        device=device,
        ffmpeg=str(ffmpeg) if isinstance(ffmpeg, str) else "",
        hash_videos=hash_videos if isinstance(hash_videos, bool) else True,
    )


def save_gui_settings(data_root: Path, settings: GuiSettings) -> None:
    """合并写入 GUI 偏好，移除旧版 accelerator 键，保留 data_root 等既有键。"""
    values = read_settings(data_root)
    values.pop("accelerator", None)
    values.update(
        {
            "scan_path": settings.scan_path,
            "device": settings.device,
            "ffmpeg": settings.ffmpeg,
            "hash_videos": settings.hash_videos,
        }
    )
    write_settings(data_root, values)


def is_writable(directory: Path) -> bool:
    """探测目录可写性：实际创建并删除一个探测文件，失败返回 False。"""
    target = Path(directory).expanduser().resolve()
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / f".cc-cover-write-probe-{uuid.uuid4().hex}"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def fallback_pointer_root() -> Path:
    """默认根不可写时，数据根指针的回退位置（用户本地应用数据目录）。"""
    environment = os.environ.get("LOCALAPPDATA")
    local = Path(environment) if environment else Path.home() / "AppData" / "Local"
    return (local / APP_DATA_DIRECTORY).resolve()


def pointer_root(default_root: Path) -> Path:
    """数据根指针所在目录：默认根可写时用默认根，否则用用户可写的回退位置。"""
    default_root = Path(default_root).expanduser().resolve()
    if settings_file(default_root).exists() or is_writable(default_root):
        return default_root
    return fallback_pointer_root()


def configured_data_root(default_root: Path) -> Path:
    """从数据根指针读取自定义数据根；未配置或非绝对路径时返回默认根。"""
    default_root = Path(default_root).expanduser().resolve()
    value = read_settings(pointer_root(default_root)).get(DATA_ROOT_KEY)
    if value in (None, ""):
        return default_root
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return default_root
    return path.resolve()


def resolve_data_root(default_root: Path) -> DataRootResolution:
    """解析启动时使用的数据根；默认根不可写且未配置自定义根时引导选择。"""
    default_root = Path(default_root).expanduser().resolve()
    configured = configured_data_root(default_root)
    if configured != default_root:
        return DataRootResolution(
            root=configured, needs_choice=not is_writable(configured)
        )
    return DataRootResolution(
        root=default_root, needs_choice=not is_writable(default_root)
    )


def apply_data_root(
    default_root: Path,
    previous_root: Path,
    new_root: Path,
) -> Path:
    """切换数据根：设置文件复制到新根并更新指针，旧文件保留不删除。

    换回默认根时只清除指针；环境与模型缓存不做自动迁移。
    """
    default_root = Path(default_root).expanduser().resolve()
    previous_root = Path(previous_root).expanduser().resolve()
    new_root = Path(new_root).expanduser().resolve()
    if new_root == default_root:
        pointer = pointer_root(default_root)
        values = read_settings(pointer)
        values.pop(DATA_ROOT_KEY, None)
        write_settings(pointer, values)
        return default_root
    values = dict(read_settings(previous_root))
    values[DATA_ROOT_KEY] = str(new_root)
    write_settings(new_root, values)
    pointer = pointer_root(default_root)
    pointer_values = read_settings(pointer)
    pointer_values[DATA_ROOT_KEY] = str(new_root)
    write_settings(pointer, pointer_values)
    return new_root

def command_environment(
    paths: RuntimePaths, inherited: Mapping[str, str] | None = None
) -> dict[str, str]:
    environment = dict(os.environ if inherited is None else inherited)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(paths.source_root) + (
        os.pathsep + existing if existing else ""
    )
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def discovery_arguments(options: GuiOptions, *, preview: bool = False) -> list[str]:
    arguments: list[str] = []
    if preview or not options.hash_videos:
        arguments.append("--no-hash-videos")
    return arguments


def pipeline_arguments(options: GuiOptions) -> list[str]:
    arguments = ["--device", options.device]
    if options.ffmpeg is not None:
        arguments.extend(["--ffmpeg", str(options.ffmpeg)])
    return arguments


def scan_command(paths: RuntimePaths, root: Path, options: GuiOptions) -> list[str]:
    return [
        str(paths.venv_python),
        "-m",
        "cc_cover",
        "scan",
        str(root),
        "--json",
        *discovery_arguments(options, preview=True),
    ]


def transcribe_command(
    paths: RuntimePaths,
    root: Path,
    options: GuiOptions,
    exclude_file: Path | None = None,
) -> list[str]:
    return [
        str(paths.venv_python),
        "-m",
        "cc_cover",
        "transcribe",
        str(root),
        "--runs-root",
        str(paths.runs_root),
        "--model-cache",
        str(paths.model_cache),
        *discovery_arguments(options),
        *pipeline_arguments(options),
        *(["--exclude", str(exclude_file)] if exclude_file is not None else []),
    ]


def resume_command(paths: RuntimePaths, run_dir: Path) -> list[str]:
    return [str(paths.venv_python), "-m", "cc_cover", "resume", str(run_dir)]


def python_candidates() -> list[list[str]]:
    candidates: list[list[str]] = []
    launcher = shutil.which("py")
    if launcher:
        candidates.extend(
            [[launcher, version] for version in ("-3.10", "-3.11", "-3.12")]
        )
    for name in ("python", "python3"):
        executable = shutil.which(name)
        if executable and [executable] not in candidates:
            candidates.append([executable])
    return candidates


def setup_commands(
    paths: RuntimePaths, base_python: Sequence[str], accelerator: str
) -> list[list[str]]:
    if accelerator not in {"cuda", "cpu"}:
        raise ValueError(f"不支持的加速器：{accelerator}")
    torch_index = (
        "https://download.pytorch.org/whl/cu121"
        if accelerator == "cuda"
        else "https://download.pytorch.org/whl/cpu"
    )
    commands: list[list[str]] = []
    if not paths.venv_python.is_file():
        commands.append([*base_python, "-m", "venv", str(paths.venv_root)])
    commands.extend(
        [
            [
                str(paths.venv_python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
            # CPU/CUDA 轮子版本号相同，必须先卸掉再装，否则 pip 会跳过替换。
            [
                str(paths.venv_python),
                "-m",
                "pip",
                "uninstall",
                "-y",
                "torch",
                "torchaudio",
            ],
            [
                str(paths.venv_python),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-cache-dir",
                f"torch=={TORCH_VERSION}",
                f"torchaudio=={TORCH_VERSION}",
                "--index-url",
                torch_index,
            ],
            [
                str(paths.venv_python),
                "-m",
                "pip",
                "install",
                *ASR_DEPENDENCIES,
            ],
        ]
    )
    return commands


def environment_check_command(
    paths: RuntimePaths, accelerator: str = "cpu"
) -> list[str]:
    if accelerator not in {"cuda", "cpu"}:
        raise ValueError(f"不支持的加速器：{accelerator}")
    require_cuda = "True" if accelerator == "cuda" else "False"
    return [
        str(paths.venv_python),
        "-c",
        (
            "import ctranslate2, funasr, faster_whisper, imageio_ffmpeg, torch; "
            f"require_cuda = {require_cuda}; "
            "cuda_ok = bool(torch.cuda.is_available()); "
            "ct2_count = int(ctranslate2.get_cuda_device_count()); "
            "print('环境检查通过'); "
            "print('PyTorch:', torch.__version__); "
            "print('CUDA:', cuda_ok); "
            "print('CTranslate2 CUDA devices:', ct2_count); "
            "print('FFmpeg:', imageio_ffmpeg.get_ffmpeg_exe()); "
            "ok = (not require_cuda) or (cuda_ok and ct2_count > 0); "
            "raise SystemExit("
            "0 if ok else "
            "(print('错误：已选择 NVIDIA GPU，但当前环境 CUDA 不可用。"
            "请确认已安装 NVIDIA 驱动，并重新执行安装 / 修复运行环境。') or 1)"
            ")"
        ),
    ]


def detect_device_command(paths: RuntimePaths) -> list[str]:
    """构造检测可用运行设备的命令：CUDA 可用输出 cuda，否则输出 cpu。"""
    return [
        str(paths.venv_python),
        "-c",
        (
            "import torch; "
            "import ctranslate2; "
            "ok = bool(torch.cuda.is_available()) and "
            "int(ctranslate2.get_cuda_device_count()) > 0; "
            "print('cuda' if ok else 'cpu')"
        ),
    ]


def parsed_device(output: str) -> str | None:
    """从检测命令输出解析运行设备；无法识别时返回 None。"""
    for line in reversed((output or "").splitlines()):
        value = line.strip()
        if value in GUI_DEVICE_CHOICES:
            return value
    return None


def nvidia_probe_command() -> list[str]:
    """构造 NVIDIA 硬件探测命令：列出 GPU 名称；无 NVIDIA 驱动时失败。"""
    return ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]


def device_probe_commands(paths: RuntimePaths) -> list[list[str]]:
    """按优先级返回运行设备探测命令：先运行时 CUDA 探测，再 NVIDIA 硬件探测。"""
    return [detect_device_command(paths), nvidia_probe_command()]


def parsed_nvidia_probe(output: str) -> str | None:
    """NVIDIA 硬件探测输出非空（存在 GPU）时视为 cuda。"""
    for line in (output or "").splitlines():
        if line.strip():
            return "cuda"
    return None


def environment_status_label(accelerator: str, _check_output: str = "") -> str:
    if accelerator == "cuda":
        return "运行环境已就绪（GPU）"
    return "运行环境已就绪（CPU）"

