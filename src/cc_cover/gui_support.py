from __future__ import annotations

import hashlib
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


APP_DATA_DIRECTORY = "CC-Cover"
SETTINGS_FILENAME = "settings.json"
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

_LOCK_FILENAME = "instance.lock"
_ERROR_ALREADY_EXISTS = 183
CLEANUP_WARNING_BYTES = 5 * 1024**3


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


RUN_DIR_PATTERN = re.compile(r"^运行目录：\s*(.+?)\s*$", re.MULTILINE)
ERROR_LINE_PATTERN = re.compile(r"^错误：\s*(.+)$", re.MULTILINE)
PROGRESS_PATTERN = re.compile(
    r"^\[[A-Za-z_\-]+\s+(\d+)/(\d+)\]\s*(.*)$", re.MULTILINE
)
PROGRESS_ENGINE_PATTERN = re.compile(
    r"^\[([A-Za-z_\-]+)\s+\d+/\d+\]\s*(.*)$", re.MULTILINE
)
VIDEO_PATH_PATTERN = re.compile(
    r"((?:[A-Za-z]:[\\/]|[\\/])[^\s,，）)\]]+\.(?:mp4|mkv|avi|mov|wmv|flv|webm|m4v|ts|m2ts|mts|ogv|mpg|mpeg|3gp|rmvb|rm|vob|asf|f4v|divx))",
    re.IGNORECASE,
)
ENGINE_VIDEO_PATTERN = re.compile(r"video=([^,\s]+)")


def detect_stage(reason: str, fallback: str) -> str:
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


def extract_file(output: str, reason: str) -> str | None:
    progress_file: str | None = None
    for match in PROGRESS_PATTERN.finditer(output):
        file_path = match.group(3).strip()
        if file_path:
            progress_file = file_path
    if progress_file:
        return progress_file
    match = ENGINE_VIDEO_PATTERN.search(reason)
    if match:
        return match.group(1).strip()
    match = VIDEO_PATH_PATTERN.search(reason)
    if match:
        return match.group(1).strip()
    return None


def failure_info(
    output: str,
    *,
    fallback_stage: str,
    reason: str | None = None,
    run_dir: Path | None = None,
) -> FailureInfo:
    """从子进程输出解析失败对话框需要的信息。"""
    text = output or ""
    if reason is None:
        error_lines = ERROR_LINE_PATTERN.findall(text)
        reason = error_lines[-1].strip() if error_lines else text.strip()
        if len(reason) > 1200:
            reason = reason[-1200:]
    run_dir_match = RUN_DIR_PATTERN.search(text)
    resolved_run_dir = (
        run_dir
        if run_dir is not None
        else (Path(run_dir_match.group(1)) if run_dir_match else None)
    )
    progress_matches = list(PROGRESS_PATTERN.finditer(text))
    done_count = total_count = None
    if progress_matches:
        last = progress_matches[-1]
        done_count = int(last.group(1))
        total_count = int(last.group(2))
    return FailureInfo(
        stage=detect_stage(reason, fallback_stage),
        reason=reason,
        file=extract_file(text, reason),
        run_dir=resolved_run_dir,
        done_count=done_count,
        total_count=total_count,
    )


def run_dir_from_output(output: str) -> Path | None:
    """从 CLI 输出解析运行目录；取最后一个「运行目录：」匹配。"""
    matches = RUN_DIR_PATTERN.findall(output or "")
    return Path(matches[-1]) if matches else None


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


def format_size(size_bytes: int) -> str:
    """字节数转人类可读大小；小于 1KB 按字节，否则保留一位小数。"""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{int(size_bytes)} B"


def format_duration(seconds: float | None) -> str:
    """秒数转中文时长；无法计算时返回「未知」。"""
    if seconds is None:
        return "未知"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} 时 {minutes} 分 {secs} 秒"
    if minutes:
        return f"{minutes} 分 {secs} 秒"
    return f"{secs} 秒"


@dataclass(frozen=True)
class ProgressSnapshot:
    current: int
    total: int
    percent: int
    elapsed_seconds: float
    remaining_seconds: float | None


class ProgressTracker:
    """解析转写进度行，跟踪「第 N / 共 M 个」与耗时/粗估剩余时间。

    进度行形如 ``[funasr 2/5] <视频路径>``，每个视频在双模型（funasr /
    faster_whisper）各打印一行；一个文件两行都出现才计为完成，百分比因此反映
    真实的转写完成度，而不是“已开始”。剩余时间按「平均耗时 × 剩余数」粗估。
    """

    _ENGINES = ("funasr", "faster_whisper")

    def __init__(self, total: int, started_at: float = 0.0):
        self.total = max(0, int(total))
        self.started_at = float(started_at)
        self._engines_by_path: dict[str, set[str]] = {}

    def on_progress_line(self, line: str) -> None:
        match = PROGRESS_ENGINE_PATTERN.search(line)
        if match is None:
            return
        engine = match.group(1)
        path = match.group(2).strip()
        if path:
            self._engines_by_path.setdefault(path, set()).add(engine)

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
    lines = [
        f"将处理 {candidate_count} 个视频并替换同名 TXT（替换前自动备份）。"
    ]
    if excluded_count:
        lines.append(f"已排除 {excluded_count} 个视频，本次不处理。")
    else:
        lines.append("本次无已排除的视频。")
    return "\n".join(lines)


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
    for directory in (
        paths.data_root,
        paths.venv_root,
        paths.model_cache,
        paths.runs_root,
        paths.temp_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)


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
    paths: RuntimePaths, root: Path, options: GuiOptions
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


class SingleInstanceLock:
    """基于数据根的单实例锁。

    Windows 使用命名互斥体（进程退出后由系统自动释放），其他平台在数据根
    写入包含 PID 的锁文件并在发现陈旧锁时自动清理。
    """

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.acquired = False
        self._handle: Any = None
        self._lock_path: Path | None = None

    def _identity(self) -> str:
        """返回数据根的规范化标识，用于生成跨进程一致的锁名。"""
        normalized = (
            os.path.normcase(str(self.data_root)).replace("\\", "/").lower()
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    def _mutex_name(self) -> str:
        return f"CC-Cover-{self._identity()}"

    def acquire(self) -> bool:
        """尝试获取锁；已有实例持有同一数据根锁时返回 False。"""
        if self.acquired:
            return True
        if os.name == "nt":
            return self._acquire_mutex()
        return self._acquire_file()

    def release(self) -> None:
        if os.name == "nt":
            self._release_mutex()
        else:
            self._release_file()

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def _acquire_mutex(self) -> bool:
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = (
                wintypes.LPVOID,
                wintypes.BOOL,
                wintypes.LPCWSTR,
            )
            kernel32.CreateMutexW.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
        except (ImportError, OSError, AttributeError) as exc:
            raise OSError(f"无法创建单实例锁：{exc}") from exc
        handle = kernel32.CreateMutexW(None, False, self._mutex_name())
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error() or 1)
        if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        self.acquired = True
        return True

    def _release_mutex(self) -> None:
        if self._handle is not None:
            try:
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
                kernel32.CloseHandle.restype = wintypes.BOOL
                kernel32.CloseHandle(self._handle)
            except (ImportError, OSError, AttributeError):
                pass
            self._handle = None
        self.acquired = False

    def _acquire_file(self) -> bool:
        if self.acquired:
            return True
        lock_path = self.data_root / _LOCK_FILENAME
        for attempt in range(2):
            try:
                descriptor = os.open(
                    lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError:
                if (
                    attempt == 0
                    and not self._pid_alive(self._read_pid(lock_path))
                ):
                    try:
                        lock_path.unlink()
                    except OSError:
                        return False
                    continue
                return False
            except OSError:
                return False
            try:
                os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            finally:
                os.close(descriptor)
            self._lock_path = lock_path
            self.acquired = True
            return True
        return False

    def _release_file(self) -> None:
        if self._lock_path is not None:
            try:
                self._lock_path.unlink()
            except OSError:
                pass
            self._lock_path = None
        self.acquired = False

    @staticmethod
    def _read_pid(lock_path: Path) -> int | None:
        try:
            text = lock_path.read_text(encoding="ascii").strip()
        except OSError:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if pid is None:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True


def focus_existing_window(prefix: str = "CC-Cover") -> bool:
    """尽力将已运行的 CC-Cover 主窗口带到前台（仅 Windows，失败时返回 False）。"""
    if os.name != "nt" or not prefix:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        enum_proc = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )
        user32.EnumWindows.argtypes = (enum_proc, wintypes.LPARAM)
        user32.EnumWindows.restype = wintypes.BOOL
        user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = (
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        )
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        user32.SetForegroundWindow.restype = wintypes.BOOL
        found: list[int] = []

        def _enum(hwnd: int, _lparam: int) -> bool:
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                if buffer.value.startswith(prefix):
                    found.append(int(hwnd))
                    return False
            return True

        if not user32.EnumWindows(enum_proc(_enum), 0):
            return False
        if not found:
            return False
        handle = found[0]
        user32.ShowWindow(handle, 9)  # SW_RESTORE
        user32.SetForegroundWindow(handle)
        return True
    except (AttributeError, OSError, TypeError, ValueError):
        return False
