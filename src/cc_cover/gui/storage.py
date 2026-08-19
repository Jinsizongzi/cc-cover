from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cc_cover.gui.data_root import RuntimePaths
from cc_cover.gui.human_readable import format_size

CLEANUP_WARNING_BYTES = 5 * 1024**3

# 磁盘预检与安装进度使用的字节估算（仅用于提示与粗估剩余，非精确计量）。
# torch + torchaudio 轮子：CPU 约 330MB，CUDA cu121 约 2.7GB，均留余量。
TORCH_CPU_BYTES = 400 * 1024**2
TORCH_CUDA_BYTES = 3200 * 1024**2
# 默认模型：FunASR（paraformer-large + fsmn-vad + ct-punc）约 2GB，large-v3 约 3GB
# （相比之前的默认 large-v3-turbo 约 1.6GB 更大——turbo 是裁剪过解码层的蒸馏版）。
FUNASR_MODELS_BYTES = 2500 * 1024**2
FAST_WHISPER_MODELS_BYTES = 3100 * 1024**2
# 其余 ASR 依赖轮子（funasr、faster-whisper、ctranslate2 等）约 250MB。
ASR_DEPENDENCIES_BYTES = 400 * 1024**2
# venv/临时目录开销与磁盘碎片余量。
INSTALL_BUFFER_BYTES = 1024**3


def install_download_bytes(
    device: str, *, include_torch: bool = True, include_asr: bool = True
) -> int:
    """安装阶段 pip 实际下载量的估算（torch 轮子 + ASR 依赖，不含模型）。

    用于安装进度条的总量与剩余时间估算；与``estimate_install_required_bytes``
    不同——后者额外计入首次运行需下载的模型与余量，用于磁盘预检。

    include_torch/include_asr 用于精简重装场景（只重装部分包时，对应那部分
    不计入总量估算，避免进度条分母虚高）；两者默认 True，保持全量安装时的
    估算不变。
    """
    total = 0
    if include_torch:
        total += TORCH_CUDA_BYTES if device == "cuda" else TORCH_CPU_BYTES
    if include_asr:
        total += ASR_DEPENDENCIES_BYTES
    return total


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
