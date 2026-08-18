from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cc_cover.data_root import RuntimePaths
from cc_cover.human_readable import format_size
from cc_cover.settings import GUI_DEVICE_CHOICES

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
class GuiOptions:
    device: str = "auto"
    hash_videos: bool = True
    ffmpeg: Path | None = None


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

