from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


APP_DATA_DIRECTORY = "CC-Cover"
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


@dataclass(frozen=True)
class FailureInfo:
    """任务失败/停止时用于对话框的结构化信息。"""

    stage: str
    reason: str
    file: str | None = None
    run_dir: Path | None = None
    done_count: int | None = None
    total_count: int | None = None


@dataclass(frozen=True)
class RuntimePaths:
    source_root: Path
    data_root: Path
    venv_root: Path
    venv_python: Path
    model_cache: Path
    runs_root: Path


@dataclass(frozen=True)
class GuiOptions:
    device: str = "auto"
    hash_videos: bool = True
    ffmpeg: Path | None = None


RUN_DIR_PATTERN = re.compile(r"^运行目录：\s*(.+?)\s*$", re.MULTILINE)
ERROR_LINE_PATTERN = re.compile(r"^错误：\s*(.+)$", re.MULTILINE)
PROGRESS_PATTERN = re.compile(
    r"^\[[A-Za-z_\-]+\s+(\d+)/(\d+)\]\s*(.*)$", re.MULTILINE
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


def runtime_paths(
    *,
    frozen: bool | None = None,
    bundle_root: Path | None = None,
    local_app_data: Path | None = None,
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
    if local_app_data is None:
        environment = os.environ.get("LOCALAPPDATA")
        local_app_data = (
            Path(environment)
            if environment
            else Path.home() / "AppData" / "Local"
        )
    data_root = (local_app_data / APP_DATA_DIRECTORY).resolve()
    venv_root = data_root / ".venv"
    return RuntimePaths(
        source_root=source_root,
        data_root=data_root,
        venv_root=venv_root,
        venv_python=venv_root / "Scripts" / "python.exe",
        model_cache=data_root / "model-cache",
        runs_root=data_root / "runs",
    )


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


def environment_status_label(accelerator: str, _check_output: str = "") -> str:
    if accelerator == "cuda":
        return "运行环境已就绪（GPU）"
    return "运行环境已就绪（CPU）"
