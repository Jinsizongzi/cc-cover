from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


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
    include_whitespace_only: bool = False
    include_missing: bool = False
    hash_videos: bool = True
    ffmpeg: Path | None = None


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


def option_arguments(options: GuiOptions, *, preview: bool = False) -> list[str]:
    arguments = ["--device", options.device]
    if options.include_whitespace_only:
        arguments.append("--include-whitespace-only")
    if options.include_missing:
        arguments.append("--include-missing")
    if preview or not options.hash_videos:
        arguments.append("--no-hash-videos")
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
        *option_arguments(options, preview=True),
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
        *option_arguments(options),
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
            [
                str(paths.venv_python),
                "-m",
                "pip",
                "install",
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


def environment_check_command(paths: RuntimePaths) -> list[str]:
    return [
        str(paths.venv_python),
        "-c",
        (
            "import ctranslate2, funasr, faster_whisper, imageio_ffmpeg, torch; "
            "print('环境检查通过'); "
            "print('PyTorch:', torch.__version__); "
            "print('CUDA:', torch.cuda.is_available()); "
            "print('FFmpeg:', imageio_ffmpeg.get_ffmpeg_exe())"
        ),
    ]
