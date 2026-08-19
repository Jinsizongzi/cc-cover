from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from cc_cover.gui.data_root import RuntimePaths
from cc_cover.gui.settings import GuiSettings


def command_environment(
    paths: RuntimePaths,
    inherited: Mapping[str, str] | None = None,
    *,
    hf_token: str = "",
) -> dict[str, str]:
    """构造子进程环境变量。

    hf_token 非空时设置 HF_TOKEN，用于 Hugging Face Hub 下载模型时避免未
    认证请求限流；留空时不碰这个键，不覆盖用户可能已经在系统环境变量里
    设置的 HF_TOKEN（这是这个设置项存在之前就能用的临时解法，两者不冲突）。
    """
    environment = dict(os.environ if inherited is None else inherited)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(paths.source_root) + (
        os.pathsep + existing if existing else ""
    )
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    if hf_token:
        environment["HF_TOKEN"] = hf_token
    return environment


def discovery_arguments(settings: GuiSettings, *, preview: bool = False) -> list[str]:
    arguments: list[str] = []
    if preview or not settings.hash_videos:
        arguments.append("--no-hash-videos")
    return arguments


def pipeline_arguments(settings: GuiSettings) -> list[str]:
    arguments = ["--device", settings.device]
    ffmpeg_text = settings.ffmpeg.strip().strip('"')
    if ffmpeg_text:
        arguments.extend(["--ffmpeg", str(Path(ffmpeg_text).resolve())])
    return arguments


def scan_command(paths: RuntimePaths, root: Path, settings: GuiSettings) -> list[str]:
    return [
        str(paths.venv_python),
        "-m",
        "cc_cover",
        "scan",
        str(root),
        "--json",
        *discovery_arguments(settings, preview=True),
    ]


def transcribe_command(
    paths: RuntimePaths,
    root: Path,
    settings: GuiSettings,
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
        *discovery_arguments(settings),
        *pipeline_arguments(settings),
        *(["--exclude", str(exclude_file)] if exclude_file is not None else []),
    ]


def resume_command(paths: RuntimePaths, run_dir: Path) -> list[str]:
    return [str(paths.venv_python), "-m", "cc_cover", "resume", str(run_dir)]
