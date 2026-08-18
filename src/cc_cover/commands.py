from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from cc_cover.data_root import RuntimePaths
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
# 受版本一致性检测管理的全部依赖：torch/torchaudio 配对（由 TORCH_VERSION 控制）
# + ASR_DEPENDENCIES 各项各自的包名，从 ASR_DEPENDENCIES 派生，避免手写第二份清单。
TRACKED_PACKAGES: tuple[str, ...] = ("torch", "torchaudio") + tuple(
    Requirement(spec).name for spec in ASR_DEPENDENCIES
)
_TORCH_PAIR = frozenset({"torch", "torchaudio"})


@dataclass(frozen=True)
class GuiOptions:
    device: str = "auto"
    hash_videos: bool = True
    ffmpeg: Path | None = None


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


def reinstall_scope(outdated: set[str] | None) -> tuple[bool, bool]:
    """outdated 落在 setup_commands() 里会实际触发哪些安装步骤。

    返回 (是否需要重装 torch/torchaudio, 是否需要重装至少一个 ASR_DEPENDENCIES
    包)。outdated 为 None（全量安装）时两者都是 True。调用方（比如 GUI 侧渲染
    日志文案、估算下载量）应该用这个函数而不是自己重新判断 outdated 里有没有
    "torch"/"torchaudio" 这类领域知识。
    """
    needs_torch = outdated is None or bool(_TORCH_PAIR & outdated)
    needs_asr = outdated is None or bool(outdated - _TORCH_PAIR)
    return needs_torch, needs_asr


def setup_commands(
    paths: RuntimePaths,
    base_python: Sequence[str],
    accelerator: str,
    outdated: set[str] | None = None,
) -> list[list[str]]:
    """构造安装/修复命令。

    outdated 为 None 时是全量安装（覆盖首次安装场景，行为与不做版本检测时
    完全一致）；传入非 None 的包名集合时，只重装集合里的包——torch 配对的
    卸载+强制重装步骤只在 torch/torchaudio 至少一个落后时才会加入。
    """
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
    commands.append(
        [
            str(paths.venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    needs_torch, _ = reinstall_scope(outdated)
    if needs_torch:
        commands.extend(
            [
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
            ]
        )
    asr_targets = (
        ASR_DEPENDENCIES
        if outdated is None
        else tuple(
            spec for spec in ASR_DEPENDENCIES if Requirement(spec).name in outdated
        )
    )
    if asr_targets:
        commands.append(
            [str(paths.venv_python), "-m", "pip", "install", *asr_targets]
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
            "from importlib.metadata import PackageNotFoundError, version as _pkg_version; "
            f"_tracked = {TRACKED_PACKAGES!r}\n"
            "_versions = {}\n"
            "for _name in _tracked:\n"
            "    try:\n"
            "        _versions[_name] = _pkg_version(_name)\n"
            "    except PackageNotFoundError:\n"
            "        pass\n"
            "print('VERSIONS: ' + ' '.join(f'{k}={v}' for k, v in _versions.items())); "
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


_VERSIONS_LINE_PATTERN = re.compile(r"^VERSIONS:\s*(?P<pairs>.+)$", re.MULTILINE)
_VERSION_PAIR_PATTERN = re.compile(r"(?P<name>[A-Za-z0-9_.-]+)=(?P<version>\S+)")


def parse_installed_versions(output: str) -> dict[str, str]:
    """从 environment_check_command() 的输出里解析已装包版本。

    找不到 VERSIONS: 那一行时返回空字典（比如脚本在打印这行之前就失败了）。
    """
    match = _VERSIONS_LINE_PATTERN.search(output)
    if not match:
        return {}
    return {
        pair.group("name"): pair.group("version")
        for pair in _VERSION_PAIR_PATTERN.finditer(match.group("pairs"))
    }


def _unsatisfied(name: str, specifier: SpecifierSet, installed: Mapping[str, str]) -> bool:
    """已装版本缺失、或不满足给定约束，即视为不满足。"""
    version_text = installed.get(name)
    return version_text is None or Version(version_text) not in specifier


def outdated_packages(installed: Mapping[str, str]) -> set[str]:
    """比对已装版本与当前代码声明的约束，返回不满足约束的包名集合。

    torch/torchaudio 配对处理：任一不满足（含缺失）就把两个都计入结果集，
    因为它们总是同一条 pip 命令、同一个版本号装出来的。
    """
    outdated: set[str] = set()
    torch_specifier = SpecifierSet(f"=={TORCH_VERSION}")
    if any(_unsatisfied(name, torch_specifier, installed) for name in _TORCH_PAIR):
        outdated |= _TORCH_PAIR
    for spec in ASR_DEPENDENCIES:
        requirement = Requirement(spec)
        if _unsatisfied(requirement.name, requirement.specifier, installed):
            outdated.add(requirement.name)
    return outdated


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


def environment_status_label(
    accelerator: str, _check_output: str = "", *, outdated: bool = False
) -> str:
    base = "运行环境已就绪（GPU）" if accelerator == "cuda" else "运行环境已就绪（CPU）"
    return f"{base}（有更新可用）" if outdated else base
