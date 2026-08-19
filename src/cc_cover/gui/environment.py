from __future__ import annotations

import re
import shutil
from typing import Mapping, Sequence

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from cc_cover.gui.data_root import RuntimePaths

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


def needs_force_reinstall_prompt(outdated: set[str]) -> bool:
    """环境检查成功、但版本比对显示全部匹配（没有可精简重装的目标）时为 True。

    这种情况下点击"安装 / 修复运行环境"本来会什么都不装，需要给用户一个
    强制完整重装的逃生舱，用来处理版本号没变但文件本身损坏（磁盘错误、
    杀软误隔离等）这类版本比对查不出来的问题。
    """
    return not outdated


def setup_commands(
    paths: RuntimePaths,
    base_python: Sequence[str],
    device: str,
    outdated: set[str] | None = None,
) -> list[list[str]]:
    """构造安装/修复命令。

    outdated 为 None 时是全量安装（覆盖首次安装场景，行为与不做版本检测时
    完全一致）；传入非 None 的包名集合时，只重装集合里的包——torch 配对的
    卸载+强制重装步骤只在 torch/torchaudio 至少一个落后时才会加入。
    """
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"不支持的设备：{device}")
    torch_index = (
        "https://download.pytorch.org/whl/cu121"
        if device == "cuda"
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
        commands.append([str(paths.venv_python), "-m", "pip", "install", *asr_targets])
    return commands


def environment_check_command(paths: RuntimePaths, device: str = "cpu") -> list[str]:
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"不支持的设备：{device}")
    require_cuda = "True" if device == "cuda" else "False"
    script = f"""\
import ctranslate2, funasr, faster_whisper, imageio_ffmpeg, torch
from importlib.metadata import PackageNotFoundError, version as _pkg_version

_tracked = {TRACKED_PACKAGES!r}
_versions = {{}}
for _name in _tracked:
    try:
        _versions[_name] = _pkg_version(_name)
    except PackageNotFoundError:
        pass
print('VERSIONS: ' + ' '.join(f'{{k}}={{v}}' for k, v in _versions.items()))

require_cuda = {require_cuda}
cuda_ok = bool(torch.cuda.is_available())
ct2_count = int(ctranslate2.get_cuda_device_count())
print('环境检查通过')
print('PyTorch:', torch.__version__)
print('CUDA:', cuda_ok)
print('CTranslate2 CUDA devices:', ct2_count)
print('FFmpeg:', imageio_ffmpeg.get_ffmpeg_exe())
ok = (not require_cuda) or (cuda_ok and ct2_count > 0)
if not ok:
    print('错误：已选择 NVIDIA GPU，但当前环境 CUDA 不可用。请确认已安装 NVIDIA 驱动，并重新执行安装 / 修复运行环境。')
raise SystemExit(0 if ok else 1)
"""
    return [str(paths.venv_python), "-c", script]


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


def _unsatisfied(
    name: str, specifier: SpecifierSet, installed: Mapping[str, str]
) -> bool:
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


def environment_status_label(
    device: str, _check_output: str = "", *, outdated: bool = False
) -> str:
    base = "运行环境已就绪（GPU）" if device == "cuda" else "运行环境已就绪（CPU）"
    return f"{base}（有更新可用）" if outdated else base


NOT_INSTALLED_STATUS_LABEL = "尚未安装（已有装好的环境？点右侧“更改…”指定位置）"
"""数据根换过目录后最常见的困惑——旧环境其实还在，只是没指过去。"""
