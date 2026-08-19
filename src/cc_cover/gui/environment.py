from __future__ import annotations

import queue
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from cc_cover.gui.background import (
    CancelledOutcome,
    DoneOutcome,
    ErrorOutcome,
    IdleOutcome,
    TaskCancelled,
    run_in_background,
)
from cc_cover.gui.data_root import RuntimePaths
from cc_cover.gui.device import device_probe_commands, parsed_device, parsed_nvidia_probe
from cc_cover.gui.dialogs import DialogHost
from cc_cover.gui.progress import failure_info_from_command
from cc_cover.gui.storage import (
    disk_precheck,
    estimate_install_required_bytes,
    install_download_bytes,
    list_runs,
    runs_total_size,
)
from cc_cover.gui.tasks import TaskRunner

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

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


class EnvironmentController:
    """环境/设备检查与安装的共用外壳。

    只造还没执行的 worker 闭包，不负责起线程或切换 busy 状态——那是
    CCCoverApp._start_worker/_set_busy 的地盘，调用方拿到闭包后自己派发到
    线程；测试时可以直接同步调用闭包，不用起真线程。is_device_auto/
    current_device/hf_token 三个读取器指向 CCCoverApp 持有的 Tk 变量/状态，
    这里只读不改。
    """

    def __init__(
        self,
        paths: RuntimePaths,
        events: "queue.Queue[Any]",
        tasks: TaskRunner,
        dialogs: DialogHost,
        *,
        is_device_auto: Callable[[], bool],
        current_device: Callable[[], str],
        hf_token: Callable[[], str],
    ) -> None:
        self.paths = paths
        self.events = events
        self.tasks = tasks
        self.dialogs = dialogs
        self.is_device_auto = is_device_auto
        self.current_device = current_device
        self.hf_token = hf_token
        self._prompt_recheck = False

    def request_recheck_prompt(self) -> None:
        self._prompt_recheck = True

    def clear_recheck_prompt(self) -> None:
        self._prompt_recheck = False

    def _outdated_packages_for(self, device: str) -> tuple[str, set[str]]:
        """跑一次环境检查命令，返回 (原始输出, 已过期的包名集合)。

        check()/setup() 都需要这两样——check() 用输出做状态上报，setup() 用
        过期集合决定精简重装范围，字面相同的调用只写一份。
        """
        output = self.tasks.run_capture(
            environment_check_command(self.paths, device), hf_token=self.hf_token()
        )
        return output, outdated_packages(parse_installed_versions(output))

    def _detect_and_report(self) -> None:
        if not self.is_device_auto():
            return
        detected: str | None = None
        for command in device_probe_commands(self.paths):
            try:
                output = self.tasks.run_capture(command, hf_token=self.hf_token())
            except Exception:
                continue
            detected = parsed_device(output) or parsed_nvidia_probe(output)
            if detected is not None:
                break
        self.events.put(("device_detected", detected or "cpu"))

    def _find_base_python(self) -> list[str]:
        for candidate in python_candidates():
            completed = subprocess.run(
                [
                    *candidate,
                    "-c",
                    (
                        "import sys; "
                        "raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)"
                    ),
                ],
                capture_output=True,
                text=True,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
            if completed.returncode == 0:
                return candidate
        raise RuntimeError(
            "未找到 Python 3.10、3.11 或 3.12。请先安装 Python，并勾选 Add Python to PATH。"
        )

    def _confirm_force_reinstall(self) -> bool:
        """版本比对全部匹配、没有可精简重装的目标时，询问是否强制完整重装。

        用于版本号没变但文件本身损坏（磁盘错误、杀软误隔离等）这类版本比对
        查不出来的场景——给用户一个手动逃生舱，而不是让"安装 / 修复运行环境"
        在这种情况下悄悄变成什么都不做。任务已停止时视为取消，跟
        CCCoverApp._confirm_start 保持同一种语义（两处传参形状不同，没有抽
        共用工具，见 ADR-0003）。
        """
        result_queue: queue.Queue[bool] = queue.Queue()
        self.events.put(("confirm_force_reinstall", result_queue))
        confirmed = result_queue.get()
        if self.tasks.cancel_requested:
            return False
        return confirmed

    def build_check_worker(self) -> Callable[[], None]:
        def worker() -> None:
            device = "auto"

            def run() -> None:
                nonlocal device
                if not self.paths.venv_python.is_file():
                    self.events.put(
                        ("environment", (False, NOT_INSTALLED_STATUS_LABEL))
                    )
                    self.events.put(IdleOutcome("请先安装运行环境"))
                    self._detect_and_report()
                    return
                device = self.current_device()
                output, outdated = self._outdated_packages_for(device)
                self.clear_recheck_prompt()
                self.events.put(
                    (
                        "environment",
                        (
                            True,
                            environment_status_label(
                                device, output, outdated=bool(outdated)
                            ),
                        ),
                    )
                )
                self.events.put(("log", output + "\n"))
                self._detect_and_report()
                self.events.put(IdleOutcome("就绪"))

            def on_cancel(exc: TaskCancelled) -> None:
                self.events.put(
                    CancelledOutcome(
                        info=failure_info_from_command(
                            [], exc, fallback_stage="环境检查"
                        )
                    )
                )

            def on_error(exc: Exception) -> None:
                # 环境未就绪是预期内状态，不当作任务失败：不生成 ErrorOutcome、
                # 不弹错误对话框，只更新状态后照常落回 idle，跟成功路径一致。
                # device 复用 run() 读到的值（而不是再读一次 current_device()），
                # 避免 device_detected 事件恰好在两次读取之间被主线程处理、
                # 导致这里选错提示文案（不受 _set_busy 的按钮禁用保护）。
                detail = str(exc).strip() or "需要安装或修复"
                label = "GPU 环境未就绪" if device == "cuda" else "需要安装或修复"
                self.events.put(("environment", (False, label)))
                self.events.put(("log", f"环境检查失败：{detail}\n"))
                self._detect_and_report()
                if self._prompt_recheck:
                    self.clear_recheck_prompt()
                    self.events.put(("device_check_failed", label))
                self.events.put(IdleOutcome("就绪"))

            run_in_background(run, on_cancel=on_cancel, on_error=on_error)

        return worker

    def precheck_setup(self) -> bool:
        """磁盘预检，不够时通过 dialogs 弹窗确认；在主线程同步执行。

        必须在 build_setup_worker() 返回的闭包开始跑之前调用——这一步要弹
        原生对话框，只有主线程、还没起后台线程时才安全。返回 False 时不应
        该再调用 build_setup_worker()。
        """
        try:
            required = estimate_install_required_bytes(self.current_device())
            check = disk_precheck(self.paths.data_root, required)
            runs_bytes = runs_total_size(list_runs(self.paths.runs_root))
        except OSError as exc:
            self.dialogs.show_disk_precheck_error(str(exc))
            return False
        if check.sufficient:
            return True
        return self.dialogs.confirm_low_disk_space(check, runs_bytes)

    def build_setup_worker(self) -> Callable[[], None]:
        def worker() -> None:
            chunks: list[str] = []

            def run() -> None:
                self.paths.data_root.mkdir(parents=True, exist_ok=True)
                base_python = self._find_base_python()
                device = self.current_device()
                outdated: set[str] | None = None
                if self.paths.venv_python.is_file():
                    try:
                        _, outdated = self._outdated_packages_for(device)
                    except TaskCancelled:
                        raise
                    except Exception:
                        # 环境本身有问题（比如 CUDA 不可用），走全量重装最保险，
                        # 不尝试用一次失败的检查结果算精简重装范围。TaskCancelled
                        # 不在此列——用户主动停止要照常走 on_cancel，不能被这里
                        # 当成"环境检查失败"吞掉、退化成继续跑完整安装。
                        outdated = None
                    else:
                        if (
                            needs_force_reinstall_prompt(outdated)
                            and self._confirm_force_reinstall()
                        ):
                            outdated = None
                commands = setup_commands(
                    self.paths, base_python, device, outdated=outdated
                )
                include_torch, include_asr = reinstall_scope(outdated)
                self.events.put(("log", "开始安装运行环境。此过程可能需要较长时间。\n"))
                if device == "cuda" and include_torch:
                    self.events.put(
                        (
                            "log",
                            "将强制重装 GPU 版 PyTorch，以便覆盖已安装的 CPU 包。\n",
                        )
                    )
                self.events.put(
                    (
                        "install_start",
                        (
                            install_download_bytes(
                                device,
                                include_torch=include_torch,
                                include_asr=include_asr,
                            ),
                            len(commands),
                        ),
                    )
                )
                for index, command in enumerate(commands, start=1):
                    self.events.put(
                        ("status", f"正在安装组件 {index}/{len(commands)}…")
                    )
                    self.events.put(("install_component", (index, len(commands))))
                    chunks.append(
                        self.tasks.run_streaming(command, hf_token=self.hf_token())
                    )
                output = self.tasks.run_capture(
                    environment_check_command(self.paths, device),
                    hf_token=self.hf_token(),
                )
                chunks.append(output)
                self.events.put(("log", output + "\n"))
                self.events.put(
                    (
                        "environment",
                        (True, environment_status_label(device, output)),
                    )
                )
                self._detect_and_report()
                self.events.put(
                    DoneOutcome(
                        title="安装完成",
                        message="运行环境安装并检查通过，可以开始扫描视频。",
                        run_dir=None,
                    )
                )

            def on_cancel(exc: TaskCancelled) -> None:
                self.events.put(
                    CancelledOutcome(
                        info=failure_info_from_command(
                            chunks, exc, fallback_stage="安装运行环境"
                        )
                    )
                )

            def on_error(exc: Exception) -> None:
                self.events.put(
                    ErrorOutcome(
                        title="环境安装失败",
                        info=failure_info_from_command(
                            chunks, exc, fallback_stage="安装运行环境"
                        ),
                    )
                )

            run_in_background(run, on_cancel=on_cancel, on_error=on_error)

        return worker
