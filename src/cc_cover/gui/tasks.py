from __future__ import annotations

import os
import queue
import subprocess
from typing import Any

from cc_cover.gui.background import TaskCancelled
from cc_cover.gui.commands import command_environment
from cc_cover.gui.data_root import RuntimePaths
from cc_cover.gui.progress import done_event_present

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


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


class TaskRunner:
    """子进程执行的共用外壳：环境变量、捕获式/流式运行、取消状态。

    ``events`` 由 CCCoverApp 传入并共享——CCCoverApp 的 ``_poll_events()``
    消费，这里只负责在流式运行时 put() 日志行；worker 生命周期（成功/取消/
    报错要弹哪种对话框）仍由 CCCoverApp 的 5 个任务入口各自决定，不在这里
    收口，因为每个入口的 fallback_stage、要不要走 best-effort summary 都
    不一样，勉强收成一个通用形状反而会把这些真实差异藏起来。
    """

    def __init__(self, paths: RuntimePaths, events: "queue.Queue[Any]") -> None:
        self.paths = paths
        self.events = events
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.stop_triggered = False

    def reset(self) -> None:
        """每次任务发起前调用，清掉上一次任务遗留的取消/停止标记。"""
        self.cancel_requested = False
        self.stop_triggered = False

    def cancel(self) -> None:
        self.cancel_requested = True
        if self.process is not None and self.process.poll() is None:
            self.stop_triggered = True
            terminate_process_tree(self.process)

    def run_capture(self, command: list[str], *, hf_token: str = "") -> str:
        if self.cancel_requested:
            raise TaskCancelled("任务已由用户停止。")
        process = subprocess.Popen(
            command,
            cwd=str(self.paths.data_root),
            env=command_environment(self.paths, hf_token=hf_token),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
        self.process = process
        assert process.stdout is not None
        return self._finish_process(process, "".join(process.stdout))

    def run_streaming(self, command: list[str], *, hf_token: str = "") -> str:
        if self.cancel_requested:
            raise TaskCancelled("任务已由用户停止。")
        self.events.put(("log", "\n▶ " + " ".join(command[1:]) + "\n"))
        process = subprocess.Popen(
            command,
            cwd=str(self.paths.data_root),
            env=command_environment(self.paths, hf_token=hf_token),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
        self.process = process
        assert process.stdout is not None
        lines: list[str] = []
        for line in process.stdout:
            lines.append(line)
            self.events.put(("log", line))
        return self._finish_process(process, "".join(lines))

    def _finish_process(self, process: subprocess.Popen[str], output: str) -> str:
        return_code = process.wait()
        self.process = None
        if self.stop_triggered:
            raise TaskCancelled(
                output.strip() or "任务已由用户停止。运行产物可以稍后继续。"
            )
        if self.cancel_requested and return_code != 0:
            raise TaskCancelled(
                output.strip() or "任务已由用户停止。运行产物可以稍后继续。"
            )
        if return_code != 0 and not done_event_present(output):
            raise RuntimeError(
                output.strip() or f"任务执行失败，退出代码：{return_code}"
            )
        return output
