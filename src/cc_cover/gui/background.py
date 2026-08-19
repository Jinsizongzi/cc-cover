from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cc_cover.gui.progress import FailureInfo


class TaskCancelled(RuntimeError):
    """用户主动停止任务时抛出，不当作任务失败处理。"""


@dataclass(frozen=True)
class IdleOutcome:
    """worker 安静结束、不弹对话框，仅更新状态文字。"""

    status: str


@dataclass(frozen=True)
class DoneOutcome:
    """worker 成功完成，弹出完成对话框。"""

    title: str
    message: str
    run_dir: Path | None


@dataclass(frozen=True)
class CancelledOutcome:
    """worker 被用户停止。"""

    info: FailureInfo


@dataclass(frozen=True)
class ErrorOutcome:
    """worker 抛出未预期的异常。"""

    title: str
    info: FailureInfo


WorkerOutcome = IdleOutcome | DoneOutcome | CancelledOutcome | ErrorOutcome
"""GUI 进程内部 worker 线程 → UI 主线程的终态消息，不跨进程、不序列化，因此不像
cc_cover.core.models.Event 那样带 kind 字段/Enum/to_dict/from_dict——那一套是 Event 为了
在跨进程 JSON 边界上还原具体子类型才需要的，同进程内类型信息不会丢，isinstance/
match 足够分发。"""


def run_in_background(
    run: Callable[[], None],
    *,
    on_cancel: Callable[[TaskCancelled], None],
    on_error: Callable[[Exception], None],
) -> None:
    """跑 worker 主体，按结果分派给调用方提供的回调。

    只负责跑 run() 和分派取消/错误这两条异常路径；成功路径（run() 正常
    返回）不触发任何回调，由 run() 自己在结束前把该发的消息发出去。
    """
    try:
        run()
    except TaskCancelled as exc:
        on_cancel(exc)
    except Exception as exc:
        on_error(exc)
