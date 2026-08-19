from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cc_cover.core.models import Event, Phase
from cc_cover.core.pipeline.errors import PipelineError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, payload: Any) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def emit_event(event: Event) -> None:
    """在人读文字打印旁边无条件追加一行结构化事件；不替代、不影响原有打印。"""
    print(json.dumps(event.to_dict(), ensure_ascii=False), flush=True)


def load_json(path: Path, *, phase: Phase) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"缺少运行产物：{path}", phase=phase) from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"运行产物 JSON 无效：{path}: {exc}", phase=phase) from exc
    if not isinstance(value, dict):
        raise PipelineError(f"运行产物顶层必须是对象：{path}", phase=phase)
    return value


def load_optional_json(path: Path) -> dict[str, Any] | None:
    """读取可选的运行产物 JSON；文件缺失或无效时返回 None。"""
    try:
        return load_json(path, phase=Phase.SETUP)
    except (PipelineError, OSError):
        return None
