from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class Phase(str, Enum):
    """流水线失败发生在哪个处理环节；不含 scan——扫描阶段的错误是 DiscoveryError。"""

    SETUP = "setup"
    AUDIO_EXTRACT = "audio_extract"
    FUNASR = "funasr"
    FASTER_WHISPER = "faster_whisper"
    QUALITY_GATE = "quality_gate"
    WRITEBACK = "writeback"
    VERIFY = "verify"


class EventKind(str, Enum):
    """CLI 子进程向 GUI 报告运行状态用的结构化事件种类。

    只在 pipeline.py/cli.py 打印人读文字的同一批打印点旁边无条件追加；
    人读文字本身不受影响。取值随生产者票（#67 pipeline.py、#68 cli.py）
    逐步补齐，消费端（#69）按需处理未知取值。
    """

    ENGINE_START = "engine_start"
    PROGRESS = "progress"
    RUN_DIR = "run_dir"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class EngineStartEvent:
    """对应 pipeline.py 里"加载 {engine}：device=..."这一行。"""

    engine: str
    device: str
    kind: EventKind = field(default=EventKind.ENGINE_START, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.kind.value, "engine": self.engine, "device": self.device}


@dataclass(frozen=True)
class ProgressEvent:
    """对应 pipeline.py 里"[{engine} {index}/{total}] {video_path}"这一行。"""

    engine: str
    index: int
    total: int
    video_path: str
    kind: EventKind = field(default=EventKind.PROGRESS, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.kind.value,
            "engine": self.engine,
            "index": self.index,
            "total": self.total,
            "video_path": self.video_path,
        }


@dataclass(frozen=True)
class RunDirEvent:
    """对应 cli.py 里"运行目录：..."这一行。"""

    path: str
    kind: EventKind = field(default=EventKind.RUN_DIR, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.kind.value, "path": self.path}


@dataclass(frozen=True)
class DoneEvent:
    """对应 cli.py 里"字幕已写回并复核通过：..."/"复核通过，共 N 个..."这几行。"""

    run_dir: str
    kind: EventKind = field(default=EventKind.DONE, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.kind.value, "run_dir": self.run_dir}


@dataclass(frozen=True)
class ErrorEvent:
    """对应 cli.py 里 main() 捕获异常后打印"错误：..."这一行。"""

    phase: Phase
    reason: str
    video_path: str | None = None
    sample_id: str | None = None
    kind: EventKind = field(default=EventKind.ERROR, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.kind.value,
            "phase": self.phase.value,
            "reason": self.reason,
            "video_path": self.video_path,
            "sample_id": self.sample_id,
        }


Event = EngineStartEvent | ProgressEvent | RunDirEvent | DoneEvent | ErrorEvent
"""共享的事件 schema，生产者 pipeline.py（#67）/cli.py（#68），消费者 gui_support.py（#69）。"""


@dataclass(frozen=True)
class Fingerprint:
    exists: bool
    size: int | None
    mtime_ns: int | None
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Fingerprint":
        return cls(
            exists=bool(value["exists"]),
            size=None if value.get("size") is None else int(value["size"]),
            mtime_ns=(
                None if value.get("mtime_ns") is None else int(value["mtime_ns"])
            ),
            sha256=None if value.get("sha256") is None else str(value["sha256"]),
        )


@dataclass(frozen=True)
class Segment:
    start_ms: int
    end_ms: int
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Segment":
        return cls(
            start_ms=int(value["start_ms"]),
            end_ms=int(value["end_ms"]),
            text=str(value["text"]),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    root: Path
    video_path: Path
    target_path: Path
    initial_state: str
    video_fingerprint: Fingerprint
    target_fingerprint: Fingerprint
    video_duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "root": str(self.root),
            "video_path": str(self.video_path),
            "target_path": str(self.target_path),
            "initial_state": self.initial_state,
            "video_fingerprint": self.video_fingerprint.to_dict(),
            "target_fingerprint": self.target_fingerprint.to_dict(),
            "video_duration_s": self.video_duration_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Candidate":
        return cls(
            sample_id=str(value["sample_id"]),
            root=Path(str(value["root"])).resolve(),
            video_path=Path(str(value["video_path"])).resolve(),
            target_path=Path(str(value["target_path"])).resolve(),
            initial_state=str(value["initial_state"]),
            video_fingerprint=Fingerprint.from_dict(value["video_fingerprint"]),
            target_fingerprint=Fingerprint.from_dict(value["target_fingerprint"]),
            video_duration_s=(
                None
                if value.get("video_duration_s") is None
                else float(value["video_duration_s"])
            ),
        )


@dataclass(frozen=True)
class ProtectedText:
    path: Path
    fingerprint: Fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "fingerprint": self.fingerprint.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtectedText":
        return cls(
            path=Path(str(value["path"])).resolve(),
            fingerprint=Fingerprint.from_dict(value["fingerprint"]),
        )


@dataclass
class PipelineOptions:
    roots: list[Path]
    runs_root: Path
    model_cache: Path
    device: str = "auto"
    compute_type: str = "auto"
    ffmpeg: Path | None = None
    language: str = "zh"
    funasr_model: str = "paraformer-zh"
    funasr_vad_model: str = "fsmn-vad"
    funasr_punc_model: str = "ct-punc"
    faster_whisper_model: str = "large-v3-turbo"
    hotwords_file: Path | None = None
    hash_videos: bool = True
    pilot_count: int = 2
