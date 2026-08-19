from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class Phase(str, Enum):
    """流水线失败发生在哪个处理环节；不含 scan——扫描阶段的错误是 DiscoveryError。"""

    SETUP = "setup"
    AUDIO_EXTRACT = "audio_extract"
    FINGERPRINT = "fingerprint"
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
    CANDIDATE_FAILED = "candidate_failed"


@dataclass(frozen=True)
class EngineStartEvent:
    """对应 pipeline.py 里"加载 {engine}：device=..."这一行。"""

    engine: str
    device: str
    kind: EventKind = field(default=EventKind.ENGINE_START, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.kind.value, "engine": self.engine, "device": self.device}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EngineStartEvent":
        return cls(engine=str(value["engine"]), device=str(value["device"]))


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProgressEvent":
        return cls(
            engine=str(value["engine"]),
            index=int(value["index"]),
            total=int(value["total"]),
            video_path=str(value["video_path"]),
        )


@dataclass(frozen=True)
class RunDirEvent:
    """对应 cli.py 里"运行目录：..."这一行。"""

    path: str
    kind: EventKind = field(default=EventKind.RUN_DIR, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.kind.value, "path": self.path}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunDirEvent":
        return cls(path=str(value["path"]))


@dataclass(frozen=True)
class DoneEvent:
    """对应 cli.py 里"字幕已写回并复核通过：..."/"复核通过，共 N 个..."这几行。"""

    run_dir: str
    kind: EventKind = field(default=EventKind.DONE, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.kind.value, "run_dir": self.run_dir}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DoneEvent":
        return cls(run_dir=str(value["run_dir"]))


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ErrorEvent":
        return cls(
            phase=Phase(str(value["phase"])),
            reason=str(value["reason"]),
            video_path=(
                None if value.get("video_path") is None else str(value["video_path"])
            ),
            sample_id=(
                None if value.get("sample_id") is None else str(value["sample_id"])
            ),
        )


@dataclass(frozen=True)
class CandidateFailedEvent:
    """对应候选处理失败但跳过继续（不中止整批）时 pipeline.py 打印的记录行。

    跟 ErrorEvent 的区别：ErrorEvent 是单次终止性错误，中止整个进程；
    CandidateFailedEvent 只标记一个候选被跳过，进程会继续处理下一个候选。
    """

    phase: Phase
    reason: str
    video_path: str
    sample_id: str
    kind: EventKind = field(default=EventKind.CANDIDATE_FAILED, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.kind.value,
            "phase": self.phase.value,
            "reason": self.reason,
            "video_path": self.video_path,
            "sample_id": self.sample_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateFailedEvent":
        return cls(
            phase=Phase(str(value["phase"])),
            reason=str(value["reason"]),
            video_path=str(value["video_path"]),
            sample_id=str(value["sample_id"]),
        )


Event = (
    EngineStartEvent
    | ProgressEvent
    | RunDirEvent
    | DoneEvent
    | ErrorEvent
    | CandidateFailedEvent
)
"""共享的事件 schema，生产者 pipeline.py（#67）/cli.py（#68），消费者 progress.py（#69）。"""

_EVENT_DECODERS: Mapping[str, Any] = {
    EventKind.ENGINE_START.value: EngineStartEvent.from_dict,
    EventKind.PROGRESS.value: ProgressEvent.from_dict,
    EventKind.RUN_DIR.value: RunDirEvent.from_dict,
    EventKind.DONE.value: DoneEvent.from_dict,
    EventKind.ERROR.value: ErrorEvent.from_dict,
    EventKind.CANDIDATE_FAILED.value: CandidateFailedEvent.from_dict,
}


def event_from_dict(value: Mapping[str, Any]) -> Event:
    """按 "event" 字段分发到对应 Event 子类型；kind 未知或字段不全时抛 ValueError/KeyError。"""
    decoder = _EVENT_DECODERS.get(value.get("event"))
    if decoder is None:
        raise ValueError(f"未知事件种类：{value.get('event')!r}")
    return decoder(value)


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


DEFAULT_FASTER_WHISPER_MODEL = "large-v3"
"""faster-whisper 默认模型；cli.py 的 DEFAULTS 字典、pipeline.py 的
options_from_dict() 都从这里导入，避免三处独立硬编码互相漏改。"""


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
    faster_whisper_model: str = DEFAULT_FASTER_WHISPER_MODEL
    hotwords_file: Path | None = None
    hash_videos: bool = True
    pilot_count: int = 2


ENGINE_NAMES = ("funasr", "faster_whisper")
"""本系统支持的两个转写引擎标识；core.pipeline.validate 与 core.pipeline.run 共用。"""
