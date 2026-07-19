from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


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
class FormatProfile:
    encoding: str = "utf-8"
    bom: bool = False
    newline_name: str = "crlf"
    style: str = "timed"
    timestamp_style: str = "mmss"
    terminal_newline: bool = True
    strip_sentence_punctuation: bool = True
    source_samples: tuple[str, ...] = ()

    @property
    def newline(self) -> str:
        return "\r\n" if self.newline_name == "crlf" else "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "bom": self.bom,
            "newline_name": self.newline_name,
            "style": self.style,
            "timestamp_style": self.timestamp_style,
            "terminal_newline": self.terminal_newline,
            "strip_sentence_punctuation": self.strip_sentence_punctuation,
            "source_samples": list(self.source_samples),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormatProfile":
        return cls(
            encoding=str(value.get("encoding", "utf-8")),
            bom=bool(value.get("bom", False)),
            newline_name=str(value.get("newline_name", "crlf")),
            style=str(value.get("style", "timed")),
            timestamp_style=str(value.get("timestamp_style", "mmss")),
            terminal_newline=bool(value.get("terminal_newline", True)),
            strip_sentence_punctuation=bool(
                value.get("strip_sentence_punctuation", True)
            ),
            source_samples=tuple(str(item) for item in value.get("source_samples", [])),
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
    profile: FormatProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "root": str(self.root),
            "video_path": str(self.video_path),
            "target_path": str(self.target_path),
            "initial_state": self.initial_state,
            "video_fingerprint": self.video_fingerprint.to_dict(),
            "target_fingerprint": self.target_fingerprint.to_dict(),
            "profile": self.profile.to_dict(),
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
            profile=FormatProfile.from_dict(value["profile"]),
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
    include_whitespace_only: bool = False
    include_missing: bool = False
    hash_videos: bool = True
    pilot_count: int = 2
