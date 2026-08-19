from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cc_cover.discovery import (
    DiscoveryReport,
    discover,
    fingerprint,
    fingerprints_match,
    fingerprints_match_quick,
)
from cc_cover.engines import (
    FasterWhisperEngine,
    FunASREngine,
    EngineError,
    extract_audio,
    ffmpeg_version,
    resolve_device,
    resolve_ffmpeg,
)
from cc_cover.formats import (
    FormatError,
    normalize_text,
    render_segments,
    validate_rendered,
)
from cc_cover.models import (
    Candidate,
    CandidateFailedEvent,
    DEFAULT_FASTER_WHISPER_MODEL,
    EngineStartEvent,
    Event,
    Phase,
    PipelineOptions,
    ProgressEvent,
    ProtectedText,
    Segment,
)


ASCII_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[._+#/-][A-Za-z0-9]+)*|\d+(?:\.\d+)?"
)
FILENAME_HOTWORD_PATTERN = re.compile(r"[A-Za-z0-9]+")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

WARNING_DENSITY_MIN = 100.0
WARNING_DENSITY_MAX = 600.0
WARNING_LENGTH_RATIO_MIN = 0.80
WARNING_LENGTH_RATIO_MAX = 1.40
WARNING_DUPLICATE_RUN_AT = 3
WARNING_MEDIAN_CHARS_MIN = 3.0
WARNING_MEDIAN_CHARS_MAX = 40.0
WARNING_AVG_LOGPROB_MIN = -1.0
WARNING_NO_SPEECH_PROB_MAX = 0.6


class PipelineError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        phase: Phase,
        video_path: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.video_path = video_path
        self.sample_id = sample_id


class _TerminalEngineFailure(Exception):
    """标记系统性失败：engine.transcribe() 自身抛出的异常、或引擎未加载这个
    编程错误。候选级失败（指纹校验、音频提取、字幕段校验）改为跳过当前
    候选、继续处理下一个；只有这两种视为系统性，需要照常向上传播、中止
    整批——用一个专门的包裹类型标记，而不是按异常类型区分，因为
    EngineError/PipelineError 两种类型在候选级失败和系统性失败里都会用到。
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


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


SUMMARY_FILENAME = "summary.txt"

RUN_STATUS_LABELS = {
    "prepared": "已准备",
    "running": "运行中",
    "staged_all": "已暂存（全部候选）",
    "staged_partial": "已暂存（部分候选）",
    "committed": "已完成",
    "unknown": "未知",
}


def run_status_label(status: str) -> str:
    """运行状态的用户可读标签；未知状态原样透传。"""
    return RUN_STATUS_LABELS.get(status, status)


def load_optional_json(path: Path) -> dict[str, Any] | None:
    """读取可选的运行产物 JSON；文件缺失或无效时返回 None。"""
    try:
        return load_json(path, phase=Phase.SETUP)
    except (PipelineError, OSError):
        return None


def _sample_line(sample: dict[str, Any]) -> str:
    return "{} {}".format(
        sample.get("sample_id", ""), sample.get("video_path", "")
    ).strip()


def build_summary_text(run_dir: Path) -> str:
    """从运行产物生成人读摘要文本；产物缺失时以 0 / 未知降级。"""
    run_dir = run_dir.expanduser().resolve()
    manifest = load_optional_json(run_dir / "manifest.json") or {}
    stage = load_optional_json(run_dir / "stage_report.json")
    commit = load_optional_json(run_dir / "commit_report.json")
    verification = load_optional_json(run_dir / "verification.json")

    run_id = str(manifest.get("run_id") or run_dir.name)
    status = str(manifest.get("status") or "unknown")
    started = manifest.get("created_at_utc")
    ended = None
    if commit is not None:
        ended = commit.get("committed_at_utc")
    if ended is None:
        ended = manifest.get("updated_at_utc")
    if ended is None:
        ended = started

    candidates = manifest.get("candidates") or []
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    excluded_videos = manifest.get("excluded_videos") or []
    excluded_count = len(excluded_videos) if isinstance(excluded_videos, list) else 0
    discovery = manifest.get("discovery")
    if isinstance(discovery, dict):
        # 已排除 = 发现视频总数 - 候选数（同 stem 冲突、显式排除项均不进入候选）。
        video_count = int(discovery.get("video_count") or 0)
        discovered_candidates = int(discovery.get("candidate_count") or 0)
        if video_count and discovered_candidates:
            excluded_count = max(0, video_count - discovered_candidates)

    samples = (stage.get("samples") or []) if stage else []
    sample_dicts = [item for item in samples if isinstance(item, dict)]
    passed_samples = [item for item in sample_dicts if bool(item.get("passed"))]
    failed_samples = [item for item in sample_dicts if not bool(item.get("passed"))]
    warning_samples = [item for item in sample_dicts if item.get("warnings")]
    warning_count = sum(int(item.get("warning_count") or 0) for item in warning_samples)

    entries = (commit.get("entries") or []) if commit else []
    committed_count = len(entries) if isinstance(entries, list) else 0

    verify_failures: list[str] = []
    if verification is not None and not bool(verification.get("passed", True)):
        verify_failures = [str(item) for item in verification.get("failures") or []]

    candidate_failures = manifest.get("candidate_failures") or []
    candidate_failure_dicts = [
        item for item in candidate_failures if isinstance(item, dict)
    ]

    lines = [
        "CC-Cover 运行摘要",
        "================",
        "",
        f"运行 ID：{run_id}",
        f"状态：{run_status_label(status)}（{status}）",
        f"开始时间：{started}",
        f"结束时间：{ended}",
        "",
        "统计",
        "----",
        f"候选总数：{candidate_count}",
        f"已排除：{excluded_count}（本次不处理）",
        f"处理失败（已跳过）：{len(candidate_failure_dicts)}",
        f"质量门禁通过：{len(passed_samples)}",
        f"质量门禁失败：{len(failed_samples)}",
        f"写回成功：{committed_count}",
        f"告警：{len(warning_samples)} 个视频，共 {warning_count} 条",
    ]
    if verification is not None:
        if verify_failures:
            lines.append(f"最终复核：失败（{len(verify_failures)} 项）")
        else:
            lines.append(
                f"最终复核：{int(verification.get('verified_count') or 0)} 项通过"
            )

    lines += ["", "告警明细", "--------"]
    if warning_samples:
        for sample in warning_samples:
            lines.append(_sample_line(sample))
            lines.extend(f"  - {warning}" for warning in sample.get("warnings") or [])
    else:
        lines.append("（无）")

    lines += ["", "处理失败（已跳过）明细", "--------"]
    if candidate_failure_dicts:
        for failure in candidate_failure_dicts:
            lines.append(
                "{} {}".format(
                    failure.get("sample_id", ""), failure.get("video_path", "")
                ).strip()
            )
            lines.append(f"  - {failure.get('reason', '')}")
    else:
        lines.append("（无）")

    lines += ["", "失败明细", "--------"]
    failure_lines: list[str] = []
    for sample in failed_samples:
        failure_lines.append(_sample_line(sample))
        failure_lines.extend(f"  - {error}" for error in sample.get("errors") or [])
    failure_lines.extend(f"  - {failure}" for failure in verify_failures)
    if failure_lines:
        lines.extend(failure_lines)
    else:
        lines.append("（无）")

    lines += ["", "写回清单", "--------"]
    if entries:
        for entry in entries:
            sample_id = entry.get("sample_id", "")
            target = entry.get("target_path", "")
            size = entry.get("target_size", "?")
            lines.append(f"{sample_id} {target}（{size} 字节）".strip())
    else:
        lines.append("（未写回）")

    lines += ["", "路径", "----", f"运行目录：{run_dir}"]
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path) -> Path:
    """在运行目录写入 summary.txt（UTF-8），返回其路径。"""
    run_dir = run_dir.expanduser().resolve()
    path = run_dir / SUMMARY_FILENAME
    write_bytes_atomic(path, build_summary_text(run_dir).encode("utf-8"))
    return path


@dataclass(frozen=True)
class CompletionStats:
    """完成弹窗展示的运行结果统计。"""

    elapsed_seconds: float | None
    written_count: int
    warning_count: int
    failed_count: int = 0


def _timestamp_epoch(value: Any) -> float | None:
    """ISO 时间戳转 epoch 秒；缺失或不可解析时返回 None。"""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def run_completion_stats(run_dir: Path) -> CompletionStats:
    """从运行产物读取完成弹窗需要的统计：总耗时、写回数、告警数。"""
    run_dir = run_dir.expanduser().resolve()
    manifest = load_optional_json(run_dir / "manifest.json")
    commit = load_optional_json(run_dir / "commit_report.json")
    stage = load_optional_json(run_dir / "stage_report.json")

    started = _timestamp_epoch(manifest.get("created_at_utc") if manifest else None)
    ended_raw = None
    if commit is not None:
        ended_raw = commit.get("committed_at_utc")
    if ended_raw is None and manifest is not None:
        ended_raw = manifest.get("updated_at_utc")
    ended = _timestamp_epoch(ended_raw)
    elapsed = ended - started if started is not None and ended is not None else None

    written = 0
    if commit is not None:
        written = int(commit.get("entry_count") or 0)
        entries = commit.get("entries")
        if not written and isinstance(entries, list):
            written = len(entries)

    warnings = int(stage.get("warning_count") or 0) if stage else 0
    candidate_failures = manifest.get("candidate_failures") if manifest else None
    failed_count = len(candidate_failures) if isinstance(candidate_failures, list) else 0
    return CompletionStats(
        elapsed_seconds=elapsed,
        written_count=written,
        warning_count=warnings,
        failed_count=failed_count,
    )


def options_to_dict(options: PipelineOptions) -> dict[str, Any]:
    return {
        "roots": [str(path) for path in options.roots],
        "runs_root": str(options.runs_root),
        "model_cache": str(options.model_cache),
        "device": options.device,
        "compute_type": options.compute_type,
        "ffmpeg": None if options.ffmpeg is None else str(options.ffmpeg),
        "language": options.language,
        "funasr_model": options.funasr_model,
        "funasr_vad_model": options.funasr_vad_model,
        "funasr_punc_model": options.funasr_punc_model,
        "faster_whisper_model": options.faster_whisper_model,
        "hotwords_file": (
            None if options.hotwords_file is None else str(options.hotwords_file)
        ),
        "hash_videos": options.hash_videos,
        "pilot_count": options.pilot_count,
    }


def options_from_dict(value: Mapping[str, Any]) -> PipelineOptions:
    return PipelineOptions(
        roots=[Path(str(path)).resolve() for path in value["roots"]],
        runs_root=Path(str(value["runs_root"])).resolve(),
        model_cache=Path(str(value["model_cache"])).resolve(),
        device=str(value.get("device", "auto")),
        compute_type=str(value.get("compute_type", "auto")),
        ffmpeg=(
            None
            if value.get("ffmpeg") in (None, "")
            else Path(str(value["ffmpeg"])).resolve()
        ),
        language=str(value.get("language", "zh")),
        funasr_model=str(value.get("funasr_model", "paraformer-zh")),
        funasr_vad_model=str(value.get("funasr_vad_model", "fsmn-vad")),
        funasr_punc_model=str(value.get("funasr_punc_model", "ct-punc")),
        faster_whisper_model=str(
            value.get("faster_whisper_model", DEFAULT_FASTER_WHISPER_MODEL)
        ),
        hotwords_file=(
            None
            if value.get("hotwords_file") in (None, "")
            else Path(str(value["hotwords_file"])).resolve()
        ),
        hash_videos=bool(value.get("hash_videos", True)),
        pilot_count=int(value.get("pilot_count", 2)),
    )


def extract_filename_hotwords(stem: str) -> list[str]:
    """从文件名主干提取热词：仅保留含英文字母的字母/数字 token，纯数字过滤。"""
    return [
        token
        for token in FILENAME_HOTWORD_PATTERN.findall(stem)
        if any(character.isalpha() for character in token)
    ]


def load_hotwords(options: PipelineOptions, candidates: Sequence[Candidate]) -> list[str]:
    values: list[str] = []
    if options.hotwords_file is not None:
        if not options.hotwords_file.is_file():
            raise PipelineError(
                f"热词文件不存在：{options.hotwords_file}", phase=Phase.SETUP
            )
        for line in options.hotwords_file.read_text(encoding="utf-8-sig").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            values.extend(part.strip() for part in text.split(",") if part.strip())
    for candidate in candidates:
        values.extend(extract_filename_hotwords(candidate.video_path.stem))
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique[:200]


def comparison_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def token_set(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text)
    return {item.casefold() for item in ASCII_TOKEN_PATTERN.findall(normalized)}


def overlap_ms(left: Segment, right: Segment) -> int:
    return max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def align_for_audit(
    funasr_segments: Sequence[Segment], faster_segments: Sequence[Segment]
) -> dict[str, Any]:
    alignments: list[dict[str, Any]] = []
    used_funasr: set[int] = set()
    for faster_index, faster in enumerate(faster_segments):
        matched = [
            (index, item)
            for index, item in enumerate(funasr_segments)
            if overlap_ms(item, faster) > 0
        ]
        used_funasr.update(index for index, _item in matched)
        funasr_text = "".join(item.text for _index, item in matched)
        left = comparison_text(funasr_text)
        right = comparison_text(faster.text)
        ratio = SequenceMatcher(None, left, right).ratio() if left and right else 0.0
        mismatch = sorted(token_set(funasr_text).symmetric_difference(token_set(faster.text)))
        alignments.append(
            {
                "faster_whisper_segment_index": faster_index,
                "funasr_segment_indexes": [index for index, _item in matched],
                "start_ms": faster.start_ms,
                "end_ms": faster.end_ms,
                "funasr_text": funasr_text,
                "faster_whisper_text": faster.text,
                "similarity_ratio": round(ratio, 6),
                "ascii_or_numeric_token_mismatch": mismatch,
                "high_risk": not matched or ratio < 0.60 or bool(mismatch),
                "decision": "keep_funasr_writeback_review_faster_whisper_only",
            }
        )
    for index, segment in enumerate(funasr_segments):
        if index in used_funasr:
            continue
        alignments.append(
            {
                "faster_whisper_segment_index": None,
                "funasr_segment_indexes": [index],
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "funasr_text": segment.text,
                "faster_whisper_text": "",
                "similarity_ratio": 0.0,
                "ascii_or_numeric_token_mismatch": sorted(token_set(segment.text)),
                "high_risk": True,
                "decision": "keep_funasr_writeback_missing_faster_whisper_overlap",
            }
        )
    alignments.sort(key=lambda item: (item["start_ms"], item["end_ms"]))
    ratios = [float(item["similarity_ratio"]) for item in alignments]
    return {
        "writeback_source_engine": "funasr",
        "faster_whisper_role": "second_candidate_and_conflict_audit_only",
        "alignment_count": len(alignments),
        "high_risk_count": sum(bool(item["high_risk"]) for item in alignments),
        "median_similarity_ratio": statistics.median(ratios) if ratios else 0.0,
        "alignments": alignments,
    }


def longest_duplicate_run(segments: Sequence[Segment]) -> int:
    longest = 1
    current = 1
    previous = ""
    for segment in segments:
        normalized = comparison_text(segment.text)
        if normalized and normalized == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
        previous = normalized
    return longest


def faster_whisper_confidence(segments: Sequence[Segment]) -> dict[str, Any]:
    avg_logprob = [
        float(item.metadata["avg_logprob"])
        for item in segments
        if item.metadata.get("avg_logprob") is not None
    ]
    no_speech_prob = [
        float(item.metadata["no_speech_prob"])
        for item in segments
        if item.metadata.get("no_speech_prob") is not None
    ]
    return {
        "avg_logprob_mean": statistics.mean(avg_logprob) if avg_logprob else None,
        "avg_logprob_min": min(avg_logprob) if avg_logprob else None,
        "no_speech_prob_max": max(no_speech_prob) if no_speech_prob else None,
        "checked_segments": len(segments),
    }


_ENGINE_NAMES = ("funasr", "faster_whisper")


def engine_phase(engine: str | None) -> Phase:
    """引擎标识（不论连字符还是下划线拼法）到处理阶段的映射。"""
    normalized = (engine or "").replace("-", "_")
    return Phase.FUNASR if normalized == "funasr" else Phase.FASTER_WHISPER


def validate_segments(
    segments: Sequence[Segment],
    duration_seconds: float,
    *,
    engine: str | None = None,
    sample_id: str | None = None,
    video_path: str | None = None,
) -> None:
    context = (
        f"engine={engine}, sample={sample_id}, video={video_path}, "
        f"duration_ms={round(duration_seconds * 1000.0)}"
    )
    phase = engine_phase(engine)
    if not segments:
        raise PipelineError(
            f"引擎字幕段为空：{context}",
            phase=phase,
            sample_id=sample_id,
            video_path=video_path,
        )
    previous_start = -1
    maximum_end = math.ceil(duration_seconds * 1000.0) + 5000
    for index, segment in enumerate(segments):
        if (
            segment.start_ms < 0
            or segment.end_ms <= segment.start_ms
            or segment.start_ms < previous_start
            or segment.end_ms > maximum_end
            or not segment.text.strip()
        ):
            raise PipelineError(
                f"引擎字幕段无效：#{index} "
                f"({context}, start_ms={segment.start_ms}, end_ms={segment.end_ms})",
                phase=phase,
                sample_id=sample_id,
                video_path=video_path,
            )
        previous_start = segment.start_ms


def validate_protected(protected: Sequence[ProtectedText], *, phase: Phase) -> None:
    failures: list[str] = []
    for item in protected:
        actual = fingerprint(item.path, include_hash=True)
        if not fingerprints_match(actual, item.fingerprint):
            failures.append(str(item.path))
    if failures:
        raise PipelineError(
            "受保护的非空 TXT 发生变化：\n" + "\n".join(failures), phase=phase
        )


def validate_candidates(
    candidates: Sequence[Candidate], require_initial_target: bool, *, phase: Phase
) -> None:
    failures: list[str] = []
    for candidate in candidates:
        current_video = fingerprint(candidate.video_path, include_hash=False)
        if not fingerprints_match_quick(current_video, candidate.video_fingerprint):
            failures.append(f"视频变化：{candidate.video_path}")
        if require_initial_target:
            current_target = fingerprint(candidate.target_path, include_hash=True)
            if not fingerprints_match(current_target, candidate.target_fingerprint):
                failures.append(f"目标 TXT 状态变化：{candidate.target_path}")
    if failures:
        raise PipelineError(
            "候选快照校验失败：\n" + "\n".join(failures), phase=phase
        )


class SubtitlePipeline:
    def __init__(
        self,
        options: PipelineOptions,
        run_dir: Path,
        candidates: Sequence[Candidate],
        protected: Sequence[ProtectedText],
        manifest: dict[str, Any],
    ):
        self.options = options
        self.run_dir = run_dir.resolve()
        self.candidates = list(candidates)
        self.protected = list(protected)
        self.manifest = manifest
        self.ffmpeg = resolve_ffmpeg(options.ffmpeg)
        self.device, self.compute_type = resolve_device(
            options.device, options.compute_type
        )
        self.hotwords = load_hotwords(options, self.candidates)
        self.candidate_failures: dict[str, dict[str, Any]] = {
            str(item["sample_id"]): dict(item)
            for item in manifest.get("candidate_failures") or []
            if isinstance(item, dict) and item.get("sample_id")
        }

    @classmethod
    def create(
        cls,
        options: PipelineOptions,
        report: DiscoveryReport,
        excluded_videos: Sequence[Path] | None = None,
    ) -> "SubtitlePipeline":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise PipelineError(f"生成的 run_id 不安全：{run_id}", phase=Phase.SETUP)
        run_dir = options.runs_root.expanduser().resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        pilot_count = max(0, min(options.pilot_count, len(report.candidates)))
        pilot = [item.sample_id for item in report.candidates[:pilot_count]]
        remaining = [
            item.sample_id for item in report.candidates if item.sample_id not in set(pilot)
        ]
        manifest = {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "prepared",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "options": options_to_dict(options),
            "discovery": {
                "roots": [str(path) for path in report.roots],
                "video_count": report.video_count,
                "matched_text_count": report.matched_text_count,
                "missing_text_count": report.missing_text_count,
                "candidate_count": len(report.candidates),
                "conflict_count": len(report.conflicts),
                "protected_nonempty_txt_count": len(report.protected_texts),
            },
            "phases": {
                "pilot": pilot,
                "remaining": remaining,
                "all": [item.sample_id for item in report.candidates],
            },
            "candidates": [item.to_dict() for item in report.candidates],
            "protected_nonempty_txt": [item.to_dict() for item in report.protected_texts],
            "excluded_videos": sorted(
                {str(path.expanduser().resolve()) for path in (excluded_videos or ())}
            ),
            "runtime": None,
            "stage": None,
            "commit": None,
            "candidate_failures": [],
        }
        write_json_atomic(run_dir / "manifest.json", manifest)
        return cls(options, run_dir, report.candidates, report.protected_texts, manifest)

    @classmethod
    def resume(cls, run_dir: Path) -> "SubtitlePipeline":
        resolved = run_dir.expanduser().resolve()
        manifest = load_json(resolved / "manifest.json", phase=Phase.SETUP)
        options = options_from_dict(manifest["options"])
        candidates = [Candidate.from_dict(item) for item in manifest["candidates"]]
        protected = [
            ProtectedText.from_dict(item)
            for item in manifest["protected_nonempty_txt"]
        ]
        return cls(options, resolved, candidates, protected, manifest)

    def update_manifest(self, **changes: Any) -> None:
        self.manifest.update(changes)
        self.manifest["updated_at_utc"] = utc_now()
        write_json_atomic(self.run_dir / "manifest.json", self.manifest)

    def engine_output(self, engine: str, sample_id: str) -> Path:
        return self.run_dir / "engines" / engine / f"{sample_id}.json"

    def load_engine_output(self, engine: str, candidate: Candidate) -> dict[str, Any]:
        phase = engine_phase(engine)
        payload = load_json(self.engine_output(engine, candidate.sample_id), phase=phase)
        if payload.get("sample_id") != candidate.sample_id:
            raise PipelineError(
                f"{engine} sample_id 不匹配",
                phase=phase,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )
        if Path(str(payload.get("source_path", ""))).resolve() != candidate.video_path:
            raise PipelineError(
                f"{engine} source_path 不匹配",
                phase=phase,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )
        if payload.get("engine") != engine:
            raise PipelineError(
                f"{engine} 引擎声明不匹配",
                phase=phase,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )
        segments = [Segment.from_dict(item) for item in payload.get("segments", [])]
        validate_segments(
            segments,
            float(payload["duration_seconds"]),
            engine=engine,
            sample_id=candidate.sample_id,
            video_path=str(candidate.video_path),
        )
        return payload

    def output_complete(self, engine: str, candidate: Candidate) -> bool:
        try:
            self.load_engine_output(engine, candidate)
            return True
        except Exception:
            return False

    def _eligible_candidates(
        self, sample_ids: Sequence[str] | None = None
    ) -> list[Candidate]:
        """排除候选级失败（self.candidate_failures）后的候选；stage()/commit()/
        verify() 都只处理这个子集，让它们对候选级失败的候选保持沉默——
        跳过继续，不是整批中止。传 sample_ids 则先限定在这个子集里筛选。
        """
        selected = (
            self.candidates
            if sample_ids is None
            else [
                candidate
                for candidate in self.candidates
                if candidate.sample_id in sample_ids
            ]
        )
        return [
            candidate
            for candidate in selected
            if candidate.sample_id not in self.candidate_failures
        ]

    def _build_engine(self, engine_name: str) -> Any:
        if engine_name == "funasr":
            return FunASREngine(self.options, self.device)
        return FasterWhisperEngine(self.options, self.device, self.compute_type)

    def _load_engines_if_needed(self, sample_ids: Sequence[str]) -> dict[str, Any]:
        """按需构造并加载两个引擎；这批候选两个引擎都已完成时不加载。

        两个引擎总是配对加载/释放，不按引擎单独判断是否需要——一旦这批
        候选里有任何一个引擎的输出缺失，就把两个都装上，理由见 spec：
        模型常驻范围覆盖整个批次，不逐引擎精细判断。
        """
        selected = [
            candidate for candidate in self.candidates if candidate.sample_id in sample_ids
        ]
        needs_any = any(
            not self.output_complete(engine_name, candidate)
            for engine_name in _ENGINE_NAMES
            for candidate in selected
        )
        if not needs_any:
            return {}
        engines: dict[str, Any] = {}
        try:
            for engine_name in _ENGINE_NAMES:
                print(f"加载 {engine_name}：device={self.device}", flush=True)
                emit_event(EngineStartEvent(engine=engine_name, device=self.device))
                engine = self._build_engine(engine_name)
                engine.load()
                engines[engine_name] = engine
        except Exception:
            # 两个引擎打包加载，第二个失败时第一个已经真的装进显存了——
            # 不补这个 except 会让它随局部变量一起丢失、永远不会 close()。
            for loaded in engines.values():
                loaded.close()
            raise
        return engines

    def run_candidates(
        self, sample_ids: Sequence[str], engines: Mapping[str, Any]
    ) -> None:
        """逐候选交替处理：一个候选紧接着跑完两个引擎，再处理下一个候选。

        音频抽取和转写前后的指纹校验都收在候选级别，两个引擎共用同一份
        wav、共用同一对前后校验——不是每个引擎各来一遍。候选级失败（指纹
        校验、音频提取、字幕段校验）记录后跳过，继续处理下一批里的下一个
        候选，不再中止整批；只有 engine.transcribe() 自身失败（或引擎未
        加载这个编程错误）视为系统性失败，照常向上传播、中止整批。已经
        记录过失败的候选（含 resume 时从 manifest 恢复的）不会被重试。
        """
        selected = [
            candidate for candidate in self.candidates if candidate.sample_id in sample_ids
        ]
        pending = [
            candidate
            for candidate in selected
            if candidate.sample_id not in self.candidate_failures
            and any(
                not self.output_complete(engine_name, candidate)
                for engine_name in _ENGINE_NAMES
            )
        ]
        if not pending:
            return
        for index, candidate in enumerate(pending, start=1):
            try:
                self._run_one_candidate(candidate, index, len(pending), engines)
            except _TerminalEngineFailure as exc:
                raise exc.original from exc
            except (PipelineError, EngineError, OSError) as exc:
                self._record_candidate_failure(candidate, exc)

    def _run_one_candidate(
        self,
        candidate: Candidate,
        index: int,
        total: int,
        engines: Mapping[str, Any],
    ) -> None:
        before = fingerprint(candidate.video_path, include_hash=False)
        if not fingerprints_match_quick(before, candidate.video_fingerprint):
            raise PipelineError(
                f"视频在转写前发生变化：{candidate.video_path}",
                phase=Phase.FINGERPRINT,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )
        wav_path = self.run_dir / "work" / f"{candidate.sample_id}.wav"
        duration: float | None = None
        try:
            for engine_name in _ENGINE_NAMES:
                if self.output_complete(engine_name, candidate):
                    continue
                if engine_name not in engines:
                    # 调用方（execute()）本该已经用 _load_engines_if_needed()
                    # 保证：这批候选里只要有任何一个未完成的引擎，两个引擎
                    # 就都会被装好传进来。走到这里说明调用方没有遵守这个
                    # 前提——是编程错误，不是这一个候选独有的问题，视为
                    # 系统性失败，中止整批而不是跳过。
                    raise _TerminalEngineFailure(
                        PipelineError(
                            f"引擎 {engine_name} 未加载，无法处理候选",
                            phase=Phase.SETUP,
                            sample_id=candidate.sample_id,
                            video_path=str(candidate.video_path),
                        )
                    )
                engine = engines[engine_name]
                print(
                    f"[{engine_name} {index}/{total}] {candidate.video_path}",
                    flush=True,
                )
                emit_event(
                    ProgressEvent(
                        engine=engine_name,
                        index=index,
                        total=total,
                        video_path=str(candidate.video_path),
                    )
                )
                if duration is None:
                    duration = extract_audio(
                        self.ffmpeg, candidate.video_path, wav_path
                    )
                started = time.perf_counter()
                try:
                    segments, metadata = engine.transcribe(
                        wav_path, duration, self.hotwords
                    )
                except _TerminalEngineFailure:
                    raise
                except Exception as exc:
                    raise _TerminalEngineFailure(exc) from exc
                validate_segments(
                    segments,
                    duration,
                    engine=engine_name,
                    sample_id=candidate.sample_id,
                    video_path=str(candidate.video_path),
                )
                write_json_atomic(
                    self.engine_output(engine_name, candidate.sample_id),
                    {
                        "schema_version": "1.0",
                        "sample_id": candidate.sample_id,
                        "source_path": str(candidate.video_path),
                        "engine": engine_name,
                        "duration_seconds": round(duration, 6),
                        "elapsed_total_seconds": round(
                            time.perf_counter() - started, 6
                        ),
                        "metadata": metadata,
                        "segments": [segment.to_dict() for segment in segments],
                    },
                )
        finally:
            if wav_path.exists():
                wav_path.unlink()
        after = fingerprint(candidate.video_path, include_hash=False)
        if not fingerprints_match_quick(after, candidate.video_fingerprint):
            raise PipelineError(
                f"视频在转写后发生变化：{candidate.video_path}",
                phase=Phase.FINGERPRINT,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )

    def _record_candidate_failure(
        self, candidate: Candidate, exc: BaseException
    ) -> None:
        """记录一次候选级失败并跳过，不中止整批；持久化到 manifest 供 resume 用。"""
        phase = getattr(exc, "phase", None) or Phase.SETUP
        reason = str(exc)
        print(
            f"候选处理失败，跳过：{candidate.video_path}（{reason}）",
            flush=True,
        )
        emit_event(
            CandidateFailedEvent(
                phase=phase,
                reason=reason,
                video_path=str(candidate.video_path),
                sample_id=candidate.sample_id,
            )
        )
        self.candidate_failures[candidate.sample_id] = {
            "sample_id": candidate.sample_id,
            "video_path": str(candidate.video_path),
            "phase": phase.value,
            "reason": reason,
        }
        self.update_manifest(
            candidate_failures=list(self.candidate_failures.values())
        )

    def quality_report(
        self,
        candidate: Candidate,
        caption_payload: bytes,
        funasr_segments: Sequence[Segment],
        faster_segments: Sequence[Segment],
        duration_seconds: float,
    ) -> dict[str, Any]:
        try:
            format_metrics = validate_rendered(caption_payload)
        except FormatError as exc:
            format_metrics = {
                "style": "timed",
                "segment_count": 0,
                "median_text_chars": 0,
                "max_text_chars": 0,
                "median_gap_seconds": 0,
                "first_timestamp": "",
                "last_timestamp": "",
            }
            format_error = str(exc)
        else:
            format_error = None
        funasr_text = "".join(
            normalize_text(item.text) for item in funasr_segments
        )
        faster_text = "".join(item.text for item in faster_segments)
        normalized_funasr = comparison_text(funasr_text)
        normalized_faster = comparison_text(faster_text)
        minutes = duration_seconds / 60.0
        density = len(normalized_funasr) / minutes
        length_ratio = (
            len(normalized_funasr) / len(normalized_faster)
            if normalized_faster
            else math.inf
        )
        duplicate_run = longest_duplicate_run(funasr_segments)
        minimum_segments = max(3, int(minutes * 5))
        errors: list[str] = []
        warnings: list[str] = []
        if format_error is not None:
            errors.append(f"格式校验失败：{format_error}")
        if len(funasr_segments) < minimum_segments:
            errors.append(
                f"FunASR 字幕段数过少：{len(funasr_segments)} < {minimum_segments}"
            )
        if density < 30 or density > 1200:
            errors.append(f"文本密度异常：{density:.1f} chars/min")
        elif density < WARNING_DENSITY_MIN or density > WARNING_DENSITY_MAX:
            warnings.append(f"文本密度告警：{density:.1f} chars/min")
        if duplicate_run > 4:
            errors.append(f"连续重复字幕过多：{duplicate_run}")
        elif duplicate_run > WARNING_DUPLICATE_RUN_AT:
            warnings.append(f"连续重复字幕告警：{duplicate_run}")
        if (
            not math.isfinite(length_ratio)
            or length_ratio < 0.45
            or length_ratio > 2.20
        ):
            errors.append(f"双模型全文长度比异常：{length_ratio:.3f}")
        elif (
            length_ratio < WARNING_LENGTH_RATIO_MIN
            or length_ratio > WARNING_LENGTH_RATIO_MAX
        ):
            warnings.append(f"双模型全文长度比告警：{length_ratio:.3f}")
        median_chars = float(format_metrics["median_text_chars"])
        if (
            format_error is None
            and (
                median_chars < WARNING_MEDIAN_CHARS_MIN
                or median_chars > WARNING_MEDIAN_CHARS_MAX
            )
        ):
            warnings.append(f"中位段长偏离常见范围：{median_chars:g} 字")
        confidence = faster_whisper_confidence(faster_segments)
        if (
            confidence["avg_logprob_min"] is not None
            and confidence["avg_logprob_min"] < WARNING_AVG_LOGPROB_MIN
        ):
            warnings.append(
                "faster-whisper 置信度告警："
                f"avg_logprob 最低 {confidence['avg_logprob_min']:.2f}"
            )
        if (
            confidence["no_speech_prob_max"] is not None
            and confidence["no_speech_prob_max"] > WARNING_NO_SPEECH_PROB_MAX
        ):
            warnings.append(
                "faster-whisper 置信度告警："
                f"no_speech_prob 最高 {confidence['no_speech_prob_max']:.2f}"
            )
        alignment = align_for_audit(funasr_segments, faster_segments)
        high_risk_count = int(alignment["high_risk_count"])
        if high_risk_count:
            warnings.append(f"high_risk 冲突审计：{high_risk_count} 段需人工复核")
        return {
            "sample_id": candidate.sample_id,
            "video_path": str(candidate.video_path),
            "target_path": str(candidate.target_path),
            "duration_seconds": round(duration_seconds, 6),
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
            "has_warnings": bool(warnings),
            "warning_count": len(warnings),
            "output_format": {
                "style": "timed",
                "timestamp": "MM:SS / H:MM:SS",
                "encoding": "utf-8",
                "bom": False,
                "newline_name": "crlf",
                "terminal_newline": True,
            },
            "format_metrics": format_metrics,
            "funasr_segment_count": len(funasr_segments),
            "faster_whisper_segment_count": len(faster_segments),
            "text_density_chars_per_minute": round(density, 3),
            "funasr_to_faster_whisper_text_length_ratio": round(length_ratio, 6),
            "longest_consecutive_duplicate_run": duplicate_run,
            "caption_sha256": hashlib.sha256(caption_payload).hexdigest(),
            "caption_size": len(caption_payload),
            "faster_whisper_confidence": confidence,
            "high_risk_count": high_risk_count,
            "alignment_summary": {
                key: value for key, value in alignment.items() if key != "alignments"
            },
            "alignment": alignment,
        }

    def stage(self, sample_ids: Sequence[str]) -> dict[str, Any]:
        validate_protected(self.protected, phase=Phase.QUALITY_GATE)
        # 候选级失败（跳过继续）的候选排除在外——它们已经在 run_candidates()
        # 里被记录过一次，这里再拿同一份指纹基线重新校验只会重复抓到同一个
        # 已知问题、把它当成新错误再抛一次，中止掉本该照常写回的其余候选。
        validate_candidates(
            self._eligible_candidates(),
            require_initial_target=True,
            phase=Phase.QUALITY_GATE,
        )
        selected = self._eligible_candidates(sample_ids)
        reports: list[dict[str, Any]] = []
        for candidate in selected:
            funasr_payload = self.load_engine_output("funasr", candidate)
            faster_payload = self.load_engine_output("faster_whisper", candidate)
            funasr_segments = [
                Segment.from_dict(item) for item in funasr_payload["segments"]
            ]
            faster_segments = [
                Segment.from_dict(item) for item in faster_payload["segments"]
            ]
            duration = float(funasr_payload["duration_seconds"])
            if abs(duration - float(faster_payload["duration_seconds"])) > 0.05:
                raise PipelineError(
                    f"双模型音频时长不一致：{candidate.sample_id}",
                    phase=Phase.QUALITY_GATE,
                    sample_id=candidate.sample_id,
                    video_path=str(candidate.video_path),
                )
            caption_payload = render_segments(funasr_segments)
            report = self.quality_report(
                candidate,
                caption_payload,
                funasr_segments,
                faster_segments,
                duration,
            )
            write_bytes_atomic(
                self.run_dir / "prepared" / f"{candidate.sample_id}.txt",
                caption_payload,
            )
            write_json_atomic(
                self.run_dir / "audit" / f"{candidate.sample_id}.json",
                report,
            )
            reports.append(
                {key: value for key, value in report.items() if key != "alignment"}
            )
        stage_path = self.run_dir / "stage_report.json"
        previous: dict[str, Any] = {}
        if stage_path.is_file():
            previous = load_json(stage_path, phase=Phase.QUALITY_GATE)
        merged = {
            str(item["sample_id"]): item
            for item in previous.get("samples", [])
            if isinstance(item, dict) and item.get("sample_id")
        }
        for report in reports:
            merged[str(report["sample_id"])] = report
        all_ids = [
            sample_id
            for sample_id in self.manifest["phases"]["all"]
            if sample_id not in self.candidate_failures
        ]
        staged_all = all(sample_id in merged for sample_id in all_ids)
        all_passed = staged_all and all(
            bool(merged[sample_id]["passed"]) for sample_id in all_ids
        )
        selected_passed = all(
            sample_id in merged and bool(merged[sample_id]["passed"])
            for sample_id in sample_ids
            if sample_id not in self.candidate_failures
        )
        stage_report = {
            "schema_version": "1.0",
            "run_id": self.manifest["run_id"],
            "updated_at_utc": utc_now(),
            "selected_ids": list(sample_ids),
            "selected_passed": selected_passed,
            "staged_ids": sorted(merged),
            "staged_all": staged_all,
            "all_passed": all_passed,
            "warning_count": sum(
                int(item.get("warning_count", 0)) for item in merged.values()
            ),
            "warning_sample_ids": sorted(
                sample_id
                for sample_id, item in merged.items()
                if item.get("warnings")
            ),
            "has_warnings": any(item.get("warnings") for item in merged.values()),
            "samples": [merged[sample_id] for sample_id in sorted(merged)],
        }
        write_json_atomic(stage_path, stage_report)
        self.update_manifest(
            status="staged_all" if staged_all else "staged_partial",
            stage={
                "report": str(stage_path),
                "staged_all": staged_all,
                "all_passed": all_passed,
            },
        )
        if not selected_passed:
            raise PipelineError(
                "候选未全部就绪或未通过质量门禁，未写回课程目录", phase=Phase.QUALITY_GATE
            )
        return stage_report

    def commit(self) -> dict[str, Any]:
        stage_report = load_json(
            self.run_dir / "stage_report.json", phase=Phase.WRITEBACK
        )
        if not stage_report.get("staged_all") or not stage_report.get("all_passed"):
            raise PipelineError("全部字幕尚未通过质量门禁", phase=Phase.WRITEBACK)
        reports = {
            str(item["sample_id"]): item for item in stage_report["samples"]
        }
        validate_protected(self.protected, phase=Phase.WRITEBACK)
        # 候选级失败（跳过继续）的候选从未进入 stage_report，写回时排除，不
        # 强求它们凑齐——写回只覆盖真正处理完、通过质量门禁的那部分候选。
        eligible = self._eligible_candidates()
        validate_candidates(
            eligible, require_initial_target=True, phase=Phase.WRITEBACK
        )
        backups = self.run_dir / "backups"
        payloads: dict[str, bytes] = {}
        for candidate in eligible:
            prepared = self.run_dir / "prepared" / f"{candidate.sample_id}.txt"
            payload = prepared.read_bytes()
            validate_rendered(payload)
            expected_hash = str(reports[candidate.sample_id]["caption_sha256"])
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise PipelineError(
                    f"暂存字幕哈希不匹配：{prepared}",
                    phase=Phase.WRITEBACK,
                    sample_id=candidate.sample_id,
                    video_path=str(candidate.video_path),
                )
            payloads[candidate.sample_id] = payload
            backup_dir = backups / candidate.sample_id
            original = candidate.target_path.read_bytes() if candidate.target_path.exists() else b""
            write_bytes_atomic(backup_dir / "original.txt", original)
            write_json_atomic(
                backup_dir / "state.json",
                {
                    "target_path": str(candidate.target_path),
                    "initial_state": candidate.initial_state,
                    "fingerprint": candidate.target_fingerprint.to_dict(),
                },
            )
        # 写回前全量复核：视频哈希保护开启时做第 2 次（也是最后一次）全量读；
        # 关闭时只做快速校验。
        for candidate in eligible:
            current_video = fingerprint(
                candidate.video_path,
                include_hash=candidate.video_fingerprint.sha256 is not None,
            )
            if not fingerprints_match(current_video, candidate.video_fingerprint):
                raise PipelineError(
                    f"写回前视频已变化：{candidate.video_path}",
                    phase=Phase.WRITEBACK,
                    sample_id=candidate.sample_id,
                    video_path=str(candidate.video_path),
                )
        committed: list[Candidate] = []
        try:
            for candidate in eligible:
                write_bytes_atomic(candidate.target_path, payloads[candidate.sample_id])
                committed.append(candidate)
            for candidate in eligible:
                actual = candidate.target_path.read_bytes()
                if actual != payloads[candidate.sample_id]:
                    raise PipelineError(
                        f"写回后内容不一致：{candidate.target_path}",
                        phase=Phase.WRITEBACK,
                        sample_id=candidate.sample_id,
                        video_path=str(candidate.video_path),
                    )
                validate_rendered(actual)
            validate_protected(self.protected, phase=Phase.WRITEBACK)
        except Exception:
            for candidate in reversed(committed):
                backup = backups / candidate.sample_id / "original.txt"
                if candidate.initial_state == "missing":
                    if candidate.target_path.exists():
                        candidate.target_path.unlink()
                else:
                    write_bytes_atomic(candidate.target_path, backup.read_bytes())
            raise
        entries = [
            {
                "sample_id": candidate.sample_id,
                "video_path": str(candidate.video_path),
                "target_path": str(candidate.target_path),
                "target_size": candidate.target_path.stat().st_size,
                "target_sha256": hashlib.sha256(
                    candidate.target_path.read_bytes()
                ).hexdigest(),
            }
            for candidate in eligible
        ]
        report = {
            "schema_version": "1.0",
            "run_id": self.manifest["run_id"],
            "status": "committed",
            "committed_at_utc": utc_now(),
            "entry_count": len(entries),
            "entries": entries,
        }
        write_json_atomic(self.run_dir / "commit_report.json", report)
        self.update_manifest(status="committed", commit=report)
        return report

    def verify(self) -> dict[str, Any]:
        validate_protected(self.protected, phase=Phase.VERIFY)
        failures: list[str] = []
        entries: list[dict[str, Any]] = []
        # 候选级失败（跳过继续）的候选从未写回，复核时排除，不然会被误报
        # 成"目标字幕为空"。
        eligible = self._eligible_candidates()
        for candidate in eligible:
            current_video = fingerprint(candidate.video_path, include_hash=False)
            if not fingerprints_match_quick(current_video, candidate.video_fingerprint):
                failures.append(f"视频变化：{candidate.video_path}")
                continue
            prepared = self.run_dir / "prepared" / f"{candidate.sample_id}.txt"
            if not candidate.target_path.is_file() or candidate.target_path.stat().st_size == 0:
                failures.append(f"目标字幕为空：{candidate.target_path}")
                continue
            actual = candidate.target_path.read_bytes()
            if actual != prepared.read_bytes():
                failures.append(f"目标字幕与暂存产物不一致：{candidate.target_path}")
                continue
            metrics = validate_rendered(actual)
            entries.append(
                {
                    "sample_id": candidate.sample_id,
                    "target_path": str(candidate.target_path),
                    "size": len(actual),
                    "sha256": hashlib.sha256(actual).hexdigest(),
                    "metrics": metrics,
                }
            )
        report = {
            "schema_version": "1.0",
            "run_id": self.manifest["run_id"],
            "verified_at_utc": utc_now(),
            "passed": not failures,
            "failures": failures,
            "verified_count": len(entries),
            "entries": entries,
        }
        write_json_atomic(self.run_dir / "verification.json", report)
        if failures:
            raise PipelineError(
                "最终复核失败：\n" + "\n".join(failures), phase=Phase.VERIFY
            )
        return report

    def _finalize_summary(self) -> None:
        """尽力写 summary.txt；失败只告警，不掩盖运行结果。"""
        try:
            write_summary(self.run_dir)
        except (OSError, PipelineError) as exc:
            print(f"警告：无法生成运行摘要：{exc}", file=sys.stderr, flush=True)

    def execute(self) -> Path:
        validate_protected(self.protected, phase=Phase.SETUP)
        # 候选级失败（跳过继续）的候选排除在外——否则 resume() 之后每次
        # execute() 一开始就会拿同一份指纹基线重新抓到同一个已经记录过的
        # 候选级失败，永远卡在这里，连 run_candidates() 都进不去。
        validate_candidates(
            self._eligible_candidates(), require_initial_target=True, phase=Phase.SETUP
        )
        self.update_manifest(
            status="running",
            runtime={
                "device": self.device,
                "compute_type": self.compute_type,
                "ffmpeg": str(self.ffmpeg),
                "ffmpeg_version": ffmpeg_version(self.ffmpeg),
                "hotword_count": len(self.hotwords),
            },
        )
        try:
            pilot = list(self.manifest["phases"]["pilot"])
            remaining = list(self.manifest["phases"]["remaining"])
            engines = self._load_engines_if_needed(pilot + remaining)
            try:
                if pilot:
                    self.run_candidates(pilot, engines)
                    self.stage(pilot)
                if remaining:
                    self.run_candidates(remaining, engines)
                    self.stage(list(self.manifest["phases"]["all"]))
                elif pilot:
                    self.stage(pilot)
            finally:
                for engine in engines.values():
                    engine.close()
            self.commit()
            self.verify()
        finally:
            self._finalize_summary()
        return self.run_dir


def discover_for_options(options: PipelineOptions) -> DiscoveryReport:
    return discover(
        options.roots,
        hash_videos=options.hash_videos,
        probe_durations=True,
        ffmpeg=options.ffmpeg,
    )
