from __future__ import annotations

import math
from typing import Sequence

from cc_cover.core.discovery import (
    fingerprint,
    fingerprints_match,
    fingerprints_match_quick,
)
from cc_cover.core.models import Candidate, Phase, ProtectedText, Segment
from cc_cover.core.pipeline.errors import PipelineError


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
        raise PipelineError("候选快照校验失败：\n" + "\n".join(failures), phase=phase)
