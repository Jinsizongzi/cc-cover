from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cc_cover.formats import text_is_whitespace_only
from cc_cover.models import Candidate, Fingerprint, ProtectedText


VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}
)


class DiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscoveryReport:
    roots: tuple[Path, ...]
    candidates: tuple[Candidate, ...]
    protected_texts: tuple[ProtectedText, ...]
    video_count: int
    matched_text_count: int
    missing_text_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path, include_hash: bool = True) -> Fingerprint:
    if not path.exists():
        return Fingerprint(False, None, None, None)
    stat = path.stat()
    return Fingerprint(
        True,
        stat.st_size,
        stat.st_mtime_ns,
        sha256_file(path) if include_hash else None,
    )


def fingerprints_match(actual: Fingerprint, expected: Fingerprint) -> bool:
    if not fingerprints_match_quick(actual, expected):
        return False
    if expected.sha256 is None:
        # 基线无 hash（视频哈希保护关闭）：只按存在性 + 大小 + mtime 匹配。
        return True
    # 基线有 hash：必须做全量复核，快速指纹（sha256=None）不算匹配。
    return actual.sha256 is not None and actual.sha256 == expected.sha256


def fingerprints_match_quick(actual: Fingerprint, expected: Fingerprint) -> bool:
    """快速校验：只比较存在性、大小与修改时间，不比较哈希。"""
    return (
        actual.exists == expected.exists
        and actual.size == expected.size
        and actual.mtime_ns == expected.mtime_ns
    )


def normalize_roots(roots: Iterable[Path]) -> list[Path]:
    normalized: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise DiscoveryError(f"扫描目录不存在：{resolved}")
        if resolved not in normalized:
            normalized.append(resolved)
    if not normalized:
        raise DiscoveryError("至少需要一个扫描目录")
    return normalized


def discover(
    roots: Iterable[Path],
    include_whitespace_only: bool = False,
    include_missing: bool = False,
    hash_videos: bool = True,
) -> DiscoveryReport:
    normalized_roots = normalize_roots(roots)
    videos: list[tuple[Path, Path]] = []
    seen_videos: set[Path] = set()
    matched_text_count = 0
    missing_text_count = 0
    root_for_video: dict[Path, Path] = {}
    candidate_states: dict[Path, str] = {}
    for root in normalized_roots:
        for video in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
            if not video.is_file() or video.suffix.casefold() not in VIDEO_EXTENSIONS:
                continue
            resolved_video = video.resolve()
            if resolved_video in seen_videos:
                continue
            seen_videos.add(resolved_video)
            root_for_video[resolved_video] = root
            target = video.with_suffix(".txt").resolve()
            videos.append((resolved_video, target))
            if not target.exists():
                missing_text_count += 1
                if include_missing:
                    candidate_states[resolved_video] = "missing"
                continue
            matched_text_count += 1
            payload = target.read_bytes()
            if len(payload) == 0:
                candidate_states[resolved_video] = "zero_byte"
                continue
            if include_whitespace_only and text_is_whitespace_only(payload):
                candidate_states[resolved_video] = "whitespace_only"
                continue
    candidates: list[Candidate] = []
    for video, target in sorted(videos, key=lambda item: str(item[0]).casefold()):
        state = candidate_states.get(video)
        if state is None:
            continue
        root = root_for_video[video]
        candidates.append(
            Candidate(
                sample_id=f"CC-MISSING-{len(candidates) + 1:05d}",
                root=root,
                video_path=video,
                target_path=target,
                initial_state=state,
                video_fingerprint=fingerprint(video, include_hash=hash_videos),
                target_fingerprint=fingerprint(target, include_hash=True),
            )
        )
    candidate_targets = {candidate.target_path for candidate in candidates}
    protected: list[ProtectedText] = []
    seen_texts: set[Path] = set()
    for root in normalized_roots:
        for path in sorted(root.rglob("*.txt"), key=lambda item: str(item).casefold()):
            resolved = path.resolve()
            if resolved in seen_texts or resolved in candidate_targets:
                continue
            seen_texts.add(resolved)
            if path.is_file() and path.stat().st_size > 0:
                protected.append(ProtectedText(resolved, fingerprint(resolved, True)))
    return DiscoveryReport(
        roots=tuple(normalized_roots),
        candidates=tuple(candidates),
        protected_texts=tuple(protected),
        video_count=len(videos),
        matched_text_count=matched_text_count,
        missing_text_count=missing_text_count,
    )
