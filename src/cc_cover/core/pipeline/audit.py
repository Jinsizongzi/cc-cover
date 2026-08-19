from __future__ import annotations

import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Sequence

from cc_cover.core.models import Segment

ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:[._+#/-][A-Za-z0-9]+)*|\d+(?:\.\d+)?")


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
        mismatch = sorted(
            token_set(funasr_text).symmetric_difference(token_set(faster.text))
        )
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
