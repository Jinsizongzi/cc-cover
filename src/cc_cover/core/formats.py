from __future__ import annotations

import codecs
import re
import statistics
import unicodedata
from typing import Sequence

from cc_cover.core.models import Segment


# 固定输出格式：MM:SS（不足 1 小时）/ H:MM:SS（满 1 小时，小时不限位数）。
# 校验按同规则放宽：分钟与小时字段均不限制位数。
MMSS_TIMESTAMP = re.compile(r"^(?P<minutes>\d+):(?P<seconds>\d{2})$")
HMMSS_TIMESTAMP = re.compile(r"^(?P<hours>\d+):(?P<minutes>\d{2}):(?P<seconds>\d{2})$")
MODEL_TAG_PATTERN = re.compile(r"<\|[^|>]+\|>")
REMOVED_PUNCTUATION = frozenset("，。！？；：、…“”‘’《》【】,.!?;:")


class FormatError(RuntimeError):
    pass


def decode_bytes(payload: bytes) -> tuple[str, str, bool]:
    if payload.startswith(codecs.BOM_UTF8):
        return payload[len(codecs.BOM_UTF8) :].decode("utf-8"), "utf-8", True
    if payload.startswith(codecs.BOM_UTF16_LE):
        return (
            payload[len(codecs.BOM_UTF16_LE) :].decode("utf-16-le"),
            "utf-16-le",
            True,
        )
    if payload.startswith(codecs.BOM_UTF16_BE):
        return (
            payload[len(codecs.BOM_UTF16_BE) :].decode("utf-16-be"),
            "utf-16-be",
            True,
        )
    for encoding in ("utf-8", "gb18030"):
        try:
            return payload.decode(encoding), encoding, False
        except UnicodeDecodeError:
            continue
    raise FormatError("无法识别 TXT 编码")


def punctuation_replacement(text: str, index: int) -> str:
    character = text[index]
    previous = text[index - 1] if index > 0 else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    previous_ascii = previous.isascii() and previous.isalnum()
    following_ascii = following.isascii() and following.isalnum()
    if character == "." and previous_ascii and following_ascii:
        return character
    if character == ":" and previous.isdigit() and following.isdigit():
        return character
    if previous_ascii and following_ascii:
        return " "
    return ""


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = MODEL_TAG_PATTERN.sub("", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    output: list[str] = []
    for index, character in enumerate(text):
        if character in REMOVED_PUNCTUATION:
            output.append(punctuation_replacement(text, index))
        else:
            output.append(character)
    text = "".join(output)
    return re.sub(r"\s+", " ", text).strip()


def timestamp(start_ms: int) -> str:
    total_seconds = max(0, start_ms // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_timestamp(timecode: str) -> int | None:
    match = MMSS_TIMESTAMP.fullmatch(timecode)
    if match:
        return int(match["minutes"]) * 60 + int(match["seconds"])
    match = HMMSS_TIMESTAMP.fullmatch(timecode)
    if match:
        return (
            int(match["hours"]) * 3600
            + int(match["minutes"]) * 60
            + int(match["seconds"])
        )
    return None


def render_timed(segments: Sequence[Segment]) -> str:
    blocks: list[str] = []
    for segment in segments:
        text = normalize_text(segment.text)
        if not text:
            continue
        blocks.append(timestamp(segment.start_ms) + "\r\n" + text)
    if not blocks:
        raise FormatError("格式化后字幕为空")
    return "\r\n\r\n".join(blocks) + "\r\n"


def render_segments(segments: Sequence[Segment]) -> bytes:
    ordered = sorted(segments, key=lambda item: (item.start_ms, item.end_ms))
    payload = render_timed(ordered).encode("utf-8")
    validate_rendered(payload)
    return payload


def validate_rendered(payload: bytes) -> dict[str, object]:
    text, encoding, bom = decode_bytes(payload)
    if encoding != "utf-8" or bom:
        raise FormatError("输出编码必须是 UTF-8 且不带 BOM")
    if re.search(r"(?<!\r)\n", text):
        raise FormatError("输出包含非 CRLF 换行")
    if not text.endswith("\r\n"):
        raise FormatError("输出缺少末尾换行")
    lines = text.split("\r\n")
    lines = lines[:-1]
    timestamps: list[int] = []
    lengths: list[int] = []
    index = 0
    while index < len(lines):
        if index + 1 >= len(lines):
            raise FormatError("时间字幕块不完整")
        timecode = lines[index]
        caption = lines[index + 1]
        total = parse_timestamp(timecode)
        if total is None or not caption:
            raise FormatError(f"时间字幕块无效：{timecode!r}")
        timestamps.append(total)
        lengths.append(len(caption))
        index += 2
        if index < len(lines):
            if lines[index] != "":
                raise FormatError("字幕段之间缺少空行")
            index += 1
    if timestamps != sorted(timestamps):
        raise FormatError("时间戳不是单调非递减")
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    return {
        "style": "timed",
        "segment_count": len(timestamps),
        "median_text_chars": statistics.median(lengths),
        "max_text_chars": max(lengths),
        "median_gap_seconds": statistics.median(gaps) if gaps else 0,
        "first_timestamp": timestamp(timestamps[0] * 1000),
        "last_timestamp": timestamp(timestamps[-1] * 1000),
    }


def text_is_whitespace_only(payload: bytes) -> bool:
    if not payload:
        return True
    try:
        text, _encoding, _bom = decode_bytes(payload)
    except FormatError:
        return False
    return not text.strip()
