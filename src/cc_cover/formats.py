from __future__ import annotations

import codecs
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from cc_cover.models import FormatProfile, Segment


MMSS_PATTERN = re.compile(r"^\d{2}:\d{2}$")
HHMMSS_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")
MODEL_TAG_PATTERN = re.compile(r"<\|[^|>]+\|>")
REMOVED_PUNCTUATION = frozenset("，。！？；：、…“”‘’《》【】,.!?;:")
SENTENCE_PUNCTUATION = frozenset("，。！？；：、,.!?;:")


class FormatError(RuntimeError):
    pass


def decode_bytes(payload: bytes) -> tuple[str, str, bool]:
    if payload.startswith(codecs.BOM_UTF8):
        return payload[len(codecs.BOM_UTF8) :].decode("utf-8"), "utf-8", True
    if payload.startswith(codecs.BOM_UTF16_LE):
        return payload[len(codecs.BOM_UTF16_LE) :].decode("utf-16-le"), "utf-16-le", True
    if payload.startswith(codecs.BOM_UTF16_BE):
        return payload[len(codecs.BOM_UTF16_BE) :].decode("utf-16-be"), "utf-16-be", True
    for encoding in ("utf-8", "gb18030"):
        try:
            return payload.decode(encoding), encoding, False
        except UnicodeDecodeError:
            continue
    raise FormatError("无法识别 TXT 编码")


def encode_text(text: str, profile: FormatProfile) -> bytes:
    payload = text.encode(profile.encoding)
    if not profile.bom:
        return payload
    if profile.encoding == "utf-8":
        return codecs.BOM_UTF8 + payload
    if profile.encoding == "utf-16-le":
        return codecs.BOM_UTF16_LE + payload
    if profile.encoding == "utf-16-be":
        return codecs.BOM_UTF16_BE + payload
    return payload


def detect_profile(path: Path) -> FormatProfile:
    payload = path.read_bytes()
    if not payload:
        raise FormatError(f"格式样本为空：{path}")
    text, encoding, bom = decode_bytes(payload)
    crlf_count = text.count("\r\n")
    bare_lf_count = len(re.findall(r"(?<!\r)\n", text))
    newline_name = "crlf" if crlf_count >= bare_lf_count else "lf"
    lines = re.split(r"\r?\n", text)
    nonempty = [line for line in lines if line]
    mmss_count = sum(bool(MMSS_PATTERN.fullmatch(line.strip())) for line in nonempty)
    hhmmss_count = sum(bool(HHMMSS_PATTERN.fullmatch(line.strip())) for line in nonempty)
    timestamp_count = mmss_count + hhmmss_count
    style = "timed" if timestamp_count >= 2 else "plain"
    timestamp_style = "hhmmss" if hhmmss_count > mmss_count else "mmss"
    content_lines = [
        line
        for line in nonempty
        if not MMSS_PATTERN.fullmatch(line.strip())
        and not HHMMSS_PATTERN.fullmatch(line.strip())
    ]
    punctuation_count = sum(
        sum(character in SENTENCE_PUNCTUATION for character in line)
        for line in content_lines
    )
    strip_punctuation = punctuation_count <= max(2, len(content_lines) // 20)
    terminal_newline = text.endswith("\r\n") or text.endswith("\n")
    return FormatProfile(
        encoding=encoding,
        bom=bom,
        newline_name=newline_name,
        style=style,
        timestamp_style=timestamp_style,
        terminal_newline=terminal_newline,
        strip_sentence_punctuation=strip_punctuation,
        source_samples=(str(path.resolve()),),
    )


def profile_signature(profile: FormatProfile) -> tuple[object, ...]:
    return (
        profile.encoding,
        profile.bom,
        profile.newline_name,
        profile.style,
        profile.timestamp_style,
        profile.terminal_newline,
        profile.strip_sentence_punctuation,
    )


def dominant_profile(profiles: Sequence[FormatProfile]) -> FormatProfile:
    if not profiles:
        return FormatProfile()
    counts = Counter(profile_signature(profile) for profile in profiles)
    signature, _count = counts.most_common(1)[0]
    matching = [profile for profile in profiles if profile_signature(profile) == signature]
    source_samples = tuple(
        sample for profile in matching for sample in profile.source_samples
    )
    return FormatProfile(
        encoding=str(signature[0]),
        bom=bool(signature[1]),
        newline_name=str(signature[2]),
        style=str(signature[3]),
        timestamp_style=str(signature[4]),
        terminal_newline=bool(signature[5]),
        strip_sentence_punctuation=bool(signature[6]),
        source_samples=source_samples[:20],
    )


def choose_profile(
    target_path: Path,
    profiles_by_directory: dict[Path, list[FormatProfile]],
    root_profiles: Sequence[FormatProfile],
) -> FormatProfile:
    sibling_profiles = profiles_by_directory.get(target_path.parent.resolve(), [])
    return dominant_profile(sibling_profiles or list(root_profiles))


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


def normalize_text(raw: str, strip_sentence_punctuation: bool) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = MODEL_TAG_PATTERN.sub("", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    if strip_sentence_punctuation:
        output: list[str] = []
        for index, character in enumerate(text):
            if character in REMOVED_PUNCTUATION:
                output.append(punctuation_replacement(text, index))
            else:
                output.append(character)
        text = "".join(output)
    return re.sub(r"\s+", " ", text).strip()


def timestamp(start_ms: int, style: str) -> str:
    total_seconds = max(0, start_ms // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if style == "hhmmss":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    total_minutes = hours * 60 + minutes
    return f"{total_minutes:02d}:{seconds:02d}"


def render_timed(segments: Sequence[Segment], profile: FormatProfile) -> str:
    blocks: list[str] = []
    for segment in segments:
        text = normalize_text(segment.text, profile.strip_sentence_punctuation)
        if not text:
            continue
        blocks.append(
            timestamp(segment.start_ms, profile.timestamp_style)
            + profile.newline
            + text
        )
    if not blocks:
        raise FormatError("格式化后字幕为空")
    result = (profile.newline * 2).join(blocks)
    if profile.terminal_newline:
        result += profile.newline
    return result


def render_plain(segments: Sequence[Segment], profile: FormatProfile) -> str:
    paragraphs: list[str] = []
    current = ""
    previous_end: int | None = None
    for segment in segments:
        text = normalize_text(segment.text, profile.strip_sentence_punctuation)
        if not text:
            continue
        gap = segment.start_ms - previous_end if previous_end is not None else 0
        if current and gap >= 2500:
            paragraphs.append(current.strip())
            current = ""
        if current and current[-1:].isascii() and text[:1].isascii():
            current += " "
        current += text
        previous_end = segment.end_ms
        if len(current) >= 220:
            paragraphs.append(current.strip())
            current = ""
    if current:
        paragraphs.append(current.strip())
    if not paragraphs:
        raise FormatError("格式化后字幕为空")
    result = (profile.newline * 2).join(paragraphs)
    if profile.terminal_newline:
        result += profile.newline
    return result


def render_segments(segments: Sequence[Segment], profile: FormatProfile) -> bytes:
    ordered = sorted(segments, key=lambda item: (item.start_ms, item.end_ms))
    text = (
        render_timed(ordered, profile)
        if profile.style == "timed"
        else render_plain(ordered, profile)
    )
    payload = encode_text(text, profile)
    validate_rendered(payload, profile)
    return payload


def validate_rendered(payload: bytes, profile: FormatProfile) -> dict[str, object]:
    text, encoding, bom = decode_bytes(payload)
    if encoding != profile.encoding or bom != profile.bom:
        raise FormatError("输出编码与格式模板不一致")
    newline = profile.newline
    if profile.newline_name == "crlf" and re.search(r"(?<!\r)\n", text):
        raise FormatError("输出包含非 CRLF 换行")
    if profile.terminal_newline and not text.endswith(newline):
        raise FormatError("输出缺少模板要求的末尾换行")
    if profile.style == "plain":
        return {"style": "plain", "characters": len(text)}
    lines = text.split(newline)
    if profile.terminal_newline:
        lines = lines[:-1]
    timestamps: list[int] = []
    lengths: list[int] = []
    index = 0
    expected_pattern = HHMMSS_PATTERN if profile.timestamp_style == "hhmmss" else MMSS_PATTERN
    while index < len(lines):
        if index + 1 >= len(lines):
            raise FormatError("时间字幕块不完整")
        timecode = lines[index]
        caption = lines[index + 1]
        if not expected_pattern.fullmatch(timecode) or not caption:
            raise FormatError(f"时间字幕块无效：{timecode!r}")
        parts = [int(value) for value in timecode.split(":")]
        if len(parts) == 2:
            total = parts[0] * 60 + parts[1]
        else:
            total = parts[0] * 3600 + parts[1] * 60 + parts[2]
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
        "first_timestamp": timestamp(timestamps[0] * 1000, profile.timestamp_style),
        "last_timestamp": timestamp(timestamps[-1] * 1000, profile.timestamp_style),
    }


def text_is_whitespace_only(payload: bytes) -> bool:
    if not payload:
        return True
    try:
        text, _encoding, _bom = decode_bytes(payload)
    except FormatError:
        return False
    return not text.strip()
