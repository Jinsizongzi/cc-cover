from __future__ import annotations

import re
from typing import Sequence

from cc_cover.core.models import Candidate, Phase, PipelineOptions
from cc_cover.core.pipeline.errors import PipelineError

FILENAME_HOTWORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def extract_filename_hotwords(stem: str) -> list[str]:
    """从文件名主干提取热词：仅保留含英文字母的字母/数字 token，纯数字过滤。"""
    return [
        token
        for token in FILENAME_HOTWORD_PATTERN.findall(stem)
        if any(character.isalpha() for character in token)
    ]


def load_hotwords(
    options: PipelineOptions, candidates: Sequence[Candidate]
) -> list[str]:
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
