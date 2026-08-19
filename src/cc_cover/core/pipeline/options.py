from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from cc_cover.core.models import DEFAULT_FASTER_WHISPER_MODEL, PipelineOptions


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
