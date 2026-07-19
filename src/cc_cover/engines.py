from __future__ import annotations

import gc
import inspect
import math
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

from cc_cover.models import PipelineOptions, Segment


class EngineError(RuntimeError):
    pass


FUNASR_CACHE_NAMES = {
    "paraformer-zh": "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch": "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "fsmn-vad": "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch": "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic--punc_ct-transformer_cn-en-common-vocab471067-large",
    "iic/punc_ct-transformer_cn-en-common-vocab471067-large": "iic--punc_ct-transformer_cn-en-common-vocab471067-large",
}


def configure_model_cache(model_cache: Path) -> None:
    model_cache.mkdir(parents=True, exist_ok=True)
    runtime_temp = model_cache / ".runtime-temp"
    paths = {
        "MODELSCOPE_CACHE": model_cache / "funasr",
        "FUNASR_HOME": model_cache / "funasr",
        "HF_HOME": model_cache / "huggingface",
        "HF_HUB_CACHE": model_cache / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": model_cache / "huggingface" / "hub",
        "TORCH_HOME": model_cache / "torch",
        "TEMP": runtime_temp,
        "TMP": runtime_temp,
        "TMPDIR": runtime_temp,
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(runtime_temp)


def local_funasr_model(identifier: str, cache: Path) -> str:
    direct = Path(identifier).expanduser()
    if direct.is_dir():
        return str(direct.resolve())
    cache_name = FUNASR_CACHE_NAMES.get(identifier)
    if cache_name is None:
        return identifier
    snapshots = cache / "models" / cache_name / "snapshots"
    if not snapshots.is_dir():
        return identifier
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    candidates.sort(key=lambda path: (path.name != "master", -path.stat().st_mtime_ns))
    return str(candidates[0].resolve()) if candidates else identifier


def local_faster_whisper_model(identifier: str, cache: Path) -> str:
    direct = Path(identifier).expanduser()
    if direct.is_dir():
        return str(direct.resolve())
    cached = cache / identifier
    return str(cached.resolve()) if cached.is_dir() else identifier


def resolve_device(requested: str, compute_type: str) -> tuple[str, str]:
    if requested not in {"auto", "cuda", "cpu"}:
        raise EngineError(f"不支持的设备：{requested}")
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    try:
        import ctranslate2

        cuda_available = cuda_available and ctranslate2.get_cuda_device_count() > 0
    except Exception:
        if requested == "cuda":
            raise EngineError("无法导入 CTranslate2，不能使用 CUDA")
        cuda_available = False
    if requested == "cuda" and not cuda_available:
        raise EngineError("请求了 CUDA，但当前环境没有可用的 CUDA ASR 运行时")
    device = "cuda" if requested == "cuda" or requested == "auto" and cuda_available else "cpu"
    if compute_type == "auto":
        compute_type = "int8_float16" if device == "cuda" else "int8"
    return device, compute_type


def resolve_ffmpeg(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    environment = os.environ.get("CC_COVER_FFMPEG")
    if environment:
        candidates.append(Path(environment).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    try:
        import imageio_ffmpeg

        candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if candidate.is_file():
            return candidate.resolve()
    except Exception:
        pass
    from shutil import which

    system = which("ffmpeg")
    if system:
        return Path(system).resolve()
    raise EngineError(
        "找不到 FFmpeg。请运行 setup.ps1，或通过 --ffmpeg/CC_COVER_FFMPEG 指定。"
    )


def ffmpeg_version(ffmpeg: Path) -> str:
    completed = subprocess.run(
        [str(ffmpeg), "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise EngineError(f"FFmpeg 预检失败：{completed.stderr.strip()}")
    return completed.stdout.splitlines()[0] if completed.stdout else str(ffmpeg)


def extract_audio(ffmpeg: Path, video: Path, output_wav: Path) -> float:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not output_wav.is_file():
        raise EngineError(f"音频提取失败：{video}: {completed.stderr.strip()}")
    with wave.open(str(output_wav), "rb") as handle:
        frames = handle.getnframes()
        frame_rate = handle.getframerate()
    if frames <= 0 or frame_rate <= 0:
        raise EngineError(f"提取出的 WAV 无有效音频：{output_wav}")
    return frames / frame_rate


def numeric_milliseconds(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 and math.isfinite(number) else None


def timestamp_bounds(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    pairs: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = numeric_milliseconds(item[0])
        end = numeric_milliseconds(item[1])
        if start is not None and end is not None and end >= start:
            pairs.append((start, end))
    if not pairs:
        return None
    return pairs[0][0], pairs[-1][1]


def text_from_mapping(value: Mapping[str, Any]) -> str:
    for key in ("text", "sentence", "value"):
        text = value.get(key)
        if text is not None and str(text).strip():
            return str(text).strip()
    return ""


def flatten_result(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, (list, tuple)):
            result.extend(flatten_result(item))
    return result


def segment_from_mapping(
    value: Mapping[str, Any], duration_ms: float
) -> Segment | None:
    text = text_from_mapping(value)
    if not text:
        return None
    start = numeric_milliseconds(
        value.get("start", value.get("start_ms", value.get("begin")))
    )
    end = numeric_milliseconds(
        value.get("end", value.get("end_ms", value.get("finish")))
    )
    bounds = timestamp_bounds(value.get("timestamp", value.get("timestamps")))
    if bounds:
        start = bounds[0] if start is None else start
        end = bounds[1] if end is None else end
    start = 0.0 if start is None else min(max(0.0, start), duration_ms)
    end = duration_ms if end is None else min(max(start, end), duration_ms)
    if end <= start:
        end = min(duration_ms, start + 500.0)
    if end <= start:
        return None
    score = value.get("score")
    metadata: dict[str, Any] = {}
    if isinstance(score, (int, float)):
        metadata["confidence"] = float(score)
    return Segment(round(start), round(end), text, metadata)


def normalize_funasr(raw_result: Any, duration_seconds: float) -> list[Segment]:
    duration_ms = duration_seconds * 1000.0
    result: list[Segment] = []
    for item in flatten_result(raw_result):
        sentence_info = item.get("sentence_info")
        sentence_segments: list[Segment] = []
        if isinstance(sentence_info, (list, tuple)):
            for sentence in sentence_info:
                if not isinstance(sentence, dict):
                    continue
                segment = segment_from_mapping(sentence, duration_ms)
                if segment is not None:
                    sentence_segments.append(segment)
        if sentence_segments:
            result.extend(sentence_segments)
            continue
        segment = segment_from_mapping(item, duration_ms)
        if segment is not None:
            result.append(segment)
    result.sort(key=lambda item: (item.start_ms, item.end_ms))
    if not result:
        raise EngineError("FunASR 没有生成有效字幕段")
    return result


class FunASREngine:
    def __init__(self, options: PipelineOptions, device: str):
        self.options = options
        self.device = device
        self.model: Any = None

    def load(self) -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise EngineError("未安装 FunASR，请先运行 setup.ps1") from exc
        configure_model_cache(self.options.model_cache)
        cache = self.options.model_cache / "funasr"
        try:
            self.model = AutoModel(
                model=local_funasr_model(self.options.funasr_model, cache),
                vad_model=local_funasr_model(self.options.funasr_vad_model, cache),
                punc_model=local_funasr_model(self.options.funasr_punc_model, cache),
                device="cuda:0" if self.device == "cuda" else "cpu",
                hub="ms",
                cache_dir=str(cache),
                disable_update=True,
                max_single_segment_time=30000,
            )
        except Exception as exc:
            raise EngineError(f"FunASR 模型加载失败：{exc}") from exc

    def transcribe(
        self, audio_path: Path, duration_seconds: float, hotwords: Sequence[str]
    ) -> tuple[list[Segment], dict[str, Any]]:
        if self.model is None:
            raise EngineError("FunASR 尚未加载")
        parameters: dict[str, Any] = {
            "input": str(audio_path),
            "cache": {},
            "batch_size_s": 60,
            "sentence_timestamp": True,
            "use_itn": True,
            "merge_vad": True,
            "merge_length_s": 15,
        }
        if hotwords:
            parameters["hotword"] = " ".join(hotwords)
        started = time.perf_counter()
        try:
            raw_result = self.model.generate(**parameters)
        except Exception as exc:
            raise EngineError(f"FunASR 推理失败：{audio_path}: {exc}") from exc
        elapsed = time.perf_counter() - started
        segments = normalize_funasr(raw_result, duration_seconds)
        return segments, {
            "engine": "funasr",
            "model": self.options.funasr_model,
            "device": self.device,
            "elapsed_seconds": round(elapsed, 6),
            "duration_seconds": round(duration_seconds, 6),
            "real_time_factor": round(elapsed / duration_seconds, 6),
            "segment_count": len(segments),
        }

    def close(self) -> None:
        self.model = None
        release_gpu()


class FasterWhisperEngine:
    def __init__(self, options: PipelineOptions, device: str, compute_type: str):
        self.options = options
        self.device = device
        self.compute_type = compute_type
        self.model: Any = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise EngineError("未安装 faster-whisper，请先运行 setup.ps1") from exc
        configure_model_cache(self.options.model_cache)
        cache = self.options.model_cache / "faster-whisper"
        cache.mkdir(parents=True, exist_ok=True)
        identifier = local_faster_whisper_model(
            self.options.faster_whisper_model, cache
        )
        try:
            self.model = WhisperModel(
                identifier,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(cache),
            )
        except Exception as exc:
            raise EngineError(f"faster-whisper 模型加载失败：{exc}") from exc

    def transcribe(
        self, audio_path: Path, duration_seconds: float, hotwords: Sequence[str]
    ) -> tuple[list[Segment], dict[str, Any]]:
        if self.model is None:
            raise EngineError("faster-whisper 尚未加载")
        signature = inspect.signature(self.model.transcribe)
        parameters = signature.parameters
        kwargs: dict[str, Any] = {
            "language": self.options.language,
            "beam_size": 5,
            "best_of": 5,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "vad_filter": True,
            "condition_on_previous_text": False,
            "word_timestamps": False,
            "compression_ratio_threshold": 2.4,
            "hallucination_silence_threshold": 2.0,
        }
        if hotwords:
            if "hotwords" in parameters:
                kwargs["hotwords"] = ", ".join(hotwords)
            else:
                kwargs["initial_prompt"] = "术语表：" + "、".join(hotwords)
        unsupported = [key for key in kwargs if key not in parameters]
        if unsupported:
            raise EngineError("当前 faster-whisper 不支持参数：" + ", ".join(unsupported))
        started = time.perf_counter()
        try:
            iterator, info = self.model.transcribe(str(audio_path), **kwargs)
            raw_segments = list(iterator)
        except Exception as exc:
            raise EngineError(f"faster-whisper 推理失败：{audio_path}: {exc}") from exc
        elapsed = time.perf_counter() - started
        segments: list[Segment] = []
        for raw in raw_segments:
            start = max(0, round(float(raw.start) * 1000.0))
            end = min(
                round(duration_seconds * 1000.0),
                max(start + 1, round(float(raw.end) * 1000.0)),
            )
            text = str(raw.text).strip()
            if not text:
                continue
            metadata = {
                "avg_logprob": getattr(raw, "avg_logprob", None),
                "no_speech_prob": getattr(raw, "no_speech_prob", None),
                "compression_ratio": getattr(raw, "compression_ratio", None),
            }
            segments.append(Segment(start, end, text, metadata))
        if not segments:
            raise EngineError("faster-whisper 没有生成有效字幕段")
        return segments, {
            "engine": "faster-whisper",
            "model": self.options.faster_whisper_model,
            "device": self.device,
            "compute_type": self.compute_type,
            "elapsed_seconds": round(elapsed, 6),
            "duration_seconds": round(duration_seconds, 6),
            "duration_after_vad_seconds": getattr(info, "duration_after_vad", None),
            "real_time_factor": round(elapsed / duration_seconds, 6),
            "segment_count": len(segments),
        }

    def close(self) -> None:
        self.model = None
        release_gpu()


def release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
