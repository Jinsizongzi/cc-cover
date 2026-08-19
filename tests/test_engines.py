from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from cc_cover.engines import (
    EngineError,
    FasterWhisperEngine,
    extract_audio,
    parse_ffmpeg_duration,
    probe_duration,
    resolve_device,
)
from cc_cover.models import Phase, PipelineOptions


def raw_segment(start: float, end: float, text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        avg_logprob=-0.5,
        no_speech_prob=0.01,
        compression_ratio=1.0,
    )


class FakeWhisperModel:
    """模拟支持 hotwords 参数的现代版本 faster-whisper。"""

    def __init__(self, raw_segments: list[SimpleNamespace]) -> None:
        self.raw_segments = raw_segments
        self.last_kwargs: dict[str, object] = {}

    def transcribe(
        self,
        path: str,
        language: str = "zh",
        beam_size: int = 5,
        best_of: int = 5,
        temperature: list[float] | None = None,
        vad_filter: bool = True,
        condition_on_previous_text: bool = False,
        word_timestamps: bool = False,
        compression_ratio_threshold: float = 2.4,
        hallucination_silence_threshold: float = 2.0,
        hotwords: str | None = None,
        initial_prompt: str | None = None,
        **kwargs: object,
    ) -> tuple[object, SimpleNamespace]:
        self.last_kwargs = dict(kwargs)
        if hotwords is not None:
            self.last_kwargs["hotwords"] = hotwords
        if initial_prompt is not None:
            self.last_kwargs["initial_prompt"] = initial_prompt
        return iter(self.raw_segments), SimpleNamespace(duration_after_vad=None)


class FakeWhisperModelWithoutHotwords:
    """模拟不支持 hotwords 参数的旧版本 faster-whisper（只有 initial_prompt）。"""

    def __init__(self, raw_segments: list[SimpleNamespace]) -> None:
        self.raw_segments = raw_segments
        self.last_kwargs: dict[str, object] = {}

    def transcribe(
        self,
        path: str,
        language: str = "zh",
        beam_size: int = 5,
        best_of: int = 5,
        temperature: list[float] | None = None,
        vad_filter: bool = True,
        condition_on_previous_text: bool = False,
        word_timestamps: bool = False,
        compression_ratio_threshold: float = 2.4,
        hallucination_silence_threshold: float = 2.0,
        initial_prompt: str | None = None,
        **kwargs: object,
    ) -> tuple[object, SimpleNamespace]:
        self.last_kwargs = dict(kwargs)
        if initial_prompt is not None:
            self.last_kwargs["initial_prompt"] = initial_prompt
        return iter(self.raw_segments), SimpleNamespace(duration_after_vad=None)


class FasterWhisperEngineTests(unittest.TestCase):
    def _engine(self, raw_segments: list[SimpleNamespace]) -> FasterWhisperEngine:
        options = PipelineOptions(
            roots=[Path.cwd()],
            runs_root=Path("runs"),
            model_cache=Path("models"),
        )
        engine = FasterWhisperEngine(options, "cpu", "int8")
        engine.model = FakeWhisperModel(raw_segments)
        return engine

    def _engine_without_hotwords_support(
        self, raw_segments: list[SimpleNamespace]
    ) -> FasterWhisperEngine:
        options = PipelineOptions(
            roots=[Path.cwd()],
            runs_root=Path("runs"),
            model_cache=Path("models"),
        )
        engine = FasterWhisperEngine(options, "cpu", "int8")
        engine.model = FakeWhisperModelWithoutHotwords(raw_segments)
        return engine

    def test_tail_segment_start_beyond_duration_is_skipped_with_log(self) -> None:
        raw_segments = [
            raw_segment(0.0, 5.0, "normal"),
            raw_segment(551.7018, 551.7020, "hallucinated tail"),
        ]
        engine = self._engine(raw_segments)

        with self.assertLogs("cc_cover.engines", level="WARNING") as logs:
            segments, metadata = engine.transcribe(
                Path("audio.wav"), 551.701312, []
            )

        self.assertEqual([segment.text for segment in segments], ["normal"])
        self.assertEqual(metadata["segment_count"], 1)
        self.assertTrue(
            any("零长度" in record for record in logs.output),
            logs.output,
        )

    def test_empty_hotwords_skip_engine_prompt_without_fallback(self) -> None:
        engine = self._engine([raw_segment(0.0, 5.0, "hello")])

        segments, _metadata = engine.transcribe(Path("audio.wav"), 5.0, [])

        self.assertEqual([segment.text for segment in segments], ["hello"])
        self.assertNotIn("hotwords", engine.model.last_kwargs)
        self.assertNotIn("initial_prompt", engine.model.last_kwargs)

    def test_nonempty_hotwords_sets_both_hotwords_and_initial_prompt_when_supported(
        self,
    ) -> None:
        """hotwords 和 initial_prompt 是两套独立机制，两个都支持时两个都传，
        不再是二选一的回退关系。"""
        engine = self._engine([raw_segment(0.0, 5.0, "hello")])

        engine.transcribe(Path("audio.wav"), 5.0, ["PyTorch", "Django"])

        self.assertEqual(engine.model.last_kwargs["hotwords"], "PyTorch, Django")
        self.assertEqual(
            engine.model.last_kwargs["initial_prompt"], "术语表：PyTorch、Django"
        )

    def test_nonempty_hotwords_falls_back_to_initial_prompt_when_hotwords_unsupported(
        self,
    ) -> None:
        """旧版本 faster-whisper 不支持 hotwords 参数时，仍然靠 initial_prompt 兜底。"""
        engine = self._engine_without_hotwords_support(
            [raw_segment(0.0, 5.0, "hello")]
        )

        engine.transcribe(Path("audio.wav"), 5.0, ["PyTorch", "Django"])

        self.assertNotIn("hotwords", engine.model.last_kwargs)
        self.assertEqual(
            engine.model.last_kwargs["initial_prompt"], "术语表：PyTorch、Django"
        )

    def test_segment_seconds_convert_to_milliseconds_with_rounding(self) -> None:
        engine = self._engine([raw_segment(0.0, 1.234)])

        segments, _metadata = engine.transcribe(Path("audio.wav"), 5.0, [])

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_ms, 0)
        self.assertEqual(segments[0].end_ms, 1234)

    def test_negative_start_is_clamped_to_zero(self) -> None:
        engine = self._engine([raw_segment(-0.5, 2.0)])

        segments, _metadata = engine.transcribe(Path("audio.wav"), 5.0, [])

        self.assertEqual(segments[0].start_ms, 0)

    def test_end_beyond_duration_is_clamped_to_duration(self) -> None:
        engine = self._engine([raw_segment(0.0, 10.0)])

        segments, _metadata = engine.transcribe(Path("audio.wav"), 5.0, [])

        self.assertEqual(segments[0].end_ms, 5000)

    def test_blank_text_segment_is_skipped(self) -> None:
        engine = self._engine(
            [raw_segment(0.0, 2.0, "  "), raw_segment(2.0, 4.0, "hello")]
        )

        segments, _metadata = engine.transcribe(Path("audio.wav"), 5.0, [])

        self.assertEqual([segment.text for segment in segments], ["hello"])

    def test_segment_metadata_is_preserved(self) -> None:
        engine = self._engine([raw_segment(0.0, 5.0)])

        segments, _metadata = engine.transcribe(Path("audio.wav"), 5.0, [])

        self.assertEqual(segments[0].metadata["avg_logprob"], -0.5)
        self.assertEqual(segments[0].metadata["no_speech_prob"], 0.01)
        self.assertEqual(segments[0].metadata["compression_ratio"], 1.0)


class EngineErrorPhaseTests(unittest.TestCase):
    def test_engine_error_requires_phase(self) -> None:
        with self.assertRaises(TypeError):
            EngineError("message")  # type: ignore[call-arg]

    def test_resolve_device_rejects_unknown_choice_with_setup_phase(self) -> None:
        with self.assertRaises(EngineError) as caught:
            resolve_device("quantum", "auto")

        self.assertEqual(caught.exception.phase, Phase.SETUP)
        self.assertIsNone(caught.exception.video_path)
        self.assertIsNone(caught.exception.sample_id)

    def test_engine_error_carries_video_path_and_sample_id_when_given(self) -> None:
        error = EngineError(
            "message",
            phase=Phase.FUNASR,
            video_path="E:/videos/sample.mp4",
            sample_id="CC-MISSING-00047",
        )

        self.assertEqual(error.video_path, "E:/videos/sample.mp4")
        self.assertEqual(error.sample_id, "CC-MISSING-00047")


class ExtractAudioTests(unittest.TestCase):
    def test_extract_audio_failure_carries_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "lesson.mp4"
            video.write_bytes(b"not a real video")
            output_wav = root / "lesson.wav"

            with self.assertRaises(EngineError) as caught:
                extract_audio(Path(sys.executable), video, output_wav)

        self.assertEqual(caught.exception.phase, Phase.AUDIO_EXTRACT)
        self.assertEqual(caught.exception.video_path, str(video))


class DurationProbeTests(unittest.TestCase):
    def test_parse_ffmpeg_duration_reads_common_formats(self) -> None:
        self.assertAlmostEqual(
            parse_ffmpeg_duration("  Duration: 00:01:23.45, start: 0.0"),
            83.45,
        )
        self.assertAlmostEqual(
            parse_ffmpeg_duration("Duration: 01:02:03, bitrate: 1 kb/s"),
            3723.0,
        )
        self.assertIsNone(parse_ffmpeg_duration("Duration: N/A"))
        self.assertIsNone(parse_ffmpeg_duration(""))
        self.assertIsNone(parse_ffmpeg_duration("no duration here"))

    def test_probe_duration_reads_real_media_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = root / "clip.wav"
            with wave.open(str(media), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)

            duration = probe_duration(media)

        self.assertIsNotNone(duration)
        self.assertAlmostEqual(duration, 1.0, delta=0.15)

    def test_probe_duration_returns_none_for_unreadable_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broken = root / "broken.mp4"
            broken.write_bytes(b"not a media file")

            duration = probe_duration(broken)

        self.assertIsNone(duration)


if __name__ == "__main__":
    unittest.main()
