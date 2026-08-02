from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from cc_cover.engines import FasterWhisperEngine
from cc_cover.models import PipelineOptions


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
        **kwargs: object,
    ) -> tuple[object, SimpleNamespace]:
        self.last_kwargs = dict(kwargs)
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


if __name__ == "__main__":
    unittest.main()
