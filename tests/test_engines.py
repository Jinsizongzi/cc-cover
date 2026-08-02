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


if __name__ == "__main__":
    unittest.main()
