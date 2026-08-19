from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from cc_cover.core.discovery import discover
from cc_cover.core.engines import (
    configure_model_cache,
    local_faster_whisper_model,
    local_funasr_model,
)
from cc_cover.core.models import (
    Candidate,
    EngineStartEvent,
    Fingerprint,
    Phase,
    PipelineOptions,
    ProgressEvent,
    Segment,
)
from cc_cover.core.pipeline import (
    PipelineError,
    SubtitlePipeline,
    emit_event,
    engine_phase,
    extract_filename_hotwords,
    load_hotwords,
    load_json,
    options_from_dict,
    options_to_dict,
    validate_segments,
    write_bytes_atomic,
)


def hotwords_options(hotwords_file: Path | None = None) -> PipelineOptions:
    return PipelineOptions(
        roots=[Path("videos").resolve()],
        runs_root=Path("runs").resolve(),
        model_cache=Path("models").resolve(),
        hotwords_file=hotwords_file,
    )


def hotword_candidate(name: str) -> Candidate:
    video = Path("videos") / f"{name}.mp4"
    return Candidate(
        sample_id=f"sample-{name}",
        root=Path("videos").resolve(),
        video_path=video,
        target_path=video.with_suffix(".txt"),
        initial_state="missing",
        video_fingerprint=Fingerprint(False, None, None, None),
        target_fingerprint=Fingerprint(False, None, None, None),
    )


class HotwordTests(unittest.TestCase):
    def test_filename_tokens_keep_only_alphanumeric_words_with_letters(self) -> None:
        cases = {
            "机器学习-01-第2讲": [],
            "PyTorch-2.0-第3章": ["PyTorch"],
            "AI入门-2026": ["AI"],
            "GPT4-4K": ["GPT4", "4K"],
            "C++基础": ["C"],
            "纯数字-123": [],
            "a1b": ["a1b"],
        }
        for stem, expected in cases.items():
            with self.subTest(stem=stem):
                self.assertEqual(extract_filename_hotwords(stem), expected)

    def test_load_hotwords_combines_user_file_then_filename_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hotwords_file = Path(temporary) / "hotwords.txt"
            hotwords_file.write_text(
                "# 注释\n\nAlpha, beta\n机器学习\n",
                encoding="utf-8",
            )
            candidates = [
                hotword_candidate("Alpha-01-第1讲"),
                hotword_candidate("beta-2.0"),
                hotword_candidate("PyTorch-入门"),
            ]

            result = load_hotwords(hotwords_options(hotwords_file), candidates)

        self.assertEqual(result, ["Alpha", "beta", "机器学习", "PyTorch"])

    def test_load_hotwords_dedupes_case_insensitively_keeping_first_form(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hotwords_file = Path(temporary) / "hotwords.txt"
            hotwords_file.write_text("GPT\ngpt\nGPT4\n", encoding="utf-8")

            result = load_hotwords(hotwords_options(hotwords_file), [])

        self.assertEqual(result, ["GPT", "GPT4"])

    def test_load_hotwords_caps_combined_unique_at_200_after_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hotwords_file = Path(temporary) / "hotwords.txt"
            file_terms = [f"term{i:03d}" for i in range(60)]
            hotwords_file.write_text("\n".join(file_terms) + "\n", encoding="utf-8")
            candidates = [hotword_candidate(f"extra{i}") for i in range(150)]

            result = load_hotwords(hotwords_options(hotwords_file), candidates)

        self.assertEqual(len(result), 200)
        self.assertEqual(result[:60], file_terms)
        self.assertEqual(result[60], "extra0")
        self.assertEqual(result[-1], "extra139")
        self.assertNotIn("extra140", result)

    def test_load_hotwords_returns_empty_without_fallback(self) -> None:
        candidates = [hotword_candidate("01"), hotword_candidate("机器学习")]

        self.assertEqual(load_hotwords(hotwords_options(), candidates), [])
        self.assertEqual(load_hotwords(hotwords_options(), []), [])

    def test_load_hotwords_raises_when_user_file_missing(self) -> None:
        with self.assertRaises(PipelineError) as caught:
            load_hotwords(hotwords_options(Path("missing-hotwords.txt")), [])

        self.assertEqual(caught.exception.phase, Phase.SETUP)


class PipelineHelperTests(unittest.TestCase):
    def test_pipeline_error_requires_phase(self) -> None:
        with self.assertRaises(TypeError):
            PipelineError("message")  # type: ignore[call-arg]

    def test_pipeline_error_carries_the_given_phase(self) -> None:
        error = PipelineError("message", phase=Phase.WRITEBACK)

        self.assertEqual(error.phase, Phase.WRITEBACK)
        self.assertIsNone(error.video_path)
        self.assertIsNone(error.sample_id)

    def test_pipeline_error_carries_video_path_and_sample_id_when_given(self) -> None:
        error = PipelineError(
            "message",
            phase=Phase.WRITEBACK,
            video_path="E:/videos/sample.mp4",
            sample_id="CC-MISSING-00047",
        )

        self.assertEqual(error.video_path, "E:/videos/sample.mp4")
        self.assertEqual(error.sample_id, "CC-MISSING-00047")

    def test_load_json_missing_file_carries_given_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"

            with self.assertRaises(PipelineError) as caught:
                load_json(missing, phase=Phase.VERIFY)

        self.assertEqual(caught.exception.phase, Phase.VERIFY)

    def test_atomic_write_replaces_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "subtitle.txt"
            write_bytes_atomic(target, b"first")
            write_bytes_atomic(target, b"second")
            self.assertEqual(target.read_bytes(), b"second")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_options_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            options = PipelineOptions(
                roots=[root],
                runs_root=root / "runs",
                model_cache=root / "models",
                hash_videos=False,
            )
            payload = options_to_dict(options)
            restored = options_from_dict(payload)

        self.assertEqual(restored.roots, options.roots)
        self.assertEqual(restored.runs_root, options.runs_root)
        self.assertFalse(restored.hash_videos)
        self.assertNotIn("include_whitespace_only", payload)
        self.assertNotIn("include_missing", payload)

    def test_existing_model_caches_resolve_to_local_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr_cache = root / "funasr"
            snapshot = (
                funasr_cache
                / "models"
                / "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch"
                / "snapshots"
                / "master"
            )
            snapshot.mkdir(parents=True)
            whisper_cache = root / "faster-whisper"
            whisper_model = whisper_cache / "large-v3"
            whisper_model.mkdir(parents=True)

            resolved_funasr = local_funasr_model("fsmn-vad", funasr_cache)
            resolved_whisper = local_faster_whisper_model("large-v3", whisper_cache)

        self.assertEqual(Path(resolved_funasr), snapshot.resolve())
        self.assertEqual(Path(resolved_whisper), whisper_model.resolve())

    def test_runtime_temp_stays_inside_funasr_cache(self) -> None:
        environment_names = ("TEMP", "TMP", "TMPDIR")
        previous_environment = {
            name: os.environ.get(name) for name in environment_names
        }
        previous_tempdir = tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory() as temporary:
                cache = Path(temporary) / "model-cache"
                configure_model_cache(cache, phase=Phase.FUNASR)
                expected = cache / "funasr" / ".runtime-temp"
                self.assertTrue(expected.is_dir())
                for name in environment_names:
                    self.assertEqual(Path(os.environ[name]), expected)
        finally:
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            tempfile.tempdir = previous_tempdir

    def test_validate_segments_error_includes_engine_and_video_context(self) -> None:
        segments = [
            Segment(0, 1000, "first"),
            Segment(1000, 1000, "zero-length tail"),
        ]

        with self.assertRaises(PipelineError) as caught:
            validate_segments(
                segments,
                10.0,
                engine="faster-whisper",
                sample_id="CC-MISSING-00047",
                video_path="E:/videos/sample.mp4",
            )

        message = str(caught.exception)
        self.assertIn("#1", message)
        self.assertIn("faster-whisper", message)
        self.assertIn("CC-MISSING-00047", message)
        self.assertIn("sample.mp4", message)
        self.assertIn("start_ms=1000", message)
        self.assertIn("end_ms=1000", message)
        self.assertIn("duration_ms=10000", message)
        self.assertEqual(caught.exception.phase, Phase.FASTER_WHISPER)
        self.assertEqual(caught.exception.sample_id, "CC-MISSING-00047")
        self.assertEqual(caught.exception.video_path, "E:/videos/sample.mp4")

    def test_validate_segments_funasr_error_carries_funasr_phase(self) -> None:
        with self.assertRaises(PipelineError) as caught:
            validate_segments([], 10.0, engine="funasr")

        self.assertEqual(caught.exception.phase, Phase.FUNASR)

    def test_engine_start_event_serializes_to_dict(self) -> None:
        event = EngineStartEvent(engine="funasr", device="cuda")

        self.assertEqual(
            event.to_dict(),
            {"event": "engine_start", "engine": "funasr", "device": "cuda"},
        )

    def test_progress_event_serializes_to_dict(self) -> None:
        event = ProgressEvent(
            engine="faster_whisper", index=2, total=5, video_path="a.mp4"
        )

        self.assertEqual(
            event.to_dict(),
            {
                "event": "progress",
                "engine": "faster_whisper",
                "index": 2,
                "total": 5,
                "video_path": "a.mp4",
            },
        )

    def test_emit_event_prints_exactly_one_json_line(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            emit_event(EngineStartEvent(engine="funasr", device="cpu"))

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(
            json.loads(lines[0]),
            {"event": "engine_start", "engine": "funasr", "device": "cpu"},
        )

    def test_engine_phase_normalizes_hyphen_and_underscore(self) -> None:
        self.assertEqual(engine_phase("funasr"), Phase.FUNASR)
        self.assertEqual(engine_phase("faster_whisper"), Phase.FASTER_WHISPER)
        self.assertEqual(engine_phase("faster-whisper"), Phase.FASTER_WHISPER)

    def test_create_manifest_records_excluded_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "keep.mp4").write_bytes(b"video-keep")
            (root / "keep.txt").write_bytes(b"")
            (root / "skip.mp4").write_bytes(b"video-skip")
            (root / "skip.txt").write_bytes(b"")
            report = discover([root], hash_videos=False)
            options = PipelineOptions(
                roots=[root],
                runs_root=root / "runs",
                model_cache=root / "models",
                hash_videos=False,
            )
            kept = tuple(
                candidate
                for candidate in report.candidates
                if candidate.video_path.name == "keep.mp4"
            )
            skipped = [
                candidate.video_path
                for candidate in report.candidates
                if candidate.video_path.name == "skip.mp4"
            ]
            filtered = replace(report, candidates=kept)

            pipeline = SubtitlePipeline.create(
                options,
                filtered,
                excluded_videos=skipped,
            )
            manifest = load_json(pipeline.run_dir / "manifest.json", phase=Phase.SETUP)

        self.assertEqual(len(manifest["candidates"]), 1)
        self.assertEqual(
            manifest["candidates"][0]["video_path"],
            str(root / "keep.mp4"),
        )
        self.assertEqual(manifest["excluded_videos"], [str(root / "skip.mp4")])


if __name__ == "__main__":
    unittest.main()
