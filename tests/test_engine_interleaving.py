from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cc_cover.discovery import discover
from cc_cover.models import Candidate, Segment
from cc_cover.pipeline import (
    PipelineOptions,
    SubtitlePipeline,
    options_to_dict,
    write_json_atomic,
)


SEGMENTS = [
    Segment(0, 1000, "你好世界"),
    Segment(1000, 2000, "欢迎使用"),
    Segment(2000, 3000, "字幕恢复工具"),
]


def _build_pipeline(
    root: Path, video_names: list[str]
) -> tuple[SubtitlePipeline, list[Path]]:
    """构造一个带 N 个候选的 pipeline，全部落在 remaining 批次里。"""
    videos: list[Path] = []
    for name in video_names:
        video = root / name
        video.write_bytes(f"video-{name}".encode("utf-8"))
        target = video.with_suffix(".txt")
        target.write_bytes(b"")
        videos.append(video)
    report = discover([root], hash_videos=False)
    candidates = sorted(report.candidates, key=lambda item: str(item.video_path))
    options = PipelineOptions(
        roots=[root],
        runs_root=root / "runs",
        model_cache=root / "models",
        hash_videos=False,
    )
    run_dir = root / "runs" / "run1"
    run_dir.mkdir(parents=True)
    sample_ids = [candidate.sample_id for candidate in candidates]
    manifest = {
        "run_id": "run1",
        "status": "running",
        "phases": {"all": sample_ids, "pilot": [], "remaining": sample_ids},
        "options": options_to_dict(options),
    }
    pipeline = SubtitlePipeline(options, run_dir, candidates, [], manifest)
    return pipeline, videos


def _write_engine_output(
    pipeline: SubtitlePipeline, engine: str, candidate: Candidate
) -> None:
    write_json_atomic(
        pipeline.engine_output(engine, candidate.sample_id),
        {
            "schema_version": "1.0",
            "sample_id": candidate.sample_id,
            "source_path": str(candidate.video_path),
            "engine": engine,
            "duration_seconds": 1.0,
            "elapsed_total_seconds": 0.1,
            "metadata": {},
            "segments": [segment.to_dict() for segment in SEGMENTS],
        },
    )


class EngineInterleavingTests(unittest.TestCase):
    def test_execute_interleaves_engines_across_multiple_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, videos = _build_pipeline(root, ["a.mp4", "b.mp4"])

            calls: list[tuple[str, str]] = []

            def make_engine(name: str) -> mock.Mock:
                engine = mock.Mock()

                def transcribe(wav_path, duration, hotwords):
                    calls.append((name, str(wav_path)))
                    return list(SEGMENTS), {}

                engine.transcribe.side_effect = transcribe
                return engine

            funasr_engine = make_engine("funasr")
            faster_engine = make_engine("faster_whisper")
            with mock.patch(
                "cc_cover.pipeline.FunASREngine", return_value=funasr_engine
            ), mock.patch(
                "cc_cover.pipeline.FasterWhisperEngine", return_value=faster_engine
            ), mock.patch(
                "cc_cover.pipeline.extract_audio", return_value=1.0
            ):
                pipeline.execute()

            engines_only = [engine for engine, _wav in calls]
            self.assertEqual(
                engines_only,
                ["funasr", "faster_whisper", "funasr", "faster_whisper"],
            )
            # 同一个候选两次调用用的是同一份 wav（共用一份音频抽取产物）。
            self.assertEqual(calls[0][1], calls[1][1])
            self.assertEqual(calls[2][1], calls[3][1])
            self.assertNotEqual(calls[0][1], calls[2][1])

    def test_execute_loads_and_closes_each_engine_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, _videos = _build_pipeline(root, ["a.mp4", "b.mp4"])
            # 一个 pilot、一个 remaining，覆盖两个批次都非空的场景。
            sample_ids = pipeline.manifest["phases"]["remaining"]
            pipeline.manifest["phases"]["pilot"] = [sample_ids[0]]
            pipeline.manifest["phases"]["remaining"] = [sample_ids[1]]

            funasr_class = mock.Mock(return_value=mock.Mock(
                transcribe=mock.Mock(return_value=(list(SEGMENTS), {}))
            ))
            faster_class = mock.Mock(return_value=mock.Mock(
                transcribe=mock.Mock(return_value=(list(SEGMENTS), {}))
            ))
            with mock.patch(
                "cc_cover.pipeline.FunASREngine", funasr_class
            ), mock.patch(
                "cc_cover.pipeline.FasterWhisperEngine", faster_class
            ), mock.patch(
                "cc_cover.pipeline.extract_audio", return_value=1.0
            ):
                pipeline.execute()

            funasr_class.assert_called_once()
            faster_class.assert_called_once()
            funasr_class.return_value.load.assert_called_once_with()
            faster_class.return_value.load.assert_called_once_with()
            funasr_class.return_value.close.assert_called_once_with()
            faster_class.return_value.close.assert_called_once_with()

    def test_load_engines_closes_already_loaded_engine_when_second_fails(
        self,
    ) -> None:
        """两个引擎打包加载，第二个失败时第一个已经装好的要被释放，不能泄漏。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, _videos = _build_pipeline(root, ["a.mp4"])

            funasr_engine = mock.Mock()
            faster_engine = mock.Mock()
            faster_engine.load.side_effect = RuntimeError("显存不足")
            with mock.patch(
                "cc_cover.pipeline.FunASREngine", return_value=funasr_engine
            ), mock.patch(
                "cc_cover.pipeline.FasterWhisperEngine", return_value=faster_engine
            ):
                with self.assertRaises(RuntimeError):
                    pipeline._load_engines_if_needed(
                        pipeline.manifest["phases"]["remaining"]
                    )

            funasr_engine.load.assert_called_once_with()
            funasr_engine.close.assert_called_once_with()
            faster_engine.close.assert_not_called()

    def test_execute_resumes_missing_engine_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, _videos = _build_pipeline(root, ["a.mp4"])
            candidate = pipeline.candidates[0]
            _write_engine_output(pipeline, "funasr", candidate)

            funasr_engine = mock.Mock()
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = (list(SEGMENTS), {})
            with mock.patch(
                "cc_cover.pipeline.FunASREngine", return_value=funasr_engine
            ), mock.patch(
                "cc_cover.pipeline.FasterWhisperEngine", return_value=faster_engine
            ), mock.patch(
                "cc_cover.pipeline.extract_audio", return_value=1.0
            ):
                pipeline.execute()

            funasr_engine.transcribe.assert_not_called()
            faster_engine.transcribe.assert_called_once()

    def test_execute_aborts_later_candidates_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, videos = _build_pipeline(root, ["a.mp4", "b.mp4"])

            funasr_engine = mock.Mock()
            funasr_engine.transcribe.side_effect = RuntimeError("推理失败")
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = (list(SEGMENTS), {})
            with mock.patch(
                "cc_cover.pipeline.FunASREngine", return_value=funasr_engine
            ), mock.patch(
                "cc_cover.pipeline.FasterWhisperEngine", return_value=faster_engine
            ), mock.patch(
                "cc_cover.pipeline.extract_audio", return_value=1.0
            ):
                with self.assertRaises(Exception):
                    pipeline.execute()

            # 第一个候选的 funasr 失败，faster_whisper 不会为它、也不会为
            # 第二个候选执行——整批中止。
            faster_engine.transcribe.assert_not_called()
            self.assertEqual(funasr_engine.transcribe.call_count, 1)

    def test_execute_extracts_audio_once_per_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, videos = _build_pipeline(root, ["a.mp4", "b.mp4"])

            funasr_engine = mock.Mock()
            funasr_engine.transcribe.return_value = (list(SEGMENTS), {})
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = (list(SEGMENTS), {})
            with mock.patch(
                "cc_cover.pipeline.FunASREngine", return_value=funasr_engine
            ), mock.patch(
                "cc_cover.pipeline.FasterWhisperEngine", return_value=faster_engine
            ), mock.patch(
                "cc_cover.pipeline.extract_audio", return_value=1.0
            ) as extract:
                pipeline.execute()

            self.assertEqual(extract.call_count, len(videos))


if __name__ == "__main__":
    unittest.main()
