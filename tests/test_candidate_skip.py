from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cc_cover.core.discovery import discover
from cc_cover.core.engines import EngineError
from cc_cover.core.models import Candidate, Phase, PipelineOptions, Segment
from cc_cover.core.pipeline import PipelineError, SubtitlePipeline
from cc_cover.core.pipeline.io import write_json_atomic
from cc_cover.core.pipeline.options import options_to_dict
from cc_cover.core.pipeline.summary import build_summary_text


SEGMENTS = [
    Segment(0, 1000, "你好世界"),
    Segment(1000, 2000, "欢迎使用"),
    Segment(2000, 3000, "字幕恢复工具"),
]


def _build_pipeline(
    root: Path, video_names: list[str], *, candidate_failures: list[dict] | None = None
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
        "candidate_failures": candidate_failures or [],
    }
    pipeline = SubtitlePipeline(options, run_dir, candidates, [], manifest)
    return pipeline, videos


def _write_engine_output(
    pipeline: SubtitlePipeline,
    engine: str,
    candidate: Candidate,
    segments: list[Segment] = SEGMENTS,
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
            "segments": [segment.to_dict() for segment in segments],
        },
    )


class CandidateSkipAndContinueTests(unittest.TestCase):
    def test_execute_skips_candidate_specific_failure_and_writes_back_the_rest(
        self,
    ) -> None:
        """候选级失败（音频提取）跳过并记录，execute() 不抛异常、其余候选照常写回。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, videos = _build_pipeline(root, ["a.mp4", "b.mp4"])

            def failing_extract(ffmpeg, video_path, output_wav):
                if video_path.name == "a.mp4":
                    raise EngineError(
                        f"音频提取失败：{video_path}",
                        phase=Phase.AUDIO_EXTRACT,
                        video_path=str(video_path),
                    )
                return 1.0

            funasr_engine = mock.Mock()
            funasr_engine.transcribe.return_value = (list(SEGMENTS), {})
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = (list(SEGMENTS), {})
            with (
                mock.patch(
                    "cc_cover.core.pipeline.run.FunASREngine",
                    return_value=funasr_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.FasterWhisperEngine",
                    return_value=faster_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.extract_audio",
                    side_effect=failing_extract,
                ),
            ):
                pipeline.execute()

            candidate_a = next(
                c for c in pipeline.candidates if c.video_path.name == "a.mp4"
            )
            candidate_b = next(
                c for c in pipeline.candidates if c.video_path.name == "b.mp4"
            )

            self.assertIn(candidate_a.sample_id, pipeline.candidate_failures)
            self.assertNotIn(candidate_b.sample_id, pipeline.candidate_failures)
            self.assertEqual(
                pipeline.candidate_failures[candidate_a.sample_id]["phase"],
                Phase.AUDIO_EXTRACT.value,
            )
            # a 从未写回（保持发现时的空文件），b 正常写回。
            self.assertEqual(candidate_a.target_path.read_bytes(), b"")
            self.assertTrue(candidate_b.target_path.stat().st_size > 0)
            # 失败记录已经持久化进 manifest，供 resume 用。
            self.assertEqual(
                pipeline.manifest["candidate_failures"][0]["sample_id"],
                candidate_a.sample_id,
            )

    def test_stage_and_commit_survive_a_recorded_fingerprint_failure(self) -> None:
        """指纹失败候选被 run_candidates() 记录后，stage()/commit() 自己
        那道全量 validate_candidates() 校验不能拿同一份基线重新抓到它、再
        把整批中止掉——这两处都直接调用 validate_candidates(self.candidates,
        ...)，如果不排除已记录的候选级失败，会比对同一份指纹基线，把刚
        跳过的候选又当成新错误抓一次。这跟音频提取失败不同：音频提取
        失败不会碰视频指纹，不会触发这个回归，所以要单独测。

        用 run_candidates() 而不是完整的 execute()：execute() 开头另有一道
        独立的、#104 之前就存在的预检（run 开始前"候选是否跟 discover()
        时一致"），职责跟这里要测的不是一回事，混在一起会把两件事测糊。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, videos = _build_pipeline(root, ["a.mp4", "b.mp4"])
            candidate_a = next(
                c for c in pipeline.candidates if c.video_path.name == "a.mp4"
            )
            candidate_b = next(
                c for c in pipeline.candidates if c.video_path.name == "b.mp4"
            )

            stat = candidate_a.video_path.stat()
            candidate_a.video_path.write_bytes(b"changed-after-discovery")
            os.utime(
                candidate_a.video_path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )

            funasr_engine = mock.Mock()
            funasr_engine.transcribe.return_value = (list(SEGMENTS), {})
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = (list(SEGMENTS), {})
            with (
                mock.patch(
                    "cc_cover.core.pipeline.run.FunASREngine",
                    return_value=funasr_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.FasterWhisperEngine",
                    return_value=faster_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.extract_audio", return_value=1.0
                ),
            ):
                engines = pipeline._load_engines_if_needed(
                    pipeline.manifest["phases"]["remaining"]
                )
                pipeline.run_candidates(
                    pipeline.manifest["phases"]["remaining"], engines
                )

            self.assertIn(candidate_a.sample_id, pipeline.candidate_failures)
            self.assertEqual(
                pipeline.candidate_failures[candidate_a.sample_id]["phase"],
                Phase.FINGERPRINT.value,
            )

            pipeline.stage(pipeline.manifest["phases"]["all"])
            pipeline.commit()

            self.assertTrue(candidate_b.target_path.stat().st_size > 0)
            # a 仍是指纹失败前记录的原样：从未被写回。
            self.assertEqual(candidate_a.target_path.read_bytes(), b"")

    def test_resume_execute_does_not_get_stuck_on_a_recorded_fingerprint_failure(
        self,
    ) -> None:
        """resume 时 execute() 一开始的全量校验也要排除已记录的候选级失败，
        不然每次 resume 都会在第一行就重新抓到同一个已知问题、永远卡住。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            built, _videos = _build_pipeline(root, ["a.mp4", "b.mp4"])
            candidate_a = next(
                c for c in built.candidates if c.video_path.name == "a.mp4"
            )
            candidate_b = next(
                c for c in built.candidates if c.video_path.name == "b.mp4"
            )
            stat = candidate_a.video_path.stat()
            candidate_a.video_path.write_bytes(b"changed-after-discovery")
            os.utime(
                candidate_a.video_path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )

            manifest = dict(built.manifest)
            manifest["candidate_failures"] = [
                {
                    "sample_id": candidate_a.sample_id,
                    "video_path": str(candidate_a.video_path),
                    "phase": Phase.FINGERPRINT.value,
                    "reason": "此前已失败",
                }
            ]
            pipeline = SubtitlePipeline(
                built.options, built.run_dir, built.candidates, [], manifest
            )

            funasr_engine = mock.Mock()
            funasr_engine.transcribe.return_value = (list(SEGMENTS), {})
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = (list(SEGMENTS), {})
            with (
                mock.patch(
                    "cc_cover.core.pipeline.run.FunASREngine",
                    return_value=funasr_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.FasterWhisperEngine",
                    return_value=faster_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.extract_audio", return_value=1.0
                ),
            ):
                pipeline.execute()

            self.assertTrue(candidate_b.target_path.stat().st_size > 0)

    def test_execute_still_aborts_batch_when_transcribe_itself_fails(self) -> None:
        """engine.transcribe() 自身失败视为系统性失败，不记录进 candidate_failures，照常中止整批。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, _videos = _build_pipeline(root, ["a.mp4", "b.mp4"])

            funasr_engine = mock.Mock()
            funasr_engine.transcribe.side_effect = RuntimeError("推理失败")
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = (list(SEGMENTS), {})
            with (
                mock.patch(
                    "cc_cover.core.pipeline.run.FunASREngine",
                    return_value=funasr_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.FasterWhisperEngine",
                    return_value=faster_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.extract_audio", return_value=1.0
                ),
            ):
                with self.assertRaises(RuntimeError):
                    pipeline.execute()

            self.assertEqual(pipeline.candidate_failures, {})

    def test_resume_does_not_retry_a_previously_failed_candidate(self) -> None:
        """resume 时从 manifest 恢复的候选级失败不会被重新处理。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            built, _videos = _build_pipeline(root, ["a.mp4", "b.mp4"])
            candidate_a = next(
                c for c in built.candidates if c.video_path.name == "a.mp4"
            )
            # 模拟 resume()：同一个 run_dir/candidates，manifest 里带着上一次
            # 已经记录的候选级失败，重新构造 pipeline 实例。
            manifest = dict(built.manifest)
            manifest["candidate_failures"] = [
                {
                    "sample_id": candidate_a.sample_id,
                    "video_path": str(candidate_a.video_path),
                    "phase": Phase.AUDIO_EXTRACT.value,
                    "reason": "此前已失败",
                }
            ]
            pipeline = SubtitlePipeline(
                built.options, built.run_dir, built.candidates, [], manifest
            )

            funasr_engine = mock.Mock()
            funasr_engine.transcribe.return_value = (list(SEGMENTS), {})
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = (list(SEGMENTS), {})
            with (
                mock.patch(
                    "cc_cover.core.pipeline.run.FunASREngine",
                    return_value=funasr_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.FasterWhisperEngine",
                    return_value=faster_engine,
                ),
                mock.patch(
                    "cc_cover.core.pipeline.run.extract_audio", return_value=1.0
                ) as extract,
            ):
                engines = pipeline._load_engines_if_needed(
                    pipeline.manifest["phases"]["remaining"]
                )
                pipeline.run_candidates(
                    pipeline.manifest["phases"]["remaining"], engines
                )

            extract.assert_called_once()
            self.assertEqual(extract.call_args.args[1].name, "b.mp4")
            self.assertIn(candidate_a.sample_id, pipeline.candidate_failures)

    def test_quality_gate_failure_still_blocks_the_whole_batch(self) -> None:
        """质量门禁失败（非处理失败）维持现状：仍然是全有全无，不属于 #104 改动范围。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, _videos = _build_pipeline(root, ["a.mp4", "b.mp4"])
            candidate_a = next(
                c for c in pipeline.candidates if c.video_path.name == "a.mp4"
            )
            candidate_b = next(
                c for c in pipeline.candidates if c.video_path.name == "b.mp4"
            )

            # a 的字幕段结构合法但数量不足，触发质量门禁失败（而非处理失败）。
            insufficient = [Segment(0, 1000, "太短")]
            _write_engine_output(pipeline, "funasr", candidate_a, insufficient)
            _write_engine_output(pipeline, "faster_whisper", candidate_a, insufficient)
            _write_engine_output(pipeline, "funasr", candidate_b)
            _write_engine_output(pipeline, "faster_whisper", candidate_b)

            with self.assertRaises(PipelineError):
                pipeline.stage(pipeline.manifest["phases"]["all"])

            # 质量门禁失败不进 candidate_failures——那是 #104 专属的候选处理
            # 失败类别；写回目录（commit()）从未被调用，b 的目标 TXT 仍是
            # discover() 时的空文件，没有被单独写回。
            self.assertEqual(pipeline.candidate_failures, {})
            self.assertEqual(candidate_b.target_path.read_bytes(), b"")

    def test_summary_lists_processing_failures_separately_from_quality_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, _videos = _build_pipeline(
                root,
                ["a.mp4"],
                candidate_failures=[
                    {
                        "sample_id": "CC-CANDIDATE-00001",
                        "video_path": str(root / "a.mp4"),
                        "phase": Phase.FINGERPRINT.value,
                        "reason": "视频在转写前发生变化",
                    }
                ],
            )
            write_json_atomic(pipeline.run_dir / "manifest.json", pipeline.manifest)

            text = build_summary_text(pipeline.run_dir)

            self.assertIn("处理失败（已跳过）：1", text)
            self.assertIn("处理失败（已跳过）明细", text)
            self.assertIn("视频在转写前发生变化", text)
            self.assertIn("失败明细", text)


if __name__ == "__main__":
    unittest.main()
