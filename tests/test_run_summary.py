from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cc_cover.core.discovery import discover
from cc_cover.core.models import PipelineOptions, Segment
from cc_cover.core.pipeline import (
    PipelineError,
    SubtitlePipeline,
    run_status_label,
    write_summary,
)
from cc_cover.core.pipeline.io import write_json_atomic
from cc_cover.core.pipeline.options import options_to_dict
from cc_cover.core.pipeline.summary import build_summary_text


def committed_run_fixture(root: Path) -> Path:
    """构造一个已写回、含告警与排除项的运行目录，返回 run_dir。"""
    run_dir = root / "20260802_213000_12345"
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / "manifest.json",
        {
            "run_id": "20260802_213000_12345",
            "status": "committed",
            "created_at_utc": "2026-08-02T21:30:00+00:00",
            "updated_at_utc": "2026-08-02T21:46:00+00:00",
            "candidates": [
                {"sample_id": "CC-CANDIDATE-00001"},
                {"sample_id": "CC-CANDIDATE-00002"},
            ],
            "excluded_videos": ["E:\\videos\\skip.mp4"],
        },
    )
    write_json_atomic(
        run_dir / "stage_report.json",
        {
            "samples": [
                {
                    "sample_id": "CC-CANDIDATE-00001",
                    "video_path": "E:\\videos\\ok.mp4",
                    "passed": True,
                    "errors": [],
                    "warnings": ["文本密度告警：120.0 chars/min"],
                    "warning_count": 1,
                },
                {
                    "sample_id": "CC-CANDIDATE-00002",
                    "video_path": "E:\\videos\\ok2.mp4",
                    "passed": True,
                    "errors": [],
                    "warnings": [],
                    "warning_count": 0,
                },
            ]
        },
    )
    write_json_atomic(
        run_dir / "commit_report.json",
        {
            "committed_at_utc": "2026-08-02T21:45:00+00:00",
            "entry_count": 1,
            "entries": [
                {
                    "sample_id": "CC-CANDIDATE-00001",
                    "target_path": "E:\\videos\\ok.txt",
                    "target_size": 2048,
                    "target_sha256": "abc",
                }
            ],
        },
    )
    return run_dir


class RunStatusLabelTests(unittest.TestCase):
    def test_maps_known_statuses_to_chinese_labels(self) -> None:
        self.assertEqual(run_status_label("prepared"), "已准备")
        self.assertEqual(run_status_label("running"), "运行中")
        self.assertEqual(run_status_label("staged_all"), "已暂存（全部候选）")
        self.assertEqual(run_status_label("staged_partial"), "已暂存（部分候选）")
        self.assertEqual(run_status_label("committed"), "已完成")

    def test_unknown_status_passes_through(self) -> None:
        self.assertEqual(run_status_label("mystery"), "mystery")


class BuildSummaryTextTests(unittest.TestCase):
    def test_committed_run_lists_times_totals_warnings_and_writeback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = committed_run_fixture(Path(temporary))

            text = build_summary_text(run_dir)

        self.assertIn("CC-Cover 运行摘要", text)
        self.assertIn("运行 ID：20260802_213000_12345", text)
        self.assertIn("状态：已完成（committed）", text)
        self.assertIn("开始时间：2026-08-02T21:30:00+00:00", text)
        self.assertIn("结束时间：2026-08-02T21:45:00+00:00", text)
        self.assertIn("候选总数：2", text)
        self.assertIn("已排除：1（本次不处理）", text)
        self.assertIn("质量门禁通过：2", text)
        self.assertIn("质量门禁失败：0", text)
        self.assertIn("写回成功：1", text)
        self.assertIn("告警：1 个视频，共 1 条", text)
        self.assertIn("CC-CANDIDATE-00001 E:\\videos\\ok.mp4", text)
        self.assertIn("- 文本密度告警：120.0 chars/min", text)
        self.assertIn("CC-CANDIDATE-00001 E:\\videos\\ok.txt（2048 字节）", text)
        self.assertIn(f"运行目录：{run_dir.resolve()}", text)

    def test_partial_failed_run_lists_failures_and_no_writeback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "20260802_220000_1"
            run_dir.mkdir()
            write_json_atomic(
                run_dir / "manifest.json",
                {
                    "run_id": "20260802_220000_1",
                    "status": "staged_partial",
                    "created_at_utc": "2026-08-02T22:00:00+00:00",
                    "updated_at_utc": "2026-08-02T22:10:00+00:00",
                    "candidates": [{"sample_id": "CC-CANDIDATE-00007"}],
                },
            )
            write_json_atomic(
                run_dir / "stage_report.json",
                {
                    "samples": [
                        {
                            "sample_id": "CC-CANDIDATE-00007",
                            "video_path": "E:\\videos\\bad.mp4",
                            "passed": False,
                            "errors": ["文本密度异常：20.0 chars/min"],
                            "warnings": [],
                            "warning_count": 0,
                        }
                    ]
                },
            )

            text = build_summary_text(run_dir)

        self.assertIn("状态：已暂存（部分候选）（staged_partial）", text)
        self.assertIn("质量门禁通过：0", text)
        self.assertIn("质量门禁失败：1", text)
        self.assertIn("写回成功：0", text)
        self.assertIn("CC-CANDIDATE-00007 E:\\videos\\bad.mp4", text)
        self.assertIn("- 文本密度异常：20.0 chars/min", text)
        self.assertIn("（未写回）", text)

    def test_missing_artifacts_are_tolerated_with_unknown_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "empty-run"
            run_dir.mkdir()

            text = build_summary_text(run_dir)

        self.assertIn("状态：未知（unknown）", text)
        self.assertIn("候选总数：0", text)
        self.assertIn("已排除：0（本次不处理）", text)
        self.assertIn(f"运行目录：{run_dir.resolve()}", text)

    def test_excluded_count_flows_from_manifest_without_stage_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "prepared-run"
            run_dir.mkdir()
            write_json_atomic(
                run_dir / "manifest.json",
                {
                    "run_id": "prepared-run",
                    "status": "prepared",
                    "created_at_utc": "2026-08-02T23:00:00+00:00",
                    "candidates": [{"sample_id": "CC-CANDIDATE-00001"}],
                    "excluded_videos": [
                        "E:\\videos\\skip1.mp4",
                        "E:\\videos\\skip2.mp4",
                    ],
                },
            )

            text = build_summary_text(run_dir)

        self.assertIn("状态：已准备（prepared）", text)
        self.assertIn("已排除：2（本次不处理）", text)

    def test_excluded_count_flows_from_discovery_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "conflict-run"
            run_dir.mkdir()
            write_json_atomic(
                run_dir / "manifest.json",
                {
                    "run_id": "conflict-run",
                    "status": "committed",
                    "created_at_utc": "2026-08-02T21:30:00+00:00",
                    "updated_at_utc": "2026-08-02T21:46:00+00:00",
                    "candidates": [{"sample_id": "CC-CANDIDATE-00001"}],
                    "discovery": {
                        "video_count": 5,
                        "candidate_count": 3,
                    },
                },
            )

            text = build_summary_text(run_dir)

        self.assertIn("候选总数：1", text)
        self.assertIn("已排除：2（本次不处理）", text)

    def test_stopped_run_still_yields_coherent_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "20260802_230000_5"
            run_dir.mkdir()
            write_json_atomic(
                run_dir / "manifest.json",
                {
                    "run_id": "20260802_230000_5",
                    "status": "running",
                    "created_at_utc": "2026-08-02T23:00:00+00:00",
                    "updated_at_utc": "2026-08-02T23:04:00+00:00",
                    "candidates": [{"sample_id": "CC-CANDIDATE-00001"}],
                },
            )

            text = build_summary_text(run_dir)

        self.assertIn("状态：运行中（running）", text)
        self.assertIn("质量门禁通过：0", text)
        self.assertIn("（未写回）", text)

    def test_failed_final_review_lists_verify_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = committed_run_fixture(Path(temporary))
            write_json_atomic(
                run_dir / "verification.json",
                {
                    "passed": False,
                    "verified_count": 1,
                    "failures": ["视频变化：E:\\videos\\ok.mp4"],
                },
            )

            text = build_summary_text(run_dir)

        self.assertIn("最终复核：失败（1 项）", text)
        self.assertIn("- 视频变化：E:\\videos\\ok.mp4", text)

    def test_passed_final_review_reports_verified_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = committed_run_fixture(Path(temporary))
            write_json_atomic(
                run_dir / "verification.json",
                {"passed": True, "verified_count": 2, "failures": []},
            )

            text = build_summary_text(run_dir)

        self.assertIn("最终复核：2 项通过", text)


class WriteSummaryTests(unittest.TestCase):
    def test_write_summary_persists_summary_txt_and_returns_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = committed_run_fixture(Path(temporary))

            path = write_summary(run_dir)

            self.assertEqual(path, (run_dir / "summary.txt").resolve())
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn("CC-Cover 运行摘要", text)
            self.assertIn(f"运行目录：{run_dir.resolve()}", text)


def segments(texts: list[str], *, duration_seconds: float = 60.0) -> list[Segment]:
    duration_ms = max(1, round(duration_seconds * 1000.0))
    step = duration_ms // len(texts)
    result: list[Segment] = []
    for index, text in enumerate(texts):
        start = index * step
        result.append(Segment(start, min(duration_ms, start + step), text, {}))
    return result


class ExecuteWritesSummaryTests(unittest.TestCase):
    """execute() 每次运行都要写 summary.txt：成功与质量门禁失败均覆盖。"""

    def _staged_pipeline(
        self,
        root: Path,
        *,
        funasr_texts: list[str],
        faster_texts: list[str],
    ) -> SubtitlePipeline:
        video = root / "clip.mp4"
        video.write_bytes(b"video-bytes")
        target = root / "clip.txt"
        target.write_bytes(b"")
        report = discover([root], hash_videos=False)
        candidate = report.candidates[0]
        options = PipelineOptions(
            roots=[root],
            runs_root=root / "runs",
            model_cache=root / "models",
            hash_videos=False,
        )
        run_dir = root / "runs" / "run1"
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": "run1",
            "status": "prepared",
            "phases": {
                "pilot": [candidate.sample_id],
                "remaining": [],
                "all": [candidate.sample_id],
            },
            "candidates": [candidate.to_dict()],
            "options": options_to_dict(options),
        }
        pipeline = SubtitlePipeline(options, run_dir, [candidate], [], manifest)
        for engine, engine_segments in (
            ("funasr", segments(funasr_texts)),
            ("faster_whisper", segments(faster_texts)),
        ):
            write_json_atomic(
                pipeline.engine_output(engine, candidate.sample_id),
                {
                    "schema_version": "1.0",
                    "sample_id": candidate.sample_id,
                    "source_path": str(candidate.video_path),
                    "engine": engine,
                    "duration_seconds": 60.0,
                    "elapsed_total_seconds": 1.0,
                    "metadata": {},
                    "segments": [item.to_dict() for item in engine_segments],
                },
            )
        return pipeline

    def test_execute_writes_committed_summary_with_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline = self._staged_pipeline(
                root,
                funasr_texts=["a" * 9 + str(index) for index in range(5)],
                faster_texts=["a" * 9 + str(index) for index in range(5)],
            )
            with patch.object(SubtitlePipeline, "run_candidates"):
                pipeline.execute()

            summary = pipeline.run_dir / "summary.txt"
            self.assertTrue(summary.is_file())
            text = summary.read_text(encoding="utf-8")
            self.assertIn("状态：已完成（committed）", text)
            self.assertIn("质量门禁通过：1", text)
            self.assertIn("质量门禁失败：0", text)
            self.assertIn("写回成功：1", text)
            self.assertIn("告警：1 个视频，共 1 条", text)
            self.assertIn("- 文本密度告警", text)

    def test_execute_writes_summary_on_quality_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline = self._staged_pipeline(
                root,
                funasr_texts=["a" * 3 + str(index) for index in range(5)],
                faster_texts=["a" * 3 + str(index) for index in range(5)],
            )
            with patch.object(SubtitlePipeline, "run_candidates"):
                with self.assertRaises(PipelineError):
                    pipeline.execute()

            summary = pipeline.run_dir / "summary.txt"
            self.assertTrue(summary.is_file())
            text = summary.read_text(encoding="utf-8")
            self.assertIn("状态：已暂存（全部候选）（staged_all）", text)
            self.assertIn("质量门禁通过：0", text)
            self.assertIn("质量门禁失败：1", text)
            self.assertIn("（未写回）", text)
            self.assertNotIn("全部通过", text)


if __name__ == "__main__":
    unittest.main()
