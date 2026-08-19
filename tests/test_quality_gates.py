from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from cc_cover.core.discovery import discover
from cc_cover.core.formats import render_segments
from cc_cover.core.models import Candidate, Fingerprint, PipelineOptions, Segment
from cc_cover.core.pipeline import SubtitlePipeline
from cc_cover.core.pipeline.io import write_json_atomic
from cc_cover.core.pipeline.options import options_to_dict


def make_candidate(root: Path) -> Candidate:
    return Candidate(
        sample_id="sample-1",
        root=root,
        video_path=root / "clip.mp4",
        target_path=root / "clip.txt",
        initial_state="missing",
        video_fingerprint=Fingerprint(False, None, None, None),
        target_fingerprint=Fingerprint(False, None, None, None),
    )


def make_pipeline(root: Path, candidate: Candidate) -> SubtitlePipeline:
    options = PipelineOptions(
        roots=[root],
        runs_root=root / "runs",
        model_cache=root / "models",
    )
    run_dir = root / "runs" / f"run-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "run1",
        "status": "prepared",
        "phases": {"all": [candidate.sample_id]},
        "options": options_to_dict(options),
    }
    return SubtitlePipeline(options, run_dir, [candidate], [], manifest)


def segments(
    texts: list[str],
    *,
    duration_seconds: float = 60.0,
    metadata: dict | list[dict] | None = None,
) -> list[Segment]:
    duration_ms = max(1, round(duration_seconds * 1000.0))
    step = duration_ms // len(texts)
    metadata_list = (
        metadata if isinstance(metadata, list) else [metadata or {}] * len(texts)
    )
    result: list[Segment] = []
    for index, text in enumerate(texts):
        start = index * step
        result.append(
            Segment(
                start,
                min(duration_ms, start + step),
                text,
                metadata_list[index],
            )
        )
    return result


def quality_report_for(
    root: Path,
    funasr_texts: list[str],
    faster_texts: list[str],
    *,
    duration_seconds: float = 60.0,
    faster_metadata: dict | None = None,
    caption_payload: bytes | None = None,
) -> dict:
    candidate = make_candidate(root)
    funasr_segments = segments(funasr_texts, duration_seconds=duration_seconds)
    faster_segments = segments(
        faster_texts,
        duration_seconds=duration_seconds,
        metadata=faster_metadata,
    )
    pipeline = make_pipeline(root, candidate)
    if caption_payload is None:
        caption_payload = render_segments(funasr_segments)
    return pipeline.quality_report(
        candidate,
        caption_payload,
        funasr_segments,
        faster_segments,
        duration_seconds,
    )


class QualityGateWarningTests(unittest.TestCase):
    def test_error_thresholds_keep_density_below_30_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr = ["a" * 3 + str(index) for index in range(5)]
            report = quality_report_for(root, funasr, funasr)

        self.assertTrue(any("文本密度异常" in error for error in report["errors"]))
        self.assertFalse(
            any("文本密度告警" in warning for warning in report["warnings"])
        )
        self.assertFalse(report["passed"])

    def test_error_thresholds_keep_density_above_1200_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr = ["a" * 33 + f"{index:02d}" for index in range(40)]
            report = quality_report_for(root, funasr, funasr)

        self.assertTrue(any("文本密度异常" in error for error in report["errors"]))
        self.assertFalse(
            any("文本密度告警" in warning for warning in report["warnings"])
        )
        self.assertFalse(report["passed"])

    def test_error_thresholds_keep_length_ratio_outside_045_220_blocking(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low = quality_report_for(
                root,
                ["a" * 7 + str(index) for index in range(5)],
                ["b" * 19 + str(index) for index in range(5)],
            )
            high = quality_report_for(
                root,
                ["a" * 45 + str(index) for index in range(5)],
                ["b" * 19 + str(index) for index in range(5)],
            )

        for report in (low, high):
            self.assertTrue(
                any("双模型全文长度比异常" in error for error in report["errors"])
            )
            self.assertFalse(
                any("双模型全文长度比告警" in warning for warning in report["warnings"])
            )
            self.assertFalse(report["passed"])

    def test_error_thresholds_keep_duplicate_run_above_4_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = quality_report_for(
                root,
                ["a" * 10 + "0"] * 5 + ["b" * 10 + "5"],
                ["a" * 10 + "0"] * 5 + ["b" * 10 + "5"],
            )

        self.assertTrue(any("连续重复字幕过多" in error for error in report["errors"]))
        self.assertFalse(
            any("连续重复字幕告警" in warning for warning in report["warnings"])
        )
        self.assertFalse(report["passed"])

    def test_error_thresholds_keep_insufficient_segments_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr = ["a" * 19 + str(index) for index in range(2)]
            report = quality_report_for(root, funasr, funasr)

        self.assertTrue(any("段数过少" in error for error in report["errors"]))
        self.assertFalse(report["passed"])

    def test_density_below_100_warns_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr = ["a" * 9 + str(index) for index in range(5)]
            report = quality_report_for(root, funasr, funasr)

        self.assertTrue(
            any("文本密度告警" in warning for warning in report["warnings"])
        )
        self.assertFalse(any("文本密度异常" in error for error in report["errors"]))
        self.assertTrue(report["passed"])
        self.assertEqual(report["warning_count"], len(report["warnings"]))
        self.assertTrue(report["has_warnings"])

    def test_density_above_600_warns_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr = ["a" * 34 + f"{index:02d}" for index in range(20)]
            report = quality_report_for(root, funasr, funasr)

        self.assertTrue(
            any("文本密度告警" in warning for warning in report["warnings"])
        )
        self.assertFalse(any("文本密度异常" in error for error in report["errors"]))
        self.assertTrue(report["passed"])

    def test_length_ratio_outside_080_140_warns_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low = quality_report_for(
                root,
                ["a" * 9 + str(index) for index in range(5)],
                ["b" * 19 + str(index) for index in range(5)],
            )
            high = quality_report_for(
                root,
                ["a" * 29 + str(index) for index in range(5)],
                ["b" * 19 + str(index) for index in range(5)],
            )

        for report in (low, high):
            self.assertTrue(
                any("双模型全文长度比告警" in warning for warning in report["warnings"])
            )
            self.assertFalse(
                any("双模型全文长度比异常" in error for error in report["errors"])
            )
            self.assertTrue(report["passed"])

    def test_duplicate_run_4_warns_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = quality_report_for(
                root,
                ["a" * 10 + "0"] * 4 + ["b" * 10 + "4", "c" * 10 + "5"],
                ["a" * 10 + "0"] * 4 + ["b" * 10 + "4", "c" * 10 + "5"],
            )

        self.assertTrue(
            any("连续重复字幕告警" in warning for warning in report["warnings"])
        )
        self.assertFalse(any("连续重复字幕过多" in error for error in report["errors"]))
        self.assertTrue(report["passed"])

    def test_median_segment_length_outside_3_40_warns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr = ["a" * 50 + str(index) for index in range(5)]
            report = quality_report_for(root, funasr, funasr)

        self.assertTrue(
            any("中位段长偏离常见范围" in warning for warning in report["warnings"])
        )
        self.assertTrue(report["passed"])

    def test_no_warnings_when_all_metrics_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr = ["a" * 29 + str(index) for index in range(5)]
            report = quality_report_for(root, funasr, funasr)

        self.assertEqual(report["errors"], [])
        self.assertEqual(report["warnings"], [])
        self.assertTrue(report["passed"])
        self.assertEqual(report["warning_count"], 0)
        self.assertFalse(report["has_warnings"])


class QualityGateConfidenceTests(unittest.TestCase):
    def test_low_avg_logprob_warns_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts = ["a" * 5 + str(index) for index in range(20)]
            report = quality_report_for(
                root,
                texts,
                texts,
                faster_metadata={"avg_logprob": -4.18, "no_speech_prob": 0.01},
            )

        self.assertTrue(
            any(
                "faster-whisper 置信度告警" in warning and "avg_logprob" in warning
                for warning in report["warnings"]
            )
        )
        self.assertFalse(
            any("no_speech_prob" in warning for warning in report["warnings"])
        )
        self.assertTrue(report["passed"])
        confidence = report["faster_whisper_confidence"]
        self.assertEqual(confidence["avg_logprob_min"], -4.18)
        self.assertEqual(confidence["avg_logprob_mean"], -4.18)
        self.assertEqual(confidence["no_speech_prob_max"], 0.01)
        self.assertEqual(confidence["checked_segments"], 20)

    def test_high_no_speech_prob_warns_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts = ["a" * 5 + str(index) for index in range(20)]
            report = quality_report_for(
                root,
                texts,
                texts,
                faster_metadata={"avg_logprob": -0.5, "no_speech_prob": 0.83},
            )

        self.assertTrue(
            any(
                "faster-whisper 置信度告警" in warning and "no_speech_prob" in warning
                for warning in report["warnings"]
            )
        )
        self.assertFalse(
            any("avg_logprob" in warning for warning in report["warnings"])
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["faster_whisper_confidence"]["no_speech_prob_max"], 0.83
        )

    def test_confidence_metrics_aggregate_across_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts = ["a" * 5 + str(index) for index in range(3)]
            metadatas = [
                {"avg_logprob": -0.5, "no_speech_prob": 0.01},
                {"avg_logprob": -1.2, "no_speech_prob": 0.40},
                {"avg_logprob": -0.3, "no_speech_prob": 0.60},
            ]
            report = quality_report_for(
                root,
                texts,
                texts,
                faster_metadata=metadatas,
            )

        confidence = report["faster_whisper_confidence"]
        self.assertAlmostEqual(confidence["avg_logprob_mean"], -0.6666667)
        self.assertEqual(confidence["avg_logprob_min"], -1.2)
        self.assertEqual(confidence["no_speech_prob_max"], 0.60)
        self.assertEqual(confidence["checked_segments"], 3)
        self.assertTrue(any("avg_logprob" in warning for warning in report["warnings"]))
        self.assertFalse(
            any("no_speech_prob" in warning for warning in report["warnings"])
        )

    def test_missing_confidence_metadata_skips_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts = ["a" * 5 + str(index) for index in range(5)]
            report = quality_report_for(root, texts, texts)

        confidence = report["faster_whisper_confidence"]
        self.assertIsNone(confidence["avg_logprob_min"])
        self.assertIsNone(confidence["avg_logprob_mean"])
        self.assertIsNone(confidence["no_speech_prob_max"])
        self.assertEqual(confidence["checked_segments"], 5)
        self.assertFalse(any("置信度告警" in warning for warning in report["warnings"]))


class QualityGateConflictAuditTests(unittest.TestCase):
    def test_high_risk_conflicts_warn_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr = ["a" * 5 + str(index) for index in range(5)]
            faster = ["b" * 5 + str(index) for index in range(5)]
            report = quality_report_for(root, funasr, faster)

        self.assertTrue(
            any("high_risk 冲突审计：5 段" in warning for warning in report["warnings"])
        )
        self.assertEqual(report["high_risk_count"], 5)
        self.assertEqual(report["alignment_summary"]["high_risk_count"], 5)
        self.assertTrue(report["passed"])

    def test_no_high_risk_warning_when_audit_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts = ["a" * 5 + str(index) for index in range(5)]
            report = quality_report_for(root, texts, texts)

        self.assertEqual(report["high_risk_count"], 0)
        self.assertFalse(
            any("high_risk 冲突审计" in warning for warning in report["warnings"])
        )


class QualityGateFormatErrorTests(unittest.TestCase):
    def test_format_validation_failure_records_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts = ["a" * 5 + str(index) for index in range(5)]
            report = quality_report_for(
                root,
                texts,
                texts,
                caption_payload=b"not a valid caption payload\r\n",
            )

        self.assertTrue(any("格式校验失败" in error for error in report["errors"]))
        self.assertFalse(report["passed"])


class QualityGateStageTests(unittest.TestCase):
    def _staged_pipeline(
        self,
        root: Path,
        *,
        funasr_texts: list[str],
        faster_texts: list[str],
        faster_metadata: dict | None = None,
    ) -> tuple[SubtitlePipeline, dict, dict]:
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
            "phases": {"all": [candidate.sample_id]},
            "options": options_to_dict(options),
        }
        pipeline = SubtitlePipeline(options, run_dir, [candidate], [], manifest)
        payloads = {
            "funasr": segments(funasr_texts, duration_seconds=60.0),
            "faster_whisper": segments(
                faster_texts,
                duration_seconds=60.0,
                metadata=faster_metadata,
            ),
        }
        for engine, engine_segments in payloads.items():
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
        stage_report = pipeline.stage([candidate.sample_id])
        audit = json.loads(
            (run_dir / "audit" / f"{candidate.sample_id}.json").read_text(
                encoding="utf-8"
            )
        )
        persisted = json.loads(
            (run_dir / "stage_report.json").read_text(encoding="utf-8")
        )
        return pipeline, audit, persisted

    def test_stage_persists_warnings_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, audit, persisted = self._staged_pipeline(
                root,
                funasr_texts=["a" * 9 + str(index) for index in range(5)],
                faster_texts=["b" * 19 + str(index) for index in range(5)],
                faster_metadata={
                    "avg_logprob": -4.18,
                    "no_speech_prob": 0.9,
                },
            )

        sample_id = pipeline.candidates[0].sample_id
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["has_warnings"])
        self.assertEqual(audit["warning_count"], len(audit["warnings"]))
        self.assertTrue(any("文本密度告警" in warning for warning in audit["warnings"]))
        self.assertTrue(
            any("双模型全文长度比告警" in warning for warning in audit["warnings"])
        )
        self.assertTrue(
            any("faster-whisper 置信度告警" in warning for warning in audit["warnings"])
        )
        self.assertTrue(
            any("high_risk 冲突审计：5 段" in warning for warning in audit["warnings"])
        )
        self.assertEqual(persisted["warning_count"], audit["warning_count"])
        self.assertEqual(persisted["warning_sample_ids"], [sample_id])
        self.assertTrue(persisted["has_warnings"])
        self.assertEqual(persisted["samples"][0]["warnings"], audit["warnings"])
        self.assertEqual(
            persisted["samples"][0]["warning_count"],
            audit["warning_count"],
        )
        self.assertEqual(persisted["samples"][0]["high_risk_count"], 5)

    def test_stage_records_zero_warnings_when_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, audit, persisted = self._staged_pipeline(
                root,
                funasr_texts=["a" * 29 + str(index) for index in range(5)],
                faster_texts=["a" * 29 + str(index) for index in range(5)],
            )

        sample_id = pipeline.candidates[0].sample_id
        self.assertEqual(audit["warnings"], [])
        self.assertFalse(audit["has_warnings"])
        self.assertEqual(persisted["warning_count"], 0)
        self.assertEqual(persisted["warning_sample_ids"], [])
        self.assertFalse(persisted["has_warnings"])
        self.assertNotIn(sample_id, persisted["warning_sample_ids"])


if __name__ == "__main__":
    unittest.main()
