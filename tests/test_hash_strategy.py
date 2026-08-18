from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from cc_cover.discovery import (
    discover,
    fingerprint,
    fingerprints_match,
    fingerprints_match_quick,
    sha256_file,
)
from cc_cover.models import Phase, PipelineOptions, Segment
from cc_cover.pipeline import (
    PipelineError,
    SubtitlePipeline,
    options_to_dict,
    validate_candidates,
)


CAPTION_PAYLOAD = (
    "00:00\r\n你好世界\r\n\r\n00:02\r\nPyTorch 2.5\r\n".encode("utf-8")
)


EXECUTE_SEGMENTS = (
    Segment(0, 1000, "你好世界"),
    Segment(1000, 2000, "欢迎使用"),
    Segment(2000, 3000, "字幕恢复工具"),
)


class FingerprintMatchingTests(unittest.TestCase):
    def test_fingerprint_without_hash_keeps_quick_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            path.write_bytes(b"content")
            quick = fingerprint(path, include_hash=False)

        self.assertTrue(quick.exists)
        self.assertEqual(quick.size, 7)
        self.assertIsNotNone(quick.mtime_ns)
        self.assertIsNone(quick.sha256)

    def test_quick_match_ignores_hash_presence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            path.write_bytes(b"content")
            full = fingerprint(path, include_hash=True)
            quick = fingerprint(path, include_hash=False)

        self.assertIsNotNone(full.sha256)
        self.assertTrue(fingerprints_match_quick(full, quick))
        self.assertTrue(fingerprints_match_quick(quick, full))

    def test_full_match_requires_hash_when_baseline_has_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            path.write_bytes(b"content")
            full = fingerprint(path, include_hash=True)
            quick = fingerprint(path, include_hash=False)

        self.assertTrue(fingerprints_match(full, full))
        self.assertFalse(fingerprints_match(quick, full))

    def test_full_match_ignores_hash_when_baseline_has_no_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            path.write_bytes(b"content")
            full = fingerprint(path, include_hash=True)
            quick = fingerprint(path, include_hash=False)

        self.assertTrue(fingerprints_match(full, quick))
        self.assertTrue(fingerprints_match(quick, quick))

    def test_quick_match_detects_size_mtime_and_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            path.write_bytes(b"aaaa")
            baseline = fingerprint(path, include_hash=False)
            path.write_bytes(b"aabb")
            os.utime(path, ns=(baseline.mtime_ns, baseline.mtime_ns))
            changed_content = fingerprint(path, include_hash=False)
            path.write_bytes(b"bb")
            changed_size = fingerprint(path, include_hash=False)
            path.unlink()
            missing = fingerprint(path, include_hash=False)

        self.assertTrue(fingerprints_match_quick(changed_content, baseline))
        self.assertFalse(fingerprints_match_quick(changed_size, baseline))
        self.assertFalse(fingerprints_match_quick(missing, baseline))


class PipelineHashStrategyTests(unittest.TestCase):
    def _pipeline(
        self, root: Path, *, hash_videos: bool
    ) -> tuple[SubtitlePipeline, Path, Path, bytes]:
        video = root / "clip.mp4"
        video.write_bytes(b"video-abc")
        target = root / "clip.txt"
        target.write_bytes(b"")
        report = discover([root], hash_videos=hash_videos)
        candidate = report.candidates[0]
        options = PipelineOptions(
            roots=[root],
            runs_root=root / "runs",
            model_cache=root / "models",
            hash_videos=hash_videos,
        )
        run_dir = root / "runs" / "run1"
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": "run1",
            "status": "committed",
            "phases": {"all": [candidate.sample_id]},
            "options": options_to_dict(options),
        }
        pipeline = SubtitlePipeline(options, run_dir, [candidate], [], manifest)
        prepared = run_dir / "prepared" / f"{candidate.sample_id}.txt"
        prepared.parent.mkdir(parents=True)
        prepared.write_bytes(CAPTION_PAYLOAD)
        return pipeline, video, target, CAPTION_PAYLOAD

    def _stage_report(self, pipeline: SubtitlePipeline) -> None:
        payload = {
            "schema_version": "1.0",
            "run_id": pipeline.manifest["run_id"],
            "staged_all": True,
            "all_passed": True,
            "samples": [
                {
                    "sample_id": pipeline.candidates[0].sample_id,
                    "passed": True,
                    "caption_sha256": hashlib.sha256(CAPTION_PAYLOAD).hexdigest(),
                }
            ],
        }
        (pipeline.run_dir / "stage_report.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def test_validate_candidates_quick_checks_video_with_hash_baseline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            video = root / "clip.mp4"
            video.write_bytes(b"video-abc")
            target = root / "clip.txt"
            target.write_bytes(b"")
            report = discover([root], hash_videos=True)
            candidate = report.candidates[0]
            stat = video.stat()
            video.write_bytes(b"video-xyz")
            os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))

            with mock.patch(
                "cc_cover.discovery.sha256_file", wraps=sha256_file
            ) as hasher:
                validate_candidates(
                    [candidate],
                    require_initial_target=True,
                    phase=Phase.QUALITY_GATE,
                )

            hashed_paths = [
                Path(call.args[0]).resolve() for call in hasher.call_args_list
            ]
            self.assertNotIn(video.resolve(), hashed_paths)
            self.assertIn(target.resolve(), hashed_paths)

    def test_verify_quick_checks_video_with_hash_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, video, target, payload = self._pipeline(
                root, hash_videos=True
            )
            target.write_bytes(payload)
            stat = video.stat()
            video.write_bytes(b"video-xyz")
            os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))

            with mock.patch(
                "cc_cover.discovery.sha256_file", wraps=sha256_file
            ) as hasher:
                report = pipeline.verify()

            hasher.assert_not_called()
            self.assertTrue(report["passed"])

    def test_run_candidates_uses_quick_checks_before_and_after(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, video, _target, _payload = self._pipeline(
                root, hash_videos=True
            )
            sample_id = pipeline.candidates[0].sample_id
            stat = video.stat()
            video.write_bytes(b"video-xyz")
            os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))

            funasr_engine = mock.Mock()
            funasr_engine.transcribe.return_value = ([Segment(0, 1000, "你好")], {})
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = ([Segment(0, 1000, "你好")], {})
            with mock.patch(
                "cc_cover.pipeline.extract_audio", return_value=1.0
            ), mock.patch(
                "cc_cover.discovery.sha256_file", wraps=sha256_file
            ) as hasher:
                pipeline.run_candidates(
                    [sample_id],
                    {"funasr": funasr_engine, "faster_whisper": faster_engine},
                )

            hasher.assert_not_called()
            self.assertTrue(
                pipeline.engine_output("funasr", sample_id).is_file()
            )
            self.assertTrue(
                pipeline.engine_output("faster_whisper", sample_id).is_file()
            )

    def test_run_candidates_interleaves_both_engines_per_candidate(self) -> None:
        """一个候选紧接着跑完两个引擎，不是分两轮——用事件顺序验证。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, video, _target, _payload = self._pipeline(
                root, hash_videos=False
            )
            sample_id = pipeline.candidates[0].sample_id

            funasr_engine = mock.Mock()
            funasr_engine.transcribe.return_value = ([Segment(0, 1000, "你好")], {})
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = ([Segment(0, 1000, "你好")], {})
            output = StringIO()
            with mock.patch(
                "cc_cover.pipeline.extract_audio", return_value=1.0
            ), redirect_stdout(output):
                pipeline.run_candidates(
                    [sample_id],
                    {"funasr": funasr_engine, "faster_whisper": faster_engine},
                )

        lines = output.getvalue().splitlines()
        self.assertIn(f"[funasr 1/1] {video}", lines)
        self.assertIn(f"[faster_whisper 1/1] {video}", lines)

        events = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "event" in payload:
                events.append(payload)

        self.assertEqual(
            events,
            [
                {
                    "event": "progress",
                    "engine": "funasr",
                    "index": 1,
                    "total": 1,
                    "video_path": str(video),
                },
                {
                    "event": "progress",
                    "engine": "faster_whisper",
                    "index": 1,
                    "total": 1,
                    "video_path": str(video),
                },
            ],
        )

    def test_run_candidates_quick_check_aborts_before_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, video, _target, _payload = self._pipeline(
                root, hash_videos=True
            )
            sample_id = pipeline.candidates[0].sample_id
            stat = video.stat()
            video.write_bytes(b"video-xyz")
            os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

            funasr_engine = mock.Mock()
            faster_engine = mock.Mock()
            with mock.patch("cc_cover.pipeline.extract_audio") as extract:
                with self.assertRaises(PipelineError) as caught:
                    pipeline.run_candidates(
                        [sample_id],
                        {"funasr": funasr_engine, "faster_whisper": faster_engine},
                    )

            self.assertIn("转写前", str(caught.exception))
            self.assertEqual(caught.exception.phase, Phase.FINGERPRINT)
            extract.assert_not_called()
            funasr_engine.transcribe.assert_not_called()
            faster_engine.transcribe.assert_not_called()

    def test_commit_full_hash_aborts_on_hidden_video_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, video, target, _payload = self._pipeline(
                root, hash_videos=True
            )
            self._stage_report(pipeline)
            stat = video.stat()
            video.write_bytes(b"video-xyz")
            os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))

            with self.assertRaises(PipelineError) as caught:
                pipeline.commit()

            self.assertIn("写回前", str(caught.exception))
            self.assertEqual(target.read_bytes(), b"")

    def test_commit_with_hash_off_uses_quick_check_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, video, target, payload = self._pipeline(
                root, hash_videos=False
            )
            self._stage_report(pipeline)
            stat = video.stat()
            video.write_bytes(b"video-xyz")
            os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))

            report = pipeline.commit()

            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(report["entry_count"], 1)

    def test_commit_with_hash_on_succeeds_when_video_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pipeline, _video, target, payload = self._pipeline(
                root, hash_videos=True
            )
            self._stage_report(pipeline)

            report = pipeline.commit()

            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(report["entry_count"], 1)

    def test_discover_records_hash_baseline_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            video = root / "clip.mp4"
            video.write_bytes(b"video-abc")
            target = root / "clip.txt"
            target.write_bytes(b"")

            on_report = discover([root], hash_videos=True)
            off_report = discover([root], hash_videos=False)

        self.assertIsNotNone(on_report.candidates[0].video_fingerprint.sha256)
        self.assertIsNone(off_report.candidates[0].video_fingerprint.sha256)

    def test_full_run_hashes_each_video_exactly_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            video = root / "clip.mp4"
            video.write_bytes(b"video-abc")
            target = root / "clip.txt"
            target.write_bytes(b"")
            options = PipelineOptions(
                roots=[root],
                runs_root=root / "runs",
                model_cache=root / "models",
                hash_videos=True,
            )
            funasr_engine = mock.Mock()
            funasr_engine.transcribe.return_value = (list(EXECUTE_SEGMENTS), {})
            faster_engine = mock.Mock()
            faster_engine.transcribe.return_value = (list(EXECUTE_SEGMENTS), {})
            with mock.patch(
                "cc_cover.discovery.sha256_file", wraps=sha256_file
            ) as hasher, mock.patch(
                "cc_cover.pipeline.FunASREngine", return_value=funasr_engine
            ), mock.patch(
                "cc_cover.pipeline.FasterWhisperEngine", return_value=faster_engine
            ), mock.patch(
                "cc_cover.pipeline.extract_audio", return_value=3.0
            ):
                report = discover([root], hash_videos=True)
                candidate = report.candidates[0]
                run_dir = root / "runs" / "run1"
                run_dir.mkdir(parents=True)
                manifest = {
                    "run_id": "run1",
                    "status": "committed",
                    "phases": {
                        "all": [candidate.sample_id],
                        "pilot": [],
                        "remaining": [candidate.sample_id],
                    },
                    "options": options_to_dict(options),
                }
                pipeline = SubtitlePipeline(
                    options, run_dir, [candidate], [], manifest
                )
                pipeline.execute()

            video_hashes = [
                call
                for call in hasher.call_args_list
                if Path(call.args[0]).resolve() == video.resolve()
            ]
            self.assertEqual(len(video_hashes), 2)
            self.assertTrue(target.is_file())


if __name__ == "__main__":
    unittest.main()
