from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest import mock

from cc_cover.core.discovery import discover
from cc_cover.core.models import PipelineOptions
from cc_cover.core.pipeline import (
    PipelineError,
    SubtitlePipeline,
    options_to_dict,
    write_bytes_atomic,
    write_json_atomic,
)


CAPTION = "00:00\r\n你好世界\r\n\r\n00:02\r\nPyTorch 2.5\r\n".encode("utf-8")


def _fail_write_on(target: Path) -> Callable[[Path, bytes], None]:
    """返回一个 write_bytes_atomic 替身：写到指定目标时抛错，其余走真实实现。"""
    real_write = write_bytes_atomic

    def fail_write(path: Path, payload: bytes) -> None:
        if Path(path).resolve() == target.resolve():
            raise OSError("模拟写回失败")
        real_write(path, payload)

    return fail_write


def _make_pipeline(
    root: Path, *, videos: list[tuple[str, bytes]], originals: dict[str, bytes]
) -> tuple[SubtitlePipeline, list[Path], list[Path]]:
    """两个候选：每个视频配一个同名 TXT，初始内容取自 originals（缺失则不存在）。"""
    root = root.resolve()
    target_paths: list[Path] = []
    for name, _content in videos:
        video = root / f"{name}.mp4"
        video.write_bytes(_content)
        target = root / f"{name}.txt"
        original = originals.get(name)
        if original is not None:
            target.write_bytes(original)
        target_paths.append(target)
    report = discover([root], hash_videos=False)
    candidates = list(report.candidates)
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
        "status": "committed",
        "phases": {
            "all": [candidate.sample_id for candidate in candidates],
            "pilot": [],
            "remaining": [candidate.sample_id for candidate in candidates],
        },
        "options": options_to_dict(options),
    }
    pipeline = SubtitlePipeline(options, run_dir, candidates, [], manifest)
    payloads: list[bytes] = []
    for candidate in candidates:
        prepared = run_dir / "prepared" / f"{candidate.sample_id}.txt"
        prepared.parent.mkdir(parents=True, exist_ok=True)
        prepared.write_bytes(CAPTION)
        payloads.append(CAPTION)
    write_json_atomic(
        run_dir / "stage_report.json",
        {
            "schema_version": "1.0",
            "run_id": "run1",
            "staged_all": True,
            "all_passed": True,
            "samples": [
                {
                    "sample_id": candidate.sample_id,
                    "passed": True,
                    "caption_sha256": hashlib.sha256(payload).hexdigest(),
                }
                for candidate, payload in zip(candidates, payloads)
            ],
        },
    )
    return pipeline, target_paths, candidates


class CommitRollbackTests(unittest.TestCase):
    def test_rolls_back_earlier_targets_when_later_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline, targets, _candidates = _make_pipeline(
                root,
                videos=[("a", b"video-a"), ("b", b"video-b")],
                originals={"a": b"original-a", "b": b"original-b"},
            )
            target_a, target_b = targets
            with mock.patch(
                "cc_cover.core.pipeline.write_bytes_atomic",
                side_effect=_fail_write_on(target_b),
            ):
                with self.assertRaises(OSError) as caught:
                    pipeline.commit()

            self.assertIn("模拟写回失败", str(caught.exception))
            self.assertEqual(target_a.read_bytes(), b"original-a")
            self.assertEqual(target_b.read_bytes(), b"original-b")

    def test_rollback_removes_created_target_for_missing_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline, targets, _candidates = _make_pipeline(
                root,
                videos=[("a", b"video-a"), ("b", b"video-b")],
                originals={"b": b"original-b"},
            )
            target_a, target_b = targets
            self.assertFalse(target_a.exists())
            with mock.patch(
                "cc_cover.core.pipeline.write_bytes_atomic",
                side_effect=_fail_write_on(target_b),
            ):
                with self.assertRaises(OSError):
                    pipeline.commit()

            self.assertFalse(target_a.exists())
            self.assertEqual(target_b.read_bytes(), b"original-b")

    def test_commit_writes_backup_of_original_before_replacing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline, targets, candidates = _make_pipeline(
                root,
                videos=[("a", b"video-a")],
                originals={"a": b"original-a"},
            )
            target_a = targets[0]

            report = pipeline.commit()

            self.assertEqual(report["entry_count"], 1)
            self.assertEqual(target_a.read_bytes(), CAPTION)
            backup_dir = pipeline.run_dir / "backups" / candidates[0].sample_id
            self.assertEqual((backup_dir / "original.txt").read_bytes(), b"original-a")
            state = json.loads((backup_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["initial_state"], "nonempty")
            self.assertEqual(state["target_path"], str(target_a.resolve()))

    def test_commit_rolls_back_when_second_write_fails_with_missing_and_existing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline, targets, _candidates = _make_pipeline(
                root,
                videos=[("a", b"video-a"), ("b", b"video-b"), ("c", b"video-c")],
                originals={"a": b"original-a", "c": b"original-c"},
            )
            target_a, target_b, target_c = targets
            with mock.patch(
                "cc_cover.core.pipeline.write_bytes_atomic",
                side_effect=_fail_write_on(target_c),
            ):
                with self.assertRaises(OSError):
                    pipeline.commit()

            self.assertEqual(target_a.read_bytes(), b"original-a")
            self.assertFalse(target_b.exists())
            self.assertEqual(target_c.read_bytes(), b"original-c")


class VerifySafetyTests(unittest.TestCase):
    def test_verify_reports_modified_target_after_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline, targets, _candidates = _make_pipeline(
                root,
                videos=[("a", b"video-a")],
                originals={"a": b"original-a"},
            )
            target_a = targets[0]
            pipeline.commit()
            target_a.write_bytes(b"tampered-by-other-process")

            with self.assertRaises(PipelineError) as caught:
                pipeline.verify()

        self.assertIn("不一致", str(caught.exception))
        self.assertIn(str(target_a.resolve()), str(caught.exception))

    def test_verify_passes_when_target_matches_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline, _targets, _candidates = _make_pipeline(
                root,
                videos=[("a", b"video-a")],
                originals={"a": b"original-a"},
            )
            pipeline.commit()

            report = pipeline.verify()

        self.assertTrue(report["passed"])
        self.assertEqual(report["verified_count"], 1)


if __name__ == "__main__":
    unittest.main()
