from __future__ import annotations

import json
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from cc_cover.cli import ConfigError, create_parser, load_exclusions, main
from cc_cover.models import Phase
from cc_cover.pipeline import PipelineError


class CliTests(unittest.TestCase):
    def test_scan_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "lesson.mp4").write_bytes(b"video")
            (root / "lesson.txt").write_bytes(b"")
            output = StringIO()
            with redirect_stdout(output):
                result = main(["scan", str(root), "--json", "--no-hash-videos"])

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["candidates"][0]["state"], "zero_byte")

    def test_scan_json_includes_duration_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = root / "lesson.mp4"
            with wave.open(str(media), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)
            expected_size = media.stat().st_size
            (root / "lesson.txt").write_bytes(b"")
            output = StringIO()
            with redirect_stdout(output):
                result = main(["scan", str(root), "--json", "--no-hash-videos"])

        payload = json.loads(output.getvalue())
        candidate = payload["candidates"][0]
        self.assertEqual(result, 0)
        self.assertAlmostEqual(candidate["video_duration_s"], 1.0, delta=0.15)
        self.assertEqual(candidate["video_size"], expected_size)

    def test_config_settings_require_explicit_scan_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            (videos / "lesson.mp4").write_bytes(b"video")
            (videos / "lesson.txt").write_bytes(b"")
            config = root / "config.json"
            config.write_text(
                json.dumps({"runs_root": "runs", "hash_videos": False}),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                result = main(
                    ["scan", str(videos), "--config", str(config), "--json"]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["candidate_count"], 1)

    def test_scan_json_lists_conflicts_and_excludes_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "episode.mp4").write_bytes(b"video")
            (root / "episode.mkv").write_bytes(b"video")
            (root / "episode.txt").write_bytes(b"")
            (root / "standalone.mp4").write_bytes(b"video")
            (root / "standalone.txt").write_bytes(b"")
            output = StringIO()
            with redirect_stdout(output):
                result = main(["scan", str(root), "--json", "--no-hash-videos"])

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["video_count"], 3)
        self.assertEqual(payload["conflict_count"], 1)
        self.assertEqual(payload["candidate_count"], 1)
        self.assertTrue(
            payload["candidates"][0]["target_path"].endswith("standalone.txt")
        )
        self.assertTrue(
            payload["conflicts"][0]["target_path"].endswith("episode.txt")
        )
        self.assertEqual(len(payload["conflicts"][0]["videos"]), 2)

    def test_obsolete_scan_options_are_rejected(self) -> None:
        parser = create_parser()
        errors = StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            with redirect_stderr(errors), self.assertRaises(SystemExit):
                parser.parse_args(["scan", temporary, "--include-missing"])
            with redirect_stderr(errors), self.assertRaises(SystemExit):
                parser.parse_args(["scan", temporary, "--include-whitespace-only"])

    def test_transcribe_requires_root_and_rejects_obsolete_write_flag(self) -> None:
        parser = create_parser()
        errors = StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit):
            parser.parse_args(["transcribe"])
        with tempfile.TemporaryDirectory() as temporary:
            obsolete_flag = "--" + "app" + "ly"
            with redirect_stderr(errors), self.assertRaises(SystemExit):
                parser.parse_args(["transcribe", temporary, obsolete_flag])

    def test_cli_device_choices_are_preserved(self) -> None:
        parser = create_parser()
        for choice in ("auto", "cuda", "cpu"):
            arguments = parser.parse_args(
                ["transcribe", "root", "--device", choice]
            )
            self.assertEqual(arguments.device, choice)

    def test_load_exclusions_accepts_valid_video_path_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            excluded_file = root / "excluded.json"
            first = root / "a.mp4"
            second = root / "b.mp4"
            excluded_file.write_text(
                json.dumps([str(first), str(second)]), encoding="utf-8"
            )

            excluded = load_exclusions(excluded_file)

        self.assertEqual(
            excluded,
            {first.resolve(), second.resolve()},
        )

    def test_load_exclusions_rejects_invalid_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.json"
            with self.assertRaises(ConfigError):
                load_exclusions(missing)

            not_a_list = root / "not-a-list.json"
            not_a_list.write_text(json.dumps({"video": "a.mp4"}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_exclusions(not_a_list)

            not_strings = root / "not-strings.json"
            not_strings.write_text(json.dumps([1, 2]), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_exclusions(not_strings)

        self.assertEqual(load_exclusions(None), set())

    def test_transcribe_accepts_exclude_file_argument(self) -> None:
        parser = create_parser()
        with tempfile.TemporaryDirectory() as temporary:
            exclude_file = Path(temporary) / "excluded.json"
            arguments = parser.parse_args(
                ["transcribe", temporary, "--exclude", str(exclude_file)]
            )

        self.assertEqual(arguments.exclude, exclude_file)

    def _write_candidates(self, root: Path, names: list[str]) -> None:
        for name in names:
            (root / f"{name}.mp4").write_bytes(f"video-{name}".encode())
            (root / f"{name}.txt").write_bytes(b"")

    def test_transcribe_filters_excluded_candidates_and_reports_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_candidates(root, ["keep", "skip"])
            exclude_file = root / "excluded.json"
            exclude_file.write_text(
                json.dumps([str((root / "skip.mp4").resolve())]), encoding="utf-8"
            )
            runs_root = root / "runs"
            model_cache = root / "models"
            output = StringIO()
            pipeline = mock.Mock()
            pipeline.run_dir = root / "runs" / "run-1"
            pipeline.candidate_failures = {}
            with mock.patch(
                "cc_cover.cli.SubtitlePipeline.create", return_value=pipeline
            ) as create:
                with redirect_stdout(output):
                    result = main(
                        [
                            "transcribe",
                            str(root),
                            "--no-hash-videos",
                            "--exclude",
                            str(exclude_file),
                            "--runs-root",
                            str(runs_root),
                            "--model-cache",
                            str(model_cache),
                        ]
                    )

        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("已排除：1 个候选", text)
        self.assertIn("实际处理：1 个候选", text)
        create.assert_called_once()
        filtered_report = create.call_args.args[1]
        self.assertEqual(len(filtered_report.candidates), 1)
        self.assertEqual(
            filtered_report.candidates[0].video_path.name,
            "keep.mp4",
        )
        self.assertEqual(
            [path.name for path in create.call_args.kwargs["excluded_videos"]],
            ["skip.mp4"],
        )
        pipeline.execute.assert_called_once_with()

    def test_transcribe_with_all_excluded_does_not_start_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_candidates(root, ["only"])
            exclude_file = root / "excluded.json"
            exclude_file.write_text(
                json.dumps([str((root / "only.mp4").resolve())]), encoding="utf-8"
            )
            output = StringIO()
            with mock.patch("cc_cover.cli.SubtitlePipeline.create") as create:
                with redirect_stdout(output):
                    result = main(
                        [
                            "transcribe",
                            str(root),
                            "--no-hash-videos",
                            "--exclude",
                            str(exclude_file),
                            "--runs-root",
                            str(root / "runs"),
                            "--model-cache",
                            str(root / "models"),
                        ]
                    )

        self.assertEqual(result, 0)
        self.assertIn("无需处理", output.getvalue())
        create.assert_not_called()

    def test_transcribe_emits_run_dir_and_done_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_candidates(root, ["only"])
            output = StringIO()
            pipeline = mock.Mock()
            pipeline.run_dir = root / "runs" / "run-1"
            pipeline.candidate_failures = {}
            with mock.patch(
                "cc_cover.cli.SubtitlePipeline.create", return_value=pipeline
            ):
                with redirect_stdout(output):
                    result = main(
                        [
                            "transcribe",
                            str(root),
                            "--no-hash-videos",
                            "--runs-root",
                            str(root / "runs"),
                            "--model-cache",
                            str(root / "models"),
                        ]
                    )

        self.assertEqual(result, 0)
        lines = output.getvalue().splitlines()
        self.assertIn(f"运行目录：{pipeline.run_dir}", lines)
        self.assertIn(f"字幕已写回并复核通过：{pipeline.run_dir}", lines)
        pipeline.execute.assert_called_once_with()
        events = [json.loads(line) for line in lines if line.startswith("{")]
        self.assertEqual(
            events,
            [
                {"event": "run_dir", "path": str(pipeline.run_dir)},
                {"event": "done", "run_dir": str(pipeline.run_dir)},
            ],
        )

    def test_transcribe_returns_nonzero_when_candidates_were_skipped(self) -> None:
        """候选级失败不再抛异常，但整体仍算失败——退出码要跟以前一样非零。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_candidates(root, ["only"])
            output = StringIO()
            errors = StringIO()
            pipeline = mock.Mock()
            pipeline.run_dir = root / "runs" / "run-1"
            pipeline.candidate_failures = {
                "CC-1": {"video_path": "bad.mp4", "reason": "音频提取失败"}
            }
            with mock.patch(
                "cc_cover.cli.SubtitlePipeline.create", return_value=pipeline
            ):
                with redirect_stdout(output), redirect_stderr(errors):
                    result = main(
                        [
                            "transcribe",
                            str(root),
                            "--no-hash-videos",
                            "--runs-root",
                            str(root / "runs"),
                            "--model-cache",
                            str(root / "models"),
                        ]
                    )

        self.assertEqual(result, 1)
        self.assertIn(f"字幕已写回并复核通过：{pipeline.run_dir}", output.getvalue())
        self.assertIn("1 个候选处理失败", errors.getvalue())
        self.assertIn("bad.mp4：音频提取失败", errors.getvalue())
        lines = output.getvalue().splitlines()
        events = [json.loads(line) for line in lines if line.startswith("{")]
        self.assertIn({"event": "done", "run_dir": str(pipeline.run_dir)}, events)

    def test_resume_already_committed_emits_done_event_without_executing(
        self,
    ) -> None:
        output = StringIO()
        pipeline = mock.Mock()
        pipeline.run_dir = Path("runs/run-1")
        pipeline.manifest = {"status": "committed"}
        pipeline.verify.return_value = {"verified_count": 3}
        pipeline.candidate_failures = {}
        with mock.patch(
            "cc_cover.cli.SubtitlePipeline.resume", return_value=pipeline
        ):
            with redirect_stdout(output):
                result = main(["resume", str(pipeline.run_dir)])

        self.assertEqual(result, 0)
        lines = output.getvalue().splitlines()
        self.assertIn("复核通过，共 3 个字幕文件。", lines)
        pipeline.execute.assert_not_called()
        events = [json.loads(line) for line in lines if line.startswith("{")]
        self.assertEqual(
            events, [{"event": "done", "run_dir": str(pipeline.run_dir)}]
        )

    def test_resume_pending_executes_and_emits_done_event(self) -> None:
        output = StringIO()
        pipeline = mock.Mock()
        pipeline.run_dir = Path("runs/run-1")
        pipeline.manifest = {"status": "staged_partial"}
        pipeline.candidate_failures = {}
        with mock.patch(
            "cc_cover.cli.SubtitlePipeline.resume", return_value=pipeline
        ):
            with redirect_stdout(output):
                result = main(["resume", str(pipeline.run_dir)])

        self.assertEqual(result, 0)
        lines = output.getvalue().splitlines()
        self.assertIn(f"字幕已写回并复核通过：{pipeline.run_dir}", lines)
        pipeline.execute.assert_called_once_with()
        events = [json.loads(line) for line in lines if line.startswith("{")]
        self.assertEqual(
            events, [{"event": "done", "run_dir": str(pipeline.run_dir)}]
        )

    def test_verify_emits_done_event(self) -> None:
        output = StringIO()
        pipeline = mock.Mock()
        pipeline.run_dir = Path("runs/run-1")
        pipeline.verify.return_value = {"verified_count": 5}
        pipeline.candidate_failures = {}
        with mock.patch(
            "cc_cover.cli.SubtitlePipeline.resume", return_value=pipeline
        ):
            with redirect_stdout(output):
                result = main(["verify", str(pipeline.run_dir)])

        self.assertEqual(result, 0)
        lines = output.getvalue().splitlines()
        self.assertIn("复核通过，共 5 个字幕文件。", lines)
        events = [json.loads(line) for line in lines if line.startswith("{")]
        self.assertEqual(
            events, [{"event": "done", "run_dir": str(pipeline.run_dir)}]
        )

    def test_pipeline_error_emits_error_event_with_phase_and_candidate_context(
        self,
    ) -> None:
        output = StringIO()
        errors = StringIO()
        pipeline = mock.Mock()
        pipeline.run_dir = Path("runs/run-1")
        pipeline.execute.side_effect = PipelineError(
            "写回后内容不一致：a.txt",
            phase=Phase.WRITEBACK,
            sample_id="CC-MISSING-00047",
            video_path="E:/videos/a.mp4",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_candidates(root, ["only"])
            with mock.patch(
                "cc_cover.cli.SubtitlePipeline.create", return_value=pipeline
            ):
                with redirect_stdout(output), redirect_stderr(errors):
                    result = main(
                        [
                            "transcribe",
                            str(root),
                            "--no-hash-videos",
                            "--runs-root",
                            str(root / "runs"),
                            "--model-cache",
                            str(root / "models"),
                        ]
                    )

        self.assertEqual(result, 1)
        self.assertIn("错误：写回后内容不一致：a.txt", errors.getvalue())
        events = [
            json.loads(line)
            for line in output.getvalue().splitlines()
            if line.startswith("{")
        ]
        error_events = [event for event in events if event["event"] == "error"]
        self.assertEqual(
            error_events,
            [
                {
                    "event": "error",
                    "phase": "writeback",
                    "reason": "写回后内容不一致：a.txt",
                    "video_path": "E:/videos/a.mp4",
                    "sample_id": "CC-MISSING-00047",
                }
            ],
        )

    def test_config_error_does_not_emit_error_event(self) -> None:
        output = StringIO()
        errors = StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_candidates(root, ["only"])
            missing_exclude = root / "missing.json"
            with redirect_stdout(output), redirect_stderr(errors):
                result = main(
                    [
                        "transcribe",
                        str(root),
                        "--no-hash-videos",
                        "--exclude",
                        str(missing_exclude),
                    ]
                )

        self.assertEqual(result, 1)
        self.assertIn("错误：", errors.getvalue())
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
