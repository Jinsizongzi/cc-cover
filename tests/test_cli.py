from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from cc_cover.cli import create_parser, main


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


if __name__ == "__main__":
    unittest.main()
