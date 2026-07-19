from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from cc_cover.cli import main


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

    def test_config_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            (videos / "lesson.mp4").write_bytes(b"video")
            (videos / "lesson.txt").write_bytes(b"")
            config = root / "config.json"
            config.write_text(
                json.dumps({"roots": ["videos"], "hash_videos": False}),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                result = main(["scan", "--config", str(config), "--json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
