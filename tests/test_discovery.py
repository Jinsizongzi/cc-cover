from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cc_cover.discovery import discover


class DiscoveryTests(unittest.TestCase):
    def test_default_only_selects_zero_byte_txt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "empty.mp4").write_bytes(b"video")
            (root / "empty.txt").write_bytes(b"")
            (root / "sample.mp4").write_bytes(b"video")
            (root / "sample.txt").write_bytes(
                "00:00\r\n格式\r\n\r\n00:02\r\n样本\r\n".encode("utf-8")
            )
            (root / "missing.mkv").write_bytes(b"video")
            (root / "notes.txt").write_text("不得修改", encoding="utf-8")

            report = discover([root], hash_videos=False)

        self.assertEqual(report.video_count, 3)
        self.assertEqual(report.missing_text_count, 1)
        self.assertEqual(len(report.candidates), 1)
        self.assertEqual(report.candidates[0].target_path.name, "empty.txt")
        self.assertEqual(report.candidates[0].initial_state, "zero_byte")
        self.assertEqual(report.candidates[0].profile.style, "timed")
        self.assertEqual(len(report.protected_texts), 2)

    def test_missing_and_whitespace_are_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "missing.mp4").write_bytes(b"video")
            (root / "spaces.mp4").write_bytes(b"video")
            (root / "spaces.txt").write_bytes(b" \r\n\t")

            default_report = discover([root], hash_videos=False)
            opt_in_report = discover(
                [root],
                include_missing=True,
                include_whitespace_only=True,
                hash_videos=False,
            )

        self.assertEqual(len(default_report.candidates), 0)
        self.assertEqual(
            {item.initial_state for item in opt_in_report.candidates},
            {"missing", "whitespace_only"},
        )


if __name__ == "__main__":
    unittest.main()
