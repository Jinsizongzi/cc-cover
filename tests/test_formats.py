from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cc_cover.formats import detect_profile, render_segments, validate_rendered
from cc_cover.models import Segment


class FormatTests(unittest.TestCase):
    def test_detect_and_render_existing_timed_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sample = Path(temporary) / "sample.txt"
            sample.write_bytes(
                "00:00\r\n已有字幕\r\n\r\n00:03\r\n格式样本\r\n".encode("utf-8")
            )
            profile = detect_profile(sample)
            payload = render_segments(
                [
                    Segment(0, 1800, "你好，世界。"),
                    Segment(2500, 4200, "PyTorch 2.5。"),
                ],
                profile,
            )

        self.assertEqual(profile.style, "timed")
        self.assertEqual(profile.newline_name, "crlf")
        self.assertEqual(
            payload,
            "00:00\r\n你好世界\r\n\r\n00:02\r\nPyTorch 2.5\r\n".encode("utf-8"),
        )
        metrics = validate_rendered(payload, profile)
        self.assertEqual(metrics["segment_count"], 2)

    def test_detect_plain_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sample = Path(temporary) / "plain.txt"
            sample.write_text("第一段。\n\n第二段。\n", encoding="utf-8", newline="")
            profile = detect_profile(sample)

        self.assertEqual(profile.style, "plain")
        self.assertEqual(profile.newline_name, "lf")
        self.assertTrue(profile.terminal_newline)


if __name__ == "__main__":
    unittest.main()
