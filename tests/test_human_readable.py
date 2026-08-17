from __future__ import annotations

import unittest

from cc_cover.human_readable import format_duration, format_size


class FormatSizeTests(unittest.TestCase):
    def test_format_size_human_readable(self) -> None:
        self.assertEqual(format_size(0), "0 B")
        self.assertEqual(format_size(512), "512 B")
        self.assertEqual(format_size(1024), "1.0 KB")
        self.assertEqual(format_size(1536), "1.5 KB")
        self.assertEqual(format_size(5 * 1024 * 1024), "5.0 MB")
        self.assertEqual(format_size(3 * 1024**3), "3.0 GB")


class FormatDurationTests(unittest.TestCase):
    def test_plain_seconds(self) -> None:
        self.assertEqual(format_duration(0), "0 秒")
        self.assertEqual(format_duration(45), "45 秒")

    def test_minutes_and_seconds(self) -> None:
        self.assertEqual(format_duration(120), "2 分 0 秒")
        self.assertEqual(format_duration(125), "2 分 5 秒")

    def test_hours_minutes_seconds(self) -> None:
        self.assertEqual(format_duration(3725), "1 时 2 分 5 秒")

    def test_unknown_duration(self) -> None:
        self.assertEqual(format_duration(None), "未知")


if __name__ == "__main__":
    unittest.main()
