from __future__ import annotations

import unittest

from cc_cover.human_readable import format_duration, format_size, strip_ansi_escapes


class StripAnsiEscapesTests(unittest.TestCase):
    def test_strips_cursor_movement(self) -> None:
        self.assertEqual(strip_ansi_escapes("100%|done\x1b[A"), "100%|done")

    def test_strips_color_codes(self) -> None:
        self.assertEqual(
            strip_ansi_escapes("\x1b[34m██████\x1b[0m"), "██████"
        )

    def test_strips_realistic_tqdm_frame(self) -> None:
        raw = "100%|\x1b[34m██████████\x1b[0m| 5/5 [00:00<00:00, 8.82it/s]\x1b[A"
        self.assertEqual(
            strip_ansi_escapes(raw),
            "100%|██████████| 5/5 [00:00<00:00, 8.82it/s]",
        )

    def test_plain_text_unaffected(self) -> None:
        self.assertEqual(strip_ansi_escapes("[funasr 31/77] a.mp4"), "[funasr 31/77] a.mp4")

    def test_does_not_touch_brackets_without_escape_byte(self) -> None:
        # 没有真正的 ESC (\x1b) 字节时，"[A"这种纯文本不应该被误删。
        self.assertEqual(strip_ansi_escapes("状态：[A]"), "状态：[A]")


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
