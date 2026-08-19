from __future__ import annotations

import unittest

from cc_cover.core.formats import (
    FormatError,
    decode_bytes,
    normalize_text,
    parse_timestamp,
    render_segments,
    text_is_whitespace_only,
    timestamp,
    validate_rendered,
)
from cc_cover.core.models import Segment


class TimestampTests(unittest.TestCase):
    def test_under_one_hour_uses_mmss(self) -> None:
        self.assertEqual(timestamp(0), "00:00")
        self.assertEqual(timestamp((59 * 60 + 59) * 1000), "59:59")

    def test_one_hour_switches_to_hmmss(self) -> None:
        self.assertEqual(timestamp(60 * 60 * 1000), "1:00:00")

    def test_hours_are_not_digit_limited(self) -> None:
        self.assertEqual(timestamp((100 * 60 + 30) * 1000), "1:40:30")
        self.assertEqual(timestamp(25 * 60 * 60 * 1000), "25:00:00")


class RenderSegmentsTests(unittest.TestCase):
    def test_fixed_output_uses_mmss_crlf_no_bom_and_trailing_newline(self) -> None:
        payload = render_segments(
            [
                Segment(0, 1800, "你好，世界。"),
                Segment(2500, 4200, "PyTorch 2.5。"),
            ]
        )

        self.assertEqual(
            payload,
            "00:00\r\n你好世界\r\n\r\n00:02\r\nPyTorch 2.5\r\n".encode("utf-8"),
        )
        self.assertNotEqual(payload[:3], b"\xef\xbb\xbf")

    def test_over_one_hour_renders_hmmss_without_validation_hang(self) -> None:
        payload = render_segments(
            [Segment(100 * 60 * 1000, 100 * 60 * 1000 + 1500, "长视频内容")]
        )

        self.assertIn(b"1:40:00\r\n", payload)
        metrics = validate_rendered(payload)
        self.assertEqual(metrics["segment_count"], 1)
        self.assertEqual(metrics["first_timestamp"], "1:40:00")

    def test_segments_are_sorted_and_blank_text_skipped(self) -> None:
        payload = render_segments(
            [
                Segment(5000, 6000, "   "),
                Segment(0, 1000, "第一条"),
                Segment(3000, 4000, "第二条"),
            ]
        )

        self.assertEqual(
            payload,
            "00:00\r\n第一条\r\n\r\n00:03\r\n第二条\r\n".encode("utf-8"),
        )

    def test_all_empty_text_raises_format_error(self) -> None:
        with self.assertRaises(FormatError):
            render_segments([Segment(0, 1000, "   ")])


class ValidateRenderedTests(unittest.TestCase):
    def test_accepts_fixed_timed_payload_with_metrics(self) -> None:
        payload = "00:00\r\n你好世界\r\n\r\n00:02\r\nPyTorch 2.5\r\n".encode("utf-8")

        metrics = validate_rendered(payload)

        self.assertEqual(metrics["style"], "timed")
        self.assertEqual(metrics["segment_count"], 2)
        self.assertEqual(metrics["first_timestamp"], "00:00")
        self.assertEqual(metrics["last_timestamp"], "00:02")

    def test_accepts_hmmss_with_unlimited_hours(self) -> None:
        payload = "1:40:00\r\n长视频内容\r\n\r\n25:00:00\r\n跨天内容\r\n".encode(
            "utf-8"
        )

        metrics = validate_rendered(payload)

        self.assertEqual(metrics["segment_count"], 2)
        self.assertEqual(metrics["first_timestamp"], "1:40:00")
        self.assertEqual(metrics["last_timestamp"], "25:00:00")

    def test_accepts_wide_minute_field_without_hanging(self) -> None:
        payload = "100:00\r\n内容\r\n".encode("utf-8")

        metrics = validate_rendered(payload)

        self.assertEqual(metrics["segment_count"], 1)

    def test_rejects_utf16_and_utf8_bom(self) -> None:
        payloads = (
            "00:00\r\n内容\r\n".encode("utf-16"),
            b"\xef\xbb\xbf" + "00:00\r\n内容\r\n".encode("utf-8"),
        )
        for payload in payloads:
            with self.subTest(payload=payload[:8]):
                with self.assertRaises(FormatError):
                    validate_rendered(payload)

    def test_rejects_non_crlf_newline(self) -> None:
        with self.assertRaises(FormatError):
            validate_rendered("00:00\n内容\n".encode("utf-8"))

    def test_rejects_missing_trailing_newline(self) -> None:
        payload = "00:00\r\n内容\r\n".encode("utf-8")[:-2]
        with self.assertRaises(FormatError):
            validate_rendered(payload)

    def test_rejects_missing_blank_line_between_blocks(self) -> None:
        payload = "00:00\r\n内容\r\n00:02\r\n更多\r\n".encode("utf-8")
        with self.assertRaises(FormatError):
            validate_rendered(payload)

    def test_rejects_non_monotonic_timestamps(self) -> None:
        payload = "00:03\r\n内容\r\n\r\n00:00\r\n更早\r\n".encode("utf-8")
        with self.assertRaises(FormatError):
            validate_rendered(payload)


class ParseTimestampTests(unittest.TestCase):
    def test_parses_mmss(self) -> None:
        self.assertEqual(parse_timestamp("00:00"), 0)
        self.assertEqual(parse_timestamp("59:59"), 59 * 60 + 59)

    def test_parses_hmmss_with_unlimited_hours(self) -> None:
        self.assertEqual(parse_timestamp("1:00:00"), 3600)
        self.assertEqual(parse_timestamp("25:00:00"), 25 * 3600)

    def test_parses_wide_minute_and_single_hour_fields(self) -> None:
        self.assertEqual(parse_timestamp("100:00"), 100 * 60)
        self.assertEqual(parse_timestamp("0:00"), 0)

    def test_rejects_malformed_timecodes(self) -> None:
        for timecode in ("0:0", "00:0", "1:2:3", "1::00", "abc", "", "12:34:56:78"):
            with self.subTest(timecode=timecode):
                self.assertIsNone(parse_timestamp(timecode))


class DecodeBytesTests(unittest.TestCase):
    def test_plain_utf8_is_not_bom(self) -> None:
        text, encoding, bom = decode_bytes("你好".encode("utf-8"))

        self.assertEqual((text, encoding, bom), ("你好", "utf-8", False))

    def test_utf8_bom_is_stripped_and_flagged(self) -> None:
        text, encoding, bom = decode_bytes(b"\xef\xbb\xbf" + "你好".encode("utf-8"))

        self.assertEqual((text, encoding, bom), ("你好", "utf-8", True))

    def test_utf16_le_is_decoded(self) -> None:
        text, encoding, bom = decode_bytes("你好".encode("utf-16"))

        self.assertEqual(text, "你好")
        self.assertTrue(bom)

    def test_gb18030_fallback(self) -> None:
        text, encoding, bom = decode_bytes(b"\xc4\xe3\xba\xc3")

        self.assertEqual((text, encoding, bom), ("你好", "gb18030", False))

    def test_unrecognizable_payload_raises(self) -> None:
        with self.assertRaises(FormatError):
            decode_bytes(b"\x81")


class NormalizeTextTests(unittest.TestCase):
    def test_strips_chinese_punctuation(self) -> None:
        self.assertEqual(normalize_text("你好，世界。"), "你好世界")

    def test_keeps_dot_between_ascii_words(self) -> None:
        self.assertEqual(normalize_text("A.B"), "A.B")

    def test_keeps_colon_between_digits(self) -> None:
        self.assertEqual(normalize_text("3:45"), "3:45")

    def test_removes_model_tags(self) -> None:
        self.assertEqual(normalize_text("你好<|en|>世界"), "你好世界")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("  PyTorch\t 2.5 \n"), "PyTorch 2.5")


class TextIsWhitespaceOnlyTests(unittest.TestCase):
    def test_empty_payload_is_whitespace(self) -> None:
        self.assertTrue(text_is_whitespace_only(b""))

    def test_whitespace_payload_is_whitespace(self) -> None:
        self.assertTrue(text_is_whitespace_only(b" \r\n\t"))

    def test_content_payload_is_not_whitespace(self) -> None:
        self.assertFalse(text_is_whitespace_only("内容".encode("utf-8")))

    def test_unrecognizable_payload_is_not_whitespace(self) -> None:
        self.assertFalse(text_is_whitespace_only(b"\x81"))


if __name__ == "__main__":
    unittest.main()
