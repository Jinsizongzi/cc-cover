from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from cc_cover.discovery import VIDEO_EXTENSIONS, discover


class VideoExtensionTests(unittest.TestCase):
    def test_whitelist_covers_all_decision_extensions(self) -> None:
        self.assertEqual(
            VIDEO_EXTENSIONS,
            frozenset(
                {
                    ".mp4",
                    ".mkv",
                    ".avi",
                    ".mov",
                    ".wmv",
                    ".flv",
                    ".webm",
                    ".m4v",
                    ".ts",
                    ".m2ts",
                    ".mts",
                    ".ogv",
                    ".mpg",
                    ".mpeg",
                    ".3gp",
                    ".rmvb",
                    ".rm",
                    ".vob",
                    ".asf",
                    ".f4v",
                    ".divx",
                }
            ),
        )

    def test_every_whitelisted_extension_is_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, extension in enumerate(sorted(VIDEO_EXTENSIONS)):
                (root / f"clip{index}{extension}").write_bytes(b"video")
            report = discover([root], hash_videos=False)

        self.assertEqual(report.video_count, len(VIDEO_EXTENSIONS))
        self.assertEqual(len(report.candidates), len(VIDEO_EXTENSIONS))


class DiscoverySemanticsTests(unittest.TestCase):
    def test_all_videos_are_candidates_regardless_of_txt_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "missing.mp4").write_bytes(b"video")
            (root / "empty.mp4").write_bytes(b"video")
            (root / "empty.txt").write_bytes(b"")
            (root / "spaces.mp4").write_bytes(b"video")
            (root / "spaces.txt").write_bytes(b" \r\n\t")
            (root / "sample.mp4").write_bytes(b"video")
            (root / "sample.txt").write_text(
                "00:00\r\n格式\r\n\r\n00:02\r\n样本\r\n", encoding="utf-8"
            )
            (root / "notes.txt").write_text("不得修改", encoding="utf-8")

            report = discover([root], hash_videos=False)

        self.assertEqual(report.video_count, 4)
        self.assertEqual(len(report.candidates), 4)
        self.assertEqual(
            {item.target_path.name: item.initial_state for item in report.candidates},
            {
                "missing.txt": "missing",
                "empty.txt": "zero_byte",
                "spaces.txt": "whitespace_only",
                "sample.txt": "nonempty",
            },
        )
        self.assertEqual(len(report.protected_texts), 1)
        self.assertEqual(report.protected_texts[0].path.name, "notes.txt")

    def test_same_stem_videos_are_conflicts_excluded_and_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "episode.mp4").write_bytes(b"video")
            (root / "episode.mkv").write_bytes(b"video")
            (root / "episode.txt").write_bytes(b"")
            (root / "standalone.mp4").write_bytes(b"video")
            (root / "standalone.txt").write_bytes(b"")

            report = discover([root], hash_videos=False)

        self.assertEqual(report.video_count, 3)
        self.assertEqual(report.conflict_count, 1)
        self.assertEqual(len(report.candidates), 1)
        self.assertEqual(report.candidates[0].target_path.name, "standalone.txt")
        conflict = report.conflicts[0]
        self.assertEqual(conflict.target_path.name, "episode.txt")
        self.assertEqual(
            sorted(path.name for path in conflict.videos),
            ["episode.mkv", "episode.mp4"],
        )

    def test_different_name_txt_is_never_a_candidate_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "lesson.mp4").write_bytes(b"video")
            (root / "notes.txt").write_bytes(b"important")

            report = discover([root], hash_videos=False)

        self.assertEqual(len(report.candidates), 1)
        self.assertEqual(report.candidates[0].target_path.name, "lesson.txt")
        self.assertEqual(
            [item.path.name for item in report.protected_texts], ["notes.txt"]
        )

    def test_case_insensitive_target_collision_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "EPISODE.MP4").write_bytes(b"video")
            (root / "episode.mkv").write_bytes(b"video")

            report = discover([root], hash_videos=False)

        self.assertEqual(report.video_count, 2)
        self.assertEqual(report.conflict_count, 1)
        self.assertEqual(len(report.candidates), 0)
        conflict = report.conflicts[0]
        self.assertEqual(
            sorted(path.name for path in conflict.videos),
            ["EPISODE.MP4", "episode.mkv"],
        )

    def test_video_in_nested_subdirectory_is_scanned_with_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subdir = root / "season" / "episodes"
            subdir.mkdir(parents=True)
            (subdir / "clip.mp4").write_bytes(b"video")
            (subdir / "clip.txt").write_bytes(b"")

            report = discover([root], hash_videos=False)

        self.assertEqual(report.video_count, 1)
        self.assertEqual(len(report.candidates), 1)
        candidate = report.candidates[0]
        self.assertEqual(candidate.video_path, (subdir / "clip.mp4").resolve())
        self.assertEqual(candidate.root, root.resolve())


class DiscoveryProbeTests(unittest.TestCase):
    def _wav_media(self, root: Path, name: str) -> Path:
        path = root / name
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 16000)
        return path

    def test_discover_probes_duration_and_size_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = self._wav_media(root, "clip.mp4")
            expected_size = media.stat().st_size
            (root / "clip.txt").write_bytes(b"")
            (root / "broken.mp4").write_bytes(b"not a media file")
            (root / "broken.txt").write_bytes(b"")

            report = discover([root], hash_videos=False, probe_durations=True)

        by_name = {item.video_path.name: item for item in report.candidates}
        self.assertAlmostEqual(by_name["clip.mp4"].video_duration_s, 1.0, delta=0.15)
        self.assertEqual(by_name["clip.mp4"].video_fingerprint.size, expected_size)
        self.assertIsNone(by_name["broken.mp4"].video_duration_s)

    def test_discover_without_probe_keeps_duration_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._wav_media(root, "clip.mp4")
            (root / "clip.txt").write_bytes(b"")

            report = discover([root], hash_videos=False)

        self.assertEqual(len(report.candidates), 1)
        self.assertIsNone(report.candidates[0].video_duration_s)


if __name__ == "__main__":
    unittest.main()
