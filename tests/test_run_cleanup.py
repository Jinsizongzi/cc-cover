from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cc_cover.gui_support import (
    CLEANUP_WARNING_BYTES,
    RunInfo,
    delete_runs,
    directory_size,
    format_size,
    list_runs,
    runs_total_size,
)


def manifest_status(run_dir: Path, status: str, created: str | None = None) -> None:
    payload: dict[str, object] = {"status": status}
    if created is not None:
        payload["created_at_utc"] = created
    (run_dir / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class ListRunsTests(unittest.TestCase):
    def test_lists_runs_sorted_with_status_size_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runs_root = root / "runs"
            committed = runs_root / "20260801_010101_1"
            committed.mkdir(parents=True)
            manifest_status(
                committed,
                "committed",
                created="2026-08-01T01:01:01+00:00",
            )
            prepared_dir = committed / "prepared" / "sample.txt"
            prepared_dir.mkdir(parents=True)
            (prepared_dir / "x.txt").write_bytes(b"x" * 10)

            staged = runs_root / "20260802_020202_2"
            staged.mkdir(parents=True)
            manifest_status(staged, "staged_partial")

            abandoned = runs_root / "abandoned-run-3"
            abandoned.mkdir()
            (abandoned / "junk.bin").write_bytes(b"y" * 5)

            runs = list_runs(runs_root)

        self.assertEqual(
            [run.run_id for run in runs],
            ["20260801_010101_1", "20260802_020202_2", "abandoned-run-3"],
        )
        self.assertEqual(runs[0].status, "committed")
        self.assertEqual(runs[0].created_at_utc, "2026-08-01T01:01:01+00:00")
        self.assertGreaterEqual(runs[0].size_bytes, 10)
        self.assertEqual(runs[1].status, "staged_partial")
        self.assertEqual(runs[2].status, "unknown")
        self.assertEqual(runs[2].size_bytes, 5)
        self.assertEqual(runs[2].path, (root / "runs" / "abandoned-run-3").resolve())

    def test_prepared_run_without_artifacts_is_still_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runs_root = root / "runs"
            prepared = runs_root / "20260802_000000_9"
            prepared.mkdir(parents=True)
            manifest_status(prepared, "prepared")

            runs = list_runs(runs_root)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, "prepared")
        self.assertLess(runs[0].size_bytes, 100)

    def test_corrupt_manifest_is_reported_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run_dir = root / "runs" / "broken"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                "{not json", encoding="utf-8"
            )

            runs = list_runs(root / "runs")

        self.assertEqual(runs[0].status, "unknown")

    def test_ignores_non_directory_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runs_root = root / "runs"
            runs_root.mkdir()
            (runs_root / "stray.txt").write_text("not a run", encoding="utf-8")
            run_dir = runs_root / "20260802_000000_1"
            run_dir.mkdir()
            manifest_status(run_dir, "running")

            runs = list_runs(runs_root)

        self.assertEqual([run.run_id for run in runs], ["20260802_000000_1"])

    def test_missing_runs_root_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(list_runs(Path(temporary) / "missing"), [])


class SizeHelperTests(unittest.TestCase):
    def test_directory_size_sums_nested_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "data"
            (directory / "a").mkdir(parents=True)
            (directory / "a" / "one.bin").write_bytes(b"1" * 3)
            (directory / "b").mkdir()
            (directory / "b" / "two.bin").write_bytes(b"2" * 7)
            (directory / "top.bin").write_bytes(b"3" * 11)

            self.assertEqual(directory_size(directory), 3 + 7 + 11)

    def test_runs_total_size_sums_run_sizes(self) -> None:
        runs = [
            RunInfo("a", Path("a"), "prepared", 100),
            RunInfo("b", Path("b"), "committed", 250),
            RunInfo("c", Path("c"), "unknown", 50),
        ]

        self.assertEqual(runs_total_size(runs), 400)


class DeleteRunsTests(unittest.TestCase):
    def test_delete_removes_only_selected_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runs_root = root / "runs"
            keep = runs_root / "keep"
            remove = runs_root / "remove"
            keep.mkdir(parents=True)
            remove.mkdir()
            manifest_status(keep, "committed")
            manifest_status(remove, "prepared")

            deleted = delete_runs([RunInfo("remove", remove, "prepared", 0)])

            self.assertEqual(deleted, 1)
            self.assertTrue(keep.is_dir())
            self.assertFalse(remove.exists())

    def test_delete_counts_only_existing_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            existing = root / "runs" / "existing"
            existing.mkdir(parents=True)
            missing = root / "runs" / "missing"

            deleted = delete_runs(
                [
                    RunInfo("existing", existing, "running", 0),
                    RunInfo("missing", missing, "unknown", 0),
                ]
            )

            self.assertEqual(deleted, 1)
            self.assertFalse(existing.exists())


class CleanupDisplayTests(unittest.TestCase):
    def test_format_size_human_readable(self) -> None:
        self.assertEqual(format_size(0), "0 B")
        self.assertEqual(format_size(512), "512 B")
        self.assertEqual(format_size(1024), "1.0 KB")
        self.assertEqual(format_size(1536), "1.5 KB")
        self.assertEqual(format_size(5 * 1024 * 1024), "5.0 MB")
        self.assertEqual(format_size(3 * 1024**3), "3.0 GB")

    def test_warning_threshold_is_five_gigabytes(self) -> None:
        self.assertEqual(CLEANUP_WARNING_BYTES, 5 * 1024**3)


if __name__ == "__main__":
    unittest.main()
