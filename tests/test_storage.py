from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cc_cover.data_root import RuntimePaths, runtime_paths
from cc_cover.storage import (
    ASR_DEPENDENCIES_BYTES,
    CLEANUP_WARNING_BYTES,
    DiskCheck,
    FAST_WHISPER_MODELS_BYTES,
    FUNASR_MODELS_BYTES,
    INSTALL_BUFFER_BYTES,
    RunInfo,
    TORCH_CPU_BYTES,
    TORCH_CUDA_BYTES,
    clean_model_cache,
    clear_all_data_text,
    clear_local_data,
    delete_runs,
    directory_size,
    disk_precheck,
    disk_precheck_text,
    estimate_install_required_bytes,
    format_size,
    install_download_bytes,
    list_runs,
    local_data_usage,
    model_cache_cleanup_text,
    runs_total_size,
)


class EstimateBytesTests(unittest.TestCase):
    def test_cuda_requires_more_disk_than_cpu(self) -> None:
        self.assertGreater(
            estimate_install_required_bytes("cuda"),
            estimate_install_required_bytes("cpu"),
        )

    def test_install_download_cuda_larger_than_cpu(self) -> None:
        self.assertGreater(install_download_bytes("cuda"), install_download_bytes("cpu"))

    def test_install_download_combines_torch_and_dependencies(self) -> None:
        self.assertEqual(
            install_download_bytes("cpu"),
            TORCH_CPU_BYTES + ASR_DEPENDENCIES_BYTES,
        )
        self.assertEqual(
            install_download_bytes("cuda"),
            TORCH_CUDA_BYTES + ASR_DEPENDENCIES_BYTES,
        )

    def test_install_download_excludes_torch_when_not_requested(self) -> None:
        self.assertEqual(
            install_download_bytes("cuda", include_torch=False),
            ASR_DEPENDENCIES_BYTES,
        )

    def test_install_download_excludes_asr_when_not_requested(self) -> None:
        self.assertEqual(
            install_download_bytes("cpu", include_asr=False),
            TORCH_CPU_BYTES,
        )

    def test_install_download_zero_when_nothing_requested(self) -> None:
        self.assertEqual(
            install_download_bytes("cpu", include_torch=False, include_asr=False),
            0,
        )

    def test_precheck_total_includes_models_and_buffer(self) -> None:
        self.assertEqual(
            estimate_install_required_bytes("cpu"),
            install_download_bytes("cpu")
            + FUNASR_MODELS_BYTES
            + FAST_WHISPER_MODELS_BYTES
            + INSTALL_BUFFER_BYTES,
        )

    def test_unknown_device_falls_back_to_cpu(self) -> None:
        self.assertEqual(
            estimate_install_required_bytes("tpu"),
            estimate_install_required_bytes("cpu"),
        )

    def test_estimates_are_a_few_gigabytes_not_tens_of_gigabytes(self) -> None:
        cpu_total = estimate_install_required_bytes("cpu")
        self.assertGreater(cpu_total, 2 * 1024**3)
        self.assertLess(cpu_total, 20 * 1024**3)
        self.assertLess(estimate_install_required_bytes("cuda"), 30 * 1024**3)

    def test_torch_cuda_wheel_is_multi_gigabyte(self) -> None:
        self.assertGreater(TORCH_CUDA_BYTES, 2 * 1024**3)
        self.assertLess(TORCH_CUDA_BYTES, 10 * 1024**3)

    def test_model_estimates_are_gigabyte_scale(self) -> None:
        self.assertGreater(FUNASR_MODELS_BYTES, 1024**3)
        self.assertLess(FUNASR_MODELS_BYTES, 5 * 1024**3)
        self.assertGreater(FAST_WHISPER_MODELS_BYTES, 1024**3)
        self.assertLess(FAST_WHISPER_MODELS_BYTES, 4 * 1024**3)


class DiskPrecheckTests(unittest.TestCase):
    def test_precheck_reports_sufficient_when_free_is_large(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data"
            target.mkdir()
            usage = _usage(free_bytes=10 * 1024**3)
            with patch("cc_cover.storage.shutil.disk_usage", return_value=usage):
                check = disk_precheck(target, required_bytes=2 * 1024**3)

        self.assertTrue(check.sufficient)
        self.assertEqual(check.required_bytes, 2 * 1024**3)
        self.assertEqual(check.free_bytes, 10 * 1024**3)
        self.assertEqual(check.target, target.resolve())

    def test_precheck_reports_insufficient_when_free_is_small(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data"
            target.mkdir()
            usage = _usage(free_bytes=1 * 1024**3)
            with patch("cc_cover.storage.shutil.disk_usage", return_value=usage):
                check = disk_precheck(target, required_bytes=2 * 1024**3)

        self.assertFalse(check.sufficient)

    def test_precheck_handles_missing_target_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data"
            usage = _usage(free_bytes=10 * 1024**3)
            with patch("cc_cover.storage.shutil.disk_usage", return_value=usage):
                check = disk_precheck(target, required_bytes=1024**3)

        self.assertTrue(check.sufficient)

    def test_precheck_text_mentions_required_and_free(self) -> None:
        check = disk_precheck_result(sufficient=False, required=4 * 1024**3, free=1 * 1024**3)

        text = disk_precheck_text(check)

        self.assertIn("至少需要 4.0 GB", text)
        self.assertIn("剩余 1.0 GB", text)
        self.assertIn("空间不足", text)

    def test_precheck_text_suggests_cleaning_runs_when_large(self) -> None:
        check = disk_precheck_result(sufficient=False, required=4 * 1024**3, free=1 * 1024**3)

        text = disk_precheck_text(check, runs_bytes=500 * 1024**3)

        self.assertIn("建议先清理运行目录", text)
        self.assertIn("500.0 GB", text)

    def test_precheck_text_no_runs_suggestion_when_zero(self) -> None:
        check = disk_precheck_result(sufficient=False, required=4 * 1024**3, free=1 * 1024**3)

        text = disk_precheck_text(check)

        self.assertNotIn("运行目录", text)

    def test_precheck_text_sufficient_has_no_warning(self) -> None:
        check = disk_precheck_result(sufficient=True, required=2 * 1024**3, free=20 * 1024**3)

        text = disk_precheck_text(check)

        self.assertNotIn("空间不足", text)
        self.assertIn("至少需要 2.0 GB", text)


class LocalCleanupTests(unittest.TestCase):
    def test_clean_model_cache_removes_contents_and_recreates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            _write(paths.model_cache / "models" / "funasr.bin", 10)
            (paths.model_cache / "huggingface").mkdir(parents=True)

            clean_model_cache(paths)

            self.assertTrue(paths.model_cache.is_dir())
            self.assertEqual(list(paths.model_cache.iterdir()), [])

    def test_clean_model_cache_creates_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))

            clean_model_cache(paths)

            self.assertTrue(paths.model_cache.is_dir())

    def test_local_data_usage_sums_all_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            _write(paths.venv_root / "python.exe", 100)
            _write(paths.model_cache / "model.bin", 200)
            _write(paths.runs_root / "run1" / "manifest.json", 300)
            _write(paths.temp_root / "scratch.tmp", 400)

            usage = local_data_usage(paths)

        self.assertEqual(usage, 100 + 200 + 300 + 400)

    def test_local_data_usage_missing_directories_count_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            usage = local_data_usage(_paths(Path(temporary)))

        self.assertEqual(usage, 0)

    def test_clear_local_data_removes_subdirs_keeps_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            _write(paths.venv_root / "python.exe", 100)
            _write(paths.model_cache / "model.bin", 200)
            _write(paths.runs_root / "run1" / "manifest.json", 300)
            _write(paths.temp_root / "scratch.tmp", 400)
            settings = paths.data_root / "settings.json"
            settings.write_text('{"device": "cpu"}', encoding="utf-8")

            clear_local_data(paths)

            for directory in (
                paths.venv_root,
                paths.model_cache,
                paths.runs_root,
                paths.temp_root,
            ):
                self.assertTrue(directory.is_dir())
                self.assertEqual(list(directory.iterdir()), [])
            self.assertTrue(settings.is_file())

    def test_clear_all_data_text_lists_items_and_size(self) -> None:
        text = clear_all_data_text(2 * 1024**3)

        self.assertIn("2.0 GB", text)
        self.assertIn("venv", text)
        self.assertIn("模型缓存", text)
        self.assertIn("运行记录", text)

    def test_model_cache_cleanup_text_mentions_redownload(self) -> None:
        text = model_cache_cleanup_text(5 * 1024**2)

        self.assertIn("5.0 MB", text)
        self.assertIn("重新下载", text)


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
    def test_warning_threshold_is_five_gigabytes(self) -> None:
        self.assertEqual(CLEANUP_WARNING_BYTES, 5 * 1024**3)


class _FakeDiskUsage:
    def __init__(self, free_bytes: int) -> None:
        self.free = free_bytes
        self.total = free_bytes + 20 * 1024**3
        self.used = 20 * 1024**3


def _usage(free_bytes: int) -> _FakeDiskUsage:
    return _FakeDiskUsage(free_bytes)


def disk_precheck_result(
    *, sufficient: bool, required: int, free: int
) -> DiskCheck:
    return DiskCheck(
        target=Path("."),
        required_bytes=required,
        free_bytes=free,
        sufficient=sufficient,
    )


def _paths(root: Path) -> RuntimePaths:
    return runtime_paths(frozen=True, bundle_root=root / "bundle", data_root=root / "data")


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


if __name__ == "__main__":
    unittest.main()
