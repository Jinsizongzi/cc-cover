from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cc_cover.gui_support import (
    ASR_DEPENDENCIES_BYTES,
    DiskCheck,
    FAST_WHISPER_MODELS_BYTES,
    FUNASR_MODELS_BYTES,
    INSTALL_BUFFER_BYTES,
    InstallProgressSnapshot,
    InstallProgressTracker,
    RuntimePaths,
    TORCH_CPU_BYTES,
    TORCH_CUDA_BYTES,
    clean_model_cache,
    clear_all_data_text,
    clear_local_data,
    disk_precheck,
    disk_precheck_text,
    estimate_install_required_bytes,
    format_size,
    install_download_bytes,
    install_progress_text,
    local_data_usage,
    model_cache_cleanup_text,
    runtime_paths,
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
        self.assertLess(FAST_WHISPER_MODELS_BYTES, 3 * 1024**3)


class DiskPrecheckTests(unittest.TestCase):
    def test_precheck_reports_sufficient_when_free_is_large(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data"
            target.mkdir()
            usage = _usage(free_bytes=10 * 1024**3)
            with patch("cc_cover.gui_support.shutil.disk_usage", return_value=usage):
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
            with patch("cc_cover.gui_support.shutil.disk_usage", return_value=usage):
                check = disk_precheck(target, required_bytes=2 * 1024**3)

        self.assertFalse(check.sufficient)

    def test_precheck_handles_missing_target_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data"
            usage = _usage(free_bytes=10 * 1024**3)
            with patch("cc_cover.gui_support.shutil.disk_usage", return_value=usage):
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


class InstallProgressTrackerTests(unittest.TestCase):
    def test_accumulates_download_header_sizes(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        tracker.on_output("Downloading torch-2.5.1-cp312.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=10.0)

        self.assertEqual(snapshot.downloaded_bytes, 500 * 1024**2)

    def test_parses_small_units(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        tracker.on_output("Downloading six.whl (10 kB)\n")

        self.assertEqual(tracker.snapshot(now=1.0).downloaded_bytes, 10 * 1024)

    def test_multiple_headers_accumulate(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        tracker.on_output("Downloading a.whl (100.0 MB)\nDownloading b.whl (50.0 MB)\n")

        self.assertEqual(
            tracker.snapshot(now=5.0).downloaded_bytes, 150 * 1024**2
        )

    def test_ignores_non_download_lines(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        for line in (
            "Collecting six\n",
            "Installing collected packages\n",
            "Successfully installed six\n",
        ):
            tracker.on_output(line)

        self.assertEqual(tracker.snapshot(now=5.0).downloaded_bytes, 0)

    def test_downloaded_capped_at_total(self) -> None:
        tracker = InstallProgressTracker(total_bytes=100 * 1024**2, component_count=4)
        tracker.on_output("Downloading huge.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=5.0)

        self.assertEqual(snapshot.downloaded_bytes, 100 * 1024**2)

    def test_percent_reflects_completed_components(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1000 * 1024**2, component_count=4)
        self.assertEqual(tracker.snapshot(now=5.0).percent, 0)

        tracker.on_component(1)
        self.assertEqual(tracker.snapshot(now=5.0).percent, 0)

        tracker.on_component(3)
        self.assertEqual(tracker.snapshot(now=5.0).percent, 50)

    def test_speed_and_remaining_estimated_from_elapsed(self) -> None:
        tracker = InstallProgressTracker(
            total_bytes=1000 * 1024**2, component_count=4, started_at=0.0
        )
        tracker.on_output("Downloading a.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=100.0)

        self.assertAlmostEqual(snapshot.speed_bytes, 5 * 1024**2)
        self.assertAlmostEqual(snapshot.remaining_seconds, 100.0)

    def test_no_speed_estimate_when_little_time_elapsed(self) -> None:
        tracker = InstallProgressTracker(
            total_bytes=1000 * 1024**2, component_count=4, started_at=0.0
        )
        tracker.on_output("Downloading a.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=1.0)

        self.assertIsNone(snapshot.speed_bytes)
        self.assertIsNone(snapshot.remaining_seconds)

    def test_no_remaining_estimate_when_downloaded(self) -> None:
        tracker = InstallProgressTracker(total_bytes=500 * 1024**2, component_count=4)
        tracker.on_output("Downloading a.whl (500.0 MB)\n")

        snapshot = tracker.snapshot(now=60.0)

        self.assertEqual(snapshot.downloaded_bytes, 500 * 1024**2)
        self.assertIsNone(snapshot.remaining_seconds)

    def test_zero_total_never_divides_by_zero(self) -> None:
        tracker = InstallProgressTracker(total_bytes=0, component_count=3)
        tracker.on_output("Downloading a.whl (10.0 MB)\n")

        snapshot = tracker.snapshot(now=5.0)

        self.assertEqual(snapshot.downloaded_bytes, 0)
        self.assertEqual(snapshot.percent, 0)
        self.assertIsNone(snapshot.speed_bytes)

    def test_component_index_tracks_worker_emissions(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=6)
        tracker.on_component(3)

        self.assertEqual(tracker.snapshot(now=1.0).component_index, 3)
        self.assertEqual(tracker.snapshot(now=1.0).component_count, 6)

    def test_parses_rich_progress_bar_frame_when_present(self) -> None:
        tracker = InstallProgressTracker(total_bytes=1024**3, component_count=4)
        tracker.on_output("\r  45%|##### | 450.0/1000.0 MB [00:10<00:12, 45MB/s]")

        self.assertEqual(tracker.snapshot(now=5.0).downloaded_bytes, 450 * 1024**2)


class InstallProgressTextTests(unittest.TestCase):
    def test_text_contains_component_downloaded_and_remaining(self) -> None:
        snapshot = InstallProgressSnapshot(
            component_index=3,
            component_count=6,
            downloaded_bytes=300 * 1024**2,
            total_bytes=900 * 1024**2,
            percent=33,
            speed_bytes=5 * 1024**2,
            remaining_seconds=120.0,
        )

        text = install_progress_text(snapshot)

        self.assertIn("组件 3/6", text)
        self.assertIn("已下载约 300.0 MB", text)
        self.assertIn("约剩余 2 分 0 秒", text)

    def test_text_omits_remaining_when_unavailable(self) -> None:
        snapshot = InstallProgressSnapshot(
            component_index=1,
            component_count=6,
            downloaded_bytes=10 * 1024**2,
            total_bytes=900 * 1024**2,
            percent=1,
            speed_bytes=None,
            remaining_seconds=None,
        )

        text = install_progress_text(snapshot)

        self.assertNotIn("约剩余", text)
        self.assertIn("已下载约 10.0 MB", text)

    def test_text_omits_downloaded_when_zero(self) -> None:
        snapshot = InstallProgressSnapshot(
            component_index=1,
            component_count=6,
            downloaded_bytes=0,
            total_bytes=900 * 1024**2,
            percent=0,
            speed_bytes=None,
            remaining_seconds=None,
        )

        text = install_progress_text(snapshot)

        self.assertNotIn("已下载约", text)
        self.assertIn("组件 1/6", text)


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
