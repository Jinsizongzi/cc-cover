from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cc_cover.settings import (
    GuiSettings,
    SettingsError,
    load_gui_settings,
    read_settings,
    resolve_default_device,
    save_gui_settings,
    settings_file,
    write_settings,
)


class SettingsFileTests(unittest.TestCase):
    def test_settings_file_lives_inside_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(settings_file(root), (root / "settings.json").resolve())

    def test_read_settings_returns_empty_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(read_settings(Path(temporary)), {})

    def test_read_settings_treats_unaddressable_root_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocker = root / "blocker"
            blocker.write_bytes(b"")
            self.assertEqual(read_settings(blocker / "app"), {})

    def test_write_settings_round_trips_utf8_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"data_root": "D:/数据", "device": "cpu"})
            self.assertEqual(
                read_settings(root),
                {"data_root": "D:/数据", "device": "cpu"},
            )
            self.assertEqual(settings_file(root).is_file(), True)

    def test_read_settings_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_file(root).write_text("{not json", encoding="utf-8")
            with self.assertRaises(SettingsError):
                read_settings(root)

    def test_read_settings_rejects_non_object_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_file(root).write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(SettingsError):
                read_settings(root)


class GuiPreferencesTests(unittest.TestCase):
    def test_load_returns_defaults_for_missing_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = load_gui_settings(Path(temporary))

        self.assertEqual(
            settings,
            GuiSettings(
                scan_path="",
                device="auto",
                ffmpeg="",
                hash_videos=True,
            ),
        )

    def test_save_then_load_round_trips_user_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = GuiSettings(
                scan_path="D:/视频",
                device="cpu",
                ffmpeg="C:/ffmpeg.exe",
                hash_videos=False,
                hf_token="hf_abc123",
            )

            save_gui_settings(root, expected)
            loaded = load_gui_settings(root)

            self.assertEqual(loaded, expected)
            self.assertIn(
                "D:/视频", settings_file(root).read_text(encoding="utf-8")
            )

    def test_load_falls_back_to_default_hf_token_for_invalid_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"hf_token": 123})

            settings = load_gui_settings(root)

        self.assertEqual(settings.hf_token, "")

    def test_load_fills_defaults_for_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"device": "cpu", "unknown": "ignored"})

            settings = load_gui_settings(root)

        self.assertEqual(
            settings,
            GuiSettings(
                scan_path="",
                device="cpu",
                ffmpeg="",
                hash_videos=True,
            ),
        )

    def test_load_falls_back_to_defaults_for_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(
                root,
                {
                    "scan_path": None,
                    "device": "tpu",
                    "accelerator": "gpu",
                    "ffmpeg": None,
                    "hash_videos": "yes",
                },
            )

            settings = load_gui_settings(root)

        self.assertEqual(
            settings,
            GuiSettings(
                scan_path="",
                device="auto",
                ffmpeg="",
                hash_videos=True,
            ),
        )

    def test_save_preserves_existing_settings_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"data_root": "D:/data"})

            save_gui_settings(
                root,
                GuiSettings(
                    scan_path="C:/videos",
                    device="cuda",
                    ffmpeg="",
                    hash_videos=True,
                ),
            )

            stored = read_settings(root)
        self.assertEqual(stored["data_root"], "D:/data")
        self.assertEqual(stored["scan_path"], "C:/videos")
        self.assertEqual(stored["device"], "cuda")

    def test_load_migrates_legacy_accelerator_to_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"accelerator": "cpu"})

            settings = load_gui_settings(root)

        self.assertEqual(settings.device, "cpu")

    def test_load_prefers_legacy_accelerator_over_device_when_both_present(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"device": "cpu", "accelerator": "cuda"})

            settings = load_gui_settings(root)

        self.assertEqual(settings.device, "cuda")

    def test_load_keeps_valid_device_when_legacy_accelerator_is_invalid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"device": "cpu", "accelerator": "tpu"})

            settings = load_gui_settings(root)

        self.assertEqual(settings.device, "cpu")

    def test_save_drops_legacy_accelerator_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_settings(root, {"accelerator": "cuda", "data_root": "D:/data"})

            save_gui_settings(
                root,
                GuiSettings(
                    scan_path="C:/videos",
                    device="cpu",
                    ffmpeg="",
                    hash_videos=True,
                ),
            )

            stored = read_settings(root)
        self.assertNotIn("accelerator", stored)
        self.assertEqual(stored["device"], "cpu")
        self.assertEqual(stored["data_root"], "D:/data")

    def test_load_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_file(root).write_text("{not json", encoding="utf-8")
            with self.assertRaises(SettingsError):
                load_gui_settings(root)

    def test_save_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_file(root).write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(SettingsError):
                save_gui_settings(root, GuiSettings())


class ResolveDefaultDeviceTests(unittest.TestCase):
    def test_saved_choice_wins_over_detection(self) -> None:
        self.assertEqual(resolve_default_device("cuda", "cpu"), "cuda")
        self.assertEqual(resolve_default_device("cpu", "cuda"), "cpu")

    def test_auto_follows_detected_device(self) -> None:
        self.assertEqual(resolve_default_device("auto", "cuda"), "cuda")
        self.assertEqual(resolve_default_device("auto", "cpu"), "cpu")

    def test_auto_falls_back_to_cpu_without_detection(self) -> None:
        self.assertEqual(resolve_default_device("auto", None), "cpu")
        self.assertEqual(resolve_default_device("auto", "garbage"), "cpu")


if __name__ == "__main__":
    unittest.main()
