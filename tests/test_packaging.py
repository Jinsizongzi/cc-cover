from __future__ import annotations

import importlib.util
import re
import struct
import unittest
from pathlib import Path

from cc_cover import gui_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_build_version_info():
    path = PROJECT_ROOT / "packaging" / "build_version_info.py"
    spec = importlib.util.spec_from_file_location(
        "build_version_info", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackagingTests(unittest.TestCase):
    def test_version_resource_matches_source_version(self) -> None:
        """version_info.txt 与 __init__.py 的 __version__ 保持一致（单一来源）。"""
        module = _load_build_version_info()
        expected = module.version_info_text(module.current_version())
        actual = (
            PROJECT_ROOT / "packaging" / "version_info.txt"
        ).read_text(encoding="utf-8")
        self.assertEqual(actual, expected)

    def test_version_resource_has_four_part_file_version(self) -> None:
        """FixedFileInfo 的 filevers/prodvers 必须是四段整数元组。"""
        module = _load_build_version_info()
        fixed = module.parse_version(module.current_version())
        self.assertEqual(len(fixed), 4)
        for part in fixed:
            self.assertIsInstance(part, int)
        self.assertEqual(fixed[3], 0)  # build 号缺省为 0

    def test_iss_default_version_matches_source_version(self) -> None:
        """CC-Cover.iss 的默认版本号与 __init__.py 一致，避免手工漂移。"""
        iss = (PROJECT_ROOT / "packaging" / "CC-Cover.iss").read_text(
            encoding="utf-8"
        )
        match = re.search(r'#define MyAppVersion "([^"]+)"', iss)
        self.assertIsNotNone(match, "CC-Cover.iss 缺少默认 MyAppVersion")
        assert match is not None
        self.assertEqual(match.group(1), _load_build_version_info().current_version())

    def test_iss_installs_per_user_to_localappdata(self) -> None:
        """安装器默认当前用户安装到 %LOCALAPPDATA%\\Programs\\CC-Cover。"""
        iss = (PROJECT_ROOT / "packaging" / "CC-Cover.iss").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "DefaultDirName={localappdata}\\Programs\\{#MyAppName}", iss
        )
        self.assertIn("PrivilegesRequired=lowest", iss)
        self.assertNotIn("PrivilegesRequired=admin", iss)

    def test_iss_packages_onedir_and_cleans_data_root_on_uninstall(self) -> None:
        """安装器打包 onedir 产物；卸载时清理数据根（与运行时布局单一来源一致）。"""
        iss = (PROJECT_ROOT / "packaging" / "CC-Cover.iss").read_text(
            encoding="utf-8"
        )
        self.assertIn("Source: \"..\\dist\\CC-Cover\\*\"", iss)
        self.assertIn("recursesubdirs", iss)
        data_root_items = set(gui_support.DATA_ROOT_SUBDIRECTORIES) | {
            gui_support.SETTINGS_FILENAME
        }
        for name in sorted(data_root_items):
            with self.subTest(name=name):
                self.assertIn(f'Name: "{{app}}\\{name}"', iss)

    def test_iss_uses_vendored_chinese_language_file(self) -> None:
        """中文语言文件随仓库自带，构建不依赖 Inno 安装是否带全语言包。"""
        iss = (PROJECT_ROOT / "packaging" / "CC-Cover.iss").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'Name: "chinesesimplified"; MessagesFile: '
            '"Languages\\ChineseSimplified.isl"',
            iss,
        )
        isl = (
            PROJECT_ROOT / "packaging" / "Languages" / "ChineseSimplified.isl"
        )
        self.assertTrue(isl.is_file(), "ChineseSimplified.isl 缺失")
        self.assertGreater(isl.stat().st_size, 1000)

    def test_icon_is_valid_ico_with_multiple_sizes(self) -> None:
        """assets/app.ico 是合法 ICO，且包含常见的多尺寸条目。"""
        path = PROJECT_ROOT / "assets" / "app.ico"
        data = path.read_bytes()
        reserved, kind, count = struct.unpack("<HHH", data[:6])
        self.assertEqual(reserved, 0)
        self.assertEqual(kind, 1)  # ICONDIR 类型：图标
        sizes = set()
        offset = 6
        for _ in range(count):
            width, height = data[offset], data[offset + 1]
            sizes.add((width or 256, height or 256))
            offset += 16
        self.assertGreaterEqual(count, 5)
        for expected in ((16, 16), (32, 32), (256, 256)):
            self.assertIn(expected, sizes)

    def test_workflow_builds_onedir_and_both_release_artifacts(self) -> None:
        """发布 CI 使用 onedir 构建，并产出安装器与绿色压缩包两种发布物。"""
        workflow = (
            PROJECT_ROOT / ".github" / "workflows" / "windows-gui.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("--onedir", workflow)
        self.assertIn("--noupx", workflow)
        self.assertIn("--version-file packaging/version_info.txt", workflow)
        self.assertIn("--icon assets/app.ico", workflow)
        # ASR 运行环境由数据根 venv 在首跑时安装，冻结包必须排除整个运行时栈，
        # 否则 onedir 会膨胀到数 GB（torch+tensorflow+modelscope）。
        for module in (
            "torch",
            "funasr",
            "modelscope",
            "faster_whisper",
            "ctranslate2",
            "tensorflow",
            "imageio_ffmpeg",
        ):
            with self.subTest(exclude=module):
                self.assertIn(f"--exclude-module {module}", workflow)
        self.assertIn("iscc.exe", workflow)
        self.assertIn("Compress-Archive", workflow)
        self.assertIn("dist/CC-Cover-Setup-*.exe", workflow)
        self.assertIn("dist/CC-Cover-*-portable.zip", workflow)


if __name__ == "__main__":
    unittest.main()
