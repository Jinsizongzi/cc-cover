from __future__ import annotations

import re
import unittest
from pathlib import Path


class ProjectContractTests(unittest.TestCase):
    def test_legacy_script_launchers_are_removed(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        for name in ("run.ps1", "setup.ps1", "start.cmd"):
            with self.subTest(name=name):
                self.assertFalse((project_root / name).exists())

    def test_user_facing_files_have_no_default_scan_path(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        files = [
            project_root / "README.md",
            project_root / "src" / "cc_cover" / "gui.py",
            project_root / "src" / "cc_cover" / "gui_support.py",
            project_root / "src" / "cc_cover" / "cli.py",
            project_root / "src" / "cc_cover" / "models.py",
            project_root / "src" / "cc_cover" / "pipeline.py",
        ]
        drive_path = re.compile(r"[A-Za-z]:\\")
        obsolete_fragment = "-" + "app" + "ly"
        for path in files:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIsNone(drive_path.search(text))
                self.assertNotIn(obsolete_fragment, text.casefold())

    def test_readme_describes_graphical_software(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        readme = (project_root / "README.md").read_text(encoding="utf-8")
        self.assertIn("CC-Cover.exe", readme)
        self.assertIn("功能说明", readme)
        self.assertIn("操作指南", readme)


if __name__ == "__main__":
    unittest.main()
