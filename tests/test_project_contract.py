from __future__ import annotations

import re
import unittest
from pathlib import Path


class ProjectContractTests(unittest.TestCase):
    def test_user_facing_files_have_no_default_path_or_write_switch(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        files = [
            project_root / "README.md",
            project_root / "config.example.json",
            project_root / "run.ps1",
            project_root / "setup.ps1",
            project_root / "start.cmd",
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

    def test_double_click_launcher_starts_interactive_script(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        launcher = (project_root / "start.cmd").read_text(encoding="utf-8")
        self.assertIn("run.ps1", launcher)

    def test_powershell_scripts_have_windows_utf8_bom(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        for name in ("run.ps1", "setup.ps1"):
            with self.subTest(name=name):
                self.assertTrue(
                    (project_root / name).read_bytes().startswith(b"\xef\xbb\xbf")
                )


if __name__ == "__main__":
    unittest.main()
