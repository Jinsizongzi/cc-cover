from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

try:
    import tkinter as tk
except ImportError:  # ubuntu CI 通常未安装 python3-tk
    tk = None

from cc_cover.gui_support import (
    SingleInstanceLock,
    ensure_data_root,
    runtime_paths,
)


@unittest.skipUnless(
    sys.platform.startswith("win") and tk is not None,
    "GUI 冒烟只在 Windows runner 上创建真实窗口",
)
class GuiSmokeTests(unittest.TestCase):
    def _build_app(self) -> tuple[tk.Tk, object, SingleInstanceLock, Path]:
        from cc_cover.gui import CCCoverApp

        temporary = tempfile.TemporaryDirectory()
        root_path = Path(temporary.name).resolve()
        paths = runtime_paths(data_root=root_path)
        ensure_data_root(paths)
        lock = SingleInstanceLock(paths.data_root)
        if not lock.acquire():
            temporary.cleanup()
            self.skipTest("单实例锁不可用")
        try:
            root = tk.Tk()
        except tk.TclError:
            lock.release()
            temporary.cleanup()
            self.skipTest("当前会话无法创建 Tk 窗口")
        root.withdraw()
        try:
            app = CCCoverApp(root, paths, lock)
        except Exception:
            root.destroy()
            lock.release()
            temporary.cleanup()
            raise
        return root, app, lock, temporary

    def _teardown(
        self, root: tk.Tk, lock: SingleInstanceLock, temporary: tempfile.TemporaryDirectory
    ) -> None:
        try:
            root.destroy()
        finally:
            lock.release()
            temporary.cleanup()

    def test_window_title_and_tabs_build(self) -> None:
        root, app, lock, temporary = self._build_app()
        try:
            root.update_idletasks()

            self.assertIn("CC-Cover", root.title())
            labels = [
                str(self._notebook_tab_text(app, index)) for index in range(4)
            ]
            self.assertEqual(
                [label.strip() for label in labels],
                ["字幕补全", "功能说明", "操作指南", "运行日志"],
            )
        finally:
            self._teardown(root, lock, temporary)

    def test_core_widgets_exist_after_construction(self) -> None:
        root, app, lock, temporary = self._build_app()
        try:
            root.update_idletasks()

            for widget in (
                app.setup_button,
                app.start_button,
                app.cancel_button,
                app.candidate_tree,
                app.environment_label,
                app.choose_button,
                app.scan_button,
                app.hash_check,
            ):
                self.assertTrue(widget.winfo_exists(), widget)
            self.assertEqual(app.status.get(), "正在检查运行环境…")
            self.assertIn(app.device.get(), {"cuda", "cpu"})
        finally:
            self._teardown(root, lock, temporary)

    @staticmethod
    def _notebook_tab_text(app: object, index: int) -> str:
        return str(app.notebook.tab(index, "text"))


if __name__ == "__main__":
    unittest.main()
