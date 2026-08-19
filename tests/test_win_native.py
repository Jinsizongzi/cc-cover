from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cc_cover.gui.win_native import SingleInstanceLock, focus_existing_window


class SingleInstanceLockTests(unittest.TestCase):
    def test_mutex_name_is_stable_for_same_data_root(self) -> None:
        first = SingleInstanceLock(Path("E:/CC-Cover/Data"))
        second = SingleInstanceLock(Path("e:/cc-cover/data"))
        third = SingleInstanceLock(Path("D:/other"))
        self.assertEqual(first._mutex_name(), second._mutex_name())
        self.assertNotEqual(first._mutex_name(), third._mutex_name())
        self.assertTrue(first._mutex_name().startswith("CC-Cover-"))

    def test_second_acquire_is_rejected_until_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = SingleInstanceLock(root)
            self.assertTrue(first.acquire())
            try:
                self.assertFalse(SingleInstanceLock(root).acquire())
            finally:
                first.release()
            self.assertTrue(SingleInstanceLock(root).acquire())

    def test_context_manager_releases_lock_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with SingleInstanceLock(root) as first:
                self.assertTrue(first.acquired)
                self.assertFalse(SingleInstanceLock(root).acquire())
            self.assertTrue(SingleInstanceLock(root).acquire())

    def test_second_process_is_rejected_until_first_exits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_code = (
                "import sys, time\n"
                "from pathlib import Path\n"
                "from cc_cover.gui.win_native import SingleInstanceLock\n"
                "lock = SingleInstanceLock(Path(sys.argv[1]))\n"
                "print(lock.acquire(), flush=True)\n"
                "time.sleep(10)\n"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", child_code, str(root)],
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                first_line = child.stdout.readline()
                self.assertEqual(first_line.strip(), "True")
                self.assertFalse(SingleInstanceLock(root).acquire())
            finally:
                child.terminate()
                child.wait(timeout=10)
                child.stdout.close()
            # wait() 确认子进程已退出，不代表内核已经同步清理完它持有的
            # 全部句柄（含命名互斥体）——这两者之间有真实但短暂的时序
            # 窗口，轮询代替单次断言，避免在这个窗口内偶发失败。
            deadline = time.monotonic() + 2.0
            acquired = False
            while time.monotonic() < deadline:
                if SingleInstanceLock(root).acquire():
                    acquired = True
                    break
                time.sleep(0.05)
            self.assertTrue(acquired)

    def test_focus_existing_window_never_raises(self) -> None:
        self.assertIsInstance(focus_existing_window(), bool)
        self.assertIsInstance(focus_existing_window(""), bool)


@unittest.skipIf(
    os.name == "nt", "Windows 使用命名互斥体，文件锁回退在非 Windows 平台验证"
)
class SingleInstanceLockFileFallbackTests(unittest.TestCase):
    def test_file_lock_rejects_live_second_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = SingleInstanceLock(root)
            self.assertTrue(first._acquire_file())
            try:
                self.assertFalse(SingleInstanceLock(root)._acquire_file())
            finally:
                first._release_file()
            self.assertTrue(SingleInstanceLock(root)._acquire_file())

    def test_file_lock_cleans_up_stale_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "instance.lock"
            dead = subprocess.Popen([sys.executable, "-c", "pass"], shell=False)
            dead.wait()
            lock_path.write_text(f"{dead.pid}\n", encoding="ascii")

            self.assertTrue(SingleInstanceLock(root)._acquire_file())
            self.assertEqual(
                lock_path.read_text(encoding="ascii").strip(),
                str(os.getpid()),
            )

    def test_file_lock_keeps_corrupt_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "instance.lock"
            lock_path.write_text("not-a-pid\n", encoding="ascii")
            self.assertFalse(SingleInstanceLock(root)._acquire_file())


if __name__ == "__main__":
    unittest.main()
