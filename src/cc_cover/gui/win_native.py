from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_LOCK_FILENAME = "instance.lock"
_ERROR_ALREADY_EXISTS = 183


def open_in_explorer(path_value: str) -> None:
    """在资源管理器里定位并选中该文件；找不到 explorer 时退化为打开父目录。"""
    if not path_value:
        return
    path = Path(path_value)
    try:
        subprocess.Popen(
            ["explorer", "/select,", str(path)],
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError:
        os.startfile(str(path.parent))


class SingleInstanceLock:
    """基于数据根的单实例锁。

    Windows 使用命名互斥体（进程退出后由系统自动释放），其他平台在数据根
    写入包含 PID 的锁文件并在发现陈旧锁时自动清理。
    """

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.acquired = False
        self._handle: Any = None
        self._lock_path: Path | None = None

    def _identity(self) -> str:
        """返回数据根的规范化标识，用于生成跨进程一致的锁名。"""
        normalized = os.path.normcase(str(self.data_root)).replace("\\", "/").lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    def _mutex_name(self) -> str:
        return f"CC-Cover-{self._identity()}"

    def acquire(self) -> bool:
        """尝试获取锁；已有实例持有同一数据根锁时返回 False。"""
        if self.acquired:
            return True
        if os.name == "nt":
            return self._acquire_mutex()
        return self._acquire_file()

    def release(self) -> None:
        if os.name == "nt":
            self._release_mutex()
        else:
            self._release_file()

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def _acquire_mutex(self) -> bool:
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = (
                wintypes.LPVOID,
                wintypes.BOOL,
                wintypes.LPCWSTR,
            )
            kernel32.CreateMutexW.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
        except (ImportError, OSError, AttributeError) as exc:
            raise OSError(f"无法创建单实例锁：{exc}") from exc
        handle = kernel32.CreateMutexW(None, False, self._mutex_name())
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error() or 1)
        if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        self.acquired = True
        return True

    def _release_mutex(self) -> None:
        if self._handle is not None:
            try:
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
                kernel32.CloseHandle.restype = wintypes.BOOL
                kernel32.CloseHandle(self._handle)
            except (ImportError, OSError, AttributeError):
                pass
            self._handle = None
        self.acquired = False

    def _acquire_file(self) -> bool:
        if self.acquired:
            return True
        lock_path = self.data_root / _LOCK_FILENAME
        for attempt in range(2):
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if attempt == 0 and not self._pid_alive(self._read_pid(lock_path)):
                    try:
                        lock_path.unlink()
                    except OSError:
                        return False
                    continue
                return False
            except OSError:
                return False
            try:
                os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            finally:
                os.close(descriptor)
            self._lock_path = lock_path
            self.acquired = True
            return True
        return False

    def _release_file(self) -> None:
        if self._lock_path is not None:
            try:
                self._lock_path.unlink()
            except OSError:
                pass
            self._lock_path = None
        self.acquired = False

    @staticmethod
    def _read_pid(lock_path: Path) -> int | None:
        try:
            text = lock_path.read_text(encoding="ascii").strip()
        except OSError:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if pid is None:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True


def focus_existing_window(prefix: str = "CC-Cover") -> bool:
    """尽力将已运行的 CC-Cover 主窗口带到前台（仅 Windows，失败时返回 False）。"""
    if os.name != "nt" or not prefix:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows.argtypes = (enum_proc, wintypes.LPARAM)
        user32.EnumWindows.restype = wintypes.BOOL
        user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = (
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        )
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        user32.SetForegroundWindow.restype = wintypes.BOOL
        found: list[int] = []

        def _enum(hwnd: int, _lparam: int) -> bool:
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                if buffer.value.startswith(prefix):
                    found.append(int(hwnd))
                    return False
            return True

        if not user32.EnumWindows(enum_proc(_enum), 0):
            return False
        if not found:
            return False
        handle = found[0]
        user32.ShowWindow(handle, 9)  # SW_RESTORE
        user32.SetForegroundWindow(handle)
        return True
    except (AttributeError, OSError, TypeError, ValueError):
        return False
