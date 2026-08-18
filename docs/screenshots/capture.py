"""用真实界面生成 README 截图（Windows，需 tkinter 与 Pillow）。

用法：cd 仓库根目录 && "python" docs/screenshots/capture.py
输出：docs/screenshots/main.png / features.png / guide.png

脚本构造真实的 CCCoverApp 窗口，等待环境检查启动与界面刷新后，
以真实渲染路径（_display_report）填充代表性候选数据，再截取窗口像素存为 PNG。
"""

from __future__ import annotations

import ctypes
import sys
import tempfile
import time
from pathlib import Path

try:  # 先声明 DPI 感知，保证 Tk 逻辑像素与 ImageGrab 物理像素一致。
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

import tkinter as tk
from PIL import ImageGrab

from cc_cover.data_root import ensure_data_root, runtime_paths
from cc_cover.win_native import SingleInstanceLock

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def _sample_report() -> dict[str, object]:
    """代表性扫描结果：与真实 manifest 同构，状态字段沿用真实取值。"""
    base = "D:\\素材\\纪录片"
    videos = [
        ("S01E01.mp4", "missing"),
        ("S01E02.mp4", "zero_byte"),
        ("S01E03.mp4", "whitespace_only"),
        ("S01E04.mp4", "nonempty"),
        ("S01E05.mp4", "missing"),
        ("S01E06.mp4", "zero_byte"),
        ("S01E07.mp4", "nonempty"),
        ("S02E01.mp4", "missing"),
    ]
    return {
        "video_count": 12,
        "candidate_count": 8,
        "conflict_count": 1,
        "protected_nonempty_txt_count": 2,
        "candidates": [
            {
                "state": state,
                "video_path": f"{base}\\{name}",
                "target_path": f"{base}\\{name.rsplit('.', 1)[0]}.txt",
            }
            for name, state in videos
        ],
        "conflicts": [
            {
                "videos": [f"{base}\\S02E02.mp4", f"{base}\\S02E02.mkv"],
                "target_path": f"{base}\\S02E02.txt",
            }
        ],
    }


def _pump(root: tk.Tk, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        root.update_idletasks()
        time.sleep(0.02)


def _capture(root: tk.Tk, target: Path) -> None:
    root.lift()
    root.attributes("-topmost", True)
    root.update_idletasks()
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    width = root.winfo_width()
    height = root.winfo_height()
    image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    image.save(target)
    root.attributes("-topmost", False)


def main() -> int:
    if not sys.platform.startswith("win"):
        print("截图脚本仅支持 Windows。")
        return 1

    temporary = tempfile.TemporaryDirectory()
    data_root = Path(temporary.name).resolve()
    paths = runtime_paths(data_root=data_root)
    ensure_data_root(paths)
    lock = SingleInstanceLock(paths.data_root)
    if not lock.acquire():
        print("单实例锁不可用，退出。")
        return 1

    root = tk.Tk()
    root.withdraw()
    root.geometry("1080x760")
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = max(0, (screen_width - 1080) // 2)
    y = max(0, (screen_height - 760) // 2 - 30)
    root.geometry(f"1080x760+{x}+{y}")
    root.deiconify()

    from cc_cover.gui import CCCoverApp

    app = CCCoverApp(root, paths, lock)
    _pump(root, 2.0)  # 等待环境检查与缓存显示刷新
    app._display_report(_sample_report())
    _pump(root, 0.8)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    _capture(root, OUTPUT / "main.png")
    for tab, name in ((app.feature_tab, "features"), (app.guide_tab, "guide")):
        app.notebook.select(tab)
        _pump(root, 0.5)
        _capture(root, OUTPUT / f"{name}.png")

    root.destroy()
    lock.release()
    temporary.cleanup()
    print("截图已生成：", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
