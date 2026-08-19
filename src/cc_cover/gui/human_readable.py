from __future__ import annotations

import re

_ANSI_CSI_PATTERN = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def strip_ansi_escapes(text: str) -> str:
    """剥掉 ANSI CSI 转义序列（光标移动、颜色控制码等）。

    tqdm 之类的进度条在真实终端里靠这些转义序列原地刷新同一行，不是给
    Tkinter Text 控件这种不解析转义码的地方看的——不剥掉会在界面上露出
    "[A"/"34m"/"0m" 这类残留字符。
    """
    return _ANSI_CSI_PATTERN.sub("", text)


def format_size(size_bytes: int) -> str:
    """字节数转人类可读大小；小于 1KB 按字节，否则保留一位小数。"""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{int(size_bytes)} B"


def format_duration(seconds: float | None) -> str:
    """秒数转中文时长；无法计算时返回「未知」。"""
    if seconds is None:
        return "未知"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} 时 {minutes} 分 {secs} 秒"
    if minutes:
        return f"{minutes} 分 {secs} 秒"
    return f"{secs} 秒"
