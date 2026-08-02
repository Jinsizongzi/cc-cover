"""生成 PyInstaller 版本资源文件 packaging/version_info.txt。

版本号单一来源为 src/cc_cover/__init__.py 的 __version__。生成的 version_info.txt
作为 --version-file 传入 PyInstaller，使发布物（exe / 安装器）的版本信息与产品版本
保持一致，避免手工维护两处版本号。

用法：
  python packaging/build_version_info.py            # 生成 version_info.txt
  python packaging/build_version_info.py --version  # 仅打印当前版本号（供 CI 使用）
输出：packaging/version_info.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = ROOT / "src" / "cc_cover" / "__init__.py"
OUTPUT = ROOT / "packaging" / "version_info.txt"

_COMPANY = "Jinsizongzi"
_PRODUCT = "CC-Cover"
_DESCRIPTION = (
    "Safely recover empty same-name video subtitle TXT files "
    "with FunASR and faster-whisper."
)
_COPYRIGHT = "Copyright (c) 2026 Jinsizongzi"


def current_version() -> str:
    """从 __init__.py 读取 __version__；解析失败视为打包配置错误。"""
    text = INIT_PATH.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError(f"无法从 {INIT_PATH} 解析 __version__")
    return match.group(1)


def parse_version(version: str) -> tuple[int, int, int, int]:
    """版本号拆为 PyInstaller 需要的四段 (major, minor, patch, build)。"""
    parts = version.split(".")
    if len(parts) < 3:
        raise ValueError(f"版本号至少需要三段：{version!r}")
    try:
        triple = [int(part) for part in parts[:3]]
    except ValueError as exc:
        raise ValueError(f"版本号含非数字段：{version!r}") from exc
    build = int(parts[3]) if len(parts) > 3 else 0
    return triple[0], triple[1], triple[2], build


def version_info_text(version: str) -> str:
    """构造 PyInstaller 版本资源文件内容（简体中文 + Unicode 语言表）。"""
    fixed = parse_version(version)
    return (
        "VSVersionInfo(\n"
        "  ffi=FixedFileInfo(\n"
        f"    filevers={fixed!r},\n"
        f"    prodvers={fixed!r},\n"
        "    mask=0x3f,\n"
        "    flags=0x0,\n"
        "    OS=0x40004,\n"
        "    fileType=0x1,\n"
        "    subtype=0x0,\n"
        "    date=(0, 0)\n"
        "  ),\n"
        "  kids=[\n"
        "    StringFileInfo(\n"
        "      [\n"
        "        StringTable(\n"
        "          '080404b0',\n"
        "          [\n"
        f"            StringStruct('CompanyName', '{_COMPANY}'),\n"
        f"            StringStruct('FileDescription', '{_DESCRIPTION}'),\n"
        f"            StringStruct('FileVersion', '{version}'),\n"
        f"            StringStruct('InternalName', '{_PRODUCT}'),\n"
        f"            StringStruct('LegalCopyright', '{_COPYRIGHT}'),\n"
        f"            StringStruct('OriginalFilename', '{_PRODUCT}.exe'),\n"
        f"            StringStruct('ProductName', '{_PRODUCT}'),\n"
        f"            StringStruct('ProductVersion', '{version}')\n"
        "          ]\n"
        "        )\n"
        "      ]\n"
        "    ),\n"
        "    VarFileInfo([VarStruct('Translation', [2052, 1200])])\n"
        "  ]\n"
        ")\n"
    )


def main() -> int:
    version = current_version()
    if "--version" in sys.argv[1:]:
        print(version)
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(version_info_text(version), encoding="utf-8")
    print(f"版本信息已生成：{OUTPUT}（版本 {version}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
