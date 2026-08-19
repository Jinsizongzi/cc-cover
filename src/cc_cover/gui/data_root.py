from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from cc_cover.gui.settings import read_settings, settings_file, write_settings


APP_DATA_DIRECTORY = "CC-Cover"
# 数据根固定子目录：与 README/安装器卸载清单保持一致，改动需同步
# packaging/CC-Cover.iss 的 [UninstallDelete] 与 tests/test_packaging.py。
DATA_ROOT_SUBDIRECTORIES = ("venv", "model-cache", "runs", "temp")
DATA_ROOT_KEY = "data_root"


@dataclass(frozen=True)
class RuntimePaths:
    source_root: Path
    data_root: Path
    venv_root: Path
    venv_python: Path
    model_cache: Path
    runs_root: Path
    temp_root: Path


@dataclass(frozen=True)
class DataRootResolution:
    root: Path
    needs_choice: bool


def default_data_root(
    *,
    frozen: bool | None = None,
    bundle_root: Path | None = None,
    app_dir: Path | None = None,
) -> Path:
    """数据根默认位置：打包后为 exe 所在目录，开发模式为项目根目录。"""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if app_dir is not None:
        return Path(app_dir).expanduser().resolve()
    if is_frozen:
        return Path(sys.executable).resolve().parent
    if bundle_root is None:
        bundle_root = Path(__file__).resolve().parents[2]
    return Path(bundle_root).resolve()


def runtime_paths(
    *,
    frozen: bool | None = None,
    bundle_root: Path | None = None,
    data_root: Path | None = None,
) -> RuntimePaths:
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if bundle_root is None:
        if is_frozen:
            bundle_root = Path(str(getattr(sys, "_MEIPASS"))).resolve()
        else:
            bundle_root = Path(__file__).resolve().parents[2]
    source_root = (
        bundle_root / "src" if is_frozen else bundle_root.resolve() / "src"
    ).resolve()
    if data_root is None:
        data_root = default_data_root(frozen=is_frozen, bundle_root=bundle_root)
    resolved_data_root = Path(data_root).expanduser().resolve()
    venv_root = resolved_data_root / "venv"
    return RuntimePaths(
        source_root=source_root,
        data_root=resolved_data_root,
        venv_root=venv_root,
        venv_python=venv_root / "Scripts" / "python.exe",
        model_cache=resolved_data_root / "model-cache",
        runs_root=resolved_data_root / "runs",
        temp_root=resolved_data_root / "temp",
    )


def ensure_data_root(paths: RuntimePaths) -> None:
    """创建数据根与固定子目录：venv、model-cache、runs、temp。"""
    paths.data_root.mkdir(parents=True, exist_ok=True)
    for name in DATA_ROOT_SUBDIRECTORIES:
        (paths.data_root / name).mkdir(parents=True, exist_ok=True)


def is_writable(directory: Path) -> bool:
    """探测目录可写性：实际创建并删除一个探测文件，失败返回 False。"""
    target = Path(directory).expanduser().resolve()
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / f".cc-cover-write-probe-{uuid.uuid4().hex}"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def fallback_pointer_root() -> Path:
    """默认根不可写时，数据根指针的回退位置（用户本地应用数据目录）。"""
    environment = os.environ.get("LOCALAPPDATA")
    local = Path(environment) if environment else Path.home() / "AppData" / "Local"
    return (local / APP_DATA_DIRECTORY).resolve()


def pointer_root(default_root: Path) -> Path:
    """数据根指针所在目录：默认根可写时用默认根，否则用用户可写的回退位置。"""
    default_root = Path(default_root).expanduser().resolve()
    if settings_file(default_root).exists() or is_writable(default_root):
        return default_root
    return fallback_pointer_root()


def configured_data_root(default_root: Path) -> Path:
    """从数据根指针读取自定义数据根；未配置或非绝对路径时返回默认根。"""
    default_root = Path(default_root).expanduser().resolve()
    value = read_settings(pointer_root(default_root)).get(DATA_ROOT_KEY)
    if value in (None, ""):
        return default_root
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return default_root
    return path.resolve()


def resolve_data_root(default_root: Path) -> DataRootResolution:
    """解析启动时使用的数据根；默认根不可写且未配置自定义根时引导选择。"""
    default_root = Path(default_root).expanduser().resolve()
    configured = configured_data_root(default_root)
    if configured != default_root:
        return DataRootResolution(
            root=configured, needs_choice=not is_writable(configured)
        )
    return DataRootResolution(
        root=default_root, needs_choice=not is_writable(default_root)
    )


def apply_data_root(
    default_root: Path,
    previous_root: Path,
    new_root: Path,
) -> Path:
    """切换数据根：设置文件复制到新根并更新指针，旧文件保留不删除。

    换回默认根时只清除指针；环境与模型缓存不做自动迁移。
    """
    default_root = Path(default_root).expanduser().resolve()
    previous_root = Path(previous_root).expanduser().resolve()
    new_root = Path(new_root).expanduser().resolve()
    if new_root == default_root:
        pointer = pointer_root(default_root)
        values = read_settings(pointer)
        values.pop(DATA_ROOT_KEY, None)
        write_settings(pointer, values)
        return default_root
    values = dict(read_settings(previous_root))
    values[DATA_ROOT_KEY] = str(new_root)
    write_settings(new_root, values)
    pointer = pointer_root(default_root)
    pointer_values = read_settings(pointer)
    pointer_values[DATA_ROOT_KEY] = str(new_root)
    write_settings(pointer, pointer_values)
    return new_root
