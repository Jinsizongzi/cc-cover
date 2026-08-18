from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SETTINGS_FILENAME = "settings.json"


class SettingsError(RuntimeError):
    """设置文件无效（JSON 损坏或顶层不是对象）时抛出。"""


DEVICE_CHOICES = ("auto", "cuda", "cpu")
GUI_DEVICE_CHOICES = ("cuda", "cpu")


@dataclass(frozen=True)
class GuiSettings:
    """GUI 用户偏好，持久化到数据根下的 settings.json。

    device 是唯一的运行设备设置；旧版 accelerator 键在读取时迁移。
    """

    scan_path: str = ""
    device: str = "auto"
    ffmpeg: str = ""
    hash_videos: bool = True


def resolve_default_device(saved: str, detected: str | None) -> str:
    """启动默认设备：已保存的明确选择优先；否则跟随检测结果，回退 CPU。"""
    if saved in GUI_DEVICE_CHOICES:
        return saved
    return detected if detected in GUI_DEVICE_CHOICES else "cpu"


def settings_file(data_root: Path) -> Path:
    """数据根内固定位置：<data_root>/settings.json。"""
    return Path(data_root).expanduser().resolve() / SETTINGS_FILENAME


def read_settings(data_root: Path) -> dict[str, Any]:
    """读取数据根下的设置文件；文件不存在时返回空字典。"""
    path = settings_file(data_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, NotADirectoryError):
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise SettingsError(f"设置文件无效：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SettingsError(f"设置文件顶层必须是 JSON 对象：{path}")
    return value


def write_settings(data_root: Path, values: Mapping[str, Any]) -> None:
    """原子写入数据根下的设置文件（UTF-8）。"""
    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = settings_file(root)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(values), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_gui_settings(data_root: Path) -> GuiSettings:
    """读取数据根下 settings.json 中的 GUI 偏好；缺失或非法字段回退默认值。"""
    values = read_settings(data_root)
    scan_path = values.get("scan_path")
    device = values.get("device")
    accelerator = values.get("accelerator")
    if accelerator in GUI_DEVICE_CHOICES:
        # 旧版 accelerator 是“运行环境”选择（NVIDIA GPU / CPU），合并后优先迁移。
        device = accelerator
    elif device not in DEVICE_CHOICES:
        device = "auto"
    ffmpeg = values.get("ffmpeg")
    hash_videos = values.get("hash_videos")
    return GuiSettings(
        scan_path=str(scan_path) if isinstance(scan_path, str) else "",
        device=device,
        ffmpeg=str(ffmpeg) if isinstance(ffmpeg, str) else "",
        hash_videos=hash_videos if isinstance(hash_videos, bool) else True,
    )


def save_gui_settings(data_root: Path, settings: GuiSettings) -> None:
    """合并写入 GUI 偏好，移除旧版 accelerator 键，保留 data_root 等既有键。"""
    values = read_settings(data_root)
    values.pop("accelerator", None)
    values.update(
        {
            "scan_path": settings.scan_path,
            "device": settings.device,
            "ffmpeg": settings.ffmpeg,
            "hash_videos": settings.hash_videos,
        }
    )
    write_settings(data_root, values)
