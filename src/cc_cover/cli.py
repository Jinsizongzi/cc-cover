from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from cc_cover import __version__
from cc_cover.discovery import DiscoveryError, DiscoveryReport
from cc_cover.engines import EngineError
from cc_cover.formats import FormatError
from cc_cover.models import PipelineOptions
from cc_cover.pipeline import (
    PipelineError,
    SubtitlePipeline,
    discover_for_options,
)


class ConfigError(RuntimeError):
    pass


DEFAULTS: dict[str, Any] = {
    "runs_root": "runs",
    "model_cache": "model-cache",
    "device": "auto",
    "compute_type": "auto",
    "ffmpeg": None,
    "language": "zh",
    "funasr_model": "paraformer-zh",
    "funasr_vad_model": "fsmn-vad",
    "funasr_punc_model": "ct-punc",
    "faster_whisper_model": "large-v3-turbo",
    "hotwords_file": None,
    "include_whitespace_only": False,
    "include_missing": False,
    "hash_videos": True,
    "pilot_count": 2,
}

CONFIG_KEYS = frozenset(DEFAULTS)
PATH_KEYS = frozenset({"runs_root", "model_cache", "ffmpeg", "hotwords_file"})


def load_config(path: Path | None) -> tuple[dict[str, Any], Path]:
    if path is None:
        return {}, Path.cwd()
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ConfigError(f"配置文件不存在：{resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置文件 JSON 无效：{resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("配置文件顶层必须是 JSON 对象")
    unknown = sorted(set(value) - CONFIG_KEYS)
    if unknown:
        raise ConfigError("配置文件包含未知字段：" + ", ".join(unknown))
    return dict(value), resolved.parent


def resolve_path(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def option_value(
    arguments: argparse.Namespace,
    config: Mapping[str, Any],
    config_base: Path,
    name: str,
) -> tuple[Any, Path]:
    command_value = getattr(arguments, name, None)
    if command_value is not None:
        return command_value, Path.cwd()
    if name in config:
        return config[name], config_base
    return DEFAULTS[name], Path.cwd()


def build_options(
    arguments: argparse.Namespace,
    config: Mapping[str, Any],
    config_base: Path,
) -> PipelineOptions:
    root_values: Any = arguments.roots
    roots_base = Path.cwd()
    if isinstance(root_values, (str, Path)):
        root_values = [root_values]
    if not isinstance(root_values, list) or not root_values:
        raise ConfigError("请提供至少一个扫描目录")
    roots = [resolve_path(value, roots_base) for value in root_values]
    if any(path is None for path in roots):
        raise ConfigError("roots 不能包含空路径")

    values: dict[str, Any] = {}
    for name in DEFAULTS:
        value, base = option_value(arguments, config, config_base, name)
        values[name] = resolve_path(value, base) if name in PATH_KEYS else value
    return PipelineOptions(
        roots=[path for path in roots if path is not None],
        runs_root=values["runs_root"],
        model_cache=values["model_cache"],
        device=str(values["device"]),
        compute_type=str(values["compute_type"]),
        ffmpeg=values["ffmpeg"],
        language=str(values["language"]),
        funasr_model=str(values["funasr_model"]),
        funasr_vad_model=str(values["funasr_vad_model"]),
        funasr_punc_model=str(values["funasr_punc_model"]),
        faster_whisper_model=str(values["faster_whisper_model"]),
        hotwords_file=values["hotwords_file"],
        include_whitespace_only=bool(values["include_whitespace_only"]),
        include_missing=bool(values["include_missing"]),
        hash_videos=bool(values["hash_videos"]),
        pilot_count=int(values["pilot_count"]),
    )


def report_payload(report: DiscoveryReport) -> dict[str, Any]:
    return {
        "roots": [str(path) for path in report.roots],
        "video_count": report.video_count,
        "matched_text_count": report.matched_text_count,
        "missing_text_count": report.missing_text_count,
        "nonempty_format_samples": report.nonempty_format_samples,
        "candidate_count": len(report.candidates),
        "protected_nonempty_txt_count": len(report.protected_texts),
        "candidates": [
            {
                "sample_id": item.sample_id,
                "state": item.initial_state,
                "video_path": str(item.video_path),
                "target_path": str(item.target_path),
                "format": item.profile.to_dict(),
            }
            for item in report.candidates
        ],
    }


def print_report(report: DiscoveryReport) -> None:
    print(f"视频文件：{report.video_count}")
    print(f"同名 TXT：{report.matched_text_count}")
    print(f"缺失 TXT：{report.missing_text_count}")
    print(f"可用格式样本：{report.nonempty_format_samples}")
    print(f"待补全字幕：{len(report.candidates)}")
    print(f"受保护非空 TXT：{len(report.protected_texts)}")
    for candidate in report.candidates:
        print(f"  [{candidate.sample_id}] {candidate.target_path} ({candidate.initial_state})")


def add_discovery_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("roots", nargs="+", help="递归扫描的视频目录，必须显式提供")
    parser.add_argument("--config", type=Path, help="JSON 配置文件")
    parser.add_argument(
        "--include-whitespace-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="同时处理只有空白字符的 TXT",
    )
    parser.add_argument(
        "--include-missing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="同时为没有同名 TXT 的视频创建字幕",
    )
    parser.add_argument(
        "--hash-videos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="扫描时计算视频 SHA-256，默认启用",
    )


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-root", type=Path, help="运行产物目录")
    parser.add_argument("--model-cache", type=Path, help="模型缓存目录")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--compute-type", help="CTranslate2 计算类型")
    parser.add_argument("--ffmpeg", type=Path, help="FFmpeg 可执行文件")
    parser.add_argument("--language", help="faster-whisper 语言代码")
    parser.add_argument("--funasr-model", help="FunASR 主模型名称或路径")
    parser.add_argument("--funasr-vad-model", help="FunASR VAD 模型名称或路径")
    parser.add_argument("--funasr-punc-model", help="FunASR 标点模型名称或路径")
    parser.add_argument(
        "--faster-whisper-model",
        help="faster-whisper 模型名称或路径",
    )
    parser.add_argument("--hotwords-file", type=Path, help="每行一个热词的 UTF-8 文件")
    parser.add_argument("--pilot-count", type=int, help="先行质量门禁的视频数量")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cc-cover",
        description="使用 FunASR 与 faster-whisper 安全补全空字幕 TXT。",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="只扫描并显示候选文件")
    add_discovery_arguments(scan)
    scan.add_argument("--json", action="store_true", help="输出机器可读 JSON")

    transcribe = commands.add_parser("transcribe", help="生成、校验并写回字幕")
    add_discovery_arguments(transcribe)
    add_pipeline_arguments(transcribe)

    resume = commands.add_parser("resume", help="继续已有运行")
    resume.add_argument("run_dir", type=Path, help="包含 manifest.json 的运行目录")

    verify = commands.add_parser("verify", help="复核已写回的运行")
    verify.add_argument("run_dir", type=Path, help="包含 manifest.json 的运行目录")
    return parser


def command_scan(arguments: argparse.Namespace) -> int:
    config, config_base = load_config(arguments.config)
    options = build_options(arguments, config, config_base)
    report = discover_for_options(options)
    if arguments.json:
        print(json.dumps(report_payload(report), ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0


def command_transcribe(arguments: argparse.Namespace) -> int:
    config, config_base = load_config(arguments.config)
    options = build_options(arguments, config, config_base)
    report = discover_for_options(options)
    print_report(report)
    if not report.candidates:
        print("没有符合条件的空字幕 TXT，无需处理。")
        return 0
    pipeline = SubtitlePipeline.create(options, report)
    print(f"运行目录：{pipeline.run_dir}")
    pipeline.execute()
    print(f"字幕已写回并复核通过：{pipeline.run_dir}")
    return 0


def command_resume(arguments: argparse.Namespace) -> int:
    pipeline = SubtitlePipeline.resume(arguments.run_dir)
    if pipeline.manifest.get("status") == "committed":
        report = pipeline.verify()
        print(f"复核通过，共 {report['verified_count']} 个字幕文件。")
        return 0
    pipeline.execute()
    print(f"字幕已写回并复核通过：{pipeline.run_dir}")
    return 0


def command_verify(arguments: argparse.Namespace) -> int:
    pipeline = SubtitlePipeline.resume(arguments.run_dir)
    report = pipeline.verify()
    print(f"复核通过，共 {report['verified_count']} 个字幕文件。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = create_parser().parse_args(argv)
    try:
        if arguments.command == "scan":
            return command_scan(arguments)
        if arguments.command == "transcribe":
            return command_transcribe(arguments)
        if arguments.command == "resume":
            return command_resume(arguments)
        if arguments.command == "verify":
            return command_verify(arguments)
        raise ConfigError(f"未知命令：{arguments.command}")
    except (ConfigError, DiscoveryError, EngineError, FormatError, PipelineError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。运行产物可通过 resume 继续。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
