from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cc_cover.discovery import (
    DiscoveryReport,
    discover,
    fingerprint,
    fingerprints_match,
)
from cc_cover.engines import (
    FasterWhisperEngine,
    FunASREngine,
    EngineError,
    extract_audio,
    ffmpeg_version,
    resolve_device,
    resolve_ffmpeg,
)
from cc_cover.formats import normalize_text, render_segments, validate_rendered
from cc_cover.models import (
    Candidate,
    PipelineOptions,
    ProtectedText,
    Segment,
)


ASCII_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[._+#/-][A-Za-z0-9]+)*|\d+(?:\.\d+)?"
)
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_HOTWORDS = (
    "AI",
    "机器学习",
    "深度学习",
    "神经网络",
    "自然语言处理",
    "计算机视觉",
    "Python",
    "PyTorch",
    "TensorFlow",
    "NumPy",
    "Tensor",
    "张量",
    "线性回归",
    "梯度下降",
    "反向传播",
    "损失函数",
    "学习率",
    "优化器",
    "SGD",
    "Adam",
    "CNN",
    "RNN",
    "LSTM",
    "GRU",
    "Transformer",
    "BERT",
    "Word2Vec",
    "Embedding",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "API",
    "GPU",
    "CUDA",
)


class PipelineError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, payload: Any) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"缺少运行产物：{path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"运行产物 JSON 无效：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"运行产物顶层必须是对象：{path}")
    return value


def options_to_dict(options: PipelineOptions) -> dict[str, Any]:
    return {
        "roots": [str(path) for path in options.roots],
        "runs_root": str(options.runs_root),
        "model_cache": str(options.model_cache),
        "device": options.device,
        "compute_type": options.compute_type,
        "ffmpeg": None if options.ffmpeg is None else str(options.ffmpeg),
        "language": options.language,
        "funasr_model": options.funasr_model,
        "funasr_vad_model": options.funasr_vad_model,
        "funasr_punc_model": options.funasr_punc_model,
        "faster_whisper_model": options.faster_whisper_model,
        "hotwords_file": (
            None if options.hotwords_file is None else str(options.hotwords_file)
        ),
        "include_whitespace_only": options.include_whitespace_only,
        "include_missing": options.include_missing,
        "hash_videos": options.hash_videos,
        "pilot_count": options.pilot_count,
    }


def options_from_dict(value: Mapping[str, Any]) -> PipelineOptions:
    return PipelineOptions(
        roots=[Path(str(path)).resolve() for path in value["roots"]],
        runs_root=Path(str(value["runs_root"])).resolve(),
        model_cache=Path(str(value["model_cache"])).resolve(),
        device=str(value.get("device", "auto")),
        compute_type=str(value.get("compute_type", "auto")),
        ffmpeg=(
            None
            if value.get("ffmpeg") in (None, "")
            else Path(str(value["ffmpeg"])).resolve()
        ),
        language=str(value.get("language", "zh")),
        funasr_model=str(value.get("funasr_model", "paraformer-zh")),
        funasr_vad_model=str(value.get("funasr_vad_model", "fsmn-vad")),
        funasr_punc_model=str(value.get("funasr_punc_model", "ct-punc")),
        faster_whisper_model=str(
            value.get("faster_whisper_model", "large-v3-turbo")
        ),
        hotwords_file=(
            None
            if value.get("hotwords_file") in (None, "")
            else Path(str(value["hotwords_file"])).resolve()
        ),
        include_whitespace_only=bool(value.get("include_whitespace_only", False)),
        include_missing=bool(value.get("include_missing", False)),
        hash_videos=bool(value.get("hash_videos", True)),
        pilot_count=int(value.get("pilot_count", 2)),
    )


def load_hotwords(options: PipelineOptions, candidates: Sequence[Candidate]) -> list[str]:
    values = list(DEFAULT_HOTWORDS)
    if options.hotwords_file is not None:
        if not options.hotwords_file.is_file():
            raise PipelineError(f"热词文件不存在：{options.hotwords_file}")
        for line in options.hotwords_file.read_text(encoding="utf-8-sig").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            values.extend(part.strip() for part in text.split(",") if part.strip())
    for candidate in candidates:
        values.extend(ASCII_TOKEN_PATTERN.findall(candidate.video_path.stem))
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique[:120]


def comparison_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def token_set(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text)
    return {item.casefold() for item in ASCII_TOKEN_PATTERN.findall(normalized)}


def overlap_ms(left: Segment, right: Segment) -> int:
    return max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def align_for_audit(
    funasr_segments: Sequence[Segment], faster_segments: Sequence[Segment]
) -> dict[str, Any]:
    alignments: list[dict[str, Any]] = []
    used_funasr: set[int] = set()
    for faster_index, faster in enumerate(faster_segments):
        matched = [
            (index, item)
            for index, item in enumerate(funasr_segments)
            if overlap_ms(item, faster) > 0
        ]
        used_funasr.update(index for index, _item in matched)
        funasr_text = "".join(item.text for _index, item in matched)
        left = comparison_text(funasr_text)
        right = comparison_text(faster.text)
        ratio = SequenceMatcher(None, left, right).ratio() if left and right else 0.0
        mismatch = sorted(token_set(funasr_text).symmetric_difference(token_set(faster.text)))
        alignments.append(
            {
                "faster_whisper_segment_index": faster_index,
                "funasr_segment_indexes": [index for index, _item in matched],
                "start_ms": faster.start_ms,
                "end_ms": faster.end_ms,
                "funasr_text": funasr_text,
                "faster_whisper_text": faster.text,
                "similarity_ratio": round(ratio, 6),
                "ascii_or_numeric_token_mismatch": mismatch,
                "high_risk": not matched or ratio < 0.60 or bool(mismatch),
                "decision": "keep_funasr_writeback_review_faster_whisper_only",
            }
        )
    for index, segment in enumerate(funasr_segments):
        if index in used_funasr:
            continue
        alignments.append(
            {
                "faster_whisper_segment_index": None,
                "funasr_segment_indexes": [index],
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "funasr_text": segment.text,
                "faster_whisper_text": "",
                "similarity_ratio": 0.0,
                "ascii_or_numeric_token_mismatch": sorted(token_set(segment.text)),
                "high_risk": True,
                "decision": "keep_funasr_writeback_missing_faster_whisper_overlap",
            }
        )
    alignments.sort(key=lambda item: (item["start_ms"], item["end_ms"]))
    ratios = [float(item["similarity_ratio"]) for item in alignments]
    return {
        "writeback_source_engine": "funasr",
        "faster_whisper_role": "second_candidate_and_conflict_audit_only",
        "alignment_count": len(alignments),
        "high_risk_count": sum(bool(item["high_risk"]) for item in alignments),
        "median_similarity_ratio": statistics.median(ratios) if ratios else 0.0,
        "alignments": alignments,
    }


def longest_duplicate_run(segments: Sequence[Segment]) -> int:
    longest = 1
    current = 1
    previous = ""
    for segment in segments:
        normalized = comparison_text(segment.text)
        if normalized and normalized == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
        previous = normalized
    return longest


def validate_segments(segments: Sequence[Segment], duration_seconds: float) -> None:
    if not segments:
        raise PipelineError("引擎字幕段为空")
    previous_start = -1
    maximum_end = math.ceil(duration_seconds * 1000.0) + 5000
    for index, segment in enumerate(segments):
        if (
            segment.start_ms < 0
            or segment.end_ms <= segment.start_ms
            or segment.start_ms < previous_start
            or segment.end_ms > maximum_end
            or not segment.text.strip()
        ):
            raise PipelineError(f"引擎字幕段无效：#{index}")
        previous_start = segment.start_ms


def validate_protected(protected: Sequence[ProtectedText]) -> None:
    failures: list[str] = []
    for item in protected:
        actual = fingerprint(item.path, include_hash=True)
        if not fingerprints_match(actual, item.fingerprint):
            failures.append(str(item.path))
    if failures:
        raise PipelineError("受保护的非空 TXT 发生变化：\n" + "\n".join(failures))


def validate_candidates(candidates: Sequence[Candidate], require_initial_target: bool) -> None:
    failures: list[str] = []
    for candidate in candidates:
        current_video = fingerprint(
            candidate.video_path,
            include_hash=candidate.video_fingerprint.sha256 is not None,
        )
        if not fingerprints_match(current_video, candidate.video_fingerprint):
            failures.append(f"视频变化：{candidate.video_path}")
        if require_initial_target:
            current_target = fingerprint(candidate.target_path, include_hash=True)
            if not fingerprints_match(current_target, candidate.target_fingerprint):
                failures.append(f"目标 TXT 状态变化：{candidate.target_path}")
    if failures:
        raise PipelineError("候选快照校验失败：\n" + "\n".join(failures))


class SubtitlePipeline:
    def __init__(
        self,
        options: PipelineOptions,
        run_dir: Path,
        candidates: Sequence[Candidate],
        protected: Sequence[ProtectedText],
        manifest: dict[str, Any],
    ):
        self.options = options
        self.run_dir = run_dir.resolve()
        self.candidates = list(candidates)
        self.protected = list(protected)
        self.manifest = manifest
        self.ffmpeg = resolve_ffmpeg(options.ffmpeg)
        self.device, self.compute_type = resolve_device(
            options.device, options.compute_type
        )
        self.hotwords = load_hotwords(options, self.candidates)

    @classmethod
    def create(cls, options: PipelineOptions, report: DiscoveryReport) -> "SubtitlePipeline":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise PipelineError(f"生成的 run_id 不安全：{run_id}")
        run_dir = options.runs_root.expanduser().resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        pilot_count = max(0, min(options.pilot_count, len(report.candidates)))
        pilot = [item.sample_id for item in report.candidates[:pilot_count]]
        remaining = [
            item.sample_id for item in report.candidates if item.sample_id not in set(pilot)
        ]
        manifest = {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "prepared",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "options": options_to_dict(options),
            "discovery": {
                "roots": [str(path) for path in report.roots],
                "video_count": report.video_count,
                "matched_text_count": report.matched_text_count,
                "missing_text_count": report.missing_text_count,
                "nonempty_format_samples": report.nonempty_format_samples,
                "candidate_count": len(report.candidates),
                "protected_nonempty_txt_count": len(report.protected_texts),
            },
            "phases": {
                "pilot": pilot,
                "remaining": remaining,
                "all": [item.sample_id for item in report.candidates],
            },
            "candidates": [item.to_dict() for item in report.candidates],
            "protected_nonempty_txt": [item.to_dict() for item in report.protected_texts],
            "runtime": None,
            "stage": None,
            "commit": None,
        }
        write_json_atomic(run_dir / "manifest.json", manifest)
        return cls(options, run_dir, report.candidates, report.protected_texts, manifest)

    @classmethod
    def resume(cls, run_dir: Path) -> "SubtitlePipeline":
        resolved = run_dir.expanduser().resolve()
        manifest = load_json(resolved / "manifest.json")
        options = options_from_dict(manifest["options"])
        candidates = [Candidate.from_dict(item) for item in manifest["candidates"]]
        protected = [
            ProtectedText.from_dict(item)
            for item in manifest["protected_nonempty_txt"]
        ]
        return cls(options, resolved, candidates, protected, manifest)

    def update_manifest(self, **changes: Any) -> None:
        self.manifest.update(changes)
        self.manifest["updated_at_utc"] = utc_now()
        write_json_atomic(self.run_dir / "manifest.json", self.manifest)

    def engine_output(self, engine: str, sample_id: str) -> Path:
        return self.run_dir / "engines" / engine / f"{sample_id}.json"

    def load_engine_output(self, engine: str, candidate: Candidate) -> dict[str, Any]:
        payload = load_json(self.engine_output(engine, candidate.sample_id))
        if payload.get("sample_id") != candidate.sample_id:
            raise PipelineError(f"{engine} sample_id 不匹配")
        if Path(str(payload.get("source_path", ""))).resolve() != candidate.video_path:
            raise PipelineError(f"{engine} source_path 不匹配")
        if payload.get("engine") != engine:
            raise PipelineError(f"{engine} 引擎声明不匹配")
        segments = [Segment.from_dict(item) for item in payload.get("segments", [])]
        validate_segments(segments, float(payload["duration_seconds"]))
        return payload

    def output_complete(self, engine: str, candidate: Candidate) -> bool:
        try:
            self.load_engine_output(engine, candidate)
            return True
        except Exception:
            return False

    def run_engine(self, engine_name: str, sample_ids: Sequence[str]) -> None:
        selected = [
            candidate for candidate in self.candidates if candidate.sample_id in sample_ids
        ]
        pending = [
            candidate
            for candidate in selected
            if not self.output_complete(engine_name, candidate)
        ]
        if not pending:
            return
        if engine_name == "funasr":
            engine: Any = FunASREngine(self.options, self.device)
        else:
            engine = FasterWhisperEngine(
                self.options, self.device, self.compute_type
            )
        print(f"加载 {engine_name}：device={self.device}", flush=True)
        engine.load()
        try:
            for index, candidate in enumerate(pending, start=1):
                print(
                    f"[{engine_name} {index}/{len(pending)}] {candidate.video_path}",
                    flush=True,
                )
                before = fingerprint(
                    candidate.video_path,
                    include_hash=candidate.video_fingerprint.sha256 is not None,
                )
                if not fingerprints_match(before, candidate.video_fingerprint):
                    raise PipelineError(f"视频在转写前发生变化：{candidate.video_path}")
                wav_path = self.run_dir / "work" / f"{candidate.sample_id}.wav"
                started = time.perf_counter()
                try:
                    duration = extract_audio(self.ffmpeg, candidate.video_path, wav_path)
                    segments, metadata = engine.transcribe(
                        wav_path, duration, self.hotwords
                    )
                finally:
                    if wav_path.exists():
                        wav_path.unlink()
                after = fingerprint(
                    candidate.video_path,
                    include_hash=candidate.video_fingerprint.sha256 is not None,
                )
                if not fingerprints_match(after, candidate.video_fingerprint):
                    raise PipelineError(f"视频在转写后发生变化：{candidate.video_path}")
                validate_segments(segments, duration)
                write_json_atomic(
                    self.engine_output(engine_name, candidate.sample_id),
                    {
                        "schema_version": "1.0",
                        "sample_id": candidate.sample_id,
                        "source_path": str(candidate.video_path),
                        "engine": engine_name,
                        "duration_seconds": round(duration, 6),
                        "elapsed_total_seconds": round(time.perf_counter() - started, 6),
                        "metadata": metadata,
                        "segments": [segment.to_dict() for segment in segments],
                    },
                )
        finally:
            engine.close()

    def quality_report(
        self,
        candidate: Candidate,
        caption_payload: bytes,
        funasr_segments: Sequence[Segment],
        faster_segments: Sequence[Segment],
        duration_seconds: float,
    ) -> dict[str, Any]:
        format_metrics = validate_rendered(caption_payload, candidate.profile)
        funasr_text = "".join(
            normalize_text(item.text, candidate.profile.strip_sentence_punctuation)
            for item in funasr_segments
        )
        faster_text = "".join(item.text for item in faster_segments)
        normalized_funasr = comparison_text(funasr_text)
        normalized_faster = comparison_text(faster_text)
        minutes = duration_seconds / 60.0
        density = len(normalized_funasr) / minutes
        length_ratio = (
            len(normalized_funasr) / len(normalized_faster)
            if normalized_faster
            else math.inf
        )
        duplicate_run = longest_duplicate_run(funasr_segments)
        minimum_segments = max(3, int(minutes * 5))
        errors: list[str] = []
        warnings: list[str] = []
        if len(funasr_segments) < minimum_segments:
            errors.append(
                f"FunASR 字幕段数过少：{len(funasr_segments)} < {minimum_segments}"
            )
        if density < 30 or density > 1200:
            errors.append(f"文本密度异常：{density:.1f} chars/min")
        if duplicate_run > 4:
            errors.append(f"连续重复字幕过多：{duplicate_run}")
        if not math.isfinite(length_ratio) or length_ratio < 0.45 or length_ratio > 2.20:
            errors.append(f"双模型全文长度比异常：{length_ratio:.3f}")
        if candidate.profile.style == "timed":
            median_chars = float(format_metrics["median_text_chars"])
            if median_chars < 3 or median_chars > 40:
                warnings.append(f"中位段长偏离常见范围：{median_chars:g} 字")
        alignment = align_for_audit(funasr_segments, faster_segments)
        return {
            "sample_id": candidate.sample_id,
            "video_path": str(candidate.video_path),
            "target_path": str(candidate.target_path),
            "duration_seconds": round(duration_seconds, 6),
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
            "profile": candidate.profile.to_dict(),
            "format_metrics": format_metrics,
            "funasr_segment_count": len(funasr_segments),
            "faster_whisper_segment_count": len(faster_segments),
            "text_density_chars_per_minute": round(density, 3),
            "funasr_to_faster_whisper_text_length_ratio": round(length_ratio, 6),
            "longest_consecutive_duplicate_run": duplicate_run,
            "caption_sha256": hashlib.sha256(caption_payload).hexdigest(),
            "caption_size": len(caption_payload),
            "alignment_summary": {
                key: value for key, value in alignment.items() if key != "alignments"
            },
            "alignment": alignment,
        }

    def stage(self, sample_ids: Sequence[str]) -> dict[str, Any]:
        validate_protected(self.protected)
        validate_candidates(self.candidates, require_initial_target=True)
        selected = [
            candidate for candidate in self.candidates if candidate.sample_id in sample_ids
        ]
        reports: list[dict[str, Any]] = []
        for candidate in selected:
            funasr_payload = self.load_engine_output("funasr", candidate)
            faster_payload = self.load_engine_output("faster_whisper", candidate)
            funasr_segments = [
                Segment.from_dict(item) for item in funasr_payload["segments"]
            ]
            faster_segments = [
                Segment.from_dict(item) for item in faster_payload["segments"]
            ]
            duration = float(funasr_payload["duration_seconds"])
            if abs(duration - float(faster_payload["duration_seconds"])) > 0.05:
                raise PipelineError(f"双模型音频时长不一致：{candidate.sample_id}")
            caption_payload = render_segments(funasr_segments, candidate.profile)
            report = self.quality_report(
                candidate,
                caption_payload,
                funasr_segments,
                faster_segments,
                duration,
            )
            write_bytes_atomic(
                self.run_dir / "prepared" / f"{candidate.sample_id}.txt",
                caption_payload,
            )
            write_json_atomic(
                self.run_dir / "audit" / f"{candidate.sample_id}.json",
                report,
            )
            reports.append(
                {key: value for key, value in report.items() if key != "alignment"}
            )
        stage_path = self.run_dir / "stage_report.json"
        previous: dict[str, Any] = {}
        if stage_path.is_file():
            previous = load_json(stage_path)
        merged = {
            str(item["sample_id"]): item
            for item in previous.get("samples", [])
            if isinstance(item, dict) and item.get("sample_id")
        }
        for report in reports:
            merged[str(report["sample_id"])] = report
        all_ids = list(self.manifest["phases"]["all"])
        staged_all = all(sample_id in merged for sample_id in all_ids)
        all_passed = staged_all and all(
            bool(merged[sample_id]["passed"]) for sample_id in all_ids
        )
        selected_passed = all(bool(merged[sample_id]["passed"]) for sample_id in sample_ids)
        stage_report = {
            "schema_version": "1.0",
            "run_id": self.manifest["run_id"],
            "updated_at_utc": utc_now(),
            "selected_ids": list(sample_ids),
            "selected_passed": selected_passed,
            "staged_ids": sorted(merged),
            "staged_all": staged_all,
            "all_passed": all_passed,
            "samples": [merged[sample_id] for sample_id in sorted(merged)],
        }
        write_json_atomic(stage_path, stage_report)
        self.update_manifest(
            status="staged_all" if staged_all else "staged_partial",
            stage={
                "report": str(stage_path),
                "staged_all": staged_all,
                "all_passed": all_passed,
            },
        )
        if not selected_passed:
            raise PipelineError("试样或全量质量门禁未通过，未写回课程目录")
        return stage_report

    def commit(self) -> dict[str, Any]:
        stage_report = load_json(self.run_dir / "stage_report.json")
        if not stage_report.get("staged_all") or not stage_report.get("all_passed"):
            raise PipelineError("全部字幕尚未通过质量门禁")
        reports = {
            str(item["sample_id"]): item for item in stage_report["samples"]
        }
        validate_protected(self.protected)
        validate_candidates(self.candidates, require_initial_target=True)
        backups = self.run_dir / "backups"
        payloads: dict[str, bytes] = {}
        for candidate in self.candidates:
            prepared = self.run_dir / "prepared" / f"{candidate.sample_id}.txt"
            payload = prepared.read_bytes()
            validate_rendered(payload, candidate.profile)
            expected_hash = str(reports[candidate.sample_id]["caption_sha256"])
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise PipelineError(f"暂存字幕哈希不匹配：{prepared}")
            payloads[candidate.sample_id] = payload
            backup_dir = backups / candidate.sample_id
            original = candidate.target_path.read_bytes() if candidate.target_path.exists() else b""
            write_bytes_atomic(backup_dir / "original.txt", original)
            write_json_atomic(
                backup_dir / "state.json",
                {
                    "target_path": str(candidate.target_path),
                    "initial_state": candidate.initial_state,
                    "fingerprint": candidate.target_fingerprint.to_dict(),
                },
            )
        committed: list[Candidate] = []
        try:
            for candidate in self.candidates:
                write_bytes_atomic(candidate.target_path, payloads[candidate.sample_id])
                committed.append(candidate)
            for candidate in self.candidates:
                actual = candidate.target_path.read_bytes()
                if actual != payloads[candidate.sample_id]:
                    raise PipelineError(f"写回后内容不一致：{candidate.target_path}")
                validate_rendered(actual, candidate.profile)
            validate_protected(self.protected)
        except Exception:
            for candidate in reversed(committed):
                backup = backups / candidate.sample_id / "original.txt"
                if candidate.initial_state == "missing":
                    if candidate.target_path.exists():
                        candidate.target_path.unlink()
                else:
                    write_bytes_atomic(candidate.target_path, backup.read_bytes())
            raise
        entries = [
            {
                "sample_id": candidate.sample_id,
                "video_path": str(candidate.video_path),
                "target_path": str(candidate.target_path),
                "target_size": candidate.target_path.stat().st_size,
                "target_sha256": hashlib.sha256(
                    candidate.target_path.read_bytes()
                ).hexdigest(),
            }
            for candidate in self.candidates
        ]
        report = {
            "schema_version": "1.0",
            "run_id": self.manifest["run_id"],
            "status": "committed",
            "committed_at_utc": utc_now(),
            "entry_count": len(entries),
            "entries": entries,
        }
        write_json_atomic(self.run_dir / "commit_report.json", report)
        self.update_manifest(status="committed", commit=report)
        return report

    def verify(self) -> dict[str, Any]:
        validate_protected(self.protected)
        failures: list[str] = []
        entries: list[dict[str, Any]] = []
        for candidate in self.candidates:
            current_video = fingerprint(
                candidate.video_path,
                include_hash=candidate.video_fingerprint.sha256 is not None,
            )
            if not fingerprints_match(current_video, candidate.video_fingerprint):
                failures.append(f"视频变化：{candidate.video_path}")
                continue
            prepared = self.run_dir / "prepared" / f"{candidate.sample_id}.txt"
            if not candidate.target_path.is_file() or candidate.target_path.stat().st_size == 0:
                failures.append(f"目标字幕为空：{candidate.target_path}")
                continue
            actual = candidate.target_path.read_bytes()
            if actual != prepared.read_bytes():
                failures.append(f"目标字幕与暂存产物不一致：{candidate.target_path}")
                continue
            metrics = validate_rendered(actual, candidate.profile)
            entries.append(
                {
                    "sample_id": candidate.sample_id,
                    "target_path": str(candidate.target_path),
                    "size": len(actual),
                    "sha256": hashlib.sha256(actual).hexdigest(),
                    "metrics": metrics,
                }
            )
        report = {
            "schema_version": "1.0",
            "run_id": self.manifest["run_id"],
            "verified_at_utc": utc_now(),
            "passed": not failures,
            "failures": failures,
            "verified_count": len(entries),
            "entries": entries,
        }
        write_json_atomic(self.run_dir / "verification.json", report)
        if failures:
            raise PipelineError("最终复核失败：\n" + "\n".join(failures))
        return report

    def execute(self) -> Path:
        validate_protected(self.protected)
        validate_candidates(self.candidates, require_initial_target=True)
        self.update_manifest(
            status="running",
            runtime={
                "device": self.device,
                "compute_type": self.compute_type,
                "ffmpeg": str(self.ffmpeg),
                "ffmpeg_version": ffmpeg_version(self.ffmpeg),
                "hotword_count": len(self.hotwords),
            },
        )
        pilot = list(self.manifest["phases"]["pilot"])
        remaining = list(self.manifest["phases"]["remaining"])
        if pilot:
            self.run_engine("funasr", pilot)
            self.run_engine("faster_whisper", pilot)
            self.stage(pilot)
        if remaining:
            self.run_engine("funasr", remaining)
            self.run_engine("faster_whisper", remaining)
            self.stage(list(self.manifest["phases"]["all"]))
        elif pilot:
            self.stage(pilot)
        self.commit()
        self.verify()
        return self.run_dir


def discover_for_options(options: PipelineOptions) -> DiscoveryReport:
    return discover(
        options.roots,
        include_whitespace_only=options.include_whitespace_only,
        include_missing=options.include_missing,
        hash_videos=options.hash_videos,
    )
