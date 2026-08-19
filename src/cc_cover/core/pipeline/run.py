from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from cc_cover.core.discovery import (
    DiscoveryReport,
    discover,
    fingerprint,
    fingerprints_match,
    fingerprints_match_quick,
)
from cc_cover.core.engines import (
    EngineError,
    FasterWhisperEngine,
    FunASREngine,
    extract_audio,
    ffmpeg_version,
    resolve_device,
    resolve_ffmpeg,
)
from cc_cover.core.formats import (
    FormatError,
    normalize_text,
    render_segments,
    validate_rendered,
)
from cc_cover.core.models import (
    Candidate,
    CandidateFailedEvent,
    ENGINE_NAMES,
    EngineStartEvent,
    Phase,
    PipelineOptions,
    ProgressEvent,
    ProtectedText,
    Segment,
)
from cc_cover.core.pipeline.audit import (
    align_for_audit,
    comparison_text,
    faster_whisper_confidence,
    longest_duplicate_run,
)
from cc_cover.core.pipeline.errors import PipelineError
from cc_cover.core.pipeline.hotwords import load_hotwords
from cc_cover.core.pipeline.io import (
    emit_event,
    load_json,
    utc_now,
    write_bytes_atomic,
    write_json_atomic,
)
from cc_cover.core.pipeline.options import options_from_dict, options_to_dict
from cc_cover.core.pipeline.summary import write_summary
from cc_cover.core.pipeline.validate import (
    engine_phase,
    validate_candidates,
    validate_protected,
    validate_segments,
)

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

WARNING_DENSITY_MIN = 100.0
WARNING_DENSITY_MAX = 600.0
WARNING_LENGTH_RATIO_MIN = 0.80
WARNING_LENGTH_RATIO_MAX = 1.40
WARNING_DUPLICATE_RUN_AT = 3
WARNING_MEDIAN_CHARS_MIN = 3.0
WARNING_MEDIAN_CHARS_MAX = 40.0
WARNING_AVG_LOGPROB_MIN = -1.0
WARNING_NO_SPEECH_PROB_MAX = 0.6


class _TerminalEngineFailure(Exception):
    """标记系统性失败：engine.transcribe() 自身抛出的异常、或引擎未加载这个
    编程错误。候选级失败（指纹校验、音频提取、字幕段校验）改为跳过当前
    候选、继续处理下一个；只有这两种视为系统性，需要照常向上传播、中止
    整批——用一个专门的包裹类型标记，而不是按异常类型区分，因为
    EngineError/PipelineError 两种类型在候选级失败和系统性失败里都会用到。
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


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
        self.candidate_failures: dict[str, dict[str, Any]] = {
            str(item["sample_id"]): dict(item)
            for item in manifest.get("candidate_failures") or []
            if isinstance(item, dict) and item.get("sample_id")
        }

    @classmethod
    def create(
        cls,
        options: PipelineOptions,
        report: DiscoveryReport,
        excluded_videos: Sequence[Path] | None = None,
    ) -> "SubtitlePipeline":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise PipelineError(f"生成的 run_id 不安全：{run_id}", phase=Phase.SETUP)
        run_dir = options.runs_root.expanduser().resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        pilot_count = max(0, min(options.pilot_count, len(report.candidates)))
        pilot = [item.sample_id for item in report.candidates[:pilot_count]]
        remaining = [
            item.sample_id
            for item in report.candidates
            if item.sample_id not in set(pilot)
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
                "candidate_count": len(report.candidates),
                "conflict_count": len(report.conflicts),
                "protected_nonempty_txt_count": len(report.protected_texts),
            },
            "phases": {
                "pilot": pilot,
                "remaining": remaining,
                "all": [item.sample_id for item in report.candidates],
            },
            "candidates": [item.to_dict() for item in report.candidates],
            "protected_nonempty_txt": [
                item.to_dict() for item in report.protected_texts
            ],
            "excluded_videos": sorted(
                {str(path.expanduser().resolve()) for path in (excluded_videos or ())}
            ),
            "runtime": None,
            "stage": None,
            "commit": None,
            "candidate_failures": [],
        }
        write_json_atomic(run_dir / "manifest.json", manifest)
        return cls(
            options, run_dir, report.candidates, report.protected_texts, manifest
        )

    @classmethod
    def resume(cls, run_dir: Path) -> "SubtitlePipeline":
        resolved = run_dir.expanduser().resolve()
        manifest = load_json(resolved / "manifest.json", phase=Phase.SETUP)
        options = options_from_dict(manifest["options"])
        candidates = [Candidate.from_dict(item) for item in manifest["candidates"]]
        protected = [
            ProtectedText.from_dict(item) for item in manifest["protected_nonempty_txt"]
        ]
        return cls(options, resolved, candidates, protected, manifest)

    def update_manifest(self, **changes: Any) -> None:
        self.manifest.update(changes)
        self.manifest["updated_at_utc"] = utc_now()
        write_json_atomic(self.run_dir / "manifest.json", self.manifest)

    def engine_output(self, engine: str, sample_id: str) -> Path:
        return self.run_dir / "engines" / engine / f"{sample_id}.json"

    def load_engine_output(self, engine: str, candidate: Candidate) -> dict[str, Any]:
        phase = engine_phase(engine)
        payload = load_json(
            self.engine_output(engine, candidate.sample_id), phase=phase
        )
        if payload.get("sample_id") != candidate.sample_id:
            raise PipelineError(
                f"{engine} sample_id 不匹配",
                phase=phase,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )
        if Path(str(payload.get("source_path", ""))).resolve() != candidate.video_path:
            raise PipelineError(
                f"{engine} source_path 不匹配",
                phase=phase,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )
        if payload.get("engine") != engine:
            raise PipelineError(
                f"{engine} 引擎声明不匹配",
                phase=phase,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )
        segments = [Segment.from_dict(item) for item in payload.get("segments", [])]
        validate_segments(
            segments,
            float(payload["duration_seconds"]),
            engine=engine,
            sample_id=candidate.sample_id,
            video_path=str(candidate.video_path),
        )
        return payload

    def output_complete(self, engine: str, candidate: Candidate) -> bool:
        try:
            self.load_engine_output(engine, candidate)
            return True
        except Exception:
            return False

    def _eligible_candidates(
        self, sample_ids: Sequence[str] | None = None
    ) -> list[Candidate]:
        """排除候选级失败（self.candidate_failures）后的候选；stage()/commit()/
        verify() 都只处理这个子集，让它们对候选级失败的候选保持沉默——
        跳过继续，不是整批中止。传 sample_ids 则先限定在这个子集里筛选。
        """
        selected = (
            self.candidates
            if sample_ids is None
            else [
                candidate
                for candidate in self.candidates
                if candidate.sample_id in sample_ids
            ]
        )
        return [
            candidate
            for candidate in selected
            if candidate.sample_id not in self.candidate_failures
        ]

    def _build_engine(self, engine_name: str) -> Any:
        if engine_name == "funasr":
            return FunASREngine(self.options, self.device)
        return FasterWhisperEngine(self.options, self.device, self.compute_type)

    def _load_engines_if_needed(self, sample_ids: Sequence[str]) -> dict[str, Any]:
        """按需构造并加载两个引擎；这批候选两个引擎都已完成时不加载。

        两个引擎总是配对加载/释放，不按引擎单独判断是否需要——一旦这批
        候选里有任何一个引擎的输出缺失，就把两个都装上，理由见 spec：
        模型常驻范围覆盖整个批次，不逐引擎精细判断。
        """
        selected = [
            candidate
            for candidate in self.candidates
            if candidate.sample_id in sample_ids
        ]
        needs_any = any(
            not self.output_complete(engine_name, candidate)
            for engine_name in ENGINE_NAMES
            for candidate in selected
        )
        if not needs_any:
            return {}
        engines: dict[str, Any] = {}
        try:
            for engine_name in ENGINE_NAMES:
                print(f"加载 {engine_name}：device={self.device}", flush=True)
                emit_event(EngineStartEvent(engine=engine_name, device=self.device))
                engine = self._build_engine(engine_name)
                engine.load()
                engines[engine_name] = engine
        except Exception:
            # 两个引擎打包加载，第二个失败时第一个已经真的装进显存了——
            # 不补这个 except 会让它随局部变量一起丢失、永远不会 close()。
            for loaded in engines.values():
                loaded.close()
            raise
        return engines

    def run_candidates(
        self, sample_ids: Sequence[str], engines: Mapping[str, Any]
    ) -> None:
        """逐候选交替处理：一个候选紧接着跑完两个引擎，再处理下一个候选。

        音频抽取和转写前后的指纹校验都收在候选级别，两个引擎共用同一份
        wav、共用同一对前后校验——不是每个引擎各来一遍。候选级失败（指纹
        校验、音频提取、字幕段校验）记录后跳过，继续处理下一批里的下一个
        候选，不再中止整批；只有 engine.transcribe() 自身失败（或引擎未
        加载这个编程错误）视为系统性失败，照常向上传播、中止整批。已经
        记录过失败的候选（含 resume 时从 manifest 恢复的）不会被重试。
        """
        selected = [
            candidate
            for candidate in self.candidates
            if candidate.sample_id in sample_ids
        ]
        pending = [
            candidate
            for candidate in selected
            if candidate.sample_id not in self.candidate_failures
            and any(
                not self.output_complete(engine_name, candidate)
                for engine_name in ENGINE_NAMES
            )
        ]
        if not pending:
            return
        for index, candidate in enumerate(pending, start=1):
            try:
                self._run_one_candidate(candidate, index, len(pending), engines)
            except _TerminalEngineFailure as exc:
                raise exc.original from exc
            except (PipelineError, EngineError, OSError) as exc:
                self._record_candidate_failure(candidate, exc)

    def _run_one_candidate(
        self,
        candidate: Candidate,
        index: int,
        total: int,
        engines: Mapping[str, Any],
    ) -> None:
        before = fingerprint(candidate.video_path, include_hash=False)
        if not fingerprints_match_quick(before, candidate.video_fingerprint):
            raise PipelineError(
                f"视频在转写前发生变化：{candidate.video_path}",
                phase=Phase.FINGERPRINT,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )
        wav_path = self.run_dir / "work" / f"{candidate.sample_id}.wav"
        duration: float | None = None
        try:
            for engine_name in ENGINE_NAMES:
                if self.output_complete(engine_name, candidate):
                    continue
                if engine_name not in engines:
                    # 调用方（execute()）本该已经用 _load_engines_if_needed()
                    # 保证：这批候选里只要有任何一个未完成的引擎，两个引擎
                    # 就都会被装好传进来。走到这里说明调用方没有遵守这个
                    # 前提——是编程错误，不是这一个候选独有的问题，视为
                    # 系统性失败，中止整批而不是跳过。
                    raise _TerminalEngineFailure(
                        PipelineError(
                            f"引擎 {engine_name} 未加载，无法处理候选",
                            phase=Phase.SETUP,
                            sample_id=candidate.sample_id,
                            video_path=str(candidate.video_path),
                        )
                    )
                engine = engines[engine_name]
                print(
                    f"[{engine_name} {index}/{total}] {candidate.video_path}",
                    flush=True,
                )
                emit_event(
                    ProgressEvent(
                        engine=engine_name,
                        index=index,
                        total=total,
                        video_path=str(candidate.video_path),
                    )
                )
                if duration is None:
                    duration = extract_audio(
                        self.ffmpeg, candidate.video_path, wav_path
                    )
                started = time.perf_counter()
                try:
                    segments, metadata = engine.transcribe(
                        wav_path, duration, self.hotwords
                    )
                except _TerminalEngineFailure:
                    raise
                except Exception as exc:
                    raise _TerminalEngineFailure(exc) from exc
                validate_segments(
                    segments,
                    duration,
                    engine=engine_name,
                    sample_id=candidate.sample_id,
                    video_path=str(candidate.video_path),
                )
                write_json_atomic(
                    self.engine_output(engine_name, candidate.sample_id),
                    {
                        "schema_version": "1.0",
                        "sample_id": candidate.sample_id,
                        "source_path": str(candidate.video_path),
                        "engine": engine_name,
                        "duration_seconds": round(duration, 6),
                        "elapsed_total_seconds": round(
                            time.perf_counter() - started, 6
                        ),
                        "metadata": metadata,
                        "segments": [segment.to_dict() for segment in segments],
                    },
                )
        finally:
            if wav_path.exists():
                wav_path.unlink()
        after = fingerprint(candidate.video_path, include_hash=False)
        if not fingerprints_match_quick(after, candidate.video_fingerprint):
            raise PipelineError(
                f"视频在转写后发生变化：{candidate.video_path}",
                phase=Phase.FINGERPRINT,
                sample_id=candidate.sample_id,
                video_path=str(candidate.video_path),
            )

    def _record_candidate_failure(
        self, candidate: Candidate, exc: BaseException
    ) -> None:
        """记录一次候选级失败并跳过，不中止整批；持久化到 manifest 供 resume 用。"""
        phase = getattr(exc, "phase", None) or Phase.SETUP
        reason = str(exc)
        print(
            f"候选处理失败，跳过：{candidate.video_path}（{reason}）",
            flush=True,
        )
        emit_event(
            CandidateFailedEvent(
                phase=phase,
                reason=reason,
                video_path=str(candidate.video_path),
                sample_id=candidate.sample_id,
            )
        )
        self.candidate_failures[candidate.sample_id] = {
            "sample_id": candidate.sample_id,
            "video_path": str(candidate.video_path),
            "phase": phase.value,
            "reason": reason,
        }
        self.update_manifest(candidate_failures=list(self.candidate_failures.values()))

    def quality_report(
        self,
        candidate: Candidate,
        caption_payload: bytes,
        funasr_segments: Sequence[Segment],
        faster_segments: Sequence[Segment],
        duration_seconds: float,
    ) -> dict[str, Any]:
        try:
            format_metrics = validate_rendered(caption_payload)
        except FormatError as exc:
            format_metrics = {
                "style": "timed",
                "segment_count": 0,
                "median_text_chars": 0,
                "max_text_chars": 0,
                "median_gap_seconds": 0,
                "first_timestamp": "",
                "last_timestamp": "",
            }
            format_error = str(exc)
        else:
            format_error = None
        funasr_text = "".join(normalize_text(item.text) for item in funasr_segments)
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
        if format_error is not None:
            errors.append(f"格式校验失败：{format_error}")
        if len(funasr_segments) < minimum_segments:
            errors.append(
                f"FunASR 字幕段数过少：{len(funasr_segments)} < {minimum_segments}"
            )
        if density < 30 or density > 1200:
            errors.append(f"文本密度异常：{density:.1f} chars/min")
        elif density < WARNING_DENSITY_MIN or density > WARNING_DENSITY_MAX:
            warnings.append(f"文本密度告警：{density:.1f} chars/min")
        if duplicate_run > 4:
            errors.append(f"连续重复字幕过多：{duplicate_run}")
        elif duplicate_run > WARNING_DUPLICATE_RUN_AT:
            warnings.append(f"连续重复字幕告警：{duplicate_run}")
        if (
            not math.isfinite(length_ratio)
            or length_ratio < 0.45
            or length_ratio > 2.20
        ):
            errors.append(f"双模型全文长度比异常：{length_ratio:.3f}")
        elif (
            length_ratio < WARNING_LENGTH_RATIO_MIN
            or length_ratio > WARNING_LENGTH_RATIO_MAX
        ):
            warnings.append(f"双模型全文长度比告警：{length_ratio:.3f}")
        median_chars = float(format_metrics["median_text_chars"])
        if format_error is None and (
            median_chars < WARNING_MEDIAN_CHARS_MIN
            or median_chars > WARNING_MEDIAN_CHARS_MAX
        ):
            warnings.append(f"中位段长偏离常见范围：{median_chars:g} 字")
        confidence = faster_whisper_confidence(faster_segments)
        if (
            confidence["avg_logprob_min"] is not None
            and confidence["avg_logprob_min"] < WARNING_AVG_LOGPROB_MIN
        ):
            warnings.append(
                "faster-whisper 置信度告警："
                f"avg_logprob 最低 {confidence['avg_logprob_min']:.2f}"
            )
        if (
            confidence["no_speech_prob_max"] is not None
            and confidence["no_speech_prob_max"] > WARNING_NO_SPEECH_PROB_MAX
        ):
            warnings.append(
                "faster-whisper 置信度告警："
                f"no_speech_prob 最高 {confidence['no_speech_prob_max']:.2f}"
            )
        alignment = align_for_audit(funasr_segments, faster_segments)
        high_risk_count = int(alignment["high_risk_count"])
        if high_risk_count:
            warnings.append(f"high_risk 冲突审计：{high_risk_count} 段需人工复核")
        return {
            "sample_id": candidate.sample_id,
            "video_path": str(candidate.video_path),
            "target_path": str(candidate.target_path),
            "duration_seconds": round(duration_seconds, 6),
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
            "has_warnings": bool(warnings),
            "warning_count": len(warnings),
            "output_format": {
                "style": "timed",
                "timestamp": "MM:SS / H:MM:SS",
                "encoding": "utf-8",
                "bom": False,
                "newline_name": "crlf",
                "terminal_newline": True,
            },
            "format_metrics": format_metrics,
            "funasr_segment_count": len(funasr_segments),
            "faster_whisper_segment_count": len(faster_segments),
            "text_density_chars_per_minute": round(density, 3),
            "funasr_to_faster_whisper_text_length_ratio": round(length_ratio, 6),
            "longest_consecutive_duplicate_run": duplicate_run,
            "caption_sha256": hashlib.sha256(caption_payload).hexdigest(),
            "caption_size": len(caption_payload),
            "faster_whisper_confidence": confidence,
            "high_risk_count": high_risk_count,
            "alignment_summary": {
                key: value for key, value in alignment.items() if key != "alignments"
            },
            "alignment": alignment,
        }

    def stage(self, sample_ids: Sequence[str]) -> dict[str, Any]:
        validate_protected(self.protected, phase=Phase.QUALITY_GATE)
        # 候选级失败（跳过继续）的候选排除在外——它们已经在 run_candidates()
        # 里被记录过一次，这里再拿同一份指纹基线重新校验只会重复抓到同一个
        # 已知问题、把它当成新错误再抛一次，中止掉本该照常写回的其余候选。
        validate_candidates(
            self._eligible_candidates(),
            require_initial_target=True,
            phase=Phase.QUALITY_GATE,
        )
        selected = self._eligible_candidates(sample_ids)
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
                raise PipelineError(
                    f"双模型音频时长不一致：{candidate.sample_id}",
                    phase=Phase.QUALITY_GATE,
                    sample_id=candidate.sample_id,
                    video_path=str(candidate.video_path),
                )
            caption_payload = render_segments(funasr_segments)
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
            previous = load_json(stage_path, phase=Phase.QUALITY_GATE)
        merged = {
            str(item["sample_id"]): item
            for item in previous.get("samples", [])
            if isinstance(item, dict) and item.get("sample_id")
        }
        for report in reports:
            merged[str(report["sample_id"])] = report
        all_ids = [
            sample_id
            for sample_id in self.manifest["phases"]["all"]
            if sample_id not in self.candidate_failures
        ]
        staged_all = all(sample_id in merged for sample_id in all_ids)
        all_passed = staged_all and all(
            bool(merged[sample_id]["passed"]) for sample_id in all_ids
        )
        selected_passed = all(
            sample_id in merged and bool(merged[sample_id]["passed"])
            for sample_id in sample_ids
            if sample_id not in self.candidate_failures
        )
        stage_report = {
            "schema_version": "1.0",
            "run_id": self.manifest["run_id"],
            "updated_at_utc": utc_now(),
            "selected_ids": list(sample_ids),
            "selected_passed": selected_passed,
            "staged_ids": sorted(merged),
            "staged_all": staged_all,
            "all_passed": all_passed,
            "warning_count": sum(
                int(item.get("warning_count", 0)) for item in merged.values()
            ),
            "warning_sample_ids": sorted(
                sample_id for sample_id, item in merged.items() if item.get("warnings")
            ),
            "has_warnings": any(item.get("warnings") for item in merged.values()),
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
            raise PipelineError(
                "候选未全部就绪或未通过质量门禁，未写回课程目录",
                phase=Phase.QUALITY_GATE,
            )
        return stage_report

    def commit(self) -> dict[str, Any]:
        stage_report = load_json(
            self.run_dir / "stage_report.json", phase=Phase.WRITEBACK
        )
        if not stage_report.get("staged_all") or not stage_report.get("all_passed"):
            raise PipelineError("全部字幕尚未通过质量门禁", phase=Phase.WRITEBACK)
        reports = {str(item["sample_id"]): item for item in stage_report["samples"]}
        validate_protected(self.protected, phase=Phase.WRITEBACK)
        # 候选级失败（跳过继续）的候选从未进入 stage_report，写回时排除，不
        # 强求它们凑齐——写回只覆盖真正处理完、通过质量门禁的那部分候选。
        eligible = self._eligible_candidates()
        validate_candidates(
            eligible, require_initial_target=True, phase=Phase.WRITEBACK
        )
        backups = self.run_dir / "backups"
        payloads: dict[str, bytes] = {}
        for candidate in eligible:
            prepared = self.run_dir / "prepared" / f"{candidate.sample_id}.txt"
            payload = prepared.read_bytes()
            validate_rendered(payload)
            expected_hash = str(reports[candidate.sample_id]["caption_sha256"])
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise PipelineError(
                    f"暂存字幕哈希不匹配：{prepared}",
                    phase=Phase.WRITEBACK,
                    sample_id=candidate.sample_id,
                    video_path=str(candidate.video_path),
                )
            payloads[candidate.sample_id] = payload
            backup_dir = backups / candidate.sample_id
            original = (
                candidate.target_path.read_bytes()
                if candidate.target_path.exists()
                else b""
            )
            write_bytes_atomic(backup_dir / "original.txt", original)
            write_json_atomic(
                backup_dir / "state.json",
                {
                    "target_path": str(candidate.target_path),
                    "initial_state": candidate.initial_state,
                    "fingerprint": candidate.target_fingerprint.to_dict(),
                },
            )
        # 写回前全量复核：视频哈希保护开启时做第 2 次（也是最后一次）全量读；
        # 关闭时只做快速校验。
        for candidate in eligible:
            current_video = fingerprint(
                candidate.video_path,
                include_hash=candidate.video_fingerprint.sha256 is not None,
            )
            if not fingerprints_match(current_video, candidate.video_fingerprint):
                raise PipelineError(
                    f"写回前视频已变化：{candidate.video_path}",
                    phase=Phase.WRITEBACK,
                    sample_id=candidate.sample_id,
                    video_path=str(candidate.video_path),
                )
        committed: list[Candidate] = []
        try:
            for candidate in eligible:
                write_bytes_atomic(candidate.target_path, payloads[candidate.sample_id])
                committed.append(candidate)
            for candidate in eligible:
                actual = candidate.target_path.read_bytes()
                if actual != payloads[candidate.sample_id]:
                    raise PipelineError(
                        f"写回后内容不一致：{candidate.target_path}",
                        phase=Phase.WRITEBACK,
                        sample_id=candidate.sample_id,
                        video_path=str(candidate.video_path),
                    )
                validate_rendered(actual)
            validate_protected(self.protected, phase=Phase.WRITEBACK)
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
            for candidate in eligible
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
        validate_protected(self.protected, phase=Phase.VERIFY)
        failures: list[str] = []
        entries: list[dict[str, Any]] = []
        # 候选级失败（跳过继续）的候选从未写回，复核时排除，不然会被误报
        # 成"目标字幕为空"。
        eligible = self._eligible_candidates()
        for candidate in eligible:
            current_video = fingerprint(candidate.video_path, include_hash=False)
            if not fingerprints_match_quick(current_video, candidate.video_fingerprint):
                failures.append(f"视频变化：{candidate.video_path}")
                continue
            prepared = self.run_dir / "prepared" / f"{candidate.sample_id}.txt"
            if (
                not candidate.target_path.is_file()
                or candidate.target_path.stat().st_size == 0
            ):
                failures.append(f"目标字幕为空：{candidate.target_path}")
                continue
            actual = candidate.target_path.read_bytes()
            if actual != prepared.read_bytes():
                failures.append(f"目标字幕与暂存产物不一致：{candidate.target_path}")
                continue
            metrics = validate_rendered(actual)
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
            raise PipelineError(
                "最终复核失败：\n" + "\n".join(failures), phase=Phase.VERIFY
            )
        return report

    def _finalize_summary(self) -> None:
        """尽力写 summary.txt；失败只告警，不掩盖运行结果。"""
        try:
            write_summary(self.run_dir)
        except (OSError, PipelineError) as exc:
            print(f"警告：无法生成运行摘要：{exc}", file=sys.stderr, flush=True)

    def execute(self) -> Path:
        validate_protected(self.protected, phase=Phase.SETUP)
        # 候选级失败（跳过继续）的候选排除在外——否则 resume() 之后每次
        # execute() 一开始就会拿同一份指纹基线重新抓到同一个已经记录过的
        # 候选级失败，永远卡在这里，连 run_candidates() 都进不去。
        validate_candidates(
            self._eligible_candidates(), require_initial_target=True, phase=Phase.SETUP
        )
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
        try:
            pilot = list(self.manifest["phases"]["pilot"])
            remaining = list(self.manifest["phases"]["remaining"])
            engines = self._load_engines_if_needed(pilot + remaining)
            try:
                if pilot:
                    self.run_candidates(pilot, engines)
                    self.stage(pilot)
                if remaining:
                    self.run_candidates(remaining, engines)
                    self.stage(list(self.manifest["phases"]["all"]))
                elif pilot:
                    self.stage(pilot)
            finally:
                for engine in engines.values():
                    engine.close()
            self.commit()
            self.verify()
        finally:
            self._finalize_summary()
        return self.run_dir


def discover_for_options(options: PipelineOptions) -> DiscoveryReport:
    return discover(
        options.roots,
        hash_videos=options.hash_videos,
        probe_durations=True,
        ffmpeg=options.ffmpeg,
    )
