"""字幕流水线：扫描后的双引擎转写、质量门禁、写回与复核。

对外只暴露 core.cli / gui.tasks 实际需要的名字；测试如果要单测某个内部
环节（热词提取、指纹校验、审计比对……），直接从对应子模块导入，见各子
模块自己的小接口。
"""

from __future__ import annotations

from cc_cover.core.pipeline.errors import PipelineError
from cc_cover.core.pipeline.io import emit_event, load_optional_json
from cc_cover.core.pipeline.run import SubtitlePipeline, discover_for_options
from cc_cover.core.pipeline.summary import (
    SUMMARY_FILENAME,
    CompletionStats,
    run_completion_stats,
    run_status_label,
    write_summary,
)

__all__ = [
    "PipelineError",
    "emit_event",
    "load_optional_json",
    "SubtitlePipeline",
    "discover_for_options",
    "SUMMARY_FILENAME",
    "CompletionStats",
    "run_completion_stats",
    "run_status_label",
    "write_summary",
]
