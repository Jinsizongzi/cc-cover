from __future__ import annotations

from cc_cover.core.models import Phase


class PipelineError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        phase: Phase,
        video_path: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.video_path = video_path
        self.sample_id = sample_id
