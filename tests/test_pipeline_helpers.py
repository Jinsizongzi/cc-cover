from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cc_cover.engines import local_faster_whisper_model, local_funasr_model
from cc_cover.models import PipelineOptions
from cc_cover.pipeline import options_from_dict, options_to_dict, write_bytes_atomic


class PipelineHelperTests(unittest.TestCase):
    def test_atomic_write_replaces_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "subtitle.txt"
            write_bytes_atomic(target, b"first")
            write_bytes_atomic(target, b"second")
            self.assertEqual(target.read_bytes(), b"second")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_options_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            options = PipelineOptions(
                roots=[root],
                runs_root=root / "runs",
                model_cache=root / "models",
                include_whitespace_only=True,
                apply=True,
            )
            restored = options_from_dict(options_to_dict(options))

        self.assertEqual(restored.roots, options.roots)
        self.assertEqual(restored.runs_root, options.runs_root)
        self.assertTrue(restored.include_whitespace_only)
        self.assertTrue(restored.apply)

    def test_existing_model_caches_resolve_to_local_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            funasr_cache = root / "funasr"
            snapshot = (
                funasr_cache
                / "models"
                / "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch"
                / "snapshots"
                / "master"
            )
            snapshot.mkdir(parents=True)
            whisper_cache = root / "faster-whisper"
            whisper_model = whisper_cache / "large-v3-turbo"
            whisper_model.mkdir(parents=True)

            resolved_funasr = local_funasr_model("fsmn-vad", funasr_cache)
            resolved_whisper = local_faster_whisper_model(
                "large-v3-turbo", whisper_cache
            )

        self.assertEqual(Path(resolved_funasr), snapshot.resolve())
        self.assertEqual(Path(resolved_whisper), whisper_model.resolve())


if __name__ == "__main__":
    unittest.main()
