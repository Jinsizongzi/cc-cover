from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cc_cover.gui.data_root import runtime_paths
from cc_cover.gui.device import (
    detect_device_command,
    device_probe_commands,
    nvidia_probe_command,
    parsed_device,
    parsed_nvidia_probe,
)


class DeviceDetectionTests(unittest.TestCase):
    def test_detect_command_probes_runtime_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            command = detect_device_command(paths)

        self.assertEqual(command[0], str(paths.venv_python))
        self.assertIn("torch.cuda.is_available()", command[-1])
        self.assertIn("ctranslate2.get_cuda_device_count()", command[-1])
        self.assertIn("print('cuda'", command[-1])

    def test_parsed_device_accepts_cuda_and_cpu_output(self) -> None:
        self.assertEqual(parsed_device("cuda\n"), "cuda")
        self.assertEqual(parsed_device("cpu\n"), "cpu")

    def test_parsed_device_rejects_unknown_output(self) -> None:
        self.assertIsNone(parsed_device(""))
        self.assertIsNone(parsed_device("无法检测"))

    def test_nvidia_probe_command_queries_gpu_list(self) -> None:
        command = nvidia_probe_command()

        self.assertEqual(command[0], "nvidia-smi")
        self.assertIn("--query-gpu=name", command[1:])
        self.assertIn("--format=csv,noheader", command[1:])

    def test_parsed_nvidia_probe_maps_gpu_names_to_cuda(self) -> None:
        self.assertEqual(parsed_nvidia_probe("NVIDIA GeForce RTX 4090\n"), "cuda")
        self.assertEqual(parsed_nvidia_probe("  NVIDIA RTX A6000  \n"), "cuda")

    def test_parsed_nvidia_probe_rejects_empty_output(self) -> None:
        self.assertIsNone(parsed_nvidia_probe(""))
        self.assertIsNone(parsed_nvidia_probe("\n"))

    def test_device_probe_commands_prefer_runtime_then_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = runtime_paths(
                frozen=True,
                bundle_root=base / "bundle",
                data_root=base / "data",
            )
            commands = device_probe_commands(paths)

        self.assertEqual(commands[0][0], str(paths.venv_python))
        self.assertEqual(commands[1][0], "nvidia-smi")


if __name__ == "__main__":
    unittest.main()
