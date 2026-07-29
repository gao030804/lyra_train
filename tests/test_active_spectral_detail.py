import ast
import math
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SOUNDSTREAM_PATH = ROOT / "audiolm_pytorch" / "soundstream.py"


def load_active_spectral_detail_method():
    tree = ast.parse(SOUNDSTREAM_PATH.read_text(encoding="utf-8"))
    soundstream_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SoundStream"
    )
    method = next(
        node
        for node in soundstream_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "active_spectral_detail_metrics"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "F": F,
        "math": math,
        "torch": torch,
        "log": lambda tensor, eps=1e-5: torch.log(tensor.clamp(min=eps)),
        "rearrange": None,
    }
    exec(compile(module, str(SOUNDSTREAM_PATH), "exec"), namespace)
    return namespace["active_spectral_detail_metrics"]


class DummySoundStream:
    target_sample_hz = 16_000
    active_spectral_detail_settings = (
        (256, 256, 64, 0.50),
        (512, 512, 128, 1.00),
        (1024, 1024, 256, 1.00),
        (2048, 2048, 512, 0.50),
    )
    active_spectral_detail_relative_db = -50.0
    active_spectral_detail_min_hz = 200.0
    active_spectral_detail_max_hz = 7_800.0
    active_spectral_detail_bands = (
        ("200_1k", 200.0, 1_000.0, 0.50),
        ("1k_3k", 1_000.0, 3_000.0, 1.00),
        ("3k_5k", 3_000.0, 5_000.0, 1.00),
        ("5k_7k", 5_000.0, 7_000.0, 1.25),
        ("7k_7p8k", 7_000.0, 7_800.0, 1.50),
    )
    spectral_envelope_relative_rms_db = -35.0
    spectral_envelope_absolute_rms = 0.003

    def __init__(self):
        self.zero = torch.tensor(0.0)
        method = load_active_spectral_detail_method()
        self.active_spectral_detail_metrics = types.MethodType(method, self)


class ActiveSpectralDetailTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = DummySoundStream()
        time = torch.arange(8_000) / self.model.target_sample_hz
        self.target = (
            0.10 * torch.sin(2 * math.pi * 440 * time)
            + 0.04 * torch.sin(2 * math.pi * 2_500 * time)
            + 0.02 * torch.sin(2 * math.pi * 7_300 * time)
        ).unsqueeze(0)

    def test_identical_audio_has_zero_loss_and_band_error(self):
        loss, metrics = self.model.active_spectral_detail_metrics(
            self.target,
            self.target.clone(),
        )
        self.assertLess(float(loss), 1e-7)
        self.assertEqual(set(metrics), {
            "200_1k", "1k_3k", "3k_5k", "5k_7k", "7k_7p8k"
        })
        for log_error, ratio_db, convergence in metrics.values():
            self.assertLess(abs(float(log_error)), 1e-6)
            self.assertLess(abs(float(ratio_db)), 1e-5)
            self.assertLess(abs(float(convergence)), 1e-6)

    def test_missing_active_detail_produces_positive_loss(self):
        recon = 0.5 * self.target
        loss, metrics = self.model.active_spectral_detail_metrics(
            self.target,
            recon,
        )
        self.assertGreater(float(loss), 0.1)
        for _, ratio_db, convergence in metrics.values():
            self.assertLess(float(ratio_db), -5.0)
            self.assertGreater(float(convergence), 0.4)

    def test_normalized_band_weights_prioritize_missing_upper_detail(self):
        time = torch.arange(16_000) / self.model.target_sample_hz
        low = 0.05 * torch.sin(2 * math.pi * 440 * time)
        upper = 0.05 * torch.sin(2 * math.pi * 7_300 * time)
        target = (low + upper).unsqueeze(0)

        missing_low_loss, _ = self.model.active_spectral_detail_metrics(
            target,
            upper.unsqueeze(0),
        )
        missing_upper_loss, _ = self.model.active_spectral_detail_metrics(
            target,
            low.unsqueeze(0),
        )
        self.assertGreater(
            float(missing_upper_loss),
            float(missing_low_loss),
        )


if __name__ == "__main__":
    unittest.main()
