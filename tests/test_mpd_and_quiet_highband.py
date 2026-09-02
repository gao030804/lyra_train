import ast
import math
import types
import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SOUNDSTREAM_PATH = ROOT / "audiolm_pytorch" / "soundstream.py"


def extract_definitions(*names):
    tree = ast.parse(SOUNDSTREAM_PATH.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    selected_names = {node.name for node in selected}
    soundstream = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SoundStream"
    )
    selected.extend(
        node for node in soundstream.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        and node.name not in selected_names
    )
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))

    def rearrange_period(tensor, pattern, *, p):
        if pattern != "b c (n p) -> b c n p":
            raise AssertionError(f"unexpected test rearrange pattern: {pattern}")
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1, p)

    namespace = {
        "F": F,
        "Module": nn.Module,
        "ModuleList": nn.ModuleList,
        "nn": nn,
        "torch": torch,
        "math": math,
        "rearrange": rearrange_period,
    }
    exec(compile(module, str(SOUNDSTREAM_PATH), "exec"), namespace)
    return namespace


class MultiPeriodDiscriminatorTests(unittest.TestCase):
    def test_period_bank_returns_aligned_logits_and_features(self):
        namespace = extract_definitions(
            "leaky_relu",
            "PeriodDiscriminator",
            "MultiPeriodDiscriminator",
        )
        discriminator = namespace["MultiPeriodDiscriminator"]((3, 7))
        waveform = torch.randn(2, 1, 1001, requires_grad=True)
        logits, features = discriminator(waveform, return_intermediates=True)

        self.assertEqual(len(logits), 2)
        self.assertEqual(len(features), 2)
        self.assertTrue(all(len(period_features) == 5 for period_features in features))
        loss = sum(period_logits.mean() for period_logits in logits)
        loss.backward()
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())


class QuietHighbandTests(unittest.TestCase):
    def setUp(self):
        namespace = extract_definitions("quiet_multiband_noise_metrics")
        self.model = types.SimpleNamespace(
            target_sample_hz=16_000,
            quiet_mask_transition_db=6.0,
            quiet_highband_margin_db=0.5,
            quiet_highband_loss_weight=1.0,
            zero=torch.tensor(0.0),
        )
        self.method = types.MethodType(
            namespace["quiet_multiband_noise_metrics"], self.model
        )

    def test_identical_quiet_audio_has_zero_excess(self):
        time = torch.arange(8_000) / self.model.target_sample_hz
        target = (0.002 * torch.sin(2 * math.pi * 440 * time)).unsqueeze(0)
        loss, excess_db = self.method(target, target.clone())
        self.assertLess(float(loss), 1e-7)
        self.assertLess(float(excess_db), 1e-6)

    def test_added_silent_region_upper_hf_is_penalized(self):
        time = torch.arange(8_000) / self.model.target_sample_hz
        target = (0.002 * torch.sin(2 * math.pi * 440 * time)).unsqueeze(0)
        recon = target + (0.004 * torch.sin(2 * math.pi * 6_200 * time)).unsqueeze(0)
        loss, excess_db = self.method(target, recon)
        self.assertGreater(float(loss), 0.1)
        self.assertGreater(float(excess_db), 1.0)


if __name__ == "__main__":
    unittest.main()
