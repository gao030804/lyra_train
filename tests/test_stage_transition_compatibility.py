import pickle

import pytest
import torch

import train_soundstream
from train_soundstream import (
    QUALITY_RETENTION_STAGES,
    STAGE_DEFAULTS,
    load_model_weights_only,
)


def test_ac320_is_diagnostic_only():
    """The retired absolute gate must not silently return."""
    assert not hasattr(train_soundstream, "AC320_ABSOLUTE_GATE_STAGES")


def test_stage1_formant_training_schedule():
    stage1 = STAGE_DEFAULTS["recon_pretrain"]

    assert stage1["spectral_envelope_loss_weight"] == pytest.approx(0.08)
    assert stage1["spectral_envelope_loss_start_steps"] == 5_000
    assert stage1["spectral_envelope_loss_warmup_steps"] == 15_000
    assert stage1["formant_peak_loss_weight"] == pytest.approx(0.02)
    assert stage1["formant_peak_loss_start_steps"] == 15_000
    assert stage1["formant_peak_loss_warmup_steps"] == 20_000
    assert stage1["stft_recon_loss_weight"] == pytest.approx(0.05)
    assert stage1["stft_recon_loss_start_steps"] == 5_000
    assert stage1["stft_recon_loss_warmup_steps"] == 15_000
    assert stage1["voiced_highband_loss_weight"] == pytest.approx(0.04)
    assert stage1["upper_highband_loss_weight"] == pytest.approx(0.0025)


def test_spectral_refine_matches_current_optional_profile():
    refine = STAGE_DEFAULTS["spectral_refine"]

    assert "spectral_refine" in QUALITY_RETENTION_STAGES
    assert refine["lr"] == pytest.approx(2e-5)
    assert refine["patience"] == 30
    assert refine["spectral_envelope_loss_weight"] == pytest.approx(0.05)
    assert refine["formant_peak_loss_weight"] == pytest.approx(0.02)
    assert refine["formant_peak_loss_warmup_steps"] == 5_000
    assert refine["stft_recon_loss_weight"] == pytest.approx(0.10)
    assert refine["frame_phase_loss_weight"] == pytest.approx(0.005)
    assert refine["decoder_x8_residual_scale_target"] == pytest.approx(0.85)
    assert refine["decoder_x8_residual_scale_ramp_steps"] == 3_000


def test_loss_gradient_diagnostic_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_soundstream.py"])
    args = train_soundstream.parse_args()

    assert args.loss_grad_diagnostics_every == 1_000


class GeneratorOnlyModel:
    decoder_upsample_mode = "linear"
    decoder_linear_upsample_kernel_min = 4
    decoder_interpolation_mode = "linear"
    decoder_split_first_upsample = False
    encoder_depthwise_separable_blocks = (2, 3)
    encoder_depthwise_separable_revision = 2

    def load_generator_state_dict(self, state_dict):
        assert state_dict == {}
        return []


def matching_config(**overrides):
    config = {
        "decoder_upsample_mode": "linear",
        "decoder_linear_upsample_kernel_min": 4,
        "decoder_interpolation_mode": "linear",
        "decoder_split_first_upsample": False,
        "encoder_depthwise_separable_blocks": (2, 3),
        "encoder_depthwise_separable_revision": 2,
    }
    config.update(overrides)
    return config


def save_generator_checkpoint(path, config):
    torch.save({"config": pickle.dumps(config), "model": {}}, path)


def test_generator_checkpoint_accepts_matching_current_topology(tmp_path):
    checkpoint = tmp_path / "matching.pt"
    save_generator_checkpoint(checkpoint, matching_config())

    config = load_model_weights_only(
        GeneratorOnlyModel(), checkpoint, generator_only=True
    )

    assert config["encoder_depthwise_separable_revision"] == 2


def test_generator_checkpoint_rejects_old_dscnn_activation_revision(tmp_path):
    checkpoint = tmp_path / "old-revision.pt"
    save_generator_checkpoint(
        checkpoint,
        matching_config(encoder_depthwise_separable_revision=1),
    )

    with pytest.raises(ValueError, match="activation topology mismatch"):
        load_model_weights_only(
            GeneratorOnlyModel(), checkpoint, generator_only=True
        )


def test_generator_checkpoint_rejects_decoder_interpolation_mismatch(tmp_path):
    checkpoint = tmp_path / "cubic.pt"
    save_generator_checkpoint(
        checkpoint,
        matching_config(decoder_interpolation_mode="cubic"),
    )

    with pytest.raises(ValueError, match="decoder interpolation mismatch"):
        load_model_weights_only(
            GeneratorOnlyModel(), checkpoint, generator_only=True
        )
