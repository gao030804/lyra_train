import pickle

import pytest
import torch

import train_soundstream
from train_soundstream import (
    AC320_ABSOLUTE_GATE_STAGES,
    QUALITY_RETENTION_STAGES,
    STAGE_DEFAULTS,
    build_model_from_checkpoint,
    checkpoint_handoff_failure_reasons,
    load_model_weights_only,
)


def test_ac320_absolute_gate_is_stage1_only():
    assert "recon_pretrain" in AC320_ABSOLUTE_GATE_STAGES
    assert "spectral_refine" not in AC320_ABSOLUTE_GATE_STAGES
    assert "gan_pretrain" not in AC320_ABSOLUTE_GATE_STAGES
    assert "hardware_qat_finetune" not in AC320_ABSOLUTE_GATE_STAGES


def test_spectral_refine_is_two_phase_bounded_frame_leakage_repair():
    refine = STAGE_DEFAULTS["spectral_refine"]

    assert "spectral_refine" in QUALITY_RETENTION_STAGES
    assert refine["lr"] == 5e-6
    assert refine["encoder_lr"] == 5e-7
    assert refine["encoder_unfreeze_step"] == 3_000
    assert refine["patience"] == 100
    assert refine["quality_retention_patience"] == 100
    assert refine["frame_phase_loss_weight"] == pytest.approx(0.05)
    assert refine["frame_phase_loss_warmup_steps"] == 3_000
    assert refine["voiced_highband_loss_weight"] == pytest.approx(0.15)
    assert refine["voiced_highband_loss_warmup_steps"] == 0
    assert refine["voiced_hf_retention_loss_weight"] == pytest.approx(0.005)
    assert refine["stft_recon_loss_weight"] == pytest.approx(0.05)
    assert refine["quality_retention_start_step"] == 3_000
    assert refine["decoder_x8_residual_scale_target"] is None
    assert refine["decoder_x8_residual_scale_ramp_steps"] == 0


def test_spectral_gradient_diagnostic_defaults_to_validation_interval(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_soundstream.py"])
    args = train_soundstream.parse_args()

    assert args.spectral_grad_diagnostics_every == 250


class GeneratorOnlyModel:
    def __init__(self, latent_context_frames):
        self.decoder_latent_context_frames = latent_context_frames

    def load_generator_state_dict(
        self,
        state_dict,
        *,
        allow_missing_latent_context=False,
    ):
        assert state_dict == {}
        self.allow_missing_latent_context = allow_missing_latent_context
        return []


class TinyCheckpointModel(torch.nn.Module):
    def __init__(self, gain=1.):
        super().__init__()
        self.register_buffer("gain", torch.tensor(float(gain)))
        self.runtime_restored = False

    def restore_decoder_runtime_state(self, config):
        self.runtime_restored = config["gain"] == float(self.gain)


def test_streaming_stages_preserve_stage2_latent_context_structure():
    assert STAGE_DEFAULTS["gan_pretrain"]["decoder_latent_context_frames"] == 4
    assert STAGE_DEFAULTS["stream_finetune"]["decoder_latent_context_frames"] == 4
    assert STAGE_DEFAULTS["stream_finetune_long"]["decoder_latent_context_frames"] == 4
    assert STAGE_DEFAULTS["hardware_qat_finetune"]["decoder_latent_context_frames"] == 4


def test_final_hardware_qat_uses_low_lr_and_fixed_full_rvq_profile():
    qat = STAGE_DEFAULTS["hardware_qat_finetune"]
    assert qat["lr"] == 5e-7
    assert qat["encoder_lr"] == 3e-6
    assert qat["gan_adversarial_max"] == 0.


def test_checkpoint_loader_reports_latent_context_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": pickle.dumps({"decoder_latent_context_frames": 4}),
            "model": {},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="latent-context structure mismatch"):
        load_model_weights_only(
            GeneratorOnlyModel(latent_context_frames=0),
            checkpoint,
            generator_only=True,
        )


def test_checkpoint_loader_allows_only_explicit_zero_to_four_context_expansion(tmp_path):
    checkpoint = tmp_path / "stage1.pt"
    torch.save(
        {
            "config": pickle.dumps({"decoder_latent_context_frames": 0}),
            "model": {},
        },
        checkpoint,
    )
    model = GeneratorOnlyModel(latent_context_frames=4)

    load_model_weights_only(
        model,
        checkpoint,
        generator_only=True,
        allow_latent_context_expansion=True,
    )

    assert model.allow_missing_latent_context is True


def test_evaluation_model_is_rebuilt_from_checkpoint_config(monkeypatch, tmp_path):
    checkpoint = tmp_path / "exact-config.pt"
    reference = TinyCheckpointModel(gain=3.5)
    torch.save(
        {
            "config": pickle.dumps({"gain": 3.5}),
            "model": reference.state_dict(),
        },
        checkpoint,
    )
    monkeypatch.setattr(train_soundstream, "SoundStream", TinyCheckpointModel)

    rebuilt, config, _ = build_model_from_checkpoint(checkpoint)

    assert isinstance(rebuilt, TinyCheckpointModel)
    assert config == {"gain": 3.5}
    assert rebuilt.gain.item() == pytest.approx(3.5)
    assert rebuilt.runtime_restored is True


def test_handoff_gate_rejects_collapsed_readback_metrics():
    reasons = checkpoint_handoff_failure_reasons({
        "score": 1.,
        "aligned_si_sdr": -38.,
        "aligned_correlation": 0.02,
        "active_code_ratio": 0.,
        "codebook_perplexity": 0.,
        "q00_validation_eligible": 0.,
        "q01_validation_eligible": 0.,
        "rvq_validation_eligible": 0.,
    })

    assert "aligned_si_sdr<-10dB" in reasons
    assert "aligned_correlation<0.10" in reasons
    assert "rvq_unhealthy" in reasons


def test_handoff_gate_rejects_excess_320hz_leakage():
    reasons = checkpoint_handoff_failure_reasons({
        "score": 1.,
        "aligned_si_sdr": 1.,
        "aligned_correlation": 0.75,
        "active_code_ratio": 0.99,
        "codebook_perplexity": 200.,
        "q00_validation_eligible": 1.,
        "q01_validation_eligible": 1.,
        "rvq_validation_eligible": 1.,
        "ac_320_isolated": 0.1378,
    })

    assert "ac_320_isolated>0.1000" in reasons
