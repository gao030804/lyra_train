import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_soundstream
from train_soundstream import (
    QUALITY_RETENTION_STAGES,
    STAGE_DEFAULTS,
    load_model_weights_only,
)
from audiolm_pytorch.trainer import SoundStreamTrainer


def test_bypass_rvq_is_saved_runtime_mode_and_emits_sentinel_indices():
    from audiolm_pytorch.soundstream import SoundStream

    model = SoundStream(
        channels=4,
        channel_mults=(2, 2),
        strides=(2, 2),
        codebook_dim=8,
        codebook_size=16,
        rq_num_quantizers=2,
        discr_multi_scales=(1,),
        pad_mode="constant",
        bypass_rvq=True,
        rq_rotation_trick=False,
    )
    latent, indices, commitment = model(
        torch.randn(1, 256),
        return_encoded=True,
    )

    assert latent.shape == (1, 64, 8)
    assert indices.shape == (1, 64, 2)
    assert torch.all(indices == -1)
    assert commitment.item() == 0
    assert model.configs["bypass_rvq"] is True
    assert model.configs["rq_rotation_trick"] is False
    assert all(
        not layer.rotation_trick
        for rvq in model.rq.rvqs
        for layer in rvq.layers
    )


ROOT = Path(__file__).resolve().parents[1]


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
    assert args.decoder_upsample_mode == "convtranspose"


def test_stage2_uses_decoder_only_150k_schedule():
    stage2 = STAGE_DEFAULTS["gan_pretrain"]

    assert stage2["steps"] == 150_000
    assert stage2["patience"] is None
    assert stage2["gan_start"] == 2_000
    assert stage2["gan_ramp"] == 15_000
    assert stage2["waveform_discr_update_every"] == (2, 2, 2)
    assert stage2["stft_recon_loss_weight"] == pytest.approx(0.02)
    assert stage2["voiced_highband_loss_weight"] == pytest.approx(0.02)
    assert stage2["voiced_hf_retention_loss_weight"] == pytest.approx(0.01)

    launcher = (ROOT / "run_stage1_stage15_stage2.sh").read_text(
        encoding="utf-8"
    )
    assert '--stage2-unfreeze-encoder-rvq-step -1' in launcher
    assert '--stage2-phase2-start-step 50000' in launcher
    assert '--stage2-phase3-start-step 100000' in launcher
    assert '--no-stage2-quality-hard-stop' in launcher
    assert '--early-stopping-patience' not in launcher


def test_stage2_phase_boundaries_adjust_lr_and_gan_weights():
    trainer = object.__new__(SoundStreamTrainer)
    torch.nn.Module.__init__(trainer)
    trainer.enable_gan = True
    trainer.gan_start_step = 2_000
    trainer.gan_ramp_steps = 15_000
    trainer.gan_adversarial_max = 2e-4
    trainer.gan_feature_max = 1.5
    trainer.stage2_phase2_start_step = 50_000
    trainer.stage2_phase3_start_step = 100_000
    trainer.stage2_phase2_generator_lr = 2e-7
    trainer.stage2_phase3_generator_lr = 1e-7
    trainer.stage2_phase3_gan_adversarial_max = 1e-4
    trainer.stage2_phase3_gan_feature_max = 1.0
    trainer.generator_hold_steps = 5_000
    trainer.generator_hold_lr = 1e-7
    trainer.generator_freeze_steps = 2_000
    trainer.generator_hold_base_lrs = (5e-7,)
    trainer.optim = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": 5e-7}]),
        sync_warmup_lrs_from_optimizer=lambda: None,
    )
    model = SimpleNamespace(
        adversarial_loss_weight=0.,
        feature_loss_weight=0.,
    )
    trainer.soundstream = model
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda value: value)

    trainer.update_gan_weights(99_999)
    assert model.adversarial_loss_weight == pytest.approx(2e-4)
    assert model.feature_loss_weight == pytest.approx(1.5)
    assert trainer.cap_generator_lr_for_retention(50_000) == pytest.approx(2e-7)

    trainer.update_gan_weights(100_000)
    assert model.adversarial_loss_weight == pytest.approx(1e-4)
    assert model.feature_loss_weight == pytest.approx(1.0)
    assert trainer.cap_generator_lr_for_retention(100_000) == pytest.approx(1e-7)


def test_distributed_test_report_keeps_formant_and_stft_metrics():
    source = (ROOT / "train_soundstream.py").read_text(encoding="utf-8")
    start = source.index("        metric_names = (")
    end = source.index("        local_num_samples = (", start)
    metric_block = source[start:end]

    for metric in (
        "spectral_envelope_fine",
        "spectral_envelope_coarse",
        "formant_f1_mae_hz",
        "formant_f2_mae_hz",
        "formant_f3_mae_hz",
        "stft_scale_512",
        "stft_scale_1024",
        "stft_scale_2048",
    ):
        assert repr(metric) in metric_block


class GeneratorOnlyModel:
    decoder_upsample_mode = "convtranspose"
    decoder_linear_upsample_kernel_min = 4
    decoder_interpolation_mode = "linear"
    decoder_split_first_upsample = False
    encoder_depthwise_separable_blocks = (1, 2, 3)
    encoder_depthwise_separable_revision = 3
    encoder_low_rank_pointwise_ranks = (
        (0, 0), (8, 16), (16, 32), (32, 64)
    )
    num_quantizers = 16
    codebook_size = 16
    codebook_dim = 64

    def load_generator_state_dict(self, state_dict):
        assert state_dict == {}
        return []


class LinearGeneratorOnlyModel(GeneratorOnlyModel):
    """Compatibility fixture for checkpoints using interpolated upsampling."""

    decoder_upsample_mode = "linear"


def matching_config(**overrides):
    config = {
        "decoder_upsample_mode": "convtranspose",
        "decoder_linear_upsample_kernel_min": 4,
        "decoder_interpolation_mode": "linear",
        "decoder_split_first_upsample": False,
        "encoder_depthwise_separable_blocks": (1, 2, 3),
        "encoder_depthwise_separable_revision": 3,
        "encoder_low_rank_pointwise_ranks": (
            (0, 0), (8, 16), (16, 32), (32, 64)
        ),
        "rq_num_quantizers": 16,
        "codebook_size": 16,
        "codebook_dim": 64,
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

    assert config["encoder_depthwise_separable_revision"] == 3


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


def test_generator_checkpoint_rejects_old_rvq_shape(tmp_path):
    checkpoint = tmp_path / "rvq-8x256.pt"
    save_generator_checkpoint(
        checkpoint,
        matching_config(rq_num_quantizers=8, codebook_size=256),
    )

    with pytest.raises(ValueError, match="RVQ topology mismatch"):
        load_model_weights_only(
            GeneratorOnlyModel(), checkpoint, generator_only=True
        )


def test_generator_checkpoint_rejects_missing_block2_dscnn(tmp_path):
    checkpoint = tmp_path / "blocks-3-4-only.pt"
    save_generator_checkpoint(
        checkpoint,
        matching_config(encoder_depthwise_separable_blocks=(2, 3)),
    )

    with pytest.raises(ValueError, match="DSCNN topology mismatch"):
        load_model_weights_only(
            GeneratorOnlyModel(), checkpoint, generator_only=True
        )


def test_generator_checkpoint_rejects_decoder_interpolation_mismatch(tmp_path):
    checkpoint = tmp_path / "cubic.pt"
    save_generator_checkpoint(
        checkpoint,
        matching_config(
            decoder_upsample_mode="linear",
            decoder_interpolation_mode="cubic",
        ),
    )

    with pytest.raises(ValueError, match="decoder interpolation mismatch"):
        load_model_weights_only(
            LinearGeneratorOnlyModel(), checkpoint, generator_only=True
        )
