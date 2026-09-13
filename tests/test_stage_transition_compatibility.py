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
        generator_waveform_discr_loss_weights=(1.,),
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


def test_projection_only_path_uses_paired_linear_maps_without_rvq_codes():
    from audiolm_pytorch.soundstream import SoundStream

    model = SoundStream(
        channels=4,
        channel_mults=(2, 2),
        strides=(2, 2),
        codebook_dim=8,
        rq_lookup_dim=4,
        codebook_size=16,
        rq_num_quantizers=2,
        discr_multi_scales=(1,),
        generator_waveform_discr_loss_weights=(1.,),
        pad_mode="constant",
        rq_projection_only=True,
    )
    latent, indices, commitment = model(
        torch.randn(1, 256),
        return_encoded=True,
    )

    assert latent.shape == (1, 64, 8)
    assert torch.all(indices == -1)
    assert commitment.item() == 0
    assert model.rq_input_projection.weight.shape == (4, 8)
    assert model.rq_output_projection.weight.shape == (8, 4)
    assert model.configs["rq_projection_only"] is True


def test_rvq_pipeline_runs_projection_gate_before_calibration():
    launcher = (ROOT / "run_bypass_rvq_stage1_stage2.sh").read_text(
        encoding="utf-8"
    )
    q0 = launcher.index('Q0 PCA projection-only pretraining')
    b1 = launcher.index('B1 RVQ calibration with codec frozen')
    assert q0 < b1
    assert '--rvq-projection-only' in launcher
    assert 'latent_pca_report.json' in launcher
    assert '--init-checkpoint "$RVQ_PROJECTION_CKPT"' in launcher
    assert 'RVQ_LOOKUP_DIM="${RVQ_LOOKUP_DIM:-32}"' in launcher
    assert 'best_rvq_projection.pt' in launcher
    assert '--rvq-projection-latent-mse-weight' in launcher
    assert '--rvq-projection-latent-cosine-weight' in launcher


def test_q0_has_independent_checkpoint_and_fixed_decoder_scale_paths():
    trainer = (ROOT / "audiolm_pytorch" / "trainer.py").read_text(
        encoding="utf-8"
    )
    entrypoint = (ROOT / "train_soundstream.py").read_text(encoding="utf-8")

    assert "if self.rvq_projection_only:" in trainer
    assert "best_rvq_projection.pt" in trainer
    assert "projection_fidelity_loss" in trainer
    assert "0\n            if args.rvq_projection_only" in entrypoint
    assert "if args.rvq_projection_only or args.stage in (" in entrypoint


ROOT = Path(__file__).resolve().parents[1]


def test_ac320_is_diagnostic_only():
    """The retired absolute gate must not silently return."""
    assert not hasattr(train_soundstream, "AC320_ABSOLUTE_GATE_STAGES")


def test_stage1_formant_training_schedule():
    stage1 = STAGE_DEFAULTS["recon_pretrain"]

    assert stage1["wave_mse_loss_weight"] == pytest.approx(0.10)
    assert stage1["energy_loss_weight"] == pytest.approx(0.)
    assert stage1["spectral_envelope_loss_weight"] == pytest.approx(0.)
    assert stage1["spectral_envelope_loss_start_steps"] == 0
    assert stage1["spectral_envelope_loss_warmup_steps"] == 0
    assert stage1["formant_peak_loss_weight"] == pytest.approx(0.)
    assert stage1["formant_peak_loss_start_steps"] == 0
    assert stage1["formant_peak_loss_warmup_steps"] == 0
    assert stage1["stft_recon_loss_weight"] == pytest.approx(0.05)
    assert stage1["stft_recon_loss_start_steps"] == 5_000
    assert stage1["stft_recon_loss_warmup_steps"] == 15_000
    assert stage1["si_sdr_loss_weight"] == pytest.approx(0.07)
    assert stage1["si_sdr_loss_start_steps"] == 15_000
    assert stage1["si_sdr_loss_warmup_steps"] == 15_000

    launcher = (ROOT / "run_bypass_rvq_stage1_stage2.sh").read_text(
        encoding="utf-8"
    )
    assert 'BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS="${BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS:-60000}"' in launcher
    assert '--early-stopping-min-steps "$BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS"' in launcher
    assert stage1["voiced_highband_loss_weight"] == pytest.approx(0.)
    assert stage1["upper_highband_loss_weight"] == pytest.approx(0.)


def test_spectral_refine_matches_formant_refinement_profile():
    refine = STAGE_DEFAULTS["spectral_refine"]

    assert "spectral_refine" in QUALITY_RETENTION_STAGES
    assert refine["lr"] == pytest.approx(1e-5)
    assert refine["encoder_lr"] == pytest.approx(2e-6)
    assert refine["use_ema"] is False
    assert refine["wave_mse_loss_weight"] == pytest.approx(0.10)
    assert refine["energy_loss_weight"] == pytest.approx(0.)
    assert refine["patience"] == 30
    assert refine["spectral_envelope_loss_weight"] == pytest.approx(0.10)
    assert refine["formant_peak_loss_weight"] == pytest.approx(0.05)
    assert refine["formant_peak_loss_warmup_steps"] == 5_000
    assert refine["stft_recon_loss_weight"] == pytest.approx(0.05)
    assert refine["frame_phase_loss_weight"] == pytest.approx(0.)
    assert refine["decoder_x8_residual_scale_target"] == pytest.approx(1.0)
    assert refine["decoder_x8_residual_scale_ramp_steps"] == 3_000


def test_a1_a15_use_random_train_and_deterministic_validation_crops():
    source = (ROOT / "train_soundstream.py").read_text(encoding="utf-8")
    trainer_source = (ROOT / "audiolm_pytorch" / "trainer.py").read_text(
        encoding="utf-8"
    )

    assert 'dataset_fixed_crop=(args.stage == "overfit")' in source
    assert "base_dataset.deterministic_crop_indices = fixed_indices" in trainer_source


def test_formant_train_and_eval_masks_are_reported_separately():
    source = (ROOT / "audiolm_pytorch" / "soundstream.py").read_text(
        encoding="utf-8"
    )

    assert "('f2', 1000., 2500." in source
    assert "0.12, -9.0, 0.16, -6.0" in source
    assert "0.16, -15.0, 0.22, -12.0" in source
    assert "f'{name}_valid_{mask_name}'" in source
    assert "('trainmask', train_confidence)" in source
    assert "('evalmask', eval_confidence)" in source


def test_loss_gradient_diagnostic_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_soundstream.py"])
    args = train_soundstream.parse_args()

    assert args.loss_grad_diagnostics_every == 1_000
    assert args.decoder_upsample_mode == "convtranspose"


def test_stage2_uses_decoder_then_encoder_tail_150k_schedule():
    stage2 = STAGE_DEFAULTS["gan_pretrain"]

    assert stage2["steps"] == 150_000
    assert stage2["encoder_lr"] == pytest.approx(1e-7)
    assert stage2["patience"] is None
    assert stage2["gan_start"] == 2_000
    assert stage2["gan_ramp"] == 15_000
    assert stage2["waveform_discr_update_every"] == (2, 4, 4)
    assert stage2["waveform_discr_loss_weights"] == (1.0, 0.25, 0.25)
    assert stage2["stft_discr_loss_weight"] == pytest.approx(0.5)
    assert stage2["gan_feature_max"] == pytest.approx(1.0)
    assert stage2["stft_recon_loss_weight"] == pytest.approx(0.02)
    assert stage2["spectral_envelope_loss_weight"] == pytest.approx(0.02)
    assert stage2["formant_peak_loss_weight"] == pytest.approx(0.)
    assert stage2["voiced_highband_loss_weight"] == pytest.approx(0.)
    assert stage2["voiced_hf_retention_loss_weight"] == pytest.approx(0.01)
    assert stage2["upper_highband_loss_weight"] == pytest.approx(0.)
    assert stage2["active_spectral_detail_loss_weight"] == pytest.approx(0.)

    launcher = (ROOT / "run_stage1_stage15_stage2.sh").read_text(
        encoding="utf-8"
    )
    assert '--stage2-encoder-unfreeze-step 10000' in launcher
    assert '--stage2-encoder-trainable-from-block 3' in launcher
    assert '--stage2-encoder-lr 1e-7' in launcher
    assert '--stage2-phase2-start-step 2000' in launcher
    assert '--stage2-phase3-start-step 10000' in launcher
    assert '--no-stage2-quality-hard-stop' in launcher
    assert '--early-stopping-patience' not in launcher

    soundstream_source = (
        ROOT / "audiolm_pytorch" / "soundstream.py"
    ).read_text(encoding="utf-8")
    assert "total_branch_weight = max(sum(branch_weights), 1e-8)" in soundstream_source
    assert "weighted_adversarial_branches" in soundstream_source
    assert "weighted_feature_branches" in soundstream_source


def test_stateful_stages_have_distinct_pre_and_post_rvq_roles():
    pre_rvq = STAGE_DEFAULTS["stream_finetune"]
    final = STAGE_DEFAULTS["stream_finetune_long"]

    assert pre_rvq["encoder_lr"] == pytest.approx(1e-7)
    assert pre_rvq["lr"] == pytest.approx(5e-7)
    assert final["steps"] == 10_000
    assert final["lr"] == pytest.approx(2e-7)
    assert final["gan_adversarial_max"] == pytest.approx(0.)
    assert final["gan_feature_max"] == pytest.approx(0.)

    launcher = (ROOT / "run_bypass_rvq_stage1_stage2.sh").read_text(
        encoding="utf-8"
    )
    assert 'ENABLE_PRE_RVQ_STATE_FT="${ENABLE_PRE_RVQ_STATE_FT:-0}"' in launcher
    assert 'FINAL_STATE_STEPS="${FINAL_STATE_STEPS:-10000}"' in launcher
    assert "FINAL_STATE_STEPS <= 0" in launcher
    assert "post-RVQ state alignment is mandatory" in launcher
    assert "--reinitialize-rvq-from-bypass-checkpoint" in launcher
    assert '--stage stream_finetune --results-dir "$PRE_RVQ_STATE_DIR"' in launcher
    assert '--stage stream_finetune_long --results-dir "$FINAL_STATE_DIR"' in launcher
    assert launcher.index('A2.5 pre-RVQ stateful alignment') < launcher.index(
        'B1 RVQ calibration with codec frozen'
    )
    assert launcher.index('B2 RVQ GAN decoder adaptation') < launcher.index(
        'Final post-RVQ stateful Decoder fine-tune'
    )
    assert '--num-train-steps "$FINAL_STATE_STEPS"' in launcher
    assert '--no-bypass-rvq-during-training' in launcher


def test_pipeline_inserts_joint_rvq_adaptation_and_shortens_b2():
    launcher = (ROOT / "run_bypass_rvq_stage1_stage2.sh").read_text(
        encoding="utf-8"
    )
    assert 'RVQ_CALIBRATION_STEPS="${RVQ_CALIBRATION_STEPS:-1000}"' in launcher
    assert 'RVQ_JOINT_ADAPT_STEPS="${RVQ_JOINT_ADAPT_STEPS:-30000}"' in launcher
    assert 'RVQ_STAGE2_STEPS="${RVQ_STAGE2_STEPS:-50000}"' in launcher
    assert 'B1.5 joint STE quantization adaptation' in launcher
    assert '--rvq-warm-in-steps 10000' in launcher
    assert '--rvq-codebook-balance-loss-weight 0.002' in launcher
    assert '--rvq-quantization-error-loss-weight 0.05' in launcher
    assert '--rvq-continuous-teacher-loss-weight 0.10' in launcher
    assert '--rvq-codebook-balance-target-perplexity 64' in launcher
    assert '--stage2-encoder-unfreeze-step -1' in launcher
    assert '--early-stopping-patience 20 --stage2-quality-hard-stop' in launcher
    assert launcher.index('B1 checkpoint=') < launcher.index(
        'B1.5 joint STE quantization adaptation'
    ) < launcher.index('B2 RVQ GAN decoder adaptation')

    source = (ROOT / "train_soundstream.py").read_text(encoding="utf-8")
    assert 'freeze_encoder_after_step=(15_000 if rvq_joint_adapt else None)' in source
    model_source = (ROOT / "audiolm_pytorch" / "soundstream.py").read_text(
        encoding="utf-8"
    )
    assert 'def rvq_codebook_balance_loss(self, encoded, indices):' in model_source
    assert 'def rvq_quantization_error_loss(self, encoded, indices):' in model_source
    assert 'x = continuous_x + alpha * (x - continuous_x)' in model_source
    assert 'teacher_latent = self.rq_output_projection' in model_source


def test_pipeline_can_start_from_bypass_formant_refine_checkpoint():
    launcher = (ROOT / "run_bypass_rvq_stage1_stage2.sh").read_text(
        encoding="utf-8"
    )

    assert 'BYPASS_STAGE1_CKPT="${BYPASS_STAGE1_CKPT:-}"' in launcher
    assert '"$START_PHASE" != "bypass_formant_refine"' in launcher
    assert 'START_PHASE=bypass_formant_refine requires an existing BYPASS_STAGE1_CKPT' in launcher
    assert 'BYPASS_S1_CKPT="$(readlink -f "$BYPASS_STAGE1_CKPT")"' in launcher
    assert 'Skipping A1; starting A1.5 from $BYPASS_S1_CKPT' in launcher


def test_stage2_phase_boundaries_adjust_lr_and_gan_weights():
    trainer = object.__new__(SoundStreamTrainer)
    torch.nn.Module.__init__(trainer)
    trainer.enable_gan = True
    trainer.gan_start_step = 2_000
    trainer.gan_ramp_steps = 15_000
    trainer.gan_adversarial_max = 2e-4
    trainer.gan_feature_max = 1.0
    trainer.stage2_phase2_start_step = 2_000
    trainer.stage2_phase3_start_step = 10_000
    trainer.stage2_phase2_generator_lr = 5e-7
    trainer.stage2_phase3_generator_lr = 5e-7
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

    trainer.update_gan_weights(9_999)
    gan_progress = (9_999 - 2_000) / 15_000
    assert model.adversarial_loss_weight == pytest.approx(2e-4 * gan_progress)
    assert model.feature_loss_weight == pytest.approx(1.0 * gan_progress)
    assert trainer.cap_generator_lr_for_retention(2_000) == pytest.approx(5e-7)

    trainer.update_gan_weights(10_000)
    assert model.adversarial_loss_weight == pytest.approx(1e-4)
    assert model.feature_loss_weight == pytest.approx(1.0)
    assert trainer.cap_generator_lr_for_retention(10_000) == pytest.approx(5e-7)


@pytest.mark.parametrize(
    ("completed_step", "instantaneous_lr"),
    (
        (0, 4e-7),
        (500, 1.004e-4),
        (999, 2e-4),
    ),
)
def test_stage1_resume_inside_warmup_preserves_undampened_base_lr(
    completed_step,
    instantaneous_lr,
):
    trainer = object.__new__(SoundStreamTrainer)
    torch.nn.Module.__init__(trainer)
    sync_calls = []
    trainer.optim = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": instantaneous_lr}]),
        warmup=SimpleNamespace(lrs=[instantaneous_lr]),
        sync_warmup_lrs_from_optimizer=lambda: sync_calls.append(True),
    )
    trainer.plateau_scheduler = SimpleNamespace(_last_lr=[])
    trainer.best_checkpoint_metric = "recon_pretrain"
    trainer.generator_warmup_steps = 1_000
    trainer.generator_hold_base_lrs = (2e-4,)
    trainer.plateau_lr_min_lr = 1e-5
    trainer.plateau_lr_start_steps = 60_000
    trainer.print = lambda *_: None

    trainer.sync_plateau_scheduler_from_optimizer(completed_step=completed_step)

    assert trainer.optim.optimizer.param_groups[0]["lr"] == pytest.approx(
        instantaneous_lr
    )
    assert trainer.optim.warmup.lrs == pytest.approx([2e-4])
    assert trainer.plateau_scheduler._last_lr == pytest.approx([instantaneous_lr])
    assert sync_calls == []


def test_stage1_resume_after_warmup_repairs_invalid_lr_and_syncs_state():
    trainer = object.__new__(SoundStreamTrainer)
    torch.nn.Module.__init__(trainer)
    sync_calls = []
    optimizer = SimpleNamespace(param_groups=[{"lr": 4e-7}])
    warmup_state = SimpleNamespace(lrs=[4e-7])

    def sync_warmup_lrs():
        sync_calls.append(True)
        warmup_state.lrs = [group["lr"] for group in optimizer.param_groups]

    trainer.optim = SimpleNamespace(
        optimizer=optimizer,
        warmup=warmup_state,
        sync_warmup_lrs_from_optimizer=sync_warmup_lrs,
    )
    trainer.plateau_scheduler = SimpleNamespace(_last_lr=[])
    trainer.best_checkpoint_metric = "recon_pretrain"
    trainer.generator_warmup_steps = 1_000
    trainer.generator_hold_base_lrs = (2e-4,)
    trainer.plateau_lr_min_lr = 1e-5
    trainer.plateau_lr_start_steps = 60_000
    trainer.print = lambda *_: None

    trainer.sync_plateau_scheduler_from_optimizer(completed_step=1_000)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert warmup_state.lrs == pytest.approx([2e-4])
    assert trainer.plateau_scheduler._last_lr == pytest.approx([2e-4])
    assert sync_calls == [True]


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
    num_quantizers = 8
    codebook_size = 256
    codebook_dim = 64
    rq_lookup_dim = 16
    rq_use_cosine_sim = True
    loaded_without_rvq = False

    def load_generator_state_dict(self, state_dict):
        assert state_dict == {}
        return []

    def load_state_dict_without_rvq(self, state_dict, *, include_discriminators):
        assert state_dict == {}
        assert include_discriminators is False
        self.loaded_without_rvq = True
        return ("rq.rvqs.0.layers.0._codebook.embed",)


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
        "rq_num_quantizers": 8,
        "codebook_size": 256,
        "codebook_dim": 64,
        "rq_lookup_dim": 16,
        "rq_use_cosine_sim": True,
    }
    config.update(overrides)
    return config


def test_current_rvq_profile_is_8_by_256_by_16_lookup_and_3p2kbps():
    config = matching_config()
    frame_rate = 16000 / 320
    bits_per_index = 8

    assert config["rq_num_quantizers"] == 8
    assert config["codebook_size"] == 256
    assert config["codebook_dim"] == 64
    assert config["rq_lookup_dim"] == 16
    assert config["rq_use_cosine_sim"] is True
    assert frame_rate * config["rq_num_quantizers"] * bits_per_index == 3200
    assert (
        config["rq_num_quantizers"] * config["codebook_size"]
        * config["rq_lookup_dim"] + 2 * config["codebook_dim"]
        * config["rq_lookup_dim"]
        == 34816
    )


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
        matching_config(
            rq_num_quantizers=16,
            codebook_size=16,
            rq_lookup_dim=64,
            rq_use_cosine_sim=False,
        ),
    )

    with pytest.raises(ValueError, match="RVQ topology mismatch"):
        load_model_weights_only(
            GeneratorOnlyModel(), checkpoint, generator_only=True
        )


def test_generator_checkpoint_rebuilds_rvq_from_confirmed_bypass(tmp_path):
    checkpoint = tmp_path / "bypass-rvq-10x64.pt"
    save_generator_checkpoint(
        checkpoint,
        matching_config(
            rq_num_quantizers=10,
            codebook_size=64,
            bypass_rvq=True,
        ),
    )
    model = GeneratorOnlyModel()

    config = load_model_weights_only(
        model,
        checkpoint,
        generator_only=True,
        reinitialize_rvq_from_bypass=True,
    )

    assert model.loaded_without_rvq is True
    assert config["bypass_rvq"] is True


def test_rvq_rebuild_rejects_checkpoint_not_marked_as_bypass(tmp_path):
    checkpoint = tmp_path / "quantized-rvq-10x64.pt"
    save_generator_checkpoint(
        checkpoint,
        matching_config(
            rq_num_quantizers=10,
            codebook_size=64,
            bypass_rvq=False,
        ),
    )

    with pytest.raises(ValueError, match="records bypass_rvq=True"):
        load_model_weights_only(
            GeneratorOnlyModel(),
            checkpoint,
            generator_only=True,
            reinitialize_rvq_from_bypass=True,
        )


def test_rvq_rebuild_rejects_checkpoint_without_config(tmp_path):
    checkpoint = tmp_path / "missing-config.pt"
    torch.save({"model": {}}, checkpoint)

    with pytest.raises(ValueError, match="metadata proving bypass_rvq=True"):
        load_model_weights_only(
            GeneratorOnlyModel(),
            checkpoint,
            generator_only=True,
            reinitialize_rvq_from_bypass=True,
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
