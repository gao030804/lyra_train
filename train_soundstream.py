from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

PROJECT_DIR = Path(__file__).resolve().parent
RUNTIME_TMP_DIR = PROJECT_DIR / ".runtime-tmp"
RUNTIME_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(RUNTIME_TMP_DIR))
os.environ.setdefault("TEMP", str(RUNTIME_TMP_DIR))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
tempfile.tempdir = str(RUNTIME_TMP_DIR)

from audiolm_pytorch import FrameStreamingSoundStream, SoundStream, SoundStreamTrainer

DEFAULT_AUDIO_DIR = (
    PROJECT_DIR
    / "data"
    / "librispeech"
    / "LibriSpeech"
    / "train-clean-100"
)

STAGE_RESULTS_DIRS = {
    "overfit": PROJECT_DIR / "results" / "overfit-lowrank-dscnn-fp-64d-9q128-l32",
    "recon_pretrain": PROJECT_DIR / "results" / "recon-pretrain-lowrank-dscnn-fp-64d-9q128-l32",
    "spectral_refine": PROJECT_DIR / "results" / "spectral-refine-lowrank-dscnn-fp-64d-9q128-l32",
    "gan_pretrain": PROJECT_DIR / "results" / "gan-pretrain-lowrank-dscnn-fp-64d-9q128-l32",
    "stream_finetune": PROJECT_DIR / "results" / "stream-finetune-lowrank-dscnn-fp-64d-9q128-l32",
    "stream_finetune_long": PROJECT_DIR / "results" / "stream-finetune-long-lowrank-dscnn-fp-64d-9q128-l32",
    "hardware_qat_finetune": PROJECT_DIR / "results" / "hardware-qat-encoder-lowrank-dscnn-int8-64d-9q128-l32",
}

RECONSTRUCTION_STAGES = frozenset((
    "recon_pretrain",
    "spectral_refine",
    "gan_pretrain",
    "stream_finetune",
    "stream_finetune_long",
    "hardware_qat_finetune",
))
# Stateful alignment is deliberately reconstruction-only. Adversarial training
# finishes in A2; neither the optional pre-RVQ nor final post-RVQ state pass
# reopens the GAN objective.
GAN_STAGES = frozenset(("gan_pretrain",))
QUALITY_RETENTION_STAGES = frozenset((
    "spectral_refine",
    "gan_pretrain",
    "stream_finetune",
    "stream_finetune_long",
    "hardware_qat_finetune",
))

STAGE_DEFAULTS = {
    "overfit": dict(
        steps=5_000, batch_size=4, segment_seconds=2.,
        save_every=500, eval_every=100, min_steps=0, patience=30,
        lr=3e-4, discr_lr=None, ema_beta=0.95,
        ema_update_after_step=0, ema_update_every=1,
        click_loss_weight=0., jump_loss_weight=0.,
        transient_loss_warmup_steps=0,
        spectral_envelope_loss_weight=0.,
        voiced_highband_loss_weight=0.,
        stft_recon_loss_weight=0.,
        gan_start=0, gan_ramp=0,
    ),
    "recon_pretrain": dict(
        steps=150_000, batch_size=4, segment_seconds=4.,
        save_every=2_000, eval_every=250, min_steps=60_000, patience=40,
        lr=2e-4, discr_lr=None, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        # The highband-k5 run showed sustained EMA lag followed by multi-layer
        # EMA codebook collapse while the online codec remained healthy near
        # its validation peak.  Stage 1 therefore selects and exports online
        # weights only; later stages may still opt into EMA independently.
        use_ema=False,
        # A very small first-difference term aligns Stage-1 optimization with
        # the click-excess clean gate without suppressing normal consonant
        # transients.  Ramp it quickly enough to affect checkpoint selection.
        click_loss_weight=0.002, jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.03,
        wave_mse_loss_weight=0.10, energy_loss_weight=0.,
        transient_loss_warmup_steps=5_000,
        # A1 is deliberately limited to stable base reconstruction.  Envelope,
        # formant and high-band objectives are measured during validation but
        # receive no gradient until the dedicated A1.5 refinement pass.
        spectral_envelope_loss_weight=0.,
        spectral_envelope_loss_start_steps=0,
        spectral_envelope_loss_warmup_steps=0,
        # Peak locations remain diagnostic-only in A1.
        formant_peak_loss_weight=0.,
        formant_peak_loss_start_steps=0,
        formant_peak_loss_warmup_steps=0,
        # Speech-clarity and Nyquist-edge shaping belong to A1.5/A2, avoiding
        # redundant spectral gradients in the base reconstruction stage.
        voiced_highband_loss_weight=0.,
        upper_highband_loss_weight=0.,
        upper_highband_energy_deficit_weight=0.,
        upper_highband_energy_margin_db=0.50,
        upper_highband_loss_start_steps=5_000,
        upper_highband_loss_warmup_steps=15_000,
        voiced_highband_loss_start_steps=5_000,
        voiced_highband_loss_warmup_steps=15_000,
        stft_recon_loss_weight=0.05,
        stft_recon_loss_start_steps=5_000,
        stft_recon_loss_warmup_steps=15_000,
        si_sdr_loss_weight=0.07,
        si_sdr_loss_start_steps=15_000,
        si_sdr_loss_warmup_steps=15_000,
        gan_start=0, gan_ramp=0,
        gan_adversarial_max=0., gan_feature_max=0.,
    ),
    # Stage 1.5 is the formant-refinement pass.  In bypass-RVQ runs it keeps
    # the continuous latent path, updates Decoder conservatively, and gives
    # Encoder a smaller LR so F2/F3 information can still enter the latent.
    "spectral_refine": dict(
        steps=20_000, batch_size=4, segment_seconds=4.,
        save_every=1_000, eval_every=250, min_steps=5_000, patience=30,
        lr=1e-5, encoder_lr=2e-6, discr_lr=None, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        use_ema=False,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.03,
        wave_mse_loss_weight=0.10, energy_loss_weight=0.,
        transient_loss_warmup_steps=0,
        # The previous 0.12 envelope objective barely moved validation
        # formants while competing with waveform quality.  Keep it as a
        # structural anchor and let the independently scheduled F2/F3 peak
        # terms do the targeted refinement.
        spectral_envelope_loss_weight=0.10,
        spectral_envelope_loss_start_steps=0,
        spectral_envelope_loss_warmup_steps=0,
        formant_peak_loss_weight=0.05,
        formant_peak_loss_start_steps=0,
        formant_peak_loss_warmup_steps=5_000,
        voiced_highband_loss_weight=0.02,
        voiced_highband_loss_start_steps=0,
        voiced_highband_loss_warmup_steps=5_000,
        upper_highband_loss_weight=0.,
        upper_highband_energy_deficit_weight=0.,
        upper_highband_energy_margin_db=0.50,
        upper_highband_loss_start_steps=0,
        upper_highband_loss_warmup_steps=5_000,
        stft_recon_loss_weight=0.05,
        stft_recon_loss_start_steps=0,
        stft_recon_loss_warmup_steps=5_000,
        frame_phase_loss_weight=0.,
        frame_phase_loss_start_steps=0,
        frame_phase_loss_warmup_steps=5_000,
        decoder_x8_residual_scale_target=1.0,
        decoder_x8_residual_scale_ramp_steps=3_000,
        si_sdr_loss_weight=0.05,
        si_sdr_loss_start_steps=0,
        si_sdr_loss_warmup_steps=0,
        gan_start=0, gan_ramp=0,
        gan_adversarial_max=0., gan_feature_max=0.,
    ),
    "gan_pretrain": dict(
        steps=150_000, batch_size=4, segment_seconds=4.,
        save_every=5_000, eval_every=500, min_steps=0, patience=None,
        early_stopping_min_delta=0.003,
        # Keep Stage-2 decoder updates smaller than discriminator updates so a
        # freshly initialized GAN cannot quickly displace the selected codec.
        lr=5e-7, encoder_lr=1e-7, discr_lr=5e-7, stft_discr_lr=2.5e-7,
        waveform_discr_lrs=(5e-7, 5e-7, 2.5e-7),
        waveform_discr_update_every=(2, 4, 4),
        waveform_discr_loss_weights=(1.0, 0.25, 0.25),
        stft_discr_update_every=4,
        stft_discr_loss_weight=0.5,
        ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        use_ema=False,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.03,
        transient_loss_warmup_steps=0,
        wave_mse_loss_weight=0.10, energy_loss_weight=0.,
        # B2 inherits the B1.5 waveform / Mel / SI-SDR / MR-STFT fidelity
        # objective.  Keep the formant-envelope auxiliary loss disabled here;
        # the frozen B1.5 teacher already retains that behavior without adding
        # a competing spectral target during GAN adaptation.
        spectral_envelope_loss_weight=0.,
        spectral_envelope_loss_start_steps=0,
        spectral_envelope_loss_warmup_steps=0,
        formant_peak_loss_weight=0.,
        formant_peak_loss_start_steps=0,
        formant_peak_loss_warmup_steps=5_000,
        # B2 begins from an already polished RVQ decoder.  Preserve the B1.5
        # reconstruction objective and let the staged GAN add texture only
        # after the reconstruction-only retention phase.
        voiced_highband_loss_weight=0.,
        voiced_hf_retention_loss_weight=0.,
        voiced_highband_loss_start_steps=0,
        voiced_highband_loss_warmup_steps=0,
        # Nyquist-edge and broad active-spectrum objectives are disabled in
        # the general GAN pass; they can be reintroduced only in a targeted
        # diagnostic experiment.
        upper_highband_loss_weight=0.,
        upper_highband_energy_deficit_weight=0.,
        upper_highband_energy_margin_db=0.50,
        upper_highband_loss_start_steps=0,
        upper_highband_loss_warmup_steps=0,
        active_spectral_detail_loss_weight=0.,
        active_spectral_detail_loss_start_steps=0,
        active_spectral_detail_loss_warmup_steps=0,
        stft_recon_loss_weight=0.05,
        stft_recon_loss_start_steps=0,
        stft_recon_loss_warmup_steps=2_500,
        # Keep frame-leakage validation diagnostics, but disable the training
        # loss because the latest diagnostic run increased ac_320.
        frame_phase_loss_weight=0.,
        frame_phase_loss_start_steps=0,
        frame_phase_loss_warmup_steps=0,
        si_sdr_loss_weight=0.12,
        si_sdr_loss_start_steps=0,
        si_sdr_loss_warmup_steps=0,
        # Phase A is reconstruction-only. Phase B introduces only the full-
        # resolution waveform discriminator, and the full discriminator set
        # is enabled later by the trainer.  The long ramp and reduced feature
        # matching weight prevent GAN gradients from displacing B1.5.
        gan_start=3_000, gan_ramp=40_000,
        gan_adversarial_max=1e-4, gan_feature_max=0.20,
    ),
    "stream_finetune": dict(
        steps=20_000, batch_size=4, segment_seconds=4.,
        save_every=1_000, eval_every=250, min_steps=5_000, patience=30,
        lr=5e-7, encoder_lr=1e-7, discr_lr=None, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        use_ema=False,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.03,
        transient_loss_warmup_steps=0,
        spectral_envelope_loss_weight=0.05,
        voiced_highband_loss_weight=0.07,
        voiced_hf_retention_loss_weight=0.02,
        stft_recon_loss_weight=0.,
        si_sdr_loss_weight=0.07,
        boundary_loss_weight=0.02,
        boundary_loss_start_steps=2_000,
        boundary_loss_warmup_steps=3_000,
        # Streaming consistency is a waveform-only teacher loss.  Keep it
        # light: ordinary Mel / high-band objectives already preserve the
        # spectrum, while reusing the log-Mel reconstruction loss here made
        # the training value orders of magnitude larger than validation.
        stream_consistency_loss_weight=0.10,
        stream_consistency_loss_start_steps=0,
        stream_consistency_loss_warmup_steps=2_000,
        gan_start=0, gan_ramp=0,
        gan_adversarial_max=0., gan_feature_max=0.,
    ),
    "stream_finetune_long": dict(
        # Final state alignment is a short calibration from an already-good
        # B2 checkpoint, not another spectral reconstruction stage.
        steps=3_000, batch_size=2, segment_seconds=4.,
        save_every=500, eval_every=250, min_steps=500, patience=3,
        lr=5e-8, discr_lr=None, stft_discr_lr=None, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        use_ema=False,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.005,
        transient_loss_warmup_steps=0,
        spectral_envelope_loss_weight=0.005,
        voiced_highband_loss_weight=0.005,
        voiced_hf_retention_loss_weight=0.002,
        stft_recon_loss_weight=0.,
        si_sdr_loss_weight=0.12,
        boundary_loss_weight=0.15,
        boundary_loss_start_steps=0,
        boundary_loss_warmup_steps=0,
        stream_consistency_loss_weight=0.75,
        stream_consistency_loss_start_steps=0,
        stream_consistency_loss_warmup_steps=0,
        waveform_recon_loss_weight=0.75,
        multi_spectral_recon_loss_weight=0.15,
        correlation_loss_weight=0.02,
        teacher_retention_weight=0.75,
        decoder_trainable_from_block=2,
        gan_start=0, gan_ramp=0,
        gan_adversarial_max=0., gan_feature_max=0.,
    ),
    "hardware_qat_finetune": dict(
        # Encoder-only hardware QAT from a validation-selected floating B2.
        steps=8_000, batch_size=4, segment_seconds=4.,
        save_every=500, eval_every=250, min_steps=1_500, patience=20,
        lr=2e-7, discr_lr=None, stft_discr_lr=None, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1, use_ema=False,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.,
        transient_loss_warmup_steps=0,
        spectral_envelope_loss_weight=0., voiced_highband_loss_weight=0.,
        stft_recon_loss_weight=0.025, si_sdr_loss_weight=0.05,
        waveform_recon_loss_weight=0.5,
        multi_spectral_recon_loss_weight=0.15,
        correlation_loss_weight=0.02,
        gan_start=0, gan_ramp=0,
        gan_adversarial_max=0., gan_feature_max=0.,
    ),
}


def stage1_lr_lambda(step: int) -> float:
    """Fallback piecewise LR multiplier for recon_pretrain after linear warmup.

    ReduceLROnPlateau is the default for stage 1. This fallback is used only
    when --no-stage1-plateau-lr is passed. The trainer applies warmup
    separately. With the recon_pretrain base LR of 2e-4 and warmup_steps=1000,
    this gives:
      0 - 1000: linear warmup to 2e-4
      1000 - 20000: 2e-4
      20000 - 35000: 1e-4
      35000+: 5e-5
    """
    if step < 20_000:
        return 1.0
    if step < 35_000:
        return 0.5
    return 0.25


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".webm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a staged configurable-bitrate streaming SoundStream speech codec."
    )
    parser.add_argument('--preceding-context-seconds', type=float, default=0.,
                        help='Real contiguous prefix excluded from waveform losses; use 0.4 for context-aware crops.')
    parser.add_argument('--stream-tbptt-frames', type=int, default=20,
                        help='Detach streaming activation caches every N frames; 0 retains the full graph.')

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_DEFAULTS),
        default="overfit",
        help="Training phase: overfit, reconstruction, GAN, or streaming fine-tuning.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional model-only checkpoint used to initialize a new stage.",
    )
    parser.add_argument(
        "--reinitialize-rvq-from-bypass-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load Encoder/Decoder (and, when applicable, discriminator) state "
            "from a checkpoint explicitly saved with bypass_rvq=True, discard "
            "all checkpoint rq.* tensors, and keep the current model's freshly "
            "initialized RVQ. Valid for RVQ projection pretraining or calibration."
        ),
    )
    parser.add_argument(
        "--predecessor-results-dir",
        type=Path,
        default=None,
        help=(
            "Optional results directory for the preceding stage. "
            "Uses best_selected.pt unless --init-checkpoint is provided."
        ),
    )
    parser.add_argument(
        "--overfit-files",
        type=int,
        default=10,
        help="Number of deterministic files used by the overfit diagnostic stage.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="Defaults depend on --stage.",
    )
    parser.add_argument(
        "--generator-lr",
        type=float,
        default=None,
        help=(
            "Optional generator learning-rate override. Useful for a fresh "
            "low-LR refinement initialized from a model-only checkpoint."
        ),
    )
    parser.add_argument(
        "--gan-adversarial-max",
        type=float,
        default=None,
        help=(
            "Optional maximum generator adversarial-loss weight override. "
            "The stage default is retained when omitted."
        ),
    )
    parser.add_argument(
        "--gan-feature-max",
        type=float,
        default=None,
        help=(
            "Optional maximum discriminator feature-matching weight override. "
            "The stage default is retained when omitted."
        ),
    )
    parser.add_argument('--hardware-qat-observer-start-step', type=int, default=0)
    parser.add_argument('--hardware-qat-start-step', type=int, default=1000,
                        help='First step of weight-only INT8 QAT.')
    parser.add_argument('--hardware-qat-activation-start-step', type=int, default=2000)
    parser.add_argument('--hardware-qat-warm-in-steps', type=int, default=600)
    parser.add_argument('--hardware-qat-block-interval-steps', type=int, default=600)
    parser.add_argument(
        '--hardware-qat-observer-freeze-step', type=int, default=1000
    )
    parser.add_argument('--hardware-qat-ema-decay', type=float, default=0.99)
    parser.add_argument(
        '--hardware-qat-observer', choices=('max', 'percentile'),
        default='percentile'
    )
    parser.add_argument('--hardware-qat-percentile', type=float, default=99.99)
    parser.add_argument(
        '--hardware-qat-validation-gated',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Require consecutive fixed-validation passes before enabling the next activation group.'
    )
    parser.add_argument('--hardware-qat-gate-required-passes', type=int, default=2)
    parser.add_argument('--hardware-qat-latent64-weight', type=float, default=0.5)
    parser.add_argument('--hardware-qat-latent32-weight', type=float, default=1.5)
    parser.add_argument('--hardware-qat-rvq-margin-weight', type=float, default=0.03)
    parser.add_argument('--hardware-qat-rvq-margin', type=float, default=0.05)
    parser.add_argument('--hardware-qat-rvq-margin-max', type=float, default=1.0)
    parser.add_argument(
        '--hardware-qat-sensitivity-scan',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--hardware-qat-fixed-scale-lr', type=float, default=1e-7)
    parser.add_argument('--hardware-qat-final-polish-step', type=int, default=5000)
    parser.add_argument('--hardware-qat-final-polish-lr', type=float, default=5e-8)
    parser.add_argument('--hardware-qat-max-latent32-nmse', type=float, default=0.03)
    parser.add_argument(
        '--hardware-qat-max-quantized-output-nmse', type=float, default=0.05
    )
    parser.add_argument('--hardware-qat-group-fail-patience', type=int, default=4)
    parser.add_argument('--hardware-qat-max-q00-index-flip', type=float, default=0.10)
    parser.add_argument('--hardware-qat-max-q01-index-flip', type=float, default=0.15)
    parser.add_argument(
        "--waveform-recon-loss-weight",
        type=float,
        default=None,
        help="Optional waveform L1 reconstruction-weight override.",
    )
    parser.add_argument(
        "--multi-spectral-recon-loss-weight",
        type=float,
        default=None,
        help="Optional multi-spectral Mel reconstruction-weight override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU batch size; defaults depend on --stage.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--save-model-every",
        type=int,
        default=None,
        help="Defaults depend on --stage.",
    )
    parser.add_argument(
        "--best-eval-every",
        type=int,
        default=None,
        help="Defaults depend on --stage.",
    )
    parser.add_argument(
        "--best-eval-batches",
        type=int,
        default=26,
        help="Number of fixed validation batches averaged for best checkpoint selection.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Validation checks without improvement before stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=None,
        help="Minimum validation-score decrease required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--early-stopping-min-steps",
        type=int,
        default=None,
        help="Minimum completed steps before patience can accumulate.",
    )
    parser.add_argument(
        "--save-results-every",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--grad-accum-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dl-num-workers",
        type=int,
        default=6,
        help="DataLoader worker processes per GPU. Use 0 for fully synchronous loading.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, PyTorch, data split, and dataloader shuffle.",
    )
    parser.add_argument(
        "--si-sdr-loss-weight",
        type=float,
        default=None,
        help="Maximum SI-SDR loss weight. Defaults depend on --stage.",
    )
    parser.add_argument(
        "--si-sdr-loss-start-steps",
        type=int,
        default=None,
        help="Keep SI-SDR loss at zero for this many steps; defaults depend on --stage.",
    )
    parser.add_argument(
        "--si-sdr-loss-warmup-steps",
        type=int,
        default=None,
        help="SI-SDR ramp duration after its start step; defaults depend on --stage.",
    )
    parser.add_argument(
        "--spectral-envelope-loss-weight",
        type=float,
        default=None,
        help=(
            "Maximum voiced spectral-envelope loss weight. Defaults to 0.05 "
            "for recon_pretrain, 0.01 for gan_pretrain, and zero otherwise."
        ),
    )
    parser.add_argument(
        "--spectral-envelope-loss-start-steps",
        type=int,
        default=None,
        help="Initial disabled steps; defaults depend on --stage.",
    )
    parser.add_argument(
        "--spectral-envelope-loss-warmup-steps",
        type=int,
        default=None,
        help=(
            "After its start step, linearly ramp the voiced spectral-envelope "
            "loss to its maximum weight; defaults depend on --stage."
        ),
    )
    parser.add_argument(
        "--bypass-rvq-during-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Bypass RVQ in every generator path (training, validation and GAN) "
            "while preserving the same Encoder/Decoder checkpoint topology."
        ),
    )
    parser.add_argument(
        "--rvq-calibration-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run Encoder -> RVQ only under no_grad so EMA codebooks are "
            "calibrated while Encoder and Decoder weights remain unchanged."
        ),
    )
    parser.add_argument(
        "--rvq-projection-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train only PCA-initialized 64->lookup->64 projections while "
            "bypassing discrete RVQ lookup and freezing Encoder/Decoder."
        ),
    )
    parser.add_argument(
        "--rvq-projection-pca-batches",
        type=int,
        default=100,
        help="Distributed batches used to estimate latent PCA before projection training.",
    )
    parser.add_argument(
        "--rvq-calibration-kmeans-batches",
        type=int,
        default=0,
        help=(
            "B1 batches used for one distributed sequential residual K-means "
            "initialization before the short EMA bootstrap; zero keeps the "
            "package's ordinary first-batch initialization."
        ),
    )
    parser.add_argument(
        "--reinitialize-rvq-codebooks-from-projection-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load a checkpoint explicitly saved with rq_projection_only=True, "
            "retain Encoder/Decoder and both trained 64<->lookup projections, "
            "but discard rq.* embeddings and EMA buffers. This is the safe "
            "Q0-reuse path for starting RVQ calibration with a new topology."
        ),
    )
    parser.add_argument(
        "--rq-lookup-dim",
        type=int,
        choices=(16, 24, 32),
        default=32,
        help="RVQ lookup/PCA dimension; bitrate is unchanged, storage scales with this value.",
    )
    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=9,
        help="Number of residual RVQ levels (default: 9).",
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=128,
        help="Uniform codebook size at every RVQ level (default: 128).",
    )
    parser.add_argument(
        "--codebook-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional per-level RVQ codebook sizes. When set, this overrides "
            "--codebook-size and must contain --num-quantizers powers of two. "
            "Recommended 3.2 kbps profile: 256 128 128 128 128 128 128 128 128."
        ),
    )
    parser.add_argument(
        "--rq-distance",
        choices=("cosine", "euclidean"),
        default="euclidean",
        help=(
            "RVQ nearest-neighbour geometry. Euclidean remains hardware-friendly "
            "through 2*dot(x,c)-||c||^2; use identical seeds/Q0 settings for A/B tests."
        ),
    )
    parser.add_argument(
        "--rvq-projection-min-evr",
        type=float,
        default=0.90,
        help="Stop Q0 before training when PCA variance retention is below this floor.",
    )
    parser.add_argument(
        "--rvq-projection-latent-mse-weight",
        type=float,
        default=0.0,
        help="Q0 normalized latent reconstruction-MSE weight.",
    )
    parser.add_argument(
        "--rvq-projection-latent-cosine-weight",
        type=float,
        default=0.0,
        help="Q0 latent cosine-distance weight.",
    )
    parser.add_argument(
        "--rvq-projection-orth-loss-weight",
        type=float,
        default=0.0,
        help="Weight for keeping Win rows approximately orthonormal in Q0/B1.5.",
    )
    parser.add_argument(
        "--rvq-projection-tie-loss-weight",
        type=float,
        default=0.0,
        help="Weight for keeping Wout close to Win transpose in Q0/B1.5.",
    )
    parser.add_argument(
        "--rvq-projection-freeze-input-steps",
        type=int,
        default=0,
        help="Q0 steps that train only Wout while keeping PCA Win fixed.",
    )
    parser.add_argument(
        "--rvq-joint-latent64-teacher-loss-weight",
        type=float,
        default=0.10,
        help="B1.5 normalized 64-D latent teacher-loss weight.",
    )
    parser.add_argument("--rvq-joint-decoder-only-steps", type=int, default=10000)
    parser.add_argument("--rvq-joint-rvq-adapt-end-steps", type=int, default=20000)
    parser.add_argument(
        "--rvq-plateau-freeze-start-steps",
        type=int,
        default=30000,
        help="Earliest B1.5 step at which lookup-NMSE plateau checks may freeze RVQ EMA.",
    )
    parser.add_argument(
        "--rvq-plateau-freeze-patience",
        type=int,
        default=7,
        help="Validation checks without lookup-NMSE improvement before freezing RVQ EMA; 0 disables.",
    )
    parser.add_argument(
        "--rvq-plateau-freeze-min-delta",
        type=float,
        default=0.001,
        help="Minimum lookup-NMSE reduction that resets the B1.5 RVQ plateau counter.",
    )
    parser.add_argument(
        "--rvq-joint-polish-decoder-lr",
        type=float,
        default=1.5e-6,
        help="Decoder LR after the RVQ EMA phase in B1.5 (default: 1.5e-6).",
    )
    parser.add_argument(
        "--rvq-joint-adapt",
        action="store_true",
        help=(
            "Run non-GAN B1.5 Decoder/RVQ-EMA adaptation while keeping "
            "Encoder and both projection matrices fixed."
        ),
    )
    parser.add_argument("--rvq-warm-in-steps", type=int, default=0,
                        help="Linearly mix continuous latent into full RVQ latent over N steps.")
    parser.add_argument("--rvq-codebook-balance-loss-weight", type=float, default=0.,
                        help="Weight of the differentiable per-level entropy-floor loss.")
    parser.add_argument("--rvq-codebook-balance-target-perplexity", type=float, default=8.,
                        help="Soft-assignment perplexity floor for each RVQ level.")
    parser.add_argument("--rvq-codebook-balance-temperature", type=float, default=1.,
                        help="Squared-distance soft-assignment temperature.")
    parser.add_argument("--rvq-quantization-error-loss-weight", type=float, default=0.,
                        help="Weight for normalized sequential RVQ residual error.")
    parser.add_argument("--rvq-continuous-teacher-loss-weight", type=float, default=0.,
                        help="Weight for stop-gradient Q0 waveform/STFT distillation.")
    parser.add_argument(
        "--formant-peak-loss-weight",
        type=float,
        default=None,
        help=(
            "Maximum differentiable F1/F2/F3 regional peak loss weight. "
            "The peak metrics are still reported when this is zero."
        ),
    )
    parser.add_argument(
        "--formant-peak-loss-start-steps",
        type=int,
        default=None,
        help="Delay peak-loss gradients while the cepstral envelope stabilizes.",
    )
    parser.add_argument(
        "--formant-peak-loss-warmup-steps",
        type=int,
        default=None,
        help="Linear ramp duration for the differentiable formant-peak loss.",
    )
    parser.add_argument(
        "--loss-grad-diagnostics-every",
        type=int,
        default=1_000,
        help=(
            "Measure decoder gradient norms for each weighted reconstruction "
            "loss every N steps; zero disables the diagnostic."
        ),
    )
    parser.add_argument(
        "--voiced-highband-loss-weight",
        type=float,
        default=None,
        help=(
            "Maximum target-voiced, frame-gain-normalized high-band log-spectrum "
            "loss weight (2.5-5.5 kHz primary, 5.5-7 kHz auxiliary); "
            "loss weight; defaults depend on --stage."
        ),
    )
    parser.add_argument(
        "--voiced-highband-loss-start-steps",
        type=int,
        default=None,
        help="Initial disabled steps for the voiced high-band objective.",
    )
    parser.add_argument(
        "--upper-highband-loss-weight",
        type=float,
        default=None,
        help=(
            "Maximum target-active 7-7.8 kHz log-spectrum loss weight. "
            "Defaults to 0.005 for recon_pretrain, 0.0025 for gan_pretrain, "
            "and zero otherwise."
        ),
    )
    parser.add_argument(
        "--upper-highband-loss-start-steps",
        type=int,
        default=None,
        help="Initial disabled steps for the independent 7-7.8 kHz objective.",
    )
    parser.add_argument(
        "--upper-highband-loss-warmup-steps",
        type=int,
        default=None,
        help="Linear ramp duration for the independent 7-7.8 kHz objective.",
    )
    parser.add_argument(
        "--active-spectral-detail-loss-weight",
        type=float,
        default=None,
        help=(
            "Maximum multi-resolution log-spectrum detail weight. Only "
            "target-voiced bins within 50 dB of the target-frame peak count."
        ),
    )
    parser.add_argument(
        "--active-spectral-detail-loss-start-steps",
        type=int,
        default=None,
        help="Initial disabled steps for the active spectral-detail objective.",
    )
    parser.add_argument(
        "--active-spectral-detail-loss-warmup-steps",
        type=int,
        default=None,
        help="Linear ramp duration for the active spectral-detail objective.",
    )
    parser.add_argument(
        "--upper-highband-energy-deficit-weight",
        type=float,
        default=None,
        help=(
            "Internal asymmetric energy-deficit coefficient inside the "
            "target-active 7-7.8 kHz loss."
        ),
    )
    parser.add_argument(
        "--upper-highband-energy-margin-db",
        type=float,
        default=None,
        help=(
            "Allowed 7-7.8 kHz reconstruction energy deficit in dB before "
            "the asymmetric squared deficit term activates."
        ),
    )
    parser.add_argument(
        "--voiced-highband-loss-warmup-steps",
        type=int,
        default=None,
        help="Linear ramp duration for the voiced high-band objective.",
    )
    parser.add_argument(
        "--voiced-highband-energy-deficit-weight",
        type=float,
        default=0.40,
        help=(
            "Internal one-sided high-band energy-deficit weight. This "
            "penalizes missing voiced high-band energy without rewarding "
            "high-frequency excess."
        ),
    )
    parser.add_argument(
        "--voiced-highband-energy-margin-db",
        type=float,
        default=0.10,
        help=(
            "Allowed voiced high-band energy deficit before the one-sided "
            "penalty activates."
        ),
    )
    parser.add_argument(
        "--voiced-hf-retention-loss-weight",
        type=float,
        default=None,
        help=(
            "One-sided target-voiced 3-7 kHz power-retention loss weight. "
            "This is aligned with the Stage-2 voiced-HF quality gate and does "
            "not reward excess high-frequency power."
        ),
    )
    parser.add_argument(
        "--voiced-hf-retention-margin-db",
        type=float,
        default=0.50,
        help="Allowed target-voiced 3-7 kHz power deficit before retention loss.",
    )

    parser.add_argument(
        "--stage1-plateau-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For recon_pretrain, keep the step scheduler constant after warmup "
            "and let ReduceLROnPlateau lower LR from validation "
            "online_aligned_si_sdr."
        ),
    )
    parser.add_argument(
        "--plateau-start-steps",
        type=int,
        default=60_000,
        help=(
            "Do not apply stage-1 ReduceLROnPlateau before this completed step. "
            "The early 10k-15k reconstruction-entry region is intentionally ignored."
        ),
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=0.5,
        help="LR multiplier used by stage-1 ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=16,
        help="Validation checks without sufficient online_aligned_si_sdr improvement before lowering LR.",
    )
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=0.03,
        help="Minimum absolute online_aligned_si_sdr improvement counted by ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--plateau-cooldown",
        type=int,
        default=2,
        help="Validation checks to wait after a plateau LR drop.",
    )
    parser.add_argument(
        "--plateau-min-lr",
        type=float,
        default=1e-5,
        help="Lower bound for stage-1 ReduceLROnPlateau generator LR.",
    )

    parser.add_argument(
        "--stage1-rvq-retention-patience",
        type=int,
        default=8,
        help=(
            "Consecutive fixed-validation checks with an unhealthy online q00/RVQ "
            "after the stage-1 minimum-step floor before a protective hard stop."
        ),
    )
    parser.add_argument(
        "--plateau-unclean-grace-checks",
        type=int,
        default=8,
        help=(
            "After plateau start, allow ReduceLROnPlateau to observe "
            "online_aligned_si_sdr if validation remains clean_ok=0 for this "
            "many consecutive validation checks. This does not relax best "
            "checkpoint clean-gate eligibility."
        ),
    )
    parser.add_argument(
        "--stage2-plateau-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For gan_pretrain, lower generator LR "
            "only after post-ramp validation plateaus while quality retention holds."
        ),
    )
    parser.add_argument(
        "--stage2-plateau-start-steps",
        type=int,
        default=10_000,
        help="Do not allow Stage-2 plateau LR reductions before this completed step.",
    )
    parser.add_argument(
        "--stage2-plateau-factor",
        type=float,
        default=0.5,
        help="Stage-2 generator-LR multiplier for each validation-plateau reduction.",
    )
    parser.add_argument(
        "--stage2-plateau-patience",
        type=int,
        default=8,
        help="Stage-2 quality-retained validation checks without a reconstruction-score improvement before reducing LR.",
    )
    parser.add_argument(
        "--stage2-plateau-threshold",
        type=float,
        default=0.01,
        help="Minimum absolute reduction in Stage-2 validation reconstruction score counted by plateau.",
    )
    parser.add_argument(
        "--stage2-plateau-cooldown",
        type=int,
        default=2,
        help="Validation checks to wait after a Stage-2 LR reduction.",
    )
    parser.add_argument(
        "--stage2-plateau-min-lr",
        type=float,
        default=2e-7,
        help="Minimum Stage-2 generator LR.",
    )
    parser.add_argument(
        "--stage2-plateau-discr-min-lr",
        type=float,
        default=2e-7,
        help="Minimum Stage-2 waveform-discriminator LR; preserves the initial G:D ratio.",
    )
    parser.add_argument(
        "--stage2-plateau-stft-discr-min-lr",
        type=float,
        default=1e-7,
        help=(
            "Minimum Stage-2 STFT-discriminator LR. Kept separate so plateau "
            "cannot raise the lower STFT-D LR to the waveform-D minimum."
        ),
    )
    parser.add_argument(
        "--click-loss-weight",
        type=float,
        default=None,
        help=(
            "Auxiliary first-difference loss weight for reducing click/electric "
            "artifacts. Defaults are stage-specific and intentionally small."
        ),
    )
    parser.add_argument(
        "--jump-loss-weight",
        type=float,
        default=None,
        help=(
            "Auxiliary soft excess-jump loss weight for reducing isolated spikes. "
            "Defaults are stage-specific and intentionally small."
        ),
    )
    parser.add_argument(
        "--preemph-loss-weight",
        type=float,
        default=None,
        help=(
            "Pre-emphasis waveform loss weight. Defaults to zero; use an "
            "explicit positive value only for a targeted experiment."
        ),
    )
    parser.add_argument(
        "--noise-floor-loss-weight",
        type=float,
        default=None,
        help=(
            "Quiet-frame multiband spectral excess noise loss weight. "
            "Defaults to 0.05 for recon_pretrain, 0.02 for gan_pretrain, "
            "and zero for later stages."
        ),
    )
    parser.add_argument(
        "--transient-loss-warmup-steps",
        type=int,
        default=None,
        help=(
            "Linearly ramp click/jump loss over this many steps. Defaults are "
            "stage-specific so transient penalties do not dominate early training."
        ),
    )
    parser.add_argument(
        "--disable-clean-gate",
        action="store_true",
        help=(
            "Disable artifact-aware checkpoint eligibility. By default, best "
            "checkpoint selection rejects validation checkpoints with poor "
            "aligned SI-SDR/correlation or abnormal peak/click/jump metrics."
        ),
    )
    parser.add_argument(
        "--clean-gate-min-aligned-si-sdr",
        type=float,
        default=0.0,
        help="Minimum aligned SI-SDR required for best checkpoint eligibility.",
    )
    parser.add_argument(
        "--clean-gate-min-aligned-corr",
        type=float,
        default=0.65,
        help="Minimum aligned correlation required for best checkpoint eligibility.",
    )
    parser.add_argument(
        "--clean-gate-min-rms-ratio",
        type=float,
        default=0.4,
        help="Minimum recon/input RMS ratio required for best checkpoint eligibility.",
    )
    parser.add_argument(
        "--clean-gate-max-rms-ratio",
        type=float,
        default=2.5,
        help="Maximum recon/input RMS ratio required for best checkpoint eligibility.",
    )
    parser.add_argument(
        "--clean-gate-max-recon-peak",
        type=float,
        default=1.2,
        help="Maximum reconstructed absolute peak for best checkpoint eligibility.",
    )
    parser.add_argument(
        "--clean-gate-max-recon-clip-fraction",
        type=float,
        default=1e-3,
        help="Maximum reconstructed clipping fraction for best checkpoint eligibility.",
    )
    parser.add_argument(
        "--clean-gate-max-click-score",
        type=float,
        default=6.0,
        help="Maximum click score for best checkpoint eligibility.",
    )
    parser.add_argument(
        "--clean-gate-max-click-excess",
        type=float,
        default=0.5,
        help=(
            "Maximum reconstructed click-score excess above the matched target. "
            "This relative gate replaces the absolute click-score gate when set."
        ),
    )
    parser.add_argument(
        "--clean-gate-max-jump-ratio",
        type=float,
        default=2.0,
        help="Maximum max-jump ratio for best checkpoint eligibility.",
    )
    parser.add_argument(
        "--clean-gate-max-p999-jump-ratio",
        type=float,
        default=1.75,
        help="Maximum p99.9 jump ratio for best checkpoint eligibility.",
    )
    parser.add_argument(
        "--clean-gate-min-voiced-hf-ratio-db",
        type=float,
        default=-1.5,
        help=(
            "Stage-1 minimum voiced 3-7 kHz reconstruction/target energy "
            "ratio for best-checkpoint eligibility."
        ),
    )
    parser.add_argument(
        "--clean-gate-max-voiced-hf-ratio-db",
        type=float,
        default=1.0,
        help=(
            "Stage-1 maximum voiced 3-7 kHz reconstruction/target energy "
            "ratio, used to reject high-frequency over-generation."
        ),
    )
    parser.add_argument(
        "--stage2-quality-retention-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For gan_pretrain, reject best-checkpoint updates that regress beyond "
            "the initialized stage-1 baseline. Hard stopping is controlled separately."
        ),
    )
    parser.add_argument(
        "--clean-gate-max-negative-fraction",
        type=float,
        default=0.01,
        help=(
            "Maximum validation batches with polarity inversion for checkpoint "
            "eligibility. Use 0.05 for exploratory B1.5/B2 selection; deployment stays 0.01."
        ),
    )
    parser.add_argument(
        "--stage2-quality-hard-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stop after sustained Stage-2 retention failures. Disabled for the "
            "150k exploratory schedule so failed candidates are rejected without "
            "terminating training."
        ),
    )
    parser.add_argument(
        "--stage2-max-aligned-si-sdr-drop",
        type=float,
        default=0.15,
        help="Maximum Stage-2 aligned SI-SDR drop from its initialization baseline in dB.",
    )
    parser.add_argument(
        "--stage2-max-aligned-corr-drop",
        type=float,
        default=0.02,
        help="Maximum Stage-2 aligned correlation drop from its initialization baseline.",
    )
    parser.add_argument(
        "--stage2-max-quiet-hf-excess-db-rise",
        type=float,
        default=0.50,
        help="Maximum Stage-2 quiet-HF excess rise from its initialization baseline in dB.",
    )
    parser.add_argument(
        "--stage2-quality-retention-patience",
        type=int,
        default=8,
        help="Consecutive validation regressions before the Stage-2 quality hard stop.",
    )
    parser.add_argument(
        "--stage2-rvq-retention-patience",
        type=int,
        default=6,
        help="Consecutive q00/q01 regressions before the Stage-2 RVQ hard stop.",
    )
    parser.add_argument(
        "--stage2-max-voiced-hf-ratio-db-drop",
        type=float,
        default=0.30,
        help=(
            "Maximum Stage-2 voiced 3-7 kHz energy-ratio drop from the "
            "initialization baseline in dB."
        ),
    )
    parser.add_argument(
        "--stage2-max-voiced-hf-ratio-db-rise",
        type=float,
        default=1.50,
        help=(
            "Maximum Stage-2 voiced 3-7 kHz energy-ratio rise from the "
            "initialization baseline in dB."
        ),
    )
    parser.add_argument(
        "--stage2-voiced-hf-score-weight",
        type=float,
        default=3.,
        help=(
            "Squared two-sided voiced-HF gate-deviation penalty added to the Stage-2 "
            "validation score used for checkpoint ranking and plateau LR."
        ),
    )
    parser.add_argument(
        "--stage2-max-click-score-rise",
        type=float,
        default=0.30,
        help="Maximum Stage-2 click-score rise above its initialization baseline.",
    )
    parser.add_argument(
        "--stage2-max-ac320-isolated-rise",
        type=float,
        default=0.005,
        help=(
            "Maximum Stage-2 ac_320_isolated rise above its initialization "
            "baseline before rejecting checkpoints and counting a quality failure."
        ),
    )
    parser.add_argument(
        "--stage2-max-comb-median-excess-db-rise",
        type=float,
        default=0.25,
        help=(
            "Maximum Stage-2 comb_median_excess_db rise above its "
            "initialization baseline."
        ),
    )
    parser.add_argument(
        "--stage2-balanced-max-aligned-si-sdr-drop",
        type=float,
        default=0.10,
        help=(
            "Maximum aligned SI-SDR drop from initialization for Stage-2 "
            "balanced-best checkpoints."
        ),
    )
    parser.add_argument(
        "--stage2-min-voiced-7k-7p8k-ratio-db",
        type=float,
        default=-1.0,
        help="Minimum voiced 7-7.8 kHz energy ratio for Stage-2 balanced bests.",
    )
    parser.add_argument(
        "--stage2-max-voiced-7k-7p8k-ratio-db",
        type=float,
        default=0.5,
        help="Maximum voiced 7-7.8 kHz energy ratio for Stage-2 balanced bests.",
    )
    parser.add_argument(
        "--stage2-max-quiet-7k-7p8k-excess-db-rise",
        type=float,
        default=0.30,
        help=(
            "Maximum quiet 7-7.8 kHz excess rise above initialization for "
            "Stage-2 balanced bests."
        ),
    )
    parser.add_argument(
        "--stage2-upper-highband-score-weight",
        type=float,
        default=0.10,
        help=(
            "Squared two-sided 7-7.8 kHz gate-deviation penalty added to "
            "the Stage-2 checkpoint and plateau score."
        ),
    )
    parser.add_argument(
        "--stage2-active-spectral-score-weight",
        type=float,
        default=0.05,
        help=(
            "Linear active-spectral-detail term added to the Stage-2 "
            "checkpoint and plateau score after reconstruction and HF gates."
        ),
    )
    parser.add_argument(
        "--waveform-r1-every",
        type=int,
        default=0,
        help="Waveform-discriminator real-only R1 interval; 0 disables it.",
    )
    parser.add_argument(
        "--waveform-r1-gamma",
        type=float,
        default=0.0,
        help="Waveform-discriminator real-only R1 gamma.",
    )
    parser.add_argument(
        "--stft-r1-every",
        type=int,
        default=32,
        help="STFT-discriminator real-only R1 interval.",
    )
    parser.add_argument(
        "--stft-r1-gamma",
        type=float,
        default=5e-3,
        help=(
            "Length-normalized STFT-discriminator R1 gamma; no lazy-interval "
            "multiplication is applied. Recalibrate from raw_mean diagnostics."
        ),
    )
    parser.add_argument(
        "--stft-discr-lr",
        type=float,
        default=None,
        help="Independent STFT-discriminator LR; Stage 2 defaults to 5e-7.",
    )
    parser.add_argument(
        "--waveform-discr-lrs",
        type=float,
        nargs=3,
        default=None,
        metavar=("SCALE1", "SCALE05", "SCALE025"),
        help="Per-scale waveform-D LRs in the order 1.0, 0.5, 0.25.",
    )
    parser.add_argument(
        "--waveform-discr-update-every",
        type=int,
        nargs=3,
        default=None,
        metavar=("SCALE1", "SCALE05", "SCALE025"),
        help="Update intervals for waveform D scales 1.0, 0.5, 0.25.",
    )
    parser.add_argument(
        "--waveform-discr-loss-weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("SCALE1", "SCALE05", "SCALE025"),
        help=(
            "Normalized waveform-discriminator branch weights. Defaults to "
            "the stage configuration."
        ),
    )
    parser.add_argument(
        "--stft-discr-update-every",
        type=int,
        default=None,
        help="Update the STFT discriminator once per this many generator steps.",
    )
    parser.add_argument(
        "--stft-discr-loss-weight",
        type=float,
        default=None,
        help="STFT discriminator branch weight in the normalized D objective.",
    )
    parser.add_argument(
        "--gan-grad-diagnostics-every",
        type=int,
        default=500,
        help=(
            "For Stage 2, measure decoder gradient norms from reconstruction "
            "and GAN objectives every N steps; 0 disables this diagnostic."
        ),
    )
    parser.add_argument(
        "--discr-max-grad-norm",
        type=float,
        default=0.5,
        help="Independent gradient-norm clipping threshold for each discriminator branch.",
    )
    parser.add_argument(
        "--stage2-encoder-unfreeze-step",
        "--stage2-unfreeze-encoder-rvq-step",
        dest="stage2_unfreeze_encoder_rvq_step",
        type=int,
        default=10_000,
        help=(
            "Unfreeze the selected Encoder tail at this Stage-2 step; -1 keeps "
            "the entire Encoder frozen. RVQ remains frozen in either case. "
            "The legacy option name is retained as an alias."
        ),
    )
    parser.add_argument(
        "--stage2-encoder-trainable-from-block",
        type=int,
        default=3,
        help=(
            "Zero-based first trainable Encoder block after Stage-2 unfreeze. "
            "The default 3 trains Block4 and the final latent convolution only."
        ),
    )
    parser.add_argument(
        "--stage2-encoder-lr",
        type=float,
        default=1e-7,
        help="Learning rate for the Stage-2 trainable Encoder tail.",
    )
    parser.add_argument(
        "--stage2-targeted-refine",
        "--stage25-encoder-refine",
        dest="stage2_targeted_refine",
        action="store_true",
        help=(
            "Run the short Stage-2.5 refinement preset from a Stage-2 checkpoint: "
            "inherit discriminators, train Decoder at 2e-7 and Encoder at 1e-7, "
            "keep RVQ frozen, and disable the plateau scheduler."
        ),
    )
    parser.add_argument(
        "--stage25-decoder-lr",
        type=float,
        default=2e-7,
        help="Decoder learning rate for --stage25-encoder-refine.",
    )
    parser.add_argument(
        "--stage25-encoder-lr",
        type=float,
        default=5e-8,
        help=(
            "Encoder learning rate for Stage-2.5 refinement. Keep this well "
            "below the Decoder learning rate; the conservative default is 5e-8."
        ),
    )
    parser.add_argument(
        "--stage25-decoder-only-refine",
        action="store_true",
        help=(
            "Run a short reconstruction-only Stage-2.5 diagnostic: disable "
            "GAN updates, optimize only Decoder parameters, keep Encoder and "
            "RVQ frozen, and use --stage25-decoder-lr."
        ),
    )
    parser.add_argument(
        "--stage25-joint-recon-refine",
        action="store_true",
        help=(
            "Run a short reconstruction-only Stage-2.5 refinement: disable "
            "GAN updates, optimize Decoder and Encoder with separate low "
            "learning rates, keep RVQ frozen, and enforce strict AC320/comb "
            "retention gates relative to the initialization checkpoint."
        ),
    )
    parser.add_argument(
        "--stage25-rvq-midband-refine",
        action="store_true",
        help=(
            "Run a short reconstruction-only Stage-2.5 refinement for overall "
            "and mid/low-band error: update Encoder and RVQ from step 0 while "
            "keeping the complete Decoder frozen for all training steps."
        ),
    )
    parser.add_argument(
        "--stream-context-frames",
        type=int,
        default=0,
        help="Deprecated compatibility option; streaming now uses per-layer activation state.",
    )
    parser.add_argument(
        "--decoder-upsample-mode",
        choices=("convtranspose", "linear"),
        default="convtranspose",
        help=(
            "Decoder upsampling block. The production default is learned "
            "causal ConvTranspose1d, avoiding explicit interpolation smoothing. "
            "'linear' remains available for old checkpoints and ablations."
        ),
    )
    parser.add_argument(
        "--decoder-linear-upsample-kernel-min",
        type=int,
        default=4,
        help=(
            "Minimum kernel size for decoder linear-upsample CausalConv1d. "
            "With the default 4, every layer uses its natural 2*stride "
            "kernel (16, 10, 8, 4 for decoder strides 8, 5, 4, 2), "
            "avoiding extra smoothing in the final x2 upsampling layer."
        ),
    )
    parser.add_argument(
        "--decoder-interpolation-mode",
        choices=("linear", "cubic"),
        default="linear",
        help=(
            "Causal interpolation between latent frames. 'cubic' uses backward-"
            "difference Hermite tangents so adjacent intervals share the same "
            "slope at every codec-frame boundary. The production default is "
            "'linear'; use 'cubic' only for a controlled ablation."
        ),
    )
    parser.add_argument(
        "--decoder-split-first-upsample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Split the first decoder x8 expansion into causal x4 then x2 "
            "interpolation/convolution stages while preserving total x320. "
            "Disabled by default for the single-x8 production baseline."
        ),
    )
    parser.add_argument(
        "--decoder-residual-scale-start",
        type=float,
        default=0.2,
        help="Decoder residual scale before the warmup window in recon_pretrain.",
    )
    parser.add_argument(
        "--decoder-residual-scale-end",
        type=float,
        default=1.0,
        help="Decoder residual scale after the warmup window in recon_pretrain.",
    )
    parser.add_argument(
        "--decoder-residual-scale-warmup-start-steps",
        type=int,
        default=0,
        help="Step where decoder residual scale begins increasing.",
    )
    parser.add_argument(
        "--stage2-generator-hold-steps",
        type=int,
        default=5_000,
        help=(
            "For gan_pretrain, keep the generator at a conservative LR during "
            "the initial GAN ramp so the Stage-1 reconstruction baseline is retained."
        ),
    )
    parser.add_argument(
        "--stage2-generator-freeze-steps",
        type=int,
        default=0,
        help=(
            "Optional initial full-generator freeze. The safe B2 schedule "
            "defaults to zero and trains Decoder reconstruction from step 0."
        ),
    )
    parser.add_argument(
        "--stage2-gan-start-step",
        type=int,
        default=None,
        help=(
            "First step that exposes the Stage-2 generator to GAN/feature "
            "losses; earlier Decoder updates are reconstruction-only."
        ),
    )
    parser.add_argument(
        "--stage2-gan-ramp-steps",
        type=int,
        default=None,
        help="Number of steps used to ramp Stage-2 GAN and feature weights.",
    )
    parser.add_argument(
        "--stage2-phase2-start-step",
        type=int,
        default=3_000,
        help="Absolute Stage-2 step where the single-scale safe GAN starts.",
    )
    parser.add_argument(
        "--stage2-phase3-start-step",
        type=int,
        default=15_000,
        help="Absolute Stage-2 step where all discriminator branches are enabled.",
    )
    parser.add_argument(
        "--stage2-phase2-generator-lr",
        type=float,
        default=2e-7,
        help="Decoder LR from phase2-start through the start of phase 3.",
    )
    parser.add_argument(
        "--stage2-phase3-generator-lr",
        type=float,
        default=2.5e-7,
        help="Decoder LR from phase3-start through the end of Stage 2.",
    )
    parser.add_argument(
        "--stage2-phase3-gan-adversarial-max",
        type=float,
        default=1e-4,
        help="Adversarial weight used from the 100k refinement phase onward.",
    )
    parser.add_argument(
        "--stage2-phase3-gan-feature-max",
        type=float,
        default=0.20,
        help="Feature-matching ceiling used in the full perceptual phase.",
    )
    parser.add_argument(
        "--stage2-teacher-retention-weight",
        type=float,
        default=0.,
        help=(
            "Frozen B1.5 Decoder waveform/log-STFT retention-loss weight "
            "during ordinary Stage-2 GAN training."
        ),
    )
    parser.add_argument(
        "--stage2-adaptive-gan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Adapt the GAN ceiling from validation SI-SDR retention and the "
            "measured GAN/reconstruction gradient ratio."
        ),
    )
    parser.add_argument(
        "--stage2-max-gan-grad-ratio",
        type=float,
        default=0.05,
        help="Maximum target GAN-to-reconstruction Decoder gradient ratio.",
    )
    parser.add_argument(
        "--stage2-decoder-lr-multipliers",
        type=float,
        nargs=3,
        default=(1., 1., 1.),
        metavar=("EARLY", "MID", "LATE"),
        help=(
            "Relative LRs for early Decoder modules, the middle block, and "
            "late/output modules."
        ),
    )
    parser.add_argument(
        "--stft-recon-loss-weight",
        type=float,
        default=None,
        help="Independent MR-STFT weight; defaults depend on --stage.",
    )
    parser.add_argument(
        "--stft-recon-loss-start-steps",
        type=int,
        default=None,
        help="Initial disabled steps for independent MR-STFT loss.",
    )
    parser.add_argument(
        "--stft-recon-loss-warmup-steps",
        type=int,
        default=None,
        help="Linear ramp duration for independent MR-STFT loss.",
    )
    parser.add_argument(
        "--frame-phase-loss-weight",
        type=float,
        default=None,
        help="Weight for the 320-sample frame-phase residual loss.",
    )
    parser.add_argument(
        "--frame-phase-loss-warmup-steps",
        type=int,
        default=None,
        help="Linear ramp duration for frame-phase residual loss.",
    )
    parser.add_argument(
        "--stage2-generator-hold-lr",
        type=float,
        default=1e-7,
        help="Generator LR at release, linearly ramped to the base LR by hold-steps.",
    )
    parser.add_argument(
        "--stage2-discriminator-hold-steps",
        type=int,
        default=0,
        help=(
            "For gan_pretrain, keep every discriminator at a conservative LR "
            "during the initial GAN ramp while still updating from step zero."
        ),
    )
    parser.add_argument(
        "--stage2-discriminator-start-steps",
        type=int,
        default=None,
        help=(
            "First Stage-2 step that updates discriminators. Defaults to GAN "
            "start; targeted refinement uses step 0 for discriminator warmup."
        ),
    )
    parser.add_argument(
        "--stage2-discriminator-hold-lr",
        type=float,
        default=5e-6,
        help="Discriminator LR during the initial Stage-2 GAN ramp.",
    )
    parser.add_argument(
        "--stage2-recon-transition-start-steps",
        type=int,
        default=None,
        help=(
            "Optional legacy Stage-2 reconstruction-weight transition start. "
            "Disabled by default so retention weights remain fixed."
        ),
    )
    parser.add_argument(
        "--stage2-recon-transition-end-steps",
        type=int,
        default=None,
        help=(
            "Optional legacy Stage-2 reconstruction-weight transition end; "
            "must be supplied together with its start."
        ),
    )
    parser.add_argument(
        "--stage2-quality-gate-start-steps",
        type=int,
        default=20_000,
        help=(
            "Record Stage-2 quality retention from initialization, but defer "
            "quality hard-stop accumulation until this step."
        ),
    )
    parser.add_argument(
        "--stage2-best-checkpoint-min-step",
        type=int,
        default=5_000,
        help="Do not save Stage-2 best candidates before this completed step.",
    )
    parser.add_argument(
        "--decoder-residual-scale-warmup-end-steps",
        type=int,
        default=15_000,
        help=(
            "Step where decoder residual scale reaches the end value. "
            "Stage-1 best checkpoints are held until this point so downstream "
            "stages never inherit an in-progress residual-scale schedule."
        ),
    )
    parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=None,
        help="Streaming boundary target weight; stage default is 0.02.",
    )
    parser.add_argument(
        "--boundary-loss-radius",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--boundary-loss-start-steps",
        type=int,
        default=None,
        help="Streaming step before boundary loss begins.",
    )
    parser.add_argument(
        "--boundary-loss-warmup-steps",
        type=int,
        default=None,
        help="Linear boundary-loss ramp length after its start step.",
    )
    parser.add_argument(
        "--stream-consistency-loss-weight",
        type=float,
        default=None,
        help=(
            "Weight for stateful-streaming output consistency against the "
            "same weights executed through the full-sequence causal path."
        ),
    )
    parser.add_argument(
        "--stream-consistency-loss-start-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--stream-consistency-loss-warmup-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--state-teacher-retention-weight",
        type=float,
        default=None,
        help="Frozen B2 Decoder waveform/log-STFT retention weight for final state calibration.",
    )
    parser.add_argument(
        "--state-decoder-trainable-from-block",
        type=int,
        default=None,
        help="First Decoder upsampling block trained by final state calibration (0-based).",
    )
    parser.add_argument(
        "--valid-frac",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--test-eval-batches",
        type=int,
        default=None,
        help="Limit final held-out test files. Defaults to the full test split.",
    )
    parser.add_argument(
        "--test-block-seconds",
        type=float,
        default=5.0,
        help="Block length used for deterministic full-file test evaluation.",
    )
    parser.add_argument(
        "--test-context-ms",
        type=float,
        default=60.0,
        help="Previous-audio context for non-streaming checkpoints; stateful streaming tests carry state continuously.",
    )
    parser.add_argument(
        "--save-test-reconstructions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save held-out test reconstructions after recon_pretrain finishes.",
    )
    parser.add_argument(
        "--test-recon-dir",
        type=Path,
        default=None,
        help="Directory for saved held-out test reconstructions. Defaults to results/stage1_test_reconstructions.",
    )
    parser.add_argument(
        "--test-report-file",
        type=Path,
        default=None,
        help="Text file for final held-out test summary. Defaults to results/stage1_test_report.txt.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and evaluate only --test-checkpoint on the held-out split.",
    )
    parser.add_argument(
        "--test-checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint for --test-only; accepts model-only or full trainer checkpoints.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Skip training and evaluate only --validation-checkpoint on the "
            "deterministic fixed validation set. This mode is intended for "
            "checkpoint handoff / fallback decisions and never evaluates the "
            "held-out test split."
        ),
    )
    parser.add_argument(
        "--validation-checkpoint",
        type=Path,
        default=None,
        help="Explicit model-only or full trainer checkpoint for --validation-only.",
    )
    parser.add_argument(
        "--validation-report-file",
        type=Path,
        default=None,
        help=(
            "TSV output for --validation-only. Defaults to "
            "results/fixed_validation_report.tsv."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reset-early-stopping-on-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When resuming a full trainer checkpoint, preserve model/optimizer/"
            "scheduler/best-score state but clear an already-triggered early stop "
            "and its bad-validation counter."
        ),
    )

    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(
        r"soundstream\.(\d+)\.pt",
        path.name,
    )
    return int(match.group(1)) if match else -1


def latest_checkpoint(results_dir: Path) -> Path | None:
    if not results_dir.exists():
        return None

    latest = results_dir / "latest.pt"
    if latest.exists():
        return latest

    checkpoints = [
        path
        for path in results_dir.glob("soundstream.*.pt")
        if checkpoint_step(path) >= 0
    ]

    return max(
        checkpoints,
        key=checkpoint_step,
        default=None,
    )



def calculate_bitrate(
    sample_rate: int,
    strides: tuple[int, ...],
    codebook_size: int | tuple[int, ...],
    num_quantizers: int | None = None,
) -> float:
    downsample_factor = math.prod(strides)
    frame_rate = sample_rate / downsample_factor
    if isinstance(codebook_size, int):
        if num_quantizers is None:
            raise ValueError("num_quantizers is required for a uniform codebook")
        codebook_sizes = (codebook_size,) * num_quantizers
    else:
        codebook_sizes = tuple(codebook_size)

    return frame_rate * sum(math.log2(size) for size in codebook_sizes)


def load_model_weights_only(
    model: torch.nn.Module,
    checkpoint: Path,
    *,
    generator_only: bool = False,
    reinitialize_rvq_from_bypass: bool = False,
    reinitialize_rvq_codebooks_from_projection: bool = False,
) -> dict:
    if reinitialize_rvq_from_bypass and reinitialize_rvq_codebooks_from_projection:
        raise ValueError("RVQ bypass and Q0 projection migration modes are mutually exclusive")
    pkg = torch.load(str(checkpoint), map_location="cpu")
    checkpoint_config = {}
    if reinitialize_rvq_from_bypass and "config" not in pkg:
        raise ValueError(
            "--reinitialize-rvq-from-bypass-checkpoint requires checkpoint "
            "configuration metadata proving bypass_rvq=True."
        )
    if reinitialize_rvq_codebooks_from_projection and "config" not in pkg:
        raise ValueError(
            "--reinitialize-rvq-codebooks-from-projection-checkpoint requires "
            "checkpoint configuration metadata proving rq_projection_only=True."
        )
    if "config" in pkg:
        checkpoint_config = pickle.loads(pkg["config"])
        rvq_fields = (
            ("rq_num_quantizers", "num_quantizers"),
            ("codebook_dim", "codebook_dim"),
            ("rq_lookup_dim", "rq_lookup_dim"),
        )
        rvq_mismatches = []
        for config_name, model_name in rvq_fields:
            checkpoint_value = checkpoint_config.get(config_name)
            model_value = getattr(model, model_name, None)
            if (
                checkpoint_value is not None and
                model_value is not None and
                int(checkpoint_value) != int(model_value)
            ):
                rvq_mismatches.append(
                    f"{config_name}: checkpoint={checkpoint_value}, "
                    f"current_model={model_value}"
                )
        checkpoint_codebook_sizes = checkpoint_config.get("codebook_size")
        if isinstance(checkpoint_codebook_sizes, int):
            checkpoint_codebook_sizes = (
                int(checkpoint_codebook_sizes),
            ) * int(checkpoint_config.get("rq_num_quantizers", 0))
        elif checkpoint_codebook_sizes is not None:
            checkpoint_codebook_sizes = tuple(
                int(size) for size in checkpoint_codebook_sizes
            )
        model_codebook_sizes = tuple(getattr(model, "codebook_sizes", ()))
        if (
            checkpoint_codebook_sizes and
            model_codebook_sizes and
            tuple(checkpoint_codebook_sizes) != model_codebook_sizes
        ):
            rvq_mismatches.append(
                "codebook_sizes: checkpoint="
                f"{tuple(checkpoint_codebook_sizes)}, "
                f"current_model={model_codebook_sizes}"
            )
        checkpoint_bypassed_rvq = bool(checkpoint_config.get("bypass_rvq", False))
        checkpoint_projection_only = bool(checkpoint_config.get("rq_projection_only", False))
        if reinitialize_rvq_from_bypass and not checkpoint_bypassed_rvq:
            raise ValueError(
                "--reinitialize-rvq-from-bypass-checkpoint requires a source "
                "checkpoint whose saved config records bypass_rvq=True."
            )
        if reinitialize_rvq_codebooks_from_projection and not checkpoint_projection_only:
            raise ValueError(
                "--reinitialize-rvq-codebooks-from-projection-checkpoint requires "
                "a source checkpoint whose saved config records rq_projection_only=True."
            )
        if reinitialize_rvq_codebooks_from_projection:
            for config_name, model_name in (
                ("codebook_dim", "codebook_dim"),
                ("rq_lookup_dim", "rq_lookup_dim"),
            ):
                checkpoint_value = checkpoint_config.get(config_name)
                model_value = getattr(model, model_name, None)
                if (
                    checkpoint_value is not None and
                    model_value is not None and
                    int(checkpoint_value) != int(model_value)
                ):
                    raise ValueError(
                        "Q0 projection dimension mismatch: "
                        f"{config_name}: checkpoint={checkpoint_value}, "
                        f"current_model={model_value}. Projection matrices "
                        "cannot be reused across different dimensions."
                    )
        if (
            rvq_mismatches and
            not reinitialize_rvq_from_bypass and
            not reinitialize_rvq_codebooks_from_projection
        ):
            raise ValueError(
                "Checkpoint RVQ topology mismatch: "
                + "; ".join(rvq_mismatches)
                + ". Start a fresh run or use a matching checkpoint."
            )
        checkpoint_cosine = bool(checkpoint_config.get("rq_use_cosine_sim", False))
        model_cosine = bool(getattr(model, "rq_use_cosine_sim", False))
        if (
            checkpoint_cosine != model_cosine and
            not reinitialize_rvq_from_bypass
        ):
            raise ValueError(
                "Checkpoint RVQ distance mismatch: "
                f"checkpoint_cosine={checkpoint_cosine}, current_model_cosine={model_cosine}. "
                "Start a fresh RVQ run or rebuild it from a confirmed bypass checkpoint."
            )
        checkpoint_upsample = checkpoint_config.get(
            "decoder_upsample_mode",
            "convtranspose",
        )
        model_upsample = getattr(model, "decoder_upsample_mode", None)
        if model_upsample is not None and checkpoint_upsample != model_upsample:
            raise ValueError(
                "Checkpoint decoder upsample mode mismatch: "
                f"checkpoint={checkpoint_upsample}, current_model={model_upsample}. "
                "Use a checkpoint trained with the same decoder structure, or "
                "rerun with --decoder-upsample-mode matching the checkpoint."
            )
        checkpoint_kernel_min = checkpoint_config.get(
            "decoder_linear_upsample_kernel_min",
            0,
        )
        model_kernel_min = getattr(
            model,
            "decoder_linear_upsample_kernel_min",
            None,
        )
        if (
            checkpoint_upsample == "linear" and
            model_kernel_min is not None and
            checkpoint_kernel_min != model_kernel_min
        ):
            raise ValueError(
                "Checkpoint decoder linear upsample kernel-min mismatch: "
                f"checkpoint={checkpoint_kernel_min}, current_model={model_kernel_min}. "
                "This changes decoder parameter shapes; use a matching checkpoint "
                "or rerun with --decoder-linear-upsample-kernel-min matching the checkpoint."
            )
        checkpoint_interpolation_mode = checkpoint_config.get(
            "decoder_interpolation_mode",
            "linear",
        )
        model_interpolation_mode = getattr(
            model,
            "decoder_interpolation_mode",
            None,
        )
        if (
            checkpoint_upsample == "linear" and
            model_interpolation_mode is not None and
            checkpoint_interpolation_mode != model_interpolation_mode
        ):
            raise ValueError(
                "Checkpoint decoder interpolation mismatch: "
                f"checkpoint={checkpoint_interpolation_mode}, "
                f"current_model={model_interpolation_mode}. "
                "Slope-continuous interpolation changes decoder behavior; "
                "start a fresh run or use a matching checkpoint."
            )
        checkpoint_split_first = bool(checkpoint_config.get(
            "decoder_split_first_upsample",
            False,
        ))
        model_split_first = getattr(
            model,
            "decoder_split_first_upsample",
            None,
        )
        if (
            model_split_first is not None and
            checkpoint_split_first != model_split_first
        ):
            raise ValueError(
                "Checkpoint first decoder upsample structure mismatch: "
                f"checkpoint_split_x8={checkpoint_split_first}, "
                f"current_split_x8={model_split_first}. "
                "The x8 versus x4+x2 decoder parameters are not shape-compatible."
            )
        checkpoint_depthwise_blocks = tuple(checkpoint_config.get(
            "encoder_depthwise_separable_blocks",
            (),
        ))
        model_depthwise_blocks = tuple(getattr(
            model,
            "encoder_depthwise_separable_blocks",
            (),
        ))
        if checkpoint_depthwise_blocks != model_depthwise_blocks:
            raise ValueError(
                "Checkpoint Encoder DSCNN topology mismatch: "
                f"checkpoint_blocks={checkpoint_depthwise_blocks}, "
                f"current_model_blocks={model_depthwise_blocks}. "
                "Encoder DSCNN block selection changes parameter names and shapes; start a "
                "fresh run or use a checkpoint trained with the same topology."
            )
        if checkpoint_depthwise_blocks:
            checkpoint_dscnn_revision = int(checkpoint_config.get(
                "encoder_depthwise_separable_revision",
                1,
            ))
            model_dscnn_revision = int(getattr(
                model,
                "encoder_depthwise_separable_revision",
                2,
            ))
            if checkpoint_dscnn_revision != model_dscnn_revision:
                raise ValueError(
                    "Checkpoint Encoder DSCNN activation topology mismatch: "
                    f"checkpoint_revision={checkpoint_dscnn_revision}, "
                    f"current_revision={model_dscnn_revision}. Revision 2 "
                    "uses full-rank Pointwise layers, while revision 3 uses "
                    "low-rank Pointwise factorization; start a fresh run."
                )
            checkpoint_low_rank_ranks = tuple(
                tuple(int(rank) for rank in ranks)
                for ranks in checkpoint_config.get(
                    "encoder_low_rank_pointwise_ranks",
                    (),
                )
            )
            model_low_rank_ranks = tuple(getattr(
                model,
                "encoder_low_rank_pointwise_ranks",
                (),
            ))
            if checkpoint_low_rank_ranks != model_low_rank_ranks:
                raise ValueError(
                    "Checkpoint Encoder low-rank Pointwise topology mismatch: "
                    f"checkpoint_ranks={checkpoint_low_rank_ranks}, "
                    f"current_model_ranks={model_low_rank_ranks}. Start a "
                    "fresh run or use a checkpoint with matching ranks."
                )
    state_dict = pkg["model"] if "model" in pkg else pkg
    if reinitialize_rvq_codebooks_from_projection:
        if not hasattr(model, "load_state_dict_without_rvq_codebooks"):
            raise TypeError(
                "model does not support retaining Q0 projections while "
                "reinitializing RVQ codebooks"
            )
        skipped_keys = model.load_state_dict_without_rvq_codebooks(
            state_dict,
            include_discriminators=not generator_only,
        )
        print(
            "Reused validated Q0 Encoder/Decoder and 64<->lookup projections; "
            "discarded all source rq.* embeddings and EMA buffers, then "
            f"initialized codebooks for topology {getattr(model, 'codebook_sizes', 'unknown')} "
            f"({len(skipped_keys)} fresh state keys)."
        )
        print("Optimizer, scheduler, codebook EMA statistics, and training step start fresh.")
    elif reinitialize_rvq_from_bypass:
        if not hasattr(model, "load_state_dict_without_rvq"):
            raise TypeError(
                "model does not support loading a bypass checkpoint while "
                "reinitializing RVQ"
            )
        skipped_keys = model.load_state_dict_without_rvq(
            state_dict,
            include_discriminators=not generator_only,
        )
        old_topology = (
            checkpoint_config.get("rq_num_quantizers", "unknown"),
            checkpoint_config.get("codebook_size", "unknown"),
            checkpoint_config.get("codebook_dim", "unknown"),
            checkpoint_config.get("rq_lookup_dim", checkpoint_config.get("codebook_dim", "unknown")),
        )
        new_topology = (
            getattr(model, "num_quantizers", "unknown"),
            getattr(model, "codebook_size", "unknown"),
            getattr(model, "codebook_dim", "unknown"),
            getattr(model, "rq_lookup_dim", "unknown"),
        )
        print(
            "Loaded non-RVQ state from a confirmed bypass checkpoint; "
            f"discarded RVQ topology {old_topology[0]}x{old_topology[1]}x"
            f"{old_topology[2]} (lookup={old_topology[3]}) and initialized current RVQ topology "
            f"{new_topology[0]}x{new_topology[1]}x{new_topology[2]} "
            f"(lookup={new_topology[3]}, cosine={getattr(model, 'rq_use_cosine_sim', False)}) "
            f"({len(skipped_keys)} state keys left freshly initialized)."
        )
        print("Optimizer, scheduler, and training step will start fresh.")
    elif generator_only:
        if not hasattr(model, "load_generator_state_dict"):
            raise TypeError("model does not support generator-only checkpoint loading")
        skipped_keys = model.load_generator_state_dict(state_dict)
        print(
            "Loaded generator/RVQ checkpoint state strictly; initialized "
            f"current discriminators from scratch ({len(skipped_keys)} state keys)."
        )
    else:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded complete checkpoint state, including all discriminators.")
    if checkpoint_config and hasattr(model, "restore_decoder_runtime_state"):
        model.restore_decoder_runtime_state(checkpoint_config)
    return checkpoint_config


def build_model(
    stage: str,
    *,
    sample_rate: int,
    strides: tuple[int, ...],
    stream_frame_size: int,
    stream_context_frames: int,
    boundary_loss_weight: float,
    boundary_loss_radius: int,
    boundary_loss_start_steps: int,
    boundary_loss_warmup_steps: int,
    stream_consistency_loss_weight: float,
    stream_consistency_loss_start_steps: int,
    stream_consistency_loss_warmup_steps: int,
    codebook_size: int | tuple[int, ...],
    num_quantizers: int,
    rq_lookup_dim: int,
    rq_use_cosine_sim: bool,
    si_sdr_loss_weight: float,
    spectral_envelope_loss_weight: float,
    formant_peak_loss_weight: float,
    voiced_highband_loss_weight: float,
    upper_highband_loss_weight: float,
    active_spectral_detail_loss_weight: float,
    active_spectral_detail_band_weights: tuple[float, ...],
    upper_highband_energy_deficit_weight: float,
    upper_highband_energy_margin_db: float,
    voiced_highband_energy_deficit_weight: float,
    voiced_highband_energy_margin_db: float,
    voiced_hf_retention_loss_weight: float,
    voiced_hf_retention_margin_db: float,
    click_loss_weight: float,
    jump_loss_weight: float,
    preemph_loss_weight: float,
    noise_floor_loss_weight: float,
    wave_mse_loss_weight: float,
    energy_loss_weight: float,
    generator_waveform_discr_loss_weights: tuple[float, ...],
    generator_stft_discr_loss_weight: float,
    stft_recon_loss_weight: float,
    frame_phase_loss_weight: float,
    gan_adversarial_max: float,
    gan_feature_max: float,
    decoder_upsample_mode: str,
    decoder_residual_scale: float,
    decoder_linear_upsample_kernel_min: int,
    decoder_interpolation_mode: str,
    decoder_split_first_upsample: bool,
    commitment_loss_weight: float | None = None,
    sync_codebook: bool | None = None,
    rq_codebook_balance_loss_weight: float = 0.,
    rq_codebook_balance_target_perplexity: float = 8.,
    rq_codebook_balance_temperature: float = 1.,
    rq_quantization_error_loss_weight: float = 0.,
    rq_continuous_teacher_loss_weight: float = 0.,
    bypass_rvq: bool = False,
    rq_projection_only: bool = False,
    recon_loss_weight_override: float | None = None,
    multi_spectral_recon_loss_weight_override: float | None = None,
    correlation_loss_weight_override: float | None = None,
    hardware_encoder_qat: bool = False,
    hardware_qat_observer_start_step: int = 0,
    hardware_qat_start_step: int = 1000,
    hardware_qat_activation_start_step: int = 2000,
    hardware_qat_warm_in_steps: int = 600,
    hardware_qat_block_interval_steps: int = 600,
    hardware_qat_observer_freeze_step: int = 1000,
    hardware_qat_ema_decay: float = 0.99,
    hardware_qat_observer: str = 'percentile',
    hardware_qat_percentile: float = 99.99,
    hardware_qat_validation_gated: bool = True,
    hardware_qat_gate_required_passes: int = 2,
    hardware_qat_group_fail_patience: int = 4,
) -> SoundStream:
    if sync_codebook is None:
        sync_codebook = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if stage == "recon_pretrain":
        recon_loss_weight = 10.
        multi_spectral_recon_loss_weight = 1.1
        correlation_loss_weight = 0.02
    elif stage == "spectral_refine":
        # Keep reconstruction objectives as anchors while allowing the
        # formant/envelope objectives to drive the refinement pass.
        recon_loss_weight = 5.
        multi_spectral_recon_loss_weight = 0.8
        correlation_loss_weight = 0.02
    elif stage == "gan_pretrain":
        recon_loss_weight = 5.
        multi_spectral_recon_loss_weight = 0.7
        correlation_loss_weight = 0.02
    else:
        recon_loss_weight = 10. if stage == "overfit" else 1.
        multi_spectral_recon_loss_weight = 0.7
        correlation_loss_weight = 0.

    if recon_loss_weight_override is not None:
        recon_loss_weight = float(recon_loss_weight_override)
    if multi_spectral_recon_loss_weight_override is not None:
        multi_spectral_recon_loss_weight = float(
            multi_spectral_recon_loss_weight_override
        )
    if correlation_loss_weight_override is not None:
        correlation_loss_weight = float(correlation_loss_weight_override)

    model_kwargs = dict(
        channels=16,
        channel_mults=(2, 4, 8, 16),
        codebook_dim=64,
        rq_lookup_dim=rq_lookup_dim,
        rq_projection_only=rq_projection_only,
        codebook_size=codebook_size,
        rq_num_quantizers=num_quantizers,
        rq_groups=1,
        # Use the classic straight-through estimator for Encoder gradients.
        # vector-quantize-pytorch falls back to standard STE when the mutually
        # exclusive rotation trick is disabled.
        rq_rotation_trick=False,
        rq_use_cosine_sim=rq_use_cosine_sim,
        rq_codebook_balance_loss_weight=rq_codebook_balance_loss_weight,
        rq_codebook_balance_target_perplexity=rq_codebook_balance_target_perplexity,
        rq_codebook_balance_temperature=rq_codebook_balance_temperature,
        rq_quantization_error_loss_weight=rq_quantization_error_loss_weight,
        rq_continuous_teacher_loss_weight=rq_continuous_teacher_loss_weight,
        use_lookup_free_quantizer=False,
        use_finite_scalar_quantizer=False,
        use_local_attn=False,
        target_sample_hz=sample_rate,
        strides=strides,
        generator_waveform_discr_loss_weights=(
            generator_waveform_discr_loss_weights
        ),
        generator_stft_discr_loss_weight=generator_stft_discr_loss_weight,
        # Zero-based Encoder Block2/3/4.  The SoundStream constructor itself
        # defaults to the legacy full-convolution topology so old checkpoint
        # configs that predate this field still rebuild correctly.
        encoder_depthwise_separable_blocks=(1, 2, 3),
        encoder_depthwise_separable_revision=3,
        # One (residual PW rank, downsample PW rank) pair per zero-based block.
        # Block1 remains dense; Block2/3/4 use hardware-aligned low-rank PW.
        encoder_low_rank_pointwise_ranks=((0, 0), (8, 16), (16, 32), (32, 64)),
        recon_loss_weight=recon_loss_weight,
        multi_spectral_recon_loss_weight=multi_spectral_recon_loss_weight,
        stft_recon_loss_weight=stft_recon_loss_weight,
        spectral_envelope_loss_weight=spectral_envelope_loss_weight,
        formant_peak_loss_weight=formant_peak_loss_weight,
        voiced_highband_loss_weight=voiced_highband_loss_weight,
        upper_highband_loss_weight=upper_highband_loss_weight,
        active_spectral_detail_loss_weight=active_spectral_detail_loss_weight,
        active_spectral_detail_band_weights=(
            active_spectral_detail_band_weights
        ),
        upper_highband_energy_deficit_weight=upper_highband_energy_deficit_weight,
        upper_highband_energy_margin_db=upper_highband_energy_margin_db,
        voiced_highband_energy_deficit_weight=voiced_highband_energy_deficit_weight,
        voiced_highband_energy_margin_db=voiced_highband_energy_margin_db,
        voiced_hf_retention_loss_weight=voiced_hf_retention_loss_weight,
        voiced_hf_retention_margin_db=voiced_hf_retention_margin_db,
        si_sdr_loss_weight=si_sdr_loss_weight,
        correlation_loss_weight=correlation_loss_weight,
        wave_mse_loss_weight=wave_mse_loss_weight,
        energy_loss_weight=energy_loss_weight,
        click_loss_weight=click_loss_weight,
        jump_loss_weight=jump_loss_weight,
        preemph_loss_weight=preemph_loss_weight,
        noise_floor_loss_weight=noise_floor_loss_weight,
        frame_phase_loss_weight=frame_phase_loss_weight,
        frame_phase_samples=320,
        commitment_loss_weight=(
            commitment_loss_weight
            if commitment_loss_weight is not None
            else (
                0.
                if stage in ("gan_pretrain", "stream_finetune", "stream_finetune_long")
                else 0.1
            )
        ),
        adversarial_loss_weight=(gan_adversarial_max if stage in GAN_STAGES else 0.),
        feature_loss_weight=(gan_feature_max if stage in GAN_STAGES else 0.),
        rq_quantize_dropout=False,
        rq_threshold_ema_dead_code=2,
        rq_kwargs=dict(sync_codebook=sync_codebook),
        bypass_rvq=bypass_rvq,
        attn_window_size=64,
        attn_dim_head=32,
        attn_heads=4,
        attn_depth=1,
        decoder_upsample_mode=decoder_upsample_mode,
        decoder_residual_scale=decoder_residual_scale,
        decoder_linear_upsample_kernel_min=decoder_linear_upsample_kernel_min,
        decoder_interpolation_mode=decoder_interpolation_mode,
        decoder_split_first_upsample=decoder_split_first_upsample,
        pad_mode="constant",
        hardware_compatible_encoder=hardware_encoder_qat,
        hardware_encoder_qat=hardware_encoder_qat,
        hardware_qat_observer_start_step=hardware_qat_observer_start_step,
        hardware_qat_start_step=hardware_qat_start_step,
        hardware_qat_activation_start_step=hardware_qat_activation_start_step,
        hardware_qat_warm_in_steps=hardware_qat_warm_in_steps,
        hardware_qat_block_interval_steps=hardware_qat_block_interval_steps,
        hardware_qat_observer_freeze_step=(
            hardware_qat_observer_freeze_step
        ),
        hardware_qat_ema_decay=hardware_qat_ema_decay,
        hardware_qat_observer=hardware_qat_observer,
        hardware_qat_percentile=hardware_qat_percentile,
        hardware_qat_validation_gated=hardware_qat_validation_gated,
        hardware_qat_gate_required_passes=hardware_qat_gate_required_passes,
        hardware_qat_group_fail_patience=hardware_qat_group_fail_patience,
    )

    if stage not in ("stream_finetune", "stream_finetune_long"):
        return SoundStream(**model_kwargs)

    return FrameStreamingSoundStream(
        stream_frame_size=stream_frame_size,
        stream_context_frames=stream_context_frames,
        boundary_loss_weight=boundary_loss_weight,
        boundary_loss_radius=boundary_loss_radius,
        boundary_loss_start_steps=boundary_loss_start_steps,
        boundary_loss_warmup_steps=boundary_loss_warmup_steps,
        stream_consistency_loss_weight=stream_consistency_loss_weight,
        stream_consistency_loss_start_steps=stream_consistency_loss_start_steps,
        stream_consistency_loss_warmup_steps=stream_consistency_loss_warmup_steps,
        **model_kwargs,
    )


def main() -> None:
    args = parse_args()
    if args.hardware_qat_observer_start_step < 0:
        raise ValueError('--hardware-qat-observer-start-step cannot be negative')
    if args.hardware_qat_start_step < args.hardware_qat_observer_start_step:
        raise ValueError(
            '--hardware-qat-start-step must be >= '
            '--hardware-qat-observer-start-step'
        )
    if args.hardware_qat_observer_freeze_step > args.hardware_qat_start_step:
        raise ValueError(
            '--hardware-qat-observer-freeze-step must be <= '
            '--hardware-qat-start-step'
        )
    if args.hardware_qat_activation_start_step < args.hardware_qat_start_step:
        raise ValueError('--hardware-qat-activation-start-step must be >= --hardware-qat-start-step')
    if args.hardware_qat_block_interval_steps < 0:
        raise ValueError('--hardware-qat-block-interval-steps cannot be negative')
    if args.hardware_qat_warm_in_steps < 0:
        raise ValueError('--hardware-qat-warm-in-steps cannot be negative')
    if not 0. < args.hardware_qat_percentile <= 100.:
        raise ValueError('--hardware-qat-percentile must be in (0, 100]')
    if args.hardware_qat_gate_required_passes < 1:
        raise ValueError('--hardware-qat-gate-required-passes must be positive')
    if args.hardware_qat_group_fail_patience < 1:
        raise ValueError('--hardware-qat-group-fail-patience must be positive')
    for name in (
        'hardware_qat_latent64_weight',
        'hardware_qat_latent32_weight',
        'hardware_qat_rvq_margin_weight',
        'hardware_qat_rvq_margin',
        'hardware_qat_rvq_margin_max',
        'hardware_qat_fixed_scale_lr',
        'hardware_qat_final_polish_lr',
        'hardware_qat_max_latent32_nmse',
        'hardware_qat_max_quantized_output_nmse',
        'hardware_qat_max_q00_index_flip',
        'hardware_qat_max_q01_index_flip',
    ):
        if getattr(args, name) < 0.:
            raise ValueError(f'--{name.replace("_", "-")} cannot be negative')
    stage_defaults = dict(STAGE_DEFAULTS[args.stage])
    stage25_decoder_only_refine = bool(args.stage25_decoder_only_refine)
    stage25_joint_recon_refine = bool(args.stage25_joint_recon_refine)
    stage25_rvq_midband_refine = bool(args.stage25_rvq_midband_refine)
    rvq_joint_adapt = bool(args.rvq_joint_adapt)
    stage25_refine_mode_count = sum((
        stage25_decoder_only_refine,
        stage25_joint_recon_refine,
        stage25_rvq_midband_refine,
        rvq_joint_adapt,
    ))
    if stage25_refine_mode_count > 1:
        raise ValueError(
            "--stage25-decoder-only-refine, --stage25-joint-recon-refine, "
            "--stage25-rvq-midband-refine, and --rvq-joint-adapt are mutually exclusive."
        )
    stage25_reconstruction_only_refine = (
        stage25_decoder_only_refine or
        stage25_joint_recon_refine or
        stage25_rvq_midband_refine or
        rvq_joint_adapt
    )
    if stage25_reconstruction_only_refine:
        # Reuse Stage-2.5 initialization and quality-retention logic while
        # selecting a reconstruction-only optimization path below.
        args.stage2_targeted_refine = True
    if args.stage2_targeted_refine:
        if args.stage != "gan_pretrain":
            raise ValueError("--stage2-targeted-refine is only valid with --stage gan_pretrain.")
        if stage25_reconstruction_only_refine:
            is_short_frameguard_refine = (
                stage25_joint_recon_refine or
                stage25_rvq_midband_refine
            )
            stage_defaults.update(
                steps=(
                    20_000
                    if rvq_joint_adapt
                    else
                    3_000
                    if stage25_rvq_midband_refine
                    else 2_000
                    if stage25_joint_recon_refine
                    else 5_000
                ),
                save_every=(
                    200
                    if stage25_rvq_midband_refine
                    else 250
                    if stage25_joint_recon_refine
                    else 500
                ),
                eval_every=(
                    100 if stage25_rvq_midband_refine else 250
                ),
                min_steps=(
                    20_000
                    if rvq_joint_adapt
                    else
                    1_000
                    if stage25_rvq_midband_refine
                    else 1_000
                    if stage25_joint_recon_refine
                    else 3_000
                ),
                patience=(
                    None
                    if rvq_joint_adapt
                    else
                    10 if stage25_rvq_midband_refine
                    else 8
                ),
                early_stopping_min_delta=(
                    0.001 if is_short_frameguard_refine else 0.002
                ),
                lr=(
                    5e-6
                    if rvq_joint_adapt
                    else
                    1e-7
                    if stage25_rvq_midband_refine
                    else args.stage25_decoder_lr
                ),
                encoder_lr=(
                    5e-7
                    if rvq_joint_adapt
                    else
                    2e-8
                    if stage25_rvq_midband_refine
                    else args.stage25_encoder_lr
                    if stage25_joint_recon_refine
                    else None
                ),
                gan_start=0,
                gan_ramp=0,
                gan_adversarial_max=0.,
                gan_feature_max=0.,
                noise_floor_loss_weight=0.03,
                spectral_envelope_loss_weight=(0. if rvq_joint_adapt else 0.05),
                voiced_highband_loss_weight=(0. if rvq_joint_adapt else 0.06),
                voiced_hf_retention_loss_weight=(0. if rvq_joint_adapt else 0.02),
                active_spectral_detail_loss_weight=(
                    0. if rvq_joint_adapt else
                    0.03 if stage25_rvq_midband_refine else 0.02
                ),
                active_spectral_detail_loss_start_steps=0,
                active_spectral_detail_loss_warmup_steps=(
                    300 if stage25_rvq_midband_refine else 0
                ),
                upper_highband_loss_weight=0.0025,
                upper_highband_loss_start_steps=0,
                upper_highband_loss_warmup_steps=0,
                frame_phase_loss_weight=(
                    0.005
                    if stage25_rvq_midband_refine
                    else 0.001
                    if stage25_joint_recon_refine
                    else 0.
                ),
                frame_phase_loss_warmup_steps=(
                    500 if stage25_joint_recon_refine else 0
                ),
                si_sdr_loss_weight=(0.07 if rvq_joint_adapt else 0.05),
                si_sdr_loss_start_steps=(0 if rvq_joint_adapt else 15_000),
                si_sdr_loss_warmup_steps=(2500 if rvq_joint_adapt else 15_000),
            )
        else:
            stage_defaults.update(
                steps=10_000,
                save_every=1_000,
                eval_every=500,
                min_steps=5_000,
                patience=12,
                early_stopping_min_delta=0.003,
                lr=args.stage25_decoder_lr,
                encoder_lr=args.stage25_encoder_lr,
                discr_lr=5e-7,
                stft_discr_lr=2.5e-7,
                waveform_discr_lrs=(5e-7, 5e-7, 2.5e-7),
                waveform_discr_update_every=(2, 4, 4),
                waveform_discr_loss_weights=(1.0, 0.25, 0.25),
                stft_discr_update_every=4,
                stft_discr_loss_weight=0.5,
                gan_start=0,
                gan_ramp=0,
                gan_adversarial_max=2e-4,
                gan_feature_max=1.5,
                noise_floor_loss_weight=0.03,
                spectral_envelope_loss_weight=0.05,
                voiced_highband_loss_weight=0.06,
                voiced_hf_retention_loss_weight=0.02,
                frame_phase_loss_weight=0.,
                frame_phase_loss_warmup_steps=0,
                si_sdr_loss_weight=0.05,
            )
        # The RVQ-midband experiment is the only Stage-2.5 preset that updates
        # Encoder weights plus codebook EMA/dead-code state. Decoder weights
        # remain fixed for the complete controlled experiment.
        args.stage2_unfreeze_encoder_rvq_step = (
            2_000 if rvq_joint_adapt else 0 if stage25_rvq_midband_refine else -1
        )
        if stage25_rvq_midband_refine:
            args.stage2_quality_retention_patience = 4
            args.stage2_rvq_retention_patience = 2
        args.stage2_generator_freeze_steps = 0
        args.stage2_generator_hold_steps = 0
        args.stage2_discriminator_hold_steps = 0
        args.stage2_discriminator_start_steps = 0
        args.stage2_recon_transition_start_steps = None
        args.stage2_recon_transition_end_steps = None
        args.stage2_max_aligned_si_sdr_drop = min(args.stage2_max_aligned_si_sdr_drop, 0.10)
        if stage25_joint_recon_refine or stage25_rvq_midband_refine:
            # The previous Decoder-only run gained only a tiny reconstruction
            # improvement while steadily increasing 50 Hz frame leakage.
            # Compare against the fixed initialization baseline from the first
            # validation and reject candidates before this drift can accumulate.
            args.stage2_max_ac320_isolated_rise = min(
                args.stage2_max_ac320_isolated_rise,
                0.0015,
            )
            args.stage2_max_comb_median_excess_db_rise = min(
                args.stage2_max_comb_median_excess_db_rise,
                0.10,
            )
            args.stage2_quality_gate_start_steps = 0
            args.stage2_best_checkpoint_min_step = (
                100 if stage25_rvq_midband_refine else 250
            )
        else:
            args.stage2_quality_gate_start_steps = 1_000
            args.stage2_best_checkpoint_min_step = 1_000

    args.boundary_loss_weight = (
        args.boundary_loss_weight
        if args.boundary_loss_weight is not None
        else stage_defaults.get("boundary_loss_weight", 0.02)
    )
    args.boundary_loss_start_steps = (
        args.boundary_loss_start_steps
        if args.boundary_loss_start_steps is not None
        else stage_defaults.get("boundary_loss_start_steps", 0)
    )
    args.boundary_loss_warmup_steps = (
        args.boundary_loss_warmup_steps
        if args.boundary_loss_warmup_steps is not None
        else stage_defaults.get("boundary_loss_warmup_steps", 0)
    )
    args.stream_consistency_loss_weight = (
        args.stream_consistency_loss_weight
        if args.stream_consistency_loss_weight is not None
        else stage_defaults.get("stream_consistency_loss_weight", 0.)
    )
    args.stream_consistency_loss_start_steps = (
        args.stream_consistency_loss_start_steps
        if args.stream_consistency_loss_start_steps is not None
        else stage_defaults.get("stream_consistency_loss_start_steps", 0)
    )
    args.stream_consistency_loss_warmup_steps = (
        args.stream_consistency_loss_warmup_steps
        if args.stream_consistency_loss_warmup_steps is not None
        else stage_defaults.get("stream_consistency_loss_warmup_steps", 0)
    )

    if args.generator_lr is not None:
        if args.generator_lr <= 0:
            raise ValueError("--generator-lr must be positive.")
        stage_defaults["lr"] = args.generator_lr
    if args.rvq_calibration_only:
        if args.bypass_rvq_during_training:
            raise ValueError(
                "--rvq-calibration-only cannot be combined with "
                "--bypass-rvq-during-training"
            )
        if args.stage != "recon_pretrain":
            raise ValueError(
                "--rvq-calibration-only requires --stage recon_pretrain"
            )
        if args.init_checkpoint is None:
            raise ValueError(
                "--rvq-calibration-only requires --init-checkpoint from the "
                "completed bypass phase"
            )
    if args.rvq_projection_only:
        if args.bypass_rvq_during_training or args.rvq_calibration_only:
            raise ValueError(
                "--rvq-projection-only is mutually exclusive with bypass and calibration modes"
            )
        if args.stage != "recon_pretrain":
            raise ValueError("--rvq-projection-only requires --stage recon_pretrain")
        if args.init_checkpoint is None:
            raise ValueError("--rvq-projection-only requires a bypass --init-checkpoint")
    if args.rvq_projection_pca_batches < 0:
        raise ValueError("--rvq-projection-pca-batches cannot be negative")
    if args.rvq_calibration_kmeans_batches < 0:
        raise ValueError("--rvq-calibration-kmeans-batches cannot be negative")
    if not 0. <= args.rvq_projection_min_evr <= 1.:
        raise ValueError("--rvq-projection-min-evr must be in [0, 1]")
    if args.rvq_projection_latent_mse_weight < 0.:
        raise ValueError("--rvq-projection-latent-mse-weight cannot be negative")
    if args.rvq_projection_latent_cosine_weight < 0.:
        raise ValueError("--rvq-projection-latent-cosine-weight cannot be negative")
    if args.rvq_projection_orth_loss_weight < 0.:
        raise ValueError("--rvq-projection-orth-loss-weight cannot be negative")
    if args.rvq_projection_tie_loss_weight < 0.:
        raise ValueError("--rvq-projection-tie-loss-weight cannot be negative")
    if args.rvq_projection_freeze_input_steps < 0:
        raise ValueError("--rvq-projection-freeze-input-steps cannot be negative")
    if args.rvq_joint_latent64_teacher_loss_weight < 0.:
        raise ValueError("--rvq-joint-latent64-teacher-loss-weight cannot be negative")
    if (
        args.state_teacher_retention_weight is not None and
        args.state_teacher_retention_weight < 0.
    ):
        raise ValueError("--state-teacher-retention-weight cannot be negative")
    if (
        args.state_decoder_trainable_from_block is not None and
        not 0 <= args.state_decoder_trainable_from_block <= 3
    ):
        raise ValueError(
            "--state-decoder-trainable-from-block must be between 0 and 3"
        )
    if args.rvq_joint_polish_decoder_lr <= 0.:
        raise ValueError("--rvq-joint-polish-decoder-lr must be positive")
    rvq_joint_boundaries = (
        args.rvq_joint_decoder_only_steps,
        args.rvq_joint_rvq_adapt_end_steps,
    )
    if any(step < 0 for step in rvq_joint_boundaries):
        raise ValueError("RVQ joint phase boundaries cannot be negative")
    if tuple(sorted(rvq_joint_boundaries)) != rvq_joint_boundaries:
        raise ValueError("RVQ joint phase boundaries must be monotonic")
    if args.rvq_plateau_freeze_start_steps < 0:
        raise ValueError("--rvq-plateau-freeze-start-steps cannot be negative")
    if args.rvq_plateau_freeze_patience < 0:
        raise ValueError("--rvq-plateau-freeze-patience cannot be negative")
    if args.rvq_plateau_freeze_min_delta < 0.:
        raise ValueError("--rvq-plateau-freeze-min-delta cannot be negative")
    if (
        args.reinitialize_rvq_from_bypass_checkpoint and
        not (args.rvq_calibration_only or args.rvq_projection_only)
    ):
        raise ValueError(
            "--reinitialize-rvq-from-bypass-checkpoint requires projection-only "
            "pretraining or RVQ calibration"
        )
    if (
        args.reinitialize_rvq_codebooks_from_projection_checkpoint and
        not args.rvq_calibration_only
    ):
        raise ValueError(
            "--reinitialize-rvq-codebooks-from-projection-checkpoint requires "
            "--rvq-calibration-only"
        )
    if (
        args.reinitialize_rvq_from_bypass_checkpoint and
        args.reinitialize_rvq_codebooks_from_projection_checkpoint
    ):
        raise ValueError("RVQ bypass and Q0 projection migration modes are mutually exclusive")
    if args.gan_adversarial_max is not None:
        if args.gan_adversarial_max < 0:
            raise ValueError("--gan-adversarial-max cannot be negative.")
        stage_defaults["gan_adversarial_max"] = args.gan_adversarial_max
    if args.rvq_warm_in_steps < 0:
        raise ValueError("--rvq-warm-in-steps cannot be negative.")
    if args.rvq_codebook_balance_loss_weight < 0.:
        raise ValueError("--rvq-codebook-balance-loss-weight cannot be negative.")
    if args.num_quantizers <= 0:
        raise ValueError("--num-quantizers must be positive.")
    codebook_sizes = (
        tuple(args.codebook_sizes)
        if args.codebook_sizes is not None
        else (args.codebook_size,) * args.num_quantizers
    )
    if len(codebook_sizes) != args.num_quantizers:
        raise ValueError(
            "--codebook-sizes must contain exactly --num-quantizers values; "
            f"got {len(codebook_sizes)} for {args.num_quantizers} levels."
        )
    if any(size <= 1 or size & (size - 1) for size in codebook_sizes):
        raise ValueError("Every RVQ codebook size must be a power of two greater than one.")
    if not 1. <= args.rvq_codebook_balance_target_perplexity <= min(codebook_sizes):
        raise ValueError(
            "--rvq-codebook-balance-target-perplexity must be in "
            f"[1, {min(codebook_sizes)}] for the smallest RVQ level."
        )
    if args.rvq_codebook_balance_temperature <= 0.:
        raise ValueError("--rvq-codebook-balance-temperature must be positive.")
    if args.rvq_quantization_error_loss_weight < 0.:
        raise ValueError("--rvq-quantization-error-loss-weight cannot be negative.")
    if args.rvq_continuous_teacher_loss_weight < 0.:
        raise ValueError("--rvq-continuous-teacher-loss-weight cannot be negative.")
    if args.rvq_joint_adapt:
        if args.stage != "gan_pretrain":
            raise ValueError("--rvq-joint-adapt requires --stage gan_pretrain.")
        if args.bypass_rvq_during_training or args.rvq_calibration_only:
            raise ValueError("--rvq-joint-adapt requires RVQ in the training path.")
        if (
            (args.num_train_steps or stage_defaults["steps"]) <
            args.rvq_joint_rvq_adapt_end_steps
        ):
            raise ValueError(
                "--rvq-joint-adapt training steps must reach the end of the "
                "configured RVQ EMA phase."
            )
    if args.gan_feature_max is not None:
        if args.gan_feature_max < 0:
            raise ValueError("--gan-feature-max cannot be negative.")
        stage_defaults["gan_feature_max"] = args.gan_feature_max
    if args.stage2_gan_start_step is not None:
        if args.stage2_gan_start_step < 0:
            raise ValueError("--stage2-gan-start-step cannot be negative.")
        stage_defaults["gan_start"] = args.stage2_gan_start_step
    if args.stage2_gan_ramp_steps is not None:
        if args.stage2_gan_ramp_steps < 0:
            raise ValueError("--stage2-gan-ramp-steps cannot be negative.")
        stage_defaults["gan_ramp"] = args.stage2_gan_ramp_steps
    if args.waveform_recon_loss_weight is not None and args.waveform_recon_loss_weight < 0.:
        raise ValueError("--waveform-recon-loss-weight cannot be negative.")
    if (
        args.multi_spectral_recon_loss_weight is not None and
        args.multi_spectral_recon_loss_weight < 0.
    ):
        raise ValueError("--multi-spectral-recon-loss-weight cannot be negative.")
    if args.stage2_teacher_retention_weight < 0.:
        raise ValueError("--stage2-teacher-retention-weight cannot be negative.")
    if args.stage2_max_gan_grad_ratio <= 0.:
        raise ValueError("--stage2-max-gan-grad-ratio must be positive.")
    if any(multiplier <= 0. for multiplier in args.stage2_decoder_lr_multipliers):
        raise ValueError("--stage2-decoder-lr-multipliers values must be positive.")

    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.si_sdr_loss_weight is not None and args.si_sdr_loss_weight < 0:
        raise ValueError("--si-sdr-loss-weight cannot be negative.")
    if args.si_sdr_loss_start_steps is not None and args.si_sdr_loss_start_steps < 0:
        raise ValueError("--si-sdr-loss-start-steps cannot be negative.")
    if args.si_sdr_loss_warmup_steps is not None and args.si_sdr_loss_warmup_steps < 0:
        raise ValueError("--si-sdr-loss-warmup-steps cannot be negative.")
    if args.boundary_loss_weight < 0:
        raise ValueError("--boundary-loss-weight cannot be negative.")
    if args.boundary_loss_start_steps < 0 or args.boundary_loss_warmup_steps < 0:
        raise ValueError("Boundary-loss schedule steps cannot be negative.")
    if args.stream_consistency_loss_weight < 0:
        raise ValueError("--stream-consistency-loss-weight cannot be negative.")
    if (
        args.stream_consistency_loss_start_steps < 0 or
        args.stream_consistency_loss_warmup_steps < 0
    ):
        raise ValueError("Stream-consistency schedule steps cannot be negative.")
    if args.spectral_envelope_loss_weight is not None and args.spectral_envelope_loss_weight < 0:
        raise ValueError("--spectral-envelope-loss-weight cannot be negative.")
    if args.formant_peak_loss_weight is not None and args.formant_peak_loss_weight < 0:
        raise ValueError("--formant-peak-loss-weight cannot be negative.")
    if (
        args.formant_peak_loss_start_steps is not None and
        args.formant_peak_loss_start_steps < 0
    ):
        raise ValueError("--formant-peak-loss-start-steps cannot be negative.")
    if (
        args.formant_peak_loss_warmup_steps is not None and
        args.formant_peak_loss_warmup_steps < 0
    ):
        raise ValueError("--formant-peak-loss-warmup-steps cannot be negative.")
    if args.loss_grad_diagnostics_every < 0:
        raise ValueError("--loss-grad-diagnostics-every cannot be negative.")
    if args.voiced_highband_loss_weight is not None and args.voiced_highband_loss_weight < 0:
        raise ValueError("--voiced-highband-loss-weight cannot be negative.")
    if args.upper_highband_loss_weight is not None and args.upper_highband_loss_weight < 0:
        raise ValueError("--upper-highband-loss-weight cannot be negative.")
    if (
        args.upper_highband_loss_start_steps is not None and
        args.upper_highband_loss_start_steps < 0
    ):
        raise ValueError("--upper-highband-loss-start-steps cannot be negative.")
    if (
        args.upper_highband_loss_warmup_steps is not None and
        args.upper_highband_loss_warmup_steps < 0
    ):
        raise ValueError("--upper-highband-loss-warmup-steps cannot be negative.")
    if (
        args.active_spectral_detail_loss_weight is not None and
        args.active_spectral_detail_loss_weight < 0
    ):
        raise ValueError(
            "--active-spectral-detail-loss-weight cannot be negative."
        )
    if (
        args.active_spectral_detail_loss_start_steps is not None and
        args.active_spectral_detail_loss_start_steps < 0
    ):
        raise ValueError(
            "--active-spectral-detail-loss-start-steps cannot be negative."
        )
    if (
        args.active_spectral_detail_loss_warmup_steps is not None and
        args.active_spectral_detail_loss_warmup_steps < 0
    ):
        raise ValueError(
            "--active-spectral-detail-loss-warmup-steps cannot be negative."
        )
    if (
        args.upper_highband_energy_deficit_weight is not None and
        args.upper_highband_energy_deficit_weight < 0
    ):
        raise ValueError("--upper-highband-energy-deficit-weight cannot be negative.")
    if (
        args.upper_highband_energy_margin_db is not None and
        args.upper_highband_energy_margin_db < 0
    ):
        raise ValueError("--upper-highband-energy-margin-db cannot be negative.")
    if args.voiced_highband_loss_start_steps is not None and args.voiced_highband_loss_start_steps < 0:
        raise ValueError("--voiced-highband-loss-start-steps cannot be negative.")
    if args.voiced_highband_loss_warmup_steps is not None and args.voiced_highband_loss_warmup_steps < 0:
        raise ValueError("--voiced-highband-loss-warmup-steps cannot be negative.")
    if args.voiced_highband_energy_deficit_weight < 0:
        raise ValueError("--voiced-highband-energy-deficit-weight cannot be negative.")
    if args.voiced_highband_energy_margin_db < 0:
        raise ValueError("--voiced-highband-energy-margin-db cannot be negative.")
    if (
        args.voiced_hf_retention_loss_weight is not None and
        args.voiced_hf_retention_loss_weight < 0
    ):
        raise ValueError("--voiced-hf-retention-loss-weight cannot be negative.")
    if args.voiced_hf_retention_margin_db < 0:
        raise ValueError("--voiced-hf-retention-margin-db cannot be negative.")
    if args.stage2_voiced_hf_score_weight < 0:
        raise ValueError("--stage2-voiced-hf-score-weight cannot be negative.")
    if args.stage2_balanced_max_aligned_si_sdr_drop < 0:
        raise ValueError(
            "--stage2-balanced-max-aligned-si-sdr-drop cannot be negative."
        )
    if (
        args.stage2_min_voiced_7k_7p8k_ratio_db >=
        args.stage2_max_voiced_7k_7p8k_ratio_db
    ):
        raise ValueError(
            "--stage2-min-voiced-7k-7p8k-ratio-db must be lower than "
            "--stage2-max-voiced-7k-7p8k-ratio-db."
        )
    if args.stage2_max_quiet_7k_7p8k_excess_db_rise < 0:
        raise ValueError(
            "--stage2-max-quiet-7k-7p8k-excess-db-rise cannot be negative."
        )
    if args.stage2_upper_highband_score_weight < 0:
        raise ValueError("--stage2-upper-highband-score-weight cannot be negative.")
    if args.stage2_active_spectral_score_weight < 0:
        raise ValueError("--stage2-active-spectral-score-weight cannot be negative.")
    if args.stft_recon_loss_weight is not None and args.stft_recon_loss_weight < 0:
        raise ValueError("--stft-recon-loss-weight cannot be negative.")
    if args.stft_recon_loss_start_steps is not None and args.stft_recon_loss_start_steps < 0:
        raise ValueError("--stft-recon-loss-start-steps cannot be negative.")
    if args.stft_recon_loss_warmup_steps is not None and args.stft_recon_loss_warmup_steps < 0:
        raise ValueError("--stft-recon-loss-warmup-steps cannot be negative.")
    if args.frame_phase_loss_weight is not None and args.frame_phase_loss_weight < 0:
        raise ValueError("--frame-phase-loss-weight cannot be negative.")
    if args.frame_phase_loss_warmup_steps is not None and args.frame_phase_loss_warmup_steps < 0:
        raise ValueError("--frame-phase-loss-warmup-steps cannot be negative.")
    if (
        args.spectral_envelope_loss_start_steps is not None and
        args.spectral_envelope_loss_start_steps < 0
    ):
        raise ValueError("--spectral-envelope-loss-start-steps cannot be negative.")
    if (
        args.spectral_envelope_loss_warmup_steps is not None and
        args.spectral_envelope_loss_warmup_steps < 0
    ):
        raise ValueError("--spectral-envelope-loss-warmup-steps cannot be negative.")
    if args.decoder_linear_upsample_kernel_min < 0:
        raise ValueError("--decoder-linear-upsample-kernel-min cannot be negative.")
    if args.decoder_residual_scale_start < 0:
        raise ValueError("--decoder-residual-scale-start cannot be negative.")
    if args.decoder_residual_scale_end < 0:
        raise ValueError("--decoder-residual-scale-end cannot be negative.")
    if args.decoder_residual_scale_warmup_start_steps < 0:
        raise ValueError("--decoder-residual-scale-warmup-start-steps cannot be negative.")
    if args.decoder_residual_scale_warmup_end_steps < args.decoder_residual_scale_warmup_start_steps:
        raise ValueError(
            "--decoder-residual-scale-warmup-end-steps must be >= "
            "--decoder-residual-scale-warmup-start-steps."
        )
    if args.plateau_start_steps < 0:
        raise ValueError("--plateau-start-steps cannot be negative.")
    if not 0. < args.plateau_factor < 1.:
        raise ValueError("--plateau-factor must be between 0 and 1.")
    if args.plateau_patience <= 0:
        raise ValueError("--plateau-patience must be greater than zero.")
    if args.plateau_threshold < 0:
        raise ValueError("--plateau-threshold cannot be negative.")
    if args.plateau_cooldown < 0:
        raise ValueError("--plateau-cooldown cannot be negative.")
    if args.plateau_min_lr < 0:
        raise ValueError("--plateau-min-lr cannot be negative.")
    if args.plateau_unclean_grace_checks < 0:
        raise ValueError("--plateau-unclean-grace-checks cannot be negative.")
    if args.stage2_plateau_start_steps < 0:
        raise ValueError("--stage2-plateau-start-steps cannot be negative.")
    if not 0. < args.stage2_plateau_factor < 1.:
        raise ValueError("--stage2-plateau-factor must be between 0 and 1.")
    if args.stage2_plateau_patience <= 0:
        raise ValueError("--stage2-plateau-patience must be greater than zero.")
    if args.stage2_plateau_threshold < 0:
        raise ValueError("--stage2-plateau-threshold cannot be negative.")
    if args.stage2_plateau_cooldown < 0:
        raise ValueError("--stage2-plateau-cooldown cannot be negative.")
    if args.stage2_plateau_min_lr < 0:
        raise ValueError("--stage2-plateau-min-lr cannot be negative.")
    if args.stage2_plateau_discr_min_lr < 0:
        raise ValueError("--stage2-plateau-discr-min-lr cannot be negative.")
    if args.stage2_plateau_stft_discr_min_lr < 0:
        raise ValueError("--stage2-plateau-stft-discr-min-lr cannot be negative.")
    if args.stft_discr_lr is not None and args.stft_discr_lr <= 0:
        raise ValueError("--stft-discr-lr must be positive.")
    if args.waveform_discr_lrs is not None and any(lr <= 0 for lr in args.waveform_discr_lrs):
        raise ValueError("--waveform-discr-lrs values must all be positive.")
    if (
        args.waveform_discr_update_every is not None and
        any(interval <= 0 for interval in args.waveform_discr_update_every)
    ):
        raise ValueError("--waveform-discr-update-every values must all be positive.")
    if args.stft_discr_update_every is not None and args.stft_discr_update_every <= 0:
        raise ValueError("--stft-discr-update-every must be positive.")
    if (
        args.waveform_discr_loss_weights is not None and
        any(weight < 0 for weight in args.waveform_discr_loss_weights)
    ):
        raise ValueError("--waveform-discr-loss-weights values cannot be negative.")
    if args.stft_discr_loss_weight is not None and args.stft_discr_loss_weight < 0:
        raise ValueError("--stft-discr-loss-weight cannot be negative.")
    if args.gan_grad_diagnostics_every < 0:
        raise ValueError("--gan-grad-diagnostics-every cannot be negative.")
    if args.discr_max_grad_norm <= 0:
        raise ValueError("--discr-max-grad-norm must be positive.")
    if args.clean_gate_max_click_excess < 0:
        raise ValueError("--clean-gate-max-click-excess cannot be negative.")
    if args.stage2_unfreeze_encoder_rvq_step < -1:
        raise ValueError("--stage2-encoder-unfreeze-step must be -1 or non-negative.")
    if not 0 <= args.stage2_encoder_trainable_from_block <= 3:
        raise ValueError("--stage2-encoder-trainable-from-block must be in [0, 3].")
    if args.stage2_encoder_lr <= 0.:
        raise ValueError("--stage2-encoder-lr must be positive.")
    if args.stage2_generator_hold_steps < 0:
        raise ValueError("--stage2-generator-hold-steps cannot be negative.")
    if args.stage2_generator_hold_lr <= 0:
        raise ValueError("--stage2-generator-hold-lr must be positive.")
    if args.stage2_max_voiced_hf_ratio_db_drop < 0:
        raise ValueError("--stage2-max-voiced-hf-ratio-db-drop cannot be negative.")
    if args.stage2_max_voiced_hf_ratio_db_rise < 0:
        raise ValueError("--stage2-max-voiced-hf-ratio-db-rise cannot be negative.")
    if args.stage2_generator_freeze_steps < 0:
        raise ValueError("--stage2-generator-freeze-steps cannot be negative.")
    if args.stage2_phase2_start_step < 0 or args.stage2_phase3_start_step < 0:
        raise ValueError("Stage-2 phase boundaries cannot be negative.")
    if args.stage2_phase3_start_step <= args.stage2_phase2_start_step:
        raise ValueError(
            "--stage2-phase3-start-step must be greater than "
            "--stage2-phase2-start-step."
        )
    for name in (
        "stage2_phase2_generator_lr",
        "stage2_phase3_generator_lr",
    ):
        if getattr(args, name) <= 0.:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in (
        "stage2_phase3_gan_adversarial_max",
        "stage2_phase3_gan_feature_max",
    ):
        if getattr(args, name) < 0.:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if args.stage2_discriminator_hold_steps < 0:
        raise ValueError("--stage2-discriminator-hold-steps cannot be negative.")
    if args.stage2_discriminator_hold_lr <= 0:
        raise ValueError("--stage2-discriminator-hold-lr must be positive.")
    if (
        args.stage2_discriminator_start_steps is not None and
        args.stage2_discriminator_start_steps < 0
    ):
        raise ValueError("--stage2-discriminator-start-steps cannot be negative.")
    transition_start = args.stage2_recon_transition_start_steps
    transition_end = args.stage2_recon_transition_end_steps
    if (transition_start is None) != (transition_end is None):
        raise ValueError(
            "--stage2-recon-transition-start-steps and "
            "--stage2-recon-transition-end-steps must be supplied together."
        )
    if transition_start is not None:
        if transition_start < 0:
            raise ValueError("--stage2-recon-transition-start-steps cannot be negative.")
        if transition_end < transition_start:
            raise ValueError(
                "--stage2-recon-transition-end-steps must be >= "
                "--stage2-recon-transition-start-steps."
            )
    if args.stage2_quality_gate_start_steps < 0:
        raise ValueError("--stage2-quality-gate-start-steps cannot be negative.")
    if args.stage2_best_checkpoint_min_step < 0:
        raise ValueError("--stage2-best-checkpoint-min-step cannot be negative.")
    for name in (
        "stage2_max_click_score_rise",
        "stage2_max_ac320_isolated_rise",
        "stage2_max_comb_median_excess_db_rise",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if args.waveform_r1_every < 0 or args.stft_r1_every < 0:
        raise ValueError("R1 intervals cannot be negative.")
    if args.waveform_r1_gamma < 0 or args.stft_r1_gamma < 0:
        raise ValueError("R1 gamma values cannot be negative.")
    if args.click_loss_weight is not None and args.click_loss_weight < 0:
        raise ValueError("--click-loss-weight cannot be negative.")
    if args.jump_loss_weight is not None and args.jump_loss_weight < 0:
        raise ValueError("--jump-loss-weight cannot be negative.")
    if args.preemph_loss_weight is not None and args.preemph_loss_weight < 0:
        raise ValueError("--preemph-loss-weight cannot be negative.")
    if args.noise_floor_loss_weight is not None and args.noise_floor_loss_weight < 0:
        raise ValueError("--noise-floor-loss-weight cannot be negative.")
    if (
        args.transient_loss_warmup_steps is not None and
        args.transient_loss_warmup_steps < 0
    ):
        raise ValueError("--transient-loss-warmup-steps cannot be negative.")
    if not -1. <= args.clean_gate_min_aligned_corr <= 1.:
        raise ValueError("--clean-gate-min-aligned-corr must be between -1 and 1.")
    if not 0. <= args.clean_gate_max_negative_fraction <= 1.:
        raise ValueError("--clean-gate-max-negative-fraction must be between 0 and 1.")
    if args.clean_gate_min_rms_ratio <= 0:
        raise ValueError("--clean-gate-min-rms-ratio must be positive.")
    if args.clean_gate_max_rms_ratio < args.clean_gate_min_rms_ratio:
        raise ValueError("--clean-gate-max-rms-ratio must be >= --clean-gate-min-rms-ratio.")
    if args.clean_gate_max_voiced_hf_ratio_db < args.clean_gate_min_voiced_hf_ratio_db:
        raise ValueError(
            "--clean-gate-max-voiced-hf-ratio-db must be >= "
            "--clean-gate-min-voiced-hf-ratio-db."
        )
    for name in (
        "clean_gate_max_recon_peak",
        "clean_gate_max_recon_clip_fraction",
        "clean_gate_max_click_score",
        "clean_gate_max_jump_ratio",
        "clean_gate_max_p999_jump_ratio",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    for name in (
        "stage2_max_aligned_si_sdr_drop",
        "stage2_max_aligned_corr_drop",
        "stage2_max_quiet_hf_excess_db_rise",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if args.stage2_targeted_refine:
        if stage_defaults["lr"] <= 0:
            raise ValueError("Stage-2.5 Decoder LR must be positive.")
        if (
            not stage25_decoder_only_refine and
            stage_defaults["encoder_lr"] <= 0
        ):
            raise ValueError("--stage25-encoder-lr must be positive.")
        if (
            not stage25_decoder_only_refine and
            stage_defaults["encoder_lr"] > stage_defaults["lr"]
        ):
            raise ValueError(
                "Stage-2.5 Encoder LR must not exceed the effective Decoder LR."
            )
    for name in (
        "stage1_rvq_retention_patience",
        "stage2_quality_retention_patience",
        "stage2_rvq_retention_patience",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero.")
    if args.dl_num_workers < 0:
        raise ValueError("--dl-num-workers cannot be negative.")
    seed_everything(args.seed)

    audio_dir = args.audio_dir.resolve()
    default_results_dir = STAGE_RESULTS_DIRS[args.stage]
    results_dir = (args.results_dir or default_results_dir).resolve()
    save_model_every = args.save_model_every or stage_defaults["save_every"]
    best_eval_every = args.best_eval_every or stage_defaults["eval_every"]
    num_train_steps = args.num_train_steps or stage_defaults["steps"]
    if (
        args.stage == "gan_pretrain" and
        args.stage2_plateau_lr and
        not args.stage2_targeted_refine and
        args.stage2_plateau_min_lr > stage_defaults["lr"]
    ):
        raise ValueError(
            "--stage2-plateau-min-lr cannot exceed the Stage-2 generator LR "
            f"({stage_defaults['lr']:.3e})."
        )
    if args.stage2_targeted_refine:
        first_gan_candidate_step = (
            stage_defaults["gan_start"] +
            (stage_defaults["gan_ramp"] + 1) // 2
        )
        if (
            not stage25_reconstruction_only_refine and
            num_train_steps <= first_gan_candidate_step
        ):
            raise ValueError(
                "--stage2-targeted-refine must run beyond step "
                f"{first_gan_candidate_step} so GAN ramp can reach 0.5 and "
                "best_gan_balanced.pt can become eligible."
            )
    batch_size = args.batch_size or stage_defaults["batch_size"]
    segment_seconds = (
        args.segment_seconds
        if args.segment_seconds is not None
        else stage_defaults["segment_seconds"]
    )
    stft_discr_lr = (
        args.stft_discr_lr
        if args.stft_discr_lr is not None
        else stage_defaults.get("stft_discr_lr", stage_defaults["discr_lr"])
    )
    waveform_discr_lrs = tuple(
        args.waveform_discr_lrs
        if args.waveform_discr_lrs is not None
        else stage_defaults.get(
            "waveform_discr_lrs",
            (stage_defaults["discr_lr"],) * 3,
        )
    ) if stage_defaults["discr_lr"] is not None else None
    waveform_discr_update_every = tuple(
        args.waveform_discr_update_every
        if args.waveform_discr_update_every is not None
        else stage_defaults.get("waveform_discr_update_every", (1, 1, 1))
    )
    waveform_discr_loss_weights = tuple(
        args.waveform_discr_loss_weights
        if args.waveform_discr_loss_weights is not None
        else stage_defaults.get("waveform_discr_loss_weights", (1., 1., 1.))
    )
    stft_discr_update_every = (
        args.stft_discr_update_every
        if args.stft_discr_update_every is not None
        else stage_defaults.get("stft_discr_update_every", 1)
    )
    stft_discr_loss_weight = (
        args.stft_discr_loss_weight
        if args.stft_discr_loss_weight is not None
        else stage_defaults.get("stft_discr_loss_weight", 1.)
    )
    voiced_hf_retention_loss_weight = (
        args.voiced_hf_retention_loss_weight
        if args.voiced_hf_retention_loss_weight is not None
        else stage_defaults.get("voiced_hf_retention_loss_weight", 0.)
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if num_train_steps <= 0:
        raise ValueError("--num-train-steps must be greater than zero.")
    early_stopping_patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else stage_defaults["patience"]
    )
    early_stopping_min_steps = (
        args.early_stopping_min_steps
        if args.early_stopping_min_steps is not None
        else stage_defaults["min_steps"]
    )
    early_stopping_min_delta = (
        args.early_stopping_min_delta
        if args.early_stopping_min_delta is not None
        else stage_defaults.get("early_stopping_min_delta", 0.)
    )
    effective_early_stopping_min_steps = early_stopping_min_steps
    if args.stage == "gan_pretrain":
        # A bounded Stage-2 diagnostic may intentionally finish before the
        # complete GAN / reconstruction / plateau schedule.  Those schedule
        # endpoints control when patience-based early stopping becomes
        # eligible; they must not make an otherwise valid short run illegal.
        # Capping only the validation floor does not shorten the actual
        # schedules used by the trainer.
        gan_schedule_end = min(
            stage_defaults["gan_start"] + stage_defaults["gan_ramp"],
            num_train_steps,
        )
        plateau_schedule_start = (
            min(args.stage2_plateau_start_steps, num_train_steps)
            if args.stage2_plateau_lr and not args.stage2_targeted_refine
            else 0
        )
        recon_transition_end = min(
            args.stage2_recon_transition_end_steps or 0,
            num_train_steps,
        )
        effective_early_stopping_min_steps = max(
            effective_early_stopping_min_steps,
            plateau_schedule_start,
            recon_transition_end,
            gan_schedule_end,
        )
    if early_stopping_min_steps < 0:
        raise ValueError("--early-stopping-min-steps cannot be negative.")
    if early_stopping_min_steps > num_train_steps:
        raise ValueError(
            "--early-stopping-min-steps cannot exceed --num-train-steps."
        )
    if effective_early_stopping_min_steps > num_train_steps:
        raise ValueError(
            "The effective early-stopping start step exceeds --num-train-steps. "
            "Increase the training length or shorten the Stage-2 schedule."
        )

    audio_files = [
        path
        for path in audio_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    if not audio_files:
        raise FileNotFoundError(
            f"No supported audio files found under: {audio_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            "Select E:\\lyra\\.venv\\Scripts\\python.exe in PyCharm."
        )

    torch.set_float32_matmul_precision("high")

    sample_rate = 16_000
    strides = (2, 4, 5, 8)
    stream_frame_size = math.prod(strides)
    stream_context_frames = args.stream_context_frames
    # Default deployment profile: q0 uses K256 and q1-q8 use K128.  At 50
    # frames/s this is 8 + 8*7 = 64 bits/frame = exactly 3.2 kbps.  The codec
    # representation remains 64-D while lookup happens in 32-D.
    codebook_sizes = (
        tuple(args.codebook_sizes)
        if args.codebook_sizes is not None
        else (args.codebook_size,) * args.num_quantizers
    )
    codebook_size = (
        codebook_sizes[0]
        if len(set(codebook_sizes)) == 1
        else codebook_sizes
    )
    num_quantizers = args.num_quantizers
    rq_lookup_dim = args.rq_lookup_dim
    if args.stage in RECONSTRUCTION_STAGES:
        si_sdr_loss_weight = (
            args.si_sdr_loss_weight
            if args.si_sdr_loss_weight is not None
            else stage_defaults.get("si_sdr_loss_weight", 0.)
        )
        si_sdr_loss_start_steps = (
            args.si_sdr_loss_start_steps
            if args.si_sdr_loss_start_steps is not None
            else stage_defaults.get("si_sdr_loss_start_steps", 0)
        )
        si_sdr_loss_warmup_steps = (
            args.si_sdr_loss_warmup_steps
            if args.si_sdr_loss_warmup_steps is not None
            else stage_defaults.get("si_sdr_loss_warmup_steps", 0)
        )
    else:
        si_sdr_loss_weight = 0.
        si_sdr_loss_start_steps = 0
        si_sdr_loss_warmup_steps = 0
    click_loss_weight = (
        args.click_loss_weight
        if args.click_loss_weight is not None
        else stage_defaults["click_loss_weight"]
    )
    jump_loss_weight = (
        args.jump_loss_weight
        if args.jump_loss_weight is not None
        else stage_defaults["jump_loss_weight"]
    )
    preemph_loss_weight = (
        args.preemph_loss_weight
        if args.preemph_loss_weight is not None
        else stage_defaults.get("preemph_loss_weight", 0.)
    )
    noise_floor_loss_weight = (
        args.noise_floor_loss_weight
        if args.noise_floor_loss_weight is not None
        else stage_defaults.get("noise_floor_loss_weight", 0.)
    )
    wave_mse_loss_weight = stage_defaults.get("wave_mse_loss_weight", 0.3)
    energy_loss_weight = stage_defaults.get("energy_loss_weight", 0.1)
    spectral_envelope_loss_weight = (
        args.spectral_envelope_loss_weight
        if args.spectral_envelope_loss_weight is not None
        else stage_defaults["spectral_envelope_loss_weight"]
    )
    spectral_envelope_loss_start_steps = (
        args.spectral_envelope_loss_start_steps
        if args.spectral_envelope_loss_start_steps is not None
        else stage_defaults.get("spectral_envelope_loss_start_steps", 0)
    )
    spectral_envelope_loss_warmup_steps = (
        args.spectral_envelope_loss_warmup_steps
        if args.spectral_envelope_loss_warmup_steps is not None
        else stage_defaults.get("spectral_envelope_loss_warmup_steps", 0)
    )
    formant_peak_loss_weight = (
        args.formant_peak_loss_weight
        if args.formant_peak_loss_weight is not None
        else stage_defaults.get("formant_peak_loss_weight", 0.)
    )
    formant_peak_loss_start_steps = (
        args.formant_peak_loss_start_steps
        if args.formant_peak_loss_start_steps is not None
        else stage_defaults.get("formant_peak_loss_start_steps", 0)
    )
    formant_peak_loss_warmup_steps = (
        args.formant_peak_loss_warmup_steps
        if args.formant_peak_loss_warmup_steps is not None
        else stage_defaults.get("formant_peak_loss_warmup_steps", 0)
    )
    voiced_highband_loss_weight = (
        args.voiced_highband_loss_weight
        if args.voiced_highband_loss_weight is not None
        else stage_defaults.get("voiced_highband_loss_weight", 0.)
    )
    upper_highband_loss_weight = (
        args.upper_highband_loss_weight
        if args.upper_highband_loss_weight is not None
        else stage_defaults.get("upper_highband_loss_weight", 0.)
    )
    upper_highband_energy_deficit_weight = (
        args.upper_highband_energy_deficit_weight
        if args.upper_highband_energy_deficit_weight is not None
        else stage_defaults.get("upper_highband_energy_deficit_weight", 0.)
    )
    upper_highband_energy_margin_db = (
        args.upper_highband_energy_margin_db
        if args.upper_highband_energy_margin_db is not None
        else stage_defaults.get("upper_highband_energy_margin_db", 0.50)
    )
    upper_highband_loss_start_steps = (
        args.upper_highband_loss_start_steps
        if args.upper_highband_loss_start_steps is not None
        else stage_defaults.get("upper_highband_loss_start_steps", 0)
    )
    upper_highband_loss_warmup_steps = (
        args.upper_highband_loss_warmup_steps
        if args.upper_highband_loss_warmup_steps is not None
        else stage_defaults.get("upper_highband_loss_warmup_steps", 0)
    )
    active_spectral_detail_loss_weight = (
        args.active_spectral_detail_loss_weight
        if args.active_spectral_detail_loss_weight is not None
        else stage_defaults.get("active_spectral_detail_loss_weight", 0.)
    )
    active_spectral_detail_loss_start_steps = (
        args.active_spectral_detail_loss_start_steps
        if args.active_spectral_detail_loss_start_steps is not None
        else stage_defaults.get(
            "active_spectral_detail_loss_start_steps",
            0
        )
    )
    active_spectral_detail_loss_warmup_steps = (
        args.active_spectral_detail_loss_warmup_steps
        if args.active_spectral_detail_loss_warmup_steps is not None
        else stage_defaults.get(
            "active_spectral_detail_loss_warmup_steps",
            0
        )
    )
    voiced_highband_loss_start_steps = (
        args.voiced_highband_loss_start_steps
        if args.voiced_highband_loss_start_steps is not None
        else stage_defaults.get("voiced_highband_loss_start_steps", 0)
    )
    voiced_highband_loss_warmup_steps = (
        args.voiced_highband_loss_warmup_steps
        if args.voiced_highband_loss_warmup_steps is not None
        else stage_defaults.get("voiced_highband_loss_warmup_steps", 0)
    )
    transient_loss_warmup_steps = (
        args.transient_loss_warmup_steps
        if args.transient_loss_warmup_steps is not None
        else stage_defaults["transient_loss_warmup_steps"]
    )
    stft_recon_loss_weight = (
        args.stft_recon_loss_weight
        if args.stft_recon_loss_weight is not None
        else stage_defaults["stft_recon_loss_weight"]
    )
    stft_recon_loss_start_steps = (
        args.stft_recon_loss_start_steps
        if args.stft_recon_loss_start_steps is not None
        else stage_defaults.get("stft_recon_loss_start_steps", 0)
    )
    stft_recon_loss_warmup_steps = (
        args.stft_recon_loss_warmup_steps
        if args.stft_recon_loss_warmup_steps is not None
        else stage_defaults.get("stft_recon_loss_warmup_steps", 0)
    )
    frame_phase_loss_weight = (
        args.frame_phase_loss_weight
        if args.frame_phase_loss_weight is not None
        else stage_defaults.get("frame_phase_loss_weight", 0.)
    )
    frame_phase_loss_warmup_steps = (
        args.frame_phase_loss_warmup_steps
        if args.frame_phase_loss_warmup_steps is not None
        else stage_defaults.get("frame_phase_loss_warmup_steps", 0)
    )
    waveform_recon_loss_weight = (
        args.waveform_recon_loss_weight
        if args.waveform_recon_loss_weight is not None
        else 7.5 if rvq_joint_adapt
        else 10.0 if args.stage in ("overfit", "recon_pretrain")
        else 5.0 if args.stage in ("spectral_refine", "gan_pretrain")
        else stage_defaults.get("waveform_recon_loss_weight", 1.0)
    )
    multi_spectral_recon_loss_weight = (
        args.multi_spectral_recon_loss_weight
        if args.multi_spectral_recon_loss_weight is not None
        else 0.6 if rvq_joint_adapt
        else 1.1 if args.stage == "recon_pretrain"
        else 0.8 if args.stage == "spectral_refine"
        else stage_defaults.get("multi_spectral_recon_loss_weight", 0.7)
    )
    correlation_loss_weight = (
        stage_defaults.get(
            "correlation_loss_weight",
            0.02 if args.stage in (*RECONSTRUCTION_STAGES, "gan_pretrain") else 0.0,
        )
    )
    decoder_residual_scale_start = args.decoder_residual_scale_start
    decoder_residual_scale_end = args.decoder_residual_scale_end
    decoder_residual_scale_warmup_start_steps = args.decoder_residual_scale_warmup_start_steps
    decoder_residual_scale_warmup_end_steps = args.decoder_residual_scale_warmup_end_steps
    if args.rvq_projection_only:
        # Loading the bypass checkpoint below restores its exact block scales.
        # Q0 must not apply the ordinary Stage-1 residual-scale schedule.
        decoder_residual_scale_start = decoder_residual_scale_end
        decoder_residual_scale_warmup_start_steps = 0
        decoder_residual_scale_warmup_end_steps = 0
    stage1_plateau_lr_enabled = (
        args.stage == "recon_pretrain" and
        args.stage1_plateau_lr
    )
    stage2_plateau_lr_enabled = (
        args.stage == "gan_pretrain" and
        args.stage2_plateau_lr and
        not args.stage2_targeted_refine
    )

    bitrate = calculate_bitrate(
        sample_rate=sample_rate,
        strides=strides,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training stage: {args.stage}")
    print(f"Audio directory: {audio_dir}")
    print(f"Audio files: {len(audio_files)}")
    print(f"Results directory: {results_dir}")
    print(f"Save model every: {save_model_every} steps")
    print(f"Best eval every: {best_eval_every} steps")
    print(f"Maximum training steps: {num_train_steps}")
    print(f"Generator learning rate: {stage_defaults['lr']}")
    print(
        "Encoder DSCNN low-rank revision 3: Block2/3/4 use Depthwise followed "
        "by rank-factorized Pointwise convolutions; residual ranks=8/16/32, "
        "downsample ranks=16/32/64; existing ReLU positions are unchanged."
    )
    print(
        "RVQ gradient estimator: standard STE "
        "(rotation_trick=False; codebook remains EMA-updated)"
    )
    if rvq_joint_adapt:
        print(
            "B1.5 curriculum: 0-"
            f"{args.rvq_joint_decoder_only_steps} Decoder-only; "
            f"{args.rvq_joint_decoder_only_steps}-"
            f"{args.rvq_joint_rvq_adapt_end_steps} RVQ EMA + Decoder "
            f"(plateau freeze from step {args.rvq_plateau_freeze_start_steps}, "
            f"patience={args.rvq_plateau_freeze_patience}, "
            f"lookup-NMSE min_delta={args.rvq_plateau_freeze_min_delta:g}); "
            f">={args.rvq_joint_rvq_adapt_end_steps} Decoder-only polish at "
            f"LR={args.rvq_joint_polish_decoder_lr:.3e}. "
            "Encoder and both projections remain frozen; "
            "SI-SDR starts immediately; GAN and HF-detail losses are disabled."
        )
    elif stage25_decoder_only_refine:
        print(
            "Stage-2.5 optimizer groups: "
            f"Decoder LR={stage_defaults['lr']:.3e}; "
            "Encoder and RVQ excluded from the generator optimizer; GAN disabled."
        )
    elif stage25_rvq_midband_refine:
        print(
            "Stage-2.5 RVQ-midband optimizer groups: "
            f"Encoder LR={stage_defaults['encoder_lr']:.3e}; "
            "RVQ EMA enabled, full Decoder frozen for the entire run, "
            "GAN disabled."
        )
    elif stage25_joint_recon_refine:
        print(
            "Stage-2.5 optimizer groups: "
            f"Decoder LR={stage_defaults['lr']:.3e}, "
            f"Encoder LR={stage_defaults['encoder_lr']:.3e}; "
            "RVQ excluded from the generator optimizer; GAN disabled."
        )
    elif args.stage2_targeted_refine:
        print(
            "Stage-2.5 optimizer groups: "
            f"Decoder LR={stage_defaults['lr']:.3e}, "
            f"Encoder LR={stage_defaults['encoder_lr']:.3e}, "
            "RVQ excluded from the generator optimizer."
        )
    if args.rvq_projection_only:
        print(
            "Q0 checkpoint eligibility begins at step 0; Decoder residual "
            "scales stay fixed at the init-checkpoint values."
        )
    elif args.stage == "recon_pretrain":
        if stage1_plateau_lr_enabled:
            print(
                "Stage-1 LR schedule: "
                "linear warmup for steps [0, 1000), "
                "then ReduceLROnPlateau on validation "
                "online_aligned_si_sdr "
                f"from step {args.plateau_start_steps} "
                f"(factor={args.plateau_factor}, "
                f"patience={args.plateau_patience}, "
                f"threshold={args.plateau_threshold}, "
                f"cooldown={args.plateau_cooldown}, "
                f"min_lr={args.plateau_min_lr}, "
                f"unclean_grace_checks={args.plateau_unclean_grace_checks})"
            )
        else:
            print(
                "Stage-1 LR schedule: "
                "linear warmup for steps [0, 1000), "
                "2.000e-04 for [1000, 20000), "
                "1.000e-04 for [20000, 35000), "
                "5.000e-05 from step 35000"
            )
    elif args.stage == "spectral_refine":
        print(
            "Stage-1.5 LR schedule: fixed "
            f"{stage_defaults['lr']:.3e} after the standard warmup."
        )
    elif args.stage == "gan_pretrain":
        waveform_lr_text = "/".join(f"{lr:.3e}" for lr in waveform_discr_lrs)
        waveform_update_text = "/".join(str(value) for value in waveform_discr_update_every)
        if stage2_plateau_lr_enabled:
            print(
                f"Stage-2 LR schedule: standard warmup, then generator={stage_defaults['lr']:.3e} / "
                f"waveform discriminators={waveform_lr_text} / STFT discriminator="
                f"{stft_discr_lr:.3e} until step "
                f"{args.stage2_plateau_start_steps}; validation ReduceLROnPlateau "
                "then lowers the generator on the HF-penalized composite score "
                "(10*wave + 1.1*Mel + voiced-HF gate penalties + "
                f"{args.stage2_active_spectral_score_weight:g}*active spectral "
                "detail; lower is better) "
                f"(factor={args.stage2_plateau_factor}, "
                f"patience={args.stage2_plateau_patience}, "
                f"threshold={args.stage2_plateau_threshold}, "
                f"cooldown={args.stage2_plateau_cooldown}, "
                f"min_g_lr={args.stage2_plateau_min_lr})"
            )
        else:
            print(
                f"Stage-2 LR schedule: fixed generator={stage_defaults['lr']:.3e}, "
                f"waveform discriminators={waveform_lr_text}, STFT discriminator="
                f"{stft_discr_lr:.3e} after the standard warmup."
            )
        print(
            "Stage-2 discriminator update intervals (scale1/scale0.5/scale0.25/STFT): "
            f"{waveform_update_text}/{stft_discr_update_every}."
        )
        print(
            "Stage-2 normalized discriminator loss weights "
            "(scale1/scale0.5/scale0.25/STFT): "
            f"{'/'.join(f'{weight:g}' for weight in waveform_discr_loss_weights)}/"
            f"{stft_discr_loss_weight:g}."
        )
        if args.stage2_targeted_refine:
            if stage25_reconstruction_only_refine:
                trainable_text = (
                    "Decoder updates from step 0; Encoder remains frozen"
                    if stage25_decoder_only_refine
                    else (
                        "Encoder/RVQ update from step 0; full Decoder remains "
                        "frozen for the complete run"
                    )
                    if stage25_rvq_midband_refine
                    else "Encoder and Decoder update from step 0"
                )
                rvq_text = (
                    "RVQ EMA/dead-code updates are enabled"
                    if stage25_rvq_midband_refine
                    else "RVQ remains frozen"
                )
                print(
                    f"Stage-2.5 reconstruction-only transition: {trainable_text}; "
                    f"{rvq_text}; waveform/STFT discriminators are loaded "
                    "only as checkpoint state and receive no updates; GAN losses "
                    "are disabled."
                )
            else:
                print(
                    "Stage-2.5 transition: Encoder and Decoder update from step 0; "
                    "RVQ remains frozen; inherited waveform/STFT discriminator weights "
                    "continue from step 0 with fresh optimizer states; GAN weights stay "
                    "at their Stage-2 endpoint values."
                )
        else:
            print(
                "Stage-2 retention phase: Generator frozen through step "
                f"{args.stage2_generator_freeze_steps}; discriminator updates begin at step "
                f"{args.stage2_discriminator_start_steps if args.stage2_discriminator_start_steps is not None else stage_defaults['gan_start']}. "
                "RVQ remains frozen throughout; Encoder is "
                + (
                    "frozen throughout decoder-only training"
                    if args.stage2_unfreeze_encoder_rvq_step < 0
                    else (
                        f"frozen through step {args.stage2_unfreeze_encoder_rvq_step}, "
                        f"then Block{args.stage2_encoder_trainable_from_block + 1} "
                        "through the final latent convolution train at "
                        f"LR={args.stage2_encoder_lr:.3e}"
                    )
                )
                + "; generator adversarial/feature losses begin after step "
                f"{stage_defaults['gan_start']}; generator LR releases linearly from "
                f"{args.stage2_generator_hold_lr:.3e} to {stage_defaults['lr']:.3e} "
                f"by step {args.stage2_generator_hold_steps}."
            )
            print(
                "Stage-2 safe schedule: "
                f"phase1 reconstruction-only=[0,{args.stage2_phase2_start_step}) "
                f"lr={stage_defaults['lr']:.3e}; "
                f"phase2 scale-1 waveform GAN=[{args.stage2_phase2_start_step},"
                f"{args.stage2_phase3_start_step}) lr="
                f"{args.stage2_phase2_generator_lr:.3e}; "
                f"phase3 full perceptual=[{args.stage2_phase3_start_step},"
                f"{num_train_steps}) "
                f"lr={args.stage2_phase3_generator_lr:.3e}, "
                f"adv={args.stage2_phase3_gan_adversarial_max:.3e}, "
                f"feature={args.stage2_phase3_gan_feature_max:g}; "
                f"teacher={args.stage2_teacher_retention_weight:g}, "
                "decoder_lr_multipliers="
                f"{tuple(args.stage2_decoder_lr_multipliers)}; "
                f"quality_hard_stop={int(args.stage2_quality_hard_stop)}."
            )
        if args.stage2_recon_transition_start_steps is None:
            print(
                "Stage-2 reconstruction weights: fixed for the full run "
                f"(wave={waveform_recon_loss_weight:g}, "
                f"mel={multi_spectral_recon_loss_weight:g}, "
                f"SI-SDR={si_sdr_loss_weight:g}, "
                f"corr=0.02, envelope={spectral_envelope_loss_weight:g}, "
                f"voiced-highband={voiced_highband_loss_weight:g}, "
                f"voiced-HF-retention={voiced_hf_retention_loss_weight:g}, "
                f"noise-floor={noise_floor_loss_weight:g}, "
                f"STFT={stft_recon_loss_weight:g}, frame-phase={frame_phase_loss_weight:g})."
            )
        else:
            print(
                "Stage-2 legacy reconstruction transition: start="
                f"{args.stage2_recon_transition_start_steps}, end="
                f"{args.stage2_recon_transition_end_steps}."
            )
        print(
            "Stage-2 candidate policy: quality baseline at initialization; "
            f"best checkpoints and quality hard-stop begin at step "
            f"{args.stage2_best_checkpoint_min_step}/"
            f"{args.stage2_quality_gate_start_steps}; "
            + (
                "reconstruction-only best candidates must preserve the "
                "AC320 and comb-median initialization gates."
                if stage25_reconstruction_only_refine
                else (
                    "best_gan_balanced.pt requires GAN ramp >= 0.5 and "
                    "best_full_gan_balanced.pt requires ramp=1.0."
                )
            )
        )
        print(
            "Stage-2 generator gradient diagnostics: every "
            f"{args.gan_grad_diagnostics_every} step(s) "
            "(0 disables; decoder GAN/reconstruction gradient ratio)."
        )
    print(f"Random seed: {args.seed}")
    print(
        "SI-SDR loss: "
        f"max_weight={si_sdr_loss_weight}, "
        f"start_steps={si_sdr_loss_start_steps if si_sdr_loss_weight > 0 else 0}, "
        f"warmup_steps={si_sdr_loss_warmup_steps if si_sdr_loss_weight > 0 else 0}"
    )
    print(
        "Voiced spectral envelope loss: "
        f"max_weight={spectral_envelope_loss_weight}, "
        f"start_steps={spectral_envelope_loss_start_steps if spectral_envelope_loss_weight > 0 else 0}, "
        f"warmup_steps={spectral_envelope_loss_warmup_steps if spectral_envelope_loss_weight > 0 else 0}, "
        "band=200-4500 Hz, cepstral_lifters=fine48/coarse32, "
        "scale_weights=0.55/0.45, shape_weights=value1/slope0.35/curvature0.15, "
        "region_weights=F1:1.0/F2:1.3/F3:1.15, "
        "voicing=target_periodicity_70-400Hz(threshold=0.35)+RMS"
    )
    print(
        "Formant peak loss: "
        f"max_weight={formant_peak_loss_weight}, "
        f"start_steps={formant_peak_loss_start_steps if formant_peak_loss_weight > 0 else 0}, "
        f"warmup_steps={formant_peak_loss_warmup_steps if formant_peak_loss_weight > 0 else 0}, "
        "regions=F1(200-1000)/F2(1000-2500)/F3(2500-4500)Hz, "
        "diagnostics_always_enabled=True, hard_gate=False"
    )
    print(
        "Voiced high-band detail loss: "
        f"max_weight={voiced_highband_loss_weight}, "
        f"start_steps={voiced_highband_loss_start_steps if voiced_highband_loss_weight > 0 else 0}, "
        f"warmup_steps={voiced_highband_loss_warmup_steps if voiced_highband_loss_weight > 0 else 0}, "
        "loss_band=2500-5500 Hz primary + 5500-7000 Hz auxiliary(0.35), "
        "diagnostic_band=3000-7000 Hz, target-voiced and frame-gain-normalized, "
        f"energy_deficit_weight={args.voiced_highband_energy_deficit_weight}, "
        f"allowed_deficit={args.voiced_highband_energy_margin_db} dB, "
        "excess_not_rewarded"
    )
    print(
        "Upper high-band detail loss: "
        f"max_weight={upper_highband_loss_weight}, "
        f"start_steps={upper_highband_loss_start_steps if upper_highband_loss_weight > 0 else 0}, "
        f"warmup_steps={upper_highband_loss_warmup_steps if upper_highband_loss_weight > 0 else 0}, "
        f"energy_deficit_weight={upper_highband_energy_deficit_weight}, "
        f"energy_margin_db={upper_highband_energy_margin_db}, "
        "band=7000-7800 Hz, taper=7600-7800 Hz, target-active only"
    )
    print(
        "Active spectral detail loss: "
        f"initial_weight={0.02 if stage25_rvq_midband_refine else 0.0}, "
        f"max_weight={active_spectral_detail_loss_weight}, "
        f"start_steps={active_spectral_detail_loss_start_steps if active_spectral_detail_loss_weight > 0 else 0}, "
        f"warmup_steps={active_spectral_detail_loss_warmup_steps if active_spectral_detail_loss_weight > 0 else 0}, "
        "windows=256/512/1024/2048, alphas=0.5/1/1/0.5, "
        "target_voiced=True, target_active_floor=-50 dB, band=200-7800 Hz, "
        "band_weights="
        + (
            "1/1.5/1/0.5/0.25"
            if stage25_rvq_midband_refine
            else "0.5/1/1/1.25/1.5"
        )
    )
    print(
        "Gate-aligned voiced-HF retention loss: "
        f"weight={voiced_hf_retention_loss_weight}, band=3000-7000 Hz, "
        f"margin={args.voiced_hf_retention_margin_db} dB, target_voiced_only=True, "
        "excess_not_rewarded=True"
    )
    print(
        "Transient noise loss: "
        f"click_weight={click_loss_weight}, "
        f"jump_weight={jump_loss_weight}, "
        f"warmup_steps={transient_loss_warmup_steps}"
    )
    print(
        "Background noise loss: "
        f"preemph_weight={preemph_loss_weight}, "
        f"multiband_noise_floor_weight={noise_floor_loss_weight}"
    )
    print(
        "Clean checkpoint gate: "
        f"enabled={not args.disable_clean_gate}, "
        f"aligned_si_sdr>={args.clean_gate_min_aligned_si_sdr}, "
        f"aligned_corr>={args.clean_gate_min_aligned_corr}, "
        f"negative_fraction<={args.clean_gate_max_negative_fraction}, "
        f"rms_ratio=[{args.clean_gate_min_rms_ratio}, {args.clean_gate_max_rms_ratio}], "
        f"peak<={args.clean_gate_max_recon_peak}, "
        f"clip<={args.clean_gate_max_recon_clip_fraction}, "
        f"click_excess<={args.clean_gate_max_click_excess} "
        f"(absolute click diagnostic threshold={args.clean_gate_max_click_score}), "
        f"jump<={args.clean_gate_max_jump_ratio}, "
        f"p999_jump<={args.clean_gate_max_p999_jump_ratio}; "
        "AC320, periodic/comb artifacts and high-frequency metrics are "
        "diagnostic-only"
    )
    if args.stage == "gan_pretrain":
        print(
            "Stage-2 quality retention gate: "
            f"enabled={args.stage2_quality_retention_gate}, "
            f"aligned_si_sdr_drop<={args.stage2_max_aligned_si_sdr_drop:.2f} dB, "
            f"aligned_corr_drop<={args.stage2_max_aligned_corr_drop:.3f}, "
            "balanced_aligned_si_sdr_drop<="
            f"{args.stage2_balanced_max_aligned_si_sdr_drop:.2f} dB, "
            "active_spectral_score_weight="
            f"{args.stage2_active_spectral_score_weight:g}, "
            "AC320/comb/high-frequency scores=diagnostic-only, "
            "quality selection rejects collapse only; normalized RVQ distribution "
            "metrics are tracked separately for deployability, "
            f"hard_stop=(quality={args.stage2_quality_retention_patience}, "
            f"rvq={args.stage2_rvq_retention_patience}) validation checks"
        )
        if stage25_rvq_midband_refine:
            print(
                "Stage-2.5 RVQ relative retention gate: "
                "q00_active_drop<=0.05, q00_perplexity_drop<=15% "
                "from the fixed initialization baseline."
            )
        print(
            "Stage-2 effective click checkpoint gate: "
            f"max(absolute={args.clean_gate_max_click_score:.4f}, "
            "initialization_baseline_click+"
            f"{args.stage2_max_click_score_rise:.4f}); "
            "validation logs report the resolved threshold and signed margin"
        )
    print(
        "Signed correlation loss weight: "
        f"{correlation_loss_weight} configured; "
        + (
            "Stage-1 runtime schedule="
            "0.20(<10k)/0.15(<20k)/0.10(<40k)/0.05(>=40k)"
            if args.stage == "recon_pretrain"
            else "fixed anchor for this stage"
        )
    )
    print(
        "Waveform reconstruction loss weight: "
        f"outer={waveform_recon_loss_weight}, L1=1, "
        f"MSE={wave_mse_loss_weight}, energy={energy_loss_weight}"
    )
    print(
        "Mel reconstruction loss weight: "
        f"{multi_spectral_recon_loss_weight}"
    )
    print(
        "STFT reconstruction loss weight: "
        f"max={stft_recon_loss_weight}, start_steps={stft_recon_loss_start_steps}, "
        f"warmup_steps={stft_recon_loss_warmup_steps}, "
        "scales=64/128/256/512/1024/2048, "
        "scale_weights=0.25/0.50/0.75/1.00/1.00/0.75, "
        "per_scale_logging=True"
    )
    print(
        "Per-loss decoder gradient diagnostics: "
        f"every={args.loss_grad_diagnostics_every} step(s) "
        "(wave/mel/MR-STFT/envelope/formant-peak/voiced-highband/"
        "SI-SDR/signed-correlation; 0 disables)"
    )
    if args.stage == "recon_pretrain":
        print(
            "Stage-1 RVQ protective stop: "
            f"patience={args.stage1_rvq_retention_patience} consecutive "
            "fixed-validation checks after the early-stopping floor"
        )
    print(
        "Frame-phase residual loss: "
        f"max={frame_phase_loss_weight}, warmup_steps={frame_phase_loss_warmup_steps}, "
        "frame_samples=320"
    )
    print(f"Decoder upsample mode: {args.decoder_upsample_mode}")
    print(f"Decoder linear upsample kernel min: {args.decoder_linear_upsample_kernel_min}")
    print(f"Decoder interpolation mode: {args.decoder_interpolation_mode}")
    print(
        "Decoder first x8 upsample: "
        f"{'split x4+x2' if args.decoder_split_first_upsample else 'single x8'}"
    )
    print(
        "Decoder residual scale schedule: "
        f"{decoder_residual_scale_start} until step "
        f"{decoder_residual_scale_warmup_start_steps}, "
        f"linear to {decoder_residual_scale_end} by step "
        f"{decoder_residual_scale_warmup_end_steps}"
    )
    if stage_defaults.get("decoder_x8_residual_scale_target") is not None:
        print(
            "Decoder x8 residual-scale refinement: "
            f"inherited -> {stage_defaults['decoder_x8_residual_scale_target']} over "
            f"{stage_defaults.get('decoder_x8_residual_scale_ramp_steps', 0)} steps"
        )
    if args.stage == "recon_pretrain":
        print(
            "Stage-1 best-checkpoint eligibility begins at step "
            f"{decoder_residual_scale_warmup_end_steps}, after decoder residual "
            f"scale reaches {decoder_residual_scale_end}."
        )
    print(f"Waveform discriminator learning rate: {stage_defaults['discr_lr']}")
    if args.stage == "gan_pretrain":
        print(f"STFT discriminator learning rate: {stft_discr_lr}")
    use_ema = stage_defaults.get("use_ema", True)
    print(f"EMA enabled: {use_ema}")
    if use_ema:
        print(f"EMA beta: {stage_defaults['ema_beta']}")
        print(
            "EMA schedule: "
            f"after_step={stage_defaults['ema_update_after_step']}, "
            f"every={stage_defaults['ema_update_every']}"
        )
    print(
        "GAN schedule: "
        f"start={stage_defaults['gan_start']}, "
        f"ramp={stage_defaults['gan_ramp']}, "
        f"adversarial_max={stage_defaults.get('gan_adversarial_max', 0.001)}, "
        f"feature_max={stage_defaults.get('gan_feature_max', 5.0)}"
    )
    if args.stage == "gan_pretrain":
        print(
            "Discriminator R1: "
            f"STFT real-only gamma={args.stft_r1_gamma:.3e} every "
            f"{args.stft_r1_every} steps (no interval multiplier); "
            f"waveform gamma={args.waveform_r1_gamma:.3e} every "
            f"{args.waveform_r1_every} steps; "
            f"per-branch grad_clip={args.discr_max_grad_norm:g}"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_global_batch = (
        batch_size * world_size * args.grad_accum_every
    )
    print(f"Per-GPU batch size: {batch_size}")
    print(f"Distributed world size: {world_size}")
    print(f"Effective global batch size: {effective_global_batch}")
    print(f"DataLoader workers per GPU: {args.dl_num_workers}")
    print(
        "Early stopping: "
        f"metric={'aligned_si_sdr' if args.stage == 'recon_pretrain' else 'score'}, "
        f"patience={early_stopping_patience} validation checks, "
        f"min_delta={early_stopping_min_delta}, "
        f"configured_min_steps={early_stopping_min_steps}, "
        f"effective_min_steps={effective_early_stopping_min_steps}"
    )
    print(f"Fixed validation batches: {args.best_eval_batches}")
    print(f"Dataset split: train {1 - args.valid_frac - args.test_frac:.2%}, valid {args.valid_frac:.2%}, test {args.test_frac:.2%}")
    print(f"Target sample rate: {sample_rate} Hz")
    print(f"Training segment: {segment_seconds:.3f} s")
    if args.stage in ("stream_finetune", "stream_finetune_long"):
        print(f"Internal streaming frame: {stream_frame_size} samples")
        print("Streaming context: per-layer causal state (no previous PCM frames)")
        print(
            "Boundary loss schedule: "
            f"weight={args.boundary_loss_weight}, "
            f"start={args.boundary_loss_start_steps}, "
            f"warmup={args.boundary_loss_warmup_steps}"
        )
        print(
            "Offline/stateful consistency schedule: "
            f"weight={args.stream_consistency_loss_weight}, "
            f"start={args.stream_consistency_loss_start_steps}, "
            f"warmup={args.stream_consistency_loss_warmup_steps}"
        )
        print(f"Boundary loss radius: {args.boundary_loss_radius} samples")
    print(f"Codebook sizes by RVQ level: {codebook_sizes}")
    print(f"RVQ quantizers: {num_quantizers}")
    print("Codec latent dimension: 64")
    print(f"RVQ lookup dimension: {rq_lookup_dim}")
    print(f"RVQ distance: {args.rq_distance}")
    print("RVQ gradient estimator: standard STE (rotation trick disabled)")
    if args.rvq_projection_only:
        print(
            "Q0 projection-only mode: PCA initialize Win/Wout from "
            f"{args.rvq_projection_pca_batches} distributed batches; "
            "Encoder/Decoder/RVQ frozen; discrete lookup bypassed."
        )
    codebook_storage = sum(codebook_sizes) * rq_lookup_dim
    projection_storage = 2 * 64 * rq_lookup_dim
    print(
        "RVQ INT8 storage: "
        f"{codebook_storage / 1024:.0f} KiB codebooks + "
        f"{projection_storage / 1024:.0f} KiB projections"
    )
    print("RVQ quantize dropout: False")
    print("RVQ dead-code threshold: 2")
    print(f"RVQ codebook synchronization: {world_size > 1}")
    if rvq_joint_adapt:
        print(
            "RVQ codebook training: EMA enabled only during the configured "
            f"{args.rvq_joint_decoder_only_steps}-"
            f"{args.rvq_joint_rvq_adapt_end_steps} B1.5 phase; "
            "frozen before and after it"
        )
    elif args.stage in ("stream_finetune", "stream_finetune_long"):
        print("RVQ codebook training: frozen")
    elif args.stage == "gan_pretrain":
        print(
            "RVQ codebook training: frozen throughout Stage 2 "
            "(EMA statistics and dead-code replacement disabled)"
        )
    elif stage_defaults.get("freeze_codebook_after_step") is not None:
        print(
            "RVQ codebook training: enabled until step "
            f"{stage_defaults['freeze_codebook_after_step']}, then frozen"
        )
    else:
        print("RVQ codebook training: enabled")
    print(f"Theoretical bitrate: {bitrate / 1000:.3f} kbps")
    if args.stage in ("stream_finetune", "stream_finetune_long"):
        print(
            f"Final test: continuous stateful full-file streaming, "
            f"{args.test_block_seconds:.1f} s metric blocks"
        )
        print(
            "Stateful optimizer policy: "
            + (
                f"Encoder Block{args.stage2_encoder_trainable_from_block + 1} "
                "and final latent convolution use LR=1e-7; Decoder uses "
                f"LR={stage_defaults['lr']:.3e}; RVQ is bypassed/frozen"
                if args.stage == "stream_finetune"
                else (
                    "Encoder and RVQ are frozen; Decoder-only LR="
                    f"{stage_defaults['lr']:.3e}; GAN is disabled"
                )
            )
        )
    else:
        print(
            f"Final test: full files in {args.test_block_seconds:.1f} s blocks, "
            f"{args.test_context_ms:.1f} ms previous context discarded from output"
        )

    # Default RVQ-v4 profile:
    # 16000 / 320 = 50 frames/s
    # log2(128) = 7 bits/token; 9 quantizers = 3.15 kbps
    soundstream = build_model(
        args.stage,
        sample_rate=sample_rate,
        strides=strides,
        stream_frame_size=stream_frame_size,
        stream_context_frames=stream_context_frames,
        boundary_loss_weight=args.boundary_loss_weight,
        boundary_loss_radius=args.boundary_loss_radius,
        boundary_loss_start_steps=args.boundary_loss_start_steps,
        boundary_loss_warmup_steps=args.boundary_loss_warmup_steps,
        stream_consistency_loss_weight=args.stream_consistency_loss_weight,
        stream_consistency_loss_start_steps=args.stream_consistency_loss_start_steps,
        stream_consistency_loss_warmup_steps=args.stream_consistency_loss_warmup_steps,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        rq_lookup_dim=rq_lookup_dim,
        rq_use_cosine_sim=(args.rq_distance == "cosine"),
        rq_projection_only=args.rvq_projection_only,
        si_sdr_loss_weight=si_sdr_loss_weight,
        click_loss_weight=click_loss_weight,
        jump_loss_weight=jump_loss_weight,
        spectral_envelope_loss_weight=spectral_envelope_loss_weight,
        formant_peak_loss_weight=formant_peak_loss_weight,
        voiced_highband_loss_weight=voiced_highband_loss_weight,
        upper_highband_loss_weight=upper_highband_loss_weight,
        active_spectral_detail_loss_weight=(
            active_spectral_detail_loss_weight
        ),
        active_spectral_detail_band_weights=(
            (1.00, 1.50, 1.00, 0.50, 0.25)
            if stage25_rvq_midband_refine
            else (0.50, 1.00, 1.00, 1.25, 1.50)
        ),
        upper_highband_energy_deficit_weight=upper_highband_energy_deficit_weight,
        upper_highband_energy_margin_db=upper_highband_energy_margin_db,
        voiced_highband_energy_deficit_weight=args.voiced_highband_energy_deficit_weight,
        voiced_highband_energy_margin_db=args.voiced_highband_energy_margin_db,
        voiced_hf_retention_loss_weight=voiced_hf_retention_loss_weight,
        voiced_hf_retention_margin_db=args.voiced_hf_retention_margin_db,
        preemph_loss_weight=preemph_loss_weight,
        noise_floor_loss_weight=noise_floor_loss_weight,
        wave_mse_loss_weight=wave_mse_loss_weight,
        energy_loss_weight=energy_loss_weight,
        generator_waveform_discr_loss_weights=waveform_discr_loss_weights,
        generator_stft_discr_loss_weight=stft_discr_loss_weight,
        stft_recon_loss_weight=stft_recon_loss_weight,
        frame_phase_loss_weight=frame_phase_loss_weight,
        gan_adversarial_max=stage_defaults.get("gan_adversarial_max", 0.001),
        gan_feature_max=stage_defaults.get("gan_feature_max", 5.0),
        decoder_upsample_mode=args.decoder_upsample_mode,
        decoder_residual_scale=decoder_residual_scale_start,
        decoder_linear_upsample_kernel_min=args.decoder_linear_upsample_kernel_min,
        decoder_interpolation_mode=args.decoder_interpolation_mode,
        decoder_split_first_upsample=args.decoder_split_first_upsample,
        # A fixed RVQ still supplies a useful commitment target when Stage 2.5
        # updates the Encoder. Decoder-only Stage 2 keeps this term disabled.
        commitment_loss_weight=(
            0.
            if stage25_decoder_only_refine
            else 0.1
            if args.stage2_targeted_refine
            else 0.
            if (
                args.stage in ("stream_finetune", "stream_finetune_long") or
                (
                    args.stage == "gan_pretrain" and
                    args.stage2_unfreeze_encoder_rvq_step < 0
                )
            )
            else 0.1
        ),
        sync_codebook=(world_size > 1),
        rq_codebook_balance_loss_weight=args.rvq_codebook_balance_loss_weight,
        rq_codebook_balance_target_perplexity=args.rvq_codebook_balance_target_perplexity,
        rq_codebook_balance_temperature=args.rvq_codebook_balance_temperature,
        rq_quantization_error_loss_weight=args.rvq_quantization_error_loss_weight,
        rq_continuous_teacher_loss_weight=args.rvq_continuous_teacher_loss_weight,
        bypass_rvq=args.bypass_rvq_during_training,
        recon_loss_weight_override=waveform_recon_loss_weight,
        multi_spectral_recon_loss_weight_override=(
            multi_spectral_recon_loss_weight
        ),
        correlation_loss_weight_override=correlation_loss_weight,
        hardware_encoder_qat=(args.stage == 'hardware_qat_finetune'),
        hardware_qat_observer_start_step=args.hardware_qat_observer_start_step,
        hardware_qat_start_step=args.hardware_qat_start_step,
        hardware_qat_activation_start_step=args.hardware_qat_activation_start_step,
        hardware_qat_warm_in_steps=args.hardware_qat_warm_in_steps,
        hardware_qat_block_interval_steps=args.hardware_qat_block_interval_steps,
        hardware_qat_observer_freeze_step=(
            args.hardware_qat_observer_freeze_step
        ),
        hardware_qat_ema_decay=args.hardware_qat_ema_decay,
        hardware_qat_observer=args.hardware_qat_observer,
        hardware_qat_percentile=args.hardware_qat_percentile,
        hardware_qat_validation_gated=args.hardware_qat_validation_gated,
        hardware_qat_gate_required_passes=args.hardware_qat_gate_required_passes,
        hardware_qat_group_fail_patience=args.hardware_qat_group_fail_patience,
    )
    if rvq_joint_adapt:
        if soundstream.recon_loss_weight != 7.5:
            raise RuntimeError(
                "B1.5 waveform loss weight did not reach SoundStream: "
                f"expected 7.5, got {soundstream.recon_loss_weight}"
            )
        if soundstream.multi_spectral_recon_loss_weight != 0.6:
            raise RuntimeError(
                "B1.5 Mel loss weight did not reach SoundStream: "
                "expected 0.6, got "
                f"{soundstream.multi_spectral_recon_loss_weight}"
            )

    warmup_steps = (
        1
        if args.stage == 'hardware_qat_finetune'
        else 100
        if stage25_rvq_midband_refine
        else min(
            1_000,
            max(1, num_train_steps // 10),
        )
    )
    scheduler = None
    scheduler_kwargs = {}
    discr_scheduler = None
    discr_scheduler_kwargs = {}
    if args.stage == "recon_pretrain" and not stage1_plateau_lr_enabled:
        scheduler = LambdaLR
        scheduler_kwargs = dict(lr_lambda=stage1_lr_lambda)

    if args.stream_tbptt_frames < 0:
        raise ValueError('--stream-tbptt-frames must be non-negative')
    if (
        args.preceding_context_seconds and
        not args.bypass_rvq_during_training and
        args.stage not in ("stream_finetune", "stream_finetune_long")
    ):
        raise ValueError(
            'Context training with an updating RVQ is not isolated. Use '
            '--bypass-rvq-during-training, or a stream stage where RVQ is frozen.'
        )
    soundstream.stream_tbptt_frames = args.stream_tbptt_frames
    print(f'Preceding real context: {args.preceding_context_seconds}s; target: {segment_seconds}s; TBPTT: {args.stream_tbptt_frames} frames')
    if args.stage in RECONSTRUCTION_STAGES:
        print(
            "Training crop: uniform random continuous crop; validation/test "
            "crop: deterministic by file index and split seed; A1/A1.5 use "
            "ordinary causal forward with real preceding context, not "
            "stateful-stream training."
        )
    trainer = SoundStreamTrainer(
        soundstream,
        folder=str(audio_dir),
        batch_size=batch_size,
        grad_accum_every=args.grad_accum_every,
        data_max_length_seconds=segment_seconds,
        preceding_context_seconds=args.preceding_context_seconds,
        dataset_max_files=(args.overfit_files if args.stage == "overfit" else None),
        dataset_fixed_crop=(args.stage == "overfit"),
        num_train_steps=num_train_steps,
        rvq_calibration_only=args.rvq_calibration_only,
        rvq_projection_only=args.rvq_projection_only,
        rvq_projection_pca_batches=args.rvq_projection_pca_batches,
        rvq_calibration_kmeans_batches=(
            args.rvq_calibration_kmeans_batches
            if args.rvq_calibration_only else 0
        ),
        rvq_projection_min_evr=args.rvq_projection_min_evr,
        rvq_projection_latent_mse_weight=(
            args.rvq_projection_latent_mse_weight
            if args.rvq_projection_only else 0.
        ),
        rvq_projection_latent_cosine_weight=(
            args.rvq_projection_latent_cosine_weight
            if args.rvq_projection_only else 0.
        ),
        rvq_projection_orth_loss_weight=args.rvq_projection_orth_loss_weight,
        rvq_projection_tie_loss_weight=args.rvq_projection_tie_loss_weight,
        rvq_projection_freeze_input_steps=(
            args.rvq_projection_freeze_input_steps
            if args.rvq_projection_only else 0
        ),
        rvq_joint_latent64_teacher_loss_weight=(
            args.rvq_joint_latent64_teacher_loss_weight
            if rvq_joint_adapt else 0.
        ),
        stage2_teacher_retention_weight=(
            args.stage2_teacher_retention_weight
            if args.stage == "gan_pretrain" and not rvq_joint_adapt
            else (
                args.state_teacher_retention_weight
                if args.stage == "stream_finetune_long" and
                args.state_teacher_retention_weight is not None
                else stage_defaults.get("teacher_retention_weight", 0.)
                if args.stage == "stream_finetune_long"
                else 0.
            )
        ),
        stage2_adaptive_gan=(
            args.stage2_adaptive_gan
            if args.stage == "gan_pretrain" and not rvq_joint_adapt
            else False
        ),
        stage2_max_gan_grad_ratio=args.stage2_max_gan_grad_ratio,
        decoder_lr_multipliers=(
            tuple(args.stage2_decoder_lr_multipliers)
            if args.stage == "gan_pretrain" and not rvq_joint_adapt
            else (1., 1., 1.)
        ),
        decoder_trainable_from_block=(
            args.state_decoder_trainable_from_block
            if args.stage == "stream_finetune_long" and
            args.state_decoder_trainable_from_block is not None
            else stage_defaults.get("decoder_trainable_from_block")
            if args.stage == "stream_finetune_long"
            else None
        ),
        # B1.5 keeps both 64<->lookup projections immutable. RVQ codebooks are
        # EMA-managed and excluded from Adam; only Decoder parameters optimize.
        projection_lr=None,
        rvq_joint_decoder_only_steps=args.rvq_joint_decoder_only_steps,
        rvq_joint_rvq_adapt_end_steps=args.rvq_joint_rvq_adapt_end_steps,
        rvq_plateau_freeze_start_steps=args.rvq_plateau_freeze_start_steps,
        rvq_plateau_freeze_patience=args.rvq_plateau_freeze_patience,
        rvq_plateau_freeze_min_delta=args.rvq_plateau_freeze_min_delta,
        lr=stage_defaults["lr"],
        encoder_lr=(
            None
            if rvq_joint_adapt
            else args.stage2_encoder_lr
            if (
                args.stage == "gan_pretrain" and
                not args.stage2_targeted_refine and
                args.stage2_unfreeze_encoder_rvq_step >= 0
            )
            else stage_defaults.get("encoder_lr")
            if args.stage != "gan_pretrain" or args.stage2_targeted_refine
            else None
        ),
        encoder_trainable_from_block=(
            args.stage2_encoder_trainable_from_block
            if (
                (
                    args.stage == "gan_pretrain" and
                    not args.stage2_targeted_refine and
                    args.stage2_unfreeze_encoder_rvq_step >= 0
                ) or
                args.stage == "stream_finetune"
            )
            else None
        ),
        # EMA codebook updates do not require optimizer parameters. Keep RVQ
        # parameters out of Adam even in the midband mode; freeze_codebook=False
        # below is what enables EMA/dead-code state updates.
        exclude_rq_from_generator_optimizer=(
            args.bypass_rvq_during_training or
            rvq_joint_adapt or
            (args.stage2_targeted_refine and not rvq_joint_adapt) or
            (args.stage == "gan_pretrain" and not rvq_joint_adapt) or
            args.stage in (
                "stream_finetune",
                "stream_finetune_long",
                "hardware_qat_finetune",
            )
        ),
        exclude_encoder_from_generator_optimizer=(
            args.rvq_projection_only or
            rvq_joint_adapt or
            stage25_decoder_only_refine or
            args.stage == "stream_finetune_long" or
            (
                args.stage == "gan_pretrain" and
                args.stage2_unfreeze_encoder_rvq_step < 0 and
                not args.stage2_targeted_refine
            )
        ),
        exclude_first_decoder_block_from_generator_optimizer=False,
        discr_lr=stage_defaults["discr_lr"],
        stft_discr_lr=stft_discr_lr,
        waveform_discr_lrs=waveform_discr_lrs,
        waveform_discr_update_every=waveform_discr_update_every,
        waveform_discr_loss_weights=waveform_discr_loss_weights,
        stft_discr_update_every=stft_discr_update_every,
        stft_discr_loss_weight=stft_discr_loss_weight,
        gan_grad_diagnostics_every=(
            args.gan_grad_diagnostics_every
            if args.stage == "gan_pretrain"
            else 0
        ),
        loss_grad_diagnostics_every=args.loss_grad_diagnostics_every,
        discr_max_grad_norm=args.discr_max_grad_norm,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        discr_scheduler=discr_scheduler,
        discr_scheduler_kwargs=discr_scheduler_kwargs,
        plateau_lr_enabled=(stage1_plateau_lr_enabled or stage2_plateau_lr_enabled),
        plateau_lr_start_steps=(
            args.stage2_plateau_start_steps if stage2_plateau_lr_enabled
            else args.plateau_start_steps
        ),
        plateau_lr_factor=(
            args.stage2_plateau_factor if stage2_plateau_lr_enabled
            else args.plateau_factor
        ),
        plateau_lr_patience=(
            args.stage2_plateau_patience if stage2_plateau_lr_enabled
            else args.plateau_patience
        ),
        plateau_lr_threshold=(
            args.stage2_plateau_threshold if stage2_plateau_lr_enabled
            else args.plateau_threshold
        ),
        plateau_lr_cooldown=(
            args.stage2_plateau_cooldown if stage2_plateau_lr_enabled
            else args.plateau_cooldown
        ),
        plateau_lr_min_lr=(
            args.stage2_plateau_min_lr if stage2_plateau_lr_enabled
            else args.plateau_min_lr
        ),
        plateau_lr_unclean_grace_checks=args.plateau_unclean_grace_checks,
        plateau_lr_metric=(
            'score' if stage2_plateau_lr_enabled else 'aligned_si_sdr'
        ),
        # Different D branches now have independent LRs and update rates.
        # A generator plateau must not silently destroy that balance.
        plateau_lr_update_discriminator=False,
        plateau_lr_discr_min_lr=(
            args.stage2_plateau_discr_min_lr if stage2_plateau_lr_enabled
            else None
        ),
        plateau_lr_stft_discr_min_lr=(
            args.stage2_plateau_stft_discr_min_lr
            if stage2_plateau_lr_enabled
            else None
        ),
        # Stage-2 plateau observes the HF/active-spectral composite score even when
        # the strict retention gate is currently failing. The gate still
        # controls checkpoint eligibility and the delayed hard stop.
        plateau_lr_require_quality_retention=False,
        generator_hold_steps=(
            args.stage2_generator_hold_steps
            if args.stage == "gan_pretrain"
            else 0
        ),
        generator_hold_lr=(
            args.stage2_generator_hold_lr
            if args.stage == "gan_pretrain" and args.stage2_generator_hold_steps > 0
            else None
        ),
        generator_freeze_steps=(
            args.stage2_generator_freeze_steps
            if args.stage == "gan_pretrain"
            else 0
        ),
        stage2_phase2_start_step=(
            args.stage2_phase2_start_step
            if args.stage == "gan_pretrain" and not args.stage2_targeted_refine
            else None
        ),
        stage2_phase3_start_step=(
            args.stage2_phase3_start_step
            if args.stage == "gan_pretrain" and not args.stage2_targeted_refine
            else None
        ),
        stage2_phase2_generator_lr=(
            args.stage2_phase2_generator_lr
            if args.stage == "gan_pretrain" and not args.stage2_targeted_refine
            else None
        ),
        stage2_phase3_generator_lr=(
            args.stage2_phase3_generator_lr
            if args.stage == "gan_pretrain" and not args.stage2_targeted_refine
            else None
        ),
        stage2_phase3_gan_adversarial_max=(
            args.stage2_phase3_gan_adversarial_max
            if args.stage == "gan_pretrain" and not args.stage2_targeted_refine
            else None
        ),
        stage2_phase3_gan_feature_max=(
            args.stage2_phase3_gan_feature_max
            if args.stage == "gan_pretrain" and not args.stage2_targeted_refine
            else None
        ),
        discriminator_hold_steps=(
            args.stage2_discriminator_hold_steps
            if args.stage == "gan_pretrain"
            else 0
        ),
        discriminator_hold_lr=(
            args.stage2_discriminator_hold_lr
            if args.stage == "gan_pretrain" and args.stage2_discriminator_hold_steps > 0
            else None
        ),
        discriminator_start_step=(
            args.stage2_discriminator_start_steps
            if args.stage == "gan_pretrain"
            else None
        ),
        stage2_recon_transition_start_steps=(
            args.stage2_recon_transition_start_steps
            if args.stage == "gan_pretrain"
            else None
        ),
        stage2_recon_transition_end_steps=(
            args.stage2_recon_transition_end_steps
            if args.stage == "gan_pretrain"
            else None
        ),
        stage2_initial_si_sdr_loss_weight=(
            0.07 if args.stage == "gan_pretrain" else 0.
        ),
        stage2_initial_correlation_loss_weight=(
            0.02 if args.stage == "gan_pretrain" else 0.
        ),
        stage2_initial_spectral_envelope_loss_weight=(
            0.05 if args.stage == "gan_pretrain" else 0.
        ),
        stage2_initial_voiced_highband_loss_weight=(
            voiced_highband_loss_weight if args.stage == "gan_pretrain" else 0.
        ),
        stage2_initial_noise_floor_loss_weight=(
            0.03 if args.stage == "gan_pretrain" else 0.
        ),
        quality_retention_start_step=(
            args.stage2_quality_gate_start_steps
            if args.stage == "gan_pretrain"
            else (
                stage_defaults.get("min_steps", 0)
                if args.stage in ("stream_finetune", "stream_finetune_long")
                else args.hardware_qat_final_polish_step
                if args.stage == 'hardware_qat_finetune'
                else 0
            )
        ),
        save_results_every=args.save_results_every,
        save_model_every=save_model_every,
        best_eval_every=best_eval_every,
        best_eval_batches=args.best_eval_batches,
        si_sdr_loss_start_steps=(
            si_sdr_loss_start_steps
            if si_sdr_loss_weight > 0
            else 0
        ),
        si_sdr_loss_warmup_steps=(
            si_sdr_loss_warmup_steps
            if si_sdr_loss_weight > 0
            else 0
        ),
        transient_loss_warmup_steps=transient_loss_warmup_steps,
        decoder_residual_scale_start=decoder_residual_scale_start,
        spectral_envelope_loss_start_steps=(
            spectral_envelope_loss_start_steps
            if spectral_envelope_loss_weight > 0
            else 0
        ),
        spectral_envelope_loss_warmup_steps=(
            spectral_envelope_loss_warmup_steps
            if spectral_envelope_loss_weight > 0
            else 0
        ),
        formant_peak_loss_start_steps=(
            formant_peak_loss_start_steps
            if formant_peak_loss_weight > 0
            else 0
        ),
        formant_peak_loss_warmup_steps=(
            formant_peak_loss_warmup_steps
            if formant_peak_loss_weight > 0
            else 0
        ),
        voiced_highband_loss_start_steps=(
            voiced_highband_loss_start_steps
            if voiced_highband_loss_weight > 0
            else 0
        ),
        voiced_highband_loss_warmup_steps=(
            voiced_highband_loss_warmup_steps
            if voiced_highband_loss_weight > 0
            else 0
        ),
        upper_highband_loss_start_steps=(
            upper_highband_loss_start_steps
            if upper_highband_loss_weight > 0
            else 0
        ),
        upper_highband_loss_warmup_steps=(
            upper_highband_loss_warmup_steps
            if upper_highband_loss_weight > 0
            else 0
        ),
        active_spectral_detail_loss_start_steps=(
            active_spectral_detail_loss_start_steps
            if active_spectral_detail_loss_weight > 0
            else 0
        ),
        active_spectral_detail_loss_initial_weight=(
            0.02 if stage25_rvq_midband_refine else 0.
        ),
        active_spectral_detail_loss_warmup_steps=(
            active_spectral_detail_loss_warmup_steps
            if active_spectral_detail_loss_weight > 0
            else 0
        ),
        stft_recon_loss_start_steps=(
            stft_recon_loss_start_steps if stft_recon_loss_weight > 0 else 0
        ),
        stft_recon_loss_warmup_steps=(
            stft_recon_loss_warmup_steps if stft_recon_loss_weight > 0 else 0
        ),
        frame_phase_loss_start_steps=stage_defaults.get("frame_phase_loss_start_steps", 0),
        frame_phase_loss_warmup_steps=(
            frame_phase_loss_warmup_steps if frame_phase_loss_weight > 0 else 0
        ),
        decoder_residual_scale_end=decoder_residual_scale_end,
        decoder_residual_scale_warmup_start_steps=decoder_residual_scale_warmup_start_steps,
        decoder_residual_scale_warmup_end_steps=decoder_residual_scale_warmup_end_steps,
        decoder_x8_residual_scale_target=stage_defaults.get("decoder_x8_residual_scale_target"),
        decoder_x8_residual_scale_ramp_steps=stage_defaults.get("decoder_x8_residual_scale_ramp_steps", 0),
        best_checkpoint_min_step=(
            0
            if args.rvq_projection_only
            else decoder_residual_scale_warmup_end_steps
            if args.stage == "recon_pretrain"
            else (
                args.stage2_best_checkpoint_min_step
                if args.stage == "gan_pretrain"
                else (
                    stage_defaults.get("min_steps", 0)
                    if args.stage in (
                        "spectral_refine",
                        "stream_finetune",
                        "stream_finetune_long",
                    )
                    else 0
                )
            )
        ),
        frame_leakage_checkpoint=(args.stage == "spectral_refine"),
        midband_checkpoint=stage25_rvq_midband_refine,
        clean_gate=not args.disable_clean_gate,
        clean_gate_min_aligned_si_sdr=args.clean_gate_min_aligned_si_sdr,
        clean_gate_min_aligned_corr=args.clean_gate_min_aligned_corr,
        clean_gate_max_negative_fraction=args.clean_gate_max_negative_fraction,
        clean_gate_min_rms_ratio=args.clean_gate_min_rms_ratio,
        clean_gate_max_rms_ratio=args.clean_gate_max_rms_ratio,
        clean_gate_max_recon_peak=args.clean_gate_max_recon_peak,
        clean_gate_max_recon_clip_fraction=args.clean_gate_max_recon_clip_fraction,
        clean_gate_max_click_score=args.clean_gate_max_click_score,
        clean_gate_max_click_excess=args.clean_gate_max_click_excess,
        clean_gate_max_jump_ratio=args.clean_gate_max_jump_ratio,
        clean_gate_max_p999_jump_ratio=args.clean_gate_max_p999_jump_ratio,
        clean_gate_min_voiced_hf_ratio_db=(
            args.clean_gate_min_voiced_hf_ratio_db
            if args.stage == "recon_pretrain"
            else None
        ),
        clean_gate_max_voiced_hf_ratio_db=(
            args.clean_gate_max_voiced_hf_ratio_db
            if args.stage == "recon_pretrain"
            else None
        ),
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_min_steps=early_stopping_min_steps,
        early_stopping_metric=(
            'aligned_si_sdr' if args.stage == 'recon_pretrain' else 'score'
        ),
        enable_gan=(
            args.stage in GAN_STAGES and
            not stage25_reconstruction_only_refine
        ),
        allow_discriminator_reinitialization=args.test_only,
        gan_start_step=stage_defaults["gan_start"],
        gan_ramp_steps=stage_defaults["gan_ramp"],
        gan_adversarial_max=stage_defaults.get("gan_adversarial_max", 0.001),
        gan_feature_max=stage_defaults.get("gan_feature_max", 5.),
        quality_retention_gate=(
            args.stage in QUALITY_RETENTION_STAGES and
            args.stage2_quality_retention_gate
        ),
        quality_retention_hard_stop=(
            False
            if args.stage == "spectral_refine"
            else (
                args.stage2_quality_hard_stop
                if args.stage == "gan_pretrain"
                else True
            )
        ),
        quality_retention_max_aligned_si_sdr_drop=(
            0.10
            if args.stage == "spectral_refine"
            else (
                0.20
                if args.stage in ("stream_finetune", "stream_finetune_long")
                else args.stage2_max_aligned_si_sdr_drop
            )
        ),
        quality_retention_max_aligned_corr_drop=(
            0.01
            if args.stage in ("stream_finetune", "stream_finetune_long")
            else args.stage2_max_aligned_corr_drop
        ),
        quality_retention_max_quiet_hf_excess_db_rise=(
            0.30
            if args.stage in ("stream_finetune", "stream_finetune_long")
            else args.stage2_max_quiet_hf_excess_db_rise
        ),
        quality_retention_max_voiced_hf_ratio_db_drop=(
            0.50
            if args.stage in ("stream_finetune", "stream_finetune_long")
            else args.stage2_max_voiced_hf_ratio_db_drop
        ),
        quality_retention_max_voiced_hf_ratio_db_rise=(
            0.50
            if args.stage in ("stream_finetune", "stream_finetune_long")
            else args.stage2_max_voiced_hf_ratio_db_rise
        ),
        quality_retention_hf_score_weight=args.stage2_voiced_hf_score_weight,
        quality_retention_max_click_score_rise=args.stage2_max_click_score_rise,
        quality_retention_max_ac320_isolated_rise=(
            0.003
            if args.stage in ("stream_finetune", "stream_finetune_long")
            else args.stage2_max_ac320_isolated_rise
        ),
        quality_retention_max_comb_median_excess_db_rise=(
            0.30
            if args.stage in ("stream_finetune", "stream_finetune_long")
            else args.stage2_max_comb_median_excess_db_rise
        ),
        balanced_checkpoint_max_aligned_si_sdr_drop=(
            args.stage2_balanced_max_aligned_si_sdr_drop
        ),
        balanced_checkpoint_min_upper_hf_ratio_db=(
            args.stage2_min_voiced_7k_7p8k_ratio_db
            if args.stage == "gan_pretrain"
            else None
        ),
        balanced_checkpoint_max_upper_hf_ratio_db=(
            args.stage2_max_voiced_7k_7p8k_ratio_db
            if args.stage == "gan_pretrain"
            else None
        ),
        balanced_checkpoint_max_quiet_upper_hf_excess_db_rise=(
            args.stage2_max_quiet_7k_7p8k_excess_db_rise
        ),
        quality_retention_upper_hf_score_weight=(
            args.stage2_upper_highband_score_weight
            if args.stage == "gan_pretrain"
            else 0.
        ),
        quality_retention_active_spectral_score_weight=(
            args.stage2_active_spectral_score_weight
            if args.stage == "gan_pretrain"
            else 0.
        ),
        quality_retention_max_q00_active_ratio_drop=(
            0.05 if stage25_rvq_midband_refine else None
        ),
        quality_retention_max_q00_perplexity_fraction_drop=(
            0.15 if stage25_rvq_midband_refine else None
        ),
        quality_retention_patience=args.stage2_quality_retention_patience,
        quality_retention_rvq_patience=args.stage2_rvq_retention_patience,
        stage1_rvq_retention_patience=args.stage1_rvq_retention_patience,
        freeze_codebook_after_step=(
            None if rvq_joint_adapt
            else stage_defaults.get("freeze_codebook_after_step")
        ),
        freeze_codebook_before_step=None,
        freeze_encoder_before_step=(
            num_train_steps + 1
            if (args.rvq_projection_only or rvq_joint_adapt)
            else
            num_train_steps + 1
            if (
                args.stage == "stream_finetune_long" or
                stage25_decoder_only_refine
            )
            else
            None
            if args.stage2_targeted_refine
            else
            (
                num_train_steps + 1
                if args.stage2_unfreeze_encoder_rvq_step < 0
                else args.stage2_unfreeze_encoder_rvq_step
            )
            if args.stage == "gan_pretrain"
            else args.hardware_qat_start_step
            if args.stage == 'hardware_qat_finetune'
            else None
        ),
        freeze_encoder_after_step=None,
        rvq_warm_in_steps=args.rvq_warm_in_steps,
        decoder_lr_after_step=(
            (
                args.rvq_joint_rvq_adapt_end_steps,
                args.rvq_joint_polish_decoder_lr,
            )
            if rvq_joint_adapt else None
        ),
        encoder_only_training=(args.stage == 'hardware_qat_finetune'),
        hardware_qat_latent64_weight=args.hardware_qat_latent64_weight,
        hardware_qat_latent32_weight=args.hardware_qat_latent32_weight,
        hardware_qat_rvq_margin_weight=args.hardware_qat_rvq_margin_weight,
        hardware_qat_rvq_margin=args.hardware_qat_rvq_margin,
        hardware_qat_rvq_margin_max=args.hardware_qat_rvq_margin_max,
        hardware_qat_sensitivity_scan=args.hardware_qat_sensitivity_scan,
        hardware_qat_fixed_scale_step=args.hardware_qat_observer_freeze_step,
        hardware_qat_fixed_scale_lr=args.hardware_qat_fixed_scale_lr,
        hardware_qat_final_polish_step=args.hardware_qat_final_polish_step,
        hardware_qat_final_polish_lr=args.hardware_qat_final_polish_lr,
        hardware_qat_max_latent32_nmse=args.hardware_qat_max_latent32_nmse,
        hardware_qat_max_quantized_output_nmse=(
            args.hardware_qat_max_quantized_output_nmse
        ),
        hardware_qat_group_fail_patience=args.hardware_qat_group_fail_patience,
        hardware_qat_max_q00_index_flip=args.hardware_qat_max_q00_index_flip,
        hardware_qat_max_q01_index_flip=args.hardware_qat_max_q01_index_flip,
        freeze_decoder_before_step=(
            num_train_steps
            if (args.rvq_projection_only or stage25_rvq_midband_refine)
            else None
        ),
        freeze_codebook_during_training=(
            args.rvq_projection_only or
            args.stage in (
                "stream_finetune",
                "stream_finetune_long",
                "hardware_qat_finetune",
            ) or
            (
                args.stage2_targeted_refine and
                not stage25_rvq_midband_refine and
                not rvq_joint_adapt
            ) or
            (
                args.stage == "gan_pretrain" and
                not args.stage2_targeted_refine
            )
        ),
        use_ema=use_ema,
        ema_beta=stage_defaults["ema_beta"],
        ema_update_after_step=stage_defaults["ema_update_after_step"],
        ema_update_every=stage_defaults["ema_update_every"],
        apply_grad_penalty_every=(
            args.waveform_r1_every if args.stage == "gan_pretrain" else 0
        ),
        waveform_grad_penalty_gamma=(
            args.waveform_r1_gamma if args.stage == "gan_pretrain" else 0.
        ),
        stft_grad_penalty_every=(
            args.stft_r1_every if args.stage == "gan_pretrain" else 0
        ),
        stft_grad_penalty_gamma=(
            args.stft_r1_gamma if args.stage == "gan_pretrain" else 0.
        ),
        results_folder=str(results_dir),
        valid_frac=(0. if args.stage == "overfit" else args.valid_frac),
        test_frac=(0. if args.stage == "overfit" else args.test_frac),
        split_by_speaker=(args.stage != "overfit"),
        random_split_seed=args.seed,
        dataloader_seed=args.seed,
        best_checkpoint_metric=args.stage,
        dl_num_workers=args.dl_num_workers,
        init_process_group_timeout_seconds=7_200,
        force_clear_prev_results=False,
    )

    if args.test_only and args.validation_only:
        raise ValueError("--test-only and --validation-only are mutually exclusive")

    evaluation_only = args.test_only or args.validation_only
    checkpoint = (
        latest_checkpoint(results_dir)
        if args.resume and not evaluation_only
        else None
    )

    if evaluation_only:
        evaluation_checkpoint = (
            args.test_checkpoint
            if args.test_only
            else args.validation_checkpoint
        )
        required_flag = (
            "--test-checkpoint"
            if args.test_only
            else "--validation-checkpoint"
        )
        if evaluation_checkpoint is None:
            raise ValueError(
                f"{'--test-only' if args.test_only else '--validation-only'} "
                f"requires {required_flag}"
            )
        evaluation_checkpoint = evaluation_checkpoint.expanduser().resolve()
        if not evaluation_checkpoint.is_file():
            raise FileNotFoundError(
                f"Evaluation checkpoint not found: {evaluation_checkpoint}"
            )
        if args.test_only:
            args.test_checkpoint = evaluation_checkpoint
            print(
                "Test-only mode; training and checkpoint writes are disabled: "
                f"{evaluation_checkpoint}"
            )
        else:
            args.validation_checkpoint = evaluation_checkpoint
            print(
                "Fixed-validation-only mode; training, held-out testing, and "
                f"checkpoint writes are disabled: {evaluation_checkpoint}"
            )
            load_model_weights_only(
                trainer.unwrapped_soundstream,
                evaluation_checkpoint,
            )
    elif checkpoint is not None:
        print(f"Resuming from checkpoint: {checkpoint}")
        trainer.load(
            str(checkpoint),
            reset_early_stopping=args.reset_early_stopping_on_resume,
        )
        if args.stage in (
            "gan_pretrain",
            "stream_finetune",
            "stream_finetune_long",
            "hardware_qat_finetune",
        ):
            resumed_model = trainer.unwrapped_soundstream
            resumed_block_scales = tuple(
                float(scale) for scale in
                resumed_model.get_decoder_block_residual_scales()
            )
            if not resumed_block_scales:
                raise RuntimeError(
                    "Resumed staged checkpoint does not expose decoder block "
                    "residual scales."
                )
            # Stage-2/3 checkpoints already encode the decoder behavior that
            # passed validation. Do not let the generic scalar warmup overwrite
            # those restored per-block values after the next optimizer step.
            resumed_model.set_decoder_block_residual_scales(
                resumed_block_scales
            )
            trainer.decoder_x8_residual_scale_start = resumed_block_scales[0]
            trainer.decoder_x8_residual_scale_target = resumed_block_scales[0]
            trainer.decoder_x8_residual_scale_ramp_steps = 0
            trainer.decoder_residual_scale_start = resumed_block_scales[0]
            trainer.decoder_residual_scale_end = resumed_block_scales[0]
            trainer.decoder_residual_scale_warmup_start_steps = 0
            trainer.decoder_residual_scale_warmup_end_steps = 0
            trainer.print(
                "Fixed resumed decoder block residual scales for staged "
                f"training: {resumed_block_scales}"
            )
    else:
        if args.reset_early_stopping_on_resume:
            raise FileNotFoundError(
                "--reset-early-stopping-on-resume requires an existing resume "
                f"checkpoint under: {results_dir}"
            )
        predecessor_stage = {
            "spectral_refine": "recon_pretrain",
            "gan_pretrain": "spectral_refine",
            "stream_finetune": "gan_pretrain",
            "stream_finetune_long": "stream_finetune",
            "hardware_qat_finetune": "gan_pretrain",
        }.get(args.stage)
        init_checkpoint = args.init_checkpoint

        if args.stage2_targeted_refine and init_checkpoint is None:
            raise FileNotFoundError(
                "--stage25-encoder-refine requires --init-checkpoint pointing "
                "to a trained Stage-2 checkpoint with discriminator weights."
            )

        if init_checkpoint is None and predecessor_stage is not None:
            predecessor_dir = (
                args.predecessor_results_dir
                if args.predecessor_results_dir is not None
                else STAGE_RESULTS_DIRS[predecessor_stage]
            )
            predecessor_candidates = (
                (
                    "best_by_clarity.pt",
                    "best_selected.pt",
                    "best_by_aligned_si_sdr.pt",
                )
                if predecessor_stage in ("recon_pretrain", "spectral_refine")
                else (
                    "best_full_gan_balanced.pt",
                    "best_gan_balanced.pt",
                    "best_selected.pt",
                )
            )
            init_checkpoint = next(
                (predecessor_dir / name for name in predecessor_candidates
                 if (predecessor_dir / name).exists()),
                None,
            )

        if predecessor_stage is not None and init_checkpoint is None:
            expected_predecessor_dir = (
                args.predecessor_results_dir
                if args.predecessor_results_dir is not None
                else STAGE_RESULTS_DIRS[predecessor_stage]
            )
            raise FileNotFoundError(
                f"{args.stage} requires --init-checkpoint or "
                f"a validation-selected checkpoint under {expected_predecessor_dir}"
            )

        if init_checkpoint is not None:
            print(f"Initializing {args.stage} from checkpoint: {init_checkpoint}")
            checkpoint_config = load_model_weights_only(
                trainer.unwrapped_soundstream,
                init_checkpoint,
                # Stage 2.5 is a continuation of an already adversarially
                # trained Stage-2 model, so its discriminator weights must be
                # inherited. Other staged transitions intentionally start new
                # discriminators and load generator weights only.
                generator_only=(
                    predecessor_stage is not None and
                    not args.stage2_targeted_refine
                ),
                reinitialize_rvq_from_bypass=(
                    args.reinitialize_rvq_from_bypass_checkpoint
                ),
                reinitialize_rvq_codebooks_from_projection=(
                    args.reinitialize_rvq_codebooks_from_projection_checkpoint
                ),
            )
            if args.rvq_projection_only or args.stage in (
                "spectral_refine",
                "gan_pretrain",
                "stream_finetune",
                "stream_finetune_long",
                "hardware_qat_finetune",
            ):
                inherited_block_scales = checkpoint_config.get(
                    "decoder_block_residual_scales"
                )
                if inherited_block_scales is None and "decoder_residual_scale" not in checkpoint_config:
                    raise ValueError(
                        "Initialization checkpoint does not record decoder residual scale state; "
                        "cannot safely continue a staged run without changing decoder behavior."
                    )
                if "decoder_residual_scale" in checkpoint_config:
                    inherited_residual_scale = float(
                        checkpoint_config["decoder_residual_scale"]
                    )
                else:
                    inherited_residual_scale = float(inherited_block_scales[0])
                inherited_block_scales = tuple(
                    float(scale) for scale in (
                        inherited_block_scales or
                        (inherited_residual_scale,) * len(strides)
                    )
                )
                if any(scale < 0. for scale in inherited_block_scales):
                    raise ValueError(
                        "Initialization checkpoint has invalid decoder block residual scales="
                        f"{inherited_block_scales}."
                    )
                trainer.unwrapped_soundstream.set_decoder_block_residual_scales(
                    inherited_block_scales
                )
                trainer.decoder_x8_residual_scale_start = inherited_block_scales[0]
                if args.stage in (
                    "gan_pretrain",
                    "stream_finetune",
                    "stream_finetune_long",
                    "hardware_qat_finetune",
                ):
                    # Keep every refined block scale fixed. Using the legacy
                    # scalar scheduler here would silently reset x5/x4/x2.
                    trainer.decoder_x8_residual_scale_target = inherited_block_scales[0]
                    trainer.decoder_x8_residual_scale_ramp_steps = 0
                trainer.decoder_residual_scale_start = inherited_residual_scale
                trainer.decoder_residual_scale_end = inherited_residual_scale
                trainer.decoder_residual_scale_warmup_start_steps = 0
                trainer.decoder_residual_scale_warmup_end_steps = 0
                print(
                    "Restored decoder block residual scales from initialization checkpoint: "
                    f"{inherited_block_scales}."
                )
            if trainer.use_ema:
                trainer.copy_online_to_ema()
                print("Synchronized EMA from initialized online weights.")
            if rvq_joint_adapt:
                trainer.capture_rvq_q0_teacher()
                print(
                    "Captured immutable Q0 Encoder/Projection/Decoder teacher "
                    "for RVQ joint adaptation."
                )
            if args.stage == "gan_pretrain" and not args.stage2_targeted_refine:
                trainer.capture_stage2_latent_reference()
                print(
                    "Captured the Stage-2 initialization Encoder as the latent "
                    "drift reference (diagnostic only; no latent penalty)."
                )
                if args.stage2_teacher_retention_weight > 0.:
                    trainer.capture_stage2_decoder_teacher()
                    print(
                        "Captured immutable B1.5 Decoder teacher for Stage-2 "
                        "waveform/log-STFT retention."
                    )
            if args.stage == 'hardware_qat_finetune':
                trainer.capture_stage2_latent_reference()
                print(
                    'Captured immutable FP32 B2 Encoder teacher for '
                    'latent64/latent32 QAT retention and RVQ index-flip diagnostics.'
                )
                print(
                    'Hardware QAT schedule: '
                    f'observer-only={args.hardware_qat_observer_start_step}..'
                    f'{args.hardware_qat_start_step - 1}; '
                    f'weight-only starts={args.hardware_qat_start_step}; '
                    f'blockwise activation starts={args.hardware_qat_activation_start_step}, '
                    f'per-block blend={args.hardware_qat_warm_in_steps}, '
                    f'block interval={args.hardware_qat_block_interval_steps}; '
                    f'observer_freeze={args.hardware_qat_observer_freeze_step}; '
                    f'observer={args.hardware_qat_observer}'
                    f'({args.hardware_qat_percentile:g}%); '
                    f'activation_groups=6; validation_gated='
                    f'{int(args.hardware_qat_validation_gated)} '
                    f'(passes={args.hardware_qat_gate_required_passes}); '
                    f'sensitivity_scan={int(args.hardware_qat_sensitivity_scan)}; '
                    f'fixed_scale_lr={args.hardware_qat_fixed_scale_lr:g}; '
                    f'final_polish={args.hardware_qat_final_polish_step} '
                    f'at lr={args.hardware_qat_final_polish_lr:g}; '
                    f'teacher_weights=64D:{args.hardware_qat_latent64_weight:g}/'
                    f'32D:{args.hardware_qat_latent32_weight:g}/'
                    f'RVQ-margin:{args.hardware_qat_rvq_margin_weight:g}; '
                    f'gates=NMSE32<={args.hardware_qat_max_latent32_nmse:g}, '
                    'VQout-NMSE<='
                    f'{args.hardware_qat_max_quantized_output_nmse:g}, '
                    'INT32-overflow=0; q00/q01 index flips are diagnostic; '
                    f'group_fail_patience={args.hardware_qat_group_fail_patience}.'
                )
            if (
                args.stage == "stream_finetune_long" and
                trainer.stage2_teacher_retention_weight > 0.
            ):
                trainer.capture_stage2_decoder_teacher()
                print(
                    "Captured immutable B2 Decoder teacher for final state "
                    "waveform/log-STFT retention."
                )

        print("Starting a new training run.")

    if (
        not evaluation_only and
        args.stage in QUALITY_RETENTION_STAGES and
        trainer.quality_retention_gate and
        not trainer.has_quality_retention_baseline
    ):
        # Evaluate once on rank 0, then distribute the numerical baseline so
        # resumed / distributed workers carry identical retention state.
        baseline_keys = (
            "score",
            "reconstruction_score",
            "aligned_si_sdr",
            "aligned_correlation",
            "alignment_negative_fraction",
            "quiet_hf_excess_db",
            "voiced_hf_energy_ratio_db",
            "voiced_7k_7p8k_ratio_db",
            "quiet_7k_7p8k_excess_db",
            "click_score",
            "ac_320_isolated",
            "comb_median_excess_db",
            "spectral_envelope_fine",
            "spectral_envelope_coarse",
            "formant_f1_mae_hz",
            "formant_f2_mae_hz",
            "formant_f3_mae_hz",
            "formant_f1_valid_fraction",
            "formant_f2_valid_fraction",
            "formant_f3_valid_fraction",
            "formant_f2_mae_trainmask_hz",
            "formant_f2_valid_trainmask",
            "formant_f2_mae_evalmask_hz",
            "formant_f2_valid_evalmask",
            "formant_f3_mae_trainmask_hz",
            "formant_f3_valid_trainmask",
            "formant_f3_mae_evalmask_hz",
            "formant_f3_valid_evalmask",
            "stft_scale_512",
            "stft_scale_1024",
            "stft_scale_2048",
            "codebook_q00_active_ratio",
            "codebook_q00_perplexity",
            "active_spec_200_1k_logmag_error",
            "active_spec_1k_3k_logmag_error",
            "active_spec_3k_5k_logmag_error",
        )
        baseline_values = torch.zeros(
            len(baseline_keys),
            device=trainer.accelerator.device,
            dtype=torch.float64,
        )
        if trainer.is_main:
            baseline_metrics = trainer.evaluate_fixed_validation_score(
                trainer.unwrapped_soundstream
            )
            baseline_values.copy_(torch.tensor(
                [baseline_metrics[key] for key in baseline_keys],
                device=baseline_values.device,
                dtype=baseline_values.dtype,
            ))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(baseline_values, src=0)
        baseline_metrics = dict(zip(
            baseline_keys,
            (float(value) for value in baseline_values.cpu().tolist()),
        ))
        trainer.set_quality_retention_baseline(baseline_metrics)
        if checkpoint is None:
            # The initialization point is a real candidate, not merely a gate
            # reference.  Seed every best-score tracker with it so later
            # checkpoints must genuinely beat the model we started from.
            trainer.best_valid_score = min(
                trainer.best_valid_score,
                baseline_metrics["score"],
            )
            trainer.early_stopping_best_score = min(
                trainer.early_stopping_best_score,
                baseline_metrics["score"],
            )
            trainer.best_aligned_si_sdr = max(
                trainer.best_aligned_si_sdr,
                baseline_metrics["aligned_si_sdr"],
            )
            trainer.best_frame_leakage_score = min(
                trainer.best_frame_leakage_score,
                baseline_metrics["ac_320_isolated"],
            )
            trainer.best_balanced_score = min(
                trainer.best_balanced_score,
                baseline_metrics["score"],
            )
            trainer.best_formant_score = min(
                trainer.best_formant_score,
                trainer.formant_refinement_score(baseline_metrics),
            )
            if stage25_rvq_midband_refine:
                trainer.best_midband_score = min(
                    trainer.best_midband_score,
                    trainer.midband_checkpoint_score(baseline_metrics),
                )
        if trainer.is_main:
            print(
                "Stage-2 quality retention baseline recorded from initialized "
                "checkpoint: "
                f"aligned_si_sdr={baseline_metrics['aligned_si_sdr']:.3f}, "
                f"aligned_corr={baseline_metrics['aligned_correlation']:.3f}, "
                f"q00_active={baseline_metrics['codebook_q00_active_ratio']:.3f}, "
                f"q00_perplexity={baseline_metrics['codebook_q00_perplexity']:.1f}, "
                f"quiet_hf_excess_db={baseline_metrics['quiet_hf_excess_db']:.3f}, "
                f"voiced_hf_ratio_db={baseline_metrics['voiced_hf_energy_ratio_db']:+.3f}, "
                "voiced_7k_7p8k_ratio_db="
                f"{baseline_metrics['voiced_7k_7p8k_ratio_db']:+.3f}, "
                "quiet_7k_7p8k_excess_db="
                f"{baseline_metrics['quiet_7k_7p8k_excess_db']:+.3f}, "
                f"click_score={baseline_metrics['click_score']:.3f}, "
                f"ac_320_isolated={baseline_metrics['ac_320_isolated']:.4f}, "
                f"comb_median_excess_db={baseline_metrics['comb_median_excess_db']:.3f}"
            )
            if checkpoint is None:
                # Keep the initialization checkpoint semantically separate
                # from every trained Stage-2 best. The numerical trackers are
                # seeded above, so a fine-tuned checkpoint still has to beat
                # the baseline before it can acquire a best_* filename.
                baseline_candidates = ((
                    results_dir / "baseline_init.pt",
                    baseline_metrics["score"],
                ),)
                for baseline_path, baseline_score in baseline_candidates:
                    if baseline_path.exists():
                        print(
                            "Keeping existing checkpoint instead of overwriting "
                            f"the initialization candidate: {baseline_path}"
                        )
                        continue
                    trainer.save_model_only(
                        baseline_path,
                        trainer.unwrapped_soundstream,
                        score=baseline_score,
                        step=-1,
                        weight_source="initialization",
                    )
                print(
                    "Saved Stage-2 initialization as baseline_init.pt only; "
                    "seeded best-score comparisons without claiming a trained "
                    "Stage-2 best checkpoint."
                )
        trainer.accelerator.wait_for_everyone()

    if args.validation_only:
        validation_metrics = None
        if trainer.is_main:
            validation_metrics = trainer.evaluate_fixed_validation_score(
                trainer.unwrapped_soundstream
            )
            validation_report_file = (
                args.validation_report_file
                or (results_dir / "fixed_validation_report.tsv")
            ).expanduser().resolve()
            validation_report_file.parent.mkdir(parents=True, exist_ok=True)
            with validation_report_file.open("w", encoding="utf-8") as report:
                report.write("metric\tvalue\n")
                for metric, value in sorted(validation_metrics.items()):
                    report.write(f"{metric}\t{value}\n")
            print(
                "Fixed validation report: "
                f"score={validation_metrics['score']:.6f}, "
                f"aligned_si_sdr={validation_metrics['aligned_si_sdr']:.6f}, "
                f"aligned_corr={validation_metrics['aligned_correlation']:.6f}, "
                f"voiced_hf_ratio_db={validation_metrics['voiced_hf_energy_ratio_db']:+.6f}, "
                f"quiet_hf_excess_db={validation_metrics['quiet_hf_excess_db']:+.6f}, "
                f"voiced_7k_7p8k_error={validation_metrics['voiced_7k_7p8k_logmag_error']:.6f}, "
                f"voiced_7k_7p8k_ratio_db={validation_metrics['voiced_7k_7p8k_ratio_db']:+.6f}, "
                f"quiet_7k_7p8k_excess_db={validation_metrics['quiet_7k_7p8k_excess_db']:+.6f}, "
                f"ac_320_isolated={validation_metrics['ac_320_isolated']:+.6f}, "
                f"click_score={validation_metrics['click_score']:.6f}, "
                f"q00_ok={int(validation_metrics['q00_validation_eligible'] >= 0.5)}, "
                f"q01_ok={int(validation_metrics['q01_validation_eligible'] >= 0.5)}, "
                f"rvq_ok={int(validation_metrics['rvq_validation_eligible'] >= 0.5)}"
            )
            if "quantization_gap_db" in validation_metrics:
                residual_ratios = ", ".join(
                    f"q{index:02d}={validation_metrics[key]:.4f}"
                    for index in range(num_quantizers)
                    for key in (f"rvq_residual_energy_ratio_q{index:02d}",)
                    if key in validation_metrics
                )
                print(
                    "RVQ fidelity diagnostics: "
                    f"projection_aligned_si_sdr="
                    f"{validation_metrics['projection_only_aligned_si_sdr']:.3f} dB, "
                    f"quantized_aligned_si_sdr="
                    f"{validation_metrics['quantized_aligned_si_sdr']:.3f} dB, "
                    f"quantization_gap={validation_metrics['quantization_gap_db']:+.3f} dB, "
                    f"lookup_nmse={validation_metrics.get('rvq_latent_nmse_lookup', float('nan')):.4f}, "
                    f"lookup_cos={validation_metrics.get('rvq_latent_cosine_lookup', float('nan')):.4f}, "
                    f"latent64_nmse={validation_metrics.get('rvq_latent_nmse_64d', float('nan')):.4f}, "
                    f"latent64_cos={validation_metrics.get('rvq_latent_cosine_64d', float('nan')):.4f}; "
                    f"residual_energy={residual_ratios}"
                )
            print(f"Fixed validation report saved to: {validation_report_file}")
        trainer.accelerator.wait_for_everyone()
        return

    if not args.test_only:
        trainer.train()
        trainer.accelerator.wait_for_everyone()
    is_main = trainer.is_main
    test_model = trainer.unwrapped_soundstream

    if is_main and not args.test_only:
        # trainer.steps points to the next step, while checkpoint filenames
        # represent the last completed step.
        next_step = int(trainer.steps.item())
        last_completed_step = next_step - 1
        final_checkpoint = None

        if last_completed_step >= 0:
            final_checkpoint = (
                results_dir
                / f"soundstream.{last_completed_step}.pt"
            )
            trainer.save(str(final_checkpoint))
            shutil.copyfile(
                final_checkpoint,
                results_dir / "latest.pt",
            )

        print("Training complete.")
        print(f"Parameters saved to: {final_checkpoint}")

    trainer.accelerator.wait_for_everyone()
    best_full_gan_balanced = results_dir / "best_full_gan_balanced.pt"
    best_gan_balanced = results_dir / "best_gan_balanced.pt"
    best_balanced = results_dir / "best_balanced.pt"
    best_by_midband = results_dir / "best_by_midband.pt"
    best_by_formant = results_dir / "best_by_formant.pt"
    baseline_init = results_dir / "baseline_init.pt"
    best_by_aligned_si_sdr = results_dir / "best_by_aligned_si_sdr.pt"
    best_selected = results_dir / "best_selected.pt"
    best_by_clarity = results_dir / "best_by_clarity.pt"
    best_raw_clarity = results_dir / "best_raw_online_by_clarity.pt"
    best_raw_online = results_dir / "best_raw_online_by_aligned_si_sdr.pt"
    best_full_qat = results_dir / "best_full_qat.pt"
    if args.test_only:
        best_checkpoint = args.test_checkpoint
    elif args.stage == "gan_pretrain":
        if stage25_reconstruction_only_refine:
            # Reconstruction-only Stage 2.5 intentionally has no eligible
            # GAN-ramp checkpoint. Test the best quality-gated reconstruction
            # candidate instead of incorrectly requiring a GAN filename.
            best_checkpoint = next((
                checkpoint for checkpoint in (
                    *(
                        (best_by_midband,)
                        if stage25_rvq_midband_refine
                        else ()
                    ),
                    best_selected,
                    best_by_aligned_si_sdr,
                    best_balanced,
                )
                if checkpoint.exists()
            ), None)
        else:
            best_checkpoint = next((
                checkpoint for checkpoint in (
                    best_full_gan_balanced,
                    best_gan_balanced,
                )
                if checkpoint.exists()
            ), None)
    elif args.stage == "recon_pretrain":
        # Production handoff and final testing prefer the clean-gated clarity
        # candidate. Raw checkpoints remain last-resort diagnostics only.
        best_checkpoint = next((
            checkpoint for checkpoint in (
                best_by_clarity,
                best_selected,
                best_by_aligned_si_sdr,
                best_raw_clarity,
                best_raw_online,
            )
            if checkpoint.exists()
        ), best_raw_online)
    elif args.stage == "spectral_refine":
        # A trained refinement is eligible only if its normalized formant
        # score beats the initialization while the retention gate passes.
        # Otherwise report/test the original A1 initialization explicitly.
        best_checkpoint = next((
            checkpoint for checkpoint in (
                best_by_formant,
                baseline_init,
            )
            if checkpoint.exists()
        ), None)
    elif args.stage == "hardware_qat_finetune":
        # Partial-QAT best files are diagnostics only.  Final test/export must
        # use a checkpoint saved with all blocks at alpha=1 and observers off.
        best_checkpoint = best_full_qat if best_full_qat.exists() else None
    elif args.stage in ("stream_finetune", "stream_finetune_long"):
        # Streaming adaptation is ranked by the gated reconstruction +
        # boundary + offline/stateful consistency score.
        best_checkpoint = next((
            checkpoint for checkpoint in (
                best_selected,
                best_by_aligned_si_sdr,
                best_raw_clarity,
                best_raw_online,
            )
            if checkpoint.exists()
        ), best_raw_online)
    else:
        best_checkpoint = next((
            checkpoint for checkpoint in (
                best_balanced,
                best_by_aligned_si_sdr,
                best_selected,
                best_raw_clarity,
                best_raw_online,
            )
            if checkpoint.exists()
        ), best_raw_online)

    if (
        is_main and
        not args.test_only and
        args.stage == "gan_pretrain" and
        best_checkpoint is None
    ):
        if stage25_reconstruction_only_refine:
            print(
                "Final held-out test skipped: no reconstruction-only candidate "
                "improved the initialization score while preserving the AC320/"
                "comb quality gates. baseline_init.pt remains an initialization "
                "reference, not a fine-tuned result."
            )
        else:
            print(
                "Final held-out test skipped: no eligible trained Stage-2 GAN "
                "checkpoint was produced. baseline_init.pt remains an initialization "
                "reference, not a Stage-2 result."
            )

    if (
        is_main and not args.test_only and
        args.stage == 'hardware_qat_finetune' and
        best_checkpoint is None
    ):
        print(
            'Hardware QAT run FAILED deployment eligibility: no '
            'best_full_qat.pt satisfied full INT8, fixed observer, NMSE32, '
            'quantized-output NMSE, and INT32 overflow gates. q00/q01 '
            'FP32 index flips are diagnostic only. Final held-out '
            'test/export is intentionally skipped.'
        )

    if best_checkpoint is not None and best_checkpoint.exists() and trainer.test_files:
        if is_main and best_checkpoint in (best_raw_clarity, best_raw_online):
            print(
                "WARNING: no clean-gated best checkpoint exists; final test is "
                f"using {best_checkpoint.name} for diagnostics only."
            )
        if is_main:
            print(f"Final test checkpoint: {best_checkpoint}")
        selected_test_files = list(trainer.test_files)
        if args.test_eval_batches is not None:
            selected_test_files = selected_test_files[:args.test_eval_batches]

        rank = trainer.accelerator.process_index
        world_size = trainer.accelerator.num_processes
        rank_test_files = selected_test_files[rank::world_size]
        save_test_reconstructions = (
            args.stage in ("recon_pretrain", "spectral_refine") and
            args.save_test_reconstructions and
            len(selected_test_files) > 0
        )
        test_recon_dir = (
            (args.test_recon_dir or (results_dir / "stage1_test_reconstructions")).resolve()
            if save_test_reconstructions
            else None
        )
        rank_metrics_path = (
            test_recon_dir / f"test_metrics_rank{rank:02d}.tsv"
            if save_test_reconstructions
            else None
        )
        default_test_report_name = (
            "stage1_test_report.txt"
            if args.stage in ("recon_pretrain", "spectral_refine")
            else "held_out_test_report.txt"
        )
        test_report_file = (
            args.test_report_file or (results_dir / default_test_report_name)
        ).resolve()

        if is_main:
            print(
                f"Evaluating {len(selected_test_files)} held-out test files with "
                f"validation-selected checkpoint on {world_size} process(es): "
                f"{best_checkpoint}"
            )
            if save_test_reconstructions:
                print(f"Saving stage-1 test reconstructions to: {test_recon_dir}")
            print(f"Writing held-out test report to: {test_report_file}")

        load_model_weights_only(test_model, best_checkpoint)
        if is_main and hasattr(test_model, "get_decoder_block_residual_scales"):
            print(
                "Final-test decoder block residual scales: "
                f"{test_model.get_decoder_block_residual_scales()}"
            )
        local_metrics = trainer.evaluate_full_audio_files(
            rank_test_files,
            model=test_model,
            block_seconds=args.test_block_seconds,
            context_ms=args.test_context_ms,
            save_recon_dir=test_recon_dir,
            metrics_path=rank_metrics_path,
        )

        metric_names = (
            'score',
            'reconstruction_score',
            'selection_score',
            'voiced_hf_score_penalty',
            'multi_spectral_recon_loss',
            'stft_recon_loss',
            'stft_scale_64',
            'stft_scale_128',
            'stft_scale_256',
            'stft_scale_512',
            'stft_scale_1024',
            'stft_scale_2048',
            'recon_loss',
            'wave_mse',
            'boundary_loss',
            'stream_consistency_loss',
            'stream_encoder_latent_l1',
            'stream_encoder_latent_cosine',
            'stream_encoder_latent_rms_ratio',
            'commitment_loss',
            'energy_loss',
            'rms_ratio',
            'correlation',
            'si_sdr',
            'aligned_correlation',
            'aligned_si_sdr',
            'target_peak',
            'recon_peak',
            'target_clip_fraction',
            'recon_clip_fraction',
            'target_max_jump',
            'recon_max_jump',
            'target_p999_jump',
            'recon_p999_jump',
            'jump_ratio',
            'p999_jump_ratio',
            'click_score',
            'spectral_envelope_loss',
            'spectral_envelope_fine',
            'spectral_envelope_coarse',
            'spectral_envelope_low',
            'spectral_envelope_mid',
            'spectral_envelope_high',
            'spectral_envelope_voiced_fraction',
            'formant_periodicity_mean',
            'formant_f1_mae_hz',
            'formant_f2_mae_hz',
            'formant_f3_mae_hz',
            'formant_f1_valid_fraction',
            'formant_f2_valid_fraction',
            'formant_f3_valid_fraction',
            'formant_f2_mae_trainmask_hz',
            'formant_f2_valid_trainmask',
            'formant_f2_mae_evalmask_hz',
            'formant_f2_valid_evalmask',
            'formant_f3_mae_trainmask_hz',
            'formant_f3_valid_trainmask',
            'formant_f3_mae_evalmask_hz',
            'formant_f3_valid_evalmask',
            'voiced_highband_loss',
            'voiced_hf_energy_ratio_db',
            'voiced_hf_logmag_error',
            'voiced_hf_energy_deficit',
            'voiced_hf_retention_loss',
            'upper_highband_loss',
            'voiced_7k_7p8k_logmag_error',
            'voiced_7k_7p8k_ratio_db',
            'quiet_7k_7p8k_excess_db',
            'active_spectral_detail_loss',
            'active_spec_200_1k_logmag_error',
            'active_spec_200_1k_energy_ratio_db',
            'active_spec_200_1k_spectral_convergence',
            'active_spec_1k_3k_logmag_error',
            'active_spec_1k_3k_energy_ratio_db',
            'active_spec_1k_3k_spectral_convergence',
            'active_spec_3k_5k_logmag_error',
            'active_spec_3k_5k_energy_ratio_db',
            'active_spec_3k_5k_spectral_convergence',
            'active_spec_5k_7k_logmag_error',
            'active_spec_5k_7k_energy_ratio_db',
            'active_spec_5k_7k_spectral_convergence',
            'active_spec_7k_7p8k_logmag_error',
            'active_spec_7k_7p8k_energy_ratio_db',
            'active_spec_7k_7p8k_spectral_convergence',
            'spectral_centroid_delta_hz',
            'spectral_slope_delta',
            'frame_diagnostic_valid',
            'ac_319',
            'ac_320',
            'ac_321',
            'ac_320_isolated',
            'frame_phase_peak_db',
            'comb_median_excess_db',
            'comb_p90_excess_db',
            'comb_lines_gt_6db',
        )
        local_num_samples = (
            float(local_metrics['num_samples'])
            if local_metrics is not None
            else 0.
        )
        packed_metrics = torch.tensor(
            [
                (
                    local_metrics[name] * local_num_samples
                    if local_metrics is not None
                    else 0.
                )
                for name in metric_names
            ] + [local_num_samples],
            dtype=torch.float64,
            device=trainer.device,
        )
        packed_metrics = trainer.accelerator.reduce(
            packed_metrics,
            reduction='sum',
        )
        local_code_counts = (
            local_metrics['code_counts']
            if local_metrics is not None
            else torch.zeros(
                num_quantizers,
                max(codebook_sizes),
                dtype=torch.float64
            )

        ).to(trainer.device)
        global_code_counts = trainer.accelerator.reduce(
            local_code_counts,
            reduction='sum',
        )

        if is_main and packed_metrics[-1].item() > 0:
            total_samples = packed_metrics[-1]
            test_metrics = {
                name: (packed_metrics[index] / total_samples).item()
                for index, name in enumerate(metric_names)
            }
            test_metrics.update(
                trainer.codebook_metrics_from_counts(
                    global_code_counts,
                    codebook_sizes,
                )
            )
            print(
                "Test report: "
                f"score={test_metrics['score']:.6f}, "
                f"reconstruction_score={test_metrics['reconstruction_score']:.6f}, "
                f"selection_score={test_metrics['selection_score']:.6f}, "
                f"hf_penalty={test_metrics['voiced_hf_score_penalty']:.6f}, "
                f"mel={test_metrics['multi_spectral_recon_loss']:.6f}, "
                f"stft={test_metrics['stft_recon_loss']:.6f}, "
                f"recon={test_metrics['recon_loss']:.6f}, "
                f"mse={test_metrics['wave_mse']:.6f}, "
                f"boundary={test_metrics['boundary_loss']:.6f}, "
                f"stream_consistency={test_metrics['stream_consistency_loss']:.6f}, "
                f"stream_encoder_l1={test_metrics['stream_encoder_latent_l1']:.6f}, "
                f"stream_encoder_cos={test_metrics['stream_encoder_latent_cosine']:.6f}, "
                f"stream_encoder_rms_ratio={test_metrics['stream_encoder_latent_rms_ratio']:.6f}, "
                f"commitment={test_metrics['commitment_loss']:.6f}, "
                f"energy={test_metrics['energy_loss']:.6f}, "
                f"rms_ratio={test_metrics['rms_ratio']:.6f}, "
                f"corr={test_metrics['correlation']:.6f}, "
                f"si_sdr={test_metrics['si_sdr']:.6f}, "
                f"aligned_corr={test_metrics['aligned_correlation']:.6f}, "
                f"aligned_si_sdr={test_metrics['aligned_si_sdr']:.6f}, "
                f"recon_peak={test_metrics.get('recon_peak', 0.):.6f}, "
                f"recon_clip={test_metrics.get('recon_clip_fraction', 0.) * 100:.6f}%, "
                f"jump_ratio={test_metrics.get('jump_ratio', 0.):.6f}, "
                f"click_score={test_metrics.get('click_score', 0.):.6f}, "
                f"voiced_hf_ratio_db={test_metrics.get('voiced_hf_energy_ratio_db', 0.):+.3f}, "
                f"voiced_hf_error={test_metrics.get('voiced_hf_logmag_error', 0.):.4f}, "
                f"voiced_hf_deficit={test_metrics.get('voiced_hf_energy_deficit', 0.):.4f}, "
                f"voiced_hf_retention={test_metrics.get('voiced_hf_retention_loss', 0.):.4f}, "
                f"voiced_7k_7p8k_error={test_metrics.get('voiced_7k_7p8k_logmag_error', 0.):.4f}, "
                f"voiced_7k_7p8k_ratio_db={test_metrics.get('voiced_7k_7p8k_ratio_db', 0.):+.3f}, "
                f"quiet_7k_7p8k_excess_db={test_metrics.get('quiet_7k_7p8k_excess_db', 0.):+.3f}, "
                f"centroid_delta_hz={test_metrics.get('spectral_centroid_delta_hz', 0.):+.1f}, "
                f"slope_delta={test_metrics.get('spectral_slope_delta', 0.):+.3f}, "
                f"ac_320_isolated={test_metrics.get('ac_320_isolated', 0.):+.6f}, "
                f"phase_peak_db={test_metrics.get('frame_phase_peak_db', 0.):+.3f}, "
                f"comb_median_db={test_metrics.get('comb_median_excess_db', 0.):+.3f}, "
                f"active_codes={test_metrics['active_code_ratio']:.6f}, "
                f"perplexity={test_metrics['codebook_perplexity']:.6f}"
            )
            if test_report_file is not None:
                test_report_file.parent.mkdir(parents=True, exist_ok=True)
                with test_report_file.open("w", encoding="utf-8") as f:
                    f.write(f"{args.stage} held-out test report\n")
                    f.write(f"checkpoint\t{best_checkpoint}\n")
                    f.write(f"results_dir\t{results_dir}\n")
                    f.write(f"num_test_files\t{len(selected_test_files)}\n")
                    f.write(f"num_processes\t{world_size}\n")
                    f.write(f"sample_rate\t{sample_rate}\n")
                    f.write(f"block_seconds\t{args.test_block_seconds}\n")
                    f.write(f"context_ms\t{args.test_context_ms}\n")
                    if test_recon_dir is not None:
                        f.write(f"reconstruction_dir\t{test_recon_dir}\n")
                        f.write("per_rank_metrics\t")
                        f.write(
                            ",".join(
                                str(test_recon_dir / f"test_metrics_rank{rank_index:02d}.tsv")
                                for rank_index in range(world_size)
                            )
                        )
                        f.write("\n")
                    f.write("\nmetric\tvalue\n")
                    for name in (
                        *metric_names,
                        'active_code_ratio',
                        'codebook_perplexity',
                        'codebook_collapsed_quantizers',
                    ):
                        if name in test_metrics:
                            f.write(f"{name}\t{test_metrics[name]}\n")
                print(f"Test report saved to: {test_report_file}")

    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.end_training()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
