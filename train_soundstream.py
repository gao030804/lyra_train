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
    "overfit": PROJECT_DIR / "results" / "overfit-64d-23q",
    "recon_pretrain": PROJECT_DIR / "results" / "recon-pretrain-64d-23q",
    "spectral_refine": PROJECT_DIR / "results" / "stage1-spectral-refine-64d-23q",
    "gan_pretrain": PROJECT_DIR / "results" / "gan-pretrain-64d-23q",
    "stream_finetune": PROJECT_DIR / "results" / "stream-finetune-64d-23q",
    "stream_finetune_long": PROJECT_DIR / "results" / "stream-finetune-long-64d-23q",
}

RECONSTRUCTION_STAGES = frozenset((
    "recon_pretrain",
    "spectral_refine",
    "gan_pretrain",
    "stream_finetune",
    "stream_finetune_long",
))
# Stage 3 is deliberately reconstruction-only.  Stage 4 is the optional weak
# streaming-GAN pass after stateful/offline consistency has been established.
GAN_STAGES = frozenset(("gan_pretrain", "stream_finetune_long"))
QUALITY_RETENTION_STAGES = frozenset((
    "gan_pretrain",
    "stream_finetune",
    "stream_finetune_long",
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
        transient_loss_warmup_steps=5_000,
        spectral_envelope_loss_weight=0.05,
        spectral_envelope_loss_start_steps=5_000,
        spectral_envelope_loss_warmup_steps=10_000,
        # The k4-online run retained a -2.2 dB voiced high-band deficit even
        # though SI-SDR and frame leakage improved.  Give the speech-clarity
        # band a little more influence without turning this into a broadband
        # high-frequency boost.
        voiced_highband_loss_weight=0.06,
        voiced_highband_loss_start_steps=5_000,
        voiced_highband_loss_warmup_steps=15_000,
        stft_recon_loss_weight=0.,
        si_sdr_loss_weight=0.07,
        si_sdr_loss_start_steps=5_000,
        si_sdr_loss_warmup_steps=15_000,
        gan_start=0, gan_ramp=0,
        gan_adversarial_max=0., gan_feature_max=0.,
    ),
    # Stage 1.5 starts from the validation-selected stage-1 codec. It keeps
    # RVQ trainable and only adds a small spectral refinement objective.
    "spectral_refine": dict(
        steps=20_000, batch_size=4, segment_seconds=4.,
        save_every=1_000, eval_every=250, min_steps=5_000, patience=30,
        lr=2e-5, discr_lr=None, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        use_ema=True,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.03,
        transient_loss_warmup_steps=0,
        spectral_envelope_loss_weight=0.05,
        spectral_envelope_loss_start_steps=0,
        spectral_envelope_loss_warmup_steps=0,
        voiced_highband_loss_weight=0.02,
        voiced_highband_loss_start_steps=0,
        voiced_highband_loss_warmup_steps=5_000,
        stft_recon_loss_weight=0.10,
        stft_recon_loss_start_steps=0,
        stft_recon_loss_warmup_steps=5_000,
        frame_phase_loss_weight=0.005,
        frame_phase_loss_start_steps=0,
        frame_phase_loss_warmup_steps=5_000,
        decoder_x8_residual_scale_target=0.85,
        decoder_x8_residual_scale_ramp_steps=3_000,
        si_sdr_loss_weight=0.03,
        si_sdr_loss_start_steps=0,
        si_sdr_loss_warmup_steps=0,
        gan_start=0, gan_ramp=0,
        gan_adversarial_max=0., gan_feature_max=0.,
    ),
    "gan_pretrain": dict(
        steps=50_000, batch_size=4, segment_seconds=4.,
        save_every=5_000, eval_every=500, min_steps=20_000, patience=12,
        early_stopping_min_delta=0.003,
        # Keep Stage-2 decoder updates smaller than discriminator updates so a
        # freshly initialized GAN cannot quickly displace the selected codec.
        lr=5e-7, discr_lr=5e-7, stft_discr_lr=2.5e-7,
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
        spectral_envelope_loss_weight=0.05,
        spectral_envelope_loss_start_steps=0,
        spectral_envelope_loss_warmup_steps=0,
        # Preserve the deterministic clarity learned in Stage 1.  Stage 2 GAN
        # gradients may add natural detail, but must not replace the paired
        # voiced high-band objective entirely.
        voiced_highband_loss_weight=0.07,
        voiced_hf_retention_loss_weight=0.02,
        voiced_highband_loss_start_steps=0,
        voiced_highband_loss_warmup_steps=0,
        stft_recon_loss_weight=0.,
        # Keep frame-leakage validation diagnostics, but disable the training
        # loss because the latest diagnostic run increased ac_320.
        frame_phase_loss_weight=0.,
        frame_phase_loss_start_steps=0,
        frame_phase_loss_warmup_steps=0,
        si_sdr_loss_weight=0.07,
        si_sdr_loss_start_steps=0,
        si_sdr_loss_warmup_steps=0,
        # Warm up fresh discriminators against a frozen Stage-1 generator for
        # 1k steps. Then introduce GAN gradients slowly over 20k steps.
        gan_start=1_000, gan_ramp=20_000,
        gan_adversarial_max=2e-4, gan_feature_max=1.25,
    ),
    "stream_finetune": dict(
        steps=20_000, batch_size=4, segment_seconds=4.,
        save_every=1_000, eval_every=250, min_steps=5_000, patience=30,
        lr=5e-7, discr_lr=5e-7, ema_beta=0.999,
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
        steps=20_000, batch_size=2, segment_seconds=4.,
        save_every=1_000, eval_every=250, min_steps=5_000, patience=20,
        lr=2.5e-7, discr_lr=5e-7, stft_discr_lr=2.5e-7, ema_beta=0.999,
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
        boundary_loss_start_steps=0,
        boundary_loss_warmup_steps=0,
        stream_consistency_loss_weight=0.10,
        stream_consistency_loss_start_steps=0,
        stream_consistency_loss_warmup_steps=0,
        gan_start=1_000, gan_ramp=10_000,
        gan_adversarial_max=5e-5, gan_feature_max=0.25,
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
        description="Train a staged 9.2 kbps streaming SoundStream speech codec."
    )

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
        default=1e-7,
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
            "the initialized stage-1 baseline and stop after sustained regression."
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
        "--stage2-unfreeze-encoder-rvq-step",
        type=int,
        default=-1,
        help=(
            "Joint Encoder/RVQ unfreeze step in Stage 2; -1 keeps both frozen "
            "for conservative decoder-only training and the 10k diagnostic."
        ),
    )
    parser.add_argument(
        "--stage2-targeted-refine",
        action="store_true",
        help=(
            "Use the conservative 10k decoder-only Stage-2 diagnostic preset: "
            "fresh discriminators, balanced per-D update rates, and no plateau scheduler."
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
        default="linear",
        help=(
            "Decoder upsampling block. 'linear' uses causal linear upsample + "
            "CausalConv1d and is stored in the checkpoint config."
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
        default=1_000,
        help=(
            "For gan_pretrain, keep all generator parameters unchanged while "
            "fresh discriminators warm up; defaults to 1000 steps."
        ),
    )
    parser.add_argument(
        "--stft-recon-loss-weight",
        type=float,
        default=None,
        help="Independent MR-STFT weight; defaults depend on --stage.",
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
        default=0,
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
    codebook_size: int,
    num_quantizers: int,
) -> float:
    downsample_factor = math.prod(strides)
    frame_rate = sample_rate / downsample_factor
    bits_per_token = math.log2(codebook_size)

    return frame_rate * num_quantizers * bits_per_token


def load_model_weights_only(
    model: torch.nn.Module,
    checkpoint: Path,
    *,
    generator_only: bool = False,
) -> dict:
    pkg = torch.load(str(checkpoint), map_location="cpu")
    checkpoint_config = {}
    if "config" in pkg:
        checkpoint_config = pickle.loads(pkg["config"])
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
    state_dict = pkg["model"] if "model" in pkg else pkg
    if generator_only:
        if not hasattr(model, "load_generator_state_dict"):
            raise TypeError("model does not support generator-only checkpoint loading")
        skipped_keys = model.load_generator_state_dict(state_dict)
        print(
            "Loaded generator/RVQ checkpoint state strictly; initialized "
            f"current discriminators from scratch ({len(skipped_keys)} state keys)."
        )
    else:
        model.load_state_dict(state_dict, strict=True)
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
    codebook_size: int,
    num_quantizers: int,
    si_sdr_loss_weight: float,
    spectral_envelope_loss_weight: float,
    voiced_highband_loss_weight: float,
    voiced_highband_energy_deficit_weight: float,
    voiced_highband_energy_margin_db: float,
    voiced_hf_retention_loss_weight: float,
    voiced_hf_retention_margin_db: float,
    click_loss_weight: float,
    jump_loss_weight: float,
    preemph_loss_weight: float,
    noise_floor_loss_weight: float,
    stft_recon_loss_weight: float,
    frame_phase_loss_weight: float,
    gan_adversarial_max: float,
    gan_feature_max: float,
    decoder_upsample_mode: str,
    decoder_residual_scale: float,
    decoder_linear_upsample_kernel_min: int,
    commitment_loss_weight: float | None = None,
    sync_codebook: bool | None = None,
) -> SoundStream:
    if sync_codebook is None:
        sync_codebook = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if stage in RECONSTRUCTION_STAGES:
        recon_loss_weight = 10.
        multi_spectral_recon_loss_weight = 1.1
        correlation_loss_weight = 0.02
    else:
        recon_loss_weight = 10. if stage == "overfit" else 1.
        multi_spectral_recon_loss_weight = 0.7
        correlation_loss_weight = 0.

    model_kwargs = dict(
        channels=16,
        channel_mults=(2, 4, 8, 16),
        codebook_dim=64,
        codebook_size=codebook_size,
        rq_num_quantizers=num_quantizers,
        rq_groups=1,
        use_lookup_free_quantizer=False,
        use_finite_scalar_quantizer=False,
        use_local_attn=False,
        target_sample_hz=sample_rate,
        strides=strides,
        recon_loss_weight=recon_loss_weight,
        multi_spectral_recon_loss_weight=multi_spectral_recon_loss_weight,
        stft_recon_loss_weight=stft_recon_loss_weight,
        spectral_envelope_loss_weight=spectral_envelope_loss_weight,
        voiced_highband_loss_weight=voiced_highband_loss_weight,
        voiced_highband_energy_deficit_weight=voiced_highband_energy_deficit_weight,
        voiced_highband_energy_margin_db=voiced_highband_energy_margin_db,
        voiced_hf_retention_loss_weight=voiced_hf_retention_loss_weight,
        voiced_hf_retention_margin_db=voiced_hf_retention_margin_db,
        si_sdr_loss_weight=si_sdr_loss_weight,
        correlation_loss_weight=correlation_loss_weight,
        energy_loss_weight=0.1,
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
        attn_window_size=64,
        attn_dim_head=32,
        attn_heads=4,
        attn_depth=1,
        decoder_upsample_mode=decoder_upsample_mode,
        decoder_residual_scale=decoder_residual_scale,
        decoder_linear_upsample_kernel_min=decoder_linear_upsample_kernel_min,
        pad_mode="constant",
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
    stage_defaults = dict(STAGE_DEFAULTS[args.stage])
    if args.stage2_targeted_refine:
        if args.stage != "gan_pretrain":
            raise ValueError("--stage2-targeted-refine is only valid with --stage gan_pretrain.")
        stage_defaults.update(
            steps=10_000,
            save_every=1_000,
            eval_every=500,
            min_steps=10_000,
            patience=20,
            early_stopping_min_delta=0.003,
            lr=2e-7,
            discr_lr=1e-6,
            stft_discr_lr=2.5e-7,
            waveform_discr_lrs=(5e-7, 5e-7, 2.5e-7),
            waveform_discr_update_every=(2, 4, 4),
            waveform_discr_loss_weights=(1.0, 0.25, 0.25),
            stft_discr_update_every=4,
            stft_discr_loss_weight=0.5,
            gan_start=1_000,
            gan_ramp=10_000,
            gan_adversarial_max=2e-4,
            gan_feature_max=5.0,
            noise_floor_loss_weight=0.03,
            spectral_envelope_loss_weight=0.05,
            voiced_highband_loss_weight=0.07,
            voiced_hf_retention_loss_weight=0.02,
            frame_phase_loss_weight=0.,
            frame_phase_loss_warmup_steps=0,
            si_sdr_loss_weight=0.07,
        )
        # The targeted preset is decoder-only by design.  Ignore an old launch
        # argument that would otherwise unfreeze Encoder/RVQ mid-run.
        args.stage2_unfreeze_encoder_rvq_step = -1
        args.stage2_generator_freeze_steps = 1_000
        args.stage2_discriminator_start_steps = 0
        args.stage2_max_aligned_si_sdr_drop = min(
            args.stage2_max_aligned_si_sdr_drop,
            0.05,
        )
        # Keep checkpoint eligibility strict from the beginning, but allow the
        # diagnostic to reach GAN ramp >= 0.5 before hard-stop patience can end
        # the whole run.  The trainer's candidate gate remains unchanged.
        args.stage2_quality_gate_start_steps = 6_000
        args.stage2_best_checkpoint_min_step = min(
            args.stage2_best_checkpoint_min_step,
            1_000,
        )

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
    if args.voiced_highband_loss_weight is not None and args.voiced_highband_loss_weight < 0:
        raise ValueError("--voiced-highband-loss-weight cannot be negative.")
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
    if args.stft_recon_loss_weight is not None and args.stft_recon_loss_weight < 0:
        raise ValueError("--stft-recon-loss-weight cannot be negative.")
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
        raise ValueError("--stage2-unfreeze-encoder-rvq-step must be -1 or non-negative.")
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
        if num_train_steps <= first_gan_candidate_step:
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
    # Lyra V2-style maximum bitrate:
    # 50 frames/s * 23 quantizers * 8 bits/index = 9.2 kbps.
    codebook_size = 256
    num_quantizers = 23
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
    voiced_highband_loss_weight = (
        args.voiced_highband_loss_weight
        if args.voiced_highband_loss_weight is not None
        else stage_defaults.get("voiced_highband_loss_weight", 0.)
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
    waveform_recon_loss_weight = 10.0 if args.stage in ("overfit", *RECONSTRUCTION_STAGES) else 1.0
    multi_spectral_recon_loss_weight = 1.1 if args.stage in RECONSTRUCTION_STAGES else 0.7
    correlation_loss_weight = (
        0.02 if args.stage in RECONSTRUCTION_STAGES else 0.0
    )
    decoder_residual_scale_start = args.decoder_residual_scale_start
    decoder_residual_scale_end = args.decoder_residual_scale_end
    decoder_residual_scale_warmup_start_steps = args.decoder_residual_scale_warmup_start_steps
    decoder_residual_scale_warmup_end_steps = args.decoder_residual_scale_warmup_end_steps
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
    if args.stage == "recon_pretrain":
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
                "(10*wave + 1.1*Mel + two-sided voiced-HF gate penalty; lower is better) "
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
        print(
            "Stage-2 retention phase: Generator frozen through step "
            f"{args.stage2_generator_freeze_steps}; discriminator updates begin at step "
            f"{args.stage2_discriminator_start_steps if args.stage2_discriminator_start_steps is not None else stage_defaults['gan_start']}. "
            "Encoder and RVQ are "
            + (
                "frozen throughout decoder-only training"
                if args.stage2_unfreeze_encoder_rvq_step < 0
                else f"jointly frozen through step {args.stage2_unfreeze_encoder_rvq_step}"
            )
            + "; generator adversarial/feature losses begin after step "
            f"{stage_defaults['gan_start']}; generator LR releases linearly from "
            f"{args.stage2_generator_hold_lr:.3e} to {stage_defaults['lr']:.3e} "
            f"by step {args.stage2_generator_hold_steps}."
        )
        if args.stage2_recon_transition_start_steps is None:
            print(
                "Stage-2 reconstruction weights: fixed for the full run "
                f"(wave=10.0, mel=1.1, SI-SDR={si_sdr_loss_weight:g}, "
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
            f"{args.stage2_quality_gate_start_steps}; best_gan_balanced.pt "
            "requires GAN ramp >= 0.5 and best_full_gan_balanced.pt requires ramp=1.0."
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
        "band=200-4500 Hz, smoothing_bins=9"
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
        f"rms_ratio=[{args.clean_gate_min_rms_ratio}, {args.clean_gate_max_rms_ratio}], "
        f"peak<={args.clean_gate_max_recon_peak}, "
        f"clip<={args.clean_gate_max_recon_clip_fraction}, "
        f"click_excess<={args.clean_gate_max_click_excess} "
        f"(absolute click diagnostic threshold={args.clean_gate_max_click_score}), "
        f"jump<={args.clean_gate_max_jump_ratio}, "
        f"p999_jump<={args.clean_gate_max_p999_jump_ratio}, "
        f"stage1_voiced_hf_ratio_db=[{args.clean_gate_min_voiced_hf_ratio_db}, "
        f"{args.clean_gate_max_voiced_hf_ratio_db}]"
    )
    if args.stage == "gan_pretrain":
        print(
            "Stage-2 quality retention gate: "
            f"enabled={args.stage2_quality_retention_gate}, "
            f"aligned_si_sdr_drop<={args.stage2_max_aligned_si_sdr_drop:.2f} dB, "
            f"aligned_corr_drop<={args.stage2_max_aligned_corr_drop:.3f}, "
            f"quiet_hf_excess_rise<={args.stage2_max_quiet_hf_excess_db_rise:.2f} dB, "
            f"voiced_hf_ratio_delta=[-{args.stage2_max_voiced_hf_ratio_db_drop:.2f}, "
            f"+{args.stage2_max_voiced_hf_ratio_db_rise:.2f}] dB, "
            f"voiced_hf_score_weight={args.stage2_voiced_hf_score_weight:g}, "
            f"ac320_rise<={args.stage2_max_ac320_isolated_rise:.4f}, "
            "comb_median_excess_rise<="
            f"{args.stage2_max_comb_median_excess_db_rise:.2f} dB, "
            "q00=(active>=0.70, perplexity>=50), q01/RVQ healthy, "
            f"hard_stop=(quality={args.stage2_quality_retention_patience}, "
            f"rvq={args.stage2_rvq_retention_patience}) validation checks"
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
        f"{correlation_loss_weight}"
    )
    print(
        "Waveform reconstruction loss weight: "
        f"{waveform_recon_loss_weight}"
    )
    print(
        "Mel reconstruction loss weight: "
        f"{multi_spectral_recon_loss_weight}"
    )
    print(
        "STFT reconstruction loss weight: "
        f"max={stft_recon_loss_weight}, warmup_steps={stft_recon_loss_warmup_steps}"
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
    print(f"Codebook size: {codebook_size}")
    print(f"RVQ quantizers: {num_quantizers}")
    print("RVQ quantize dropout: False")
    print("RVQ dead-code threshold: 2")
    print(f"RVQ codebook synchronization: {world_size > 1}")
    if args.stage in ("stream_finetune", "stream_finetune_long"):
        print("RVQ codebook training: frozen")
    elif args.stage == "gan_pretrain":
        if args.stage2_unfreeze_encoder_rvq_step < 0:
            print(
                "RVQ codebook training: frozen throughout Stage 2 "
                "(EMA statistics and dead-code replacement disabled)"
            )
        elif args.stage2_unfreeze_encoder_rvq_step == 0:
            print(
                "RVQ codebook training: enabled throughout Stage 2 "
                "(including EMA statistics and dead-code replacement)"
            )
        else:
            print(
                "RVQ codebook training: frozen for steps [0, "
                f"{args.stage2_unfreeze_encoder_rvq_step}), then enabled "
                "jointly with the Encoder (including EMA statistics and "
                "dead-code replacement)"
            )
    elif stage_defaults.get("freeze_codebook_after_step") is not None:
        print(
            "RVQ codebook training: enabled until step "
            f"{stage_defaults['freeze_codebook_after_step']}, then frozen"
        )
    else:
        print("RVQ codebook training: enabled")
    print(f"Theoretical bitrate: {bitrate / 1000:.1f} kbps")
    if args.stage in ("stream_finetune", "stream_finetune_long"):
        print(
            f"Final test: continuous stateful full-file streaming, "
            f"{args.test_block_seconds:.1f} s metric blocks"
        )
    else:
        print(
            f"Final test: full files in {args.test_block_seconds:.1f} s blocks, "
            f"{args.test_context_ms:.1f} ms previous context discarded from output"
        )

    # Lyra V2-style scalable bitrate configuration:
    # 16000 / 320 = 50 frames/s
    # log2(256) = 8 bits/token
    # 8 / 15 / 23 quantizers = 3.2 / 6.0 / 9.2 kbps
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
        si_sdr_loss_weight=si_sdr_loss_weight,
        click_loss_weight=click_loss_weight,
        jump_loss_weight=jump_loss_weight,
        spectral_envelope_loss_weight=spectral_envelope_loss_weight,
        voiced_highband_loss_weight=voiced_highband_loss_weight,
        voiced_highband_energy_deficit_weight=args.voiced_highband_energy_deficit_weight,
        voiced_highband_energy_margin_db=args.voiced_highband_energy_margin_db,
        voiced_hf_retention_loss_weight=voiced_hf_retention_loss_weight,
        voiced_hf_retention_margin_db=args.voiced_hf_retention_margin_db,
        preemph_loss_weight=preemph_loss_weight,
        noise_floor_loss_weight=noise_floor_loss_weight,
        stft_recon_loss_weight=stft_recon_loss_weight,
        frame_phase_loss_weight=frame_phase_loss_weight,
        gan_adversarial_max=stage_defaults.get("gan_adversarial_max", 0.001),
        gan_feature_max=stage_defaults.get("gan_feature_max", 5.0),
        decoder_upsample_mode=args.decoder_upsample_mode,
        decoder_residual_scale=decoder_residual_scale_start,
        decoder_linear_upsample_kernel_min=args.decoder_linear_upsample_kernel_min,
        # With frozen Encoder/RVQ the commitment term has no trainable target.
        # Restore it automatically if the user explicitly unfreezes that path.
        commitment_loss_weight=(
            0.
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
    )

    warmup_steps = min(
        1_000,
        max(1, num_train_steps // 10),
    )
    scheduler = None
    scheduler_kwargs = {}
    discr_scheduler = None
    discr_scheduler_kwargs = {}
    if args.stage == "recon_pretrain" and not stage1_plateau_lr_enabled:
        scheduler = LambdaLR
        scheduler_kwargs = dict(lr_lambda=stage1_lr_lambda)

    trainer = SoundStreamTrainer(
        soundstream,
        folder=str(audio_dir),
        batch_size=batch_size,
        grad_accum_every=args.grad_accum_every,
        data_max_length_seconds=segment_seconds,
        dataset_max_files=(args.overfit_files if args.stage == "overfit" else None),
        dataset_fixed_crop=(args.stage == "overfit"),
        num_train_steps=num_train_steps,
        lr=stage_defaults["lr"],
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
        # Stage-2 plateau observes the HF-penalized composite score even when
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
        stft_recon_loss_start_steps=stage_defaults.get("stft_recon_loss_start_steps", 0),
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
            decoder_residual_scale_warmup_end_steps
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
        clean_gate=not args.disable_clean_gate,
        clean_gate_min_aligned_si_sdr=args.clean_gate_min_aligned_si_sdr,
        clean_gate_min_aligned_corr=args.clean_gate_min_aligned_corr,
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
        enable_gan=args.stage in GAN_STAGES,
        allow_discriminator_reinitialization=args.test_only,
        gan_start_step=stage_defaults["gan_start"],
        gan_ramp_steps=stage_defaults["gan_ramp"],
        gan_adversarial_max=stage_defaults.get("gan_adversarial_max", 0.001),
        gan_feature_max=stage_defaults.get("gan_feature_max", 5.),
        quality_retention_gate=(
            args.stage in QUALITY_RETENTION_STAGES and
            args.stage2_quality_retention_gate
        ),
        quality_retention_max_aligned_si_sdr_drop=(
            0.20
            if args.stage in ("stream_finetune", "stream_finetune_long")
            else args.stage2_max_aligned_si_sdr_drop
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
        quality_retention_patience=args.stage2_quality_retention_patience,
        quality_retention_rvq_patience=args.stage2_rvq_retention_patience,
        stage1_rvq_retention_patience=args.stage1_rvq_retention_patience,
        freeze_codebook_after_step=stage_defaults.get("freeze_codebook_after_step"),
        freeze_codebook_before_step=(
            args.stage2_unfreeze_encoder_rvq_step
            if (
                args.stage == "gan_pretrain" and
                args.stage2_unfreeze_encoder_rvq_step >= 0
            )
            else None
        ),
        freeze_encoder_before_step=(
            num_train_steps + 1
            if args.stage in ("stream_finetune", "stream_finetune_long")
            else
            (
                num_train_steps + 1
                if args.stage2_unfreeze_encoder_rvq_step < 0
                else args.stage2_unfreeze_encoder_rvq_step
            )
            if args.stage == "gan_pretrain"
            else None
        ),
        freeze_codebook_during_training=(
            args.stage in ("stream_finetune", "stream_finetune_long") or
            (
                args.stage == "gan_pretrain" and
                args.stage2_unfreeze_encoder_rvq_step < 0
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
    else:
        if args.reset_early_stopping_on_resume:
            raise FileNotFoundError(
                "--reset-early-stopping-on-resume requires an existing resume "
                f"checkpoint under: {results_dir}"
            )
        predecessor_stage = {
            "spectral_refine": "recon_pretrain",
            "gan_pretrain": "recon_pretrain",
            "stream_finetune": "gan_pretrain",
            "stream_finetune_long": "stream_finetune",
        }.get(args.stage)
        init_checkpoint = args.init_checkpoint

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
                generator_only=(predecessor_stage is not None),
            )
            if args.stage in (
                "spectral_refine",
                "gan_pretrain",
                "stream_finetune",
                "stream_finetune_long",
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
            "aligned_si_sdr",
            "aligned_correlation",
            "quiet_hf_excess_db",
            "voiced_hf_energy_ratio_db",
            "click_score",
            "ac_320_isolated",
            "comb_median_excess_db",
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
        if trainer.is_main:
            print(
                "Stage-2 quality retention baseline recorded from initialized "
                "checkpoint: "
                f"aligned_si_sdr={baseline_metrics['aligned_si_sdr']:.3f}, "
                f"aligned_corr={baseline_metrics['aligned_correlation']:.3f}, "
                f"quiet_hf_excess_db={baseline_metrics['quiet_hf_excess_db']:.3f}, "
                f"voiced_hf_ratio_db={baseline_metrics['voiced_hf_energy_ratio_db']:+.3f}, "
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
                f"ac_320_isolated={validation_metrics['ac_320_isolated']:+.6f}, "
                f"click_score={validation_metrics['click_score']:.6f}, "
                f"q00_ok={int(validation_metrics['q00_validation_eligible'] >= 0.5)}, "
                f"q01_ok={int(validation_metrics['q01_validation_eligible'] >= 0.5)}, "
                f"rvq_ok={int(validation_metrics['rvq_validation_eligible'] >= 0.5)}"
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
    best_by_aligned_si_sdr = results_dir / "best_by_aligned_si_sdr.pt"
    best_selected = results_dir / "best_selected.pt"
    best_by_clarity = results_dir / "best_by_clarity.pt"
    best_raw_clarity = results_dir / "best_raw_online_by_clarity.pt"
    best_raw_online = results_dir / "best_raw_online_by_aligned_si_sdr.pt"
    if args.test_only:
        best_checkpoint = args.test_checkpoint
    elif args.stage == "gan_pretrain":
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
        print(
            "Final held-out test skipped: no eligible trained Stage-2 GAN "
            "checkpoint was produced. baseline_init.pt remains an initialization "
            "reference, not a Stage-2 result."
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
            'recon_loss',
            'wave_mse',
            'boundary_loss',
            'stream_consistency_loss',
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
            'voiced_highband_loss',
            'voiced_hf_energy_ratio_db',
            'voiced_hf_logmag_error',
            'voiced_hf_energy_deficit',
            'voiced_hf_retention_loss',
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
                codebook_size,
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
                trainer.codebook_metrics_from_counts(global_code_counts)
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
