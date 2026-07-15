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

RECONSTRUCTION_STAGES = frozenset(("recon_pretrain", "spectral_refine", "gan_pretrain"))
GAN_STAGES = frozenset(("gan_pretrain", "stream_finetune", "stream_finetune_long"))

STAGE_DEFAULTS = {
    "overfit": dict(
        steps=5_000, batch_size=4, segment_seconds=2.,
        save_every=500, eval_every=100, min_steps=0, patience=30,
        lr=3e-4, discr_lr=None, ema_beta=0.95,
        ema_update_after_step=0, ema_update_every=1,
        click_loss_weight=0., jump_loss_weight=0.,
        transient_loss_warmup_steps=0,
        spectral_envelope_loss_weight=0.,
        stft_recon_loss_weight=0.,
        gan_start=0, gan_ramp=0,
    ),
    "recon_pretrain": dict(
        steps=150_000, batch_size=4, segment_seconds=4.,
        save_every=2_000, eval_every=250, min_steps=10_000, patience=40,
        lr=2e-4, discr_lr=None, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        use_ema=True,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.03,
        transient_loss_warmup_steps=15_000,
        spectral_envelope_loss_weight=0.05,
        spectral_envelope_loss_start_steps=5_000,
        spectral_envelope_loss_warmup_steps=10_000,
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
        steps=200_000, batch_size=4, segment_seconds=4.,
        save_every=5_000, eval_every=500, min_steps=30_000, patience=30,
        lr=1e-5, discr_lr=2e-5, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        use_ema=True,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.02,
        transient_loss_warmup_steps=0,
        spectral_envelope_loss_weight=0.,
        spectral_envelope_loss_start_steps=0,
        spectral_envelope_loss_warmup_steps=0,
        stft_recon_loss_weight=0.,
        si_sdr_loss_weight=0.03,
        si_sdr_loss_start_steps=0,
        si_sdr_loss_warmup_steps=0,
        # Stage 2 starts from a validation-selected Stage-1 codec.  Train G
        # and D from step zero, while their low-LR retention window and the
        # slow GAN ramp protect reconstruction quality.
        gan_start=0, gan_ramp=20_000,
        gan_adversarial_max=0.0002, gan_feature_max=1.0,
    ),
    "stream_finetune": dict(
        steps=20_000, batch_size=4, segment_seconds=2.,
        save_every=1_000, eval_every=250, min_steps=5_000, patience=30,
        lr=3e-5, discr_lr=5e-5, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.,
        transient_loss_warmup_steps=0,
        spectral_envelope_loss_weight=0.,
        stft_recon_loss_weight=0.,
        gan_start=5_000, gan_ramp=10_000,
    ),
    "stream_finetune_long": dict(
        steps=5_000, batch_size=2, segment_seconds=4.,
        save_every=1_000, eval_every=250, min_steps=1_000, patience=15,
        lr=1e-5, discr_lr=2e-5, ema_beta=0.999,
        ema_update_after_step=0, ema_update_every=1,
        click_loss_weight=0., jump_loss_weight=0.,
        preemph_loss_weight=0., noise_floor_loss_weight=0.,
        transient_loss_warmup_steps=0,
        spectral_envelope_loss_weight=0.,
        stft_recon_loss_weight=0.,
        gan_start=1_000, gan_ramp=2_000,
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
        default=0.0,
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
        default=2e-5,
        help="Lower bound for stage-1 ReduceLROnPlateau generator LR.",
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
            "For gan_pretrain, lower generator and discriminator LR together "
            "only after post-ramp validation plateaus while quality retention holds."
        ),
    )
    parser.add_argument(
        "--stage2-plateau-start-steps",
        type=int,
        default=60_000,
        help="Do not allow Stage-2 plateau LR reductions before this completed step.",
    )
    parser.add_argument(
        "--stage2-plateau-factor",
        type=float,
        default=0.5,
        help="Stage-2 G/D LR multiplier for each validation-plateau reduction.",
    )
    parser.add_argument(
        "--stage2-plateau-patience",
        type=int,
        default=16,
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
        default=2e-6,
        help="Minimum Stage-2 generator LR.",
    )
    parser.add_argument(
        "--stage2-plateau-discr-min-lr",
        type=float,
        default=4e-6,
        help="Minimum Stage-2 discriminator LR.",
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
        default=0.30,
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
        default=3,
        help="Consecutive validation regressions before the Stage-2 quality hard stop.",
    )
    parser.add_argument(
        "--stage2-rvq-retention-patience",
        type=int,
        default=2,
        help="Consecutive q00/q01 regressions before the Stage-2 RVQ hard stop.",
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
        default=7,
        help=(
            "Minimum kernel size for decoder linear-upsample CausalConv1d. "
            "With the default 7, large-stride layers keep 2*stride while the "
            "final small-stride layer is lightly smoothed."
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
        default=20_000,
        help=(
            "For gan_pretrain, keep the generator at a conservative LR during "
            "the initial GAN ramp so the Stage-1 reconstruction baseline is retained."
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
        default=5e-6,
        help="Generator LR during the initial Stage-2 GAN ramp.",
    )
    parser.add_argument(
        "--stage2-discriminator-hold-steps",
        type=int,
        default=20_000,
        help=(
            "For gan_pretrain, keep every discriminator at a conservative LR "
            "during the initial GAN ramp while still updating from step zero."
        ),
    )
    parser.add_argument(
        "--stage2-discriminator-hold-lr",
        type=float,
        default=1e-5,
        help="Discriminator LR during the initial Stage-2 GAN ramp.",
    )
    parser.add_argument(
        "--stage2-recon-transition-start-steps",
        type=int,
        default=20_000,
        help=(
            "For gan_pretrain, begin linearly transitioning Stage-1 "
            "reconstruction weights to their Stage-2 endpoint at this step."
        ),
    )
    parser.add_argument(
        "--stage2-recon-transition-end-steps",
        type=int,
        default=40_000,
        help="For gan_pretrain, finish the reconstruction-weight transition at this step.",
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
        default=20_000,
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
        default=0.1,
    )
    parser.add_argument(
        "--boundary-loss-radius",
        type=int,
        default=8,
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
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    codebook_size: int,
    num_quantizers: int,
    si_sdr_loss_weight: float,
    spectral_envelope_loss_weight: float,
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
    sync_codebook: bool | None = None,
) -> SoundStream:
    if sync_codebook is None:
        sync_codebook = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if stage in RECONSTRUCTION_STAGES:
        recon_loss_weight = 10.
        multi_spectral_recon_loss_weight = 1.1
        correlation_loss_weight = (
            0.02 if stage in ("recon_pretrain", "spectral_refine") else 0.
        )
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
        si_sdr_loss_weight=si_sdr_loss_weight,
        correlation_loss_weight=correlation_loss_weight,
        energy_loss_weight=0.1,
        click_loss_weight=click_loss_weight,
        jump_loss_weight=jump_loss_weight,
        preemph_loss_weight=preemph_loss_weight,
        noise_floor_loss_weight=noise_floor_loss_weight,
        frame_phase_loss_weight=frame_phase_loss_weight,
        frame_phase_samples=320,
        commitment_loss_weight=0.1,
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
        **model_kwargs,
    )


def main() -> None:
    args = parse_args()
    stage_defaults = STAGE_DEFAULTS[args.stage]

    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.si_sdr_loss_weight is not None and args.si_sdr_loss_weight < 0:
        raise ValueError("--si-sdr-loss-weight cannot be negative.")
    if args.si_sdr_loss_start_steps is not None and args.si_sdr_loss_start_steps < 0:
        raise ValueError("--si-sdr-loss-start-steps cannot be negative.")
    if args.si_sdr_loss_warmup_steps is not None and args.si_sdr_loss_warmup_steps < 0:
        raise ValueError("--si-sdr-loss-warmup-steps cannot be negative.")
    if args.spectral_envelope_loss_weight is not None and args.spectral_envelope_loss_weight < 0:
        raise ValueError("--spectral-envelope-loss-weight cannot be negative.")
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
    if args.stage2_generator_hold_steps < 0:
        raise ValueError("--stage2-generator-hold-steps cannot be negative.")
    if args.stage2_generator_hold_lr <= 0:
        raise ValueError("--stage2-generator-hold-lr must be positive.")
    if args.stage2_discriminator_hold_steps < 0:
        raise ValueError("--stage2-discriminator-hold-steps cannot be negative.")
    if args.stage2_discriminator_hold_lr <= 0:
        raise ValueError("--stage2-discriminator-hold-lr must be positive.")
    if args.stage2_recon_transition_start_steps < 0:
        raise ValueError("--stage2-recon-transition-start-steps cannot be negative.")
    if args.stage2_recon_transition_end_steps < args.stage2_recon_transition_start_steps:
        raise ValueError(
            "--stage2-recon-transition-end-steps must be >= "
            "--stage2-recon-transition-start-steps."
        )
    if args.stage2_quality_gate_start_steps < 0:
        raise ValueError("--stage2-quality-gate-start-steps cannot be negative.")
    if args.stage2_best_checkpoint_min_step < 0:
        raise ValueError("--stage2-best-checkpoint-min-step cannot be negative.")
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
    for name in ("stage2_quality_retention_patience", "stage2_rvq_retention_patience"):
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
    batch_size = args.batch_size or stage_defaults["batch_size"]
    segment_seconds = (
        args.segment_seconds
        if args.segment_seconds is not None
        else stage_defaults["segment_seconds"]
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
    if early_stopping_min_steps < 0:
        raise ValueError("--early-stopping-min-steps cannot be negative.")
    if early_stopping_min_steps > num_train_steps:
        raise ValueError(
            "--early-stopping-min-steps cannot exceed --num-train-steps."
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
    correlation_loss_weight = 0.02 if args.stage in RECONSTRUCTION_STAGES else 0.0
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
        args.stage2_plateau_lr
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
        if stage2_plateau_lr_enabled:
            print(
                "Stage-2 LR schedule: standard warmup, then generator=1.000e-05 / "
                "discriminator=2.000e-05 until step "
                f"{args.stage2_plateau_start_steps}; validation ReduceLROnPlateau "
                "then lowers both together on quality-retained reconstruction score "
                "(10*wave + 1.1*Mel; lower is better) "
                f"(factor={args.stage2_plateau_factor}, "
                f"patience={args.stage2_plateau_patience}, "
                f"threshold={args.stage2_plateau_threshold}, "
                f"cooldown={args.stage2_plateau_cooldown}, "
                f"min_g_lr={args.stage2_plateau_min_lr}, "
                f"min_d_lr={args.stage2_plateau_discr_min_lr})"
            )
        else:
            print("Stage-2 LR schedule: fixed generator=1.000e-05, discriminator=2.000e-05 after the standard warmup.")
        print(
            "Stage-2 retention phase: generator is capped at "
            f"{args.stage2_generator_hold_lr:.3e} for steps [0, "
            f"{args.stage2_generator_hold_steps}), then returns to 1.000e-05; "
            "discriminators update from step 0 at "
            f"{args.stage2_discriminator_hold_lr:.3e} for steps [0, "
            f"{args.stage2_discriminator_hold_steps}), then return to 2.000e-05."
        )
        print(
            "Stage-2 reconstruction transition: retain Stage-1 weights "
            "(SI-SDR=0.07, corr=0.02, envelope=0.05, noise-floor=0.03) "
            f"through step {args.stage2_recon_transition_start_steps}, then "
            "linearly reach Stage-2 endpoints "
            "(SI-SDR=0.03, corr=0, envelope=0, noise-floor=0.02) by step "
            f"{args.stage2_recon_transition_end_steps}."
        )
        print(
            "Stage-2 candidate policy: quality baseline at initialization; "
            f"best checkpoints and quality hard-stop begin at step "
            f"{args.stage2_best_checkpoint_min_step}/"
            f"{args.stage2_quality_gate_start_steps}."
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
        f"click<={args.clean_gate_max_click_score}, "
        f"jump<={args.clean_gate_max_jump_ratio}, "
        f"p999_jump<={args.clean_gate_max_p999_jump_ratio}"
    )
    if args.stage == "gan_pretrain":
        print(
            "Stage-2 quality retention gate: "
            f"enabled={args.stage2_quality_retention_gate}, "
            f"aligned_si_sdr_drop<={args.stage2_max_aligned_si_sdr_drop:.2f} dB, "
            f"aligned_corr_drop<={args.stage2_max_aligned_corr_drop:.3f}, "
            f"quiet_hf_excess_rise<={args.stage2_max_quiet_hf_excess_db_rise:.2f} dB, "
            "q00/q01=(active>=0.30, perplexity>=16), "
            f"hard_stop=(quality={args.stage2_quality_retention_patience}, "
            f"rvq={args.stage2_rvq_retention_patience}) validation checks"
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
    print(f"Discriminator learning rate: {stage_defaults['discr_lr']}")
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
        f"patience={early_stopping_patience} validation checks, "
        f"min_delta={args.early_stopping_min_delta}, "
        f"min_steps={early_stopping_min_steps}"
    )
    print(f"Fixed validation batches: {args.best_eval_batches}")
    print(f"Dataset split: train {1 - args.valid_frac - args.test_frac:.2%}, valid {args.valid_frac:.2%}, test {args.test_frac:.2%}")
    print(f"Target sample rate: {sample_rate} Hz")
    print(f"Training segment: {segment_seconds:.3f} s")
    if args.stage in ("stream_finetune", "stream_finetune_long"):
        print(f"Internal streaming frame: {stream_frame_size} samples")
        print("Streaming context: per-layer causal state (no previous PCM frames)")
        print(f"Boundary loss weight: {args.boundary_loss_weight}")
        print(f"Boundary loss radius: {args.boundary_loss_radius} samples")
    print(f"Codebook size: {codebook_size}")
    print(f"RVQ quantizers: {num_quantizers}")
    print("RVQ quantize dropout: False")
    print("RVQ dead-code threshold: 2")
    print(f"RVQ codebook synchronization: {world_size > 1}")
    if args.stage in ("stream_finetune", "stream_finetune_long"):
        print("RVQ codebook training: frozen")
    elif args.stage == "gan_pretrain":
        print(
            "RVQ codebook training: frozen for steps [0, 5000), then enabled "
            "(including EMA statistics and dead-code replacement)"
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
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        si_sdr_loss_weight=si_sdr_loss_weight,
        click_loss_weight=click_loss_weight,
        jump_loss_weight=jump_loss_weight,
        spectral_envelope_loss_weight=spectral_envelope_loss_weight,
        preemph_loss_weight=preemph_loss_weight,
        noise_floor_loss_weight=noise_floor_loss_weight,
        stft_recon_loss_weight=stft_recon_loss_weight,
        frame_phase_loss_weight=frame_phase_loss_weight,
        gan_adversarial_max=stage_defaults.get("gan_adversarial_max", 0.001),
        gan_feature_max=stage_defaults.get("gan_feature_max", 5.0),
        decoder_upsample_mode=args.decoder_upsample_mode,
        decoder_residual_scale=decoder_residual_scale_start,
        decoder_linear_upsample_kernel_min=args.decoder_linear_upsample_kernel_min,
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
        discr_max_grad_norm=0.5,
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
        plateau_lr_update_discriminator=stage2_plateau_lr_enabled,
        plateau_lr_discr_min_lr=(
            args.stage2_plateau_discr_min_lr if stage2_plateau_lr_enabled
            else None
        ),
        plateau_lr_require_quality_retention=stage2_plateau_lr_enabled,
        generator_hold_steps=(
            args.stage2_generator_hold_steps
            if args.stage == "gan_pretrain"
            else 0
        ),
        generator_hold_lr=(
            args.stage2_generator_hold_lr
            if args.stage == "gan_pretrain"
            else None
        ),
        discriminator_hold_steps=(
            args.stage2_discriminator_hold_steps
            if args.stage == "gan_pretrain"
            else 0
        ),
        discriminator_hold_lr=(
            args.stage2_discriminator_hold_lr
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
        stage2_initial_noise_floor_loss_weight=(
            0.03 if args.stage == "gan_pretrain" else 0.
        ),
        quality_retention_start_step=(
            args.stage2_quality_gate_start_steps
            if args.stage == "gan_pretrain"
            else 0
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
                    if args.stage == "spectral_refine"
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
        clean_gate_max_jump_ratio=args.clean_gate_max_jump_ratio,
        clean_gate_max_p999_jump_ratio=args.clean_gate_max_p999_jump_ratio,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_steps=early_stopping_min_steps,
        enable_gan=args.stage in GAN_STAGES,
        gan_start_step=stage_defaults["gan_start"],
        gan_ramp_steps=stage_defaults["gan_ramp"],
        gan_adversarial_max=stage_defaults.get("gan_adversarial_max", 0.001),
        gan_feature_max=stage_defaults.get("gan_feature_max", 5.),
        quality_retention_gate=(
            args.stage == "gan_pretrain" and args.stage2_quality_retention_gate
        ),
        quality_retention_max_aligned_si_sdr_drop=args.stage2_max_aligned_si_sdr_drop,
        quality_retention_max_aligned_corr_drop=args.stage2_max_aligned_corr_drop,
        quality_retention_max_quiet_hf_excess_db_rise=args.stage2_max_quiet_hf_excess_db_rise,
        quality_retention_patience=args.stage2_quality_retention_patience,
        quality_retention_rvq_patience=args.stage2_rvq_retention_patience,
        freeze_codebook_after_step=stage_defaults.get("freeze_codebook_after_step"),
        freeze_codebook_before_step=(
            5_000 if args.stage == "gan_pretrain" else None
        ),
        freeze_codebook_during_training=args.stage in (
            "stream_finetune",
            "stream_finetune_long",
        ),
        use_ema=use_ema,
        ema_beta=stage_defaults["ema_beta"],
        ema_update_after_step=stage_defaults["ema_update_after_step"],
        ema_update_every=stage_defaults["ema_update_every"],
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

    checkpoint = (
        latest_checkpoint(results_dir)
        if args.resume and not args.test_only
        else None
    )

    if args.test_only:
        if args.test_checkpoint is None:
            raise ValueError("--test-only requires --test-checkpoint")
        args.test_checkpoint = args.test_checkpoint.expanduser().resolve()
        if not args.test_checkpoint.is_file():
            raise FileNotFoundError(
                f"Test checkpoint not found: {args.test_checkpoint}"
            )
        print(f"Test-only mode; training and checkpoint writes are disabled: {args.test_checkpoint}")
    elif checkpoint is not None:
        print(f"Resuming from checkpoint: {checkpoint}")
        trainer.load(str(checkpoint))
    else:
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
                ("best_by_aligned_si_sdr.pt", "best_selected.pt")
                if predecessor_stage in ("recon_pretrain", "spectral_refine")
                else ("best_selected.pt",)
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
            )
            if args.stage in ("spectral_refine", "gan_pretrain"):
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
                if args.stage == "gan_pretrain":
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
        not args.test_only and
        args.stage == "gan_pretrain" and
        trainer.quality_retention_gate and
        not trainer.has_quality_retention_baseline
    ):
        # Evaluate once on rank 0, then distribute the numerical baseline so
        # resumed / distributed workers carry identical retention state.
        baseline_keys = (
            "aligned_si_sdr",
            "aligned_correlation",
            "quiet_hf_excess_db",
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
        if trainer.is_main:
            print(
                "Stage-2 quality retention baseline recorded from initialized "
                "checkpoint: "
                f"aligned_si_sdr={baseline_metrics['aligned_si_sdr']:.3f}, "
                f"aligned_corr={baseline_metrics['aligned_correlation']:.3f}, "
                f"quiet_hf_excess_db={baseline_metrics['quiet_hf_excess_db']:.3f}"
            )
        trainer.accelerator.wait_for_everyone()

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
    best_by_aligned_si_sdr = results_dir / "best_by_aligned_si_sdr.pt"
    best_selected = results_dir / "best_selected.pt"
    best_checkpoint = (
        args.test_checkpoint
        if args.test_only
        else (
            best_by_aligned_si_sdr
            if best_by_aligned_si_sdr.exists()
            else best_selected
        )
    )

    if best_checkpoint.exists() and trainer.test_files:
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
        test_report_file = (
            (args.test_report_file or (results_dir / "stage1_test_report.txt")).resolve()
            if args.stage in ("recon_pretrain", "spectral_refine")
            else None
        )

        if is_main:
            print(
                f"Evaluating {len(selected_test_files)} held-out test files with "
                f"validation-selected checkpoint on {world_size} process(es): "
                f"{best_checkpoint}"
            )
            if save_test_reconstructions:
                print(f"Saving stage-1 test reconstructions to: {test_recon_dir}")
                print(f"Writing stage-1 test report to: {test_report_file}")

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
            'multi_spectral_recon_loss',
            'stft_recon_loss',
            'recon_loss',
            'wave_mse',
            'boundary_loss',
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
                f"mel={test_metrics['multi_spectral_recon_loss']:.6f}, "
                f"stft={test_metrics['stft_recon_loss']:.6f}, "
                f"recon={test_metrics['recon_loss']:.6f}, "
                f"mse={test_metrics['wave_mse']:.6f}, "
                f"boundary={test_metrics['boundary_loss']:.6f}, "
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
                f"ac_320_isolated={test_metrics.get('ac_320_isolated', 0.):+.6f}, "
                f"phase_peak_db={test_metrics.get('frame_phase_peak_db', 0.):+.3f}, "
                f"comb_median_db={test_metrics.get('comb_median_excess_db', 0.):+.3f}, "
                f"active_codes={test_metrics['active_code_ratio']:.6f}, "
                f"perplexity={test_metrics['codebook_perplexity']:.6f}"
            )
            if test_report_file is not None:
                test_report_file.parent.mkdir(parents=True, exist_ok=True)
                with test_report_file.open("w", encoding="utf-8") as f:
                    f.write("Stage-1/1.5 held-out test report\n")
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
