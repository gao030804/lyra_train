from __future__ import annotations

import re
import copy
import pickle
import random
from math import isfinite, sqrt
from datetime import timedelta
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import partial
from collections import Counter, deque
from contextlib import contextmanager, nullcontext

from beartype.typing import Type
from typing_extensions import Annotated
import numpy as np

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset, random_split

import pytorch_warmup as warmup

from einops import rearrange

from audiolm_pytorch.optimizer import get_optimizer
import wandb
from ema_pytorch import EMA

from audiolm_pytorch.soundstream import SoundStream
from audiolm_pytorch.encodec import EncodecWrapper

from audiolm_pytorch.audiolm_pytorch import (
    SemanticTransformer,
    SemanticTransformerWrapper,
    CoarseTransformer,
    CoarseTransformerWrapper,
    FineTransformer,
    FineTransformerWrapper,
    FairseqVQWav2Vec,
    HubertWithKmeans
)

from audiolm_pytorch.data import (
    SoundDataset,
    get_dataloader,
    load_audio_file,
    save_audio_file
)
from audiolm_pytorch.utils import AudioConditionerBase

from audiolm_pytorch.version import __version__
from packaging import version

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.tracking import WandBTracker

# constants

DEFAULT_SAMPLE_RATE = 16000

# make sure only one trainer is instantiated

ONE_TRAINER_INSTANTIATED = False

def check_one_trainer():
    global ONE_TRAINER_INSTANTIATED
    assert not ONE_TRAINER_INSTANTIATED, 'only one Trainer can be instantiated at a time for training'
    ONE_TRAINER_INSTANTIATED = True

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters = True)

# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
    raw_wave = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
    text = list[str],
    text_embeds = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim == 3]
    ],
)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def safe_float(val, default_value = 0.):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default_value

def noop(*args, **kwargs):
    pass

def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def dict_values_to_device(d: dict, device):
    out = {}
    for k, v in d.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict(Counter(tup))
    return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/semantic.transformer.20000.pt" which is
    for 20k train steps. Returns -1 when the filename has no numeric step suffix.
    """
    filename = Path(checkpoint_path).name
    match = re.search(r'\.(\d+)\.pt$', filename)
    return int(match.group(1)) if match else -1

# optimizer with scheduler + warmup

class OptimizerWithWarmupSchedule(nn.Module):
    @beartype
    def __init__(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: Type[LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        warmup_steps: int = 0
    ):
        super().__init__()

        if exists(scheduler):
            self.scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            self.scheduler = None

        # The LR scheduler must capture the undampened base LR before warmup
        # modifies the optimizer's current LR.
        self.warmup = warmup.LinearWarmup(
            optimizer,
            warmup_period = warmup_steps
        )
        self.optimizer = optimizer

        if exists(self.scheduler):
            self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)
        else:
            self.optimizer = accelerator.prepare(self.optimizer)
        self.accelerator = accelerator

    def state_dict(self):
        return dict(
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict() if exists(self.scheduler) else None,
            warmup = self.warmup.state_dict()
        )

    def load_state_dict(self, pkg):
        self.optimizer.load_state_dict(pkg['optimizer'])
        if exists(self.scheduler) and exists(pkg.get('scheduler')):
            self.scheduler.load_state_dict(pkg['scheduler'])
        self.warmup.load_state_dict(pkg['warmup'])

    def bind_schedulers_to_current_optimizer(self):
        # `accelerator.prepare()` may replace the optimizer object held by this
        # wrapper. Keep warmup / scheduler references pointed at the optimizer
        # that is actually stepped and logged during training.
        self.warmup.optimizer = self.optimizer
        if exists(self.scheduler) and hasattr(self.scheduler, 'optimizer'):
            self.scheduler.optimizer = self.optimizer

    def sync_warmup_lrs_from_optimizer(self):
        # pytorch_warmup restores `warmup.lrs` at the start of every dampening
        # context.  External schedulers such as ReduceLROnPlateau step outside
        # that context, so copy the current optimizer LR back into warmup state
        # after any external LR change.
        self.warmup.lrs = [
            group['lr']
            for group in self.optimizer.param_groups
        ]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

        if not self.accelerator.optimizer_step_was_skipped:
            with self.warmup.dampening():
                if exists(self.scheduler):
                    self.scheduler.step()

# main trainer class

class SoundStreamTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        soundstream: SoundStream,
        *,
        num_train_steps: int,
        batch_size: int,
        data_max_length: int = None,
        data_max_length_seconds: int | float = None,
        dataset_max_files: int | None = None,
        dataset_fixed_crop: bool = False,
        dataset_min_rms_db: float | None = None,
        folder: str = None,
        dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        lr: float = 2e-4,
        discr_lr: float | None = None,
        stft_discr_lr: float | None = None,
        grad_accum_every: int = 4,
        wd: float = 0.,
        warmup_steps: int = 1000,
        scheduler: Type[LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        plateau_lr_enabled: bool = False,
        plateau_lr_start_steps: int = 20_000,
        plateau_lr_factor: float = 0.5,
        plateau_lr_patience: int = 12,
        plateau_lr_threshold: float = 0.05,
        plateau_lr_cooldown: int = 2,
        plateau_lr_min_lr: float = 2e-5,
        plateau_lr_unclean_grace_checks: int = 8,
        plateau_lr_metric: str = 'aligned_si_sdr',
        plateau_lr_update_discriminator: bool = False,
        plateau_lr_discr_min_lr: float | None = None,
        plateau_lr_stft_discr_min_lr: float | None = None,
        plateau_lr_require_quality_retention: bool = False,
        generator_hold_steps: int = 0,
        generator_hold_lr: float | None = None,
        discriminator_hold_steps: int = 0,
        discriminator_hold_lr: float | None = None,
        stage2_recon_transition_start_steps: int | None = None,
        stage2_recon_transition_end_steps: int | None = None,
        stage2_initial_si_sdr_loss_weight: float = 0.,
        stage2_initial_correlation_loss_weight: float = 0.,
        stage2_initial_spectral_envelope_loss_weight: float = 0.,
        stage2_initial_noise_floor_loss_weight: float = 0.,
        quality_retention_start_step: int = 0,
        discr_warmup_steps: int | None = None,
        discr_scheduler: Type[LRScheduler] | None = None,
        discr_scheduler_kwargs: dict = dict(),
        max_grad_norm: float = 0.5,
        discr_max_grad_norm: float = None,
        save_results_every: int = 100,
        save_model_every: int = 1000,
        best_eval_every: int | None = None,
        best_eval_batches: int = 8,
        si_sdr_loss_start_steps: int = 0,
        si_sdr_loss_warmup_steps: int = 0,
        spectral_envelope_loss_start_steps: int = 0,
        spectral_envelope_loss_warmup_steps: int = 0,
        stft_recon_loss_start_steps: int = 0,
        stft_recon_loss_warmup_steps: int = 0,
        frame_phase_loss_start_steps: int = 0,
        frame_phase_loss_warmup_steps: int = 0,
        transient_loss_warmup_steps: int = 0,
        decoder_residual_scale_start: float = 1.,
        decoder_residual_scale_end: float = 1.,
        decoder_residual_scale_warmup_start_steps: int = 0,
        decoder_residual_scale_warmup_end_steps: int = 0,
        decoder_x8_residual_scale_target: float | None = None,
        decoder_x8_residual_scale_ramp_steps: int = 0,
        best_checkpoint_min_step: int = 0,
        frame_leakage_checkpoint: bool = False,
        clean_gate: bool = True,
        clean_gate_min_aligned_si_sdr: float = 0.,
        clean_gate_min_aligned_corr: float = 0.65,
        clean_gate_min_rms_ratio: float = 0.4,
        clean_gate_max_rms_ratio: float = 2.5,
        clean_gate_max_recon_peak: float = 1.2,
        clean_gate_max_recon_clip_fraction: float = 1e-3,
        clean_gate_max_click_score: float = 6.,
        clean_gate_max_jump_ratio: float = 2.,
        clean_gate_max_p999_jump_ratio: float = 1.75,
        quality_retention_gate: bool = False,
        quality_retention_max_aligned_si_sdr_drop: float = 0.30,
        quality_retention_max_aligned_corr_drop: float = 0.02,
        quality_retention_max_quiet_hf_excess_db_rise: float = 0.50,
        quality_retention_max_click_score_rise: float = 0.30,
        quality_retention_q00_min_active_ratio: float = 0.70,
        quality_retention_q00_min_perplexity: float = 50.,
        quality_retention_q00_warn_active_ratio: float = 0.80,
        quality_retention_q00_warn_perplexity: float = 70.,
        quality_retention_patience: int = 3,
        quality_retention_rvq_patience: int = 2,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.,
        early_stopping_min_steps: int = 0,
        enable_gan: bool = True,
        allow_discriminator_reinitialization: bool = False,
        gan_start_step: int = 0,
        gan_ramp_steps: int = 0,
        gan_adversarial_max: float = 0.1,
        gan_feature_max: float = 10.,
        freeze_codebook_after_step: int | None = None,
        freeze_codebook_before_step: int | None = None,
        freeze_codebook_during_training: bool = False,
        freeze_encoder_before_step: int | None = None,
        log_losses_every: int = 1,
        results_folder: str = './results',
        valid_frac: float = 0.05,
        test_frac: float = 0.,
        split_by_speaker: bool = False,
        best_checkpoint_metric: str | None = None,
        random_split_seed: int = 42,
        dataloader_seed: int | None = None,
        use_ema: bool = True,
        ema_beta: float = 0.995,
        ema_update_after_step: int = 500,
        ema_update_every: int = 10,
        apply_grad_penalty_every: int = 4,
        waveform_grad_penalty_gamma: float = 0.,
        stft_grad_penalty_every: int = 32,
        stft_grad_penalty_gamma: float = 5e-3,
        dl_num_workers: int = 0,
        accelerator: Accelerator | None = None,
        accelerate_kwargs: dict = dict(),
        init_process_group_timeout_seconds = 1800,
        dataloader_drop_last = True,
        split_batches = False,
        use_wandb_tracking = False,
        force_clear_prev_results: bool = None  # set to True | False to skip the prompt
    ):
        """
        Initialize with a SoundStream instance and either a folder containing audio data or
        train/val DataLoader instances.
        """
        super().__init__()
        check_one_trainer()

        self.accelerator = accelerator
        assert not (exists(accelerator) and len(accelerate_kwargs) > 0)

        self.use_wandb_tracking = use_wandb_tracking

        if not exists(self.accelerator):
            init_process_kwargs = InitProcessGroupKwargs(timeout = timedelta(seconds = init_process_group_timeout_seconds))

            if use_wandb_tracking:
                accelerate_kwargs.update(log_with = 'wandb')

            # AcceleratedScheduler otherwise advances once per process when
            # split_batches=False. A six-GPU run would therefore consume a
            # step-based LR schedule six times faster than intended.
            accelerate_kwargs.setdefault(
                'step_scheduler_with_optimizer',
                False
            )
            self.accelerator = Accelerator(
                kwargs_handlers = [DEFAULT_DDP_KWARGS, init_process_kwargs],
                split_batches = split_batches,
                **accelerate_kwargs
            )

        self.soundstream = soundstream

        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.ema_update_after_step = ema_update_after_step
        self.ema_update_every = ema_update_every

        self.register_buffer('steps', torch.tensor(0))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        self.si_sdr_loss_max_weight = float(
            getattr(soundstream, 'si_sdr_loss_weight', 0.)
        )
        self.si_sdr_loss_start_steps = si_sdr_loss_start_steps
        self.si_sdr_loss_warmup_steps = si_sdr_loss_warmup_steps
        self.spectral_envelope_loss_max_weight = float(
            getattr(soundstream, 'spectral_envelope_loss_weight', 0.)
        )
        self.spectral_envelope_loss_start_steps = spectral_envelope_loss_start_steps
        self.spectral_envelope_loss_warmup_steps = spectral_envelope_loss_warmup_steps
        self.stft_recon_loss_max_weight = float(
            getattr(soundstream, 'stft_recon_loss_weight', 0.)
        )
        self.stft_recon_loss_start_steps = stft_recon_loss_start_steps
        self.stft_recon_loss_warmup_steps = stft_recon_loss_warmup_steps
        self.frame_phase_loss_max_weight = float(
            getattr(soundstream, 'frame_phase_loss_weight', 0.)
        )
        self.frame_phase_loss_start_steps = frame_phase_loss_start_steps
        self.frame_phase_loss_warmup_steps = frame_phase_loss_warmup_steps
        # The Stage-2 reconstruction transition interpolates toward these
        # configured endpoint values rather than hard-coding a second copy.
        self.stage2_final_correlation_loss_weight = float(
            getattr(soundstream, 'correlation_loss_weight', 0.)
        )
        self.stage2_final_noise_floor_loss_weight = float(
            getattr(soundstream, 'noise_floor_loss_weight', 0.)
        )
        self.click_loss_max_weight = float(
            getattr(soundstream, 'click_loss_weight', 0.)
        )
        self.jump_loss_max_weight = float(
            getattr(soundstream, 'jump_loss_weight', 0.)
        )
        self.transient_loss_warmup_steps = transient_loss_warmup_steps
        assert decoder_residual_scale_start >= 0.
        assert decoder_residual_scale_end >= 0.
        assert decoder_residual_scale_warmup_start_steps >= 0
        assert decoder_residual_scale_warmup_end_steps >= decoder_residual_scale_warmup_start_steps
        assert best_checkpoint_min_step >= 0
        self.decoder_residual_scale_start = decoder_residual_scale_start
        self.decoder_residual_scale_end = decoder_residual_scale_end
        self.decoder_residual_scale_warmup_start_steps = decoder_residual_scale_warmup_start_steps
        self.decoder_residual_scale_warmup_end_steps = decoder_residual_scale_warmup_end_steps
        if exists(decoder_x8_residual_scale_target):
            assert decoder_x8_residual_scale_target >= 0.
        assert decoder_x8_residual_scale_ramp_steps >= 0
        self.decoder_x8_residual_scale_target = decoder_x8_residual_scale_target
        self.decoder_x8_residual_scale_ramp_steps = decoder_x8_residual_scale_ramp_steps
        initial_block_scales = getattr(soundstream, 'decoder_block_residual_scales', ())
        self.decoder_x8_residual_scale_start = (
            float(initial_block_scales[0]) if initial_block_scales else decoder_residual_scale_end
        )
        self.best_checkpoint_min_step = best_checkpoint_min_step
        self.frame_leakage_checkpoint = frame_leakage_checkpoint
        self.clean_gate = clean_gate
        self.clean_gate_min_aligned_si_sdr = clean_gate_min_aligned_si_sdr
        self.clean_gate_min_aligned_corr = clean_gate_min_aligned_corr
        self.clean_gate_min_rms_ratio = clean_gate_min_rms_ratio
        self.clean_gate_max_rms_ratio = clean_gate_max_rms_ratio
        self.clean_gate_max_recon_peak = clean_gate_max_recon_peak
        self.clean_gate_max_recon_clip_fraction = clean_gate_max_recon_clip_fraction
        self.clean_gate_max_click_score = clean_gate_max_click_score
        self.clean_gate_max_jump_ratio = clean_gate_max_jump_ratio
        self.clean_gate_max_p999_jump_ratio = clean_gate_max_p999_jump_ratio
        assert quality_retention_max_aligned_si_sdr_drop >= 0.
        assert quality_retention_max_aligned_corr_drop >= 0.
        assert quality_retention_max_quiet_hf_excess_db_rise >= 0.
        assert quality_retention_max_click_score_rise >= 0.
        assert 0. <= quality_retention_q00_min_active_ratio <= 1.
        assert quality_retention_q00_min_perplexity >= 0.
        assert 0. <= quality_retention_q00_warn_active_ratio <= 1.
        assert quality_retention_q00_warn_perplexity >= 0.
        assert quality_retention_patience > 0
        assert quality_retention_rvq_patience > 0
        self.quality_retention_gate = quality_retention_gate
        self.quality_retention_max_aligned_si_sdr_drop = quality_retention_max_aligned_si_sdr_drop
        self.quality_retention_max_aligned_corr_drop = quality_retention_max_aligned_corr_drop
        self.quality_retention_max_quiet_hf_excess_db_rise = quality_retention_max_quiet_hf_excess_db_rise
        self.quality_retention_max_click_score_rise = quality_retention_max_click_score_rise
        self.quality_retention_q00_min_active_ratio = quality_retention_q00_min_active_ratio
        self.quality_retention_q00_min_perplexity = quality_retention_q00_min_perplexity
        self.quality_retention_q00_warn_active_ratio = quality_retention_q00_warn_active_ratio
        self.quality_retention_q00_warn_perplexity = quality_retention_q00_warn_perplexity
        self.quality_retention_patience = quality_retention_patience
        self.quality_retention_rvq_patience = quality_retention_rvq_patience
        self.quality_retention_baseline = None
        self.quality_retention_bad_evals = 0
        self.quality_retention_rvq_bad_evals = 0
        self.stft_saturation_history = deque(maxlen = 1000)

        hyperparameters = {
            "num_train_steps": num_train_steps,
            "batch_size": batch_size,
            "gradient_accum_every": grad_accum_every,
            "learning_rate": lr,
            "discriminator_learning_rate": discr_lr,
            "stft_discriminator_learning_rate": stft_discr_lr,
            "target_sample_hz": soundstream.target_sample_hz,
        }

        # optimizers

        self.optim = OptimizerWithWarmupSchedule(
            self.accelerator,
            get_optimizer(soundstream.non_discr_parameters(), lr = lr, wd = wd),
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            warmup_steps = warmup_steps
        )
        # Keep the undampened target LR so a temporary Stage-2 retention cap
        # can be released exactly at its configured boundary.
        self.generator_hold_base_lrs = tuple(self.optim.warmup.lrs)

        assert plateau_lr_start_steps >= 0
        assert 0. < plateau_lr_factor < 1.
        assert plateau_lr_patience > 0
        assert plateau_lr_threshold >= 0.
        assert plateau_lr_cooldown >= 0
        assert plateau_lr_min_lr >= 0.
        assert plateau_lr_metric in {'aligned_si_sdr', 'score'}
        assert plateau_lr_unclean_grace_checks >= 0
        plateau_lr_discr_min_lr = default(
            plateau_lr_discr_min_lr,
            plateau_lr_min_lr
        )
        plateau_lr_stft_discr_min_lr = default(
            plateau_lr_stft_discr_min_lr,
            plateau_lr_discr_min_lr
        )
        assert plateau_lr_discr_min_lr >= 0.
        assert plateau_lr_stft_discr_min_lr >= 0.
        assert generator_hold_steps >= 0
        if exists(generator_hold_lr):
            assert generator_hold_lr > 0.
        self.plateau_lr_enabled = plateau_lr_enabled
        self.plateau_lr_start_steps = plateau_lr_start_steps
        self.plateau_lr_factor = plateau_lr_factor
        self.plateau_lr_patience = plateau_lr_patience
        self.plateau_lr_threshold = plateau_lr_threshold
        self.plateau_lr_cooldown = plateau_lr_cooldown
        self.plateau_lr_min_lr = plateau_lr_min_lr
        self.plateau_lr_metric = plateau_lr_metric
        self.plateau_lr_unclean_grace_checks = plateau_lr_unclean_grace_checks
        self.plateau_lr_update_discriminator = plateau_lr_update_discriminator
        self.plateau_lr_discr_min_lr = plateau_lr_discr_min_lr
        self.plateau_lr_stft_discr_min_lr = plateau_lr_stft_discr_min_lr
        self.plateau_lr_require_quality_retention = plateau_lr_require_quality_retention
        self.generator_hold_steps = generator_hold_steps
        self.generator_hold_lr = generator_hold_lr
        assert discriminator_hold_steps >= 0
        if exists(discriminator_hold_lr):
            assert discriminator_hold_lr > 0.
        self.discriminator_hold_steps = discriminator_hold_steps
        self.discriminator_hold_lr = discriminator_hold_lr
        if exists(stage2_recon_transition_start_steps):
            assert stage2_recon_transition_start_steps >= 0
            assert exists(stage2_recon_transition_end_steps)
            assert stage2_recon_transition_end_steps >= stage2_recon_transition_start_steps
        self.stage2_recon_transition_start_steps = stage2_recon_transition_start_steps
        self.stage2_recon_transition_end_steps = stage2_recon_transition_end_steps
        self.stage2_initial_si_sdr_loss_weight = stage2_initial_si_sdr_loss_weight
        self.stage2_initial_correlation_loss_weight = stage2_initial_correlation_loss_weight
        self.stage2_initial_spectral_envelope_loss_weight = stage2_initial_spectral_envelope_loss_weight
        self.stage2_initial_noise_floor_loss_weight = stage2_initial_noise_floor_loss_weight
        assert quality_retention_start_step >= 0
        self.quality_retention_start_step = quality_retention_start_step
        self.plateau_lr_unclean_checks = 0
        # Build this after the trainer modules are passed through
        # accelerator.prepare().  Otherwise the scheduler can hold a reference
        # to the pre-prepare optimizer while training uses the prepared one.
        self.plateau_scheduler = None

        discr_lr = default(discr_lr, lr)
        stft_discr_lr = default(stft_discr_lr, discr_lr)
        assert discr_lr > 0.
        assert stft_discr_lr > 0.
        discr_warmup_steps = default(discr_warmup_steps, warmup_steps)

        for discr_optimizer_key, discr in self.multiscale_discriminator_iter():
            one_multiscale_discr_optimizer = OptimizerWithWarmupSchedule(
                self.accelerator,
                get_optimizer(discr.parameters(), lr = discr_lr, wd = wd),
                scheduler = discr_scheduler,
                scheduler_kwargs = discr_scheduler_kwargs,
                warmup_steps = discr_warmup_steps
            )
            setattr(self, discr_optimizer_key, one_multiscale_discr_optimizer)

        self.discr_optim = OptimizerWithWarmupSchedule(
            self.accelerator,
            get_optimizer(soundstream.stft_discriminator.parameters(), lr = stft_discr_lr, wd = wd),
            scheduler = discr_scheduler,
            scheduler_kwargs = discr_scheduler_kwargs,
            warmup_steps = discr_warmup_steps
        )
        self.discriminator_hold_base_lrs = {
            name: tuple(float(lr) for lr in optimizer.warmup.lrs)
            for name, optimizer in self.multiscale_discriminator_optim_iter()
        }
        self.discriminator_hold_base_lrs['stft'] = tuple(
            float(lr) for lr in self.discr_optim.warmup.lrs
        )

        # max grad norm

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        if exists(folder):
            assert not exists(dataset)
            assert not exists(val_dataset)
            assert not exists(train_dataloader)
            assert not exists(val_dataloader)

            # create dataset

            if exists(data_max_length_seconds):
                assert not exists(data_max_length)
                data_max_length = int(data_max_length_seconds * soundstream.target_sample_hz)
            else:
                assert exists(data_max_length)

            hyperparameters['data_max_length'] = data_max_length

            dataset = SoundDataset(
                folder,
                max_length = data_max_length,
                target_sample_hz = soundstream.target_sample_hz,
                seq_len_multiple_of = soundstream.seq_len_multiple_of,
                max_files = dataset_max_files,
                fixed_crop = dataset_fixed_crop,
                min_rms_db = dataset_min_rms_db
            )

            assert len(dataset) >= batch_size, 'dataset must have sufficient samples for training'

        if exists(dataset):
            assert not exists(train_dataloader)
            assert not exists(val_dataloader)

            # maybe split for validation and held-out test

            if valid_frac > 0 or test_frac > 0:
                assert valid_frac >= 0 and test_frac >= 0
                assert (valid_frac + test_frac) < 1, 'valid_frac + test_frac must be less than 1'

                train_frac = 1 - valid_frac - test_frac
                if split_by_speaker:
                    assert hasattr(dataset, 'files'), 'speaker split requires a dataset with a files attribute'
                    root = Path(folder).resolve()
                    speaker_to_indices = {}

                    for index, file in enumerate(dataset.files):
                        relative_parts = Path(file).resolve().relative_to(root).parts
                        assert len(relative_parts) >= 2, f'audio file must be inside a speaker subdirectory: {file}'
                        speaker_to_indices.setdefault(relative_parts[0], []).append(index)

                    speakers = sorted(speaker_to_indices)
                    permutation = torch.randperm(
                        len(speakers),
                        generator = torch.Generator().manual_seed(random_split_seed)
                    ).tolist()
                    speakers = [speakers[index] for index in permutation]

                    train_speaker_count = int(round(train_frac * len(speakers)))
                    valid_speaker_count = int(round(valid_frac * len(speakers)))
                    test_speaker_count = len(speakers) - train_speaker_count - valid_speaker_count

                    if test_frac == 0:
                        valid_speaker_count = len(speakers) - train_speaker_count
                        test_speaker_count = 0

                    train_speakers = set(speakers[:train_speaker_count])
                    valid_speakers = set(speakers[train_speaker_count:train_speaker_count + valid_speaker_count])
                    test_speakers = set(speakers[train_speaker_count + valid_speaker_count:])

                    def indices_for(selected_speakers, interleave = False):
                        grouped_indices = [
                            speaker_to_indices[speaker]
                            for speaker in sorted(selected_speakers)
                        ]

                        if not interleave:
                            return [
                                index
                                for group in grouped_indices
                                for index in group
                            ]

                        return [
                            index
                            for position in range(max(map(len, grouped_indices)))
                            for group in grouped_indices
                            if position < len(group)
                            for index in (group[position],)
                        ]

                    base_dataset = dataset
                    dataset = Subset(base_dataset, indices_for(train_speakers))
                    val_dataset = Subset(base_dataset, indices_for(valid_speakers, interleave = True))
                    test_indices = indices_for(test_speakers, interleave = True)
                    self.test_dataset = Subset(base_dataset, test_indices) if test_speaker_count > 0 else None
                    self.test_files = [base_dataset.files[index] for index in test_indices]

                    split_msg = (
                        f'speaker split: train {len(train_speakers)} speakers / {len(dataset)} samples, '
                        f'valid {len(valid_speakers)} speakers / {len(val_dataset)} samples'
                    )
                    if exists(self.test_dataset):
                        split_msg += f', test {len(test_speakers)} speakers / {len(self.test_dataset)} samples'
                else:
                    train_size = int(round(train_frac * len(dataset)))
                    valid_size = int(round(valid_frac * len(dataset)))
                    test_size = len(dataset) - train_size - valid_size

                    if test_frac == 0:
                        valid_size = len(dataset) - train_size
                        test_size = 0

                    split_sizes = [train_size, valid_size]

                    if test_frac > 0:
                        split_sizes.append(test_size)

                    split_datasets = random_split(dataset, split_sizes, generator = torch.Generator().manual_seed(random_split_seed))
                    dataset, val_dataset = split_datasets[:2]
                    self.test_dataset = split_datasets[2] if test_frac > 0 else None

                    split_msg = f'training with dataset of {len(dataset)} samples and validating with randomly splitted {len(val_dataset)} samples'

                if exists(self.test_dataset):
                    if not split_by_speaker:
                        split_msg += f'; holding out {len(self.test_dataset)} test samples'

                self.print(split_msg)
            else:
                val_dataset = dataset
                self.test_dataset = None
                self.print(f'training with shared training and valid dataset of {len(dataset)} samples')

            assert len(val_dataset) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(val_dataset)}) for training'
            assert not exists(self.test_dataset) or len(self.test_dataset) >= batch_size, f'test dataset must have sufficient number of samples (currently {len(self.test_dataset)})'

            dataloader_kwargs = dict(
                pin_memory = self.accelerator.device.type == 'cuda'
            )
            if dl_num_workers > 0:
                dataloader_kwargs.update(
                    persistent_workers = True,
                    prefetch_factor = 2
                )
            if exists(dataloader_seed):
                generator = torch.Generator()
                generator.manual_seed(dataloader_seed)

                def seed_worker(worker_id):
                    worker_seed = dataloader_seed + worker_id
                    random.seed(worker_seed)
                    np.random.seed(worker_seed % (2 ** 32))
                    torch.manual_seed(worker_seed)

                dataloader_kwargs.update(
                    generator = generator,
                    worker_init_fn = seed_worker
                )

            train_dataloader = get_dataloader(dataset, batch_size = batch_size, num_workers = dl_num_workers, shuffle = True, drop_last = dataloader_drop_last, **dataloader_kwargs)
            val_dataloader = get_dataloader(val_dataset, batch_size = batch_size, num_workers = dl_num_workers, shuffle = False, drop_last = dataloader_drop_last, **dataloader_kwargs)
            self.test_dl = get_dataloader(self.test_dataset, batch_size = batch_size, num_workers = dl_num_workers, shuffle = False, drop_last = False, **dataloader_kwargs) if exists(self.test_dataset) else None

        # dataloader

        self.dl = train_dataloader
        self.valid_dl = val_dataloader
        self.valid_dataset = val_dataset
        self.test_dl = getattr(self, 'test_dl', None)
        self.test_files = getattr(self, 'test_files', None)

        assert exists(self.dl) and exists(self.valid_dl)

        # prepare with accelerator

        (
            self.soundstream,
            self.optim,
            self.discr_optim,
            self.dl
        ) = self.accelerator.prepare(
            self.soundstream,
            self.optim,
            self.discr_optim,
            self.dl
        )

        self.optim.bind_schedulers_to_current_optimizer()
        self.discr_optim.bind_schedulers_to_current_optimizer()
        self.plateau_scheduler = self.create_plateau_scheduler()

        # Build EMA after Accelerator has placed and wrapped the online model.
        # Creating it before prepare leaves EMA's source model on CPU under DDP,
        # which fails on the first moving-average update against the CUDA copy.
        if self.use_ema:
            # The RVQ codebooks already maintain their own EMA statistics.
            # Copy those buffers directly so the outer model EMA does not
            # introduce a second lag between encoder outputs and code vectors.
            codebook_buffer_names = {
                name
                for name, _ in self.unwrapped_soundstream.named_buffers()
                if (
                    name.startswith('rq.') and
                    '._codebook.' in name
                )
            }
            self.ema_soundstream = EMA(
                self.unwrapped_soundstream,
                beta = self.ema_beta,
                update_after_step = self.ema_update_after_step,
                update_every = self.ema_update_every,
                param_or_buffer_names_no_ema = codebook_buffer_names
            ).to(self.device)

        # prepare the multiscale discriminators with accelerator

        for name, _ in self.multiscale_discriminator_iter():
            optimizer = getattr(self, name)
            optimizer = self.accelerator.prepare(optimizer)
            optimizer.bind_schedulers_to_current_optimizer()
            setattr(self, name, optimizer)

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.best_eval_every = best_eval_every
        assert best_eval_batches > 0
        self.best_eval_batches = best_eval_batches
        self.best_validation_seed = random_split_seed
        self.fixed_best_valid_waves = None
        self.fixed_result_valid_samples = None
        self.save_results_every = save_results_every
        self.log_losses_every = log_losses_every
        self.best_checkpoint_metric = best_checkpoint_metric
        self.best_valid_score = float('inf')
        self.best_aligned_si_sdr = float('-inf')
        self.best_frame_leakage_score = float('inf')
        assert not exists(early_stopping_patience) or early_stopping_patience > 0
        assert early_stopping_min_delta >= 0.
        assert early_stopping_min_steps >= 0
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_min_steps = early_stopping_min_steps
        self.early_stopping_bad_evals = 0
        self.early_stopping_triggered = False
        self.enable_gan = enable_gan
        self.allow_discriminator_reinitialization = allow_discriminator_reinitialization
        self.gan_start_step = gan_start_step
        self.gan_ramp_steps = gan_ramp_steps
        self.gan_adversarial_max = gan_adversarial_max
        self.gan_feature_max = gan_feature_max
        assert not exists(freeze_codebook_after_step) or freeze_codebook_after_step >= 0
        assert not exists(freeze_codebook_before_step) or freeze_codebook_before_step >= 0
        assert not exists(freeze_encoder_before_step) or freeze_encoder_before_step >= 0
        self.freeze_codebook_after_step = freeze_codebook_after_step
        self.freeze_codebook_before_step = freeze_codebook_before_step
        self.freeze_codebook_during_training = freeze_codebook_during_training
        self.freeze_encoder_before_step = freeze_encoder_before_step

        assert apply_grad_penalty_every >= 0
        assert waveform_grad_penalty_gamma >= 0.
        assert stft_grad_penalty_every >= 0
        assert stft_grad_penalty_gamma >= 0.
        self.apply_grad_penalty_every = apply_grad_penalty_every
        self.waveform_grad_penalty_gamma = waveform_grad_penalty_gamma
        self.stft_grad_penalty_every = stft_grad_penalty_every
        self.stft_grad_penalty_gamma = stft_grad_penalty_gamma

        self.results_folder = Path(results_folder)

        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.accelerator.wait_for_everyone()
        self.results_folder.mkdir(parents = True, exist_ok = True)

        # save tracker hyperparameters

        self.tracker_hps = hyperparameters

        assert self.accelerator.distributed_type != DistributedType.FSDP, 'FSDP not supported for soundstream trainer due to complex-valued stft discriminator'

    @property
    def ema_tokenizer(self):
        return self.ema_soundstream.ema_model

    def tokenize(self, audio):
        return self.ema_tokenizer.tokenize(audio)

    def copy_online_to_ema(self):
        """Synchronize EMA weights and buffers from the current online model."""
        assert self.use_ema
        self.ema_soundstream.ema_model.load_state_dict(
            self.unwrapped_soundstream.state_dict()
        )
        self.sync_ema_runtime_state()

    def sync_ema_runtime_state(self):
        """Copy non-state-dict model controls to the EMA reconstruction path.

        Decoder residual scaling and scheduled loss weights are ordinary Python
        attributes, so EMA parameter updates do not copy them automatically.
        Keeping them synchronized makes EMA validation match the configuration
        stored with an EMA checkpoint.
        """
        if not self.use_ema:
            return

        online_model = self.unwrapped_soundstream
        ema_model = self.ema_soundstream.ema_model

        if (
            hasattr(online_model, 'get_decoder_block_residual_scales') and
            hasattr(ema_model, 'set_decoder_block_residual_scales')
        ):
            ema_model.set_decoder_block_residual_scales(
                online_model.get_decoder_block_residual_scales()
            )
        elif (
            hasattr(online_model, 'decoder_residual_scale') and
            hasattr(ema_model, 'set_decoder_residual_scale')
        ):
            ema_model.set_decoder_residual_scale(
                online_model.decoder_residual_scale
            )

        for attribute in (
            'si_sdr_loss_weight',
            'spectral_envelope_loss_weight',
            'stft_recon_loss_weight',
            'frame_phase_loss_weight',
            'click_loss_weight',
            'jump_loss_weight',
            'preemph_loss_weight',
            'noise_floor_loss_weight'
        ):
            if hasattr(online_model, attribute) and hasattr(ema_model, attribute):
                setattr(ema_model, attribute, getattr(online_model, attribute))

    def save(self, path):
        path = Path(path)
        filename_step = checkpoint_num_steps(path)
        trainer_step = (
            filename_step
            if filename_step >= 0
            else int(self.steps.item())
        )
        pkg = dict(
            model = self.accelerator.get_state_dict(self.soundstream),
            optim = self.optim.state_dict(),
            plateau_scheduler = self.plateau_scheduler.state_dict() if exists(self.plateau_scheduler) else None,
            plateau_lr_unclean_checks = self.plateau_lr_unclean_checks,
            config = self.unwrapped_soundstream._configs,
            discr_optim = self.discr_optim.state_dict(),
            best_valid_score = self.best_valid_score,
            best_aligned_si_sdr = self.best_aligned_si_sdr,
            best_frame_leakage_score = self.best_frame_leakage_score,
            early_stopping_bad_evals = self.early_stopping_bad_evals,
            early_stopping_triggered = self.early_stopping_triggered,
            quality_retention_baseline = self.quality_retention_baseline,
            quality_retention_bad_evals = self.quality_retention_bad_evals,
            quality_retention_rvq_bad_evals = self.quality_retention_rvq_bad_evals,
            trainer_step = trainer_step,
            version = __version__
        )

        if self.use_ema:
            pkg['ema_model'] = self.ema_soundstream.state_dict()

        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            pkg[key] = discr_optim.state_dict()

        torch.save(pkg, str(path))

    def save_model_only(
        self,
        path,
        model,
        *,
        score = None,
        step = None,
        weight_source = None
    ):
        path = Path(path)
        pkg = dict(
            model = model.state_dict(),
            config = self.unwrapped_soundstream._configs,
            version = __version__
        )

        if exists(score):
            pkg['best_score'] = score

        if exists(step):
            pkg['best_step'] = step

        if exists(weight_source):
            pkg['weight_source'] = weight_source

        torch.save(pkg, str(path))

    def saved_model_only_score(self, path):
        path = Path(path)
        if not path.exists():
            return None

        try:
            pkg = torch.load(str(path), map_location = 'cpu')
        except Exception as exc:
            self.print(
                f"warning: could not read checkpoint score from {path}: {exc}"
            )
            return None

        if not isinstance(pkg, dict) or 'best_score' not in pkg:
            return None

        score = pkg['best_score']
        if isinstance(score, torch.Tensor):
            score = score.detach().cpu().item()

        try:
            score = float(score)
        except (TypeError, ValueError):
            return None

        return score if isfinite(score) else None

    def sync_best_scores_from_disk(self):
        """Use existing best checkpoint metadata as the source of truth.

        This prevents a restarted run, or a run launched with --no-resume into a
        non-empty results directory, from overwriting a better checkpoint only
        because the in-memory best score was reset.
        """
        best_selected_score = self.saved_model_only_score(
            self.results_folder / 'best_selected.pt'
        )
        if exists(best_selected_score):
            self.best_valid_score = min(
                self.best_valid_score,
                best_selected_score
            )

        best_aligned_si_sdr = self.saved_model_only_score(
            self.results_folder / 'best_by_aligned_si_sdr.pt'
        )
        if exists(best_aligned_si_sdr):
            self.best_aligned_si_sdr = max(
                self.best_aligned_si_sdr,
                best_aligned_si_sdr
            )

        best_frame_leakage_score = self.saved_model_only_score(
            self.results_folder / 'best_by_frame_leakage.pt'
        )
        if exists(best_frame_leakage_score):
            self.best_frame_leakage_score = min(
                self.best_frame_leakage_score,
                best_frame_leakage_score
            )

    def codebook_metrics(self, model, indices):
        counts = self.codebook_counts(model, indices)
        return self.codebook_metrics_from_counts(counts)

    def codebook_counts(self, model, indices):
        flattened_indices = indices.reshape(-1, indices.shape[-1])
        all_counts = []

        for quantizer_index in range(flattened_indices.shape[-1]):
            codes = flattened_indices[:, quantizer_index]
            codes = codes[codes >= 0]
            if codes.numel() == 0:
                all_counts.append(torch.zeros(
                    model.codebook_size,
                    dtype=torch.float64,
                    device=indices.device
                ))
                continue
            counts = torch.bincount(
                codes.long(),
                minlength = model.codebook_size
            ).to(dtype=torch.float64)
            all_counts.append(counts)

        if not all_counts:
            return torch.zeros(
                0,
                model.codebook_size,
                dtype=torch.float64,
                device=indices.device
            )

        return torch.stack(all_counts)

    def codebook_metrics_from_counts(self, counts):
        active_ratios = []
        perplexities = []
        collapsed_quantizers = 0
        metrics = {}

        for quantizer_index, quantizer_counts in enumerate(counts):
            codebook_size = quantizer_counts.numel()
            if quantizer_counts.sum() <= 0:
                active_ratio = quantizer_counts.new_tensor(0.)
                perplexity = quantizer_counts.new_tensor(0.)
                probabilities = torch.zeros_like(quantizer_counts)
            else:
                probabilities = quantizer_counts / quantizer_counts.sum()
                nonzero = probabilities > 0
                entropy = -(
                    probabilities[nonzero] *
                    probabilities[nonzero].log()
                ).sum()
                active_ratio = (quantizer_counts > 0).float().mean()
                perplexity = entropy.exp()

            active_codes = int((quantizer_counts > 0).sum().detach().cpu())
            dead_codes = codebook_size - active_codes
            normalized_perplexity = perplexity / max(1, codebook_size)
            top_probabilities = probabilities.sort(descending=True).values

            active_ratios.append(active_ratio)
            perplexities.append(perplexity)
            active_ratio_value = float(active_ratio.detach().cpu())
            perplexity_value = float(perplexity.detach().cpu())
            collapsed_quantizers += int(
                active_ratio_value < 0.10 or perplexity_value < 16.
            )
            prefix = f'codebook_q{quantizer_index:02d}'
            metrics[f'{prefix}_active_codes'] = active_codes
            metrics[f'codebook_q{quantizer_index:02d}_active_ratio'] = float(
                active_ratio.detach().cpu()
            )
            metrics[f'codebook_q{quantizer_index:02d}_perplexity'] = float(
                perplexity.detach().cpu()
            )
            metrics[f'{prefix}_normalized_perplexity'] = float(
                normalized_perplexity.detach().cpu()
            )
            metrics[f'{prefix}_dead_codes'] = dead_codes
            metrics[f'{prefix}_top1_prob'] = float(
                top_probabilities[:1].sum().detach().cpu()
            )
            metrics[f'{prefix}_top5_prob'] = float(
                top_probabilities[:5].sum().detach().cpu()
            )
            metrics[f'{prefix}_top10_prob'] = float(
                top_probabilities[:10].sum().detach().cpu()
            )

        metrics.update(
            active_code_ratio = float(
                torch.stack(active_ratios).mean().detach().cpu()
                if active_ratios
                else 0.
            ),
            codebook_perplexity = float(
                torch.stack(perplexities).mean().detach().cpu()
                if perplexities
                else 0.
            ),
            codebook_collapsed_quantizers = collapsed_quantizers
        )
        return metrics

    def format_codebook_diagnostics(self, metrics):
        quantizer_metrics = []
        quantizer_index = 0

        while (
            f'codebook_q{quantizer_index:02d}_active_ratio'
            in metrics
        ):
            active_ratio = metrics[
                f'codebook_q{quantizer_index:02d}_active_ratio'
            ]
            active_codes = int(metrics[
                f'codebook_q{quantizer_index:02d}_active_codes'
            ])
            perplexity = metrics[
                f'codebook_q{quantizer_index:02d}_perplexity'
            ]
            normalized_perplexity = metrics[
                f'codebook_q{quantizer_index:02d}_normalized_perplexity'
            ]
            dead_codes = int(metrics[
                f'codebook_q{quantizer_index:02d}_dead_codes'
            ])
            top1_prob = metrics[
                f'codebook_q{quantizer_index:02d}_top1_prob'
            ]
            top5_prob = metrics[
                f'codebook_q{quantizer_index:02d}_top5_prob'
            ]
            top10_prob = metrics[
                f'codebook_q{quantizer_index:02d}_top10_prob'
            ]
            quantizer_metrics.append(
                (
                    quantizer_index,
                    active_codes,
                    active_ratio,
                    perplexity,
                    normalized_perplexity,
                    dead_codes,
                    top1_prob,
                    top5_prob,
                    top10_prob
                )
            )
            quantizer_index += 1

        warnings = [
            index
            for (
                index, _, active_ratio, _, normalized_perplexity, *_
            ) in quantizer_metrics
            if active_ratio < 0.30 or normalized_perplexity < 0.10
        ]
        collapsed = [
            index
            for (
                index, _, active_ratio, perplexity, _, *_
            ) in quantizer_metrics
            if active_ratio < 0.10 or perplexity < 16.
        ]
        details = " ".join(
            (
                f"q{index:02d}="
                f"u{active_codes},"
                f"a{active_ratio:.2f},"
                f"p{perplexity:.1f},"
                f"np{normalized_perplexity:.3f},"
                f"t1{top1_prob:.3f},"
                f"t5{top5_prob:.3f},"
                f"t10{top10_prob:.3f},"
                f"d{dead_codes}"
            )
            for (
                index,
                active_codes,
                active_ratio,
                perplexity,
                normalized_perplexity,
                dead_codes,
                top1_prob,
                top5_prob,
                top10_prob
            ) in quantizer_metrics
        )
        warning_text = (
            ",".join(f"q{index:02d}" for index in warnings)
            if warnings
            else "none"
        )
        collapsed_text = (
            ",".join(f"q{index:02d}" for index in collapsed)
            if collapsed
            else "none"
        )
        return (
            f"{details} | warning={warning_text} "
            f"| suspected_collapsed={collapsed_text}"
        )

    def score_model_on_wave(self, model, wave):
        model.eval()
        wave = wave.to(next(model.parameters()).device)

        with torch.inference_mode():
            recon = model(wave, return_recons_only = True)
            _, indices, commitment_loss = model(wave, return_encoded = True)

        metrics = self.reconstruction_metrics(model, wave, recon)
        metrics['commitment_loss'] = float(commitment_loss.sum().detach().cpu())
        return metrics, self.codebook_counts(model, indices)

    def evaluate_dataloader_score(self, dataloader, model = None, max_batches = None):
        if not exists(dataloader):
            return None

        model = default(model, self.ema_soundstream.ema_model if self.use_ema else self.unwrapped_soundstream)
        device = self.device
        model = model.to(device)

        totals = Counter()
        total_code_counts = None
        num_batches = 0

        for wave, in dataloader:
            wave = wave.to(device)
            metrics, code_counts = self.score_model_on_wave(model, wave)
            total_code_counts = (
                code_counts
                if total_code_counts is None
                else total_code_counts + code_counts
            )

            for key, value in metrics.items():
                totals[key] += value

            num_batches += 1

            if exists(max_batches) and num_batches >= max_batches:
                break

        if num_batches == 0:
            return None

        averaged_metrics = {
            key: value / num_batches
            for key, value in totals.items()
        }
        averaged_metrics.update(
            self.codebook_metrics_from_counts(total_code_counts)
        )
        return averaged_metrics

    def reconstruction_metrics(self, model, target, recon):
        wave_l1 = F.l1_loss(recon, target)
        wave_mse = F.mse_loss(recon, target)
        target_rms = target.square().mean(dim = -1).clamp_min(1e-8).sqrt()
        recon_rms = recon.square().mean(dim = -1).clamp_min(1e-8).sqrt()
        energy_loss = F.l1_loss(recon_rms, target_rms)
        rms_ratio = (recon_rms / target_rms).mean()
        boundary_loss = target.new_tensor(0.)

        # Validation-only 50 Hz / 320-sample leakage diagnostics.  These use
        # exactly the same full residual for every checkpoint candidate and do
        # not alter the reconstruction passed to the ordinary quality metrics.
        frame_samples = int(getattr(model, 'frame_phase_samples', 320))
        usable_samples = (
            min(target.shape[-1], recon.shape[-1]) // frame_samples
        ) * frame_samples
        frame_diagnostic_valid = target.new_tensor(float(usable_samples >= frame_samples * 4))
        ac_319 = target.new_tensor(0.)
        ac_320 = target.new_tensor(0.)
        ac_321 = target.new_tensor(0.)
        ac_320_isolated = target.new_tensor(0.)
        frame_phase_peak_db = target.new_tensor(0.)
        comb_median_excess_db = target.new_tensor(0.)
        comb_p90_excess_db = target.new_tensor(0.)
        comb_lines_gt_6db = target.new_tensor(0.)
        if frame_diagnostic_valid.item() >= 0.5:
            frame_residual = (
                recon[..., :usable_samples] - target[..., :usable_samples]
            ).float().reshape(-1, usable_samples)
            residual_rms = frame_residual.square().mean(dim = -1).clamp_min(1e-12).sqrt()
            residual_hp = frame_residual[..., 1:] - frame_residual[..., :-1]
            residual_hp = residual_hp - residual_hp.mean(dim = -1, keepdim = True)

            def normalized_ac(lag):
                left = residual_hp[..., :-lag]
                right = residual_hp[..., lag:]
                numerator = (left * right).sum(dim = -1)
                denominator = (
                    left.square().sum(dim = -1) * right.square().sum(dim = -1)
                ).clamp_min(1e-12).sqrt()
                return (numerator / denominator).mean()

            ac_319 = normalized_ac(frame_samples - 1)
            ac_320 = normalized_ac(frame_samples)
            ac_321 = normalized_ac(frame_samples + 1)
            ac_320_isolated = ac_320 - 0.5 * (ac_319 + ac_321)

            phase_pattern = frame_residual.reshape(
                -1, usable_samples // frame_samples, frame_samples
            ).mean(dim = 1)
            phase_pattern = phase_pattern - phase_pattern.mean(dim = -1, keepdim = True)
            phase_peak = phase_pattern.abs().amax(dim = -1)
            frame_phase_peak_db = (
                20. * torch.log10((phase_peak / residual_rms).clamp_min(1e-8))
            ).mean()

            spectrum_db = 20. * torch.log10(
                torch.fft.rfft(frame_residual, dim = -1).abs().clamp_min(1e-8)
            )
            frame_count = usable_samples // frame_samples
            harmonic_bins = torch.arange(
                frame_count,
                spectrum_db.shape[-1] - 1,
                frame_count,
                device = spectrum_db.device
            )
            if harmonic_bins.numel() > 0:
                line_db = spectrum_db[:, harmonic_bins]
                neighbor_db = 0.5 * (
                    spectrum_db[:, harmonic_bins - 1] +
                    spectrum_db[:, harmonic_bins + 1]
                )
                comb_excess = line_db - neighbor_db
                comb_median_excess_db = comb_excess.median(dim = -1).values.mean()
                comb_p90_excess_db = torch.quantile(comb_excess, 0.9, dim = -1).mean()
                comb_lines_gt_6db = (comb_excess > 6.).float().sum(dim = -1).mean()

        target_peak = target.abs().amax(dim = -1).mean()
        recon_peak = recon.abs().amax(dim = -1).mean()
        target_clip_fraction = (target.abs() >= 0.98).float().mean()
        recon_clip_fraction = (recon.abs() >= 0.98).float().mean()

        # Validation-only DC / very-low-frequency diagnostics. Keep target and
        # reconstruction untouched: correlation may center internally, but codec
        # output is never de-meaned or filtered here.
        target_dc = target.mean(dim = -1).mean()
        recon_dc = recon.mean(dim = -1).mean()
        dc_abs_error = (
            target.mean(dim = -1) - recon.mean(dim = -1)
        ).abs().mean()
        low_n_fft = min(1024, target.shape[-1], recon.shape[-1])
        if low_n_fft >= 2:
            target_low_spec = torch.fft.rfft(target.float(), n = low_n_fft, dim = -1)
            recon_low_spec = torch.fft.rfft(recon.float(), n = low_n_fft, dim = -1)
            low_bin_count = min(
                target_low_spec.shape[-1],
                max(1, int(50 * low_n_fft / model.target_sample_hz) + 1)
            )
            target_low_freq_0_50_power = target_low_spec[..., :low_bin_count].abs().square().mean()
            recon_low_freq_0_50_power = recon_low_spec[..., :low_bin_count].abs().square().mean()
            target_low_freq_0_50_db = 10 * torch.log10(target_low_freq_0_50_power.clamp_min(1e-12))
            recon_low_freq_0_50_db = 10 * torch.log10(recon_low_freq_0_50_power.clamp_min(1e-12))
            low_freq_0_50_excess_db = recon_low_freq_0_50_db - target_low_freq_0_50_db
            low_freq_0_50_abs_error_db = low_freq_0_50_excess_db.abs()
        else:
            target_low_freq_0_50_db = target.new_tensor(0.)
            recon_low_freq_0_50_db = target.new_tensor(0.)
            low_freq_0_50_excess_db = target.new_tensor(0.)
            low_freq_0_50_abs_error_db = target.new_tensor(0.)

        if target.shape[-1] > 1 and recon.shape[-1] > 1:
            target_jump = target.diff(dim = -1).abs()
            recon_jump = recon.diff(dim = -1).abs()
            target_max_jump = target_jump.amax(dim = -1).mean()
            recon_max_jump = recon_jump.amax(dim = -1).mean()
            target_p999_jump = torch.quantile(
                target_jump.flatten(),
                0.999
            )
            recon_p999_jump = torch.quantile(
                recon_jump.flatten(),
                0.999
            )
        else:
            target_max_jump = target.new_tensor(0.)
            recon_max_jump = recon.new_tensor(0.)
            target_p999_jump = target.new_tensor(0.)
            recon_p999_jump = recon.new_tensor(0.)

        jump_ratio = recon_max_jump / target_max_jump.clamp_min(1e-8)
        p999_jump_ratio = recon_p999_jump / target_p999_jump.clamp_min(1e-8)
        click_score = (
            recon_max_jump /
            recon_rms.mean().clamp_min(1e-8)
        )

        if hasattr(model, 'frame_boundary_loss'):
            boundary_loss = model.frame_boundary_loss(recon)

        min_mel_samples = max(
            transform.n_fft
            for transform in model.mel_spec_transforms
        )

        if target.shape[-1] < min_mel_samples:
            padding = min_mel_samples - target.shape[-1]
            target = F.pad(target, (0, padding))
            recon = F.pad(recon, (0, padding))

        _, multi_spectral_recon_loss, stft_recon_loss, _ = model.reconstruction_losses(target, recon)
        (
            spectral_envelope_loss,
            spectral_envelope_low,
            spectral_envelope_mid,
            spectral_envelope_high,
            spectral_envelope_voiced_fraction
        ) = model.spectral_envelope_metrics(target, recon)
        _, quiet_hf_excess_db = model.quiet_multiband_noise_metrics(
            target,
            recon
        )

        target_centered = target - target.mean(dim = -1, keepdim = True)
        recon_centered = recon - recon.mean(dim = -1, keepdim = True)
        correlation = (
            (target_centered * recon_centered).sum(dim = -1) /
            (
                target_centered.norm(dim = -1) *
                recon_centered.norm(dim = -1)
            ).clamp_min(1e-8)
        ).mean()

        projection_scale = (
            (recon * target).sum(dim = -1, keepdim = True) /
            target.square().sum(dim = -1, keepdim = True).clamp_min(1e-8)
        )
        projected = projection_scale * target
        residual = recon - projected
        si_sdr = (
            10 * torch.log10(
                projected.square().sum(dim = -1).clamp_min(1e-8) /
                residual.square().sum(dim = -1).clamp_min(1e-8)
            )
        ).mean()

        aligned_correlation, aligned_si_sdr = (
            self.lag_aligned_reconstruction_metrics(
                target,
                recon,
                max_lag_samples = model.target_sample_hz // 25
            )
        )

        wave_loss = (
            wave_l1 +
            0.3 * wave_mse +
            model.energy_loss_weight * energy_loss +
            (
                model.correlation_loss_weight /
                max(float(model.recon_loss_weight), 1e-8)
            ) * (1. - correlation)
        )
        score = (
            multi_spectral_recon_loss *
            model.multi_spectral_recon_loss_weight +
            stft_recon_loss *
            getattr(model, 'stft_recon_loss_weight', 0.) +
            wave_loss * model.recon_loss_weight
        )
        if self.best_checkpoint_metric in (
            'stream_finetune',
            'stream_finetune_long'
        ):
            score = score + boundary_loss * model.boundary_loss_weight

        validation_eligible = (
            (rms_ratio >= 0.25) &
            (rms_ratio <= 4.)
        )
        if self.best_checkpoint_metric == 'overfit':
            validation_eligible = (
                validation_eligible &
                (aligned_correlation >= 0.5) &
                (aligned_si_sdr >= -5.)
            )
        elif self.best_checkpoint_metric in ('recon_pretrain', 'spectral_refine'):
            validation_eligible = (
                validation_eligible &
                (aligned_correlation >= 0.5) &
                (aligned_si_sdr >= -5.)
            )
        elif self.best_checkpoint_metric in (
            'gan_pretrain',
            'stream_finetune',
            'stream_finetune_long'
        ):
            validation_eligible = (
                validation_eligible &
                (aligned_correlation >= 0.5) &
                # These stages start from a recon_pretrain checkpoint whose
                # aligned SI-SDR is around -3.5 dB. Requiring 0 dB made every
                # checkpoint ineligible even when its validation score
                # improved, so no best_selected.pt could be produced.
                (aligned_si_sdr >= -5.)
            )

        return dict(
            score = float(score.detach().cpu()),
            recon_loss = float(wave_l1.detach().cpu()),
            wave_mse = float(wave_mse.detach().cpu()),
            multi_spectral_recon_loss = float(multi_spectral_recon_loss.detach().cpu()),
            stft_recon_loss = float(stft_recon_loss.detach().cpu()),
            commitment_loss = 0.,
            spectral_envelope_loss = float(spectral_envelope_loss.detach().cpu()),
            spectral_envelope_low = float(spectral_envelope_low.detach().cpu()),
            spectral_envelope_mid = float(spectral_envelope_mid.detach().cpu()),
            spectral_envelope_high = float(spectral_envelope_high.detach().cpu()),
            spectral_envelope_voiced_fraction = float(spectral_envelope_voiced_fraction.detach().cpu()),
            quiet_hf_excess_db = float(quiet_hf_excess_db.detach().cpu()),
            boundary_loss = float(boundary_loss.detach().cpu()),
            energy_loss = float(energy_loss.detach().cpu()),
            rms_ratio = float(rms_ratio.detach().cpu()),
            target_peak = float(target_peak.detach().cpu()),
            recon_peak = float(recon_peak.detach().cpu()),
            target_dc = float(target_dc.detach().cpu()),
            recon_dc = float(recon_dc.detach().cpu()),
            dc_abs_error = float(dc_abs_error.detach().cpu()),
            target_low_freq_0_50_db = float(target_low_freq_0_50_db.detach().cpu()),
            recon_low_freq_0_50_db = float(recon_low_freq_0_50_db.detach().cpu()),
            low_freq_0_50_excess_db = float(low_freq_0_50_excess_db.detach().cpu()),
            low_freq_0_50_abs_error_db = float(low_freq_0_50_abs_error_db.detach().cpu()),
            target_clip_fraction = float(target_clip_fraction.detach().cpu()),
            recon_clip_fraction = float(recon_clip_fraction.detach().cpu()),
            target_max_jump = float(target_max_jump.detach().cpu()),
            recon_max_jump = float(recon_max_jump.detach().cpu()),
            target_p999_jump = float(target_p999_jump.detach().cpu()),
            recon_p999_jump = float(recon_p999_jump.detach().cpu()),
            jump_ratio = float(jump_ratio.detach().cpu()),
            p999_jump_ratio = float(p999_jump_ratio.detach().cpu()),
            click_score = float(click_score.detach().cpu()),
            correlation = float(correlation.detach().cpu()),
            si_sdr = float(si_sdr.detach().cpu()),
            aligned_correlation = float(aligned_correlation.detach().cpu()),
            aligned_si_sdr = float(aligned_si_sdr.detach().cpu()),
            frame_diagnostic_valid = float(frame_diagnostic_valid.detach().cpu()),
            ac_319 = float(ac_319.detach().cpu()),
            ac_320 = float(ac_320.detach().cpu()),
            ac_321 = float(ac_321.detach().cpu()),
            ac_320_isolated = float(ac_320_isolated.detach().cpu()),
            frame_phase_peak_db = float(frame_phase_peak_db.detach().cpu()),
            comb_median_excess_db = float(comb_median_excess_db.detach().cpu()),
            comb_p90_excess_db = float(comb_p90_excess_db.detach().cpu()),
            comb_lines_gt_6db = float(comb_lines_gt_6db.detach().cpu()),
            validation_eligible = float(validation_eligible.detach().cpu())
        )

    def lag_aligned_reconstruction_metrics(
        self,
        target,
        recon,
        *,
        max_lag_samples
    ):
        target_rows = target.reshape(-1, target.shape[-1])
        recon_rows = recon.reshape(-1, recon.shape[-1])
        num_samples = min(target_rows.shape[-1], recon_rows.shape[-1])
        max_lag_samples = min(max_lag_samples, num_samples - 1)

        target_rows = target_rows[..., :num_samples]
        recon_rows = recon_rows[..., :num_samples]
        target_centered = target_rows - target_rows.mean(dim = -1, keepdim = True)
        recon_centered = recon_rows - recon_rows.mean(dim = -1, keepdim = True)

        fft_size = 1 << (2 * num_samples - 1).bit_length()
        cross_correlation = torch.fft.irfft(
            torch.fft.rfft(recon_centered, n = fft_size) *
            torch.fft.rfft(target_centered, n = fft_size).conj(),
            n = fft_size
        )
        lag_window = torch.cat((
            cross_correlation[..., -max_lag_samples:],
            cross_correlation[..., :max_lag_samples + 1]
        ), dim = -1)
        best_lags = lag_window.argmax(dim = -1) - max_lag_samples

        correlations = []
        si_sdrs = []
        for row_index, lag_tensor in enumerate(best_lags):
            lag = int(lag_tensor.item())
            target_row = target_rows[row_index]
            recon_row = recon_rows[row_index]

            if lag > 0:
                target_row = target_row[:-lag]
                recon_row = recon_row[lag:]
            elif lag < 0:
                target_row = target_row[-lag:]
                recon_row = recon_row[:lag]

            target_row = target_row - target_row.mean()
            recon_row = recon_row - recon_row.mean()
            correlation = (
                (target_row * recon_row).sum() /
                (target_row.norm() * recon_row.norm()).clamp_min(1e-8)
            )
            projection_scale = (
                (recon_row * target_row).sum() /
                target_row.square().sum().clamp_min(1e-8)
            )
            projected = projection_scale * target_row
            residual = recon_row - projected
            si_sdr = 10 * torch.log10(
                projected.square().sum().clamp_min(1e-8) /
                residual.square().sum().clamp_min(1e-8)
            )
            correlations.append(correlation)
            si_sdrs.append(si_sdr)

        return (
            torch.stack(correlations).mean(),
            torch.stack(si_sdrs).mean()
        )

    def evaluate_full_audio_files(
        self,
        files,
        *,
        model = None,
        block_seconds = 5.,
        context_ms = 60.,
        max_files = None,
        save_recon_dir = None,
        metrics_path = None
    ):
        if not files:
            return None

        model = default(
            model,
            self.ema_soundstream.ema_model if self.use_ema else self.unwrapped_soundstream
        )
        model = model.to(self.device)
        model.eval()
        sample_rate = model.target_sample_hz
        frame_size = model.seq_len_multiple_of
        block_samples = int(round(block_seconds * sample_rate))
        context_samples = int(round(context_ms * sample_rate / 1000.))
        stateful_streaming = (
            hasattr(model, 'encode_frame') and
            hasattr(model, 'decode_frame')
        )

        assert block_samples > 0 and block_samples % frame_size == 0
        assert context_samples >= 0
        assert stateful_streaming or context_samples % frame_size == 0

        totals = Counter()
        total_samples = 0
        total_code_counts = torch.zeros(
            model.num_quantizers * model.rq_groups,
            model.codebook_size,
            dtype=torch.float64,
            device=self.device
        )
        selected_files = list(files)
        save_recon_dir = Path(save_recon_dir) if exists(save_recon_dir) else None
        metrics_path = Path(metrics_path) if exists(metrics_path) else None
        file_metric_rows = []

        if exists(max_files):
            selected_files = selected_files[:max_files]

        with torch.inference_mode():
            for file_index, file in enumerate(selected_files):
                wave, source_sample_rate = load_audio_file(file)

                if wave.shape[0] > 1:
                    wave = wave.mean(dim = 0, keepdim = True)

                if source_sample_rate != sample_rate:
                    wave = torchaudio.functional.resample(
                        wave,
                        source_sample_rate,
                        sample_rate
                    )

                wave = wave.to(self.device)
                encoder_state = None
                decoder_state = None
                file_totals = Counter()
                file_samples = 0
                reconstructed_blocks = []

                for start in range(0, wave.shape[-1], block_samples):
                    current = wave[..., start:(start + block_samples)]
                    valid_samples = current.shape[-1]
                    padded_length = (
                        (valid_samples + frame_size - 1)
                        // frame_size
                        * frame_size
                    )
                    current_padded = F.pad(
                        current,
                        (0, padded_length - valid_samples)
                    )

                    if stateful_streaming:
                        reconstructed_frames = []
                        block_indices = []
                        block_commitment_losses = []

                        for frame in current_padded.unsqueeze(0).split(
                            model.stream_frame_size,
                            dim=-1
                        ):
                            quantized, frame_indices, frame_commitment_loss, encoder_state = model.encode_frame(
                                frame,
                                state=encoder_state,
                                freeze_codebook=True
                            )
                            reconstructed_frame, decoder_state = model.decode_frame(
                                quantized,
                                state=decoder_state
                            )
                            reconstructed_frames.append(reconstructed_frame)
                            block_indices.append(frame_indices)
                            block_commitment_losses.append(
                                frame_commitment_loss.sum()
                            )

                        reconstructed = torch.cat(
                            reconstructed_frames,
                            dim=-1
                        )[..., :valid_samples]
                        indices = torch.cat(block_indices, dim=2)
                        commitment_loss = torch.stack(
                            block_commitment_losses
                        ).mean()
                    else:
                        previous_start = max(0, start - context_samples)
                        context = wave[..., previous_start:start]
                        codec_input = torch.cat(
                            (context, current_padded),
                            dim=-1
                        )
                        # load_audio_file returns [channels, samples].  The
                        # regular SoundStream input packs every leading axis
                        # and restores a channel axis on output, so adding an
                        # extra batch dimension here produced [1, 1, 1, N]
                        # reconstructions during full-file evaluation.
                        reconstructed = model(
                            codec_input,
                            return_recons_only=True
                        )
                        _, indices, commitment_loss = model(
                            codec_input,
                            return_encoded=True
                        )
                        reconstructed = reconstructed[
                            ...,
                            context.shape[-1]:(context.shape[-1] + valid_samples)
                        ]

                    target = current.unsqueeze(0)
                    metrics = self.reconstruction_metrics(
                        model,
                        target,
                        reconstructed
                    )
                    metrics['commitment_loss'] = float(
                        commitment_loss.sum().detach().cpu()
                    )
                    total_code_counts += self.codebook_counts(model, indices)
                    block_reconstructed = reconstructed.detach().cpu()
                    while (
                        block_reconstructed.ndim > 2 and
                        block_reconstructed.shape[0] == 1
                    ):
                        block_reconstructed = block_reconstructed.squeeze(0)
                    if block_reconstructed.ndim == 1:
                        block_reconstructed = block_reconstructed.unsqueeze(0)
                    reconstructed_blocks.append(
                        block_reconstructed[..., :valid_samples]
                    )

                    for key, value in metrics.items():
                        totals[key] += value * valid_samples
                        file_totals[key] += value * valid_samples

                    total_samples += valid_samples
                    file_samples += valid_samples

                reconstructed_path = None
                if exists(save_recon_dir) and reconstructed_blocks:
                    file_path = Path(file)
                    file_id = "__".join(file_path.with_suffix('').parts[-3:])
                    file_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', file_id)
                    reconstructed_path = (
                        save_recon_dir /
                        f"rank{self.accelerator.process_index:02d}_"
                        f"{file_index:05d}_{file_id}.flac"
                    )
                    reconstructed_wave = torch.cat(
                        reconstructed_blocks,
                        dim = -1
                    ).clamp(-1., 1.)
                    save_audio_file(
                        reconstructed_path,
                        reconstructed_wave,
                        sample_rate
                    )

                if file_samples > 0:
                    file_metrics = {
                        key: value / file_samples
                        for key, value in file_totals.items()
                    }
                    file_metric_rows.append((
                        str(file),
                        str(reconstructed_path) if exists(reconstructed_path) else "",
                        file_samples,
                        file_metrics
                    ))

                if (file_index + 1) % 50 == 0:
                    self.print(
                        f'tested {file_index + 1}/{len(selected_files)} full audio files'
                    )

        if exists(metrics_path):
            metrics_path.parent.mkdir(parents = True, exist_ok = True)
            metric_names = sorted({
                key
                for *_, file_metrics in file_metric_rows
                for key in file_metrics
            })
            with metrics_path.open('w', encoding = 'utf-8') as f:
                f.write(
                    "source_file\treconstructed_file\tnum_samples\t" +
                    "\t".join(metric_names) +
                    "\n"
                )
                for source_file, reconstructed_file, file_samples, file_metrics in file_metric_rows:
                    values = [
                        source_file,
                        reconstructed_file,
                        str(file_samples),
                    ] + [
                        f"{file_metrics.get(name, float('nan')):.9f}"
                        for name in metric_names
                    ]
                    f.write("\t".join(values) + "\n")

        if total_samples == 0:
            return None

        averaged_metrics = {
            key: value / total_samples
            for key, value in totals.items()
        }
        averaged_metrics.update(
            self.codebook_metrics_from_counts(total_code_counts)
        )
        averaged_metrics['code_counts'] = total_code_counts.detach().cpu()
        averaged_metrics['num_samples'] = total_samples
        return averaged_metrics

    def dataset_file_at(self, dataset, index):
        if isinstance(dataset, Subset):
            return self.dataset_file_at(dataset.dataset, dataset.indices[index])

        files = getattr(dataset, 'files', None)
        if exists(files) and index < len(files):
            return Path(files[index])

        return None

    def speaker_gender_map(self, files):
        speaker_file = None
        for file in files:
            if not exists(file):
                continue
            for parent in Path(file).parents:
                candidate = parent / 'SPEAKERS.TXT'
                if candidate.exists():
                    speaker_file = candidate
                    break
            if exists(speaker_file):
                break

        if not exists(speaker_file):
            return {}

        genders = {}
        try:
            lines = speaker_file.read_text(
                encoding = 'utf-8',
                errors = 'ignore'
            ).splitlines()
        except OSError:
            return {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith(';') or '|' not in line:
                continue

            parts = [part.strip() for part in line.split('|')]
            if len(parts) < 2:
                continue

            speaker_id, gender = parts[0], parts[1].upper()
            if gender in {'M', 'F'}:
                genders[speaker_id] = gender

        return genders

    def validation_sample_profile(self, index):
        wave = self.valid_dataset[index]
        wave = wave[0] if isinstance(wave, (tuple, list)) else wave
        wave = wave.detach().cpu().float()

        file = self.dataset_file_at(self.valid_dataset, index)
        speaker_id = Path(file).parts[-3] if exists(file) and len(Path(file).parts) >= 3 else ''

        rms = wave.square().mean().clamp_min(1e-8).sqrt().item()
        peak = wave.abs().max().item()
        silence_fraction = (
            wave.abs() < max(0.005, rms * 0.1)
        ).float().mean().item()
        duration = wave.numel() / max(float(self.unwrapped_soundstream.target_sample_hz), 1.)
        original_duration = duration

        if exists(file):
            try:
                info = torchaudio.info(str(file))
                original_duration = info.num_frames / max(float(info.sample_rate), 1.)
            except Exception:
                original_duration = duration

        onset_samples = min(
            wave.numel(),
            max(1, int(0.08 * self.unwrapped_soundstream.target_sample_hz))
        )
        onset = wave[:onset_samples]
        onset_peak = onset.abs().max().item() if onset.numel() > 0 else 0.
        onset_jump = (
            onset.diff().abs().max().item()
            if onset.numel() > 1
            else 0.
        )

        return dict(
            index = index,
            file = str(file) if exists(file) else '',
            speaker_id = speaker_id,
            gender = '',
            wave = wave,
            rms = rms,
            peak = peak,
            silence_fraction = silence_fraction,
            duration = duration,
            original_duration = original_duration,
            onset_peak = onset_peak,
            onset_jump = onset_jump,
            onset_score = onset_peak / max(rms, 1e-8) + onset_jump / max(rms, 1e-8)
        )

    def fixed_result_validation_samples(self):
        if exists(getattr(self, 'fixed_result_valid_samples', None)):
            return self.fixed_result_valid_samples

        max_candidates = min(len(self.valid_dataset), max(128, self.best_eval_batches * 20))
        profiles = [
            self.validation_sample_profile(index)
            for index in range(max_candidates)
        ]
        genders = self.speaker_gender_map(
            [profile['file'] for profile in profiles if profile['file']]
        )
        for profile in profiles:
            profile['gender'] = genders.get(profile['speaker_id'], '')

        selected = []
        selected_indices = set()

        def add_best(label, key, *, reverse = True, predicate = lambda profile: True):
            candidates = [
                profile
                for profile in profiles
                if (
                    profile['index'] not in selected_indices and
                    predicate(profile)
                )
            ]
            if not candidates:
                return

            chosen = sorted(
                candidates,
                key = lambda profile: safe_float(profile.get(key)),
                reverse = reverse
            )[0]
            chosen = dict(chosen)
            chosen['label'] = label
            selected.append(chosen)
            selected_indices.add(chosen['index'])

        add_best('male_loud', 'rms', predicate = lambda profile: profile['gender'] == 'M')
        add_best('male_quiet', 'rms', reverse = False, predicate = lambda profile: profile['gender'] == 'M')
        add_best('female_loud', 'rms', predicate = lambda profile: profile['gender'] == 'F')
        add_best('female_quiet', 'rms', reverse = False, predicate = lambda profile: profile['gender'] == 'F')
        add_best('short_utterance', 'original_duration', reverse = False)
        add_best('long_utterance', 'original_duration')
        add_best('opening_plosive_or_transient', 'onset_score')
        add_best('silence_rich', 'silence_fraction')
        add_best('high_peak', 'peak')

        median_candidates = [
            profile
            for profile in profiles
            if profile['index'] not in selected_indices
        ]
        if median_candidates:
            median_rms = float(np.median([profile['rms'] for profile in profiles]))
            chosen = sorted(
                median_candidates,
                key = lambda profile: abs(profile['rms'] - median_rms)
            )[0]
            chosen = dict(chosen)
            chosen['label'] = 'median_reference'
            selected.append(chosen)
            selected_indices.add(chosen['index'])

        for profile in profiles:
            if len(selected) >= 10:
                break
            if profile['index'] in selected_indices:
                continue
            profile = dict(profile)
            profile['label'] = f'extra_{len(selected):02d}'
            selected.append(profile)
            selected_indices.add(profile['index'])

        selected = selected[:10]
        self.fixed_result_valid_samples = selected

        if self.is_main:
            sample_dir = self.results_folder / 'fixed_validation_samples'
            sample_dir.mkdir(parents = True, exist_ok = True)
            manifest = sample_dir / 'manifest.tsv'
            with manifest.open('w', encoding = 'utf-8') as f:
                f.write(
                    'slot\tlabel\tgender\tspeaker_id\trms\tpeak\t'
                    'silence_fraction\toriginal_duration\tonset_peak\t'
                    'onset_jump\tsource_file\n'
                )
                for slot, profile in enumerate(selected):
                    f.write(
                        f"{slot:02d}\t{profile['label']}\t{profile['gender']}\t"
                        f"{profile['speaker_id']}\t{profile['rms']:.9f}\t"
                        f"{profile['peak']:.9f}\t{profile['silence_fraction']:.9f}\t"
                        f"{profile['original_duration']:.3f}\t"
                        f"{profile['onset_peak']:.9f}\t"
                        f"{profile['onset_jump']:.9f}\t{profile['file']}\n"
                    )

        return selected

    def fixed_result_validation_batch(self):
        samples = self.fixed_result_validation_samples()
        waves = [sample['wave'] for sample in samples]
        min_len = min(wave.shape[-1] for wave in waves)
        waves = [wave[:min_len] for wave in waves]
        return torch.stack(waves), samples

    def fixed_validation_waves(self):
        if exists(self.fixed_best_valid_waves):
            return self.fixed_best_valid_waves

        waves = []

        if exists(self.valid_dataset):
            num_waves = min(self.best_eval_batches, len(self.valid_dataset))

            with torch.random.fork_rng(devices = []):
                torch.manual_seed(self.best_validation_seed)

                for index in range(num_waves):
                    wave = self.valid_dataset[index]
                    wave = wave[0] if isinstance(wave, (tuple, list)) else wave
                    waves.append(wave.detach().cpu().unsqueeze(0))
        else:
            for _ in range(self.best_eval_batches):
                wave, = next(self.valid_dl_iter)
                waves.append(wave.detach().cpu())

        self.fixed_best_valid_waves = waves
        return waves

    def effective_clean_gate_max_click_score(self):
        """Resolve the Stage-2 absolute/relative click gates consistently."""
        allowed = self.clean_gate_max_click_score
        if (
            self.best_checkpoint_metric == 'gan_pretrain' and
            self.quality_retention_gate and
            self.has_quality_retention_baseline
        ):
            allowed = max(
                allowed,
                self.quality_retention_baseline['click_score'] +
                self.quality_retention_max_click_score_rise
            )
        return float(allowed)

    def clean_gate_passes(self, metrics):
        if not self.clean_gate:
            return True

        return (
            metrics.get('aligned_si_sdr', float('-inf')) >= self.clean_gate_min_aligned_si_sdr and
            metrics.get('aligned_correlation', float('-inf')) >= self.clean_gate_min_aligned_corr and
            self.clean_gate_min_rms_ratio <= metrics.get('rms_ratio', float('inf')) <= self.clean_gate_max_rms_ratio and
            metrics.get('recon_peak', float('inf')) <= self.clean_gate_max_recon_peak and
            metrics.get('recon_clip_fraction', float('inf')) <= self.clean_gate_max_recon_clip_fraction and
            metrics.get('click_score', float('inf')) <= self.effective_clean_gate_max_click_score() and
            metrics.get('jump_ratio', float('inf')) <= self.clean_gate_max_jump_ratio and
            metrics.get('p999_jump_ratio', float('inf')) <= self.clean_gate_max_p999_jump_ratio
        )

    def clean_gate_failure_reasons(self, metrics):
        if not self.clean_gate:
            return []

        reasons = []

        if metrics.get('aligned_si_sdr', float('-inf')) < self.clean_gate_min_aligned_si_sdr:
            reasons.append('aligned_si_sdr')

        if metrics.get('aligned_correlation', float('-inf')) < self.clean_gate_min_aligned_corr:
            reasons.append('aligned_corr')

        rms_ratio = metrics.get('rms_ratio', float('inf'))
        if not self.clean_gate_min_rms_ratio <= rms_ratio <= self.clean_gate_max_rms_ratio:
            reasons.append('rms_ratio')

        if metrics.get('recon_peak', float('inf')) > self.clean_gate_max_recon_peak:
            reasons.append('peak')

        if metrics.get('recon_clip_fraction', float('inf')) > self.clean_gate_max_recon_clip_fraction:
            reasons.append('clip')

        if metrics.get('click_score', float('inf')) > self.effective_clean_gate_max_click_score():
            reasons.append('click')

        if metrics.get('jump_ratio', float('inf')) > self.clean_gate_max_jump_ratio:
            reasons.append('jump')

        if metrics.get('p999_jump_ratio', float('inf')) > self.clean_gate_max_p999_jump_ratio:
            reasons.append('p999_jump')

        return reasons

    @property
    def has_quality_retention_baseline(self):
        return isinstance(self.quality_retention_baseline, dict)

    def set_quality_retention_baseline(self, metrics):
        required = (
            'aligned_si_sdr',
            'aligned_correlation',
            'quiet_hf_excess_db',
            'click_score',
        )
        missing = [key for key in required if key not in metrics]
        if missing:
            raise KeyError(f"quality-retention baseline missing metrics: {missing}")
        self.quality_retention_baseline = {
            key: float(metrics[key]) for key in required
        }
        self.quality_retention_bad_evals = 0
        self.quality_retention_rvq_bad_evals = 0

    def quality_retention_failure_reasons(self, metrics):
        if not self.quality_retention_gate or not self.has_quality_retention_baseline:
            return []

        baseline = self.quality_retention_baseline
        reasons = []
        if metrics.get('aligned_si_sdr', float('-inf')) < (
            baseline['aligned_si_sdr'] - self.quality_retention_max_aligned_si_sdr_drop
        ):
            reasons.append('aligned_si_sdr')
        if metrics.get('aligned_correlation', float('-inf')) < (
            baseline['aligned_correlation'] - self.quality_retention_max_aligned_corr_drop
        ):
            reasons.append('aligned_corr')
        if metrics.get('quiet_hf_excess_db', float('inf')) > (
            baseline['quiet_hf_excess_db'] +
            self.quality_retention_max_quiet_hf_excess_db_rise
        ):
            reasons.append('quiet_hf')
        if metrics.get('click_score', float('inf')) > (
            baseline.get('click_score', self.clean_gate_max_click_score) +
            self.quality_retention_max_click_score_rise
        ):
            reasons.append('click')
        if (
            metrics.get('codebook_q00_active_ratio', 0.) <
            self.quality_retention_q00_min_active_ratio or
            metrics.get('codebook_q00_perplexity', 0.) <
            self.quality_retention_q00_min_perplexity
        ):
            reasons.append('q00')
        if metrics.get('q01_validation_eligible', 0.) < 0.5:
            reasons.append('q01')
        return reasons

    def evaluate_fixed_validation_score(self, model):
        totals = Counter()
        total_code_counts = None
        waves = self.fixed_validation_waves()

        for wave in waves:
            metrics, code_counts = self.score_model_on_wave(model, wave)
            total_code_counts = (
                code_counts
                if total_code_counts is None
                else total_code_counts + code_counts
            )

            for key, value in metrics.items():
                totals[key] += value

        averaged_metrics = {
            key: value / len(waves)
            for key, value in totals.items()
        }
        averaged_metrics.update(
            self.codebook_metrics_from_counts(total_code_counts)
        )
        # Eligibility must be evaluated from the final aggregate. Averaging
        # per-wave booleans allowed checkpoints whose reported mean
        # correlation and SI-SDR were both below their required thresholds.
        eligible = (
            0.25 <= averaged_metrics['rms_ratio'] <= 4. and
            averaged_metrics['aligned_correlation'] >= 0.5 and
            averaged_metrics['aligned_si_sdr'] >= -5.
        )
        q00_eligible = (
            averaged_metrics.get('codebook_q00_active_ratio', 0.) >= (
                self.quality_retention_q00_min_active_ratio
                if self.best_checkpoint_metric == 'gan_pretrain' else 0.30
            ) and
            averaged_metrics.get('codebook_q00_perplexity', 0.) >= (
                self.quality_retention_q00_min_perplexity
                if self.best_checkpoint_metric == 'gan_pretrain' else 16.
            )
        )
        averaged_metrics['q00_retention_warning'] = float(
            self.best_checkpoint_metric == 'gan_pretrain' and (
                averaged_metrics.get('codebook_q00_active_ratio', 0.) <
                self.quality_retention_q00_warn_active_ratio or
                averaged_metrics.get('codebook_q00_perplexity', 0.) <
                self.quality_retention_q00_warn_perplexity
            )
        )
        q01_eligible = (
            averaged_metrics.get('codebook_q01_active_ratio', 0.) >= 0.30 and
            averaged_metrics.get('codebook_q01_perplexity', 0.) >= 16.
        )
        rvq_eligible = (
            averaged_metrics.get('active_code_ratio', 0.) >= 0.25 and
            averaged_metrics.get('codebook_perplexity', 0.) >= 20. and
            averaged_metrics.get('codebook_collapsed_quantizers', 0) <= 2
        )
        averaged_metrics['q00_validation_eligible'] = float(q00_eligible)
        averaged_metrics['q01_validation_eligible'] = float(q01_eligible)
        averaged_metrics['rvq_validation_eligible'] = float(rvq_eligible)
        click_allowed = self.effective_clean_gate_max_click_score()
        averaged_metrics['click_allowed'] = click_allowed
        averaged_metrics['click_fail_margin'] = (
            averaged_metrics.get('click_score', float('inf')) - click_allowed
        )
        clean_eligible = self.clean_gate_passes(averaged_metrics)
        averaged_metrics['clean_validation_eligible'] = float(clean_eligible)
        if self.best_checkpoint_metric in ('recon_pretrain', 'spectral_refine', 'gan_pretrain'):
            # Stage-1 depends on a healthy full RVQ stack. Avoid selecting
            # checkpoints whose aggregate score improves after q00 or later
            # residual quantizers have collapsed. Stage-2 starts from that
            # codec and can otherwise select GAN-polished but noisy weights.
            eligible = eligible and q00_eligible and rvq_eligible and clean_eligible
        if self.best_checkpoint_metric in (
            'stream_finetune',
            'stream_finetune_long'
        ):
            # Preserve the healthy stage-2 RVQ utilization during streaming
            # adaptation instead of selecting a low-loss collapsed codebook.
            eligible = (
                eligible and
                averaged_metrics['active_code_ratio'] >= 0.15 and
                averaged_metrics['codebook_perplexity'] >= 24. and
                clean_eligible
            )
        averaged_metrics['validation_eligible'] = float(eligible)
        return averaged_metrics

    @property
    def unwrapped_soundstream(self):
        return self.accelerator.unwrap_model(self.soundstream)

    def create_plateau_scheduler(self):
        if not self.plateau_lr_enabled:
            return None

        return ReduceLROnPlateau(
            self.optim.optimizer,
            mode = 'min' if self.plateau_lr_metric == 'score' else 'max',
            factor = self.plateau_lr_factor,
            patience = self.plateau_lr_patience,
            threshold = self.plateau_lr_threshold,
            threshold_mode = 'abs',
            cooldown = self.plateau_lr_cooldown,
            min_lr = self.plateau_lr_min_lr
        )

    def sync_optimizer_lrs_from_plateau_scheduler(self):
        if not exists(self.plateau_scheduler):
            return

        last_lrs = getattr(self.plateau_scheduler, '_last_lr', None)
        if not last_lrs:
            return

        for group, lr in zip(self.optim.optimizer.param_groups, last_lrs):
            group['lr'] = float(lr)

        self.optim.sync_warmup_lrs_from_optimizer()

    def scale_discriminator_lrs_from_plateau(self, factor):
        """Apply a generator plateau reduction to every Stage-2 discriminator.

        The discriminator has no independent validation target. Scaling it by
        the actual generator LR ratio preserves the intended 1:2 G:D schedule,
        including a final partial reduction when either minimum LR is reached.
        """
        if not self.plateau_lr_update_discriminator:
            return []

        optimizers = [
            ('stft', self.discr_optim),
            *self.multiscale_discriminator_optim_iter(),
        ]
        changes = []
        for name, discriminator_optim in optimizers:
            old_lrs = [
                group['lr']
                for group in discriminator_optim.optimizer.param_groups
            ]
            min_lr = (
                self.plateau_lr_stft_discr_min_lr
                if name == 'stft'
                else self.plateau_lr_discr_min_lr
            )
            for group in discriminator_optim.optimizer.param_groups:
                group['lr'] = max(
                    group['lr'] * factor,
                    min_lr
                )
            discriminator_optim.sync_warmup_lrs_from_optimizer()
            new_lrs = [
                group['lr']
                for group in discriminator_optim.optimizer.param_groups
            ]
            changes.append((name, old_lrs, new_lrs))
        return changes

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        # if loading from old version, make a hacky guess

        if len(pkg.keys()) > 20:
            self.unwrapped_soundstream.load_state_dict(pkg)

            if self.use_ema:
                self.ema_soundstream.ema_model.load_state_dict(pkg)
            return

        # check version

        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')

        # otherwise load things normally

        discriminator_reinitialized = False
        try:
            self.unwrapped_soundstream.load_state_dict(pkg['model'])
        except RuntimeError as full_load_error:
            # A corrected discriminator architecture changes only discriminator
            # tensor shapes.  Verify the generator/RVQ state still matches
            # strictly before allowing a non-GAN stage to resume.
            try:
                skipped_keys = self.unwrapped_soundstream.load_generator_state_dict(
                    pkg['model']
                )
            except RuntimeError:
                raise full_load_error

            if self.enable_gan and not self.allow_discriminator_reinitialization:
                raise RuntimeError(
                    "This checkpoint uses an incompatible discriminator "
                    "architecture. Do not resume the old GAN stage; initialize "
                    "a fresh stage 2 from the stage-1 validation-best checkpoint."
                ) from full_load_error

            discriminator_reinitialized = True
            self.print(
                "Resumed generator/RVQ state with the corrected discriminator "
                f"architecture; reinitialized {len(skipped_keys)} discriminator "
                "state keys."
            )
        if (
            'config' in pkg and
            hasattr(self.unwrapped_soundstream, 'restore_decoder_runtime_state')
        ):
            checkpoint_config = pickle.loads(pkg['config'])
            restored_scales = self.unwrapped_soundstream.restore_decoder_runtime_state(
                checkpoint_config
            )
            self.print(
                "Restored decoder runtime block scales from training checkpoint: "
                f"{restored_scales}"
            )

        if self.use_ema:
            if 'ema_model' in pkg:
                if discriminator_reinitialized:
                    ema_pkg = pkg['ema_model']
                    ema_generator_state = {
                        key[len('ema_model.'):]: value
                        for key, value in ema_pkg.items()
                        if key.startswith('ema_model.')
                    }
                    if ema_generator_state:
                        self.ema_soundstream.ema_model.load_generator_state_dict(
                            ema_generator_state
                        )
                        current_ema_state = self.ema_soundstream.state_dict()
                        compatible_ema_meta = {
                            key: value
                            for key, value in ema_pkg.items()
                            if (
                                not key.startswith('ema_model.') and
                                key in current_ema_state and
                                hasattr(value, 'shape') and
                                current_ema_state[key].shape == value.shape
                            )
                        }
                        self.ema_soundstream.load_state_dict(
                            compatible_ema_meta,
                            strict = False
                        )
                    else:
                        self.copy_online_to_ema()
                else:
                    self.ema_soundstream.load_state_dict(pkg['ema_model'])
                self.sync_ema_runtime_state()
            else:
                self.print(
                    "checkpoint has no EMA state; initializing EMA from the "
                    "loaded online model"
                )
                self.copy_online_to_ema()

        self.optim.load_state_dict(pkg['optim'])
        if exists(self.plateau_scheduler) and exists(pkg.get('plateau_scheduler')):
            self.plateau_scheduler.load_state_dict(pkg['plateau_scheduler'])
            self.sync_optimizer_lrs_from_plateau_scheduler()
        self.plateau_lr_unclean_checks = pkg.get('plateau_lr_unclean_checks', 0)
        if not discriminator_reinitialized:
            self.discr_optim.load_state_dict(pkg['discr_optim'])
        self.best_valid_score = pkg.get('best_valid_score', float('inf'))
        self.best_aligned_si_sdr = pkg.get('best_aligned_si_sdr', float('-inf'))
        self.best_frame_leakage_score = pkg.get('best_frame_leakage_score', float('inf'))
        self.early_stopping_bad_evals = pkg.get('early_stopping_bad_evals', 0)
        self.early_stopping_triggered = pkg.get('early_stopping_triggered', False)
        self.quality_retention_baseline = pkg.get('quality_retention_baseline')
        self.quality_retention_bad_evals = pkg.get('quality_retention_bad_evals', 0)
        self.quality_retention_rvq_bad_evals = pkg.get('quality_retention_rvq_bad_evals', 0)

        if not discriminator_reinitialized:
            for key, _ in self.multiscale_discriminator_iter():
                discr_optim = getattr(self, key)
                discr_optim.load_state_dict(pkg[key])

        # + 1 to start from the next step and avoid overwriting the last checkpoint

        filename_step = checkpoint_num_steps(path)
        completed_step = (
            filename_step
            if filename_step >= 0
            else pkg.get('trainer_step', -1)
        )
        self.steps = torch.tensor([completed_step + 1], device=self.device)

    def multiscale_discriminator_iter(self):
        for ind, discr in enumerate(self.unwrapped_soundstream.discriminators):
            yield f'multiscale_discr_optimizer_{ind}', discr

    def multiscale_discriminator_optim_iter(self):
        for name, _ in self.multiscale_discriminator_iter():
            yield name, getattr(self, name)

    def print(self, msg):
        self.accelerator.print(msg)

    def log(self, **logs_as_kwargs):
        self.accelerator.log(logs_as_kwargs, step = self.steps.item())

    @contextmanager
    def wandb_tracker(self, project, run = None, hps = None):
        assert self.use_wandb_tracking, '`use_wandb_tracking` must be set to True on SoundStreamTrainer'

        hps = default(hps, self.tracker_hps)

        self.accelerator.init_trackers(project, config = None)

        if exists(run):
            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), self.accelerator.trackers)
            assert exists(wandb_tracker)

            wandb_tracker.run.name = run

        yield

        self.accelerator.end_training()

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def update_gan_weights(self, steps):
        model = self.unwrapped_soundstream
        if not self.enable_gan:
            model.adversarial_loss_weight = 0.
            model.feature_loss_weight = 0.
            return 0.

        if steps < self.gan_start_step:
            model.adversarial_loss_weight = 0.
            model.feature_loss_weight = 0.
            return 0.

        progress = 1.
        if self.gan_ramp_steps > 0:
            progress = min(
                1.,
                max(
                    0.,
                    (steps - self.gan_start_step + 1) / self.gan_ramp_steps
                )
            )

        model.adversarial_loss_weight = self.gan_adversarial_max * progress
        model.feature_loss_weight = self.gan_feature_max * progress
        return progress

    def update_si_sdr_loss_weight(self, steps):
        model = self.unwrapped_soundstream
        if self.si_sdr_loss_max_weight <= 0:
            model.si_sdr_loss_weight = 0.
            return 0.

        if steps < self.si_sdr_loss_start_steps:
            model.si_sdr_loss_weight = 0.
            return 0.

        if self.si_sdr_loss_warmup_steps <= 0:
            progress = 1.
        else:
            progress = min(
                1.,
                max(
                    0.,
                    (steps - self.si_sdr_loss_start_steps) /
                    self.si_sdr_loss_warmup_steps
                )
            )

        model.si_sdr_loss_weight = self.si_sdr_loss_max_weight * progress
        return progress

    def update_spectral_envelope_loss_weight(self, steps):
        model = self.unwrapped_soundstream
        if self.spectral_envelope_loss_max_weight <= 0:
            model.spectral_envelope_loss_weight = 0.
            return 0.

        if steps < self.spectral_envelope_loss_start_steps:
            model.spectral_envelope_loss_weight = 0.
            return 0.

        if self.spectral_envelope_loss_warmup_steps <= 0:
            progress = 1.
        else:
            progress = min(
                1.,
                max(
                    0.,
                    (steps - self.spectral_envelope_loss_start_steps) /
                    self.spectral_envelope_loss_warmup_steps
                )
            )

        model.spectral_envelope_loss_weight = self.spectral_envelope_loss_max_weight * progress
        return progress

    @staticmethod
    def scheduled_loss_weight(steps, start_steps, warmup_steps, max_weight):
        if max_weight <= 0 or steps < start_steps:
            return 0., 0.
        progress = 1. if warmup_steps <= 0 else min(
            1., max(0., (steps - start_steps) / warmup_steps)
        )
        return max_weight * progress, progress

    def update_spectral_refinement_weights(self, steps):
        model = self.unwrapped_soundstream
        stft_weight, stft_progress = self.scheduled_loss_weight(
            steps,
            self.stft_recon_loss_start_steps,
            self.stft_recon_loss_warmup_steps,
            self.stft_recon_loss_max_weight
        )
        frame_weight, frame_progress = self.scheduled_loss_weight(
            steps,
            self.frame_phase_loss_start_steps,
            self.frame_phase_loss_warmup_steps,
            self.frame_phase_loss_max_weight
        )
        model.stft_recon_loss_weight = stft_weight
        model.frame_phase_loss_weight = frame_weight
        return stft_progress, frame_progress

    def update_stage2_reconstruction_transition(self, steps):
        """Bridge Stage-1 reconstruction constraints into the final GAN loss.

        Direct Stage-1 -> Stage-2 training should not abruptly remove
        correlation and voiced-envelope constraints while the GAN objective is
        being introduced.  Before the transition window this restores the
        Stage-1 values; across the window it linearly reaches the configured
        Stage-2 endpoint values already stored on the model.
        """
        if not exists(self.stage2_recon_transition_start_steps):
            return None

        model = self.unwrapped_soundstream
        start = self.stage2_recon_transition_start_steps
        end = self.stage2_recon_transition_end_steps
        if steps < start:
            progress = 0.
        elif end == start or steps >= end:
            progress = 1.
        else:
            progress = (steps - start) / (end - start)

        def interpolate(initial, final):
            return initial + (final - initial) * progress

        model.si_sdr_loss_weight = interpolate(
            self.stage2_initial_si_sdr_loss_weight,
            self.si_sdr_loss_max_weight,
        )
        model.correlation_loss_weight = interpolate(
            self.stage2_initial_correlation_loss_weight,
            self.stage2_final_correlation_loss_weight,
        )
        model.spectral_envelope_loss_weight = interpolate(
            self.stage2_initial_spectral_envelope_loss_weight,
            self.spectral_envelope_loss_max_weight,
        )
        model.noise_floor_loss_weight = interpolate(
            self.stage2_initial_noise_floor_loss_weight,
            self.stage2_final_noise_floor_loss_weight,
        )
        return progress

    def update_transient_loss_weights(self, steps):
        model = self.unwrapped_soundstream
        if (
            self.click_loss_max_weight <= 0 and
            self.jump_loss_max_weight <= 0
        ):
            model.click_loss_weight = 0.
            model.jump_loss_weight = 0.
            return 0.

        if self.transient_loss_warmup_steps <= 0:
            progress = 1.
        else:
            progress = min(
                1.,
                max(0., steps / self.transient_loss_warmup_steps)
            )

        model.click_loss_weight = self.click_loss_max_weight * progress
        model.jump_loss_weight = self.jump_loss_max_weight * progress
        return progress

    def update_decoder_residual_scale(self, steps):
        model = self.unwrapped_soundstream

        if not hasattr(model, 'set_decoder_residual_scale'):
            return 1.

        if exists(self.decoder_x8_residual_scale_target):
            if not hasattr(model, 'set_decoder_block_residual_scales'):
                raise RuntimeError('per-block decoder residual scaling is not supported by this model')
            progress = 1. if self.decoder_x8_residual_scale_ramp_steps <= 0 else min(
                1., max(0., steps / self.decoder_x8_residual_scale_ramp_steps)
            )
            x8_scale = self.decoder_x8_residual_scale_start + (
                self.decoder_x8_residual_scale_target - self.decoder_x8_residual_scale_start
            ) * progress
            block_scales = list(model.get_decoder_block_residual_scales())
            block_scales[0] = x8_scale
            model.set_decoder_block_residual_scales(block_scales)
            return x8_scale

        start_scale = self.decoder_residual_scale_start
        end_scale = self.decoder_residual_scale_end
        warmup_start = self.decoder_residual_scale_warmup_start_steps
        warmup_end = self.decoder_residual_scale_warmup_end_steps

        if steps < warmup_start:
            scale = start_scale
        elif warmup_end == warmup_start:
            scale = end_scale
        elif steps >= warmup_end:
            scale = end_scale
        else:
            progress = (steps - warmup_start) / (warmup_end - warmup_start)
            scale = start_scale + (end_scale - start_scale) * progress

        model.set_decoder_residual_scale(scale)
        return scale

    def cap_generator_lr_for_retention(self, steps):
        """Cap early Stage-2 generator updates without disturbing warmup/plateau.

        The cap is applied immediately before the optimizer update.  It leaves
        the standard initial warmup intact when it is already below the cap,
        and it ends cleanly before GAN/discriminator training begins.
        """
        current_lr = self.optim.optimizer.param_groups[0]['lr']
        if (
            exists(self.generator_hold_lr) and
            self.generator_hold_steps > 0 and
            steps < self.generator_hold_steps
        ):
            applied_lr = min(current_lr, self.generator_hold_lr)
            for group in self.optim.optimizer.param_groups:
                group['lr'] = min(group['lr'], self.generator_hold_lr)
            return applied_lr

        if (
            exists(self.generator_hold_lr) and
            self.generator_hold_steps > 0 and
            steps == self.generator_hold_steps
        ):
            for group, base_lr in zip(
                self.optim.optimizer.param_groups,
                self.generator_hold_base_lrs,
            ):
                group['lr'] = base_lr
            self.optim.sync_warmup_lrs_from_optimizer()
            return self.generator_hold_base_lrs[0]

        return current_lr

    def cap_discriminator_lr_for_retention(self, steps):
        """Keep every Stage-2 discriminator conservative during GAN ramp-up.

        The discriminator still trains from step zero, but its learning rate is
        capped until the generator has spent the full retention/ramp window
        adapting to adversarial gradients.  The cap covers both the STFT and
        waveform-scale discriminators and is released to their configured base
        rate at one shared step.
        """
        optimizers = [('stft', self.discr_optim), *self.multiscale_discriminator_optim_iter()]
        current_lr = self.discr_optim.optimizer.param_groups[0]['lr']

        if (
            exists(self.discriminator_hold_lr) and
            self.discriminator_hold_steps > 0 and
            steps < self.discriminator_hold_steps
        ):
            # Do not call ``sync_warmup_lrs_from_optimizer`` here.  During the
            # regular warmup, the optimizer LR is deliberately a dampened
            # value, whereas ``warmup.lrs`` must retain the undampened base LR.
            # Synchronizing it every step makes the warmup factor compound
            # multiplicatively (for example 1e-5 -> 4e-11 -> ...), effectively
            # freezing every discriminator before the retention window ends.
            applied_lr = min(current_lr, self.discriminator_hold_lr)
            for _, optimizer in optimizers:
                for group in optimizer.optimizer.param_groups:
                    group['lr'] = min(group['lr'], self.discriminator_hold_lr)
            return applied_lr

        if (
            exists(self.discriminator_hold_lr) and
            self.discriminator_hold_steps > 0 and
            steps == self.discriminator_hold_steps
        ):
            for name, optimizer in optimizers:
                for group, base_lr in zip(
                    optimizer.optimizer.param_groups,
                    self.discriminator_hold_base_lrs[name],
                ):
                    group['lr'] = base_lr
                optimizer.sync_warmup_lrs_from_optimizer()
            return self.discriminator_hold_base_lrs['stft'][0]

        return current_lr

    def update_discriminators(self, device, steps, logs):
        apply_waveform_grad_penalty = (
            self.waveform_grad_penalty_gamma > 0. and
            self.apply_grad_penalty_every > 0 and
            not (steps % self.apply_grad_penalty_every)
        )
        apply_stft_grad_penalty = (
            self.stft_grad_penalty_gamma > 0. and
            self.stft_grad_penalty_every > 0 and
            not (steps % self.stft_grad_penalty_every)
        )
        self.discr_optim.zero_grad()

        for _, multiscale_discr_optim in self.multiscale_discriminator_optim_iter():
            multiscale_discr_optim.zero_grad()

        for i in range(self.grad_accum_every):
            is_last = i == (self.grad_accum_every - 1)
            context = (
                partial(self.accelerator.no_sync, self.soundstream)
                if not is_last
                else nullcontext
            )
            wave, = next(self.dl_iter)
            wave = wave.to(device)

            with self.accelerator.autocast(), context():
                discr_losses = self.soundstream(
                    wave,
                    apply_grad_penalty = apply_waveform_grad_penalty,
                    apply_stft_grad_penalty = apply_stft_grad_penalty,
                    waveform_grad_penalty_gamma = self.waveform_grad_penalty_gamma,
                    stft_grad_penalty_gamma = self.stft_grad_penalty_gamma,
                    return_discr_loss = True,
                    return_discr_losses_separately = True,
                    freeze_codebook = True
                )
                diagnostic_names = {
                    'stft',
                    'stft_real_logits_mean',
                    'stft_fake_logits_mean',
                    'stft_saturated',
                    'stft_r1_raw_mean',
                    'stft_r1_raw_sum',
                    'stft_r1_weighted',
                }
                optimization_losses = []
                for name, discr_loss in discr_losses:
                    if not torch.isfinite(discr_loss).all():
                        raise RuntimeError(
                            f"non-finite discriminator loss at step {steps}: "
                            f"{name}={discr_loss.detach().float().cpu().item()}"
                        )
                    if name not in diagnostic_names:
                        optimization_losses.append(discr_loss)
                    accum_log(
                        logs,
                        {name: discr_loss.item() / self.grad_accum_every}
                    )
                if not optimization_losses:
                    raise RuntimeError('no discriminator optimization losses were returned')

                # The STFT branch already packages hinge + R1 as stft_total.
                # Backpropagate the complete D objective once; never backward
                # the R1 diagnostic as a second graph branch.
                total_discriminator_loss = torch.stack(optimization_losses).sum()
                self.accelerator.backward(
                    total_discriminator_loss / self.grad_accum_every
                )

                discr_loss_map = dict(discr_losses)
                if 'stft_r1_weighted' in discr_loss_map:
                    weighted = float(
                        discr_loss_map['stft_r1_weighted'].float().cpu().item()
                    )
                    raw_mean = float(
                        discr_loss_map['stft_r1_raw_mean'].float().cpu().item()
                    )
                    stft_hinge = float(discr_loss_map['stft'].float().cpu().item())
                    ratio = weighted / max(stft_hinge, 0.1)
                    accum_log(logs, {
                        'stft_r1_gamma': self.stft_grad_penalty_gamma / self.grad_accum_every,
                        'stft_r1_effective_per_step': (
                            weighted /
                            max(self.stft_grad_penalty_every, 1) /
                            self.grad_accum_every
                        ),
                        'stft_r1_to_hinge_ratio': ratio / self.grad_accum_every,
                    })
                    if weighted > 1.:
                        self.print(
                            f"{steps}: warning: normalized STFT R1 weighted "
                            f"penalty is unusually high ({weighted:.6f} > 1.0; "
                            f"raw_mean={raw_mean:.6f}, "
                            f"gamma={self.stft_grad_penalty_gamma:.3e})"
                        )

        if 'stft_saturated' in logs:
            self.stft_saturation_history.append(float(logs['stft_saturated']))
            logs['stft_saturated_fraction_1000'] = (
                sum(self.stft_saturation_history) /
                len(self.stft_saturation_history)
            )

        if exists(self.discr_max_grad_norm):
            model = self.unwrapped_soundstream
            waveform_grad_norm = self.accelerator.clip_grad_norm_(
                model.discriminators.parameters(),
                self.discr_max_grad_norm
            )
            stft_grad_norm = self.accelerator.clip_grad_norm_(
                model.stft_discriminator.parameters(),
                self.discr_max_grad_norm
            )
            accum_log(logs, {
                'waveform_d_grad_norm_pre_clip': float(waveform_grad_norm),
                'stft_d_grad_norm_pre_clip': float(stft_grad_norm),
                'waveform_d_grad_norm_post_clip_upper_bound': min(
                    float(waveform_grad_norm), self.discr_max_grad_norm
                ),
                'stft_d_grad_norm_post_clip_upper_bound': min(
                    float(stft_grad_norm), self.discr_max_grad_norm
                ),
            })

        self.ensure_finite_optimizer_lrs('stft discriminator', self.discr_optim)
        self.discr_optim.step()
        for name, multiscale_discr_optim in self.multiscale_discriminator_optim_iter():
            self.ensure_finite_optimizer_lrs(name, multiscale_discr_optim)
            multiscale_discr_optim.step()

    @contextmanager
    def frozen_discriminators(self):
        discriminators = [
            self.unwrapped_soundstream.stft_discriminator,
            *self.unwrapped_soundstream.discriminators,
        ]
        parameters = [
            parameter
            for discriminator in discriminators
            for parameter in discriminator.parameters()
        ]
        original_requires_grad = [
            parameter.requires_grad
            for parameter in parameters
        ]
        for parameter in parameters:
            parameter.requires_grad_(False)

        try:
            yield
        finally:
            for parameter, requires_grad in zip(
                parameters,
                original_requires_grad
            ):
                parameter.requires_grad_(requires_grad)

    @contextmanager
    def frozen_encoder_and_rvq(self):
        """Temporarily train only the decoder side of the generator."""
        model = self.unwrapped_soundstream
        modules = [
            getattr(model, name, None)
            for name in ('encoder', 'encoder_attn', 'encoder_film', 'rq')
        ]
        parameters = [
            parameter
            for module in modules
            if isinstance(module, nn.Module)
            for parameter in module.parameters()
        ]
        original_requires_grad = [parameter.requires_grad for parameter in parameters]
        for parameter in parameters:
            parameter.requires_grad_(False)

        try:
            yield
        finally:
            for parameter, requires_grad in zip(parameters, original_requires_grad):
                parameter.requires_grad_(requires_grad)

    @staticmethod
    def ensure_finite_optimizer_lrs(name, optimizer):
        lrs = [float(group['lr']) for group in optimizer.optimizer.param_groups]
        if not all(isfinite(lr) and lr >= 0. for lr in lrs):
            raise RuntimeError(f"{name} has invalid learning rate(s): {lrs}")
        return lrs

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())
        log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)
        gan_progress = self.update_gan_weights(steps)
        si_sdr_loss_progress = self.update_si_sdr_loss_weight(steps)
        spectral_envelope_loss_progress = self.update_spectral_envelope_loss_weight(steps)
        stft_loss_progress, frame_phase_loss_progress = self.update_spectral_refinement_weights(steps)
        stage2_recon_transition_progress = self.update_stage2_reconstruction_transition(steps)
        transient_loss_progress = self.update_transient_loss_weights(steps)
        freeze_codebook = (
            self.freeze_codebook_during_training or
            (exists(self.freeze_codebook_before_step) and steps < self.freeze_codebook_before_step) or
            (exists(self.freeze_codebook_after_step) and steps >= self.freeze_codebook_after_step)
        )
        freeze_encoder = (
            exists(self.freeze_encoder_before_step) and
            steps < self.freeze_encoder_before_step
        )
        decoder_residual_scale = self.update_decoder_residual_scale(steps)
        self.sync_ema_runtime_state()

        self.soundstream.train()

        # logs

        logs = {}

        # update vae (generator)

        for i in range(self.grad_accum_every):
            is_last = i == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.soundstream) if not is_last else nullcontext

            wave, = next(self.dl_iter)
            wave = wave.to(device)

            freeze_discriminators = (
                self.frozen_discriminators
                if gan_progress > 0
                else nullcontext
            )
            freeze_encoder_and_rvq = (
                self.frozen_encoder_and_rvq
                if freeze_encoder
                else nullcontext
            )
            with (
                self.accelerator.autocast(),
                context(),
                freeze_discriminators(),
                freeze_encoder_and_rvq()
            ):
                loss, (
                    recon_loss,
                    multi_spectral_recon_loss,
                    stft_recon_loss,
                    spectral_envelope_loss,
                    adversarial_loss,
                    feature_loss,
                    all_commitment_loss,
                    si_sdr_loss,
                    correlation_loss,
                    click_loss,
                    jump_loss,
                    preemph_loss,
                    noise_floor_loss,
                    frame_phase_or_boundary_loss
                ) = self.soundstream(
                    wave,
                    return_loss_breakdown = True,
                    freeze_codebook = freeze_codebook
                )

                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        f"non-finite generator loss at step {steps}: "
                        f"{loss.detach().float().cpu().item()}"
                    )

                self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, dict(
                loss = loss.item() / self.grad_accum_every,
                recon_loss = recon_loss.item() / self.grad_accum_every,
            ))

            accum_log(logs, dict(
                multi_spectral_recon_loss = multi_spectral_recon_loss.item() / self.grad_accum_every,
                stft_recon_loss = stft_recon_loss.item() / self.grad_accum_every,
                spectral_envelope_loss = spectral_envelope_loss.item() / self.grad_accum_every,
                adversarial_loss = adversarial_loss.item() / self.grad_accum_every,
                feature_loss = feature_loss.item() / self.grad_accum_every,
                all_commitment_loss = all_commitment_loss.item() / self.grad_accum_every,
                si_sdr_loss = si_sdr_loss.item() / self.grad_accum_every,
                correlation_loss = correlation_loss.item() / self.grad_accum_every,
                click_loss = click_loss.item() / self.grad_accum_every,
                jump_loss = jump_loss.item() / self.grad_accum_every,
                preemph_loss = preemph_loss.item() / self.grad_accum_every,
                noise_floor_loss = noise_floor_loss.item() / self.grad_accum_every,
                frame_phase_loss = (
                    frame_phase_or_boundary_loss.item() / self.grad_accum_every
                    if not hasattr(self.unwrapped_soundstream, 'frame_boundary_loss') else 0.
                ),
                boundary_loss = (
                    frame_phase_or_boundary_loss.item() / self.grad_accum_every
                    if hasattr(self.unwrapped_soundstream, 'frame_boundary_loss') else 0.
                ),
            ))

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.soundstream.parameters(), self.max_grad_norm)

        generator_update_lr = self.cap_generator_lr_for_retention(steps)
        self.ensure_finite_optimizer_lrs('generator', self.optim)
        self.optim.step()
        self.optim.zero_grad()

        if self.enable_gan and gan_progress > 0:
            discriminator_update_lr = self.cap_discriminator_lr_for_retention(steps)
            self.update_discriminators(device, steps, logs)
        else:
            discriminator_update_lr = self.discr_optim.optimizer.param_groups[0]['lr']

        # build pretty printed losses

        model = self.unwrapped_soundstream
        generator_lr = self.optim.optimizer.param_groups[0]['lr']
        stft_discriminator_lr = self.discr_optim.optimizer.param_groups[0]['lr']
        waveform_discriminator_lrs = {
            name: optimizer.optimizer.param_groups[0]['lr']
            for name, optimizer in self.multiscale_discriminator_optim_iter()
        }
        waveform_discriminator_lr = next(
            iter(waveform_discriminator_lrs.values()),
            stft_discriminator_lr
        )
        logs['lr_g'] = generator_lr
        logs['lr_d_stft'] = stft_discriminator_lr
        for index, lr in enumerate(waveform_discriminator_lrs.values()):
            logs[f'lr_d_wave_{index}'] = lr
        losses_str = (
            f"{steps}: "
            f"g_total={logs['loss']:.6f} | "
            f"wave={logs['recon_loss']:.6f}(w={model.recon_loss_weight:.4g}) | "
            f"mel={logs['multi_spectral_recon_loss']:.6f}"
            f"(w={model.multi_spectral_recon_loss_weight:.4g}) | "
            f"stft={logs['stft_recon_loss']:.6f}"
            f"(w={model.stft_recon_loss_weight:.4g},ramp={stft_loss_progress:.4f}) | "
            f"frame_phase={logs['frame_phase_loss']:.6f}"
            f"(w={getattr(model, 'frame_phase_loss_weight', 0.):.4g},ramp={frame_phase_loss_progress:.4f}) | "
            f"formant_env={logs['spectral_envelope_loss']:.6f}"
            f"(w={model.spectral_envelope_loss_weight:.4g},ramp={spectral_envelope_loss_progress:.4f}) | "
            f"corr_loss={logs['correlation_loss']:.6f}"
            f"(w={model.correlation_loss_weight:.4g}) | "
            f"si_sdr_loss={logs['si_sdr_loss']:.6f}"
            f"(w={model.si_sdr_loss_weight:.4g},ramp={si_sdr_loss_progress:.4f}) | "
            f"click_loss={logs['click_loss']:.6f}"
            f"(w={model.click_loss_weight:.4g},ramp={transient_loss_progress:.4f}) | "
            f"jump_loss={logs['jump_loss']:.6f}"
            f"(w={model.jump_loss_weight:.4g},ramp={transient_loss_progress:.4f}) | "
            f"preemph_loss={logs['preemph_loss']:.6f}"
            f"(w={model.preemph_loss_weight:.4g}) | "
            f"noise_floor_loss={logs['noise_floor_loss']:.6f}"
            f"(w={model.noise_floor_loss_weight:.4g}) | "
            f"decoder_res_scale={decoder_residual_scale:.4f} | "
            f"rvq_frozen={int(freeze_codebook)} | "
            f"encoder_frozen={int(freeze_encoder)} | "
            f"commit={logs['all_commitment_loss']:.6f}"
        )

        if hasattr(model, 'get_decoder_block_residual_scales'):
            block_scales = ','.join(
                f'{scale:.3f}' for scale in model.get_decoder_block_residual_scales()
            )
            losses_str += f" | decoder_block_scales=[{block_scales}]"

        if hasattr(model, 'boundary_loss_weight'):
            losses_str += (
                f" | boundary={logs['boundary_loss']:.6f}"
                f"(w={model.boundary_loss_weight:.4g})"
            )

        losses_str += (
            f" | adv={logs['adversarial_loss']:.6f}"
            f"(w={model.adversarial_loss_weight:.4g}) | "
            f"feature={logs['feature_loss']:.6f}"
            f"(w={model.feature_loss_weight:.4g}) | "
            f"gan_ramp={gan_progress:.4f} | "
            f"lr_g_update={generator_update_lr:.3e} | "
            f"lr_g_next={generator_lr:.3e} | "
            f"lr_d_stft_update={discriminator_update_lr:.3e} | "
            f"lr_d_stft_next={stft_discriminator_lr:.3e} | "
            f"lr_d_wave_next={waveform_discriminator_lr:.3e}"
        )

        if exists(stage2_recon_transition_progress):
            losses_str += (
                f" | stage2_recon_transition={stage2_recon_transition_progress:.4f}"
            )

        if log_losses:
            self.log(**logs)

        discriminator_log_names = {
            'stft': 'd_stft',
            'stft_total': 'd_stft_total',
            'stft_real_logits_mean': 'd_stft_real_mean',
            'stft_fake_logits_mean': 'd_stft_fake_mean',
            'stft_saturated': 'd_stft_saturated',
            'stft_saturated_fraction_1000': 'd_stft_saturated_fraction_1000',
            'stft_r1_raw_mean': 'stft_r1_raw_mean',
            'stft_r1_raw_sum': 'stft_r1_raw_sum',
            'stft_r1_gamma': 'stft_r1_gamma',
            'stft_r1_weighted': 'stft_r1_weighted',
            'stft_r1_effective_per_step': 'stft_r1_effective_per_step',
            'stft_r1_to_hinge_ratio': 'stft_r1_to_hinge_ratio',
            'waveform_d_grad_norm_pre_clip': 'waveform_d_grad_norm_pre_clip',
            'stft_d_grad_norm_pre_clip': 'stft_d_grad_norm_pre_clip',
            'waveform_d_grad_norm_post_clip_upper_bound': 'waveform_d_grad_norm_post_clip_upper_bound',
            'stft_d_grad_norm_post_clip_upper_bound': 'stft_d_grad_norm_post_clip_upper_bound',
        }
        for key, value in logs.items():
            if key.startswith('scale:'):
                log_name = f"d_scale_{key.split(':', 1)[1]}"
            elif key.startswith('scale_grad_penalty:'):
                log_name = f"gp_scale_{key.split(':', 1)[1]}"
            elif key in discriminator_log_names:
                log_name = discriminator_log_names[key]
            else:
                continue

            losses_str += f" | {log_name}={value:.6f}"

            if log_losses:
                self.log(**{log_name: value})

        # log

        self.print(losses_str)

        # update exponential moving averaged generator

        self.accelerator.wait_for_everyone()

        if self.use_ema:
            self.ema_soundstream.update()
            self.sync_ema_runtime_state()

        # sample results every so often

        self.accelerator.wait_for_everyone()

        if not (steps % self.save_results_every):
            models = [(self.unwrapped_soundstream, str(steps))]
            if self.use_ema:
                models.append((self.ema_soundstream.ema_model if self.use_ema else self.unwrapped_soundstream, f'{steps}.ema'))

            sample_infos = None
            if self.best_checkpoint_metric == 'recon_pretrain':
                wave, sample_infos = self.fixed_result_validation_batch()
            else:
                wave, = next(self.valid_dl_iter)
            wave = wave.to(device)

            for model, label in models:
                model.eval()
                model = model.to(device)

                with torch.inference_mode():
                    recons = model(wave, return_recons_only = True)

                if self.is_main:
                    fixed_sample_dir = self.results_folder / 'fixed_validation_samples'
                    for ind, recon in enumerate(recons.unbind(dim = 0)):
                        if exists(sample_infos):
                            sample_info = sample_infos[ind]
                            sample_label = re.sub(
                                r'[^A-Za-z0-9_.-]+',
                                '_',
                                sample_info['label']
                            )
                            source_filename = (
                                fixed_sample_dir /
                                f"input_{ind:02d}_{sample_label}.flac"
                            )
                            if not source_filename.exists():
                                source_wave = wave[ind].detach().cpu()
                                if source_wave.ndim == 1:
                                    source_wave = source_wave.unsqueeze(0)
                                save_audio_file(
                                    source_filename,
                                    source_wave,
                                    self.unwrapped_soundstream.target_sample_hz
                                )

                            filename = (
                                fixed_sample_dir /
                                f"recon_{label}_{ind:02d}_{sample_label}.flac"
                            )
                            recon_wave = recon.cpu().detach()
                            if recon_wave.ndim == 1:
                                recon_wave = recon_wave.unsqueeze(0)
                            save_audio_file(
                                filename,
                                recon_wave,
                                self.unwrapped_soundstream.target_sample_hz
                            )
                        else:
                            filename = str(
                                self.results_folder /
                                f'sample_{label}_{ind:02d}.flac'
                            )
                            recon_wave = recon.cpu().detach()
                            if recon_wave.ndim == 1:
                                recon_wave = recon_wave.unsqueeze(0)
                            save_audio_file(
                                filename,
                                recon_wave,
                                self.unwrapped_soundstream.target_sample_hz
                            )

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        # save model every so often

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'soundstream.{steps}.pt')
            self.save(model_path)
            self.save(str(self.results_folder / 'latest.pt'))

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        should_eval_best = (
            exists(self.best_checkpoint_metric) and
            exists(self.best_eval_every) and
            not (steps % self.best_eval_every)
        )
        plateau_metric_tensor = None
        if should_eval_best and exists(self.plateau_scheduler):
            plateau_metric_tensor = torch.zeros(3, device = device)

        if self.is_main and should_eval_best:
            online_model = self.unwrapped_soundstream
            online_score = self.evaluate_fixed_validation_score(online_model)
            ema_model = self.ema_soundstream.ema_model if self.use_ema else None
            ema_score = (
                self.evaluate_fixed_validation_score(ema_model.to(device))
                if self.use_ema
                else None
            )
            candidate_models = [('online', online_model, online_score)]
            if self.use_ema:
                candidate_models.append(('ema', ema_model, ema_score))

            quality_reasons = {
                name: self.quality_retention_failure_reasons(metrics)
                for name, _, metrics in candidate_models
            }
            eligible_candidates = [
                (name, model, metrics)
                for name, model, metrics in candidate_models
                if (
                    steps >= self.best_checkpoint_min_step and
                    metrics['validation_eligible'] >= 0.5 and
                    not quality_reasons[name]
                )
            ]
            if eligible_candidates:
                selected_name, selected_model, selected_metrics = min(
                    eligible_candidates,
                    key = lambda candidate: candidate[2]['score']
                )
            else:
                selected_name = 'none'
                selected_model = None
                selected_metrics = online_score
            selection_score = selected_metrics['score']

            si_sdr_candidates = eligible_candidates
            if si_sdr_candidates:
                si_sdr_selected_name, si_sdr_selected_model, si_sdr_selected_metrics = max(
                    si_sdr_candidates,
                    key = lambda candidate: candidate[2]['aligned_si_sdr']
                )
            else:
                si_sdr_selected_name = 'none'
                si_sdr_selected_model = None
                si_sdr_selected_metrics = None

            leakage_candidates = [
                candidate for candidate in eligible_candidates
                if candidate[2].get('frame_diagnostic_valid', 0.) >= 0.5
            ]
            if leakage_candidates:
                leakage_selected_name, leakage_selected_model, leakage_selected_metrics = min(
                    leakage_candidates,
                    key = lambda candidate: candidate[2]['ac_320_isolated']
                )
            else:
                leakage_selected_name = 'none'
                leakage_selected_model = None
                leakage_selected_metrics = None

            self.sync_best_scores_from_disk()

            online_clean_fail = (
                ",".join(self.clean_gate_failure_reasons(online_score)) or
                "none"
            )
            ema_clean_fail = (
                ",".join(self.clean_gate_failure_reasons(ema_score)) or
                "none"
                if self.use_ema
                else "disabled"
            )

            ema_summary = (
                f"ema={ema_score['score']:.6f}, "
                f"ema_rms={ema_score['rms_ratio']:.3f}, "
                f"ema_corr={ema_score['correlation']:.3f}, "
                f"ema_si_sdr={ema_score['si_sdr']:.3f}, "
                f"ema_aligned_corr={ema_score['aligned_correlation']:.3f}, "
                f"ema_aligned_si_sdr={ema_score['aligned_si_sdr']:.3f}, "
                f"ema_env={ema_score.get('spectral_envelope_loss', 0.):.4f}, "
                f"ema_env_low={ema_score.get('spectral_envelope_low', 0.):.4f}, "
                f"ema_env_mid={ema_score.get('spectral_envelope_mid', 0.):.4f}, "
                f"ema_env_high={ema_score.get('spectral_envelope_high', 0.):.4f}, "
                f"ema_voiced={ema_score.get('spectral_envelope_voiced_fraction', 0.):.3f}, "
                f"ema_quiet_hf_excess_db={ema_score.get('quiet_hf_excess_db', 0.):.2f}, "
                f"ema_peak={ema_score.get('recon_peak', 0.):.3f}, "
                f"ema_clip={ema_score.get('recon_clip_fraction', 0.) * 100:.3f}%, "
                f"ema_jump_ratio={ema_score.get('jump_ratio', 0.):.2f}, "
                f"ema_click={ema_score.get('click_score', 0.):.4f}, "
                f"ema_click_allowed={ema_score.get('click_allowed', self.effective_clean_gate_max_click_score()):.4f}, "
                f"ema_click_fail_margin={ema_score.get('click_fail_margin', 0.):+.4f}, "
                f"ema_clean_ok={ema_score.get('clean_validation_eligible', 0.):.0f}, "
                f"ema_clean_fail={ema_clean_fail}, "
                f"ema_active_codes={ema_score['active_code_ratio']:.3f}, "
                f"ema_perplexity={ema_score['codebook_perplexity']:.1f}, "
                f"ema_rvq_ok={ema_score.get('rvq_validation_eligible', 0.):.0f}, "
                f"ema_collapsed_q={int(ema_score.get('codebook_collapsed_quantizers', 0))}"
                if self.use_ema
                else "ema=disabled"
            )
            self.print(
                f"{steps}: validation score online={online_score['score']:.6f}, "
                f"{ema_summary}, best={self.best_valid_score:.6f}, "
                f"selected={selected_name} | "
                f"online_rms={online_score['rms_ratio']:.3f}, "
                f"online_corr={online_score['correlation']:.3f}, "
                f"online_si_sdr={online_score['si_sdr']:.3f}, "
                f"online_aligned_corr={online_score['aligned_correlation']:.3f}, "
                f"online_aligned_si_sdr={online_score['aligned_si_sdr']:.3f}, "
                f"online_env={online_score.get('spectral_envelope_loss', 0.):.4f}, "
                f"online_env_low={online_score.get('spectral_envelope_low', 0.):.4f}, "
                f"online_env_mid={online_score.get('spectral_envelope_mid', 0.):.4f}, "
                f"online_env_high={online_score.get('spectral_envelope_high', 0.):.4f}, "
                f"online_voiced={online_score.get('spectral_envelope_voiced_fraction', 0.):.3f}, "
                f"online_quiet_hf_excess_db={online_score.get('quiet_hf_excess_db', 0.):.2f}, "
                f"online_target_dc={online_score.get('target_dc', 0.):+.5f}, "
                f"online_recon_dc={online_score.get('recon_dc', 0.):+.5f}, "
                f"online_lf0_50_excess_db={online_score.get('low_freq_0_50_excess_db', 0.):+.2f}, "
                f"online_ac320_iso={online_score.get('ac_320_isolated', 0.):+.4f}, "
                f"online_phase_peak_db={online_score.get('frame_phase_peak_db', 0.):+.2f}, "
                f"online_comb_med_db={online_score.get('comb_median_excess_db', 0.):+.2f}, "
                f"online_peak={online_score.get('recon_peak', 0.):.3f}, "
                f"online_clip={online_score.get('recon_clip_fraction', 0.) * 100:.3f}%, "
                f"online_jump_ratio={online_score.get('jump_ratio', 0.):.2f}, "
                f"online_click={online_score.get('click_score', 0.):.4f}, "
                f"online_click_allowed={online_score.get('click_allowed', self.effective_clean_gate_max_click_score()):.4f}, "
                f"online_click_fail_margin={online_score.get('click_fail_margin', 0.):+.4f}, "
                f"online_clean_ok={online_score.get('clean_validation_eligible', 0.):.0f}, "
                f"online_clean_fail={online_clean_fail}, "
                f"online_active_codes={online_score['active_code_ratio']:.3f}, "
                f"online_perplexity={online_score['codebook_perplexity']:.1f}, "
                f"online_q00_ok={online_score.get('q00_validation_eligible', 0.):.0f}, "
                f"online_q00_warning={online_score.get('q00_retention_warning', 0.):.0f}, "
                f"online_q01_ok={online_score.get('q01_validation_eligible', 0.):.0f}, "
                f"online_rvq_ok={online_score.get('rvq_validation_eligible', 0.):.0f}, "
                f"online_collapsed_q={int(online_score.get('codebook_collapsed_quantizers', 0))} | "
                f"best_aligned_si_sdr={self.best_aligned_si_sdr:.3f}, "
                f"si_sdr_selected={si_sdr_selected_name}"
            )
            self.print(
                f"{steps}: codebook online "
                f"{self.format_codebook_diagnostics(online_score)}"
            )
            if self.use_ema:
                self.print(
                    f"{steps}: codebook ema "
                    f"{self.format_codebook_diagnostics(ema_score)}"
                )

            if self.quality_retention_gate and steps >= self.quality_retention_start_step:
                passing_candidates = [
                    name for name, _, _ in candidate_models
                    if not quality_reasons[name]
                ]
                if passing_candidates:
                    self.quality_retention_bad_evals = 0
                    self.quality_retention_rvq_bad_evals = 0
                else:
                    self.quality_retention_bad_evals += 1
                    rvq_bad = all(
                        ('q00' in quality_reasons[name] or 'q01' in quality_reasons[name])
                        for name, _, _ in candidate_models
                    )
                    self.quality_retention_rvq_bad_evals = (
                        self.quality_retention_rvq_bad_evals + 1 if rvq_bad else 0
                    )
                    online_quality_fail = ','.join(quality_reasons['online']) or 'none'
                    self.print(
                        f"{steps}: Stage-2 quality retention gate rejected all candidates "
                        f"(online_fail={online_quality_fail}, "
                        f"quality_bad={self.quality_retention_bad_evals}/"
                        f"{self.quality_retention_patience}, rvq_bad="
                        f"{self.quality_retention_rvq_bad_evals}/"
                        f"{self.quality_retention_rvq_patience})"
                    )
                    if (
                        self.quality_retention_bad_evals >= self.quality_retention_patience or
                        self.quality_retention_rvq_bad_evals >= self.quality_retention_rvq_patience
                    ):
                        self.early_stopping_triggered = True
                        self.save(str(self.results_folder / 'latest.pt'))
                        self.print(
                            f"{steps}: Stage-2 quality retention hard stop triggered; "
                            "the initialized reconstruction baseline was not preserved."
                        )
            elif self.quality_retention_gate and self.is_main:
                self.print(
                    f"{steps}: Stage-2 quality retention monitored but hard-stop "
                    f"is deferred until step {self.quality_retention_start_step}."
                )

            if exists(self.plateau_scheduler):
                plateau_default = (
                    float('inf')
                    if self.plateau_lr_metric == 'score'
                    else float('-inf')
                )
                plateau_metric = float(
                    online_score.get(self.plateau_lr_metric, plateau_default)
                )
                plateau_after_start = steps >= self.plateau_lr_start_steps
                plateau_unclean_fallback = False
                plateau_quality_ready = (
                    not self.plateau_lr_require_quality_retention or
                    not self.quality_retention_failure_reasons(online_score)
                )
                if self.plateau_lr_require_quality_retention:
                    # Stage-2 must not lower LR merely because GAN training is
                    # plateauing while it is already regressing from Stage 1.5.
                    plateau_ready = (
                        plateau_after_start and
                        isfinite(plateau_metric) and
                        plateau_quality_ready
                    )
                else:
                    plateau_clean_ready = (
                        online_score.get('clean_validation_eligible', 0.) >= 0.5
                    )
                    if plateau_after_start and isfinite(plateau_metric):
                        if plateau_clean_ready:
                            self.plateau_lr_unclean_checks = 0
                        else:
                            self.plateau_lr_unclean_checks += 1
                            plateau_unclean_fallback = (
                                self.plateau_lr_unclean_checks >=
                                self.plateau_lr_unclean_grace_checks
                            )
                    plateau_ready = (
                        plateau_after_start and
                        isfinite(plateau_metric) and
                        (
                            plateau_clean_ready or
                            plateau_unclean_fallback
                        )
                    )
                if plateau_ready:
                    plateau_metric_tensor[0] = plateau_metric
                    plateau_metric_tensor[1] = 1.
                    plateau_metric_tensor[2] = float(plateau_unclean_fallback)

            improved = (
                exists(selected_model) and
                selection_score < (
                    self.best_valid_score - self.early_stopping_min_delta
                )
            )
            si_sdr_improved = (
                exists(si_sdr_selected_model) and
                si_sdr_selected_metrics['aligned_si_sdr'] > self.best_aligned_si_sdr
            )
            frame_leakage_improved = (
                self.frame_leakage_checkpoint and
                exists(leakage_selected_model) and
                leakage_selected_metrics['ac_320_isolated'] < self.best_frame_leakage_score
            )

            if frame_leakage_improved:
                self.best_frame_leakage_score = leakage_selected_metrics['ac_320_isolated']
                self.save_model_only(
                    self.results_folder / 'best_by_frame_leakage.pt',
                    leakage_selected_model,
                    score = self.best_frame_leakage_score,
                    step = steps,
                    weight_source = leakage_selected_name
                )
                self.save(str(self.results_folder / 'latest.pt'))
                self.print(
                    f"{steps}: saving best_by_frame_leakage.pt "
                    f"({leakage_selected_name}, ac_320_isolated="
                    f"{self.best_frame_leakage_score:+.5f})"
                )

            if si_sdr_improved:
                self.best_aligned_si_sdr = si_sdr_selected_metrics['aligned_si_sdr']
                self.save_model_only(
                    self.results_folder / 'best_by_aligned_si_sdr.pt',
                    si_sdr_selected_model,
                    score = si_sdr_selected_metrics['aligned_si_sdr'],
                    step = steps,
                    weight_source = si_sdr_selected_name
                )
                self.save(str(self.results_folder / 'latest.pt'))
                self.print(
                    f"{steps}: saving best_by_aligned_si_sdr.pt "
                    f"({si_sdr_selected_name}, "
                    f"aligned_si_sdr={self.best_aligned_si_sdr:.3f})"
                )

            if improved:
                self.best_valid_score = selection_score
                self.early_stopping_bad_evals = 0
                self.save_model_only(
                    self.results_folder / 'best.pt',
                    online_model,
                    score = online_score['score'],
                    step = steps
                )
                if self.use_ema:
                    self.save_model_only(
                        self.results_folder / 'best_ema.pt',
                        ema_model,
                        score = ema_score['score'],
                        step = steps
                    )
                self.save_model_only(
                    self.results_folder / 'best_selected.pt',
                    selected_model,
                    score = selection_score,
                    step = steps,
                    weight_source = selected_name
                )
                self.save(str(self.results_folder / 'latest.pt'))
                self.print(f'{steps}: saving new best checkpoints to {str(self.results_folder)}')
            elif not isfinite(self.best_valid_score):
                self.early_stopping_bad_evals = 0
                self.print(
                    f"{steps}: validation is not yet eligible; "
                    "early-stopping patience starts after the first "
                    "eligible checkpoint"
                )
            elif (
                exists(self.early_stopping_patience) and
                steps >= self.early_stopping_min_steps
            ):
                self.early_stopping_bad_evals += 1
                self.print(
                    f"{steps}: validation did not improve by "
                    f"{self.early_stopping_min_delta:.6g}; "
                    f"early-stopping patience "
                    f"{self.early_stopping_bad_evals}/{self.early_stopping_patience}"
                )

                if self.early_stopping_bad_evals >= self.early_stopping_patience:
                    self.early_stopping_triggered = True
                    self.save(str(self.results_folder / 'latest.pt'))
                    self.print(
                        f"{steps}: early stopping triggered; "
                        f"best validation score={self.best_valid_score:.6f}"
                    )
            elif exists(self.early_stopping_patience):
                self.early_stopping_bad_evals = 0
                self.print(
                    f"{steps}: early stopping is inactive until "
                    f"step {self.early_stopping_min_steps}"
                )

        if should_eval_best and exists(self.plateau_scheduler):
            if (
                self.is_distributed and
                torch.distributed.is_available() and
                torch.distributed.is_initialized()
            ):
                torch.distributed.broadcast(plateau_metric_tensor, src = 0)

            plateau_metric = float(plateau_metric_tensor[0].item())
            plateau_ready = plateau_metric_tensor[1].item() >= 0.5
            plateau_unclean_fallback = plateau_metric_tensor[2].item() >= 0.5
            if plateau_ready:
                old_lrs = [
                    group['lr']
                    for group in self.optim.optimizer.param_groups
                ]
                self.plateau_scheduler.step(plateau_metric)
                self.optim.sync_warmup_lrs_from_optimizer()
                new_lrs = [
                    group['lr']
                    for group in self.optim.optimizer.param_groups
                ]
                lr_changed = any(
                    abs(old_lr_i - new_lr_i) > 1e-12
                    for old_lr_i, new_lr_i in zip(old_lrs, new_lrs)
                )
                discr_lr_changes = []
                if lr_changed and old_lrs[0] > 0.:
                    discr_lr_changes = self.scale_discriminator_lrs_from_plateau(
                        new_lrs[0] / old_lrs[0]
                    )
                if self.is_main:
                    old_lr = old_lrs[0]
                    new_lr = new_lrs[0]
                    metric_name = f"online_{self.plateau_lr_metric}"
                    metric_value = (
                        f"{plateau_metric:.6f}"
                        if self.plateau_lr_metric == 'score'
                        else f"{plateau_metric:.3f}"
                    )
                    source = (
                        'quality_retention'
                        if self.plateau_lr_require_quality_retention
                        else ('unclean_fallback' if plateau_unclean_fallback else 'clean')
                    )
                    if lr_changed:
                        discr_summary = ''
                        if discr_lr_changes:
                            _, old_discr_lrs, new_discr_lrs = discr_lr_changes[0]
                            discr_summary = (
                                f"; discriminator LR {old_discr_lrs[0]:.3e} "
                                f"to {new_discr_lrs[0]:.3e}"
                            )
                        self.print(
                            f"{steps}: ReduceLROnPlateau lowered generator LR "
                            f"from {old_lr:.3e} to {new_lr:.3e}{discr_summary} "
                            f"({metric_name}={metric_value}, "
                            f"source={source})"
                        )
                    else:
                        self.print(
                            f"{steps}: ReduceLROnPlateau observed "
                            f"{metric_name}={metric_value}; "
                            f"generator_lr={new_lr:.3e}; "
                            f"source={source}"
                        )
            elif self.is_main:
                inactive_reasons = []
                if steps < self.plateau_lr_start_steps:
                    inactive_reasons.append(
                        f"before_start_step_{self.plateau_lr_start_steps}"
                    )
                if not isfinite(plateau_metric):
                    inactive_reasons.append(
                        f"invalid_online_{self.plateau_lr_metric}"
                    )
                if self.plateau_lr_require_quality_retention:
                    quality_reasons = (
                        self.quality_retention_failure_reasons(online_score)
                        if 'online_score' in locals() else ['missing_online_score']
                    )
                    if quality_reasons:
                        inactive_reasons.append(
                            "quality_retention:" + ",".join(quality_reasons)
                        )
                elif (
                    'online_score' in locals() and
                    online_score.get('clean_validation_eligible', 0.) < 0.5
                ):
                    inactive_reasons.append(
                        "online_clean_ok_0:"
                        f"{self.plateau_lr_unclean_checks}/"
                        f"{self.plateau_lr_unclean_grace_checks}"
                    )
                reason = ",".join(inactive_reasons) or "not_ready"
                self.print(
                    f"{steps}: ReduceLROnPlateau inactive "
                    f"({reason})"
                )

        self.accelerator.wait_for_everyone()

        stop_flag = torch.tensor(
            int(self.early_stopping_triggered),
            device = self.device
        )
        stop_flag = self.accelerator.reduce(stop_flag, reduction = 'max')
        self.early_stopping_triggered = bool(stop_flag.item())

        self.steps.add_(1)
        return logs

    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps and not self.early_stopping_triggered:
            logs = self.train_step()
            log_fn(logs)

        reason = 'early stopping' if self.early_stopping_triggered else 'maximum steps'
        self.print(f'training complete ({reason})')

# semantic transformer trainer

class SemanticTransformerTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        wav2vec: FairseqVQWav2Vec | HubertWithKmeans | None,
        transformer: SemanticTransformer,
        *,
        num_train_steps,
        batch_size,
        audio_conditioner: AudioConditionerBase | None = None,
        dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
        data_max_length = None,
        data_max_length_seconds = None,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        init_process_group_timeout_seconds = 1800,
        use_wandb_tracking = False,
        split_batches = False,
        drop_last = False,
        force_clear_prev_results = None,
        average_valid_loss_over_grad_accum_every: bool = True, # if False, valid loss on a single batch
    ):
        super().__init__()
        check_one_trainer()

        init_process_kwargs = InitProcessGroupKwargs(timeout = timedelta(seconds = init_process_group_timeout_seconds))
        self.use_wandb_tracking = use_wandb_tracking
        if use_wandb_tracking:
            accelerate_kwargs.update(log_with = 'wandb')
        self.accelerator = Accelerator(
            kwargs_handlers = [DEFAULT_DDP_KWARGS, init_process_kwargs],
            split_batches = split_batches,
            **accelerate_kwargs
        )
        self.wav2vec = wav2vec
        self.transformer = transformer
        self.audio_conditioner = audio_conditioner

        self.train_wrapper = SemanticTransformerWrapper(
            wav2vec = wav2vec,
            transformer = transformer,
            audio_conditioner = audio_conditioner
        )

        self.register_buffer('steps', torch.tensor(0))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr = lr, wd = wd)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = dataset
        if not exists(self.ds):
            assert exists(folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            assert not (exists(data_max_length) and exists(data_max_length_seconds))

            if exists(data_max_length_seconds):
                data_max_length = data_max_length_seconds * wav2vec.target_sample_hz

            self.ds = SoundDataset(
                folder,
                max_length = data_max_length,
                target_sample_hz = wav2vec.target_sample_hz,
                seq_len_multiple_of = wav2vec.seq_len_multiple_of
            )

        self.ds_fields = None

        # split for validation

        self.valid_ds = valid_dataset

        if not exists(self.valid_ds):
            if valid_frac > 0:
                train_size = int((1 - valid_frac) * len(self.ds))
                valid_size = len(self.ds) - train_size
                self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
                self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
            else:
                self.valid_ds = self.ds
                self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        # prepare with accelerator

        (
            self.train_wrapper,
            self.optim,
            self.dl
        ) = self.accelerator.prepare(
            self.train_wrapper,
            self.optim,
            self.dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.accelerator.wait_for_everyone()
        self.results_folder.mkdir(parents = True, exist_ok = True)

        hps = {"num_train_steps": num_train_steps, "data_max_length": data_max_length, "learning_rate": lr}
        self.tracker_hps = hps

        self.accelerator.init_trackers("semantic", config=hps)
        self.average_valid_loss_over_grad_accum_every = average_valid_loss_over_grad_accum_every

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),
            optim = self.optim.state_dict(),
            version = __version__
        )
        torch.save(pkg, path)

    def load(self, path):
        transformer = self.accelerator.unwrap_model(self.transformer)
        pkg = transformer.load(path)
        # trainer-specific things
        self.optim.load_state_dict(pkg['optim'])

        # + 1 to start from the next step and avoid overwriting the last checkpoint
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)


    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def data_tuple_to_kwargs(self, data):
        if not exists(self.ds_fields):
            self.ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'

        return dict(zip(self.ds_fields, data))

    @contextmanager
    def wandb_tracker(self, project, run = None, hps = None):
        assert self.use_wandb_tracking, '`use_wandb_tracking` must be set to True on SemanticTransformerTrainer'

        hps = default(hps, self.tracker_hps)

        self.accelerator.init_trackers(project, config = None)

        if exists(run):
            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), self.accelerator.trackers)
            assert exists(wandb_tracker)

            wandb_tracker.run.name = run

        yield

        self.accelerator.end_training()

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.transformer.train()

        # logs

        logs = {}

        # update transformer

        for i in range(self.grad_accum_every):
            is_last = i == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.train_wrapper) if not is_last else nullcontext

            data_kwargs = self.data_tuple_to_kwargs(next(self.dl_iter))

            with self.accelerator.autocast(), context():
                loss = self.train_wrapper(**data_kwargs, return_loss = True)

                self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            valid_loss = 0
            unwrapped_model = self.accelerator.unwrap_model(self.train_wrapper)

            for _ in range(self.average_valid_loss_over_grad_accum_every):
                data_kwargs = self.data_tuple_to_kwargs(next(self.valid_dl_iter))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)

                with torch.inference_mode():
                    unwrapped_model.eval()
                    valid_loss += unwrapped_model(**data_kwargs, return_loss = True)

            valid_loss = valid_loss.clone() # avoid inference mode to non-inference mode error
            valid_loss /= self.average_valid_loss_over_grad_accum_every

            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'semantic.transformer.{steps}.pt')
            self.save(model_path)
            if self.use_wandb_tracking:
                wandb.save(model_path)
            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.accelerator.wait_for_everyone()

        self.steps.add_(1)
        return logs

    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

# fine transformer trainer

class CoarseTransformerTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        transformer: CoarseTransformer,
        codec: SoundStream | EncodecWrapper,
        wav2vec: FairseqVQWav2Vec | HubertWithKmeans | None,
        *,
        num_train_steps,
        batch_size,
        audio_conditioner: AudioConditionerBase | None = None,
        dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
        ds_fields: tuple[str, ...] = ('raw_wave', 'raw_wave_for_codec', 'text'),
        data_max_length = None,
        data_max_length_seconds = None,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        init_process_group_timeout_seconds = 1800,
        split_batches = False,
        drop_last = False,
        force_clear_prev_results = None,
        use_wandb_tracking = False,
        average_valid_loss_over_grad_accum_every: bool = True,  # if False, valid loss on a single batch
    ):
        super().__init__()
        check_one_trainer()
        self.use_wandb_tracking = use_wandb_tracking
        if use_wandb_tracking:
            accelerate_kwargs.update(log_with = 'wandb')
        init_process_kwargs = InitProcessGroupKwargs(timeout = timedelta(seconds = init_process_group_timeout_seconds))

        self.accelerator = Accelerator(
            kwargs_handlers = [DEFAULT_DDP_KWARGS, init_process_kwargs],
            split_batches = split_batches,
            **accelerate_kwargs
        )

        self.transformer = transformer
        self.codec = codec
        self.wav2vec = wav2vec
        self.audio_conditioner = audio_conditioner

        self.train_wrapper = CoarseTransformerWrapper(
            codec = codec,
            wav2vec = wav2vec,
            transformer = transformer,
            audio_conditioner = audio_conditioner
        )

        self.register_buffer('steps', torch.tensor(0))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr = lr, wd = wd)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = dataset

        if not exists(self.ds):
            assert exists(folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            assert not (exists(data_max_length) and exists(data_max_length_seconds))

            if exists(data_max_length_seconds):
                data_max_length = max(data_max_length_seconds * hz for hz in (wav2vec.target_sample_hz, codec.target_sample_hz))

            self.ds = SoundDataset(
                folder,
                max_length = data_max_length,
                target_sample_hz = (
                    wav2vec.target_sample_hz,
                    codec.target_sample_hz
                ), # need 2 waves resampled differently here
                seq_len_multiple_of = codec.seq_len_multiple_of
            )

        self.ds_fields = ds_fields

        # split for validation

        self.valid_ds = valid_dataset

        if not exists(self.valid_ds):
            if valid_frac > 0:
                train_size = int((1 - valid_frac) * len(self.ds))
                valid_size = len(self.ds) - train_size
                self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
                self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
            else:
                self.valid_ds = self.ds
                self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        # prepare with accelerator

        (
            self.train_wrapper,
            self.optim,
            self.dl
        ) = self.accelerator.prepare(
            self.train_wrapper,
            self.optim,
            self.dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        hps = {"num_train_steps": num_train_steps, "data_max_length": data_max_length, "learning_rate": lr}
        self.tracker_hps = hps

        self.accelerator.init_trackers("coarse", config=hps)

        self.train_wrapper.to(self.device)
        self.average_valid_loss_over_grad_accum_every = average_valid_loss_over_grad_accum_every

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),
            optim = self.optim.state_dict(),
            version = __version__
        )
        torch.save(pkg, path)

    def load(self, path):
        transformer = self.accelerator.unwrap_model(self.transformer)
        pkg = transformer.load(path)
        # trainer-specific things
        self.optim.load_state_dict(pkg['optim'])

        # + 1 to start from the next step and avoid overwriting the last checkpoint
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @contextmanager
    def wandb_tracker(self, project, run = None, hps = None):
        assert self.use_wandb_tracking, '`use_wandb_tracking` must be set to True on CoarseTransformerTrainer'

        hps = default(hps, self.tracker_hps)

        self.accelerator.init_trackers(project, config = None)

        if exists(run):
            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), self.accelerator.trackers)
            assert exists(wandb_tracker)

            wandb_tracker.run.name = run

        yield

        self.accelerator.end_training()  

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.transformer.train()

        # logs

        logs = {}

        # update transformer

        for i in range(self.grad_accum_every):
            is_last = i == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.train_wrapper) if not is_last else nullcontext

            data_kwargs = dict(zip(self.ds_fields, next(self.dl_iter)))

            with self.accelerator.autocast(), context():
                loss = self.train_wrapper(
                    **data_kwargs,
                    return_loss = True
                )

                self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            valid_loss = 0
            unwrapped_model = self.accelerator.unwrap_model(self.train_wrapper)

            for i in range(self.average_valid_loss_over_grad_accum_every):
                data_kwargs = dict(zip(self.ds_fields, next(self.valid_dl_iter)))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)

                with torch.no_grad():
                    unwrapped_model.eval()

                    valid_loss += unwrapped_model(
                        **data_kwargs,
                        return_loss = True
                    )

            valid_loss = valid_loss.clone() # avoid inference mode to non-inference mode error
            valid_loss /= self.average_valid_loss_over_grad_accum_every

            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'coarse.transformer.{steps}.pt')
            self.save(model_path)
            if self.use_wandb_tracking:
                wandb.save(model_path)
            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.accelerator.wait_for_everyone()

        self.steps.add_(1)
        return logs

    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

# fine transformer trainer

class FineTransformerTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        transformer: FineTransformer,
        codec: SoundStream | EncodecWrapper,
        *,
        num_train_steps,
        batch_size,
        audio_conditioner: AudioConditionerBase | None = None,
        dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
        data_max_length = None,
        data_max_length_seconds = None,
        dataset_normalize = False,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        init_process_group_timeout_seconds = 1800,
        split_batches = False,
        drop_last = False,
        use_wandb_tracking = False,
        force_clear_prev_results = None,
        average_valid_loss_over_grad_accum_every: bool = True, # if False, valid loss on a single batch
    ):
        super().__init__()
        check_one_trainer()
        self.use_wandb_tracking = use_wandb_tracking
        if use_wandb_tracking:
            accelerate_kwargs.update(log_with = 'wandb')
        init_process_kwargs = InitProcessGroupKwargs(timeout = timedelta(seconds = init_process_group_timeout_seconds))

        self.accelerator = Accelerator(
            kwargs_handlers = [DEFAULT_DDP_KWARGS, init_process_kwargs],
            split_batches = split_batches,
            **accelerate_kwargs
        )

        self.transformer = transformer
        self.codec = codec
        self.audio_conditioner = audio_conditioner

        self.train_wrapper = FineTransformerWrapper(
            codec = codec,
            transformer = transformer,
            audio_conditioner = audio_conditioner
        )

        self.register_buffer('steps', torch.tensor(0))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr = lr, wd = wd)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = dataset

        if not exists(self.ds):
            assert exists(folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            assert not (exists(data_max_length) and exists(data_max_length_seconds))

            if exists(data_max_length_seconds):
                data_max_length = data_max_length_seconds * codec.target_sample_hz

            self.ds = SoundDataset(
                folder,
                max_length = data_max_length,
                target_sample_hz = codec.target_sample_hz,
                seq_len_multiple_of = codec.seq_len_multiple_of
            )

        self.ds_fields = None

        # split for validation

        self.valid_ds = valid_dataset

        if not exists(self.valid_ds):
            if valid_frac > 0:
                train_size = int((1 - valid_frac) * len(self.ds))
                valid_size = len(self.ds) - train_size
                self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
                self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
            else:
                self.valid_ds = self.ds
                self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        # prepare with accelerator

        (
            self.transformer,
            self.optim,
            self.dl
        ) = self.accelerator.prepare(
            self.transformer,
            self.optim,
            self.dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.accelerator.wait_for_everyone()
        self.results_folder.mkdir(parents = True, exist_ok = True)

        hps = {"num_train_steps": num_train_steps, "data_max_length": data_max_length, "learning_rate": lr}
        self.tracker_hps = hps

        self.accelerator.init_trackers("fine", config=hps)

        self.train_wrapper.to(self.device)
        self.average_valid_loss_over_grad_accum_every = average_valid_loss_over_grad_accum_every

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),
            optim = self.optim.state_dict(),
            version = __version__
        )
        torch.save(pkg, path)

    def load(self, path):
        transformer = self.accelerator.unwrap_model(self.transformer)
        pkg = transformer.load(path)
        # trainer-specific things
        self.optim.load_state_dict(pkg['optim'])

        # + 1 to start from the next step and avoid overwriting the last checkpoint
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @contextmanager
    def wandb_tracker(self, project, run = None, hps = None):
        assert self.use_wandb_tracking, '`use_wandb_tracking` must be set to True on FineTransformerTrainer'

        hps = default(hps, self.tracker_hps)

        self.accelerator.init_trackers(project, config = None)

        if exists(run):
            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), self.accelerator.trackers)
            assert exists(wandb_tracker)

            wandb_tracker.run.name = run

        yield

        self.accelerator.end_training() 

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def data_tuple_to_kwargs(self, data):
        if not exists(self.ds_fields):
            self.ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'

        return dict(zip(self.ds_fields, data))

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.transformer.train()

        # logs

        logs = {}

        # update transformer

        for i in range(self.grad_accum_every):
            is_last = i == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.train_wrapper) if not is_last else nullcontext

            data_kwargs = self.data_tuple_to_kwargs(next(self.dl_iter))

            with self.accelerator.autocast(), context():
                loss = self.train_wrapper(**data_kwargs, return_loss = True)

                self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            unwrapped_model = self.accelerator.unwrap_model(self.train_wrapper)
            valid_loss = 0

            for i in range(self.average_valid_loss_over_grad_accum_every):
                data_kwargs = self.data_tuple_to_kwargs(next(self.valid_dl_iter))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)

                with torch.inference_mode():
                    unwrapped_model.eval()
                    valid_loss += unwrapped_model(**data_kwargs, return_loss = True)

            valid_loss = valid_loss.clone() # avoid inference mode to non-inference mode error
            valid_loss /= self.average_valid_loss_over_grad_accum_every

            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'fine.transformer.{steps}.pt')
            self.save(model_path)
            if self.use_wandb_tracking:
                wandb.save(model_path)
            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.accelerator.wait_for_everyone()

        self.steps.add_(1)
        return logs

    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
