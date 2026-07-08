from __future__ import annotations

import re
import copy
import random
from math import isfinite, sqrt
from datetime import timedelta
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import partial
from collections import Counter
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
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
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

ConstantLRScheduler = partial(LambdaLR, lr_lambda = lambda step: 1.)

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
            self.scheduler = ConstantLRScheduler(optimizer)

        # The LR scheduler must capture the undampened base LR before warmup
        # modifies the optimizer's current LR.
        self.warmup = warmup.LinearWarmup(
            optimizer,
            warmup_period = warmup_steps
        )
        self.optimizer = optimizer

        self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)
        self.accelerator = accelerator

    def state_dict(self):
        return dict(
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            warmup = self.warmup.state_dict()
        )

    def load_state_dict(self, pkg):
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.scheduler.load_state_dict(pkg['scheduler'])
        self.warmup.load_state_dict(pkg['warmup'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

        if not self.accelerator.optimizer_step_was_skipped:
            with self.warmup.dampening():
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
        grad_accum_every: int = 4,
        wd: float = 0.,
        warmup_steps: int = 1000,
        scheduler: Type[LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
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
        transient_loss_warmup_steps: int = 0,
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
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.,
        early_stopping_min_steps: int = 0,
        enable_gan: bool = True,
        gan_start_step: int = 0,
        gan_ramp_steps: int = 0,
        gan_adversarial_max: float = 0.1,
        gan_feature_max: float = 10.,
        freeze_codebook_during_training: bool = False,
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
        self.click_loss_max_weight = float(
            getattr(soundstream, 'click_loss_weight', 0.)
        )
        self.jump_loss_max_weight = float(
            getattr(soundstream, 'jump_loss_weight', 0.)
        )
        self.transient_loss_warmup_steps = transient_loss_warmup_steps
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

        hyperparameters = {
            "num_train_steps": num_train_steps,
            "batch_size": batch_size,
            "gradient_accum_every": grad_accum_every,
            "learning_rate": lr,
            "discriminator_learning_rate": discr_lr,
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

        discr_lr = default(discr_lr, lr)
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
            get_optimizer(soundstream.stft_discriminator.parameters(), lr = discr_lr, wd = wd),
            scheduler = discr_scheduler,
            scheduler_kwargs = discr_scheduler_kwargs,
            warmup_steps = discr_warmup_steps
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
        assert not exists(early_stopping_patience) or early_stopping_patience > 0
        assert early_stopping_min_delta >= 0.
        assert early_stopping_min_steps >= 0
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_min_steps = early_stopping_min_steps
        self.early_stopping_bad_evals = 0
        self.early_stopping_triggered = False
        self.enable_gan = enable_gan
        self.gan_start_step = gan_start_step
        self.gan_ramp_steps = gan_ramp_steps
        self.gan_adversarial_max = gan_adversarial_max
        self.gan_feature_max = gan_feature_max
        self.freeze_codebook_during_training = freeze_codebook_during_training

        self.apply_grad_penalty_every = apply_grad_penalty_every

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
            config = self.unwrapped_soundstream._configs,
            discr_optim = self.discr_optim.state_dict(),
            best_valid_score = self.best_valid_score,
            best_aligned_si_sdr = self.best_aligned_si_sdr,
            early_stopping_bad_evals = self.early_stopping_bad_evals,
            early_stopping_triggered = self.early_stopping_triggered,
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

        target_peak = target.abs().amax(dim = -1).mean()
        recon_peak = recon.abs().amax(dim = -1).mean()
        target_clip_fraction = (target.abs() >= 0.98).float().mean()
        recon_clip_fraction = (recon.abs() >= 0.98).float().mean()

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

        _, multi_spectral_recon_loss, stft_recon_loss = model.reconstruction_losses(target, recon)

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
        elif self.best_checkpoint_metric == 'recon_pretrain':
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
            boundary_loss = float(boundary_loss.detach().cpu()),
            energy_loss = float(energy_loss.detach().cpu()),
            rms_ratio = float(rms_ratio.detach().cpu()),
            target_peak = float(target_peak.detach().cpu()),
            recon_peak = float(recon_peak.detach().cpu()),
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
                        reconstructed = model(
                            codec_input.unsqueeze(0),
                            return_recons_only=True
                        )
                        _, indices, commitment_loss = model(
                            codec_input.unsqueeze(0),
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

    def clean_gate_passes(self, metrics):
        if not self.clean_gate:
            return True

        return (
            metrics.get('aligned_si_sdr', float('-inf')) >= self.clean_gate_min_aligned_si_sdr and
            metrics.get('aligned_correlation', float('-inf')) >= self.clean_gate_min_aligned_corr and
            self.clean_gate_min_rms_ratio <= metrics.get('rms_ratio', float('inf')) <= self.clean_gate_max_rms_ratio and
            metrics.get('recon_peak', float('inf')) <= self.clean_gate_max_recon_peak and
            metrics.get('recon_clip_fraction', float('inf')) <= self.clean_gate_max_recon_clip_fraction and
            metrics.get('click_score', float('inf')) <= self.clean_gate_max_click_score and
            metrics.get('jump_ratio', float('inf')) <= self.clean_gate_max_jump_ratio and
            metrics.get('p999_jump_ratio', float('inf')) <= self.clean_gate_max_p999_jump_ratio
        )

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
            averaged_metrics.get('codebook_q00_active_ratio', 0.) >= 0.30 and
            averaged_metrics.get('codebook_q00_perplexity', 0.) >= 16.
        )
        rvq_eligible = (
            averaged_metrics.get('active_code_ratio', 0.) >= 0.25 and
            averaged_metrics.get('codebook_perplexity', 0.) >= 20. and
            averaged_metrics.get('codebook_collapsed_quantizers', 0) <= 2
        )
        averaged_metrics['q00_validation_eligible'] = float(q00_eligible)
        averaged_metrics['rvq_validation_eligible'] = float(rvq_eligible)
        clean_eligible = self.clean_gate_passes(averaged_metrics)
        averaged_metrics['clean_validation_eligible'] = float(clean_eligible)
        if self.best_checkpoint_metric in ('recon_pretrain', 'gan_pretrain'):
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

        self.unwrapped_soundstream.load_state_dict(pkg['model'])

        if self.use_ema:
            assert 'ema_model' in pkg
            self.ema_soundstream.load_state_dict(pkg['ema_model'])

        self.optim.load_state_dict(pkg['optim'])
        self.discr_optim.load_state_dict(pkg['discr_optim'])
        self.best_valid_score = pkg.get('best_valid_score', float('inf'))
        self.best_aligned_si_sdr = pkg.get('best_aligned_si_sdr', float('-inf'))
        self.early_stopping_bad_evals = pkg.get('early_stopping_bad_evals', 0)
        self.early_stopping_triggered = pkg.get('early_stopping_triggered', False)

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

    def update_discriminators(self, device, steps, logs):
        apply_grad_penalty = (
            self.apply_grad_penalty_every > 0 and
            not (steps % self.apply_grad_penalty_every)
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
                    apply_grad_penalty = apply_grad_penalty,
                    return_discr_loss = True,
                    return_discr_losses_separately = True,
                    freeze_codebook = True
                )
                for name, discr_loss in discr_losses:
                    self.accelerator.backward(
                        discr_loss / self.grad_accum_every,
                        retain_graph = True
                    )
                    accum_log(
                        logs,
                        {name: discr_loss.item() / self.grad_accum_every}
                    )

        if exists(self.discr_max_grad_norm):
            model = self.unwrapped_soundstream
            self.accelerator.clip_grad_norm_(
                model.discriminators.parameters(),
                self.discr_max_grad_norm
            )
            self.accelerator.clip_grad_norm_(
                model.stft_discriminator.parameters(),
                self.discr_max_grad_norm
            )

        self.discr_optim.step()
        for _, multiscale_discr_optim in self.multiscale_discriminator_optim_iter():
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

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())
        log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)
        gan_progress = self.update_gan_weights(steps)
        si_sdr_loss_progress = self.update_si_sdr_loss_weight(steps)
        transient_loss_progress = self.update_transient_loss_weights(steps)

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
            with (
                self.accelerator.autocast(),
                context(),
                freeze_discriminators()
            ):
                loss, (
                    recon_loss,
                    multi_spectral_recon_loss,
                    stft_recon_loss,
                    adversarial_loss,
                    feature_loss,
                    all_commitment_loss,
                    si_sdr_loss,
                    click_loss,
                    jump_loss,
                    boundary_loss
                ) = self.soundstream(
                    wave,
                    return_loss_breakdown = True,
                    freeze_codebook = self.freeze_codebook_during_training
                )

                self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, dict(
                loss = loss.item() / self.grad_accum_every,
                recon_loss = recon_loss.item() / self.grad_accum_every,
            ))

            accum_log(logs, dict(
                multi_spectral_recon_loss = multi_spectral_recon_loss.item() / self.grad_accum_every,
                stft_recon_loss = stft_recon_loss.item() / self.grad_accum_every,
                adversarial_loss = adversarial_loss.item() / self.grad_accum_every,
                feature_loss = feature_loss.item() / self.grad_accum_every,
                all_commitment_loss = all_commitment_loss.item() / self.grad_accum_every,
                si_sdr_loss = si_sdr_loss.item() / self.grad_accum_every,
                click_loss = click_loss.item() / self.grad_accum_every,
                jump_loss = jump_loss.item() / self.grad_accum_every,
                boundary_loss = boundary_loss.item() / self.grad_accum_every,
            ))

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.soundstream.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        if self.enable_gan and gan_progress > 0:
            self.update_discriminators(device, steps, logs)

        # build pretty printed losses

        model = self.unwrapped_soundstream
        generator_lr = self.optim.optimizer.param_groups[0]['lr']
        discriminator_lr = self.discr_optim.optimizer.param_groups[0]['lr']
        losses_str = (
            f"{steps}: "
            f"g_total={logs['loss']:.6f} | "
            f"wave={logs['recon_loss']:.6f} | "
            f"mel={logs['multi_spectral_recon_loss']:.6f} | "
            f"stft={logs['stft_recon_loss']:.6f} | "
            f"si_sdr_loss={logs['si_sdr_loss']:.6f}"
            f"(w={model.si_sdr_loss_weight:.4g},ramp={si_sdr_loss_progress:.4f}) | "
            f"click_loss={logs['click_loss']:.6f}"
            f"(w={model.click_loss_weight:.4g},ramp={transient_loss_progress:.4f}) | "
            f"jump_loss={logs['jump_loss']:.6f}"
            f"(w={model.jump_loss_weight:.4g},ramp={transient_loss_progress:.4f}) | "
            f"commit={logs['all_commitment_loss']:.6f}"
        )

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
            f"lr_g={generator_lr:.3e} | "
            f"lr_d={discriminator_lr:.3e}"
        )

        if log_losses:
            self.log(**logs)

        discriminator_log_names = {
            'stft': 'd_stft',
            'stft_grad_penalty': 'gp_stft',
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

        if self.is_main and self.use_ema:
            self.ema_soundstream.update()

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

            eligible_candidates = [
                (name, model, metrics)
                for name, model, metrics in candidate_models
                if metrics['validation_eligible'] >= 0.5
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

            self.sync_best_scores_from_disk()

            ema_summary = (
                f"ema={ema_score['score']:.6f}, "
                f"ema_rms={ema_score['rms_ratio']:.3f}, "
                f"ema_corr={ema_score['correlation']:.3f}, "
                f"ema_si_sdr={ema_score['si_sdr']:.3f}, "
                f"ema_aligned_corr={ema_score['aligned_correlation']:.3f}, "
                f"ema_aligned_si_sdr={ema_score['aligned_si_sdr']:.3f}, "
                f"ema_peak={ema_score.get('recon_peak', 0.):.3f}, "
                f"ema_clip={ema_score.get('recon_clip_fraction', 0.) * 100:.3f}%, "
                f"ema_jump_ratio={ema_score.get('jump_ratio', 0.):.2f}, "
                f"ema_click={ema_score.get('click_score', 0.):.1f}, "
                f"ema_clean_ok={ema_score.get('clean_validation_eligible', 0.):.0f}, "
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
                f"online_peak={online_score.get('recon_peak', 0.):.3f}, "
                f"online_clip={online_score.get('recon_clip_fraction', 0.) * 100:.3f}%, "
                f"online_jump_ratio={online_score.get('jump_ratio', 0.):.2f}, "
                f"online_click={online_score.get('click_score', 0.):.1f}, "
                f"online_clean_ok={online_score.get('clean_validation_eligible', 0.):.0f}, "
                f"online_active_codes={online_score['active_code_ratio']:.3f}, "
                f"online_perplexity={online_score['codebook_perplexity']:.1f}, "
                f"online_q00_ok={online_score.get('q00_validation_eligible', 0.):.0f}, "
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
