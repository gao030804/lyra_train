from __future__ import annotations

from pathlib import Path
from functools import partial, wraps

from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable

import torchaudio
from torchaudio.functional import resample

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from audiolm_pytorch.utils import curtail_to_multiple

from einops import rearrange, reduce

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional runtime fallback
    sf = None

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def is_unique(arr):
    return len(set(arr)) == len(arr)

def load_audio_file(file):
    path = Path(file)
    if sf is not None and path.suffix.lower() in {'.flac', '.wav', '.ogg'}:
        audio, sample_hz = sf.read(str(path), dtype = 'float32', always_2d = True)
        return torch.from_numpy(audio.T).contiguous(), sample_hz

    return torchaudio.load(str(path))

def save_audio_file(file, wave, sample_hz):
    path = Path(file)
    path.parent.mkdir(parents = True, exist_ok = True)

    if sf is not None and path.suffix.lower() in {'.flac', '.wav', '.ogg'}:
        sf.write(
            str(path),
            wave.detach().cpu().squeeze(0).numpy(),
            sample_hz
        )
        return

    torchaudio.save(str(path), wave.detach().cpu(), sample_hz)

# dataset functions

class SoundDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        target_sample_hz: int | Tuple[int, ...],  # target sample hz must be specified, or a tuple of them if one wants to return multiple resampled
        exts = ['flac', 'wav', 'mp3', 'webm'],
        max_length: int | None = None,               # max length would apply to the highest target_sample_hz, if there are multiple
        seq_len_multiple_of: int | tuple[int | None, ...] | None = None,
        max_files: int | None = None,
        fixed_crop: bool = False,
        min_rms_db: float | None = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), f'folder "{str(path)}" does not exist'

        files = sorted(file for ext in exts for file in path.glob(f'**/*.{ext}'))
        if exists(max_files):
            files = files[:max_files]
        assert len(files) > 0, 'no sound files found'

        self.files = files

        self.max_length = max_length
        self.fixed_crop = fixed_crop
        self.min_rms = (
            10 ** (min_rms_db / 20)
            if exists(min_rms_db)
            else None
        )

        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        # strategy, if there are multiple target sample hz, would be to resample to the highest one first
        # apply the max lengths, and then resample to all the others

        self.max_target_sample_hz = max(self.target_sample_hz)
        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.target_sample_hz) == len(self.seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        data, sample_hz = load_audio_file(file)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = reduce(data, 'c ... -> 1 ...', 'mean')

        # first resample data to the max target freq

        data = resample(data, sample_hz, self.max_target_sample_hz)
        sample_hz = self.max_target_sample_hz

        # then curtail or pad the audio depending on the max length

        max_length = self.max_length
        audio_length = data.size(1)

        if exists(max_length):
            if audio_length > max_length:
                max_start = audio_length - max_length
                if self.fixed_crop:
                    # Preserve the deterministic overfit/debug behavior.
                    starts = torch.linspace(
                        0,
                        max_start,
                        steps = 10
                    ).long()
                    candidates = [
                        data[:, int(start):int(start) + max_length]
                        for start in starts
                    ]
                    data = max(
                        candidates,
                        key = lambda candidate: candidate.square().mean().item()
                    )
                else:
                    # Use one uniformly random crop for every ordinary
                    # training sample.  Do not rank candidate windows by RMS:
                    # the codec must also model pauses, quiet speech, and the
                    # real background-noise distribution.
                    start = torch.randint(0, max_start + 1, ()).item()
                    data = data[:, start:start + max_length]
            else:
                data = F.pad(data, (0, max_length - audio_length), 'constant')

        data = rearrange(data, '1 ... -> ...')

        # resample if target_sample_hz is not None in the tuple

        num_outputs = len(self.target_sample_hz)
        data = cast_tuple(data, num_outputs)

        data_tuple = tuple(resample(d, sample_hz, target_sample_hz) for d, target_sample_hz in zip(data, self.target_sample_hz))

        output = []

        # process each of the data resample at different frequencies individually for curtailing to multiple

        for data, seq_len_multiple_of in zip(data_tuple, self.seq_len_multiple_of):
            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]

        return output

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = fn(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
