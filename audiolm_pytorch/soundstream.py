from __future__ import annotations

import functools
import math
from pathlib import Path
from functools import partial, wraps
from itertools import cycle, zip_longest

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

import torchaudio.transforms as T
from torchaudio.functional import resample

from einops import rearrange, reduce, pack, unpack

from vector_quantize_pytorch import (
    GroupedResidualVQ,
    GroupedResidualLFQ,
    GroupedResidualFSQ
)

from local_attention import LocalMHA
from local_attention.transformer import FeedForward, DynamicPositionBias

from gateloop_transformer import SimpleGateLoopLayer as GateLoop

from audiolm_pytorch.utils import curtail_to_multiple

from audiolm_pytorch.version import __version__
from packaging import version
parsed_version = version.parse(__version__)

import pickle

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def filter_by_keys(fn, d):
    return {k: v for k, v in d.items() if fn(k)}

def map_keys(fn, d):
    return {fn(k): v for k, v in d.items()}

# gan losses

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def r1_gradient_penalty(wave, logits, gamma):
    """Length-normalized real-sample R1 using one score per sample.

    The conventional squared-gradient sum grows with the waveform length.  For
    audio segments of different duration, use the per-input-element mean as the
    optimization penalty and return the sum as a diagnostic only.
    """
    if not wave.requires_grad:
        raise ValueError('wave must require gradients for R1 penalty')
    if gamma <= 0:
        zero = wave.new_zeros(())
        return zero, zero

    batch_scores = rearrange(logits, 'b ... -> b (...)').mean(dim = -1)
    gradients = torch_grad(
        outputs = batch_scores,
        inputs = wave,
        grad_outputs = torch.ones_like(batch_scores),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]
    gradients = rearrange(gradients, 'b ... -> b (...)')
    squared_gradients = gradients.square()
    raw_sum = squared_gradients.sum(dim = 1).mean()
    raw_mean = squared_gradients.mean()
    weighted_penalty = 0.5 * gamma * raw_mean
    return weighted_penalty, raw_mean.detach(), raw_sum.detach()

def aggregate_discriminator_losses(
    waveform_losses,
    stft_loss = None,
    waveform_penalties = (),
    stft_penalty = None
):
    """Average discriminator branches after adding each branch's regularizer."""
    branch_losses = []
    for index, loss in enumerate(waveform_losses):
        if index < len(waveform_penalties):
            loss = loss + waveform_penalties[index]
        branch_losses.append(loss)

    if exists(stft_loss):
        if exists(stft_penalty):
            stft_loss = stft_loss + stft_penalty
        branch_losses.append(stft_loss)

    if not branch_losses:
        raise RuntimeError('no discriminator branch losses were produced')

    return torch.stack(branch_losses).mean()

# better sequential

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# discriminators

class MultiScaleDiscriminator(Module):
    def __init__(
        self,
        channels = 16,
        layers = 4,
        groups = (4, 16, 64, 256),
        chan_max = 1024,
        input_channels = 1
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(input_channels, channels, 15, padding = 7)
        self.conv_layers = ModuleList([])

        curr_channels = channels

        for _, group in zip(range(layers), groups):
            chan_out = min(curr_channels * 4, chan_max)

            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(curr_channels, chan_out, 41, stride = 4, padding = 20, groups = group),
                leaky_relu()
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv1d(curr_channels, curr_channels, 5, padding = 2),
            leaky_relu(),
            nn.Conv1d(curr_channels, 1, 3, padding = 1),
        )

    def forward(
        self,
        x,
        return_intermediates = False
    ):
        x = self.init_conv(x)
        intermediates = []

        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)

        out = self.final_conv(x)

        if not return_intermediates:
            return out

        return out, intermediates

# autoregressive squeeze excitation
# https://arxiv.org/abs/1709.01507

class SqueezeExcite(Module):
    def __init__(self, dim, reduction_factor = 4, dim_minimum = 8):
        super().__init__()
        dim_inner = max(dim_minimum, dim // reduction_factor)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, 1),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq, device = x.shape[-2], x.device

        # cumulative mean - since it is autoregressive

        cum_sum = x.cumsum(dim = -2)
        denom = torch.arange(1, seq + 1, device = device).float()
        cum_mean = cum_sum / rearrange(denom, 'n -> n 1')

        # glu gate

        gate = self.net(cum_mean)

        return x * gate

# real-valued STFT discriminator with real / imaginary input channels

def STFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        Residual(Sequential(
            nn.Conv2d(chan_in, chan_in, 3, padding = 1),
            leaky_relu(),
            nn.Conv2d(chan_in, chan_in, 3, padding = 1)
        )),
        nn.Conv2d(
            chan_in,
            chan_out,
            kernel_sizes,
            stride = strides,
            padding = paddings
        )
    )

class STFTDiscriminator(Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = ((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)),
        chan_mults = (1, 2, 4, 4, 8, 8),
        input_channels = 2,
        n_fft = 1024,
        hop_length = 256,
        win_length = 1024,
        stft_normalized = False,
        stft_window_fn = torch.hann_window,
        logits_abs = False
    ):
        super().__init__()
        if input_channels != 2:
            raise ValueError(
                "STFTDiscriminator expects two real-valued channels "
                "containing the STFT real and imaginary parts"
            )
        if logits_abs:
            raise ValueError(
                "STFTDiscriminator requires signed logits for hinge loss; "
                "logits_abs=True is unsupported"
            )

        self.init_conv = nn.Conv2d(input_channels, channels, 7, padding = 3)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        curr_channels = channels

        self.layers = ModuleList([])

        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            self.layers.append(STFTResidualUnit(chan_in, chan_out, layer_stride))

        # torch.stft returns n_fft // 2 + 1 bins.  Drop Nyquist in forward so
        # the paper's F = W / 2 convention is exact, then compute the actual
        # post-block frequency width from the configured strides.  This avoids
        # a hard-coded kernel that leaves multiple frequency logits behind.
        final_frequency_bins = n_fft // 2
        for _, frequency_stride in strides:
            kernel_size = frequency_stride + 2
            padding = kernel_size // 2
            final_frequency_bins = (
                final_frequency_bins + 2 * padding - kernel_size
            ) // frequency_stride + 1

        self.final_frequency_bins = final_frequency_bins
        self.final_conv = nn.Conv2d(
            layer_channels[-1],
            1,
            (1, final_frequency_bins)
        )

        # stft settings

        self.stft_normalized = stft_normalized
        self.stft_window_fn = stft_window_fn

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, return_intermediates = False):
        x = rearrange(x, 'b 1 n -> b n')

        '''
        reference: The content of the paper( https://arxiv.org/pdf/2107.03312.pdf)is as follows:
        The STFT-based discriminator is illustrated in Figure 4
        and operates on a single scale, computing the STFT with a
        window length of W = 1024 samples and a hop length of
        H = 256 samples
        '''

        # Always compute the complex STFT in float32.  Complex autocast support
        # varies across PyTorch versions and previously made this discriminator
        # unnecessarily fragile under mixed precision.
        with torch.autocast(device_type = x.device.type, enabled = False):
            x = x.float()
            stft_window = self.stft_window_fn(
                self.win_length,
                device = x.device,
                dtype = torch.float32
            )
            x = torch.stft(
                x,
                self.n_fft,
                hop_length = self.hop_length,
                win_length = self.win_length,
                window = stft_window,
                normalized = self.stft_normalized,
                return_complex = True
            )

        # [batch, frequency, time] -> [batch, real_or_imag, time, frequency].
        # The residual-unit strides are defined as (time, frequency), matching
        # the SoundStream paper rather than accidentally swapping both axes.
        x = x[:, :-1, :]
        x = torch.view_as_real(x)
        x = rearrange(x, 'b f t c -> b c t f').contiguous()

        intermediates = []

        x = self.init_conv(x)

        intermediates.append(x)

        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        logits = self.final_conv(x)
        if logits.shape[-1] != 1:
            raise RuntimeError(
                "STFT discriminator final convolution did not aggregate all "
                f"frequency bins: output_shape={tuple(logits.shape)}, "
                f"kernel_frequency={self.final_frequency_bins}"
            )
        logits = logits.squeeze(-1)

        if not return_intermediates:
            return logits

        return logits, intermediates

# Backward-compatible import name.  The implementation is intentionally real
# valued; "Complex" now only describes its STFT source representation.
ComplexSTFTDiscriminator = STFTDiscriminator

# sound stream

class Residual(Module):
    def __init__(self, fn: Module, scale = 1.):
        super().__init__()
        self.fn = fn
        self.residual_scale = float(scale)

    def set_residual_scale(self, scale: float):
        self.residual_scale = float(scale)

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.residual_scale + x

class ChannelTranspose(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c n -> b n c')
        out = self.fn(x, **kwargs) + x
        return rearrange(out, 'b n c -> b c n')

class CausalConv1d(Module):
    def __init__(self, chan_in, chan_out, kernel_size, pad_mode = 'reflect', **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0), mode = self.pad_mode)
        return self.conv(x)

    def forward_stream(self, x, state = None):
        cache_size = self.causal_padding

        if cache_size == 0:
            return self.conv(x), None

        if not exists(state):
            state = x.new_zeros(*x.shape[:-1], cache_size)

        x_with_cache = torch.cat((state, x), dim = -1)
        out = self.conv(x_with_cache)
        next_state = x_with_cache[..., -cache_size:]
        return out, next_state

class CausalConvTranspose1d(Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.stream_overlap = kernel_size - stride
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out

    def forward_stream(self, x, state = None):
        num_input_frames = x.shape[-1]
        output_length = num_input_frames * self.upsample_factor
        overlap_length = self.stream_overlap

        out = F.conv_transpose1d(
            x,
            self.conv.weight,
            bias = None,
            stride = self.conv.stride,
            padding = self.conv.padding,
            output_padding = self.conv.output_padding,
            groups = self.conv.groups,
            dilation = self.conv.dilation
        )

        if not exists(state):
            state = out.new_zeros(*out.shape[:-1], overlap_length)

        out = torch.cat(
            (out[..., :overlap_length] + state, out[..., overlap_length:]),
            dim = -1
        )
        next_state = out[..., output_length:(output_length + overlap_length)]
        out = out[..., :output_length]

        if exists(self.conv.bias):
            out = out + rearrange(self.conv.bias, 'c -> 1 c 1')

        return out, next_state

class CausalLinearUpsampleConv1d(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size,
        stride,
        pad_mode = 'reflect',
        interpolation_mode = 'linear',
        **kwargs
    ):
        super().__init__()
        if interpolation_mode not in ('linear', 'cubic'):
            raise ValueError(
                f'unknown causal interpolation mode: {interpolation_mode}'
            )
        self.upsample_factor = stride
        self.interpolation_mode = interpolation_mode
        self.conv = CausalConv1d(chan_in, chan_out, kernel_size, pad_mode = pad_mode, **kwargs)

    @property
    def history_size(self):
        return 1 if self.interpolation_mode == 'linear' else 2

    def _initial_history(self, x):
        return x[..., :1].expand(
            *x.shape[:-1],
            self.history_size
        )

    def _upsample(self, x, history = None):
        factor = self.upsample_factor

        if factor == 1:
            return x

        if not exists(history):
            history = self._initial_history(x)
        elif history.shape[-1] != self.history_size:
            raise ValueError(
                'causal interpolation history has the wrong length: '
                f'expected {self.history_size}, got {history.shape[-1]}'
            )

        prev = history[..., -1:]
        start = torch.cat((prev, x[..., :-1]), dim = -1)
        end = x
        weights = torch.linspace(
            1. / factor,
            1.,
            steps = factor,
            device = x.device,
            dtype = x.dtype
        )
        weights = rearrange(weights, 's -> 1 1 1 s')

        if self.interpolation_mode == 'linear':
            out = start[..., None] * (1. - weights) + end[..., None] * weights
        else:
            # Causal cubic Hermite interpolation.  The tangent at each latent
            # knot is its backward difference, so the end tangent of one
            # interval exactly matches the start tangent of the next interval.
            # This removes the repeated first-derivative discontinuity that
            # ordinary piecewise-linear interpolation introduces at the
            # 50 Hz codec-frame boundary, without looking at future latents.
            interpolation_input = torch.cat((history, x), dim = -1)
            prev_prev = interpolation_input[..., :-2]
            start_tangent = start - prev_prev
            end_tangent = end - start

            t = weights
            t2 = t.square()
            t3 = t2 * t
            h00 = 2. * t3 - 3. * t2 + 1.
            h10 = t3 - 2. * t2 + t
            h01 = -2. * t3 + 3. * t2
            h11 = t3 - t2
            out = (
                h00 * start[..., None] +
                h10 * start_tangent[..., None] +
                h01 * end[..., None] +
                h11 * end_tangent[..., None]
            )

        return rearrange(out, 'b c n s -> b c (n s)')

    def forward(self, x):
        x = self._upsample(x)
        return self.conv(x)

    def forward_stream(self, x, state = None):
        history = None
        conv_state = None

        if exists(state):
            history, conv_state = state

        upsampled = self._upsample(x, history = history)
        out, next_conv_state = self.conv.forward_stream(upsampled, conv_state)
        history_source = torch.cat(
            (
                self._initial_history(x) if not exists(history) else history,
                x
            ),
            dim = -1
        )
        next_history = history_source[..., -self.history_size:]

        return out, (next_history, next_conv_state)

class DepthwiseSeparableCausalConv1d(Module):
    """Causal depthwise filtering followed directly by pointwise mixing."""

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size,
        *,
        stride = 1,
        dilation = 1,
        pad_mode = 'reflect',
        pointwise_rank = None
    ):
        super().__init__()
        pointwise = (
            LowRankPointwiseCausalConv1d(
                chan_in,
                chan_out,
                pointwise_rank,
                pad_mode = pad_mode
            )
            if exists(pointwise_rank)
            else CausalConv1d(chan_in, chan_out, 1, pad_mode = pad_mode)
        )
        self.net = nn.Sequential(
            CausalConv1d(
                chan_in,
                chan_in,
                kernel_size,
                stride = stride,
                dilation = dilation,
                groups = chan_in,
                pad_mode = pad_mode
            ),
            pointwise
        )

    def forward(self, x):
        return self.net(x)

    def forward_stream(self, x, state = None):
        return stream_module(self.net, x, state)

class LowRankPointwiseCausalConv1d(Module):
    """Factor a pointwise matrix as Cin -> rank -> Cout without activation."""

    def __init__(self, chan_in, chan_out, rank, *, pad_mode = 'reflect'):
        super().__init__()
        rank = int(rank)
        if rank <= 0:
            raise ValueError(f'pointwise rank must be positive, got {rank}')
        self.rank = rank
        self.net = nn.Sequential(
            CausalConv1d(chan_in, rank, 1, pad_mode = pad_mode),
            CausalConv1d(rank, chan_out, 1, pad_mode = pad_mode)
        )

    def forward(self, x):
        return self.net(x)

    def forward_stream(self, x, state = None):
        return stream_module(self.net, x, state)

def ResidualUnit(
    chan_in,
    chan_out,
    dilation,
    kernel_size = 7,
    squeeze_excite = False,
    pad_mode = 'reflect',
    residual_scale = 1.,
    depthwise_separable = False,
    pointwise_rank = None
):
    temporal_conv = (
        DepthwiseSeparableCausalConv1d(
            chan_in,
            chan_out,
            kernel_size,
            dilation = dilation,
            pad_mode = pad_mode,
            pointwise_rank = pointwise_rank
        )
        if depthwise_separable
        else CausalConv1d(
            chan_in,
            chan_out,
            kernel_size,
            dilation = dilation,
            pad_mode = pad_mode
        )
    )
    output_pointwise = (
        LowRankPointwiseCausalConv1d(
            chan_out,
            chan_out,
            pointwise_rank,
            pad_mode = pad_mode
        )
        if exists(pointwise_rank)
        else CausalConv1d(chan_out, chan_out, 1, pad_mode = pad_mode)
    )
    return Residual(Sequential(
        temporal_conv,
        nn.ReLU(),
        output_pointwise,
        nn.ReLU(),
        SqueezeExcite(chan_out) if squeeze_excite else None
    ), scale = residual_scale)

def EncoderBlock(
    chan_in,
    chan_out,
    stride,
    cycle_dilations = (1, 3, 9),
    squeeze_excite = False,
    pad_mode = 'reflect',
    depthwise_separable = False,
    residual_pointwise_rank = None,
    downsample_pointwise_rank = None
):
    it = cycle(cycle_dilations)
    residual_unit = partial(
        ResidualUnit,
        squeeze_excite = squeeze_excite,
        pad_mode = pad_mode,
        depthwise_separable = depthwise_separable,
        pointwise_rank = residual_pointwise_rank
    )
    downsample = (
        DepthwiseSeparableCausalConv1d(
            chan_in,
            chan_out,
            2 * stride,
            stride = stride,
            pad_mode = pad_mode,
            pointwise_rank = downsample_pointwise_rank
        )
        if depthwise_separable
        else CausalConv1d(
            chan_in,
            chan_out,
            2 * stride,
            stride = stride,
            pad_mode = pad_mode
        )
    )

    return nn.Sequential(
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        downsample
    )

def DecoderBlock(
    chan_in,
    chan_out,
    stride,
    cycle_dilations = (1, 3, 9),
    squeeze_excite = False,
    pad_mode = 'reflect',
    upsample_mode = 'convtranspose',
    residual_scale = 1.,
    linear_upsample_kernel_min = 0,
    interpolation_mode = 'linear',
    split_upsample = False
):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    residual_unit = partial(
        ResidualUnit,
        squeeze_excite = squeeze_excite,
        pad_mode = pad_mode,
        residual_scale = residual_scale
    )

    if split_upsample:
        if upsample_mode != 'linear':
            raise ValueError(
                'split decoder upsampling requires causal interpolation mode'
            )
        if stride != 8:
            raise ValueError(
                f'split decoder upsampling expects stride 8, got {stride}'
            )

        # Replace the abrupt first x8 expansion with x4 -> shaping -> x2.
        # Both stages keep a causal convolution, and ReLU between them gives the
        # second stage a chance to reshape interpolation images before the
        # ordinary residual stack.  The total temporal factor remains x8.
        first_stride, second_stride = 4, 2
        first_kernel = max(2 * first_stride, linear_upsample_kernel_min)
        second_kernel = max(2 * second_stride, linear_upsample_kernel_min)
        upsample = nn.Sequential(
            CausalLinearUpsampleConv1d(
                chan_in,
                chan_out,
                first_kernel,
                stride = first_stride,
                pad_mode = pad_mode,
                interpolation_mode = interpolation_mode
            ),
            nn.ReLU(),
            CausalLinearUpsampleConv1d(
                chan_out,
                chan_out,
                second_kernel,
                stride = second_stride,
                pad_mode = pad_mode,
                interpolation_mode = interpolation_mode
            )
        )
    elif upsample_mode == 'convtranspose':
        upsample = CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride)
    elif upsample_mode == 'linear':
        upsample_kernel_size = max(2 * stride, linear_upsample_kernel_min)
        upsample = CausalLinearUpsampleConv1d(
            chan_in,
            chan_out,
            upsample_kernel_size,
            stride = stride,
            pad_mode = pad_mode,
            interpolation_mode = interpolation_mode
        )
    else:
        raise ValueError(f'unknown decoder upsample mode: {upsample_mode}')

    it = cycle(cycle_dilations)
    return nn.Sequential(
        upsample,
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
    )

def stream_module(module, x, state = None):
    if isinstance(module, (
        CausalConv1d,
        CausalConvTranspose1d,
        CausalLinearUpsampleConv1d,
        DepthwiseSeparableCausalConv1d,
    )):
        return module.forward_stream(x, state)

    if isinstance(module, Residual):
        residual, next_state = stream_module(module.fn, x, state)
        return residual * module.residual_scale + x, next_state

    if isinstance(module, nn.Sequential):
        states = state if exists(state) else [None] * len(module)
        assert len(states) == len(module)
        next_states = []

        for submodule, substate in zip(module, states):
            x, next_substate = stream_module(submodule, x, substate)
            next_states.append(next_substate)

        return x, next_states

    return module(x), None

class LocalTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        window_size,
        dynamic_pos_bias = False,
        **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.layers = ModuleList([])

        self.pos_bias = None
        if dynamic_pos_bias:
            self.pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        for _ in range(depth):
            self.layers.append(ModuleList([
                LocalMHA(
                    dim = dim,
                    heads = heads,
                    qk_rmsnorm = True,
                    window_size = window_size,
                    use_rotary_pos_emb = not dynamic_pos_bias,
                    gate_values_per_head = True,
                    use_xpos = True,
                    **kwargs
                ),
                FeedForward(dim = dim)
            ]))

    def forward(self, x):
        w = self.window_size

        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        return x

class FiLM(Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
        return x * gamma + beta

class SoundStream(Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        codebook_dim = 512,
        codebook_size: int | None = None,
        finite_scalar_quantizer_levels: list[int] | None = None,
        rq_num_quantizers = 8,
        rq_commitment_weight = 1.,
        rq_ema_decay = 0.99,
        rq_quantize_dropout_multiple_of = 1,
        rq_quantize_dropout = True,
        rq_groups = 1,
        rq_stochastic_sample_codes = False,
        rq_rotation_trick = True,
        rq_threshold_ema_dead_code = 2,
        rq_kwargs: dict = {},
        bypass_rvq = False,
        use_lookup_free_quantizer = False,              # proposed in https://arxiv.org/abs/2310.05737, adapted for residual quantization
        use_finite_scalar_quantizer = False,            # proposed in https://arxiv.org/abs/2309.15505, adapted for residual quantization
        input_channels = 1,
        discr_multi_scales = (1, 0.5, 0.25),
        stft_normalized = False,
        enc_cycle_dilations = (1, 3, 9),
        encoder_depthwise_separable_blocks = (),
        encoder_depthwise_separable_revision = 2,
        encoder_low_rank_pointwise_ranks = (),
        dec_cycle_dilations = (1, 3, 9),
        decoder_upsample_mode = 'convtranspose',
        decoder_residual_scale = 1.,
        decoder_block_residual_scales: tuple[float, ...] | None = None,
        decoder_linear_upsample_kernel_min = 0,
        decoder_interpolation_mode = 'linear',
        decoder_split_first_upsample = False,
        multi_spectral_window_powers_of_two = tuple(range(6, 12)),
        multi_spectral_n_ffts = 512,
        multi_spectral_n_mels = 64,
        recon_loss_weight = 1.,
        multi_spectral_recon_loss_weight = 1.,
        stft_recon_loss_weight = 0.,
        spectral_envelope_loss_weight = 0.,
        formant_peak_loss_weight = 0.,
        voiced_highband_loss_weight = 0.,
        upper_highband_loss_weight = 0.,
        active_spectral_detail_loss_weight = 0.,
        active_spectral_detail_band_weights = (
            0.50, 1.00, 1.00, 1.25, 1.50
        ),
        upper_highband_energy_deficit_weight = 0.,
        upper_highband_energy_margin_db = 0.50,
        voiced_highband_energy_deficit_weight = 0.35,
        voiced_highband_energy_margin_db = 0.10,
        voiced_hf_retention_loss_weight = 0.,
        voiced_hf_retention_margin_db = 0.50,
        si_sdr_loss_weight = 0.,
        correlation_loss_weight = 0.,
        energy_loss_weight = 0.1,
        click_loss_weight = 0.,
        jump_loss_weight = 0.,
        preemph_loss_weight = 0.,
        noise_floor_loss_weight = 0.,
        frame_phase_loss_weight = 0.,
        frame_phase_samples = 320,
        commitment_loss_weight = 0.1,
        adversarial_loss_weight = 1.,
        feature_loss_weight = 100,
        quantize_dropout_cutoff_index = 1,
        target_sample_hz = 16000,
        use_local_attn = True,
        attn_window_size = 128,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_depth = 1,
        attn_xpos_scale_base = None,
        attn_dynamic_pos_bias = False,
        use_gate_loop_layers = False,
        squeeze_excite = False,
        complex_stft_discr_logits_abs = False,
        pad_mode = 'reflect',
        stft_discriminator: Module | None = None,  # can pass in own stft discriminator
        complex_stft_discr_kwargs: dict = dict()
    ):
        super().__init__()

        # for autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        # rest of the class

        self.target_sample_hz = target_sample_hz # for resampling on the fly

        self.single_channel = input_channels == 1
        self.strides = strides
        encoder_depthwise_separable_blocks = tuple(
            int(index) for index in encoder_depthwise_separable_blocks
        )
        invalid_depthwise_blocks = tuple(
            index for index in encoder_depthwise_separable_blocks
            if index < 0 or index >= len(strides)
        )
        if invalid_depthwise_blocks:
            raise ValueError(
                'encoder_depthwise_separable_blocks contains invalid zero-based '
                f'indices {invalid_depthwise_blocks}; encoder has {len(strides)} blocks'
            )
        if len(set(encoder_depthwise_separable_blocks)) != len(
            encoder_depthwise_separable_blocks
        ):
            raise ValueError('encoder_depthwise_separable_blocks cannot contain duplicates')
        self.encoder_depthwise_separable_blocks = encoder_depthwise_separable_blocks
        encoder_depthwise_separable_revision = int(
            encoder_depthwise_separable_revision
        )
        if encoder_depthwise_separable_revision not in (2, 3):
            raise ValueError(
                'only Encoder DSCNN revisions 2 and 3 are supported'
            )
        low_rank_ranks = tuple(
            tuple(int(rank) for rank in ranks)
            for ranks in encoder_low_rank_pointwise_ranks
        )
        if low_rank_ranks and len(low_rank_ranks) != len(strides):
            raise ValueError(
                'encoder_low_rank_pointwise_ranks must contain one '
                f'(residual_rank, downsample_rank) pair per Encoder block; '
                f'expected {len(strides)}, got {len(low_rank_ranks)}'
            )
        if any(len(ranks) != 2 for ranks in low_rank_ranks):
            raise ValueError(
                'each encoder_low_rank_pointwise_ranks entry must contain '
                '(residual_rank, downsample_rank)'
            )
        if encoder_depthwise_separable_revision == 3:
            if not low_rank_ranks:
                raise ValueError(
                    'Encoder DSCNN revision 3 requires low-rank pointwise ranks'
                )
            for block_index in encoder_depthwise_separable_blocks:
                if min(low_rank_ranks[block_index]) <= 0:
                    raise ValueError(
                        'every revision-3 DSCNN block requires positive '
                        'residual and downsample pointwise ranks'
                    )
        elif low_rank_ranks and any(any(ranks) for ranks in low_rank_ranks):
            raise ValueError(
                'low-rank pointwise ranks require Encoder DSCNN revision 3'
            )
        self.encoder_depthwise_separable_revision = (
            encoder_depthwise_separable_revision
        )
        self.encoder_low_rank_pointwise_ranks = low_rank_ranks
        if decoder_upsample_mode not in ('convtranspose', 'linear'):
            raise ValueError(f'unknown decoder upsample mode: {decoder_upsample_mode}')
        if decoder_residual_scale < 0:
            raise ValueError(f'decoder_residual_scale must be >= 0, got {decoder_residual_scale}')
        if exists(decoder_block_residual_scales):
            if len(decoder_block_residual_scales) != len(strides):
                raise ValueError(
                    'decoder_block_residual_scales must contain one value per decoder block '
                    f'({len(strides)} expected, got {len(decoder_block_residual_scales)})'
                )
            if any(scale < 0 for scale in decoder_block_residual_scales):
                raise ValueError('decoder block residual scales must all be >= 0')
        if decoder_linear_upsample_kernel_min < 0:
            raise ValueError(f'decoder_linear_upsample_kernel_min must be >= 0, got {decoder_linear_upsample_kernel_min}')
        if decoder_interpolation_mode not in ('linear', 'cubic'):
            raise ValueError(
                f'unknown decoder interpolation mode: {decoder_interpolation_mode}'
            )
        if decoder_split_first_upsample:
            if decoder_upsample_mode != 'linear':
                raise ValueError(
                    'decoder_split_first_upsample requires decoder_upsample_mode="linear"'
                )
            if strides[-1] != 8:
                raise ValueError(
                    'decoder_split_first_upsample requires the first decoder '
                    f'upsampling stride to be 8, got {strides[-1]}'
                )
        self.decoder_upsample_mode = decoder_upsample_mode
        self.decoder_residual_scale = float(decoder_residual_scale)
        self.decoder_block_residual_scales = tuple(
            float(scale) for scale in default(
                decoder_block_residual_scales,
                (decoder_residual_scale,) * len(strides)
            )
        )
        self.decoder_linear_upsample_kernel_min = int(decoder_linear_upsample_kernel_min)
        self.decoder_interpolation_mode = decoder_interpolation_mode
        self.decoder_split_first_upsample = bool(decoder_split_first_upsample)

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for block_index, ((chan_in, chan_out), layer_stride) in enumerate(
            zip(chan_in_out_pairs, strides)
        ):
            residual_rank = None
            downsample_rank = None
            if (
                block_index in self.encoder_depthwise_separable_blocks and
                self.encoder_depthwise_separable_revision == 3
            ):
                residual_rank, downsample_rank = (
                    self.encoder_low_rank_pointwise_ranks[block_index]
                )
            encoder_blocks.append(EncoderBlock(
                chan_in,
                chan_out,
                layer_stride,
                enc_cycle_dilations,
                squeeze_excite,
                pad_mode,
                depthwise_separable=(
                    block_index in self.encoder_depthwise_separable_blocks
                ),
                residual_pointwise_rank=residual_rank,
                downsample_pointwise_rank=downsample_rank
            ))

            if use_gate_loop_layers:
                encoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_out, use_heinsen = False))))

        self.encoder = nn.Sequential(
            CausalConv1d(input_channels, channels, 7, pad_mode = pad_mode),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], codebook_dim, 3, pad_mode = pad_mode)
        )

        attn_kwargs = dict(
            dim = codebook_dim,
            dim_head = attn_dim_head,
            heads = attn_heads,
            depth = attn_depth,
            window_size = attn_window_size,
            xpos_scale_base = attn_xpos_scale_base,
            dynamic_pos_bias = attn_dynamic_pos_bias,
            prenorm = True,
            causal = True
        )

        self.encoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None

        self.encoder_film = FiLM(codebook_dim, dim_cond = 2)

        self.num_quantizers = rq_num_quantizers

        self.codebook_dim = codebook_dim

        self.rq_groups = rq_groups
        # Training-time codec mode.  This is intentionally part of the saved
        # model config (rather than a diagnostic-only forward) so reconstruction,
        # validation and GAN paths all see exactly the same latent signal.
        self.bypass_rvq = bool(bypass_rvq)

        assert not (use_lookup_free_quantizer and use_finite_scalar_quantizer)

        self.use_lookup_free_quantizer = use_lookup_free_quantizer
        self.use_finite_scalar_quantizer = use_finite_scalar_quantizer

        if use_lookup_free_quantizer:
            assert exists(codebook_size) and not exists(finite_scalar_quantizer_levels), 'if use_finite_scalar_quantizer is set to False, `codebook_size` must be set (and not `finite_scalar_quantizer_levels`)'

            self.rq = GroupedResidualLFQ(
                dim = codebook_dim,
                num_quantizers = rq_num_quantizers,
                codebook_size = codebook_size,
                groups = rq_groups,
                quantize_dropout = rq_quantize_dropout,
                quantize_dropout_cutoff_index = quantize_dropout_cutoff_index,
                **rq_kwargs
            )

            self.codebook_size = codebook_size

        elif use_finite_scalar_quantizer:
            assert not exists(codebook_size) and exists(finite_scalar_quantizer_levels), 'if use_finite_scalar_quantizer is set to True, `finite_scalar_quantizer_levels` must be set (and not `codebook_size`). the effective codebook size is the cumulative product of all the FSQ levels'

            self.rq = GroupedResidualFSQ(
                dim = codebook_dim,
                levels = finite_scalar_quantizer_levels,
                num_quantizers = rq_num_quantizers,
                groups = rq_groups,
                quantize_dropout = rq_quantize_dropout,
                quantize_dropout_cutoff_index = quantize_dropout_cutoff_index,
                **rq_kwargs
            )

            self.codebook_size = self.rq.codebook_size

        else:
            assert exists(codebook_size) and not exists(finite_scalar_quantizer_levels), 'if use_finite_scalar_quantizer is set to False, `codebook_size` must be set (and not `finite_scalar_quantizer_levels`)'
            self.rq = GroupedResidualVQ(
                dim = codebook_dim,
                num_quantizers = rq_num_quantizers,
                codebook_size = codebook_size,
                groups = rq_groups,
                decay = rq_ema_decay,
                commitment_weight = rq_commitment_weight,
                quantize_dropout_multiple_of = rq_quantize_dropout_multiple_of,
                kmeans_init = True,
                threshold_ema_dead_code = rq_threshold_ema_dead_code,
                quantize_dropout = rq_quantize_dropout,
                quantize_dropout_cutoff_index = quantize_dropout_cutoff_index,
                stochastic_sample_codes = rq_stochastic_sample_codes,
                rotation_trick = rq_rotation_trick,
                **rq_kwargs
            )

            self.codebook_size = codebook_size

        self.decoder_film = FiLM(codebook_dim, dim_cond = 2)

        self.decoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None

        decoder_blocks = []

        for block_index, (((chan_in, chan_out), layer_stride), residual_scale) in enumerate(zip(
            zip(reversed(chan_in_out_pairs), reversed(strides)),
            self.decoder_block_residual_scales
        )):
            decoder_blocks.append(DecoderBlock(
                chan_out,
                chan_in,
                layer_stride,
                dec_cycle_dilations,
                squeeze_excite,
                pad_mode,
                upsample_mode = decoder_upsample_mode,
                residual_scale = residual_scale,
                linear_upsample_kernel_min = decoder_linear_upsample_kernel_min,
                interpolation_mode = decoder_interpolation_mode,
                split_upsample = (
                    decoder_split_first_upsample and block_index == 0
                )
            ))

            if use_gate_loop_layers:
                decoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_in))))

        self.decoder = nn.Sequential(
            CausalConv1d(codebook_dim, layer_channels[-1], 7, pad_mode = pad_mode),
            *decoder_blocks,
            CausalConv1d(channels, input_channels, 7, pad_mode = pad_mode)
        )
        self._update_config(
            decoder_block_residual_scales = self.decoder_block_residual_scales
        )

        # discriminators

        self.discr_multi_scales = discr_multi_scales
        self.discriminators = ModuleList([MultiScaleDiscriminator() for _ in range(len(discr_multi_scales))])
        discr_rel_factors = [int(s1 / s2) for s1, s2 in zip(discr_multi_scales[:-1], discr_multi_scales[1:])]
        self.downsamples = ModuleList([nn.Identity()] + [nn.AvgPool1d(2 * factor, stride = factor, padding = factor) for factor in discr_rel_factors])

        self.stft_discriminator = stft_discriminator

        if not exists(self.stft_discriminator):
            self.stft_discriminator = ComplexSTFTDiscriminator(
                stft_normalized = stft_normalized,
                logits_abs = complex_stft_discr_logits_abs,
                **complex_stft_discr_kwargs
            )

        # multi spectral reconstruction

        self.mel_spec_transforms = ModuleList([])
        self.mel_spec_recon_alphas = []
        self.mel_spec_log_recon_alphas = []
        self.stft_recon_settings = []
        self.stft_recon_alphas = []

        # Stage-1 formant / vocal-tract envelope objective. This stays
        # separate from the raw multi-scale STFT loss: the frequency-smoothed,
        # voiced-only envelope does not reward individual narrow-band bins.
        self.spectral_envelope_n_fft = 1024
        self.spectral_envelope_win_length = 640
        self.spectral_envelope_hop_length = 160
        # Dual-scale real-cepstral envelopes: the 48-coefficient branch keeps
        # narrower formant shape, while the 32-coefficient branch suppresses
        # harmonic ripple.  This weakens the old single 9-bin smoothing
        # without removing the stable coarse-envelope reference.
        self.spectral_envelope_fine_lifter = 48
        self.spectral_envelope_coarse_lifter = 32
        self.spectral_envelope_fine_weight = 0.55
        self.spectral_envelope_coarse_weight = 0.45
        self.spectral_envelope_slope_weight = 0.35
        self.spectral_envelope_curvature_weight = 0.15
        self.spectral_envelope_min_hz = 200.
        self.spectral_envelope_max_hz = 4500.
        self.spectral_envelope_relative_rms_db = -35.
        self.spectral_envelope_absolute_rms = 0.003
        self.spectral_envelope_periodicity_threshold = 0.35
        self.spectral_envelope_f0_min_hz = 70.
        self.spectral_envelope_f0_max_hz = 400.
        self.formant_peak_softmax_temperature = 8.
        # Optimize the band that carries speech presence and consonant detail
        # more strongly than the upper "air" band.  A flat 3-7 kHz objective
        # left the k4-online checkpoint measurably dull, while uniformly
        # increasing the whole band risks learning broadband hiss.
        self.voiced_highband_loss_min_hz = 2500.
        self.voiced_highband_core_max_hz = 5500.
        self.voiced_highband_upper_weight = 0.35
        # Keep the original 3-7 kHz diagnostic definition so new validation
        # numbers remain comparable with earlier runs.
        self.voiced_highband_min_hz = 3000.
        self.voiced_highband_max_hz = min(7000., target_sample_hz / 2.)
        self.voiced_highband_reference_min_hz = 200.
        self.voiced_highband_slope_min_hz = 1000.
        # The 16 kHz model has only a narrow octave-edge band between 7 kHz
        # and Nyquist.  Keep it separate from the established 2.5-7 kHz
        # objective so a small, target-activity-gated term can recover real
        # upper-band detail without rewarding broadband hiss.  The last
        # 200 Hz is tapered to zero before Nyquist.
        self.upper_highband_min_hz = 7000.
        self.upper_highband_taper_start_hz = 7600.
        self.upper_highband_max_hz = min(7800., target_sample_hz / 2.)
        self.upper_highband_relative_power_floor_db = -30.
        self.upper_highband_reference_ratio_floor_db = -40.
        # A lightweight multi-resolution detail objective for target-supported
        # voiced bins.  Unlike the full MR-STFT loss, it ignores quiet bins and
        # short 64/128-sample windows, so it does not reward broadband hiss.
        self.active_spectral_detail_settings = (
            (256, 256, 64, 0.50),
            (512, 512, 128, 1.00),
            (1024, 1024, 256, 1.00),
            (2048, 2048, 512, 0.50),
        )
        self.active_spectral_detail_relative_db = -50.
        self.active_spectral_detail_min_hz = 200.
        self.active_spectral_detail_max_hz = min(
            7800.,
            target_sample_hz / 2.
        )
        if len(active_spectral_detail_band_weights) != 5:
            raise ValueError(
                'active_spectral_detail_band_weights must contain exactly '
                'five values for 200-1k, 1-3k, 3-5k, 5-7k, and 7-7.8k'
            )
        active_spectral_detail_band_weights = tuple(
            float(weight) for weight in active_spectral_detail_band_weights
        )
        if any(weight < 0. for weight in active_spectral_detail_band_weights):
            raise ValueError(
                'active_spectral_detail_band_weights must all be >= 0'
            )
        self.active_spectral_detail_band_weights = (
            active_spectral_detail_band_weights
        )
        self.active_spectral_detail_bands = tuple(
            (name, min_hz, max_hz, weight)
            for (name, min_hz, max_hz), weight in zip(
                (
                    ('200_1k', 200., 1000.),
                    ('1k_3k', 1000., 3000.),
                    ('3k_5k', 3000., 5000.),
                    ('5k_7k', 5000., 7000.),
                    ('7k_7p8k', 7000., 7800.),
                ),
                active_spectral_detail_band_weights,
            )
        )
        if voiced_highband_energy_deficit_weight < 0:
            raise ValueError('voiced_highband_energy_deficit_weight must be >= 0')
        if voiced_highband_energy_margin_db < 0:
            raise ValueError('voiced_highband_energy_margin_db must be >= 0')
        if upper_highband_energy_deficit_weight < 0:
            raise ValueError('upper_highband_energy_deficit_weight must be >= 0')
        if upper_highband_energy_margin_db < 0:
            raise ValueError('upper_highband_energy_margin_db must be >= 0')
        if voiced_hf_retention_loss_weight < 0:
            raise ValueError('voiced_hf_retention_loss_weight must be >= 0')
        if formant_peak_loss_weight < 0:
            raise ValueError('formant_peak_loss_weight must be >= 0')
        if voiced_hf_retention_margin_db < 0:
            raise ValueError('voiced_hf_retention_margin_db must be >= 0')
        self.voiced_highband_energy_deficit_weight = float(
            voiced_highband_energy_deficit_weight
        )
        self.voiced_highband_energy_margin_db = float(
            voiced_highband_energy_margin_db
        )
        self.upper_highband_energy_deficit_weight = float(
            upper_highband_energy_deficit_weight
        )
        self.upper_highband_energy_margin_db = float(
            upper_highband_energy_margin_db
        )
        self.voiced_hf_retention_loss_weight = float(
            voiced_hf_retention_loss_weight
        )
        self.voiced_hf_retention_margin_db = float(
            voiced_hf_retention_margin_db
        )
        self.register_buffer(
            'spectral_envelope_window',
            torch.hann_window(self.spectral_envelope_win_length),
            persistent = False
        )

        num_transforms = len(multi_spectral_window_powers_of_two)
        multi_spectral_n_ffts = cast_tuple(multi_spectral_n_ffts, num_transforms)
        multi_spectral_n_mels = cast_tuple(multi_spectral_n_mels, num_transforms)
        mel_weight_by_win_length = {
            64: 0.50,
            128: 0.75,
            256: 1.00,
            512: 1.00,
            1024: 1.00,
            2048: 0.75
        }
        # Short windows are auxiliary. Mid-scale STFTs dominate speech-formant
        # reconstruction and reduce pressure to fit high-frequency spikes.
        stft_weight_by_win_length = {
            64: 0.25,
            128: 0.50,
            256: 0.75,
            512: 1.00,
            1024: 1.00,
            2048: 0.75
        }

        for powers, n_fft, n_mels in zip_longest(multi_spectral_window_powers_of_two, multi_spectral_n_ffts, multi_spectral_n_mels):
            win_length = 2 ** powers
            alpha = mel_weight_by_win_length.get(win_length, 1.)
            stft_alpha = stft_weight_by_win_length.get(win_length, 1.)

            calculated_n_fft = default(max(n_fft, win_length), win_length)  # @AndreyBocharnikov said this is usually win length, but overridable

            # if any audio experts have an opinion about these settings, please submit a PR

            melspec_transform = T.MelSpectrogram(
                sample_rate = target_sample_hz,
                n_fft = calculated_n_fft,
                win_length = win_length,
                hop_length = win_length // 4,
                n_mels = n_mels,
                normalized = stft_normalized
            )

            self.mel_spec_transforms.append(melspec_transform)
            self.mel_spec_recon_alphas.append(alpha)
            # SoundStream gives the log-Mel branch more frequency-resolution
            # emphasis at long windows.  The final aggregation normalizes by
            # the sum of these weights, so its scale stays stable.
            self.mel_spec_log_recon_alphas.append(math.sqrt(win_length / 2))
            self.stft_recon_settings.append((
                calculated_n_fft,
                win_length,
                win_length // 4
            ))
            self.stft_recon_alphas.append(stft_alpha)

        # loss weights

        self.recon_loss_weight = recon_loss_weight
        self.multi_spectral_recon_loss_weight = multi_spectral_recon_loss_weight
        self.stft_recon_loss_weight = stft_recon_loss_weight
        self.spectral_envelope_loss_weight = spectral_envelope_loss_weight
        self.formant_peak_loss_weight = formant_peak_loss_weight
        self.voiced_highband_loss_weight = voiced_highband_loss_weight
        self.upper_highband_loss_weight = upper_highband_loss_weight
        self.active_spectral_detail_loss_weight = (
            active_spectral_detail_loss_weight
        )
        self.si_sdr_loss_weight = si_sdr_loss_weight
        self.correlation_loss_weight = correlation_loss_weight
        self.energy_loss_weight = energy_loss_weight
        self.click_loss_weight = click_loss_weight
        self.jump_loss_weight = jump_loss_weight
        self.preemph_loss_weight = preemph_loss_weight
        self.noise_floor_loss_weight = noise_floor_loss_weight
        self.frame_phase_loss_weight = frame_phase_loss_weight
        if frame_phase_samples <= 1:
            raise ValueError(f'frame_phase_samples must be > 1, got {frame_phase_samples}')
        self.frame_phase_samples = int(frame_phase_samples)
        self.commitment_loss_weight = commitment_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def configs(self):
        return pickle.loads(self._configs)

    def _update_config(self, **updates):
        config = self.configs
        config.update(updates)
        self._configs = pickle.dumps(config)

    def set_decoder_residual_scale(self, scale: float):
        scale = float(scale)
        if scale < 0:
            raise ValueError(f'decoder residual scale must be >= 0, got {scale}')

        self.decoder_residual_scale = scale

        for module in self.decoder.modules():
            if isinstance(module, Residual):
                module.set_residual_scale(scale)

        block_count = len(self.get_decoder_residual_blocks())
        self.decoder_block_residual_scales = (scale,) * block_count
        self._update_config(
            decoder_residual_scale = scale,
            decoder_block_residual_scales = self.decoder_block_residual_scales
        )

    def get_decoder_residual_blocks(self):
        """Return decoder upsampling blocks in temporal order (x8, x5, x4, x2)."""
        return tuple(
            module for module in self.decoder
            if isinstance(module, nn.Sequential) and
            any(isinstance(child, Residual) for child in module.modules())
        )

    def get_decoder_block_residual_scales(self):
        scales = []
        for block in self.get_decoder_residual_blocks():
            residuals = [module for module in block.modules() if isinstance(module, Residual)]
            scales.append(residuals[0].residual_scale if residuals else 1.)
        return tuple(scales)

    def set_decoder_block_residual_scales(self, scales):
        blocks = self.get_decoder_residual_blocks()
        scales = tuple(float(scale) for scale in scales)
        if len(scales) != len(blocks):
            raise ValueError(f'expected {len(blocks)} decoder block scales, got {len(scales)}')
        if any(scale < 0 for scale in scales):
            raise ValueError('decoder block residual scales must all be >= 0')

        for block, scale in zip(blocks, scales):
            for module in block.modules():
                if isinstance(module, Residual):
                    module.set_residual_scale(scale)

        self.decoder_block_residual_scales = scales
        self._update_config(decoder_block_residual_scales = scales)

    def restore_decoder_runtime_state(self, config):
        """Restore non-parameter decoder controls stored in a checkpoint config."""
        block_scales = config.get('decoder_block_residual_scales')
        if exists(block_scales):
            self.set_decoder_block_residual_scales(block_scales)
            return self.decoder_block_residual_scales

        if 'decoder_residual_scale' in config:
            self.set_decoder_residual_scale(config['decoder_residual_scale'])
            return self.decoder_block_residual_scales

        return self.get_decoder_block_residual_scales()

    def frame_phase_residual_loss(self, target, recon):
        """Penalize *additional* codec-frame periodicity in the residual.

        The loss is evaluated on ``recon - target``, rather than on the
        reconstruction alone.  Real 50 Hz structure already present in the
        source therefore cancels when it is reconstructed correctly.  Besides
        the mean phase pattern, use differentiable counterparts of the fixed
        validation diagnostics: the isolated lag-320 autocorrelation peak and
        50 Hz comb-line excess over adjacent FFT bins.
        """
        frame_samples = self.frame_phase_samples
        usable_samples = (min(target.shape[-1], recon.shape[-1]) // frame_samples) * frame_samples
        if usable_samples < frame_samples * 2:
            return self.zero

        residual = (recon[..., :usable_samples] - target[..., :usable_samples]).float()
        residual_frames = residual.reshape(*residual.shape[:-1], -1, frame_samples)
        phase_pattern = residual_frames.mean(dim = -2)
        phase_pattern = phase_pattern - phase_pattern.mean(dim = -1, keepdim = True)
        phase_rms = phase_pattern.square().mean(dim = -1).clamp_min(1e-12).sqrt()
        # Quiet segments must not turn a tiny absolute phase residual into an
        # unbounded relative loss.  -60 dBFS is a conservative denominator
        # floor; the separate quiet-noise loss still supervises those regions.
        target_rms = (
            target[..., :usable_samples]
            .float()
            .square()
            .mean(dim = -1)
            .sqrt()
            .clamp_min(1e-3)
        )
        phase_pattern_loss = (phase_rms / target_rms).mean()

        # Match the validation AC320 diagnostic: first-difference the
        # residual, remove DC, then measure how much the lag-320 correlation
        # exceeds the average of its immediate neighbours.  A small positive
        # margin avoids optimizing harmless numerical fluctuations.
        residual_rows = residual.reshape(-1, usable_samples)
        residual_hp = residual_rows[..., 1:] - residual_rows[..., :-1]
        residual_hp = residual_hp - residual_hp.mean(dim = -1, keepdim = True)

        def normalized_ac(lag):
            left = residual_hp[..., :-lag]
            right = residual_hp[..., lag:]
            numerator = (left * right).sum(dim = -1)
            denominator = (
                left.square().sum(dim = -1) *
                right.square().sum(dim = -1)
            ).clamp_min(1e-12).sqrt()
            return numerator / denominator

        ac_319 = normalized_ac(frame_samples - 1)
        ac_320 = normalized_ac(frame_samples)
        ac_321 = normalized_ac(frame_samples + 1)
        ac_320_isolated = ac_320 - 0.5 * (ac_319 + ac_321)
        ac_excess_loss = F.relu(ac_320_isolated - 0.01).mean()

        # FFT-bin spacing is sample_rate / usable_samples.  Advancing by the
        # number of codec frames therefore advances exactly one codec-frame
        # rate (50 Hz for 16 kHz / 320 samples).  Penalize only positive line
        # excess above a 1 dB margin and normalize the dB scale so this remains
        # an auxiliary term.
        comb_excess_loss = self.zero
        if usable_samples >= frame_samples * 4:
            spectrum_db = 20. * torch.log10(
                torch.fft.rfft(residual_rows, dim = -1).abs().clamp_min(1e-8)
            )
            frame_count = usable_samples // frame_samples
            harmonic_bins = torch.arange(
                frame_count,
                spectrum_db.shape[-1] - 1,
                frame_count,
                device = spectrum_db.device,
            )
            if harmonic_bins.numel() > 0:
                line_db = spectrum_db[:, harmonic_bins]
                neighbour_db = 0.5 * (
                    spectrum_db[:, harmonic_bins - 1] +
                    spectrum_db[:, harmonic_bins + 1]
                )
                comb_excess_loss = (
                    F.relu(line_db - neighbour_db - 1.).mean() / 6.
                )

        return (
            phase_pattern_loss +
            0.5 * ac_excess_loss +
            0.1 * comb_excess_loss
        )

    def reconstruction_losses(self, target, recon):
        wave_l1 = F.l1_loss(recon, target)
        wave_mse = F.mse_loss(recon, target)
        target_rms = target.square().mean(dim = -1).clamp_min(1e-8).sqrt()
        recon_rms = recon.square().mean(dim = -1).clamp_min(1e-8).sqrt()
        energy_loss = F.l1_loss(recon_rms, target_rms)
        target_centered = target - target.mean(dim = -1, keepdim = True)
        recon_centered = recon - recon.mean(dim = -1, keepdim = True)
        signed_correlation = (
            (target_centered * recon_centered).sum(dim = -1) /
            (
                target_centered.norm(dim = -1) *
                recon_centered.norm(dim = -1)
            ).clamp_min(1e-8)
        )
        correlation_loss = (1. - signed_correlation).mean()
        wave_loss = (
            wave_l1 +
            0.3 * wave_mse +
            self.energy_loss_weight * energy_loss
        )

        linear_spectral_losses = []
        linear_spectral_weights = []
        log_spectral_losses = []
        log_spectral_weights = []
        for mel_transform, linear_alpha, log_alpha in zip(
            self.mel_spec_transforms,
            self.mel_spec_recon_alphas,
            self.mel_spec_log_recon_alphas
        ):
            target_mel, recon_mel = map(mel_transform, (target, recon))
            linear_loss = F.l1_loss(recon_mel, target_mel)
            log_loss = F.mse_loss(
                log(recon_mel, eps = 1e-5),
                log(target_mel, eps = 1e-5)
            )
            linear_spectral_losses.append(0.8 * linear_alpha * linear_loss)
            linear_spectral_weights.append(linear_alpha)
            log_spectral_losses.append(log_alpha * log_loss)
            log_spectral_weights.append(log_alpha)

        linear_spectral_loss = (
            torch.stack(linear_spectral_losses).sum() /
            max(sum(linear_spectral_weights), 1e-8)
            if linear_spectral_losses
            else self.zero
        )
        log_spectral_loss = (
            torch.stack(log_spectral_losses).sum() /
            max(sum(log_spectral_weights), 1e-8)
            if log_spectral_losses
            else self.zero
        )
        spectral_loss = linear_spectral_loss + 0.5 * log_spectral_loss

        stft_loss = self.zero
        self._last_stft_scale_losses = {}
        if self.stft_recon_loss_weight > 0:
            stft_losses = []
            stft_weights = []
            target_for_stft = target.float()
            recon_for_stft = recon.float()
            if target_for_stft.ndim == 2:
                target_for_stft = rearrange(target_for_stft, 'b n -> b 1 n')
            if recon_for_stft.ndim == 2:
                recon_for_stft = rearrange(recon_for_stft, 'b n -> b 1 n')

            for (n_fft, win_length, hop_length), alpha in zip(
                self.stft_recon_settings,
                self.stft_recon_alphas
            ):
                min_samples = n_fft
                target_padded = target_for_stft
                recon_padded = recon_for_stft
                if target_padded.shape[-1] < min_samples:
                    padding = min_samples - target_padded.shape[-1]
                    target_padded = F.pad(target_padded, (0, padding))
                    recon_padded = F.pad(recon_padded, (0, padding))

                target_rows = rearrange(target_padded, 'b c n -> (b c) n')
                recon_rows = rearrange(recon_padded, 'b c n -> (b c) n')
                window = torch.hann_window(
                    win_length,
                    device = target_rows.device,
                    dtype = torch.float32
                )
                target_stft = torch.stft(
                    target_rows,
                    n_fft,
                    hop_length = hop_length,
                    win_length = win_length,
                    window = window,
                    return_complex = True
                )
                recon_stft = torch.stft(
                    recon_rows,
                    n_fft,
                    hop_length = hop_length,
                    win_length = win_length,
                    window = window,
                    return_complex = True
                )
                target_mag = target_stft.abs()
                recon_mag = recon_stft.abs()
                diff_norm = torch.linalg.norm(
                    target_mag - recon_mag,
                    dim = (-2, -1)
                )
                target_norm = torch.linalg.norm(
                    target_mag,
                    dim = (-2, -1)
                ).clamp_min(1e-8)
                spectral_convergence = (diff_norm / target_norm).mean()
                log_mag_loss = F.l1_loss(
                    log(recon_mag, eps = 1e-5),
                    log(target_mag, eps = 1e-5)
                )
                one_stft_loss = (
                    0.5 * spectral_convergence +
                    1.0 * log_mag_loss
                )
                self._last_stft_scale_losses[str(win_length)] = one_stft_loss
                stft_losses.append(alpha * one_stft_loss)
                stft_weights.append(alpha)

            stft_loss = (
                torch.stack(stft_losses).sum() /
                max(sum(stft_weights), 1e-8)
                if stft_losses
                else self.zero
            )

        return wave_loss, spectral_loss, stft_loss, correlation_loss

    def si_sdr_loss(self, target, recon):
        projection_scale = (
            (recon * target).sum(dim = -1, keepdim = True) /
            target.square().sum(dim = -1, keepdim = True).clamp_min(1e-8)
        )
        projected = projection_scale * target
        residual = recon - projected
        si_sdr = 10 * torch.log10(
            projected.square().sum(dim = -1).clamp_min(1e-8) /
            residual.square().sum(dim = -1).clamp_min(1e-8)
        )
        return -si_sdr.mean()

    def spectral_envelope_metrics(self, target, recon):
        """Return periodic-voiced, dual-cepstral vocal-tract envelope errors."""
        target_rows = target.float()
        recon_rows = recon.float()
        if target_rows.ndim == 3:
            target_rows = rearrange(target_rows, 'b c n -> (b c) n')
        if recon_rows.ndim == 3:
            recon_rows = rearrange(recon_rows, 'b c n -> (b c) n')

        num_samples = min(target_rows.shape[-1], recon_rows.shape[-1])
        target_rows = target_rows[..., :num_samples]
        recon_rows = recon_rows[..., :num_samples]
        if num_samples < self.spectral_envelope_n_fft:
            padding = self.spectral_envelope_n_fft - num_samples
            target_rows = F.pad(target_rows, (0, padding))
            recon_rows = F.pad(recon_rows, (0, padding))

        window = self.spectral_envelope_window.to(
            device = target_rows.device,
            dtype = target_rows.dtype
        )
        stft_kwargs = dict(
            n_fft = self.spectral_envelope_n_fft,
            hop_length = self.spectral_envelope_hop_length,
            win_length = self.spectral_envelope_win_length,
            window = window,
            center = True,
            pad_mode = 'constant',
            return_complex = True
        )
        target_log_mag = log(
            torch.stft(target_rows, **stft_kwargs).abs(),
            eps = 1e-5
        )
        recon_log_mag = log(
            torch.stft(recon_rows, **stft_kwargs).abs(),
            eps = 1e-5
        )

        def cepstral_envelope(log_magnitude, lifter_order):
            cepstrum = torch.fft.irfft(
                log_magnitude,
                n = self.spectral_envelope_n_fft,
                dim = 1
            )
            liftered = torch.zeros_like(cepstrum)
            liftered[:, :lifter_order + 1] = cepstrum[:, :lifter_order + 1]
            liftered[:, -lifter_order:] = cepstrum[:, -lifter_order:]
            return torch.fft.rfft(liftered, dim = 1).real

        target_fine = cepstral_envelope(
            target_log_mag, self.spectral_envelope_fine_lifter
        )
        recon_fine = cepstral_envelope(
            recon_log_mag, self.spectral_envelope_fine_lifter
        )
        target_coarse = cepstral_envelope(
            target_log_mag, self.spectral_envelope_coarse_lifter
        )
        recon_coarse = cepstral_envelope(
            recon_log_mag, self.spectral_envelope_coarse_lifter
        )

        frequencies = torch.fft.rfftfreq(
            self.spectral_envelope_n_fft,
            d = 1. / self.target_sample_hz,
            device = target_rows.device
        )
        voice_band = (
            (frequencies >= self.spectral_envelope_min_hz) &
            (frequencies <= self.spectral_envelope_max_hz)
        )
        target_fine = target_fine[:, voice_band]
        recon_fine = recon_fine[:, voice_band]
        target_coarse = target_coarse[:, voice_band]
        recon_coarse = recon_coarse[:, voice_band]

        def remove_frame_gain(envelope):
            return envelope - envelope.mean(dim = 1, keepdim = True)

        target_fine = remove_frame_gain(target_fine)
        recon_fine = remove_frame_gain(recon_fine)
        target_coarse = remove_frame_gain(target_coarse)
        recon_coarse = remove_frame_gain(recon_coarse)

        # Match torch.stft(center=True) frame centers exactly, then compute a
        # normalized target autocorrelation.  This rejects energetic unvoiced
        # fricatives that the previous RMS-only mask mislabeled as voiced.
        with torch.no_grad():
            padded = F.pad(
                target_rows,
                (self.spectral_envelope_n_fft // 2,) * 2
            )
            frames = padded.unfold(
                -1,
                self.spectral_envelope_n_fft,
                self.spectral_envelope_hop_length
            )
            crop_start = (
                self.spectral_envelope_n_fft -
                self.spectral_envelope_win_length
            ) // 2
            frames = frames[..., crop_start:crop_start + self.spectral_envelope_win_length]
            frame_rms = frames.square().mean(dim = -1).clamp_min(1e-12).sqrt()
            centered_frames = frames - frames.mean(dim = -1, keepdim = True)
            fft_size = 2 * self.spectral_envelope_win_length
            autocorrelation = torch.fft.irfft(
                torch.fft.rfft(centered_frames, n = fft_size).abs().square(),
                n = fft_size
            )[..., :self.spectral_envelope_win_length]
            autocorrelation = autocorrelation / autocorrelation[..., :1].clamp_min(1e-8)
            min_lag = max(1, int(self.target_sample_hz / self.spectral_envelope_f0_max_hz))
            max_lag = min(
                self.spectral_envelope_win_length - 1,
                int(self.target_sample_hz / self.spectral_envelope_f0_min_hz)
            )
            periodicity = autocorrelation[..., min_lag:max_lag + 1].amax(dim = -1)

        num_frames = min(
            frame_rms.shape[-1],
            target_fine.shape[-1],
            target_coarse.shape[-1]
        )
        frame_rms = frame_rms[..., :num_frames]
        periodicity = periodicity[..., :num_frames]
        target_fine = target_fine[..., :num_frames]
        recon_fine = recon_fine[..., :num_frames]
        target_coarse = target_coarse[..., :num_frames]
        recon_coarse = recon_coarse[..., :num_frames]
        relative_floor = (
            frame_rms.amax(dim = -1, keepdim = True) *
            (10. ** (self.spectral_envelope_relative_rms_db / 20.))
        )
        voiced_mask = (
            (frame_rms >= relative_floor) &
            (frame_rms >= self.spectral_envelope_absolute_rms) &
            (periodicity >= self.spectral_envelope_periodicity_threshold)
        )
        voiced_weight = voiced_mask.unsqueeze(1).to(target_fine.dtype)

        voice_frequencies = frequencies[voice_band]
        region_weights = torch.ones_like(voice_frequencies)
        region_weights = torch.where(
            (voice_frequencies >= 1000.) & (voice_frequencies < 2500.),
            region_weights.new_tensor(1.30), region_weights
        )
        region_weights = torch.where(
            voice_frequencies >= 2500.,
            region_weights.new_tensor(1.15), region_weights
        )

        def masked_band_mean(error, band_mask, use_region_weights = False):
            band_error = error[:, band_mask]
            available_region_weights = region_weights[:error.shape[1]]
            frequency_weight = (
                available_region_weights[band_mask].view(1, -1, 1)
                if use_region_weights
                else band_error.new_ones((1, band_error.shape[1], 1))
            )
            denominator = (
                voiced_weight.sum() * frequency_weight.sum()
            ).clamp_min(1.)
            return (band_error * voiced_weight * frequency_weight).sum() / denominator

        all_frequencies = torch.ones_like(voice_frequencies, dtype = torch.bool)

        def envelope_objective(target_envelope, recon_envelope):
            value_error = (recon_envelope - target_envelope).abs()
            slope_error = (
                recon_envelope.diff(dim = 1) -
                target_envelope.diff(dim = 1)
            ).abs()
            curvature_error = (
                recon_envelope.diff(n = 2, dim = 1) -
                target_envelope.diff(n = 2, dim = 1)
            ).abs()
            value_loss = masked_band_mean(value_error, all_frequencies, True)
            slope_loss = masked_band_mean(
                slope_error,
                all_frequencies[:-1],
                True
            )
            curvature_loss = masked_band_mean(
                curvature_error,
                all_frequencies[:-2],
                True
            )
            return (
                value_loss +
                self.spectral_envelope_slope_weight * slope_loss +
                self.spectral_envelope_curvature_weight * curvature_loss,
                value_error,
                value_loss,
                slope_loss,
                curvature_loss,
            )

        fine_loss, fine_error, fine_value, fine_slope, fine_curvature = envelope_objective(
            target_fine, recon_fine
        )
        coarse_loss, coarse_error, coarse_value, coarse_slope, coarse_curvature = envelope_objective(
            target_coarse, recon_coarse
        )
        total_loss = (
            self.spectral_envelope_fine_weight * fine_loss +
            self.spectral_envelope_coarse_weight * coarse_loss
        )
        combined_error = (
            self.spectral_envelope_fine_weight * fine_error +
            self.spectral_envelope_coarse_weight * coarse_error
        )
        low_loss = masked_band_mean(
            combined_error,
            (voice_frequencies >= 200.) & (voice_frequencies < 1000.)
        )
        mid_loss = masked_band_mean(
            combined_error,
            (voice_frequencies >= 1000.) & (voice_frequencies < 2500.)
        )
        high_loss = masked_band_mean(
            combined_error,
            (voice_frequencies >= 2500.) & (voice_frequencies <= 4500.)
        )
        voiced_fraction = voiced_mask.float().mean()

        # Stage 1 of direct F1/F2/F3 supervision is always-on diagnostics.
        # Stage 2 is activated only when formant_peak_loss_weight is ramped by
        # the trainer.  Target peaks and confidence masks are detached.
        formant_bands = (
            ('f1', 200., 1000., 1.0),
            ('f2', 1000., 2500., 1.3),
            ('f3', 2500., 4500., 0.8),
        )
        peak_loss_terms = []
        peak_weights = []
        peak_diagnostics = {}
        for name, minimum_hz, maximum_hz, formant_weight in formant_bands:
            band_mask = (
                (voice_frequencies >= minimum_hz) &
                (voice_frequencies < maximum_hz)
            )
            band_frequencies = voice_frequencies[band_mask].view(1, -1, 1)
            target_band = target_fine[:, band_mask]
            recon_band = recon_fine[:, band_mask]
            target_distribution = torch.softmax(
                self.formant_peak_softmax_temperature * target_band.detach(),
                dim = 1
            )
            recon_distribution = torch.softmax(
                self.formant_peak_softmax_temperature * recon_band,
                dim = 1
            )
            target_position = (target_distribution * band_frequencies).sum(dim = 1)
            recon_position = (recon_distribution * band_frequencies).sum(dim = 1)
            prominence = (
                target_band.detach().amax(dim = 1) -
                target_band.detach().mean(dim = 1)
            )
            confidence = (prominence >= 0.10) & voiced_mask
            confidence_weight = confidence.to(target_band.dtype)
            denominator = confidence_weight.sum().clamp_min(1.)
            position_error_hz = (recon_position - target_position).abs()
            position_loss = (
                position_error_hz * confidence_weight
            ).sum() / denominator / max(maximum_hz - minimum_hz, 1.)
            target_amplitude = (target_distribution * target_band.detach()).sum(dim = 1)
            recon_amplitude = (target_distribution * recon_band).sum(dim = 1)
            amplitude_loss = (
                (recon_amplitude - target_amplitude).abs() * confidence_weight
            ).sum() / denominator
            target_shape = target_band.detach() - target_band.detach().mean(dim = 1, keepdim = True)
            recon_shape = recon_band - recon_band.mean(dim = 1, keepdim = True)
            shape_error = (recon_shape - target_shape).abs().mean(dim = 1)
            shape_loss = (shape_error * confidence_weight).sum() / denominator
            one_peak_loss = position_loss + 0.30 * amplitude_loss + 0.30 * shape_loss
            peak_loss_terms.append(formant_weight * one_peak_loss)
            peak_weights.append(formant_weight)
            peak_diagnostics[f'{name}_mae_hz'] = (
                position_error_hz * confidence_weight
            ).sum() / denominator
            peak_diagnostics[f'{name}_valid_fraction'] = confidence.float().mean()

        formant_peak_loss = sum(peak_loss_terms) / max(sum(peak_weights), 1e-8)
        self._last_formant_metrics = {
            'fine_loss': fine_loss,
            'coarse_loss': coarse_loss,
            'fine_value': fine_value,
            'fine_slope': fine_slope,
            'fine_curvature': fine_curvature,
            'coarse_value': coarse_value,
            'coarse_slope': coarse_slope,
            'coarse_curvature': coarse_curvature,
            'periodicity_mean': (
                (periodicity * voiced_mask.float()).sum() /
                voiced_mask.float().sum().clamp_min(1.)
            ),
            'voiced_fraction': voiced_fraction,
            'peak_loss': formant_peak_loss,
            **peak_diagnostics,
        }
        return total_loss, low_loss, mid_loss, high_loss, voiced_fraction

    def voiced_highband_metrics(self, target, recon):
        """Measure voiced high-frequency detail without rewarding noise.

        Voicing is selected only from target-frame RMS.  Per-frame log spectra
        are normalized by their 200-7000 Hz mean before the high-band error is
        computed, so this objective follows harmonic/formant detail rather than
        encouraging a broadband gain increase.  A small one-sided term only
        penalizes high-band power that remains more than the configured margin
        below the target; excess high-frequency power receives no reward.  The
        remaining values are diagnostics and do not participate in checkpoint
        gates by default.  A separate target-active 7-7.8 kHz objective uses a
        cosine taper above 7.6 kHz and reports voiced log-magnitude error,
        voiced energy ratio, and quiet-frame excess independently.
        """
        target_rows = target.float()
        recon_rows = recon.float()
        if target_rows.ndim == 3:
            target_rows = rearrange(target_rows, 'b c n -> (b c) n')
        if recon_rows.ndim == 3:
            recon_rows = rearrange(recon_rows, 'b c n -> (b c) n')

        num_samples = min(target_rows.shape[-1], recon_rows.shape[-1])
        target_rows = target_rows[..., :num_samples]
        recon_rows = recon_rows[..., :num_samples]
        if num_samples < self.spectral_envelope_n_fft:
            padding = self.spectral_envelope_n_fft - num_samples
            target_rows = F.pad(target_rows, (0, padding))
            recon_rows = F.pad(recon_rows, (0, padding))

        window = self.spectral_envelope_window.to(
            device = target_rows.device,
            dtype = target_rows.dtype
        )
        stft_kwargs = dict(
            n_fft = self.spectral_envelope_n_fft,
            hop_length = self.spectral_envelope_hop_length,
            win_length = self.spectral_envelope_win_length,
            window = window,
            center = True,
            pad_mode = 'constant',
            return_complex = True
        )
        target_mag = torch.stft(target_rows, **stft_kwargs).abs()
        recon_mag = torch.stft(recon_rows, **stft_kwargs).abs()
        target_log_mag = log(target_mag, eps = 1e-5)
        recon_log_mag = log(recon_mag, eps = 1e-5)

        frequencies = torch.fft.rfftfreq(
            self.spectral_envelope_n_fft,
            d = 1. / self.target_sample_hz,
            device = target_rows.device
        )
        reference_band = (
            (frequencies >= self.voiced_highband_reference_min_hz) &
            (frequencies <= self.voiced_highband_max_hz)
        )
        highband = (
            (frequencies >= self.voiced_highband_min_hz) &
            (frequencies <= self.voiced_highband_max_hz)
        )
        loss_highband = (
            (frequencies >= self.voiced_highband_loss_min_hz) &
            (frequencies <= self.voiced_highband_max_hz)
        )
        slope_band = (
            (frequencies >= self.voiced_highband_slope_min_hz) &
            (frequencies <= self.voiced_highband_max_hz)
        )
        upper_highband = (
            (frequencies > self.upper_highband_min_hz) &
            (frequencies <= self.upper_highband_max_hz)
        )
        if (
            not reference_band.any() or
            not highband.any() or
            not loss_highband.any() or
            not slope_band.any()
        ):
            return (
                self.zero, self.zero, self.zero,
                self.zero, self.zero, self.zero, self.zero,
                self.zero, self.zero, self.zero, self.zero, self.zero
            )

        frame_rms = F.avg_pool1d(
            target_rows.square().unsqueeze(1),
            kernel_size = self.spectral_envelope_win_length,
            stride = self.spectral_envelope_hop_length,
            padding = self.spectral_envelope_win_length // 2,
            count_include_pad = False
        ).squeeze(1).clamp_min(1e-12).sqrt()
        num_frames = min(frame_rms.shape[-1], target_mag.shape[-1], recon_mag.shape[-1])
        frame_rms = frame_rms[..., :num_frames]
        target_mag = target_mag[..., :num_frames]
        recon_mag = recon_mag[..., :num_frames]
        target_log_mag = target_log_mag[..., :num_frames]
        recon_log_mag = recon_log_mag[..., :num_frames]
        relative_floor = (
            frame_rms.amax(dim = -1, keepdim = True) *
            (10. ** (self.spectral_envelope_relative_rms_db / 20.))
        )
        voiced_mask = (
            (frame_rms >= relative_floor) &
            (frame_rms >= self.spectral_envelope_absolute_rms)
        )
        voiced_weight = voiced_mask.to(target_mag.dtype)
        voiced_count = voiced_weight.sum().clamp_min(1.)
        quiet_threshold = torch.quantile(
            frame_rms.detach(), 0.30, dim = -1, keepdim = True
        ).clamp_max(0.03)
        quiet_weight = (frame_rms <= quiet_threshold).to(target_mag.dtype)
        quiet_count = quiet_weight.sum().clamp_min(1.)

        # Normalize each frame with the same broad speech-band statistic.  A
        # lower loss therefore requires structured high-band agreement rather
        # than merely adding energy above 3 kHz.
        target_frame_gain = target_log_mag[:, reference_band].mean(dim = 1, keepdim = True)
        recon_frame_gain = recon_log_mag[:, reference_band].mean(dim = 1, keepdim = True)
        target_highband = target_log_mag[:, loss_highband] - target_frame_gain
        recon_highband = recon_log_mag[:, loss_highband] - recon_frame_gain
        highband_error = (recon_highband - target_highband).abs()
        loss_highband_frequencies = frequencies[loss_highband]
        highband_bin_weights = torch.where(
            loss_highband_frequencies <= self.voiced_highband_core_max_hz,
            torch.ones_like(loss_highband_frequencies),
            torch.full_like(
                loss_highband_frequencies,
                self.voiced_highband_upper_weight
            )
        )
        highband_weight_sum = highband_bin_weights.sum().clamp_min(1e-8)
        voiced_highband_logmag_error = (
            highband_error *
            highband_bin_weights[None, :, None] *
            voiced_weight.unsqueeze(1)
        ).sum() / (voiced_count * highband_weight_sum)

        # The one-sided training penalty follows the same clarity weighting.
        # Diagnostics below deliberately retain the historical flat 3-7 kHz
        # definition for apples-to-apples comparisons.
        target_loss_hf_power = (
            target_mag[:, loss_highband].square() *
            highband_bin_weights[None, :, None]
        ).sum(dim = 1) / highband_weight_sum
        recon_loss_hf_power = (
            recon_mag[:, loss_highband].square() *
            highband_bin_weights[None, :, None]
        ).sum(dim = 1) / highband_weight_sum

        upper_highband_loss = self.zero
        voiced_upper_logmag_error = self.zero
        voiced_upper_energy_deficit = self.zero
        voiced_upper_energy_ratio_db = self.zero
        quiet_upper_excess_db = self.zero
        if upper_highband.any():
            upper_frequencies = frequencies[upper_highband]
            upper_bin_weights = torch.ones_like(upper_frequencies)
            taper_mask = upper_frequencies > self.upper_highband_taper_start_hz
            if taper_mask.any():
                taper_position = (
                    (
                        upper_frequencies[taper_mask] -
                        self.upper_highband_taper_start_hz
                    ) /
                    max(
                        self.upper_highband_max_hz -
                        self.upper_highband_taper_start_hz,
                        1.
                    )
                ).clamp(0., 1.)
                upper_bin_weights[taper_mask] = (
                    0.5 * (1. + torch.cos(math.pi * taper_position))
                )
            upper_weight_sum = upper_bin_weights.sum().clamp_min(1e-8)

            target_upper_power = (
                target_mag[:, upper_highband].square() *
                upper_bin_weights[None, :, None]
            ).sum(dim = 1) / upper_weight_sum
            recon_upper_power = (
                recon_mag[:, upper_highband].square() *
                upper_bin_weights[None, :, None]
            ).sum(dim = 1) / upper_weight_sum
            target_reference_power = (
                target_mag[:, reference_band].square().mean(dim = 1)
            )

            relative_upper_floor = (
                target_upper_power.amax(dim = -1, keepdim = True) *
                10. ** (
                    self.upper_highband_relative_power_floor_db / 10.
                )
            )
            reference_ratio_floor = 10. ** (
                self.upper_highband_reference_ratio_floor_db / 10.
            )
            upper_active_mask = (
                voiced_mask &
                (target_upper_power >= relative_upper_floor) &
                (
                    target_upper_power >=
                    target_reference_power * reference_ratio_floor
                )
            )
            upper_active_weight = upper_active_mask.to(target_mag.dtype)
            upper_active_count = upper_active_weight.sum().clamp_min(1.)

            target_upper_logmag = (
                target_log_mag[:, upper_highband] - target_frame_gain
            )
            recon_upper_logmag = (
                recon_log_mag[:, upper_highband] - recon_frame_gain
            )
            upper_logmag_error = (
                target_upper_logmag - recon_upper_logmag
            ).abs()
            voiced_upper_logmag_error = (
                upper_logmag_error *
                upper_bin_weights[None, :, None] *
                upper_active_weight.unsqueeze(1)
            ).sum() / (upper_active_count * upper_weight_sum)

            upper_ratio_db = 10. * torch.log10(
                (recon_upper_power + 1e-10) /
                (target_upper_power + 1e-10)
            )
            upper_deficit_db = F.relu(
                -self.upper_highband_energy_margin_db - upper_ratio_db
            )
            voiced_upper_energy_deficit = (
                upper_deficit_db.square() * upper_active_weight
            ).sum() / upper_active_count
            upper_highband_loss = (
                voiced_upper_logmag_error +
                self.upper_highband_energy_deficit_weight *
                voiced_upper_energy_deficit
            )
            voiced_upper_energy_ratio_db = (
                upper_ratio_db.clamp(-40., 40.) * upper_active_weight
            ).sum() / upper_active_count
            quiet_upper_excess_db = (
                F.relu(upper_ratio_db) * quiet_weight
            ).sum() / quiet_count

        target_hf_power = target_mag[:, highband].square().mean(dim = 1)
        recon_hf_power = recon_mag[:, highband].square().mean(dim = 1)
        log_power_deficit = F.relu(
            torch.log(target_loss_hf_power + 1e-10) -
            torch.log(recon_loss_hf_power + 1e-10) -
            self.voiced_highband_energy_margin_db * math.log(10.) / 10.
        )
        voiced_hf_energy_deficit = (
            log_power_deficit * voiced_weight
        ).sum() / voiced_count
        voiced_highband_loss = (
            voiced_highband_logmag_error +
            self.voiced_highband_energy_deficit_weight *
            voiced_hf_energy_deficit
        )
        hf_ratio_db = 10. * torch.log10(
            (recon_hf_power + 1e-10) / (target_hf_power + 1e-10)
        )
        # Gate-aligned, target-voiced 3-7 kHz retention objective.  It only
        # penalizes a reconstructed power deficit beyond the allowed margin;
        # excess high-frequency power receives no reward.  Squaring in dB
        # gives the gate boundary a smooth zero-gradient side while strongly
        # discouraging the persistent voiced-HF loss seen in Stage 2.
        voiced_hf_retention_loss = (
            F.relu(-hf_ratio_db - self.voiced_hf_retention_margin_db).square() *
            voiced_weight
        ).sum() / voiced_count
        voiced_hf_energy_ratio_db = (
            hf_ratio_db.clamp(-40., 40.) * voiced_weight
        ).sum() / voiced_count

        reference_frequencies = frequencies[reference_band]
        target_reference_mag = target_mag[:, reference_band]
        recon_reference_mag = recon_mag[:, reference_band]
        target_centroid = (
            target_reference_mag * reference_frequencies[None, :, None]
        ).sum(dim = 1) / target_reference_mag.sum(dim = 1).clamp_min(1e-8)
        recon_centroid = (
            recon_reference_mag * reference_frequencies[None, :, None]
        ).sum(dim = 1) / recon_reference_mag.sum(dim = 1).clamp_min(1e-8)
        spectral_centroid_delta_hz = (
            (recon_centroid - target_centroid) * voiced_weight
        ).sum() / voiced_count

        slope_frequency_khz = frequencies[slope_band] / 1000.
        slope_frequency_khz = slope_frequency_khz - slope_frequency_khz.mean()
        slope_denominator = slope_frequency_khz.square().sum().clamp_min(1e-8)
        db_scale = 20. / math.log(10.)
        target_slope = (
            target_log_mag[:, slope_band] *
            slope_frequency_khz[None, :, None]
        ).sum(dim = 1) * db_scale / slope_denominator
        recon_slope = (
            recon_log_mag[:, slope_band] *
            slope_frequency_khz[None, :, None]
        ).sum(dim = 1) * db_scale / slope_denominator
        spectral_slope_delta = (
            (recon_slope - target_slope) * voiced_weight
        ).sum() / voiced_count

        return (
            voiced_highband_loss,
            voiced_hf_energy_ratio_db,
            voiced_highband_logmag_error,
            voiced_hf_energy_deficit,
            spectral_centroid_delta_hz,
            spectral_slope_delta,
            voiced_hf_retention_loss,
            upper_highband_loss,
            voiced_upper_logmag_error,
            voiced_upper_energy_deficit,
            voiced_upper_energy_ratio_db,
            quiet_upper_excess_db
        )

    def active_spectral_detail_metrics(self, target, recon):
        """Compare voiced, target-supported spectral detail across speech bands.

        The training value is normalized first across speech bands and then
        across STFT resolutions. Each band combines target-supported log
        magnitude, its frame-to-frame delta, and spectral convergence with
        weights 0.7 / 0.2 / 0.1. A bin contributes only when its target
        magnitude is within 50 dB of that frame's target peak and the target
        frame is voiced. Diagnostics use the 1024-sample scale and report each
        speech band independently, which makes broad yellow spectrogram
        differences easier to localize without turning every quiet
        time-frequency bin into a loss.
        """
        target_rows = target.float()
        recon_rows = recon.float()
        if target_rows.ndim == 3:
            target_rows = rearrange(target_rows, 'b c n -> (b c) n')
        if recon_rows.ndim == 3:
            recon_rows = rearrange(recon_rows, 'b c n -> (b c) n')

        num_samples = min(target_rows.shape[-1], recon_rows.shape[-1])
        target_rows = target_rows[..., :num_samples]
        recon_rows = recon_rows[..., :num_samples]

        weighted_losses = []
        scale_weights = []
        band_metrics = {}

        for n_fft, win_length, hop_length, alpha in (
            self.active_spectral_detail_settings
        ):
            target_padded = target_rows
            recon_padded = recon_rows
            if num_samples < n_fft:
                padding = n_fft - num_samples
                target_padded = F.pad(target_padded, (0, padding))
                recon_padded = F.pad(recon_padded, (0, padding))

            window = torch.hann_window(
                win_length,
                device = target_rows.device,
                dtype = target_rows.dtype
            )
            stft_kwargs = dict(
                n_fft = n_fft,
                hop_length = hop_length,
                win_length = win_length,
                window = window,
                center = True,
                pad_mode = 'constant',
                return_complex = True
            )
            target_mag = torch.stft(target_padded, **stft_kwargs).abs()
            recon_mag = torch.stft(recon_padded, **stft_kwargs).abs()
            target_log_mag = log(target_mag, eps = 1e-5)
            recon_log_mag = log(recon_mag, eps = 1e-5)

            frequencies = torch.fft.rfftfreq(
                n_fft,
                d = 1. / self.target_sample_hz,
                device = target_rows.device
            )
            detail_band = (
                (frequencies >= self.active_spectral_detail_min_hz) &
                (frequencies <= self.active_spectral_detail_max_hz)
            )
            if not detail_band.any():
                continue

            frame_rms = F.avg_pool1d(
                target_padded.square().unsqueeze(1),
                kernel_size = win_length,
                stride = hop_length,
                padding = win_length // 2,
                count_include_pad = False
            ).squeeze(1).clamp_min(1e-12).sqrt()
            num_frames = min(
                frame_rms.shape[-1],
                target_mag.shape[-1],
                recon_mag.shape[-1]
            )
            frame_rms = frame_rms[..., :num_frames]
            target_mag = target_mag[..., :num_frames]
            recon_mag = recon_mag[..., :num_frames]
            target_log_mag = target_log_mag[..., :num_frames]
            recon_log_mag = recon_log_mag[..., :num_frames]

            relative_rms_floor = (
                frame_rms.amax(dim = -1, keepdim = True) *
                (10. ** (self.spectral_envelope_relative_rms_db / 20.))
            )
            voiced_mask = (
                (frame_rms >= relative_rms_floor) &
                (frame_rms >= self.spectral_envelope_absolute_rms)
            )
            target_detail_log = target_log_mag[:, detail_band]
            frame_peak = target_detail_log.amax(dim = 1, keepdim = True)
            active_mask = (
                target_detail_log >=
                frame_peak + self.active_spectral_detail_relative_db * (
                    math.log(10.) / 20.
                )
            )
            active_mask = active_mask & voiced_mask.unsqueeze(1)
            active_weight = active_mask.to(target_mag.dtype)

            detail_frequencies = frequencies[detail_band]
            target_detail_mag = target_mag[:, detail_band]
            recon_detail_mag = recon_mag[:, detail_band]
            recon_detail_log = recon_log_mag[:, detail_band]
            band_losses = []
            band_weights = []
            for _, min_hz, max_hz, band_alpha in (
                self.active_spectral_detail_bands
            ):
                band = (
                    (detail_frequencies >= min_hz) &
                    (
                        detail_frequencies <= max_hz
                        if max_hz >= self.active_spectral_detail_max_hz
                        else detail_frequencies < max_hz
                    )
                )
                if not band.any():
                    continue

                band_weight = active_weight[:, band]
                band_count = band_weight.sum().clamp_min(1.)
                target_band_log = target_detail_log[:, band]
                recon_band_log = recon_detail_log[:, band]
                target_band_mag = target_detail_mag[:, band]
                recon_band_mag = recon_detail_mag[:, band]

                logmag_loss = (
                    F.smooth_l1_loss(
                        recon_band_log,
                        target_band_log,
                        reduction = 'none',
                        beta = 0.10
                    ) *
                    band_weight
                ).sum() / band_count

                pair_weight = (
                    band_weight[..., 1:] *
                    band_weight[..., :-1]
                )
                pair_count = pair_weight.sum().clamp_min(1.)
                target_log_delta = (
                    target_band_log[..., 1:] -
                    target_band_log[..., :-1]
                )
                recon_log_delta = (
                    recon_band_log[..., 1:] -
                    recon_band_log[..., :-1]
                )
                temporal_delta_loss = (
                    F.smooth_l1_loss(
                        recon_log_delta,
                        target_log_delta,
                        reduction = 'none',
                        beta = 0.10
                    ) *
                    pair_weight
                ).sum() / pair_count

                spectral_convergence = torch.sqrt(
                    (
                        (recon_band_mag - target_band_mag).square() *
                        band_weight
                    ).sum() /
                    (
                        target_band_mag.square() * band_weight
                    ).sum().clamp_min(1e-10)
                )
                band_loss = (
                    0.70 * logmag_loss +
                    0.20 * temporal_delta_loss +
                    0.10 * spectral_convergence
                )
                band_losses.append(float(band_alpha) * band_loss)
                band_weights.append(float(band_alpha))

            scale_loss = (
                torch.stack(band_losses).sum() /
                max(sum(band_weights), 1e-8)
                if band_losses
                else self.zero.to(target_rows)
            )
            weighted_losses.append(float(alpha) * scale_loss)
            scale_weights.append(float(alpha))

            # A single fixed diagnostic scale keeps validation values directly
            # comparable while the training loss remains multi-resolution.
            if win_length != 1024:
                continue

            for name, min_hz, max_hz, _ in self.active_spectral_detail_bands:
                band = (
                    (detail_frequencies >= min_hz) &
                    (
                        detail_frequencies <= max_hz
                        if max_hz >= self.active_spectral_detail_max_hz
                        else detail_frequencies < max_hz
                    )
                )
                if not band.any():
                    zero = self.zero.to(target_rows)
                    band_metrics[name] = (zero, zero, zero)
                    continue

                band_weight = active_weight[:, band]
                band_count = band_weight.sum().clamp_min(1.)
                target_band_mag = target_detail_mag[:, band]
                recon_band_mag = recon_detail_mag[:, band]
                logmag_error = (
                    (
                        recon_log_mag[:, detail_band][:, band] -
                        target_detail_log[:, band]
                    ).abs() *
                    band_weight
                ).sum() / band_count
                target_power = (
                    target_band_mag.square() * band_weight
                ).sum().clamp_min(1e-10)
                recon_power = (
                    recon_band_mag.square() * band_weight
                ).sum().clamp_min(1e-10)
                energy_ratio_db = 10. * torch.log10(
                    recon_power / target_power
                )
                spectral_convergence = torch.sqrt(
                    (
                        (recon_band_mag - target_band_mag).square() *
                        band_weight
                    ).sum() /
                    (
                        target_band_mag.square() * band_weight
                    ).sum().clamp_min(1e-10)
                )
                band_metrics[name] = (
                    logmag_error,
                    energy_ratio_db,
                    spectral_convergence
                )

        total_loss = (
            torch.stack(weighted_losses).sum() /
            max(sum(scale_weights), 1e-8)
            if weighted_losses
            else self.zero.to(target_rows)
        )
        zero = self.zero.to(target_rows)
        for name, _, _, _ in self.active_spectral_detail_bands:
            band_metrics.setdefault(name, (zero, zero, zero))
        return total_loss, band_metrics

    def transient_noise_losses(self, target, recon):
        if (
            self.click_loss_weight <= 0 and
            self.jump_loss_weight <= 0
        ):
            return self.zero, self.zero

        if target.shape[-1] < 2 or recon.shape[-1] < 2:
            return self.zero, self.zero

        target_delta = target.diff(dim = -1)
        recon_delta = recon.diff(dim = -1)

        # Smooth, local click proxy. This keeps the reconstructed local slope
        # close to the source slope without over-penalizing natural speech
        # transients.
        diff_l1 = F.l1_loss(recon_delta, target_delta)

        # Clean-gate-aligned click proxy. The validation gate is driven by a
        # normalized reconstructed jump score (recon jump / recon RMS), so this
        # adds a differentiable top-k excess penalty against the same failure
        # mode instead of only optimizing the average first-difference error.
        recon_abs_delta = recon_delta.abs()
        recon_rms = recon.square().mean(dim = -1, keepdim = True).clamp_min(1e-8).sqrt()
        norm_recon_delta = recon_abs_delta / recon_rms.clamp_min(1e-8)
        flat_norm_delta = rearrange(
            norm_recon_delta.float(),
            'b c n -> b (c n)'
        )
        topk_count = max(1, int(flat_norm_delta.shape[-1] * 0.001))
        topk_click = flat_norm_delta.topk(topk_count, dim = -1).values.mean(dim = -1)
        gate_click_loss = F.relu(topk_click - 5.5).square().mean()

        click_loss = diff_l1 + 0.5 * gate_click_loss

        # Soft excess-jump proxy. Allow reconstructed slopes to follow real
        # speech transients, but penalize slopes that exceed the local target
        # slope envelope plus a detached robust per-example high-percentile
        # margin. This keeps the penalty small unless the model creates spikes
        # that are not present in the source.
        target_abs_delta = target_delta.detach().abs()
        flat_target_delta = rearrange(
            target_abs_delta.float(),
            'b c n -> b (c n)'
        )
        target_p999 = torch.quantile(
            flat_target_delta,
            0.999,
            dim = -1,
            keepdim = True
        ).to(dtype = recon_abs_delta.dtype)
        while target_p999.ndim < recon_abs_delta.ndim:
            target_p999 = target_p999.unsqueeze(-1)

        allowed_delta = (
            1.5 * target_abs_delta +
            0.5 * target_p999 +
            1e-4
        )
        jump_excess = F.relu(recon_abs_delta - allowed_delta)
        jump_loss = jump_excess.square().mean().clamp_min(1e-12).sqrt()

        return click_loss, jump_loss

    def quiet_multiband_noise_metrics(self, target, recon):
        """One-sided quiet-frame band loss plus a validation-only 2-8 kHz metric."""
        target_rows, recon_rows = target.float(), recon.float()
        if target_rows.ndim == 3:
            target_rows = rearrange(target_rows, 'b c n -> (b c) n')
        if recon_rows.ndim == 3:
            recon_rows = rearrange(recon_rows, 'b c n -> (b c) n')

        num_samples = min(target_rows.shape[-1], recon_rows.shape[-1])
        target_rows = target_rows[..., :num_samples]
        recon_rows = recon_rows[..., :num_samples]
        if num_samples == 0:
            return self.zero, self.zero

        frame_length = max(1, int(round(0.040 * self.target_sample_hz)))
        hop_length = max(1, int(round(0.020 * self.target_sample_hz)))
        if num_samples < frame_length:
            padding = frame_length - num_samples
            target_rows = F.pad(target_rows, (0, padding))
            recon_rows = F.pad(recon_rows, (0, padding))

        target_frames = target_rows.unfold(-1, frame_length, hop_length)
        recon_frames = recon_rows.unfold(-1, frame_length, hop_length)
        frame_rms = target_frames.square().mean(dim = -1).clamp_min(1e-10).sqrt()
        quiet_threshold = torch.quantile(
            frame_rms.detach(), 0.30, dim = -1, keepdim = True
        ).clamp_max(0.03)
        quiet_mask = (frame_rms <= quiet_threshold).to(target_frames.dtype)

        window = torch.hann_window(
            frame_length, device = target_frames.device, dtype = target_frames.dtype
        )
        n_fft = 1 << (frame_length - 1).bit_length()
        normalization = window.square().sum().clamp_min(1e-8)
        target_power = torch.fft.rfft(
            target_frames * window, n = n_fft, dim = -1
        ).abs().square() / normalization
        recon_power = torch.fft.rfft(
            recon_frames * window, n = n_fft, dim = -1
        ).abs().square() / normalization
        frequencies = torch.fft.rfftfreq(
            n_fft, d = 1. / self.target_sample_hz, device = target_frames.device
        )

        bands = (
            (0., 1_000., 0.25), (1_000., 2_000., 0.50),
            (2_000., 4_000., 1.00),
            (4_000., min(8_000., self.target_sample_hz / 2), 1.25),
        )
        losses, loss_weights, hf_values, hf_weights = [], [], [], []
        quiet_count = quiet_mask.sum().clamp_min(1.)
        margin = math.log(10.) / 10.
        db_scale = 10. / math.log(10.)
        for low_hz, high_hz, alpha in bands:
            band_mask = (frequencies >= low_hz) & (frequencies < high_hz)
            if high_hz <= low_hz or not band_mask.any():
                continue
            target_band = target_power[..., band_mask].mean(dim = -1)
            recon_band = recon_power[..., band_mask].mean(dim = -1)
            log_ratio = torch.log(recon_band + 1e-10) - torch.log(target_band + 1e-10)
            excess = (F.relu(log_ratio - margin) * quiet_mask).sum() / quiet_count
            losses.append(alpha * excess)
            loss_weights.append(alpha)
            if low_hz >= 2_000.:
                excess_db = (F.relu(log_ratio * db_scale) * quiet_mask).sum() / quiet_count
                hf_values.append(alpha * excess_db)
                hf_weights.append(alpha)

        loss = torch.stack(losses).sum() / max(sum(loss_weights), 1e-8) if losses else self.zero
        quiet_hf_excess_db = (
            torch.stack(hf_values).sum() / max(sum(hf_weights), 1e-8)
            if hf_values else self.zero
        )
        return loss, quiet_hf_excess_db

    def background_noise_losses(self, target, recon):
        """Stage-specific pre-emphasis and quiet-frame multiband constraints."""
        if (
            self.preemph_loss_weight <= 0 and
            self.noise_floor_loss_weight <= 0
        ):
            return self.zero, self.zero

        if target.shape[-1] < 2 or recon.shape[-1] < 2:
            return self.zero, self.zero

        target_float = target.float()
        recon_float = recon.float()
        preemph_loss = self.zero
        if self.preemph_loss_weight > 0:
            preemphasis = 0.97
            target_highpass = (
                target_float[..., 1:] -
                preemphasis * target_float[..., :-1]
            )
            recon_highpass = (
                recon_float[..., 1:] -
                preemphasis * recon_float[..., :-1]
            )
            preemph_loss = F.l1_loss(recon_highpass, target_highpass)

        noise_floor_loss = (
            self.quiet_multiband_noise_metrics(target_float, recon_float)[0]
            if self.noise_floor_loss_weight > 0
            else self.zero
        )
        return preemph_loss, noise_floor_loss

    def generator_perceptual_losses(self, real, fake):
        if self.adversarial_loss_weight <= 0 and self.feature_loss_weight <= 0:
            return self.zero, self.zero

        adversarial_losses = []
        discriminator_intermediates = []

        (stft_real_logits, stft_real_intermediates), (
            stft_fake_logits,
            stft_fake_intermediates
        ) = map(
            partial(self.stft_discriminator, return_intermediates = True),
            (real, fake)
        )
        discriminator_intermediates.append(
            (stft_real_intermediates, stft_fake_intermediates)
        )

        scaled_real, scaled_fake = real, fake
        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))
            (real_logits, real_intermediates), (
                fake_logits,
                fake_intermediates
            ) = map(
                partial(discr, return_intermediates = True),
                (scaled_real, scaled_fake)
            )
            discriminator_intermediates.append((real_intermediates, fake_intermediates))
            adversarial_losses.append(hinge_gen_loss(fake_logits))

        per_discriminator_feature_losses = []
        for real_features, fake_features in discriminator_intermediates:
            if len(real_features) != len(fake_features) or len(real_features) == 0:
                raise RuntimeError(
                    "feature-matching intermediates must be non-empty and aligned: "
                    f"real={len(real_features)}, fake={len(fake_features)}"
                )
            layer_losses = [
                F.l1_loss(real_feature, fake_feature)
                for real_feature, fake_feature in zip(real_features, fake_features)
            ]
            per_discriminator_feature_losses.append(
                torch.stack(layer_losses).mean()
            )

        feature_loss = torch.stack(per_discriminator_feature_losses).mean()
        adversarial_losses.append(hinge_gen_loss(stft_fake_logits))
        adversarial_loss = torch.stack(adversarial_losses).mean()
        return adversarial_loss, feature_loss

    def decode_from_codebook_indices(self, quantized_indices):
        assert quantized_indices.dtype in (torch.long, torch.int32)

        if quantized_indices.ndim == 3:
            quantized_indices = rearrange(quantized_indices, 'b n (g q) -> g b n q', g = self.rq_groups)

        x = self.rq.get_output_from_indices(quantized_indices)

        return self.decode(x)

    def decode(self, x, quantize = False):
        if quantize:
            x, *_ = self.rq(x)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')
        return self.decoder(x)

    def save(self, path):
        path = Path(path)
        pkg = dict(
            model = self.state_dict(),
            config = self._configs,
            version = __version__
        )

        torch.save(pkg, str(path))

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        soundstream = cls(**config)
        soundstream.load(path, strict = strict)
        soundstream.eval()
        return soundstream

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        # check version

        if 'version' in pkg and version.parse(pkg['version']) < parsed_version:
            print(f'soundstream model being loaded was trained on an older version of audiolm-pytorch ({pkg["version"]})')

        has_ema = 'ema_model' in pkg
        model_pkg = pkg['ema_model'] if has_ema else pkg['model']

        if has_ema:
            model_pkg = filter_by_keys(lambda k: k.startswith('ema_model.'), model_pkg)
            model_pkg = map_keys(lambda k: k[len('ema_model.'):], model_pkg)

        self.load_state_dict(model_pkg, strict = strict)

    def load_from_trainer_saved_obj(self, path):
        path = Path(path)
        assert path.exists()
        obj = torch.load(str(path))
        self.load_state_dict(obj['model'])

    @staticmethod
    def is_discriminator_state_key(key):
        return key.startswith(('stft_discriminator.', 'discriminators.'))

    def load_generator_state_dict(self, state_dict):
        """Strictly load generator/RVQ state while reinitializing discriminators."""
        generator_state = {
            key: value
            for key, value in state_dict.items()
            if not self.is_discriminator_state_key(key)
        }
        incompatible = self.load_state_dict(generator_state, strict = False)
        unexpected = list(incompatible.unexpected_keys)
        missing_generator = [
            key for key in incompatible.missing_keys
            if not self.is_discriminator_state_key(key)
        ]
        if unexpected or missing_generator:
            raise RuntimeError(
                "generator-only checkpoint load was not strict for generator "
                f"state: missing={missing_generator}, unexpected={unexpected}"
            )
        return tuple(incompatible.missing_keys)

    def non_discr_parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *(self.encoder_attn.parameters() if exists(self.encoder_attn) else []),
            *(self.decoder_attn.parameters() if exists(self.decoder_attn) else []),
            *self.encoder_film.parameters(),
            *self.decoder_film.parameters(),
            *self.rq.parameters()
        ]

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    @property
    def downsample_factor(self):
        return self.seq_len_multiple_of

    def process_input(
        self,
        x,
        input_sample_hz = None,
        curtail_from_left = False
    ):
        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        x = curtail_to_multiple(x, self.seq_len_multiple_of, from_left = curtail_from_left)

        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')

        return x, ps

    @torch.no_grad()
    def tokenize(self, audio):
        self.eval()
        return self.forward(audio, return_codes_only = True)

    def forward_bypass_rvq(
        self,
        x,
        is_denoising = None,
        return_recons_only = True,
        input_sample_hz = None,
        curtail_from_left = False
    ):
        """Diagnostic path: Encoder -> Decoder, skipping RVQ quantization.

        This is intended for reconstruction debugging only.  It keeps the same
        preprocessing, encoder attention, optional FiLM conditioning, decoder
        attention, and output unpacking as the normal forward path, but does not
        call `self.rq`.
        """
        process_input = partial(self.process_input, input_sample_hz = input_sample_hz, curtail_from_left = curtail_from_left)

        x, ps = process_input(x)

        x = self.encoder(x)
        x = rearrange(x, 'b c n -> b n c')

        denoise_input = None
        if exists(self.encoder_attn):
            x = self.encoder_attn(x)

        if exists(is_denoising):
            denoise_input = torch.tensor([is_denoising, not is_denoising], dtype = x.dtype, device = self.device)
            x = self.encoder_film(x, denoise_input)

        if exists(is_denoising):
            x = self.decoder_film(x, denoise_input)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')
        recon_x = self.decoder(x)

        if return_recons_only:
            recon_x, = unpack(recon_x, ps, '* c n')

        return recon_x

    def forward(
        self,
        x,
        target = None,
        is_denoising = None, # if you want to learn film conditioners that teach the soundstream to denoise - target would need to be passed in above
        return_encoded = False,
        return_codes_only = False,
        return_discr_loss = False,
        return_discr_losses_separately = False,
        return_loss_breakdown = False,
        return_recons_only = False,
        input_sample_hz = None,
        apply_grad_penalty = False,
        apply_stft_grad_penalty = False,
        waveform_grad_penalty_gamma = 0.,
        stft_grad_penalty_gamma = 5e-3,
        curtail_from_left = False,
        num_quantizers = None,
        freeze_codebook = False
    ):
        assert not (exists(is_denoising) and not exists(target))
        assert not (exists(num_quantizers) and self.training), 'num_quantizers is an inference-only option'

        process_input = partial(self.process_input, input_sample_hz = input_sample_hz, curtail_from_left = curtail_from_left)

        x, ps = process_input(x)

        if exists(target):
            target, _ = process_input(target)

        orig_x = x.clone()

        x = self.encoder(x)

        x = rearrange(x, 'b c n -> b n c')

        if exists(self.encoder_attn):
            x = self.encoder_attn(x)

        if exists(is_denoising):
            denoise_input = torch.tensor([is_denoising, not is_denoising], dtype = x.dtype, device = self.device) # [1, 0] for denoise, [0, 1] for not denoising
            x = self.encoder_film(x, denoise_input)

        if self.bypass_rvq:
            indices = torch.full(
                (self.rq_groups, x.shape[0], x.shape[1], self.num_quantizers),
                -1,
                dtype = torch.long,
                device = x.device,
            )
            commit_loss = self.zero
        elif not self.use_finite_scalar_quantizer:
            rq_kwargs = (
                dict(freeze_codebook = freeze_codebook)
                if not self.use_lookup_free_quantizer
                else {}
            )
            x, indices, commit_loss = self.rq(x, **rq_kwargs)
        else:
            # finite scalar quantizer does not have any aux loss

            x, indices = self.rq(x)
            commit_loss = self.zero

        if exists(num_quantizers) and not self.bypass_rvq:
            assert 0 < num_quantizers <= self.num_quantizers
            indices = indices[..., :num_quantizers]
            x = self.rq.get_output_from_indices(indices)

        if return_codes_only:
            return indices

        if return_encoded:
            indices = rearrange(indices, 'g b n q -> b n (g q)')
            return x, indices, commit_loss

        if exists(is_denoising):
            x = self.decoder_film(x, denoise_input)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')

        recon_x = self.decoder(x)

        if return_recons_only:
            recon_x, = unpack(recon_x, ps, '* c n')
            return recon_x

        # multi-scale discriminator loss

        if return_discr_loss:
            real, fake = orig_x.detach(), recon_x.detach()

            stft_discr_loss = None
            stft_grad_penalty = None
            discr_losses = []
            discr_grad_penalties = []
            discr_real_active_fractions = []
            discr_fake_active_fractions = []

            if self.single_channel:
                real = orig_x.detach().requires_grad_(apply_stft_grad_penalty)
                fake = recon_x.detach()
                stft_real_logits = self.stft_discriminator(real)
                stft_fake_logits = self.stft_discriminator(fake)
                stft_discr_loss = hinge_discr_loss(stft_fake_logits, stft_real_logits)

                if apply_stft_grad_penalty:
                    (
                        stft_grad_penalty,
                        stft_grad_penalty_raw_mean,
                        stft_grad_penalty_raw_sum
                    ) = r1_gradient_penalty(
                        real,
                        stft_real_logits,
                        stft_grad_penalty_gamma
                    )

            scaled_real, scaled_fake = real, fake
            for discr, downsample in zip(self.discriminators, self.downsamples):
                scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

                if apply_grad_penalty:
                    scaled_real.requires_grad_(True)
                real_logits = discr(scaled_real)
                fake_logits = discr(scaled_fake)
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                discr_real_active_fractions.append((real_logits < 1.).float().mean().detach())
                discr_fake_active_fractions.append((fake_logits > -1.).float().mean().detach())
                if apply_grad_penalty:
                    one_grad_penalty, _, _ = r1_gradient_penalty(
                        scaled_real,
                        real_logits,
                        waveform_grad_penalty_gamma
                    )
                    discr_grad_penalties.append(one_grad_penalty)

            if not return_discr_losses_separately:
                return aggregate_discriminator_losses(
                    discr_losses,
                    stft_discr_loss,
                    discr_grad_penalties,
                    stft_grad_penalty
                )

            # return a list of discriminator losses with List[Tuple[str, Tensor]]

            discr_losses_pkg = []

            discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in zip(self.discr_multi_scales, discr_losses)])

            discr_losses_pkg.extend([
                (f'scale_real_hinge_active:{scale}', active_fraction)
                for scale, active_fraction in zip(
                    self.discr_multi_scales,
                    discr_real_active_fractions,
                )
            ])
            discr_losses_pkg.extend([
                (f'scale_fake_hinge_active:{scale}', active_fraction)
                for scale, active_fraction in zip(
                    self.discr_multi_scales,
                    discr_fake_active_fractions,
                )
            ])

            discr_losses_pkg.extend([(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in zip(self.discr_multi_scales, discr_grad_penalties)])

            if exists(stft_discr_loss):
                stft_total_loss = stft_discr_loss
                if exists(stft_grad_penalty):
                    stft_total_loss = stft_total_loss + stft_grad_penalty
                discr_losses_pkg.extend((
                    ('stft_total', stft_total_loss),
                    ('stft', stft_discr_loss.detach()),
                    ('stft_real_logits_mean', stft_real_logits.detach().mean()),
                    ('stft_fake_logits_mean', stft_fake_logits.detach().mean()),
                    ('stft_saturated', (stft_discr_loss.detach() < 1e-4).float()),
                ))

            if exists(stft_grad_penalty):
                discr_losses_pkg.extend((
                    ('stft_r1_raw_mean', stft_grad_penalty_raw_mean),
                    ('stft_r1_raw_sum', stft_grad_penalty_raw_sum),
                    ('stft_r1_weighted', stft_grad_penalty.detach()),
                ))

            return discr_losses_pkg

        # recon loss

        target = default(target, orig_x)  # target can also be passed in, in the case of denoising

        (
            recon_loss,
            multi_spectral_recon_loss,
            stft_recon_loss,
            correlation_loss
        ) = self.reconstruction_losses(
            target,
            recon_x
        )
        si_sdr_loss = (
            self.si_sdr_loss(target, recon_x)
            if self.si_sdr_loss_weight > 0
            else self.zero
        )
        click_loss, jump_loss = self.transient_noise_losses(target, recon_x)
        spectral_envelope_loss = (
            self.spectral_envelope_metrics(target, recon_x)[0]
            if (
                self.spectral_envelope_loss_weight > 0 or
                self.formant_peak_loss_weight > 0
            )
            else self.zero
        )
        formant_peak_loss = (
            self._last_formant_metrics['peak_loss']
            if hasattr(self, '_last_formant_metrics')
            else self.zero
        )
        voiced_highband_metrics = (
            self.voiced_highband_metrics(target, recon_x)
            if (
                self.voiced_highband_loss_weight > 0 or
                self.voiced_hf_retention_loss_weight > 0 or
                self.upper_highband_loss_weight > 0
            )
            else (self.zero,) * 12
        )
        voiced_highband_loss = voiced_highband_metrics[0]
        voiced_hf_retention_loss = voiced_highband_metrics[6]
        upper_highband_loss = voiced_highband_metrics[7]
        active_spectral_detail_loss = (
            self.active_spectral_detail_metrics(target, recon_x)[0]
            if self.active_spectral_detail_loss_weight > 0
            else self.zero
        )
        preemph_loss, noise_floor_loss = self.background_noise_losses(target, recon_x)
        frame_phase_loss = (
            self.frame_phase_residual_loss(target, recon_x)
            if self.frame_phase_loss_weight > 0
            else self.zero
        )
        adversarial_loss, feature_loss = self.generator_perceptual_losses(
            orig_x,
            recon_x
        )

        # sum commitment loss

        all_commitment_loss = commit_loss.sum()

        total_loss = (
            recon_loss * self.recon_loss_weight +
            multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight +
            stft_recon_loss * self.stft_recon_loss_weight +
            si_sdr_loss * self.si_sdr_loss_weight +
            spectral_envelope_loss * self.spectral_envelope_loss_weight +
            formant_peak_loss * self.formant_peak_loss_weight +
            voiced_highband_loss * self.voiced_highband_loss_weight +
            voiced_hf_retention_loss * self.voiced_hf_retention_loss_weight +
            upper_highband_loss * self.upper_highband_loss_weight +
            active_spectral_detail_loss *
            self.active_spectral_detail_loss_weight +
            correlation_loss * self.correlation_loss_weight +
            click_loss * self.click_loss_weight +
            jump_loss * self.jump_loss_weight +
            preemph_loss * self.preemph_loss_weight +
            noise_floor_loss * self.noise_floor_loss_weight +
            frame_phase_loss * self.frame_phase_loss_weight +
            adversarial_loss * self.adversarial_loss_weight +
            feature_loss * self.feature_loss_weight +
            all_commitment_loss * self.commitment_loss_weight
        )

        if return_loss_breakdown:
            return total_loss, (
                recon_loss,
                multi_spectral_recon_loss,
                stft_recon_loss,
                spectral_envelope_loss,
                voiced_highband_loss,
                voiced_hf_retention_loss,
                adversarial_loss,
                feature_loss,
                all_commitment_loss,
                si_sdr_loss,
                correlation_loss,
                click_loss,
                jump_loss,
                preemph_loss,
                noise_floor_loss,
                frame_phase_loss,
                upper_highband_loss,
                active_spectral_detail_loss,
                formant_peak_loss
            )

        return total_loss

class FrameStreamingSoundStream(SoundStream):
    def __init__(
        self,
        *,
        stream_frame_size = 320,
        stream_context_frames = 0,
        boundary_loss_weight = 0.1,
        boundary_loss_radius = 8,
        boundary_loss_start_steps = 0,
        boundary_loss_warmup_steps = 0,
        stream_consistency_loss_weight = 0.,
        stream_consistency_loss_start_steps = 0,
        stream_consistency_loss_warmup_steps = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stream_frame_size = stream_frame_size
        self.stream_context_frames = stream_context_frames
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_loss_radius = boundary_loss_radius
        self.boundary_loss_start_steps = int(boundary_loss_start_steps)
        self.boundary_loss_warmup_steps = int(boundary_loss_warmup_steps)
        self.stream_consistency_loss_weight = float(stream_consistency_loss_weight)
        self.stream_consistency_loss_start_steps = int(stream_consistency_loss_start_steps)
        self.stream_consistency_loss_warmup_steps = int(stream_consistency_loss_warmup_steps)
        self.runtime_boundary_loss_weight = (
            0.
            if self.boundary_loss_start_steps > 0 or self.boundary_loss_warmup_steps > 0
            else float(boundary_loss_weight)
        )
        self.runtime_stream_consistency_loss_weight = (
            0.
            if (
                self.stream_consistency_loss_start_steps > 0 or
                self.stream_consistency_loss_warmup_steps > 0
            )
            else float(stream_consistency_loss_weight)
        )

        config = pickle.loads(self._configs)
        config['stream_frame_size'] = stream_frame_size
        config['stream_context_frames'] = stream_context_frames
        config['boundary_loss_weight'] = boundary_loss_weight
        config['boundary_loss_radius'] = boundary_loss_radius
        config['boundary_loss_start_steps'] = boundary_loss_start_steps
        config['boundary_loss_warmup_steps'] = boundary_loss_warmup_steps
        config['stream_consistency_loss_weight'] = stream_consistency_loss_weight
        config['stream_consistency_loss_start_steps'] = stream_consistency_loss_start_steps
        config['stream_consistency_loss_warmup_steps'] = stream_consistency_loss_warmup_steps
        self._configs = pickle.dumps(config)

    @staticmethod
    def _scheduled_stream_weight(step, start_steps, warmup_steps, target_weight):
        if step < start_steps:
            return 0.
        if warmup_steps <= 0:
            return float(target_weight)
        progress = min(max((step - start_steps) / warmup_steps, 0.), 1.)
        return float(target_weight) * progress

    def update_stream_loss_weights(self, step):
        """Update only runtime weights; checkpoint configuration keeps targets."""
        self.runtime_boundary_loss_weight = self._scheduled_stream_weight(
            step,
            self.boundary_loss_start_steps,
            self.boundary_loss_warmup_steps,
            self.boundary_loss_weight,
        )
        self.runtime_stream_consistency_loss_weight = self._scheduled_stream_weight(
            step,
            self.stream_consistency_loss_start_steps,
            self.stream_consistency_loss_warmup_steps,
            self.stream_consistency_loss_weight,
        )
        return (
            self.runtime_boundary_loss_weight,
            self.runtime_stream_consistency_loss_weight,
        )

    def offline_reconstruction(self, x):
        """Run the same weights through the full-sequence causal path."""
        return SoundStream.forward(
            self,
            x,
            return_recons_only = True,
            freeze_codebook = True,
        )

    def offline_decode_quantized(self, quantized, is_denoising = None):
        """Decode an existing latent sequence once through the offline path.

        Reusing the exact latent sequence produced by ``stream_codec`` avoids
        a second encoder / RVQ call in training.  That both removes needless
        work and prevents train-mode quantizer behaviour from contaminating a
        loss whose purpose is to compare the two decoder execution paths.
        """
        x = quantized
        if exists(is_denoising):
            denoise_input = torch.tensor(
                [is_denoising, not is_denoising],
                dtype = x.dtype,
                device = x.device,
            )
            x = self.decoder_film(x, denoise_input)
        x = rearrange(x, 'b n c -> b c n')
        return self.decoder(x)

    def stream_consistency_loss(
        self,
        x,
        stream_recon,
        quantized = None,
        is_denoising = None,
    ):
        if self.stream_consistency_loss_weight <= 0.:
            return self.zero

        with torch.no_grad():
            offline_recon = (
                self.offline_decode_quantized(
                    quantized,
                    is_denoising = is_denoising,
                )
                if exists(quantized)
                else self.offline_reconstruction(x)
            )

        offline_recon = offline_recon[..., :stream_recon.shape[-1]]
        stream_recon = stream_recon[..., :offline_recon.shape[-1]]
        # This objective measures implementation consistency, not perceptual
        # reconstruction quality.  The previous reuse of reconstruction_losses
        # included a multi-scale log-Mel L2 term.  Tiny near-silence differences
        # between the offline and stateful paths could therefore produce raw
        # training losses around 5-15 while validation stayed near 0.04.  A
        # direct waveform Smooth-L1 is deterministic, robust to isolated
        # outliers, and keeps train / validation values in the same units.
        return F.smooth_l1_loss(
            stream_recon,
            offline_recon,
            beta = 0.01,
        )

    def frame_boundary_loss(self, recon_x):
        weight = self.boundary_loss_weight
        radius = self.boundary_loss_radius

        if weight <= 0 or radius <= 0:
            return self.zero

        frame_size = self.stream_frame_size
        seq_len = recon_x.shape[-1]

        if seq_len <= frame_size:
            return self.zero

        losses = []

        for boundary in range(frame_size, seq_len, frame_size):
            if boundary < 2 or boundary + 1 >= seq_len:
                continue

            left_slope = (
                recon_x[..., boundary - 1] -
                recon_x[..., boundary - 2]
            )
            right_slope = (
                recon_x[..., boundary + 1] -
                recon_x[..., boundary]
            )
            seam_slope = (
                recon_x[..., boundary] -
                recon_x[..., boundary - 1]
            )
            expected_slope = 0.5 * (left_slope + right_slope)
            losses.append(F.l1_loss(seam_slope, expected_slope))
            losses.append(F.l1_loss(right_slope, left_slope))

            left = recon_x[..., max(0, boundary - radius):boundary]
            right = recon_x[..., boundary:min(seq_len, boundary + radius)]
            if left.shape[-1] > 1 and right.shape[-1] > 1:
                left_delta_rms = (
                    left.diff(dim=-1).square().mean(dim=-1).clamp_min(1e-8).sqrt()
                )
                right_delta_rms = (
                    right.diff(dim=-1).square().mean(dim=-1).clamp_min(1e-8).sqrt()
                )
                losses.append(F.l1_loss(right_delta_rms, left_delta_rms))

        if not losses:
            return self.zero

        return torch.stack(losses).mean()

    def encode_frame(
        self,
        frame,
        state = None,
        is_denoising = None,
        num_quantizers = None,
        freeze_codebook = False
    ):
        x, next_state = self.encode_frame_latent(
            frame,
            state = state,
            is_denoising = is_denoising
        )

        if self.bypass_rvq:
            indices = torch.full(
                (self.rq_groups, x.shape[0], x.shape[1], self.num_quantizers),
                -1,
                dtype = torch.long,
                device = x.device,
            )
            commit_loss = self.zero
        elif not self.use_finite_scalar_quantizer:
            rq_kwargs = (
                dict(freeze_codebook = freeze_codebook)
                if not self.use_lookup_free_quantizer
                else {}
            )
            x, indices, commit_loss = self.rq(x, **rq_kwargs)
        else:
            x, indices = self.rq(x)
            commit_loss = self.zero

        if exists(num_quantizers) and not self.bypass_rvq:
            assert 0 < num_quantizers <= self.num_quantizers
            indices = indices[..., :num_quantizers]
            x = self.rq.get_output_from_indices(indices)

        return x, indices, commit_loss, next_state

    def encode_frame_latent(
        self,
        frame,
        state = None,
        is_denoising = None
    ):
        assert frame.shape[-1] == self.stream_frame_size
        assert not exists(self.encoder_attn), 'stateful frame encoding does not support encoder attention'

        x, next_state = stream_module(self.encoder, frame, state)
        x = rearrange(x, 'b c n -> b n c')
        assert x.shape[1] == 1, 'each input frame must produce exactly one latent frame'

        denoise_input = None
        if exists(is_denoising):
            denoise_input = torch.tensor([is_denoising, not is_denoising], dtype = x.dtype, device = self.device)
            x = self.encoder_film(x, denoise_input)

        return x, next_state

    def decode_frame(self, quantized, state = None, is_denoising = None):
        assert quantized.shape[1] == 1, 'decoder expects exactly one latent frame'
        assert not exists(self.decoder_attn), 'stateful frame decoding does not support decoder attention'

        x = quantized
        if exists(is_denoising):
            denoise_input = torch.tensor([is_denoising, not is_denoising], dtype = x.dtype, device = self.device)
            x = self.decoder_film(x, denoise_input)

        x = rearrange(x, 'b n c -> b c n')
        recon, next_state = stream_module(self.decoder, x, state)
        assert recon.shape[-1] == self.stream_frame_size
        return recon, next_state

    def decode_codes_frame(self, indices, state = None, is_denoising = None):
        if indices.ndim == 3:
            indices = rearrange(indices, 'b n (g q) -> g b n q', g = self.rq_groups)

        quantized = self.rq.get_output_from_indices(indices)
        return self.decode_frame(quantized, state = state, is_denoising = is_denoising)

    def _codec_frame(
        self,
        frame,
        state = None,
        is_denoising = None,
        num_quantizers = None,
        freeze_codebook = False
    ):
        state = default(state, {})
        quantized, indices, commit_loss, encoder_state = self.encode_frame(
            frame,
            state = state.get('encoder'),
            is_denoising = is_denoising,
            num_quantizers = num_quantizers,
            freeze_codebook = freeze_codebook
        )
        recon, decoder_state = self.decode_frame(
            quantized,
            state = state.get('decoder'),
            is_denoising = is_denoising
        )
        next_state = dict(encoder = encoder_state, decoder = decoder_state)
        return recon, quantized, indices, commit_loss, next_state

    def stream_codec(
        self,
        x,
        is_denoising = None,
        num_quantizers = None,
        freeze_codebook = False
    ):
        frame_size = self.stream_frame_size
        assert x.shape[-1] % frame_size == 0, f'input length must be a multiple of {frame_size}'

        latent_frames = []
        encoder_state = None

        for start in range(0, x.shape[-1], frame_size):
            frame = x[..., start:(start + frame_size)]
            frame_latent, encoder_state = self.encode_frame_latent(
                frame,
                state = encoder_state,
                is_denoising = is_denoising
            )
            latent_frames.append(frame_latent)

        all_latents = torch.cat(latent_frames, dim = 1)
        if self.bypass_rvq:
            all_quantized = all_latents
            all_indices = torch.full(
                (
                    self.rq_groups,
                    all_latents.shape[0],
                    all_latents.shape[1],
                    self.num_quantizers,
                ),
                -1,
                dtype = torch.long,
                device = all_latents.device,
            )
            all_commit_loss = self.zero
        elif not self.use_finite_scalar_quantizer:
            rq_kwargs = (
                dict(freeze_codebook = freeze_codebook)
                if not self.use_lookup_free_quantizer
                else {}
            )
            all_quantized, all_indices, all_commit_loss = self.rq(
                all_latents,
                **rq_kwargs
            )
            all_commit_loss = all_commit_loss.sum()
        else:
            all_quantized, all_indices = self.rq(all_latents)
            all_commit_loss = self.zero

        if exists(num_quantizers) and not self.bypass_rvq:
            assert 0 < num_quantizers <= self.num_quantizers
            all_indices = all_indices[..., :num_quantizers]
            all_quantized = self.rq.get_output_from_indices(all_indices)

        recons = []
        decoder_state = None
        for frame_quantized in all_quantized.split(1, dim = 1):
            recon, decoder_state = self.decode_frame(
                frame_quantized,
                state = decoder_state,
                is_denoising = is_denoising
            )
            recons.append(recon)

        recon_x = torch.cat(recons, dim = -1)

        return recon_x, all_quantized, all_indices, all_commit_loss

    def forward(
        self,
        x,
        target = None,
        is_denoising = None,
        return_encoded = False,
        return_codes_only = False,
        return_discr_loss = False,
        return_discr_losses_separately = False,
        return_loss_breakdown = False,
        return_recons_only = False,
        input_sample_hz = None,
        apply_grad_penalty = False,
        apply_stft_grad_penalty = False,
        waveform_grad_penalty_gamma = 0.,
        stft_grad_penalty_gamma = 5e-3,
        curtail_from_left = False,
        num_quantizers = None,
        freeze_codebook = False
    ):
        assert not (exists(is_denoising) and not exists(target))
        assert not (exists(num_quantizers) and self.training), 'num_quantizers is an inference-only option'

        process_input = partial(self.process_input, input_sample_hz = input_sample_hz, curtail_from_left = curtail_from_left)

        x, ps = process_input(x)

        if exists(target):
            target, _ = process_input(target)

        orig_x = x.clone()
        recon_x, quantized, indices, all_commitment_loss = self.stream_codec(
            x,
            is_denoising = is_denoising,
            num_quantizers = num_quantizers,
            freeze_codebook = freeze_codebook
        )

        if return_codes_only:
            return indices

        if return_encoded:
            indices = rearrange(indices, 'g b n q -> b n (g q)')
            return quantized, indices, all_commitment_loss

        if return_recons_only:
            recon_x, = unpack(recon_x, ps, '* c n')
            return recon_x

        if return_discr_loss:
            real, fake = orig_x.detach(), recon_x.detach()

            stft_discr_loss = None
            stft_grad_penalty = None
            discr_losses = []
            discr_grad_penalties = []
            discr_real_active_fractions = []
            discr_fake_active_fractions = []

            if self.single_channel:
                real = orig_x.detach().requires_grad_(apply_stft_grad_penalty)
                fake = recon_x.detach()
                stft_real_logits = self.stft_discriminator(real)
                stft_fake_logits = self.stft_discriminator(fake)
                stft_discr_loss = hinge_discr_loss(stft_fake_logits, stft_real_logits)

                if apply_stft_grad_penalty:
                    (
                        stft_grad_penalty,
                        stft_grad_penalty_raw_mean,
                        stft_grad_penalty_raw_sum
                    ) = r1_gradient_penalty(
                        real,
                        stft_real_logits,
                        stft_grad_penalty_gamma
                    )

            scaled_real, scaled_fake = real, fake
            for discr, downsample in zip(self.discriminators, self.downsamples):
                scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

                if apply_grad_penalty:
                    scaled_real.requires_grad_(True)
                real_logits = discr(scaled_real)
                fake_logits = discr(scaled_fake)
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                discr_real_active_fractions.append((real_logits < 1.).float().mean().detach())
                discr_fake_active_fractions.append((fake_logits > -1.).float().mean().detach())
                if apply_grad_penalty:
                    one_grad_penalty, _, _ = r1_gradient_penalty(
                        scaled_real,
                        real_logits,
                        waveform_grad_penalty_gamma
                    )
                    discr_grad_penalties.append(one_grad_penalty)

            if not return_discr_losses_separately:
                return aggregate_discriminator_losses(
                    discr_losses,
                    stft_discr_loss,
                    discr_grad_penalties,
                    stft_grad_penalty
                )

            discr_losses_pkg = []

            discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in zip(self.discr_multi_scales, discr_losses)])

            discr_losses_pkg.extend([
                (f'scale_real_hinge_active:{scale}', active_fraction)
                for scale, active_fraction in zip(
                    self.discr_multi_scales,
                    discr_real_active_fractions,
                )
            ])
            discr_losses_pkg.extend([
                (f'scale_fake_hinge_active:{scale}', active_fraction)
                for scale, active_fraction in zip(
                    self.discr_multi_scales,
                    discr_fake_active_fractions,
                )
            ])

            discr_losses_pkg.extend([(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in zip(self.discr_multi_scales, discr_grad_penalties)])

            if exists(stft_discr_loss):
                stft_total_loss = stft_discr_loss
                if exists(stft_grad_penalty):
                    stft_total_loss = stft_total_loss + stft_grad_penalty
                discr_losses_pkg.extend((
                    ('stft_total', stft_total_loss),
                    ('stft', stft_discr_loss.detach()),
                    ('stft_real_logits_mean', stft_real_logits.detach().mean()),
                    ('stft_fake_logits_mean', stft_fake_logits.detach().mean()),
                    ('stft_saturated', (stft_discr_loss.detach() < 1e-4).float()),
                ))

            if exists(stft_grad_penalty):
                discr_losses_pkg.extend((
                    ('stft_r1_raw_mean', stft_grad_penalty_raw_mean),
                    ('stft_r1_raw_sum', stft_grad_penalty_raw_sum),
                    ('stft_r1_weighted', stft_grad_penalty.detach()),
                ))

            return discr_losses_pkg

        target = default(target, orig_x)

        (
            recon_loss,
            multi_spectral_recon_loss,
            stft_recon_loss,
            correlation_loss
        ) = self.reconstruction_losses(
            target,
            recon_x
        )
        si_sdr_loss = (
            self.si_sdr_loss(target, recon_x)
            if self.si_sdr_loss_weight > 0
            else self.zero
        )
        click_loss, jump_loss = self.transient_noise_losses(target, recon_x)
        preemph_loss, noise_floor_loss = self.background_noise_losses(target, recon_x)
        spectral_envelope_loss = (
            self.spectral_envelope_metrics(target, recon_x)[0]
            if (
                self.spectral_envelope_loss_weight > 0 or
                self.formant_peak_loss_weight > 0
            )
            else self.zero
        )
        formant_peak_loss = (
            self._last_formant_metrics['peak_loss']
            if hasattr(self, '_last_formant_metrics')
            else self.zero
        )
        voiced_highband_metrics = (
            self.voiced_highband_metrics(target, recon_x)
            if (
                self.voiced_highband_loss_weight > 0 or
                self.voiced_hf_retention_loss_weight > 0 or
                self.upper_highband_loss_weight > 0
            )
            else (self.zero,) * 12
        )
        voiced_highband_loss = voiced_highband_metrics[0]
        voiced_hf_retention_loss = voiced_highband_metrics[6]
        upper_highband_loss = voiced_highband_metrics[7]
        active_spectral_detail_loss = (
            self.active_spectral_detail_metrics(target, recon_x)[0]
            if self.active_spectral_detail_loss_weight > 0
            else self.zero
        )
        adversarial_loss, feature_loss = self.generator_perceptual_losses(
            orig_x,
            recon_x
        )

        boundary_loss = self.frame_boundary_loss(recon_x)
        stream_consistency_loss = self.stream_consistency_loss(
            orig_x,
            recon_x,
            quantized = quantized,
            is_denoising = is_denoising,
        )

        total_loss = (
            recon_loss * self.recon_loss_weight +
            multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight +
            stft_recon_loss * self.stft_recon_loss_weight +
            si_sdr_loss * self.si_sdr_loss_weight +
            spectral_envelope_loss * self.spectral_envelope_loss_weight +
            formant_peak_loss * self.formant_peak_loss_weight +
            voiced_highband_loss * self.voiced_highband_loss_weight +
            voiced_hf_retention_loss * self.voiced_hf_retention_loss_weight +
            upper_highband_loss * self.upper_highband_loss_weight +
            active_spectral_detail_loss *
            self.active_spectral_detail_loss_weight +
            correlation_loss * self.correlation_loss_weight +
            click_loss * self.click_loss_weight +
            jump_loss * self.jump_loss_weight +
            preemph_loss * self.preemph_loss_weight +
            noise_floor_loss * self.noise_floor_loss_weight +
            adversarial_loss * self.adversarial_loss_weight +
            feature_loss * self.feature_loss_weight +
            all_commitment_loss * self.commitment_loss_weight +
            boundary_loss * self.runtime_boundary_loss_weight +
            stream_consistency_loss * self.runtime_stream_consistency_loss_weight
        )

        if return_loss_breakdown:
            return total_loss, (
                recon_loss,
                multi_spectral_recon_loss,
                stft_recon_loss,
                spectral_envelope_loss,
                voiced_highband_loss,
                voiced_hf_retention_loss,
                adversarial_loss,
                feature_loss,
                all_commitment_loss,
                si_sdr_loss,
                correlation_loss,
                click_loss,
                jump_loss,
                preemph_loss,
                noise_floor_loss,
                boundary_loss,
                stream_consistency_loss,
                upper_highband_loss,
                active_spectral_detail_loss,
                formant_peak_loss
            )

        return total_loss

# some default soundstreams

def AudioLMSoundStream(
    strides = (2, 4, 5, 8),
    target_sample_hz = 16000,
    rq_num_quantizers = 12,
    **kwargs
):
    return SoundStream(
        strides = strides,
        target_sample_hz = target_sample_hz,
        rq_num_quantizers = rq_num_quantizers,
        **kwargs
    )

def MusicLMSoundStream(
    strides = (3, 4, 5, 8),
    target_sample_hz = 24000,
    rq_num_quantizers = 12,
    **kwargs
):
    return SoundStream(
        strides = strides,
        target_sample_hz = target_sample_hz,
        rq_num_quantizers = rq_num_quantizers,
        **kwargs
    )
