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
from torch.linalg import vector_norm

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

def gradient_penalty(wave, output, weight = 10, center = 0.):
    batch_size, device = wave.shape[0], wave.device

    gradients = torch_grad(
        outputs = output,
        inputs = wave,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim = 1) - center) ** 2).mean()

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

# complex stft discriminator

class ModReLU(Module):
    """
    https://arxiv.org/abs/1705.09792
    https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801
    """
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x))

class ComplexConv2d(Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        stride = 1,
        padding = 0
    ):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype = torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))

        x = x.to(weight.dtype)
        return F.conv2d(x, weight, bias, stride = self.stride, padding = self.padding)

def ComplexSTFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        Residual(Sequential(
            ComplexConv2d(chan_in, chan_in, 3, padding = 1),
            ModReLU(),
            ComplexConv2d(chan_in, chan_in, 3, padding = 1)
        )),
        ComplexConv2d(chan_in, chan_out, kernel_sizes, stride = strides, padding = paddings)
    )

class ComplexSTFTDiscriminator(Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = ((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)),
        chan_mults = (1, 2, 4, 4, 8, 8),
        input_channels = 1,
        n_fft = 1024,
        hop_length = 256,
        win_length = 1024,
        stft_normalized = False,
        stft_window_fn = torch.hann_window,
        logits_abs = False
    ):
        super().__init__()
        self.init_conv = ComplexConv2d(input_channels, channels, 7, padding = 3)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        curr_channels = channels

        self.layers = ModuleList([])

        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            self.layers.append(ComplexSTFTResidualUnit(chan_in, chan_out, layer_stride))

        self.final_conv = ComplexConv2d(layer_channels[-1], 1, (16, 1)) # todo: remove hardcoded 16

        # stft settings

        self.stft_normalized = stft_normalized
        self.stft_window_fn = stft_window_fn

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # how to output the logits into real space

        self.logits_abs = logits_abs

    def forward(self, x, return_intermediates = False):
        x = rearrange(x, 'b 1 n -> b n')

        '''
        reference: The content of the paper( https://arxiv.org/pdf/2107.03312.pdf)is as follows:
        The STFT-based discriminator is illustrated in Figure 4
        and operates on a single scale, computing the STFT with a
        window length of W = 1024 samples and a hop length of
        H = 256 samples
        '''

        stft_window = self.stft_window_fn(self.win_length, device = x.device)

        x = torch.stft(
            x,
            self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = stft_window,
            normalized = self.stft_normalized,
            return_complex = True
        )

        x = rearrange(x, 'b ... -> b 1 ...')

        intermediates = []

        x = self.init_conv(x)

        intermediates.append(x)

        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        complex_logits = self.final_conv(x)

        if self.logits_abs:
            # Kept only as a compatibility option.  Magnitude logits are
            # non-negative and therefore cannot satisfy the fake branch of a
            # hinge discriminator (which requires D(fake) <= -1).
            complex_logits = complex_logits.abs()
        else:
            # Hinge GAN needs one signed real-valued score per logit.  Do not
            # use view_as_real here: treating real and imaginary parts as two
            # independent logits gives the imaginary component an unrelated
            # hinge target.
            complex_logits = complex_logits.real

        if not return_intermediates:
            return complex_logits

        return complex_logits, intermediates

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
    def __init__(self, chan_in, chan_out, kernel_size, stride, pad_mode = 'reflect', **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.conv = CausalConv1d(chan_in, chan_out, kernel_size, pad_mode = pad_mode, **kwargs)

    def _upsample(self, x, prev = None):
        factor = self.upsample_factor

        if factor == 1:
            return x

        if not exists(prev):
            prev = x[..., :1]

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

        out = start[..., None] * (1. - weights) + end[..., None] * weights
        return rearrange(out, 'b c n s -> b c (n s)')

    def forward(self, x):
        x = self._upsample(x)
        return self.conv(x)

    def forward_stream(self, x, state = None):
        prev = None
        conv_state = None

        if exists(state):
            prev, conv_state = state

        upsampled = self._upsample(x, prev = prev)
        out, next_conv_state = self.conv.forward_stream(upsampled, conv_state)
        next_prev = x[..., -1:]

        return out, (next_prev, next_conv_state)

def ResidualUnit(chan_in, chan_out, dilation, kernel_size = 7, squeeze_excite = False, pad_mode = 'reflect', residual_scale = 1.):
    return Residual(Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation = dilation, pad_mode = pad_mode),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1, pad_mode = pad_mode),
        nn.ELU(),
        SqueezeExcite(chan_out) if squeeze_excite else None
    ), scale = residual_scale)

def EncoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9), squeeze_excite = False, pad_mode = 'reflect'):
    it = cycle(cycle_dilations)
    residual_unit = partial(ResidualUnit, squeeze_excite = squeeze_excite, pad_mode = pad_mode)

    return nn.Sequential(
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride = stride, pad_mode = pad_mode)
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
    linear_upsample_kernel_min = 0
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

    if upsample_mode == 'convtranspose':
        upsample = CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride)
    elif upsample_mode == 'linear':
        upsample_kernel_size = max(2 * stride, linear_upsample_kernel_min)
        upsample = CausalLinearUpsampleConv1d(chan_in, chan_out, upsample_kernel_size, stride = stride, pad_mode = pad_mode)
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
    if isinstance(module, (CausalConv1d, CausalConvTranspose1d, CausalLinearUpsampleConv1d)):
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
        use_lookup_free_quantizer = False,              # proposed in https://arxiv.org/abs/2310.05737, adapted for residual quantization
        use_finite_scalar_quantizer = False,            # proposed in https://arxiv.org/abs/2309.15505, adapted for residual quantization
        input_channels = 1,
        discr_multi_scales = (1, 0.5, 0.25),
        stft_normalized = False,
        enc_cycle_dilations = (1, 3, 9),
        dec_cycle_dilations = (1, 3, 9),
        decoder_upsample_mode = 'convtranspose',
        decoder_residual_scale = 1.,
        decoder_block_residual_scales: tuple[float, ...] | None = None,
        decoder_linear_upsample_kernel_min = 0,
        multi_spectral_window_powers_of_two = tuple(range(6, 12)),
        multi_spectral_n_ffts = 512,
        multi_spectral_n_mels = 64,
        recon_loss_weight = 1.,
        multi_spectral_recon_loss_weight = 1.,
        stft_recon_loss_weight = 0.,
        spectral_envelope_loss_weight = 0.,
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
        self.decoder_upsample_mode = decoder_upsample_mode
        self.decoder_residual_scale = float(decoder_residual_scale)
        self.decoder_block_residual_scales = tuple(
            float(scale) for scale in default(
                decoder_block_residual_scales,
                (decoder_residual_scale,) * len(strides)
            )
        )
        self.decoder_linear_upsample_kernel_min = int(decoder_linear_upsample_kernel_min)

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride, enc_cycle_dilations, squeeze_excite, pad_mode))

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

        for ((chan_in, chan_out), layer_stride), residual_scale in zip(
            zip(reversed(chan_in_out_pairs), reversed(strides)),
            self.decoder_block_residual_scales
        ):
            decoder_blocks.append(DecoderBlock(
                chan_out,
                chan_in,
                layer_stride,
                dec_cycle_dilations,
                squeeze_excite,
                pad_mode,
                upsample_mode = decoder_upsample_mode,
                residual_scale = residual_scale,
                linear_upsample_kernel_min = decoder_linear_upsample_kernel_min
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
        self.spectral_envelope_smoothing_bins = 9
        self.spectral_envelope_min_hz = 200.
        self.spectral_envelope_max_hz = 4500.
        self.spectral_envelope_relative_rms_db = -35.
        self.spectral_envelope_absolute_rms = 0.003
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
            256: 1.00,
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
        """Penalize residual energy that is phase-locked to the codec frame period."""
        frame_samples = self.frame_phase_samples
        usable_samples = (min(target.shape[-1], recon.shape[-1]) // frame_samples) * frame_samples
        if usable_samples < frame_samples * 2:
            return self.zero

        residual = (recon[..., :usable_samples] - target[..., :usable_samples]).float()
        residual_frames = residual.reshape(*residual.shape[:-1], -1, frame_samples)
        phase_pattern = residual_frames.mean(dim = -2)
        phase_pattern = phase_pattern - phase_pattern.mean(dim = -1, keepdim = True)
        phase_rms = phase_pattern.square().mean(dim = -1).clamp_min(1e-12).sqrt()
        target_rms = target[..., :usable_samples].float().square().mean(dim = -1).clamp_min(1e-8).sqrt()
        return (phase_rms / target_rms).mean()

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
        """Return voiced, frequency-smoothed vocal-tract envelope errors.

        The per-frame mean log magnitude is removed before comparison so this
        objective follows the formant-envelope shape rather than absolute
        loudness or a stationary recording-noise floor.
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
        target_log_mag = log(
            torch.stft(target_rows, **stft_kwargs).abs(),
            eps = 1e-5
        )
        recon_log_mag = log(
            torch.stft(recon_rows, **stft_kwargs).abs(),
            eps = 1e-5
        )

        smoothing_bins = self.spectral_envelope_smoothing_bins
        smooth_padding = smoothing_bins // 2
        target_envelope = F.avg_pool2d(
            target_log_mag.unsqueeze(1),
            kernel_size = (smoothing_bins, 1),
            stride = 1,
            padding = (smooth_padding, 0)
        ).squeeze(1)
        recon_envelope = F.avg_pool2d(
            recon_log_mag.unsqueeze(1),
            kernel_size = (smoothing_bins, 1),
            stride = 1,
            padding = (smooth_padding, 0)
        ).squeeze(1)

        frequencies = torch.fft.rfftfreq(
            self.spectral_envelope_n_fft,
            d = 1. / self.target_sample_hz,
            device = target_rows.device
        )
        voice_band = (
            (frequencies >= self.spectral_envelope_min_hz) &
            (frequencies <= self.spectral_envelope_max_hz)
        )
        target_envelope = target_envelope[:, voice_band]
        recon_envelope = recon_envelope[:, voice_band]

        # Remove frame-level gain so the loss focuses on envelope shape.
        target_envelope = target_envelope - target_envelope.mean(
            dim = 1,
            keepdim = True
        )
        recon_envelope = recon_envelope - recon_envelope.mean(
            dim = 1,
            keepdim = True
        )
        envelope_error = (recon_envelope - target_envelope).abs()

        frame_rms = F.avg_pool1d(
            target_rows.square().unsqueeze(1),
            kernel_size = self.spectral_envelope_win_length,
            stride = self.spectral_envelope_hop_length,
            padding = self.spectral_envelope_win_length // 2,
            count_include_pad = False
        ).squeeze(1).clamp_min(1e-12).sqrt()
        num_frames = min(frame_rms.shape[-1], envelope_error.shape[-1])
        frame_rms = frame_rms[..., :num_frames]
        envelope_error = envelope_error[..., :num_frames]
        relative_floor = (
            frame_rms.amax(dim = -1, keepdim = True) *
            (10. ** (self.spectral_envelope_relative_rms_db / 20.))
        )
        voiced_mask = (
            (frame_rms >= relative_floor) &
            (frame_rms >= self.spectral_envelope_absolute_rms)
        )
        voiced_weight = voiced_mask.unsqueeze(1).to(envelope_error.dtype)

        def masked_band_mean(error, band_mask):
            band_error = error[:, band_mask]
            denominator = (
                voiced_weight.sum() * max(band_error.shape[1], 1)).clamp_min(1.)
            return (band_error * voiced_weight).sum() / denominator

        voice_frequencies = frequencies[voice_band]
        total_loss = masked_band_mean(
            envelope_error,
            torch.ones_like(voice_frequencies, dtype = torch.bool)
        )
        low_loss = masked_band_mean(
            envelope_error,
            (voice_frequencies >= 200.) & (voice_frequencies < 1000.)
        )
        mid_loss = masked_band_mean(
            envelope_error,
            (voice_frequencies >= 1000.) & (voice_frequencies < 2500.)
        )
        high_loss = masked_band_mean(
            envelope_error,
            (voice_frequencies >= 2500.) & (voice_frequencies <= 4500.)
        )
        voiced_fraction = voiced_mask.float().mean()
        return total_loss, low_loss, mid_loss, high_loss, voiced_fraction
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

        feature_losses = [
            F.l1_loss(real_feature, fake_feature)
            for real_features, fake_features in discriminator_intermediates
            for real_feature, fake_feature in zip(real_features, fake_features)
        ]
        feature_loss = torch.stack(feature_losses).mean()
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

        if not self.use_finite_scalar_quantizer:
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

        if exists(num_quantizers):
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
            real, fake = orig_x, recon_x.detach()

            stft_discr_loss = None
            stft_grad_penalty = None
            discr_losses = []
            discr_grad_penalties = []

            if self.single_channel:
                real, fake = orig_x.clone(), recon_x.detach()
                stft_real_logits, stft_fake_logits = map(self.stft_discriminator, (real.requires_grad_(), fake.requires_grad_()))
                stft_discr_loss = hinge_discr_loss(stft_fake_logits, stft_real_logits)

                if apply_grad_penalty:
                    stft_grad_penalty = gradient_penalty(real, stft_discr_loss) + gradient_penalty(fake, stft_discr_loss)

            scaled_real, scaled_fake = real, fake
            for discr, downsample in zip(self.discriminators, self.downsamples):
                scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

                real_logits, fake_logits = map(discr, (scaled_real.requires_grad_(), scaled_fake.requires_grad_()))
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                if apply_grad_penalty:
                    discr_grad_penalties.extend([
                        gradient_penalty(scaled_real, one_discr_loss),
                        gradient_penalty(scaled_fake, one_discr_loss)
                    ])

            if not return_discr_losses_separately:
                all_discr_losses = torch.stack(discr_losses).mean()

                if exists(stft_discr_loss):
                    all_discr_losses = all_discr_losses + stft_discr_loss

                if exists(stft_grad_penalty):
                    all_discr_losses = all_discr_losses + stft_grad_penalty

                return all_discr_losses

            # return a list of discriminator losses with List[Tuple[str, Tensor]]

            discr_losses_pkg = []

            discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in zip(self.discr_multi_scales, discr_losses)])

            discr_losses_pkg.extend([(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in zip(self.discr_multi_scales, discr_grad_penalties)])

            if exists(stft_discr_loss):
                discr_losses_pkg.append(('stft', stft_discr_loss))

            if exists(stft_grad_penalty):
                discr_losses_pkg.append(('stft_grad_penalty', stft_grad_penalty))

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
            if self.spectral_envelope_loss_weight > 0
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
                adversarial_loss,
                feature_loss,
                all_commitment_loss,
                si_sdr_loss,
                correlation_loss,
                click_loss,
                jump_loss,
                preemph_loss,
                noise_floor_loss,
                frame_phase_loss
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stream_frame_size = stream_frame_size
        self.stream_context_frames = stream_context_frames
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_loss_radius = boundary_loss_radius

        config = pickle.loads(self._configs)
        config['stream_frame_size'] = stream_frame_size
        config['stream_context_frames'] = stream_context_frames
        config['boundary_loss_weight'] = boundary_loss_weight
        config['boundary_loss_radius'] = boundary_loss_radius
        self._configs = pickle.dumps(config)

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

        if not self.use_finite_scalar_quantizer:
            rq_kwargs = (
                dict(freeze_codebook = freeze_codebook)
                if not self.use_lookup_free_quantizer
                else {}
            )
            x, indices, commit_loss = self.rq(x, **rq_kwargs)
        else:
            x, indices = self.rq(x)
            commit_loss = self.zero

        if exists(num_quantizers):
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
        if not self.use_finite_scalar_quantizer:
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

        if exists(num_quantizers):
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
            real, fake = orig_x, recon_x.detach()

            stft_discr_loss = None
            stft_grad_penalty = None
            discr_losses = []
            discr_grad_penalties = []

            if self.single_channel:
                real, fake = orig_x.clone(), recon_x.detach()
                stft_real_logits, stft_fake_logits = map(self.stft_discriminator, (real.requires_grad_(), fake.requires_grad_()))
                stft_discr_loss = hinge_discr_loss(stft_fake_logits, stft_real_logits)

                if apply_grad_penalty:
                    stft_grad_penalty = gradient_penalty(real, stft_discr_loss) + gradient_penalty(fake, stft_discr_loss)

            scaled_real, scaled_fake = real, fake
            for discr, downsample in zip(self.discriminators, self.downsamples):
                scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

                real_logits, fake_logits = map(discr, (scaled_real.requires_grad_(), scaled_fake.requires_grad_()))
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                if apply_grad_penalty:
                    discr_grad_penalties.extend([
                        gradient_penalty(scaled_real, one_discr_loss),
                        gradient_penalty(scaled_fake, one_discr_loss)
                    ])

            if not return_discr_losses_separately:
                all_discr_losses = torch.stack(discr_losses).mean()

                if exists(stft_discr_loss):
                    all_discr_losses = all_discr_losses + stft_discr_loss

                if exists(stft_grad_penalty):
                    all_discr_losses = all_discr_losses + stft_grad_penalty

                return all_discr_losses

            discr_losses_pkg = []

            discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in zip(self.discr_multi_scales, discr_losses)])

            discr_losses_pkg.extend([(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in zip(self.discr_multi_scales, discr_grad_penalties)])

            if exists(stft_discr_loss):
                discr_losses_pkg.append(('stft', stft_discr_loss))

            if exists(stft_grad_penalty):
                discr_losses_pkg.append(('stft_grad_penalty', stft_grad_penalty))

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
            if self.spectral_envelope_loss_weight > 0
            else self.zero
        )
        adversarial_loss, feature_loss = self.generator_perceptual_losses(
            orig_x,
            recon_x
        )

        boundary_loss = self.frame_boundary_loss(recon_x)

        total_loss = (
            recon_loss * self.recon_loss_weight +
            multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight +
            stft_recon_loss * self.stft_recon_loss_weight +
            si_sdr_loss * self.si_sdr_loss_weight +
            spectral_envelope_loss * self.spectral_envelope_loss_weight +
            correlation_loss * self.correlation_loss_weight +
            click_loss * self.click_loss_weight +
            jump_loss * self.jump_loss_weight +
            preemph_loss * self.preemph_loss_weight +
            noise_floor_loss * self.noise_floor_loss_weight +
            adversarial_loss * self.adversarial_loss_weight +
            feature_loss * self.feature_loss_weight +
            all_commitment_loss * self.commitment_loss_weight +
            boundary_loss * self.boundary_loss_weight
        )

        if return_loss_breakdown:
            return total_loss, (
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
                boundary_loss
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
