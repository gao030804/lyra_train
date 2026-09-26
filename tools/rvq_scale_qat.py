"""Scale-only surrogate and fixed W8A16 projection contract."""
from __future__ import annotations

import math
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tools.integer_rvq_reference import round_away, ratio_parameters, requant, clip16


def fake_quant(x, scale, low, high):
    normalized = x / scale
    rounded = normalized.sign() * (normalized.abs() + .5).floor()
    integer = (normalized + (rounded-normalized).detach()).clamp(low, high)
    return integer * scale


class ScaleRVQ(nn.Module):
    def __init__(self, books, scales, output_scale, max_ratio=1.15, topk=8, temperature=.1, scale_mode='v1'):
        super().__init__()
        if max_ratio <= 1 or temperature <= 0 or topk < 2:
            raise ValueError('Invalid scale bounds or distance distillation configuration')
        self.register_buffer('initial', torch.tensor(scales, dtype=torch.float64))
        self.register_buffer('output_scale', torch.tensor(output_scale, dtype=torch.float64))
        for q, book in enumerate(books):
            self.register_buffer(f'book_{q}', torch.as_tensor(book, dtype=torch.float64))
        self.delta = nn.Parameter(torch.zeros(len(scales), dtype=torch.float64))
        self.log_range = math.log(max_ratio)
        self.topk, self.temperature = topk, temperature
        if scale_mode not in ('v1', 'v2'):
            raise ValueError('scale_mode must be v1 or v2')
        self.scale_mode = scale_mode

    def scales(self):
        # Exact interval [s0 / max_ratio, s0 * max_ratio].
        return self.initial * torch.exp(self.log_range * torch.tanh(self.delta))

    def forward(self, query, teacher_query):
        scales = self.scales()
        residual_scales = scales if self.scale_mode == 'v1' else self.initial
        residual = fake_quant(query.double(), residual_scales[0], -32768, 32767)
        teacher_residual = teacher_query.detach().double()
        quantized = torch.zeros_like(residual)
        dist_loss, cb_loss = residual.new_zeros(()), residual.new_zeros(())
        weights = (1.,1.,.8,.65,.55,.45,.35,.30,.25)
        for q, scale in enumerate(scales):
            book = getattr(self, f'book_{q}')
            qb = fake_quant(book, scale, -128, 127)
            search = fake_quant(residual, scale, -32768, 32767)
            distances = (search.unsqueeze(-2)-qb).square().sum(-1)
            with torch.no_grad():
                td = (teacher_residual.unsqueeze(-2)-book).square().sum(-1)
                tk = td.argmin(-1)
                candidates = td.topk(min(self.topk, book.shape[0]), largest=False).indices
                td_top = td.gather(-1, candidates)
                normalization = td_top.mean(-1, keepdim=True).clamp_min(1e-8)
                prob = (-td_top/normalization/self.temperature).softmax(-1)
            student_top = distances.gather(-1, candidates)
            dist_loss = dist_loss + weights[min(q,8)] * F.kl_div(
                (-student_top/normalization/self.temperature).log_softmax(-1), prob,
                reduction='none').sum(-1).mean()
            selected = qb[distances.detach().argmin(-1)]
            quantized = quantized + fake_quant(selected, self.output_scale, -2**31, 2**31-1)
            residual = residual - (selected if self.scale_mode == 'v1' else fake_quant(selected, self.initial[q], -32768,32767))
            if q+1 < len(scales):
                residual = fake_quant(residual, residual_scales[q+1], -32768,32767)
            teacher_residual = teacher_residual - book[tk]
            cb_loss = cb_loss + (qb-book).square().mean()/book.square().mean().clamp_min(1e-12)
        quantized = fake_quant(quantized, self.output_scale, -32768,32767)
        return quantized, dist_loss/sum(weights[min(q,8)] for q in range(len(scales))), cb_loss/len(scales)


def make_projection(weight, bias, input_scale, output_scale):
    w = np.asarray(weight, dtype=float)
    ws = np.maximum(np.abs(w).max(-1)/127., 1e-12)
    qw = np.clip(round_away(w/ws[:,None]),-128,127).astype(np.int8)
    qb = round_away(np.zeros(w.shape[0]) if bias is None else np.asarray(bias)/(input_scale*ws)).astype(np.int64)
    if np.any((qb < -2**39)|(qb > 2**39-1)):
        raise OverflowError('Projection bias exceeds ACC40')
    params = [ratio_parameters(input_scale*s/output_scale) for s in ws]
    return dict(weight=qw, bias=qb, weight_scale=ws, input_scale=float(input_scale),
                output_scale=float(output_scale), multiplier=np.array([p[0] for p in params]),
                shift=np.array([p[1] for p in params]))


def project_integer(x, spec):
    if np.asarray(x).dtype != np.int16:
        raise ValueError('Projection input must be INT16')
    acc = np.asarray(x,dtype=np.int64) @ spec['weight'].astype(np.int64).T + spec['bias']
    if np.any((acc < -2**39)|(acc > 2**39-1)):
        raise OverflowError('Projection accumulation exceeds ACC40')
    channels=[]
    for i,(m,s) in enumerate(zip(spec['multiplier'],spec['shift'])):
        m,s=int(m),int(s)
        # ACC40 * INT32 can require 72 bits. Python integers avoid wraparound.
        def scalar(value):
            product=int(value)*m
            magnitude=(abs(product)+((1<<(s-1)) if s else 0))>>s
            return -magnitude if product<0 else magnitude
        values=[scalar(v) for v in acc[...,i].reshape(-1)]
        channels.append(np.asarray(values,dtype=np.int64).reshape(acc.shape[:-1]))
    raw=np.stack(channels,axis=-1)
    return clip16(raw), int(np.count_nonzero((raw < -32768)|(raw > 32767)))


def projection_output_scale(spec, output_scale):
    """Change only requant units, never weights/bias or the projection geometry."""
    params = [ratio_parameters(spec['input_scale'] * s / output_scale) for s in spec['weight_scale']]
    return dict(spec, output_scale=float(output_scale), multiplier=np.array([p[0] for p in params]),
                shift=np.array([p[1] for p in params]))


def audio_first_key(summary):
    return (-summary['mean_si_sdr_delta'], -summary['p10_si_sdr_delta'],
            -summary['worst_si_sdr_delta'], -summary['mean_corr_delta'], summary['mean_vqout_nmse'])


def export_projection(directory, spec):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    k,d = spec['weight'].shape
    packed = np.zeros(((k+7)//8,(d+3)//4,4,8),dtype=np.int8)
    for c in range(k):
        for j in range(d):
            packed[c//8,j//4,j%4,c%8] = spec['weight'][c,j]
    packed.tofile(root/'projection_weights_4x8_int8.bin')
    spec['bias'].astype('<i8').tofile(root/'projection_bias_int40_in_int64.bin')
    spec['multiplier'].astype('<i4').tofile(root/'projection_multiplier_int32.bin')
    spec['shift'].astype('u1').tofile(root/'projection_shift_uint8.bin')
    manifest = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in spec.items() if k not in ('weight','bias')}
    manifest.update(cin=d,cout=k,weight_bits=8,activation_bits=16,accumulator_bits=40,
                    rounding='half_away_from_zero',zero_point=0,layout='output_group,reduction_group,4,8')
    (root/'projection_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def integer_encoder(model, wave):
    """Recursively evaluate actual module graph, including residual/ReLU edges."""
    from audiolm_pytorch.soundstream import CausalConv1d, Residual, DepthwiseSeparableCausalConv1d, LowRankPointwiseCausalConv1d
    from audiolm_pytorch.hardware_quantization import HardwareReLU
    from tools.export_encoder_integer_goldens import integer_conv1d_layer, align_to_scale, quantize_activation
    first = next(m for m in model.encoder.modules() if isinstance(m,CausalConv1d))
    scale = float(first.hardware_input_fake_quant.scale.item())
    x = quantize_activation(wave.cpu(),first.hardware_input_fake_quant.scale.cpu(),16)
    def walk(module,x,scale):
        if isinstance(module,(DepthwiseSeparableCausalConv1d,LowRankPointwiseCausalConv1d)):
            return walk(module.net,x,scale)
        if isinstance(module,nn.Sequential):
            for child in module:
                x,scale=walk(child,x,scale)
            return x,scale
        if isinstance(module,CausalConv1d):
            ins=float(module.hardware_input_fake_quant.scale.item())
            if not math.isclose(scale,ins,rel_tol=1e-6):
                raise RuntimeError(f'Unspecified Encoder edge scale {scale} -> {ins}')
            return integer_conv1d_layer(x,module,16,40),float(module.hardware_output_fake_quant.scale.item())
        if isinstance(module,HardwareReLU):
            out=float(module.fake_quant.scale.item())
            return align_to_scale(x,scale/out).clamp(0,32767).short(),out
        if isinstance(module,Residual):
            branch,bs=walk(module.fn,x.clone(),scale)
            out=float(module.hardware_residual_add.fake_quant.scale.item())
            identity=align_to_scale(x,scale/out).clamp(-32768,32767)
            branch=align_to_scale(branch,bs/out).clamp(-32768,32767)
            branch=align_to_scale(branch,float(module.residual_scale))
            return (identity+branch).clamp(-32768,32767).short(),out
        if isinstance(module,nn.Identity):
            return x,scale
        raise TypeError(f'Unsupported integer Encoder module: {type(module).__name__}')
    return walk(model.encoder,x,scale)


def retention_summary(rows):
    sdr=np.array([r['aligned_si_sdr_delta'] for r in rows])
    errors=np.array([r['vqout_nmse'] for r in rows])
    result=dict(mean_si_sdr_delta=float(sdr.mean()),median_si_sdr_delta=float(np.median(sdr)),
                p10_si_sdr_delta=float(np.percentile(sdr,10)),worst_si_sdr_delta=float(sdr.min()),
                mean_corr_delta=float(np.mean([r['aligned_corr_delta'] for r in rows])),
                mean_vqout_nmse=float(errors.mean()),p90_vqout_nmse=float(np.percentile(errors,90)),
                worst_vqout_nmse=float(errors.max()),saturation=sum(r['saturation'] for r in rows))
    result['passed']=bool(result['mean_si_sdr_delta'] >= -.05 and result['p10_si_sdr_delta'] >= -.20
                          and result['worst_si_sdr_delta'] >= -.30 and result['mean_vqout_nmse'] <= .04
                          and result['p90_vqout_nmse'] <= .08 and result['saturation']==0)
    return result
