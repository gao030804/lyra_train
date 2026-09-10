from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from audiolm_pytorch.hardware_quantization import (
    INT8_QMAX,
    INT8_QMIN,
    INT32_QMAX,
    INT32_QMIN,
    approximate_requant_multiplier,
    quantize_bias_int32,
    round_half_away_from_zero,
)


def _int_tensor_bytes(tensor: torch.Tensor, dtype: torch.dtype) -> bytes:
    return tensor.detach().cpu().to(dtype).contiguous().numpy().tobytes()


def pack_conv1d_weights_4x8(qweight: torch.Tensor) -> torch.Tensor:
    """Pack [Cout,Cin,K] as RTL [o_group][k_group][4][8]."""
    if qweight.ndim != 3:
        raise ValueError(f"expected [Cout,Cin,K], got {tuple(qweight.shape)}")
    cout, cin, kernel = map(int, qweight.shape)
    k_total = cin * kernel
    k_groups = math.ceil(k_total / 4)
    n_groups = math.ceil(cout / 8)
    packed = torch.zeros(n_groups, k_groups, 4, 8, dtype=torch.int8)
    source = qweight.detach().cpu().to(torch.int8)
    for n in range(cout):
        for k in range(kernel):
            for c in range(cin):
                reduction_index = k * cin + c
                packed[n // 8, reduction_index // 4, reduction_index % 4, n % 8] = (
                    source[n, c, k]
                )
    return packed.contiguous()


def integer_conv1d_reference(
    qinput: torch.Tensor,
    qweight: torch.Tensor,
    qbias: torch.Tensor | None,
    multiplier: torch.Tensor,
    shift: torch.Tensor,
    *,
    stride: int = 1,
    dilation: int = 1,
    left_pad: int = 0,
) -> dict[str, torch.Tensor]:
    """Pure integer Conv1d reference following the RTL reduction order.

    Accumulation is performed in INT64 and checked after every 4-element
    k-group.  The RTL does not saturate accumulator additions, so overflow is
    an error rather than something this reference silently clamps.
    """
    if qinput.ndim != 3 or qweight.ndim != 3:
        raise ValueError("qinput and qweight must be [B,C,T] and [Cout,Cin,K]")
    batch, cin, input_length = map(int, qinput.shape)
    cout, weight_cin, kernel = map(int, qweight.shape)
    if cin != weight_cin:
        raise ValueError(f"input channels {cin} != weight channels {weight_cin}")
    output_length = (input_length + left_pad - dilation * (kernel - 1) - 1) // stride + 1
    if output_length <= 0:
        raise ValueError("non-positive Conv1d output length")

    x = qinput.detach().cpu().to(torch.int64)
    w = qweight.detach().cpu().to(torch.int64)
    bias = (
        torch.zeros(cout, dtype=torch.int64)
        if qbias is None
        else qbias.detach().cpu().to(torch.int64).flatten()
    )
    accumulator = torch.zeros(batch, cout, output_length, dtype=torch.int64)
    reduction = [(k, c) for k in range(kernel) for c in range(cin)]
    for group_start in range(0, len(reduction), 4):
        partial = torch.zeros_like(accumulator)
        for k, c in reduction[group_start:(group_start + 4)]:
            for m in range(output_length):
                source_index = m * stride - left_pad + k * dilation
                if 0 <= source_index < input_length:
                    partial[:, :, m] += (
                        x[:, c, source_index].view(batch, 1) * w[:, c, k].view(1, cout)
                    )
        if bool((partial < -(2 ** 19)).any() or (partial > (2 ** 19) - 1).any()):
            raise OverflowError("INT20 partial sum overflow")
        accumulator += partial
        if bool((accumulator < INT32_QMIN).any() or (accumulator > INT32_QMAX).any()):
            raise OverflowError("INT32 accumulator overflow")

    biased = (accumulator + bias.view(1, cout, 1)).clamp(INT32_QMIN, INT32_QMAX)
    mul = multiplier.detach().cpu().to(torch.int64).view(1, cout, 1)
    right_shift = shift.detach().cpu().to(torch.int64).view(1, cout, 1)
    product = biased * mul
    half = torch.where(
        right_shift > 0,
        torch.bitwise_left_shift(torch.ones_like(right_shift), (right_shift - 1).clamp_min(0)),
        torch.zeros_like(right_shift),
    )
    magnitude = torch.bitwise_right_shift(product.abs() + half, right_shift)
    requant = torch.where(product < 0, -magnitude, magnitude)
    output = requant.clamp(INT8_QMIN, INT8_QMAX).to(torch.int8)
    return {
        "accumulator_int32": accumulator.to(torch.int32),
        "biased_int32": biased.to(torch.int32),
        "requant_int8": output,
    }


def _conv_followed_by_hardware_relu(encoder: torch.nn.Module) -> set[str]:
    result: set[str] = set()
    for parent_name, parent in encoder.named_modules():
        if not isinstance(parent, torch.nn.Sequential):
            continue
        children = tuple(parent.named_children())
        for (producer_key, producer), (_, consumer) in zip(children, children[1:]):
            if (
                hasattr(producer, "hardware_output_fake_quant") and
                consumer.__class__.__name__ == "HardwareReLU"
            ):
                result.add(".".join(filter(None, (parent_name, producer_key))))
    return result


@torch.no_grad()
def export_hardware_encoder_package(
    model: torch.nn.Module,
    output_dir: str | Path,
) -> Path:
    """Export fixed integer parameters; refuse uncalibrated QAT state."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = model.encoder
    relu_layers = _conv_followed_by_hardware_relu(encoder)
    weight_blob = bytearray()
    bias_blob = bytearray()
    parameter_blob = bytearray()
    parameter_group_blob = bytearray()
    residual_blob = bytearray()
    layers: list[dict[str, Any]] = []

    for name, module in encoder.named_modules():
        if not (
            hasattr(module, "conv") and
            hasattr(module, "hardware_input_fake_quant") and
            hasattr(module, "hardware_output_fake_quant") and
            hasattr(module, "hardware_weight_fake_quant")
        ):
            continue
        if module.hardware_input_fake_quant is None or module.hardware_output_fake_quant is None:
            continue
        for label, quantizer in (
            ("input", module.hardware_input_fake_quant),
            ("output", module.hardware_output_fake_quant),
        ):
            if int(quantizer.num_observations.item()) <= 0:
                raise RuntimeError(f"{name}: {label} activation scale was never calibrated")
            if quantizer.observer_enabled:
                raise RuntimeError(f"{name}: freeze observers before exporting")

        conv = module.conv
        input_scale = module.hardware_input_fake_quant.scale.detach().float()
        output_scale = module.hardware_output_fake_quant.scale.detach().float()
        qweight, weight_scale = module.hardware_weight_fake_quant.quantize(conv.weight)
        qweight = qweight.round().clamp(INT8_QMIN, INT8_QMAX).to(torch.int8)
        qbias = quantize_bias_int32(conv.bias, input_scale, weight_scale)
        if qbias is None:
            qbias = torch.zeros(conv.out_channels, device=conv.weight.device)
        qbias = qbias.round().clamp(INT32_QMIN, INT32_QMAX).to(torch.int32)
        real_multiplier = input_scale * weight_scale.flatten() / output_scale
        _, multiplier, shift = approximate_requant_multiplier(real_multiplier)
        multiplier = multiplier.to(torch.int32)
        shift_u8 = shift.to(torch.uint8)
        packed_weight = pack_conv1d_weights_4x8(qweight)

        padded_cout = math.ceil(conv.out_channels / 8) * 8
        def pad_channels(tensor: torch.Tensor, value: int = 0) -> torch.Tensor:
            padding = padded_cout - tensor.numel()
            return F.pad(tensor.flatten(), (0, padding), value=value)

        padded_bias = pad_channels(qbias).to(torch.int32)
        padded_multiplier = pad_channels(multiplier).to(torch.int32)
        padded_shift = pad_channels(shift_u8).to(torch.uint8)

        weight_offset = len(weight_blob)
        bias_offset = len(bias_blob)
        parameter_offset = len(parameter_blob)
        parameter_group_offset = len(parameter_group_blob)
        weight_blob.extend(_int_tensor_bytes(packed_weight, torch.int8))
        bias_blob.extend(_int_tensor_bytes(padded_bias, torch.int32))
        parameter_blob.extend(_int_tensor_bytes(padded_multiplier, torch.int32))
        parameter_blob.extend(_int_tensor_bytes(padded_shift, torch.uint8))
        parameter_blob.extend(bytes(padded_cout))  # zero_point = 0
        for group in range(padded_cout // 8):
            channel_slice = slice(group * 8, (group + 1) * 8)
            parameter_group_blob.extend(_int_tensor_bytes(
                padded_bias[channel_slice], torch.int32
            ))
            parameter_group_blob.extend(_int_tensor_bytes(
                padded_multiplier[channel_slice], torch.int32
            ))
            parameter_group_blob.extend(_int_tensor_bytes(
                padded_shift[channel_slice], torch.uint8
            ))
            parameter_group_blob.extend(bytes(8))

        packed_bytes = packed_weight.numel()
        layers.append({
            "name": name,
            "cin": conv.in_channels,
            "cout": conv.out_channels,
            "kernel": conv.kernel_size[0],
            "stride": conv.stride[0],
            "dilation": conv.dilation[0],
            # RTL 必须据此选择 Dense 脉动阵列或 group-aware Depthwise 通路。
            "groups": conv.groups,
            "is_depthwise": (
                conv.groups == conv.in_channels and
                conv.out_channels == conv.in_channels
            ),
            "weight_cin_per_group": conv.weight.shape[1],
            "left_pad": module.causal_padding,
            "input_scale": float(input_scale.item()),
            "output_scale": float(output_scale.item()),
            "weight_scale": weight_scale.flatten().detach().cpu().tolist(),
            "zero_point": 0,
            "weight_layout": "output_group_k_group_4x8",
            "weight_offset": weight_offset,
            "weight_bytes": packed_bytes,
            "bias_offset": bias_offset,
            "parameter_offset": parameter_offset,
            "parameter_group_offset": parameter_group_offset,
            "parameter_group_bytes": (padded_cout // 8) * 80,
            "relu_enabled": name in relu_layers,
            "fits_128k_weight_bank": packed_bytes <= 128 * 1024,
        })

    if not layers:
        raise RuntimeError("model does not contain calibrated hardware-QAT Encoder layers")

    residuals: list[dict[str, Any]] = []
    for name, module in encoder.named_modules():
        residual_add = getattr(module, "hardware_residual_add", None)
        if residual_add is None:
            continue
        convs = [
            child for child in module.fn.modules()
            if hasattr(child, "conv") and hasattr(child, "hardware_output_fake_quant")
        ]
        if not convs:
            raise RuntimeError(f"{name}: hardware residual has no Conv1d branch")
        channels = int(convs[-1].conv.out_channels)
        multiplier, shift = residual_add.integer_parameters(
            channels,
            module.residual_scale,
        )
        offset = len(residual_blob)
        residual_blob.extend(_int_tensor_bytes(multiplier, torch.int32))
        residual_blob.extend(_int_tensor_bytes(shift, torch.uint8))
        residuals.append({
            "name": name,
            "channels": channels,
            "residual_scale": float(module.residual_scale),
            "parameter_offset": offset,
            "parameter_bytes": channels * 5,
            "accumulator": "int32",
            "output_saturation": [INT8_QMIN, INT8_QMAX],
        })

    (output_dir / "encoder_weights_int8.bin").write_bytes(weight_blob)
    (output_dir / "encoder_bias_int32.bin").write_bytes(bias_blob)
    (output_dir / "encoder_quant_params.bin").write_bytes(parameter_blob)
    (output_dir / "encoder_parameter_groups.bin").write_bytes(parameter_group_blob)
    (output_dir / "encoder_residual_params.bin").write_bytes(residual_blob)
    manifest = {
        "format_version": 1,
        "int8_range": [INT8_QMIN, INT8_QMAX],
        "zero_point": 0,
        "rounding": "half_away_from_zero",
        "weight_layout": "output_group_k_group_4x8",
        "parameter_group_layout": (
            "8xbias_int32,8xmultiplier_int32,8xshift_uint8,8xzero_point_int8"
        ),
        "layers": layers,
        "residuals": residuals,
    }
    manifest_path = output_dir / "encoder_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path
