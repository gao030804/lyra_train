from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from audiolm_pytorch.hardware_quantization import (
    HARDWARE_QUANTIZATION_CONTRACT_VERSION,
    INT8_QMIN,
    INT8_QMAX,
    approximate_requant_multiplier,
    quantize_bias_integer,
    round_half_away_from_zero,
    signed_integer_range,
)
from audiolm_pytorch.soundstream import CausalConv1d
from infer_soundstream import load_audio, load_checkpoint, soundstream_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one real PCM segment and per-Conv1d W8A16 activation "
            "goldens for RTL layer-by-layer comparison."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=320)
    parser.add_argument("--weights", choices=("online", "ema"), default="online")
    parser.add_argument(
        "--recover-missing-contract-version", action="store_true"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_int16(path: Path, values: torch.Tensor) -> None:
    array = values.detach().cpu().contiguous().numpy().astype("<i2", copy=False)
    path.write_bytes(array.tobytes(order="C"))


def write_hex16(path: Path, values: torch.Tensor) -> None:
    flat = values.detach().cpu().contiguous().view(-1).to(torch.int64)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for value in flat.tolist():
            handle.write(f"{value & 0xffff:04x}\n")


def quantize_activation(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    qmin, qmax = signed_integer_range(bits)
    scalar_scale = scale.detach().float().reshape(-1)
    if scalar_scale.numel() != 1 or not torch.isfinite(scalar_scale).all():
        raise RuntimeError(f"Expected one finite activation scale, got {scale}")
    if float(scalar_scale.item()) <= 0.0:
        raise RuntimeError(f"Activation scale must be positive, got {scale}")
    return round_half_away_from_zero(
        tensor.detach().float() / scalar_scale.item()
    ).clamp(qmin, qmax).to(torch.int16)


def full_qat_proven(package: dict) -> bool:
    progression = package.get("hardware_qat_progression_state") or {}
    accepted = progression.get("accepted_groups")
    progression_ok = (
        isinstance(accepted, (list, tuple))
        and len(accepted) == 6
        and all(bool(value) for value in accepted)
    )
    gated_best = package.get("weight_source") == "online_full_int8_qat_gated"
    return bool(progression_ok or gated_best)


def integer_conv1d_layer(
    qinput: torch.Tensor,
    module: CausalConv1d,
    activation_bits: int,
    accumulator_bits: int,
) -> torch.Tensor:
    """按部署RTL公式计算一层Conv1d，返回BCT布局的整数激活。

    这里不使用浮点Conv输出。输入、权重、Bias、Multiplier和Shift均按
    hardware_export.py的部署参数生成规则计算。INT64仅作为Python容器，
    每个需要受硬件位宽约束的边界都会显式检查或饱和。
    """
    conv = module.conv
    qmin, qmax = signed_integer_range(activation_bits)
    acc_min, acc_max = signed_integer_range(accumulator_bits)
    input_scale = module.hardware_input_fake_quant.scale.detach().float()
    output_scale = module.hardware_output_fake_quant.scale.detach().float()
    qweight, weight_scale = module.hardware_weight_fake_quant.quantize(conv.weight)
    qweight = qweight.round().clamp(INT8_QMIN, INT8_QMAX).to(torch.int64).cpu()
    qbias = quantize_bias_integer(
        conv.bias, input_scale, weight_scale,
        accumulator_bits=accumulator_bits,
    )
    if qbias is None:
        qbias = torch.zeros(conv.out_channels, dtype=torch.int64)
    qbias = qbias.round().clamp(acc_min, acc_max).to(torch.int64).cpu()
    _, multiplier, shift = approximate_requant_multiplier(
        input_scale * weight_scale.flatten() / output_scale
    )
    multiplier = multiplier.to(torch.int64).cpu()
    shift = shift.to(torch.int64).cpu()

    x = qinput.to(torch.int64).cpu()
    left_pad = int(module.causal_padding)
    if left_pad:
        # RTL causal AGU对负time地址使用mask补零，不使用reflect padding。
        x = torch.nn.functional.pad(x, (left_pad, 0), mode="constant", value=0)
    kernel = int(conv.kernel_size[0])
    stride = int(conv.stride[0])
    dilation = int(conv.dilation[0])
    output_length = (
        qinput.shape[-1] + left_pad - dilation * (kernel - 1) - 1
    ) // stride + 1
    accumulator = torch.zeros(
        qinput.shape[0], conv.out_channels, output_length, dtype=torch.int64
    )

    if conv.groups == 1:
        for tap in range(kernel):
            samples = x[:, :, tap * dilation:tap * dilation + output_length * stride:stride]
            contribution = torch.matmul(
                samples.permute(0, 2, 1), qweight[:, :, tap].t()
            ).permute(0, 2, 1)
            accumulator += contribution
            if bool(((accumulator < acc_min) | (accumulator > acc_max)).any()):
                raise OverflowError(
                    f"{module}: INT{accumulator_bits} accumulator overflow"
                )
    elif conv.groups == conv.in_channels == conv.out_channels:
        for tap in range(kernel):
            samples = x[:, :, tap * dilation:tap * dilation + output_length * stride:stride]
            accumulator += samples * qweight[:, 0, tap].view(1, -1, 1)
            if bool(((accumulator < acc_min) | (accumulator > acc_max)).any()):
                raise OverflowError(
                    f"{module}: INT{accumulator_bits} depthwise accumulator overflow"
                )
    else:
        raise RuntimeError(
            f"Unsupported grouped Conv1d: groups={conv.groups}, "
            f"Cin={conv.in_channels}, Cout={conv.out_channels}"
        )

    biased = (accumulator + qbias.view(1, -1, 1)).clamp(acc_min, acc_max)
    # ACC40×INT32需要72位，用Python int避免torch INT64静默溢出。
    result = torch.empty_like(biased, dtype=torch.int16)
    for b in range(biased.shape[0]):
        for c in range(biased.shape[1]):
            for t in range(biased.shape[2]):
                value = round_product(int(biased[b,c,t]), int(multiplier[c]), int(shift[c]))
                result[b,c,t] = max(qmin, min(qmax, value))
    return result


def round_product(value, multiplier, shift):
    product = int(value) * int(multiplier)
    magnitude = (abs(product) + ((1 << (shift-1)) if shift else 0)) >> shift
    return -magnitude if product < 0 else magnitude


def align_to_scale(tensor, ratio):
    import math
    shift = 48
    while shift and math.floor(ratio * (1 << shift) + .5) > 0x7fffffff:
        shift -= 1
    multiplier = math.floor(ratio * (1 << shift) + .5)
    if ratio < 0 or multiplier > 0x7fffffff:
        raise ValueError("Residual ratio cannot be represented")
    values = [round_product(v, multiplier, shift) for v in tensor.flatten().tolist()]
    return torch.tensor(values, dtype=torch.int64).reshape(tensor.shape)


def main() -> None:
    args = parse_args()
    if args.start_sample < 0:
        raise ValueError("--start-sample must be non-negative")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    checkpoint = args.checkpoint.expanduser().resolve()
    audio_path = args.audio.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    package = load_checkpoint(checkpoint)
    contract = package.get("hardware_quantization_contract_version")
    recovered = contract is None and args.recover_missing_contract_version
    if contract != HARDWARE_QUANTIZATION_CONTRACT_VERSION and not recovered:
        raise RuntimeError(
            f"Checkpoint contract {contract!r} is not "
            f"{HARDWARE_QUANTIZATION_CONTRACT_VERSION}; use the explicit "
            "recovery flag only for the verified missing-tag QAT checkpoint."
        )
    if not full_qat_proven(package):
        raise RuntimeError("Checkpoint is not a validation-gated full-QAT artifact")

    model = soundstream_from_checkpoint(
        package, use_ema=args.weights == "ema"
    )
    if int(model.hardware_qat_activation_bits) != 16:
        raise RuntimeError(
            f"This exporter requires W8A16, got A{model.hardware_qat_activation_bits}"
        )
    if int(model.hardware_qat_accumulator_bits) != 40:
        raise RuntimeError(
            f"This exporter requires ACC40, got ACC{model.hardware_qat_accumulator_bits}"
        )
    model.set_hardware_qat_sensitivity_group(-1, accepted_groups=range(6))
    if not model.hardware_qat_is_full():
        raise RuntimeError("Could not restore the full hardware-QAT state")
    model.eval()

    wave = load_audio(audio_path, int(model.target_sample_hz))
    end_sample = args.start_sample + args.num_samples
    if wave.shape[-1] < end_sample:
        raise RuntimeError(
            f"Audio has {wave.shape[-1]} samples, requested through {end_sample}"
        )
    segment = wave[..., args.start_sample:end_sample].contiguous()
    if segment.ndim != 2:
        raise RuntimeError(
            f"Expected loaded audio in [channels, time], got {tuple(segment.shape)}"
        )
    # ``infer_soundstream.load_audio`` returns [channels, time], whereas the
    # Encoder contract is [batch, channels, time].  Calling Encoder directly
    # without this batch axis makes the residual requantizer interpret the
    # 320-sample time dimension as channels.
    segment = segment.unsqueeze(0)

    pcm = round_half_away_from_zero(
        segment.clamp(-1.0, 32767.0 / 32768.0) * 32768.0
    ).clamp(-32768, 32767).to(torch.int16)
    write_int16(output_dir / "input_pcm_s16le.bin", pcm)
    write_hex16(output_dir / "input_pcm_s16le.hex", pcm)

    layer_modules = [
        (name, module)
        for name, module in model.encoder.named_modules()
        if isinstance(module, CausalConv1d)
    ]
    if len(layer_modules) != 63:
        raise RuntimeError(f"Expected 63 physical Conv1d layers, got {len(layer_modules)}")

    # RTL ROM中的Residual capture/add边界。Scratchpad保存capture层的输入，
    # 对应add层卷积、激活完成后进行同Scale INT16饱和加法。
    residual_capture_layers = {1, 3, 5, 8, 13, 18, 26, 31, 36, 44, 49, 54}
    residual_add_layers = {2, 4, 6, 12, 17, 22, 30, 35, 40, 48, 53, 58}
    # 直接复用导出器的结构判定，避免手工维护ReLU层号。
    from audiolm_pytorch.hardware_export import _conv_followed_by_hardware_relu
    relu_names = _conv_followed_by_hardware_relu(model.encoder)

    first_module = layer_modules[0][1]
    current_q = quantize_activation(
        segment,
        first_module.hardware_input_fake_quant.scale,
        model.hardware_qat_activation_bits,
    )
    captures: list[dict[str, object]] = []
    residual_q = None
    residual_input_scale = None
    current_scale = float(first_module.hardware_input_fake_quant.scale.item())
    module_by_name = dict(model.encoder.named_modules())
    for index, (name, module) in enumerate(layer_modules):
        expected_scale = float(module.hardware_input_fake_quant.scale.item())
        if abs(current_scale-expected_scale) > 1e-6 * max(current_scale, expected_scale):
            raise RuntimeError(f"Layer {index}: missing edge requant {current_scale} -> {expected_scale}")
        input_q = current_q.contiguous()
        if index in residual_capture_layers:
            residual_q = input_q.clone()
            residual_input_scale = float(module.hardware_input_fake_quant.scale.item())

        output_q = integer_conv1d_layer(
            input_q,
            module,
            int(model.hardware_qat_activation_bits),
            int(model.hardware_qat_accumulator_bits),
        )
        current_scale = float(module.hardware_output_fake_quant.scale.item())
        if name in relu_names:
            output_q = output_q.clamp_min(0)
        if index in residual_add_layers:
            if residual_q is None or residual_q.shape != output_q.shape:
                raise RuntimeError(
                    f"Layer {index} residual shape mismatch: "
                    f"saved={None if residual_q is None else tuple(residual_q.shape)}, "
                    f"main={tuple(output_q.shape)}"
                )
            owner = module_by_name[name.split(".fn.")[0]]
            add_scale = float(owner.hardware_residual_add.fake_quant.scale.item())
            identity = align_to_scale(residual_q, residual_input_scale/add_scale).clamp(-32768,32767)
            branch = align_to_scale(output_q, float(module.hardware_output_fake_quant.scale.item())/add_scale).clamp(-32768,32767)
            branch = align_to_scale(branch, float(owner.residual_scale))
            output_q = (identity + branch).clamp(-32768,32767).to(torch.int16)
            current_scale = add_scale
            residual_q = None

        captures.append({
            "name": name,
            "input": input_q,
            "output": output_q,
            "input_scale": float(module.hardware_input_fake_quant.scale.item()),
            "output_scale": current_scale,
        })
        current_q = output_q

    encoder_output = current_q

    manifest_layers = []
    for index, capture_data in enumerate(captures):
        prefix = f"layer_{index:02d}"
        input_q = capture_data.pop("input")
        output_q = capture_data.pop("output")
        input_bct = output_dir / f"{prefix}_input_bct_int16.bin"
        output_bct = output_dir / f"{prefix}_output_bct_int16.bin"
        output_tm = output_dir / f"{prefix}_output_time_major_int16.bin"
        output_hex = output_dir / f"{prefix}_output_time_major_int16.hex"
        write_int16(input_bct, input_q)
        write_int16(output_bct, output_q)
        time_major = output_q.permute(0, 2, 1).contiguous()
        write_int16(output_tm, time_major)
        write_hex16(output_hex, time_major)
        manifest_layers.append(
            {
                "index": index,
                "name": capture_data["name"],
                "input_shape_bct": list(input_q.shape),
                "output_shape_bct": list(output_q.shape),
                "input_scale": capture_data["input_scale"],
                "output_scale": capture_data["output_scale"],
                "input_bct_file": input_bct.name,
                "output_bct_file": output_bct.name,
                "output_time_major_file": output_tm.name,
                "output_time_major_hex": output_hex.name,
                "output_time_major_sha256": sha256(output_tm),
            }
        )

    manifest = {
        "format_version": 1,
        "description": "Bit-exact W8A16/ACC40 integer goldens for RTL layer comparison",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256(audio_path),
        "sample_rate": int(model.target_sample_hz),
        "start_sample": args.start_sample,
        "num_samples": args.num_samples,
        "pcm_format": "signed_int16_little_endian",
        "tensor_layout": "BCT; time-major files are BTC with B=1",
        "weight_bits": 8,
        "activation_bits": 16,
        "accumulator_bits": 40,
        "rounding": "half_away_from_zero",
        "generation_mode": "recursive_integer_rtl_contract",
        "contract_version": HARDWARE_QUANTIZATION_CONTRACT_VERSION,
        "contract_version_recovered": recovered,
        "encoder_output_shape_bct": list(encoder_output.shape),
        "layer_count": len(manifest_layers),
        "layers": manifest_layers,
    }
    manifest_path = output_dir / "golden_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(manifest_layers)} layer goldens: {manifest_path}")


if __name__ == "__main__":
    main()
