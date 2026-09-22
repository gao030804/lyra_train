from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# The RTL saturates signed activations and post-processing outputs to the full
# two's-complement range.  The scale denominator remains +127 so zero stays
# exact and the positive endpoint is representable; -128 is an extra hardware
# saturation code, not a different zero-point.
INT8_QMIN = -128
INT8_QMAX = 127
INT32_QMIN = -(2 ** 31)
INT32_QMAX = (2 ** 31) - 1
HARDWARE_QUANTIZATION_CONTRACT_VERSION = 3


def signed_integer_range(bits: int) -> tuple[int, int]:
    bits = int(bits)
    if bits < 2 or bits > 62:
        raise ValueError("signed integer width must be in [2, 62]")
    return -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1


def round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    """Match the Conv1d NPU requantizer's tie-breaking rule."""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def ste_replace(reference: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Use the quantized forward value and the reference backward gradient."""
    return reference + (quantized - reference).detach()


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Hardware rounding in forward, straight-through gradient in backward."""
    return ste_replace(x, round_half_away_from_zero(x))


def clamp_ste(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Saturate like hardware while retaining an STE gradient."""
    return ste_replace(x, x.clamp(qmin, qmax))


class SymmetricActivationFakeQuant(nn.Module):
    """Observed, per-tensor, signed symmetric activation fake quantization.

    ``percentile`` keeps the selected hardware format unchanged while
    preventing rare audio transients from consuming most of its dynamic range.
    """

    def __init__(
        self,
        ema_decay: float = 0.99,
        eps: float = 1e-8,
        observer: str = "max",
        percentile: float = 99.99,
        bits: int = 8,
    ):
        super().__init__()
        if not 0. <= ema_decay < 1.:
            raise ValueError("ema_decay must be in [0, 1)")
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        if observer not in {"max", "percentile"}:
            raise ValueError("observer must be 'max' or 'percentile'")
        if not 0. < percentile <= 100.:
            raise ValueError("percentile must be in (0, 100]")
        self.observer = observer
        self.percentile = float(percentile)
        self.bits = int(bits)
        self.qmin, self.qmax = signed_integer_range(self.bits)
        self.register_buffer("amax", torch.tensor(0.))
        # Keep scale observer-owned and fixed after calibration.  It retains
        # the historical state-dict key, but is deliberately not optimized;
        # zero_point remains fixed at 0 to match the MAC datapath.
        self.register_buffer("scale", torch.tensor(1.))
        self.register_buffer("num_observations", torch.tensor(0, dtype=torch.long))
        # Runtime QAT state is persistent.  A model-only best checkpoint must
        # not silently come back in observer/FP32 mode when it is evaluated or
        # exported in a fresh process.
        self.register_buffer("enabled_state", torch.tensor(False))
        self.register_buffer("observer_enabled_state", torch.tensor(False))
        self.register_buffer("blend_alpha_state", torch.tensor(1.))

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_state.item())

    @property
    def observer_enabled(self) -> bool:
        return bool(self.observer_enabled_state.item())

    @property
    def blend_alpha(self) -> float:
        return float(self.blend_alpha_state.item())

    def set_state(
        self,
        *,
        enabled: bool,
        observer_enabled: bool,
        blend_alpha: float = 1.,
    ) -> None:
        if not 0. <= blend_alpha <= 1.:
            raise ValueError("blend_alpha must be in [0, 1]")
        self.enabled_state.fill_(bool(enabled))
        self.observer_enabled_state.fill_(bool(observer_enabled))
        self.blend_alpha_state.fill_(float(blend_alpha))

    @torch.no_grad()
    def reset_observer(self, *, percentile: float | None = None) -> None:
        """Clear calibration history before a local activation-group pass."""
        if percentile is not None:
            if not 0. < float(percentile) <= 100.:
                raise ValueError("percentile must be in (0, 100]")
            self.percentile = float(percentile)
        self.amax.zero_()
        self.scale.fill_(1.)
        self.num_observations.zero_()

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        absolute = x.detach().abs().float().flatten()
        if absolute.numel() == 0:
            return
        # Bound calibration overhead on early high-resolution feature maps.
        # Deterministic striding keeps runs reproducible and does not alter the
        # tensor that is quantized in the forward path.
        if absolute.numel() > 65_536:
            stride = max(1, absolute.numel() // 65_536)
            absolute = absolute[::stride][:65_536]
        current_amax = (
            absolute.amax()
            if self.observer == "max" or self.percentile >= 100.
            else torch.quantile(absolute, self.percentile / 100.)
        )
        if not torch.isfinite(current_amax):
            return
        # DDP broadcasts buffers at forward boundaries, but that does not make
        # the current batch statistic global.  All ranks must update the EMA
        # from the same conservative statistic or the saved rank-0 scale is not
        # representative of the distributed training run.  MAX is deliberate:
        # it avoids an unsafe scale even though each rank estimates its local
        # percentile independently.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                current_amax,
                op=torch.distributed.ReduceOp.MAX,
            )
        if self.num_observations.item() == 0:
            updated_amax = current_amax
        else:
            updated_amax = self.ema_decay * self.amax + (1. - self.ema_decay) * current_amax
        self.amax.copy_(updated_amax)
        self.scale.copy_((updated_amax / self.qmax).clamp_min(self.eps))
        self.num_observations.add_(1)

    def effective_scale(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.scale.to(device=x.device, dtype=x.dtype)
            .abs()
            .clamp_min(self.eps)
            .detach()
        )

    def quantize(self, x: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        scale = self.effective_scale(x) if scale is None else scale
        return round_ste(x / scale).clamp(self.qmin, self.qmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.observer_enabled:
            self.observe(x)
        if not self.enabled:
            return x
        scale = self.effective_scale(x)
        quantized = self.quantize(x, scale=scale) * scale
        return x + self.blend_alpha * (quantized - x)


class PerOutputChannelWeightFakeQuant(nn.Module):
    """Signed symmetric INT8 weight fake quantization along Conv1d Cout."""

    def quantize(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reduce_dims = tuple(range(1, weight.ndim))
        scale = (
            weight.detach().abs().amax(dim=reduce_dims, keepdim=True) / INT8_QMAX
        ).clamp_min(torch.finfo(weight.dtype).eps)
        qweight = round_ste(weight / scale).clamp(INT8_QMIN, INT8_QMAX)
        return qweight, scale

    def forward(
        self,
        weight: torch.Tensor,
        blend_alpha: float = 1.,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qweight, scale = self.quantize(weight)
        quantized = qweight * scale
        blended = weight + float(blend_alpha) * (quantized - weight)
        return blended, scale


def approximate_requant_multiplier(
    real_multiplier: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate positive ratios as int32 multiplier / 2**shift.

    The returned effective ratio has the hardware value in forward and the
    real-ratio gradient in backward.  Shifts are limited to the RTL's 6 bits.
    """
    safe = real_multiplier.clamp_min(torch.finfo(real_multiplier.dtype).tiny)
    detached = safe.detach().double()
    shift = torch.floor(torch.log2(INT32_QMAX / detached)).clamp(0, 63).to(torch.long)
    multiplier = round_half_away_from_zero(detached * torch.pow(2., shift.double()))
    multiplier = multiplier.clamp(1, INT32_QMAX).to(torch.long)
    hardware_ratio = (
        multiplier.double() / torch.pow(2., shift.double())
    ).to(device=real_multiplier.device, dtype=real_multiplier.dtype)
    return ste_replace(safe, hardware_ratio), multiplier, shift


def requantize_int32_with_multiplier_shift(
    accumulator: torch.Tensor,
    effective_multiplier: torch.Tensor,
    multiplier: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Bit-exact detached INT64 requant forward with differentiable STE path."""
    reference = accumulator * effective_multiplier.view(1, -1, 1)
    with torch.no_grad():
        product = (
            accumulator.detach().round().to(torch.int64) *
            multiplier.view(1, -1, 1).to(device=accumulator.device, dtype=torch.int64)
        )
        channel_shift = shift.view(1, -1, 1).to(
            device=accumulator.device,
            dtype=torch.int64,
        )
        half = torch.where(
            channel_shift > 0,
            torch.bitwise_left_shift(
                torch.ones_like(channel_shift),
                (channel_shift - 1).clamp_min(0),
            ),
            torch.zeros_like(channel_shift),
        )
        rounded_magnitude = torch.bitwise_right_shift(product.abs() + half, channel_shift)
        hardware_output = torch.where(product < 0, -rounded_magnitude, rounded_magnitude)
        hardware_output = hardware_output.to(dtype=accumulator.dtype)
    return ste_replace(reference, hardware_output)


class HardwareReLU(nn.Module):
    """ReLU with shared-scale signed-INT8 input and output."""

    def __init__(
        self, ema_decay: float = 0.99, observer: str = "max",
        percentile: float = 99.99, activation_bits: int = 8
    ):
        super().__init__()
        self.fake_quant = SymmetricActivationFakeQuant(
            ema_decay=ema_decay, observer=observer, percentile=percentile,
            bits=activation_bits,
        )
        self.observer_owned_by_producer = False

    def share_producer_fake_quant(
        self,
        fake_quant: SymmetricActivationFakeQuant,
    ) -> None:
        """Use the Conv requant scale without observing the same edge twice."""
        self.fake_quant = fake_quant
        self.observer_owned_by_producer = True

    def set_hardware_qat_state(
        self, *, enabled: bool, observer_enabled: bool, blend_alpha: float = 1.
    ) -> None:
        self.fake_quant.set_state(
            enabled=enabled,
            observer_enabled=observer_enabled,
            blend_alpha=blend_alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        float_output = F.relu(x)
        if (
            self.training and
            self.fake_quant.observer_enabled and
            not self.observer_owned_by_producer
        ):
            # One scale is deliberately shared by the ReLU input and output.
            self.fake_quant.observe(torch.cat((x.flatten(), float_output.flatten())))
        if not self.fake_quant.enabled:
            return float_output

        scale = self.fake_quant.effective_scale(x)
        qinput = self.fake_quant.quantize(x, scale=scale)
        qoutput = qinput.clamp_min(0.)
        quantized = qoutput * scale
        return float_output + self.fake_quant.blend_alpha * (
            quantized - float_output
        )


class HardwareResidualAdd(nn.Module):
    """Shared-scale signed-integer residual merge with hardware saturation."""

    def __init__(
        self, ema_decay: float = 0.99, observer: str = "max",
        percentile: float = 99.99, activation_bits: int = 8
    ):
        super().__init__()
        self.fake_quant = SymmetricActivationFakeQuant(
            ema_decay=ema_decay, observer=observer, percentile=percentile,
            bits=activation_bits,
        )

    def set_hardware_qat_state(
        self, *, enabled: bool, observer_enabled: bool, blend_alpha: float = 1.
    ) -> None:
        self.fake_quant.set_state(
            enabled=enabled,
            observer_enabled=observer_enabled,
            blend_alpha=blend_alpha,
        )

    def forward(
        self,
        identity: torch.Tensor,
        residual: torch.Tensor,
        residual_scale: float = 1.,
    ) -> torch.Tensor:
        """Align both branches, apply residual scale as integer requant, add.

        The previous implementation multiplied ``residual`` in floating point
        before entering this module.  That operation had no deployable RTL
        parameter.  Here the multiplier/shift approximation is in the forward
        path, so the same pair can be exported for the residual-align unit.
        """
        if residual_scale < 0.:
            raise ValueError("hardware residual scale must be non-negative")
        scaled_residual = residual * float(residual_scale)
        float_output = identity + scaled_residual
        if self.training and self.fake_quant.observer_enabled:
            self.fake_quant.observe(torch.cat((
                identity.flatten(),
                scaled_residual.flatten(),
                float_output.flatten(),
            )))
        if not self.fake_quant.enabled:
            return float_output

        scale = self.fake_quant.effective_scale(float_output)
        qidentity = self.fake_quant.quantize(identity, scale=scale)
        qresidual = self.fake_quant.quantize(residual, scale=scale)
        channel_multiplier = torch.full(
            (residual.shape[1],),
            float(residual_scale),
            device=residual.device,
            dtype=residual.dtype,
        )
        effective, multiplier, shift = approximate_requant_multiplier(
            channel_multiplier
        )
        aligned_residual = requantize_int32_with_multiplier_shift(
            qresidual,
            effective,
            multiplier,
            shift,
        )
        qoutput = clamp_ste(
            qidentity + aligned_residual,
            self.fake_quant.qmin,
            self.fake_quant.qmax,
        )
        quantized = qoutput * scale
        return float_output + self.fake_quant.blend_alpha * (
            quantized - float_output
        )

    def integer_parameters(
        self,
        channels: int,
        residual_scale: float,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fixed per-channel residual multiplier and shift."""
        ratio = torch.full(
            (int(channels),),
            float(residual_scale),
            dtype=torch.float32,
            device=device,
        )
        _, multiplier, shift = approximate_requant_multiplier(ratio)
        return multiplier, shift


def quantize_bias_integer(
    bias: torch.Tensor | None,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    accumulator_bits: int = 32,
) -> torch.Tensor | None:
    """Return bias in the selected signed accumulator integer domain."""
    if bias is None:
        return None
    bias_scale = input_scale.to(device=bias.device, dtype=bias.dtype) * weight_scale.flatten()
    bias_scale = bias_scale.clamp_min(torch.finfo(bias.dtype).eps)
    qmin, qmax = signed_integer_range(accumulator_bits)
    return clamp_ste(round_ste(bias / bias_scale), qmin, qmax)


def quantize_bias_int32(
    bias: torch.Tensor | None,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor | None:
    """Backward-compatible fixed INT32 bias helper."""
    return quantize_bias_integer(bias, input_scale, weight_scale, 32)


def quantize_bias_dequantized(
    bias: torch.Tensor | None,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    accumulator_bits: int = 32,
) -> torch.Tensor | None:
    """Simulate per-output-channel INT32 bias storage for QAT."""
    if bias is None:
        return None
    bias_scale = input_scale.to(device=bias.device, dtype=bias.dtype) * weight_scale.flatten()
    bias_scale = bias_scale.clamp_min(torch.finfo(bias.dtype).eps)
    qbias = quantize_bias_integer(
        bias, input_scale, weight_scale, accumulator_bits
    )
    return qbias * bias_scale
