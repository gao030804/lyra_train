import torch

from audiolm_pytorch.hardware_quantization import (
    HardwareReLU,
    HardwareResidualAdd,
    SymmetricActivationFakeQuant,
    approximate_requant_multiplier,
    round_half_away_from_zero,
)
from audiolm_pytorch.hardware_export import (
    integer_conv1d_reference,
    pack_conv1d_weights_4x8,
)
from audiolm_pytorch.soundstream import (
    CausalConv1d,
    SoundStream,
    hardware_input_fake_quant,
)


def test_half_away_from_zero_matches_hardware_ties():
    values = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    expected = torch.tensor([-3., -2., -1., 1., 2., 3.])
    torch.testing.assert_close(round_half_away_from_zero(values), expected)


def test_symmetric_activation_fake_quant_uses_signed_int8_and_ste():
    fake_quant = SymmetricActivationFakeQuant(ema_decay=0.).train()
    fake_quant.set_state(enabled=True, observer_enabled=True)
    x = torch.tensor([-2., -0.25, 0.5, 1.], requires_grad=True)

    output = fake_quant(x)
    output.sum().backward()

    torch.testing.assert_close(
        fake_quant.scale,
        torch.tensor(2. / 127.),
    )
    assert output.min() >= -2.
    assert output.max() <= 2.
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_percentile_observer_rejects_rare_activation_outlier():
    values = torch.cat((torch.ones(10_000), torch.tensor([100.])))
    percentile = SymmetricActivationFakeQuant(
        ema_decay=0., observer="percentile", percentile=99.9
    )
    maximum = SymmetricActivationFakeQuant(
        ema_decay=0., observer="max", percentile=99.9
    )
    percentile.observe(values)
    maximum.observe(values)
    assert percentile.amax.item() < 2.
    assert maximum.amax.item() == 100.
    assert percentile.scale.item() < maximum.scale.item() / 50.


def test_activation_uses_full_rtl_negative_saturation_code():
    fake_quant = SymmetricActivationFakeQuant().eval()
    fake_quant.set_state(enabled=True, observer_enabled=False)
    fake_quant.scale.fill_(1.)
    output = fake_quant(torch.tensor([-200., -128., 127., 200.]))
    torch.testing.assert_close(output, torch.tensor([-128., -128., 127., 127.]))


def test_activation_scale_is_observer_owned_and_fixed_after_freeze():
    fake_quant = SymmetricActivationFakeQuant(ema_decay=0.).train()
    x = torch.tensor([-1.3, -0.2, 0.7, 1.1], requires_grad=True)

    fake_quant.set_state(enabled=True, observer_enabled=True)
    fake_quant(x).square().sum().backward()
    assert fake_quant.scale.grad is None
    calibrated_scale = fake_quant.scale.clone()

    x.grad = None
    fake_quant.set_state(enabled=True, observer_enabled=False)
    fake_quant(x).square().sum().backward()
    assert fake_quant.scale.grad is None
    torch.testing.assert_close(fake_quant.scale, calibrated_scale)
    assert "scale" not in dict(fake_quant.named_parameters())


def test_requant_multiplier_uses_int32_multiplier_and_six_bit_shift():
    ratio = torch.tensor([0.003, 0.25, 1.5], requires_grad=True)
    effective, multiplier, shift = approximate_requant_multiplier(ratio)

    assert torch.all(multiplier >= 1)
    assert torch.all(multiplier <= (2 ** 31) - 1)
    assert torch.all(shift >= 0)
    assert torch.all(shift <= 63)
    torch.testing.assert_close(effective, ratio, rtol=1e-6, atol=1e-9)
    effective.sum().backward()
    torch.testing.assert_close(ratio.grad, torch.ones_like(ratio))


def test_hardware_relu_shares_one_scale_and_preserves_gradients():
    relu = HardwareReLU(ema_decay=0.).train()
    relu.set_hardware_qat_state(enabled=True, observer_enabled=True)
    x = torch.tensor([-2., -0.5, 0., 1.], requires_grad=True)

    output = relu(x)
    output.sum().backward()

    assert relu.fake_quant.scale.item() > 0.
    torch.testing.assert_close(output[:3], torch.zeros(3))
    assert output[3] > 0.
    assert torch.isfinite(x.grad).all()


def test_hardware_relu_does_not_double_update_shared_conv_observer():
    producer = SymmetricActivationFakeQuant(ema_decay=0.).train()
    producer.set_state(enabled=True, observer_enabled=True)
    relu = HardwareReLU(ema_decay=0.).train()
    relu.share_producer_fake_quant(producer)
    producer.observe(torch.tensor([-2., 1.]))
    observations = int(producer.num_observations.item())
    relu(torch.tensor([-2., 1.]))
    assert int(producer.num_observations.item()) == observations

    consumer = CausalConv1d(1, 1, 1, pad_mode="constant")
    consumer.configure_hardware_qat(ema_decay=0.)
    consumer.share_hardware_input_fake_quant(producer)
    consumer.set_hardware_qat_state(enabled=True, observer_enabled=True)
    consumer(torch.tensor([[[-2., 1.]]]))
    assert int(producer.num_observations.item()) == observations


def test_residual_scale_is_applied_through_integer_requant():
    residual = HardwareResidualAdd(ema_decay=0.).train()
    residual.set_hardware_qat_state(enabled=True, observer_enabled=False)
    residual.fake_quant.scale.fill_(1.)
    identity = torch.tensor([[[10., -10.]]])
    branch = torch.tensor([[[8., 8.]]])
    output = residual(identity, branch, residual_scale=0.5)
    torch.testing.assert_close(output, torch.tensor([[[14., -6.]]]))


def test_hardware_relu_preserves_large_positive_branch_without_overflow():
    relu = HardwareReLU().train()
    relu.set_hardware_qat_state(enabled=True, observer_enabled=False)
    with torch.no_grad():
        relu.fake_quant.scale.fill_(1.)
    x = torch.tensor([-2., 100.], requires_grad=True)

    output = relu(x)
    output.sum().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert relu.fake_quant.scale.grad is None


def test_hardware_qat_supports_current_lowrank_dscnn_encoder_and_keeps_shape():
    model = SoundStream(
        channels=16,
        channel_mults=(2, 4, 8, 16),
        codebook_dim=64,
        codebook_size=256,
        rq_num_quantizers=8,
        use_local_attn=False,
        encoder_depthwise_separable_blocks=(1, 2, 3),
        encoder_depthwise_separable_revision=3,
        encoder_low_rank_pointwise_ranks=((0, 0), (8, 16), (16, 32), (32, 64)),
        hardware_compatible_encoder=True,
        hardware_encoder_qat=True,
        hardware_qat_observer_start_step=0,
        hardware_qat_start_step=1,
        hardware_qat_activation_start_step=2,
        hardware_qat_warm_in_steps=2,
        hardware_qat_block_interval_steps=0,
        hardware_qat_validation_gated=False,
        hardware_qat_observer_freeze_step=1,
        pad_mode="constant",
    ).eval()

    report = model.hardware_encoder_weight_report()
    assert report
    assert all(layer["fits_weight_bank"] for layer in report)
    assert any(layer["groups"] > 1 for layer in report)
    assert max(layer["packed_weight_bytes"] for layer in report) <= 128 * 1024

    with torch.no_grad():
        encoded = model.encoder(torch.randn(1, 1, 320))
    assert encoded.shape == (1, 64, 1)

    assert model.update_hardware_qat(0) == (False, True)
    assert model.hardware_qat_blend_alpha == 0.
    assert model.update_hardware_qat(1) == (True, False)
    assert model.hardware_qat_blend_alpha == 0.
    assert model.encoder[0].hardware_weight_qat_enabled
    assert not model.encoder[0].hardware_activation_qat_enabled
    assert model.update_hardware_qat(2) == (True, False)
    assert model.hardware_qat_blend_alpha == 0.5
    assert model.update_hardware_qat(3) == (True, False)
    assert model.hardware_qat_blend_alpha == 1.
    assert model.hardware_qat_is_full()
    assert (
        model.encoder[0].hardware_output_fake_quant is
        hardware_input_fake_quant(model.encoder[1])
    )


def test_hardware_qat_uses_six_groups_and_requires_two_validation_passes():
    model = SoundStream(
        channels=16,
        channel_mults=(2, 4, 8, 16),
        codebook_dim=64,
        codebook_size=256,
        rq_num_quantizers=8,
        use_local_attn=False,
        hardware_compatible_encoder=True,
        hardware_encoder_qat=True,
        hardware_qat_start_step=1,
        hardware_qat_activation_start_step=2,
        hardware_qat_observer_freeze_step=1,
        hardware_qat_warm_in_steps=1,
        hardware_qat_validation_gated=True,
        hardware_qat_gate_required_passes=2,
        pad_mode="constant",
    ).eval()
    model.update_hardware_qat(2)
    assert model.hardware_qat_group_alphas == (1., 0., 0., 0., 0., 0.)
    passing = {
        "aligned_si_sdr": 2.0,
        "qat_latent32_nmse": 0.01,
        "qat_quantized_output_nmse": 0.01,
        "qat_int32_overflow_max": 0.0,
        "qat_index_flip_q00": 0.05,
        "qat_index_flip_q01": 0.08,
    }
    baseline = {"aligned_si_sdr": 2.0}
    assert model.update_hardware_qat_validation_gate(
        passing, baseline, 2
    ) == (False, "awaiting_consecutive_pass")
    assert model.update_hardware_qat_validation_gate(
        passing, baseline, 3
    ) == (True, "accepted_rescan_required")
    model.hardware_qat_active_group.fill_(1)
    model.hardware_qat_group_start_step.fill_(4)
    model.update_hardware_qat(4)
    assert model.hardware_qat_group_alphas == (1., 1., 0., 0., 0., 0.)


def test_hardware_qat_validation_gated_warm_in_uses_physical_group_order():
    model = SoundStream(
        channels=16,
        channel_mults=(2, 4, 8, 16),
        codebook_dim=64,
        codebook_size=256,
        rq_num_quantizers=8,
        use_local_attn=False,
        hardware_compatible_encoder=True,
        hardware_encoder_qat=True,
        hardware_qat_start_step=1,
        hardware_qat_activation_start_step=2,
        hardware_qat_observer_freeze_step=1,
        hardware_qat_warm_in_steps=2,
        hardware_qat_validation_gated=True,
        pad_mode="constant",
    ).eval()
    model.hardware_qat_group_order.copy_(torch.tensor([2, 4, 0, 5, 3, 1]))
    model.hardware_qat_active_group.fill_(2)

    model.update_hardware_qat(2)
    assert model.hardware_qat_group_alphas == (0., 0., .5, 0., 0., 0.)
    model.update_hardware_qat(3)
    assert model.hardware_qat_group_alphas == (0., 0., 1., 0., 0., 0.)

    model.hardware_qat_accepted_groups[2] = True
    model.hardware_qat_active_group.fill_(4)
    model.hardware_qat_group_start_step.fill_(4)
    model.update_hardware_qat(4)
    assert model.hardware_qat_group_alphas == (0., 0., 1., 0., .5, 0.)


def test_hardware_qat_gate_uses_vq_output_and_defers_after_patience():
    model = SoundStream(
        channels=16,
        channel_mults=(2, 4, 8, 16),
        codebook_dim=64,
        codebook_size=256,
        rq_num_quantizers=8,
        use_local_attn=False,
        hardware_compatible_encoder=True,
        hardware_encoder_qat=True,
        hardware_qat_start_step=1,
        hardware_qat_activation_start_step=2,
        hardware_qat_observer_freeze_step=1,
        hardware_qat_warm_in_steps=1,
        hardware_qat_validation_gated=True,
        hardware_qat_gate_required_passes=1,
        hardware_qat_group_fail_patience=2,
        pad_mode="constant",
    ).eval()
    baseline = {"aligned_si_sdr": 2.0}
    model.update_hardware_qat(2)

    # Large FP32 index flips are diagnostics only when quantized output is
    # faithful to the teacher and integer arithmetic remains safe.
    passing = {
        "aligned_si_sdr": 2.0,
        "qat_latent32_nmse": 0.01,
        "qat_quantized_output_nmse": 0.02,
        "qat_int32_overflow_max": 0.0,
        "qat_index_flip_q00": 0.9,
        "qat_index_flip_q01": 0.9,
    }
    assert model.update_hardware_qat_validation_gate(
        passing, baseline, 2
    ) == (True, "accepted_rescan_required")

    model.hardware_qat_active_group.fill_(1)
    model.hardware_qat_group_start_step.fill_(3)
    model.update_hardware_qat(3)
    failing = dict(passing, qat_quantized_output_nmse=0.2)
    assert model.update_hardware_qat_validation_gate(
        failing, baseline, 3
    ) == (False, "metrics_failed")
    assert model.update_hardware_qat_validation_gate(
        failing, baseline, 4
    ) == (False, "rollback_deferred")
    assert model.hardware_qat_active_group.item() == -1
    assert model.hardware_qat_deferred_groups[1]
    assert model.hardware_qat_accepted_groups[0]


def test_hardware_qat_disabled_stage_rebuild_starts_in_float_mode():
    model = SoundStream(
        channels=16,
        channel_mults=(2, 4, 8, 16),
        codebook_dim=64,
        codebook_size=256,
        rq_num_quantizers=8,
        use_local_attn=False,
        hardware_compatible_encoder=True,
        hardware_encoder_qat=True,
        hardware_qat_start_step=(2 ** 31) - 1,
        hardware_qat_observer_freeze_step=(2 ** 31) - 1,
        pad_mode="constant",
    ).eval()

    convs = [module for module in model.encoder.modules() if isinstance(module, CausalConv1d)]
    assert convs
    assert all(not module.hardware_input_fake_quant.enabled for module in convs)
    assert all(not module.hardware_output_fake_quant.enabled for module in convs)

    rebuilt = SoundStream(**model.configs).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    encoder_input = torch.randn(1, 1, 320)
    with torch.no_grad():
        encoded = model.encoder(encoder_input)
        rebuilt_encoded = rebuilt.encoder(encoder_input)
    assert torch.count_nonzero(encoded).item() > 0
    assert torch.isfinite(encoded).all()
    torch.testing.assert_close(rebuilt_encoded, encoded)


def test_hardware_qat_runtime_state_survives_model_only_state_dict():
    fake_quant = SymmetricActivationFakeQuant()
    fake_quant.set_state(enabled=True, observer_enabled=False, blend_alpha=1.)
    state = fake_quant.state_dict()
    rebuilt = SymmetricActivationFakeQuant()
    rebuilt.load_state_dict(state, strict=True)
    assert rebuilt.enabled
    assert not rebuilt.observer_enabled
    assert rebuilt.blend_alpha == 1.


def test_hardware_qat_causal_conv_quantizes_bias_and_backpropagates():
    layer = CausalConv1d(4, 8, 3, pad_mode="constant")
    layer.configure_hardware_qat(ema_decay=0.)
    layer.set_hardware_qat_state(enabled=True, observer_enabled=True)
    x = torch.randn(2, 4, 16, requires_grad=True)

    output = layer(x)
    output.square().mean().backward()

    assert output.shape == (2, 8, 16)
    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.conv.weight.grad).all()
    report = layer.hardware_quantization_diagnostics()
    assert report["accumulator_values"] > 0
    assert report["output_values"] > 0
    assert 0. <= report["accumulator_overflow_rate"] <= 1.
    assert 0. <= report["output_saturation_rate"] <= 1.
    assert len(report["requant_multiplier"]) == 8
    assert len(report["requant_shift"]) == 8


def test_observer_only_calibrates_conv_output_without_changing_float_forward():
    layer = CausalConv1d(1, 2, 3, pad_mode="constant")
    layer.configure_hardware_qat(ema_decay=0.)
    layer.train()
    layer.set_hardware_qat_state(
        enabled=False,
        observer_enabled=True,
        blend_alpha=0.,
    )
    x = torch.randn(2, 1, 32)
    expected = layer.conv(torch.nn.functional.pad(x, (2, 0)))
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    assert layer.hardware_input_fake_quant.num_observations.item() == 1
    assert layer.hardware_output_fake_quant.num_observations.item() == 1


def test_weight_packing_matches_rtl_4x8_byte_order():
    qweight = torch.zeros(9, 2, 2, dtype=torch.int8)
    qweight[0, 0, 0] = 11
    qweight[7, 1, 0] = 22
    qweight[8, 0, 1] = 33
    packed = pack_conv1d_weights_4x8(qweight)
    assert packed.shape == (2, 1, 4, 8)
    assert packed[0, 0, 0, 0].item() == 11
    assert packed[0, 0, 1, 7].item() == 22
    assert packed[1, 0, 2, 0].item() == 33


def test_integer_reference_keeps_bias_after_accumulation():
    result = integer_conv1d_reference(
        torch.tensor([[[2, 3]]], dtype=torch.int8),
        torch.tensor([[[4]]], dtype=torch.int8),
        torch.tensor([5], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        result["accumulator_int32"],
        torch.tensor([[[8, 12]]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        result["biased_int32"],
        torch.tensor([[[13, 17]]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        result["requant_int8"],
        torch.tensor([[[13, 17]]], dtype=torch.int8),
    )
