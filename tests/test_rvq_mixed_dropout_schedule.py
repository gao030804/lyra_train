from types import SimpleNamespace

from audiolm_pytorch.soundstream import SoundStream


def make_scheduler_model(*, enabled=True):
    residual_vq = SimpleNamespace(
        num_quantizers=23,
        quantize_dropout=False,
    )
    model = SimpleNamespace(
        num_quantizers=23,
        rq_mixed_quantize_dropout=enabled,
        rq_mixed_quantize_dropout_start_step=50_000,
        rq_mixed_quantize_dropout_warmup_steps=20_000,
        rq_mixed_quantize_dropout_hold_end_step=95_000,
        rq_mixed_quantize_dropout_cooldown_steps=10_000,
        rq_mixed_quantize_dropout_max_probability=0.05,
        rq_mixed_quantize_dropout_depths=(8, 12, 16, 20),
        rq_mixed_quantize_dropout_depth_weights=(0.10, 0.15, 0.25, 0.50),
        rq_mixed_quantize_dropout_seed=42,
        rq_mixed_quantize_dropout_probability=0.,
        rq_mixed_quantize_dropout_active=False,
        rq_mixed_quantize_dropout_depth=23,
        rq_mixed_quantize_dropout_hf_scale=1.,
        rq_mixed_quantize_dropout_safety_disabled=False,
        rq=SimpleNamespace(rvqs=(residual_vq,)),
    )
    return model, residual_vq


def update(model, step):
    return SoundStream.update_rq_mixed_quantize_dropout(model, step)


def test_mixed_dropout_probability_schedule_and_final_polish():
    model, residual_vq = make_scheduler_model()

    probability, active, depth, hf_scale = update(model, 49_999)
    assert probability == 0.
    assert active is False
    assert depth == 23
    assert hf_scale == 1.
    assert residual_vq.quantize_dropout is False

    probability, *_ = update(model, 60_000)
    assert probability == 0.025

    probability, *_ = update(model, 70_000)
    assert probability == 0.05

    probability, *_ = update(model, 95_000)
    assert probability == 0.05

    probability, *_ = update(model, 100_000)
    assert probability == 0.025

    probability, active, depth, hf_scale = update(model, 105_000)
    assert probability == 0.
    assert active is False
    assert depth == 23
    assert hf_scale == 1.


def test_mixed_dropout_decision_is_step_deterministic():
    model, _ = make_scheduler_model()

    first = update(model, 70_123)
    second = update(model, 70_123)

    assert first == second


def test_mixed_dropout_is_deep_biased_and_keeps_library_dropout_disabled():
    model, residual_vq = make_scheduler_model()
    selected_depths = []

    for step in range(70_000, 95_000):
        probability, active, depth, hf_scale = update(model, step)
        assert probability == 0.05
        assert residual_vq.quantize_dropout is False
        if active:
            selected_depths.append(depth)
            expected_scale = 0.25 if depth <= 12 else 0.50 if depth <= 16 else 1.
            assert hf_scale == expected_scale

    observed_probability = len(selected_depths) / 25_000
    assert 0.04 <= observed_probability <= 0.06

    counts = {
        depth: selected_depths.count(depth)
        for depth in model.rq_mixed_quantize_dropout_depths
    }
    assert counts[20] > counts[16] > counts[12] > counts[8]


def test_safety_disable_forces_full_depth_forever():
    model, residual_vq = make_scheduler_model()
    SoundStream.disable_rq_mixed_quantize_dropout_for_safety(model)

    probability, active, depth, hf_scale = update(model, 80_000)

    assert probability == 0.
    assert active is False
    assert depth == 23
    assert hf_scale == 1.
    assert residual_vq.quantize_dropout is False


def test_disabled_mixed_dropout_never_activates():
    model, residual_vq = make_scheduler_model(enabled=False)

    probability, active, depth, hf_scale = update(model, 80_000)

    assert probability == 0.
    assert active is False
    assert depth == 23
    assert hf_scale == 1.
    assert residual_vq.quantize_dropout is False


def test_fixed_eight_quantizer_profile_does_not_attenuate_hf_losses():
    model, residual_vq = make_scheduler_model(enabled=False)
    model.num_quantizers = 8
    model.rq_mixed_quantize_dropout_depth = 8
    residual_vq.num_quantizers = 8

    probability, active, depth, hf_scale = update(model, 80_000)

    assert probability == 0.
    assert active is False
    assert depth == 8
    assert hf_scale == 1.
    assert residual_vq.quantize_dropout is False
