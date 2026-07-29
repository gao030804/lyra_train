import ast
from pathlib import Path

import pytest


def load_static_method(method_name):
    source_path = (
        Path(__file__).resolve().parents[1] /
        "audiolm_pytorch" /
        "trainer.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SoundStreamTrainer"
    )
    method = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[method_name]

def test_active_spectral_weight_starts_at_point02_and_reaches_point03():
    schedule = load_static_method("scheduled_loss_weight_from_initial")

    assert schedule(0, 0, 300, 0.02, 0.03) == pytest.approx((0.02, 0.0))
    assert schedule(150, 0, 300, 0.02, 0.03) == pytest.approx((0.025, 0.5))
    assert schedule(300, 0, 300, 0.02, 0.03) == pytest.approx((0.03, 1.0))
    assert schedule(3_000, 0, 300, 0.02, 0.03) == pytest.approx((0.03, 1.0))


def test_midband_checkpoint_score_uses_requested_band_penalties():
    midband_checkpoint_score = load_static_method("midband_checkpoint_score")
    metrics = {
        "score": 9.0,
        "reconstruction_score": 1.5,
        "active_spec_200_1k_logmag_error": 0.4,
        "active_spec_1k_3k_logmag_error": 0.6,
        "active_spec_3k_5k_logmag_error": 0.8,
    }

    assert midband_checkpoint_score(metrics) == pytest.approx(
        1.5 + 0.10 * 0.4 + 0.15 * 0.6 + 0.05 * 0.8
    )


def test_stage25_launch_script_matches_controlled_experiment():
    script = (
        Path(__file__).resolve().parents[1] /
        "run_stage25_rvq_midband_refine.sh"
    ).read_text(encoding="utf-8")

    expected_fragments = (
        "--num-train-steps 3000",
        "--save-model-every 200",
        "--best-eval-every 100",
        "--early-stopping-min-steps 1000",
        "--early-stopping-patience 10",
        "--active-spectral-detail-loss-weight 0.03",
        "--active-spectral-detail-loss-warmup-steps 300",
        "--stage2-quality-retention-patience 4",
        "--stage2-rvq-retention-patience 2",
    )
    for fragment in expected_fragments:
        assert fragment in script


def test_stage25_mode_freezes_decoder_and_enables_midband_checkpoint():
    source = (
        Path(__file__).resolve().parents[1] /
        "train_soundstream.py"
    ).read_text(encoding="utf-8")

    assert "num_train_steps if stage25_rvq_midband_refine else None" in source
    assert "midband_checkpoint=stage25_rvq_midband_refine" in source
    assert "(1.00, 1.50, 1.00, 0.50, 0.25)" in source
    assert "full Decoder frozen through step 500" not in source
    assert "later Decoder modules" not in source
