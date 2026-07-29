from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage2_baseline_broadcast_includes_upper_highband_metrics():
    source = (ROOT / "train_soundstream.py").read_text(encoding="utf-8")

    start = source.index("        baseline_keys = (")
    end = source.index("        baseline_values = torch.zeros(", start)
    baseline_block = source[start:end]

    assert '"voiced_7k_7p8k_ratio_db"' in baseline_block
    assert '"quiet_7k_7p8k_excess_db"' in baseline_block
    assert '"codebook_q00_active_ratio"' in baseline_block
    assert '"codebook_q00_perplexity"' in baseline_block


def test_trainer_persists_and_gates_on_upper_highband_baseline():
    source = (
        ROOT / "audiolm_pytorch" / "trainer.py"
    ).read_text(encoding="utf-8")

    assert "'voiced_7k_7p8k_ratio_db'," in source
    assert "'quiet_7k_7p8k_excess_db'," in source
    assert "'codebook_q00_active_ratio'," in source
    assert "'codebook_q00_perplexity'," in source
    assert "reasons.append('missing_upper_hf_baseline')" in source
    assert "reasons.append('q00_active_drop')" in source
    assert "reasons.append('q00_perplexity_drop')" in source
