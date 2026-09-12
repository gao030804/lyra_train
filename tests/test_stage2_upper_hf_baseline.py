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
    assert '"spectral_envelope_fine"' in baseline_block
    assert '"spectral_envelope_coarse"' in baseline_block
    assert '"formant_f1_mae_hz"' in baseline_block
    assert '"formant_f2_mae_hz"' in baseline_block
    assert '"formant_f3_mae_hz"' in baseline_block
    assert '"stft_scale_512"' in baseline_block
    assert '"stft_scale_1024"' in baseline_block
    assert '"stft_scale_2048"' in baseline_block


def test_trainer_keeps_upper_highband_diagnostic_only():
    source = (
        ROOT / "audiolm_pytorch" / "trainer.py"
    ).read_text(encoding="utf-8")

    assert "'voiced_7k_7p8k_ratio_db'," in source
    assert "'quiet_7k_7p8k_excess_db'," in source
    assert "'codebook_q00_active_ratio'," in source
    assert "'codebook_q00_perplexity'," in source
    assert "reasons.append('missing_upper_hf_baseline')" not in source
    assert "High-frequency metrics remain diagnostics/loss targets" in source
    assert "reasons.append('q00_active_drop')" not in source
    assert "reasons.append('q00_perplexity_drop')" not in source
    assert "rvq_deployable_distribution_eligible" in source
