import pytest
import torch
import torch.nn.functional as F

from audiolm_pytorch.soundstream import canonicalize_waveform_pair
from audiolm_pytorch.trainer import SoundStreamTrainer


def test_waveform_pair_adds_missing_channel_dimension_without_broadcasting():
    target = torch.tensor([[0.0, 1.0], [10.0, 11.0]])
    recon = target.unsqueeze(1)

    aligned_target, aligned_recon = canonicalize_waveform_pair(target, recon)

    assert aligned_target.shape == aligned_recon.shape == (2, 1, 2)
    assert F.l1_loss(aligned_recon, aligned_target).item() == 0.0


def test_waveform_pair_rejects_mismatched_audio_shapes():
    target = torch.zeros(2, 8)
    recon = torch.zeros(2, 1, 7)

    with pytest.raises(ValueError, match="shapes must match"):
        canonicalize_waveform_pair(target, recon)


def test_waveform_pair_accepts_unbatched_audio():
    target = torch.zeros(8)
    recon = torch.zeros(1, 1, 8)

    aligned_target, aligned_recon = canonicalize_waveform_pair(target, recon)

    assert aligned_target.shape == aligned_recon.shape == (1, 1, 8)


def test_lag_alignment_reports_inverted_waveform_instead_of_hiding_polarity():
    trainer = SoundStreamTrainer.__new__(SoundStreamTrainer)
    target = torch.sin(torch.linspace(0, 20, 2048)).unsqueeze(0)
    recon = -target

    correlation, _ = trainer.lag_aligned_reconstruction_metrics(
        target,
        recon,
        max_lag_samples=64,
    )

    assert correlation.item() < -0.99
    assert trainer._last_alignment_metrics["alignment_negative_fraction"] == 1.0
