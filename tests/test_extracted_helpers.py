"""Regression fixtures for the extracted helpers; execute on the HPC."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_compare.data.annotations import MONTAGE, parse_rec
from eeg_compare.models.separable_sae import SeparableSAE
from eeg_compare.models.waveform_sae import WaveformTrainSpec
from eeg_compare.preprocessing.resampling import resample_exact
from eeg_compare.preprocessing.robust import robust_scale, valid_rows


def test_row_validity_and_robust_scaling() -> None:
    """Missing/flat channels stay invalid; offset and positive gain cancel."""
    rows = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, float("nan"), 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    assert valid_rows(rows).tolist() == [False, False, True]
    x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
    expected = x / 1.4826
    torch.testing.assert_close(robust_scale(x), expected)
    torch.testing.assert_close(robust_scale(3.0 * x + 7.0), expected)
    torch.testing.assert_close(robust_scale(rows[:1]), torch.zeros_like(rows[:1]))


def test_resampling_preserves_sample_clock_without_integer_length_stretch() -> None:
    """An awkward crop length must retain the requested 128/125 sample ratio."""
    sfreq, target = 250.0, 256.0
    source_time = np.arange(1003) / sfreq
    source = np.stack([np.sin(2 * np.pi * 10 * source_time), np.cos(2 * np.pi * 10 * source_time)])
    result = resample_exact(source, sfreq, target)
    assert result.shape == (2, 1028)
    target_time = np.arange(result.shape[-1]) / target
    expected = np.stack([np.sin(2 * np.pi * 10 * target_time), np.cos(2 * np.pi * 10 * target_time)])
    np.testing.assert_allclose(result[:, 32:-32], expected[:, 32:-32], atol=0.005, rtol=0)


def test_annotations_preserve_raw_intervals_and_ear_derivations() -> None:
    """The reader must not inherit the old experiment's channel exclusions."""
    assert MONTAGE[8] == "A1-T3"
    assert MONTAGE[13] == "T4-A2"
    assert parse_rec("8,0.25,1.75,1\n\n21,4,5,6.0\n") == [(8, 0.25, 1.75, "spsw"), (21, 4.0, 5.0, "bckg")]


@pytest.mark.parametrize("row", ["22,0,1,1", "0,0,1,7", "0,2,1,1", "0,1,1,1", "0,0,1"])
def test_annotations_reject_invalid_channels_codes_and_intervals(row: str) -> None:
    with pytest.raises(ValueError):
        parse_rec(row)


def test_excluded_patient_trainer_is_rejected_at_construction() -> None:
    """An old configuration must not silently train with different sampling."""
    with pytest.raises(ValueError, match="sampling_unit"):
        SeparableSAE(train_spec=WaveformTrainSpec(sampling_unit="patient", recenter_atoms=False), device="cpu")
