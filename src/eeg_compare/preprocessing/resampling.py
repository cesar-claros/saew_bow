"""Exact-ratio resampling shared by corpus readers."""

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def resample_exact(data: np.ndarray, sfreq: float, target_sfreq: float) -> np.ndarray:
    """Polyphase resampling along the last axis at the exact rational ratio ``target_sfreq / sfreq``.

    MNE's default FFT method shifted 250 Hz data by about 2 ms (half an input sample), and MNE's
    polyphase wrapper adjusts the ratio so the output length is an integer, which slightly stretched
    a crop whose length is not a multiple of 125 samples (1 percent error on a test tone; 2026-10-26).
    ``resample_poly`` keeps the ratio (128/125 for 250 to 256 Hz) for any length and only has a
    local edge effect of a few input samples.
    """
    ratio = Fraction(target_sfreq / sfreq).limit_denominator(10_000)
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)
