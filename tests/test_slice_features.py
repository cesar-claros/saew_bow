"""Readouts and features of the slice: token counts, the spectral control and the bag of spectra; run on the HPC."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eeg_compare.cells.common import CellAtoms, unit_norm
from eeg_compare.models.spectral_features import SpectralFeatures
from eeg_compare.slice.features import BANDS, band_features, cell_tokens, count_features, frame_starts, window_spectra
from eeg_compare.slice.readouts import READOUT_NAMES, make_readouts, window_support
from eeg_compare.slice.spectra import fit_spectra_bag, spectra_tokens


def _sine(frequency: float, n_times: int = 1250, sfreq: float = 250.0) -> torch.Tensor:
    return torch.sin(2 * math.pi * frequency * torch.arange(n_times) / sfreq)


def _waves(atom_len: int = 25) -> torch.Tensor:
    """Three unit waveforms with one, two and three cycles."""
    time = torch.arange(atom_len) / atom_len
    return unit_norm(torch.stack([torch.sin(2 * math.pi * cycles * time) for cycles in (1, 2, 3)]))


def test_the_readouts_are_those_of_the_protocol() -> None:
    readouts = make_readouts()
    assert tuple(readouts) == READOUT_NAMES
    assert readouts["full"].windows == tuple(range(10)) and readouts["full"].ranges == ((0, 1250),)
    assert readouts["event"].windows == (4, 5) and readouts["event"].ranges == ((438, 812),)
    assert readouts["context"].windows == (0, 1, 2, 7, 8, 9)
    assert readouts["context"].ranges == ((0, 437), (813, 1250))
    assert readouts["displaced"].windows == tuple(range(10)) and readouts["displaced"].displaced
    assert not any(readouts[name].displaced for name in ("full", "event", "context"))
    assert window_support(0, 1250, 125, 125) == (0, 187) and window_support(9, 1250, 125, 125) == (1063, 1250)
    # the tiling level: windows of 250 positions
    wide = make_readouts(pool_len=250)
    assert wide["full"].windows == (0, 1, 2, 3, 4) and wide["event"].windows == (2,)
    assert wide["event"].ranges == ((438, 812),)
    assert wide["context"].windows == (0, 4) and wide["context"].ranges == ((0, 312), (938, 1250))
    with pytest.raises(ValueError, match="whole number"):
        make_readouts(n_times=1200)
    with pytest.raises(ValueError, match="without a window"):
        make_readouts(n_times=250, event=(0, 250))


def test_frames_cover_a_range_with_an_overlap_of_at_least_half() -> None:
    assert frame_starts(1250, 250) == list(range(0, 1001, 125))
    assert frame_starts(374, 250) == [0, 124]
    assert frame_starts(437, 250) == [0, 94, 187]
    assert frame_starts(250, 250) == [0]
    for length in (250, 374, 437, 600, 1250):
        starts = frame_starts(length, 250)
        assert starts[0] == 0 and starts[-1] == length - 250
        assert all(second - first <= 125 for first, second in zip(starts, starts[1:], strict=False))
    with pytest.raises(ValueError, match="no frame"):
        frame_starts(249, 250)


def test_band_features_equal_the_retained_control_on_a_whole_context() -> None:
    rows = torch.randn((5, 3, 1250), generator=torch.Generator().manual_seed(0))
    retained = SpectralFeatures(sfreq=250.0, bands=BANDS, nperseg=250, device="cpu")(rows).double().numpy()
    found = band_features(rows, [(0, 1250)])
    assert found.shape == (5, 3 * 8) and found.dtype == np.float64
    np.testing.assert_allclose(found, retained, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(band_features(rows, [(0, 1250)], chunk=2), found, rtol=1e-6, atol=1e-7)
    # the shares of a channel add up to 1, and a 10 Hz sine lies in the band of 8 to 13 Hz
    assert np.allclose(found.reshape(5, 3, 8)[..., 4:].sum(axis=-1), 1.0, atol=1e-5)
    alpha = band_features(_sine(10.0)[None, None, :], [(0, 1250)])[0]
    assert alpha[4 + 2] > 0.95 and int(np.argmax(alpha[:4])) == 2
    with pytest.raises(ValueError, match="leaves"):
        band_features(rows, [(0, 1300)])
    with pytest.raises(ValueError, match="at least one range"):
        band_features(rows, [])
    with pytest.raises(ValueError, match="no frame"):
        band_features(rows, [(0, 200)])


def test_band_features_of_a_readout_use_its_samples_only() -> None:
    readouts = make_readouts()
    rows = torch.randn((4, 2, 1250), generator=torch.Generator().manual_seed(1))
    changed = rows.clone()
    changed[..., 500:750] += 20.0 * _sine(10.0)[500:750]
    context = readouts["context"].ranges
    np.testing.assert_array_equal(band_features(changed, context), band_features(rows, context))
    event = readouts["event"].ranges
    assert not np.allclose(band_features(changed, event), band_features(rows, event))
    # the two parts of the context readout enter one mean
    left = rows.clone()
    left[..., :437] *= 3.0
    assert not np.allclose(band_features(left, context), band_features(rows, context))
    right = rows.clone()
    right[..., 813:] *= 3.0
    assert not np.allclose(band_features(right, context), band_features(rows, context))


def test_a_single_channel_dictionary_gives_one_token_per_channel_and_window() -> None:
    waves = _waves()
    rows = torch.zeros((3, 4, 500))
    rows[1, 3, 213:238] = 2.0 * waves[2]
    for score in ("cosine", "raw"):
        tokens = cell_tokens(rows, CellAtoms("S1", waveform=waves), score, 50, device="cpu")
        assert tokens.shape == (3, 10, 4) and tokens.dtype == torch.long
        expected = torch.full((3, 10, 4), -1)
        expected[1, 4, 3] = 2
        assert torch.equal(tokens, expected)
    noisy = rows + 0.01 * torch.randn(rows.shape, generator=torch.Generator().manual_seed(2))
    whole = cell_tokens(noisy, CellAtoms("S1", waveform=waves), "cosine", 50, device="cpu")
    assert int(whole[1, 4, 3]) == 2 and int(whole.min()) >= -1 and int(whole.max()) <= 2
    assert torch.equal(cell_tokens(noisy, CellAtoms("S1", waveform=waves), "cosine", 50, device="cpu", chunk=2), whole)
    assert cell_tokens(rows[:0], CellAtoms("S1", waveform=waves), "cosine", 50, device="cpu").shape == (0, 10, 4)


def test_a_full_template_dictionary_gives_one_token_per_window() -> None:
    waves = _waves()
    template = torch.zeros((2, 4, 25))
    template[0, 0] = waves[0]
    template[1, 1] = waves[1]
    template[1, 2] = -waves[1]
    atoms = CellAtoms("SF", template=unit_norm(template))
    rows = torch.zeros((3, 4, 500))
    assert atoms.template is not None
    rows[0, :, 113:138] = 3.0 * atoms.template[1]
    rows[2, :, 413:438] = atoms.template[0]
    tokens = cell_tokens(rows, atoms, "cosine", 50, device="cpu")
    expected = torch.full((3, 10, 1), -1)
    expected[0, 2, 0] = 1
    expected[2, 8, 0] = 0
    assert torch.equal(tokens, expected)
    rank_one = CellAtoms("SR", waveform=waves, pattern=torch.ones((3, 4, 1)))
    with pytest.raises(ValueError, match="S1 and SF"):
        cell_tokens(rows, rank_one, "cosine", 50, device="cpu")


def test_counts_are_shares_of_the_tokens_of_the_chosen_windows() -> None:
    tokens = torch.tensor(
        [
            [[0, 1, -1], [2, 2, 0], [1, 1, 1], [-1, -1, -1]],
            [[2, 2, 2], [2, 2, 2], [0, 0, 0], [0, -1, 1]],
        ]
    )
    found = count_features(tokens, [0, 1], 3)
    np.testing.assert_allclose(found, np.array([[2, 1, 2, 1], [0, 0, 6, 0]]) / 6.0)
    np.testing.assert_allclose(count_features(tokens, [3], 3), np.array([[0, 0, 0, 3], [1, 1, 0, 1]]) / 3.0)
    every = count_features(tokens, [0, 1, 2, 3], 3)
    assert every.shape == (2, 4) and np.allclose(every.sum(axis=1), 1.0)
    # a dictionary with more atoms than the tokens use keeps a column for each
    assert count_features(tokens, [0], 5).shape == (2, 6)
    for windows, n_atoms in (([0, 4], 3), ([0, 0], 3), ([], 3), ([0], 2)):
        with pytest.raises(ValueError):
            count_features(tokens, windows, n_atoms)
    with pytest.raises(ValueError, match="tokens"):
        count_features(tokens[0], [0], 3)


def test_window_spectra_hold_the_frequency_of_a_window_and_no_edge_step() -> None:
    rows = torch.zeros((2, 2, 1250))
    rows[0, 0] = _sine(10.0)
    rows[1, 0] = 5.0
    rows[1, 1] = _sine(20.0)
    spectra = window_spectra(rows, 125, 125, device="cpu")
    assert spectra.shape == (2, 10, 2, 30) and spectra.dtype == torch.float32
    # bins of 1 Hz from 1 to 30 Hz: the 10 Hz sine has its peak in bin 9, in every window, also at the edges
    assert bool((spectra[0, :, 0].argmax(dim=-1) == 9).all())
    assert bool((spectra[1, :, 1].argmax(dim=-1) == 19).all())
    assert bool((spectra[0, :, 0, 7:12].pow(2).sum(dim=-1) > 0.99).all())
    norms = torch.linalg.vector_norm(spectra, dim=-1)
    assert torch.allclose(norms[0, :, 0], torch.ones(10), atol=1e-5)
    # a window without signal, and a constant window, have a zero spectrum
    assert not bool(spectra[0, :, 1].any()) and not bool(spectra[1, :, 0].any())
    # a constant offset does not change a spectrum
    shifted = window_spectra(rows + 3.0, 125, 125, device="cpu")
    assert torch.allclose(shifted[0, :, 0], spectra[0, :, 0], atol=1e-3)
    assert torch.allclose(window_spectra(rows, 125, 125, device="cpu", chunk=1), spectra, atol=1e-6)
    # the tiling level has longer windows and finer bins
    assert window_spectra(rows, 125, 250, device="cpu").shape == (2, 5, 2, 59)


def test_the_bag_of_spectra_gives_the_nearest_center_and_the_null_token() -> None:
    rows = torch.zeros((6, 2, 1250))
    for index in range(6):
        rows[index, 0] = _sine(6.0 if index % 2 else 22.0)
    rows += 0.01 * torch.randn(rows.shape, generator=torch.Generator().manual_seed(3))
    rows[:, 1] = 0.0
    spectra = window_spectra(rows, 125, 125, device="cpu")
    model = fit_spectra_bag(spectra[:4], 2, seed=0)
    tokens = spectra_tokens(model, spectra)
    assert tokens.shape == (6, 10, 2) and tokens.dtype == torch.long
    assert bool((tokens[:, :, 1] == -1).all())
    slow, fast = tokens[1::2, :, 0], tokens[0::2, :, 0]
    assert slow.unique().numel() == 1 and fast.unique().numel() == 1 and int(slow[0, 0]) != int(fast[0, 0])
    assert set(tokens[:, :, 0].unique().tolist()) == {0, 1}
    again = spectra_tokens(fit_spectra_bag(spectra[:4], 2, seed=0), spectra)
    assert torch.equal(again, tokens)
    features = count_features(tokens, list(range(10)), 2)
    assert np.allclose(features[:, 2], 0.5) and np.allclose(features.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="centers"):
        fit_spectra_bag(spectra[:1, :1], 32, seed=0)
