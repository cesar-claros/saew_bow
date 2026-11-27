"""Features of the slice: token counts, the spectral control and the spectra of the bag of spectra (protocol, sections 6 and 7).

Three families give the features of a context ``(C, T)``:

- **Token counts.** A dictionary gives one token per pooling window: per channel
  for a single-channel dictionary, per window for a full-template dictionary.
  The features are the counts per atom over the tokens of a readout, divided by
  the number of tokens. The null token has its own count.
- **Spectral control.** Log energy and relative share in fixed bands, per channel,
  from the mean power of Hann frames. On a whole context this is the retained
  ``SpectralFeatures`` with frames of 1 s. On a part of a context the frames are
  spaced evenly, with an overlap of at least 50%, so that they cover the part.
- **Bag of spectra.** The unit-norm amplitude spectrum of every single-channel
  window. A k-means model (``eeg_compare.slice.spectra``) turns each spectrum
  into a token, and the tokens are counted as those of a dictionary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from eeg_compare.cells.alternating import Score, encode_alternating
from eeg_compare.cells.common import CellAtoms, extended_windows, same_padding
from eeg_compare.device import resolve_device

__all__ = ["BANDS", "band_features", "cell_tokens", "count_features", "frame_starts", "window_spectra"]

BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
_FLOOR = 1e-20
_FLAT = 1e-12


def frame_starts(length: int, nperseg: int) -> list[int]:
    """First samples of the frames that cover ``length`` samples with an overlap of at least 50%.

    The frames are spaced evenly from the first sample to the last possible
    start. When ``length - nperseg`` is a multiple of half a frame, these are the
    frames of a short-time transform with a hop of half a frame.

    Raises
    ------
    ValueError
        If the range is shorter than one frame.
    """
    if nperseg < 2 or length < nperseg:
        raise ValueError(f"a range of {length} samples holds no frame of {nperseg}")
    count = math.ceil((length - nperseg) / (nperseg // 2)) + 1
    if count == 1:
        return [0]
    return [round(index * (length - nperseg) / (count - 1)) for index in range(count)]


@torch.no_grad()
def band_features(
    rows: torch.Tensor,
    ranges: Sequence[tuple[int, int]],
    sfreq: float = 250.0,
    bands: Sequence[tuple[float, float]] = BANDS,
    nperseg: int = 250,
    device: str = "cpu",
    chunk: int = 512,
) -> NDArray[np.float64]:
    """Log band energies and relative shares per channel, from the samples of the given ranges.

    Parameters
    ----------
    rows : torch.Tensor
        Contexts ``(n, C, T)``.
    ranges : Sequence[tuple[int, int]]
        Half-open sample ranges of a readout. The frames of all ranges enter one mean.
    sfreq : float
        Sampling rate in Hz.
    bands : Sequence[tuple[float, float]]
        Bands ``(low, high)`` in Hz; a bin belongs to a band when ``low <= f < high``.
    nperseg : int
        Frame length in samples.
    device : str
        ``cpu``, ``cuda``, ``cuda:N`` or ``auto``.
    chunk : int
        Contexts per block.

    Returns
    -------
    numpy.ndarray
        ``(n, C * 2 * len(bands))``; per channel the log10 energies, then the shares.

    Raises
    ------
    ValueError
        If the contexts are not 3-D, there is no range, or a range leaves the context or holds no frame.
    """
    if rows.ndim != 3 or not ranges:
        raise ValueError("band features need contexts (n, C, T) and at least one range")
    if any(low < 0 or high > rows.shape[-1] for low, high in ranges):
        raise ValueError("a range leaves the context")
    target = resolve_device(device)
    starts = [low + first for low, high in ranges for first in frame_starts(high - low, nperseg)]
    window = torch.hann_window(nperseg, periodic=False, device=target)
    freqs = torch.arange(nperseg // 2 + 1, device=target) * sfreq / nperseg
    masks = [(freqs >= low) & (freqs < high) for low, high in bands]
    blocks = []
    for first in range(0, rows.shape[0], chunk):
        block = rows[first : first + chunk].to(target).float()
        frames = torch.stack([block[..., start : start + nperseg] for start in starts], dim=2)
        power = torch.fft.rfft(frames * window, dim=-1).abs().pow(2).mean(dim=2)
        energy = torch.stack([power[..., mask].sum(dim=-1) for mask in masks], dim=-1)
        share = energy / energy.sum(dim=-1, keepdim=True).clamp_min(_FLOOR)
        features = torch.cat([torch.log10(energy.clamp_min(_FLOOR)), share], dim=-1)
        blocks.append(features.flatten(1).double().cpu())
    if not blocks:
        return np.zeros((0, rows.shape[1] * 2 * len(bands)))
    return torch.cat(blocks).numpy()


@torch.no_grad()
def cell_tokens(
    rows: torch.Tensor, atoms: CellAtoms, score: Score, pool_len: int, device: str = "auto", chunk: int = 256
) -> torch.Tensor:
    """The tokens of contexts under a dictionary, by the assignment rule of the alternating cells.

    Parameters
    ----------
    rows : torch.Tensor
        Contexts ``(n, C, T)``; ``T`` is a multiple of ``pool_len``.
    atoms : CellAtoms
        A dictionary of the structure ``S1`` or ``SF``.
    score : str
        ``cosine`` or ``raw``: the score that selects the atom and the placement.
    pool_len : int
        Pooling window ``P``.
    device : str
        ``cpu``, ``cuda``, ``cuda:N`` or ``auto``.
    chunk : int
        Contexts per block.

    Returns
    -------
    torch.Tensor
        ``(n, M, c)`` atom indices on the CPU, ``-1`` for the null token. ``c`` is
        ``C`` for ``S1`` (one token per channel and window) and 1 for ``SF``.

    Raises
    ------
    ValueError
        If the dictionary has another structure.
    """
    if atoms.structure == "S1":
        assert atoms.waveform is not None
        bank = atoms.waveform.unsqueeze(1)
    elif atoms.structure == "SF":
        assert atoms.template is not None
        bank = atoms.template
    else:
        raise ValueError("the slice has tokens for the structures S1 and SF only")
    target = resolve_device(device)
    bank = bank.to(target, torch.float32)
    single = atoms.structure == "S1"
    blocks = []
    for first in range(0, rows.shape[0], chunk):
        block = rows[first : first + chunk].to(target, torch.float32)
        n_rows, n_channels, n_times = block.shape
        if single:
            flat = encode_alternating(block.reshape(n_rows * n_channels, 1, n_times), bank, score, pool_len, True)
            tokens = flat.atom.reshape(n_rows, n_channels, -1).permute(0, 2, 1)
        else:
            tokens = encode_alternating(block, bank, score, pool_len, False).atom.unsqueeze(-1)
        blocks.append(tokens.cpu())
    if not blocks:
        return torch.zeros((0, rows.shape[-1] // pool_len, rows.shape[1] if single else 1), dtype=torch.long)
    return torch.cat(blocks)


def count_features(tokens: torch.Tensor, windows: Sequence[int], n_atoms: int) -> NDArray[np.float64]:
    """Counts per atom over the tokens of the given windows, divided by the number of tokens.

    Parameters
    ----------
    tokens : torch.Tensor
        ``(n, M, c)`` atom indices; ``-1`` is the null token.
    windows : Sequence[int]
        Pooling windows of the readout.
    n_atoms : int
        Atoms ``K``.

    Returns
    -------
    numpy.ndarray
        ``(n, K + 1)``; the last column is the null token. Every row adds up to 1.

    Raises
    ------
    ValueError
        If the tokens are not 3-D, a window does not exist, or a token is not an atom of the dictionary.
    """
    if tokens.ndim != 3 or not windows:
        raise ValueError("count features need tokens (n, M, c) and at least one window")
    if min(windows) < 0 or max(windows) >= tokens.shape[1] or len(set(windows)) != len(windows):
        raise ValueError(f"the windows {tuple(windows)} are not distinct windows of {tokens.shape[1]}")
    chosen = tokens[:, list(windows), :].flatten(1).long()
    if chosen.numel() and (int(chosen.min()) < -1 or int(chosen.max()) >= n_atoms):
        raise ValueError("a token is not an atom of the dictionary")
    index = torch.where(chosen < 0, torch.full_like(chosen, n_atoms), chosen)
    counts = torch.zeros((chosen.shape[0], n_atoms + 1), dtype=torch.long)
    counts.scatter_add_(1, index, torch.ones_like(index))
    return (counts.double() / chosen.shape[1]).numpy()


@torch.no_grad()
def window_spectra(
    rows: torch.Tensor,
    atom_len: int,
    pool_len: int,
    sfreq: float = 250.0,
    band: tuple[float, float] = (1.0, 30.0),
    device: str = "auto",
    chunk: int = 256,
) -> torch.Tensor:
    """Unit-norm amplitude spectra of the single-channel windows of contexts.

    A window is the support of one pooling window of one channel, as for the
    token arms. Only the samples inside the context enter: their mean is removed
    and a Hann taper of their length is applied, so a window at the edge of a
    context has no step to zero. The transform has bins of 1 Hz or finer.

    Parameters
    ----------
    rows : torch.Tensor
        Contexts ``(n, C, T)``.
    atom_len : int
        Atom length ``L`` of the token arms.
    pool_len : int
        Pooling window ``P``.
    sfreq : float
        Sampling rate in Hz, a whole number.
    band : tuple[float, float]
        Lowest and highest frequency of the kept bins, both inside.
    device : str
        ``cpu``, ``cuda``, ``cuda:N`` or ``auto``.
    chunk : int
        Contexts per block.

    Returns
    -------
    torch.Tensor
        ``(n, M, C, F)`` on the CPU in single precision. A window without signal has a zero spectrum.
    """
    if abs(sfreq - round(sfreq)) > 1e-9:
        raise ValueError("the spectra need a whole sampling rate")
    target = resolve_device(device)
    n_times = rows.shape[-1]
    length = pool_len + atom_len - 1
    n_fft = int(round(sfreq)) * math.ceil(length / sfreq)
    freqs = torch.arange(n_fft // 2 + 1) * sfreq / n_fft
    keep = ((freqs >= band[0]) & (freqs <= band[1])).to(target)
    sample = torch.arange(n_times // pool_len)[:, None] * pool_len - same_padding(atom_len)[0] + torch.arange(length)
    real = (sample >= 0) & (sample < n_times)
    taper = torch.zeros(real.shape)
    for window in range(real.shape[0]):
        taper[window, real[window]] = torch.hann_window(int(real[window].sum()), periodic=False)
    real_on, taper_on = real.to(target)[None, :, None, :], taper.to(target)[None, :, None, :]
    blocks = []
    for first in range(0, rows.shape[0], chunk):
        windows, _ = extended_windows(rows[first : first + chunk].to(target, torch.float32), atom_len, pool_len)
        mean = (windows * real_on).sum(dim=-1, keepdim=True) / real_on.sum(dim=-1, keepdim=True)
        amplitude = torch.fft.rfft((windows - mean) * taper_on, n=n_fft, dim=-1).abs()[..., keep]
        norm = torch.linalg.vector_norm(amplitude, dim=-1, keepdim=True)
        blocks.append(torch.where(norm > _FLAT, amplitude / norm.clamp_min(_FLAT), torch.zeros_like(amplitude)).cpu())
    if not blocks:
        return torch.zeros((0, n_times // pool_len, rows.shape[1], int(keep.sum())))
    return torch.cat(blocks)
