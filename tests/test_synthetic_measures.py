"""Measures of the synthetic benchmark on hand-made dictionaries and tokens; run on the HPC."""

from __future__ import annotations

import pytest
import torch

from eeg_compare.cells.common import CellAtoms, Tokens
from eeg_compare.synthetic.generator import Events
from eeg_compare.synthetic.measures import (
    Recovery,
    atom_map,
    decoy_recovery,
    detection,
    shifted_correlation,
    template_recovery,
    waveform_recovery,
)

MAX_SHIFT = 5


def _bump(at: int = 6) -> torch.Tensor:
    """A positive unit waveform of 16 samples with 4 samples of support."""
    waveform = torch.zeros(16, dtype=torch.float64)
    waveform[at : at + 4] = torch.tensor([1.0, 2.0, 2.0, 1.0], dtype=torch.float64)
    return waveform / waveform.norm()


WAVEFORM, DELAYED = _bump(6), _bump(9)  # DELAYED holds WAVEFORM three samples later
PATTERN = torch.tensor([0.6, 0.8, 0.0], dtype=torch.float64)
OTHER = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
TEMPLATE = PATTERN[:, None] * WAVEFORM[None, :]


def test_a_shift_is_the_delay_of_the_planted_frame_inside_the_atom() -> None:
    correlation = shifted_correlation(WAVEFORM[None, None], DELAYED[None, None], MAX_SHIFT)
    assert correlation.shape == (1, 1, 2 * MAX_SHIFT + 1)
    assert int(correlation[0, 0].argmax()) == MAX_SHIFT + 3
    assert abs(float(correlation[0, 0, MAX_SHIFT + 3]) - 1.0) < 1e-12
    assert float(correlation[0, 0, :MAX_SHIFT].abs().max()) < 1e-12, (
        "an atom that holds the frame later has no match earlier"
    )
    with pytest.raises(ValueError, match="do not agree"):
        shifted_correlation(WAVEFORM[None, None], DELAYED[None, None, :8], MAX_SHIFT)


def test_waveform_recovery_of_single_channel_atoms_is_signed() -> None:
    atoms = CellAtoms("S1", waveform=torch.stack([-WAVEFORM, DELAYED]))
    recovery = waveform_recovery(atoms, WAVEFORM[None], None, MAX_SHIFT)
    assert abs(float(recovery.value[0]) - 1.0) < 1e-12
    assert int(recovery.per_atom[0].argmax()) == 1 and int(recovery.shift[0, 1]) == 3
    assert float(recovery.per_atom[0, 0]) <= 1e-12, "an inverted waveform is not a recovery"
    decoys = torch.stack([_bump(8), DELAYED])  # one sample and no sample away from the second atom
    assert torch.allclose(decoy_recovery(atoms, decoys, PATTERN[None], MAX_SHIFT), torch.ones(2, dtype=torch.float64))


def test_recovery_of_full_templates_projects_on_the_planted_pattern() -> None:
    held = PATTERN[:, None] * DELAYED[None, :]
    atoms = CellAtoms("SF", template=torch.stack([held, -held]))
    recovery = waveform_recovery(atoms, WAVEFORM[None], PATTERN[None], MAX_SHIFT)
    assert abs(float(recovery.value[0]) - 1.0) < 1e-12 and int(recovery.shift[0, 0]) == 3
    assert float(recovery.per_atom[0, 1]) <= 1e-12
    full = template_recovery(atoms, TEMPLATE[None], MAX_SHIFT)
    assert abs(float(full.value[0]) - 1.0) < 1e-12 and int(full.shift[0, 0]) == 3
    off_pattern = CellAtoms("SF", template=(OTHER[:, None] * DELAYED[None, :])[None])
    assert float(waveform_recovery(off_pattern, WAVEFORM[None], PATTERN[None], MAX_SHIFT).value[0]) <= 1e-12
    assert float(template_recovery(off_pattern, TEMPLATE[None], MAX_SHIFT).value[0]) <= 1e-12
    assert torch.allclose(
        decoy_recovery(atoms, DELAYED[None], torch.stack([OTHER, PATTERN]), MAX_SHIFT),
        torch.ones(1, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="patterns"):
        waveform_recovery(atoms, WAVEFORM[None], None, MAX_SHIFT)


def test_recovery_of_rank_one_atoms_follows_the_template_and_not_the_factors() -> None:
    # Atom 0 holds the template with its first pattern. Atom 1 holds it with both factors reversed.
    waveform = torch.stack([DELAYED, -DELAYED])
    pattern = torch.stack([torch.stack([PATTERN, OTHER], dim=-1), torch.stack([-PATTERN, -PATTERN], dim=-1)])
    atoms = CellAtoms("SR", waveform=waveform, pattern=pattern)
    recovery = waveform_recovery(atoms, WAVEFORM[None], PATTERN[None], MAX_SHIFT)
    assert torch.allclose(recovery.per_atom[0], torch.ones(2, dtype=torch.float64))
    assert recovery.shift[0].tolist() == [3, 3]
    full = template_recovery(atoms, TEMPLATE[None], MAX_SHIFT)
    assert torch.allclose(full.per_atom[0], torch.ones(2, dtype=torch.float64)) and full.shift[0].tolist() == [3, 3]
    inverted = CellAtoms("SR", waveform=DELAYED[None], pattern=-PATTERN[None, :, None])
    assert float(waveform_recovery(inverted, WAVEFORM[None], PATTERN[None], MAX_SHIFT).value[0]) <= 1e-12
    assert float(template_recovery(inverted, TEMPLATE[None], MAX_SHIFT).value[0]) <= 1e-12
    with pytest.raises(ValueError, match="no multichannel"):
        template_recovery(CellAtoms("S1", waveform=WAVEFORM[None]), TEMPLATE[None], MAX_SHIFT)


def test_an_atom_maps_to_its_best_template_above_the_floor() -> None:
    recovery = Recovery(
        value=torch.tensor([0.9, 0.6]),
        per_atom=torch.tensor([[0.9, 0.2, 0.4], [0.3, 0.6, 0.45]]),
        shift=torch.tensor([[1, 2, 3], [4, 5, 6]]),
    )
    template, shift = atom_map(recovery, floor=0.5)
    assert template.tolist() == [0, 1, -1] and shift.tolist() == [1, 5, 6]


def _tokens() -> Tokens:
    atom = torch.tensor([[0, 1, -1], [2, -1, -1]])
    start = torch.tensor([[8, 70, -1], [30, -1, -1]])
    zeros = torch.zeros(2, 3, dtype=torch.float64)
    return Tokens(atom=atom, start=start, value=zeros, response=zeros)


def _events() -> Events:
    return Events(
        segment=torch.tensor([0, 0, 1]),
        template=torch.tensor([0, 1, 0]),
        start=torch.tensor([10, 50, 30]),
        amplitude=torch.ones(3, dtype=torch.float64),
        sign=torch.tensor([1, -1, 1]),
    )


def test_detection_counts_claims_of_mapped_atoms_near_the_event_start() -> None:
    template_of_atom = torch.tensor([0, 1, -1])
    found = detection(_tokens(), _events(), template_of_atom, torch.tensor([3, -20, 0]), tolerance=2)
    assert (found["n_events"], found["n_tokens"], found["n_claims"]) == (3, 3, 2)
    assert abs(found["recall"] - 2 / 3) < 1e-12 and found["precision"] == 1.0 and abs(found["f1"] - 0.8) < 1e-12
    assert found["position_error"] == 0.5
    assert found["recall_by_template"] == {"0": 0.5, "1": 1.0} and found["recall_by_sign"] == {"-1": 1.0, "1": 0.5}
    late = detection(_tokens(), _events(), template_of_atom, torch.tensor([5, -20, 0]), tolerance=2)
    assert abs(late["recall"] - 1 / 3) < 1e-12 and late["precision"] == 0.5 and late["position_error"] == 0.0


def test_detection_without_claims_or_without_events() -> None:
    unmapped = torch.tensor([-1, -1, -1])
    found = detection(_tokens(), _events(), unmapped, torch.zeros(3, dtype=torch.long), tolerance=2)
    assert found["n_claims"] == 0 and found["recall"] == 0.0 and found["precision"] is None and found["f1"] is None
    assert found["position_error"] is None
    empty = Events(
        *(torch.zeros(0, dtype=torch.long) for _ in range(3)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    )
    quiet = detection(_tokens(), empty, torch.tensor([0, 1, -1]), torch.zeros(3, dtype=torch.long), tolerance=2)
    assert quiet["n_events"] == 0 and quiet["n_claims"] == 2 and quiet["recall"] is None and quiet["precision"] is None
