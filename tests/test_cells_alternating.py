"""Alternating learners of the comparison cells (rules L1 to L4); run on the HPC.

The checks use small deterministic arrays in float64. The search and the update
are compared with direct loops, and one planted waveform must stay recovered
under every rule.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from eeg_compare.cells.alternating import (
    RULES,
    AlternatingSpec,
    assign,
    encode_alternating,
    fit_alternating,
    training_units,
    update,
)
from eeg_compare.cells.common import extended_windows, same_padding, unit_norm

ATOM_LEN, POOL_LEN = 8, 16


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _direct_assignment(
    units: torch.Tensor, atoms: torch.Tensor, score: str, valid: torch.Tensor
) -> tuple[list[int], list[int], list[float]]:
    """Labels, shifts and scores by a loop over atoms and shifts; the first largest score wins."""
    labels, shifts, values = [], [], []
    atom_len = atoms.shape[-1]
    for index in range(units.shape[0]):
        best, label, shift = 0.0, -1, 0
        for atom in range(atoms.shape[0]):
            for start in range(units.shape[-1] - atom_len + 1):
                if not bool(valid[index, start]):
                    continue
                part = units[index, :, start : start + atom_len]
                value = float((part * atoms[atom]).sum())
                if score == "cosine":
                    value /= float(part.norm() * atoms[atom].norm())
                if value > best:
                    best, label, shift = value, atom, start
        labels.append(label)
        shifts.append(shift)
        values.append(best)
    return labels, shifts, values


def test_rules_are_the_path_of_the_plan() -> None:
    assert {name: (rule.score, rule.weight) for name, rule in RULES.items()} == {
        "L1": ("cosine", "unit"),
        "L2": ("cosine", "norm"),
        "L3": ("cosine", "least_squares"),
        "L4": ("raw", "least_squares"),
    }
    with pytest.raises(ValueError, match="rule"):
        AlternatingSpec("L5", ATOM_LEN, POOL_LEN, True)


def test_extended_windows_follow_the_pooling_grid_of_the_adapters() -> None:
    rows = _randn(2, 3, 32, seed=1)
    windows, valid = extended_windows(rows, 8, 8)
    assert same_padding(8) == (3, 4)
    assert windows.shape == (2, 4, 3, 15) and valid.shape == (4, 8)
    padded = F.pad(rows, (3, 4))
    for window in range(4):
        assert torch.equal(windows[:, window], padded[..., window * 8 : window * 8 + 15])
    centers = torch.nonzero(valid.flatten()).flatten().tolist()
    assert centers == list(range(3, 28)), "the valid placements are the T - L + 1 placements inside the segment"
    with pytest.raises(ValueError, match="multiple"):
        extended_windows(_randn(1, 3, 30), 8, 8)
    with pytest.raises(ValueError, match="no complete placement"):
        extended_windows(_randn(1, 3, 8), 16, 8)


def test_training_units_keep_segment_window_and_channel_order() -> None:
    rows = _randn(2, 3, 32, seed=2)
    windows, valid = extended_windows(rows, 8, 8)
    single, single_valid = training_units(rows, 8, 8, single_channel=True)
    full, full_valid = training_units(rows, 8, 8, single_channel=False)
    assert single.shape == (2 * 4 * 3, 1, 15) and full.shape == (2 * 4, 3, 15)
    for segment, window, channel in ((0, 0, 0), (1, 2, 1), (1, 3, 2)):
        index = (segment * 4 + window) * 3 + channel
        assert torch.equal(single[index, 0], windows[segment, window, channel])
        assert torch.equal(single_valid[index], valid[window])
        assert torch.equal(full[segment * 4 + window], windows[segment, window])
        assert torch.equal(full_valid[segment * 4 + window], valid[window])


@pytest.mark.parametrize("score", ["cosine", "raw"])
def test_assign_equals_a_direct_search(score: str) -> None:
    units = _randn(40, 2, 15, seed=3)
    atoms = unit_norm(_randn(3, 2, 8, seed=4))
    valid = torch.rand(40, 8, generator=torch.Generator().manual_seed(5)) > 0.3
    valid[:, 0] = True
    valid[7] = False
    labels, shifts, values = _direct_assignment(units, atoms, score, valid)
    found = assign(units, atoms, score, valid, chunk=16)
    assert found.label.tolist() == labels and found.shift.tolist() == shifts
    assert int(found.label[7]) == -1 and float(found.response[7]) == 0.0
    ours = found.cosine if score == "cosine" else found.response
    assert torch.allclose(ours, torch.tensor(values, dtype=torch.float64), atol=1e-12)
    assigned = found.label >= 0
    ratio = found.response[assigned] / found.cosine[assigned]
    assert bool((ratio > 0).all()), "the response and the cosine of a placement have one sign"


def test_a_unit_without_a_positive_response_gets_no_atom() -> None:
    atoms = torch.full((1, 1, 8), 8**-0.5, dtype=torch.float64)
    units = torch.stack([-torch.ones(1, 15), torch.zeros(1, 15), torch.ones(1, 15)]).double()
    for score in ("cosine", "raw"):
        found = assign(units, atoms, score)  # type: ignore[arg-type]
        assert found.label.tolist() == [-1, -1, 0]
        assert found.response[:2].tolist() == [0.0, 0.0] and found.cosine[:2].tolist() == [0.0, 0.0]
        assert abs(float(found.cosine[2]) - 1.0) < 1e-12
    with pytest.raises(ValueError, match="score"):
        assign(units, atoms, "euclidean")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shorter"):
        assign(units[..., :4], atoms, "raw")


@pytest.mark.parametrize("weight", ["unit", "norm", "least_squares"])
def test_update_is_the_weighted_sum_of_the_aligned_sub_windows(weight: str) -> None:
    units = _randn(30, 2, 15, seed=6)
    atoms = unit_norm(_randn(3, 2, 8, seed=7))
    generator = torch.Generator().manual_seed(8)
    label = torch.randint(0, 2, (30,), generator=generator)
    label[4] = -1
    shift = torch.randint(0, 8, (30,), generator=generator)
    expected = torch.zeros(3, 2, 8, dtype=torch.float64)
    for index in range(30):
        atom = int(label[index])
        if atom < 0:
            continue
        part = units[index, :, int(shift[index]) : int(shift[index]) + 8]
        if weight == "unit":
            expected[atom] += part / part.norm()
        elif weight == "norm":
            expected[atom] += part
        else:
            expected[atom] += (part * atoms[atom]).sum() * part
    updated, counts = update(units, label, shift, atoms, weight, chunk=7)  # type: ignore[arg-type]
    assert counts.tolist() == [int((label == 0).sum()), int((label == 1).sum()), 0]
    assert torch.allclose(updated[:2], unit_norm(expected[:2]), atol=1e-12)
    assert torch.equal(updated[2], atoms[2]), "an atom without a unit keeps its value"
    assert torch.allclose(updated.flatten(1).norm(dim=1), torch.ones(3, dtype=torch.float64))


def test_one_pass_is_one_assignment_and_one_update() -> None:
    rows = _randn(6, 3, 64, seed=9)
    for single, channels in ((True, 1), (False, 3)):
        initial = _randn(4, channels, ATOM_LEN, seed=10)
        units, valid = training_units(rows, ATOM_LEN, POOL_LEN, single)
        for name, rule in RULES.items():
            fit = fit_alternating(rows, initial, AlternatingSpec(name, ATOM_LEN, POOL_LEN, single, max_passes=1))
            start = unit_norm(initial)
            found = assign(units, start, rule.score, valid)
            expected, _ = update(units, found.label, found.shift, start, rule.weight)
            assert fit.passes == 1 and len(fit.history) == 1
            assert torch.equal(fit.atoms, expected)
            assert fit.converged == (fit.history[0]["change"] < 1e-4)
    with pytest.raises(ValueError, match="initial atoms"):
        fit_alternating(rows, _randn(4, 2, ATOM_LEN), AlternatingSpec("L1", ATOM_LEN, POOL_LEN, False))


def _planted_rows() -> tuple[torch.Tensor, torch.Tensor]:
    """Segments with one waveform on channel 0 in the middle of pooling windows 1 and 2, in weak noise.

    With ``P = 2 L`` and these starts, no other pooling window overlaps an event.
    """
    waveform = torch.tensor([0.0, 1.0, 3.0, -2.0, -1.0, 0.5, 0.0, 0.0], dtype=torch.float64)
    waveform = waveform / waveform.norm()
    rows = 0.1 * _randn(24, 2, 64, seed=11)
    for start in (20, 36):
        rows[:, 0, start : start + ATOM_LEN] += 4.0 * waveform
    return rows, waveform


@pytest.mark.parametrize("rule", ["L1", "L2", "L3", "L4"])
def test_a_planted_waveform_stays_recovered_by_every_rule(rule: str) -> None:
    rows, waveform = _planted_rows()
    single_start = torch.stack([rows[0, 0:1, 20:28], rows[0, 1:2, 0:8]])
    fit = fit_alternating(rows, single_start, AlternatingSpec(rule, ATOM_LEN, POOL_LEN, True, max_passes=20))
    assert float((fit.atoms[:, 0] @ waveform).max()) >= 0.95
    assert fit.passes == len(fit.history) <= 20
    for record in fit.history:
        assert 0.0 <= record["residual_share"] <= 1.0 and 0.0 < record["assigned_share"] <= 1.0
    template = torch.zeros(2, ATOM_LEN, dtype=torch.float64)
    template[0] = waveform
    full_start = torch.stack([rows[0, :, 20:28], rows[0, :, 0:8]])
    full = fit_alternating(rows, full_start, AlternatingSpec(rule, ATOM_LEN, POOL_LEN, False, max_passes=20))
    assert float((full.atoms * template).sum(dim=(1, 2)).max()) >= 0.95


def test_tokens_find_a_planted_atom_at_its_place() -> None:
    atoms = unit_norm(_randn(2, 1, ATOM_LEN, seed=12))
    templates = unit_norm(_randn(2, 3, ATOM_LEN, seed=13))
    for start in (0, 20, 36, 56):
        window = (start + same_padding(ATOM_LEN)[0]) // POOL_LEN
        for score in ("cosine", "raw"):
            rows = torch.zeros(1, 3, 64, dtype=torch.float64)
            rows[0, 2, start : start + ATOM_LEN] = 2.5 * atoms[1, 0]
            tokens = encode_alternating(rows, atoms, score, POOL_LEN, single_channel=True)  # type: ignore[arg-type]
            assert tokens.atom.shape == (1, 4) and tokens.active.sum() == 1
            assert int(tokens.atom[0, window]) == 1 and int(tokens.start[0, window]) == start
            assert tokens.channel is not None and int(tokens.channel[0, window]) == 2
            assert abs(float(tokens.response[0, window]) - 2.5) < 1e-10
            assert abs(float(tokens.value[0, window]) - (1.0 if score == "cosine" else 2.5)) < 1e-10
            assert int(tokens.start[0, (window + 1) % 4]) == -1
            rows = torch.zeros(1, 3, 64, dtype=torch.float64)
            rows[0, :, start : start + ATOM_LEN] = 2.5 * templates[0]
            tokens = encode_alternating(rows, templates, score, POOL_LEN, single_channel=False)  # type: ignore[arg-type]
            assert tokens.channel is None and tokens.active.sum() == 1
            assert int(tokens.atom[0, window]) == 0 and int(tokens.start[0, window]) == start


def test_the_single_channel_token_is_that_of_the_best_channel_by_the_score_of_the_cell() -> None:
    atom = torch.zeros(1, 1, ATOM_LEN, dtype=torch.float64)
    atom[0, 0, 3] = 1.0
    rows = torch.zeros(1, 2, 64, dtype=torch.float64)
    rows[0, 0, 23] = 1.0  # the atom at sample 20 with amplitude 1: cosine 1
    rows[0, 1, 25] = 2.4  # the atom at sample 22 with amplitude 2.4 ...
    rows[0, 1, 27] = 1.8  # ... plus an orthogonal part: cosine 0.8
    by_cosine = encode_alternating(rows, atom, "cosine", POOL_LEN, single_channel=True)
    by_raw = encode_alternating(rows, atom, "raw", POOL_LEN, single_channel=True)
    assert by_cosine.channel is not None and by_raw.channel is not None
    assert (int(by_cosine.channel[0, 1]), int(by_cosine.start[0, 1])) == (0, 20)
    assert (int(by_raw.channel[0, 1]), int(by_raw.start[0, 1])) == (1, 22)
    assert abs(float(by_cosine.value[0, 1]) - 1.0) < 1e-12 and abs(float(by_raw.value[0, 1]) - 2.4) < 1e-12
