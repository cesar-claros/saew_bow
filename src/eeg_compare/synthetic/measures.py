"""Measures of the synthetic benchmark (stage 2 protocol, section 5).

Recovery is a cross-correlation of unit vectors over shifts: the planted
waveform (or template) against an atom, with zeros outside the atom. A shift
``d`` means that the atom holds the planted frame ``d`` samples after the first
sample of the atom. So a token of that atom that starts at sample ``s`` places
the planted frame at ``s + d``.

Event detection maps every atom to the template that it recovers best, when
that recovery reaches a floor. A token of a mapped atom is a claim. An event is
detected when a claim of its template lies within a tolerance of its start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from eeg_compare.cells.common import CellAtoms, Tokens
from eeg_compare.synthetic.generator import Events

__all__ = [
    "Recovery",
    "atom_map",
    "decoy_recovery",
    "detection",
    "shifted_correlation",
    "template_recovery",
    "waveform_recovery",
]

_EPS = 1e-12


@dataclass(frozen=True)
class Recovery:
    """Recovery of ``N`` planted items by ``K`` atoms.

    Parameters
    ----------
    value : torch.Tensor
        ``(N,)`` largest correlation over atoms and shifts.
    per_atom : torch.Tensor
        ``(N, K)`` largest correlation of every atom over shifts.
    shift : torch.Tensor
        ``(N, K)`` shift of that largest correlation, in samples.
    """

    value: torch.Tensor
    per_atom: torch.Tensor
    shift: torch.Tensor


def _unit(values: torch.Tensor) -> torch.Tensor:
    """Unit norm over all axes but the first; a zero item stays zero."""
    norms = torch.linalg.vector_norm(values.flatten(1), dim=1).clamp_min(_EPS)
    return values / norms.view(-1, *([1] * (values.ndim - 1)))


def shifted_correlation(planted: torch.Tensor, atoms: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Correlations ``(N, K, 2 * max_shift + 1)`` of planted items ``(N, c, L)`` with atoms ``(K, c, L)``.

    Entry ``j`` is the shift ``d = j - max_shift``: the sum of ``planted[t] * atom[t + d]`` over the samples
    where both exist.
    """
    if planted.ndim != 3 or atoms.ndim != 3 or planted.shape[1:] != atoms.shape[1:]:
        raise ValueError(f"planted {tuple(planted.shape)} and atoms {tuple(atoms.shape)} do not agree")
    return F.conv1d(F.pad(atoms, (max_shift, max_shift)), planted).transpose(0, 1)


def _reduce(correlation: torch.Tensor, max_shift: int) -> Recovery:
    """The best shift of every atom, then the best atom, of correlations ``(N, K, J)``."""
    per_atom, index = correlation.max(dim=-1)
    return Recovery(per_atom.max(dim=-1).values, per_atom, index - max_shift)


def waveform_recovery(
    atoms: CellAtoms, waveforms: torch.Tensor, patterns: torch.Tensor | None, max_shift: int
) -> Recovery:
    """Recovery of planted waveforms ``(N, L)`` by the waveforms of the atoms. The value is signed.

    For ``S1`` the waveform of an atom is the atom. For ``SF`` it is the atom
    projected on the planted spatial pattern, scaled to unit norm. For ``SR`` it
    is the waveform of the atom with the sign of the projection of each of its
    spatial patterns on the planted pattern; the best of them counts.

    Parameters
    ----------
    atoms : CellAtoms
        The dictionary of a cell.
    waveforms : torch.Tensor
        ``(N, L)`` unit waveforms.
    patterns : torch.Tensor | None
        ``(N, C)`` planted spatial pattern of every waveform; needed for ``SF`` and ``SR``.
    max_shift : int
        Largest shift in samples.

    Raises
    ------
    ValueError
        If ``SF`` or ``SR`` atoms come without patterns.
    """
    planted = waveforms.double().unsqueeze(1)
    if atoms.structure == "S1":
        assert atoms.waveform is not None
        return _reduce(
            shifted_correlation(planted, _unit(atoms.waveform.double().cpu()).unsqueeze(1), max_shift), max_shift
        )
    if patterns is None:
        raise ValueError("the structures SF and SR need the planted spatial patterns")
    patterns = patterns.double()
    if atoms.structure == "SF":
        assert atoms.template is not None
        template = atoms.template.double().cpu()
        rows = []
        for index in range(planted.shape[0]):
            projected = _unit(torch.einsum("c,kcl->kl", patterns[index], template)).unsqueeze(1)
            rows.append(shifted_correlation(planted[index : index + 1], projected, max_shift))
        return _reduce(torch.cat(rows), max_shift)
    assert atoms.waveform is not None and atoms.pattern is not None
    correlation = shifted_correlation(planted, _unit(atoms.waveform.double().cpu()).unsqueeze(1), max_shift)
    signs = torch.sign(torch.einsum("nc,kcd->nkd", patterns, atoms.pattern.double().cpu()))
    signed = (signs.unsqueeze(-1) * correlation.unsqueeze(2)).max(dim=2).values
    return _reduce(signed, max_shift)


def template_recovery(atoms: CellAtoms, templates: torch.Tensor, max_shift: int) -> Recovery:
    """Recovery of planted templates ``(N, C, L)`` by the full templates of ``SF`` or ``SR`` atoms.

    A rank-1 atom has one template ``u_d w^T`` per spatial pattern; the best of them counts.

    Raises
    ------
    ValueError
        For ``S1`` atoms, which have no multichannel template.
    """
    planted = templates.double()
    if atoms.structure == "SF":
        assert atoms.template is not None
        return _reduce(shifted_correlation(planted, _unit(atoms.template.double().cpu()), max_shift), max_shift)
    if atoms.structure != "SR":
        raise ValueError("the structure S1 has no multichannel template")
    assert atoms.waveform is not None and atoms.pattern is not None
    waveform, pattern = atoms.waveform.double().cpu(), atoms.pattern.double().cpu()
    n_atoms, _, n_projections = pattern.shape
    full = _unit(torch.einsum("kcd,kl->kdcl", pattern, waveform).flatten(0, 1))
    correlation = shifted_correlation(planted, full, max_shift).unflatten(1, (n_atoms, n_projections))
    return _reduce(correlation.max(dim=2).values, max_shift)


def decoy_recovery(atoms: CellAtoms, decoys: torch.Tensor, patterns: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Recovery ``(N',)`` of waveforms that are never planted; for ``SF`` and ``SR`` the best over the planted patterns."""
    if atoms.structure == "S1":
        return waveform_recovery(atoms, decoys, None, max_shift).value
    values = [
        waveform_recovery(atoms, decoys, pattern.expand(decoys.shape[0], -1), max_shift).value for pattern in patterns
    ]
    return torch.stack(values).max(dim=0).values


def atom_map(recovery: Recovery, floor: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """The template of every atom and the shift of that recovery.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(K,)`` template with the largest recovery by the atom, or ``-1`` when
        that recovery is below ``floor``; and ``(K,)`` the shift of that recovery.
    """
    best, template = recovery.per_atom.max(dim=0)
    shift = recovery.shift.gather(0, template.unsqueeze(0)).squeeze(0)
    return torch.where(best >= floor, template, torch.full_like(template, -1)), shift


def detection(
    tokens: Tokens, events: Events, template_of_atom: torch.Tensor, shift_of_atom: torch.Tensor, tolerance: int
) -> dict[str, Any]:
    """Event detection of one token set.

    Parameters
    ----------
    tokens : Tokens
        ``(B, M)`` tokens of the test segments.
    events : Events
        The events of those segments.
    template_of_atom, shift_of_atom : torch.Tensor
        ``(K,)`` results of ``atom_map``.
    tolerance : int
        Largest distance in samples between the claimed and the true start of a detected event.

    Returns
    -------
    dict[str, Any]
        ``n_events``, ``n_tokens``, ``n_claims``, ``recall``, ``precision``,
        ``f1`` and ``position_error`` (``None`` where a ratio has no
        denominator), ``recall_by_template``, ``recall_by_sign``, and ``pairs``
        (for events in pairs: the shares of the pairs with both events, one
        event and no event detected).
    """
    atom, start = tokens.atom.cpu(), tokens.start.cpu()
    mapped = torch.where(atom >= 0, template_of_atom.cpu()[atom.clamp_min(0)], torch.full_like(atom, -1))
    claimed = start + shift_of_atom.cpu()[atom.clamp_min(0)]
    claims = mapped >= 0
    n_events, n_claims = len(events), int(claims.sum())
    result: dict[str, Any] = {
        "n_events": n_events,
        "n_tokens": int((atom >= 0).sum()),
        "n_claims": n_claims,
        "recall": None,
        "precision": None,
        "f1": None,
        "position_error": None,
        "recall_by_template": {},
        "recall_by_sign": {},
        "pairs": None,
    }
    if n_events == 0:
        return result
    distance = (claimed[events.segment] - events.start[:, None]).abs()
    match = (mapped[events.segment] == events.template[:, None]) & (distance <= tolerance)
    detected = match.any(dim=1)
    correct = torch.zeros(atom.shape, dtype=torch.long).index_add_(0, events.segment, match.long()) > 0

    def share(selected: torch.Tensor) -> float:
        """The share of true entries, from integer counts, so that a ratio has no float32 rounding."""
        return int(selected.sum()) / int(selected.numel())

    recall = share(detected)
    result["recall"] = recall
    if n_claims:
        precision = share(correct[claims])
        result["precision"] = precision
        result["f1"] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    if bool(detected.any()):
        nearest = torch.where(match, distance, torch.full_like(distance, 10**9)).min(dim=1).values
        result["position_error"] = int(nearest[detected].sum()) / int(detected.sum())
    for template in events.template.unique().tolist():
        result["recall_by_template"][str(template)] = share(detected[events.template == template])
    for sign in events.sign.unique().tolist():
        result["recall_by_sign"][str(sign)] = share(detected[events.sign == sign])
    if events.group is not None:
        found = torch.zeros(int(events.group.max()) + 1, dtype=torch.long).index_add_(0, events.group, detected.long())
        result["pairs"] = {
            "n": int(found.numel()),
            "both": share(found == 2),
            "one": share(found == 1),
            "none": share(found == 0),
        }
    return result
