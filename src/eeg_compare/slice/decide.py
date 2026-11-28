"""The reads of step 3 of the slice: M1 and M4 to M11, the predictions S1 to S7 and the rules of section 13.

The module has two parts. ``measure`` takes the numbers from a stored run.
``evaluate`` applies the rules to those numbers and reads no data, so the rules
can be checked by hand. Both follow the protocol as it was fixed before the run
(sections 10, 11, 13 and 16.3).

- A difference is the first arm minus the second, on the ``full`` readout, as a
  mean over the seeds of each arm. A level has one seed; it is compared with the
  same seed of the primary arm.
- ``h`` is the half-width of the paired interval of ``S1-L2`` minus
  ``S1-frozen-data`` with the logistic head.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from eeg_compare.slice.evaluate import (
    bootstrap_multiplicities,
    confusion,
    interval,
    patient_strata,
    recall_draws,
    summarize,
)
from eeg_compare.slice.report import RunData, _table

__all__ = ["HEADS", "PRIMARY_ARMS", "decide", "evaluate", "measure", "render_decisions"]

HEADS = ("logistic", "forest")
LEARNED = ("S1-L1", "S1-L2")
FROZEN = ("S1-frozen-data", "S1-frozen-gauss", "SF-frozen-data")
PRIMARY_ARMS = ("spectral", "spectra_bag", *LEARNED, *FROZEN)
EPILEPTIFORM = ("gped", "pled")
RECORD_LIMIT = 0.10


class _Scores:
    """Scores and bootstrap draws of the arms of a run, on one set of draws."""

    def __init__(self, data: RunData, n_draws: int, seed: int) -> None:
        self.data = data
        self.n_classes = len(data.labels)
        self.index, stratum = patient_strata(data.targets, data.patients, self.n_classes)
        self.multiplicities = bootstrap_multiplicities(stratum, n_draws, seed)

    def has(self, arm: str, head: str = "logistic", readout: str = "full") -> bool:
        return (arm, head, readout) in self.data.groups

    def _runs(self, arm: str, head: str, readout: str, seeds: tuple[int, ...] | None) -> list[NDArray[np.int64]]:
        runs = self.data.groups[(arm, head, readout)]
        found = [values for (arm_seed, _), values in runs.items() if seeds is None or arm_seed in seeds]
        if not found:
            raise ValueError(f"the run holds no predictions of {arm} for the seeds {seeds}")
        return found

    def recall(
        self, arm: str, head: str, readout: str = "full", seeds: tuple[int, ...] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The recall of every class: its value, and its draws ``(n_draws, n_classes)``."""
        runs = self._runs(arm, head, readout, seeds)
        targets = self.data.targets
        point = np.mean([summarize(confusion(targets, values, self.n_classes))["recall"] for values in runs], axis=0)
        draws = np.mean(
            [recall_draws(targets, values, self.index, self.multiplicities, self.n_classes) for values in runs], axis=0
        )
        return point, draws

    def score(
        self, arm: str, head: str, readout: str = "full", seeds: tuple[int, ...] | None = None
    ) -> tuple[float, NDArray[np.float64]]:
        """The balanced accuracy: its value, and its draws."""
        point, draws = self.recall(arm, head, readout, seeds)
        return float(np.nanmean(point)), np.nanmean(draws, axis=1)

    def pair(
        self,
        first: str,
        second: str,
        head: str,
        seeds_first: tuple[int, ...] | None = None,
        seeds_second: tuple[int, ...] | None = None,
        readouts: tuple[str, str] = ("full", "full"),
    ) -> dict[str, Any]:
        """The paired difference of two arms: value, interval and half-width."""
        a, draws_a = self.score(first, head, readouts[0], seeds_first)
        b, draws_b = self.score(second, head, readouts[1], seeds_second)
        return {"first": first, "second": second, "head": head, "difference": a - b, **interval(draws_a - draws_b)}


def measure(data: RunData, n_draws: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """The numbers of step 3 from a stored run; ``None`` when the run holds no learned arm.

    Returns
    -------
    dict
        ``h`` and its pair (M1), the primary scores, the tables ``m4`` to
        ``m11`` (lists of paired differences), the differences that the
        predictions need, and the record effect of the best arm by class.
    """
    scores = _Scores(data, n_draws, seed)
    if not scores.has("S1-L2"):
        return None
    settings = data.report["settings"]
    first = (min(arm_seed for arm_seed, _ in data.groups[("S1-L2", "logistic", "full")]),)
    wide = f"P{settings['wide_pool_len']}"

    def both(a: str, b: str, seeds_b: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
        return [scores.pair(a, b, head, None, seeds_b) for head in HEADS if scores.has(a, head) and scores.has(b, head)]

    m1 = scores.pair("S1-L2", "S1-frozen-data", "logistic")
    primary = {arm: scores.score(arm, "logistic")[0] for arm in PRIMARY_ARMS}
    forest = {arm: scores.score(arm, "forest")[0] for arm in PRIMARY_ARMS}
    best = max(primary, key=lambda arm: primary[arm])
    effect = []
    full_point, full_draws = scores.recall(best, "logistic", "full")
    for other in ("displaced", "context"):
        point, draws = scores.recall(best, "logistic", other)
        for label, name in enumerate(data.labels):
            effect.append(
                {
                    "class": name,
                    "against": other,
                    "difference": float(full_point[label] - point[label]),
                    **interval(full_draws[:, label] - draws[:, label]),
                }
            )
    levels = {
        **{f"K{count}": both(f"S1-L2@K{count}", "S1-L2", first) for count in settings["level_atoms"]},
        "raw tokens": both("S1-L2@raw", "S1-L2"),
        f"tiling {wide}": both(f"S1-L2@{wide}", "S1-L2", first),
        "profile input": both("S1-L2@profile-raw", "S1-L2@raw", first),
        "10 minutes": both("S1-L2@10min", "S1-L2", first),
    }
    return {
        "h": m1["half_width"],
        "m1": m1,
        "primary_scores": primary,
        "forest_scores": forest,
        "best_arm": best,
        "learned_minus_frozen": m1["difference"],
        "bag_minus_learned": scores.pair("spectra_bag", "S1-L2", "logistic")["difference"],
        "cosine_minus_raw": scores.pair("S1-L2", "S1-L2@raw", "logistic")["difference"],
        "profile_minus_primary_spectral": scores.pair("spectral@profile", "spectral", "logistic")["difference"],
        "record_effect": effect,
        "levels": levels,
        "m4": [row for a in LEARNED for b in FROZEN[:2] for row in both(a, b)],
        "m5": [row for arm in (*LEARNED, *FROZEN) for row in both(arm, f"{arm}@raw")],
        "m6": [
            row for arm in LEARNED for count in settings["level_atoms"] for row in both(f"{arm}@K{count}", arm, first)
        ],
        "m7": [
            *both("spectral@profile", "spectral"),
            *both("SF-frozen-data@profile", "SF-frozen-data", first),
            *both("S1-L2@profile-raw", "S1-L2@raw", first),
        ],
        "m8": both("S1-L2@10min", "S1-L2", first),
        "m9": both("S1-L2@half-in", "S1-L2@half-out"),
        "m11": [row for arm in ("S1-L2", *FROZEN) for row in both(f"{arm}@{wide}", arm, first)],
    }


def evaluate(numbers: dict[str, Any]) -> dict[str, Any]:
    """Apply the rules of the protocol to measured numbers; no data is read.

    Returns
    -------
    dict
        ``h``; ``margin`` (the larger of 0.02 and ``h``); ``resolution``; the
        predictions ``S1`` to ``S7`` as booleans; ``defaults`` (does a level
        replace the primary value: only if it is above the primary by more than
        ``h`` on both heads); and ``no_waveform_claim`` (the classes whose
        ``displaced`` or ``context`` readout is within the margin of the
        ``full`` readout for the best arm).
    """
    h = float(numbers["h"])
    margin = max(0.02, h)
    effect = numbers["record_effect"]
    displaced = {row["class"]: row["difference"] for row in effect if row["against"] == "displaced"}
    within = sorted({row["class"] for row in effect if row["difference"] <= margin})
    return {
        "h": h,
        "margin": margin,
        "resolution": "described comparison" if h > 0.05 else "ranking possible",
        "S1": 0.02 <= h <= 0.05,
        "S2": numbers["learned_minus_frozen"] > h,
        "S3": numbers["bag_minus_learned"] >= -h,
        "S4": numbers["best_arm"] == "spectral",
        "S5": all(displaced[name] <= RECORD_LIMIT for name in EPILEPTIFORM),
        "S6": numbers["cosine_minus_raw"] > h,
        "S7": abs(numbers["profile_minus_primary_spectral"]) < h,
        "defaults": {
            level: len(rows) == len(HEADS) and all(row["difference"] > h for row in rows)
            for level, rows in numbers["levels"].items()
        },
        "no_waveform_claim": within,
    }


def decide(data: RunData, n_draws: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """The numbers of step 3 and the outcome of every rule; ``None`` when the run holds no learned arm."""
    numbers = measure(data, n_draws, seed)
    if numbers is None:
        return None
    return {"numbers": numbers, "outcomes": evaluate(numbers)}


def _pairs(rows: list[dict[str, Any]]) -> list[str]:
    return _table(
        ["first", "second", "head", "difference", "interval", "half"],
        [
            [
                row["first"],
                row["second"],
                row["head"],
                f"{row['difference']:+.3f}",
                f"{row['low']:+.3f} to {row['high']:+.3f}",
                f"{row['half_width']:.3f}",
            ]
            for row in rows
        ],
    )


def render_decisions(decisions: dict[str, Any]) -> str:
    """The reads of step 3 as text."""
    numbers, outcomes = decisions["numbers"], decisions["outcomes"]
    h, m1 = outcomes["h"], numbers["m1"]
    state = {True: "holds", False: "does not hold"}
    lines = [
        "Step 3: measures, predictions and rules",
        "",
        f"M1  h = {h:.3f}: S1-L2 minus S1-frozen-data = {m1['difference']:+.3f} ({m1['low']:+.3f} to {m1['high']:+.3f})",
        f"    margin of a useful gain = {outcomes['margin']:.3f}; resolution: {outcomes['resolution']}",
        "",
        "Primary scores (full readout): logistic, forest",
    ]
    lines += _table(
        ["arm", "logistic", "forest"],
        [
            [arm, f"{value:.3f}", f"{numbers['forest_scores'][arm]:.3f}"]
            for arm, value in sorted(numbers["primary_scores"].items(), key=lambda item: -item[1])
        ],
    )
    texts = {
        "S1": "h lies between 0.02 and 0.05",
        "S2": f"S1-L2 is above S1-frozen-data by more than h ({numbers['learned_minus_frozen']:+.3f})",
        "S3": f"spectra_bag is within h of S1-L2, or above it ({numbers['bag_minus_learned']:+.3f})",
        "S4": f"spectral has the highest primary score (the highest is {numbers['best_arm']})",
        "S5": f"gped and pled: displaced within 0.10 of full for the best arm ({numbers['best_arm']})",
        "S6": f"cosine tokens are above raw tokens by more than h for S1-L2 ({numbers['cosine_minus_raw']:+.3f})",
        "S7": f"the input level changes spectral by less than h ({numbers['profile_minus_primary_spectral']:+.3f})",
    }
    lines += ["", "Predictions"]
    lines += [f"  {name}  {state[bool(outcomes[name])]:<13}  {text}" for name, text in texts.items()]
    lines += ["", "Defaults of stage 4: does a level replace the primary value (above by more than h on both heads)?"]
    for level, replaces in outcomes["defaults"].items():
        found = ", ".join(f"{row['head']} {row['difference']:+.3f}" for row in numbers["levels"][level])
        lines.append(f"  {level:<14} {'yes' if replaces else 'no ':<3}  {found}")
    lines += ["", f"Record effect of the best arm ({numbers['best_arm']}), logistic head: full minus the other readout"]
    lines += _table(
        ["class", "against", "difference", "interval"],
        [
            [row["class"], row["against"], f"{row['difference']:+.3f}", f"{row['low']:+.3f} to {row['high']:+.3f}"]
            for row in numbers["record_effect"]
        ],
    )
    lines.append(f"No waveform claim (within the margin): {', '.join(outcomes['no_waveform_claim']) or 'none'}")
    titles = {
        "m4": "M4  learned against frozen",
        "m5": "M5  cosine against raw tokens on one dictionary",
        "m6": "M6  number of atoms against 32",
        "m7": "M7  profile input against primary input",
        "m8": "M8  10 against 5 minutes per patient",
        "m9": "M9  head on the dictionary half against head on the other half",
        "m11": "M11 wide tiling against the primary tiling",
    }
    for key, title in titles.items():
        lines += ["", title, *_pairs(numbers[key])]
    return "\n".join(lines) + "\n"
