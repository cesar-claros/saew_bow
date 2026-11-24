"""Audit of the preprocessed signals of a TUEV manifest; execute on the HPC.

The manifest audit proves that the examples follow the annotation files. This
script reads signals and asks three more things of the training set:

1. Values. Are the samples finite, on the expected scale, and seldom clipped?
2. Band. Is the power inside the configured passband?
3. Events against signal. Do the marked channels of an event carry more energy
   than the other channels in the marked second, and not in the other seconds
   or in the displaced control of the same recording? This holds only when the
   annotation times and the channel order agree with the signal.

Only training patients are read: each fold's validation partition, which
together hold every training patient once. The evaluation set stays closed.

The thresholds of the verdicts are fixed in this file before any run:

- the median share of power above the low-pass edge plus 10 Hz is at most 0.01;
- the median per-channel standard deviation of background contexts is between
  0.2 and 5 units of the configured normalization (checked only with "mad" or
  "zscore" normalization of the recording);
- for eye movement and for artifact, the mean marked-channel AUROC in the marked
  second is at least 0.60 and at least 0.05 above that of the displaced control.

Run from the repository root with ``PYTHONPATH=src``. The numeric helpers need
NumPy only; the package and Torch are imported when signals are read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

LABELS = ("spsw", "gped", "pled", "eyem", "artf", "bckg")
HIGH_BAND_MARGIN_HZ = 10.0
HIGH_BAND_SHARE_MAX = 0.01
BACKGROUND_STD_RANGE = (0.2, 5.0)
MARKED_AUROC_MIN = 0.60
MARKED_AUROC_GAIN_MIN = 0.05
CHANNEL_LOCAL_CLASSES = ("eyem", "artf")


def channel_auroc(energy: NDArray[np.float64], marked: NDArray[np.bool_]) -> float:
    """Probability that a marked channel has more energy than another channel (ties count one half).

    Returns NaN when the marked channels or the other channels are empty.
    """
    inside, outside = energy[marked], energy[~marked]
    if inside.size == 0 or outside.size == 0:
        return float("nan")
    greater = (inside[:, None] > outside[None, :]).mean()
    equal = (inside[:, None] == outside[None, :]).mean()
    return float(greater + 0.5 * equal)


def second_energies(signal: NDArray[np.floating[Any]], samples_per_second: int) -> NDArray[np.float64]:
    """Mean squared value per channel and whole second, shape ``(channels, seconds)``; a rest is dropped."""
    seconds = signal.shape[-1] // samples_per_second
    blocks = np.asarray(signal[:, : seconds * samples_per_second], dtype=np.float64)
    return (blocks.reshape(signal.shape[0], seconds, samples_per_second) ** 2).mean(axis=-1)


def band_shares(signal: NDArray[np.floating[Any]], sfreq: float, edges: tuple[float, ...]) -> list[float]:
    """Shares of the power of all channels below, between and above the ascending ``edges`` in Hz.

    A Hann window limits leakage. The constant term belongs to the lowest band.
    Returns NaNs for a signal without power.
    """
    window = np.hanning(signal.shape[-1])
    power = np.abs(np.fft.rfft(np.asarray(signal, dtype=np.float64) * window, axis=-1)) ** 2
    frequency = np.fft.rfftfreq(signal.shape[-1], 1.0 / sfreq)
    total = float(power.sum())
    if total <= 0:
        return [float("nan")] * (len(edges) + 1)
    bounds = (-np.inf, *edges, np.inf)
    return [
        float(power[:, (frequency >= low) & (frequency < high)].sum() / total) for low, high in zip(bounds, bounds[1:])
    ]


def _finite(values: Any, reduce: Any) -> float:
    """``reduce`` of the finite values, or NaN when there is none."""
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    return float(reduce(finite)) if finite.size else float("nan")


def summary(values: list[float]) -> dict[str, float | int]:
    """Count, median, 5th and 95th percentile and mean of the finite values."""
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0, "median": float("nan"), "p05": float("nan"), "p95": float("nan"), "mean": float("nan")}
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "mean": float(finite.mean()),
    }


def example_measures(
    signal: NDArray[np.floating[Any]],
    valid_channels: NDArray[np.bool_],
    marked_channels: tuple[int, ...],
    labeled_second: int,
    sfreq: float,
    clip: float | None,
    edges: tuple[float, ...],
) -> dict[str, Any]:
    """The measures of one context ``(channels, samples)``.

    ``marked_channels`` are positions in the signal. Flat channels take no part
    in the channel statistics. ``marked_auroc`` has one value per whole second of
    the context; ``second_ratio`` is the base-10 log of the energy of the labeled
    second over the mean energy of the other seconds, over all usable channels.
    """
    usable = np.asarray(valid_channels, dtype=bool)
    energies = second_energies(signal, round(sfreq))
    marked = np.zeros(signal.shape[0], dtype=bool)
    marked[list(marked_channels)] = True
    aurocs = [channel_auroc(energies[usable, second], marked[usable]) for second in range(energies.shape[1])]
    montage = energies[usable].mean(axis=0) if usable.any() else np.full(energies.shape[1], np.nan)
    others = np.delete(montage, labeled_second)
    comparable = others.size > 0 and others.mean() > 0 and montage[labeled_second] > 0
    ratio = float(np.log10(montage[labeled_second] / others.mean())) if comparable else float("nan")
    values = np.asarray(signal, dtype=np.float64)
    return {
        "finite": bool(np.isfinite(values).all()),
        "clipped_share": float((np.abs(values) >= clip - 1e-6).mean()) if clip is not None else float("nan"),
        "channel_std": float(np.median(values[usable].std(axis=-1))) if usable.any() else float("nan"),
        "abs_max": float(np.abs(values).max()),
        "band_shares": band_shares(values[usable], sfreq, edges) if usable.any() else [float("nan")] * (len(edges) + 1),
        "marked_auroc": aurocs,
        "second_ratio": ratio,
        "labeled_energy": energies[:, labeled_second].tolist(),
    }


def select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """At most ``limit`` rows per class, those with the smallest example IDs; ``limit <= 0`` keeps all."""
    if limit <= 0:
        return rows
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["example_id"]):
        if len(by_class[row["label"]]) < limit:
            by_class[row["label"]].append(row)
    return [row for label in sorted(by_class) for row in by_class[label]]


def collect(root: Path, manifest_dir: Path, cache_dir: Path | None, limit: int) -> dict[str, Any]:
    """Read the selected training events and their displaced controls, and gather their measures by class."""
    from eeg_compare.data.dataset import Partition, ReaderOptions, TUEVDataset

    options = ReaderOptions(cache_dir=cache_dir)
    first = TUEVDataset(root, manifest_dir, Partition("validation", 0), options)
    config = first.manifest.config
    sfreq = config.preprocessing.target_sfreq
    if not float(config.examples.before_seconds).is_integer() or not float(sfreq).is_integer():
        raise ValueError("the signal audit needs a whole number of seconds before the onset and a whole sampling rate")
    labeled_second = round(config.examples.before_seconds)
    low_pass = config.preprocessing.lowpass_hz
    high_pass = config.preprocessing.highpass_hz
    edges = tuple(
        edge
        for edge in (high_pass, low_pass, None if low_pass is None else low_pass + HIGH_BAND_MARGIN_HZ)
        if edge is not None
    )
    channels = list(config.preprocessing.channels)
    from eeg_compare.data.annotations import MONTAGE

    position = {MONTAGE.index(name): index for index, name in enumerate(channels)}
    datasets: dict[int, tuple[Any, Any, dict[str, int]]] = {}
    rows: list[dict[str, Any]] = []
    for fold in range(config.splits.n_folds):
        events = first if fold == 0 else TUEVDataset(root, manifest_dir, Partition("validation", fold), options)
        try:
            controls: TUEVDataset | None = TUEVDataset(
                root, manifest_dir, Partition("validation", fold, kind="displaced"), options
            )
        except ValueError:
            controls = None
        control_index = (
            {source: index for index, source in enumerate(controls.examples["source_example_id"].to_list())}
            if controls is not None
            else {}
        )
        datasets[fold] = (events, controls, control_index)
        for index, (example_id, label) in enumerate(
            zip(events.examples["example_id"].to_list(), events.examples["label"].to_list())
        ):
            rows.append({"fold": fold, "index": index, "example_id": example_id, "label": label})
    chosen = sorted(select(rows, limit), key=lambda row: (row["fold"], row["index"]))
    measures: dict[str, dict[str, list[Any]]] = {label: defaultdict(list) for label in LABELS}
    for row in chosen:
        events, controls, control_index = datasets[row["fold"]]
        meta = events.example_metadata(row["index"])
        marked = tuple(position[channel] for channel in meta.annotation_channels if channel in position)
        loaded = events.reader.example(meta)
        result = example_measures(
            loaded.signal, loaded.valid_channels, marked, labeled_second, sfreq, config.preprocessing.clip, edges
        )
        bucket = measures[meta.label]
        for key, value in result.items():
            bucket[key].append(value)
        if controls is not None and meta.example_id in control_index:
            control = events.reader.example(controls.example_metadata(control_index[meta.example_id]))
            shifted = example_measures(
                control.signal, control.valid_channels, marked, labeled_second, sfreq, config.preprocessing.clip, edges
            )
            bucket["control_auroc"].append(shifted["marked_auroc"][labeled_second])
            bucket["control_ratio"].append(shifted["second_ratio"])
            bucket["control_std"].append(shifted["channel_std"])
    return {
        "measures": measures,
        "channels": channels,
        "edges": list(edges),
        "labeled_second": labeled_second,
        "normalization": [config.preprocessing.normalization, config.preprocessing.normalization_scope],
        "clip": config.preprocessing.clip,
        "low_pass": low_pass,
        "selected": len(chosen),
        "available": len(rows),
    }


def build_report(collected: dict[str, Any]) -> dict[str, Any]:
    """Class summaries and the verdicts of the fixed thresholds."""
    measures, channels = collected["measures"], collected["channels"]
    second = collected["labeled_second"]
    classes: dict[str, Any] = {}
    for label in LABELS:
        bucket = measures[label]
        count = len(bucket["finite"])
        if not count:
            classes[label] = {"n": 0}
            continue
        aurocs = np.asarray(bucket["marked_auroc"], dtype=np.float64)
        shares = np.asarray(bucket["band_shares"], dtype=np.float64)
        energy = np.asarray(bucket["labeled_energy"], dtype=np.float64).mean(axis=0)
        order = np.argsort(energy)[::-1][:4]
        classes[label] = {
            "n": count,
            "all_finite": bool(all(bucket["finite"])),
            "clipped_share": summary(bucket["clipped_share"]),
            "examples_with_a_clipped_sample": int(sum(value > 0 for value in bucket["clipped_share"])),
            "channel_std": summary(bucket["channel_std"]),
            "abs_max": summary(bucket["abs_max"]),
            "band_share_medians": [_finite(shares[:, band], np.median) for band in range(shares.shape[1])],
            "marked_auroc_by_second": [_finite(aurocs[:, index], np.mean) for index in range(aurocs.shape[1])],
            "examples_with_channel_contrast": int(np.isfinite(aurocs[:, second]).sum()),
            "control_marked_auroc": summary(bucket["control_auroc"]),
            "second_ratio": summary(bucket["second_ratio"]),
            "control_second_ratio": summary(bucket["control_ratio"]),
            "control_channel_std": summary(bucket["control_std"]),
            "channels_with_most_energy": [channels[index] for index in order],
        }
    checks = []
    all_shares = [row[-1] for label in LABELS for row in measures[label]["band_shares"]]
    if collected["low_pass"] is not None and all_shares:
        value = _finite(all_shares, np.median)
        checks.append(
            {
                "check": f"median share of power above {collected['low_pass'] + HIGH_BAND_MARGIN_HZ:g} Hz",
                "value": value,
                "rule": f"at most {HIGH_BAND_SHARE_MAX}",
                "ok": bool(value <= HIGH_BAND_SHARE_MAX),
            }
        )
    if collected["normalization"][0] in ("mad", "zscore") and collected["normalization"][1] == "recording":
        value = classes["bckg"].get("channel_std", {}).get("median", float("nan"))
        low, high = BACKGROUND_STD_RANGE
        checks.append(
            {
                "check": "median per-channel standard deviation of background contexts",
                "value": value,
                "rule": f"between {low} and {high}",
                "ok": bool(low <= value <= high),
            }
        )
    for label in CHANNEL_LOCAL_CLASSES:
        if not classes[label].get("n"):
            continue
        event = classes[label]["marked_auroc_by_second"][second]
        if classes[label]["control_marked_auroc"]["n"]:
            gain = event - classes[label]["control_marked_auroc"]["mean"]
            checks.append(
                {
                    "check": f"{label}: marked-channel AUROC in the marked second, and its gain over the displaced control",
                    "value": [event, gain],
                    "rule": f"at least {MARKED_AUROC_MIN}, and a gain of at least {MARKED_AUROC_GAIN_MIN}",
                    "ok": bool(event >= MARKED_AUROC_MIN and gain >= MARKED_AUROC_GAIN_MIN),
                }
            )
        else:
            checks.append(
                {
                    "check": f"{label}: marked-channel AUROC in the marked second (the manifest has no displaced control)",
                    "value": [event],
                    "rule": f"at least {MARKED_AUROC_MIN}",
                    "ok": bool(event >= MARKED_AUROC_MIN),
                }
            )
    finite = all(classes[label].get("all_finite", True) for label in LABELS)
    checks.append({"check": "all samples are finite", "value": finite, "rule": "true", "ok": finite})
    return {
        "scope": "Training patients only (the validation partition of every fold); the evaluation set is not read.",
        "selected_examples": collected["selected"],
        "available_examples": collected["available"],
        "band_edges_hz": collected["edges"],
        "labeled_second": second,
        "normalization": collected["normalization"],
        "clip": collected["clip"],
        "classes": classes,
        "checks": checks,
        "ok": all(check["ok"] for check in checks),
    }


def render(report: dict[str, Any]) -> str:
    """A text view of the report for the terminal."""
    edges = report["band_edges_hz"]
    bands = ["all"]
    if edges:
        bands = [f"below {edges[0]:g}", *[f"{low:g}-{high:g}" for low, high in zip(edges, edges[1:])]]
        bands.append(f"above {edges[-1]:g}")
    lines = [
        report["scope"],
        f"examples read: {report['selected_examples']} of {report['available_examples']} | "
        f"normalization {report['normalization']} | clip {report['clip']} | bands in Hz: {bands}",
        "",
    ]
    for label in LABELS:
        data = report["classes"][label]
        if not data["n"]:
            lines.append(f"{label}: no example")
            continue
        aurocs = " ".join(f"{value:.2f}" for value in data["marked_auroc_by_second"])
        lines.extend(
            [
                f"{label}: n {data['n']} | per-channel std median {data['channel_std']['median']:.2f} "
                f"(control {data['control_channel_std']['median']:.2f}) | largest absolute value median "
                f"{data['abs_max']['median']:.1f} | clipped share mean {data['clipped_share']['mean']:.5f} in "
                f"{data['examples_with_a_clipped_sample']} examples",
                "  band shares (median): " + " ".join(f"{value:.4f}" for value in data["band_share_medians"]),
                f"  marked-channel AUROC by second: {aurocs} | marked second is number {report['labeled_second'] + 1} | "
                f"control {data['control_marked_auroc']['mean']:.2f} | examples with a contrast "
                f"{data['examples_with_channel_contrast']}",
                f"  log10 energy of the marked second over the other seconds: median "
                f"{data['second_ratio']['median']:+.2f} (control {data['control_second_ratio']['median']:+.2f})",
                f"  channels with most energy in the marked second: {data['channels_with_most_energy']}",
            ]
        )
    lines.append("")
    for check in report["checks"]:
        lines.append(f"{'PASS' if check['ok'] else 'FAIL'}: {check['check']}: {check['value']} ({check['rule']})")
    lines.append("signal audit: " + ("all checks pass" if report["ok"] else "at least one check fails"))
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    """The report with non-finite numbers as ``None``, so the file is strict JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    """Write the JSON report and print its text view; return 0 when all checks pass, 1 otherwise, 2 on an error."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Folder that holds the official directories")
    parser.add_argument("--manifest", type=Path, required=True, help="A completed manifest directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report, outside the manifest")
    parser.add_argument("--cache-dir", type=Path, help="Recording cache of the reader")
    parser.add_argument("--limit-per-class", type=int, default=300, help="Events per class; 0 reads all")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.resolve().is_relative_to(args.manifest.resolve()):
            raise ValueError("the output must be a new file outside the manifest directory")
        report = build_report(collect(args.root, args.manifest, args.cache_dir, args.limit_per_class))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(report), indent=2) + "\n")
    except (OSError, ValueError, KeyError) as error:
        sys.stderr.write(f"Signal audit failed: {error}\n")
        return 2
    sys.stdout.write(render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
