"""Audit of the recording scales of a TUEV manifest; execute on the HPC.

The signal contract divides every channel of a recording by its own robust
scale (median and MAD over the recording). The alternative is one recording
scale times a channel profile that is the same for all recordings. This script
measures, on the training recordings only, whether the two differ and whether
events enter the scale. It fits no model. The evaluation set stays closed.

Three measures:

1. Profile. The log scale of channel c in recording r is split into a recording
   part, a channel part (the fixed profile) and a rest. The rest is what a scale
   per channel removes and a fixed profile keeps.
2. Events in the scale. For a channel with marks of a class, the scale from all
   samples is divided by the scale from the samples that are at least
   ``--gap-seconds`` from every mark of the recording. The channels of the same
   recordings without any mark are the control.
3. Loud channels. The share of channels whose scale is more than 3 times, or
   less than a third of, what the recording part and the profile give. Without a
   scale per channel, such a channel needs a step of its own.

The rules are fixed in this file before any run:

- measure 1 passes when the standard deviation of the rest is at most 0.10
  (natural log units, about 10% of the scale);
- measure 2 passes when the median ratio is at most 1.5 for every class with at
  least 20 marked channels (a ratio above 1.5 means that the scale per channel
  removes a third of the amplitude of that class);
- outcome "per_channel": measure 1 passes. The scale per channel and the fixed
  profile are the same in practice.
- outcome "per_channel_with_profile_level": measure 1 fails and measure 2
  passes. The scale per channel stays primary; the fixed profile is a declared
  second level for the full and the rank-1 cells.
- outcome "profile_primary": both fail. The fixed profile becomes primary, with
  a cap on loud channels.

Limits. Measure 2 uses the class marks of the training patients for a decision
of the design. The marks of TUEV cover a part of the events, so the samples far
from every mark can still hold events; the ratio is then too small. The mean
rest on the marked channels of a class is reported beside it as a second view.

The script reads and filters every training recording again, without the
normalization, so it takes some minutes. Run from the repository root with
``PYTHONPATH=src``. The numeric helpers need NumPy only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

LABELS = ("spsw", "gped", "pled", "eyem", "artf", "bckg")
REST_SD_MAX = 0.10
EVENT_RATIO_MAX = 1.5
LOUD_FACTOR = 3.0
MIN_UNITS = 20
GAP_SECONDS = 10.0
MIN_FAR_SECONDS = 30.0


def channel_scales(data: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """The scale of the pipeline for every channel of ``(channels, samples)`` data.

    It is 1.4826 times the lower median of the absolute deviations from the lower
    median, with a floor of 1% of the standard deviation and of 1e-12, as in
    ``eeg_compare.data.recordings.normalize_signal``.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("the scale needs (channels, nonempty samples) data")
    middle = (values.shape[-1] - 1) // 2
    centered = values - np.partition(values, middle, axis=-1)[:, middle : middle + 1]
    mad = np.partition(np.abs(centered), middle, axis=-1)[:, middle] * 1.4826
    return np.maximum(mad, 0.01 * centered.std(axis=-1)).clip(1e-12)


def far_mask(n_samples: int, sfreq: float, intervals: list[tuple[float, float]], gap: float) -> NDArray[np.bool_]:
    """The samples that are at least ``gap`` seconds from every interval ``(start, stop)`` in seconds."""
    near = np.zeros(n_samples, dtype=bool)
    for start, stop in intervals:
        first = max(int(math.floor((start - gap) * sfreq)), 0)
        last = min(int(math.ceil((stop + gap) * sfreq)), n_samples)
        near[first:last] = True
    return ~near


def two_way(log_scale: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split ``(recordings, channels)`` log scales into a recording part, a channel part and a rest.

    The recording part is the mean over the channels. The channel part is the mean
    over the recordings of what remains; its own mean is zero. The rest has a zero
    mean in every row and every column.
    """
    level = log_scale.mean(axis=1)
    profile = (log_scale - level[:, None]).mean(axis=0)
    return level, profile, log_scale - level[:, None] - profile[None, :]


def quartiles(values: list[float]) -> dict[str, float | int]:
    """Count, median and quartiles of the values; NaN for an empty list."""
    if not values:
        return {"n": 0, "q25": float("nan"), "median": float("nan"), "q75": float("nan")}
    q25, median, q75 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])
    return {"n": len(values), "q25": float(q25), "median": float(median), "q75": float(q75)}


def collect(root: Path, manifest_dir: Path, limit: int, gap: float, min_far_seconds: float) -> dict[str, Any]:
    """Read every training recording without the normalization and gather its scales and its marked channels.

    A row holds the scale of every channel from all samples, the scale from the
    samples far from every mark (``None`` when fewer than ``min_far_seconds`` are
    far), and the channels that carry a mark of each class. A cluster is a number,
    so the report holds no patient identifier.
    """
    from dataclasses import replace

    from eeg_compare.data.annotations import CLASSES, MONTAGE
    from eeg_compare.data.dataset import TUEVReader
    from eeg_compare.data.manifest import _read_annotations, load_manifest
    from eeg_compare.data.recordings import read_recording

    manifest = load_manifest(manifest_dir)
    TUEVReader(root, manifest)  # refuses a manifest of another configuration or runtime
    config = manifest.config.preprocessing
    unscaled = replace(config, normalization="none", clip=None)
    sfreq = float(config.target_sfreq)
    position = {MONTAGE.index(name): index for index, name in enumerate(config.channels)}
    training = sorted(
        (r for r in manifest.recordings if r.official_split == "train" and not r.missing_channels),
        key=lambda recording: recording.recording_id,
    )
    chosen = training[:limit] if limit > 0 else training
    clusters = {name: index for index, name in enumerate(sorted({r.cluster_id for r in training}))}
    rows: list[dict[str, Any]] = []
    for number, recording in enumerate(chosen, start=1):
        data = np.asarray(read_recording(root, recording, unscaled), dtype=np.float64)
        annotations = _read_annotations(root, recording)
        intervals = [(row.start_seconds, row.stop_seconds) for row in annotations]
        far = far_mask(data.shape[1], sfreq, intervals, gap)
        marked: dict[str, set[int]] = {}
        for row in annotations:
            if row.channel in position:
                marked.setdefault(CLASSES[row.class_code], set()).add(position[row.channel])
        enough = int(far.sum()) >= min_far_seconds * sfreq
        rows.append(
            {
                "recording_id": recording.recording_id,
                "cluster": clusters[recording.cluster_id],
                "seconds": data.shape[1] / sfreq,
                "far_seconds": int(far.sum()) / sfreq,
                "scale": channel_scales(data).tolist(),
                "far_scale": channel_scales(data[:, far]).tolist() if enough else None,
                "marked": {label: sorted(channels) for label, channels in marked.items()},
            }
        )
        if number % 25 == 0 or number == len(chosen):
            sys.stderr.write(f"scale audit: {number} of {len(chosen)} recordings read\n")
    return {
        "rows": rows,
        "channels": list(config.channels),
        "normalization": [config.normalization, config.normalization_scope],
        "band_hz": [config.highpass_hz, config.lowpass_hz],
        "gap_seconds": gap,
        "min_far_seconds": min_far_seconds,
        "available": len(training),
    }


def build_report(collected: dict[str, Any], min_units: int = MIN_UNITS) -> dict[str, Any]:
    """Apply the fixed rules to the collected scales.

    Parameters
    ----------
    collected : dict[str, Any]
        The result of ``collect``.
    min_units : int
        Marked channels that a class needs for its rule of measure 2.

    Returns
    -------
    dict[str, Any]
        The three measures, the outcome, and the scale of every channel of every
        recording (for a later fit of the profile).
    """
    rows = collected["rows"]
    channels = collected["channels"]
    if len(rows) < 2:
        raise ValueError("the scale audit needs at least two recordings")
    scale = np.array([row["scale"] for row in rows], dtype=np.float64)
    level, profile, rest = two_way(np.log(scale))
    rest_sd = float(rest.std())
    spread = float(1.4826 * np.median(np.abs(rest - np.median(rest))))
    variance = {
        "recording": float(level.var()),
        "channel": float(profile.var()),
        "rest": float(rest.var()),
    }
    profile_passes = rest_sd <= REST_SD_MAX

    classes: dict[str, Any] = {}
    for label in LABELS:
        ratios, rests, controls = [], [], []
        recordings, clusters = set(), set()
        for index, row in enumerate(rows):
            marked = row["marked"].get(label, [])
            if not marked:
                continue
            rests.extend(float(rest[index, channel]) for channel in marked)
            if row["far_scale"] is None:
                continue
            recordings.add(index)
            clusters.add(row["cluster"])
            any_mark = {channel for found in row["marked"].values() for channel in found}
            for channel in range(len(channels)):
                ratio = row["scale"][channel] / row["far_scale"][channel]
                if channel in marked:
                    ratios.append(ratio)
                elif channel not in any_mark:
                    controls.append(ratio)
        summary = quartiles(ratios)
        evaluated = summary["n"] >= min_units
        classes[label] = {
            "marked_channels": summary["n"],
            "recordings": len(recordings),
            "clusters": len(clusters),
            "ratio_all_to_far": summary,
            "control_ratio_on_unmarked_channels": quartiles(controls),
            "mean_rest_on_marked_channels": float(np.mean(rests)) if rests else float("nan"),
            "scale_factor_of_that_rest": float(np.exp(np.mean(rests))) if rests else float("nan"),
            "evaluated": bool(evaluated),
            "passes": bool(summary["median"] <= EVENT_RATIO_MAX) if evaluated else None,
        }
    decided = [entry["passes"] for entry in classes.values() if entry["evaluated"]]
    events_pass = bool(decided) and all(decided)

    loud = rest > math.log(LOUD_FACTOR)
    quiet = rest < -math.log(LOUD_FACTOR)
    if profile_passes:
        outcome = "per_channel"
    elif events_pass:
        outcome = "per_channel_with_profile_level"
    elif decided:
        outcome = "profile_primary"
    else:
        outcome = "undecided_no_class_with_enough_marked_channels"
    return {
        "question": "one scale per channel and recording, or one recording scale times a fixed channel profile",
        "recordings_read": len(rows),
        "recordings_available": collected["available"],
        "normalization_of_the_manifest": collected["normalization"],
        "band_hz": collected["band_hz"],
        "gap_seconds": collected["gap_seconds"],
        "thresholds": {
            "rest_sd_max": REST_SD_MAX,
            "event_ratio_max": EVENT_RATIO_MAX,
            "loud_factor": LOUD_FACTOR,
            "min_marked_channels": min_units,
        },
        "measure_1_profile": {
            "rest_sd": rest_sd,
            "rest_robust_sd": spread,
            "variance_of_the_log_scale": variance,
            "rest_sd_by_channel": dict(zip(channels, rest.std(axis=0).tolist())),
            "profile_factor_by_channel": dict(zip(channels, np.exp(profile).tolist())),
            "recording_scale_quartiles_volts": quartiles(np.exp(level).tolist()),
            "passes": bool(profile_passes),
        },
        "measure_2_events": {"classes": classes, "passes": events_pass if decided else None},
        "measure_3_loud_channels": {
            "share_of_channels_above": float(loud.mean()),
            "share_of_channels_below": float(quiet.mean()),
            "share_of_recordings_with_one": float((loud | quiet).any(axis=1).mean()),
            "channels_above_by_channel": dict(zip(channels, loud.sum(axis=0).astype(int).tolist())),
        },
        "outcome": outcome,
        "scales": [
            {"recording_id": row["recording_id"], "cluster": row["cluster"], "scale": row["scale"]} for row in rows
        ],
    }


def render(report: dict[str, Any]) -> str:
    """The text view of a report."""
    first, third = report["measure_1_profile"], report["measure_3_loud_channels"]
    variance = first["variance_of_the_log_scale"]
    lines = [
        f"Scale audit of {report['recordings_read']} of {report['recordings_available']} training recordings",
        "Measure 1, profile",
        f"  standard deviation of the rest: {first['rest_sd']:.3f} (robust {first['rest_robust_sd']:.3f}); "
        f"at most {report['thresholds']['rest_sd_max']} passes: {'PASS' if first['passes'] else 'FAIL'}",
        f"  variance of the log scale: recording {variance['recording']:.3f}, channel {variance['channel']:.3f}, "
        f"rest {variance['rest']:.3f}",
        "Measure 2, events in the scale: median ratio of the scale from all samples to the scale far from marks",
    ]
    for label, entry in report["measure_2_events"]["classes"].items():
        verdict = "not evaluated" if not entry["evaluated"] else ("PASS" if entry["passes"] else "FAIL")
        lines.append(
            f"  {label}: {entry['ratio_all_to_far']['median']:.3f} on {entry['marked_channels']} marked channels of "
            f"{entry['recordings']} recordings; control {entry['control_ratio_on_unmarked_channels']['median']:.3f}; "
            f"rest on marked channels x{entry['scale_factor_of_that_rest']:.2f}; {verdict}"
        )
    lines += [
        "Measure 3, loud channels",
        f"  channels above {report['thresholds']['loud_factor']} times the profile: "
        f"{third['share_of_channels_above']:.4f}; below a third: {third['share_of_channels_below']:.4f}; "
        f"recordings with one: {third['share_of_recordings_with_one']:.3f}",
        f"Outcome: {report['outcome']}",
    ]
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    """NaN and infinity as ``None``, so that the report is strict JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    """Write the JSON report and print its text view; return 0 when the audit ran, 2 on an error."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Folder that holds the official directories")
    parser.add_argument("--manifest", type=Path, required=True, help="A completed manifest directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report, outside the manifest")
    parser.add_argument("--limit", type=int, default=0, help="Recordings to read; 0 reads all")
    parser.add_argument("--gap-seconds", type=float, default=GAP_SECONDS, help="Distance from every mark")
    parser.add_argument("--min-far-seconds", type=float, default=MIN_FAR_SECONDS, help="Far samples that a scale needs")
    parser.add_argument("--min-units", type=int, default=MIN_UNITS, help="Marked channels that a class needs")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.resolve().is_relative_to(args.manifest.resolve()):
            raise ValueError("the output must be a new file outside the manifest directory")
        collected = collect(args.root, args.manifest, args.limit, args.gap_seconds, args.min_far_seconds)
        report = build_report(collected, args.min_units)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(report), indent=2) + "\n")
    except (OSError, ValueError, KeyError) as error:
        sys.stderr.write(f"Scale audit failed: {error}\n")
        return 2
    sys.stdout.write(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
