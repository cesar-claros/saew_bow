"""Read-only audit of a built TUEV manifest: counts, conflicts, annotation agreement and coverage.

The script reads ``metadata.json`` and the example shards. It reads no signal and
changes nothing. It needs Polars and no other part of this package. Its report
is the source of the audit tables in ``docs/Comparison Plan.md``.

A conflict is an event interval that overlaps, in time, an interval of another
class on any channel of the same recording. The report shows which classes meet,
and what four candidate rules would keep:

- ``drop``: remove every conflicting interval.
- ``background_yields``: a background interval that meets another class is
  removed; an interval of another class is kept when background is all it meets.
- ``precedence``: an interval is kept when every class it meets ranks lower
  (epileptiform classes, then eye movement, then artifact, then background).
- ``keep``: keep every interval.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl

LABELS = ("spsw", "gped", "pled", "eyem", "artf", "bckg")
RANK = {"spsw": 3, "gped": 3, "pled": 3, "eyem": 2, "artf": 1, "bckg": 0}
RULES = ("drop", "background_yields", "precedence", "keep")
EDGE = "context_outside_recording"
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def keeps(rule: str, label: str, partners: set[str]) -> bool:
    """Whether a conflict rule keeps an interval of ``label`` that meets the classes in ``partners``."""
    if rule == "keep":
        return True
    if rule == "drop":
        return not partners
    if rule == "background_yields":
        return not partners if label == "bckg" else partners <= {"bckg"}
    if rule == "precedence":
        return all(RANK[other] < RANK[label] for other in partners)
    raise ValueError(f"unknown conflict rule: {rule}")


def load_events(directory: Path) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """All examples of the manifest, and its event examples with the classes that each one meets.

    Every event row gains ``partners`` (other classes that overlap its interval),
    ``exact`` (a partner has the identical interval) and ``cross_file`` (partners
    that come only from the other annotation file of a merged recording).

    Raises:
        ValueError: The stored conflict flag of an example differs from the overlaps found here.
    """
    examples = pl.read_parquet(sorted((directory / "examples").glob("*.parquet")))
    events = examples.filter(pl.col("kind") == "event").to_dicts()
    by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        by_recording[row["recording_id"]].append(row)
    for rows in by_recording.values():
        for row in rows:
            row["partners"], row["cross_file"], row["exact"] = set(), set(), False
            for other in rows:
                if other is row or other["label"] == row["label"]:
                    continue
                if (
                    other["annotation_start_seconds"] < row["annotation_stop_seconds"]
                    and other["annotation_stop_seconds"] > row["annotation_start_seconds"]
                ):
                    row["partners"].add(other["label"])
                    row["exact"] |= (other["annotation_start_seconds"], other["annotation_stop_seconds"]) == (
                        row["annotation_start_seconds"],
                        row["annotation_stop_seconds"],
                    )
                    if not set(row["annotation_sources"]) & set(other["annotation_sources"]):
                        row["cross_file"].add(other["label"])
            if bool(row["partners"]) != row["conflicting_annotation"]:
                raise ValueError(f"conflict flag of example {row['example_id']} differs from its overlaps")
    return examples, events


def _electrode(name: str) -> tuple[str, str]:
    """The canonical electrode name of an EDF label and its reference suffix, by the rule of the reader."""
    name = name.strip().upper()
    if name.startswith("EEG "):
        name = name[4:].strip()
    reference = ""
    for suffix in ("-REF", "-LE"):
        if name.endswith(suffix):
            name, reference = name[: -len(suffix)], suffix[1:]
            break
    return "-".join(ALIASES.get(part.strip(), part.strip()) for part in name.split("-")), reference


def table_checks(directory: Path, metadata: dict[str, Any], examples: pl.DataFrame) -> list[str]:
    """Timing of the contexts, distance of the controls and channel mapping, from the tables alone.

    The mapping check covers referential input: every row of a channel matrix
    must hold +1 on the first electrode of its derivation and -1 on the second,
    found by name, and 0 elsewhere.
    """
    rate = metadata["config"]["preprocessing"]["target_sfreq"]
    before = metadata["config"]["examples"]["before_seconds"]
    gap = metadata["config"]["examples"]["displaced_gap_seconds"]
    channels = metadata["config"]["preprocessing"]["channels"]
    recordings = pl.read_parquet(directory / "recordings.parquet")
    events = examples.filter(pl.col("kind") == "event")
    kept = examples.filter(pl.col("status") == "kept")
    starts = zip(events["annotation_start_seconds"].to_list(), events["start_sample"].to_list())
    moved = sum(math.floor((onset - before) * rate + 0.5) != start for onset, start in starts)
    length = {row["recording_id"]: math.ceil(row["n_samples"] * rate / row["sfreq"]) for row in recordings.to_dicts()}
    outside = sum(
        row["start_sample"] < 0 or row["start_sample"] + row["n_samples"] > length[row["recording_id"]]
        for row in kept.select("recording_id", "start_sample", "n_samples").to_dicts()
    )
    marks: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in events.select("recording_id", "annotation_start_seconds", "annotation_stop_seconds").to_dicts():
        marks[row["recording_id"]].append((row["annotation_start_seconds"], row["annotation_stop_seconds"]))
    distances = []
    controls = examples.filter(pl.col("kind") == "displaced").select("recording_id", "start_sample", "n_samples")
    for row in controls.to_dicts():
        low, high = row["start_sample"] / rate, (row["start_sample"] + row["n_samples"]) / rate
        distances.append(min(max(start - high, low - stop) for start, stop in marks[row["recording_id"]]))
    wrong = 0
    references: Counter[str] = Counter()
    for sources, matrix in zip(recordings["source_channels"].to_list(), recordings["channel_matrix"].to_list()):
        names = [_electrode(source) for source in sources]
        references.update({reference for _, reference in names})
        index = {name: position for position, (name, _) in enumerate(names)}
        for channel, row in zip(channels, matrix):
            first, second = channel.split("-")
            expected = [0.0] * len(sources)
            if first in index and second in index:
                expected[index[first]], expected[index[second]] = 1.0, -1.0
            wrong += list(row) != expected
    hours = float((recordings["n_samples"] / recordings["sfreq"]).sum()) / 3600
    offsets = events["onset_offset_samples"]
    padded = int(((kept["pad_left"] > 0) | (kept["pad_right"] > 0)).sum())
    lines = [
        "",
        "timing and channel mapping, from the tables",
        f"  contexts whose start differs from round((onset - {before:g} s) * {rate:g}): {moved} of {events.height}",
        f"  onset offset in samples: {offsets.min():.4f} to {offsets.max():.4f}",
        f"  kept examples with padding: {padded}; kept examples that leave their recording: {outside}",
    ]
    if distances:
        near = sum(distance < gap - 1e-9 for distance in distances)
        lines.append(
            f"  displaced controls closer than {gap:g} s to a mark: {near} of {len(distances)}; "
            f"smallest distance {min(distances):.3f} s"
        )
    rows = recordings.height * len(channels)
    rates = dict(Counter(recordings["sfreq"].to_list()))
    lines.append(f"  rows of the channel matrices that differ from +first -second electrode by name: {wrong} of {rows}")
    lines.append(f"  reference suffixes {dict(references)}; native sampling rates {rates}; signal hours {hours:.1f}")
    return lines


def _matrix(counter: Counter[tuple[str, str]], scale: float = 1.0) -> list[str]:
    lines = ["          " + " ".join(f"{label:>6s}" for label in LABELS)]
    for row in LABELS:
        lines.append(f"    {row:5s} " + " ".join(f"{counter.get((row, column), 0) / scale:6.0f}" for column in LABELS))
    return lines


def report(directory: Path) -> list[str]:
    """The lines of the audit report of one completed manifest."""
    metadata = json.loads((directory / "metadata.json").read_text())
    examples, events = load_events(directory)
    statistics = metadata["statistics"]
    lines = [
        f"manifest: {directory.name} | corpus {metadata['corpus_version']} | schema {metadata['schema_version']}",
        f"recordings {metadata['recording_count']} | clusters {metadata['cluster_count']} | "
        f"training patients {metadata['training_patient_count']} | fold sizes {metadata['fold_sizes']}",
        f"source annotation rows {statistics['source_annotation_rows']} | event intervals {len(events)} | "
        f"kept {statistics['kept']} | displaced controls {statistics.get('displaced', 0)}",
    ]
    durations = Counter(round(row["annotation_stop_seconds"] - row["annotation_start_seconds"], 3) for row in events)
    lines.append(f"interval durations in seconds: {dict(durations.most_common(4))}")
    rule = metadata["config"]["examples"]["conflict_policy"]
    if rule in RULES:
        others = {"conflicting_annotation", "background_with_other_class"}
        differing = sum(
            (row["status"] == "kept")
            != (not set(row["exclusion_reasons"]) - others and keeps(rule, row["label"], row["partners"]))
            for row in events
        )
        lines.append(f"conflict rule of the manifest: {rule}; kept events that differ from this audit: {differing}")

    lines.extend(table_checks(directory, metadata, examples))

    lines.append("")
    lines.append(
        "intervals by split and class: all, outside the recording, conflicting, rows and channels per interval"
    )
    for split in ("train", "eval"):
        for label in LABELS:
            rows = [row for row in events if row["official_split"] == split and row["label"] == label]
            lines.append(
                f"  {split:5s} {label}: all {len(rows):5d} | rows {sum(row['row_weight'] for row in rows):6d} | "
                f"edge {sum(EDGE in row['exclusion_reasons'] for row in rows):4d} | "
                f"conflicting {sum(bool(row['partners']) for row in rows):4d} | "
                f"rows per interval {sum(row['row_weight'] for row in rows) / max(len(rows), 1):4.1f}"
            )

    within: dict[str, Counter[tuple[str, str]]] = {"train": Counter(), "eval": Counter()}
    cross: Counter[tuple[str, str]] = Counter()
    exact = sum(row["exact"] for row in events)
    for row in events:
        for other in row["partners"]:
            target = cross if other in row["cross_file"] else within[row["official_split"]]
            target[(row["label"], other)] += 1
    lines.append("")
    lines.append("intervals of a class (row) that meet another class (column)")
    lines.append("  training set, both marks in one annotation file")
    lines.extend(_matrix(within["train"]))
    lines.append("  training set, marks in the two annotation files of a merged recording")
    lines.extend(_matrix(cross))
    lines.append("  evaluation set")
    lines.extend(_matrix(within["eval"]))
    conflicting = sum(bool(row["partners"]) for row in events)
    lines.append(f"  conflicting intervals: {conflicting}; with a partner on the identical second: {exact}")

    lines.append("")
    lines.append("intervals, rows and clusters that each conflict rule keeps (after the edge rule)")
    for split in ("train", "eval"):
        lines.append(f"  {split}")
        totals: Counter[str] = Counter()
        for label in LABELS:
            cells = []
            for rule in RULES:
                rows = [
                    row
                    for row in events
                    if row["official_split"] == split
                    and row["label"] == label
                    and EDGE not in row["exclusion_reasons"]
                    and keeps(rule, label, row["partners"])
                ]
                totals[rule] += len(rows)
                clusters = len({row["cluster_id"] for row in rows})
                cells.append(
                    f"{rule} {len(rows):5d} int {sum(row['row_weight'] for row in rows):6d} row {clusters:3d} cl"
                )
            lines.append(f"    {label:5s} " + " | ".join(cells))
        lines.append("    total " + " | ".join(f"{rule} {totals[rule]} int" for rule in RULES))

    merged = defaultdict(list)
    for row in events:
        merged[row["recording_id"]].append(row)
    both = agree = 0
    one_file: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    recordings = 0
    for rows in merged.values():
        if not any(1 in row["annotation_sources"] for row in rows):
            continue
        recordings += 1
        bins: tuple[dict[int, set[str]], dict[int, set[str]]] = (defaultdict(set), defaultdict(set))
        for row in rows:
            low, high = round(row["annotation_start_seconds"] * 10), round(row["annotation_stop_seconds"] * 10)
            for source in set(row["annotation_sources"]):
                for tick in range(low, high):
                    bins[min(source, 1)][tick].add(row["label"])
        for tick in set(bins[0]) | set(bins[1]):
            first, second = bins[0].get(tick), bins[1].get(tick)
            if first and second:
                both += 1
                agree += first == second
                confusion.update((left, right) for left in first for right in second)
            else:
                one_file.update(first or second or ())
    lines.append("")
    lines.append(f"merged recordings: {recordings}; agreement of their two annotation files in time (any channel)")
    lines.append(
        f"  time marked by both files {both / 10:.0f} s; same classes in {agree / 10:.0f} s "
        f"({100 * agree / max(both, 1):.0f}%)"
    )
    lines.append(
        f"  time marked by one file only, in seconds: { {k: round(one_file[k] / 10) for k in LABELS if one_file[k]} }"
    )
    lines.append("  seconds by class in the first file (row) and class in the second file (column)")
    lines.extend(_matrix(confusion, scale=10.0))

    lines.append("")
    lines.append("best balanced accuracy when identical seconds carry several classes (keep rule, after the edge rule)")
    for split in ("train", "eval"):
        rows = [row for row in events if row["official_split"] == split and EDGE not in row["exclusion_reasons"]]
        for name in ("row", "interval"):
            total: Counter[str] = Counter()
            groups: dict[tuple[str, float, float], Counter[str]] = defaultdict(Counter)
            for row in rows:
                weight = row["row_weight"] if name == "row" else 1
                total[row["label"]] += weight
                key = (row["recording_id"], row["annotation_start_seconds"], row["annotation_stop_seconds"])
                groups[key][row["label"]] += weight
            won: Counter[str] = Counter()
            for group in groups.values():
                best = max(group, key=lambda label: group[label] / total[label])
                won[best] += group[best]
            ceiling = sum(won[label] / total[label] for label in LABELS) / len(LABELS)
            several = sum(len(group) > 1 for group in groups.values())
            lines.append(
                f"  {split:5s} {name:8s}-weighted: {ceiling:.3f} | seconds with several classes {several} of {len(groups)}"
            )

    kept = [row for row in events if row["status"] == "kept"]
    displaced = examples.filter(pl.col("kind") == "displaced")
    lines.append("")
    lines.append("kept events under the manifest's own rule")
    lines.append(
        f"  context of 5 s holds another class: {sum(len(row['context_class_codes']) > 1 for row in kept)} of {len(kept)}"
    )
    for split in ("train", "eval"):
        rows = [row for row in kept if row["official_split"] == split]
        lines.append(
            f"  {split}: {len(rows)} events in {len({row['overlap_group_id'] for row in rows})} overlap groups, "
            f"{len({row['recording_id'] for row in rows})} recordings, {len({row['cluster_id'] for row in rows})} clusters"
        )
        for label in LABELS:
            sizes = sorted(Counter(row["cluster_id"] for row in rows if row["label"] == label).values(), reverse=True)
            lines.append(
                f"    {label}: {sum(sizes):5d} events in {len(sizes):3d} clusters; the 3 largest clusters hold "
                f"{100 * sum(sizes[:3]) / max(sum(sizes), 1):.0f}%"
            )
    lines.append(
        f"  displaced controls: {displaced.height}; kept events without a control: "
        f"{len(kept) - displaced['source_example_id'].n_unique()}"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Print the audit report of a manifest; return 0 on success, otherwise 1."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="A completed manifest directory")
    parser.add_argument("--output", type=Path, help="Also write the report to this new text file")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        text = "\n".join(report(args.manifest)) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text)
    except (OSError, ValueError, KeyError) as error:
        sys.stderr.write(f"Manifest audit failed: {error}\n")
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
