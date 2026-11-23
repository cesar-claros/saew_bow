"""TUEV annotation codes and canonical channel indices."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

CLASSES = {1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}


MONTAGE = (
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
)


def parse_rec(text: str) -> list[tuple[int, float, float, str]]:
    """The rows of a ``.rec`` file: channel index, start and stop in seconds, and the class.

    Raises
    ------
    ValueError
        If a row does not have four fields, or holds an unknown channel or code, or stops before it starts.
    """
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"row {number} does not have four fields: {line!r}")
        channel, start, stop, code = int(fields[0]), float(fields[1]), float(fields[2]), int(float(fields[3]))
        if not 0 <= channel < len(MONTAGE) or code not in CLASSES or stop <= start:
            raise ValueError(f"row {number} holds a bad channel, code or interval: {line!r}")
        rows.append((channel, start, stop, CLASSES[code]))
    return rows


@dataclass(frozen=True)
class Annotation:
    """One validated source row, retaining the physical one-based line number."""

    line_number: int
    channel: int
    start_seconds: float
    stop_seconds: float
    class_code: int


def parse_annotations(text: str) -> list[Annotation]:
    """Parse strict TUEV annotations without discarding row identity.

    Blank lines are ignored. Nonfinite/negative times, fractional channel/class
    values, unknown codes and nonpositive intervals raise ``ValueError`` with a
    line number. Recording-duration checks belong to manifest construction.
    """
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 4:
            raise ValueError(f"annotation line {number}: expected four fields")
        try:
            channel, start, stop, code = (float(field.strip()) for field in fields)
        except ValueError as error:
            raise ValueError(f"annotation line {number}: fields must be numeric") from error
        if not all(isfinite(value) for value in (channel, start, stop, code)):
            raise ValueError(f"annotation line {number}: fields must be finite")
        if not channel.is_integer() or not code.is_integer() or not 0 <= channel < len(MONTAGE) or code not in CLASSES:
            raise ValueError(f"annotation line {number}: invalid channel or class code")
        if start < 0 or stop <= start:
            raise ValueError(f"annotation line {number}: expected 0 <= start < stop")
        rows.append(Annotation(number, int(channel), start, stop, int(code)))
    return rows
