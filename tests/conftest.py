"""Synthetic TUEV fixtures; all runtime tests are reserved for the HPC."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from eeg_compare.data.annotations import MONTAGE
from eeg_compare.data.config import PipelineConfig, PreprocessingConfig, SplitConfig
from eeg_compare.data.manifest import CorpusLayout, build_manifest
from eeg_compare.data.provenance import content_hash, file_hash
from eeg_compare.data.recordings import ClusterKind, Recording, RecordingIdentity, cluster_of


@pytest.fixture
def recording() -> Recording:
    """A 20-second, 22-channel recording on the 250 Hz sample clock."""
    return Recording(
        recording_id=content_hash("train/example.edf"),
        edf_path="train/example.edf",
        rec_path="train/example.rec",
        patient_id="patient-0",
        session_id="session-0",
        identity_source="synthetic fixture",
        cluster_id="patient:patient-0",
        cluster_kind="patient",
        official_split="train",
        sfreq=250.0,
        n_samples=5000,
        source_channels=MONTAGE,
        channel_matrix=tuple(tuple(float(i == j) for j in range(len(MONTAGE))) for i in range(len(MONTAGE))),
        missing_channels=(),
        edf_sha256="synthetic-edf",
        rec_sha256="synthetic-rec",
    )


@pytest.fixture
def corpus_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording: Recording) -> tuple[Path, Path]:
    """Synthetic EDF/REC files and their identity CSV, with mocked header inspection and no real EEG data."""
    root = tmp_path / "corpus"
    identities = tmp_path / "identities.csv"
    rows = []
    for index, split in enumerate(("train", "train", "train", "eval")):
        path = root / split / f"record-{index}.edf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic header {index}".encode())
        path.with_suffix(".rec").write_text("\n".join(f"0,{3 + 2 * i},{4 + 2 * i},{i + 1}" for i in range(6)))
        rows.append(
            {
                "edf_path": path.relative_to(root).as_posix(),
                "patient_id": f"patient-{index}",
                "session_id": f"session-{index}",
                "identity_source": "synthetic fixture",
            }
        )
    with identities.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["edf_path", "patient_id", "session_id", "identity_source"])
        writer.writeheader()
        writer.writerows(rows)

    def inspect(
        corpus_root: Path,
        identity: RecordingIdentity,
        split: Literal["train", "eval"],
        config: PreprocessingConfig,
        eval_cluster: ClusterKind = "patient",
    ) -> Recording:
        path = corpus_root / identity.edf_path
        cluster_id, cluster_kind = cluster_of(identity, split, eval_cluster)
        return replace(
            recording,
            recording_id=content_hash([identity.edf_path, file_hash(path), file_hash(path.with_suffix(".rec"))]),
            edf_path=identity.edf_path,
            rec_path=path.with_suffix(".rec").relative_to(corpus_root).as_posix(),
            patient_id=identity.patient_id,
            session_id=identity.session_id,
            identity_source=identity.identity_source,
            cluster_id=cluster_id,
            cluster_kind=cluster_kind,
            official_split=split,
            edf_sha256=file_hash(path),
            rec_sha256=file_hash(path.with_suffix(".rec")),
        )

    monkeypatch.setattr("eeg_compare.data.manifest.inspect_recording", inspect)
    return root, identities


@pytest.fixture
def mini_corpus(tmp_path: Path, corpus_files: tuple[Path, Path]) -> tuple[Path, Path]:
    """Build real metadata artifacts from the synthetic files, with the dataclass defaults of the split rules."""
    root, identities = corpus_files
    config = PipelineConfig(
        preprocessing=PreprocessingConfig(input_mode="bipolar", highpass_hz=None, notch_hz=()),
        splits=SplitConfig(n_folds=2),
    )
    manifest_dir = tmp_path / "manifest"
    build_manifest(CorpusLayout(root, "synthetic"), identities, config, manifest_dir)
    return root, manifest_dir
