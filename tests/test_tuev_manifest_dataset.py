"""End-to-end metadata and lazy-reader checks with mocked EDF content; run on the HPC."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import orjson
import pytest
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from eeg_compare.data.annotations import parse_annotations
from eeg_compare.data.cli import main
from eeg_compare.data.config import ExampleConfig, PipelineConfig, PreprocessingConfig
from eeg_compare.data.dataset import Partition, ReaderOptions, TUEVDataset, TUEVReader, runtime_scope
from eeg_compare.data.examples import make_examples
from eeg_compare.data.manifest import CorpusLayout, build_manifest, load_manifest, read_identities
from eeg_compare.data.provenance import content_hash, resolve_path
from eeg_compare.data.recordings import Recording


def _signal(root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
    time = np.arange(recording.n_samples) / recording.sfreq
    return np.tile(np.sin(2 * np.pi * 10 * time), (len(config.channels), 1)).astype(np.float32)


def test_manifest_roundtrip_and_class_coverage(mini_corpus: tuple[Path, Path]) -> None:
    _, directory = mini_corpus
    manifest = load_manifest(directory)
    assert len(manifest.recordings) == len(manifest.assignments) == 4
    assert manifest.metadata["statistics"]["kept"] == 24
    assert manifest.scan_examples().select("example_id").collect().height == 24
    for fold in range(2):
        manifest.require_class_coverage(fold)


def test_artifact_tampering_and_incomplete_manifests_are_rejected(
    mini_corpus: tuple[Path, Path], tmp_path: Path
) -> None:
    _, directory = mini_corpus
    with (directory / "splits.parquet").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(directory)
    incomplete = tmp_path / "unfinished"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError):
        load_manifest(incomplete)


def test_dataset_partition_isolation_and_tensor_collation(
    mini_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory = mini_corpus
    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", _signal)
    training = TUEVDataset(root, directory, Partition("train", 0))
    validation = TUEVDataset(root, directory, Partition("validation", 0))
    evaluation = TUEVDataset(root, directory, Partition("eval", 0, allow_eval=True))
    groups = [set(dataset.examples["patient_id"]) for dataset in (training, validation, evaluation)]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    batch = next(iter(DataLoader(training, batch_size=2)))
    assert batch["signal"].shape == (2, 22, 1250)
    assert batch["valid_samples"].all() and batch["target"].shape == (2,)
    assert len(batch["example_id"]) == 2
    with pytest.raises(ValueError, match="allow_eval"):
        Partition("eval")


def test_disk_cache_reuse_and_checksum_failure(
    mini_corpus: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory = mini_corpus
    calls = []

    def source(corpus_root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
        calls.append(recording.recording_id)
        return _signal(corpus_root, recording, config)

    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", source)
    manifest = load_manifest(directory)
    options = ReaderOptions(cache_dir=tmp_path / "cache", memory_cache_records=0)
    record = manifest.recordings[0]
    first = TUEVReader(root, manifest, options).recording(record.recording_id)
    second = TUEVReader(root, manifest, options).recording(record.recording_id)
    np.testing.assert_array_equal(first, second)
    assert calls == [record.recording_id]
    key = content_hash([record.edf_sha256, TUEVReader(root, manifest, options).signal_key])
    assert options.cache_dir is not None
    with (options.cache_dir / f"{key}.npy").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        TUEVReader(root, manifest, options).recording(record.recording_id)


def test_source_mutation_is_detected_even_with_memory_cache(
    mini_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory = mini_corpus
    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", _signal)
    manifest = load_manifest(directory)
    reader = TUEVReader(root, manifest)
    record = manifest.recordings[0]
    reader.recording(record.recording_id)
    (root / record.edf_path).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source EDF"):
        reader.recording(record.recording_id)


def test_padding_does_not_enter_normalization_statistics(
    mini_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory = mini_corpus
    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", _signal)
    manifest = load_manifest(directory)
    config = PipelineConfig(
        preprocessing=replace(manifest.config.preprocessing, normalization="zscore"),
        examples=ExampleConfig(edge_policy="zero_pad"),
        splits=manifest.config.splits,
    )
    metadata = {
        **manifest.metadata,
        "config": config.to_dict(),
        "config_hash": content_hash(config.to_dict()),
        "preprocessing_hash": content_hash([asdict(config.preprocessing), manifest.metadata["runtime"]]),
    }
    reader = TUEVReader(root, replace(manifest, config=config, metadata=metadata))
    row = make_examples(manifest.recordings[0], parse_annotations("0,0,1,1"), config)[0]
    result = reader.example(row)
    assert row.pad_left == 500 and result.valid_samples.sum() == 750
    np.testing.assert_array_equal(result.signal[:, :500], 0)
    np.testing.assert_allclose(result.signal[:, 500:].mean(axis=-1), 0, atol=1e-6)
    np.testing.assert_allclose(result.signal[:, 500:].std(axis=-1), 1, atol=1e-6)


def test_runtime_changes_refuse_stale_manifest(mini_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    root, directory = mini_corpus
    manifest = load_manifest(directory)
    monkeypatch.setattr("eeg_compare.data.dataset.runtime_fingerprint", lambda: {"different": True})
    with pytest.raises(ValueError, match="rebuild"):
        TUEVReader(root, manifest)


def test_identity_mapping_preserves_leading_zeros_and_requires_evidence(tmp_path: Path) -> None:
    mapping = tmp_path / "identities.csv"
    mapping.write_text("edf_path,patient_id,session_id,identity_source\ntrain/a.edf,0001,01,curated lookup\n")
    assert read_identities(mapping)["train/a.edf"].patient_id == "0001"
    mapping.write_text("edf_path,patient_id,session_id,identity_source\ntrain/a.edf,,01,\n")
    with pytest.raises(ValueError, match="required"):
        read_identities(mapping)
    with pytest.raises(ValueError):
        resolve_path(tmp_path, "../outside.edf")


def test_unknown_config_or_changed_preprocessing_hash_fails(mini_corpus: tuple[Path, Path]) -> None:
    _, directory = mini_corpus
    metadata_path = directory / "metadata.json"
    metadata = orjson.loads(metadata_path.read_bytes())
    metadata["preprocessing_hash"] = "wrong"
    metadata_path.write_bytes(orjson.dumps(metadata))
    with pytest.raises(ValueError, match="preprocessing hash"):
        load_manifest(directory)
    with pytest.raises(ValueError, match="unknown"):
        PipelineConfig.from_dict({"preprocesing": {}})


def test_concurrent_cache_readers_share_one_completed_entry(
    mini_corpus: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory = mini_corpus
    manifest = load_manifest(directory)
    calls = []

    def source(corpus_root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
        calls.append(recording.recording_id)
        return _signal(corpus_root, recording, config)

    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", source)
    options = ReaderOptions(cache_dir=tmp_path / "shared-cache")
    readers = [TUEVReader(root, manifest, options) for _ in range(2)]
    identifier = manifest.recordings[0].recording_id
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(reader.recording, identifier) for reader in readers]
        first, second = [future.result(timeout=30) for future in futures]
    np.testing.assert_array_equal(first, second)
    assert calls == [identifier]


def test_completed_manifest_cannot_be_overwritten(mini_corpus: tuple[Path, Path]) -> None:
    root, directory = mini_corpus
    before = (directory / "metadata.json").read_bytes()
    manifest = load_manifest(directory)
    with pytest.raises(FileExistsError):
        build_manifest(CorpusLayout(root, "synthetic"), root.parent / "identities.csv", manifest.config, directory)
    assert (directory / "metadata.json").read_bytes() == before


def test_cli_check_records_scope_and_requires_explicit_eval(
    mini_corpus: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory = mini_corpus
    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", _signal)
    output = tmp_path / "signal-check.json"
    args = ["check", "--root", str(root), "--manifest", str(directory), "--output", str(output), "--limit", "2"]
    assert main([*args, "--split", "eval"]) == 2
    assert not output.exists()
    assert main(args) == 0
    report = orjson.loads(output.read_bytes())
    assert len(report["checked_examples"]) == 2 and "smoke check only" in report["scope"]
    assert main(args) == 2


def test_only_code_that_makes_tables_or_signals_invalidates_a_manifest(
    mini_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory = mini_corpus
    manifest = load_manifest(directory)
    built = manifest.metadata["runtime"]
    key = TUEVReader(root, manifest).signal_key

    def changed(path: str | None = None, library: str | None = None) -> dict[str, object]:
        runtime = {**built, "source_hashes": dict(built["source_hashes"]), "libraries": dict(built["libraries"])}
        if path is not None:
            assert path in runtime["source_hashes"]
            runtime["source_hashes"][path] = "edited"
        if library is not None:
            runtime["libraries"][library] = "0.0.0"
        return runtime

    for path in ("data/dataset.py", "data/cli.py"):
        monkeypatch.setattr("eeg_compare.data.dataset.runtime_fingerprint", lambda path=path: changed(path))
        assert TUEVReader(root, manifest).signal_key == key
    for path, library in (
        ("data/examples.py", None),
        ("data/recordings.py", None),
        ("preprocessing/resampling.py", None),
        (None, "polars"),
        (None, "numpy"),
    ):
        monkeypatch.setattr(
            "eeg_compare.data.dataset.runtime_fingerprint", lambda path=path, library=library: changed(path, library)
        )
        with pytest.raises(ValueError, match="rebuild"):
            TUEVReader(root, manifest)
    signal = runtime_scope(built, "signal")
    assert runtime_scope(changed("data/examples.py"), "signal") == signal
    assert runtime_scope(changed(None, "polars"), "signal") == signal
    assert runtime_scope(changed("data/recordings.py"), "signal") != signal
    assert runtime_scope(changed("preprocessing/temporal_filtering.py"), "signal") != signal
    assert runtime_scope(changed(None, "scipy"), "signal") != signal
    assert runtime_scope({"different": True}, "tables") != runtime_scope(built, "tables")
