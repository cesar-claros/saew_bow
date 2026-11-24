# TUEV Data Pipeline

Implemented 2026-11-23. Revised the same day for the changes that the plan review required: session clusters, stratified folds, row weights, a conflict report, displaced control contexts and the common signal contract ([Comparison Plan](Comparison%20Plan.md)). Runtime tests and corpus validation have not been run. All commands in this note are for the HPC. The implementation supplies stage 1 of the comparison plan; its empirical acceptance checks remain pending.

## Inputs and Identity

The corpus root contains separate official training and evaluation directories. Every EDF needs one same-stem `.rec` companion. Directory names are configurable with `--train-subdir` and `--eval-subdir`; paths in artifacts remain relative to the corpus root, so the dataset can move between mounts.

An explicit identity CSV is required:

```csv
edf_path,patient_id,session_id,identity_source
train/00000001/00000001_00000001.edf,00000001,,patient folder of the corpus README
eval/032/bckg_032_a_.edf,,032,session index of the corpus README
```

These rows are fictional. Paths are relative to the corpus root that `--root` names (`$DATA_DIR/v2.0.1/edf`). The corpus README describes a training path as `train/<patient>/<patient>_<file>.edf` and an evaluation path as `eval/<session index>/<file>.edf`; check the folders on disk against it, because the earlier inventory found patient folders with letter codes. Keep the real mapping under ignored `data/` or outside the repository. `identity_source` records the evidence; the software cannot independently prove that the supplied IDs identify people. A row needs its evidence and at least one of the two IDs. Duplicated mapping rows, missing mappings and unused mappings are errors. The reader never guesses an identity from a path.

Each recording belongs to one cluster, the independent unit for folds, partitions and resampling:

- A training recording clusters by patient and needs a `patient_id`. Its cluster ID is `patient:<patient_id>`.
- An evaluation recording uses the kind that `[splits] eval_cluster` declares. With `"patient"` (the dataclass default) it needs a `patient_id`. With `"session"` (the value in `configs/tuev.toml`) it needs a `session_id`, and its cluster ID is `session:<session_id>`. TUEV v2.0.1 gives a random session index for each evaluation file and no patient identity.

`scripts/make_tuev_identities.py` writes this CSV from the folder layout of the corpus. The folder that holds a training file names its patient, and the folder that holds an evaluation file names its session index; the earlier inventory of this corpus used the same rule. The script is a separate, explicit step: the manifest build itself never reads an identity from a path. It needs only the standard library, refuses to overwrite its output, and prints a summary to review before the build: the files and groups of each official split, the files without one `.rec` companion, the files whose name does not contain the name of their folder, and whether the counts match v2.0.1 (359 files of 290 patients, 159 files of 80 sessions).

```bash
python scripts/make_tuev_identities.py --root "$DATA_DIR/v2.0.1/edf" --output data/tuev_identities.csv
```

Patient IDs that are present use one namespace across the official splits, and a patient in both splits is an error. With session clusters this check cannot cover the evaluation set: its disjointness from the training patients then rests on the corpus documentation, and two sessions can belong to one person. The manifest records this in `cluster_policy`. Do not report evaluation sessions as verified independent patients.

### Identical EDF files

TUEV v2.0.1 holds EDF files with identical bytes. A scan of the corpus in November 2026 found 20 groups, all in the training set. Each group is one file in two patient folders with adjacent codes, and the two annotation files of each group differ. The 20 groups join 16 pairs of folders, so the 290 training folders hold at most 274 independent patients. No group joins the training and the evaluation set. The evaluation files are pruned segments, so identical bytes cannot show if an evaluation session comes from a training patient.

A recording belongs to one person. Two folders with the same recording therefore name the same person, or one of them is wrong. `[splits] duplicate_content` sets the rule:

- `"error"` (the dataclass default) stops the build before the output directory exists and names the files of each group.
- `"merge"` (the value in the configuration files) keeps one recording per group, the one with the first path. Its annotation rows are the rows of all copies. Rows that both files hold raise `row_weight` and count as duplicate rows, so the row-weighted score still counts each row as the published row-level protocol does. A mark in one file that overlaps a mark of another class in the other file meets the usual conflict rule. The clusters of a group become one cluster, and the join is transitive. The merged cluster ID lists the folder IDs with `+` between them, for example `patient:patient_001+patient_002`, and every recording of the joined folders carries it.

A group that joins the training and the evaluation set always stops the build, because it is a leak between the official sets and needs its own decision. `metadata.json` holds `duplicate_content`: the policy and, per group, the files, the joined clusters, the rows and classes of each annotation file, the intervals that both files mark, and the conflicts. Each example records the file of each of its rows in `annotation_sources`, parallel to `annotation_lines`.

## Signal Contract

[configs/tuev.toml](../configs/tuev.toml) holds the common comparison protocol. The dataclass defaults in `config.py` differ from it (0.1 Hz high-pass, 60 Hz notch, no normalization), so always pass a configuration file.

| Component | `configs/tuev.toml` | Alternatives |
|---|---|---|
| Input | Referential electrodes with one reference suffix | Already bipolar EDF channels |
| Output montage | All 22 canonical TUEV derivations, in annotation order | Any explicit unique subset/order of those names |
| Native-rate filtering | Band-pass 1 to 30 Hz, no notch (the low-pass removes the 60 Hz line) | `configs/tuev_wide.toml`: 1 to 70 Hz with a 60 Hz notch; 0 disables a passband edge, `[]` disables the notch |
| Resampling | 250 Hz, rational polyphase ratio | Explicit target frequency |
| Normalization | Lower-median/MAD scaling per channel | Per-channel z-score, or none (physical values in volts) |
| Normalization scope | Complete recording | Real samples of an example |
| Clipping | 20 robust units, the row clip of the retained model | Another threshold in the resulting units, or none |
| Missing channels | Error | Exclude the complete recording with a recorded reason |

These are study choices, not verified reproduction settings. Scaling per recording removes electrode gain and keeps the amplitude relations inside a recording. Scaling each 5 s example would divide every channel by the amplitude of an event that fills the example. The clip value is a proposal that copies the retained model; confirm it. Inspect the corpus's physical units and channel labels on the HPC before treating the result as validated. No implicit CAR, interpolation, bad-channel sidecar lookup, or old experiment preprocessing is applied. A manifest holds one band; the wide band needs its own manifest and cache entries.

Channel matching accepts an optional `EEG ` prefix and `-REF`/`-LE` suffix. T7/T8/P7/P8 map to the legacy T3/T4/T5/T6 names. Duplicated canonical inputs and mixed reference suffixes are rejected. Already bipolar inputs can use a reversed pair; its coefficient is recorded as -1. `recordings.parquet` stores the ordered EDF names and the exact channel matrix.

MNE includes only the required channels. Waveform loading uses `preload=True`, which avoids per-slice mixed-rate resampling artifacts described in the [MNE EDF reader documentation](https://mne.tools/stable/generated/mne.io.read_raw_edf.html). The selected-channel native clock is recorded in the manifest. Filtering uses complete recordings, followed by resampling from recording time zero, then normalization of the recording, then example extraction. FIR settings are explicit: zero phase, Hamming window, `firwin`, and `reflect_limited` boundary padding, following the [MNE filter API](https://mne.tools/stable/generated/mne.filter.filter_data.html). Reflection effects at actual recording boundaries still need inspection.

Z-scores use population standard deviation. MAD scaling uses the lower median, factor 1.4826, a 1% standard-deviation floor, and an absolute floor of 1e-12 volts. These are loader transformations. The retained encoder has its own per-segment scaling; with this configuration use the comparison adapters (`eeg_compare.adapters.models`), whose `input_scaling="external"` turns that scaling off, so the signal is not normalized twice.

## Examples and Exclusions

Defaults define a five-second context from two seconds before annotation onset to three seconds after it. The output has 1250 samples at 250 Hz. Start positions round to the nearest target sample, with exact ties toward positive infinity. Original annotation times and the fractional onset offset are retained. There is no crop-to-pooling-length step and no shifting of an edge example into the record.

- `unit="interval"` merges channels with identical onset, offset and class into one multichannel example. `unit="annotation"` retains separate channel marks. Exact duplicate rows merge in both modes, retaining every source line number.
- `row_weight` is the number of source annotation rows of an example, duplicates included. The published TUEV protocol uses one example per row, so a score weighted by `row_weight` is the one that compares with the literature. The unweighted score on merged intervals is the primary one.
- Conflicts mean overlapping half-open annotated intervals with different labels on any channel. Each example records the other classes that it meets in `conflict_class_codes`, whatever the rule. Other labels inside the wider context are recorded separately.
- `conflict_policy` decides what a conflict excludes. `"background_yields"` (the value in the configuration files) reads a background mark as "no event on this channel": a background interval that meets another class is excluded as `background_with_other_class`, an interval of another class stays when background is all that it meets, and two other classes that meet exclude each other as `conflicting_annotation`. `"drop"` (the dataclass default) excludes every conflicting interval. `"keep"` excludes none; a manifest with this rule holds the examples of the published row-level protocol.
- `metadata.json` holds a `conflict_report`: per official split and class, the intervals and rows in all, those with a conflict, and those kept under each of the three rules. One build gives all rules; `build` also logs the classes with conflicts. In TUEV v2.0.1 one interval in five has a conflict, mostly an artifact on some channels with another class on other channels (see the audit in the comparison plan).
- Contexts crossing recording boundaries are excluded by default. `edge_policy="zero_pad"` pads after normalizing real samples and returns a sample mask. Padding never contributes to example normalization statistics.
- Marks wholly on excluded output channels are excluded by name; canonical `.rec` channel indices are never renumbered. Missing-channel recording exclusions also appear on every affected example.
- Overlapping contexts receive a connected-component group ID, including excluded contexts, within each example kind. All remain on their recording's cluster partition.
- Unannotated time creates no examples or background labels. Source class codes 1–6 map to targets 0–5 in the order spsw, gped, pled, eyem, artf, bckg.

With `displaced_controls = true`, each kept event example also gets one displaced control (`kind="displaced"`): a context of the same recording whose every sample is at least `displaced_gap_seconds` (10 s) from every annotation row on any channel. A hash of the event ID draws its start uniformly from the free starts of the recording, so the draw is reproducible and needs no seed. The control records its event in `source_example_id` and carries that event's class and `row_weight`. It serves the record-only control of the plan: it asks whether the recording alone predicts the class. It is never an event example, and unannotated time stays unlabeled for every other purpose. A recording without a free start yields no control (`recordings_without_displaced_context` in the statistics). The displaced settings do not change the IDs of event examples.

Malformed annotations, nonfinite/negative times, fractional class/channel codes, unknown codes, nonpositive intervals, and annotations extending beyond a record are errors. The strict `parse_annotations` API is used here; the original `parse_rec` compatibility helper remains unchanged.

Record IDs bind the relative EDF path and EDF/REC content hashes. Example IDs bind that recording snapshot, annotation identity, context geometry (the seconds before and after onset, and the unit) and target rate. The edge rule and the conflict rule decide which examples are kept and do not enter the ID, so one example has one ID in two manifests that differ only in those rules, and predictions pair across them. They are independent of the input sequence order within that snapshot. The manifest hash is still required when comparing outputs across preprocessing arms.

## Manifests and Folds

`build` writes one Parquet example shard per recording, plus `recordings.parquet`, `splits.parquet`, and a final `metadata.json` completion marker. Excluded examples remain in the shards with their reasons. No signal arrays enter these manifests. Content hashing reads the complete EDF files, so the build is still an I/O pass over the corpus even though it does not decode waveform samples. The manifest schema version is 2; a version-1 manifest is refused.

`[splits] stratify` selects the fold rule for the training patients:

- `"class_presence"` (the value in `configs/tuev.toml`): iterative stratification by the classes of each patient's kept event examples. The class with the fewest unassigned patients goes first. Each of its patients goes to the fold that still wants most patients of that class, then to the fold that wants most patients in all, then by a seeded hash. The number of patients per class is then as even over the folds as the counts allow. Fold sizes are close to equal but not guaranteed within one; `fold_sizes` reports them. The rule uses training labels only.
- `"none"` (the dataclass default): patients are hash-shuffled with the declared seed and dealt across the folds. Patient counts differ by at most one, and labels are not used.

The stratified rule reads the classes of the kept examples, so a manifest that is built with another edge or conflict rule can have other folds. Compare such rules inside one manifest, not across two.

Fold `f` uses the patients assigned to `f` for validation and the remaining training patients for fitting. Official evaluation clusters always have fold -1 and stay in evaluation. Duplicate EDF bytes and a known patient in both official splits fail explicitly. `load_manifest` recomputes the assignments from the recordings and, for the stratified rule, from the class presence in the example shards, and refuses a manifest whose stored assignments differ.

The manifest reports, for every fold and partition, the examples per class, the clusters that hold each class, and the missing classes. The cluster counts are the effective sample size of a patient-level resampling. Missing classes do not trigger silent reshuffling. Call `manifest.require_class_coverage(f)` before a six-class development benchmark, and report any later split-policy change.

The manifest stores resolved configuration, identity-map hash, cluster policy, source hashes, artifact hashes, data-pipeline source hashes and numerical-library versions. The loader verifies artifacts and fold assignments. Changed preprocessing code or library versions require rebuilding the manifest. Existing output directories are refused; an interrupted build without a valid completion marker is unusable.

## HPC Workflow

For the existing EEG container, complete [HPC Container Setup](HPC%20Container.md) and its environment report first. That guide preserves the image's Torch installation and records the image identity and installed package snapshot. For an isolated environment, resolve and commit its lockfile on the cluster. In either case, run the tests before creating manifests.

The commands use `python`, the interpreter of the active environment. If `eeg_compare` is not installed in that interpreter, prefix a command with `PYTHONPATH=src`. The corpus is TUEV v2.0.1 at `$DATA_DIR/v2.0.1`, with the official directories `edf/train` and `edf/eval` below it. `DATA_DIR` comes from the `.env` file of the HPC project; if the shell does not hold it, load it with `set -a; source .env; set +a`.

```bash
set -e
python -m pytest -q

python -m eeg_compare.data.cli build \
  --root "$DATA_DIR/v2.0.1/edf" --version v2.0.1 \
  --identities data/tuev_identities.csv --config configs/tuev.toml \
  --output artifacts/tuev-manifest-001

python -m eeg_compare.data.cli check \
  --root "$DATA_DIR/v2.0.1/edf" --manifest artifacts/tuev-manifest-001 \
  --split train --fold 0 --limit 8 --cache-dir cache/tuev \
  --output artifacts/tuev-signal-check-001.json
```

`check` deterministically samples the selected partition and records shapes, validity counts and timing offsets. `--kind displaced` checks the control contexts. It is a smoke check, not a completed corpus audit. Evaluation loading requires explicit `--allow-eval` or `allow_eval=True`. Keep evaluation examples out of method development.

After `build`, read these parts of `metadata.json` before any fit: `statistics` (kept examples and rows, exclusions, displaced controls), `duplicate_content`, `conflict_report`, `fold_sizes`, `class_coverage` and `cluster_policy`. Reconcile the row counts with the corpus inventory of the plan (83,932 training and 29,421 evaluation rows in v2.0.1).

The programmatic dataset interface is:

```python
import os
from pathlib import Path
from torch.utils.data import DataLoader
from eeg_compare.adapters.models import ComparisonSeparableSAE, ComparisonSeparableSpec, SignalPairs
from eeg_compare.data.dataset import Partition, ReaderOptions, TUEVDataset

root = Path(os.environ["DATA_DIR"]) / "v2.0.1" / "edf"
dataset = TUEVDataset(
    root,
    Path("artifacts/tuev-manifest-001"),
    Partition("train", validation_fold=0),
    ReaderOptions(cache_dir=Path("cache/tuev")),
)
dataset.manifest.require_class_coverage(0)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

controls = TUEVDataset(
    root,
    Path("artifacts/tuev-manifest-001"),
    Partition("train", validation_fold=0, kind="displaced"),
)

model = ComparisonSeparableSAE(ComparisonSeparableSpec(n_atoms=32, atom_len=250, pool_len=250, n_projections=4))
pairs = DataLoader(SignalPairs(dataset), batch_size=16)
```

Each batch contains `signal` (B,C,T), `target` (B,), `example_id`, `row_weight` (B,), `valid_samples` (B,T), and `valid_channels` (B,C). Flat channels are flagged, not silently repaired. `dataset.example_metadata(i)` exposes annotation lines, channels, intervals, the example kind, the source example of a control, and overlap groups. A dataset holds one example kind. `Partition(conflict_view="keep")` also selects the events that only the conflict rule excludes. This view holds the examples of the published row-level protocol with the folds and IDs of the same manifest, so use it, and not a second manifest, for a comparison with published numbers. The edge, channel and recording rules still apply to it, and it has no displaced controls. `check --conflict-view keep` reads examples of this view. `SignalPairs` gives the `(signal, target)` items that the retained trainer expects; that trainer rebuilds its own loader with uniform shuffling, so patient-balanced exposure still needs the common training adapter.

Processed recordings use a bounded per-worker LRU cache (one recording by default). Optional disk caching uses source/preprocessing/runtime keys, checksums, atomic completion markers and POSIX locks. Use storage with working POSIX advisory locks for multiple workers, such as cluster scratch. Cache data are read-only and cannot be modified by example normalization. Peak preprocessing memory still scales with the largest recording; block processing of very long EDFs is future work. Source hashes are checked on first access per worker and again when file metadata changes.

## Validation Status and Acceptance

Local checks cover Python syntax, Ruff, mypy on the package, import isolation, TOML parsing, and static review. Mypy uses the configured missing-import handling, so it does not check the calls into Torch, MNE or Polars; repeat it with the installed HPC dependencies. Runtime tests were saved but not executed. They cover annotation policies, channel mapping, resampling timing, split leakage, manifest integrity, dataset collation, normalization masks and cache integrity. The review changes add tests for cluster kinds, stratified folds, row weights, the conflict report, displaced controls, identical EDF files and the comparison adapters. Their expected values were traced by hand, not run.

Before treating stage 1 as empirically accepted, run the suite and smoke check, review the supplied identity evidence, reconcile counts/exclusions with the actual release, and manually check at least one EDF against its raw annotation times and channel names. Inspect signal units, filtering near recording boundaries, flat channels, and representative classes. Read the conflict report. None of those corpus checks has been claimed complete by this implementation.
