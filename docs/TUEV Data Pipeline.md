# TUEV Data Pipeline

Implemented 2026-11-23. Runtime tests and corpus validation have not been run.
All commands in this note are for the HPC. The implementation supplies stage 1
of the comparison plan; its empirical acceptance checks remain pending.

## Inputs and Identity

The corpus root contains separate official training and evaluation directories.
Every EDF needs one same-stem `.rec` companion. Directory names are configurable
with `--train-subdir` and `--eval-subdir`; paths in artifacts remain relative to
the corpus root, so the dataset can move between mounts.

An explicit identity CSV is required:

```csv
edf_path,patient_id,session_id,identity_source
train/example_a.edf,patient_001,session_01,reviewed authoritative identity mapping
eval/example_b.edf,patient_002,session_02,reviewed authoritative identity mapping
```

These rows are fictional. Keep the real mapping under ignored `data/` or outside
the repository. Patient IDs must use the same namespace across official splits.
`identity_source` records the evidence; the software cannot independently prove
that the supplied IDs identify people. Session IDs may be empty. Unknown patient
IDs, duplicated mapping rows, missing mappings, unused mappings, and official
patient overlap are errors. The reader never guesses patient IDs from paths.
If patient identities cannot be recovered, this pipeline cannot certify a
patient-disjoint comparison; a weaker grouping study needs a separately declared
protocol.

## Signal Contract

The starter [configuration](../configs/tuev.toml) proposes these explicit choices:

| Component | Default | Alternatives |
|---|---|---|
| Input | Referential electrodes with one reference suffix | Already bipolar EDF channels |
| Output montage | All 22 canonical TUEV derivations, in annotation order | Any explicit unique subset/order of those names |
| Native-rate filtering | 0.1 Hz high-pass, 60 Hz notch | Configurable passband and notch list; 0 disables a passband edge, `[]` disables notch |
| Resampling | 250 Hz, rational polyphase ratio | Explicit target frequency |
| Normalization | None; retain MNE's physical signal values in volts | Per-channel z-score or lower-median/MAD scaling |
| Normalization scope | Real samples of an example | Complete recording |
| Clipping | None | Explicit threshold in the resulting units |
| Missing channels | Error | Exclude the complete recording with a recorded reason |

These are proposed comparison settings, not verified reproduction settings.
Inspect the corpus's physical units and channel labels on the HPC before treating
the resulting scale as validated. No implicit CAR, interpolation, bad-channel
sidecar lookup, or old experiment preprocessing is applied.

Channel matching accepts an optional `EEG ` prefix and `-REF`/`-LE` suffix.
T7/T8/P7/P8 map to the legacy T3/T4/T5/T6 names. Duplicated canonical inputs and
mixed reference suffixes are rejected. Already bipolar inputs can use a reversed
pair; its coefficient is recorded as -1. `recordings.parquet` stores the ordered
EDF names and the exact channel matrix.

MNE includes only the required channels. Waveform loading uses `preload=True`,
which avoids per-slice mixed-rate resampling artifacts described in the
[MNE EDF reader documentation](https://mne.tools/stable/generated/mne.io.read_raw_edf.html).
The selected-channel native clock is recorded in the manifest. Filtering uses
complete recordings, followed by resampling from recording time zero, then
example extraction. FIR settings are explicit: zero phase, Hamming window,
`firwin`, and `reflect_limited` boundary padding, following the
[MNE filter API](https://mne.tools/stable/generated/mne.filter.filter_data.html).
Reflection effects at actual recording boundaries still need inspection.

Z-scores use population standard deviation. MAD scaling uses the lower median,
factor 1.4826, a 1% standard-deviation floor, and an absolute floor of 1e-12 volts.
These are loader transformations, separate from the retained encoder's internal
normalization. Keep loader normalization off when retaining that reference path;
the later method adapter must prevent unintended double normalization.

## Examples and Exclusions

Defaults define a five-second context from two seconds before annotation onset
to three seconds after it. The output has 1250 samples at 250 Hz. Start positions
round to the nearest target sample, with exact ties toward positive infinity.
Original annotation times and the fractional onset offset are retained. There is
no crop-to-pooling-length step and no shifting of an edge example into the record.

- `unit="interval"` merges channels with identical onset, offset and class into
  one multichannel example. `unit="annotation"` retains separate channel marks.
  Exact duplicate rows merge in both modes, retaining every source line number.
- Conflicts mean overlapping half-open annotated intervals with different labels
  on any channel. They are excluded by default; `conflict_policy="keep"` retains
  them with a flag. Other labels inside the wider context are recorded separately.
- Contexts crossing recording boundaries are excluded by default.
  `edge_policy="zero_pad"` pads after normalizing real samples and returns a sample
  mask. Padding never contributes to example normalization statistics.
- Marks wholly on excluded output channels are excluded by name; canonical `.rec`
  channel indices are never renumbered. Missing-channel recording exclusions also
  appear on every affected example.
- Overlapping contexts receive a connected-component group ID, including excluded
  contexts. All remain on their recording's patient partition.
- Unannotated time creates no examples or background labels. Source class codes
  1–6 map to targets 0–5 in the order spsw, gped, pled, eyem, artf, bckg.

Malformed annotations, nonfinite/negative times, fractional class/channel codes,
unknown codes, nonpositive intervals, and annotations extending beyond a record
are errors. The strict `parse_annotations` API is used here; the original
`parse_rec` compatibility helper remains unchanged.

Record IDs bind the relative EDF path and EDF/REC content hashes. Example IDs bind
that recording snapshot, annotation identity, context settings and target rate.
They are independent of the input sequence order within that snapshot. The
manifest hash is still required when comparing outputs across preprocessing arms.

## Manifests and Folds

`build` writes one Parquet example shard per recording, plus `recordings.parquet`,
`splits.parquet`, and a final `metadata.json` completion marker. Excluded examples
remain in the shards with their reasons. No signal arrays enter these manifests.
Content hashing reads the complete EDF files, so the build is still an I/O pass
over the corpus even though it does not decode waveform samples.

Patients from official training data are hash-shuffled with the declared seed and
dealt across development folds. Patient counts differ by at most one. Fold `f`
uses those patients assigned to `f` for validation and the remaining training
patients for fitting. Official evaluation patients always have fold -1 and stay
in evaluation. Duplicate EDF bytes and official patient overlap fail explicitly.

This baseline split balances patient counts, not class distributions. The manifest
reports six-class coverage for every fold and partition. Missing classes do not
trigger silent reshuffling. Call `manifest.require_class_coverage(f)` before a
six-class development benchmark, and report any later split-policy change.

The manifest stores resolved configuration, identity-map hash, source hashes,
artifact hashes, data-pipeline source hashes and numerical-library versions. The
loader verifies artifacts and fold assignments. Changed preprocessing code or
library versions require rebuilding the manifest. Existing output directories
are refused; an interrupted build without a valid completion marker is unusable.

## HPC Workflow

For the existing EEG container, complete [HPC Container Setup](HPC%20Container.md)
and its environment report first. That guide preserves the image's Torch
installation and records the image identity and installed package snapshot.
For an isolated environment, resolve and commit its lockfile on the cluster.
In either case, run the tests before creating manifests. The commands below use
the prepared environment directly. Replace the corpus paths and `YOUR_RELEASE`
with the installed release:

```bash
set -e
.venv/bin/python -m pytest -q

.venv/bin/python -m eeg_compare.data.cli build \
  --root /path/to/tuev/edf --version YOUR_RELEASE \
  --identities data/tuev_identities.csv --config configs/tuev.toml \
  --output artifacts/tuev-manifest-001

.venv/bin/python -m eeg_compare.data.cli check \
  --root /path/to/tuev/edf --manifest artifacts/tuev-manifest-001 \
  --split train --fold 0 --limit 8 --cache-dir cache/tuev \
  --output artifacts/tuev-signal-check-001.json
```

`check` deterministically samples the selected partition and records shapes,
validity counts and timing offsets. It is a smoke check, not a completed corpus
audit. Evaluation loading requires explicit `--allow-eval` or `allow_eval=True`.
Keep evaluation examples out of method development.

The programmatic dataset interface is:

```python
from pathlib import Path
from torch.utils.data import DataLoader
from eeg_compare.data.dataset import Partition, ReaderOptions, TUEVDataset

dataset = TUEVDataset(
    Path("/path/to/tuev/edf"),
    Path("artifacts/tuev-manifest-001"),
    Partition("train", validation_fold=0),
    ReaderOptions(cache_dir=Path("cache/tuev")),
)
dataset.manifest.require_class_coverage(0)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
```

Each batch contains `signal` (B,C,T), `target` (B,), `example_id`, `valid_samples`
(B,T), and `valid_channels` (B,C). Flat channels are flagged, not silently repaired.
`dataset.example_metadata(i)` exposes annotation lines, channels, intervals and
overlap groups. Model adapters still need to consume this interface and its masks;
the retained trainer expects `(X,y)` and rebuilds its own sampler.

Processed recordings use a bounded per-worker LRU cache (one recording by default).
Optional disk caching uses source/preprocessing/runtime keys, checksums, atomic
completion markers and POSIX locks. Use storage with working POSIX advisory locks
for multiple workers, such as cluster scratch. Cache data are read-only and cannot
be modified by example normalization. Peak preprocessing memory still scales with
the largest recording; block processing of very long EDFs is future work. Source
hashes are checked on first access per worker and again when file metadata changes.

## Validation Status and Acceptance

Local checks cover Python 3.12 syntax, Ruff, mypy, import isolation, TOML parsing,
and static review. Mypy uses the configured missing-import handling; repeat it
with the installed HPC dependencies. Runtime tests were saved but not executed.
They cover annotation policies, channel mapping, resampling timing, split leakage,
manifest integrity, dataset collation, normalization masks and cache integrity.

Before treating stage 1 as empirically accepted, run the suite and smoke check,
review the supplied identity evidence, reconcile counts/exclusions with the actual
release, and manually check at least one EDF against its raw annotation times and
channel names. Inspect signal units, filtering near recording boundaries, flat
channels, and representative classes. None of those corpus checks has been claimed
complete by this implementation.
