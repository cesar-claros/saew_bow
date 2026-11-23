# Code Reuse

The source is `cesar-claros/tuh-eeg-epilepsy`, revision
`5ef7deee956ab7fd1713d475f81960df30150483`. The selected source files were unchanged
relative to that revision. `provenance.json` records their SHA-256 hashes and the
changes made during extraction. Files are independent copies, not symlinks.

## Retained components

| Destination in the new package | Original source | Reason |
|---|---|---|
| `models/separable_sae.py` | `src/models/components/separable_sae.py` | Current proposal reference |
| `models/waveform_sae.py` | `src/models/components/waveform_sae.py` | Parent training and selection implementation; full-template ablation |
| `models/spectral_features.py` | `src/models/components/spectral_features.py` | Spectral control with explicit sampling rate |
| `preprocessing/robust.py` | Three helpers and constants from `shapeconv_sae.py` | Preserve exact row scaling without importing the older model family |
| `device.py` | `_resolve_device` from `hydra_transform.py` | Preserve device selection without copying HYDRA |
| `preprocessing/resampling.py` | `_resample` from `window_dataset.py` | Exact rational polyphase resampling |
| `preprocessing/temporal_filtering.py` | Same-named data component | Correct context for zero-phase FIR filtering |
| `data/annotations.py` | Selected definitions from `event_dictionary_core.py` | Six event classes, canonical 22-channel annotation map, `.rec` parser |
| `tests/test_separable_sae.py` | Same-named source test | Adjointness, geometry, selection, reconstruction, initialization, fit, and checkpoint invariants |
| `tests/test_temporal_filtering.py` | Same-named source test | Crop-context versus whole-record filtering equivalence |

## Extraction changes

Imports use `eeg_compare`. The parent model calls extracted standalone helpers;
their arithmetic is preserved. The TUEP-specific patient-training branch is
removed and that mode is rejected explicitly. The comparison's patient-aware
sampler will be implemented independently of TUEP manifests. Retained schedule
fields for the former patient mode are compatibility metadata only.
The retained trainer rebuilds its loader with uniform shuffling and does not
preserve custom samplers or collate functions. The common training adapter must
address this before claiming patient-balanced exposure.

Checkpoint model tags are `eeg_compare.waveform_sae` and
`eeg_compare.separable_sae`. They prevent an old experiment checkpoint from being
mistaken for a new comparison artifact. The numerical model and segment-training
path are otherwise retained. Checkpoint tags and source revision alone are not
sufficient provenance for a benchmark: the future runner must also bind the
dataset, split, preprocessing, and method configuration hashes.

The annotation parser retains raw interval boundaries and all 22 channel indices.
The old five-minute record rules, one-second scoring rules, ear-channel exclusion,
and event-specific pass/fail thresholds are not part of this copy.
Its original `parse_rec` checks are retained for compatibility. Stage 1 adds a
separate strict `parse_annotations` boundary with line numbers, finite/nonnegative
times and integral class/channel values; the new reader uses that API.

The spectral component preserves the original FFT-bin power calculation. Its
absolute features are implementation-specific log energies, not yet a calibrated
physical PSD. Keep sampling rate and FFT geometry fixed for a comparison, and
audit scaling before making amplitude claims.

## Excluded components

- `.git`, remotes, data listings, private patient manifests, caches, fitted models,
  results, logs, notebooks, and post-hoc analyses.
- Historical `run_*.sh` files, SLURM account settings, Hydra experiment configs,
  fixed run paths, and code that selects outputs from earlier campaigns.
- The class-mask and hinge variants. They are proposed experiments, not the
  unchanged separable reference.
- The full TUEP engine, Lightning orchestration, ICA machinery, foundation-model
  dependencies, and the previous patient classifier.
- The old TUEV reader. It imports the TUEP engine and fixes 20 channels and 256 Hz.
  The new benchmark uses its independent reader with an explicit montage,
  sampling rate, and five-second extraction policy.
- Binary patient-AUROC decision code. The primary task now needs six-class
  metrics and a patient-clustered evaluation of event predictions.

## Initial Extraction Validation

Completed locally on 2026-11-23, before the stage-1 additions:

- Python 3.12 syntax compilation of all 15 Python files, without executing them.
- Ruff lint and formatting checks for all source and test files.
- Mypy: no issues in the 12 package files, with the configured missing-library
  import handling. Repeat in the resolved HPC environment with its installed
  dependencies.
- Static resolution of 12 local imports; no imports of the former `src` package,
  root discovery, absolute machine paths, symlinks, or copied Git metadata.
- SHA-256 verification of all 10 source records and structural comparison of 115
  retained function bodies. The comparison allows the documented helper renames,
  removal of the patient-training branch, and the new sampling-mode guard; it
  ignores docstrings and formatting.
- Relative documentation links and package configuration checked.

`provenance.json` includes the validation counts and destination hashes. The new
helper fixtures cover row scaling, exact-rate resampling, annotation channel
mapping, malformed annotations, and rejection of the excluded trainer.

Runtime equivalence, the retained and new pytest suite, environment locking, and
a real EDF reader check still need to run on the HPC. No dataset was loaded and
no training or evaluation was performed during this extraction.

The copied tests contain small synthetic fits. They belong to the HPC validation
step even though they can use a CPU. A passed static check is not a passed model
regression or a completed paper reproduction.
