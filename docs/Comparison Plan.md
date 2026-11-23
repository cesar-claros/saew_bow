# EEG Waveform Comparison Plan

Created 2026-11-23. TUEV event classification is the first benchmark. Patient-level
epilepsy prediction is a later transfer study. The purpose is to determine which
changes improve waveform identification, prediction, and computational scaling
when the data and evaluation are controlled.

The three families are the original single-channel BOW method, the spatial and
temporal BOW extension, and the separable atom encoder. The existing proposal is
a reference arm. Improvements to it must have separate names and configurations.

## Questions and evidence

| Question | Evidence needed |
|---|---|
| Does learning shape rather than captured energy recover useful morphology? | Matched cosine versus reconstruction objectives; random and spectral controls |
| Does spatial structure help? | Same examples and head with temporal-only, regional, full multichannel, and separable representations |
| Do waveform transitions help? | Unigrams versus transitions with the same dictionary and classifier |
| Does multi-atom coding help? | R=1 versus R>1, followed separately by coefficient refinement |
| Can the method scale? | Data-efficiency curves, fitting cost, inference throughput, peak memory, and complete pipeline time |

Classification performance, event localization, and waveform interpretability are
different outcomes. A visually plausible atom is not evidence of a validated
event detector, and reconstruction error is not the classification endpoint.

## Reference definitions

| Family | Reference recipe | Status in this copy |
|---|---|---|
| Original BOW | Cosine assignment over shifts, aligned mean updates, independent class dictionaries, counts, TF-IDF, linear classifier [1] | Author implementation identified; adapter planned |
| Spatial and temporal BOW | Regional or multichannel atoms, hard tokens, count and transition features, feature selection, downstream classifier [2] | Paper specification available; exact implementation still to verify |
| Separable encoder | Unit-norm temporal and spatial factors; inner products; maximum over spatial patterns and positions; top-R selection; tied independent-window reconstruction | Core copied from the pinned local source |

The newer paper's TUEV experiments include whole-record and label-balanced
dictionary sources and their union. Label-balanced sampling uses label
information even if the dictionary loss itself is unsupervised. Its nonlinear
classifier and its count/transition representation must be treated as separate
factors. These details matter for attributing a gain [2].

For the original BOW code, pin the author repository at
`df37f52509cd7932f8587460ca8a9833d935793f` as an initial reference [3]. Audit the
settings used by the paper rather than assuming the newest defaults are identical:
the repository now includes SVD options. Do not silently replace cosine assignment
with a least-squares update while calling the result a reproduction.

For [2], the exact region map, multichannel normalization, union tokenization,
dictionary-training source, stopping rule, and selected TUEV configuration must
be resolved before claiming faithful reproduction. The inspected paper page does
not provide a verified author-code revision. A reconstruction from the paper must
be labeled a reimplementation with its assumptions recorded.

## Mathematical distinctions to test

The shared abstraction is a shifted dictionary match followed by a sparse or
discrete local code and an aggregate representation. This common structure does
not make the objectives equivalent. The following deductions use the local
specification and the cosine-matching definition of the reference methods.

Let $X_t$ be a candidate multichannel patch and let
$A_{k,d}=u_{k,d}w_k^\top$ have unit Frobenius norm. The separable score is
$s_{k,d,t}=\langle A_{k,d},X_t\rangle_F$. A shape-normalized score instead divides
by $\|X_t\|_F+\epsilon$. At a fixed location this denominator is common to all
unit templates, but it changes competition across locations. Consequently, raw
correlation can choose a higher-energy, poorer-shape match inside a pooling
window. Per-channel MAD scaling over the whole segment does not remove this
local energy preference. Local centering and polarity conventions must also be
matched explicitly.

For one unit template $a$ placed completely inside the scored support,
$\min_{\alpha\ge0}\|x-\alpha a\|_2^2$ is attained at
$\alpha=[a^\top x]_+$, with loss reduction $[a^\top x]_+^2$. Thus the proposal's
R=1 selection has a precise one-template reconstruction interpretation when all
channels are valid and the complete template is scored. Masked channels or
truncated templates change the effective norm; this argument then needs that
norm in the denominator. Cosine assignment and aligned averaging optimize a
different criterion, and cannot be equated to the proposal's Adam training merely
because both learn interpretable templates.

For several selected placed templates collected as columns of $B$, the loss is
$\|x\|^2-2\alpha^\top B^\top x+\alpha^\top(B^\top B)\alpha$. Using the encoder's
independent positive correlations as coefficients ignores off-diagonal Gram
terms. It is not generally the least-squares or nonnegative least-squares optimum.
This motivates an R>1 coefficient-refinement ablation, with its inference cost
reported, rather than an assumption that increasing R must improve recovery.

Each separable candidate has rank one across channels and time. Multiple spatial
patterns for the same waveform allow alternative topographies, but do not allow
independent channel delays or channel-specific waveform shapes in a single
candidate. The stored parameter count is $K(L+CD)$, versus $KDCL$ for the same
number of unrestricted candidates. These are storage counts, not matched model
capacities. Projection-first correlation costs approximately
$O(BKDT(C+L))$, compared with $O(BKDTCL)$ for direct unrestricted correlation,
but activation memory, filtering, and data I/O can still dominate. Scaling gains
must be measured on the same hardware.

Independent extended-window reconstruction scores overlapping target samples
more than once and does not enforce agreement between neighboring reconstructions.
It is not a global convolutional sparse-coding objective. The overlap multiplicity
and boundary treatment must be fixed before comparing residual errors. The
current count readout also collapses spatial winners; the representation learned
by the decoder is richer than the representation passed to the classifier.

Finally, morphology, polarity, and spatial interpretability need separate checks.
Jointly reversing both factors leaves a separable template unchanged, whereas
reversing only its signal polarity generally changes a ReLU code. Bipolar montage
dependencies can make spatial weights non-identifiable outside the observed
signal subspace; per-channel scaling further changes their physical meaning.
Compare observable templates and matched responses, not raw factor coordinates
alone. Any signed matching, spatial subspace constraint, soft selection, or
higher-rank/delay extension should be a separately named improvement.

## Two complementary comparisons

### Published pipeline comparison

Preserve each method's defining preprocessing, dictionary update, feature
construction, and classifier. Report the original BOW method adapted to TUEV as
an adaptation, not as a reproduction of its mouse-genotype results. For the newer
paper, compare with its reported TUEV numbers only after reconciling the corpus
version, example construction, montage, split, and evaluation code.

This comparison measures whole-pipeline performance. It does not isolate whether
the dictionary, normalization, transitions, or classifier caused a difference.

### Controlled component comparison

Use the same raw recordings, folds, event examples, physical signal exposure,
preprocessing contract, tuning budget, and downstream classifier. A shared
unlabeled training source and unigram features with regularized multinomial
logistic regression form the first controlled comparison. This makes the original
BOW shared-dictionary arm an explicit adaptation.

Use R=1 initially so every local window has at most one positive token. Define
an explicit null state when all responses are nonpositive; dropping nulls must
not create transitions across missing windows. Compare a
shared temporal dictionary applied channel by channel, full multichannel BOW
templates, and separable templates. Report temporal-only and spatially resolved
separable features separately. Add regional dictionaries after fixing their
electrode membership and aggregation rule.

## TUEV data audit before model fitting

The published TUEV protocol uses five-second contexts around event onset [2].
The previous project evaluated annotated channel-seconds within longer records.
Those are different examples and different tasks. Build a new canonical example
manifest rather than reusing the previous score tables.

1. Inventory the available TUEV version, EDF files, annotation rows, class counts,
   recordings, sessions, and recoverable patient identities. Reconcile the result
   with the benchmark preprocessing implementation; do not force published sample
   totals by duplicating examples.
2. Define one example precisely: the originating annotation, its start/stop,
   five-second context, channel mapping, and target. Store an annotation-group ID
   so overlapping contexts and duplicate marks can be traced. Fix policies for
   conflicting labels, multi-channel annotations, missing channels, and edges.
3. Verify the train/evaluation split by patient identity. Where the source hides
   patient IDs, document the grouping evidence and use conservative available
   groups; do not relabel sessions as verified independent patients.
4. Preserve the official evaluation set for final scoring. Draw development folds
   by patient from training data. All examples and overlapping contexts from a
   recording stay in one partition.
5. Validate annotation-to-channel mapping by names, never by assuming that a
   sorted model channel order equals the `.rec` channel order.
6. Keep unannotated time unlabeled. It can enter an explicitly unlabeled dictionary
   pool, but cannot become negative examples for detection automatically.

The proposed common signal contract is 22 canonical TCP derivations at 250 Hz,
with a fixed filter chain suitable for the available signals. Confirm this during
the audit. If 20 channels are necessary, apply the same documented exclusions to
every controlled arm and report that the published montage was changed.

Filtering should use recording context before cropping. Make notch, passband,
resampling, clipping, and normalization scope explicit. Common-average reference
belongs to electrode potentials, not an unexplained additional operation on
bipolar derivations. Store both the common physical signal and normalization
metadata. Per-window normalization versus record-level scaling is a controlled
factor rather than an accidental implementation difference.

## Training data and supervision

Keep three training-source conditions separate:

| Condition | Dictionary sees | Purpose |
|---|---|---|
| Whole recordings | Patient-balanced unlabeled training intervals | Primary shared-source comparison |
| Annotated contexts | The same training event examples across all methods | Effect of event-centered sampling |
| Class-balanced or class-specific | Equalized classes or separate class dictionaries | Label-informed representation comparison |

Apply each condition to all eligible methods with the same selected patients and
signal exposure. A union of two dictionaries has twice the vocabulary unless its
components are reduced; compare it with a single dictionary of the same total
capacity as well. Never use evaluation examples to fit even an unlabeled dictionary.

Patient-aware training must cap or weight contributions so long recordings do not
dominate by accident. Record unique signal seconds, repeated draws, patients,
optimizer updates, and complete passes. Equal epoch counts alone are not a matched
learning budget. Early stopping uses development patients only.
Within each development fold, fit dictionaries, IDF, scaling, variance selection,
and the head using that fold's training patients only. Validation patients may
select configurations or stopping points but may not enter dictionary updates.
Use nested patient folds when reporting a tuned cross-validation estimate, or
predeclare a fixed training/development split for tuning and reserve the official
evaluation set for the final estimate.
The retained segment trainer reconstructs its loader with uniform shuffling, so
a custom sampler on the incoming loader is not sufficient. Implement and verify
the common training adapter before assigning the patient-balanced condition.

## Features and controlled ablations

| Change | Fixed comparison | Question |
|---|---|---|
| Matching and learning objective | Same separable family, raw correlations/MSE versus normalized shape criterion | Is the learning target appropriate? |
| Spatial information | Temporal counts versus counts indexed by waveform and spatial winner | Does the readout discard useful structure? |
| Transitions | Same dictionary, head, and training split; unigrams then bigrams | Does order add predictive information? |
| Classifier | Same frozen features; logistic regression then random forest | How much gain comes from the head? |
| Multi-atom inference | R=1, then R=4, then R=4 with NNLS refinement | Does explaining overlapping content help? |
| Window objective | Independent patches versus global synthesis | Does the current reconstruction geometry limit recovery? |

Run a small pilot before selecting a finite search budget. Candidate dictionary
sizes are 16, 32, and 64 temporal waveforms; candidate durations are 0.5, 1, and
2 seconds. These are proposals, not source-paper defaults. Add short transient
atoms as a separately named extension if spike recovery remains a target.

Match physical durations, not sample counts. A paper window of length N with
an L-sample atom has N-L+1 valid placements. In the proposal, a pooling window P
has an extended target P+L-1. Setting N=P does not match either support or search
range. The common adapter must expose target length, allowed centers, stride,
padding, and scored samples separately. Do not silently truncate the five-second
example to make P divide its sample count.

For transitions, never concatenate unrelated channels, dictionary vocabularies,
sessions, patients, or event examples into one token stream. Count only transitions
within their declared stream. Report how many tokens and valid n-grams exist in
each five-second example. A long atom may leave too few tokens for a trigram.
Zero valid transitions is a real feature outcome, not a reason to cross boundaries.

R>1 gives simultaneous selections rather than one ordered token per window. Do
not turn atom-index sorting into a temporal sequence. Keep the first transition
comparison at R=1; evaluate a separately defined multihot or timed-event transition
representation if needed later.

Report K, D, K times D candidate templates, parameter count, feature dimension,
and nonzero placements per second. No single setting simultaneously equalizes all
these capacities. Use both fixed-vocabulary comparisons and resource curves.

## Metrics and interpretation

The proposed primary prediction metric is balanced accuracy over the six event
classes. Report macro F1, weighted F1, Cohen kappa, per-class precision/recall/F1,
and confusion matrices. Include the spectral control, random/frozen dictionaries,
and a simple class-prior predictor. If superiority is claimed, define the minimum
useful gain before scoring the held-out evaluation set.

Use paired patient-cluster bootstrap intervals for differences between methods.
For incomplete patient identifiers use the validated grouping unit and state the
limitation. Also report a patient-weighted metric so prolific patients cannot
silently determine the conclusion. Keep initialization variability distinct from
sampling uncertainty. Apply a declared multiplicity correction to confirmatory
contrasts; treat the remaining ablations as exploratory.

Report waveform stability across seeds and training subsets after matching atoms
up to permitted sign and shift conventions. Show representative held-out matches,
their spatial patterns, class association, artifact matches, and residuals. A
blinded expert review, if available, should use a fixed rubric and examples
selected without cherry-picking. Synthetic rank-1, delayed, overlapping, and
artifact events supply separate recovery tests with known ground truth.

Continuous event detection is a later evaluation. It needs onset matching,
deduplication, sensitivity at fixed false alarms per hour, and enough adjudicated
background to estimate that rate. Five-second classification accuracy alone does
not establish continuous detection performance.

## Scaling study

Use the same patient subsets at 10, 25, 50, and 100 percent of the available
training cohort, subject to class coverage. Separate the number of patients from
the number of minutes per patient. Fit all data-dependent transforms anew within
each subset; do not reuse a dictionary or IDF fitted on the full cohort.

Measure dictionary fitting, feature extraction, classifier fitting, end-to-end
time, peak CPU/GPU memory, and inference signal-seconds per wall-second. Compare
convergence curves as well as a fixed compute budget. Benchmark serially on the
same allocated hardware with warm-up and CUDA synchronization. Distinguish cached
signal computation from raw EDF I/O. Hardware-specific acceleration is allowed,
but report its cost and environment.

## Implementation sequence

Stage 1 code was implemented on 2026-11-23. Its identity-map contract, configurable
policies, HPC commands and pending acceptance checks are documented in
[TUEV Data Pipeline](TUEV%20Data%20Pipeline.md). Runtime and corpus validation remain
pending; the stage's empirical completion condition below has not been asserted.

| Stage | Deliverable | Completion condition |
|---|---|---|
| 0 | Selective standalone source copy and provenance | No imports or artifacts from the old repository; static validation |
| 1 | TUEV reader, manifest, split audit, configurable preprocessing, deterministic example loader | One recording checked against raw annotations; no group overlap; event timing and channel order verified |
| 2 | Original BOW and spatial/temporal BOW adapters; separable adapter with spatial winners | Source-pinned or assumption-documented references; synthetic assignment and update parity |
| 3 | Common unigram features, TF-IDF, feature selection, logistic head, six-class evaluator | Train-only fitting; identical example IDs across arms; no split leakage |
| 4 | Small development pilot | Meaningful fit, stable token exposure, measured runtime; finalize search budget |
| 5 | Frozen controlled study and published-recipe comparison | All predefined arms complete with paired metrics and provenance |
| 6 | Targeted improvements and scaling curves | Each variant changes one factor; benefits and failures are quantified on development data |
| 7 | Final held-out evaluation and waveform review | Configuration frozen; one complete final score table with uncertainty |

The first implementation should stop at a trustworthy small comparison before
adding the NNLS, supervision, or global-decoder improvements. Do not modify the
reference model while implementing those variants.

Every run should write its code revision, resolved configuration, environment,
input/split/preprocessing hashes, random seeds, training exposure, model and
feature schemas, predictions keyed by example ID, metrics, timing, and memory.
The new runner should require an explicit run ID and refuse to overwrite an
existing run. Model adapters must expose their normalization and event support;
the retained reference's `forward` method alone is insufficient for this contract.

## References

1. Cano Achuri et al. (2026), [Interpretable EEG biomarkers for neurological
   disease models in mice using bag-of-waves classifiers](https://doi.org/10.1088/1741-2552/ae4d8c).
2. Papastathopoulos-Katsaros et al. (2026), [Interpretable EEG biomarkers with
   bag-of-waves: Spatial and temporal waveform dictionaries for low-data
   regimes](https://arxiv.org/html/2607.22508v1), version 1.
3. [Original BOW author code](https://github.com/ajbrockmeier/bowaves-mice-genotyping/tree/df37f52509cd7932f8587460ca8a9833d935793f).
4. The local source document is `documentation/Separable Atom Encoder Mathematics.md`
   (updated 2026-11-14), especially sections 2, 4, 5, and 6. Its hash and the local
   PDF hash are in `provenance.json`. [Code Reuse](Code%20Reuse.md) identifies the
   copied implementation; the original documents remain in the source project.
