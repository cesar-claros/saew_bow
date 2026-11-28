# EEG Waveform Comparison Plan

Created 2026-11-23. Revised 2026-11-23 after a review of this plan against the three method sources, the stage 1 code, and the earlier TUEV inventory [5]. The last section records what changed and what is still unverified. TUEV event classification is the first benchmark. Patient-level epilepsy prediction is a later transfer study. The purpose is to determine which changes improve waveform identification, prediction, and computational scaling when the data and evaluation are controlled.

The three families are the original single-channel BOW method [1], the spatial and temporal BOW extension [2], and the separable atom encoder [4]. The existing proposal is a reference arm. Improvements to it must have separate names and configurations.

The study must explain a difference between the methods. A table of three scores cannot do that, because the families differ in at least eight design choices. The controlled comparison therefore places each published method in one design space and connects the methods with bridge arms. Two neighboring arms differ in one choice.

## Questions and evidence

| Question | Evidence needed |
|---|---|
| Which part of the learning rule matters: the position score, the weight of a window, or the optimizer? | The five-step path of learning rules on one atom structure; frozen, spectral, and phase controls |
| Does spatial structure help, and in which form? | The same tiling, readout, and heads with shared single-channel atoms, full multichannel atoms, and rank-1 atoms; a per-channel readout of single-channel atoms as the control for spatial readout without spatial atoms |
| Does a score read the event or the record? | Full, event-only, and context-only readouts; the displaced-context control; agreement with the annotated channels |
| Does the readout matter? | Hard counts versus amplitude sums; two heads from the start |
| Do waveform transitions help? | Unigrams versus transitions with the same dictionary and classifier |
| Does multi-atom coding help? | R=1 versus R>1, followed separately by coefficient refinement |
| Can the method scale? | Data-efficiency curves, fitting cost, inference throughput, peak memory, and complete pipeline time |

Classification performance, event localization, and waveform interpretability are different outcomes. A visually plausible atom is not evidence of a validated event detector, and reconstruction error is not the classification endpoint.

## Reference definitions

| Family | Reference recipe | Status in this copy |
|---|---|---|
| Original BOW [1] | Cosine assignment over shifts; mean of the aligned raw windows; one dictionary per class (200 atoms of 1 s, windows of 2 s without overlap); hard counts over 1 h segments; TF-IDF and L2 norm; multinomial logistic regression. The animals that train a dictionary do not train or test its classifier. | Author implementation identified; adapter planned |
| Spatial and temporal BOW [2] | Windows of twice the atom duration without overlap; z-score of each channel in each window (with CAR for multichannel atoms); cosine assignment; aligned mean; hard tokens; counts and n-grams; TF-IDF; selection of the features with the largest variance (300 for TUEV); random forest | Paper specification only; a reimplementation with recorded assumptions |
| Separable encoder [4] | Unit-norm temporal and spatial factors; inner products; maximum over spatial patterns and positions; top-R selection; tied independent-window reconstruction; Adam on the mean squared error | Core copied from the pinned local source |

The review established these facts about the references:

- [2] reports about 84,000 training and 29,000 evaluation windows. The earlier inventory counts 83,932 and 29,421 annotation rows, each exactly 1 s long [5]. So [2] uses one example per annotation row, which is one channel and one second.
- [2] gives no code. It does not state the corpus version, the channels, the filters, the atom length for TUEV, whether its TUEV atoms are multichannel, the balancing method, the forest depth, or the split that selected the configuration. Its kappa of 0.44 is the best of a sweep with a range of 0.36 to 0.44, from one run with no interval.
- [1] built each bag from 1,800 windows of a 1 h segment. A 5 s TUEV example holds 2 windows of 2 s per channel. The adaptation moves the method out of the range it was designed for, which is occurrence rates over long records. Reports must say so.
- [1] reports two comparisons that this plan reuses. Bag of spectra (BOS) clusters the normalized FFT magnitudes of the same windows. Soft counted waveforms (SCW) sum amplitudes without windows. BOS tied BOW on most tasks. SCW against BOW changed the strain accuracy from 70% to 91% and the DBA2 genotype accuracy from 86% to 63%. The readout was the largest factor in that paper.
- The `forward` method of the retained model returns the sum of the selected amplitudes of each atom, divided by the number of pooling windows. It is not an occurrence count, and it has no spatial index.

The TUEV experiments of [2] include whole-record and label-balanced dictionary sources and their union. Label-balanced sampling uses label information even if the dictionary loss itself is unsupervised. Its nonlinear classifier and its count and transition representation must be treated as separate factors. These details matter for attributing a gain [2].

For the original BOW code, pin the author repository at `df37f52509cd7932f8587460ca8a9833d935793f` as an initial reference [3]. Audit the settings used by the paper rather than assuming the newest defaults are identical: the repository now includes SVD options. Do not silently replace cosine assignment with a least-squares update while calling the result a reproduction.

A reconstruction of [2] from the paper must be labeled a reimplementation with its assumptions recorded. The region map, the multichannel normalization, the union tokenization, the dictionary-training source, and the selected TUEV configuration are assumptions until the authors supply them.

## Mathematical distinctions to test

The shared abstraction is a shifted dictionary match followed by a sparse or discrete local code and an aggregate representation. The deductions below use the local specification [4] and the definitions of [1].

### One loss, three learning choices

For R=1, the separable loss of one window is $\|X_m\|_F^2-(\max_k Z[k,m])^2$ ([4], section 6.12). Equation 5 of [1] is the same loss for a single-channel atom: one shifted, scaled template per window. The families then differ in three choices, and each choice can change alone.

1. **Position score.** The exact minimizer selects the atom and the shift with the largest raw inner product (equation 7 of [1]). [1] and [2] select the largest cosine (equation 8), which divides by the norm of the sub-window. At a fixed position, the two scores rank unit templates identically. They differ in the position that wins inside a window: raw correlation can select a sub-window with more energy and a poorer shape. Per-channel MAD scaling over a whole segment does not remove this local preference.
2. **Window weight.** Given the assignments, write each aligned sub-window as $y_i=\|y_i\|\,\hat y_i$. Every update of an atom is a weighted mean of the unit windows $\hat y_i$. The mean of normalized windows has the weight 1 and is optimal for the summed cosine. The mean of raw windows, which [1] uses, has the weight $\|y_i\|$. The least-squares update has the weight $\|y_i\|^2\cos_i$: its fixed point is $c\propto\sum_i\alpha_i y_i$ with $\alpha_i=\langle y_i,c\rangle$, the top singular vector that [1] describes. The gradient of the separable loss has the same stationary points. So the exponent of the window norm orders the three methods: [2] is near 0 because it z-scores each window, [1] is at 1, and the separable encoder is at 2.
3. **Optimizer.** BOW alternates the assignment and a closed-form mean over all windows. The separable encoder takes Adam steps on mini-batches. At the exponent 2 the two aim at the same stationary points. Their paths differ, and Adam rescales each coordinate.

In the separable family, the exponent 0 needs one change: divide the loss of each window by the energy of its target. This variant has the name `window-normalized loss`.

Expectation, stated before any run: the window weight has the largest effect. Earlier screens of this project fitted atoms to high-amplitude artifacts and to content below 1 Hz, which the exponent 2 rewards.

Local centering and polarity conventions must also be matched explicitly. Masked channels or truncated templates change the effective norm of a template, and the one-template argument then needs that norm in the denominator.

### Tiling

A BOW window of $N$ samples with an atom of $L$ samples has $N-L+1$ valid placements. A pooling window of $P$ positions has the support $P+L-1$. The two agree when $P=N-L+1$. BOW windows with the stride $N-L+1$, an option that [1] names, are the pooling grid. The published BOW tiling ($N=2L$, no overlap) tests about half of all placements and never a placement that crosses a window boundary. With windows of 2 s it also discards the last second of a 5 s example. Setting $N=P$ matches neither the support nor the search range.

### Several atoms per window

For several selected placed templates collected as columns of $B$, the loss is $\|x\|^2-2\alpha^\top B^\top x+\alpha^\top(B^\top B)\alpha$. Using the encoder's independent positive correlations as coefficients ignores the off-diagonal Gram terms. It is not generally the least-squares or nonnegative least-squares optimum. This motivates an R>1 coefficient-refinement ablation, with its inference cost reported, rather than an assumption that increasing R must improve recovery.

### Atom structure and cost

Each separable candidate has rank one across channels and time. Multiple spatial patterns for the same waveform allow alternative topographies, but do not allow independent channel delays or channel-specific waveform shapes in a single candidate. The stored parameter count is $K(L+CD)$, versus $KDCL$ for the same number of unrestricted candidates. These are storage counts, not matched model capacities. Projection-first correlation costs approximately $O(BKDT(C+L))$, compared with $O(BKDTCL)$ for direct unrestricted correlation, but activation memory, filtering, and data I/O can still dominate. Scaling gains must be measured on the same hardware.

With $D=C$ and $U_k=I_C$, the separable response is the response of the best single channel ([4], section 6.4). So a shared single-channel atom with one token per window, on its best channel, is a special case of the separable model with fixed spatial patterns. This gives a structural step from single-channel atoms to learned spatial patterns at an equal number of tokens.

### What "spatial" means under each normalization

[2] z-scores each channel inside each window (stated for its dementia data; its TUEV section says only "gain-stripped"). That removes the amplitude differences between channels, so a multichannel atom of [2] codes the shape and polarity of each channel. A rank-1 template $u w^\top$ codes the amplitude pattern across channels. The two methods use different definitions of spatial structure. A z-score of each channel in each window, applied to all arms as a shared input step, would remove what the rank-1 template codes.

The retained model scales each channel of each segment by its own median and MAD. On the 2 min segments of its source project, that removed electrode gain. On a 5 s example that a periodic pattern fills, it divides each channel by the amplitude of the event. The earlier TUEV protocol scaled per recording for this reason [5].

### Independent windows, polarity, and identifiability

Independent extended-window reconstruction scores overlapping target samples more than once and does not enforce agreement between neighboring reconstructions. It is not a global convolutional sparse-coding objective. The overlap multiplicity and boundary treatment must be fixed before comparing residual errors.

Morphology, polarity, and spatial interpretability need separate checks. Jointly reversing both factors leaves a separable template unchanged, whereas reversing only its signal polarity generally changes a ReLU code. Bipolar montage dependencies can make spatial weights non-identifiable outside the observed signal subspace; per-channel scaling further changes their physical meaning. Compare observable templates and matched responses, not raw factor coordinates alone. Any signed matching, spatial subspace constraint, soft selection, or higher-rank or delay extension should be a separately named improvement.

## Two complementary comparisons

### Published pipeline comparison

Preserve each method's defining preprocessing, dictionary update, feature construction, and classifier. Report the original BOW method adapted to TUEV as an adaptation, not as a reproduction of its mouse-genotype results.

For [2], use row-level scoring: each merged interval carries the number of its annotation rows as a weight (see the data audit), on the `keep` view of the primary manifest, because the published protocol keeps every row. The reproduction target is the kappa range of 0.36 to 0.44, not the value 0.44. If the evaluation set selected that configuration, the true value is lower. Compare our arms with each other first. The literature rows of Table II(b) of [2] come from other papers with their own protocols, so they give context only. Limit the effort on this arm, and record each assumption.

This comparison measures whole-pipeline performance. It does not isolate whether the dictionary, normalization, transitions, or classifier caused a difference.

### Controlled component comparison

All cells use the same recordings, folds, examples, input signal, tiling, number of atoms, atom duration, readout, heads, and tuning budget. A shared unlabeled training source, R=1, hard unigram counts, and two heads form the first comparison.

**Learning rules.** One path connects the three published rules. Each step changes one choice.

| Rule | Position score | Window weight exponent | Optimizer | Published point |
|---|---|---|---|---|
| `L1` | Cosine | 0 (normalized windows) | Alternating mean | Rule of [2] |
| `L2` | Cosine | 1 (raw windows) | Alternating mean | Rule of [1] |
| `L3` | Cosine | 2 (least squares) | Alternating update | Bridge |
| `L4` | Raw | 2 | Alternating update | Exact alternating minimization of the common loss |
| `L5` | Raw | 2 | Adam | Rule of the separable encoder |

Note of 2026-11-24 on the published points. The table takes them from the text of the papers. The author checkout of [1] that the build read (the mouse pipeline, local HEAD b7716e1) always calls its clustering update with normalized sub-windows, which is the update of `L1`, and with one start. The update function also has the raw mean of `L2`. Which update made the published results of [1] is a question for the authors. The cells do not change, because the path holds both rules. The parity gate of stage 2 checks both updates against the author functions.

**Atom structures.** `S1` is one set of 1 x L atoms that all channels share. `SF` is a set of full C x L atoms. `SR` is a set of rank-1 atoms with D spatial patterns.

| Structure | Rules | Fits | Purpose |
|---|---|---|---|
| `S1` | `L1` to `L5` | 5 | The learning path at the lowest cost. `S1-L2` is the shared-dictionary adaptation of [1]. |
| `SF` | `L1`, `L5` | 2 | `SF-L1` is the reimplementation of the rule of [2]. `SF-L5` is the full-template parent of the separable model. |
| `SR` | `L5`, and `L5` with the window-normalized loss | 2 | `SR-L5` is the separable reference. The second cell is a named variant. |

This gives nine dictionary configurations at one K and one atom duration. Conditions on the cells:

- `SF-L5` and `SR-L5` must score the same loss support. The retained `WaveformSAE` scores the central P samples, and the separable reference scores the extended window. Either both cells use the central support, or the full-template cell gets the extended loss. Record the choice.
- Rank-1 atoms with a shared waveform have no closed-form update here, so `SR` has no alternating cells.
- If one step of the `S1` path has an effect above the margin, repeat that step in `SF`. Declare the repeat before any read of `SF`.

**Tokens.** Use R=1 initially so every local window has at most one positive token. Define an explicit null state when all responses are nonpositive; dropping nulls must not create transitions across missing windows. `S1` has two tokenizations: one token per channel and window, and one token per window on the best channel. The second has the token count of `SF` and `SR`. Add regional dictionaries after fixing their electrode membership and aggregation rule.

## TUEV data audit before model fitting

The published TUEV protocol uses five-second contexts around event onset [2]. The previous project evaluated annotated channel-seconds within longer records. Those are different examples and different tasks. Build a new canonical example manifest rather than reusing the previous score tables.

The earlier inventory of TUEV v2.0.1 gives these counts [5]. The audit of this repository must reproduce them under its own rules.

| Quantity | Training | Evaluation |
|---|---|---|
| Files | 359 | 159 |
| Independent units | 290 patients | 80 sessions, with no patient identity |
| Hours of signal | 100.5 | 48.3 |
| Annotation rows, each 1 s on one channel | 83,932 | 29,421 |
| `spsw` marked seconds | About 77, in 27 patients | 38, in 9 sessions |
| `gped` marked seconds | About 1,170, in 30 patients | 464, in 21 sessions |
| `pled` marked seconds | About 980, in 29 patients | 428, in 20 sessions |

The 290 training patients of this table are patient folders. A scan of the corpus in November 2026 found 20 groups of EDF files with identical bytes, all in the training set, in 16 pairs of folders with adjacent codes and with different annotation files. The pipeline merges each group into one recording with the annotation rows of both copies, and each pair of folders into one cluster, so the training set has at most 274 independent patients ([TUEV Data Pipeline](TUEV%20Data%20Pipeline.md), section "Identical EDF files"). The counts by patient in this plan come from the earlier inventory, which treated each folder as a patient; the manifest replaces them. That inventory also drew its folds by folder, so a pair of folders could lie in two folds.

Class and patient are confounded. Of the 30 training patients with `gped`, 1 has a background mark. For `pled` it is 4 of 29, and for `spsw` 11 of 27. 107 patients have background marks only. In the earlier pilot, one band-power number read `gped` at 0.85 across patients and at 0.56 inside a record [5]. So a six-class score rewards a feature that separates patients, and all three methods gain from it. The readouts and controls below measure this.

1. **Inventory.** List the corpus version, EDF files, annotation rows, class counts, recordings, sessions, and recoverable patient identities. Reconcile the result with the table above; do not force published sample totals by duplicating examples.
2. **Example unit.** The primary unit is the merged interval (`unit="interval"`): the rows of one second hold the same multichannel signal. Each interval carries the number of its source rows as a weight, which the manifest can derive from the stored annotation lines. Report the row-weighted score next to the primary score, because only the row-weighted score compares with the literature. Store an annotation-group ID so overlapping contexts and duplicate marks can be traced.
3. **Conflicts.** A conflict is an interval that overlaps an interval of another class on any channel. The rule is `conflict_policy="background_yields"`, fixed on 2026-11-24 after the audit of the first manifest (below): a background interval that meets another class is removed, an interval of another class stays when background is all that it meets, and two other classes that meet remove each other.
4. **Identical files.** Report each group of EDF files with identical bytes: its files, the clusters it joins, the rows and classes of each annotation file, and the conflicts between the files. A group across the official sets is a leak and stops the build.
5. **Identities.** Training files have patient identities. Evaluation files have a random session index only. The evaluation set is therefore clustered by session, and its disjointness from the training patients rests on the corpus README. Record the kind of each cluster. Do not relabel sessions as verified independent patients.
6. **Folds.** Draw development folds by patient from training data, stratified by the classes that each patient holds. A fold of 58 patients that is drawn without stratification holds 5.4 of the 27 `spsw` patients on average, with a standard deviation of about 2 (hypergeometric). All examples and overlapping contexts from a recording stay in one partition. Preserve the official evaluation set for final scoring.
7. **Channel mapping.** Validate annotation-to-channel mapping by names, never by assuming that a sorted model channel order equals the `.rec` channel order.
8. **Unannotated time.** Keep it unlabeled. It can enter an explicitly unlabeled dictionary pool, but cannot become negative examples for detection automatically. Displaced contexts (see the readouts) are a control kind of their own. They carry the label of their source event for the control only, and no model that is scored on events trains on them.

### Audit of the manifests

The first complete build is `artifacts/tuev-manifest-002` (2026-11-24, `configs/tuev.toml`, merge rule for identical files, drop rule for conflicts). The build with the chosen conflict rule is `artifacts/tuev-manifest-003`. `scripts/audit_manifest.py` makes every number of this section from a manifest; it reads no signal. The counts that do not depend on the conflict rule are identical in the two builds.

**The counts match the corpus.** The build read 113,353 annotation rows, which is 83,932 training rows plus 29,421 evaluation rows, the counts of the earlier inventory and of [2]. The rows form 13,495 intervals, each exactly 1 s long: 10,062 in the training set and 3,433 in the evaluation set. After the merge of the 20 groups of identical files there are 498 recordings, 274 training patients and 80 evaluation sessions. The folds of the first build hold 55, 54, 55, 55 and 55 patients, with 4 or 5 `spsw` patients, 5 `gped` patients and 4 or 5 `pled` patients in each. The folds of `tuev-manifest-003` hold 56, 55, 55, 54 and 54 patients. The assignment changed between the builds, because the stratification reads the classes of the kept examples and the conflict rule changes which examples are kept.

**One interval in five is a conflict.** 2,710 intervals overlap an interval of another class on some channel. 1,015 of them have their partner on the identical second; the others have a partner that is shifted by less than 1 s. Inside one annotation file the partner is nearly always an artifact: an artifact on some channels with background (352 training and 376 evaluation artifact intervals), with `gped` (360 `gped` intervals), with `pled` (160) or with eye movement (139) on other channels. Two epileptiform classes meet inside one file in 9 training intervals. So a conflict is mostly two things that are present in one second on different channels, not a labeling error.

The table gives the intervals that each candidate rule keeps, after the 771 contexts that cross a recording boundary are removed. `drop` removes every conflicting interval. `background_yields` removes a background interval that meets another class and keeps an interval of another class when background is all that it meets. `precedence` keeps an interval when every class it meets ranks lower (epileptiform classes, then eye movement, then artifact, then background). `keep` keeps all.

| Set | Class | All | `drop` | `background_yields` | `precedence` | `keep` |
|---|---|---|---|---|---|---|
| Training | `spsw` | 141 | 105 | 105 | 109 | 134 |
| Training | `gped` | 2,416 | 1,986 | 1,986 | 2,342 | 2,361 |
| Training | `pled` | 2,001 | 1,631 | 1,635 | 1,833 | 1,870 |
| Training | `eyem` | 529 | 334 | 372 | 509 | 510 |
| Training | `artf` | 1,699 | 864 | 1,171 | 1,171 | 1,565 |
| Training | `bckg` | 3,276 | 2,694 | 2,694 | 2,694 | 3,015 |
| Training | Total | 10,062 | 7,614 | 7,963 | 8,658 | 9,455 |
| Evaluation | `spsw` | 46 | 40 | 40 | 46 | 46 |
| Evaluation | `gped` | 827 | 802 | 802 | 802 | 803 |
| Evaluation | `pled` | 719 | 682 | 682 | 683 | 684 |
| Evaluation | `eyem` | 124 | 114 | 116 | 124 | 124 |
| Evaluation | `artf` | 528 | 122 | 457 | 457 | 485 |
| Evaluation | `bckg` | 1,189 | 827 | 827 | 827 | 1,127 |
| Evaluation | Total | 3,433 | 2,587 | 2,924 | 2,939 | 3,269 |

The `drop` rule removes 75% of the evaluation artifact intervals and leaves 122 of them in 20 sessions. With `background_yields` the evaluation set has 457 artifact intervals in 41 sessions. The other classes of the evaluation set change little between the rules.

**Decision of 2026-11-24: the rule is `background_yields`.** A background mark on one channel does not contradict another class on another channel, so it is no reason to remove an event from a task that reads the whole montage. The rule adds one assumption and no rank of the classes. The primary examples are then 7,963 training intervals and 2,924 evaluation intervals. `precedence` and `keep` are sensitivity checks. `tuev-manifest-003` holds these examples: 10,887 kept events, and the audit script finds no difference between the kept events of the manifest and its own count for the rule. A manifest with another conflict rule has the same example IDs but other folds, for the reason given above. So the examples of the published row-level protocol (the `keep` rule) come from this manifest as a second view of its shards, which still hold the excluded examples with their reasons: `Partition(conflict_view="keep")`. On the first manifest this view selects 9,455 training and 3,269 evaluation events, the `keep` column of the table.

**Identical seconds limit a score very little.** With the `keep` rule, 351 of 9,104 training seconds and 122 of 3,147 evaluation seconds carry more than one class. The best balanced accuracy that any classifier of the whole montage can then reach is 0.98 on intervals and 0.99 on rows. The low published scores do not come from this limit.

**The duplicated recordings were annotated twice.** The two annotation files of the 20 merged recordings hold 2,974 and 3,627 rows. Of their 1,773 intervals, 44 are in both files with the same second and class, and 470 are conflicts. In time, the two files mark the same moment for 124 s and only one file marks it for 704 s. Where both mark, the classes are the same in 88 s (71%). No second that one file marks `spsw` is `spsw` in the other file: the other file marks `pled` (10 s) or `gped` (6 s). This comparison counts a mark on any channel, the 20 recordings are a small sample, and nothing tells that the two files come from independent annotators. With these limits, it is the only measure of annotation agreement that the corpus gives: which seconds get a mark depends on the annotation pass, and `spsw` against the periodic classes is the weakest distinction.

**The effective sample is small.** In `tuev-manifest-003` the evaluation set has 40 `spsw` intervals in 8 sessions, and the 3 largest sessions hold 62% of them. The 7,963 kept training events lie in 741 groups of overlapping contexts, and the 2,924 kept evaluation events in 362 groups. A marked `spsw` second has 4.6 rows in the training set and 12.3 rows in the evaluation set, so the annotation style differs between the sets and the row-weighted score counts an evaluation spike more than twice as much. 1,250 of the 10,887 kept contexts hold a mark of another class inside their 5 s. The number was 564 of 10,201 under the `drop` rule: the 686 intervals that the chosen rule restores all have a background mark next to them.

**Displaced controls.** 10,879 of the 10,887 kept events have a control; two recordings have no time that is 10 s from every mark.

**Timing and channel mapping hold in the tables.** All 13,495 contexts start at the sample that the rule gives, 2 s before the onset. No kept example has padding or leaves its recording. No displaced control is closer than 10 s to a mark; the smallest distance is 10.008 s. All 10,956 rows of the channel matrices (498 recordings, 22 derivations) hold +1 on the first electrode of the derivation and -1 on the second, found by name. All recordings have the `REF` reference and a native rate of 250 Hz, so this corpus does not use the resampler. After the merge the corpus holds 142.4 hours of signal.

**Signal audit of the training patients (2026-11-24).** `scripts/audit_signals.py` read 1,605 of the 7,963 training events of `tuev-manifest-003` (300 per class and all 105 `spsw`) and their displaced controls. Its thresholds were fixed in the script before the run. Four of its five checks pass and one fails.

- **Band.** The median share of the power above 40 Hz is below 1e-7, so the low-pass works. Between 8% and 21% of the power of a context lies below 1 Hz. The 1 Hz high-pass has a transition band from 0 to 1 Hz, and the spectrum of a 5 s context leaks about 0.4 Hz, so this measure overstates the share.
- **Scale.** The median per-channel standard deviation of a background context is 1.04 robust units, and 1.03 in its control.
- **Timing and channel order.** For eye movement, the probability that a marked channel has more energy than another channel is 0.92 in the marked second, 0.66 to 0.70 in the other four seconds and 0.49 in the control. The four channels with the most energy are the four frontal pairs. For `spsw` the value is 0.68 in the marked second, 0.54 to 0.56 in the other seconds and 0.55 in the control. The annotation times and the channel order thus agree with the signal.
- **The failed check.** For artifact the value is 0.57 in the marked second and 0.56 in the control, against the fixed rule of at least 0.60 with a gain of 0.05. It is flat over the five seconds. The rule assumed that an artifact raises the energy of its marked channels in its marked second, in the 1 to 30 Hz band. The data do not support this assumption. The same code passes for eye movement, so the failure describes the artifact class in this band and not the timing of the pipeline. The earlier pilot read artifacts best with a measure on 1 to 70 Hz [5], so the low-pass could be what removes the artifact energy. The follow-up audit on the wide band tested this explanation (below).
- **Periodic classes.** `pled` has 0.66 to 0.69 in all five seconds and 0.63 in the control, and `gped` has 0.62 against 0.55. A periodic pattern is on the same channels in the whole record, so its marked second does not stand out. The earlier pilot found the same.
- **Clip.** 32% of the eye-movement examples and 21% of the artifact examples hold a sample at the clip of 20 robust units, and the 95th percentile of their largest absolute value is the clip itself. The clip cuts the peak of a large eye movement, so its value is open again. The follow-up audit on `configs/tuev_noclip.toml` gives the amplitudes that are needed to set it (below).

**Follow-up audits (2026-11-24).** The same 1,605 events were read from two more manifests, `tuev-manifest-wide-001` (1 to 70 Hz with the notch) and `tuev-manifest-noclip-001` (the primary band without the clip).

- **The wide band changes little.** For artifact the value in the marked second rises from 0.57 to 0.60, with 0.56 in the control, and it stays flat over the five seconds (0.58 to 0.62). The explanation that the low-pass removes the artifact energy holds for a small part only. The values of the other classes do not change: eye movement 0.92, `spsw` 0.67, `gped` 0.61, `pled` 0.69. So the audit gives no reason to change the primary band.
- **What an artifact mark is in this corpus.** In both bands the marked channels of an artifact are only a little more energetic than the other channels, in every second of the context and also 10 s away from any mark. The energy of the marked second equals that of its neighbors. An artifact mark is thus not a short event on its channels. Two readings fit: the artifact lasts much longer than one second, or the scaling hides it, because the robust scale of each channel over the recording absorbs a noise that is present for most of the recording. The audit cannot separate the two; a single scale for all channels of a recording, the declared second scaling level, can.
- **Amplitudes without the clip.** In robust units, the largest absolute value of an eye-movement example has a median of 15, a 99th percentile of 44 and a maximum of 56. The maxima of `spsw` and `gped` are 38 and 41. Artifact examples reach 2,426, and 18 of 300 exceed 100. One `pled` example and one `bckg` example exceed 100. A clip at 20 touches 95 of 300 eye-movement examples; a clip at 50 touches one of them, no `spsw` and no `gped` example, and 24 of 300 artifact examples.

Three consequences follow for the design. The primary band stays 1 to 30 Hz and the wide band stays the second level. For the artifact class, expect the `context` and `displaced` readouts to be near the `full` readout in both bands; a claim about artifact waveforms needs the scaling test first. And the clip must move from 20 to a value that leaves the physiological classes whole and still bounds the artifacts.

**Decision of 2026-11-24: the clip is 50 robust units**, in the two study configurations. It touches 1 of 300 eye-movement examples, no `spsw` and no `gped` example, and 24 of 300 artifact examples. The examples and the folds do not change with the clip, but the signals do, so the study needs a new manifest.

**The manifest of the study: `tuev-manifest-clip50-001` (2026-11-24).** Its configuration differs from that of `tuev-manifest-003` in the clip only. Its 24,374 example rows (13,495 events and 10,879 displaced controls) have the same IDs, recordings, clusters, classes, statuses and row weights, and its fold assignments and recordings are identical. Its table audit equals that of `tuev-manifest-003` line by line: 10,887 kept events, no difference from the conflict rule, and no failed table check. Its signal audit shows what the audit without a clip predicted: a sample at the clip in 1 of 300 eye-movement examples, 24 of 300 artifact examples, 1 `pled` example and 2 `bckg` examples, and in no `spsw` or `gped` example. The values of the marked channels do not change with the clip. The same four checks pass, and the artifact check fails as before.

**The first smoke checks.** Two smoke checks on the HPC read 8 event examples and 8 displaced controls of `tuev-manifest-003` (training partition of fold 0) without an error. Each example has 22 channels, 1,250 real samples and no flat channel. The 16 examples come from 8 recordings and hold two classes only, `bckg` and `pled`. The checks record shapes and not values; the signal audit above measures the values.

### Signal contract

- **Montage and rate.** 22 canonical TCP derivations at 250 Hz. If 20 channels are necessary, apply the same documented exclusions to every controlled arm and report that the published montage was changed. Common-average reference belongs to electrode potentials, not an unexplained additional operation on bipolar derivations.
- **Band.** The primary band is 1 to 30 Hz. One wide band, 1 to 70 Hz with the 60 Hz notch, is a declared second level. A high-pass at 0.1 Hz with no low-pass repeats the condition of an earlier result of this project, in which the separable code explained only content below 1 Hz. The earlier TUEV pilot found 1 to 30 Hz best for spikes and no gain from the wide band [5]. A z-score per window does not remove drift inside a window, so the band matters for all families. The primary configuration has no notch, because the 30 Hz low-pass removes the 60 Hz line.
- **Filtering.** Filter with recording context before cropping. Make notch, passband, resampling, and clipping explicit.
- **Scaling.** All arms receive one input, scaled per recording: one robust scale (median and MAD) per channel and recording, computed in the loader. A single scale per recording for all channels is a second level, because it keeps the full amplitude pattern across channels. Note of 2026-11-26: this second level is under review before stage 3. A scale per channel and recording keeps a rank-1 event rank-1; it changes the weight of a channel, the meaning of a spatial pattern, and the agreement of a pattern between recordings. The alternative is one recording scale times a fixed channel profile, which keeps what a single scale keeps and removes the fixed effect of the montage. The scale audit (`scripts/audit_scales.py`, [TUEV Data Pipeline](TUEV%20Data%20Pipeline.md), section "Scale Audit") decides between three outcomes by rules that are fixed before its run. Result of 2026-11-26: the outcome is `per_channel_with_profile_level`. The scale per channel stays primary. The second level is one recording scale times the fixed channel profile, with a cap on loud channels, and it replaces the plain single scale. The two inputs differ by a pattern of about 28% per channel that belongs to the patient; the marked events raise the scale of their channels by about 12% at most. The loader clips the scaled signal at 50 robust units. The first value, 20, was the row clip of the retained model; the signal audit found a sample at that clip in 32% of the eye-movement examples, and the amplitudes without a clip gave the value 50. The adapter turns the internal scaling of the retained model off and records this as a deviation from the reference. Removal of gain inside a window belongs to a method (cosine does it), not to the shared input.
- **Information set and edges.** Every arm reads the same 1,250 real samples of an example, with valid placements only. For a 1 s atom, zero padding touches 249 of 1,250 positions (20%) and 2 of 5 pooling windows, against 0.85% on the 2 min segments of the source project. The adapter masks the responses at the L-1 edge positions, as section 3 of [4] allows, and reports wins near the edges.
- **Common tiling.** The tiling of the controlled comparison is the pooling grid with P positions per window. P must divide 1,250, so P is 125, 250, or 625 samples. With P=250 the third window is the labeled second. The BOW cells use the same windows of centers. The common adapter must expose target length, allowed centers, stride, padding, and scored samples separately. Do not truncate the five-second example.

## Training data and supervision

Keep three training-source conditions separate:

| Condition | Dictionary sees | Purpose |
|---|---|---|
| Whole recordings | Patient-balanced unlabeled training intervals | Primary shared-source comparison |
| Annotated contexts | The same training event examples across all methods | Effect of event-centered sampling |
| Class-balanced or class-specific | Equalized classes or separate class dictionaries | Label-informed representation comparison |

Apply each condition to all eligible methods with the same selected patients and signal exposure. A union of two dictionaries has twice the vocabulary unless its components are reduced; compare it with a single dictionary of the same total capacity as well. Never use evaluation examples to fit even an unlabeled dictionary.

The annotated-context and class-specific conditions need cross-fitting. [1] learns its dictionaries on one half of the animals and trains the classifier on other animals, because features differ between dictionary individuals and new individuals. A spike dictionary can store its 77 training seconds; the supervised reference of the earlier protocol read 0.836 on its training folds and 0.581 held out [5]. So the training features of a patient come from a dictionary that did not see that patient (K-fold cross-fitting inside the training patients). For the whole-recording condition, measure the difference between in-sample and cross-fitted training features once.

Patient-aware training must cap or weight contributions so long recordings do not dominate by accident. Record unique signal seconds, repeated draws, patients, optimizer updates, and complete passes. Equal epoch counts alone are not a matched learning budget. Early stopping uses development patients only. Within each development fold, fit dictionaries, IDF, scaling, variance selection, and the head using that fold's training patients only. Validation patients may select configurations or stopping points but may not enter dictionary updates. The development folds select configurations, and the official evaluation set gives the final estimate; label a tuned development score as optimistic.

The retained segment trainer reconstructs its loader with uniform shuffling, so a custom sampler on the incoming loader is not sufficient. Implement and verify the common training adapter before assigning the patient-balanced condition.

**Decision of 2026-11-26: an exposure plan, not a sampler.** The balance between patients is fixed once in a stored list of training units per fold and source, drawn without replacement with a quota per patient. Every cell reads the same list, so the alternating cells, which make no draws, and the gradient cells see the same data, and the retained trainer needs no change. The quotas are 5 minutes of signal per patient for the whole-recording source and 20 events per patient and class for the annotated contexts, with 10 minutes and 50 events as second levels. On the training set of the study manifest, the signal has 204 effective patients of 274 and the kept events have 64; the quotas raise these to 274 and 217. Weights and draws with replacement are not used: weights mix with the window weight of the least-squares rules, and repeated draws repeat the few events of small patients. The heads get their balance by weights in the loss ([Slice Protocol](Slice%20Protocol.md), section 8).

## Features, readouts, and controlled ablations

### Readouts of each fitted arm

The label belongs to 1 s of the 5 s context. A bag over the whole context gives that second 1 of 5 tokens at P=250. Each fitted arm is therefore read in four ways, with no new fit.

| Readout | Windows that enter the features | Question |
|---|---|---|
| `full` | All windows of the context | The published form of the task |
| `event` | The windows of the labeled second | Does the event carry the score? |
| `context` | The windows whose support holds no sample of the labeled second | Does the surrounding signal carry the score? |
| `displaced` | A 5 s context of the same recording, at least 10 s from any mark | Does the record alone carry the score? |

The `displaced` head trains and tests on displaced contexts with the folds of the events. If `context` or `displaced` is near `full` for a class, the arm reads the record for that class, and no waveform claim follows. A periodic pattern can also run through unmarked time, so `displaced` is an upper limit of the record effect. Report the number of tokens of each readout.

### Feature forms

- **Hard counts** per atom are the primary form. **Amplitude sums**, which the retained `forward` returns, are an ablation.
- `S1` has pooled counts (K features) and counts per channel (K x C features). `SR` has counts per waveform (K) and counts per waveform and spatial winner (K x D). The model adapter must return the atom, the spatial winner, and the position of each placement.
- TF-IDF, scaling, and variance selection fit on the training patients of a fold.

### Ablations

| Change | Fixed comparison | Question |
|---|---|---|
| Learning rule | `L1` to `L5` in `S1` | Which of the position score, the window weight, and the optimizer matters? |
| Atom structure | `S1`, `SF`, `SR` at `L5`; `S1`, `SF` at `L1` | Do joint templates help? |
| Spatial readout | `S1` pooled versus per channel; `SR` by waveform versus by waveform and spatial winner | Does the readout carry the spatial gain? |
| Count form | Hard counts versus amplitude sums | Does amplitude add to occurrence? |
| Classifier | Logistic regression and random forest on the same frozen features | Does a ranking hold on both heads? |
| Transitions | Same dictionary, head, and split; unigrams then bigrams | Does order add predictive information? |
| Multi-atom inference | R=1, then R=4, then R=4 with NNLS refinement | Does explaining overlapping content help? |
| Window objective | Independent patches versus global synthesis | Does the current reconstruction geometry limit recovery? |
| Dictionary source | The three conditions of the previous section | Does event-centered or label-informed sampling help? |

In [2], logistic regression and the random forest gave a balanced accuracy of 0.43 and 0.44, but a kappa of 0.31 and 0.44. So both heads run from the start, with one class weighting for all arms, and a ranking of representations must hold on both.

### Controls

- A class-prior predictor.
- The retained spectral features.
- **Bag of spectra** [1]: the same windows and K, with k-means on normalized FFT magnitudes. It is the matched control for the question of whether the time-domain shape adds to the spectrum.
- **Phase-randomized atoms:** keep the amplitude spectrum of each learned atom, draw a random phase, and encode again. This one control applies to all three families. If the score holds, the learned morphology carries nothing beyond its spectrum.
- **Frozen dictionaries of two kinds** for each structure: Gaussian draws and random data windows. BOW starts from data windows and the separable reference has its own start, so one kind alone does not compare. The earlier TUEV stage read crop dictionaries at 0.788 and random waveforms at 0.770 for spikes [5].
- **HYDRA** on the same examples, as an anchor and not as a contrast. It is not in this copy; add it by a recorded extraction.

### Search space

Run a small pilot before selecting a finite search budget. Candidate dictionary sizes are 16, 32, and 64 temporal waveforms; candidate durations are 0.5 and 1 s. These are proposals, not source-paper defaults. A 2 s atom leaves one token per stream under the published BOW tiling and two windows at P=625, so it is a separately named extension. Short transient atoms are a second named extension if spike recovery remains a target. Match physical durations, not sample counts.

Each family gets the same number of random configurations from declared ranges. Report the expected best development score against the number of configurations [6]. The separable family has more settings, so an equal effort per setting would favor it.

For transitions, never concatenate unrelated channels, dictionary vocabularies, sessions, patients, or event examples into one token stream. Count only transitions within their declared stream. Report how many tokens and valid n-grams exist in each five-second example. A long atom may leave too few tokens for a trigram. Zero valid transitions is a real feature outcome, not a reason to cross boundaries.

R>1 gives simultaneous selections rather than one ordered token per window. Do not turn atom-index sorting into a temporal sequence. Keep the first transition comparison at R=1; evaluate a separately defined multihot or timed-event transition representation if needed later.

Report K, D, K times D candidate templates, parameter count, feature dimension, and nonzero placements per second. No single setting simultaneously equalizes all these capacities. Use both fixed-vocabulary comparisons and resource curves.

## Metrics and interpretation

The primary prediction metric is balanced accuracy over the six event classes, on merged intervals. Report macro F1, weighted F1, Cohen kappa, per-class precision, recall, and F1, confusion matrices, and the row-weighted version of each score. Also report a patient-weighted metric so prolific patients cannot silently determine the conclusion.

**Resolution of the benchmark.** In the evaluation set, `spsw` has 38 marked seconds in 9 sessions, so one sixth of the balanced accuracy depends on 9 sessions. A rough estimate gives a 95% interval of about ±0.04 to ±0.05 for one arm. It assumes a binomial error for `spsw` at a recall of 0.3 and a standard deviation of recall between sessions of 0.3 for `gped` and `pled`. Most published methods lie between 0.40 and 0.47. Paired differences are tighter. The out-of-fold predictions of the slice (stage 3) give the measured width.

**Confirmatory contrasts.** At most three, declared before any read of the evaluation set:

1. Learned against frozen (random data windows) for the reference cell of each family: `S1-L2`, `SF-L1`, `SR-L5`.
2. The best dictionary cell against bag of spectra.
3. The separable reference `SR-L5` against `SF-L1` under the common conditions.

The margin of a useful gain is fixed after the slice and before any read of the evaluation set. It may not be smaller than the half-width of the paired interval that the slice measures. Apply a declared multiplicity correction to these contrasts. Only the arms of these contrasts and the anchors are scored on the evaluation set. All other ablations stay on development data and are exploratory.

**Intervals.** Use a paired cluster bootstrap for differences between methods: patients on development data, sessions on the evaluation set. Draw inside strata of class presence, so that each resample holds each class. On development data, pool the out-of-fold predictions of all training patients before scoring; do not average five fold scores. Keep initialization variability distinct from sampling uncertainty, with at least three seeds per cell.

**Agreement with the annotated channels.** TUEV marks channels, which gives a ground truth for the spatial claim. For the winning placement of the labeled second, rank the channels by the absolute spatial weight (`SR`), by the energy of each row of the template (`SF`), or by the response of each channel (`S1`). Score the annotated channels against the other channels with an AUROC per example, on the examples whose marks do not cover all channels. The comparison value is the same score from the energy of each channel in that second.

**Pre-stated readings.**

| Result on development data | Consequence |
|---|---|
| All dictionary cells are within the margin of bag of spectra and of the frozen dictionaries | Learned morphology adds nothing to prediction on TUEV. Stop the predictive comparison and keep the synthetic and stability results. |
| A BOW cell is above the bar and the separable cell is not | The path names the factor. Add that factor to the separable family as a named variant. |
| `context` or `displaced` is near `full` for a class | Report record classification for that class, with no waveform claim. |
| The interval of a paired difference is wider than the margin | The benchmark cannot rank the methods. Report the comparison as descriptive. |

Prediction recorded on 2026-11-23: about 50% for the first row, from the earlier TUEV stage in which crop dictionaries did not beat random waveforms or theta power [5].

Report waveform stability across seeds and training subsets after matching atoms up to permitted sign and shift conventions. Show representative held-out matches, their spatial patterns, class association, artifact matches, and residuals. A blinded expert review, if available, should use a fixed rubric and examples selected without cherry-picking.

Continuous event detection is a later evaluation. It needs onset matching, deduplication, sensitivity at fixed false alarms per hour, and enough adjudicated background to estimate that rate. Five-second classification accuracy alone does not establish continuous detection performance.

## Synthetic benchmark

The synthetic benchmark runs before any corpus fit, on all nine cells. It is the only place where waveform identification has a known answer. Write the predictions below into the run record before the run. The metrics are atom recovery (largest absolute cosine with a planted template, over shifts) and event detection (precision and recall of the placements) on independent draws.

| Condition | Prediction |
|---|---|
| Events of variable amplitude in a background with high-amplitude artifacts | The window weight exponent 0 recovers more planted atoms than the exponent 2. |
| An artifact burst close to an event inside one window | The cosine score places the atom on the event more often than the raw score. With stationary noise the two scores agree. |
| Events with a delay across channels | `SF` recovers the template. `SR` loses in proportion to the delay. |
| Rank-1 events and few training examples | `SR` recovers with fewer examples than `SF`. |
| Two overlapping events in one window | R=1 codes one of them. R>1 without refinement overshoots when the templates are coherent. |
| Events of inverted polarity | D=1 does not respond. D>1 responds in part. |
| Background only | The learned and the frozen dictionaries give the same downstream score. This is the null. |

**Outcome (read on 2026-11-26; protocol, section 15).** Nine cells, 14 levels, 5 seeds. The gates of the protocol pass. Of its eight predictions, three hold, one is opposite, three are undecided, and the null fails.

- **Atom structure: support.** With a delay of 16 samples between channels, the full template recovers more than the rank-1 template (0.68 against 0.49). With 10 training segments, the rank-1 template recovers more (0.83 against 0.72). At no delay and 200 segments the two are equal.
- **Optimizer: not a factor.** The alternating update and Adam reach the same recovery, and the alternating fit takes about one hundredth of the time.
- **Window weight: the expectation of this plan did not hold.** With variable amplitudes and rare artifacts, equal weights recover less than the least-squares weight (0.81 against 0.91), because the weight by energy gains from the strong events. With a burst of high energy behind every event, read after the results, the picture turns: the cells with the raw score and the least-squares weight do not learn the templates, and the window-normalized loss helps. So the window weight matters, and its direction depends on what holds the energy.
- **Position score: the largest effect, at inference.** With the planted dictionary and bursts, cosine tokens find 0.70 of the events and raw tokens 0.07. The prediction on the learned cells is undecided, because neither cell learns the templates in that level.
- **The null fails.** A dictionary learned on background matches a spike with a slow wave at about 0.75. A learned waveform needs a background dictionary as its null before any claim on morphology.
- **Number of atoms.** With 8 atoms for 4 templates the cells lose a template that they all recover with 16.

These results show that a mechanism exists on ideal data. They do not show its size on EEG.

The exact protocol is [Synthetic Benchmark Protocol](Synthetic%20Benchmark%20Protocol.md). The user confirmed it on 2026-11-24, with every test and run on the HPC. It fixes the generator, the cells, the measures, the gates and the decision rule of each prediction. It keeps the table above and makes four changes, each with its reason in the protocol:

- The window-weight prediction has two forms: the absolute form of the table, and the loss of recovery from the base condition to the artifact condition. The second is the test of the mechanism, because equal weights also give atoms to the background in proportion to its share of the windows.
- The artifacts of the first condition come from random families and are not noise alone. Noise windows of high energy do not add up in an atom.
- A condition without events has no downstream task. The null tests the recovery measure in its place.
- Recovery from a common start is a result. The gates are the parity with the author code, the loss identity, the detection by the planted dictionary, and the stability of the planted dictionary under each rule.

## Scaling study

Use nested patient subsets at 10, 25, 50, and 100 percent of the available training cohort, stratified by class presence, with at least five chains of subsets. Ten percent of 290 patients holds about 3 `spsw` patients, so one chain is not an estimate. Separate the number of patients from the number of minutes per patient. Fit all data-dependent transforms anew within each subset; do not reuse a dictionary or IDF fitted on the full cohort.

Measure dictionary fitting, feature extraction, classifier fitting, end-to-end time, peak CPU and GPU memory, and inference signal-seconds per wall-second. Compare convergence curves as well as a fixed compute budget. Benchmark serially on the same allocated hardware with warm-up and CUDA synchronization. Distinguish cached signal computation from raw EDF I/O. Hardware-specific acceleration is allowed, but report its cost and environment.

## Implementation sequence

Stage 1 code was implemented on 2026-11-23. Its identity-map contract, configurable policies, HPC commands and pending acceptance checks are documented in [TUEV Data Pipeline](TUEV%20Data%20Pipeline.md). The test suite passed on the HPC on 2026-11-24, and the audits of the manifests and of the signals are recorded above. The manifest with the clip of 50 passes the table audit. Its signal audit passes four of its five fixed checks and fails the artifact check, in both bands and with every clip. The plan reads this failure as a property of the artifact marks and not as a defect of the pipeline (section "Audit of the manifests"). **Stage 1 is complete: the user accepted it on 2026-11-24 with this one recorded exception.** The single scale per recording and the patient-balanced training sampler stay open; they are not conditions of this stage. The review added the changes of the next section to stage 1, and they are implemented.

**Stage 2 is complete: the user accepted it on 2026-11-26** with the results of section 15 of the [Synthetic Benchmark Protocol](Synthetic%20Benchmark%20Protocol.md). The gates of the protocol passed on the HPC, and each of its eight predictions has a result by its fixed rule. Three points come from the stage as proposals for the slice of stage 3 and wait for a decision: cosine against raw tokens on one dictionary, a number of atoms with a margin, and a dictionary learned on background as the null of any claim on morphology. The points that were set aside at the end of stage 1 are due now, because stage 3 processes real data: the scale of a recording and the patient-balanced training sampler. One more point is open since the build of stage 2: the author checkout of [1] uses the update of `L1`, and the slice names `S1-L2` as the adaptation of [1].

**Stage 3 has run (2026-11-28) and waits for the acceptance of the user.** All gates of the [Slice Protocol](Slice%20Protocol.md) pass, and every fold repeats bit by bit. The margin of a useful gain is 0.033. The five best arms lie inside 0.026 with the logistic head: the two learned single-channel arms read 0.460 and 0.458, an untrained full-template dictionary 0.453 and the spectral control 0.441, and their order changes with the head. Learning adds about 0.02 to its own start, which is below the margin. No level (atoms, token score, tiling, input, exposure) replaces a primary value. A control context with no event reads 0.32 to 0.37, about two thirds of the gain over the prior, and the three epileptiform classes get no waveform claim by the rule of the protocol. Three of the seven predictions hold. My recommendation, in section 18.9 of the protocol, was not to build the full grid of stage 4 on this target before a test that removes the recording. The user chose that test on 2026-11-28. Its design, rules and predictions are section 19 of the protocol; it is built and has not run.

| Stage | Deliverable | Completion condition |
|---|---|---|
| 0 | Selective standalone source copy and provenance | No imports or artifacts from the old repository; static validation |
| 1 (complete, 2026-11-24) | TUEV reader, manifest, split audit, configurable preprocessing, deterministic example loader, and the changes of the next section | Counts reconciled with the inventory table; conflicts counted by class; no group overlap; event timing and channel order verified for all recordings by the table audit and by the signal audit, in place of a check of one recording by hand |
| 2 (complete, 2026-11-26) | Synthetic benchmark for the nine cells, by the [Synthetic Benchmark Protocol](Synthetic%20Benchmark%20Protocol.md) | Predictions recorded before the run; assignment and update parity with the author code of [1]; the six gates of the protocol; recovery table and the result of each prediction |
| 3 (protocol confirmed 2026-11-26; steps 1 and 2 of 3 passed their gates on the HPC on 2026-11-27, results in sections 17 and 18 of the protocol; step 3 ran on 2026-11-28; the stage waits for the acceptance of the user and the decision on stage 4) | The slice, by the [Slice Protocol](Slice%20Protocol.md): common tiling, hard-count features, two heads, six-class evaluator; spectral features, bag of spectra, `S1-L1` and `S1-L2`, and the frozen dictionaries | Out-of-fold predictions on the training patients; the interval width, the size of the record effect, and the bar are measured; the decision on stage 4 is recorded |
| 4 | The remaining cells and their adapters | Each neighbor differs by one choice; identical example IDs across cells; train-only fitting; no split leakage |
| 5 | Frozen controlled study and published-recipe comparison | All predefined cells complete with paired metrics and provenance; margin and contrasts fixed |
| 6 | Targeted improvements and scaling curves | Each variant changes one factor; benefits and failures are quantified on development data |
| 7 | Final held-out evaluation and waveform review | Configuration frozen; one read of the declared contrasts and anchors, with uncertainty |

The slice of stage 3 comes before the reimplementation of [2]. It is cheap, and its result decides whether the full set of cells can rank anything. Do not modify the reference model while implementing variants.

### Changes that the review required in stage 1

Implemented on 2026-11-23 and 2026-11-24. The suite of 133 tests passed on the HPC on 2026-11-24. [TUEV Data Pipeline](TUEV%20Data%20Pipeline.md) describes each item.

1. `configs/tuev.toml` holds the primary band (`highpass_hz = 1.0`, `lowpass_hz = 30.0`, no notch) and `normalization = "mad"` with `normalization_scope = "recording"`. `configs/tuev_wide.toml` holds the wide band. Both set `clip = 50.0` (first 20, the row clip of the retained model; changed on 2026-11-24 after the audit without a clip).
2. `[splits] eval_cluster = "session"` declares the cluster kind of the evaluation set. Recordings, examples and fold assignments carry a `cluster_id` with its kind as a prefix. The patient ID of an evaluation file can be empty.
3. `[splits] stratify = "class_presence"` gives folds by iterative stratification on the classes that each patient holds.
4. Each example has a `row_weight`. The manifest metadata has a `conflict_report` with the counts by class under both conflict policies, in intervals and in rows.
5. `[examples] displaced_controls = true` adds one `displaced` example per kept event, with its source example recorded. `Partition(kind="displaced")` selects them.
6. `eeg_compare.adapters.models` has `ComparisonSeparableSAE` and `ComparisonWaveformSAE`: internal scaling off, the edge mask, initial patches from inside the segment, and `placements` with hard counts, amplitude sums, and the atom, spatial winner and position of each placement. The retained model files are unchanged.
7. `README.md` states that the `forward` output is an amplitude sum.
8. `[splits] duplicate_content = "merge"` handles EDF files with identical bytes, which the first build on the HPC found. `scripts/make_tuev_identities.py` writes the identity CSV from the folder layout.

Still open in stage 1: the single scale per recording for all channels (the second scaling level), the patient-balanced training sampler, and all runtime and corpus checks.

## Run records

Every run should write its code revision, resolved configuration, environment, input, split, and preprocessing hashes, random seeds, training exposure, model and feature schemas, predictions keyed by example ID, metrics, timing, and memory. The new runner should require an explicit run ID and refuse to overwrite an existing run. Model adapters must expose their normalization and event support; the retained reference's `forward` method alone is insufficient for this contract.

## Review record (2026-11-23)

What the review changed:

- The question on the learning objective is now three separate choices (position score, window weight, optimizer), connected by the path `L1` to `L5`.
- The controlled comparison is a set of nine cells in which neighbors differ in one choice.
- The data audit states the example unit, the row weights, the conflict count, the session clusters of the evaluation set, and stratified folds.
- The signal contract fixes the band, the scaling per recording, the information set, and the common tiling.
- Label-informed dictionaries need cross-fitting.
- New readouts and controls: event, context, and displaced readouts; channel agreement; bag of spectra; phase-randomized atoms; frozen dictionaries of two kinds; HYDRA as an anchor.
- New rules for the evaluation read: at most three contrasts, a margin fixed after the slice, and no evaluation read for exploratory ablations.
- The synthetic benchmark and the slice come before the adapters.

What is still unverified:

- No code ran for this review.
- [2] was read through an automated extraction of its HTML page. An item marked as not stated can be in a figure.
- `tuev-manifest-003` has the chosen conflict rule, and its audit agrees with the counts of this plan. The signal audit passes for the band, the scale, the timing and the channel order, and fails its fixed rule for the artifact class in both bands. The clip is 50 robust units since 2026-11-24, and `tuev-manifest-clip50-001` is the manifest of the study. The single scale per recording is not implemented.
- The interval width of the benchmark is a rough estimate until the slice measures it.
- The marked-second counts of the training set follow the rules of the earlier protocol [5].
- The counts by patient come from 290 folders. After the merge of identical files the number of independent training patients is at most 274, and the counts by class can change.
- The comparison adapters pass their tests on the HPC, but no model has been fitted on the corpus with them.

## References

1. Cano Achuri et al. (2026), [Interpretable EEG biomarkers for neurological disease models in mice using bag-of-waves classifiers](https://doi.org/10.1088/1741-2552/ae4d8c).
2. Papastathopoulos-Katsaros et al. (2026), [Interpretable EEG biomarkers with bag-of-waves: Spatial and temporal waveform dictionaries for low-data regimes](https://arxiv.org/html/2607.22508v1), version 1.
3. [Original BOW author code](https://github.com/ajbrockmeier/bowaves-mice-genotyping/tree/df37f52509cd7932f8587460ca8a9833d935793f).
4. The local source document is `documentation/Separable Atom Encoder Mathematics.md` (updated 2026-11-14), especially sections 2 to 7. Its hash and the local PDF hash are in `provenance.json`. [Code Reuse](Code%20Reuse.md) identifies the copied implementation; the original documents remain in the source project.
5. `documentation/event_corpus_dictionary_protocol.md` of the source project, sections 18 to 23: the inventory of TUEV v2.0.1, the pilot, and stage 2a. The document remains in the source project.
6. Dodge et al. (2019), Show Your Work: Improved Reporting of Experimental Results, EMNLP-IJCNLP 2019.
