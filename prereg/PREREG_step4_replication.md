# Pre-registration: confirmatory replication of Step 4

**Written and committed BEFORE any training run.** No analysis choice in this
document may change after inspecting any base-versus-adaLN replication output
from drug_split_0 or drug_split_2. Those splits were previously characterised
under base-model analyses - their SNR and base DPC-50 appear in section 7 - so
they are NOT pristine external test sets. No adaLN result from either split has
been seen, and neither drove the architecture or this rule, so they remain
prospective confirmation sets.

Author: Noha Samir Ismail. Date: 2026-07-29.

## 1. Why this is needed

drug_split_1 was **selected because it had the best measured signal-to-noise
ratio** (2.12x, 41.0% detectable conditions) - a choice made using test data.
It then carried the Step 1 noise floor, the failure diagnosis, the architecture
selection, the Step 4 gate, the fingerprint ablation, the eta^2 mechanism
analysis and the leave-one-drug-out influence check.

Each individual test was implemented carefully. The problem is cumulative:
repeated scientific decisions against the same test labels create a material
risk of adaptive overfitting at the research-program level. Numerical
overfitting is not demonstrated; the risk is enough to warrant confirmation.

**drug_split_1 is therefore reclassified as the development split.** Step 4's
result stands as a development-split finding. This document specifies the
confirmatory test on two unseen-compound splits that did not drive any
architectural choice.

## 2. What is frozen

| item | value |
|---|---|
| Code commit | recorded at launch; no edits to model or training code after |
| Architecture | adaLN as committed in `4657ab0`, `--use_adaln True` |
| Training args | within a split, base and adaLN launch commands differ ONLY in output/log identifiers and `--use_adaln False` vs `True`. Same data files, steps, optimizer, batch size, seeds, GPU type, preprocessing, checkpoint rule. A normalized command diff is saved before launch. |
| Seeds | 0,1,2,3,4 per arm |
| GPU | h100, pinned, matching the Step 4 runs |
| cuDNN | deterministic, benchmark off, as `set_seed()` sets |
| Steps | 100,000 (`lr_anneal_steps`), from scratch, not fine-tuned |
| Checkpoint | `model_0.9999.pt` (EMA) |
| Evaluation | `leakage_test_base_sciplex_array.py`, unmodified |
| Primary metric | chance-corrected DE-overlap Jaccard |
| Gate statistic | Welch 95% CI on the arm difference, per split |

## 3. Design

Two splits x two arms x five seeds = **20 training runs**, ~30 min each,
parallelisable. Both splits are unseen-compound: 0 test drugs appear in
training.

The existing `results_base_corrected/base_drug_split_{0,2}` checkpoints are
**not** used as the base arm. They predate the Step 4 protocol - untyped GPU
allocation, no cuDNN determinism, single checkpoints rather than five seeds -
so using them would confound architecture with training protocol.

## 4. Endpoints

**Primary, and the only gated quantity:** chance-corrected DE-overlap Jaccard,
adaLN minus base, assessed **separately on each split**.

    J_cc = (J - c) / (1 - c)

**Seed-level estimand.** For each checkpoint, the primary seed value is the
ARITHMETIC MEAN of condition-level raw Jaccard over all eligible test
conditions in that split. The arm estimand is the mean of five adaLN seed
values minus the mean of five base seed values. No median seed summary is used
for the primary endpoint. (Step 1 used per-seed medians and Step 4 per-seed
means, which produced an apparent discrepancy that was purely definitional.)

**Missingness.** Every primary Jaccard value must be finite for every eligible
condition in all ten models within a split. The analysis asserts identical
condition labels, order and finite masks across all ten, and ABORTS before
printing arm summaries if any value is undefined. Conditions and seeds are
never dropped post hoc. Exactly five completed checkpoints per arm are
required; existence of five filenames is not sufficient.

**Split-specific active-universe reference.** Let `M_s` be the number of panel
genes whose true treated-minus-control condition mean is nonzero in at least
one eligible condition of split s, computed from the same stored `true` and `C`
vectors used for evaluation. The reference is the exact expected Jaccard
between two independent random top-50 sets drawn from `M_s` genes:

    c_s = E[ I / (2K - I) ],  I ~ Hypergeometric(M_s, K, K),  K = 50

    c_s = sum_i [ i / (2K - i) ] * C(K,i) C(M_s-K, K-i) / C(M_s, K)

computed deterministically from the hypergeometric pmf, NOT by simulation. Note
E[I/(2K-I)] != E[I]/E[2K-I]; the ratio-of-expectations approximation
underestimates it by about 0.002 at these values.

This reference uses true test responses to define the active universe, so it is
oracle-informed. Call it the **split-specific active-universe random-set
reference**, not a chance floor.

Chance-corrected Jaccard is `(J - c_s)/(1 - c_s)`. Because the transform is
positive affine, **the gate is algebraically identical on raw and corrected
Jaccard.** Both are reported. Corrected values are for interpretation; RAW
differences are used for cross-split magnitude comparison, since each split has
a different `c_s` and comparing corrected effects would mix the architecture
effect with a different affine scale.

**Secondary, reported but NOT gated.** Exploratory, unadjusted for
multiplicity:

- **B-C on top-50 DEGs (Pearson).** Seed summary is the condition-level MEAN,
  matching Step 4. The median is reported separately.
- **DPC-50** = `corr( (yhat_i - C_i)[G_i], (y_i - C_i)[G_i] )` where `G_i` is
  the condition's true top-50 set by `|y_i - C_i|`. This is an ORACLE-SELECTED
  evaluation subset: the true response chooses the genes. Seed summary is the
  condition-level MEDIAN, matching prior DPC reporting.
- **Paired per-condition difference.** Statistic: per-condition adaLN-minus-base
  difference on seed-averaged values. Reported centre: MEDIAN. Bootstrap unit:
  unique test drug, drawn with replacement, member condition indices
  CONCATENATED (a set-membership test would drop duplicates and narrow the
  interval). 5000 replicates, `RandomState(0)`, percentile 2.5-97.5. Seed means
  are treated as FIXED, so this quantifies condition-level variation for these
  ten models, not uncertainty for a future retraining.

**Supporting:** exhaustive two-sided 5-vs-5 label permutation over all
C(10,5) = 252 allocations, no random sampling. Statistic: **absolute difference
in arm means**. Minimum attainable two-sided p is 2/252 = 0.0079. Under unequal
arm variances this tests exchangeability rather than equality of means, which
is why it supports rather than gates.

## 5. Uncertainty

Each split's Welch CI uses that split's own observed arm SDs. **The Step 1
noise floor (Jaccard SD 0.0088) was measured on drug_split_1 and does not
transfer.** No separate noise-floor run is required: five seeds per arm supply
the dispersion estimate directly.

Arms are treated as **unpaired**. Base seed i and adaLN seed i share a numeric
seed but not an optimisation trajectory once the parameterisations differ.

## 6. Success criteria

Four mutually exclusive outcomes, decided per split on the Welch 95% CI:

1. **Full replication** - both splits have a CI lower bound above zero.
2. **Partial replication** - exactly one split has a lower bound above zero AND
   the other's CI includes zero.
3. **Directional contradiction** - at least one split has an upper bound below
   zero. Reported as such even if the other split is significantly positive.
   This is NOT partial replication.
4. **No replication** - neither split is significantly positive and neither is
   significantly negative.

Requiring both splits is an intersection-union rule: the joint claim is
accepted only when both individual nulls are rejected, so no multiplicity
correction is needed. It is conservative at five seeds per arm, which is
intended.

Do not pool the splits post hoc to rescue a partial result.

**Direction gates; magnitude does not.** Effect sizes legitimately vary with
split difficulty. The observed differences will be reported alongside
drug_split_1's +0.0601 (chance-corrected) for comparison, but a smaller effect
that clears the CI criterion is a successful replication.

## 7. Stated in advance: drug_split_0 is the harder test

Recorded now so a weaker result there is interpretable rather than explained
afterwards.

| split | SNR | detectable conditions | base DPC-50 |
|---|---|---|---|
| drug_split_1 (development) | 2.12x | 41.0% | 0.0364 |
| drug_split_2 | 1.89x | 40.1% | 0.0528 |
| drug_split_0 | **1.37x** | **35.8%** | **0.0047** |

drug_split_0's base DPC-50 was indistinguishable from zero (t ~ 0.29).

These are CONTEXTUAL DESCRIPTORS and do not change the section 6
classification. A failure on drug_split_0 is a failed or partial replication,
full stop. Insufficient signal is a **pre-specified candidate explanation**, to
be argued from evidence if it arises - not a pre-authorised excuse.

**No signal-stratified secondary analysis is pre-specified.** If one is wanted
later, it must be defined - vehicle-null procedure, threshold, condition- or
drug-level, metric, uncertainty method - BEFORE the new arm results are
inspected, in a superseding document. It may not be introduced afterwards.

Sources for the table above: SNR and detectable fraction from
`null_results/null_<split>.txt` via `deg_permutation_null.py`; base DPC-50 from
`benchmark_results/benchmark_metrics.json` via `eval_benchmark_metrics.py`.

## 8. What would change the conclusion

- Neither split replicates: the Step 4 result is development-split specific and
  the chapter reports it as such.
- Both replicate: the architecture claim generalises across unseen-compound
  splits, and Step 4 becomes a confirmed finding rather than a selected one.
- Direction reverses on either split with a CI excluding zero: report it. Do
  not exclude the split.

**No architecture, hyperparameter, metric or analysis change is permitted after
inspecting either new split.** If something must change, this document is
superseded by a new pre-registration and the affected split is thereafter a
development split too.

## 9. Provenance requirements

Every run writes to the tracked `provenance/` directory, not to a gitignored
output dir. Each record must contain:

- git commit and `git status --porcelain` at launch
- SLURM job ID, node name, and GPU type from `nvidia-smi`
- torch, CUDA and cuDNN versions
- checkpoint output path
- the exact training command
- seed, split, arm
- training and test data paths with SHA-256 hashes
- launch-script and evaluation-script hashes
- checkpoint SHA-256 after training
- this pre-registration's commit hash
- condition count and number of unique held-out drugs
- runtime assertion that zero test drugs appear in training

**The launch aborts unless `git status --porcelain` is empty**, apart from
explicitly listed ignored output paths. Every arm-split-seed pair uses a unique
output directory and aborts if it already contains a checkpoint. No run is
resumed. A technical SLURM failure may be rerun with the same seed and
unchanged command; no completed seed is excluded or replaced based on its
metric, and no seed may be dropped as an outlier. Architecture is detected from
each checkpoint at evaluation and all ten must log the expected value.

The GPU type is mandatory. An untyped `--gres=gpu:1` request already produced a
real confound in this project: the base and adaLN latent extractions landed on
A100 and h100 respectively, leaving GPU architecture aligned with arm on the
latent side of the mechanism analysis.

## 10. What this does not establish

**Capacity versus conditioning.** adaLN adds 3,957,120 parameters (+7.3%).
Replication confirms the architecture improves the endpoint; it does not
isolate multiplicative modulation as the cause. That requires the
parameter-matched control arm - identical modulation pathway conditioned on the
timestep embedding alone - which is a separate experiment.

That control isolates **semantic-latent-dependent** modulation, not
drug-dependent modulation, because `z_sem` is built from control-cell
expression plus drug fingerprint and therefore mixes drug identity with
baseline cell state.

**Mechanism.** The eta^2 attenuation result is provisional pending the
uniform-hardware recompute and is not part of this replication.

**Generalisation beyond sci-Plex.** Two additional splits of one dataset.

---

## 11. Technical amendment, 2026-07-29: base arm omits the flag

**Committed before any valid base-arm result exists. No replication endpoint
has been evaluated.**

Section 3 required that within a split, base and adaLN launch commands differ
only in output identifiers and `--use_adaln False` versus `True`. That is
**withdrawn**. The base arm now OMITS `--use_adaln` entirely.

**Reason.** `train_squidiff.py`'s `parse_args` builds its parser with
`type=type(value)`. For a Boolean default this is `type=bool`, and
`bool('False')` is `True` because every non-empty string is truthy. The correct
converter `str2bool` exists in `script_util.py` and is used by
`add_dict_to_argparser`, but `train_squidiff.py` has its own loop that bypasses
it. Passing `--use_adaln False` therefore trained adaLN.

All 10 base-labelled runs of array 16930893 are adaLN (12 `adaln_mod` tensors,
58,186,374 parameters against an expected 54,229,254). They are archived and
are NOT used - neither as the base arm, nor as additional adaLN seeds, since
they share seed identifiers with the adaLN-labelled runs on the same splits and
are not independent draws.

**Verified before amending:** omitting the flag resolves to Boolean `False`
through the exact parser path (`use_adaln = False, type=bool`).

**The 10 adaLN runs are retained.** They used the intended architecture, their
post-training assertion passed, their provenance is complete, and no
replication metric has been inspected. Only the technical mechanism for
obtaining the pre-registered base setting has changed; model and optimisation
source remain frozen at the same commit.

Omission also reproduces Step 4's base protocol exactly:
`train_sciplex_seedvar.sh` predates the flag and never passed it.

**train_squidiff.py is deliberately NOT fixed before the recovery**, so the
recovered base runs share identical training source with the retained adaLN
runs. The parser defect is repaired afterwards as a separate commit.

**Unchanged:** architecture, seeds, GPU type, steps, data, metric hierarchy,
success criteria, and the rule that no analysis choice may change after
inspecting any base-versus-adaLN replication output.

**Incidental finding, recorded for Methods.** Training is bit-reproducible
WITHIN a node but not across nodes of the same GPU model. drug_split_0's base
and adaLN runs both landed on cn274 and produced byte-identical checkpoints;
drug_split_2's split across cn274 and cn270 and differ. All 20 reported
NVIDIA H100 NVL. Cross-node variation is far below the seed noise floor and
does not affect any comparison, but bit-reproducibility may only be claimed
within a node.

---

## 12. Amendment, 2026-07-30: ridge comparison on the confirmation splits

### Disclosure of partial knowledge

**The point estimates are already visible.** After the replication analysis
completed, the reported median DPC-50 values were adaLN 0.0878 on drug_split_0
and 0.0666 on drug_split_2, against ridge medians of 0.0719 and 0.0638 recorded
earlier in `paired_ridge_all_splits.json`.

This is therefore **NOT a blind pre-registration** and must not be described as
one. What it fixes, before any test statistic, interval or p-value is computed,
is the inferential procedure: the estimand, the pairing, the condition
alignment, the uncertainty method and the success criterion. That is weaker than
a blind pre-specification and stronger than analysing after the fact.

### Why it is not covered by section 4

Section 4 named chance-corrected DE-overlap as the sole gated endpoint and
listed DPC-50 as secondary and exploratory. No ridge comparison was specified
for the replication at all. This amendment adds one as an **explicitly
exploratory** analysis. It cannot alter the section 6 replication verdict, which
is already decided.

### Estimand

Per condition, DPC-50 is

    corr( (pred_i - C_i)[G_i], (y_i - C_i)[G_i] ),  G_i = top-50 by |y_i - C_i|

`G_i` is chosen using the true response, so DPC-50 is an **oracle-selected
evaluation subset** for both methods equally.

- **ridge:** `ridge_fp__dpc50` from `baseline_results/per_condition_<split>.npz`
- **adaLN:** computed from the five replication vectors and averaged per
  condition across seeds

Both arms are evaluated on the same conditions, so the comparison is
**paired at the condition level**.

Primary statistic: **median per-condition difference (adaLN minus ridge)**.
Medians, matching how both quantities were previously summarised.

### Alignment requirements — abort, do not adjust

1. Condition labels in the baseline `.npz` and the replication vectors must be
   **identical and in the same order**. If they differ, abort and investigate;
   do not reindex, intersect or subset.
2. Condition counts must match the replication analysis: 444 for drug_split_0,
   455 for drug_split_2.
3. Any non-finite DPC-50 on either side aborts. Conditions are not dropped.
4. Drug counts must match those established per split: 37 and 38.

### Uncertainty

Drug-clustered percentile bootstrap on the median difference: clusters drawn
with replacement, member condition indices **concatenated**, 5000 replicates,
`RandomState(0)`, 2.5-97.5 percentiles.

Reported alongside, so seed variation is visible: the same paired median
computed **separately for each of the five adaLN seeds**, with its range. Ridge
is deterministic and has no seed variation, so the arms are asymmetric in this
respect and the bootstrap interval treats the five adaLN models as fixed.

Also reported: the fraction of conditions on which adaLN exceeds ridge.

### Success criterion

**adaLN exceeds ridge** if the median paired difference is positive with a
drug-clustered 95% interval excluding zero, **on both splits**.

One split only is reported as such. No pooling to rescue a partial result.

### What this cannot establish

- It does not estimate the uncertainty of retraining a fresh adaLN model; the
  five seeds are treated as fixed and only condition-level variation is
  quantified.
- It concerns DPC-50 only. adaLN remains below control-mean parity on B-C and
  below the expression-only heuristic on DE-overlap on every split measured.
- It is exploratory relative to the Step 4 gate and does not change it.
- Ridge here is ridge on drug fingerprints as implemented in
  `baseline_models_sciplex.py`, not a tuned or optimised linear baseline.

---

## 13. Amendment, 2026-07-30: parameter-matched capacity control

### This amendment IS blind

No time-only model has been trained. Unlike amendment 12, nothing about the
outcome is visible, so this is a genuine prospective pre-registration.

### Question

adaLN adds 3,957,120 parameters (+7.3%) in a rank-128 modulation pathway. The
confirmed Step 4 result supports "the adaLN architecture improves DE-overlap"
but not "conditioning the modulation on the semantic latent is what improves
it". An extra pathway of that size could help through capacity alone, or through
modulation driven by the timestep embedding only.

### Design: a third arm

`use_adaln_timeonly`. The modulation MLP is IDENTICAL in parameter count,
topology, rank-128 bottleneck, zero-initialisation and placement. Only its input
differs:

- **adaLN (existing):** `time_dense(emb) + zsem_dense(z_sem)`
- **time-only (new):** `time_dense(emb)` alone

The base additive path `h + time_dense(emb) + zsem_dense(z_sem)` is retained
unchanged in both. Only what drives the scale-and-shift differs.

Five seeds (0-4), **drug_split_1**, from scratch, h100, cuDNN deterministic,
100,000 steps, all other arguments identical to `train_sciplex_adaln.sh`.

**Split choice.** Attribution is a mechanism question, not a generalisation
question, and reusing a confirmation split's adaLN arm would erode its
confirmatory status. This is a DEVELOPMENT-SPLIT attribution analysis and does
not alter or re-open the section 6 replication verdict.

### What this isolates, and what it does not

`z_sem` is produced from control-cell expression PLUS the drug fingerprint
(`MLPModel.py:201` assigns `concat([control_feature, drug_dose])`). Removing it
from the modulation input removes BOTH drug identity and baseline cell state.
This arm isolates **semantic-latent-dependent modulation**, not drug-dependent
modulation. Isolating drug alone would need a fingerprint-only modulation input,
which is not pre-registered here.

### Endpoint and criterion

**Primary, and the only gated quantity:** chance-corrected DE-overlap Jaccard,
adaLN minus time-only, on drug_split_1. Seed summary is the arithmetic mean of
condition-level raw Jaccard. `c_s` recomputed for drug_split_1 by the same exact
hypergeometric method. Welch 95% CI, arms unpaired, five seeds each. Exhaustive
252-allocation permutation as support.

**Secondary, not gated.** B-C, DPC-50, and the three-arm decomposition so the
ordering is visible:

    delta(time-only - base),  delta(adaLN - time-only),  delta(adaLN - base)

### Three mutually exclusive outcomes, fixed now

1. **Semantic-latent modulation contributes beyond matched time-only
   modulation** - the adaLN-minus-time-only CI lies entirely above zero.
   Because the arms share parameter count and modulation topology, this supports
   an INCREMENTAL benefit from including z_sem in the modulation input. It does
   NOT imply the entire adaLN-versus-base gain is unrelated to generic
   modulation or capacity; both components may contribute, and the three-arm
   decomposition is reported to show their relative size.

2. **Incremental semantic-modulation effect UNRESOLVED** - the CI includes zero.
   This experiment does not resolve whether semantic-latent-dependent
   modulation improves the endpoint beyond matched time-only modulation. The
   Step 4 architecture result remains valid; the semantic-modulation
   interpretation is simply unsupported by this experiment.

   **This is explicitly NOT a claim of equivalence.** A difference CI containing
   zero is also consistent with a real effect this five-seed design lacks power
   to resolve. No equivalence margin has been justified in advance, so no
   equivalence claim may be made afterwards. Any future equivalence claim
   requires a prospectively defined margin and a procedure such as two
   one-sided tests.

3. **Time-only exceeds adaLN** with a CI entirely below zero. Reported as such;
   would indicate the semantic latent actively harms the modulation.

No further arm may be introduced after seeing these results to rescue the
mechanistic reading.

### Verification: the state dict CANNOT identify this arm

Semantic adaLN and time-only adaLN have **identical parameter structures and
identical counts**, which is what makes the control capacity-matched. The
post-training state-dict architecture check used in Step 4 and the replication
therefore CANNOT distinguish them. A behavioural test is required instead:

- **Functional discrimination.** With the timestep fixed, feed two different
  z_sem values and compare the modulation output. For **time-only** it must be
  bitwise IDENTICAL. For **semantic adaLN** it must differ. Run on every
  checkpoint after training, and record the result in provenance.
- **Identity at initialisation** for the time-only arm, exactly 0.0 against the
  base model at block and model level.
- **Parameter count EXACTLY 58,186,374**, identical to adaLN. A different count
  means the arm is not capacity-matched and the comparison is void.
- **Provenance** per run in the tracked `provenance/` directory: commit, GPU
  type read from torch, data and script hashes, checkpoint hash, resolved
  command, and the resolved flag value.

### The freeze tag must move, and why that is safe

Adding `use_adaln_timeonly` modifies `Squidiff/MLPModel.py`, so "byte-identical
to `step4-replication-freeze`" is impossible for this arm and that requirement
is WITHDRAWN for it. Instead:

- A new tag `step4b-control-freeze` is created and all five runs verified
  against it.
- **Regression requirement:** the modified code must reproduce an existing
  Step 4 checkpoint's output BITWISE for both the base and semantic-adaLN paths,
  demonstrating the change is a strict superset that leaves prior behaviour
  untouched. If it does not, the existing adaLN arm must be retrained under the
  new code rather than compared across implementations.

### What this cannot establish

- Whether a fingerprint-only modulation input would behave differently.
- Anything about generalisation: one split, and the development one.
- Whether adaLN beats ridge; amendment 12 found no evidence for that.
- The absolute quality of any arm. All three are expected to remain below the
  expression-only heuristic on DE-overlap and below control-mean parity on B-C.

---

## 14. Amendment, 2026-07-30: mechanism metric under the capacity control

### Scope of blindness, stated precisely

No output drug eta^2, output excess eta^2, latent drug eta^2, latent excess
eta^2, or latent/output ratio has been computed or inspected for ANY arm since
amendment 13 completed.

This is NOT a claim that no relevant data exists. Time-only prediction vectors
already exist and have been used for DE-overlap, B-C and DPC-50. The primary
endpoint here could have been computed from them at any point; it has not been.
This amendment is blind to the eta^2 outcomes, not to the underlying
predictions.

### Two separate questions

**Primary.** Does semantic adaLN produce greater OUTPUT drug-associated excess
eta^2 than parameter-matched time-only modulation?

**Secondary.** How do base, time-only and semantic compare on LATENT excess
eta^2 and on the latent/output excess ratio?

These can disagree. Time-only could match semantic on output excess while
differing in latent excess, giving a different ratio. Each is reported
separately and the primary does not determine the secondary.

### Estimator, fixed now

For each checkpoint independently:

- output is the condition-mean predicted delta, B - C; latent is z_sem
- both matrices are centred WITHIN cell line before estimation
- drug-associated eta^2 is variance-weighted:
  sum_j SS_between,j / sum_j SS_total,j, over latent coordinates or output genes
- the stratified null permutes drug labels within each of the 12 dose-by-cell-
  line strata, 500 permutations, RandomState(0), matching the existing mechanism
  protocol
- the same precomputed permutation-label arrays are reused across all 15
  checkpoints and both representations
- excess = observed eta^2 minus the mean permutation-null eta^2

Latent coordinates are never averaged or aligned across checkpoints: the spaces
are independently learned. Only scalar per-checkpoint summaries are compared
five versus five. Shared permutation arrays reduce unrelated Monte Carlo
variation; they do NOT make training seeds paired.

**Non-positive excesses.** Output excess eta^2 may be zero or negative and
remains a valid primary scalar. The latent/output ratio is reported ONLY when
both excesses are finite and positive; otherwise that checkpoint's ratio is
labelled undefined. It is not clipped, imputed, or silently dropped from any
other statistic.

### Inputs and provenance

Primary contrast uses vectors produced under the SAME current sampler:

- semantic:  vectors_adaln_v2_seed_0..4.npz
- time-only: vectors_timeonly_seed_0..4.npz

For every file the analysis verifies checkpoint SHA-256, vector SHA-256,
effective mode, evaluation commit, identical condition order, 443 conditions,
and identical true and C arrays. Time-only additionally requires a sidecar whose
checkpoint SHA-256 matches, genuine JSON Boolean fields, and effective model and
block mode (True, True); semantic requires (True, False).

**Base arm is CROSS-VERSION.** Base vectors were produced under the earlier
sampler, so the three-arm ordering is not uniform. Base is retained as a
SECONDARY cross-version reference and labelled as such wherever it appears. The
primary semantic-versus-time-only contrast is unaffected.

**Latent extraction.** All 15 extractions run sequentially in ONE allocation on
NVIDIA H100 NVL, under one recorded git commit and one PyTorch/CUDA/cuDNN
environment, verified from torch at runtime. The manifest requires exactly 15
files and records per checkpoint: path, checkpoint SHA-256, latent SHA-256,
effective mode, node, GPU name, and the torch/CUDA/cuDNN versions. This also
retires the standing caveat that the original base and adaLN latents were
extracted on A100 and h100 respectively.

### Three mutually exclusive outcomes, fixed now

1. **Semantic modulation increases output drug-associated variance** - the
   adaLN-minus-time-only CI lies entirely above zero. The mechanism metric
   separates the arms where DE-overlap did not, giving independent support to
   the amendment 13 secondary finding.

2. **UNRESOLVED** - the CI includes zero. NOT equivalence; no margin is
   justified here either. Combined with amendment 13 it would mean neither the
   pre-registered endpoint nor the mechanism metric attributes the Step 4 gain
   to semantic conditioning, and the 7.92x to 5.37x result should be reported
   as an architecture effect rather than evidence of semantic conditioning.

3. **Time-only exceeds semantic** with a CI entirely below zero. Reported as
   such.

### What this cannot establish

- It does not reopen the replication verdict; drug_split_1 only.
- z_sem still reaches the decoder additively in BOTH modulation arms, so this
  does not test whether the semantic latent matters to prediction at all.
- eta^2 is computed on condition-mean profiles, not per cell.
- The arms were trained under different commits, 592924e and 3ee4531; that
  limitation is unchanged here.
