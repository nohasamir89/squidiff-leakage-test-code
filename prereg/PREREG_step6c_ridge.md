# Pre-registration: Step 6c, flow matching against fingerprint ridge

**Written and committed BEFORE any DPC-50 value is computed for a
flow-matching checkpoint.** Confirmed by inspection: `step6_analysis.json` and
`step6b_analysis.json` contain P1 (DE-overlap Jaccard) and P2 (energy distance)
only. No flow-matching DPC-50 exists in any stored artifact.

This is a **blind** pre-registration, unlike amendment 12 of the Step 4
pre-registration, which had to disclose that adaLN point estimates were already
visible when its ridge comparison was specified.

Author: Noha Samir Ismail. Date: 2026-08-19.

## 1. Question

Flow matching improves DE-overlap over DDIM, confirmed on two held-out splits
(+0.03206 and +0.02802, both intervals above zero at the 97.5% level). DE-overlap
measures which genes are identified as responding.

Ridge regression on drug fingerprints beats base Squidiff on DPC-50 by three to
ten times across all six splits, and amendment 12 found no evidence that adaLN
exceeds ridge on the two confirmation splits.

**Does the confirmed flow-matching improvement close that gap?**

DPC-50 measures the correlation between predicted and true expression CHANGE on
the true top-50 DEGs - accuracy of the predicted response, not selection of
which genes respond. A model can improve one without the other; Step 6 already
showed `fm_control_ot` leading on DE-overlap while failing distributionally.

## 2. Arms and scope

**Flow matching:** `fm_gaussian` at 20 NFE with Euler integration, five seeds,
on all three splits where checkpoints exist - `drug_split_1` from Step 6,
`drug_split_0` and `drug_split_2` from Step 6b.

The 50-NFE configuration is NOT used. It was identified by inspecting a
post-hoc sweep and is excluded here for the same reason it was excluded from
Step 6b.

**Ridge:** the stored per-condition `ridge_fp__dpc50` from
`per_condition_<split>.npz`, unchanged from Step 4. Ridge is not refitted.

All three splits are reported. No split may be selected after the fact.

## 3. Endpoint and estimator

**Primary:** the **median of paired per-condition differences**,
flow matching minus ridge, on DPC-50, computed per split.

This is not the difference of marginal medians. Amendment 12 established that
the two can disagree in sign: adaLN's marginal median DPC-50 exceeded ridge's
while the paired median difference was negative on both confirmation splits.
The paired form is the estimand.

**Uncertainty:** 95% percentile interval from a **drug-clustered bootstrap**,
resampling the 37 or 38 compounds of the split with replacement, 5000
replicates, `RandomState(0)`. Conditions within a compound are not independent;
the compound is the resampling unit, matching amendment 12.

**Aggregation across seeds:** the five flow-matching checkpoints are pooled per
condition by their arithmetic mean before differencing against ridge, so each
split yields one paired-difference distribution over its conditions. This
matches amendment 12's treatment of the adaLN arm.

**Verification:** every vector file is checked against its evaluation
provenance record - status, split, arm, seed, checkpoint hash and vector hash -
before use, and the analysis ABORTS on any mismatch. Filename correctness is
not sufficient: the first Step 4 replication attempt produced correctly named
base artifacts from adaLN checkpoints.

## 4. Outcome categories, fixed now

Assessed per split, then combined. The first applicable category applies.

1. **FLOW MATCHING EXCEEDS RIDGE** - the paired median difference is positive
   with its 95% interval entirely above zero on all three splits.
2. **RIDGE EXCEEDS FLOW MATCHING** - interval entirely below zero on all three.
3. **MIXED** - intervals resolve in different directions across splits, or
   resolve on some and not others. Reported per split with no pooling.
4. **UNRESOLVED** - every interval includes zero. NOT equivalence; no margin is
   justified and none may be claimed.

## 5. Expectation, recorded in advance

**A negative result is expected.** Ridge DPC-50 runs 0.064-0.117 across splits
against base Squidiff's 0.007-0.035. The flow-matching improvement was +0.03 on
DE-overlap, a gene-ranking metric, and its pooled predicted-versus-true delta
correlations on `drug_split_0` were 0.09-0.11 - well below ridge's DPC-50 range.

This is recorded so that a negative result reads as anticipated rather than as
a disappointment discovered late, and so that a positive result cannot be
presented as though it had been expected.

## 6. What this cannot establish

- Whether ridge would remain competitive on a different gene panel, dataset, or
  perturbation type.
- Whether flow matching at another NFE, path or solver would differ.
- Anything about DE-overlap, which is separately confirmed and not at issue here.
- Whether either method is useful in absolute terms. Both remain below the
  control-mean baseline on B-C.
