# Pre-registration: Step 7, the drug x cell-line interaction baseline

**NOT BLIND, and the disclosure is the point of this section.** An exploratory
diagnostic on `drug_split_1` was run before this document and its results are
reproduced in section 2. What has not been computed is the replication on
`drug_split_0` and `drug_split_2`, or any paired contrast against flow matching
under any feature set. Those are what this document fixes in advance.

This mirrors amendment 12 of the Step 4 pre-registration, which likewise fixed
an inferential procedure after point estimates were visible.

Author: Noha Samir Ismail. Date: 2026-08-21.

## 1. Why this exists

Every baseline comparison in this thesis has used ridge on drug fingerprints
alone. Step 6c found flow matching does not reach it: paired median differences
of -0.0219, -0.0506 and -0.0118 across three splits.

An exploratory diagnostic then asked where the remaining signal lives. It found
that an oracle knowing the true response of the same drug in the same cell line
reaches DPC-50 0.3273 even at extrapolated doses, against ridge's 0.1155 - so
roughly two thirds of recoverable signal is unreached by anything tested.

Adding an explicit **drug x cell-line interaction** to ridge captured a large
part of that gap. This raises a problem for the thesis: **the baseline the
model has been compared against is weaker than a well-specified simple model.**

## 2. What was already observed, on drug_split_1 only

| feature set | DPC-50 |
|---|---|
| fingerprint | 0.1155 |
| fingerprint + line | 0.1240 |
| fingerprint x line | 0.1557 |
| **fingerprint + line + fingerprint x line** | **0.1884** |
| fingerprint + line + control expression | 0.1238 |
| fingerprint x line + control expression | 0.1885 |

Alpha sweep for the chosen set: 0.1884 at alpha 0.01, 0.1, and 1.0; 0.1845 at
10; 0.1648 at 100; 0.1231 at 10000. The solution is data-determined at the low
end, not penalty-determined.

Reference values on the same split: base Squidiff 0.0215, global-mean predictor
0.0810, flow matching marginal median 0.0924, oracle drug-only 0.1770, oracle
drug x line at edge doses 0.3273.

**Seven feature sets were tried.** The one carried forward was the best of
them. That is selection, and fixing it here before the other two splits are
touched is what makes the replication a test rather than a continued search.

## 3. What is fixed now

**Baseline definition.** Ridge, `alpha = 1.0`, on
`[fingerprint, line indicators, fingerprint x line]` - 1024 + 3 + 3072 = 4099
columns. No other feature set may be substituted after seeing results. Control
expression is NOT included: it added 0.0001 on drug_split_1 and has now failed
to contribute in three separate tests.

**Cross-validation.** `GroupKFold`, 5 folds, grouped by compound, so no drug
appears in both training and test folds. This matches the unseen-compound task.

**Splits.** All three. Every one reported.

**Estimand.** Median of paired per-condition differences, with a drug-clustered
percentile bootstrap over the split's compounds, 5000 replicates,
`RandomState(0)`. Identical to Step 6c and amendment 12, so the three are
comparable. The difference of marginal medians is reported alongside but is NOT
the estimand: it reversed sign against the paired form twice in this project.

**Contrasts, both reported per split:**

1. **interaction-ridge minus plain ridge** - does the interaction help?
2. **flow matching minus interaction-ridge** - does the model reach the
   stronger baseline?

**Flow matching arm.** `fm_gaussian` at 20 NFE, five seeds, pooled per
condition by arithmetic mean before differencing. Same treatment as Step 6c.

## 4. Outcome categories, fixed now

Per contrast, per split, then combined. First applicable category applies.

**For contrast 1, interaction-ridge minus plain ridge:**

1. **INTERACTION HELPS** - interval entirely above zero on all three splits.
2. **MIXED** - resolves in different directions, or on some splits only.
3. **NO RESOLVED GAIN** - every interval includes zero. The drug_split_1
   observation would then be a single-split result that did not replicate, and
   the thesis continues to report plain ridge as the baseline.

**For contrast 2, flow matching minus interaction-ridge:**

1. **FLOW MATCHING EXCEEDS** - interval above zero on all three splits.
2. **INTERACTION-RIDGE EXCEEDS** - interval below zero on all three.
3. **MIXED** - resolves in different directions or on some splits only.
4. **UNRESOLVED** - every interval includes zero. NOT equivalence.

## 5. Expectation, recorded in advance

**Contrast 1 is expected to resolve in favour of the interaction.** The
drug_split_1 gain was 0.1240 to 0.1884, and the oracle diagnostic gives a
mechanistic reason to expect it: drug-and-line knowledge reaches 0.3273 where
drug-only knowledge reaches 0.1770.

**Contrast 2 is expected to resolve AGAINST flow matching.** Its marginal
median on drug_split_1 is 0.0924 against the interaction baseline's 0.1884, and
Step 6c already found it behind plain ridge on the paired estimand.

Recorded so that the anticipated outcome cannot later be presented as a
surprise, and so that an unanticipated one is visibly unanticipated.

## 6. What this changes for the thesis if contrast 1 resolves

The baseline comparison reported throughout Steps 1, 4 and 6c used ridge on
fingerprints alone. If the interaction baseline replicates, then **the
published-model comparison understates how strong a simple model can be**, and
the thesis must say so.

That does not invalidate the earlier results: base Squidiff lost to plain ridge,
and losing to a stronger baseline is a fortiori. But every statement of the form
"the model does not reach ridge" becomes "the model does not reach ridge, and
the gap to a well-specified linear model is larger still."

## 7. What this cannot establish

- Whether a deep model could reach the oracle ceiling of 0.3273. The remaining
  gap above the interaction baseline is nonlinear by construction, since linear
  crossing has been exhausted, but nothing here tests whether a network reaches
  it.
- Whether the interaction helps on DE-overlap. This is a DPC-50 analysis only.
- Whether the result transfers beyond sci-Plex3, three cell lines, or a
  200-gene panel. Three cell lines is a small basis for an interaction claim.
- Anything causal about why the model underperforms. The diagnostic locates
  headroom; it does not explain the architecture's failure to use it.
