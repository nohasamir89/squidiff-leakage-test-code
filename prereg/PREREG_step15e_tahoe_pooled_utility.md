# PREREG — Step 15e: pooled utility of the Tahoe predictions against PRISM sensitivity (all held-out drugs)

Written 2026-10-06, after Step 15d (`step15d_findings.md`; records 67d8f9a, b29c2c2) and before any number of this step
exists. No training; the same frozen Step 15b arrays. This document is committed before `score_step15e_pooled.py`
runs; nothing in it is edited after results exist (pre-run corrections, if any, are dated paragraphs inside a section).

## 1. Question

Step 15d's layer A ended UNRESOLVED: with 13–33 labelled held-out drugs per split, the per-split test could not decide
whether the recipe's predicted effects carry sensitivity information, and the three splits disagreed. The three
held-out sets are disjoint and together cover every usable drug once, so the pooled design has 85 labelled held-out
drugs (27 / 33 / 25 at the highest available dose; `step15d_utility.json`) instead of 13–33.

**Q1.** Pooled over the three held-out sets, is U(recipe) above zero — do the predicted effects rank the held-out
drugs within a line the way PRISM measured their sensitivity? **Q2.** Is U(recipe) above U(ridge-train)? Reported
next to them, not gated: the ceiling (measured effects), QSAR, effect size, and the calibrated ratio
U(recipe) / U(ceiling).

## 2. What is known (from `step15d_utility.json`, commit 67d8f9a), and what this step adds

Highest-available-dose variant of Step 15d, per split 1 / 0 / 2: labelled held-out drugs 27 / 33 / 25 (pairs 240 /
289 / 214); U(recipe) −0.134 / +0.187 / +0.410; U(ridge-train) +0.240 / +0.146 / +0.144; U(ceiling) +0.332 / +0.265 /
+0.134; U(QSAR) +0.517 / +0.159 / +0.187; U(measured effect size) +0.509 / +0.345 / +0.290; f CV R² +0.064 / +0.076 /
+0.088; QSAR CV R² +0.031 / +0.095 / +0.125. The 5 µM rule of Step 15d dropped 101 / 153 / 118 of the labelled pairs
and the most potent drugs; Step 15e therefore uses the highest available dose per drug as its primary dose rule. This
choice is made knowing the per-split 15d values and is stated as such; the pooled analysis itself has not been run.

**Stated in advance:** the point estimate of the primary pooling (§4, reading A) is the drug-weighted mean of the three
known split values and is therefore already determined: recipe (27 × (−0.134) + 33 × 0.187 + 25 × 0.410) / 85 =
+0.151; ridge-train +0.175; ceiling +0.248; QSAR +0.281; measured effect size +0.381. What Step 15e adds is (i) the
interval of each pooled quantity and of the contrasts, from a stratified drug bootstrap, and (ii) the fully pooled
reading B of §4, whose estimates are new.

## 3. Submissions and inputs (no new predictions)

As Step 15d §3: the five arms (three seeds, per-seed values averaged), the four shift baselines rebuilt by the Step 15b
code path and asserted to reproduce the Step 15b per-condition record to 1e-9, the measured effects (ceiling), QSAR and
effect size; PRISM labels, the inventory-(d) drug matching and the dose-availability rule exactly as in
`score_step15d_utility.py` (section-5 variant (a)); all files hash-checked as there.

## 4. Method

**Per split s**, exactly the highest-dose variant of Step 15d: every drug at the highest dose level at which it has
conditions in the PRISM lines; f_s = ridge (alpha from {0.1, 1, 10, 100, 1,000}, 5-fold GroupKFold over training drugs)
from the standardised measured training effects of split s to AUC; QSAR_s likewise from Step 15b's compound features;
per-pair values v_s(ℓ, d) for every labelled held-out pair of split s under every submission; per-line Spearman ρ_{s,ℓ}
over the labelled held-out drugs of s in ℓ (≥ 8 pairs), U_s = mean over lines. f_s never sees a drug held out in s; a
drug is held out in exactly one split, so every held-out drug is scored by an instrument that never saw it.

**Reading A (primary pooling).** U_pool = Σ_s n_s U_s / Σ_s n_s with n_s the number of labelled held-out drugs of split
s. Contrasts are differences of U_pool between submissions. Intervals: stratified drug-clustered percentile bootstrap —
in every replicate the held-out drugs of each split are resampled with replacement within that split, U_s and U_pool
recomputed for every submission on the same resample, 5,000 replicates, `RandomState(0)`, 97.5 %.

**Reading B (fully pooled, secondary).** For every (split, line) the per-pair values of a submission are centred by
their own mean (removes the instrument offset between splits), then the three splits' pairs are pooled and one Spearman
per line is computed over all labelled held-out drugs of that line (up to 85), U_B = mean over the 10 lines; same
bootstrap. Reported next to A; not gated.

**Ratio (descriptive).** U_pool(recipe) / U_pool(ceiling) with its bootstrap interval, reported only when the ceiling's
interval is above zero.

**Within-drug across-line (secondary).** For every held-out drug with ≥ 6 labelled lines, Spearman across lines between
v(ℓ, d) and AUC; mean over the 85 drugs; stratified drug bootstrap; recipe, ridge-train, ceiling, QSAR. Per-line U_pool
for the three brain lines versus the seven gut lines (descriptive).

Floors: control-paste and mean-response are constant within a line, ρ = 0 by definition (asserted). The 5 µM reading of
Step 15d is **not** recomputed here; Step 15e reports the highest-dose rule only, as stated in §2.

## 5. Endpoints

- **Primary, gated (Q1):** U_pool(recipe) − 0, reading A.
- **Co-primary, gated (Q2):** U_pool(recipe) − U_pool(ridge-train), reading A.
- **Secondary, not gated:** every submission's U_pool − 0; ceiling, QSAR, effect size (recipe), measured effect size −
  0; recipe − ceiling; recipe − QSAR; recipe − effect size; guidance w=2 − recipe; attn_cross − mlp_ctrl; the ratio;
  reading B for every quantity above; per-split U_s (identical to the 15d record, asserted to 1e-9); per seed; per line;
  brain versus gut; within-drug across-line; the CV R² of every f_s and QSAR_s.

## 6. Outcome categories, fixed now (precedence 4 → 1 → 2 → 3, coded in the scorer)

4. **LABEL SET CANNOT RESOLVE** — the ceiling's U_pool − 0 is not above zero: the instrument finds no sensitivity
   information in the *measured* effects, so no submission can be judged; reported with the instrument's CV R².
1. **SENSITIVITY INFORMATION BEYOND LINEAR** — primary above zero and co-primary above zero.
2. **SENSITIVITY INFORMATION, NOT BEYOND LINEAR** — primary above zero; co-primary not above zero.
3. **NO RESOLVED SENSITIVITY INFORMATION** — primary includes or is below zero while the ceiling is above zero: the
   measured effects carry the information and the predictions do not carry a resolvable amount of it; the ratio is
   reported with its interval.

Reading B carries no category.

## 7. Expectation, recorded in advance

The reading-A point estimates are known (§2). The bootstrap intervals, from the three per-split half-widths of Step 15d
(≈ 0.33, 0.26, 0.21) combined with weights 27 / 33 / 25, should be about ± 0.16 around each pooled U. I therefore expect:
ceiling (+0.248) above zero, measured effect size (+0.381) above zero, QSAR (+0.281) above zero; the recipe (+0.151)
at the boundary — above zero or including it, with no basis to call which; ridge-train (+0.175) likewise; the
co-primary (−0.024) including zero. So **category 2 or 3**, decided by whether +0.151 clears its lower bound, and a
ratio near 0.6 with an interval wide enough to include both "half" and "none". Reading B, with up to 85 drugs per line,
should give narrower intervals; I expect its estimates within ± 0.1 of reading A's. I record no expectation for brain
versus gut or for the within-drug read.

## 8. What this step cannot establish

- A better model: frozen predictions, as Step 15d.
- Sensitivity for the four lines absent from PRISM, or for drugs outside the 135 matched.
- Independence from Step 15d: same arrays, same labels, same instrument; the pooling is the only new element, and
  reading A's point estimates are Step 15d's numbers re-weighted. The step is honest about that; its value is the
  interval and reading B.
- That the ratio is a property of the model rather than of this label set and instrument.

## 9. Analysis plan and provenance

`score_step15e_pooled.py`, committed before it runs: guards as `score_step15d_utility.py` (this document committed and
unchanged in section count ≥ 9; scorer and imported scorers committed and clean; arrays hash-checked; baselines
reproduced to 1e-9; PRISM files checked); the per-split highest-dose values asserted equal to `step15d_utility.json`
(`exploratory.variants.highest_dose.U`) to 1e-9 before pooling; then readings A and B, the contrasts, the ratio, the
secondary reads, the category. One run over all three splits (no development split: the per-split values are already
known and asserted). Outputs `tahoe_results/step15e_pooled.json`, `step15e_pooled_long.csv`, `step15e_pooled_pairs.csv`;
log `tahoe_results/step15e/pooled.log`. Never overwrites.

- Step 15d records: commit 67d8f9a; findings b29c2c2. Step 15b records: 6b40c72.
- Pre-registration commit: [hash]. Scorer commit: [hash].
