# Pre-registration: Step 12b, confirming classifier-free guidance on held-out compounds

**NOT BLIND to the development result.** The Step 12 value being confirmed is
reproduced in section 2. What has not been computed is any dropout-trained
model or any guided sample on `drug_split_0` or `drug_split_2`.

Author: Noha Samir Ismail. Date: 2026-09-21.

## 1. Question

Step 12 found, on the development split, that classifier-free guidance at
w = 2 raises DE-overlap by +0.00854 [+0.00561, +0.01142] with no resolved
change in energy distance. The dropout rate, the guidance weight, the endpoints
and the analysis were all fixed before that result, but they were fixed *for*
that split, and the result lives on its 37 held-out compounds.

**Does the w = 2 gain appear on compounds that played no part in designing it?**

Same recipe, same weight, same analysis, two other splits. Nothing is chosen.

## 2. What was observed, and its status

From `analyse_step12.py` at commit `cea765a`, `drug_split_1`, 443 conditions:

| contrast, w = 2 minus w = 1, same checkpoints | value | 95% CI |
|---|---|---|
| DE-overlap (primary) | +0.00854 | [+0.00561, +0.01142] |
| energy distance (co-primary) | +0.00469 | [−0.04281, +0.05133] |
| paired MDE at 80% power | 0.00271 | — |

The mechanism recorded in the Step 12 findings: the unguided model predicts a
drug effect whose mean magnitude is 70% of the true one, and w = 2 brings it to
98%. Nothing in that mechanism refers to the identity of the held-out compounds,
which is the reason to expect replication — and the reason a failure to
replicate would be informative.

**Status of the sweep.** w = 3 gave a larger DE-overlap gain (+0.01308) with an
unresolved 26% rise in energy distance; the half-trajectory variant at w = 2
kept 80% of the gain with a lower energy point estimate. Both are
development-split observations. They are carried here as EXPLORATORY (section
4) and cannot become claims through this document.

## 3. The intervention

**Ten new training runs.** Five dropout-trained seeds on each of `drug_split_0`
and `drug_split_2`: Gaussian-source flow matching, semantic adaLN, Morgan-1024
dose-scaled fingerprint, 100,000 steps, uniform time sampler, EMA 0.9999,
`--cond_drop 0.1`, at the commit of this document.

**Data construction: `--drug_encoding fingerprint`, not `fingerprint_s10`.**
The `fingerprint_s10` string exists to drop three compounds that lack a
ChemicalVAE embedding, so that Step 10's arms saw identical data. No ChemicalVAE
arm is involved here. On these two splits the existing flow-matching reference
— Step 6b's checkpoints, evaluated at 50 NFE in Step 11 — was trained on the
full construction, and the dropout-effect secondary is only like-for-like if the
new arm is too. The gated contrast is within-checkpoint and does not depend on
this choice. The encoder and parameter count (58,178,180) are identical under
both strings; the guard enforces the count.

| | |
|---|---|
| arms | one: dropout-trained flow matching, `cfg_fingerprint` |
| splits | `drug_split_0` (444 conditions, 37 drugs), `drug_split_2` (455 conditions, 38 drugs) — **fourth use of each**: Steps 4, 6b, 11, 12b |
| seeds | 5 per split |
| reference for the dropout secondary | Step 11 vectors `vectors_s11_<split>_fm_gaussian_nfe50_seed_<k>.npz`, Step 6b checkpoints at 50 NFE; no new reference runs |
| sampling | 50 NFE Euler on cn274, per-cell storage on |
| gated evaluations per split | unguided path (5), guided w = 1 (5, identity check), w = 2 (5) |
| exploratory evaluations per split | w = 3 (5), w = 2 for t ≥ 0.5 only (5) |
| total | 25 per split, 50 |

**Implementation assertions, unchanged from Step 12 and part of the run:**
parsed `cond_drop` equals the float 0.1; parameter count 58,178,180; realised
dropout rate within 0.005 of 0.1 over 100,000 calls; the guided path at w = 1
reproduces the unguided path to 10⁻⁶ on condition means for every seed before
any w > 1 sample is drawn; every provenance record binds the guidance weight the
sampler itself printed; the Step 10 magnitude gate on the unguided arm against
the reference.

**One clarification carried from Step 12.** The evaluation count there listed
35; 45 were run, because the identity check needs the five unguided-path runs.
The counts above include them.

## 4. Endpoints

**Primary, gated: DE-overlap Jaccard**, w = 2 minus w = 1 on the same five
checkpoints, per split, mean over conditions of seed-mean per-condition
differences, drug-clustered percentile bootstrap over the split's compounds,
5000 replicates, `RandomState(0)`, at the **97.5% level per split** — the level
Steps 6b and 11 used for two-split confirmation, so that the three confirmations
are directly comparable.

**Co-primary, gated: energy distance**, amendment-13 U-statistic, mean over
conditions, same contrast, same level.

**Secondary, reported and not gated:**
- dropout itself: unguided dropout arm (w = 1) minus the Step 11 reference
  vectors at 50 NFE, per split, different checkpoints, drug-clustered bootstrap;
- DPC-50, B − C and predicted-delta magnitude (mean |B − C| against mean
  |true − C|) at every evaluated weight;
- **EXPLORATORY:** w = 3 and the half-trajectory w = 2, each against w = 1 and
  against w = 2, both endpoints. Declared here before any run, reported
  separately, and unable to revise the gated claim. If either looks better
  than w = 2 on both splits, that is a new hypothesis for a new pre-registration.

**Sensitivity.** The paired MDE on the development split was 0.00271; the
observed effect was 3.2 times it. The same formula is applied per split here
after the fact, reported next to the result, and not used to interpret it.

## 5. Outcome categories, fixed now

1. **CONFIRMS** — DE-overlap interval above zero on BOTH splits AND energy
   interval not above zero on both. Guidance at w = 2 becomes reportable as a
   confirmed result, with the size of the effect stated.
2. **TRADE-OFF** — DE-overlap above zero on both, energy above zero on either.
   Reported as a trade on held-out compounds; not described as a gain.
3. **PARTIAL** — DE-overlap resolved above zero on one split only. Reported per
   split, no pooling, not described as confirmed.
4. **NOT CONFIRMED** — DE-overlap unresolved on both. The Step 12 value stands
   as a development-split observation and nothing more.
5. **REVERSED** — DE-overlap resolved below zero on either split. Reported as
   observed.

"Not above zero" for energy distance means what it says: an interval that
includes zero is *unresolved*, not evidence of no cost. The findings will use
the phrase "no resolved cost", not "no cost".

## 6. Expectation, recorded in advance

**I expect CONFIRMS.** The effect is 3.2 times the MDE, the five seeds agree
(ratios within a narrow band), and the mechanism — restoring the scale of a
drug direction the model already holds — has no reason to depend on which
compounds are held out. I expect the effect size to be of the same order,
+0.005 to +0.012, and energy distance to be unresolved at w = 2 on both splits.

**I expect the exploratory w = 3 to show a larger DE-overlap gain with a larger
energy point estimate**, as on the development split.

If DE-overlap does NOT confirm on either split while the identity check and
magnitude gate pass, the mechanism as stated is wrong or split-specific, and
that would need explaining rather than reporting.

## 7. What this cannot establish

- **That w = 2 is the best weight.** Only w = 2 is gated; w = 3 and the
  half-trajectory variant are exploratory.
- **That energy distance is unchanged.** Unresolved is not equivalence; no
  margin is set.
- **Independence from Steps 4, 6b and 11.** This is the fourth use of both
  splits. No unspent compound split exists in this design, and no result from
  this thesis has an untouched held-out partition to fall back on.
- **Anything about DDIM**, other NFE, other dropout rates, or the Step 10 data
  construction on these splits.
- **Anything about DPC-50 or ridge as a claim.** The DPC-50 values are means;
  the standing ridge comparator is a paired median; they are not compared here.

## 8. Analysis plan, fixed now

`analyse_step12b.py`, committed before any dropout model exists on these splits,
reads the 50 vector files and the 10 Step 11 reference files, checks every
provenance record's guidance weight against its filename and every vectors hash
against its record, computes the contrasts of section 4 per split, applies
section 5 mechanically, and writes `leakage_results/step12b_analysis.json` with
commit and input hashes. It refuses to run if any input is missing.

## 9. Provenance

- Development result: `leakage_results/step12_analysis.json`, commit `cea765a`.
- Step 12 pre-registration: `PREREG_step12_guidance.md`, commit `998778e`.
- Code: dropout in `Squidiff/flow_matching.py` (`aa402fc`), guidance
  (`aa65b30`), evaluation and analysis pattern (`b9a1d6e`).
- Reference vectors: Step 11, `provenance/s11ev_s11_<split>_fm_gaussian_nfe50_seed_<k>.json`.
- Training commit for the new arms: [hash at submission]
- Pre-registration commit: [hash]
