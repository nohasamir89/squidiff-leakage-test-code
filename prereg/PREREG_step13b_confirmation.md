# Pre-registration: Step 13b, confirming drug × cell-state cross-attention on held-out compounds

**NOT BLIND to the development result.** The Step 13 values being confirmed are
reproduced in section 2. What has not been computed is any attention arm, any
control arm, or any vector of either on `drug_split_0` or `drug_split_2`.

Author: Noha Samir Ismail. Date: 2026-09-22.

## 1. Question

Step 13 found, on the development split, that the `attn_cross` encoder — the drug
fingerprint as a query over 16 tokens made from the control-cell expression, a
zero-initialised residual on the drug input — raised DE-overlap over the
reference and over a parameter-matched non-attention control, and lowered energy
distance. Two of its three findings were against the recorded expectation. The
recipe, the widths, the endpoints and the analysis were fixed before that
result, but they were fixed *for* that split, and the result lives on its 37
held-out compounds.

**Does the `attn_cross` gain appear, and still exceed the control, on compounds
that played no part in designing it?**

Same recipe, same arm, same control, same analysis, two other splits. Nothing is
chosen. `attn_fp` and `attn_mem` were UNRESOLVED and are not carried forward.

## 2. What was observed, and its status

From `analyse_step13.py` at commit `731eb18`, `drug_split_1`, 443 conditions, 95%:

| contrast | value | 95% CI |
|---|---|---|
| DE-overlap, `attn_cross` minus reference (primary) | +0.00570 | [+0.00276, +0.00875] |
| DE-overlap, `attn_cross` minus `mlp_ctrl` (attribution) | +0.00316 | [+0.00133, +0.00519] |
| energy distance, `attn_cross` minus reference (co-primary) | −0.02084 | [−0.04314, −0.00637] |
| DE-overlap, `mlp_ctrl` minus reference (secondary) | +0.00254 | [+0.00050, +0.00458] |
| paired DPC-50, `attn_cross` minus interaction ridge | −0.0466 | [−0.1148, −0.0089] |

Recorded mechanism: the arm's predicted effect magnitude is *smaller* than the
reference's (0.68 vs 0.71 of the true size); the gain is a better direction
(DPC-50 +0.019), not amplification, and it improves both endpoints at once. The
control alone resolved above the reference, so the attribution contrast is part
of what has to replicate.

## 3. The intervention

**Twenty new training runs.** On each of `drug_split_0` and `drug_split_2`: five
`attn_cross` seeds and five `mlp_ctrl` seeds — Gaussian-source flow matching,
semantic adaLN, `--drug_encoding fingerprint` (the full construction, as the
Step 11 reference on these splits), 100,000 steps, uniform time sampler, EMA
0.9999, `cond_drop` 0, at the commit of this document. Whole-model parameter
counts asserted at construction: `attn_cross` 58,532,804; `mlp_ctrl` 58,532,300
(reference 58,178,180). Identity at initialisation asserted in every run.

| | |
|---|---|
| arms | `attn_cross` and `mlp_ctrl` (the attribution control travels with the arm) |
| splits | `drug_split_0` (444 conditions, 37 drugs), `drug_split_2` (455 conditions, 38 drugs) — **fifth use of each**: Steps 4, 6b, 11, 12b, 13b |
| seeds | 5 per arm per split |
| reference | Step 11 vectors, `vectors_s11_<split>_fm_gaussian_nfe50_seed_<k>.npz` (Step 6b checkpoints at 50 NFE), no new reference runs |
| sampling | 50 NFE Euler on cn274, per-cell storage on, no guidance |
| evaluations | 10 per split, 20 |

**Implementation assertions, unchanged from Step 13 and part of the run:**
parameter counts as above; the arm's residual exactly zero at initialisation;
the sampler's detected arm bound into every evaluation record with
`guidance_w = None`; the Step 10 magnitude gate for each arm against the
reference.

## 4. Endpoints

All contrasts: mean over conditions of seed-mean per-condition values,
drug-clustered percentile bootstrap over the split's compounds, 5000 replicates,
`RandomState(0)`, at the **97.5% level per split** — the level Steps 6b, 11 and
12b used for two-split confirmation.

**Primary, gated:** DE-overlap Jaccard, `attn_cross` minus reference, per split.

**Attribution, gated:** DE-overlap Jaccard, `attn_cross` minus `mlp_ctrl`, per split.

**Co-primary, gated:** energy distance, `attn_cross` minus reference, per split.
Higher is worse.

**Secondary, reported, not gated:** `mlp_ctrl` minus reference on both endpoints;
DPC-50 as the paired condition median of each arm against the Step 7
interaction ridge, reproduced per split with Step 7's exact procedure and
checked against the Step 7 record (marginal medians 0.2473 on `drug_split_0`,
0.0816 on `drug_split_2`); DPC-50 mean, B − C and predicted-delta magnitude for
each arm.

**Sensitivity.** The development-split primary had an interval half-width of
0.0030 at 95%; at 97.5% with the same seeds the half-width will be about 0.0035
per split, so an effect below about 0.004 will not resolve. Stated so that
"unresolved" is read against it.

**Escalation, carried from Step 13.** If `attn_cross`'s paired DPC-50 difference
against the interaction ridge resolves above zero on both splits, the thesis
headline may change; otherwise a DE-overlap confirmation is reported as a
narrowing of the gap.

## 5. Outcome categories, fixed now

1. **CONFIRMS** — DE-overlap above zero against the reference on BOTH splits,
   above zero against `mlp_ctrl` on BOTH splits, and energy not above zero on
   both. Cross-attention on the drug condition becomes reportable as a confirmed
   effect attributable to attention, with its size stated.
2. **CONFIRMS AS CAPACITY** — DE-overlap above zero against the reference on
   both splits, but not above zero against `mlp_ctrl` on both. Reported as a
   confirmed effect of the residual pathway, not of attention.
3. **TRADE-OFF** — DE-overlap above zero against the reference on both splits
   and energy above zero on either. Reported as a trade, not a gain.
4. **PARTIAL** — DE-overlap against the reference above zero on one split only.
   Reported per split, no pooling, not described as confirmed.
5. **NOT CONFIRMED** — DE-overlap against the reference unresolved on both.
   The Step 13 result stands as a development-split observation and nothing more.
6. **REVERSED** — DE-overlap against the reference below zero on either split.

"Not above zero" for energy means an interval that includes zero or lies below
it; an interval that includes zero is unresolved, not evidence of no cost.

## 6. Expectation, recorded in advance

**I expect CONFIRMS**, with the same reservation Step 13's expectation earned:
the development result was larger than I predicted, and confirmations in this
thesis have so far been of the same order as, or larger than, their development
values (Step 12b). I expect the primary between +0.003 and +0.008 on each split,
the attribution contrast smaller and the one most at risk of not resolving at
97.5%, and energy distance not above zero on both splits, plausibly below. I
expect `mlp_ctrl` above the reference on at least one split. On paired DPC-50 I
expect `attn_cross` to remain below the interaction ridge on `drug_split_0`,
where the ridge's advantage over flow matching was resolved in Step 7, and to be
unresolved on `drug_split_2`, where it was not.

If the primary confirms but the attribution does not, the honest reading is that
the residual pathway, not attention, is what generalises, and that is what the
thesis will say.

## 7. What this cannot establish

- **The mechanism.** Consistent with a drug × cell-state interaction; not
  isolated from a reversed-direction or bilinear variant.
- **That the widths are right.** Fixed a priori; no tuning on any split.
- **Anything about guidance on this encoder.** A separate pre-registration.
- **Independence from Steps 4, 6b, 11 and 12b.** Fifth use of both splits. No
  unspent compound split exists in this design.
- **That energy distance is unchanged** where it is unresolved.
- **A claim about ridge** unless the escalation clause is met.

## 8. Analysis plan, fixed now

`analyse_step13b.py`, committed before any attention or control checkpoint exists
on these splits, reads the 20 arm vector files and the 10 Step 11 reference files,
checks every provenance record's arm, parameter count and guidance setting
against its filename and every vectors hash against its record, reproduces the
Step 7 interaction ridge per split with Step 7's procedure and stops if the
reproduced marginal median differs from the Step 7 record, computes the contrasts
of section 4 per split, applies section 5 mechanically, prints every number it
uses, and writes `leakage_results/step13b_analysis.json` with commit and input
hashes. It refuses to run if any input is missing.

## 9. Provenance

- Development result: `leakage_results/step13_analysis.json`, commit `731eb18`;
  findings `step13_findings.md`.
- Step 13 pre-registration: `PREREG_step13_attention.md`, commit `556dc62`.
- Code: `Squidiff/drug_attention.py` and the construction path (`11e05cb`);
  training and evaluation pattern (`9f64cac`, `d42759b`).
- Reference vectors: Step 11, `provenance/s11ev_s11_<split>_fm_gaussian_nfe50_seed_<k>.json`.
- Step 7 comparator: `leakage_results/step7_interaction_baseline.json`.
- Training commit for the new arms: [hash at submission]
- Pre-registration commit: [hash]
