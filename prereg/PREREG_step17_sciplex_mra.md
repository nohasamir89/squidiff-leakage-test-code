# Pre-registration — Step 17: response-magnitude agreement (MRA) on sci-Plex, a further read of the frozen predictions

Date: 2026-10-09. Written before any MRA value on sci-Plex existed. The metric was defined on Tahoe (Step 16,
`PREREG_step16_tahoe_targets.md` §4.4, 2026-10-06) and has never been computed on the sci-Plex arrays. No training;
every number is a further read of predictions frozen by Steps 5, 11, 12, 12b, 13 and 13b and already scored by Step 14.

## 1. Question

On Tahoe the flow-matching recipe's predicted response size ranks held-out conditions within a line and dose
(Step 16 §4.4: MRA +0.32 / +0.29 / +0.31, above zero on every split; base −0.08 / −0.06 / −0.11; the fingerprint ridges
resolved on one split only). Does the same hold on sci-Plex — the thesis's first dataset, three lines, four doses —
for the equivalent arm, against the same baselines?

- **Q1 (primary, gated):** MRA(fm_50) − 0, per split. fm_50 is the sci-Plex recipe (Step 6b / Step 11 checkpoints:
  adaLN, Gaussian flow matching, fingerprint, 50 NFE Euler, unguided; five seeds).
- **Q2 (co-primary, gated):** MRA(fm_50) − MRA(ridge-train), per split — size ranking beyond the fingerprint-linear
  shift fitted on the training compounds.

## 2. Data and submissions

Per split (`drug_split_1` development; `drug_split_0`, `drug_split_2` confirmation): the frozen mean vectors
`leakage_results/vectors_<tag>.npz` (`cond`, `true`, `B`, `C`), verified against their provenance records exactly as
`score_step14.py` does (same registry, same checks). Submissions, the Step 14 set:

| submission | source | seeds |
|---|---|---|
| fm_50 (the recipe) | Step 6 sweep at 50 NFE (split 1) / Step 11 (splits 0, 2) | 5 |
| base Squidiff | Step 5 (split 1) / replication (splits 0, 2); mean vectors only | 5 |
| guidance w = 2 | Step 12 / 12b | 5 |
| attn_cross, mlp_ctrl | Step 13 / 13b | 5 each |
| ridge-S7, ridge-train, mean-response, control-paste | rebuilt by Step 14's code, verbatim | 1 |

Conditions 443 / 444 / 455, compounds 37 / 37 / 38 (Step 14), lines A549 / K562 / MCF7, the test file's four dose
levels; contexts = (line, dose), 12 per split. Metadata from `sci_plex_test_<split>.h5ad` (`Group`, `SMILES`, `dose`,
`cell_name`), as Step 14.

## 3. Metric

MRA as Step 16 §4.4, computed by `score_step16_tahoe.py`'s own functions (`prep_ctx`, `mra_pre`, `ctx_split`): for each
context with ≥ 10 conditions, the Spearman correlation between ‖B − C‖ and ‖true − C‖ over the context's conditions
(L2 norms over the 200 genes); MRA = the mean over contexts. Arms: per seed, then the seed mean. 0 = no information.
Control-paste and mean-response are 0 by construction (asserted: ‖B − C‖ is constant within every context).

Descriptive anchor, not gated and not a ceiling claim: the Step 14 split-half construction (five seeded half-splits
of each condition's stored real cells and of its paired vehicle cells; `RandomState(r)`, r = 0–4) gives, per
half-split, the Spearman within each context between ‖profA − cA‖ and ‖profB − cB‖; reported as the mean over the
five. It is the split-half repeatability of the measured size (cell-sampling noise only, same wells), so it bounds
from above what any predictor could reach against one measurement; it is scored on exactly the same conditions as
the submissions.

## 4. Intervals and contrasts

Drug(compound)-clustered percentile bootstrap, 5,000 replicates, `RandomState(0)`, 97.5 % per split (1.25 / 98.75
percentiles), the MRA recomputed on every resampled set as in Step 16. Contrasts: Q1 and Q2; secondaries: base − 0,
ridge-S7 − 0, ridge-train − 0, fm_50 − ridge-S7, guidance − fm_50, attn_cross − fm_50, mlp_ctrl − fm_50, each of the
four arms − 0. Descriptive (point values, intervals reported, nothing gated): MRA per dose level (3 contexts each) and
per line (4 contexts each) for every submission; the split-half anchor; worst seed per arm.

## 5. Verdict rule (coded in the scorer, precedence 1 → 2 → 5 → 3 → 4)

1. **SIZE SIGNAL BEYOND LINEAR** — Q1 above zero on all three splits and Q2 above zero on all three.
2. **SIZE SIGNAL** — Q1 above zero on all three splits (Q2 not all above).
5. **UNRESOLVED** — Q1 includes zero on all three splits.
3. **REVERSED** — Q1 below zero on any split.
4. **PARTIAL** — otherwise (Q1 above zero on one or two splits).

"Includes zero" is "unresolved", never "equivalent".

## 6. Splits and discipline

`drug_split_1` is scored first and alone (`_development` records); the final run scores all three splits; nothing in
the scorer changes between the two, and the scorer refuses to run uncommitted. These splits have been read before
(Steps 4, 6b, 11, 12b, 13b, 14); this step adds no training and no selection, and the metric was fixed on another
dataset before any sci-Plex value was seen. Reproduction guards before any MRA is reported: ridge-S7's DPC-50 median
equals the Step 7 record; every submission's DE-overlap and DPC-50 mean, rebuilt from the same arrays by the same
code, equal the Step 14 record (`leakage_results/step14_analysis.json`, `thesis_metrics`) to 1e-9; condition order,
`true` and `C` identical across submissions.

## 7. Expectation (written before the run)

Q1 above zero on all three splits, at +0.2 to +0.4; base at or below zero; ridge-train unresolved or small positive;
Q2 unresolved on at least one split → **category 2, SIZE SIGNAL** (category 1 if Q2 resolves everywhere). Guidance
below fm_50 (as on Tahoe); attn_cross and mlp_ctrl within ± 0.05 of fm_50. Per dose: no prediction — on Tahoe the
middle dose was strongest and the lowest weakest (Step 16 §10), and sci-Plex has four doses on a different scale.

## 8. Limits

Three lines and twelve contexts per split (Tahoe: 14 lines, 42), so wider intervals; no biological replicate exists
in the processed sci-Plex files, so the anchor is split-half only and is an upper bound, not a ceiling in the Tahoe
sense; per-dose and per-line values rest on three or four contexts and are descriptive; the arrays are the frozen
Step 5–13b predictions at their own evaluation protocols (50 NFE, unguided, per-cell storage), not re-sampled; the
splits are not independent of earlier steps (§6).

## 9. Implementation and records

`score_step17_sciplex_mra.py` (imports `score_step14.py`'s registry, loader and ridge construction by copying the
exact code; imports `score_step16_tahoe.py`'s MRA functions), committed before any run, refuses to run if it, this
pre-registration or the Step 14 record are uncommitted or modified. Runs on a CPU node. Records:
`leakage_results/step17_mra[_development].json`, `step17_mra_long[_development].csv`,
`step17_mra_per_condition[_development].csv` (every submission's effect norm and the measured norm per condition);
logs under `leakage_results/step17_logs/`. Never overwrites. Findings in `step17_findings.md`.
