# Pre-registration: Step 15b, Tahoe-100M (brain and gut lines) — does the sci-Plex recipe generalise?

**BLIND for every quantity this step produces: no Tahoe checkpoint, vector or metric exists.** NOT BLIND for
the sci-Plex results of Steps 1–14, which fix the expectations in §7, and NOT BLIND for the Tahoe data record
(`tahoe_results/step15_data_record.json`, commit 9647935), which was built without any model and is cited where used.

Author: Noha Samir Ismail. Date: 2026-09-26. Branch `step15-tahoe`; records in `tahoe_results/`.

## 1. Questions

Steps 1–14 established, on sci-Plex three-line unseen-compound prediction, that (a) the recipe — semantic adaLN
plus Gaussian-source flow matching, sampled at 50 NFE unguided — raises held-out-compound DE-overlap far above the
base Squidiff model, and (b) a linear shift of the paired control cells by a ridge-predicted mean delta matches the
recipe on DE-overlap and beats it on the compound-specific metric (DPC-50), so the generative model's standing claim
rests on distributional fidelity (energy distance), not on gene ranking. Step 15a prepared the Tahoe-100M brain and
gut cancer lines under the same protocol. This step asks whether those two findings hold on the new dataset.

**Q1 (generalisation of the recipe).** On the Tahoe brain/gut lines, does the recipe raise held-out-compound
DE-overlap above the base model, as it did on sci-Plex?

**Q2 (the linear-shift comparison, ported).** On the same data, does the recipe beat a ridge linear shift of the
paired control cells on the thesis's three metrics, or does the Step 14 pattern (shift ahead on DPC-50, close on
DE-overlap, model ahead on energy) recur?

**Q3 (the guidance trade-off, ported).** Does classifier-free guidance at w = 2 raise DE-overlap above the unguided
recipe on the Tahoe lines, and at what cost in energy distance? On sci-Plex (Step 12b) guidance was the DE-overlap
leader but worse on energy on every split — a TRADE-OFF, not a strict win. Q3 tests whether that trade-off recurs;
it is judged on DE-overlap **and** energy jointly, never on DE-overlap alone.

**Q4 (cross-attention re-test, exploratory).** Cross-attention over drug × cell-state (`attn_cross`) was NOT
CONFIRMED on sci-Plex (Step 13b, §1): on the confirmation splits it did not beat its parameter-matched non-attention
control (`mlp_ctrl`) on DE-overlap (point estimates −0.00135, −0.00195), so its development-split edge over the recipe
(+0.0057 [+0.0028, +0.0088]) was extra capacity plus development-split variation, not the attention mechanism. Tahoe
has 14 cell lines against sci-Plex's 3, so there is more cell-state variation for an attention mechanism to use; this
step re-tests, prospectively and on new data, whether attention beats its own parameter-matched control here. It is
exploratory, separate from the recipe's claim, and judged only by the attention-versus-control contrast
(`attn_cross` − `mlp_ctrl`), never by `attn_cross` against the recipe.

Q1 is the generalisation claim the thesis needs; Q2 is the honesty check that the recipe earns its cost over a linear
baseline; Q3 asks whether the guided variant, which led the sci-Plex ladder on DE-overlap, does so here too and
whether its fidelity cost transfers; Q4 re-tests, as an exploratory pair, a mechanism sci-Plex rejected. Q1–Q3 are
answered on all three splits with `drug_split_1` as development and `drug_split_0`, `drug_split_2` as confirmation;
Q4 is reported on all three but carries no confirmation weight (§5).

## 2. What is known, and its status

All sci-Plex numbers below are from `step14_findings.md` (commit 9647935 on this branch carries it) and are NOT BLIND;
they fix expectations only, and nothing here is gated on them.

| fact (sci-Plex, splits 1 / 0 / 2) | source |
|---|---|
| DE-overlap: base 0.5080 / 0.5047 / 0.4908; recipe (unguided 50 NFE) 0.6100 / 0.6124 / 0.6106 | Step 14 §2 |
| DE-overlap: ridge-S7 shift 0.5544 / 0.5684 / 0.5535 | Step 14 §2 |
| energy (lower better): recipe 0.1795 / 0.0483 / 0.1178; ridge-S7 0.2308 / 0.0322 / 0.1612 | Step 14 §2 |
| DPC-50 median: recipe 0.0940 / 0.1004 / 0.0538; ridge-S7 0.1884 / 0.2473 / 0.0816; mean-response 0.1264 / 0.3042 / 0.0858 | Step 14 §2 |
| Step 14 verdict PARTIAL: recipe ahead of ridge-S7 on DE-overlap on all three splits; energy mixed; shift ahead on DPC-50 | Step 14 §1 |
| DE-overlap chance floor on a 200-gene / top-50 panel ≈ 0.206 | findings |

| fact (Tahoe, from the data record) | value |
|---|---|
| gene panel | 200 genes, one panel for all splits (27,099-gene universe, 11,523 eligible) |
| lines | 14 (brain: A-172, H4, SW 1088; gut/gastric: the other 11); 13 on every plate |
| held-out drugs per split (1 / 0 / 2) | 75 / 75 / 76 (of 372 structures) |
| test conditions per split (1 / 0 / 2) | 2,899 / 3,115 / 2,853 |
| training conditions per split (1 / 0 / 2) | 11,464 / 11,248 / 11,510 |
| cells per test condition | median 500 (capped), 5th percentile ≈ 340 |
| training rows per split | ≈ 1.7 M |
| excluded | Sacubitril/Valsartan (no single structure); 77 SW 1088 conditions on plate 13 (no same-plate vehicle) |
| natural replicate | plate 14 replicates plate 6 in full (≈ 1,157 line × drug × dose conditions on both) |

## 3. Arms, per split

Five arms: the three confirmed rungs of the sci-Plex ladder (base, recipe, guidance) and one exploratory pair
(`attn_cross` and its parameter-matched control `mlp_ctrl`, Q4). All use `train_squidiff.py`,
`--use_drug_structure True --gene_size 200 --output_dim 200`, fingerprint drug encoding, on the Step 15a files.
**3 seeds each** (seeds 0, 1, 2), pooled per condition by the arithmetic mean of per-condition metric values, as
every earlier step. Three seeds rather than sci-Plex's five: the ≈ 1.7 M-cell Tahoe training set is ≈ 6.5×
sci-Plex's per split, which lowers per-run variance, and 5 arms × 3 seeds × 3 splits is already 45 H100 runs.

| arm | role | training flags beyond the shared set | sampler at evaluation |
|---|---|---|---|
| **base** Squidiff | Q1 comparator | none (parser defaults: DDPM objective, 1000 diffusion steps, batch 64, lr 1e-4, EMA 0.9999, `lr_anneal_steps` 1e5, `fm_source` none, `cond_drop` 0.0) | `sample_squidiff.sampler`, the base DDIM path, as Steps 1–5 |
| **recipe** | the confirmed top; Q1/Q2/Q3 reference | `--use_adaln True --fm_source gaussian --lr_anneal_steps 100000` (cond_drop 0.0) | Gaussian-source flow matching, 50 NFE, Euler, unguided (`leakage_test_fm_sciplex.py … fm_gaussian 50 euler`), as Steps 6/11/13 |
| **guidance w=2** | Q3 | `--use_adaln True --fm_source gaussian --cond_drop 0.1 --lr_anneal_steps 100000` | the same 50 NFE Euler flow-matching sampler with classifier-free guidance w = 2 (`S12_GUIDANCE_W=2`, `S12_GUIDE_TMIN=0.0`), as Step 12/12b |
| **attn_cross** | Q4, exploratory | `--use_adaln True --fm_source gaussian --encoder_arm attn_cross --lr_anneal_steps 100000` (cond_drop 0.0) | 50 NFE, Euler, unguided (the Step 13 harness; encoder arm detected from the checkpoint) |
| **mlp_ctrl** | Q4 control, exploratory | `--use_adaln True --fm_source gaussian --encoder_arm mlp_ctrl --lr_anneal_steps 100000` (cond_drop 0.0) | 50 NFE, Euler, unguided (the Step 13 harness) |

The recipe is the Step 6/11 configuration exactly; the guidance arm is Step 12b exactly (recipe plus condition
dropout 0.1 at training, guided at sampling — guidance needs the dropout-trained checkpoint and cannot be applied to
the recipe's cond_drop = 0 weights); `attn_cross` and `mlp_ctrl` are the Step 13 configurations exactly (recipe plus
`--encoder_arm`, no condition dropout). `attn_cross` and `mlp_ctrl` enter only as the Q4 pair and never as part of
the recipe; the other Step 9/10 extras (pathway, ChemVAE) and the Step 13 `attn_fp`/`attn_mem` arms are not carried
over. Training runs for 100,000 steps on all five arms, as on sci-Plex; on the ≈ 1.7 M-cell Tahoe training set that
is ≈ 3.8 passes over the data, against ≈ 37 on sci-Plex. Holding the step count fixed keeps each configuration
identical; the epoch difference is a property of the larger dataset, recorded here and in §8, not a change.

**Shift baselines** (for Q2), built as in Step 14 §3, per held-out condition, from that condition's paired vehicle
cells (Step 15a pairing) and a per-condition delta, clipped at 0, one shifted cell per real cell (cap 500):

| baseline | delta |
|---|---|
| **ridge-S7** | interaction ridge (fingerprint + line + fingerprint × line, alpha 1.0), fitted within the held-out compounds by structure-grouped 5-fold CV, exactly Step 7's procedure |
| **ridge-train** | the same ridge fitted on the training split's condition deltas and applied to the held-out conditions |
| **mean-response** | per line, the equal-weight mean over the line's held-out conditions of (condition mean − paired-control mean); the same delta for every condition of the line |
| **control-paste** | delta = 0 |

*Pre-run correction, 2026-09-29 (before any evaluation):* the per-condition cell cap for the shift baselines is **200**, not 500. The harness's `MAX_CELLS_PER_COND` is 200 in the unchanged sci-Plex code that §4 requires, and the baselines are built, exactly as Step 14 §3, as the paired vehicle cells of the 200 real cells the harness stored per condition plus the delta, so every arm and every baseline is scored against the same real cells. The 500-cell test cap of Step 15a still feeds the true condition mean.

**Replicate reference** (descriptive, not a submission): for the ≈ 1,157 conditions on both plate 6 and plate 14,
the plate-6 cells scored against the plate-14 cells of the same condition (and vice versa), giving a genuine
experimental upper reference on each metric. Reported per metric; nothing is gated on it.

## 4. Metrics

The thesis's three, exactly as in every earlier step and computed by the same code:

- **DE-overlap Jaccard** — Jaccard of the true and predicted top-50 genes by |mean delta| against the paired control.
- **Energy distance** — the amendment-13 U-statistic on per-cell predicted vs real expression (lower is better).
- **DPC-50** — Pearson correlation on the true top-50 genes between predicted and true mean delta; reported as the
  mean over conditions and the paired condition median.

All three are computed per condition, then summarised per split and (for reporting) per cell line. The VCC-style
facets of Step 14 are not computed here.

## 5. Endpoints

All intervals: **drug-clustered percentile bootstrap over the split's held-out structures, 5,000 replicates,
`RandomState(0)`, 97.5% per split**, as Step 14. Clustering is by structure, so a stereo/solvate pair (§Step 15a 4.2)
resamples as one unit.

Development vs confirmation, as on sci-Plex: **`drug_split_1` is the development split** — read first, used to
confirm the arms trained and the metrics behave, and the only split on which any unplanned look is allowed.
**`drug_split_0` and `drug_split_2` are confirmation splits**; the outcome categories of §6 require agreement across
all three.

**Primary, gated (Q1):** DE-overlap Jaccard, recipe minus base, per split.

**Co-primary, gated (Q2):** DE-overlap Jaccard, recipe minus ridge-S7 shift, per split.

**Secondary, reported, not gated:** energy distance and DPC-50, recipe minus base and recipe minus each shift
baseline, per split; the **guidance trade-off (Q3): guidance w=2 minus recipe on DE-overlap and on energy jointly**,
per split, with the same 97.5% intervals; every arm's and baseline's absolute value on all three metrics, per split
and per line; the replicate reference on all three metrics; the brain-line (A-172, H4, SW 1088) versus gut-line
contrast on each metric, as an exploratory per-line summary.

**Exploratory (Q4), reported with its interval but carrying no confirmation weight:** `attn_cross` minus `mlp_ctrl`
on DE-overlap, per split, with the same 97.5% drug-clustered bootstrap. This is the attention-versus-parameters
contrast; on sci-Plex it was not above zero on the confirmation splits (§2). It is read on its own, never as
`attn_cross` minus the recipe, and does not enter the outcome categories of §6.

*Pre-run addition, 2026-09-29 (before any evaluation), exploratory, no confirmation weight:* (i) every absolute value is also reported per seed, and the worst seed and the worst split are shown next to the mean; (ii) a per-condition **learnability** descriptor, ‖true − C‖² / (‖true − C‖² + ‖C − C̄_other‖²), with C̄_other the mean paired-control profile of the other cell lines (adapted from Wang et al. 2026, *Adv. Sci.*, eq. 11), summarised per split with Spearman correlations against the recipe's DE-overlap and against the energy gain over control-paste. Neither enters §6.

**Sensitivity.** On sci-Plex the recipe-minus-base DE-overlap gap was ≈ 0.10 and resolved comfortably; the
recipe-minus-ridge gap was ≈ 0.05 and resolved on all three splits at 97.5%. The Tahoe conditions are ≈ 6.5× as many
per split, so intervals should be no wider; no power is claimed in advance for the per-line contrasts, whose
condition counts are smaller.

## 6. Outcome categories, fixed now

Applied to the primary (Q1) and co-primary (Q2) contrasts jointly:

1. **RECIPE GENERALISES** — primary above zero on all three splits (recipe beats base on DE-overlap) and co-primary
   above zero on all three (recipe beats the ridge shift on DE-overlap). The sci-Plex recipe and its advantage over
   a linear baseline both transfer to the brain/gut lines.
2. **RECIPE TRANSFERS, SHIFT COMPETITIVE** — primary above zero on all three, co-primary not above zero on at least
   one split. The architecture generalises over base, but the linear shift matches it on gene ranking, as on
   sci-Plex; the thesis reports the recipe's remaining claim (energy) per §5 secondaries.
3. **NO GENERALISATION** — primary not above zero on at least one split. The recipe's DE-overlap advantage over base
   does not transfer; reported per split, and the base-model behaviour on the new dataset is characterised.
4. **PARTIAL** — a contrast resolved (interval excludes zero) on one or two splits only. Reported per split.
5. **UNRESOLVED** — a gated contrast includes zero on all three splits. Not equivalence.

## 7. Expectation, recorded in advance

I expect **RECIPE TRANSFERS, SHIFT COMPETITIVE**. The recipe's advantage over base on DE-overlap was large and
consistent on sci-Plex (≈ +0.10 on every split) and is architectural, so I expect the primary above zero on all
three Tahoe splits. The ridge shift matched the recipe on DE-overlap on sci-Plex (recipe − ridge-S7 ≈ +0.05, but
the shift was built from the held-out compounds' own responses), so I expect the co-primary positive but small and
possibly unresolved on at least one split. On the secondaries I expect, as on sci-Plex, the shift ahead on DPC-50
and the recipe ahead on energy, though the Tahoe lines are more numerous and more sparsely detected (median panel
detection ≈ 6%), so the absolute levels may differ. For Q3 I expect the sci-Plex **TRADE-OFF to recur**: guidance
w=2 above the recipe on DE-overlap (≈ +0.01) and worse on energy, on most or all splits; if guidance is better on
DE-overlap *and* no worse on energy, the trade-off did not transfer and guidance is the preferred configuration on
Tahoe, which the thesis would then say. For Q4 I expect the sci-Plex result to hold — `attn_cross` **not** above
`mlp_ctrl` on DE-overlap — since the rejection there was clear on both confirmation splits; the reason to run it is
that the 14-line setting is the one condition under which it might not, and a positive, resolved contrast here would
be a new finding worth a dedicated follow-up, not a Step 15b conclusion. I record no expectation for the
brain-versus-gut contrast; it is exploratory. If the primary is not above zero on all three splits, the recipe does
not generalise and the thesis says so plainly.

## 8. What this step cannot establish

- **That 100,000 steps is optimal on Tahoe.** It is the sci-Plex recipe's step count, held fixed to keep the recipe
  identical; on the larger dataset it is fewer epochs (§3). A step or epoch sweep is not part of this step.
- **Cross-dataset transfer of weights.** Nothing is loaded from a sci-Plex checkpoint; only the recipe and protocol
  carry over. Each Tahoe arm is trained from scratch.
- **That the linear shift is reachable in practice.** As in Step 14, ridge-S7 uses the held-out compounds' own
  responses and bounds what a linear model could do, not what a deployed predictor would achieve; ridge-train is the
  like-for-like version and both are reported.
- **A dose-response claim.** Three doses (0.05 / 0.5 / 5 µM) enter through the inherited fingerprint × log10(dose+1)
  encoding; dose is not modelled separately.
- **Anything about the brain–gut axis biology.** The per-line brain-versus-gut contrast is a descriptive summary of
  prediction quality, not a biological result.
- **Independence from the sci-Plex splits.** The Tahoe splits are new, but the recipe and every threshold were fixed
  on sci-Plex; this is a test of a pre-specified recipe on new data, not a fresh model search.

## 9. Analysis plan and provenance

Training: `train_step15_tahoe.sh` (SLURM array, one task per arm × seed × split), committed before it runs, with the
per-task guards of the sci-Plex training scripts (prereg committed, sources clean, parameter count asserted,
provenance sidecar written). Evaluation: `eval_step15_tahoe.sh` driving the Step 13 harness ported to the Tahoe file
prefix and line list, per-cell storage on, condition counts asserted against the data record. Scoring:
`score_step15_tahoe.py`, committed before any result exists, reads the per-cell vectors of all five arms and builds
the four shift baselines and the replicate reference, computes the three metrics of §4, applies §6 mechanically
(Q1/Q2 gated; Q3 and Q4 reported per §5), prints every number, and writes `tahoe_results/step15_model_analysis.json`
and `tahoe_results/step15_model_long.csv`. It stops if any input is missing, if any condition count differs from the
record, or if a shift baseline's DPC-50 does not reproduce its Step-7-style construction.

- Data record: `tahoe_results/step15_data_record.json`, commit 9647935; preparation `prep_step15_tahoe.py`, PREREG_step15a.
- sci-Plex references: `step14_findings.md`; Step 7 ridge `analyse_step7.py`.
- Pre-registration commit: [hash]. Training / evaluation / scoring script commits: [hash].
