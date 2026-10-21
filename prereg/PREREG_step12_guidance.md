# Pre-registration: Step 12, classifier-free guidance on the drug condition

**BLIND.** No model has been trained with condition dropout and no guided sample
has been drawn. Every number quoted in section 2 is a stored result from an
earlier step; nothing in this document was computed for it.

Author: Noha Samir Ismail. Date: 2026-09-20.

The reference arm's training configuration, commit and data construction were
read from `train_step10.sh`, its job log and `scrna_datasets.py` before this was
written; they are recorded in section 9.

## 1. Question

Every diagnostic in this thesis that asks how much the drug moves the output has
returned "not much". On the development split, substituting a wrong fingerprint
changed the semantic-adaLN DDIM output by a median 1 − r of 0.0734 (Figure 4),
and excess drug variance fell from 0.4493 in the latent to 0.0734 at the output,
a median ratio of 5.83× (eta15). Both measurements are on the DDIM arm; no
equivalent exists for flow matching, so the premise is carried over, not shown.

scDiffusion (Luo, Hao, Wei & Zhang, *Bioinformatics* 2024;40(9):btae518) steers
sampling toward a requested condition with the gradient of a separately trained
classifier. A classifier needs a class per condition, and this thesis is scored
on compounds the model never saw, so that form cannot be used. Its classifier-
free form (Ho & Salimans, 2022) can: train with the condition randomly hidden,
and at sampling time step along v_null + w·(v_drug − v_null), which for w > 1
moves further in the direction that distinguishes the drug. The direction is the
model's own; guidance changes nothing about what the model has learned, only how
hard the sampler leans on it.

**Does leaning harder on the drug direction at sampling time raise DE-overlap,
and at what cost in energy distance?**

The test-loss analysis (exploratory, `test_loss_v2.json`) bears on why this is
the intervention chosen: the training objective is met on held-out compounds
almost as well as on seen ones, so the failure is not one of fitting the
objective, and a change that leaves the fit untouched and acts only on sampling
is the appropriate kind.

## 2. What exists, and its status

| quantity | value | source |
|---|---|---|
| flow matching, fingerprint, drug_split_1, 20 NFE, DE-overlap | 0.5888 | Step 10 arm A (`fingerprint_s10`), 5 seeds |
| same, 50 NFE | not computed | — |
| Step 6 flow matching, drug_split_1, 50 NFE | 0.6100 | Step 6 sweep, later confirmed in Step 11 on held-out splits |
| energy distance, `fingerprint_s10`, 20 NFE | 0.1859 | Step 10 |
| control-source OT-FM energy distance penalty | 51% worse than Gaussian source | Step 6, arm dropped |

The last row is the reason energy distance is co-primary here: the one earlier
change that pulled samples toward a condition-specific point matched Gaussian
flow matching on DE-overlap and paid for it in distributional fidelity.

## 3. The intervention

**One new arm, five seeds, one split.** Gaussian-source flow matching with the
semantic-adaLN backbone, Morgan-1024 dose-scaled fingerprint, `drug_split_1`,
100,000 steps, uniform time sampler, EMA 0.9999 — identical to `fingerprint_s10`
in every respect but one: during training, for a random 10% of rows in each
batch, `drug_dose` is replaced by the all-zero vector before `x_start` is
rebuilt. The all-zero vector is never a real condition: every training target
has dose 10–10,000 and the fingerprint is scaled by log10(dose + 1), so zeros
occur nowhere in the training data. Trained this way, the null input learns the
model's average treated response for the given control cell.

| | |
|---|---|
| new training | 5 seeds, `drug_split_1`, `train_step10.sh` arm-0 flags plus `--cond_drop 0.1` |
| reference arm | `fingerprint_s10`, 5 seeds, no dropout, commit `8d5b65f` |
| data construction | `--drug_encoding fingerprint_s10` on both: this string triggers the Step 10 three-compound exclusion in `prepared_data` and then resolves to the plain fingerprint encoder; with `--drug_encoding fingerprint` the filter would not fire and the arms would see different data |
| sampling | 50 NFE Euler (Step 11 configuration), guided at w ∈ {1, 1.5, 2, 3, 5, 8} |
| evaluations | 5 × 6 guided + 5 reference at 50 NFE = 35 |
| decoder seed | fixed, as in every step |

**Guided sampling.** At each Euler step the model is evaluated twice, with the
fingerprint and with the null; the step uses v_null + w·(v_drug − v_null).
w = 1 is the unguided model. Two evaluations per step, so 50 NFE costs 100.

**Gated value of w: 2, fixed now.** There is no data on this problem to choose
from; 2 is a conventional mild weight and is fixed before any run. Whether a
different w does better is a secondary question answered by the sweep (section
5), and any value the sweep prefers is a development-split observation with the
same status the 50-NFE value had before Step 11.

**Implementation assertions, part of the run and not optional.**
- Dropout is applied before `x_start` is rebuilt from `control_feature` and
  `drug_dose` (the custom objective reinserts `x_start`; masking after that
  point would leave the fingerprint in the encoder input). Asserted per batch:
  the fingerprint half of `x_start` is exactly zero on masked rows.
- A guided run at w = 1 must reproduce the unguided vectors to within 1e-6;
  the identity v_null + 1·(v_drug − v_null) = v_drug is the check that the
  guided code path is wired correctly. Run before any w > 1 is sampled.
- Parameter count must equal 58,178,180, the `fingerprint_s10` guard value:
  dropout adds no parameters, so any other count means the wrong model.
- Control-arm prediction magnitude assertion from Step 10 stays in the
  evaluation path.
- `guidance_w` and the null definition are written to the provenance sidecar
  and bound to the checkpoint hash, as `mode` already is.

## 4. Endpoints

**Primary, gated: DE-overlap Jaccard**, guided (w = 2) minus unguided (w = 1) on
the **same five dropout-trained checkpoints**, paired per condition, with a
drug-clustered percentile bootstrap over the split's 37 compounds, 5000
replicates, `RandomState(0)`, 95% interval. Pairing on the same checkpoints
isolates guidance from training.

**Co-primary, gated: energy distance** (amendment-13 U-statistic), the same
contrast. This endpoint decides whether a DE-overlap gain is a gain or a trade.

**Secondary, reported and not gated:**
- the w sweep, all six values, both endpoints;
- dropout itself: unguided dropout arm (w = 1) minus `fingerprint_s10`, both at
  50 NFE, different checkpoints, drug-clustered bootstrap — whether hiding the
  fingerprint 10% of the time costs anything by itself;
- DPC-50 and B − C at every w, reported;
- **EXPLORATORY**, one evaluation per seed: guidance applied only for t ≥ 0.5
  (the data half of the trajectory) at w = 2, following scDiffusion's practice
  of guiding only in the low-noise half of its chain.

**Sensitivity.** No minimum detectable effect is quoted in advance: the contrast
is paired within checkpoint and no earlier step has that structure to borrow
from. It is computed from the w = 1 and w = 2 vectors by the Step 6b procedure,
reported next to the result, and not used to interpret it.

## 5. Outcome categories, fixed now

1. **GUIDANCE HELPS** — DE-overlap interval above zero AND energy-distance
   interval not above zero (guidance does not worsen fidelity).
2. **TRADE-OFF** — DE-overlap interval above zero AND energy-distance interval
   above zero. Reported as a trade, not a gain. The sweep is then read for
   whether a smaller w keeps the gain without the cost; that reading is a
   development-split observation and not a confirmed result.
3. **UNRESOLVED** — DE-overlap interval includes zero. Not equivalence; no
   margin was set.
4. **GUIDANCE HARMS** — DE-overlap interval below zero.

**On the dropout secondary:** if the unguided dropout arm is resolved below
`fingerprint_s10` by more than the gated gain, the gain is recovering what the
dropout cost, and the primary result is reported with that sentence attached.

**Confirmation.** A result in category 1 or 2 is a development-split result. It
becomes confirmable only by training dropout arms on `drug_split_0` and
`drug_split_2` under a Step 12b pre-registration at the same w = 2. Both splits
have been used three times (Steps 4, 6b, 11); a fourth use is recorded as such.
No confirmation is run for categories 3 or 4.

## 6. Expectation, recorded in advance

**I expect UNRESOLVED at w = 2, with energy distance rising monotonically in w
across the sweep.** The drug direction the model holds is faint and, on the
evidence of Step 3, largely not the true one — ridge on the same fingerprint
reaches a DPC-50 of 0.1090 on this split against the model's 0.0215.
Amplifying a direction that is mostly wrong should not raise DE-overlap much,
and guidance is known to trade sample diversity for adherence to the condition,
which is what energy distance measures.

If GUIDANCE HELPS occurs, the interesting quantity is the sweep: whether the
gain saturates at small w (a faint but correct direction) or keeps growing
(the model held more drug information than its samples showed).

## 7. What this cannot establish

- **Anything about held-out compounds beyond the development split.** This is
  a `drug_split_1` experiment; section 5 says what confirmation would take.
- **Anything about DDIM.** Guidance is tested on the flow-matching arm only.
- **That the null is a well-formed marginal.** One dropout rate, 0.1, chosen by
  convention; the null's quality is not measured separately.
- **Optimality of w.** The gate is at 2 by fiat; the sweep is descriptive.
- **Interaction with NFE.** Everything is at 50 NFE.
- **Anything about DPC-50 as a gated claim**, on which every model in this
  thesis loses to a linear baseline; the ridge value above is the standing
  comparator whatever guidance does to DE-overlap.
- **That the scDiffusion result transfers.** Their classifier guidance targets
  cell-type labels with a class per condition and is evaluated on realism of
  known classes; nothing in their paper concerns unseen conditions, and none of
  its numbers bear on this design.

## 8. Analysis plan, fixed now

`analyse_step12.py`, written and committed before the guided vectors exist,
reads the vector files, computes both primary contrasts and every secondary,
writes `leakage_results/step12_analysis.json` with commit, vector hashes and
`guidance_w` per input, and refuses to run if any expected vector file is
missing or any sidecar's `guidance_w` disagrees with its filename.

## 9. Provenance

- Reference arm: `step10/fingerprint_s10/seed_{0..4}/model_0.9999.pt`, trained
  under commit `8d5b65f` (job log `logger_files/s10_17949440_0.out`), by
  `train_step10.sh` array task 0 with
  `--use_adaln True --fm_source gaussian --drug_encoding fingerprint_s10
  --use_drug_structure True --gene_size 200 --output_dim 200
  --lr_anneal_steps 100000`, parameter guard 58,178,180.
- Data construction: `prepared_data` in `Squidiff/scrna_datasets.py` applies
  the three-compound exclusion when `drug_encoding` is `chemvae`, `fp_chemvae`
  or `fingerprint_s10`, filters the paired control file by the same row mask,
  and then maps `fingerprint_s10` to the fingerprint encoder. The new arm passes
  the same string, so its training data are identical to the reference's.
- New flag `--cond_drop` (float, default 0.0, so the `bool("False")` parser trap
  does not apply); the parsed value is printed by the guard block and asserted
  equal to 0.1 before training starts.
- Training commit for the new arm: [hash at submission]
- Pre-registration commit: [hash]
