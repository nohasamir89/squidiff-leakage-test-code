# Step 26 — where does the compound-free discrimination come from? Split-control scoring, no-model baselines and calibration (pre-registration)

Written 2026-11-02, after the Step 25 reproduction gates passed (job 19024369, five bitwise-identical lines) and
before any Step 25 swap value was computed or read; the 30 swap runs (job 19029254) were in the queue when this file
was written. Evaluation only, CPU only: no model is run, and every quantity is computed from committed Step 15b/24/25
arrays plus the committed Step 15a data files. After this file is committed, changes are made only as dated
paragraphs, each written before the data it concerns are read.

## 1. Question

Step 24 found on the development split that the rewired recipe's PDS does not require the compound: 0.7223 with its
own fingerprint, 0.7160 with another held-out compound's, 0.7360 with none (records at 35af542). Step 25 asks whether
that replicates and whether the base model behaves the same way; its verdicts are not inputs here, and every
question below is defined whatever they turn out to be, because `fpshuf` PDS far above 0.5 is already established on
the development split and is the thing needing a source.

Three candidate sources, not mutually exclusive:

- **S1, shared control estimate.** The scoring subtracts the same control mean C from the predicted and the measured
  change. Reusing one finite control sample on both sides of a comparison inflates similarity metrics even for
  non-informative predictions (Nicol et al. 2026, bioRxiv, doi:10.64898/2026.05.07.723486; shared reference shifts,
  Mejía et al. 2025, arXiv:2506.22641; systematic variation, Viñas Torné et al. 2025, Nat Biotechnol,
  doi:10.1038/s41587-025-02777-8). In this pipeline C is the mean of all vehicle cells of the condition's (line,
  plate) pool (`leakage_test_base_tahoe.py`, sha256 caddf58d…, condition loop: `predC = pool.mean(0)`), so this
  route requires candidate sets that span plates — candidates sharing one plate share one C, which cancels in the
  ranking. Whether the sets span plates is an empirical fact of the data layout, reported before any verdict is read.
- **S2, conditioning information.** The model's prediction B is conditioned on cfB, a per-condition draw of n vehicle
  cells from the same (line, plate) pool. A generative model can carry line- and plate-identifying state from cfB
  into B without any compound information, and that is real model behaviour, not a scoring fault.
- **S3, compound signal masked by S1.** A compound-specific component could exist that shared-control noise hid in
  Steps 24/25. The design must be able to see this even though Step 24 gave no sign of it.

Step 26 separates S1 from S2+S3 by re-scoring the committed predictions under scoring schemes that remove the shared
estimate (Nicol's control splitting), and bounds S1's size with no-model baselines and a split-half ceiling
(interpolated duplicate and dynamic range, Miller et al. 2025, bioRxiv, doi:10.1101/2025.10.20.683304; positive and
negative calibration baselines, Mejía et al. 2025). A candidate-set restriction of our own construction is run
alongside as convergent evidence.

## 2. Inputs and populations

**Committed arrays** (hash-checked against provenance as in Steps 24/25): the Step 15b vectors and per-cell files of
the recipe and base arms (3 seeds each, 3 splits), the Step 24 swap records (split 1, recipe `fpshuf`/`fpzero`,
3 seeds), the Step 25 swap records (the remaining split × arm × seed cells of the Step 25 design table). Each
vectors file stores, per condition: `true` (mean over all the condition's treated cells in the test file), `A`
(prediction with the encoder shown the treated cells), `B` (prediction with the encoder shown cfB; the published
arm), `C` (the (line, plate) pool mean). Each per-cell file stores the predicted cells and the stored real cells
(n ≤ 200, drawn without replacement).

**Data files** (sha256 re-checked against the Step 15 data record): `tahoe_test_<split>.h5ad` and
`tahoe_test_<split>_control.h5ad` under `/scratch/ni2071/Diffusion/tahoe/datasets_step15/`. The control file's cells
are pooled by (line, plate) exactly as the harness pools them; pool sizes are reported per split (minimum, median,
and the count of pools smaller than 2n of their largest condition).

**Rebuild gates** (§7.2) force this file's reading of the harness to be true before anything else is computed: the
per-condition mean over all test-file cells must equal stored `true`, and the (line, plate) pool mean must equal
stored `C`.

## 3. Scoring schemes

All schemes feed the same committed library functions used for every published number — `score_step15c_tahoe.pds_sets`
on change matrices, `score_step15_tahoe.thesis_per_cond` with the reference argument set to zero vectors so its
internal subtraction reproduces the scheme's changes, and the Step 25 scorer's MRA computation — with different
change matrices as inputs. Nothing is reimplemented. Energy distance compares cell populations and never touches C;
the committed energy values stand under every scheme and are not recomputed.

Per split, with D the predicted-change and Y the measured-change matrix:

- **S-orig** (reproduction): D = B − C, Y = true − C. Must reproduce the committed records (§7.3); every other
  scheme is read only after it does.
- **S-split** (primary; Nicol): each (line, plate) pool's cells are shuffled once by `RandomState(0)` and split into
  two disjoint halves, h1 and h2 (odd cell to h1); C1 = mean(h1), C2 = mean(h2), shared by that pool's conditions
  exactly as C is. D = B − C1, Y = true − C2. The predicted and measured changes then share no control cells.
- **S-flip** (reference): D = B − C2, Y = true − C1. Reported to show the half assignment does not matter.
- **S-linemean** (reference): C replaced by the mean over all control-file cells of the condition's line in that
  split, pooled across its plates. Separates plate-level from line-level reference structure.
- **S-none** (reference): D = B, Y = true. No subtraction.

**Candidate-set restriction** (our construction, convergent with S-split): under S-orig and S-split, PDS recomputed
with each condition's candidates restricted to the same (line, dose, plate), sets with ≥ 2 members. Within such a
set every candidate shares C exactly, so S1 is structurally absent; surviving discrimination is S2/S3. The number of
covered sets and conditions is reported; if coverage is thin this stays descriptive, which is why S-split is primary.

**Plate-structure report** (required in the findings before any verdict): per split, the distribution of distinct
plates per (line, dose) candidate set and the fraction of candidate pairs sharing a plate.

## 4. Arms

Per split, scored under every scheme; swap arms per seed and averaged per condition as in Steps 24/25:

| arm | construction |
|---|---|
| `recipe`, `base` | committed B of the own-fingerprint runs (3 seeds) |
| `recipe_fpshuf`, `base_fpshuf` | committed B of the swap runs (3 seeds) |
| `recipe_fpzero`, `base_fpzero` | committed B of the zero runs (3 seeds) |
| `recon_recipe`, `recon_base` | committed A (encoder shown the treated cells; reference only) |
| `ctrl_sample` | no model: prediction_i = mean of n_i cells drawn from the condition's full (line, plate) pool, without replacement unless the pool is smaller than n_i, `RandomState(1000000 + i)` with i the condition's row index and n_i the condition's stored per-cell count — cfB's construction with a fresh seed |
| `ctrl_sample_h1` | no model: as `ctrl_sample` but drawn from h1 only, `RandomState(2000000 + i)`; under S-split it shares no cells with C2 and is a negative control for the scheme itself |
| `duplicate` | ceiling: the stored real cells of each condition are shuffled by `RandomState(3000000 + i)` and split into halves T1/T2; D_i = mean(T1) − (the scheme's prediction-side reference), Y_i = mean(T2) − (the scheme's measured-side reference), both matrices built this way for every condition (split-half reliability of the measured changes; interpolated-duplicate construction, Miller et al. 2025) |

Conditions whose (line, plate) pool holds fewer than 20 cells, or fewer than 4 stored real cells, are flagged and
counted; all conditions are scored (as in Steps 24/25) and the gated contrasts are additionally reported with the
flagged conditions removed, as a sensitivity line.

## 5. Measures, gated contrasts and verdicts

Measures per scheme: PDS (chance 0.5), DE-overlap, DPC-50, MRA. Intervals: 97.5 % structure-clustered percentile
bootstrap, 5,000 replicates, fresh `RandomState(0)` per scored set, as Steps 15c–25. Gates are on PDS; the other
measures, the per-dose and salt-twin breakdowns, S-flip, S-linemean, S-none, the recon arms and the base-model
analogues of G1–G3 are reported ungated.

Per split, on PDS:

- **G1** — `recipe` under S-split − 0.5.
- **G2** — `recipe_fpshuf` under S-split − 0.5.
- **G3** — `recipe` − `recipe_fpshuf` under S-split (paired per condition).
- **G4** — `ctrl_sample` under S-orig − 0.5.
- **G5** — `ctrl_sample_h1` under S-split − 0.5 (scheme validity: expected chance by construction).

**Verdict D (source of the published discrimination)**, read across the three splits, development first; a split
"shows" a pattern when its interval supports it:

1. **S1-DOMINATED** — G1 includes or lies below zero on all three splits while G4 is above zero. The published PDS
   discrimination is the shared-control bias; PDS as published measured neither compound nor condition specificity.
2. **CONDITION-REAL, COMPOUND-FREE** — G1 above zero with G3 including or below zero. Real discrimination survives
   unbiased scoring and still does not need the compound: the model carries condition-identifying state from its
   control input (S2). G4's size then states how much of the published number was S1 on top of it.
3. **COMPOUND SIGNAL RESTORED** — G1 above zero and G3 above zero on all three splits. Removing the shared estimate
   reveals a compound-specific component that S1 noise masked; Steps 24/25's reading is revised in the manuscript
   with both scorings shown.
4. **MIXED** — different splits show different patterns; reported split by split, no general claim.
5. **UNINTERPRETABLE** — any reproduction or rebuild gate fails, or G5 excludes zero on any split (the scheme's own
   negative control failing means a fault, not a result).

**Verdict E (baseline attribution, descriptive but committed):** G4's point estimate as a fraction of the recipe's
S-orig PDS − 0.5, with the structure-clustered interval on the paired per-condition difference recipe − `ctrl_sample`
under S-orig, per split. Stated in the manuscript as "a no-model vehicle resample scores X of the published effect".

**Verdict F (calibration, descriptive):** the `duplicate` ceiling under S-orig and under S-split per split, and each
model arm's position as a dynamic-range fraction (arm − `ctrl_sample`) / (`duplicate` − `ctrl_sample`) within each
scheme (Miller et al. 2025). The S-orig − S-split ceiling drop is reported as the shared estimate's contribution to
the ceiling itself.

## 6. Expectations (written before any Step 26 value; Step 24 §6 was wrong, so confidence is stated per item)

- Plate report: I expect candidate sets to span several plates — without that, no identified route explains the
  Step 24 `fpzero` result — but this is a layout fact I have not seen, so low confidence.
- G4 above zero and substantial: moderately confident (Nicol's result, and `fpzero` at 0.7360 with no compound and
  no dose).
- G1: genuinely uncertain between outcome 1 (includes zero) and outcome 2 (above zero but well below the published
  +0.22). I decline to predict which; both rewrite the manuscript in the ways §5 states.
- G2 tracks G1 within its interval: moderately confident (Steps 24/25 show own ≈ fpshuf).
- G3 includes zero on all splits: moderately confident from Step 24; above zero would be outcome 3 and I record now
  that I do not expect it.
- G5 includes zero everywhere: confident; anything else is a coding fault and stops the step.
- Duplicate ceiling higher under S-orig than S-split: confident in direction, no size prediction.

## 7. Guards and reproduction checks (before any Step 26 value is printed)

1. This file, the scorer and its launcher committed and identical to HEAD; the Step 24 and Step 25 records and
   findings committed first; every input array hash-checked against its provenance; no Step 26 output exists.
2. **Rebuild gates:** per condition, the mean over all its test-file cells equals stored `true`, and its (line,
   plate) control-pool mean equals stored `C`, each to 1e-4 in float32 with the max deviation printed; the stored
   real cells are recovered in the test file; pool sizes and flags of §4 computed and printed. A failure stops the
   step — it would mean this file's reading of the harness is wrong.
3. **Reproduction:** S-orig rescoring reproduces the committed per-condition records to 1e-9 — recipe and base PDS
   (Step 15c), DE-overlap and DPC-50 (Step 15b), the Step 24 and Step 25 swap per-condition records — and the
   bootstrap of recipe and base PDS − 0.5 under S-orig reproduces the Step 15c published intervals (the §7.7
   constants of PREREG_step25_swap_confirmation.md). No S-split, baseline or ceiling number is read before this
   passes.
4. The S-split halves are disjoint and exhaustive within every pool; every draw uses the §4 seeds; the per-arm
   change matrices differ between arms at every condition except where §4 defines them to coincide.
5. Outputs written once (`step26_source.json`, `step26_source_per_condition.csv`, `step26_source_summary.csv`),
   never overwritten.

## 8. Scripts, records, run order

1. Committed before any run: this file, `score_step26_tahoe.py`, `run_step26_score.sh`. No harness, no permutation,
   no GPU job: the step reads committed outputs and the Step 15a files only.
2. Step 25 is scored and its records and findings committed first; Step 26's scorer refuses to run otherwise.
3. One CPU job (compute partition, as `run_step25_score.sh`), tested beforehand on synthetic data in which each
   source S1/S2/S3 is planted alone and must be recovered, and every guard is exercised.
4. Findings in `step26_findings.md`; records committed to the three repositories as Steps 24/25 were. No rerun with
   changed settings; a fault fixed after a run is reported as such, with both runs kept.

## 9. Limits

- One halving realization per pool and one draw per baseline, each under a pre-registered seed; no interval covers
  the choice of split or draw (Nicol's procedure has the same property). S-flip shows the halving's other
  assignment; agreement between them is reported, not assumed.
- S-split removes the shared control estimate; it does not and cannot remove S2 — that separation is exactly its
  purpose, so a surviving G1 attributes discrimination to the conditioning route only as the remaining identified
  source, and §3's candidate-set restriction is the convergent check.
- `true` averages all of a condition's test-file cells while the `duplicate` ceiling uses the stored ≤ 200; both
  counts are reported, and the ceiling is a property of the stored sample.
- Pools smaller than twice a condition's n give thin halves and a noisier C1/C2; flagged, counted, and covered by
  the sensitivity line of §4.
- `fpzero` remains outside both models' training distribution, and zeroing the fingerprint still zeroes dose
  (encoder entanglement, Step 24 §2); unchanged here because no model is re-run.
- This step attributes the discrimination among identified sources; it does not prove the absence of an
  unidentified one. What it can end with is stated per verdict in §5, and nothing stronger will be written.

## Amendment, 2026-11-02 — the S1 probe (written after the synthetic test and before any real Step 26 quantity;
no Step 25 swap value had been computed either)

The §8.3 synthetic test, run before any real data was read, showed that `ctrl_sample` cannot detect S1, for an
exact reason: for any prediction built as a weighted average of the pool's own cells (weights summing to one),
cov(P − C, −C) = −v/N + v/N = 0, so a pool resample — and any model output that merely reweights its control
input — carries no shared-estimate advantage under S-orig by construction. The shared estimate rewards only
predictions that are (partly) independent of the pool sample, such as a smooth model output or an out-of-pool
control estimate, where the +var(C) term survives. Three changes, none of which read any data:

1. A twelfth arm is added: `ctrl_line` — no model; prediction_i = the condition's line's pooled control-file mean
   (the S-linemean reference CL used as a prediction, under each scheme's own references). It is compound- and
   plate-blind, and out-of-pool by the fraction 1 − N_pool/N_line, so under S-orig it carries the shared estimate;
   under S-split it does not. It is the S1 probe.
2. **G4 is redefined** as `ctrl_line` under S-orig − 0.5. The old contrast, `ctrl_sample` under S-orig − 0.5, is
   kept and reported as **G4b**, the input-copying null: by the identity above it is expected at chance under
   every scheme, so it joins G5 as a validity check. Because each null is at chance in expectation but its single
   pre-registered draw is not resampled by the structure bootstrap, a lone 97.5 % interval excluding zero is a
   ~2.5 % event per null per split, and with two nulls on three splits a one-split exclusion must not void the
   step: verdict outcome 5 fires when G5 or G4b excludes zero on **two or more** splits; a single-split exclusion
   is reported as a caution attached to verdict D and does not change it (this rule was set from the synthetic
   test, before any real value existed). §5's verdict D and §6's expectations read G4 as the `ctrl_line` probe.
   The §6 expectation for the old G4 is replaced by: G4b includes zero everywhere (confident, by construction);
   G4 (ctrl_line) above zero if and to the extent that the shared estimate is large enough to matter at the
   pools' sizes — genuinely uncertain, and that size is exactly what the step measures.
3. Verdict E's attribution baseline becomes `ctrl_line` (the no-model prediction that does carry S1), reported as
   before; the `ctrl_sample` fraction is reported alongside as the copying-null reference.

**Note, 2026-11-04 (before any Step 25 swap value or Step 26 value existed):** §7.3's reproduction of the Step 15c
published intervals is done by replaying Step 15c's bootstrap draw order with its own functions
(`score_step25_tahoe.replay_step15c`), not by a fresh generator — see the amendment of the same date in
PREREG_step25_swap_confirmation.md for the cause and the proof. The gate stays exact at 5e-6.
4. The halving introduces draw noise clustered at the plate level, which the structure bootstrap does not resample,
   and the synthetic test showed it can push a single split's G1 interval above zero when no real signal exists.
   Therefore a split counts as showing "G1 above zero" (and likewise "G3 above zero") only when the S-split
   interval is above zero AND the same contrast under S-flip is above zero — a survival or compound claim must
   hold under both half assignments. "Not above zero" needs no agreement, which makes the rule conservative
   toward outcomes 1 and 2 and against outcome 3; the same rule applies to the base-model analogues. Both
   schemes' intervals are reported in full either way.
