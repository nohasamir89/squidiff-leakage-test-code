# Step 27 — condition and compound signal once the plate is uninformative: within-plate scoring with intervals (pre-registration)

Written 2026-11-04, after the Step 26 records and findings were committed (research `fa2dd1d`, package `cb6614e`) and
before any Step 27 quantity was computed. Evaluation only, CPU only: no model is run; every quantity is computed from
the committed Step 15b/24/25 arrays, the committed Step 15a data files, and the arm constructions of Step 26 with
their pre-registered seeds, unchanged. After this file is committed, changes are made only as dated paragraphs,
each written before the data it concerns are read.

## 1. Question

Step 26 found that the published condition discrimination (recipe PDS 0.7223 / 0.7181 / 0.7142 on splits 1 / 0 / 2,
base 0.6415 / 0.6444 / 0.6549) is reproduced, and exceeded, by a no-model prediction that carries only the plate:
the line's pooled control mean (`ctrl_line`, 0.7513 / 0.7335 / 0.7321), and that this survives split-control scoring,
so it is not the shared-estimate bias (S1 ≈ 0.004 PDS at these pool sizes). A condition's PDS candidates sit on a
median of five plates and only ~22 % of candidate pairs share the condition's plate (`step26_findings.md` §3). When
candidates were restricted to the condition's own plate — a pre-registered **descriptive** analysis of Step 26,
recipe arm only, no interval — the recipe's PDS fell to 0.5025 / 0.5177 / 0.5270 (S-orig) and 0.5030 / 0.5172 /
0.5264 (S-split), on 2,645 / 2,678 / 2,532 conditions.

Those three point values have been seen. What has not been measured is their uncertainty, the other arms under the
same restriction, and the mechanism. Step 27 asks, with intervals and for every arm:

1. **Condition signal beyond plate.** Once the plate cannot tell conditions apart, does any model discriminate
   conditions at all?
2. **The recipe-over-base gap.** Does the ~0.08 PDS advantage survive within plate?
3. **Compound signal.** Does the correct compound matter once plate is removed (Steps 24–26 could not resolve this
   under cross-plate scoring, where the plate dominated the ranking)?
4. **The mechanism.** Does the measured change of treated cells, after subtracting their own plate's control mean,
   still point along that plate's control offset — the direct evidence for why a plate-only prediction wins PDS?

## 2. Inputs

Identical to Step 26 §2: the committed per-condition arrays (`true`, `A`, `B`, `C`; stored real and predicted cells)
of the recipe and base arms (3 seeds × 3 splits), the Step 24 and Step 25 swap records, and the Step 15a data files
`tahoe_test_<split>.h5ad` / `tahoe_test_<split>_control.h5ad` under `/scratch/ni2071/Diffusion/tahoe/datasets_step15/`,
sha256-checked against the Step 15 data record. The (line, plate) pools, the S-split halving (`RandomState(0)` per
pool), and the no-model arms are rebuilt by calling the Step 26 scorer's functions with the Step 26 seeds, so every
arm's prediction is bitwise the Step 26 arm. The plate of each condition is read from the committed per-condition
records (`plate` column), as in Step 26.

## 3. Scoring

**Schemes.** S-orig (D = B − C, Y = true − C), S-split (D = B − C1, Y = true − C2) and S-flip (D = B − C2,
Y = true − C1), defined exactly as in Step 26 §3. S-linemean and S-none are not run: under the within-plate
restriction every candidate shares the plate, so the plate-level reference question those schemes addressed is moot.

**Candidate sets.** For condition i, the Step 15c candidate set (same line and dose, other structures, plus i
itself) intersected with the conditions on i's plate; sets with ≥ 10 members (`MIN_CAND` of `score_step15c_tahoe`,
as Step 26 used); conditions in smaller sets are not scored and are counted. Coverage per split is printed and must
equal Step 26's (2,645 / 2,678 / 2,532 of 2,899 / 3,115 / 2,853) — a mismatch is a fault and stops the step. PDS is
computed by the committed `score_step15c_tahoe.pds_sets` on the restricted candidate dictionary; nothing is
reimplemented.

**Arms.** The twelve of Step 26 §4, unchanged (plus `ctrl_sample_h2`, used only as the H5 null, §4): `recipe`, `recipe_fpshuf`, `recipe_fpzero`, `base`, `base_fpshuf`,
`base_fpzero`, `recon_recipe`, `recon_base`, `ctrl_sample`, `ctrl_sample_h1`, `ctrl_line`, `duplicate`. Swap arms are
scored per seed and averaged per condition as in Steps 24–26.

**Plate-residual measure (question 4).** For condition i on plate p of line ℓ, let o_i = L_ℓ − C_i be the plate's
control offset (line pooled control mean minus the (line, plate) pool mean; `ctrl_line`'s predicted change under
S-orig). Per condition and per arm, the cosine between the arm's change vector and o_i:

- under S-orig: cos(X_i − C_i, o_i), and
- under S-split: cos(X_i − C2_i, L_ℓ − C1_i), where the offset is built from the prediction half and the change from
  the measured half, so the two share no control cells (the S-orig version shares C_i on both sides and is inflated
  by the shared estimate; S-split is the primary reading, S-orig is reported).

X_i is `true` for the **measured** residual (the quantity of question 4), and each arm's prediction for the
**predicted** residual (how much of each arm's output is plate offset; `ctrl_line` is 1 by construction under S-orig).
The null for the measured residual is `ctrl_sample_h2` (§4, H5). Cosines are averaged over all scored conditions
(not only the within-plate-covered ones) with the same structure-clustered interval as the PDS contrasts.

**Intervals.** 97.5 % structure-clustered percentile bootstrap, 5,000 replicates, fresh `RandomState(0)` per scored
set (`score_step23_screen.Boot`), as Steps 15c–26.

## 4. Gated contrasts and verdicts

Per split, on within-plate PDS:

- **H1** — `recipe` under S-split − 0.5 (and S-flip; a split "shows above zero" only if both do, Step 26 rule 4).
- **H2** — `recipe` − `base` under S-split, paired per condition.
- **H3** — `recipe` − `recipe_fpshuf` under S-split, paired.
- **H4** — `ctrl_line` under S-orig − 0.5. Within one (line, dose, plate) set `ctrl_line`'s change vector is the
  same for every member, so when every member has a distinct structure its PDS values are a permutation of
  0, 1/(m−1), …, 1 and their mean is exactly 0.5 **by construction**; members sharing a structure (stereo or salt
  twins, Step 23) have smaller candidate sets and move the mean slightly. H4 is therefore a validity null, read
  with the same rule as H4b, and the printed value is expected within a few thousandths of 0.5 — a check that the
  restriction was applied, not a measurement of anything.
- **H4b** — `ctrl_sample_h1` under S-split − 0.5: the non-trivial validity null (expected at chance).
- **H5** — measured plate residual: mean cos(true − C2, L − C1) − 0. Its null is a thirteenth arm used only for
  this measure, `ctrl_sample_h2`: n_i cells drawn from the condition's pool half **h2** (`RandomState(4000000 + i)`,
  without replacement unless h2 is smaller than n_i), whose change X − C2 is independent of the offset's half h1 —
  a resample from h1 would correlate with C1 through its own cells, and one from the full pool with both halves. Its
  cosine is expected to include zero.
- Base analogues of H1 and H3, `recon` arms, `recipe_fpzero`, `duplicate` and all S-orig values: reported ungated.

**Verdict G (condition and compound signal within plate)**, read across the three splits, development first:

1. **NO RESOLVED SIGNAL BEYOND PLATE** — H1 includes zero on all three splits. The models' published discrimination
   is plate recognition to the resolution of this study; H2 and H3 are then reported but carry no claim.
2. **SMALL CONDITION SIGNAL, COMPOUND-FREE** — H1 above zero (both halvings) on at least two splits, H3 not above
   zero on any. Some condition information beyond plate exists; its size is the H1 point estimate; the compound is
   still not resolved.
3. **COMPOUND SIGNAL WITHIN PLATE** — H1 and H3 above zero (both halvings) on at least two splits each. The correct
   compound matters once the plate is removed; Steps 24–26's reading is revised with both scorings shown.
4. **MIXED** — any other pattern; reported split by split, no general claim.
5. **UNINTERPRETABLE** — any reproduction or coverage gate fails, or H4 or H4b excludes zero on two or more
   splits (a one-split exclusion is a caution attached to verdict G, as in Step 26).

**Verdict H (mechanism, descriptive but committed):** H5's sign and size per split, with the predicted residual of
each arm beside it. If H5 is above zero on all three splits and its null includes zero, the manuscript states that
treated cells carry less of their plate's control offset than the controls do, so subtracting the plate's control
mean leaves a plate-aligned residual in the measured change — the route by which a plate-only prediction scores
above chance under cross-plate candidate sets. If H5 includes zero, that route is not supported and the Step 26 §5.1
inference is withdrawn.

**Verdict I (gap attribution, descriptive):** H2's point estimate as a fraction of the published cross-plate gap
(Step 26 `recipe` − `base` under S-orig: 0.0808 / 0.0737 / 0.0593 from the Step 25 records). Stated as "X of the
recipe's advantage over the base model survives within plate".

## 5. Expectations (written before any Step 27 interval or cosine exists; the three recipe within-plate point
values are already known from Step 26 and are excluded from what counts as a prediction)

- H1: the point estimates are +0.0030 / +0.0172 / +0.0264 (S-split). I expect split 1's interval to include zero
  (confident) and genuinely do not know whether splits 0 and 2 exclude it; the Step 26 cross-plate intervals had
  half-widths of ~0.03–0.04, and within-plate sets are smaller, so I lean to "includes zero" on all three (outcome 1)
  with low confidence.
- H2: recipe above base within plate by a small amount, interval including zero — low confidence either way; the
  base's cross-plate advantage over `ctrl_sample` came from the same plate route, so most of the gap should vanish.
- H3: includes zero on all splits — moderately confident (Steps 24, 25, 26 G3, all including zero).
- H4 within a few thousandths of 0.5 with an interval including zero on every split — confident (exact only
  without structure twins); an interval excluding zero is a coding fault.
- H4b includes zero everywhere — confident (Step 26 G5 did); the `ctrl_sample_h2` cosine null likewise.
- H5 above zero on all three splits with the null including zero — moderately confident; it is the only identified
  route to `ctrl_line`'s 0.75 under S-split, but it is an inference from scores and this is its first direct test.
- Predicted residual: `ctrl_line` cosine = 1 by construction under S-orig; model arms between 0 and the `true`
  value; `ctrl_sample` ≈ 0. Low confidence on the models' position.
- `duplicate` within plate: well above 0.5 (confident), lower than its cross-plate 0.95 (moderately confident —
  part of the ceiling was plate too).

## 6. Guards and reproduction checks (before any Step 27 value is printed)

1. This file, `score_step27_tahoe.py` and `run_step27_score.sh` committed and identical to HEAD; Step 26 records and
   findings committed; no Step 27 output exists.
2. Rebuild gates of Step 26 §7.2 re-run (stored `true` and `C` from the data files to 1e-4; stored cells recovered
   bitwise; halves disjoint and exhaustive; pool sizes and flags printed).
3. **Reproduction:** cross-plate S-orig PDS of `recipe` and `base` equals the committed Step 15c per-condition
   record to 1e-9; every arm's cross-plate PDS under S-orig and S-split equals the committed `step26_source.json`
   value to 1e-9 — per condition, against `step26_source_per_condition.csv`, so the pools, halving, no-model draws
   and ceiling are shown to be rebuilt bitwise; the within-plate `recipe` values equal Step 26's recorded
   0.5025 / 0.5030, 0.5177 / 0.5172, 0.5270 / 0.5264 (full precision from `step26_source.json`, to 1e-9) and the
   coverage counts match exactly; the Step 15c published intervals replay exactly
   (`score_step25_tahoe.replay_step15c`, 5e-6).
4. The within-plate restriction is applied to every arm through the same candidate dictionary; the `ctrl_line`
   change vector is identical across the members of every within-plate set (checked to 1e-12).
5. Outputs written once (`step27_within_plate.json`, `step27_within_plate_per_condition.csv`,
   `step27_within_plate_summary.csv`), never overwritten; log to `tahoe_results/step27_logs/`.

## 7. Scripts, records, run order

1. Committed before any run: this file, `score_step27_tahoe.py` (imports the Step 15, 15c, 23, 25 and 26 scorers;
   reimplements nothing), `run_step27_score.sh` (CPU, compute partition, as `run_step26_score.sh`).
2. Synthetic test before the real run (36 structures on 3 plates, 144 conditions per split; the Step 26 scorer run
   on the same mock first so its records exist): (a) plate response only, (b) a planted within-plate condition signal
   without compound signal, (c) a planted compound signal, (d) a planted plate residual with no condition signal —
   recovered as verdict G outcomes 1, 2, 3 and 1, with verdict H "supported" in (d) only and never "supported" in
   (a)–(c); nine negative tests (existing output, uncommitted script, tampered Step 26 per-condition value, tampered
   coverage count, tampered within-plate value, missing Step 26 findings, tampered replay target, a command-line
   option, an incomplete pre-registration) each stop the step.
3. One CPU job. Findings in `step27_findings.md`; records committed to the three repositories as Steps 24–26 were.
   No rerun with changed settings; a fault fixed after a run is reported as such, with both runs kept.

## 8. Limits

- The within-plate restriction drops 9–13 % of conditions (sets < 10) and shrinks the candidate sets, so intervals
  are wider than the cross-plate ones; a verdict of "includes zero" bounds the signal, it does not prove zero, and
  the interval's upper limit is stated wherever the verdict is.
- The restriction removes plate as a discriminating variable; it does not remove any other structure shared within a
  (line, dose, plate) set, nor does it address S-linemean's question of what the models' outputs are closest to.
- H5 measures alignment with the plate's control offset, one direction in gene space; a plate × treatment interaction
  along other directions is not measured.
- The offset is one vector per (line, plate) pool, so the residual cosine varies mostly between pools, and its
  structure-clustered interval does not resample pools. With 180 pools per split that uncertainty is small but not
  zero; the per-condition record carries the plate column so a verdict H resting on a few pools would be visible.
  The synthetic test (six pools per split) showed H5 excluding zero on single splits in both directions when no
  residual was planted, which is why verdict H requires all three splits and the null together.
- The Step 26 descriptive point values for the recipe were seen before this file was written; §5 says so, and the
  verdict rules do not depend on them beyond the reproduction gate.
- As in Step 26: one halving per pool, one draw per no-model arm, each under its pre-registered seed; S-flip is the
  agreement check, not a second sample.
- This step bounds the within-plate condition and compound signal of the committed predictions; it says nothing about
  what a model trained or scored differently could achieve. Nothing stronger than §4's verdict sentences will be
  written.
