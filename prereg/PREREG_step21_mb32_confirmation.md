# Pre-registration — Step 21: does minibatch OT at *m* = 32 keep full OT's realism at less cost in compound discrimination? A confirmation on splits 0 and 2

Date: 2026-10-12. Written before any Step 21 checkpoint exists. Inputs: the Step 20 records (`step20_findings.md`,
`tahoe_results/step20_analysis_development.json`); the Step 18 records (`step18_findings.md`,
`tahoe_results/step18_analysis.json`, `tahoe_results/step18_per_condition.csv`, the `s18_*` and `s18r_*` arrays); the
minibatch pairing record (`tahoe_results/minibatch_pairing_check.json`, commit f85ea15); and the two *m* = 32 control
files written for this step by the committed `check_minibatch_pairing_tahoe.py` (commit eff074f; job 18772868), each
sha256-pinned in `tahoe_results/minibatch_pairing_written_drug_split_0_m32.json` and
`tahoe_results/minibatch_pairing_written_drug_split_2_m32.json`, committed together with this document. Section
numbers refer to this document unless another document is named.

## 1. Question

Step 18: exact OT pairing of the training pairs ("full OT") brings the generated populations closer to the real ones
(energy) on all three splits, at a cost in DE-overlap and PDS. Step 20, on the development split only, weakened the
matching with minibatch OT; no setting avoided the DE-overlap cost (verdict 0). One secondary result stood out. At
*m* = 32, energy could not be told apart from full OT's (*m* = 32 − full OT: +0.00011 [−0.00469, +0.00446]), while PDS
was higher (+0.01896 [+0.01194, +0.02670]); and at all three settings the PDS cost was smaller than the straight line
from the recipe to full OT predicts (Step 20 findings §2–§3).

Step 21 asks whether this holds on the two splits whose test data Step 20 never read: **does *m* = 32 keep full OT's
realism while losing less compound discrimination?** If it does, the thesis can recommend *m* = 32 over exact OT. It
would not remove the DE-overlap cost against the recipe (−0.01239 at *m* = 32 on split 1, Step 20); that cost is
reported as a secondary (§4.2).

The hypothesis comes from the development split, and *m* = 32 is tested because it looked best there. The
confirmation splits are the only protection against that selection (§8).

## 2. Arms and comparators

- **mb32**: the Step 20 arm exactly — the Step 15b recipe (adaLN, Gaussian-source flow matching, 1,024-bit
  fingerprint drug encoding, mlp encoder, cond_drop 0, 100,000 steps, EMA, the same code path and arguments) — with
  `--control_data_path` pointing at `tahoe_train_<split>_control_mb32.h5ad`. Each file holds exactly the rows of the
  original control file, re-ordered within each condition (asserted at writing); its sha256 is pinned and asserted by
  the launcher; the parameter count is asserted equal to 58,178,180. Seeds 0–2 on `drug_split_0` and `drug_split_2`:
  6 runs. The two files were written by job 18772868 at commit f5abcd7. The mean centred cosine between paired cells is
  0.279963 (split 0) and 0.279780 (split 2), each equal to the measure record; on split 1 it was 0.279397 (Step 20
  pre-registration §1).
- **Comparators, already trained and evaluated under the same harness, not retrained:** full OT
  (`s18_<split>_ot_pair_seed_<k>`) and the recipe (`s18r_<split>_recipe_seed_<k>`, bitwise equal to the Step 15b
  arrays). Their Step 18 values are re-derived and asserted to 1e-9 before any Step 21 number is printed (§6).
- **Development split:** not re-run. The Step 20 *m* = 32 results on `drug_split_1` are shown beside the
  confirmation results, copied from the committed Step 20 record, descriptively.

## 3. Evaluation

As in Step 20: the Step 18 harness `leakage_test_fm_step18.py` (which stores the conditioning control cells), through
the Step 18 evaluation path as patched in commit eb23315. 50 NFE Euler, unguided, 200-cell cap, Protocol B with random
vehicle cells of the condition's line and plate. Six evaluations.

## 4. Endpoints

Per split, 97.5 % structure-clustered percentile bootstrap intervals (5,000 replicates), with the bootstrap generator
reset to `RandomState(0)` before every contrast, as in Step 20; per-condition metrics fixed under resampling; arms per
seed, then averaged per condition. Energy, DE-overlap and PDS are defined exactly as in Step 18 §4.1.

### 4.1 Gated, on each confirmation split *s*

**Q1, realism kept (non-inferiority).** D_E(*s*) = energy(mb32) − energy(full OT). Lower energy is closer to the real
cells, so D_E > 0 means mb32 is less realistic. Q1 holds on *s* if the upper bound of D_E's 97.5 % interval is below the
margin δ_*s*.

The margin follows the non-inferiority framework of the FDA guidance (2016, §III.C and §IV.C.1). M1 is the whole effect
of the active comparator, estimated conservatively by its confidence bound closest to no effect; the margin M2 is a
stated fraction of M1. Here the active comparator is full OT, and its effect is Step 18's pre-registered energy gain
over the recipe (Step 18 Q1). M1_*s* is the absolute value of the upper bound of that interval, read by the scorer from
`tahoe_results/step18_analysis.json` (printed in Step 18 §1 as 0.04353 on split 0 and 0.04154 on split 2).
**δ_*s* = 0.25 × M1_*s*** (about 0.0109 and 0.0104): mb32 must keep at least 75 % of full OT's conservatively estimated
realism gain. The guidance notes that choosing M2 as 50 % of M1 "has become usual practice for cardiovascular (CV)
outcome studies". Here 75 % is kept, which is stricter, because the claim under test is that *m* = 32 *keeps* the
realism, not that it keeps some of it. The margin is fixed here, before any Step 21 output, as the CONSORT extension for
non-inferiority trials asks (Piaggio et al. 2012, item 7a).

**Q2, less PDS loss (superiority).** D_P(*s*) = PDS(mb32) − PDS(full OT). Q2 holds on *s* if D_P's 97.5 % interval is
above zero.

### 4.2 Secondary, per confirmation split

DE-overlap, mb32 − full OT. The Step 20 gated contrasts, mb32 − recipe, on energy, DE-overlap and PDS. DPC-50 and MRA
(42 contexts), mb32 − recipe. Control tracking (Step 18 §4.4), mb32 − recipe and mb32 − its permutation null.

### 4.3 Exploratory

The departure from the straight line between the recipe and full OT at *f*(32), for energy, DE-overlap and PDS, on
splits 0 and 2 (defined as in the Step 20 pre-registration §4.4). Does the shape seen on split 1 recur: realism gained
early, PDS cost paid late? Descriptive, with intervals; not gated.

## 5. Verdict rule (coded in the scorer)

E = Q1 holds on both confirmation splits; P = Q2 holds on both.

1. **CONFIRMED** — E and P.
2. **LESS PDS LOSS, REALISM NOT SHOWN** — P, not E.
3. **REALISM KEPT, NO PDS ADVANTAGE SHOWN** — E, not P.
4. **NOT CONFIRMED** — neither E nor P.

The four categories are exclusive. If D_P's interval is below zero on any split, or D_E's lower bound is above δ_*s* on
any split, the verdict carries the clause "with a reversal on <split>". "Not shown" and "includes zero" mean
unresolved, never "equivalent" or "absent". Every verdict is reported with D_E's upper bounds against δ_*s* and D_P's
lower bounds, and the figure shows each interval against its margin (Piaggio et al. 2012, item 17a).

Category 1 requires all four tests to pass (two endpoints on two splits). A test that rejects only when every
component test rejects has level α whenever each component has level α (the intersection–union principle, Berger
1982), so no multiplicity correction is applied to the category claims.

## 6. Splits and discipline

The development split (Step 20) motivated this step. The confirmation runs are on splits 0 and 2 only, and no setting,
margin or endpoint is chosen after any Step 21 output. The scorer refuses to run uncommitted, refuses unless this
document is committed with at least 9 sections, and asserts every Step 18 value it reuses. **All of these
reproductions finish before the first Step 21 number is printed**, so the print-order defect of Step 20 (Step 20
findings §7 item 10) is not repeated. This is the eighth use of the confirmation splits (Steps 15b, 15c, 15d, 15e, 16,
18, 19 and 21; Step 20 read only their training files). Their test data are read in Step 21 only through the six
evaluations and the scorer.

## 7. Expectation (written before the run)

Category **1, CONFIRMED**, held with moderate confidence. The reasoning, which the findings will audit:

- **Q1.** On split 1 the D_E interval was +0.00011 with a half-width of about 0.0046. Step 18's energy intervals on
  splits 0 and 2 were narrower than on split 1 (half-widths about 0.027 and 0.028, against 0.038), so similar or better
  precision is expected. With D_E near zero, the upper bound should fall well inside δ (about 0.0109 and 0.0104).
- **Q2.** If mb32 keeps the split-1 share of full OT's PDS cost (about half), the projected D_P is 0.5 × 0.02820 =
  0.0141 on split 0 and 0.5 × 0.03661 = 0.0183 on split 2. **These are projections from recorded values, not data.**
  With intervals as on split 1 (half-width about 0.007), both would be above zero.
- **Risks.** *m* = 32 was picked after seeing split 1, so its advantage there is probably overstated. Full OT's PDS
  cost is smallest on split 0 (−0.02820), which leaves the least room for an advantage. Two earlier development-split
  effects (Steps 13b and 16) did not replicate. The weakest test is Q2 on split 0.

## 8. Limits

- **Selection.** *m* = 32 was chosen on the development split because it looked best there; only the confirmation
  splits protect against that.
- **One setting**, a pairing drawn once and fixed through training (Step 20 pre-registration §8), and plain Euclidean
  cost on the 200-gene panel.
- **The margin is a judgment**: keep 75 % of the conservatively estimated gain, fixed here before any Step 21 output.
  M1 comes from Step 18's intervals, whose exact bounds depend on draw order (Step 18 findings §9 item 12); on split 1
  the bounds moved by about 0.0005 or less, which would move δ by about 0.0001.
- **Three seeds**, per-split 97.5 % intervals; "not shown" is not "absent".
- **The test is against full OT, not the recipe.** A CONFIRMED verdict would still leave mb32 with the DE-overlap cost
  against the recipe, reported in §4.2.
- **Eighth use of the confirmation splits.**

## 9. Implementation and records

- `train_step21_tahoe.sh` (array of 6: split = (`drug_split_0`, `drug_split_2`)[T // 3], seed = T % 3, *m* = 32;
  guards as Step 20: this pre-registration committed with at least 9 sections, scripts committed and clean, the data
  record and both control files hash-checked against their pins, parameter count 58,178,180, `[step21]` log lines,
  dry run) and `finalize_step21.py` (certification and `provenance/s21tr_<TAG>.json`). Committed, and dry-run on the
  real model and files, before training starts.
- `eval_step21_tahoe.sh` (the Step 20 evaluation path; tags `s21_<split>_mb32_seed_<k>`, provenance `s21ev_*`),
  `score_step21_tahoe.py` and `plot_step21.py`. Committed before the first Step 21 evaluation.
- Checkpoints under `step21/mb32/` (gitignored, sha256-pinned); arrays
  `tahoe_results/vectors_s21_<split>_mb32_seed_<k>.npz` and `_pc.npz`; records `tahoe_results/step21_analysis.json`,
  `step21_long.csv` and `step21_per_condition.csv`; logs under `tahoe_results/step21_logs/`; findings
  `step21_findings.md`.
- GPU: `--partition=condo -q a2s2 --gres=gpu:h100:1`, two tasks in parallel.

## 10. References

- U.S. Food and Drug Administration (2016). *Non-Inferiority Clinical Trials to Establish Effectiveness: Guidance for
  Industry.* November 2016. Definitions of M1 and M2: §III.C; the conservative estimate of M1 and the 50 % practice:
  §IV.C.1.
- Piaggio G, Elbourne DR, Pocock SJ, Evans SJW, Altman DG; CONSORT Group (2012). Reporting of noninferiority and
  equivalence randomized trials: extension of the CONSORT 2010 statement. *JAMA* 308(24):2594–2604.
  doi:10.1001/jama.2012.87802. Item 7a (the margin, specified and justified); item 17a (intervals shown against the
  margin).
- Berger RL (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics* 24(4):295–300. The
  level-α property of the overall test is Theorem 1 of the 1980 technical report of the same title (Florida State
  University; DTIC ADA091607).
