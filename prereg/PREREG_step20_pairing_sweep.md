# Pre-registration — Step 20: how strongly should training pairs be matched? A minibatch-OT sweep between the recipe and Step 18

Date: 2026-10-12. Written before any Step 20 checkpoint exists. Inputs: the Step 15a data (`tahoe_results/step15_data_record.json`);
the Step 18 records (`step18_findings.md`, `tahoe_results/step18_analysis.json`, the `s18_*` and `s18r_*` arrays); the
minibatch pairing record written by `check_minibatch_pairing_tahoe.py` (script commit eff074f; record
`tahoe_results/minibatch_pairing_check.json`, commit f85ea15) and the three development control files it wrote, each
sha256-pinned in `tahoe_results/minibatch_pairing_written_drug_split_1_m<m>.json`.

## 1. Question

Step 18 trained the recipe on treated–control pairs matched by exact optimal transport within each condition. On all
three splits the generated populations came closer to the real ones (energy, `ot_pair` − recipe: −0.07706 / −0.06825 /
−0.06802), at a cost on the drug-specific endpoints (DE-overlap −0.01457 / −0.01241 / −0.01213; PDS −0.03783 /
−0.02820 / −0.03661), and each generated cell followed the control cell it was conditioned on (tracking +0.30 against
+0.015 for the recipe). Step 20 asks whether a **weaker** matching keeps the fidelity gain while losing less of the drug
signal — that is, whether the gain and the cost can be separated along the strength of the pairing.

**The dial.** Minibatch optimal transport: inside each condition the treated cells and the control cells are split at
random into groups of about *m*, and the exact matching is made only inside each group. *m* = 1 is a random pairing
(the recipe), *m* ≥ the condition size is exact OT (Step 18). Sources, each checked against the original text: Fatras
et al. 2021 (arXiv:2101.01792, §3.2.5: minibatch OT interpolates "between OT, when m=n and averaged pairwise distance,
when m=1"; §3.3: the plan "densifies when m decreases, which is a similar effect to entropic regularization"); Fatras
et al. 2020 (AISTATS, PMLR 108:2131–2141: minibatch OT is "an implicit regularization of the original problem");
Pooladian et al. 2023 (ICML, PMLR 202: transport cost falls with batch size, Theorem D.8, and approaches the OT cost,
Theorem 4.2); Tong et al. 2024 (TMLR, §3.2.3: a batch covering the support recovers exact OT). Exact matching inside each
group: `scipy.optimize.linear_sum_assignment` (modified Jonker–Volgenant; Crouse 2016, IEEE TAES 52(4):1679–1696), plain
Euclidean cost on the 200-gene panel, as in Step 18.

**The three settings**, chosen before this document by the rule fixed in `check_minibatch_pairing_tahoe.py` (on
`drug_split_1`, the candidate *m* closest to 25 / 50 / 75 % of the way from the recipe to full OT on the centred cosine
between a treated cell and its paired control). Measured on the development split (training conditions: 11,464;
cells per condition 43 / 150 / 150):

| pairing | pair distance | cosine | fraction of the way, cosine (f) | fraction, distance |
|---|---|---|---|---|
| recipe | 11.792285 | 0.000355 | 0 | 0 |
| *m* = 4 | 11.070632 | 0.122683 | 0.333 | 0.310 |
| *m* = 8 | 10.703655 | 0.182157 | 0.495 | 0.467 |
| *m* = 32 | 10.074031 | 0.279397 | 0.759 | 0.738 |
| full OT (Step 18) | 9.462471 | 0.367880 | 1 | 1 |

The two ends reproduce the Step 18 pairing record to 1e-9, and at the chosen settings the fractions on the three
splits agree to within 0.001 (`minibatch_pairing_check.json`).

## 2. Arms and comparators

- **mb4, mb8, mb32**: the Step 18 arm exactly — the Step 15b recipe (adaLN, Gaussian-source flow matching, fingerprint
  drug encoding of 1,024 bits, mlp encoder, cond_drop 0, 100,000 steps, EMA, the same code path and arguments) — with
  `--control_data_path` pointing at `tahoe_train_<split>_control_mb<m>.h5ad`. Each file holds exactly the rows of the
  original control file, re-ordered within each condition (asserted at writing); its sha256 is pinned and asserted by
  the launcher; parameter count asserted equal to 58,178,180. Development: seeds 0–2 on `drug_split_1`, 9 runs.
- **Comparators, already trained and evaluated under the same harness:** the recipe (`s18r_<split>_recipe_seed_<k>`,
  bitwise equal to the Step 15b arrays) and `ot_pair` (`s18_<split>_ot_pair_seed_<k>`). They are not retrained; their
  Step 18 values are re-derived and asserted to 1e-9 before any Step 20 number is printed.
- **Confirmation** (§5): only the setting selected on the development split, seeds 0–2 on `drug_split_0` and
  `drug_split_2`, 6 runs; its control files are written by the same script after selection.

## 3. Evaluation

The Step 18 harness `leakage_test_fm_step18.py` (stores the conditioning control cells `ctrl`), through the Step 18
evaluation launcher as patched in commit eb23315 (per-condition reads hoisted out of the loops). 50 NFE Euler,
unguided, 200-cell cap, Protocol B with random vehicle cells of the condition's line and plate — as every Tahoe
evaluation. Nine evaluations on the development split; six more if confirmation runs.

## 4. Endpoints

Per split, 97.5 % structure-clustered percentile bootstrap intervals (5,000 replicates, `RandomState(0)`), per-condition
metrics fixed under resampling, arms per seed then averaged per condition — as Step 18.

### 4.1 Gated, per setting: Q1 energy, Q2 DE-overlap, Q3 PDS (setting − recipe), defined exactly as Step 18 §4.1.

### 4.2 Qualification and selection (development split only)

A setting **qualifies** if, on `drug_split_1`, Q1 is below zero and neither Q2 nor Q3 is below zero. If one or more
settings qualify, the qualifying setting with the **largest m** (the strongest matching that still shows no resolved
cost) is selected and confirmed (§5). If none qualifies, no confirmation run is made. The rule is coded in the scorer
before the development run.

### 4.3 Secondary, per setting: the setting − `ot_pair` on energy, DE-overlap and PDS (how much of Step 18's gain and
cost it keeps); DPC-50 and MRA (42 contexts) against the recipe; control tracking (Step 18 §4.4), which should rise with
*m* between the recipe (+0.015) and `ot_pair` (+0.30) if the dial acts on the model as it acts on the pairs.

### 4.4 Exploratory: departure from a straight line. For each setting *s* and endpoint X (energy, DE-overlap, PDS):
excess(*s*) = [X(*s*) − X(recipe)] − f(*s*) · [X(`ot_pair`) − X(recipe)], with f(*s*) the cosine fraction of §1 (a fixed
constant from the pairing record). A gain larger than the line, or a cost smaller than the line, is what a separable
trade-off would look like. Descriptive, with intervals; not gated.

## 5. Verdict rule (coded in the scorer)

**Development stage** (`drug_split_1` alone):
- **0 NO QUALIFYING SETTING** — no setting qualifies (§4.2). Step 20 ends with the development curve; no confirmation.
- **SELECTED m = X** — the setting chosen by §4.2 goes to confirmation.

**Final stage** (selected setting, three splits; precedence 1 → 2 → 5 → 3 → 4, as Step 18 §5):
1. **FIDELITY GAIN** — Q1 below zero on all three splits; Q2 and Q3 not below zero on any split.
2. **FIDELITY GAIN AT A COST** — Q1 below zero on all three splits; Q2 or Q3 below zero on at least one split.
5. **UNRESOLVED** — Q1 includes zero on all three splits.
3. **REVERSED** — Q1 above zero on any split.
4. **PARTIAL** — otherwise.

"Not below zero" and "includes zero" are "unresolved", never "equivalent"; a category-1 result is reported with the
lower bounds of Q2 and Q3, so the size of a cost that cannot be excluded is visible.

## 6. Splits and discipline

Development first: the nine `drug_split_1` runs are evaluated and scored alone (`_development` records) before any
confirmation control file is written. Nothing in the scorer changes between the two stages; any correction before the
confirmation runs is dated into an existing section, never a new one. The scorer refuses to run uncommitted, refuses
unless this document is committed with at least 9 sections, and asserts every Step 18 value it reuses. Eighth use of
the Tahoe splits (Steps 15b, 15c, 15d, 15e, 16, 18, 19, 20); the selection among the three settings is made on the
development split only.

## 7. Expectation (written before the run)

Category **0, NO QUALIFYING SETTING**. Reasoning, auditable against the outcome: if the gain and the costs both scale
with the pairing fraction f — the simplest reading of Step 18, where both arise from the generated cell following its
control cell — then the projected development contrasts are Step 18's multiplied by f. **These are projections from
recorded values, not data:** energy −0.026 / −0.038 / −0.058, DE-overlap −0.0049 / −0.0072 / −0.0111, PDS −0.0126 /
−0.0187 / −0.0287 at *m* = 4 / 8 / 32 (f = 0.333 / 0.495 / 0.759 times −0.07706, −0.01457, −0.03783). Against Step 18's
development half-widths (energy 0.038, DE-overlap 0.006, PDS 0.024), the DE-overlap cost would resolve wherever the
energy gain does, so no setting would qualify. A setting qualifies only if the gain is front-loaded (larger than the
line at small f) or the cost back-loaded. Tracking: I expect it to rise roughly with f — projected as
recipe + f × (`ot_pair` − recipe) from the split-1 values +0.0152 and +0.3029, about +0.11 / +0.16 / +0.23.

If category 0 is the outcome, Step 20 still answers the question it was run for: along the pairing axis the fidelity
gain cannot be had without the drug-signal cost, and optimising the pairing strength does not remove it.

## 8. Limits

The pairing is drawn once and fixed through training (as in Step 18), whereas minibatch OT in flow-matching training
re-draws groups every step; the low end of the dial is coarse (*m* = 2 gives f = 0.165, *m* = 4 gives 0.333); f is
measured on the cosine, one of two pairing statistics (the distance fractions are 0.02–0.03 lower); the cost is plain
Euclidean on the 200-gene panel, not the squared Euclidean of the flow-matching sources; selection uses the
development split only, so the confirmed setting carries no winner's-curse correction beyond the two fresh splits;
the qualification rule relies on "not below zero", which with three seeds and these intervals cannot exclude a cost
of about the DE-overlap half-width (0.006); three splits used eight times.

## 9. Implementation and records

- `train_step20_tahoe.sh` (array of 9 on the development split: *m* = 4, 8, 32 × seeds 0–2; guards as Step 18: this
  pre-registration ≥ 9 sections, scripts committed and clean, data record and minibatch control files hash-checked
  against their pins, parameter count 58,178,180, `[step20]` log lines, dry run); `eval_step20_tahoe.sh` (the patched
  Step 18 evaluation path); `finalize_step20.py` (certification and `provenance/s20tr_<TAG>.json`);
  `score_step20_tahoe.py`; `plot_step20.py`. All committed before the first run.
- Checkpoints under `step20/mb<m>/` (gitignored, sha256-pinned); arrays `tahoe_results/vectors_s20_<split>_mb<m>_seed_<k>.npz`
  and `_pc.npz`, provenance `s20ev_*`; records `tahoe_results/step20_analysis[_development].json`, `step20_long[_development].csv`,
  `step20_per_condition[_development].csv`; logs under `tahoe_results/step20_logs/`; findings `step20_findings.md`.
- GPU: `--partition=condo -q a2s2 --gres=gpu:h100:1`, two tasks in parallel.
- *Dated 2026-10-12, before any Step 20 run.* The order in the first bullet is changed for the training scripts
  only, because training produces no endpoint: `train_step20_tahoe.sh` and `finalize_step20.py` are committed first
  (commit 5330817; a dry run on the real model and files passed) and training may start then. The evaluation
  launcher, the scorer and the plot script are committed before the first Step 20 evaluation run, so every analysis
  script still precedes every Step 20 array and every result. Nothing else in this document changes.
