# Step 25 findings — fingerprint-swap confirmation on three held-out splits

Date: 2026-11-04
Pre-registration: `PREREG_step25_swap_confirmation.md` (md5 `77feda36ec06752b09bdb73d73748283`), with the dated §7.7 amendment of 2026-11-04 (scoring-gate incident, see §6 below).
Scorer: `score_step25_tahoe.py` (md5 `4bd60023513feeaef285724109dda52d`).
Records: `tahoe_results/step25_swap.json`, `tahoe_results/step25_swap_per_condition.csv`, `tahoe_results/step25_swap_summary.csv` (research-repo commit `2857a16`, branch `step15-tahoe`).
Logs: `tahoe_results/step25_logs/s25ev_19024369_{0..4}.out` (reproduction runs), `s25ev_19029254_{5..34}.out` (swap runs), `s25score_19044477.log` (failed first scoring run, kept), `s25score_19061929.log` (scoring run that produced the records), `verify_step15c_intervals.log`.

Every number below is copied from `step25_swap_summary.csv` or from `s25score_19061929.log`. Values are rounded to three or four decimals; the CSV holds full precision.

---

## 1. Question and design (from the pre-registration)

Step 24 found, on the development split (`drug_split_1`), that the recipe model discriminates held-out conditions about as well when it is given a **wrong** compound fingerprint (a derangement across conditions, `fpshuf`) or **no** fingerprint (`fpzero`) as when it is given the correct one. Step 25 asked three pre-registered questions:

- **A.** Does the swap null replicate on the two confirmation splits (`drug_split_0`, `drug_split_2`) for the recipe model?
- **B.** Does the base (unmodified Squidiff) model show the same pattern?
- **C.** Is the recipe-over-base PDS advantage compound-specific? That is, does the gap between recipe and base shrink when both models are given wrong compounds?

Design: for each split and each model (recipe, base), three seeds each of the own-compound arm (the committed Step 15 evaluations), the `fpshuf` arm and the `fpzero` arm. Metrics per condition: PDS (chance 0.5), DE-overlap, DPC-50, energy distance, MRA. Contrasts are differences of per-condition PDS, with structure-clustered percentile bootstrap intervals (5,000 reps, 97.5 %). Decision rules were fixed in advance (prereg §5–§6): an interval that includes zero reads "no resolved difference"; one above zero reads "above zero".

Derangement permutations: split 0 `tahoe_results/step25/permutation_drug_split_0.json` (sha `27299dbf01e9`), split 2 `permutation_drug_split_2.json` (sha `d1a8b9f25921`); split 1 reuses the Step 24 permutation (sha `69ff344f2ce3`).

Conditions scored: 2,899 (split 1), 3,115 (split 0), 2,853 (split 2).

---

## 2. Reproduction gates (all passed)

Before any swap run was scored, the scorer required:

- the five reproduction evaluations (job 19024369: recipe and base, seed 0, each split that lacked a committed record) to match the committed Step 15 predictions **bitwise** — five of five passed;
- the committed per-condition records of Steps 15b, 15c, 24 to be reproduced to 1e-9;
- the Step 15c bootstrap intervals (recipe and base, each split) to be reproduced to 5e-6 — passed on all three splits after the §7.7 amendment (see §6);
- the Step 16 development-split MRA seeds to be reproduced: 0.3112 / 0.3237 / 0.3265 — passed.

All 30 swap evaluations (job 19029254) completed without `FATAL`; each log reports its fingerprint substitution (derangement or zero vector) and the number of substituted conditions.

---

## 3. Pre-registered contrasts

### 3.1 Question A — recipe model, own compound vs wrong compound

| split | Q1: own − shuf | verdict | Q2: shuf − 0.5 | verdict |
|---|---|---|---|---|
| drug_split_1 (dev) | +0.0063 [−0.0153, +0.0288] | includes zero | +0.2160 [+0.1862, +0.2455] | above zero |
| drug_split_0 | +0.0169 [−0.0044, +0.0406] | includes zero | +0.2012 [+0.1572, +0.2386] | above zero |
| drug_split_2 | +0.0235 [−0.0029, +0.0527] | includes zero | +0.1907 [+0.1557, +0.2248] | above zero |

**Verdict A: 1 — NULL REPLICATED.** On every split the recipe model with a wrong compound still discriminates conditions far above chance (Q2), and its advantage from having the right compound (Q1) is not resolved from zero.

### 3.2 Question B — base model, own compound vs wrong compound

| split | Q1: own − shuf | verdict | Q2: shuf − 0.5 | verdict |
|---|---|---|---|---|
| drug_split_1 (dev) | −0.0011 [−0.0090, +0.0068] | includes zero | +0.1426 [+0.1035, +0.1804] | above zero |
| drug_split_0 | +0.0084 [−0.0035, +0.0226] | includes zero | +0.1360 [+0.0959, +0.1769] | above zero |
| drug_split_2 | +0.0107 [−0.0010, +0.0248] | includes zero | +0.1441 [+0.0985, +0.1873] | above zero |

**Verdict B: 1 — NULL REPLICATED.** The base model shows the same pattern. This is not a property introduced by the recipe.

### 3.3 Question C — is the recipe-over-base gap compound-specific?

| split | gap with own compounds | gap with wrong compounds | Q3: gap(own) − gap(shuf) | verdict |
|---|---|---|---|---|
| drug_split_1 (dev) | +0.0808 [+0.0362, +0.1239] | +0.0734 [+0.0282, +0.1143] | +0.0075 [−0.0116, +0.0271] | includes zero |
| drug_split_0 | +0.0737 [+0.0176, +0.1237] | +0.0652 [+0.0122, +0.1133] | +0.0085 [−0.0077, +0.0258] | includes zero |
| drug_split_2 | +0.0593 [+0.0175, +0.1029] | +0.0465 [+0.0014, +0.0935] | +0.0128 [−0.0088, +0.0354] | includes zero |

**Verdict C: GAP NOT COMPOUND-SPECIFIC.** The recipe beats the base model by about 0.05–0.08 PDS on every split, and it does so to the same extent whether or not either model is given the correct compound. The gap is therefore a gain in *condition* discrimination (cell line, dose, plate context), not in *compound* discrimination.

### 3.4 Reference contrasts (not decision rules; recorded for completeness)

| split | recipe: zero − 0.5 | recipe: shuf − zero | recipe: own − zero | base: zero − 0.5 | base: shuf − zero | base: own − zero |
|---|---|---|---|---|---|---|
| drug_split_1 | +0.2360 [+0.2076, +0.2653] | −0.0200 [−0.0399, −0.0019] below zero | −0.0136 [−0.0328, +0.0041] | +0.1431 [+0.1036, +0.1822] | −0.0004 [−0.0065, +0.0058] | −0.0016 [−0.0094, +0.0061] |
| drug_split_0 | +0.2180 [+0.1761, +0.2542] | −0.0168 [−0.0310, −0.0031] below zero | +0.0001 [−0.0198, +0.0213] | +0.1366 [+0.0991, +0.1756] | −0.0006 [−0.0098, +0.0086] | +0.0078 [−0.0018, +0.0212] |
| drug_split_2 | +0.2144 [+0.1801, +0.2472] | −0.0238 [−0.0456, −0.0018] below zero | −0.0002 [−0.0196, +0.0200] | +0.1460 [+0.1004, +0.1895] | −0.0018 [−0.0106, +0.0063] | +0.0089 [−0.0000, +0.0186] |

One reading worth noting: for the recipe model, a **wrong** fingerprint scores slightly *below* **no** fingerprint on all three splits (shuf − zero: −0.020, −0.017, −0.024, each interval below zero), while the correct fingerprint scores no better than no fingerprint (own − zero includes zero on every split). A wrong compound is a small, consistent distraction; the right compound adds nothing resolvable over an empty vector. For the base model, own, wrong and empty are indistinguishable.

---

## 4. Absolute metrics per arm (mean over three seeds, mean over conditions)

PDS chance = 0.5. Energy distance: lower is better. MRA: higher is better.

### drug_split_1 (dev, n = 2,899)

| arm | PDS | DE-overlap | DPC-50 | energy | MRA |
|---|---|---|---|---|---|
| recipe own | 0.7223 | 0.4640 | 0.5197 | 0.2250 | 0.3205 |
| recipe fpshuf | 0.7160 | 0.4623 | 0.5229 | 0.2163 | 0.3603 |
| recipe fpzero | 0.7360 | 0.4564 | 0.5288 | 0.1641 | 0.3553 |
| base own | 0.6415 | 0.2683 | 0.2269 | 2.2681 | −0.0755 |
| base fpshuf | 0.6426 | 0.2680 | 0.2269 | 2.2610 | −0.0745 |
| base fpzero | 0.6431 | 0.2672 | 0.2193 | 2.3116 | −0.1356 |

### drug_split_0 (n = 3,115)

| arm | PDS | DE-overlap | DPC-50 | energy | MRA |
|---|---|---|---|---|---|
| recipe own | 0.7181 | 0.4603 | 0.5005 | 0.2438 | 0.2864 |
| recipe fpshuf | 0.7012 | 0.4545 | 0.4850 | 0.2846 | 0.2138 |
| recipe fpzero | 0.7180 | 0.4534 | 0.4868 | 0.2330 | 0.2491 |
| base own | 0.6444 | 0.2850 | 0.2490 | 2.1353 | −0.0576 |
| base fpshuf | 0.6360 | 0.2849 | 0.2409 | 2.1454 | −0.0538 |
| base fpzero | 0.6366 | 0.2831 | 0.2373 | 2.1447 | −0.1138 |

### drug_split_2 (n = 2,853)

| arm | PDS | DE-overlap | DPC-50 | energy | MRA |
|---|---|---|---|---|---|
| recipe own | 0.7142 | 0.4585 | 0.5231 | 0.2181 | 0.3117 |
| recipe fpshuf | 0.6907 | 0.4560 | 0.5102 | 0.2149 | 0.3283 |
| recipe fpzero | 0.7144 | 0.4525 | 0.5226 | 0.1728 | 0.3261 |
| base own | 0.6549 | 0.2908 | 0.2705 | 2.0210 | −0.1069 |
| base fpshuf | 0.6441 | 0.2897 | 0.2629 | 2.0151 | −0.0716 |
| base fpzero | 0.6460 | 0.2881 | 0.2584 | 2.0385 | −0.1169 |

Per-seed MRA (from the scoring log): recipe own 0.3112 / 0.3237 / 0.3265 (split 1), 0.2820 / 0.2917 / 0.2855 (split 0), 0.3168 / 0.3103 / 0.3081 (split 2); base own −0.0575 / −0.0739 / −0.0950, −0.0469 / −0.0364 / −0.0896, −0.1213 / −0.0895 / −0.1099.

Reading of the absolute table: across all five metrics the recipe's three arms sit close together, and the base's three arms sit close together; the large differences are between models, not between fingerprint arms. The recipe's advantage over the base model on DE-overlap (~0.46 vs ~0.27–0.29), DPC-50 (~0.49–0.53 vs ~0.22–0.27), energy distance (~0.16–0.28 vs ~2.0–2.3) and MRA (~0.21–0.36 vs negative) survives fingerprint shuffling and zeroing intact.

---

## 5. Audit against the pre-registered expectations

| prereg expectation (§6) | outcome | held? |
|---|---|---|
| A: Q1 includes zero and Q2 above zero on both confirmation splits → "null replicated" | Q1 includes zero on splits 0, 2 (and 1); Q2 above zero on all three | yes |
| B: same pattern for the base model | Q1 includes zero on all three; Q2 above zero on all three | yes |
| C: Q3 includes zero on all splits → "gap not compound-specific" | Q3 includes zero on all three | yes |

No expectation was revised after seeing data. No measure or verdict rule changed.

**Nuance that must travel with the verdicts.** The recipe's Q1 point estimates are positive on every split and grow from the development split to the confirmation splits (+0.006, +0.017, +0.024), and two lower bounds sit just under zero (−0.0044 on split 0, −0.0029 on split 2; base split 2: −0.0010). The correct statement is: *no resolved own-compound advantage on any split; an advantage of up to about +0.05 PDS is not excluded by these intervals.* "Zero effect" is not what the data say. Under Protocol B scoring, the shared control estimate adds noise that the swap cannot remove; Step 26 (G3) re-asks the own-vs-wrong question under split-control scoring, where that noise is gone, and is the right place to tighten this.

---

## 6. Scoring-gate incident (prereg §7.7 amendment, 2026-11-04)

The first scoring run (job 19044477, log kept) stopped at the reproduction gate for the Step 15c base interval on split 1: the point estimate matched exactly, but the interval bounds differed by about 0.001 from the committed record.

Cause: Step 15c computed every one of its bootstrap intervals from a single module-level `RandomState(0)` (`score_step15_tahoe.py`, line 66), advancing it across contrasts and splits in the order of that run. Only the first draw of that run (recipe, split 1) is reproducible by a fresh `RandomState(0)`; every later interval depends on how many draws preceded it. This is a draw-order dependence of the committed record, not a change in the data or in the estimator.

Proof: `verify_step15c_intervals.py` replays the Step 15c draw sequence against the committed `tahoe_results/step15c_pds_per_condition.csv` and reproduces all six intervals (recipe and base, three splits) — **ALL PASS**; the drift seen with a fresh generator was 0.0004–0.0022 for the five non-first intervals (log `verify_step15c_intervals.log`).

Fix: `replay_step15c()` was added to the scorer and the gate kept exact (tolerance 5e-6). Step 25's own contrasts are unaffected: they use `s23.Boot`, which seeds a fresh `RandomState(0)` per contrast. The research-repo commit `2547893` carries the fix, the verify script, the failed log and the dated amendment. The rescored run (job 19061929, elapsed 00:30:11) passed every gate.

Lesson recorded for the manuscript methods: every bootstrap interval must be seeded per contrast, never from a shared generator, or the record cannot be reproduced piecewise.

---

## 7. What Step 25 settles, and what it does not

Settled:
1. The compound-fingerprint swap null is not a development-split artefact. It holds on all three held-out drug splits, for both the recipe and the base model.
2. The recipe's PDS, DE-overlap, DPC-50, energy and MRA gains over the base model do not depend on the model receiving the correct compound. They are gains in condition discrimination.
3. Under Protocol B scoring, the recipe model with a wrong compound is slightly *worse* than with no compound (three splits, intervals below zero); with the right compound it is not resolvably better than with none.

Not settled here:
1. Whether a small own-compound advantage (≤ ~0.05 PDS) exists — the intervals do not exclude it. Step 26 G3 addresses this under split-control scoring.
2. *Where* the condition discrimination comes from (shared control estimate, conditioning information, or masked compound signal) — this is exactly Step 26's S1/S2/S3 decomposition.
3. Anything about the fingerprint design itself (Morgan vs other featurisations) — all arms here use the committed fingerprint; swapping and zeroing test whether the model *uses* it, not whether a different one would be used.

---

## 8. Next step

Step 26 (`run_step26_score.sh`, CPU) is staged and refuses to run until this document and the Step 25 records are committed and packaged. Its scorer reproduces the Step 25 records to 1e-9 as its own entry gate.
