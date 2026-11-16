# Step 29 — findings: do the statements about the cross-validated ridge depend on the fold assignment?

Pre-registration: `PREREG_step29_fold_sensitivity.md` (with its dated paragraphs of 15 and 16 November 2026).
Every number in this file is copied by a script from the two job outputs named in section 7; none is typed.

## 1. What was done

The cross-validated ridge is fitted within the held-out compounds, with five folds grouped by compound (sci-Plex) or
chemical structure (Tahoe-100M). The folds of the paper are one assignment among many that the same rule can give:
compounds with equal numbers of conditions are ordered by a sort whose result depends on the processor. Step 29
refitted the ridge with the folds of the paper and with 200 random assignments (the same rule, ties in random order;
seeds 20261113 onwards) and counted, for every statement of the paper that compares something with this ridge, how
many random assignments give the verdict the paper states. A statement is **robust** on a split if at least 95 % of
the assignments (190 of 200; 48 of 50 for the feedforward network) give that verdict, and robust overall if it is
robust on every split it is made for. No generative model was retrained or re-sampled. No value of the paper is
replaced.

## 2. Guards

All guards of the pre-registration passed before any new value was printed.

- **sci-Plex** (job 19385877, node cn054, AMD EPYC 7742). With the folds of the paper the script gave back the committed
  values: the Step 7 response correlation with a difference of 0.0 on the three splits, the per-condition rows to
  1.1e-16, the records of Steps 17 and 28, and the feedforward network of the Note 1 comparison with a difference of
  0.000000 on every split and seed.
- **Tahoe-100M, folds of the paper** (guard 4; run 19391428 on cn274). The folds were not stored. Of three candidate
  orders of ties, exactly one reproduced the committed per-condition record on each split: numpy's default sort without
  AVX-512 (largest differences 5.6e-17 for gene identity and 3.0e-12 to 6.6e-12 for response correlation; no condition
  differed). The other two differed on every condition. Record: `tahoe_results/step29_tahoe_folds.json`.
- **Tahoe-100M, scorer** (guard 5; job 19391610, node cn066, AMD EPYC 7742). With those folds the ridge reproduced the
  committed per-condition gene identity, response correlation, population distance and discrimination score to
  8.9e-16 or better, and the response-magnitude agreement with its interval exactly. With the folds of the paper and
  intervals from a fresh generator, every item gave the verdict the paper states.

## 3. Result on sci-Plex

Number of random assignments that give the verdict of the paper.

| Item | Statement of the paper | Split 1 | Split 0 | Split 2 | Verdict |
|---|---|---|---|---|---|
| A1 | Gene identity, rewired model minus ridge: positive, interval above zero (the headline; Table S5c) | 200 of 200 | 200 of 200 | 200 of 200 | robust |
| A2 | The same without the salt-twin compound (Supplementary Note 7); Splits 0 and 2 | — | 200 of 200 | 200 of 200 | robust |
| A3 | Gene identity, original model minus ridge: negative | 200 of 200 | 200 of 200 | 200 of 200 | robust |
| A4 | Mean response correlation: every generative model below the ridge | 200 of 200 | 200 of 200 | 199 of 200 | robust |
| A5 | Median response correlation: every generative model below the ridge on Splits 1 and 0; cross-attention and its control above it on Split 2 (caption of Table S6) | 199 of 200 | 200 of 200 | 96 of 200 | **not robust** |
| A6 | Response-magnitude agreement of the ridge: positive; interval above zero on Split 1, includes zero on Splits 0 and 2 (Note 4, "resolved on one split") | 94 of 200 | 189 of 200 | 138 of 200 | **not robust** |
| A7 | Feedforward network minus ridge on response correlation: negative on Split 0 (Results 3, "below it on one"); 50 assignments | — | 45 of 50 | — | **not robust** |
| C4 | Response-magnitude agreement, rewired model minus ridge: interval includes zero (Table S16b) | 197 of 200 | 122 of 200 | 198 of 200 | **not robust** |

Ranges of the quantities behind the items that are not robust (and of the headline, for comparison):

| Quantity | Split | Folds of the paper | Median | 2.5th to 97.5th percentile | Range | Assignments at or below the paper |
|---|---|---|---|---|---|---|
| Gene identity, rewired minus ridge (A1) | 1 | +0.0556 | +0.0582 | +0.0542 to +0.0633 | +0.0528 to +0.0651 | 25 of 200 |
| Gene identity, rewired minus ridge (A1) | 0 | +0.0439 | +0.0444 | +0.0400 to +0.0488 | +0.0376 to +0.0509 | 81 of 200 |
| Gene identity, rewired minus ridge (A1) | 2 | +0.0570 | +0.0558 | +0.0519 to +0.0601 | +0.0507 to +0.0613 | 140 of 200 |
| Median response correlation of the ridge | 1 | +0.1884 | +0.1766 | +0.1485 to +0.2107 | +0.1200 to +0.2206 | 152 of 200 |
| Median response correlation of the ridge | 0 | +0.2473 | +0.2297 | +0.2032 to +0.2567 | +0.1774 to +0.2655 | 167 of 200 |
| Median response correlation of the ridge | 2 | +0.0816 | +0.0973 | +0.0734 to +0.1175 | +0.0663 to +0.1250 | 25 of 200 |
| Cross-attention minus ridge, median (A5) | 1 | -0.0671 | -0.0553 | -0.0894 to -0.0272 | -0.0993 to +0.0013 | 48 of 200 |
| Cross-attention minus ridge, median (A5) | 0 | -0.1587 | -0.1411 | -0.1681 to -0.1146 | -0.1770 to -0.0888 | 33 of 200 |
| Cross-attention minus ridge, median (A5) | 2 | +0.0164 | +0.0007 | -0.0196 to +0.0245 | -0.0271 to +0.0316 | 175 of 200 |
| Its control minus ridge, median (A5) | 1 | -0.0835 | -0.0717 | -0.1057 to -0.0436 | -0.1157 to -0.0151 | 48 of 200 |
| Its control minus ridge, median (A5) | 0 | -0.1672 | -0.1496 | -0.1766 to -0.1231 | -0.1854 to -0.0973 | 33 of 200 |
| Its control minus ridge, median (A5) | 2 | +0.0152 | -0.0006 | -0.0208 to +0.0233 | -0.0283 to +0.0304 | 175 of 200 |
| Response-magnitude agreement of the ridge (A6) | 1 | +0.2778 | +0.2368 | +0.0421 to +0.3441 | -0.0018 to +0.3607 | 152 of 200 |
| Response-magnitude agreement of the ridge (A6) | 0 | +0.1267 | +0.0739 | +0.0164 to +0.1736 | -0.0533 to +0.1973 | 176 of 200 |
| Response-magnitude agreement of the ridge (A6) | 2 | +0.2387 | +0.2242 | +0.0754 to +0.3070 | -0.0996 to +0.3605 | 125 of 200 |
| Rewired minus ridge, response magnitude (C4) | 1 | +0.0563 | +0.0973 | -0.0100 to +0.2920 | -0.0267 to +0.3358 | 48 of 200 |
| Rewired minus ridge, response magnitude (C4) | 0 | +0.1140 | +0.1668 | +0.0670 to +0.2243 | +0.0434 to +0.2940 | 24 of 200 |
| Rewired minus ridge, response magnitude (C4) | 2 | +0.0312 | +0.0457 | -0.0371 to +0.1945 | -0.0906 to +0.3695 | 75 of 200 |
| Network minus ridge (A7; 50 assignments) | 1 | +0.0046 | -0.0138 | -0.0377 to +0.0257 | -0.0428 to +0.0288 | 42 of 50 |
| Network minus ridge (A7; 50 assignments) | 0 | -0.0361 | -0.0191 | -0.0397 to +0.0034 | -0.0446 to +0.0237 | 4 of 50 |
| Network minus ridge (A7; 50 assignments) | 2 | +0.0266 | +0.0116 | -0.0072 to +0.0333 | -0.0113 to +0.0345 | 45 of 50 |

## 4. Result on Tahoe-100M

| Item | Statement of the paper | Split 1 | Split 0 | Split 2 | Verdict |
|---|---|---|---|---|---|
| B1 | Gene identity, compound-blind average minus ridge: positive | 200 of 200 | 200 of 200 | 200 of 200 | robust |
| B2 | Gene identity, original model minus ridge: negative | 200 of 200 | 200 of 200 | 200 of 200 | robust |
| B3 | Discrimination score of the ridge minus 0.5: interval includes zero | 200 of 200 | 200 of 200 | 200 of 200 | robust |
| B4 | Response-magnitude agreement of the ridge: interval includes zero | 197 of 200 | 200 of 200 | 186 of 200 | **not robust** |
| C1 | Gene identity, rewired model minus ridge: interval includes zero (Table S5c) | 101 of 200 | 198 of 200 | 40 of 200 | **not robust** |
| C2 | Population distance, rewired model minus ridge: positive; interval above zero on Splits 1 and 2, includes zero on Split 0 | 200 of 200 | 200 of 200 | 200 of 200 | robust |
| C3 | Discrimination score, rewired model minus ridge: positive, interval above zero | 200 of 200 | 200 of 200 | 200 of 200 | robust |

Ranges:

| Quantity | Split | Folds of the paper | Median | 2.5th to 97.5th percentile | Range | Assignments at or below the paper |
|---|---|---|---|---|---|---|
| Gene identity, rewired minus ridge (C1) | 1 | -0.0132 | -0.0136 | -0.0173 to -0.0092 | -0.0183 to -0.0069 | 123 of 200 |
| Gene identity, rewired minus ridge (C1) | 0 | -0.0050 | -0.0055 | -0.0089 to -0.0015 | -0.0114 to +0.0003 | 124 of 200 |
| Gene identity, rewired minus ridge (C1) | 2 | -0.0093 | -0.0113 | -0.0157 to -0.0079 | -0.0174 to -0.0070 | 176 of 200 |
| Gene identity of the ridge | 1 | +0.4772 | +0.4777 | +0.4733 to +0.4814 | +0.4709 to +0.4824 | 77 of 200 |
| Gene identity of the ridge | 0 | +0.4653 | +0.4658 | +0.4618 to +0.4692 | +0.4600 to +0.4717 | 76 of 200 |
| Gene identity of the ridge | 2 | +0.4678 | +0.4697 | +0.4664 to +0.4742 | +0.4654 to +0.4759 | 24 of 200 |
| Response-magnitude agreement of the ridge (B4) | 1 | -0.0499 | -0.0098 | -0.1275 to +0.0750 | -0.1754 to +0.0940 | 45 of 200 |
| Response-magnitude agreement of the ridge (B4) | 0 | +0.0043 | -0.0188 | -0.0963 to +0.0642 | -0.1183 to +0.0854 | 142 of 200 |
| Response-magnitude agreement of the ridge (B4) | 2 | +0.0944 | +0.0682 | -0.0399 to +0.1655 | -0.0698 to +0.1941 | 141 of 200 |
| Population distance, rewired minus ridge (C2) | 1 | +0.0654 | +0.0671 | +0.0612 to +0.0704 | +0.0593 to +0.0714 | 52 of 200 |
| Population distance, rewired minus ridge (C2) | 0 | +0.0208 | +0.0208 | +0.0157 to +0.0272 | +0.0146 to +0.0292 | 100 of 200 |
| Population distance, rewired minus ridge (C2) | 2 | +0.0558 | +0.0580 | +0.0542 to +0.0613 | +0.0520 to +0.0620 | 22 of 200 |
| Discrimination score, rewired minus ridge (C3) | 1 | +0.2184 | +0.2223 | +0.2117 to +0.2392 | +0.2048 to +0.2431 | 50 of 200 |
| Discrimination score, rewired minus ridge (C3) | 0 | +0.2266 | +0.2242 | +0.2048 to +0.2384 | +0.1959 to +0.2427 | 118 of 200 |
| Discrimination score, rewired minus ridge (C3) | 2 | +0.2283 | +0.2270 | +0.2130 to +0.2390 | +0.2072 to +0.2462 | 116 of 200 |
| Discrimination score of the ridge minus 0.5 (B3) | 1 | +0.0039 | +0.0000 | -0.0169 to +0.0106 | -0.0207 to +0.0175 | 150 of 200 |
| Discrimination score of the ridge minus 0.5 (B3) | 0 | -0.0085 | -0.0061 | -0.0203 to +0.0133 | -0.0246 to +0.0222 | 82 of 200 |
| Discrimination score of the ridge minus 0.5 (B3) | 2 | -0.0141 | -0.0128 | -0.0248 to +0.0012 | -0.0320 to +0.0070 | 84 of 200 |

## 5. Reading, by the rule fixed in advance

**Robust.** The headline of the sci-Plex comparison does not depend on the folds: the rewired model lies above the
cross-validated ridge on gene identity, with an interval above zero, in every assignment on every split, with and
without the salt-twin compound (A1, A2). The original model lies below the ridge on gene identity on both datasets
(A3, B2). On sci-Plex every generative model lies below the ridge on mean response correlation (A4). On Tahoe-100M
the compound-blind average lies above the ridge on gene identity (B1), the discrimination score of the ridge is not
distinguishable from 0.5 (B3), and the comparisons of the rewired model with the ridge on population distance and on
the discrimination score hold in every assignment (C2, C3).

**Not robust.** Six statements depend on the folds.

- **A5** (sci-Plex, Split 2, median response correlation). That cross-attention and its control lie above the ridge
  holds in about half of the assignments. The pre-registration recorded that this outcome was already known from an
  earlier record (the Note 1 record); this item was not blind.
- **A6** (sci-Plex, response-magnitude agreement of the ridge). The estimate is positive, but whether its interval
  excludes zero changes with the folds on every split. On Split 0 the count is one short of the threshold.
- **A7** (sci-Plex, feedforward network against the ridge, Split 0). The count is three short of the threshold. The
  networks of the random assignments were trained with one thread each, which by the dated paragraph of 15 November
  changes a network value by up to about 0.007; the differences counted here are of that size on some assignments.
  The ridge side of this item could also be worked out from the Note 1 record, so it was not fully blind.
- **C4** (sci-Plex, Split 0, response magnitude, rewired model against ridge). In part of the assignments the interval
  excludes zero, with the rewired model above the ridge.
- **B4** (Tahoe-100M, Split 2, response-magnitude agreement of the ridge). In a minority of the assignments the interval
  excludes zero.
- **C1** (Tahoe-100M, gene identity, rewired model against ridge). With the folds of the paper the interval includes
  zero on the three splits, on Splits 1 and 2 with an upper bound within 0.002 of zero. Over the random assignments the
  difference is negative in every assignment on Splits 1 and 2 (the rewired model below the ridge), and the interval
  excludes zero in about half of them on Split 1 and in four fifths on Split 2. On Split 0 the statement holds. The
  statement "not distinguishable" therefore reflects the particular folds; across assignments the ridge scores
  slightly higher than the rewired model on gene identity on two of three splits. The size of the difference is about
  0.01 on a scale where the two values are near 0.47.

**Consequences fixed by the pre-registration (section 6).** The sentences of the paper that carry A5, A6 and A7 are
reworded so that they state the count; the table rows that carry C4, B4 and C1 receive a footnote with the count; a
sentence that states C1 or B4 is reworded likewise. No value of the paper is replaced: every value stays the one
obtained with the folds of the paper. A supplementary table reports all counts and ranges, and the Methods state how
the folds were assigned and that the statements were checked against 200 random assignments.

## 6. Expectations written before the run, and what happened

- A1 to A3 and B1, B2 keep their direction: **as expected** (every assignment).
- A4 holds on Splits 1 and 0, open on Split 2: it **held on the three splits**.
- A5 not robust on Split 2: **as expected** (not blind).
- A6 and C1, whose intervals have a bound close to zero, may not be robust: **both are not robust** (A6 on the three
  splits, C1 on Splits 1 and 2).
- B3 is robust: **as expected** (every assignment).
- A7, B4, C2, C3 and C4: no expectation was written. A7, B4 and C4 are not robust; C2 and C3 are robust.

## 7. Limits

- The count measures agreement with the verdict of the paper under the paper's own rule for an interval. It is not a
  new test, and a count below 95 % does not show that a statement is false; it shows that the statement depends on an
  arbitrary choice.
- The random assignments follow the dealing rule of the paper (largest group first). Assignments that the rule cannot
  produce were not examined.
- Intervals of gene identity, population distance and the discrimination score were recomputed from a fresh generator.
  The committed intervals came from a generator shared between contrasts and differ from the recomputed ones in the
  third or fourth decimal; both are in the records.
- The feedforward network (A7) was examined for 50 assignments, with one training thread.
- The compound-specific comparisons that do not involve the cross-validated ridge are outside this step.

## 8. Records

| What | File | Commit |
|---|---|---|
| Pre-registration | `PREREG_step29_fold_sensitivity.md` | research `cfd6fb6`, `3f4c85c`, `2ab0e33` |
| sci-Plex scorer and job file | `score_step29_sciplex.py`, `run_step29_sciplex.sh` | research `3f4c85c` |
| sci-Plex records | `leakage_results/step29_fold_sensitivity_sciplex.json`, `..._per_assignment.csv` | research `d62d91a` |
| sci-Plex log | `tahoe_results/step28_logs/s29sciplex_19385877.log` | research `d62d91a` |
| Tahoe folds: script, record, log | `diag_step29_tahoe_folds.py`, `tahoe_results/step29_tahoe_folds.json`, `tahoe_results/step28_logs/s29tfolds_19391428.log` | research `4796dc2`, `00f61e7` |
| Tahoe scorer and job file | `score_step29_tahoe.py`, `run_step29_tahoe.sh` | research `2ab0e33` |
| Tahoe records | `tahoe_results/step29_fold_sensitivity_tahoe.json`, `..._per_assignment.csv` | research `fe362ae` |
| Tahoe log | `tahoe_results/step28_logs/s29tahoe_19391610.log` | research `fe362ae` |
| All of the above in the code package | folders `prereg`, `03_evaluation`, `04_scoring`, `07_records` | package `27707b2` |

The tables of sections 3 and 4 were extracted from the two logs (`s29sciplex_19385877.log`, `s29tahoe_19391610.log`).
