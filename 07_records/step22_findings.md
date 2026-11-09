# Step 22 findings — RNA fingerprinting (Grabski et al. 2026) applied to the generated cells: verdict 3 NOT CONFIRMED

Pre-registration: `PREREG_step22_fingerprint_identifiability.md` (committed a75dfca; dated notes added before the
runs they concern: §2 f177e50, §4 amendment 042ab8b/afb9ff2, §3.1 note ae02490, §9 note 7ccadc4). Record:
`tahoe_results/step22_analysis.json` (final, commit 7ccadc4, 137 s), `step22_long.csv`, `step22_per_condition.csv`;
development record `step22_development_*`. Every number below is copied from the final record. Written 2026-10-14.

## 1. Verdict

**3 NOT CONFIRMED** on all three splits. The gate passed on every split; neither primary held on any split.

| split | K kept of 75 | covered conditions | gate G1 | gate G2 | Q1 (group) | Q2 (cell) |
|---|---|---|---|---|---|---|
| 1 (development) | 63 | 2,477 of 2,899 | above zero | above zero | includes zero | includes zero |
| 0 | 66 | 2,786 of 3,115 | above zero | above zero | includes zero | includes zero |
| 2 | 65 | 2,539 of 2,853 | above zero | above zero | includes zero | includes zero |

In words: the authors' package, learning one fingerprint per held-out compound from half of the real cells, does
recognise the compound in the other half of the real cells more often than chance and more often than in the
compound-blind reference — so the test is informative — but it does not recognise the compound in the recipe's
generated populations or cells any more often than in a vehicle cell shifted by the line's mean response. No
reversal: no Q1 or Q2 interval lies below zero.

## 2. Measurements

Chance-corrected hit = hit − chance per query, averaged per condition (arms: per seed, then per condition), over the
covered conditions (prereg §4 with the amendment of 2026-10-14). Hit: the true structure is in the top credible set
with sign +. Chance: members of the top set with sign + divided by K. Unassigned: fraction of queries with no credible
set.

### 2.1 Chance-corrected hit per set (hit, chance, unassigned fraction in brackets)

| set | level | split 1 | split 0 | split 2 |
|---|---|---|---|---|
| real cells, other half | group | 0.0685 (hit 0.0820, chance 0.0134, unassigned 0.46) | 0.1041 (hit 0.1195, chance 0.0154, unassigned 0.43) | 0.0620 (hit 0.0748, chance 0.0128, unassigned 0.48) |
| recipe | group | 0.0034 (hit 0.0147, chance 0.0112, unassigned 0.59) | 0.0132 (hit 0.0276, chance 0.0144, unassigned 0.54) | 0.0084 (hit 0.0205, chance 0.0121, unassigned 0.57) |
| m = 32 | group | 0.0129 (hit 0.0242, chance 0.0113, unassigned 0.59) | 0.0185 (hit 0.0319, chance 0.0134, unassigned 0.57) | 0.0101 (hit 0.0211, chance 0.0110, unassigned 0.59) |
| full OT | group | 0.0148 (hit 0.0264, chance 0.0116, unassigned 0.59) | 0.0211 (hit 0.0345, chance 0.0134, unassigned 0.57) | 0.0127 (hit 0.0238, chance 0.0111, unassigned 0.61) |
| ridge-S7 | group | 0.0095 (hit 0.0206, chance 0.0111, unassigned 0.53) | 0.0085 (hit 0.0190, chance 0.0106, unassigned 0.54) | 0.0095 (hit 0.0205, chance 0.0109, unassigned 0.59) |
| ridge-train | group | 0.0173 (hit 0.0283, chance 0.0110, unassigned 0.54) | 0.0195 (hit 0.0287, chance 0.0093, unassigned 0.57) | 0.0163 (hit 0.0272, chance 0.0108, unassigned 0.59) |
| compound-blind mean | group | 0.0087 (hit 0.0202, chance 0.0115, unassigned 0.53) | 0.0077 (hit 0.0187, chance 0.0110, unassigned 0.55) | 0.0092 (hit 0.0201, chance 0.0109, unassigned 0.58) |
| vehicle cells | group | 0.0060 (hit 0.0129, chance 0.0069, unassigned 0.59) | 0.0032 (hit 0.0086, chance 0.0054, unassigned 0.63) | 0.0064 (hit 0.0138, chance 0.0074, unassigned 0.65) |
| real cells, other half | cell | 0.0094 (hit 0.0158, chance 0.0064, unassigned 0.77) | 0.0114 (hit 0.0184, chance 0.0070, unassigned 0.80) | 0.0066 (hit 0.0129, chance 0.0064, unassigned 0.80) |
| recipe | cell | 0.0012 (hit 0.0070, chance 0.0058, unassigned 0.80) | 0.0022 (hit 0.0082, chance 0.0061, unassigned 0.84) | 0.0015 (hit 0.0071, chance 0.0056, unassigned 0.82) |
| m = 32 | cell | 0.0021 (hit 0.0081, chance 0.0060, unassigned 0.80) | 0.0033 (hit 0.0095, chance 0.0062, unassigned 0.83) | 0.0026 (hit 0.0084, chance 0.0058, unassigned 0.82) |
| full OT | cell | 0.0020 (hit 0.0081, chance 0.0061, unassigned 0.79) | 0.0030 (hit 0.0092, chance 0.0062, unassigned 0.83) | 0.0027 (hit 0.0086, chance 0.0059, unassigned 0.82) |
| compound-blind mean | cell | 0.0017 (hit 0.0075, chance 0.0059, unassigned 0.78) | 0.0024 (hit 0.0087, chance 0.0063, unassigned 0.82) | 0.0017 (hit 0.0070, chance 0.0054, unassigned 0.81) |
| vehicle cells | cell | 0.0016 (hit 0.0071, chance 0.0055, unassigned 0.78) | 0.0025 (hit 0.0081, chance 0.0056, unassigned 0.82) | 0.0014 (hit 0.0065, chance 0.0050, unassigned 0.81) |

Per seed, group level, chance-corrected hit: recipe 0.0047 / 0.0026 / 0.0030 (split 1), 0.0165 / 0.0097 / 0.0135
(split 0), 0.0062 / 0.0092 / 0.0099 (split 2); full OT 0.0146 / 0.0131 / 0.0167, 0.0210 / 0.0216 / 0.0206,
0.0135 / 0.0124 / 0.0120; m = 32 0.0147 / 0.0107 / 0.0133, 0.0167 / 0.0200 / 0.0188, 0.0094 / 0.0106 / 0.0105.
Mean size of the top credible set: 0.8–1.2 at group level, 0.4–0.6 at cell level (unassigned counted as 0).

### 2.2 Contrasts (97.5 % structure-clustered intervals; chance-corrected hit unless marked raw)

| contrast | split 1 | split 0 | split 2 |
|---|---|---|---|
| G1 real cells above 0 (group) | +0.0685 [+0.0474, +0.0951] above zero | +0.1041 [+0.0620, +0.1618] above zero | +0.0620 [+0.0404, +0.0876] above zero |
| G2 real cells − compound-blind mean (group) | +0.0598 [+0.0387, +0.0862] above zero | +0.0965 [+0.0540, +0.1545] above zero | +0.0528 [+0.0287, +0.0802] above zero |
| **Q1 recipe − compound-blind mean (group)** | −0.0052 [−0.0135, +0.0032] includes zero | +0.0056 [−0.0106, +0.0354] includes zero | −0.0008 [−0.0132, +0.0192] includes zero |
| **Q2 recipe − compound-blind mean (cell)** | −0.0005 [−0.0022, +0.0009] includes zero | −0.0003 [−0.0023, +0.0017] includes zero | −0.0002 [−0.0024, +0.0016] includes zero |
| full OT − recipe (group) | +0.0114 [+0.0045, +0.0206] above zero | +0.0078 [+0.0024, +0.0142] above zero | +0.0042 [+0.0004, +0.0081] above zero |
| m = 32 − recipe (group) | +0.0095 [+0.0037, +0.0172] above zero | +0.0053 [+0.0005, +0.0108] above zero | +0.0017 [−0.0022, +0.0056] includes zero |
| full OT − recipe (cell) | +0.0008 [−0.0006, +0.0025] includes zero | +0.0008 [−0.0003, +0.0020] includes zero | +0.0012 [+0.0000, +0.0025] above zero |
| m = 32 − recipe (cell) | +0.0009 [−0.0002, +0.0022] includes zero | +0.0011 [+0.0002, +0.0021] above zero | +0.0011 [+0.0002, +0.0020] above zero |
| real cells − recipe (group) | +0.0651 [+0.0441, +0.0915] above zero | +0.0909 [+0.0552, +0.1418] above zero | +0.0536 [+0.0308, +0.0790] above zero |
| real cells − recipe (cell) | +0.0083 [+0.0048, +0.0129] above zero | +0.0093 [+0.0055, +0.0146] above zero | +0.0051 [+0.0025, +0.0080] above zero |
| recipe − vehicle cells (group) | −0.0026 [−0.0096, +0.0050] includes zero | +0.0100 [−0.0059, +0.0394] includes zero | +0.0020 [−0.0106, +0.0227] includes zero |
| recipe − vehicle cells (cell) | −0.0004 [−0.0021, +0.0010] includes zero | −0.0003 [−0.0024, +0.0017] includes zero | +0.0001 [−0.0021, +0.0019] includes zero |
| recipe − ridge-train (group) | −0.0138 [−0.0241, −0.0042] below zero | −0.0062 [−0.0155, +0.0026] includes zero | −0.0079 [−0.0164, −0.0001] below zero |
| recipe − ridge-S7 (group) | −0.0061 [−0.0141, +0.0024] includes zero | +0.0048 [−0.0115, +0.0342] includes zero | −0.0011 [−0.0142, +0.0190] includes zero |
| Q1 on the raw hit (group) | −0.0055 [−0.0138, +0.0029] includes zero | +0.0090 [−0.0073, +0.0379] includes zero | +0.0004 [−0.0126, +0.0208] includes zero |
| Q2 on the raw hit (cell) | −0.0006 [−0.0022, +0.0009] includes zero | −0.0005 [−0.0028, +0.0016] includes zero | +0.0001 [−0.0022, +0.0019] includes zero |
| recipe − vehicle cells, raw hit (group) | +0.0017 [−0.0050, +0.0092] includes zero | +0.0190 [+0.0032, +0.0481] above zero | +0.0067 [−0.0063, +0.0281] includes zero |
| Q1 variants, group: sign-agnostic / any set | −0.0075 / −0.0077 includes zero | +0.0085 / +0.0127 includes zero | +0.0008 / −0.0003 includes zero |

The sign-aware and the sign-blind chance floors agree to within 0.003 for every set except the vehicle cells (whose
assigned answers carry random signs; §3 of the prereg amendment), so the amendment did not change any verdict: Q1
and Q2 on the raw hit also include zero on every split.

## 3. Reading

1. **The instrument works here, but its ceiling is low.** Real half-populations are recognised 8.2 %, 12.0 % and
   7.5 % of the time against a chance of 1.3–1.5 %; 43–48 % of them receive no credible set; per cell, 1.3–1.8 %
   with 77–80 % unassigned. The package's own low-signal filter dropped 12, 9 and 10 of the 75 held-out
   structures. On this panel (200 genes, one fingerprint per compound pooled over 14 lines and three dose levels)
   most Tahoe conditions do not carry a compound signature the method can find even in real cells, which agrees
   with the small effects measured throughout Steps 14–21.
2. **The recipe's generated populations carry no recognisable compound signature beyond the shared response.**
   Recognised 1.5 %, 2.8 % and 2.1 % of the time against chance 1.1–1.4 %; not distinguishable from the
   compound-blind mean (Q1) nor from the untouched vehicle cells (−0.003 / +0.010 / +0.002, all including zero), on
   any split and at either level.
3. **Pairing helps a little, and ridge-train is at least as good.** Full OT is recognised more often than the recipe
   on all three splits (+0.011, +0.008, +0.004) and m = 32 on two; both reach about the ridge-train reference
   (0.017 / 0.020 / 0.016), which beats the recipe outright on two splits. All these differences are of the order of
   one population in a hundred; the real-cell ceiling is 5–9 points higher than any generated set.
4. **The compound-blind mean is itself barely recognised** (0.009 / 0.008 / 0.009), unlike on the mock, where it
   collected large credible sets: on real Tahoe data the line's mean response does not resemble any single
   compound's fingerprint closely enough to be assigned.

## 4. Exploratory (prereg §4.4; descriptive)

- **By dose (group level, hit rate; real / recipe / compound-blind mean).** Split 1: 50 nM 0.092 / 0.013 / 0.024
  (993 conditions), 500 nM 0.079 / 0.015 / 0.023 (1,010), 5,000 nM 0.075 / 0.016 / 0.013 (896). Split 0: 0.087 /
  0.016 / 0.016 (1,172), 0.136 / 0.021 / 0.019 (991), 0.142 / 0.048 / 0.021 (952). Split 2: 0.058 / 0.012 / 0.018
  (1,020), 0.074 / 0.021 / 0.021 (959), 0.094 / 0.030 / 0.021 (874). Recognition of the recipe rises with dose on
  splits 0 and 2 (to 4.8 % and 3.0 % at 5 µM).
- **Recognition concentrates in a few compounds.** The best-recognised real structures per split: split 1 P001
  0.571, P050 0.357, P035 0.333 (hit rates over their 28, 42 and 27 conditions); split 0 P066 0.846, P013 0.764, P025
  0.717, P009 0.463; split 2 P020 0.643, P062 0.310, P048 0.264. For most of these the recipe scores 0.000–0.05; the
  two exceptions are P025 on split 0 (recipe 0.528 against real 0.717) and P048 on split 2 (recipe 0.340 against
  real 0.264). Labels map to SMILES in each split's `manifest.json`.
- **Uncovered conditions** (structures the filter dropped: 422 / 329 / 314 conditions): real half-populations were
  unassigned 53.6 % / 45.6 % / 46.5 % of the time, the recipe's 61.7 % / 56.8 % / 60.9 %.
- **Log Bayes factors of the top set (group, real cells):** hits average 53.6 / 54.9 / 53.6, misses 26.8 / 32.4 / 30.7
  (203 / 333 / 190 hits against 1,144 / 1,267 / 1,123 assigned misses); the recipe 39.0 / 39.4 / 40.5 against 28.0 /
  29.2 / 29.1 (109 / 231 / 156 hits).

## 5. Expectation audit (prereg §7)

- Gate: expected to pass; passed. The level was lower than imagined: "strong group-level recognition" as in the
  authors' sci-Plex analyses did not transfer to a 200-gene, pooled-line dictionary.
- Q1: expected to hold; did not. The named risk was the right one: the recipe's gene-level signal does not exceed the
  shared response on this instrument either. The category expected was 2; the outcome is 3.
- Q2: expected not to hold; did not.
- Secondary: full OT was expected *below* the recipe at the cell level because its cells track their vehicle cell;
  observed above the recipe at group level on all splits and at cell level on one (OT) or two (m = 32), by about one
  population in a hundred. A tentative reading: the paired models' populations are closer to the real ones (Step 18),
  and the fingerprint test rewards that closeness slightly; the expectation was wrong.

## 6. Implementation record

- Toolchain: `provenance/R_env/` (R 4.5.3, Seurat 5.5.1, presto 1.1.0 at b5df6ee, RNAFingerprinting 0.0.0.9000 at
  4004c8d; commit 312cc37). The package is called through its two exported functions with default parameters; input
  layer per prereg §3.2 (standardised log-expression, clipped at ±10, parameters fitted on the dictionary cells).
- Scripts: `prep_step22_fingerprint.py` (f177e50), `run_step22_fingerprint.R`, `run_step22_tahoe.sh` (da38d5f),
  `score_step22_fingerprint.py` (042ab8b), `plot_step22.py` (b4e2ff9); later storage-only revisions (7ccadc4).
  Mock run job 18785013 and mock scoring passed (PASS) before the development run.
- Development run: job 18786995 (stage 1 272 s; K = 63) stopped in the package's cell-level mapping of the
  compound-blind mean set. Diagnosis (jobs 18790851, 18791328, 18791538): three cells of one condition triggered an
  unguarded branch of the package — after a joint-fingerprint proposal the refit found no effect, and the package
  called its calibration on an empty set. A one-line guard returning "unassigned" (the package's own answer for a fit
  without credible sets) was applied as `provenance/R_env/rnafingerprinting_classify_guard.patch` (ae02490); the R
  stage and the scorer require it. The run resumed as job 18792770 (the sets already answered were kept, since the
  guard only acts where the unpatched code crashed). Development scoring: verdict 3 on split 1.
- Confirmation runs: jobs 18794425 (split 0) and 18794426 (split 2), scripts frozen at 7ccadc4, 2,431 s and 2,649 s of
  R time. Final scoring over the three splits: commit 7ccadc4, 137 s.
- Records committed: `tahoe_results/step22/<split>/` (manifest, dictionary, answers, R session, timings), logs under
  `tahoe_results/step22_logs/`, analysis records, this document. Matrices and per-cell metadata CSVs are gitignored
  (sha256 in each manifest).
- Order of events: the §4 amendment (chance-corrected hit; sign-aware floor) was made after the mock run and before
  any real run; the §3.1 guard note after the development run stopped and before any result was scored; the §9
  storage note after the development scoring and before the confirmation runs. No endpoint, gate, cap, label or
  verdict rule changed after a development result was seen.

## 7. Limits

1. **Input layer.** Standardised log-expression instead of SCTransform residuals (prereg §3.2); the real-cell ceiling
   (7–12 %) is measured on that input, and it is not known how much of the low ceiling is the input and how much the
   panel.
2. **One fingerprint per compound** pooled over 14 lines and three doses. A per-line dictionary would ask a
   different question (within-line recognition) and might raise the ceiling; not pre-registered, not run.
3. **The test is a recognition test**: the dictionary is learned from the held-out compounds' real cells. It does not
   ask whether the model predicts, only whether its output looks like the compound's real response.
4. **Package patch.** One guard in the package's joint-refit branch (§6); it changes no answer of any query that did
   not crash. The upstream package has not been changed; the defect should be reported to its authors.
5. **Cell level at 10 cells per condition**; 77–84 % of cells unassigned, so the cell-level contrasts are decided by
   a few hundred assigned cells per set.
6. **Three seeds; 97.5 % per-split intervals;** "includes zero" is not "absent": Q1 on split 0 is +0.006 with an
   upper bound of +0.035.
7. **Ninth use of the confirmation splits.**

## 8. Figure

`figures/step22_fingerprint.png` from `plot_step22.py` (reads only the two committed records; every plotted value
checked against the JSON to 1e-12). Panels A and B: chance-corrected hit per set at group and cell level; C: gate,
primaries and pairing secondaries with intervals. Printed values equal the record.

## 9. References

- Grabski IN, Blair JD, … Knowles DA, Satija R (2026). Mapping transcriptional responses to cellular perturbations
  with RNA fingerprinting. *Cell* 189, in press. doi:10.1016/j.cell.2026.09.017. Package
  github.com/satijalab/rna-fingerprinting, commit 4004c8d.
- Wang G, Sarkar A, Carbonetto P, Stephens M (2020). *J R Stat Soc Ser B* 82:1273–1300. doi:10.1111/rssb.12388.
- Hao Y et al. (2024). *Nat Biotechnol* 42:293–304. doi:10.1038/s41587-023-01767-y (Seurat v5).
- Berger RL (1982). *Technometrics* 24(4):295–300.

## 10. What the thesis can say

An external, published recognition method (RNA fingerprinting), given one fingerprint per held-out compound learned
from half of the real cells, recognises the compound in the other half of the real cells above chance on every
split (7–12 % of populations, against 1–2 % by chance; corrected 10 Nov 2026 from 8–12 %, since split 2 is 7.48 %), but recognises it in the recipe's generated populations no
more often than in a vehicle cell shifted by the line's mean response, and no more often than in the untouched
vehicle cells (pre-registered verdict NOT CONFIRMED, all three splits). The optimal-transport-paired models are
recognised slightly more often than the recipe (about one population in a hundred, resolved on all three splits for
full OT), as is the ridge fitted on the training compounds. On this 200-gene panel the compound-specific signal that
the method can find is small even in real cells, and the generated populations carry none of it that the shared
response does not already carry.
