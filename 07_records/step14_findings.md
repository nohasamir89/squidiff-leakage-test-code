# Step 14 findings: linear-shift baselines and the six VCC-style facets

Pre-registration: `PREREG_step14_shift_baselines_vcc_scoring.md` (commit 5473582; Amendment 1, written before any
real data were scored, commit f23b4d3). Scorer: `score_step14.py` at commit 1ef3c07 (md5
c7506f0635debbcf8d9d4d090ccafa0c, 409 lines) with `step14_metrics.py` (md5 1b7dd6a018190d87d457ad0cd0d6eaf7,
sha256 e136b81c…). Records: `leakage_results/step14_analysis.json`, `leakage_results/step14_long.csv`,
`leakage_results/step14_analysis.log`, commit c354314. Run time 1147 s. Date: 2026-09-23.

Every number below is copied from `step14_analysis.json` or, where stated, from the committed log. Intervals are
97.5% drug-clustered percentile bootstrap intervals over the split's compounds (5000 replicates, `RandomState(0)`),
on all three splits alike (PREREG §5: nothing in this step is chosen on any split).

## 1. Pre-registered verdict: PARTIAL

Both gated contrasts (ridge-S7 shift minus unguided 50 NFE flow matching) resolved on every split. The primary agrees
across splits; the co-primary does not.

| split | primary: DE-overlap Jaccard | co-primary: energy distance (+ = shift worse) | pattern on this split |
|---|---|---|---|
| drug_split_1 | −0.0556 [−0.0722, −0.0355], below zero | +0.0513 [+0.0061, +0.1062], above zero | generative advantage |
| drug_split_0 | −0.0439 [−0.0536, −0.0340], below zero | −0.0161 [−0.0260, −0.0087], **below zero** | trade (shift closer in distribution) |
| drug_split_2 | −0.0570 [−0.0702, −0.0443], below zero | +0.0434 [+0.0041, +0.0908], above zero | generative advantage |

Applying PREREG §6 mechanically (the rule committed in `score_step14.py` before the run): category 1 (LINEAR SHIFT
AHEAD) fails, the primary is below zero everywhere; category 2 (GENERATIVE ADVANTAGE) fails, the co-primary is not
above zero on split 0; category 3 (TRADE) fails in both directions; category 5 (UNRESOLVED) fails. The scorer
assigns category 4, **PARTIAL**, as the residual, and §6 requires it to be reported per split, which the table does.

**Wording note, recorded for the thesis.** §6 phrases category 4 as "either contrast resolved on one or two splits
only". The observed pattern — the co-primary resolved on all three splits, in the generative model's favour on two
and against it on one — is not literally described by that phrase, nor by any other category. The verdict stands as
the committed rule produced it; it is not upgraded to GENERATIVE ADVANTAGE on the strength of two splits, because
§5 makes agreement across all three the condition for categories 1–3.

## 2. Measurements: the thesis's three metrics, every submission

DE-overlap and DPC-50 are profile-level (mean prediction vs condition mean); energy is cell-level, against the
harness's stored real-cell draw (§8). Arms: five checkpoints, metrics averaged per condition. "median" is the
condition median of DPC-50. `control_paste` has no top-50 set (zero delta), so its DE-overlap and DPC-50 are
undefined. Base Squidiff has no per-cell record, so no energy.

**drug_split_1** (443 conditions, 37 compounds)

| submission | DE-overlap | energy | DPC-50 mean | DPC-50 median |
|---|---|---|---|---|
| ridge-S7 shift | 0.5544 | 0.2308 | 0.1891 | 0.1884 |
| ridge-train shift | 0.5644 | 0.1630 | 0.1963 | 0.1795 |
| mean-response shift | 0.5440 | 0.2082 | 0.1795 | 0.1264 |
| control-paste | — | 0.2000 | — | — |
| adaLN DDIM 1000 | 0.5548 | 0.1889 | 0.0875 | 0.0644 |
| flow matching 20 NFE | 0.5858 | 0.1862 | 0.0977 | 0.0924 |
| flow matching 50 NFE (comparator) | 0.6100 | 0.1795 | 0.0991 | 0.0940 |
| guidance w = 2 | 0.6172 | 0.1846 | 0.1177 | 0.0927 |
| cross-attention | 0.6157 | 0.1587 | 0.1183 | 0.1213 |
| matched control | 0.6125 | 0.1762 | 0.1072 | 0.1049 |
| Step 9 pathway / both | 0.5858 / 0.5890 | 0.1768 / 0.1807 | 0.1137 / 0.1057 | 0.1084 / 0.1056 |
| Step 10 ChemVAE / FP + ChemVAE | 0.5794 / 0.5820 | 0.2237 / 0.2056 | 0.0550 / 0.0823 | 0.0445 / 0.0810 |
| base Squidiff (profile only) | 0.5080 | — | 0.0452 | 0.0384 |

**drug_split_0** (444 conditions, 37 compounds)

| submission | DE-overlap | energy | DPC-50 mean | DPC-50 median |
|---|---|---|---|---|
| ridge-S7 shift | 0.5684 | 0.0322 | 0.2023 | 0.2473 |
| ridge-train shift | 0.5369 | 0.0591 | 0.1234 | 0.1058 |
| mean-response shift | 0.5538 | 0.0288 | 0.2504 | 0.3042 |
| control-paste | — | 0.0304 | — | — |
| adaLN DDIM 1000 | 0.5585 | 0.0563 | 0.0602 | 0.0921 |
| flow matching 20 NFE | 0.5906 | 0.0531 | 0.0670 | 0.0888 |
| flow matching 50 NFE (comparator) | 0.6124 | 0.0483 | 0.0698 | 0.1004 |
| guidance w = 2 | 0.6275 | 0.0697 | 0.0793 | 0.0935 |
| cross-attention | 0.6110 | 0.0406 | 0.0790 | 0.0886 |
| matched control | 0.6123 | 0.0465 | 0.0675 | 0.0801 |
| base Squidiff (profile only) | 0.5047 | — | 0.0097 | 0.0184 |

**drug_split_2** (455 conditions, 38 compounds)

| submission | DE-overlap | energy | DPC-50 mean | DPC-50 median |
|---|---|---|---|---|
| ridge-S7 shift | 0.5535 | 0.1612 | 0.0982 | 0.0816 |
| ridge-train shift | 0.5473 | 0.1420 | 0.1211 | 0.0950 |
| mean-response shift | 0.5325 | 0.1428 | 0.1310 | 0.0858 |
| control-paste | — | 0.1388 | — | — |
| adaLN DDIM 1000 | 0.5579 | 0.1370 | 0.0708 | 0.0743 |
| flow matching 20 NFE | 0.5859 | 0.1235 | 0.0661 | 0.0626 |
| flow matching 50 NFE (comparator) | 0.6106 | 0.1178 | 0.0691 | 0.0538 |
| guidance w = 2 | 0.6222 | 0.1238 | 0.0781 | 0.0647 |
| cross-attention | 0.6116 | 0.1071 | 0.0856 | 0.0979 |
| matched control | 0.6135 | 0.1127 | 0.0848 | 0.0967 |
| base Squidiff (profile only) | 0.4908 | — | 0.0508 | 0.0257 |

Consistency with earlier records: the arms' DE-overlap and energy values equal the Figure 17/18 records (unguided
50 NFE 0.6100 / 0.6124 / 0.6106 and 0.1795 / 0.0483 / 0.1178; guidance w = 2 0.6172 / 0.6275 / 0.6222) to four
decimals, and the ridge-S7 DPC-50 medians equal the Step 7 record to nine decimals (§8).

Fraction of shifted-cell entries clipped at zero (committed log): ridge-S7 29.7% / 35.2% / 31.7%, ridge-train
43.7% / 39.7% / 42.0%, mean-response 11.6% / 14.7% / 12.8% (splits 1 / 0 / 2).

## 3. Q1 read per split, with the secondary contrasts

All contrasts are "baseline minus unguided 50 NFE" unless stated; on energy, positive means the baseline is worse.

| contrast | drug_split_1 | drug_split_0 | drug_split_2 |
|---|---|---|---|
| DE-overlap, ridge-train − 50 NFE | −0.0456 [−0.0645, −0.0233] below | −0.0755 [−0.0856, −0.0655] below | −0.0633 [−0.0767, −0.0492] below |
| energy, ridge-train − 50 NFE | −0.0166 [−0.0816, +0.0377] includes 0 | +0.0109 [−0.0032, +0.0282] includes 0 | +0.0242 [−0.0203, +0.0723] includes 0 |
| DE-overlap, mean-response − 50 NFE | −0.0660 [−0.0858, −0.0419] below | −0.0586 [−0.0652, −0.0518] below | −0.0781 [−0.0892, −0.0661] below |
| energy, mean-response − 50 NFE | +0.0287 [+0.0103, +0.0515] above | −0.0195 [−0.0295, −0.0117] **below** | +0.0250 [−0.0157, +0.0930] includes 0 |
| energy, control-paste − 50 NFE | +0.0205 [−0.0142, +0.0675] includes 0 | −0.0179 [−0.0284, −0.0093] **below** | +0.0210 [−0.0251, +0.0986] includes 0 |
| DE-overlap, ridge-S7 − guidance w = 2 | −0.0628 [−0.0795, −0.0424] below | −0.0591 [−0.0699, −0.0479] below | −0.0686 [−0.0805, −0.0559] below |
| energy, ridge-S7 − guidance w = 2 | +0.0462 [−0.0112, +0.1170] includes 0 | −0.0376 [−0.0614, −0.0202] below | +0.0375 [−0.0410, +0.1444] includes 0 |

What this establishes:

- **Gene identity (DE-overlap).** The unguided 50 NFE model is ahead of every shift baseline on every split, and
  the intervals are far from zero (half-widths 0.007–0.022 against differences of 0.044–0.078). The ridge shifts
  add little over the compound-blind mean response on this metric (ridge-S7 0.5544 / 0.5684 / 0.5535 against
  mean-response 0.5440 / 0.5538 / 0.5325); the generative model adds 0.059–0.078.
- **Distribution shape (energy).** Against the ridge-S7 shift the model is closer to the real cells on splits 1
  and 2 and farther on split 0. On split 0 it is also farther than the unshifted control cells (control-paste,
  −0.0179 [−0.0284, −0.0093]) and than the mean-response shift. On splits 1 and 2 the model-versus-control-paste
  contrast includes zero. The model's energy advantage is therefore established only against the ridge shift,
  and the ridge shift is a weak opponent on this metric: it is worse than an unshifted control on all three splits
  (0.2308 vs 0.2000; 0.0322 vs 0.0304; 0.1612 vs 0.1388), with 30–35% of its cell entries clipped at zero.
- **Split 0 is the small-effect split.** Its energy distances are several times smaller than the other splits'
  for every submission (control-paste 0.0304 against 0.2000 and 0.1388; 50 NFE 0.0483 against 0.1795 and 0.1178),
  and its MSE ratios (§6) are above 1 for every arm. §7 of the pre-registration fixed the consequence in advance: where the shift is not worse on
  energy, "the generative model's remaining claim is gone and the thesis says so". On split 0 it is gone.

## 4. The like-for-like ridge (ridge-train)

The ridge-train shift is fitted on the training split's 1344 / 1344 / 1332 condition deltas (112 / 112 / 111
compounds) and applied to the held-out compounds — the generative arms' information regime. Against it the model
is ahead on DE-overlap on all three splits (§3), indistinguishable on energy on all three, and below on DPC-50
(medians 0.0940 / 0.1004 / 0.0538 against 0.1795 / 0.1058 / 0.0950; no interval was pre-registered for this
contrast).

PREREG §7 expected ridge-train below ridge-S7 on DPC-50 and DE-overlap. Observed: DPC-50 median below on splits 1
and 0 (0.1795 vs 0.1884; 0.1058 vs 0.2473), above on split 2 (0.0950 vs 0.0816); DE-overlap above on split 1
(0.5644 vs 0.5544), below on splits 0 and 2; energy better than ridge-S7 on splits 1 and 2, worse on split 0.

## 5. DPC-50 and the mean-response shift

The mean-response shift carries no compound information: one delta per cell line, the equal-weight mean of the
line's held-out condition deltas (transductive; PREREG §3, §8). Its condition-median DPC-50 is 0.1264 / 0.3042 /
0.0858 (mean 0.1795 / 0.2504 / 0.1310). Against this:

- the Step 7 ridge (within-held-out, fingerprint × line) reaches 0.1884 / 0.2473 / 0.0816 — above the
  compound-blind shift on split 1 by 0.062, below it on split 0 by 0.057, level on split 2;
- every generative arm on every split is below it (best arm medians 0.1213 / 0.1004 / 0.0979);
- base Squidiff is at 0.0384 / 0.0184 / 0.0257.

*Dated correction, 2026-11-09 (found while checking the Step 28 sci-Plex result).* The second bullet is wrong on
split 2 as written: it compares medians, and on split 2 the best arm medians (cross-attention 0.0979, matched
control 0.0967) are above the mean-response median (0.0858). It holds for the condition mean on every split (every
generative arm at most 0.1183 / 0.0793 / 0.0856 against 0.1795 / 0.2504 / 0.1310) and for the median on splits 1
and 0. The manuscript's Discussion now states the comparison for the mean (manuscript rev 167).

DPC-50 correlates the predicted and measured deltas on the condition's true top-50 |delta| genes. These genes are
dominated by the line's shared response (§6, reach), so a submission that predicts only that shared direction
scores well on DPC-50 without any compound-specific information. Two consequences for the thesis:

1. The DPC-50 margin of the Step 7 ridge over the generative arms (Steps 7–13b) is, on splits 0 and 2, no larger
   than the margin of a compound-blind shift; the phrase "recovers more of the compound-specific signal" in
   PREREG_step14 §1 (and in the diagnosis language of Step 13 §8 / Step 13b §10) overstates what DPC-50 isolates.
   The arms' weakness on DPC-50 is a weakness on the shared direction as much as on the compound-specific one.
2. The generative arms' advantage is on which genes move (DE-overlap, §3), where they beat both ridges and the
   mean response on every split.

The mean-response shift uses the held-out conditions' own means and is not a predictor. A legitimate compound-blind
reference would be the training split's per-line mean delta; it was not pre-registered here and is not computed.
The ridge-train shift contains it implicitly (line one-hot columns) but adds shrunk fingerprint terms.

## 6. The six facets (descriptive; nothing gated) — first run; scaled values superseded by §11

*Dated note, 2026-09-24.* The raw facets below stand (the second run changed them by at most 1e-4 on fidelity and
Jaccard and 0.015 on reach, apart from control-paste reach, §9 item 1). The scaled values and the "overall" below
were computed with five departures from the cell-eval2 procedure (Amendment 2) and are superseded by §11.

Raw facets are per-line values (mean over the line's gated conditions; MSE is the line ratio) averaged over the
three lines. Effects are taken against the condition's paired-control mean (Amendment 1). The replicate anchor
splits each condition's real cells and its paired vehicle cells into halves, five seeds.

**Replicate anchors (raw, per line)**

| split | line | PDS | MSE | fidelity | reach | sig-Jaccard | lfc-NMAE |
|---|---|---|---|---|---|---|---|
| 1 | A549 / K562 / MCF7 | 0.690 / 0.646 / 0.659 | 0.191 / 0.471 / 0.064 | 0.810 / 0.647 / 0.749 | 1.000 / 0.972 / 0.966 | 0.921 / 0.893 / 0.915 | 0.173 / — / 0.418 |
| 0 | A549 / K562 / MCF7 | 0.603 / 0.633 / 0.583 | 0.697 / 0.677 / 0.433 | 0.764 / 0.610 / 0.491 | 0.979 / 0.961 / 0.860 | 0.984 / 0.932 / 0.948 | — / — / 0.472 |
| 2 | A549 / K562 / MCF7 | 0.665 / 0.655 / 0.636 | 0.261 / 0.546 / 0.071 | 0.743 / 0.574 / 0.751 | 1.000 / 0.950 / 0.970 | 0.954 / 0.868 / 0.913 | 0.247 / — / 0.474 |

"—": no gated condition of that line reached ten significant genes with finite fold change in the half-sized
reference, so the replicate NMAE is undefined there.

**Raw facets, split means over lines** (the CSV holds every line)

| split | submission | PDS | MSE ratio | fidelity | reach | sig-Jaccard | lfc-NMAE |
|---|---|---|---|---|---|---|---|
| 1 | ridge-S7 | 0.5544 | 0.873 | 0.587 | 0.823 | 0.016 | 0.303 |
| 1 | ridge-train | 0.5919 | 0.650 | 0.587 | 0.815 | 0.013 | 0.241 |
| 1 | mean-response | 0.5000 | 0.918 | 0.586 | 0.941 | 0.015 | 0.338 |
| 1 | control-paste | 0.5000 | 1.000 | 0.000 | 0.265* | 0.795 | 1.000 |
| 1 | 50 NFE | 0.5303 | 0.931 | 0.739 | 0.763 | 0.008 | 0.334 |
| 1 | guidance w = 2 | 0.5565 | 0.956 | 0.731 | 0.781 | 0.008 | 0.276 |
| 1 | cross-attention | 0.5379 | 0.844 | 0.746 | 0.765 | 0.008 | 0.324 |
| 1 | base | 0.5019 | 1.400 | — | — | — | — |
| 0 | ridge-S7 | 0.5373 | 1.039 | 0.581 | 0.832 | 0.003 | 0.547 |
| 0 | ridge-train | 0.5365 | 1.927 | 0.552 | 0.748 | 0.002 | 0.382 |
| 0 | mean-response | 0.5000 | 0.945 | 0.581 | 0.924 | 0.003 | 0.508 |
| 0 | control-paste | 0.5000 | 1.000 | 0.000 | 0.444* | 0.919 | 1.000 |
| 0 | 50 NFE | 0.5300 | 1.608 | 0.738 | 0.628 | 0.001 | 0.402 |
| 0 | guidance w = 2 | 0.5430 | 2.141 | 0.733 | 0.583 | 0.001 | 0.413 |
| 0 | cross-attention | 0.5309 | 1.436 | 0.745 | 0.639 | 0.001 | 0.415 |
| 0 | base | 0.5059 | 4.280 | — | — | — | — |
| 2 | ridge-S7 | 0.5010 | 0.890 | 0.581 | 0.756 | 0.011 | 0.363 |
| 2 | ridge-train | 0.5520 | 0.787 | 0.572 | 0.715 | 0.009 | 0.339 |
| 2 | mean-response | 0.5000 | 0.941 | 0.586 | 0.902 | 0.010 | 0.435 |
| 2 | control-paste | 0.5000 | 1.000 | 0.000 | 0.421* | 0.808 | 1.000 |
| 2 | 50 NFE | 0.5158 | 0.906 | 0.737 | 0.664 | 0.005 | 0.395 |
| 2 | guidance w = 2 | 0.5325 | 0.930 | 0.731 | 0.646 | 0.005 | 0.350 |
| 2 | cross-attention | 0.5246 | 0.836 | 0.741 | 0.681 | 0.005 | 0.385 |
| 2 | base | 0.5074 | 1.524 | — | — | — | — |

\* control-paste reach is a rounding artefact and should read 0 (§9).

Facet contrasts, ridge-S7 minus 50 NFE, per condition (97.5%): PDS +0.024 [−0.025, +0.072], +0.007 [−0.046,
+0.061], −0.015 [−0.067, +0.040] (includes zero on all three); direction fidelity −0.152 [−0.174, −0.126], −0.157
[−0.170, −0.144], −0.156 [−0.179, −0.133] (below zero on all three); significant-set Jaccard +0.008 [+0.002,
+0.017], +0.001 [+0.000, +0.002], +0.005 [+0.001, +0.011] (above zero on all three); reach and NMAE undefined
(fewer than half the conditions carry a value, next paragraph).

**What the raw facets show.**

- *The reference is empty in most conditions.* The control-paste cells are the paired vehicle cells themselves, so
  its significant set is empty in every condition and its significant-set Jaccard equals 1 exactly when the
  reference (real cells vs paired vehicle, Wilcoxon, BH 0.05 within the condition) has no significant gene and 0
  otherwise. Its per-line values therefore give the fraction of gated conditions with no significant reference
  gene on the 200 genes: 0.82 / 0.79 / 0.77 (A549 / K562 / MCF7) on split 1, 0.94 / 0.89 / 0.93 on split 0, 0.86 /
  0.79 / 0.77 on split 2. Reach and NMAE exist only in the remaining conditions, which is why their contrasts are
  undefined by the scorer's rule (fewer than half the conditions finite).
- *PDS.* Every submission lies within 0.06 of the 0.5 no-information point; the replicate anchor itself is only
  0.58–0.69. Ridge-train is the highest on splits 1 and 2 (0.592 / 0.552) and guidance w = 2 on split 0 (0.543);
  ridge-S7 is at 0.554 / 0.537 / 0.501, at the no-information point on split 2. PREREG §7's expectation (arms
  within 0.05 of 0.5, ridge shifts above 0.5) held except guidance w = 2 on split 1 (0.5565) and ridge-S7 on
  split 2 (0.5010).
- *MSE ratio.* Values above 1 mean the prediction is farther from the condition mean than the paired-control mean
  is. On split 0 every arm is above 1 (50 NFE 1.61, guidance 2.14, base 4.28) and ridge-S7 is at 1.04; only the
  mean response (0.94) is below. On splits 1 and 2 the ladder arms from 20 NFE upward are at 0.84–0.96, adaLN
  DDIM (1.00 / 1.05) and the Step 10 arms (1.12 / 1.02) at or above 1, and ridge-train at 0.65 / 0.79.
- *Direction fidelity.* The arms are at 0.71–0.75 on every split against 0.55–0.59 for every shift, and the
  contrast is resolved on all three splits. Fidelity counts the submission's significant calls that agree in sign
  with the reference, over max(n_pred, n_real); the arms' 200 low-dispersion cells make many calls (their
  significant-set Jaccard with the reference is 0.001–0.008), and about three quarters of those calls have the
  reference's sign.
- *Reach.* The mean-response shift is the highest (0.90–0.94, at or near the replicate's 0.86–1.00): ranking the
  reference's significant genes by the submission's confidence, the shared line response has the right sign for
  nearly all of them. The arms are at 0.58–0.78.
- *NMAE.* Ridge-train is the lowest on every split (0.241 / 0.382 / 0.339); among arms, guidance w = 2 is the
  lowest on splits 1 and 2 (0.276 / 0.350) and adaLN DDIM on split 0 (0.386); values range 0.24–0.55 across
  submissions.

**Scaled scores.** `s = (u − b)/(r − b)` per line, then averaged; MSE clipped to [0, 1], NMAE floored at −6, as
specified. The split-level scaled values and the six-facet overall are in the JSON and CSV (50 NFE overall:
+0.476 / −1.254 / +3.013 on splits 1 / 0 / 2). They are recorded because they were pre-registered and are not
interpretable here, for three structural reasons visible in the replicate table: (i) on reach and NMAE the
replicate and the mean-response baseline nearly coincide or invert (replicate minus baseline on reach: +0.079 /
+0.063 / −0.028 per line on split 1, +0.145 / +0.023 / −0.140 on split 0, +0.143 / −0.009 / +0.079 on split 2),
so the denominator is near zero or negative and the scaled values run from −8 to +30; (ii) the replicate's DE
facets are computed from half the cells and so have less power than any full-size submission — the arms' scaled
fidelity is 0.7–1.2 on splits 1 and 0 and 2.8–3.5 on split 2, at or above the "1" point on two splits — so the
anchor is not a ceiling for the DE facets; (iii) the
replicate NMAE is undefined for K562 on every split and for A549 on split 0, so those lines drop out of the
scaled NMAE mean. PREREG §7 expected every scaled score far below 1; that expectation was wrong for these reasons,
not because any submission approaches replicate quality — the raw PDS and MSE columns say the opposite.

## 7. Expectation audit (PREREG §7)

| expectation | outcome |
|---|---|
| TRADE, with the ridge-S7 shift above 50 NFE on DE-overlap on splits 0 and 1 | wrong: the shift is below on all three splits |
| split 2 the one most likely unresolved on DE-overlap | wrong: resolved, in the model's favour, on split 2 as well |
| shift worse on energy on all three | right on splits 1 and 2; wrong on split 0 |
| "if the shift is not worse on energy, the generative model's remaining claim is gone" | applies on split 0, where the model is also farther from the real cells than the unshifted control |
| ridge-train below ridge-S7 on DPC-50 and DE-overlap | mixed (§4) |
| arms' raw PDS within 0.05 of 0.5; ridge shifts above | held with two exceptions (§6) |
| every scaled score far below 1, fidelity and reach lowest | wrong, for structural reasons (§6) |

## 8. Implementation record

- Inputs: 11 submissions × 5 checkpoints on split 1 and 7 × 5 on splits 0 and 2, every vectors and per-cell file
  hash-checked against its provenance record; the hashes are written under `inputs` in the JSON. Split files
  `sci_plex_{test,train}_<split>[_control].h5ad` from `datasets_corrected`.
- **Step 7 reproduction.** The ridge-S7 shift's DPC-50 median equals the Step 7 record on every split:
  0.18838950991630554, 0.24729523062705994, 0.08158174157142639. The guard tolerance is 1e-9. Two earlier runs
  stopped at this guard before any endpoint was computed: the first reproduced the ridge from the 50 NFE target
  file (its `true − C` differs from the Step 7 file by up to 6.0e-8 / 5.2e-8 / 1.1e-7 per entry) and missed the
  tolerance while agreeing to six printed decimals; the second used float64 arithmetic and missed by 8e-9
  (0.188389502). The committed scorer follows
  `analyse_step7.py` operation for operation — `true − C` read as stored (float32), fingerprint matrix uncast,
  float32 prediction store, features `[FP, L, FP × L]`, folds grouped by SMILES, Pearson on
  `argsort(|Y|)[-50:]` — and casts to float64 only afterwards.
- **Harness draw.** The Step 6–13b harness draws each condition's real cells with replacement (seeded, identical
  across arms), so the stored per-cell "true" arrays hold duplicates in 263 / 280 / 279 conditions. Every stored
  cell was recovered exactly in the test file (443 / 444 / 455 conditions, none within tolerance only, no seeded
  draw needed). The distinct matched cells (7–200 / 15–199 / 10–200 per condition) and their row-aligned paired
  vehicle cells supply the DE reference, the cell gate, the shift baselines' cells and the replicate anchor; the
  stored draw as-is supplies the real side of every energy distance, so the baselines are scored exactly as the
  arms were and the arms' energy values reproduce the Figure 17/18 records. The duplicates make the earlier energy
  distances internally consistent, not wrong; an energy distance against distinct cells would differ slightly and
  was not pre-registered.
- **Cell gate.** ≥ 20 distinct real cells: 439 / 443, 442 / 444, 452 / 455 conditions; the excluded conditions
  drop out of the four DE facets and the replicate anchor for every submission alike; the thesis metrics use all
  conditions.
- **Shift baselines.** Per condition, the distinct paired vehicle cells plus the delta, clipped at zero; profile
  metrics (DE-overlap, DPC-50, PDS, MSE) use the unclipped profile `C + delta`, as PREREG §2 defines the ridge-S7
  DPC-50 by construction; cell metrics (energy, DE facets) use the clipped cells.
- **Ridge-train.** Training deltas are per-condition means of the train file minus the row-aligned train-control
  file; the scorer stops if any held-out compound appears in the training file (none did).
- **Synthetic check.** The scorer was run end to end on an invented environment (structure only, no real file read)
  before each real run; the mock's Step 7 record is computed with the same float32 path.
- Verdict rule as coded: category 1 `all(primary above) and all(co-primary not above)`; 2 `all(primary below) and
  all(co-primary above)`; 3 `all above / all above` or `all(primary below) and all(co-primary not above)`; 5
  `all include zero`; 4 otherwise. The coded reverse-TRADE clause admits "includes zero" on the co-primary where
  §6 says "below zero"; the observed pattern is not affected (the co-primary is above zero on two splits).

## 9. Limits and known defects

1. **Control-paste reach in the first record is a rounding artefact.** The control-paste cells were built in float64
   from float32 paired cells; `expm1` in the two precisions differs at the eighth digit, so a fold change that should
   be exactly 0 has a random sign, and the reach facet (which reads sign agreement) records 0.26 / 0.44 / 0.42 instead
   of 0. No significant call results (fidelity is 0.0000 exactly on every line) and no endpoint or contrast involves
   it. *Fixed in the second run (§11): the cells are cast to float32 and control-paste reach reads 0.000.*
2. **The shift construction is crude.** Adding a mean delta to sparse log-expression and clipping at zero sets
   30–44% of entries to zero; this is why the ridge shifts are worse than an unshifted control on energy. A shift
   that respected sparsity would be a different baseline and needs its own pre-registration.
3. **Mean-response is transductive**, as in the challenge; it bounds triviality and is not a predictor (§5).
4. **The scaled scores of the first run are superseded** (§11). The specification's half-split replicate is not a
   ceiling for DE facets under either computation.
5. **Sixth use of splits 0 and 2 and of the development split**; no model choice was made in this step.
6. **The within-held-out ridge** (ridge-S7) learns from the held-out compounds' own responses; ridge-train is the
   like-for-like comparator and both are reported.
7. **Category-4 wording** (§1) does not literally describe the observed pattern; the residual rule assigned it.
8. Nothing here is comparable to a challenge leaderboard entry (PREREG §4, §8).

## 10. What the thesis can say

On the sci-Plex held-out compounds, the flow-matching model identifies which genes respond better than any linear
shift of the control cells — the within-held-out fingerprint × line ridge, the like-for-like ridge fitted on the
training compounds, and a compound-blind mean response — on all three splits (DE-overlap 0.61 against 0.53–0.57).
On the shape of the predicted cell population it is closer to the real cells than the ridge shift on two splits,
farther on the small-effect split, indistinguishable from the like-for-like ridge on all three, not resolved
against pasting the unshifted control cells on splits 1 and 2, and worse than them on split 0. On the correlation of the
predicted delta with the measured one over the strongest genes (DPC-50), every generative arm is below not only
the ridges but a compound-blind shift, which shows that DPC-50 mostly rewards the line's shared response and that
the ridge's margin on this metric is not, in itself, compound-specific signal. The six VCC-style facets place every
submission near the no-information point on perturbation discrimination (PDS 0.49–0.59 against a replicate of
0.58–0.69), give the arms the higher direction fidelity and the shifts the higher reach, and show that in
77–94% of conditions the real cells are not distinguishable from their paired vehicle cells on any of the 200 genes.
The pre-registered category is PARTIAL, reported per split, and the pre-registered expectation was wrong on the
primary endpoint's direction.

*Dated correction, 2026-11-09.* "Every generative arm is below not only the ridges but a compound-blind shift"
holds for the condition mean of DPC-50 on every split. For the condition median it fails on split 2, where
cross-attention (0.0979) and matched control (0.0967) are above the ridge-train (0.0950), ridge-S7 (0.0816) and
mean-response (0.0858) medians; see the correction under §5.

---

## 11. Facets v2 — the six facets recomputed against the cell-eval2 source (Amendment 2; added 2026-09-24)

Amendment: `PREREG_step14_amendment2_facets_v2.md` (written after the first run's results were seen; it changes
nothing that Q1 gates). Scorer: `score_step14_facets_v2.py` with `step14_metrics_v2.py`, committed before the run
(scorer commit 3cde5a4). Record: `leakage_results/step14_facets_v2.json`, `leakage_results/step14_long_v2.csv`,
`leakage_results/step14_facets_v2.log`. Run time 980 s. Every number below is copied from that record.

**What changed and why.** An audit of the first run against the cell-eval2 0.16.0 source found five points of
the surrounding procedure that were not the challenge's and not forced by the data (Amendment 2 §2): the Wilcoxon
p-value carried a continuity correction (scanpy applies none), the NMAE replicate anchor was gated on the half
table instead of the full one, a replicate on the wrong side of the baseline was scored instead of refused, the 0
point was an additive log-space shift instead of the challenge's resampled-and-scaled emission of control cells,
and the overall was averaged over lines before facets. A sixth item was the control-paste dtype bug of §9. All six
are corrected in the second run; inputs, matching, gate, Step 7 reproduction and the four raw facet definitions
are unchanged, and the script stops unless the gate counts and the matching reproduce the first record.

**Cross-checks against the first record.** PDS and the expression-error ratio of every submission equal the
first record to 1e-9 (asserted). The DE facets moved by at most 1.1×10⁻⁴ (fidelity) and 3.5×10⁻⁴ (Jaccard) on
any line; reach by at most 0.048 (ridge-train, split 2; every other submission ≤ 0.015) apart from control-paste,
whose reach now reads 0.000 on every split (0.265 / 0.444 / 0.421 in the first record, the artefact of §9). NMAE moved by at most 0.003 on splits 1
and 0; on split 2 it moved by 0.06–0.12 on the K562 line for every submission, because the gate (≥ 10
reference-significant genes with finite fold change) admits a different set of K562 conditions once the
continuity correction is removed. The facet contrasts ridge-S7 minus 50 NFE are unchanged in verdict on every
split (PDS includes zero; fidelity below zero; Jaccard above zero; reach and NMAE undefined).

**The 0 point, per line.** The challenge's emission — the condition's paired vehicle cells resampled with
replacement and scaled gene-wise, in normalised-count space, by the line's mean-response profile over its control
profile — has these properties on this data: 124 / 113 / 107 of the 200 genes (A549 / K562 / MCF7) have a zero
control profile and are emitted as zeros; the median gene scale is 1.0–1.9; the largest scales (up to 136×) fall
on near-silent genes. A multiplicative rescaling of a sparse, heavy-tailed distribution moves ranks very little,
so the emission **calls no gene significant in any condition on any split**: fidelity 0.000 on every line,
significant-set Jaccard equal to control-paste's to four decimals (the fraction of gated conditions whose
reference set is empty), NMAE above 1 (its fold changes are the log of the scale, the wrong size on the gate
genes), reach 0.18–0.64 (its fold-change signs are largely right), PDS 0.46–0.51, expression-error ratio
1.00–1.70 (worse than pasting the control). Its profile, by contrast, is the strongest of any submission on the
thesis's gene-identity metric: DE-overlap 0.6651 / 0.6761 / 0.6698 (50 NFE: 0.6100 / 0.6124 / 0.6106), with
DPC-50 medians 0.0747 / 0.0712 / 0.0465. The additive shift of the first run is kept as an ordinary submission
("mean-response (additive)").

| split | line | replicate half: PDS / MSE / fid / reach / Jac / NMAE | 0 point: PDS / MSE / fid / reach / Jac / NMAE |
|---|---|---|---|
| 1 | A549 | 0.690 / 0.191 / 0.813 / 1.000 / 0.921 / 0.247 | 0.464 / 1.066 / 0.000 / 0.385 / 0.818 / 1.046 |
| 1 | K562 | 0.646 / 0.471 / 0.642 / 0.972 / 0.892 / 0.582 | 0.465 / 1.275 / 0.000 / 0.418 / 0.793 / 1.124 |
| 1 | MCF7 | 0.659 / 0.064 / 0.749 / 0.966 / 0.915 / 0.588 | 0.515 / 1.000 / 0.000 / 0.516 / 0.774 / 0.995 |
| 0 | A549 | 0.603 / 0.697 / 0.764 / 0.979 / 0.984 / 0.230 | 0.511 / 1.700 / 0.000 / 0.333 / 0.939 / 1.098 |
| 0 | K562 | 0.633 / 0.677 / 0.609 / 0.964 / 0.927 / 0.844 | 0.490 / 1.624 / 0.000 / 0.183 / 0.891 / 1.293 |
| 0 | MCF7 | 0.583 / 0.433 / 0.491 / 0.860 / 0.948 / 0.573 | 0.473 / 1.479 / 0.000 / 0.642 / 0.926 / 1.223 |
| 2 | A549 | 0.665 / 0.261 / 0.735 / 1.000 / 0.953 / 0.240 | 0.492 / 1.132 / 0.000 / 0.328 / 0.860 / 1.031 |
| 2 | K562 | 0.655 / 0.546 / 0.578 / 0.937 / 0.866 / 2.066 | 0.485 / 1.349 / 0.000 / 0.432 / 0.788 / 1.730 |
| 2 | MCF7 | 0.636 / 0.071 / 0.745 / 0.970 / 0.912 / 0.578 | 0.502 / 1.020 / 0.000 / 0.492 / 0.775 / 1.026 |

The NMAE anchor is now defined on every line (the first run had it undefined on K562 everywhere and on A549
on split 0). One scale is refused, exactly as cell-eval2 would refuse it: K562 on split 2, NMAE, where the
replicate half (2.066) is worse than the 0 point (1.730); the K562 overall on split 2 is therefore undefined and
the split's overall is the mean over A549 and MCF7.

**Raw facets, split means over lines** (the CSV holds every line and every submission)

| split | submission | PDS | MSE ratio | fidelity | reach | sig-Jaccard | lfc-NMAE |
|---|---|---|---|---|---|---|---|
| 1 | mean-response, challenge emission (0 point) | 0.4809 | 1.114 | 0.000 | 0.440 | 0.795 | 1.055 |
| 1 | control-paste | 0.5000 | 1.000 | 0.000 | 0.000 | 0.795 | 1.000 |
| 1 | mean-response (additive) | 0.5000 | 0.918 | 0.586 | 0.941 | 0.015 | 0.339 |
| 1 | ridge-S7 | 0.5544 | 0.873 | 0.587 | 0.820 | 0.016 | 0.304 |
| 1 | ridge-train | 0.5919 | 0.650 | 0.587 | 0.815 | 0.013 | 0.242 |
| 1 | 50 NFE | 0.5303 | 0.931 | 0.739 | 0.767 | 0.008 | 0.335 |
| 1 | guidance w = 2 | 0.5565 | 0.956 | 0.731 | 0.784 | 0.008 | 0.277 |
| 1 | cross-attention | 0.5379 | 0.844 | 0.746 | 0.769 | 0.008 | 0.325 |
| 1 | base | 0.5019 | 1.400 | — | — | — | — |
| 0 | mean-response, challenge emission (0 point) | 0.4911 | 1.601 | 0.000 | 0.386 | 0.919 | 1.205 |
| 0 | control-paste | 0.5000 | 1.000 | 0.000 | 0.000 | 0.919 | 1.000 |
| 0 | mean-response (additive) | 0.5000 | 0.945 | 0.581 | 0.922 | 0.003 | 0.508 |
| 0 | ridge-S7 | 0.5373 | 1.039 | 0.581 | 0.832 | 0.003 | 0.547 |
| 0 | ridge-train | 0.5365 | 1.927 | 0.552 | 0.748 | 0.002 | 0.382 |
| 0 | 50 NFE | 0.5300 | 1.608 | 0.738 | 0.628 | 0.001 | 0.402 |
| 0 | guidance w = 2 | 0.5430 | 2.141 | 0.733 | 0.583 | 0.001 | 0.413 |
| 0 | cross-attention | 0.5309 | 1.436 | 0.745 | 0.639 | 0.001 | 0.415 |
| 0 | base | 0.5059 | 4.280 | — | — | — | — |
| 2 | mean-response, challenge emission (0 point) | 0.4930 | 1.167 | 0.000 | 0.418 | 0.808 | 1.262 |
| 2 | control-paste | 0.5000 | 1.000 | 0.000 | 0.000 | 0.808 | 1.000 |
| 2 | mean-response (additive) | 0.5000 | 0.941 | 0.586 | 0.902 | 0.010 | 0.459 |
| 2 | ridge-S7 | 0.5010 | 0.890 | 0.581 | 0.756 | 0.011 | 0.382 |
| 2 | ridge-train | 0.5520 | 0.787 | 0.572 | 0.699 | 0.009 | 0.360 |
| 2 | 50 NFE | 0.5158 | 0.906 | 0.737 | 0.664 | 0.005 | 0.436 |
| 2 | guidance w = 2 | 0.5325 | 0.930 | 0.731 | 0.646 | 0.005 | 0.387 |
| 2 | cross-attention | 0.5246 | 0.836 | 0.741 | 0.681 | 0.005 | 0.422 |
| 2 | base | 0.5074 | 1.524 | — | — | — | — |

**Scaled facets, cell-eval2's procedure** (s = (u − b)/(r − b) per line with b the challenge emission and r the
replicate half; MSE clipped to [0, 1], NMAE floored at −6; a line's overall is the mean of its six; the split's
overall is the mean over the lines with all six defined)

| split | submission | PDS | MSE | fidelity | reach | sig-Jaccard | lfc-NMAE | overall (lines) |
|---|---|---|---|---|---|---|---|---|
| 1 | control-paste | +0.085 | +0.139 | +0.000 | −0.844 | +0.000 | +0.091 | −0.088 (A549, K562, MCF7) |
| 1 | mean-response (additive) | +0.085 | +0.236 | +0.803 | +0.940 | −7.003 | +1.298 | −0.607 (A549, K562, MCF7) |
| 1 | ridge-S7 | +0.401 | +0.287 | +0.804 | +0.698 | −6.997 | +1.373 | −0.572 (A549, K562, MCF7) |
| 1 | ridge-train | +0.615 | +0.537 | +0.803 | +0.688 | −7.025 | +1.484 | −0.483 (A549, K562, MCF7) |
| 1 | 50 NFE | +0.264 | +0.218 | +1.012 | +0.595 | −7.065 | +1.309 | −0.611 (A549, K562, MCF7) |
| 1 | guidance w = 2 | +0.410 | +0.183 | +1.002 | +0.621 | −7.065 | +1.415 | −0.572 (A549, K562, MCF7) |
| 1 | cross-attention | +0.305 | +0.314 | +1.021 | +0.598 | −7.066 | +1.328 | −0.583 (A549, K562, MCF7) |
| 1 | base | +0.107 | +0.012 | — | — | — | — | — |
| 0 | control-paste | +0.067 | +0.605 | +0.000 | −1.233 | +0.000 | +0.370 | −0.032 (A549, K562, MCF7) |
| 0 | mean-response (additive) | +0.067 | +0.661 | +0.967 | +1.126 | −28.978 | +1.111 | −4.174 (A549, K562, MCF7) |
| 0 | ridge-S7 | +0.404 | +0.568 | +0.968 | +0.872 | −28.977 | +1.087 | −4.179 (A549, K562, MCF7) |
| 0 | ridge-train | +0.415 | +0.149 | +0.924 | +0.648 | −28.994 | +1.275 | −4.264 (A549, K562, MCF7) |
| 0 | 50 NFE | +0.368 | +0.091 | +1.220 | +0.243 | −29.015 | +1.283 | −4.302 (A549, K562, MCF7) |
| 0 | guidance w = 2 | +0.476 | +0.000 | +1.212 | +0.037 | −29.015 | +1.263 | −4.338 (A549, K562, MCF7) |
| 0 | cross-attention | +0.381 | +0.168 | +1.232 | +0.269 | −29.014 | +1.262 | −4.284 (A549, K562, MCF7) |
| 0 | base | +0.163 | +0.000 | — | — | — | — | — |
| 2 | control-paste | +0.040 | +0.202 | +0.000 | −0.792 | +0.000 | +0.049 | −0.102 (A549, MCF7) |
| 2 | mean-response (additive) | +0.040 | +0.272 | +0.863 | +0.888 | −8.248 | +1.274 | −0.718 (A549, MCF7) |
| 2 | ridge-S7 | +0.041 | +0.326 | +0.852 | +0.582 | −8.241 | +1.492 | −0.699 (A549, MCF7) |
| 2 | ridge-train | +0.373 | +0.438 | +0.841 | +0.492 | −8.263 | +1.570 | −0.590 (A549, MCF7) |
| 2 | 50 NFE | +0.143 | +0.300 | +1.085 | +0.418 | −8.293 | +1.386 | −0.661 (A549, MCF7) |
| 2 | guidance w = 2 | +0.247 | +0.262 | +1.077 | +0.378 | −8.292 | +1.496 | −0.622 (A549, MCF7) |
| 2 | cross-attention | +0.195 | +0.378 | +1.091 | +0.453 | −8.294 | +1.408 | −0.620 (A549, MCF7) |
| 2 | base | +0.088 | +0.059 | — | — | — | — | — |

(The 0-point submission scores 0.000 on every facet by construction; the split-1 extras — Step 9 and Step 10
arms — are in the record and behave as the ladder arms do.)

**Reading.**

- *The overall is not a ranking on this data.* Every submission that calls any gene significant — every arm and
  every shift — sits 7 to 29 scale units below the 0 point on significant-set Jaccard, because the 0 point calls
  nothing, the reference sets are empty in 77–94% of conditions (control-paste Jaccard per line), and the Jaccard
  of a non-empty set against an empty one is 0. The mean of six is dominated by that term: control-paste has the
  highest overall on every split (−0.03 to −0.10) and every model and every shift sits at −0.5 to −0.8 (splits 1
  and 2) or −4.2 to −4.3 (split 0). Applied faithfully, the challenge's aggregate rewards calling nothing, which
  is a property of the data's empty reference sets, not of any submission.
- *Five facets read sensibly on this scale.* Direction fidelity: every generative arm at 1.0–1.2 (at or above
  the replicate half, whose fidelity is computed from half the cells), every shift at 0.80–0.97. PDS: ridge-train
  the highest on splits 1 and 2 (0.62 / 0.37), guidance w = 2 on split 0 (0.48); the arms 0.14–0.48; ridge-S7
  at the 0 point on split 2 (0.04). Expression error: control-paste scores 0.14–0.61 above the 0 point, because
  the emission's error ratio is 1.07–1.70 — the challenge's baseline predicts the condition means worse than
  pasting the control does on this data; the arms 0.0–0.38, ridge-train the highest on splits 1 and 2. Reach:
  the additive mean response the highest (0.89–1.13), the ridges 0.49–0.87, the arms 0.04–0.62, control-paste
  negative. NMAE: every calling submission 1.1–1.6, above the replicate half, because the 0 point's NMAE is above
  1 and the half-sized replicate's fold changes are noisy (0.23–2.07 raw).
- *Against the first run.* The raw facets are the same to three decimals except control-paste reach and K562
  NMAE on split 2; the scaled values differ because the 0 point does: the additive shift called many genes
  (fidelity 0.58, Jaccard 0.015, reach 0.92) and the challenge's emission calls none. Neither 0 point is the
  challenge's situation (raw counts, 18,533 genes, 400 cells, CRISPRi responses with large reference sets), and
  the difference between the two shows how much the scaled DE facets depend on how the baseline is emitted as
  cells on sparse data.
- *For the thesis.* The raw facets (§6 and this section) carry the information; the scaled PDS, expression
  error, fidelity and reach are interpretable with their 0 and 1 points stated; the scaled Jaccard, and therefore
  the overall, are not usable here, and the text says why rather than reporting a rank.

**Limits added by this run.** The emission is the challenge's construction transplanted to normalised-count
space on 200 genes; its gene scales are computed from the same paired vehicle cells it resamples. The K562
NMAE scale on split 2 is refused, so that split's overall covers two lines. The first run's scaled values are
kept in §6 as superseded, not deleted.
