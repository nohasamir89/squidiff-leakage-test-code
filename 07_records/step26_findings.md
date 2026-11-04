# Step 26 findings — where the published condition discrimination comes from

Date: 2026-11-04
Pre-registration: `PREREG_step26_control_source.md` (md5 `fb74d1ea9ba1bf0317380e91a6f7b9ae`), with the amendment of 2026-11-02 (ctrl_line as the S1 probe, G4 redefined, G4b and the two-split void rule) and the note of 2026-11-04 (Step 15c interval replay).
Scorer: `score_step26_tahoe.py` (md5 `a1ff350e26e46ddd9e435751f56b4aba`); launcher `run_step26_score.sh` (md5 `70e339c2d11300977f91d557fac33a46`).
Run: job 19070662, CPU, elapsed 00:23:47, peak RSS 7.68 GB; guards passed at research-repo commit `5628d4f`.
Records: `tahoe_results/step26_source.json`, `step26_source_per_condition.csv` (8,867 rows), `step26_source_summary.csv` (795 rows); log `tahoe_results/step26_logs/s26score_19070662.log`; research-repo commit `b490f3a`.

Every number below is copied from `step26_source_summary.csv` or from `s26score_19070662.log`. Rounded to four decimals; the CSV holds full precision. The `_sens` contrasts (flagged conditions excluded) are identical to the main contrasts because no condition was flagged on any split.

---

## 1. Question and design

Steps 24–25 showed that both the recipe and the base model discriminate held-out conditions (PDS ≈ 0.72 and 0.64, chance 0.5) equally well with the wrong compound or no compound. Step 26 asked where that discrimination comes from. Three candidate sources were named in advance:

- **S1** — the shared control estimate: predicted change B − C and measured change true − C both subtract the same noisy pool mean C, so they correlate even for an uninformative prediction (Nicol et al.).
- **S2** — condition-identifying state carried from the control cells the model is given as input.
- **S3** — compound signal masked by S1 noise.

Method: the committed predictions were re-scored, unchanged, under five scoring schemes that differ only in which control reference is subtracted (§3 of the prereg): **S-orig** (reproduction, C = the (line, plate) pool mean), **S-split** (primary: the pool is halved once by `RandomState(0)`; the prediction's change uses one half's mean, the truth's change the other's — the two changes share no control cells, so S1 is removed), **S-flip** (halves swapped), **S-linemean** (C replaced by the line's control mean pooled over plates) and **S-none** (no subtraction). Twelve arms: the six model arms of Step 25 (recipe/base × own, fpshuf, fpzero), two reconstruction arms, and four no-model arms — `ctrl_sample` (a resample of the condition's own pool; the input-copying null), `ctrl_sample_h1` (a resample of the prediction half only; the S-split validity null), `ctrl_line` (the line's pooled control mean used as the prediction; the pre-registered S1 probe) and `duplicate` (a second draw of the condition's own treated cells; the ceiling). Two pre-registered descriptive analyses: a plate-structure report, and PDS with candidates restricted to the same (line, dose, **plate**).

Intervals: structure-clustered percentile bootstrap, 5,000 replicates, 97.5 %, fresh `RandomState(0)` per scored set. Conditions: 2,899 / 3,115 / 2,853 on splits 1 / 0 / 2, 75 structures each, none flagged.

---

## 2. Guards, gates and reproduction (all passed)

- Pre-registration, scorer and launcher identical to HEAD; Step 24 and Step 25 records and findings committed (`5628d4f`).
- Pools: 180 / 179 / 180 (line, plate) pools per split, size min/median/max 104/8,500/11,000, 135/8,971/11,500 and 389/8,313/10,000 cells; 0 conditions flagged (pool < 20 or stored cells < 4).
- Rebuild gates: stored `true` and `C` reproduced from the test and control h5ad files to max deviation 3.24e-06 / 2.15e-05 (split 1), 3.75e-06 / 1.96e-05 (split 0), 4.44e-06 / 2.27e-05 (split 2), all under the 1e-4 tolerance; every stored real cell recovered bitwise.
- Reproduction: S-orig equals the committed Step 15b / 15c / 24 / 25 per-condition records to 1e-9 on every split; the Step 15c published intervals replayed exactly (recipe +0.22234 [+0.18829, +0.25411], base +0.14150 [+0.10358, +0.18063] on split 1; +0.21811 [+0.17822, +0.25899], +0.14441 [+0.10481, +0.18450] on split 0; +0.21420 [+0.18261, +0.24477], +0.15488 [+0.11017, +0.19767] on split 2).
- Validity nulls: G4b (`ctrl_sample`, S-orig) and G5 (`ctrl_sample_h1`, S-split) include zero on all three splits — the schemes behave as designed and no void condition fired.
- `FATAL` count 0; no `S26_TEST_REF` warning (the synthetic reference was not used).

---

## 3. Plate structure (pre-registered report, required before any verdict)

| split | (line, dose) contexts | distinct plates per context min/median/max | fraction of candidate pairs sharing the condition's plate |
|---|---|---|---|
| drug_split_1 (dev) | 42 | 3 / 5 / 5 | 0.2177 |
| drug_split_0 | 42 | 3 / 5 / 12 | 0.2137 |
| drug_split_2 | 42 | 3 / 5 / 5 | 0.2234 |

A condition's PDS candidates — other compounds in the same line and dose — sit on a median of five plates, and only about 22 % of candidate pairs were measured on the condition's own plate. Any prediction that carries the plate's signature therefore has a route to above-chance PDS that has nothing to do with the compound or the condition.

---

## 4. Pre-registered gates (PDS)
#### recipe
| contrast | split 1 (dev) | split 0 | split 2 |
|---|---|---|---|
| `G1_recipe_split_minus_0.5` | +0.2181 [+0.1844, +0.2499] above zero | +0.2132 [+0.1717, +0.2525] above zero | +0.2106 [+0.1802, +0.2401] above zero |
| `G1f_recipe_flip_minus_0.5` | +0.2172 [+0.1836, +0.2489] above zero | +0.2145 [+0.1730, +0.2530] above zero | +0.2089 [+0.1780, +0.2389] above zero |
| `G2_fpshuf_split_minus_0.5` | +0.2123 [+0.1828, +0.2417] above zero | +0.1968 [+0.1530, +0.2344] above zero | +0.1868 [+0.1519, +0.2210] above zero |
| `G2f_fpshuf_flip_minus_0.5` | +0.2112 [+0.1814, +0.2409] above zero | +0.1969 [+0.1531, +0.2344] above zero | +0.1851 [+0.1495, +0.2192] above zero |
| `G3_split_recipe_minus_fpshuf` | +0.0058 [-0.0156, +0.0280] includes zero | +0.0164 [-0.0048, +0.0400] includes zero | +0.0238 [-0.0028, +0.0531] includes zero |
| `G3f_flip_recipe_minus_fpshuf` | +0.0059 [-0.0157, +0.0282] includes zero | +0.0176 [-0.0035, +0.0416] includes zero | +0.0238 [-0.0031, +0.0531] includes zero |

#### nulls
| contrast | split 1 (dev) | split 0 | split 2 |
|---|---|---|---|
| `G4_ctrl_line_orig_minus_0.5` | +0.2513 [+0.2243, +0.2795] above zero | +0.2335 [+0.2017, +0.2617] above zero | +0.2321 [+0.2008, +0.2616] above zero |
| `G4b_ctrl_sample_orig_minus_0.5` | -0.0008 [-0.0112, +0.0092] includes zero | +0.0023 [-0.0098, +0.0154] includes zero | -0.0053 [-0.0196, +0.0092] includes zero |
| `G5_ctrl_h1_split_minus_0.5` | +0.0000 [-0.0107, +0.0113] includes zero | -0.0085 [-0.0209, +0.0031] includes zero | +0.0056 [-0.0061, +0.0163] includes zero |

#### base
| contrast | split 1 (dev) | split 0 | split 2 |
|---|---|---|---|
| `base_G1_split_minus_0.5` | +0.1392 [+0.1001, +0.1775] above zero | +0.1419 [+0.1034, +0.1825] above zero | +0.1528 [+0.1092, +0.1939] above zero |
| `base_G1f_flip_minus_0.5` | +0.1394 [+0.1005, +0.1773] above zero | +0.1421 [+0.1033, +0.1828] above zero | +0.1520 [+0.1078, +0.1937] above zero |
| `base_G2_split_minus_0.5` | +0.1402 [+0.1010, +0.1783] above zero | +0.1338 [+0.0940, +0.1749] above zero | +0.1421 [+0.0967, +0.1852] above zero |
| `base_G3_split_own_minus_fpshuf` | -0.0010 [-0.0087, +0.0068] includes zero | +0.0081 [-0.0037, +0.0222] includes zero | +0.0107 [-0.0010, +0.0250] includes zero |
| `base_G3f_flip_own_minus_fpshuf` | -0.0011 [-0.0089, +0.0069] includes zero | +0.0081 [-0.0038, +0.0223] includes zero | +0.0107 [-0.0011, +0.0248] includes zero |

#### E
| contrast | split 1 (dev) | split 0 | split 2 |
|---|---|---|---|
| `E_orig_recipe_minus_ctrl_line` | -0.0289 [-0.0538, -0.0047] below zero | -0.0154 [-0.0405, +0.0094] includes zero | -0.0179 [-0.0435, +0.0070] includes zero |
| `E_orig_recipe_minus_ctrl_sample` | +0.2231 [+0.1880, +0.2556] above zero | +0.2158 [+0.1741, +0.2548] above zero | +0.2195 [+0.1855, +0.2530] above zero |

### 4.1 Verdict D (pre-registered decision tree, §5)

Per split: G1 above zero | G2 above zero | G3 includes zero | G4 above zero | G4b includes zero | G5 includes zero — identical on all three splits. G1 and G3 agree under S-split and S-flip, as the 2026-11-02 rule requires.

**Verdict D: 2 — CONDITION-REAL, COMPOUND-FREE** (G1 above zero on all splits, G3 not above zero on any). By the tree's wording: discrimination survives unbiased (split-control) scoring and still does not need the compound. The recipe loses only 0.004 PDS from S-orig to S-split (0.7223 → 0.7181 on split 1); the base model likewise (0.6415 → 0.6392). **S1, the shared-estimate bias, is small here**: with pools of ~8,500 cells the pool mean is precise, and Nicol's effect, while present in the expected direction on every arm and on the ceiling (duplicate 0.9531 → 0.9512), is worth about 0.004 PDS.

G3 under split-control scoring is +0.0058 / +0.0164 / +0.0238, all including zero, with lower bounds −0.0156 / −0.0048 / −0.0028 — within 0.001 of Step 25's Q1 under S-orig. Removing the shared estimate did not tighten the own-compound question: there is still no resolved own-compound advantage on any split, and an advantage up to about +0.05 is still not excluded. The base model's G3 is −0.0010 / +0.0081 / +0.0107, all including zero.

### 4.2 Verdict E (baseline attribution)

| split | recipe S-orig PDS | ctrl_line S-orig PDS | ctrl_line share of the recipe effect | recipe − ctrl_line (paired) | ctrl_sample share |
|---|---|---|---|---|---|
| drug_split_1 (dev) | 0.7223 | 0.7513 | 113.0 % | −0.0289 [−0.0538, −0.0047] below zero | −0.3 % |
| drug_split_0 | 0.7181 | 0.7335 | 107.0 % | −0.0154 [−0.0405, +0.0094] includes zero | 1.1 % |
| drug_split_2 | 0.7142 | 0.7321 | 108.3 % | −0.0179 [−0.0435, +0.0070] includes zero | −2.5 % |

**A prediction with no model — the cell line's pooled control mean, blind to compound, dose and plate — scores higher PDS than the recipe on every split**, resolvably so on the development split. It reproduces 107–113 % of the recipe's published effect. The input-copying null (`ctrl_sample`) reproduces none of it, as the pool-average identity predicts.

### 4.3 Verdict F (calibration)

| split | duplicate ceiling S-orig | S-split | recipe dynamic-range fraction S-orig | S-split |
|---|---|---|---|---|
| drug_split_1 (dev) | 0.9531 | 0.9512 | 0.492 | 0.500 |
| drug_split_0 | 0.9622 | 0.9600 | 0.469 | 0.483 |
| drug_split_2 | 0.9530 | 0.9500 | 0.479 | 0.492 |

The recipe sits halfway between the copying null and a second draw of the real treated cells; the ceiling drops 0.002 under split-control scoring, the shared estimate's contribution to the ceiling itself.

---

## 5. The source, identified: plate identity

### 5.1 Why ctrl_line's score is not S1

The 2026-11-02 amendment introduced `ctrl_line` as the S1 probe on the premise that, being out-of-pool, it carries the shared estimate under S-orig and "under S-split it does not". The data contradict the premise: `ctrl_line` scores 0.7475 / 0.7284 / 0.7292 under S-split — 0.004 below its S-orig values, the same small drop as every other arm. Whatever makes the line mean discriminate conditions is not the shared control estimate.

What `ctrl_line` predicts, as a change, is D = (line mean) − C_plate: minus the plate's control offset from its line. It carries exactly one piece of information — which plate the condition was measured on. For it to win PDS against candidates on other plates, the measured change Y = true − C_plate must still carry that plate's signature; that is, subtracting the plate's own control mean does **not** remove plate structure from treated cells. Under S-linemean, where the reference is the line mean itself, `ctrl_line` is exactly 0.5000 on every split (its change is zero) and `ctrl_sample` — the plate's own control resample, whose change is now the plate offset — jumps to 0.7919 / 0.7890 / 0.7920. Plate offset alone, with no model and no treatment information, is as discriminative as anything in this study.

### 5.2 Within-plate candidates (pre-registered descriptive analysis)

Restricting each condition's candidates to compounds measured on the **same** (line, dose, plate), sets of ≥ 10 members:

| split | conditions covered | recipe PDS, S-orig | recipe PDS, S-split |
|---|---|---|---|
| drug_split_1 (dev) | 2,645 of 2,899 | 0.5025 | 0.5030 |
| drug_split_0 | 2,678 of 3,115 | 0.5177 | 0.5172 |
| drug_split_2 | 2,532 of 2,853 | 0.5270 | 0.5264 |

When the plate cannot be used to tell conditions apart, the recipe's PDS falls from about 0.72 to 0.50–0.53 on 87–91 % of conditions. The published discrimination is almost entirely plate recognition: the model is shown the plate's control cells as input, its output carries that plate's signature, and PDS rewards it because the competing candidates were measured on other plates. These within-plate values were pre-registered as descriptive — no intervals, recipe arm only — so "chance" is not claimed here; +0.018 and +0.027 on the confirmation splits may be a small real condition effect. That is what Step 27 must measure.

### 5.3 What the pre-registered label means after §5

Verdict D's label "CONDITION-REAL" is the tree's wording for "survives split-control scoring", and that is true. But the condition information that survives is, to the resolution of this step, **the plate**. The three named sources were S1, S2 and S3; the data point to S2 in a specific and deflating form — the plate signature carried from the control input — plus a plate × treatment interaction in the data that the standard control subtraction does not remove.

---

## 6. The other measures under the five schemes

Full tables in §9. The readings that matter:

- **DE-overlap**: `ctrl_line` (0.5028 / 0.4930 / 0.4942 under S-orig) is **above the recipe** (0.4640 / 0.4603 / 0.4585) and above the `duplicate` ceiling (0.4443 / 0.4415 / 0.4367). The genes that differ between a plate's controls and the line mean overlap the "true" DE genes (relative to the plate's controls) about as well as a second draw of the real treated cells does. DE-overlap, as published, is also largely a plate measure. The recipe-over-base gap on DE-overlap (~0.46 vs ~0.27–0.29) survives every scheme.
- **DPC-50**: `ctrl_line` 0.3815 / 0.3414 / 0.3679 versus recipe 0.5197 / 0.5005 / 0.5231, `duplicate` 0.7061 / 0.7218 / 0.7092, `ctrl_sample` ≈ 0. Here the recipe does carry direction beyond the plate offset (~+0.14 over `ctrl_line`). DPC-50 is not discriminative across conditions, so that extra direction may be a generic treated-versus-control shift shared across compounds; this step cannot tell.
- **MRA**: `ctrl_line` ≈ 0 (0.0439 / −0.0428 / −0.0093), `ctrl_sample` ≈ 0.08–0.09, recipe 0.3205 / 0.2864 / 0.3117, `duplicate` ≈ 0.92. Magnitude ranking is the one measure on which the recipe is clearly above every no-model arm — but `recipe_fpshuf` (0.3603 / 0.2138 / 0.3283) and `recipe_fpzero` (0.3553 / 0.2491 / 0.3261) match or exceed it, so the magnitude information is compound-free too. The base model is negative on MRA under every subtracting scheme.
- **S-none** (no subtraction): PDS ≈ 0.51–0.54 for every model arm, DE-overlap ≈ 0.82, DPC-50 ≈ 0.94 — raw profiles are dominated by the line, which every arm reproduces; the numbers are trivially high and carry no information about response.
- **S-linemean**: model arms fall to PDS 0.51–0.55 while `ctrl_sample` rises to 0.79. The models' outputs sit closer to the line mean than to their plate's control mean — they reproduce the plate signature only partly, which is why `ctrl_line` (the full plate offset, with the sign PDS rewards) beats them.

---

## 7. Audit against the pre-registered expectations (§6, as amended)

| expectation | outcome | held? |
|---|---|---|
| candidate sets span several plates (low confidence) | median 5 plates per context; 22 % same-plate pairs | yes |
| G4 (`ctrl_line`, S-orig) above zero to the extent S1 matters | above zero on all splits (+0.25 / +0.23 / +0.23) — but for a reason other than S1 (§5.1) | held as a contrast; the mechanism the expectation assumed is wrong |
| G4b includes zero everywhere (confident) | includes zero on all three | yes |
| G1: undecided between outcome 1 and 2 | outcome 2; drop S-orig → S-split only 0.004 | — |
| G2 tracks G1 | +0.2123 / +0.1968 / +0.1868 vs +0.2181 / +0.2132 / +0.2106 | yes |
| G3 includes zero on all splits | yes | yes |
| G5 includes zero everywhere (confident) | yes | yes |
| duplicate ceiling higher under S-orig than S-split (direction only) | 0.9531 > 0.9512, 0.9622 > 0.9600, 0.9530 > 0.9500 | yes |

One expectation was wrong in its mechanism, not its direction: the amendment's claim that `ctrl_line` carries S1 under S-orig and not under S-split. Recorded as written; the §5 reading is post-hoc interpretation of two pre-registered descriptive outputs (the plate report and the within-plate PDS) and is labelled as such.

---

## 8. Consequences for the earlier steps and the manuscript

1. **PDS on Tahoe-100M, with candidate sets that cross plates, measures plate identity.** Every PDS reported in Steps 15–25 (recipe ≈ 0.72, base ≈ 0.64, the fingerprint-swap arms, the Step 15c submission ladder) is to be read with that in mind. The recipe-over-base gap of ~0.08 is how much more faithfully the recipe reproduces its input plate's control state — a realism property, consistent with the OT result — not a difference in drug-response prediction.
2. **DE-overlap is also plate-dominated**; `ctrl_line` beats the models and the ceiling on it.
3. **Compound information: none resolved** (Steps 24, 25, 26 G3). **Condition information beyond plate: at most ~0.03 PDS**, unmeasured with intervals (Step 27).
4. **Nicol's shared-estimate bias is present but small** (~0.004 PDS) at these pool sizes. The manuscript should report it as measured, not as the explanation.
5. A no-model baseline — the line's control mean — must appear in every PDS and DE-overlap table from here on. Any published model score is interpretable only relative to it and to the within-plate restriction.
6. The Results and Discussion written around "condition discrimination" (Steps 15c, 24, 25) need rewriting; the abstract's claim set changes from "compound-free but condition-specific" to "plate recognition under a cross-plate candidate design, with no resolved compound and little resolved within-plate condition signal".

---

## 9. Full tables (from `step26_source_summary.csv`)
#### PDS (chance 0.5) — split 1 (dev)
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.7223 | 0.7181 | 0.7172 | 0.5290 | 0.5259 |
| recipe_fpshuf | 0.7160 | 0.7123 | 0.7112 | 0.5314 | 0.5371 |
| recipe_fpzero | 0.7360 | 0.7316 | 0.7310 | 0.5399 | 0.5246 |
| base | 0.6415 | 0.6392 | 0.6394 | 0.5103 | 0.5065 |
| base_fpshuf | 0.6426 | 0.6402 | 0.6405 | 0.5163 | 0.5113 |
| base_fpzero | 0.6431 | 0.6408 | 0.6410 | 0.5131 | 0.5065 |
| recon_recipe | 0.7368 | 0.7329 | 0.7315 | 0.5361 | 0.5297 |
| recon_base | 0.6564 | 0.6540 | 0.6544 | 0.5150 | 0.5090 |
| ctrl_sample | 0.4992 | 0.4847 | 0.4765 | 0.7919 | 0.7500 |
| ctrl_sample_h1 | 0.4964 | 0.5000 | 0.4548 | 0.7894 | 0.7485 |
| ctrl_line | 0.7513 | 0.7475 | 0.7463 | 0.5000 | 0.5001 |
| duplicate | 0.9531 | 0.9512 | 0.9506 | 0.9567 | 0.9631 |

#### PDS (chance 0.5) — split 0
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.7181 | 0.7132 | 0.7145 | 0.5505 | 0.5332 |
| recipe_fpshuf | 0.7012 | 0.6968 | 0.6969 | 0.5387 | 0.5252 |
| recipe_fpzero | 0.7180 | 0.7132 | 0.7140 | 0.5311 | 0.5218 |
| base | 0.6444 | 0.6419 | 0.6421 | 0.5275 | 0.5144 |
| base_fpshuf | 0.6360 | 0.6338 | 0.6340 | 0.5163 | 0.5119 |
| base_fpzero | 0.6366 | 0.6342 | 0.6348 | 0.5168 | 0.5095 |
| recon_recipe | 0.7304 | 0.7255 | 0.7265 | 0.5540 | 0.5382 |
| recon_base | 0.6591 | 0.6567 | 0.6570 | 0.5317 | 0.5178 |
| ctrl_sample | 0.5023 | 0.4829 | 0.4857 | 0.7890 | 0.7372 |
| ctrl_sample_h1 | 0.4950 | 0.4915 | 0.4625 | 0.7888 | 0.7397 |
| ctrl_line | 0.7335 | 0.7284 | 0.7294 | 0.5000 | 0.5013 |
| duplicate | 0.9622 | 0.9600 | 0.9601 | 0.9678 | 0.9706 |

#### PDS (chance 0.5) — split 2
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.7142 | 0.7106 | 0.7089 | 0.5284 | 0.5272 |
| recipe_fpshuf | 0.6907 | 0.6868 | 0.6851 | 0.5286 | 0.5205 |
| recipe_fpzero | 0.7144 | 0.7108 | 0.7087 | 0.5387 | 0.5286 |
| base | 0.6549 | 0.6528 | 0.6520 | 0.5294 | 0.5159 |
| base_fpshuf | 0.6441 | 0.6421 | 0.6413 | 0.5222 | 0.5124 |
| base_fpzero | 0.6460 | 0.6437 | 0.6428 | 0.5185 | 0.5111 |
| recon_recipe | 0.7262 | 0.7225 | 0.7209 | 0.5292 | 0.5335 |
| recon_base | 0.6688 | 0.6664 | 0.6660 | 0.5340 | 0.5180 |
| ctrl_sample | 0.4947 | 0.4785 | 0.4732 | 0.7920 | 0.7494 |
| ctrl_sample_h1 | 0.5033 | 0.5056 | 0.4641 | 0.7915 | 0.7512 |
| ctrl_line | 0.7321 | 0.7292 | 0.7259 | 0.5000 | 0.5000 |
| duplicate | 0.9530 | 0.9500 | 0.9504 | 0.9588 | 0.9652 |

#### DE-overlap — split 1 (dev)
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.4640 | 0.4615 | 0.4616 | 0.4566 | 0.8188 |
| recipe_fpshuf | 0.4623 | 0.4600 | 0.4599 | 0.4563 | 0.8207 |
| recipe_fpzero | 0.4564 | 0.4539 | 0.4542 | 0.4285 | 0.8240 |
| base | 0.2683 | 0.2684 | 0.2687 | 0.2518 | 0.5954 |
| base_fpshuf | 0.2680 | 0.2682 | 0.2681 | 0.2521 | 0.5957 |
| base_fpzero | 0.2672 | 0.2673 | 0.2669 | 0.2492 | 0.5981 |
| recon_recipe | 0.4655 | 0.4630 | 0.4628 | 0.4572 | 0.8170 |
| recon_base | 0.2710 | 0.2711 | 0.2710 | 0.2532 | 0.5935 |
| ctrl_sample | 0.3843 | 0.3857 | 0.3855 | 0.4881 | 0.8282 |
| ctrl_sample_h1 | 0.3859 | 0.3857 | 0.3851 | 0.4890 | 0.8278 |
| ctrl_line | 0.5028 | 0.4982 | 0.4972 | n/a | 0.8529 |
| duplicate | 0.4443 | 0.4423 | 0.4412 | 0.4720 | 0.7990 |

#### DE-overlap — split 0
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.4603 | 0.4578 | 0.4577 | 0.4511 | 0.8182 |
| recipe_fpshuf | 0.4545 | 0.4524 | 0.4521 | 0.4496 | 0.8149 |
| recipe_fpzero | 0.4534 | 0.4510 | 0.4503 | 0.4336 | 0.8200 |
| base | 0.2850 | 0.2849 | 0.2848 | 0.2683 | 0.6078 |
| base_fpshuf | 0.2849 | 0.2845 | 0.2845 | 0.2684 | 0.6070 |
| base_fpzero | 0.2831 | 0.2828 | 0.2828 | 0.2635 | 0.6094 |
| recon_recipe | 0.4608 | 0.4580 | 0.4580 | 0.4517 | 0.8165 |
| recon_base | 0.2874 | 0.2870 | 0.2869 | 0.2697 | 0.6060 |
| ctrl_sample | 0.3820 | 0.3820 | 0.3815 | 0.4890 | 0.8263 |
| ctrl_sample_h1 | 0.3817 | 0.3813 | 0.3813 | 0.4892 | 0.8271 |
| ctrl_line | 0.4930 | 0.4887 | 0.4879 | n/a | 0.8480 |
| duplicate | 0.4415 | 0.4396 | 0.4401 | 0.4760 | 0.8014 |

#### DE-overlap — split 2
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.4585 | 0.4556 | 0.4561 | 0.4497 | 0.8205 |
| recipe_fpshuf | 0.4560 | 0.4530 | 0.4538 | 0.4534 | 0.8236 |
| recipe_fpzero | 0.4525 | 0.4499 | 0.4505 | 0.4349 | 0.8267 |
| base | 0.2908 | 0.2904 | 0.2906 | 0.2723 | 0.6178 |
| base_fpshuf | 0.2897 | 0.2891 | 0.2898 | 0.2723 | 0.6187 |
| base_fpzero | 0.2881 | 0.2878 | 0.2883 | 0.2677 | 0.6202 |
| recon_recipe | 0.4606 | 0.4579 | 0.4579 | 0.4504 | 0.8190 |
| recon_base | 0.2938 | 0.2933 | 0.2934 | 0.2747 | 0.6154 |
| ctrl_sample | 0.3812 | 0.3824 | 0.3812 | 0.4863 | 0.8313 |
| ctrl_sample_h1 | 0.3816 | 0.3811 | 0.3816 | 0.4855 | 0.8302 |
| ctrl_line | 0.4942 | 0.4875 | 0.4904 | n/a | 0.8552 |
| duplicate | 0.4367 | 0.4340 | 0.4344 | 0.4653 | 0.8001 |

#### DPC-50 — split 1 (dev)
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.5197 | 0.5118 | 0.5121 | 0.3822 | 0.9390 |
| recipe_fpshuf | 0.5229 | 0.5153 | 0.5154 | 0.3972 | 0.9405 |
| recipe_fpzero | 0.5288 | 0.5197 | 0.5222 | 0.4241 | 0.9480 |
| base | 0.2269 | 0.2239 | 0.2230 | 0.1194 | 0.7323 |
| base_fpshuf | 0.2269 | 0.2239 | 0.2231 | 0.1235 | 0.7325 |
| base_fpzero | 0.2193 | 0.2162 | 0.2157 | 0.1122 | 0.7381 |
| recon_recipe | 0.5368 | 0.5292 | 0.5291 | 0.3906 | 0.9382 |
| recon_base | 0.2409 | 0.2379 | 0.2369 | 0.1257 | 0.7276 |
| ctrl_sample | -0.0010 | -0.0110 | -0.0185 | 0.4214 | 0.9501 |
| ctrl_sample_h1 | -0.0049 | -0.0015 | -0.0348 | 0.4198 | 0.9498 |
| ctrl_line | 0.3815 | 0.3722 | 0.3716 | n/a | 0.9485 |
| duplicate | 0.7061 | 0.6994 | 0.6989 | 0.7523 | 0.9682 |

#### DPC-50 — split 0
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.5005 | 0.4936 | 0.4943 | 0.3946 | 0.9314 |
| recipe_fpshuf | 0.4850 | 0.4782 | 0.4790 | 0.3707 | 0.9245 |
| recipe_fpzero | 0.4868 | 0.4794 | 0.4805 | 0.3830 | 0.9332 |
| base | 0.2490 | 0.2456 | 0.2468 | 0.1500 | 0.7688 |
| base_fpshuf | 0.2409 | 0.2374 | 0.2389 | 0.1366 | 0.7661 |
| base_fpzero | 0.2373 | 0.2339 | 0.2352 | 0.1308 | 0.7747 |
| recon_recipe | 0.5154 | 0.5087 | 0.5094 | 0.3998 | 0.9309 |
| recon_base | 0.2629 | 0.2594 | 0.2607 | 0.1559 | 0.7651 |
| ctrl_sample | 0.0002 | -0.0164 | -0.0087 | 0.4091 | 0.9392 |
| ctrl_sample_h1 | 0.0036 | -0.0007 | -0.0174 | 0.4133 | 0.9392 |
| ctrl_line | 0.3414 | 0.3323 | 0.3339 | n/a | 0.9365 |
| duplicate | 0.7218 | 0.7156 | 0.7156 | 0.7700 | 0.9686 |

#### DPC-50 — split 2
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.5231 | 0.5167 | 0.5147 | 0.3878 | 0.9376 |
| recipe_fpshuf | 0.5102 | 0.5031 | 0.5020 | 0.4006 | 0.9389 |
| recipe_fpzero | 0.5226 | 0.5161 | 0.5130 | 0.4219 | 0.9466 |
| base | 0.2705 | 0.2673 | 0.2675 | 0.1604 | 0.7579 |
| base_fpshuf | 0.2629 | 0.2595 | 0.2600 | 0.1598 | 0.7586 |
| base_fpzero | 0.2584 | 0.2551 | 0.2553 | 0.1481 | 0.7643 |
| recon_recipe | 0.5366 | 0.5305 | 0.5282 | 0.3897 | 0.9361 |
| recon_base | 0.2841 | 0.2809 | 0.2810 | 0.1660 | 0.7526 |
| ctrl_sample | -0.0047 | -0.0198 | -0.0179 | 0.4040 | 0.9497 |
| ctrl_sample_h1 | 0.0091 | 0.0094 | -0.0189 | 0.4081 | 0.9504 |
| ctrl_line | 0.3679 | 0.3582 | 0.3577 | n/a | 0.9487 |
| duplicate | 0.7092 | 0.7021 | 0.7020 | 0.7523 | 0.9676 |

#### MRA — split 1 (dev)
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.3205 | 0.3164 | 0.3171 | 0.0574 | 0.0295 |
| recipe_fpshuf | 0.3603 | 0.3561 | 0.3589 | -0.0200 | 0.1002 |
| recipe_fpzero | 0.3553 | 0.3525 | 0.3508 | 0.1301 | 0.3285 |
| base | -0.0755 | -0.0735 | -0.0779 | 0.0360 | 0.0245 |
| base_fpshuf | -0.0745 | -0.0738 | -0.0755 | 0.0344 | 0.0555 |
| base_fpzero | -0.1356 | -0.1328 | -0.1377 | 0.0117 | 0.1006 |
| recon_recipe | 0.3937 | 0.3902 | 0.3904 | 0.1235 | 0.0889 |
| recon_base | -0.0346 | -0.0333 | -0.0376 | 0.0514 | 0.0619 |
| ctrl_sample | 0.0914 | 0.0948 | 0.0938 | 0.2018 | 0.5652 |
| ctrl_sample_h1 | 0.0899 | 0.0867 | 0.1026 | 0.1865 | 0.5500 |
| ctrl_line | 0.0439 | 0.0318 | 0.0337 | 0.0000 | 0.0000 |
| duplicate | 0.9191 | 0.9129 | 0.9150 | 0.8856 | 0.9276 |

#### MRA — split 0
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.2864 | 0.2847 | 0.2835 | 0.0595 | 0.1840 |
| recipe_fpshuf | 0.2138 | 0.2130 | 0.2117 | 0.0337 | 0.1819 |
| recipe_fpzero | 0.2491 | 0.2479 | 0.2478 | 0.1228 | 0.3365 |
| base | -0.0576 | -0.0573 | -0.0576 | 0.0185 | 0.1489 |
| base_fpshuf | -0.0538 | -0.0528 | -0.0523 | 0.0302 | 0.1023 |
| base_fpzero | -0.1138 | -0.1127 | -0.1135 | -0.0091 | 0.1040 |
| recon_recipe | 0.3356 | 0.3346 | 0.3328 | 0.1071 | 0.2311 |
| recon_base | -0.0021 | -0.0017 | -0.0017 | 0.0539 | 0.1794 |
| ctrl_sample | 0.0789 | 0.0770 | 0.0813 | 0.1819 | 0.5674 |
| ctrl_sample_h1 | 0.0869 | 0.0858 | 0.0838 | 0.1631 | 0.5652 |
| ctrl_line | -0.0428 | -0.0405 | -0.0380 | 0.0000 | 0.0000 |
| duplicate | 0.9207 | 0.9178 | 0.9150 | 0.8933 | 0.9441 |

#### MRA — split 2
| arm | S-orig | S-split | S-flip | S-linemean | S-none |
|---|---|---|---|---|---|
| recipe | 0.3117 | 0.3159 | 0.3060 | 0.0882 | 0.1876 |
| recipe_fpshuf | 0.3283 | 0.3331 | 0.3239 | 0.0229 | 0.1323 |
| recipe_fpzero | 0.3261 | 0.3302 | 0.3209 | 0.0934 | 0.3392 |
| base | -0.1069 | -0.1063 | -0.1064 | 0.0226 | 0.1413 |
| base_fpshuf | -0.0716 | -0.0702 | -0.0722 | 0.0231 | 0.0891 |
| base_fpzero | -0.1169 | -0.1165 | -0.1167 | 0.0187 | 0.1103 |
| recon_recipe | 0.3903 | 0.3959 | 0.3827 | 0.1465 | 0.2188 |
| recon_base | -0.0473 | -0.0463 | -0.0476 | 0.0477 | 0.1666 |
| ctrl_sample | 0.0894 | 0.0977 | 0.0856 | 0.1589 | 0.5827 |
| ctrl_sample_h1 | 0.1042 | 0.0978 | 0.1136 | 0.1771 | 0.5914 |
| ctrl_line | -0.0093 | 0.0018 | -0.0215 | 0.0000 | 0.0000 |
| duplicate | 0.9154 | 0.9111 | 0.9079 | 0.8820 | 0.9320 |

---

## 10. Limits

- The within-plate PDS has no interval and was computed for the recipe only; `ctrl_line`, base and the swap arms were not restricted. Step 27 pre-registers the full within-plate analysis (all arms, intervals, DE-overlap and DPC-50 within plate, and a plate-blind reference).
- The plate × treatment interaction inferred in §5.1 is an inference from scores, not a direct measurement; Step 27 should measure it directly (e.g. the correlation of true − C_plate with the plate offset across conditions).
- Energy distance was not recomputed under any scheme (it compares cell populations and never touches C); the committed values stand.
- The pool halving uses one draw (`RandomState(0)`) per pool; S-flip shows the assignment does not matter at the reported precision.

## 11. Next step

Step 27 pre-registration (within-plate scoring, all arms, with intervals), then rewriting of Results 3, the abstract and the Discussion.
