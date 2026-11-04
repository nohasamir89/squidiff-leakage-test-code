# Step 27 findings — within-plate scoring: no resolved condition signal beyond plate; the plate residual measured

Date: 2026-11-05
Pre-registration: `PREREG_step27_within_plate.md` (md5 `bf20110283f6cca827d4148c54d5540e`), committed `a9f37f4` before the run; no amendments.
Scorer: `score_step27_tahoe.py` (md5 `fc021f05eb647857be5c8b0ca6d33111`); launcher `run_step27_score.sh` (md5 `d662574310f9de353b95f389a46c826d`).
Run: job 19072136, CPU, elapsed 00:10:56, peak RSS 8.71 GB; guards passed at commit `a9f37f4`.
Records: `tahoe_results/step27_within_plate.json`, `step27_within_plate_per_condition.csv` (8,867 rows), `step27_within_plate_summary.csv` (255 rows); log `tahoe_results/step27_logs/s27score_19072136.log`; research-repo commit `bcde730`.

Every number below is copied from `s27score_19072136.log` (the committed record; the summary CSV holds the same values at full precision). Contrasts are rounded to four decimals.

---

## 1. Question

Step 26 found that a no-model, plate-only prediction (the line's pooled control mean) reproduces the recipe's published PDS, that this is not the shared-estimate bias, and — descriptively, recipe arm only — that restricting PDS candidates to the condition's own plate drops the recipe to 0.50–0.53. Step 27 put intervals on that, extended it to every arm, and tested the mechanism directly: does the measured change of treated cells (true − C_plate) point along the plate's control offset (L − C_plate)?

Design (prereg §3–4): the twelve Step 26 arms, rebuilt with the Step 26 seeds; candidates restricted to the same (line, dose, plate), sets ≥ 10; S-orig, S-split, S-flip; structure-clustered percentile bootstrap, 5,000 reps, 97.5 %; a split "shows above zero" only under both halvings. The plate residual is the per-condition cosine between a change vector and the plate's control offset, primary version built from disjoint halves (change relative to C2, offset L − C1), null = a resample from h2 (`ctrl_sample_h2`, seed 4000000 + i).

---

## 2. Guards, gates and reproduction (all passed)

- Pre-registration, scorer, launcher and every record read committed and clean at `a9f37f4`; Step 26 records and findings committed first.
- Rebuild gates: stored `true` and `C` reproduced from the data files to 3.24e-06 / 2.15e-05 (split 1), 3.75e-06 / 1.96e-05 (split 0), 4.44e-06 / 2.27e-05 (split 2); all stored real cells recovered bitwise; 180 / 179 / 180 pools; 0 conditions flagged.
- **Reproduction: every arm's cross-plate PDS under S-orig, S-split and S-flip equals the committed Step 26 per-condition record to 1e-9 on every split** — the pools, the halving, the no-model draws and the duplicate ceiling were rebuilt bitwise. The Step 15c published intervals replayed exactly.
- Within-plate coverage equals Step 26's: 2,645 of 2,899 (split 1), 2,678 of 3,115 (split 0), 2,532 of 2,853 (split 2); 162 / 149 / 149 (line, dose, plate) sets; recipe within-plate values reproduce Step 26's (0.5025 / 0.5030, 0.5177 / 0.5172, 0.5270 / 0.5264).
- Validity nulls: H4 (`ctrl_line` within plate) +0.0000 / +0.0000 / −0.0001, intervals including zero — chance by construction, as pre-registered; H4b (`ctrl_sample_h1`, S-split) and the H5 null (`ctrl_sample_h2` cosine) include zero on every split. `FATAL` count 0; no synthetic-reference warning.

---

## 3. Pre-registered verdicts

**Verdict G: 1 — NO RESOLVED SIGNAL BEYOND PLATE.** H1 (recipe within-plate PDS − 0.5, S-split): +0.0030 [−0.0409, +0.0476], +0.0172 [−0.0394, +0.0713], +0.0264 [−0.0213, +0.0762]; the same under S-flip and S-orig. Once the plate cannot tell conditions apart, the recipe's discrimination is not resolved from chance on any split. The intervals are wide (half-widths ≈ 0.045–0.055; within-plate sets are small and the per-structure values vary), so the bound is "at most about +0.07–0.08", not "zero".

**Verdict H: SUPPORTED.** The measured change points along the plate's control offset on every split: cosine +0.3642 [+0.3180, +0.4105], +0.3195 [+0.2787, +0.3575], +0.3381 [+0.2911, +0.3811] (S-split, disjoint halves); the h2 resample null is −0.0029 / +0.0005 / +0.0019, all including zero. Treated cells, after subtracting their own plate's control mean, still carry about a third of a unit of plate direction: subtracting the plate's control mean does not remove plate structure from treated cells. This is the direct measurement of the route inferred in `step26_findings.md` §5.1.

**Verdict I (gap attribution):** within plate, recipe − base (S-split) is +0.0096 / +0.0094 / +0.0065, all including zero — 11.8 % / 12.7 % / 10.9 % of the published cross-plate gap (+0.0808 / +0.0737 / +0.0593). About 88 % of the recipe's PDS advantage over the base model was plate.

Per split: H1 includes zero/includes zero | H2 includes zero | H3 includes zero/includes zero (split 1), includes zero/above zero (split 0), above zero/above zero (split 2) | H4 includes zero | H4b includes zero | H5 above zero (null includes zero).

---

## 4. Three readings the tables force

### 4.1 The within-plate task is solvable from the data; the models do not solve it
The `duplicate` ceiling — a second draw of each condition's own treated cells — scores 0.8846 / 0.9111 / 0.8936 within plate (duplicate − 0.5: +0.3836 [+0.3665, +0.4002], +0.4095 [+0.3913, +0.4263], +0.3911 [+0.3721, +0.4094]). Real conditions on the same plate differ enough to be told apart nine times in ten from ~100–200 cells. The recipe scores 0.50–0.53, the base 0.49–0.52, and the reconstruction arms — the encoder shown the treated cells themselves — only 0.5236–0.5387 (recipe) and 0.5050–0.5302 (base). Even given the answer as input, the models' outputs barely distinguish same-plate conditions.

### 4.2 Why the recipe beat the base model under cross-plate scoring
The predicted-residual rows explain the published gap. The cosine of each arm's output change with the plate offset (S-split): recipe 0.6277 / 0.6352 / 0.6235; `recipe_fpzero` 0.7224 / 0.7122 / 0.7006; base 0.2508 / 0.2589 / 0.2636; real treated cells 0.3642 / 0.3195 / 0.3381. The recipe's output is **more** aligned with the plate's control offset than real treated cells are — it over-shrinks toward the line mean — and the base less. Cross-plate PDS rewards alignment with the plate offset (§3, verdict H), so the model that shrinks harder scores higher. `ctrl_line`, pure offset (cosine 1.0000 / 0.98 under S-split), scores highest of all (Step 26). The recipe's ~0.08 advantage is the amount by which it reproduces its input plate's control signature better than the base does; it is not better drug-response prediction.

### 4.3 The compound within plate: a wrong one hurts, the right one adds little
`recipe_fpshuf` sits below chance on all three splits within plate (0.4886 / 0.4906 / 0.4859, S-split) while `recipe_fpzero` is at chance (0.4969 / 0.4983 / 0.4972; fpzero − 0.5 includes zero on every split). H3 (own − wrong compound, S-split): +0.0144 [−0.0157, +0.0425], +0.0266 [−0.0006, +0.0582], +0.0405 [+0.0053, +0.0820] — above zero under both halvings on split 2 only (split 0: above zero under S-flip, not under S-split; by the pre-registered rule one split "shows"). The base model's analogue is above zero on split 2 as well (+0.0206 [+0.0051, +0.0384], both halvings). Verdict G's tree does not reach outcome 3 because H1 never resolved — the correct reading — and H3 is reported as it is: the models use the compound input, but mostly in the sense that a wrong compound misleads them; the right one is worth at most a few hundredths of PDS, unresolved on two of three splits. Steps 24–26 could not see this under cross-plate scoring because the plate dominated the ranking.

---

## 5. Audit against the pre-registered expectations (§5)

| expectation | outcome | held? |
|---|---|---|
| H1 includes zero on split 1 (confident); splits 0 and 2 undecided, leaning includes zero (low confidence) | includes zero on all three | yes |
| H2 small, including zero (low confidence) | +0.0096 / +0.0094 / +0.0065, all including zero | yes |
| H3 includes zero on all splits (moderately confident) | above zero on split 2 (both halvings); split 0 under one halving only | **no** on split 2 — recorded as written |
| H4 ≈ 0.5 with interval including zero | +0.0000 / +0.0000 / −0.0001 | yes |
| H4b and the h2 null include zero | yes, all splits | yes |
| H5 above zero on all splits with null including zero (moderately confident) | yes | yes |
| predicted residual: ctrl_line = 1, models between 0 and the true value, ctrl_sample ≈ 0 (low confidence on the models) | ctrl_line 1.0000; ctrl_sample ≈ 0; **the recipe (0.63) and fpzero (0.72) exceed the true value (0.32–0.36); the base (0.25) is below it** | partly: the models' position was wrong — they are not between 0 and the truth, they straddle it |
| duplicate within plate well above 0.5 but below its cross-plate 0.95 | 0.88–0.91 vs 0.95 | yes |

Two expectations failed in an informative direction: the compound is not inert within plate (one split resolves own > wrong), and the recipe's output is more plate-aligned than real cells, not less. Neither changes a verdict; both go in the manuscript.

---

## 6. What Steps 24–27 together establish

1. Under cross-plate candidate sets, PDS on Tahoe-100M measures plate recognition (Step 26) through a measured route: treated cells keep about a third of their plate's control-offset direction after control subtraction (Step 27, verdict H), so any prediction aligned with that offset scores above chance against candidates from other plates.
2. The recipe's published advantage over the base model (~0.08 PDS) is ~88 % that alignment (verdict I); the recipe over-shrinks toward the line mean more than the base does (§4.2).
3. Within plate, where the route is closed: no resolved condition discrimination for either model (H1 includes zero, upper bounds ≈ +0.05 to +0.08) against a data ceiling of 0.88–0.91 (§4.1).
4. The compound fingerprint: no resolved own-compound advantage cross-plate (Steps 24–26); within plate, a wrong compound lowers PDS below chance on every split and own > wrong resolves on one split (§4.3).
5. The shared-estimate bias (Nicol) is real and small here (~0.004 PDS; Step 26).

Manuscript consequences, adding to `step26_findings.md` §8: every PDS/DE-overlap table gets the `ctrl_line` row and a within-plate column; the Results sentence on the recipe-over-base gap becomes an attribution (88 % plate alignment, 11–13 % surviving within plate, unresolved); the Discussion gains the over-shrinkage mechanism (§4.2) as the explanation of why a realism-improving recipe scored higher on a discrimination metric; the `duplicate` within-plate ceiling is the honest statement of how far any model is from the data.

---

## 7. Limits

- Within-plate intervals are wide (half-width ≈ 0.045–0.055 for the model arms): "includes zero" bounds the signal at roughly +0.08 at most, it does not show zero. The nulls' intervals are narrow (≈ 0.012) because their per-structure values do not vary; the models' do.
- H3's split-2 result is one split under the pre-registered rule; it is reported, not generalised.
- The residual cosine varies mostly between pools and its interval does not resample pools (prereg §8); 179–180 pools per split and three splits agreeing make a pool artefact unlikely but not excluded.
- Within-plate scoring drops 9–13 % of conditions (sets < 10) and removes plate only; other structure shared within a (line, dose, plate) set is not addressed.
- DE-overlap and DPC-50 have no within-plate form; Step 26's comparisons stand for them.
- Nothing here says what a differently trained or differently scored model could do; the statements are about the committed predictions.

---

## 8. Full tables (from the scoring log)

#### CONTRASTS
| contrast | split 1 (dev) | split 0 | split 2 |
|---|---|---|---|
| `H1_recipe_split_minus_0.5` | +0.0030 [-0.0409, +0.0476] includes zero | +0.0172 [-0.0394, +0.0713] includes zero | +0.0264 [-0.0213, +0.0762] includes zero |
| `H1f_recipe_flip_minus_0.5` | +0.0025 [-0.0412, +0.0470] includes zero | +0.0192 [-0.0369, +0.0732] includes zero | +0.0271 [-0.0199, +0.0758] includes zero |
| `H1o_recipe_orig_minus_0.5` | +0.0025 [-0.0415, +0.0470] includes zero | +0.0177 [-0.0387, +0.0715] includes zero | +0.0270 [-0.0206, +0.0763] includes zero |
| `H2_split_recipe_minus_base` | +0.0096 [-0.0528, +0.0686] includes zero | +0.0094 [-0.0717, +0.0820] includes zero | +0.0065 [-0.0542, +0.0648] includes zero |
| `H2f_flip_recipe_minus_base` | +0.0086 [-0.0526, +0.0671] includes zero | +0.0123 [-0.0687, +0.0846] includes zero | +0.0071 [-0.0531, +0.0647] includes zero |
| `H2o_orig_recipe_minus_base` | +0.0089 [-0.0531, +0.0676] includes zero | +0.0102 [-0.0706, +0.0827] includes zero | +0.0070 [-0.0534, +0.0650] includes zero |
| `H3_split_recipe_minus_fpshuf` | +0.0144 [-0.0157, +0.0425] includes zero | +0.0266 [-0.0006, +0.0582] includes zero | +0.0405 [+0.0053, +0.0820] above zero |
| `H3f_flip_recipe_minus_fpshuf` | +0.0147 [-0.0148, +0.0428] includes zero | +0.0292 [+0.0013, +0.0621] above zero | +0.0415 [+0.0068, +0.0820] above zero |
| `H3o_orig_recipe_minus_fpshuf` | +0.0150 [-0.0151, +0.0436] includes zero | +0.0273 [-0.0002, +0.0595] includes zero | +0.0411 [+0.0064, +0.0824] above zero |
| `H4_ctrl_line_orig_minus_0.5` | +0.0000 [-0.0393, +0.0413] includes zero | +0.0000 [-0.0403, +0.0408] includes zero | -0.0001 [-0.0440, +0.0442] includes zero |
| `H4b_ctrl_h1_split_minus_0.5` | +0.0023 [-0.0097, +0.0141] includes zero | -0.0042 [-0.0168, +0.0089] includes zero | -0.0008 [-0.0124, +0.0109] includes zero |
| `H5_residual_true_split` | +0.3642 [+0.3180, +0.4105] above zero | +0.3195 [+0.2787, +0.3575] above zero | +0.3381 [+0.2911, +0.3811] above zero |
| `H5null_residual_ctrl_h2_split` | -0.0029 [-0.0114, +0.0058] includes zero | +0.0005 [-0.0066, +0.0077] includes zero | +0.0019 [-0.0072, +0.0106] includes zero |
| `H5o_residual_true_orig` | +0.3730 [+0.3266, +0.4192] above zero | +0.3292 [+0.2882, +0.3667] above zero | +0.3470 [+0.2991, +0.3905] above zero |
| `base_H1_split_minus_0.5` | -0.0066 [-0.0610, +0.0471] includes zero | +0.0078 [-0.0442, +0.0617] includes zero | +0.0200 [-0.0352, +0.0775] includes zero |
| `base_H1f_flip_minus_0.5` | -0.0060 [-0.0601, +0.0477] includes zero | +0.0069 [-0.0449, +0.0609] includes zero | +0.0199 [-0.0355, +0.0777] includes zero |
| `base_H3_split_own_minus_fpshuf` | -0.0027 [-0.0115, +0.0054] includes zero | +0.0071 [-0.0103, +0.0284] includes zero | +0.0206 [+0.0051, +0.0384] above zero |
| `base_H3f_flip_own_minus_fpshuf` | -0.0023 [-0.0114, +0.0060] includes zero | +0.0069 [-0.0109, +0.0285] includes zero | +0.0203 [+0.0050, +0.0380] above zero |
| `fpzero_split_minus_0.5` | -0.0031 [-0.0420, +0.0382] includes zero | -0.0017 [-0.0533, +0.0484] includes zero | -0.0028 [-0.0455, +0.0414] includes zero |
| `duplicate_split_minus_0.5` | +0.3836 [+0.3665, +0.4002] above zero | +0.4095 [+0.3913, +0.4263] above zero | +0.3911 [+0.3721, +0.4094] above zero |
| `ctrl_sample_split_minus_0.5` | +0.0024 [-0.0091, +0.0142] includes zero | +0.0049 [-0.0096, +0.0201] includes zero | -0.0062 [-0.0223, +0.0096] includes zero |

### Within-plate PDS, mean over covered conditions

#### WP split 1 (dev)
| arm | S-orig | S-split | S-flip |
|---|---|---|---|
| recipe | 0.5025 | 0.5030 | 0.5025 |
| recipe_fpshuf | 0.4875 | 0.4886 | 0.4878 |
| recipe_fpzero | 0.4969 | 0.4969 | 0.4968 |
| base | 0.4936 | 0.4934 | 0.4940 |
| base_fpshuf | 0.4962 | 0.4961 | 0.4963 |
| base_fpzero | 0.4971 | 0.4970 | 0.4973 |
| recon_recipe | 0.5236 | 0.5245 | 0.5230 |
| recon_base | 0.5050 | 0.5049 | 0.5053 |
| ctrl_sample | 0.5028 | 0.5024 | 0.5028 |
| ctrl_sample_h1 | 0.5032 | 0.5023 | 0.5044 |
| ctrl_line | 0.5000 | 0.5000 | 0.5000 |
| duplicate | 0.8846 | 0.8836 | 0.8830 |

#### WP split 0
| arm | S-orig | S-split | S-flip |
|---|---|---|---|
| recipe | 0.5177 | 0.5172 | 0.5192 |
| recipe_fpshuf | 0.4904 | 0.4906 | 0.4900 |
| recipe_fpzero | 0.4990 | 0.4983 | 0.4994 |
| base | 0.5075 | 0.5078 | 0.5069 |
| base_fpshuf | 0.5001 | 0.5007 | 0.5001 |
| base_fpzero | 0.4981 | 0.4986 | 0.4980 |
| recon_recipe | 0.5323 | 0.5314 | 0.5334 |
| recon_base | 0.5209 | 0.5210 | 0.5208 |
| ctrl_sample | 0.5058 | 0.5049 | 0.5054 |
| ctrl_sample_h1 | 0.4966 | 0.4958 | 0.4964 |
| ctrl_line | 0.5000 | 0.5000 | 0.5000 |
| duplicate | 0.9111 | 0.9095 | 0.9093 |

#### WP split 2
| arm | S-orig | S-split | S-flip |
|---|---|---|---|
| recipe | 0.5270 | 0.5264 | 0.5271 |
| recipe_fpshuf | 0.4859 | 0.4859 | 0.4856 |
| recipe_fpzero | 0.4981 | 0.4972 | 0.4983 |
| base | 0.5200 | 0.5200 | 0.5199 |
| base_fpshuf | 0.4994 | 0.4994 | 0.4996 |
| base_fpzero | 0.5024 | 0.5022 | 0.5026 |
| recon_recipe | 0.5387 | 0.5384 | 0.5390 |
| recon_base | 0.5302 | 0.5295 | 0.5298 |
| ctrl_sample | 0.4929 | 0.4938 | 0.4933 |
| ctrl_sample_h1 | 0.5004 | 0.4992 | 0.5004 |

### Plate-residual cosine, mean over all conditions (S-orig: change vs C, offset L − C; S-split: change vs C2, offset L − C1)
| ctrl_line | 0.4999 | 0.4999 | 0.4999 |
| duplicate | 0.8936 | 0.8911 | 0.8921 |

#### RES split 1 (dev)
| change vector | S-orig | S-split |
|---|---|---|
| true | 0.3730 | 0.3642 |
| ctrl_sample_h2 | 0.0011 | -0.0029 |
| ctrl_sample_h1 | -0.0057 | -0.0369 |
| ctrl_sample | -0.0029 | -0.0209 |
| ctrl_line | 1.0000 | 0.9817 |
| duplicate | 0.3206 | 0.3134 |
| recipe | 0.6373 | 0.6277 |
| recipe_fpshuf | 0.6237 | 0.6141 |
| recipe_fpzero | 0.7323 | 0.7224 |
| base | 0.2541 | 0.2508 |
| base_fpshuf | 0.2521 | 0.2488 |
| base_fpzero | 0.2490 | 0.2458 |
| recon_recipe | 0.6356 | 0.6261 |
| recon_base | 0.2632 | 0.2597 |

#### RES split 0
| change vector | S-orig | S-split |
|---|---|---|
| true | 0.3292 | 0.3195 |
| ctrl_sample_h2 | 0.0015 | 0.0005 |
| ctrl_sample_h1 | -0.0079 | -0.0332 |
| ctrl_sample | 0.0070 | -0.0067 |
| ctrl_line | 1.0000 | 0.9830 |
| duplicate | 0.2850 | 0.2771 |
| recipe | 0.6443 | 0.6352 |
| recipe_fpshuf | 0.6457 | 0.6364 |
| recipe_fpzero | 0.7223 | 0.7122 |
| base | 0.2617 | 0.2589 |
| base_fpshuf | 0.2616 | 0.2588 |
| base_fpzero | 0.2639 | 0.2610 |
| recon_recipe | 0.6424 | 0.6336 |
| recon_base | 0.2701 | 0.2674 |

#### RES split 2
| change vector | S-orig | S-split |
|---|---|---|
| true | 0.3470 | 0.3381 |
| ctrl_sample_h2 | 0.0009 | 0.0019 |
| ctrl_sample_h1 | 0.0092 | -0.0183 |
| ctrl_sample | -0.0034 | -0.0166 |
| ctrl_line | 1.0000 | 0.9807 |
| duplicate | 0.3018 | 0.2944 |
| recipe | 0.6319 | 0.6235 |
| recipe_fpshuf | 0.6162 | 0.6074 |
| recipe_fpzero | 0.7104 | 0.7006 |
| base | 0.2659 | 0.2636 |
| base_fpshuf | 0.2639 | 0.2617 |
| base_fpzero | 0.2637 | 0.2613 |
| recon_recipe | 0.6279 | 0.6198 |
| recon_base | 0.2729 | 0.2706 |
