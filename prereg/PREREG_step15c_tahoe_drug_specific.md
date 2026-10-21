# PREREG — Step 15c: drug-specific signal in the Tahoe predictions (perturbation discrimination)

Written 2026-10-03, after the Step 15b results (`step15_findings.md`) and before any number of this step exists. No
training; every input is a Step 15b array already scored and hash-pinned. This document is committed before
`score_step15c_tahoe.py` runs; nothing in it is edited after results exist (pre-run corrections, if any, are dated
paragraphs inside a section, never a new section).

## 1. Question

Step 15b established that on the Tahoe brain and gut lines the recipe beats base Squidiff on every metric, is
indistinguishable from the within-held-out ridge shift on DE-overlap, and is **below a compound-blind per-line
mean response on DE-overlap on all three splits**, with a per-condition learnability of ≈ 0.05 (the drug shift is
≈ 5 % of the between-line shift). None of the three thesis metrics separates "the prediction carries the line's
shared response" from "the prediction carries this drug's response". Step 15c asks the single question those
results leave open:

**Q1. Does any submission's prediction identify the held-out drug?** Given the predicted effect of condition p
(line, drug, dose, plate), is it closer to p's measured effect than to the measured effects of the other held-out
drugs at the same dose in the same line? Perturbation discrimination (PDS, Step 14's facet) answers this with a
no-information point of exactly 0.5 and an experimental ceiling from the plate 6 / plate 14 replicates.

**Q2.** Is the recipe's discrimination above that of the like-for-like linear shift (ridge-train), i.e. does the
generative model add drug-specific information that a linear model fitted on the training drugs does not have?

Secondary: the same for every other submission; whether classifier-free guidance, which gained DE-overlap and lost
energy in Step 15b, gains discrimination; whether discrimination rises with dose and with learnability.

## 2. What is known, and its status

| fact | source |
|---|---|
| sci-Plex (Step 14 §6/§11), raw PDS, split means over lines: ridge-S7 0.5544 / 0.5373 / 0.5010; ridge-train 0.5919 / 0.5365 / 0.5520; mean-response 0.5000 (by construction); control-paste 0.5000 (by construction); 50 NFE recipe 0.5303 / 0.5300 / 0.5158; guidance w = 2 0.5565 / 0.5430 / 0.5325; cross-attention 0.5379 / 0.5309 / 0.5246; base 0.5019 / 0.5059 / 0.5074; replicate half-split anchor 0.58–0.69 per line | Step 14 findings §6, §11 |
| sci-Plex facet contrast ridge-S7 minus 50 NFE on PDS includes zero on all three splits (+0.024 / +0.007 / −0.015) | Step 14 findings §6 |
| Tahoe (Step 15b): recipe below mean-response on DE-overlap on all three splits (−0.045 / −0.040 / −0.050); learnability mean 0.049 / 0.057 / 0.049; recipe's generated mean 1.67 from the real treated mean against a real shift of 1.37 (split 1, seed 0, post-hoc diagnostic) | step15_findings §3, §6, §9 |
| Tahoe design: 2,899 / 3,115 / 2,853 held-out conditions, 75 held-out structures per split, 14 lines, 3 doses (0.05 / 0.5 / 5 µM), plate 14 replicates plate 6 (259 / 299 / 246 condition pairs) | step15_data_record.json; step15_findings §2 |
| The arrays: `tahoe_results/vectors_s15_<split>_<arm>_seed_<s>.npz` (`cond`, `true`, `B`, `C`) for 45 runs, sha256 in `provenance/s15ev_*.json`; the shift baselines are rebuilt from the test / train / control files exactly as `score_step15_tahoe.py` (commit ed423ba) built them | Step 15b records, commit 6b40c72 |

PDS on sci-Plex placed every submission within 0.06 of the no-information point and the half-split replicate at
0.58–0.69, so the sci-Plex drug-specific signal was small for everything. Whether the Tahoe lines, with 6.5× the
conditions and ≈ 75 competing drugs per (line, dose), show more or less is the open point.

## 3. Submissions and inputs (no new computation of predictions)

Nine submissions, exactly Step 15b's: the five arms (base, recipe, guidance w = 2, attn_cross, mlp_ctrl; three seeds
each, PDS computed per seed and averaged per condition, as every metric in Steps 6–15) and the four shift baselines
(ridge-S7, ridge-train, mean-response, control-paste), rebuilt by the same code path as Step 15b and **asserted to
reproduce** the per-condition DE-overlap and DPC-50 of `tahoe_results/step15_model_per_condition.csv` to 1e-9 before
anything else runs. Effects are profile minus paired-control profile: predicted effect = B − C (arms) or the
baseline's delta (shifts); measured effect = true − C. All from the committed arrays and the Step 15a files.

## 4. Metric: perturbation discrimination, Step 14's definition with one stated adaptation

`pds_cosine` of `step14_metrics.py` (commit as recorded in the Step 14 findings): cosine distance between the
predicted effect of condition p and the measured effects of a candidate set containing p; k_p = rank of the own match
(ties averaged; a zero-norm vector has distance 1 to everything); PDS_p = 1 − k_p / (n_p − 1).

**Adaptation for Tahoe (fixed now).** On sci-Plex the candidate set was the line's conditions. On Tahoe the same
(line, drug, dose) appears on plates 6 and 14 and the same drug appears at three doses, so the candidate set for
condition p is **the held-out conditions of p's line at p's dose, minus every other condition of p's drug** (which
removes the replicate copy; the dose match removes the dose confound). n_p is therefore the number of distinct
held-out drugs present at that dose in that line, ≈ 75. A condition whose candidate set has fewer than 10 members is
excluded from every submission alike and the count is reported. PDS is computed from the condition's own predicted
effect; the candidate measured effects are plate-corrected by construction (each relative to its own paired
control), so candidates from other plates are admissible.

By construction: mean-response (one delta per line) and control-paste (zero delta) give PDS = 0.5 exactly; both are
computed and reported as implementation checks.

*Pre-run correction, 2026-10-03 (before the scorer is committed or run):* "exactly 0.5 by construction" holds for control-paste under any candidate set (a zero-norm prediction ties with every candidate). For mean-response it holds only when every condition of a (line, dose) context has the same candidate set; the sibling exclusion gives the replicate copies smaller sets, so on the adapted sets its mean is near, not exactly, 0.5. The scorer therefore (i) asserts control-paste = 0.5 to 1e-9 on the adapted sets, (ii) asserts mean-response = 0.5 to 1e-9 per (line, dose) context on the unadapted full-context sets, as the check of the ranking code, and (iii) reports mean-response on the adapted sets as a descriptive value. This replaces the corresponding guard wording in section 9.

**Replicate ceiling** (descriptive): for each plate-6 / plate-14 pair, the plate-6 measured effect offered as the
prediction of the plate-14 condition, scored against the plate-14 condition's candidate set, and vice versa.

**Drug-level discrimination** (secondary): PDS averaged over a drug's conditions (lines × doses); per dose
(0.05 / 0.5 / 5 µM); per line; brain (A-172, H4, SW 1088) versus gut.

## 5. Endpoints

All intervals: structure-clustered percentile bootstrap over the split's held-out structures, 5,000 replicates,
`RandomState(0)`, 97.5 % per split, as Steps 14–15b. PDS_p is treated as fixed per condition under resampling (the
candidate sets are not re-drawn), as in Step 14's facet contrasts. Split 1 is read first (development); splits 0
and 2 are confirmation; categories require all three.

- **Primary, gated (Q1):** mean PDS of the recipe minus 0.5, per split.
- **Co-primary, gated (Q2):** mean PDS, recipe minus ridge-train, per split.
- **Secondary, reported, not gated:** each other submission's mean PDS minus 0.5; recipe minus ridge-S7; guidance
  w = 2 minus recipe; attn_cross minus mlp_ctrl (exploratory, as in Step 15b); the replicate ceiling; PDS per dose,
  per line, brain versus gut, per drug; Spearman of per-condition PDS against learnability (Step 15b's descriptor,
  read from `step15_model_per_condition.csv`); per-seed values and the worst seed / worst split.

## 6. Outcome categories, fixed now (precedence 1 → 2 → 5 → 3 → 4, coded in the scorer)

1. **DRUG-SPECIFIC SIGNAL BEYOND LINEAR** — primary above zero on all three splits and co-primary above zero on all
   three.
2. **DRUG-SPECIFIC SIGNAL, NOT BEYOND LINEAR** — primary above zero on all three; co-primary not above zero on at
   least one.
3. **NO DRUG-SPECIFIC SIGNAL** — primary not above zero on at least one split; the thesis then says the generative
   model's Tahoe predictions do not identify the held-out drug, whatever their DE-overlap, and Step 16's arms are
   aimed at that.
4. **PARTIAL** — a gated contrast resolved on one or two splits only; reported per split.
5. **UNRESOLVED** — a gated contrast includes zero on all three splits. Not equivalence.

## 7. Expectation, recorded in advance

I expect **category 2**. With ≈ 2,900 conditions per split the interval on the primary will be narrow, and on
sci-Plex every generative arm sat 0.02–0.06 above 0.5, so I expect the recipe above 0.5 on all three splits, at
0.52–0.56. I expect ridge-train at or above the recipe (on sci-Plex it was the highest submission on two splits),
so the co-primary negative or unresolved. I expect guidance w = 2 above the recipe on PDS (its DE-overlap gain in
Step 15b suggests sharper drug-specific ranking), the replicate ceiling well above every submission, PDS rising
with dose, and a positive Spearman against learnability. I record no expectation for attn_cross versus mlp_ctrl or
for brain versus gut. If the primary is not above zero on all three splits, the thesis says so plainly.

## 8. What this step cannot establish

- **A better model.** It measures what the Step 15b predictions contain; it changes no prediction.
- **Discrimination across lines or across doses.** Candidates are restricted to the same line and dose by design.
- **That ridge-train's discrimination is reachable by a generative model**, or the reverse; it is a comparator.
- **Anything about the biology of the discriminated drugs.** A per-drug PDS table is descriptive; target or MoA
  annotation is not part of this step (the coverage inventory is a separate, descriptive task).
- **Independence from Step 15b.** Same arrays, same splits; this is a further read of a frozen prediction set.

## 9. Analysis plan and provenance

`score_step15c_tahoe.py`, committed before it runs: guards (this document committed and unchanged in section count
≥ 9; scorer committed and clean; Step 15b records present and hash-checked against `provenance/s15ev_*.json`;
baseline reproduction to 1e-9 against `step15_model_per_condition.csv`; mean-response and control-paste PDS equal
0.5 to 1e-9); then PDS for every submission, the replicate ceiling, the endpoints of §5, the categories of §6 when
all three splits are present. Development run on `drug_split_1` alone first (outputs with the suffix `_development`),
then the final run. Outputs: `tahoe_results/step15c_pds.json`, `step15c_pds_long.csv`, `step15c_pds_per_condition.csv`.
Never overwrites. Stops on any missing input, hash mismatch, failed reproduction, or candidate-set size below 10 in
more than 1 % of conditions.

- Step 15b records: commit 6b40c72 (arrays hash-pinned in `provenance/s15ev_*`), findings `step15_findings.md`.
- Step 14 facet code: `step14_metrics.py` (`pds_cosine`), as cited in the Step 14 findings.
- Pre-registration commit: [hash]. Scorer commit: [hash].
