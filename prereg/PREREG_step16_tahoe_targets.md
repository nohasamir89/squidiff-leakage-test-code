# PREREG — Step 16: target-aware compound representation on the Tahoe brain/gut lines

Written 2026-10-06, after Steps 15b–15e (`step15_findings.md`, `step15c_findings.md`, `step15d_findings.md`,
`step15e_findings.md`) and before any Step 16 model exists. Two new training arms (GPU); everything else — data files,
splits, steps, seeds, sampler, scorer code paths — is Step 15b's. This document is committed before the training
launcher runs; nothing in it is edited after a checkpoint exists (pre-run corrections, if any, are dated paragraphs
inside a section).

## 1. Question

Step 15c showed that the recipe's predicted *direction* of response identifies a held-out drug (PDS 0.72 / 0.72 / 0.71
against chance 0.5 and a plate-free replicate ceiling of 0.84); Step 15b showed that the size and location of the
predicted shift are not better than compound-blind baselines; Step 15e showed that the predictions carry only a
boundary-level amount of sensitivity information (U_pool +0.151 [−0.008, +0.305]). The compound enters the model as a
1,024-bit structural fingerprint and nothing else. Step 16 asks whether telling the model *what the compound targets*
changes any of this.

**Q1 (primary).** Does the target-aware arm identify held-out drugs better than the recipe — PDS(targets) − PDS(recipe)
above zero? **Q2 (co-primary).** Does it change the thesis's primary metric — DE-overlap(targets) − DE-overlap(recipe)?
**Q3 (information control).** Is any gain due to the *information* and not to the wider input — PDS(targets) −
PDS(targets-shuffled) above zero, where the shuffled arm receives the same input width filled with another drug's
targets? Secondary: the response-magnitude score of §4.4 (new), energy distance and DPC-50 as in Step 15b, the pooled
PRISM utility of Step 15e for the new arms, and the "less unseen" analysis of §4.5.

## 2. What is known, and its status

| fact | source |
|---|---|
| Recipe: PDS 0.7223 / 0.7181 / 0.7142 (splits 1 / 0 / 2); DE-overlap 0.464 / 0.460 / 0.459; replicate PDS ceiling 0.7263 / 0.8060 / 0.7341 (pre-registered sets), 0.8393 plate-removed (split 1) | step15c_findings, step15_findings |
| attn_cross and mlp_ctrl (Step 15b/15c): attn_cross − mlp_ctrl on PDS −0.001 / +0.005 / +0.009 (includes / includes / above); on DE-overlap tiny, exploratory — the precedent for a parameter-matched control arm | step15c_findings |
| Curated targets (Repurposing Hub 2020-03-24, current HGNC symbols): 282 / 375 usable drugs annotated, 544 target symbols; held-out drugs with targets 58 / 59 / 56 (splits 0 / 1 / 2); 4 of the 278 GPT target symbols on the panel (the curated count is reported by the scorer) | `step15_targets_curated_hgnc.csv` (d9120cd), coverage note |
| Pooled PRISM utility of the recipe +0.151 [−0.008, +0.305] (reading A), +0.189 [+0.020, +0.343] (reading B); ceiling +0.248; measured effect size +0.381 | step15e_findings (73bda2b) |
| Training recipe of Step 15b: adaLN + Gaussian flow matching, cond_drop 0, encoder_arm mlp, 100,000 steps, fingerprint encoding (drug_dimension 1,024), seeds 0–2, 58,178,180 parameters; sampling 50 NFE Euler, unguided | `train_step15_tahoe.sh`, `eval_step15_tahoe.sh`, PREREG_step15b |
| The encoder's first layer reads gene_size + drug_dimension inputs; the samplers infer the drug encoding from that layer's width (`encoder.fc1.weight`) | `Squidiff/MLPModel.py`, `sample_squidiff.py`, `leakage_test_fm_tahoe.py` |

## 3. Arms (two new; the recipe and its Step 15b arrays are the comparators, not retrained)

- **targets** — the recipe with `drug_encoding = fp_targets`: the 1,024-bit fingerprint, then a **bag of curated targets**
  (one input per symbol of the fixed vocabulary V = the sorted union of the curated target symbols of all 375 usable
  drugs, |V| = 544), then one **has-targets flag**; every block scaled by log10(dose + 1) exactly as the published
  encoder, so a zero-dose control is all zeros. A drug without curated targets has zeros in the bag and the flag. The
  lookup is SMILES → drug name(s) (the Step 15a record's `table_smiles`) → `curated_targets_hgnc`; when two names share
  a SMILES their target sets are united (counted and printed). drug_dimension = 1,024 + 544 + 1 = 1,569.
- **targets_shuf** — identical in every respect, with the bag and flag of each drug replaced by those of another drug
  under one fixed cyclic order of the 375 usable drugs (a `RandomState(0)` permutation of the sorted drug names; each
  drug receives the bag of the next drug in the cycle, so no drug keeps its own; the order is printed with its hash
  and is the same at training and at sampling — the evaluation refuses a run whose hash differs from training). The
  input width, the parameter count and the input statistics (the multiset of bags over the usable drugs) are those of
  the targets arm; the correspondence between a structure and its targets is destroyed. Two drugs can have the same bag
  (both empty, say), so the shuffled bag need not differ from the drug's own for every drug; the number of drugs for
  which it does is printed by the encoder and recorded.

Everything else equals the recipe: adaLN, Gaussian flow matching, cond_drop 0, encoder_arm mlp, 100,000 steps, batch,
learning rate, EMA, the Step 15a files, seeds 0–2 on `drug_split_0/1/2` (3 × 3 × 2 = 18 training runs), 50 NFE Euler
unguided sampling by the Step 15b harness, the same certified-array format and provenance.

**Parameter count.** The only change is the first encoder layer: expected parameters = 58,178,180 + (1,569 − 1,024) ×
(width of `encoder.fc1`); the launcher computes the expectation from the built model and prints it; the two arms must
have identical counts.

**Dry run, 2026-10-06 (before any Step 16 model exists).** The launcher guard on drug_split_1 reports: the encoder's
first layer is 2,048 wide, so both arms have 59,294,340 parameters = 58,178,180 + 545 x 2,048, and the recipe rebuilt by
the same code path gives 58,178,180. The encoder resolves the 375 usable drugs to 372 SMILES -- three SMILES carry two
drug names each ((S)-Crizotinib and crizotinib; Doxorubicin (hydrochloride) and Epirubicin (hydrochloride); Trametinib
and Trametinib (DMSO_TF solvate)), and their target sets are united -- of which 279 have a non-empty curated bag; the
vocabulary is 544 symbols, as pre-registered. The fixed cyclic order (sha256 3b9848626e13) gives a bag different from the
drug's own for 350 of the 372 SMILES. One asymmetry follows from shuffling over all 375 drugs while each arm trains on a
subset: on drug_split_1 the targets arm sees 220 of 297 training SMILES with a non-empty bag (mean bag size 2.63, max 49)
and the shuffled arm 221 (mean 2.81, max 49); on the held-out side 59 of 75 in both (mean 3.45 and 2.77). The two arms
match exactly in input width, parameter count and the bag multiset over all usable drugs, and approximately in the bag
distribution within a split; the findings report these counts next to Q3.

**Less-unseen caveat (stated now).** A held-out drug with targets that also occur among the training drugs' targets
enters the model with an input it has partly seen; a held-out drug without targets enters exactly as in the recipe.
§4.5 quantifies this per drug and the findings report the gain by stratum.

## 4. Metrics

**4.1 PDS** — Step 15c's definition on Step 15c's candidate sets (same line, same dose, minus other conditions of the
same drug; `score_step15c_tahoe.pds_sets`), per seed, averaged per condition; the replicate ceiling as in Step 15c; the
cross-plate facet as in Step 15c §8 (standing facet).

**4.2 DE-overlap, energy distance, DPC-50** — Step 15b's definitions and code (`score_step15_tahoe.thesis_per_cond`,
`energy`), per seed, averaged per condition.

**4.3 Baselines** — the Step 15b shift baselines rebuilt by the Step 15b code path and asserted to reproduce
`step15_model_per_condition.csv` to 1e-9; the Step 15b arms' certified arrays loaded as in Step 15c (hash-checked); the
new arms' arrays certified by the same harness.

**4.4 Response-magnitude score (new; MRA).** For every (line, dose) context of held-out conditions with ≥ 10 members
(the Step 15c contexts): the Spearman correlation across the context's conditions between the L2 norm of the predicted
effect (B − C) and the norm of the measured effect (true − C); MRA = mean over contexts. Arms: computed per seed, the
arm's MRA is the mean over seeds. Chance 0. Control-paste (zero norm everywhere) and mean-response (one norm per line)
are constant within a context → 0 by definition (asserted). Replicate ceiling: within each context, over the
conditions that have a copy on the other plate (plates 6 and 14), the copy's measured norm offered as the prediction
of the condition's own measured norm; mean over the contexts with ≥ 10 such conditions, their number reported.
Reported for every submission (Step 15b arms, baselines, new arms), per seed, per split, brain versus gut.
Pre-registered reason: Step 15d/15e found that the *size* of the measured response is what carries sensitivity
(+0.381 pooled), and Step 15b that the recipe does not learn size.

**4.5 Less-unseen analysis (descriptive).** For every held-out drug: number of curated targets; number of those present
in the training drugs' target sets of that split ("seen"); strata: no targets / targets, none seen / targets, ≥ 1 seen.
PDS and DE-overlap of targets, targets_shuf and recipe by stratum, with the stratum sizes.

**Addition of 2026-10-08 (after the development run on drug_split_1, before any confirmation run or result exists;
exploratory, not gated).** The stratum means on their own cannot say whether the gain depends on the stratum, so the
scorer also reports the **stratum gap**: an arm's gain over the recipe among the drugs whose targets were seen in
training, minus its gain among the drugs with no curated targets at all, with the same structure-clustered percentile
bootstrap as every other contrast (5,000 replicates, RandomState(0), 97.5 %; a drug belongs to exactly one stratum,
asserted; a replicate drawing no drug from one stratum is dropped and counted). A drug with no curated targets enters
the target-aware arm as the published fingerprint followed by 545 zeros, so its input differs from the recipe's in
width alone; if the gain is target information, this gap must be above zero. Computed for targets and for
targets_shuf, each against the recipe, on PDS and on DE-overlap. For targets_shuf the strata are labelled by the drug's
own bag while its input carries another drug's, so the shuffled gap is a consistency check, not a test of information.
Reason for adding it now: on drug_split_1 the stratum PDS gains of the targets arm were +0.0098 (no targets), +0.0081
(targets, none seen) and +0.0093 (targets, at least one seen) -- no visible dependence on the stratum, the opposite of
the section 7 expectation -- and an interval on the difference is needed before that reading can be written down.

**4.6 Pooled PRISM utility (secondary).** The Step 15e computation (same instruments f_s and QSAR_s, same 85 drugs,
reading A and reading B) applied to the new arms' predicted effects; contrast targets − recipe, targets_shuf − recipe
and targets − targets_shuf on U_pool (reading A) and U_B (reading B), same stratified bootstrap. The per-split U_s of
the new arms (the Step 15d highest-dose computation) is reported in the development run with a per-split drug
bootstrap; the pooled readings need all three splits and are computed in the final run only. Every Step 15e value
(U_s per split, U_pool, U_B and the recipe's pooled interval) is asserted to reproduce the Step 15e record to 1e-9.

## 5. Endpoints

All intervals: structure-clustered percentile bootstrap over the split's held-out structures, 5,000 replicates,
`RandomState(0)`, 97.5 % per split, as Steps 14–15c; per-condition values (PDS, DE-overlap, energy, DPC-50) fixed under
resampling; MRA and the utility U_s recomputed on the resampled conditions in every replicate (as the Step 15d/15e
utility was), the same resample for every submission. Split 1 is read first
(development: the three seeds of both arms on `drug_split_1` are trained and scored first); splits 0 and 2 are
confirmation; categories require all three.

- **Primary, gated (Q1):** mean PDS, targets − recipe, per split.
- **Co-primary, gated (Q2):** mean DE-overlap, targets − recipe, per split.
- **Information control, gated (Q3):** mean PDS, targets − targets_shuf, per split.
- **Secondary, not gated:** targets_shuf − recipe on PDS and DE-overlap; targets − recipe and targets − targets_shuf on
  MRA, energy and DPC-50; every submission's MRA − 0 and the MRA ceiling; the §4.5 strata; the §4.6 utility contrasts
  (reading A and B); per seed and worst seed / worst split; cross-plate PDS; brain versus gut; per line, per dose.

## 6. Outcome categories, fixed now (precedence 1 → 2 → 5 → 3 → 4, coded in the scorer)

1. **TARGET INFORMATION HELPS DIRECTION** — Q1 above zero on all three splits and Q3 above zero on all three.
2. **WIDTH, NOT INFORMATION** — Q1 above zero on all three; Q3 not above zero on at least one (the shuffled bag does as
   well: the gain is not target information).
3. **TARGET INFORMATION HURTS** — Q1 below zero on at least one split (and not category 5).
4. **PARTIAL** — Q1 above zero on one or two splits, including zero on the rest.
5. **UNRESOLVED** — Q1 includes zero on all three splits. Not equivalence.

Q2 carries its own three-split verdict (above / includes / below on each split), reported next to the category and in
the findings; it does not change the category. MRA and the utility contrasts carry verdicts, not categories.

## 7. Expectation, recorded in advance

I expect a **small PDS gain** for the targets arm, +0.01 to +0.03 per split, concentrated in the stratum "targets, ≥ 1
seen" and absent in "no targets" (where the arm's input equals the recipe's plus a zero flag), and the shuffled arm
within ± 0.01 of the recipe; hence **category 1 or 5** — the intervals of Step 15c's exploratory arm contrast were about
± 0.01 wide, so a +0.02 gain would resolve and a +0.005 gain would not. I expect DE-overlap unchanged (the shared
response dominates it; the targets say which drug, not how the line's generic programme moves), MRA unchanged for all
three arms and near zero for the recipe (targets do not say how strong the response is), and the pooled PRISM U within
± 0.05 of the recipe's. If the targets arm is below the recipe on any split, the thesis says so plainly: a wider,
sparser input hurt a 300-drug training set.

## 8. What this step cannot establish

- That target information helps a model that never saw the targets' companions: the less-unseen caveat of §3; the
  stratum analysis bounds it but does not remove it.
- Anything about magnitude *learning*: MRA is a new read of the same frozen and new predictions; no arm is designed to
  learn magnitude here.
- Transfer to sci-Plex or to lines outside the 14; dose extrapolation; guidance (dropped after Steps 15b/15c).
- The thesis's external-validation story does not change unless §4.6 moves: a PDS gain alone is an internal gain.

## 9. Analysis plan and provenance

Code, all committed before the launcher runs, every edit to an existing file applied by `patch_step16.py` (exact-string
replacements with count assertions; never a hand edit; the patched files' hashes recorded):

- `Squidiff/scrna_datasets.py` (edited by `patch_step16.py`): `_load_target_map()` (reads the Step 15a record and
  `step15_targets_curated_hgnc.csv` from the repository root; pins |V| = 544 as `TARGET_VOCAB_SIZE`; prints the
  shared-SMILES unions and the shuffled order's hash), `Target_dose_encoder`, `Fp_targets_dose_encoder` (plain and
  shuffled), entries `fp_targets` and `fp_targets_shuf` in `DRUG_ENCODERS` / `DRUG_ENCODER_DIMS` (both 1,569); the
  shuffled variant is selected by name, never by width.
- `train_squidiff.py` (edited by `patch_step16.py`): the parameter-count expectation becomes base + (drug_dimension −
  1,024) × width(fc1), for the mlp encoder arm only.
- The samplers infer the encoding from the first-layer width; two encodings share the width 1,569, so the launcher
  names the arm's encoding in the environment variable `SQUIDIFF_DRUG_ENCODING`, asserted against the inferred width by
  `sample_squidiff.py` (edited by `patch_step16.py`) and by `leakage_test_fm_step16.py` (generated from
  `leakage_test_fm_tahoe.py` by `port_harness_step16.py`; one block replaced, header with the source sha256); the
  Step 15b arms and their harness are untouched (width 1,024 maps to one encoding).
- `train_step16_tahoe.sh` (18 tasks: split = T / 6, arm = (T % 6) / 3, seed = T % 3; same partition, guards and post
  block as Step 15b; the guard builds the arm's model and the recipe's by the same code path and checks n(arm) −
  n(recipe) = (1,569 − 1,024) × width(fc1); `finalize_step16.py` = Step 15's finaliser with the encoding passed through
  and the two encoder inputs sha256-pinned), `eval_step16_tahoe.sh` (as Step 15b's with the encoding passed through;
  refuses a run whose sampler or harness reports another encoding, or whose shuffled-order hash differs from training),
  `score_step16_tahoe.py` (loads the Step 15b arrays and the new arrays, reproduces the Step 15b per-condition record,
  the Step 15c PDS record and the Step 15e utility record to 1e-9, computes §4, the contrasts of §5 and the category
  of §6; development run on `drug_split_1` first, outputs with the suffix `_development`, then the final run).
- Outputs: `tahoe_results/step16_analysis[_development].json`, `step16_long[_development].csv`,
  `step16_per_condition[_development].csv`; arrays `tahoe_results/vectors_s16_<split>_<arm>_seed_<s>[_pc].npz`
  (gitignored, sha256 in `provenance/s16ev_*.json`); checkpoints under `step16/` (gitignored, sha256 in
  `provenance/s16tr_*.json`). Never overwrites.

Order: patch and code committed → dry run of the launcher (guards, parameter rule, encoder self-check on every SMILES of
the split's training and test files: the fingerprint block equals the published encoder, each drug's bag equals its
curated set, the flag equals has-targets, the shuffled bag equals the partner's, a zero-dose control is all zeros) →
training of the six development runs (`drug_split_1`, both arms, three seeds) → their evaluation → development scoring
→ confirmation training (12 runs) → evaluation → final scoring → findings.

- Step 15b records: 6b40c72; Step 15c: 24c16d8; coverage and provenance: ea7acf0, e6cbd2c, d9120cd; Step 15d/15e:
  67d8f9a, 73bda2b.
- Pre-registration commit: [hash]. Code commit: [hash]. Training array: [job id]. Evaluation arrays: [job ids].
