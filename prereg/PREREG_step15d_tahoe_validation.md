# PREREG — Step 15d: external validation of the Tahoe predictions (sensitivity labels, drug targets, tumour proteomes)

Written 2026-10-05, after Steps 15b/15c (`step15_findings.md`, `step15c_findings.md`) and the coverage inventory with
roadmap steps 1–3 (`claude/coverage_inventory_abc.md`; commits ea7acf0, e6cbd2c, d9120cd), and before any number of
this step exists. No training; every prediction is a Step 15b array already scored and hash-pinned. This document is
committed before any of the three scripts of section 9 runs; nothing in it is edited after results exist (pre-run
corrections, if any, are dated paragraphs inside a section, never a new section).

## 1. Questions

Step 15c established that the recipe's predicted *direction* of response identifies the held-out drug among ≈ 75
alternatives (PDS 0.72 / 0.72 / 0.71 against chance 0.5 and a plate-free replicate ceiling of 0.84), while Step 15b
showed that the size and location of the predicted shift are not better than compound-blind baselines. Every number
so far comes from the Tahoe measurements themselves. Step 15d asks what the predictions are worth against data the
model never saw, in three layers of decreasing strength:

**A. Utility against independent sensitivity labels (gated).** PRISM measured, for 10 of the 14 Tahoe lines, the
viability response (AUC of an 8-point dose curve) to 135 of the 375 Tahoe drugs. A predictor "expression effect →
AUC" is fitted on the *training* drugs from their *measured* effects. Fed the recipe's *predicted* effects for the
*held-out* drugs, does it rank the held-out drugs within a line the way PRISM measured them (Q1)? Does it do so better
than the like-for-like linear shift fitted on the training drugs, ridge-train (Q2)?

**B. Target plausibility (secondary, not gated).** For a held-out drug with curated targets, are the 20 panel genes the
recipe predicts to respond most closer, in a protein-interaction network, to that drug's own targets than to the
targets of the other held-out drugs?

**C. Tumour layer (descriptive, no endpoint).** With CPTAC GBM and COAD: (C1) is each drug's curated target
quantified as a protein in the tumour cohort that matches the line's tissue; (C2) which panel genes have mRNA that
tracks protein across tumours ("protein-reflected") and which do not; (C3) does each line's untreated profile resemble
the GBM cohort or the colon cohort more.

## 2. What is known, and its status (every number from a committed record)

| fact | source |
|---|---|
| Tahoe design: 2,899 / 3,115 / 2,853 held-out conditions (splits 1 / 0 / 2), 75 / 75 / 76 held-out drugs, 14 lines (brain: A-172, H4, SW 1088; gut: COLO 205, HCT15, HT-29, KATO III, LS 180, LoVo, RKO, SNU-1, SW1417, SW48, SW480), doses 0.05 / 0.5 / 5 µM, plate 14 replicates plate 6 at 5 µM | `step15_data_record.json`; step15_findings §2 |
| Step 15b: recipe below mean-response on DE-overlap on all three splits; learnability ≈ 0.05; the recipe's spread and library size match the real cells, the base arm's do not | step15_findings §3, §6, §9 |
| Step 15c: PDS recipe 0.7223 / 0.7181 / 0.7142 (splits 1 / 0 / 2); ridge-train 0.5164 / 0.5400 / 0.5258; ridge-S7 0.5039 / 0.4915 / 0.4859; mean-response 0.5002 / 0.5000 / 0.5003; replicate ceiling 0.7263 / 0.8060 / 0.7341 (pre-registered sets), 0.8393 with the plate removed (split 1) | step15c_findings; `step15c_pds.json`, `step15c_diag_crossplate_drug_split_1.json` |
| PRISM Repurposing 19Q4 secondary screen (Corsello et al. 2020, figshare 10.6084/m9.figshare.9393293.v4): 10 / 14 Tahoe lines (absent: COLO 205, SW1417, KATO III, SNU-1); 135 / 375 usable drugs matched (20 by name and structure, 106 name only, 9 structure only); 1,234 of 1,350 (line, drug) pairs with a fitted AUC, all passed STR profiling; held-out drugs in the screen 33 / 27 / 25 (splits 0 / 1 / 2) with 303 / 253 / 223 labelled pairs | `step15_coverage_inventory_d.json`, `step15_prism_match.csv` (commit ea7acf0) |
| Curated targets = Broad Repurposing Hub 2020-03-24: 282 / 375 drugs; held-out 58 / 59 / 56 (splits 0 / 1 / 2); 544 target symbols, all current HGNC symbols; GPT targets kept only as a flag | `step15_targets_curated.csv`, `step15_targets_curated_hgnc.csv` (commits e6cbd2c, d9120cd) |
| Panel: 157 protein-coding, 26 non-coding RNA, 1 other, 16 without an HGNC entry (HGNC 2026-01-06); of the 278 GPT target symbols, 4 were on the panel (the count for the 544 curated symbols is not yet computed; layer B reports it) | `step15_panel_harmonised.csv`; coverage note (b) |
| CPTAC (cptac 1.5.14, Zenodo 10.5281/zenodo.8394329): GBM UMich proteome 100 tumours / 12,100 symbols, WashU transcriptome 99 tumours, both 99, clinical 100 × 124; COAD UMich proteome 97 tumours / 9,153 symbols, transcriptome 106, both 96, clinical 109 × 124 | `tahoe_results/external/cptac_fetch_record.json`, `step15_coverage_inventory_c.json` |
| Harmonised coverage: panel in GBM proteome / transcriptome / both 125 / 192 / 125; COAD 91 / 192 / 91; both proteomes 80; drugs with ≥ 1 curated target quantified 238 / 282 (GBM; held-out 47 / 56 / 45) and 210 / 282 (COAD; 41 / 48 / 43) | `step15_symbol_harmonisation.json` (commit d9120cd) |
| The arrays: `tahoe_results/vectors_s15_<split>_<arm>_seed_<s>.npz` (`cond`, `true`, `B`, `C`), 45 runs, sha256 in `provenance/s15ev_*.json`; shift baselines rebuilt exactly as `score_step15_tahoe.py` builds them and asserted to reproduce `step15_model_per_condition.csv` to 1e-9 (the Step 15c code path) | Step 15b/15c records |

Not yet in hand (fetched by the scripts of section 9, recorded with sizes and hashes before any scoring): the STRING
v12.0 human network files (links with combined score, protein info, aliases) for layer B. The CPTAC clinical tables
have not been read beyond their shape; whether they carry a published subtype field is unknown and layer C3 does not
depend on it.

## 3. Submissions and inputs (no new predictions)

Nine submissions, exactly Step 15b's: the five arms (base, recipe, guidance w = 2, attn_cross, mlp_ctrl; three seeds
each; every per-condition quantity is computed per seed and averaged over seeds, as in Steps 6–15) and the four shift
baselines (ridge-S7, ridge-train, mean-response, control-paste). Predicted effect of a condition = B − C (arms) or the
baseline's delta; measured effect = true − C; all from the committed arrays and the Step 15a files. Only the **5 µM**
conditions are used in layers A and B: PRISM's labels integrate an 8-point curve up to 10 µM, so the highest Tahoe dose
is the closest, and it is the dose at which the plate-14 replicate exists. A (line, drug) with 5 µM conditions on both
plates gets the average of the two conditions' effects (predicted and measured alike); a pair with no 5 µM condition
is excluded and counted.

External inputs, each hash-checked against its fetch record before use: PRISM `secondary-screen-dose-response-curve-parameters.csv`
and `secondary-screen-cell-line-info.csv` (figshare record `tahoe_results/external/prism_figshare_9393293_files.json`);
the drug matching of inventory (d) reused verbatim (`match` in `step15_coverage_inventory_d.json`; no re-matching);
curated targets as current symbols (`step15_targets_curated_hgnc.csv`); STRING v12.0 files (fetch record written by
`fetch_string.py`); CPTAC tables (`cptac_fetch_record.json`); HGNC 2026-01-06 (md5 pinned in `harmonise_symbols.py`).

## 4. Methods

### 4A. Utility against PRISM sensitivity (item 4)

**Label.** For line ℓ (one of the 10 PRISM lines) and drug d: the median AUC over every PRISM secondary-screen row of
ℓ's DepMap id whose `broad_id` is among d's matched ids and whose `passed_str_profiling` is true. Lower AUC = more
sensitive. Pairs without such a row have no label.

**Features.** The 200-gene effect vector of (ℓ, d) at 5 µM, standardised per gene with the mean and standard
deviation of the training pairs' measured effects (statistics fixed by the training set; applied unchanged to every
submission's predictions).

**Predictor f.** Ridge regression (scikit-learn `Ridge`) from the standardised effect to AUC, fitted on the split's
**training pairs**: training drugs of the split that are matched to PRISM, in the 10 lines, with a label and with a
5 µM condition in the training file, using the *measured* training effects (per-condition mean treated minus mean
paired control, averaged over plates, the ridge-train construction of Step 15b). Alpha from {0.1, 1, 10, 100, 1000} by
the lowest mean squared error in 5-fold `GroupKFold` over training drugs; the chosen alpha is recorded. f is fitted
once per split and never sees a held-out drug or a prediction.

**Score.** For submission s and line ℓ: ρ_ℓ(s) = Spearman correlation, over the labelled held-out drugs of ℓ, between
f(predicted effect of (ℓ, d) under s) and the AUC label. U(s) = mean of ρ_ℓ(s) over the lines with at least 8 labelled
held-out pairs. Positive U = the predicted effects rank the held-out drugs in the measured direction. A submission whose
predicted effects are identical for every drug of a line (control-paste: zero effect; mean-response: one delta per
line) has no ranking; its ρ_ℓ is set to 0 by definition and the fact is stated with the result.

**Comparators, all scored by the same f and U.** Ceiling: the measured held-out effects (what f achieves on real
data). Floors: control-paste and mean-response (U = 0 by definition). Linear: ridge-train (co-primary) and ridge-S7.
Two further references outside the "effect → f" route: (i) **QSAR**, a ridge from Step 15b's compound features
(the harness's fingerprint at 5 µM, line one-hot, fingerprint × line; the ridge-train design matrix) directly to AUC,
fitted on the same training pairs with the same alpha grid and grouped CV — the standard baseline that uses no
expression at all; (ii) **effect size**, the Spearman correlation per line between the negative L2 norm of the
predicted effect and AUC (no fitting) — tests whether sensitivity is carried by magnitude alone.

**Secondary read (descriptive).** Within-drug across-line Spearman (over the lines with labels, drugs with ≥ 6 lines),
averaged over held-out drugs: whether the predictions say *which line* is sensitive to a drug. With 10 lines this is
reported, not gated.

### 4B. Target discrimination in a protein network (item 8)

**Network.** STRING v12.0, *Homo sapiens* (9606): `protein.links` filtered to combined score ≥ 700 (high confidence),
undirected, unweighted; nodes restricted to the largest connected component. Proteins are named by STRING's
`preferred_name`; a current HGNC symbol is mapped to a node by preferred name, else through STRING's aliases file;
unmapped symbols are counted and listed.

*Pre-run correction, 2026-10-05 (before `fetch_string.py` exists or runs):* STRING's current release is 12.5, not 12.0
(download page checked on 2026-10-05). The files fetched are the 12.5 release (`9606.protein.links.v12.5.txt.gz`,
`9606.protein.info.v12.5.txt.gz`, `9606.protein.aliases.v12.5.txt.gz`; CC BY 4.0) and "v12.0" above reads 12.5;
the score threshold, the node set and everything else are unchanged.

**Sets.** For held-out drug d of a split with ≥ 1 curated target mapped to a node: T_d = its mapped targets. For line
ℓ (all 14) and submission s: S_dℓ(s) = the 20 network-mapped panel genes with the largest |predicted effect| at 5 µM
(plate-averaged, seed-averaged). Candidate target sets for d: T_e of every other held-out drug e of the split with
mapped targets and **no target in common with d**; a drug with fewer than 10 candidates is excluded and counted.

**Score.** Closest-distance proximity (Guney et al. 2016): d_c(S, T) = mean over g ∈ S of the shortest-path length
from g to the nearest member of T. TDS_dℓ(s) = 1 − (number of candidate sets with d_c strictly smaller than d_c(S_dℓ,
T_d) + ½ × number of ties) / (number of candidates), in [0, 1], chance 0.5 (a drug's own targets no closer to its
predicted response genes than another drug's targets). Per drug: mean over lines; per split: mean over drugs.
Comparators: ridge-train, ridge-S7, mean-response (one gene set per line, so a drug's TDS measures only how often
the line's shared response happens to sit near its targets) and the ceiling from the measured effects' top-20 genes.
Descriptive: TDS for brain versus gut lines; per drug.

### 4C. CPTAC tumour layer (items 5, 6, 7), descriptive

Tumour samples only (proteomics sample ids ending in `.N` are normals and are dropped). Symbols on both sides are
the harmonised current symbols of step 3 (`step15_panel_harmonised.csv`; the CPTAC column mapping of
`harmonise_symbols.py` re-applied).

**C1, target presence (item 5).** For every drug with curated targets and each cohort: the number of its targets in
the proteome table and, per target, the fraction of tumours with a non-missing value; a drug is flagged
"target protein detected" in a cohort when at least one target is non-missing in ≥ 50 % of that cohort's tumours.
Brain lines read the GBM flag, gut lines the COAD flag (no cohort for the two gastric lines; stated).

**C2, mRNA–protein concordance (item 6).** For every panel gene present in both the proteome and the transcriptome of
a cohort (125 GBM / 91 COAD): Spearman correlation across the tumours with both measurements (99 / 96), over the
tumours where both values are non-missing (≥ 30 required, else "not testable"); Benjamini–Hochberg over the genes
tested in that cohort. Label: "protein-reflected" if FDR < 0.05 and ρ > 0; "transcript-only" otherwise; panel genes
without a protein measurement in the cohort are "no protein measured" (the 43 non-protein-coding genes by HGNC class
always fall here). Summary per cohort: number of genes per label, median ρ.

**C3, baseline resemblance (item 7).** For each line: the mean log-normalised expression over the panel of its distinct
vehicle cells (by barcode) in the Step 15a control files of `drug_split_1` (train and test pooled), and for each
cohort the median tumour RNA value per gene over its tumours; r_cohort(line) = Spearman across the panel genes present
in that cohort's transcriptome (192) between the line's profile and the cohort median. Reported per line: r_GBM, r_COAD
and the difference r_GBM − r_COAD. The expected pattern (brain lines positive, gut lines negative) is a sanity check
on tissue identity over 192 genes, nothing more; no subtype assignment is attempted in 15d.

## 5. Endpoints

All intervals: drug-clustered percentile bootstrap over the split's held-out drugs that enter the endpoint, 5,000
replicates, `RandomState(0)`, 97.5 % per split, as Steps 14–15c; in every replicate the resampled drug set is applied
to both members of a contrast, lines with fewer than 8 labelled pairs (A) are skipped in that replicate, and
per-condition quantities are fixed under resampling. Split 1 is read first (development); splits 0 and 2 are
confirmation; categories require all three.

- **Primary, gated (Q1, layer A):** U(recipe) − 0, per split.
- **Co-primary, gated (Q2, layer A):** U(recipe) − U(ridge-train), per split.
- **Secondary, reported, not gated:** U of every submission, of the ceiling, of QSAR and of effect size, each minus 0;
  recipe − ceiling; recipe − QSAR; recipe − effect size; guidance w = 2 − recipe; attn_cross − mlp_ctrl
  (exploratory); per-seed U and the worst seed / worst split; the chosen alphas; the within-drug across-line read;
  layer B: mean TDS of every submission minus 0.5, recipe − ridge-train, recipe − ceiling, brain versus gut, per
  drug; layer C: the tables of 4C.

*Pre-final additions, 2026-10-05 (after the development run of layer A on `drug_split_1`, commit 515de10, and before
layer B's development run or any final run). Exploratory, not gated, reported next to the pre-registered quantities;
the endpoints, the categories and the pre-registered f are unchanged.* The development run showed two things the
design had not anticipated: 10 of the 27 PRISM-labelled held-out drugs of split 1 have no 5 µM condition in the Tahoe
plates at all (the dose rule of §3 kept 152 of 253 labelled pairs and 17 of 27 drugs, and 61 training drugs), and f
chose the upper bound of its alpha grid. The scorer therefore also reports: (a) a *highest available dose per drug*
variant — every drug, training and held-out alike, enters at the highest dose level at which it has conditions in the
PRISM lines, with f, QSAR and the standardisation refitted on those training pairs — with U for recipe, ridge-train,
ceiling, QSAR, effect size and measured effect size and the contrasts recipe − 0, recipe − ridge-train, ceiling − 0,
QSAR − 0, recipe − QSAR, measured effect size − 0, recipe − ceiling, each with the same bootstrap; (b) f refitted on the
pre-registered pairs with the alpha grid extended by {10⁴, 10⁵}, same quantities; (c) the cross-validated R² of f and
QSAR on their training pairs (1 − CV MSE / label variance), so the reader can see how much skill the instrument has
before judging any submission; (d) the number of labelled held-out pairs lost by the 5 µM rule and each held-out
drug's available dose levels. None of this changes section 6.

*Pre-run addition for layer B, 2026-10-05 (before its development run):* the same dose limitation applies to layer B, so
its scorer also reports the highest-available-dose-per-drug variant of (a) for the TDS of recipe, ridge-train and the
ceiling (each minus 0.5, recipe − ridge-train, recipe − ceiling), exploratory, next to the pre-registered 5 µM values.

## 6. Outcome categories for layer A, fixed now (precedence 1 → 2 → 5 → 3 → 4, coded in the scorer)

1. **SENSITIVITY INFORMATION BEYOND LINEAR** — primary above zero on all three splits and co-primary above zero on
   all three.
2. **SENSITIVITY INFORMATION, NOT BEYOND LINEAR** — primary above zero on all three; co-primary not above zero on at
   least one.
3. **NO SENSITIVITY INFORMATION** — primary not above zero on at least one split (and not category 5); the thesis
   then says the predictions' drug-specific direction (Step 15c) does not translate into a usable sensitivity ranking
   on this label set.
4. **PARTIAL** — a gated contrast resolved on one or two splits only; reported per split.
5. **UNRESOLVED** — a gated contrast includes zero on all three splits. Not equivalence; the ceiling's own interval
   is reported next to it, because if the ceiling is unresolved the label set cannot resolve any submission.

Layers B and C carry no category; their results are reported as pre-registered secondary and descriptive outputs.

## 7. Expectation, recorded in advance

Power first: 25–33 labelled held-out drugs per line gives a per-line Spearman with a standard error near 0.2, and the
10 lines share the same drugs, so I expect the 97.5 % interval on U to be roughly ± 0.2 to ± 0.25 wide. Only effects
larger than that resolve.

Layer A: I expect the **ceiling** (measured effects through f) above zero on all three splits, at U ≈ 0.3–0.5,
because a strong 24-hour transcriptional response and loss of viability go together. I expect the **recipe** at
roughly a third to a half of the ceiling, U ≈ 0.1–0.25, hence **category 4 or 5** — above zero on at most two splits.
I expect ridge-train close to the recipe (the shared-response part of the effect carries much of the sensitivity
signal, and ridge-train has it), so the co-primary unresolved. I expect QSAR to be the strongest reference after the
ceiling, and effect size alone to be positive — it is the simplest form of the "strong response ↔ sensitive"
relation. If the ceiling itself is unresolved, the honest statement is that 253–303 pairs per split cannot test
utility, and the step reports that.

Layer B: I expect the recipe's mean TDS slightly above 0.5 (0.52–0.58), with intervals that include 0.5 on at least
one split (45–56 drugs with targets per split, 20 of 200 panel genes), and ridge-train at 0.5. I record no expectation
for brain versus gut.

Layer C: I expect targets of most annotated drugs to be detected as proteins (the inventory already shows 238 / 282 in
GBM and 210 / 282 in COAD have a quantified target); a minority of protein-coding panel genes to be
"protein-reflected" at FDR 0.05 with 99 / 96 tumours; and the sign pattern of C3 to hold for most lines. If C3 fails
for a line, that is reported as a limitation of the panel, not of the line.

## 8. What this step cannot establish

- **A better model.** It reads frozen predictions; it changes none of them.
- **Sensitivity for the four lines absent from PRISM** (COLO 205, SW1417, KATO III, SNU-1) or for drugs outside the
  135 matched; layer A speaks for 10 lines and ≈ 25–33 held-out drugs per split.
- **A causal link between predicted response genes and targets.** Layer B is a proximity rank over a 200-gene panel
  of highly variable genes in a generic protein network; it cannot say which gene mediates what.
- **Patient-level claims.** CPTAC is bulk tumour with stroma, without drug exposure; layer C annotates the candidate
  list and the panel, it validates no prediction.
- **Independence from Step 15b/15c.** Same arrays, same splits; the labels are new, the predictions are not.
- **The ceiling is not an upper bound on any method.** It is what *measured* Tahoe effects achieve through the same f;
  a different predictor or label (IC50, GDSC2) could move it.

## 9. Analysis plan and provenance

Four scripts, all committed before they run, all CPU, all never overwriting, all stopping on any missing input, hash
mismatch or failed reproduction:

- `fetch_string.py` — downloads the STRING v12.0 9606 links, protein info and aliases files to
  `/scratch/ni2071/Diffusion/tahoe/external/string/`, records names, sizes, sha256 and the filtered network's node and
  edge counts in `tahoe_results/external/string_fetch_record.json`.
- `score_step15d_utility.py` — layer A. Guards: this document committed and unchanged in section count ≥ 9; the
  script committed and clean; Step 15b arrays hash-checked against `provenance/s15ev_*.json`; baseline reproduction to
  1e-9 against `step15_model_per_condition.csv`; PRISM files hash-checked against the figshare record; exactly three
  dose levels in the test file and the 5 µM level identified as the highest; f's training pairs contain no held-out
  drug (asserted by name and by structure); control-paste and mean-response give identical predictions within every
  line (asserted, then U = 0). Outputs `tahoe_results/step15d_utility[_development].json`, `_long.csv` (every U and
  contrast), `_pairs.csv` (every labelled held-out pair with its label and each submission's f value).
- `score_step15d_targets.py` — layer B. Guards as above plus the STRING fetch record; outputs
  `tahoe_results/step15d_targets[_development].json`, `_per_drug.csv`.
- `inventory_step15d_cptac.py` — layer C; CPTAC tables hash-checked against the fetch record; outputs
  `tahoe_results/step15d_cptac_targets.csv`, `step15d_cptac_concordance_<cohort>.csv`, `step15d_cptac_alignment.json`.

Order: `fetch_string.py` → layer C (no split dependence) → layer A development on `drug_split_1` → layer B development
on `drug_split_1` → the findings document's development section → both final runs (all three splits) → categories →
findings. Logs under `tahoe_results/step15d/`.

- Step 15b records: commit 6b40c72; Step 15c records: commit 24c16d8; coverage and provenance: commits ea7acf0,
  e6cbd2c, d9120cd.
- Pre-registration commit: [hash]. Script commits: [hash].
