# Pre-registration — Step 22: can an independent method tell which compound a generated population came from? RNA fingerprinting of the recipe, full OT and *m* = 32 on all three splits

Date: 2026-10-13. Written before any Step 22 matrix, dictionary or mapping exists. Inputs: the stored real cells,
conditioning vehicle cells and generated cells of the Step 18 re-evaluations of the recipe (`s18r_*`), the Step 18
full-OT runs (`s18_*`) and the Step 21 *m* = 32 runs (`s21_*`), every array hash-checked against its provenance by the
Step 18 and Step 21 loaders; the Step 15b per-condition record (`tahoe_results/step15_model_per_condition.csv`) for the
annotations; the data record (`tahoe_results/step15_data_record.json`); the training files, read only to rebuild the
ridge-train reference by the Step 15b code path. Toolchain: `provenance/R_env/` (commit 312cc37; R 4.5.3, Seurat 5.5.1,
presto 1.1.0 at b5df6ee, RNAFingerprinting 0.0.0.9000 at 4004c8d). Section numbers refer to this document unless
another document is named.

## 1. Question

Every endpoint so far is ours: DE-overlap, energy, PDS, MRA and tracking are computed by our scorers against the real
mean profiles. Grabski et al. (2026) published a method, RNA fingerprinting, that learns a *fingerprint* for each
perturbation from real cells and then asks, for a query cell or a query population, which perturbation it carries,
with a Bayesian credible set and a Bayes factor. It is an independent instrument, built for exactly this question,
with its own statistical model.

Step 22 asks: **when the authors' package, unmodified, learns the fingerprints of the held-out compounds from one half
of the real cells, does it recognise those compounds in the cells our model generates — and more often than in the
compound-blind reference, a vehicle cell shifted by the line's mean response?** Two levels: a generated population
(group level) and single generated cells (cell level). Secondary: does pairing the training cells by optimal
transport (full OT, *m* = 32), which makes a generated cell track its vehicle cell (Step 18 §4.4), lose this
identity?

No model is trained or chosen here. The step adds an external check of the compound signal that Steps 15b–21 measured
with our own endpoints.

## 2. Inputs

Per split (`drug_split_1`, `drug_split_0`, `drug_split_2`; 2,899 / 3,115 / 2,853 held-out conditions, 75 held-out
structures, 14 lines, 42 (line, dose) contexts; Step 15b findings §2):

- **Real cells.** The stored real cells of every condition (the harness draw, cap 200; 43–200 per condition), bitwise
  identical across all runs (asserted, as Steps 18–21).
- **Vehicle cells.** The conditioning control cells stored with every run (`ctrl`): the vehicle cells of the condition's
  line and plate that each generated cell was conditioned on; identical across arms per condition (asserted).
- **Arms**, three seeds each, cells per condition equal to the stored real cells: **recipe** (Step 18 re-evaluations),
  **full OT** (`ot_pair`, Step 18), ***m* = 32** (`mb32`, Step 21). *Note added 2026-10-13, before any run:* on
  `drug_split_1` the *m* = 32 arrays are Step 20's (`s20_drug_split_1_mb32_seed_<k>`, loaded by
  `score_step20_tahoe.load_eval20` against the split-1 *m* = 32 pin), because Step 21 trained *m* = 32 on splits 0
  and 2 only.
- **References**, rebuilt by the Step 15b code path (`score_step16_tahoe.rebuild`) and asserted to reproduce the Step
  15b per-condition record (DE-overlap and DPC-50 to 1e-9, as Step 18): cells = conditioning vehicle cells + delta,
  clipped at 0. **control-paste** (delta 0: the vehicle cells themselves), **compound-blind mean** (`mean_response`:
  the line's mean measured response over all its held-out conditions), **ridge-train** (fitted on the training
  compounds) and **ridge-S7** (fitted within the held-out compounds, structure-grouped 5-fold).
- **Annotations.** Line, plate, structure (SMILES) and dose from the Step 15b per-condition record, in the arrays'
  order (asserted).

## 3. The method, and how it is applied here

### 3.1 The package, unmodified

`RNAFingerprinting` (github.com/satijalab/rna-fingerprinting, commit 4004c8d), the authors' implementation of Grabski
et al. (2026), called through its two exported entry points with their default parameters, as the authors state they
did for every analysis in their paper (their Quantification and Statistical Analysis section):

- `LearnDictionary(object, perturbation_meta, control_class, features)`: for each perturbation with at least 10 cells,
  a baseline is fitted by PCA of the control cells (30 components, `irlba`), every cell's residual after baseline
  subtraction is taken, the perturbation's fingerprint is the first principal component of its residuals, sign-fixed
  against a Wilcoxon statistic (`presto`), with the iterative outlier removal (cells at ≥ 10 MADs) and the
  low-signal filter (a fingerprint is kept only if its scores separate perturbed from control cells, Wilcoxon
  p < 0.05, and at least 2 of its 10 largest-weight genes are differentially expressed at p < 0.05). Returns the
  fingerprints with per-entry means and standard deviations. Seed `seed.use` default 1448145.
- `FingerprintCells(object, condition_meta, control_class, dictionary, group_level, top = 100)`: the query cells'
  residuals after baseline subtraction (PCA of the query's own control cells), regressed on the fingerprints with the
  package's SuSiE variant (L = 10, estimated prior, 95 % credible sets with purity ≥ 0.4, sets with log Bayes factor
  below 10 % of the largest dropped, 50 resamples of the fingerprints to carry their uncertainty, joint fingerprints
  proposed for correlated sets). At cell level, per-cell outlier terms (the 5 largest residuals and any residual ≥ 10).
  At group level, residuals are summed within each condition before the regression, without outlier terms. Returns,
  per cell or per condition, the top credible set, its log Bayes factor, all credible sets, their Bayes factors and the
  sign of each member.

Nothing in the package is edited. The only choices made here are what goes into the Seurat object and which labels
are used.

*Note added 2026-10-14, after the development run stopped (job 18786995) and before any result was scored.* The
package's cell-level mapping crashed on three cells of one condition of the compound-blind mean set. Diagnosis (jobs
18790851, 18791328, 18791538; `diag_step22_cell_error.R`, `diag_step22_cell_trace.R`): for those cells the package
proposed a joint fingerprint and refitted SuSiE with it; the refit found no effect at all (every prior variance 0), so
its list of credible sets was empty, and the joint-refit branch, unlike the first pass, has no guard for that case
and calls `calibrate()` with nothing to calibrate. One guard was added to that branch, returning "unassigned" — the
package's own output whenever a fit yields no credible set (its first pass, and its joint path after calibration).
The guard is the unified diff `provenance/R_env/rnafingerprinting_classify_guard.patch` against commit 4004c8d, applied
to the clone and reinstalled; the R stage refuses to run unless the installed package carries it and records the
patch's sha256 in `r_session.txt`; the scorer refuses a split whose record lacks that line. The guard changes no
answer of any cell that did not crash, so the answer files written before the crash are kept and the job resumes.
Every other statement of this section stands.

### 3.2 Input layer (the one deviation from the authors' pipeline, disclosed)

The package reads the `scale.data` layer of an assay named `SCT`, which in the authors' pipeline holds SCTransform v2
Pearson residuals computed from UMI counts. Our model generates log-normalised expression on the 200-gene panel, not
counts, so SCTransform cannot be applied to generated cells, and real and generated cells must enter through the same
transform. Here the `SCT` assay's `scale.data` layer is filled with **per-gene standardised log-expression** (centred
and scaled, clipped at ±10): Seurat's standard `ScaleData` transform (Hao et al. 2024), which, like a Pearson
residual, puts every gene on a unit-variance scale so that the package's ≥ 10 outlier rule and PCA steps apply as
designed. The means and standard deviations are computed once per split on the dictionary cells (§3.3) and applied
unchanged to every query cell; a gene with zero standard deviation stops the run (none is expected on a panel of
highly variable genes). The real split-half ceiling (§3.3) measures what the method achieves on this input, so the
deviation is calibrated, not assumed harmless.

### 3.3 Dictionary and queries (per split)

All draws use `numpy.random.RandomState` with the seeds given; every draw is written to the manifest.

- **Real-cell halves.** For each condition *i* with *n_i* stored real cells, a permutation (one `RandomState(22)` per
  split, advanced over conditions in the arrays' order); half **A** = the first ⌊*n_i*/2⌋ cells (≥ 21), half **B** =
  the rest (≥ 22).
- **Vehicle halves.** The unique rows of all conditioning vehicle cells of the split (`numpy.unique` on rows), permuted
  with `RandomState(23)`; the first half to the dictionary (**control-A**), the rest to every query (**control-B**).
  Each half must hold at least 30 cells (the package's minimum).
- **Dictionary.** One per split: half A of every condition labelled with its structure (`P001`–`P075` in sorted-SMILES
  order, because the package rewrites `,`, `;` and `_` in names and splits joint names on `_`; the map is in the
  manifest) plus control-A labelled `control`; `features` = all 200 genes. The package's low-signal filter is left on:
  **K** = the number of fingerprints kept, and the filtered structures, are recorded. A condition is **covered** if its
  structure is in the dictionary; endpoints are computed over covered conditions, and the coverage (structures,
  conditions) is reported.
- **Query sets**, each a Seurat object holding the set's treated cells (labelled with their condition id and structure)
  and control-B labelled `control`. For every condition, each set contributes exactly |B_i| cells: for **real-half**
  the half-B cells themselves; for the arms and references, |B_i| of the condition's cells drawn with a fresh
  `RandomState(24)` per set, advanced over conditions in order. Sets: `real_half`; `recipe`, `ot_pair`, `mb32` (seeds
  0–2, one set each); `control_paste`, `mean_response`, `ridge_train`, `ridge_S7`. Fourteen sets.
- **Group level**: every set, `group_level = TRUE`, one answer per condition.
- **Cell level**: `group_level = FALSE` on the first **C_cell = 10** of each condition's |B_i| query cells (a subset of
  the group-level cells), for `real_half`, the nine arm sets, `control_paste` and `mean_response` (twelve sets; the
  ridges are group level only, for cost). One answer per cell.
- Group-level and cell-level answers are written per set as CSV (§9) and scored in Python.

## 4. Endpoints

Per split, 97.5 % structure-clustered percentile bootstrap intervals (5,000 replicates), generator reset to
`RandomState(0)` before every contrast (Step 20 convention), per-condition values fixed under resampling, paired by
condition; arms per seed, then averaged per condition. Conditions not covered by the dictionary are excluded from every
contrast (they carry no fingerprint to hit; they are reported in §4.4).

**Hit.** For a query (a condition at group level, a cell at cell level): hit = 1 if the true structure is a member of
the *top* credible set *with sign +*, else 0 (an unassigned query is 0). Per condition: the hit itself at group level;
the mean hit over its C_cell cells at cell level. **Hit rate** = mean over covered conditions.

**Chance floor** of a set: per query, |top credible set| / K (0 if unassigned); per condition as above; the mean over
covered conditions. It is the hit rate the set would reach if its true structure were a random one of the K.

*Amendment 2026-10-14, before any real run, after the mock run (job 18785013).* On the synthetic split the
compound-blind reference was answered with credible sets of about 7 of the 12 structures, so its true structure was
often "in the top set" without being singled out, while the real and generated cells were answered with sets of about
one. A hit inside a large set is weak evidence, and the raw hit rate rewards it as much as a precise one. Every gated
contrast (G1, G2, Q1, Q2) is therefore computed on the **chance-corrected hit**: per query, hit − chance, where
chance is the number of members of the top set carrying sign + divided by K — the probability that a random one of
the K structures would score a hit as defined (a hit requires sign +; on the mock the vehicle cells, when assigned,
received near-complete sets with random signs, which the sign-blind count |top set| / K would have scored as −0.2
instead of 0). The chance-corrected hit is 0 for an unassigned query, 1 − 1/K for a correct singleton, 0 for a
correct set that holds every structure with sign +, and negative for a wrong set; it is averaged per condition
exactly as the hit. G1 becomes "the chance-corrected hit of real_half above zero", which is what it already was. The
raw hit rate, the sign-blind floor |top set| / K and their contrasts are reported as secondaries. Nothing else changes.

### 4.1 Gate, per split (both must hold on every split, else UNINFORMATIVE)

- **G1**: hit(real_half) − chance(real_half), group level, interval above zero: the real cells are recognised above
  chance.
- **G2**: hit(real_half) − hit(mean_response), group level, interval above zero: the real cells carry compound
  identity beyond the line's shared response. If the real cells do not, nothing can be asked of the model.

### 4.2 Gated primary, per split

- **Q1, group level**: D_G = hit(recipe) − hit(mean_response). Holds if the interval is above zero.
- **Q2, cell level**: D_C = hit(recipe) − hit(mean_response). Holds if the interval is above zero.

### 4.3 Secondary (intervals, not gated)

- **Pairing**: full OT − recipe and *m* = 32 − recipe, group and cell level.
- **Distance to the ceiling**: real_half − recipe, group and cell level.
- **Other references**: recipe − control_paste (group and cell), recipe − ridge_train and recipe − ridge_S7 (group).
- **Unassigned fraction** per set and level; **credible-set size** (mean |top set|) per set; **sign-agnostic hit** (the
  true structure in the top set with either sign) and **any-set hit** (in any credible set, sign +), per set, as hit
  rates with the Q1/Q2 contrasts repeated on them.
- **K**, the filtered structures, and the coverage.

### 4.4 Exploratory (descriptive, no interval)

Hit rate per structure (which compounds the method recognises in real and in generated cells); hit rate by dose level;
what uncovered conditions map to (unassigned, or another structure); the top-set log Bayes factors of hits and misses.

## 5. Verdict rule (coded in the scorer)

G = G1 and G2 hold on all three splits; E = Q1 holds on all three; P = Q2 holds on all three.

0. **UNINFORMATIVE** — not G. Everything is reported descriptively; no claim about the model is made.
1. **SIGNATURE CONFIRMED AT CELL LEVEL** — G, E and P.
2. **SIGNATURE CONFIRMED AT GROUP LEVEL ONLY** — G and E, not P.
3. **NOT CONFIRMED** — G, not E.

The categories are exclusive. If any Q1 or Q2 interval is below zero on any split, the verdict carries the clause
"with a reversal on <split, level>". "Includes zero" means unresolved, never "equivalent" or "absent". Categories 1
and 2 each require every component test to pass; a test that rejects only when every component rejects has level α
whenever each component has level α (the intersection–union principle, Berger 1982), so no multiplicity correction is
applied to the category claims.

## 6. Splits and discipline

No model is trained. The pipeline (prep → R mapping → scorer) runs first on the development split alone
(`drug_split_1`); its outputs are read to check that the pipeline runs end to end, and any defect found is fixed in the
scripts before the confirmation run and recorded in §9 by a dated paragraph with the commit. Endpoints, gate, seeds,
caps, labels and the verdict rule are not changed after any development output. The confirmation run then processes
`drug_split_0` and `drug_split_2` with the frozen scripts, and the final scorer runs on all three splits; the verdict
uses all three, and the development split's values are reported as development evidence, as in Step 15b. The scorer
refuses to run uncommitted, refuses unless this document is committed with at least 10 sections, refuses if any
mapping file or manifest hash differs from its record, and reproduces the Step 15b reference record before printing
any Step 22 number. This is the ninth use of the confirmation splits (Steps 15b, 15c, 15d, 15e, 16, 18, 19, 21, 22).
Their test data are read in Step 22 only through the stored real cells of the certified arrays.

## 7. Expectation (written before the run)

Category **2, SIGNATURE CONFIRMED AT GROUP LEVEL ONLY**, held with low-to-moderate confidence.

- **Gate.** Expected to pass. The real cells discriminate compounds within a context well above chance (the replicate
  PDS ceiling in Step 15c), and the authors report strong group-level recognition on drug data (sci-Plex). Risk: a
  200-gene panel, fingerprints pooled over 14 lines and several doses, and many weak compounds; the low-signal filter
  may remove a large share of the 75 structures. K is a measurement; no value is projected.
- **Q1.** Expected to hold. The recipe's mean profiles carry compound-specific signal within contexts (PDS 0.718 and
  0.714 on splits 0 and 2, Step 21 findings §2, against the chance value 0.5 that any compound-blind prediction
  gets, since it ranks every candidate alike). Risk: on Tahoe the recipe's DE-overlap is
  below the compound-blind mean's (−0.045, −0.040, −0.050; Step 15b), so if the method's fingerprints are dominated by
  genes of the shared response, the compound-blind mean may be recognised as often as the recipe. The weakest test is
  Q1 on whichever split has the smallest K.
- **Q2.** Not expected to hold on every split. Cell-level mapping of 200 genes with outlier terms is the noisier task,
  and the authors' own cell-level accuracy falls with signal strength.
- **Secondary.** Full OT below the recipe at the cell level (its cells track their vehicle cell: cosine +0.30 against
  +0.015, Step 18), *m* = 32 between the two; at the group level the differences small, in line with PDS costs of
  −0.038 / −0.028 / −0.037 (full OT) and −0.019 / −0.013 / −0.021 (*m* = 32, Steps 20–21).

## 8. Limits

- **Input layer.** Standardised log-expression in place of SCTransform residuals (§3.2); the real split-half ceiling
  calibrates it, but the method's published accuracy was obtained on its own input.
- **One fingerprint per compound**, pooled over lines, plates and doses. Line-specific responses blur it; the ceiling
  carries the same blur.
- **Recognition, not prediction.** The dictionary is learned from real cells of the held-out compounds; the model never
  saw them. The test asks whether the generated cells look like the real response of their compound, which is the
  intended use of such a model (screening a compound's generated response against known responses).
- **Credible-set hits reward large sets**; the chance floor and set sizes are reported, and the gate and Q1/Q2 are
  paired contrasts in which set size affects both sides.
- **C_cell = 10 cells per condition** at cell level, for cost; the group-level test uses all |B_i| cells.
- **Three seeds**; per-split 97.5 % intervals; "includes zero" is not "absent".
- **Versions.** Seurat 5.5.1 (the authors: v5.0); package commit 4004c8d (2025-09-21), presto b5df6ee; pinned in
  `provenance/R_env/`.
- **The method is linear and rank-one per perturbation** (the authors' own limitation); a compound whose response is
  not well summarised by one direction is harder to recognise for every set alike.
- **Ninth use of the confirmation splits.**

## 9. Implementation and records

- `prep_step22_fingerprint.py` (Python, `squidiff` env): loads the certified arrays through the Step 18 and Step 21
  loaders, rebuilds and asserts the references, draws the halves and query cells (§3.3), standardises, and writes per
  split under `tahoe_results/step22/<split>/`: one `float32` row-major matrix file and one metadata CSV per Seurat
  object (dictionary and fourteen query sets), the gene list, and `manifest.json` (every seed, draw size, the
  standardisation parameters, the structure-label map, sha256 of every matrix). `--mock` writes a synthetic split with
  planted fingerprints under `tahoe_results/step22_mock/` for the end-to-end test.
- `run_step22_fingerprint.R` (`/scratch/ni2071/conda_envs/R_env/bin/Rscript`): reads the matrices, builds the Seurat
  objects (`SCT` assay, `scale.data` layer), `LearnDictionary`, `FingerprintCells` at group level for every set and at
  cell level for the twelve sets, with `mc.cores` = the job's cores; writes `dictionary.csv` (structure, kept, cells),
  `dictionary.rds`, `map_<set>_group.csv`, `map_<set>_cell.csv` and `r_session.txt` (every package version), and
  prints timings.
- `score_step22_fingerprint.py` (Python): hits, chance floors, gate, contrasts, verdict; writes
  `tahoe_results/step22_analysis.json`, `step22_long.csv`, `step22_per_condition.csv`; never overwrites.
  `plot_step22.py` reads only those records.
- `run_step22_tahoe.sh`: `sbatch -p compute`, one task per split (prep, then R), 32 cores, 120 GB, 24 h; development
  on split 1 first; logs under `tahoe_results/step22_logs/`.
- Matrices and `dictionary.rds` are gitignored (sha256 in the manifest and in the R log); every CSV, manifest, log and
  record is committed. All four scripts and the launcher are committed, and the mock run has passed, before the
  development run.
- *Note added 2026-10-14, after the development run and before the confirmation runs (storage only, no analysis
  change).* The per-cell metadata CSVs of the matrices (`*_meta.csv`, one row per cell, about 40 MB each,
  regenerable by the prep, sha256 in the manifest) are gitignored like the matrices; the cell-level answer files are
  stored gzipped (`map_<set>_cell.csv.gz`, written by the launcher after the R stage); the R stage treats a gzipped
  file as existing when it resumes and the scorer reads either form. The development run of split 1 was committed in
  this form.

## 10. References

- Grabski IN, Blair JD, … Knowles DA, Satija R (2026). Mapping transcriptional responses to cellular perturbations
  with RNA fingerprinting. *Cell* 189, in press. doi:10.1016/j.cell.2026.09.017. Software: Grabski IN (2026), RNA
  Fingerprinting, github.com/satijalab/rna-fingerprinting (commit 4004c8d used here; the authors' reference 34 cites
  the Zenodo record doi:10.5281/zenodo.21464506).
- Wang G, Sarkar A, Carbonetto P, Stephens M (2020). A simple new approach to variable selection in regression, with
  application to genetic fine mapping. *J R Stat Soc Ser B* 82:1273–1300. doi:10.1111/rssb.12388. (SuSiE.)
- Hao Y, Stuart T, Kowalski MH, et al. (2024). Dictionary learning for integrative, multimodal and scalable
  single-cell analysis. *Nat Biotechnol* 42:293–304. doi:10.1038/s41587-023-01767-y. (Seurat v5; `ScaleData`.)
- Hafemeister C, Satija R (2019). Normalization and variance stabilization of single-cell RNA-seq data using
  regularized negative binomial regression. *Genome Biol* 20:296. doi:10.1186/s13059-019-1874-1; Choudhary S, Satija R
  (2022). Comparison and evaluation of statistical error models for scRNA-seq. *Genome Biol* 23:27.
  doi:10.1186/s13059-021-02584-9. (SCTransform v2, the authors' input layer, not used here; §3.2.)
- Korsunsky I, Nathan A, Millard N, Raychaudhuri S (2019). Presto scales Wilcoxon and auROC analyses to millions of
  observations. bioRxiv. doi:10.1101/653253. Baglama J, Reichel L (2005). Augmented implicitly restarted Lanczos
  bidiagonalization methods. *SIAM J Sci Comput* 27:19–42. (irlba.)
- Berger RL (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics* 24(4):295–300.
