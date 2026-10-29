# Step 23 — richer compound information: a model-free screen, then a model test (pre-registration)

Written 2026-10-26, before any Step 23 quantity was computed from Tahoe expression or response data. The compound
feature tables (§2) were built from chemical structure and ChEMBL only and read no expression data. After this file
is committed, changes are made only as dated paragraphs, each written before the data it concerns are read.

## 1. Question

The recipe identifies a held-out compound from its predicted response: PDS 0.7223 / 0.7181 / 0.7142 on splits
1 / 0 / 2 (Step 15c), against 0.4859–0.5400 for the fingerprint ridges. Its only compound input is a 1,024-bit FCFP4
fingerprint of the study SMILES (counter-ions included), scaled by log10(dose + 1). Step 16 added the compound's
curated protein targets and did not improve it (PDS +0.009 / −0.001 / +0.003). Step 23 asks whether chemical
information the fingerprint does not carry — atom identity, the arrangement of pharmacophore features across the
whole molecule, and whole-molecule physicochemical properties — tells us more about how an unseen compound changes
the cells. It also asks whether the drug's measured potency (IC50, Ki, Kd) tells the model how strong the response
will be.

The end point of the model is the single-cell response to an unseen compound. A compound representation can only
help that prediction if it carries information about the response, so the step has two stages:

- **Stage 1, the screen (this pre-registration's first run).** Model-free, CPU only, development split only. For each
  held-out condition, the measured responses of the k training compounds most similar to the held-out compound (same
  line, same dose) are averaged and offered as the prediction (k-nearest-neighbour transfer), and scored with the
  paper's own measures on the paper's own candidate sets. A representation that knows more about the response finds
  neighbours whose responses point the right way more often.
- **Stage 2, the model tests (§6; each run only if Stage 1 selects it).** The recipe retrained with the selected
  representation (§6.1) or with the drug's potency (§6.2) added to its compound input, against the recipe and a
  shuffled control, on all three splits, with the single-cell measure (energy distance) as a gated endpoint.

Potency is handled apart from the chemistry because it carries one number, how strong the drug is, and no
direction: it cannot say which genes move, only how much. Its gate is therefore on response size (§4.4), and its
model test is on the single cells and on response size (§6.2).

Why a screen first: each candidate costs 18 training runs (two arms × three seeds × three splits, as Step 16), and
there are five (four chemical, one potency). Why k-nearest neighbours and not a ridge: the fingerprint ridges sit at chance on PDS,
so a ridge null would say nothing; transfer from similar compounds needs no linear map from features to response
(the similarity principle of medicinal chemistry: similar structures tend to act alike; Maggiora et al. 2014).

## 2. Data, and what is read

**Compound tables** (`tahoe_results/step23_features/`, built by `step23_compound_features.py` from the Step 15 data
record and the ChEMBL_37 summaries of `step23_chembl_coverage.py` and `step23_chembl_targets.py`; no expression
data): 372 structures (the study's unit: canonical SMILES without stereochemistry, counter-ions kept) from 350 parent
molecules (largest organic fragment). For each structure: four fingerprints — FCFP4 as encoded (checked bit for bit
against the model's encoder), and FCFP4, ECFP4 and the Gobbi–Poppinger 2D pharmacophore fingerprint of the parent
(39,972 possible bits, median 480 on per structure); 49 numbers — 16 RDKit descriptors, 25 atom and bond counts, 8
pharmacophore-feature counts; and three ChEMBL potency summaries (pChEMBL: strongest on any human protein target,
strongest on a curated mechanism target, median growth inhibition over cell lines; missing for 60, 253 and 179
structures). Removing counter-ions changes the fingerprint of 65 structures.

How the requested properties enter: lipophilicity (Crippen logP and molar refractivity; Wildman and Crippen 1999);
hydrogen bonding (donor and acceptor counts, topological polar surface area; Ertl et al. 2000); size and shape
(molecular weight, heavy atoms, rotatable bonds, rings, aromatic rings, fraction sp3, Labute surface area; Labute
2000); electronic properties (Gasteiger partial charges, Gasteiger and Marsili 1980; E-state indices, Hall and Kier
1995; formal charges); atom level (element counts, hybridisation, charged atoms, ring sizes; atom identity, degree,
hydrogen count, charge and ring membership also enter ECFP4's atom invariants; Rogers and Hahn 2010); bond level
(double, triple, aromatic and conjugated bond counts); pharmacophores (feature counts, and the 2D pharmacophore
fingerprint: pairs and triplets of features with the bond distances between them; Gobbi and Poppinger 1998).
Not used, and why: chirality and bond stereochemistry (the study SMILES carry none; 4 of 375 table SMILES had any);
3D pharmacophores (they need stereochemistry and conformers).

Potency (pChEMBL = −log10 of the molar IC50, Ki or Kd, the three types collected here; Bento et al. 2014): one summary may become a model input —
the strongest pChEMBL on any human protein target, which 312 of the 372 structures have — through the share of the
target it would occupy at the dose used (§6.2). The other two summaries stay checks: potency on a curated mechanism
target is too sparse to be an input (119 of 372 structures), and growth inhibition in cell lines (193 of 372) is
itself a measured cell response, sometimes in the same lines (HT-29 for 60 structures); as an input it would blur
what the model predicts from. A model given potency predicts only for drugs that have been tested, which suits
repurposing known drugs and not new molecules.

**What Stage 1 reads.** Only the development split, `drug_split_1`: the Step 15b mean-vector files of the recipe and
base arms (hash-checked), the Step 15b and Step 15c per-condition records, the Step 15a test file's annotations (no
expression) and the Step 15a training file with its paired control file. Nothing from splits 0 and 2. As in Steps
15b–22, split 1's training compounds include compounds held out in splits 0 and 2; the separation that matters, the
held-out compounds of the confirmation splits, is kept.

**Scored set.** A held-out structure whose parent molecule also occurs among the split's training structures (a salt
or hydrate form of a training compound; same first 14 characters of the parent's InChIKey) is not scored, in either
stage: its nearest neighbour would be itself. The scorer computes this set itself from the feature tables; an
earlier, uncommitted check found 8 such structures on split 1. Their conditions stay in every candidate set as
competitors, so the candidate sets are exactly Step 15c's. Every other held-out condition is scored (expected: 67
structures).

## 3. Representations

Similarity between two structures: Tanimoto for fingerprints; for the numbers, minus the Euclidean distance between
z-scored values (mean and SD over the split's training structures; a missing value takes the training median; a
column constant among the training structures contributes nothing). A two-block representation ranks every training
structure by each block's similarity (rank 1 = most similar, ties averaged) and orders them by the mean of the two
ranks, so both blocks count equally whatever their scale; remaining ties go to the higher similarity of the first
block, then to the lower structure id. A one-block representation orders by similarity, ties to the lower structure
id.

Gated (§5):

- **R1** FCFP4 as encoded — the model's current input; the positive control.
- **R2** FCFP4 of the parent (counter-ions removed).
- **R3** R2 + ECFP4 of the parent (atom identity).
- **R4** R2 + 2D pharmacophore fingerprint of the parent, folded to 1,024 bits (bit i → i mod 1,024) —
  pharmacophore arrangement, in the form Stage 2 would give the model (39,972 inputs would add about 82 million
  parameters to the first layer).
- **R5** R2 + the 49 numbers (physicochemical properties, atom and bond counts, pharmacophore counts).

Exploratory (reported, never gated): E1 ECFP4 alone; E2 2D pharmacophore fingerprint alone (all 39,972 bits); E3
the 49 numbers alone; E3a the 16 descriptors alone; E3b the 25 atom and bond counts alone; E3c the 8 pharmacophore
counts alone; E4 R2 + the unfolded 2D pharmacophore fingerprint, with E4 − R4 showing what folding costs. Every
gated representation is exactly what Stage 2 would give the model.

References (not predictors; reported): **X1** response-profile neighbours — similarity is the cosine between the
held-out compound's measured responses in the other 13 lines (all doses) and each training compound's responses in
the same contexts, after removing each context's mean training response. It uses measured data of the held-out
compound, never from the line being predicted, and shows how far a perfect compound similarity could take this
probe. **X2** R2 + X1 — checks that the equal-weight combination of §3 can register an informative second block.

## 4. Stage 1 — the screen

**4.1 Transfer.** For a held-out condition (structure s, line L, dose d), the training structures with a training
condition at (L, d) are ordered by similarity to s (§3). The prediction is the unweighted mean of the measured
training responses of the first k of them. A training response is the mean treated expression minus the mean of the
row-paired control cells over the condition's cells — the ridge-train construction of Steps 15b and 15c — averaged
over plate copies. k = 10 primary, k = 5 secondary; neither was tuned.

**4.2 PDS (primary).** Step 15c's per-condition perturbation discrimination (`score_step15c_tahoe.pds_sets`, its
candidate sets: same line and dose, the other conditions of the same structure removed), with the transferred
response as the predicted effect; the mean over the scored set.

**4.3 MRA (secondary).** Step 16 §4.4: within each (line, dose) context, the Spearman correlation between the length
of the predicted response and the length of the measured response; the mean over the 42 contexts; a prediction of
constant length scores 0. On the scored set.

**4.4 Qp — does potency say something about response size that the model does not?** For each of the three
potency summaries, on the scored conditions of the structures that have a value: (a) the MRA of §4.3 with the
potency value in place of the predicted length, with the recipe's MRA on the same conditions beside it; (b) the
partial Spearman correlation between potency and the measured length, removing the recipe's predicted length (the
mean of its three seeds' lengths), within each (line, dose) context and averaged over the contexts. (b) is computed as
the correlation of rank residuals (the construction of Step 19), so it measures what potency adds beyond the model.
Within a context the dose is fixed, so ranking by pChEMBL is ranking by the occupancy of §6.2. (b) for the
protein-target summary is gated (§5, rule 5.4); everything else in Qp is descriptive.

**4.5 Other readings (descriptive, no interval).** DE-overlap and DPC-50 of the transferred responses (Step 15b
code). The recipe's PDS and MRA on the scored set. R1's and the recipe's PDS on the excluded structures (the size of
the salt-twin effect).

**4.6 Intervals.** Structure-clustered percentile bootstrap: 5,000 replicates, 97.5 % (1.25th and 98.75th
percentiles), clusters drawn exactly as `score_step15_tahoe.boot` draws them, from a fresh `RandomState(0)` for each
set of scored conditions, so all intervals on one set use the same resamples (and none depends on how many intervals
were drawn before it; Step 20 §6). Per-condition PDS is fixed under resampling (Step 15c); MRA is recomputed on each
resample (Step 16).

## 5. Stage 1 — gated contrasts and the selection rule

At k = 10, PDS, development split, scored set:

- **P** (positive control): R1 − 0.5.
- **Q1a** salt removal: R2 − R1.
- **Q1b** atom identity: R3 − R2.
- **Q1c** pharmacophore arrangement: R4 − R2.
- **Q1d** physicochemical properties: R5 − R2.

A contrast passes if its interval lies above zero. The chemistry route, applied mechanically:

1. If one or more of Q1b–Q1d pass, the block with the largest point estimate goes to Stage 2 (§6.1), and only that
   block.
2. Otherwise, if Q1a passes, Stage 2 tests the parent fingerprint alone (§6.1).
3. Otherwise the chemistry arms are not run. If P passed, the reading is "no added information detected by a probe
   that detects the fingerprint's own information"; if P failed, "screen uninformative" — the probe cannot read even
   the information the model is known to use, and a direct model test would need its own pre-registration.

A passing block goes forward whatever P's outcome: a gain over R2 is evidence by itself; P only qualifies a null.

The potency route, decided separately (rule 5.4): **Qp** — §4.4 (b) for the protein-target summary — passes if its
interval lies above zero; then Stage 2 also runs the potency arms (§6.2). Otherwise they are not run.

No secondary, exploratory or reference result changes these rules.

## 6. Stage 2 — the model tests (each run only if its route is selected in §5)

**6.1 Chemistry route (rule 5.1 or 5.2).** **Arms**, on the Step 15b recipe (adaLN + Gaussian flow matching,
cond_drop 0, mlp encoder, 100,000 steps, 200 genes, 50-step Euler sampling without guidance, 200-cell cap), changing
only the compound input:

- `chem` — the selected representation: FCFP4 of the parent, plus the selected block (ECFP4, 1,024 bits; or the 2D
  pharmacophore fingerprint folded to 1,024 bits; or the 49 numbers, z-scored with the split's training
  statistics), every block multiplied by log10(dose + 1) as the recipe's fingerprint is.
- `chem_shuf` (only when a block was selected) — the same width, but the block of another structure under one fixed
  permutation of the 372 structures (a cyclic shift of a `RandomState(0)` ordering; the number of structures that
  receive a block identical to their own is reported); the fingerprint stays the structure's own. It separates the
  compound's own block from width, scale and salt removal.

Seeds 0–2. **Development split 1 first** — a pipeline check and first read that cannot confirm, because the block
was selected on this split — **then confirmation splits 0 and 2**. The comparator is the recipe's existing certified
predictions (Step 15b). Scored set per split as in §2, from that split's own training structures.

**Gated, on confirmation splits 0 and 2** (97.5 % structure-clustered intervals as §4.6):

- **M1 primary** — PDS, `chem` − recipe.
- **M2 co-primary (single cells)** — energy distance between generated and real cells (Step 15b), `chem` − recipe;
  below zero = closer to the real cells.
- **M3 control** — PDS, `chem` − `chem_shuf` (not applicable under §5.2).

Secondary: DE-overlap, DPC-50 and MRA, `chem` − recipe and `chem` − `chem_shuf`; energy, `chem` − `chem_shuf`;
per-seed values; the split-1 values.

**Verdict**, first match:

1. BETTER COMPOUND SIGNAL AND BETTER CELLS — M1 above zero on both confirmation splits and M2 below zero on both.
2. BETTER COMPOUND SIGNAL — M1 above zero on both, M2 not above zero on either.
3. BETTER COMPOUND SIGNAL AT A SINGLE-CELL COST — M1 above zero on both, M2 above zero on at least one.
4. BETTER CELLS ONLY — M2 below zero on both, M1 not above zero on both.
5. NO CONFIRMED GAIN — otherwise.

Under 1–3, if M3 applies and is not above zero on both splits, the verdict carries the clause "not shown to come from
the compound's own features".

**6.2 Potency route (rule 5.4).** **Arms**, on the same recipe, changing only the compound input:

- `pot` — the recipe's input unchanged (FCFP4 as encoded × log10(dose + 1)) plus two numbers: the share of the
  target occupied at the dose used, θ = d / (d + 10^(9 − p)), with d the dose in nM and p the strongest
  protein-target pChEMBL — the occupancy relation of receptor theory with a slope of 1 (Clark 1926), taking IC50 as
  the affinity (Cheng and Prusoff 1973) — and a has-value flag. A structure without a value gets θ = 0 and flag 0,
  as Step 16 did for drugs without targets. θ is not multiplied by log10(dose + 1): it already contains the dose.
- `pot_shuf` — the same, but θ and the flag computed from another structure's value, under the same permutation
  as `chem_shuf` (the number of structures that keep their own value, or their own missing flag, is reported).

Seeds, splits, comparator and scored sets as §6.1. Gated, on confirmation splits 0 and 2:

- **P1 primary (single cells)** — energy distance, `pot` − recipe; below zero = closer to the real cells.
- **P2 co-primary** — MRA, `pot` − recipe; above zero = response size ranked better.
- **P3 control** — energy distance, `pot` − `pot_shuf`.

Secondary: PDS, DE-overlap and DPC-50, `pot` − recipe; MRA, `pot` − `pot_shuf`; per-seed values; the split-1 values.

**Verdict**, first match:

1. BETTER CELLS AND BETTER SIZE RANKING — P1 below zero on both confirmation splits and P2 above zero on both.
2. BETTER CELLS — P1 below zero on both, P2 not below zero on either.
3. BETTER CELLS AT A SIZE-RANKING COST — P1 below zero on both, P2 below zero on at least one.
4. BETTER SIZE RANKING ONLY — P2 above zero on both, P1 not below zero on both.
5. NO CONFIRMED GAIN — otherwise.

Under 1–3, if P3 is not below zero on both splits, the verdict carries the clause "not shown to come from the drug's
own potency"; under 4 the same clause applies, with MRA, `pot` − `pot_shuf`, in place of P3.

What cannot be fixed before Stage 1 — for either route, the harness change, its dry-run checks (input width,
parameter count, permutation hash, the count of unchanged blocks or values) and the job scripts — is committed with a
dated paragraph in this section before the first Stage 2 training run.

## 7. Expectations (written before any Step 23 response value)

Stage 1: P above zero but small (R1 PDS 0.52–0.58): the fingerprint ridges were at chance, and the model's 0.72 needs
a non-linear read. Q1a includes zero (65 of 372 fingerprints change, and the twins are not scored). Q1b–Q1d include
zero; R5 may sit below R2, since an equal-weight block with weak information dilutes the ranking. Expected outcome:
rule 5.3, no chemistry arms. X1 well above R1 (0.6–0.7); X2 above R2. Qp: (a) cell-line potency positive (MRA
about 0.1–0.3), protein-target potency weaker (0–0.15); (b) the protein-target partial small and positive (0 to
+0.15) with an interval that includes zero, so no potency arms; every interval wide (fewer than 67 structures with a
value). Stage 2, if run: category 5 on either route.

## 8. Guards and reproduction checks (all before any Step 23 value is printed)

1. This file, `score_step23_screen.py`, the scorers it imports (`score_step15_tahoe.py`, `score_step15c_tahoe.py`,
   `step14_metrics.py`), `step23_compound_features.py`, the data record, the Step 15b and 15c per-condition records
   and the three feature tables are committed and identical to HEAD (a staged but uncommitted edit also fails). The
   feature tables' recorded commit is an ancestor of HEAD and `step23_compound_features.py` is unchanged since it.
   No output exists. The scorer takes no options and reads no split other than `drug_split_1`.
2. Vectors hash-checked against provenance (`score_step15c_tahoe.load_vectors`); conditions, `true` and `C` identical
   across files; test-file annotations equal the Step 15b record.
3. Every held-out and training SMILES in the files is a SMILES of the data record (canonical or table form) and maps
   to exactly one structure id; the held-out ids equal the record's `held_structures`; FCFP4 as encoded equals the
   model's encoder for all 372 structures and equals the dose-scaled input of every held-out condition.
4. The ridge-train shift rebuilt from the training files, and the recipe, reproduce the Step 15c per-condition PDS to
   1e-9, with the same candidate-set sizes — a joint check of the training responses, the candidate sets and the PDS
   code.
5. The MRA implementation reproduces Step 16 §4.4 on split 1 to its four printed decimals: ridge-S7 −0.0499,
   ridge-train +0.1537, recipe per seed 0.3112 / 0.3237 / 0.3265 (mean 0.3205), base −0.0755.
6. The bootstrap reproduces Step 15c's split-1 Q1 interval for the recipe, +0.22234 [+0.18829, +0.25411], to its
   five printed decimals; and the fast (vectorised) replicate statistics equal those computed on the first three
   resamples written out in full.
7. Every held-out (line, dose) context has at least k training structures.
8. The partial Spearman of §4.4 (b), computed from rank residuals, equals the closed formula from the three pairwise
   Spearman correlations on every context where neither variable is nearly collinear with the third (|ρ| < 0.99), to
   1e-9; and its fast replicates equal those computed on the first three resamples written out in full. Checked on
   quantities that are not Step 23 results (the ridge-train shift's length against the measured length, beyond the
   recipe's).

## 9. Scripts, records, run order

1. Committed together, before any Step 23 response value: this file; `step23_chembl_coverage.py`,
   `step23_chembl_targets.py`, `step23_compound_features.py`; `score_step23_screen.py` and `run_step23_screen.sh`.
2. The feature tables are rebuilt under that commit and committed with the ChEMBL summaries (the per-molecule cache
   is ignored).
3. One run of the screen (CPU; `tahoe_results/step23_screen.json`, `step23_screen_per_condition.csv`,
   `step23_screen_summary.csv`; the log in `tahoe_results/step23_logs/`), committed.
4. Findings in `step23_findings.md`. No rerun with changed settings; a fault fixed after the run is reported as such,
   with both runs kept.

## 10. Limits

- A screen, not a test of the model. A null from neighbour transfer does not show that the model cannot use a block,
  and a block that helps the transfer may not help the model. Stage 2 is the test.
- Equal-weight rank combination is one way to join two blocks; a block with weak but real information can lower the
  combined ranking. X2 shows whether the combination can register an informative block at all.
- One split, about 67 scored structures: the intervals are wide. Five gated contrasts at 97.5 % each, without a
  multiplicity correction: a pass is a reason to run Stage 2, not a finding.
- Neighbours come from about 300 training structures; a held-out compound with no close analogue receives a generic
  prediction under every representation.
- No stereochemistry, no 3D, no learned molecular embeddings (graph networks, pretrained chemical language models);
  each of these would be a separate arm.
- Potency. ChEMBL values are sparse and come from different assays and targets. The strongest potency on any
  protein is crude: that protein may not drive the response in these cells, and drugs tested on more targets reach a
  higher maximum just by being tested more. The nominal dose in the medium is not the free concentration at the
  target; occupancy assumes a slope of 1 and takes IC50 as the affinity. The has-value flag can carry "well-studied
  drug" rather than potency; `pot_shuf` keeps the flags' overall frequency but not their pairing with the drug.

## 11. References

- Ertl P, Rohde B, Selzer P (2000). Fast calculation of molecular polar surface area as a sum of fragment-based
  contributions and its application to the prediction of drug transport properties. *J Med Chem* 43:3714–3717.
  doi:10.1021/jm000942e
- Bento AP, Gaulton A, Hersey A, et al. (2014). The ChEMBL bioactivity database: an update. *Nucleic Acids Res*
  42:D1083–D1090. doi:10.1093/nar/gkt1031
- Cheng Y, Prusoff WH (1973). Relationship between the inhibition constant (K1) and the concentration of inhibitor
  which causes 50 per cent inhibition (I50) of an enzymatic reaction. *Biochem Pharmacol* 22:3099–3108.
  doi:10.1016/0006-2952(73)90196-2
- Clark AJ (1926). The reaction between acetyl choline and muscle cells. *J Physiol* 61:530–546.
  doi:10.1113/jphysiol.1926.sp002314
- Gasteiger J, Marsili M (1980). Iterative partial equalization of orbital electronegativity — a rapid access to
  atomic charges. *Tetrahedron* 36:3219–3228. [DOI to check]
- Gobbi A, Poppinger D (1998). Genetic optimization of combinatorial libraries. *Biotechnol Bioeng* 61:47–54.
  doi:10.1002/(sici)1097-0290(199824)61:1<47::aid-bit9>3.0.co;2-z
- Hall LH, Kier LB (1995). Electrotopological state indices for atom types: a novel combination of electronic,
  topological, and valence state information. *J Chem Inf Comput Sci* 35:1039–1045. doi:10.1021/ci00028a014
- Hetzel L, Böhm S, Kilbertus N, Günnemann S, Lotfollahi M, Theis F (2022). Predicting cellular responses to novel
  drug perturbations at a single-cell resolution. *Advances in Neural Information Processing Systems* 35 (NeurIPS
  2022). arXiv:2204.13545 — prior work that varied the molecular encoder of a perturbation model.
- Labute P (2000). A widely applicable set of descriptors. *J Mol Graph Model* 18:464–477.
  doi:10.1016/s1093-3263(00)00068-1
- Maggiora G, Vogt M, Stumpfe D, Bajorath J (2014). Molecular similarity in medicinal chemistry. *J Med Chem*
  57:3186–3204. doi:10.1021/jm401411z
- Rogers D, Hahn M (2010). Extended-connectivity fingerprints. *J Chem Inf Model* 50:742–754. doi:10.1021/ci100050t
- Wildman SA, Crippen GM (1999). Prediction of physicochemical parameters by atomic contributions. *J Chem Inf Comput
  Sci* 39:868–873. doi:10.1021/ci990307l
- Zdrazil B et al. (2024). The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data
  types and time periods. *Nucleic Acids Res* 52:D1180–D1192. doi:10.1093/nar/gkad1004
