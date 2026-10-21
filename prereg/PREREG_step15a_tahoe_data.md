# Pre-registration: Step 15a, Tahoe-100M (brain and gut lines) — data preparation

**BLIND for every quantity this step produces: no Tahoe split file, gene panel, structure group
or paired control exists. NOT BLIND for the readiness record** (`tahoe_results/tahoe_readiness.json`,
commit deec6fd), which is the evidence for the rules below and is cited where used.

Author: Noha Samir Ismail. Date: 2026-09-25. Branch `step15-tahoe`; records in `tahoe_results/`.

## 1. Purpose and scope

Step 15 asks whether the sci-Plex findings of Steps 1–14 — the recipe (semantic adaLN, Gaussian-source
flow matching, 50 NFE, unguided) and its standing against the linear-shift baselines — hold on a second
drug screen: the Tahoe-100M plates restricted to brain and gut cancer lines. This document fixes only how
the data are prepared. The model arms, endpoints, outcome categories and expectations are a separate
pre-registration (Step 15b), written after the data record of section 6 exists and before any training.

The preparation is written so that every quantity below is either (a) a rule fixed here, or (b) a number
the script derives from the data and records, never a choice made after seeing a model result.

## 2. What is known, and its status

| fact | source |
|---|---|
| 13 plate files (plate 12 excluded during preprocessing for a data problem, by decision before this step); 0.76–2.26 M cells per plate; 14 lines, 13 on every plate (SW 1088 absent from plate 7) | readiness record |
| 376 treated drugs; 192 at three doses (0.05 / 0.5 / 5 µM), 165 at two, 19 at one; a drug–dose sits on one plate; vehicle `DMSO_TF`, two wells per plate, 374–10,008 cells per line per plate | readiness record |
| cells per condition (line × drug × dose): median 900–1,500, 5th percentile 84–573; every condition ≥ 20 cells | readiness record |
| `layers['counts']` integral raw UMIs; stored `X` = `normalize_total(1e4, exclude_highly_expressed=True)` + `log1p`, run per plate; the excluded gene set therefore differs by plate | `preprocessing_4/hvg.py` lines 229–245; normalisation probe of 2026-09-25 |
| the stored 3,000-gene panels were selected per plate (`seurat_v3` on counts) and differ between plates (overlap with plate 1: 2,460–2,704) | `hvg.py` lines 257–263; readiness record |
| full-gene files (`cpa_ready_plate<k>_full_genes.h5ad`, 30,287 genes on plate 1) carry the same cells with counts for all genes | probe of 2026-09-25 |
| drug → SMILES: `preprocessing/tahoe_drug_metadata.csv` (`canonical_smiles`, `pubchem_cid`) covers 374 of 376 drugs once names are whitespace-stripped; Verteporfin has none there (obtained from PubChem CID 5362420, formula-verified, 2026-09-25); Sacubitril/Valsartan is a two-molecule product with no single structure | checks of 2026-09-25 |
| two name pairs share one canonical SMILES (crizotinib / (S)-Crizotinib; Doxorubicin / Epirubicin, hydrochlorides): stereoisomers, indistinguishable to the fingerprint | same |
| on sci-Plex the model operates on a 200-gene panel; the DE-overlap chance floor (0.206) is a property of that panel size and top-50 rule; the base model failed to reconstruct 2,000-gene panels (93% zeros) and worked at 100–800 | `findings`, gene-count robustness section |
| the encoder reads the drug as fingerprint × log10(dose + 1) with the dose in the sci-Plex files' unit | `PREREG_step13_attention.md` §1 |

## 3. Inputs

- Plates 1–11, 13, 14: `/scratch/ni2071/Tahoa_2/preprocessing_4/cpa_hvg_plates/cpa_ready_plate<k>_full_genes.h5ad`
  (the full-gene files; the stored 3,000-gene files and the stored `X` are not used).
- Drug table: `/scratch/ni2071/Tahoa_2/preprocessing/tahoe_drug_metadata.csv`, plus the Verteporfin row
  (PubChem CID 5362420 ConnectivitySMILES, recorded in the script).
- sci-Plex split files `datasets_corrected/sci_plex_{train,test}_drug_split_1.h5ad` and their
  `_control.h5ad` companions: read only for their obs schema, which the Tahoe files reproduce (read on
  2026-09-26: obs `Group`, `dose`, `SMILES`, `cell_name`; `X` float32, 200 genes; no `uns`; the control
  file has the same number of rows as its treated file and the same labels row for row).

## 4. Rules

**4.1 Cells and conditions.** Every cell of the 13 files is eligible (the files are post-QC). A condition is
`cell line × drug × dose`; each lies on one plate. Vehicle cells are `drug == 'DMSO_TF'` (`dose = 0`).
The paired control of a condition is the vehicle cells of the same line on the same plate — the analogue
of the sci-Plex line-wise pairing, with the plate added because every condition and its controls share a
plate. All 14 lines are kept; SW 1088 simply has no plate-7 conditions.

*Correction, 2026-09-26, before any output was written (first batch run of `prep_step15_tahoe.py`, commit ddb31af,
stopped at plate 13): SW 1088 has treated cells but no vehicle cells on plate 13. A condition whose line has no vehicle
cells on its plate cannot be paired under this rule and is excluded; the script counts and lists such conditions in
the record instead of stopping. No other rule changes.*

**4.2 Drugs and structures.** Names are matched to the drug table after stripping whitespace. Structure
= RDKit canonical SMILES of the table's SMILES (stereochemistry removed, as the fingerprint removes it).
Sacubitril/Valsartan is excluded (no single structure); 375 drugs remain, forming 373 structures. Splits,
bootstrap clustering and "held-out compound" are defined on structures, so a stereo pair is always on
one side of a split. The two pairs receive identical fingerprints and, at equal dose and line, identical
conditioning; this is recorded as a limitation, not corrected.

**4.3 Gene universe.** Genes present in all 13 full-gene files. Per-cell totals are taken over this universe.

**4.4 Normalisation.** `X = log1p(counts / total × 10⁴)`, the total being the cell's counts over the gene
universe; no gene is excluded from the total. This is the sci-Plex files' construction (single log1p on
library-size-normalised counts) and the GBM preprocessing (`prep_gbm_labels.py`), so all three datasets
of the thesis share one transform. The stored Tahoe `X` is not used because its size factor depends on a
per-plate excluded gene set (section 2), which cannot be stated as one rule. The rejected alternative —
a size factor excluding a fixed list of dominant transcripts (MALAT1, MT-RNR1, MT-RNR2) — is closer to
the stored values but adds a data-derived list that the sci-Plex and GBM values do not have.

**4.5 Gene panel: 200 genes, one panel for every split.** Statistics come from a fixed random sample of
300,000 cells per plate (`RandomState(k)` for plate k; contiguous-block reads as in
`tahoe_readiness_check.py`). A gene is *eligible* if it is in the universe, is detected (count > 0) in
≥ 1% of the sampled cells on every plate, does not start with `MT-`, `RPS` or `RPL`, and does not exceed
5% of a cell's counts in ≥ 1% of the sampled cells of any plate (the dominant transcripts; expected
MALAT1 and the mitochondrial rRNAs). Eligible genes are ranked per plate by scanpy's
`highly_variable_genes(flavor='seurat_v3', layer='counts')` normalised variance (rank 1 = most
variable), and pooled by the median rank over the 13 plates; the panel is the 200 best pooled ranks.
The size is sci-Plex's, so that the DE-overlap chance floor, the top-50 rule and the model's operating
regime (section 2) carry over unchanged. The panel is selected on all cells before any split, as the
sci-Plex panel was; this is a dataset-level property, not a label, and is stated as such.

**4.6 Splits: three, compound-held-out, disjoint.** The 373 structures are shuffled once
(`RandomState(15)`); split k (k = 0, 1, 2) holds out structures 75k … 75k + 74 of the shuffled order
(75 structures ≈ 20%; their drugs at every dose and line are the test conditions), and trains on the
remaining 298. The held-out sets of the three splits are disjoint. As on sci-Plex, `drug_split_1` is the
development split and `drug_split_0`, `drug_split_2` are confirmation splits; Step 15b fixes what, if
anything, may be chosen on the development split.

**4.7 Cell caps and pairing, drawn once with `RandomState(0)`, by condition.** Training conditions: up
to 150 cells per condition, matching the sci-Plex density (172,060 cells over 1,344 training conditions,
128 per condition). Test conditions: up to 500 cells per condition (the evaluation harness applies its
own cap of 200 real cells per condition, as on sci-Plex). Pairing follows the sci-Plex layout exactly:
each treated file has a `_control` companion with the same number of rows, and row *i* of the companion
is a vehicle cell of the same line and plate as treated row *i*, drawn with replacement from that
line-plate's vehicle cells and carrying treated row *i*'s labels. The draws are recorded.

**4.8 Schema.** The Tahoe files carry exactly the obs columns of the sci-Plex split files, with the same
names, types and units: `Group` = `<cell_name>_<drug>_<dose / 10000>` (sci-Plex writes the dose suffix
as a fraction of 10 µM); `dose` in nM (50, 500, 5000; vehicle rows carry the treated row's dose, as in
sci-Plex); `SMILES` as given by the drug table (salts and stereochemistry kept, as the sci-Plex strings
keep them; the structure grouping of 4.2 is separate); `cell_name` the line. Two columns are added:
`plate` and `sample` (well; the vehicle cell's own well in the control files). Drug or line names
containing `_` would break the `Group` convention; the script asserts there are none, and if there are,
replaces `_` by `-` in the name and records the mapping. `X` is float32, 200 genes in the panel
order, stored as the sci-Plex files store it; `var_names` are gene symbols; no `uns` entries that the training code could misread — provenance
goes in `uns['step15']` only if a dry run confirms the loader ignores it, otherwise in the record alone.

*Correction, 2026-09-26, before any run (dry run of `prep_step15_tahoe.py`, commit da9150f): the sci-Plex `X` is stored
sparse (CSR, float32), not dense as this section first said; the Tahoe files are written the same way. The dry run also
showed that a drug can sit at two doses on one plate (Adagrasib, plate 8); each dose is its own condition under 4.1 and
nothing changes.*

*Correction, 2026-09-26, after the first complete run of the script and before its outputs are used or its record
committed: 130 (drug, dose) pairs recur on several plates (plate 14 replicates plate 6 in full, plate 13 re-runs drug–doses
of plates 1, 5, 7, 8, 9, Adagrasib 50 nM sits on 12 plates), so 1,743 of the 14,363 conditions shared their `Group` label
with a condition on another plate. `Group` is therefore `<cell_name>_<drug>_<dose / 10000>_p<plate>` on every row — the
sci-Plex form with the plate appended — so that a label names exactly one condition, as 4.1 defines it. The training code
carries `Group` as an opaque string. The outputs of the first run were deleted and the script rerun; draws, panel and
splits are unchanged.*

## 5. Outputs

`/scratch/ni2071/Diffusion/tahoe/datasets_step15/tahoe_{train,test}_drug_split_<k>.h5ad` and
`tahoe_{train,test}_drug_split_<k>_control.h5ad`, k = 0, 1, 2 — twelve files, the sci-Plex layout with
`sci_plex` replaced by `tahoe`. The `/scratch` file-count quota is the binding limit; nothing else is
written there.

## 6. Record

`tahoe_results/step15_data_record.json` (committed) and `tahoe_results/step15_gene_panel.txt`: the gene
universe size; the per-plate eligibility counts and the excluded dominant transcripts by name; the panel
with each gene's pooled rank and detection rate; the structure table (drug → canonical SMILES → structure
id, the stereo pairs, the excluded drug); per split, the held-out structures and drugs, the number of
conditions, cells and vehicle cells in each file, cells per condition (quantiles), and the SHA-256 of each
output file. The script prints every number it records.

## 7. Analysis plan and provenance

`prep_step15_tahoe.py`, committed before it runs, on a compute node (`sbatch`, 4 CPUs, 64 GB, 6 h), in
stages that each write nothing until their assertions pass: inventory and gene universe; per-plate samples
and gene statistics; panel; drug table and structures; splits; cell selection and files; record. It stops
if any input is missing, if a drug name has no structure (other than the one excluded above), if the
sci-Plex schema cannot be matched, or if any condition in a test file has fewer than 20 cells.

- Readiness record: `tahoe_results/tahoe_readiness.json`, commit deec6fd; script `tahoe_readiness_check.py`, commit 8e47dcb.
- Pre-registration commit: [hash]. Preparation script commit: [hash].

## 8. What this cannot establish

- **Anything about the model.** No arm is trained or evaluated here.
- **That the panel is optimal.** It is a rule fixed in advance; Step 15b does not revisit it.
- **Independence of plate and drug.** Each drug–dose sits on one plate, so plate is confounded with
  drug–dose by design of the screen; pairing controls within plate removes the plate offset from every
  delta, and nothing more is claimed.
- **Dose comparability with sci-Plex.** Three doses (0.05 / 0.5 / 5 µM) against four; the dose
  transform is inherited, not re-fitted.
