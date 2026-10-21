# Manuscript code inventory — figure → script → record → pipeline

Built 20–21 Oct 2026 from three sources: the nine figure scripts themselves (what each reads, read by the tracer
from the committed files), the step findings documents (each names its scorer, harness, launcher and records),
and the manuscript Methods (which name the pipelines that live outside the repository). Repository:
`/scratch/ni2071/Diffusion/Squidiff_git/Squidiff`, branch `step15-tahoe`, commit `2b33b12`.

Flags: **✓** named in the step's findings document · **◦** found by the tracer or the Methods, confirm by eye ·
**?** not located yet.

---

## 1. Main figures

| Panel | What it shows | Drawn by | Reads (record) | Record written by | Upstream |
|---|---|---|---|---|---|
| 1a | study design (schematic) | BioRender | — | — | — |
| 1b | gene identity, original model vs references, both datasets | `plot_fig1b.R` ◦ | `figures/fig1b_values.csv` | `plot_fig1b.py` ◦ (the one remaining Python extractor) ← `leakage_results/step14_analysis.json`, `tahoe_results/step15_model_long.csv`, `tahoe_results/step15_model_analysis.json` (the three plate-replicate rows) | `score_step14.py` ✓, `score_step15_tahoe.py` ✓ |
| 1c | per-cell breakdown, original model vs real cells (Tahoe split 1 seed 0) | `plot_fig1c.R` ◦ | `tahoe_results/step15_diag_cells_drug_split_1_seed_0_per_condition.csv` | `diag_step15_cells.py` ✓ | Step 15b evaluation arrays |
| 1d | gene-panel sweep (glioblastoma) | `plot_fig1d.R` ◦ | `/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/gene_sensitivity_SEEDED.csv` (outside repo) | `eval_gbm_sensitivity.py` ◦ via `run_gbm_seeded_reeval.sh` ◦ | `prep_gbm_sensitivity.py`, `prep_gbm_labels.py`, `run_gbm_sensitivity.sh` ◦ (same folder) |
| 1e | reproduction of the published glioblastoma analysis | `plot_fig1e.R` ◦ | `/scratch/ni2071/Diffusion/gbm_run/logs/eval_16377417.out`, `/scratch/ni2071/Diffusion/gbm_run_200/logs/eval_200_16378137.out` (SLURM logs, outside repo) | `gbm_run_200/run_eval_200.sh` ◦; the 2,000-gene counterpart in `gbm_run/` ? | `gbm_run/train_squidiff_gbm.sh`, `gbm_run_200/train_squidiff_gbm_200.sh` ◦ |
| 2a | gene identity, original vs rewired vs references | `plot_fig2.R` ◦ | `leakage_results/step14_analysis.json`; `tahoe_results/step15_model_analysis.json`, `step15_model_long.csv` | `score_step14.py` + `step14_metrics.py` ✓; `score_step15_tahoe.py` ✓ | sci-Plex and Tahoe pipelines (§4) |
| 2b | population distance on Tahoe | `plot_fig2.R` | same Tahoe records | same | |
| 3a | conditioning pathway, original model (schematic) | BioRender | — | — | — |
| 3b | interventions on the compound input | `plot_fig3.R` ◦ | `leakage_results/step9_analysis.json`, `step10_analysis.json`, `step13_analysis.json`, `step13b_analysis.json` | `analyse_step9.py` ◦, `analyse_step10.py` ◦, `analyse_step13.py` ✓, `analyse_step13b.py` ◦ | Steps 9, 10, 13, 13b training + evaluation (§4) |
| 3c | cross-attention, development vs confirmation | `plot_fig3.R` | `step13_analysis.json`, `step13b_analysis.json` | as above | |
| 4a | conditioning pathway, rewired model (schematic) | BioRender | — | — | — |
| 4b | rewired vs original on Tahoe | `plot_fig4.R` ◦ | `tahoe_results/step15_model_analysis.json`, `step15_model_long.csv` | `score_step15_tahoe.py` ✓ | Step 15b |
| 4c | optimal-transport pairing | `plot_fig4.R` | `tahoe_results/step18_analysis.json`; `tahoe_results/posthoc_ot_energy_vs_references.json` | `score_step18_tahoe.py` ✓; `posthoc_ot_energy_vs_references.py` ✓ | Step 18 |
| 4d | matching-strength sweep | `plot_fig4.R` | `tahoe_results/step20_analysis_development.json` | `score_step20_tahoe.py` ✓ | Step 20 |
| 4e | m = 32 vs exact matching | `plot_fig4.R` | `tahoe_results/step21_analysis.json` | `score_step21_tahoe.py` ✓ | Step 21 |
| 5a | compound discrimination (PDS) | `plot_fig5.R` ◦ | `tahoe_results/step15c_pds.json` | `score_step15c_tahoe.py` ◦ | Step 15c (`diag_step15c_crossplate.py` ◦ if the cross-plate ceiling is plotted) |
| 5b | response-magnitude agreement | `plot_fig5.R` | `tahoe_results/step16_analysis.json`; `leakage_results/step17_mra.json` | `score_step16_tahoe.py` ✓; `score_step17_sciplex_mra.py` ◦ | Step 16 (per-dose sentence: `diag_step16_mra_dose.py` ✓), Step 17 |
| 5c | spread agreement | `plot_fig5.R` | `tahoe_results/step19_heterogeneity.json` | `score_step19_heterogeneity.py` ✓ (+ `diag_step19_control_route.py` ✓ for the decomposition in Table S17) | Step 19 |
| 5d | RNA-fingerprinting recognition | `plot_fig5.R` | `tahoe_results/step22_analysis.json` | `score_step22_fingerprint.py` ✓ | Step 22: `prep_step22_fingerprint.py`, `run_step22_fingerprint.R`, `run_step22_tahoe.sh` ✓, `provenance/R_env/` ✓ |

Each `plot_fig*.R` also writes `fig*_values.csv` next to itself — the values that feed the supplementary tables
(all committed in `2b33b12`).

## 2. Results and tables that have no figure

| Where in the paper | Record | Script |
|---|---|---|
| §1.3 held-out vs training loss (Table S2) | `leakage_results/test_loss_v2.json` | `compute_test_loss.py` ◦ |
| §3 η² at the latent and at the output ("Localisation of the loss of compound identity", Methods) | `leakage_results/eta_latent_vs_output.json` (`_adaln.json` for the rewired model) | `eta_latent_vs_output.py` ◦ |
| §3 supporting ablation (fingerprint of a dissimilar compound) | `leakage_results/fingerprint_ablation_meta.json` (+ per-split .npz) | `fingerprint_ablation.py` ◦, launchers `run_fp_ablation*.sh` ◦ |
| §3.2 feedforward / linear network given the ridge's features (Table S10) | ? | Step 8 scripts — **not located**; `grep -l "step8\|sample_regime" *.py *.sh` |
| §3.2 mechanism labels: ridge given the same labels | in `step9_analysis.json` | `analyse_step9.py` ◦ |
| §2 ridge references (training-set ridge, cross-validated ridge, compound-blind average) | recomputed inside the Step 14 and Step 15b scorers | `score_step14.py` ✓, `score_step15_tahoe.py` ✓ (Step 7's own `analyse_step7.py` ◦ only if Table S5/S6 quote it) |
| §4.3 guidance on sci-Plex (Table S13) | `leakage_results/step14_analysis.json` (guidance w = 2 rows) | `score_step14.py` ✓ |
| Table S14 facets (six VCC-style facets) if used | `leakage_results/step14_facets_v2.json` | `score_step14_facets_v2.py` + `step14_metrics_v2.py` ✓ |
| Table S1 datasets / Table S9 settings | — | hand-made from Methods and `tahoe_results/step15_data_record.json` (`prep_step15_tahoe.py` ◦) |

Tables S8, S11 and S14 are never cited in the text; check whether they exist in the supplementary file.

## 3. Pipelines behind the records

### Shared (both datasets)
`Squidiff/` package (model, `train_util.py` with the 6a12f8a fix, `flow_matching.py`, `drug_attention.py` ✓ for Step 13),
`train_squidiff.py` ◦ (training entry point), `sample_squidiff.py` ◦ (sampler), the harness
`leakage_test_base_sciplex_array.py` ◦ and the flow-matching wrapper `leakage_test_fm_sciplex.py` ? (the Tahoe
wrappers were ported from it, so it exists — confirm the name).

### Tahoe-100M (Steps 15a–22)
- 15a data: `prep_step15_tahoe.py` ◦ → `tahoe_results/step15_data_record.json`
- 15b: `train_step15_tahoe.sh`, `finalize_step15.py`, `port_harness_step15.py` → `leakage_test_base_tahoe.py` + `leakage_test_fm_tahoe.py`, `eval_step15_tahoe.sh`, `score_step15_tahoe.py`, `diag_step15_cells.py` ✓ (all named in step15_findings)
- 15c: `score_step15c_tahoe.py` ◦, `diag_step15c_crossplate.py` ◦
- 16: `patch_step16.py`, `port_harness_step16.py` → `leakage_test_fm_step16.py`, `finalize_step16.py`, `train_step16_tahoe.sh`, `eval_step16_tahoe.sh`, `score_step16_tahoe.py`, `diag_step16_mra_ceiling.py`, `diag_step16_mra_dose.py` ✓ (target tables: `crosscheck_targets.py`, `harmonise_symbols.py` ◦)
- 18: `check_ot_pairing_tahoe.py`, `train_step18_tahoe.sh`, `finalize_step18.py`, `port_harness_step18.py` → `leakage_test_fm_step18.py`, `eval_step18_tahoe.sh`, `patch_eval_step18_speed.py`, `certify_step18_ondisk.py`, `score_step18_tahoe.py`, `posthoc_ot_energy_vs_references.py` ✓
- 19: `score_step19_heterogeneity.py`, `diag_step19_control_route.py` ✓
- 20: `check_minibatch_pairing_tahoe.py`, `train_step20_tahoe.sh`, `finalize_step20.py`, `eval_step20_tahoe.sh`, `score_step20_tahoe.py` ✓
- 21: `train_step21_tahoe.sh`, `finalize_step21.py`, `eval_step21_tahoe.sh`, `score_step21_tahoe.py` ✓
- 22: `prep_step22_fingerprint.py`, `run_step22_fingerprint.R`, `run_step22_tahoe.sh`, `score_step22_fingerprint.py`, `provenance/R_env/` (toolchain record + `rnafingerprinting_classify_guard.patch`) ✓

### sci-Plex (Steps 6–14, 17)
- preprocessing: `/scratch/ni2071/Diffusion/sciplex/prep_sciplex_stage1.py` ◦ (outside repo)
- original model: `train_sciplex_base.sh` ◦, evaluation `run_leakage_array.sh` ◦
- adaptive normalisation alone (DDIM): `train_sciplex_adaln.sh`, `eval_adaln_v2.sh` / `run_leakage_adaln.sh` ◦
- flow matching (the rewired model), Step 6/6b: `train_step6.sh`, `eval_step6.sh`, `analyse_step6.py`, `sweep_step6_nfe.sh`, `train_step6b.sh`, `eval_step6b.sh`, `analyse_step6b.py` ◦
- guidance, Step 12/12b: `train_step12.sh`, `eval_step12.sh`, `train_step12b.sh`, `eval_step12b.sh` ◦ (+ their analyse scripts ?)
- attention, Step 13/13b: `train_step13.sh`, `eval_step13.sh`, `analyse_step13.py` ✓; `train_step13b.sh`, `eval_step13b.sh`, `analyse_step13b.py` ◦
- mechanism labels, Step 9: `train_step9.sh`, `eval_step9.sh`, `analyse_step9.py`, `extract_step9_results.py` ◦
- chemical embedding, Step 10: `train_step10.sh`, `eval_step10.sh`, `analyse_step10.py`, `extract_step10_results.py` ◦
- Step 14 scoring: `score_step14.py`, `step14_metrics.py`, `score_step14_facets_v2.py`, `step14_metrics_v2.py` ✓
- Step 17 (sci-Plex MRA): `score_step17_sciplex_mra.py` ◦
- test loss: `compute_test_loss.py` ◦; bottleneck: `eta_latent_vs_output.py`, `fingerprint_ablation.py` ◦
- Step 8 (features network): **?**

### Glioblastoma (outside repo)
`/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/`: `prep_gbm_sensitivity.py`, `prep_gbm_labels.py`,
`eval_gbm_sensitivity.py`, `run_gbm_sensitivity.sh`, `run_gbm_seeded_reeval.sh`, `plot_gbm_sensitivity.py` ◦ and the
record `gene_sensitivity_SEEDED.csv`; `/scratch/ni2071/Diffusion/gbm_run/` and `gbm_run_200/`: the training and
evaluation scripts and the two logs Fig. 1e reads ◦.

## 4. Pre-registration documents (one per step, all tracked)
`PREREG_step6_flow_matching.md`, `PREREG_step6b_confirmation.md`, `PREREG_step6c_ridge.md`, `PREREG_step7_interaction_baseline.md`,
`PREREG_step8_sample_regime.md`, Step 9 ?, Step 10 ?, `PREREG_step12_guidance.md`, `PREREG_step12b_confirmation.md`,
`PREREG_step13_attention.md`, `PREREG_step13b_confirmation.md`, `PREREG_step14_shift_baselines_vcc_scoring.md`,
`PREREG_step14_amendment2_facets_v2.md`, `PREREG_step15a_tahoe_data.md`, `PREREG_step15b_tahoe_model.md`, Step 15c ?,
`PREREG_step16_tahoe_targets.md`, `PREREG_step17_sciplex_mra.md`, `PREREG_step18_tahoe_ot_pairing.md`,
`PREREG_step19_heterogeneity.md`, `PREREG_step20_pairing_sweep.md`, `PREREG_step21_mb32_confirmation.md`,
`PREREG_step22_fingerprint_identifiability.md`.

## 5. Package layout (what the collaborator receives)

```
squidiff_manuscript_code/
  README.md, MANIFEST.csv (package path, original path, sha256, commit), FIGURE_MAP.md (section 1–2 of this file)
  01_data_preparation/   prep_step15_tahoe.py · prep_sciplex_stage1.py · the GBM prep scripts
  02_training/           Squidiff/ package · train_squidiff.py · every train_*.sh and finalize_*.py · check_*_pairing_tahoe.py · patch_step16.py
  03_evaluation/         sample_squidiff.py · leakage_test_* harness files · port_harness_*.py · every eval_*.sh · run_leakage_*.sh · certify_step18_ondisk.py · patch_eval_step18_speed.py
  04_scoring/            score_*.py · step14_metrics*.py · analyse_step*.py · extract_step*_results.py · diag_* used by the paper · posthoc_* · compute_test_loss.py · eta_latent_vs_output.py · fingerprint_ablation.py
  05_figures/            plot_fig1b.R … plot_fig5.R · plot_fig1b.py
  06_supplementary/      the fig*_values.csv tables and whatever assembles Tables S1–S18
  07_records/            every JSON/CSV in sections 1–2 · external/ holds the two GBM logs and gene_sensitivity_SEEDED.csv
  08_external_pipelines/ sciplex/ · gbm_squidiff_robustness_test/ · gbm_run/ · gbm_run_200/
  prereg/                the pre-registration documents
```
Not packaged (too large, listed in MANIFEST with their location): checkpoints, per-cell `.npz` arrays, `.h5ad` data.
