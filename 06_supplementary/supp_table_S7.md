Supplementary Table S7 | What the population distance of the rewired model is made of, on Tahoe-100M (Results 2). The analysis of Supplementary Table S4 applied to the rewired model, on the same 2,899 held-out conditions of split 1 (seed 0) and the same measured cells; quantities, statistics and layout are those of that table. The analysis was exploratory.

**a** | Cells: mean (median) over conditions.

| Quantity | Generated cells | Measured cells | Generated ÷ measured |
|---|---|---|---|
| Spread (mean distance between two cells of the population) | 11.50 (11.15) | 11.63 (11.27) | 0.99 |
| Total expression per cell | 49.1 (43.1) | 50.3 (44.4) | 0.98 |
| Zero entries (%) | 37.1 (37.2) | 89.3 (90.5) | — |

**b** | Distances and mean profiles.

| Quantity | Mean over conditions | Median over conditions |
|---|---|---|
| Mean distance between a generated and a measured cell | 11.68 | 11.34 |
| Population distance | 0.228 | 0.136 |
| Correlation between the generated and the measured mean profile | 0.965 | 0.974 |
| Slope of the generated on the measured mean profile | 0.917 | 0.921 |
| Distance from the generated mean to the measured treated mean | 1.67 | 1.45 |
| Distance from the generated mean to the vehicle mean | 1.73 | 1.52 |
| Distance from the measured treated mean to the vehicle mean (the measured response) | 1.37 | 1.21 |
| Generated-to-treated distance ÷ measured response (ratio of the means) | 1.21 | — |

Sources: step15_diag_cells_drug_split_1_seed_0.json (sha256 ec835b903935acd7; diag_step15_cells.py). The table of the same run with one row per condition, from which Fig. 1c is drawn (step15_diag_cells_drug_split_1_seed_0_per_condition.csv, sha256 4810be4a928db638), reproduces every mean and median given here. The population distance is the scored value of this seed in step15_model_analysis.json (sha256 5fd700b5a030cf5a; Supplementary Table S12b). Assembled by make_supp_table_s4_s7.py.
