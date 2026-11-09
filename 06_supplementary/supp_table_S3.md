# Supplementary Table S3 — glioblastoma: gene-panel sweep and reproduction of the published analysis

All correlations are Pearson correlations between mean expression profiles on the log scale, from the re-evaluation on the corrected scale (24 October 2026). Vehicle reconstruction: between the mean expression of reconstructed and of measured vehicle cells, over all genes of the panel. Prediction: between the predicted and measured mean expression of the treated cells, with the compound direction estimated in myeloid cells and added to the encoded vehicle cells of the held-out cell type; averaged over the five held-out compounds. Vehicle mean: the same correlation with the vehicle mean of the held-out cell type used as the prediction. 50 most responsive genes: for each compound, the 50 genes with the largest absolute difference between the treated and vehicle means. Margin = prediction minus vehicle mean on those genes (Fig. 1d, bottom). A size was interpreted only if vehicle reconstruction reached 0.70 in both cell types. The sweep model at 200 genes (a, b) and the model of the reproduction (c) were trained independently, on gene panels constructed independently (Supplementary Note 3). Values are given at the precision of the records: four decimals in the sweep record, three in the evaluation logs.

**a | Gene-panel sweep: panel and vehicle reconstruction (Fig. 1d, top).**

| Genes in the panel | Fraction of zero entries | Vehicle reconstruction, tumour cells | Vehicle reconstruction, oligodendrocytes | Interpreted |
|---|---|---|---|---|
| 100 | 0.9443 | 0.9767 | 0.9986 | yes |
| 200 | 0.9399 | 0.9816 | 0.9970 | yes |
| 300 | 0.9350 | 0.6092 | 0.8527 | no |
| 400 | 0.9310 | 0.4233 | 0.6591 | no |
| 500 | 0.9304 | 0.6585 | 0.7592 | no |
| 600 | 0.9315 | 0.7409 | 0.8353 | yes |
| 700 | 0.9306 | 0.8539 | 0.9193 | yes |
| 800 | 0.9336 | 0.8910 | 0.9340 | yes |
| 900 | 0.9360 | 0.7726 | 0.4452 | no |
| 1,000 | 0.9375 | −0.0372 | −0.0159 | no |

**b | Gene-panel sweep: prediction against the vehicle mean (Fig. 1d, bottom; Results 1).**

| Genes in the panel | Cell type | Prediction, all genes | Vehicle mean, all genes | Prediction, 50 most responsive genes | Vehicle mean, 50 most responsive genes | Margin |
|---|---|---|---|---|---|---|
| 100 | Tumour cells | 0.7527 | 0.7071 | 0.6937 | 0.6313 | 0.0624 |
| 100 | Oligodendrocytes | 0.9337 | 0.9241 | 0.9202 | 0.9075 | 0.0127 |
| 200 | Tumour cells | 0.8010 | 0.7355 | 0.6653 | 0.5322 | 0.1331 |
| 200 | Oligodendrocytes | 0.9317 | 0.9273 | 0.8953 | 0.8851 | 0.0102 |
| 300 | Tumour cells | 0.5461 | 0.7912 | 0.5929 | 0.5613 | 0.0316 |
| 300 | Oligodendrocytes | 0.7900 | 0.9228 | 0.7662 | 0.8306 | −0.0644 |
| 400 | Tumour cells | 0.3432 | 0.7977 | 0.2020 | 0.4927 | −0.2908 |
| 400 | Oligodendrocytes | 0.5907 | 0.9219 | 0.7157 | 0.8157 | −0.1000 |
| 500 | Tumour cells | 0.5741 | 0.7896 | 0.5429 | 0.3873 | 0.1555 |
| 500 | Oligodendrocytes | 0.6478 | 0.9138 | 0.6517 | 0.7590 | −0.1073 |
| 600 | Tumour cells | 0.6503 | 0.7948 | 0.4513 | 0.3795 | 0.0718 |
| 600 | Oligodendrocytes | 0.7350 | 0.9136 | 0.7038 | 0.7593 | −0.0554 |
| 700 | Tumour cells | 0.7480 | 0.8236 | 0.5201 | 0.4389 | 0.0811 |
| 700 | Oligodendrocytes | 0.8243 | 0.9117 | 0.7645 | 0.7399 | 0.0245 |
| 800 | Tumour cells | 0.7906 | 0.8332 | 0.6684 | 0.4279 | 0.2405 |
| 800 | Oligodendrocytes | 0.8551 | 0.9119 | 0.7902 | 0.7317 | 0.0585 |
| 900 | Tumour cells | 0.6404 | 0.8371 | 0.1604 | 0.4086 | −0.2482 |
| 900 | Oligodendrocytes | 0.3057 | 0.9172 | −0.3413 | 0.7234 | −1.0647 |
| 1,000 | Tumour cells | −0.0233 | 0.8451 | −0.0141 | 0.4325 | −0.4465 |
| 1,000 | Oligodendrocytes | −0.0056 | 0.9171 | −0.0751 | 0.7098 | −0.7849 |

**c | Reproduction of the published analysis (Fig. 1e).** Correlation between predicted and measured mean expression of the treated cells, over all genes of the panel, per held-out compound.

| | Tumour cells, 200 genes | Tumour cells, 2,000 genes | Oligodendrocytes, 200 genes | Oligodendrocytes, 2,000 genes |
|---|---|---|---|---|
| panobinostat | 0.831 | 0.414 | 0.858 | 0.434 |
| RO4929097 | 0.890 | 0.364 | 0.960 | 0.445 |
| tazemetostat | 0.795 | 0.350 | 0.937 | 0.446 |
| ispinesib | 0.857 | 0.397 | 0.940 | 0.465 |
| ANA-12 | 0.846 | 0.409 | 0.954 | 0.469 |
| Mean over the five compounds | 0.844 | 0.387 | 0.930 | 0.452 |
| Published value (He et al.) | 0.912 | — | 0.915 | — |
| Vehicle reconstruction | 0.990 | 0.433 | 0.996 | 0.514 |

Sources: (a, b) gene_sensitivity_SEEDED_v2.csv (job 18932187; eval_gbm_sensitivity_v2.py via run_gbm_seeded_reeval_v2.sh; decoder seed 42), sha256 1024091bb2f2a792; (c) eval_200_v2_18927698.out (sha256 097e88b73ccb57aa) and eval_v2_18927697.out (sha256 f3fb22c2bafe0474), through figures/fig1e_values_v2.csv (plot_fig1e_v2.R, research commit 7dbbcab), sha256 e54b44c8462d20ad. Means in (c) are those printed in the logs, computed there from the unrounded per-compound values. Published values: He et al. (Squidiff, Nat. Methods 2026). Assembled by make_supp_table_s3.py.
