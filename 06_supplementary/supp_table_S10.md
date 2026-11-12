Supplementary Table S10 | Interventions on the compound input of the rewired model on sci-Plex (Results 3; Fig. 3b, c; Supplementary Note 1). Gene identity, response correlation and population distance as in Supplementary Tables S5 and S6. Each variant was trained with five seeds and compared with five seeds of the rewired model evaluated with the same number of integration steps. Differences are given with percentile bootstrap intervals (5,000 replicates, resampling held-out compounds), at 95 % on the development split (split 1; 443 conditions of 37 held-out compounds) and at 97.5 % on the confirmation splits (splits 0 and 2; 444 and 455 conditions); an asterisk marks an interval that excludes zero. For population distance a negative difference is closer to the measured cells. a, The interventions on the development split. Response correlation: median over conditions of the paired difference. Gene identity: mean over conditions of the difference. The first, the third and the last three rows are the primary comparisons of the experiments; six of the seven rows are drawn in Fig. 3b. b, The two attention modules that were not taken further: every comparison specified for them (mean differences). c, Cross-attention and its control on the three splits (mean differences); the first three rows are drawn in Fig. 3c, and the last rows give the values the differences are computed from and the difference in population distance relative to the rewired model. Parts a to c come from analyses specified before the data were scored. d, Ridge models fitted within the held-out compounds, with folds grouped by compound, on the cell line, the compound features and their products, each at the penalty that was best for it. The compound features were the fingerprint, the mechanism-of-action label or the annotated target; the label model was the best of nine feature sets tried on the development split and was then fitted on the other two (Supplementary Note 1). Its fingerprint model therefore differs slightly from the cross-validated ridge of part e, which has a fixed penalty. e, A feedforward network given the features of the linear references (4,099: dose-scaled fingerprint, cell-line indicator and their products), fitted within the held-out compounds with five folds grouped by compound, as the cross-validated ridge is. One hidden layer with ReLU, AdamW, weight decay 10, 400 epochs (the best of twelve settings on the development split: weight decay 0.01–10, 25–400 epochs), minibatch 32, learning rate 10⁻³, inputs standardised on the training folds; three seeds per width. Values of parts d and e are the median over conditions of the response correlation, for the network the mean over seeds. Parts d and e were exploratory.

**Mechanism-of-action labels.** The annotated mechanism class of each compound (55 classes) was given as an indicator vector scaled by log10(dose + 1), as the fingerprint is. It was added to the fingerprint (input width 1,079 instead of 1,024; 0.19 % more parameters) or used in its place. Cells were generated in 20 integration steps.

**Learned chemical embedding.** Each compound was represented by an embedding from a pretrained chemical variational autoencoder (ChemicalVAE; 196 dimensions, 64 draws per compound), added to the fingerprint (input width 1,220) or used in its place. The embedding was drawn once per training cell when the dataset was built and kept for the whole of training, not re-drawn at each access as pre-registered; evaluation averaged 16 draws per condition, as planned (Methods, Variants of the rewired model). Three training compounds that the autoencoder could not encode (4,577 cells, 2.7 % of the training cells) were left out of all three arms, so the rewired model of this experiment was retrained without them; the held-out conditions were unchanged. Cells were generated in 20 integration steps.

**Attention modules.** Each module computes a correction of width 1,024 that is added to the fingerprint before the unchanged encoder; its last layer starts at zero, so every variant equals the rewired model before training. Fingerprint self-attention: one attention layer (width 64, four heads) over the active bits of the fingerprint (182,080 added parameters). Attention over similar training compounds: the compound attends over one learned entry per training compound (112 on the development split), addressed by the fingerprints of those compounds (204,928 added parameters); a held-out compound is never an entry. Cross-attention between compound and cell state and its control (354,624 and 354,120 added parameters) are described in the Methods. Cells were generated in 50 integration steps.

**a** | The interventions on the development split.

| Intervention | Measure | Difference from the rewired model |
|---|---|---|
| Mechanism-of-action labels added to the fingerprint | Response correlation | +0.002 [−0.003, +0.009] |
| Mechanism-of-action labels in place of the fingerprint | Response correlation | +0.002 [−0.020, +0.021] |
| Learned chemical embedding added to the fingerprint | Response correlation | −0.010 [−0.036, +0.005] |
| Learned chemical embedding in place of the fingerprint | Response correlation | −0.023 [−0.060, +0.003] |
| Fingerprint self-attention | Gene identity | −0.0003 [−0.0029, +0.0024] |
| Attention over similar training compounds | Gene identity | +0.0016 [−0.0006, +0.0039] |
| Cross-attention between compound and cell state | Gene identity | +0.0057 [+0.0028, +0.0088]* |

**b** | The two attention modules that were not taken further (development split).

| Module | Comparison | Difference |
|---|---|---|
| Fingerprint self-attention | Gene identity, against the rewired model | −0.0003 [−0.0029, +0.0024] |
| Fingerprint self-attention | Gene identity, against the parameter-matched control | −0.0029 [−0.0053, −0.0006]* |
| Fingerprint self-attention | Population distance, against the rewired model | −0.013 [−0.032, +0.002] |
| Fingerprint self-attention | Response correlation, against the rewired model | +0.010 [+0.001, +0.019]* |
| Attention over similar training compounds | Gene identity, against the rewired model | +0.0016 [−0.0006, +0.0039] |
| Attention over similar training compounds | Gene identity, against the parameter-matched control | −0.0009 [−0.0027, +0.0009] |
| Attention over similar training compounds | Population distance, against the rewired model | −0.011 [−0.022, −0.002]* |
| Attention over similar training compounds | Response correlation, against the rewired model | +0.002 [−0.004, +0.009] |

**c** | Cross-attention and its control on the three splits.

| Entry | Measure | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|---|
| Cross-attention − rewired model | Gene identity | +0.0057 [+0.0028, +0.0088]* | −0.0014 [−0.0041, +0.0016] | +0.0010 [−0.0016, +0.0037] |
| Cross-attention − parameter-matched control | Gene identity | +0.0032 [+0.0013, +0.0052]* | −0.0013 [−0.0038, +0.0011] | −0.0020 [−0.0047, +0.0007] |
| Cross-attention − rewired model | Population distance | −0.021 [−0.043, −0.006]* | −0.008 [−0.010, −0.006]* | −0.011 [−0.036, +0.001] |
| Cross-attention − rewired model | Response correlation | +0.019 [+0.006, +0.032]* | +0.009 [−0.001, +0.020] | +0.016 [+0.004, +0.030]* |
| Control − rewired model | Gene identity | +0.0025 [+0.0005, +0.0046]* | −0.0001 [−0.0023, +0.0021] | +0.0030 [+0.0009, +0.0052]* |
| Control − rewired model | Population distance | −0.003 [−0.009, +0.001] | −0.002 [−0.003, 0.000]* | −0.005 [−0.024, +0.006] |
| Control − rewired model | Response correlation | +0.008 [−0.002, +0.018] | −0.002 [−0.012, +0.007] | +0.016 [+0.004, +0.028]* |
| Rewired model | Gene identity | 0.610 | 0.612 | 0.611 |
| Cross-attention | Gene identity | 0.616 | 0.611 | 0.612 |
| Control | Gene identity | 0.613 | 0.612 | 0.614 |
| Rewired model | Population distance | 0.180 | 0.048 | 0.118 |
| Cross-attention | Population distance | 0.159 | 0.041 | 0.107 |
| Control | Population distance | 0.176 | 0.047 | 0.113 |
| Cross-attention − rewired model, relative (%) | Population distance | −11.6 | −15.8 | −9.1 |

**d** | Linear models given the fingerprint, the labels or the targets (exploratory).

| Compound features | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Fingerprint | 0.189 | 0.247 | 0.082 |
| Mechanism-of-action label | 0.213 | 0.265 | 0.129 |
| Annotated target | 0.170 | 0.264 | 0.101 |
| Mechanism-of-action label − fingerprint | +0.024 | +0.018 | +0.046 |

**e** | Feedforward network given the features of the linear references (exploratory).

|  | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Cross-validated ridge | 0.188 | 0.247 | 0.082 |
| Network, width 64 | 0.193 | 0.211 | 0.108 |
| Network, width 256 | 0.165 | 0.174 | 0.070 |
| Network, width 1,024 | 0.191 | 0.157 | 0.110 |

Sources: step9_analysis.json (sha256 6f84c17f4a7e5344), step10_analysis.json (sha256 83a30129d62491d6), step13_analysis.json (sha256 71b9a94e7bd80f36) and step13b_analysis.json (sha256 4ca621d99af27031; parts a to c); pathway_vs_fingerprint.json (sha256 8bbe7aeec8c36d55; part d); ridge_vs_mlp_v2.json (sha256 7ec3fadafe8b8489; part e, script ridge_vs_mlp_v2.py). The values of parts a and c that are drawn in Fig. 3b and 3c equal those the figures were drawn from (fig3_values.csv). The numbers in the description of the variants are those of the pre-registrations of the three experiments. Assembled by make_supp_table_s10.py.
