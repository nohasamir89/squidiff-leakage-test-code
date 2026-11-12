Supplementary Table S13 | Classifier-free guidance (Results 4). The guided variant is the rewired model trained with condition dropout (the compound input replaced by zeros for a random 10 % of the cells of each training batch) and sampled with guidance weight 2. Gene identity, response correlation and population distance are defined as in Supplementary Table S12; values are means over the held-out conditions of each compound split, each model averaged over its seeds within a condition first (three seeds on Tahoe-100M, five on sci-Plex). a, Tahoe-100M (2,899, 3,115 and 2,853 conditions on splits 1, 0 and 2): the guided variant against the rewired model. The two were trained separately, so their difference combines the effect of condition dropout with that of guidance. b, sci-Plex (443, 444 and 455 conditions): the same five dropout-trained models sampled with and without guidance, which isolates guidance, and the dropout-trained models sampled without guidance against the rewired model, which isolates condition dropout. Differences carry percentile bootstrap intervals (5,000 replicates, resampling held-out compounds; chemical structures on Tahoe-100M) at the 97.5 % level, and at the 95 % level on the sci-Plex development split, as fixed before each analysis; an asterisk marks an interval that excludes zero. The comparisons on gene identity and on population distance were specified before the data were scored; response correlation was a reported secondary measure. † Difference of the two means; the analysis of the confirmation splits computed no interval for this comparison. Differences in gene identity are given to four decimals because they lie in the third decimal. On the sci-Plex development split, the rewired model of this comparison was a second set of five models; like the dropout-trained models, it was trained without the three training compounds that the learned-embedding comparison could not encode (Supplementary Table S10). Its values differ in the third decimal from those of the rewired model in Supplementary Tables S5 and S6 (gene identity 0.609 against 0.610).

**a** | Tahoe-100M.

| Measure | Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|---|
| Gene identity | Rewired model | 0.464 | 0.460 | 0.458 |
|  | Trained with condition dropout, sampled with guidance (weight 2) | 0.470 | 0.463 | 0.463 |
|  | Guided − rewired model | +0.0064 [+0.0034, +0.0095]* | +0.0025 [−0.0010, +0.0060] | +0.0041 [+0.0001, +0.0079]* |
| Response correlation | Rewired model | 0.520 | 0.501 | 0.523 |
|  | Trained with condition dropout, sampled with guidance (weight 2) | 0.456 | 0.460 | 0.470 |
|  | Guided − rewired model | −0.064 [−0.079, −0.049]* | −0.040 [−0.062, −0.015]* | −0.053 [−0.071, −0.037]* |
| Population distance | Rewired model | 0.225 | 0.244 | 0.218 |
|  | Trained with condition dropout, sampled with guidance (weight 2) | 0.393 | 0.349 | 0.363 |
|  | Guided − rewired model | +0.168 [+0.113, +0.231]* | +0.105 [+0.065, +0.152]* | +0.145 [+0.102, +0.191]* |

**b** | sci-Plex.

| Measure | Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|---|
| Gene identity | Rewired model | 0.609 | 0.612 | 0.611 |
|  | Trained with condition dropout, sampled without guidance | 0.609 | 0.613 | 0.611 |
|  | Trained with condition dropout, sampled with guidance (weight 2) | 0.617 | 0.628 | 0.622 |
|  | Guidance − no guidance | +0.0085 [+0.0056, +0.0114]* | +0.0144 [+0.0110, +0.0177]* | +0.0113 [+0.0081, +0.0150]* |
|  | Condition dropout − rewired model | −0.0008 [−0.0026, +0.0010] | +0.0008 [−0.0013, +0.0028] | +0.0003 [−0.0017, +0.0023] |
| Response correlation | Rewired model | 0.100 | 0.070 | 0.069 |
|  | Trained with condition dropout, sampled without guidance | 0.100 | 0.068 | 0.075 |
|  | Trained with condition dropout, sampled with guidance (weight 2) | 0.118 | 0.079 | 0.078 |
|  | Guidance − no guidance | +0.018 [+0.002, +0.035]* | +0.011† | +0.003† |
|  | Condition dropout − rewired model | 0.000 [−0.009, +0.009] | −0.002 [−0.011, +0.007] | +0.006 [−0.002, +0.014] |
| Population distance | Rewired model | 0.179 | 0.048 | 0.118 |
|  | Trained with condition dropout, sampled without guidance | 0.180 | 0.045 | 0.117 |
|  | Trained with condition dropout, sampled with guidance (weight 2) | 0.185 | 0.070 | 0.124 |
|  | Guidance − no guidance | +0.005 [−0.043, +0.051] | +0.025 [+0.013, +0.043]* | +0.006 [−0.059, +0.059] |
|  | Condition dropout − rewired model | +0.001 [−0.002, +0.004] | −0.003 [−0.005, −0.002]* | 0.000 [−0.005, +0.005] |

Sources: step15_model_analysis.json (sha256 5fd700b5a030cf5a; Tahoe-100M), step12_analysis.json (sha256 c3d1c642dca278d5; sci-Plex development split) and step12b_analysis.json (sha256 6cadb30a2bb7bfd5; sci-Plex confirmation splits). On sci-Plex, the guided models are those of Supplementary Table S6, and the rewired model of the confirmation splits is that of Supplementary Tables S5 and S6 (step14_analysis.json, sha256 c7f021b27272bbac). Assembled by make_supp_table_s13.py.
