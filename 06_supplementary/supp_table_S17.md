Supplementary Table S17 | Spread agreement on Tahoe-100M and its decomposition (Supplementary Note 5; Supplementary Fig. S1). The spread of a population is the mean Euclidean distance of its cells from their mean, over the 200 genes and at most 200 cells (a different quantity from the mean pairwise distance of Supplementary Tables S4 and S7). The spread ratio of a condition is the spread of its treated cells divided by the spread of the matched vehicle cells given to the model for that condition; it was computed for the measured and for the predicted cells. Spread agreement is the Spearman correlation over conditions between the predicted and the measured ratio, within each context (one cell line at one dose) holding at least ten held-out conditions; the measure is its mean over the 42 contexts and, for a model, over three seeds. Values are given with 97.5 % percentile bootstrap intervals against zero (5,000 replicates, resampling held-out chemical structures); an asterisk marks an interval that excludes zero. Splits 1, 0 and 2 hold 2,899, 3,115 and 2,853 held-out conditions of 75 structures each. a, Spread agreement. Each linear reference adds one shift to every vehicle cell and sets negative values to zero; the unaltered vehicle cells have a ratio of 1 in every condition and score zero by construction. For the plate replicate, the ratio measured on one plate is the prediction for the same condition on the other; it is scored on the 14 contexts that have a plate copy (259, 299 and 246 condition pairs on splits 1, 0 and 2). The last row compares the two models. b, The measured ratio over conditions; the generated spread divided by the measured spread (mean over conditions and seeds); and the standard deviation over conditions of the ratio predicted by each linear reference, which would be zero without the clipping at zero. Parts a and b come from an analysis specified before the data were scored. c, Decomposition. Agreement is the measure of part a, computed on the prediction averaged over the three seeds (for the two models it therefore differs from part a, by at most 0.007). “Vehicle spread removed” is the partial Spearman correlation between predicted and measured ratio given the spread of the vehicle cells, the denominator shared by both ratios. “Compound-blind prediction removed” is the same given the ratio predicted by the compound-blind average. “Agreement on spread itself” is the partial Spearman correlation between predicted and measured spread, with no ratio, given the vehicle spread. d, Spearman correlation of each quantity with the vehicle spread. Parts c and d use the contexts and the bootstrap resamples of part a. They were exploratory: the decomposition was added after the compound-blind average had scored above zero on the development split, and its tests were fixed before it was run.

**a** | Spread agreement.

| Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Rewired model | +0.161 [+0.040, +0.275]* | +0.188 [+0.057, +0.306]* | +0.176 [+0.055, +0.284]* |
| Rewired model with optimal-transport pairing | +0.170 [+0.056, +0.275]* | +0.156 [+0.045, +0.254]* | +0.170 [+0.038, +0.287]* |
| Compound-blind average | +0.131 [+0.049, +0.210]* | +0.112 [+0.028, +0.193]* | +0.133 [+0.036, +0.226]* |
| Training-set ridge | +0.066 [−0.058, +0.178] | +0.027 [−0.094, +0.155] | +0.015 [−0.117, +0.150] |
| Cross-validated ridge | +0.115 [−0.015, +0.238] | −0.028 [−0.156, +0.099] | −0.056 [−0.162, +0.061] |
| Plate replicate (14 contexts) | +0.327 [+0.058, +0.458]* | +0.332 [−0.020, +0.567] | +0.382 [+0.090, +0.505]* |
| Rewired model with optimal-transport pairing − rewired model | +0.009 [−0.063, +0.084] | −0.033 [−0.087, +0.021] | −0.006 [−0.066, +0.049] |

**b** | Size of the measured and predicted quantities.

| Quantity | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Measured ratio, mean | 0.995 | 0.994 | 0.996 |
| Measured ratio, median | 0.995 | 0.994 | 0.995 |
| Measured ratio, standard deviation | 0.037 | 0.038 | 0.038 |
| Conditions with a measured ratio above 1 (%) | 43.7 | 41.5 | 43.2 |
| Generated spread / measured spread, rewired model | 0.992 | 0.996 | 0.995 |
| Generated spread / measured spread, rewired model with optimal-transport pairing | 0.985 | 0.988 | 0.987 |
| Standard deviation of the predicted ratio, compound-blind average | 0.006 | 0.005 | 0.006 |
| Standard deviation of the predicted ratio, training-set ridge | 0.019 | 0.018 | 0.027 |
| Standard deviation of the predicted ratio, cross-validated ridge | 0.013 | 0.015 | 0.011 |

**c** | Decomposition (exploratory).

| Entry | Quantity | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|---|
| Rewired model | Agreement | +0.162 [+0.039, +0.277]* | +0.188 [+0.057, +0.308]* | +0.176 [+0.055, +0.284]* |
| Rewired model | Agreement, vehicle spread removed | −0.050 [−0.166, +0.073] | +0.074 [−0.046, +0.184] | −0.016 [−0.137, +0.114] |
| Rewired model | Agreement, compound-blind prediction removed | +0.192 [+0.081, +0.295]* | +0.214 [+0.092, +0.325]* | +0.208 [+0.080, +0.316]* |
| Rewired model | Agreement on spread itself, vehicle spread removed | −0.028 [−0.150, +0.093] | +0.098 [−0.029, +0.216] | +0.001 [−0.122, +0.130] |
| Rewired model with optimal-transport pairing | Agreement | +0.174 [+0.057, +0.281]* | +0.162 [+0.050, +0.263]* | +0.173 [+0.040, +0.292]* |
| Rewired model with optimal-transport pairing | Agreement, vehicle spread removed | +0.009 [−0.104, +0.122] | +0.050 [−0.049, +0.148] | +0.029 [−0.095, +0.156] |
| Rewired model with optimal-transport pairing | Agreement, compound-blind prediction removed | +0.194 [+0.085, +0.291]* | +0.177 [+0.065, +0.276]* | +0.192 [+0.055, +0.310]* |
| Rewired model with optimal-transport pairing | Agreement on spread itself, vehicle spread removed | +0.077 [−0.031, +0.187] | +0.129 [+0.032, +0.224]* | +0.100 [−0.014, +0.218] |
| Compound-blind average | Agreement | +0.131 [+0.049, +0.210]* | +0.112 [+0.028, +0.193]* | +0.133 [+0.036, +0.226]* |
| Compound-blind average | Agreement, vehicle spread removed | +0.137 [+0.072, +0.199]* | +0.123 [+0.041, +0.200]* | +0.177 [+0.094, +0.252]* |
| Compound-blind average | Agreement on spread itself, vehicle spread removed | +0.070 [+0.014, +0.107]* | +0.060 [+0.006, +0.105]* | +0.078 [+0.020, +0.116]* |
| Training-set ridge | Agreement | +0.066 [−0.058, +0.178] | +0.027 [−0.094, +0.155] | +0.015 [−0.117, +0.150] |
| Training-set ridge | Agreement, vehicle spread removed | +0.033 [−0.107, +0.158] | +0.038 [−0.088, +0.171] | +0.027 [−0.113, +0.161] |
| Training-set ridge | Agreement, compound-blind prediction removed | +0.073 [−0.048, +0.179] | +0.022 [−0.099, +0.150] | +0.015 [−0.109, +0.143] |
| Training-set ridge | Agreement on spread itself, vehicle spread removed | +0.094 [−0.013, +0.197] | +0.137 [+0.022, +0.259]* | +0.102 [−0.050, +0.243] |
| Cross-validated ridge | Agreement | +0.115 [−0.015, +0.238] | −0.028 [−0.156, +0.099] | −0.056 [−0.162, +0.061] |
| Cross-validated ridge | Agreement, vehicle spread removed | +0.146 [+0.004, +0.276]* | −0.031 [−0.160, +0.096] | −0.062 [−0.175, +0.058] |
| Cross-validated ridge | Agreement, compound-blind prediction removed | +0.085 [−0.041, +0.207] | −0.037 [−0.159, +0.087] | −0.061 [−0.171, +0.056] |
| Cross-validated ridge | Agreement on spread itself, vehicle spread removed | +0.162 [+0.056, +0.263]* | +0.065 [−0.040, +0.176] | +0.010 [−0.081, +0.117] |

**d** | Correlation with the vehicle spread (exploratory).

| Quantity | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Measured treated spread | +0.505 [+0.430, +0.550]* | +0.562 [+0.491, +0.610]* | +0.508 [+0.421, +0.572]* |
| Measured ratio | −0.292 [−0.398, −0.180]* | −0.195 [−0.301, −0.087]* | −0.283 [−0.377, −0.179]* |
| Predicted ratio, rewired model | −0.699 [−0.739, −0.637]* | −0.717 [−0.766, −0.642]* | −0.686 [−0.758, −0.586]* |
| Predicted ratio, rewired model with optimal-transport pairing | −0.573 [−0.642, −0.485]* | −0.565 [−0.635, −0.476]* | −0.520 [−0.631, −0.377]* |
| Predicted ratio, compound-blind average | +0.049 [−0.085, +0.186] | +0.105 [−0.005, +0.215] | +0.137 [+0.004, +0.260]* |
| Predicted ratio, training-set ridge | −0.113 [−0.238, +0.017] | +0.039 [−0.072, +0.138] | +0.054 [−0.094, +0.194] |
| Predicted ratio, cross-validated ridge | +0.140 [+0.006, +0.258]* | +0.026 [−0.106, +0.167] | +0.012 [−0.117, +0.129] |

Sources: step19_heterogeneity.json (sha256 de4dd1396cca78bb; parts a and b) and step19_diag_control_route.json (sha256 6112c94c13cb14ba; parts c and d). The values of part a, other than its last row, equal those Supplementary Fig. S1 was drawn from (fig5_values.csv, panel c). Assembled by make_supp_table_s17.py.
