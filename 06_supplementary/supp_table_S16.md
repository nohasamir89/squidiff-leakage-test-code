Supplementary Table S16 | Response-magnitude agreement (Supplementary Note 4; Supplementary Fig. S2). Within each context (one cell line at one dose) holding at least ten held-out conditions, the Spearman correlation over conditions between the size of the predicted change and the size of the measured change (Euclidean norms, over the 200 genes, of the mean change from the matched vehicle cells) was computed; the measure is its mean over contexts and, for a model, over seeds (three on Tahoe-100M, five on sci-Plex). Zero means no ranking; the compound-blind average and the unaltered vehicle cells have one size per context and score zero by construction. Values are given with 97.5 % percentile bootstrap intervals against zero (5,000 replicates, resampling held-out chemical structures on Tahoe-100M and compounds on sci-Plex); an asterisk marks an interval that excludes zero. a, Tahoe-100M, 42 contexts (14 cell lines at three doses); the plate replicate is scored on the 14 contexts that have a plate copy (259, 299 and 246 condition pairs on splits 1, 0 and 2). b, sci-Plex, 12 contexts (three cell lines at four doses); the split-half entry scores one half of the measured cells against the other, as the mean of five random halvings, and the last two rows compare the rewired model with each ridge. c, Tahoe-100M: the two models given the fingerprint of another held-out compound or none, and the cell-line vehicle average used as a prediction without a model (Results 5; Supplementary Table S15). Intervals were computed for the rewired model on the development split only; the other entries of part c are point values. Parts a to c come from analyses specified before the data were scored. d, Tahoe-100M by dose (14 contexts per dose; 993 / 1,172 / 1,020 conditions at 50 nM, 1,010 / 991 / 959 at 500 nM and 896 / 952 / 874 at 5,000 nM); values for splits 1 / 0 / 2. e, Tahoe-100M restricted to the conditions that have a plate replicate: 14 contexts at the highest dose (518, 570 and 492 conditions). Parts d and e were exploratory.

**a** | Tahoe-100M, all conditions.

| Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Original model | −0.075 [−0.145, −0.003]* | −0.058 [−0.123, +0.006] | −0.107 [−0.175, −0.036]* |
| Rewired model | +0.320 [+0.170, +0.443]* | +0.286 [+0.135, +0.416]* | +0.312 [+0.197, +0.413]* |
| Rewired model with classifier-free guidance (weight 2) | +0.227 [+0.076, +0.365]* | +0.238 [+0.080, +0.378]* | +0.218 [+0.079, +0.351]* |
| Rewired model with cross-attention | +0.325 [+0.177, +0.445]* | +0.268 [+0.114, +0.399]* | +0.318 [+0.202, +0.423]* |
| Rewired model, parameter-matched control without attention | +0.324 [+0.182, +0.440]* | +0.285 [+0.143, +0.417]* | +0.318 [+0.213, +0.415]* |
| Rewired model with target annotation | +0.342 [+0.196, +0.456]* | +0.300 [+0.153, +0.427]* | +0.290 [+0.168, +0.405]* |
| Rewired model, target sets reassigned between compounds | +0.300 [+0.147, +0.434]* | +0.286 [+0.135, +0.414]* | +0.299 [+0.180, +0.407]* |
| Training-set ridge | +0.154 [−0.011, +0.299] | +0.173 [+0.026, +0.332]* | +0.051 [−0.137, +0.229] |
| Cross-validated ridge | −0.050 [−0.202, +0.109] | +0.004 [−0.167, +0.154] | +0.094 [−0.046, +0.228] |
| Plate replicate (14 contexts) | +0.483 [+0.074, +0.685]* | +0.483 [+0.039, +0.708]* | +0.328 [−0.061, +0.557] |

**b** | sci-Plex.

| Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Original model | +0.227 [+0.067, +0.371]* | +0.182 [+0.066, +0.291]* | +0.184 [−0.001, +0.374] |
| Rewired model | +0.334 [+0.138, +0.503]* | +0.241 [+0.111, +0.355]* | +0.270 [+0.072, +0.466]* |
| Rewired model with classifier-free guidance (weight 2) | +0.290 [+0.069, +0.483]* | +0.211 [+0.077, +0.327]* | +0.258 [+0.069, +0.449]* |
| Rewired model with cross-attention | +0.356 [+0.155, +0.535]* | +0.269 [+0.131, +0.390]* | +0.292 [+0.063, +0.497]* |
| Rewired model, parameter-matched control without attention | +0.328 [+0.122, +0.510]* | +0.265 [+0.136, +0.377]* | +0.234 [+0.030, +0.444]* |
| Training-set ridge | +0.403 [+0.156, +0.608]* | +0.090 [−0.078, +0.259] | +0.209 [−0.057, +0.454] |
| Cross-validated ridge | +0.278 [+0.031, +0.481]* | +0.127 [−0.020, +0.262] | +0.239 [−0.008, +0.451] |
| Split-half of the measured cells | +0.520 [+0.299, +0.693]* | +0.396 [+0.267, +0.512]* | +0.536 [+0.399, +0.653]* |
| Rewired model − training-set ridge | −0.069 [−0.180, +0.066] | +0.151 [−0.045, +0.341] | +0.061 [−0.089, +0.225] |
| Rewired model − cross-validated ridge | +0.056 [−0.170, +0.278] | +0.114 [−0.039, +0.283] | +0.031 [−0.128, +0.192] |

**c** | Tahoe-100M, compound input replaced or removed.

| Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Rewired model, its own fingerprint | +0.320 [+0.170, +0.443]* | +0.286 | +0.312 |
| Rewired model, fingerprint of another held-out compound | +0.360 [+0.240, +0.461]* | +0.214 | +0.328 |
| Rewired model, no fingerprint | +0.355 [+0.253, +0.438]* | +0.249 | +0.326 |
| Original model, its own fingerprint | −0.075 | −0.058 | −0.107 |
| Original model, fingerprint of another held-out compound | −0.075 | −0.054 | −0.072 |
| Original model, no fingerprint | −0.136 | −0.114 | −0.117 |
| Cell-line vehicle average (no model) | +0.044 | −0.043 | −0.009 |
| Rewired model: own − another compound’s fingerprint | −0.040 [−0.176, +0.102] | — | — |
| Rewired model: own − no fingerprint | −0.035 [−0.141, +0.067] | — | — |

**d** | Tahoe-100M by dose (splits 1 / 0 / 2).

| Entry | 50 nM | 500 nM | 5,000 nM |
|---|---|---|---|
| Original model | −0.069 / −0.097 / −0.155* | −0.076 / −0.093 / −0.123* | −0.098 / −0.015 / −0.091 |
| Rewired model | +0.169 / +0.128 / +0.173 | +0.475* / +0.400* / +0.557* | +0.318* / +0.336* / +0.213 |
| Rewired model with classifier-free guidance (weight 2) | +0.135 / +0.122 / +0.080 | +0.332* / +0.308* / +0.446* | +0.224 / +0.296* / +0.134 |
| Rewired model with cross-attention | +0.189* / +0.083 / +0.183* | +0.476* / +0.399* / +0.562* | +0.320* / +0.330* / +0.216 |
| Rewired model, parameter-matched control without attention | +0.185* / +0.121 / +0.173* | +0.491* / +0.414* / +0.575* | +0.319* / +0.332* / +0.218 |
| Rewired model with target annotation | +0.192* / +0.162* / +0.163 | +0.512* / +0.421* / +0.541* | +0.327* / +0.331* / +0.174 |
| Rewired model, target sets reassigned between compounds | +0.134 / +0.137 / +0.193* | +0.458* / +0.411* / +0.564* | +0.313* / +0.318* / +0.149 |
| Training-set ridge | +0.076 / +0.146 / −0.014 | +0.223* / +0.218* / −0.027 | +0.162 / +0.155 / +0.192 |
| Cross-validated ridge | +0.031 / +0.147 / −0.015 | −0.083 / +0.016 / +0.224* | −0.097 / −0.150 / +0.074 |

**e** | Tahoe-100M, conditions with a plate replicate.

| Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Original model | −0.046 [−0.156, +0.075] | +0.044 [−0.123, +0.237] | +0.014 [−0.188, +0.170] |
| Rewired model | −0.069 [−0.416, +0.311] | +0.134 [−0.304, +0.514] | +0.154 [−0.228, +0.509] |
| Rewired model with classifier-free guidance (weight 2) | −0.084 [−0.447, +0.318] | +0.146 [−0.304, +0.528] | +0.127 [−0.242, +0.477] |
| Rewired model with cross-attention | −0.094 [−0.426, +0.288] | +0.132 [−0.306, +0.512] | +0.167 [−0.208, +0.524] |
| Rewired model, parameter-matched control without attention | −0.048 [−0.390, +0.321] | +0.071 [−0.335, +0.436] | +0.128 [−0.235, +0.474] |
| Rewired model with target annotation | −0.025 [−0.355, +0.320] | +0.127 [−0.281, +0.495] | +0.018 [−0.293, +0.339] |
| Rewired model, target sets reassigned between compounds | −0.061 [−0.453, +0.330] | +0.048 [−0.356, +0.420] | −0.065 [−0.378, +0.269] |
| Training-set ridge | +0.254 [−0.202, +0.580] | +0.203 [−0.124, +0.521] | +0.157 [−0.236, +0.470] |
| Cross-validated ridge | +0.028 [−0.376, +0.384] | −0.297 [−0.540, +0.019] | +0.063 [−0.311, +0.400] |
| Plate replicate | +0.483 [+0.074, +0.685]* | +0.483 [+0.039, +0.708]* | +0.328 [−0.061, +0.557] |
| Rewired model − plate replicate | −0.552 [−0.935, −0.004]* | −0.349 [−0.861, +0.193] | −0.174 [−0.610, +0.322] |

Sources: step16_analysis.json (sha256 561a2159e359965c; part a), step17_mra.json (sha256 9d2744b9fbd95619; part b), step24_swap.json (sha256 8055d34ceec1c7e7), step25_swap.json (sha256 abf409df3a8a06dd) and step26_source.json (sha256 2604f5cf8535d4ac; part c), step16_diag_mra_dose.json (sha256 f4489af6d3eddb6a; part d) and step16_diag_mra_ceiling.json (sha256 20f4406f38ded4d6; part e). The values of the original model, the rewired model, the two ridges, the plate replicate and the split-half entry in parts a and b equal those Supplementary Fig. S2 was drawn from (fig5_values.csv, panel b). Part d shows values and asterisks only; its intervals are in supp_table_S16.csv in the code repository. Assembled by make_supp_table_s16.py.
