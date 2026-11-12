Supplementary Table S5 | Gene identity and population distance of the models and the references (Results 1 and 2; Figs 1b, 2a and 2b). Values are means over the held-out conditions of each compound split (splits 1 / 0 / 2: 2,899 / 3,115 / 2,853 conditions on Tahoe-100M; 443 / 444 / 455 on sci-Plex); the generative models are averaged over their seeds within a condition first (three on Tahoe-100M, five on sci-Plex). a, Gene identity: Jaccard index between the 50 genes with the largest absolute predicted change and the 50 with the largest absolute measured change, both relative to the matched vehicle mean. Chance is the expected index of two random sets of 50 genes out of 200. The unaltered vehicle cells have no predicted change and are not scored on this measure. b, Population distance on Tahoe-100M: energy distance between the generated or reference population and the measured treated cells; lower is better. The plate replicate is scored on the conditions measured on both plates (259 / 299 / 246 pairs). c, Contrasts between the rewired model and each reference, each specified before the data were scored, with 97.5 % percentile bootstrap intervals (5,000 replicates, resampling held-out compounds; chemical structures on Tahoe-100M); an asterisk marks an interval that excludes zero.

**a** | Gene identity (splits 1 / 0 / 2).

| Entry | Tahoe-100M | sci-Plex |
|---|---|---|
| Original model | 0.268 / 0.285 / 0.291 | 0.508 / 0.505 / 0.491 |
| Rewired model | 0.464 / 0.460 / 0.458 | 0.610 / 0.612 / 0.611 |
| Training-set ridge | 0.429 / 0.433 / 0.425 | 0.564 / 0.537 / 0.547 |
| Cross-validated ridge | 0.477 / 0.465 / 0.468 | 0.554 / 0.568 / 0.554 |
| Compound-blind average | 0.509 / 0.500 / 0.509 | 0.544 / 0.554 / 0.532 |
| Plate replicate | 0.502 / 0.507 / 0.477 | — |
| Chance | 0.144 | 0.144 |

**b** | Population distance on Tahoe-100M.

| Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Original model | 2.268 | 2.135 | 2.021 |
| Rewired model | 0.225 | 0.244 | 0.218 |
| Training-set ridge | 0.260 | 0.264 | 0.250 |
| Cross-validated ridge | 0.160 | 0.223 | 0.162 |
| Compound-blind average | 0.146 | 0.197 | 0.146 |
| Unaltered vehicle cells | 0.182 | 0.229 | 0.179 |
| Plate replicate | 0.194 | 0.202 | 0.165 |

**c** | Contrasts (point estimate and 97.5 % interval).

| Contrast | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| **Gene identity, Tahoe-100M** |  |  |  |
| Rewired model − training-set ridge | +0.035 [+0.025, +0.045]* | +0.028 [+0.018, +0.036]* | +0.034 [+0.022, +0.045]* |
| Rewired model − cross-validated ridge | −0.013 [−0.027, +0.001] | −0.005 [−0.014, +0.005] | −0.009 [−0.019, +0.001] |
| Rewired model − compound-blind average | −0.045 [−0.052, −0.037]* | −0.040 [−0.048, −0.032]* | −0.050 [−0.058, −0.042]* |
| **Gene identity, sci-Plex** |  |  |  |
| Rewired model − training-set ridge | +0.046 [+0.023, +0.064]* | +0.075 [+0.066, +0.086]* | +0.063 [+0.049, +0.077]* |
| Rewired model − cross-validated ridge | +0.056 [+0.035, +0.072]* | +0.044 [+0.034, +0.054]* | +0.057 [+0.044, +0.070]* |
| Rewired model − compound-blind average | +0.066 [+0.042, +0.086]* | +0.059 [+0.052, +0.065]* | +0.078 [+0.066, +0.089]* |
| **Population distance, Tahoe-100M** |  |  |  |
| Rewired model − training-set ridge | −0.035 [−0.098, +0.022] | −0.020 [−0.065, +0.023] | −0.032 [−0.066, +0.005] |
| Rewired model − cross-validated ridge | +0.065 [+0.020, +0.114]* | +0.021 [−0.060, +0.079] | +0.056 [+0.018, +0.095]* |
| Rewired model − compound-blind average | +0.079 [+0.031, +0.129]* | +0.047 [−0.030, +0.102] | +0.072 [+0.037, +0.110]* |
| Rewired model − unaltered vehicle cells | +0.043 [−0.004, +0.094] | +0.015 [−0.072, +0.073] | +0.039 [+0.001, +0.078]* |

Sources: step15_model_analysis.json (sha256 5fd700b5a030cf5a; Tahoe-100M) and step14_analysis.json (sha256 c7f021b27272bbac; sci-Plex). The values of parts a and b equal those the figures were drawn from (fig1b_values.csv, fig2_values.csv). The sci-Plex record stores each contrast as reference minus rewired model; it is given here with the sign reversed. Assembled by make_supp_table_s5.py.
