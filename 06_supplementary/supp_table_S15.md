**Supplementary Table S15a** | Score per arm, mean over conditions (splits 1 / 0 / 2; 0.5 = chance).

| Arm | Cross-plate, published (S-orig) | Cross-plate, vehicle halves (S-split) | Within plate (S-split) | Cosine with the plate offset (S-split) |
|---|---|---|---|---|
| Rewired model | 0.722 / 0.718 / 0.714 | 0.718 / 0.713 / 0.711 | 0.503 / 0.517 / 0.526 | 0.628 / 0.635 / 0.623 |
| Rewired model, fingerprint of another held-out compound | 0.716 / 0.701 / 0.691 | 0.712 / 0.697 / 0.687 | 0.489 / 0.491 / 0.486 | 0.614 / 0.636 / 0.607 |
| Rewired model, no fingerprint | 0.736 / 0.718 / 0.714 | 0.732 / 0.713 / 0.711 | 0.497 / 0.498 / 0.497 | 0.722 / 0.712 / 0.701 |
| Rewired model, encoder given the measured treated cells | 0.737 / 0.730 / 0.726 | 0.733 / 0.726 / 0.722 | 0.525 / 0.531 / 0.538 | 0.626 / 0.634 / 0.620 |
| Original model | 0.641 / 0.644 / 0.655 | 0.639 / 0.642 / 0.653 | 0.493 / 0.508 / 0.520 | 0.251 / 0.259 / 0.264 |
| Original model, fingerprint of another held-out compound | 0.643 / 0.636 / 0.644 | 0.640 / 0.634 / 0.642 | 0.496 / 0.501 / 0.499 | 0.249 / 0.259 / 0.262 |
| Original model, no fingerprint | 0.643 / 0.637 / 0.646 | 0.641 / 0.634 / 0.644 | 0.497 / 0.499 / 0.502 | 0.246 / 0.261 / 0.261 |
| Original model, encoder given the measured treated cells | 0.656 / 0.659 / 0.669 | 0.654 / 0.657 / 0.666 | 0.505 / 0.521 / 0.530 | 0.260 / 0.267 / 0.271 |
| Rewired model with classifier-free guidance (weight 2) | 0.686 / 0.714 / 0.694 | — | — | — |
| Rewired model with cross-attention | 0.726 / 0.723 / 0.720 | — | — | — |
| Rewired model, parameter-matched control without attention | 0.727 / 0.718 / 0.711 | — | — | — |
| Training-set ridge | 0.516 / 0.540 / 0.526 | — | — | — |
| Cross-validated ridge | 0.504 / 0.492 / 0.486 | — | — | — |
| Compound-blind average | 0.500 / 0.500 / 0.500 | — | — | — |
| Cell-line vehicle average (no model) | 0.751 / 0.733 / 0.732 | 0.747 / 0.728 / 0.729 | 0.500 / 0.500 / 0.500 | 0.982 / 0.983 / 0.981 |
| Resample of the condition's vehicle pool (no model) | 0.499 / 0.502 / 0.495 | 0.485 / 0.483 / 0.479 | 0.502 / 0.505 / 0.494 | −0.021 / −0.007 / −0.017 |
| Second draw of the measured cells (ceiling) | 0.953 / 0.962 / 0.953 | 0.951 / 0.960 / 0.950 | 0.884 / 0.910 / 0.891 | 0.313 / 0.277 / 0.294 |
| Measured change of the real treated cells | — | — | — | 0.364 / 0.320 / 0.338 |
| Vehicle resample from the measured-change half (null) | — | — | — | −0.003 / 0.000 / 0.002 |

**Supplementary Table S15b** | Pre-registered contrasts, 97.5 % structure-clustered bootstrap intervals (* excludes zero).

| Contrast | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| **Published construction (cross-plate, S-orig)** | | | |
| Rewired model − 0.5 | +0.222 [+0.188, +0.254]* | +0.218 [+0.178, +0.259]* | +0.214 [+0.183, +0.245]* |
| Original model − 0.5 | +0.141 [+0.104, +0.181]* | +0.144 [+0.105, +0.184]* | +0.155 [+0.110, +0.198]* |
| Training-set ridge − 0.5 | +0.016 [−0.033, +0.064] | +0.040 [−0.015, +0.098] | +0.026 [−0.028, +0.075] |
| Cross-validated ridge − 0.5 | +0.004 [−0.055, +0.062] | −0.008 [−0.057, +0.044] | −0.014 [−0.071, +0.042] |
| Compound-blind average − 0.5 | 0.000 [−0.062, +0.060] | 0.000 [−0.055, +0.057] | 0.000 [−0.065, +0.059] |
| Rewired model − training-set ridge | +0.206 [+0.151, +0.262]* | +0.178 [+0.110, +0.240]* | +0.188 [+0.131, +0.246]* |
| Rewired model − cross-validated ridge | +0.218 [+0.149, +0.289]* | +0.227 [+0.170, +0.282]* | +0.228 [+0.165, +0.289]* |
| Rewired model with guidance − 0.5 | +0.186 [+0.149, +0.222]* | +0.214 [+0.178, +0.251]* | +0.194 [+0.160, +0.228]* |
| Rewired model with guidance − rewired model | −0.036 [−0.053, −0.023]* | −0.004 [−0.023, +0.023] | −0.020 [−0.038, −0.006]* |
| Rewired model with cross-attention − 0.5 | +0.226 [+0.191, +0.259]* | +0.223 [+0.183, +0.261]* | +0.220 [+0.188, +0.251]* |
| Parameter-matched control − 0.5 | +0.227 [+0.195, +0.258]* | +0.218 [+0.172, +0.258]* | +0.211 [+0.180, +0.241]* |
| Cross-attention − parameter-matched control (exploratory) | −0.001 [−0.008, +0.004] | +0.005 [−0.003, +0.015] | +0.009 [+0.001, +0.018]* |
| Rewired model, another compound’s fingerprint − 0.5 | +0.216 [+0.186, +0.246]* | +0.201 [+0.157, +0.239]* | +0.191 [+0.156, +0.225]* |
| Rewired model, no fingerprint − 0.5 | +0.236 [+0.208, +0.265]* | +0.218 [+0.176, +0.254]* | +0.214 [+0.180, +0.247]* |
| Rewired model: own − another compound’s fingerprint | +0.006 [−0.015, +0.029] | +0.017 [−0.004, +0.041] | +0.024 [−0.003, +0.053] |
| Rewired model: own − no fingerprint | −0.014 [−0.033, +0.004] | 0.000 [−0.020, +0.021] | 0.000 [−0.020, +0.020] |
| Original model, another compound’s fingerprint − 0.5 | +0.143 [+0.103, +0.180]* | +0.136 [+0.096, +0.177]* | +0.144 [+0.099, +0.187]* |
| Original model: own − another compound’s fingerprint | −0.001 [−0.009, +0.007] | +0.008 [−0.004, +0.023] | +0.011 [−0.001, +0.025] |
| Rewired model − original model | +0.081 [+0.036, +0.124]* | +0.074 [+0.018, +0.124]* | +0.059 [+0.017, +0.103]* |
| (Rewired − original): own − another compound’s fingerprint | +0.007 [−0.012, +0.027] | +0.009 [−0.008, +0.026] | +0.013 [−0.009, +0.035] |
| Cell-line vehicle average − 0.5 | +0.251 [+0.224, +0.279]* | +0.233 [+0.202, +0.262]* | +0.232 [+0.201, +0.262]* |
| Vehicle resample − 0.5 | −0.001 [−0.011, +0.009] | +0.002 [−0.010, +0.015] | −0.005 [−0.020, +0.009] |
| Rewired model − cell-line vehicle average | −0.029 [−0.054, −0.005]* | −0.015 [−0.040, +0.009] | −0.018 [−0.044, +0.007] |
| **Cross-plate, disjoint vehicle halves (S-split)** | | | |
| Rewired model − 0.5 | +0.218 [+0.184, +0.250]* | +0.213 [+0.172, +0.253]* | +0.211 [+0.180, +0.240]* |
| Rewired model, another compound’s fingerprint − 0.5 | +0.212 [+0.183, +0.242]* | +0.197 [+0.153, +0.234]* | +0.187 [+0.152, +0.221]* |
| Rewired model: own − another compound’s fingerprint | +0.006 [−0.016, +0.028] | +0.016 [−0.005, +0.040] | +0.024 [−0.003, +0.053] |
| Original model − 0.5 | +0.139 [+0.100, +0.177]* | +0.142 [+0.103, +0.182]* | +0.153 [+0.109, +0.194]* |
| **Within plate (S-split unless stated)** | | | |
| Rewired model − 0.5 | +0.003 [−0.041, +0.048] | +0.017 [−0.039, +0.071] | +0.026 [−0.021, +0.076] |
| Original model − 0.5 | −0.007 [−0.061, +0.047] | +0.008 [−0.044, +0.062] | +0.020 [−0.035, +0.077] |
| Rewired model, no fingerprint − 0.5 | −0.003 [−0.042, +0.038] | −0.002 [−0.053, +0.048] | −0.003 [−0.045, +0.041] |
| Rewired model − original model | +0.010 [−0.053, +0.069] | +0.009 [−0.072, +0.082] | +0.006 [−0.054, +0.065] |
| Rewired model: own − another compound’s fingerprint | +0.014 [−0.016, +0.042] | +0.027 [−0.001, +0.058] | +0.041 [+0.005, +0.082]* |
| Rewired model: own − another compound’s fingerprint (S-flip) | +0.015 [−0.015, +0.043] | +0.029 [+0.001, +0.062]* | +0.041 [+0.007, +0.082]* |
| Original model: own − another compound’s fingerprint | −0.003 [−0.012, +0.005] | +0.007 [−0.010, +0.028] | +0.021 [+0.005, +0.038]* |
| Cell-line vehicle average − 0.5 (S-orig) | 0.000 [−0.039, +0.041] | 0.000 [−0.040, +0.041] | 0.000 [−0.044, +0.044] |
| Vehicle resample − 0.5 | +0.002 [−0.009, +0.014] | +0.005 [−0.010, +0.020] | −0.006 [−0.022, +0.010] |
| Second draw of the measured cells − 0.5 (ceiling) | +0.384 [+0.367, +0.400]* | +0.410 [+0.391, +0.426]* | +0.391 [+0.372, +0.409]* |
| **Alignment with the plate offset (cosine, S-split)** | | | |
| Measured change of the real treated cells | +0.364 [+0.318, +0.410]* | +0.320 [+0.279, +0.357]* | +0.338 [+0.291, +0.381]* |
| Vehicle resample from the measured-change half (null) | −0.003 [−0.011, +0.006] | 0.000 [−0.007, +0.008] | +0.002 [−0.007, +0.011] |

**Supplementary Table S15c** | Layout of the candidate sets and the shares quoted in the text (point estimates).

| Quantity | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|
| Conditions scored within a plate, of all held-out conditions | 2,645 of 2,899 | 2,678 of 3,115 | 2,532 of 2,853 |
| Plates per cell line and dose: median (range) | 5 (3–5) | 5 (3–12) | 5 (3–5) |
| Share of candidate pairs measured on the condition's own plate | 0.218 | 0.214 | 0.223 |
| Cell-line vehicle average: distance from chance as a share of the rewired model's (S-orig) | 113.0 % | 107.0 % | 108.3 % |
| Rewired − original difference remaining within a plate, as a share of the cross-plate difference | 11.8 % | 12.7 % | 10.9 % |
| Shared vehicle estimate: rewired model, S-orig minus S-split (cross-plate) | 0.0042 | 0.0049 | 0.0036 |

Sources: step15c_pds.json and step15c_pds_per_condition.csv (Step 15c), step25_swap_summary.csv (Step 25), step26_source_summary.csv and step26_source.json (Step 26), step27_within_plate_summary.csv (Step 27); research commit b474ed5; assembled by make_supp_table_s15.py.
