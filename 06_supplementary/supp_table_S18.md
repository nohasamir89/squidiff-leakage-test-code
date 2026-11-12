Supplementary Table S18 | Recognition by RNA fingerprinting on Tahoe-100M (Supplementary Note 6; Supplementary Fig. S3). For each split, one fingerprint per held-out chemical structure was learned from half of the measured cells of every held-out condition; the other half, the generated populations and the references were then queried, each with the same number of cells per condition (at most 100). A query is recognised when its true structure is in the top credible set with a positive sign; a query that receives no credible set is not recognised. The chance level of a query is the number of structures in its top credible set with a positive sign, divided by the number of structures with a fingerprint. The low-signal filter of the method kept fingerprints for 63, 66 and 65 of the 75 held-out structures on splits 1, 0 and 2, and all values are averages over the 2,477, 2,786 and 2,539 conditions of those structures (of 2,899, 3,115 and 2,853); values of a model are also averaged over three seeds. Parts a and b give values for splits 1 / 0 / 2. a, Whole populations, one query per condition: the share of conditions recognised, the chance level, their difference and the share of queries to which no compound was assigned. b, Single cells: the first ten cells of each query population (the ridges were not queried at this level). c, Comparisons of recognised minus chance, in percentage points with 97.5 % percentile bootstrap intervals (5,000 replicates, resampling held-out chemical structures). An asterisk marks an interval that excludes zero; it is set from the unrounded limits, so a limit printed as 0.00 may lie on either side of zero. The outcome fixed in advance rested on four of these comparisons: the first three rows for whole populations and the first row for single cells. d, The same comparisons on the recognition rate without the correction for chance, and the rewired model against the compound-blind average under two other rules of recognition: the true structure in the top credible set with either sign, and the true structure in any credible set with a positive sign. All comparisons of parts c and d, and the use of recognised minus chance for the four deciding ones, were specified before the data were scored.

**a** | Whole populations (splits 1 / 0 / 2).

| Query set | Recognised (%) | Chance (%) | Recognised − chance (percentage points) | No compound assigned (%) |
|---|---|---|---|---|
| Measured cells (other half) | 8.20 / 11.95 / 7.48 | 1.34 / 1.54 / 1.28 | 6.85 / 10.41 / 6.20 | 45.6 / 42.6 / 48.3 |
| Rewired model | 1.47 / 2.76 / 2.05 | 1.12 / 1.44 / 1.21 | 0.34 / 1.32 / 0.84 | 59.3 / 53.5 / 56.8 |
| Rewired model with optimal-transport pairing | 2.64 / 3.45 / 2.38 | 1.16 / 1.34 / 1.11 | 1.48 / 2.11 / 1.27 | 58.6 / 57.2 / 60.7 |
| Rewired model with minibatch optimal-transport pairing (m = 32) | 2.42 / 3.19 / 2.11 | 1.13 / 1.34 / 1.10 | 1.29 / 1.85 / 1.01 | 59.1 / 57.1 / 59.3 |
| Compound-blind average | 2.02 / 1.87 / 2.01 | 1.15 / 1.10 / 1.09 | 0.87 / 0.77 / 0.92 | 52.7 / 55.5 / 58.1 |
| Unaltered vehicle cells | 1.29 / 0.86 / 1.38 | 0.69 / 0.54 / 0.74 | 0.60 / 0.32 / 0.64 | 58.7 / 63.5 / 64.9 |
| Training-set ridge | 2.83 / 2.87 / 2.72 | 1.10 / 0.93 / 1.08 | 1.73 / 1.95 / 1.63 | 54.1 / 57.0 / 58.6 |
| Cross-validated ridge | 2.06 / 1.90 / 2.05 | 1.11 / 1.06 / 1.09 | 0.95 / 0.85 / 0.95 | 52.6 / 53.6 / 58.6 |

**b** | Single cells (splits 1 / 0 / 2).

| Query set | Recognised (%) | Chance (%) | Recognised − chance (percentage points) | No compound assigned (%) |
|---|---|---|---|---|
| Measured cells (other half) | 1.58 / 1.84 / 1.29 | 0.64 / 0.70 / 0.64 | 0.94 / 1.14 / 0.66 | 77.2 / 80.1 / 79.6 |
| Rewired model | 0.70 / 0.82 / 0.71 | 0.58 / 0.61 / 0.56 | 0.12 / 0.22 / 0.15 | 80.3 / 83.7 / 82.5 |
| Rewired model with optimal-transport pairing | 0.81 / 0.92 / 0.86 | 0.61 / 0.62 / 0.59 | 0.20 / 0.30 / 0.27 | 79.4 / 82.9 / 82.0 |
| Rewired model with minibatch optimal-transport pairing (m = 32) | 0.81 / 0.95 / 0.84 | 0.60 / 0.62 / 0.58 | 0.21 / 0.33 / 0.26 | 79.7 / 83.3 / 82.2 |
| Compound-blind average | 0.75 / 0.87 / 0.70 | 0.59 / 0.63 / 0.54 | 0.17 / 0.24 / 0.17 | 77.9 / 82.2 / 81.5 |
| Unaltered vehicle cells | 0.71 / 0.81 / 0.65 | 0.55 / 0.56 / 0.50 | 0.16 / 0.25 / 0.14 | 77.6 / 82.4 / 81.4 |

**c** | Comparisons of recognised minus chance (percentage points).

| Queries | Comparison | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|---|
| Whole populations | Measured cells: recognised − chance | +6.85 [+4.74, +9.51]* | +10.41 [+6.20, +16.18]* | +6.20 [+4.04, +8.76]* |
| Whole populations | Measured cells − compound-blind average | +5.98 [+3.87, +8.62]* | +9.65 [+5.40, +15.45]* | +5.28 [+2.87, +8.02]* |
| Whole populations | Rewired model − compound-blind average | −0.52 [−1.35, +0.32] | +0.56 [−1.06, +3.54] | −0.08 [−1.32, +1.92] |
| Whole populations | Rewired model − unaltered vehicle cells | −0.26 [−0.96, +0.50] | +1.00 [−0.59, +3.94] | +0.20 [−1.06, +2.27] |
| Whole populations | Rewired model − training-set ridge | −1.38 [−2.41, −0.42]* | −0.62 [−1.55, +0.26] | −0.79 [−1.64, −0.01]* |
| Whole populations | Rewired model − cross-validated ridge | −0.61 [−1.41, +0.24] | +0.48 [−1.15, +3.42] | −0.11 [−1.42, +1.90] |
| Whole populations | Optimal-transport pairing − rewired model | +1.14 [+0.45, +2.06]* | +0.78 [+0.24, +1.42]* | +0.42 [+0.04, +0.81]* |
| Whole populations | Minibatch optimal-transport pairing (m = 32) − rewired model | +0.95 [+0.37, +1.72]* | +0.53 [+0.05, +1.08]* | +0.17 [−0.22, +0.56] |
| Whole populations | Measured cells − rewired model | +6.51 [+4.41, +9.15]* | +9.09 [+5.52, +14.18]* | +5.36 [+3.08, +7.90]* |
| Single cells | Rewired model − compound-blind average | −0.05 [−0.22, +0.09] | −0.03 [−0.23, +0.17] | −0.02 [−0.24, +0.16] |
| Single cells | Rewired model − unaltered vehicle cells | −0.04 [−0.21, +0.10] | −0.03 [−0.24, +0.17] | +0.01 [−0.21, +0.19] |
| Single cells | Optimal-transport pairing − rewired model | +0.08 [−0.06, +0.25] | +0.08 [−0.03, +0.20] | +0.12 [0.00, +0.25]* |
| Single cells | Minibatch optimal-transport pairing (m = 32) − rewired model | +0.09 [−0.02, +0.22] | +0.11 [+0.02, +0.21]* | +0.11 [+0.02, +0.20]* |
| Single cells | Measured cells − rewired model | +0.83 [+0.48, +1.29]* | +0.93 [+0.55, +1.46]* | +0.51 [+0.25, +0.80]* |

**d** | Comparisons without the correction for chance (percentage points).

| Queries | Comparison | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|---|
| Whole populations | Measured cells − compound-blind average | +6.18 [+4.03, +8.84]* | +10.09 [+5.78, +15.90]* | +5.47 [+3.03, +8.25]* |
| Whole populations | Rewired model − compound-blind average | −0.55 [−1.38, +0.29] | +0.90 [−0.73, +3.79] | +0.04 [−1.26, +2.08] |
| Whole populations | Rewired model − unaltered vehicle cells | +0.17 [−0.50, +0.92] | +1.90 [+0.32, +4.81]* | +0.67 [−0.63, +2.81] |
| Whole populations | Rewired model − training-set ridge | −1.36 [−2.41, −0.37]* | −0.11 [−1.01, +0.75] | −0.67 [−1.57, +0.17] |
| Whole populations | Rewired model − cross-validated ridge | −0.59 [−1.42, +0.29] | +0.86 [−0.79, +3.78] | 0.00 [−1.35, +2.04] |
| Whole populations | Optimal-transport pairing − rewired model | +1.17 [+0.46, +2.13]* | +0.68 [+0.08, +1.36]* | +0.33 [−0.06, +0.72] |
| Whole populations | Minibatch optimal-transport pairing (m = 32) − rewired model | +0.96 [+0.35, +1.74]* | +0.43 [−0.07, +1.00] | +0.07 [−0.31, +0.44] |
| Whole populations | Measured cells − rewired model | +6.73 [+4.62, +9.39]* | +9.19 [+5.57, +14.32]* | +5.44 [+3.12, +8.04]* |
| Whole populations | Rewired model − compound-blind average, either sign counted | −0.75 [−1.58, +0.11] | +0.85 [−0.82, +3.76] | +0.08 [−1.20, +2.14] |
| Whole populations | Rewired model − compound-blind average, any credible set counted | −0.77 [−1.66, +0.14] | +1.27 [−0.60, +4.84] | −0.03 [−1.41, +2.23] |
| Single cells | Rewired model − compound-blind average | −0.06 [−0.22, +0.09] | −0.05 [−0.28, +0.16] | +0.01 [−0.22, +0.19] |
| Single cells | Rewired model − unaltered vehicle cells | −0.01 [−0.18, +0.14] | +0.01 [−0.21, +0.23] | +0.07 [−0.16, +0.25] |
| Single cells | Optimal-transport pairing − rewired model | +0.11 [−0.03, +0.27] | +0.10 [−0.02, +0.22] | +0.15 [+0.03, +0.28]* |
| Single cells | Minibatch optimal-transport pairing (m = 32) − rewired model | +0.11 [0.00, +0.24] | +0.13 [+0.03, +0.23]* | +0.12 [+0.04, +0.22]* |
| Single cells | Measured cells − rewired model | +0.88 [+0.53, +1.34]* | +1.02 [+0.62, +1.57]* | +0.58 [+0.31, +0.88]* |
| Single cells | Rewired model − compound-blind average, either sign counted | −0.14 [−0.34, +0.03] | −0.07 [−0.29, +0.15] | −0.17 [−0.42, +0.03] |
| Single cells | Rewired model − compound-blind average, any credible set counted | −0.05 [−0.21, +0.10] | −0.06 [−0.28, +0.16] | +0.02 [−0.22, +0.20] |

Sources: step22_analysis.json (sha256 cec58c9d4a74dcb4). The values of part a for the share recognised, the chance level and the share with no compound assigned equal those Supplementary Fig. S3 was drawn from (fig5_values.csv, panel d). The record stores fractions; supp_table_S18.csv in the code repository holds them at full precision. Assembled by make_supp_table_s18.py.
