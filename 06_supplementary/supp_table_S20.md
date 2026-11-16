
**a** | sci-Plex.

| Item | Statement of the paper | Split | Folds of the paper | Median | Range | Assignments with the verdict of the paper | At least 95 % | Statement |
|---|---|---|---|---|---|---|---|---|
| A1 | Gene identity, rewired model minus ridge: positive, interval above zero | 1 | +0.056 | +0.058 | +0.053 to +0.065 | 200 of 200 | yes | robust |
|  |  | 0 | +0.044 | +0.044 | +0.038 to +0.051 | 200 of 200 | yes |  |
|  |  | 2 | +0.057 | +0.056 | +0.051 to +0.061 | 200 of 200 | yes |  |
| A2 | The same without the compound that has a salt twin in training: positive, interval above zero | 0 | +0.045 | +0.045 | +0.038 to +0.051 | 200 of 200 | yes | robust |
|  |  | 2 | +0.056 | +0.055 | +0.050 to +0.060 | 200 of 200 | yes |  |
| A3 | Gene identity, original model minus ridge: negative | 1 | −0.046 | −0.044 | −0.049 to −0.037 | 200 of 200 | yes | robust |
|  |  | 0 | −0.064 | −0.063 | −0.070 to −0.057 | 200 of 200 | yes |  |
|  |  | 2 | −0.063 | −0.064 | −0.069 to −0.059 | 200 of 200 | yes |  |
| A4 | Response correlation (mean over conditions), highest generative model minus ridge: negative | 1 | −0.071 | −0.072 | −0.096 to −0.018 | 200 of 200 | yes | robust |
|  |  | 0 | −0.123 | −0.121 | −0.145 to −0.089 | 200 of 200 | yes |  |
|  |  | 2 | −0.013 | −0.026 | −0.051 to +0.001 | 199 of 200 | yes |  |
| A5 | Response correlation (median over conditions), highest generative model minus ridge: negative on Splits 1 and 0; on Split 2, cross-attention and its control both above the ridge | 1 | −0.067 | −0.055 | −0.099 to +0.001 | 199 of 200 | yes | not robust |
|  |  | 0 | −0.147 | −0.129 | −0.165 to −0.077 | 200 of 200 | yes |  |
|  |  | 2 | +0.016 | +0.001 | −0.027 to +0.032 | 96 of 200 | no |  |
| A6 | Response-magnitude agreement of the ridge: positive; interval above zero on Split 1, includes zero on Splits 0 and 2 | 1 | +0.278 | +0.237 | −0.002 to +0.361 | 94 of 200 | no | not robust |
|  |  | 0 | +0.127 | +0.074 | −0.053 to +0.197 | 189 of 200 | no |  |
|  |  | 2 | +0.239 | +0.224 | −0.100 to +0.360 | 138 of 200 | no |  |
| C4 | Response-magnitude agreement, rewired model minus ridge: interval includes zero | 1 | +0.056 | +0.097 | −0.027 to +0.336 | 197 of 200 | yes | not robust |
|  |  | 0 | +0.114 | +0.167 | +0.043 to +0.294 | 122 of 200 | no |  |
|  |  | 2 | +0.031 | +0.046 | −0.091 to +0.370 | 198 of 200 | yes |  |
| A7 | Response correlation, feedforward network minus ridge: negative on Split 0 (50 assignments) | 0 | −0.036 | −0.019 | −0.045 to +0.024 | 45 of 50 | no | not robust |

**b** | Tahoe-100M.

| Item | Statement of the paper | Split | Folds of the paper | Median | Range | Assignments with the verdict of the paper | At least 95 % | Statement |
|---|---|---|---|---|---|---|---|---|
| B1 | Gene identity, compound-blind average minus ridge: positive | 1 | +0.031 | +0.031 | +0.026 to +0.038 | 200 of 200 | yes | robust |
|  |  | 0 | +0.035 | +0.034 | +0.028 to +0.040 | 200 of 200 | yes |  |
|  |  | 2 | +0.041 | +0.039 | +0.033 to +0.043 | 200 of 200 | yes |  |
| B2 | Gene identity, original model minus ridge: negative | 1 | −0.209 | −0.209 | −0.214 to −0.203 | 200 of 200 | yes | robust |
|  |  | 0 | −0.180 | −0.181 | −0.187 to −0.175 | 200 of 200 | yes |  |
|  |  | 2 | −0.177 | −0.179 | −0.185 to −0.175 | 200 of 200 | yes |  |
| C1 | Gene identity, rewired model minus ridge: interval includes zero | 1 | −0.013 | −0.014 | −0.018 to −0.007 | 101 of 200 | no | not robust |
|  |  | 0 | −0.005 | −0.005 | −0.011 to 0.000 | 198 of 200 | yes |  |
|  |  | 2 | −0.009 | −0.011 | −0.017 to −0.007 | 40 of 200 | no |  |
| C2 | Population distance, rewired model minus ridge: positive; interval above zero on Splits 1 and 2, includes zero on Split 0 | 1 | +0.065 | +0.067 | +0.059 to +0.071 | 200 of 200 | yes | robust |
|  |  | 0 | +0.021 | +0.021 | +0.015 to +0.029 | 200 of 200 | yes |  |
|  |  | 2 | +0.056 | +0.058 | +0.052 to +0.062 | 200 of 200 | yes |  |
| B3 | Discrimination score of the ridge minus 0.5: interval includes zero | 1 | +0.004 | 0.000 | −0.021 to +0.018 | 200 of 200 | yes | robust |
|  |  | 0 | −0.008 | −0.006 | −0.025 to +0.022 | 200 of 200 | yes |  |
|  |  | 2 | −0.014 | −0.013 | −0.032 to +0.007 | 200 of 200 | yes |  |
| C3 | Discrimination score, rewired model minus ridge: positive, interval above zero | 1 | +0.218 | +0.222 | +0.205 to +0.243 | 200 of 200 | yes | robust |
|  |  | 0 | +0.227 | +0.224 | +0.196 to +0.243 | 200 of 200 | yes |  |
|  |  | 2 | +0.228 | +0.227 | +0.207 to +0.246 | 200 of 200 | yes |  |
| B4 | Response-magnitude agreement of the ridge: interval includes zero | 1 | −0.050 | −0.010 | −0.175 to +0.094 | 197 of 200 | yes | not robust |
|  |  | 0 | +0.004 | −0.019 | −0.118 to +0.085 | 200 of 200 | yes |  |
|  |  | 2 | +0.094 | +0.068 | −0.070 to +0.194 | 186 of 200 | no |  |

**c** | The cross-validated ridge itself.

| Dataset | Measure | Split | Folds of the paper | Median | 2.5th to 97.5th percentile | Range | Assignments at or below the paper |
|---|---|---|---|---|---|---|---|
| sci-Plex | Gene identity | 1 | 0.554 | 0.552 | 0.547 to 0.556 | 0.545 to 0.557 | 175 of 200 |
|  |  | 0 | 0.568 | 0.568 | 0.564 to 0.572 | 0.561 to 0.575 | 119 of 200 |
|  |  | 2 | 0.554 | 0.555 | 0.551 to 0.559 | 0.549 to 0.560 | 60 of 200 |
| sci-Plex | Response correlation, mean over conditions | 1 | 0.189 | 0.190 | 0.148 to 0.211 | 0.136 to 0.214 | 95 of 200 |
|  |  | 0 | 0.202 | 0.200 | 0.183 to 0.218 | 0.169 to 0.225 | 122 of 200 |
|  |  | 2 | 0.098 | 0.112 | 0.090 to 0.131 | 0.085 to 0.136 | 21 of 200 |
| sci-Plex | Response correlation, median over conditions | 1 | 0.188 | 0.177 | 0.149 to 0.211 | 0.120 to 0.221 | 152 of 200 |
|  |  | 0 | 0.247 | 0.230 | 0.203 to 0.257 | 0.177 to 0.266 | 167 of 200 |
|  |  | 2 | 0.082 | 0.097 | 0.073 to 0.118 | 0.066 to 0.125 | 25 of 200 |
| sci-Plex | Response-magnitude agreement | 1 | 0.278 | 0.237 | 0.042 to 0.344 | −0.002 to 0.361 | 152 of 200 |
|  |  | 0 | 0.127 | 0.074 | 0.016 to 0.174 | −0.053 to 0.197 | 176 of 200 |
|  |  | 2 | 0.239 | 0.224 | 0.075 to 0.307 | −0.100 to 0.360 | 125 of 200 |
| Tahoe-100M | Gene identity | 1 | 0.477 | 0.478 | 0.473 to 0.481 | 0.471 to 0.482 | 77 of 200 |
|  |  | 0 | 0.465 | 0.466 | 0.462 to 0.469 | 0.460 to 0.472 | 76 of 200 |
|  |  | 2 | 0.468 | 0.470 | 0.466 to 0.474 | 0.465 to 0.476 | 24 of 200 |
| Tahoe-100M | Response correlation, mean over conditions | 1 | 0.411 | 0.413 | 0.393 to 0.429 | 0.387 to 0.435 | 88 of 200 |
|  |  | 0 | 0.378 | 0.367 | 0.347 to 0.393 | 0.339 to 0.405 | 158 of 200 |
|  |  | 2 | 0.398 | 0.407 | 0.388 to 0.425 | 0.382 to 0.428 | 39 of 200 |
| Tahoe-100M | Response correlation, median over conditions | 1 | 0.501 | 0.512 | 0.482 to 0.535 | 0.462 to 0.542 | 50 of 200 |
|  |  | 0 | 0.467 | 0.461 | 0.438 to 0.496 | 0.416 to 0.514 | 120 of 200 |
|  |  | 2 | 0.503 | 0.519 | 0.494 to 0.544 | 0.479 to 0.551 | 28 of 200 |
| Tahoe-100M | Population distance | 1 | 0.160 | 0.158 | 0.155 to 0.164 | 0.154 to 0.166 | 148 of 200 |
|  |  | 0 | 0.223 | 0.223 | 0.217 to 0.228 | 0.215 to 0.229 | 100 of 200 |
|  |  | 2 | 0.162 | 0.160 | 0.157 to 0.164 | 0.156 to 0.166 | 178 of 200 |
| Tahoe-100M | Discrimination score | 1 | 0.504 | 0.500 | 0.483 to 0.511 | 0.479 to 0.518 | 150 of 200 |
|  |  | 0 | 0.492 | 0.494 | 0.480 to 0.513 | 0.475 to 0.522 | 82 of 200 |
|  |  | 2 | 0.486 | 0.487 | 0.475 to 0.501 | 0.468 to 0.507 | 84 of 200 |
| Tahoe-100M | Response-magnitude agreement | 1 | −0.050 | −0.010 | −0.128 to 0.075 | −0.175 to 0.094 | 45 of 200 |
|  |  | 0 | 0.004 | −0.019 | −0.096 to 0.064 | −0.118 to 0.085 | 142 of 200 |
|  |  | 2 | 0.094 | 0.068 | −0.040 to 0.166 | −0.070 to 0.194 | 141 of 200 |

Sources: step29_fold_sensitivity_sciplex.json (sha256 73bd35c74988db00; parts a and c) and step29_fold_sensitivity_tahoe.json (sha256 a88bbfd8221bca59; parts b and c); 200 random assignments per split, 50 for the feedforward network. The .csv file holds every value at full precision, with the 2.5th and 97.5th percentiles of parts a and b. Assembled by make_supp_table_s20.py.
