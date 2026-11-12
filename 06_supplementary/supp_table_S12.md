Supplementary Table S12 | The rewired model against the original model on Tahoe-100M (Results 4; Fig. 4b). Gene identity: Jaccard index between the 50 genes with the largest absolute predicted change and the 50 with the largest absolute measured change. Response correlation: Pearson correlation between the predicted and the measured mean change over the 50 genes with the largest absolute measured change. Population distance: energy distance between the generated and the measured treated cells; lower is better. a, Mean over the held-out conditions of each compound split (2,899, 3,115 and 2,853 conditions on splits 1, 0 and 2), each model averaged over its three seeds within a condition first, and the difference between the two models, specified before the data were scored, with its 97.5 % percentile bootstrap interval (5,000 replicates, resampling held-out chemical structures); an asterisk marks an interval that excludes zero. b, The same means for each training seed (seeds 0 / 1 / 2).

**a** | Values and differences.

| Measure | Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|---|
| Gene identity | Original model | 0.268 | 0.285 | 0.291 |
|  | Rewired model | 0.464 | 0.460 | 0.458 |
|  | Rewired − original | +0.196 [+0.184, +0.207]* | +0.175 [+0.165, +0.186]* | +0.168 [+0.157, +0.178]* |
| Response correlation | Original model | 0.227 | 0.249 | 0.271 |
|  | Rewired model | 0.520 | 0.501 | 0.523 |
|  | Rewired − original | +0.293 [+0.263, +0.322]* | +0.252 [+0.219, +0.283]* | +0.253 [+0.223, +0.282]* |
| Population distance | Original model | 2.268 | 2.135 | 2.021 |
|  | Rewired model | 0.225 | 0.244 | 0.218 |
|  | Rewired − original | −2.043 [−2.113, −1.975]* | −1.892 [−1.949, −1.828]* | −1.803 [−1.864, −1.743]* |

**b** | Values per seed (seeds 0 / 1 / 2).

| Measure | Entry | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|---|
| Gene identity | Original model | 0.264 / 0.271 / 0.270 | 0.277 / 0.283 / 0.295 | 0.289 / 0.287 / 0.296 |
|  | Rewired model | 0.464 / 0.463 / 0.465 | 0.458 / 0.462 / 0.461 | 0.458 / 0.460 / 0.458 |
| Response correlation | Original model | 0.276 / 0.219 / 0.186 | 0.302 / 0.243 / 0.201 | 0.308 / 0.272 / 0.232 |
|  | Rewired model | 0.519 / 0.518 / 0.522 | 0.500 / 0.502 / 0.499 | 0.521 / 0.524 / 0.524 |
| Population distance | Original model | 2.109 / 2.279 / 2.416 | 2.018 / 2.111 / 2.277 | 1.944 / 1.896 / 2.223 |
|  | Rewired model | 0.228 / 0.225 / 0.222 | 0.247 / 0.242 / 0.242 | 0.221 / 0.214 / 0.219 |

Source: step15_model_analysis.json (sha256 5fd700b5a030cf5a). The values of part a equal those Fig. 4b was drawn from (fig4_values.csv, panel b). Assembled by make_supp_table_s12.py.
