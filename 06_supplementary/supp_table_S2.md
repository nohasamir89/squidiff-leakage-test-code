Supplementary Table S2 | Training objective on training and on held-out cells (Results 1). The training objective of each model was evaluated on 2,560 training cells and on 2,560 cells of held-out compounds (40 batches of 64 cells, seed 0, uniform sampling of the diffusion time), for five seeds on each of the two sci-Plex confirmation splits. Losses are the mean over the five seeds, with the sample standard deviation over seeds in parentheses. The ratio is the mean over seeds of held-out loss divided by training loss, with its range over seeds in parentheses. The diffusion and the flow-matching objectives are different quantities, so losses are comparable within a row only; the ratio is the quantity compared in the text. The analysis was exploratory.

| Model | Objective | Split | Training loss | Held-out loss | Held-out ÷ training |
|---|---|---|---|---|---|
| Original model | Diffusion | Split 0 | 0.01223 (0.00008) | 0.01110 (0.00009) | 0.908 (0.903–0.914) |
| Original model | Diffusion | Split 2 | 0.01203 (0.00017) | 0.01232 (0.00011) | 1.024 (1.017–1.031) |
| Original model with adaptive normalisation | Diffusion | Split 0 | 0.01057 (0.00024) | 0.00973 (0.00020) | 0.921 (0.915–0.926) |
| Original model with adaptive normalisation | Diffusion | Split 2 | 0.01056 (0.00003) | 0.01077 (0.00010) | 1.020 (1.007–1.032) |
| Rewired model (adaptive normalisation and flow matching) | Flow matching | Split 0 | 0.04541 (0.00008) | 0.04828 (0.00010) | 1.063 (1.062–1.064) |
| Rewired model (adaptive normalisation and flow matching) | Flow matching | Split 2 | 0.04788 (0.00012) | 0.05314 (0.00021) | 1.110 (1.105–1.116) |

Sources: test_loss_base.json (sha256 2fa258669da2c667; compute_test_loss_base.py) and test_loss_v2.json (sha256 d5d9c48e469034d9; compute_test_loss.py). Before the original model was scored, the procedure reproduced the recorded value of drug_split_0_ddim_seed0 in test_loss_v2.json to within 1e-9. Assembled by make_supp_table_s2.py.
