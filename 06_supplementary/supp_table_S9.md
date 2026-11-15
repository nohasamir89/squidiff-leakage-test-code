Supplementary Table S9 | Settings of the experimental references, sampling steps and entries set to zero (Methods; Results 2 and 4). a, Settings of the references. b, Number of sampling steps of every evaluated configuration on splits 1 / 0 / 2 (a dash: not evaluated on that split). The diffusion models were sampled with DDIM and the flow-matching models with the Euler method. Source of the step count: record, the evaluation record of each run; run name, the number carried by the name of the run; sampler default, the default of the sampler, with which the original model was sampled. c, Entries of the shifted cells set to zero after the shift was added, in per cent of all entries, on splits 1 / 0 / 2.

**a** | Settings of the references.

| Reference | Setting |
|---|---|
| Unaltered vehicle cells | The matched vehicle cells, returned unchanged. |
| Training-set ridge | Multi-output ridge regression (scikit-learn Ridge, penalty 1.0 fixed before any result, intercept, unscaled features). Features: the 1,024-bit fingerprint, a one-hot cell-line indicator and their products (4,099 features on sci-Plex, 15,374 on Tahoe-100M). Target: the mean expression change of a condition from its matched vehicle cells. Fitted on the training conditions. |
| Cross-validated ridge | The same regression, fitted within the held-out compounds by five-fold cross-validation with folds grouped by structure. |
| Compound-blind average | One shift per cell line: the mean change over the held-out conditions of that line. |
| Use of the three shifts | The shift is added to the matched vehicle cells and negative values are set to zero; one shifted cell per measured cell, at most 200 per condition. |

**b** | Sampling steps of every configuration (splits 1 / 0 / 2).

| Dataset | Configuration | Steps | Sampler | Source of the step count |
|---|---|---|---|---|
| sci-Plex | Original model | 1,000 / 1,000 / 1,000 | DDIM | sampler default |
|  | Original model with adaptive normalisation | 1,000 / 1,000 / 1,000 | DDIM | run name / record / record |
|  | Rewired model | 50 / 50 / 50 | Euler | run name / record / record |
|  | Rewired model, 20 integration steps | 20 / 20 / 20 | Euler | run name / record / record |
|  | Rewired model with classifier-free guidance (weight 2) | 50 / 50 / 50 | Euler | record |
|  | Rewired model with cross-attention | 50 / 50 / 50 | Euler | record |
|  | Rewired model, parameter-matched control without attention | 50 / 50 / 50 | Euler | record |
|  | Variant: mechanism-of-action labels in place of the fingerprint | 20 / — / — | Euler | record |
|  | Variant: mechanism-of-action labels added to the fingerprint | 20 / — / — | Euler | record |
|  | Variant: learned chemical embedding in place of the fingerprint | 20 / — / — | Euler | record |
|  | Variant: learned chemical embedding added to the fingerprint | 20 / — / — | Euler | record |
| Tahoe-100M | Original model | 1,000 / 1,000 / 1,000 | DDIM | sampler default |
|  | Rewired model | 50 / 50 / 50 | Euler | record |
|  | Rewired model with classifier-free guidance (weight 2) | 50 / 50 / 50 | Euler | record |
|  | Rewired model with cross-attention | 50 / 50 / 50 | Euler | record |
|  | Rewired model, parameter-matched control without attention | 50 / 50 / 50 | Euler | record |

**c** | Entries set to zero (%; splits 1 / 0 / 2).

| Reference | sci-Plex | Tahoe-100M |
|---|---|---|
| Training-set ridge | 43.7 / 39.7 / 42.0 | 41.6 / 39.5 / 38.2 |
| Cross-validated ridge | 29.7 / 35.2 / 31.7 | 36.3 / 33.0 / 31.3 |
| Compound-blind average | 11.6 / 14.7 / 12.8 | 24.3 / 20.4 / 21.3 |

Sources: part a, Methods, with the number of features derived and the cap read from step14_analysis.json and step15_model_analysis.json; part b, the evaluation records in provenance/ of the 170 runs listed in those two files, and sample_squidiff.py for the sampler default (1,000 steps); part c, step14_analysis.log and s15score_final.log. Assembled by make_supp_table_s9.py.
