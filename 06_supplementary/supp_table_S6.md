Supplementary Table S6 | Response correlation on sci-Plex (Results 2). Pearson correlation between the predicted and the measured mean change over the 50 genes with the largest absolute measured change, computed for each held-out condition (443, 444 and 455 conditions on splits 1, 0 and 2) and summarised as the mean and as the median over conditions (splits 1 / 0 / 2); the generative models are averaged over five seeds within a condition first. On the mean over conditions, every generative model lies below the two ridge regressions and the compound-blind average on every split; this is the statement made in the text. On the median over conditions the same holds on splits 1 and 0, but on split 2 the model with cross-attention and its parameter-matched control lie above the three references. The variants in the last four rows were trained on the development split only (Supplementary Table S10), which a dash marks. The unaltered vehicle cells have no predicted change and are not scored on this measure.

| Entry | Mean over conditions | Median over conditions |
|---|---|---|
| Training-set ridge | 0.196 / 0.123 / 0.121 | 0.180 / 0.106 / 0.095 |
| Cross-validated ridge | 0.189 / 0.202 / 0.098 | 0.188 / 0.247 / 0.082 |
| Compound-blind average | 0.180 / 0.250 / 0.131 | 0.126 / 0.304 / 0.086 |
| Original model | 0.045 / 0.010 / 0.051 | 0.038 / 0.018 / 0.026 |
| Original model with adaptive normalisation | 0.087 / 0.060 / 0.071 | 0.064 / 0.092 / 0.074 |
| Rewired model | 0.099 / 0.070 / 0.069 | 0.094 / 0.100 / 0.054 |
| Rewired model, 20 integration steps | 0.098 / 0.067 / 0.066 | 0.092 / 0.089 / 0.063 |
| Rewired model with classifier-free guidance (weight 2) | 0.118 / 0.079 / 0.078 | 0.093 / 0.093 / 0.065 |
| Rewired model with cross-attention | 0.118 / 0.079 / 0.086 | 0.121 / 0.089 / 0.098 |
| Rewired model, parameter-matched control without attention | 0.107 / 0.068 / 0.085 | 0.105 / 0.080 / 0.097 |
| Variant: mechanism-of-action labels in place of the fingerprint | 0.114 / — / — | 0.108 / — / — |
| Variant: mechanism-of-action labels added to the fingerprint | 0.106 / — / — | 0.106 / — / — |
| Variant: learned chemical embedding in place of the fingerprint | 0.055 / — / — | 0.045 / — / — |
| Variant: learned chemical embedding added to the fingerprint | 0.082 / — / — | 0.081 / — / — |

Source: step14_analysis.json (sha256 c7f021b27272bbac). Assembled by make_supp_table_s6.py.
