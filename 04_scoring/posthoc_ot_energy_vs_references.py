"""posthoc_ot_energy_vs_references.py - post hoc intervals for two energy contrasts of the OT-paired model.

Purpose: Section 4 states that, with optimal-transport pairing, the generated populations were closer to the
         measured treated cells than the unaltered vehicle cells (and not closer than the compound-blind
         average on two splits). Neither contrast had a pre-registered interval (step 18 record: "descriptive").
         This script computes those intervals after the fact, with the same bootstrap as every other contrast
         in the study, so the sentence can carry an interval and be labelled a post hoc comparison.
Logic:   reads tahoe_results/step18_per_condition.csv (one row per held-out condition and split; metrics are
         already averaged over the three sampling seeds). For each split and each contrast
             ot_pair_energy - control_paste_energy     (below zero = closer than the unaltered vehicle cells)
             ot_pair_energy - mean_response_energy     (below zero = closer than the compound-blind average)
         the point estimate is the mean over conditions of the per-condition difference, as in the step 18
         primary. The interval is a 97.5 % percentile bootstrap interval: 5,000 replicates, RandomState(0),
         each replicate resampling the split's held-out structures with replacement and taking every
         condition of each drawn structure (structure-clustered), as in Methods. A contrast is 'below zero'
         when the whole interval is below zero, 'above zero' when above, else 'includes zero'.
         Nothing is typed in; the per-split means are printed so they can be checked against the step 18
         record (ot_pair 0.1479 / 0.1755 / 0.1501; control-paste 0.1820 / 0.2291 / 0.1788;
         mean-response 0.1458 / 0.1970 / 0.1457 for splits 1 / 0 / 2).
Output:  tahoe_results/posthoc_ot_energy_vs_references.json and .csv, and a printed table.
"""
import json
import numpy as np
import pandas as pd

CSV        = "tahoe_results/step18_per_condition.csv"
OUT        = "tahoe_results/posthoc_ot_energy_vs_references"
SPLITS     = ["drug_split_1", "drug_split_0", "drug_split_2"]        # development, confirmation, confirmation
CONTRASTS  = [("control_paste", "unaltered vehicle cells"),
              ("mean_response", "compound-blind average")]
N_BOOT     = 5000
LEVEL      = 0.975                                                     # 97.5 % percentile interval
SEED       = 0

d = pd.read_csv(CSV)
for col in ["split", "structure", "ot_pair_energy", "control_paste_energy", "mean_response_energy"]:
    assert col in d.columns, f"missing column {col}"


def cluster_bootstrap_mean(diff, groups, n_boot, seed):
    """Mean of `diff` under resampling of whole groups (structures) with replacement.

    diff   : per-condition differences (1-D array)
    groups : the structure of each condition (same length)
    Returns the array of replicate means.
    """
    codes, uniques = pd.factorize(groups)
    n_groups = len(uniques)
    # sum and count of the differences per structure: a replicate is a weighted mean over drawn structures
    sums   = np.bincount(codes, weights=diff, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    rng = np.random.RandomState(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.randint(0, n_groups, size=n_groups)              # structures with replacement
        out[b] = sums[draw].sum() / counts[draw].sum()               # mean over all conditions of drawn structures
    return out


rows = []
for split in SPLITS:
    s = d[d["split"] == split]
    assert len(s) > 0, split
    means = {k: float(s[f"{k}_energy"].mean()) for k in ["ot_pair", "control_paste", "mean_response"]}
    print(f"{split}: conditions {len(s)}, structures {s['structure'].nunique()}; "
          f"mean energy  ot_pair {means['ot_pair']:.4f}  control_paste {means['control_paste']:.4f}  "
          f"mean_response {means['mean_response']:.4f}")
    for ref, ref_label in CONTRASTS:
        diff = (s["ot_pair_energy"] - s[f"{ref}_energy"]).to_numpy()
        assert not np.isnan(diff).any(), (split, ref, "NaN in energy")
        point = float(diff.mean())
        boot = cluster_bootstrap_mean(diff, s["structure"].to_numpy(), N_BOOT, SEED)
        lo, hi = np.percentile(boot, [100 * (1 - LEVEL) / 2, 100 * (1 - (1 - LEVEL) / 2)])
        verdict = "below zero" if hi < 0 else ("above zero" if lo > 0 else "includes zero")
        rows.append(dict(split=split, contrast=f"ot_pair - {ref}", reference=ref_label,
                         n_conditions=int(len(s)), n_structures=int(s["structure"].nunique()),
                         point=point, lo=float(lo), hi=float(hi), verdict=verdict,
                         frac_conditions_closer=float((diff < 0).mean())))

res = pd.DataFrame(rows)
res.to_csv(OUT + ".csv", index=False)
json.dump(dict(source=CSV, n_boot=N_BOOT, level=LEVEL, seed=SEED,
               resampling="held-out structures with replacement (cluster bootstrap)",
               note="post hoc: computed after the step 18 run; no pre-registered interval for these contrasts",
               results=rows), open(OUT + ".json", "w"), indent=2)

print()
print("energy contrast (below zero = OT-paired model closer to the measured treated cells than the reference)")
for _, r in res.iterrows():
    print(f"  {r['split']:13s} {r['contrast']:26s} {r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']:13s}"
          f"  closer in {100 * r['frac_conditions_closer']:.1f} % of conditions")
print("written:", OUT + ".json", OUT + ".csv")
