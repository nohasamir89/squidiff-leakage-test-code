"""plot_fig1b.py - Fig. 1b: gene identity of the original model against the references.

Purpose: draw the gene-identity panel of Figure 1 from the committed result files only.
Logic:   Tahoe values = tahoe_results/step15_model_long.csv (line == 'ALL', metric == 'de_overlap');
         Tahoe plate replicate = tahoe_results/step15_model_analysis.json;
         sci-Plex values = leakage_results/step14_analysis.json.
         The chance level is computed from the hypergeometric distribution, not typed in.
         Writes figures/fig1b_gene_identity.{png,pdf} and figures/fig1b_values.csv, and prints the values.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

# ---- input files, relative to the repo root ----
TAHOE_LONG   = "tahoe_results/step15_model_long.csv"
TAHOE_JSON   = "tahoe_results/step15_model_analysis.json"
SCIPLEX_JSON = "leakage_results/step14_analysis.json"
OUT          = "figures/fig1b_gene_identity"
os.makedirs("figures", exist_ok=True)

# ---- labels in the result files -> labels in the manuscript (bottom-to-top on the y axis) ----
ROWS = [
    ("base",          "Original model"),
    ("ridge_train",   "Like-for-like ridge"),
    ("ridge_S7",      "Cross-validated ridge"),
    ("mean_response", "Compound-blind average"),
    ("replicate",     "Plate replicate"),
]
# split id in the files, label in the paper, colour in the paper
SPLITS = [("drug_split_1", "Split 1", "#2a78d6"),
          ("drug_split_0", "Split 0", "#eb6834"),
          ("drug_split_2", "Split 2", "#1baf7a")]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# ---- chance level: expected Jaccard of two random sets of 50 genes out of 200 ----
# overlap I ~ Hypergeom(N=200, K=50, n=50); Jaccard = I / (100 - I); take the expectation exactly
rv = hypergeom(200, 50, 50)
i = np.arange(0, 51)
CHANCE = float(np.sum(rv.pmf(i) * i / (100 - i)))

# ---- Tahoe: one pooled value per split and submission ----
d = pd.read_csv(TAHOE_LONG)
d = d[(d["line"] == "ALL") & (d["metric"] == "de_overlap")]
tahoe = {}
for split, _, _ in SPLITS:
    for key, _ in ROWS[:-1]:
        v = d[(d["split"] == split) & (d["submission"] == key)]["value"]
        assert len(v) == 1, ("expected one row", split, key, len(v))   # stop if the file is not as expected
        tahoe[(split, key)] = float(v.iloc[0])
a = json.load(open(TAHOE_JSON))
for split, _, _ in SPLITS:
    tahoe[(split, "replicate")] = float(a["per_split"][split]["replicate"]["de_overlap"])

# ---- sci-Plex: one value per split and submission; no plate replicate exists ----
b = json.load(open(SCIPLEX_JSON))
sciplex = {}
for split, _, _ in SPLITS:
    tm = b["per_split"][split]["thesis_metrics"]
    for key, _ in ROWS[:-1]:
        sciplex[(split, key)] = float(tm[key]["de_overlap"])

# ---- write the plotted values to a CSV (also feeds Supplementary Table S5) ----
rows = []
for dataset, vals in [("Tahoe-100M", tahoe), ("sci-Plex", sciplex)]:
    for (split, key), v in vals.items():
        rows.append(dict(dataset=dataset, split=split, submission=key,
                         label=dict(ROWS)[key], gene_identity=v))
rows.append(dict(dataset="both", split="all", submission="chance", label="Chance (50 of 200)", gene_identity=CHANCE))
pd.DataFrame(rows).to_csv(OUT.replace("gene_identity", "values") + ".csv", index=False)

# ---- draw: two panels, horizontal dot plot, one dot per split ----
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, facecolor=SURFACE)
panels = [("Tahoe-100M, 14 cell lines", tahoe, True), ("sci-Plex, 3 cell lines", sciplex, False)]
for ax, (title, vals, has_replicate) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    for y, (key, label) in enumerate(ROWS):
        if key == "replicate" and not has_replicate:
            ax.text(0.02, y, "no plate replicate", va="center", fontsize=7, color=MUTED, style="italic")
            continue
        for j, (split, _, colour) in enumerate(SPLITS):
            # small vertical offset per split so the three dots never hide each other
            ax.scatter(vals[(split, key)], y + (j - 1) * 0.2, s=28, color=colour,
                       zorder=3, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(CHANCE, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(CHANCE + 0.012, len(ROWS) - 0.6, f"chance {CHANCE:.2f}", fontsize=7, color=INK2, va="center")
    ax.set_xlim(0, 0.7); ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([lab for _, lab in ROWS], fontsize=8, color=INK)
    ax.set_ylim(-0.6, len(ROWS) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=INK2, labelsize=7, length=2)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=4)
    ax.set_xlabel("Gene identity (Jaccard, top-50 genes)", fontsize=8, color=INK2)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n) for _, n, c in SPLITS]
axes[1].legend(handles=handles, loc="lower right", fontsize=7, frameon=False, handletextpad=0.3)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT + ".png", dpi=300, facecolor=SURFACE)
fig.savefig(OUT + ".pdf", facecolor=SURFACE)

# ---- print the values for checking against the text ----
print(f"chance (exact hypergeometric expectation): {CHANCE:.4f}")
for dataset, vals in [("Tahoe-100M", tahoe), ("sci-Plex", sciplex)]:
    print(dataset)
    for key, label in ROWS:
        got = [vals.get((s, key)) for s, _, _ in SPLITS]
        if all(g is None for g in got):
            continue
        print(f"  {label:24s} " + "  ".join(f"{n}: {g:.3f}" for (_, n, _), g in zip(SPLITS, got)))
print("written:", OUT + ".png", OUT + ".pdf", OUT.replace("gene_identity", "values") + ".csv")
