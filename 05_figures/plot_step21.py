#!/usr/bin/env python3
"""
Step 21 figure: does m = 32 keep full OT's realism while losing less compound discrimination? Each interval is shown
against its margin, as the CONSORT extension for non-inferiority trials suggests (Piaggio et al. 2012, item 17a).

Three panels, each the difference m = 32 minus full OT with its 97.5 % structure-clustered interval, per split:
  A  Q1, realism kept (energy; lower = closer to the real cells). The short dashed line at each split is that split's
     non-inferiority margin delta = 0.25 x M1 (prereg 4.1). Q1 holds where the whole interval lies below the margin.
  B  Q2, less PDS loss (PDS; above 0 = m = 32 tells compounds apart better than full OT). Q2 holds where the whole
     interval lies above zero.
  C  Secondary: DE-overlap (above 0 = m = 32 keeps more of the responding genes than full OT).
Splits 0 and 2 (the confirmation splits, gated) in their colours; split 1 (the development split, Step 20) shown
descriptively, set apart and labelled as such. Filled marker: the test holds (A, B) or the interval excludes zero (C);
hollow: it does not.

Reads ONLY the committed records tahoe_results/step21_analysis.json and step21_long.csv. Prints every number it plots
and checks every plotted CSV row against the analysis JSON to 1e-12, so a wrong row cannot pass silently.
Writes figures/step21_mb32_confirmation.png.

Run on Jubail in the squidiff env:  python -u plot_step21.py
"""
import json
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

JSON_PATH = "tahoe_results/step21_analysis.json"
CSV_PATH = "tahoe_results/step21_long.csv"
OUT = "figures/step21_mb32_confirmation.png"
DEV, CONF = "drug_split_1", ("drug_split_0", "drug_split_2")
ORDER = (DEV,) + CONF                                  # left to right: development, then the two confirmation splits
XPOS = {DEV: 0.0, "drug_split_0": 1.3, "drug_split_2": 2.3}

# palette (Noha's standing choices)
SPLIT_COLOR = {"drug_split_1": "#2a78d6", "drug_split_0": "#eb6834", "drug_split_2": "#1baf7a"}
SPLIT_MARK = {"drug_split_1": "o", "drug_split_0": "s", "drug_split_2": "^"}
SPLIT_LABEL = {"drug_split_1": "split 1\n(development,\ndescriptive)", "drug_split_0": "split 0\n(confirmation)",
               "drug_split_2": "split 2\n(confirmation)"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

PANELS = [("Q1_energy_mb32_minus_ot_pair", "A   Q1  Realism kept?",
           "energy, m = 32 minus full OT\n(below the dashed margin = realism kept)"),
          ("Q2_pds_mb32_minus_ot_pair", "B   Q2  Less PDS loss?",
           "PDS, m = 32 minus full OT\n(above 0 = compounds told apart better)"),
          ("secondary_de_mb32_minus_ot_pair", "C   DE-overlap (secondary)",
           "DE-overlap, m = 32 minus full OT\n(above 0 = more responding genes kept)")]


def rec_of(rec, split, key):
    """The analysis JSON's record of one contrast, and whether its test holds (A: non-inferior; B: above zero;
    C: interval excludes zero)."""
    if split == DEV:
        d = rec["development_descriptive"]
        c = d["contrasts"][key]
        res = d["Q1_result"] if key.startswith("Q1") else c["verdict"]
    else:
        c = rec["per_split"][split]["contrasts"][key]
        res = rec["per_split"][split]["tests"]["Q1"]["result"] if key.startswith("Q1") else c["verdict"]
    if key.startswith("Q1"):
        holds = res == "non-inferior"
    elif key.startswith("Q2"):
        holds = res == "above zero"
    else:
        holds = res != "includes zero"
    return c, res, holds


def csv_row(long_, split, key):
    level = "development_descriptive" if split == DEV else "contrast"
    r = long_[(long_["split"] == split) & (long_["arm"] == "mb32") & (long_["level"] == level) & (long_["key"] == key)]
    assert len(r) == 1, (split, key, level, len(r))
    r = r.iloc[0]
    return float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"]), str(r["verdict"])


def delta_of(rec, long_, split):
    want = rec["development_descriptive"]["delta"] if split == DEV else rec["per_split"][split]["delta"]
    r = long_[(long_["split"] == split) & (long_["level"] == "margin") & (long_["key"] == "delta")]
    assert len(r) == 1 and abs(float(r.iloc[0]["value"]) - float(want)) < 1e-12, (split, "delta")
    return float(want)


def style(ax, ylab, title):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylab, fontsize=9, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, pad=8, loc="left")
    ax.tick_params(colors=AXIS, labelcolor=INK2, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color=INK2, lw=1.0, ls="--", zorder=1)
    ax.axvline(0.65, color=AXIS, lw=0.8, ls=":", zorder=1)          # sets the development split apart
    ax.set_xlim(-0.6, 2.9)
    ax.set_xticks([XPOS[s] for s in ORDER])
    ax.set_xticklabels([SPLIT_LABEL[s] for s in ORDER], fontsize=8.2, color=INK)


rec = json.load(open(JSON_PATH))
long_ = pd.read_csv(CSV_PATH, float_precision="round_trip")
print("verdict: %s" % rec["verdict"])
deltas = {s: delta_of(rec, long_, s) for s in ORDER}
print("margins delta: %s" % {s: round(v, 6) for s, v in deltas.items()})

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), facecolor=SURFACE)
for ax, (key, title, ylab) in zip(axes, PANELS):
    print("\n%s [%s]" % (title.split("   ")[1], key))
    for s in ORDER:
        c, res, holds = rec_of(rec, s, key)
        got = csv_row(long_, s, key)
        want = (float(c["diff"]), float(c["ci"][0]), float(c["ci"][1]))
        assert max(abs(g - w) for g, w in zip(got[:3], want)) < 1e-12 and got[3] == c["verdict"], (s, key, got, want)
        x, col = XPOS[s], SPLIT_COLOR[s]
        ax.plot([x, x], [got[1], got[2]], color=col, lw=1.6, solid_capstyle="butt", zorder=2,
                alpha=0.75 if s == DEV else 1.0)
        ax.plot([x], [got[0]], marker=SPLIT_MARK[s], ms=8, mfc=(col if holds else SURFACE), mec=col, mew=1.6, ls="none",
                zorder=3, alpha=0.75 if s == DEV else 1.0)
        if key.startswith("Q1"):
            ax.plot([x - 0.28, x + 0.28], [deltas[s], deltas[s]], color=INK, lw=1.3, ls=(0, (3, 2)), zorder=2)
        print("  %-13s %+.5f [%+.5f, %+.5f] %-13s -> %s%s" % (s, got[0], got[1], got[2], got[3], res,
              ("   (margin %.5f)" % deltas[s]) if key.startswith("Q1") else ""))
    style(ax, ylab, title)
fig.suptitle("Step 21   m = 32 against full OT on the confirmation splits      (pre-registered verdict: %s)" % rec["verdict"],
             fontsize=11, color=INK, x=0.01, ha="left", y=0.995)
handles = [Line2D([], [], marker=SPLIT_MARK[s], color=SPLIT_COLOR[s], ms=7, lw=1.5, label=SPLIT_LABEL[s].replace("\n", " "))
           for s in ORDER]
handles += [Line2D([], [], color=INK, lw=1.3, ls=(0, (3, 2)), label="margin delta (keep 75 % of full OT's cautious gain)"),
            Line2D([], [], marker="o", color=MUTED, mfc=MUTED, ms=7, ls="none", label="filled: test holds / excludes 0"),
            Line2D([], [], marker="o", color=MUTED, mfc=SURFACE, mew=1.6, ms=7, ls="none", label="hollow: does not")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=8.4)
fig.tight_layout(rect=(0, 0, 1, 0.86))
os.makedirs("figures", exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print("\nwrote", OUT)
