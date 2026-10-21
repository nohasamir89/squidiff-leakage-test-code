#!/usr/bin/env python3
"""
Step 20 figure: how strongly should training pairs be matched? The trade-off curve between the recipe (random pairs)
and Step 18 (exact OT pairs), with the three minibatch settings in between.

x-axis: how far each pairing is from random toward full OT (the cosine fraction f of the pairing record; recipe 0,
ot_pair 1). Four panels, each the change against the recipe with its 97.5 % structure-clustered interval:
  A  energy (lower = generated cells closer to the real ones; the gain)
  B  DE-overlap (lower = drug-specific signal lost; a cost)
  C  PDS (lower = compounds harder to tell apart; a cost)
  D  control tracking (how much each generated cell follows its conditioning control cell; the mechanism)
The dashed line joins the recipe (0, 0) to ot_pair (1, its change): the straight-line reference of prereg section 4.4.
A setting above that line on A (a smaller gain) or below it on B/C (a larger cost) does worse than proportional; the
reverse is what a separable trade-off would look like. Filled marker: interval excludes zero; hollow: includes zero.

Development split (drug_split_1) in blue. If the final record exists, the confirmed setting's values on splits 0 and 2
are added in their split colours at the same f.

Reads ONLY committed records: tahoe_results/step20_analysis_development.json and step20_long_development.csv (and the
final pair if present). Prints every number it plots and checks the long CSV against the analysis JSON to 1e-12, so a
wrong row cannot pass silently. Writes figures/step20_pairing_sweep.png.

Run on Jubail in the squidiff env:  python -u plot_step20.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEV_JSON = "tahoe_results/step20_analysis_development.json"
DEV_CSV = "tahoe_results/step20_long_development.csv"
FIN_JSON = "tahoe_results/step20_analysis.json"
FIN_CSV = "tahoe_results/step20_long.csv"
OUT = "figures/step20_pairing_sweep.png"
DEV = "drug_split_1"

# palette (Noha's standing choices)
SPLIT_COLOR = {"drug_split_1": "#2a78d6", "drug_split_0": "#eb6834", "drug_split_2": "#1baf7a"}
SPLIT_MARK = {"drug_split_1": "o", "drug_split_0": "s", "drug_split_2": "^"}
SPLIT_LABEL = {"drug_split_1": "split 1 (development)", "drug_split_0": "split 0 (confirmation)", "drug_split_2": "split 2 (confirmation)"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

PANELS = [("Q1_energy_minus_recipe", "A   Realism (energy)", "energy change vs recipe\n(below 0 = closer to the real cells)"),
          ("Q2_de_minus_recipe", "B   DE-overlap", "DE-overlap change vs recipe\n(below 0 = drug signal lost)"),
          ("Q3_pds_minus_recipe", "C   PDS", "PDS change vs recipe\n(below 0 = compounds harder to tell apart)"),
          ("secondary_tracking_minus_recipe", "D   Tracking (mechanism)", "tracking change vs recipe\n(above 0 = follows its control cell)")]


def load(json_path, csv_path):
    rec = json.load(open(json_path))
    long_ = pd.read_csv(csv_path, float_precision="round_trip")
    return rec, long_


def row(long_, split, arm, key):
    r = long_[(long_["split"] == split) & (long_["arm"] == arm) & (long_["level"] == "contrast") & (long_["key"] == key)]
    assert len(r) == 1, (split, arm, key, len(r))
    r = r.iloc[0]
    return float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"]), str(r["verdict"])


def check(rec, split, arm, key, got):
    """The CSV row must equal the analysis JSON to 1e-12 (difference and interval)."""
    c = rec["per_split"][split]["contrasts"][arm][key]
    want = (float(c["diff"]), float(c["ci"][0]), float(c["ci"][1]))
    assert max(abs(g - w) for g, w in zip(got[:3], want)) < 1e-12 and got[3] == c["verdict"], (split, arm, key, got, want)


def dot(ax, x, v, lo, hi, verdict, split):
    c = SPLIT_COLOR[split]
    ax.plot([x, x], [lo, hi], color=c, lw=1.5, solid_capstyle="butt", zorder=2)
    ax.plot([x], [v], marker=SPLIT_MARK[split], ms=7.5, mfc=(c if verdict != "includes zero" else SURFACE), mec=c, mew=1.6,
            ls="none", zorder=3)


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


dev, dlong = load(DEV_JSON, DEV_CSV)
assert dev["final"] is False and dev["selection"]["split"] == DEV, "not the development record"
arms = dev["per_split"][DEV]["arms"]
f = {nm: float(dev["per_split"][DEV]["f_cos"][nm]) for nm in arms}
f["ot_pair"] = 1.0
order = sorted(arms, key=lambda nm: f[nm]) + ["ot_pair"]
fin = None
if os.path.exists(FIN_JSON) and os.path.exists(FIN_CSV):
    fin, flong = load(FIN_JSON, FIN_CSV)
    assert fin["final"] is True
print("development verdict: %s" % dev["verdict"])
print("pairing fractions f: %s" % {nm: round(f[nm], 3) for nm in order})

fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.6), facecolor=SURFACE)
for ax, (key, title, ylab) in zip(axes, PANELS):
    print("\n%s [%s]" % (title.split("   ")[1], key))
    print("  %-8s f=0.000  0 by definition (the reference)" % "recipe")
    vals = {}
    for nm in order:
        got = row(dlong, DEV, nm, key)
        check(dev, DEV, nm, key, got)
        vals[nm] = got
        dot(ax, f[nm], *got, DEV)
        print("  %-8s f=%.3f  %+.5f [%+.5f, %+.5f] %s" % (nm, f[nm], *got))
    ax.plot([0.0], [0.0], marker="D", ms=6.5, color=INK2, ls="none", zorder=3)               # the recipe: the reference
    ax.plot([0.0, 1.0], [0.0, vals["ot_pair"][0]], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)  # straight-line reference
    if fin is not None:
        m_sel = "mb%d" % int(fin["selection"]["confirmed_m"])
        for sp in ("drug_split_0", "drug_split_2"):
            got = row(flong, sp, m_sel, key)
            check(fin, sp, m_sel, key, got)
            dot(ax, f[m_sel] + (0.018 if sp == "drug_split_0" else 0.036), *got, sp)
            print("  %-8s %s  %+.5f [%+.5f, %+.5f] %s" % (m_sel, sp, *got))
    ax.set_xlim(-0.06, 1.08)
    ax.set_xticks([0.0] + [f[nm] for nm in order])
    ax.set_xticklabels(["recipe"] + [("m=%s\n%.3f" % (nm[2:], f[nm])) if nm != "ot_pair" else "full OT\n1.000" for nm in order],
                       fontsize=8.2, color=INK)
    style(ax, ylab, title)
fig.text(0.5, 0.005, "how far the training pairs are from random toward exact optimal transport (cosine fraction f)",
         ha="center", fontsize=9.5, color=INK)
sub = "development verdict: %s" % dev["verdict"]
if fin is not None:
    sub += "   |   final verdict for the confirmed setting: %s" % fin["verdict"]
fig.suptitle("Step 20   Matching strength of the training pairs: what is gained and what is lost      (%s)" % sub,
             fontsize=11, color=INK, x=0.01, ha="left", y=0.995)
handles = [Line2D([], [], marker="D", color=INK2, ls="none", ms=6, label="recipe (reference, 0)"),
           Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="straight line recipe → full OT")]
handles += [Line2D([], [], marker=SPLIT_MARK[sp], color=SPLIT_COLOR[sp], ms=7, lw=1.5, label=SPLIT_LABEL[sp])
            for sp in (["drug_split_1"] + (["drug_split_0", "drug_split_2"] if fin is not None else []))]
handles += [Line2D([], [], marker="o", color=MUTED, mfc=MUTED, ms=7, ls="none", label="filled: interval excludes 0"),
            Line2D([], [], marker="o", color=MUTED, mfc=SURFACE, mew=1.6, ms=7, ls="none", label="hollow: includes 0")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(handles), frameon=False, fontsize=8.4)
fig.tight_layout(rect=(0, 0.03, 1, 0.91))
os.makedirs("figures", exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print("\nwrote", OUT)
