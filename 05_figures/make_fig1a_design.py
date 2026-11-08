#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Purpose: draw Fig. 1a (study design) of the Squidiff manuscript as a vector
#          figure (PDF) and a 600-dpi PNG, replacing the BioRender export, which
#          carried the typo "Glioblasome" and could not be re-exported.
# Logic:   three boxes, read left to right, in the same flat style and palette
#          as the Section 5 schematic (make_fig5_schematic.py).
#            Data            - the three datasets with the counts stated in the
#                              Methods (Datasets): Tahoe-100M, 14 cell lines
#                              (3 brain, 2 gastric, 9 colorectal), 372 compound
#                              structures, 3 doses; sci-Plex, 3 cell lines
#                              (A549, K562, MCF7), 188 compounds, 4 doses;
#                              glioblastoma slice cultures, used only for the
#                              gene-panel sweep and the reproduction.
#            Compound splits - three disjoint held-out sets of 75 structures
#                              (Tahoe; Methods, Compound splits), split 1 used
#                              for development and splits 0 and 2 for
#                              confirmation. Compounds are drawn as hexagon
#                              glyphs; no real structure is depicted.
#            Evaluation      - vehicle cell + held-out compound -> generative
#                              model -> generated cells, compared with the
#                              measured treated cells on gene identity,
#                              population distance and response correlation,
#                              beside two references taken from the experiment:
#                              the unaltered vehicle cells and the plate
#                              replicate.
#          Every count printed is copied from the Methods text of the
#          manuscript (rev 153); nothing is computed here.
# Output:  figures/fig1a_design.pdf and figures/fig1a_design.png next to the
#          script. Width 180 mm, height 95 mm, editable text in the PDF.
# -----------------------------------------------------------------------------
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, RegularPolygon, FancyArrowPatch

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED = "#222222", "#6b6b6b"
BOX = "#f4f4f4"; BOX_EC = "#9a9a9a"
SPLIT = {1: ("#7f9fd0", "#e4ecf7"), 0: ("#e3a053", "#fbeedb"), 2: ("#5aae9b", "#dcf0ea")}
CELL = ("#d4879a", "#b45a72")      # body, nucleus
CELL2 = ("#9b8fc4", "#6e5fa3")
MODEL = ("#c9cde8", "#5e6aa8")
rng = np.random.RandomState(1)

fig = plt.figure(figsize=(180 / 25.4, 95 / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 95); ax.set_aspect("equal"); ax.axis("off")


def rbox(x, y, w, h, fc=BOX, ec=BOX_EC, lw=0.8, r=2.0, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z))


def cell(x, y, r=2.2, col=CELL, z=3):
    ax.add_patch(Ellipse((x, y), 2 * r, 1.8 * r, fc=col[0], ec="none", zorder=z))
    ax.add_patch(Circle((x - 0.25 * r, y + 0.1 * r), 0.55 * r, fc=col[1], ec="none", zorder=z + 1))


def cells(x, y, n, col, spread=3.0, r=1.7, z=3):
    for _ in range(n):
        cell(x + rng.uniform(-spread, spread), y + rng.uniform(-spread * 0.7, spread * 0.7), r, col, z)


def compound(x, y, col, r=1.5, z=3):
    ax.add_patch(RegularPolygon((x, y), 6, radius=r, orientation=np.pi / 6, fc="white", ec=col, lw=0.9, zorder=z))
    ax.plot([x + r * 0.87, x + r * 1.6], [y + r * 0.5, y + r * 0.9], color=col, lw=0.9, zorder=z)


def arrow(p, q, color=INK, lw=1.0, hl=2.2, hw=1.6, z=4, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={hl},head_width={hw}", color=color,
                                 lw=lw, linestyle=ls, mutation_scale=1, shrinkA=0, shrinkB=0, zorder=z))


def title(x, y, s):
    ax.text(x, y, s, fontsize=8, fontweight="bold", ha="center", va="top", color=INK)


# ---- column 1: Data ---------------------------------------------------------
rbox(3, 4, 46, 88, fc="white")
title(26, 90, "Data")
for i, (name, lines, l2, l3) in enumerate([
        ("Tahoe-100M", "14 cell lines: 3 brain, 2 gastric,\n9 colorectal", "372 compound structures", "3 doses"),
        ("sci-Plex", "3 cell lines (A549, K562, MCF7)", "188 compounds", "4 doses"),
        ("Glioblastoma slice cultures", "gene-panel sweep and\nreproduction only", "", "")]):
    y0 = 60 - i * 27.5
    rbox(6, y0, 40, 25)
    ax.text(26, y0 + 23, name, fontsize=7, fontweight="bold", ha="center", va="top", color=INK)
    if i == 0:
        cells(17, y0 + 15.5, 4, CELL, spread=2.5, r=1.4); cells(35, y0 + 15.5, 4, CELL2, spread=2.5, r=1.4)
    elif i == 1:
        for k, c in enumerate([CELL2, CELL, ("#c58b8b", "#8f4f4f")]):
            cell(17 + 9 * k, y0 + 15.5, 2.0, c)
    else:
        # a brain-shaped blob for slice cultures: two overlapping ellipses with a tumour spot
        ax.add_patch(Ellipse((22, y0 + 15.5), 9, 6.5, fc="#e9c9d1", ec="#b45a72", lw=0.7, zorder=3))
        ax.add_patch(Ellipse((30, y0 + 15.5), 7, 5.5, fc="#e9c9d1", ec="#b45a72", lw=0.7, zorder=3))
        ax.add_patch(Circle((25, y0 + 15), 1.6, fc="#b45a72", ec="none", zorder=4))
    txt = "\n".join(t for t in (lines, l2, l3) if t)
    ax.text(26, y0 + 10.5, txt, fontsize=5.8, ha="center", va="top", color=INK, linespacing=1.25)
arrow((50, 48), (56, 48), lw=1.1)

# ---- column 2: Compound splits ----------------------------------------------
rbox(57, 4, 52, 88, fc="white")
title(83, 90, "Compound splits")
ax.text(83, 84.5, "three disjoint held-out sets of 75 structures (Tahoe)", fontsize=5.5, ha="center", va="top", color=MUTED)
ax.text(73.5, 80.5, "training compounds", fontsize=5.8, ha="center", va="top", color=INK)
ax.text(100, 80.5, "held-out\ncompounds", fontsize=5.8, ha="center", va="top", color=INK)
for i, s in enumerate([1, 0, 2]):
    col, fill = SPLIT[s]
    y0 = 55 - i * 24
    role = {1: "development", 0: "confirmation", 2: "confirmation"}[s]
    ax.text(60, y0 + 18.3, f"Split {s}  \u00b7  {role}", fontsize=6.3, ha="left", va="bottom", color=col, fontweight="bold")
    rbox(60, y0, 27, 18, fc=fill, ec=col, lw=0.9)
    for (dx, dy) in [(5, 13), (12, 14), (20, 12), (7, 5), (15, 6), (22, 4.5)]:
        compound(60 + dx, y0 + dy, col)
    arrow((88, y0 + 9), (93, y0 + 9), color=col, lw=0.9)
    rbox(94, y0 + 2.5, 12, 13, fc=fill, ec=col, lw=0.9)
    compound(98, y0 + 11.5, col); compound(102, y0 + 6, col)
arrow((110, 48), (116, 48), lw=1.1)

# ---- column 3: Evaluation ----------------------------------------------------
rbox(117, 4, 60, 88, fc="white")
title(147, 90, "Evaluation")
# inputs
cell(123, 68, 2.4, CELL)
ax.text(123, 72.5, "vehicle cell\n(DMSO)", fontsize=5.5, ha="center", va="bottom", color=INK)
compound(123, 52, "#555555", r=1.8)
ax.text(123, 48.5, "held-out\ncompound", fontsize=5.5, ha="center", va="top", color=INK)
# model box
rbox(131, 54, 16, 12, fc=MODEL[0], ec=MODEL[1], lw=0.9)
ax.text(139, 60, "generative\nmodel", fontsize=6, ha="center", va="center", color=INK)
arrow((126, 67), (131, 62.5), lw=0.9); arrow((126, 53), (131, 57.5), lw=0.9)
arrow((147, 60), (151.5, 60), lw=0.9)
# generated vs measured
cells(156, 60, 5, CELL, spread=2.6, r=1.5)
ax.text(156, 66.5, "generated\ncells", fontsize=5.5, ha="center", va="bottom", color=INK)
cells(171, 60, 5, ("#c58b8b", "#8f4f4f"), spread=2.6, r=1.5)
ax.text(171, 66.5, "measured\ntreated cells", fontsize=5.5, ha="center", va="bottom", color=INK)
ax.annotate("", (167.5, 60), (159.5, 60), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
ax.text(163.5, 56.5, "gene identity\npopulation distance\nresponse correlation", fontsize=5.3,
        ha="center", va="top", color=INK, linespacing=1.25)
# references
ax.text(147, 36, "references taken from the experiment", fontsize=5.8, ha="center", va="bottom", color=MUTED)
rbox(121, 10, 25, 24, fc="white", ec="#9a9a9a", ls="--", lw=0.8)
cell(133.5, 27, 2.4, CELL)
ax.text(133.5, 22.5, "unaltered vehicle cells\n(same line and plate)\nlower reference", fontsize=5.3, ha="center", va="top", color=INK, linespacing=1.25)
rbox(149, 10, 25, 24, fc="white", ec="#9a9a9a", ls="--", lw=0.8)
cells(161.5, 27, 3, ("#c58b8b", "#8f4f4f"), spread=1.8, r=1.4)
ax.text(161.5, 22.5, "plate replicate\n(same condition,\nsecond plate)\nempirical ceiling", fontsize=5.3, ha="center", va="top", color=INK, linespacing=1.25)
ax.text(147, 6.5, "each reference is scored on the same measures as the model", fontsize=5.3, ha="center", va="bottom", color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig1a_design.pdf"))
fig.savefig(os.path.join(out, "fig1a_design.png"), dpi=600)
print("written", out)
