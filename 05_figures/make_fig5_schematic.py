#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Purpose: draw the conceptual schematic for Results Section 5 of the Squidiff
#          manuscript ("The discrimination score measures alignment with the
#          plate rather than recognition of the compound") as a vector figure
#          (PDF) and a 600-dpi PNG. It replaces a BioRender draft whose panel b
#          labels were wrong and which could not be corrected (no AI credits).
# Logic:   three panels, read left to right, matching the four analyses of the
#          section.
#            a  Cross-plate comparison (the published score): five culture
#               plates, the scored condition on plate 3, and a ranking ladder in
#               which candidates from all five plates compete. Same-plate
#               candidates sit near the top because they share the plate offset.
#            b  The plate offset: the pooled cell-line vehicle mean at the
#               centre, five plate vehicle means displaced from it, the treated
#               cells of plate 3 keeping about one third of that offset, and
#               three predictions (cell-line vehicle average (line mean in the records), rewired model, original model)
#               drawn as arrows from the plate-3 mean back towards the centre;
#               the angle to the offset direction encodes the cosine with it
#               (arccos of the recorded value), so the cell-line vehicle average (line mean in the records) lies exactly on
#               the offset, the rewired model close to it, the original model
#               far from it.
#            c  Within-plate comparison: one plate only, same-colour candidates,
#               and the four within-plate scores on a 0.4-1.0 axis with chance
#               at 0.5 and the split-half ceiling at the top.
#          Every number printed is a range copied from the committed records
#          (Step 25-27; results_table_steps24_27.md, Tables 1-3):
#            cross-plate PDS (S-orig): rewired 0.71-0.72, original 0.64-0.65,
#              cell-line vehicle average (line mean in the records) 0.73-0.75
#            within-plate PDS (S-split): rewired 0.50-0.53, original 0.49-0.52,
#              cell-line vehicle average (line mean in the records) 0.50, split-half duplicate 0.88-0.91
#            cosine with plate offset (S-split): real treated cells 0.32-0.36,
#              rewired 0.62-0.64, original 0.25-0.26, cell-line vehicle average (line mean in the records) 1 by
#              construction (0.98 under S-split)
#            candidate plate structure: median 5 plates; 21-22 % same-plate pairs
#          Nothing is computed from data here; the script only draws. Dot
#          positions inside plates and ladders are decorative (fixed seed).
# Output:  figures/fig5_schematic.pdf and figures/fig5_schematic.png next to the
#          script (folder created if absent). Width 180 mm (Nature double
#          column), height 88 mm, sans-serif 5.5-8 pt, editable text in the PDF.
# -----------------------------------------------------------------------------
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK, MUTED = "#222222", "#6b6b6b"
PLATE = ["#7f9fd0", "#e3a053", "#8e62b4", "#5aae9b", "#d4879a"]       # plates 1-5
PLATE_FILL = ["#e4ecf7", "#fbeedb", "#ebe1f3", "#dcf0ea", "#f8e3e8"]
SCORED = 2                                                            # plate 3
ORANGE, BLUE, PURPLE, GREY = "#d9711c", "#2f6db5", "#7a3e9d", "#8a8a8a"
rng = np.random.RandomState(0)

# Each panel is an equal-aspect axes 10 units wide and 13 units tall.
W, H = 10.0, 13.0
fig = plt.figure(figsize=(180 / 25.4, 88 / 25.4))
gs = fig.add_gridspec(1, 3, wspace=0.04, left=0.005, right=0.995, top=0.995, bottom=0.075)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
for ax in axes:
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
axA, axB, axC = axes


def panel_label(ax, letter, title):
    ax.text(0.1, H - 0.1, letter, fontsize=9, fontweight="bold", va="top", color=INK)
    ax.text(0.9, H - 0.17, title, fontsize=7, fontweight="bold", va="top", color=INK)


def plate(ax, x, y, w, h, k, star=False):
    """Rounded plate with an 8 x 4 grid of wells; treated wells in the plate
    colour, vehicle wells grey. star marks the scored condition."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=PLATE_FILL[k], ec=PLATE[k], lw=0.8, zorder=1))
    xs = np.linspace(x + 0.13 * w / 2, x + w - 0.13 * w / 2, 8)
    ys = np.linspace(y + 0.22 * h / 2, y + h - 0.22 * h / 2, 4)
    r = 0.36 * (xs[1] - xs[0])
    for i, xi in enumerate(xs):
        for j, yj in enumerate(ys):
            treated = (i + 2 * j) % 3 != 0
            ax.add_patch(Circle((xi, yj), r, fc=PLATE[k] if treated else "#c4c4c4", ec="none", zorder=2))
    if star:
        ax.plot(xs[3], ys[1], marker="*", ms=7, mfc="#f2c200", mec=INK, mew=0.5, zorder=5)
    return xs[3], ys[1]


def arrow(ax, p, q, color, lw=1.1, ls="-", hl=0.28, hw=0.2, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={hl},head_width={hw}",
                                 color=color, lw=lw, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0, mutation_scale=10))


def ladder(ax, x, y0, y1, w=0.9):
    ax.add_patch(Rectangle((x - w / 2, y0), w, y1 - y0, fc="#f4f4f4", ec="#bdbdbd", lw=0.6, zorder=1))
    ax.text(x, y1 + 0.12, "rank", fontsize=6, ha="center", va="bottom", color=INK)
    ax.text(x + w / 2 + 0.1, y1 - 0.05, "most\nsimilar", fontsize=5, ha="left", va="top", color=MUTED)
    ax.text(x + w / 2 + 0.1, y0 + 0.05, "least\nsimilar", fontsize=5, ha="left", va="bottom", color=MUTED)


# ---- panel a -----------------------------------------------------------------
panel_label(axA, "a", "Published score")
axA.text(0.9, H - 0.75, "candidates from all plates", fontsize=6.5, va="top", color=MUTED)
pw, ph, gap = 4.2, 1.35, 0.42
for k in range(5):
    y = H - 1.9 - ph - k * (ph + gap)
    sx, sy = plate(axA, 0.7, y, pw, ph, k, star=(k == SCORED))
    axA.text(0.45, y + ph / 2, f"{k+1}", fontsize=6, ha="right", va="center", color=MUTED)
    if k == SCORED:
        star_xy = (0.7 + pw + 0.1, y + ph / 2)
axA.text(0.7 + pw / 2, H - 1.9 - 5 * ph - 4 * gap - 0.15, "five plates, one cell line, one dose",
         fontsize=5.5, ha="center", va="top", color=MUTED)
lx, ly0, ly1 = 7.3, 2.6, H - 2.4
ladder(axA, lx, ly0, ly1)
n = 55
same = rng.rand(n) < 0.21
pos = np.where(same, rng.uniform(0.72, 0.98, n), rng.uniform(0.02, 0.85, n))
plate_of = np.where(same, SCORED, rng.choice([0, 1, 3, 4], n))
for p_, k in zip(pos, plate_of):
    axA.plot(lx + rng.uniform(-0.28, 0.28), ly0 + p_ * (ly1 - ly0), "o", ms=3,
             mfc=PLATE[k], mec="white", mew=0.35, zorder=3)
arrow(axA, star_xy, (lx - 0.55, ly1 - 0.5), INK, lw=0.9)
axA.text(5.05, H - 6.05, "prediction\nfor the\nstarred\ncondition", fontsize=5.3, ha="left", va="top", color=INK)
axA.text(W / 2, 0.15, "~74 candidates on ~5 plates; only ~1 in 5\nshares the plate. Same-plate candidates\n"
         "rank high because they share the plate offset.", fontsize=5.8, ha="center", va="bottom", color=INK)

# ---- panel b -----------------------------------------------------------------
panel_label(axB, "b", "What the score rewards")
axB.text(0.9, H - 0.75, "the plate offset", fontsize=6.5, va="top", color=MUTED)
cx, cy = 4.0, 7.2
axB.add_patch(Circle((cx, cy), 0.5, fc="#d9d9d9", ec=INK, lw=0.6, zorder=3))
axB.text(cx + 0.45, cy - 0.75, "cell-line vehicle mean\n(all plates pooled)", fontsize=5.5, ha="left", va="top", color=INK)
angles = {0: 160, 1: 105, 2: 25, 3: 200, 4: 265}
R = 2.4
for k, a in angles.items():
    ar = np.radians(a)
    px, py = cx + R * np.cos(ar), cy + R * np.sin(ar)
    arrow(axB, (cx + 0.55 * np.cos(ar), cy + 0.55 * np.sin(ar)),
          (px - 0.35 * np.cos(ar), py - 0.35 * np.sin(ar)), MUTED, lw=0.7, hl=0.2, hw=0.15)
    axB.add_patch(Circle((px, py), 0.3, fc=PLATE[k], ec=INK, lw=0.5, zorder=3))
    off = {0: (-0.45, 0), 1: (0, 0.45), 3: (-0.45, -0.1), 4: (0, -0.45), 2: (0.0, 0.45)}[k]
    ha = {0: "right", 1: "center", 3: "right", 4: "center", 2: "center"}[k]
    va = {0: "center", 1: "bottom", 3: "center", 4: "top", 2: "bottom"}[k]
    axB.text(px + off[0], py + off[1], f"plate {k+1}", fontsize=5.5, ha=ha, va=va, color=INK)
axB.text(cx - 0.9, cy + 1.65, "plate offset", fontsize=5.5, ha="right", va="center", color=MUTED)
# plate 3 geometry
a3 = np.radians(angles[SCORED]); u = np.array([np.cos(a3), np.sin(a3)]); v = np.array([-u[1], u[0]])
p3 = np.array([cx, cy]) + R * u
t = p3 + 0.8 * u + 2.3 * v                      # treated cells: 1/3 of |offset| along u
for _ in range(16):
    axB.plot(t[0] + rng.normal(0, 0.2), t[1] + rng.normal(0, 0.2), "o", ms=2.5,
             mfc=ORANGE, mec="white", mew=0.3, zorder=3)
axB.text(t[0], t[1] + 0.55, "treated cells, plate 3", fontsize=5.5, ha="center", va="bottom", color=ORANGE)
d = (t - p3) / np.linalg.norm(t - p3)
arrow(axB, tuple(p3 + 0.32 * d), tuple(t - 0.45 * d), ORANGE, lw=1.2)
proj = p3 + 0.8 * u
axB.plot([proj[0], t[0]], [proj[1], t[1]], ls=":", lw=0.6, color=MUTED, zorder=2)
axB.plot([p3[0], proj[0]], [p3[1], proj[1]], ls="-", lw=2.2, color=ORANGE, alpha=0.45, zorder=2)
axB.text(proj[0] + 0.35, proj[1] - 0.45, "measured change keeps\n~1/3 of the offset\n(cosine 0.32–0.36)",
         fontsize=5.3, ha="left", va="top", color=ORANGE)
# predictions: arrows from plate 3 back towards the centre, angle = arccos(cosine)
back = -u
def pred(cos_val, color, length, lw):
    th = np.arccos(cos_val)
    dd = np.array([np.cos(th) * back[0] - np.sin(th) * back[1],
                   np.sin(th) * back[0] + np.cos(th) * back[1]])
    arrow(axB, tuple(p3 + 0.32 * dd), tuple(p3 + length * dd), color, lw=lw)
pred(1.00, BLUE, 1.85, 1.3)
pred(0.63, PURPLE, 1.75, 1.6)
pred(0.255, GREY, 1.6, 1.1)
# legend for the three predictions
ly = 3.25
for i, (col, lab) in enumerate([
        (BLUE, "cell-line vehicle average (no model): exactly along\nthe offset, cosine 1; cross-plate score 0.73–0.75"),
        (PURPLE, "rewired model: cosine 0.62–0.64, more aligned\nthan the real cells; score 0.71–0.72"),
        (GREY, "original model: cosine 0.25–0.26; score 0.64–0.65")]):
    axB.plot([0.5, 1.1], [ly - i * 0.72] * 2, color=col, lw=2.2, solid_capstyle="round")
    axB.text(1.3, ly - i * 0.72, lab, fontsize=5.3, ha="left", va="center", color=INK)
axB.text(W / 2, 0.15, "A prediction that points back along the plate offset is\nclosest to the same-plate "
         "candidates and wins the ranking\nwithout any information about the compound.",
         fontsize=5.8, ha="center", va="bottom", color=INK)

# ---- panel c -----------------------------------------------------------------
panel_label(axC, "c", "Same plate only")
axC.text(0.9, H - 0.75, "no plate to align with", fontsize=6.5, va="top", color=MUTED)
sx, sy = plate(axC, 0.7, H - 3.6, pw, ph, SCORED, star=True)
axC.text(0.7 + pw / 2, H - 3.75, "plate 3 only", fontsize=5.5, ha="center", va="top", color=MUTED)
lx, ly0, ly1 = 6.6, H - 4.3, H - 1.3
ladder(axC, lx, ly0, ly1)
for _ in range(13):
    axC.plot(lx + rng.uniform(-0.28, 0.28), ly0 + rng.uniform(0.05, 0.95) * (ly1 - ly0), "o",
             ms=3, mfc=PLATE[SCORED], mec="white", mew=0.35, zorder=3)
arrow(axC, (0.7 + pw + 0.1, H - 3.6 + ph / 2), (lx - 0.55, ly1 - 0.6), INK, lw=0.9)
axC.text(W / 2, H - 4.75, "every candidate shares the plate offset,\nso the offset cannot separate them",
         fontsize=5.5, ha="center", va="top", color=INK)
# score axis
x0, x1, ay = 1.0, 9.3, 1.9
def xp(val): return x0 + (val - 0.4) / 0.6 * (x1 - x0)
axC.plot([x0, x1], [ay, ay], color=INK, lw=0.6)
for val in np.arange(0.4, 1.01, 0.1):
    axC.plot([xp(val)] * 2, [ay - 0.1, ay], color=INK, lw=0.6)
    axC.text(xp(val), ay - 0.2, f"{val:.1f}", fontsize=5.3, ha="center", va="top", color=INK)
axC.text((x0 + x1) / 2, ay - 0.7, "within-plate discrimination score", fontsize=5.8, ha="center", va="top", color=INK)
top = H - 6.6
axC.plot([xp(0.5)] * 2, [ay, top + 0.3], color=MUTED, lw=0.6, ls="--")
axC.text(xp(0.5), top + 0.4, "chance", fontsize=5.3, ha="center", va="bottom", color=MUTED)
rows = [("split-half of the\nreal cells (ceiling)", 0.88, 0.91, ORANGE),
        ("cell-line vehicle average", 0.50, 0.50, BLUE),
        ("rewired model", 0.50, 0.53, PURPLE),
        ("original model", 0.49, 0.52, GREY)]
for i, (lab, lo, hi, col) in enumerate(rows):
    y = top - i * 0.95
    axC.plot([xp(lo), xp(hi)], [y, y], color=col, lw=3.2, solid_capstyle="round", zorder=3)
    if lo == hi:
        axC.plot(xp(lo), y, "o", ms=4, mfc=col, mec=col, zorder=4)
    txt = f"{lo:.2f}" if lo == hi else f"{lo:.2f}–{hi:.2f}"
    if lo > 0.6:
        axC.text(xp(lo) - 0.2, y, lab, fontsize=5.3, ha="right", va="center", color=INK)
        axC.text(xp(hi) + 0.15, y, txt, fontsize=5.3, ha="left", va="center", color=MUTED)
    else:
        axC.text(xp(hi) + 0.2, y, f"{lab}  {txt}", fontsize=5.3, ha="left", va="center", color=INK)

fig.text(0.5, 0.008, "The score rewards alignment with the plate offset. Within a plate both models are at chance, "
         "although the data separate the conditions (0.88–0.91).",
         fontsize=6.3, ha="center", va="bottom", color=INK, style="italic")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig5_schematic.pdf"))
fig.savefig(os.path.join(out, "fig5_schematic.png"), dpi=600)
print("written", out)
