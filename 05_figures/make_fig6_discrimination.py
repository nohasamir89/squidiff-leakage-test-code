#!/usr/bin/env python3
"""
make_fig6_discrimination.py - Figure 6 of the manuscript (panels a-c; formerly Fig. 5a): the discrimination score under cross-plate and within-plate
candidate sets, for every arm, with the no-model references and the ceiling.

Purpose
-------
The old Figure 5a showed the cross-plate score read as compound identification. The new panel shows, side by side,
(a) the published cross-plate score for every arm, including predictions made without a model, (b) the same arms
with candidates restricted to the condition's own plate, and (c) how aligned each arm's predicted change is with
the plate's vehicle offset. Every number is read from a committed record; nothing is typed from memory except the
Step 15c published intervals and the Step 27 values, which are copied from the committed scoring logs and listed
below with their source line.

Logic
-----
- Panel a (cross-plate, S-orig): points = PDS per arm and split. Intervals, where a pre-registered contrast against
  chance exists: recipe and base from the Step 15c published intervals (replayed in the Step 26/27 logs); fpshuf and
  fpzero from the Step 25 contrasts (Q2 and ref_zero); ctrl_line and ctrl_sample from the Step 26 contrasts G4 and
  G4b. The duplicate ceiling has no test (diamond).
- Panel b (within-plate, S-split): points and intervals from the Step 27 contrasts (H1, base_H1, fpzero, ctrl_sample,
  duplicate against chance; ctrl_line under S-orig, H4, which is chance by construction); recipe_fpshuf and
  base_fpshuf have within-plate points but no pre-registered interval against chance (shown as points only).
- Panel c (alignment with the plate offset, S-split): per-arm mean cosine between the predicted change and the plate's
  vehicle offset; the measured change (true) with its interval (H5) and the h2 resample null (H5null).
Filled marker = interval excludes chance (or zero); hollow = includes; diamond = no test; small dot = no interval
computed. Colours by split as in the other figures.

Inputs (relative to the research worktree root, or pass --records DIR):
  tahoe_results/step25_swap_summary.csv, tahoe_results/step26_source_summary.csv,
  tahoe_results/step27_within_plate_summary.csv (if present; otherwise the values transcribed from the log below)
Output: figures/fig6_discrimination_plate_R.png and .pdf
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SPLITS = ['drug_split_1', 'drug_split_0', 'drug_split_2']
LABEL = {'drug_split_1': 'Split 1', 'drug_split_0': 'Split 0', 'drug_split_2': 'Split 2'}
COLOR = {'drug_split_1': '#1f6fe0', 'drug_split_0': '#f26b21', 'drug_split_2': '#1aa37a'}

# Step 15c published intervals (difference from chance), replayed exactly in the Step 26 and 27 scoring logs
# (s26score_19070662.log lines 8/50/92; s27score_19072136.log lines 4/61/118).
S15C = {('recipe', 'drug_split_1'): (0.22234, 0.18829, 0.25411), ('base', 'drug_split_1'): (0.14150, 0.10358, 0.18063),
        ('recipe', 'drug_split_0'): (0.21811, 0.17822, 0.25899), ('base', 'drug_split_0'): (0.14441, 0.10481, 0.18450),
        ('recipe', 'drug_split_2'): (0.21420, 0.18261, 0.24477), ('base', 'drug_split_2'): (0.15488, 0.11017, 0.19767)}

# Step 27 values transcribed from tahoe_results/step27_logs/s27score_19072136.log (used only when the summary CSV
# is not beside the other records). Within-plate PDS under S-split (mean over covered conditions) and the
# contrasts against chance / zero with their 97.5 % intervals.
S27_WP_SPLIT = {  # arm -> split -> PDS
    'recipe':         {'drug_split_1': 0.5030, 'drug_split_0': 0.5172, 'drug_split_2': 0.5264},
    'recipe_fpshuf':  {'drug_split_1': 0.4886, 'drug_split_0': 0.4906, 'drug_split_2': 0.4859},
    'recipe_fpzero':  {'drug_split_1': 0.4969, 'drug_split_0': 0.4983, 'drug_split_2': 0.4972},
    'base':           {'drug_split_1': 0.4934, 'drug_split_0': 0.5078, 'drug_split_2': 0.5200},
    'base_fpshuf':    {'drug_split_1': 0.4961, 'drug_split_0': 0.5007, 'drug_split_2': 0.4994},
    'ctrl_line':      {'drug_split_1': 0.5000, 'drug_split_0': 0.5000, 'drug_split_2': 0.4999},
    'ctrl_sample':    {'drug_split_1': 0.5024, 'drug_split_0': 0.5049, 'drug_split_2': 0.4938},
    'duplicate':      {'drug_split_1': 0.8836, 'drug_split_0': 0.9095, 'drug_split_2': 0.8911}}
S27_CON = {  # name -> split -> (value, lo, hi)
    'H1_recipe_split_minus_0.5':   {'drug_split_1': (0.00296, -0.04093, 0.04755), 'drug_split_0': (0.01722, -0.03936, 0.07134), 'drug_split_2': (0.02644, -0.02134, 0.07621)},
    'base_H1_split_minus_0.5':     {'drug_split_1': (-0.00662, -0.06104, 0.04711), 'drug_split_0': (0.00784, -0.04416, 0.06168), 'drug_split_2': (0.01998, -0.03523, 0.07746)},
    'fpzero_split_minus_0.5':      {'drug_split_1': (-0.00309, -0.04197, 0.03820), 'drug_split_0': (-0.00166, -0.05331, 0.04840), 'drug_split_2': (-0.00280, -0.04546, 0.04138)},
    'ctrl_sample_split_minus_0.5': {'drug_split_1': (0.00237, -0.00910, 0.01419), 'drug_split_0': (0.00489, -0.00964, 0.02008), 'drug_split_2': (-0.00623, -0.02228, 0.00961)},
    'H4_ctrl_line_orig_minus_0.5': {'drug_split_1': (0.00000, -0.03934, 0.04131), 'drug_split_0': (0.00000, -0.04031, 0.04078), 'drug_split_2': (-0.00013, -0.04405, 0.04419)},
    'duplicate_split_minus_0.5':   {'drug_split_1': (0.38363, 0.36655, 0.40020), 'drug_split_0': (0.40953, 0.39126, 0.42631), 'drug_split_2': (0.39106, 0.37209, 0.40945)},
    'H5_residual_true_split':      {'drug_split_1': (0.36417, 0.31798, 0.41046), 'drug_split_0': (0.31953, 0.27873, 0.35747), 'drug_split_2': (0.33805, 0.29113, 0.38107)},
    'H5null_residual_ctrl_h2_split': {'drug_split_1': (-0.00289, -0.01138, 0.00581), 'drug_split_0': (0.00047, -0.00656, 0.00773), 'drug_split_2': (0.00189, -0.00721, 0.01058)}}
S27_RES_SPLIT = {  # change vector -> split -> mean cosine with the plate offset (S-split)
    'recipe':        {'drug_split_1': 0.6277, 'drug_split_0': 0.6352, 'drug_split_2': 0.6235},
    'recipe_fpshuf': {'drug_split_1': 0.6141, 'drug_split_0': 0.6364, 'drug_split_2': 0.6074},
    'recipe_fpzero': {'drug_split_1': 0.7224, 'drug_split_0': 0.7122, 'drug_split_2': 0.7006},
    'base':          {'drug_split_1': 0.2508, 'drug_split_0': 0.2589, 'drug_split_2': 0.2636},
    'base_fpshuf':   {'drug_split_1': 0.2488, 'drug_split_0': 0.2588, 'drug_split_2': 0.2617},
    'ctrl_line':     {'drug_split_1': 0.9817, 'drug_split_0': 0.9830, 'drug_split_2': 0.9807},
    'ctrl_sample':   {'drug_split_1': -0.0209, 'drug_split_0': -0.0067, 'drug_split_2': -0.0166},
    'duplicate':     {'drug_split_1': 0.3134, 'drug_split_0': 0.2771, 'drug_split_2': 0.2944},
    'true':          {'drug_split_1': 0.3642, 'drug_split_0': 0.3195, 'drug_split_2': 0.3381},
    'ctrl_sample_h2': {'drug_split_1': -0.0029, 'drug_split_0': 0.0005, 'drug_split_2': 0.0019}}

ARMS = [('duplicate', 'Second draw of the measured cells'),
        ('ctrl_line', 'Line vehicle mean (no model)'),
        ('ctrl_sample', 'Vehicle resample (no model)'),
        ('base_fpshuf', 'Original model, wrong fingerprint'),
        ('base', 'Original model'),
        ('recipe_fpzero', 'Rewired model, no fingerprint'),
        ('recipe_fpshuf', 'Rewired model, wrong fingerprint'),
        ('recipe', 'Rewired model')]


def load_summary(path):
    rows = list(csv.DictReader(open(path)))
    vals = {(r['split'], r['name']): float(r['value']) for r in rows if r['kind'] == 'value'}
    con = {(r['split'], r['name']): (float(r['value']), float(r['lo']), float(r['hi'])) for r in rows if r['kind'] == 'contrast'}
    return vals, con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='tahoe_results')
    ap.add_argument('--out', default='figures/fig6_discrimination_plate_R')
    a = ap.parse_args()
    v25, c25 = load_summary(os.path.join(a.records, 'step25_swap_summary.csv'))
    v26, c26 = load_summary(os.path.join(a.records, 'step26_source_summary.csv'))
    p27 = os.path.join(a.records, 'step27_within_plate_summary.csv')
    if os.path.isfile(p27):
        v27, c27 = load_summary(p27)
        wp = {arm: {s: v27[(s, '%s__split_wp_pds' % arm)] for s in SPLITS} for arm, _ in ARMS}
        con27 = {k: {s: c27[(s, k)] for s in SPLITS} for k in S27_CON}
        res = {nm: {s: v27[(s, 'residual__%s__split' % nm)] for s in SPLITS} for nm in S27_RES_SPLIT}
        src27 = 'step27_within_plate_summary.csv'
    else:
        wp, con27, res, src27 = S27_WP_SPLIT, S27_CON, S27_RES_SPLIT, 's27score_19072136.log (transcribed)'

    # ---- panel a: cross-plate S-orig PDS with the committed intervals against chance
    def cross(arm, s):
        pt = v26[(s, '%s__orig_pds' % arm)]
        if arm in ('recipe', 'base'):
            d, lo, hi = S15C[(arm, s)]; return pt, 0.5 + lo, 0.5 + hi, 'test'
        if arm == 'recipe_fpshuf':
            d, lo, hi = c25[(s, 'recipe_Q2_shuf_minus_0.5')]; return pt, 0.5 + lo, 0.5 + hi, 'test'
        if arm == 'recipe_fpzero':
            d, lo, hi = c25[(s, 'recipe_ref_zero_minus_0.5')]; return pt, 0.5 + lo, 0.5 + hi, 'test'
        if arm == 'base_fpshuf':
            d, lo, hi = c25[(s, 'base_Q2_shuf_minus_0.5')]; return pt, 0.5 + lo, 0.5 + hi, 'test'
        if arm == 'ctrl_line':
            d, lo, hi = c26[(s, 'G4_ctrl_line_orig_minus_0.5')]; return pt, 0.5 + lo, 0.5 + hi, 'test'
        if arm == 'ctrl_sample':
            d, lo, hi = c26[(s, 'G4b_ctrl_sample_orig_minus_0.5')]; return pt, 0.5 + lo, 0.5 + hi, 'test'
        return pt, None, None, 'ceiling'

    # ---- panel b: within-plate S-split PDS with the Step 27 intervals against chance
    KEY = {'recipe': 'H1_recipe_split_minus_0.5', 'base': 'base_H1_split_minus_0.5', 'recipe_fpzero': 'fpzero_split_minus_0.5',
           'ctrl_sample': 'ctrl_sample_split_minus_0.5', 'ctrl_line': 'H4_ctrl_line_orig_minus_0.5', 'duplicate': 'duplicate_split_minus_0.5'}

    def within(arm, s):
        pt = wp[arm][s]
        if arm in KEY:
            d, lo, hi = con27[KEY[arm]][s]; return pt, 0.5 + lo, 0.5 + hi, 'test'
        return pt, None, None, 'point'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.0]))
    yoff = {'drug_split_1': 0.22, 'drug_split_0': 0.0, 'drug_split_2': -0.22}
    for ax, getter, title, xlab, ref in [
            (axes[0], cross, 'a  Cross-plate candidates (published construction)', 'Discrimination score (0.5 = chance)', 0.5),
            (axes[1], within, 'b  Within-plate candidates', 'Discrimination score (0.5 = chance)', 0.5)]:
        for i, (arm, lab) in enumerate(ARMS):
            for s in SPLITS:
                pt, lo, hi, kind = getter(arm, s)
                y = i + yoff[s]; c = COLOR[s]
                if kind == 'ceiling':
                    ax.plot(pt, y, marker='D', ms=7, color=c, ls='none')
                elif kind == 'point':
                    ax.plot(pt, y, marker='o', ms=4, color=c, ls='none')
                else:
                    excl = lo > ref or hi < ref
                    ax.plot([lo, hi], [y, y], color=c, lw=1.6)
                    ax.plot(pt, y, marker='o', ms=7, color=c, mfc=c if excl else 'white', mew=1.6, ls='none')
        ax.axvline(ref, ls=':', color='k', lw=1)
        ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([lab for _, lab in ARMS])
        ax.set_title(title, loc='left', fontsize=11); ax.set_xlabel(xlab)
        ax.grid(axis='x', color='0.9'); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    axes[1].set_yticklabels([])
    axes[0].set_xlim(0.42, 1.0); axes[1].set_xlim(0.42, 1.0)

    # ---- panel c: alignment with the plate offset
    ax = axes[2]
    rows_c = ARMS + [('true', 'Measured change (real treated cells)'), ('ctrl_sample_h2', 'Vehicle resample, measured-change half (null)')]
    for i, (nm, lab) in enumerate(rows_c):
        for s in SPLITS:
            y = i + yoff[s]; c = COLOR[s]; pt = res[nm][s]
            if nm == 'true':
                d, lo, hi = con27['H5_residual_true_split'][s]
                ax.plot([lo, hi], [y, y], color=c, lw=1.6); ax.plot(pt, y, 'o', ms=7, color=c, ls='none')
            elif nm == 'ctrl_sample_h2':
                d, lo, hi = con27['H5null_residual_ctrl_h2_split'][s]
                ax.plot([lo, hi], [y, y], color=c, lw=1.6); ax.plot(pt, y, 'o', ms=7, color=c, mfc='white', mew=1.6, ls='none')
            else:
                ax.plot(pt, y, marker='o', ms=4, color=c, ls='none')
    ax.axvline(0, ls=':', color='k', lw=1)
    ax.set_yticks(range(len(rows_c))); ax.set_yticklabels([lab for _, lab in rows_c])
    ax.yaxis.tick_right()
    ax.set_title('c  Alignment with the plate\'s vehicle offset', loc='left', fontsize=11)
    ax.set_xlabel('Cosine between the change and the plate offset')
    ax.grid(axis='x', color='0.9'); ax.set_axisbelow(True)
    for sp in ('top', 'left'): ax.spines[sp].set_visible(False)

    handles = [plt.Line2D([], [], marker='o', color=COLOR[s], ls='none', ms=7, label=LABEL[s]) for s in SPLITS]
    handles += [plt.Line2D([], [], marker='o', color='0.3', ls='none', ms=7, label='filled: interval excludes chance / zero'),
                plt.Line2D([], [], marker='o', color='0.3', mfc='white', ls='none', ms=7, label='hollow: includes'),
                plt.Line2D([], [], marker='D', color='0.3', ls='none', ms=7, label='diamond: ceiling, no test'),
                plt.Line2D([], [], marker='o', color='0.3', ls='none', ms=4, label='small: point only, no interval computed')]
    fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    import textwrap
    note = ('a: candidates are the other held-out compounds in the same cell line and dose, on any plate; the vehicle mean of the '
            'condition\'s own plate is subtracted on both sides. b: candidates restricted to compounds measured on the condition\'s own '
            'plate (sets of at least 10); predicted and measured changes subtract disjoint halves of the plate\'s vehicle pool. '
            'c: mean cosine between each arm\'s predicted change (against one half-pool mean) and the plate\'s vehicle offset (line mean '
            'minus the other half-pool mean); the measured change is the real treated cells. Intervals: 97.5 %% structure-clustered '
            'bootstrap. Sources: step25_swap_summary.csv, step26_source_summary.csv, %s.' % src27)
    fig.text(0.01, -0.06, '\n'.join(textwrap.wrap(note, 190)), fontsize=8, color='0.35', va='top')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out + '.png', dpi=300, bbox_inches='tight'); fig.savefig(a.out + '.pdf', bbox_inches='tight')
    print('wrote', a.out + '.png/.pdf')


if __name__ == '__main__':
    main()
