#!/usr/bin/env python3
"""
verify_step15c_intervals.py - prove why the Step 25 scorer's bootstrap gate failed on the base arm.

Purpose
-------
The Step 25 scorer (job 19044477) reproduced every per-condition PDS to 1e-9 and the base arm's point estimate
exactly (+0.14150), but its 97.5 % interval differed from the committed Step 15c value by ~0.001 per bound. The
suspected cause is the documented draw-order dependence of Step 15c's bootstrap (Step 18 findings section 9 item
12; Step 20 findings section 6 item 6): Step 15c drew every interval from ONE generator created once per run
(score_step15_tahoe.py line 66, RandomState(0)), so an interval's bounds depend on how many draws preceded it.
Only the first draw of the run - split 1's recipe Q1 - is reproducible by a fresh RandomState(0), which is why
that gate passed in Steps 23, 24 and 25 and the base gate did not.

Logic
-----
Replay Step 15c's draw sequence exactly: reset score_step15_tahoe.rng to RandomState(0) once, then for each split
in Step 15c's order (ALL_SPLITS: 1, 0, 2) call Step 15c's own committed `contrast` function with the committed
per-condition PDS columns, in the order score_step15c_tahoe.main() calls it (Q1, Q2, the secondary <name> - 0.5
loop over SUBS skipping recipe and control_paste, then recipe - ridge_S7, guidance_w2 - recipe,
attn_cross - mlp_ctrl). Compare the recipe Q1 and base - 0.5 intervals of every split to the committed Step 15c
constants (the REF_BOOT of score_step25_tahoe.py) at 5e-6. Beside each, print the fresh-RandomState(0) interval
(the Step 20+ convention, score_step23_screen.Boot) to show the two conventions side by side.

Reads only the committed tahoe_results/step15c_pds_per_condition.csv. Writes nothing. Run from the worktree root.
PASS on every line = the diagnosis is proven and the committed Step 15c numbers are exactly recoverable.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15      # noqa: E402  the shared generator lives here (module-level rng)
import score_step15c_tahoe as s15c    # noqa: E402  Step 15c's contrast wrapper and SUBS order
import score_step23_screen as s23     # noqa: E402  the fresh-generator convention used from Step 20 on

# committed Step 15c findings, section 1 (recipe) and section 3 (base): (point, lo, hi)
REF = {
    'recipe': {'drug_split_1': (0.22234, 0.18829, 0.25411), 'drug_split_0': (0.21811, 0.17822, 0.25899),
               'drug_split_2': (0.21420, 0.18261, 0.24477)},
    'base':   {'drug_split_1': (0.14150, 0.10358, 0.18063), 'drug_split_0': (0.14441, 0.10481, 0.18450),
               'drug_split_2': (0.15488, 0.11017, 0.19767)},
}
TOL = 5e-6 + 1e-9
PC = os.path.join(s15.RES, 'step15c_pds_per_condition.csv')


def step15c_sequence(pds, half, drugs):
    """The contrast calls of score_step15c_tahoe.main(), in its order; each advances the shared generator."""
    seq = [('Q1_recipe', pds['recipe'], half), ('Q2_recipe_ridge_train', pds['recipe'], pds['ridge_train'])]
    for name in s15c.SUBS:
        if name in ('recipe', 'control_paste'):
            continue
        seq.append(('sec_%s' % name, pds[name], half))
    seq += [('sec_recipe_ridge_S7', pds['recipe'], pds['ridge_S7']),
            ('sec_guidance_w2_recipe', pds['guidance_w2'], pds['recipe']),
            ('exp_attn_cross_mlp_ctrl', pds['attn_cross'], pds['mlp_ctrl'])]
    return seq


def main():
    if not os.path.isfile(PC):
        sys.exit('FATAL: missing %s' % PC)
    pcc = pd.read_csv(PC)
    s15.rng = np.random.RandomState(0)          # Step 15c's single generator, at the state it had when its run began
    print('replaying Step 15c draw order with its committed functions on %s' % PC)
    print('SUBS order: %s\n' % ', '.join(s15c.SUBS))
    n_fail = 0
    for split in s15.ALL_SPLITS:
        d = pcc[pcc['split'] == split].reset_index(drop=True)
        if len(d) == 0:
            sys.exit('FATAL: no rows for %s' % split)
        drugs = d['structure'].astype(str).values
        n = len(d)
        half = np.full(n, 0.5)
        pds = {name: d['%s_pds' % name].values.astype(np.float64) for name in s15c.SUBS}
        got = {}
        for lab, xb, xa in step15c_sequence(pds, half, drugs):
            got[lab] = s15c.contrast(xb, xa, drugs)      # consumes s15.rng exactly as Step 15c did
        print('%s  (%d conditions, draws so far advance the shared generator)' % (split, n))
        for arm, lab in (('recipe', 'Q1_recipe'), ('base', 'sec_base')):
            c = got[lab]
            mine = (c['diff'], c['ci'][0], c['ci'][1])
            ref = REF[arm][split]
            ok = all(abs(a - b) <= TOL for a, b in zip(mine, ref))
            n_fail += 0 if ok else 1
            # the fresh-generator convention (Step 20 onward), for comparison
            b = s23.Boot(np.arange(n), drugs, n)
            pt, reps = b.mean_reps(pds[arm] - 0.5)
            q = s23.interval(pt, reps)
            print('  %-7s replay  %+.5f [%+.5f, %+.5f]   committed %+.5f [%+.5f, %+.5f]   %s'
                  % (arm, *mine, *ref, 'PASS' if ok else 'FAIL'))
            print('          fresh   %+.5f [%+.5f, %+.5f]   (RandomState(0) per contrast; bounds move %+.5f / %+.5f)'
                  % (q['diff'], q['ci'][0], q['ci'][1], q['ci'][0] - ref[1], q['ci'][1] - ref[2]))
    print('\nRESULT: %s' % ('ALL PASS - Step 15c intervals reproduced exactly by replaying its draw order'
                            if n_fail == 0 else '%d FAIL - diagnosis not confirmed; do not amend' % n_fail))


if __name__ == '__main__':
    main()
