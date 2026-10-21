#!/usr/bin/env python
"""Step 15c, post-hoc diagnostic on the development split (2026-10-03, exploratory, no endpoint depends on it).

Why. In the development run the recipe's PDS (0.7223) equals the plate 6 / 14 replicate ceiling (0.7263) and is flat
across doses (0.73 / 0.71 / 0.73), while every fingerprint-based linear shift sits at 0.50-0.52. The arms are
conditioned on control cells of the condition's own line AND PLATE (protocol B), so a prediction can carry a
plate-specific component of the treated state; in the pre-registered candidate set (same line, same dose, all
plates) the own match shares the prediction's plate and most competitors do not, so a plate component would raise
the own match's rank without any drug information. A plate effect does not depend on dose, which the flat profile
is consistent with. This script measures discrimination with the plate taken out of the comparison.

What. For every plate 6 / 14 replicate pair (p6, p14) and both directions (src -> dst):
  cross-plate PDS   prediction = the arm's predicted effect for the plate-src condition; candidates = the (line, dose)
                    context's conditions on plates other than src, minus other conditions of the same drug, with the
                    plate-dst copy as the own match. The prediction's plate is absent from the candidates, so a plate
                    component cannot help it; the own match is on another plate, so if anything it is penalised.
  ceiling           the measured plate-src effect offered as the prediction, on exactly the same candidate set.
  ordinary PDS      the pre-registered value of the plate-src condition (same line, same dose, all plates), for the
                    same conditions, read from the development per-condition record.
Per arm and per dose, averaged over seeds and directions. If the cross-plate PDS stays at the ordinary value and near
the ceiling, the discrimination is drug-specific; if it falls toward 0.5, it was plate.

Usage: python -u diag_step15c_crossplate.py            (development split only)
Writes tahoe_results/step15c_diag_crossplate_drug_split_1.json and ..._pairs.csv
"""
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15        # noqa: E402
import score_step15c_tahoe as s15c      # noqa: E402

SPLIT = 'drug_split_1'
PC_DEV = os.path.join(s15.RES, 'step15c_pds_per_condition_development.csv')
T0 = time.time()


def main():
    for q in (PC_DEV, s15c.PC15):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    stem = os.path.join(s15.RES, 'step15c_diag_crossplate_%s' % SPLIT)
    if os.path.exists(stem + '.json'):
        sys.exit('FATAL: %s.json exists' % stem)
    arms = {arm: [s15c.load_vectors(SPLIT, arm, s) for s in range(s15.N_SEEDS)] for arm in s15.ARMS}
    ref = arms['recipe'][0]
    cond, n = ref['cond'], len(ref['cond'])
    Y = ref['true'] - ref['C']
    pc = pd.read_csv(s15c.PC15)
    pc = pc[pc['split'] == SPLIT].reset_index(drop=True)
    dev = pd.read_csv(PC_DEV)
    dev = dev[dev['split'] == SPLIT].reset_index(drop=True)
    for df in (pc, dev):
        if len(df) != n or not np.array_equal(df['condition'].astype(str).values, cond):
            sys.exit('FATAL: a per-condition record does not match the vectors')
    lines, plates, drugs, doses = (pc['line'].astype(str).values, pc['plate'].astype(str).values,
                                   pc['structure'].astype(str).values, pc['dose_nM'].astype(float).values)
    ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
    members_of = {u: np.where(ctx_key == u)[0] for u in np.unique(ctx_key)}

    keyed = {}
    for i, c in enumerate(cond):
        key, plate = s15.parse_cond(c)
        if plate in s15.REP_PLATES:
            keyed.setdefault(key, {})[plate] = i
    pairs = [(v['6'], v['14']) for v in keyed.values() if '6' in v and '14' in v]
    print('[design] %s: %d conditions, %d plate 6/14 pairs' % (SPLIT, n, len(pairs)))

    pred = {arm: [v['B'] - v['C'] for v in L] for arm, L in arms.items()}
    rows = []
    for i6, i14 in pairs:
        for src, dst in ((i6, i14), (i14, i6)):
            members = members_of[ctx_key[dst]]
            cand = np.array([q for q in members if plates[q] != plates[src] and (q == dst or drugs[q] != drugs[dst])])
            if len(cand) < s15c.MIN_CAND:
                continue
            r = dict(split=SPLIT, src=cond[src], dst=cond[dst], line=lines[dst], dose_nM=doses[dst], n_candidates=len(cand))
            E = Y.copy()
            E[dst] = Y[src]
            r['ceiling'] = s15c.pds_sets(E, Y, [members], {dst: cand})[dst]
            for arm in s15.ARMS:
                vals = []
                for P in pred[arm]:
                    E = Y.copy()
                    E[dst] = P[src]                                   # the arm's prediction for the plate-src condition, scored at dst
                    vals.append(s15c.pds_sets(E, Y, [members], {dst: cand})[dst])
                r['%s_crossplate' % arm] = float(np.mean(vals))
                r['%s_ordinary' % arm] = float(dev['%s_pds' % arm].values[src])
            rows.append(r)
    if not rows:
        sys.exit('FATAL: no pair had a candidate set of at least %d' % s15c.MIN_CAND)
    df = pd.DataFrame(rows)
    summary = dict(split=SPLIT, n_pairs=len(pairs), n_evaluations=len(df), min_candidates=s15c.MIN_CAND,
                   candidates=[int(df['n_candidates'].min()), int(df['n_candidates'].max())],
                   note='post-hoc exploratory diagnostic of 2026-10-03 on the development split; no endpoint depends on it',
                   ceiling=dict(mean=float(df['ceiling'].mean()), per_dose={('%g' % d): float(df.loc[np.isclose(df['dose_nM'], d), 'ceiling'].mean()) for d in sorted(set(df['dose_nM']))}),
                   arms={})
    print('[ceiling] cross-plate replicate: mean %.4f | per dose %s' % (summary['ceiling']['mean'], {k: round(v, 4) for k, v in summary['ceiling']['per_dose'].items()}))
    for arm in s15.ARMS:
        o, x = df['%s_ordinary' % arm], df['%s_crossplate' % arm]
        summary['arms'][arm] = dict(ordinary_mean=float(o.mean()), crossplate_mean=float(x.mean()),
                                    crossplate_per_dose={('%g' % d): float(x[np.isclose(df['dose_nM'], d)].mean()) for d in sorted(set(df['dose_nM']))},
                                    ordinary_per_dose={('%g' % d): float(o[np.isclose(df['dose_nM'], d)].mean()) for d in sorted(set(df['dose_nM']))})
        a = summary['arms'][arm]
        print('[%-11s] ordinary PDS on these conditions %.4f | cross-plate PDS %.4f | cross-plate per dose %s | ordinary per dose %s'
              % (arm, a['ordinary_mean'], a['crossplate_mean'], {k: round(v, 4) for k, v in a['crossplate_per_dose'].items()},
                 {k: round(v, 4) for k, v in a['ordinary_per_dose'].items()}))
    json.dump(summary, open(stem + '.json', 'w'), indent=1)
    df.to_csv(stem + '_pairs.csv', index=False)
    print('wrote %s.json and %s_pairs.csv  (%d s)' % (stem, stem, time.time() - T0))


if __name__ == '__main__':
    main()
