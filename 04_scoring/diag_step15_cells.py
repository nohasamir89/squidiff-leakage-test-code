#!/usr/bin/env python
"""Step 15b, post-hoc diagnostic of the generated cell populations (exploratory, 2026-10-01).

Why: in the final scoring (commit ed423ba) the base arm's energy distance is 2.27 / 2.14 / 2.02 on splits 1 / 0 / 2
while every other submission, including control-paste (the vehicle cells themselves), lies between 0.15 and 0.39.
A value that far from the vehicle cells is not a ranking problem; it says the base model's cells differ from the
real cells in scale or spread. This script measures which, from the stored per-cell populations the scorer used.
It changes no endpoint and feeds no contrast; its numbers are reported as a labelled diagnostic only.

What it computes, per condition, from vectors_<TAG>_pc.npz (pred = generated cells, true = the stored real cells,
cap 200) and vectors_<TAG>.npz (true / C = real-treated and paired-control mean vectors):
  energy terms   xy = mean ||x - y||, xx = mean ||x - x'||, yy = mean ||y - y'||, E = 2 xy - xx - yy  (the scorer's
                 U-statistic, so E here reproduces the scored energy per condition);
  library size   mean per-cell sum over genes, generated vs real;
  zeros          fraction of exactly-zero entries, generated vs real; and the minimum generated value;
  profile        Pearson r between the generated and real mean profiles across genes, and the least-squares slope
                 of generated on real (1 = same scale);
  distances      ||mean(x) - mean(y)|| (generated mean to real-treated mean), ||mean(x) - C|| (to the paired
                 control mean) and ||true - C|| (the real vehicle-to-treated shift, the scale of the effect).

Reading the output: xx >> yy with xy large = over-dispersed (noisy) cells; xx << yy = collapsed cells; xx ~ yy with
||mean(x) - mean(y)|| large = a location (scale) error; slope far from 1 = a global scale error.

Usage (from the repo root, conda env active):
  python -u diag_step15_cells.py --split drug_split_1 --seed 0 --arms base recipe
Writes tahoe_results/step15_diag_cells_<split>_seed_<seed>.json (summary) and ..._per_condition.csv.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr

RES = 'tahoe_results'
ARMS = ('base', 'recipe', 'guidance_w2', 'attn_cross', 'mlp_ctrl')


def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            m.update(c)
    return m.hexdigest()


def load(split, arm, seed):
    """The same files the scorer loaded, verified against the same provenance hashes."""
    tag = 's15_%s_%s_seed_%d' % (split, arm, seed)
    f, fpc, pj = (os.path.join(RES, 'vectors_%s.npz' % tag), os.path.join(RES, 'vectors_%s_pc.npz' % tag),
                  os.path.join('provenance', 's15ev_%s.json' % tag))
    for q in (f, fpc, pj):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    pv = json.load(open(pj))
    if pv.get('vectors_sha256') != sha(f) or pv.get('per_cell_sha256') != sha(fpc):
        sys.exit('FATAL: %s arrays differ from their provenance hashes' % tag)
    v = np.load(f, allow_pickle=True)
    p = np.load(fpc, allow_pickle=True)
    cond = np.array([str(x) for x in v['cond']])
    if len(p['pred']) != len(cond) or len(p['true']) != len(cond):
        sys.exit('FATAL: %s per-cell length differs from vectors' % tag)
    return tag, cond, np.asarray(v['true'], np.float64), np.asarray(v['C'], np.float64), p['pred'], p['true']


def per_condition(X, Y, true_mean, C):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    xy, xx, yy = cdist(X, Y).mean(), pdist(X).mean(), pdist(Y).mean()
    mx, my = X.mean(0), Y.mean(0)
    r = pearsonr(mx, my)[0] if mx.std() > 1e-12 and my.std() > 1e-12 else np.nan
    slope = float(np.dot(mx - mx.mean(), my - my.mean()) / np.dot(my - my.mean(), my - my.mean())) if my.std() > 1e-12 else np.nan
    return dict(n_pred=len(X), n_true=len(Y), xy=xy, xx=xx, yy=yy, energy=2 * xy - xx - yy,
                lib_pred=float(X.sum(1).mean()), lib_true=float(Y.sum(1).mean()),
                zero_pred=float((X == 0).mean()), zero_true=float((Y == 0).mean()), min_pred=float(X.min()),
                profile_r=float(r), slope_pred_on_true=slope,
                d_mean_pred_true=float(np.linalg.norm(mx - my)), d_mean_pred_ctrl=float(np.linalg.norm(mx - C)),
                d_true_ctrl=float(np.linalg.norm(true_mean - C)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='drug_split_1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arms', nargs='+', default=['base', 'recipe'], choices=ARMS)
    ap.add_argument('--max-conditions', type=int, default=0, help='0 = all conditions')
    a = ap.parse_args()
    t0 = time.time()
    keys = ('xy', 'xx', 'yy', 'energy', 'lib_pred', 'lib_true', 'zero_pred', 'zero_true', 'min_pred', 'profile_r',
            'slope_pred_on_true', 'd_mean_pred_true', 'd_mean_pred_ctrl', 'd_true_ctrl')
    summary, rows = {}, []
    for arm in a.arms:
        tag, cond, true_mean, C, pred, true_cells = load(a.split, arm, a.seed)
        n = len(cond) if a.max_conditions == 0 else min(a.max_conditions, len(cond))
        recs = []
        for i in range(n):
            r = per_condition(pred[i], true_cells[i], true_mean[i], C[i])
            recs.append(r)
            rows.append(dict(arm=arm, condition=cond[i], **r))
        summary[arm] = dict(tag=tag, n_conditions=n,
                            mean={k: float(np.nanmean([r[k] for r in recs])) for k in keys},
                            median={k: float(np.nanmedian([r[k] for r in recs])) for k in keys})
        m, md = summary[arm]['mean'], summary[arm]['median']
        print('[%s] %d conditions | energy mean %.4f (xy %.3f xx %.3f yy %.3f) | library pred %.1f true %.1f | '
              'zeros pred %.3f true %.3f | min pred %.3f | profile r %.3f slope %.3f | ||mean(x)-mean(y)|| %.3f '
              '||mean(x)-C|| %.3f ||true-C|| %.3f  [medians: energy %.4f r %.3f slope %.3f]'
              % (arm, n, m['energy'], m['xy'], m['xx'], m['yy'], m['lib_pred'], m['lib_true'], m['zero_pred'],
                 m['zero_true'], m['min_pred'], m['profile_r'], m['slope_pred_on_true'], m['d_mean_pred_true'],
                 m['d_mean_pred_ctrl'], m['d_true_ctrl'], md['energy'], md['profile_r'], md['slope_pred_on_true']))
        sys.stdout.flush()
    stem = os.path.join(RES, 'step15_diag_cells_%s_seed_%d' % (a.split, a.seed))
    json.dump(dict(split=a.split, seed=a.seed, arms=a.arms, max_conditions=a.max_conditions,
                   note='post-hoc exploratory diagnostic of 2026-10-01; no pre-registered endpoint depends on it',
                   summary=summary), open(stem + '.json', 'w'), indent=1)
    with open(stem + '_per_condition.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['arm', 'condition', 'n_pred', 'n_true'] + list(keys))
        w.writeheader()
        w.writerows(rows)
    print('wrote %s.json and %s_per_condition.csv  (%d s)' % (stem, stem, time.time() - t0))


if __name__ == '__main__':
    main()
