#!/usr/bin/env python
"""Step 16, post-hoc diagnostic (2026-10-09, after the final Step 16 record; exploratory, no endpoint depends on it).

Why. In step16_findings.md section 4.4 every submission's MRA is a mean over all 42 (line, dose) contexts, while the
replicate ceiling can only be scored where a condition has a copy on the other plate - plate 14 replicates plate 6 at
the top dose only - so the ceiling is a mean over 14 contexts and over the replicated conditions inside them. The two
numbers are therefore not scored on the same set, and the sentence "about 0.6 of a repeat experiment" rests on an
approximation. This script scores every submission on EXACTLY the ceiling's conditions.

What. From tahoe_results/step16_per_condition.csv (commit 6baefc3; every submission's seed-mean predicted effect norm
and the measured norm per condition), per split:
  restricted MRA     for each (line, dose) context with >= 10 conditions that have a plate 6 / 14 copy: the Spearman
                     between a submission's predicted norm and the measured norm over those replicated conditions
                     only; mean over those contexts. The ceiling on the same conditions = the copy's measured norm.
  full MRA           the same submissions over all 42 contexts, from the same seed-mean norms - printed next to the
                     findings' per-seed values so the reader sees how much the seed-mean convention changes them
                     (the per-condition record stores seed means, not per-seed norms).
  intervals          drug-clustered percentile bootstrap, 5,000 replicates, RandomState(0), 97.5 % per split, the
                     statistic recomputed on the resampled drugs (as the MRA bootstrap of score_step16_tahoe);
                     contrasts recipe - ceiling, targets - ceiling, targets_shuf - ceiling; ratio recipe / ceiling.
Guards: the record committed and clean; the restricted contexts count equals the findings' 14 per split; the ceiling on
the restricted set equals the Step 16 record's replicate MRA to 1e-9 (same definition, same conditions).

Usage: python -u diag_step16_mra_ceiling.py
Writes tahoe_results/step16_diag_mra_ceiling.json and _long.csv. Never overwrites.
"""
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402  parse_cond, REP_PLATES, verdict, fmt
import score_step16_tahoe as s16                      # noqa: E402  fast_rank, MIN_CTX, N_BOOT, LO, HI

PC = os.path.join(s15.RES, 'step16_per_condition.csv')
REC = os.path.join(s15.RES, 'step16_analysis.json')
OUT = os.path.join(s15.RES, 'step16_diag_mra_ceiling.json')
OUT_CSV = os.path.join(s15.RES, 'step16_diag_mra_ceiling_long.csv')
SUBS = ('ridge_S7', 'ridge_train', 'base', 'recipe', 'guidance_w2', 'attn_cross', 'mlp_ctrl', 'targets', 'targets_shuf')
T0 = time.time()


def spearman(x, y):
    """Average-rank Spearman; 0 when either side is constant (score_step15d_utility.spearman0's convention)."""
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    rx, ry = s16.fast_rank(np.asarray(x, np.float64)), s16.fast_rank(np.asarray(y, np.float64))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def mra(pred, true, ctx_id, n_ctx, sel):
    """Mean over contexts (>= MIN_CTX selected conditions) of the Spearman between pred and true on the selection."""
    c_of = ctx_id[sel]
    rhos = [spearman(pred[m], true[m]) for c in range(n_ctx) for m in [sel[c_of == c]] if len(m) >= s16.MIN_CTX]
    return float(np.mean(rhos)) if rhos else np.nan


def main():
    for q in (OUT, OUT_CSV):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    tracked = '%s %s diag_step16_mra_ceiling.py score_step16_tahoe.py' % (PC, REC)
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: record or scripts not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0:
        sys.exit('FATAL: record or scripts uncommitted')
    rec = json.load(open(REC))
    if not rec.get('final'):
        sys.exit('FATAL: %s is not the final record' % REC)
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    pc_all = pd.read_csv(PC)
    print('[inputs] %s (%d rows), record commit %s, this commit %s' % (PC, len(pc_all), str(rec.get('commit'))[:7], commit[:7]))

    results, rows = {}, []
    for split in s15.ALL_SPLITS:
        pc = pc_all[pc_all['split'] == split].reset_index(drop=True)
        n = len(pc)
        cond = pc['condition'].astype(str).values
        lines, doses, drugs = pc['line'].astype(str).values, pc['dose_nM'].astype(float).values, pc['structure'].astype(str).values
        true = pc['true_effect_norm'].astype(float).values
        ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
        ctx_u = np.unique(ctx_key)
        ctx_id = np.array([int(np.where(ctx_u == k)[0][0]) for k in ctx_key])
        n_ctx = len(ctx_u)
        # replicate partner: the same (line, drug, dose) on the other of plates 6 / 14
        keyed = {}
        for i, c in enumerate(cond):
            key, plate = s15.parse_cond(c)
            if plate in s15.REP_PLATES:
                keyed.setdefault(key, {})[plate] = i
        partner = np.full(n, -1)
        for v in keyed.values():
            if '6' in v and '14' in v:
                partner[v['6']], partner[v['14']] = v['14'], v['6']
        rep = np.where(partner >= 0)[0]
        if any(ctx_id[partner[i]] != ctx_id[i] for i in rep):
            sys.exit('FATAL: a replicate pair spans two contexts')
        ceil_pred = np.zeros(n)
        ceil_pred[rep] = true[partner[rep]]
        elig = [c for c in range(n_ctx) if (ctx_id[rep] == c).sum() >= s16.MIN_CTX]
        rec_split = rec['per_split'][split]
        if len(elig) != int(rec_split['replicate']['mra_n_contexts']):
            sys.exit('FATAL: %d eligible contexts, the record says %d' % (len(elig), rec_split['replicate']['mra_n_contexts']))
        full = np.arange(n)
        preds = {nm: pc['%s_effect_norm' % nm].astype(float).values for nm in SUBS}
        preds['ceiling'] = ceil_pred

        def restricted(sel):
            s_ = sel[partner[sel] >= 0]
            return {nm: mra(p, true, ctx_id, n_ctx, s_) for nm, p in preds.items()}

        R = restricted(full)
        ceil_rec = rec_split['replicate']['mra']
        if ceil_rec is None or abs(R['ceiling'] - float(ceil_rec)) > 1e-9:
            sys.exit('FATAL: restricted ceiling %.9f does not reproduce the record %r' % (R['ceiling'], ceil_rec))
        F = {nm: mra(preds[nm], true, ctx_id, n_ctx, full) for nm in SUBS}
        print('\n== %s: %d conditions, %d contexts, %d with >= %d replicated conditions (%d replicated conditions in them)'
              % (split, n, n_ctx, len(elig), s16.MIN_CTX, int(sum((ctx_id[rep] == c).sum() for c in elig))))
        print('   restricted ceiling %.4f reproduces the Step 16 record' % R['ceiling'])
        print('   %-13s %10s %12s %14s' % ('submission', 'restricted', 'full (mean)', 'full (record)'))
        for nm in SUBS:
            print('   %-13s %+10.4f %+12.4f %+14.4f' % (nm, R[nm], F[nm], float(rec_split['metrics'][nm]['mra'])))
        print('   %-13s %+10.4f' % ('ceiling', R['ceiling']))

        # bootstrap over drugs, the statistic recomputed on the restricted set of the resampled drugs
        rng = np.random.RandomState(0)
        groups = [np.where(drugs == u)[0] for u in np.unique(drugs)]
        B = {nm: np.full(s16.N_BOOT, np.nan) for nm in preds}
        for b in range(s16.N_BOOT):
            sel = np.concatenate([groups[g] for g in rng.randint(0, len(groups), len(groups))])
            got = restricted(sel)
            for nm in preds:
                B[nm][b] = got[nm]

        def con(name, point, dist):
            lo, hi = float(np.nanpercentile(dist, s16.LO)), float(np.nanpercentile(dist, s16.HI))
            c = dict(diff=float(point), ci=[lo, hi], verdict=s15.verdict(lo, hi), n_nan=int(np.isnan(dist).sum()))
            print('   %-36s %s' % (name, s15.fmt(c)))
            rows.append(dict(split=split, quantity=name, value=point, ci_lo=lo, ci_hi=hi, verdict=c['verdict']))
            return c

        print('   intervals on the restricted set (%d replicates):' % s16.N_BOOT)
        contrasts = {nm + '_minus_0': con('%s - 0' % nm, R[nm], B[nm]) for nm in preds}
        for nm in ('recipe', 'targets', 'targets_shuf', 'ridge_train'):
            contrasts[nm + '_minus_ceiling'] = con('%s - ceiling' % nm, R[nm] - R['ceiling'], B[nm] - B['ceiling'])
        ratio = dict(point=float(R['recipe'] / R['ceiling']) if R['ceiling'] != 0 else None,
                     ci=[float(np.nanpercentile(B['recipe'] / B['ceiling'], s16.LO)), float(np.nanpercentile(B['recipe'] / B['ceiling'], s16.HI))]
                     if contrasts['ceiling_minus_0']['verdict'] == 'above zero' else None)
        print('   ratio recipe / ceiling on the same conditions: %s %s' % ('%.3f' % ratio['point'] if ratio['point'] is not None else 'undefined',
                                                                           ('[%.3f, %.3f]' % tuple(ratio['ci'])) if ratio['ci'] else '(ceiling not above zero: no interval)'))
        results[split] = dict(n_conditions=n, n_contexts=n_ctx, n_contexts_restricted=len(elig), n_replicated_conditions=int(len(rep)),
                              restricted={nm: float(v) for nm, v in R.items()}, full_seed_mean={nm: float(v) for nm, v in F.items()},
                              full_record={nm: float(rec_split['metrics'][nm]['mra']) for nm in SUBS}, contrasts=contrasts, ratio=ratio)
        for nm in preds:
            rows.append(dict(split=split, quantity='restricted_%s' % nm, value=R[nm], ci_lo=None, ci_hi=None, verdict=None))
    json.dump(dict(note='post-hoc exploratory diagnostic of 2026-10-09; no endpoint depends on it', record=REC, record_commit=rec.get('commit'),
                   record_sha256=s15.sha(REC), per_condition_sha256=s15.sha(PC), commit=commit, n_boot=s16.N_BOOT, min_context=s16.MIN_CTX,
                   per_split=results, seconds=round(time.time() - T0)), open(OUT, 'w'), indent=1)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'quantity', 'value', 'ci_lo', 'ci_hi', 'verdict'])
        w.writeheader(); w.writerows(rows)
    print('\nwrote %s and %s (%.0f s)' % (OUT, OUT_CSV, time.time() - T0))


if __name__ == '__main__':
    main()
