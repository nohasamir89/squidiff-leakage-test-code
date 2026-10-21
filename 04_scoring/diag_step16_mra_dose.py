#!/usr/bin/env python
"""Step 16, second post-hoc diagnostic (2026-10-09, after section 9 of step16_findings.md; exploratory, no endpoint depends on it).

Why. Section 9 showed that on the replicate ceiling's own conditions (top dose, plates 6 / 14 only, 14 contexts) no
submission's MRA resolves, while over all 42 (line, dose) contexts the recipe scores about +0.3. The restricted set
changes three things at once: the dose (5 uM only), the plates (6 and 14 only) and the number of drugs per context.
This script separates them, from the same committed per-condition record, with no new predictions.

What, per split, for every submission (seed-mean effect norms, as section 9):
  MRA per dose        the 14 contexts of each dose level over ALL their conditions (the 42-context MRA of section 4.4
                      is the mean of these three, which is asserted);
  top dose, split     the 14 top-dose contexts split into the conditions WITH a plate 6 / 14 copy (= section 9's
                      restricted set, asserted to reproduce it to 1e-9) and the conditions WITHOUT one (other plates);
  intervals           drug-clustered percentile bootstrap, 5,000 replicates, RandomState(0), 97.5 % per split, the
                      statistic recomputed on the resampled drugs, with the same draws as diag_step16_mra_ceiling.py;
                      every quantity against zero, and for the recipe: top dose minus each lower dose, and
                      (other plates) minus (plates 6 / 14) at the top dose.
Written before the run, as section 9 says: if the recipe's top-dose MRA over all conditions is near zero, the section
4.4 result is a low-dose result; if it is near +0.3 and the other-plate conditions score near +0.3 while the plate
6 / 14 conditions do not, the loss belongs to the replicated subset; anything in between is unresolved.

The MRA is computed by the Step 16 scorer's own functions (score_step16_tahoe.prep_ctx / mra_pre / ctx_split).
Guards: the record, the section 9 record and the scripts committed and clean; exactly three dose levels with 14
contexts each; every eligible replicate context at the top dose; the per-dose mean reproduces the full-set seed-mean
MRA of section 9 to 1e-9; the plate 6 / 14 values reproduce section 9's restricted values to 1e-9.

Usage: python -u diag_step16_mra_dose.py            (--n-boot N only for synthetic tests; the record stores the value used)
Writes tahoe_results/step16_diag_mra_dose.json and _long.csv. Never overwrites.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402  parse_cond, REP_PLATES, verdict, fmt, sha
import score_step16_tahoe as s16                      # noqa: E402  prep_ctx, mra_pre, ctx_split, MIN_CTX, N_BOOT, LO, HI

PC = os.path.join(s15.RES, 'step16_per_condition.csv')
REC = os.path.join(s15.RES, 'step16_analysis.json')
CEIL = os.path.join(s15.RES, 'step16_diag_mra_ceiling.json')
OUT = os.path.join(s15.RES, 'step16_diag_mra_dose.json')
OUT_CSV = os.path.join(s15.RES, 'step16_diag_mra_dose_long.csv')
SUBS = ('ridge_S7', 'ridge_train', 'base', 'recipe', 'guidance_w2', 'attn_cross', 'mlp_ctrl', 'targets', 'targets_shuf')
N_DOSES = 3
T0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=s16.N_BOOT)
    a = ap.parse_args()
    n_boot = int(a.n_boot)
    for q in (OUT, OUT_CSV):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    tracked = '%s %s %s diag_step16_mra_dose.py diag_step16_mra_ceiling.py score_step16_tahoe.py' % (PC, REC, CEIL)
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: record or scripts not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0:
        sys.exit('FATAL: record or scripts uncommitted')
    rec = json.load(open(REC))
    if not rec.get('final'):
        sys.exit('FATAL: %s is not the final record' % REC)
    ceil = json.load(open(CEIL))
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    pc_all = pd.read_csv(PC)
    print('[inputs] %s (%d rows), record commit %s, section 9 record commit %s, this commit %s, %d bootstrap replicates'
          % (PC, len(pc_all), str(rec.get('commit'))[:7], str(ceil.get('commit'))[:7], commit[:7], n_boot))
    if n_boot != s16.N_BOOT:
        print('[WARNING] n_boot %d differs from the pre-registered %d: synthetic test only' % (n_boot, s16.N_BOOT))

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
        dose_levels = np.array(sorted(np.unique(doses)))
        if len(dose_levels) != N_DOSES:
            sys.exit('FATAL: %d dose levels, expected %d' % (len(dose_levels), N_DOSES))
        labels = ['%g uM' % (d / 1000.0) for d in dose_levels]
        top = dose_levels[-1]
        for d in dose_levels:
            k = len(set(ctx_id[doses == d]))
            if k != n_ctx // N_DOSES:
                sys.exit('FATAL: dose %g nM has %d contexts, expected %d' % (d, k, n_ctx // N_DOSES))
        # replicate partner: the same (line, drug, dose) on the other of plates 6 / 14 (as diag_step16_mra_ceiling.py)
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
        elig = [c for c in range(n_ctx) if (ctx_id[rep] == c).sum() >= s16.MIN_CTX]
        if len(elig) != int(ceil['per_split'][split]['n_contexts_restricted']):
            sys.exit('FATAL: %d eligible replicate contexts, section 9 says %d' % (len(elig), ceil['per_split'][split]['n_contexts_restricted']))
        if any(doses[ctx_id == c][0] != top for c in elig):
            sys.exit('FATAL: an eligible replicate context is not at the top dose')
        preds = {nm: pc['%s_effect_norm' % nm].astype(float).values for nm in SUBS}
        full = np.arange(n)

        def subsets(sel):
            """The named selections of one draw: each dose over all its conditions; the top dose split by plate copy."""
            out = {}
            for d, lab in zip(dose_levels, labels):
                out[lab] = sel[doses[sel] == d]
            st = sel[doses[sel] == top]
            out['top, plates 6/14'] = st[partner[st] >= 0]
            out['top, other plates'] = st[partner[st] < 0]
            return out

        def values(sel):
            """MRA of every submission on every named selection, by the scorer's own functions."""
            got = {}
            for name, s_ in subsets(sel).items():
                pre = s16.prep_ctx(true, s16.ctx_split(ctx_id, n_ctx, s_))
                got[name] = {nm: s16.mra_pre(p, pre) for nm, p in preds.items()}
                got[name]['n_contexts'] = int(sum(item is not None for item in pre))
            return got

        V = values(full)
        names = list(V)
        # --- guards against the committed records
        fsm = ceil['per_split'][split]['full_seed_mean']
        for nm in SUBS:
            m3 = float(np.mean([V[lab][nm] for lab in labels]))
            if abs(m3 - float(fsm[nm])) > 1e-9:
                sys.exit('FATAL: %s per-dose mean %.9f does not reproduce the full-set seed-mean MRA %.9f' % (nm, m3, float(fsm[nm])))
            if abs(V['top, plates 6/14'][nm] - float(ceil['per_split'][split]['restricted'][nm])) > 1e-9:
                sys.exit('FATAL: %s on plates 6/14 %.9f does not reproduce section 9 %.9f' % (nm, V['top, plates 6/14'][nm], float(ceil['per_split'][split]['restricted'][nm])))
        sizes = {name: int(len(s_)) for name, s_ in subsets(full).items()}
        print('\n== %s: %d conditions, %d contexts; dose levels %s nM; conditions per selection %s'
              % (split, n, n_ctx, ', '.join('%g' % d for d in dose_levels), ', '.join('%s %d (%d contexts)' % (k, sizes[k], V[k]['n_contexts']) for k in names)))
        print('   per-dose means reproduce the full-set seed-mean MRA of section 9; plates 6/14 values reproduce section 9 (both to 1e-9)')
        print('   %-13s %s' % ('submission', ' '.join('%18s' % k for k in names)))
        for nm in SUBS:
            print('   %-13s %s' % (nm, ' '.join('%+18.4f' % V[k][nm] for k in names)))

        # --- bootstrap over drugs, the same draws as diag_step16_mra_ceiling.py (RandomState(0) per split)
        rng = np.random.RandomState(0)
        groups = [np.where(drugs == u)[0] for u in np.unique(drugs)]
        B = {k: {nm: np.full(n_boot, np.nan) for nm in SUBS} for k in names}
        for b in range(n_boot):
            sel = np.concatenate([groups[g] for g in rng.randint(0, len(groups), len(groups))])
            got = values(sel)
            for k in names:
                for nm in SUBS:
                    B[k][nm][b] = got[k][nm]

        def con(name, point, dist):
            lo, hi = float(np.nanpercentile(dist, s16.LO)), float(np.nanpercentile(dist, s16.HI))
            c = dict(diff=float(point), ci=[lo, hi], verdict=s15.verdict(lo, hi), n_nan=int(np.isnan(dist).sum()))
            print('   %-44s %s' % (name, s15.fmt(c)))
            rows.append(dict(split=split, quantity=name, value=float(point), ci_lo=lo, ci_hi=hi, verdict=c['verdict']))
            return c

        print('   intervals (%d replicates):' % n_boot)
        contrasts = {}
        for k in names:
            for nm in SUBS:
                contrasts['%s | %s - 0' % (k, nm)] = con('%s | %s - 0' % (k, nm), V[k][nm], B[k][nm])
        for lab in labels[:-1]:
            key = 'recipe | %s - %s' % (labels[-1], lab)
            contrasts[key] = con(key, V[labels[-1]]['recipe'] - V[lab]['recipe'], B[labels[-1]]['recipe'] - B[lab]['recipe'])
        key = 'recipe | top, other plates - top, plates 6/14'
        contrasts[key] = con(key, V['top, other plates']['recipe'] - V['top, plates 6/14']['recipe'],
                             B['top, other plates']['recipe'] - B['top, plates 6/14']['recipe'])
        # the plate 6/14 interval of the recipe should coincide with section 9's (same draws, same statistic)
        s9 = ceil['per_split'][split]['contrasts']['recipe_minus_0']['ci']
        here = contrasts['top, plates 6/14 | recipe - 0']['ci']
        print('   recipe on plates 6/14: interval here [%+.5f, %+.5f], section 9 [%+.5f, %+.5f] (max difference %.2e%s)'
              % (here[0], here[1], s9[0], s9[1], max(abs(here[0] - s9[0]), abs(here[1] - s9[1])),
                 '' if n_boot == s16.N_BOOT else '; not comparable at this n_boot'))
        results[split] = dict(n_conditions=n, n_contexts=n_ctx, dose_levels_nM=[float(d) for d in dose_levels], selections=names,
                              n_conditions_per_selection=sizes, n_contexts_per_selection={k: V[k]['n_contexts'] for k in names},
                              values={k: {nm: float(V[k][nm]) for nm in SUBS} for k in names}, contrasts=contrasts)
        for k in names:
            for nm in SUBS:
                rows.append(dict(split=split, quantity='value | %s | %s' % (k, nm), value=float(V[k][nm]), ci_lo=None, ci_hi=None, verdict=None))
    json.dump(dict(note='post-hoc exploratory diagnostic of 2026-10-09 (second); no endpoint depends on it', record=REC,
                   record_commit=rec.get('commit'), record_sha256=s15.sha(REC), per_condition_sha256=s15.sha(PC),
                   section9_record=CEIL, section9_sha256=s15.sha(CEIL), commit=commit, n_boot=n_boot, min_context=s16.MIN_CTX,
                   per_split=results, seconds=round(time.time() - T0)), open(OUT, 'w'), indent=1)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'quantity', 'value', 'ci_lo', 'ci_hi', 'verdict'])
        w.writeheader(); w.writerows(rows)
    print('\nwrote %s and %s (%.0f s)' % (OUT, OUT_CSV, time.time() - T0))


if __name__ == '__main__':
    main()
