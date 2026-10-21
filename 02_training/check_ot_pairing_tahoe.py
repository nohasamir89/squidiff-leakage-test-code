#!/usr/bin/env python
"""Code-and-data check before roadmap item 12 (OT-paired training pairs), 2026-10-09. CPU only; no model is touched.

What the recipe does now (prep_step15_tahoe.py stage 6, scrna_datasets.AnnDataDataset): every treated training cell is
paired BY ROW with one vehicle cell of the same line and plate, drawn at random with replacement; the paired cell is
the model's `control_feature` (the conditioning input of Change B). Random pairing makes the control cell independent
of the treated cell given the condition, so the only information it carries is the condition's line and plate.

What item 12 would change: the same control cells, re-ordered within each condition so that each treated cell is
paired with the vehicle cell closest to it - the exact minimum-cost perfect matching (discrete optimal transport
between two equal-weight empirical measures), Euclidean distance on the 200-gene log-normalised panel, the model's
own input space. Nothing else changes: same treated cells, same multiset of control cells, same marginal distribution
of the conditioning input, same test files, same sampler (the harness draws random vehicle cells at sampling, as the
training marginal). The training script takes the control file as --control_data_path, so the arm needs no code
change, only a re-ordered control file.

What this script measures, per split and per training condition (Group), from the certified h5ad files:
  cost_random    mean distance between a treated cell and its current (random) paired vehicle
  cost_ot        the same after the optimal matching (scipy linear_sum_assignment; exact)
  cost_nearest   mean distance to the nearest vehicle cell (not a matching; a lower bound on cost_ot)
  spread_T       mean distance between a treated cell and the condition's treated mean (what remains if the
                 control cell told the model everything about the condition mean and nothing about the cell)
  dist_cmean     mean distance between a treated cell and the paired-control mean (the metrics' origin)
  cos_random     mean cosine between the centred treated cell and its centred paired control (random pairing:
                 about 0 by construction)
  cos_ot         the same under the optimal matching - how much per-cell state the matched vehicle shares with
                 the treated cell. This is the quantity that decides whether item 12 can matter: the model can only
                 use what the matched control tells it about the cell beyond the condition mean.
Reading guide, written before the run (a check, not a pre-registered endpoint): cos_ot below about 0.2 on most
conditions means the matched vehicle carries little per-cell information and item 12 is not worth 18 GPU runs;
above about 0.5 means the pairing carries real cell-state information and the arm should be pre-registered.
--write also writes the re-ordered control files (<train>_control_ot.h5ad, rows permuted within each Group; the
sorted row names and the matrix multiset are asserted identical to the original) and pins their sha256 in the record.

Usage (repo root, conda env active):  python -u check_ot_pairing_tahoe.py [--write] [--data DIR]
Writes tahoe_results/ot_pairing_check.json and tahoe_results/ot_pairing_check_per_condition.csv. Never overwrites.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

RES = 'tahoe_results'
RECORD = os.path.join(RES, 'step15_data_record.json')
OUT = os.path.join(RES, 'ot_pairing_check.json')
OUT_CSV = os.path.join(RES, 'ot_pairing_check_per_condition.csv')
SPLITS = ('drug_split_1', 'drug_split_0', 'drug_split_2')
T0 = time.time()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def dense(X):
    return np.asarray(X.todense() if hasattr(X, 'todense') else X, dtype=np.float32)


def cosines(A, B):
    """Row-wise cosine between two matrices after centring each by its own column means."""
    Ac, Bc = A - A.mean(axis=0, keepdims=True), B - B.mean(axis=0, keepdims=True)
    num = (Ac * Bc).sum(axis=1)
    den = np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1)
    ok = den > 0
    return float(np.mean(num[ok] / den[ok])) if ok.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='also write the OT-paired control files')
    ap.add_argument('--data', default=None, help='override the data directory recorded in the data record (synthetic tests only)')
    a = ap.parse_args()
    for q in (OUT, OUT_CSV):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if subprocess.run('git ls-files --error-unmatch check_ot_pairing_tahoe.py %s' % RECORD, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: script or data record not committed')
    if subprocess.run('git diff --quiet -- check_ot_pairing_tahoe.py %s' % RECORD, shell=True).returncode != 0:
        sys.exit('FATAL: script or data record modified')
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    rec = json.load(open(RECORD))
    print('[inputs] %s (prep commit %s), this commit %s' % (RECORD, str(rec.get('commit'))[:7], commit[:7]))

    results, rows, written = {}, [], {}
    for split in SPLITS:
        s = split.split('_')[-1]
        files = rec['splits'][s]['files']
        ft, fc = files['train']['path'], files['train_control']['path']
        if a.data:
            ft, fc = os.path.join(a.data, os.path.basename(ft)), os.path.join(a.data, os.path.basename(fc))
        for key, path in (('train', ft), ('train_control', fc)):
            if not os.path.isfile(path):
                sys.exit('FATAL: missing %s' % path)
            got = sha256(path)
            if got != files[key]['sha256']:
                sys.exit('FATAL: %s sha256 %s differs from the data record %s' % (path, got[:16], files[key]['sha256'][:16]))
        T = ad.read_h5ad(ft); C = ad.read_h5ad(fc)
        if T.n_obs != C.n_obs or T.n_vars != C.n_vars:
            sys.exit('FATAL: %s and its control file differ in shape' % split)
        for col in ('Group', 'cell_name', 'dose', 'SMILES'):
            if not np.array_equal(T.obs[col].astype(str).values, C.obs[col].astype(str).values):
                sys.exit('FATAL: %s: column %s differs between treated and control rows (the 1:1 row pairing does not hold)' % (split, col))
        if 'plate' in T.obs and 'plate' in C.obs and not np.array_equal(T.obs['plate'].values, C.obs['plate'].values):
            sys.exit('FATAL: %s: plate differs between treated and control rows' % split)
        X, Xc = dense(T.X), dense(C.X)
        groups = T.obs['Group'].astype(str).values
        codes, uniq = pd.factorize(pd.Series(groups), sort=False)
        order = np.argsort(codes, kind='stable')
        bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        print('[%s] %d rows, %d conditions, %d genes, hashes match the data record (%.0f s)' % (split, T.n_obs, len(uniq), T.n_vars, time.time() - T0))
        perm = np.arange(T.n_obs)
        rng = np.random.RandomState(0)
        agg = dict(cost_random=[], cost_ot=[], cost_nearest=[], cost_reshuffled=[], spread_T=[], spread_C=[], dist_cmean=[], cos_random=[], cos_ot=[], n=[])
        for k in range(len(uniq)):
            idx = order[bounds[k]:bounds[k + 1]]
            n = len(idx)
            Tk, Ck = X[idx].astype(np.float64), Xc[idx].astype(np.float64)
            D = cdist(Tk, Ck)
            r_, c_ = linear_sum_assignment(D)
            if not np.array_equal(r_, np.arange(n)):
                sys.exit('FATAL: assignment rows are not in order')
            perm[idx] = idx[c_]                                        # treated row idx[i] <- control row idx[c_[i]]
            sh = rng.permutation(n)
            vals = dict(cost_random=float(np.mean(np.diag(D))), cost_ot=float(D[r_, c_].mean()), cost_nearest=float(D.min(axis=1).mean()),
                        cost_reshuffled=float(D[np.arange(n), sh].mean()),
                        spread_T=float(np.linalg.norm(Tk - Tk.mean(axis=0), axis=1).mean()), spread_C=float(np.linalg.norm(Ck - Ck.mean(axis=0), axis=1).mean()),
                        dist_cmean=float(np.linalg.norm(Tk - Ck.mean(axis=0), axis=1).mean()),
                        cos_random=cosines(Tk, Ck), cos_ot=cosines(Tk, Ck[c_]), n=int(n))
            for key, v in vals.items():
                agg[key].append(v)
            rows.append(dict(split=split, group=str(uniq[k]), **vals))
        A = {key: np.array(v, dtype=np.float64) for key, v in agg.items()}
        red = (A['cost_random'] - A['cost_ot']) / A['cost_random']
        summary = dict(n_conditions=int(len(uniq)), n_rows=int(T.n_obs), cells_per_condition=[int(A['n'].min()), int(np.median(A['n'])), int(A['n'].max())],
                       mean={key: float(np.nanmean(A[key])) for key in A if key != 'n'}, median={key: float(np.nanmedian(A[key])) for key in A if key != 'n'},
                       reduction_random_to_ot=dict(mean=float(red.mean()), median=float(np.median(red)), q05=float(np.percentile(red, 5)), q95=float(np.percentile(red, 95)),
                                                   frac_above_10pct=float((red > 0.10).mean()), frac_above_25pct=float((red > 0.25).mean())),
                       cos_ot_quantiles=[float(x) for x in np.nanpercentile(A['cos_ot'], [5, 25, 50, 75, 95])],
                       frac_cos_ot_above_0_2=float(np.nanmean(A['cos_ot'] > 0.2)), frac_cos_ot_above_0_5=float(np.nanmean(A['cos_ot'] > 0.5)))
        print('   cells per condition min/median/max %s' % summary['cells_per_condition'])
        print('   mean over conditions: pair distance random %.4f | reshuffled %.4f | OT %.4f | nearest %.4f || treated spread %.4f | control spread %.4f | treated to control mean %.4f'
              % (summary['mean']['cost_random'], summary['mean']['cost_reshuffled'], summary['mean']['cost_ot'], summary['mean']['cost_nearest'],
                 summary['mean']['spread_T'], summary['mean']['spread_C'], summary['mean']['dist_cmean']))
        print('   reduction of the pair distance by OT: mean %.3f, median %.3f, 5-95%% [%.3f, %.3f]; above 10%%: %.1f%% of conditions, above 25%%: %.1f%%'
              % (red.mean(), np.median(red), np.percentile(red, 5), np.percentile(red, 95), 100 * (red > 0.10).mean(), 100 * (red > 0.25).mean()))
        print('   centred cosine treated vs paired control: random %.4f | OT %.4f (quantiles 5/25/50/75/95: %s); cos_ot > 0.2 in %.1f%% of conditions, > 0.5 in %.1f%%'
              % (summary['mean']['cos_random'], summary['mean']['cos_ot'], [round(x, 3) for x in summary['cos_ot_quantiles']],
                 100 * summary['frac_cos_ot_above_0_2'], 100 * summary['frac_cos_ot_above_0_5']))
        results[split] = dict(train=dict(path=ft, sha256=files['train']['sha256']), train_control=dict(path=fc, sha256=files['train_control']['sha256']), **summary)
        if a.write:
            out_path = fc.replace('_control.h5ad', '_control_ot.h5ad')
            if out_path == fc or os.path.exists(out_path):
                sys.exit('FATAL: cannot write %s' % out_path)
            Cot = C[perm].copy()
            if not np.array_equal(np.sort(Cot.obs_names.values), np.sort(C.obs_names.values)):
                sys.exit('FATAL: the re-ordered control file does not hold the same rows')
            if abs(float(dense(Cot.X).sum()) - float(Xc.sum())) > 1e-3 * max(1.0, abs(float(Xc.sum()))):
                sys.exit('FATAL: the re-ordered control matrix differs in total')
            for col in ('Group', 'cell_name'):
                if not np.array_equal(Cot.obs[col].astype(str).values, T.obs[col].astype(str).values):
                    sys.exit('FATAL: re-ordered control rows do not align with the treated rows on %s' % col)
            Cot.write_h5ad(out_path)
            written[split] = dict(path=out_path, sha256=sha256(out_path), rows=int(Cot.n_obs), source_control_sha256=files['train_control']['sha256'])
            print('   wrote %s (%s)' % (out_path, written[split]['sha256'][:16]))
        del T, C, X, Xc
    json.dump(dict(note='check before roadmap item 12 (OT-paired training pairs); no model touched', record=RECORD, record_sha256=sha256(RECORD),
                   commit=commit, per_split=results, written=written, seconds=round(time.time() - T0)), open(OUT, 'w'), indent=1)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print('\nwrote %s and %s (%.0f s)' % (OUT, OUT_CSV, time.time() - T0))


if __name__ == '__main__':
    main()
