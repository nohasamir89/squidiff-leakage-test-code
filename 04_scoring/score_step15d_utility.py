#!/usr/bin/env python
"""Step 15d, layer A: utility of the frozen Tahoe predictions against PRISM sensitivity labels.

Pre-registration: PREREG_step15d_tahoe_validation.md (sections 3, 4A, 5, 6, 9), committed before this file. This
script, committed before any result exists, does only what that document says:

  1. guards: the pre-registration and this scorer committed and clean (also score_step15_tahoe.py and
     score_step15c_tahoe.py, which it imports, and the Step 15b per-condition record); the Step 15b arrays hash-checked
     against provenance/s15ev_*.json; the four shift baselines rebuilt by the Step 15b code path and asserted to
     reproduce the Step 15b per-condition record to 1e-9; the PRISM files checked against the figshare record; exactly
     three dose levels in the test file, the highest (5 uM) used; no held-out drug among the training pairs;
  2. labels: for a PRISM line and a Tahoe drug, the median AUC over every secondary-screen row of the line's DepMap id
     whose broad_id is among the drug's matched ids (inventory (d), reused verbatim) and that passed STR profiling;
  3. effects at 5 uM per (line, drug), averaged over plates: predicted (B - C per seed for the arms; the baseline deltas)
     and measured (true - C); training effects from the training file (mean treated - mean paired control);
  4. the predictor f: ridge from the standardised 200-gene effect to AUC, fitted on the training pairs' MEASURED
     effects (alpha from {0.1, 1, 10, 100, 1000} by 5-fold GroupKFold over training drugs, lowest MSE); QSAR: the same
     ridge procedure from the Step 15b compound features (fingerprint, line one-hot, interactions) to AUC;
  5. U(s) = mean over lines (>= 8 labelled held-out pairs) of the Spearman correlation between f(predicted effect) and
     AUC over the labelled held-out drugs; a submission with identical predictions within a line (control-paste,
     mean-response) gets rho = 0 by definition (asserted constant first); the ceiling = measured effects through f;
     effect size = the negative L2 norm of the recipe's predicted effect (no fitting); arms per seed, then averaged;
  6. the section-5 contrasts with a drug-clustered percentile bootstrap (5,000 replicates, RandomState(0), 97.5 %;
     the same resampled drug set for both members of a contrast; lines with < 8 pairs skipped in a replicate);
     the within-drug across-line read (drugs with >= 6 labelled lines); section 6 when all three splits are present
     with the precedence 1 -> 2 -> 5 -> 3 -> 4 (3 = primary below zero on a split; 4 = primary above zero on one or
     two splits and including zero on the rest, as coded for Step 15b).

  7. exploratory additions (section 5, dated paragraph of 2026-10-05, after the development run on drug_split_1 and
     before any final run; not gated, reported next to the pre-registered quantities): (a) a "highest available dose
     per drug" variant - each drug enters at the highest dose level at which it has conditions in the PRISM lines, on
     both the training and the held-out side, because 10 of the 27 PRISM-labelled held-out drugs of split 1 have no
     5 uM condition at all; (b) f refitted with the alpha grid extended to {1e4, 1e5} on the pre-registered pairs
     (the development run chose the grid's upper bound); (c) the CV R^2 of f and QSAR on the training pairs; (d) the
     number of labelled held-out pairs lost by the 5 uM rule and each held-out drug's available dose levels.

Usage: python -u score_step15d_utility.py [--splits drug_split_1]     (development run first, then no --splits = final)
Outputs: tahoe_results/step15d_utility[_development].json, step15d_utility_long[_development].csv,
         step15d_utility_pairs[_development].csv. Never overwrites.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402  Step 15b: loaders, features, baselines
import score_step15c_tahoe as s15c                    # noqa: E402  Step 15c: mean-vector loader, close()

PRE = 'PREREG_step15d_tahoe_validation.md'
PC15 = os.path.join(s15.RES, 'step15_model_per_condition.csv')
INV_D = os.path.join(s15.RES, 'step15_coverage_inventory_d.json')
PRISM_REC = os.path.join(s15.RES, 'external', 'prism_figshare_9393293_files.json')
PRISM_DIR = '/scratch/ni2071/Diffusion/tahoe/external/prism'
DRC = os.path.join(PRISM_DIR, 'secondary-screen-dose-response-curve-parameters.csv')
CLI = os.path.join(PRISM_DIR, 'secondary-screen-cell-line-info.csv')
LINES = ['A-172', 'COLO 205', 'H4', 'HCT15', 'HT-29', 'KATO III', 'LS 180', 'LoVo', 'RKO', 'SNU-1', 'SW 1088', 'SW1417', 'SW48', 'SW480']
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
ALPHAS_EXT = ALPHAS + (1e4, 1e5)                                       # exploratory (b)
EXPLORE_SUBS = ('recipe', 'ridge_train', 'ceiling', 'qsar', 'effect_size', 'effect_size_measured')
MIN_PAIRS_LINE, MIN_LINES_DRUG, N_BOOT = 8, 6, 5000
LO, HI = 1.25, 98.75
SUBS = list(s15.BASELINES) + list(s15.ARMS)
EXTRA = ['ceiling', 'qsar', 'effect_size']
T0 = time.time()


def md5(p):
    m = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            m.update(c)
    return m.hexdigest()


def norm_line(s):
    return re.sub(r'[^A-Za-z0-9]', '', str(s)).upper()


def col(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    sys.exit('FATAL: none of %s among the columns %s' % (names, list(df.columns)))


def cond_name(c, line):
    """'<cell_name>_<drug>_<dose>_p<plate>' -> drug name as written in the file (names carry no underscore)."""
    if not c.startswith(line + '_'):
        sys.exit('FATAL: condition %r does not start with its line %r' % (c, line))
    return c[len(line) + 1:].rsplit('_', 2)[0]


def fit_ridge(X, y, groups, tag, alphas=ALPHAS):
    """Ridge with alpha chosen by 5-fold GroupKFold (lowest MSE); returns the fitted model, the alpha, the CV MSE and the
    CV R^2 (1 - CV MSE / variance of y)."""
    best = None
    for a in alphas:
        err = []
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=groups):
            if set(groups[tr]) & set(groups[te]):
                sys.exit('FATAL: %s CV fold shares a drug between fit and prediction' % tag)
            p = Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[te])
            err.append(float(np.mean((p - y[te]) ** 2)))
        m = float(np.mean(err))
        if best is None or m < best[1] - 1e-12:
            best = (a, m)
    r2 = 1.0 - best[1] / float(np.var(y))
    print('[%s] alpha %g (CV MSE %.5f, label variance %.5f, CV R^2 %+.3f) on %d training pairs x %d features' % (tag, best[0], best[1], float(np.var(y)), r2, X.shape[0], X.shape[1]))
    return Ridge(alpha=best[0]).fit(X, y), best[0], best[1], r2


def spearman0(x, y):
    """Spearman rho; 0 when x (or y) is constant (no ranking), as section 4A defines for the floors."""
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    r = spearmanr(x, y)[0]
    return float(r) if np.isfinite(r) else 0.0


def U_of(vals, labels, line_ix, n_lines, sel):
    """Mean per-line Spearman over the pair subset sel (with multiplicity); lines with < MIN_PAIRS_LINE pairs skipped."""
    rhos = []
    for l in range(n_lines):
        m = sel[line_ix[sel] == l]
        if len(m) >= MIN_PAIRS_LINE:
            rhos.append(spearman0(vals[m], labels[m]))
    return float(np.mean(rhos)) if rhos else np.nan, len(rhos)


def W_of(vals, labels, drug_ix, drugs_sel):
    """Within-drug across-line Spearman, mean over the selected drugs with >= MIN_LINES_DRUG labelled lines."""
    rhos = []
    for d in drugs_sel:
        m = np.where(drug_ix == d)[0]
        if len(m) >= MIN_LINES_DRUG:
            rhos.append(spearman0(vals[m], labels[m]))
    return float(np.mean(rhos)) if rhos else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default=','.join(s15.ALL_SPLITS))
    a = ap.parse_args()
    splits = [s.strip() for s in a.splits.split(',') if s.strip()]
    for s in splits:
        if s not in s15.ALL_SPLITS:
            sys.exit('FATAL: unknown split %s' % s)
    final = sorted(splits) == sorted(s15.ALL_SPLITS)
    suffix = '' if final else '_development'
    OUT = os.path.join(s15.RES, 'step15d_utility%s.json' % suffix)
    OUT_CSV = os.path.join(s15.RES, 'step15d_utility_long%s.csv' % suffix)
    OUT_PAIRS = os.path.join(s15.RES, 'step15d_utility_pairs%s.csv' % suffix)
    for q in (OUT, OUT_CSV, OUT_PAIRS):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 9:
        sys.exit('FATAL: prereg incomplete')
    tracked = '%s score_step15d_utility.py score_step15_tahoe.py score_step15c_tahoe.py %s %s' % (PRE, PC15, INV_D)
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: prereg, scorer, imported scorers or records not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0:
        sys.exit('FATAL: prereg, scorer, imported scorers or records uncommitted')
    for q in (DRC, CLI, PRISM_REC, s15.RECORD):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    record = json.load(open(s15.RECORD))
    name_map = record.get('name_map', {})
    inv_map = {v: k for k, v in name_map.items()}                       # file name -> usable (inventory) name
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    print('[prereg] committed; scorer committed; commit %s; splits %s (%s)' % (commit[:7], splits, 'final' if final else 'development'))

    # ------------------------------------------------------------------ PRISM: files, lines, labels
    prism_rec = json.load(open(PRISM_REC))
    entries = prism_rec['files'] if isinstance(prism_rec, dict) and 'files' in prism_rec else prism_rec
    hashes = {}
    for path in (DRC, CLI):
        h = md5(path)
        ent = next((e for e in entries if isinstance(e, dict) and str(e.get('name', '')) == os.path.basename(path)), None)
        exp = next((ent[k] for k in ('computed_md5', 'md5', 'supplied_md5') if ent and ent.get(k)), None)
        if exp is None:
            print('[prism] WARNING: no md5 for %s in the figshare record; md5 recorded, not checked' % os.path.basename(path))
        elif exp != h:
            sys.exit('FATAL: %s md5 %s differs from the figshare record %s' % (path, h, exp))
        hashes[os.path.basename(path)] = h
    print('[prism] files checked: %s' % hashes)
    cli = pd.read_csv(CLI)
    drc = pd.read_csv(DRC, low_memory=False)
    d_dep, d_auc, d_broad, d_pass = col(drc, 'depmap_id'), col(drc, 'auc'), col(drc, 'broad_id'), col(drc, 'passed_str_profiling')
    c_ccle, c_dep = col(cli, 'ccle_name'), col(cli, 'depmap_id')
    prefix = {str(x).split('_')[0].upper(): str(y) for x, y in zip(cli[c_ccle], cli[c_dep])}
    dep_of_line = {ln: prefix[norm_line(ln)] for ln in LINES if norm_line(ln) in prefix}
    if len(dep_of_line) != 10:
        sys.exit('FATAL: %d Tahoe lines matched in PRISM, inventory (d) found 10' % len(dep_of_line))
    match = json.load(open(INV_D))['match']
    passed = drc[drc[d_pass].astype(str).str.lower().isin(['true', '1', '1.0']) & drc[d_auc].notna()]
    passed = passed[passed[d_dep].isin(dep_of_line.values())]
    bid_rows = {b: g for b, g in passed.groupby(passed[d_broad].astype(str))}
    label = {}                                                            # (line, usable drug) -> median AUC
    for ln, dep in dep_of_line.items():
        for d, mt in match.items():
            vals = []
            for b in mt['broad_ids']:
                g = bid_rows.get(b)
                if g is not None:
                    vals += list(g.loc[g[d_dep] == dep, d_auc].astype(float).values)
            if vals:
                label[(ln, d)] = float(np.median(vals))
    print('[prism] %d lines | %d matched drugs | %d (line, drug) labels (median AUC over %d rows that passed STR profiling)'
          % (len(dep_of_line), sum(1 for m in match.values() if m['broad_ids']), len(label), len(passed)))

    pc15 = pd.read_csv(PC15)
    results, inputs, long_rows, pair_rows = {}, {}, [], []
    rng = np.random.RandomState(0)
    for split in splits:
        print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
        k = split.split('_')[-1]
        rec_split = record['splits'][k]
        n_drugs = len(rec_split['held_structures'])

        # --- certified mean vectors (identical cond / C / true across the 15 runs)
        arms = {arm: [s15c.load_vectors(split, arm, s) for s in range(s15.N_SEEDS)] for arm in s15.ARMS}
        ref = arms['recipe'][0]
        cond, n = ref['cond'], len(ref['cond'])
        if n != int(rec_split['n_test_conditions']):
            sys.exit('FATAL: %d conditions, record expects %d' % (n, rec_split['n_test_conditions']))
        for arm, L in arms.items():
            for v in L:
                if not np.array_equal(v['cond'], cond) or np.abs(v['C'] - ref['C']).max() > 1e-5 or np.abs(v['true'] - ref['true']).max() > 1e-5:
                    sys.exit('FATAL: condition order, control means or true means differ in %s' % v['tag'])
        inputs[split] = {arm: [v['rec'] for v in L] for arm, L in arms.items()}
        true_mean, C = ref['true'], ref['C']
        Y = true_mean - C
        pc = pc15[pc15['split'] == split].reset_index(drop=True)
        if len(pc) != n or not np.array_equal(pc['condition'].astype(str).values, cond):
            sys.exit('FATAL: the Step 15b per-condition record does not list the conditions of %s in the vectors order' % split)
        lines = pc['line'].astype(str).values
        drugs = pc['structure'].astype(str).values
        doses = pc['dose_nM'].astype(float).values
        lv = sorted(set(lines))
        if len(np.unique(drugs)) != n_drugs:
            sys.exit('FATAL: %d structures in the record, expected %d' % (len(np.unique(drugs)), n_drugs))
        levels = sorted(set(np.round(doses, 6)))
        if len(levels) != 3:
            sys.exit('FATAL: %d dose levels in the test conditions, expected 3' % len(levels))
        top = levels[-1]
        names = np.array([inv_map.get(cond_name(c, lines[i]), cond_name(c, lines[i])) for i, c in enumerate(cond)])
        for nm in set(names):
            if nm not in match:
                sys.exit('FATAL: drug %r of the test file is not in the inventory (d) match table' % nm)
        print('[design] %d conditions | %d held-out structures | dose levels %s -> using %g' % (n, n_drugs, [float(x) for x in levels], top))

        # --- shift baselines by the Step 15b code path, with the training-condition effects kept for f
        FP = np.array([s15.Drug_dose_encoder([drugs[i]], [doses[i]], comb_num=1)[0] for i in range(n)], dtype=np.float64)
        X = s15.features(FP, lines, lv)
        P7 = s15.ridge_s7(X, Y, drugs)
        tr = ad.read_h5ad('%s/tahoe_train_%s.h5ad' % (s15.DATA, split))
        trc = ad.read_h5ad('%s/tahoe_train_%s_control.h5ad' % (s15.DATA, split))
        if tr.n_obs != trc.n_obs or not np.array_equal(tr.obs['Group'].astype(str).values, trc.obs['Group'].astype(str).values):
            sys.exit('FATAL: train and control files are not row-aligned')
        gtr, smi_tr, dose_tr, line_tr = (tr.obs['Group'].astype(str).values, tr.obs['SMILES'].astype(str).values,
                                         tr.obs['dose'].astype(float).values, tr.obs['cell_name'].astype(str).values)
        if set(smi_tr) & set(drugs):
            sys.exit('FATAL: a held-out structure appears in the training file')
        codes, utr = pd.factorize(pd.Series(gtr), sort=False)
        Xtr, Xtrc = s15.dense(tr.X), s15.dense(trc.X)
        order = np.argsort(codes, kind='stable')
        bounds = np.searchsorted(codes[order], np.arange(len(utr)))
        cnt = np.bincount(codes, minlength=len(utr)).astype(np.float64)
        Ytr = np.add.reduceat((Xtr[order].astype(np.float64) - Xtrc[order].astype(np.float64)), bounds, axis=0) / cnt[:, None]
        first = order[bounds]
        dose_sorted = dose_tr[order]
        ends = np.append(bounds[1:], len(order))
        dose_med = np.array([float(np.median(dose_sorted[bounds[q]:ends[q]])) for q in range(len(utr))])
        smi_c, line_c = smi_tr[first], line_tr[first]
        FPtr = np.array([s15.Drug_dose_encoder([smi_c[q]], [dose_med[q]], comb_num=1)[0] for q in range(len(utr))], dtype=np.float64)
        Ptr = Ridge(alpha=s15.ALPHA).fit(s15.features(FPtr, line_c, lv), Ytr).predict(X)
        del tr, trc, Xtr, Xtrc
        mr = np.zeros_like(Y)
        for ln in lv:
            ixl = np.where(lines == ln)[0]
            mr[ixl] = Y[ixl].mean(axis=0)[None, :]
        deltas = {'ridge_S7': P7, 'ridge_train': Ptr, 'mean_response': mr, 'control_paste': np.zeros_like(Y)}
        for bname, D in deltas.items():
            j, d = s15.thesis_per_cond(true_mean, C + D, C)
            if not s15c.close(j, pc['%s_de_overlap' % bname].values) or not s15c.close(d, pc['%s_dpc50' % bname].values):
                sys.exit('FATAL: %s does not reproduce the Step 15b per-condition record' % bname)
        for arm, L in arms.items():
            jj, dd = zip(*[s15.thesis_per_cond(v['true'], v['B'], v['C']) for v in L])
            if not s15c.close(np.nanmean(np.vstack(jj), axis=0), pc['%s_de_overlap' % arm].values) or not s15c.close(np.nanmean(np.vstack(dd), axis=0), pc['%s_dpc50' % arm].values):
                sys.exit('FATAL: %s does not reproduce the Step 15b per-condition record' % arm)
        print('[reproduction] DE-overlap and DPC-50 of all nine submissions equal the Step 15b record to 1e-9')

        # --- training pairs at the top dose: measured effects, averaged over plates; PRISM lines with a label only
        tr_levels = sorted(set(np.round(dose_med, 6)))
        if tr_levels != levels:
            sys.exit('FATAL: training dose levels %s differ from the test levels %s' % (tr_levels, levels))
        tr_names = np.array([inv_map.get(cond_name(u, line_c[q]), cond_name(u, line_c[q])) for q, u in enumerate(utr)])
        tp = {}
        for q in range(len(utr)):
            if abs(dose_med[q] - top) > 1e-6 or line_c[q] not in dep_of_line:
                continue
            key = (line_c[q], tr_names[q])
            if key in label:
                tp.setdefault(key, []).append(q)
        tr_keys = sorted(tp)
        if set(nm for _, nm in tr_keys) & set(names):
            sys.exit('FATAL: a held-out drug name among the training pairs')
        Etr = np.array([Ytr[tp[key]].mean(axis=0) for key in tr_keys])
        ytr = np.array([label[key] for key in tr_keys])
        gtr_drug = np.array([nm for _, nm in tr_keys])
        mu, sd = Etr.mean(axis=0), Etr.std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        print('[training] %d training pairs (%d drugs, %d lines) with a label at the top dose; standardisation fixed on them'
              % (len(tr_keys), len(set(gtr_drug)), len(set(ln for ln, _ in tr_keys))))
        f, alpha_f, mse_f, r2_f = fit_ridge((Etr - mu) / sd, ytr, gtr_drug, 'f')
        FPq = np.array([s15.Drug_dose_encoder([smi_c[tp[key][0]]], [top], comb_num=1)[0] for key in tr_keys], dtype=np.float64)
        Xq = s15.features(FPq, np.array([ln for ln, _ in tr_keys]), lv)
        qsar, alpha_q, mse_q, r2_q = fit_ridge(Xq, ytr, gtr_drug, 'qsar')
        n_tr_unmatched = int(sum(1 for nm in set(tr_names) if nm not in match))
        print('[training] drug names of the training file not in the inventory (d) match table: %d' % n_tr_unmatched)

        # --- held-out pairs at the top dose: predicted and measured effects averaged over plates
        hp = {}
        for i in range(n):
            if abs(doses[i] - top) > 1e-6 or lines[i] not in dep_of_line:
                continue
            key = (lines[i], names[i])
            hp.setdefault(key, []).append(i)
        keys = [key for key in sorted(hp) if key in label]
        n_unlabelled = len(hp) - len(keys)
        line_ix = np.array([lv.index(ln) for ln, _ in keys])
        drug_names = np.array([nm for _, nm in keys])
        drug_ix = pd.factorize(pd.Series(drug_names), sort=True)[0]
        labels = np.array([label[key] for key in keys])
        n_lines_eff = len(set(line_ix))
        per_line_n = {lv[l]: int((line_ix == l).sum()) for l in sorted(set(line_ix))}
        print('[held-out] %d (line, drug) pairs at the top dose in PRISM lines, %d with a label (%d drugs), %d without | pairs per line %s'
              % (len(hp), len(keys), len(set(drug_names)), n_unlabelled, per_line_n))

        def pair_effects(E):
            return np.array([E[hp[key]].mean(axis=0) for key in keys])

        def through_f(E):
            return f.predict((pair_effects(E) - mu) / sd)

        vals, per_seed = {}, {}
        for bname, D in deltas.items():
            vals[bname] = through_f(D)
        for arm, L in arms.items():
            per_seed[arm] = [through_f(v['B'] - v['C']) for v in L]
            vals[arm] = np.mean(np.vstack(per_seed[arm]), axis=0)
        vals['ceiling'] = through_f(Y)
        FPh = np.array([s15.Drug_dose_encoder([drugs[hp[key][0]]], [top], comb_num=1)[0] for key in keys], dtype=np.float64)
        vals['qsar'] = qsar.predict(s15.features(FPh, np.array([ln for ln, _ in keys]), lv))
        es_seeds = [-np.linalg.norm(pair_effects(v['B'] - v['C']), axis=1) for v in arms['recipe']]
        vals['effect_size'] = np.mean(np.vstack(es_seeds), axis=0)
        vals['effect_size_measured'] = -np.linalg.norm(pair_effects(Y), axis=1)
        # floors must be constant within every line (section 4A): asserted, then rho = 0 by spearman0
        for bname in ('control_paste', 'mean_response'):
            for l in set(line_ix):
                v_ = vals[bname][line_ix == l]
                if np.ptp(v_) > 1e-9:
                    sys.exit('FATAL: %s predictions are not identical within a line' % bname)
        print('[checks] control-paste and mean-response are constant within every line (rho = 0 by definition)')

        # --- point estimates
        all_sel = np.arange(len(keys))
        drugs_all = np.unique(drug_ix)
        U, W = {}, {}
        for nm, v_ in vals.items():
            U[nm], n_l = U_of(v_, labels, line_ix, len(lv), all_sel)
            W[nm] = W_of(v_, labels, drug_ix, drugs_all)
            print('  U %-22s %+.4f over %d lines | within-drug %+.4f' % (nm, U[nm], n_l, W[nm]))
        U_seed = {arm: [U_of(v_, labels, line_ix, len(lv), all_sel)[0] for v_ in per_seed[arm]] for arm in s15.ARMS}
        for arm in s15.ARMS:
            print('  per seed %-18s %s' % (arm, ' / '.join('%+.4f' % u for u in U_seed[arm])))

        # --- bootstrap: resample held-out drugs; U per submission in every replicate; contrasts as paired differences
        groups = [np.where(drug_ix == d)[0] for d in drugs_all]
        names_boot = list(vals)
        B = {nm: np.full(N_BOOT, np.nan) for nm in names_boot}
        Wb = {nm: np.full(N_BOOT, np.nan) for nm in ('recipe', 'ridge_train', 'ceiling', 'qsar')}
        for b in range(N_BOOT):
            draw = rng.randint(0, len(groups), len(groups))
            sel = np.concatenate([groups[g] for g in draw])
            for nm in names_boot:
                B[nm][b] = U_of(vals[nm], labels, line_ix, len(lv), sel)[0]
            for nm in Wb:
                Wb[nm][b] = W_of(vals[nm], labels, drug_ix, drugs_all[draw])
        print('[bootstrap] %d replicates over %d held-out drugs (%d s)' % (N_BOOT, len(groups), time.time() - T0))

        def ci(arr):
            return [float(np.nanpercentile(arr, LO)), float(np.nanpercentile(arr, HI))]

        def con(name, b_name, a_name=None, point=None, boot_b=None, boot_a=None):
            if boot_b is None:
                boot_b, boot_a = B[b_name], (B[a_name] if a_name else 0.0)
            d = (U[b_name] - (U[a_name] if a_name else 0.0)) if point is None else point
            lo, hi = ci(boot_b - boot_a)
            c = dict(diff=float(d), ci=[lo, hi], verdict=s15.verdict(lo, hi), half_width=float((hi - lo) / 2))
            print('  %-52s %s' % (name, s15.fmt(c)))
            return c

        contrasts = {}
        contrasts['Q1_primary_U_recipe_minus_0'] = con('Q1 primary: U recipe - 0', 'recipe')
        contrasts['Q2_coprimary_U_recipe_minus_ridge_train'] = con('Q2 co-primary: U recipe - ridge-train', 'recipe', 'ridge_train')
        for nm in names_boot:
            if nm != 'recipe':
                contrasts['secondary_U_%s_minus_0' % nm] = con('secondary: U %s - 0' % nm, nm)
        contrasts['secondary_U_recipe_minus_ceiling'] = con('secondary: U recipe - ceiling', 'recipe', 'ceiling')
        contrasts['secondary_U_recipe_minus_qsar'] = con('secondary: U recipe - qsar', 'recipe', 'qsar')
        contrasts['secondary_U_recipe_minus_effect_size'] = con('secondary: U recipe - effect size', 'recipe', 'effect_size')
        contrasts['secondary_U_recipe_minus_ridge_S7'] = con('secondary: U recipe - ridge-S7', 'recipe', 'ridge_S7')
        contrasts['secondary_U_guidance_w2_minus_recipe'] = con('secondary: U guidance w=2 - recipe', 'guidance_w2', 'recipe')
        contrasts['exploratory_U_attn_cross_minus_mlp_ctrl'] = con('exploratory: U attn_cross - mlp_ctrl', 'attn_cross', 'mlp_ctrl')
        for nm in Wb:
            contrasts['secondary_withindrug_%s_minus_0' % nm] = con('secondary: within-drug %s - 0' % nm, nm, point=W[nm], boot_b=Wb[nm], boot_a=0.0)

        # ---------------------------------------------------------------- exploratory additions (section 5, dated 2026-10-05)
        print('[exploratory] section-5 additions of 2026-10-05 (not gated)')
        held_names = sorted(set(names))
        avail_h = {nm: sorted(set(float(np.round(doses[i], 6)) for i in range(n) if names[i] == nm and lines[i] in dep_of_line)) for nm in held_names}
        avail_t = {}
        for q in range(len(utr)):
            if line_c[q] in dep_of_line:
                avail_t.setdefault(tr_names[q], set()).add(float(np.round(dose_med[q], 6)))
        n_lab_held = sum(1 for (ln, d) in label if d in set(held_names))
        lost = n_lab_held - len(keys)
        no_top = sorted(nm for nm in held_names if avail_h[nm] and top not in avail_h[nm] and any((ln, nm) in label for ln in dep_of_line))
        print('  (d) labelled held-out pairs in PRISM lines %d | scored at the top dose %d | lost by the 5 uM rule %d | labelled held-out drugs without a top-dose condition %d: %s'
              % (n_lab_held, len(keys), lost, len(no_top), no_top))
        # (b) extended alpha grid on the pre-registered pairs
        f_ext, alpha_ext, mse_ext, r2_ext = fit_ridge((Etr - mu) / sd, ytr, gtr_drug, 'f, extended grid', alphas=ALPHAS_EXT)
        vals_ext = {}
        vals_ext['recipe'] = np.mean(np.vstack([f_ext.predict((pair_effects(v['B'] - v['C']) - mu) / sd) for v in arms['recipe']]), axis=0)
        vals_ext['ridge_train'] = f_ext.predict((pair_effects(deltas['ridge_train']) - mu) / sd)
        vals_ext['ceiling'] = f_ext.predict((pair_effects(Y) - mu) / sd)
        vals_ext['qsar'] = vals['qsar']
        vals_ext['effect_size'], vals_ext['effect_size_measured'] = vals['effect_size'], vals['effect_size_measured']
        # (a) highest available dose per drug, training and held-out alike
        hd_t = {nm: max(v_) for nm, v_ in avail_t.items() if v_}
        tp2 = {}
        for q in range(len(utr)):
            nm = tr_names[q]
            if line_c[q] in dep_of_line and nm in hd_t and abs(dose_med[q] - hd_t[nm]) < 1e-6 and (line_c[q], nm) in label:
                tp2.setdefault((line_c[q], nm), []).append(q)
        tr_keys2 = sorted(tp2)
        Etr2 = np.array([Ytr[tp2[key]].mean(axis=0) for key in tr_keys2])
        ytr2 = np.array([label[key] for key in tr_keys2])
        g2 = np.array([nm for _, nm in tr_keys2])
        mu2, sd2 = Etr2.mean(axis=0), Etr2.std(axis=0)
        sd2 = np.where(sd2 > 1e-8, sd2, 1.0)
        print('  (a) highest-dose variant: %d training pairs (%d drugs)' % (len(tr_keys2), len(set(g2))))
        f2, alpha_f2, mse_f2, r2_f2 = fit_ridge((Etr2 - mu2) / sd2, ytr2, g2, 'f, highest dose')
        FPq2 = np.array([s15.Drug_dose_encoder([smi_c[tp2[key][0]]], [hd_t[key[1]]], comb_num=1)[0] for key in tr_keys2], dtype=np.float64)
        qsar2, alpha_q2, mse_q2, r2_q2 = fit_ridge(s15.features(FPq2, np.array([ln for ln, _ in tr_keys2]), lv), ytr2, g2, 'qsar, highest dose')
        hd_h = {nm: max(v_) for nm, v_ in avail_h.items() if v_}
        hp2 = {}
        for i in range(n):
            nm = names[i]
            if lines[i] in dep_of_line and nm in hd_h and abs(doses[i] - hd_h[nm]) < 1e-6:
                hp2.setdefault((lines[i], nm), []).append(i)
        keys2 = [key for key in sorted(hp2) if key in label]
        line_ix2 = np.array([lv.index(ln) for ln, _ in keys2])
        drug_ix2 = pd.factorize(pd.Series([nm for _, nm in keys2]), sort=True)[0]
        labels2 = np.array([label[key] for key in keys2])
        print('  (a) highest-dose variant: %d labelled held-out pairs (%d drugs) | pairs per line %s'
              % (len(keys2), len(set(nm for _, nm in keys2)), {lv[l]: int((line_ix2 == l).sum()) for l in sorted(set(line_ix2))}))

        def pe2(E):
            return np.array([E[hp2[key]].mean(axis=0) for key in keys2])

        vals2 = {}
        vals2['recipe'] = np.mean(np.vstack([f2.predict((pe2(v['B'] - v['C']) - mu2) / sd2) for v in arms['recipe']]), axis=0)
        vals2['ridge_train'] = f2.predict((pe2(deltas['ridge_train']) - mu2) / sd2)
        vals2['ceiling'] = f2.predict((pe2(Y) - mu2) / sd2)
        FPh2 = np.array([s15.Drug_dose_encoder([drugs[hp2[key][0]]], [hd_h[key[1]]], comb_num=1)[0] for key in keys2], dtype=np.float64)
        vals2['qsar'] = qsar2.predict(s15.features(FPh2, np.array([ln for ln, _ in keys2]), lv))
        vals2['effect_size'] = np.mean(np.vstack([-np.linalg.norm(pe2(v['B'] - v['C']), axis=1) for v in arms['recipe']]), axis=0)
        vals2['effect_size_measured'] = -np.linalg.norm(pe2(Y), axis=1)
        explore = dict(lost_pairs_top_dose=int(lost), labelled_heldout_pairs_prism_lines=int(n_lab_held), heldout_drugs_without_top_dose=no_top,
                       dose_levels_available_per_heldout_drug=avail_h, training_names_not_in_match=n_tr_unmatched,
                       cv=dict(f=dict(alpha=alpha_f, mse=mse_f, r2=r2_f), qsar=dict(alpha=alpha_q, mse=mse_q, r2=r2_q),
                               f_extended_grid=dict(alpha=alpha_ext, mse=mse_ext, r2=r2_ext), f_highest_dose=dict(alpha=alpha_f2, mse=mse_f2, r2=r2_f2),
                               qsar_highest_dose=dict(alpha=alpha_q2, mse=mse_q2, r2=r2_q2), label_variance_training=float(np.var(ytr))),
                       variants={})
        for vname, vv, lab_, lix_, dix_ in (('extended_alpha_grid', vals_ext, labels, line_ix, drug_ix), ('highest_dose', vals2, labels2, line_ix2, drug_ix2)):
            sel_ = np.arange(len(lab_))
            dr_ = np.unique(dix_)
            Uv = {nm: U_of(vv[nm], lab_, lix_, len(lv), sel_)[0] for nm in vv}
            grp_ = [np.where(dix_ == d)[0] for d in dr_]
            Bv = {nm: np.full(N_BOOT, np.nan) for nm in vv}
            for b in range(N_BOOT):
                sel_b = np.concatenate([grp_[g] for g in rng.randint(0, len(grp_), len(grp_))])
                for nm in vv:
                    Bv[nm][b] = U_of(vv[nm], lab_, lix_, len(lv), sel_b)[0]
            cv_ = {}
            for cname, bn, an in (('U_recipe_minus_0', 'recipe', None), ('U_recipe_minus_ridge_train', 'recipe', 'ridge_train'),
                                  ('U_ceiling_minus_0', 'ceiling', None), ('U_qsar_minus_0', 'qsar', None), ('U_recipe_minus_qsar', 'recipe', 'qsar'),
                                  ('U_effect_size_measured_minus_0', 'effect_size_measured', None), ('U_recipe_minus_ceiling', 'recipe', 'ceiling')):
                dd = Bv[bn] - (Bv[an] if an else 0.0)
                lo, hi = float(np.nanpercentile(dd, LO)), float(np.nanpercentile(dd, HI))
                cv_[cname] = dict(diff=float(Uv[bn] - (Uv[an] if an else 0.0)), ci=[lo, hi], verdict=s15.verdict(lo, hi))
                print('  %-22s %-36s %s' % (vname, cname, s15.fmt(cv_[cname])))
            explore['variants'][vname] = dict(n_pairs=int(len(lab_)), n_drugs=int(len(dr_)), U={nm: float(x) for nm, x in Uv.items()}, contrasts=cv_)
            for nm, x in Uv.items():
                long_rows.append(dict(split=split, quantity='exploratory_%s_U' % vname, submission=nm, value=x))
            for cname, c in cv_.items():
                long_rows.append(dict(split=split, quantity='exploratory_%s_contrast' % vname, submission=cname, value=c['diff'], ci_lo=c['ci'][0], ci_hi=c['ci'][1], verdict=c['verdict']))
        del Ytr

        results[split] = dict(n_conditions=n, n_held_structures=n_drugs, dose_levels=[float(x) for x in levels], top_dose=float(top),
                              n_training_pairs=len(tr_keys), n_training_drugs=len(set(gtr_drug)), alpha_f=alpha_f, alpha_qsar=alpha_q,
                              n_heldout_pairs_top_dose=len(hp), n_labelled_pairs=len(keys), n_labelled_drugs=len(set(drug_names)),
                              n_unlabelled_pairs=n_unlabelled, pairs_per_line=per_line_n, n_lines_scored=n_lines_eff,
                              U={nm: float(v) for nm, v in U.items()}, U_per_seed=U_seed, within_drug={nm: float(v) for nm, v in W.items()},
                              contrasts=contrasts, exploratory=explore)
        for nm in vals:
            long_rows.append(dict(split=split, quantity='U', submission=nm, value=U[nm]))
            long_rows.append(dict(split=split, quantity='within_drug', submission=nm, value=W[nm]))
        for arm in s15.ARMS:
            for s_, u in enumerate(U_seed[arm]):
                long_rows.append(dict(split=split, quantity='U_seed_%d' % s_, submission=arm, value=u))
        for cname, c in contrasts.items():
            long_rows.append(dict(split=split, quantity='contrast', submission=cname, value=c['diff'], ci_lo=c['ci'][0], ci_hi=c['ci'][1], verdict=c['verdict']))
        for p, key in enumerate(keys):
            row = dict(split=split, line=key[0], drug=key[1], auc=labels[p], n_conditions=len(hp[key]))
            row.update({'f_%s' % nm: float(vals[nm][p]) for nm in vals})
            pair_rows.append(row)
        del arms, Etr

    out = dict(prereg=PRE, commit=commit, final=final, splits=splits, prism_files=hashes, n_labels=len(label), lines=sorted(dep_of_line),
               min_pairs_per_line=MIN_PAIRS_LINE, min_lines_per_drug=MIN_LINES_DRUG, n_boot=N_BOOT, alphas=list(ALPHAS),
               per_split=results, inputs=inputs, elapsed_s=int(time.time() - T0))
    if final:
        v1 = [results[s]['contrasts']['Q1_primary_U_recipe_minus_0']['verdict'] for s in s15.ALL_SPLITS]
        v2 = [results[s]['contrasts']['Q2_coprimary_U_recipe_minus_ridge_train']['verdict'] for s in s15.ALL_SPLITS]
        if all(v == 'above zero' for v in v1) and all(v == 'above zero' for v in v2):
            cat = '1 SENSITIVITY INFORMATION BEYOND LINEAR'
        elif all(v == 'above zero' for v in v1):
            cat = '2 SENSITIVITY INFORMATION, NOT BEYOND LINEAR'
        elif all(v == 'includes zero' for v in v1) or all(v == 'includes zero' for v in v2):
            cat = '5 UNRESOLVED'
        elif any(v == 'below zero' for v in v1):
            cat = '3 NO SENSITIVITY INFORMATION'
        else:
            cat = '4 PARTIAL'
        out['verdict'] = cat
        out['ceiling_verdicts'] = [results[s]['contrasts']['secondary_U_ceiling_minus_0']['verdict'] for s in s15.ALL_SPLITS]
        print('\n[section 6] primary %s | co-primary %s | ceiling %s -> %s' % (v1, v2, out['ceiling_verdicts'], cat))
    json.dump(out, open(OUT, 'w'), indent=1)
    pd.DataFrame(long_rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(pair_rows).to_csv(OUT_PAIRS, index=False)
    print('wrote %s, %s, %s (%d s)' % (OUT, OUT_CSV, OUT_PAIRS, time.time() - T0))


if __name__ == '__main__':
    main()
