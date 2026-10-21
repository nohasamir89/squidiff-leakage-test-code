#!/usr/bin/env python
"""Step 15c scoring: perturbation discrimination (PDS) on the frozen Step 15b Tahoe predictions.

Pre-registration: PREREG_step15c_tahoe_drug_specific.md (committed before this file; its dated section-4 paragraph of
2026-10-03 fixes the two exact-0.5 checks). This script, committed before any result exists, does only what that
document says:

  1. guards: the pre-registration and this scorer committed and clean (also score_step15_tahoe.py and step14_metrics.py,
     which it imports); the Step 15b records present; every vectors array hash-checked against provenance/s15ev_*.json;
  2. loads the 15 certified mean-vector files per split (cond, true, B, C) - no per-cell arrays are needed, PDS is
     profile-level - and the Step 15b per-condition record (tahoe_results/step15_model_per_condition.csv) for the
     condition annotations and the learnability descriptor;
  3. rebuilds the four shift baselines by the Step 15b code path (ridge-S7, ridge-train, mean-response, control-paste)
     and ASSERTS that their per-condition DE-overlap and DPC-50, and every arm's, reproduce the Step 15b record to 1e-9;
  4. computes PDS per condition for every submission with the section-4 candidate set: the held-out conditions of the
     same line at the same dose, minus every other condition of the same drug (structure); conditions with fewer than
     10 candidates are excluded for every submission alike; arms are scored per seed and averaged per condition;
  5. checks the ranking code: the adapted function equals step14_metrics.pds_cosine on full sets (random data at start),
     control-paste gives 0.5 exactly per condition on the adapted sets, mean-response gives 0.5 exactly per (line, dose)
     context on the full-context sets;
  6. replicate ceiling (plate 6 effect offered as the prediction of the plate-14 copy and vice versa), summaries per
     seed / line / dose / drug, brain vs gut, Spearman against learnability;
  7. the section-5 contrasts with the structure-clustered percentile bootstrap of Step 15b (score_step15_tahoe.contrast:
     5,000 replicates, RandomState(0), 97.5%), and section 6 when all three splits are present.

Usage: python -u score_step15c_tahoe.py [--splits drug_split_1]     (development run first, then no --splits = final)
Outputs: tahoe_results/step15c_pds[_development].json, step15c_pds_long[_development].csv,
         step15c_pds_per_condition[_development].csv. Never overwrites.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402  the Step 15b scorer: loaders, baselines, bootstrap
from step14_metrics import pds_cosine                  # noqa: E402  Step 14's facet, the reference for the self-check

PRE = 'PREREG_step15c_tahoe_drug_specific.md'
PC15 = os.path.join(s15.RES, 'step15_model_per_condition.csv')
SUBS = list(s15.BASELINES) + list(s15.ARMS)
MIN_CAND = int(os.environ.get('S15C_MIN_CAND', 10))   # pre-registered value 10; the env override exists for the synthetic test only and is written into the record
DOSES_LABEL = lambda d: ('%g' % d)
T0 = time.time()


# ------------------------------------------------------------------ inputs
def load_vectors(split, arm, seed):
    """One certified evaluation's mean vectors, provenance-checked as Step 15b (vectors hash only; no per-cell array)."""
    tag = 's15_%s_%s_seed_%d' % (split, arm, seed)
    f = os.path.join(s15.RES, 'vectors_%s.npz' % tag)
    pj = os.path.join('provenance', 's15ev_%s.json' % tag)
    for q in (f, pj):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    pv = json.load(open(pj))
    s15.check_prov(pv, tag, dict(status='completed', arm=arm, split=split, seed=seed, drug_encoding='fingerprint',
                                 guidance_w=(2.0 if arm == 'guidance_w2' else None), nfe=(None if arm == 'base' else 50),
                                 max_cells_per_cond=s15.CAP))
    if pv.get('vectors_sha256') != s15.sha(f):
        sys.exit('FATAL: %s vectors hash differs from provenance' % tag)
    v = np.load(f, allow_pickle=True)
    return dict(tag=tag, cond=np.array([str(x) for x in v['cond']]), true=np.asarray(v['true'], np.float64),
                B=np.asarray(v['B'], np.float64), C=np.asarray(v['C'], np.float64),
                rec=dict(tag=tag, vectors_sha256=pv['vectors_sha256'], checkpoint_sha256=pv.get('checkpoint_sha256')))


# ------------------------------------------------------------------ the metric
def unit_rows(M):
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    return np.where(nrm > 0, M / np.where(nrm > 0, nrm, 1.0), 0.0), (nrm[:, 0] > 0)


def pds_sets(pred_eff, true_eff, contexts, cand):
    """Step 14's pds_cosine with per-condition candidate sets. contexts: list of index arrays (one per (line, dose));
    cand[p]: the candidate indices of condition p (a subset of p's context containing p). Distances are computed per
    context as 1 - P @ T.T exactly as pds_cosine does; a zero-norm vector has distance 1 to everything; ties averaged;
    PDS_p = 1 - k_p / (n_p - 1); NaN where n_p < MIN_CAND."""
    P, okp = unit_rows(np.asarray(pred_eff, np.float64))
    T, okt = unit_rows(np.asarray(true_eff, np.float64))
    out = np.full(len(P), np.nan)
    for members in contexts:
        D = 1.0 - P[members] @ T[members].T
        D[~okp[members], :] = 1.0
        D[:, ~okt[members]] = 1.0
        pos = {q: j for j, q in enumerate(members)}
        for i, p in enumerate(members):
            if p not in cand:                                   # only the conditions with a candidate set are scored
                continue
            cols = np.array([pos[q] for q in cand[p]])
            if len(cols) < MIN_CAND:
                continue
            d = D[i, cols]
            own = D[i, pos[p]]
            k = np.sum(d < own) + 0.5 * (np.sum(d == own) - 1)
            out[p] = 1.0 - k / (len(cols) - 1)
    return out


def self_check():
    """On random data with full candidate sets the adapted function must equal step14_metrics.pds_cosine."""
    r = np.random.RandomState(7)
    for n, G in ((12, 5), (40, 200)):
        A, B = r.randn(n, G), r.randn(n, G)
        A[0] = 0.0; B[1] = 0.0                                  # zero-norm rows on both sides
        ref = pds_cosine(A, B)
        got = pds_sets(A, B, [np.arange(n)], {p: np.arange(n) for p in range(n)})
        if np.nanmax(np.abs(ref - got)) > 1e-9:
            sys.exit('FATAL: pds_sets does not reproduce pds_cosine on full sets')
    print('[self-check] pds_sets == pds_cosine on full candidate sets (two random cases incl. zero-norm rows)')


# ------------------------------------------------------------------ helpers
def contrast(x_b, x_a, drugs):
    """Step 15b's contrast with the cluster count taken from the finite entries (excluded conditions drop out)."""
    m = np.isfinite(x_b) & np.isfinite(x_a)
    return s15.contrast(x_b, x_a, drugs, len(np.unique(drugs[m])))


def fmt(c):
    return s15.fmt(c)


def close(a, b, tol=1e-9):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    return np.array_equal(na, nb) and (np.abs(a[~na] - b[~nb]).max() if (~na).any() else 0.0) <= tol


# ------------------------------------------------------------------ main
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
    OUT = os.path.join(s15.RES, 'step15c_pds%s.json' % suffix)
    OUT_CSV = os.path.join(s15.RES, 'step15c_pds_long%s.csv' % suffix)
    OUT_PC = os.path.join(s15.RES, 'step15c_pds_per_condition%s.csv' % suffix)
    for q in (OUT, OUT_CSV, OUT_PC):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 9:
        sys.exit('FATAL: prereg incomplete')
    tracked = '%s score_step15c_tahoe.py score_step15_tahoe.py step14_metrics.py %s' % (PRE, PC15)
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: prereg, scorer, imported scorers or the Step 15b per-condition record not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0:
        sys.exit('FATAL: prereg, scorer, imported scorers or the Step 15b per-condition record uncommitted')
    record = json.load(open(s15.RECORD))
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    print('[prereg] committed; scorer committed; commit %s; splits %s (%s)' % (commit[:7], splits, 'final' if final else 'development'))
    self_check()
    pc15 = pd.read_csv(PC15)

    results, inputs, rows, pc_rows = {}, {}, [], []
    for split in splits:
        print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
        k = split.split('_')[-1]
        rec_split = record['splits'][k]
        n_drugs = len(rec_split['held_structures'])

        # --- the 15 certified mean-vector files; identical cond / C / true across runs
        arms = {arm: [load_vectors(split, arm, s) for s in range(s15.N_SEEDS)] for arm in s15.ARMS}
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

        # --- Step 15b per-condition record: annotations and learnability, in the vectors' condition order
        pc = pc15[pc15['split'] == split].reset_index(drop=True)
        if len(pc) != n or not np.array_equal(pc['condition'].astype(str).values, cond):
            sys.exit('FATAL: the Step 15b per-condition record does not list the conditions of %s in the vectors order' % split)
        lines = pc['line'].astype(str).values
        plates = pc['plate'].astype(str).values
        drugs = pc['structure'].astype(str).values
        doses = pc['dose_nM'].astype(float).values
        learn = pc['learnability'].astype(float).values
        lv = sorted(set(lines))
        if len(np.unique(drugs)) != n_drugs:
            sys.exit('FATAL: %d structures in the record, expected %d' % (len(np.unique(drugs)), n_drugs))

        # --- the test file's annotations must agree with the record (obs only, no X)
        te = ad.read_h5ad('%s/tahoe_test_%s.h5ad' % (s15.DATA, split), backed='r')
        g_all = te.obs['Group'].astype(str).values
        idx = s15.condition_index(g_all)
        smi_all, dose_all, line_all = te.obs['SMILES'].astype(str).values, te.obs['dose'].astype(float).values, te.obs['cell_name'].astype(str).values
        for i, c in enumerate(cond):
            ix = idx[c]
            if smi_all[ix[0]] != drugs[i] or line_all[ix[0]] != lines[i] or s15.parse_cond(c)[1] != plates[i] or abs(float(np.median(dose_all[ix])) - doses[i]) > 1e-9:
                sys.exit('FATAL: annotation of %s differs between the test file and the Step 15b record' % c)
        te.file.close()
        print('[design] %d conditions | %d held-out structures | %d lines | doses %s' % (n, n_drugs, len(lv), sorted(set(DOSES_LABEL(d) for d in doses), key=float)))

        # --- shift baselines, the Step 15b code path
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
        FPtr = np.array([s15.Drug_dose_encoder([smi_tr[j]], [dose_med[q]], comb_num=1)[0] for q, j in enumerate(first)], dtype=np.float64)
        Ptr = Ridge(alpha=s15.ALPHA).fit(s15.features(FPtr, line_tr[first], lv), Ytr).predict(X)
        del tr, trc, Xtr, Xtrc, Ytr
        mr = np.zeros_like(Y)
        for ln in lv:
            ixl = np.where(lines == ln)[0]
            mr[ixl] = Y[ixl].mean(axis=0)[None, :]
        deltas = {'ridge_S7': P7, 'ridge_train': Ptr, 'mean_response': mr, 'control_paste': np.zeros_like(Y)}

        # --- reproduction guard against the Step 15b record (profile metrics; energy needs per-cell arrays and is not redone)
        for bname, D in deltas.items():
            j, d = s15.thesis_per_cond(true_mean, C + D, C)
            if not close(j, pc['%s_de_overlap' % bname].values) or not close(d, pc['%s_dpc50' % bname].values):
                sys.exit('FATAL: %s does not reproduce the Step 15b per-condition record' % bname)
        for arm, L in arms.items():
            jj, dd = zip(*[s15.thesis_per_cond(v['true'], v['B'], v['C']) for v in L])
            if not close(np.nanmean(np.vstack(jj), axis=0), pc['%s_de_overlap' % arm].values) or not close(np.nanmean(np.vstack(dd), axis=0), pc['%s_dpc50' % arm].values):
                sys.exit('FATAL: %s does not reproduce the Step 15b per-condition record' % arm)
        print('[reproduction] DE-overlap and DPC-50 of all nine submissions equal the Step 15b record to 1e-9')

        # --- candidate sets: same line, same dose, minus other conditions of the same structure (section 4)
        ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
        contexts = [np.where(ctx_key == u)[0] for u in np.unique(ctx_key)]
        cand, cand_full = {}, {}
        for members in contexts:
            for p in members:
                cand_full[p] = members
                cand[p] = np.array([q for q in members if q == p or drugs[q] != drugs[p]])
        n_cand = np.array([len(cand[p]) for p in range(n)])
        excluded = int((n_cand < MIN_CAND).sum())
        if excluded > 0.01 * n:
            sys.exit('FATAL: %d conditions have fewer than %d candidates' % (excluded, MIN_CAND))
        print('[candidates] %d (line, dose) contexts | candidates per condition %d-%d (median %d) | excluded (< %d): %d'
              % (len(contexts), n_cand.min(), n_cand.max(), int(np.median(n_cand)), MIN_CAND, excluded))

        # --- PDS per submission
        pds = {}
        per_seed = {}
        for bname, D in deltas.items():
            pds[bname] = pds_sets(D, Y, contexts, cand)
            per_seed[bname] = [pds[bname]]
        for arm, L in arms.items():
            per_seed[arm] = [pds_sets(v['B'] - v['C'], Y, contexts, cand) for v in L]
            pds[arm] = np.nanmean(np.vstack(per_seed[arm]), axis=0)
        # exact checks of section 4 (dated paragraph): control-paste on the adapted sets; mean-response on full sets per context
        cp = pds['control_paste']
        if np.nanmax(np.abs(cp - 0.5)) > 1e-9:
            sys.exit('FATAL: control-paste PDS is not 0.5 on every scored condition')
        mr_full = pds_sets(mr, Y, contexts, cand_full)
        for members in contexts:
            v_ = mr_full[members]
            if len(members) >= MIN_CAND and abs(np.nanmean(v_) - 0.5) > 1e-9:
                sys.exit('FATAL: mean-response PDS on the full-context sets is not 0.5 in a context')
        print('[checks] control-paste = 0.5 on the adapted sets; mean-response = 0.5 per context on the full sets')

        # --- replicate ceiling: plate 6 effect as the prediction of the plate-14 copy, and vice versa
        keyed = {}
        for i, c in enumerate(cond):
            key, plate = s15.parse_cond(c)
            if plate in s15.REP_PLATES:
                keyed.setdefault(key, {})[plate] = i
        pairs = [(v['6'], v['14']) for v in keyed.values() if '6' in v and '14' in v]
        rep_vals = []
        for i6, i14 in pairs:
            for src, dst in ((i6, i14), (i14, i6)):
                E = Y.copy()
                E[dst] = Y[src]                                             # the other plate's measured effect offered at dst
                rep_vals.append(pds_sets(E, Y, [cand_full[dst]], {dst: cand[dst]})[dst])   # dst's own candidate set excludes src (same drug)
        rep_vals = np.array(rep_vals, dtype=np.float64)
        replicate = dict(n_pairs=len(pairs), pds_mean=float(np.nanmean(rep_vals)) if len(pairs) else None,
                         pds_median=float(np.nanmedian(rep_vals)) if len(pairs) else None)
        print('[replicate] plate 6 vs 14: %d pairs | PDS mean %.4f median %.4f' % (len(pairs), replicate['pds_mean'], replicate['pds_median']))

        # --- summaries
        ib = np.isin(lines, s15.BRAIN)
        metrics = {}
        for name in SUBS:
            x = pds[name]
            m_ = dict(pds_mean=float(np.nanmean(x)), pds_median=float(np.nanmedian(x)), n_scored=int(np.isfinite(x).sum()),
                      per_seed=[float(np.nanmean(s_)) for s_ in per_seed[name]],
                      per_line={ln: float(np.nanmean(x[lines == ln])) for ln in lv},
                      per_dose={DOSES_LABEL(d): float(np.nanmean(x[np.isclose(doses, d)])) for d in sorted(set(doses))},
                      per_drug={dr: float(np.nanmean(x[drugs == dr])) for dr in sorted(set(drugs))},
                      brain_minus_gut=float(np.nanmean(x[ib]) - np.nanmean(x[~ib])))
            metrics[name] = m_
            for ln, val in m_['per_line'].items():
                rows.append(dict(split=split, level='line', key=ln, submission=name, value=val))
            for ds, val in m_['per_dose'].items():
                rows.append(dict(split=split, level='dose', key=ds, submission=name, value=val))
            for dr, val in m_['per_drug'].items():
                rows.append(dict(split=split, level='drug', key=dr, submission=name, value=val))
            rows.append(dict(split=split, level='ALL', key='mean', submission=name, value=m_['pds_mean']))
            rows.append(dict(split=split, level='ALL', key='median', submission=name, value=m_['pds_median']))
            print('[pds] %-14s mean %.4f median %.4f | per-seed %s | per-dose %s | brain-gut %+.4f'
                  % (name, m_['pds_mean'], m_['pds_median'], [round(v, 4) for v in m_['per_seed']],
                     {d_: round(v, 4) for d_, v in m_['per_dose'].items()}, m_['brain_minus_gut']))
        rho = spearmanr(learn, pds['recipe'], nan_policy='omit')
        learn_corr = [float(rho[0]), float(rho[1])]
        print('[learnability] Spearman of recipe PDS vs learnability rho %.4f (p %.2e)' % tuple(learn_corr))
        for i in range(n):
            r = dict(split=split, condition=cond[i], line=lines[i], plate=plates[i], structure=drugs[i], dose_nM=doses[i],
                     n_candidates=int(n_cand[i]), learnability=learn[i])
            for name in SUBS:
                r['%s_pds' % name] = pds[name][i]
            pc_rows.append(r)

        # --- the pre-registered contrasts (section 5)
        half = np.full(n, 0.5)
        con = {}
        con['Q1_primary_pds_recipe_minus_0.5'] = contrast(pds['recipe'], half, drugs)
        con['Q2_coprimary_pds_recipe_minus_ridge_train'] = contrast(pds['recipe'], pds['ridge_train'], drugs)
        print('\nQ1 PRIMARY     PDS, recipe minus 0.5            %s' % fmt(con['Q1_primary_pds_recipe_minus_0.5']))
        print('Q2 CO-PRIMARY  PDS, recipe minus ridge-train    %s' % fmt(con['Q2_coprimary_pds_recipe_minus_ridge_train']))
        for name in SUBS:
            if name in ('recipe', 'control_paste'):
                continue
            con['secondary_pds_%s_minus_0.5' % name] = contrast(pds[name], half, drugs)
            print('secondary   PDS, %-14s minus 0.5      %s' % (name, fmt(con['secondary_pds_%s_minus_0.5' % name])))
        con['secondary_pds_recipe_minus_ridge_S7'] = contrast(pds['recipe'], pds['ridge_S7'], drugs)
        print('secondary   PDS, recipe minus ridge-S7         %s' % fmt(con['secondary_pds_recipe_minus_ridge_S7']))
        con['secondary_pds_guidance_w2_minus_recipe'] = contrast(pds['guidance_w2'], pds['recipe'], drugs)
        print('secondary   PDS, guidance w=2 minus recipe     %s' % fmt(con['secondary_pds_guidance_w2_minus_recipe']))
        con['exploratory_pds_attn_cross_minus_mlp_ctrl'] = contrast(pds['attn_cross'], pds['mlp_ctrl'], drugs)
        print('exploratory PDS, attn_cross minus mlp_ctrl     %s' % fmt(con['exploratory_pds_attn_cross_minus_mlp_ctrl']))
        results[split] = dict(n_conditions=n, n_structures=n_drugs, n_contexts=len(contexts), min_candidates=MIN_CAND,
                              candidates_per_condition=[int(n_cand.min()), int(n_cand.max())], excluded=excluded,
                              metrics=metrics, replicate=replicate, learnability_spearman_recipe=learn_corr, contrasts=con)

    # ------------------------------------------------------------------ section 6
    cat = None
    if final:
        q1 = {s: results[s]['contrasts']['Q1_primary_pds_recipe_minus_0.5']['verdict'] for s in s15.ALL_SPLITS}
        q2 = {s: results[s]['contrasts']['Q2_coprimary_pds_recipe_minus_ridge_train']['verdict'] for s in s15.ALL_SPLITS}
        v1, v2 = list(q1.values()), list(q2.values())
        if all(v == 'above zero' for v in v1) and all(v == 'above zero' for v in v2):
            cat = 'DRUG-SPECIFIC SIGNAL BEYOND LINEAR'
        elif all(v == 'above zero' for v in v1):
            cat = 'DRUG-SPECIFIC SIGNAL, NOT BEYOND LINEAR'
        elif all(v == 'includes zero' for v in v1) or all(v == 'includes zero' for v in v2):
            cat = 'UNRESOLVED'
        elif 'below zero' in v1:
            cat = 'NO DRUG-SPECIFIC SIGNAL'
        elif 1 <= v1.count('above zero') <= 2:
            cat = 'PARTIAL'
        else:
            cat = 'NO DRUG-SPECIFIC SIGNAL'
        print('\n' + '=' * 72 + '\nPRE-REGISTERED VERDICT (section 6): %s\n  Q1 recipe - 0.5 per split: %s\n  Q2 recipe - ridge-train per split: %s\n'
              % (cat, q1, q2) + '  Section 7 expected: DRUG-SPECIFIC SIGNAL, NOT BEYOND LINEAR.\n' + '=' * 72)
    else:
        print('\n[development] %s scored; section 6 is applied only when all three splits are present' % ', '.join(splits))

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'level', 'key', 'submission', 'value'])
        w.writeheader(); w.writerows(rows)
    with open(OUT_PC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0].keys()))
        w.writeheader(); w.writerows(pc_rows)
    json.dump(dict(prereg=PRE, final=final, verdict=cat, splits=splits, interval='97.5% percentile per split, structure-clustered',
                   n_boot=s15.N_BOOT, n_seeds=s15.N_SEEDS, min_candidates=MIN_CAND, per_split=results, inputs=inputs,
                   step15b_per_condition_sha256=s15.sha(PC15), commit=commit, seconds=round(time.time() - T0)),
              open(OUT, 'w'), indent=2, allow_nan=True)
    print('\nwrote %s, %s and %s  (%.0f s)' % (OUT, OUT_CSV, OUT_PC, time.time() - T0))


if __name__ == '__main__':
    main()
