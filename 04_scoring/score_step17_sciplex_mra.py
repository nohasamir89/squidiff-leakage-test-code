#!/usr/bin/env python
"""Step 17 scoring under PREREG_step17_sciplex_mra.md: response-magnitude agreement (MRA) on the frozen sci-Plex predictions.

What it does, per split (drug_split_1 alone = development run; all three = final run):
  1. loads the Step 14 submission set from leakage_results/vectors_*.npz, every file checked against its provenance
     record (the registry, the loader and the checks are copied verbatim from score_step14.py, which cannot be imported
     because it runs at import); mean vectors only, except the comparator's seed-0 per-cell file for the anchor;
  2. rebuilds the four shift baselines exactly as score_step14.py (ridge-S7 = Step 7's within-held-out interaction
     ridge, asserted equal to the Step 7 record; ridge-train on the training split's condition deltas; mean-response
     per line; control-paste);
  3. reproduction guard: every submission's DE-overlap and DPC-50 mean, from the same arrays by the same code, must
     equal the Step 14 record (leakage_results/step14_analysis.json, thesis_metrics) to 1e-9 before any MRA is printed;
  4. MRA as Step 16 section 4.4, by score_step16_tahoe's own functions: per (line, dose) context with >= 10 conditions,
     the Spearman between ||B - C|| and ||true - C|| (C = the comparator's paired-control mean, one origin for every
     submission); mean over contexts; arms per seed then averaged; control-paste and mean-response asserted 0;
  5. the split-half anchor (section 3): five seeded half-splits of each condition's matched real cells and paired
     vehicle cells (the Step 14 construction, copied), Spearman within context between ||profA - cA|| and ||profB - cB||;
  6. intervals: compound-clustered percentile bootstrap, 5,000 replicates, RandomState(0), 97.5 % per split, the MRA
     recomputed per replicate; Q1 fm_50 - 0, Q2 fm_50 - ridge_train (gated), the secondaries of section 4, and the
     descriptive per-dose / per-line / anchor intervals;
  7. the section 5 verdict across the splits of the run (final run only), and the records of section 9.
Every number it uses is printed. It refuses to run if any input is missing, any hash differs, the pre-registration or
the code is uncommitted, or an output exists.

Usage (repo root, conda env active):
  python -u score_step17_sciplex_mra.py --splits drug_split_1        # development
  python -u score_step17_sciplex_mra.py                              # final, all three splits
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Squidiff.scrna_datasets import Drug_dose_encoder   # noqa: E402  the harness's fingerprint + dose encoder
import score_step16_tahoe as s16                        # noqa: E402  fast_rank, prep_ctx, mra_pre, ctx_split, mra_self_check, MIN_CTX

RES = 'leakage_results'
DATA = os.environ.get('S17_DATA', os.environ.get('S14_DATA', '/scratch/ni2071/Diffusion/sciplex/datasets_corrected'))
PRE = 'PREREG_step17_sciplex_mra.md'
REC14 = os.path.join(RES, 'step14_analysis.json')
ALL_SPLITS = ('drug_split_1', 'drug_split_0', 'drug_split_2')          # development first
N_DRUGS = {'drug_split_1': 37, 'drug_split_0': 37, 'drug_split_2': 38}   # held-out compounds per split (Step 14)
N_SEEDS, K, N_BOOT, ALPHA = 5, 50, 5000, 1.0
LO, HI = 1.25, 98.75                                                    # 97.5 % per split
MIN_CELLS, CAP, N_REP = 20, 200, 5
COMPARATOR = 'fm_50'
ARMS = ('fm_50', 'guidance_w2', 'attn_cross', 'mlp_ctrl')
BASELINES = ('ridge_S7', 'ridge_train', 'mean_response', 'control_paste')
SUBS = BASELINES + ('base',) + ARMS
MIN_CTX = s16.MIN_CTX
T0 = time.time()


# ------------------------------------------------------------------ verbatim from score_step14.py (which runs at import and cannot be imported)
def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): m.update(c)
    return m.hexdigest()

def jac(a, b):
    i = len(np.intersect1d(a, b)); return i / (len(a) + len(b) - i)

def thesis_per_cond(true, B, C):
    """DE-overlap Jaccard and DPC-50 per condition from mean vectors (true, predicted B, paired control C)."""
    n = len(true); j = np.full(n, np.nan); d = np.full(n, np.nan)
    for i in range(n):
        td = true[i] - C[i]; p = B[i] - C[i]
        g = np.argsort(np.abs(td))[-K:]
        if np.abs(p).std() > 1e-12: j[i] = jac(g, np.argsort(np.abs(p))[-K:])
        if td[g].std() > 1e-12 and p[g].std() > 1e-12: d[i] = pearsonr(p[g], td[g])[0]
    return j, d

def verdict(lo, hi):
    return 'above zero' if lo > 0 else 'below zero' if hi < 0 else 'includes zero'

def dense(X):
    return np.asarray(X.todense() if hasattr(X, 'todense') else X, dtype=np.float32)

def registry(split):
    if split == 'drug_split_1':
        R = {'fm_50': ('sw_fm_gaussian_seed_%d_n50', 'provenance/s6sw_sw_fm_gaussian_seed_%d_n50.json', dict(arm='fm_gaussian', steps=50)),
             'guidance_w2': ('s12_cfg_w2_seed_%d', 'provenance/s12ev_s12_cfg_w2_seed_%d.json', dict(guidance_w=2.0)),
             'attn_cross': ('s13_drug_split_1_attn_cross_seed_%d', 'provenance/s13ev_s13_drug_split_1_attn_cross_seed_%d.json', dict(arm='attn_cross')),
             'mlp_ctrl': ('s13_drug_split_1_mlp_ctrl_seed_%d', 'provenance/s13ev_s13_drug_split_1_mlp_ctrl_seed_%d.json', dict(arm='mlp_ctrl'))}
        base = ('step5_base_seed_%d', 'provenance/s5eval_step5_base_seed_%d.json', dict(arm='base'))
    else:
        s = split
        R = {'fm_50': ('s11_%s_fm_gaussian_nfe50_seed_%%d' % s, 'provenance/s11ev_s11_%s_fm_gaussian_nfe50_seed_%%d.json' % s, dict(split=s, nfe=50)),
             'guidance_w2': ('s12b_%s_w2_seed_%%d' % s, 'provenance/s12bev_s12b_%s_w2_seed_%%d.json' % s, dict(split=s, guidance_w=2.0)),
             'attn_cross': ('s13b_%s_attn_cross_seed_%%d' % s, 'provenance/s13bev_s13b_%s_attn_cross_seed_%%d.json' % s, dict(arm='attn_cross', split=s)),
             'mlp_ctrl': ('s13b_%s_mlp_ctrl_seed_%%d' % s, 'provenance/s13bev_s13b_%s_mlp_ctrl_seed_%%d.json' % s, dict(arm='mlp_ctrl', split=s))}
        base = ('repl_%s_base_seed_%%d' % s, 'provenance/repleval_repl_%s_base_seed_%%d.json' % s, dict(arm='base', split=s))
    return R, base

def check_prov(pv, name, checks):
    """A provenance key that exists must carry the expected value; keys absent from a record are reported, not assumed."""
    for k, v in checks.items():
        if k not in pv: print('  [note] %s: provenance has no key %r (expected %r)' % (name, k, v)); continue
        got = pv[k]
        ok = (abs(float(got) - float(v)) < 1e-9) if isinstance(v, (int, float)) else (str(got) == str(v))
        if not ok: sys.exit('FATAL: %s provenance %s=%r, expected %r' % (name, k, got, v))

def load_vectors(name, pj, checks, need_pc=True):
    f = os.path.join(RES, 'vectors_%s.npz' % name); fpc = os.path.join(RES, 'vectors_%s_pc.npz' % name)
    for q in ([f, pj] + ([fpc] if need_pc else [])):
        if not os.path.isfile(q): sys.exit('FATAL: missing %s' % q)
    pv = json.load(open(pj)); check_prov(pv, name, checks)
    if pv.get('vectors_sha256') != sha(f): sys.exit('FATAL: %s vectors hash differs from provenance' % name)
    if need_pc and pv.get('per_cell_sha256') != sha(fpc): sys.exit('FATAL: %s per-cell hash differs from provenance' % name)
    v = np.load(f, allow_pickle=True)
    out = dict(cond=np.array([str(x) for x in v['cond']]), true=np.asarray(v['true'], np.float64), B=np.asarray(v['B'], np.float64), C=np.asarray(v['C'], np.float64),
               rec=dict(tag=name, vectors_sha256=pv.get('vectors_sha256'), per_cell_sha256=pv.get('per_cell_sha256') if need_pc else None, checkpoint_sha256=pv.get('checkpoint_sha256')))
    if need_pc:
        p = np.load(fpc, allow_pickle=True)
        out['pred'] = [np.asarray(a, np.float32) for a in p['pred']]; out['true_cells'] = [np.asarray(a, np.float32) for a in p['true']]
        if len(out['pred']) != len(out['cond']): sys.exit('FATAL: %s per-cell length differs from vectors' % name)
    return out


# ------------------------------------------------------------------ intervals
def verdict_of(point, dist):
    lo, hi = float(np.nanpercentile(dist, LO)), float(np.nanpercentile(dist, HI))
    return dict(diff=float(point), ci=[lo, hi], verdict=verdict(lo, hi), half_width=float((hi - lo) / 2), n_nan=int(np.isnan(dist).sum()))

def fmt(c):
    return '%+.5f  [%+.5f, %+.5f] -> %s' % (c['diff'], *c['ci'], c['verdict'])


# ------------------------------------------------------------------ one split
def score_split(split, rec14, rows, pc_rows):
    print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
    n_drugs = N_DRUGS[split]
    R, base = registry(split)
    arms = {}
    for name, (vec, pj, checks) in R.items():
        arms[name] = [load_vectors(vec % s, pj % s, checks, need_pc=(name == COMPARATOR and s == 0)) for s in range(N_SEEDS)]
    base_v = [load_vectors(base[0] % s, base[1] % s, base[2], need_pc=False) for s in range(N_SEEDS)]
    cond = arms[COMPARATOR][0]['cond']; n = len(cond)
    for name, L in list(arms.items()) + [('base', base_v)]:
        for v in L:
            if not np.array_equal(v['cond'], cond): sys.exit('FATAL: condition order differs in %s' % v['rec']['tag'])
            if np.abs(v['C'] - arms[COMPARATOR][0]['C']).max() > 1e-5: sys.exit('FATAL: paired-control means differ in %s' % v['rec']['tag'])
            if np.abs(v['true'] - arms[COMPARATOR][0]['true']).max() > 1e-5: sys.exit('FATAL: true means differ in %s' % v['rec']['tag'])
    inputs = {name: [v['rec'] for v in L] for name, L in list(arms.items()) + [('base', base_v)]}
    true_mean = arms[COMPARATOR][0]['true']; C = arms[COMPARATOR][0]['C']
    print('[inputs] %d submissions with %d seeds each loaded and hash-checked; %d conditions, %d genes' % (len(arms) + 1, N_SEEDS, n, true_mean.shape[1]))

    # --- data: test cells, paired vehicle cells (as score_step14.py)
    te = ad.read_h5ad('%s/sci_plex_test_%s.h5ad' % (DATA, split)); tc = ad.read_h5ad('%s/sci_plex_test_%s_control.h5ad' % (DATA, split))
    if not np.array_equal(te.obs_names, tc.obs_names): sys.exit('FATAL: test and control files are not row-aligned')
    Xte = dense(te.X); Xtc = dense(tc.X)
    g_all = te.obs['Group'].astype(str).values; smi_all = te.obs['SMILES'].astype(str).values
    dose_all = te.obs['dose'].astype(float).values; line_all = te.obs['cell_name'].astype(str).values
    drugs = np.array([smi_all[g_all == c][0] for c in cond]); lines = np.array([line_all[g_all == c][0] for c in cond])
    doses = np.array([float(np.median(dose_all[g_all == c])) for c in cond])
    if len(np.unique(drugs)) != n_drugs: sys.exit('FATAL: %d drugs, expected %d' % (len(np.unique(drugs)), n_drugs))
    real = []; paired = []; rs = np.random.RandomState(0); recovered = exact = tolerant = with_dup = 0; n_stored = []
    for i, c in enumerate(cond):
        ix = np.where(g_all == c)[0]
        stored = arms[COMPARATOR][0]['true_cells'][i]; n_stored.append(len(stored))
        lookup = {Xte[j].tobytes(): j for j in ix}
        hit = [lookup.get(np.asarray(r, np.float32).tobytes()) for r in stored]
        if all(h is not None for h in hit):
            sel = np.array(sorted(set(hit))); recovered += 1; exact += 1
        else:                                                    # nearest test row within 1e-5 (max-abs)
            D = cdist(np.asarray(stored, np.float64), Xte[ix].astype(np.float64), metric='chebyshev')
            if (D.min(axis=1) < 1e-5).all(): sel = np.array(sorted(set(ix[np.argmin(D, axis=1)].tolist()))); recovered += 1; tolerant += 1
            else: sel = np.sort(rs.choice(ix, CAP, replace=False)) if len(ix) > CAP else ix
        if len(sel) < len(stored): with_dup += 1
        real.append(Xte[sel]); paired.append(Xtc[sel])
    n_real = np.array([len(a) for a in real]); gated = n_real >= MIN_CELLS
    if n_real.min() < 2: sys.exit('FATAL: a condition has fewer than 2 matched real cells; the split-half anchor is undefined')
    print('[design] %d conditions | %d drugs | lines %s | dose levels %s | stored cells per condition %d-%d, distinct %d-%d | >= %d distinct cells: %d | harness draw recovered in %d conditions (%d exact, %d within 1e-5), seeded draw for %d'
          % (n, n_drugs, sorted(set(str(x) for x in lines)), sorted(set(doses.tolist())), min(n_stored), max(n_stored), n_real.min(), n_real.max(), MIN_CELLS, gated.sum(), recovered, exact, tolerant, n - recovered))

    # --- the ridges, exactly as score_step14.py
    FP = np.array([Drug_dose_encoder([smi_all[g_all == c][0]], [float(np.median(dose_all[g_all == c]))], comb_num=1)[0] for c in cond], dtype=np.float64)
    lv = sorted(set(lines)); Lmat = np.array([[float(l == u) for u in lv] for l in lines])
    Xfeat = np.hstack([FP, Lmat, np.concatenate([FP * Lmat[:, [k]] for k in range(len(lv))], axis=1)])
    Y = true_mean - C
    v7 = ('vectors_step6_fm_gaussian_seed_0.npz', 'provenance/s6eval_step6_fm_gaussian_seed_0.json') if split == 'drug_split_1' \
         else ('vectors_s6b_%s_fm_gaussian_seed_0.npz' % split, 'provenance/s6bev_s6b_%s_fm_gaussian_seed_0.json' % split)
    f7, e7 = os.path.join(RES, v7[0]), v7[1]
    for q in (f7, e7):
        if not os.path.isfile(q): sys.exit('FATAL: missing ' + q)
    if json.load(open(e7)).get('vectors_sha256') != sha(f7): sys.exit('FATAL: %s changed since Step 7' % f7)
    with np.load(f7, allow_pickle=True) as v:
        Y7 = np.asarray(v['true']) - np.asarray(v['C']); cond7 = np.array([str(x) for x in v['cond']])   # float32, as analyse_step13 computes it
    if not np.array_equal(cond7, cond): sys.exit('FATAL: Step 7 target condition order differs from the arms on %s' % split)
    FP7 = np.array([Drug_dose_encoder([smi_all[g_all == c][0]], [float(np.median(dose_all[g_all == c]))], comb_num=1)[0] for c in cond])
    X7 = np.hstack([FP7, Lmat, np.concatenate([FP7 * Lmat[:, [k]] for k in range(len(lv))], axis=1)])
    P7 = np.zeros_like(Y7)                                        # float32 store, as analyse_step13's ridge_pred
    for a_, b_ in GroupKFold(n_splits=5).split(X7, Y7, groups=drugs):
        P7[b_] = Ridge(alpha=ALPHA).fit(X7[a_], Y7[a_]).predict(X7[b_])
    d7 = np.full(n, np.nan)
    for i in range(n):
        d = np.argsort(np.abs(Y7[i]))[-K:]
        if Y7[i][d].std() > 1e-12 and P7[i][d].std() > 1e-12: d7[i] = pearsonr(P7[i][d], Y7[i][d])[0]
    P7 = P7.astype(np.float64); Y7 = Y7.astype(np.float64)
    s7 = json.load(open(os.path.join(RES, 'step7_interaction_baseline.json')))
    rec_med = float(s7['per_split'][split]['marginal']['interaction'])
    if abs(float(np.nanmedian(d7)) - rec_med) > 1e-9: sys.exit('FATAL: ridge-S7 DPC-50 median %.9f differs from the Step 7 record %.9f' % (np.nanmedian(d7), rec_med))
    print('[ridge-S7] reproduced from %s: marginal median DPC-50 %.6f (Step 7 record %.6f)' % (v7[0], np.nanmedian(d7), rec_med))
    tr = ad.read_h5ad('%s/sci_plex_train_%s.h5ad' % (DATA, split)); trc = ad.read_h5ad('%s/sci_plex_train_%s_control.h5ad' % (DATA, split))
    if not np.array_equal(tr.obs_names, trc.obs_names): sys.exit('FATAL: train and control files are not row-aligned')
    gtr = tr.obs['Group'].astype(str).values; utr = np.unique(gtr); Xtr = dense(tr.X); Xtrc = dense(trc.X)
    idx = {c: np.where(gtr == c)[0] for c in utr}
    Ytr = np.array([Xtr[idx[c]].mean(axis=0) - Xtrc[idx[c]].mean(axis=0) for c in utr], dtype=np.float64)
    smi_tr = tr.obs['SMILES'].astype(str).values; dose_tr = tr.obs['dose'].astype(float).values; line_tr = tr.obs['cell_name'].astype(str).values
    if set(smi_tr) & set(drugs): sys.exit('FATAL: a held-out compound appears in the training file')
    FPtr = np.array([Drug_dose_encoder([smi_tr[idx[c]][0]], [float(np.median(dose_tr[idx[c]]))], comb_num=1)[0] for c in utr], dtype=np.float64)
    Ltr = np.array([[float(line_tr[idx[c]][0] == u) for u in lv] for c in utr])
    Xtrf = np.hstack([FPtr, Ltr, np.concatenate([FPtr * Ltr[:, [k]] for k in range(len(lv))], axis=1)])
    Ptr = Ridge(alpha=ALPHA).fit(Xtrf, Ytr).predict(Xfeat)
    print('[ridge-train] fitted on %d training conditions (%d compounds), predicting %d held-out conditions' % (len(utr), len(set(smi_tr)), n))
    del tr, trc, Xtr, Xtrc

    # --- shift baselines: profile vectors C + delta (as score_step14.py)
    deltas = {'ridge_S7': P7, 'ridge_train': Ptr, 'control_paste': np.zeros_like(Y)}
    mr = np.zeros_like(Y)
    for ln in lv:
        ix = np.where(lines == ln)[0]; mr[ix] = Y[ix].mean(axis=0)[None, :]
    deltas['mean_response'] = mr
    Bs = {}                                                        # name -> list over seeds of B (profile vectors)
    jac_ = {}; dpc_ = {}
    for bname, D in deltas.items():
        Bs[bname] = [C + D]
        j, d = thesis_per_cond(true_mean, C + D, C)
        jac_[bname] = [j]; dpc_[bname] = [d]
    for name, L in list(arms.items()) + [('base', base_v)]:
        Bs[name] = [v['B'] for v in L]
        jac_[name] = []; dpc_[name] = []
        for v in L:
            j, d = thesis_per_cond(v['true'], v['B'], v['C'])          # each file's own arrays, as Step 14
            jac_[name].append(j); dpc_[name].append(d)

    # --- reproduction guard against the Step 14 record (thesis_metrics: DE-overlap and DPC-50 mean, seed-mean per condition)
    tm = rec14['per_split'][split]['thesis_metrics']
    for nm in SUBS:
        de = float(np.nanmean(np.nanmean(np.vstack(jac_[nm]), axis=0))); dp = float(np.nanmean(np.nanmean(np.vstack(dpc_[nm]), axis=0)))
        if nm not in tm: sys.exit('FATAL: %s has no entry in the Step 14 record' % nm)
        if abs(de - float(tm[nm]['de_overlap'])) > 1e-9 or abs(dp - float(tm[nm]['dpc_mean'])) > 1e-9:
            sys.exit('FATAL: %s DE-overlap %.9f / DPC-50 %.9f do not reproduce the Step 14 record %.9f / %.9f' % (nm, de, dp, tm[nm]['de_overlap'], tm[nm]['dpc_mean']))
    print('[guard] every submission reproduces the Step 14 record (DE-overlap and DPC-50 mean) to 1e-9')

    # --- MRA (section 3): contexts = (line, dose); norms against the comparator's C
    ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
    ctx_u = np.unique(ctx_key); n_ctx = len(ctx_u)
    ctx_id = np.array([int(np.where(ctx_u == k)[0][0]) for k in ctx_key])
    full = np.arange(n)
    true_norm = np.linalg.norm(true_mean - C, axis=1)
    # baselines: the norm of the delta itself (B - C recomputed from B = C + delta differs from delta at 1e-16 and would
    # break the exact ties that make control-paste and mean-response 0 by construction); arms: ||B - C||
    norms = {nm: ([np.linalg.norm(deltas[nm], axis=1)] if nm in deltas else [np.linalg.norm(B - C, axis=1) for B in Bs[nm]]) for nm in SUBS}
    # the anchor (section 3): five seeded half-splits, as Step 14's replicate anchor
    anc_A, anc_B = [], []
    for r in range(N_REP):
        rr = np.random.RandomState(r)
        halfA, halfB = [], []
        for i in range(n):
            perm = rr.permutation(len(real[i])); h = len(perm) // 2
            halfA.append(real[i][perm[:h]]); halfB.append(real[i][perm[h:2 * h]])
        vA, vB = [], []
        for i in range(n):
            perm = rr.permutation(len(paired[i])); h = len(perm) // 2
            vA.append(paired[i][perm[:h]]); vB.append(paired[i][perm[h:2 * h]])
        profA = np.array([a.mean(axis=0) for a in halfA], np.float64); profB = np.array([b.mean(axis=0) for b in halfB], np.float64)
        cA = np.array([a.mean(axis=0) for a in vA], np.float64); cB = np.array([b.mean(axis=0) for b in vB], np.float64)
        anc_A.append(np.linalg.norm(profA - cA, axis=1)); anc_B.append(np.linalg.norm(profB - cB, axis=1))
    sel_dose = {('dose %g' % d): np.where(doses == d)[0] for d in sorted(set(doses.tolist()))}
    sel_line = {('line %s' % ln): np.where(lines == ln)[0] for ln in lv}
    dsel = dict(sel_dose, **sel_line)

    def mra_all(sel):
        """Every submission's MRA on one selection (arms: per seed, then the mean), the anchor's (mean over the five
        half-splits), and the per-dose / per-line values; the measured norms' ranks per context are shared."""
        pre = s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, n_ctx, sel))
        out = {nm: float(np.mean([s16.mra_pre(x, pre) for x in norms[nm]])) for nm in SUBS}
        out['anchor'] = float(np.mean([s16.mra_pre(a, s16.prep_ctx(b, s16.ctx_split(ctx_id, n_ctx, sel))) for a, b in zip(anc_A, anc_B)]))
        for key, ix in dsel.items():
            s_ = sel[np.isin(sel, ix)]
            pre_s = s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, n_ctx, s_))
            for nm in SUBS:
                out['%s | %s' % (key, nm)] = float(np.mean([s16.mra_pre(x, pre_s) for x in norms[nm]])) if len(s_) else np.nan
        return out

    pre_full = s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, n_ctx, full))
    n_ctx_used = sum(item is not None for item in pre_full)
    mra_seed = {nm: [s16.mra_pre(x, pre_full) for x in norms[nm]] for nm in SUBS}
    V = mra_all(full)
    for nm in SUBS:
        if abs(V[nm] - float(np.mean(mra_seed[nm]))) > 1e-12: sys.exit('FATAL: shared-context MRA differs from the per-seed computation for %s' % nm)
    for nm in ('control_paste', 'mean_response'):
        if abs(V[nm]) > 1e-12: sys.exit('FATAL: %s MRA is %.3g, expected 0 by definition' % (nm, V[nm]))
    anchor_rep = [s16.mra_pre(a, s16.prep_ctx(b, s16.ctx_split(ctx_id, n_ctx, full))) for a, b in zip(anc_A, anc_B)]
    print('[mra] %d contexts (%d with >= %d conditions), %d-%d conditions each | split-half anchor %+.4f (per half-split %s)'
          % (n_ctx, n_ctx_used, MIN_CTX, min(np.bincount(ctx_id)), max(np.bincount(ctx_id)), V['anchor'], [round(x, 4) for x in anchor_rep]))
    for nm in SUBS:
        print('[mra] %-14s %+.4f | per-seed %s | worst seed %+.4f' % (nm, V[nm], [round(x, 4) for x in mra_seed[nm]], min(mra_seed[nm])))
    print('[mra] per dose / per line:')
    for key in dsel:
        print('   %-12s %s' % (key, '  '.join('%s %+.3f' % (nm, V['%s | %s' % (key, nm)]) for nm in SUBS if nm not in ('control_paste', 'mean_response'))))

    # --- bootstrap (section 4): recomputed per replicate on the resampled compounds, one resample for every quantity
    rng = np.random.RandomState(0)
    gs = [np.where(drugs == u)[0] for u in np.unique(drugs)]
    if len(gs) != n_drugs: sys.exit('FATAL: %d compound clusters, expected %d' % (len(gs), n_drugs))
    keys = list(V)
    B = {k: np.full(N_BOOT, np.nan) for k in keys}
    for b in range(N_BOOT):
        sel = np.concatenate([gs[g] for g in rng.randint(0, len(gs), len(gs))])
        got = mra_all(sel)
        for k in keys: B[k][b] = got[k]
    print('[bootstrap] %d replicates (%.0f s)' % (N_BOOT, time.time() - T0))

    con = {}
    def add(key, label, point, dist):
        con[key] = verdict_of(point, dist)
        print('%-11s %-52s %s' % (label, key, fmt(con[key])))
        rows.append(dict(split=split, quantity=key, value=float(point), ci_lo=con[key]['ci'][0], ci_hi=con[key]['ci'][1], verdict=con[key]['verdict']))
    print()
    add('Q1_primary_mra_fm_50_minus_0', 'PRIMARY', V['fm_50'], B['fm_50'])
    add('Q2_coprimary_mra_fm_50_minus_ridge_train', 'CO-PRIMARY', V['fm_50'] - V['ridge_train'], B['fm_50'] - B['ridge_train'])
    for nm in ('base', 'ridge_S7', 'ridge_train', 'guidance_w2', 'attn_cross', 'mlp_ctrl'):
        add('secondary_mra_%s_minus_0' % nm, 'secondary', V[nm], B[nm])
    add('secondary_mra_fm_50_minus_ridge_S7', 'secondary', V['fm_50'] - V['ridge_S7'], B['fm_50'] - B['ridge_S7'])
    for nm in ('guidance_w2', 'attn_cross', 'mlp_ctrl'):
        add('secondary_mra_%s_minus_fm_50' % nm, 'secondary', V[nm] - V['fm_50'], B[nm] - B['fm_50'])
    add('descriptive_mra_anchor_minus_0', 'descriptive', V['anchor'], B['anchor'])
    for key in dsel:
        for nm in SUBS:
            if nm in ('control_paste', 'mean_response'): continue
            add('descriptive_mra_%s_%s_minus_0' % (key.replace(' ', '_'), nm), 'descriptive', V['%s | %s' % (key, nm)], B['%s | %s' % (key, nm)])

    for i in range(n):
        pc_rows.append(dict(split=split, condition=cond[i], line=lines[i], dose=doses[i], drug=drugs[i], true_effect_norm=float(true_norm[i]),
                            **{'%s_effect_norm' % nm: float(np.mean([x[i] for x in norms[nm]])) for nm in SUBS},
                            anchor_norm_A=float(np.mean([a[i] for a in anc_A])), anchor_norm_B=float(np.mean([b[i] for b in anc_B]))))
    for nm in SUBS:
        rows.append(dict(split=split, quantity='value_mra_%s' % nm, value=V[nm], ci_lo=None, ci_hi=None, verdict=None))
    return dict(n_conditions=n, n_compounds=n_drugs, n_genes=int(true_mean.shape[1]), n_contexts=n_ctx, n_contexts_used=n_ctx_used,
                conditions_per_context=[int(np.bincount(ctx_id).min()), int(np.bincount(ctx_id).max())], dose_levels=sorted(set(doses.tolist())), lines=lv,
                cells=dict(stored_min=int(min(n_stored)), stored_max=int(max(n_stored)), distinct_min=int(n_real.min()), distinct_max=int(n_real.max()),
                           recovered=int(recovered), recovered_exact=int(exact), recovered_within_1e5=int(tolerant), n_at_least_min_cells=int(gated.sum())),
                ridge_S7_dpc_median=float(np.nanmedian(d7)), step7_record=rec_med,
                metrics={nm: dict(mra=V[nm], mra_per_seed=mra_seed[nm], mra_worst_seed=float(min(mra_seed[nm])),
                                  per_dose={k: V['%s | %s' % (k, nm)] for k in sel_dose}, per_line={k: V['%s | %s' % (k, nm)] for k in sel_line},
                                  de_overlap=float(tm[nm]['de_overlap']), dpc_mean=float(tm[nm]['dpc_mean'])) for nm in SUBS},
                anchor=dict(mra=V['anchor'], per_half_split=[float(x) for x in anchor_rep]), contrasts=con, inputs=inputs)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default=','.join(ALL_SPLITS))
    a = ap.parse_args()
    splits = [s.strip() for s in a.splits.split(',') if s.strip()]
    for s in splits:
        if s not in ALL_SPLITS: sys.exit('FATAL: unknown split %s' % s)
    final = sorted(splits) == sorted(ALL_SPLITS)
    suffix = '' if final else '_development'
    OUT = os.path.join(RES, 'step17_mra%s.json' % suffix)
    OUT_CSV = os.path.join(RES, 'step17_mra_long%s.csv' % suffix)
    OUT_PC = os.path.join(RES, 'step17_mra_per_condition%s.csv' % suffix)
    for q in (OUT, OUT_CSV, OUT_PC):
        if os.path.exists(q): sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 9: sys.exit('FATAL: prereg incomplete')
    tracked = '%s %s score_step17_sciplex_mra.py score_step16_tahoe.py score_step15_tahoe.py score_step14.py step14_metrics.py' % (PRE, REC14)
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0: sys.exit('FATAL: prereg, record or code not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0: sys.exit('FATAL: prereg, record or code uncommitted')
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    rec14 = json.load(open(REC14))
    print('[prereg] %s committed; code committed; commit %s; Step 14 record %s (verdict %s)' % (PRE, commit[:7], REC14, rec14.get('verdict')))
    s16.mra_self_check()
    print('[self-check] fast_rank == scipy rankdata; mra_pre == the Step 15e Spearman')
    rows, pc_rows, results = [], [], {}
    for split in splits:
        results[split] = score_split(split, rec14, rows, pc_rows)

    cat = None
    if final:
        q1 = [results[s]['contrasts']['Q1_primary_mra_fm_50_minus_0']['verdict'] for s in ALL_SPLITS]
        q2 = [results[s]['contrasts']['Q2_coprimary_mra_fm_50_minus_ridge_train']['verdict'] for s in ALL_SPLITS]
        if all(v == 'above zero' for v in q1) and all(v == 'above zero' for v in q2): cat = '1 SIZE SIGNAL BEYOND LINEAR'
        elif all(v == 'above zero' for v in q1): cat = '2 SIZE SIGNAL'
        elif all(v == 'includes zero' for v in q1): cat = '5 UNRESOLVED'
        elif any(v == 'below zero' for v in q1): cat = '3 REVERSED'
        else: cat = '4 PARTIAL'
        print('\n' + '=' * 72 + '\nPRE-REGISTERED VERDICT: %s\n  Q1 per split: %s\n  Q2 per split: %s\n  Section 7 expected: 2 SIZE SIGNAL\n' % (cat, dict(zip(ALL_SPLITS, q1)), dict(zip(ALL_SPLITS, q2))) + '=' * 72)
    else:
        print('\n[development] %s only: no verdict; the verdict needs all three splits' % ', '.join(splits))
    json.dump(dict(prereg=PRE, prereg_sha256=sha(PRE), final=final, verdict=cat, interval='97.5 % percentile per split, compound-clustered', n_boot=N_BOOT,
                   min_context=MIN_CTX, comparator=COMPARATOR, step14_record_sha256=sha(REC14), per_split=results, commit=commit, seconds=round(time.time() - T0)),
              open(OUT, 'w'), indent=1, allow_nan=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'quantity', 'value', 'ci_lo', 'ci_hi', 'verdict']); w.writeheader(); w.writerows(rows)
    pd.DataFrame(pc_rows).to_csv(OUT_PC, index=False)
    print('\nwrote %s, %s and %s (%.0f s)' % (OUT, OUT_CSV, OUT_PC, time.time() - T0))


if __name__ == '__main__':
    main()
