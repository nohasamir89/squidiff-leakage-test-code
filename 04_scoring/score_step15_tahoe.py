#!/usr/bin/env python3
"""Step 15b scoring under PREREG_step15b_tahoe_model.md (sections 4-6 and 9).

For each requested split (development: drug_split_1 alone; confirmation: all three):
  1. loads the five arms x three seeds from tahoe_results/vectors_s15_<split>_<arm>_seed_<s>.npz (+ per-cell file),
     each checked against provenance/s15ev_<tag>.json (arm, split, seed, guidance, NFE, sha256 of both files);
  2. reproduces the harness's real-cell draw per condition from the Tahoe test file (file order, RandomState(0),
     cap 200) and asserts it equals the stored real cells, so the paired vehicle cells (row-aligned control file,
     Step 15a pairing) are exactly the cells behind every arm's stored population;
  3. builds the four shift baselines of section 3 - ridge-S7 (fingerprint + line + fingerprint x line, alpha 1.0,
     structure-grouped 5-fold CV within the held-out compounds, Step 7's procedure), ridge-train (same features,
     fitted on the training split's condition deltas), mean-response (per-line mean delta over the held-out
     conditions) and control-paste (delta 0) - as the paired vehicle cells plus the delta, clipped at 0 (cap 200,
     section 3 correction of 2026-09-29), with profile vector C + delta;
  4. builds the replicate reference: conditions present on both plate 6 and plate 14, each plate's cells scored
     against the other's (both directions averaged);
  5. scores the thesis's three metrics per condition exactly as Step 14 (DE-overlap Jaccard on the true top-50
     |delta| genes, energy-distance U-statistic on cells, DPC-50) for every arm (seed-mean per condition) and baseline;
  6. computes the pre-registered contrasts with the drug(structure)-clustered percentile bootstrap (5,000 replicates,
     RandomState(0), 97.5% per split): Q1 recipe - base and Q2 recipe - ridge-S7 on DE-overlap (gated); secondaries
     on energy and DPC-50; Q3 guidance - recipe on DE-overlap and energy; Q4 attn_cross - mlp_ctrl on DE-overlap;
     reports absolute values per submission, per seed (worst case), per line, brain vs gut, and the exploratory
     per-condition learnability (section 5 addition of 2026-09-29);
  7. applies section 6 mechanically when all three splits are present. The written categories overlap, so the
     precedence fixed here is: 1 RECIPE GENERALISES -> 2 RECIPE TRANSFERS, SHIFT COMPETITIVE -> 5 UNRESOLVED (either
     gated contrast includes zero on all three splits) -> 3 NO GENERALISATION (the primary below zero on any split)
     -> 4 PARTIAL (the primary above zero on one or two splits, including zero on the rest).
Writes tahoe_results/step15_model_analysis.json, step15_model_long.csv and step15_model_per_condition.csv (final,
all three splits) or the same names with the suffix _development (any other split selection). Never overwrites.
Stops if any input is missing, a hash differs from its provenance, a condition count differs from the data record,
the stored real cells differ from the reproduced draw, or the ridge-S7 construction does not reproduce.
Usage:  python -u score_step15_tahoe.py [--splits drug_split_1[,drug_split_0,...]]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Squidiff.scrna_datasets import Drug_dose_encoder  # noqa: E402  (the harness's fingerprint + dose encoder)

RES = 'tahoe_results'
DATA = os.environ.get('S15_DATA', '/scratch/ni2071/Diffusion/tahoe/datasets_step15')
PRE = 'PREREG_step15b_tahoe_model.md'
RECORD = os.path.join(RES, 'step15_data_record.json')
ALL_SPLITS = ('drug_split_1', 'drug_split_0', 'drug_split_2')          # development first, as section 5
ARMS = ('base', 'recipe', 'guidance_w2', 'attn_cross', 'mlp_ctrl')
BASELINES = ('ridge_S7', 'ridge_train', 'mean_response', 'control_paste')
N_SEEDS, K, N_BOOT, ALPHA, CAP = 3, 50, 5000, 1.0, 200
LO, HI = 1.25, 98.75                                                    # 97.5% per split
BRAIN = ('A-172', 'H4', 'SW 1088')
REP_PLATES = ('6', '14')
rng = np.random.RandomState(0)
T0 = time.time()


# ------------------------------------------------------------------ the thesis's metrics, verbatim from score_step14.py
def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            m.update(c)
    return m.hexdigest()


def jac(a, b):
    i = len(np.intersect1d(a, b))
    return i / (len(a) + len(b) - i)


def thesis_per_cond(true, B, C):
    """DE-overlap Jaccard and DPC-50 per condition from mean vectors (true, predicted B, paired control C)."""
    n = len(true)
    j = np.full(n, np.nan)
    d = np.full(n, np.nan)
    for i in range(n):
        td = true[i] - C[i]
        p = B[i] - C[i]
        g = np.argsort(np.abs(td))[-K:]
        if np.abs(p).std() > 1e-12:
            j[i] = jac(g, np.argsort(np.abs(p))[-K:])
        if td[g].std() > 1e-12 and p[g].std() > 1e-12:
            d[i] = pearsonr(p[g], td[g])[0]
    return j, d


def energy(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return 2 * cdist(X, Y).mean() - pdist(X).mean() - pdist(Y).mean()


def verdict(lo, hi):
    return 'above zero' if lo > 0 else 'below zero' if hi < 0 else 'includes zero'


def dense(X):
    return np.asarray(X.todense() if hasattr(X, 'todense') else X, dtype=np.float32)


def boot(x, drugs, n_drugs, stat=np.nanmean):
    """Drug(structure)-clustered percentile bootstrap of stat over the finite entries of x (97.5%)."""
    ix = np.where(np.isfinite(x))[0]
    gs = [ix[drugs[ix] == u] for u in np.unique(drugs[ix])]
    if len(gs) != n_drugs:
        sys.exit('FATAL: %d drug clusters, expected %d' % (len(gs), n_drugs))
    o = np.array([stat(x[np.concatenate([gs[k] for k in rng.randint(0, len(gs), len(gs))])]) for _ in range(N_BOOT)])
    return float(np.percentile(o, LO)), float(np.percentile(o, HI))


def contrast(x_b, x_a, drugs, n_drugs):
    diff = x_b - x_a
    if np.isfinite(diff).sum() < len(diff) // 2:              # control-paste has no top-50 set, so its DE-overlap is undefined
        return dict(diff=None, ci=None, verdict='undefined (metric not defined for this submission)', half_width=None)
    m = float(np.nanmean(diff))
    lo, hi = boot(diff, drugs, n_drugs)
    return dict(diff=m, ci=[lo, hi], verdict=verdict(lo, hi), half_width=float((hi - lo) / 2))


def fmt(c):
    return '%+.5f  [%+.5f, %+.5f] -> %s' % (c['diff'], *c['ci'], c['verdict']) if c['diff'] is not None else c['verdict']


# ------------------------------------------------------------------ inputs
def check_prov(pv, name, checks):
    for k, v in checks.items():
        if k not in pv:
            sys.exit('FATAL: %s provenance has no key %r' % (name, k))
        got = pv[k]
        if v is None:
            ok = got is None
        elif isinstance(v, (int, float)):
            ok = got is not None and abs(float(got) - float(v)) < 1e-9
        else:
            ok = str(got) == str(v)
        if not ok:
            sys.exit('FATAL: %s provenance %s=%r, expected %r' % (name, k, got, v))


def load_eval(split, arm, seed):
    """One certified evaluation: mean vectors, per-cell populations, provenance record."""
    tag = 's15_%s_%s_seed_%d' % (split, arm, seed)
    f = os.path.join(RES, 'vectors_%s.npz' % tag)
    fpc = os.path.join(RES, 'vectors_%s_pc.npz' % tag)
    pj = os.path.join('provenance', 's15ev_%s.json' % tag)
    for q in (f, fpc, pj):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    pv = json.load(open(pj))
    check_prov(pv, tag, dict(status='completed', arm=arm, split=split, seed=seed, drug_encoding='fingerprint',
                             guidance_w=(2.0 if arm == 'guidance_w2' else None), nfe=(None if arm == 'base' else 50),
                             max_cells_per_cond=CAP))
    if pv.get('vectors_sha256') != sha(f):
        sys.exit('FATAL: %s vectors hash differs from provenance' % tag)
    if pv.get('per_cell_sha256') != sha(fpc):
        sys.exit('FATAL: %s per-cell hash differs from provenance' % tag)
    v = np.load(f, allow_pickle=True)
    p = np.load(fpc, allow_pickle=True)
    out = dict(tag=tag, cond=np.array([str(x) for x in v['cond']]), true=np.asarray(v['true'], np.float64),
               B=np.asarray(v['B'], np.float64), C=np.asarray(v['C'], np.float64),
               pred=[np.asarray(a, np.float32) for a in p['pred']], true_cells=[np.asarray(a, np.float32) for a in p['true']],
               rec=dict(tag=tag, vectors_sha256=pv['vectors_sha256'], per_cell_sha256=pv['per_cell_sha256'],
                        checkpoint_sha256=pv.get('checkpoint_sha256'), training_commit=pv.get('training_commit'),
                        eval_commit=pv.get('commit')))
    if len(out['pred']) != len(out['cond']) or len(out['true_cells']) != len(out['cond']):
        sys.exit('FATAL: %s per-cell length differs from vectors' % tag)
    return out


def parse_cond(c):
    """'<cell_name>_<drug>_<dose/10000>_p<plate>' -> (key without plate, plate)."""
    key, plate = c.rsplit('_p', 1)
    if not plate.isdigit():
        sys.exit('FATAL: cannot parse plate from condition %r' % c)
    return key, plate


def condition_index(groups):
    """Row indices per condition label, in file order, without a per-label scan of the whole file."""
    codes, uniques = pd.factorize(pd.Series(groups), sort=False)
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    return {str(uniques[k]): order[bounds[k]:bounds[k + 1]] for k in range(len(uniques))}


def features(fp, lines, lv):
    L = np.array([[float(l == u) for u in lv] for l in lines])
    return np.hstack([fp, L, np.concatenate([fp * L[:, [k]] for k in range(len(lv))], axis=1)])


def ridge_s7(X, Y, groups):
    """Step 7's within-held-out interaction ridge: structure-grouped 5-fold CV predictions."""
    P = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=5).split(X, Y, groups=groups):
        if set(groups[tr]) & set(groups[te]):
            sys.exit('FATAL: ridge-S7 fold shares a structure between fit and prediction')
        P[te] = Ridge(alpha=ALPHA).fit(X[tr], Y[tr]).predict(X[te])
    return P


def dpc_only(P, Y):
    d = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        g = np.argsort(np.abs(Y[i]))[-K:]
        if Y[i][g].std() > 1e-12 and P[i][g].std() > 1e-12:
            d[i] = pearsonr(P[i][g], Y[i][g])[0]
    return d


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default=','.join(ALL_SPLITS))
    a = ap.parse_args()
    splits = [s.strip() for s in a.splits.split(',') if s.strip()]
    for s in splits:
        if s not in ALL_SPLITS:
            sys.exit('FATAL: unknown split %s' % s)
    final = sorted(splits) == sorted(ALL_SPLITS)
    suffix = '' if final else '_development'
    OUT = os.path.join(RES, 'step15_model_analysis%s.json' % suffix)
    OUT_CSV = os.path.join(RES, 'step15_model_long%s.csv' % suffix)
    OUT_PC = os.path.join(RES, 'step15_model_per_condition%s.csv' % suffix)
    for q in (OUT, OUT_CSV, OUT_PC):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 9:
        sys.exit('FATAL: prereg incomplete')
    if subprocess.run('git ls-files --error-unmatch %s score_step15_tahoe.py' % PRE, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: prereg or scoring code not committed')
    if subprocess.run('git diff --quiet -- %s score_step15_tahoe.py' % PRE, shell=True).returncode != 0:
        sys.exit('FATAL: prereg or scoring code uncommitted')
    if not os.path.isfile(RECORD):
        sys.exit('FATAL: missing %s' % RECORD)
    record = json.load(open(RECORD))
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    print('[prereg] committed; scoring code committed; commit %s; splits %s (%s)' % (commit[:7], splits, 'final' if final else 'development'))

    results, inputs, rows, pc_rows = {}, {}, [], []
    for split in splits:
        print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
        k = split.split('_')[-1]
        rec_split = record['splits'][k]
        expect_n = int(rec_split['n_test_conditions'])
        n_drugs = len(rec_split['held_structures'])

        # --- arms: 5 x 3 certified evaluations
        arms = {arm: [load_eval(split, arm, s) for s in range(N_SEEDS)] for arm in ARMS}
        ref = arms['recipe'][0]
        cond = ref['cond']
        n = len(cond)
        if n != expect_n:
            sys.exit('FATAL: %d conditions in the vectors, data record expects %d' % (n, expect_n))
        for arm, L in arms.items():
            for v in L:
                if not np.array_equal(v['cond'], cond):
                    sys.exit('FATAL: condition order differs in %s' % v['tag'])
                if np.abs(v['C'] - ref['C']).max() > 1e-5:
                    sys.exit('FATAL: paired-control means differ in %s' % v['tag'])
                if np.abs(v['true'] - ref['true']).max() > 1e-5:
                    sys.exit('FATAL: true means differ in %s' % v['tag'])
                for i in range(n):
                    if not np.array_equal(v['true_cells'][i], ref['true_cells'][i]):
                        sys.exit('FATAL: stored real cells differ in %s, condition %d' % (v['tag'], i))
        inputs[split] = {arm: [v['rec'] for v in L] for arm, L in arms.items()}
        true_mean, C = ref['true'], ref['C']
        real = ref['true_cells']

        # --- test data: reproduce the harness's draw, paired vehicles, condition annotations
        te = ad.read_h5ad('%s/tahoe_test_%s.h5ad' % (DATA, split))
        tc = ad.read_h5ad('%s/tahoe_test_%s_control.h5ad' % (DATA, split))
        # Row alignment is positional (Step 15a pairing): the control file holds the vehicle cells under their own
        # barcodes, and each control row carries the Group of the treated condition it is paired to.
        if te.n_obs != tc.n_obs or not np.array_equal(te.obs['Group'].astype(str).values, tc.obs['Group'].astype(str).values):
            sys.exit('FATAL: test and control files are not row-aligned (row count or paired Group differs)')
        Xte, Xtc = dense(te.X), dense(tc.X)
        g_all = te.obs['Group'].astype(str).values
        smi_all = te.obs['SMILES'].astype(str).values
        dose_all = te.obs['dose'].astype(float).values
        line_all = te.obs['cell_name'].astype(str).values
        idx = condition_index(g_all)
        if set(idx) != set(cond):
            sys.exit('FATAL: the test file conditions differ from the vectors')
        paired, n_cells = [], []
        drugs = np.empty(n, dtype=object)
        lines = np.empty(n, dtype=object)
        plates = np.empty(n, dtype=object)
        doses = np.zeros(n)
        for i, c in enumerate(cond):
            ix = idx[c]
            sel = np.arange(len(ix))
            if len(sel) > CAP:
                sel = np.random.RandomState(0).choice(sel, CAP, replace=False)      # the harness's draw, verbatim
            if not np.array_equal(Xte[ix][sel], real[i]):
                sys.exit('FATAL: reproduced real-cell draw differs from the stored cells for %s' % c)
            paired.append(Xtc[ix][sel])
            n_cells.append(len(ix))
            drugs[i] = smi_all[ix[0]]
            lines[i] = line_all[ix[0]]
            plates[i] = parse_cond(c)[1]
            doses[i] = float(np.median(dose_all[ix]))
        drugs = drugs.astype(str)
        lines = lines.astype(str)
        plates = plates.astype(str)
        if len(np.unique(drugs)) != n_drugs:
            sys.exit('FATAL: %d held-out structures among the test conditions, record says %d' % (len(np.unique(drugs)), n_drugs))
        # The record lists held-out structures by id (S0003, ...) and held-out drugs by name; the file carries SMILES and the
        # drug name inside Group (<cell_name>_<drug>_<dose/10000>_p<plate>; drug names carry no underscore), so the
        # identity check is on names, normalised through the record's name_map; the structure COUNT is checked above.
        name_map = record.get('name_map', {})
        names = set()
        for i, c in enumerate(cond):
            if not c.startswith(lines[i] + '_'):
                sys.exit('FATAL: condition %r does not start with its cell line %r' % (c, lines[i]))
            names.add(c[len(lines[i]) + 1:].rsplit('_', 2)[0])
        held = set(name_map.get(x, x) for x in rec_split['held_drugs'])
        if names != held:
            sys.exit('FATAL: held-out drug names in the test file differ from the record (%d vs %d), e.g. %s'
                     % (len(names), len(held), sorted(names ^ held)[:3]))
        lv = sorted(set(str(x) for x in lines))
        print('[design] %d conditions | %d held-out structures | %d lines %s | plates %s | cells per condition %d-%d (stored cap %d)'
              % (n, n_drugs, len(lv), lv, sorted(set(str(x) for x in plates), key=int), min(n_cells), max(n_cells), CAP))

        # --- ridge-S7 and ridge-train
        FP = np.array([Drug_dose_encoder([drugs[i]], [doses[i]], comb_num=1)[0] for i in range(n)], dtype=np.float64)
        X = features(FP, lines, lv)
        Y = true_mean - C
        P7 = ridge_s7(X, Y, drugs)
        if not np.array_equal(P7, ridge_s7(X, Y, drugs)):
            sys.exit('FATAL: ridge-S7 construction does not reproduce')
        d7 = dpc_only(P7, Y)
        print('[ridge-S7] %d features, %d conditions: DPC-50 mean %.6f median %.6f (structure-grouped 5-fold CV within the held-out compounds)'
              % (X.shape[1], n, np.nanmean(d7), np.nanmedian(d7)))
        tr = ad.read_h5ad('%s/tahoe_train_%s.h5ad' % (DATA, split))
        trc = ad.read_h5ad('%s/tahoe_train_%s_control.h5ad' % (DATA, split))
        if tr.n_obs != trc.n_obs or not np.array_equal(tr.obs['Group'].astype(str).values, trc.obs['Group'].astype(str).values):
            sys.exit('FATAL: train and control files are not row-aligned (row count or paired Group differs)')
        gtr = tr.obs['Group'].astype(str).values
        smi_tr = tr.obs['SMILES'].astype(str).values
        dose_tr = tr.obs['dose'].astype(float).values
        line_tr = tr.obs['cell_name'].astype(str).values
        if set(smi_tr) & set(drugs):
            sys.exit('FATAL: a held-out structure appears in the training file')
        codes, utr = pd.factorize(pd.Series(gtr), sort=False)
        Xtr, Xtrc = dense(tr.X), dense(trc.X)
        order = np.argsort(codes, kind='stable')
        bounds = np.searchsorted(codes[order], np.arange(len(utr)))
        cnt = np.bincount(codes, minlength=len(utr)).astype(np.float64)
        Ytr = np.add.reduceat((Xtr[order].astype(np.float64) - Xtrc[order].astype(np.float64)), bounds, axis=0) / cnt[:, None]   # per-condition mean(treated) - mean(paired control)
        first = order[bounds]                                                   # first row of each condition, in code order
        dose_sorted = dose_tr[order]
        ends = np.append(bounds[1:], len(order))
        dose_med = np.array([float(np.median(dose_sorted[bounds[k]:ends[k]])) for k in range(len(utr))])
        FPtr = np.array([Drug_dose_encoder([smi_tr[j]], [dose_med[k]], comb_num=1)[0] for k, j in enumerate(first)], dtype=np.float64)
        Xtrf = features(FPtr, line_tr[first], lv)
        Ptr = Ridge(alpha=ALPHA).fit(Xtrf, Ytr).predict(X)
        print('[ridge-train] fitted on %d training conditions (%d structures), predicting %d held-out conditions' % (len(utr), len(set(smi_tr)), n))
        del tr, trc, Xtr, Xtrc, Ytr, Xtrf

        # --- shift baselines: profile C + delta; cells = paired vehicles + delta, clipped at 0
        mr = np.zeros_like(Y)
        for ln in lv:
            ixl = np.where(lines == ln)[0]
            mr[ixl] = Y[ixl].mean(axis=0)[None, :]
        deltas = {'ridge_S7': P7, 'ridge_train': Ptr, 'mean_response': mr, 'control_paste': np.zeros_like(Y)}
        subs = {}
        for bname, D in deltas.items():
            cells = [np.clip(paired[i] + D[i][None, :], 0, None) for i in range(n)]
            clipped = float(np.mean([(paired[i] + D[i][None, :] < 0).mean() for i in range(n)]))
            j, d = thesis_per_cond(true_mean, C + D, C)
            en = np.array([energy(cells[i], real[i]) for i in range(n)])
            subs[bname] = [dict(jac=j, dpc=d, en=en)]
            print('[baseline] %-14s DE-overlap %.4f  energy %.4f  DPC-50 mean %.4f  median %.4f  (entries clipped at 0: %.4f%%)'
                  % (bname, np.nanmean(j), en.mean(), np.nanmean(d), np.nanmedian(d), 100 * clipped))
        for arm, L in arms.items():
            subs[arm] = []
            for v in L:
                j, d = thesis_per_cond(v['true'], v['B'], v['C'])
                en = np.array([energy(v['pred'][i], v['true_cells'][i]) for i in range(n)])
                subs[arm].append(dict(jac=j, dpc=d, en=en))
                print('[arm] %-14s seed %d  DE-overlap %.4f  energy %.4f  DPC-50 mean %.4f  median %.4f'
                      % (arm, L.index(v), np.nanmean(j), en.mean(), np.nanmean(d), np.nanmedian(d)))
        cm = lambda name, key: np.nanmean(np.vstack([x[key] for x in subs[name]]), axis=0)     # seed-mean per condition

        # --- replicate reference: plate 6 vs plate 14, both directions
        keyed = {}
        for i, c in enumerate(cond):
            key, plate = parse_cond(c)
            if plate in REP_PLATES:
                keyed.setdefault(key, {})[plate] = i
        pairs = [(v['6'], v['14']) for v in keyed.values() if '6' in v and '14' in v]
        rj, rd, re_ = [], [], []
        for i6, i14 in pairs:
            for a_, b_ in ((i6, i14), (i14, i6)):
                j, d = thesis_per_cond(true_mean[[b_]], true_mean[[a_]] - C[a_] + C[b_], C[[b_]])   # a's delta offered as the prediction of b
                rj.append(j[0]); rd.append(d[0])
            re_.append(energy(real[i6], real[i14]))
        replicate = dict(n_pairs=len(pairs), de_overlap=float(np.nanmean(rj)) if pairs else None,
                         energy=float(np.mean(re_)) if pairs else None,
                         dpc_mean=float(np.nanmean(rd)) if pairs else None, dpc_median=float(np.nanmedian(rd)) if pairs else None)
        print('[replicate] plate 6 vs 14: %d condition pairs | DE-overlap %s  energy %s  DPC-50 mean %s median %s'
              % (len(pairs), *[('%.4f' % replicate[q]) if replicate[q] is not None else 'n/a' for q in ('de_overlap', 'energy', 'dpc_mean', 'dpc_median')]))

        # --- exploratory learnability per condition (section 5 addition 2026-09-29): drug shift / (drug shift + between-line shift)
        learn = np.zeros(n)
        other_by_line = {ln: C[lines != ln].mean(axis=0) for ln in lv}
        for i in range(n):
            other = other_by_line[lines[i]]
            a_ = float(((true_mean[i] - C[i]) ** 2).sum())
            b_ = float(((C[i] - other) ** 2).sum())
            learn[i] = a_ / (a_ + b_) if (a_ + b_) > 0 else np.nan
        rho_de = spearmanr(learn, cm('recipe', 'jac'), nan_policy='omit')
        rho_en = spearmanr(learn, cm('control_paste', 'en') - cm('recipe', 'en'), nan_policy='omit')
        learnability = dict(mean=float(np.nanmean(learn)), median=float(np.nanmedian(learn)),
                            spearman_vs_recipe_de_overlap=[float(rho_de[0]), float(rho_de[1])],
                            spearman_vs_energy_gain_over_control_paste=[float(rho_en[0]), float(rho_en[1])])
        print('[learnability] mean %.4f median %.4f | Spearman vs recipe DE-overlap rho %.4f (p %.2e) | vs energy gain over control-paste rho %.4f (p %.2e)'
              % (learnability['mean'], learnability['median'], *learnability['spearman_vs_recipe_de_overlap'], *learnability['spearman_vs_energy_gain_over_control_paste']))

        # --- absolute values: per submission (seed-mean per condition), per seed, per line, brain vs gut
        def summ(v_jac, v_en, v_dpc):
            return dict(de_overlap=float(np.nanmean(v_jac)) if np.isfinite(v_jac).any() else None, energy=float(np.nanmean(v_en)),
                        dpc_mean=float(np.nanmean(v_dpc)), dpc_median=float(np.nanmedian(v_dpc)))
        metrics = {}
        for name in subs:
            m_ = summ(cm(name, 'jac'), cm(name, 'en'), cm(name, 'dpc'))
            m_['per_seed'] = [summ(x['jac'], x['en'], x['dpc']) for x in subs[name]]
            m_['per_line'] = {}
            for ln in lv:
                ixl = np.where(lines == ln)[0]
                m_['per_line'][ln] = summ(cm(name, 'jac')[ixl], cm(name, 'en')[ixl], cm(name, 'dpc')[ixl])
                for key_, val_ in m_['per_line'][ln].items():
                    rows.append(dict(split=split, line=ln, submission=name, metric=key_, value=val_))
            for key_ in ('de_overlap', 'energy', 'dpc_mean', 'dpc_median'):
                rows.append(dict(split=split, line='ALL', submission=name, metric=key_, value=m_[key_]))
            ib = np.isin(lines, BRAIN)
            m_['brain_minus_gut'] = {key_: (float(np.nanmean(cm(name, mk)[ib]) - np.nanmean(cm(name, mk)[~ib])) if np.isfinite(cm(name, mk)).any() else None)
                                     for key_, mk in (('de_overlap', 'jac'), ('energy', 'en'), ('dpc_mean', 'dpc'))}
            metrics[name] = m_
            print('[abs] %-14s DE-overlap %s  energy %.4f  DPC-50 mean %.4f median %.4f | per-seed DE-overlap %s | brain-gut DE %s'
                  % (name, ('%.4f' % m_['de_overlap']) if m_['de_overlap'] is not None else 'n/a', m_['energy'], m_['dpc_mean'], m_['dpc_median'],
                     [None if s_['de_overlap'] is None else round(s_['de_overlap'], 4) for s_ in m_['per_seed']],
                     None if m_['brain_minus_gut']['de_overlap'] is None else round(m_['brain_minus_gut']['de_overlap'], 4)))
        for i, c in enumerate(cond):
            r = dict(split=split, condition=c, line=lines[i], plate=plates[i], structure=drugs[i], dose_nM=doses[i], n_cells=n_cells[i], learnability=learn[i])
            for name in subs:
                r['%s_de_overlap' % name] = cm(name, 'jac')[i]
                r['%s_energy' % name] = cm(name, 'en')[i]
                r['%s_dpc50' % name] = cm(name, 'dpc')[i]
            pc_rows.append(r)

        # --- the pre-registered contrasts
        con = {}
        con['Q1_primary_de_recipe_minus_base'] = contrast(cm('recipe', 'jac'), cm('base', 'jac'), drugs, n_drugs)
        con['Q2_coprimary_de_recipe_minus_ridgeS7'] = contrast(cm('recipe', 'jac'), cm('ridge_S7', 'jac'), drugs, n_drugs)
        print('\nQ1 PRIMARY     DE-overlap, recipe minus base       %s' % fmt(con['Q1_primary_de_recipe_minus_base']))
        print('Q2 CO-PRIMARY  DE-overlap, recipe minus ridge-S7   %s' % fmt(con['Q2_coprimary_de_recipe_minus_ridgeS7']))
        for key_, lab in (('en', 'energy'), ('dpc', 'DPC-50')):
            con['secondary_%s_recipe_minus_base' % key_] = contrast(cm('recipe', key_), cm('base', key_), drugs, n_drugs)
            print('secondary   %-8s recipe minus base            %s' % (lab, fmt(con['secondary_%s_recipe_minus_base' % key_])))
        for bname in BASELINES:
            for key_, lab in (('jac', 'DE-overlap'), ('en', 'energy'), ('dpc', 'DPC-50')):
                if bname == 'ridge_S7' and key_ == 'jac':
                    continue
                c_ = contrast(cm('recipe', key_), cm(bname, key_), drugs, n_drugs)
                con['secondary_%s_recipe_minus_%s' % (key_, bname)] = c_
                print('secondary   %-10s recipe minus %-14s %s' % (lab, bname, fmt(c_)))
        for key_, lab in (('jac', 'DE-overlap'), ('en', 'energy'), ('dpc', 'DPC-50')):
            c_ = contrast(cm('guidance_w2', key_), cm('recipe', key_), drugs, n_drugs)
            con['Q3_%s_guidance_w2_minus_recipe' % key_] = c_
            print('Q3          %-10s guidance w=2 minus recipe        %s' % (lab, fmt(c_)))
        for key_, lab in (('jac', 'DE-overlap'), ('en', 'energy'), ('dpc', 'DPC-50')):
            c_ = contrast(cm('attn_cross', key_), cm('mlp_ctrl', key_), drugs, n_drugs)
            con['Q4_%s_attn_cross_minus_mlp_ctrl' % key_] = c_
            print('Q4 explor.  %-10s attn_cross minus mlp_ctrl        %s' % (lab, fmt(c_)))
        results[split] = dict(n_conditions=n, n_structures=n_drugs, lines=lv, cells_per_condition=[int(min(n_cells)), int(max(n_cells))],
                              ridge_S7_dpc=dict(mean=float(np.nanmean(d7)), median=float(np.nanmedian(d7)), n_features=int(X.shape[1])),
                              metrics=metrics, replicate=replicate, learnability=learnability, contrasts=con)
        del te, tc, Xte, Xtc, arms, subs

    # ------------------------------------------------------------------ section 6, when all three splits are present
    cat = None
    if final:
        q1 = {s: results[s]['contrasts']['Q1_primary_de_recipe_minus_base']['verdict'] for s in ALL_SPLITS}
        q2 = {s: results[s]['contrasts']['Q2_coprimary_de_recipe_minus_ridgeS7']['verdict'] for s in ALL_SPLITS}
        v1, v2 = list(q1.values()), list(q2.values())
        if all(v == 'above zero' for v in v1) and all(v == 'above zero' for v in v2):
            cat = 'RECIPE GENERALISES'
        elif all(v == 'above zero' for v in v1):
            cat = 'RECIPE TRANSFERS, SHIFT COMPETITIVE'
        elif all(v == 'includes zero' for v in v1) or all(v == 'includes zero' for v in v2):
            cat = 'UNRESOLVED'                     # a gated contrast includes zero on all three splits
        elif 'below zero' in v1:
            cat = 'NO GENERALISATION'              # the primary is contradicted on at least one split
        elif 1 <= v1.count('above zero') <= 2:
            cat = 'PARTIAL'                        # the primary resolved on one or two splits only, including zero on the rest
        else:
            cat = 'NO GENERALISATION'
        print('\n' + '=' * 72 + '\nPRE-REGISTERED VERDICT (section 6): %s\n  Q1 recipe - base, DE-overlap per split: %s\n  Q2 recipe - ridge-S7, DE-overlap per split: %s\n'
              % (cat, q1, q2) + '  Section 7 expected: RECIPE TRANSFERS, SHIFT COMPETITIVE.\n' + '=' * 72)
    else:
        print('\n[development] %s scored; section 6 is applied only when all three splits are present' % ', '.join(splits))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(pc_rows).to_csv(OUT_PC, index=False)
    json.dump(dict(prereg=PRE, final=final, verdict=cat, splits=splits, interval='97.5% percentile per split, structure-clustered',
                   n_boot=N_BOOT, cap=CAP, n_seeds=N_SEEDS, per_split=results, inputs=inputs,
                   data_record_sha256=sha(RECORD), commit=commit, seconds=round(time.time() - T0)),
              open(OUT, 'w'), indent=2, allow_nan=True)
    print('\nwrote %s, %s and %s  (%.0f s)' % (OUT, OUT_CSV, OUT_PC, time.time() - T0))


if __name__ == '__main__':
    main()
