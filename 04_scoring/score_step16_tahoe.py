#!/usr/bin/env python
"""Step 16 scoring: the target-aware arms against the recipe on the frozen Tahoe test sets.

Pre-registration: PREREG_step16_tahoe_targets.md (sections 4-6, 9), committed with the code before any Step 16 model
exists. This script does only what that document says, by the Step 15b/15c/15e code paths:

  1. guards: the pre-registration, this scorer and the scorers it imports committed and clean; the Step 15b, 15c and
     15e records present and committed; every Step 15b array (mean vectors) and every Step 16 array (mean vectors and
     per-cell populations) hash-checked against its provenance; the Step 16 provenance names the arm's encoding;
  2. the four shift baselines rebuilt by the Step 15b code path and asserted to reproduce the Step 15b per-condition
     record to 1e-9 (DE-overlap, DPC-50), with every Step 15b arm; PDS of all nine Step 15b submissions asserted to
     reproduce the Step 15c per-condition record to 1e-9 (so the candidate sets are Step 15c's);
  3. the new arms: DE-overlap, DPC-50 (section 4.2), energy distance against the stored real cells (asserted identical
     to the recipe's), PDS on the Step 15c sets (4.1), per seed and seed-averaged per condition;
  4. MRA (4.4): per (line, dose) context with >= 10 conditions, the Spearman between the norm of the predicted effect
     and the norm of the measured effect; mean over contexts; arms per seed then averaged; control-paste and
     mean-response asserted 0; replicate ceiling on the conditions with a plate 6 / 14 copy;
  5. the less-unseen strata (4.5) from the encoder's own lookup, and — dated addition of 2026-10-08, written after the
     development run on drug_split_1 and before any confirmation result exists, exploratory and not gated — the
     stratum gap of 4.5: an arm's gain over the recipe among drugs whose targets were seen in training minus its gain
     among drugs with no curated targets at all (whose input is the fingerprint plus 545 zeros, so they differ from the
     recipe in width alone), with the same drug-clustered bootstrap; the cross-plate PDS facet (Step 15c section 8) for
     the recipe and the new arms;
  6. the pooled PRISM utility (4.6): the Step 15e computation per split (U_s asserted equal to the Step 15e record for
     every Step 15e submission) extended to the new arms, with a per-split drug bootstrap; when all three splits are
     present, readings A and B pooled over the 85 labelled drugs with the Step 15e stratified bootstrap (the recipe's
     interval asserted to reproduce the record);
  7. the section-5 contrasts (structure-clustered percentile bootstrap, 5,000 replicates, RandomState(0), 97.5 % per
     split; MRA and utility recomputed on the resampled drugs) and section 6 when all three splits are present.

Usage: python -u score_step16_tahoe.py --splits drug_split_1      (development run first; no --splits = final)
Outputs: tahoe_results/step16_analysis[_development].json, step16_long[_development].csv,
         step16_per_condition[_development].csv. Never overwrites.
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
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402  loaders, metrics, baselines, bootstrap
import score_step15c_tahoe as s15c                    # noqa: E402  pds_sets, load_vectors, close
import score_step15d_utility as s15d                  # noqa: E402  PRISM helpers, fit_ridge, U_of
import score_step15e_pooled as s15e                   # noqa: E402  spearman_fast, U_fast
from Squidiff.scrna_datasets import _load_target_map, TARGET_VOCAB_SIZE, DRUG_ENCODER_DIMS   # noqa: E402  the encoder's own lookup

PRE = 'PREREG_step16_tahoe_targets.md'
PC15 = s15d.PC15
PC15C = os.path.join(s15.RES, 'step15c_pds_per_condition.csv')
REC15B = os.path.join(s15.RES, 'step15_model_analysis.json')
REC15C = os.path.join(s15.RES, 'step15c_pds.json')
REC15E = os.path.join(s15.RES, 'step15e_pooled.json')
TARGETS = os.path.join(s15.RES, 'step15_targets_curated_hgnc.csv')
NEW_ARMS = ('targets', 'targets_shuf')
ENC_OF = {'targets': 'fp_targets', 'targets_shuf': 'fp_targets_shuf'}
OLD_ARMS = tuple(s15.ARMS)
SUBS = list(s15.BASELINES) + list(OLD_ARMS) + list(NEW_ARMS)
MIN_CTX = 10
N_BOOT, LO, HI = s15.N_BOOT, s15.LO, s15.HI
STRATA = ('no targets', 'targets, none seen', 'targets, >=1 seen')
UTIL_SUBS = ('recipe', 'ridge_train', 'ceiling', 'qsar', 'effect_size_measured', 'targets', 'targets_shuf')
T0 = time.time()


# ------------------------------------------------------------------ inputs
def load_eval16(split, arm, seed):
    """One certified Step 16 evaluation: mean vectors, per-cell populations, provenance (encoding checked by name)."""
    tag = 's16_%s_%s_seed_%d' % (split, arm, seed)
    f = os.path.join(s15.RES, 'vectors_%s.npz' % tag)
    fpc = os.path.join(s15.RES, 'vectors_%s_pc.npz' % tag)
    pj = os.path.join('provenance', 's16ev_%s.json' % tag)
    for q in (f, fpc, pj):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    pv = json.load(open(pj))
    s15.check_prov(pv, tag, dict(status='completed', arm=arm, split=split, seed=seed, drug_encoding=ENC_OF[arm],
                                 drug_dimension=DRUG_ENCODER_DIMS[ENC_OF[arm]], encoder_arm='mlp', guidance_w=None, nfe=50,
                                 max_cells_per_cond=s15.CAP))
    if pv.get('vectors_sha256') != s15.sha(f):
        sys.exit('FATAL: %s vectors hash differs from provenance' % tag)
    if pv.get('per_cell_sha256') != s15.sha(fpc):
        sys.exit('FATAL: %s per-cell hash differs from provenance' % tag)
    v = np.load(f, allow_pickle=True)
    p = np.load(fpc, allow_pickle=True)
    out = dict(tag=tag, cond=np.array([str(x) for x in v['cond']]), true=np.asarray(v['true'], np.float64),
               B=np.asarray(v['B'], np.float64), C=np.asarray(v['C'], np.float64),
               pred=[np.asarray(a, np.float32) for a in p['pred']], true_cells=[np.asarray(a, np.float32) for a in p['true']],
               rec=dict(tag=tag, vectors_sha256=pv['vectors_sha256'], per_cell_sha256=pv['per_cell_sha256'],
                        checkpoint_sha256=pv.get('checkpoint_sha256'), training_commit=pv.get('training_commit'),
                        eval_commit=pv.get('commit'), shuffled_order_sha256_prefix=pv.get('shuffled_order_sha256_prefix'),
                        targets_table_sha256=pv.get('targets_table_sha256'), data_record_sha256=pv.get('data_record_sha256')))
    if len(out['pred']) != len(out['cond']) or len(out['true_cells']) != len(out['cond']):
        sys.exit('FATAL: %s per-cell length differs from vectors' % tag)
    return out


def rebuild(split, lines, drugs, doses, lv, Y):
    """The Step 15b shift baselines by the Step 15b code path (verbatim from score_step15e_pooled.main), with the
    training-side objects the utility instrument needs."""
    n = len(lines)
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
    train = dict(utr=utr, smi_c=smi_c, line_c=line_c, dose_med=dose_med, Ytr=Ytr, smiles=sorted(set(smi_tr)))
    return deltas, train


def spearman_fast(x, y):
    return s15e.spearman_fast(np.asarray(x, np.float64), np.asarray(y, np.float64))


def fast_rank(x):
    """Average ranks (scipy.stats.rankdata, ties averaged), vectorised: the MRA bootstrap needs millions of these and
    rankdata's per-call overhead is 110 us against 16 us here. Checked against rankdata in mra_self_check()."""
    n = x.size
    o = np.argsort(x, kind='stable')
    xs = x[o]
    neq = np.empty(n, dtype=bool)
    neq[0] = True
    neq[1:] = xs[1:] != xs[:-1]
    st = np.flatnonzero(neq)
    en = np.append(st[1:], n)
    r = np.empty(n)
    r[o] = np.repeat((st + en - 1) / 2.0 + 1.0, en - st)
    return r


def prep_ctx(true_norm, ctx_sel):
    """Per context: the centred ranks of the measured norms and their length, or None where the context is too small or
    the measured norms are constant (no ranking: the context contributes nothing)."""
    out = []
    for m in ctx_sel:
        if len(m) < MIN_CTX or np.ptp(true_norm[m]) == 0:
            out.append(None)
            continue
        ry = fast_rank(true_norm[m])
        ry = ry - ry.mean()
        out.append((m, ry, float(np.sqrt((ry * ry).sum()))))
    return out


def mra_pre(pred_norm, pre):
    """MRA given the prepared contexts: mean over contexts of the Spearman between predicted and measured norms; a
    submission constant within a context scores 0 there, as score_step15d_utility.spearman0 defines."""
    rhos = []
    for item in pre:
        if item is None:
            continue
        m, ry, dy = item
        x = pred_norm[m]
        if np.ptp(x) == 0:
            rhos.append(0.0)
            continue
        rx = fast_rank(x)
        rx = rx - rx.mean()
        dx = np.sqrt((rx * rx).sum())
        rhos.append(float((rx * ry).sum() / (dx * dy)) if dx > 0 and dy > 0 else 0.0)
    return float(np.mean(rhos)) if rhos else np.nan


def ctx_split(ctx_id, n_ctx, sel):
    """The selected conditions of each context, in one pass over sel (not one pass per submission)."""
    c_of = ctx_id[sel]
    return [sel[c_of == c] for c in range(n_ctx)]


def mra_of(pred_norm, true_norm, ctx_id, n_ctx, sel):
    """MRA for one prediction over the selection sel (condition indices with multiplicity)."""
    return mra_pre(pred_norm, prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, sel)))


def mra_self_check():
    """fast_rank equals scipy's rankdata (ties included) and mra_pre equals the same quantity computed with
    score_step15e_pooled.spearman_fast, which Step 15e checked against the Step 15d scipy computation."""
    from scipy.stats import rankdata
    r = np.random.RandomState(16)
    for n in (5, 12, 30, 61):
        for t in range(50):
            x = r.rand(n)
            if t % 3:
                x = np.round(x, 1)                       # ties
            if t % 7 == 0:
                x = x[r.randint(0, n, n)]                # repeated entries, as a bootstrap draw produces
            if np.abs(fast_rank(x) - rankdata(x)).max() > 1e-12:
                sys.exit('FATAL: fast_rank differs from scipy rankdata')
    for case in range(20):
        n, k = 120, 4
        ctx_id = r.randint(0, k, n)
        true_norm, pred_norm = r.rand(n), r.rand(n)
        sel = r.randint(0, n, n)
        got = mra_of(pred_norm, true_norm, ctx_id, k, sel)
        rhos = []
        for c in range(k):
            m = sel[ctx_id[sel] == c]
            if len(m) >= MIN_CTX:
                rhos.append(s15e.spearman_fast(pred_norm[m], true_norm[m]))
        want = float(np.mean(rhos)) if rhos else np.nan
        if abs(got - want) > 1e-12:
            sys.exit('FATAL: mra_pre differs from the reference computation (%r vs %r)' % (got, want))
    print('[self-check] fast_rank == scipy rankdata (ties and repeats); MRA == the Step 15e Spearman routine')


def verdict_of(point, dist):
    lo, hi = float(np.nanpercentile(dist, LO)), float(np.nanpercentile(dist, HI))
    return dict(diff=float(point), ci=[lo, hi], verdict=s15.verdict(lo, hi), half_width=float((hi - lo) / 2))


def contrast_pc(x_b, x_a, drugs):
    """Per-condition values fixed under resampling (Steps 14-15c): the Step 15c wrapper of score_step15_tahoe.contrast."""
    return s15c.contrast(x_b, x_a, drugs)


def stratum_gap(diff, drugs, stratum, hi, lo):
    """Exploratory (section 4.5, dated addition of 2026-10-08, before any confirmation result exists): the gain of an arm
    over the recipe in stratum `hi` minus its gain in stratum `lo`, with the same drug-clustered percentile bootstrap as
    every other contrast (5,000 replicates, RandomState(0), 97.5 %).

    Why: a drug with no curated targets enters the target-aware arm as the published fingerprint followed by 545 zeros,
    so its input differs from the recipe's only in WIDTH; a drug whose targets occur among the training drugs' targets
    enters with a bag the model has seen before. If the arm's gain comes from target information, the gain must be
    larger in 'targets, >= 1 seen' than in 'no targets'. This statistic is that difference, with an interval; the two
    stratum means on their own cannot say whether they differ.

    A drug belongs to exactly one stratum (asserted), so the bootstrap resamples drugs and recomputes both stratum means
    from the resampled set; a replicate that draws no drug from one of the two strata contributes NaN and is dropped by
    the percentile (their number is recorded)."""
    ok = np.isfinite(diff)
    groups, labs = [], []
    for u in np.unique(drugs[ok]):
        m = np.where((drugs == u) & ok)[0]
        s = set(stratum[m])
        if len(s) != 1:
            sys.exit('FATAL: drug %s spans %d strata; the stratum must be a property of the drug' % (u, len(s)))
        groups.append(m)
        labs.append(s.pop())
    labs = np.array(labs, dtype=object)
    if not (labs == hi).any() or not (labs == lo).any():
        return dict(diff=None, ci=None, verdict='undefined (a stratum has no drug on this split)',
                    n_drugs_hi=int((labs == hi).sum()), n_drugs_lo=int((labs == lo).sum()))

    def stat(sel):
        a = [groups[g] for g in sel if labs[g] == hi]
        b = [groups[g] for g in sel if labs[g] == lo]
        if not a or not b:
            return np.nan
        return float(np.mean(diff[np.concatenate(a)]) - np.mean(diff[np.concatenate(b)]))

    point = stat(np.arange(len(groups)))
    rng = np.random.RandomState(0)
    dist = np.array([stat(rng.randint(0, len(groups), len(groups))) for _ in range(N_BOOT)])
    lo_, hi_ = float(np.nanpercentile(dist, LO)), float(np.nanpercentile(dist, HI))
    return dict(diff=point, ci=[lo_, hi_], verdict=s15.verdict(lo_, hi_), half_width=float((hi_ - lo_) / 2),
                n_drugs_hi=int((labs == hi).sum()), n_drugs_lo=int((labs == lo).sum()),
                n_replicates_dropped=int(np.isnan(dist).sum()))


def fmt(c):
    return s15.fmt(c)


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
    OUT = os.path.join(s15.RES, 'step16_analysis%s.json' % suffix)
    OUT_CSV = os.path.join(s15.RES, 'step16_long%s.csv' % suffix)
    OUT_PC = os.path.join(s15.RES, 'step16_per_condition%s.csv' % suffix)
    for q in (OUT, OUT_CSV, OUT_PC):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 9:
        sys.exit('FATAL: prereg incomplete')
    tracked = ('%s score_step16_tahoe.py score_step15_tahoe.py score_step15c_tahoe.py score_step15d_utility.py score_step15e_pooled.py '
               'step14_metrics.py Squidiff/scrna_datasets.py %s %s %s %s %s %s %s %s'
               % (PRE, PC15, PC15C, REC15B, REC15C, REC15E, s15d.INV_D, TARGETS, s15.RECORD))
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers, encoder or records not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers, encoder or records uncommitted')
    for q in (s15d.DRC, s15d.CLI, s15d.PRISM_REC):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    record = json.load(open(s15.RECORD))
    name_map = record.get('name_map', {})
    inv_map = {v: k for k, v in name_map.items()}
    rec15b, rec15c, rec15e = json.load(open(REC15B)), json.load(open(REC15C)), json.load(open(REC15E))
    for nm, r in (('15b', rec15b), ('15c', rec15c)):
        if not r.get('final'):
            sys.exit('FATAL: the Step %s record is not the final one' % nm)
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    print('[prereg] committed; scorer committed; commit %s; splits %s (%s)' % (commit[:7], splits, 'final' if final else 'development'))
    s15c.self_check()
    mra_self_check()
    TM = _load_target_map()                                             # the encoder's own SMILES -> bag lookup
    if TM['table'] != os.path.abspath(TARGETS) or TM['record'] != os.path.abspath(s15.RECORD):
        sys.exit('FATAL: the encoder reads %s / %s, the scorer tracks %s / %s' % (TM['table'], TM['record'], TARGETS, s15.RECORD))
    pc15, pc15c = pd.read_csv(PC15), pd.read_csv(PC15C)

    # ------------------------------------------------------------------ PRISM labels, as Step 15d / 15e
    prism_rec = json.load(open(s15d.PRISM_REC))
    entries = prism_rec['files'] if isinstance(prism_rec, dict) and 'files' in prism_rec else prism_rec
    hashes = {}
    for path in (s15d.DRC, s15d.CLI):
        h = s15d.md5(path)
        ent = next((e for e in entries if isinstance(e, dict) and str(e.get('name', '')) == os.path.basename(path)), None)
        exp = next((ent[k] for k in ('computed_md5', 'md5', 'supplied_md5') if ent and ent.get(k)), None)
        if exp is not None and exp != h:
            sys.exit('FATAL: %s md5 differs from the figshare record' % path)
        hashes[os.path.basename(path)] = h
    if hashes != rec15e['prism_files']:
        sys.exit('FATAL: PRISM file hashes differ from the Step 15e record')
    cli = pd.read_csv(s15d.CLI)
    drc = pd.read_csv(s15d.DRC, low_memory=False)
    d_dep, d_auc, d_broad, d_pass = s15d.col(drc, 'depmap_id'), s15d.col(drc, 'auc'), s15d.col(drc, 'broad_id'), s15d.col(drc, 'passed_str_profiling')
    c_ccle, c_dep = s15d.col(cli, 'ccle_name'), s15d.col(cli, 'depmap_id')
    prefix = {str(x).split('_')[0].upper(): str(y) for x, y in zip(cli[c_ccle], cli[c_dep])}
    dep_of_line = {ln: prefix[s15d.norm_line(ln)] for ln in s15d.LINES if s15d.norm_line(ln) in prefix}
    if len(dep_of_line) != 10:
        sys.exit('FATAL: %d Tahoe lines matched in PRISM, expected 10' % len(dep_of_line))
    match = json.load(open(s15d.INV_D))['match']
    passed = drc[drc[d_pass].astype(str).str.lower().isin(['true', '1', '1.0']) & drc[d_auc].notna()]
    passed = passed[passed[d_dep].isin(dep_of_line.values())]
    bid_rows = {b: g for b, g in passed.groupby(passed[d_broad].astype(str))}
    label = {}
    for ln, dep in dep_of_line.items():
        for d, mt in match.items():
            vals = []
            for b in mt['broad_ids']:
                g = bid_rows.get(b)
                if g is not None:
                    vals += list(g.loc[g[d_dep] == dep, d_auc].astype(float).values)
            if vals:
                label[(ln, d)] = float(np.median(vals))
    if len(label) != rec15e['n_labels']:
        sys.exit('FATAL: %d labels, Step 15e record has %d' % (len(label), rec15e['n_labels']))
    print('[prism] %d lines | %d labels (as Step 15e)' % (len(dep_of_line), len(label)))

    results, inputs, rows, pc_rows, util_split = {}, {}, [], [], {}
    lv_ref = None
    for split in splits:
        print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
        k = split.split('_')[-1]
        rec_split = record['splits'][k]
        n_drugs = len(rec_split['held_structures'])

        # --- Step 15b mean vectors (15 runs), the recipe's per-cell file for the real cells, the six Step 16 runs
        old = {arm: [s15c.load_vectors(split, arm, s) for s in range(s15.N_SEEDS)] for arm in OLD_ARMS}
        ref = old['recipe'][0]
        cond, n = ref['cond'], len(ref['cond'])
        if n != int(rec_split['n_test_conditions']):
            sys.exit('FATAL: %d conditions, record expects %d' % (n, rec_split['n_test_conditions']))
        ref_pc = s15.load_eval(split, 'recipe', 0)
        if not np.array_equal(ref_pc['cond'], cond):
            sys.exit('FATAL: the recipe per-cell file lists different conditions')
        real = ref_pc['true_cells']
        new = {arm: [load_eval16(split, arm, s) for s in range(s15.N_SEEDS)] for arm in NEW_ARMS}
        for arm, L in list(old.items()) + list(new.items()):
            for v in L:
                if not np.array_equal(v['cond'], cond) or np.abs(v['C'] - ref['C']).max() > 1e-5 or np.abs(v['true'] - ref['true']).max() > 1e-5:
                    sys.exit('FATAL: condition order, control means or true means differ in %s' % v['tag'])
        for arm, L in new.items():
            for v in L:
                for i in range(n):
                    if not np.array_equal(v['true_cells'][i], real[i]):
                        sys.exit('FATAL: stored real cells differ in %s, condition %d' % (v['tag'], i))
            sh = set(v['rec']['shuffled_order_sha256_prefix'] for v in L)
            tt = set(v['rec']['targets_table_sha256'] for v in L)
            if len(sh) != 1 or len(tt) != 1 or tt != {s15.sha(TARGETS)}:
                sys.exit('FATAL: %s runs disagree on the shuffled order or the target table, or the table changed' % arm)
        inputs[split] = {arm: [v['rec'] for v in L] for arm, L in list(old.items()) + list(new.items())}
        true_mean, C = ref['true'], ref['C']
        Y = true_mean - C
        del ref_pc

        # --- Step 15b / 15c per-condition records: annotations in the vectors' order
        pc = pc15[pc15['split'] == split].reset_index(drop=True)
        pcc = pc15c[pc15c['split'] == split].reset_index(drop=True)
        for df, nm in ((pc, 'Step 15b'), (pcc, 'Step 15c')):
            if len(df) != n or not np.array_equal(df['condition'].astype(str).values, cond):
                sys.exit('FATAL: the %s per-condition record does not list the conditions of %s in the vectors order' % (nm, split))
        lines, plates, drugs, doses = (pc['line'].astype(str).values, pc['plate'].astype(str).values,
                                       pc['structure'].astype(str).values, pc['dose_nM'].astype(float).values)
        learn = pc['learnability'].astype(float).values
        lv = sorted(set(lines))
        if lv_ref is None:
            lv_ref = lv
        elif lv != lv_ref:
            sys.exit('FATAL: line set differs between splits')
        if len(np.unique(drugs)) != n_drugs:
            sys.exit('FATAL: %d structures in the record, expected %d' % (len(np.unique(drugs)), n_drugs))
        names = np.array([inv_map.get(s15d.cond_name(c, lines[i]), s15d.cond_name(c, lines[i])) for i, c in enumerate(cond)])
        ib = np.isin(lines, s15.BRAIN)
        print('[design] %d conditions | %d held-out structures | %d lines | cells per condition up to %d' % (n, n_drugs, len(lv), s15.CAP))

        # --- baselines by the Step 15b code path; reproduction guards against the Step 15b and 15c records
        deltas, train = rebuild(split, lines, drugs, doses, lv, Y)
        for bname, D in deltas.items():
            j, d = s15.thesis_per_cond(true_mean, C + D, C)
            if not s15c.close(j, pc['%s_de_overlap' % bname].values) or not s15c.close(d, pc['%s_dpc50' % bname].values):
                sys.exit('FATAL: %s does not reproduce the Step 15b per-condition record' % bname)
        for arm, L in old.items():
            jj, dd = zip(*[s15.thesis_per_cond(v['true'], v['B'], v['C']) for v in L])
            if not s15c.close(np.nanmean(np.vstack(jj), axis=0), pc['%s_de_overlap' % arm].values) or not s15c.close(np.nanmean(np.vstack(dd), axis=0), pc['%s_dpc50' % arm].values):
                sys.exit('FATAL: %s does not reproduce the Step 15b per-condition record' % arm)
        print('[reproduction] DE-overlap and DPC-50 of the nine Step 15b submissions equal the Step 15b record to 1e-9')
        ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
        ctx_u = np.unique(ctx_key)
        contexts = [np.where(ctx_key == u)[0] for u in ctx_u]
        ctx_id = np.array([int(np.where(ctx_u == ctx_key[i])[0][0]) for i in range(n)])
        cand, cand_full = {}, {}
        for members in contexts:
            for p in members:
                cand_full[p] = members
                cand[p] = np.array([q for q in members if q == p or drugs[q] != drugs[p]])
        n_cand = np.array([len(cand[p]) for p in range(n)])
        if not np.array_equal(n_cand, pcc['n_candidates'].astype(int).values):
            sys.exit('FATAL: the candidate sets differ from the Step 15c record')
        pds, pds_seed = {}, {}
        for bname, D in deltas.items():
            pds[bname] = s15c.pds_sets(D, Y, contexts, cand)
            pds_seed[bname] = [pds[bname]]
        for arm, L in old.items():
            pds_seed[arm] = [s15c.pds_sets(v['B'] - v['C'], Y, contexts, cand) for v in L]
            pds[arm] = np.nanmean(np.vstack(pds_seed[arm]), axis=0)
        for nm in list(deltas) + list(OLD_ARMS):
            if not s15c.close(pds[nm], pcc['%s_pds' % nm].values):
                sys.exit('FATAL: PDS of %s does not reproduce the Step 15c per-condition record' % nm)
        print('[reproduction] PDS of the nine Step 15b submissions equals the Step 15c record to 1e-9 (%d contexts, candidates %d-%d)'
              % (len(contexts), n_cand.min(), n_cand.max()))

        # --- the new arms: DE-overlap, DPC-50, energy, PDS (per seed, then seed-mean per condition)
        jac, dpc, en = {}, {}, {}
        for nm in list(deltas) + list(OLD_ARMS):
            jac[nm], dpc[nm], en[nm] = pc['%s_de_overlap' % nm].values, pc['%s_dpc50' % nm].values, pc['%s_energy' % nm].values
        seed_vals = {}
        for arm, L in new.items():
            jj, dd, ee, pp = [], [], [], []
            for v in L:
                j, d = s15.thesis_per_cond(v['true'], v['B'], v['C'])
                e_ = np.array([s15.energy(v['pred'][i], real[i]) for i in range(n)])
                p_ = s15c.pds_sets(v['B'] - v['C'], Y, contexts, cand)
                jj.append(j); dd.append(d); ee.append(e_); pp.append(p_)
                print('[arm] %-13s seed %d  DE-overlap %.4f  energy %.4f  DPC-50 mean %.4f  PDS %.4f'
                      % (arm, L.index(v), np.nanmean(j), e_.mean(), np.nanmean(d), np.nanmean(p_)))
            seed_vals[arm] = dict(jac=jj, dpc=dd, en=ee, pds=pp)
            jac[arm], dpc[arm], en[arm] = (np.nanmean(np.vstack(jj), axis=0), np.nanmean(np.vstack(dd), axis=0), np.nanmean(np.vstack(ee), axis=0))
            pds_seed[arm] = pp
            pds[arm] = np.nanmean(np.vstack(pp), axis=0)
        for arm, L in new.items():
            for v in L:
                v['pred'] = None                                          # per-cell arrays no longer needed
        del real

        # --- MRA (section 4.4): norms of predicted and measured effects, per-context Spearman
        true_norm = np.linalg.norm(Y, axis=1)
        norms = {}                                                       # submission -> list of per-seed norm arrays
        for bname, D in deltas.items():
            norms[bname] = [np.linalg.norm(D, axis=1)]
        for arm, L in list(old.items()) + list(new.items()):
            norms[arm] = [np.linalg.norm(v['B'] - v['C'], axis=1) for v in L]
        n_ctx = len(contexts)
        full = np.arange(n)
        # replicate ceiling: conditions with a plate 6 / 14 copy; the copy's measured norm offered as the prediction
        keyed = {}
        for i, c in enumerate(cond):
            key, plate = s15.parse_cond(c)
            if plate in s15.REP_PLATES:
                keyed.setdefault(key, {})[plate] = i
        pairs = [(v['6'], v['14']) for v in keyed.values() if '6' in v and '14' in v]
        partner = np.full(n, -1)
        for i6, i14 in pairs:
            partner[i6], partner[i14] = i14, i6
        rep_ix = np.where(partner >= 0)[0]
        for i in rep_ix:
            if ctx_id[partner[i]] != ctx_id[i]:
                sys.exit('FATAL: a replicate pair spans two (line, dose) contexts')
        rep_norm = np.zeros(n)
        rep_norm[rep_ix] = true_norm[partner[rep_ix]]

        def mra_all(sel):
            """Every submission's MRA and the ceiling's on one selection; the per-context split and the measured norms'
            ranks are computed once and shared, which is what makes 5,000 replicates affordable."""
            pre = prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, sel))
            out = {nm: float(np.mean([mra_pre(x, pre) for x in norms[nm]])) for nm in norms}
            s_ = sel[partner[sel] >= 0]
            out['ceiling'] = mra_pre(rep_norm, prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, s_))) if len(s_) else np.nan
            return out

        pre_full = prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, full))
        mra_seed = {nm: [mra_pre(x, pre_full) for x in norms[nm]] for nm in norms}
        mra = {nm: float(np.mean(v_)) for nm, v_ in mra_seed.items()}
        full_all = mra_all(full)
        for nm in norms:
            if abs(full_all[nm] - mra[nm]) > 1e-12:
                sys.exit('FATAL: the shared-context MRA differs from the per-seed computation for %s' % nm)
        for nm in ('control_paste', 'mean_response'):
            if abs(mra[nm]) > 1e-12:
                sys.exit('FATAL: %s MRA is %.3g, expected 0 by definition' % (nm, mra[nm]))
        mra_ceil = full_all['ceiling']
        n_ctx_rep = sum(1 for c in range(n_ctx) if (ctx_id[rep_ix] == c).sum() >= MIN_CTX)
        n_ctx_used = sum(1 for c in range(n_ctx) if len(contexts[c]) >= MIN_CTX)
        print('[mra] %d contexts (%d with >= %d conditions) | replicate ceiling %s over %d contexts with >= %d replicated conditions (%d pairs)'
              % (n_ctx, n_ctx_used, MIN_CTX, 'n/a' if not np.isfinite(mra_ceil) else '%.4f' % mra_ceil, n_ctx_rep, MIN_CTX, len(pairs)))
        for nm in SUBS:
            print('[mra] %-14s %+.4f | per-seed %s' % (nm, mra[nm], [round(x, 4) for x in mra_seed[nm]]))

        # --- less-unseen strata (section 4.5), from the encoder's own lookup
        train_targets = set().union(*[TM['bag'][s] for s in train['smiles']])
        n_t = np.array([len(TM['bag'][drugs[i]]) for i in range(n)])
        n_seen = np.array([len(TM['bag'][drugs[i]] & train_targets) for i in range(n)])
        stratum = np.array([STRATA[0] if n_t[i] == 0 else STRATA[1] if n_seen[i] == 0 else STRATA[2] for i in range(n)], dtype=object)
        strata = {}
        for st in STRATA:
            m = stratum == st
            dsub = sorted(set(drugs[m]))
            entry = dict(n_drugs=len(dsub), n_conditions=int(m.sum()),
                         targets_per_drug=[int(n_t[drugs == d_][0]) for d_ in dsub], seen_per_drug=[int(n_seen[drugs == d_][0]) for d_ in dsub])
            for nm in ('recipe', 'targets', 'targets_shuf'):
                entry['pds_%s' % nm] = float(np.nanmean(pds[nm][m])) if m.any() else None
                entry['de_%s' % nm] = float(np.nanmean(jac[nm][m])) if m.any() else None
            if len(dsub) >= 2:                                           # descriptive: the gated contrast within the stratum
                for key_, arr in (('pds', pds), ('de', jac)):
                    entry['%s_targets_minus_recipe' % key_] = contrast_pc(arr['targets'][m], arr['recipe'][m], drugs[m])
                    entry['%s_targets_minus_targets_shuf' % key_] = contrast_pc(arr['targets'][m], arr['targets_shuf'][m], drugs[m])
            strata[st] = entry
            print('[strata] %-20s %d drugs, %d conditions | PDS recipe %s targets %s shuf %s | DE recipe %s targets %s shuf %s'
                  % (st, len(dsub), int(m.sum()), *['n/a' if entry[q] is None else '%.4f' % entry[q] for q in
                     ('pds_recipe', 'pds_targets', 'pds_targets_shuf', 'de_recipe', 'de_targets', 'de_targets_shuf')]))
        print('[strata] training drugs %d with %d distinct curated targets; held-out drugs with targets %d of %d'
              % (len(train['smiles']), len(train_targets), len(set(drugs[n_t > 0])), n_drugs))
        # Exploratory, dated 2026-10-08 (before any confirmation result exists): does the gain depend on the stratum?
        # 'no targets' is the width-only stratum (fingerprint + 545 zeros); 'targets, >= 1 seen' is the one where target
        # information could act. A gain that comes from the targets must be larger in the second.
        gaps = {}
        for key_, arr in (('pds', pds), ('de', jac)):
            for arm in NEW_ARMS:
                d_ = arr[arm] - arr['recipe']
                g = stratum_gap(d_, drugs, stratum, STRATA[2], STRATA[0])
                gaps['%s_%s_minus_recipe_seen_minus_notargets' % (key_, arm)] = g
                print('[strata-gap] %-10s %-13s minus recipe: gain(%s) - gain(%s) %s | %d vs %d drugs'
                      % (key_, arm, STRATA[2], STRATA[0],
                         fmt(g) if g['diff'] is not None else g['verdict'], g['n_drugs_hi'], g['n_drugs_lo']))

        # --- cross-plate PDS facet (Step 15c section 8) for the recipe and the new arms, plus the ceiling
        members_of = {u: np.where(ctx_key == u)[0] for u in ctx_u}
        xp = {nm: [] for nm in ('recipe', 'targets', 'targets_shuf')}
        xp_ceil = []
        pred_of = {nm: [v['B'] - v['C'] for v in (old if nm in old else new)[nm]] for nm in xp}
        for i6, i14 in pairs:
            for src, dst in ((i6, i14), (i14, i6)):
                members = members_of[ctx_key[dst]]
                cs = np.array([q for q in members if plates[q] != plates[src] and (q == dst or drugs[q] != drugs[dst])])
                if len(cs) < s15c.MIN_CAND:
                    continue
                E = Y.copy(); E[dst] = Y[src]
                xp_ceil.append(s15c.pds_sets(E, Y, [members], {dst: cs})[dst])
                for nm in xp:
                    vals_ = []
                    for P in pred_of[nm]:
                        E = Y.copy(); E[dst] = P[src]
                        vals_.append(s15c.pds_sets(E, Y, [members], {dst: cs})[dst])
                    xp[nm].append(float(np.mean(vals_)))
        crossplate = dict(n_pairs=len(pairs), n_evaluations=len(xp_ceil), ceiling=float(np.mean(xp_ceil)) if xp_ceil else None,
                          **{nm: (float(np.mean(v_)) if v_ else None) for nm, v_ in xp.items()})
        print('[cross-plate] %d evaluations | ceiling %s | recipe %s | targets %s | targets_shuf %s'
              % (len(xp_ceil), *['n/a' if crossplate[q] is None else '%.4f' % crossplate[q] for q in ('ceiling', 'recipe', 'targets', 'targets_shuf')]))

        # --- absolute values per submission
        pre_brain = prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, full[ib]))
        pre_gut = prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, full[~ib]))
        metrics = {}
        for nm in SUBS:
            x = pds[nm]
            m_ = dict(pds_mean=float(np.nanmean(x)), pds_median=float(np.nanmedian(x)),
                      de_overlap=float(np.nanmean(jac[nm])) if np.isfinite(jac[nm]).any() else None,
                      energy=float(np.nanmean(en[nm])), dpc_mean=float(np.nanmean(dpc[nm])), mra=mra[nm],
                      pds_per_seed=[float(np.nanmean(s_)) for s_ in pds_seed[nm]], mra_per_seed=mra_seed[nm],
                      pds_per_line={ln: float(np.nanmean(x[lines == ln])) for ln in lv},
                      pds_per_dose={('%g' % d_): float(np.nanmean(x[np.isclose(doses, d_)])) for d_ in sorted(set(doses))},
                      pds_brain_minus_gut=float(np.nanmean(x[ib]) - np.nanmean(x[~ib])),
                      mra_brain=float(np.mean([mra_pre(q, pre_brain) for q in norms[nm]])),
                      mra_gut=float(np.mean([mra_pre(q, pre_gut) for q in norms[nm]])))
            if nm in new:
                m_['de_per_seed'] = [float(np.nanmean(s_)) for s_ in seed_vals[nm]['jac']]
                m_['energy_per_seed'] = [float(np.nanmean(s_)) for s_ in seed_vals[nm]['en']]
            elif nm in OLD_ARMS:
                m_['de_per_seed'] = [s_['de_overlap'] for s_ in rec15b['per_split'][split]['metrics'][nm]['per_seed']]
                m_['energy_per_seed'] = [s_['energy'] for s_ in rec15b['per_split'][split]['metrics'][nm]['per_seed']]
            metrics[nm] = m_
            for key_ in ('pds_mean', 'pds_median', 'de_overlap', 'energy', 'dpc_mean', 'mra'):
                rows.append(dict(split=split, level='ALL', key=key_, submission=nm, value=m_[key_]))
            for ln, val in m_['pds_per_line'].items():
                rows.append(dict(split=split, level='line', key=ln, submission=nm, value=val))
            for ds, val in m_['pds_per_dose'].items():
                rows.append(dict(split=split, level='dose', key=ds, submission=nm, value=val))
            print('[abs] %-14s PDS %.4f (seeds %s) | DE-overlap %s | energy %.4f | DPC-50 %.4f | MRA %+.4f | brain-gut PDS %+.4f'
                  % (nm, m_['pds_mean'], [round(v_, 4) for v_ in m_['pds_per_seed']], 'n/a' if m_['de_overlap'] is None else '%.4f' % m_['de_overlap'],
                     m_['energy'], m_['dpc_mean'], m_['mra'], m_['pds_brain_minus_gut']))
        replicate = dict(pds=rec15c['per_split'][split]['replicate'], de_overlap_energy_dpc=rec15b['per_split'][split]['replicate'],
                         mra=mra_ceil if np.isfinite(mra_ceil) else None, mra_n_contexts=n_ctx_rep, n_pairs=len(pairs))
        for i in range(n):
            r = dict(split=split, condition=cond[i], line=lines[i], plate=plates[i], structure=drugs[i], drug=names[i], dose_nM=doses[i],
                     n_candidates=int(n_cand[i]), learnability=learn[i], n_targets=int(n_t[i]), n_targets_seen=int(n_seen[i]), stratum=stratum[i],
                     true_effect_norm=float(true_norm[i]))
            for nm in SUBS:
                r['%s_pds' % nm] = pds[nm][i]; r['%s_de_overlap' % nm] = jac[nm][i]; r['%s_dpc50' % nm] = dpc[nm][i]; r['%s_energy' % nm] = en[nm][i]
                r['%s_effect_norm' % nm] = float(np.mean([q[i] for q in norms[nm]]))
            pc_rows.append(r)

        # --- the pre-registered contrasts (section 5): per-condition metrics with fixed values under resampling
        con = {}
        con['Q1_primary_pds_targets_minus_recipe'] = contrast_pc(pds['targets'], pds['recipe'], drugs)
        con['Q2_coprimary_de_targets_minus_recipe'] = contrast_pc(jac['targets'], jac['recipe'], drugs)
        con['Q3_control_pds_targets_minus_targets_shuf'] = contrast_pc(pds['targets'], pds['targets_shuf'], drugs)
        print('\nQ1 PRIMARY     PDS, targets minus recipe              %s' % fmt(con['Q1_primary_pds_targets_minus_recipe']))
        print('Q2 CO-PRIMARY  DE-overlap, targets minus recipe       %s' % fmt(con['Q2_coprimary_de_targets_minus_recipe']))
        print('Q3 CONTROL     PDS, targets minus targets_shuf        %s' % fmt(con['Q3_control_pds_targets_minus_targets_shuf']))
        for key_, arr, lab in (('pds', pds, 'PDS'), ('de', jac, 'DE-overlap'), ('en', en, 'energy'), ('dpc', dpc, 'DPC-50')):
            for a_, b_ in (('targets_shuf', 'recipe'), ('targets', 'recipe'), ('targets', 'targets_shuf')):
                key = 'secondary_%s_%s_minus_%s' % (key_, a_, b_)
                if key in ('secondary_pds_targets_minus_recipe', 'secondary_de_targets_minus_recipe', 'secondary_pds_targets_minus_targets_shuf'):
                    continue                                            # gated above
                con[key] = contrast_pc(arr[a_], arr[b_], drugs)
                print('secondary   %-10s %-12s minus %-12s %s' % (lab, a_, b_, fmt(con[key])))
        half = np.full(n, 0.5)
        for nm in NEW_ARMS:
            con['secondary_pds_%s_minus_0.5' % nm] = contrast_pc(pds[nm], half, drugs)
            print('secondary   PDS, %-12s minus 0.5                %s' % (nm, fmt(con['secondary_pds_%s_minus_0.5' % nm])))

        # --- MRA contrasts: recomputed per replicate on the resampled drugs (same resample for every submission)
        rng = np.random.RandomState(0)
        gs = [np.where(drugs == u)[0] for u in np.unique(drugs)]
        BM = {nm: np.full(N_BOOT, np.nan) for nm in SUBS}
        BC = np.full(N_BOOT, np.nan)
        for b in range(N_BOOT):
            sel = np.concatenate([gs[g] for g in rng.randint(0, len(gs), len(gs))])
            got = mra_all(sel)
            for nm in SUBS:
                BM[nm][b] = got[nm]
            BC[b] = got['ceiling']
        for nm in SUBS:
            if nm in ('control_paste', 'mean_response'):
                continue
            con['secondary_mra_%s_minus_0' % nm] = verdict_of(mra[nm], BM[nm])
            print('secondary   MRA, %-14s minus 0                %s' % (nm, fmt(con['secondary_mra_%s_minus_0' % nm])))
        if np.isfinite(mra_ceil):
            con['secondary_mra_ceiling_minus_0'] = verdict_of(mra_ceil, BC)
            print('secondary   MRA, replicate ceiling minus 0          %s' % fmt(con['secondary_mra_ceiling_minus_0']))
        for a_, b_ in (('targets', 'recipe'), ('targets', 'targets_shuf'), ('targets_shuf', 'recipe')):
            con['secondary_mra_%s_minus_%s' % (a_, b_)] = verdict_of(mra[a_] - mra[b_], BM[a_] - BM[b_])
            print('secondary   MRA, %-12s minus %-12s      %s' % (a_, b_, fmt(con['secondary_mra_%s_minus_%s' % (a_, b_)])))
        print('[bootstrap] MRA: %d replicates (%d s)' % (N_BOOT, time.time() - T0))

        # ------------------------------------------------------------------ utility (section 4.6): the Step 15e per-split computation
        utr, smi_c, line_c, dose_med, Ytr = train['utr'], train['smi_c'], train['line_c'], train['dose_med'], train['Ytr']
        tr_names = np.array([inv_map.get(s15d.cond_name(u, line_c[q]), s15d.cond_name(u, line_c[q])) for q, u in enumerate(utr)])
        avail_t = {}
        for q in range(len(utr)):
            if line_c[q] in dep_of_line:
                avail_t.setdefault(tr_names[q], set()).add(float(np.round(dose_med[q], 6)))
        hd_t = {nm: max(v_) for nm, v_ in avail_t.items() if v_}
        tp2 = {}
        for q in range(len(utr)):
            nm = tr_names[q]
            if line_c[q] in dep_of_line and nm in hd_t and abs(dose_med[q] - hd_t[nm]) < 1e-6 and (line_c[q], nm) in label:
                tp2.setdefault((line_c[q], nm), []).append(q)
        tr_keys2 = sorted(tp2)
        if set(nm for _, nm in tr_keys2) & set(names):
            sys.exit('FATAL: a held-out drug name among the training pairs')
        Etr2 = np.array([Ytr[tp2[key]].mean(axis=0) for key in tr_keys2])
        ytr2 = np.array([label[key] for key in tr_keys2])
        g2 = np.array([nm for _, nm in tr_keys2])
        mu2, sd2 = Etr2.mean(axis=0), Etr2.std(axis=0)
        sd2 = np.where(sd2 > 1e-8, sd2, 1.0)
        f2, alpha_f2, mse_f2, r2_f2 = s15d.fit_ridge((Etr2 - mu2) / sd2, ytr2, g2, 'f, highest dose')
        FPq2 = np.array([s15.Drug_dose_encoder([smi_c[tp2[key][0]]], [hd_t[key[1]]], comb_num=1)[0] for key in tr_keys2], dtype=np.float64)
        qsar2, alpha_q2, mse_q2, r2_q2 = s15d.fit_ridge(s15.features(FPq2, np.array([ln for ln, _ in tr_keys2]), lv), ytr2, g2, 'qsar, highest dose')
        del Ytr, train['Ytr']
        avail_h = {nm: sorted(set(float(np.round(doses[i], 6)) for i in range(n) if names[i] == nm and lines[i] in dep_of_line)) for nm in sorted(set(names))}
        hd_h = {nm: max(v_) for nm, v_ in avail_h.items() if v_}
        hp2 = {}
        for i in range(n):
            nm = names[i]
            if lines[i] in dep_of_line and nm in hd_h and abs(doses[i] - hd_h[nm]) < 1e-6:
                hp2.setdefault((lines[i], nm), []).append(i)
        keys2 = [key for key in sorted(hp2) if key in label]
        line_ix2 = np.array([lv.index(ln) for ln, _ in keys2])
        drug_names2 = np.array([nm for _, nm in keys2])
        drug_ix2 = pd.factorize(pd.Series(drug_names2), sort=True)[0]
        labels2 = np.array([label[key] for key in keys2])

        def pe2(E):
            return np.array([E[hp2[key]].mean(axis=0) for key in keys2])

        def through(E):
            return f2.predict((pe2(E) - mu2) / sd2)

        vals, per_seed_u = {}, {}
        for bname, D in deltas.items():
            vals[bname] = through(D)
        for arm, L in list(old.items()) + list(new.items()):
            per_seed_u[arm] = [through(v['B'] - v['C']) for v in L]
            vals[arm] = np.mean(np.vstack(per_seed_u[arm]), axis=0)
        vals['ceiling'] = through(Y)
        FPh2 = np.array([s15.Drug_dose_encoder([drugs[hp2[key][0]]], [hd_h[key[1]]], comb_num=1)[0] for key in keys2], dtype=np.float64)
        vals['qsar'] = qsar2.predict(s15.features(FPh2, np.array([ln for ln, _ in keys2]), lv))
        vals['effect_size'] = np.mean(np.vstack([-np.linalg.norm(pe2(v['B'] - v['C']), axis=1) for v in old['recipe']]), axis=0)
        vals['effect_size_measured'] = -np.linalg.norm(pe2(Y), axis=1)
        all_sel = np.arange(len(keys2))
        U = {nm: s15d.U_of(v_, labels2, line_ix2, len(lv), all_sel)[0] for nm, v_ in vals.items()}
        expect = rec15e['U_per_split'][split]
        for nm in expect:
            if nm not in U or abs(U[nm] - expect[nm]) > 1e-9:
                sys.exit('FATAL: U_s(%s) = %s does not reproduce the Step 15e record %.9f' % (nm, U.get(nm), expect[nm]))
        print('[utility] %d labelled held-out pairs (%d drugs), %d training pairs (%d drugs) | U_s reproduces the Step 15e record to 1e-9 for all %d Step 15e submissions'
              % (len(keys2), len(set(drug_names2)), len(tr_keys2), len(set(g2)), len(expect)))
        for nm in UTIL_SUBS:
            print('  U_s %-22s %+.4f%s' % (nm, U[nm], (' | per-seed %s' % [round(x, 4) for x in [s15d.U_of(v_, labels2, line_ix2, len(lv), all_sel)[0] for v_ in per_seed_u[nm]]]) if nm in per_seed_u else ''))
        rng = np.random.RandomState(0)
        ug = [np.where(drug_ix2 == d_)[0] for d_ in np.unique(drug_ix2)]
        BU = {nm: np.full(N_BOOT, np.nan) for nm in UTIL_SUBS}
        for b in range(N_BOOT):
            sel = np.concatenate([ug[g] for g in rng.randint(0, len(ug), len(ug))])
            for nm in UTIL_SUBS:
                BU[nm][b] = s15e.U_fast(vals[nm], labels2, line_ix2, len(lv), sel)
        ucon = {}
        for nm in NEW_ARMS:
            ucon['U_%s_minus_0' % nm] = verdict_of(U[nm], BU[nm])
            print('  utility U_s %-14s minus 0            %s' % (nm, fmt(ucon['U_%s_minus_0' % nm])))
        for a_, b_ in (('targets', 'recipe'), ('targets_shuf', 'recipe'), ('targets', 'targets_shuf')):
            ucon['U_%s_minus_%s' % (a_, b_)] = verdict_of(U[a_] - U[b_], BU[a_] - BU[b_])
            print('  utility U_s %-12s minus %-12s   %s' % (a_, b_, fmt(ucon['U_%s_minus_%s' % (a_, b_)])))
        util_split[split] = dict(labels=labels2, line_ix=line_ix2, drug_ix=drug_ix2, vals={nm: vals[nm] for nm in UTIL_SUBS},
                                 n_drugs=int(len(set(drug_names2))), n_pairs=len(keys2))
        results[split] = dict(n_conditions=n, n_structures=n_drugs, n_contexts=len(contexts), n_contexts_mra=n_ctx_used, min_candidates=s15c.MIN_CAND,
                              min_context=MIN_CTX, metrics=metrics, replicate=replicate, strata=strata, strata_gap=gaps, crossplate=crossplate, contrasts=con,
                              utility=dict(U={nm: float(U[nm]) for nm in UTIL_SUBS}, U_per_seed={nm: [float(s15d.U_of(v_, labels2, line_ix2, len(lv), all_sel)[0]) for v_ in per_seed_u[nm]] for nm in NEW_ARMS},
                                           contrasts=ucon, n_pairs=len(keys2), n_drugs=int(len(set(drug_names2))),
                                           cv=dict(f=dict(alpha=alpha_f2, mse=mse_f2, r2=r2_f2), qsar=dict(alpha=alpha_q2, mse=mse_q2, r2=r2_q2))))
        del old, new

    # ------------------------------------------------------------------ pooled utility (section 4.6), final only
    pooled = None
    if final:
        sp = list(s15.ALL_SPLITS)
        nsd = {s: util_split[s]['n_drugs'] for s in sp}
        wsum = float(sum(nsd.values()))
        if nsd != rec15e['n_drugs_per_split']:
            sys.exit('FATAL: labelled drugs per split %s differ from the Step 15e record %s' % (nsd, rec15e['n_drugs_per_split']))

        def U_pool_of(sel):
            out = {}
            for nm in UTIL_SUBS:
                acc = 0.0
                for s in sp:
                    ps = util_split[s]
                    u = s15e.U_fast(ps['vals'][nm], ps['labels'], ps['line_ix'], len(lv_ref), sel[s])
                    acc += nsd[s] * (u if np.isfinite(u) else 0.0)
                out[nm] = acc / wsum
            return out

        def U_B_of(sel):
            out = {}
            for nm in UTIL_SUBS:
                rhos = []
                for l in range(len(lv_ref)):
                    xs, ys = [], []
                    for s in sp:
                        ps = util_split[s]
                        m = sel[s][ps['line_ix'][sel[s]] == l]
                        if len(m) == 0:
                            continue
                        v_ = ps['vals'][nm][m]
                        xc = v_ - v_.mean()
                        xc[np.abs(xc) < 1e-9] = 0.0
                        xs.append(xc); ys.append(ps['labels'][m])
                    if xs:
                        x, y = np.concatenate(xs), np.concatenate(ys)
                        if len(x) >= s15d.MIN_PAIRS_LINE:
                            rhos.append(s15e.spearman_fast(x, y))
                out[nm] = float(np.mean(rhos)) if rhos else np.nan
            return out

        full_sel = {s: np.arange(util_split[s]['n_pairs']) for s in sp}
        full_drugs = {s: np.unique(util_split[s]['drug_ix']) for s in sp}
        UA, UB = U_pool_of(full_sel), U_B_of(full_sel)
        for nm in UTIL_SUBS:
            if nm in rec15e['U_pool'] and (abs(UA[nm] - rec15e['U_pool'][nm]) > 1e-9 or abs(UB[nm] - rec15e['U_B'][nm]) > 1e-9):
                sys.exit('FATAL: pooled U of %s does not reproduce the Step 15e record' % nm)
        print('\n[pooled] labelled drugs per split %s (total %d); readings A and B reproduce the Step 15e record to 1e-9' % (nsd, int(wsum)))
        for nm in UTIL_SUBS:
            print('  pooled A %-22s %+.4f | reading B %+.4f' % (nm, UA[nm], UB[nm]))
        rng = np.random.RandomState(0)
        groups = {s: [np.where(util_split[s]['drug_ix'] == d_)[0] for d_ in full_drugs[s]] for s in sp}
        BA = {nm: np.full(N_BOOT, np.nan) for nm in UTIL_SUBS}
        BB = {nm: np.full(N_BOOT, np.nan) for nm in UTIL_SUBS}
        for b in range(N_BOOT):
            sel = {}
            for s in sp:
                draw = rng.randint(0, len(groups[s]), len(groups[s]))
                sel[s] = np.concatenate([groups[s][g] for g in draw])
            ua, ub = U_pool_of(sel), U_B_of(sel)
            for nm in UTIL_SUBS:
                BA[nm][b], BB[nm][b] = ua[nm], ub[nm]
        print('[bootstrap] pooled utility: %d stratified replicates (%d s)' % (N_BOOT, time.time() - T0))
        pcon, pconB = {}, {}
        for nm in UTIL_SUBS:
            pcon['Upool_%s_minus_0' % nm] = verdict_of(UA[nm], BA[nm])
            pconB['UB_%s_minus_0' % nm] = verdict_of(UB[nm], BB[nm])
        r0 = rec15e['contrasts']['Q1_primary_Upool_recipe_minus_0']
        if abs(pcon['Upool_recipe_minus_0']['ci'][0] - r0['ci'][0]) > 1e-9 or abs(pcon['Upool_recipe_minus_0']['ci'][1] - r0['ci'][1]) > 1e-9:
            sys.exit('FATAL: the recipe\'s pooled interval %s does not reproduce the Step 15e record %s' % (pcon['Upool_recipe_minus_0']['ci'], r0['ci']))
        print('[reproduction] the recipe\'s pooled interval (reading A) equals the Step 15e record')
        for a_, b_ in (('targets', 'recipe'), ('targets_shuf', 'recipe'), ('targets', 'targets_shuf'), ('targets', 'ceiling'), ('targets', 'qsar')):
            pcon['Upool_%s_minus_%s' % (a_, b_)] = verdict_of(UA[a_] - UA[b_], BA[a_] - BA[b_])
            pconB['UB_%s_minus_%s' % (a_, b_)] = verdict_of(UB[a_] - UB[b_], BB[a_] - BB[b_])
        for key, c in list(pcon.items()) + list(pconB.items()):
            print('  %-40s %s' % (key, fmt(c)))
        pooled = dict(n_drugs_per_split=nsd, n_drugs_total=int(wsum), U_pool={nm: float(v) for nm, v in UA.items()}, U_B={nm: float(v) for nm, v in UB.items()},
                      contrasts=pcon, contrasts_B=pconB)

    # ------------------------------------------------------------------ section 6 (precedence 1 -> 2 -> 5 -> 3 -> 4)
    cat, q2 = None, None
    if final:
        q1 = {s: results[s]['contrasts']['Q1_primary_pds_targets_minus_recipe']['verdict'] for s in s15.ALL_SPLITS}
        q2 = {s: results[s]['contrasts']['Q2_coprimary_de_targets_minus_recipe']['verdict'] for s in s15.ALL_SPLITS}
        q3 = {s: results[s]['contrasts']['Q3_control_pds_targets_minus_targets_shuf']['verdict'] for s in s15.ALL_SPLITS}
        v1, v3 = list(q1.values()), list(q3.values())
        if all(v == 'above zero' for v in v1) and all(v == 'above zero' for v in v3):
            cat = '1 TARGET INFORMATION HELPS DIRECTION'
        elif all(v == 'above zero' for v in v1):
            cat = '2 WIDTH, NOT INFORMATION'
        elif all(v == 'includes zero' for v in v1):
            cat = '5 UNRESOLVED'
        elif 'below zero' in v1:
            cat = '3 TARGET INFORMATION HURTS'
        else:
            cat = '4 PARTIAL'
        print('\n' + '=' * 72 + '\nPRE-REGISTERED VERDICT (section 6): %s\n  Q1 targets - recipe, PDS per split: %s\n  Q3 targets - targets_shuf, PDS per split: %s\n  Q2 targets - recipe, DE-overlap per split: %s\n'
              % (cat, q1, q3, q2) + '  Section 7 expected: category 1 or 5 (small PDS gain, +0.01 to +0.03).\n' + '=' * 72)
    else:
        print('\n[development] %s scored; section 6 is applied only when all three splits are present' % ', '.join(splits))

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'level', 'key', 'submission', 'value'])
        w.writeheader(); w.writerows(rows)
    with open(OUT_PC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0].keys()))
        w.writeheader(); w.writerows(pc_rows)
    json.dump(dict(prereg=PRE, final=final, verdict=cat, q2_verdicts=q2, splits=splits, interval='97.5% percentile per split, structure-clustered',
                   n_boot=N_BOOT, n_seeds=s15.N_SEEDS, min_candidates=s15c.MIN_CAND, min_context=MIN_CTX, vocab_size=TARGET_VOCAB_SIZE,
                   shuffled_order_sha256=TM['perm_sha256'], per_split=results, pooled=pooled, inputs=inputs,
                   records=dict(step15b=s15.sha(PC15), step15c=s15.sha(PC15C), step15e=s15.sha(REC15E), targets_table=s15.sha(TARGETS), data_record=s15.sha(s15.RECORD)),
                   commit=commit, seconds=round(time.time() - T0)),
              open(OUT, 'w'), indent=1, allow_nan=True)
    print('\nwrote %s, %s and %s  (%.0f s)' % (OUT, OUT_CSV, OUT_PC, time.time() - T0))


if __name__ == '__main__':
    main()
