#!/usr/bin/env python
"""Step 15e: pooled utility of the frozen Tahoe predictions against PRISM sensitivity, over all held-out drugs.

Pre-registration: PREREG_step15e_tahoe_pooled_utility.md (sections 3-6, 9), committed before this file. One run over
the three splits (no development split: the per-split values are already known from Step 15d and are asserted).

  1. guards as score_step15d_utility.py (prereg and scorers committed and clean; arrays hash-checked; baselines
     reproduced to 1e-9 against the Step 15b record; PRISM files checked; three dose levels);
  2. per split, exactly the highest-available-dose variant of Step 15d: f_s and QSAR_s fitted on the split's training
     pairs, per-pair values for every submission on the split's labelled held-out pairs, per-line Spearman, U_s;
     U_s for recipe, ridge-train, ceiling, QSAR, effect size and measured effect size asserted equal to
     tahoe_results/step15d_utility.json (exploratory.variants.highest_dose.U) to 1e-9;
  3. reading A: U_pool = sum_s n_s U_s / sum_s n_s (n_s = labelled held-out drugs of split s); stratified drug bootstrap
     (drugs resampled within each split; 5,000 replicates; RandomState(0); 97.5 %), the same resample for every
     submission; contrasts of section 5; the ratio U_pool(recipe) / U_pool(ceiling);
  4. reading B: per-(split, line) centred values pooled across splits, one Spearman per line over all labelled held-out
     drugs of the line, mean over lines; same bootstrap;
  5. within-drug across-line read (drugs with >= 6 labelled lines), per-line and brain/gut summaries, per-seed U;
  6. section 6 with the precedence 4 -> 1 -> 2 -> 3.

Usage: python -u score_step15e_pooled.py
Outputs: tahoe_results/step15e_pooled.json, step15e_pooled_long.csv, step15e_pooled_pairs.csv. Never overwrites.
"""
import json
import os
import subprocess
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402
import score_step15c_tahoe as s15c                    # noqa: E402
import score_step15d_utility as s15d                  # noqa: E402  fit_ridge, U_of, W_of, spearman0, cond_name, md5, col, norm_line

PRE = 'PREREG_step15e_tahoe_pooled_utility.md'
REC15D = os.path.join(s15.RES, 'step15d_utility.json')
OUT = os.path.join(s15.RES, 'step15e_pooled.json')
OUT_CSV = os.path.join(s15.RES, 'step15e_pooled_long.csv')
OUT_PAIRS = os.path.join(s15.RES, 'step15e_pooled_pairs.csv')
N_BOOT, LO, HI = 5000, 1.25, 98.75
ASSERTED = ('recipe', 'ridge_train', 'ceiling', 'qsar', 'effect_size', 'effect_size_measured')
BRAIN = set(s15.BRAIN)
T0 = time.time()


def spearman_fast(x, y):
    """Spearman rho via average ranks and Pearson (identical to scipy.stats.spearmanr up to floating point); 0 when x
    or y is constant, as score_step15d_utility.spearman0. Used in the bootstrap for speed; checked against spearman0."""
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def U_fast(vals, labels, line_ix, n_lines, sel):
    rhos = []
    for l in range(n_lines):
        m = sel[line_ix[sel] == l]
        if len(m) >= s15d.MIN_PAIRS_LINE:
            rhos.append(spearman_fast(vals[m], labels[m]))
    return float(np.mean(rhos)) if rhos else np.nan


def main():
    for q in (OUT, OUT_CSV, OUT_PAIRS):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 9:
        sys.exit('FATAL: prereg incomplete')
    tracked = '%s score_step15e_pooled.py score_step15d_utility.py score_step15_tahoe.py score_step15c_tahoe.py %s %s %s' % (PRE, s15d.PC15, s15d.INV_D, REC15D)
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers or records not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers or records uncommitted')
    for q in (s15d.DRC, s15d.CLI, s15d.PRISM_REC, s15.RECORD):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    rec15d = json.load(open(REC15D))
    if not rec15d.get('final'):
        sys.exit('FATAL: %s is not the final Step 15d record' % REC15D)
    record = json.load(open(s15.RECORD))
    name_map = record.get('name_map', {})
    inv_map = {v: k for k, v in name_map.items()}
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    print('[prereg] committed; scorer committed; commit %s; splits %s (pooled)' % (commit[:7], list(s15.ALL_SPLITS)))

    # ------------------------------------------------------------------ PRISM labels, as Step 15d
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
    if hashes != rec15d['prism_files']:
        sys.exit('FATAL: PRISM file hashes differ from the Step 15d record')
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
    if len(label) != rec15d['n_labels']:
        sys.exit('FATAL: %d labels, Step 15d record has %d' % (len(label), rec15d['n_labels']))
    print('[prism] %d lines | %d labels (as Step 15d)' % (len(dep_of_line), len(label)))

    pc15 = pd.read_csv(s15d.PC15)
    per_split, pair_rows, long_rows = {}, [], []
    lv_ref = None
    for split in s15.ALL_SPLITS:
        print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
        k = split.split('_')[-1]
        rec_split = record['splits'][k]
        n_drugs = len(rec_split['held_structures'])
        arms = {arm: [s15c.load_vectors(split, arm, s) for s in range(s15.N_SEEDS)] for arm in s15.ARMS}
        ref = arms['recipe'][0]
        cond, n = ref['cond'], len(ref['cond'])
        if n != int(rec_split['n_test_conditions']):
            sys.exit('FATAL: %d conditions, record expects %d' % (n, rec_split['n_test_conditions']))
        for arm, L in arms.items():
            for v in L:
                if not np.array_equal(v['cond'], cond) or np.abs(v['C'] - ref['C']).max() > 1e-5 or np.abs(v['true'] - ref['true']).max() > 1e-5:
                    sys.exit('FATAL: condition order, control means or true means differ in %s' % v['tag'])
        true_mean, C = ref['true'], ref['C']
        Y = true_mean - C
        pc = pc15[pc15['split'] == split].reset_index(drop=True)
        if len(pc) != n or not np.array_equal(pc['condition'].astype(str).values, cond):
            sys.exit('FATAL: the Step 15b per-condition record does not list the conditions of %s in the vectors order' % split)
        lines, drugs, doses = pc['line'].astype(str).values, pc['structure'].astype(str).values, pc['dose_nM'].astype(float).values
        lv = sorted(set(lines))
        if lv_ref is None:
            lv_ref = lv
        elif lv != lv_ref:
            sys.exit('FATAL: line set differs between splits')
        if len(np.unique(drugs)) != n_drugs:
            sys.exit('FATAL: %d structures in the record, expected %d' % (len(np.unique(drugs)), n_drugs))
        levels = sorted(set(np.round(doses, 6)))
        if len(levels) != 3:
            sys.exit('FATAL: %d dose levels, expected 3' % len(levels))
        names = np.array([inv_map.get(s15d.cond_name(c, lines[i]), s15d.cond_name(c, lines[i])) for i, c in enumerate(cond)])

        # --- baselines by the Step 15b code path, reproduction guard, training effects kept
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

        # --- highest available dose per drug: training side (Step 15d variant (a), verbatim)
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
        print('[training] %d training pairs (%d drugs) at each drug\'s highest available dose' % (len(tr_keys2), len(set(g2))))
        f2, alpha_f2, mse_f2, r2_f2 = s15d.fit_ridge((Etr2 - mu2) / sd2, ytr2, g2, 'f, highest dose')
        FPq2 = np.array([s15.Drug_dose_encoder([smi_c[tp2[key][0]]], [hd_t[key[1]]], comb_num=1)[0] for key in tr_keys2], dtype=np.float64)
        qsar2, alpha_q2, mse_q2, r2_q2 = s15d.fit_ridge(s15.features(FPq2, np.array([ln for ln, _ in tr_keys2]), lv), ytr2, g2, 'qsar, highest dose')
        del Ytr

        # --- held-out side
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

        vals, per_seed = {}, {}
        for bname, D in deltas.items():
            vals[bname] = through(D)
        for arm, L in arms.items():
            per_seed[arm] = [through(v['B'] - v['C']) for v in L]
            vals[arm] = np.mean(np.vstack(per_seed[arm]), axis=0)
        vals['ceiling'] = through(Y)
        FPh2 = np.array([s15.Drug_dose_encoder([drugs[hp2[key][0]]], [hd_h[key[1]]], comb_num=1)[0] for key in keys2], dtype=np.float64)
        vals['qsar'] = qsar2.predict(s15.features(FPh2, np.array([ln for ln, _ in keys2]), lv))
        vals['effect_size'] = np.mean(np.vstack([-np.linalg.norm(pe2(v['B'] - v['C']), axis=1) for v in arms['recipe']]), axis=0)
        vals['effect_size_measured'] = -np.linalg.norm(pe2(Y), axis=1)
        for bname in ('control_paste', 'mean_response'):
            for l in set(line_ix2):
                if np.ptp(vals[bname][line_ix2 == l]) > 1e-9:
                    sys.exit('FATAL: %s predictions are not identical within a line' % bname)
        all_sel = np.arange(len(keys2))
        U = {nm: s15d.U_of(v_, labels2, line_ix2, len(lv), all_sel)[0] for nm, v_ in vals.items()}
        expect = rec15d['per_split'][split]['exploratory']['variants']['highest_dose']['U']
        for nm in ASSERTED:
            if abs(U[nm] - expect[nm]) > 1e-9:
                sys.exit('FATAL: U_s(%s) = %.9f does not reproduce the Step 15d record %.9f' % (nm, U[nm], expect[nm]))
        print('[held-out] %d labelled pairs (%d drugs) | U_s reproduces the Step 15d highest-dose record to 1e-9 for %s' % (len(keys2), len(set(drug_names2)), ', '.join(ASSERTED)))
        for nm in vals:
            print('  U_s %-22s %+.4f' % (nm, U[nm]))
        U_seed = {arm: [s15d.U_of(v_, labels2, line_ix2, len(lv), all_sel)[0] for v_ in per_seed[arm]] for arm in s15.ARMS}
        rho_line = {nm: {lv[l]: s15d.spearman0(vals[nm][line_ix2 == l], labels2[line_ix2 == l]) for l in sorted(set(line_ix2)) if (line_ix2 == l).sum() >= s15d.MIN_PAIRS_LINE} for nm in vals}
        per_split[split] = dict(keys=keys2, labels=labels2, line_ix=line_ix2, drug_ix=drug_ix2, drug_names=drug_names2, vals=vals, U=U, U_seed=U_seed,
                                rho_line=rho_line, n_drugs=int(len(set(drug_names2))), n_pairs=len(keys2),
                                cv=dict(f=dict(alpha=alpha_f2, mse=mse_f2, r2=r2_f2), qsar=dict(alpha=alpha_q2, mse=mse_q2, r2=r2_q2), label_variance=float(np.var(ytr2))),
                                n_training_pairs=len(tr_keys2), n_training_drugs=int(len(set(g2))))
        for p_, key in enumerate(keys2):
            row = dict(split=split, line=key[0], drug=key[1], auc=labels2[p_], n_conditions=len(hp2[key]), dose_nM=hd_h[key[1]])
            row.update({'f_%s' % nm: float(vals[nm][p_]) for nm in vals})
            pair_rows.append(row)
        del arms

    # ------------------------------------------------------------------ pooling
    splits = list(s15.ALL_SPLITS)
    subs = list(per_split[splits[0]]['vals'])
    nsd = {s: per_split[s]['n_drugs'] for s in splits}
    wsum = float(sum(nsd.values()))
    print('\n[pooling] labelled held-out drugs per split %s (total %d) | weights n_s / %d' % (nsd, int(wsum), int(wsum)))

    def U_pool_of(sel):                                   # sel: {split: index array into that split's pairs}
        out = {}
        for nm in subs:
            acc = 0.0
            for s in splits:
                ps = per_split[s]
                u = U_fast(ps['vals'][nm], ps['labels'], ps['line_ix'], len(lv_ref), sel[s])
                acc += nsd[s] * (u if np.isfinite(u) else 0.0)
            out[nm] = acc / wsum
        return out

    def U_B_of(sel):                                      # reading B: per-(split, line) centred values pooled per line
        out = {}
        for nm in subs:
            rhos = []
            for l in range(len(lv_ref)):
                xs, ys = [], []
                for s in splits:
                    ps = per_split[s]
                    m = sel[s][ps['line_ix'][sel[s]] == l]
                    if len(m) == 0:
                        continue
                    v_ = ps['vals'][nm][m]
                    xc = v_ - v_.mean()
                    xc[np.abs(xc) < 1e-9] = 0.0                  # a submission constant within (split, line) stays constant
                    xs.append(xc)
                    ys.append(ps['labels'][m])
                if xs:
                    x, y = np.concatenate(xs), np.concatenate(ys)
                    if len(x) >= s15d.MIN_PAIRS_LINE:
                        rhos.append(spearman_fast(x, y))
            out[nm] = float(np.mean(rhos)) if rhos else np.nan
        return out

    def W_pool_of(drug_sel):                              # within-drug across-line, mean over the selected drugs of every split
        out = {}
        for nm in ('recipe', 'ridge_train', 'ceiling', 'qsar'):
            rhos = []
            for s in splits:
                ps = per_split[s]
                for d in drug_sel[s]:
                    m = np.where(ps['drug_ix'] == d)[0]
                    if len(m) >= s15d.MIN_LINES_DRUG:
                        rhos.append(spearman_fast(ps['vals'][nm][m], ps['labels'][m]))
            out[nm] = float(np.mean(rhos)) if rhos else np.nan
        return out

    full_sel = {s: np.arange(per_split[s]['n_pairs']) for s in splits}
    full_drugs = {s: np.unique(per_split[s]['drug_ix']) for s in splits}
    for s in splits:                                      # self-check: the fast Spearman reproduces the Step 15d one
        ps = per_split[s]
        for nm in subs:
            if abs(U_fast(ps['vals'][nm], ps['labels'], ps['line_ix'], len(lv_ref), full_sel[s]) - ps['U'][nm]) > 1e-9:
                sys.exit('FATAL: fast Spearman differs from the Step 15d computation for %s / %s' % (s, nm))
    print('[self-check] fast Spearman equals the Step 15d per-line computation for every submission and split')
    UA, UB, W = U_pool_of(full_sel), U_B_of(full_sel), W_pool_of(full_drugs)
    for nm in subs:
        print('  pooled A %-22s %+.4f | reading B %+.4f' % (nm, UA[nm], UB[nm]))
    for nm, w in W.items():
        print('  within-drug (pooled) %-12s %+.4f' % (nm, w))

    rng = np.random.RandomState(0)
    groups = {s: [np.where(per_split[s]['drug_ix'] == d)[0] for d in full_drugs[s]] for s in splits}
    BA = {nm: np.full(N_BOOT, np.nan) for nm in subs}
    BB = {nm: np.full(N_BOOT, np.nan) for nm in subs}
    BW = {nm: np.full(N_BOOT, np.nan) for nm in W}
    for b in range(N_BOOT):
        sel, dsel = {}, {}
        for s in splits:
            draw = rng.randint(0, len(groups[s]), len(groups[s]))
            sel[s] = np.concatenate([groups[s][g] for g in draw])
            dsel[s] = full_drugs[s][draw]
        ua, ub, w = U_pool_of(sel), U_B_of(sel), W_pool_of(dsel)
        for nm in subs:
            BA[nm][b], BB[nm][b] = ua[nm], ub[nm]
        for nm in W:
            BW[nm][b] = w[nm]
    print('[bootstrap] %d stratified replicates (%d s)' % (N_BOOT, time.time() - T0))

    def ci(arr):
        return [float(np.nanpercentile(arr, LO)), float(np.nanpercentile(arr, HI))]

    def con(name, point, dist):
        lo, hi = ci(dist)
        c = dict(diff=float(point), ci=[lo, hi], verdict=s15.verdict(lo, hi), half_width=float((hi - lo) / 2))
        print('  %-54s %s' % (name, s15.fmt(c)))
        return c

    contrasts = {}
    contrasts['Q1_primary_Upool_recipe_minus_0'] = con('Q1 primary: U_pool recipe - 0', UA['recipe'], BA['recipe'])
    contrasts['Q2_coprimary_Upool_recipe_minus_ridge_train'] = con('Q2 co-primary: U_pool recipe - ridge-train', UA['recipe'] - UA['ridge_train'], BA['recipe'] - BA['ridge_train'])
    for nm in subs:
        if nm != 'recipe':
            contrasts['secondary_Upool_%s_minus_0' % nm] = con('secondary: U_pool %s - 0' % nm, UA[nm], BA[nm])
    for a_, b_ in (('recipe', 'ceiling'), ('recipe', 'qsar'), ('recipe', 'effect_size'), ('recipe', 'ridge_S7'), ('guidance_w2', 'recipe'), ('attn_cross', 'mlp_ctrl')):
        contrasts['secondary_Upool_%s_minus_%s' % (a_, b_)] = con('secondary: U_pool %s - %s' % (a_, b_), UA[a_] - UA[b_], BA[a_] - BA[b_])
    ceiling_above = contrasts['secondary_Upool_ceiling_minus_0']['verdict'] == 'above zero'
    ratio = dict(point=float(UA['recipe'] / UA['ceiling']) if UA['ceiling'] != 0 else None, ci=ci(BA['recipe'] / BA['ceiling']) if ceiling_above else None,
                 reported=bool(ceiling_above))
    print('  ratio U_pool(recipe) / U_pool(ceiling): %s %s' % ('%.3f' % ratio['point'] if ratio['point'] is not None else 'undefined',
                                                              ('[%.3f, %.3f]' % tuple(ratio['ci'])) if ratio['ci'] else '(ceiling not above zero: interval not reported)'))
    contrasts_B = {}
    contrasts_B['UB_recipe_minus_0'] = con('reading B: U_B recipe - 0', UB['recipe'], BB['recipe'])
    contrasts_B['UB_recipe_minus_ridge_train'] = con('reading B: U_B recipe - ridge-train', UB['recipe'] - UB['ridge_train'], BB['recipe'] - BB['ridge_train'])
    for nm in subs:
        if nm != 'recipe':
            contrasts_B['UB_%s_minus_0' % nm] = con('reading B: U_B %s - 0' % nm, UB[nm], BB[nm])
    for a_, b_ in (('recipe', 'ceiling'), ('recipe', 'qsar'), ('recipe', 'effect_size')):
        contrasts_B['UB_%s_minus_%s' % (a_, b_)] = con('reading B: U_B %s - %s' % (a_, b_), UB[a_] - UB[b_], BB[a_] - BB[b_])
    within = {nm: con('within-drug pooled: %s - 0' % nm, W[nm], BW[nm]) for nm in W}

    # per-line pooled U (reading A weights) and brain / gut
    per_line = {}
    for ln in lv_ref:
        for nm in ('recipe', 'ridge_train', 'ceiling', 'qsar', 'effect_size_measured'):
            acc, wt = 0.0, 0.0
            for s in splits:
                r = per_split[s]['rho_line'][nm].get(ln)
                if r is not None:
                    acc += nsd[s] * r
                    wt += nsd[s]
            per_line.setdefault(ln, {})[nm] = (acc / wt) if wt else None
    brain_gut = {}
    for nm in ('recipe', 'ridge_train', 'ceiling', 'qsar', 'effect_size_measured'):
        b_ = [per_line[ln][nm] for ln in lv_ref if ln in BRAIN and per_line[ln][nm] is not None]
        g_ = [per_line[ln][nm] for ln in lv_ref if ln not in BRAIN and per_line[ln][nm] is not None]
        brain_gut[nm] = dict(brain=float(np.mean(b_)) if b_ else None, gut=float(np.mean(g_)) if g_ else None, n_brain=len(b_), n_gut=len(g_))
        print('  per-line pooled %-22s brain %s (%d lines) | gut %s (%d lines)' % (nm, 'n/a' if not b_ else '%+.3f' % np.mean(b_), len(b_), 'n/a' if not g_ else '%+.3f' % np.mean(g_), len(g_)))

    # section 6, precedence 4 -> 1 -> 2 -> 3
    v1 = contrasts['Q1_primary_Upool_recipe_minus_0']['verdict']
    v2 = contrasts['Q2_coprimary_Upool_recipe_minus_ridge_train']['verdict']
    if not ceiling_above:
        cat = '4 LABEL SET CANNOT RESOLVE'
    elif v1 == 'above zero' and v2 == 'above zero':
        cat = '1 SENSITIVITY INFORMATION BEYOND LINEAR'
    elif v1 == 'above zero':
        cat = '2 SENSITIVITY INFORMATION, NOT BEYOND LINEAR'
    else:
        cat = '3 NO RESOLVED SENSITIVITY INFORMATION'
    print('\n[section 6] primary %s | co-primary %s | ceiling %s -> %s' % (v1, v2, contrasts['secondary_Upool_ceiling_minus_0']['verdict'], cat))

    out = dict(prereg=PRE, commit=commit, step15d_record_commit=rec15d.get('commit'), splits=splits, prism_files=hashes, n_labels=len(label),
               n_drugs_per_split=nsd, n_drugs_total=int(wsum), n_pairs_per_split={s: per_split[s]['n_pairs'] for s in splits},
               cv_per_split={s: per_split[s]['cv'] for s in splits}, training_pairs_per_split={s: dict(pairs=per_split[s]['n_training_pairs'], drugs=per_split[s]['n_training_drugs']) for s in splits},
               U_per_split={s: {nm: float(v) for nm, v in per_split[s]['U'].items()} for s in splits},
               U_seed_per_split={s: per_split[s]['U_seed'] for s in splits},
               U_pool={nm: float(v) for nm, v in UA.items()}, U_B={nm: float(v) for nm, v in UB.items()}, within_drug={nm: float(v) for nm, v in W.items()},
               contrasts=contrasts, contrasts_B=contrasts_B, within_contrasts=within, ratio=ratio, per_line=per_line, brain_gut=brain_gut,
               n_boot=N_BOOT, verdict=cat, elapsed_s=int(time.time() - T0))
    json.dump(out, open(OUT, 'w'), indent=1)
    for nm in subs:
        long_rows.append(dict(quantity='U_pool', submission=nm, value=UA[nm]))
        long_rows.append(dict(quantity='U_B', submission=nm, value=UB[nm]))
        for s in splits:
            long_rows.append(dict(quantity='U_split_%s' % s, submission=nm, value=per_split[s]['U'][nm]))
    for cname, c in list(contrasts.items()) + list(contrasts_B.items()) + [('within_%s' % k, v) for k, v in within.items()]:
        long_rows.append(dict(quantity='contrast', submission=cname, value=c['diff'], ci_lo=c['ci'][0], ci_hi=c['ci'][1], verdict=c['verdict']))
    pd.DataFrame(long_rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(pair_rows).to_csv(OUT_PAIRS, index=False)
    print('wrote %s, %s, %s (%d s)' % (OUT, OUT_CSV, OUT_PAIRS, time.time() - T0))


if __name__ == '__main__':
    main()
