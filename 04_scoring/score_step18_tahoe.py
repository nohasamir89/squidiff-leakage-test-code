#!/usr/bin/env python
"""Step 18 scoring: the OT-paired arm against the recipe on the frozen Tahoe test sets.

Pre-registration: PREREG_step18_tahoe_ot_pairing.md (sections 4-6, 9), committed with the code before any Step 18 model
exists. This script does only what that document says, by the Step 15b/15c/16 code paths:

  1. guards: the pre-registration, this scorer and the scorers it imports committed and clean; the Step 15b and 15c
     records present and committed; every Step 15b array and every Step 18 array hash-checked against its provenance;
     the ot_pair provenance pinned to the OT control file of the pairing record;
  2. the four shift baselines rebuilt by the Step 15b code path and asserted to reproduce the Step 15b per-condition
     record to 1e-9 (DE-overlap, DPC-50) with every Step 15b arm; PDS of all nine Step 15b submissions asserted to
     reproduce the Step 15c per-condition record to 1e-9;
  3. the recipe re-evaluations under the Step 18 harness asserted to carry the Step 15b conditions, true means and
     control means (bitwise) and mean vectors within 1e-3; the recipe is scored from its Step 15b arrays;
  4. the new arm: DE-overlap, DPC-50, energy against the stored real cells (asserted identical to the recipe's), PDS on
     the Step 15c sets, per seed and seed-averaged per condition; MRA (Step 16 section 4.4) with the replicate ceiling;
     the cross-plate PDS facet (Step 15c section 8);
  5. control tracking (section 4.4): per condition, the mean cosine between each generated cell's deviation from the
     generated mean and its conditioning control cell's deviation from the control mean, for the recipe (from its
     re-evaluations) and ot_pair; and the same against a within-condition permutation of the control cells (the null);
  6. the section-5 contrasts (structure-clustered percentile bootstrap, 5,000 replicates, RandomState(0), 97.5 % per
     split; MRA recomputed on the resampled drugs), the secondaries of section 4, and the section-5 verdict when all
     three splits are present.

Usage: python -u score_step18_tahoe.py --splits drug_split_1      (development run first; no --splits = final)
Outputs: tahoe_results/step18_analysis[_development].json, step18_long[_development].csv,
         step18_per_condition[_development].csv. Never overwrites.
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402  loaders, metrics, baselines, bootstrap
import score_step15c_tahoe as s15c                    # noqa: E402  pds_sets, load_vectors, close
import score_step16_tahoe as s16                      # noqa: E402  rebuild, prep_ctx, mra_pre, ctx_split, verdict_of, contrast_pc

PRE = 'PREREG_step18_tahoe_ot_pairing.md'
PC15, PC15C, REC15B, REC15C = s16.PC15, s16.PC15C, s16.REC15B, s16.REC15C
REC16 = os.path.join(s15.RES, 'step16_analysis.json')
OTREC = os.path.join(s15.RES, 'ot_pairing_check.json')
NEW = 'ot_pair'
OLD_ARMS = tuple(s15.ARMS)
SUBS = list(s15.BASELINES) + list(OLD_ARMS) + [NEW]
MIN_CTX, N_BOOT, LO, HI = s16.MIN_CTX, s16.N_BOOT, s16.LO, s16.HI
T0 = time.time()


# ------------------------------------------------------------------ inputs
def load_eval18(split, seed, kind, ot_sha):
    """One certified Step 18 evaluation (kind 'ot': the new arm; 'recipe': a Step 15b recipe checkpoint re-evaluated under
    the Step 18 harness): mean vectors, per-cell populations with their conditioning control cells, provenance."""
    if kind == 'ot':
        tag, pj = 's18_%s_ot_pair_seed_%d' % (split, seed), 'provenance/s18ev_s18_%s_ot_pair_seed_%d.json' % (split, seed)
        arm = 'ot_pair'
    else:
        tag, pj = 's18r_%s_recipe_seed_%d' % (split, seed), 'provenance/s18rv_s18r_%s_recipe_seed_%d.json' % (split, seed)
        arm = 'recipe'
    f = os.path.join(s15.RES, 'vectors_%s.npz' % tag)
    fpc = os.path.join(s15.RES, 'vectors_%s_pc.npz' % tag)
    for q in (f, fpc, pj):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    pv = json.load(open(pj))
    s15.check_prov(pv, tag, dict(status='completed', kind=kind, arm=arm, split=split, seed=seed, drug_encoding='fingerprint',
                                 drug_dimension=1024, encoder_arm='mlp', guidance_w=None, nfe=50, max_cells_per_cond=s15.CAP, ctrl_stored=True))
    if kind == 'ot' and pv.get('control_ot_sha256') != ot_sha:
        sys.exit('FATAL: %s was trained on a control file other than the pinned OT file' % tag)
    if pv.get('vectors_sha256') != s15.sha(f):
        sys.exit('FATAL: %s vectors hash differs from provenance' % tag)
    if pv.get('per_cell_sha256') != s15.sha(fpc):
        sys.exit('FATAL: %s per-cell hash differs from provenance' % tag)
    v = np.load(f, allow_pickle=True)
    p = np.load(fpc, allow_pickle=True)
    if 'ctrl' not in p.files:
        sys.exit('FATAL: %s has no ctrl key' % fpc)
    out = dict(tag=tag, cond=np.array([str(x) for x in v['cond']]), true=np.asarray(v['true'], np.float64),
               B=np.asarray(v['B'], np.float64), C=np.asarray(v['C'], np.float64),
               pred=[np.asarray(a, np.float32) for a in p['pred']], true_cells=[np.asarray(a, np.float32) for a in p['true']],
               ctrl=[np.asarray(a, np.float32) for a in p['ctrl']],
               rec=dict(tag=tag, kind=kind, vectors_sha256=pv['vectors_sha256'], per_cell_sha256=pv['per_cell_sha256'],
                        checkpoint_sha256=pv.get('checkpoint_sha256'), training_commit=pv.get('training_commit'),
                        eval_commit=pv.get('commit'), control_ot_sha256=pv.get('control_ot_sha256'), reproduction=pv.get('reproduction')))
    n = len(out['cond'])
    if len(out['pred']) != n or len(out['true_cells']) != n or len(out['ctrl']) != n:
        sys.exit('FATAL: %s per-cell length differs from vectors' % tag)
    for i in range(n):
        if out['ctrl'][i].shape != out['pred'][i].shape:
            sys.exit('FATAL: %s ctrl / pred shapes differ in condition %d' % (tag, i))
    return out


def tracking(pred, ctrl, rng=None):
    """Per condition: the mean over cells of the cosine between (generated cell - generated mean) and (its conditioning
    control cell - control mean); with rng, the control cells are permuted within the condition first (the null)."""
    out = np.full(len(pred), np.nan)
    for i, (P, K) in enumerate(zip(pred, ctrl)):
        P, K = np.asarray(P, np.float64), np.asarray(K, np.float64)
        if rng is not None:
            K = K[rng.permutation(len(K))]
        Pc, Kc = P - P.mean(axis=0, keepdims=True), K - K.mean(axis=0, keepdims=True)
        den = np.linalg.norm(Pc, axis=1) * np.linalg.norm(Kc, axis=1)
        ok = den > 0
        if ok.any():
            out[i] = float(np.mean((Pc[ok] * Kc[ok]).sum(axis=1) / den[ok]))
    return out


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
    OUT = os.path.join(s15.RES, 'step18_analysis%s.json' % suffix)
    OUT_CSV = os.path.join(s15.RES, 'step18_long%s.csv' % suffix)
    OUT_PC = os.path.join(s15.RES, 'step18_per_condition%s.csv' % suffix)
    for q in (OUT, OUT_CSV, OUT_PC):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 9:
        sys.exit('FATAL: prereg incomplete')
    tracked = ('%s score_step18_tahoe.py score_step16_tahoe.py score_step15_tahoe.py score_step15c_tahoe.py score_step15d_utility.py '
               'score_step15e_pooled.py step14_metrics.py Squidiff/scrna_datasets.py %s %s %s %s %s %s %s' % (PRE, PC15, PC15C, REC15B, REC15C, REC16, OTREC, s15.RECORD))
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers or records not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers or records uncommitted')
    record = json.load(open(s15.RECORD))
    otrec = json.load(open(OTREC))
    rec15b, rec15c, rec16 = json.load(open(REC15B)), json.load(open(REC15C)), json.load(open(REC16))
    for nm, r in (('15b', rec15b), ('15c', rec15c), ('16', rec16)):
        if not r.get('final'):
            sys.exit('FATAL: the Step %s record is not the final one' % nm)
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    print('[prereg] committed; scorer committed; commit %s; splits %s (%s)' % (commit[:7], splits, 'final' if final else 'development'))
    s15c.self_check()
    s16.mra_self_check()
    pc15, pc15c = pd.read_csv(PC15), pd.read_csv(PC15C)

    results, inputs, rows, pc_rows = {}, {}, [], []
    for split in splits:
        print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
        k = split.split('_')[-1]
        rec_split = record['splits'][k]
        n_drugs = len(rec_split['held_structures'])
        ot_sha = otrec['written'][split]['sha256']

        # --- Step 15b mean vectors (15 runs), the recipe's per-cell file for the real cells, the Step 18 runs
        old = {arm: [s15c.load_vectors(split, arm, s) for s in range(s15.N_SEEDS)] for arm in OLD_ARMS}
        ref = old['recipe'][0]
        cond, n = ref['cond'], len(ref['cond'])
        if n != int(rec_split['n_test_conditions']):
            sys.exit('FATAL: %d conditions, record expects %d' % (n, rec_split['n_test_conditions']))
        ref_pc = s15.load_eval(split, 'recipe', 0)
        if not np.array_equal(ref_pc['cond'], cond):
            sys.exit('FATAL: the recipe per-cell file lists different conditions')
        real = ref_pc['true_cells']
        del ref_pc
        true_mean, C = ref['true'], ref['C']
        Y = true_mean - C
        for arm, L in old.items():
            for v in L:
                if not np.array_equal(v['cond'], cond) or np.abs(v['C'] - C).max() > 1e-5 or np.abs(v['true'] - true_mean).max() > 1e-5:
                    sys.exit('FATAL: condition order, control means or true means differ in %s' % v['tag'])

        # --- the recipe re-evaluations (tracking only) and the new arm, loaded one at a time; per-cell arrays dropped after use
        track, track_null, inp = {}, {}, {}
        seed_vals = dict(jac=[], dpc=[], en=[], pds=[])
        new_B = []
        rng_null = np.random.RandomState(0)
        for kind, arm in (('recipe', 'recipe'), ('ot', NEW)):
            tr_, tn_ = [], []
            inp[arm] = []
            for s in range(s15.N_SEEDS):
                v = load_eval18(split, s, kind, ot_sha)
                if not np.array_equal(v['cond'], cond):
                    sys.exit('FATAL: condition order differs in %s' % v['tag'])
                for i in range(n):
                    if not np.array_equal(v['true_cells'][i], real[i]):
                        sys.exit('FATAL: stored real cells differ in %s, condition %d' % (v['tag'], i))
                if kind == 'recipe':
                    r15 = old['recipe'][s]
                    if not np.array_equal(v['true'], r15['true']) or not np.array_equal(v['C'], r15['C']):
                        sys.exit('FATAL: %s true or C differ from the Step 15b arrays' % v['tag'])
                    dB = float(np.abs(v['B'] - r15['B']).max())
                    if dB > 1e-3:
                        sys.exit('FATAL: %s mean vectors differ from the Step 15b arrays by %.2e' % (v['tag'], dB))
                    print('[repro] %s: true, C, cond bitwise as Step 15b; max |dB| %.2e' % (v['tag'], dB))
                else:
                    if np.abs(v['C'] - C).max() > 1e-5 or np.abs(v['true'] - true_mean).max() > 1e-5:
                        sys.exit('FATAL: control or true means differ in %s' % v['tag'])
                tr_.append(tracking(v['pred'], v['ctrl']))
                tn_.append(tracking(v['pred'], v['ctrl'], rng_null))
                if kind == 'ot':
                    j, d = s15.thesis_per_cond(v['true'], v['B'], v['C'])
                    e_ = np.array([s15.energy(v['pred'][i], real[i]) for i in range(n)])
                    seed_vals['jac'].append(j); seed_vals['dpc'].append(d); seed_vals['en'].append(e_)
                    new_B.append(v['B'])
                    print('[arm] %-8s seed %d  DE-overlap %.4f  energy %.4f  DPC-50 mean %.4f  tracking %.4f (null %.4f)'
                          % (NEW, s, np.nanmean(j), e_.mean(), np.nanmean(d), np.nanmean(tr_[-1]), np.nanmean(tn_[-1])))
                else:
                    print('[arm] %-8s seed %d  tracking %.4f (null %.4f)' % ('recipe', s, np.nanmean(tr_[-1]), np.nanmean(tn_[-1])))
                inp[arm].append(v['rec'])
                del v
            track[arm] = np.nanmean(np.vstack(tr_), axis=0)
            track_null[arm] = np.nanmean(np.vstack(tn_), axis=0)
            track[arm + '_per_seed'] = [float(np.nanmean(x)) for x in tr_]
        del real
        inputs[split] = dict({arm: [v['rec'] for v in L] for arm, L in old.items()}, **inp)

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
        if len(np.unique(drugs)) != n_drugs:
            sys.exit('FATAL: %d structures in the record, expected %d' % (len(np.unique(drugs)), n_drugs))
        ib = np.isin(lines, s15.BRAIN)
        print('[design] %d conditions | %d held-out structures | %d lines | cells per condition up to %d' % (n, n_drugs, len(lv), s15.CAP))

        # --- baselines by the Step 15b code path; reproduction guards against the Step 15b and 15c records
        deltas, train = s16.rebuild(split, lines, drugs, doses, lv, Y)
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
        cand = {}
        for members in contexts:
            for p in members:
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

        # --- the new arm's per-condition metrics (seed means) next to the Step 15b ones
        jac, dpc, en = {}, {}, {}
        for nm in list(deltas) + list(OLD_ARMS):
            jac[nm], dpc[nm], en[nm] = pc['%s_de_overlap' % nm].values, pc['%s_dpc50' % nm].values, pc['%s_energy' % nm].values
        jac[NEW] = np.nanmean(np.vstack(seed_vals['jac']), axis=0)
        dpc[NEW] = np.nanmean(np.vstack(seed_vals['dpc']), axis=0)
        en[NEW] = np.nanmean(np.vstack(seed_vals['en']), axis=0)
        pds_seed[NEW] = [s15c.pds_sets(B - C, Y, contexts, cand) for B in new_B]
        pds[NEW] = np.nanmean(np.vstack(pds_seed[NEW]), axis=0)
        for s_, p_ in enumerate(pds_seed[NEW]):
            print('[arm] %-8s seed %d  PDS %.4f' % (NEW, s_, np.nanmean(p_)))

        # --- MRA (Step 16 section 4.4) with the replicate ceiling
        true_norm = np.linalg.norm(Y, axis=1)
        norms = {bname: [np.linalg.norm(D, axis=1)] for bname, D in deltas.items()}
        for arm, L in old.items():
            norms[arm] = [np.linalg.norm(v['B'] - v['C'], axis=1) for v in L]
        norms[NEW] = [np.linalg.norm(B - C, axis=1) for B in new_B]
        n_ctx = len(contexts)
        full = np.arange(n)
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
        rep_norm = np.zeros(n)
        rep_norm[rep_ix] = true_norm[partner[rep_ix]]

        def mra_all(sel):
            pre = s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, n_ctx, sel))
            out = {nm: float(np.mean([s16.mra_pre(x, pre) for x in norms[nm]])) for nm in norms}
            s_ = sel[partner[sel] >= 0]
            out['ceiling'] = s16.mra_pre(rep_norm, s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, n_ctx, s_))) if len(s_) else np.nan
            return out

        pre_full = s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, n_ctx, full))
        mra_seed = {nm: [s16.mra_pre(x, pre_full) for x in norms[nm]] for nm in norms}
        mra = {nm: float(np.mean(v_)) for nm, v_ in mra_seed.items()}
        full_all = mra_all(full)
        for nm in ('control_paste', 'mean_response'):
            if abs(mra[nm]) > 1e-12:
                sys.exit('FATAL: %s MRA is %.3g, expected 0 by definition' % (nm, mra[nm]))
        mra_ceil = full_all['ceiling']
        n_ctx_rep = sum(1 for c in range(n_ctx) if (ctx_id[rep_ix] == c).sum() >= MIN_CTX)
        for nm in list(deltas) + list(OLD_ARMS):                        # the Step 16 record holds the same MRA for these
            want = rec16['per_split'][split]['metrics'][nm]['mra']
            if abs(mra[nm] - float(want)) > 1e-9:
                sys.exit('FATAL: MRA of %s (%.9f) does not reproduce the Step 16 record (%.9f)' % (nm, mra[nm], want))
        if (mra_ceil is None) != (rec16['per_split'][split]['replicate']['mra'] is None) or (np.isfinite(mra_ceil) and abs(mra_ceil - float(rec16['per_split'][split]['replicate']['mra'])) > 1e-9):
            sys.exit('FATAL: the replicate MRA ceiling does not reproduce the Step 16 record')
        print('[reproduction] MRA of the nine Step 15b submissions and the replicate ceiling equal the Step 16 record to 1e-9')
        print('[mra] %d contexts | replicate ceiling %s over %d contexts (%d pairs)' % (n_ctx, 'n/a' if not np.isfinite(mra_ceil) else '%.4f' % mra_ceil, n_ctx_rep, len(pairs)))
        for nm in SUBS:
            print('[mra] %-14s %+.4f | per-seed %s' % (nm, mra[nm], [round(x, 4) for x in mra_seed[nm]]))

        # --- cross-plate PDS facet (Step 15c section 8) for the recipe and the new arm, plus the ceiling
        members_of = {u: np.where(ctx_key == u)[0] for u in ctx_u}
        xp = {'recipe': [], NEW: []}
        xp_ceil = []
        pred_of = {'recipe': [v['B'] - v['C'] for v in old['recipe']], NEW: [B - C for B in new_B]}
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
        print('[cross-plate] %d evaluations | ceiling %s | recipe %s | %s %s'
              % (len(xp_ceil), *['n/a' if crossplate[q] is None else '%.4f' % crossplate[q] for q in ('ceiling', 'recipe')], NEW, 'n/a' if crossplate[NEW] is None else '%.4f' % crossplate[NEW]))

        # --- absolute values per submission
        metrics = {}
        for nm in SUBS:
            x = pds[nm]
            m_ = dict(pds_mean=float(np.nanmean(x)), pds_median=float(np.nanmedian(x)),
                      de_overlap=float(np.nanmean(jac[nm])) if np.isfinite(jac[nm]).any() else None,
                      energy=float(np.nanmean(en[nm])), dpc_mean=float(np.nanmean(dpc[nm])), mra=mra[nm],
                      pds_per_seed=[float(np.nanmean(s_)) for s_ in pds_seed[nm]], mra_per_seed=mra_seed[nm],
                      pds_per_line={ln: float(np.nanmean(x[lines == ln])) for ln in lv},
                      pds_brain_minus_gut=float(np.nanmean(x[ib]) - np.nanmean(x[~ib])))
            if nm == NEW:
                m_['de_per_seed'] = [float(np.nanmean(s_)) for s_ in seed_vals['jac']]
                m_['energy_per_seed'] = [float(np.nanmean(s_)) for s_ in seed_vals['en']]
                m_['dpc_per_seed'] = [float(np.nanmean(s_)) for s_ in seed_vals['dpc']]
            elif nm in OLD_ARMS:
                m_['de_per_seed'] = [s_['de_overlap'] for s_ in rec15b['per_split'][split]['metrics'][nm]['per_seed']]
                m_['energy_per_seed'] = [s_['energy'] for s_ in rec15b['per_split'][split]['metrics'][nm]['per_seed']]
            if nm in track:
                m_['tracking'] = float(np.nanmean(track[nm])); m_['tracking_null'] = float(np.nanmean(track_null[nm])); m_['tracking_per_seed'] = track[nm + '_per_seed']
            metrics[nm] = m_
            for key_ in ('pds_mean', 'pds_median', 'de_overlap', 'energy', 'dpc_mean', 'mra'):
                rows.append(dict(split=split, level='ALL', key=key_, submission=nm, value=m_[key_]))
            print('[abs] %-14s PDS %.4f | DE-overlap %s | energy %.4f (seeds %s) | DPC-50 %.4f | MRA %+.4f%s'
                  % (nm, m_['pds_mean'], 'n/a' if m_['de_overlap'] is None else '%.4f' % m_['de_overlap'], m_['energy'],
                     [round(v_, 4) for v_ in m_.get('energy_per_seed', [])], m_['dpc_mean'], m_['mra'],
                     (' | tracking %+.4f (null %+.4f)' % (m_['tracking'], m_['tracking_null'])) if 'tracking' in m_ else ''))
        replicate = dict(pds=rec15c['per_split'][split]['replicate'], de_overlap_energy_dpc=rec15b['per_split'][split]['replicate'],
                         mra=mra_ceil if np.isfinite(mra_ceil) else None, mra_n_contexts=n_ctx_rep, n_pairs=len(pairs))
        for i in range(n):
            r = dict(split=split, condition=cond[i], line=lines[i], plate=plates[i], structure=drugs[i], dose_nM=doses[i],
                     n_candidates=int(n_cand[i]), learnability=learn[i], true_effect_norm=float(true_norm[i]),
                     recipe_tracking=track['recipe'][i], recipe_tracking_null=track_null['recipe'][i],
                     ot_pair_tracking=track[NEW][i], ot_pair_tracking_null=track_null[NEW][i])
            for nm in SUBS:
                r['%s_pds' % nm] = pds[nm][i]; r['%s_de_overlap' % nm] = jac[nm][i]; r['%s_dpc50' % nm] = dpc[nm][i]; r['%s_energy' % nm] = en[nm][i]
                r['%s_effect_norm' % nm] = float(np.mean([q[i] for q in norms[nm]]))
            pc_rows.append(r)

        # --- the pre-registered contrasts (section 5): per-condition metrics fixed under resampling
        con = {}
        con['Q1_primary_energy_ot_pair_minus_recipe'] = s16.contrast_pc(en[NEW], en['recipe'], drugs)
        con['Q2_gate_de_ot_pair_minus_recipe'] = s16.contrast_pc(jac[NEW], jac['recipe'], drugs)
        con['Q3_gate_pds_ot_pair_minus_recipe'] = s16.contrast_pc(pds[NEW], pds['recipe'], drugs)
        print('\nQ1 PRIMARY  energy,     ot_pair minus recipe   %s' % s16.fmt(con['Q1_primary_energy_ot_pair_minus_recipe']))
        print('Q2 GATE     DE-overlap, ot_pair minus recipe   %s' % s16.fmt(con['Q2_gate_de_ot_pair_minus_recipe']))
        print('Q3 GATE     PDS,        ot_pair minus recipe   %s' % s16.fmt(con['Q3_gate_pds_ot_pair_minus_recipe']))
        con['secondary_dpc_ot_pair_minus_recipe'] = s16.contrast_pc(dpc[NEW], dpc['recipe'], drugs)
        print('secondary   DPC-50,     ot_pair minus recipe   %s' % s16.fmt(con['secondary_dpc_ot_pair_minus_recipe']))
        con['secondary_pds_ot_pair_minus_0.5'] = s16.contrast_pc(pds[NEW], np.full(n, 0.5), drugs)
        print('secondary   PDS,        ot_pair minus 0.5      %s' % s16.fmt(con['secondary_pds_ot_pair_minus_0.5']))
        for arm in OLD_ARMS:
            if arm == 'recipe':
                continue
            con['secondary_energy_ot_pair_minus_%s' % arm] = s16.contrast_pc(en[NEW], en[arm], drugs)
        con['exploratory_tracking_ot_pair_minus_recipe'] = s16.contrast_pc(track[NEW], track['recipe'], drugs)
        con['exploratory_tracking_recipe_minus_null'] = s16.contrast_pc(track['recipe'], track_null['recipe'], drugs)
        con['exploratory_tracking_ot_pair_minus_null'] = s16.contrast_pc(track[NEW], track_null[NEW], drugs)
        print('exploratory tracking,   ot_pair minus recipe   %s' % s16.fmt(con['exploratory_tracking_ot_pair_minus_recipe']))
        print('exploratory tracking,   recipe minus its null  %s' % s16.fmt(con['exploratory_tracking_recipe_minus_null']))
        print('exploratory tracking,   ot_pair minus its null %s' % s16.fmt(con['exploratory_tracking_ot_pair_minus_null']))
        gain = en['recipe'] - en[NEW]
        rho, pval = spearmanr(learn, gain)
        print('exploratory learnability vs energy gain of ot_pair over the recipe: Spearman %+.4f (p %.2e)' % (rho, pval))

        # --- MRA contrasts: recomputed per replicate on the resampled drugs
        rng = np.random.RandomState(0)
        gs = [np.where(drugs == u)[0] for u in np.unique(drugs)]
        BM = {nm: np.full(N_BOOT, np.nan) for nm in norms}
        BC = np.full(N_BOOT, np.nan)
        for b in range(N_BOOT):
            sel = np.concatenate([gs[g] for g in rng.randint(0, len(gs), len(gs))])
            got = mra_all(sel)
            for nm in norms:
                BM[nm][b] = got[nm]
            BC[b] = got['ceiling']
        con['secondary_mra_ot_pair_minus_0'] = s16.verdict_of(mra[NEW], BM[NEW])
        con['secondary_mra_ot_pair_minus_recipe'] = s16.verdict_of(mra[NEW] - mra['recipe'], BM[NEW] - BM['recipe'])
        con['secondary_mra_recipe_minus_0'] = s16.verdict_of(mra['recipe'], BM['recipe'])
        if np.isfinite(mra_ceil):
            con['secondary_mra_ceiling_minus_0'] = s16.verdict_of(mra_ceil, BC)
        print('secondary   MRA,        ot_pair minus 0        %s' % s16.fmt(con['secondary_mra_ot_pair_minus_0']))
        print('secondary   MRA,        ot_pair minus recipe   %s' % s16.fmt(con['secondary_mra_ot_pair_minus_recipe']))
        print('[bootstrap] MRA: %d replicates (%d s)' % (N_BOOT, time.time() - T0))
        for key, c in con.items():
            rows.append(dict(split=split, level='contrast', key=key, submission='', value=c['diff'], ci_lo=c['ci'][0], ci_hi=c['ci'][1], verdict=c['verdict']))
        results[split] = dict(n_conditions=n, n_structures=n_drugs, n_contexts=n_ctx, min_candidates=s15c.MIN_CAND, min_context=MIN_CTX,
                              metrics=metrics, replicate=replicate, crossplate=crossplate, contrasts=con,
                              learnability_vs_energy_gain=dict(spearman=float(rho), p=float(pval)),
                              tracking=dict(recipe=float(np.nanmean(track['recipe'])), ot_pair=float(np.nanmean(track[NEW])),
                                            recipe_null=float(np.nanmean(track_null['recipe'])), ot_pair_null=float(np.nanmean(track_null[NEW])),
                                            recipe_per_seed=track['recipe_per_seed'], ot_pair_per_seed=track[NEW + '_per_seed']))
        del old, new_B

    # ------------------------------------------------------------------ section 5 (precedence 1 -> 2 -> 5 -> 3 -> 4)
    cat = None
    if final:
        q1 = {s: results[s]['contrasts']['Q1_primary_energy_ot_pair_minus_recipe']['verdict'] for s in s15.ALL_SPLITS}
        q2 = {s: results[s]['contrasts']['Q2_gate_de_ot_pair_minus_recipe']['verdict'] for s in s15.ALL_SPLITS}
        q3 = {s: results[s]['contrasts']['Q3_gate_pds_ot_pair_minus_recipe']['verdict'] for s in s15.ALL_SPLITS}
        v1 = list(q1.values()); gates = list(q2.values()) + list(q3.values())
        if all(v == 'below zero' for v in v1) and all(v != 'below zero' for v in gates):
            cat = '1 FIDELITY GAIN'
        elif all(v == 'below zero' for v in v1):
            cat = '2 FIDELITY GAIN AT A COST'
        elif all(v == 'includes zero' for v in v1):
            cat = '5 UNRESOLVED'
        elif 'above zero' in v1:
            cat = '3 REVERSED'
        else:
            cat = '4 PARTIAL'
        print('\n' + '=' * 72 + '\nPRE-REGISTERED VERDICT (section 5): %s\n  Q1 energy ot_pair - recipe per split: %s\n  Q2 DE-overlap per split: %s\n  Q3 PDS per split: %s\n'
              % (cat, q1, q2, q3) + '  Section 7 expected: 5 UNRESOLVED.\n' + '=' * 72)
    else:
        print('\n[development] %s scored; section 5 is applied only when all three splits are present' % ', '.join(splits))

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'level', 'key', 'submission', 'value', 'ci_lo', 'ci_hi', 'verdict'])
        w.writeheader(); w.writerows(rows)
    with open(OUT_PC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0].keys()))
        w.writeheader(); w.writerows(pc_rows)
    json.dump(dict(prereg=PRE, final=final, verdict=cat, splits=splits, interval='97.5% percentile per split, structure-clustered',
                   n_boot=N_BOOT, n_seeds=s15.N_SEEDS, min_candidates=s15c.MIN_CAND, min_context=MIN_CTX, per_split=results, inputs=inputs,
                   records=dict(step15b=s15.sha(PC15), step15c=s15.sha(PC15C), step16=s15.sha(REC16), pairing=s15.sha(OTREC), data_record=s15.sha(s15.RECORD)),
                   commit=commit, seconds=round(time.time() - T0)),
              open(OUT, 'w'), indent=1, allow_nan=True)
    print('\nwrote %s, %s and %s  (%.0f s)' % (OUT, OUT_CSV, OUT_PC, time.time() - T0))


if __name__ == '__main__':
    main()
