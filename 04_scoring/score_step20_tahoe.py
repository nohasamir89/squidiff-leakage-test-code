#!/usr/bin/env python
"""Step 20 scoring: the pairing-strength sweep (minibatch OT) against the recipe and Step 18's OT pairing.

Pre-registration: PREREG_step20_pairing_sweep.md (sections 4-6, 9 and the dated note in 9), committed before any Step 20
checkpoint existed; this script is committed before any Step 20 evaluation exists. It does only what that document says,
by the Step 15b / 15c / 16 / 18 code paths (imported, not copied).

Two stages (prereg section 5):
  development   python -u score_step20_tahoe.py --splits drug_split_1
                the three settings mb4, mb8, mb32 on drug_split_1: Q1 energy, Q2 DE-overlap, Q3 PDS (setting - recipe),
                the secondaries of 4.3 and the exploratory of 4.4; then the qualification and selection rule of 4.2.
                Writes step20_analysis_development.json whose `selection` block the confirmation launchers read:
                {split, per_setting, qualifying, selected_m, verdict}.
  final         python -u score_step20_tahoe.py
                the selected setting on all three splits (its drug_split_1 runs and the confirmation runs) and the
                section-5 category. Refuses if the committed development record selected no setting.

Before any Step 20 number is printed, on every split (prereg section 2):
  1. the recipe's DE-overlap, DPC-50 and PDS are recomputed from its Step 15b arrays, its energy taken from the Step 15b
     record, its control tracking (and null) recomputed from the Step 18 re-evaluations; ot_pair's DE-overlap, DPC-50,
     energy, PDS and tracking (and null) are recomputed from the Step 18 arrays - every value asserted equal to the Step 18
     per-condition record (tahoe_results/step18_per_condition.csv) to 1e-9;
  2. MRA of the recipe and ot_pair asserted equal to the Step 18 record to 1e-9;
  3. the Step 18 contrasts ot_pair - recipe recomputed: the differences of Q1 energy, Q2 DE-overlap and Q3 PDS, and the
     MRA secondary's difference AND interval (its resamples come from a fresh RandomState(0) in both scripts), asserted
     equal to the Step 18 record to 1e-9. The Q1-Q3 intervals are not compared: Step 18 drew them from the single
     generator score_step15_tahoe creates at import (line 66), so their values depend on how many intervals that run
     computed before them.
Every Step 20 array is hash-checked against its provenance, must have been trained on its pinned minibatch control file,
and must carry the same conditions, the same real cells and THE SAME conditioning control cells as the comparators
(bitwise, per condition): the arms differ only in how their training pairs were matched.

Conventions kept from Step 18 so the comparators reproduce exactly: DE-overlap and DPC-50 from each run's own true and
control means; PDS and MRA of the arms from B - C with C the recipe's Step 15b control means; energy against the stored
real cells; per seed, then averaged per condition; contrasts by score_step16_tahoe.contrast_pc (structure-clustered
percentile bootstrap, 5,000 replicates, 97.5 % per split) with score_step15_tahoe's generator reset to RandomState(0)
before EVERY contrast, so all Step 20 contrasts use the same resamples - paired across settings and identical between
the development and final stages; MRA recomputed per replicate on the resampled structures. The permutation null of control tracking: one RandomState(0) per split for the comparators, advanced in Step
18's order (recipe seeds, then ot_pair seeds); one fresh RandomState(0) per Step 20 arm.

Outputs: tahoe_results/step20_analysis[_development].json, step20_long[_development].csv,
         step20_per_condition[_development].csv. Never overwrites.
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
import score_step15_tahoe as s15                      # noqa: E402  loaders, thesis_per_cond, energy, check_prov, sha
import score_step15c_tahoe as s15c                    # noqa: E402  pds_sets, load_vectors, close, self_check
import score_step16_tahoe as s16                      # noqa: E402  prep_ctx, mra_pre, ctx_split, verdict_of, contrast_pc, fmt
import score_step18_tahoe as s18                      # noqa: E402  load_eval18, tracking, OTREC

PRE = 'PREREG_step20_pairing_sweep.md'
ME = 'score_step20_tahoe.py'
PC15, PC15C, REC15B, REC15C = s16.PC15, s16.PC15C, s16.REC15B, s16.REC15C
REC18 = os.path.join(s15.RES, 'step18_analysis.json')
PC18 = os.path.join(s15.RES, 'step18_per_condition.csv')
OTREC = s18.OTREC
MBREC = os.path.join(s15.RES, 'minibatch_pairing_check.json')
DEVREC = os.path.join(s15.RES, 'step20_analysis_development.json')
DEV = 'drug_split_1'
DEV_MS = (4, 8, 32)                                   # the settings chosen by check_minibatch_pairing_tahoe.py
MIN_CTX, N_BOOT = s16.MIN_CTX, s16.N_BOOT
T0 = time.time()


def pin_path(split, m):
    return os.path.join(s15.RES, 'minibatch_pairing_written_%s_m%d.json' % (split, m))


def load_eval20(split, m, seed, pin_sha):
    """One certified Step 20 evaluation (s20_<split>_mb<m>_seed_<k>): mean vectors, per-cell populations with the
    conditioning control cells, provenance - as score_step18_tahoe.load_eval18, with the minibatch control pin."""
    tag = 's20_%s_mb%d_seed_%d' % (split, m, seed)
    pj = 'provenance/s20ev_%s.json' % tag
    f = os.path.join(s15.RES, 'vectors_%s.npz' % tag)
    fpc = os.path.join(s15.RES, 'vectors_%s_pc.npz' % tag)
    for q in (f, fpc, pj):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    pv = json.load(open(pj))
    s15.check_prov(pv, tag, dict(status='completed', kind='mb', arm='mb%d' % m, m=m, split=split, seed=seed,
                                 drug_encoding='fingerprint', drug_dimension=1024, encoder_arm='mlp', guidance_w=None, nfe=50,
                                 max_cells_per_cond=s15.CAP, ctrl_stored=True))
    if pv.get('control_mb_sha256') != pin_sha:
        sys.exit('FATAL: %s was trained on a control file other than its pinned minibatch file' % tag)
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
               rec=dict(tag=tag, kind='mb', m=m, vectors_sha256=pv['vectors_sha256'], per_cell_sha256=pv['per_cell_sha256'],
                        checkpoint_sha256=pv.get('checkpoint_sha256'), training_commit=pv.get('training_commit'),
                        eval_commit=pv.get('commit'), control_mb_sha256=pv.get('control_mb_sha256')))
    n = len(out['cond'])
    if len(out['pred']) != n or len(out['true_cells']) != n or len(out['ctrl']) != n:
        sys.exit('FATAL: %s per-cell length differs from vectors' % tag)
    return out


def contrast(x_b, x_a, drugs):
    """score_step16_tahoe.contrast_pc with score_step15_tahoe's bootstrap generator reset first, so the interval does
    not depend on how many intervals were drawn before it in this run (see the module docstring)."""
    s15.rng = np.random.RandomState(0)
    return s16.contrast_pc(x_b, x_a, drugs)


def guard_tracked(paths):
    joined = ' '.join(paths)
    if subprocess.run('git ls-files --error-unmatch ' + joined, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers or records not committed')
    if subprocess.run('git diff --quiet -- ' + joined, shell=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers or records uncommitted')


def same_contrast(got, want, label, interval):
    """A recomputed contrast must equal its Step 18 record to 1e-9: the difference always, the interval and verdict
    only when both scripts drew the resamples from a fresh RandomState(0) (interval=True)."""
    ok = abs(got['diff'] - float(want['diff'])) <= 1e-9
    if interval:
        ok = ok and (abs(got['ci'][0] - float(want['ci'][0])) <= 1e-9 and abs(got['ci'][1] - float(want['ci'][1])) <= 1e-9
                     and got['verdict'] == want['verdict'])
    if not ok:
        sys.exit('FATAL: %s does not reproduce the Step 18 record: %s vs %s' % (label, s16.fmt(got), want))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default=','.join(s15.ALL_SPLITS))
    a = ap.parse_args()
    splits = [s.strip() for s in a.splits.split(',') if s.strip()]
    final = sorted(splits) == sorted(s15.ALL_SPLITS)
    if not final and splits != [DEV]:
        sys.exit('FATAL: score the development split alone (--splits %s) or all three splits' % DEV)
    suffix = '' if final else '_development'
    OUT = os.path.join(s15.RES, 'step20_analysis%s.json' % suffix)
    OUT_CSV = os.path.join(s15.RES, 'step20_long%s.csv' % suffix)
    OUT_PC = os.path.join(s15.RES, 'step20_per_condition%s.csv' % suffix)
    for q in (OUT, OUT_CSV, OUT_PC):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 9:
        sys.exit('FATAL: prereg incomplete')

    # --- which arm(s) on which split: the development stage scores all three settings on drug_split_1; the final stage
    #     scores the setting the committed development record selected, on every split
    tracked = [PRE, ME, 'score_step18_tahoe.py', 'score_step16_tahoe.py', 'score_step15_tahoe.py', 'score_step15c_tahoe.py',
               'score_step15d_utility.py', 'score_step15e_pooled.py', 'step14_metrics.py', 'Squidiff/scrna_datasets.py',
               PC15, PC15C, REC15B, REC15C, REC18, PC18, OTREC, MBREC, s15.RECORD]
    if final:
        tracked.append(DEVREC)
        guard_tracked([DEVREC])
        dev = json.load(open(DEVREC))
        sel = dev.get('selection', {})
        if dev.get('final') is not False or sel.get('split') != DEV:
            sys.exit('FATAL: %s is not the development record' % DEVREC)
        if sel.get('selected_m') is None:
            sys.exit('FATAL: the development stage selected no setting (%s); prereg section 5 ends Step 20 there' % sel.get('verdict'))
        arms_of = {s: [int(sel['selected_m'])] for s in splits}
    else:
        dev = None
        arms_of = {DEV: list(DEV_MS)}
    for s in splits:
        for m in arms_of[s]:
            tracked.append(pin_path(s, m))
    guard_tracked(tracked)
    record = json.load(open(s15.RECORD))
    otrec, mbrec, rec18 = json.load(open(OTREC)), json.load(open(MBREC)), json.load(open(REC18))
    if not rec18.get('final'):
        sys.exit('FATAL: the Step 18 record is not the final one')
    if [int(x) for x in mbrec['selection']['chosen']] != list(DEV_MS):
        sys.exit('FATAL: the pairing record chose %s, this scorer expects %s' % (mbrec['selection']['chosen'], list(DEV_MS)))
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    print('[prereg] committed; scorer committed; commit %s; splits %s (%s); arms %s'
          % (commit[:7], splits, 'final' if final else 'development', arms_of))
    s15c.self_check()
    s16.mra_self_check()
    pc15, pc15c = pd.read_csv(PC15), pd.read_csv(PC15C)
    pc18all = pd.read_csv(PC18, float_precision='round_trip')

    results, inputs, rows, pc_rows = {}, {}, [], []
    for split in splits:
        print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
        k = split.split('_')[-1]
        rec_split = record['splits'][k]
        n_drugs = len(rec_split['held_structures'])
        ot_sha = otrec['written'][split]['sha256']

        # ---------------- the comparators' inputs (Step 15b recipe vectors and real cells), as Step 18 ----------------
        old = [s15c.load_vectors(split, 'recipe', s) for s in range(s15.N_SEEDS)]
        ref = old[0]
        cond, n = ref['cond'], len(ref['cond'])
        if n != int(rec_split['n_test_conditions']):
            sys.exit('FATAL: %d conditions, record expects %d' % (n, rec_split['n_test_conditions']))
        for v in old:
            if not np.array_equal(v['cond'], cond) or np.abs(v['C'] - ref['C']).max() > 1e-5 or np.abs(v['true'] - ref['true']).max() > 1e-5:
                sys.exit('FATAL: condition order, control means or true means differ in %s' % v['tag'])
        ref_pc = s15.load_eval(split, 'recipe', 0)
        if not np.array_equal(ref_pc['cond'], cond):
            sys.exit('FATAL: the recipe per-cell file lists different conditions')
        real = ref_pc['true_cells']
        del ref_pc
        true_mean, C = ref['true'], ref['C']
        Y = true_mean - C

        pc = pc15[pc15['split'] == split].reset_index(drop=True)
        pcc = pc15c[pc15c['split'] == split].reset_index(drop=True)
        p18 = pc18all[pc18all['split'] == split].reset_index(drop=True)
        for df, nm in ((pc, 'Step 15b'), (pcc, 'Step 15c'), (p18, 'Step 18')):
            if len(df) != n or not np.array_equal(df['condition'].astype(str).values, cond):
                sys.exit('FATAL: the %s per-condition record does not list the conditions of %s in the vectors order' % (nm, split))
        lines, plates, drugs, doses = (pc['line'].astype(str).values, pc['plate'].astype(str).values,
                                       pc['structure'].astype(str).values, pc['dose_nM'].astype(float).values)
        if len(np.unique(drugs)) != n_drugs:
            sys.exit('FATAL: %d structures in the record, expected %d' % (len(np.unique(drugs)), n_drugs))
        ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
        ctx_u = np.unique(ctx_key)
        contexts = [np.where(ctx_key == u)[0] for u in ctx_u]
        ctx_id = np.array([int(np.where(ctx_u == ctx_key[i])[0][0]) for i in range(n)])
        cand = {}
        for members in contexts:
            for p_ in members:
                cand[p_] = np.array([q for q in members if q == p_ or drugs[q] != drugs[p_]])
        n_cand = np.array([len(cand[p_]) for p_ in range(n)])
        if not np.array_equal(n_cand, pcc['n_candidates'].astype(int).values):
            sys.exit('FATAL: the candidate sets differ from the Step 15c record')
        print('[design] %d conditions | %d held-out structures | %d contexts | cells per condition up to %d'
              % (n, n_drugs, len(contexts), s15.CAP))

        M = {}            # per-condition metrics per submission: jac, dpc, en, pds, trk, trk_null; norms per seed
        norms = {}

        # ---------------- the recipe: recomputed and asserted against the Step 18 per-condition record ----------------
        jj, dd = zip(*[s15.thesis_per_cond(v['true'], v['B'], v['C']) for v in old])
        M['recipe'] = dict(jac=np.nanmean(np.vstack(jj), axis=0), dpc=np.nanmean(np.vstack(dd), axis=0),
                           en=pc['recipe_energy'].astype(float).values,
                           pds=np.nanmean(np.vstack([s15c.pds_sets(v['B'] - v['C'], Y, contexts, cand) for v in old]), axis=0))
        norms['recipe'] = [np.linalg.norm(v['B'] - v['C'], axis=1) for v in old]
        rng_null = np.random.RandomState(0)                    # one per split, advanced in Step 18's order
        tr_, tn_ = [], []
        ctrl_ref = None
        inp = {'recipe': [v['tag'] for v in old]}
        for s in range(s15.N_SEEDS):
            v = s18.load_eval18(split, s, 'recipe', ot_sha)
            if not np.array_equal(v['cond'], cond):
                sys.exit('FATAL: condition order differs in %s' % v['tag'])
            for i in range(n):
                if not np.array_equal(v['true_cells'][i], real[i]):
                    sys.exit('FATAL: stored real cells differ in %s, condition %d' % (v['tag'], i))
            if ctrl_ref is None:
                ctrl_ref = v['ctrl']                            # the conditioning control cells every arm must share
            else:
                for i in range(n):
                    if not np.array_equal(v['ctrl'][i], ctrl_ref[i]):
                        sys.exit('FATAL: conditioning control cells differ across the recipe re-evaluations at condition %d' % i)
            tr_.append(s18.tracking(v['pred'], v['ctrl']))
            tn_.append(s18.tracking(v['pred'], v['ctrl'], rng_null))
            inp['recipe'].append(v['tag'])
            del v
        M['recipe']['trk'] = np.nanmean(np.vstack(tr_), axis=0)
        M['recipe']['trk_null'] = np.nanmean(np.vstack(tn_), axis=0)

        # ---------------- an arm evaluated under the Step 18 harness (ot_pair, or a Step 20 setting) ----------------
        def score_arm(loader, label, rng):
            per = dict(jac=[], dpc=[], en=[], pds=[], trk=[], trk_null=[])
            nrm, tags = [], []
            for s in range(s15.N_SEEDS):
                v = loader(s)
                if not np.array_equal(v['cond'], cond):
                    sys.exit('FATAL: condition order differs in %s' % v['tag'])
                if np.abs(v['C'] - C).max() > 1e-5 or np.abs(v['true'] - true_mean).max() > 1e-5:
                    sys.exit('FATAL: control or true means differ in %s' % v['tag'])
                for i in range(n):
                    if not np.array_equal(v['true_cells'][i], real[i]):
                        sys.exit('FATAL: stored real cells differ in %s, condition %d' % (v['tag'], i))
                    if not np.array_equal(v['ctrl'][i], ctrl_ref[i]):
                        sys.exit('FATAL: %s was conditioned on different control cells than the comparators at condition %d'
                                 % (v['tag'], i))
                j, d = s15.thesis_per_cond(v['true'], v['B'], v['C'])
                per['jac'].append(j); per['dpc'].append(d)
                per['en'].append(np.array([s15.energy(v['pred'][i], real[i]) for i in range(n)]))
                per['pds'].append(s15c.pds_sets(v['B'] - C, Y, contexts, cand))
                per['trk'].append(s18.tracking(v['pred'], v['ctrl']))
                per['trk_null'].append(s18.tracking(v['pred'], v['ctrl'], rng))
                nrm.append(np.linalg.norm(v['B'] - C, axis=1))
                tags.append(v['tag'])
                print('[arm] %-8s seed %d  energy %.4f  DE-overlap %.4f  PDS %.4f  DPC-50 %.4f  tracking %+.4f (null %+.4f)'
                      % (label, s, per['en'][-1].mean(), np.nanmean(j), np.nanmean(per['pds'][-1]), np.nanmean(d),
                         np.nanmean(per['trk'][-1]), np.nanmean(per['trk_null'][-1])))
                del v
            out = {key: np.nanmean(np.vstack(vals), axis=0) for key, vals in per.items()}
            out['per_seed'] = {key: [float(np.nanmean(x)) for x in vals] for key, vals in per.items()}
            return out, nrm, tags

        M['ot_pair'], norms['ot_pair'], inp['ot_pair'] = score_arm(lambda s: s18.load_eval18(split, s, 'ot', ot_sha), 'ot_pair', rng_null)

        # ---------------- reproduction 1: the comparators equal the Step 18 per-condition record ----------------
        for nm in ('recipe', 'ot_pair'):
            for key, col in (('jac', 'de_overlap'), ('dpc', 'dpc50'), ('en', 'energy'), ('pds', 'pds'),
                             ('trk', 'tracking'), ('trk_null', 'tracking_null')):
                if not s15c.close(M[nm][key], p18['%s_%s' % (nm, col)].astype(float).values):
                    sys.exit('FATAL: %s %s does not reproduce the Step 18 per-condition record' % (nm, col))
        print('[reproduction] recipe and ot_pair: DE-overlap, DPC-50, energy, PDS, tracking and its null equal the Step 18 '
              'per-condition record to 1e-9')

        # ---------------- MRA, as Step 18 (no replicate ceiling here) ----------------
        true_norm = np.linalg.norm(Y, axis=1)
        n_ctx = len(contexts)
        full = np.arange(n)
        pre_full = s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, n_ctx, full))

        # ---------------- the Step 20 settings on this split ----------------
        arms = []
        f_cos = {}
        for m in arms_of[split]:
            pin = json.load(open(pin_path(split, m)))
            if int(pin['m']) != m or pin['split'] != split:
                sys.exit('FATAL: %s pins %s m = %s' % (pin_path(split, m), pin['split'], pin['m']))
            label = 'mb%d' % m
            M[label], norms[label], inp[label] = score_arm(lambda s, m=m, sha=pin['sha256']: load_eval20(split, m, s, sha),
                                                           label, np.random.RandomState(0))
            f_cos[label] = float(mbrec['per_split'][split]['f_cos'][str(m)])
            arms.append(label)
        print('[shared] every Step 20 run carries the comparators\' conditions, real cells and conditioning control cells (bitwise)')
        del real, ctrl_ref

        mra_seed = {nm: [s16.mra_pre(x, pre_full) for x in norms[nm]] for nm in norms}
        mra = {nm: float(np.mean(v_)) for nm, v_ in mra_seed.items()}
        for nm in ('recipe', 'ot_pair'):
            want = float(rec18['per_split'][split]['metrics'][nm]['mra'])
            if abs(mra[nm] - want) > 1e-9:
                sys.exit('FATAL: MRA of %s (%.9f) does not reproduce the Step 18 record (%.9f)' % (nm, mra[nm], want))
        print('[reproduction] MRA of the recipe and ot_pair equals the Step 18 record to 1e-9')

        # ---------------- reproduction 2: the Step 18 contrasts, difference and interval ----------------
        c18 = rec18['per_split'][split]['contrasts']
        same_contrast(contrast(M['ot_pair']['en'], M['recipe']['en'], drugs), c18['Q1_primary_energy_ot_pair_minus_recipe'], 'Q1 energy', False)
        same_contrast(contrast(M['ot_pair']['jac'], M['recipe']['jac'], drugs), c18['Q2_gate_de_ot_pair_minus_recipe'], 'Q2 DE-overlap', False)
        same_contrast(contrast(M['ot_pair']['pds'], M['recipe']['pds'], drugs), c18['Q3_gate_pds_ot_pair_minus_recipe'], 'Q3 PDS', False)

        rng = np.random.RandomState(0)                          # MRA: the Step 18 resamples, replicate for replicate
        gs = [np.where(drugs == u)[0] for u in np.unique(drugs)]
        BM = {nm: np.full(N_BOOT, np.nan) for nm in norms}
        for b in range(N_BOOT):
            sel = np.concatenate([gs[g] for g in rng.randint(0, len(gs), len(gs))])
            pre = s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, n_ctx, sel))
            for nm in norms:
                BM[nm][b] = float(np.mean([s16.mra_pre(x, pre) for x in norms[nm]]))
        same_contrast(s16.verdict_of(mra['ot_pair'] - mra['recipe'], BM['ot_pair'] - BM['recipe']),
                      c18['secondary_mra_ot_pair_minus_recipe'], 'MRA secondary', True)
        print('[reproduction] the Step 18 contrasts ot_pair - recipe equal the Step 18 record to 1e-9: energy, DE-overlap and '
              'PDS differences; MRA difference and interval (same resamples)')

        # ---------------- absolute values ----------------
        metrics = {}
        for nm in ['recipe', 'ot_pair'] + arms:
            m_ = dict(energy=float(np.nanmean(M[nm]['en'])), de_overlap=float(np.nanmean(M[nm]['jac'])),
                      pds_mean=float(np.nanmean(M[nm]['pds'])), dpc_mean=float(np.nanmean(M[nm]['dpc'])), mra=mra[nm],
                      tracking=float(np.nanmean(M[nm]['trk'])), tracking_null=float(np.nanmean(M[nm]['trk_null'])),
                      mra_per_seed=mra_seed[nm])
            if 'per_seed' in M[nm]:
                m_['per_seed'] = M[nm]['per_seed']
            if nm in f_cos:
                m_['f_cos'] = f_cos[nm]
            metrics[nm] = m_
            for key_ in ('energy', 'de_overlap', 'pds_mean', 'dpc_mean', 'mra', 'tracking'):
                rows.append(dict(split=split, arm=nm, level='metric', key=key_, value=m_[key_], ci_lo='', ci_hi='', verdict=''))
            print('[abs] %-8s energy %.4f | DE-overlap %.4f | PDS %.4f | DPC-50 %.4f | MRA %+.4f | tracking %+.4f (null %+.4f)%s'
                  % (nm, m_['energy'], m_['de_overlap'], m_['pds_mean'], m_['dpc_mean'], m_['mra'], m_['tracking'],
                     m_['tracking_null'], ('  | f_cos %.3f' % f_cos[nm]) if nm in f_cos else ''))

        # ---------------- contrasts per setting (sections 4.1, 4.3, 4.4) ----------------
        con = {}
        for nm in arms:
            f = f_cos[nm]
            c = {}
            c['Q1_energy_minus_recipe'] = contrast(M[nm]['en'], M['recipe']['en'], drugs)
            c['Q2_de_minus_recipe'] = contrast(M[nm]['jac'], M['recipe']['jac'], drugs)
            c['Q3_pds_minus_recipe'] = contrast(M[nm]['pds'], M['recipe']['pds'], drugs)
            c['secondary_energy_minus_ot_pair'] = contrast(M[nm]['en'], M['ot_pair']['en'], drugs)
            c['secondary_de_minus_ot_pair'] = contrast(M[nm]['jac'], M['ot_pair']['jac'], drugs)
            c['secondary_pds_minus_ot_pair'] = contrast(M[nm]['pds'], M['ot_pair']['pds'], drugs)
            c['secondary_dpc_minus_recipe'] = contrast(M[nm]['dpc'], M['recipe']['dpc'], drugs)
            c['secondary_mra_minus_recipe'] = s16.verdict_of(mra[nm] - mra['recipe'], BM[nm] - BM['recipe'])
            c['secondary_tracking_minus_recipe'] = contrast(M[nm]['trk'], M['recipe']['trk'], drugs)
            c['secondary_tracking_minus_null'] = contrast(M[nm]['trk'], M[nm]['trk_null'], drugs)
            for key, metric in (('energy', 'en'), ('de', 'jac'), ('pds', 'pds')):
                line = M['recipe'][metric] + f * (M['ot_pair'][metric] - M['recipe'][metric])
                c['exploratory_excess_%s_over_line' % key] = contrast(M[nm][metric], line, drugs)
            con[nm] = c
            print('\n%s (m = %s, f_cos %.3f)' % (nm, nm[2:], f))
            for key, cc in c.items():
                print('  %-38s %s' % (key, s16.fmt(cc)))
                rows.append(dict(split=split, arm=nm, level='contrast', key=key, value=cc['diff'], ci_lo=cc['ci'][0],
                                 ci_hi=cc['ci'][1], verdict=cc['verdict']))
        # the comparators' own Step 18 contrasts, carried for the curve (reproduced above)
        con['ot_pair'] = {'Q1_energy_minus_recipe': contrast(M['ot_pair']['en'], M['recipe']['en'], drugs),
                          'Q2_de_minus_recipe': contrast(M['ot_pair']['jac'], M['recipe']['jac'], drugs),
                          'Q3_pds_minus_recipe': contrast(M['ot_pair']['pds'], M['recipe']['pds'], drugs),
                          'secondary_tracking_minus_recipe': contrast(M['ot_pair']['trk'], M['recipe']['trk'], drugs)}
        for key, cc in con['ot_pair'].items():
            rows.append(dict(split=split, arm='ot_pair', level='contrast', key=key, value=cc['diff'], ci_lo=cc['ci'][0],
                             ci_hi=cc['ci'][1], verdict=cc['verdict']))
        print('[bootstrap] %d replicates (%d s)' % (N_BOOT, time.time() - T0))

        for i in range(n):
            r = dict(split=split, condition=cond[i], line=lines[i], plate=plates[i], structure=drugs[i], dose_nM=doses[i])
            for nm in ['recipe', 'ot_pair'] + arms:
                for key, col in (('en', 'energy'), ('jac', 'de_overlap'), ('pds', 'pds'), ('dpc', 'dpc50'), ('trk', 'tracking')):
                    r['%s_%s' % (nm, col)] = M[nm][key][i]
            pc_rows.append(r)
        results[split] = dict(n_conditions=n, n_structures=n_drugs, n_contexts=n_ctx, min_context=MIN_CTX, arms=arms,
                              f_cos=f_cos, metrics=metrics, contrasts=con)
        inputs[split] = inp
        del old, M, norms, BM

    # ------------------------------------------------------------------ verdicts (section 5)
    if not final:
        per = {}
        for nm in results[DEV]['arms']:
            c = results[DEV]['contrasts'][nm]
            q1, q2, q3 = (c['Q1_energy_minus_recipe']['verdict'], c['Q2_de_minus_recipe']['verdict'], c['Q3_pds_minus_recipe']['verdict'])
            per[nm] = dict(m=int(nm[2:]), Q1=q1, Q2=q2, Q3=q3, qualifies=bool(q1 == 'below zero' and q2 != 'below zero' and q3 != 'below zero'))
        qualifying = sorted(v['m'] for v in per.values() if v['qualifies'])
        selected = max(qualifying) if qualifying else None
        verdict = ('SELECTED m = %d' % selected) if selected is not None else '0 NO QUALIFYING SETTING'
        selection = dict(split=DEV, per_setting=per, qualifying=qualifying, selected_m=selected, verdict=verdict)
        print('\n' + '=' * 72 + '\nPRE-REGISTERED DEVELOPMENT VERDICT (sections 4.2 and 5): %s' % verdict)
        for nm, v in per.items():
            print('  %-6s Q1 energy %-13s | Q2 DE-overlap %-13s | Q3 PDS %-13s | qualifies: %s' % (nm, v['Q1'], v['Q2'], v['Q3'], v['qualifies']))
        print('  Section 7 expected: 0 NO QUALIFYING SETTING.\n' + '=' * 72)
    else:
        m_sel = int(dev['selection']['selected_m'])
        nm = 'mb%d' % m_sel
        for key in ('Q1_energy_minus_recipe', 'Q2_de_minus_recipe', 'Q3_pds_minus_recipe'):   # split 1 equals the development record
            got, want = results[DEV]['contrasts'][nm][key], dev['per_split'][DEV]['contrasts'][nm][key]
            if abs(got['diff'] - want['diff']) > 1e-9 or abs(got['ci'][0] - want['ci'][0]) > 1e-9 or abs(got['ci'][1] - want['ci'][1]) > 1e-9:
                sys.exit('FATAL: %s %s on %s differs from the development record' % (nm, key, DEV))
        print('[reproduction] %s on %s equals the development record (Q1-Q3, difference and interval) to 1e-9' % (nm, DEV))
        q1 = {s: results[s]['contrasts'][nm]['Q1_energy_minus_recipe']['verdict'] for s in s15.ALL_SPLITS}
        q2 = {s: results[s]['contrasts'][nm]['Q2_de_minus_recipe']['verdict'] for s in s15.ALL_SPLITS}
        q3 = {s: results[s]['contrasts'][nm]['Q3_pds_minus_recipe']['verdict'] for s in s15.ALL_SPLITS}
        v1 = list(q1.values()); gates = list(q2.values()) + list(q3.values())
        if all(v == 'below zero' for v in v1) and all(v != 'below zero' for v in gates):
            verdict = '1 FIDELITY GAIN'
        elif all(v == 'below zero' for v in v1):
            verdict = '2 FIDELITY GAIN AT A COST'
        elif all(v == 'includes zero' for v in v1):
            verdict = '5 UNRESOLVED'
        elif 'above zero' in v1:
            verdict = '3 REVERSED'
        else:
            verdict = '4 PARTIAL'
        selection = dict(dev['selection'], confirmed_m=m_sel)
        print('\n' + '=' * 72 + '\nPRE-REGISTERED FINAL VERDICT (section 5) for %s: %s' % (nm, verdict))
        print('  Q1 energy per split: %s\n  Q2 DE-overlap per split: %s\n  Q3 PDS per split: %s' % (q1, q2, q3))
        for s in s15.ALL_SPLITS:
            c = results[s]['contrasts'][nm]
            print('  %s: Q2 lower bound %+.5f, Q3 lower bound %+.5f' % (s, c['Q2_de_minus_recipe']['ci'][0], c['Q3_pds_minus_recipe']['ci'][0]))
        print('=' * 72)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'arm', 'level', 'key', 'value', 'ci_lo', 'ci_hi', 'verdict'])
        w.writeheader(); w.writerows(rows)
    with open(OUT_PC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0].keys()))
        w.writeheader(); w.writerows(pc_rows)
    json.dump(dict(prereg=PRE, final=final, verdict=verdict, selection=selection, splits=splits, arms_of=arms_of,
                   interval='97.5% percentile per split, structure-clustered', n_boot=N_BOOT, n_seeds=s15.N_SEEDS,
                   per_split=results, inputs=inputs,
                   records=dict(step15b=s15.sha(PC15), step15c=s15.sha(PC15C), step18=s15.sha(REC18), step18_per_condition=s15.sha(PC18),
                                ot_pairing=s15.sha(OTREC), minibatch_pairing=s15.sha(MBREC), data_record=s15.sha(s15.RECORD),
                                development=s15.sha(DEVREC) if final else None),
                   commit=commit, seconds=round(time.time() - T0)),
              open(OUT, 'w'), indent=1, allow_nan=True)
    print('\nwrote %s, %s and %s  (%.0f s)' % (OUT, OUT_CSV, OUT_PC, time.time() - T0))


if __name__ == '__main__':
    main()
