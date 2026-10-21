#!/usr/bin/env python
"""Step 21 scoring: minibatch OT at m = 32 against full OT on the confirmation splits.

Pre-registration: PREREG_step21_mb32_confirmation.md (sections 4-6 and 9), committed before any Step 21 checkpoint
existed; this script is committed before any Step 21 evaluation exists. It does only what that document says, through
the Step 15b / 15c / 16 / 18 code paths (imported, not copied) and the Step 20 conventions.

    python -u score_step21_tahoe.py          (no options: both confirmation splits, drug_split_0 then drug_split_2)

Two phases (prereg section 6: every reproduction, on BOTH splits, finishes before the first Step 21 number is printed,
so the print-order defect of Step 20 is not repeated). Phase A, on each confirmation split:
  1. the comparators: the recipe's DE-overlap, DPC-50 and PDS recomputed from its Step 15b arrays, its energy taken
     from the Step 15b record, its control tracking (and null) recomputed from the Step 18 re-evaluations; full OT's
     (ot_pair) DE-overlap, DPC-50, energy, PDS and tracking (and null) recomputed from the Step 18 arrays - every value
     asserted equal to the Step 18 per-condition record (tahoe_results/step18_per_condition.csv) to 1e-9;
  2. MRA of the recipe and full OT asserted equal to the Step 18 record to 1e-9;
  3. the Step 18 contrasts full OT - recipe recomputed: the differences of Q1 energy, Q2 DE-overlap and Q3 PDS, and the
     MRA secondary's difference AND interval (fresh RandomState(0) resamples in both scripts), asserted equal to the
     Step 18 record to 1e-9. (The Q1-Q3 intervals are not compared: Step 18 drew them from one generator per run.)
  4. the margin delta from the committed Step 18 record.
Phase B, only after phase A has passed on both splits: the six Step 21 arrays are loaded and scored. Each is
hash-checked against its provenance, must have been trained on its pinned m = 32 control file, and must carry the
comparators' conditions, real cells and conditioning control cells (bitwise, per condition). The MRA bootstrap is
re-run with the same fresh RandomState(0) and must reproduce phase A's recipe replicates exactly.

Endpoints (prereg section 4), per split, 97.5 % structure-clustered percentile bootstrap intervals (5,000 replicates)
with score_step15_tahoe's generator reset to RandomState(0) before EVERY contrast (as Step 20):
  Q1 (gated, non-inferiority)  D_E = energy(mb32) - energy(full OT); holds if the interval's upper bound is below
                               delta = 0.25 * M1, with M1 = |upper bound of Step 18's Q1 interval on that split|, read
                               from tahoe_results/step18_analysis.json.
  Q2 (gated, superiority)      D_P = PDS(mb32) - PDS(full OT); holds if the interval is above zero.
  secondary                    DE-overlap mb32 - full OT; mb32 - recipe on energy, DE-overlap, PDS, DPC-50, MRA,
                               tracking; tracking mb32 - its permutation null.
  exploratory                  excess over the straight line recipe -> full OT at f(32), for energy, DE-overlap, PDS.
  reference                    full OT - recipe on energy, DE-overlap, PDS and tracking on the same (fresh) resamples.
Verdict (prereg section 5): E = Q1 holds on both splits, P = Q2 holds on both; 1 CONFIRMED (E and P), 2 LESS PDS LOSS,
REALISM NOT SHOWN (P only), 3 REALISM KEPT, NO PDS ADVANTAGE SHOWN (E only), 4 NOT CONFIRMED (neither); the clause
"with a reversal on <split>" if D_P's interval is below zero or D_E's lower bound is above delta on that split.
The development split (Step 20) is reported descriptively from the committed Step 20 record, under the same rule.

Outputs: tahoe_results/step21_analysis.json, step21_long.csv, step21_per_condition.csv. Never overwrites.
"""
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

PRE = 'PREREG_step21_mb32_confirmation.md'
ME = 'score_step21_tahoe.py'
PC15, PC15C, REC15B, REC15C = s16.PC15, s16.PC15C, s16.REC15B, s16.REC15C
REC18 = os.path.join(s15.RES, 'step18_analysis.json')
PC18 = os.path.join(s15.RES, 'step18_per_condition.csv')
OTREC = s18.OTREC
MBREC = os.path.join(s15.RES, 'minibatch_pairing_check.json')
REC20 = os.path.join(s15.RES, 'step20_analysis_development.json')
SPLITS = ('drug_split_0', 'drug_split_2')             # the confirmation splits, in this order
DEV = 'drug_split_1'
M = 32
ARM = 'mb32'
MARGIN_FRACTION = 0.25                                # prereg 4.1: keep at least 75 % of full OT's cautious energy gain
Q1_18 = 'Q1_primary_energy_ot_pair_minus_recipe'      # the Step 18 key that defines M1
MIN_CTX, N_BOOT = s16.MIN_CTX, s16.N_BOOT
OUT = os.path.join(s15.RES, 'step21_analysis.json')
OUT_CSV = os.path.join(s15.RES, 'step21_long.csv')
OUT_PC = os.path.join(s15.RES, 'step21_per_condition.csv')
T0 = time.time()


def pin_path(split):
    return os.path.join(s15.RES, 'minibatch_pairing_written_%s_m%d.json' % (split, M))


def load_eval21(split, seed, pin_sha):
    """One certified Step 21 evaluation (s21_<split>_mb32_seed_<k>): mean vectors, per-cell populations with the
    conditioning control cells, provenance - as score_step20_tahoe.load_eval20, with the Step 21 names."""
    tag = 's21_%s_%s_seed_%d' % (split, ARM, seed)
    pj = 'provenance/s21ev_%s.json' % tag
    f = os.path.join(s15.RES, 'vectors_%s.npz' % tag)
    fpc = os.path.join(s15.RES, 'vectors_%s_pc.npz' % tag)
    for q in (f, fpc, pj):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    pv = json.load(open(pj))
    s15.check_prov(pv, tag, dict(status='completed', kind='mb', step=21, arm=ARM, m=M, split=split, seed=seed,
                                 drug_encoding='fingerprint', drug_dimension=1024, encoder_arm='mlp', guidance_w=None, nfe=50,
                                 max_cells_per_cond=s15.CAP, ctrl_stored=True, prereg=PRE))
    if pv.get('control_mb_sha256') != pin_sha:
        sys.exit('FATAL: %s was trained on a control file other than its pinned m = 32 file' % tag)
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
               rec=dict(tag=tag, kind='mb', m=M, vectors_sha256=pv['vectors_sha256'], per_cell_sha256=pv['per_cell_sha256'],
                        checkpoint_sha256=pv.get('checkpoint_sha256'), training_commit=pv.get('training_commit'),
                        eval_commit=pv.get('commit'), control_mb_sha256=pv.get('control_mb_sha256')))
    n = len(out['cond'])
    if len(out['pred']) != n or len(out['true_cells']) != n or len(out['ctrl']) != n:
        sys.exit('FATAL: %s per-cell length differs from vectors' % tag)
    return out


def contrast(x_b, x_a, drugs):
    """score_step16_tahoe.contrast_pc with score_step15_tahoe's bootstrap generator reset first, so every contrast uses
    the same resamples, whatever was drawn before it (Step 20 convention; Step 18 findings section 9 item 12)."""
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


def mra_boot(norms_sub, true_norm, ctx_id, n_ctx, drugs):
    """MRA per bootstrap replicate for the given submissions: a fresh RandomState(0) drawing structures with replacement,
    exactly as score_step18_tahoe and score_step20_tahoe, so two calls give identical resamples."""
    rng = np.random.RandomState(0)
    gs = [np.where(drugs == u)[0] for u in np.unique(drugs)]
    BM = {nm: np.full(N_BOOT, np.nan) for nm in norms_sub}
    for b in range(N_BOOT):
        sel = np.concatenate([gs[g] for g in rng.randint(0, len(gs), len(gs))])
        pre = s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, n_ctx, sel))
        for nm in norms_sub:
            BM[nm][b] = float(np.mean([s16.mra_pre(x, pre) for x in norms_sub[nm]]))
    return BM


def ni_test(c, delta):
    """Prereg 4.1 Q1: non-inferior if the upper bound is below delta; a reversal if the lower bound is above delta."""
    if c['ci'][1] < delta:
        return 'non-inferior'
    if c['ci'][0] > delta:
        return 'reversal (lower bound above the margin)'
    return 'not shown'


def score_arm(st, loader, label, rng):
    """One arm evaluated under the Step 18 harness (full OT, or the Step 21 arm), scored exactly as in Step 20; the arm
    must carry the comparators' conditions, real cells and conditioning control cells (bitwise, per condition)."""
    cond, n, real, ctrl_ref, C, true_mean = st['cond'], st['n'], st['real'], st['ctrl_ref'], st['C'], st['true_mean']
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
                sys.exit('FATAL: %s was conditioned on different control cells than the comparators at condition %d' % (v['tag'], i))
        j, d = s15.thesis_per_cond(v['true'], v['B'], v['C'])
        per['jac'].append(j); per['dpc'].append(d)
        per['en'].append(np.array([s15.energy(v['pred'][i], real[i]) for i in range(n)]))
        per['pds'].append(s15c.pds_sets(v['B'] - C, st['Y'], st['contexts'], st['cand']))
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


def comparators(split, record, otrec, rec18, pc15, pc15c, pc18all):
    """Phase A for one split: build the design and the comparators (recipe, full OT) exactly as Steps 18 and 20, and
    assert them against the Step 18 records (per-condition values, MRA, contrast differences, MRA interval). Prints
    comparator numbers only. Returns everything phase B needs."""
    print('\n' + '#' * 72 + '\n# %s: comparators\n' % split + '#' * 72)
    k = split.split('_')[-1]
    rec_split = record['splits'][k]
    n_drugs = len(rec_split['held_structures'])
    ot_sha = otrec['written'][split]['sha256']
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

    Mt, norms = {}, {}
    # ---- the recipe: recomputed (energy from the Step 15b record), as Steps 18 and 20
    jj, dd = zip(*[s15.thesis_per_cond(v['true'], v['B'], v['C']) for v in old])
    Mt['recipe'] = dict(jac=np.nanmean(np.vstack(jj), axis=0), dpc=np.nanmean(np.vstack(dd), axis=0),
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
    Mt['recipe']['trk'] = np.nanmean(np.vstack(tr_), axis=0)
    Mt['recipe']['trk_null'] = np.nanmean(np.vstack(tn_), axis=0)
    st = dict(split=split, cond=cond, n=n, n_drugs=n_drugs, real=real, ctrl_ref=ctrl_ref, C=C, true_mean=true_mean, Y=Y,
              contexts=contexts, cand=cand, ctx_id=ctx_id, n_ctx=len(contexts), lines=lines, plates=plates, drugs=drugs,
              doses=doses, Mt=Mt, norms=norms, inp=inp)
    # ---- full OT, under the Step 18 harness; its null generator continues the recipe's (Step 18's order)
    Mt['ot_pair'], norms['ot_pair'], inp['ot_pair'] = score_arm(st, lambda s: s18.load_eval18(split, s, 'ot', ot_sha), 'ot_pair', rng_null)
    del old

    # ---- reproduction 1: the comparators equal the Step 18 per-condition record
    for nm in ('recipe', 'ot_pair'):
        for key, col in (('jac', 'de_overlap'), ('dpc', 'dpc50'), ('en', 'energy'), ('pds', 'pds'),
                         ('trk', 'tracking'), ('trk_null', 'tracking_null')):
            if not s15c.close(Mt[nm][key], p18['%s_%s' % (nm, col)].astype(float).values):
                sys.exit('FATAL: %s %s does not reproduce the Step 18 per-condition record' % (nm, col))
    print('[reproduction] recipe and full OT: DE-overlap, DPC-50, energy, PDS, tracking and its null equal the Step 18 '
          'per-condition record to 1e-9')

    # ---- reproduction 2: MRA of the comparators
    true_norm = np.linalg.norm(Y, axis=1)
    pre_full = s16.prep_ctx(true_norm, s16.ctx_split(ctx_id, st['n_ctx'], np.arange(n)))
    mra_seed = {nm: [s16.mra_pre(x, pre_full) for x in norms[nm]] for nm in ('recipe', 'ot_pair')}
    mra = {nm: float(np.mean(v_)) for nm, v_ in mra_seed.items()}
    for nm in ('recipe', 'ot_pair'):
        want = float(rec18['per_split'][split]['metrics'][nm]['mra'])
        if abs(mra[nm] - want) > 1e-9:
            sys.exit('FATAL: MRA of %s (%.9f) does not reproduce the Step 18 record (%.9f)' % (nm, mra[nm], want))
    print('[reproduction] MRA of the recipe and full OT equals the Step 18 record to 1e-9')

    # ---- reproduction 3: the Step 18 contrasts (differences; the MRA interval on the same resamples)
    c18 = rec18['per_split'][split]['contrasts']
    ref_c = {'reference_energy_ot_pair_minus_recipe': contrast(Mt['ot_pair']['en'], Mt['recipe']['en'], drugs),
             'reference_de_ot_pair_minus_recipe': contrast(Mt['ot_pair']['jac'], Mt['recipe']['jac'], drugs),
             'reference_pds_ot_pair_minus_recipe': contrast(Mt['ot_pair']['pds'], Mt['recipe']['pds'], drugs),
             'reference_tracking_ot_pair_minus_recipe': contrast(Mt['ot_pair']['trk'], Mt['recipe']['trk'], drugs)}
    same_contrast(ref_c['reference_energy_ot_pair_minus_recipe'], c18[Q1_18], 'Q1 energy', False)
    same_contrast(ref_c['reference_de_ot_pair_minus_recipe'], c18['Q2_gate_de_ot_pair_minus_recipe'], 'Q2 DE-overlap', False)
    same_contrast(ref_c['reference_pds_ot_pair_minus_recipe'], c18['Q3_gate_pds_ot_pair_minus_recipe'], 'Q3 PDS', False)
    BM1 = mra_boot({nm: norms[nm] for nm in ('recipe', 'ot_pair')}, true_norm, ctx_id, st['n_ctx'], drugs)
    same_contrast(s16.verdict_of(mra['ot_pair'] - mra['recipe'], BM1['ot_pair'] - BM1['recipe']),
                  c18['secondary_mra_ot_pair_minus_recipe'], 'MRA secondary', True)
    print('[reproduction] the Step 18 contrasts full OT - recipe equal the Step 18 record to 1e-9: energy, DE-overlap and '
          'PDS differences; MRA difference and interval (same resamples)')

    # ---- the margin, from the committed Step 18 record (prereg 4.1)
    q1_hi = float(c18[Q1_18]['ci'][1])
    if not q1_hi < 0:
        sys.exit('FATAL: the Step 18 Q1 interval on %s does not exclude zero (upper bound %+.6f); M1 is undefined' % (split, q1_hi))
    M1 = abs(q1_hi)
    delta = MARGIN_FRACTION * M1
    print('[margin] M1 = |upper bound of Step 18 Q1| = %.6f  ->  delta = %.2f x M1 = %.6f' % (M1, MARGIN_FRACTION, delta))
    st.update(true_norm=true_norm, pre_full=pre_full, mra_seed=mra_seed, mra=mra, BM1=BM1, ref_c=ref_c, M1=M1, delta=delta)
    return st


def score_mb32(split, st, mbrec, rows, pc_rows):
    """Phase B for one split: load and score the Step 21 arm against the checked comparators (prereg 4.1-4.3)."""
    print('\n' + '#' * 72 + '\n# %s: the Step 21 arm (m = %d)\n' % (split, M) + '#' * 72)
    pin = json.load(open(pin_path(split)))
    if int(pin['m']) != M or pin['split'] != split:
        sys.exit('FATAL: %s pins %s m = %s' % (pin_path(split), pin['split'], pin['m']))
    Mt, norms, inp, drugs, n = st['Mt'], st['norms'], st['inp'], st['drugs'], st['n']
    Mt[ARM], norms[ARM], inp[ARM] = score_arm(st, lambda s, sha=pin['sha256']: load_eval21(split, s, sha), ARM, np.random.RandomState(0))
    f32 = float(mbrec['per_split'][split]['f_cos'][str(M)])
    print('[shared] every Step 21 run carries the comparators\' conditions, real cells and conditioning control cells (bitwise)')
    st['real'] = st['ctrl_ref'] = None

    mra_seed, mra = st['mra_seed'], st['mra']
    mra_seed[ARM] = [s16.mra_pre(x, st['pre_full']) for x in norms[ARM]]
    mra[ARM] = float(np.mean(mra_seed[ARM]))
    BM2 = mra_boot({nm: norms[nm] for nm in ('recipe', ARM)}, st['true_norm'], st['ctx_id'], st['n_ctx'], drugs)
    if not np.array_equal(BM2['recipe'], st['BM1']['recipe']):
        sys.exit('FATAL: the second MRA bootstrap did not reproduce the first one\'s resamples')

    metrics = {}
    for nm in ('recipe', 'ot_pair', ARM):
        m_ = dict(energy=float(np.nanmean(Mt[nm]['en'])), de_overlap=float(np.nanmean(Mt[nm]['jac'])),
                  pds_mean=float(np.nanmean(Mt[nm]['pds'])), dpc_mean=float(np.nanmean(Mt[nm]['dpc'])), mra=mra[nm],
                  tracking=float(np.nanmean(Mt[nm]['trk'])), tracking_null=float(np.nanmean(Mt[nm]['trk_null'])),
                  mra_per_seed=mra_seed[nm])
        if 'per_seed' in Mt[nm]:
            m_['per_seed'] = Mt[nm]['per_seed']
        metrics[nm] = m_
        for key_ in ('energy', 'de_overlap', 'pds_mean', 'dpc_mean', 'mra', 'tracking'):
            rows.append(dict(split=split, arm=nm, level='metric', key=key_, value=m_[key_], ci_lo='', ci_hi='', verdict=''))
        print('[abs] %-8s energy %.4f | DE-overlap %.4f | PDS %.4f | DPC-50 %.4f | MRA %+.4f | tracking %+.4f (null %+.4f)'
              % (nm, m_['energy'], m_['de_overlap'], m_['pds_mean'], m_['dpc_mean'], m_['mra'], m_['tracking'], m_['tracking_null']))

    c = {}
    c['Q1_energy_mb32_minus_ot_pair'] = contrast(Mt[ARM]['en'], Mt['ot_pair']['en'], drugs)
    c['Q2_pds_mb32_minus_ot_pair'] = contrast(Mt[ARM]['pds'], Mt['ot_pair']['pds'], drugs)
    c['secondary_de_mb32_minus_ot_pair'] = contrast(Mt[ARM]['jac'], Mt['ot_pair']['jac'], drugs)
    c['secondary_energy_mb32_minus_recipe'] = contrast(Mt[ARM]['en'], Mt['recipe']['en'], drugs)
    c['secondary_de_mb32_minus_recipe'] = contrast(Mt[ARM]['jac'], Mt['recipe']['jac'], drugs)
    c['secondary_pds_mb32_minus_recipe'] = contrast(Mt[ARM]['pds'], Mt['recipe']['pds'], drugs)
    c['secondary_dpc_mb32_minus_recipe'] = contrast(Mt[ARM]['dpc'], Mt['recipe']['dpc'], drugs)
    c['secondary_mra_mb32_minus_recipe'] = s16.verdict_of(mra[ARM] - mra['recipe'], BM2[ARM] - BM2['recipe'])
    c['secondary_tracking_mb32_minus_recipe'] = contrast(Mt[ARM]['trk'], Mt['recipe']['trk'], drugs)
    c['secondary_tracking_mb32_minus_null'] = contrast(Mt[ARM]['trk'], Mt[ARM]['trk_null'], drugs)
    for key, metric in (('energy', 'en'), ('de', 'jac'), ('pds', 'pds')):
        line = Mt['recipe'][metric] + f32 * (Mt['ot_pair'][metric] - Mt['recipe'][metric])
        c['exploratory_excess_%s_over_line' % key] = contrast(Mt[ARM][metric], line, drugs)
    c.update(st['ref_c'])
    delta = st['delta']
    tests = dict(Q1=dict(c['Q1_energy_mb32_minus_ot_pair'], M1=st['M1'], delta=delta,
                         result=ni_test(c['Q1_energy_mb32_minus_ot_pair'], delta)),
                 Q2=dict(c['Q2_pds_mb32_minus_ot_pair'], result=c['Q2_pds_mb32_minus_ot_pair']['verdict']))
    print('\n%s on %s (f_cos %.3f)' % (ARM, split, f32))
    for key, cc in c.items():
        print('  %-42s %s' % (key, s16.fmt(cc)))
        rows.append(dict(split=split, arm='ot_pair' if key.startswith('reference') else ARM, level='contrast', key=key,
                         value=cc['diff'], ci_lo=cc['ci'][0], ci_hi=cc['ci'][1], verdict=cc['verdict']))
    rows.append(dict(split=split, arm=ARM, level='margin', key='delta', value=delta, ci_lo='', ci_hi='', verdict=''))
    print('  Q1 (non-inferiority): upper bound %+.5f against delta %.5f -> %s' % (c['Q1_energy_mb32_minus_ot_pair']['ci'][1], delta, tests['Q1']['result']))
    print('  Q2 (superiority):     lower bound %+.5f against 0 -> %s' % (c['Q2_pds_mb32_minus_ot_pair']['ci'][0], tests['Q2']['result']))
    print('[bootstrap] %d replicates (%d s)' % (N_BOOT, time.time() - T0))

    for i in range(n):
        r = dict(split=split, condition=st['cond'][i], line=st['lines'][i], plate=st['plates'][i], structure=drugs[i],
                 dose_nM=st['doses'][i])
        for nm in ('recipe', 'ot_pair', ARM):
            for key, col in (('en', 'energy'), ('jac', 'de_overlap'), ('pds', 'pds'), ('dpc', 'dpc50'), ('trk', 'tracking')):
                r['%s_%s' % (nm, col)] = Mt[nm][key][i]
        pc_rows.append(r)
    res = dict(n_conditions=n, n_structures=st['n_drugs'], n_contexts=st['n_ctx'], min_context=MIN_CTX, f_cos_m32=f32,
               M1=st['M1'], delta=delta, tests=tests, metrics=metrics, contrasts=c)
    return res, inp


def main():
    if len(sys.argv) > 1:
        sys.exit('FATAL: this scorer takes no options; it scores %s' % (SPLITS,))
    for q in (OUT, OUT_CSV, OUT_PC):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 9:
        sys.exit('FATAL: prereg incomplete')
    tracked = [PRE, ME, 'score_step18_tahoe.py', 'score_step16_tahoe.py', 'score_step15_tahoe.py', 'score_step15c_tahoe.py',
               'score_step15d_utility.py', 'score_step15e_pooled.py', 'step14_metrics.py', 'Squidiff/scrna_datasets.py',
               PC15, PC15C, REC15B, REC15C, REC18, PC18, OTREC, MBREC, s15.RECORD, REC20] + [pin_path(s) for s in SPLITS]
    guard_tracked(tracked)
    record = json.load(open(s15.RECORD))
    otrec, mbrec, rec18, rec20 = json.load(open(OTREC)), json.load(open(MBREC)), json.load(open(REC18)), json.load(open(REC20))
    if not rec18.get('final'):
        sys.exit('FATAL: the Step 18 record is not the final one')
    if M not in [int(x) for x in mbrec['selection']['chosen']]:
        sys.exit('FATAL: m = %d was not chosen by the pairing rule (%s)' % (M, mbrec['selection']['chosen']))
    if rec20.get('final') is not False or rec20.get('selection', {}).get('split') != DEV:
        sys.exit('FATAL: %s is not the Step 20 development record' % REC20)
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    print('[prereg] committed; scorer committed; commit %s; confirmation splits %s; arm %s (m = %d)'
          % (commit[:7], list(SPLITS), ARM, M))
    s15c.self_check()
    s16.mra_self_check()
    pc15, pc15c = pd.read_csv(PC15), pd.read_csv(PC15C)
    pc18all = pd.read_csv(PC18, float_precision='round_trip')

    # Two phases, so that every comparator check on BOTH splits finishes before the first Step 21 number is printed
    # (prereg section 6): phase A builds and checks the comparators of each split and keeps what phase B needs; phase B
    # loads and scores the Step 21 arm.
    state = {}
    for split in SPLITS:
        state[split] = comparators(split, record, otrec, rec18, pc15, pc15c, pc18all)
    print('\n[reproduction] complete on %s: every comparator value checked; no Step 21 number has been printed yet'
          % ' and '.join(SPLITS))

    results, inputs, rows, pc_rows = {}, {}, [], []
    for split in SPLITS:
        results[split], inputs[split] = score_mb32(split, state.pop(split), mbrec, rows, pc_rows)

    # ------------------------------------------------------------------ the development split, descriptively (prereg 2)
    c20 = rec20['per_split'][DEV]['contrasts'][ARM]
    q1_dev = float(rec18['per_split'][DEV]['contrasts'][Q1_18]['ci'][1])
    dev = dict(source=REC20, M1=abs(q1_dev), delta=MARGIN_FRACTION * abs(q1_dev),
               contrasts={'Q1_energy_mb32_minus_ot_pair': c20['secondary_energy_minus_ot_pair'],
                          'Q2_pds_mb32_minus_ot_pair': c20['secondary_pds_minus_ot_pair'],
                          'secondary_de_mb32_minus_ot_pair': c20['secondary_de_minus_ot_pair']})
    dev['Q1_result'] = ni_test(dev['contrasts']['Q1_energy_mb32_minus_ot_pair'], dev['delta'])
    dev['Q2_result'] = dev['contrasts']['Q2_pds_mb32_minus_ot_pair']['verdict']
    for key, cc in dev['contrasts'].items():
        rows.append(dict(split=DEV, arm=ARM, level='development_descriptive', key=key, value=cc['diff'], ci_lo=cc['ci'][0],
                         ci_hi=cc['ci'][1], verdict=cc['verdict']))
    rows.append(dict(split=DEV, arm=ARM, level='margin', key='delta', value=dev['delta'], ci_lo='', ci_hi='', verdict=''))

    # ------------------------------------------------------------------ verdict (prereg section 5)
    E = all(results[s]['tests']['Q1']['result'] == 'non-inferior' for s in SPLITS)
    P = all(results[s]['tests']['Q2']['result'] == 'above zero' for s in SPLITS)
    if E and P:
        verdict = '1 CONFIRMED'
    elif P:
        verdict = '2 LESS PDS LOSS, REALISM NOT SHOWN'
    elif E:
        verdict = '3 REALISM KEPT, NO PDS ADVANTAGE SHOWN'
    else:
        verdict = '4 NOT CONFIRMED'
    reversals = [s for s in SPLITS if results[s]['tests']['Q2']['result'] == 'below zero'
                 or results[s]['tests']['Q1']['result'].startswith('reversal')]
    if reversals:
        verdict += ' with a reversal on %s' % ', '.join(reversals)
    print('\n' + '=' * 72 + '\nPRE-REGISTERED VERDICT (section 5): %s' % verdict)
    for s in SPLITS:
        q1, q2 = results[s]['tests']['Q1'], results[s]['tests']['Q2']
        print('  %s: Q1 D_E %+.5f [%+.5f, %+.5f], delta %.5f -> %s | Q2 D_P %+.5f [%+.5f, %+.5f] -> %s'
              % (s, q1['diff'], q1['ci'][0], q1['ci'][1], q1['delta'], q1['result'], q2['diff'], q2['ci'][0], q2['ci'][1], q2['result']))
    print('  development split (descriptive, Step 20 record): Q1 %s (delta %.5f), Q2 %s' % (dev['Q1_result'], dev['delta'], dev['Q2_result']))
    print('  Section 7 expected: 1 CONFIRMED.\n' + '=' * 72)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'arm', 'level', 'key', 'value', 'ci_lo', 'ci_hi', 'verdict'])
        w.writeheader(); w.writerows(rows)
    with open(OUT_PC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0].keys()))
        w.writeheader(); w.writerows(pc_rows)
    json.dump(dict(prereg=PRE, verdict=verdict, E=E, P=P, reversal_splits=reversals, margin_fraction=MARGIN_FRACTION,
                   splits=list(SPLITS), arm=ARM, m=M, interval='97.5% percentile per split, structure-clustered',
                   n_boot=N_BOOT, n_seeds=s15.N_SEEDS, per_split=results, development_descriptive=dev, inputs=inputs,
                   records=dict(step15b=s15.sha(PC15), step15c=s15.sha(PC15C), step18=s15.sha(REC18), step18_per_condition=s15.sha(PC18),
                                ot_pairing=s15.sha(OTREC), minibatch_pairing=s15.sha(MBREC), data_record=s15.sha(s15.RECORD),
                                step20_development=s15.sha(REC20), pins={s: s15.sha(pin_path(s)) for s in SPLITS}),
                   commit=commit, seconds=round(time.time() - T0)),
              open(OUT, 'w'), indent=1, allow_nan=True)
    print('\nwrote %s, %s and %s  (%.0f s)' % (OUT, OUT_CSV, OUT_PC, time.time() - T0))


if __name__ == '__main__':
    main()
