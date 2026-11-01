#!/usr/bin/env python3
"""
score_step24_tahoe.py - Step 24: does the rewired recipe's compound discrimination require the right fingerprint?

Purpose
-------
Step 15c reported that the rewired recipe identifies a held-out compound from its predicted response (PDS 0.7223 on
the development split, chance 0.5). Step 24 hands the same trained checkpoints a different compound's fingerprint
(`fpshuf`, one fixed derangement of the split's held-out structures) or no compound information at all (`fpzero`, an
all-zero fingerprint) at sampling time, and asks whether the discrimination survives. Pre-registration:
PREREG_step24_fingerprint_swap.md. This script does only what its sections 4, 5 and 7 say.

Logic
-----
1. Guards (prereg 7.1-7.2): the pre-registration, this scorer and its job script, the other Step 24 scripts and the
   generated harness, the scorers it imports and the records it reads are committed and identical to HEAD; the
   permutation record is self-consistent and is a derangement of exactly the data record's held-out structures; no
   output exists. Only drug_split_1 is read; the script takes no options.
2. Inputs: the recipe's three Step 15b evaluations (vectors and per-cell populations, hash-checked against their
   provenance by score_step15_tahoe.load_eval); the six Step 24 arm runs and the reproduction run, hash-checked
   against provenance/s24ev_*.json, which must name the arm, the substitution mode, the same checkpoint as the Step
   15b evaluation of that seed and - for fpshuf - the committed permutation's sha256.
3. Reproduction (prereg 7.3-7.5, 7.7), before any Step 24 value is printed: the reproduction run equals the recipe's
   seed-0 Step 15b arrays bitwise (cond, true, A, B, C and every stored real cell); every arm shares the recipe's
   conditions, measured means, control means and real cells, so only the prediction differs; every fpshuf and fpzero
   prediction differs from the recipe's same-seed prediction at every condition; the permutation's strings are the
   SMILES of the held-out conditions; the recipe's per-condition PDS, DE-overlap, DPC-50 and energy rebuilt here
   equal the Step 15c and 15b records to 1e-9, its MRA equals Step 16 to four decimals, and the bootstrap reproduces
   Step 15c's split-1 interval.
4. Measures (prereg 4), per seed and then averaged per condition: PDS on the Step 15c candidate sets, DE-overlap and
   DPC-50 (Step 15b), energy distance on the stored cells (Step 15b), and MRA (Step 16 section 4.4; per seed, then
   averaged). All held-out conditions are scored; the salt-twin conditions of Step 23 (read from its per-condition
   record) are also reported apart.
5. Intervals: 97.5 % structure-clustered percentile bootstrap, 5,000 replicates, a fresh RandomState(0) for each set
   of scored conditions (the Step 23 implementation, imported). Gated contrasts and the verdict of prereg section 5,
   then the secondary contrasts it lists.
Outputs (never overwritten): tahoe_results/step24_swap.json, step24_swap_per_condition.csv, step24_swap_summary.csv.

Run on a compute node from the worktree root, after all seven evaluation runs, through run_step24_score.sh.
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15        # noqa: E402  loaders, metrics, provenance checks (Step 15b)
import score_step15c_tahoe as s15c      # noqa: E402  PDS on candidate sets (Step 15c)
import score_step23_screen as s23       # noqa: E402  bootstrap, MRA, intervals, git guards (Step 23, validated)

PRE = 'PREREG_step24_fingerprint_swap.md'
SPLIT = 'drug_split_1'                                     # the development split; no other split is read
PERM = os.path.join(s15.RES, 'step24', 'permutation.json')
S23_REC = os.path.join(s15.RES, 'step23_screen.json')
S23_PC = os.path.join(s15.RES, 'step23_screen_per_condition.csv')
PC15C = os.path.join(s15.RES, 'step15c_pds_per_condition.csv')
OUT_JSON = os.path.join(s15.RES, 'step24_swap.json')
OUT_PC = os.path.join(s15.RES, 'step24_swap_per_condition.csv')
OUT_SUM = os.path.join(s15.RES, 'step24_swap_summary.csv')
ARMS24 = {'fpshuf': 'shuf', 'fpzero': 'zero'}               # arm -> S24_MODE of its runs
SUBS = ('recipe', 'fpshuf', 'fpzero')
PAIRS = (('recipe', 'fpshuf'), ('recipe', 'fpzero'), ('fpshuf', 'fpzero'))
MIN_SECTIONS = 9
T0 = time.time()
say, die = s23.say, s23.die


def load_s24(arm, seed, perm_sha, ck_sha):
    """One Step 24 evaluation, checked against the provenance record the launcher's post-check wrote."""
    tag = 's24_%s_%s_seed_%d' % (SPLIT, arm, seed)
    f = os.path.join(s15.RES, 'vectors_%s.npz' % tag)
    fpc = os.path.join(s15.RES, 'vectors_%s_pc.npz' % tag)
    pj = os.path.join('provenance', 's24ev_%s.json' % tag)
    for q in (f, fpc, pj):
        if not os.path.isfile(q):
            die('missing %s' % q)
    pv = json.load(open(pj))
    s15.check_prov(pv, tag, dict(status='completed', arm=arm, split=SPLIT, seed=seed, s24_mode=ARMS24.get(arm, 'off'),
                                 checkpoint_arm='recipe', drug_encoding='fingerprint', guidance_w=None, nfe=50,
                                 max_cells_per_cond=s15.CAP))
    if arm == 'fpshuf' and pv.get('permutation_sha256') != perm_sha:
        die('%s used permutation %s, the committed record is %s' % (tag, pv.get('permutation_sha256'), perm_sha))
    if pv.get('checkpoint_sha256') != ck_sha:
        die('%s was sampled from checkpoint %s, the Step 15b evaluation of seed %d used %s'
            % (tag, str(pv.get('checkpoint_sha256'))[:12], seed, str(ck_sha)[:12]))
    if pv.get('vectors_sha256') != s15.sha(f) or pv.get('per_cell_sha256') != s15.sha(fpc):
        die('%s vectors or per-cell hash differs from its provenance' % tag)
    v = np.load(f, allow_pickle=True)
    p = np.load(fpc, allow_pickle=True)
    out = dict(tag=tag, file=f, cond=np.array([str(x) for x in v['cond']]), true=np.asarray(v['true'], np.float64),
               B=np.asarray(v['B'], np.float64), C=np.asarray(v['C'], np.float64),
               pred=[np.asarray(x, np.float32) for x in p['pred']],
               true_cells=[np.asarray(x, np.float32) for x in p['true']],
               rec=dict(tag=tag, vectors_sha256=pv['vectors_sha256'], per_cell_sha256=pv['per_cell_sha256'],
                        checkpoint_sha256=pv['checkpoint_sha256'], eval_commit=pv.get('commit'),
                        slurm_array_job=pv.get('slurm_array_job'), permutation_sha256=pv.get('permutation_sha256')))
    if len(out['pred']) != len(out['cond']) or len(out['true_cells']) != len(out['cond']):
        die('%s per-cell length differs from vectors' % tag)
    return out


def per_seed(v, Y, contexts, cand):
    """The per-condition measures of one evaluation (prereg 4): PDS, DE-overlap, DPC-50, energy, response size."""
    D = v['B'] - v['C']
    j, d = s15.thesis_per_cond(v['true'], v['B'], v['C'])
    return dict(pds=s15c.pds_sets(D, Y, contexts, cand), de=j, dpc=d,
                en=np.array([s15.energy(v['pred'][i], v['true_cells'][i]) for i in range(len(D))]),
                size=np.linalg.norm(D, axis=1))


def main():
    if len(sys.argv) > 1:
        die('this script takes no options (prereg 8: development split only)')
    for q in (OUT_JSON, OUT_PC, OUT_SUM):
        if os.path.exists(q):
            die('%s exists; refusing to overwrite a record' % q)
    if not os.path.isfile(PRE) or open(PRE).read().count('\n## ') < MIN_SECTIONS:
        die('pre-registration missing or incomplete')
    s23.committed_and_clean([PRE, 'score_step24_tahoe.py', 'run_step24_score.sh', 'port_harness_step24.py',
                             'make_step24_permutation.py', 'leakage_test_fm_step24.py', 'eval_step24_tahoe.sh',
                             'score_step23_screen.py', 'score_step15_tahoe.py', 'score_step15c_tahoe.py',
                             'step14_metrics.py'], 'pre-registration or scripts')
    s23.committed_and_clean([PERM, s15.RECORD, s15c.PC15, PC15C, S23_REC, S23_PC], 'records')
    commit = s23.git('rev-parse', 'HEAD').stdout.strip()
    ref = s23.REFERENCE
    if s23.TEST_REFERENCE:
        ref = json.load(open(s23.TEST_REFERENCE))
        say('[WARNING] S23_TEST_REFERENCE set: reproduction targets read from %s (synthetic test only)'
            % s23.TEST_REFERENCE)

    # ---------------------------------------------------------------- the permutation (prereg 3, 7.2)
    perm = json.load(open(PERM))
    psids = perm['structures']
    if hashlib.sha256('\n'.join('%s>%s' % (k, perm['map'][k]) for k in psids).encode()).hexdigest() != perm['sha256']:
        die('the permutation record does not match its own sha256')
    record = json.load(open(s15.RECORD))
    rec_split = record['splits'][SPLIT.split('_')[-1]]
    held = sorted(rec_split['held_structures'])
    if psids != held or perm.get('split') != SPLIT:
        die('the permutation does not cover exactly the held-out structures of %s' % SPLIT)
    if any(perm['map'][k] == k for k in psids) or sorted(perm['map'].values()) != psids:
        die('the permutation is not a derangement')
    say('[guards] pre-registration, scripts and records committed and clean; commit %s; permutation %s, a derangement '
        'of the %d held-out structures of %s' % (commit[:7], perm['sha256'][:12], len(psids), SPLIT))
    s15c.self_check()

    # ---------------------------------------------------------------- inputs
    rec = [s15.load_eval(SPLIT, 'recipe', s) for s in range(s15.N_SEEDS)]
    arms = {a: [load_s24(a, s, perm['sha256'], rec[s]['rec']['checkpoint_sha256']) for s in range(s15.N_SEEDS)]
            for a in ARMS24}
    repro = load_s24('recipe_repro', 0, perm['sha256'], rec[0]['rec']['checkpoint_sha256'])
    ref0 = rec[0]
    cond, n = ref0['cond'], len(ref0['cond'])
    if n != int(rec_split['n_test_conditions']):
        die('%d conditions, the data record expects %d' % (n, rec_split['n_test_conditions']))
    say('[inputs] recipe (Step 15b) and the Step 24 runs, every file hash-checked against its provenance; each Step 24 '
        'run sampled from the same checkpoint as the Step 15b evaluation of its seed')

    # ---------------------------------------------------------------- reproduction (prereg 7.3, 7.4)
    # every array the Step 15b file stores (cond, true, A, B, C), and every stored cell, predicted and real
    r0 = np.load(os.path.join(s15.RES, 'vectors_%s.npz' % ref0['tag']), allow_pickle=True)
    rr = np.load(repro['file'], allow_pickle=True)
    if sorted(r0.files) != sorted(rr.files):
        die('the reproduction run stores %s, Step 15b seed 0 stores %s' % (sorted(rr.files), sorted(r0.files)))
    for k in r0.files:
        x0, x1 = (np.array([str(c) for c in r0[k]]), np.array([str(c) for c in rr[k]])) if k == 'cond' else (r0[k], rr[k])
        if x0.dtype != x1.dtype or not np.array_equal(x0, x1):
            die('the reproduction run differs from Step 15b seed 0 in %s' % k)
    if not all(np.array_equal(repro['true_cells'][i], ref0['true_cells'][i]) for i in range(n)) \
            or not all(np.array_equal(repro['pred'][i], ref0['pred'][i]) for i in range(n)):
        die('the reproduction run differs from Step 15b seed 0 in the stored real or predicted cells')
    say('[reproduction] the substitution switched off reproduces the Step 15b recipe (seed 0) bitwise: %s and every '
        'stored real and predicted cell' % ', '.join(sorted(r0.files)))
    # each Step 24 run against the recipe run of its own seed (bitwise); the recipe seeds against each other as the
    # Step 15b scorer checks them (means to 1e-5, real cells exactly)
    for L in arms.values():
        for s, v in enumerate(L):
            r = rec[s]
            if not np.array_equal(v['cond'], cond) or not np.array_equal(v['true'], r['true']) \
                    or not np.array_equal(v['C'], r['C']) \
                    or not all(np.array_equal(v['true_cells'][i], r['true_cells'][i]) for i in range(n)):
                die('%s does not share the conditions, measured means, control means and real cells of %s'
                    % (v['tag'], r['tag']))
    for r in rec:
        if not np.array_equal(r['cond'], cond) or np.abs(r['true'] - ref0['true']).max() > 1e-5 \
                or np.abs(r['C'] - ref0['C']).max() > 1e-5 \
                or not all(np.array_equal(r['true_cells'][i], ref0['true_cells'][i]) for i in range(n)):
            die('%s differs from %s in conditions, means or real cells' % (r['tag'], ref0['tag']))
    say('[reproduction] every Step 24 run shares the %d conditions, measured and control means and real cells of the '
        'recipe run of its seed, bitwise; only the prediction differs' % n)

    # ---------------------------------------------------------------- the substitution reached the model (prereg 7.7)
    unchanged = {}
    for a, L in arms.items():
        for s, v in enumerate(L):
            same = np.all(v['B'] == rec[s]['B'], axis=1)
            unchanged[v['tag']] = int(same.sum())
            if same.any():
                die('%s: the predicted mean equals the recipe\'s seed-%d prediction bitwise at %d of %d conditions '
                    '(e.g. %s); the substitution did not reach the model there - inspect the run before scoring'
                    % (v['tag'], s, int(same.sum()), n, cond[np.flatnonzero(same)[0]]))
    say('[substitution] every fpshuf and fpzero prediction differs from the recipe\'s same-seed prediction at all %d '
        'conditions' % n)

    # ---------------------------------------------------------------- annotations and structure ids (Step 23 path)
    pc15 = pd.read_csv(s15c.PC15)
    pc = pc15[pc15['split'] == SPLIT].reset_index(drop=True)
    if len(pc) != n or not np.array_equal(pc['condition'].astype(str).values, cond):
        die('the Step 15b per-condition record does not list the split-1 conditions in the vectors order')
    lines, drugs = pc['line'].astype(str).values, pc['structure'].astype(str).values
    doses = pc['dose_nM'].astype(float).values
    dr = record['drugs']
    smi2sid = {}
    for name, smi in dr['canonical_smiles'].items():
        for form in {smi, dr.get('table_smiles', {}).get(name, smi)}:            # canonical, or the table form
            if smi2sid.setdefault(form, dr['structure'][name]) != dr['structure'][name]:
                die('the data record maps one SMILES to two structures (%s)' % name)
    miss = sorted(set(drugs) - set(smi2sid))
    if miss:
        die('%d held-out SMILES are not SMILES of the data record, e.g. %s' % (len(miss), miss[:2]))
    sid_c = np.array([smi2sid[x] for x in drugs])
    if sorted(set(sid_c)) != held:
        die('the structure ids of the held-out conditions differ from the data record held_structures')
    bad = [i for i in range(n) if perm['smiles'][sid_c[i]]['own'] != drugs[i]]
    if bad:
        die('the permutation\'s own SMILES differs from the condition\'s SMILES at %d conditions (e.g. %s)'
            % (len(bad), cond[bad[0]]))
    say('[identity] every held-out SMILES maps to one structure id; the permutation\'s strings are the SMILES of the '
        'held-out conditions')

    # ---------------------------------------------------------------- candidate sets (Step 15c section 4)
    Y = ref0['true'] - ref0['C']
    tn = np.linalg.norm(Y, axis=1)
    ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
    cpos = {c: j for j, c in enumerate(sorted(set(ctx_key)))}
    ctx_code = np.array([cpos[c] for c in ctx_key])
    contexts = [np.where(ctx_key == u)[0] for u in np.unique(ctx_key)]
    cand = {}
    for members in contexts:
        for p_ in members:
            cand[p_] = np.array([q for q in members if q == p_ or drugs[q] != drugs[p_]])

    # ---------------------------------------------------------------- per-seed measures
    say('[measures] scoring %d evaluations on %d conditions in %d (line, dose) contexts (energy on the stored cells)'
        % (len(SUBS) * s15.N_SEEDS, n, len(contexts)))
    M = {'recipe': [per_seed(v, Y, contexts, cand) for v in rec]}
    for a, L in arms.items():
        M[a] = [per_seed(v, Y, contexts, cand) for v in L]

    def seedmean(a, k):
        return np.nanmean(np.vstack([m[k] for m in M[a]]), axis=0)

    # ---------------------------------------------------------------- the recipe reproduces its records (prereg 7.5)
    p15c = pd.read_csv(PC15C)
    p15c = p15c[p15c['split'] == SPLIT].reset_index(drop=True)
    if not np.array_equal(p15c['condition'].astype(str).values, cond):
        die('the Step 15c per-condition record does not list the split-1 conditions in the vectors order')
    if not np.array_equal(np.array([len(cand[i]) for i in range(n)]), p15c['n_candidates'].values):
        die('candidate-set sizes differ from the Step 15c record')
    for name, mine, theirs in (('PDS', seedmean('recipe', 'pds'), p15c['recipe_pds'].values),
                               ('DE-overlap', seedmean('recipe', 'de'), pc['recipe_de_overlap'].values),
                               ('DPC-50', seedmean('recipe', 'dpc'), pc['recipe_dpc50'].values),
                               ('energy', seedmean('recipe', 'en'), pc['recipe_energy'].values)):
        if not s15c.close(mine, theirs):
            die('the recipe\'s %s does not reproduce the Step 15b/15c per-condition record to 1e-9' % name)
    allidx = np.arange(n)
    mra_seeds = [s23.mra_point(m['size'], tn, ctx_code, allidx) for m in M['recipe']]
    if any(abs(x - y) > ref['mra_tol'] for x, y in zip(mra_seeds, ref['mra']['recipe_seeds'])):
        die('the recipe\'s MRA does not reproduce %s: %s' % (ref['source_mra'], ['%.6f' % x for x in mra_seeds]))
    b_all = s23.Boot(allidx, drugs, n)
    pt, reps = b_all.mean_reps(seedmean('recipe', 'pds') - 0.5)
    q1_15c = s23.interval(pt, reps)
    bt = ref['boot_tol']
    if abs(q1_15c['diff'] - ref['boot_q1']['diff']) > bt or abs(q1_15c['ci'][0] - ref['boot_q1']['lo']) > bt \
            or abs(q1_15c['ci'][1] - ref['boot_q1']['hi']) > bt:
        die('the bootstrap does not reproduce %s: got %s' % (ref['source_boot'], s23.fmt(q1_15c)))
    say('[reproduction] the recipe\'s PDS, DE-overlap, DPC-50 and energy equal the Step 15c/15b records to 1e-9; its '
        'MRA equals Step 16 (seeds %s); the bootstrap reproduces Step 15c (recipe - 0.5): %s'
        % (' '.join('%.4f' % x for x in mra_seeds), s23.fmt(q1_15c)))

    # ---------------------------------------------------------------- salt-twin conditions, from the Step 23 records
    p23 = pd.read_csv(S23_PC, dtype={'structure_id': str})
    if not np.array_equal(p23['condition'].astype(str).values, cond) \
            or not np.array_equal(p23['structure_id'].astype(str).values, sid_c):
        die('the Step 23 per-condition record does not list the split-1 conditions and structures in the vectors order')
    twin = p23['scored'].values == 0
    s23rec = json.load(open(S23_REC))
    twin_ids = sorted(t['structure_id'] for t in s23rec['design']['twins'])
    if sorted(set(sid_c[twin])) != twin_ids or set(sid_c[~twin]) & set(twin_ids):
        die('the conditions Step 23 did not score are not exactly those of its twin structures')
    say('[twins] %d salt-twin structures (%d conditions) from the Step 23 records: scored with the rest, also reported '
        'apart' % (len(twin_ids), int(twin.sum())))

    # ---------------------------------------------------------------- values
    keys = (('pds', 'PDS'), ('de', 'DE-overlap'), ('dpc', 'DPC-50'), ('en', 'energy'))
    vals = {}
    for a in SUBS:
        vals[a] = {k: float(np.nanmean(seedmean(a, k))) for k, _ in keys}
        vals[a]['per_seed_pds'] = [float(np.nanmean(m['pds'])) for m in M[a]]
        vals[a]['mra_per_seed'] = [s23.mra_point(m['size'], tn, ctx_code, allidx) for m in M[a]]
        vals[a]['mra'] = float(np.mean(vals[a]['mra_per_seed']))
        vals[a]['per_dose_pds'] = {'%g' % d: float(np.nanmean(seedmean(a, 'pds')[np.isclose(doses, d)]))
                                   for d in sorted(set(doses))}
        vals[a]['twins_pds'] = float(np.nanmean(seedmean(a, 'pds')[twin])) if twin.any() else None
        vals[a]['non_twins_pds'] = float(np.nanmean(seedmean(a, 'pds')[~twin]))
    say('\n' + '=' * 100 + '\nSTEP 24, %s: %d held-out conditions, %d structures, three seeds per arm\n'
        % (SPLIT, n, b_all.G) + '=' * 100)
    say('  %-8s %8s %10s %8s %8s %8s   per-seed PDS' % ('arm', 'PDS', 'DE-overlap', 'DPC-50', 'energy', 'MRA'))
    for a in SUBS:
        v = vals[a]
        say('  %-8s %8.4f %10.4f %8.4f %8.4f %+8.4f   %s' % (a, v['pds'], v['de'], v['dpc'], v['en'], v['mra'],
                                                          ' '.join('%.4f' % x for x in v['per_seed_pds'])))
    for a in SUBS:
        say('  PDS by dose  %-7s %s' % (a, '  '.join('%s nM %.4f' % (d, x) for d, x in vals[a]['per_dose_pds'].items())))
    if twin.any():
        say('  PDS on the %d salt-twin conditions (descriptive): %s | on the other %d: %s'
            % (int(twin.sum()), ', '.join('%s %.4f' % (a, vals[a]['twins_pds']) for a in SUBS), int((~twin).sum()),
               ', '.join('%s %.4f' % (a, vals[a]['non_twins_pds']) for a in SUBS)))

    # ---------------------------------------------------------------- contrasts (prereg 5)
    con = {}

    def contrast(name, xa, xb):
        diff = np.asarray(xa, np.float64) - np.asarray(xb, np.float64)
        ok = np.isfinite(diff)
        b = b_all if ok.all() else s23.Boot(np.flatnonzero(ok), drugs, n)   # fresh RandomState(0) for that set
        pt_, reps_ = b.mean_reps(diff)
        con[name] = s23.interval(pt_, reps_)
        con[name]['n_conditions'] = int(ok.sum())
        return con[name]

    half = np.full(n, 0.5)
    say('\nGATED (PDS):')
    say('  Q1 primary     recipe - fpshuf    %s' % s23.fmt(contrast('Q1_pds_recipe_minus_fpshuf',
                                                                     seedmean('recipe', 'pds'), seedmean('fpshuf', 'pds'))))
    say('  Q2 co-primary  fpshuf - 0.5       %s' % s23.fmt(contrast('Q2_pds_fpshuf_minus_0.5',
                                                                     seedmean('fpshuf', 'pds'), half)))
    say('reference (prereg Q3):')
    say('  Q3             fpzero - 0.5       %s' % s23.fmt(contrast('Q3_pds_fpzero_minus_0.5',
                                                                     seedmean('fpzero', 'pds'), half)))
    say('  Q3             fpshuf - fpzero    %s' % s23.fmt(contrast('Q3_pds_fpshuf_minus_fpzero',
                                                                     seedmean('fpshuf', 'pds'), seedmean('fpzero', 'pds'))))
    say('  Q3             recipe - fpzero    %s' % s23.fmt(contrast('Q3_pds_recipe_minus_fpzero',
                                                                     seedmean('recipe', 'pds'), seedmean('fpzero', 'pds'))))
    say('\nsecondary (not gated):')
    for k, lab in keys[1:]:
        for a_, b_ in PAIRS:
            c_ = contrast('sec_%s_%s_minus_%s' % (k, a_, b_), seedmean(a_, k), seedmean(b_, k))
            say('  %-10s %-6s - %-6s  %s%s' % (lab, a_, b_, s23.fmt(c_),
                                               '' if c_['n_conditions'] == n else '  (%d conditions)' % c_['n_conditions']))
    mpt, mreps = b_all.mra_reps([[m['size'] for m in M[a]] for a in SUBS], tn, ctx_code)
    for j, a in enumerate(SUBS):
        con['sec_mra_%s' % a] = s23.interval(mpt[j], mreps[:, j])
        say('  MRA        %-15s  %s' % (a, s23.fmt(con['sec_mra_%s' % a])))
    for a_, b_ in PAIRS:
        ja, jb = SUBS.index(a_), SUBS.index(b_)
        con['sec_mra_%s_minus_%s' % (a_, b_)] = s23.interval(mpt[ja] - mpt[jb], mreps[:, ja] - mreps[:, jb])
        say('  MRA        %-6s - %-6s  %s' % (a_, b_, s23.fmt(con['sec_mra_%s_minus_%s' % (a_, b_)])))
    if abs(mpt[0] - vals['recipe']['mra']) > 1e-10:
        die('the vectorised MRA differs from the plain MRA for the recipe')

    # ---------------------------------------------------------------- verdict (prereg 5), first match
    q1 = con['Q1_pds_recipe_minus_fpshuf']['verdict']
    q2 = con['Q2_pds_fpshuf_minus_0.5']['verdict']
    if q1 == 'above zero' and q2 != 'above zero':
        verdict = '1 DISCRIMINATION IS COMPOUND-SPECIFIC'
    elif q1 == 'above zero':
        verdict = '2 PARTLY COMPOUND-SPECIFIC'
    elif q2 == 'above zero':
        verdict = '3 NOT SHOWN TO BE COMPOUND-SPECIFIC'
    else:
        verdict = '4 UNINTERPRETABLE'
    if q1 == 'below zero':
        extra = '\n  Q1 lies below zero: the wrong fingerprint discriminates better than the right one.'
    elif q1 == 'above zero':
        extra = '\n  discrimination attributable to compound identity (Q1): %+.5f' % con['Q1_pds_recipe_minus_fpshuf']['diff']
    else:
        extra = ''
    say('\n' + '=' * 100 + '\nPRE-REGISTERED VERDICT (section 5): %s\n  Q1 recipe - fpshuf: %s | Q2 fpshuf - 0.5: %s%s\n'
        '  Section 6 expected: 2 PARTLY COMPOUND-SPECIFIC (outcome 3 also live).\n' % (verdict, q1, q2, extra) + '=' * 100)

    # ---------------------------------------------------------------- records (built in memory, then written)
    rows = [dict(kind='value', name=a, metric=k, value=vals[a][k], lo='', hi='', verdict='', n_conditions=n)
            for a in SUBS for k in ('pds', 'de', 'dpc', 'en', 'mra')]
    rows += [dict(kind='interval', name=k_, metric=k_.split('_')[1], value=c_['diff'], lo=c_['ci'][0], hi=c_['ci'][1],
                  verdict=c_['verdict'], n_conditions=c_.get('n_conditions', n)) for k_, c_ in con.items()]
    pc_rows = []
    for i in range(n):
        r = dict(split=SPLIT, condition=cond[i], line=lines[i], structure=drugs[i], structure_id=sid_c[i],
                 substituted_structure_id=perm['map'][sid_c[i]], dose_nM=doses[i], twin=int(twin[i]),
                 n_candidates=int(len(cand[i])), measured_size=tn[i])
        for a in SUBS:
            for k, _ in keys:
                r['%s_%s' % (a, k)] = seedmean(a, k)[i]
            r['%s_size' % a] = float(np.mean([m['size'][i] for m in M[a]]))
        pc_rows.append(r)
    text = json.dumps(dict(prereg=PRE, split=SPLIT, commit=commit, verdict=verdict, n_boot=s23.N_BOOT,
                           interval='97.5% percentile, structure-clustered, fresh RandomState(0) per scored set',
                           reference=ref, test_reference_override=s23.TEST_REFERENCE,
                           permutation=dict(file=PERM, sha256=perm['sha256'], draws=perm.get('draws_until_derangement'),
                                            tanimoto=perm.get('tanimoto_summary')),
                           reproduction=dict(bitwise_seed0=True, recipe_mra_seeds=mra_seeds, bootstrap_15c_q1=q1_15c,
                                             unchanged_conditions=unchanged),
                           design=dict(n_conditions=int(n), n_structures=int(b_all.G), n_contexts=len(contexts),
                                       twin_structures=twin_ids, twin_conditions=int(twin.sum())),
                           values=vals, contrasts=con,
                           inputs=dict(recipe=[v['rec'] for v in rec],
                                       arms={a: [v['rec'] for v in L] for a, L in arms.items()}, repro=repro['rec'],
                                       step15c_per_condition_sha256=s15.sha(PC15C),
                                       step23_per_condition_sha256=s15.sha(S23_PC)),
                           seconds=int(round(time.time() - T0))), indent=2, allow_nan=True)
    with open(OUT_SUM, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    with open(OUT_PC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(pc_rows)
    with open(OUT_JSON, 'w') as f:
        f.write(text)
    say('\nwrote %s, %s and %s  (%.0f s)' % (OUT_JSON, OUT_PC, OUT_SUM, time.time() - T0))


if __name__ == '__main__':
    main()
