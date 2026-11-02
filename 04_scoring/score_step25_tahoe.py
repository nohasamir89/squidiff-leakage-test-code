#!/usr/bin/env python3
"""
score_step25_tahoe.py - Step 25: does the fingerprint-swap null replicate on the confirmation splits, and does the
base model behave the same way?

Purpose
-------
Step 24 found on the development split that the rewired recipe's compound discrimination does not require the right
compound (PDS own 0.7223, wrong 0.7160, none 0.7360). This scorer reads the Step 25 evaluation runs
(PREREG_step25_swap_confirmation.md): the recipe swap arms on drug_split_0 and drug_split_2, and the base-model swap
arms on all three splits. It asks whether the null replicates (verdict A), whether the base model behaves the same
way (verdict B), and whether the recipe-over-base PDS gap depends on either model receiving the right compound
(verdict C, Q3). It does only what the pre-registration's sections 3-5 and 7 describe.

Logic
-----
1. Guards (prereg 7.1-7.2): the pre-registration, this scorer and its job script, the Step 24 and Step 25 scripts and
   the two generated harnesses, the imported scorers and the records read are committed and identical to HEAD; each
   split's permutation record is self-consistent and is a derangement of exactly that split's held-out structures;
   no output exists.
2. Inputs, per split and per checkpoint arm (recipe, base): the Step 15b own-fingerprint evaluation (3 seeds, via
   score_step15_tahoe.load_eval, hash-checked against provenance), and the swap runs - shuf and zero, 3 seeds each,
   plus one reproduction run. Split-1 recipe swap runs are read from the committed Step 24 records (tag s24_*); every
   other swap run from the Step 25 records (tag s25_*). Each swap run's provenance must name the checkpoint arm, the
   substitution mode, the same checkpoint as the Step 15b evaluation of that arm/split/seed, and - for shuf - that
   split's committed permutation sha256.
3. Reproduction (prereg 7.3-7.7), before any Step 25 value is printed: per split and arm, the reproduction run equals
   the Step 15b arrays bitwise; every swap run shares the own run's conditions, measured and control means and real
   cells; every shuf/zero prediction differs from the own prediction at every condition; each own arm's PDS,
   DE-overlap, DPC-50 and energy reproduce the committed Step 15c/15b per-condition records to 1e-9; each own arm's
   bootstrap of PDS - 0.5 reproduces the Step 15c published interval; recipe MRA on the development split reproduces
   Step 16 to four decimals, and MRA is otherwise verified internally (vectorised == plain) and reported.
4. Measures (prereg 3): PDS, DE-overlap, DPC-50, energy, MRA, per seed then averaged per condition, on each split's
   own Step 15c candidate sets. All held-out conditions scored.
5. Contrasts and verdicts (prereg 4-5): per split, Q1/Q2 for recipe and for base; Q3 across arms; verdicts A, B, C.
   Intervals: 97.5 % structure-clustered percentile bootstrap, 5,000 replicates, fresh RandomState(0) per scored set,
   per split (the Step 23 implementation, imported).
Outputs (never overwritten): tahoe_results/step25_swap.json, step25_swap_per_condition.csv, step25_swap_summary.csv.

Run on a compute node from the worktree root, after all 35 evaluation runs, through run_step25_score.sh.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15        # noqa: E402
import score_step15c_tahoe as s15c      # noqa: E402
import score_step23_screen as s23       # noqa: E402

PRE = 'PREREG_step25_swap_confirmation.md'
SPLITS = ('drug_split_1', 'drug_split_0', 'drug_split_2')      # development first (prereg 5)
CKARMS = ('recipe', 'base')
MODES = (('own', 'off'), ('shuf', 'shuf'), ('zero', 'zero'))   # local name -> S24_MODE
PC15B = s15c.PC15                                              # step15_model_per_condition.csv (DE, energy, DPC)
PC15C = os.path.join(s15.RES, 'step15c_pds_per_condition.csv')  # <arm>_pds, n_candidates
S23_PC = os.path.join(s15.RES, 'step23_screen_per_condition.csv')   # split-1 salt-twin flags (descriptive)
OUT_JSON = os.path.join(s15.RES, 'step25_swap.json')
OUT_PC = os.path.join(s15.RES, 'step25_swap_per_condition.csv')
OUT_SUM = os.path.join(s15.RES, 'step25_swap_summary.csv')
MIN_SECTIONS = 9
T0 = time.time()
say, die = s23.say, s23.die

# Published values this scorer reproduces, from the committed project docs (prereg 7). Step 15c findings section 1
# (recipe PDS - 0.5) and section 3 (base PDS - 0.5), five decimals; Step 16 section 4.4 development-split recipe MRA
# per seed, four decimals.
REF_BOOT = {
    'recipe': {'drug_split_1': (0.22234, 0.18829, 0.25411), 'drug_split_0': (0.21811, 0.17822, 0.25899),
               'drug_split_2': (0.21420, 0.18261, 0.24477)},
    'base':   {'drug_split_1': (0.14150, 0.10358, 0.18063), 'drug_split_0': (0.14441, 0.10481, 0.18450),
               'drug_split_2': (0.15488, 0.11017, 0.19767)},
}
REF_MRA_DEV_RECIPE = (0.3112, 0.3237, 0.3265)      # step16_findings.md section 4.4, drug_split_1 recipe, per seed
BOOT_TOL = 5e-6 + 1e-9
MRA_TOL = 5e-5 + 1e-9
# synthetic test only: replaces the published reproduction targets with ones computed from the mock, written into
# the record. The real run never sets this; on the real data the constants above are the committed published values.
TEST_REF = os.environ.get('S25_TEST_REF')
if TEST_REF:
    _tr = json.load(open(TEST_REF))
    REF_BOOT = _tr['boot']
    REF_MRA_DEV_RECIPE = tuple(_tr['mra_dev_recipe'])


def perm_path(split):
    return os.path.join(s15.RES, 'step24', 'permutation.json') if split == 'drug_split_1' \
        else os.path.join(s15.RES, 'step25', 'permutation_%s.json' % split)


def arm_name(ckarm, mode):
    """The run's arm label in its tag."""
    base = {'off': 'repro', 'shuf': 'fpshuf', 'zero': 'fpzero'}[mode]
    if ckarm == 'recipe':
        return {'repro': 'recipe_repro', 'fpshuf': 'fpshuf', 'fpzero': 'fpzero'}[base]
    return {'repro': 'base_repro', 'fpshuf': 'base_fpshuf', 'fpzero': 'base_fpzero'}[base]


def load_swap(split, ckarm, mode, seed, perm_sha, ck_sha):
    """One swap or reproduction run, checked against the provenance the launcher wrote. Split-1 recipe runs come from
    the committed Step 24 records (s24ev / s24_); all others from the Step 25 records (s25ev / s25_)."""
    arm = arm_name(ckarm, mode)
    if ckarm == 'recipe' and split == 'drug_split_1':
        s24arm = {'recipe_repro': 'recipe_repro', 'fpshuf': 'fpshuf', 'fpzero': 'fpzero'}[arm]
        tag, prov = 's24_%s_%s_seed_%d' % (split, s24arm, seed), 's24ev'
    else:
        tag, prov = 's25_%s_%s_seed_%d' % (split, arm, seed), 's25ev'
    f = os.path.join(s15.RES, 'vectors_%s.npz' % tag)
    fpc = os.path.join(s15.RES, 'vectors_%s_pc.npz' % tag)
    pj = os.path.join('provenance', '%s_%s.json' % (prov, tag))
    for q in (f, fpc, pj):
        if not os.path.isfile(q):
            die('missing %s' % q)
    pv = json.load(open(pj))
    nfe = 50 if ckarm == 'recipe' else None
    s15.check_prov(pv, tag, dict(status='completed', split=split, seed=seed, s24_mode=mode, checkpoint_arm=ckarm,
                                 drug_encoding='fingerprint', guidance_w=None, nfe=nfe, max_cells_per_cond=s15.CAP))
    if pv.get('checkpoint_sha256') != ck_sha:
        die('%s used checkpoint %s, the Step 15b %s eval of seed %d used %s'
            % (tag, str(pv.get('checkpoint_sha256'))[:12], ckarm, seed, str(ck_sha)[:12]))
    if mode == 'shuf' and pv.get('permutation_sha256') != perm_sha:
        die('%s used permutation %s, this split\'s committed record is %s'
            % (tag, pv.get('permutation_sha256'), perm_sha))
    if pv.get('vectors_sha256') != s15.sha(f) or pv.get('per_cell_sha256') != s15.sha(fpc):
        die('%s vectors or per-cell hash differs from its provenance' % tag)
    v = np.load(f, allow_pickle=True)
    p = np.load(fpc, allow_pickle=True)
    return dict(tag=tag, file=f, cond=np.array([str(x) for x in v['cond']]), true=np.asarray(v['true'], np.float64),
                B=np.asarray(v['B'], np.float64), C=np.asarray(v['C'], np.float64),
                pred=[np.asarray(x, np.float32) for x in p['pred']],
                true_cells=[np.asarray(x, np.float32) for x in p['true']],
                rec=dict(tag=tag, vectors_sha256=pv['vectors_sha256'], per_cell_sha256=pv['per_cell_sha256'],
                         checkpoint_sha256=pv['checkpoint_sha256'], permutation_sha256=pv.get('permutation_sha256'),
                         source='step24' if prov == 's24ev' else 'step25'))


def per_seed(v, Y, contexts, cand):
    D = v['B'] - v['C']
    j, d = s15.thesis_per_cond(v['true'], v['B'], v['C'])
    return dict(pds=s15c.pds_sets(D, Y, contexts, cand), de=j, dpc=d,
                en=np.array([s15.energy(v['pred'][i], v['true_cells'][i]) for i in range(len(D))]),
                size=np.linalg.norm(D, axis=1))


def main():
    if len(sys.argv) > 1:
        die('this script takes no options (prereg 2: all three splits, fixed arms)')
    for q in (OUT_JSON, OUT_PC, OUT_SUM):
        if os.path.exists(q):
            die('%s exists; refusing to overwrite a record' % q)
    if not os.path.isfile(PRE) or open(PRE).read().count('\n## ') < MIN_SECTIONS:
        die('pre-registration missing or incomplete')
    s23.committed_and_clean(
        [PRE, 'score_step25_tahoe.py', 'run_step25_score.sh', 'port_harness_base_step25.py',
         'leakage_test_base_step25.py', 'eval_step25_tahoe.sh', 'port_harness_step24.py',
         'make_step24_permutation.py', 'leakage_test_fm_step24.py', 'score_step23_screen.py',
         'score_step15_tahoe.py', 'score_step15c_tahoe.py', 'step14_metrics.py'], 'pre-registration or scripts')
    records = [PC15B, PC15C, s15.RECORD] + [perm_path(s) for s in SPLITS]
    if os.path.isfile(S23_PC):
        records.append(S23_PC)
    s23.committed_and_clean(records, 'records')
    commit = s23.git('rev-parse', 'HEAD').stdout.strip()
    say('[guards] pre-registration, scripts and records committed and clean; commit %s' % commit[:7])
    if TEST_REF:
        say('[WARNING] S25_TEST_REF set: reproduction targets read from %s (synthetic test only)' % TEST_REF)
    s15c.self_check()
    record = json.load(open(s15.RECORD))
    pc15b = pd.read_csv(PC15B)
    pc15c = pd.read_csv(PC15C)

    results, pc_rows, summary_rows = {}, [], []
    for split in SPLITS:
        k = split.split('_')[-1]
        held = sorted(record['splits'][k]['held_structures'])
        perm = json.load(open(perm_path(split)))
        if perm.get('split') != split or sorted(perm['structures']) != held \
                or any(perm['map'][x] == x for x in perm['structures']) \
                or sorted(perm['map'].values()) != sorted(perm['structures']):
            die('%s: the permutation is not a derangement of this split\'s held-out structures' % split)
        perm_sha = perm['sha256']

        # ---- inputs: own (Step 15b), both checkpoint arms. Per-cell arrays are large (~1 GB per run at 200 genes
        # ---- and 200 cells over ~2,900 conditions), so each run is checked and scored as it is loaded and its
        # ---- per-cell arrays are freed immediately; only the per-condition metrics and B are kept.
        own, cksha = {}, {}
        for ckarm in CKARMS:
            own[ckarm] = [s15.load_eval(split, ckarm, s) for s in range(s15.N_SEEDS)]
            cksha[ckarm] = [own[ckarm][s]['rec']['checkpoint_sha256'] for s in range(s15.N_SEEDS)]
        ref0 = own['recipe'][0]
        cond, n = ref0['cond'], len(ref0['cond'])
        if n != int(record['splits'][k]['n_test_conditions']):
            die('%s: %d conditions, the data record expects %d' % (split, n, record['splits'][k]['n_test_conditions']))

        # ---- reproduction gate per arm (prereg 7.3): off-run bitwise identical to Step 15b seed 0
        for ckarm in CKARMS:
            rep = load_swap(split, ckarm, 'off', 0, perm_sha, cksha[ckarm][0])
            base0 = own[ckarm][0]
            r0 = np.load(os.path.join(s15.RES, 'vectors_%s.npz' % base0['tag']), allow_pickle=True)
            rr = np.load(rep['file'], allow_pickle=True)
            if sorted(r0.files) != sorted(rr.files):
                die('%s %s reproduction stores %s, Step 15b stores %s'
                    % (split, ckarm, sorted(rr.files), sorted(r0.files)))
            for key in r0.files:
                a, b = (np.array([str(x) for x in r0[key]]), np.array([str(x) for x in rr[key]])) \
                    if key == 'cond' else (r0[key], rr[key])
                if a.dtype != b.dtype or not np.array_equal(a, b):
                    die('%s %s reproduction differs from Step 15b seed 0 in %s' % (split, ckarm, key))
            if not all(np.array_equal(rep['true_cells'][i], base0['true_cells'][i]) for i in range(n)) \
                    or not all(np.array_equal(rep['pred'][i], base0['pred'][i]) for i in range(n)):
                die('%s %s reproduction differs from Step 15b seed 0 in the stored real or predicted cells'
                    % (split, ckarm))
        say('[%s] reproduction: recipe and base off-runs are bitwise identical to Step 15b seed 0' % split)

        # ---- every own run shares the reference conditions and means
        for ckarm in CKARMS:
            for s in range(s15.N_SEEDS):
                o = own[ckarm][s]
                if not np.array_equal(o['cond'], cond) or np.abs(o['true'] - ref0['true']).max() > 1e-5 \
                        or np.abs(o['C'] - ref0['C']).max() > 1e-5:
                    die('%s %s seed %d differs from the recipe seed-0 conditions, true or C' % (split, ckarm, s))

        # ---- annotations, candidate sets (Step 15c path)
        pcb = pc15b[pc15b['split'] == split].reset_index(drop=True)
        pcc = pc15c[pc15c['split'] == split].reset_index(drop=True)
        if len(pcb) != n or not np.array_equal(pcb['condition'].astype(str).values, cond):
            die('%s: the Step 15b per-condition record is not in the vectors order' % split)
        if len(pcc) != n or not np.array_equal(pcc['condition'].astype(str).values, cond):
            die('%s: the Step 15c per-condition record is not in the vectors order' % split)
        lines, drugs = pcb['line'].astype(str).values, pcb['structure'].astype(str).values
        doses = pcb['dose_nM'].astype(float).values
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
        if not np.array_equal(np.array([len(cand[i]) for i in range(n)]), pcc['n_candidates'].values):
            die('%s: candidate-set sizes differ from the Step 15c record' % split)
        allidx = np.arange(n)
        boot = s23.Boot(allidx, drugs, n)

        # ---- per-seed measures. Own runs are already loaded; each swap run is loaded, checked against the own run
        # ---- of its arm and seed, scored, and then released before the next is read (peak memory ~2 runs).
        say('[%s] scoring %d evaluations on %d conditions in %d contexts' % (split, 6 * s15.N_SEEDS, n, len(contexts)))
        M, swap_rec = {}, {}
        for ckarm in CKARMS:
            M[(ckarm, 'own')] = [per_seed(v, Y, contexts, cand) for v in own[ckarm]]
            for name in ('shuf', 'zero'):
                M[(ckarm, name)], swap_rec['%s_%s' % (ckarm, name)] = [], []
                for s in range(s15.N_SEEDS):
                    w = load_swap(split, ckarm, dict(MODES)[name], s, perm_sha, cksha[ckarm][s])
                    o = own[ckarm][s]
                    if not np.array_equal(w['cond'], cond) or not np.array_equal(w['true'], o['true']) \
                            or not np.array_equal(w['C'], o['C']) \
                            or not all(np.array_equal(w['true_cells'][i], o['true_cells'][i]) for i in range(n)):
                        die('%s does not share the own run\'s conditions, means and real cells' % w['tag'])
                    same = np.all(w['B'] == o['B'], axis=1)
                    if same.any():
                        die('%s: the prediction equals the own prediction at a condition (e.g. %s)'
                            % (w['tag'], cond[np.flatnonzero(same)[0]]))
                    M[(ckarm, name)].append(per_seed(w, Y, contexts, cand))
                    swap_rec['%s_%s' % (ckarm, name)].append(w['rec'])
                    w['pred'] = w['true_cells'] = None        # release the per-cell arrays
                    del w
        say('[%s] every swap run shares the own conditions, means and real cells; the prediction differs everywhere'
            % split)
        sm = lambda key, metric: np.nanmean(np.vstack([m[metric] for m in M[key]]), axis=0)

        # ---- reproduction of the own records (prereg 7.7)
        for ckarm in CKARMS:
            checks = (('PDS', sm((ckarm, 'own'), 'pds'), pcc['%s_pds' % ckarm].values),
                      ('DE-overlap', sm((ckarm, 'own'), 'de'), pcb['%s_de_overlap' % ckarm].values),
                      ('DPC-50', sm((ckarm, 'own'), 'dpc'), pcb['%s_dpc50' % ckarm].values),
                      ('energy', sm((ckarm, 'own'), 'en'), pcb['%s_energy' % ckarm].values))
            for nm, mine, theirs in checks:
                if not s15c.close(mine, theirs):
                    die('%s %s own %s does not reproduce the committed per-condition record to 1e-9' % (split, ckarm, nm))
            pt, reps = boot.mean_reps(sm((ckarm, 'own'), 'pds') - 0.5)
            q = s23.interval(pt, reps)
            tgt = REF_BOOT[ckarm][split]
            if abs(q['diff'] - tgt[0]) > BOOT_TOL or abs(q['ci'][0] - tgt[1]) > BOOT_TOL or abs(q['ci'][1] - tgt[2]) > BOOT_TOL:
                die('%s %s bootstrap of PDS - 0.5 does not reproduce Step 15c %s, got %s'
                    % (split, ckarm, tgt, s23.fmt(q)))
            mra_seeds = [s23.mra_point(m['size'], tn, ctx_code, allidx) for m in M[(ckarm, 'own')]]
            mpt, _ = boot.mra_reps([[m['size'] for m in M[(ckarm, 'own')]]], tn, ctx_code)
            if abs(mpt[0] - float(np.mean(mra_seeds))) > 1e-10:
                die('%s %s vectorised MRA differs from the plain MRA' % (split, ckarm))
            if split == 'drug_split_1' and ckarm == 'recipe' \
                    and any(abs(a - b) > MRA_TOL for a, b in zip(mra_seeds, REF_MRA_DEV_RECIPE)):
                die('development recipe MRA does not reproduce Step 16 %s: %s' % (REF_MRA_DEV_RECIPE, mra_seeds))
            say('[%s] %s own reproduces PDS/DE/DPC/energy to 1e-9; bootstrap %s; MRA seeds %s'
                % (split, ckarm, s23.fmt(q), ' '.join('%.4f' % x for x in mra_seeds)))

        # ---- values and contrasts
        keys = (('pds', 'PDS'), ('de', 'DE-overlap'), ('dpc', 'DPC-50'), ('en', 'energy'))
        vals = {}
        for ckarm in CKARMS:
            for name, _ in MODES:
                v = {kk: float(np.nanmean(sm((ckarm, name), kk))) for kk, _ in keys}
                v['mra'] = float(np.mean([s23.mra_point(m['size'], tn, ctx_code, allidx) for m in M[(ckarm, name)]]))
                v['per_seed_pds'] = [float(np.nanmean(m['pds'])) for m in M[(ckarm, name)]]
                vals['%s_%s' % (ckarm, name)] = v
        con = {}
        half = np.full(n, 0.5)

        def contrast(name, xa, xb):
            diff = np.asarray(xa, np.float64) - np.asarray(xb, np.float64)
            ok = np.isfinite(diff)
            b = boot if ok.all() else s23.Boot(np.flatnonzero(ok), drugs, n)
            pt_, reps_ = b.mean_reps(diff)
            con[name] = s23.interval(pt_, reps_)
            con[name]['n_conditions'] = int(ok.sum())
            return con[name]

        say('\n' + '=' * 96 + '\n%s: %d conditions, %d structures\n' % (split, n, boot.G) + '=' * 96)
        say('  %-14s %8s %10s %8s %8s %8s' % ('arm', 'PDS', 'DE-overlap', 'DPC-50', 'energy', 'MRA'))
        for ckarm in CKARMS:
            for name, _ in MODES:
                v = vals['%s_%s' % (ckarm, name)]
                say('  %-14s %8.4f %10.4f %8.4f %8.4f %+8.4f'
                    % ('%s_%s' % (ckarm, name), v['pds'], v['de'], v['dpc'], v['en'], v['mra']))
        for ckarm in CKARMS:
            say('  %s   Q1 own - shuf   %s' % (ckarm, s23.fmt(contrast('%s_Q1_own_minus_shuf' % ckarm,
                sm((ckarm, 'own'), 'pds'), sm((ckarm, 'shuf'), 'pds')))))
            say('  %s   Q2 shuf - 0.5   %s' % (ckarm, s23.fmt(contrast('%s_Q2_shuf_minus_0.5' % ckarm,
                sm((ckarm, 'shuf'), 'pds'), half))))
            contrast('%s_ref_zero_minus_0.5' % ckarm, sm((ckarm, 'zero'), 'pds'), half)
            contrast('%s_ref_shuf_minus_zero' % ckarm, sm((ckarm, 'shuf'), 'pds'), sm((ckarm, 'zero'), 'pds'))
            contrast('%s_ref_own_minus_zero' % ckarm, sm((ckarm, 'own'), 'pds'), sm((ckarm, 'zero'), 'pds'))
        gap_own = sm(('recipe', 'own'), 'pds') - sm(('base', 'own'), 'pds')
        gap_shuf = sm(('recipe', 'shuf'), 'pds') - sm(('base', 'shuf'), 'pds')
        say('  Q3 (recipe-base) own - shuf   %s' % s23.fmt(contrast('Q3_gap_own_minus_shuf', gap_own, gap_shuf)))
        con['ref_gap_own'] = contrast('ref_gap_own', sm(('recipe', 'own'), 'pds'), sm(('base', 'own'), 'pds'))
        con['ref_gap_shuf'] = contrast('ref_gap_shuf', sm(('recipe', 'shuf'), 'pds'), sm(('base', 'shuf'), 'pds'))

        # ---- per-condition rows
        for i in range(n):
            r = dict(split=split, condition=cond[i], line=lines[i], structure=drugs[i], dose_nM=doses[i],
                     n_candidates=int(len(cand[i])), measured_size=tn[i])
            for ckarm in CKARMS:
                for name, _ in MODES:
                    for kk, _ in keys:
                        r['%s_%s_%s' % (ckarm, name, kk)] = sm((ckarm, name), kk)[i]
            pc_rows.append(r)
        for kk, c in con.items():
            summary_rows.append(dict(split=split, kind='contrast', name=kk, value=c['diff'],
                                     lo=c['ci'][0], hi=c['ci'][1], verdict=c['verdict'],
                                     n_conditions=c.get('n_conditions', n)))
        for aname, v in vals.items():
            for kk in ('pds', 'de', 'dpc', 'en', 'mra'):
                summary_rows.append(dict(split=split, kind='value', name='%s_%s' % (aname, kk), value=v[kk],
                                         lo='', hi='', verdict='', n_conditions=n))
        for ckarm in CKARMS:                                  # release this split's own per-cell arrays
            for o in own[ckarm]:
                o['pred'] = o['true_cells'] = None
        results[split] = dict(n_conditions=int(n), n_structures=int(boot.G), n_contexts=len(contexts),
                              permutation_sha256=perm_sha, values=vals, contrasts=con,
                              inputs=dict(own={a: [o['rec'] for o in own[a]] for a in CKARMS}, swap=swap_rec))

    # ---- verdicts across splits (prereg 5)
    conf = ('drug_split_0', 'drug_split_2')

    def q(split, name):
        return results[split]['contrasts'][name]['verdict']

    def verdict_null(ckarm):
        q1 = {s: q(s, '%s_Q1_own_minus_shuf' % ckarm) for s in conf}
        q2 = {s: q(s, '%s_Q2_shuf_minus_0.5' % ckarm) for s in conf}
        if any(q2[s] != 'above zero' for s in conf):
            return '4 UNINTERPRETABLE'
        repl = {s: (q1[s] != 'above zero') for s in conf}      # null replicated on a split = Q1 not above zero
        if all(repl.values()):
            return '1 NULL REPLICATED'
        if any(repl.values()):
            return '2 NULL PARTLY REPLICATED'
        return '3 NULL NOT REPLICATED'

    vA = verdict_null('recipe')
    vB = verdict_null('base')
    q3 = {s: q(s, 'Q3_gap_own_minus_shuf') for s in SPLITS}
    if all(v == 'above zero' for v in q3.values()):
        vC = 'GAP PARTLY COMPOUND-SPECIFIC (Q3 above zero on all three splits)'
    elif all(v == 'includes zero' for v in q3.values()):
        vC = 'GAP NOT COMPOUND-SPECIFIC (Q3 includes zero on all three splits)'
    else:
        vC = 'GAP MIXED (Q3: %s)' % ', '.join('%s %s' % (s.split('_')[-1], q3[s]) for s in SPLITS)
    say('\n' + '=' * 96)
    say('PRE-REGISTERED VERDICTS (section 5)')
    say('  A, recipe null on confirmation splits: %s' % vA)
    say('  B, base model:                         %s' % vB)
    say('  C, recipe-over-base gap:               %s' % vC)
    say('  development split is reported but not part of A/B replication (prereg 5).')
    say('=' * 96)

    text = json.dumps(dict(prereg=PRE, commit=commit, splits=list(SPLITS), n_boot=s23.N_BOOT,
                           interval='97.5% percentile, structure-clustered, fresh RandomState(0) per scored set',
                           reference=dict(boot=REF_BOOT, mra_dev_recipe=list(REF_MRA_DEV_RECIPE),
                                          source='step15c_findings.md s1/s3, step16_findings.md s4.4'),
                           verdict=dict(A_recipe_null=vA, B_base=vB, C_gap=vC, Q3_by_split=q3),
                           per_split=results, seconds=int(round(time.time() - T0))), indent=2, allow_nan=True)
    with open(OUT_SUM, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(summary_rows)
    with open(OUT_PC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(pc_rows)
    with open(OUT_JSON, 'w') as f:
        f.write(text)
    say('\nwrote %s, %s and %s  (%.0f s)' % (OUT_JSON, OUT_PC, OUT_SUM, time.time() - T0))


if __name__ == '__main__':
    main()
