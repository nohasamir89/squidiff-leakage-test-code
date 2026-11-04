#!/usr/bin/env python3
"""
score_step26_tahoe.py - Step 26: where does the compound-free discrimination come from?

Purpose
-------
Steps 24/25 showed that the recipe's PDS discrimination survives swapping or zeroing the compound fingerprint. This
scorer attributes that discrimination among the three pre-registered sources (PREREG_step26_control_source.md):
S1, the shared control estimate (the same control mean C is subtracted from the predicted and the measured change -
the bias of Nicol et al. 2026, bioRxiv 10.64898/2026.05.07.723486); S2, condition information the model carries from
its control-cell input; S3, compound signal that S1 noise masked. It re-scores the committed Step 15b/24/25
predictions under scoring schemes that remove the shared estimate (control splitting), adds no-model baselines and a
split-half duplicate ceiling (interpolated duplicate and dynamic range, Miller et al. 2025, bioRxiv
10.1101/2025.10.20.683304; positive/negative calibration baselines, Mejia et al. 2025, arXiv:2506.22641). CPU only;
no model is run; energy distance never touches C, so the committed energy values stand and are not recomputed.

Logic
-----
1. Guards (prereg 7.1): this file, the pre-registration, the imported scorers and every record read - including the
   Step 24/25 swap records and both findings files, so Step 25 must be scored and filed first - committed and
   identical to HEAD; no output exists.
2. Data rebuild gates (prereg 7.2), per split: the test and control h5ad files hash-match the Step 15 data record;
   the files are row-aligned (positional Group equality); the mean over ALL of a condition's test rows equals the
   stored `true` and the (line, plate) control-pool mean equals the stored `C`, each to 1e-4 with the maximum
   deviation printed (the pools are rebuilt from the per-condition record's line and plate columns, so this gate
   also validates that mapping); every stored real cell is recovered bitwise among its condition's test rows; pool
   sizes, flags (pool < 20 cells or stored cells < 4) and unmatched file groups are reported. A failure stops the
   step: it would mean the pre-registration's reading of the harness is wrong.
3. Reproduction (prereg 7.3): under S-orig this scorer's per-condition PDS / DE-overlap / DPC-50 equal the committed
   records to 1e-9 - own arms against Step 15c/15b, swap arms against the Step 24 (split 1) and Step 25 records -
   and the published Step 15c intervals for recipe and base PDS-0.5 are reproduced exactly by replaying Step 15c's
   bootstrap draw order with its own functions (score_step25_tahoe.replay_step15c; Step 25 prereg 7.7 amendment of
   2026-11-04 - Step 15c's one-generator-per-run convention makes its intervals depend on draw order, so a fresh
   generator reproduces only its first draw). No other scheme's number is read before this passes.
4. Schemes (prereg 3), all through the committed library functions on substituted change matrices (pds_sets takes
   change matrices directly; thesis_per_cond is called with a zero reference so its internal subtraction reproduces
   the scheme's changes exactly; MRA sizes are the norms of those changes):
     orig      D = B - C,  Y = true - C          (the published construction; the reproduction gate)
     split     D = B - C1, Y = true - C2         (C1/C2 = means of disjoint halves of each (line, plate) pool,
                                                  one RandomState(0) shuffle per pool; primary, Nicol)
     flip      D = B - C2, Y = true - C1         (the other half assignment; reference)
     linemean  D = B - CL, Y = true - CL         (CL = the line's pooled control mean across its plates; reference)
     none      D = B,      Y = true              (no subtraction; reference)
5. Arms (prereg 4): six model arms (own / fpshuf / fpzero x recipe / base, 3 seeds each, seed-averaged per
   condition); recon_recipe / recon_base (the stored A - the encoder shown the treated cells; reference);
   ctrl_sample (mean of n_i pool cells, RandomState(1000000+i), mirroring cfB's construction); ctrl_sample_h1
   (drawn from h1 only, RandomState(2000000+i); under S-split it shares no cells with C2 - the scheme's negative
   control); duplicate (stored real cells halved by RandomState(3000000+i); D from half 1, its own Y from half 2;
   the ceiling).
6. Contrasts and verdicts (prereg 5): G1-G5 gated on PDS, with base-model analogues and a flagged-conditions
   sensitivity line; verdict D (source), E (baseline attribution), F (calibration / dynamic-range fractions).
   Ungated: every other measure and scheme, per-dose, salt-twin (split 1, from the Step 24 record), within-plate
   candidate restriction, and the plate-structure report (distinct plates per candidate set and the fraction of
   candidate pairs sharing a plate - the facts that decide which sources are structurally possible).
Intervals: 97.5 % structure-clustered percentile bootstrap, 5,000 replicates, fresh RandomState(0) per scored set
(the Step 23 implementation, imported). Outputs (never overwritten): tahoe_results/step26_source.json,
step26_source_per_condition.csv, step26_source_summary.csv.

Run on a compute node from the worktree root through run_step26_score.sh, after Step 25 is scored and committed.
"""
import csv
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15        # noqa: E402  thesis_per_cond, dense, sha, load_eval, RES, RECORD
import score_step15c_tahoe as s15c      # noqa: E402  pds_sets, close, self_check, MIN_CAND
import score_step23_screen as s23       # noqa: E402  Boot, interval, fmt, mra_point, committed_and_clean, git
import score_step25_tahoe as s25        # noqa: E402  load_swap, perm_path, REF_BOOT, BOOT_TOL (checked loaders)

PRE = 'PREREG_step26_control_source.md'
SPLITS = s25.SPLITS                                      # development first
CKARMS = ('recipe', 'base')
SCHEMES = ('orig', 'split', 'flip', 'linemean', 'none')
MODEL_ARMS = ('recipe', 'recipe_fpshuf', 'recipe_fpzero', 'base', 'base_fpshuf', 'base_fpzero')
ALL_ARMS = MODEL_ARMS + ('recon_recipe', 'recon_base', 'ctrl_sample', 'ctrl_sample_h1', 'ctrl_line', 'duplicate')
DATA = os.environ.get('S15_DATA', '/scratch/ni2071/Diffusion/tahoe/datasets_step15')
PC15B, PC15C = s25.PC15B, s25.PC15C
S24_PC = os.path.join(s15.RES, 'step24_swap_per_condition.csv')
S25_PC = os.path.join(s15.RES, 'step25_swap_per_condition.csv')
S25_JSON = os.path.join(s15.RES, 'step25_swap.json')
OUT_JSON = os.path.join(s15.RES, 'step26_source.json')
OUT_PC = os.path.join(s15.RES, 'step26_source_per_condition.csv')
OUT_SUM = os.path.join(s15.RES, 'step26_source_summary.csv')
MIN_SECTIONS = 9
REBUILD_TOL = 1e-4                                       # prereg 7.2: float32 rebuild tolerance
FLAG_POOL, FLAG_CELLS = 20, 4                            # prereg 4: flag thresholds
SEED_CTRL, SEED_H1, SEED_DUP = 1000000, 2000000, 3000000  # prereg 4: pre-registered draw seeds (+ row index)
T0 = time.time()
say, die = s23.say, s23.die

# Published reproduction targets (prereg 7.3) - the same constants Step 25 used (its prereg section 7.7). On a
# synthetic test S26_TEST_REF replaces them with targets computed from the mock; the real run never sets it.
REF_BOOT = s25.REF_BOOT
BOOT_TOL = s25.BOOT_TOL
TEST_REF = os.environ.get('S26_TEST_REF')
if TEST_REF:
    REF_BOOT = json.load(open(TEST_REF))['boot']


def find_file_sha(obj, basename):
    """Search the Step 15 data record recursively for a sha256 recorded for this data file: either a dict holding
    the file name in one value and a 64-hex string in another, or a direct {basename: sha} entry."""
    hex64 = lambda s: isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s.lower())
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) == basename and hex64(v):
                return v
        if any(isinstance(v, str) and basename in v for v in obj.values()):
            for k, v in obj.items():
                if hex64(v) and ('sha' in str(k).lower() or 'hash' in str(k).lower()):
                    return v
        for v in obj.values():
            r = find_file_sha(v, basename)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_file_sha(v, basename)
            if r:
                return r
    return None


def load_A(run):
    """The stored reconstruction arm (encoder shown the treated cells) of an own run whose vectors file the checked
    loader already verified against its provenance hash."""
    f = os.path.join(s15.RES, 'vectors_%s.npz' % run['tag'])
    return np.asarray(np.load(f, allow_pickle=True)['A'], np.float64)


def metrics(D, Y, contexts, cand, Z):
    """One arm under one scheme, through the committed library functions only. thesis_per_cond(y, d, 0) computes
    y - 0 and d - 0 internally, so it scores exactly the scheme's change vectors."""
    j, d = s15.thesis_per_cond(Y, D, Z)
    return dict(pds=s15c.pds_sets(D, Y, contexts, cand), de=j, dpc=d, size=np.linalg.norm(D, axis=1))


def main():
    if len(sys.argv) > 1:
        die('this script takes no options (prereg 2: all three splits, fixed arms and schemes)')
    for q in (OUT_JSON, OUT_PC, OUT_SUM):
        if os.path.exists(q):
            die('%s exists; refusing to overwrite a record' % q)
    if not os.path.isfile(PRE) or open(PRE).read().count('\n## ') < MIN_SECTIONS:
        die('pre-registration missing or incomplete')
    # ---- prereg 7.1 and 8.2: everything read is committed and clean; Step 25 scored and filed first
    s23.committed_and_clean(
        [PRE, 'score_step26_tahoe.py', 'run_step26_score.sh', 'PREREG_step25_swap_confirmation.md',
         'score_step25_tahoe.py', 'score_step23_screen.py', 'score_step15_tahoe.py', 'score_step15c_tahoe.py',
         'step14_metrics.py'], 'pre-registration or scripts')
    records = [PC15B, PC15C, s15.RECORD, S24_PC, S25_PC, S25_JSON, 'step24_findings.md', 'step25_findings.md'] \
        + [s25.perm_path(s) for s in SPLITS]
    s23.committed_and_clean(records, 'records (Step 25 must be scored and its findings committed first)')
    commit = s23.git('rev-parse', 'HEAD').stdout.strip()
    say('[guards] pre-registration, scripts and records committed and clean; commit %s' % commit[:7])
    if TEST_REF:
        say('[WARNING] S26_TEST_REF set: reproduction targets read from %s (synthetic test only)' % TEST_REF)
    s15c.self_check()
    import anndata                                           # noqa: E402  only the data gates need it
    record = json.load(open(s15.RECORD))
    pc15b, pc15c = pd.read_csv(PC15B), pd.read_csv(PC15C)
    pc24 = pd.read_csv(S24_PC)
    pc25 = pd.read_csv(S25_PC)

    # Step 15c's single bootstrap generator at its run-start state; s25.replay_step15c advances it once per split in
    # Step 15c's split order (SPLITS == s15.ALL_SPLITS). Nothing else in this scorer touches s15.rng.
    if tuple(SPLITS) != tuple(s15.ALL_SPLITS):
        die('SPLITS must equal Step 15c\'s split order for the interval replay')
    s15.rng = np.random.RandomState(0)

    results, pc_rows, summary_rows = {}, [], []
    gate_verdicts = {}                                       # split -> {G1..G5, base analogues} verdict strings
    for split in SPLITS:
        k = split.split('_')[-1]
        perm = json.load(open(s25.perm_path(split)))
        perm_sha = perm['sha256']

        # ---- own runs (checked loaders); only ref0's per-cell arrays are kept
        own = {a: [s15.load_eval(split, a, s) for s in range(s15.N_SEEDS)] for a in CKARMS}
        cksha = {a: [own[a][s]['rec']['checkpoint_sha256'] for s in range(s15.N_SEEDS)] for a in CKARMS}
        ref0 = own['recipe'][0]
        cond, n = ref0['cond'], len(ref0['cond'])
        if n != int(record['splits'][k]['n_test_conditions']):
            die('%s: %d conditions, the data record expects %d' % (split, n, record['splits'][k]['n_test_conditions']))
        for a in CKARMS:
            for s in range(s15.N_SEEDS):
                o = own[a][s]
                if not np.array_equal(o['cond'], cond) or np.abs(o['true'] - ref0['true']).max() > 1e-5 \
                        or np.abs(o['C'] - ref0['C']).max() > 1e-5 \
                        or not all(np.array_equal(o['true_cells'][i], ref0['true_cells'][i]) for i in range(n)):
                    die('%s %s seed %d differs from recipe seed 0 in conditions, true, C or stored real cells'
                        % (split, a, s))

        # ---- per-condition records: order, annotations, candidate sets (the Step 15c path, as Step 25 built them)
        pcb = pc15b[pc15b['split'] == split].reset_index(drop=True)
        pcc = pc15c[pc15c['split'] == split].reset_index(drop=True)
        p25 = pc25[pc25['split'] == split].reset_index(drop=True)
        for nm, df in (('Step 15b', pcb), ('Step 15c', pcc), ('Step 25', p25)):
            if len(df) != n or not np.array_equal(df['condition'].astype(str).values, cond):
                die('%s: the %s per-condition record is not in the vectors order' % (split, nm))
        lines = pcb['line'].astype(str).values
        drugs = pcb['structure'].astype(str).values
        doses = pcb['dose_nM'].astype(float).values
        plates = pcc['plate'].astype(str).values
        ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
        ctx_code = np.unique(ctx_key, return_inverse=True)[1]
        contexts = [np.where(ctx_key == u)[0] for u in np.unique(ctx_key)]
        cand = {}
        for members in contexts:
            for p_ in members:
                cand[p_] = np.array([q for q in members if q == p_ or drugs[q] != drugs[p_]])
        if not np.array_equal(np.array([len(cand[i]) for i in range(n)]), pcc['n_candidates'].values):
            die('%s: candidate-set sizes differ from the Step 15c record' % split)
        allidx = np.arange(n)
        boot = s23.Boot(allidx, drugs, n)
        Z = np.zeros((n, ref0['true'].shape[1]))

        # ---- plate-structure report (prereg 3): which sources are structurally possible
        plates_per_ctx = np.array([len(set(plates[m])) for m in contexts])
        share = np.array([np.mean(plates[cand[i][cand[i] != i]] == plates[i]) if len(cand[i]) > 1 else np.nan
                          for i in range(n)])
        say('[%s] plate structure: %d contexts, distinct plates per context min/median/max %d/%d/%d; '
            'fraction of candidate pairs sharing the condition\'s plate %.4f'
            % (split, len(contexts), plates_per_ctx.min(), int(np.median(plates_per_ctx)), plates_per_ctx.max(),
               float(np.nanmean(share))))

        # ---- data files: hashes against the Step 15 data record, row alignment, rebuild gates (prereg 7.2)
        tf = os.path.join(DATA, 'tahoe_test_%s.h5ad' % split)
        cf = os.path.join(DATA, 'tahoe_test_%s_control.h5ad' % split)
        for q in (tf, cf):
            if not os.path.isfile(q):
                die('missing data file %s' % q)
            want = find_file_sha(record, os.path.basename(q))
            if want is None:
                die('the Step 15 data record holds no sha256 for %s; amend the pre-registration before running'
                    % os.path.basename(q))
            got = s15.sha(q)
            if got != want:
                die('%s sha256 %s differs from the data record %s' % (q, got[:12], want[:12]))
        te, co = anndata.read_h5ad(tf), anndata.read_h5ad(cf)
        for nm, a_ in (('test', te), ('control', co)):
            if 'Group' not in a_.obs:
                die('%s file has no Group column (columns: %s)' % (nm, list(a_.obs.columns)))
        tg = te.obs['Group'].astype(str).values
        cg = co.obs['Group'].astype(str).values
        if len(tg) != len(cg) or not np.array_equal(tg, cg):
            die('%s: test and control files are not row-aligned (positional Group equality fails)' % split)
        tX = s15.dense(te.X).astype(np.float32)
        cX = s15.dense(co.X).astype(np.float32)
        if tX.shape[1] != ref0['true'].shape[1] or cX.shape[1] != ref0['true'].shape[1]:
            die('%s: data files carry %d/%d genes, the stored vectors %d'
                % (split, tX.shape[1], cX.shape[1], ref0['true'].shape[1]))
        del te, co
        # condition -> file rows (grouped once; per-condition masks over ~1e6 rows would be quadratic)
        order = np.argsort(tg, kind='stable')
        sorted_tg = tg[order]
        starts = np.searchsorted(sorted_tg, cond, side='left')
        ends = np.searchsorted(sorted_tg, cond, side='right')
        if (ends <= starts).any():
            die('%s: %d conditions have no rows in the test file (e.g. %s)'
                % (split, int((ends <= starts).sum()), cond[np.flatnonzero(ends <= starts)[0]]))
        rows = [np.sort(order[starts[i]:ends[i]]) for i in range(n)]
        n_unmatched_rows = len(tg) - int((ends - starts).sum())
        say('[%s] data: %d test rows, %d in the %d scored conditions, %d in unscored groups'
            % (split, len(tg), int((ends - starts).sum()), n, n_unmatched_rows))

        # ---- pools per (line, plate), from the per-condition record's own line/plate columns
        lp_of = [(lines[i], plates[i]) for i in range(n)]
        pool_idx = {}
        for i in range(n):
            pool_idx.setdefault(lp_of[i], []).append(rows[i])
        pool_idx = {lp: np.sort(np.concatenate(v)) for lp, v in pool_idx.items()}
        pool_size = np.array([len(pool_idx[lp_of[i]]) for i in range(n)])
        n_cells = np.array([len(ref0['true_cells'][i]) for i in range(n)])
        flagged = (pool_size < FLAG_POOL) | (n_cells < FLAG_CELLS)
        say('[%s] %d (line, plate) pools, size min/median/max %d/%d/%d; %d conditions flagged '
            '(pool < %d or stored cells < %d)'
            % (split, len(pool_idx), int(pool_size.min()), int(np.median(pool_size)), int(pool_size.max()),
               int(flagged.sum()), FLAG_POOL, FLAG_CELLS))

        # ---- rebuild gates: stored true from the test rows, stored C from the pools, stored cells recovered
        dev_t = max(float(np.abs(tX[rows[i]].mean(0, dtype=np.float64) - ref0['true'][i]).max()) for i in range(n))
        if dev_t > REBUILD_TOL:
            die('%s: rebuilt per-condition means deviate from stored true by %.2e (> %.0e)' % (split, dev_t, REBUILD_TOL))
        cmean = {lp: cX[idx].mean(0, dtype=np.float64) for lp, idx in pool_idx.items()}
        dev_c = max(float(np.abs(cmean[lp_of[i]] - ref0['C'][i]).max()) for i in range(n))
        if dev_c > REBUILD_TOL:
            die('%s: rebuilt (line, plate) pool means deviate from stored C by %.2e (> %.0e)' % (split, dev_c, REBUILD_TOL))
        miss = 0
        for i in range(n):
            have = {r.tobytes() for r in tX[rows[i]]}
            miss += sum(1 for c_ in ref0['true_cells'][i] if np.asarray(c_, np.float32).tobytes() not in have)
        if miss:
            die('%s: %d stored real cells are not rows of their condition in the test file' % (split, miss))
        say('[%s] rebuild gates: stored true and C reproduced from the data files '
            '(max deviation %.2e and %.2e); all stored real cells recovered bitwise' % (split, dev_t, dev_c))
        del tX, order, sorted_tg
        gc.collect()

        # ---- scheme references (prereg 3): one RandomState(0) shuffle per pool; halves disjoint and exhaustive
        C1p, C2p, h1p = {}, {}, {}
        for lp, idx in pool_idx.items():
            pm = np.random.RandomState(0).permutation(len(idx))
            a_, b_ = idx[pm[:(len(idx) + 1) // 2]], idx[pm[(len(idx) + 1) // 2:]]
            if len(b_) == 0:                                  # a 1-cell pool cannot be split
                C1p[lp], C2p[lp], h1p[lp] = cX[a_].mean(0, dtype=np.float64), np.full(cX.shape[1], np.nan), a_
                continue
            if np.intersect1d(a_, b_).size or len(a_) + len(b_) != len(idx):
                die('pool halving is not a disjoint exhaustive split (%s)' % (lp,))
            C1p[lp], C2p[lp], h1p[lp] = cX[a_].mean(0, dtype=np.float64), cX[b_].mean(0, dtype=np.float64), a_
        C = ref0['C']
        C1 = np.vstack([C1p[lp_of[i]] for i in range(n)])
        C2 = np.vstack([C2p[lp_of[i]] for i in range(n)])
        line_rows = {}
        for i in range(n):
            line_rows.setdefault(lines[i], set()).add(lp_of[i])
        lmean = {ln: cX[np.sort(np.concatenate([pool_idx[lp] for lp in lps]))].mean(0, dtype=np.float64)
                 for ln, lps in line_rows.items()}
        CL = np.vstack([lmean[lines[i]] for i in range(n)])
        REFS = {'orig': (C, C), 'split': (C1, C2), 'flip': (C2, C1), 'linemean': (CL, CL), 'none': (Z, Z)}

        # ---- no-model baselines and the duplicate ceiling (prereg 4): single pre-registered realizations
        P_ctrl, P_h1, D1h, D2h = np.empty((n, Z.shape[1])), np.empty((n, Z.shape[1])), \
            np.empty((n, Z.shape[1])), np.empty((n, Z.shape[1]))
        for i in range(n):
            idx = pool_idx[lp_of[i]]
            t1 = np.random.RandomState(SEED_CTRL + i).choice(len(idx), n_cells[i], replace=(len(idx) < n_cells[i]))
            P_ctrl[i] = cX[idx[t1]].mean(0, dtype=np.float64)
            h1 = h1p[lp_of[i]]
            t2 = np.random.RandomState(SEED_H1 + i).choice(len(h1), n_cells[i], replace=(len(h1) < n_cells[i]))
            P_h1[i] = cX[h1[t2]].mean(0, dtype=np.float64)
            cells = np.asarray(ref0['true_cells'][i], np.float64)
            pm = np.random.RandomState(SEED_DUP + i).permutation(len(cells))
            D1h[i] = cells[pm[:(len(cells) + 1) // 2]].mean(0)
            D2h[i] = cells[pm[(len(cells) + 1) // 2:]].mean(0) if len(cells) > 1 else np.nan
        del cX
        gc.collect()

        # ---- swap runs through the Step 25 checked loader; per-cell arrays released immediately
        B = {('recipe', 'own'): [o['B'] for o in own['recipe']], ('base', 'own'): [o['B'] for o in own['base']]}
        A_arm = {a: [load_A(o) for o in own[a]] for a in CKARMS}
        swap_rec = {}
        for a in CKARMS:
            for mode in ('shuf', 'zero'):
                B[(a, mode)], swap_rec['%s_%s' % (a, mode)] = [], []
                for s in range(s15.N_SEEDS):
                    w = s25.load_swap(split, a, mode, s, perm_sha, cksha[a][s])
                    if not np.array_equal(w['cond'], cond) or np.abs(w['true'] - ref0['true']).max() > 1e-5 \
                            or np.abs(w['C'] - ref0['C']).max() > 1e-5:
                        die('%s does not share the own run\'s conditions and means' % w['tag'])
                    B[(a, mode)].append(w['B'])
                    swap_rec['%s_%s' % (a, mode)].append(w['rec'])
                    w['pred'] = w['true_cells'] = None
                    del w
        for a in CKARMS:                                      # the own per-cell arrays are no longer needed either,
            for s in range(s15.N_SEEDS):                      # except ref0's, kept for the record of n_cells
                if own[a][s] is not ref0:
                    own[a][s]['pred'] = own[a][s]['true_cells'] = None
        gc.collect()
        armkey = {'recipe': ('recipe', 'own'), 'recipe_fpshuf': ('recipe', 'shuf'), 'recipe_fpzero': ('recipe', 'zero'),
                  'base': ('base', 'own'), 'base_fpshuf': ('base', 'shuf'), 'base_fpzero': ('base', 'zero')}
        for a, mode in B:                                     # prereg 7.4: arms differ at every condition
            if mode == 'own':
                continue
            for s in range(s15.N_SEEDS):
                same = np.all(B[(a, mode)][s] == B[(a, 'own')][s], axis=1)
                if same.any():
                    die('%s %s %s seed %d equals the own prediction at a condition' % (split, a, mode, s))

        # ---- score every arm under every scheme (prereg 4-5)
        say('[%s] scoring %d arms x %d schemes on %d conditions' % (split, len(ALL_ARMS), len(SCHEMES), n))
        M = {}
        for sch in SCHEMES:
            R1, R2 = REFS[sch]
            Y = ref0['true'] - R2
            for arm in MODEL_ARMS:
                a, mode = armkey[arm]
                if sch == 'orig':                             # exactly the committed construction, per run
                    M[(arm, sch)] = [metrics((own[a][s]['B'] if mode == 'own' else B[(a, mode)][s]) -
                                             (own[a][s]['C'] if mode == 'own' else C), ref0['true'] - C,
                                             contexts, cand, Z) for s in range(s15.N_SEEDS)]
                else:
                    M[(arm, sch)] = [metrics(B[(a, mode)][s] - R1, Y, contexts, cand, Z)
                                     for s in range(s15.N_SEEDS)]
            for a in CKARMS:
                M[('recon_%s' % a, sch)] = [metrics(A_arm[a][s] - R1, Y, contexts, cand, Z)
                                            for s in range(s15.N_SEEDS)]
            M[('ctrl_sample', sch)] = [metrics(P_ctrl - R1, Y, contexts, cand, Z)]
            M[('ctrl_sample_h1', sch)] = [metrics(P_h1 - R1, Y, contexts, cand, Z)]
            # ctrl_line (amendment 2026-11-02): the line's pooled control mean as a prediction - the S1 probe,
            # out-of-pool by 1 - N_pool/N_line, so it carries the shared estimate where a pool resample cannot
            M[('ctrl_line', sch)] = [metrics(CL - R1, Y, contexts, cand, Z)]
            M[('duplicate', sch)] = [metrics(D1h - R1, D2h - R2, contexts, cand, Z)]
        sm = lambda arm, sch, kk: np.nanmean(np.vstack([m[kk] for m in M[(arm, sch)]]), axis=0)

        # ---- reproduction gate (prereg 7.3): S-orig equals the committed records to 1e-9
        for arm, src, col in (('recipe', pcc, 'recipe_pds'), ('base', pcc, 'base_pds')):
            if not s15c.close(sm(arm, 'orig', 'pds'), src[col].values):
                die('%s: %s S-orig PDS does not reproduce the committed Step 15c record to 1e-9' % (split, arm))
        for arm in ('recipe', 'base'):
            for kk, col in (('de', '%s_de_overlap' % arm), ('dpc', '%s_dpc50' % arm)):
                if not s15c.close(sm(arm, 'orig', kk), pcb[col].values):
                    die('%s: %s S-orig %s does not reproduce the committed Step 15b record to 1e-9' % (split, arm, kk))
        for arm in MODEL_ARMS:
            a, mode = armkey[arm]
            if mode == 'own':
                continue
            s25col = '%s_%s' % (a, mode)
            for kk in ('pds', 'de', 'dpc'):
                if not s15c.close(sm(arm, 'orig', kk), p25['%s_%s' % (s25col, kk)].values):
                    die('%s: %s S-orig %s does not reproduce the committed Step 25 record to 1e-9' % (split, arm, kk))
        if split == 'drug_split_1':
            p24 = pc24.reset_index(drop=True)
            if len(p24) != n or not np.array_equal(p24['condition'].astype(str).values, cond):
                die('the Step 24 per-condition record is not in the vectors order')
            for arm, pre in (('recipe_fpshuf', 'fpshuf'), ('recipe_fpzero', 'fpzero')):
                for kk in ('pds', 'de', 'dpc'):
                    if not s15c.close(sm(arm, 'orig', kk), p24['%s_%s' % (pre, kk)].values):
                        die('%s S-orig %s does not reproduce the committed Step 24 record to 1e-9' % (arm, kk))
        rep15c = s25.replay_step15c(pcc, pcc['structure'].astype(str).values)   # once per split, in order
        for a in CKARMS:
            got, tgt = rep15c[a], REF_BOOT[a][split]
            if any(abs(x - y) > BOOT_TOL for x, y in zip(got, tgt)):
                die('%s %s replay of Step 15c\'s bootstrap draw order does not reproduce its interval %s, got '
                    '%+.5f [%+.5f, %+.5f]' % (split, a, tgt, *got))
        say('[%s] reproduction: S-orig equals the committed Step 15b/15c/24/25 per-condition records to 1e-9; '
            'Step 15c intervals replayed exactly (recipe %+.5f [%+.5f, %+.5f], base %+.5f [%+.5f, %+.5f])'
            % (split, *rep15c['recipe'], *rep15c['base']))

        # ---- within-plate candidate restriction (prereg 3; S1 structurally absent inside a plate)
        cand_pl = {}
        for i in range(n):
            cp = np.array([q for q in cand[i] if q == i or plates[q] == plates[i]])
            if len(cp) >= s15c.MIN_CAND:
                cand_pl[i] = cp
        wp = {}
        for sch in ('orig', 'split'):
            R1, R2 = REFS[sch]
            Y = ref0['true'] - R2
            for arm in MODEL_ARMS:
                a, mode = armkey[arm]
                vv = [s15c.pds_sets((own[a][s]['B'] - own[a][s]['C']) if (mode == 'own' and sch == 'orig')
                                    else ((own[a][s]['B'] if mode == 'own' else B[(a, mode)][s]) - R1),
                                    (ref0['true'] - C) if sch == 'orig' else Y, contexts, cand_pl)
                      for s in range(s15.N_SEEDS)]
                v = np.nanmean(np.vstack(vv), axis=0)
                wp['%s_%s' % (arm, sch)] = dict(pds=float(np.nanmean(v)), n=int(np.isfinite(v).sum()))
        say('[%s] within-plate candidates: %d of %d conditions covered (sets >= %d); recipe orig %.4f split %.4f'
            % (split, wp['recipe_orig']['n'], n, s15c.MIN_CAND, wp['recipe_orig']['pds'], wp['recipe_split']['pds']))

        # ---- values, contrasts, verdicts (prereg 5)
        vals = {}
        for arm in ALL_ARMS:
            for sch in SCHEMES:
                v = dict(pds=float(np.nanmean(sm(arm, sch, 'pds'))), de=float(np.nanmean(sm(arm, sch, 'de'))),
                         dpc=float(np.nanmean(sm(arm, sch, 'dpc'))))
                v['mra'] = float(np.mean([s23.mra_point(m['size'], np.linalg.norm(ref0['true'] - REFS[sch][1], axis=1),
                                                        ctx_code, allidx)
                                          for m in M[(arm, sch)]]))
                vals['%s__%s' % (arm, sch)] = v
        con = {}
        half = np.full(n, 0.5)

        def contrast(name, xa, xb, sens=False):
            diff = np.asarray(xa, np.float64) - np.asarray(xb, np.float64)
            if sens:
                diff = diff.copy()
                diff[flagged] = np.nan
            ok = np.isfinite(diff)
            b = boot if ok.all() else s23.Boot(np.flatnonzero(ok), drugs, n)
            pt_, reps_ = b.mean_reps(diff)
            con[name] = s23.interval(pt_, reps_)
            con[name]['n_conditions'] = int(ok.sum())
            return con[name]

        say('\n' + '=' * 110 + '\n%s: %d conditions, %d structures, %d flagged\n' % (split, n, boot.G, int(flagged.sum()))
            + '=' * 110)
        say('  %-18s %s' % ('arm', '  '.join('%12s' % sch for sch in SCHEMES)) + '   (PDS per scheme)')
        for arm in ALL_ARMS:
            say('  %-18s %s' % (arm, '  '.join('%12.4f' % vals['%s__%s' % (arm, sch)]['pds'] for sch in SCHEMES)))
        gates = (('G1_recipe_split_minus_0.5', sm('recipe', 'split', 'pds'), half),
                 ('G1f_recipe_flip_minus_0.5', sm('recipe', 'flip', 'pds'), half),
                 ('G2_fpshuf_split_minus_0.5', sm('recipe_fpshuf', 'split', 'pds'), half),
                 ('G2f_fpshuf_flip_minus_0.5', sm('recipe_fpshuf', 'flip', 'pds'), half),
                 ('G3_split_recipe_minus_fpshuf', sm('recipe', 'split', 'pds'), sm('recipe_fpshuf', 'split', 'pds')),
                 ('G3f_flip_recipe_minus_fpshuf', sm('recipe', 'flip', 'pds'), sm('recipe_fpshuf', 'flip', 'pds')),
                 ('G4_ctrl_line_orig_minus_0.5', sm('ctrl_line', 'orig', 'pds'), half),
                 ('G4b_ctrl_sample_orig_minus_0.5', sm('ctrl_sample', 'orig', 'pds'), half),
                 ('G5_ctrl_h1_split_minus_0.5', sm('ctrl_sample_h1', 'split', 'pds'), half),
                 ('base_G1_split_minus_0.5', sm('base', 'split', 'pds'), half),
                 ('base_G1f_flip_minus_0.5', sm('base', 'flip', 'pds'), half),
                 ('base_G2_split_minus_0.5', sm('base_fpshuf', 'split', 'pds'), half),
                 ('base_G3_split_own_minus_fpshuf', sm('base', 'split', 'pds'), sm('base_fpshuf', 'split', 'pds')),
                 ('base_G3f_flip_own_minus_fpshuf', sm('base', 'flip', 'pds'), sm('base_fpshuf', 'flip', 'pds')),
                 ('E_orig_recipe_minus_ctrl_line', sm('recipe', 'orig', 'pds'), sm('ctrl_line', 'orig', 'pds')),
                 ('E_orig_recipe_minus_ctrl_sample', sm('recipe', 'orig', 'pds'), sm('ctrl_sample', 'orig', 'pds')))
        for nm, xa, xb in gates:
            say('  %-34s %s' % (nm, s23.fmt(contrast(nm, xa, xb))))
            if nm.startswith(('G1', 'G2', 'G3', 'G4', 'G5')):
                contrast(nm + '_sens', xa, xb, sens=True)
        gate_verdicts[split] = {nm: con[nm]['verdict'] for nm, _, _ in gates}

        # ---- verdicts E and F inputs (prereg 5)
        # the fraction is only meaningful against a real published effect; near zero it is reported as undefined.
        # Amendment 2026-11-02: the attribution baseline is ctrl_line (the S1 probe); the ctrl_sample fraction is
        # reported alongside as the input-copying null.
        denom = vals['recipe__orig']['pds'] - 0.5
        e_frac = (vals['ctrl_line__orig']['pds'] - 0.5) / denom if abs(denom) > 0.02 else None
        e_frac_sample = (vals['ctrl_sample__orig']['pds'] - 0.5) / denom if abs(denom) > 0.02 else None
        drf = {}
        for sch in SCHEMES:
            lo_, hi_ = vals['ctrl_sample__%s' % sch]['pds'], vals['duplicate__%s' % sch]['pds']
            for arm in MODEL_ARMS:
                drf['%s__%s' % (arm, sch)] = (vals['%s__%s' % (arm, sch)]['pds'] - lo_) / (hi_ - lo_) \
                    if abs(hi_ - lo_) > 1e-12 else None
        say('  verdict E: ctrl_line reproduces %s of the published recipe effect (paired %s); '
            'ctrl_sample (copying null) %s'
            % ('%.1f%%' % (100 * e_frac) if e_frac is not None else 'n/a (recipe S-orig effect < 0.02)',
               s23.fmt(con['E_orig_recipe_minus_ctrl_line']),
               '%.1f%%' % (100 * e_frac_sample) if e_frac_sample is not None else 'n/a'))
        say('  verdict F: duplicate ceiling orig %.4f split %.4f; recipe dynamic-range fraction orig %s split %s'
            % (vals['duplicate__orig']['pds'], vals['duplicate__split']['pds'],
               '%.3f' % drf['recipe__orig'] if drf['recipe__orig'] is not None else 'n/a',
               '%.3f' % drf['recipe__split'] if drf['recipe__split'] is not None else 'n/a'))

        # ---- ungated breakdowns: per dose, salt twins (split 1 only)
        per_dose = {}
        for arm in ('recipe', 'recipe_fpshuf', 'ctrl_sample'):
            for sch in ('orig', 'split'):
                per_dose['%s__%s' % (arm, sch)] = {'%g' % d: float(np.nanmean(sm(arm, sch, 'pds')[np.isclose(doses, d)]))
                                                   for d in sorted(set(doses))}
        twins = None
        if split == 'drug_split_1' and 'twin' in pc24.columns:
            tw = pc24['twin'].values.astype(bool)
            twins = {arm: dict(twin=float(np.nanmean(sm(arm, 'split', 'pds')[tw])),
                               non_twin=float(np.nanmean(sm(arm, 'split', 'pds')[~tw])), n_twin=int(tw.sum()))
                     for arm in ('recipe', 'recipe_fpshuf')}

        # ---- per-condition rows
        for i in range(n):
            r = dict(split=split, condition=cond[i], line=lines[i], plate=plates[i], structure=drugs[i],
                     dose_nM=doses[i], n_candidates=int(len(cand[i])), pool_size=int(pool_size[i]),
                     n_stored_cells=int(n_cells[i]), flagged=int(flagged[i]))
            for arm in ALL_ARMS:
                for sch in SCHEMES:
                    for kk in ('pds', 'de', 'dpc'):
                        r['%s__%s__%s' % (arm, sch, kk)] = sm(arm, sch, kk)[i]
            pc_rows.append(r)
        for kk, c_ in con.items():
            summary_rows.append(dict(split=split, kind='contrast', name=kk, value=c_['diff'], lo=c_['ci'][0],
                                     hi=c_['ci'][1], verdict=c_['verdict'], n_conditions=c_['n_conditions']))
        for aname, v in vals.items():
            for kk in ('pds', 'de', 'dpc', 'mra'):
                summary_rows.append(dict(split=split, kind='value', name='%s_%s' % (aname, kk), value=v[kk],
                                         lo='', hi='', verdict='', n_conditions=n))
        results[split] = dict(
            n_conditions=int(n), n_structures=int(boot.G), n_contexts=len(contexts),
            plate_structure=dict(plates_per_context_min=int(plates_per_ctx.min()),
                                 plates_per_context_median=int(np.median(plates_per_ctx)),
                                 plates_per_context_max=int(plates_per_ctx.max()),
                                 candidate_pair_same_plate_fraction=float(np.nanmean(share))),
            pools=dict(n_pools=len(pool_idx), size_min=int(pool_size.min()), size_median=int(np.median(pool_size)),
                       size_max=int(pool_size.max()), n_flagged=int(flagged.sum()),
                       unmatched_file_rows=n_unmatched_rows),
            rebuild=dict(true_max_dev=dev_t, C_max_dev=dev_c),
            data_files={os.path.basename(tf): s15.sha(tf), os.path.basename(cf): s15.sha(cf)},
            values=vals, contrasts=con, within_plate=wp, per_dose=per_dose, twins=twins,
            verdict_E_fraction=e_frac, verdict_E_fraction_copying_null=e_frac_sample, dynamic_range_fraction=drf,
            inputs=dict(own={a: [o['rec'] for o in own[a]] for a in CKARMS}, swap=swap_rec))
        for a in CKARMS:
            for o in own[a]:
                o['pred'] = o['true_cells'] = None
        # release this split's large objects by rebinding, not `del`: the sm lambda closes over M, and deleting a
        # name a live closure refers to is an unbound-name hazard (flagged by ruff F821 in the package CI)
        M = B = A_arm = P_ctrl = P_h1 = D1h = D2h = C1 = C2 = CL = REFS = None
        gc.collect()

    # ---- verdict D across splits (prereg 5, with the 2026-11-02 amendment: G4 = ctrl_line; G4b joins G5 as a
    # ---- validity check - both are at-chance-by-construction nulls)
    g1 = {s: gate_verdicts[s]['G1_recipe_split_minus_0.5'] for s in SPLITS}
    g3 = {s: gate_verdicts[s]['G3_split_recipe_minus_fpshuf'] for s in SPLITS}
    g4 = {s: gate_verdicts[s]['G4_ctrl_line_orig_minus_0.5'] for s in SPLITS}
    g4b = {s: gate_verdicts[s]['G4b_ctrl_sample_orig_minus_0.5'] for s in SPLITS}
    g5 = {s: gate_verdicts[s]['G5_ctrl_h1_split_minus_0.5'] for s in SPLITS}
    # a null excluding zero on ONE split is a ~2.5% event (its draw is not resampled by the structure bootstrap)
    # and is a caution; on two or more splits it voids the step (amendment 2026-11-02)
    n_bad_g5 = sum(1 for v in g5.values() if v != 'includes zero')
    n_bad_g4b = sum(1 for v in g4b.values() if v != 'includes zero')
    caution = ['G5 %s %s' % (s.split('_')[-1], g5[s]) for s in SPLITS if g5[s] != 'includes zero'] \
        + ['G4b %s %s' % (s.split('_')[-1], g4b[s]) for s in SPLITS if g4b[s] != 'includes zero']
    if n_bad_g5 >= 2 or n_bad_g4b >= 2:
        vD = '5 UNINTERPRETABLE (an at-chance-by-construction null excludes zero on two or more splits: %s)' \
            % ', '.join(caution)
    elif all(v != 'above zero' for v in g1.values()) and all(v == 'above zero' for v in g4.values()):
        vD = '1 S1-DOMINATED (shared control estimate; G1 not above zero on any split, G4 above zero on all)'
    elif all(v == 'above zero' for v in g1.values()) and all(v != 'above zero' for v in g3.values()):
        vD = '2 CONDITION-REAL, COMPOUND-FREE (G1 above zero on all splits, G3 not above zero on any)'
    elif all(v == 'above zero' for v in g1.values()) and all(v == 'above zero' for v in g3.values()):
        vD = '3 COMPOUND SIGNAL RESTORED (G1 and G3 above zero on all splits)'
    else:
        vD = '4 MIXED (G1: %s; G3: %s; G4: %s)' % tuple(
            ', '.join('%s %s' % (s.split('_')[-1], d[s]) for s in SPLITS) for d in (g1, g3, g4))
    if caution and not vD.startswith('5'):
        vD += '  [caution: one-split null exclusion, %s]' % ', '.join(caution)
    say('\n' + '=' * 110)
    say('PRE-REGISTERED VERDICT D (section 5): %s' % vD)
    for s in SPLITS:
        say('  %s  G1 %s | G2 %s | G3 %s | G4 %s | G4b %s | G5 %s'
            % (s, g1[s], gate_verdicts[s]['G2_fpshuf_split_minus_0.5'], g3[s], g4[s], g4b[s], g5[s]))
    say('=' * 110)

    text = json.dumps(dict(prereg=PRE, commit=commit, splits=list(SPLITS), n_boot=s23.N_BOOT,
                           interval='97.5% percentile, structure-clustered, fresh RandomState(0) per scored set',
                           schemes=list(SCHEMES), arms=list(ALL_ARMS),
                           seeds=dict(pool_halving='RandomState(0) per pool', ctrl_sample=SEED_CTRL,
                                      ctrl_sample_h1=SEED_H1, duplicate=SEED_DUP),
                           reference=dict(boot=REF_BOOT, test_reference_override=bool(TEST_REF)),
                           verdict=dict(D_source=vD, gates_by_split=gate_verdicts),
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
