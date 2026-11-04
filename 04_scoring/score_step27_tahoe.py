#!/usr/bin/env python3
"""
score_step27_tahoe.py - Step 27: condition and compound signal once the plate is uninformative.

Purpose
-------
Step 26 showed that the published PDS discrimination is reproduced by a no-model prediction carrying only the plate
(the line's pooled control mean), that this survives split-control scoring, and - descriptively, recipe arm only - that
restricting PDS candidates to the condition's own plate drops the recipe from ~0.72 to 0.50-0.53. This scorer makes
that a measured claim (PREREG_step27_within_plate.md): every Step 26 arm is re-scored with candidates restricted to the
same (line, dose, plate), with structure-clustered intervals, under S-orig, S-split and S-flip; and the mechanism
inferred in step26_findings.md 5.1 is tested directly - the cosine between the measured change (true - C2) and the
plate's control offset (L - C1), against a resample null drawn from the other half. CPU only; no model is run.

Logic
-----
1. Guards (prereg 6.1): this file, the pre-registration, every imported scorer and every record read - including the
   Step 26 records and findings, so Step 26 must be scored and filed first - committed and identical to HEAD; no
   output exists.
2. Per split, the preparation of score_step26_tahoe.py is repeated with its code and its seeds (data-file hashes,
   row alignment, rebuild gates to 1e-4, pools by (line, plate), the RandomState(0) halving, the ctrl_sample /
   ctrl_sample_h1 / duplicate draws with seeds 1000000+i / 2000000+i / 3000000+i, the swap arms through the Step 25
   checked loader). The block is copied, not imported, because Step 26's main() is one function; the copy is then
   PROVEN equal to Step 26 by the reproduction gate of step 3, which is why it is safe.
3. Reproduction (prereg 6.3): every arm's cross-plate PDS under S-orig / S-split / S-flip equals the committed
   step26_source_per_condition.csv per condition to 1e-9 (so pools, halves, draws and ceiling are rebuilt bitwise);
   the recipe's within-plate values and coverage equal step26_source.json; the Step 15c published intervals replay
   exactly (score_step25_tahoe.replay_step15c). No within-plate interval or cosine is read before this passes.
4. Within-plate candidate sets (prereg 3): the Step 15c set intersected with the condition's plate, >= MIN_CAND
   members, scored for all arms through score_step15c_tahoe.pds_sets (nothing reimplemented). The ctrl_line change
   vector is checked identical within every set (prereg 6.4).
5. Plate residual (prereg 3-4): per condition the cosine between a change vector and the plate's control offset,
   under S-orig (X - C vs CL - C) and S-split (X - C2 vs CL - C1); X = true (measured residual, H5), each arm's
   prediction (predicted residual), and ctrl_sample_h2 - a thirteenth arm drawn from half h2 with RandomState(4000000+i),
   used only here as the null whose change is independent of the offset's half.
6. Contrasts (prereg 4): H1-H5 with S-flip agreement, H4/H4b validity nulls under the two-split void rule, base
   analogues and S-orig versions ungated; verdicts G (within-plate signal), H (mechanism), I (gap attribution).
Intervals: 97.5 % structure-clustered percentile bootstrap, 5,000 replicates, fresh RandomState(0) per scored set
(score_step23_screen.Boot). Outputs (never overwritten): tahoe_results/step27_within_plate.json,
step27_within_plate_per_condition.csv, step27_within_plate_summary.csv.

Run on a compute node from the worktree root through run_step27_score.sh, after Step 26 is scored and committed.
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
import score_step15_tahoe as s15        # noqa: E402  dense, sha, load_eval, RES, RECORD, N_SEEDS, ALL_SPLITS
import score_step15c_tahoe as s15c      # noqa: E402  pds_sets, close, self_check, MIN_CAND
import score_step23_screen as s23       # noqa: E402  Boot, interval, fmt, committed_and_clean, git, say, die
import score_step25_tahoe as s25        # noqa: E402  load_swap, perm_path, replay_step15c, REF_BOOT, BOOT_TOL
import score_step26_tahoe as s26        # noqa: E402  constants, find_file_sha, load_A (the arm definitions)

PRE = 'PREREG_step27_within_plate.md'
SPLITS = s25.SPLITS                                      # development first
CKARMS = ('recipe', 'base')
SCHEMES = ('orig', 'split', 'flip')                      # prereg 3: linemean / none are moot within a plate
MODEL_ARMS = s26.MODEL_ARMS
ALL_ARMS = s26.ALL_ARMS
DATA = s26.DATA
PC15B, PC15C = s25.PC15B, s25.PC15C
S24_PC = os.path.join(s15.RES, 'step24_swap_per_condition.csv')
S25_PC = os.path.join(s15.RES, 'step25_swap_per_condition.csv')
S25_JSON = os.path.join(s15.RES, 'step25_swap.json')
S26_JSON = os.path.join(s15.RES, 'step26_source.json')
S26_PC = os.path.join(s15.RES, 'step26_source_per_condition.csv')
OUT_JSON = os.path.join(s15.RES, 'step27_within_plate.json')
OUT_PC = os.path.join(s15.RES, 'step27_within_plate_per_condition.csv')
OUT_SUM = os.path.join(s15.RES, 'step27_within_plate_summary.csv')
MIN_SECTIONS = 7
REBUILD_TOL, FLAG_POOL, FLAG_CELLS = s26.REBUILD_TOL, s26.FLAG_POOL, s26.FLAG_CELLS
SEED_CTRL, SEED_H1, SEED_DUP = s26.SEED_CTRL, s26.SEED_H1, s26.SEED_DUP
SEED_H2 = 4000000                                        # prereg 4 (H5): the h2 resample null, + row index
T0 = time.time()
say, die = s23.say, s23.die
REF_BOOT, BOOT_TOL = s25.REF_BOOT, s25.BOOT_TOL
TEST_REF = os.environ.get('S26_TEST_REF')                # the synthetic test's replay targets (never set for real)
if TEST_REF:
    REF_BOOT = json.load(open(TEST_REF))['boot']


def cosine_rows(X, Y):
    """Row-wise cosine between two matrices; NaN where either row has zero norm."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    nx, ny = np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1)
    out = np.full(len(X), np.nan)
    ok = (nx > 0) & (ny > 0)
    out[ok] = np.einsum('ij,ij->i', X[ok], Y[ok]) / (nx[ok] * ny[ok])
    return out


def main():
    if len(sys.argv) > 1:
        die('this script takes no options (prereg: all three splits, fixed arms and schemes)')
    for q in (OUT_JSON, OUT_PC, OUT_SUM):
        if os.path.exists(q):
            die('%s exists; refusing to overwrite a record' % q)
    if not os.path.isfile(PRE) or open(PRE).read().count('\n## ') < MIN_SECTIONS:
        die('pre-registration missing or incomplete')
    # ---- prereg 6.1: everything read is committed and clean; Step 26 scored and filed first
    s23.committed_and_clean(
        [PRE, 'score_step27_tahoe.py', 'run_step27_score.sh', 'PREREG_step26_control_source.md',
         'score_step26_tahoe.py', 'PREREG_step25_swap_confirmation.md', 'score_step25_tahoe.py',
         'score_step23_screen.py', 'score_step15_tahoe.py', 'score_step15c_tahoe.py', 'step14_metrics.py'],
        'pre-registration or scripts')
    records = [PC15B, PC15C, s15.RECORD, S24_PC, S25_PC, S25_JSON, S26_JSON, S26_PC,
               'step24_findings.md', 'step25_findings.md', 'step26_findings.md'] + [s25.perm_path(s) for s in SPLITS]
    s23.committed_and_clean(records, 'records (Step 26 must be scored and its findings committed first)')
    commit = s23.git('rev-parse', 'HEAD').stdout.strip()
    say('[guards] pre-registration, scripts and records committed and clean; commit %s' % commit[:7])
    if TEST_REF:
        say('[WARNING] S26_TEST_REF set: reproduction targets read from %s (synthetic test only)' % TEST_REF)
    s15c.self_check()
    import anndata                                           # noqa: E402  only the data gates need it
    record = json.load(open(s15.RECORD))
    pc15b, pc15c = pd.read_csv(PC15B), pd.read_csv(PC15C)
    pc25 = pd.read_csv(S25_PC)
    s26json = json.load(open(S26_JSON))
    s26pc = pd.read_csv(S26_PC)
    if tuple(SPLITS) != tuple(s15.ALL_SPLITS):
        die('SPLITS must equal Step 15c\'s split order for the interval replay')
    s15.rng = np.random.RandomState(0)                       # Step 15c's one generator, advanced by the replay only

    results, pc_rows, summary_rows, gate_verdicts = {}, [], [], {}
    for split in SPLITS:
        k = split.split('_')[-1]
        perm = json.load(open(s25.perm_path(split)))
        perm_sha = perm['sha256']

        # ============================================================================================ preparation
        # Verbatim the Step 26 scorer's per-split preparation (score_step26_tahoe.py main(), own runs through the
        # swap-arm checks). Any divergence is caught by the per-condition reproduction gate below.
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
        pcb = pc15b[pc15b['split'] == split].reset_index(drop=True)
        pcc = pc15c[pc15c['split'] == split].reset_index(drop=True)
        p25 = pc25[pc25['split'] == split].reset_index(drop=True)
        p26 = s26pc[s26pc['split'] == split].reset_index(drop=True)
        for nm, df in (('Step 15b', pcb), ('Step 15c', pcc), ('Step 25', p25), ('Step 26', p26)):
            if len(df) != n or not np.array_equal(df['condition'].astype(str).values, cond):
                die('%s: the %s per-condition record is not in the vectors order' % (split, nm))
        lines = pcb['line'].astype(str).values
        drugs = pcb['structure'].astype(str).values
        doses = pcb['dose_nM'].astype(float).values
        plates = pcc['plate'].astype(str).values
        ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
        contexts = [np.where(ctx_key == u)[0] for u in np.unique(ctx_key)]
        cand = {}
        for members in contexts:
            for p_ in members:
                cand[p_] = np.array([q for q in members if q == p_ or drugs[q] != drugs[p_]])
        if not np.array_equal(np.array([len(cand[i]) for i in range(n)]), pcc['n_candidates'].values):
            die('%s: candidate-set sizes differ from the Step 15c record' % split)
        Z = np.zeros((n, ref0['true'].shape[1]))

        tf = os.path.join(DATA, 'tahoe_test_%s.h5ad' % split)
        cf = os.path.join(DATA, 'tahoe_test_%s_control.h5ad' % split)
        for q in (tf, cf):
            if not os.path.isfile(q):
                die('missing data file %s' % q)
            want = s26.find_file_sha(record, os.path.basename(q))
            if want is None:
                die('the Step 15 data record holds no sha256 for %s' % os.path.basename(q))
            got = s15.sha(q)
            if got != want:
                die('%s sha256 %s differs from the data record %s' % (q, got[:12], want[:12]))
        te, co = anndata.read_h5ad(tf), anndata.read_h5ad(cf)
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
        order = np.argsort(tg, kind='stable')
        sorted_tg = tg[order]
        starts = np.searchsorted(sorted_tg, cond, side='left')
        ends = np.searchsorted(sorted_tg, cond, side='right')
        if (ends <= starts).any():
            die('%s: %d conditions have no rows in the test file' % (split, int((ends <= starts).sum())))
        rows = [np.sort(order[starts[i]:ends[i]]) for i in range(n)]
        lp_of = [(lines[i], plates[i]) for i in range(n)]
        pool_idx = {}
        for i in range(n):
            pool_idx.setdefault(lp_of[i], []).append(rows[i])
        pool_idx = {lp: np.sort(np.concatenate(v)) for lp, v in pool_idx.items()}
        pool_size = np.array([len(pool_idx[lp_of[i]]) for i in range(n)])
        n_cells = np.array([len(ref0['true_cells'][i]) for i in range(n)])
        flagged = (pool_size < FLAG_POOL) | (n_cells < FLAG_CELLS)
        dev_t = max(float(np.abs(tX[rows[i]].mean(0, dtype=np.float64) - ref0['true'][i]).max()) for i in range(n))
        if dev_t > REBUILD_TOL:
            die('%s: rebuilt per-condition means deviate from stored true by %.2e' % (split, dev_t))
        cmean = {lp: cX[idx].mean(0, dtype=np.float64) for lp, idx in pool_idx.items()}
        dev_c = max(float(np.abs(cmean[lp_of[i]] - ref0['C'][i]).max()) for i in range(n))
        if dev_c > REBUILD_TOL:
            die('%s: rebuilt (line, plate) pool means deviate from stored C by %.2e' % (split, dev_c))
        miss = 0
        for i in range(n):
            have = {r.tobytes() for r in tX[rows[i]]}
            miss += sum(1 for c_ in ref0['true_cells'][i] if np.asarray(c_, np.float32).tobytes() not in have)
        if miss:
            die('%s: %d stored real cells are not rows of their condition in the test file' % (split, miss))
        say('[%s] rebuild gates: stored true and C reproduced from the data files (max deviation %.2e and %.2e); '
            'all stored real cells recovered bitwise; %d pools, %d conditions flagged'
            % (split, dev_t, dev_c, len(pool_idx), int(flagged.sum())))
        del tX, order, sorted_tg
        gc.collect()
        C1p, C2p, h1p, h2p = {}, {}, {}, {}
        for lp, idx in pool_idx.items():
            pm = np.random.RandomState(0).permutation(len(idx))
            a_, b_ = idx[pm[:(len(idx) + 1) // 2]], idx[pm[(len(idx) + 1) // 2:]]
            if len(b_) == 0:
                C1p[lp], C2p[lp], h1p[lp], h2p[lp] = cX[a_].mean(0, dtype=np.float64), np.full(cX.shape[1], np.nan), a_, b_
                continue
            if np.intersect1d(a_, b_).size or len(a_) + len(b_) != len(idx):
                die('pool halving is not a disjoint exhaustive split (%s)' % (lp,))
            C1p[lp], C2p[lp], h1p[lp], h2p[lp] = cX[a_].mean(0, dtype=np.float64), cX[b_].mean(0, dtype=np.float64), a_, b_
        C = ref0['C']
        C1 = np.vstack([C1p[lp_of[i]] for i in range(n)])
        C2 = np.vstack([C2p[lp_of[i]] for i in range(n)])
        line_rows = {}
        for i in range(n):
            line_rows.setdefault(lines[i], set()).add(lp_of[i])
        lmean = {ln: cX[np.sort(np.concatenate([pool_idx[lp] for lp in lps]))].mean(0, dtype=np.float64)
                 for ln, lps in line_rows.items()}
        CL = np.vstack([lmean[lines[i]] for i in range(n)])
        REFS = {'orig': (C, C), 'split': (C1, C2), 'flip': (C2, C1)}
        P_ctrl, P_h1, P_h2, D1h, D2h = (np.empty((n, Z.shape[1])) for _ in range(5))
        for i in range(n):
            idx = pool_idx[lp_of[i]]
            t1 = np.random.RandomState(SEED_CTRL + i).choice(len(idx), n_cells[i], replace=(len(idx) < n_cells[i]))
            P_ctrl[i] = cX[idx[t1]].mean(0, dtype=np.float64)
            h1 = h1p[lp_of[i]]
            t2 = np.random.RandomState(SEED_H1 + i).choice(len(h1), n_cells[i], replace=(len(h1) < n_cells[i]))
            P_h1[i] = cX[h1[t2]].mean(0, dtype=np.float64)
            # --- Step 27 addition (prereg 4, H5 null): a resample from the OTHER half, h2, whose change relative to
            # C2 is independent of the offset's half h1
            h2 = h2p[lp_of[i]]
            if len(h2) == 0:
                P_h2[i] = np.nan
            else:
                t3 = np.random.RandomState(SEED_H2 + i).choice(len(h2), n_cells[i], replace=(len(h2) < n_cells[i]))
                P_h2[i] = cX[h2[t3]].mean(0, dtype=np.float64)
            cells = np.asarray(ref0['true_cells'][i], np.float64)
            pm = np.random.RandomState(SEED_DUP + i).permutation(len(cells))
            D1h[i] = cells[pm[:(len(cells) + 1) // 2]].mean(0)
            D2h[i] = cells[pm[(len(cells) + 1) // 2:]].mean(0) if len(cells) > 1 else np.nan
        del cX
        gc.collect()
        B = {('recipe', 'own'): [o['B'] for o in own['recipe']], ('base', 'own'): [o['B'] for o in own['base']]}
        A_arm = {a: [s26.load_A(o) for o in own[a]] for a in CKARMS}
        for a in CKARMS:
            for mode in ('shuf', 'zero'):
                B[(a, mode)] = []
                for s in range(s15.N_SEEDS):
                    w = s25.load_swap(split, a, mode, s, perm_sha, cksha[a][s])
                    if not np.array_equal(w['cond'], cond) or np.abs(w['true'] - ref0['true']).max() > 1e-5 \
                            or np.abs(w['C'] - ref0['C']).max() > 1e-5:
                        die('%s does not share the own run\'s conditions and means' % w['tag'])
                    B[(a, mode)].append(w['B'])
                    w['pred'] = w['true_cells'] = None
                    del w
        for a in CKARMS:
            for s in range(s15.N_SEEDS):
                if own[a][s] is not ref0:
                    own[a][s]['pred'] = own[a][s]['true_cells'] = None
        gc.collect()
        armkey = {'recipe': ('recipe', 'own'), 'recipe_fpshuf': ('recipe', 'shuf'), 'recipe_fpzero': ('recipe', 'zero'),
                  'base': ('base', 'own'), 'base_fpshuf': ('base', 'shuf'), 'base_fpzero': ('base', 'zero')}
        # ============================================================================================ end of copy

        # ---- every arm's change matrices per scheme (prediction side D, measured side Y); the orig construction
        # ---- uses each own run's stored C exactly as Step 26 did
        def changes(arm, sch):
            """List over seeds of (D, Y) for one arm under one scheme."""
            R1, R2 = REFS[sch]
            Y = ref0['true'] - R2
            if arm in MODEL_ARMS:
                a, mode = armkey[arm]
                out = []
                for s in range(s15.N_SEEDS):
                    P = own[a][s]['B'] if mode == 'own' else B[(a, mode)][s]
                    if sch == 'orig':
                        out.append((P - (own[a][s]['C'] if mode == 'own' else C), ref0['true'] - C))
                    else:
                        out.append((P - R1, Y))
                return out
            if arm.startswith('recon_'):
                return [(A_arm[arm[6:]][s] - R1, Y) for s in range(s15.N_SEEDS)]
            if arm == 'ctrl_sample':
                return [(P_ctrl - R1, Y)]
            if arm == 'ctrl_sample_h1':
                return [(P_h1 - R1, Y)]
            if arm == 'ctrl_line':
                return [(CL - R1, Y)]
            if arm == 'duplicate':
                return [(D1h - R1, D2h - R2)]
            die('unknown arm %s' % arm)

        # ---- reproduction gate (prereg 6.3): cross-plate PDS of every arm equals the Step 26 record per condition
        for arm in ALL_ARMS:
            for sch in SCHEMES:
                v = np.nanmean(np.vstack([s15c.pds_sets(D, Y, contexts, cand) for D, Y in changes(arm, sch)]), axis=0)
                col = '%s__%s__pds' % (arm, sch)
                if col not in p26.columns:
                    die('%s: the Step 26 per-condition record has no column %s' % (split, col))
                if not s15c.close(v, p26[col].values):
                    die('%s: %s %s cross-plate PDS does not reproduce the committed Step 26 record to 1e-9'
                        % (split, arm, sch))
        rep15c = s25.replay_step15c(pcc, pcc['structure'].astype(str).values)
        for a in CKARMS:
            got, tgt = rep15c[a], REF_BOOT[a][split]
            if any(abs(x - y) > BOOT_TOL for x, y in zip(got, tgt)):
                die('%s %s replay of Step 15c\'s bootstrap draw order does not reproduce its interval' % (split, a))
        say('[%s] reproduction: every arm\'s cross-plate PDS under S-orig/S-split/S-flip equals the committed Step 26 '
            'per-condition record to 1e-9 (pools, halves, draws and ceiling rebuilt bitwise); Step 15c intervals '
            'replayed exactly' % split)

        # ---- within-plate candidate sets (prereg 3): the Step 15c set intersected with the condition's plate
        cand_pl = {}
        for i in range(n):
            cp = np.array([q for q in cand[i] if q == i or plates[q] == plates[i]])
            if len(cp) >= s15c.MIN_CAND:
                cand_pl[i] = cp
        covered = np.array([i in cand_pl for i in range(n)])
        wp26 = s26json['per_split'][split]['within_plate']
        if int(covered.sum()) != int(wp26['recipe_orig']['n']):
            die('%s: within-plate coverage %d differs from Step 26\'s %d' % (split, int(covered.sum()), wp26['recipe_orig']['n']))
        # prereg 6.4: ctrl_line's change vector is identical across the members of every within-plate set
        for i, cp in cand_pl.items():
            if np.abs((CL - C)[cp] - (CL - C)[i]).max() > 1e-12:
                die('%s: ctrl_line change differs within a within-plate set (condition %s)' % (split, cond[i]))
        W = {}                                                 # (arm, sch) -> per-condition within-plate PDS (NaN uncovered)
        for arm in ALL_ARMS:
            for sch in SCHEMES:
                W[(arm, sch)] = np.nanmean(np.vstack([s15c.pds_sets(D, Y, contexts, cand_pl)
                                                      for D, Y in changes(arm, sch)]), axis=0)
                if np.isfinite(W[(arm, sch)]).sum() != int(covered.sum()):
                    die('%s: %s %s within-plate PDS is finite on %d conditions, coverage is %d'
                        % (split, arm, sch, int(np.isfinite(W[(arm, sch)]).sum()), int(covered.sum())))
        for sch in ('orig', 'split'):
            got, want = float(np.nanmean(W[('recipe', sch)])), float(wp26['recipe_%s' % sch]['pds'])
            if abs(got - want) > 1e-9:
                die('%s: recipe within-plate %s PDS %.10f differs from Step 26\'s %.10f' % (split, sch, got, want))
        say('[%s] within-plate: %d of %d conditions covered (sets >= %d), %d (line, dose, plate) sets; recipe orig '
            '%.4f split %.4f reproduce Step 26' % (split, int(covered.sum()), n, s15c.MIN_CAND,
                                                   len({(lines[i], doses[i], plates[i]) for i in cand_pl}),
                                                   np.nanmean(W[('recipe', 'orig')]), np.nanmean(W[('recipe', 'split')])))

        # ---- plate residual (prereg 3-4): cosine with the plate's control offset
        off = {'orig': CL - C, 'split': CL - C1}
        # the measured-side reference for each residual scheme: orig subtracts C, split subtracts C2
        res_ref = {'orig': C, 'split': C2}
        RES_ = {}                                              # (name, sch) -> per-condition cosine
        for sch in ('orig', 'split'):
            RES_[('true', sch)] = cosine_rows(ref0['true'] - res_ref[sch], off[sch])
            RES_[('ctrl_sample_h2', sch)] = cosine_rows(P_h2 - res_ref[sch], off[sch])
            RES_[('ctrl_sample_h1', sch)] = cosine_rows(P_h1 - res_ref[sch], off[sch])
            RES_[('ctrl_sample', sch)] = cosine_rows(P_ctrl - res_ref[sch], off[sch])
            RES_[('ctrl_line', sch)] = cosine_rows(CL - res_ref[sch], off[sch])
            RES_[('duplicate', sch)] = cosine_rows(D2h - res_ref[sch], off[sch])
            for arm in MODEL_ARMS + ('recon_recipe', 'recon_base'):
                a = arm[6:] if arm.startswith('recon_') else armkey[arm][0]
                mats = A_arm[a] if arm.startswith('recon_') else \
                    [own[a][s]['B'] if armkey[arm][1] == 'own' else B[armkey[arm]][s] for s in range(s15.N_SEEDS)]
                RES_[(arm, sch)] = np.nanmean(np.vstack([cosine_rows(P - res_ref[sch], off[sch]) for P in mats]), axis=0)

        # ---- contrasts (prereg 4)
        con = {}
        half = np.full(n, 0.5)

        def contrast(name, xa, xb):
            diff = np.asarray(xa, np.float64) - np.asarray(xb, np.float64)
            ok = np.isfinite(diff)
            b = s23.Boot(np.flatnonzero(ok), drugs, n)     # fresh RandomState(0) per scored set (the Step 23 way)
            pt_, reps_ = b.mean_reps(diff)
            con[name] = s23.interval(pt_, reps_)
            con[name]['n_conditions'] = int(ok.sum())
            con[name]['n_structures'] = int(b.G)
            return con[name]

        wpv = lambda arm, sch: W[(arm, sch)]
        gates = [('H1_recipe_split_minus_0.5', wpv('recipe', 'split'), half),
                 ('H1f_recipe_flip_minus_0.5', wpv('recipe', 'flip'), half),
                 ('H1o_recipe_orig_minus_0.5', wpv('recipe', 'orig'), half),
                 ('H2_split_recipe_minus_base', wpv('recipe', 'split'), wpv('base', 'split')),
                 ('H2f_flip_recipe_minus_base', wpv('recipe', 'flip'), wpv('base', 'flip')),
                 ('H2o_orig_recipe_minus_base', wpv('recipe', 'orig'), wpv('base', 'orig')),
                 ('H3_split_recipe_minus_fpshuf', wpv('recipe', 'split'), wpv('recipe_fpshuf', 'split')),
                 ('H3f_flip_recipe_minus_fpshuf', wpv('recipe', 'flip'), wpv('recipe_fpshuf', 'flip')),
                 ('H3o_orig_recipe_minus_fpshuf', wpv('recipe', 'orig'), wpv('recipe_fpshuf', 'orig')),
                 ('H4_ctrl_line_orig_minus_0.5', wpv('ctrl_line', 'orig'), half),
                 ('H4b_ctrl_h1_split_minus_0.5', wpv('ctrl_sample_h1', 'split'), half),
                 ('H5_residual_true_split', RES_[('true', 'split')], np.zeros(n)),
                 ('H5null_residual_ctrl_h2_split', RES_[('ctrl_sample_h2', 'split')], np.zeros(n)),
                 ('H5o_residual_true_orig', RES_[('true', 'orig')], np.zeros(n)),
                 ('base_H1_split_minus_0.5', wpv('base', 'split'), half),
                 ('base_H1f_flip_minus_0.5', wpv('base', 'flip'), half),
                 ('base_H3_split_own_minus_fpshuf', wpv('base', 'split'), wpv('base_fpshuf', 'split')),
                 ('base_H3f_flip_own_minus_fpshuf', wpv('base', 'flip'), wpv('base_fpshuf', 'flip')),
                 ('fpzero_split_minus_0.5', wpv('recipe_fpzero', 'split'), half),
                 ('duplicate_split_minus_0.5', wpv('duplicate', 'split'), half),
                 ('ctrl_sample_split_minus_0.5', wpv('ctrl_sample', 'split'), half)]
        say('\n' + '=' * 110 + '\n%s: %d conditions, %d covered within plate\n' % (split, n, int(covered.sum())) + '=' * 110)
        say('  %-18s %s' % ('arm', '  '.join('%12s' % sch for sch in SCHEMES)) + '   (within-plate PDS, mean over covered)')
        for arm in ALL_ARMS:
            say('  %-18s %s' % (arm, '  '.join('%12.4f' % np.nanmean(W[(arm, sch)]) for sch in SCHEMES)))
        say('  %-18s %s   (plate-residual cosine, mean over all conditions)' % ('change vector', '  '.join('%12s' % s_ for s_ in ('orig', 'split'))))
        for nm in ('true', 'ctrl_sample_h2', 'ctrl_sample_h1', 'ctrl_sample', 'ctrl_line', 'duplicate') + MODEL_ARMS + ('recon_recipe', 'recon_base'):
            say('  %-18s %s' % (nm, '  '.join('%12.4f' % np.nanmean(RES_[(nm, s_)]) for s_ in ('orig', 'split'))))
        for nm, xa, xb in gates:
            say('  %-34s %s  (n=%d)' % (nm, s23.fmt(contrast(nm, xa, xb)), con[nm]['n_conditions']))
        gate_verdicts[split] = {nm: con[nm]['verdict'] for nm, _, _ in gates}

        # ---- verdict I inputs: the published cross-plate gap from the Step 25 record (recipe - base, S-orig)
        gap_cross = float(np.nanmean(p25['recipe_own_pds'].values - p25['base_own_pds'].values))
        frac_I = con['H2_split_recipe_minus_base']['diff'] / gap_cross if abs(gap_cross) > 0.02 else None
        say('  verdict I: within-plate recipe - base (S-split) %+.4f is %s of the published cross-plate gap %+.4f'
            % (con['H2_split_recipe_minus_base']['diff'], '%.1f%%' % (100 * frac_I) if frac_I is not None else 'n/a', gap_cross))

        # ---- records
        for i in range(n):
            r = dict(split=split, condition=cond[i], line=lines[i], plate=plates[i], structure=drugs[i],
                     dose_nM=doses[i], n_candidates=int(len(cand[i])),
                     n_candidates_within_plate=int(len(cand_pl[i])) if i in cand_pl else 0,
                     covered=int(covered[i]), pool_size=int(pool_size[i]), n_stored_cells=int(n_cells[i]),
                     flagged=int(flagged[i]))
            for arm in ALL_ARMS:
                for sch in SCHEMES:
                    r['%s__%s__wp_pds' % (arm, sch)] = W[(arm, sch)][i]
            for (nm, sch), v in RES_.items():
                r['residual__%s__%s' % (nm, sch)] = v[i]
            pc_rows.append(r)
        for kk, c_ in con.items():
            summary_rows.append(dict(split=split, kind='contrast', name=kk, value=c_['diff'], lo=c_['ci'][0],
                                     hi=c_['ci'][1], verdict=c_['verdict'], n_conditions=c_['n_conditions']))
        vals = {}
        for arm in ALL_ARMS:
            for sch in SCHEMES:
                vals['%s__%s' % (arm, sch)] = float(np.nanmean(W[(arm, sch)]))
                summary_rows.append(dict(split=split, kind='value', name='%s__%s_wp_pds' % (arm, sch),
                                         value=vals['%s__%s' % (arm, sch)], lo='', hi='', verdict='',
                                         n_conditions=int(covered.sum())))
        resid = {}
        for (nm, sch), v in RES_.items():
            resid['%s__%s' % (nm, sch)] = float(np.nanmean(v))
            summary_rows.append(dict(split=split, kind='value', name='residual__%s__%s' % (nm, sch),
                                     value=resid['%s__%s' % (nm, sch)], lo='', hi='', verdict='',
                                     n_conditions=int(np.isfinite(v).sum())))
        results[split] = dict(
            n_conditions=int(n), n_covered=int(covered.sum()),
            n_within_plate_sets=len({(lines[i], doses[i], plates[i]) for i in cand_pl}),
            rebuild=dict(true_max_dev=dev_t, C_max_dev=dev_c),
            data_files={os.path.basename(tf): s15.sha(tf), os.path.basename(cf): s15.sha(cf)},
            within_plate_pds=vals, residual_cosine=resid, contrasts=con,
            published_cross_plate_gap=gap_cross, verdict_I_fraction=frac_I,
            step15c_replay=dict(recipe=list(rep15c['recipe']), base=list(rep15c['base'])))
        for a in CKARMS:
            for o in own[a]:
                o['pred'] = o['true_cells'] = None
        # release by rebinding, not del (closures over these names; ruff F821 in the package CI)
        W = RES_ = B = A_arm = P_ctrl = P_h1 = P_h2 = D1h = D2h = C1 = C2 = CL = REFS = None
        gc.collect()

    # ---- verdict G (prereg 4): a split "shows above zero" only under both halvings; nulls under the two-split rule
    def above(s, key, keyf):
        return gate_verdicts[s][key] == 'above zero' and gate_verdicts[s][keyf] == 'above zero'
    h1 = {s: above(s, 'H1_recipe_split_minus_0.5', 'H1f_recipe_flip_minus_0.5') for s in SPLITS}
    h3 = {s: above(s, 'H3_split_recipe_minus_fpshuf', 'H3f_flip_recipe_minus_fpshuf') for s in SPLITS}
    h4 = {s: gate_verdicts[s]['H4_ctrl_line_orig_minus_0.5'] for s in SPLITS}
    h4b = {s: gate_verdicts[s]['H4b_ctrl_h1_split_minus_0.5'] for s in SPLITS}
    caution = ['H4 %s %s' % (s.split('_')[-1], h4[s]) for s in SPLITS if h4[s] != 'includes zero'] \
        + ['H4b %s %s' % (s.split('_')[-1], h4b[s]) for s in SPLITS if h4b[s] != 'includes zero']
    n_bad4 = sum(1 for v in h4.values() if v != 'includes zero')
    n_bad4b = sum(1 for v in h4b.values() if v != 'includes zero')
    n_h1 = sum(h1.values())
    n_h3 = sum(h3.values())
    if n_bad4 >= 2 or n_bad4b >= 2:
        vG = '5 UNINTERPRETABLE (a validity null excludes zero on two or more splits: %s)' % ', '.join(caution)
    elif n_h1 == 0:
        vG = '1 NO RESOLVED SIGNAL BEYOND PLATE (H1 includes zero on all three splits)'
    elif n_h1 >= 2 and n_h3 == 0:
        vG = '2 SMALL CONDITION SIGNAL, COMPOUND-FREE (H1 above zero under both halvings on %d splits; H3 on none)' % n_h1
    elif n_h1 >= 2 and n_h3 >= 2:
        vG = '3 COMPOUND SIGNAL WITHIN PLATE (H1 above zero on %d splits, H3 on %d)' % (n_h1, n_h3)
    else:
        vG = '4 MIXED (H1 above zero on %d split(s): %s; H3 on %d: %s)' % (
            n_h1, ', '.join(s.split('_')[-1] for s in SPLITS if h1[s]) or '-',
            n_h3, ', '.join(s.split('_')[-1] for s in SPLITS if h3[s]) or '-')
    if caution and not vG.startswith('5'):
        vG += '  [caution: one-split null exclusion, %s]' % ', '.join(caution)
    h5 = {s: gate_verdicts[s]['H5_residual_true_split'] for s in SPLITS}
    h5n = {s: gate_verdicts[s]['H5null_residual_ctrl_h2_split'] for s in SPLITS}
    if all(v == 'above zero' for v in h5.values()) and all(v == 'includes zero' for v in h5n.values()):
        vH = 'SUPPORTED: the measured change points along the plate\'s control offset on all three splits; the h2 null includes zero'
    elif all(v == 'includes zero' for v in h5.values()):
        vH = 'NOT SUPPORTED: the measured residual includes zero on every split; the Step 26 5.1 inference is withdrawn'
    else:
        vH = 'MIXED (H5: %s; null: %s)' % (', '.join('%s %s' % (s.split('_')[-1], h5[s]) for s in SPLITS),
                                            ', '.join('%s %s' % (s.split('_')[-1], h5n[s]) for s in SPLITS))
    say('\n' + '=' * 110)
    say('PRE-REGISTERED VERDICT G (section 4): %s' % vG)
    say('VERDICT H (mechanism): %s' % vH)
    for s in SPLITS:
        say('  %s  H1 %s/%s | H2 %s | H3 %s/%s | H4 %s | H4b %s | H5 %s (null %s)'
            % (s, gate_verdicts[s]['H1_recipe_split_minus_0.5'], gate_verdicts[s]['H1f_recipe_flip_minus_0.5'],
               gate_verdicts[s]['H2_split_recipe_minus_base'], gate_verdicts[s]['H3_split_recipe_minus_fpshuf'],
               gate_verdicts[s]['H3f_flip_recipe_minus_fpshuf'], h4[s], h4b[s], h5[s], h5n[s]))
    say('=' * 110)

    text = json.dumps(dict(prereg=PRE, commit=commit, splits=list(SPLITS), n_boot=s23.N_BOOT,
                           interval='97.5% percentile, structure-clustered, fresh RandomState(0) per scored set',
                           schemes=list(SCHEMES), arms=list(ALL_ARMS) + ['ctrl_sample_h2 (H5 null only)'],
                           min_cand=s15c.MIN_CAND,
                           seeds=dict(pool_halving='RandomState(0) per pool', ctrl_sample=SEED_CTRL,
                                      ctrl_sample_h1=SEED_H1, ctrl_sample_h2=SEED_H2, duplicate=SEED_DUP),
                           reference=dict(boot=REF_BOOT, test_reference_override=bool(TEST_REF)),
                           verdict=dict(G_within_plate=vG, H_mechanism=vH, gates_by_split=gate_verdicts),
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
