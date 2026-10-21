#!/usr/bin/env python3
"""
Step 22, stage 1 of 4: build the cell matrices that the RNA-fingerprinting package will read
(PREREG_step22_fingerprint_identifiability.md, sections 2, 3.2, 3.3 and 9).

What it does, per split
  1. Loads the certified arrays through the Step 18 / 20 / 21 loaders (every hash checked against provenance): the
     recipe (s18r_*), full OT (s18_*) and m = 32 (s21_* on splits 0 and 2; s20_* on the development split, where
     Step 20 trained m = 32), three seeds each. Asserts that every run carries the same conditions, the same stored
     real cells and the same conditioning vehicle cells (bitwise), as Steps 18-21 did.
  2. Rebuilds the four Step 15b references by the Step 15b code path (score_step16_tahoe.rebuild) and asserts that
     their DE-overlap and DPC-50 reproduce the Step 15b per-condition record to 1e-9 (as Step 18). Reference cells =
     the conditioning vehicle cells + the reference's delta, clipped at 0 (Step 15b section 3).
  3. Draws the halves (prereg 3.3): half A / half B of each condition's real cells (RandomState(22)); the unique
     vehicle cells split into control-A / control-B (RandomState(23)); for every query set, |B_i| cells per condition
     (RandomState(24), fresh per set); the first C_CELL = 10 of each condition's query cells are flagged for the
     cell-level mapping.
  4. Standardises every cell per gene with the mean and standard deviation of the dictionary cells (half A +
     control-A), clipped at +-10 (prereg 3.2), and writes float32 row-major matrices (cells x genes) with a metadata
     CSV each, the gene list and manifest.json (seeds, sizes, parameters, label map, half-A indices, sha256 of every
     file, the provenance of every loaded run).

Outputs: tahoe_results/step22/<split>/  dictionary.f32 + dictionary_meta.csv, control_B.f32 + control_B_meta.csv,
         query_<set>.f32 + query_<set>_meta.csv for the fourteen sets, genes.txt, manifest.json. Never overwrites.
--mock:  writes a synthetic split with planted fingerprints to tahoe_results/step22_mock/mock_split/ in the same
         layout (no loaders, no guards) for the end-to-end test of the R stage and the scorer.

Run (inside run_step22_tahoe.sh, squidiff env):  python -u prep_step22_fingerprint.py --split drug_split_1
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402  sha, check_prov, thesis_per_cond, RES, RECORD, CAP
import score_step15c_tahoe as s15c                    # noqa: E402  close
import score_step16_tahoe as s16                      # noqa: E402  rebuild (the Step 15b references), PC15
import score_step18_tahoe as s18                      # noqa: E402  load_eval18, OTREC
import score_step20_tahoe as s20                      # noqa: E402  load_eval20, pin_path (m = 32 on the development split)
import score_step21_tahoe as s21                      # noqa: E402  load_eval21, guard_tracked (m = 32 on splits 0 and 2)

PRE = 'PREREG_step22_fingerprint_identifiability.md'
ME = 'prep_step22_fingerprint.py'
OUT_ROOT = os.path.join(s15.RES, 'step22')
MOCK_ROOT = os.path.join(s15.RES, 'step22_mock')
SPLITS = ('drug_split_1', 'drug_split_0', 'drug_split_2')
DEV = 'drug_split_1'
M = 32                                                # the minibatch setting confirmed in Step 21
N_SEEDS = s15.N_SEEDS
SEED_HALF, SEED_VEHICLE, SEED_QUERY = 22, 23, 24      # prereg 3.3
C_CELL = 10                                           # prereg 3.3: cells per condition at cell level
CLIP = 10.0                                           # prereg 3.2: Seurat ScaleData's scale.max
MIN_CONTROL = 30                                      # the package's minimum number of control cells in a query
ARMS = ('recipe', 'ot_pair', 'mb32')
REFS = ('control_paste', 'mean_response', 'ridge_train', 'ridge_S7')
SETS_GROUP = ['real_half'] + ['%s_seed_%d' % (a, s) for a in ARMS for s in range(N_SEEDS)] + list(REFS)
SETS_CELL = [x for x in SETS_GROUP if x not in ('ridge_train', 'ridge_S7')]       # prereg 3.3: twelve sets
T0 = time.time()


# ------------------------------------------------------------------ small helpers
def log(msg):
    print('[%6.0fs] %s' % (time.time() - T0, msg), flush=True)


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_matrix(path, X):
    """float32, row-major, cells x genes; R reads it back as genes x cells with readBin + matrix()."""
    X = np.ascontiguousarray(X, dtype='<f4')
    X.tofile(path)
    return dict(path=os.path.basename(path), n_rows=int(X.shape[0]), n_cols=int(X.shape[1]), sha256=s15.sha(path))


def write_meta(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return dict(path=os.path.basename(path), n_rows=len(rows), sha256=s15.sha(path))


def standardiser(mu, sd):
    """Prereg 3.2: per-gene centring and scaling on the dictionary cells, clipped at +-CLIP (Seurat ScaleData)."""
    def f(X):
        return np.clip((np.asarray(X, np.float64) - mu) / sd, -CLIP, CLIP)
    return f


# ------------------------------------------------------------------ loaders (the certified arrays)
def load_arm(split, arm, seed, ot_sha, mb_sha):
    if arm == 'recipe':
        return s18.load_eval18(split, seed, 'recipe', ot_sha)
    if arm == 'ot_pair':
        return s18.load_eval18(split, seed, 'ot', ot_sha)
    if arm == 'mb32':
        if split == DEV:
            return s20.load_eval20(split, M, seed, mb_sha)      # Step 20 trained m = 32 on the development split
        return s21.load_eval21(split, seed, mb_sha)             # Step 21 trained it on the confirmation splits
    sys.exit('FATAL: unknown arm %s' % arm)


def check_shared(v, cond, real, ctrl_ref, C, true_mean):
    """Every run must carry the comparators' conditions, real cells and conditioning vehicle cells (Step 21 rule)."""
    if not np.array_equal(v['cond'], cond):
        sys.exit('FATAL: condition order differs in %s' % v['tag'])
    if np.abs(v['C'] - C).max() > 1e-5 or np.abs(v['true'] - true_mean).max() > 1e-5:
        sys.exit('FATAL: control or true means differ in %s' % v['tag'])
    for i in range(len(cond)):
        if not np.array_equal(v['true_cells'][i], real[i]):
            sys.exit('FATAL: stored real cells differ in %s, condition %d' % (v['tag'], i))
        if not np.array_equal(v['ctrl'][i], ctrl_ref[i]):
            sys.exit('FATAL: %s was conditioned on different vehicle cells than the comparators at condition %d' % (v['tag'], i))
        if v['pred'][i].shape != real[i].shape:
            sys.exit('FATAL: %s generated %s cells at condition %d, real %s' % (v['tag'], v['pred'][i].shape, i, real[i].shape))


# ------------------------------------------------------------------ the writer shared by the real and the mock path
def build_split(out, split, cond, real, ctrl_ref, ann, arm_cells, ref_cells, genes, inputs, extra):
    """Draw the halves, standardise, write every matrix and the manifest (prereg 3.2-3.3).
    real[i]: stored real cells of condition i (n_i x G); ctrl_ref[i]: its conditioning vehicle cells (n_i x G);
    ann: per-condition annotations (structure SMILES, line, plate, dose); arm_cells[set](i) -> generated cells of
    condition i; ref_cells[set][i] -> reference cells of condition i."""
    n, G = len(cond), len(genes)
    drugs = ann['structure']
    smi_sorted = sorted(set(drugs))
    label = {smi: 'P%03d' % (k + 1) for k, smi in enumerate(smi_sorted)}      # package-safe names (prereg 3.3)

    # ---- half A / half B of the real cells (one generator per split, advanced over conditions in order)
    rng = np.random.RandomState(SEED_HALF)
    A, B = [], []
    for i in range(n):
        perm = rng.permutation(len(real[i]))
        A.append(perm[:len(perm) // 2])                 # dictionary half (order kept as drawn)
        B.append(perm[len(perm) // 2:])                 # query half; its first C_CELL cells go to the cell level
    nA, nB = np.array([len(a) for a in A]), np.array([len(b) for b in B])
    if nA.min() < 1:
        sys.exit('FATAL: a condition has too few stored cells for the halves (min A %d, min B %d)' % (nA.min(), nB.min()))
    if nB.min() < C_CELL:                              # cannot happen on the real data (43-200 stored cells); the mock may be smaller
        log('WARNING: %d conditions have fewer than %d query cells; they enter the cell level with all of them' % (int((nB < C_CELL).sum()), C_CELL))

    # ---- unique vehicle cells, split into control-A (dictionary) and control-B (every query)
    V = np.unique(np.vstack(ctrl_ref), axis=0)
    n_dup = int(sum(len(c) for c in ctrl_ref) - len(V))
    permv = np.random.RandomState(SEED_VEHICLE).permutation(len(V))
    cA, cB = permv[:len(V) // 2], permv[len(V) // 2:]
    if len(cA) < MIN_CONTROL or len(cB) < MIN_CONTROL:
        sys.exit('FATAL: fewer than %d unique vehicle cells in a half (%d, %d)' % (MIN_CONTROL, len(cA), len(cB)))
    log('%s: %d conditions, %d structures, real cells per condition %d-%d, half A %d-%d, half B %d-%d; vehicle cells %d '
        'stored, %d unique (%d duplicates), control-A %d, control-B %d'
        % (split, n, len(smi_sorted), min(len(r) for r in real), max(len(r) for r in real), nA.min(), nA.max(), nB.min(),
           nB.max(), sum(len(c) for c in ctrl_ref), len(V), n_dup, len(cA), len(cB)))

    # ---- standardisation on the dictionary cells (prereg 3.2)
    Xd = np.vstack([real[i][A[i]] for i in range(n)] + [V[cA]]).astype(np.float64)
    mu, sd = Xd.mean(axis=0), Xd.std(axis=0, ddof=1)
    if not np.all(np.isfinite(sd)) or sd.min() <= 0:
        sys.exit('FATAL: a gene has zero or undefined standard deviation on the dictionary cells')
    f = standardiser(mu, sd)
    os.makedirs(out, exist_ok=False)
    with open(os.path.join(out, 'genes.txt'), 'w') as fh:
        fh.write('\n'.join(genes) + '\n')
    files = {}
    meta_fields = ['cell_id', 'label', 'condition', 'structure', 'line', 'plate', 'dose_nM', 'role', 'cell_level']

    # ---- dictionary: half A labelled by structure + control-A labelled control
    rows = []
    for i in range(n):
        for j in A[i]:
            rows.append(dict(cell_id='%s|A|%d' % (cond[i], j), label=label[drugs[i]], condition=cond[i], structure=drugs[i],
                             line=ann['line'][i], plate=ann['plate'][i], dose_nM=ann['dose_nM'][i], role='treated', cell_level=0))
    for k in cA:
        rows.append(dict(cell_id='vehicle|A|%d' % k, label='control', condition='control', structure='control', line='', plate='',
                         dose_nM='', role='control', cell_level=0))
    files['dictionary'] = dict(matrix=write_matrix(os.path.join(out, 'dictionary.f32'), f(Xd)),
                               meta=write_meta(os.path.join(out, 'dictionary_meta.csv'), rows, meta_fields))
    del Xd
    log('dictionary written: %d treated + %d control cells' % (int(nA.sum()), len(cA)))

    # ---- control-B, shared by every query object
    rows = [dict(cell_id='vehicle|B|%d' % k, label='control', condition='control', structure='control', line='', plate='',
                 dose_nM='', role='control', cell_level=0) for k in cB]
    files['control_B'] = dict(matrix=write_matrix(os.path.join(out, 'control_B.f32'), f(V[cB])),
                              meta=write_meta(os.path.join(out, 'control_B_meta.csv'), rows, meta_fields))
    del V

    # ---- query sets: |B_i| cells per condition; the first C_CELL flagged for the cell level
    draws = {}
    for name in SETS_GROUP:
        rngq = np.random.RandomState(SEED_QUERY)      # fresh per set (prereg 3.3)
        blocks, rows, idx_sha = [], [], hashlib.sha256()
        for i in range(n):
            if name == 'real_half':
                cells, sel, src = real[i], B[i], 'B'
            else:
                src_cells = arm_cells[name](i) if name in arm_cells else ref_cells[name][i]
                if len(src_cells) < nB[i]:
                    sys.exit('FATAL: %s has %d cells at condition %d, fewer than |B| = %d' % (name, len(src_cells), i, nB[i]))
                sel = rngq.permutation(len(src_cells))[:nB[i]]
                cells, src = src_cells, 'Q'
            idx_sha.update(np.asarray(sel, np.int64).tobytes())
            blocks.append(f(cells[sel]))
            for r, j in enumerate(sel):
                rows.append(dict(cell_id='%s|%s|%d' % (cond[i], src, j), label=label[drugs[i]], condition=cond[i], structure=drugs[i],
                                 line=ann['line'][i], plate=ann['plate'][i], dose_nM=ann['dose_nM'][i], role='treated',
                                 cell_level=int(r < C_CELL)))
        files['query_' + name] = dict(matrix=write_matrix(os.path.join(out, 'query_%s.f32' % name), np.vstack(blocks)),
                                      meta=write_meta(os.path.join(out, 'query_%s_meta.csv' % name), rows, meta_fields))
        draws[name] = dict(seed=SEED_QUERY, indices_sha256=idx_sha.hexdigest(), n_rows=len(rows))
        log('query set %-16s written: %d cells (%d at cell level)' % (name, len(rows), sum(min(C_CELL, b) for b in nB)))

    manifest = dict(step=22, prereg=PRE, split=split, written=time.strftime('%Y-%m-%d %H:%M:%S'), n_conditions=n, n_genes=G,
                    n_structures=len(smi_sorted), label_map=label, structures_sorted=smi_sorted,
                    seeds=dict(half=SEED_HALF, vehicle=SEED_VEHICLE, query=SEED_QUERY), c_cell=C_CELL, clip=CLIP,
                    sets_group=SETS_GROUP, sets_cell=SETS_CELL,
                    vehicle=dict(stored=int(sum(len(c) for c in ctrl_ref)), unique=int(len(cA) + len(cB)), duplicates=n_dup,
                                 control_A=int(len(cA)), control_B=int(len(cB))),
                    standardisation=dict(fit_on='half A + control-A', ddof=1, mu=[float(x) for x in mu], sd=[float(x) for x in sd]),
                    conditions=[dict(condition=cond[i], label=label[drugs[i]], structure=drugs[i], line=ann['line'][i],
                                     plate=ann['plate'][i], dose_nM=ann['dose_nM'][i], n_real=int(len(real[i])), n_A=int(nA[i]),
                                     n_B=int(nB[i]), half_A=[int(j) for j in A[i]]) for i in range(n)],
                    query_draws=draws, files=files, inputs=inputs, **extra)
    with open(os.path.join(out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    log('manifest written: %s' % os.path.join(out, 'manifest.json'))


# ------------------------------------------------------------------ the real path
def real_path(split):
    out = os.path.join(OUT_ROOT, split)
    if os.path.exists(out):
        sys.exit('FATAL: %s exists (never overwritten)' % out)
    if open(PRE).read().count('\n## ') < 10:
        sys.exit('FATAL: prereg incomplete')
    mb_pin = s20.pin_path(split, M)
    tracked = [PRE, ME, 'score_step15_tahoe.py', 'score_step15c_tahoe.py', 'score_step16_tahoe.py', 'score_step18_tahoe.py',
               'score_step20_tahoe.py', 'score_step21_tahoe.py', 'Squidiff/scrna_datasets.py', s16.PC15, s18.OTREC, s15.RECORD, mb_pin]
    s21.guard_tracked(tracked)
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    record = json.load(open(s15.RECORD))
    otrec = json.load(open(s18.OTREC))
    ot_sha = otrec['written'][split]['sha256']
    mb_sha = json.load(open(mb_pin))['sha256']
    log('[prereg] committed; scripts committed; commit %s; split %s; m = 32 arrays from %s' % (commit[:7], split, 'Step 20' if split == DEV else 'Step 21'))

    # ---- the comparators' cells: recipe seed 0 defines the conditions, the real cells and the vehicle cells
    ref = load_arm(split, 'recipe', 0, ot_sha, mb_sha)
    cond, real, ctrl_ref, C, true_mean = ref['cond'], ref['true_cells'], ref['ctrl'], ref['C'], ref['true']
    n = len(cond)
    rec_split = record['splits'][split.split('_')[-1]]
    if n != int(rec_split['n_test_conditions']):
        sys.exit('FATAL: %d conditions, record expects %d' % (n, rec_split['n_test_conditions']))
    for i in range(n):
        if ctrl_ref[i].shape != real[i].shape:
            sys.exit('FATAL: vehicle / real cell counts differ at condition %d' % i)
    pc = pd.read_csv(s16.PC15)
    pc = pc[pc['split'] == split].reset_index(drop=True)
    if len(pc) != n or not np.array_equal(pc['condition'].astype(str).values, cond):
        sys.exit('FATAL: the Step 15b per-condition record does not list the conditions of %s in the arrays order' % split)
    ann = dict(structure=pc['structure'].astype(str).values, line=pc['line'].astype(str).values,
               plate=pc['plate'].astype(str).values, dose_nM=pc['dose_nM'].astype(float).values)
    if len(set(ann['structure'])) != len(rec_split['held_structures']):
        sys.exit('FATAL: %d structures, record expects %d' % (len(set(ann['structure'])), len(rec_split['held_structures'])))
    import anndata as ad
    genes = [str(g) for g in ad.read_h5ad('%s/tahoe_test_%s.h5ad' % (s15.DATA, split), backed='r').var_names]
    if len(genes) != real[0].shape[1]:
        sys.exit('FATAL: %d gene names, arrays have %d genes' % (len(genes), real[0].shape[1]))
    inputs = {'recipe_seed_0': ref['rec']}

    # ---- the arms: loaded one at a time, checked, kept as per-condition cells
    arm_cells = {}
    for arm in ARMS:
        for s in range(N_SEEDS):
            v = ref if (arm == 'recipe' and s == 0) else load_arm(split, arm, s, ot_sha, mb_sha)
            check_shared(v, cond, real, ctrl_ref, C, true_mean)
            inputs['%s_seed_%d' % (arm, s)] = v['rec']
            pred = v['pred']
            arm_cells['%s_seed_%d' % (arm, s)] = (lambda P: (lambda i: P[i]))(pred)
            log('loaded %s (%s): shared conditions, real cells and vehicle cells confirmed bitwise' % (v['tag'], arm))
            del v
    del ref

    # ---- the references by the Step 15b code path, asserted against the Step 15b record (as Step 18)
    lv = sorted(set(ann['line']))
    Y = true_mean - C
    deltas, _ = s16.rebuild(split, ann['line'], ann['structure'], ann['dose_nM'], lv, Y)
    for bname, D in deltas.items():
        j, d = s15.thesis_per_cond(true_mean, C + D, C)
        if not s15c.close(j, pc['%s_de_overlap' % bname].values) or not s15c.close(d, pc['%s_dpc50' % bname].values):
            sys.exit('FATAL: %s does not reproduce the Step 15b per-condition record' % bname)
    log('[reproduction] the four references reproduce the Step 15b per-condition record (DE-overlap, DPC-50) to 1e-9')
    ref_cells = {bname: [np.clip(ctrl_ref[i] + D[i][None, :], 0, None).astype(np.float32) for i in range(n)] for bname, D in deltas.items()}
    clipped = {bname: float(np.mean([(ctrl_ref[i] + D[i][None, :] < 0).mean() for i in range(n)])) for bname, D in deltas.items()}
    extra = dict(commit=commit, ot_sha256=ot_sha, mb32_sha256=mb_sha, reference_clipped_fraction=clipped,
                 records=dict(step15b_per_condition=s15.sha(s16.PC15), ot_pairing=s15.sha(s18.OTREC), data_record=s15.sha(s15.RECORD),
                              mb32_pin=s15.sha(mb_pin)))
    build_split(out, split, cond, real, ctrl_ref, ann, arm_cells, ref_cells, genes, inputs, extra)


# ------------------------------------------------------------------ the mock path (end-to-end test of the R stage and the scorer)
def mock_path():
    """A synthetic split with planted fingerprints: 12 structures x 4 lines x 2 doses x 2 plates = 192 conditions, 40 real
    cells each, 200 genes. Control cells of line l ~ N(line mean, 1). Treated cell = control-like draw + dose factor x
    (shared response + the compound's sparse signature) + noise. Arms carry a fraction of that signal (recipe 0.8, m = 32
    0.7, full OT 0.6); ridge-S7 0.7 and ridge-train 0.5 of it on the vehicle cells; mean_response = vehicle + the line's
    mean measured delta; control_paste = the vehicle cells. Expected: real_half and recipe recognised far above chance,
    control_paste near chance, mean_response in between; verdict 1 on the mock."""
    out = os.path.join(MOCK_ROOT, 'mock_split')
    if os.path.exists(out):
        sys.exit('FATAL: %s exists (remove the mock folder to rerun)' % out)
    rs = np.random.RandomState(2222)
    G, n_struct, lines, doses, plates, n_cells = 200, 12, ['L1', 'L2', 'L3', 'L4'], [1000.0, 5000.0], ['6', '14'], 40
    line_mu = {ln: rs.normal(0, 2.0, G) for ln in lines}
    shared = rs.normal(0, 1.0, G)                                   # the response every compound shares (the line's mean)
    sig = {}
    for k in range(n_struct):
        v = np.zeros(G)
        g = rs.choice(G, 20, replace=False)
        v[g] = rs.choice([-2.0, 2.0], 20)
        sig['S%02d' % k] = v
    structs = sorted(sig)
    cond, real, ctrl_ref, ann = [], [], [], dict(structure=[], line=[], plate=[], dose_nM=[])
    arm_frac = {'recipe': 0.8, 'mb32': 0.7, 'ot_pair': 0.6}
    arm_cells_raw = {'%s_seed_%d' % (a, s): [] for a in arm_frac for s in range(N_SEEDS)}
    ref_raw = {r: [] for r in REFS}
    for ln in lines:
        for smi in structs:
            for dose in doses:
                for pl in plates:
                    c = '%s_%s_%g_p%s' % (ln, smi, dose / 10000, pl)
                    df = 0.6 if dose < 2000 else 1.0
                    base = line_mu[ln] + 0.3 * float(pl == '14')     # a small plate offset
                    ctrl = np.clip(base + rs.normal(0, 1.0, (n_cells, G)), 0, None)
                    treated = np.clip(base + rs.normal(0, 1.0, (n_cells, G)) + df * (shared + sig[smi]), 0, None)
                    cond.append(c); real.append(treated.astype(np.float32)); ctrl_ref.append(ctrl.astype(np.float32))
                    ann['structure'].append(smi); ann['line'].append(ln); ann['plate'].append(pl); ann['dose_nM'].append(dose)
                    for a, fr in arm_frac.items():
                        for s in range(N_SEEDS):
                            gen = np.clip(ctrl + fr * df * (shared + sig[smi]) + rs.normal(0, 0.7, (n_cells, G)), 0, None)
                            arm_cells_raw['%s_seed_%d' % (a, s)].append(gen.astype(np.float32))
                    ref_raw['control_paste'].append(ctrl.astype(np.float32))
                    ref_raw['ridge_S7'].append(np.clip(ctrl + 0.7 * df * (shared + sig[smi]), 0, None).astype(np.float32))
                    ref_raw['ridge_train'].append(np.clip(ctrl + 0.5 * df * (shared + sig[smi]), 0, None).astype(np.float32))
    ann = {k: np.array(v) for k, v in ann.items()}
    n = len(cond)
    Y = np.array([real[i].mean(0) - ctrl_ref[i].mean(0) for i in range(n)], np.float64)
    for i in range(n):                                                # mean_response as Step 15b: the line's mean delta
        mr = Y[ann['line'] == ann['line'][i]].mean(axis=0)
        ref_raw['mean_response'].append(np.clip(ctrl_ref[i] + mr[None, :], 0, None).astype(np.float32))
    arm_cells = {k: (lambda P: (lambda i: P[i]))(v) for k, v in arm_cells_raw.items()}
    genes = ['G%03d' % g for g in range(G)]
    truth = dict(planted=dict(signal_genes_per_structure=20, effect=2.0, shared_scale=1.0, dose_factors=[0.6, 1.0],
                              arm_fractions=arm_frac, ridge_S7=0.7, ridge_train=0.5),
                 expectation='real_half and recipe far above chance; control_paste near chance; verdict 1 (cell level)')
    build_split(out, 'mock_split', np.array(cond), real, ctrl_ref, ann, arm_cells, ref_raw, genes, {'mock': True}, dict(mock=truth))
    log('mock split written under %s' % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default=None, help='one of %s' % (SPLITS,))
    ap.add_argument('--mock', action='store_true', help='write the synthetic split for the end-to-end test')
    a = ap.parse_args()
    if a.mock == (a.split is not None):
        sys.exit('FATAL: give exactly one of --split <split> or --mock')
    if a.mock:
        mock_path()
    else:
        if a.split not in SPLITS:
            sys.exit('FATAL: unknown split %s' % a.split)
        real_path(a.split)
    log('done')


if __name__ == '__main__':
    main()
