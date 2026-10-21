#!/usr/bin/env python
"""Step 20, step 1: build the in-between pairings for the pairing-strength sweep, and measure them. CPU only; no
model is touched. Written 2026-10-11, before the Step 20 pre-registration and before any Step 20 training.

PURPOSE
Step 18 paired every treated training cell with the vehicle cell of its condition chosen by EXACT optimal transport
(one matching over the whole condition). The recipe pairs them at random. Step 20 trains the model on pairings in
between. The dial is the MINIBATCH SIZE m: inside each condition, the treated cells and the control cells are each
split at random into groups of about m, and the exact matching is made only inside each group.

  m = 1        each group holds one treated and one control cell: a random pairing, as the recipe
               (this follows from the construction; it is not a quotation from the sources below)
  m >= n       one group holds the whole condition: exact OT, as Step 18
  in between   a pairing partly matched, partly random

Every control cell is used exactly once at every m, so the set of conditioning cells the model sees is identical
across all settings; only who is paired with whom changes. Nothing else changes: same treated cells, same control
cells, same files layout, same training script (it reads the control file given by --control_data_path).

SOURCES for the dial (each checked against the original text on 2026-10-11):
  Fatras, Zine, Majewski, Flamary, Gribonval, Courty 2021, "Minibatch optimal transport distances; analysis and
    applications", arXiv:2101.01792, section 3.2.5: "the minibatch procedure allows us to interpolate between OT,
    when m=n and averaged pairwise distance, when m=1"; section 3.3: "the OT matrix densifies when m decreases,
    which is a similar effect to entropic regularization". (Preprint.)
  Fatras, Zine, Flamary, Gribonval, Courty 2020, "Learning with minibatch Wasserstein: asymptotic and gradient
    properties", AISTATS, PMLR 108:2131-2141: minibatch OT "is equivalent to an implicit regularization of the
    original problem".
  Pooladian et al. 2023, "Multisample flow matching", ICML, PMLR 202: the transport cost decreases with the batch
    size k (Theorem D.8) and approaches the optimal transport cost as k grows (Theorem 4.2).
  Tong et al. 2024, "Improving and generalizing flow-based generative models with minibatch optimal transport",
    TMLR: "when the OT batch size equals the support size ... we recover exact OT" (section 3.2.3).
  The exact matching inside each group is scipy.optimize.linear_sum_assignment, a modified Jonker-Volgenant
    algorithm (Crouse 2016, IEEE Trans. Aerosp. Electron. Syst. 52(4):1679-1696, doi 10.1109/TAES.2016.140952),
    on the plain Euclidean distance over the 200-gene log-normalised panel: the same cost and solver as Step 18
    (check_ot_pairing_tahoe.py), so the top of the sweep is exactly the Step 18 arm.

LOGIC
Measure mode (default): for every split and every training condition, compute
  - the recipe pairing as it is in the control file (the 0 % anchor),
  - the exact OT pairing over the whole condition (the 100 % anchor; Step 18),
  - the minibatch pairing for every candidate m in CANDIDATES,
and for each the two "how matched" numbers already used before Step 18:
  cost  mean Euclidean distance between a treated cell and its paired control cell (lower = closer pairs)
  cos   mean centred cosine between a treated cell and its paired control cell (higher = the paired control
        shares more of that particular cell's state; the quantity behind Step 18's tracking result)
Each per-split mean is an average over conditions, exactly as in check_ot_pairing_tahoe.py, and the two anchors
must reproduce the Step 18 record (tahoe_results/ot_pairing_check.json) to 1e-9 - proof that this script reads
the same data and computes the same matching.
Then, for each m, the fraction of the way from the recipe to full OT:
  f_cos(m)  = (cos(m)  - cos_recipe)  / (cos_OT  - cos_recipe)
  f_cost(m) = (cost_recipe - cost(m)) / (cost_recipe - cost_OT)

SELECTION RULE, fixed here before any output of this script exists: on the DEVELOPMENT split (drug_split_1) only,
for each target t in TARGETS = (0.25, 0.50, 0.75), in that order, choose the candidate m not already chosen whose
f_cos(m) is closest to t (ties to the smaller m). These three m values are the three Step 20 settings. The rule uses
pairing statistics only - no model, no outcome - so it can be fixed before the pre-registration is written.

Write mode (--write --split S --m M [M ...]): recompute the minibatch pairing for split S at each given m (the
random groups are seeded by split, condition name and m, so they are identical to the measured ones), check that
its mean cosine equals the measure record to 1e-12, and write <train>_control_mb<m>.h5ad with rows re-ordered within
each condition (the row names and the matrix multiset asserted identical to the original control file). Each file's
sha256 is pinned in tahoe_results/minibatch_pairing_written_<S>_m<m>.json. Write mode refuses unless the measure
record is committed and every m was chosen by the rule.

Usage (repo root, conda env active):
  python -u check_minibatch_pairing_tahoe.py                                       # measure + select
  python -u check_minibatch_pairing_tahoe.py --write --split drug_split_1 --m 4 8 16  # after committing the record
Never overwrites.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ME = 'check_minibatch_pairing_tahoe.py'
RES = 'tahoe_results'
RECORD = os.path.join(RES, 'step15_data_record.json')           # the certified data files (paths + sha256)
OTREC = os.path.join(RES, 'ot_pairing_check.json')              # the Step 18 pairing record (the anchors)
OUT = os.path.join(RES, 'minibatch_pairing_check.json')
OUT_CSV = os.path.join(RES, 'minibatch_pairing_check_per_condition.csv')
SPLITS = ('drug_split_1', 'drug_split_0', 'drug_split_2')       # development first
DEV = 'drug_split_1'
CANDIDATES = (2, 4, 8, 16, 32, 64, 128)                          # candidate minibatch sizes m
TARGETS = (0.25, 0.50, 0.75)                                     # fractions of the way from recipe to full OT
BASE_SEED = 20                                                   # Step 20
T0 = time.time()


# ------------------------------------------------------------------ helpers (sha256, dense, cosines: verbatim from
# check_ot_pairing_tahoe.py, so the anchors are computed exactly as in Step 18)
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def dense(X):
    return np.asarray(X.todense() if hasattr(X, 'todense') else X, dtype=np.float32)


def cosines(A, B):
    """Row-wise cosine between two matrices after centring each by its own column means."""
    Ac, Bc = A - A.mean(axis=0, keepdims=True), B - B.mean(axis=0, keepdims=True)
    num = (Ac * Bc).sum(axis=1)
    den = np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1)
    ok = den > 0
    return float(np.mean(num[ok] / den[ok])) if ok.any() else np.nan


def rng_for(split, group, m):
    """A random generator fixed by (Step 20, split, condition name, m) and nothing else, so the same groups are
    drawn in measure mode and in write mode, whatever order the conditions are visited in."""
    seq = np.random.SeedSequence([BASE_SEED, SPLITS.index(split), zlib.crc32(group.encode('utf-8')), int(m)])
    return np.random.RandomState(int(seq.generate_state(1)[0]))


def minibatch_assignment(D, m, rng):
    """Minibatch OT inside one condition. D is the n x n distance table (treated rows, control columns). The treated
    cells and the control cells are shuffled independently and cut into the same number of groups (sizes differ by
    at most one); the exact minimum-cost matching is solved inside each group. Returns a, where treated cell i is
    paired with control cell a[i]; a is a permutation, so every control cell is used exactly once."""
    n = D.shape[0]
    a = np.full(n, -1, dtype=np.int64)
    if m >= n:                                                   # one group: exact OT over the whole condition
        r_, c_ = linear_sum_assignment(D)
        a[r_] = c_
        return a
    k = int(np.ceil(n / m))                                      # number of groups
    tg = np.array_split(rng.permutation(n), k)                   # treated groups
    cg = np.array_split(rng.permutation(n), k)                   # control groups, same sizes as the treated ones
    for t_idx, c_idx in zip(tg, cg):
        r_, c_ = linear_sum_assignment(D[np.ix_(t_idx, c_idx)])
        a[t_idx[r_]] = c_idx[c_]
    if np.any(a < 0) or len(np.unique(a)) != n:
        sys.exit('FATAL: minibatch assignment is not a permutation')
    return a


def guard_committed(paths):
    joined = ' '.join(paths)
    if subprocess.run('git ls-files --error-unmatch ' + joined, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: not committed: %s' % joined)
    if subprocess.run('git diff --quiet -- ' + joined, shell=True).returncode != 0:
        sys.exit('FATAL: modified since commit: %s' % joined)


def load_split(rec, split, data_override):
    """Read the treated training file and its control file, check them against the data record, and return the
    dense matrices plus the condition index (rows of each condition, in file order)."""
    s = split.split('_')[-1]
    files = rec['splits'][s]['files']
    ft, fc = files['train']['path'], files['train_control']['path']
    if data_override:
        ft, fc = os.path.join(data_override, os.path.basename(ft)), os.path.join(data_override, os.path.basename(fc))
    for key, path in (('train', ft), ('train_control', fc)):
        if not os.path.isfile(path):
            sys.exit('FATAL: missing %s' % path)
        got = sha256(path)
        if got != files[key]['sha256']:
            sys.exit('FATAL: %s sha256 %s differs from the data record %s' % (path, got[:16], files[key]['sha256'][:16]))
    T = ad.read_h5ad(ft)
    C = ad.read_h5ad(fc)
    if T.n_obs != C.n_obs or T.n_vars != C.n_vars:
        sys.exit('FATAL: %s and its control file differ in shape' % split)
    for col in ('Group', 'cell_name', 'dose', 'SMILES'):
        if not np.array_equal(T.obs[col].astype(str).values, C.obs[col].astype(str).values):
            sys.exit('FATAL: %s: column %s differs between treated and control rows' % (split, col))
    groups = T.obs['Group'].astype(str).values
    codes, uniq = pd.factorize(pd.Series(groups), sort=False)
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return T, C, fc, files, uniq, order, bounds


# ------------------------------------------------------------------ measure mode
def measure(a, rec, otrec, commit):
    for q in (OUT, OUT_CSV):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    results, rows = {}, []
    for split in SPLITS:
        T, C, fc, files, uniq, order, bounds = load_split(rec, split, a.data)
        X, Xc = dense(T.X), dense(C.X)
        print('[%s] %d rows, %d conditions, %d genes, hashes match the data record (%.0f s)'
              % (split, T.n_obs, len(uniq), T.n_vars, time.time() - T0))
        keys = ['cost_recipe', 'cos_recipe', 'cost_ot', 'cos_ot'] + \
               ['%s_m%d' % (q, m) for m in (1,) + CANDIDATES for q in ('cost', 'cos')]
        agg = {k: [] for k in keys}
        ns = []
        for k in range(len(uniq)):
            idx = order[bounds[k]:bounds[k + 1]]
            n = len(idx)
            Tk, Ck = X[idx].astype(np.float64), Xc[idx].astype(np.float64)
            D = cdist(Tk, Ck)
            r_, c_ = linear_sum_assignment(D)                                   # full exact OT, as Step 18
            if not np.array_equal(r_, np.arange(n)):
                sys.exit('FATAL: assignment rows are not in order')
            vals = dict(cost_recipe=float(np.mean(np.diag(D))), cos_recipe=cosines(Tk, Ck),
                        cost_ot=float(D[r_, c_].mean()), cos_ot=cosines(Tk, Ck[c_]))
            for m in (1,) + CANDIDATES:
                am = minibatch_assignment(D, m, rng_for(split, str(uniq[k]), m))
                vals['cost_m%d' % m] = float(D[np.arange(n), am].mean())
                vals['cos_m%d' % m] = cosines(Tk, Ck[am])
            for key, v in vals.items():
                agg[key].append(v)
            ns.append(n)
            rows.append(dict(split=split, group=str(uniq[k]), n=int(n), **vals))
        A = {key: np.array(v, dtype=np.float64) for key, v in agg.items()}
        mean = {key: float(np.nanmean(A[key])) for key in A}                    # mean over conditions, as Step 18

        # reproduction: the two anchors equal the Step 18 record to 1e-9
        st18 = otrec['per_split'][split]['mean']
        for mine, theirs in (('cost_recipe', 'cost_random'), ('cos_recipe', 'cos_random'),
                             ('cost_ot', 'cost_ot'), ('cos_ot', 'cos_ot')):
            if abs(mean[mine] - float(st18[theirs])) > 1e-9:
                sys.exit('FATAL: %s %s %.12f does not reproduce the Step 18 record %s %.12f'
                         % (split, mine, mean[mine], theirs, float(st18[theirs])))
        gap_cos = mean['cos_ot'] - mean['cos_recipe']
        gap_cost = mean['cost_recipe'] - mean['cost_ot']
        if gap_cos <= 0 or gap_cost <= 0:
            sys.exit('FATAL: %s: full OT is not closer than the recipe (cos gap %.4g, cost gap %.4g)'
                     % (split, gap_cos, gap_cost))
        f_cos = {m: (mean['cos_m%d' % m] - mean['cos_recipe']) / gap_cos for m in (1,) + CANDIDATES}
        f_cost = {m: (mean['cost_recipe'] - mean['cost_m%d' % m]) / gap_cost for m in (1,) + CANDIDATES}
        ns = np.array(ns)
        print('   [reproduction] recipe and full-OT anchors equal the Step 18 record to 1e-9')
        print('   cells per condition min/median/max %d / %d / %d' % (ns.min(), np.median(ns), ns.max()))
        print('   %-10s %12s %12s %10s %10s' % ('pairing', 'distance', 'cosine', 'f_cos', 'f_cost'))
        print('   %-10s %12.6f %12.6f %10.3f %10.3f' % ('recipe', mean['cost_recipe'], mean['cos_recipe'], 0.0, 0.0))
        for m in (1,) + CANDIDATES:
            print('   %-10s %12.6f %12.6f %10.3f %10.3f'
                  % ('m = %d' % m, mean['cost_m%d' % m], mean['cos_m%d' % m], f_cos[m], f_cost[m]))
        print('   %-10s %12.6f %12.6f %10.3f %10.3f' % ('full OT', mean['cost_ot'], mean['cos_ot'], 1.0, 1.0))
        mono = all(f_cos[CANDIDATES[i]] <= f_cos[CANDIDATES[i + 1]] for i in range(len(CANDIDATES) - 1))
        print('   f_cos rises with m across the candidates: %s' % ('yes' if mono else 'NO'))
        results[split] = dict(train=dict(path=files['train']['path'], sha256=files['train']['sha256']),
                              train_control=dict(path=fc, sha256=files['train_control']['sha256']),
                              n_conditions=int(len(uniq)), n_rows=int(T.n_obs),
                              cells_per_condition=[int(ns.min()), int(np.median(ns)), int(ns.max())],
                              mean=mean, f_cos={str(m): v for m, v in f_cos.items()},
                              f_cost={str(m): v for m, v in f_cost.items()}, f_cos_monotone=bool(mono))
        del T, C, X, Xc

    # selection rule (docstring), on the development split only
    f_dev = {m: results[DEV]['f_cos'][str(m)] for m in CANDIDATES}
    chosen = []
    for t in TARGETS:
        best = min((m for m in CANDIDATES if m not in chosen), key=lambda m: (abs(f_dev[m] - t), m))
        chosen.append(best)
    print('\n[selection] development split %s, targets %s -> m = %s (f_cos %s)'
          % (DEV, list(TARGETS), chosen, [round(f_dev[m], 3) for m in chosen]))
    for split in SPLITS:
        print('            %s at the chosen m: f_cos %s, f_cost %s' % (
            split, [round(results[split]['f_cos'][str(m)], 3) for m in chosen],
            [round(results[split]['f_cost'][str(m)], 3) for m in chosen]))
    json.dump(dict(note='Step 20 step 1: minibatch OT pairings, measured; no model touched', script=ME,
                   record=RECORD, record_sha256=sha256(RECORD), step18_pairing_record=OTREC,
                   step18_pairing_record_sha256=sha256(OTREC), candidates=list(CANDIDATES), targets=list(TARGETS),
                   base_seed=BASE_SEED, selection=dict(split=DEV, chosen=chosen,
                                                       f_cos=[f_dev[m] for m in chosen]),
                   per_split=results, commit=commit, seconds=round(time.time() - T0)), open(OUT, 'w'), indent=1)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print('\nwrote %s and %s (%.0f s)' % (OUT, OUT_CSV, time.time() - T0))


# ------------------------------------------------------------------ write mode
def write(a, rec, commit):
    if a.split not in SPLITS:
        sys.exit('FATAL: unknown split %s' % a.split)
    if not os.path.exists(OUT):
        sys.exit('FATAL: run the measure mode first; %s does not exist' % OUT)
    guard_committed([OUT, OUT_CSV])
    mrec = json.load(open(OUT))
    for m in a.m:
        if m not in mrec['selection']['chosen']:
            sys.exit('FATAL: m = %d was not chosen by the selection rule (chosen: %s)' % (m, mrec['selection']['chosen']))
    T, C, fc, files, uniq, order, bounds = load_split(rec, a.split, a.data)
    if files['train_control']['sha256'] != mrec['per_split'][a.split]['train_control']['sha256']:
        sys.exit('FATAL: the control file differs from the one measured')
    X, Xc = dense(T.X), dense(C.X)
    for m in a.m:
        out_path = fc.replace('_control.h5ad', '_control_mb%d.h5ad' % m)
        pin = os.path.join(RES, 'minibatch_pairing_written_%s_m%d.json' % (a.split, m))
        if out_path == fc or os.path.exists(out_path) or os.path.exists(pin):
            sys.exit('FATAL: cannot write %s (or its record %s already exists)' % (out_path, pin))
        perm = np.arange(T.n_obs)
        coss = []
        for k in range(len(uniq)):
            idx = order[bounds[k]:bounds[k + 1]]
            Tk, Ck = X[idx].astype(np.float64), Xc[idx].astype(np.float64)
            am = minibatch_assignment(cdist(Tk, Ck), m, rng_for(a.split, str(uniq[k]), m))
            perm[idx] = idx[am]                                  # treated row idx[i] <- control row idx[am[i]]
            coss.append(cosines(Tk, Ck[am]))
        got = float(np.nanmean(coss))
        want = float(mrec['per_split'][a.split]['mean']['cos_m%d' % m])
        if abs(got - want) > 1e-12:
            sys.exit('FATAL: the recomputed pairing (cos %.15f) is not the measured one (%.15f)' % (got, want))
        Cmb = C[perm].copy()
        if not np.array_equal(np.sort(Cmb.obs_names.values), np.sort(C.obs_names.values)):
            sys.exit('FATAL: the re-ordered control file does not hold the same rows')
        if abs(float(dense(Cmb.X).sum()) - float(Xc.sum())) > 1e-3 * max(1.0, abs(float(Xc.sum()))):
            sys.exit('FATAL: the re-ordered control matrix differs in total')
        for col in ('Group', 'cell_name'):
            if not np.array_equal(Cmb.obs[col].astype(str).values, T.obs[col].astype(str).values):
                sys.exit('FATAL: re-ordered control rows do not align with the treated rows on %s' % col)
        Cmb.write_h5ad(out_path)
        info = dict(split=a.split, m=int(m), path=out_path, sha256=sha256(out_path), rows=int(Cmb.n_obs),
                    source_control_sha256=files['train_control']['sha256'], measured_cos=want,
                    measure_record=OUT, measure_record_sha256=sha256(OUT), script=ME, commit=commit,
                    seconds=round(time.time() - T0))
        json.dump(info, open(pin, 'w'), indent=1)
        print('[write] %s m = %d: %s (%s); mean cosine %.6f equals the measure record'
              % (a.split, m, out_path, info['sha256'][:16], got))
        del Cmb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='write the control files for --split at the given --m')
    ap.add_argument('--split', default=None)
    ap.add_argument('--m', type=int, nargs='+', default=None)
    ap.add_argument('--data', default=None, help='override the data directory (synthetic tests only)')
    a = ap.parse_args()
    guard_committed([ME, RECORD, OTREC])
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    rec = json.load(open(RECORD))
    otrec = json.load(open(OTREC))
    print('[inputs] %s and %s committed; this commit %s' % (RECORD, OTREC, commit[:7]))
    if a.write:
        if not a.split or not a.m:
            sys.exit('FATAL: --write needs --split and --m')
        write(a, rec, commit)
    else:
        if a.split or a.m:
            sys.exit('FATAL: --split and --m are for --write only')
        measure(a, rec, otrec, commit)


if __name__ == '__main__':
    main()
