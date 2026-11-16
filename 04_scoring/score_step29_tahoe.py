#!/usr/bin/env python3
"""Step 29, Tahoe-100M part: do the statements that compare something with the cross-validated ridge depend on
the fold assignment?   (PREREG_step29_fold_sensitivity.md, items B1 to B4 and C1 to C3)

Purpose
-------
The cross-validated ridge is fitted within the held-out structures with five folds. The folds of the paper are
one assignment among many that the same rule can give (section 1 of the pre-registration; on Tahoe-100M they
were identified by diag_step29_tahoe_folds.py and are stored in tahoe_results/step29_tahoe_folds.json). This
script refits the ridge with the folds of the paper and with random assignments, and counts, for every
statement of the paper about Tahoe-100M that involves this ridge, how many assignments give the verdict the
paper states. No generative model is retrained or re-sampled, and no value of the paper is replaced.

Logic
-----
For each compound split:
 1. Read what does not depend on the folds, from committed records only: conditions, cell line, structure and
    dose, and the per-condition values of the rewired model, the original model and the compound-blind
    average (step15_model_per_condition.csv, step15c_pds_per_condition.csv). Read the measured change of every
    condition from the stored vectors (hash-checked), and the measured cells and their paired vehicle cells
    from the test files, drawn exactly as the evaluation drew them (file order, RandomState(0), at most 200).
 2. Refit the ridge for one fold assignment as score_step15_tahoe.py does (fingerprint and dose code, cell
    line, fingerprint x cell line; penalty 1; double precision), and compute per condition, with the code of
    the paper: gene identity and response correlation (score_step15_tahoe.thesis_per_cond), population
    distance (vehicle cells shifted by the predicted change, clipped at zero, energy distance to the measured
    cells), the discrimination score (score_step15c_tahoe.pds_sets, its candidate sets) and the size of the
    predicted change for response-magnitude agreement (score_step16_tahoe.py).
 3. Intervals are those of the paper: 97.5 % percentile bootstrap, 5,000 replicates, resampling held-out
    structures. Each is drawn from a fresh RandomState(0), so that every assignment meets the same resamples
    (pre-registration, section 3). The paper drew the intervals of gene identity, population distance and the
    discrimination score from one generator shared between contrasts, so a committed interval can differ from
    the recomputed one in the third decimal; both are printed for the folds of the paper. The interval of
    response-magnitude agreement was drawn from a fresh generator in the paper too and must reproduce exactly.
 4. Guards before any new value is printed (pre-registration, section 7, guards 1 and 5): no output exists;
    no tracked file is modified; this script, the pre-registration and the records are committed; the folds
    record has the expected sha256 and its structures are those of the data; and, with the folds of the paper,
    the ridge reproduces to 1e-9 the committed per-condition gene identity, response correlation and population
    distance (Step 15b), discrimination score (Step 15c) and the response-magnitude agreement with its
    interval (Step 16). If a guard fails nothing is written.
 5. The random assignments (200; seeds 20261113 onwards; the rule of the pre-registration, section 2) are
    scored in worker processes, one thread each. The folds of the paper are scored in the main process with
    the threads of the job.
 6. For every item and split: the number of random assignments that give the verdict the paper states; the
    item is robust on the split if that is at least 95 %.

Items (what is counted; section 4 of the pre-registration)
  B1  gene identity, compound-blind average minus ridge: positive
  B2  gene identity, original model minus ridge: negative
  B3  discrimination score of the ridge minus 0.5: interval includes zero
  B4  response-magnitude agreement of the ridge: interval includes zero
  C1  gene identity, rewired model minus ridge: interval includes zero
  C2  population distance, rewired model minus ridge: positive; interval above zero on Splits 1 and 2,
      includes zero on Split 0
  C3  discrimination score, rewired model minus ridge: positive, interval above zero

Outputs (written once, at the end; never overwritten)
  tahoe_results/step29_fold_sensitivity_tahoe.json
  tahoe_results/step29_fold_sensitivity_tahoe_per_assignment.csv

Run:  sbatch run_step29_tahoe.sh
"""
import os
import sys
sys.dont_write_bytecode = True          # leave no __pycache__ behind in the worktree

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)                                             # the Squidiff package of this worktree

RES = "tahoe_results"
DATA_DIR = "/scratch/ni2071/Diffusion/tahoe/datasets_step15"        # as score_step15_tahoe.py
PRE = "PREREG_step29_fold_sensitivity.md"
REC_FOLDS = os.path.join(RES, "step29_tahoe_folds.json")            # the folds of the paper (guard 4)
FOLDS_SHA256 = "a6e36a3f1de054704526ec0ad10b92c83226765e65ea244b93db645fee7bdd82"
PC15 = os.path.join(RES, "step15_model_per_condition.csv")          # Step 15b, per condition
REC15 = os.path.join(RES, "step15_model_analysis.json")
PC15C = os.path.join(RES, "step15c_pds_per_condition.csv")          # Step 15c, per condition
REC15C = os.path.join(RES, "step15c_pds.json")
REC16 = os.path.join(RES, "step16_analysis.json")                   # Step 16 (response-magnitude agreement)
TRACKED = [PRE, REC_FOLDS, PC15, REC15, PC15C, REC15C, REC16]
OUT_JSON, OUT_CSV = "step29_fold_sensitivity_tahoe.json", "step29_fold_sensitivity_tahoe_per_assignment.csv"

SPLITS = ("drug_split_1", "drug_split_0", "drug_split_2")
NFOLD, ALPHA, K, CAP = 5, 1.0, 50, 200                               # score_step15_tahoe.py
N_BOOT, LO, HI = 5000, 1.25, 98.75                                    # 97.5 % per split
MIN_CAND, MIN_CTX = 10, 10                                            # score_step15c_tahoe.py, score_step16_tahoe.py
TOL, THRESHOLD = 1e-9, 0.95
N_RANDOM, SEED0 = 200, 20261113                                       # pre-registration, section 2
REWIRED, ORIGINAL, BLIND, RIDGE = "recipe", "base", "mean_response", "ridge_S7"   # names in the records
ITEMS = ("B1", "B2", "B3", "B4", "C1", "C2", "C3")
# The verdict the paper states, per item and split (pre-registration, section 4).
STATED = {
    "B1": {sp: dict(sign="positive") for sp in SPLITS},
    "B2": {sp: dict(sign="negative") for sp in SPLITS},
    "B3": {sp: dict(ci="includes zero") for sp in SPLITS},
    "B4": {sp: dict(ci="includes zero") for sp in SPLITS},
    "C1": {sp: dict(ci="includes zero") for sp in SPLITS},
    "C2": {"drug_split_1": dict(sign="positive", ci="above zero"), "drug_split_0": dict(sign="positive", ci="includes zero"),
           "drug_split_2": dict(sign="positive", ci="above zero")},
    "C3": {sp: dict(sign="positive", ci="above zero") for sp in SPLITS},
}
D = {}                                                                # data of the splits; inherited by the workers


def die(msg):
    sys.exit("FATAL: " + msg)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:
        return ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version_of(package):
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return "?"


# ----------------------------------------------------------------------------- the paper's measures, verbatim
def verdict(lo, hi):                                                  # score_step15_tahoe.py
    return "above zero" if lo > 0 else ("below zero" if hi < 0 else "includes zero")


def jac(a, b):                                                        # score_step15_tahoe.py
    i = len(np.intersect1d(a, b))
    return i / (len(a) + len(b) - i)


def thesis_per_cond(true, B, C):                                      # score_step15_tahoe.py, verbatim
    """DE-overlap Jaccard and DPC-50 per condition from mean vectors (true, predicted B, paired control C)."""
    n = len(true)
    j = np.full(n, np.nan)
    d = np.full(n, np.nan)
    for i in range(n):
        td = true[i] - C[i]
        p = B[i] - C[i]
        g = np.argsort(np.abs(td))[-K:]
        if np.abs(p).std() > 1e-12:
            j[i] = jac(g, np.argsort(np.abs(p))[-K:])
        if td[g].std() > 1e-12 and p[g].std() > 1e-12:
            d[i] = pearsonr(p[g], td[g])[0]
    return j, d


def features(fp, lines, lv):                                          # score_step15_tahoe.py, verbatim
    L = np.array([[float(l == u) for u in lv] for l in lines])
    return np.hstack([fp, L, np.concatenate([fp * L[:, [k]] for k in range(len(lv))], axis=1)])


def dense(X):                                                         # score_step15_tahoe.py
    return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)


def condition_index(groups):                                          # score_step15_tahoe.py, verbatim
    """Row indices per condition label, in file order, without a per-label scan of the whole file."""
    codes, uniques = pd.factorize(pd.Series(groups), sort=False)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    return {str(uniques[k]): order[bounds[k]:bounds[k + 1]] for k in range(len(uniques))}


def energy_to_real(cells, real, real_within):
    """score_step15_tahoe.energy(cells, real) = 2 E|X - Y| - E|X - X'| - E|Y - Y'|, with the last term (the measured
    cells among themselves, the same for every assignment) computed once per condition and passed in."""
    X = np.asarray(cells, dtype=np.float64)
    Y = np.asarray(real, dtype=np.float64)
    return 2 * cdist(X, Y).mean() - pdist(X).mean() - real_within


def unit_rows(M):                                                     # score_step15c_tahoe.py, verbatim
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    return np.where(nrm > 0, M / np.where(nrm > 0, nrm, 1.0), 0.0), (nrm[:, 0] > 0)


def pds_sets(pred_eff, true_eff, contexts, cand):                     # score_step15c_tahoe.py, verbatim
    """Step 14's pds_cosine with per-condition candidate sets. contexts: list of index arrays (one per (line, dose));
    cand[p]: the candidate indices of condition p (a subset of p's context containing p). Distances are computed per
    context as 1 - P @ T.T exactly as pds_cosine does; a zero-norm vector has distance 1 to everything; ties averaged;
    PDS_p = 1 - k_p / (n_p - 1); NaN where n_p < MIN_CAND."""
    P, okp = unit_rows(np.asarray(pred_eff, np.float64))
    T, okt = unit_rows(np.asarray(true_eff, np.float64))
    out = np.full(len(P), np.nan)
    for members in contexts:
        Dm = 1.0 - P[members] @ T[members].T
        Dm[~okp[members], :] = 1.0
        Dm[:, ~okt[members]] = 1.0
        pos = {q: j for j, q in enumerate(members)}
        for i, p in enumerate(members):
            if p not in cand:
                continue
            cols = np.array([pos[q] for q in cand[p]])
            if len(cols) < MIN_CAND:
                continue
            d = Dm[i, cols]
            own = Dm[i, pos[p]]
            k = np.sum(d < own) + 0.5 * (np.sum(d == own) - 1)
            out[p] = 1.0 - k / (len(cols) - 1)
    return out


def fast_rank(x):                                                     # score_step16_tahoe.py, verbatim
    """Average ranks (scipy.stats.rankdata, ties averaged), vectorised."""
    n = x.size
    o = np.argsort(x, kind="stable")
    xs = x[o]
    neq = np.empty(n, dtype=bool)
    neq[0] = True
    neq[1:] = xs[1:] != xs[:-1]
    st = np.flatnonzero(neq)
    en = np.append(st[1:], n)
    r = np.empty(n)
    r[o] = np.repeat((st + en - 1) / 2.0 + 1.0, en - st)
    return r


def prep_ctx(true_norm, ctx_sel):                                     # score_step16_tahoe.py, verbatim
    """Per context: the centred ranks of the measured norms and their length, or None where the context is too small or
    the measured norms are constant (no ranking: the context contributes nothing)."""
    out = []
    for m in ctx_sel:
        if len(m) < MIN_CTX or np.ptp(true_norm[m]) == 0:
            out.append(None)
            continue
        ry = fast_rank(true_norm[m])
        ry = ry - ry.mean()
        out.append((m, ry, float(np.sqrt((ry * ry).sum()))))
    return out


def mra_pre(pred_norm, pre):                                          # score_step16_tahoe.py, verbatim
    """MRA given the prepared contexts: mean over contexts of the Spearman between predicted and measured norms; a
    submission constant within a context scores 0 there."""
    rhos = []
    for item in pre:
        if item is None:
            continue
        m, ry, dy = item
        x = pred_norm[m]
        if np.ptp(x) == 0:
            rhos.append(0.0)
            continue
        rx = fast_rank(x)
        rx = rx - rx.mean()
        dx = np.sqrt((rx * rx).sum())
        rhos.append(float((rx * ry).sum() / (dx * dy)) if dx > 0 and dy > 0 else 0.0)
    return float(np.mean(rhos)) if rhos else np.nan


def ctx_split(ctx_id, n_ctx, sel):                                    # score_step16_tahoe.py, verbatim
    """The selected conditions of each context, in one pass over sel."""
    c_of = ctx_id[sel]
    return [sel[c_of == c] for c in range(n_ctx)]


class Boot:                                                           # score_step23_screen.py (as score_step29_sciplex.py)
    """Structure-clustered percentile bootstrap over one set of conditions. Clusters are the structures in the set, in
    sorted order; the draws are those of score_step15_tahoe.boot (rng.randint(0, G, G) per replicate) from a fresh
    RandomState(0), so every interval of one set uses the same 5,000 resamples."""

    def __init__(self, idx, clusters):
        self.idx = np.asarray(idx)
        lab = np.asarray(clusters)[self.idx]
        self.labels = np.unique(lab)
        self.G = len(self.labels)
        self.member = np.searchsorted(self.labels, lab)              # cluster number of each condition in the set
        rng = np.random.RandomState(0)
        draws = np.array([rng.randint(0, self.G, self.G) for _ in range(N_BOOT)])
        self.cnt = np.zeros((N_BOOT, self.G))                        # how often each cluster is drawn, per replicate
        np.add.at(self.cnt, (np.repeat(np.arange(N_BOOT), self.G), draws.ravel()), 1.0)
        self.N = np.bincount(self.member, minlength=self.G).astype(np.float64)

    def interval(self, x):
        """Mean of the per-condition values x over the set, and its 97.5 % percentile interval."""
        v = np.asarray(x, np.float64)[self.idx]
        if not np.isfinite(v).all():
            die("non-finite per-condition value inside a scored set")
        S = np.bincount(self.member, weights=v, minlength=self.G)
        reps = (self.cnt @ S) / (self.cnt @ self.N)
        lo, hi = float(np.percentile(reps, LO)), float(np.percentile(reps, HI))
        return dict(diff=float(v.mean()), lo=lo, hi=hi, ci=verdict(lo, hi))


# ----------------------------------------------------------------------------- folds
def deal(counts, order):
    """GroupKFold's own rule for a given order of the groups: each group goes to the fold that is lightest so far (the
    first one if several are equally light). Returns the fold of each group."""
    weight = np.zeros(NFOLD)
    fold_of_group = np.zeros(len(counts), dtype=int)
    for k in order:
        f = int(np.argmin(weight))
        weight[f] += counts[k]
        fold_of_group[k] = f
    return fold_of_group


def canonical(fold_of_group):
    """Digit string with the folds renumbered by first appearance, so that one partition has one spelling."""
    remap, out = {}, []
    for f in fold_of_group:
        if int(f) not in remap:
            remap[int(f)] = len(remap)
        out.append(str(remap[int(f)]))
    return "".join(out)


def random_fold_of_group(counts, seed):
    """One random assignment (pre-registration, section 2): the structures in random order, then largest first (stable,
    so that ties keep the random order), then GroupKFold's rule."""
    perm = np.random.RandomState(seed).permutation(len(counts))
    order = perm[np.argsort(-counts[perm], kind="stable")]
    return deal(counts, order)


# ----------------------------------------------------------------------------- one assignment on one split
def score_assignment(sp, fold_of_group):
    """Everything of one fold assignment on one split. Returns scalars and, under 'pc', the per-condition arrays."""
    d = D[sp]
    fold = np.asarray(fold_of_group)[d["inv"]]                        # fold of every condition
    X, Y = d["X"], d["Y"]
    P = np.zeros_like(Y)                                              # score_step15_tahoe.ridge_s7, the folds held out in turn
    for f in range(NFOLD):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        if len(te) == 0:
            die("an empty fold")
        P[te] = Ridge(alpha=ALPHA).fit(X[tr], Y[tr]).predict(X[te])
    j, dp = thesis_per_cond(d["true"], d["C"] + P, d["C"])
    en = np.array([energy_to_real(np.clip(d["paired"][i] + P[i][None, :], 0, None), d["real"][i], d["real_within"][i])
                   for i in range(d["n"])])
    pds = pds_sets(P, Y, d["contexts"], d["cand"])
    norm = np.linalg.norm(P, axis=1)
    m = d["models"]
    r = dict(gi=float(np.nanmean(j)), dpc_mean=float(np.nanmean(dp)), dpc_median=float(np.nanmedian(dp)),
             energy=float(en.mean()), pds_mean=float(np.nanmean(pds)))
    r["B1"] = dict(diff=float(np.nanmean(m["jac"][BLIND])) - r["gi"])
    r["B2"] = dict(diff=float(np.nanmean(m["jac"][ORIGINAL])) - r["gi"])
    r["B3"] = d["boot_pds"].interval(pds - 0.5)
    reps = np.array([mra_pre(norm, pre) for pre in d["pres"]])
    lo, hi = float(np.nanpercentile(reps, LO)), float(np.nanpercentile(reps, HI))       # score_step16_tahoe.verdict_of
    r["B4"] = dict(diff=mra_pre(norm, d["pre_full"]), lo=lo, hi=hi, ci=verdict(lo, hi))
    r["C1"] = d["boot_all"].interval(m["jac"][REWIRED] - j)
    r["C2"] = d["boot_all"].interval(m["en"][REWIRED] - en)
    r["C3"] = d["boot_pds"].interval(m["pds"][REWIRED] - pds)
    r["same"] = {it: same_verdict(it, sp, r) for it in ITEMS}
    r["pc"] = dict(jac=j, dpc=dp, en=en, pds=pds)
    return r


def same_verdict(item, sp, r):
    """Does one assignment give, on one split, the verdict the pre-registration states for the item?"""
    st, x = STATED[item][sp], r[item]
    ok = True
    if "sign" in st:
        ok = ok and ((x["diff"] > 0) if st["sign"] == "positive" else (x["diff"] < 0))
    if "ci" in st:
        ok = ok and (x["ci"] == st["ci"])
    return bool(ok)


def worker(task):
    """One random assignment on one split, in a worker process (one thread; the data are inherited)."""
    sp, k = task
    g = random_fold_of_group(D[sp]["counts"], SEED0 + k)
    r = score_assignment(sp, g)
    r.pop("pc")
    r["partition"] = canonical(g)
    return sp, k, r


def worker_init():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)                                   # only ever lowered
    except Exception:
        pass


# ----------------------------------------------------------------------------- loading
def max_diff(mine, record):
    mine, record = np.asarray(mine, float), np.asarray(record, float)
    if not np.array_equal(np.isnan(mine), np.isnan(record)):
        return float("inf")
    ok = ~np.isnan(mine)
    return float(np.abs(mine[ok] - record[ok]).max()) if ok.any() else 0.0


def load_split(sp, data_dir, pc15, pc15c, folds_rec):
    import anndata as ad
    from Squidiff.scrna_datasets import Drug_dose_encoder             # the harness's fingerprint + dose encoder
    t0 = time.time()
    rec = pc15[pc15["split"] == sp].reset_index(drop=True)
    rec_c = pc15c[pc15c["split"] == sp].reset_index(drop=True)
    n = len(rec)
    if n == 0 or len(rec_c) != n or not np.array_equal(rec["condition"].values, rec_c["condition"].values):
        die("%s: the two per-condition records do not hold the same conditions in the same order" % sp)
    cond = rec["condition"].values.astype(str)
    drugs, lines = rec["structure"].values.astype(str), rec["line"].values.astype(str)
    doses = rec["dose_nM"].values.astype(float)
    lv = sorted(set(str(x) for x in lines))

    # measured change, from the stored vectors of the rewired model, seed 0 (hash-checked against provenance)
    tag = "s15_%s_recipe_seed_0" % sp
    pj = os.path.join(REPO, "provenance", "s15ev_%s.json" % tag)
    places = [os.path.join(REPO, RES), os.environ.get("S29_VECTORS_DIR", ""), os.path.join(REPO, "..", "Squidiff", RES)]
    f = next((os.path.join(p, "vectors_%s.npz" % tag) for p in places
              if p and os.path.isfile(os.path.join(p, "vectors_%s.npz" % tag))), None)
    if f is None or not os.path.isfile(pj):
        die("%s: the stored vectors or their provenance record are missing" % tag)
    if json.load(open(pj)).get("vectors_sha256") != sha256(f):
        die("%s vectors hash differs from provenance" % tag)
    v = np.load(f, allow_pickle=True)
    if not np.array_equal(np.array([str(x) for x in v["cond"]]), cond):
        die("%s: the conditions of the stored vectors differ from the per-condition record" % sp)
    true, C = np.asarray(v["true"], np.float64), np.asarray(v["C"], np.float64)
    Y = true - C

    # measured cells and paired vehicle cells, drawn as the evaluation drew them (score_step15_tahoe.py)
    te = ad.read_h5ad("%s/tahoe_test_%s.h5ad" % (data_dir, sp))
    tc = ad.read_h5ad("%s/tahoe_test_%s_control.h5ad" % (data_dir, sp))
    if te.n_obs != tc.n_obs or not np.array_equal(te.obs["Group"].astype(str).values, tc.obs["Group"].astype(str).values):
        die("%s: test and control files are not row-aligned" % sp)
    Xte, Xtc = dense(te.X), dense(tc.X)
    idx = condition_index(te.obs["Group"].astype(str).values)
    if set(idx) != set(cond):
        die("%s: the conditions of the test file differ from the record" % sp)
    smi_all = te.obs["SMILES"].astype(str).values
    real, paired, real_within, mean_gap = [], [], np.zeros(n), 0.0
    for i, c in enumerate(cond):
        ix = idx[c]
        sel = np.arange(len(ix))
        if len(sel) > CAP:
            sel = np.random.RandomState(0).choice(sel, CAP, replace=False)      # the harness's draw, verbatim
        if smi_all[ix[0]] != drugs[i]:
            die("%s: structure of %s differs between the test file and the record" % (sp, c))
        real.append(Xte[ix][sel])
        paired.append(Xtc[ix][sel])
        real_within[i] = pdist(np.asarray(real[i], dtype=np.float64)).mean()
        mean_gap = max(mean_gap, float(np.abs(real[i].mean(axis=0) - true[i]).max()))   # reported only; guard 5 decides
    del te, tc, Xte, Xtc

    # features and groups, as score_step15_tahoe.py
    FP = np.array([Drug_dose_encoder([drugs[i]], [doses[i]], comb_num=1)[0] for i in range(n)], dtype=np.float64)
    X = features(FP, lines, lv)
    uniq, inv = np.unique(drugs, return_inverse=True)
    counts = np.bincount(inv)
    fr = folds_rec["per_split"][sp]
    if [str(u) for u in uniq] != fr["structures"] or [int(c) for c in counts] != fr["conditions_per_structure"]:
        die("%s: the structures of the folds record are not those of the data" % sp)
    paper = np.array(fr["folds_of_the_paper"]["fold_of_structure"], dtype=int)

    # values of the models, read from the records (they do not depend on the folds)
    models = dict(jac={m: rec["%s_de_overlap" % m].values.astype(float) for m in (REWIRED, ORIGINAL, BLIND)},
                  en={REWIRED: rec["%s_energy" % REWIRED].values.astype(float)},
                  pds={REWIRED: rec_c["%s_pds" % REWIRED].values.astype(float)})
    committed = dict(jac=rec["%s_de_overlap" % RIDGE].values.astype(float), dpc=rec["%s_dpc50" % RIDGE].values.astype(float),
                     en=rec["%s_energy" % RIDGE].values.astype(float), pds=rec_c["%s_pds" % RIDGE].values.astype(float))

    # discrimination score: contexts and candidate sets (score_step15c_tahoe.py, verbatim)
    ctx_key = np.array(["%s|%.6f" % (lines[i], doses[i]) for i in range(n)])
    ctx_u = np.unique(ctx_key)
    contexts = [np.where(ctx_key == u)[0] for u in ctx_u]
    cand = {}
    for members in contexts:
        for p in members:
            cand[p] = np.array([q for q in members if q == p or drugs[q] != drugs[p]])
    scored = np.where(np.isfinite(committed["pds"]))[0]              # conditions with at least MIN_CAND candidates

    # response-magnitude agreement: contexts and the resamples of score_step16_tahoe.py
    ctx_id = np.array([int(np.where(ctx_u == k)[0][0]) for k in ctx_key])
    n_ctx = len(contexts)
    true_norm = np.linalg.norm(Y, axis=1)
    pre_full = prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, np.arange(n)))
    rng = np.random.RandomState(0)
    gs = [np.where(drugs == u)[0] for u in uniq]
    pres = [prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, np.concatenate([gs[g] for g in rng.randint(0, len(gs), len(gs))])))
            for _ in range(N_BOOT)]

    print("[data] %s: %d conditions, %d structures, %d lines, %d features, %d contexts; %d conditions scored for the "
          "discrimination score; cells per condition up to %d; mean of the drawn measured cells against the stored "
          "vectors: largest difference %.1e   [%.0f s]"
          % (sp, n, len(uniq), len(lv), X.shape[1], n_ctx, len(scored), max(len(a) for a in real), mean_gap,
             time.time() - t0))
    return dict(n=n, cond=cond, drugs=drugs, lines=lines, X=X, Y=Y, true=true, C=C, real=real, paired=paired,
                real_within=real_within, inv=inv, counts=counts, uniq=uniq, paper=paper, models=models, committed=committed,
                contexts=contexts, cand=cand, pre_full=pre_full, pres=pres,
                boot_all=Boot(np.arange(n), drugs), boot_pds=Boot(scored, drugs), n_ctx=n_ctx, n_scored=int(len(scored)))


def summary(values, paper):
    v = np.asarray(values, dtype=np.float64)
    return {"paper": float(paper), "median": float(np.median(v)), "p2.5": float(np.percentile(v, 2.5)),
            "p97.5": float(np.percentile(v, 97.5)), "min": float(v.min()), "max": float(v.max()),
            "n_random_not_above_paper": int((v <= paper).sum()), "n": int(len(v))}


QUANT = [("ridge_gene_identity", lambda r: r["gi"]), ("ridge_response_correlation_mean", lambda r: r["dpc_mean"]),
         ("ridge_response_correlation_median", lambda r: r["dpc_median"]), ("ridge_population_distance", lambda r: r["energy"]),
         ("ridge_discrimination_score", lambda r: r["pds_mean"]), ("ridge_response_magnitude", lambda r: r["B4"]["diff"]),
         ("B1_blind_minus_ridge_gene_identity", lambda r: r["B1"]["diff"]),
         ("B2_original_minus_ridge_gene_identity", lambda r: r["B2"]["diff"]),
         ("B3_ridge_discrimination_minus_half", lambda r: r["B3"]["diff"]),
         ("C1_rewired_minus_ridge_gene_identity", lambda r: r["C1"]["diff"]),
         ("C2_rewired_minus_ridge_population_distance", lambda r: r["C2"]["diff"]),
         ("C3_rewired_minus_ridge_discrimination", lambda r: r["C3"]["diff"])]


def flat(sp, name, partition, r):
    row = dict(split=sp, assignment=name, partition=partition)
    for key, fn in QUANT[:6]:
        row[key] = fn(r)
    for it in ITEMS:
        for q in ("diff", "lo", "hi", "ci"):
            if q in r[it]:
                row["%s_%s" % (it, q)] = r[it][q]
        row["%s_same_verdict" % it] = int(r["same"][it])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=RES)
    ap.add_argument("--n-random", type=int, default=N_RANDOM, help="100 only by a dated paragraph of the pre-registration")
    ap.add_argument("--folds-sha256", default=FOLDS_SHA256, help="test only")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    t0 = time.time()
    os.chdir(REPO)
    registered = (args.n_random in (200, 100) and args.folds_sha256 == FOLDS_SHA256 and args.data_dir == DATA_DIR)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    if not registered and os.path.abspath(out_dir) == os.path.abspath(os.path.join(REPO, RES)):
        die("an option differs from the registered settings; such a run must write elsewhere (--out-dir)")
    out_json, out_csv = os.path.join(out_dir, OUT_JSON), os.path.join(out_dir, OUT_CSV)

    # ---- guard 1: no output, nothing modified, everything committed ------------------------------------------
    for q in (out_json, out_csv):
        if os.path.exists(q):
            die("%s exists; refusing to overwrite a record" % q)
    dirty = sh("git status --porcelain --untracked-files=no")
    if dirty:
        die("tracked files are modified:\n" + dirty)
    for q in TRACKED + [os.path.basename(__file__)]:
        if subprocess.run("git ls-files --error-unmatch '%s'" % q, shell=True, capture_output=True).returncode != 0:
            die("%s is not committed" % q)
    if sha256(REC_FOLDS) != args.folds_sha256:
        die("%s does not have the expected sha256" % REC_FOLDS)
    commit = sh("git rev-parse --short HEAD")
    print("[guard 1] no output exists; no tracked file is modified; the pre-registration, this script and the %d records "
          "it reads are committed (commit %s); the folds record has the expected sha256" % (len(TRACKED) - 1, commit))
    cpu = sh("grep -m1 'model name' /proc/cpuinfo").split(":", 1)[-1].strip()
    env = dict(node=platform.node(), cpu=cpu, python=platform.python_version(), numpy=np.__version__,
               scipy=version_of("scipy"), sklearn=version_of("scikit-learn"), pandas=pd.__version__,
               anndata=version_of("anndata"), threads=os.environ.get("OMP_NUM_THREADS", ""), workers=args.workers,
               slurm_job=os.environ.get("SLURM_JOB_ID", ""))
    print("[env] %s" % env)

    str_cols = {"split": str, "condition": str, "line": str, "plate": str, "structure": str}
    pc15 = pd.read_csv(PC15, dtype=str_cols, keep_default_na=False, na_values=[""])
    pc15c = pd.read_csv(PC15C, dtype=str_cols, keep_default_na=False, na_values=[""])
    for col in ["%s_de_overlap" % m for m in (RIDGE, REWIRED, ORIGINAL, BLIND)] + ["%s_dpc50" % RIDGE, "%s_energy" % RIDGE,
                                                                                  "%s_energy" % REWIRED]:
        if col not in pc15.columns:
            die("%s has no column %s" % (PC15, col))
    for col in ("%s_pds" % RIDGE, "%s_pds" % REWIRED):
        if col not in pc15c.columns:
            die("%s has no column %s" % (PC15C, col))
    folds_rec = json.load(open(REC_FOLDS))
    rec15, rec15c, rec16 = json.load(open(REC15)), json.load(open(REC15C)), json.load(open(REC16))

    for sp in SPLITS:
        D[sp] = load_split(sp, args.data_dir, pc15, pc15c, folds_rec)

    # ---- guard 5: with the folds of the paper the ridge reproduces the committed records ---------------------
    paper = {}
    for sp in SPLITS:
        d = D[sp]
        r = score_assignment(sp, d["paper"])
        c, pc = d["committed"], r["pc"]
        diffs = {k: max_diff(pc[k], c[k]) for k in ("jac", "dpc", "en", "pds")}
        m16 = rec16["per_split"][sp]
        diffs["mra"] = abs(r["B4"]["diff"] - m16["metrics"][RIDGE]["mra"])
        ci16 = m16["contrasts"]["secondary_mra_%s_minus_0" % RIDGE]["ci"]
        diffs["mra_interval"] = max(abs(r["B4"]["lo"] - ci16[0]), abs(r["B4"]["hi"] - ci16[1]))
        print("[guard 5] %s: largest differences from the committed records: gene identity %.1e, response correlation "
              "%.1e, population distance %.1e, discrimination score %.1e, response-magnitude agreement %.1e, its "
              "interval %.1e   [%.0f s]" % (sp, diffs["jac"], diffs["dpc"], diffs["en"], diffs["pds"], diffs["mra"],
                                           diffs["mra_interval"], time.time() - t0))
        bad = [k for k, x in diffs.items() if not x <= TOL]
        if bad:
            die("guard 5, %s: not reproduced to %.0e: %s. Nothing is written." % (sp, TOL, ", ".join(bad)))
        r["guard_differences"] = diffs
        paper[sp] = r

    print("\n" + "=" * 100 + "\nGUARDS PASSED. Values with the folds of the paper (intervals from a fresh generator; the "
          "committed interval in brackets after 'paper:'):\n" + "=" * 100)
    com = {}
    for sp in SPLITS:
        r = paper[sp]
        c15, c15c = rec15["per_split"][sp]["contrasts"], rec15c.get("per_split", rec15c.get("results", {})).get(sp, {}).get("contrasts", {})
        com[sp] = dict(C1=c15.get("Q2_coprimary_de_recipe_minus_ridgeS7", {}).get("ci"),
                       C2=c15.get("secondary_en_recipe_minus_ridge_S7", {}).get("ci"),
                       B3=c15c.get("secondary_pds_%s_minus_0.5" % RIDGE, {}).get("ci"),
                       C3=c15c.get("secondary_pds_recipe_minus_%s" % RIDGE, {}).get("ci"),
                       B4=rec16["per_split"][sp]["contrasts"]["secondary_mra_%s_minus_0" % RIDGE]["ci"])
        print("  %s  ridge: gene identity %.4f, response correlation mean %.4f median %.4f, population distance %.4f, "
              "discrimination score %.4f, response magnitude %+.4f"
              % (sp, r["gi"], r["dpc_mean"], r["dpc_median"], r["energy"], r["pds_mean"], r["B4"]["diff"]))
        print("      B1 %+.4f   B2 %+.4f" % (r["B1"]["diff"], r["B2"]["diff"]))
        for it in ("B1", "B2"):
            if not r["same"][it]:
                print("      %s: NOT the stated verdict (%s)" % (it, STATED[it][sp]))
        for it in ("B3", "B4", "C1", "C2", "C3"):
            cc = com[sp].get(it)
            print("      %s %+.4f [%+.4f, %+.4f] %-13s paper: %s; stated: %s -> %s"
                  % (it, r[it]["diff"], r[it]["lo"], r[it]["hi"], r[it]["ci"],
                     ("[%+.4f, %+.4f]" % tuple(cc)) if cc else "n/a", STATED[it][sp],
                     "same verdict" if r["same"][it] else "NOT the stated verdict"))

    # ---- the random assignments, in worker processes ---------------------------------------------------------
    tasks = [(sp, k) for sp in SPLITS for k in range(args.n_random)]
    rnd = {sp: {} for sp in SPLITS}
    print("\n[random] %d assignments per split on %d process(es)" % (args.n_random, max(1, args.workers)))
    if args.workers > 1:
        ctx = multiprocessing.get_context("fork")                     # the workers inherit the data; nothing is copied
        with ctx.Pool(args.workers, initializer=worker_init) as pool:
            for done, (sp, k, r) in enumerate(pool.imap_unordered(worker, tasks, chunksize=1), 1):
                rnd[sp][k] = r
                if done % 50 == 0 or done == len(tasks):
                    print("[random] %d of %d   [%.0f s]" % (done, len(tasks), time.time() - t0))
    else:
        for done, t in enumerate(tasks, 1):
            sp, k, r = worker(t)
            rnd[sp][k] = r
            if done % 10 == 0 or done == len(tasks):
                print("[random] %d of %d   [%.0f s]" % (done, len(tasks), time.time() - t0))

    # ---- counts, ranges, records -----------------------------------------------------------------------------
    need = int(math.ceil(THRESHOLD * args.n_random - 1e-9))
    items, rows, ranges = {}, [], {}
    print("\n" + "=" * 100 + "\nRESULT: assignments that give the verdict of the paper (needed: %d of %d)\n" % (need, args.n_random)
          + "=" * 100)
    for it in ITEMS:
        per = {}
        for sp in SPLITS:
            k_ok = int(sum(rnd[sp][k]["same"][it] for k in range(args.n_random)))
            per[sp] = dict(n_same=k_ok, n=args.n_random, needed=need, robust=bool(k_ok >= need), stated=STATED[it][sp],
                           folds_of_the_paper_same_verdict=bool(paper[sp]["same"][it]))
        items[it] = dict(robust=bool(all(v["robust"] for v in per.values())), per_split=per)
        print("  %s  %s" % (it, "ROBUST" if items[it]["robust"] else "NOT ROBUST"))
        for sp in SPLITS:
            print("      %s  %3d of %3d  %-10s stated: %s%s" % (sp, per[sp]["n_same"], args.n_random,
                  "robust" if per[sp]["robust"] else "NOT robust", per[sp]["stated"],
                  "" if per[sp]["folds_of_the_paper_same_verdict"] else "   (the folds of the paper do not give the stated verdict here)"))
    print("\nRanges over the random assignments (value of the paper | median, 2.5th to 97.5th percentile, range | at or "
          "below the paper)")
    for sp in SPLITS:
        print("  " + sp)
        ranges[sp] = {}
        for key, fn in QUANT:
            s = summary([fn(rnd[sp][k]) for k in range(args.n_random)], fn(paper[sp]))
            ranges[sp][key] = s
            print("      %-46s %+.4f | %+.4f, %+.4f to %+.4f, %+.4f to %+.4f | %d of %d"
                  % (key, s["paper"], s["median"], s["p2.5"], s["p97.5"], s["min"], s["max"], s["n_random_not_above_paper"], s["n"]))
        rows.append(flat(sp, "paper", canonical(D[sp]["paper"]), paper[sp]))
        rows += [flat(sp, str(SEED0 + k), rnd[sp][k]["partition"], rnd[sp][k]) for k in range(args.n_random)]

    def clean(r):
        return {k: v for k, v in r.items() if k not in ("pc",)}
    record = dict(step="29, Tahoe-100M part", prereg=PRE, prereg_sha256=sha256(PRE), script=os.path.basename(__file__),
                  script_sha256=sha256(__file__), commit=commit, env=env,
                  settings=dict(n_random=args.n_random, first_seed=SEED0, threshold=THRESHOLD, needed=need, n_boot=N_BOOT,
                                interval="97.5 % percentile, structures resampled, fresh RandomState(0) per interval",
                                tolerance=TOL, registered=bool(registered)),
                  inputs={q: sha256(q) for q in TRACKED}, items=items, stated=STATED,
                  per_split={sp: dict(n_conditions=D[sp]["n"], n_structures=int(len(D[sp]["uniq"])), n_contexts=D[sp]["n_ctx"],
                                      n_scored_discrimination=D[sp]["n_scored"], partition_of_the_paper=canonical(D[sp]["paper"]),
                                      folds_of_the_paper=clean(paper[sp]), committed_intervals=com[sp], ranges=ranges[sp])
                             for sp in SPLITS},
                  seconds=round(time.time() - t0, 1))
    for q in (out_json, out_csv):
        if os.path.exists(q):
            die("%s appeared during the run; not overwritten" % q)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=1)
    cols = []
    for row in rows:
        cols += [c for c in row if c not in cols]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("\n[written] %s  sha256 %s" % (out_json, sha256(out_json)))
    print("[written] %s  sha256 %s  (%d rows)   [%.0f s]" % (out_csv, sha256(out_csv), len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
