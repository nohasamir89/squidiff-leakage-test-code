#!/usr/bin/env python3
"""
score_step29_sciplex.py - Step 29 on sci-Plex: do the comparisons with the cross-validated ridge
depend on the fold assignment?  (PREREG_step29_fold_sensitivity.md, items A1 to A7 and C4)

Purpose
-------
The cross-validated ridge of the paper is fitted within the held-out compounds with five folds
grouped by compound. The folds came from scikit-learn's GroupKFold, whose order of ties depends on
the processor, so the assignment of the paper is one of several that the same code can give
(leakage_results/diag_note1_ridge_19363649.json). This script refits that ridge with the fold
assignment of the paper and with 200 random assignments, and counts for every statement of the
paper that compares something with the ridge on sci-Plex in how many assignments the statement
keeps its verdict. Nothing else is refitted: the values of the generative models are read from the
committed per-condition record. The one exception is the small feedforward network of Results 3,
which is itself fitted with the folds and is retrained together with the ridge.

Items (section 4 of the pre-registration; verdict = what the paper states)
---------------------------------------------------------------------------
A1  gene identity, rewired model minus ridge: positive, interval above zero (Splits 1, 0, 2)
A2  the same without the compound whose salt twin is in training (Splits 0 and 2)
A3  gene identity, original model minus ridge: negative
A4  response correlation, mean over conditions: the highest generative model is below the ridge
A5  response correlation, median over conditions: Splits 1 and 0 as A4; Split 2: the model with
    cross-attention and its control are both above the ridge
A6  response-magnitude agreement of the ridge: positive; interval above zero on Split 1 and
    including zero on Splits 0 and 2
A7  feedforward network minus ridge (median response correlation): negative on Split 0; the
    range on Splits 1 and 2 is reported without a verdict (first 50 assignments)
C4  response-magnitude agreement, rewired model minus ridge: interval includes zero
Rule: an item is robust on a split if at least 95 % of the random assignments give the verdict of
the paper (190 of 200; 48 of 50 for A7), and robust if it is robust on every split it names.

Logic
-----
0. One numpy on every node. numpy sorts equal values in a different order when it runs its AVX-512
   code; that is why the folds of the paper depend on the node, and it also decides which genes
   enter a top-50 set when the 50th place is tied (one condition of Split 1). The committed
   sci-Plex records were written with numpy's plain code, because each of their scripts had to
   reproduce the Step 7 record. On a node whose processor has AVX-512 this script therefore starts
   itself again, once, with that code of numpy switched off (NPY_DISABLE_CPU_FEATURES), and
   records that it did.
1. Guards before anything is computed: no output exists; no tracked file is modified; the
   pre-registration, this script and every record it reads are committed.
2. Data per split, read as score_step14.py reads them: the measured change Y7 = true - C of the
   Step 7 target file (single precision), the mean vectors of the rewired model (five seeds, each
   file checked against its provenance record), and SMILES, dose and cell line per condition from
   the test file. Features: dose-scaled fingerprint, cell-line indicator and their products
   (4,099 columns), built as score_step14.py builds them.
3. Fold assignments. Of the paper: read from the committed Note 1 record
   (leakage_results/pathway_vs_fingerprint_v2.json, sha256 pinned below). Random: the rule and the
   seeds of that record (compounds in random order, largest first with a stable sort, each to the
   fold that is lightest so far). GUARD: the 200 assignments built here equal the 200 stored there.
4. The ridge under one assignment, operation for operation as score_step14.py: sklearn Ridge,
   penalty 1, features in double precision, target and stored predictions in single precision.
   From its predicted change, with the functions of the paper copied verbatim:
     gene identity and response correlation per condition   (thesis_per_cond, score_step14.py)
     response-magnitude agreement                           (fast_rank, prep_ctx, mra_pre, ctx_split, score_step16_tahoe.py)
   Intervals: the bootstrap class of score_step23_screen.py copied verbatim (clusters = compounds,
   5,000 replicates, rng.randint(0, G, G) per replicate from a fresh RandomState(0); 1.25th and
   98.75th percentile). It is the construction of Steps 14, 17 and 28, and every assignment meets
   the same resamples. Step 14 drew its intervals from one generator shared between contrasts, so
   the committed interval of Table S5c can differ from the recomputed one in the third decimal;
   for the folds of the paper both are printed and stored, and only the recomputed one is judged.
5. GUARDS on the folds of the paper, all checked before any value of a random assignment is
   computed (pre-registration, section 7, guard 3; printed as guards 3a to 3g):
     a. the ridge gives back the Step 7 record (median response correlation, to 1e-9);
     b. its per-condition gene identity and response correlation equal the rows of the
        cross-validated ridge in leakage_results/step14_per_condition.csv (1e-9); a difference is
        listed for all three splits (how many conditions, the largest ones) before the run stops;
     c. the mean (and median) of every model in that record equal step14_analysis.json (1e-9), and so
        does the point value of the contrast rewired model minus ridge;
     d. its response-magnitude agreement and interval, and the contrast with the rewired model,
        equal step17_mra.json (1e-9);
     e. the contrasts of Supplementary Note 7 (all conditions, and without the twin compound) equal
        the Step 28 record with their intervals (1e-9);
     f. the feedforward network and its ridge give back ridge_vs_mlp_v2.json (0.005);
     g. every item has, with the folds of the paper, the verdict the pre-registration states.
   If a guard fails the script stops and writes nothing.
6. The 200 random assignments: the same quantities, one row each. The network is retrained for the
   first 50 (three seeds, width 64, weight decay 10, 400 epochs, as ridge_vs_mlp_v2.py; the fold
   numbers that seed each network follow GroupKFold's own dealing, and for the folds of the paper
   they are those of the node the record was written on, read from the diagnostic record).
   These 150 trainings (50 assignments x 3 splits, three seeds each) are independent of each other
   and are spread over the CPUs of the job, one thread each; every training is seeded, so the
   result does not depend on the number of CPUs. The network of guard 3f runs alone, with the
   thread setting of the job. The ridge of this item is fitted in single precision, where the
   number of threads can move the median in the fifth decimal (diagnostic of 13 Nov 2026). The
   values of item A7 with the folds of the paper (threads of the job) and with the random
   assignments (one thread) are therefore comparable to four decimals.
   Threads. The predictions of the paper's ridge are computed in double precision and stored in
   single precision. A last digit of the double-precision value depends on the linear-algebra
   code of the processor and on its number of threads, and now and then it decides how a stored
   value is rounded; that can move a per-condition response correlation by about 1e-9. Guard 3b
   compares with the per-condition record to 1e-9, so run_step29_sciplex.sh gives this script the
   thread setting of the job that wrote that record (run_step14_rescore.sh: 8 threads) and asks
   for the standard compute nodes. The network record was written with 16 threads, and a training
   can end elsewhere with another number (run of 15 Nov 2026 with 8: Split 0 identical, Split 2
   off by 0.007). The training library is therefore set to 16 threads for the network of guard 3f
   (NET_THREADS), whatever the job exports; its guard allows 0.005.
7. Counts, the 95 % rule, and for every quantity the value of the paper, the median, the 2.5th to
   97.5th percentile and the range over the random assignments.

Outputs (written once, at the end, never overwritten)
------------------------------------------------------
leakage_results/step29_fold_sensitivity_sciplex.json                 everything above
leakage_results/step29_fold_sensitivity_sciplex_per_assignment.csv   one row per split and assignment

Usage (repository root, squidiff environment, through run_step29_sciplex.sh):
    python -B score_step29_sciplex.py
The options exist for tests on an invented environment. A run with any option changed refuses to
write into leakage_results.
"""
import os
import sys
sys.dont_write_bytecode = True          # leave no __pycache__ behind in the worktree


def numpy_without_avx512():
    """Step 0 of the logic. Returns the numpy features that are switched off in this process (a string, empty on a
    node without AVX-512). If numpy would use AVX-512 code here, the script is started again with the same command
    line and those features switched off; the second start finds nothing left to switch off and carries on."""
    def avx512_in_use():
        for mod in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
            try:
                m = __import__(mod, fromlist=["__cpu_features__", "__cpu_dispatch__"])
                feats, disp = dict(m.__cpu_features__), list(m.__cpu_dispatch__)
                return [f for f in disp if feats.get(f, False) and ("AVX512" in f or f == "X86_V4")]
            except Exception:
                continue
        return None
    on = avx512_in_use()
    if on is None:
        sys.exit("FATAL: numpy does not say which processor-specific code it uses")
    if on and os.environ.get("STEP29_RESTARTED") != "1":
        prev = os.environ.get("NPY_DISABLE_CPU_FEATURES", "").split()
        os.environ["NPY_DISABLE_CPU_FEATURES"] = " ".join(prev + [f for f in on if f not in prev])
        os.environ["STEP29_RESTARTED"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        print("[numpy] this node has AVX-512; starting again with numpy's AVX-512 code switched off (%s)" % " ".join(on))
        sys.stdout.flush(); sys.stderr.flush()
        os.execv(sys.executable, [sys.executable] + list(getattr(sys, "orig_argv", [None] + sys.argv))[1:])
    if on:
        sys.exit("FATAL: numpy still uses AVX-512 code after the restart: %s" % " ".join(on))
    return os.environ.get("NPY_DISABLE_CPU_FEATURES", "")


# In the main process only (worker processes inherit the setting through the environment).
NUMPY_SWITCHED_OFF = numpy_without_avx512() if __name__ == "__main__" else os.environ.get("NPY_DISABLE_CPU_FEATURES", "")

import argparse      # noqa: E402
import csv           # noqa: E402
import hashlib       # noqa: E402
import json          # noqa: E402
import math          # noqa: E402
import platform      # noqa: E402
import subprocess    # noqa: E402
import time          # noqa: E402
import warnings      # noqa: E402
from datetime import datetime, timezone   # noqa: E402

import numpy as np   # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------------- settings
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)                                             # the Squidiff package of this worktree
RES = "leakage_results"
DATA_DIR = "/scratch/ni2071/Diffusion/sciplex/datasets_corrected"
PRE = "PREREG_step29_fold_sensitivity.md"
REC_NOTE1 = os.path.join(RES, "pathway_vs_fingerprint_v2.json")       # folds of the paper and the 200 random assignments
NOTE1_SHA256 = "aebc1c61610bd1c0cbf29269facfdb051474293c9ac13dfb2efbe45e5cb58066"
REC_DIAG = os.path.join(RES, "diag_note1_ridge_19363649.json")        # dealing order of the folds of the paper (for A7)
REC_S7 = os.path.join(RES, "step7_interaction_baseline.json")         # the Step 7 record (anchor)
PC14 = os.path.join(RES, "step14_per_condition.csv")                  # per-condition values of every model
REC14 = os.path.join(RES, "step14_analysis.json")
REC17 = os.path.join(RES, "step17_mra.json")
REC28 = os.path.join(RES, "step28_sciplex.json")
PQ28 = os.path.join(RES, "step28_sciplex_per_quantity.csv")
REC_MLP = os.path.join(RES, "ridge_vs_mlp_v2.json")
OUT_JSON = "step29_fold_sensitivity_sciplex.json"
OUT_CSV = "step29_fold_sensitivity_sciplex_per_assignment.csv"
TRACKED = [PRE, REC_NOTE1, REC_DIAG, REC_S7, PC14, REC14, REC17, REC28, PQ28, REC_MLP]

SPLITS = ["drug_split_1", "drug_split_0", "drug_split_2"]            # development split first, as everywhere in the paper
N_DRUGS = {"drug_split_1": 37, "drug_split_0": 37, "drug_split_2": 38}
N_SEEDS, K, ALPHA, NFOLD = 5, 50, 1.0, 5                             # seeds of the rewired model; top genes; penalty; folds
N_BOOT, LO, HI = 5000, 1.25, 98.75                                   # 97.5 % percentile interval per split
MIN_CTX = 10                                                         # smallest context scored (score_step16_tahoe.MIN_CTX)
N_RANDOM, SEED0, N_NETWORK = 200, 20261113, 50                       # registered: assignments, first seed, assignments for A7
THRESHOLD = 0.95                                                     # share of assignments that must give the verdict of the paper
TOL, NET_TOL = 1e-9, 0.005                                           # reproduction of committed values; of the network
COMPARATOR, ORIGINAL = "fm_50", "base"                               # rewired model (50 steps) and original model in the records
REFERENCES = ("ridge_S7", "ridge_train", "mean_response", "control_paste")
GENERATIVE = {                                                        # the generative models of Supplementary Table S6
    "drug_split_1": ("base", "adaln_ddim_1000", "fm_20", "fm_50", "guidance_w2", "attn_cross", "mlp_ctrl",
                     "s9_pathway", "s9_both", "s10_chemvae", "s10_fp_chemvae"),
    "drug_split_0": ("base", "adaln_ddim_1000", "fm_20", "fm_50", "guidance_w2", "attn_cross", "mlp_ctrl"),
    "drug_split_2": ("base", "adaln_ddim_1000", "fm_20", "fm_50", "guidance_w2", "attn_cross", "mlp_ctrl"),
}
# feedforward network of Results 3, as chosen in ridge_vs_mlp_v2.json (checked against that record below)
WIDTH, WEIGHT_DECAY, EPOCHS, LR, BATCH, NET_SEEDS = 64, 10.0, 400, 1e-3, 32, (0, 1, 2)
NET_THREADS = 16      # threads of the training library in the job that wrote ridge_vs_mlp_v2.json (run_ridge_vs_mlp_v2.sh).
                      # Dated paragraph of 15 Nov 2026 in the pre-registration: with 8 threads some trainings came out
                      # identical to the record and some did not, so the network of guard 3f runs with 16.

# The verdict of every item under the folds of the paper, as section 4 of the pre-registration states it.
# "ci" is the reading of the interval: 'above zero', 'below zero' or 'includes zero'.
STATED = {
    "A1": {s: {"sign": "positive", "ci": "above zero"} for s in SPLITS},
    "A2": {s: {"sign": "positive", "ci": "above zero"} for s in ("drug_split_0", "drug_split_2")},
    "A3": {s: {"sign": "negative"} for s in SPLITS},
    "A4": {s: {"order": "every generative model below the ridge"} for s in SPLITS},
    "A5": {"drug_split_1": {"order": "every generative model below the ridge"},
           "drug_split_0": {"order": "every generative model below the ridge"},
           "drug_split_2": {"order": "cross-attention and its control above the ridge"}},
    "A6": {"drug_split_1": {"sign": "positive", "ci": "above zero"},
           "drug_split_0": {"sign": "positive", "ci": "includes zero"},
           "drug_split_2": {"sign": "positive", "ci": "includes zero"}},
    "A7": {"drug_split_0": {"sign": "negative"}},
    "C4": {s: {"ci": "includes zero"} for s in SPLITS},
}


QUIET = False                                                         # set in worker processes: no repeated notes


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
def verdict(lo, hi):                                                  # score_step14.py
    return "above zero" if lo > 0 else "below zero" if hi < 0 else "includes zero"


def jac(a, b):                                                        # score_step14.py
    i = len(np.intersect1d(a, b)); return i / (len(a) + len(b) - i)


def thesis_per_cond(true, B, C):                                      # score_step14.py, verbatim
    """DE-overlap Jaccard and DPC-50 per condition from mean vectors (true, predicted B, paired control C)."""
    from scipy.stats import pearsonr
    n = len(true); j = np.full(n, np.nan); d = np.full(n, np.nan)
    for i in range(n):
        td = true[i] - C[i]; p = B[i] - C[i]
        g = np.argsort(np.abs(td))[-K:]
        if np.abs(p).std() > 1e-12: j[i] = jac(g, np.argsort(np.abs(p))[-K:])
        if td[g].std() > 1e-12 and p[g].std() > 1e-12: d[i] = pearsonr(p[g], td[g])[0]
    return j, d


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


class Boot:                                                           # score_step23_screen.py, verbatim (two methods)
    """Structure-clustered percentile bootstrap over one set of conditions. Clusters are the structures (SMILES) in the
    set, in sorted order; the draws are those of score_step15_tahoe.boot (rng.randint(0, G, G) per replicate) from a
    fresh RandomState(0), so every interval of one set uses the same 5,000 resamples."""

    def __init__(self, idx, clusters, n_total):
        self.idx = np.asarray(idx)
        lab = np.asarray(clusters)[self.idx]
        self.labels = np.unique(lab)
        self.G = len(self.labels)
        self.member = np.searchsorted(self.labels, lab)              # cluster number of each condition in the set
        self.groups = [self.idx[self.member == g] for g in range(self.G)]
        self.mem_full = np.full(n_total, -1)
        self.mem_full[self.idx] = self.member
        rng = np.random.RandomState(0)
        self.D = np.array([rng.randint(0, self.G, self.G) for _ in range(N_BOOT)])
        self.cnt = np.zeros((N_BOOT, self.G))                      # how often each cluster is drawn, per replicate
        np.add.at(self.cnt, (np.repeat(np.arange(N_BOOT), self.G), self.D.ravel()), 1.0)

    def mean_reps(self, x):
        """Replicate means of per-condition values x over the set; the values are fixed under resampling."""
        v = np.asarray(x, np.float64)[self.idx]
        if not np.isfinite(v).all():
            die("non-finite per-condition value inside a scored set")
        S = np.bincount(self.member, weights=v, minlength=self.G)
        N = np.bincount(self.member, minlength=self.G).astype(np.float64)
        return float(v.mean()), (self.cnt @ S) / (self.cnt @ N)


def pct(reps):                                                        # score_step28_salt_twins.py
    """The published two-sided 97.5 % percentile interval of a replicate array."""
    return float(np.percentile(reps, LO)), float(np.percentile(reps, HI))


# ----------------------------------------------------------------------------- loading, as score_step17_sciplex_mra.py
def check_prov(pv, name, checks):
    """A provenance key that exists must carry the expected value; keys absent from a record are reported, not assumed."""
    for k, v in checks.items():
        if k not in pv:
            if not QUIET:
                print("  [note] %s: provenance has no key %r (expected %r)" % (name, k, v))
            continue
        got = pv[k]
        ok = (abs(float(got) - float(v)) < 1e-9) if isinstance(v, (int, float)) else (str(got) == str(v))
        if not ok:
            die("%s provenance %s=%r, expected %r" % (name, k, got, v))


def load_vectors(name, pj, checks):
    """Mean vectors of one evaluation (cond, true, B, C), hash-checked against its provenance record."""
    f = os.path.join(REPO, RES, "vectors_%s.npz" % name); pjp = os.path.join(REPO, pj)
    for q in (f, pjp):
        if not os.path.isfile(q):
            die("missing %s" % q)
    pv = json.load(open(pjp)); check_prov(pv, name, checks)
    digest = sha256(f)
    if pv.get("vectors_sha256") != digest:
        die("%s vectors hash differs from provenance" % name)
    v = np.load(f, allow_pickle=True)
    return dict(cond=np.array([str(x) for x in v["cond"]]), true=np.asarray(v["true"], np.float64),
                B=np.asarray(v["B"], np.float64), C=np.asarray(v["C"], np.float64),
                rec=dict(tag=name, vectors_sha256=digest, checkpoint_sha256=pv.get("checkpoint_sha256")))


def rewired_registry(split):
    """Vectors name, provenance path and checks of the rewired model at 50 steps (registry of score_step17_sciplex_mra.py)."""
    if split == "drug_split_1":
        return ("sw_fm_gaussian_seed_%d_n50", "provenance/s6sw_sw_fm_gaussian_seed_%d_n50.json", dict(arm="fm_gaussian", steps=50))
    s = split
    return ("s11_%s_fm_gaussian_nfe50_seed_%%d" % s, "provenance/s11ev_s11_%s_fm_gaussian_nfe50_seed_%%d.json" % s, dict(split=s, nfe=50))


def step7_target(split):
    """The file the Step 7 ridge was fitted on, and its provenance record (score_step14.py)."""
    if split == "drug_split_1":
        return ("vectors_step6_fm_gaussian_seed_0.npz", "provenance/s6eval_step6_fm_gaussian_seed_0.json")
    return ("vectors_s6b_%s_fm_gaussian_seed_0.npz" % split, "provenance/s6bev_s6b_%s_fm_gaussian_seed_0.json" % split)


def load_split(split, data_dir):
    """Everything one split needs, read and checked as score_step14.py and score_step17_sciplex_mra.py do."""
    import anndata as ad
    from Squidiff.scrna_datasets import Drug_dose_encoder             # the harness's fingerprint + dose encoder
    vec, pj, checks = rewired_registry(split)
    arms = [load_vectors(vec % s, pj % s, checks) for s in range(N_SEEDS)]
    cond = arms[0]["cond"]; n = len(cond)
    for v in arms:
        if not np.array_equal(v["cond"], cond): die("condition order differs in %s" % v["rec"]["tag"])
        if np.abs(v["C"] - arms[0]["C"]).max() > 1e-5: die("paired-control means differ in %s" % v["rec"]["tag"])
        if np.abs(v["true"] - arms[0]["true"]).max() > 1e-5: die("true means differ in %s" % v["rec"]["tag"])
    true_mean, C = arms[0]["true"], arms[0]["C"]

    tp = os.path.join(data_dir, "sci_plex_test_%s.h5ad" % split)
    if not os.path.isfile(tp): die("missing " + tp)
    obs = ad.read_h5ad(tp).obs
    g_all = obs["Group"].astype(str).values; smi_all = obs["SMILES"].astype(str).values
    dose_all = obs["dose"].astype(float).values; line_all = obs["cell_name"].astype(str).values
    for c in cond:
        if not (g_all == c).any(): die("%s: condition %r of the vectors file is absent from the test file" % (split, c))
    drugs = np.array([smi_all[g_all == c][0] for c in cond]); lines = np.array([line_all[g_all == c][0] for c in cond])
    doses = np.array([float(np.median(dose_all[g_all == c])) for c in cond])
    if len(np.unique(drugs)) != N_DRUGS[split]: die("%s: %d drugs, expected %d" % (split, len(np.unique(drugs)), N_DRUGS[split]))

    # the Step 7 target, hash-checked; float32 as stored (score_step14.py)
    f7, e7 = step7_target(split)
    f7p, e7p = os.path.join(REPO, RES, f7), os.path.join(REPO, e7)
    for q in (f7p, e7p):
        if not os.path.isfile(q): die("missing " + q)
    sha7 = sha256(f7p)
    if json.load(open(e7p)).get("vectors_sha256") != sha7: die("%s changed since Step 7" % f7)
    with np.load(f7p, allow_pickle=True) as v:
        Y7 = np.asarray(v["true"]) - np.asarray(v["C"]); cond7 = np.array([str(x) for x in v["cond"]])
    if not np.array_equal(cond7, cond): die("Step 7 target condition order differs from the rewired model on %s" % split)

    # features, as score_step14.py builds X7 (the encoder's output is not cast)
    enc = [Drug_dose_encoder([smi_all[g_all == c][0]], [float(np.median(dose_all[g_all == c]))], comb_num=1)[0] for c in cond]
    FP7 = np.array(enc)
    lv = sorted(set(lines)); Lmat = np.array([[float(l == u) for u in lv] for l in lines])
    X7 = np.hstack([FP7, Lmat, np.concatenate([FP7 * Lmat[:, [k]] for k in range(len(lv))], axis=1)])
    if X7.shape[1] != 4099: die("%s: %d features, expected 4,099" % (split, X7.shape[1]))
    # the same features in single precision, as ridge_vs_mlp_v2.py builds them (for the network, item A7)
    FP32 = np.array(enc, dtype=np.float32)
    L32 = np.array([[float(l == u) for u in lv] for l in lines], dtype=np.float32)
    X32 = np.hstack([FP32, L32] + [FP32 * L32[:, [k]] for k in range(len(lv))]).astype(np.float32)
    Y32 = Y7.astype(np.float32)

    # compounds in sorted order of their SMILES strings: the order of the fold strings
    compounds = sorted(set(drugs.tolist()))
    index = {s: k for k, s in enumerate(compounds)}
    comp = np.array([index[s] for s in drugs])                        # compound number of every condition
    counts = np.bincount(comp, minlength=len(compounds))
    return dict(cond=cond, n=n, true_mean=true_mean, C=C, arms=arms, drugs=drugs, lines=lines, doses=doses, lv=lv,
                Y7=Y7, X7=X7, X32=X32, Y32=Y32, compounds=compounds, comp=comp, counts=counts,
                inputs=dict(test_h5ad=tp, step7_target=f7, step7_target_sha256=sha7, rewired=[v["rec"] for v in arms],
                            n_conditions=int(n), n_compounds=int(len(compounds)), n_features=int(X7.shape[1]), lines=lv,
                            encoder_dtype=str(FP7.dtype), target_dtype=str(Y7.dtype)))


# ----------------------------------------------------------------------------- folds (pathway_vs_fingerprint_v2.py, verbatim)
def deal(counts, order):
    """GroupKFold's own rule for a given order of the groups: each group goes to the fold that is
    lightest so far (the first one if several are equally light). Returns the fold of each group."""
    weight = np.zeros(NFOLD)
    fold_of_group = np.zeros(len(counts), dtype=int)
    for k in order:
        f = int(np.argmin(weight))
        weight[f] += counts[k]
        fold_of_group[k] = f
    return fold_of_group


def canonical(fold_of_group):
    """Digit string with the folds renumbered by first appearance, so that one partition always
    has one spelling."""
    remap, out = {}, []
    for f in fold_of_group:
        if int(f) not in remap:
            remap[int(f)] = len(remap)
        out.append(str(remap[int(f)]))
    return "".join(out)


def random_fold_of_group(counts, seed):
    """One random assignment: the groups in random order, then largest first (stable, so that ties
    keep the random order), then GroupKFold's rule."""
    perm = np.random.RandomState(seed).permutation(len(counts))
    order = perm[np.argsort(-counts[perm], kind="stable")]
    return deal(counts, order)


# ----------------------------------------------------------------------------- the ridge of the paper under one assignment
def ridge_paper(X7, Y7, fold):
    """Cross-validated predictions of the ridge, operation for operation as score_step14.py: features in double
    precision, target and stored predictions in single precision, sklearn Ridge with penalty 1. 'fold' is the fold
    of every condition; the numbering of the folds does not matter here."""
    from sklearn.linear_model import Ridge
    P7 = np.zeros_like(Y7)                                            # float32 store, as score_step14.py
    for f in range(NFOLD):
        b_, a_ = np.where(fold == f)[0], np.where(fold != f)[0]
        if len(b_) == 0: die("an empty fold")
        P7[b_] = Ridge(alpha=ALPHA).fit(X7[a_], Y7[a_]).predict(X7[b_])
    return P7


def anchor_median(P7, Y7):
    """The Step 7 quantity: median over conditions of the response correlation, on the single-precision arrays
    (score_step14.py, lines of 'd7')."""
    from scipy.stats import pearsonr
    n = len(Y7); d7 = np.full(n, np.nan)
    for i in range(n):
        d = np.argsort(np.abs(Y7[i]))[-K:]
        if Y7[i][d].std() > 1e-12 and P7[i][d].std() > 1e-12: d7[i] = pearsonr(P7[i][d], Y7[i][d])[0]
    return float(np.nanmedian(d7))


# ----------------------------------------------------------------------------- the network of Results 3 (ridge_vs_mlp_v2.py, verbatim)
def dpc50_mlp(pred, Y):
    """Per condition: Pearson over the 50 genes with the largest |measured change|.
    Returns (median, mean) over conditions - the paper's response correlation."""
    r = []
    for i in range(Y.shape[0]):
        idx = np.argsort(np.abs(Y[i]))[-K:]
        a, b = pred[i, idx].astype(np.float64), Y[i, idx].astype(np.float64)
        if a.std() == 0 or b.std() == 0:
            r.append(0.0)
        else:
            r.append(float(np.corrcoef(a, b)[0, 1]))
    r = np.array(r)
    return float(np.median(r)), float(np.mean(r))


def ridge_cv(X, Y, fold_list):
    from sklearn.linear_model import Ridge
    pred = np.zeros_like(Y)
    for tr, te in fold_list:
        pred[te] = Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])
    return pred


def mlp_cv(X, Y, fold_list, width, wd, epochs, seed):
    """Cross-validated predictions of a one-hidden-layer network; one seed."""
    import torch as th
    import torch.nn as nn
    pred = np.zeros_like(Y)
    for f, (tr, te) in enumerate(fold_list):
        th.manual_seed(seed * 1000 + f)                      # weights
        rs = np.random.RandomState(seed * 1000 + f)           # minibatch order
        xa, xb = X[tr], X[te]
        keep = xa.std(0) > 1e-9                               # drop constant columns
        xa, xb = xa[:, keep], xb[:, keep]
        mu, sd = xa.mean(0, keepdims=True), xa.std(0, keepdims=True) + 1e-6
        za, zb = th.tensor((xa - mu) / sd), th.tensor((xb - mu) / sd)
        ya = th.tensor(Y[tr])
        m = nn.Sequential(nn.Linear(za.shape[1], width), nn.ReLU(), nn.Linear(width, Y.shape[1]))
        opt = th.optim.AdamW(m.parameters(), lr=LR, weight_decay=wd)
        n = za.shape[0]
        for _ in range(epochs):
            order = rs.permutation(n)
            for s in range(0, n, BATCH):
                idx = th.tensor(order[s:s + BATCH])
                opt.zero_grad()
                loss = ((m(za[idx]) - ya[idx]) ** 2).mean()
                loss.backward()
                opt.step()
        with th.no_grad():
            pred[te] = m(zb).numpy()
    return pred


def fold_list_of(fold):
    """(train rows, test rows) for fold 0 to 4, rows in ascending order - the list GroupKFold yields."""
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(NFOLD)]


def network_minus_ridge(d, fold_raw, epochs):
    """Item A7 for one assignment: the median response correlation of the network (mean over its three seeds)
    and of its ridge, both fitted in single precision with the same folds as ridge_vs_mlp_v2.py does."""
    fl = fold_list_of(fold_raw)
    ridge_med = dpc50_mlp(ridge_cv(d["X32"], d["Y32"], fl), d["Y32"])[0]
    seeds = [dpc50_mlp(mlp_cv(d["X32"], d["Y32"], fl, WIDTH, WEIGHT_DECAY, epochs, s), d["Y32"])[0] for s in NET_SEEDS]
    return dict(network=float(np.mean(seeds)), network_per_seed=[float(x) for x in seeds], ridge=float(ridge_med),
                diff=float(np.mean(seeds) - ridge_med))


_WORKER_DATA = {}                                                     # data of the splits, loaded once per worker process


def network_task(task):
    """Item A7 for one random assignment on one split, in a worker process: one thread, its own copy of the data.
    task = (split, number of the assignment, epochs, data directory). Every training is seeded inside mlp_cv, so
    the result is the same whichever process runs it."""
    global QUIET
    sp, k, epochs, data_dir = task
    import torch as th
    th.set_num_threads(1)
    if sp not in _WORKER_DATA:
        QUIET = True
        _WORKER_DATA[sp] = load_split(sp, data_dir)
    d = _WORKER_DATA[sp]
    g = random_fold_of_group(d["counts"], SEED0 + k)                  # GroupKFold's own fold numbers for this order
    return sp, k, network_minus_ridge(d, g[d["comp"]], epochs)


# ----------------------------------------------------------------------------- small helpers
def summary(values, paper):
    """Median, percentiles and range of one quantity over the random assignments, and where the value of the
    paper falls among them."""
    v = np.asarray(values, dtype=np.float64)
    return {"paper": float(paper), "median": float(np.median(v)), "p2.5": float(np.percentile(v, 2.5)),
            "p97.5": float(np.percentile(v, 97.5)), "min": float(v.min()), "max": float(v.max()),
            "n_random_not_above_paper": int((v <= paper).sum()), "n": int(len(v))}


def need_of(n):
    """Smallest number of assignments that is at least 95 % of n."""
    return int(math.ceil(THRESHOLD * n - 1e-9))


def close(a, b, what, tol=TOL):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        die("%s: shapes differ (%s and %s)" % (what, a.shape, b.shape))
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        die("%s: undefined values are not in the same places" % what)
    worst = float(np.nanmax(np.abs(a - b))) if a.size and np.isfinite(a).any() else 0.0
    if worst > tol:
        die("%s: largest difference %.3e, tolerance %g" % (what, worst, tol))
    return worst


def per_condition_check(mine, record, tol=TOL):
    """Guard 3b for one measure: how many conditions differ from the record by more than tol, the largest difference,
    and the (up to five) largest ones as (position, value here, value in the record). None if the two arrays do not
    have their undefined values in the same places."""
    a, b = np.asarray(mine, np.float64), np.asarray(record, np.float64)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return None
    diff = np.nan_to_num(np.abs(a - b), nan=0.0)
    order = np.argsort(-diff, kind="stable")[:5]
    return int((diff > tol).sum()), (float(diff.max()) if diff.size else 0.0), [(int(i), float(a[i]), float(b[i])) for i in order if diff[i] > tol]


# ----------------------------------------------------------------------------- per-assignment computation
def score_assignment(d, m, fold):
    """Everything of one fold assignment on one split, except the response-magnitude intervals and the network.
    d: the data of the split; m: the committed values of the models; fold: fold of every condition."""
    P7 = ridge_paper(d["X7"], d["Y7"], fold)
    D = P7.astype(np.float64)                                         # the ridge-S7 delta of score_step14.py
    j, dp = thesis_per_cond(d["true_mean"], d["C"] + D, d["C"])       # gene identity and response correlation per condition
    out = dict(P7=P7, jac=j, dpc=dp, norm=np.linalg.norm(D, axis=1),  # the norm of the delta itself, as score_step17
               gi=float(np.nanmean(j)), dpc_mean=float(np.nanmean(dp)), dpc_median=float(np.nanmedian(dp)))
    # A1: rewired model minus ridge, all conditions; A2: the same without the twin compound
    point, reps = m["boot_all"].mean_reps(m["jac"][COMPARATOR] - j)
    lo, hi = pct(reps)
    out["A1"] = dict(diff=point, lo=lo, hi=hi, ci=verdict(lo, hi))
    if m["boot_ret"] is not None:
        point, reps = m["boot_ret"].mean_reps(m["jac"][COMPARATOR] - j)
        lo, hi = pct(reps)
        out["A2"] = dict(diff=point, lo=lo, hi=hi, ci=verdict(lo, hi))
    # A3: original model minus ridge (means over conditions)
    out["A3"] = dict(diff=float(np.nanmean(m["jac"][ORIGINAL])) - out["gi"])
    # A4, A5: the generative models against the ridge on the mean and on the median over conditions
    out["A4"] = dict(ridge=out["dpc_mean"], highest_model=m["gen_mean_max"], diff=m["gen_mean_max"] - out["dpc_mean"])
    out["A5"] = dict(ridge=out["dpc_median"], highest_model=m["gen_median_max"], diff=m["gen_median_max"] - out["dpc_median"],
                     attn_cross=m["dpc_median"]["attn_cross"] - out["dpc_median"],
                     mlp_ctrl=m["dpc_median"]["mlp_ctrl"] - out["dpc_median"])
    return out


def same_verdict(item, split, r):
    """Does one assignment give, on one split, the verdict the pre-registration states for the item?"""
    st = STATED[item][split]
    if item in ("A1", "A2"):
        return bool(r[item]["diff"] > 0 and r[item]["ci"] == st["ci"])
    if item == "A3":
        return bool(r["A3"]["diff"] < 0)
    if item == "A4":
        return bool(r["A4"]["diff"] < 0)
    if item == "A5":
        if split == "drug_split_2":
            return bool(r["A5"]["attn_cross"] > 0 and r["A5"]["mlp_ctrl"] > 0)
        return bool(r["A5"]["diff"] < 0)
    if item == "A6":
        return bool(r["A6"]["mra"] > 0 and r["A6"]["ci"] == st["ci"])
    if item == "C4":
        return bool(r["C4"]["ci"] == st["ci"])
    if item == "A7":
        return bool(r["A7"]["diff"] < 0)
    die("unknown item " + item)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=RES)
    ap.add_argument("--n-random", type=int, default=N_RANDOM, help="test only")
    ap.add_argument("--n-network", type=int, default=N_NETWORK, help="test only")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="test only")
    ap.add_argument("--note1-sha256", default=NOTE1_SHA256, help="test only")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="processes for the networks of the random assignments (default: the CPUs of the job)")
    args = ap.parse_args()
    t0 = time.time()
    registered = (args.n_random == N_RANDOM and args.n_network == N_NETWORK and args.epochs == EPOCHS
                  and args.note1_sha256 == NOTE1_SHA256 and args.data_dir == DATA_DIR)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    if not registered and os.path.abspath(out_dir) == os.path.abspath(os.path.join(REPO, RES)):
        die("an option differs from the registered settings; such a run must write elsewhere (--out-dir)")
    if args.n_network > args.n_random or args.n_random < 1 or args.n_network < 1:
        die("the network needs between 1 and --n-random assignments")
    out_json, out_csv = os.path.join(out_dir, OUT_JSON), os.path.join(out_dir, OUT_CSV)

    # ---- 1. guards -------------------------------------------------------------------------------------
    for q in (out_json, out_csv):
        if os.path.exists(q):
            die("%s exists; refusing to overwrite a record" % q)
    dirty = sh("git -C '%s' status --porcelain --untracked-files=no" % REPO)
    if dirty:
        die("tracked files are modified; commit first.\n" + dirty)
    me = os.path.basename(os.path.abspath(__file__))
    for q in TRACKED + [me]:
        if not os.path.isfile(os.path.join(REPO, q)):
            die("missing " + q)
        if subprocess.run("git -C '%s' ls-files --error-unmatch '%s'" % (REPO, q), shell=True, capture_output=True).returncode != 0:
            die("%s is not committed" % q)
    commit = sh("git -C '%s' rev-parse HEAD" % REPO)
    print("[guard 1] no output exists; no tracked file is modified; the pre-registration, this script and the %d records "
          "it reads are committed (commit %s)" % (len(TRACKED) - 1, commit[:7]))
    if sha256(os.path.join(REPO, REC_NOTE1)) != args.note1_sha256:
        die("%s is not the record the folds of the paper were taken from" % REC_NOTE1)

    import pandas as pd
    note1 = json.load(open(os.path.join(REPO, REC_NOTE1)))
    diag = json.load(open(os.path.join(REPO, REC_DIAG)))
    step7 = json.load(open(os.path.join(REPO, REC_S7)))
    rec14 = json.load(open(os.path.join(REPO, REC14)))
    rec17 = json.load(open(os.path.join(REPO, REC17)))
    rec28 = json.load(open(os.path.join(REPO, REC28)))
    mlp = json.load(open(os.path.join(REPO, REC_MLP)))
    pc14 = pd.read_csv(os.path.join(REPO, PC14))
    pq28 = pd.read_csv(os.path.join(REPO, PQ28))
    fsens = note1["fold_sensitivity"]
    if fsens["seeds"][0] != SEED0 or fsens["n_assignments"] != N_RANDOM:
        die("the Note 1 record does not hold %d assignments from seed %d" % (N_RANDOM, SEED0))
    if rec14.get("comparator") != COMPARATOR or rec28.get("comparator") != COMPARATOR or rec17.get("comparator") != COMPARATOR:
        die("a record names another comparator than %s" % COMPARATOR)
    if mlp["chosen"]["weight_decay"] != WEIGHT_DECAY or mlp["chosen"]["epochs"] != EPOCHS or WIDTH not in mlp["settings"]["widths"] \
            or list(mlp["settings"]["seeds"]) != list(NET_SEEDS):
        die("the settings of the network differ from ridge_vs_mlp_v2.json")

    flags = " " + sh("grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2") + " "
    env = {"node": platform.node(), "cpu": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
           "cpu_has_avx512f": " avx512f " in flags, "python": platform.python_version(), "numpy": np.__version__,
           "scipy": version_of("scipy"), "sklearn": version_of("scikit-learn"), "anndata": version_of("anndata"),
           "pandas": version_of("pandas"), "torch": version_of("torch"), "threads": os.environ.get("OMP_NUM_THREADS", ""),
           "numpy_features_switched_off": NUMPY_SWITCHED_OFF, "restarted_without_avx512": os.environ.get("STEP29_RESTARTED") == "1",
           "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
    try:                                                               # which linear-algebra code runs here (it can differ between node types)
        import importlib
        importlib.import_module("scipy.linalg")                       # numpy and scipy each carry their own copy of the library; list both
        from threadpoolctl import threadpool_info
        env["linear_algebra"] = [{k: (os.path.basename(str(i.get(k))) if k == "filepath" else i.get(k))
                                  for k in ("user_api", "internal_api", "version", "architecture", "num_threads", "filepath")}
                                 for i in threadpool_info()]
    except Exception:
        env["linear_algebra"] = []
    import torch as th
    th.set_num_threads(NET_THREADS)                                    # as the job of ridge_vs_mlp_v2.py had it
    env["torch_threads"] = int(th.get_num_threads())
    print("[env] %s" % env)

    # ---- 2. data, folds, committed values of the models --------------------------------------------------
    data, models, folds, guards = {}, {}, {}, {s: {} for s in SPLITS}
    for sp in SPLITS:
        d = load_split(sp, args.data_dir)
        n = d["n"]
        # folds of the paper (Note 1 record) and their layout
        nf = note1["folds"][sp]
        if list(nf["compounds_sorted_smiles"]) != d["compounds"]:
            die("%s: the compounds differ from those of the Note 1 record" % sp)
        if [int(c) for c in nf["conditions_per_compound"]] != [int(c) for c in d["counts"]]:
            die("%s: conditions per compound differ from the Note 1 record" % sp)
        paper_group = np.array([int(c) for c in nf["fold_of_compound"]])
        paper_string = canonical(paper_group)
        # the 200 random assignments, by the rule of the Note 1 record; guard 2
        rand_group = [random_fold_of_group(d["counts"], SEED0 + r) for r in range(args.n_random)]
        strings = [canonical(g) for g in rand_group]
        if strings != list(fsens["per_split"][sp]["assignments"][:args.n_random]):
            die("%s: the random assignments built here differ from those stored in the Note 1 record" % sp)
        if paper_string in strings:
            die("%s: the assignment of the paper is among the random ones" % sp)
        # GroupKFold's own fold numbers of the paper's assignment (they seed the networks of ridge_vs_mlp_v2.py):
        # the dealing order numpy gives without AVX-512 code, stored by the diagnostic of 13 Nov 2026
        ds = diag["splits"][sp]
        order = ds.get("tie_orders", {}).get("no512")
        if order is None or sorted(int(x) for x in order) != list(range(len(d["compounds"]))):
            die("%s: the diagnostic record holds no dealing order without AVX-512 code" % sp)
        if [int(c) for c in ds["group_sizes"]] != [int(c) for c in d["counts"]] or list(ds["compounds_sorted"]) != d["compounds"]:
            die("%s: the diagnostic record describes other compounds" % sp)
        paper_raw = deal(d["counts"], [int(x) for x in order])
        if canonical(paper_raw) != paper_string:
            die("%s: the dealing order of the diagnostic record does not give the folds of the paper" % sp)
        folds[sp] = dict(paper=paper_group[d["comp"]], paper_raw=paper_raw[d["comp"]], paper_string=paper_string,
                         paper_raw_string="".join(str(int(x)) for x in paper_raw),
                         random=[g[d["comp"]] for g in rand_group], strings=strings)
        print("[data] %s: %d conditions, %d compounds, %d features (encoder %s, target %s); folds of the paper %s; "
              "%d random assignments equal those of the Note 1 record (guard 2)"
              % (sp, n, len(d["compounds"]), d["X7"].shape[1], d["inputs"]["encoder_dtype"], d["inputs"]["target_dtype"],
                 paper_string, args.n_random))

        # committed per-condition values of every model, in the condition order of this script
        sub = pc14[pc14["split"] == sp]
        names = sorted(set(sub["submission"].astype(str)))
        if set(names) != set(GENERATIVE[sp]) | set(REFERENCES):
            die("%s: the per-condition record holds %s" % (sp, names))
        jac_, dpc_ = {}, {}
        for nm in names:
            one = sub[sub["submission"].astype(str) == nm]
            if not np.array_equal(one["condition"].astype(str).values, d["cond"]):
                die("%s: the conditions of %s in the per-condition record are not those of the vectors files" % (sp, nm))
            if not np.array_equal(one["structure"].astype(str).values, d["drugs"]) or \
                    not np.array_equal(one["line"].astype(str).values, d["lines"]):
                die("%s: structure or cell line of %s differ from the test file" % (sp, nm))
            jac_[nm] = one["de_overlap"].astype(float).values
            dpc_[nm] = one["dpc50"].astype(float).values
        # guard 3c: the per-condition record gives back the committed means and medians
        tm, worst = rec14["per_split"][sp]["thesis_metrics"], 0.0
        for nm in names:
            for key, val in (("de_overlap", np.nanmean(jac_[nm])), ("dpc_mean", np.nanmean(dpc_[nm])), ("dpc_median", np.nanmedian(dpc_[nm]))):
                want = tm[nm].get(key)
                if want is None or not np.isfinite(want):
                    if np.isfinite(val): die("%s %s %s: defined here, undefined in step14_analysis.json" % (sp, nm, key))
                    continue
                worst = max(worst, abs(float(val) - float(want)))
                if abs(float(val) - float(want)) > TOL:
                    die("%s %s %s: %.12f from the per-condition record, %.12f committed" % (sp, nm, key, val, want))
        guards[sp]["models_vs_step14_record"] = worst
        gen = GENERATIVE[sp]
        all_idx = np.arange(n)
        ex = [e["matched_structure"] for e in rec28["splits"][sp]["excluded"]]
        if (sp in STATED["A2"]) != (len(ex) == 1):
            die("%s: %d excluded compounds in the Step 28 record" % (sp, len(ex)))
        retained = np.flatnonzero(~np.isin(d["drugs"], ex)) if ex else None
        if ex and (n - len(retained) != rec28["splits"][sp]["excluded_conditions"] or len(retained) != rec28["splits"][sp]["retained_conditions"]):
            die("%s: the retained conditions are not those of the Step 28 record" % sp)
        models[sp] = dict(jac=jac_, dpc=dpc_, names=names,
                          dpc_mean={nm: float(np.nanmean(dpc_[nm])) for nm in names},
                          dpc_median={nm: float(np.nanmedian(dpc_[nm])) for nm in names},
                          gen_mean_max=max(float(np.nanmean(dpc_[nm])) for nm in gen),
                          gen_median_max=max(float(np.nanmedian(dpc_[nm])) for nm in gen),
                          gen_mean_argmax=max(gen, key=lambda nm: float(np.nanmean(dpc_[nm]))),
                          gen_median_argmax=max(gen, key=lambda nm: float(np.nanmedian(dpc_[nm]))),
                          boot_all=Boot(all_idx, d["drugs"], n), boot_ret=Boot(retained, d["drugs"], n) if ex else None,
                          excluded=ex, n_retained=int(len(retained)) if ex else int(n))
        if models[sp]["boot_all"].G != N_DRUGS[sp]:
            die("%s: %d compound clusters, expected %d" % (sp, models[sp]["boot_all"].G, N_DRUGS[sp]))
        print("[guard 3c] %s: means and medians of the %d models of the per-condition record equal step14_analysis.json "
              "(largest difference %.1e)" % (sp, len(names), worst))
        data[sp] = d

    # ---- 3. response-magnitude agreement: contexts and the resamples of score_step17_sciplex_mra.py --------
    mra_ctx = {}
    for sp in SPLITS:
        d = data[sp]; n = d["n"]
        ctx_key = np.array(["%s|%.6f" % (d["lines"][i], d["doses"][i]) for i in range(n)])
        ctx_u = np.unique(ctx_key); n_ctx = len(ctx_u)
        ctx_id = np.array([int(np.where(ctx_u == k)[0][0]) for k in ctx_key])
        true_norm = np.linalg.norm(d["true_mean"] - d["C"], axis=1)
        fm_norms = [np.linalg.norm(v["B"] - d["C"], axis=1) for v in d["arms"]]   # the rewired model, per seed
        pre_full = prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, np.arange(n)))
        rng = np.random.RandomState(0)                                 # one generator per split, as score_step17
        gs = [np.where(d["drugs"] == u)[0] for u in np.unique(d["drugs"])]
        if len(gs) != N_DRUGS[sp]: die("%s: %d compound clusters, expected %d" % (sp, len(gs), N_DRUGS[sp]))
        pres, fm_reps = [], np.full(N_BOOT, np.nan)
        for b in range(N_BOOT):
            sel = np.concatenate([gs[g] for g in rng.randint(0, len(gs), len(gs))])
            pre = prep_ctx(true_norm, ctx_split(ctx_id, n_ctx, sel))
            pres.append(pre)
            fm_reps[b] = float(np.mean([mra_pre(x, pre) for x in fm_norms]))
        mra_ctx[sp] = dict(pre_full=pre_full, pres=pres, fm=float(np.mean([mra_pre(x, pre_full) for x in fm_norms])),
                           fm_reps=fm_reps, n_ctx=int(n_ctx), n_ctx_used=int(sum(item is not None for item in pre_full)))
        print("[mra] %s: %d contexts (%d used), %d resamples prepared   [%.0f s]"
              % (sp, n_ctx, mra_ctx[sp]["n_ctx_used"], N_BOOT, time.time() - t0))

    def add_mra(sp, r):
        """Items A6 and C4 of one assignment: point and interval, the agreement recomputed in every resample."""
        c = mra_ctx[sp]
        reps = np.array([mra_pre(r["norm"], pre) for pre in c["pres"]])
        point = mra_pre(r["norm"], c["pre_full"])
        lo, hi = float(np.nanpercentile(reps, LO)), float(np.nanpercentile(reps, HI))
        r["A6"] = dict(mra=float(point), lo=lo, hi=hi, ci=verdict(lo, hi))
        dr = c["fm_reps"] - reps
        lo, hi = float(np.nanpercentile(dr, LO)), float(np.nanpercentile(dr, HI))
        r["C4"] = dict(diff=float(c["fm"] - point), lo=lo, hi=hi, ci=verdict(lo, hi))

    # ---- 4. the folds of the paper, and guards 3a, 3b, 3d, 3e, 3f, 3g -------------------------------------
    paper, failed_3b = {}, []
    for sp in SPLITS:
        d, m = data[sp], models[sp]
        r = score_assignment(d, m, folds[sp]["paper"])
        # a. the Step 7 record
        want = float(step7["per_split"][sp]["marginal"]["interaction"])
        got = anchor_median(r["P7"], d["Y7"])
        guards[sp]["step7_anchor"] = dict(step7_record=want, recomputed=got, abs_diff=abs(got - want))
        if abs(got - want) > TOL:
            die("guard 3a, %s: the ridge gives %.12f, the Step 7 record %.12f" % (sp, got, want))
        # b. the per-condition rows of the cross-validated ridge. A difference does not stop the run at once: it is
        #    printed for every split first, so that one run shows the whole picture; the run stops after the splits.
        for key, label, mine_, record_ in (("gene_identity", "gene identity", r["jac"], m["jac"]["ridge_S7"]),
                                           ("response_correlation", "response correlation", r["dpc"], m["dpc"]["ridge_S7"])):
            res = per_condition_check(mine_, record_)
            if res is None:
                die("guard 3b, %s, %s per condition: undefined values are not in the same places as in the record" % (sp, label))
            n_bad, worst, examples = res
            guards[sp]["ridge_per_condition_" + key] = worst
            if n_bad:
                failed_3b.append("%s, %s: %d of %d conditions, largest difference %.3e" % (sp, label, n_bad, d["n"], worst))
                print("[guard 3b] DIFFERS on %s, %s: %d of %d conditions differ from the per-condition record by more than %g; "
                      "largest difference %.3e. The largest:" % (sp, label, n_bad, d["n"], TOL, worst))
                for i, here, rec_ in examples:
                    print("            %-40s %.12f here, %.12f in the record (difference %+.3e)" % (d["cond"][i], here, rec_, here - rec_))
        # d. response-magnitude agreement (step17_mra.json)
        add_mra(sp, r)
        c17 = rec17["per_split"][sp]
        w = [close(r["A6"]["mra"], c17["metrics"]["ridge_S7"]["mra"], "guard 3d, %s, agreement of the ridge" % sp),
             close([r["A6"]["lo"], r["A6"]["hi"]], c17["contrasts"]["secondary_mra_ridge_S7_minus_0"]["ci"], "guard 3d, %s, its interval" % sp),
             close(mra_ctx[sp]["fm"], c17["metrics"][COMPARATOR]["mra"], "guard 3d, %s, agreement of the rewired model" % sp),
             close(r["C4"]["diff"], c17["contrasts"]["secondary_mra_fm_50_minus_ridge_S7"]["diff"], "guard 3d, %s, rewired minus ridge" % sp),
             close([r["C4"]["lo"], r["C4"]["hi"]], c17["contrasts"]["secondary_mra_fm_50_minus_ridge_S7"]["ci"], "guard 3d, %s, interval of rewired minus ridge" % sp)]
        guards[sp]["response_magnitude_vs_step17_record"] = max(w)
        # e. the contrasts of Supplementary Note 7 (Step 28 record)
        row = pq28[(pq28["split"] == sp) & (pq28["group"] == "B-contrast") & (pq28["quantity"] == "gene identity") & (pq28["arm"] == "rewired minus ridge_S7")]
        if len(row) != 1: die("guard 3e, %s: %d rows 'rewired minus ridge_S7' in the Step 28 record" % (sp, len(row)))
        row = row.iloc[0]
        w = [close([r["A1"]["diff"], r["A1"]["lo"], r["A1"]["hi"]], [row["published"], row["published_lo"], row["published_hi"]],
                   "guard 3e, %s, rewired minus ridge on all conditions" % sp)]
        if "A2" in r:
            if int(row["n_twin_free"]) != m["n_retained"]: die("guard 3e, %s: %d retained conditions in the Step 28 record" % (sp, int(row["n_twin_free"])))
            w.append(close([r["A2"]["diff"], r["A2"]["lo"], r["A2"]["hi"]], [row["twin_free"], row["twin_free_lo"], row["twin_free_hi"]],
                           "guard 3e, %s, rewired minus ridge without the twin compound" % sp))
        guards[sp]["note7_contrasts_vs_step28_record"] = max(w)
        # The interval of this contrast as Step 14 committed it (Table S5c). It was drawn there from a generator shared
        # between contrasts, so its bounds can differ from the recomputed ones in the third decimal (pre-registration,
        # section 3). It is printed and stored beside the recomputed interval and is never used for a verdict. The
        # point value does not depend on the generator and must agree (record: ridge minus rewired model, so the sign
        # and the bounds are turned round).
        c14 = rec14["per_split"][sp]["contrasts"]["primary_de_ridgeS7_minus_fm50"]
        committed = dict(diff=-float(c14["diff"]), lo=-float(c14["ci"][1]), hi=-float(c14["ci"][0]))
        guards[sp]["A1_point_vs_step14_contrast"] = close(r["A1"]["diff"], committed["diff"],
                                                         "guard 3c, %s, the committed contrast rewired minus ridge" % sp)
        r["A1"]["committed_step14"] = committed
        paper[sp] = r
        print("[guards 3a, 3b, 3d, 3e] %s: Step 7 record %.10f, recomputed %.10f (difference %.1e); per-condition rows "
              "to %.1e and %.1e; step17_mra.json to %.1e; Step 28 record to %.1e   [%.0f s]"
              % (sp, want, got, abs(got - want), guards[sp]["ridge_per_condition_gene_identity"],
                 guards[sp]["ridge_per_condition_response_correlation"], guards[sp]["response_magnitude_vs_step17_record"],
                 guards[sp]["note7_contrasts_vs_step28_record"], time.time() - t0))
    if failed_3b:
        die("guard 3b: per-condition values of the ridge differ from the per-condition record by more than %g (%s). "
            "Nothing is written." % (TOL, "; ".join(failed_3b)))
    # f. the network and its ridge under the folds of the paper (ridge_vs_mlp_v2.json)
    for sp in SPLITS:
        r = paper[sp]
        r["A7"] = network_minus_ridge(data[sp], folds[sp]["paper_raw"], args.epochs)
        rec = mlp["per_split"][sp]
        want_net, want_ridge = float(rec["mlp"][str(WIDTH)]["median_mean_over_seeds"]), float(rec["ridge"]["median"])
        guards[sp]["network_vs_record"] = dict(network_record=want_net, network=r["A7"]["network"], ridge_record=want_ridge,
                                              ridge=r["A7"]["ridge"], per_seed_record=[float(x["median"]) for x in rec["mlp"][str(WIDTH)]["seeds"]],
                                              per_seed=r["A7"]["network_per_seed"])
        print("[guard 3f] %s: network %.6f (record %.6f, difference %+.6f; per seed %s against %s); its ridge %.6f (record %.6f, "
              "difference %+.1e)   [%.0f s]"
              % (sp, r["A7"]["network"], want_net, r["A7"]["network"] - want_net, ["%.4f" % x for x in r["A7"]["network_per_seed"]],
                 ["%.4f" % float(x["median"]) for x in rec["mlp"][str(WIDTH)]["seeds"]], r["A7"]["ridge"], want_ridge,
                 r["A7"]["ridge"] - want_ridge, time.time() - t0))
        if abs(r["A7"]["network"] - want_net) > NET_TOL or abs(r["A7"]["ridge"] - want_ridge) > NET_TOL:
            die("guard 3f, %s: the network or its ridge does not give back ridge_vs_mlp_v2.json to %g" % (sp, NET_TOL))
    # g. the verdict of every item under the folds of the paper is the one the pre-registration states
    for item in sorted(STATED):
        for sp in STATED[item]:
            if not same_verdict(item, sp, paper[sp]):
                die("guard 3g: item %s on %s does not have, with the folds of the paper, the verdict the pre-registration "
                    "states: %s" % (item, sp, {k: v for k, v in paper[sp][item].items() if not isinstance(v, list)}))
    print("[guard 3g] every item has, with the folds of the paper, the verdict the pre-registration states")
    print("\n" + "=" * 100 + "\nALL GUARDS PASSED. Values with the folds of the paper:\n" + "=" * 100)
    for sp in SPLITS:
        r, m = paper[sp], models[sp]
        print("  %s  ridge: gene identity %.4f, response correlation mean %.4f median %.4f, response magnitude %+.4f"
              % (sp, r["gi"], r["dpc_mean"], r["dpc_median"], r["A6"]["mra"]))
        print("      A1 %+.4f [%+.4f, %+.4f]" % (r["A1"]["diff"], r["A1"]["lo"], r["A1"]["hi"])
              + ("   A2 %+.4f [%+.4f, %+.4f]" % (r["A2"]["diff"], r["A2"]["lo"], r["A2"]["hi"]) if "A2" in r else "")
              + "   A3 %+.4f   A4 %+.4f (%s)   A5 %+.4f (%s), cross-attention %+.4f, control %+.4f"
              % (r["A3"]["diff"], r["A4"]["diff"], m["gen_mean_argmax"], r["A5"]["diff"], m["gen_median_argmax"], r["A5"]["attn_cross"], r["A5"]["mlp_ctrl"]))
        print("      A6 %+.4f [%+.4f, %+.4f]   C4 %+.4f [%+.4f, %+.4f]   A7 network %.4f, ridge %.4f, difference %+.4f"
              % (r["A6"]["mra"], r["A6"]["lo"], r["A6"]["hi"], r["C4"]["diff"], r["C4"]["lo"], r["C4"]["hi"],
                 r["A7"]["network"], r["A7"]["ridge"], r["A7"]["diff"]))
        c = r["A1"]["committed_step14"]
        print("      A1 as Step 14 committed it (Table S5c; generator shared between contrasts): %+.4f [%+.4f, %+.4f]; "
              "recomputed bounds minus committed bounds: %+.1e, %+.1e" % (c["diff"], c["lo"], c["hi"], r["A1"]["lo"] - c["lo"], r["A1"]["hi"] - c["hi"]))

    # ---- 5. the random assignments ------------------------------------------------------------------------
    rand = {sp: [] for sp in SPLITS}
    for sp in SPLITS:
        d, m = data[sp], models[sp]
        for k in range(args.n_random):
            r = score_assignment(d, m, folds[sp]["random"][k])
            add_mra(sp, r)
            del r["P7"]                                                # keep memory small; the summaries stay
            rand[sp].append(r)
        print("[random] %s: %d assignments scored (ridge, gene identity, response correlation, response magnitude)   [%.0f s]"
              % (sp, args.n_random, time.time() - t0))
    # the networks of the first assignments: independent tasks, spread over the CPUs of the job (one thread each)
    tasks = [(sp, k, args.epochs, args.data_dir) for k in range(args.n_network) for sp in SPLITS]
    workers = max(1, min(int(args.workers), len(tasks)))
    print("[random] network: %d trainings of three seeds each on %d process(es)" % (len(tasks), workers))
    done = 0
    if workers == 1:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):                              # one thread, as in the worker processes below
            for sp, k, res in (network_task(task) for task in tasks):
                rand[sp][k]["A7"] = res; done += 1
                if done % 15 == 0 or done == len(tasks):
                    print("[random] network: %d of %d trainings   [%.0f s]" % (done, len(tasks), time.time() - t0))
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        saved = {v: os.environ.get(v) for v in thread_vars}
        for v in thread_vars:
            os.environ[v] = "1"                                        # the worker processes start with one thread each
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for sp, k, res in pool.map(network_task, tasks, chunksize=1):
                    rand[sp][k]["A7"] = res; done += 1
                    if done % 15 == 0 or done == len(tasks):
                        print("[random] network: %d of %d trainings   [%.0f s]" % (done, len(tasks), time.time() - t0))
        finally:
            for v in thread_vars:
                if saved[v] is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = saved[v]
    if any("A7" not in rand[sp][k] for sp in SPLITS for k in range(args.n_network)):
        die("a network result is missing")

    # ---- 6. counts and the rule ------------------------------------------------------------------------------
    items = {}
    for item in sorted(STATED):
        per_split, robust = {}, True
        for sp in STATED[item]:
            pool = rand[sp][:args.n_network] if item == "A7" else rand[sp]
            n_same = int(sum(same_verdict(item, sp, r) for r in pool))
            need = need_of(len(pool))
            per_split[sp] = dict(stated=STATED[item][sp], n_same=n_same, n=len(pool), needed=need, robust=bool(n_same >= need))
            robust = robust and n_same >= need
        items[item] = dict(per_split=per_split, robust=bool(robust))
    quantities = {}                                                    # ranges of every value and difference
    for sp in SPLITS:
        q, P, R = {}, paper[sp], rand[sp]
        for key, f in (("ridge_gene_identity", lambda r: r["gi"]), ("ridge_response_correlation_mean", lambda r: r["dpc_mean"]),
                       ("ridge_response_correlation_median", lambda r: r["dpc_median"]), ("ridge_response_magnitude", lambda r: r["A6"]["mra"]),
                       ("A1_rewired_minus_ridge_gene_identity", lambda r: r["A1"]["diff"]),
                       ("A3_original_minus_ridge_gene_identity", lambda r: r["A3"]["diff"]),
                       ("A4_highest_model_minus_ridge_mean", lambda r: r["A4"]["diff"]),
                       ("A5_highest_model_minus_ridge_median", lambda r: r["A5"]["diff"]),
                       ("A5_cross_attention_minus_ridge_median", lambda r: r["A5"]["attn_cross"]),
                       ("A5_control_minus_ridge_median", lambda r: r["A5"]["mlp_ctrl"]),
                       ("C4_rewired_minus_ridge_response_magnitude", lambda r: r["C4"]["diff"])):
            q[key] = summary([f(r) for r in R], f(P))
        if "A2" in P:
            q["A2_rewired_minus_ridge_gene_identity_without_twin"] = summary([r["A2"]["diff"] for r in R], P["A2"]["diff"])
        for key in ("network", "ridge", "diff"):
            q["A7_" + {"network": "network", "ridge": "ridge_single_precision", "diff": "network_minus_ridge"}[key]] = \
                summary([r["A7"][key] for r in R[:args.n_network]], P["A7"][key])
        quantities[sp] = q

    print("\n" + "=" * 100 + "\nRESULT: assignments that give the verdict of the paper (needed: 95 %)\n" + "=" * 100)
    for item in sorted(items):
        print("  %s  %s" % (item, "ROBUST" if items[item]["robust"] else "NOT ROBUST"))
        for sp, v in items[item]["per_split"].items():
            print("      %-13s %3d of %3d (needed %d)  %-10s stated: %s" % (sp, v["n_same"], v["n"], v["needed"],
                                                                      "robust" if v["robust"] else "NOT robust", v["stated"]))
    print("\nRanges over the random assignments (value of the paper | median, 2.5th to 97.5th percentile, range | at or below the paper)")
    for sp in SPLITS:
        print("  " + sp)
        for key, s in quantities[sp].items():
            print("      %-52s %+.4f | %+.4f, %+.4f to %+.4f, %+.4f to %+.4f | %d of %d"
                  % (key, s["paper"], s["median"], s["p2.5"], s["p97.5"], s["min"], s["max"], s["n_random_not_above_paper"], s["n"]))

    # ---- 7. records ---------------------------------------------------------------------------------------------
    def flat(sp, label, seed, string, r):
        """One row of the per-assignment table."""
        row = dict(split=sp, assignment=label, seed=seed, folds=string, ridge_gene_identity=r["gi"],
                   ridge_response_correlation_mean=r["dpc_mean"], ridge_response_correlation_median=r["dpc_median"],
                   A1_diff=r["A1"]["diff"], A1_lo=r["A1"]["lo"], A1_hi=r["A1"]["hi"],
                   A2_diff=r["A2"]["diff"] if "A2" in r else "", A2_lo=r["A2"]["lo"] if "A2" in r else "", A2_hi=r["A2"]["hi"] if "A2" in r else "",
                   A3_diff=r["A3"]["diff"], A4_diff=r["A4"]["diff"], A5_diff=r["A5"]["diff"],
                   A5_cross_attention=r["A5"]["attn_cross"], A5_control=r["A5"]["mlp_ctrl"],
                   A6_mra=r["A6"]["mra"], A6_lo=r["A6"]["lo"], A6_hi=r["A6"]["hi"],
                   C4_diff=r["C4"]["diff"], C4_lo=r["C4"]["lo"], C4_hi=r["C4"]["hi"],
                   A7_network=r["A7"]["network"] if "A7" in r else "", A7_ridge=r["A7"]["ridge"] if "A7" in r else "",
                   A7_diff=r["A7"]["diff"] if "A7" in r else "")
        for item in sorted(STATED):
            row["same_" + item] = (int(same_verdict(item, sp, r)) if sp in STATED[item] and item in r else "")
        return row

    rows = []
    for sp in SPLITS:
        rows.append(flat(sp, "paper", "", folds[sp]["paper_string"], paper[sp]))
        for k, r in enumerate(rand[sp]):
            rows.append(flat(sp, "random_%03d" % k, SEED0 + k, folds[sp]["strings"][k], r))

    def keep(r):
        """The values of one assignment that go into the JSON record (arrays left out; the ridge's own values named in full)."""
        names = {"gi": "ridge_gene_identity", "dpc_mean": "ridge_response_correlation_mean",
                 "dpc_median": "ridge_response_correlation_median"}
        return {names.get(k, k): v for k, v in r.items() if k not in ("P7", "jac", "dpc", "norm")}

    rec = {
        "analysis": "Step 29 (pre-registered), sci-Plex: fold sensitivity of the comparisons with the cross-validated ridge; "
                    "items A1 to A7 and C4",
        "prereg": PRE, "prereg_sha256": sha256(os.path.join(REPO, PRE)),
        "registered_settings": bool(registered),
        "settings": {
            "random_assignments": int(args.n_random), "seeds": [SEED0, SEED0 + int(args.n_random) - 1],
            "rule_of_the_assignments": fsens["rule"], "network_assignments": int(args.n_network),
            "threshold": THRESHOLD, "robust": "at least 95 % of the random assignments give the verdict of the paper, on every "
                                              "split the item names; the assignment of the paper is not counted among them",
            "ridge": "sklearn Ridge, penalty 1, intercept, features in double precision, target and stored predictions in "
                     "single precision (score_step14.py)",
            "interval": "97.5 %% percentile bootstrap per split, %d replicates, compounds resampled, rng.randint(0, G, G) per "
                        "replicate from a fresh RandomState(0) for every contrast and assignment" % N_BOOT,
            "network": {"width": WIDTH, "weight_decay": WEIGHT_DECAY, "epochs": int(args.epochs), "learning_rate": LR,
                        "minibatch": BATCH, "seeds": list(NET_SEEDS), "fold_numbers": "GroupKFold's own dealing",
                        "processes_for_the_random_assignments": int(workers)},
            "tolerances": {"committed_values": TOL, "network": NET_TOL}, "stated_verdicts": STATED,
            "generative_models": {sp: list(GENERATIVE[sp]) for sp in SPLITS},
        },
        "environment": env,
        "inputs": {"records": {q: sha256(os.path.join(REPO, q)) for q in TRACKED}, "per_split": {sp: data[sp]["inputs"] for sp in SPLITS}},
        "guards": guards,
        "items": items,
        "per_split": {sp: {"n_conditions": int(data[sp]["n"]), "n_compounds": int(len(data[sp]["compounds"])),
                           "excluded_for_A2": models[sp]["excluded"], "n_retained_for_A2": models[sp]["n_retained"],
                           "response_magnitude_contexts": mra_ctx[sp]["n_ctx_used"],
                           "highest_generative_model": {"mean": models[sp]["gen_mean_argmax"], "median": models[sp]["gen_median_argmax"]},
                           "folds_of_the_paper": folds[sp]["paper_string"],
                           "folds_of_the_paper_groupkfold_numbers": folds[sp]["paper_raw_string"], "paper": keep(paper[sp]),
                           "quantities": quantities[sp],
                           "random": [dict(keep(r), seed=SEED0 + k, folds=folds[sp]["strings"][k]) for k, r in enumerate(rand[sp])]}
                      for sp in SPLITS},
        "commit": commit, "branch": sh("git -C '%s' rev-parse --abbrev-ref HEAD" % REPO),
        "script_sha256": sha256(os.path.abspath(__file__)), "seconds": round(time.time() - t0),
        "written": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=1)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print("\n[written] %s  sha256 %s" % (out_json, sha256(out_json)))
    print("[written] %s  sha256 %s  (%d rows)   [%.0f s]" % (out_csv, sha256(out_csv), len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
