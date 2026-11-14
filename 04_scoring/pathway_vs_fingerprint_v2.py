#!/usr/bin/env python3
"""
pathway_vs_fingerprint_v2.py - re-create the record behind Supplementary Note 1 with the folds
written down, and measure how much its numbers depend on the fold assignment.

Purpose
-------
Supplementary Note 1, part d of Supplementary Table S10 and one sentence of Results 3 quote
leakage_results/pathway_vs_fingerprint.json (commit f6676f9, 9 Sep 2026): the response
correlation (DPC-50) of a cross-validated ridge on the three compound splits of sci-Plex, given
the fingerprint, the mechanism-of-action label (pathway_level_2) or the annotated target, each
crossed with the cell line. The script that wrote that record was never committed. The
diagnostic diag_note1_ridge.py (job 19363649, 13 Nov 2026) found the settings that give the
stored numbers back, and found a second thing: the folds of the cross-validated ridge are not
fixed by the code. This script re-creates the analysis with every setting written out and
writes a new record, leakage_results/pathway_vs_fingerprint_v2.json, that supersedes the old one.

What the diagnostic established (and this script builds on)
------------------------------------------------------------
a. Procedure of the old record. Features [A, line, A x line], with A = the dose-scaled
   fingerprint of the Squidiff encoder, or a plain one-hot of the label (not scaled by dose).
   Five folds grouped by compound. Ridge with intercept at the penalties 0.01, 0.1, 1, 10, 100;
   for each split and feature set the largest score is kept ("each feature set at its own best
   alpha" - chosen on the scored conditions themselves, which is optimistic for all three
   feature sets alike). The six label values came back to within 2e-6.
b. The old run worked in single precision. That is harmless for the label models but not for
   the fingerprint model at small penalties, where the single-precision solution depends on
   rounding (at penalty 0.001 it gave 0.159-0.164 where double precision gives 0.188). The
   stored fingerprint values of splits 1 and 2 (best penalty below 1) therefore cannot be
   repeated on another processor to better than about 1e-3. This script works in double
   precision.
c. The folds. sklearn's GroupKFold deals the compounds to the folds in the order of
   np.argsort(group sizes). Nearly every compound has 12 conditions, so the order of ties decides
   the folds, and that order depends on the processor (numpy sorts ties differently with and
   without AVX-512 code) and on the scikit-learn version (newer versions use a stable sort).
   The same code gave a response correlation of the ridge of 0.188, 0.174 or 0.164 on split 1.
   The values of the paper (analyse_step7.py, ridge_vs_mlp_v2.py and the old Note 1 record)
   belong to the assignment that numpy gives WITHOUT AVX-512 code. That assignment is written
   out below (PAPER_FOLDS), so this script no longer depends on the node it runs on.

Logic
-----
1. Data, as analyse_step7.py reads them: Y = true - C (measured mean change of 200 genes per
   held-out condition) from the harness vectors file; SMILES, dose and cell line per condition
   from datasets_corrected/sci_plex_test_<split>.h5ad. The labels are read from the obs table of
   the full sci-Plex object, joined on the condition name (the test files carry no label
   column). Stops if the two files disagree on SMILES, dose or cell line, if a label is not
   constant within a compound, or if the numbers of label classes differ from the old record.
2. Folds: PAPER_FOLDS, one digit per compound in sorted SMILES order. Stops if the group sizes
   are not those of the diagnostic.
   ANCHOR: with these folds, the fingerprint ridge at penalty 1, computed operation for
   operation as analyse_step7.py does, must equal leakage_results/step7_interaction_baseline.json
   to 1e-9 on every split. Otherwise the folds are not the paper's and the script stops.
3. The three models at the five penalties, in double precision (sklearn Ridge, intercept,
   unscaled features). Score: the paper's response correlation, DPC-50 = per condition the
   Pearson correlation between predicted and measured change over the 50 genes with the largest
   |measured change|, then the median over conditions (the mean is stored too). Stored: every
   penalty, the best penalty per split and model (as the old record), and penalty 1 (the fixed
   penalty of the paper's ridge), with the gains of the two label models over the fingerprint.
4. Checks before anything is written: each new value against the old record (tolerance 1e-4 for
   the label models, 1e-3 for the fingerprint, see b) and against the six-decimal values of the
   diagnostic (5e-5). If a check fails, nothing is written.
5. Fold sensitivity (new, exploratory): the same three models under 200 random fold
   assignments. Each one follows GroupKFold's own rule (largest groups first, each to the fold
   that is lightest so far) with the ties ordered at random (numpy RandomState, seeds
   SEED0 .. SEED0 + 199). Stored: every assignment and its fifteen scores per split; summarised:
   median, 2.5th and 97.5th percentile, minimum and maximum of each model and of each gain, the
   number of assignments with a positive gain, and where the paper's assignment falls. Because
   the fingerprint model at penalty 1 is the cross-validated ridge of the paper, this also shows
   the spread of that reference on this measure.
   For speed these 200 runs use one eigen-decomposition per fold for all five penalties (the
   same ridge solution written in its dual form). The script first proves on the paper's folds
   that this gives the same predictions as sklearn's Ridge (to 1e-5) and the same scores (1e-6).

Output
------
leakage_results/pathway_vs_fingerprint_v2.json. Guards: refuses to run if a tracked file is
modified or if the output exists.

Usage (from the repository root, inside the squidiff environment; use sbatch):
    python pathway_vs_fingerprint_v2.py
The options exist for tests on an invented environment; the defaults are the real paths.
"""
import sys
sys.dont_write_bytecode = True          # leave no __pycache__ behind in the worktree

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------------- settings
REPO = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/scratch/ni2071/Diffusion/sciplex/datasets_corrected"
FULL_H5AD = "/scratch/ni2071/Diffusion/sciplex/Sci_Plex.h5ad"
SPLITS = ["drug_split_1", "drug_split_0", "drug_split_2"]            # development split first
VECTORS = {                                                            # carrier of 'true' and 'C'
    "drug_split_1": "leakage_results/vectors_step6_fm_gaussian_seed_0.npz",
    "drug_split_0": "leakage_results/vectors_s6b_drug_split_0_fm_gaussian_seed_0.npz",
    "drug_split_2": "leakage_results/vectors_s6b_drug_split_2_fm_gaussian_seed_0.npz",
}
OLD_RECORD = "leakage_results/pathway_vs_fingerprint.json"            # the record this one supersedes
STEP7_RECORD = "leakage_results/step7_interaction_baseline.json"      # the paper's cross-validated ridge
OUT_DEFAULT = "leakage_results/pathway_vs_fingerprint_v2.json"

KEY = "cov_drug_dose_name"                # condition name in the full object (= 'Group' in the test files)
LABEL_COLUMN = {"pathway": "pathway_level_2", "target": "target"}    # label models and their obs column
MODELS = ["fingerprint", "pathway", "target"]
OLD_KEY = {"fingerprint": "fp_x_line", "pathway": "pathway_x_line", "target": "target_x_line"}
OLD_COUNT = {"pathway": "n_pathways", "target": "n_targets"}
PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)  # the grid of the old record (established by the diagnostic)
NFOLD, TOP = 5, 50

# The fold assignment behind the cross-validated ridge of the paper on sci-Plex: one digit per
# compound, compounds in sorted order of their SMILES strings, digit = fold (0-4). It is what
# sklearn's GroupKFold(5) returns when numpy orders ties WITHOUT AVX-512 code (job 19363649,
# assignment "A2"; the same strings follow from the group sizes alone with numpy 2.5.3).
# The ANCHOR check of step 2 proves on every run that these are the folds of the Step 7 record.
PAPER_FOLDS = {
    "drug_split_1": "0102341023410234413023141023410234120",
    "drug_split_0": "0012340123401234403112340123401234021",
    "drug_split_2": "01023410304102341432123410234010234122",
}
# Conditions per compound in the same order, one hexadecimal digit each (c = 12, b = 11).
GROUP_SIZES = {
    "drug_split_1": "ccccccccccccccccccccccbcccccccccccccc",
    "drug_split_0": "ccccccccccccccccccccccccccccccccccccc",
    "drug_split_2": "ccccccccccccccccccccbccccccccccccccccc",
}

N_RANDOM, SEED0 = 200, 20261113           # fold sensitivity: number of random assignments and first seed
ANCHOR_TOL = 1e-9                         # Step 7 record against its re-computation
SOLVER_TOL, SCORE_TOL = 1e-5, 1e-6        # dual-form solver against sklearn Ridge: predictions, scores
OLD_TOL = {"fingerprint": 1e-3, "pathway": 1e-4, "target": 1e-4}     # new value against the old record
# Values of the diagnostic (job 19363649, double precision, paper's folds, best of the five
# penalties), six decimals as printed. Used as a check only: this script must not drift from what
# the diagnostic did. Never copied into the record.
DIAGNOSTIC = {
    "drug_split_1": {"fingerprint": 0.188414, "pathway": 0.213244, "target": 0.169558},
    "drug_split_0": {"fingerprint": 0.247295, "pathway": 0.264919, "target": 0.263885},
    "drug_split_2": {"fingerprint": 0.082486, "pathway": 0.128675, "target": 0.101170},
}
DIAGNOSTIC_TOL = 5e-5


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


# ----------------------------------------------------------------------------- 1. data
def load_full_obs(path):
    """obs table of the full sci-Plex object, first row of every condition. The expression
    matrix is not read. Returns a table indexed by the condition name."""
    import anndata
    if not os.path.isfile(path):
        die("missing " + path)
    try:
        ad = anndata.read_h5ad(path, backed="r")
        obs = ad.obs.copy()
        try:
            ad.file.close()
        except Exception:
            pass
    except Exception as e:                                  # some anndata versions cannot open every file this way
        print("[data] could not open the full object without its matrix (%s); reading it whole" % type(e).__name__)
        obs = anndata.read_h5ad(path).obs.copy()
    for col in [KEY, "SMILES", "dose", "cell_type"] + list(LABEL_COLUMN.values()):
        if col not in obs.columns:
            die("the full object has no obs column %r" % col)
    first = obs.drop_duplicates(subset=KEY, keep="first").copy()
    first.index = first[KEY].astype(str).values
    return first


def load_split(split, data_dir, first, old):
    """Everything one split needs: target, annotation, labels, compounds. All checks of step 1."""
    import anndata
    vp = os.path.join(REPO, VECTORS[split])
    tp = os.path.join(data_dir, "sci_plex_test_%s.h5ad" % split)
    for p in (vp, tp):
        if not os.path.isfile(p):
            die("missing " + p)
    v = np.load(vp, allow_pickle=True)
    Y = np.asarray(v["true"]) - np.asarray(v["C"])           # as stored (single precision), as analyse_step7.py
    cond = np.array([str(x) for x in v["cond"]])
    obs = anndata.read_h5ad(tp).obs
    g = obs["Group"].astype(str).values
    S = obs["SMILES"].astype(str).values
    Dz = obs["dose"].astype(float).values
    Ln = obs["cell_name"].astype(str).values
    smi, dose, line = [], [], []
    for c in cond:
        m = (g == c)
        if not m.any():
            die("%s: condition %r of the vectors file is absent from the test file" % (split, c))
        if len(set(S[m])) > 1 or len(set(Ln[m])) > 1 or len(set(Dz[m])) > 1:
            die("%s: condition %r has more than one SMILES, dose or cell line in the test file" % (split, c))
        smi.append(S[m][0])
        dose.append(float(np.median(Dz[m])))
        line.append(Ln[m][0])
    smi, dose, line = np.array(smi), np.array(dose), np.array(line)

    # labels from the full object; the two files must agree on what the features are built from
    sub = first.reindex(cond)
    if int(sub[KEY].isna().sum()):
        die("%s: %d conditions have no row in the full object" % (split, int(sub[KEY].isna().sum())))
    if not np.array_equal(sub["SMILES"].astype(str).values, smi):
        die("%s: SMILES differ between the test file and the full object" % split)
    if not np.allclose(np.array([float(x) for x in sub["dose"].astype(str).values]), dose):
        die("%s: doses differ between the test file and the full object" % split)
    if not np.array_equal(sub["cell_type"].astype(str).values, line):
        die("%s: cell lines differ between the test file and the full object" % split)
    labels = {}
    for model, col in LABEL_COLUMN.items():
        lab = sub[col].astype(str).values
        per_compound = {}
        for s, x in zip(smi, lab):
            per_compound.setdefault(s, set()).add(x)
        if any(len(x) > 1 for x in per_compound.values()):
            die("%s: %s is not constant within a compound" % (split, col))
        n_classes = len(set(lab))
        if old is not None and n_classes != old["per_split"][split][OLD_COUNT[model]]:
            die("%s: %d classes of %s, the old record has %d"
                % (split, n_classes, col, old["per_split"][split][OLD_COUNT[model]]))
        labels[model] = lab

    # compounds in sorted SMILES order, and the check that this is the layout the folds were written for
    compounds = sorted(set(smi.tolist()))
    index = {s: k for k, s in enumerate(compounds)}
    comp = np.array([index[s] for s in smi])                 # compound number of every condition
    counts = np.bincount(comp, minlength=len(compounds))
    sizes = "".join("%x" % min(int(c), 15) for c in counts)
    if sizes != GROUP_SIZES[split]:
        die("%s: conditions per compound are %s, the folds were written for %s" % (split, sizes, GROUP_SIZES[split]))
    return {"Y": Y, "cond": cond, "smi": smi, "dose": dose, "line": line, "labels": labels,
            "compounds": compounds, "comp": comp, "counts": counts,
            "inputs": {"vectors": VECTORS[split], "vectors_sha256": sha256(vp), "test_h5ad": tp,
                       "test_h5ad_sha256": sha256(tp), "n_conditions": int(len(cond)),
                       "n_drugs": int(len(compounds)), "lines": sorted(set(line.tolist()))}}


# ----------------------------------------------------------------------------- features
def one_hot(values):
    levels = sorted(set(values))
    return np.array([[float(a == u) for u in levels] for a in values])


def cross(A, L):
    """[A, L, A x L]: the layout of the Step 7/8 scripts (crossing in blocks by cell line)."""
    return np.hstack([A, L, np.concatenate([A * L[:, [k]] for k in range(L.shape[1])], axis=1)])


def build_features(d):
    """The three feature sets of one split, in double precision."""
    from Squidiff.scrna_datasets import Drug_dose_encoder     # the encoder of this worktree
    n = len(d["cond"])
    FP = np.array([Drug_dose_encoder([d["smi"][i]], [d["dose"][i]], comb_num=1)[0] for i in range(n)])  # uncast
    L = one_hot(d["line"].tolist())
    X = {"fingerprint": np.asarray(cross(FP, L), dtype=np.float64)}
    for model in LABEL_COLUMN:
        X[model] = np.asarray(cross(one_hot(d["labels"][model].tolist()), L), dtype=np.float64)
    return X


# ----------------------------------------------------------------------------- 2. folds
def paper_fold(split, d):
    """Fold of every condition under the assignment of the paper."""
    digits = PAPER_FOLDS[split]
    if len(digits) != len(d["compounds"]):
        die("%s: %d compounds, the folds were written for %d" % (split, len(d["compounds"]), len(digits)))
    return np.array([int(digits[k]) for k in d["comp"]])


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
    has one spelling (the spelling of PAPER_FOLDS)."""
    remap, out = {}, []
    for f in fold_of_group:
        if int(f) not in remap:
            remap[int(f)] = len(remap)
        out.append(str(remap[int(f)]))
    return "".join(out)


def random_fold_of_group(counts, seed):
    """One random assignment: the groups in random order, then largest first (stable, so that ties
    keep the random order), then GroupKFold's rule. RandomState streams do not change between
    numpy versions, so the 200 assignments can be regenerated anywhere."""
    perm = np.random.RandomState(seed).permutation(len(counts))
    order = perm[np.argsort(-counts[perm], kind="stable")]
    return deal(counts, order)


def sklearn_fold_string(d):
    """What sklearn's GroupKFold gives on THIS node, in the canonical spelling (information only)."""
    from sklearn.model_selection import GroupKFold
    fold = np.full(len(d["smi"]), -1, dtype=int)
    for f, (_, te) in enumerate(GroupKFold(n_splits=NFOLD).split(np.zeros((len(fold), 1)), None, groups=d["smi"])):
        fold[te] = f
    fold_of_group = [int(fold[np.where(d["comp"] == k)[0][0]]) for k in range(len(d["compounds"]))]
    return canonical(fold_of_group)


# ----------------------------------------------------------------------------- ridge and score
def step7_ridge(X, Y, fold):
    """The fingerprint ridge at penalty 1, operation for operation as analyse_step7.py: features
    in double precision, target and stored predictions in single precision, scipy's Pearson,
    conditions without variance left out, median. Used for the ANCHOR only."""
    from scipy.stats import pearsonr
    from sklearn.linear_model import Ridge
    P = np.zeros_like(Y)
    for f in range(NFOLD):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        P[te] = Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])
    r = []
    for i in range(len(Y)):
        idx = np.argsort(np.abs(Y[i]))[-TOP:]
        if Y[i][idx].std() > 1e-12 and P[i][idx].std() > 1e-12:
            r.append(pearsonr(P[i][idx], Y[i][idx])[0])
    return float(np.median(r))


def cv_sklearn(X, Y, fold, penalty):
    """Cross-validated predictions with sklearn's Ridge (intercept, unscaled features)."""
    from sklearn.linear_model import Ridge
    P = np.zeros_like(Y)
    for f in range(NFOLD):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        P[te] = Ridge(alpha=penalty).fit(X[tr], Y[tr]).predict(X[te])
    return P


def cv_dual(X, Y, fold):
    """The same cross-validated ridge for all penalties at once. With the training rows centred
    (Xc, Yc), the ridge prediction for new rows x is (x - mean) Xc' (K + penalty I)^-1 Yc + mean_y,
    K = Xc Xc'. One eigen-decomposition K = V diag(w) V' serves every penalty:
    (K + penalty I)^-1 = V diag(1 / (w + penalty)) V'. Returns {penalty: predictions}."""
    P = {a: np.zeros_like(Y) for a in PENALTIES}
    for f in range(NFOLD):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        Xc, Yc = X[tr] - mx, Y[tr] - my
        w, V = np.linalg.eigh(Xc @ Xc.T)
        w = np.clip(w, 0.0, None)                           # rounding can leave tiny negative eigenvalues
        VtY = V.T @ Yc
        KV = ((X[te] - mx) @ Xc.T) @ V
        for a in PENALTIES:
            P[a][te] = KV @ (VtY / (w[:, None] + a)) + my
    return P


def top_genes(Y):
    """Per condition the 50 genes with the largest |measured change|. A stable sort, so that a tie
    at the 50th place is settled by gene order on every machine."""
    return np.argsort(np.abs(Y), axis=1, kind="stable")[:, -TOP:]


def dpc50(P, Y, idx):
    """The paper's response correlation: per condition the Pearson correlation of predicted and
    measured change over its 50 genes; a condition without variance counts as 0 (as dpc50 of
    ridge_vs_mlp_v2.py). Returns (median, mean, number of conditions without variance)."""
    a = np.take_along_axis(P, idx, axis=1)
    b = np.take_along_axis(Y, idx, axis=1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    na, nb = np.sqrt((a * a).sum(1)), np.sqrt((b * b).sum(1))
    ok = (na > 0) & (nb > 0)
    r = np.zeros(len(Y))
    r[ok] = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
    return float(np.median(r)), float(np.mean(r)), int((~ok).sum())


def summary(values, paper=None, count_positive=False):
    """Median, percentiles and range of one quantity over the random assignments."""
    v = np.asarray(values, dtype=np.float64)
    out = {"median": float(np.median(v)), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)),
           "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    if count_positive:
        out["n_positive"] = int((v > 0).sum())
    if paper is not None:
        out["paper"] = float(paper)
        out["n_random_not_above_paper"] = int((v <= paper).sum())   # where the paper's assignment falls
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--full-h5ad", default=FULL_H5AD)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--diagnostic-json", default="",
                    help="test only: a JSON file that replaces the built-in DIAGNOSTIC values")
    args = ap.parse_args()
    t0 = time.time()
    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)

    # ---- guards ----------------------------------------------------------------------------
    if os.path.exists(out_path):
        die("%s exists; refusing to overwrite a record" % out_path)
    dirty = sh("git -C '%s' status --porcelain --untracked-files=no" % REPO)
    if dirty:
        die("tracked files are modified; commit first.\n" + dirty)
    diagnostic = json.load(open(args.diagnostic_json)) if args.diagnostic_json else DIAGNOSTIC
    old_path, s7_path = os.path.join(REPO, OLD_RECORD), os.path.join(REPO, STEP7_RECORD)
    for p in (old_path, s7_path):
        if not os.path.isfile(p):
            die("missing " + p)
    old, step7 = json.load(open(old_path)), json.load(open(s7_path))

    flags = " " + sh("grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2") + " "
    env = {"node": platform.node(), "cpu": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
           "cpu_has_avx512f": " avx512f " in flags, "python": platform.python_version(), "numpy": np.__version__,
           "scipy": version_of("scipy"), "sklearn": version_of("scikit-learn"), "anndata": version_of("anndata"),
           "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
    print("[env] %s" % env)

    # ---- 1. data ----------------------------------------------------------------------------
    first = load_full_obs(args.full_h5ad)
    data, feats, rec_inputs = {}, {}, {}
    for sp in SPLITS:
        d = load_split(sp, args.data_dir, first, old)
        X = build_features(d)
        if X["fingerprint"].shape[1] != 4099:
            die("%s: %d fingerprint features, expected 4,099" % (sp, X["fingerprint"].shape[1]))
        data[sp], feats[sp] = d, X
        rec_inputs[sp] = dict(d["inputs"], n_features={m: int(X[m].shape[1]) for m in MODELS},
                              n_classes={m: int(len(set(d["labels"][m].tolist()))) for m in LABEL_COLUMN})
        print("[data] %s: %d conditions, %d compounds; features %s; classes %s"
              % (sp, len(d["cond"]), len(d["compounds"]), rec_inputs[sp]["n_features"], rec_inputs[sp]["n_classes"]))
    rec_inputs["full_object"] = {"path": args.full_h5ad, "sha256": sha256(args.full_h5ad),
                                 "bytes": os.path.getsize(args.full_h5ad), "join": KEY,
                                 "label_columns": LABEL_COLUMN}
    print("[data] the test files and the full object agree on SMILES, dose and cell line; labels constant per compound")

    # ---- 2. folds and the anchor --------------------------------------------------------------
    folds, anchor, rec_folds = {}, {}, {}
    for sp in SPLITS:
        d = data[sp]
        folds[sp] = paper_fold(sp, d)
        here = sklearn_fold_string(d)
        want = float(step7["per_split"][sp]["marginal"]["interaction"])
        got = step7_ridge(feats[sp]["fingerprint"], d["Y"], folds[sp])
        anchor[sp] = {"step7_record": want, "recomputed": got, "abs_diff": abs(got - want)}
        rec_folds[sp] = {"compounds_sorted_smiles": d["compounds"], "fold_of_compound": [int(c) for c in PAPER_FOLDS[sp]],
                         "conditions_per_compound": [int(c) for c in d["counts"]],
                         "conditions_per_fold": [int(np.sum(folds[sp] == f)) for f in range(NFOLD)],
                         "groupkfold_on_this_node": here, "groupkfold_on_this_node_is_paper": here == PAPER_FOLDS[sp]}
        print("[anchor] %s: Step 7 record %.10f, recomputed %.10f, difference %.1e | GroupKFold on this node gives "
              "the paper's folds: %s" % (sp, want, got, abs(got - want), here == PAPER_FOLDS[sp]))
        if abs(got - want) > ANCHOR_TOL:
            die("%s: the written folds do not give the Step 7 ridge back (difference %.2e); nothing written"
                % (sp, abs(got - want)))

    # ---- 3. the three models on the paper's folds ------------------------------------------------
    per_split, scores, worst_pred, worst_score = {}, {}, 0.0, 0.0
    Y64 = {sp: data[sp]["Y"].astype(np.float64) for sp in SPLITS}
    top = {sp: top_genes(Y64[sp]) for sp in SPLITS}
    for sp in SPLITS:
        d, scores[sp] = data[sp], {}
        block = {"per_penalty": {}, "best": {}, "at_penalty_1": {}, "gain_over_fingerprint": {}}
        for m in MODELS:
            dual = cv_dual(feats[sp][m], Y64[sp], folds[sp])  # for the proof that the fast form is the same ridge
            block["per_penalty"][m], scores[sp][m] = {}, {}
            for a in PENALTIES:
                P = cv_sklearn(feats[sp][m], Y64[sp], folds[sp], a)
                med, mean, undefined = dpc50(P, Y64[sp], top[sp])
                block["per_penalty"][m]["%g" % a] = {"median": med, "mean": mean, "n_without_variance": undefined}
                scores[sp][m][a] = med
                worst_pred = max(worst_pred, float(np.abs(P - dual[a]).max()))
                worst_score = max(worst_score, abs(med - dpc50(dual[a], Y64[sp], top[sp])[0]))
            best = max(PENALTIES, key=lambda a: scores[sp][m][a])
            block["best"][m] = {"penalty": best, "median": scores[sp][m][best]}
            block["at_penalty_1"][m] = scores[sp][m][1.0]
            block[OLD_KEY[m]] = scores[sp][m][best]            # the key of the old record, same meaning
        fp_best, fp_one = block["best"]["fingerprint"]["median"], block["at_penalty_1"]["fingerprint"]
        block["gain_over_fingerprint"] = {                    # label model minus fingerprint model
            "best_penalty": {m: block["best"][m]["median"] - fp_best for m in LABEL_COLUMN},
            "penalty_1": {m: block["at_penalty_1"][m] - fp_one for m in LABEL_COLUMN}}
        block.update({"n": int(len(d["cond"])), "n_drugs": int(len(d["compounds"])),
                      "n_pathways": rec_inputs[sp]["n_classes"]["pathway"],
                      "n_targets": rec_inputs[sp]["n_classes"]["target"],
                      "pathway_beats_fp": bool(block["gain_over_fingerprint"]["best_penalty"]["pathway"] > 0)})
        per_split[sp] = block
        print("[paper folds] %s   penalties %s" % (sp, " ".join("%g" % a for a in PENALTIES)))
        for m in MODELS:
            print("    %-11s %s   best %.6f at %g" % (m, " ".join("%.6f" % scores[sp][m][a] for a in PENALTIES),
                                                      block["best"][m]["median"], block["best"][m]["penalty"]))
    print("[solver] dual form against sklearn Ridge on the paper's folds: largest difference of a prediction %.1e, "
          "of a score %.1e" % (worst_pred, worst_score))
    if worst_pred > SOLVER_TOL or worst_score > SCORE_TOL:
        die("the dual-form solver does not reproduce sklearn's Ridge; nothing written")

    # ---- 4. checks against the old record and the diagnostic ---------------------------------------
    comparison, failed = {}, []
    for sp in SPLITS:
        comparison[sp] = {}
        for m in MODELS:
            new, was = per_split[sp][OLD_KEY[m]], float(old["per_split"][sp][OLD_KEY[m]])
            comparison[sp][OLD_KEY[m]] = {"old": was, "new": new, "new_minus_old": new - was, "tolerance": OLD_TOL[m]}
            flag = ""
            if abs(new - was) > OLD_TOL[m]:
                failed.append("%s %s: new %.6f, old %.6f" % (sp, m, new, was))
                flag = "   <-- beyond the tolerance"
            if abs(new - float(diagnostic[sp][m])) > DIAGNOSTIC_TOL:
                failed.append("%s %s: new %.6f, diagnostic %.6f" % (sp, m, new, float(diagnostic[sp][m])))
                flag += "   <-- differs from the diagnostic"
            print("[old record] %s %-11s old %.6f  new %.6f  difference %+.6f%s" % (sp, m, was, new, new - was, flag))
    if failed:
        die("the re-creation does not agree with the stored values; nothing written:\n  " + "\n  ".join(failed))

    # ---- 5. fold sensitivity: 200 random assignments ------------------------------------------------
    sens = {"n_assignments": int(args.n_random), "seeds": [SEED0, SEED0 + int(args.n_random) - 1],
            "rule": "compounds in random order (numpy RandomState(seed).permutation), largest first (stable), each to the "
                    "fold that is lightest so far; five folds", "penalties": list(PENALTIES), "per_split": {}}
    for sp in SPLITS:
        d = data[sp]
        strings, raw = [], {m: [] for m in MODELS}
        for r in range(int(args.n_random)):
            fold_of_group = random_fold_of_group(d["counts"], SEED0 + r)
            strings.append(canonical(fold_of_group))
            fold = fold_of_group[d["comp"]]
            for m in MODELS:
                dual = cv_dual(feats[sp][m], Y64[sp], fold)
                raw[m].append([dpc50(dual[a], Y64[sp], top[sp])[0] for a in PENALTIES])
        arr = {m: np.array(raw[m]) for m in MODELS}           # assignments x penalties
        one = PENALTIES.index(1.0)
        best = {m: arr[m].max(1) for m in MODELS}
        fixed = {m: arr[m][:, one] for m in MODELS}
        blk = per_split[sp]
        paper = {"best_penalty": {m: blk["best"][m]["median"] for m in MODELS}, "penalty_1": blk["at_penalty_1"]}
        summ = {}
        for tag, val in (("best_penalty", best), ("penalty_1", fixed)):
            summ[tag] = {"models": {m: summary(val[m], paper[tag][m]) for m in MODELS},
                         "gain_over_fingerprint": {
                             m: summary(val[m] - val["fingerprint"], blk["gain_over_fingerprint"][tag][m],
                                        count_positive=True) for m in LABEL_COLUMN}}
        sens["per_split"][sp] = {"assignments": strings, "n_distinct": int(len(set(strings))),
                                 "paper_assignment_among_them": PAPER_FOLDS[sp] in strings,
                                 "median_dpc50": {m: arr[m].tolist() for m in MODELS}, "summary": summ}
        print("[spread] %s: %d random fold assignments (%d distinct)   [%.0f s]"
              % (sp, int(args.n_random), len(set(strings)), time.time() - t0))
        for tag, title in (("best_penalty", "best penalty"), ("penalty_1", "penalty 1")):
            for m in MODELS:
                s = summ[tag]["models"][m]
                print("    %-12s %-11s paper %.4f | random: median %.4f, 2.5%% %.4f, 97.5%% %.4f, range %.4f to %.4f; "
                      "at or below the paper's value: %d of %d"
                      % (title, m, s["paper"], s["median"], s["p2.5"], s["p97.5"], s["min"], s["max"],
                         s["n_random_not_above_paper"], int(args.n_random)))
            for m in LABEL_COLUMN:
                s = summ[tag]["gain_over_fingerprint"][m]
                print("    %-12s gain %-7s paper %+.4f | random: median %+.4f, 2.5%% %+.4f, 97.5%% %+.4f, "
                      "positive in %d of %d" % (title, m, s["paper"], s["median"], s["p2.5"], s["p97.5"],
                                                s["n_positive"], int(args.n_random)))

    # ---- record ---------------------------------------------------------------------------------------
    rec = {
        "analysis": "EXPLORATORY, not pre-registered; re-creation of leakage_results/pathway_vs_fingerprint.json "
                    "(commit f6676f9), whose script was never committed; this record supersedes it. The fold "
                    "sensitivity is new and exploratory.",
        "settings": {
            "features": "[A, line one-hot, A x line]; A = dose-scaled 1,024-bit fingerprint (Squidiff Drug_dose_encoder), "
                        "or a plain one-hot of pathway_level_2, or a plain one-hot of target (labels not scaled by dose)",
            "ridge": "sklearn Ridge, intercept, unscaled features, double precision",
            "penalties": list(PENALTIES),
            "best_penalty": "largest condition median per split and model, chosen on the scored conditions "
                            "(as the old record)",
            "folds": "five folds grouped by compound; the assignment of the paper, written out in PAPER_FOLDS and stored "
                     "under 'folds'; it is what sklearn GroupKFold returns when numpy orders ties without AVX-512 code",
            "score": "DPC-50: Pearson of predicted and measured change over the 50 genes with the largest |measured change| "
                     "(stable sort), per condition; condition median (mean also stored); no variance counts as 0",
            "tolerances": {"anchor": ANCHOR_TOL, "old_record": OLD_TOL, "diagnostic": DIAGNOSTIC_TOL,
                           "dual_solver_prediction": SOLVER_TOL, "dual_solver_score": SCORE_TOL},
        },
        "environment": env,
        "inputs": rec_inputs,
        "folds": rec_folds,
        "anchor_step7_ridge_penalty_1": anchor,
        "per_split": per_split,
        "old_record": {"path": OLD_RECORD, "sha256": sha256(old_path), "comparison": comparison,
                       "note": "the old run used single precision; its fingerprint values at penalties below 1 depend on "
                               "rounding and are not repeatable to better than about 1e-3 on another processor"},
        "dual_solver_check": {"largest_prediction_difference": worst_pred, "largest_score_difference": worst_score},
        "fold_sensitivity": sens,
        "commit": sh("git -C '%s' rev-parse HEAD" % REPO), "branch": sh("git -C '%s' rev-parse --abbrev-ref HEAD" % REPO),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "written": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1)
    print("[written] %s  sha256 %s  (%.0f s)" % (out_path, sha256(out_path), time.time() - t0))


if __name__ == "__main__":
    main()
