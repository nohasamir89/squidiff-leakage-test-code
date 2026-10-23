#!/usr/bin/env python3
"""
ridge_vs_mlp_v2.py — re-create the record behind the feedforward sentence of Results 3.

Purpose
-------
The committed record leakage_results/ridge_vs_mlp.json (commit 82c19c3, 3 Sep 2026)
holds the numbers quoted in Results 3 ("a feedforward network given exactly the
features of the linear references ... reached a response correlation of 0.04–0.06
against 0.08–0.25 for the ridge"), but the script that produced it was never
committed. This script re-creates that experiment from the record's own
description and from the code the other Step 7/8 scripts share, and writes a new
record, leakage_results/ridge_vs_mlp_v2.json, that supersedes the old one. The
manuscript must quote the new record.

Logic (what the old record says it did, and what this script does)
------------------------------------------------------------------
Per compound split (drug_split_1 = development, drug_split_0 and 2 = confirmation):

1. Target.  One row per held-out condition: the measured mean expression change
   Y = true − C (200 genes), read from the harness vectors file of that split
   ('true' = condition mean of the measured treated cells, 'C' = mean of the
   matched vehicle cells; both are model-independent, so the fm_gaussian seed-0
   file is used only as the carrier, exactly as step8_convergence.py does).

2. Features.  Exactly the features of the linear references (Methods,
   "Experimental references"): the dose-scaled 1,024-bit fingerprint from the
   Squidiff encoder (Drug_dose_encoder), a one-hot cell-line indicator, and their
   products — 1,024 + 3 + 3 × 1,024 = 4,099 columns on sci-Plex.

3. Folds.  Five-fold cross-validation grouped by compound (sklearn GroupKFold on
   SMILES, deterministic), as in analyse_step7.py and step8_*.py, so every
   condition receives a prediction from a model that never saw its compound.

4. Ridge.  sklearn Ridge(alpha = 1.0), intercept, unscaled features (the
   cross-validated ridge of the paper). Its score must reproduce the Step 7
   record (0.1884 / 0.2473 / 0.0816 condition medians on splits 1 / 0 / 2) to
   within 0.005, otherwise the features or folds are wrong and the script stops.

5. Feedforward network.  One hidden layer (ReLU) of width 64, 256 or 1,024,
   linear output of 200 genes, trained by AdamW on the standardised features of
   the training folds (mean and SD of the training fold; constant columns
   dropped), mean squared error, minibatches of 32, fixed number of epochs,
   no early stopping. Three seeds per width. Regularisation setting chosen as the
   old record describes: twelve settings (weight decay 0.01, 0.1, 1, 10 ×
   25, 100, 400 epochs) swept on drug_split_1 only, at width 1,024 and seed 0,
   and the best carried to every split and width.
   Unknowns of the old run, fixed here and recorded: learning rate 1e-3,
   minibatch 32, width and seed of the sweep. Because of them the new values
   need not equal the old ones to the third decimal; the new record is the
   one to quote.

6. Score.  The paper's response correlation (DPC-50): per condition, Pearson
   correlation between predicted and measured change over the 50 genes with the
   largest |measured change| (argsort(|Y|)[-50:], as analyse_step7.py), then the
   median over conditions (the mean is also recorded).

Outputs
-------
leakage_results/ridge_vs_mlp_v2.json — settings, the sweep, every width × seed
value per split, ridge per split, mlp_best per split (max over widths of the
seed mean), the commit, input hashes and library versions.

Usage
-----
    python ridge_vs_mlp_v2.py                 # real run, from the repo root
    python ridge_vs_mlp_v2.py --mock --out /tmp/mock.json   # structure test, no data read

Guards: refuses to run if any tracked file is modified, or if the output exists.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------------- settings
DATA_DIR = "/scratch/ni2071/Diffusion/sciplex/datasets_corrected"
SPLITS = ["drug_split_1", "drug_split_0", "drug_split_2"]          # development first
VECTORS = {                                                          # carrier of 'true' and 'C'
    "drug_split_1": "leakage_results/vectors_step6_fm_gaussian_seed_0.npz",
    "drug_split_0": "leakage_results/vectors_s6b_drug_split_0_fm_gaussian_seed_0.npz",
    "drug_split_2": "leakage_results/vectors_s6b_drug_split_2_fm_gaussian_seed_0.npz",
}
STEP7_RIDGE = {"drug_split_1": 0.1884, "drug_split_0": 0.2473, "drug_split_2": 0.0816}
RIDGE_TOL = 0.005
WIDTHS = [64, 256, 1024]
SEEDS = [0, 1, 2]
SWEEP_WD = [0.01, 0.1, 1.0, 10.0]
SWEEP_EPOCHS = [25, 100, 400]
SWEEP_WIDTH, SWEEP_SEED = 1024, 0
LR, BATCH, TOP = 1e-3, 32, 50
OUT_DEFAULT = "leakage_results/ridge_vs_mlp_v2.json"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- data
def load_split(split):
    """Features X (n_cond × 4,099), target Y (n_cond × 200), compound labels (for the folds).
    Mirrors step8_convergence.py line for line, except that it is a function of the split."""
    import scanpy as sc
    from Squidiff.scrna_datasets import Drug_dose_encoder

    v = np.load(VECTORS[split], allow_pickle=True)
    Y = (np.asarray(v["true"]) - np.asarray(v["C"])).astype(np.float32)
    cond = np.array([str(x) for x in v["cond"]])
    ad = sc.read_h5ad(f"{DATA_DIR}/sci_plex_test_{split}.h5ad")
    g = ad.obs["Group"].astype(str).values
    smi = {c: ad.obs["SMILES"].astype(str).values[g == c][0] for c in cond}
    dose = {c: float(np.median(ad.obs["dose"].astype(float).values[g == c])) for c in cond}
    line = {c: ad.obs["cell_name"].astype(str).values[g == c][0] for c in cond}
    drugs = np.array([smi[c] for c in cond])
    FP = np.array([Drug_dose_encoder([smi[c]], [dose[c]], comb_num=1)[0] for c in cond], dtype=np.float32)
    lv = sorted(set(line.values()))
    L = np.array([[float(line[c] == u) for u in lv] for c in cond], dtype=np.float32)
    X = np.hstack([FP, L] + [FP * L[:, [k]] for k in range(len(lv))]).astype(np.float32)
    return X, Y, drugs, {"n_conditions": int(len(cond)), "n_drugs": int(len(set(drugs))),
                         "n_features": int(X.shape[1]), "lines": lv}


def mock_split(split, rng):
    """Invented arrays with the real shapes, for a structure test only. Never a result."""
    n, n_drugs = {"drug_split_1": 443, "drug_split_0": 444, "drug_split_2": 455}[split], 37
    drugs = np.array([f"D{i % n_drugs}" for i in range(n)])
    FP = (rng.rand(n, 1024) < 0.05).astype(np.float32)
    L = np.eye(3, dtype=np.float32)[rng.randint(0, 3, n)]
    X = np.hstack([FP, L] + [FP * L[:, [k]] for k in range(3)]).astype(np.float32)
    W = rng.randn(X.shape[1], 200).astype(np.float32) * 0.05
    Y = (X @ W + rng.randn(n, 200).astype(np.float32)).astype(np.float32)
    return X, Y, drugs, {"n_conditions": n, "n_drugs": n_drugs, "n_features": int(X.shape[1]), "lines": ["A", "B", "C"]}


# ----------------------------------------------------------------------------- scoring
def dpc50(pred, Y):
    """Per condition: Pearson over the 50 genes with the largest |measured change|.
    Returns (median, mean) over conditions — the paper's response correlation."""
    r = []
    for i in range(Y.shape[0]):
        idx = np.argsort(np.abs(Y[i]))[-TOP:]
        a, b = pred[i, idx].astype(np.float64), Y[i, idx].astype(np.float64)
        if a.std() == 0 or b.std() == 0:
            r.append(0.0)
        else:
            r.append(float(np.corrcoef(a, b)[0, 1]))
    r = np.array(r)
    return float(np.median(r)), float(np.mean(r))


def folds(X, Y, drugs):
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=5).split(X, Y, groups=drugs))


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


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--mock", action="store_true", help="structure test on invented arrays; writes only to --out")
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"FATAL: {args.out} exists; refusing to overwrite a record")
    if not args.mock:
        dirty = sh("git status --porcelain --untracked-files=no")
        if dirty:
            sys.exit("FATAL: tracked files are modified; commit first.\n" + dirty)
    commit = sh("git rev-parse HEAD")

    import torch as th
    th.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    rng = np.random.RandomState(0)

    rec = {
        "analysis": "EXPLORATORY, not pre-registered; re-creation of leakage_results/ridge_vs_mlp.json "
                    "(commit 82c19c3), whose script was never committed; this record supersedes it",
        "settings": {"features": "[FP(1024, dose-scaled), line one-hot, FP x line] = 4,099", "folds": "GroupKFold(5) by SMILES",
                     "ridge": "sklearn Ridge(alpha=1.0), intercept, unscaled",
                     "mlp": "one hidden layer, ReLU, AdamW, MSE, standardised inputs, minibatch %d, lr %g" % (BATCH, LR),
                     "widths": WIDTHS, "seeds": SEEDS, "sweep": {"split": "drug_split_1", "width": SWEEP_WIDTH, "seed": SWEEP_SEED,
                                                                  "weight_decay": SWEEP_WD, "epochs": SWEEP_EPOCHS},
                     "score": "DPC-50: Pearson on argsort(|Y|)[-50:] per condition; median (and mean) over conditions",
                     "mock": bool(args.mock)},
        "sweep": {}, "per_split": {}, "inputs": {}, "commit": commit, "torch": th.__version__,
        "written": datetime.now(timezone.utc).isoformat(),
    }

    data = {}
    for split in SPLITS:
        X, Y, drugs, meta = (mock_split(split, rng) if args.mock else load_split(split))
        if meta["n_features"] != 4099:
            sys.exit(f"FATAL: {split}: {meta['n_features']} features, expected 4,099")
        data[split] = (X, Y, drugs, folds(X, Y, drugs), meta)
        if not args.mock:
            rec["inputs"][split] = {"vectors": VECTORS[split], "vectors_sha256": sha256(VECTORS[split]),
                                    "test_h5ad": f"{DATA_DIR}/sci_plex_test_{split}.h5ad", **meta}
        print(f"[data] {split}: {meta['n_conditions']} conditions, {meta['n_drugs']} compounds, {meta['n_features']} features")

    # ---- ridge, and the reproduction check against the Step 7 record ----------
    ridge = {}
    for split in SPLITS:
        X, Y, drugs, fl, _ = data[split]
        med, mean = dpc50(ridge_cv(X, Y, fl), Y)
        ridge[split] = {"median": med, "mean": mean}
        print(f"[ridge] {split}: DPC-50 median {med:.4f} (mean {mean:.4f}); Step 7 record {STEP7_RIDGE[split]:.4f}")
        if not args.mock and abs(med - STEP7_RIDGE[split]) > RIDGE_TOL:
            sys.exit(f"FATAL: ridge on {split} differs from the Step 7 record by more than {RIDGE_TOL}: "
                     f"features or folds are not the paper's")

    # ---- sweep on the development split --------------------------------------
    X, Y, drugs, fl, _ = data["drug_split_1"]
    best, best_key = -np.inf, None
    for wd in SWEEP_WD:
        for ep in SWEEP_EPOCHS:
            med, mean = dpc50(mlp_cv(X, Y, fl, SWEEP_WIDTH, wd, ep, SWEEP_SEED), Y)
            key = f"wd{wd:g}_ep{ep}"
            rec["sweep"][key] = {"median": med, "mean": mean}
            print(f"[sweep] drug_split_1 width {SWEEP_WIDTH} seed {SWEEP_SEED} {key}: median {med:.4f}")
            if med > best:
                best, best_key = med, (wd, ep)
    wd, ep = best_key
    rec["chosen"] = {"weight_decay": wd, "epochs": ep, "chosen_on": "drug_split_1 (development split)"}
    print(f"[chosen] weight decay {wd:g}, {ep} epochs")

    # ---- every width x seed on every split, at the chosen setting ---------------
    for split in SPLITS:
        X, Y, drugs, fl, meta = data[split]
        per_width = {}
        for w in WIDTHS:
            vals = []
            for s in SEEDS:
                med, mean = dpc50(mlp_cv(X, Y, fl, w, wd, ep, s), Y)
                vals.append({"seed": s, "median": med, "mean": mean})
                print(f"[mlp] {split} width {w} seed {s}: median {med:.4f}")
            per_width[str(w)] = {"seeds": vals, "median_mean_over_seeds": float(np.mean([v["median"] for v in vals]))}
        mlp_best = max(v["median_mean_over_seeds"] for v in per_width.values())
        rec["per_split"][split] = {"n": meta["n_conditions"], "n_drugs": meta["n_drugs"], "ridge": ridge[split],
                                   "mlp": per_width, "mlp_best": mlp_best,
                                   "ratio_ridge_over_mlp": ridge[split]["median"] / mlp_best if mlp_best > 0 else None}
        print(f"[result] {split}: ridge {ridge[split]['median']:.4f}  best feedforward {mlp_best:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=2)
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
