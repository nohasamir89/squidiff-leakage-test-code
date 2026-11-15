#!/usr/bin/env python3
"""Step 29, guard 4: which fold assignment did the cross-validated ridge of the paper use on Tahoe-100M?

Purpose
-------
The cross-validated ridge is fitted within the held-out compounds, with five folds grouped by chemical
structure (scikit-learn's GroupKFold). GroupKFold deals the structures to the folds in the order of their
number of conditions, largest first. Structures with the same number of conditions are ordered by numpy's
sort, and that order differs between processors with and without AVX-512 instructions. So the same code can
give different folds on different compute nodes (PREREG_step29_fold_sensitivity.md, section 1).

On sci-Plex the folds of the paper are stored in a record. On Tahoe-100M they are not. Before the Tahoe part
of Step 29 can refit the ridge "with the folds of the paper", those folds must be identified. This script
does only that. It reads; it changes no record of the paper and makes no statement about the paper.

Logic
-----
For each compound split (drug_split_1, drug_split_0, drug_split_2):
 1. Read the committed per-condition record of the paper (tahoe_results/step15_model_per_condition.csv): the
    conditions in their order, the cell line, the structure (SMILES), the dose, and the two values of the
    cross-validated ridge that depend on the folds (gene identity, column ridge_S7_de_overlap; response
    correlation, column ridge_S7_dpc50).
 2. Read the measured change of every condition (mean of the measured cells minus mean of the paired vehicle
    cells) from the stored evaluation vectors of the rewired model, seed 0, the file score_step15_tahoe.py
    took them from. The file is checked against the sha256 in its provenance record, and its conditions must
    be those of the per-condition record, in the same order. (The vectors are not in git; the file is taken
    from this worktree, from S29_VECTORS_DIR, or from the main checkout, and the hash decides.)
 3. Build the features exactly as score_step15_tahoe.py: fingerprint and dose code (the encoder of the
    Squidiff package), cell line, and fingerprint x cell line.
 4. Build the candidate orders of the structures. All use GroupKFold's own rule (each structure goes to the
    fold that is lightest so far); they differ only in the order of structures with equal counts:
      a. "default"        numpy's default sort as it runs in this process;
      b. "stable"         numpy's stable sort;
      c. "without_avx512" numpy's default sort with its AVX-512 code switched off. Only made on a node whose
                          numpy uses AVX-512 (there "default" is the order with AVX-512); it is computed in a
                          child process started with NPY_DISABLE_CPU_FEATURES.
    On a node without AVX-512, "default" is the order without AVX-512, and the order with AVX-512 cannot be
    made; the output says so. scikit-learn's GroupKFold is also run directly, and must give the same
    partition as candidate "default" (a check of this script's copy of the rule).
 5. For each distinct partition: refit the ridge as score_step15_tahoe.py does (penalty 1, five folds, the
    held-out folds predicted in turn), compute gene identity and response correlation per condition with
    the function of the paper, and compare them with the two columns of the record.
 6. A partition REPRODUCES the record if both columns agree to 1e-9 on every condition (missing values must
    be missing on both sides). The mean and median response correlation are also compared with
    step15_model_analysis.json.

Outcome (PREREG section 7, guard 4)
-----------------------------------
 - Every split has a reproducing partition: tahoe_results/step29_tahoe_folds.json is written, with the fold
   of every structure, the candidates that gave it, and all differences. Written once, never overwritten.
 - Some split has none: nothing is written and the script stops. The pre-registration is then amended
   before anything else is read. (If this happens on a node without AVX-512, the first thing to do is to run
   the script on a node with AVX-512, which adds the third candidate.)

Guards: the output must not exist; no tracked file may be modified; the pre-registration, this script and
the two records it reads must be committed.

Run:  sbatch run_step29_tahoe_folds.sh      (CPU only; needs scikit-learn and the Squidiff package, so it is
                                              a job, not a login-node command)
"""
import os
import sys
sys.dont_write_bytecode = True          # leave no __pycache__ behind in the worktree

import hashlib
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)                                             # the Squidiff package of this worktree
os.chdir(REPO)

RES = "tahoe_results"
PREREG = "PREREG_step29_fold_sensitivity.md"
REC_PC = os.path.join(RES, "step15_model_per_condition.csv")        # per-condition record of the paper
REC_AN = os.path.join(RES, "step15_model_analysis.json")            # its summary (mean and median of the ridge)
OUT = os.path.join(RES, "step29_tahoe_folds.json")
SPLITS = ("drug_split_1", "drug_split_0", "drug_split_2")
NFOLD, ALPHA, K, TOL = 5, 1.0, 50, 1e-9                              # as score_step15_tahoe.py; TOL as PREREG guard 4
T0 = time.time()


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


# ----------------------------------------------------------------------------- the paper's code, verbatim
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


# ----------------------------------------------------------------------------- folds
def deal(counts, order):
    """GroupKFold's own rule for a given order of the groups: each group goes to the fold that is lightest so
    far (the first one if several are equally light). Returns the fold of each group."""
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


def avx512_in_use():
    """The AVX-512 code paths numpy uses in this process (empty list on a node without AVX-512)."""
    for mod in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            m = __import__(mod, fromlist=["__cpu_features__", "__cpu_dispatch__"])
            feats, disp = dict(m.__cpu_features__), list(m.__cpu_dispatch__)
            return [f for f in disp if feats.get(f, False) and ("AVX512" in f or f == "X86_V4")]
        except Exception:
            continue
    die("numpy does not say which processor-specific code it uses")


def order_without_avx512(counts, features_off):
    """numpy's default sort of the counts in a child process whose numpy has its AVX-512 code switched off.
    Returns GroupKFold's order (largest first), or stops if the child still uses AVX-512."""
    code = ("import sys, json, numpy as np\n"
            "import importlib\n"
            "m = None\n"
            "for mod in ('numpy._core._multiarray_umath', 'numpy.core._multiarray_umath'):\n"
            "    try:\n"
            "        m = importlib.import_module(mod); break\n"
            "    except Exception: pass\n"
            "on = [f for f in m.__cpu_dispatch__ if dict(m.__cpu_features__).get(f, False) and ('AVX512' in f or f == 'X86_V4')]\n"
            "c = np.array(json.loads(sys.stdin.read()))\n"
            "print(json.dumps(dict(on=on, order=[int(x) for x in np.argsort(c)[::-1]])))\n")
    env = dict(os.environ)
    prev = env.get("NPY_DISABLE_CPU_FEATURES", "").split()
    env["NPY_DISABLE_CPU_FEATURES"] = " ".join(prev + [f for f in features_off if f not in prev])
    r = subprocess.run([sys.executable, "-B", "-c", code], input=json.dumps([int(x) for x in counts]),
                       capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        die("the child process for the order without AVX-512 failed: %s" % r.stderr.strip()[-400:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    if out["on"]:
        die("numpy still uses AVX-512 code in the child process: %s" % " ".join(out["on"]))
    return np.array(out["order"], dtype=int)


# ----------------------------------------------------------------------------- the ridge of the paper
def ridge_s7_with(X, Y, fold):
    """score_step15_tahoe.py's ridge_s7 with a given fold for every condition: the folds 0..4 are held out in
    turn, as GroupKFold yields them; double precision throughout, sklearn Ridge with penalty 1."""
    P = np.zeros_like(Y)
    for f in range(NFOLD):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        if len(te) == 0:
            die("an empty fold")
        P[te] = Ridge(alpha=ALPHA).fit(X[tr], Y[tr]).predict(X[te])
    return P


def max_diff(mine, record):
    """Largest absolute difference; infinite if a value is missing on one side only."""
    mine, record = np.asarray(mine, float), np.asarray(record, float)
    if not np.array_equal(np.isnan(mine), np.isnan(record)):
        return float("inf")
    ok = ~np.isnan(mine)
    return float(np.abs(mine[ok] - record[ok]).max()) if ok.any() else 0.0


def load_measured_change(split):
    """Conditions and measured change (true mean minus paired vehicle mean) from the stored vectors of the
    rewired model, seed 0, the reference evaluation of score_step15_tahoe.py; hash-checked."""
    tag = "s15_%s_recipe_seed_0" % split
    pj = os.path.join("provenance", "s15ev_%s.json" % tag)
    if not os.path.isfile(pj):
        die("missing %s" % pj)
    # The vectors are large and not in git. They are looked for in this worktree, then in the folder named by
    # S29_VECTORS_DIR, then in the main checkout beside this worktree. Wherever the file is found, it must have
    # the sha256 of its provenance record, so the place does not matter.
    places = [RES, os.environ.get("S29_VECTORS_DIR", ""), os.path.join("..", "Squidiff", RES)]
    f = next((os.path.join(p, "vectors_%s.npz" % tag) for p in places
              if p and os.path.isfile(os.path.join(p, "vectors_%s.npz" % tag))), None)
    if f is None:
        die("vectors_%s.npz not found in %s" % (tag, ", ".join(os.path.abspath(p) for p in places if p)))
    pv = json.load(open(pj))
    for key, want in (("status", "completed"), ("arm", "recipe"), ("split", split), ("seed", 0)):
        if key in pv and str(pv[key]) != str(want):
            die("%s provenance %s=%r, expected %r" % (tag, key, pv[key], want))
    digest = sha256(f)
    if pv.get("vectors_sha256") != digest:
        die("%s vectors hash differs from provenance" % tag)
    v = np.load(f, allow_pickle=True)
    cond = np.array([str(x) for x in v["cond"]])
    true, C = np.asarray(v["true"], np.float64), np.asarray(v["C"], np.float64)     # as score_step15_tahoe.py
    print("[vectors] %s (sha256 equals its provenance record)" % os.path.abspath(f))
    return cond, true, C, dict(tag=tag, vectors_sha256=digest, read_from=os.path.abspath(f))


def main():
    # ---- guards: nothing written before, nothing modified, everything this script rests on is committed
    if os.path.exists(OUT):
        die("%s exists; this script never overwrites" % OUT)
    if sh("git rev-parse --is-inside-work-tree") != "true":
        die("not inside a git worktree")
    dirty = sh("git status --porcelain --untracked-files=no")
    if dirty:
        die("tracked files are modified:\n" + dirty)
    for q in (PREREG, os.path.basename(__file__), REC_PC, REC_AN):
        if subprocess.run("git ls-files --error-unmatch '%s'" % q, shell=True, capture_output=True).returncode != 0:
            die("%s is not committed" % q)
    commit = sh("git rev-parse --short HEAD")
    print("[guard] no output exists; no tracked file is modified; the pre-registration, this script and the two "
          "records it reads are committed (commit %s)" % commit)

    flags = ""
    try:
        flags = " " + next(l for l in open("/proc/cpuinfo") if l.startswith("flags")).split(":", 1)[1].strip() + " "
    except Exception:
        pass
    cpu = sh("grep -m1 'model name' /proc/cpuinfo").split(":", 1)[-1].strip()
    avx_on = avx512_in_use()
    env = dict(node=platform.node(), cpu=cpu, cpu_has_avx512f=(" avx512f " in flags), numpy_avx512_code_in_use=avx_on,
               python=platform.python_version(), numpy=np.__version__, scipy=version_of("scipy"),
               sklearn=version_of("scikit-learn"), pandas=pd.__version__,
               threads=os.environ.get("OMP_NUM_THREADS", ""), slurm_job=os.environ.get("SLURM_JOB_ID", ""))
    print("[env] %s" % env)
    if avx_on:
        print("[env] numpy uses AVX-512 here: 'default' is the order WITH AVX-512; the order without it is made in a "
              "child process")
    else:
        print("[env] numpy uses no AVX-512 here: 'default' is the order WITHOUT AVX-512; the order with AVX-512 cannot "
              "be made on this node")

    from Squidiff.scrna_datasets import Drug_dose_encoder             # the harness's fingerprint + dose encoder

    pc = pd.read_csv(REC_PC, dtype={"split": str, "condition": str, "line": str, "plate": str, "structure": str},
                     keep_default_na=False, na_values={"ridge_S7_de_overlap": [""], "ridge_S7_dpc50": [""]})
    for col in ("split", "condition", "line", "structure", "dose_nM", "ridge_S7_de_overlap", "ridge_S7_dpc50"):
        if col not in pc.columns:
            die("%s has no column %r" % (REC_PC, col))
    analysis = json.load(open(REC_AN))
    an_split = analysis.get("per_split", analysis.get("results", {}))

    result, all_found = {}, True
    for split in SPLITS:
        print("\n" + "#" * 100 + "\n# %s\n" % split + "#" * 100)
        rec = pc[pc["split"] == split].reset_index(drop=True)
        cond, true, C, vec = load_measured_change(split)
        if len(rec) != len(cond) or not np.array_equal(rec["condition"].values.astype(str), cond):
            die("%s: the conditions of the per-condition record differ from the stored vectors (number or order)" % split)
        n = len(cond)
        drugs = rec["structure"].values.astype(str)                   # SMILES of the held-out structure
        lines = rec["line"].values.astype(str)
        doses = rec["dose_nM"].values.astype(float)                   # the dose value given to the encoder
        lv = sorted(set(str(x) for x in lines))

        # features and target, as score_step15_tahoe.py
        FP = np.array([Drug_dose_encoder([drugs[i]], [doses[i]], comb_num=1)[0] for i in range(n)], dtype=np.float64)
        X = features(FP, lines, lv)
        Y = true - C

        # the groups as GroupKFold sees them: sorted unique structures and their number of conditions
        uniq, inv = np.unique(drugs, return_inverse=True)
        counts = np.bincount(inv)
        tied = int(sum(c for c in np.bincount(counts) if c > 1))      # structures that share their count with another
        print("[design] %d conditions, %d structures, %d lines, %d features; conditions per structure %d to %d; "
              "%d structures share their count with another (%d distinct counts)"
              % (n, len(uniq), len(lv), X.shape[1], counts.min(), counts.max(), tied, len(set(counts.tolist()))))

        # candidate orders (largest first, as GroupKFold: argsort then reversed)
        cands = {"default": np.argsort(counts)[::-1], "stable": np.argsort(counts, kind="stable")[::-1]}
        if avx_on:
            cands["without_avx512"] = order_without_avx512(counts, avx_on)
        folds = {name: deal(counts, order) for name, order in cands.items()}

        # scikit-learn's GroupKFold itself must give the partition of candidate "default"
        sk = np.full(n, -1)
        for f, (_, te) in enumerate(GroupKFold(n_splits=NFOLD).split(X, Y, groups=drugs)):
            sk[te] = f
        sk_of_group = np.array([sk[inv == g][0] for g in range(len(uniq))])
        if any(len(set(sk[inv == g].tolist())) != 1 for g in range(len(uniq))):
            die("%s: GroupKFold put one structure into two folds" % split)
        if canonical(sk_of_group) != canonical(folds["default"]):
            die("%s: scikit-learn's GroupKFold does not give the partition of candidate 'default'; this script's copy "
                "of its rule is wrong for this version of scikit-learn" % split)
        print("[check] scikit-learn's GroupKFold gives the partition of candidate 'default'")

        # distinct partitions, each fitted once
        by_part = {}
        for name, fg in folds.items():
            by_part.setdefault(canonical(fg), []).append(name)
        print("[candidates] %d candidate orders give %d distinct partition(s): %s"
              % (len(cands), len(by_part), "; ".join("%s" % "+".join(v) for v in by_part.values())))

        rec_j = rec["ridge_S7_de_overlap"].values.astype(float)
        rec_d = rec["ridge_S7_dpc50"].values.astype(float)
        an = an_split.get(split, {}).get("ridge_S7_dpc", {})
        parts, found = [], None
        for part, names in by_part.items():
            fg = folds[names[0]]
            P = ridge_s7_with(X, Y, fg[inv])
            j, d = thesis_per_cond(true, C + P, C)
            dj, dd = max_diff(j, rec_j), max_diff(d, rec_d)
            ok = bool(dj <= TOL and dd <= TOL)
            mean_d, med_d = float(np.nanmean(d)), float(np.nanmedian(d))
            n_diff = int(np.sum(~np.isclose(j, rec_j, rtol=0, atol=TOL, equal_nan=True) |
                                ~np.isclose(d, rec_d, rtol=0, atol=TOL, equal_nan=True)))
            print("[partition] %-28s gene identity %.6f (record %.6f, largest difference %.1e); response correlation "
                  "mean %.6f median %.6f (analysis record %s, %s), largest difference %.1e; conditions that differ %d "
                  "of %d -> %s   [%.0f s]"
                  % ("+".join(names), np.nanmean(j), np.nanmean(rec_j), dj, mean_d, med_d,
                     ("%.6f" % an["mean"]) if "mean" in an else "n/a", ("%.6f" % an["median"]) if "median" in an else "n/a",
                     dd, n_diff, n, "REPRODUCES the record" if ok else "does not reproduce", time.time() - T0))
            parts.append(dict(candidates=names, partition=part, fold_of_structure=[int(x) for x in fg],
                              reproduces=ok, largest_difference_gene_identity=dj,
                              largest_difference_response_correlation=dd, conditions_that_differ=n_diff,
                              gene_identity_mean=float(np.nanmean(j)), response_correlation_mean=mean_d,
                              response_correlation_median=med_d))
            if ok:
                if found is not None:
                    die("%s: two different partitions reproduce the record; this cannot be" % split)
                found = parts[-1]
        if found is None:
            all_found = False
            print("[guard 4] %s: NO candidate reproduces the record" % split)
        else:
            print("[guard 4] %s: the folds of the paper are those of candidate(s) %s" % (split, "+".join(found["candidates"])))
        result[split] = dict(n_conditions=n, n_structures=int(len(uniq)), structures=[str(u) for u in uniq],
                             conditions_per_structure=[int(c) for c in counts], structures_with_tied_count=tied,
                             vectors=vec, candidates_available=sorted(cands), partitions=parts,
                             folds_of_the_paper=(None if found is None else dict(
                                 candidates=found["candidates"], partition=found["partition"],
                                 fold_of_structure=found["fold_of_structure"])))

    print("\n" + "=" * 100)
    if not all_found:
        print("GUARD 4 NOT MET: on at least one split no candidate reproduces the committed record. Nothing is written.")
        if not avx_on:
            print("This node has no AVX-512, so the order with AVX-512 was not tried. Run the script on a node with "
                  "AVX-512 before anything else.")
        print("=" * 100)
        sys.exit(1)
    print("GUARD 4 MET on the three splits.")
    print("=" * 100)
    for split in SPLITS:
        r = result[split]["folds_of_the_paper"]
        print("  %-13s candidate(s) %-32s folds %s" % (split, "+".join(r["candidates"]), r["partition"]))
    record = dict(step="29, guard 4", prereg=PREREG, prereg_sha256=sha256(PREREG), script=os.path.basename(__file__),
                  script_sha256=sha256(__file__), commit=commit, env=env, tolerance=TOL,
                  inputs={REC_PC: sha256(REC_PC), REC_AN: sha256(REC_AN)},
                  note=("fold_of_structure gives the fold (0 to 4, GroupKFold's numbering) of each entry of 'structures' "
                        "(sorted, as numpy.unique returns them). 'partition' is the same with folds renumbered by first "
                        "appearance."),
                  per_split=result, seconds=round(time.time() - T0, 1))
    if os.path.exists(OUT):
        die("%s appeared during the run; not overwritten" % OUT)
    with open(OUT, "w") as f:
        json.dump(record, f, indent=1)
    print("[written] %s  sha256 %s   [%.0f s]" % (OUT, sha256(OUT), time.time() - T0))


if __name__ == "__main__":
    main()
