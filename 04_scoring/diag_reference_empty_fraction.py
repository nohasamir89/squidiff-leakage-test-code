#!/usr/bin/env python3
"""
diag_reference_empty_fraction.py

Purpose
-------
Give the manuscript's "77–94 % of conditions had no gene significantly changed"
(sci-Plex, Results 1) a direct, committed record. The number is read from the
Step 14 facets-v2 record, leakage_results/step14_long_v2.csv, and written to
leakage_results/reference_empty_fraction.json together with the commit hash.

Logic
-----
In Step 14 the differential-expression reference of a condition is the set of
genes called significant between the measured treated cells and their paired
vehicle cells (Wilcoxon rank-sum, Benjamini–Hochberg 5 % within the condition,
200 genes). The control-paste submission is the paired vehicle cells themselves,
so its own significant set is empty in every condition. The significant-set
Jaccard of control-paste against the reference is therefore

    1  if the reference set is empty   (Jaccard of two empty sets, as scored)
    0  if the reference set is not empty

and its mean over a line's gated conditions is exactly the fraction of those
conditions whose reference set is empty. No approximation is involved; the
script only reads that facet back and prints it per split and line.

The gate: a condition enters the DE facets only with >= 20 distinct real cells
(Step 14 record: 439/443, 442/444, 452/455 conditions on splits 1, 0, 2), so the
fraction is over gated conditions.

Usage
-----
    python diag_reference_empty_fraction.py            # from the repo root
    python diag_reference_empty_fraction.py --csv PATH --out PATH

The script refuses to run if any tracked file is modified (same rule as the
analysis scripts of this repository), so that the commit it records is the
commit that produced the output.
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def git(cmd):
    """Run a git command in the current directory and return its stdout, stripped."""
    return subprocess.run(["git"] + cmd, capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="leakage_results/step14_long_v2.csv",
                    help="Step 14 facets-v2 long record (committed)")
    ap.add_argument("--out", default="leakage_results/reference_empty_fraction.json",
                    help="where to write the JSON record")
    args = ap.parse_args()

    # ---- git-clean guard: tracked files must be unmodified -------------------
    dirty = git(["status", "--porcelain", "--untracked-files=no"])
    if dirty:
        sys.exit("FATAL: tracked files are modified; commit first.\n" + dirty)
    commit = git(["rev-parse", "HEAD"])

    # ---- read the record ----------------------------------------------------
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        sys.exit(f"FATAL: record not found: {csv_path}")

    rows = list(csv.DictReader(open(csv_path, newline="")))
    expected_cols = {"split", "line", "submission", "facet", "kind", "value"}
    missing = expected_cols - set(rows[0].keys())
    if missing:
        sys.exit(f"FATAL: columns missing from {csv_path}: {sorted(missing)}")

    # The facet is the significant-set Jaccard. Its exact name is taken from the
    # file: the unique facet whose name contains 'jac' (case-insensitive).
    facets = sorted({r["facet"] for r in rows})
    jac = [f for f in facets if "jac" in f.lower()]
    if len(jac) != 1:
        sys.exit(f"FATAL: expected one Jaccard facet, found {jac}; facets in file: {facets}")
    jac = jac[0]

    # The control-paste submission: the unique submission name containing
    # 'control' and 'paste' (the record writes it as control_paste).
    subs = sorted({r["submission"] for r in rows})
    cp = [s for s in subs if "control" in s.lower() and "paste" in s.lower()]
    if len(cp) != 1:
        sys.exit(f"FATAL: expected one control-paste submission, found {cp}; submissions: {subs}")
    cp = cp[0]

    # ---- select: control-paste, Jaccard facet, raw values -------------------
    sel = [r for r in rows if r["submission"] == cp and r["facet"] == jac and r["kind"] == "raw"]
    if not sel:
        sys.exit("FATAL: no raw control-paste Jaccard rows found")

    per_split = defaultdict(dict)
    for r in sel:
        per_split[r["split"]][r["line"]] = float(r["value"])

    # ---- print ----------------------------------------------------------------
    print(f"[record] {csv_path}  commit {commit[:7]}")
    print(f"[facet] '{jac}' of submission '{cp}' (raw) = fraction of gated conditions "
          f"with an empty reference set")
    allvals = []
    for split in sorted(per_split):
        lines = per_split[split]
        s = "  ".join(f"{line} {v:.4f}" for line, v in sorted(lines.items()))
        print(f"  {split}: {s}")
        allvals.extend(lines.values())
    lo, hi = min(allvals), max(allvals)
    print(f"[range over splits and lines] {lo:.4f} – {hi:.4f}  "
          f"({100*lo:.0f}–{100*hi:.0f} % of gated conditions)")

    # ---- write the JSON record -------------------------------------------------
    out = {
        "what": ("fraction of gated conditions whose DE reference set (measured treated vs paired "
                 "vehicle, Wilcoxon, BH 5 % within condition, 200 genes) is empty; equals the raw "
                 "significant-set Jaccard of the control-paste submission, read from the Step 14 "
                 "facets-v2 record"),
        "source_csv": str(csv_path),
        "facet": jac,
        "submission": cp,
        "per_split": {s: dict(sorted(l.items())) for s, l in per_split.items()},
        "min": lo,
        "max": hi,
        "commit": commit,
        "written": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
