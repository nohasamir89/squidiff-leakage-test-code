#!/usr/bin/env python3
"""
patch_gbm_eval_scale.py — write _v2 copies of the three GBM evaluation scripts
that compare predictions and truth on the same scale.

Purpose
-------
The GBM models were trained on normalize_total(1e4) + log1p expression
(prep_gbm_labels.py), so the decoder returns log1p-scale values. The three
evaluation scripts then applied np.log1p(...) to the decoded predictions once
more ("match log1p truth scale") before correlating them with the singly-logged
truth. Pearson correlation is not invariant to that second log, so every GBM
number in the manuscript (Fig. 1d sweep, Fig. 1e reproduction, the 0.70
reconstruction gate) must be recomputed with the predictions on the scale they
were produced on. Nothing is retrained; the checkpoints are evaluated again.

Logic
-----
For each script, a _v2 copy is written in the same folder:
  1. every line containing  np.log1p(np.clip(decode_mean(  loses the outer
     np.log1p( ... ) — the clip at zero is kept;
  2. the count of such lines must equal the expected count (3 / 2 / 2), else stop;
  3. sensitivity script only: RESULTS_CSV -> gene_sensitivity_SEEDED_v2.csv, and
     the N = 200 smoke-test abort (threshold 0.95, set for the old scale) becomes
     a printed warning so the row is still written;
  4. a header note and a run-time banner are added;
  5. the original files are not modified; an existing _v2 file stops the run.

Usage
-----
    python3 patch_gbm_eval_scale.py            # patches the three scripts in place as _v2
    python3 patch_gbm_eval_scale.py --check    # only reports what it would change
"""
import argparse
import os
import sys

SCRIPTS = [
    # (path, expected number of double-log lines)
    ("/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/eval_gbm_sensitivity.py", 3),
    ("/scratch/ni2071/Diffusion/gbm_run/eval_gbm_deltaz.py", 2),
    ("/scratch/ni2071/Diffusion/gbm_run_200/eval_gbm_deltaz_200.py", 2),
]
NEEDLE = "np.log1p(np.clip(decode_mean("
SUFFIX = ", 0, None))"
BANNER = ('print("[scale] v2: predictions compared on the training (log1p) scale; '
          'the second log1p of the original evaluation is removed")\n')
HEADER = ("# v2 (patched by patch_gbm_eval_scale.py): the decoded predictions are NOT logged a\n"
          "# second time; they are clipped at zero and compared with the log1p truth directly.\n"
          "# The original script is kept unchanged next to this file for the record.\n")


def patch(path, expected, check_only):
    src = open(path).read().splitlines(keepends=True)
    out, n_fixed = [], 0
    for line in src:
        if NEEDLE in line:
            if line.count(NEEDLE) != 1 or SUFFIX not in line:
                sys.exit(f"FATAL: unexpected form in {path}: {line.rstrip()}")
            head, tail = line.split(NEEDLE, 1)
            inner, rest = tail.split(SUFFIX, 1)
            line = f"{head}np.clip(decode_mean({inner}, 0, None){rest}"
            n_fixed += 1
        out.append(line)
    if n_fixed != expected:
        sys.exit(f"FATAL: {path}: found {n_fixed} double-log lines, expected {expected}")

    text = "".join(out)
    if path.endswith("eval_gbm_sensitivity.py"):
        old_csv = "'gene_sensitivity_SEEDED.csv'"
        if text.count(old_csv) != 1:
            sys.exit(f"FATAL: RESULTS_CSV line not found once in {path}")
        text = text.replace(old_csv, "'gene_sensitivity_SEEDED_v2.csv'")
        old_abort = 'raise SystemExit("Smoke test failed at N=200. Aborting before writing CSV.")'
        if text.count(old_abort) != 1:
            sys.exit(f"FATAL: smoke-test abort line not found once in {path}")
        text = text.replace(old_abort,
                            'print("[v2] smoke-test threshold (0.95) was set for the old scale; '
                            'not aborting, row is written")')

    # banner after the imports: before the first 'def ' at column 0
    first_def = text.find("\ndef ")
    if first_def < 0:
        sys.exit(f"FATAL: no top-level def in {path}")
    text = text[:first_def + 1] + BANNER + text[first_def + 1:]
    # header after the shebang / docstring: simplest is to prepend after line 1 if shebang
    lines = text.splitlines(keepends=True)
    if lines[0].startswith("#!"):
        text = lines[0] + HEADER + "".join(lines[1:])
    else:
        text = HEADER + text

    dst = path[:-3] + "_v2.py"
    if os.path.exists(dst):
        sys.exit(f"FATAL: {dst} exists")
    print(f"[{'check' if check_only else 'write'}] {dst}: {n_fixed} lines corrected")
    if not check_only:
        open(dst, "w").write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for path, expected in SCRIPTS:
        if not os.path.isfile(path):
            sys.exit(f"FATAL: missing {path}")
        patch(path, expected, a.check)


if __name__ == "__main__":
    main()
