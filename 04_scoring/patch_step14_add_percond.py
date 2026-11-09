#!/usr/bin/env python3
"""
patch_step14_add_percond.py — add a per-condition output to score_step14.py for the Step 28 sci-Plex re-aggregation.

Why
---
Step 28 (salt-twin sensitivity, PREREG_step28_salt_twins.md) re-aggregates committed per-condition records with the
parent-twin compounds removed. On Tahoe-100M those records exist. On sci-Plex they do not: score_step14.py writes
only the aggregated step14_long.csv. This patch adds a per-condition record (step14_per_condition.csv) so the
sci-Plex gene identity can be re-aggregated the same way, without re-implementing any of score_step14's loading,
hash checks or scoring.

What it changes (additive only — no existing output is altered)
--------------------------------------------------------------
1. a new path constant OUT_PC next to OUT/OUT_CSV;
2. an accumulator pc_rows alongside the existing rows list;
3. right after the per-split `cm` lambda is defined, a loop that records, for every submission and condition, the
   seed-averaged thesis metrics cm(name,'jac'/'en'/'dpc') with the condition's cell line and structure (SMILES);
4. a final write of pc_rows to OUT_PC, beside the existing write of rows to OUT_CSV.

The reproduction gate after re-running is that step14_long.csv is byte-identical to its committed version (it has
no commit field, so it must not change); step14_analysis.json legitimately changes only in its `commit` field.

This script applies four exact-match edits. Each anchor must occur exactly once or the script stops and changes
nothing. It also refuses to run twice (if OUT_PC is already present it has already been applied).

Run from the worktree:
    python patch_step14_add_percond.py
Then verify:
    grep -nE "OUT_PC|pc_rows|step14_per_condition" score_step14.py
    python -c "import ast; ast.parse(open('score_step14.py').read()); print('parses')"
"""
import sys

SRC = 'score_step14.py'

EDITS = [
    # 1. path constant
    ("OUT, OUT_CSV = os.path.join(RES, 'step14_analysis.json'), os.path.join(RES, 'step14_long.csv')",
     "OUT, OUT_CSV = os.path.join(RES, 'step14_analysis.json'), os.path.join(RES, 'step14_long.csv')\n"
     "OUT_PC = os.path.join(RES, 'step14_per_condition.csv')   # Step 28: per-condition thesis metrics (additive)"),

    # 2. accumulator
    ("results, inputs, gate, rows = {}, {}, {}, []",
     "results, inputs, gate, rows, pc_rows = {}, {}, {}, [], []    # pc_rows: Step 28 per-condition record"),

    # 3. per-condition collection, right after the cm lambda
    ("    cm = lambda name, k: np.nanmean(np.vstack([x[k] for x in subs[name]]), axis=0)\n",
     "    cm = lambda name, k: np.nanmean(np.vstack([x[k] for x in subs[name]]), axis=0)\n"
     "    # Step 28 (salt-twin sensitivity): record each submission's seed-averaged thesis metrics per condition,\n"
     "    # with the condition's cell line and structure (SMILES), so sci-Plex gene identity can be re-aggregated\n"
     "    # with the parent-twin compounds removed. This only appends to pc_rows; nothing above is altered, and the\n"
     "    # gate after re-running is that step14_long.csv stays byte-identical.\n"
     "    for _nm in subs:\n"
     "        _j, _e, _d = cm(_nm, 'jac'), cm(_nm, 'en'), cm(_nm, 'dpc')\n"
     "        for _i in range(n):\n"
     "            pc_rows.append(dict(split=split, condition=str(cond[_i]), line=lines[_i], structure=drugs[_i],\n"
     "                                submission=_nm, de_overlap=float(_j[_i]),\n"
     "                                energy=float(_e[_i]) if np.isfinite(_e[_i]) else None,\n"
     "                                dpc50=float(_d[_i])))\n"),

    # 4. final write, beside the existing one
    ("pd.DataFrame(rows).to_csv(OUT_CSV, index=False)",
     "pd.DataFrame(rows).to_csv(OUT_CSV, index=False)\n"
     "pd.DataFrame(pc_rows).to_csv(OUT_PC, index=False)   # Step 28 per-condition record"),
]


def main():
    s = open(SRC).read()
    if 'OUT_PC' in s or 'pc_rows' in s or 'step14_per_condition' in s:
        sys.exit('FATAL: score_step14.py already contains the per-condition addition; nothing to do.')
    for old, new in EDITS:
        c = s.count(old)
        if c != 1:
            sys.exit('FATAL: anchor occurs %d times, expected 1:\n  %r' % (c, old[:70]))
        s = s.replace(old, new)
    # sanity: the file still parses
    import ast
    ast.parse(s)
    open(SRC, 'w').write(s)
    print('applied 4 edits to %s; it still parses. Added: OUT_PC, pc_rows, the per-condition loop, the final write.'
          % SRC)
    print('verify:  grep -nE "OUT_PC|pc_rows|step14_per_condition" %s' % SRC)


if __name__ == '__main__':
    main()
