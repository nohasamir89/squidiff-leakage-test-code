#!/usr/bin/env python3
"""
diag_step28_sciplex_rank.py - POST HOC context for Step 28 verdict L on sci-Plex (salt-twin sensitivity).

Purpose
-------
Step 28 group B (PREREG_step28_salt_twins.md section 4, amendments 15-16; scorer score_step28_sciplex.py, job
19289646) found that on sci-Plex split 2 the rewired model (comparator `fm_50`) scored the one held-out compound
whose salt twin is in training 0.040 above the retained compounds on gene identity (verdict L, pre-registered),
and 0.006 below them on split 0. With a single compound per split, the structure-clustered bootstrap cannot test
either difference (one cluster). This script records, as POST HOC context - written after that result was seen,
not pre-registered - where the excluded compound sits among ALL held-out compounds of its split, for:

  1. "rewired"        the rewired model's gene identity per compound;
  2. "compound_blind" the compound-blind average's gene identity per compound (`mean_response`: one delta per cell
                      line, no compound information, so it cannot benefit from a twin in training);
  3. "gain"           their difference per compound (rewired minus compound-blind) - the rewired model's gain over
                      the compound-blind average.

For each it records the excluded compound's value, the mean and sample standard deviation over the split's
compounds, the excluded compound's rank (1 = highest) and how many other compounds have a strictly larger value.
Supplementary Note 7 cites item 3 (split 2: +0.108, exceeded by six of the 37 other compounds; split 0: rank 26 of
37) and verdict L's compound-blind difference (+0.009, from step28_sciplex.json, not from this script).

Logic
-----
- Inputs are committed records only:
    leakage_results/step28_sciplex.json        which structure was excluded on each split, and the comparator;
    leakage_results/step14_per_condition.csv   per condition and submission, the seed-averaged gene identity
                                               (`de_overlap`), with the condition's `structure` (test-file SMILES).
- Guard: both inputs must be tracked by git and identical to HEAD, and the output must not exist yet (outputs are
  never overwritten). Anything else stops the script before a value is computed.
- Per split that has an exclusion (splits 0 and 2; split 1 has none by construction): the per-compound mean of
  `de_overlap` over the compound's conditions (all cell lines and doses), for the comparator and for
  `mean_response`; their per-compound difference; then the summary statistics above (pandas mean; pandas std,
  i.e. ddof = 1; rank = 1 + number of compounds with a strictly larger value).
- Reproduction check: the same quantities were first printed by an inline diagnostic on 2026-11-09, quoted in
  step28_findings.md. The script must reproduce those printed values to their 4 printed decimals (tolerance 5e-5,
  half a unit in the last printed place) and the ranks and compound counts exactly; a mismatch stops it before
  anything is written. The EXPECTED block below is that check, never an input to any computation.
- Output: leakage_results/step28_sciplex_posthoc_rank.json, plus the printed table. CPU only, a few seconds; it
  imports pandas but not scipy or anndata, so it can run on the login node.

Run from the worktree (squidiff env):
    python diag_step28_sciplex_rank.py
"""
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd

# ---------------------------------------------------------------------------------------------------------------
# Paths and fixed settings
# ---------------------------------------------------------------------------------------------------------------
RES = 'leakage_results'
IN_JSON = os.path.join(RES, 'step28_sciplex.json')          # Step 28 group-B record (exclusions, comparator)
IN_PC = os.path.join(RES, 'step14_per_condition.csv')        # per-condition thesis metrics (amendment 15)
OUT = os.path.join(RES, 'step28_sciplex_posthoc_rank.json')  # this script's record

REF = 'mean_response'   # the compound-blind average: one delta per cell line, carries no compound information
TOL = 5e-5              # the inline diagnostic printed 4 decimals; half a unit in the last printed place

# Values printed by the inline diagnostic of 2026-11-09 (quoted in step28_findings.md, sci-Plex section):
# (excluded value, mean over compounds, sd over compounds, rank with 1 = highest, number of compounds).
# A reproduction check only - nothing below is computed from these numbers.
EXPECTED = {
    'drug_split_0': {'rewired':        (0.6069, 0.6124, 0.0179, 24, 37),
                     'compound_blind': (0.5579, 0.5538, 0.0127, 14, 37),
                     'gain':           (0.0491, 0.0586, 0.0181, 26, 37)},
    'drug_split_2': {'rewired':        (0.6496, 0.6106, 0.0267, 4, 38),
                     'compound_blind': (0.5415, 0.5324, 0.0273, 9, 38),
                     'gain':           (0.1081, 0.0782, 0.0323, 7, 38)},
}


def git(*args):
    """Run one git command in the current directory (the worktree); return (exit code, stripped stdout)."""
    p = subprocess.run(['git'] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return p.returncode, p.stdout.strip()


def md5(path):
    """md5 of a file, recorded in the output so the record names exactly the inputs it was computed from."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def guard():
    """Stop unless both inputs are committed and unmodified and the output does not exist yet."""
    for f in (IN_JSON, IN_PC):
        rc, _ = git('ls-files', '--error-unmatch', f)
        if rc != 0:
            sys.exit('FATAL: %s is not tracked by git; commit the Step 28 records first.' % f)
        rc, _ = git('diff', '--quiet', 'HEAD', '--', f)
        if rc != 0:
            sys.exit('FATAL: %s differs from HEAD; this diagnostic reads committed records only.' % f)
    if os.path.exists(OUT):
        sys.exit('FATAL: %s already exists; outputs are never overwritten.' % OUT)


def per_compound(pc, split, submission):
    """Mean gene identity per compound (structure) over its conditions, for one submission on one split.
    Every condition appears once per submission in the per-condition record, so this is the plain mean over the
    compound's conditions (all cell lines and doses); NaN entries are ignored and empty compounds dropped."""
    rows = pc[(pc['split'] == split) & (pc['submission'] == submission)]
    if rows.empty:
        sys.exit('FATAL: no rows for submission %s on %s in %s.' % (submission, split, IN_PC))
    return rows.groupby('structure')['de_overlap'].mean().dropna()


def place(series, key):
    """Where compound `key` sits in `series` (one value per compound): its value, the mean and sample sd over all
    compounds, its rank (1 = highest; ties share the better rank) and how many compounds are strictly larger."""
    if key not in series.index:
        sys.exit('FATAL: the excluded structure is missing from the per-compound values.')
    value = float(series[key])
    n_larger = int((series > value).sum())
    return dict(excluded=value, mean=float(series.mean()), sd=float(series.std()),
                rank=n_larger + 1, n_compounds=int(len(series)), n_larger=n_larger)


def check(split, name, got):
    """Compare one computed row with the value printed by the inline diagnostic; stop on any mismatch."""
    exp = EXPECTED[split][name]
    for label, a, b in (('excluded', got['excluded'], exp[0]), ('mean', got['mean'], exp[1]),
                        ('sd', got['sd'], exp[2])):
        if abs(a - b) > TOL:
            sys.exit('FATAL: %s %s %s = %.6f does not reproduce the printed %.4f.' % (split, name, label, a, b))
    if got['rank'] != exp[3] or got['n_compounds'] != exp[4]:
        sys.exit('FATAL: %s %s rank %d of %d does not reproduce the printed %d of %d.'
                 % (split, name, got['rank'], got['n_compounds'], exp[3], exp[4]))


def main():
    guard()
    _, commit = git('rev-parse', '--short', 'HEAD')          # recorded so the output names the code it came from
    j = json.load(open(IN_JSON))
    comparator = j['comparator']                              # 'fm_50', the rewired model, as Step 28 recorded it
    pc = pd.read_csv(IN_PC)

    out = dict(
        purpose='POST HOC context for Step 28 verdict L on sci-Plex: rank of the excluded (salt-twin) compound '
                'among all held-out compounds of its split, on gene identity. Written after the group-B result '
                '(job 19289646) was seen; not pre-registered.',
        post_hoc=True, commit=commit, comparator=comparator, reference=REF,
        inputs={IN_JSON: md5(IN_JSON), IN_PC: md5(IN_PC)},
        rank_rule='1 = highest; rank = 1 + number of compounds with a strictly larger value',
        sd_rule='sample standard deviation over compounds (ddof = 1)',
        splits={})

    print('split          quantity         excluded   mean     sd      rank        larger')
    for split in sorted(j['splits']):
        excluded = j['splits'][split]['excluded']
        if not excluded:
            continue                                          # split 1: no twin, nothing to place
        if len(excluded) != 1:
            sys.exit('FATAL: %s lists %d excluded structures; this diagnostic is for one compound per split.'
                     % (split, len(excluded)))
        key = excluded[0]['matched_structure']                # the structure value as it appears in the record

        rewired = per_compound(pc, split, comparator)
        blind = per_compound(pc, split, REF)
        gain = (rewired - blind).dropna()                     # aligned on structure; one value per compound

        rows = {}
        for name, series in (('rewired', rewired), ('compound_blind', blind), ('gain', gain)):
            got = place(series, key)
            check(split, name, got)                           # stop before writing if anything differs
            rows[name] = got
            print('%-14s %-16s %+.4f   %+.4f  %.4f  %2d of %2d    %d'
                  % (split, name, got['excluded'], got['mean'], got['sd'], got['rank'], got['n_compounds'],
                     got['n_larger']))
        out['splits'][split] = dict(excluded_structure=key, record_smiles=excluded[0].get('record_smiles'),
                                    quantities=rows)

    if not out['splits']:
        sys.exit('FATAL: no split with an exclusion was found in %s.' % IN_JSON)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('[reproduced] every value matches the 2026-11-09 inline diagnostic (tolerance %g); ranks exact' % TOL)
    print('[written] %s' % OUT)


if __name__ == '__main__':
    main()
