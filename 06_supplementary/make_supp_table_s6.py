#!/usr/bin/env python3
"""
make_supp_table_s6.py - Supplementary Table S6: response correlation of every model and reference on sci-Plex
(Results 2).

Purpose
-------
Results 2 states that, on the correlation between predicted and measured change across the 50 most responsive
genes, every generative model on sci-Plex scored below both ridge regressions and below the compound-blind average,
and cites Supplementary Table S6. This script assembles that table from the committed sci-Plex analysis record, so
that every number traces to the record and is rounded once, from full precision.

One row per entry (three references, then every generative model scored in that analysis). Two columns: the mean
and the median of the per-condition correlation, each for splits 1 / 0 / 2. The statement in the text is about the
mean over conditions. For the median it does not hold everywhere, and the table and its caption say where.

Logic
-----
- Input (committed record only; see supp_table_common.guard):
      leakage_results/step14_analysis.json   per_split/<split>/thesis_metrics/<arm>/{dpc_mean, dpc_median}
  Record names -> names in the paper: ridge_train = training-set ridge; ridge_S7 = cross-validated ridge;
  mean_response = compound-blind average; base = original model; adaln_ddim_1000 = original model with adaptive
  normalisation; fm_50 = rewired model; fm_20 = rewired model sampled in 20 steps; guidance_w2, attn_cross and
  mlp_ctrl = the rewired model with guidance, with cross-attention, and the parameter-matched control; s9_* and
  s10_* = the variants given mechanism-of-action labels or a learned chemical embedding (development split only).
  control_paste (unaltered vehicle cells) has no predicted change and no value on this measure.
- Checks, all before anything is written (a failure stops the script):
    1. Every arm of the record is placed in the table (none is left out), and each arm has a value on exactly the
       splits listed for it.
    2. Mean over conditions: every generative model is below each of the three references on every split on which
       it was scored. This is the statement of Results 2.
    3. Median over conditions: the cases in which a generative model is not below all three references are found
       by the script and must be exactly the ones the caption names (EXPECTED_MEDIAN_EXCEPTIONS).
    4. The numbers of conditions and the values quoted in the manuscript reproduce (EXPECTED below; checks only,
       never inputs to a computation).
- Rounding: three decimals, each value rounded once from full precision.
- Outputs (never overwritten): supplementary_tables/supp_table_S6.csv and .md (see supp_table_common).

Run from the research worktree root (supp_table_common.py must be in the same folder):
    python3 make_supp_table_s6.py
"""
import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import SPLITS, die, guard, is_number, load_json, same, table, triple, write_outputs

TABLE = 'S6'
IN = dict(
    sciplex=('leakage_results/step14_analysis.json',
             'c7f021b27272bbac983db2f58d694dddc5dd7de234869ed8ec5d5e64bcf6f0ae'),
)
ALL = tuple(SPLITS)
DEV = ('drug_split_1',)                                  # the development split
# (label in the paper, arm key in the record, kind, splits on which the arm was scored)
ROWS = [
    ('Training-set ridge', 'ridge_train', 'reference', ALL),
    ('Cross-validated ridge', 'ridge_S7', 'reference', ALL),
    ('Compound-blind average', 'mean_response', 'reference', ALL),
    ('Original model', 'base', 'model', ALL),
    ('Original model with adaptive normalisation', 'adaln_ddim_1000', 'model', ALL),
    ('Rewired model', 'fm_50', 'model', ALL),
    ('Rewired model, 20 integration steps', 'fm_20', 'model', ALL),
    ('Rewired model with classifier-free guidance (weight 2)', 'guidance_w2', 'model', ALL),
    ('Rewired model with cross-attention', 'attn_cross', 'model', ALL),
    ('Rewired model, parameter-matched control without attention', 'mlp_ctrl', 'model', ALL),
    ('Variant: mechanism-of-action labels in place of the fingerprint', 's9_pathway', 'model', DEV),
    ('Variant: mechanism-of-action labels added to the fingerprint', 's9_both', 'model', DEV),
    ('Variant: learned chemical embedding in place of the fingerprint', 's10_chemvae', 'model', DEV),
    ('Variant: learned chemical embedding added to the fingerprint', 's10_fp_chemvae', 'model', DEV),
]
NOT_SCORED = ('control_paste',)                          # in the record, but with no value on this measure

# What the manuscript and the caption state (checks only, never inputs).
EXPECTED_CONDITIONS = [443, 444, 455]                    # Methods: held-out conditions of splits 1, 0 and 2
EXPECTED_MEDIAN_EXCEPTIONS = [                           # caption: where the median statement does not hold
    ('drug_split_2', 'attn_cross'),
    ('drug_split_2', 'mlp_ctrl'),
]


def main():
    guard(IN, TABLE)
    rec = load_json(IN['sciplex'][0])
    per = dict((s, rec['per_split'][s]['thesis_metrics']) for s in SPLITS)

    def value(arm, measure, split):
        v = per[split].get(arm, {}).get(measure)
        return v if is_number(v) else None

    # ---- check 1: every arm placed; values on exactly the listed splits ---------------------------------------
    placed = set(arm for _, arm, _, _ in ROWS) | set(NOT_SCORED)
    for s in SPLITS:
        extra = sorted(set(per[s]) - placed)
        if extra:
            die('check 1: arm(s) of the record not placed in the table on %s: %s' % (s, ', '.join(extra)))
    for label, arm, kind, splits in ROWS:
        for s in SPLITS:
            has = value(arm, 'dpc_mean', s) is not None and value(arm, 'dpc_median', s) is not None
            if has != (s in splits):
                die('check 1: %s %s a value on %s' % (arm, 'has' if has else 'lacks', s))
    for arm in NOT_SCORED:
        if any(value(arm, 'dpc_mean', s) is not None for s in SPLITS):
            die('check 1: %s has a value but is listed as not scored' % arm)

    refs = [(label, arm) for label, arm, kind, _ in ROWS if kind == 'reference']
    models = [(label, arm, splits) for label, arm, kind, splits in ROWS if kind == 'model']

    # ---- check 2: mean over conditions - every model below every reference -----------------------------------------
    n2 = 0
    for label, arm, splits in models:
        for s in splits:
            for rlabel, rarm in refs:
                if not value(arm, 'dpc_mean', s) < value(rarm, 'dpc_mean', s):
                    die('check 2: on the condition mean, %s is not below %s on %s' % (arm, rarm, s))
                n2 += 1

    # ---- check 3: median over conditions - find the exceptions ------------------------------------------------------
    exceptions = []
    for s in SPLITS:
        for label, arm, splits in models:
            if s in splits and not all(value(arm, 'dpc_median', s) < value(rarm, 'dpc_median', s)
                                       for _, rarm in refs):
                exceptions.append((s, arm))
    same(sorted(exceptions), sorted(EXPECTED_MEDIAN_EXCEPTIONS), 'check 3, exceptions on the condition median')
    for s, arm in exceptions:                                # the caption says "above the three references"
        if not all(value(arm, 'dpc_median', s) > value(rarm, 'dpc_median', s) for _, rarm in refs):
            die('check 3: %s on %s is not above all three references on the median' % (arm, s))

    # ---- check 4: numbers of conditions -----------------------------------------------------------------------------
    n_cond = [rec['per_split'][s]['gate']['n_conditions'] for s in SPLITS]
    same(n_cond, EXPECTED_CONDITIONS, 'check 4, held-out conditions per split')

    # ---- the table --------------------------------------------------------------------------------------------------
    long_rows = []
    rows = []
    for label, arm, kind, splits in ROWS:
        means = [value(arm, 'dpc_mean', s) for s in SPLITS]
        medians = [value(arm, 'dpc_median', s) for s in SPLITS]
        rows.append([label, triple(means), triple(medians)])
        for s, m, md_ in zip(SPLITS, means, medians):
            if m is None:
                continue
            for stat, v in (('mean over conditions', m), ('median over conditions', md_)):
                long_rows.append(dict(row=label, kind=kind, split=s, statistic=stat, value=v,
                                      source_record='step14_analysis.json',
                                      source_key='thesis_metrics/%s/%s' % (arm, 'dpc_mean' if stat.startswith('mean') else 'dpc_median')))
    md = ['Supplementary Table S6 | Response correlation on sci-Plex (Results 2). Pearson correlation between the '
          'predicted and the measured mean change over the 50 genes with the largest absolute measured change, '
          'computed for each held-out condition (%s conditions on splits 1, 0 and 2) and summarised as the mean '
          'and as the median over conditions (splits 1 / 0 / 2); the generative models are averaged over five seeds '
          'within a condition first. On the mean over conditions, every generative model lies below the two ridge '
          'regressions and the compound-blind average on every split; this is the statement made in the text. On '
          'the median over conditions the same holds on splits 1 and 0, but on split 2 the model with '
          'cross-attention and its parameter-matched control lie above the three references. The variants in the '
          'last four rows were trained on the development split only (Supplementary Table S10), which a dash '
          'marks. The unaltered vehicle cells have no predicted change and are not scored on this measure.'
          % ('%d, %d and %d' % tuple(n_cond)),
          '']
    md += table(['Entry', 'Mean over conditions', 'Median over conditions'], rows)
    md += ['',
           'Source: step14_analysis.json (sha256 %s). Assembled by make_supp_table_s6.py.' % IN['sciplex'][1][:16]]
    write_outputs(TABLE, long_rows, md)
    print('[checks] 1 (every arm placed, values on the listed splits), 2 (%d comparisons on the mean: every model '
          'below every reference), 3 (median: exceptions are exactly %s), 4 (conditions per split) all passed'
          % (n2, ', '.join('%s on %s' % (a, s) for s, a in sorted(exceptions))))


if __name__ == '__main__':
    main()
