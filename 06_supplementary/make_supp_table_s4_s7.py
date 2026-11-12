#!/usr/bin/env python3
"""
make_supp_table_s4_s7.py - Supplementary Tables S4 and S7: what the population distance on Tahoe-100M is made of,
for the original model (S4; Results 1; Fig. 1c) and for the rewired model (S7; Results 2).

Purpose
-------
Results 1 states that the original model's generated cells were 2.5 times more dispersed than the measured cells,
carried 1.7 times their total expression, were 27 % zeros where the data were 89 % zeros, and had a mean profile
displaced by 3.3 times the measured response; it cites Supplementary Table S4, and so does the caption of Fig. 1c.
Results 2 states that the rewired model's cells matched the spread and library size of the measured cells, that
their mean profile correlated with the measured mean at 0.97, and that the mean still lay 1.67 from the measured
treated mean against 1.37 for the vehicle mean; it cites Supplementary Table S7. Both sets of numbers come from one
exploratory analysis (diag_step15_cells.py: split 1, seed 0, 2,899 conditions) run on the two models. This script
assembles the two tables from its committed record, in one layout, so that every number traces to the record and is
rounded once. One script builds both tables because they are the same analysis applied to two models.

Logic
-----
- Inputs (committed records only; see supp_table_common.guard):
      tahoe_results/step15_diag_cells_drug_split_1_seed_0.json
            summary/<arm>/{n_conditions, mean/<quantity>, median/<quantity>}; arms base (original model) and
            recipe (rewired model). Every value of the two tables comes from this file.
      tahoe_results/step15_diag_cells_drug_split_1_seed_0_per_condition.csv
            one row per arm and condition, written by the same run; Fig. 1c is drawn from it. Check only.
      tahoe_results/step15_model_analysis.json
            the scored population distance of seed 0 on split 1. Check only.
  Quantities of the record (definitions from diag_step15_cells.py) -> names in the tables:
      xx, yy             mean Euclidean distance between two generated / two measured cells       -> spread
      lib_pred, lib_true mean over cells of the sum of the 200 expression values                  -> total expression
      zero_pred, zero_true  fraction of entries that are exactly zero                             -> zero entries
      min_pred           smallest generated value (zero in every condition; stated in the caption, not tabulated)
      xy                 mean Euclidean distance between a generated and a measured cell
      energy             2 xy - xx - yy, the scored population distance                            -> population distance
      profile_r          Pearson correlation across genes between the generated and the measured mean profile
      slope_pred_on_true least-squares slope of the generated on the measured mean profile
      d_mean_pred_true, d_mean_pred_ctrl, d_true_ctrl   Euclidean distances between means: generated to measured
                         treated, generated to matched vehicle, measured treated to matched vehicle (the response)
- Ratios are ratios of the two means over conditions, as plot_fig1c.R prints them in the panels of Fig. 1c.
- Checks, all before anything is written (a failure stops the script):
    1. The record is the analysis the text describes: split 1, seed 0, the two arms, 2,899 conditions each.
    2. Within the record: the population distance equals 2 xy - xx - yy (1e-9); the measured-cell quantities are
       identical for the two arms (both models were compared with the same measured cells); the smallest generated
       value is zero.
    3. The population distance equals the scored value of seed 0 of each model on split 1 in the Tahoe-100M
       analysis record (1e-9), which ties these tables to Supplementary Table S12b.
    4. The per-condition table reproduces every mean and median of the record (relative 1e-9), has 2,899 rows per
       arm, no population above 200 cells and no generated value below zero. This is the file Fig. 1c reads.
    5. The values quoted in Results 1 and 2 reproduce at their printed precision (EXPECTED below; checks only,
       never inputs to a computation).
- Rounding: each value is rounded once from full precision; decimals per quantity are set in ROWS_A and ROWS_B.
- Outputs (never overwritten): supplementary_tables/supp_table_S4.csv and .md, supp_table_S7.csv and .md. Nothing is
  written unless all four can be: the outputs of both tables must be absent at the start, and if the second table
  cannot be written the first is removed again.

Run from the research worktree root (supp_table_common.py must be in the same folder):
    python3 make_supp_table_s4_s7.py
"""
import math
import os
import statistics
import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import (DASH, close, die, guard, load_csv, load_json, num, out_paths, same, table,
                               write_outputs)

IN = dict(
    cells=('tahoe_results/step15_diag_cells_drug_split_1_seed_0.json',
           'ec835b903935acd75fdae417690576de64f1c6bda7dd7abd8f3f0ba8222f88a1'),
    percond=('tahoe_results/step15_diag_cells_drug_split_1_seed_0_per_condition.csv',
             '4810be4a928db6388c5fe38358cb5fd5ba0f0c2c4790b7a8fc494ce70275e445'),
    tahoe=('tahoe_results/step15_model_analysis.json',
           '5fd700b5a030cf5a8ffea57e9a9b2ad840aa32fa9948b32754b9742362a1165e'),
)
SPLIT, SEED, N_COND, CAP = 'drug_split_1', 0, 2899, 200
TOL = 1e-9
KEYS = ('xy', 'xx', 'yy', 'energy', 'lib_pred', 'lib_true', 'zero_pred', 'zero_true', 'min_pred', 'profile_r',
        'slope_pred_on_true', 'd_mean_pred_true', 'd_mean_pred_ctrl', 'd_true_ctrl')
MEASURED_KEYS = ('yy', 'lib_true', 'zero_true', 'd_true_ctrl')      # do not depend on the model
# (table, arm in the record, name in the paper)
TABLES = [('S4', 'base', 'original model'), ('S7', 'recipe', 'rewired model')]

# part a: (label, key of the generated cells, key of the measured cells, factor, decimals, ratio shown?)
ROWS_A = [
    ('Spread (mean distance between two cells of the population)', 'xx', 'yy', 1.0, 2, True),
    ('Total expression per cell', 'lib_pred', 'lib_true', 1.0, 1, True),
    ('Zero entries (%)', 'zero_pred', 'zero_true', 100.0, 1, False),
]
# part b: (label, key, decimals)
ROWS_B = [
    ('Mean distance between a generated and a measured cell', 'xy', 2),
    ('Population distance', 'energy', 3),
    ('Correlation between the generated and the measured mean profile', 'profile_r', 3),
    ('Slope of the generated on the measured mean profile', 'slope_pred_on_true', 3),
    ('Distance from the generated mean to the measured treated mean', 'd_mean_pred_true', 2),
    ('Distance from the generated mean to the vehicle mean', 'd_mean_pred_ctrl', 2),
    ('Distance from the measured treated mean to the vehicle mean (the measured response)', 'd_true_ctrl', 2),
]
L_RATIO_B = 'Generated-to-treated distance ÷ measured response'

# Printed values the manuscript quotes (checks only, never inputs).
EXPECTED_S4 = dict(spread_ratio='2.5',        # Results 1: "2.5 times more dispersed than the measured cells"
                   lib_ratio='1.7',           # "carried 1.7 times their total expression"
                   zero_pred_pct='27',        # "were 27 % zeros
                   zero_true_pct='89',        #  where the data were 89 % zeros"
                   shift_ratio='3.3')         # "at 3.3 times the length of the measured response"
EXPECTED_S7 = dict(profile_r='0.97',          # Results 2: "correlated with the measured mean at 0.97"
                   d_pred_true='1.67',        # "it lay 1.67 from the measured treated mean,
                   d_true_ctrl='1.37')        #  against 1.37 for the vehicle mean"


def check_per_condition(path, summary):
    """Check 4: the per-condition table (one row per arm and condition) reproduces the record.
    Returns the number of means and medians compared. NaN cells are skipped, as the record's nanmean/nanmedian do."""
    rows = load_csv(path)
    n = 0
    for _, arm, _ in TABLES:
        mine = [r for r in rows if r['arm'] == arm]
        same(len(mine), summary[arm]['n_conditions'], 'check 4, rows of %s in the per-condition table' % arm)
        for key in ('n_pred', 'n_true'):
            biggest = max(int(float(r[key])) for r in mine)
            if biggest > CAP:
                die('check 4: %s reaches %d cells for %s; the caption says at most %d' % (key, biggest, arm, CAP))
        if min(float(r['min_pred']) for r in mine) < 0:
            die('check 4: a generated value of %s is below zero; the caption says none is' % arm)
        for key in KEYS:
            vals = [float(r[key]) for r in mine]
            vals = [v for v in vals if v == v]
            for stat, got in (('mean', math.fsum(vals) / len(vals)), ('median', statistics.median(vals))):
                want = summary[arm][stat][key]
                close(got, want, TOL * max(1.0, abs(want)), 'check 4, %s of %s for %s' % (stat, key, arm))
                n += 1
    return n


def build(table_id, arm, model, s):
    """Markdown lines and long-format rows of one table, from the record's summary of one arm."""
    mean, med = s['mean'], s['median']
    long_rows = []

    def keep(part, quantity, population, statistic, v, key):
        long_rows.append(dict(table=table_id, model=model, part=part, quantity=quantity, population=population,
                              statistic=statistic, value=v,
                              source_record='step15_diag_cells_drug_split_1_seed_0.json', source_key=key))

    def pair(key, f, dp):
        return '%s (%s)' % (num(f * mean[key], dp), num(f * med[key], dp))

    if table_id == 'S4':
        md = ['Supplementary Table S4 | What the population distance of the original model is made of, on '
              'Tahoe-100M (Results 1; Fig. 1c). For each of the %s held-out conditions of split 1, the cells '
              'generated by the original model (seed 0; at most %d cells) were compared with the measured treated '
              'cells of the same condition (at most %d). a, Cells: mean over conditions, with the median in '
              'parentheses. Spread is the mean Euclidean distance between two cells of the same population; total '
              'expression is the sum of the 200 log-normalised expression values of a cell; zero entries are exact '
              'zeros (no generated value was below zero). The last column is the ratio of the two means, as printed '
              'in Fig. 1c. b, Distances and mean profiles. The population distance is the energy distance: twice '
              'the mean distance between a generated and a measured cell, minus the two spreads. The mean profile '
              'of a population is its mean over cells. Its correlation with the measured mean profile is the '
              'Pearson correlation across the 200 genes, and the slope is that of the least-squares line of the '
              'generated on the measured mean profile (1 means the same scale). Distances between means are '
              'Euclidean; the distance from the measured treated mean to the mean of the matched vehicle cells is '
              'the size of the measured response. The analysis was exploratory.' % ('{:,}'.format(N_COND), CAP, CAP)]
    else:
        md = ['Supplementary Table S7 | What the population distance of the rewired model is made of, on '
              'Tahoe-100M (Results 2). The analysis of Supplementary Table S4 applied to the rewired model, on the '
              'same %s held-out conditions of split 1 (seed 0) and the same measured cells; quantities, statistics '
              'and layout are those of that table. The analysis was exploratory.' % '{:,}'.format(N_COND)]

    md += ['', '**a** | Cells: mean (median) over conditions.', '']
    rows = []
    for label, kg, km, f, dp, with_ratio in ROWS_A:
        ratio = mean[kg] / mean[km]
        rows.append([label, pair(kg, f, dp), pair(km, f, dp), num(ratio, 2) if with_ratio else DASH])
        for pop, key in (('generated', kg), ('measured', km)):
            keep('a', label, pop, 'mean', f * mean[key], 'summary/%s/mean/%s' % (arm, key))
            keep('a', label, pop, 'median', f * med[key], 'summary/%s/median/%s' % (arm, key))
        if with_ratio:
            keep('a', label, 'generated ÷ measured', 'ratio of means', ratio,
                 'summary/%s/mean/%s divided by summary/%s/mean/%s' % (arm, kg, arm, km))
    md += table(['Quantity', 'Generated cells', 'Measured cells', 'Generated ÷ measured'], rows)

    md += ['', '**b** | Distances and mean profiles.', '']
    rows = []
    for label, key, dp in ROWS_B:
        rows.append([label, num(mean[key], dp), num(med[key], dp)])
        keep('b', label, 'both', 'mean', mean[key], 'summary/%s/mean/%s' % (arm, key))
        keep('b', label, 'both', 'median', med[key], 'summary/%s/median/%s' % (arm, key))
    shift = mean['d_mean_pred_true'] / mean['d_true_ctrl']
    rows.append([L_RATIO_B + ' (ratio of the means)', num(shift, 2), DASH])
    keep('b', L_RATIO_B, 'both', 'ratio of means', shift,
         'summary/%s/mean/d_mean_pred_true divided by summary/%s/mean/d_true_ctrl' % (arm, arm))
    md += table(['Quantity', 'Mean over conditions', 'Median over conditions'], rows)

    md += ['',
           'Sources: step15_diag_cells_drug_split_1_seed_0.json (sha256 %s; diag_step15_cells.py). The table of '
           'the same run with one row per condition, from which Fig. 1c is drawn '
           '(step15_diag_cells_drug_split_1_seed_0_per_condition.csv, sha256 %s), reproduces every mean and median '
           'given here. The population distance is the scored value of this seed in step15_model_analysis.json '
           '(sha256 %s; Supplementary Table S12b). Assembled by make_supp_table_s4_s7.py.'
           % (IN['cells'][1][:16], IN['percond'][1][:16], IN['tahoe'][1][:16])]
    return long_rows, md


def main():
    for table_id, _, _ in TABLES:                        # inputs committed and unchanged; none of the four outputs exists
        guard(IN, table_id)
    rec, tah = load_json(IN['cells'][0]), load_json(IN['tahoe'][0])
    summary = rec['summary']

    # ---- check 1: the analysis the text describes --------------------------------------------------------------------
    same((rec['split'], rec['seed'], rec['arms'], rec['max_conditions']), (SPLIT, SEED, ['base', 'recipe'], 0),
         'check 1, split, seed, arms and condition limit of the record')
    for _, arm, _ in TABLES:
        same(summary[arm]['n_conditions'], N_COND, 'check 1, conditions of %s' % arm)
        same(summary[arm]['tag'], 's15_%s_%s_seed_%d' % (SPLIT, arm, SEED), 'check 1, run label of %s' % arm)
        same(sorted(summary[arm]['mean']), sorted(KEYS), 'check 1, quantities of %s' % arm)
    same(tah['per_split'][SPLIT]['n_conditions'], N_COND, 'check 1, conditions of split 1 in the Tahoe-100M record')

    # ---- check 2: within the record ----------------------------------------------------------------------------------
    n2 = 0
    for _, arm, _ in TABLES:
        m = summary[arm]['mean']
        close(m['energy'], 2 * m['xy'] - m['xx'] - m['yy'], TOL, 'check 2, energy identity of %s' % arm)
        same((m['min_pred'], summary[arm]['median']['min_pred']), (0.0, 0.0), 'check 2, smallest generated value')
        n2 += 2
    for stat in ('mean', 'median'):
        for key in MEASURED_KEYS:
            same(summary['base'][stat][key], summary['recipe'][stat][key],
                 'check 2, %s of %s is the same for both models' % (stat, key))
            n2 += 1

    # ---- check 3: the scored population distance of this seed ---------------------------------------------------------
    for _, arm, _ in TABLES:
        scored = tah['per_split'][SPLIT]['metrics'][arm]['per_seed'][SEED]['energy']
        same(tah['inputs'][SPLIT][arm][SEED]['tag'], summary[arm]['tag'], 'check 3, the scored run of %s' % arm)
        close(summary[arm]['mean']['energy'], scored, TOL, 'check 3, scored population distance of %s' % arm)

    # ---- check 4: the per-condition table that Fig. 1c reads ------------------------------------------------------------
    n4 = check_per_condition(IN['percond'][0], summary)

    # ---- check 5: the values the manuscript quotes ----------------------------------------------------------------------
    b, r = summary['base']['mean'], summary['recipe']['mean']
    same(dict(spread_ratio='%.1f' % (b['xx'] / b['yy']), lib_ratio='%.1f' % (b['lib_pred'] / b['lib_true']),
              zero_pred_pct='%.0f' % (100 * b['zero_pred']), zero_true_pct='%.0f' % (100 * b['zero_true']),
              shift_ratio='%.1f' % (b['d_mean_pred_true'] / b['d_true_ctrl'])), EXPECTED_S4,
         'check 5, values quoted in Results 1')
    # "as far from the treated mean as from the vehicle mean": the two distances agree at one decimal
    same('%.1f' % b['d_mean_pred_true'], '%.1f' % b['d_mean_pred_ctrl'],
         'check 5, Results 1: generated mean as far from the treated mean as from the vehicle mean')
    same(dict(profile_r='%.2f' % r['profile_r'], d_pred_true='%.2f' % r['d_mean_pred_true'],
              d_true_ctrl='%.2f' % r['d_true_ctrl']), EXPECTED_S7, 'check 5, values quoted in Results 2')
    # "their spread and library size matched the measured cells": both ratios are 1.0 at one decimal
    same(('%.1f' % (r['xx'] / r['yy']), '%.1f' % (r['lib_pred'] / r['lib_true'])), ('1.0', '1.0'),
         'check 5, Results 2: spread and library size of the rewired model match the measured cells')

    # ---- the two tables: all four files or none --------------------------------------------------------------------------
    built = [(t, build(t, arm, model, summary[arm])) for t, arm, model in TABLES]
    done = []
    for table_id, (long_rows, md) in built:
        try:
            write_outputs(table_id, long_rows, md)
        except SystemExit:
            for t in done:                               # the earlier table goes too: both tables or neither
                for p in out_paths(t):
                    if os.path.exists(p):
                        os.remove(p)
            raise
        done.append(table_id)
        print('')
    print('[checks] 1 (the record is split 1, seed 0, %s conditions per model), 2 (%d identities within the record), '
          '3 (scored population distance of both models), 4 (%d means and medians of the per-condition table), '
          '5 (values quoted in Results 1 and 2) all passed' % ('{:,}'.format(N_COND), n2, n4))


if __name__ == '__main__':
    main()
