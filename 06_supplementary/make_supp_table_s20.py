#!/usr/bin/env python3
"""Supplementary Table S20: sensitivity of the comparisons with the cross-validated ridge to the fold assignment.

Purpose
-------
Step 29 (PREREG_step29_fold_sensitivity.md) refitted the cross-validated ridge with the fold assignment of the paper
and with 200 random assignments, and counted, for every statement of the paper that compares something with this
ridge, how many random assignments give the verdict the paper states. This script assembles the supplementary table
of that analysis from the two committed records. It computes nothing new: every value is read from a record.

Logic
-----
1. Guard (supp_table_common.guard): both records are tracked, identical to HEAD and have the sha256 recorded here;
   neither output exists (outputs are never overwritten).
2. Checks before anything is written:
   a. every item and split of the list ITEMS below is in its record, and the record holds no further item;
   b. for every item and split, "robust" in the record equals (count >= needed), needed is the smallest whole number
      that is at least 95 % of the assignments, and "robust" of the item equals robust on all its splits;
   c. the value with the folds of the paper in the summary of a quantity equals the value in the block of the folds
      of the paper (the two places of a record that hold it);
   d. with the folds of the paper, every Tahoe-100M item gives the stated verdict (recorded by the scorer), and the
      sci-Plex scorer stops unless that holds (its guard), so the column "folds of the paper" always shows a value
      that carries the verdict the paper states.
3. Parts a (sci-Plex) and b (Tahoe-100M): one row per statement and split, with the value obtained with the folds of
   the paper, the median and the range over the random assignments, and the number of assignments that give the
   verdict of the paper. Only point values are shown; the intervals of the paper are in the tables the statements
   come from (the records of this analysis hold intervals recomputed from a fresh random generator, which differ
   from the published ones in the third or fourth decimal).
4. Part c: the values of the ridge itself, with the folds of the paper and over the random assignments.

Outputs:  supplementary_tables/supp_table_S20.md and supp_table_S20.csv (long format, full precision)
Run:      python3 make_supp_table_s20.py     (from the research worktree root; standard library only)
"""
import math

from supp_table_common import SPLITS, die, guard, load_json, num, same, table, write_outputs

TABLE = 'S20'
IN = dict(
    sciplex=('leakage_results/step29_fold_sensitivity_sciplex.json',
             '73bd35c74988db00d71b77b795b5c3f035441c0ff3577e6a5fa07d96700a36ad'),
    tahoe=('tahoe_results/step29_fold_sensitivity_tahoe.json',
           'a88bbfd8221bca59a76ab6da9366d0428257bc9af6cadd5945c5190f77a7b456'),
)
SPLIT_NO = {'drug_split_1': '1', 'drug_split_0': '0', 'drug_split_2': '2'}
THRESHOLD = 0.95

# item, the quantity whose value is shown, the statement of the paper (what is counted)
ITEMS = dict(
    sciplex=[
        ('A1', 'A1_rewired_minus_ridge_gene_identity',
         'Gene identity, rewired model minus ridge: positive, interval above zero'),
        ('A2', 'A2_rewired_minus_ridge_gene_identity_without_twin',
         'The same without the compound that has a salt twin in training: positive, interval above zero'),
        ('A3', 'A3_original_minus_ridge_gene_identity',
         'Gene identity, original model minus ridge: negative'),
        ('A4', 'A4_highest_model_minus_ridge_mean',
         'Response correlation (mean over conditions), highest generative model minus ridge: negative'),
        ('A5', 'A5_highest_model_minus_ridge_median',
         'Response correlation (median over conditions), highest generative model minus ridge: negative on '
         'Splits 1 and 0; on Split 2, cross-attention and its control both above the ridge'),
        ('A6', 'ridge_response_magnitude',
         'Response-magnitude agreement of the ridge: positive; interval above zero on Split 1, includes zero on '
         'Splits 0 and 2'),
        ('C4', 'C4_rewired_minus_ridge_response_magnitude',
         'Response-magnitude agreement, rewired model minus ridge: interval includes zero'),
        ('A7', 'A7_network_minus_ridge',
         'Response correlation, feedforward network minus ridge: negative on Split 0 (50 assignments)'),
    ],
    tahoe=[
        ('B1', 'B1_blind_minus_ridge_gene_identity',
         'Gene identity, compound-blind average minus ridge: positive'),
        ('B2', 'B2_original_minus_ridge_gene_identity',
         'Gene identity, original model minus ridge: negative'),
        ('C1', 'C1_rewired_minus_ridge_gene_identity',
         'Gene identity, rewired model minus ridge: interval includes zero'),
        ('C2', 'C2_rewired_minus_ridge_population_distance',
         'Population distance, rewired model minus ridge: positive; interval above zero on Splits 1 and 2, '
         'includes zero on Split 0'),
        ('B3', 'B3_ridge_discrimination_minus_half',
         'Discrimination score of the ridge minus 0.5: interval includes zero'),
        ('C3', 'C3_rewired_minus_ridge_discrimination',
         'Discrimination score, rewired model minus ridge: positive, interval above zero'),
        ('B4', 'ridge_response_magnitude',
         'Response-magnitude agreement of the ridge: interval includes zero'),
    ],
)
# part c: the ridge itself
RIDGE = dict(
    sciplex=[('ridge_gene_identity', 'Gene identity'),
             ('ridge_response_correlation_mean', 'Response correlation, mean over conditions'),
             ('ridge_response_correlation_median', 'Response correlation, median over conditions'),
             ('ridge_response_magnitude', 'Response-magnitude agreement')],
    tahoe=[('ridge_gene_identity', 'Gene identity'),
           ('ridge_response_correlation_mean', 'Response correlation, mean over conditions'),
           ('ridge_response_correlation_median', 'Response correlation, median over conditions'),
           ('ridge_population_distance', 'Population distance'),
           ('ridge_discrimination_score', 'Discrimination score'),
           ('ridge_response_magnitude', 'Response-magnitude agreement')],
)
NAME = dict(sciplex='sci-Plex', tahoe='Tahoe-100M')


def summaries(rec, ds, sp):
    """The summaries of the quantities of one split: {quantity: {paper, median, p2.5, p97.5, min, max, ...}}."""
    return rec['per_split'][sp]['quantities' if ds == 'sciplex' else 'ranges']


def paper_block_value(rec, ds, sp, item, quantity):
    """The value with the folds of the paper, read from the block of the folds of the paper (check c), or None
    where that block does not hold this quantity under a name this script knows."""
    if ds == 'tahoe':
        return rec['per_split'][sp]['folds_of_the_paper'][item]['diff']
    p = rec['per_split'][sp]['paper']
    if item == 'A6':
        return p['A6']['mra']
    if item in ('A1', 'A2', 'A3', 'A4', 'A5', 'C4', 'A7') and item in p and 'diff' in p[item]:
        return p[item]['diff']
    return None


def main():
    guard(IN, TABLE)
    rec = dict(sciplex=load_json(IN['sciplex'][0]), tahoe=load_json(IN['tahoe'][0]))
    long_rows, md = [], []

    def keep(part, ds, item, sp, field, value):
        long_rows.append(dict(table=TABLE, part=part, dataset=NAME[ds], item=item, split=SPLIT_NO.get(sp, sp),
                              field=field, value=value))

    n_counts = 0
    for part, ds in (('a', 'sciplex'), ('b', 'tahoe')):
        r = rec[ds]
        same(sorted(r['items']), sorted(i for i, _, _ in ITEMS[ds]), '%s: the items of the record' % NAME[ds])   # check a
        rows = []
        for item, quantity, statement in ITEMS[ds]:
            it = r['items'][item]
            per = it['per_split']
            if not set(per) <= set(SPLITS):
                die('%s item %s: unknown split' % (NAME[ds], item))
            same(bool(it['robust']), all(bool(per[sp]['robust']) for sp in per), '%s %s: robust overall' % (NAME[ds], item))
            first = True
            for sp in SPLITS:
                if sp not in per:
                    continue
                c = per[sp]
                need = int(math.ceil(THRESHOLD * c['n'] - 1e-9))                                               # check b
                same(c['needed'], need, '%s %s %s: needed' % (NAME[ds], item, sp))
                same(bool(c['robust']), c['n_same'] >= need, '%s %s %s: robust' % (NAME[ds], item, sp))
                s = summaries(r, ds, sp)[quantity]
                same(s['n'], c['n'], '%s %s %s: number of assignments' % (NAME[ds], item, sp))
                pv = paper_block_value(r, ds, sp, item, quantity)                                             # check c
                if pv is not None and abs(pv - s['paper']) > 1e-12:
                    die('%s %s %s: the two places of the record disagree on the value of the paper' % (NAME[ds], item, sp))
                if ds == 'tahoe' and not c['folds_of_the_paper_same_verdict']:                                # check d
                    die('%s %s %s: the folds of the paper do not give the stated verdict' % (NAME[ds], item, sp))
                rows.append([item if first else '', statement if first else '', SPLIT_NO[sp], num(s['paper'], 3, True),
                             num(s['median'], 3, True), '%s to %s' % (num(s['min'], 3, True), num(s['max'], 3, True)),
                             '%d of %d' % (c['n_same'], c['n']), 'yes' if c['robust'] else 'no',
                             ('robust' if it['robust'] else 'not robust') if first else ''])
                for field in ('paper', 'median', 'p2.5', 'p97.5', 'min', 'max'):
                    keep(part, ds, item, sp, 'difference_or_value_' + field, s[field])
                keep(part, ds, item, sp, 'assignments_with_the_verdict_of_the_paper', c['n_same'])
                keep(part, ds, item, sp, 'assignments', c['n'])
                keep(part, ds, item, sp, 'needed', c['needed'])
                keep(part, ds, item, sp, 'robust_on_split', bool(c['robust']))
                keep(part, ds, item, sp, 'robust_overall', bool(it['robust']))
                first = False
                n_counts += 1
        md += ['', '**%s** | %s.' % (part, NAME[ds]), '']
        md += table(['Item', 'Statement of the paper', 'Split', 'Folds of the paper', 'Median', 'Range',
                     'Assignments with the verdict of the paper', 'At least 95 %', 'Statement'], rows)

    rows = []
    for ds in ('sciplex', 'tahoe'):
        for quantity, label in RIDGE[ds]:
            first = True
            for sp in SPLITS:
                s = summaries(rec[ds], ds, sp)[quantity]
                rows.append([NAME[ds] if first else '', label if first else '', SPLIT_NO[sp], num(s['paper'], 3),
                             num(s['median'], 3), '%s to %s' % (num(s['p2.5'], 3), num(s['p97.5'], 3)),
                             '%s to %s' % (num(s['min'], 3), num(s['max'], 3)),
                             '%d of %d' % (s['n_random_not_above_paper'], s['n'])])
                for field in ('paper', 'median', 'p2.5', 'p97.5', 'min', 'max', 'n_random_not_above_paper', 'n'):
                    keep('c', ds, quantity, sp, field, s[field])
                first = False
    md += ['', '**c** | The cross-validated ridge itself.', '']
    md += table(['Dataset', 'Measure', 'Split', 'Folds of the paper', 'Median', '2.5th to 97.5th percentile', 'Range',
                 'Assignments at or below the paper'], rows)

    st, tt = rec['sciplex']['settings'], rec['tahoe']['settings']
    same(st['random_assignments'], tt['n_random'], 'number of random assignments of the two records')
    same(st['threshold'], THRESHOLD, 'threshold of the sci-Plex record')
    same(tt['threshold'], THRESHOLD, 'threshold of the Tahoe record')
    md += ['',
           'Sources: step29_fold_sensitivity_sciplex.json (sha256 %s; parts a and c) and '
           'step29_fold_sensitivity_tahoe.json (sha256 %s; parts b and c); %d random assignments per split, %d for the '
           'feedforward network. The .csv file holds every value at full precision, with the 2.5th and 97.5th '
           'percentiles of parts a and b. Assembled by make_supp_table_s20.py.'
           % (IN['sciplex'][1][:16], IN['tahoe'][1][:16], st['random_assignments'], st['network_assignments'])]
    write_outputs(TABLE, long_rows, md)
    print('[checks] a (items), b (counts, thresholds and verdicts of %d item-split pairs), c (value of the paper in two '
          'places), d (stated verdict with the folds of the paper) all passed' % n_counts)


if __name__ == '__main__':
    main()
