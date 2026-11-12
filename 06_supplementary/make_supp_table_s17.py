#!/usr/bin/env python3
"""
make_supp_table_s17.py - Supplementary Table S17: spread agreement on Tahoe-100M and its decomposition
(Supplementary Note 5; Supplementary Fig. S1).

Purpose
-------
Supplementary Note 5 asks whether a model predicts how a compound changes the SPREAD of a cell population (does the
compound make the cells more alike or more varied?). It cites this table for every number, and the caption of
Supplementary Fig. S1 cites it for the values of the figure and for the decomposition. The numbers come from two
committed records: the analysis specified before the data were scored, and an exploratory decomposition that was
added after its first result was seen. This script assembles them in one table, so that each number traces to a
record and is rounded once, and it keeps the two kinds of analysis in separate parts.

The measure (as defined before the analysis):
    spread of a population   S = mean Euclidean distance of its cells from their mean (200 genes, at most 200 cells)
    spread ratio             R = S(treated cells) / S(matched vehicle cells given to the model for that condition)
                             R > 1: the compound made the population more varied; R < 1: more uniform
    spread agreement         within a context (one cell line at one dose) holding at least ten held-out conditions,
                             the Spearman correlation over conditions between the predicted and the measured R;
                             then the mean over contexts and, for a model, over its three seeds. Zero = no agreement.

Four parts:
  a  spread agreement of every entry against zero, the plate replicate, and the difference between the two models
     (specified before scoring; the values of Supplementary Fig. S1);
  b  how large the quantities are: the measured ratio, the generated spread relative to the measured spread, and how
     much the ratio predicted by a linear reference varies (specified before scoring; descriptive);
  c  the decomposition: agreement with the vehicle spread removed, with the compound-blind prediction removed, and
     agreement on the spread itself (exploratory);
  d  how strongly each quantity follows the vehicle spread (exploratory).

Why the decomposition matters for reading part a: the predicted and the measured ratio are divided by the SAME
vehicle spread. Two ratios that share a noisy denominator correlate even when their numerators are unrelated (the
spurious correlation of ratios). Part c removes that shared denominator and shows what is left.

Logic
-----
- Inputs (committed records only; see supp_table_common.guard):
      tahoe_results/step19_heterogeneity.json        parts a, b: per_split/<s>/metrics/<arm>/ha, contrasts,
                                                     r_true, clip_variation, replicate
      tahoe_results/step19_diag_control_route.json   parts c, d: per_split/<s>/contrasts/<measure>|<arm>
      figures/squidiff_publication_figures/fig5_values.csv   the values Supplementary Fig. S1 was drawn from
                                                             (panel c of the old Figure 5); check only
  Record names -> names in the paper: recipe = rewired model; ot_pair = rewired model with optimal-transport
  pairing; mean_response = compound-blind average; ridge_train = training-set ridge; ridge_S7 = cross-validated
  ridge; control_paste = unaltered vehicle cells; replicate / ceiling = plate replicate; ha = spread agreement;
  ha_ctrl = agreement with the vehicle spread removed; ha_blind = agreement with the compound-blind prediction
  removed; sa_ctrl = agreement on the spread itself with the vehicle spread removed; coupling = correlation with
  the vehicle spread; R_true / S_true = measured ratio / measured treated spread.
- Checks, all before anything is written (a failure stops the script):
    1. First record: settings; 42 contexts and 14 for the replicate; each value against zero equals the stored
       measure (1e-9); each model value is the mean of its three seeds; the unaltered vehicle cells score exactly
       zero and their predicted ratio does not vary; the difference between the two models equals the difference
       of their two values; the outcome stored in the record follows from the intervals.
    2. Second record: settings; it was computed from the first record (same conditions and contexts, same stored
       agreement, and for the three linear references the same value and the same interval to 1e-9, which shows
       that the same bootstrap resamples were used); it holds exactly the 26 quantities of parts c and d; each
       stored point value equals its interval's value; the seed-averaged agreement of a model stays within 0.01 of
       part a; the three rules stored in the record follow from the intervals.
    3. Every value of the figure (18 rows of panel c: value, lower, upper, and whether the interval excludes zero)
       equals the value given here (1e-9).
    4. The statements of Supplementary Note 5 hold: the ranges it quotes at their printed precision, that the
       agreement of both models is not distinguishable from zero once the vehicle spread is removed, and that the
       agreement of the compound-blind average remains (EXPECTED below; checks only, never inputs to a computation).
- Rounding: three decimals (one for the percentage), each value rounded once from full precision; values against
  zero carry their sign.
- Outputs (never overwritten): supplementary_tables/supp_table_S17.csv and .md (see supp_table_common).

Run from the research worktree root (supp_table_common.py must be in the same folder):
    python3 make_supp_table_s17.py
"""
import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import (SPLITS, SPLIT_HEAD, close, die, guard, interval, load_csv, load_json, num, same, table,
                               write_outputs)

TABLE = 'S17'
IN = dict(
    het=('tahoe_results/step19_heterogeneity.json',
         'de4dd1396cca78bba97dfbedf555c2a4dee9ad81412b811bb13f65be4e3310da'),
    dec=('tahoe_results/step19_diag_control_route.json',
         '6112c94c13cb14ba54924d66766b695173026c8d9d9768859a1501a765b81eaa'),
    fig5=('figures/squidiff_publication_figures/fig5_values.csv',
          '95c5f189b195dc4d273b00aeacf1ba3b85ec64da7e4bf01a3bd70c323b1f9ca3'),
)
TOL = 1e-9

L_REW = 'Rewired model'
L_OT = 'Rewired model with optimal-transport pairing'
L_BLIND = 'Compound-blind average'
L_RT, L_RCV = 'Training-set ridge', 'Cross-validated ridge'
L_REP = 'Plate replicate (14 contexts)'
L_DIFF = 'Rewired model with optimal-transport pairing − rewired model'
# part a: (label, arm in the record, name of its comparison with zero, number of seeds)
A_ARMS = [(L_REW, 'recipe', 'secondary_ha_recipe_minus_0', 3),
          (L_OT, 'ot_pair', 'Q1_primary_ha_ot_pair_minus_0', 3),
          (L_BLIND, 'mean_response', 'Q3_gate_ha_mean_response_minus_0', 1),
          (L_RT, 'ridge_train', 'Q3_gate_ha_ridge_train_minus_0', 1),
          (L_RCV, 'ridge_S7', 'Q3_gate_ha_ridge_S7_minus_0', 1)]
MODELS = ('recipe', 'ot_pair')                            # the two generative models
LINEAR = ('mean_response', 'ridge_train', 'ridge_S7')     # one shift added to the vehicle cells, clipped at zero
PASTE = 'control_paste'                                   # the unaltered vehicle cells: ratio 1 in every condition
N_REP, N_DIFF = 'secondary_ha_ceiling_minus_0', 'Q2_coprimary_ha_ot_pair_minus_recipe'
# part c: the measures of the decomposition, in the order of the table: (key in the record, label)
C_MEASURES = [('ha', 'Agreement'),
              ('ha_ctrl', 'Agreement, vehicle spread removed'),
              ('ha_blind', 'Agreement, compound-blind prediction removed'),
              ('sa_ctrl', 'Agreement on spread itself, vehicle spread removed')]
# part d: (label, key in the record)
D_ROWS = [('Measured treated spread', 'coupling|S_true'),
          ('Measured ratio', 'coupling|R_true'),
          ('Predicted ratio, rewired model', 'coupling|recipe'),
          ('Predicted ratio, rewired model with optimal-transport pairing', 'coupling|ot_pair'),
          ('Predicted ratio, compound-blind average', 'coupling|mean_response'),
          ('Predicted ratio, training-set ridge', 'coupling|ridge_train'),
          ('Predicted ratio, cross-validated ridge', 'coupling|ridge_S7')]
NOT_SEPARABLE = 'not separable from the control route on this measure'      # the wording stored in the record

# Printed values Supplementary Note 5 quotes (checks only, never inputs): smallest and largest over the splits.
EXPECTED = dict(
    models=('0.16', '0.19'),            # "the rewired model and its optimal-transport variant agreed ... (0.16-0.19)"
    blind=('0.11', '0.13'),             # "but so did the compound-blind average (0.11-0.13)"
    models_removed=('-0.05', '0.07'),   # "their agreement was not distinguishable from zero (-0.05 to +0.07)"
    blind_removed=('0.12', '0.18'),     # "That of the compound-blind average remained (0.12-0.18)"
)


def tri(c):
    """(value, lower, upper) of one stored comparison."""
    return float(c['diff']), float(c['ci'][0]), float(c['ci'][1])


def above_zero(c):
    return c[1] > 0


def includes_zero(c):
    return c[1] <= 0 <= c[2]


def word(c):
    """The record's own word for an interval."""
    return 'above zero' if c[1] > 0 else ('below zero' if c[2] < 0 else 'includes zero')


def span(values):
    """Smallest and largest value at two decimals, as the text prints ranges."""
    return ('%.2f' % min(values), '%.2f' % max(values))


def mean(values):
    return sum(values) / len(values)


def main():
    guard(IN, TABLE)
    het, dec = load_json(IN['het'][0]), load_json(IN['dec'][0])
    H = dict((s, het['per_split'][s]) for s in SPLITS)
    G = dict((s, dec['per_split'][s]) for s in SPLITS)

    # ---- check 1: the record of the analysis specified before scoring ------------------------------------------------
    same(het['splits'], SPLITS, 'check 1, split order of the first record')
    same((het['prereg'], het['final'], het['interval'], het['n_boot'], het['n_seeds'], het['min_context']),
         ('PREREG_step19_heterogeneity.md', True, '97.5% structure-clustered percentile bootstrap', 5000, 3, 10),
         'check 1, settings of the first record')
    n1 = 0
    A = {}                                                # part a: A[split][arm] = (value, lower, upper)
    for s in SPLITS:
        same((H[s]['n_contexts'], H[s]['min_context'], H[s]['replicate']['ha_n_contexts']), (42, 10, 14),
             'check 1, contexts on %s' % s)
        A[s] = {}
        for label, arm, name, n_seeds in A_ARMS:
            c = tri(H[s]['contrasts'][name])
            close(c[0], H[s]['metrics'][arm]['ha'], TOL, 'check 1, %s on %s' % (arm, s))
            per_seed = H[s]['metrics'][arm]['ha_per_seed']
            same(len(per_seed), n_seeds, 'check 1, seeds of %s on %s' % (arm, s))
            close(c[0], mean(per_seed), TOL, 'check 1, seed mean of %s on %s' % (arm, s))
            A[s][arm] = c
            n1 += 2
        # the unaltered vehicle cells: predicted ratio 1 in every condition, so no agreement and no variation
        same((H[s]['metrics'][PASTE]['ha'], H[s]['clip_variation'][PASTE]), (0.0, 0.0),
             'check 1, the unaltered vehicle cells score zero and do not vary on %s' % s)
        c = tri(H[s]['contrasts'][N_REP])
        close(c[0], H[s]['replicate']['ha'], TOL, 'check 1, plate replicate on %s' % s)
        A[s]['replicate'] = c
        c = tri(H[s]['contrasts'][N_DIFF])
        close(c[0], A[s]['ot_pair'][0] - A[s]['recipe'][0], TOL, 'check 1, difference between the two models on %s' % s)
        A[s]['diff'] = c
        n1 += 3
    # The outcome stored in the record is a category fixed before scoring: "2 SPREAD PREDICTED" means that the model
    # with optimal-transport pairing is above zero on every split and not above the rewired model on every split.
    # It is a category label, not a conclusion (Supplementary Note 5 explains why); here it is only checked against
    # the intervals it was derived from.
    same(het['verdict'], '2 SPREAD PREDICTED', 'check 1, outcome stored in the first record')
    if not all(above_zero(A[s]['ot_pair']) for s in SPLITS) or all(above_zero(A[s]['diff']) for s in SPLITS):
        die('check 1: the outcome stored in the first record does not follow from its intervals')
    conds = [H[s]['n_conditions'] for s in SPLITS]
    structs = [H[s]['n_structures'] for s in SPLITS]
    pairs = [H[s]['replicate']['n_pairs'] for s in SPLITS]
    same(len(set(structs)), 1, 'check 1, the three splits hold the same number of held-out structures')

    # ---- check 2: the record of the exploratory decomposition ---------------------------------------------------------
    same((dec['script'], dec['exploratory'], dec['final'], dec['splits'], dec['n_boot'], dec['min_context']),
         ('diag_step19_control_route.py', True, True, SPLITS, 5000, 10), 'check 2, settings of the second record')
    same(dec['interval'], "97.5% structure-clustered percentile bootstrap (the Step 19 scorer's resamples)",
         'check 2, interval of the second record')
    same(dec['inputs']['record'], IN['het'][0], 'check 2, the second record was computed from the first')
    arms5 = [arm for _, arm, _, _ in A_ARMS]
    keys = set(k for _, k in D_ROWS)
    for arm in arms5:
        for measure, _ in C_MEASURES:
            if not (measure == 'ha_blind' and arm == 'mean_response'):   # an entry cannot be removed from itself
                keys.add('%s|%s' % (measure, arm))
    n2, seed_gap = 0, 0.0
    C = {}                                                # parts c and d: C[split][key] = (value, lower, upper)
    for s in SPLITS:
        same((G[s]['n_conditions'], G[s]['n_contexts']), (H[s]['n_conditions'], H[s]['n_contexts']),
             'check 2, conditions and contexts of the two records on %s' % s)
        same((set(G[s]['contrasts']), set(G[s]['point'])), (keys, keys), 'check 2, quantities held for %s' % s)
        C[s] = {}
        for k in sorted(keys):
            c = tri(G[s]['contrasts'][k])
            close(c[0], float(G[s]['point'][k]), TOL, 'check 2, %s on %s' % (k, s))
            same(G[s]['contrasts'][k]['verdict'], word(c), 'check 2, stored word for the interval of %s on %s' % (k, s))
            C[s][k] = c
            n2 += 1
        for arm in arms5:                                 # the agreement the second record copied from the first
            close(float(G[s]['ha_record'][arm]), A[s][arm][0], TOL, 'check 2, %s in the two records on %s' % (arm, s))
            n2 += 1
        for arm in LINEAR:                                # one prediction each: value and interval must be identical
            for i, what in enumerate(('value', 'lower limit', 'upper limit')):
                close(C[s]['ha|%s' % arm][i], A[s][arm][i], TOL,
                      'check 2, %s of %s in the two records on %s (same resamples)' % (what, arm, s))
            n2 += 1
        for arm in MODELS:                                # seed-averaged prediction against the mean over seeds
            gap = abs(C[s]['ha|%s' % arm][0] - A[s][arm][0])
            if gap > 0.01:
                die('check 2: the seed-averaged agreement of %s on %s is %.4f away from part a' % (arm, s, gap))
            seed_gap = max(seed_gap, gap)
        # the three rules the record applied, recomputed here from its intervals
        r1 = not includes_zero(C[s]['coupling|R_true'])                  # the measured ratio follows its denominator
        r3 = above_zero(C[s]['ha_ctrl|mean_response'])                   # compound-blind: more than the denominator
        r2 = dict((arm, 'beyond the control cells'
                   if all(above_zero(C[s]['%s|%s' % (m, arm)]) for m in ('ha_ctrl', 'ha_blind', 'sa_ctrl'))
                   else NOT_SEPARABLE) for arm in MODELS)
        same((G[s]['rules']['R1_r_true_coupled'], G[s]['rules']['R3_control_beyond_spread'], G[s]['rules']['R2']),
             (r1, r3, r2), 'check 2, rules stored in the second record for %s' % s)
        n2 += 3

    # ---- check 3: the values Supplementary Fig. S1 was drawn from -------------------------------------------------------
    fig_split = {'Split 1': 'drug_split_1', 'Split 0': 'drug_split_0', 'Split 2': 'drug_split_2'}
    fig_row = {'Rewired model': 'recipe', 'Rewired model, OT pairing': 'ot_pair', 'Compound-blind average': 'mean_response',
               'Training-set ridge': 'ridge_train', 'Cross-validated ridge': 'ridge_S7',
               'Plate replicate (14 contexts)': 'replicate'}
    seen = set()
    for r in load_csv(IN['fig5'][0]):
        if r['panel'] != 'c':
            continue
        what = 'check 3, fig5_values.csv panel c: %s, %s' % (r['row'], r['split'])
        same((r['facet'], float(r['chance'])), ('Spread agreement', 0.0), what + ' (facet and reference line)')
        if r['row'] not in fig_row or r['split'] not in fig_split:      # a point of the figure with no row here
            die(what + ': the figure holds a row or a split that this table does not')
        c = A[fig_split[r['split']]][fig_row[r['row']]]
        close(c[0], float(r['point']), TOL, what)
        close(c[1], float(r['lo']), TOL, what + ' (lower)')
        close(c[2], float(r['hi']), TOL, what + ' (upper)')
        same(r['excludes'], 'FALSE' if includes_zero(c) else 'TRUE', what + ' (filled or open point)')
        seen.add((r['row'], r['split']))
    n3 = len(seen)
    same(n3, 18, 'check 3, number of panel-c values in fig5_values.csv')

    # ---- check 4: the statements of Supplementary Note 5 ----------------------------------------------------------------
    got = dict(
        models=span([A[s][arm][0] for s in SPLITS for arm in MODELS]),
        blind=span([A[s]['mean_response'][0] for s in SPLITS]),
        models_removed=span([C[s]['ha_ctrl|%s' % arm][0] for s in SPLITS for arm in MODELS]),
        blind_removed=span([C[s]['ha_ctrl|mean_response'][0] for s in SPLITS]),
    )
    same(got, EXPECTED, 'check 4, ranges quoted in Supplementary Note 5')
    for s in SPLITS:
        for arm in MODELS + ('mean_response',):           # "agreed with the measured spread ratio": above zero as scored
            if not above_zero(A[s][arm]):
                die('check 4: %s is not above zero on %s' % (arm, s))
        for arm in MODELS:                                # "once it was removed, ... not distinguishable from zero"
            if not includes_zero(C[s]['ha_ctrl|%s' % arm]):
                die('check 4: with the vehicle spread removed, %s is still resolved on %s' % (arm, s))
        if not above_zero(C[s]['ha_ctrl|mean_response']):  # the compound-blind average keeps its agreement
            die('check 4: with the vehicle spread removed, the compound-blind average is not above zero on %s' % s)

    # ---- the table ---------------------------------------------------------------------------------------------------
    long_rows = []

    def keep(part, quantity, entry, split, c, status, record, key):
        v, lo, hi = c if isinstance(c, tuple) else (c, None, None)
        long_rows.append(dict(part=part, quantity=quantity, entry=entry, split=split, value=v, lo=lo, hi=hi,
                              excludes_zero='' if lo is None else int(lo > 0 or hi < 0), status=status,
                              source_record=record, source_key=key))

    def cells(get):
        return [interval(*get(s)) for s in SPLITS]

    REC_H, REC_D = 'step19_heterogeneity.json', 'step19_diag_control_route.json'
    PRE, DESC, EXPL = 'specified before scoring', 'specified before scoring (descriptive)', 'exploratory'
    md = ['Supplementary Table S17 | Spread agreement on Tahoe-100M and its decomposition (Supplementary Note 5; '
          'Supplementary Fig. S1). The spread of a population is the mean Euclidean distance of its cells from their '
          'mean, over the 200 genes and at most 200 cells (a different quantity from the mean pairwise distance of '
          'Supplementary Tables S4 and S7). The spread ratio of a condition is the spread of its treated cells '
          'divided by the spread of the matched vehicle cells given to the model for that condition; it was computed '
          'for the measured and for the predicted cells. Spread agreement is the Spearman correlation over conditions '
          'between the predicted and the measured ratio, within each context (one cell line at one dose) holding at '
          'least ten held-out conditions; the measure is its mean over the 42 contexts and, for a model, over three '
          'seeds. Values are given with 97.5 %% percentile bootstrap intervals against zero (5,000 replicates, '
          'resampling held-out chemical structures); an asterisk marks an interval that excludes zero. Splits 1, 0 '
          'and 2 hold %s held-out conditions of %d structures each. a, Spread agreement. '
          'Each linear reference adds one shift to every vehicle cell and sets negative values to zero; the unaltered '
          'vehicle cells have a ratio of 1 in every condition and score zero by construction. For the plate '
          'replicate, the ratio measured on one plate is the prediction for the same condition on the other; it is '
          'scored on the 14 contexts that have a plate copy (%s condition pairs on splits 1, 0 and 2). The last row '
          'compares the two models. b, The measured ratio over conditions; the generated spread divided by the '
          'measured spread (mean over conditions and seeds); and the standard deviation over conditions of the ratio '
          'predicted by each linear reference, which would be zero without the clipping at zero. Parts a and b come '
          'from an analysis specified before the data were scored. c, Decomposition. Agreement is the measure of '
          'part a, computed on the prediction averaged over the three seeds (for the two models it therefore differs '
          'from part a, by at most %s). “Vehicle spread removed” is the partial Spearman correlation between '
          'predicted and measured ratio given the spread of the vehicle cells, the denominator shared by both ratios. '
          '“Compound-blind prediction removed” is the same given the ratio predicted by the compound-blind average. '
          '“Agreement on spread itself” is the partial Spearman correlation between predicted and measured spread, '
          'with no ratio, given the vehicle spread. d, Spearman correlation of each quantity with the vehicle spread. '
          'Parts c and d use the contexts and the bootstrap resamples of part a. They were exploratory: the '
          'decomposition was added after the compound-blind average had scored above zero on the development split, '
          'and its tests were fixed before it was run.'
          % ('{:,}, {:,} and {:,}'.format(*conds), structs[0], '{:,}, {:,} and {:,}'.format(*pairs), num(seed_gap, 3)),
          '',
          '**a** | Spread agreement.',
          '']
    rows = []
    for label, arm, name, _ in A_ARMS:
        rows.append([label] + cells(lambda s, arm=arm: A[s][arm]))
        for s in SPLITS:
            keep('a', 'spread agreement', label, s, A[s][arm], PRE, REC_H, 'per_split/%s/contrasts/%s' % (s, name))
    rows.append([L_REP] + cells(lambda s: A[s]['replicate']))
    rows.append([L_DIFF] + cells(lambda s: A[s]['diff']))
    for s in SPLITS:
        keep('a', 'spread agreement', L_REP, s, A[s]['replicate'], PRE, REC_H, 'per_split/%s/contrasts/%s' % (s, N_REP))
        keep('a', 'difference in spread agreement', L_DIFF, s, A[s]['diff'], PRE, REC_H,
             'per_split/%s/contrasts/%s' % (s, N_DIFF))
    md += table(['Entry'] + SPLIT_HEAD, rows)

    md += ['', '**b** | Size of the measured and predicted quantities.', '']
    rows = []
    for label, key in (('Measured ratio, mean', 'mean'), ('Measured ratio, median', 'median'),
                       ('Measured ratio, standard deviation', 'sd')):
        rows.append([label] + [num(H[s]['r_true'][key], 3) for s in SPLITS])
        for s in SPLITS:
            keep('b', label[0].lower() + label[1:], 'measured cells', s, float(H[s]['r_true'][key]), DESC, REC_H,
                 'per_split/%s/r_true/%s' % (s, key))
    rows.append(['Conditions with a measured ratio above 1 (%)']
                + [num(100 * H[s]['r_true']['frac_above_one'], 1) for s in SPLITS])
    for s in SPLITS:                                      # the csv keeps the fraction, as the record stores it
        keep('b', 'fraction of conditions with a measured ratio above 1', 'measured cells', s,
             float(H[s]['r_true']['frac_above_one']), DESC, REC_H, 'per_split/%s/r_true/frac_above_one' % s)
    for label, arm in ((L_REW, 'recipe'), (L_OT, 'ot_pair')):
        rows.append(['Generated spread / measured spread, %s' % (label[0].lower() + label[1:])]
                    + [num(H[s]['metrics'][arm]['pred_over_measured_spread'], 3) for s in SPLITS])
        for s in SPLITS:
            keep('b', 'generated spread / measured spread', label, s,
                 float(H[s]['metrics'][arm]['pred_over_measured_spread']), DESC, REC_H,
                 'per_split/%s/metrics/%s/pred_over_measured_spread' % (s, arm))
    for label, arm in ((L_BLIND, 'mean_response'), (L_RT, 'ridge_train'), (L_RCV, 'ridge_S7')):
        rows.append(['Standard deviation of the predicted ratio, %s' % (label[0].lower() + label[1:])]
                    + [num(H[s]['clip_variation'][arm], 3) for s in SPLITS])
        for s in SPLITS:
            keep('b', 'standard deviation of the predicted ratio', label, s, float(H[s]['clip_variation'][arm]), DESC,
                 REC_H, 'per_split/%s/clip_variation/%s' % (s, arm))
    md += table(['Quantity'] + SPLIT_HEAD, rows)

    md += ['', '**c** | Decomposition (exploratory).', '']
    rows = []
    for label, arm, _, _ in A_ARMS:
        for measure, mlabel in C_MEASURES:
            k = '%s|%s' % (measure, arm)
            if k not in keys:
                continue
            rows.append([label, mlabel] + cells(lambda s, k=k: C[s][k]))
            for s in SPLITS:
                keep('c', mlabel[0].lower() + mlabel[1:], label, s, C[s][k], EXPL, REC_D,
                     'per_split/%s/contrasts/%s' % (s, k))
    md += table(['Entry', 'Quantity'] + SPLIT_HEAD, rows)

    md += ['', '**d** | Correlation with the vehicle spread (exploratory).', '']
    rows = []
    for label, k in D_ROWS:
        rows.append([label] + cells(lambda s, k=k: C[s][k]))
        for s in SPLITS:
            keep('d', 'correlation with the vehicle spread', label, s, C[s][k], EXPL, REC_D,
                 'per_split/%s/contrasts/%s' % (s, k))
    md += table(['Quantity'] + SPLIT_HEAD, rows)

    md += ['',
           'Sources: step19_heterogeneity.json (sha256 %s; parts a and b) and step19_diag_control_route.json '
           '(sha256 %s; parts c and d). The values of part a, other than its last row, equal those Supplementary '
           'Fig. S1 was drawn from (fig5_values.csv, panel c). Assembled by make_supp_table_s17.py.'
           % (IN['het'][1][:16], IN['dec'][1][:16])]
    write_outputs(TABLE, long_rows, md)
    print('[checks] 1 (%d values of the first record), 2 (%d values and agreements of the decomposition record), '
          '3 (%d values of Supplementary Fig. S1), 4 (statements of Supplementary Note 5) all passed' % (n1, n2, n3))


if __name__ == '__main__':
    main()
