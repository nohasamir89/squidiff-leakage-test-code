#!/usr/bin/env python3
"""
make_supp_table_s16.py - Supplementary Table S16: response-magnitude agreement on Tahoe-100M and on sci-Plex
(Supplementary Note 4; Supplementary Fig. S2).

Purpose
-------
Supplementary Note 4 reports how well each model ranks conditions by the size of their response, and cites this
table for every number; the caption of Supplementary Fig. S2 cites it for the values of the figure. The numbers come
from seven committed records of five analyses. This script assembles them in one table, so that each number traces
to a record and is rounded once, and it marks which parts were specified before the data were scored and which were
exploratory.

The measure (as defined before both analyses): within a context (one cell line at one dose) holding at least ten
held-out conditions, the Spearman correlation over conditions between the size of the predicted change and the size
of the measured change; then the mean over contexts and, for a model, over its seeds. Zero means no ranking.

Five parts:
  a  Tahoe-100M, all conditions: every model and linear reference against zero, and the plate replicate.
  b  sci-Plex: the same, the split-half of the measured cells, and the rewired model against each ridge.
  c  Tahoe-100M with the compound input replaced or removed, and a prediction made without a model.
  d  Tahoe-100M by dose (exploratory).
  e  Tahoe-100M on the conditions that have a plate replicate (exploratory).

Logic
-----
- Inputs (committed records only; see supp_table_common.guard):
      tahoe_results/step16_analysis.json            parts a: per_split/<s>/metrics/<arm>/mra, contrasts/secondary_mra_*
      leakage_results/step17_mra.json               part b: per_split/<s>/metrics/<arm>/mra, anchor, contrasts
      tahoe_results/step24_swap.json                part c, development split: contrasts/sec_mra_* (with intervals)
      tahoe_results/step25_swap.json                part c, three splits: per_split/<s>/values/<arm>/mra (points)
      tahoe_results/step26_source.json              part c: per_split/<s>/values/ctrl_line__orig/mra (points)
      tahoe_results/step16_diag_mra_dose.json       part d: per_split/<s>/values/<dose>/<arm>, contrasts
      tahoe_results/step16_diag_mra_ceiling.json    part e: per_split/<s>/restricted/<arm>, contrasts
      figures/squidiff_publication_figures/fig5_values.csv   the values Supplementary Fig. S2 was drawn from
                                                             (panel b of the old Figure 5); check only
  Record names -> names in the paper: base = original model; recipe (Tahoe-100M) and fm_50 (sci-Plex) = rewired
  model; guidance_w2, attn_cross, mlp_ctrl, targets, targets_shuf = its variants; ridge_train = training-set ridge;
  ridge_S7 = cross-validated ridge; replicate / ceiling = plate replicate; anchor = split-half of the measured
  cells; recipe_shuf / fpshuf = the fingerprint of another held-out compound; recipe_zero / fpzero = no fingerprint;
  ctrl_line = the cell-line vehicle average used as a prediction; mra = the measure.
- Checks, all before anything is written (a failure stops the script):
    1. Tahoe-100M record: each value against zero equals the stored measure (1e-9); each model value is the mean of
       its three seeds; the compound-blind average and the unaltered vehicle cells score exactly zero; 42 contexts,
       14 for the replicate.
    2. sci-Plex record: the same with five seeds and 12 contexts; the split-half value is the mean of its five
       halvings; each difference between the rewired model and a ridge equals the difference of the two values.
    3. Every value of the figure (30 rows of panel b) equals the value given here (1e-9).
    4. Part c: the three records agree with each other and with the Tahoe-100M record wherever they hold the same
       quantity (1e-9), and each difference of the development split equals the difference of its two values.
    5. Parts d and e: both exploratory records were computed from the Tahoe-100M record used here (its sha256 is
       stored in them); each value against zero equals the stored value; the replicate-restricted values equal the
       values of the highest dose on the two replicate plates in the by-dose record (1e-9).
    6. The statements of Supplementary Note 4 hold: the ranges it quotes at their printed precision, the number of
       splits on which a ridge is resolved, the order of the three doses for the rewired model, and that no model
       is resolved on the replicate conditions (EXPECTED below; checks only, never inputs to a computation).
- Rounding: three decimals, each value rounded once from full precision; values carry their sign.
- Outputs (never overwritten): supplementary_tables/supp_table_S16.csv and .md (see supp_table_common).

Run from the research worktree root (supp_table_common.py must be in the same folder):
    python3 make_supp_table_s16.py
"""
import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import (DASH, SPLITS, SPLIT_HEAD, close, die, guard, interval, load_csv, load_json, num, same,
                               table, write_outputs)

TABLE = 'S16'
IN = dict(
    tahoe=('tahoe_results/step16_analysis.json',
           '561a2159e359965cef804549391afad95d1be6b144d335e61caea45b88dbcad8'),
    sci=('leakage_results/step17_mra.json',
         '9d2744b9fbd956195cfb4e1c1defb285f4d1738b41645e1de8100cd59df5e717'),
    swap_dev=('tahoe_results/step24_swap.json',
              '8055d34ceec1c7e705c1ce1ec837c948fcc721bda1eb1f25f10788c09d274a70'),
    swap=('tahoe_results/step25_swap.json',
          'abf409df3a8a06dd154487c0a177876059ea22ffa85b2e33c3da8b1fa3123974'),
    source=('tahoe_results/step26_source.json',
            '2604f5cf8535d4acaf33a741f0074e3d99c5db06012c2cc9ec57dcb1c0296726'),
    dose=('tahoe_results/step16_diag_mra_dose.json',
          'f4489af6d3eddb6a143d79b2875d42b3609c99e1aca0198017a16e13814f9c9d'),
    rep=('tahoe_results/step16_diag_mra_ceiling.json',
         '20f4406f38ded4d6f00438662bd22e6415c558118a83cada0540f89523c47157'),
    fig5=('figures/squidiff_publication_figures/fig5_values.csv',
          '95c5f189b195dc4d273b00aeacf1ba3b85ec64da7e4bf01a3bd70c323b1f9ca3'),
)
TOL = 1e-9
DEV = 'drug_split_1'

L_ORIG, L_REW = 'Original model', 'Rewired model'
L_GUID = 'Rewired model with classifier-free guidance (weight 2)'
L_ATTN = 'Rewired model with cross-attention'
L_CTRL = 'Rewired model, parameter-matched control without attention'
L_TARG = 'Rewired model with target annotation'
L_TSHUF = 'Rewired model, target sets reassigned between compounds'
L_RT, L_RCV = 'Training-set ridge', 'Cross-validated ridge'
L_REP, L_HALF = 'Plate replicate', 'Split-half of the measured cells'
# Tahoe-100M arms of the record, in the order of the table: (label, arm, a model with seeds?)
T_ARMS = [(L_ORIG, 'base', True), (L_REW, 'recipe', True), (L_GUID, 'guidance_w2', True),
          (L_ATTN, 'attn_cross', True), (L_CTRL, 'mlp_ctrl', True), (L_TARG, 'targets', True),
          (L_TSHUF, 'targets_shuf', True), (L_RT, 'ridge_train', False), (L_RCV, 'ridge_S7', False)]
# sci-Plex arms: (label, arm, name of its comparison with zero, a model with seeds?)
S_ARMS = [(L_ORIG, 'base', 'secondary_mra_base_minus_0', True),
          (L_REW, 'fm_50', 'Q1_primary_mra_fm_50_minus_0', True),
          (L_GUID, 'guidance_w2', 'secondary_mra_guidance_w2_minus_0', True),
          (L_ATTN, 'attn_cross', 'secondary_mra_attn_cross_minus_0', True),
          (L_CTRL, 'mlp_ctrl', 'secondary_mra_mlp_ctrl_minus_0', True),
          (L_RT, 'ridge_train', 'secondary_mra_ridge_train_minus_0', False),
          (L_RCV, 'ridge_S7', 'secondary_mra_ridge_S7_minus_0', False)]
# part c: (label, arm in the three-split record, name in the development-split record or None)
C_ARMS = [('Rewired model, its own fingerprint', 'recipe_own', 'recipe'),
          ('Rewired model, fingerprint of another held-out compound', 'recipe_shuf', 'fpshuf'),
          ('Rewired model, no fingerprint', 'recipe_zero', 'fpzero'),
          ('Original model, its own fingerprint', 'base_own', None),
          ('Original model, fingerprint of another held-out compound', 'base_shuf', None),
          ('Original model, no fingerprint', 'base_zero', None)]
L_LINE = 'Cell-line vehicle average (no model)'
C_DIFFS = [('Rewired model: own − another compound’s fingerprint', 'sec_mra_recipe_minus_fpshuf', 'recipe', 'fpshuf'),
           ('Rewired model: own − no fingerprint', 'sec_mra_recipe_minus_fpzero', 'recipe', 'fpzero')]
DOSES = [('50 nM', '0.05 uM', 50), ('500 nM', '0.5 uM', 500), ('5,000 nM', '5 uM', 5000)]
SEL_REP = 'top, plates 6/14'                              # highest dose on the two plates of the replicate pair

# Printed values Supplementary Note 4 quotes (checks only, never inputs): smallest and largest over the splits.
EXPECTED = dict(
    tahoe_rewired=('0.29', '0.32'),          # "On Tahoe the rewired model reached 0.29-0.32 on every split."
    tahoe_original=('-0.11', '-0.06'),       # "The original model did not rank conditions (-0.11 to -0.06)"
    line_mean=('-0.04', '0.04'),             # "the line-mean prediction ... did not (-0.04 to +0.04)"
    swapped=('0.21', '0.36'),                # "with another compound's fingerprint or with none ... (0.21-0.36)"
    sci_rewired=('0.24', '0.33'),            # "On sci-Plex the rewired model reached 0.24-0.33"
    replicate=('0.33', '0.48'),              # "whereas the replicate reached 0.33-0.48"
)


def tri(c):
    """(value, lower, upper) of one stored comparison."""
    return float(c['diff']), float(c['ci'][0]), float(c['ci'][1])


def resolved_above(c):
    return c[1] > 0


def includes_zero(c):
    return c[1] <= 0 <= c[2]


def span(values):
    """Smallest and largest value at two decimals, as the text prints ranges."""
    return ('%.2f' % min(values), '%.2f' % max(values))


def mean(values):
    return sum(values) / len(values)


def main():
    guard(IN, TABLE)
    tah, sci, s24, s25, s26, dose, rep = (load_json(IN[k][0]) for k in
                                          ('tahoe', 'sci', 'swap_dev', 'swap', 'source', 'dose', 'rep'))
    T = dict((s, tah['per_split'][s]) for s in SPLITS)
    P = dict((s, sci['per_split'][s]) for s in SPLITS)

    # ---- check 1: the Tahoe-100M record -------------------------------------------------------------------------------
    same(tah['splits'], SPLITS, 'check 1, split order of the Tahoe-100M record')
    same((tah['interval'], tah['n_boot'], tah['n_seeds'], tah['min_context']),
         ('97.5% percentile per split, structure-clustered', 5000, 3, 10), 'check 1, settings of the Tahoe-100M record')
    n1 = 0
    A = {}                                                # part a: A[split][arm] = (value, lower, upper)
    for s in SPLITS:
        same((T[s]['n_contexts'], T[s]['n_contexts_mra'], T[s]['replicate']['mra_n_contexts']), (42, 42, 14),
             'check 1, contexts on %s' % s)
        A[s] = {}
        for label, arm, seeded in T_ARMS:
            c = tri(T[s]['contrasts']['secondary_mra_%s_minus_0' % arm])
            close(c[0], T[s]['metrics'][arm]['mra'], TOL, 'check 1, %s on %s' % (arm, s))
            per_seed = T[s]['metrics'][arm]['mra_per_seed']
            same(len(per_seed), 3 if seeded else 1, 'check 1, seeds of %s on %s' % (arm, s))
            close(c[0], mean(per_seed), TOL, 'check 1, seed mean of %s on %s' % (arm, s))
            A[s][arm] = c
            n1 += 2
        c = tri(T[s]['contrasts']['secondary_mra_ceiling_minus_0'])
        close(c[0], T[s]['replicate']['mra'], TOL, 'check 1, plate replicate on %s' % s)
        A[s]['replicate'] = c
        for arm in ('mean_response', 'control_paste'):   # one size per context: no ranking, by construction
            same(T[s]['metrics'][arm]['mra'], 0.0, 'check 1, %s scores zero on %s' % (arm, s))
        n1 += 3
    pairs = [T[s]['replicate']['n_pairs'] for s in SPLITS]

    # ---- check 2: the sci-Plex record ---------------------------------------------------------------------------------
    same((sci['interval'], sci['n_boot'], sci['min_context'], sci['comparator']),
         ('97.5 % percentile per split, compound-clustered', 5000, 10, 'fm_50'), 'check 2, settings of the sci-Plex record')
    n2 = 0
    B, BD = {}, {}                                        # part b: values against zero, and the two differences
    for s in SPLITS:
        same((P[s]['n_contexts'], P[s]['n_contexts_used']), (12, 12), 'check 2, contexts on %s' % s)
        B[s] = {}
        for label, arm, name, seeded in S_ARMS:
            c = tri(P[s]['contrasts'][name])
            close(c[0], P[s]['metrics'][arm]['mra'], TOL, 'check 2, %s on %s' % (arm, s))
            if seeded:
                per_seed = P[s]['metrics'][arm]['mra_per_seed']
                same(len(per_seed), 5, 'check 2, seeds of %s on %s' % (arm, s))
                close(c[0], mean(per_seed), TOL, 'check 2, seed mean of %s on %s' % (arm, s))
            B[s][arm] = c
            n2 += 1
        c = tri(P[s]['contrasts']['descriptive_mra_anchor_minus_0'])
        halves = P[s]['anchor']['per_half_split']
        same(len(halves), 5, 'check 2, halvings on %s' % s)
        close(c[0], P[s]['anchor']['mra'], TOL, 'check 2, split-half on %s' % s)
        close(c[0], mean(halves), TOL, 'check 2, split-half as the mean of its halvings on %s' % s)
        B[s]['anchor'] = c
        for arm in ('mean_response', 'control_paste'):
            same(P[s]['metrics'][arm]['mra'], 0.0, 'check 2, %s scores zero on %s' % (arm, s))
        BD[s] = {}
        for key, name, other in (('rt', 'Q2_coprimary_mra_fm_50_minus_ridge_train', 'ridge_train'),
                                 ('rcv', 'secondary_mra_fm_50_minus_ridge_S7', 'ridge_S7')):
            c = tri(P[s]['contrasts'][name])
            close(c[0], B[s]['fm_50'][0] - B[s][other][0], TOL, 'check 2, %s on %s' % (name, s))
            BD[s][key] = c
        n2 += 4

    # ---- check 3: the values Supplementary Fig. S2 was drawn from -------------------------------------------------------
    fig_split = {'Split 1': 'drug_split_1', 'Split 0': 'drug_split_0', 'Split 2': 'drug_split_2'}
    fig_row = {'Tahoe-100M': {'Original model': 'base', 'Rewired model': 'recipe', 'Training-set ridge': 'ridge_train',
                              'Cross-validated ridge': 'ridge_S7', 'Plate replicate (14 contexts)': 'replicate'},
               'sci-Plex': {'Original model': 'base', 'Rewired model': 'fm_50', 'Training-set ridge': 'ridge_train',
                            'Cross-validated ridge': 'ridge_S7', 'Split-half anchor': 'anchor'}}
    n3 = 0
    for r in load_csv(IN['fig5'][0]):
        if r['panel'] != 'b':
            continue
        arm = fig_row[r['facet']][r['row']]
        c = (A if r['facet'] == 'Tahoe-100M' else B)[fig_split[r['split']]][arm]
        what = 'check 3, fig5_values.csv panel b: %s, %s, %s' % (r['facet'], r['row'], r['split'])
        close(c[0], float(r['point']), TOL, what)
        if arm in ('replicate', 'anchor'):               # drawn as points only
            same((r['lo'], r['hi']), ('NA', 'NA'), what + ' (drawn without an interval)')
        else:
            close(c[1], float(r['lo']), TOL, what + ' (lower)')
            close(c[2], float(r['hi']), TOL, what + ' (upper)')
        n3 += 1
    same(n3, 30, 'check 3, number of panel-b values in fig5_values.csv')

    # ---- check 4: part c, the compound input replaced or removed ---------------------------------------------------------
    same(s24['split'], DEV, 'check 4, split of the development-split record')
    for r_ in (s24, s25, s26):
        same(r_['interval'], '97.5% percentile, structure-clustered, fresh RandomState(0) per scored set',
             'check 4, interval of a record of part c')
    n4 = 0
    C, CL = {}, {}                                        # C[split][arm] = point value; CL[split] = line-mean value
    for s in SPLITS:
        v25, v26 = s25['per_split'][s]['values'], s26['per_split'][s]['values']
        C[s] = dict((arm, float(v25[arm]['mra'])) for _, arm, _ in C_ARMS)
        close(C[s]['recipe_own'], A[s]['recipe'][0], TOL, 'check 4, rewired model with its own fingerprint on %s' % s)
        close(C[s]['base_own'], A[s]['base'][0], TOL, 'check 4, original model with its own fingerprint on %s' % s)
        for arm25, arm26 in (('recipe_own', 'recipe__orig'), ('recipe_shuf', 'recipe_fpshuf__orig'),
                             ('recipe_zero', 'recipe_fpzero__orig'), ('base_own', 'base__orig'),
                             ('base_shuf', 'base_fpshuf__orig'), ('base_zero', 'base_fpzero__orig')):
            close(C[s][arm25], float(v26[arm26]['mra']), TOL, 'check 4, %s in the two records on %s' % (arm25, s))
            n4 += 1
        CL[s] = float(v26['ctrl_line__orig']['mra'])
        n4 += 2
    CI = {}                                               # development split: intervals of three arms
    for _, arm25, arm24 in C_ARMS:
        if arm24 is None:
            continue
        c = tri(s24['contrasts']['sec_mra_%s' % arm24])
        close(c[0], float(s24['values'][arm24]['mra']), TOL, 'check 4, %s against its stored value' % arm24)
        close(c[0], C[DEV][arm25], TOL, 'check 4, %s in the development-split and the three-split record' % arm24)
        CI[arm25] = c
        n4 += 2
    CD = {}
    for label, name, a_, b_ in C_DIFFS:
        c = tri(s24['contrasts'][name])
        close(c[0], float(s24['values'][a_]['mra']) - float(s24['values'][b_]['mra']), TOL, 'check 4, %s' % name)
        CD[name] = c
        n4 += 1

    # ---- check 5: parts d and e, the two exploratory records ----------------------------------------------------------
    same((dose['record_sha256'], rep['record_sha256']), (IN['tahoe'][1],) * 2,
         'check 5, the exploratory records were computed from the Tahoe-100M record used here')
    same(dose['section9_sha256'], IN['rep'][1], 'check 5, the by-dose record refers to the replicate record used here')
    same((dose['n_boot'], dose['min_context'], rep['n_boot'], rep['min_context']), (5000, 10, 5000, 10),
         'check 5, settings of the exploratory records')
    n5 = 0
    D, E, ED = {}, {}, {}
    for s in SPLITS:
        d_, r_ = dose['per_split'][s], rep['per_split'][s]
        same(d_['dose_levels_nM'], [x[2] for x in DOSES], 'check 5, dose levels on %s' % s)
        same((r_['n_contexts'], r_['n_contexts_restricted'], r_['n_replicated_conditions']),
             (42, 14, 2 * T[s]['replicate']['n_pairs']),
             'check 5, contexts and replicated conditions on %s' % s)
        D[s], E[s] = {}, {}
        for _, sel, _ in DOSES:
            same(d_['n_contexts_per_selection'][sel], 14, 'check 5, contexts at %s on %s' % (sel, s))
            for label, arm, _ in T_ARMS:
                c = tri(d_['contrasts']['%s | %s - 0' % (sel, arm)])
                close(c[0], float(d_['values'][sel][arm]), TOL, 'check 5, %s at %s on %s' % (arm, sel, s))
                D[s][(sel, arm)] = c
                n5 += 1
        for label, arm, _ in T_ARMS:
            c = tri(r_['contrasts']['%s_minus_0' % arm])
            close(c[0], float(r_['restricted'][arm]), TOL, 'check 5, %s on the replicate conditions, %s' % (arm, s))
            close(c[0], float(d_['values'][SEL_REP][arm]), TOL,
                  'check 5, %s: replicate conditions against the highest dose on the replicate plates, %s' % (arm, s))
            close(float(r_['full_record'][arm]), A[s][arm][0], TOL, 'check 5, %s on all conditions, %s' % (arm, s))
            E[s][arm] = c
            n5 += 3
        c = tri(r_['contrasts']['ceiling_minus_0'])
        close(c[0], A[s]['replicate'][0], TOL, 'check 5, plate replicate in the two records on %s' % s)
        E[s]['replicate'] = c
        c = tri(r_['contrasts']['recipe_minus_ceiling'])
        close(c[0], E[s]['recipe'][0] - E[s]['replicate'][0], TOL, 'check 5, rewired model minus replicate on %s' % s)
        ED[s] = c
        n5 += 2
    n_rep = [dose['per_split'][s]['n_conditions_per_selection'][SEL_REP] for s in SPLITS]
    n_dose = dict((sel, [dose['per_split'][s]['n_conditions_per_selection'][sel] for s in SPLITS]) for _, sel, _ in DOSES)

    # ---- check 6: the statements of Supplementary Note 4 --------------------------------------------------------------
    got = dict(
        tahoe_rewired=span([A[s]['recipe'][0] for s in SPLITS]),
        tahoe_original=span([A[s]['base'][0] for s in SPLITS]),
        line_mean=span([CL[s] for s in SPLITS]),
        swapped=span([C[s][a] for s in SPLITS for a in ('recipe_shuf', 'recipe_zero')]),
        sci_rewired=span([B[s]['fm_50'][0] for s in SPLITS]),
        replicate=span([A[s]['replicate'][0] for s in SPLITS]),
    )
    same(got, EXPECTED, 'check 6, ranges quoted in Supplementary Note 4')
    for arm in ('ridge_train', 'ridge_S7'):              # "the ridges did so on at most one split"
        if sum(resolved_above(A[s][arm]) for s in SPLITS) > 1:
            die('check 6: %s is resolved above zero on more than one Tahoe-100M split' % arm)
    for arm in ('ridge_train', 'ridge_S7'):              # sci-Plex: "positive estimates, resolved on one split"
        same(([B[s][arm][0] > 0 for s in SPLITS], sum(resolved_above(B[s][arm]) for s in SPLITS)), ([True] * 3, 1),
             'check 6, %s on sci-Plex: positive on every split, resolved on one' % arm)
    same([includes_zero(BD[s]['rt']) for s in SPLITS], [True] * 3,
         'check 6, the rewired model does not differ from the training-set ridge on sci-Plex')
    for s in SPLITS:                                      # "strongest at the intermediate dose and weakest at the lowest"
        low, mid, top = (D[s][(sel, 'recipe')][0] for _, sel, _ in DOSES)
        if not (mid > top > low):
            die('check 6: on %s the rewired model is not strongest at 500 nM and weakest at 50 nM' % s)
    for s in SPLITS:                                      # "no model's ranking was distinguishable from zero"
        for label, arm, _ in T_ARMS:
            if not includes_zero(E[s][arm]):
                die('check 6: %s is resolved on the replicate conditions of %s' % (arm, s))

    # ---- the table ---------------------------------------------------------------------------------------------------
    long_rows = []

    def keep(part, dataset, selection, row, split, c, status, record, key):
        v, lo, hi = c if isinstance(c, tuple) else (c, None, None)
        long_rows.append(dict(part=part, dataset=dataset, selection=selection, row=row, split=split, value=v, lo=lo,
                              hi=hi, excludes_zero='' if lo is None else int(lo > 0 or hi < 0), status=status,
                              source_record=record, source_key=key))

    def cells(get):
        return [interval(*get(s)) for s in SPLITS]

    md = ['Supplementary Table S16 | Response-magnitude agreement (Supplementary Note 4; Supplementary Fig. S2). '
          'Within each context (one cell line at one dose) holding at least ten held-out conditions, the Spearman '
          'correlation over conditions between the size of the predicted change and the size of the measured change '
          '(Euclidean norms, over the 200 genes, of the mean change from the matched vehicle cells) was computed; '
          'the measure is its mean over contexts and, for a model, over seeds (three on Tahoe-100M, five on '
          'sci-Plex). Zero means no ranking; the compound-blind average and the unaltered vehicle cells have one '
          'size per context and score zero by construction. Values are given with 97.5 %% percentile bootstrap '
          'intervals against zero (5,000 replicates, resampling held-out chemical structures on Tahoe-100M and '
          'compounds on sci-Plex); an asterisk marks an interval that excludes zero. a, Tahoe-100M, 42 contexts '
          '(14 cell lines at three doses); the plate replicate is scored on the 14 contexts that have a plate copy '
          '(%s condition pairs on splits 1, 0 and 2). b, sci-Plex, 12 contexts (three cell lines at four doses); '
          'the split-half entry scores one half of the measured cells against the other, as the mean of five random '
          'halvings, and the last two rows compare the rewired model with each ridge. c, Tahoe-100M: the two models '
          'given the fingerprint of another held-out compound or none, and the cell-line vehicle average used as a '
          'prediction without a model (Results 5; Supplementary Table S15). Intervals were computed for the rewired '
          'model on the development split only; the other entries of part c are point values. Parts a to c come '
          'from analyses specified before the data were scored. d, Tahoe-100M by dose (14 contexts per dose; %s '
          'conditions at 50 nM, %s at '
          '500 nM and %s at 5,000 nM); values for splits 1 / 0 / 2. e, Tahoe-100M restricted to the conditions that '
          'have a plate replicate: 14 contexts at the highest dose (%s conditions). Parts d and e were exploratory.'
          % ('{:,}, {:,} and {:,}'.format(*pairs), ' / '.join('{:,}'.format(x) for x in n_dose['0.05 uM']),
             ' / '.join('{:,}'.format(x) for x in n_dose['0.5 uM']), ' / '.join('{:,}'.format(x) for x in n_dose['5 uM']),
             '{:,}, {:,} and {:,}'.format(*n_rep)),
          '',
          '**a** | Tahoe-100M, all conditions.',
          '']
    rows = []
    for label, arm, _ in T_ARMS:
        rows.append([label] + cells(lambda s, arm=arm: A[s][arm]))
        for s in SPLITS:
            keep('a', 'Tahoe-100M', 'all conditions', label, s, A[s][arm], 'specified before scoring',
                 'step16_analysis.json', 'per_split/%s/contrasts/secondary_mra_%s_minus_0' % (s, arm))
    rows.append([L_REP + ' (14 contexts)'] + cells(lambda s: A[s]['replicate']))
    for s in SPLITS:
        keep('a', 'Tahoe-100M', 'all conditions', L_REP, s, A[s]['replicate'], 'specified before scoring',
             'step16_analysis.json', 'per_split/%s/contrasts/secondary_mra_ceiling_minus_0' % s)
    md += table(['Entry'] + SPLIT_HEAD, rows)

    md += ['', '**b** | sci-Plex.', '']
    rows = []
    for label, arm, name, _ in S_ARMS:
        rows.append([label] + cells(lambda s, arm=arm: B[s][arm]))
        for s in SPLITS:
            keep('b', 'sci-Plex', 'all conditions', label, s, B[s][arm], 'specified before scoring',
                 'step17_mra.json', 'per_split/%s/contrasts/%s' % (s, name))
    rows.append([L_HALF] + cells(lambda s: B[s]['anchor']))
    for s in SPLITS:
        keep('b', 'sci-Plex', 'all conditions', L_HALF, s, B[s]['anchor'], 'descriptive', 'step17_mra.json',
             'per_split/%s/contrasts/descriptive_mra_anchor_minus_0' % s)
    for label, key, name in (('Rewired model − training-set ridge', 'rt', 'Q2_coprimary_mra_fm_50_minus_ridge_train'),
                             ('Rewired model − cross-validated ridge', 'rcv', 'secondary_mra_fm_50_minus_ridge_S7')):
        rows.append([label] + cells(lambda s, key=key: BD[s][key]))
        for s in SPLITS:
            keep('b', 'sci-Plex', 'all conditions', label, s, BD[s][key], 'specified before scoring',
                 'step17_mra.json', 'per_split/%s/contrasts/%s' % (s, name))
    md += table(['Entry'] + SPLIT_HEAD, rows)

    md += ['', '**c** | Tahoe-100M, compound input replaced or removed.', '']
    rows = []
    for label, arm25, arm24 in C_ARMS:
        first = interval(*CI[arm25]) if arm25 in CI else num(C[DEV][arm25], 3, True)
        rows.append([label, first] + [num(C[s][arm25], 3, True) for s in SPLITS[1:]])
        for s in SPLITS:
            if s == DEV and arm25 in CI:
                keep('c', 'Tahoe-100M', 'all conditions', label, s, CI[arm25], 'specified before scoring',
                     'step24_swap.json', 'contrasts/sec_mra_%s' % arm24)
            else:
                keep('c', 'Tahoe-100M', 'all conditions', label, s, C[s][arm25], 'point value',
                     'step25_swap.json', 'per_split/%s/values/%s/mra' % (s, arm25))
    rows.append([L_LINE] + [num(CL[s], 3, True) for s in SPLITS])
    for s in SPLITS:
        keep('c', 'Tahoe-100M', 'all conditions', L_LINE, s, CL[s], 'point value', 'step26_source.json',
             'per_split/%s/values/ctrl_line__orig/mra' % s)
    for label, name, _, _ in C_DIFFS:
        rows.append([label, interval(*CD[name]), DASH, DASH])
        keep('c', 'Tahoe-100M', 'all conditions', label, DEV, CD[name], 'specified before scoring',
             'step24_swap.json', 'contrasts/%s' % name)
    md += table(['Entry'] + SPLIT_HEAD, rows)

    md += ['', '**d** | Tahoe-100M by dose (splits 1 / 0 / 2).', '']
    rows = []
    for label, arm, _ in T_ARMS:
        row = [label]
        for dlabel, sel, _ in DOSES:
            row.append(' / '.join(num(D[s][(sel, arm)][0], 3, True) + ('*' if not includes_zero(D[s][(sel, arm)]) else '')
                                  for s in SPLITS))
            for s in SPLITS:
                keep('d', 'Tahoe-100M', dlabel, label, s, D[s][(sel, arm)], 'exploratory',
                     'step16_diag_mra_dose.json', 'per_split/%s/contrasts/%s | %s - 0' % (s, sel, arm))
        rows.append(row)
    md += table(['Entry'] + [d[0] for d in DOSES], rows)

    md += ['', '**e** | Tahoe-100M, conditions with a plate replicate.', '']
    rows = []
    for label, arm, _ in T_ARMS:
        rows.append([label] + cells(lambda s, arm=arm: E[s][arm]))
        for s in SPLITS:
            keep('e', 'Tahoe-100M', 'replicate conditions', label, s, E[s][arm], 'exploratory',
                 'step16_diag_mra_ceiling.json', 'per_split/%s/contrasts/%s_minus_0' % (s, arm))
    rows.append([L_REP] + cells(lambda s: E[s]['replicate']))
    rows.append(['Rewired model − plate replicate'] + cells(lambda s: ED[s]))
    for s in SPLITS:
        keep('e', 'Tahoe-100M', 'replicate conditions', L_REP, s, E[s]['replicate'], 'exploratory',
             'step16_diag_mra_ceiling.json', 'per_split/%s/contrasts/ceiling_minus_0' % s)
        keep('e', 'Tahoe-100M', 'replicate conditions', 'Rewired model − plate replicate', s, ED[s], 'exploratory',
             'step16_diag_mra_ceiling.json', 'per_split/%s/contrasts/recipe_minus_ceiling' % s)
    md += table(['Entry'] + SPLIT_HEAD, rows)

    md += ['',
           'Sources: step16_analysis.json (sha256 %s; part a), step17_mra.json (sha256 %s; part b), '
           'step24_swap.json (sha256 %s), step25_swap.json (sha256 %s) and step26_source.json (sha256 %s; part c), '
           'step16_diag_mra_dose.json (sha256 %s; part d) and step16_diag_mra_ceiling.json (sha256 %s; part e). '
           'The values of the original model, the rewired model, the two ridges, the plate replicate and the '
           'split-half entry in parts a and b equal those Supplementary Fig. S2 was drawn from (fig5_values.csv, '
           'panel b). Part d shows values and asterisks only; its intervals are in supp_table_S16.csv in the '
           'code repository. Assembled by make_supp_table_s16.py.'
           % tuple(IN[k][1][:16] for k in ('tahoe', 'sci', 'swap_dev', 'swap', 'source', 'dose', 'rep'))]
    write_outputs(TABLE, long_rows, md)
    print('[checks] 1 (%d values of the Tahoe-100M record), 2 (%d of the sci-Plex record), 3 (%d values of '
          'Supplementary Fig. S2), 4 (%d agreements between the records of part c), 5 (%d values of the exploratory '
          'records), 6 (statements of Supplementary Note 4) all passed' % (n1, n2, n3, n4, n5))


if __name__ == '__main__':
    main()
