#!/usr/bin/env python3
"""
make_supp_table_s5.py - Supplementary Table S5: gene identity and population distance of the models and the
references on both datasets (Results 1 and 2; Figs 1b, 2a and 2b).

Purpose
-------
Figs 1b, 2a and 2b and the text of Results 1 and 2 cite Supplementary Table S5 for their values: how well the
original model, the rewired model and each reference recover the responsive genes (gene identity) on Tahoe-100M and
on sci-Plex, and how far each population lies from the measured treated cells (population distance) on Tahoe-100M.
This script assembles the table from the two committed analysis records, so that every number traces to a record
and is rounded once, from full precision.

Three parts:
  a  gene identity of every entry on both datasets (the values of Figs 1b and 2a);
  b  population distance of every entry on Tahoe-100M (the values of Fig. 2b);
  c  the pre-specified contrasts between the rewired model and each reference, with their 97.5 % bootstrap
     intervals (the statements "above", "below" and "no closer" in Results 2 rest on these).

Logic
-----
- Inputs (committed records only; see supp_table_common.guard):
      tahoe_results/step15_model_analysis.json   per_split/<split>/metrics/<arm>/{de_overlap, energy, per_seed};
                                                 per_split/<split>/replicate; per_split/<split>/contrasts/<name>
      leakage_results/step14_analysis.json       per_split/<split>/thesis_metrics/<arm>/de_overlap;
                                                 per_split/<split>/contrasts/<name>
      figures/squidiff_publication_figures/fig2_values.csv, fig1b_values.csv   the values the figures were drawn
                                                 from (used as a cross-check only)
  Record names -> names in the paper: base = original model; recipe (Tahoe) and fm_50 (sci-Plex) = rewired model;
  ridge_train = training-set ridge; ridge_S7 = cross-validated ridge; mean_response = compound-blind average;
  control_paste = unaltered vehicle cells; replicate = plate replicate; de_overlap = gene identity; energy =
  population distance.
- The sci-Plex record stores each contrast as "reference minus rewired model". The table gives every contrast as
  "rewired model minus reference", so those are negated: difference -> -difference, [lo, hi] -> [-hi, -lo].
- Chance level of gene identity: the expected Jaccard index of two random sets of 50 genes out of 200, computed
  exactly from the hypergeometric distribution of their overlap.
- Checks, all before anything is written (a failure stops the script):
    1. Every contrast shown equals the difference of the two arm values shown (1e-9).
    2. Every value of fig2_values.csv and fig1b_values.csv equals the record value shown here (1e-9), so the table
       holds exactly what the figures show; the chance level computed here equals the figure's (1e-12).
    3. The values quoted in the manuscript reproduce at their printed precision (EXPECTED below; checks only,
       never inputs to a computation).
- Rounding: three decimals, each value rounded once from full precision.
- Outputs (never overwritten): supplementary_tables/supp_table_S5.csv and .md (see supp_table_common).

Run from the research worktree root (supp_table_common.py must be in the same folder; Python 3.8 or newer):
    python3 make_supp_table_s5.py
"""
import math
from fractions import Fraction

import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import (SPLITS, SPLIT_HEAD, close, die, guard, interval, load_csv, load_json, num, same,
                               table, triple, write_outputs)

TABLE = 'S5'
IN = dict(
    tahoe=('tahoe_results/step15_model_analysis.json',
           '5fd700b5a030cf5a8ffea57e9a9b2ad840aa32fa9948b32754b9742362a1165e'),
    sciplex=('leakage_results/step14_analysis.json',
             'c7f021b27272bbac983db2f58d694dddc5dd7de234869ed8ec5d5e64bcf6f0ae'),
    fig2=('figures/squidiff_publication_figures/fig2_values.csv',
          '981b7f1284be867a6a26f3183dbfeefe509061cb1604d4890d864d1737b84991'),
    fig1b=('figures/squidiff_publication_figures/fig1b_values.csv',
           'd47a6699ccac78d582199c913ef0ab486aa955de9ce8f8d055ed8efc9f3202db'),
)
TOL = 1e-9

# (label in the paper, arm key in the Tahoe record, arm key in the sci-Plex record); None = no such entry
ENTRIES = [
    ('Original model', 'base', 'base'),
    ('Rewired model', 'recipe', 'fm_50'),
    ('Training-set ridge', 'ridge_train', 'ridge_train'),
    ('Cross-validated ridge', 'ridge_S7', 'ridge_S7'),
    ('Compound-blind average', 'mean_response', 'mean_response'),
    ('Unaltered vehicle cells', 'control_paste', 'control_paste'),
    ('Plate replicate', 'replicate', None),
]
# part c: (label, dataset, measure, record contrast name, sign) ; sign -1 = stored as "reference minus rewired"
CONTRASTS = [
    ('Gene identity, Tahoe-100M', None, None, None, None),
    ('Rewired model − training-set ridge', 'tahoe', 'de_overlap', 'secondary_jac_recipe_minus_ridge_train', 1),
    ('Rewired model − cross-validated ridge', 'tahoe', 'de_overlap', 'Q2_coprimary_de_recipe_minus_ridgeS7', 1),
    ('Rewired model − compound-blind average', 'tahoe', 'de_overlap', 'secondary_jac_recipe_minus_mean_response', 1),
    ('Gene identity, sci-Plex', None, None, None, None),
    ('Rewired model − training-set ridge', 'sciplex', 'de_overlap', 'secondary_jac_ridge_train_minus_fm50', -1),
    ('Rewired model − cross-validated ridge', 'sciplex', 'de_overlap', 'primary_de_ridgeS7_minus_fm50', -1),
    ('Rewired model − compound-blind average', 'sciplex', 'de_overlap', 'secondary_jac_mean_response_minus_fm50', -1),
    ('Population distance, Tahoe-100M', None, None, None, None),
    ('Rewired model − training-set ridge', 'tahoe', 'energy', 'secondary_en_recipe_minus_ridge_train', 1),
    ('Rewired model − cross-validated ridge', 'tahoe', 'energy', 'secondary_en_recipe_minus_ridge_S7', 1),
    ('Rewired model − compound-blind average', 'tahoe', 'energy', 'secondary_en_recipe_minus_mean_response', 1),
    ('Rewired model − unaltered vehicle cells', 'tahoe', 'energy', 'secondary_en_recipe_minus_control_paste', 1),
]
# which two arms each contrast compares (first minus second, after the sign is applied), for check 1
CONTRAST_ARMS = {
    'secondary_jac_recipe_minus_ridge_train': ('recipe', 'ridge_train'),
    'Q2_coprimary_de_recipe_minus_ridgeS7': ('recipe', 'ridge_S7'),
    'secondary_jac_recipe_minus_mean_response': ('recipe', 'mean_response'),
    'secondary_jac_ridge_train_minus_fm50': ('fm_50', 'ridge_train'),
    'primary_de_ridgeS7_minus_fm50': ('fm_50', 'ridge_S7'),
    'secondary_jac_mean_response_minus_fm50': ('fm_50', 'mean_response'),
    'secondary_en_recipe_minus_ridge_train': ('recipe', 'ridge_train'),
    'secondary_en_recipe_minus_ridge_S7': ('recipe', 'ridge_S7'),
    'secondary_en_recipe_minus_mean_response': ('recipe', 'mean_response'),
    'secondary_en_recipe_minus_control_paste': ('recipe', 'control_paste'),
}

# Printed values the manuscript quotes (checks only, never inputs). "lo-hi" = smallest and largest of the three
# splits, each rounded to two decimals; a single value = all three splits round to it.
EXPECTED_RANGE_2DP = {
    # Results 1 and 2, gene identity
    ('tahoe', 'base', 'de_overlap'): ('0.27', '0.29'),
    ('tahoe', 'recipe', 'de_overlap'): ('0.46', '0.46'),
    ('tahoe', 'ridge_S7', 'de_overlap'): ('0.47', '0.48'),
    ('tahoe', 'mean_response', 'de_overlap'): ('0.50', '0.51'),
    ('tahoe', 'replicate', 'de_overlap'): ('0.48', '0.51'),
    ('sciplex', 'base', 'de_overlap'): ('0.49', '0.51'),
    ('sciplex', 'fm_50', 'de_overlap'): ('0.61', '0.61'),
    ('sciplex', 'ridge_S7', 'de_overlap'): ('0.55', '0.57'),
    ('sciplex', 'mean_response', 'de_overlap'): ('0.53', '0.55'),
    # Results 4, population distance on Tahoe-100M ("from 2.0-2.3 to 0.22-0.24")
    ('tahoe', 'recipe', 'energy'): ('0.22', '0.24'),
}
EXPECTED_BASE_ENERGY_1DP = ('2.0', '2.3')            # Results 4: "from 2.0-2.3"
EXPECTED_CHANCE_2DP = '0.14'                         # Results 1: "the 0.14 expected for 50 genes chosen at random"
EXPECTED_SEED_RANGE_2DP = '0.02'                     # Results 1: "Seeds within a split differed by up to 0.02"
EXPECTED_RIDGE_TRAIN_GAIN_2DP = {'tahoe': ('0.03', '0.03'), 'sciplex': ('0.05', '0.08')}   # Results 2
EXPECTED_VEHICLE_FARTHER_SPLITS = 1                  # Results 2: "no closer ... on any split, and farther on one"


def chance_jaccard(n_genes=200, k=50):
    """Expected Jaccard index of two random sets of k genes out of n_genes: sum over the overlap j of
    P(overlap = j) * j / (2k - j), with the hypergeometric probability; exact fractions, then one division."""
    total = Fraction(0)
    for j in range(k + 1):
        p = Fraction(math.comb(k, j) * math.comb(n_genes - k, k - j), math.comb(n_genes, k))
        total += p * Fraction(j, 2 * k - j)
    return float(total)


def rng2(values, dp=2):
    """('0.27', '0.29'): smallest and largest value, each rounded to dp decimals."""
    return ('%.*f' % (dp, min(values)), '%.*f' % (dp, max(values)))


def main():
    guard(IN, TABLE)
    tahoe = load_json(IN['tahoe'][0])
    sciplex = load_json(IN['sciplex'][0])
    same(tahoe['splits'], SPLITS, 'split order of the Tahoe record')
    for s in SPLITS:
        if s not in sciplex['per_split']:
            die('%s missing in %s' % (s, IN['sciplex'][0]))
    # the caption describes the intervals; these four fields are what it rests on
    same(tahoe['interval'], '97.5% percentile per split, structure-clustered', 'interval type of the Tahoe record')
    same(sciplex['interval'], '97.5% percentile per split, drug-clustered', 'interval type of the sci-Plex record')
    same((tahoe['n_boot'], sciplex['n_boot']), (5000, 5000), 'number of bootstrap replicates')
    same((tahoe['n_seeds'], [len(tahoe['per_split'][s]['metrics']['base']['per_seed']) for s in SPLITS]),
         (3, [3, 3, 3]), 'number of seeds on Tahoe-100M')

    def value(dataset, arm, measure, split):
        """One arm value from a record; None when the record has none (for example gene identity of the
        unaltered vehicle cells, which have no predicted change)."""
        if arm is None:
            return None
        if dataset == 'tahoe':
            block = tahoe['per_split'][split]
            x = block['replicate'] if arm == 'replicate' else block['metrics'][arm]
        else:
            x = sciplex['per_split'][split]['thesis_metrics'][arm]
        v = x.get(measure)
        return v if isinstance(v, (int, float)) and v == v else None

    def contrast(dataset, name, sign, split):
        """(difference, lo, hi) as "rewired model minus reference"."""
        rec = tahoe if dataset == 'tahoe' else sciplex
        c = rec['per_split'][split]['contrasts'][name]
        d, (lo, hi) = float(c['diff']), c['ci']
        return (d, float(lo), float(hi)) if sign == 1 else (-d, -float(hi), -float(lo))

    # ---- check 1: each contrast is the difference of the two arm values shown ----------------------------------
    n1 = 0
    for label, dataset, measure, name, sign in CONTRASTS:
        if name is None:
            continue
        a, b = CONTRAST_ARMS[name]
        for s in SPLITS:
            d = contrast(dataset, name, sign, s)[0]
            close(d, value(dataset, a, measure, s) - value(dataset, b, measure, s), TOL,
                  'check 1, %s on %s' % (name, s))
            n1 += 1

    # ---- check 2: the figure values -------------------------------------------------------------------------------
    chance = chance_jaccard()
    n2 = 0
    fig_dataset = {'Tahoe-100M': 'tahoe', 'sci-Plex': 'sciplex'}
    for key, col_arm, col_measure, col_value in (('fig2', 'source_key', 'metric', 'value'),
                                                 ('fig1b', 'submission', None, 'gene_identity')):
        for r in load_csv(IN[key][0]):
            arm = r[col_arm]
            measure = r[col_measure] if col_measure else 'de_overlap'
            if arm in ('computed', 'chance'):                       # the chance line of the figures
                close(chance, float(r[col_value]), 1e-12, 'check 2, chance level in %s' % IN[key][0])
            else:
                v = value(fig_dataset[r['dataset']], arm, measure, r['split'])
                close(v, float(r[col_value]), TOL, 'check 2, %s: %s %s %s %s'
                      % (IN[key][0], r['dataset'], r['split'], arm, measure))
            n2 += 1

    # ---- check 3: the values the manuscript quotes -----------------------------------------------------------------
    for (dataset, arm, measure), want in EXPECTED_RANGE_2DP.items():
        same(rng2([value(dataset, arm, measure, s) for s in SPLITS]), want,
             'check 3, %s of %s on %s' % (measure, arm, dataset))
    same(rng2([value('tahoe', 'base', 'energy', s) for s in SPLITS], 1), EXPECTED_BASE_ENERGY_1DP,
         'check 3, population distance of the original model on Tahoe-100M')
    same('%.2f' % chance, EXPECTED_CHANCE_2DP, 'check 3, chance level')
    seed_range = max(max(x['de_overlap'] for x in tahoe['per_split'][s]['metrics']['base']['per_seed'])
                     - min(x['de_overlap'] for x in tahoe['per_split'][s]['metrics']['base']['per_seed'])
                     for s in SPLITS)
    same('%.2f' % seed_range, EXPECTED_SEED_RANGE_2DP, 'check 3, largest difference between seeds (original model)')
    for dataset, name, sign in (('tahoe', 'secondary_jac_recipe_minus_ridge_train', 1),
                                ('sciplex', 'secondary_jac_ridge_train_minus_fm50', -1)):
        cs = [contrast(dataset, name, sign, s) for s in SPLITS]
        same(rng2([c[0] for c in cs]), EXPECTED_RIDGE_TRAIN_GAIN_2DP[dataset],
             'check 3, gain of the rewired model over the training-set ridge on %s' % dataset)
        if not all(c[1] > 0 for c in cs):
            die('check 3: the rewired model is not above the training-set ridge on every split of %s' % dataset)
    veh = [contrast('tahoe', 'secondary_en_recipe_minus_control_paste', 1, s) for s in SPLITS]
    if any(c[2] < 0 for c in veh):
        die('check 3: the rewired model is closer than the unaltered vehicle cells on a split')
    same(sum(1 for c in veh if c[1] > 0), EXPECTED_VEHICLE_FARTHER_SPLITS,
         'check 3, number of splits on which the rewired model is farther than the unaltered vehicle cells')

    # ---- the table -----------------------------------------------------------------------------------------------
    long_rows = []

    def keep(part, row, dataset, measure, split, v, lo=None, hi=None, record='', key=''):
        long_rows.append(dict(part=part, row=row, dataset=dataset, measure=measure, split=split, value=v, lo=lo,
                              hi=hi, excludes_zero='' if lo is None else int(lo > 0 or hi < 0),
                              source_record=record, source_key=key))

    names = dict(tahoe='Tahoe-100M', sciplex='sci-Plex')
    short = dict(tahoe='step15_model_analysis.json', sciplex='step14_analysis.json')
    n_t = [tahoe['per_split'][s]['n_conditions'] for s in SPLITS]
    n_s = [sciplex['per_split'][s]['gate']['n_conditions'] for s in SPLITS]
    n_rep = [tahoe['per_split'][s]['replicate']['n_pairs'] for s in SPLITS]
    md = ['Supplementary Table S5 | Gene identity and population distance of the models and the references '
          '(Results 1 and 2; Figs 1b, 2a and 2b). Values are means over the held-out conditions of each compound '
          'split (splits 1 / 0 / 2: %s conditions on Tahoe-100M; %s on sci-Plex); the '
          'generative models are averaged over their seeds within a condition first (three on Tahoe-100M, five on '
          'sci-Plex). a, Gene identity: Jaccard index between the 50 genes with the largest absolute predicted '
          'change and the 50 with the largest absolute measured change, both relative to the matched vehicle mean. '
          'Chance is the expected index of two random sets of 50 genes out of 200. The unaltered vehicle cells have '
          'no predicted change and are not scored on this measure. b, Population distance on Tahoe-100M: energy '
          'distance between the generated or reference population and the measured treated cells; lower is better. '
          'The plate replicate is scored on the conditions measured on both plates (%s pairs). c, Contrasts between '
          'the rewired model and each reference, each specified before the data were scored, with 97.5 %% percentile '
          'bootstrap intervals (5,000 replicates, resampling held-out compounds; chemical structures on '
          'Tahoe-100M); an asterisk marks an interval that excludes zero.'
          % (' / '.join('{:,}'.format(n) for n in n_t), ' / '.join(str(n) for n in n_s),
             ' / '.join(str(n) for n in n_rep)),
          '',
          '**a** | Gene identity (splits 1 / 0 / 2).',
          '']
    rows_a = []
    for label, arm_t, arm_s in ENTRIES:
        vt = [value('tahoe', arm_t, 'de_overlap', s) for s in SPLITS]
        vs = [value('sciplex', arm_s, 'de_overlap', s) for s in SPLITS]
        if not any(v is not None for v in vt + vs):
            continue                                              # the unaltered vehicle cells: no gene identity
        rows_a.append([label, triple(vt), triple(vs)])
        for dataset, arm, vals in (('tahoe', arm_t, vt), ('sciplex', arm_s, vs)):
            for s, v in zip(SPLITS, vals):
                if v is not None:
                    keep('a', label, names[dataset], 'gene identity', s, v, record=short[dataset], key=arm)
    rows_a.append(['Chance', num(chance), num(chance)])
    keep('a', 'Chance', 'both', 'gene identity', 'all', chance, record='computed',
         key='expected Jaccard, two random sets of 50 of 200')
    md += table(['Entry', 'Tahoe-100M', 'sci-Plex'], rows_a)

    md += ['', '**b** | Population distance on Tahoe-100M.', '']
    rows_b = []
    for label, arm_t, arm_s in ENTRIES:
        vt = [value('tahoe', arm_t, 'energy', s) for s in SPLITS]
        rows_b.append([label] + [num(v) for v in vt])
        for s, v in zip(SPLITS, vt):
            keep('b', label, 'Tahoe-100M', 'population distance', s, v, record=short['tahoe'], key=arm_t)
    md += table(['Entry'] + SPLIT_HEAD, rows_b)

    md += ['', '**c** | Contrasts (point estimate and 97.5 % interval).', '']
    rows_c = []
    group = ''
    for label, dataset, measure, name, sign in CONTRASTS:
        if name is None:
            group = label
            rows_c.append(['**%s**' % label, '', '', ''])
            continue
        cs = [contrast(dataset, name, sign, s) for s in SPLITS]
        rows_c.append([label] + [interval(*c) for c in cs])
        for s, c in zip(SPLITS, cs):
            keep('c', '%s: %s' % (group, label), names[dataset],
                 'gene identity' if measure == 'de_overlap' else 'population distance', s, c[0], c[1], c[2],
                 record=short[dataset], key=name + (' (negated)' if sign == -1 else ''))
    md += table(['Contrast'] + SPLIT_HEAD, rows_c)

    md += ['',
           'Sources: %s (sha256 %s; Tahoe-100M) and %s (sha256 %s; sci-Plex). The values of parts a and b equal '
           'those the figures were drawn from (fig1b_values.csv, fig2_values.csv). The sci-Plex record stores each '
           'contrast as reference minus rewired model; it is given here with the sign reversed. Assembled by '
           'make_supp_table_s5.py.'
           % (short['tahoe'], IN['tahoe'][1][:16], short['sciplex'], IN['sciplex'][1][:16])]
    write_outputs(TABLE, long_rows, md)
    print('[checks] 1 (%d contrasts = difference of the two arm values), 2 (%d figure values), 3 (values quoted in '
          'Results 1, 2 and 4) all passed' % (n1, n2))


if __name__ == '__main__':
    main()
