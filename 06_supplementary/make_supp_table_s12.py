#!/usr/bin/env python3
"""
make_supp_table_s12.py - Supplementary Table S12: the rewired model against the original model on Tahoe-100M
(Results 4; Fig. 4b).

Purpose
-------
Results 4 states what the rewiring changed on Tahoe-100M (gene identity from 0.27-0.29 to 0.46, response correlation
raised by 0.25-0.29, population distance cut about tenfold, from 2.0-2.3 to 0.22-0.24) and cites Supplementary
Table S12; the caption of Fig. 4b cites it for the values of the figure. This script assembles the table from the
committed Tahoe analysis record, so that every number traces to the record and is rounded once.

Two parts:
  a  the three measures for the original and the rewired model on each split, and the pre-specified difference
     between the two with its 97.5 % bootstrap interval;
  b  the same measures for each of the three training seeds (Results 1 quotes the spread between seeds).

Logic
-----
- Inputs (committed records only; see supp_table_common.guard):
      tahoe_results/step15_model_analysis.json   per_split/<split>/metrics/<arm>/{de_overlap, dpc_mean, energy,
                                                 per_seed}; per_split/<split>/contrasts/<name>; inputs/<split>/<arm>
      figures/squidiff_publication_figures/fig4_values.csv   the values Fig. 4 was drawn from (cross-check only)
  Record names -> names in the paper: base = original model; recipe = rewired model; de_overlap = gene identity;
  dpc_mean = response correlation (mean over conditions); energy = population distance.
- The per-seed values of the record are stored in the order the seeds were loaded; the record's own list of
  inputs (tags ending in seed_0, seed_1, seed_2) is checked to be in that order.
- Checks, all before anything is written (a failure stops the script):
    1. Each difference equals the rewired value minus the original value (1e-9), and each model value equals the
       mean of its three per-seed values (1e-9).
    2. Every value of panel b of fig4_values.csv equals the value shown here (1e-9).
    3. The values quoted in the manuscript reproduce at their printed precision (EXPECTED below; checks only,
       never inputs to a computation).
- Rounding: three decimals, each value rounded once from full precision.
- Outputs (never overwritten): supplementary_tables/supp_table_S12.csv and .md (see supp_table_common).

Run from the research worktree root (supp_table_common.py must be in the same folder):
    python3 make_supp_table_s12.py
"""
import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import (SPLITS, SPLIT_HEAD, close, die, guard, interval, load_csv, load_json, num, same,
                               table, triple, write_outputs)

TABLE = 'S12'
IN = dict(
    tahoe=('tahoe_results/step15_model_analysis.json',
           '5fd700b5a030cf5a8ffea57e9a9b2ad840aa32fa9948b32754b9742362a1165e'),
    fig4=('figures/squidiff_publication_figures/fig4_values.csv',
          'bfb58226d3cac9ea77d203a83f9ce91349d594b26a8763535dacd692ac1c7fee'),
)
TOL = 1e-9
# (label in the paper, key in the record, contrast "rewired minus original" in the record)
MEASURES = [
    ('Gene identity', 'de_overlap', 'Q1_primary_de_recipe_minus_base'),
    ('Response correlation', 'dpc_mean', 'secondary_dpc_recipe_minus_base'),
    ('Population distance', 'energy', 'secondary_en_recipe_minus_base'),
]
MODELS = [('Original model', 'base'), ('Rewired model', 'recipe')]
SEEDS = [0, 1, 2]

# Printed values the manuscript quotes (checks only, never inputs): smallest and largest of the three splits.
EXPECTED = {
    ('base', 'de_overlap', 2): ('0.27', '0.29'),       # Results 4: "from 0.27-0.29 to 0.46 on every split"
    ('recipe', 'de_overlap', 2): ('0.46', '0.46'),
    ('base', 'energy', 1): ('2.0', '2.3'),             # "from 2.0-2.3 to 0.22-0.24"
    ('recipe', 'energy', 2): ('0.22', '0.24'),
}
EXPECTED_DPC_GAIN_2DP = ('0.25', '0.29')               # "raised response correlation by 0.25-0.29"
EXPECTED_SEED_RANGE_2DP = '0.02'                       # Results 1: "Seeds within a split differed by up to 0.02"


def main():
    guard(IN, TABLE)
    rec = load_json(IN['tahoe'][0])
    same(rec['splits'], SPLITS, 'split order of the record')
    same(rec['interval'], '97.5% percentile per split, structure-clustered', 'interval type of the record')
    same((rec['n_boot'], rec['n_seeds']), (5000, 3), 'bootstrap replicates and seeds of the record')
    metrics = dict((s, rec['per_split'][s]['metrics']) for s in SPLITS)
    for s in SPLITS:                                     # per-seed values are in seed order 0, 1, 2
        for _, arm in MODELS:
            tags = [x['tag'] for x in rec['inputs'][s][arm]]
            same([t.rsplit('_', 1)[-1] for t in tags], [str(k) for k in SEEDS], 'seed order of %s on %s' % (arm, s))
            same(len(metrics[s][arm]['per_seed']), len(SEEDS), 'number of per-seed values of %s on %s' % (arm, s))

    def contrast(name, s):
        c = rec['per_split'][s]['contrasts'][name]
        return float(c['diff']), float(c['ci'][0]), float(c['ci'][1])

    # ---- check 1 --------------------------------------------------------------------------------------------------
    n1 = 0
    for label, key, name in MEASURES:
        for s in SPLITS:
            close(contrast(name, s)[0], metrics[s]['recipe'][key] - metrics[s]['base'][key], TOL,
                  'check 1, %s on %s' % (name, s))
            for _, arm in MODELS:
                seeds = [x[key] for x in metrics[s][arm]['per_seed']]
                close(metrics[s][arm][key], sum(seeds) / len(seeds), TOL,
                      'check 1, %s of %s on %s against its per-seed values' % (key, arm, s))
            n1 += 3

    # ---- check 2: panel b of the figure values ------------------------------------------------------------------------
    fig_measure = {'Gene identity': 'de_overlap', 'Response correlation': 'dpc_mean', 'Population distance': 'energy'}
    fig_model = {'Original': 'base', 'Rewired': 'recipe'}
    fig_split = {'Split 1': 'drug_split_1', 'Split 0': 'drug_split_0', 'Split 2': 'drug_split_2'}
    n2 = 0
    for r in load_csv(IN['fig4'][0]):
        if r['panel'] != 'b':
            continue
        key = fig_measure[r['facet'].split('\n')[0].strip()]
        arm = fig_model[r['x'].split('\n')[0].strip()]
        close(metrics[fig_split[r['split']]][arm][key], float(r['point']), TOL,
              'check 2, fig4_values.csv panel b: %s %s %s' % (key, arm, r['split']))
        n2 += 1
    same(n2, len(MEASURES) * len(MODELS) * len(SPLITS), 'check 2, number of panel-b values in fig4_values.csv')

    # ---- check 3: the values the manuscript quotes ----------------------------------------------------------------------
    for (arm, key, dp), want in EXPECTED.items():
        vals = [metrics[s][arm][key] for s in SPLITS]
        same(('%.*f' % (dp, min(vals)), '%.*f' % (dp, max(vals))), want, 'check 3, %s of %s' % (key, arm))
    gains = [contrast('secondary_dpc_recipe_minus_base', s)[0] for s in SPLITS]
    same(('%.2f' % min(gains), '%.2f' % max(gains)), EXPECTED_DPC_GAIN_2DP, 'check 3, gain in response correlation')
    spread = max(max(x['de_overlap'] for x in metrics[s]['base']['per_seed'])
                 - min(x['de_overlap'] for x in metrics[s]['base']['per_seed']) for s in SPLITS)
    same('%.2f' % spread, EXPECTED_SEED_RANGE_2DP, 'check 3, largest difference between seeds (original model)')
    for label, key, name in MEASURES:                    # every difference is resolved, in the stated direction
        for s in SPLITS:
            d, lo, hi = contrast(name, s)
            if not ((lo > 0) if key != 'energy' else (hi < 0)):
                die('check 3: %s on %s is not resolved in the direction the text states' % (name, s))

    # ---- the table ---------------------------------------------------------------------------------------------------
    long_rows = []

    def keep(part, measure, row, split, seed, v, lo=None, hi=None, key=''):
        long_rows.append(dict(part=part, measure=measure, row=row, split=split, seed=seed, value=v, lo=lo, hi=hi,
                              excludes_zero='' if lo is None else int(lo > 0 or hi < 0),
                              source_record='step15_model_analysis.json', source_key=key))

    n_cond = [rec['per_split'][s]['n_conditions'] for s in SPLITS]
    md = ['Supplementary Table S12 | The rewired model against the original model on Tahoe-100M (Results 4; '
          'Fig. 4b). Gene identity: Jaccard index between the 50 genes with the largest absolute predicted change '
          'and the 50 with the largest absolute measured change. Response correlation: Pearson correlation between '
          'the predicted and the measured mean change over the 50 genes with the largest absolute measured change. '
          'Population distance: energy distance between the generated and the measured treated cells; lower is '
          'better. a, Mean over the held-out conditions of each compound split (%s conditions on splits 1, 0 and '
          '2), each model averaged over its three seeds within a condition first, and the difference between the '
          'two models, specified before the data were scored, with its 97.5 %% percentile bootstrap interval '
          '(5,000 replicates, resampling held-out chemical structures); an asterisk marks an interval that excludes '
          'zero. b, The same means for each training seed (seeds 0 / 1 / 2).'
          % ('{:,}, {:,} and {:,}'.format(*n_cond)),
          '',
          '**a** | Values and differences.',
          '']
    rows_a = []
    for label, key, name in MEASURES:
        for i, (mlabel, arm) in enumerate(MODELS):
            vals = [metrics[s][arm][key] for s in SPLITS]
            rows_a.append([label if i == 0 else '', mlabel] + [num(v) for v in vals])
            for s, v in zip(SPLITS, vals):
                keep('a', label, mlabel, s, '', v, key='metrics/%s/%s' % (arm, key))
        cs = [contrast(name, s) for s in SPLITS]
        rows_a.append(['', 'Rewired − original'] + [interval(*c) for c in cs])
        for s, c in zip(SPLITS, cs):
            keep('a', label, 'Rewired − original', s, '', c[0], c[1], c[2], key='contrasts/%s' % name)
    md += table(['Measure', 'Entry'] + SPLIT_HEAD, rows_a)

    md += ['', '**b** | Values per seed (seeds 0 / 1 / 2).', '']
    rows_b = []
    for label, key, name in MEASURES:
        for i, (mlabel, arm) in enumerate(MODELS):
            cells = []
            for s in SPLITS:
                seeds = [x[key] for x in metrics[s][arm]['per_seed']]
                cells.append(triple(seeds))
                for k, v in zip(SEEDS, seeds):
                    keep('b', label, mlabel, s, k, v, key='metrics/%s/per_seed[%d]/%s' % (arm, k, key))
            rows_b.append([label if i == 0 else '', mlabel] + cells)
    md += table(['Measure', 'Entry'] + SPLIT_HEAD, rows_b)

    md += ['',
           'Source: step15_model_analysis.json (sha256 %s). The values of part a equal those Fig. 4b was drawn from '
           '(fig4_values.csv, panel b). Assembled by make_supp_table_s12.py.' % IN['tahoe'][1][:16]]
    write_outputs(TABLE, long_rows, md)
    print('[checks] 1 (%d differences and seed means), 2 (%d values of Fig. 4b), 3 (values quoted in Results 1 and 4) '
          'all passed' % (n1, n2))


if __name__ == '__main__':
    main()
