#!/usr/bin/env python3
"""
make_supp_table_s13.py - Supplementary Table S13: classifier-free guidance on Tahoe-100M and on sci-Plex (Results 4).

Purpose
-------
Results 4 states what classifier-free guidance (weight 2) did to gene identity, population distance and response
correlation on Tahoe-100M and on sci-Plex, and cites Supplementary Table S13. This script assembles the table from
the committed analysis records, so that every number traces to a record and is rounded once.

What was compared is not the same on the two datasets, and the table keeps them apart:
  a  Tahoe-100M. The guided variant (trained with condition dropout, sampled with guidance weight 2) against the
     rewired model. The two are separately trained models, so the difference combines condition dropout and guidance.
  b  sci-Plex. (i) The same five dropout-trained models sampled with guidance weight 2 and without guidance
     (weight 1): this isolates guidance. (ii) The dropout-trained models without guidance against the rewired model:
     this isolates condition dropout.

Logic
-----
- Inputs (committed records only; see supp_table_common.guard):
      tahoe_results/step15_model_analysis.json   Tahoe-100M, three splits, 97.5 % intervals.
            per_split/<split>/metrics/<arm>/{de_overlap, dpc_mean, energy}; per_split/<split>/contrasts/Q3_*
      leakage_results/step12_analysis.json       sci-Plex development split (drug_split_1), 95 % intervals.
            results/arm_means/<arm>/{jac, dpc, en}; results/primary/*; results/secondary/*
      leakage_results/step12b_analysis.json      sci-Plex confirmation splits (0 and 2), 97.5 % intervals.
            per_split/<split>/arm_means/<arm>/{jac, dpc, en}; per_split/<split>/{primary, secondary, exploratory}
      leakage_results/step14_analysis.json       cross-check only: ties this table to Supplementary Tables S5 and S6.
  Record names -> names in the paper:
      Tahoe-100M: recipe = rewired model; guidance_w2 = guided variant; de_overlap = gene identity;
                  dpc_mean = response correlation (mean over conditions); energy = population distance.
      sci-Plex:   ref = rewired model; w1 = dropout-trained models sampled without guidance (guidance weight 1 is the
                  unguided model); w2 = the same models with guidance weight 2; jac, dpc, en = the three measures.
- Which stored comparison is used:
      Tahoe-100M           contrasts Q3_{jac,dpc,en}_guidance_w2_minus_recipe.
      sci-Plex, guidance   gene identity and population distance: the two gated comparisons
                           (primary/de_overlap_w2_minus_w1, primary/energy_w2_minus_w1) on every split. On the
                           development split the record holds these two differences a second time, in the sweep
                           over guidance weights (secondary/w2_minus_w1), with the same point estimate and an
                           interval from a separate bootstrap draw; the gated one is used. Response correlation:
                           secondary/w2_minus_w1/dpc on the development split. On the confirmation splits the
                           record holds no interval for it; the difference of the two arm means is given and marked
                           with a dagger. Check 2 shows that this difference is the statistic the record itself
                           reports for its other comparisons.
      sci-Plex, dropout    secondary/w1_minus_ref (development) and secondary/w1_minus_step11_reference
                           (confirmation).
- Checks, all before anything is written (a failure stops the script):
    1. Tahoe-100M: each difference equals the guided value minus the rewired value (1e-9).
    2. sci-Plex: each stored difference equals the difference of the two arm means (1e-9); guidance weight 1
       reproduces the unguided model (the records' own identity comparison, 1e-6); on the confirmation splits the
       dagger difference equals the one obtained from two stored pairs of comparisons, (w3 - w1) - (w3 - w2) and
       (w2h - w1) - (w2h - w2) (1e-9); the inputs of the records carry the guidance weights 1 and 2 and five seeds.
    3. Cross-check with the record behind Supplementary Tables S5 and S6: the guided models are the same on every
       split (1e-9) and the rewired model is the same on the confirmation splits (1e-9); on the development split
       the rewired model of this analysis is a separate set of models, and the two gene-identity values printed in
       the caption must differ at three decimals.
    4. The statements of Results 4 about Tahoe-100M (gene identity raised on two of three splits, population
       distance increased and response correlation lowered on all three), and what the table shows for sci-Plex
       (gene identity raised on all three splits, population distance increased on split 0 only, response
       correlation higher with guidance on all three), hold as intervals excluding zero or as signs.
- Rounding: values three decimals; differences three decimals, except gene identity (four decimals, because the
  effects of guidance on it lie in the third decimal). Each value is rounded once from full precision.
- Outputs (never overwritten): supplementary_tables/supp_table_S13.csv and .md (see supp_table_common).

Run from the research worktree root (supp_table_common.py must be in the same folder):
    python3 make_supp_table_s13.py
"""
import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import (SPLITS, SPLIT_HEAD, close, die, guard, interval, load_json, num, same, table,
                               write_outputs)

TABLE = 'S13'
IN = dict(
    tahoe=('tahoe_results/step15_model_analysis.json',
           '5fd700b5a030cf5a8ffea57e9a9b2ad840aa32fa9948b32754b9742362a1165e'),
    dev=('leakage_results/step12_analysis.json',
         'c3d1c642dca278d500d553c68d558e2e64c12dafe153339fcd49362fb540fa82'),
    conf=('leakage_results/step12b_analysis.json',
          '6cadb30a2bb7bfd56e5d4be43e770bef0fb3f331b08402146b0f4d303edfb328'),
    s14=('leakage_results/step14_analysis.json',
         'c7f021b27272bbac983db2f58d694dddc5dd7de234869ed8ec5d5e64bcf6f0ae'),
)
TOL = 1e-9
DEV, CONF = 'drug_split_1', ['drug_split_0', 'drug_split_2']
DAGGER = '†'
# (label in the paper, Tahoe key, Tahoe contrast, sci-Plex key, key in the Step 14 record, decimals of differences)
MEASURES = [
    ('Gene identity', 'de_overlap', 'Q3_jac_guidance_w2_minus_recipe', 'jac', 'de_overlap', 4),
    ('Response correlation', 'dpc_mean', 'Q3_dpc_guidance_w2_minus_recipe', 'dpc', 'dpc_mean', 3),
    ('Population distance', 'energy', 'Q3_en_guidance_w2_minus_recipe', 'en', 'energy', 3),
]
L_REWIRED = 'Rewired model'
L_DROPOUT = 'Trained with condition dropout, sampled without guidance'
L_GUIDED = 'Trained with condition dropout, sampled with guidance (weight 2)'
L_T_DIFF = 'Guided − rewired model'
L_G_DIFF = 'Guidance − no guidance'
L_D_DIFF = 'Condition dropout − rewired model'


def triplet(c):
    """(difference, lower, upper) of one stored comparison."""
    return float(c['diff']), float(c['ci'][0]), float(c['ci'][1])


def above(c):
    return c[1] > 0


def below(c):
    return c[2] < 0


def main():
    guard(IN, TABLE)
    tah, dev, conf, s14 = (load_json(IN[k][0]) for k in ('tahoe', 'dev', 'conf', 's14'))

    # ---- the records are the ones this table was designed on ---------------------------------------------------------
    same(tah['splits'], SPLITS, 'split order of the Tahoe-100M record')
    same(tah['interval'], '97.5% percentile per split, structure-clustered', 'interval of the Tahoe-100M record')
    same((tah['n_boot'], tah['n_seeds']), (5000, 3), 'bootstrap replicates and seeds of the Tahoe-100M record')
    same((dev['split'], dev['interval'], dev['n_boot']), (DEV, '95% percentile, drug-clustered', 5000),
         'split, interval and replicates of the sci-Plex development record')
    same((conf['interval'], conf['n_boot'], sorted(conf['per_split'])),
         ('97.5% percentile per split, drug-clustered', 5000, sorted(CONF)),
         'interval, replicates and splits of the sci-Plex confirmation record')
    same((dev['statistic'], conf['statistic']), ('mean over conditions of seed-mean per-condition values',) * 2,
         'statistic of the sci-Plex records')
    # the outcome each record assigned by its rule fixed in advance (not printed in the table)
    same((dev['results']['primary']['category'], conf['verdict']), ('GUIDANCE HELPS', 'TRADE-OFF'),
         'outcome labels of the sci-Plex records')

    tmet = dict((s, tah['per_split'][s]['metrics']) for s in SPLITS)
    tcon = dict((s, tah['per_split'][s]['contrasts']) for s in SPLITS)

    def sci(split):
        """Arm means, guidance comparisons, dropout comparisons and interval level of one sci-Plex split.
        A guidance comparison without a stored interval is None."""
        if split == DEV:
            r = dev['results']
            g = dict(jac=triplet(r['primary']['de_overlap_w2_minus_w1']),
                     en=triplet(r['primary']['energy_w2_minus_w1']),
                     dpc=triplet(r['secondary']['w2_minus_w1']['dpc']))
            d = dict((k, triplet(r['secondary']['w1_minus_ref'][k])) for k in ('jac', 'dpc', 'en'))
            return r['arm_means'], g, d, '95'
        v = conf['per_split'][split]
        g = dict(jac=triplet(v['primary']['de_overlap_w2_minus_w1']),
                 en=triplet(v['primary']['energy_w2_minus_w1']), dpc=None)
        d = dict((k, triplet(v['secondary']['w1_minus_step11_reference'][k])) for k in ('jac', 'dpc', 'en'))
        return v['arm_means'], g, d, '97.5'

    # ---- check 1: Tahoe-100M differences ----------------------------------------------------------------------------------
    n1 = 0
    for label, tkey, tname, skey, key14, dp in MEASURES:
        for s in SPLITS:
            close(triplet(tcon[s][tname])[0], tmet[s]['guidance_w2'][tkey] - tmet[s]['recipe'][tkey], TOL,
                  'check 1, %s on %s' % (tname, s))
            n1 += 1

    # ---- check 2: sci-Plex differences, identity of weight 1, the dagger differences, the inputs --------------------------
    n2 = 0
    dagger = {}
    for s in SPLITS:
        am, g, d, level = sci(s)
        for label, tkey, tname, skey, key14, dp in MEASURES:
            if g[skey] is not None:
                close(g[skey][0], am['w2'][skey] - am['w1'][skey], TOL, 'check 2, guidance on %s, %s' % (skey, s))
                n2 += 1
            close(d[skey][0], am['w1'][skey] - am['ref'][skey], TOL, 'check 2, dropout on %s, %s' % (skey, s))
            close(am['wnone'][skey], am['w1'][skey], 1e-6, 'check 2, weight 1 against the unguided model, %s, %s'
                  % (skey, s))
            n2 += 2
        ident = (dev['results'] if s == DEV else conf['per_split'][s])['secondary']['w1_minus_wnone_identity']
        for k in ('jac', 'en'):
            close(float(ident[k]['diff']), 0.0, 1e-6, 'check 2, identity comparison of the record, %s, %s' % (k, s))
            n2 += 1
    r = dev['results']
    for k, name in (('jac', 'de_overlap_w2_minus_w1'), ('en', 'energy_w2_minus_w1')):
        close(float(r['secondary']['w2_minus_w1'][k]['diff']), float(r['primary'][name]['diff']), TOL,
              'check 2, the two copies of the %s difference on the development split' % k)
        n2 += 1
    for s in CONF:                                        # the dagger differences (response correlation)
        v = conf['per_split'][s]
        am, ex = v['arm_means'], v['exploratory']
        dagger[s] = am['w2']['dpc'] - am['w1']['dpc']
        close(dagger[s], float(ex['w3_minus_w1']['dpc']['diff']) - float(ex['w3_minus_w2']['dpc']['diff']), TOL,
              'check 2, dagger difference on %s against the weight-3 comparisons' % s)
        close(dagger[s], float(ex['w2h_minus_w1']['dpc']['diff']) - float(ex['w2h_minus_w2']['dpc']['diff']), TOL,
              'check 2, dagger difference on %s against the half-trajectory comparisons' % s)
        n2 += 2
    for arm, w in (('w1', 1.0), ('w2', 2.0), ('ref', None)):
        lists = [dev['inputs'][arm]] + [conf['inputs']['%s/%s' % (s, arm)] for s in CONF]
        for one in lists:
            same(len(one), 5, 'check 2, number of seeds of %s' % arm)
            same(sorted(set(x['guidance_w'] for x in one), key=str), [w], 'check 2, guidance weight of %s' % arm)
            n2 += 2

    # ---- check 3: the same models as in Supplementary Tables S5 and S6 ---------------------------------------------------
    n3 = 0
    t14 = dict((s, s14['per_split'][s]['thesis_metrics']) for s in SPLITS)
    for s in SPLITS:
        am = sci(s)[0]
        for label, tkey, tname, skey, key14, dp in MEASURES:
            close(am['w2'][skey], t14[s]['guidance_w2'][key14], TOL, 'check 3, guided models, %s, %s' % (skey, s))
            n3 += 1
            if s in CONF:
                close(am['ref'][skey], t14[s]['fm_50'][key14], TOL, 'check 3, rewired model, %s, %s' % (skey, s))
                n3 += 1
    ref_dev, s5_dev = num(sci(DEV)[0]['ref']['jac']), num(t14[DEV]['fm_50']['de_overlap'])
    if ref_dev == s5_dev:
        die('check 3: the caption says the two rewired models of the development split differ at three decimals '
            '(%s and %s)' % (ref_dev, s5_dev))
    n_sci = [s14['per_split'][s]['gate']['n_conditions'] for s in SPLITS]
    same(n_sci[0], dev['n_conditions'], 'check 3, conditions of the development split in the two sci-Plex records')
    n_tah = [tah['per_split'][s]['n_conditions'] for s in SPLITS]

    # ---- check 4: what the text and the table state ----------------------------------------------------------------------
    t = dict((m[3], [triplet(tcon[s][m[2]]) for s in SPLITS]) for m in MEASURES)
    same([above(c) for c in t['jac']], [True, False, True], 'check 4, Tahoe-100M gene identity raised on splits 1 and 2')
    same([above(c) for c in t['en']], [True] * 3, 'check 4, Tahoe-100M population distance increased on all splits')
    same([below(c) for c in t['dpc']], [True] * 3, 'check 4, Tahoe-100M response correlation lowered on all splits')
    g = dict((s, sci(s)[1]) for s in SPLITS)
    same([above(g[s]['jac']) for s in SPLITS], [True] * 3, 'check 4, sci-Plex gene identity raised on all splits')
    same([above(g[s]['en']) for s in SPLITS], [False, True, False],
         'check 4, sci-Plex population distance increased on split 0 only')
    same(above(g[DEV]['dpc']), True, 'check 4, sci-Plex response correlation raised on the development split')
    same([dagger[s] > 0 for s in CONF], [True] * 2,
         'check 4, sci-Plex response correlation higher with guidance on the confirmation splits')

    # ---- the table ---------------------------------------------------------------------------------------------------
    long_rows = []

    def keep(part, dataset, measure, row, split, v, lo=None, hi=None, level='', record='', key='', note=''):
        long_rows.append(dict(part=part, dataset=dataset, measure=measure, row=row, split=split, value=v, lo=lo,
                              hi=hi, interval_level=level,
                              excludes_zero='' if lo is None else int(lo > 0 or hi < 0),
                              source_record=record, source_key=key, note=note))

    md = ['Supplementary Table S13 | Classifier-free guidance (Results 4). The guided variant is the rewired model '
          'trained with condition dropout (the compound input replaced by zeros for a random 10 %% of the cells of '
          'each training batch) and sampled with guidance weight 2. Gene identity, response correlation and '
          'population distance are defined as in Supplementary Table S12; values are means over the held-out '
          'conditions of each compound split, each model averaged over its seeds within a condition first (three '
          'seeds on Tahoe-100M, five on sci-Plex). a, Tahoe-100M (%s conditions on splits 1, 0 and 2): the guided '
          'variant against the rewired model. The two were trained separately, so their difference combines the '
          'effect of condition dropout with that of guidance. b, sci-Plex (%s conditions): the same five '
          'dropout-trained models sampled with and without guidance, which isolates guidance, and the '
          'dropout-trained models sampled without guidance against the rewired model, which isolates condition '
          'dropout. Differences carry percentile bootstrap intervals (5,000 replicates, resampling held-out '
          'compounds; chemical structures on Tahoe-100M) at the 97.5 %% level, and at the 95 %% level on the '
          'sci-Plex development split, as fixed before each analysis; an asterisk marks an interval that excludes '
          'zero. The comparisons on gene identity and on population distance were specified before the data were '
          'scored; response correlation was a reported secondary measure. %s Difference of the two means; the '
          'analysis of the confirmation splits computed no interval for this comparison. Differences in gene '
          'identity are given to four decimals because they lie in the third decimal. On the sci-Plex development '
          'split, the rewired model of this comparison was a second set of five models; like the dropout-trained '
          'models, it was trained without the three training compounds that the learned-embedding comparison could '
          'not encode (Supplementary Table S10). Its values differ in the third decimal from those of the rewired '
          'model in Supplementary Tables S5 and S6 (gene identity %s against %s).'
          % ('{:,}, {:,} and {:,}'.format(*n_tah), '{:,}, {:,} and {:,}'.format(*n_sci), DAGGER, ref_dev, s5_dev),
          '',
          '**a** | Tahoe-100M.',
          '']
    rows_a = []
    for label, tkey, tname, skey, key14, dp in MEASURES:
        for i, (rlabel, arm) in enumerate(((L_REWIRED, 'recipe'), (L_GUIDED, 'guidance_w2'))):
            vals = [tmet[s][arm][tkey] for s in SPLITS]
            rows_a.append([label if i == 0 else '', rlabel] + [num(v) for v in vals])
            for s, v in zip(SPLITS, vals):
                keep('a', 'Tahoe-100M', label, rlabel, s, v, record='step15_model_analysis.json',
                     key='per_split/%s/metrics/%s/%s' % (s, arm, tkey))
        cs = [triplet(tcon[s][tname]) for s in SPLITS]
        rows_a.append(['', L_T_DIFF] + [interval(c[0], c[1], c[2], dp) for c in cs])
        for s, c in zip(SPLITS, cs):
            keep('a', 'Tahoe-100M', label, L_T_DIFF, s, c[0], c[1], c[2], '97.5', 'step15_model_analysis.json',
                 'per_split/%s/contrasts/%s' % (s, tname))
    md += table(['Measure', 'Entry'] + SPLIT_HEAD, rows_a)

    md += ['', '**b** | sci-Plex.', '']
    rows_b = []
    for label, tkey, tname, skey, key14, dp in MEASURES:
        per = dict((s, sci(s)) for s in SPLITS)
        for i, (rlabel, arm) in enumerate(((L_REWIRED, 'ref'), (L_DROPOUT, 'w1'), (L_GUIDED, 'w2'))):
            vals = [per[s][0][arm][skey] for s in SPLITS]
            rows_b.append([label if i == 0 else '', rlabel] + [num(v) for v in vals])
            for s, v in zip(SPLITS, vals):
                rec = 'step12_analysis.json' if s == DEV else 'step12b_analysis.json'
                key = ('results/arm_means/%s/%s' % (arm, skey) if s == DEV
                       else 'per_split/%s/arm_means/%s/%s' % (s, arm, skey))
                keep('b', 'sci-Plex', label, rlabel, s, v, record=rec, key=key)
        cells = []
        for s in SPLITS:
            am, g1, d1, level = per[s]
            rec = 'step12_analysis.json' if s == DEV else 'step12b_analysis.json'
            if g1[skey] is None:                         # response correlation on the confirmation splits
                cells.append(num(dagger[s], dp, True) + DAGGER)
                keep('b', 'sci-Plex', label, L_G_DIFF, s, dagger[s], record=rec,
                     key='per_split/%s/arm_means/w2/dpc minus per_split/%s/arm_means/w1/dpc' % (s, s),
                     note='difference of the two means; no interval in the record')
            else:
                c = g1[skey]
                cells.append(interval(c[0], c[1], c[2], dp))
                if s == DEV:
                    key = ('results/secondary/w2_minus_w1/dpc' if skey == 'dpc' else 'results/primary/%s_w2_minus_w1'
                           % ('de_overlap' if skey == 'jac' else 'energy'))
                else:
                    key = 'per_split/%s/primary/%s_w2_minus_w1' % (s, 'de_overlap' if skey == 'jac' else 'energy')
                keep('b', 'sci-Plex', label, L_G_DIFF, s, c[0], c[1], c[2], level, rec, key)
        rows_b.append(['', L_G_DIFF] + cells)
        cells = []
        for s in SPLITS:
            am, g1, d1, level = per[s]
            c = d1[skey]
            cells.append(interval(c[0], c[1], c[2], dp))
            rec = 'step12_analysis.json' if s == DEV else 'step12b_analysis.json'
            key = ('results/secondary/w1_minus_ref/%s' % skey if s == DEV
                   else 'per_split/%s/secondary/w1_minus_step11_reference/%s' % (s, skey))
            keep('b', 'sci-Plex', label, L_D_DIFF, s, c[0], c[1], c[2], level, rec, key)
        rows_b.append(['', L_D_DIFF] + cells)
    md += table(['Measure', 'Entry'] + SPLIT_HEAD, rows_b)

    md += ['',
           'Sources: step15_model_analysis.json (sha256 %s; Tahoe-100M), step12_analysis.json (sha256 %s; sci-Plex '
           'development split) and step12b_analysis.json (sha256 %s; sci-Plex confirmation splits). On sci-Plex, '
           'the guided models are those of Supplementary Table S6, and the rewired model of the confirmation '
           'splits is that of Supplementary Tables S5 and S6 (step14_analysis.json, sha256 %s). Assembled by '
           'make_supp_table_s13.py.'
           % (IN['tahoe'][1][:16], IN['dev'][1][:16], IN['conf'][1][:16], IN['s14'][1][:16])]
    write_outputs(TABLE, long_rows, md)
    print('[checks] 1 (%d Tahoe-100M differences), 2 (%d sci-Plex differences, identities and inputs), 3 (%d values '
          'shared with Tables S5 and S6), 4 (statements of the text and of the table) all passed' % (n1, n2, n3))


if __name__ == '__main__':
    main()
