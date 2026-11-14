#!/usr/bin/env python3
"""
make_supp_table_s10.py - Supplementary Table S10: interventions on the compound input of the rewired model on
sci-Plex (Results 3; Fig. 3b and 3c; Supplementary Note 1).

Purpose
-------
Results 3 reports that five ways of giving the generative model more or different compound information produced no
confirmed gain, and cites this table for the numbers and for the description of the variants. The numbers come
from six committed records of four pre-registered experiments and two exploratory analyses (and from the record
that one of them supersedes, read for checks only). This script assembles
them in one table, so that each number traces to a record and is rounded once, and it keeps the pre-registered
and the exploratory parts apart.

Five parts:
  a  the seven contrasts of the interventions against the rewired model on the development split
     (the six values of Fig. 3b, and the labels given in place of the fingerprint);
  b  the two attention modules that were not taken further: every contrast specified for them;
  c  cross-attention and its control on the three splits (the nine values of Fig. 3c, the remaining contrasts
     and the values the percentages in the text are computed from);
  d  linear models given the fingerprint, the mechanism-of-action labels or the targets, with the fold
     assignment of the paper and over 200 random fold assignments (exploratory; the numbers of
     Supplementary Note 1);
  e  a feedforward network given the features of the linear references (exploratory).
The description of the variants is printed under the caption. Every number in it is checked against the
committed pre-registration it is taken from.

Logic
-----
- Inputs (committed files only; see supp_table_common.guard). There are seven records:
      leakage_results/step9_analysis.json           mechanism-of-action labels: contrasts (paired median)
      leakage_results/step10_analysis.json          learned chemical embedding: contrasts (paired median)
      leakage_results/step13_analysis.json          attention modules, development split: results/arms, arm_means
      leakage_results/step13b_analysis.json         cross-attention, confirmation splits: per_split/<s>/primary
      leakage_results/pathway_vs_fingerprint_v2.json   linear models with three feature sets (exploratory): part d
      leakage_results/pathway_vs_fingerprint.json   the record it supersedes (single precision, script not kept);
                                                    checks only
      leakage_results/ridge_vs_mlp_v2.json          feedforward network and cross-validated ridge (exploratory)
      figures/squidiff_publication_figures/fig3_values.csv   the values Fig. 3b and 3c were drawn from; check only
      PREREG_step9_pathway_conditioning.md, PREREG_step10_chemvae_stochastic.md, PREREG_step13_attention.md
                                                    the pre-registrations; read only to check the numbers quoted
                                                    in the description of the variants
  Record names -> names in the paper: fingerprint = the rewired model of the experiment; pathway = mechanism-of-
  action labels; both = labels added to the fingerprint; chemvae = learned chemical embedding; fp_chemvae =
  embedding added to the fingerprint; attn_fp = fingerprint self-attention; attn_mem = attention over similar
  training compounds; attn_cross = cross-attention between compound and cell state; mlp_ctrl = parameter-matched
  control without attention; ref = rewired model; de_overlap / jac = gene identity; energy / en = population
  distance; dpc = response correlation; pathway_x_line, fp_x_line, target_x_line = ridge with that feature set
  (in version 2 of the record also: fingerprint, pathway, target).
- Checks, all before anything is written (a failure stops the script):
    1. Records of the interventions: development split, 443 conditions of 37 compounds, five seeds per arm,
       5,000 bootstrap replicates; each stored word for an interval follows from the interval; the differences
       of the attention modules equal the differences of the stored means (1e-9); the outcome stored for each
       module and for the confirmation follows from the intervals.
    2. Every value of Fig. 3b and 3c (15 rows: value, lower, upper, filled or open) equals the value given here.
    3. Exploratory records. Linear models: the best penalty, the value at penalty 1 and the gains follow from the
       scores stored per penalty; every summary over the 200 fold assignments (median, 2.5th and 97.5th
       percentile, number of positive gains) is recomputed here from the 200 x 5 stored scores; the comparison
       stored in the record is the comparison with the superseded file, within the tolerances of the record; the
       anchor of the record holds (Step 7 ridge, 1e-9) and the fingerprint model at penalty 1 equals the
       cross-validated ridge of part e (1e-4). Network: each value is the mean of its three seeds; the settings
       named in the caption are those stored in the record.
    4. Every number in the description of the variants is found in the pre-registration it is taken from.
    5. The statements of Results 3, of Supplementary Note 1 and of the captions of Fig. 3b and 3c hold at their
       printed precision (EXPECTED and NOTE1 below; checks only, never inputs to a computation).
- Rounding: three decimals, four for differences in gene identity (they are of the order of 0.001); each value is
  rounded once from full precision; differences carry their sign.
- Outputs (never overwritten): supplementary_tables/supp_table_S10.csv and .md (see supp_table_common).

Run from the research worktree root (supp_table_common.py must be in the same folder):
    python3 make_supp_table_s10.py
"""
import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import (DASH, SPLITS, SPLIT_HEAD, close, die, guard, interval, load_csv, load_json, num, same,
                               table, write_outputs)

TABLE = 'S10'
IN = dict(
    s9=('leakage_results/step9_analysis.json', '6f84c17f4a7e5344abf9c67ddbd5b00b1af5906519172656602eabbf7cd629ac'),
    s10=('leakage_results/step10_analysis.json', '83a30129d62491d6243e1de4d1514c834a64c64d923817bd54da8ee5f2d746ae'),
    s13=('leakage_results/step13_analysis.json', '71b9a94e7bd80f36c372a60850b46f108064eb95c3891cc17e816a0b9b05f7ce'),
    s13b=('leakage_results/step13b_analysis.json', '4ca621d99af270311ececc4abc8b91af6cf41ea7d32b6f04bb6e2ac6f3e9e2a5'),
    lin2=('leakage_results/pathway_vs_fingerprint_v2.json',
          'aebc1c61610bd1c0cbf29269facfdb051474293c9ac13dfb2efbe45e5cb58066'),
    lin=('leakage_results/pathway_vs_fingerprint.json', '8bbe7aeec8c36d551cdcc1814b3101af394f745a95eef62c26dc2a6cd52026fa'),
    mlp=('leakage_results/ridge_vs_mlp_v2.json', '7ec3fadafe8b84899b379bfcecdc269d1e53a7594cb4aaaab877e2dfe0a19d5b'),
    fig3=('figures/squidiff_publication_figures/fig3_values.csv',
          'e7c36860a45337d9b9c71beca69c0f348350ae0436961a240071f3a74ebc85c1'),
    pre9=('PREREG_step9_pathway_conditioning.md', '7b93f0bbedeeda84fdd6e780fae97d69bd5a89f7e6a8c1b3e90458b36ddf71c3'),
    pre10=('PREREG_step10_chemvae_stochastic.md', 'c1b3bf6eb8a7d5391131ac713657f6b9f9a938c7c6c27f7c4591e54ef443b8f7'),
    pre13=('PREREG_step13_attention.md', '6ca1f85b83faa2d71ce6401ebebd0b78ee56f684702f9a7c3b23c938bb9e6346'),
)
TOL = 1e-9
DEV = 'drug_split_1'
CONF = ('drug_split_0', 'drug_split_2')                   # the two confirmation splits

L_REW = 'Rewired model'
L_MOA_ADD = 'Mechanism-of-action labels added to the fingerprint'
L_MOA_ONLY = 'Mechanism-of-action labels in place of the fingerprint'
L_EMB_ADD = 'Learned chemical embedding added to the fingerprint'
L_EMB_ONLY = 'Learned chemical embedding in place of the fingerprint'
L_FP = 'Fingerprint self-attention'
L_MEM = 'Attention over similar training compounds'
L_CROSS = 'Cross-attention between compound and cell state'
L_CTRL = 'Parameter-matched control without attention'
M_RC, M_GI, M_PD = 'Response correlation', 'Gene identity', 'Population distance'
# the comparisons stored for an attention module: (key in the record, measure, compared with, decimals)
ARM_KEYS = [('de_overlap_minus_reference', M_GI, 'rewired model', 4),
            ('de_overlap_minus_control', M_GI, 'parameter-matched control', 4),
            ('energy_minus_reference', M_PD, 'rewired model', 3),
            ('dpc_minus_reference', M_RC, 'rewired model', 3)]
# the comparisons stored for the control: (key, measure, decimals)
CTRL_KEYS = [('jac', M_GI, 4), ('en', M_PD, 3), ('dpc', M_RC, 3)]
WIDTHS = [('64', 'Network, width 64'), ('256', 'Network, width 256'), ('1024', 'Network, width 1,024')]
# the linear models of part d: (name in version 2 of the record, key of the superseded record, label in the table)
LINEAR = [('fingerprint', 'fp_x_line', 'Fingerprint'), ('pathway', 'pathway_x_line', 'Mechanism-of-action label'),
          ('target', 'target_x_line', 'Annotated target')]
# the gains shown in part d: (label model, which penalty, label in the table)
GAINS = [('pathway', 'best', 'Mechanism-of-action label − fingerprint'),
         ('target', 'best', 'Annotated target − fingerprint'),
         ('pathway', 'one', 'Mechanism-of-action label − fingerprint, both at penalty 1')]
PENALTIES = [0.01, 0.1, 1.0, 10.0, 100.0]
N_FOLDSETS = 200                                         # random fold assignments in the record

# Numbers quoted in the description of the variants, each with the pre-registration that states it (check 4).
QUOTED = dict(
    pre9=['| B. pathway | pathway one-hot, dose-scaled | 55 |',
          '| **C. both** | **fingerprint concatenated with pathway** | **1079** |',
          '+112,640 (+0.19%)', 'five seeds each', 'flow matching at 20 NFE',
          'DPC-50, ridge, compound-grouped folds, each feature set at its own best alpha',
          'Nine feature sets were tried on drug_split_1'],
    pre10=['64 ChemicalVAE draws per compound', 'Latent dimension\n196', 'concatenated | 1220 |', 'ChemicalVAE alone | 196 |',
           '3 compounds, 4,577 cells (2.7%)', 'averaged over 16 independent draws per condition', 'flow matching at 20 NFE',
           'five seeds each'],
    pre13=['residual `r` of width 1024', 'd = 64, 4 heads', 'one attention layer, 16 cell tokens', '| 182,080 |',
           'M = 112 on `drug_split_1`', '197,760 + 64 M = 204,928', '| 354,624 |', '| 354,120 |', 'Five seeds per arm',
           'sampled at 50 NFE'],
)

# Printed values the manuscript quotes (checks only, never inputs).
EXPECTED = dict(
    moa='0.002',                        # Results 3: "changed the response correlation of the generative model by 0.002"
    linear=['0.025', '0.018', '0.046'], # "gained 0.025 on the development split and 0.018 and 0.046 on the two
                                        #  confirmation splits"
    linear_folds=(188, 197, 200),       # "the gain was present in 188-197 of 200 random fold assignments"
    network=('0.11', '0.21'),           # "reached a response correlation of 0.11-0.21"
    ridge=('0.08', '0.25'),             # "against 0.08-0.25 for the ridge"
    cross=('0.006', '0.003', '12'),     # "improved gene identity by 0.006 ... (0.003 of it over a control ...) ... by 12 %"
    conf=['-0.001', '0.001'],           # "the gene-identity effect was absent (-0.001 and +0.001)"
    others=('-0.011', '0.010'),         # "(population distance, -0.011) and ... (response correlation, +0.010)"
    precision=['0.003', '0.003'],       # "although their precision (+-0.003)"
)

# Printed values of Supplementary Note 1, in the order of the splits 1, 0, 2 (checks only, never inputs).
NOTE1 = dict(
    classes_label=[20, 23, 21],                 # "20, 23 and 21 classes among the held-out compounds"
    classes_target=[24, 27, 27],                # "the annotated target (24, 27 and 27 classes)"
    n_penalties=5, penalty_range=(0.01, 100.0),  # "the best of five penalties (0.01-100)"
    label=['0.213', '0.265', '0.129'],          # "the label model reached a response correlation of ..."
    fingerprint=['0.188', '0.247', '0.082'],    # "against ... for the fingerprint model"
    gain=['0.025', '0.018', '0.046'],           # "a gain of ..."
    positive=[194, 188, 197],                   # "Over 200 random assignments the gain was positive in ..."
    median_gain=['0.034', '0.025', '0.056'],    # "(median ...)"
    positive_at_penalty_1=[187, 185, 195],      # "with both models at the fixed penalty of 1 it was positive in ..."
    target_positive=[95, 184, 193],             # "The annotated target ... (positive in 95 ...) ... (184 and 193)"
    change_below=0.001,                         # "they differ by less than 0.001 from those of the original run"
)


def tri(c, key='diff'):
    """(value, lower, upper) of one stored comparison; key names the value ('diff' or 'paired_median')."""
    return float(c[key]), float(c['ci'][0]), float(c['ci'][1])


def above_zero(c):
    return c[1] > 0


def below_zero(c):
    return c[2] < 0


def includes_zero(c):
    return c[1] <= 0 <= c[2]


def word(c):
    """The records' own word for an interval."""
    return 'above zero' if c[1] > 0 else ('below zero' if c[2] < 0 else 'includes zero')


def pick(contrasts, pattern, without=None):
    """The one stored comparison whose name contains pattern (and not `without`); the names hold free text."""
    names = [k for k in contrasts if pattern in k and not (without and without in k)]
    if len(names) != 1:
        die('expected one comparison named "%s", found %r' % (pattern, names))
    return names[0]


def plain(x, dp):
    """An unsigned printed value for the comparison with the text, e.g. 0.00233 -> '0.002'."""
    return '%.*f' % (dp, x)


def span(values, dp=2):
    return (plain(min(values), dp), plain(max(values), dp))


def mean(values):
    return sum(values) / len(values)


def percentile(values, q):
    """The q-th percentile with linear interpolation between neighbouring order statistics (the default rule of
    numpy, which the record used); q = 50 gives the median."""
    v = sorted(values)
    pos = (len(v) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (pos - lo) * (v[hi] - v[lo])


def spread(values):
    """(median, 2.5th percentile, 97.5th percentile) of one quantity over the fold assignments."""
    return percentile(values, 50.0), percentile(values, 2.5), percentile(values, 97.5)


def main():
    guard(IN, TABLE)
    s9, s10, s13, s13b, lin2, lin, mlp = (load_json(IN[k][0]) for k in ('s9', 's10', 's13', 's13b', 'lin2', 'lin', 'mlp'))

    # ---- check 1: the records of the interventions -------------------------------------------------------------------
    n1 = 0
    A = {}                                               # part a: A[label] = (value, lower, upper, record, key)
    for rec, name, prereg, arms, picks in (
            (s9, 'step9_analysis.json', 'PREREG_step9_pathway_conditioning.md', ['fingerprint', 'pathway', 'both'],
             [(L_MOA_ADD, 'both minus fingerprint', None, True), (L_MOA_ONLY, 'pathway minus fingerprint', None, False)]),
            (s10, 'step10_analysis.json', 'PREREG_step10_chemvae_stochastic.md', ['fingerprint_s10', 'fp_chemvae', 'chemvae'],
             [(L_EMB_ADD, 'fp_chemvae minus fingerprint', None, True),
              (L_EMB_ONLY, 'chemvae minus fingerprint', 'fp_chemvae', False)])):
        same((rec['prereg'], rec['n_conditions'], rec['n_drugs'], len(rec['contrasts'])), (prereg, 443, 37, 2),
             'check 1, design stored in %s' % name)
        same(sorted(rec['de_overlap']), sorted(arms), 'check 1, arms of %s' % name)
        for arm in arms:                                  # five trained models per arm
            same((len(rec['de_overlap'][arm]), len(rec['energy'][arm])), (5, 5), 'check 1, seeds of %s in %s' % (arm, name))
        for label, pattern, without, gated in picks:
            k = pick(rec['contrasts'], pattern, without)
            c = tri(rec['contrasts'][k], 'paired_median')
            same((rec['contrasts'][k]['verdict'], rec['contrasts'][k]['gated']), (word(c), gated),
                 'check 1, stored word and role of "%s" in %s' % (k, name))
            A[label] = c + (name, 'contrasts/%s' % k)
            n1 += 1
    same((s13['prereg'], s13['split'], s13['interval'], s13['n_boot']),
         ('PREREG_step13_attention.md', DEV, '95% percentile, drug-clustered', 5000), 'check 1, design of the attention record')
    same((s13b['prereg'], s13b['interval'], s13b['n_boot'], sorted(s13b['per_split'])),
         ('PREREG_step13b_confirmation.md', '97.5% percentile per split, drug-clustered', 5000, sorted(CONF)),
         'check 1, design of the confirmation record')
    arms13 = s13['results']['arms']
    means13 = s13['results']['arm_means']
    same(sorted(arms13), ['attn_cross', 'attn_fp', 'attn_mem'], 'check 1, attention modules in the record')
    same(sorted(len(v) for v in s13['inputs'].values()), [5] * 5, 'check 1, five trained models per arm of the attention record')
    same(s13['results']['attn_mem_memory_attention']['memory_size'], 112, 'check 1, training compounds the memory holds')
    B = {}                                               # B[arm][key] = (value, lower, upper), development split
    for arm in sorted(arms13):
        B[arm] = {}
        for key, _, _, _ in ARM_KEYS:
            c = tri(arms13[arm][key])
            same(arms13[arm][key]['verdict'], word(c), 'check 1, stored word of %s, %s' % (arm, key))
            B[arm][key] = c
            n1 += 1
        for key, short, other in (('de_overlap_minus_reference', 'jac', 'ref'), ('de_overlap_minus_control', 'jac', 'mlp_ctrl'),
                                  ('energy_minus_reference', 'en', 'ref'), ('dpc_minus_reference', 'dpc', 'ref')):
            close(B[arm][key][0], means13[arm][short] - means13[other][short], TOL,
                  'check 1, %s of %s against the stored means' % (key, arm))
        # the outcome stored for a module, from the rule fixed in advance (gene identity decides)
        gi, ctrl, pd = B[arm]['de_overlap_minus_reference'], B[arm]['de_overlap_minus_control'], B[arm]['energy_minus_reference']
        if below_zero(gi):
            want = 'HARMS'
        elif not above_zero(gi):
            want = 'UNRESOLVED'
        elif above_zero(pd):
            want = 'TRADE-OFF'
        elif above_zero(ctrl):
            want = 'ATTENTION HELPS'
        else:
            want = 'CAPACITY'
        same((arms13[arm]['category'], s13['categories'][arm]), (want, want), 'check 1, outcome stored for %s' % arm)
    BC = {}                                              # the control against the rewired model, per split
    BC[DEV] = dict((short, tri(s13['results']['control_minus_reference'][short])) for short, _, _ in CTRL_KEYS)
    X = {DEV: dict((key, B['attn_cross'][key]) for key, _, _, _ in ARM_KEYS)}   # cross-attention, per split
    MEANS = {DEV: means13}
    for s in CONF:
        p = s13b['per_split'][s]
        X[s] = dict((key, tri(p['primary'][key])) for key, _, _, _ in ARM_KEYS)
        BC[s] = dict((short, tri(p['control_minus_reference'][short])) for short, _, _ in CTRL_KEYS)
        MEANS[s] = p['arm_means']
        for key, short, other in (('de_overlap_minus_reference', 'jac', 'ref'), ('de_overlap_minus_control', 'jac', 'mlp_ctrl'),
                                  ('energy_minus_reference', 'en', 'ref'), ('dpc_minus_reference', 'dpc', 'ref')):
            same(p['primary'][key]['verdict'], word(X[s][key]), 'check 1, stored word of %s on %s' % (key, s))
            close(X[s][key][0], p['arm_means']['attn_cross'][short] - p['arm_means'][other][short], TOL,
                  'check 1, %s on %s against the stored means' % (key, s))
            n1 += 1
    for s in SPLITS:
        for short, _, _ in CTRL_KEYS:
            close(BC[s][short][0], MEANS[s]['mlp_ctrl'][short] - MEANS[s]['ref'][short], TOL,
                  'check 1, control minus rewired model (%s) on %s against the stored means' % (short, s))
            n1 += 1
    # the confirmation: "NOT CONFIRMED" is stored when gene identity against the rewired model is unresolved on both
    same((s13b['verdict'], [word(X[s]['de_overlap_minus_reference']) for s in CONF]),
         ('NOT CONFIRMED', ['includes zero', 'includes zero']), 'check 1, outcome stored for the confirmation')

    # ---- check 2: the values Fig. 3b and 3c were drawn from --------------------------------------------------------------
    fig_split = {'Split 1': DEV, 'Split 0': 'drug_split_0', 'Split 2': 'drug_split_2'}
    fig_b = {'Mechanism-of-action labels': A[L_MOA_ADD][:3], 'Learned chemical embedding': A[L_EMB_ONLY][:3],
             'Chemical embedding + fingerprint': A[L_EMB_ADD][:3],
             'Fingerprint self-attention': B['attn_fp']['de_overlap_minus_reference'],
             'Attention over similar training compounds': B['attn_mem']['de_overlap_minus_reference'],
             'Cross-attention, compound × cell state': B['attn_cross']['de_overlap_minus_reference']}
    fig_c = {'Gene identity vs rewired model': 'de_overlap_minus_reference',
             'Gene identity vs matched control': 'de_overlap_minus_control',
             'Population distance vs rewired model': 'energy_minus_reference'}
    seen = set()
    for r in load_csv(IN['fig3'][0]):
        quantity = r['quantity'].replace('\n', ' ')
        what = 'check 2, fig3_values.csv panel %s: %s, %s, %s' % (r['panel'], r['intervention'], quantity, r['split'])
        if r['split'] not in fig_split:
            die(what + ': a split this table does not hold')
        s = fig_split[r['split']]
        if r['panel'] == 'b' and r['intervention'] in fig_b and s == DEV:
            c = fig_b[r['intervention']]
        elif r['panel'] == 'c' and r['intervention'] == 'Cross-attention' and quantity in fig_c:
            c = X[s][fig_c[quantity]]
        else:
            die(what + ': a point of the figure with no row in this table')
        same(r['level'], '95 %' if s == DEV else '97.5 %', what + ' (level of the interval)')
        for i, col in enumerate(('point', 'lo', 'hi')):
            close(c[i], float(r[col]), TOL, what + ' (%s)' % col)
        same(r['excludes_zero'], 'FALSE' if includes_zero(c) else 'TRUE', what + ' (filled or open point)')
        seen.add((r['panel'], r['intervention'], quantity, r['split']))
    n2 = len(seen)
    same(n2, 15, 'check 2, number of values in fig3_values.csv')

    # ---- check 3: the exploratory records --------------------------------------------------------------------------------
    # (a) the linear models: version 2 of the record (part d) and the record it supersedes (checks only)
    same(lin['analysis'], 'EXPLORATORY, not pre-registered; best of nine feature sets on drug_split_1, replicated here',
         'check 3, status stored in the superseded record of the linear models')
    same(lin2['analysis'].startswith('EXPLORATORY, not pre-registered; re-creation of leakage_results/'
                                     'pathway_vs_fingerprint.json'), True,
         'check 3, status stored in the record of the linear models')
    fs = lin2['fold_sensitivity']
    same((lin2['settings']['penalties'], fs['penalties'], fs['n_assignments']), (PENALTIES, PENALTIES, N_FOLDSETS),
         'check 3, penalties and number of fold assignments of the linear models')
    same(lin2['old_record']['sha256'], IN['lin'][1], 'check 3, the file that version 2 was compared with')
    one = PENALTIES.index(1.0)
    OLDV, LIN, n3 = {}, {}, 0
    for s in SPLITS:
        p, f, q = lin2['per_split'][s], fs['per_split'][s], lin['per_split'][s]
        same((p['n'], p['n_drugs'], p['n_pathways'], p['n_targets']),
             (q['n'], q['n_drugs'], q['n_pathways'], q['n_targets']),
             'check 3, conditions, compounds and label classes of %s in the two records of the linear models' % s)
        if float(lin2['anchor_step7_ridge_penalty_1'][s]['abs_diff']) > TOL:
            die('check 3: the Step 7 ridge is not reproduced by the folds of the record on %s' % s)
        OLDV[s] = dict((k, float(q[k])) for _, k, _ in LINEAR)
        d = dict(best={}, one={}, gain={}, fold={}, positive={})
        for m, old_key, _ in LINEAR:
            scores = dict((float(a), float(v['median'])) for a, v in p['per_penalty'][m].items())
            same(sorted(scores), PENALTIES, 'check 3, penalties stored for the %s model on %s' % (m, s))
            d['best'][m], d['one'][m] = max(scores.values()), scores[1.0]
            close(float(p['best'][m]['median']), d['best'][m], TOL, 'check 3, best penalty of the %s model on %s' % (m, s))
            close(scores[float(p['best'][m]['penalty'])], d['best'][m], TOL,
                  'check 3, the penalty named best for the %s model on %s' % (m, s))
            close(float(p[old_key]), d['best'][m], TOL, 'check 3, %s of %s' % (old_key, s))
            close(float(p['at_penalty_1'][m]), d['one'][m], TOL, 'check 3, %s model at penalty 1 on %s' % (m, s))
            c = lin2['old_record']['comparison'][s][old_key]     # the comparison with the superseded record
            close(float(c['old']), OLDV[s][old_key], TOL, 'check 3, superseded value of %s on %s' % (old_key, s))
            close(float(c['new']), d['best'][m], TOL, 'check 3, compared value of %s on %s' % (old_key, s))
            if abs(d['best'][m] - OLDV[s][old_key]) > float(c['tolerance']):
                die('check 3: %s on %s differs from the superseded record by more than its tolerance' % (old_key, s))
            n3 += 1
        for m, tag, _ in GAINS:
            d['gain'][(m, tag)] = d[tag][m] - d[tag]['fingerprint']
            close(float(p['gain_over_fingerprint'][{'best': 'best_penalty', 'one': 'penalty_1'}[tag]][m]),
                  d['gain'][(m, tag)], TOL, 'check 3, gain of the %s model (%s) on %s' % (m, tag, s))
        same(p['pathway_beats_fp'], d['gain'][('pathway', 'best')] > 0,
             'check 3, stored flag of the labels against the fingerprint on %s' % s)
        # the 200 random fold assignments: every summary is recomputed from the stored scores
        same((len(f['assignments']), len(set(f['assignments'])), f['n_distinct']), (N_FOLDSETS, N_FOLDSETS, N_FOLDSETS),
             'check 3, number of distinct fold assignments on %s' % s)
        per = dict(best={}, one={})                      # per[penalty rule][model] = one value per fold assignment
        for m, _, _ in LINEAR:
            runs = [[float(x) for x in row] for row in f['median_dpc50'][m]]
            same((len(runs), sorted(set(len(r) for r in runs))), (N_FOLDSETS, [len(PENALTIES)]),
                 'check 3, stored scores of the %s model over the fold assignments on %s' % (m, s))
            per['best'][m], per['one'][m] = [max(r) for r in runs], [r[one] for r in runs]
        for tag, key in (('best', 'best_penalty'), ('one', 'penalty_1')):
            for m, _, _ in LINEAR:
                got, st = spread(per[tag][m]), f['summary'][key]['models'][m]
                for x, name in zip(got, ('median', 'p2.5', 'p97.5')):
                    close(float(st[name]), x, 1e-12, 'check 3, %s of the %s model (%s) over the fold assignments on %s'
                          % (name, m, tag, s))
                close(float(st['paper']), d[tag][m], TOL,
                      'check 3, value of the paper stored with the spread (%s, %s)' % (m, s))
                d['fold'][(m, tag)] = got
                n3 += 1
        for m, tag, _ in GAINS:
            key = {'best': 'best_penalty', 'one': 'penalty_1'}[tag]
            diffs = [a - b for a, b in zip(per[tag][m], per[tag]['fingerprint'])]
            got, st = spread(diffs), f['summary'][key]['gain_over_fingerprint'][m]
            for x, name in zip(got, ('median', 'p2.5', 'p97.5')):
                close(float(st[name]), x, 1e-12, 'check 3, %s of the gain of the %s model (%s) on %s' % (name, m, tag, s))
            d['fold'][('gain', m, tag)] = got
            d['positive'][(m, tag)] = sum(1 for x in diffs if x > 0)
            same(st['n_positive'], d['positive'][(m, tag)],
                 'check 3, positive gains of the %s model (%s) on %s' % (m, tag, s))
            n3 += 1
        d['target_positive'] = d['positive'][('target', 'best')]
        LIN[s] = d
    # (b) the feedforward network
    same(mlp['analysis'].startswith('EXPLORATORY, not pre-registered'), True, 'check 3, status of the network record')
    st = mlp['settings']
    same((st['features'], st['folds'], st['ridge'], st['mlp'], st['widths'], st['seeds']),
         ('[FP(1024, dose-scaled), line one-hot, FP x line] = 4,099', 'GroupKFold(5) by SMILES',
          'sklearn Ridge(alpha=1.0), intercept, unscaled',
          'one hidden layer, ReLU, AdamW, MSE, standardised inputs, minibatch 32, lr 0.001', [64, 256, 1024], [0, 1, 2]),
         'check 3, settings of the network record')
    same((st['sweep']['split'], st['sweep']['weight_decay'], st['sweep']['epochs'], len(mlp['sweep'])),
         (DEV, [0.01, 0.1, 1.0, 10.0], [25, 100, 400], 12), 'check 3, the twelve settings tried on the development split')
    best = max(mlp['sweep'], key=lambda k: mlp['sweep'][k]['median'])
    same((mlp['chosen']['weight_decay'], mlp['chosen']['epochs'], best), (10.0, 400, 'wd10_ep400'),
         'check 3, the setting chosen is the best of the twelve on the development split')
    NET, RIDGE = {}, {}
    for s in SPLITS:
        p = mlp['per_split'][s]
        same((p['n'], p['n_drugs']), (lin2['per_split'][s]['n'], lin2['per_split'][s]['n_drugs']),
             'check 3, conditions and compounds of %s in the two exploratory records' % s)
        RIDGE[s] = float(p['ridge']['median'])
        # at penalty 1 the fingerprint model of part d is the cross-validated ridge of part e (the two records differ
        # only in details of the arithmetic)
        close(LIN[s]['one']['fingerprint'], RIDGE[s], 1e-4,
              'check 3, fingerprint model at penalty 1 against the ridge on %s' % s)
        NET[s] = {}
        for w, _ in WIDTHS:
            seeds = [float(x['median']) for x in p['mlp'][w]['seeds']]
            same(len(seeds), 3, 'check 3, seeds of width %s on %s' % (w, s))
            close(float(p['mlp'][w]['median_mean_over_seeds']), mean(seeds), TOL,
                  'check 3, seed mean of width %s on %s' % (w, s))
            NET[s][w] = float(p['mlp'][w]['median_mean_over_seeds'])
            n3 += 1
    # Part d gives each model at the penalty that was best for it; part e uses the fixed penalty of the paper's ridge
    # (penalty 1). They are given in separate rows and never mixed.
    conds = [lin2['per_split'][s]['n'] for s in SPLITS]
    drugs = [lin2['per_split'][s]['n_drugs'] for s in SPLITS]
    same((conds[0], drugs[0]), (s9['n_conditions'], s9['n_drugs']), 'check 3, development split in all records')
    width_dev = max((w for w, _ in WIDTHS), key=lambda w: NET[DEV][w])     # the width that is best on the development split

    # ---- check 4: numbers quoted in the description of the variants -------------------------------------------------------
    n4 = 0
    # the pre-registration of the label experiment states the values of the superseded record of the linear models
    quoted = dict((k, list(v)) for k, v in QUOTED.items())
    for s in SPLITS:
        quoted['pre9'].append('| %s | **%.4f** | %.4f | %.4f |' % (s, OLDV[s]['pathway_x_line'], OLDV[s]['fp_x_line'],
                                                               OLDV[s]['target_x_line']))
    for key, phrases in sorted(quoted.items()):
        with open(IN[key][0], encoding='utf-8') as f:
            text = f.read()
        for phrase in phrases:
            if phrase not in text:
                die('check 4: %s does not state %r' % (IN[key][0], phrase))
            n4 += 1

    # ---- check 5: the statements of the manuscript -----------------------------------------------------------------------
    cross = X[DEV]
    rel = dict((s, 100.0 * X[s]['energy_minus_reference'][0] / MEANS[s]['ref']['en']) for s in SPLITS)   # per cent
    got = dict(
        moa=plain(A[L_MOA_ADD][0], 3),
        linear=[plain(LIN[s]['gain'][('pathway', 'best')], 3) for s in SPLITS],
        linear_folds=(min(LIN[s]['positive'][('pathway', 'best')] for s in SPLITS),
                      max(LIN[s]['positive'][('pathway', 'best')] for s in SPLITS), fs['n_assignments']),
        network=span([NET[s][width_dev] for s in SPLITS]),
        ridge=span([RIDGE[s] for s in SPLITS]),
        cross=(plain(cross['de_overlap_minus_reference'][0], 3), plain(cross['de_overlap_minus_control'][0], 3),
               plain(-rel[DEV], 0)),
        conf=[plain(X[s]['de_overlap_minus_reference'][0], 3) for s in CONF],
        others=(plain(B['attn_mem']['energy_minus_reference'][0], 3), plain(B['attn_fp']['dpc_minus_reference'][0], 3)),
        precision=[plain((X[s]['de_overlap_minus_reference'][2] - X[s]['de_overlap_minus_reference'][1]) / 2.0, 3)
                   for s in CONF],
    )
    same(got, EXPECTED, 'check 5, values quoted in Results 3')
    # Supplementary Note 1
    note = dict(
        classes_label=[lin2['per_split'][s]['n_pathways'] for s in SPLITS],
        classes_target=[lin2['per_split'][s]['n_targets'] for s in SPLITS],
        n_penalties=len(PENALTIES), penalty_range=(min(PENALTIES), max(PENALTIES)),
        label=[plain(LIN[s]['best']['pathway'], 3) for s in SPLITS],
        fingerprint=[plain(LIN[s]['best']['fingerprint'], 3) for s in SPLITS],
        gain=[plain(LIN[s]['gain'][('pathway', 'best')], 3) for s in SPLITS],
        positive=[LIN[s]['positive'][('pathway', 'best')] for s in SPLITS],
        median_gain=[plain(LIN[s]['fold'][('gain', 'pathway', 'best')][0], 3) for s in SPLITS],
        positive_at_penalty_1=[LIN[s]['positive'][('pathway', 'one')] for s in SPLITS],
        target_positive=[LIN[s]['target_positive'] for s in SPLITS],
        change_below=NOTE1['change_below'],
    )
    same(note, NOTE1, 'check 5, values quoted in Supplementary Note 1')
    worst = max(abs(LIN[s]['best'][m] - OLDV[s][k]) for s in SPLITS for m, k, _ in LINEAR)
    if not worst < NOTE1['change_below']:
        die('check 5: a value of the linear models differs from the original run by %.6f, not by less than %s'
            % (worst, NOTE1['change_below']))
    same(width_dev, '64', 'check 5, the network width that is best on the development split')
    if not includes_zero(A[L_MOA_ADD]):
        die('check 5: the interval of the mechanism-of-action labels excludes zero')
    # Fig. 3b: "Only cross-attention cleared zero" among the six plotted contrasts
    same(sorted(k for k, c in fig_b.items() if not includes_zero(c)), ['Cross-attention, compound × cell state'],
         'check 5, the plotted contrasts of Fig. 3b that exclude zero')
    # Results 3: "Two other modules moved a further pre-specified measure on that split ...; neither improved gene
    # identity": closer in population distance, higher in response correlation, gene identity unresolved for both
    if not (below_zero(B['attn_mem']['energy_minus_reference']) and above_zero(B['attn_fp']['dpc_minus_reference'])
            and includes_zero(B['attn_mem']['de_overlap_minus_reference'])
            and includes_zero(B['attn_fp']['de_overlap_minus_reference'])):
        die('check 5: the two attention modules that were not taken further')
    # Fig. 3c: "absent on both confirmation splits; the reduction in population distance held on one of them"
    same(([includes_zero(X[s]['de_overlap_minus_reference']) for s in CONF],
          sum(below_zero(X[s]['energy_minus_reference']) for s in CONF)), ([True, True], 1),
         'check 5, cross-attention on the confirmation splits')
    if not (above_zero(cross['de_overlap_minus_reference']) and above_zero(cross['de_overlap_minus_control'])
            and below_zero(cross['energy_minus_reference'])):
        die('check 5: cross-attention on the development split')

    # ---- the table ---------------------------------------------------------------------------------------------------
    long_rows = []

    def keep(part, measure, entry, comparison, split, c, status, record, key, test=True):
        """One row of the long-format file. c is a value or (value, lower, upper). test=False marks a range that is
        not the interval of a comparison (the spread over fold assignments): no excludes-zero flag is written."""
        v, lo, hi = c if isinstance(c, tuple) else (c, None, None)
        long_rows.append(dict(part=part, measure=measure, entry=entry, comparison=comparison, split=split, value=v, lo=lo,
                              hi=hi, excludes_zero='' if (lo is None or not test) else int(lo > 0 or hi < 0),
                              status=status, source_record=record, source_key=key))

    PRE, EXPL = 'specified before scoring', 'exploratory'
    md = ['Supplementary Table S10 | Interventions on the compound input of the rewired model on sci-Plex (Results 3; '
          'Fig. 3b, c; Supplementary Note 1). Gene identity, response correlation and population distance as in '
          'Supplementary Tables S5 and S6. Each variant was trained with five seeds and compared with five seeds of the '
          'rewired model evaluated with the same number of integration steps. Differences are given with percentile '
          'bootstrap intervals (5,000 '
          'replicates, resampling held-out compounds), at 95 %% on the development split (split 1; %d conditions of %d '
          'held-out compounds) and at 97.5 %% on the confirmation splits (splits 0 and 2; %d and %d conditions); an '
          'asterisk marks an interval that excludes zero. For population distance a negative difference is closer to '
          'the measured cells. a, The interventions on the development split. Response correlation: median over '
          'conditions of the paired difference. Gene identity: mean over conditions of the difference. The first, the '
          'third and the last three rows are the primary comparisons of the experiments; six of the seven rows are '
          'drawn in Fig. 3b. b, The two attention modules that were not taken further: every comparison specified for '
          'them (mean differences). c, Cross-attention and its control on the three splits (mean differences); the '
          'first three rows are drawn in Fig. 3c, and the last rows give the values the differences are computed from '
          'and the difference in population distance relative to the rewired model. Parts a to c come from '
          'analyses specified before the data were scored. d, Ridge models fitted within the held-out compounds, '
          'with five folds grouped by compound, on the cell line, the compound features and their products. The '
          'compound features were the fingerprint, the mechanism-of-action label or the annotated target, the last '
          'two as indicator vectors; the label model was the best of nine feature sets tried on the development '
          'split and was then fitted on the other two (Supplementary Note 1). Each model is given at the best of '
          'five penalties (0.01–100), chosen on the scored conditions; the last two rows give the difference with '
          'both models at the fixed penalty of 1, at which the fingerprint model is the cross-validated ridge of '
          'part e. Of the paper: the fold assignment of the cross-validated ridge. 200 random: median (2.5th to '
          '97.5th percentile) over 200 random assignments of the compounds to the five folds and, for a difference, '
          'the number of assignments in which it was positive. e, A '
          'feedforward network given the features of the linear references (4,099: dose-scaled fingerprint, cell-line '
          'indicator and their products), fitted within the held-out compounds with five folds grouped by compound, as '
          'the cross-validated ridge is. One hidden layer with ReLU, AdamW, weight decay 10, 400 epochs (the best of '
          'twelve settings on the development split: weight decay 0.01–10, 25–400 epochs), minibatch 32, learning rate '
          '10⁻³, inputs standardised on the training folds; three seeds per width. Values of parts d and e are the '
          'median over conditions of the response correlation, for the network the mean over seeds. Parts d and e were '
          'exploratory.' % (conds[0], drugs[0], conds[1], conds[2]),
          '',
          '**Mechanism-of-action labels.** The annotated mechanism class of each compound (55 classes) was given as an '
          'indicator vector scaled by log10(dose + 1), as the fingerprint is. It was added to the fingerprint (input width '
          '1,079 instead of 1,024; 0.19 % more parameters) or used in its place. Cells were generated in 20 integration '
          'steps.',
          '',
          '**Learned chemical embedding.** Each compound was represented by an embedding from a pretrained chemical '
          'variational autoencoder (ChemicalVAE; 196 dimensions, 64 draws per compound), added to the fingerprint (input '
          'width 1,220) or used in its place. The embedding was drawn once per training cell when the dataset was built and '
          'kept for the whole of training, not re-drawn at each access as pre-registered; evaluation averaged 16 draws per '
          'condition, as planned (Methods, Variants of the rewired model). Three training compounds that the autoencoder '
          'could not encode (4,577 cells, 2.7 % of the training cells) were left out of all three arms, so the rewired '
          'model of this experiment was retrained without them; the held-out conditions were unchanged. Cells were '
          'generated in 20 integration steps.',
          '',
          '**Attention modules.** Each module computes a correction of width 1,024 that is added to the fingerprint before '
          'the unchanged encoder; its last layer starts at zero, so every variant equals the rewired model before training. '
          'Fingerprint self-attention: one attention layer (width 64, four heads) over the active bits of the fingerprint '
          '(182,080 added parameters). Attention over similar training compounds: the compound attends over one learned '
          'entry per training compound (112 on the development split), addressed by the fingerprints of those compounds '
          '(204,928 added parameters); a held-out compound is never an entry. Cross-attention between compound and cell '
          'state and its control (354,624 and 354,120 added parameters) are described in the Methods. Cells were generated '
          'in 50 integration steps.',
          '',
          '**a** | The interventions on the development split.',
          '']
    rows = []
    for label in (L_MOA_ADD, L_MOA_ONLY, L_EMB_ADD, L_EMB_ONLY):
        c = A[label]
        rows.append([label, M_RC, interval(c[0], c[1], c[2], 3)])
        keep('a', M_RC, label, 'rewired model', DEV, c[:3], PRE, c[3], c[4])
    for label, arm in ((L_FP, 'attn_fp'), (L_MEM, 'attn_mem'), (L_CROSS, 'attn_cross')):
        c = B[arm]['de_overlap_minus_reference']
        rows.append([label, M_GI, interval(c[0], c[1], c[2], 4)])
        keep('a', M_GI, label, 'rewired model', DEV, c, PRE, 'step13_analysis.json',
             'results/arms/%s/de_overlap_minus_reference' % arm)
    md += table(['Intervention', 'Measure', 'Difference from the rewired model'], rows)

    md += ['', '**b** | The two attention modules that were not taken further (development split).', '']
    rows = []
    for label, arm in ((L_FP, 'attn_fp'), (L_MEM, 'attn_mem')):
        for key, measure, other, dp in ARM_KEYS:
            c = B[arm][key]
            rows.append([label, '%s, against the %s' % (measure, other), interval(c[0], c[1], c[2], dp)])
            keep('b', measure, label, other, DEV, c, PRE, 'step13_analysis.json', 'results/arms/%s/%s' % (arm, key))
    md += table(['Module', 'Comparison', 'Difference'], rows)

    md += ['', '**c** | Cross-attention and its control on the three splits.', '']

    def rec_of(s):
        """The record that holds split s: the development record or the confirmation record."""
        return 'step13_analysis.json' if s == DEV else 'step13b_analysis.json'

    def key_of(s, dev_key, conf_key):
        """Where a value of split s sits in its record."""
        return dev_key if s == DEV else 'per_split/%s/%s' % (s, conf_key)

    def cells3(get, dp):
        return [interval(get(s)[0], get(s)[1], get(s)[2], dp) for s in SPLITS]

    rows = []
    for key, measure, other, dp in ARM_KEYS:
        rows.append(['Cross-attention − %s' % other, measure] + cells3(lambda s, key=key: X[s][key], dp))
        for s in SPLITS:
            keep('c', measure, L_CROSS, other, s, X[s][key], PRE, rec_of(s),
                 key_of(s, 'results/arms/attn_cross/%s' % key, 'primary/%s' % key))
    for short, measure, dp in CTRL_KEYS:
        rows.append(['Control − rewired model', measure] + cells3(lambda s, short=short: BC[s][short], dp))
        for s in SPLITS:
            keep('c', measure, L_CTRL, 'rewired model', s, BC[s][short], PRE, rec_of(s),
                 key_of(s, 'results/control_minus_reference/%s' % short, 'control_minus_reference/%s' % short))
    for short, measure in (('jac', M_GI), ('en', M_PD)):
        for label, entry, arm in ((L_REW, L_REW, 'ref'), ('Cross-attention', L_CROSS, 'attn_cross'),
                                  ('Control', L_CTRL, 'mlp_ctrl')):
            rows.append([label, measure] + [num(MEANS[s][arm][short], 3) for s in SPLITS])
            for s in SPLITS:
                keep('c', measure, entry, '', s, float(MEANS[s][arm][short]), PRE, rec_of(s),
                     key_of(s, 'results/arm_means/%s/%s' % (arm, short), 'arm_means/%s/%s' % (arm, short)))
    rows.append(['Cross-attention − rewired model, relative (%)', M_PD] + [num(rel[s], 1, True) for s in SPLITS])
    md += table(['Entry', 'Measure'] + SPLIT_HEAD, rows)

    md += ['', '**d** | Linear models given the fingerprint, the labels or the targets (exploratory).', '']
    REC2 = 'pathway_vs_fingerprint_v2.json'
    PAPER, RANDOM = 'of the paper', '200 random'
    C_PAPER = 'fold assignment of the paper'
    C_RANDOM = '%d random fold assignments: median, 2.5th and 97.5th percentile' % N_FOLDSETS
    C_COUNT = '%d random fold assignments: number with a positive difference' % N_FOLDSETS
    RULE = {'best': 'best_penalty', 'one': 'penalty_1'}

    def range_cell(t, sign=False):
        """'0.178 (0.152–0.212)' for a value, '+0.034 (−0.006 to +0.073)' for a difference."""
        return '%s (%s%s%s)' % (num(t[0], 3, sign), num(t[1], 3, sign), ' to ' if sign else '–', num(t[2], 3, sign))

    rows = []
    for m, _, label in LINEAR:
        rows.append([label, PAPER] + [num(LIN[s]['best'][m], 3) for s in SPLITS])
        rows.append(['', RANDOM] + [range_cell(LIN[s]['fold'][(m, 'best')]) for s in SPLITS])
        for s in SPLITS:
            keep('d', M_RC, label, C_PAPER, s, LIN[s]['best'][m], EXPL, REC2, 'per_split/%s/best/%s/median' % (s, m))
            keep('d', M_RC, label, C_RANDOM, s, LIN[s]['fold'][(m, 'best')], EXPL, REC2,
                 'fold_sensitivity/per_split/%s/summary/best_penalty/models/%s' % (s, m), test=False)
    for m, tag, label in GAINS:
        rows.append([label, PAPER] + [num(LIN[s]['gain'][(m, tag)], 3, True) for s in SPLITS])
        rows.append(['', RANDOM] + ['%s; positive in %d' % (range_cell(LIN[s]['fold'][('gain', m, tag)], True),
                                                            LIN[s]['positive'][(m, tag)]) for s in SPLITS])
        for s in SPLITS:
            where = 'summary/%s/gain_over_fingerprint/%s' % (RULE[tag], m)
            keep('d', M_RC, label, C_PAPER, s, LIN[s]['gain'][(m, tag)], EXPL, REC2,
                 'per_split/%s/gain_over_fingerprint/%s/%s' % (s, RULE[tag], m))
            keep('d', M_RC, label, C_RANDOM, s, LIN[s]['fold'][('gain', m, tag)], EXPL, REC2,
                 'fold_sensitivity/per_split/%s/%s' % (s, where), test=False)
            keep('d', M_RC, label, C_COUNT, s, LIN[s]['positive'][(m, tag)], EXPL, REC2,
                 'fold_sensitivity/per_split/%s/%s/n_positive' % (s, where))
    md += table(['Compound features', 'Fold assignment'] + SPLIT_HEAD, rows)

    md += ['', '**e** | Feedforward network given the features of the linear references (exploratory).', '']
    rows = [['Cross-validated ridge'] + [num(RIDGE[s], 3) for s in SPLITS]]
    for s in SPLITS:
        keep('e', M_RC, 'Cross-validated ridge', '', s, RIDGE[s], EXPL, 'ridge_vs_mlp_v2.json', 'per_split/%s/ridge/median' % s)
    for w, label in WIDTHS:
        rows.append([label] + [num(NET[s][w], 3) for s in SPLITS])
        for s in SPLITS:
            keep('e', M_RC, label, '', s, NET[s][w], EXPL, 'ridge_vs_mlp_v2.json',
                 'per_split/%s/mlp/%s/median_mean_over_seeds' % (s, w))
    md += table([''] + SPLIT_HEAD, rows)

    md += ['',
           'Sources: step9_analysis.json (sha256 %s), step10_analysis.json (sha256 %s), step13_analysis.json (sha256 %s) '
           'and step13b_analysis.json (sha256 %s; parts a to c); pathway_vs_fingerprint_v2.json (sha256 %s; part d, '
           'script pathway_vs_fingerprint_v2.py; it supersedes pathway_vs_fingerprint.json, sha256 %s, from whose values '
           'its own differ by less than 0.001); '
           'ridge_vs_mlp_v2.json (sha256 %s; part e, script ridge_vs_mlp_v2.py). The values of parts a and c that are '
           'drawn in Fig. 3b and 3c equal those the figures were drawn from (fig3_values.csv). The numbers in the '
           'description of the variants are those of the pre-registrations of the three experiments. Assembled by '
           'make_supp_table_s10.py.' % tuple(IN[k][1][:16] for k in ('s9', 's10', 's13', 's13b', 'lin2', 'lin', 'mlp'))]
    if DASH in '\n'.join(md):
        die('a cell of the table has no value')
    write_outputs(TABLE, long_rows, md)
    print('[checks] 1 (%d comparisons of the four experiments), 2 (%d values of Fig. 3b and 3c), 3 (%d values of the two '
          'exploratory records, with the summaries over %d fold assignments recomputed), 4 (%d numbers of the description '
          'found in the pre-registrations), 5 (statements of Results 3, of Supplementary Note 1 and of the two captions) '
          'all passed' % (n1, n2, n3, N_FOLDSETS, n4))


if __name__ == '__main__':
    main()
