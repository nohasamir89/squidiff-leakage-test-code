#!/usr/bin/env python3
"""
make_supp_table_s9.py - Supplementary Table S9: settings of the experimental references, number of sampling steps
of every configuration, and entries of the shifted cells set to zero.

Purpose
-------
The paper cites Supplementary Table S9 three times: the Methods for the full settings of the references,
Results 4 for the step count of every configuration, and Results 2 for the fraction of entries set to zero
("12-44 % on sci-Plex and 20-42 % on Tahoe-100M (per reference and split ...)"). This script assembles the three
parts from committed files, so that every value traces to its source.

Three parts:
  a  settings of the references (text of the Methods; the numbers in it are checked or derived here);
  b  number of sampling steps and sampler of every evaluated configuration, per split, with the source of each
     value;
  c  fraction of entries of the shifted cells set to zero, per reference and split.

Logic
-----
- Inputs (committed files only; see supp_table_common.guard):
      leakage_results/step14_analysis.json          inputs/<split>/<model>: the evaluated sci-Plex runs (tags); cap
      tahoe_results/step15_model_analysis.json      the same for Tahoe-100M
      tahoe_results/step15_data_record.json         the 14 cell lines (for the number of features)
      leakage_results/step14_analysis.log           '# <split>' and '[baseline] <reference> ... (entries clipped at
      tahoe_results/step15_score_logs/s15score_final.log     0: x%)': the fractions of part c
      sample_squidiff.py                            the sampler; its default for the diffusion models
      provenance/*.json                             one record per evaluation run, written by the evaluation job:
                                                    tag, and for most runs nfe (number of steps) and sampler
  The provenance records are many small files; each one used must be tracked by git and identical to HEAD (no
  sha256 is pinned for them: the outputs of this script are compared with the reference run instead).
- Part b, source of each step count, in this order:
    1. the field nfe of the run's provenance record ("record");
    2. if the record has no step count and the run name ends in _n<number> (the first sweep on Split 1), that
       number ("run name");
    3. if the record has no step count and the run is a diffusion model sampled with the sampler's defaults (the
       original model), the default of sample_squidiff.py, read from that file ('diffusion_steps': 1000 with
       'use_ddim': True) ("sampler default").
  All seeds of a configuration on a split must agree, else the script stops.
- Part a: the number of features is derived (1,024 fingerprint bits + L cell lines + 1,024 x L products) and must
  equal the numbers of the Methods; the cap of shifted cells per condition is read from the two analysis records.
  The penalty of the ridge (alpha = 1.0) is a constant of score_step14.py and score_step15_tahoe.py, not a record
  field; it is stated here as in the Methods.
- Checks, all before anything is written (a failure stops the script):
    1. every evaluated run has exactly one provenance record, and that record names the same run;
    2. the step counts agree within a configuration and split, and the statements of the paper hold (EXPECTED);
    3. each log gives one fraction per reference and split, and the ranges quoted in Results 2 reproduce.
- Outputs (never overwritten): supplementary_tables/supp_table_S9.csv and .md (see supp_table_common).

Run from the research worktree root (supp_table_common.py must be in the same folder):
    python3 make_supp_table_s9.py
"""
import glob
import json
import re
import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import DASH, SPLITS, die, git, guard, load_json, same, table, write_outputs

TABLE = 'S9'
IN = dict(
    sciplex=('leakage_results/step14_analysis.json',
             'c7f021b27272bbac983db2f58d694dddc5dd7de234869ed8ec5d5e64bcf6f0ae'),
    tahoe=('tahoe_results/step15_model_analysis.json',
           '5fd700b5a030cf5a8ffea57e9a9b2ad840aa32fa9948b32754b9742362a1165e'),
    data=('tahoe_results/step15_data_record.json',
          '06c4b547a8a1831a8e7fcaa1e7dc1d91c1bfbec850c8893b5b643b5b7c7195ba'),
    log_sciplex=('leakage_results/step14_analysis.log',
                 'a94544ef01bdd114fa0faf648864b8560161819a046ead8efb4bd7a32496c94e'),
    log_tahoe=('tahoe_results/step15_score_logs/s15score_final.log',
               'c23734e7fa4dcd853ef173a2ab0059efcaf71f0ff7bf65835afb8cb703ff932a'),
    sampler=('sample_squidiff.py',
             '092d7580032939f0b3a9b381cb1d46823da7eff14159b3ef1edd0c5d89fad0ed'),
)
PROV_DIR = 'provenance'
FP_BITS = 1024
SCIPLEX_LINES = 3                    # A549, K562, MCF7 (Methods)

# Configurations in the order of Supplementary Table S6: (name in the paper, key in the analysis record).
CONFIG = dict(
    sciplex=[('Original model', 'base'),
             ('Original model with adaptive normalisation', 'adaln_ddim_1000'),
             ('Rewired model', 'fm_50'),
             ('Rewired model, 20 integration steps', 'fm_20'),
             ('Rewired model with classifier-free guidance (weight 2)', 'guidance_w2'),
             ('Rewired model with cross-attention', 'attn_cross'),
             ('Rewired model, parameter-matched control without attention', 'mlp_ctrl'),
             ('Variant: mechanism-of-action labels in place of the fingerprint', 's9_pathway'),
             ('Variant: mechanism-of-action labels added to the fingerprint', 's9_both'),
             ('Variant: learned chemical embedding in place of the fingerprint', 's10_chemvae'),
             ('Variant: learned chemical embedding added to the fingerprint', 's10_fp_chemvae')],
    tahoe=[('Original model', 'base'),
           ('Rewired model', 'recipe'),
           ('Rewired model with classifier-free guidance (weight 2)', 'guidance_w2'),
           ('Rewired model with cross-attention', 'attn_cross'),
           ('Rewired model, parameter-matched control without attention', 'mlp_ctrl')],
)
DIFFUSION = ('base', 'adaln_ddim_1000')          # sampled with DDIM; every other configuration is flow matching
REFERENCES = [('Training-set ridge', 'ridge_train'), ('Cross-validated ridge', 'ridge_S7'),
              ('Compound-blind average', 'mean_response')]

# What the paper states (checks only, never inputs).
EXPECTED = dict(
    features=dict(sciplex=4099, tahoe=15374),                    # Methods, Experimental references
    cap=200,                                                     # "at most 200 per condition"
    steps={('sciplex', 'base'): 1000, ('tahoe', 'base'): 1000,   # Results 4 and the caption of Table S6
           ('sciplex', 'fm_50'): 50, ('tahoe', 'recipe'): 50, ('sciplex', 'fm_20'): 20},
    zero_range=dict(sciplex=('12', '44'), tahoe=('20', '42')),   # Results 2: "12-44 % ... and 20-42 %"
)


def n(x):
    return '{:,}'.format(int(x))


def tracked_and_committed(path):
    if git('ls-files', '--error-unmatch', path)[0] != 0:
        die('%s is not tracked by git' % path)
    if git('diff', '--quiet', 'HEAD', '--', path)[0] != 0:
        die('%s differs from HEAD; this script reads committed records only' % path)


def fractions(path):
    """{split: {reference: percent}} from the '[baseline]' lines of a scoring log; the split is the last '# <split>'."""
    out, split = {}, None
    for line in open(path, encoding='utf-8'):
        m = re.match(r'# (drug_split_\d)\s*$', line)
        if m:
            split = m.group(1)
            continue
        m = re.match(r'\[baseline\] (\S+) .*\(entries clipped at 0: ([0-9.]+)%\)', line)
        if m:
            if split is None:
                die('%s: a [baseline] line comes before any split' % path)
            if m.group(1) in out.setdefault(split, {}):
                die('%s: two [baseline] lines for %s on %s' % (path, m.group(1), split))
            out[split][m.group(1)] = float(m.group(2))
    return out


def main():
    guard(IN, TABLE)
    rec = dict(sciplex=load_json(IN['sciplex'][0]), tahoe=load_json(IN['tahoe'][0]))
    data = load_json(IN['data'][0])
    sampler_src = open(IN['sampler'][0], encoding='utf-8').read()
    m = re.search(r"'diffusion_steps':\s*(\d+)", sampler_src)
    if not m or not re.search(r"'use_ddim':\s*True", sampler_src):
        die('sample_squidiff.py does not state its default number of diffusion steps and the DDIM sampler')
    default_steps = int(m.group(1))

    # ---- provenance records: one per run --------------------------------------------------------------------------
    by_tag = {}
    for path in sorted(glob.glob(PROV_DIR + '/*.json')):
        try:
            d = json.load(open(path, encoding='utf-8'))
        except ValueError:
            continue
        if isinstance(d, dict) and 'tag' in d:
            by_tag.setdefault(d['tag'], []).append((path, d))

    def steps_of(dataset, model, run):
        """(steps, sampler, source) of one evaluated run; check 1 inside."""
        found = by_tag.get(run['tag'], [])
        if len(found) != 1:
            die('check 1: %d provenance records for the run %s' % (len(found), run['tag']))
        path, d = found[0]
        tracked_and_committed(path)
        nfe, sampler = d.get('nfe'), d.get('sampler')
        if isinstance(nfe, int) and not isinstance(nfe, bool):
            return nfe, sampler or ('ddim' if model in DIFFUSION else 'euler_ode'), 'record'
        tail = re.search(r'_n(\d+)$', run['tag'])
        if tail:
            return int(tail.group(1)), sampler or ('ddim' if model in DIFFUSION else 'euler_ode'), 'run name'
        if model == 'base' and sampler in (None, 'ddim'):
            return default_steps, 'ddim', 'sampler default'
        die('no step count for the run %s (%s)' % (run['tag'], path))

    # ---- check 2 and the cells of part b ------------------------------------------------------------------------------
    cells, n_runs = {}, 0
    for dataset in ('sciplex', 'tahoe'):
        placed = set(k for _, k in CONFIG[dataset])
        for s in SPLITS:
            same(set(rec[dataset]['inputs'][s]) - placed, set(), 'configurations of %s on %s without a row' % (dataset, s))
            for label, key in CONFIG[dataset]:
                runs = rec[dataset]['inputs'][s].get(key)
                if runs is None:
                    cells[(dataset, key, s)] = None
                    continue
                got = set(steps_of(dataset, key, r) for r in runs)
                n_runs += len(runs)
                if len(got) != 1:
                    die('check 2: the runs of %s on %s (%s) do not agree: %r' % (key, s, dataset, sorted(got)))
                cells[(dataset, key, s)] = got.pop()
    for (dataset, key), want in EXPECTED['steps'].items():
        for s in SPLITS:
            if cells[(dataset, key, s)] is not None:
                same(cells[(dataset, key, s)][0], want, 'check 2, steps of %s on %s (%s)' % (key, s, dataset))

    # ---- check 3: fractions set to zero ---------------------------------------------------------------------------------
    frac = dict(sciplex=fractions(IN['log_sciplex'][0]), tahoe=fractions(IN['log_tahoe'][0]))
    for dataset in ('sciplex', 'tahoe'):
        same(sorted(frac[dataset]), sorted(SPLITS), 'check 3, splits of the %s log' % dataset)
        vals = []
        for s in SPLITS:
            for _, key in REFERENCES:
                if key not in frac[dataset][s]:
                    die('check 3: no fraction for %s on %s in the %s log' % (key, s, dataset))
                vals.append(frac[dataset][s][key])
            same(frac[dataset][s].get('control_paste'), 0.0, 'check 3, unaltered vehicle cells on %s (%s)' % (s, dataset))
        same(('%.0f' % min(vals), '%.0f' % max(vals)), EXPECTED['zero_range'][dataset],
             'check 3, range of the fractions on %s' % dataset)

    # ---- part a: numbers --------------------------------------------------------------------------------------------------
    n_lines = dict(sciplex=SCIPLEX_LINES, tahoe=len(set(c['line'] for c in data['conditions'])))
    feat = dict((k, FP_BITS + v + FP_BITS * v) for k, v in n_lines.items())
    same(feat, EXPECTED['features'], 'number of features of the ridge')
    same((rec['sciplex']['cap'], rec['tahoe']['cap']), (EXPECTED['cap'], EXPECTED['cap']), 'cap of shifted cells per condition')
    cap = rec['sciplex']['cap']

    # ---- the table ----------------------------------------------------------------------------------------------------------
    long_rows = []

    def keep(part, row, column, value, source):
        long_rows.append(dict(part=part, row=row, column=column, value=value, source=source))

    md = ['Supplementary Table S9 | Settings of the experimental references, sampling steps and entries set to zero '
          '(Methods; Results 2 and 4). a, Settings of the references. b, Number of sampling steps of every evaluated '
          'configuration on splits 1 / 0 / 2 (a dash: not evaluated on that split). The diffusion models were sampled '
          'with DDIM and the flow-matching models with the Euler method. Source of the step count: record, the '
          'evaluation record of each run; run name, the number carried by the name of the run; sampler default, the '
          'default of the sampler, with which the original model was sampled. c, Entries of the shifted cells set to '
          'zero after the shift was added, in per cent of all entries, on splits 1 / 0 / 2.',
          '', '**a** | Settings of the references.', '']
    rows_a = [
        ['Unaltered vehicle cells', 'The matched vehicle cells, returned unchanged.'],
        ['Training-set ridge', 'Multi-output ridge regression (scikit-learn Ridge, penalty 1.0 fixed before any '
         'result, intercept, unscaled features). Features: the %s-bit fingerprint, a one-hot cell-line indicator '
         'and their products (%s features on sci-Plex, %s on Tahoe-100M). Target: the mean expression change of a '
         'condition from its matched vehicle cells. Fitted on the training conditions.'
         % (n(FP_BITS), n(feat['sciplex']), n(feat['tahoe']))],
        ['Cross-validated ridge', 'The same regression, fitted within the held-out compounds by five-fold '
         'cross-validation with folds grouped by structure.'],
        ['Compound-blind average', 'One shift per cell line: the mean change over the held-out conditions of that '
         'line.'],
        ['Use of the three shifts', 'The shift is added to the matched vehicle cells and negative values are set to '
         'zero; one shifted cell per measured cell, at most %s per condition.' % n(cap)],
    ]
    for r in rows_a:
        keep('a', r[0], 'setting', r[1], 'Methods; score_step14.py; score_step15_tahoe.py')
    md += table(['Reference', 'Setting'], rows_a)

    md += ['', '**b** | Sampling steps of every configuration (splits 1 / 0 / 2).', '']
    rows_b = []
    for dataset, title in (('sciplex', 'sci-Plex'), ('tahoe', 'Tahoe-100M')):
        for i, (label, key) in enumerate(CONFIG[dataset]):
            got = [cells[(dataset, key, s)] for s in SPLITS]
            steps = ' / '.join(DASH if g is None else n(g[0]) for g in got)
            samplers = sorted(set(g[1] for g in got if g is not None))
            sources = sorted(set(g[2] for g in got if g is not None))
            same(len(samplers), 1, 'one sampler for %s (%s)' % (key, dataset))
            src = ' / '.join(DASH if g is None else g[2] for g in got) if len(sources) > 1 else sources[0]
            rows_b.append([title if i == 0 else '', label, steps, 'DDIM' if samplers[0] == 'ddim' else 'Euler', src])
            for s, g in zip(SPLITS, got):
                keep('b', '%s: %s' % (title, label), s, '' if g is None else g[0], '' if g is None else g[2])
    md += table(['Dataset', 'Configuration', 'Steps', 'Sampler', 'Source of the step count'], rows_b)

    md += ['', '**c** | Entries set to zero (%; splits 1 / 0 / 2).', '']
    rows_c = []
    for label, key in REFERENCES:
        row = [label]
        for dataset, src in (('sciplex', 'step14_analysis.log'), ('tahoe', 's15score_final.log')):
            vals = [frac[dataset][s][key] for s in SPLITS]
            row.append(' / '.join('%.1f' % v for v in vals))
            for s, v in zip(SPLITS, vals):
                keep('c', label, '%s, %s' % (dataset, s), v, src)
        rows_c.append(row)
    md += table(['Reference', 'sci-Plex', 'Tahoe-100M'], rows_c)

    md += ['',
           'Sources: part a, Methods, with the number of features derived and the cap read from '
           'step14_analysis.json and step15_model_analysis.json; part b, the evaluation records in provenance/ of '
           'the %d runs listed in those two files, and sample_squidiff.py for the sampler default (%s steps); part '
           'c, step14_analysis.log and s15score_final.log. Assembled by make_supp_table_s9.py.' % (n_runs, n(default_steps))]
    write_outputs(TABLE, long_rows, md)
    print('[checks] 1 (%d runs, one evaluation record each), 2 (step counts agree and match the paper), '
          '3 (fractions set to zero) all passed' % n_runs)


if __name__ == '__main__':
    main()
