#!/usr/bin/env python3
"""
make_supp_table_s1.py - Supplementary Table S1: the datasets of the study (cell lines, plates, compound splits, seeds).

Purpose
-------
Results 1 cites Supplementary Table S1 for the design on Tahoe-100M ("14 brain, gastric and colorectal cancer cell
lines ... three compound splits that each held out 75 chemical structures, with three seeds per split"), and the
Methods cite it for the 14 cell lines. This script assembles the table from the committed records, so that every
count traces to a record.

Four parts:
  a  the 14 Tahoe-100M cell lines: tissue, plates, compounds, conditions and treated cells;
  b  the 13 plates: cells in the plate file, cell lines, compounds, conditions and treated cells;
  c  the three compound splits of each dataset: held-out structures or compounds, test and training conditions
     and, on Tahoe-100M, the cells of the test and training files;
  d  the training seeds of the models on each dataset.

Logic
-----
- Inputs (committed records only; see supp_table_common.guard):
      tahoe_results/step15_data_record.json      written by prep_step15_tahoe.py: constants, inputs/plates/<plate>,
                                                 drugs, splits/<k>, conditions (one entry per condition: plate,
                                                 line, drug, dose, n_cells, n_test, n_train, held_split),
                                                 skipped_conditions
      tahoe_results/step15_model_analysis.json   inputs/<split>/<model>: the evaluated runs, one tag per seed
      leakage_results/step14_analysis.json       the same for sci-Plex
      leakage_results/pathway_vs_fingerprint_v2.json   folds/<split>: the held-out compounds of each sci-Plex
                                                 split and their number of conditions
- A condition is cell line x compound name x dose x plate (Methods). "Treated cells" is the number of treated
  cells of a condition on its plate before any cap (n_cells). The test file holds at most 500 cells per condition
  and the training file at most 150 (constants of the record).
- The tissue of each cell line is NOT in a record. It is the grouping stated in the Methods (three brain, two
  gastric, nine colorectal), held below as a constant; the three brain lines are also listed in
  score_step15_tahoe.py (BRAIN). The script checks that the constant names exactly the 14 lines of the record.
- Checks, all before anything is written (a failure stops the script):
    1. Internal consistency of the data record: for every split, the conditions marked as held out equal the
       record's own count and the conditions of its test file; the others equal the training count; the sum of the
       capped cells equals the rows of the test and of the training file; the held-out sets are disjoint and hold
       75 structures each; the conditions of all plates add up; every condition has one of the three doses.
    2. The analysis record of Tahoe-100M was computed on this data record (its data_record_sha256), reports the
       same number of test conditions per split, and every model has the same seeds on every split. The same seed
       check for sci-Plex.
    3. The counts the manuscript quotes reproduce (EXPECTED below; checks only, never inputs to a computation).
- Outputs (never overwritten): supplementary_tables/supp_table_S1.csv and .md (see supp_table_common).

Run from the research worktree root (supp_table_common.py must be in the same folder):
    python3 make_supp_table_s1.py
"""
import re
import sys

sys.dont_write_bytecode = True      # no __pycache__ folder next to the scripts (file quota on /scratch)

from supp_table_common import SPLITS, SPLIT_HEAD, die, guard, load_json, same, table, write_outputs

TABLE = 'S1'
IN = dict(
    data=('tahoe_results/step15_data_record.json',
          '06c4b547a8a1831a8e7fcaa1e7dc1d91c1bfbec850c8893b5b643b5b7c7195ba'),
    tahoe=('tahoe_results/step15_model_analysis.json',
           '5fd700b5a030cf5a8ffea57e9a9b2ad840aa32fa9948b32754b9742362a1165e'),
    sciplex=('leakage_results/step14_analysis.json',
             'c7f021b27272bbac983db2f58d694dddc5dd7de234869ed8ec5d5e64bcf6f0ae'),
    note1=('leakage_results/pathway_vs_fingerprint_v2.json',
           'aebc1c61610bd1c0cbf29269facfdb051474293c9ac13dfb2efbe45e5cb58066'),
)

# Tissue of each cell line: the grouping of the Methods, in the order the Methods name the lines. Not from a record.
TISSUE = [
    ('A-172', 'Brain'), ('H4', 'Brain'), ('SW 1088', 'Brain'),
    ('KATO III', 'Gastric'), ('SNU-1', 'Gastric'),
    ('COLO 205', 'Colorectal'), ('HCT15', 'Colorectal'), ('HT-29', 'Colorectal'), ('LS 180', 'Colorectal'),
    ('LoVo', 'Colorectal'), ('RKO', 'Colorectal'), ('SW1417', 'Colorectal'), ('SW48', 'Colorectal'),
    ('SW480', 'Colorectal'),
]
# Counts the manuscript quotes (checks only, never inputs).
EXPECTED = dict(
    plates=13, lines=14, names=375, structures=372, held=75,     # Methods, Datasets and Compound splits
    doses=[50.0, 500.0, 5000.0],                                 # "Doses were 50, 500 and 5,000 nM"
    test_conditions=[2899, 3115, 2853],                          # splits 1, 0, 2: "(2,899, 3,115 and 2,853 conditions)"
    skipped=77,                                                  # "77 conditions of one line on one plate"
    cap_train=150, cap_test=500,                                 # "capped at 150 per condition ... test cells at 500"
    tahoe_seeds=3,                                               # Results 1: "three seeds per split"
    sciplex_test_conditions=[443, 444, 455], sciplex_compounds=[37, 37, 38],   # splits 1, 0, 2
)


def n(x):
    """An integer with thousands separators, as the manuscript prints counts."""
    return '{:,}'.format(int(x))


def seeds_of(runs, what):
    """Seed numbers of the runs of one model, read from the end of each tag ('..._seed_2' or '..._seed_2_n50')."""
    out = []
    for r in runs:
        m = re.search(r'seed_(\d+)', r['tag'])
        if not m:
            die('no seed in the tag %r (%s)' % (r['tag'], what))
        out.append(int(m.group(1)))
    return out


def main():
    guard(IN, TABLE)
    rec = load_json(IN['data'][0])
    con, cst = rec['conditions'], rec['constants']
    plates = [int(p) for p in cst['plates']]
    lines = [l for l, _ in TISSUE]
    same(sorted(set(c['line'] for c in con)), sorted(lines), 'cell lines of the record against the tissue list')

    # ---- check 1: internal consistency of the data record ---------------------------------------------------------
    same(sorted(set(c['dose'] for c in con)), EXPECTED['doses'], 'check 1, doses of the conditions')
    same(sorted(set(c['plate'] for c in con)), plates, 'check 1, plates of the conditions')
    held_sets = []
    for k in range(3):
        sp = rec['splits'][str(k)]
        held_sets.append(set(sp['held_structures']))
        same(len(sp['held_structures']), cst['n_held'], 'check 1, held-out structures of split %d' % k)
        test = [c for c in con if c['held_split'] == k]
        train = [c for c in con if c['held_split'] != k]
        same(len(test), sp['n_test_conditions'], 'check 1, test conditions of split %d' % k)
        same(len(train), sp['n_train_conditions'], 'check 1, training conditions of split %d' % k)
        same(sp['files']['test']['n_conditions'], len(test), 'check 1, conditions of the test file of split %d' % k)
        same(sp['files']['train']['n_conditions'], len(train), 'check 1, conditions of the training file of split %d' % k)
        same(sum(c['n_test'] for c in test), sp['files']['test']['rows'], 'check 1, cells of the test file of split %d' % k)
        same(sum(c['n_train'] for c in train), sp['files']['train']['rows'],
             'check 1, cells of the training file of split %d' % k)
        same(set(c['drug'] for c in test), set(sp['held_drugs']), 'check 1, held-out compound names of split %d' % k)
    if held_sets[0] & held_sets[1] or held_sets[0] & held_sets[2] or held_sets[1] & held_sets[2]:
        die('check 1: the held-out sets of the three splits are not disjoint')
    for c in con:
        if c['n_test'] != min(cst['cap_test'], c['n_cells']) or c['n_train'] != min(cst['cap_train'], c['n_cells']):
            die('check 1: the caps do not explain the cells of condition %r' % c)
    same(len(set(rec['drugs']['structure'].values())), rec['drugs']['n_structures'], 'check 1, number of structures')
    same(len(set(c['drug'] for c in con)), rec['drugs']['n_usable'], 'check 1, compound names in the conditions')

    # ---- check 2: the analysis records -----------------------------------------------------------------------------
    tah, sci, note1 = load_json(IN['tahoe'][0]), load_json(IN['sciplex'][0]), load_json(IN['note1'][0])
    same(tah['data_record_sha256'], IN['data'][1], 'check 2, data record named by the Tahoe analysis record')
    for s in SPLITS:
        same(tah['per_split'][s]['n_conditions'], rec['splits'][s[-1]]['n_test_conditions'],
             'check 2, test conditions of %s in the analysis record' % s)
    seeds = {}
    for name, r in (('tahoe', tah), ('sciplex', sci)):
        found = set()
        for s in SPLITS:
            for model, runs in r['inputs'][s].items():
                found.add(tuple(seeds_of(runs, '%s, %s, %s' % (name, s, model))))
        if len(found) != 1:
            die('check 2: the models of %s do not all have the same seeds on every split: %r' % (name, sorted(found)))
        seeds[name] = list(found.pop())
        same(seeds[name], list(range(len(seeds[name]))), 'check 2, seeds of %s are 0, 1, 2, ...' % name)
    sci_comp = [len(note1['folds'][s]['compounds_sorted_smiles']) for s in SPLITS]
    sci_cond = [sum(int(x) for x in note1['folds'][s]['conditions_per_compound']) for s in SPLITS]

    # ---- check 3: the counts the manuscript quotes -------------------------------------------------------------------
    order = [1, 0, 2]                                        # the paper's order of the splits
    same(len(plates), EXPECTED['plates'], 'check 3, number of plates')
    same(len(lines), EXPECTED['lines'], 'check 3, number of cell lines')
    same(rec['drugs']['n_usable'], EXPECTED['names'], 'check 3, compound names')
    same(rec['drugs']['n_structures'], EXPECTED['structures'], 'check 3, structures')
    same(cst['n_held'], EXPECTED['held'], 'check 3, held-out structures per split')
    same([rec['splits'][str(k)]['n_test_conditions'] for k in order], EXPECTED['test_conditions'], 'check 3, test conditions')
    same(rec['skipped_conditions']['n'], EXPECTED['skipped'], 'check 3, conditions without vehicle cells')
    same((cst['cap_train'], cst['cap_test']), (EXPECTED['cap_train'], EXPECTED['cap_test']), 'check 3, caps')
    same(len(seeds['tahoe']), EXPECTED['tahoe_seeds'], 'check 3, seeds on Tahoe-100M')
    same(sci_cond, EXPECTED['sciplex_test_conditions'], 'check 3, sci-Plex test conditions')
    same(sci_comp, EXPECTED['sciplex_compounds'], 'check 3, sci-Plex held-out compounds')

    # ---- the table -------------------------------------------------------------------------------------------------------
    long_rows = []

    def keep(part, row, column, value, source):
        long_rows.append(dict(part=part, row=row, column=column, value=value, source_record=source))

    skipped = rec['skipped_conditions']
    (sk_key,) = list(skipped['by_plate_line'])               # one plate and line: '13|SW 1088'
    sk_plate, sk_line = sk_key.split('|')
    md = ['Supplementary Table S1 | Datasets of the study (Results 1; Methods). a, The %d Tahoe-100M cell lines. A '
          'condition is one cell line, compound name, dose and plate; treated cells are counted on the plate before '
          'any cap. b, The %d plates used (plate 12 was excluded before analysis). c, The three compound splits of '
          'each dataset. On Tahoe-100M each split holds out %d of the %s chemical structures (%s compound names); '
          'the test file holds at most %s cells per condition and the training file at most %s. d, Training seeds. '
          '%s conditions of %s on plate %s had no vehicle cells on their plate and are not counted (%s treated '
          'cells).'
          % (len(lines), len(plates), cst['n_held'], n(rec['drugs']['n_structures']), n(rec['drugs']['n_usable']),
             n(cst['cap_test']), n(cst['cap_train']), n(skipped['n']), sk_line, sk_plate, n(skipped['n_cells'])),
          '', '**a** | Cell lines of the Tahoe-100M subset.', '']
    D = 'step15_data_record.json'
    rows = []
    for line, tissue in TISSUE:
        cs = [c for c in con if c['line'] == line]
        vals = [len(set(c['plate'] for c in cs)), len(set(c['drug'] for c in cs)), len(cs), sum(c['n_cells'] for c in cs)]
        rows.append([line, tissue] + [n(v) for v in vals])
        keep('a', line, 'tissue', tissue, 'Methods (not a record)')
        for col, v in zip(('plates', 'compounds', 'conditions', 'treated cells'), vals):
            keep('a', line, col, v, D)
    tot = [len(plates), len(set(c['drug'] for c in con)), len(con), sum(c['n_cells'] for c in con)]
    rows.append(['All', ''] + [n(v) for v in tot])
    for col, v in zip(('plates', 'compounds', 'conditions', 'treated cells'), tot):
        keep('a', 'All', col, v, D)
    md += table(['Cell line', 'Tissue', 'Plates', 'Compounds', 'Conditions', 'Treated cells'], rows)

    md += ['', '**b** | Plates of the Tahoe-100M subset. Cells in the plate file include the vehicle cells.', '']
    rows = []
    for p in plates:
        cs = [c for c in con if c['plate'] == p]
        vals = [rec['inputs']['plates'][str(p)]['n_cells'], len(set(c['line'] for c in cs)),
                len(set(c['drug'] for c in cs)), len(cs), sum(c['n_cells'] for c in cs)]
        rows.append([str(p)] + [n(v) for v in vals])
        for col, v in zip(('cells in the plate file', 'cell lines', 'compounds', 'conditions', 'treated cells'), vals):
            keep('b', 'plate %d' % p, col, v, D)
    same(sum(len([c for c in con if c['plate'] == p]) for p in plates), len(con), 'conditions summed over plates')
    md += table(['Plate', 'Cells in the plate file', 'Cell lines', 'Compounds', 'Conditions', 'Treated cells'], rows)

    md += ['', '**c** | Compound splits. Split 1 was used for development and Splits 0 and 2 for confirmation.', '']
    rows = []
    spec = [('Held-out structures', lambda sp: len(sp['held_structures'])),
            ('Held-out compound names', lambda sp: len(sp['held_drugs'])),
            ('Test conditions', lambda sp: sp['n_test_conditions']),
            ('Cells in the test file', lambda sp: sp['files']['test']['rows']),
            ('Training conditions', lambda sp: sp['n_train_conditions']),
            ('Cells in the training file', lambda sp: sp['files']['train']['rows'])]
    for i, (label, f) in enumerate(spec):
        vals = [f(rec['splits'][str(k)]) for k in order]
        rows.append(['Tahoe-100M' if i == 0 else '', label] + [n(v) for v in vals])
        for s, v in zip(SPLITS, vals):
            keep('c', 'Tahoe-100M: ' + label, s, v, D)
    for i, (label, vals) in enumerate((('Held-out compounds', sci_comp), ('Test conditions', sci_cond))):
        rows.append(['sci-Plex' if i == 0 else '', label] + [n(v) for v in vals])
        for s, v in zip(SPLITS, vals):
            keep('c', 'sci-Plex: ' + label, s, v, 'pathway_vs_fingerprint_v2.json')
    md += table(['Dataset', 'Quantity'] + SPLIT_HEAD, rows)

    md += ['', '**d** | Training seeds. Every evaluated model had the same seeds on every split.', '']
    rows = []
    for label, key, src in (('Tahoe-100M', 'tahoe', 'step15_model_analysis.json'),
                            ('sci-Plex', 'sciplex', 'step14_analysis.json')):
        cell = ', '.join(str(k) for k in seeds[key])
        rows.append([label, str(len(seeds[key])), cell])
        keep('d', label, 'number of seeds', len(seeds[key]), src)
        keep('d', label, 'seeds', cell, src)
    md += table(['Dataset', 'Seeds per model and split', 'Seed numbers'], rows)

    md += ['',
           'Sources: step15_data_record.json (sha256 %s; parts a to c, Tahoe-100M), pathway_vs_fingerprint_v2.json '
           '(part c, sci-Plex), step15_model_analysis.json and step14_analysis.json (part d). The tissue of each cell '
           'line is the grouping stated in the Methods. Assembled by make_supp_table_s1.py.' % IN['data'][1][:16]]
    write_outputs(TABLE, long_rows, md)
    print('[checks] 1 (data record consistent with itself), 2 (analysis records), 3 (counts quoted in the manuscript) '
          'all passed')


if __name__ == '__main__':
    main()
