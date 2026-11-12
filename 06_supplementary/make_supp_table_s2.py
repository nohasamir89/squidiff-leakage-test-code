#!/usr/bin/env python3
"""
make_supp_table_s2.py - Supplementary Table S2: the training objective on training and on held-out cells (Results 1).

Purpose
-------
Results 1 asks whether the failure of the original model is overfitting. The answer is given by one comparison: each
model's own training objective, evaluated on cells of held-out compounds and on training cells by the same
procedure. The text quotes the ratio of the two losses (0.91 and 1.02 for the original model; 0.92-1.11 for the two
models with adaptive normalisation, one case below 1) and cites Supplementary Table S2 for the values. This script
assembles that table from the two committed records, so that every number in it traces to a record and is rounded
once, from full precision.

One row per model and split (three models x the two sci-Plex confirmation splits):
    Original model                                   diffusion objective    test_loss_base.json  (arm "base")
    Original model with adaptive normalisation       diffusion objective    test_loss_v2.json    (arm "ddim")
    Rewired model (adaptive normalisation + flow     flow-matching          test_loss_v2.json    (arm "fm_gaussian")
    matching)                                        objective
Columns: training loss and held-out loss (mean over the five seeds, sample SD over seeds), and held-out / training
(mean over seeds of the per-seed ratio, with its range over seeds).

Logic
-----
- Inputs (committed records only). The script stops unless each input is tracked by git, identical to HEAD, and has
  the sha256 it had when this table was designed (SHA256 below):
      leakage_results/test_loss_base.json   per_checkpoint/<split>_base_seed<k>: train, test, ratio; summary/<split>
      leakage_results/test_loss_v2.json     per_checkpoint/<split>_<arm>_seed<k>: train, test, ratio
- Arithmetic: mean = sum / 5; SD = sqrt(sum of squared deviations / 4) (sample SD); ratio = mean of the five
  per-seed ratios (held-out / training of the same checkpoint); range = smallest and largest per-seed ratio.
  Sums use math.fsum, so the result does not depend on the Python version.
- Checks, all before anything is written (a failure stops the script):
    1. Each model and split has exactly the seeds 0-4, and every stored ratio equals held-out / training (1e-12).
    2. The summary block that test_loss_base.json carries (written when the original model was scored) equals the
       summary recomputed here from its ten runs (1e-12).
    3. The reproduction gate stored in test_loss_base.json refers to the value that test_loss_v2.json holds for the
       same checkpoint (exact), and its recomputation agrees with it to within 1e-9.
    4. Both records are marked exploratory, as the caption states.
    5. The values quoted in the manuscript reproduce (EXPECTED below): the ratios in Results 1 at two decimals, and
       the two rows drafted for the original model at their printed precision. EXPECTED holds printed values only;
       they are checks, never inputs to a computation.
- Rounding: losses and SDs with 5 decimals, ratios with 3; every value is rounded once from full precision.
- Outputs (never overwritten):
      supplementary_tables/supp_table_S2.csv   long format: one line per run and quantity, and one per summary value
      supplementary_tables/supp_table_S2.md    caption, table and sources in Markdown (also printed)
  The sha256 of both files is printed, so the files written on Jubail can be compared with a test run elsewhere.
  Nothing in the outputs depends on the date, the machine or the commit.

Reads JSON only (no numpy, scipy or anndata), so it runs on the login node.

Run from the research worktree root:
    python3 make_supp_table_s2.py
"""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------------------------------------------------------
# Inputs, outputs, fixed settings
# ---------------------------------------------------------------------------------------------------------------
REC = 'leakage_results'
IN = dict(
    base=os.path.join(REC, 'test_loss_base.json'),
    v2=os.path.join(REC, 'test_loss_v2.json'),
)
# sha256 of each record at research commit 02054d7 (the commit the table was designed on)
SHA256 = dict(
    base='2fa258669da2c667bafd8a2f3422f7661c5bc2f8e93ae29674c269ab6b6a38bb',
    v2='d5d9c48e469034d9d763cd51701712cf4ac1a16906a44542ccd4a06cc25371ac',
)
OUT_DIR = 'supplementary_tables'
OUT_CSV = os.path.join(OUT_DIR, 'supp_table_S2.csv')
OUT_MD = os.path.join(OUT_DIR, 'supp_table_S2.md')

SPLITS = ['drug_split_0', 'drug_split_2']             # the two sci-Plex confirmation splits
SPLIT_HEAD = {'drug_split_0': 'Split 0', 'drug_split_2': 'Split 2'}
SEEDS = [0, 1, 2, 3, 4]
# (row label, objective, record, arm key in that record)
MODELS = [
    ('Original model', 'Diffusion', 'base', 'base'),
    ('Original model with adaptive normalisation', 'Diffusion', 'v2', 'ddim'),
    ('Rewired model (adaptive normalisation and flow matching)', 'Flow matching', 'v2', 'fm_gaussian'),
]
EXPLORATORY = 'EXPLORATORY, not pre-registered'       # the "analysis" field both records must carry
TOL_EXACT = 1e-12         # checks 1-2: the same quantity computed twice
TOL_GATE = 1e-9           # check 3: the reproduction stated in the Sources line

# Printed values the manuscript quotes (checks only, never inputs).
#   Results 1: "0.91 times the training loss on one split and 1.02 times on the other" (original model);
#              "0.92-1.11 times the training loss, and in one case below it" (the two models with adaptive
#              normalisation: four model-by-split ratios).
#   Supplement additions, Table S2, rows drafted on 12 Nov 2026 for the original model.
EXPECTED_ORIGINAL_RATIO_2DP = {'drug_split_0': '0.91', 'drug_split_2': '1.02'}
EXPECTED_ADALN_RANGE_2DP = ('0.92', '1.11')
EXPECTED_ADALN_BELOW_ONE = 1
EXPECTED_DRAFTED_ROWS = {
    'drug_split_0': ('0.01223 (0.00008)', '0.01110 (0.00009)', '0.908 (0.903–0.914)'),
    'drug_split_2': ('0.01203 (0.00017)', '0.01232 (0.00011)', '1.024 (1.017–1.031)'),
}


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------
def die(msg):
    sys.exit('FATAL: ' + msg)


def git(*args):
    """Run one git command in the worktree; return (exit code, stripped stdout)."""
    p = subprocess.run(['git'] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return p.returncode, p.stdout.strip()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def guard():
    """Every input tracked, identical to HEAD and with its recorded sha256; no output present."""
    for key, path in IN.items():
        if not os.path.isfile(path):
            die('%s not found; run this script from the research worktree root' % path)
        if git('ls-files', '--error-unmatch', path)[0] != 0:
            die('%s is not tracked by git' % path)
        if git('diff', '--quiet', 'HEAD', '--', path)[0] != 0:
            die('%s differs from HEAD; this script reads committed records only' % path)
        if sha256_of(path) != SHA256[key]:
            die('%s is not the record this table was designed on (sha256 %s, expected %s)'
                % (path, sha256_of(path), SHA256[key]))
    for path in (OUT_CSV, OUT_MD):
        if os.path.exists(path):
            die('%s exists; outputs are never overwritten' % path)


def mean(xs):
    return math.fsum(xs) / len(xs)


def sample_sd(xs):
    """Sample standard deviation (n - 1 in the denominator)."""
    m = mean(xs)
    return math.sqrt(math.fsum((x - m) * (x - m) for x in xs) / (len(xs) - 1))    # a product, not a power: exact rounding on every machine


def close(a, b, tol, what):
    if a is None or b is None or not (abs(a - b) <= tol):
        die('%s: %r does not reproduce %r (tolerance %g)' % (what, a, b, tol))


def same(a, b, what):
    if a != b:
        die('%s: %r is not %r' % (what, a, b))


def loss_cell(m, sd):
    """'0.01223 (0.00008)': mean and sample SD, 5 decimals, each rounded once."""
    return '%.5f (%.5f)' % (m, sd)


def ratio_cell(m, lo, hi):
    """'0.908 (0.903-0.914)' with an en dash: mean ratio and its range over seeds, 3 decimals."""
    return '%.3f (%.3f–%.3f)' % (m, lo, hi)


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def main():
    guard()
    rec = {}
    for key, path in IN.items():
        with open(path) as f:
            rec[key] = json.load(f)

    # ---- check 4: both records are marked exploratory -----------------------------------------------------------
    for key in IN:
        same(rec[key].get('analysis'), EXPLORATORY, 'check 4, "analysis" field of %s' % IN[key])

    # ---- read the runs; check 1 -----------------------------------------------------------------------------------
    runs = {}        # (record, arm, split) -> list of (seed, train, test, ratio, source key)
    for label, objective, key, arm in MODELS:
        per = rec[key]['per_checkpoint']
        for split in SPLITS:
            prefix = '%s_%s_seed' % (split, arm)
            found = sorted(int(k[len(prefix):]) for k in per if k.startswith(prefix) and k[len(prefix):].isdigit())
            same(found, SEEDS, 'check 1, seeds of %s on %s' % (label, split))
            rows = []
            for seed in SEEDS:
                name = '%s%d' % (prefix, seed)
                r = per[name]
                train, test, ratio = float(r['train']), float(r['test']), float(r['ratio'])
                if not (train > 0 and test > 0):
                    die('check 1, %s: a loss is not positive' % name)
                close(ratio, test / train, TOL_EXACT, 'check 1, stored ratio of %s' % name)
                rows.append((seed, train, test, ratio, name))
            runs[(key, arm, split)] = rows

    # ---- summaries --------------------------------------------------------------------------------------------------
    summ = {}        # (record, arm, split) -> dict of the seven summary values
    for k, rows in runs.items():
        tr = [r[1] for r in rows]
        te = [r[2] for r in rows]
        ra = [r[3] for r in rows]
        summ[k] = dict(training_loss_mean=mean(tr), training_loss_sd=sample_sd(tr),
                       heldout_loss_mean=mean(te), heldout_loss_sd=sample_sd(te),
                       ratio_mean=mean(ra), ratio_min=min(ra), ratio_max=max(ra))

    # ---- check 2: the summary block stored in the original-model record -----------------------------------------------
    stored_to_here = dict(train_mean='training_loss_mean', train_sd='training_loss_sd',
                          test_mean='heldout_loss_mean', test_sd='heldout_loss_sd',
                          ratio_mean='ratio_mean', ratio_min='ratio_min', ratio_max='ratio_max')
    n_check2 = 0
    for split in SPLITS:
        block = rec['base']['summary'][split]
        for stored, here in stored_to_here.items():
            close(summ[('base', 'base', split)][here], float(block[stored]), TOL_EXACT,
                  'check 2, summary/%s/%s of %s' % (split, stored, IN['base']))
            n_check2 += 1

    # ---- check 3: the reproduction gate ---------------------------------------------------------------------------------
    gate = rec['base']['gate']
    gate_run = rec['v2']['per_checkpoint'].get(gate['key'])
    if gate_run is None:
        die('check 3: the gate key %r is not a run of %s' % (gate['key'], IN['v2']))
    same(float(gate['record_train']), float(gate_run['train']), 'check 3, gate record_train vs %s' % IN['v2'])
    same(float(gate['record_test']), float(gate_run['test']), 'check 3, gate record_test vs %s' % IN['v2'])
    gate_diff = max(abs(float(gate['train']) - float(gate_run['train'])), abs(float(gate['test']) - float(gate_run['test'])))
    if not gate_diff <= TOL_GATE:
        die('check 3: the gate recomputation differs from the record by %g (limit %g)' % (gate_diff, TOL_GATE))

    # ---- check 5: the values the manuscript quotes -------------------------------------------------------------------------
    for split in SPLITS:
        s = summ[('base', 'base', split)]
        same('%.2f' % s['ratio_mean'], EXPECTED_ORIGINAL_RATIO_2DP[split],
             'check 5, Results 1, ratio of the original model on %s' % split)
        cells = (loss_cell(s['training_loss_mean'], s['training_loss_sd']),
                 loss_cell(s['heldout_loss_mean'], s['heldout_loss_sd']),
                 ratio_cell(s['ratio_mean'], s['ratio_min'], s['ratio_max']))
        same(cells, EXPECTED_DRAFTED_ROWS[split], 'check 5, drafted row of the original model on %s' % split)
    adaln = [summ[('v2', arm, split)]['ratio_mean'] for arm in ('ddim', 'fm_gaussian') for split in SPLITS]
    same(('%.2f' % min(adaln), '%.2f' % max(adaln)), EXPECTED_ADALN_RANGE_2DP,
         'check 5, Results 1, range of the ratios of the models with adaptive normalisation')
    same(sum(1 for x in adaln if x < 1), EXPECTED_ADALN_BELOW_ONE,
         'check 5, Results 1, number of ratios below 1 among the models with adaptive normalisation')

    # ---- the table --------------------------------------------------------------------------------------------------------
    short = dict((k, os.path.basename(p)) for k, p in IN.items())
    md = ['Supplementary Table S2 | Training objective on training and on held-out cells (Results 1). The training '
          'objective of each model was evaluated on 2,560 training cells and on 2,560 cells of held-out compounds '
          '(40 batches of 64 cells, seed 0, uniform sampling of the diffusion time), for five seeds on each of the two '
          'sci-Plex confirmation splits. Losses are the mean over the five seeds, with the sample standard deviation '
          'over seeds in parentheses. The ratio is the mean over seeds of held-out loss divided by training loss, with '
          'its range over seeds in parentheses. The diffusion and the flow-matching objectives are different '
          'quantities, so losses are comparable within a row only; the ratio is the quantity compared in the text. '
          'The analysis was exploratory.',
          '',
          '| Model | Objective | Split | Training loss | Held-out loss | Held-out ÷ training |',
          '|---|---|---|---|---|---|']
    long_rows = []
    for label, objective, key, arm in MODELS:
        for split in SPLITS:
            s = summ[(key, arm, split)]
            md.append('| %s | %s | %s | %s | %s | %s |' % (
                label, objective, SPLIT_HEAD[split],
                loss_cell(s['training_loss_mean'], s['training_loss_sd']),
                loss_cell(s['heldout_loss_mean'], s['heldout_loss_sd']),
                ratio_cell(s['ratio_mean'], s['ratio_min'], s['ratio_max'])))
            for seed, train, test, ratio, name in runs[(key, arm, split)]:
                for quantity, value in (('training_loss', train), ('heldout_loss', test), ('ratio', ratio)):
                    long_rows.append(dict(row=label, objective=objective, split=split, seed=seed, quantity=quantity,
                                          value=repr(value), source_record=short[key],
                                          source_key='per_checkpoint/%s' % name))
            for quantity in ('training_loss_mean', 'training_loss_sd', 'heldout_loss_mean', 'heldout_loss_sd',
                             'ratio_mean', 'ratio_min', 'ratio_max'):
                long_rows.append(dict(row=label, objective=objective, split=split, seed='', quantity=quantity,
                                      value=repr(s[quantity]), source_record=short[key],
                                      source_key='computed from per_checkpoint/%s_%s_seed0..4' % (split, arm)))
    md += ['',
           'Sources: %s (sha256 %s; compute_test_loss_base.py) and %s (sha256 %s; compute_test_loss.py). Before the '
           'original model was scored, the procedure reproduced the recorded value of %s in %s to within 1e-9. '
           'Assembled by make_supp_table_s2.py.'
           % (short['base'], SHA256['base'][:16], short['v2'], SHA256['v2'][:16], gate['key'], short['v2'])]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(long_rows)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('')
    print('[checks] 1 (%d runs: seeds 0-4, ratio = held-out / training), 2 (%d stored summary values of the original '
          'model), 3 (reproduction gate, largest difference %.2e), 4 (both records exploratory), 5 (values quoted in '
          'Results 1 and the two drafted rows) all passed' % (len(MODELS) * len(SPLITS) * len(SEEDS), n_check2, gate_diff))
    print('[written] %s  sha256 %s' % (OUT_CSV, sha256_of(OUT_CSV)))
    print('[written] %s  sha256 %s' % (OUT_MD, sha256_of(OUT_MD)))


if __name__ == '__main__':
    main()
