#!/usr/bin/env python3
"""
gen28.py — the synthetic test of PREREG_step28_salt_twins.md section 8.2, for score_step28_salt_twins.py.

Purpose
-------
Prove, before the real run, that the Step 28 scorer recovers the pre-registered verdicts when the answer is known,
and that it stops rather than reporting a number when a guard is violated. Nothing here touches the real records:
every file is generated in a scratch directory and the scorer is pointed at it with S28_TEST_DIR.

Logic
-----
Three scenarios are generated on a mock study of 36 structures per split, 4 conditions each (144 conditions), with
9 structures marked as having a parent twin in training (the same 1-in-4 proportion as the real splits, rounded up
so the planted effect is measurable):

  (a) null         - the excluded structures are drawn from the same distribution as the retained ones.
                     Expected: verdict K1 NO CHANGE, verdict L near zero.
  (b) planted high - the excluded structures are shifted far enough that removing them moves a mean past its
                     interval and flips a contrast's sign.
                     Expected: verdict K3 A CONCLUSION CHANGES, verdict L clearly positive.
  (c) planted low  - the excluded structures are shifted the other way, by the same size.
                     Expected: verdict K3 as well (the manuscript's claims can be damaged in either direction: a
                     conclusion that changes because the twins were *dragging the average down* is still a
                     conclusion that changes), and verdict L clearly negative.

  Note on (c): the pre-registration's section 8.2 expected outcome 1 for (c). Running it is what shows the
  expectation was wrong — a large shift in either direction moves the retained mean. This is recorded in the test
  output and in step28_findings.md rather than quietly adjusted, and it does not change the scorer: K3 for (c) is
  the correct behaviour, because the verdict asks whether a reported value changes, not whether it improves.

Five negative tests run in the sandbox (the sixth of section 8.2, "uncommitted script", cannot run there because
the sandbox deliberately relaxes the git guards; it is exercised by the real run, which refuses an unclean tree):

  1. an output file already exists
  2. the exclusion list in the scorer differs from the record
  3. a published per-line value is tampered with (gate 1 must fail)
  4. an excluded structure id is absent from the data record
  5. the per-condition records are out of condition order

Each must stop the step with a non-zero exit.

Run from the repository worktree, squidiff environment:
    python gen28.py
It prints one line per scenario and per negative test, and exits non-zero if any expectation is unmet.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

SPLITS = ['drug_split_1', 'drug_split_0', 'drug_split_2']
N_STRUCT = 36
N_PER_STRUCT = 4
N_EXCL = 9                                   # structures marked as having a parent twin in training
LINES = ['L1', 'L2', 'L3']
DOSES = [50.0, 500.0, 5000.0]
PLATES = ['P1', 'P2', 'P3']
ARMS_A = ['base', 'recipe', 'ridge_train', 'ridge_S7', 'mean_response', 'control_paste']
SUFFIXES = ['de_overlap', 'energy', 'dpc50']
SCORER = 'score_step28_salt_twins.py'


def build(tmp, shift):
    """Write a complete synthetic record set into tmp. `shift` is added to every value of the excluded structures.

    The arms are given different levels so that contrasts have a definite sign in the null case: recipe above base
    above the references. Values are drawn once with a fixed seed so the three scenarios differ only by `shift`.
    """
    rng = np.random.RandomState(28)
    sids = ['S%04d' % (i + 1) for i in range(N_STRUCT)]
    excluded = sids[:N_EXCL]

    rows_b, rows_c, rows_25, rows_26, rows_27, rows_long = [], [], [], [], [], []
    for sp in SPLITS:
        cond_i = 0
        per_line_vals = {}
        for si, sid in enumerate(sids):
            for k in range(N_PER_STRUCT):
                cond = '%s_c%04d' % (sp, cond_i)
                line = LINES[cond_i % len(LINES)]
                dose = DOSES[cond_i % len(DOSES)]
                plate = PLATES[cond_i % len(PLATES)]
                bump = shift if sid in excluded else 0.0
                r_b = dict(split=sp, condition=cond, line=line, plate=plate, structure=sid, dose_nM=dose,
                           n_cells=100, learnability=0.05)
                base_level = {'base': 0.30, 'recipe': 0.46, 'ridge_train': 0.40, 'ridge_S7': 0.44,
                              'mean_response': 0.50, 'control_paste': 0.35}
                for arm in ARMS_A:
                    for suf in SUFFIXES:
                        v = base_level[arm] + rng.normal(0, 0.05) + bump
                        r_b['%s_%s' % (arm, suf)] = v
                        per_line_vals.setdefault((sp, line, arm, suf), []).append(v)
                rows_b.append(r_b)

                r_c = dict(split=sp, condition=cond, line=line, plate=plate, structure=sid, dose_nM=dose,
                           n_candidates=30, learnability=0.05)
                for arm in ARMS_A:
                    r_c['%s_pds' % arm] = 0.5 + (0.22 if arm == 'recipe' else 0.10) + rng.normal(0, 0.03) + bump
                rows_c.append(r_c)

                rows_25.append(dict(split=sp, condition=cond, line=line, structure=sid, dose_nM=dose,
                                    recipe_own_pds=0.72 + rng.normal(0, 0.03) + bump,
                                    recipe_shuf_pds=0.71 + rng.normal(0, 0.03) + bump))
                rows_26.append(dict(split=sp, condition=cond, line=line, plate=plate, structure=sid, dose_nM=dose,
                                    recipe__orig__pds=0.72 + rng.normal(0, 0.03) + bump,
                                    ctrl_line__orig__pds=0.74 + rng.normal(0, 0.03)))
                rows_27.append(dict(split=sp, condition=cond, line=line, plate=plate, structure=sid, dose_nM=dose,
                                    recipe__split__wp_pds=0.50 + rng.normal(0, 0.03) + bump,
                                    residual__true__split=0.34 + rng.normal(0, 0.05)))
                cond_i += 1

        for (s_, ln, arm, suf), vals in per_line_vals.items():
            metric = {'de_overlap': 'de_overlap', 'energy': 'energy', 'dpc50': 'dpc_mean'}[suf]
            rows_long.append(dict(split=s_, line=ln, submission=arm, metric=metric,
                                  value=float(np.nanmean(vals))))

    pd.DataFrame(rows_b).to_csv(os.path.join(tmp, 'step15_model_per_condition.csv'), index=False)
    pd.DataFrame(rows_c).to_csv(os.path.join(tmp, 'step15c_pds_per_condition.csv'), index=False)
    pd.DataFrame(rows_25).to_csv(os.path.join(tmp, 'step25_swap_per_condition.csv'), index=False)
    pd.DataFrame(rows_26).to_csv(os.path.join(tmp, 'step26_source_per_condition.csv'), index=False)
    pd.DataFrame(rows_27).to_csv(os.path.join(tmp, 'step27_within_plate_per_condition.csv'), index=False)
    pd.DataFrame(rows_long).to_csv(os.path.join(tmp, 'step15_model_long.csv'), index=False)

    # the data record: structure id -> canonical SMILES, through a compound name
    drugs = {'structure': {('D%03d' % i): s for i, s in enumerate(sids)},
             'canonical_smiles': {('D%03d' % i): ('C' * (i + 1)) for i in range(len(sids))}}
    json.dump(dict(drugs=drugs), open(os.path.join(tmp, 'step15_data_record.json'), 'w'))

    # the overlap record, in the shape check_split_parent_overlap.py writes
    ov = {'tahoe': {sp: {'held_out_structures': N_STRUCT,
                         'held_out_with_parent_in_training': [{'structure_id': s} for s in excluded]}
                    for sp in SPLITS},
          'sciplex': {}}
    json.dump(ov, open(os.path.join(tmp, 'split_parent_overlap.json'), 'w'))
    return excluded


def patch_scorer(tmp, excluded):
    """Copy the scorer into tmp with its EXCLUDE constants replaced by the synthetic ones.

    The scorer compares its literal lists against the record, so the test must set both; the point of the negative
    test 2 is that a mismatch between them stops the step.
    """
    src = open(SCORER).read()
    start = src.index('EXCLUDE = {')
    end = src.index('}', src.index('drug_split_2', start)) + 1
    new = 'EXCLUDE = {\n' + ''.join("    '%s': %r,\n" % (sp, excluded) for sp in SPLITS) + '}'
    src = src[:start] + new + src[end:]
    src = src.replace("EXPECT_N = {'drug_split_1': 8, 'drug_split_0': 4, 'drug_split_2': 9}",
                      "EXPECT_N = {%s}" % ', '.join("'%s': %d" % (sp, len(excluded)) for sp in SPLITS))
    path = os.path.join(tmp, SCORER)
    open(path, 'w').write(src)
    return path


def run(tmp, path):
    """Run the patched scorer against the sandbox; return (exit code, combined output)."""
    env = dict(os.environ, S28_TEST_DIR=tmp, PYTHONPATH=os.getcwd())
    p = subprocess.run([sys.executable, path], env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def scenario(name, shift, expect_verdict, expect_L_sign):
    tmp = tempfile.mkdtemp(prefix='gen28_%s_' % name)
    try:
        excluded = build(tmp, shift)
        path = patch_scorer(tmp, excluded)
        rc, out = run(tmp, path)
        if rc != 0:
            if rc < 0 or 'KeyboardInterrupt' in out:
                print('[%s] FAIL: the scorer was killed by signal %d before it finished (usually the login node '
                      'reaping a slow scipy/anndata import). Run this test on a compute node: sbatch run_gen28.sh'
                      % (name, -rc))
            else:
                print('[%s] FAIL: scorer exited %d\n%s' % (name, rc, out[-1500:]))
            return False
        j = json.load(open(os.path.join(tmp, 'step28_salt_twins.json')))
        got = j['verdict_K']['verdict']
        if not j.get('synthetic_test'):
            print('[%s] FAIL: output is not marked as a synthetic test' % name)
            return False
        head = j['verdict_L']['headline']
        mean_L = float(np.mean([x['excluded_minus_retained'] for x in head])) if head else 0.0
        ok_v = got.startswith(expect_verdict)
        ok_L = (abs(mean_L) < 0.02 if expect_L_sign == 0
                else (mean_L > 0.05 if expect_L_sign > 0 else mean_L < -0.05))
        print('[%s] verdict %s (expected %s*) | verdict L mean %+.4f (expected %s) | %s'
              % (name, got, expect_verdict, mean_L,
                 {0: 'near zero', 1: 'positive', -1: 'negative'}[expect_L_sign],
                 'OK' if (ok_v and ok_L) else 'FAIL'))
        return ok_v and ok_L
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def negative(name, mutate, expect):
    """A guard test: mutate the sandbox, then require THIS guard's own message.

    `expect` is a substring of the FATAL line the mutated run must print. Requiring only a FATAL is not enough:
    the synthetic run of 2026-11-09 showed a test passing on an unrelated guard that fired earlier, so the test
    never reached the thing it was meant to check.
    """
    tmp = tempfile.mkdtemp(prefix='gen28_neg_')
    try:
        excluded = build(tmp, 0.0)
        path = patch_scorer(tmp, excluded)
        mutate(tmp, path, excluded)
        rc, out = run(tmp, path)
        # A non-zero exit is NOT enough: a crash (a killed import, a missing package) also exits non-zero and would
        # report a guard as working when the guard never ran. The test requires the guard's own message, which
        # score_step23_screen.die() prints as a line beginning FATAL, and rejects a run killed by a signal.
        fatal = next((l for l in out.splitlines() if 'FATAL' in l), '')
        killed = rc < 0 or 'KeyboardInterrupt' in out
        right = expect in fatal
        ok = bool(fatal) and not killed and right
        why = ('OK' if ok else
               'FAIL (killed by signal %d, the guard never ran)' % -rc if killed else
               'FAIL (stopped without a FATAL message)' if not fatal else
               'FAIL (a different guard fired; expected %r)' % expect)
        print('[negative: %s] exit %d | %s | %s' % (name, rc, (fatal or out.strip().splitlines()[-1:] or [''])[0][:110]
                                                    if not fatal else fatal[:110], why))
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if not os.path.isfile(SCORER):
        sys.exit('FATAL: run this from the worktree that holds %s' % SCORER)
    results = []
    print('--- scenarios (prereg 8.2) ---')
    results.append(scenario('a_null', 0.00, 'K1', 0))
    results.append(scenario('b_planted_high', 0.25, 'K3', 1))
    results.append(scenario('c_planted_low', -0.25, 'K3', -1))

    print('--- negative tests ---')

    def m_exists(tmp, path, excl):
        open(os.path.join(tmp, 'step28_salt_twins.json'), 'w').write('{}')

    def m_list(tmp, path, excl):
        ov = json.load(open(os.path.join(tmp, 'split_parent_overlap.json')))
        ov['tahoe'][SPLITS[0]]['held_out_with_parent_in_training'] = [{'structure_id': 'S0099'}]
        json.dump(ov, open(os.path.join(tmp, 'split_parent_overlap.json'), 'w'))

    def m_published(tmp, path, excl):
        f = os.path.join(tmp, 'step15_model_long.csv')
        d = pd.read_csv(f)
        d.loc[0, 'value'] = float(d.loc[0, 'value']) + 0.01
        d.to_csv(f, index=False)

    def m_missing_id(tmp, path, excl):
        rec = json.load(open(os.path.join(tmp, 'step15_data_record.json')))
        name = next(n for n, s in rec['drugs']['structure'].items() if s == excl[0])
        rec['drugs']['structure'][name] = 'S9999'
        json.dump(rec, open(os.path.join(tmp, 'step15_data_record.json'), 'w'))
        d = pd.read_csv(os.path.join(tmp, 'step15_model_per_condition.csv'))
        d['structure'] = d['structure'].replace(excl[0], 'S9999')
        d.to_csv(os.path.join(tmp, 'step15_model_per_condition.csv'), index=False)

    def m_order(tmp, path, excl):
        f = os.path.join(tmp, 'step26_source_per_condition.csv')
        d = pd.read_csv(f)
        d = pd.concat([d.iloc[1:], d.iloc[:1]]).reset_index(drop=True)
        d.to_csv(f, index=False)

    results.append(negative('output already exists', m_exists, 'refusing to overwrite'))
    results.append(negative('exclusion list differs from the record', m_list, 'exclusion list differs'))
    results.append(negative('tampered published per-line value (gate 1)', m_published, 'gate 1 FAILED'))
    results.append(negative('excluded structure absent from the data record', m_missing_id,
                            'match the `structure` column'))
    results.append(negative('per-condition records out of order', m_order,
                            'is not in the Step 15b condition order'))

    n_ok = sum(bool(r) for r in results)
    print('\n%d of %d checks passed' % (n_ok, len(results)))
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == '__main__':
    main()
