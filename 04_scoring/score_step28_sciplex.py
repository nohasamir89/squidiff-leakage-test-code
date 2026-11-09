#!/usr/bin/env python3
"""
score_step28_sciplex.py - Step 28, group B: sensitivity of the sci-Plex gene-level results to the held-out compound
whose salt twin was in training (PREREG_step28_salt_twins.md section 4 group B; amendments 15 and 16).

Purpose
-------
sci-Plex gene identity is the one positive, compound-dependent result of the paper: the rewired model is above every
linear reference. On sci-Plex one compound - tofacitinib, supplied as free base and as citrate - has its parent
molecule on both sides of splits 0 and 2 (none on split 1). This script re-aggregates the per-condition record of
score_step14.py with that compound's conditions removed, and reports (verdict K) whether any stated ordering changes
and (verdict L) whether the removed conditions scored better than the rest, which is what leakage would predict.

No model is run and no cell is re-sampled. Every number comes from a committed record.

Logic
-----
1. Guards: the pre-registration (with amendment 16), this script, its runner, the scorers it relies on and every
   record it reads are committed and clean; no output exists.

2. Exclusion (section 3; amendment 16b). The excluded compound of each split is read from the committed overlap
   record leakage_results/split_parent_overlap.json ('sciplex' section), which must list none on split 1 and exactly
   one on each of splits 0 and 2, with test_conditions_affected = 12 per listed compound. Each is matched to the
   per-condition `structure` column (the test file's SMILES) exactly, or failing that by RDKit canonical SMILES; the
   match must select exactly one structure value and exactly 12 conditions (one compound x 3 lines x 4 doses), or
   the step stops. The parent twin itself sits in the training set, so it is not in this test record and cannot be
   matched by mistake.

3. Gate 1 (amendments 15, 16). For every split and submission, the nanmean (and, for DPC-50, the nanmedian) of the
   per-condition DE-overlap, energy and DPC-50 over all conditions must reproduce the committed thesis metrics in
   step14_analysis.json to 1e-9; and every committed contrast point (a baseline minus the comparator) must reproduce
   to 1e-9 from the per-condition values. This proves the per-condition record and this script's arithmetic are the
   ones the published sci-Plex numbers came from, before any compound is removed.

4. Intervals. 97.5 % structure-clustered percentile bootstrap (score_step23_screen.Boot, fresh RandomState(0) per
   set) for the full set, the retained set and the excluded set - one construction for all, so the comparison of
   published and twin-free is internally consistent. The committed contrast intervals were drawn by score_step14's
   own bootstrap, which cannot be called without re-executing that whole script; they are compared with the
   recomputed full-set intervals descriptively (written to the output as interval_replay), not used as a gate.
   The summarise / paired arithmetic is imported from score_step28_salt_twins.py, the code the synthetic test gen28
   exercised (8 of 8).

5. Verdicts K and L as pre-registered (section 5), over the sci-Plex rows: K flags any contrast whose interval
   crosses zero or whose sign reverses, and any value that moves by more than half its interval width; L is the
   excluded-minus-retained difference, headline for the rewired model's gene identity on splits 0 and 2.

Reads (all committed):
  PREREG_step28_salt_twins.md
  leakage_results/step14_per_condition.csv    per-condition DE-overlap, energy, DPC-50 per submission (amendment 15)
  leakage_results/step14_analysis.json        committed thesis metrics and contrasts; the comparator name
  leakage_results/split_parent_overlap.json   the exclusion record

Writes (once, never overwritten):
  leakage_results/step28_sciplex.json
  leakage_results/step28_sciplex_per_quantity.csv

Run from the repository worktree, squidiff environment, no options (via run_step28_sciplex.sh):
    python score_step28_sciplex.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import score_step23_screen as s23
import score_step28_salt_twins as s28          # summarise, paired: the arithmetic gen28 exercised (8 of 8)

PRE = 'PREREG_step28_salt_twins.md'
MIN_SECTIONS = 16                              # through amendment 16
RES = 'leakage_results'
PC14 = os.path.join(RES, 'step14_per_condition.csv')
J14 = os.path.join(RES, 'step14_analysis.json')
OVERLAP = os.path.join(RES, 'split_parent_overlap.json')
OUT_JSON = os.path.join(RES, 'step28_sciplex.json')
OUT_PQ = os.path.join(RES, 'step28_sciplex_per_quantity.csv')
SPLITS = ['drug_split_1', 'drug_split_0', 'drug_split_2']           # development first, as everywhere in the paper
EXPECT_COMPOUNDS = {'drug_split_1': 0, 'drug_split_0': 1, 'drug_split_2': 1}   # prereg section 1 table
CONDS_PER_COMPOUND = 12                                             # 3 lines x 4 doses (amendment 15)
REPRO_TOL = 1e-9
# the registered group-B quantities: (label, per-condition column)
QUANT = [('gene identity', 'de_overlap'), ('response correlation', 'dpc50')]
# the linear and model-free references the manuscript compares the rewired model with
REFS = ['base', 'ridge_train', 'ridge_S7', 'mean_response', 'control_paste']
T0 = time.time()
say, die = s23.say, s23.die


def resolve(record_smiles, structure_values, split):
    """Find the per-condition structure value that is the record's compound: exact match first, then RDKit
    canonical SMILES. Exactly one structure value must match; anything else stops the step."""
    present = sorted(set(map(str, structure_values)))
    if record_smiles in present:
        return record_smiles, 'exact SMILES'
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else None
    target = canon(record_smiles)
    hits = [p for p in present if target is not None and canon(p) == target]
    if len(hits) == 1:
        return hits[0], 'RDKit canonical SMILES'
    die('%s: the record compound %r matches %d structure values in the per-condition record (exact or canonical); '
        'expected exactly 1' % (split, record_smiles[:60], len(hits)))


def verdicts(rows):
    """Verdicts K and L of prereg section 5 over the sci-Plex rows (mirrors score_step28_salt_twins.compute_verdicts,
    with group 'B' in place of group 'A'). Contrasts are read against zero."""
    crossed = []
    for r in rows:
        if r['group'] != 'B-contrast' or r['twin_free'] is None or r['published'] is None:
            continue
        was = r['published_lo'] * r['published_hi'] > 0
        now = r['twin_free_lo'] * r['twin_free_hi'] > 0
        if was != now:
            crossed.append(dict(split=r['split'], quantity=r['quantity'], arm=r['arm'],
                                published=[r['published_lo'], r['published_hi']],
                                twin_free=[r['twin_free_lo'], r['twin_free_hi']],
                                direction='now includes zero' if was else 'now excludes zero'))
    reversed_sign = [r for r in rows if r['group'] == 'B-contrast' and r['twin_free'] is not None
                     and r['published'] is not None and np.sign(r['published']) != np.sign(r['twin_free'])]
    moved = [r for r in rows if r['twin_free'] is not None and r['published'] is not None
             and r['published_hi'] is not None
             and abs(r['twin_free'] - r['published']) > 0.5 * (r['published_hi'] - r['published_lo'])]
    if crossed or reversed_sign:
        verdict = 'K3 A CONCLUSION CHANGES'
    elif moved:
        verdict = 'K2 CHANGE CONFINED TO MAGNITUDES'
    else:
        verdict = 'K1 NO CHANGE'
    keys = ('split', 'quantity', 'arm', 'published', 'twin_free')
    k = dict(verdict=verdict, crossed=crossed,
             sign_reversals=[{x: r[x] for x in keys} for r in reversed_sign],
             moved=[{x: r[x] for x in keys} for r in moved])
    say('[verdict K] %s - %d interval crossings, %d contrast sign reversals, %d values moved by more than half the '
        'published interval' % (verdict, len(crossed), len(reversed_sign), len(moved)))
    L = [dict(split=r['split'], group=r['group'], quantity=r['quantity'], arm=r['arm'],
              excluded_minus_retained=float(r['excluded_mean'] - r['twin_free']), n_excluded=r['n_excluded'])
         for r in rows if r['excluded_mean'] is not None and r['twin_free'] is not None]
    return k, L


def main():
    if len(sys.argv) > 1:
        die('this script takes no options')
    for q in (OUT_JSON, OUT_PQ):
        if os.path.exists(q):
            die('%s exists; refusing to overwrite a record' % q)
    if not os.path.isfile(PRE) or open(PRE).read().count('\n## ') < MIN_SECTIONS:
        die('pre-registration missing or without amendment 16 (needs %d sections)' % MIN_SECTIONS)

    s23.committed_and_clean([PRE, 'score_step28_sciplex.py', 'run_step28_sciplex.sh', 'score_step28_salt_twins.py',
                             'score_step14.py', 'score_step23_screen.py'], 'pre-registration or scripts')
    s23.committed_and_clean([PC14, J14, OVERLAP, 'step28_findings.md'], 'records')
    commit = s23.git('rev-parse', 'HEAD').stdout.strip()
    say('[guards] pre-registration, scripts and records committed and clean; commit %s' % commit[:7])

    pc = pd.read_csv(PC14)
    j14 = json.load(open(J14))
    ov = json.load(open(OVERLAP))
    comparator = j14.get('comparator')
    if not comparator:
        die('step14_analysis.json has no comparator field')
    arms = [comparator] + REFS
    say('[comparator] the rewired model on sci-Plex is Step 14\'s pre-registered comparator: %s' % comparator)

    rows, summary, replay = [], {}, []
    for sp in SPLITS:
        d = pc[pc['split'] == sp]
        subs = list(dict.fromkeys(d['submission']))                  # file order
        for a in arms:
            if a not in subs:
                die('%s: submission %s missing from %s' % (sp, a, PC14))
        # integrity: every submission covers the same conditions in the same order
        ref_cond = d.loc[d['submission'] == subs[0], 'condition'].astype(str).values
        for s_ in subs[1:]:
            if not np.array_equal(d.loc[d['submission'] == s_, 'condition'].astype(str).values, ref_cond):
                die('%s: submission %s does not cover the conditions in the same order' % (sp, s_))
        first = d[d['submission'] == subs[0]].reset_index(drop=True)
        n = len(first)
        clusters = first['structure'].astype(str).values
        all_idx = np.arange(n)

        def col(s_, k):
            return pd.to_numeric(d.loc[d['submission'] == s_, k], errors='coerce').values.astype(np.float64)

        say('[%s] %d conditions, %d structures; full-set gene identity: %s'
            % (sp, n, len(set(clusters)),
               '  '.join('%s %.4f' % (s_, np.nanmean(col(s_, 'de_overlap'))) for s_ in subs)))

        # ---- gate 1: the committed thesis metrics and contrast points, from the per-condition record
        tm = j14['per_split'][sp]['thesis_metrics']
        n_checked = 0
        for s_ in subs:
            if s_ not in tm:
                die('%s: submission %s is not in the committed thesis metrics' % (sp, s_))
            for key, pcol, stat in (('de_overlap', 'de_overlap', np.nanmean), ('energy', 'energy', np.nanmean),
                                    ('dpc_mean', 'dpc50', np.nanmean), ('dpc_median', 'dpc50', np.nanmedian)):
                want = tm[s_].get(key)
                if want is None or not np.isfinite(want):
                    continue
                got = float(stat(col(s_, pcol)))
                if abs(got - want) > REPRO_TOL:
                    die('gate 1 FAILED: %s %s %s: %.12f from the per-condition record, %.12f committed'
                        % (sp, s_, key, got, want))
                n_checked += 1
        cons = j14['per_split'][sp]['contrasts']
        cmap = {'primary_de_ridgeS7_minus_fm50': ('ridge_S7', 'de_overlap'),
                'coprimary_energy_ridgeS7_minus_fm50': ('ridge_S7', 'energy')}
        for b_ in ('ridge_train', 'mean_response', 'control_paste'):
            cmap['secondary_jac_%s_minus_fm50' % b_] = (b_, 'de_overlap')
            cmap['secondary_en_%s_minus_fm50' % b_] = (b_, 'energy')
        for ckey, (b_, pcol) in cmap.items():
            c = cons.get(ckey)
            if c is None or c.get('diff') is None:
                continue
            full = s28.paired(col(b_, pcol), col(comparator, pcol), all_idx, clusters, n)
            if abs(full['mean'] - float(c['diff'])) > REPRO_TOL:
                die('gate 1 FAILED: %s contrast %s: %.12f recomputed, %.12f committed'
                    % (sp, ckey, full['mean'], float(c['diff'])))
            n_checked += 1
            replay.append(dict(split=sp, contrast=ckey, committed=[float(x) for x in c['ci']],
                               recomputed=[full['lo'], full['hi']],
                               max_abs_diff=max(abs(full['lo'] - float(c['ci'][0])),
                                                abs(full['hi'] - float(c['ci'][1])))))
        say('[gate 1] %s: %d committed values reproduced to %g (thesis metrics of %d submissions and the committed '
            'contrast points)' % (sp, n_checked, REPRO_TOL, len(subs)))

        # ---- exclusion
        block = ov['sciplex'][sp]
        entries = block['held_out_with_parent_in_training']
        if len(entries) != EXPECT_COMPOUNDS[sp]:
            die('%s: the overlap record lists %d compounds, the pre-registration says %d'
                % (sp, len(entries), EXPECT_COMPOUNDS[sp]))
        aff = block.get('test_conditions_affected')
        if aff is not None and int(aff) != CONDS_PER_COMPOUND * len(entries):
            die('%s: the record says %s affected conditions, expected %d' % (sp, aff, CONDS_PER_COMPOUND * len(entries)))
        excluded, matched = np.array([], dtype=int), []
        for x in entries:
            if 'smiles' not in x:
                die('%s: overlap entry has no smiles key; keys are %r' % (sp, sorted(x)))
            val, how = resolve(str(x['smiles']), clusters, sp)
            idx = np.flatnonzero(clusters == val)
            if len(idx) != CONDS_PER_COMPOUND:
                die('%s: the matched compound has %d conditions, expected %d' % (sp, len(idx), CONDS_PER_COMPOUND))
            excluded = np.union1d(excluded, idx)
            matched.append(dict(record_smiles=str(x['smiles']), matched_structure=val, matched_by=how,
                                n_conditions=int(len(idx))))
        retained = np.setdiff1d(all_idx, excluded)
        say('[%s] excluding %d compound(s) = %d conditions (%.1f %%), %d retained%s'
            % (sp, len(entries), len(excluded), 100.0 * len(excluded) / n, len(retained),
               (' - matched by ' + ', '.join(m['matched_by'] for m in matched)) if matched else
               ' - none listed: this split is free of the route by construction'))

        def add(group, quantity, arm, va, vb=None):
            """One row of the per-quantity table: the published (all conditions), twin-free (retained) and
            excluded-only values of a quantity va, or of the paired per-condition contrast va - vb."""
            def stat(idx):
                return s28.summarise(va, idx, clusters, n) if vb is None else s28.paired(va, vb, idx, clusters, n)
            full, kept, drop = stat(all_idx), stat(retained), stat(excluded)
            rows.append(dict(split=sp, group=group, quantity=quantity, arm=arm,
                             published=full['mean'], published_lo=full['lo'], published_hi=full['hi'],
                             twin_free=kept['mean'], twin_free_lo=kept['lo'], twin_free_hi=kept['hi'],
                             excluded_mean=drop['mean'], excluded_lo=drop['lo'], excluded_hi=drop['hi'],
                             n_published=full['n'], n_twin_free=kept['n'], n_excluded=drop['n'],
                             structures_twin_free=kept['n_structures']))

        for label, pcol in QUANT:
            for a in arms:
                add('B', label, a, col(a, pcol))
            for r_ in REFS:
                add('B-contrast', label, 'rewired minus %s' % r_, col(comparator, pcol), col(r_, pcol))

        summary[sp] = dict(n_conditions=int(n), n_structures=int(len(set(clusters))), excluded=matched,
                           excluded_conditions=int(len(excluded)), retained_conditions=int(len(retained)),
                           excluded_fraction=float(len(excluded)) / n)

    verdict_k, L = verdicts(rows)
    headline = [x for x in L if x['arm'] == comparator and x['quantity'] == 'gene identity']
    for x in headline:
        say('[verdict L] %s gene identity %s: excluded minus retained %+.4f (%d excluded conditions)'
            % (x['split'], comparator, x['excluded_minus_retained'], x['n_excluded']))
    mx = max([r['max_abs_diff'] for r in replay], default=float('nan'))
    say('[interval replay] committed vs recomputed full-set contrast intervals: largest difference %.3g over %d '
        'contrasts (descriptive; amendment 16b)' % (mx, len(replay)))

    with open(OUT_PQ, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(prereg=PRE, commit=commit, comparator=comparator, splits=summary, verdict_K=verdict_k,
                   verdict_L=dict(headline=headline, all=L), interval_replay=replay,
                   interval='97.5% percentile per split, structure-clustered, fresh RandomState(0) per set, for the '
                            'full, retained and excluded sets alike',
                   gates='gate 1: committed thesis metrics and contrast points reproduced to %g' % REPRO_TOL,
                   seconds=round(time.time() - T0, 1)), open(OUT_JSON, 'w'), indent=2)
    say('[written] %s, %s  (%.1f s)' % (OUT_JSON, OUT_PQ, time.time() - T0))


if __name__ == '__main__':
    main()
