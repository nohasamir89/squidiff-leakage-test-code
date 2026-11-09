#!/usr/bin/env python3
"""
score_step28_salt_twins.py — Step 28: are the reported Tahoe-100M results carried by held-out compounds whose salt,
hydrate or solvate twin was in training? (PREREG_step28_salt_twins.md)

Purpose
-------
Re-aggregate the committed per-condition records with the affected conditions removed, and report, for every
quantity the manuscript states, the published value, the twin-free value, their difference with a structure-
clustered interval, whether any pre-registered conclusion changes (verdict K), and whether the excluded compounds
scored better than the retained ones (verdict L).

No model is run, no cell is re-sampled, no .h5ad file is read. Every number comes from a committed CSV.

Logic
-----
1. Guards (prereg 7.1): the pre-registration, this script and its runner are committed and unchanged; the Step 27
   records and findings are committed; no Step 28 output exists.

2. Exclusion list (prereg 3): read from the committed leakage_results/split_parent_overlap.json, compared against
   the literal constants below (prereg 7.3), and resolved to the values the per-condition records actually carry in
   their `structure` column through the Step 15 data record. The record lists structure ids (S0012); the CSVs may
   carry the id or the canonical SMILES, so both are tried and the one that matches all of them is used.

3. Two reproduction gates, both run before any twin-free number is read (prereg 7.2). They replace reading a
   published summary value out of a JSON, because they are addressable without assuming that file's shape and they
   test precisely the two things that could silently differ:

     Gate 1 (aggregation). For every (split, line, submission, metric) row of tahoe_results/step15_model_long.csv,
     the nanmean of the matching per-condition column over that line's conditions must equal the recorded value to
     1e-9. score_step15_tahoe builds its per-line entries as np.nanmean over that line's conditions (its summ(),
     applied to [ixl]), so agreement proves the published numbers are a mean over conditions and that this script
     reads the right column for each (submission, metric) pair.

     Gate 2 (interval equivalence). For a fixed set of columns, the interval from
     s23.Boot(all_conditions).mean_reps + the published percentiles must equal score_step15_tahoe.boot(x, drugs,
     n_drugs) to 5e-6. Both draw rng.randint(0, G, G) per replicate from a fresh RandomState(0), so they should be
     identical; the gate proves it on this data before Boot is used on a subset. s15.boot itself cannot be used
     once structures are removed - it aborts when the cluster count differs from the split's held-out structure
     count (its line 119) - which is why s23.Boot exists and why this equivalence must be shown.

4. Re-aggregation: for each split and quantity, the mean over the retained conditions with the 97.5 % structure-
   clustered percentile bootstrap of s23.Boot over the retained indices; the same for the excluded conditions alone
   (verdict L) and for the full set (which must reproduce the published value, gate 1).

5. Verdicts K and L exactly as pre-registered (prereg 5).

Scope
-----
Tahoe-100M, prereg 4 groups A (gene identity, population distance, response correlation), C (cross-plate
discrimination, every arm and scheme of Steps 15c/25/26) and D (within-plate discrimination and the plate residual,
Step 27).

Group E (response-magnitude agreement) is NOT computed here; see the dated amendment in the pre-registration. It is
a rank correlation computed across the conditions of a context, not a per-condition quantity, so removing
conditions changes the ranking itself rather than the average of a per-condition value, and the question this step
asks is not well posed for it.

Group B (sci-Plex gene identity) is a separate script: score_step14.py writes one aggregated CSV keyed by
(split, line, submission, facet) and no per-condition record, so that arm re-runs it with an exclusion list.

Reads (all committed):
  PREREG_step28_salt_twins.md
  leakage_results/split_parent_overlap.json             the exclusion list
  tahoe_results/step15_data_record.json                 structure id <-> canonical SMILES
  tahoe_results/step15_model_per_condition.csv          group A
  tahoe_results/step15_model_long.csv                   gate 1 targets
  tahoe_results/step15c_pds_per_condition.csv           group C, published construction
  tahoe_results/step25_swap_per_condition.csv           group C, fingerprint-swap arms
  tahoe_results/step26_source_per_condition.csv         group C, control-source schemes and no-model arms
  tahoe_results/step27_within_plate_per_condition.csv   group D, within-plate PDS and plate residual

Writes (once, never overwritten):
  tahoe_results/step28_salt_twins.json
  tahoe_results/step28_salt_twins_per_quantity.csv
  tahoe_results/step28_salt_twins_summary.csv

Run from the repository worktree, squidiff environment, no options:
    python score_step28_salt_twins.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import score_step15_tahoe as s15
import score_step23_screen as s23
import score_step25_tahoe as s25

PRE = 'PREREG_step28_salt_twins.md'
MIN_SECTIONS = 8
SPLITS = s25.SPLITS                                   # development first: drug_split_1, drug_split_0, drug_split_2
OVERLAP = 'leakage_results/split_parent_overlap.json'
PC15B = s25.PC15B                                     # step15_model_per_condition.csv
PC15C = s25.PC15C                                     # step15c_pds_per_condition.csv
LONG15B = os.path.join(s15.RES, 'step15_model_long.csv')
S25_PC = os.path.join(s15.RES, 'step25_swap_per_condition.csv')
S26_PC = os.path.join(s15.RES, 'step26_source_per_condition.csv')
S27_PC = os.path.join(s15.RES, 'step27_within_plate_per_condition.csv')
OUT_JSON = os.path.join(s15.RES, 'step28_salt_twins.json')
OUT_PQ = os.path.join(s15.RES, 'step28_salt_twins_per_quantity.csv')
OUT_SUM = os.path.join(s15.RES, 'step28_salt_twins_summary.csv')
REPRO_TOL = 1e-9                                      # prereg 7.2, gate 1
# Gate 2 (prereg 7.2, as amended 2026-11-09 #12). s23.Boot and score_step15_tahoe.boot draw the same resamples
# (rng.randint(0, G, G) per replicate) but from different generators: Boot makes a fresh RandomState(0) for every
# set, whereas s15.boot draws from the module-level s15.rng, which every call advances. So they agree only on the
# first call after import - exactly what the synthetic runs showed (recipe_de_overlap, tested first, matched to
# better than 5e-6 every time; the next columns differed by Monte-Carlo noise of 0.3-3 % of the interval width).
# Step 27 handles the same generator the same way (s15.rng = RandomState(0) before its replay). The gate therefore
# resets s15.rng before each s15.boot call and requires agreement to an absolute 5e-6, the tolerance the
# pre-registration first stated; amendment #11's relative tolerance rested on a wrong diagnosis and is withdrawn.
INTERVAL_TOL = 5e-6
T0 = time.time()
say, die = s23.say, s23.die

# The synthetic test of prereg 8.2 runs this scorer against generated records in a scratch directory that is not a
# git worktree. S28_TEST_DIR redirects every record path there and turns the git guards into warnings; it is never
# set for a real run, and when it is set the outputs record the fact so a test result can never be mistaken for one.
TEST_DIR = os.environ.get('S28_TEST_DIR')
if TEST_DIR:
    def _t(p):
        return os.path.join(TEST_DIR, os.path.basename(p))
    PC15B, PC15C, LONG15B = _t(PC15B), _t(PC15C), _t(LONG15B)
    S25_PC, S26_PC, S27_PC = _t(S25_PC), _t(S26_PC), _t(S27_PC)
    OVERLAP = _t(OVERLAP)
    OUT_JSON, OUT_PQ, OUT_SUM = _t(OUT_JSON), _t(OUT_PQ), _t(OUT_SUM)
    RECORD_PATH = _t(s15.RECORD)
else:
    RECORD_PATH = s15.RECORD

# Prereg 3 and 7.3: the exclusion lists, written out literally so a change to the record is caught.
EXCLUDE = {
    'drug_split_1': ['S0012', 'S0052', 'S0077', 'S0142', 'S0190', 'S0216', 'S0317', 'S0326'],
    'drug_split_0': ['S0058', 'S0141', 'S0204', 'S0287'],
    'drug_split_2': ['S0011', 'S0095', 'S0137', 'S0139', 'S0191', 'S0241', 'S0258', 'S0343', 'S0345'],
}
EXPECT_N = {'drug_split_1': 8, 'drug_split_0': 4, 'drug_split_2': 9}

# Prereg 4 group A. The per-condition column of a (submission, metric) pair is '<arm>_<suffix>'; the long record
# names the same metric by its own key, which gate 1 maps and checks. Only pairs present in both are used.
METRIC_COLUMN = {'de_overlap': '%s_de_overlap',       # gene identity
                 'energy': '%s_energy',               # population distance
                 'dpc_mean': '%s_dpc50',              # response correlation (mean over conditions of DPC-50)
                 'dpc50': '%s_dpc50'}
METRIC_LABEL = {'de_overlap': 'gene identity', 'energy': 'population distance',
                'dpc_mean': 'response correlation', 'dpc50': 'response correlation'}
ARMS_A = ['base', 'recipe', 'ridge_train', 'ridge_S7', 'mean_response', 'control_paste']

# Prereg 4: the contrasts the manuscript states, paired per condition.
CONTRASTS_A = [('rewired minus original', 'recipe', 'base'),
               ('rewired minus training-set ridge', 'recipe', 'ridge_train'),
               ('rewired minus cross-validated ridge', 'recipe', 'ridge_S7'),
               ('rewired minus compound-blind average', 'recipe', 'mean_response'),
               ('rewired minus unaltered vehicle cells', 'recipe', 'control_paste'),
               ('original minus unaltered vehicle cells', 'base', 'control_paste')]

# Columns of the Step 25/26/27 per-condition records that carry a per-condition score.
def is_scored_column(col):
    """A per-condition score column: a PDS of any scheme, a within-plate PDS, or a plate-residual cosine."""
    return col.endswith('_pds') or col.endswith('__wp_pds') or col.startswith('residual__')


def pct(reps):
    """The published two-sided 97.5 % percentile interval of a replicate array (s23.LO = 1.25, s23.HI = 98.75)."""
    return float(np.percentile(reps, s23.LO)), float(np.percentile(reps, s23.HI))


def summarise(values, idx, clusters, n_total):
    """Point mean over the conditions in idx and its structure-clustered interval, the published construction.

    values    per-condition array for the whole split (NaN allowed)
    idx       the condition indices to average over (retained, excluded, or all)
    clusters  per-condition structure labels for the whole split
    n_total   the split's condition count (s23.Boot builds a full-length membership map)

    Non-finite entries are dropped from the set, as the originals do (their stat is np.nanmean over finite
    entries); the number kept is reported so a quantity that is defined on fewer conditions is visible.
    """
    v = np.asarray(values, np.float64)
    keep = np.asarray([i for i in np.asarray(idx) if np.isfinite(v[i])], dtype=int)
    if len(keep) == 0:
        return dict(n=0, n_structures=0, mean=None, lo=None, hi=None)
    b = s23.Boot(keep, clusters, n_total)
    point, reps = b.mean_reps(v)
    lo, hi = pct(reps)
    return dict(n=int(len(keep)), n_structures=int(b.G), mean=float(point), lo=lo, hi=hi)


def paired(values_a, values_b, idx, clusters, n_total):
    """Paired per-condition contrast a - b over the given conditions, the construction of Steps 25-27."""
    d = np.asarray(values_a, np.float64) - np.asarray(values_b, np.float64)
    return summarise(d, idx, clusters, n_total)


def resolve_exclusions(record, col_values, split, ids):
    """Map the record's structure ids to the values the per-condition `structure` column carries.

    The Step 15 data record maps compound name -> structure id (drugs['structure']) and name -> several SMILES
    forms: 'table_smiles' (RDKit-canonicalised from the data table, the form the per-condition records carry) and
    'canonical_smiles' (a different canonicalisation). A structure id can map to more than one name (salt forms of
    one study structure), so each id resolves to the set of SMILES of all its names in a given form.

    The `structure` column is written as drugs[i], which is a structure id on some steps and a SMILES on others, so
    three forms are tried in order - id, table_smiles, canonical_smiles - and the first that covers every excluded
    structure is used. If none does, the step stops rather than silently exclude fewer (prereg 7.3).
    """
    drugs = record['drugs']
    present = set(map(str, col_values))
    names_of_id = {}
    for name, sid in drugs['structure'].items():
        names_of_id.setdefault(sid, []).append(name)

    def smiles_forms(field):
        m = drugs.get(field, {})
        out = {}
        for sid, names in names_of_id.items():
            out[sid] = {str(m[n]) for n in names if n in m}
        return out

    candidates = [('structure id', {sid: {sid} for sid in names_of_id}),
                  ('table_smiles', smiles_forms('table_smiles')),
                  ('canonical_smiles', smiles_forms('canonical_smiles'))]
    report = []
    for how, mapping in candidates:
        matched_vals, n_ok = set(), 0
        for s in ids:
            hit = {v for v in mapping.get(s, set()) if v in present}
            if hit:
                n_ok += 1
                matched_vals |= hit
        report.append('%s %d/%d' % (how, n_ok, len(ids)))
        if n_ok == len(ids):
            return matched_vals, how
    die('%s: no form of the structure identifier matches every excluded structure in the `structure` column '
        '(%s); the column holds values such as %r'
        % (split, ', '.join(report), sorted(present)[:2]))


def gate_aggregation(pc, long_df):
    """Gate 1 (prereg 7.2): every per-line value of the long record is the nanmean of its per-condition column.

    Returns the (submission, metric) pairs that were checked, so the quantities this script reports are exactly
    the ones whose aggregation has been proved.
    """
    checked, skipped, n_all, odd_lines = [], [], 0, set()
    lines_of = {sp: set(pc.loc[pc['split'] == sp, 'line'].astype(str)) for sp in pc['split'].unique()}
    for (split, line, sub, metric), g in long_df.groupby(['split', 'line', 'submission', 'metric'], sort=True):
        tmpl = METRIC_COLUMN.get(metric)
        if tmpl is None:
            skipped.append((sub, metric))
            continue
        col = tmpl % sub
        if col not in pc.columns:
            skipped.append((sub, metric))
            continue
        line = str(line)
        if line == 'ALL':
            # the split-level published value: the manuscript's headline number, aggregated over every condition
            # of the split (score_step15_tahoe's summ() on the full index). Tested like the per-line rows; a
            # failure here would mean the headline is not a mean over conditions, which is worth knowing.
            sel = pc[pc['split'] == split]
            n_all += 1
        elif line in lines_of.get(split, set()):
            sel = pc[(pc['split'] == split) & (pc['line'].astype(str) == line)]
        else:
            # a long-record line label with no conditions in the per-condition file (a tissue aggregate such as
            # 'brain' or 'gut', if the record carries one). It cannot be reproduced from the per-condition rows,
            # so it is listed and skipped rather than failing the gate; reporting it keeps the omission visible.
            odd_lines.add((split, line))
            continue
        got = float(np.nanmean(pd.to_numeric(sel[col], errors='coerce').values))
        want = float(g['value'].iloc[0])
        if not np.isfinite(want):
            continue
        if abs(got - want) > REPRO_TOL:
            die('gate 1 FAILED: %s line=%s %s %s: nanmean of %s is %.12f, the long record says %.12f'
                % (split, line, sub, metric, col, got, want))
        checked.append((sub, metric))
    pairs = sorted(set(checked))
    say('[gate 1] %d values reproduced to %g over %d (submission, metric) pairs (%d of them split-level ALL rows); '
        '%d long-record rows had no per-condition column, %d had a line absent from the per-condition record%s'
        % (len(checked), REPRO_TOL, len(pairs), n_all, len(set(skipped)), len(odd_lines),
           (' (' + ', '.join('%s/%s' % t for t in sorted(odd_lines)) + ')') if odd_lines else ''))
    if not pairs:
        die('gate 1: no (submission, metric) pair could be checked; the column map is wrong')
    return pairs


def gate_interval(pc, split, clusters, n, n_drugs):
    """Gate 2 (prereg 7.2, amendment #12): s23.Boot + percentiles equals score_step15_tahoe.boot on the full set.

    s15.rng is reset to RandomState(0) before each s15.boot call, because that function draws from the module-level
    generator and every call advances it; Boot makes a fresh RandomState(0) per set. Returns the largest absolute
    difference seen, so the caller can record it.
    """
    tested, worst, worst_col = 0, 0.0, None
    for arm in ('recipe', 'base'):
        for suffix in ('_de_overlap', '_energy', '_dpc50'):
            col = arm + suffix
            if col not in pc.columns:
                continue
            v = pd.to_numeric(pc[col], errors='coerce').values.astype(np.float64)
            mine = summarise(v, np.arange(n), clusters, n)
            s15.rng = np.random.RandomState(0)          # s15.boot draws from this module-level generator
            theirs = s15.boot(v, clusters, n_drugs)
            # s15.boot's return shape is read rather than assumed: a (point, lo, hi) sequence, or a mapping with
            # recognisable interval keys. Anything else stops the gate instead of being guessed at.
            if isinstance(theirs, dict):
                keys = {k.lower(): k for k in theirs}
                k_lo = next((keys[k] for k in ('lo', 'low', 'ci_lo', 'l') if k in keys), None)
                k_hi = next((keys[k] for k in ('hi', 'high', 'ci_hi', 'h') if k in keys), None)
                if k_lo is None or k_hi is None:
                    die('gate 2: score_step15_tahoe.boot returned a mapping with keys %r, which this gate does '
                        'not understand' % sorted(theirs))
                lo_t, hi_t = float(theirs[k_lo]), float(theirs[k_hi])
            elif np.ndim(theirs) == 1 and len(theirs) == 2:
                # score_step15_tahoe.boot returns the interval alone, (lo, hi); confirmed by the synthetic test of
                # 2026-11-09, which stopped here rather than assume a (point, lo, hi) shape.
                lo_t, hi_t = float(theirs[0]), float(theirs[1])
            elif np.ndim(theirs) == 1 and len(theirs) >= 3:
                lo_t, hi_t = float(theirs[1]), float(theirs[2])
            else:
                die('gate 2: score_step15_tahoe.boot returned %r, which this gate does not understand' % (theirs,))
            d = max(abs(mine['lo'] - lo_t), abs(mine['hi'] - hi_t))
            if d > worst:
                worst, worst_col = d, col
            if d > INTERVAL_TOL:
                die('gate 2 FAILED: %s %s interval [%.9f, %.9f] from s23.Boot differs from s15.boot (fresh '
                    'RandomState(0)) [%.9f, %.9f] by %.3g, more than %g'
                    % (split, col, mine['lo'], mine['hi'], lo_t, hi_t, d, INTERVAL_TOL))
            tested += 1
    if tested == 0:
        die('gate 2: no column available to test the interval equivalence')
    say('[gate 2] %s: %d intervals from s23.Boot equal score_step15_tahoe.boot to %g; largest difference %.3g (%s)'
        % (split, tested, INTERVAL_TOL, worst, worst_col))
    return worst


def compute_verdicts(rows_pq):
    """Verdicts K and L of prereg 5, from the per-quantity rows. Kept callable so the synthetic test of prereg 8.2
    can plant an effect and check that the verdict fires.

    Verdict K asks whether any reported conclusion changes. Two ways it can:
      - an interval that excluded its reference now includes it, or the reverse. The reference is chance (0.5) for
        a discrimination score, zero for a contrast and for a plate-residual cosine. A bare group-A value (a gene
        identity, say) has no reference of its own, so it is not tested here; the orderings the manuscript states
        between arms are the A-contrast rows, which are tested against zero.
      - a contrast changes sign, which reverses a stated ordering.
    Failing both, a point estimate that moved by more than half the published interval width is reported as a
    change in magnitude only (K2).

    Verdict L asks the opposite question: did the excluded compounds actually score better than the retained ones?
    If the leakage route were operating they should. It is descriptive and does not feed verdict K.
    """
    def reference_of(row):
        if row['group'].endswith('contrast') or row['quantity'] == 'plate residual':
            return 0.0
        return 0.5 if 'discrimination' in row['quantity'] else 0.0

    crossed = []
    for r in rows_pq:
        if r['group'] == 'A' or r['twin_free'] is None or r['published'] is None:
            continue
        ref = reference_of(r)
        was = (r['published_lo'] - ref) * (r['published_hi'] - ref) > 0
        now = (r['twin_free_lo'] - ref) * (r['twin_free_hi'] - ref) > 0
        if was != now:
            crossed.append(dict(split=r['split'], group=r['group'], quantity=r['quantity'], arm=r['arm'],
                                reference=ref, published=[r['published_lo'], r['published_hi']],
                                twin_free=[r['twin_free_lo'], r['twin_free_hi']],
                                direction='now includes the reference' if was else 'now excludes the reference'))
    reversed_sign = [r for r in rows_pq
                     if r['group'].endswith('contrast') and r['twin_free'] is not None
                     and r['published'] is not None and np.sign(r['published']) != np.sign(r['twin_free'])]
    moved = [r for r in rows_pq
             if r['twin_free'] is not None and r['published'] is not None and r['published_hi'] is not None
             and abs(r['twin_free'] - r['published']) > 0.5 * (r['published_hi'] - r['published_lo'])]

    if crossed or reversed_sign:
        verdict = 'K3 A CONCLUSION CHANGES'
    elif moved:
        verdict = 'K2 CHANGE CONFINED TO MAGNITUDES'
    else:
        verdict = 'K1 NO CHANGE'
    keys = ('split', 'quantity', 'arm', 'published', 'twin_free')
    verdict_k = dict(verdict=verdict, crossed=crossed,
                     sign_reversals=[{k: r[k] for k in keys} for r in reversed_sign],
                     moved=[{k: r[k] for k in keys} for r in moved])
    say('[verdict K] %s - %d interval crossings, %d contrast sign reversals, %d point estimates moved by more '
        'than half the published interval' % (verdict, len(crossed), len(reversed_sign), len(moved)))

    L = [dict(split=r['split'], group=r['group'], quantity=r['quantity'], arm=r['arm'],
              excluded_minus_retained=float(r['excluded_mean'] - r['twin_free']), n_excluded=r['n_excluded'])
         for r in rows_pq if r['excluded_mean'] is not None and r['twin_free'] is not None]
    headline = [x for x in L if x['arm'] in ('recipe', 'recipe_pds')
                and x['quantity'] in ('gene identity', 'cross-plate discrimination (published construction)')]
    for x in headline:
        say('[verdict L] %s %s %s: excluded minus retained %+.4f (%d excluded conditions)'
            % (x['split'], x['quantity'], x['arm'], x['excluded_minus_retained'], x['n_excluded']))
    return verdict_k, dict(headline=headline, all=L), headline


def main():
    if len(sys.argv) > 1:
        die('this script takes no options (prereg: all three splits, fixed quantities and exclusion lists)')
    for q in (OUT_JSON, OUT_PQ, OUT_SUM):
        if os.path.exists(q):
            die('%s exists; refusing to overwrite a record' % q)
    if not os.path.isfile(PRE) or open(PRE).read().count('\n## ') < MIN_SECTIONS:
        die('pre-registration missing or incomplete')

    # ---- prereg 7.1
    if TEST_DIR:
        say('[WARNING] S28_TEST_DIR is set: records read from %s and the commit guards are not enforced. This is '
            'the synthetic test of prereg 8.2; its outputs are not a result.' % TEST_DIR)
        commit = 'SYNTHETIC-TEST'
    else:
        s23.committed_and_clean(
            [PRE, 'score_step28_salt_twins.py', 'run_step28_score.sh', 'check_split_parent_overlap.py',
             'PREREG_step27_within_plate.md', 'score_step27_tahoe.py', 'score_step26_tahoe.py',
             'score_step25_tahoe.py', 'score_step15_tahoe.py', 'score_step15c_tahoe.py', 'score_step23_screen.py'],
            'pre-registration or scripts')
        s23.committed_and_clean(
            [OVERLAP, PC15B, PC15C, LONG15B, S25_PC, S26_PC, S27_PC, s15.RECORD,
             'step25_findings.md', 'step26_findings.md', 'step27_findings.md'],
            'records (Step 27 must be scored and its findings committed first)')
        commit = s23.git('rev-parse', 'HEAD').stdout.strip()
        say('[guards] pre-registration, scripts and records committed and clean; commit %s' % commit[:7])

    record = json.load(open(RECORD_PATH))
    ov = json.load(open(OVERLAP))

    # ---- prereg 7.3: the literal lists equal the committed record
    for sp in SPLITS:
        from_record = sorted(x['structure_id'] for x in ov['tahoe'][sp]['held_out_with_parent_in_training'])
        if from_record != sorted(EXCLUDE[sp]):
            die('%s: exclusion list differs from %s\n  script: %s\n  record: %s'
                % (sp, OVERLAP, sorted(EXCLUDE[sp]), from_record))
        if len(EXCLUDE[sp]) != EXPECT_N[sp]:
            die('%s: %d excluded structures, the pre-registration says %d' % (sp, len(EXCLUDE[sp]), EXPECT_N[sp]))
    say('[exclusion] lists match the committed record: %s'
        % ', '.join('%s %d' % (sp, len(EXCLUDE[sp])) for sp in SPLITS))

    pc15b = pd.read_csv(PC15B)
    long15b = pd.read_csv(LONG15B)
    pc15c = pd.read_csv(PC15C)
    p25 = pd.read_csv(S25_PC)
    p26 = pd.read_csv(S26_PC)
    p27 = pd.read_csv(S27_PC)

    # ---- gate 1, over every split and line at once
    proved_pairs = gate_aggregation(pc15b, long15b)
    metrics_proved = sorted({m for _, m in proved_pairs})

    rows_pq, summary, gate2_worst = [], {}, {}
    for sp in SPLITS:
        b = pc15b[pc15b['split'] == sp].reset_index(drop=True)
        n = len(b)
        clusters = b['structure'].astype(str).values
        n_drugs = len(set(clusters))
        excl_vals, how = resolve_exclusions(record, clusters, sp, EXCLUDE[sp])
        is_excl = np.array([c in excl_vals for c in clusters])
        if is_excl.sum() == 0:
            die('%s: the exclusion list matched no condition' % sp)
        retained = np.flatnonzero(~is_excl)
        excluded = np.flatnonzero(is_excl)
        all_idx = np.arange(n)

        # ---- gate 2, on this split
        gate2_worst[sp] = gate_interval(b, sp, clusters, n, n_drugs)

        say('[%s] %d conditions, %d structures; excluding %d structures (matched by %s) = %d conditions (%.1f %%), '
            '%d conditions retained' % (sp, n, n_drugs, len(EXCLUDE[sp]), how, len(excluded),
                                        100.0 * len(excluded) / n, len(retained)))

        def add(group, quantity, arm, values, values_b=None):
            """One row of the per-quantity table: published (all), twin-free (retained), excluded-only."""
            if values_b is None:
                full = summarise(values, all_idx, clusters, n)
                kept = summarise(values, retained, clusters, n)
                drop = summarise(values, excluded, clusters, n)
            else:
                full = paired(values, values_b, all_idx, clusters, n)
                kept = paired(values, values_b, retained, clusters, n)
                drop = paired(values, values_b, excluded, clusters, n)
            rows_pq.append(dict(split=sp, group=group, quantity=quantity, arm=arm,
                                published=full['mean'], published_lo=full['lo'], published_hi=full['hi'],
                                twin_free=kept['mean'], twin_free_lo=kept['lo'], twin_free_hi=kept['hi'],
                                excluded_mean=drop['mean'], excluded_lo=drop['lo'], excluded_hi=drop['hi'],
                                n_published=full['n'], n_twin_free=kept['n'], n_excluded=drop['n'],
                                structures_twin_free=kept['n_structures']))

        # group A: only the metrics whose aggregation gate 1 proved
        for metric in metrics_proved:
            tmpl, label = METRIC_COLUMN[metric], METRIC_LABEL[metric]
            for arm in ARMS_A:
                col = tmpl % arm
                if col in b.columns:
                    add('A', label, arm, pd.to_numeric(b[col], errors='coerce').values)
            for cname, a_, b_ in CONTRASTS_A:
                ca, cb = tmpl % a_, tmpl % b_
                if ca in b.columns and cb in b.columns:
                    add('A-contrast', label, cname, pd.to_numeric(b[ca], errors='coerce').values,
                        pd.to_numeric(b[cb], errors='coerce').values)

        # group C: the published cross-plate discrimination (Step 15c construction)
        c = pc15c[pc15c['split'] == sp].reset_index(drop=True)
        if len(c) != n or not np.array_equal(c['condition'].astype(str).values, b['condition'].astype(str).values):
            die('%s: the Step 15c per-condition record is not in the Step 15b order' % sp)
        for col in [x for x in c.columns if x.endswith('_pds')]:
            add('C', 'cross-plate discrimination (published construction)', col[:-4],
                pd.to_numeric(c[col], errors='coerce').values)

        # groups C and D: every scored column of Steps 25, 26 and 27
        for src, df, group, label in ((S25_PC, p25, 'C', 'cross-plate discrimination (fingerprint-swap arms)'),
                                      (S26_PC, p26, 'C', 'cross-plate discrimination (control-source schemes)'),
                                      (S27_PC, p27, 'D', 'within-plate discrimination and plate residual')):
            d = df[df['split'] == sp].reset_index(drop=True)
            if len(d) != n:
                die('%s: %s has %d rows for this split, expected %d' % (sp, src, len(d), n))
            if not np.array_equal(d['condition'].astype(str).values, b['condition'].astype(str).values):
                die('%s: %s is not in the Step 15b condition order' % (sp, src))
            for col in d.columns:
                if not is_scored_column(col):
                    continue
                v = pd.to_numeric(d[col], errors='coerce').values
                if not np.isfinite(v).any():
                    continue
                q = 'plate residual' if col.startswith('residual__') else label
                add(group, q, col, v)

        summary[sp] = dict(n_conditions=int(n), n_structures=int(n_drugs),
                           excluded_structures=sorted(EXCLUDE[sp]), matched_by=how,
                           excluded_conditions=int(len(excluded)), retained_conditions=int(len(retained)),
                           excluded_fraction=float(len(excluded)) / n)

    verdict_k, verdict_l, headline = compute_verdicts(rows_pq)

    # ---- outputs (prereg 7.5)
    with open(OUT_PQ, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_pq[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows_pq)
    with open(OUT_SUM, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['split', 'n_conditions', 'n_structures', 'excluded_structures', 'excluded_conditions',
                    'retained_conditions', 'excluded_fraction', 'matched_by'])
        for sp in SPLITS:
            s_ = summary[sp]
            w.writerow([sp, s_['n_conditions'], s_['n_structures'], ' '.join(s_['excluded_structures']),
                        s_['excluded_conditions'], s_['retained_conditions'],
                        '%.6f' % s_['excluded_fraction'], s_['matched_by']])
    json.dump(dict(prereg=PRE, commit=commit, synthetic_test=bool(TEST_DIR), splits=summary,
                   verdict_K=verdict_k, verdict_L=verdict_l,
                   metrics_proved=metrics_proved, n_rows=len(rows_pq),
                   interval='97.5% percentile per split, structure-clustered, fresh RandomState(0) per scored set',
                   gates='gate 1: per-line nanmean equals step15_model_long to %g; gate 2: s23.Boot equals '
                         'score_step15_tahoe.boot (s15.rng reset to RandomState(0) per call) to %g'
                         % (REPRO_TOL, INTERVAL_TOL),
                   gate2_largest_difference=gate2_worst,
                   scope='Tahoe-100M, prereg groups A, C, D; group E withdrawn (dated amendment); group B '
                         '(sci-Plex) is a separate script',
                   seconds=round(time.time() - T0, 1)), open(OUT_JSON, 'w'), indent=2)
    say('[written] %s, %s, %s  (%.1f s)' % (OUT_JSON, OUT_PQ, OUT_SUM, time.time() - T0))


if __name__ == '__main__':
    main()
