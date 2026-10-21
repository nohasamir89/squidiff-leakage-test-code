#!/usr/bin/env python3
"""
Step 22, stage 3 of 4: hits, chance floors, gate, contrasts and verdict from the R stage's answers
(PREREG_step22_fingerprint_identifiability.md, sections 4, 5 and 6).

Per split it reads tahoe_results/step22/<split>/: manifest.json (conditions, labels, draws, hashes), dictionary.csv
(which structures the package kept: K), map_<set>_group.csv (one answer per condition) and map_<set>_cell.csv (one
answer per cell) for the sets the manifest lists, timings.csv and r_session.txt.

Before any Step 22 number is printed (prereg 6) it checks, per split: the manifest was written at a commit on this
branch from this pre-registration; the Step 15b record, data record, OT record and m = 32 pin have the sha256 the
manifest recorded; every run the prep loaded still has the provenance it recorded; every matrix and metadata file on
disk has the manifest's sha256; the dictionary labels equal the manifest's; every mapping file lists every condition
(group level) or every flagged cell (cell level) exactly once.

Hit (prereg 4): the true structure is a member of the package's TOP credible set with sign '+'; an unassigned query
scores 0. Chance floor (prereg 4, amendment of 2026-10-14): the number of members of the top set with sign '+'
divided by K (0 if unassigned) - the probability that a random structure scores a hit; the sign-blind floor
|top set| / K is kept as 'chance_size'. Chance-corrected hit: hit - chance per query. Per condition: the value at group level; the mean over the condition's cells at cell level.
Arms: per seed, then averaged per condition. Conditions whose structure the package filtered out of the dictionary
are not covered and are excluded from every contrast (NaN).
Contrasts: 97.5 % structure-clustered percentile bootstrap, 5,000 replicates, generator reset to RandomState(0)
before every contrast (Step 20 convention), paired by condition, through score_step16_tahoe.contrast_pc.
Gated, on the chance-corrected hit: G1 real_half above zero, G2 real_half - mean_response (group level); Q1 recipe -
mean_response (group), Q2 recipe - mean_response (cell). Secondaries (prereg 4.3): the same contrasts on the raw hit,
pairing and reference contrasts on both, sign-agnostic and any-set variants; exploratory as 4.4. Verdict as prereg 5.

Outputs: tahoe_results/step22_analysis.json, step22_long.csv, step22_per_condition.csv (suffix _development when
fewer than the three splits are scored). Never overwrites.
--mock: scores tahoe_results/step22_mock/mock_split with relaxed guards and checks the planted expectations
(prep_step22_fingerprint.py --mock); outputs under tahoe_results/step22_mock/.

Run (squidiff env):  python -u score_step22_fingerprint.py [--splits drug_split_1] [--mock]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402  sha, RES, RECORD, rng (the bootstrap generator)
import score_step16_tahoe as s16                      # noqa: E402  contrast_pc, fmt, PC15
import score_step18_tahoe as s18                      # noqa: E402  OTREC
import score_step21_tahoe as s21                      # noqa: E402  guard_tracked
import prep_step22_fingerprint as p22                 # noqa: E402  the layout: OUT_ROOT, MOCK_ROOT, SETS, ARMS, REFS, C_CELL

PRE = p22.PRE
ME = 'score_step22_fingerprint.py'
ALL_SPLITS = p22.SPLITS                               # development first
ARMS, REFS, N_SEEDS = p22.ARMS, p22.REFS, p22.N_SEEDS
T0 = time.time()


def log(msg):
    print('[%6.0fs] %s' % (time.time() - T0, msg), flush=True)


def contrast(x_b, x_a, drugs):
    """score_step16_tahoe.contrast_pc with the bootstrap generator reset first (Step 20 convention)."""
    s15.rng = np.random.RandomState(0)
    return s16.contrast_pc(x_b, x_a, drugs)


def fmt(c):
    return s16.fmt(c)


# ------------------------------------------------------------------ parsing the package's answers
def members_of(s):
    """'P003,P010' -> ['P003', 'P010']; 'unassigned', 'NA' or '' -> []."""
    s = '' if s is None else str(s).strip()
    if s in ('', 'NA', 'unassigned', 'nan'):
        return []
    return [m for m in s.split(',') if m]


def parse_answer(label, top_set, all_sets, all_signs, K):
    """One query's answer -> dict(hit, hit_any_sign, any_set_hit, cs_size, unassigned, chance)."""
    top = members_of(top_set)
    if not top:
        return dict(hit=0.0, excess=0.0, hit_any_sign=0.0, any_set_hit=0.0, cs_size=0.0, unassigned=1.0, chance=0.0, chance_size=0.0)
    sets = [members_of(x) for x in str(all_sets).split(';')] if members_of(all_sets) or ';' in str(all_sets) else [top]
    signs = [x.split(',') for x in str(all_signs).split(';')] if str(all_signs) not in ('', 'NA', 'nan', 'None') else []
    if not sets or sets[0] != top:
        sys.exit('FATAL: the top credible set %r is not the first of all_credible_sets %r' % (top_set, all_sets))
    if len(signs) != len(sets) or any(len(a) != len(b) for a, b in zip(sets, signs)):
        sys.exit('FATAL: signs %r do not match the credible sets %r' % (all_signs, all_sets))
    in_top = label in top
    plus_top = in_top and signs[0][top.index(label)] == '+'
    any_plus = any(label in s_ and sg[s_.index(label)] == '+' for s_, sg in zip(sets, signs))
    chance = sum(1 for sg in signs[0] if sg == '+') / float(K)      # P(a random structure is a '+' member of the top set)
    return dict(hit=float(plus_top), excess=float(plus_top) - chance, hit_any_sign=float(in_top), any_set_hit=float(any_plus),
                cs_size=float(len(top)), unassigned=0.0, chance=chance, chance_size=len(top) / float(K))


METRICS = ('hit', 'excess', 'hit_any_sign', 'any_set_hit', 'cs_size', 'unassigned', 'chance', 'chance_size')


def map_path(folder, name, level):
    """The answer file of a set at a level: map_<set>_<level>.csv, or its gzipped form (the launcher gzips the cell-level
    files after the R stage); None if neither exists."""
    for ext in ('.csv', '.csv.gz'):
        q = os.path.join(folder, 'map_%s_%s%s' % (name, level, ext))
        if os.path.isfile(q):
            return q
    return None


def read_group(path, cond, label_of, K):
    """map_<set>_group.csv -> per-condition arrays in the manifest's condition order."""
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    if sorted(df['condition']) != sorted(cond) or len(df) != len(cond):
        sys.exit('FATAL: %s does not list every condition exactly once' % path)
    df = df.set_index('condition').loc[cond]
    for c in cond:
        if df.loc[c, 'label'] != label_of[c]:
            sys.exit('FATAL: %s labels %s as %s, manifest says %s' % (path, c, df.loc[c, 'label'], label_of[c]))
    out = {m: np.zeros(len(cond)) for m in METRICS}
    lbf = np.full(len(cond), np.nan)
    for i, c in enumerate(cond):
        r = df.loc[c]
        a = parse_answer(label_of[c], r['top_credible_set'], r['all_credible_sets'], r['all_signs'], K)
        for m in METRICS:
            out[m][i] = a[m]
        if r['top_lbf'] not in ('', 'NA'):
            lbf[i] = float(r['top_lbf'])
    out['top_lbf'] = lbf
    return out


def read_cell(path, cond, label_of, K, n_expected):
    """map_<set>_cell.csv -> per-condition means over the condition's cells."""
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    counts = df['condition'].value_counts()
    for i, c in enumerate(cond):
        if int(counts.get(c, 0)) != int(n_expected[i]):
            sys.exit('FATAL: %s has %d cells for %s, expected %d' % (path, counts.get(c, 0), c, n_expected[i]))
    if len(df) != int(np.sum(n_expected)):
        sys.exit('FATAL: %s has %d rows, expected %d' % (path, len(df), int(np.sum(n_expected))))
    vals = {m: [] for m in METRICS}
    conds = df['condition'].values
    for k in range(len(df)):
        r = df.iloc[k]
        if r['label'] != label_of[conds[k]]:
            sys.exit('FATAL: %s labels a cell of %s as %s' % (path, conds[k], r['label']))
        a = parse_answer(r['label'], r['top_credible_set'], r['all_credible_sets'], r['all_signs'], K)
        for m in METRICS:
            vals[m].append(a[m])
    per = pd.DataFrame(vals)
    per['condition'] = conds
    g = per.groupby('condition').mean()
    out = {m: g.loc[cond, m].values.astype(float) for m in METRICS}
    out['n_cells'] = counts.loc[cond].values.astype(int)
    return out


# ------------------------------------------------------------------ one split
def score_split(folder, split, mock, rows, pc_rows):
    if not os.path.isfile(os.path.join(folder, 'manifest.json')):
        sys.exit('FATAL: no manifest.json under %s (stage 1 not run for %s)' % (folder, split))
    man = json.load(open(os.path.join(folder, 'manifest.json')))
    if man.get('step') != 22 or man.get('split') != split or man.get('prereg') != PRE:
        sys.exit('FATAL: %s is not the Step 22 manifest of %s' % (folder, split))
    cond = [c['condition'] for c in man['conditions']]
    n = len(cond)
    label_of = {c['condition']: c['label'] for c in man['conditions']}
    drugs = np.array([c['structure'] for c in man['conditions']])
    lines = np.array([c['line'] for c in man['conditions']])
    doses = np.array([float(c['dose_nM']) for c in man['conditions']])
    n_B = np.array([int(c['n_B']) for c in man['conditions']])
    n_cell_expected = np.minimum(n_B, int(man['c_cell']))

    # ---- checks before any Step 22 number (prereg 6)
    if not mock:
        branch_commits = subprocess.run('git rev-list HEAD', shell=True, capture_output=True, text=True).stdout.split()
        if man['commit'] not in branch_commits:
            sys.exit('FATAL: the manifest of %s was written at commit %s, not on this branch' % (split, man['commit'][:7]))
        want = dict(step15b_per_condition=s16.PC15, ot_pairing=s18.OTREC, data_record=s15.RECORD, mb32_pin=p22.s20.pin_path(split, p22.M))
        for key, path in want.items():
            if s15.sha(path) != man['records'][key]:
                sys.exit('FATAL: %s has changed since the prep of %s' % (path, split))
        for name, rec in man['inputs'].items():
            tag = rec['tag']
            pj = ('provenance/s21ev_%s.json' % tag if tag.startswith('s21_') else 'provenance/s20ev_%s.json' % tag if tag.startswith('s20_')
                  else 'provenance/s18rv_%s.json' % tag if tag.startswith('s18r_') else 'provenance/s18ev_%s.json' % tag)
            pv = json.load(open(pj))
            if pv.get('vectors_sha256') != rec['vectors_sha256'] or pv.get('per_cell_sha256') != rec['per_cell_sha256']:
                sys.exit('FATAL: the provenance of %s differs from what the prep recorded' % tag)
    for name, f in man['files'].items():
        for part in ('matrix', 'meta'):
            path = os.path.join(folder, f[part]['path'])
            if not os.path.isfile(path) or s15.sha(path) != f[part]['sha256']:
                sys.exit('FATAL: %s differs from the manifest or is missing' % path)
    dic = pd.read_csv(os.path.join(folder, 'dictionary.csv'), keep_default_na=False, dtype=str)
    if sorted(dic['label']) != sorted(man['label_map'].values()):
        sys.exit('FATAL: dictionary.csv labels differ from the manifest')
    kept = set(dic.loc[dic['kept'].str.upper() == 'TRUE', 'label'])
    K = len(kept)
    if K < 2:
        sys.exit('FATAL: the dictionary kept %d fingerprint(s); the test needs at least 2' % K)
    covered = np.array([label_of[c] in kept for c in cond])
    for name in man['sets_group']:
        if map_path(folder, name, 'group') is None:
            sys.exit('FATAL: missing group-level answers for %s' % name)
    for name in man['sets_cell']:
        if map_path(folder, name, 'cell') is None:
            sys.exit('FATAL: missing cell-level answers for %s' % name)
    r_session = os.path.join(folder, 'r_session.txt')
    if not mock:
        if not os.path.isfile(r_session) or 'classify guard: present' not in open(r_session).read():
            sys.exit('FATAL: %s has no record that the package carried the classify guard (prereg 3.1 note of 2026-10-14)' % r_session)
    timings = pd.read_csv(os.path.join(folder, 'timings.csv')) if os.path.isfile(os.path.join(folder, 'timings.csv')) else None
    log('[checks] %s: manifest at commit %s; records, provenance, %d matrices and the dictionary labels verified; K = %d of %d '
        'structures kept; covered conditions %d of %d' % (split, man['commit'][:7] if not mock else 'mock', len(man['files']), K,
                                                          man['n_structures'], int(covered.sum()), n))

    # ---- read every answer file (no number printed yet)
    G = {name: read_group(map_path(folder, name, 'group'), cond, label_of, K) for name in man['sets_group']}
    Cc = {name: read_cell(map_path(folder, name, 'cell'), cond, label_of, K, n_cell_expected) for name in man['sets_cell']}
    log('[checks] every mapping file lists every condition (group) or every flagged cell (cell) exactly once')

    # ---- per-condition values: arms averaged over seeds; uncovered conditions NaN
    def collect(level_dict, names):
        out = {}
        for arm in ARMS:
            seeds = ['%s_seed_%d' % (arm, s) for s in range(N_SEEDS)]
            if all(s_ in level_dict for s_ in seeds):
                out[arm] = {m: np.nanmean(np.vstack([level_dict[s_][m] for s_ in seeds]), axis=0) for m in METRICS}
                out[arm]['per_seed'] = {m: [float(np.nanmean(np.where(covered, level_dict[s_][m], np.nan))) for s_ in seeds] for m in ('hit', 'excess', 'unassigned')}
        for name in names:
            if name in level_dict and not any(name.startswith(a + '_seed_') for a in ARMS):
                out[name] = {m: level_dict[name][m] for m in METRICS}
        for name in out:
            for m in METRICS:
                out[name][m] = np.where(covered, out[name][m], np.nan)
        return out
    GV = collect(G, man['sets_group'])
    CV = collect(Cc, man['sets_cell'])

    def rates(V):
        return {name: {m: float(np.nanmean(v[m])) for m in METRICS} for name, v in V.items()}
    res = dict(split=split, n_conditions=n, n_structures=int(man['n_structures']), K=K,
               kept=sorted(kept), filtered=sorted(set(man['label_map'].values()) - kept),
               coverage=dict(conditions=int(covered.sum()), structures=K, fraction_conditions=float(covered.mean())),
               c_cell=int(man['c_cell']), n_B=dict(min=int(n_B.min()), max=int(n_B.max()), mean=float(n_B.mean())),
               rates=dict(group=rates(GV), cell=rates(CV)),
               per_seed=dict(group={a: GV[a]['per_seed'] for a in ARMS if a in GV}, cell={a: CV[a]['per_seed'] for a in ARMS if a in CV}))
    print('\n' + '#' * 72 + '\n# %s: hits (top credible set, sign +) and chance-corrected hits, covered conditions only\n' % split + '#' * 72)
    print('%-16s %-7s %8s %8s %8s %8s %8s %8s %8s %8s' % ('set', 'level', 'hit', 'excess', 'chance', 'chance|.|', 'hit±', 'anyset', 'unassgn', '|topCS|'))
    for level, V in (('group', GV), ('cell', CV)):
        for name in V:
            r = res['rates'][level][name]
            print('%-16s %-7s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.2f' % (name, level, r['hit'], r['excess'], r['chance'], r['chance_size'],
                                                                                 r['hit_any_sign'], r['any_set_hit'], r['unassigned'], r['cs_size']))
    for a in ARMS:
        if a in GV:
            print('[per seed] %-8s group excess %s | cell excess %s' % (a, ['%.4f' % x for x in GV[a]['per_seed']['excess']],
                                                                     ['%.4f' % x for x in CV[a]['per_seed']['excess']] if a in CV else 'n/a'))

    # ---- contrasts (prereg 4.1-4.3)
    # gated contrasts on the chance-corrected hit (prereg 4, amendment of 2026-10-14); G1 = excess of real_half above zero
    c = {}
    c['G1_real_half_excess_group'] = contrast(GV['real_half']['excess'], np.zeros(n), drugs)
    c['G2_real_half_minus_mean_response_excess_group'] = contrast(GV['real_half']['excess'], GV['mean_response']['excess'], drugs)
    c['Q1_recipe_minus_mean_response_excess_group'] = contrast(GV['recipe']['excess'], GV['mean_response']['excess'], drugs)
    c['Q2_recipe_minus_mean_response_excess_cell'] = contrast(CV['recipe']['excess'], CV['mean_response']['excess'], drugs)
    # secondaries: the same four on the raw hit; pairing, ceiling and reference contrasts on both; the variants
    c['secondary_rawhit_G2_real_half_minus_mean_response_group'] = contrast(GV['real_half']['hit'], GV['mean_response']['hit'], drugs)
    c['secondary_rawhit_Q1_recipe_minus_mean_response_group'] = contrast(GV['recipe']['hit'], GV['mean_response']['hit'], drugs)
    c['secondary_rawhit_Q2_recipe_minus_mean_response_cell'] = contrast(CV['recipe']['hit'], CV['mean_response']['hit'], drugs)
    for level, V in (('group', GV), ('cell', CV)):
        for m, tag in (('excess', 'excess'), ('hit', 'rawhit')):
            c['secondary_%s_ot_pair_minus_recipe_%s' % (tag, level)] = contrast(V['ot_pair'][m], V['recipe'][m], drugs)
            c['secondary_%s_mb32_minus_recipe_%s' % (tag, level)] = contrast(V['mb32'][m], V['recipe'][m], drugs)
            c['secondary_%s_real_half_minus_recipe_%s' % (tag, level)] = contrast(V['real_half'][m], V['recipe'][m], drugs)
            c['secondary_%s_recipe_minus_control_paste_%s' % (tag, level)] = contrast(V['recipe'][m], V['control_paste'][m], drugs)
        for m in ('hit_any_sign', 'any_set_hit'):
            c['secondary_%s_recipe_minus_mean_response_%s' % (m, level)] = contrast(V['recipe'][m], V['mean_response'][m], drugs)
    for ref in ('ridge_train', 'ridge_S7'):
        for m, tag in (('excess', 'excess'), ('hit', 'rawhit')):
            c['secondary_%s_recipe_minus_%s_group' % (tag, ref)] = contrast(GV['recipe'][m], GV[ref][m], drugs)
    print('\n[gate] G1 real_half excess (group)                 %s\n[gate] G2 real_half - mean_response excess (group) %s'
          % (fmt(c['G1_real_half_excess_group']), fmt(c['G2_real_half_minus_mean_response_excess_group'])))
    print('[primary] Q1 recipe - mean_response excess (group) %s\n[primary] Q2 recipe - mean_response excess (cell)  %s'
          % (fmt(c['Q1_recipe_minus_mean_response_excess_group']), fmt(c['Q2_recipe_minus_mean_response_excess_cell'])))
    for key in c:
        if key.startswith('secondary'):
            print('[secondary] %-60s %s' % (key, fmt(c[key])))
    tests = dict(G1=c['G1_real_half_excess_group']['verdict'], G2=c['G2_real_half_minus_mean_response_excess_group']['verdict'],
                 Q1=c['Q1_recipe_minus_mean_response_excess_group']['verdict'], Q2=c['Q2_recipe_minus_mean_response_excess_cell']['verdict'])
    res['contrasts'], res['tests'] = c, tests
    res['gate'] = bool(tests['G1'] == 'above zero' and tests['G2'] == 'above zero')

    # ---- exploratory (prereg 4.4): per structure, per dose, uncovered conditions, Bayes factors
    per_struct = {}
    for lab in sorted(set(label_of.values())):
        ix = np.array([label_of[x] == lab for x in cond])
        per_struct[lab] = dict(structure=man['structures_sorted'][int(lab[1:]) - 1], n_conditions=int(ix.sum()), kept=lab in kept,
                               **{'%s_group' % nm: (float(np.nanmean(GV[nm]['hit'][ix])) if lab in kept else None) for nm in ('real_half', 'recipe', 'mean_response')},
                               **{'%s_cell' % nm: (float(np.nanmean(CV[nm]['hit'][ix])) if lab in kept else None) for nm in ('real_half', 'recipe', 'mean_response')})
    by_dose = {}
    for d in sorted(set(doses)):
        ix = doses == d
        by_dose['%g' % d] = {nm: float(np.nanmean(GV[nm]['hit'][ix])) for nm in ('real_half', 'recipe', 'mean_response')}
        by_dose['%g' % d]['n_conditions'] = int(ix.sum())
    def raw(name, m):
        # unmasked per-condition values of a set (arms: the mean over seeds), for the uncovered conditions too
        if name in ARMS:
            return np.nanmean(np.vstack([G['%s_seed_%d' % (name, s)][m] for s in range(N_SEEDS)]), axis=0)
        return G[name][m]
    unc = ~covered
    uncovered = dict(n_conditions=int(unc.sum()))
    if unc.any():
        for nm in ('real_half', 'recipe'):
            uncovered['%s_unassigned_fraction' % nm] = float(np.mean(raw(nm, 'unassigned')[unc]))
            uncovered['%s_mean_top_set_size' % nm] = float(np.mean(raw(nm, 'cs_size')[unc]))
    lbf = {}
    for nm in ('real_half', 'recipe'):
        sets_ = ['%s_seed_%d' % (nm, s) for s in range(N_SEEDS)] if nm in ARMS else [nm]
        h, mi = [], []
        for s_ in sets_:
            hit_, un_, lb_ = G[s_]['hit'], G[s_]['unassigned'], G[s_]['top_lbf']
            h.extend(lb_[covered & (hit_ == 1)].tolist())
            mi.extend(lb_[covered & (hit_ == 0) & (un_ == 0)].tolist())
        lbf[nm] = dict(hits=float(np.nanmean(h)) if h else None, misses=float(np.nanmean(mi)) if mi else None, n_hits=len(h), n_misses=len(mi))
    res['exploratory'] = dict(per_structure=per_struct, by_dose_group=by_dose, uncovered=uncovered, top_lbf_group=lbf)
    res['timings'] = (timings.to_dict(orient='records') if timings is not None else None)
    res['records'] = dict(manifest=s15.sha(os.path.join(folder, 'manifest.json')), dictionary_csv=s15.sha(os.path.join(folder, 'dictionary.csv')),
                          r_session=s15.sha(r_session) if os.path.isfile(r_session) else None,
                          maps={('%s_group' % nm): dict(file=os.path.basename(map_path(folder, nm, 'group')), sha256=s15.sha(map_path(folder, nm, 'group')))
                                for nm in man['sets_group']})
    res['records']['maps'].update({('%s_cell' % nm): dict(file=os.path.basename(map_path(folder, nm, 'cell')), sha256=s15.sha(map_path(folder, nm, 'cell')))
                                   for nm in man['sets_cell']})
    res['manifest_commit'] = man.get('commit')

    # ---- long rows and per-condition rows
    for level, V in (('group', GV), ('cell', CV)):
        for name in V:
            for m in METRICS:
                rows.append(dict(split=split, level='rate', metric=m, key='%s_%s' % (name, level), value=res['rates'][level][name][m], ci_lo='', ci_hi='', verdict=''))
    for key, cc in c.items():
        metric = ('excess' if 'excess' in key else 'hit' if 'rawhit' in key else 'hit_any_sign' if 'hit_any_sign' in key
                  else 'any_set_hit' if 'any_set_hit' in key else 'hit')
        rows.append(dict(split=split, level='contrast', metric=metric, key=key, value=cc['diff'], ci_lo=cc['ci'][0], ci_hi=cc['ci'][1],
                         verdict=cc['verdict']))
    for i in range(n):
        r = dict(split=split, condition=cond[i], label=label_of[cond[i]], structure=drugs[i], line=lines[i], dose_nM=doses[i],
                 covered=int(covered[i]), n_B=int(n_B[i]), n_cell=int(n_cell_expected[i]))
        for level, V in (('group', GV), ('cell', CV)):
            for name in V:
                r['%s_%s_hit' % (name, level)] = V[name]['hit'][i]
                r['%s_%s_excess' % (name, level)] = V[name]['excess'][i]
                r['%s_%s_chance' % (name, level)] = V[name]['chance'][i]
        pc_rows.append(r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default=','.join(ALL_SPLITS))
    ap.add_argument('--mock', action='store_true')
    a = ap.parse_args()
    if a.mock:
        splits, final, root, outdir, stem = ['mock_split'], False, p22.MOCK_ROOT, p22.MOCK_ROOT, 'step22_mock'
    else:
        splits = [s_.strip() for s_ in a.splits.split(',') if s_.strip()]
        for s_ in splits:
            if s_ not in ALL_SPLITS:
                sys.exit('FATAL: unknown split %s' % s_)
        final = sorted(splits) == sorted(ALL_SPLITS)
        root, outdir, stem = p22.OUT_ROOT, s15.RES, 'step22' + ('' if final else '_development')
    OUT, OUT_CSV, OUT_PC = [os.path.join(outdir, '%s_%s' % (stem, x)) for x in ('analysis.json', 'long.csv', 'per_condition.csv')]
    for q in (OUT, OUT_CSV, OUT_PC):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    if open(PRE).read().count('\n## ') < 10:
        sys.exit('FATAL: prereg incomplete')
    tracked = [PRE, ME, 'prep_step22_fingerprint.py', 'run_step22_fingerprint.R', 'run_step22_tahoe.sh', 'score_step15_tahoe.py',
               'score_step15c_tahoe.py', 'score_step16_tahoe.py', 'score_step18_tahoe.py', 'score_step20_tahoe.py', 'score_step21_tahoe.py']
    if not a.mock:
        tracked += [s16.PC15, s18.OTREC, s15.RECORD] + [p22.s20.pin_path(s_, p22.M) for s_ in splits]
    s21.guard_tracked(tracked)
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    log('[prereg] committed; scripts committed; commit %s; splits %s (%s)' % (commit[:7], splits, 'mock' if a.mock else 'final' if final else 'development'))

    # every split's stage-1 and stage-2 outputs must be complete before the first number of any split is printed
    for split in splits:
        folder = os.path.join(root, split)
        if not os.path.isfile(os.path.join(folder, 'manifest.json')):
            sys.exit('FATAL: no manifest.json under %s (stage 1 not run for %s)' % (folder, split))
        m_ = json.load(open(os.path.join(folder, 'manifest.json')))
        missing = ([q for q in ['dictionary.csv'] if not os.path.isfile(os.path.join(folder, q))]
                   + ['map_%s_group' % x for x in m_['sets_group'] if map_path(folder, x, 'group') is None]
                   + ['map_%s_cell' % x for x in m_['sets_cell'] if map_path(folder, x, 'cell') is None])
        if missing:
            sys.exit('FATAL: %s lacks %d stage-2 output(s), e.g. %s' % (split, len(missing), missing[:3]))
    results, rows, pc_rows = {}, [], []
    for split in splits:
        results[split] = score_split(os.path.join(root, split), split, a.mock, rows, pc_rows)

    # ---- verdict (prereg 5) over the scored splits
    Gt = all(results[s_]['gate'] for s_ in splits)
    E = all(results[s_]['tests']['Q1'] == 'above zero' for s_ in splits)
    P = all(results[s_]['tests']['Q2'] == 'above zero' for s_ in splits)
    if not Gt:
        verdict = '0 UNINFORMATIVE'
    elif E and P:
        verdict = '1 SIGNATURE CONFIRMED AT CELL LEVEL'
    elif E:
        verdict = '2 SIGNATURE CONFIRMED AT GROUP LEVEL ONLY'
    else:
        verdict = '3 NOT CONFIRMED'
    rev = [(s_, k) for s_ in splits for k in ('Q1', 'Q2') if results[s_]['tests'][k] == 'below zero']
    if rev:
        verdict += ' with a reversal on ' + ', '.join('%s %s' % (s_, 'group' if k == 'Q1' else 'cell') for s_, k in rev)
    scope = 'all three splits' if final else ('the mock split' if a.mock else 'the scored split(s) %s (development; not the final verdict)' % splits)
    print('\n' + '=' * 72 + '\nVERDICT (%s): %s\n' % (scope, verdict) + '=' * 72)
    for s_ in splits:
        t = results[s_]['tests']
        print('  %-13s gate G1 %-14s G2 %-14s | Q1 %-14s Q2 %-14s | K %d, covered %d/%d'
              % (s_, t['G1'], t['G2'], t['Q1'], t['Q2'], results[s_]['K'], results[s_]['coverage']['conditions'], results[s_]['n_conditions']))

    mock_check = None
    if a.mock:
        r = results['mock_split']
        checks = dict(gate=r['gate'], Q1_above=r['tests']['Q1'] == 'above zero', Q2_above=r['tests']['Q2'] == 'above zero',
                      real_half_group_excess_above_0_5=r['rates']['group']['real_half']['excess'] > 0.5,
                      control_paste_group_excess_within_0_1_of_zero=abs(r['rates']['group']['control_paste']['excess']) < 0.1)
        mock_check = dict(checks=checks, passed=all(checks.values()))
        print('\n[mock] planted expectations: %s -> %s' % (checks, 'PASS' if mock_check['passed'] else 'FAIL'))

    rec = dict(step=22, prereg=PRE, final=final, mock=a.mock, splits=splits, verdict=verdict, G=Gt, E=E, P=P, reversals=rev,
               per_split=results, mock_check=mock_check,
               definitions=dict(hit='true structure in the top credible set with sign +; unassigned = 0',
                                chance='members of the top set with sign + / K (prereg 4 amendment 2026-10-14); chance_size = |top set| / K',
                                excess='hit - chance per query (the chance-corrected hit; gated contrasts use it, prereg 4 amendment 2026-10-14)',
                                interval='97.5 % structure-clustered percentile bootstrap, 5000 replicates, RandomState(0) reset per contrast',
                                arms='per seed, then averaged per condition', covered='conditions whose structure the package kept'),
               commit=commit, seconds=round(time.time() - T0))
    os.makedirs(outdir, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(rec, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'level', 'metric', 'key', 'value', 'ci_lo', 'ci_hi', 'verdict'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    pd.DataFrame(pc_rows).to_csv(OUT_PC, index=False)
    log('wrote %s, %s, %s' % (OUT, OUT_CSV, OUT_PC))


if __name__ == '__main__':
    main()
