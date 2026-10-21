#!/usr/bin/env python3
"""
Step 19 diagnostic: how much of heterogeneity agreement (HA) comes from the control cells rather than the compound?

EXPLORATORY. Not a pre-registered endpoint, not gated, and it changes nothing in the Step 19 verdict, endpoints,
contexts or categories. It exists because the development split showed the compound-blind baseline (mean_response)
resolving above zero on HA (+0.13112 [+0.04919, +0.20958]). mean_response adds ONE vector per cell line
(score_step16_tahoe.rebuild: mr[ixl] = Y[ixl].mean(axis=0)), so inside a (line, dose) context its predicted spread
change varies only through the control cells, by clipping at zero. HA can therefore be earned without knowing the
compound. This script measures how much, from the committed Step 19 records alone (no arrays, no model, no GPU).
It is committed before it is run, like every scorer in this project.

Two routes by which the control cells can enter HA, and the measure that removes each:
  denominator route   R_true = S_true / S_ctrl and R_pred = S_pred / S_ctrl share S_ctrl. A prediction whose spread
                      does not follow the control (the recipe) still correlates with R_true through 1 / S_ctrl.
                      Removed by partialling out S_ctrl.
  composition route   the control sample's make-up decides what clipping does to a shift (and, through tracking, what
                      the OT-paired model generates). Removed by partialling out the compound-blind baseline's own
                      prediction, which is built from nothing but the control sample and a per-line constant.

Measures (each a mean over the SAME (line, dose) contexts as HA, >= 10 conditions, Spearman within each context):
  HA               Spearman(R_pred, R_true): the pre-registered measure, recomputed here as a check.
  coupling         Spearman(X, S_ctrl): how strongly X follows the control's spread. Descriptive.
  HA|ctrl          partial Spearman(R_pred, R_true | S_ctrl): HA with the denominator route removed.
  HA|blind         partial Spearman(R_pred, R_true | R_pred of mean_response): HA with the composition route, as the
                   compound-blind baseline sees it, removed.
  SA|ctrl          partial Spearman(S_pred, S_true | S_ctrl): agreement on the spread itself, with no ratio anywhere,
                   beyond what the control's spread already says. The ratio-free check on HA|ctrl.

Interpretation rules, fixed before any output of this script exists:
  R1  The real spread ratio is coupled to its denominator if the interval of coupling(R_true, S_ctrl) excludes zero.
  R2  An arm (recipe, ot_pair) carries spread information beyond the control cells only if HA|ctrl, HA|blind and
      SA|ctrl are ALL above zero. Otherwise its HA is reported as "not separable from the control route on this
      measure". Not separable is not "absent": it means these data cannot tell the two apart.
  R3  If mean_response's HA|ctrl is above zero, the control cells reach HA through more than their spread, and
      HA|blind is the operative test for the arms; the findings say so.
  R4  Development split first, its log read, then all three splits, as for the scorer. Nothing here is used to
      change a Step 19 endpoint, and the confirmation splits are scored as pre-registered whatever this shows.

Intervals: 97.5 % structure-clustered percentile bootstrap, 5,000 replicates, RandomState(0) -- the SAME resamples as
the Step 19 scorer, checked: the HA intervals of the three additive shifts must reproduce the scorer's Q3 intervals
to 1e-9. The arms' predictions in the per-condition record are averaged over the three seeds, so their HA here is
the HA of the seed-averaged prediction, printed beside the pre-registered per-seed-averaged HA; every plain-versus-
partial comparison is made on that same seed-averaged prediction, so it is like for like.

Reads   tahoe_results/step19_heterogeneity[_development].json, step19_per_condition[_development].csv
Writes  tahoe_results/step19_diag_control_route[_development].json, step19_diag_control_route_long[_development].csv
Run     python -u diag_step19_control_route.py --splits drug_split_1     (development, first)
        python -u diag_step19_control_route.py                           (all three, after the final Step 19 run)
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
import score_step15_tahoe as s15                      # noqa: E402  ALL_SPLITS, RES
import score_step16_tahoe as s16                      # noqa: E402  fast_rank, prep_ctx, mra_pre, ctx_split, verdict_of, fmt

ME = 'diag_step19_control_route.py'
SUBS = ('ridge_S7', 'ridge_train', 'mean_response', 'recipe', 'ot_pair')   # control_paste: R = 1, HA 0 by definition
ARMS = ('recipe', 'ot_pair')
BLIND = 'mean_response'                               # the compound-blind baseline
PASTE = 'control_paste'                               # returns the control cells unchanged: S_pred = S_ctrl
MIN_CTX, N_BOOT = s16.MIN_CTX, s16.N_BOOT
T0 = time.time()


# ------------------------------------------------------------------ rank statistics
def crank(v):
    """Centred average ranks (ties averaged). Spearman = the Pearson correlation of these. A constant vector gives
    all zeros, which rho() scores as 0 -- the same convention as score_step16_tahoe.mra_pre."""
    r = s16.fast_rank(np.asarray(v, np.float64))
    return r - r.mean()


def rho(a, b):
    """Pearson correlation of two centred rank vectors; 0 when either is constant."""
    da, db = np.sqrt(a @ a), np.sqrt(b @ b)
    return float(a @ b / (da * db)) if da > 0 and db > 0 else 0.0


def resid(c, cz):
    """What is left of centred ranks c after removing their least-squares share with centred ranks cz. A remainder
    smaller than 1e-9 of the original is set to exactly zero: c is then fully explained by cz, and the partial
    correlation is undefined, which rho() scores as 0 (the closed formula would return 0/0 rounding noise here)."""
    zz = cz @ cz
    if zz == 0:
        return c                                       # constant covariate: nothing to remove
    e = c - cz * ((c @ cz) / zz)
    return e if np.sqrt(e @ e) > 1e-9 * np.sqrt(c @ c) else np.zeros_like(c)


def partial(x, y, z):
    """Partial Spearman of x and y given z, from centred ranks: the correlation between what is left of x and of y
    after removing what each shares with z. If z is constant this is the plain Spearman; if x or y is constant, 0."""
    return rho(resid(x, z), resid(y, z))


def partial_self_check():
    """partial() is checked three ways: (1) against the closed formula (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)
    (1 - r_yz^2)), an independent derivation, wherever that formula is numerically safe; (2) a constant covariate
    gives the plain Spearman; (3) x ranked exactly like z gives 0. Ties, repeated entries (as a bootstrap draw makes)
    and a strong shared component (as a shared denominator makes) are all included."""
    r = np.random.RandomState(19)
    n_formula = 0
    for t in range(300):
        n = int(r.randint(10, 70))
        x, y, z = r.rand(n), r.rand(n), r.rand(n)
        if t % 3 == 0:
            x, y = np.round(x, 1), np.round(y, 1)
        if t % 5 == 0:
            k = r.randint(0, n, n)
            x, y, z = x[k], y[k], z[k]
        if t % 7 == 0:
            y = y - 3.0 * z
            x = x - 2.0 * z
        cx, cy, cz = crank(x), crank(y), crank(z)
        got = partial(cx, cy, cz)
        rxy, rxz, ryz = rho(cx, cy), rho(cx, cz), rho(cy, cz)
        if max(abs(rxz), abs(ryz)) < 0.999:
            want = (rxy - rxz * ryz) / np.sqrt((1 - rxz * rxz) * (1 - ryz * ryz))
            n_formula += 1
            if abs(got - want) > 1e-10:
                sys.exit('FATAL: partial() differs from the closed formula (%r vs %r, case %d)' % (got, want, t))
        if abs(partial(cx, cy, crank(np.ones(n))) - rxy) > 1e-12:
            sys.exit('FATAL: partial() with a constant covariate is not the plain Spearman (case %d)' % t)
        if partial(cz, cy, cz) != 0.0:
            sys.exit('FATAL: partial() of a variable fully explained by the covariate is not 0 (case %d)' % t)
    print('[self-check] partial Spearman == closed formula (%d well-conditioned cases); constant covariate -> plain '
          'Spearman; fully explained -> 0 (300 cases: ties, repeats, shared component)' % n_formula)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=list(s15.ALL_SPLITS))
    splits = list(ap.parse_args().splits)
    final = sorted(splits) == sorted(s15.ALL_SPLITS)
    if not final and splits != ['drug_split_1']:
        sys.exit('FATAL: run the development split alone (--splits drug_split_1) or all three splits')
    suffix = '' if final else '_development'
    REC = os.path.join(s15.RES, 'step19_heterogeneity%s.json' % suffix)
    PCF = os.path.join(s15.RES, 'step19_per_condition%s.csv' % suffix)
    OUT = os.path.join(s15.RES, 'step19_diag_control_route%s.json' % suffix)
    OUT_CSV = os.path.join(s15.RES, 'step19_diag_control_route_long%s.csv' % suffix)
    for q in (OUT, OUT_CSV):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)

    # --- guards: this script and the records it reads are committed and unmodified ----------------------------
    tracked = '%s %s %s score_step16_tahoe.py score_step15_tahoe.py score_step19_heterogeneity.py' % (ME, REC, PCF)
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: this script or the Step 19 records it reads are not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0:
        sys.exit('FATAL: this script or the Step 19 records it reads have uncommitted changes')
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    rec = json.load(open(REC))
    if bool(rec.get('final')) != final:
        sys.exit('FATAL: %s is not the %s record' % (REC, 'final' if final else 'development'))
    print('[guards] committed; commit %s; splits %s (%s)' % (commit[:7], splits, 'final' if final else 'development'))
    s16.mra_self_check()
    partial_self_check()

    # round_trip: read every float back exactly as the scorer wrote it, so the 1e-9 reproductions below are exact
    pcall = pd.read_csv(PCF, dtype={'condition': str, 'line': str, 'plate': str, 'structure': str},
                        float_precision='round_trip')

    results, rows = {}, []
    for split in splits:
        print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
        R19 = rec['per_split'][split]
        pc = pcall[pcall['split'] == split].reset_index(drop=True)
        n = len(pc)
        if n != int(R19['n_conditions']):
            sys.exit('FATAL: %d conditions in the per-condition record; the Step 19 record says %d'
                     % (n, R19['n_conditions']))
        lines, drugs = pc['line'].values, pc['structure'].values
        doses = pc['dose_nM'].values.astype(float)
        S_true = pc['spread_true'].values.astype(np.float64)
        S_ctrl = pc['spread_ctrl'].values.astype(np.float64)
        R_true = pc['R_true'].values.astype(np.float64)
        if np.abs(R_true - S_true / S_ctrl).max() > 1e-12 * np.abs(R_true).max():
            sys.exit('FATAL: R_true is not spread_true / spread_ctrl')
        R = {nm: pc['%s_R_pred' % nm].values.astype(np.float64) for nm in SUBS + (PASTE,)}
        if np.abs(R[PASTE] - 1.0).max() > 1e-9:
            sys.exit('FATAL: control_paste R is not 1')
        # predicted spread: R_pred = S_pred / S_ctrl with S_ctrl shared by all seeds, so S_pred = R_pred * S_ctrl
        S = {nm: R[nm] * S_ctrl for nm in SUBS}

        # --- contexts, exactly as the scorer builds them ---------------------------------------------------------
        ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
        ctx_u = np.unique(ctx_key)
        ctx_id = np.array([int(np.where(ctx_u == ctx_key[i])[0][0]) for i in range(n)])
        n_ctx = len(ctx_u)
        if n_ctx != int(R19['n_contexts']):
            sys.exit('FATAL: %d contexts; the Step 19 record says %d' % (n_ctx, R19['n_contexts']))
        full = np.arange(n)

        # --- reproduction 1: the pre-registered HA of the three additive shifts (one seed each, so exact) ---------
        pre_full = s16.prep_ctx(R_true, s16.ctx_split(ctx_id, n_ctx, full))
        for nm in ('ridge_S7', 'ridge_train', BLIND):
            got, want = s16.mra_pre(R[nm], pre_full), float(R19['metrics'][nm]['ha'])
            if abs(got - want) > 1e-9:
                sys.exit('FATAL: HA of %s (%.9f) does not reproduce the Step 19 record (%.9f)' % (nm, got, want))
        print('[reproduction] HA of the three additive shifts equals the Step 19 record to 1e-9')

        def stats(sel):
            """Every measure over the selection sel (condition indices; repeated entries in a bootstrap draw)."""
            acc = {}

            def put(key, v):
                acc.setdefault(key, []).append(v)

            for m in s16.ctx_split(ctx_id, n_ctx, sel):
                if len(m) < MIN_CTX:
                    continue                                       # too small to rank: contributes nothing, as in HA
                cz = crank(S_ctrl[m])                              # the control's spread
                if np.ptp(R_true[m]) > 0:                          # the HA contexts (score_step16_tahoe.prep_ctx rule)
                    cy = crank(R_true[m])
                    cb = crank(R[BLIND][m])
                    put(('coupling', 'R_true'), rho(cy, cz))
                    for nm in SUBS:
                        cx = crank(R[nm][m])
                        put(('ha', nm), rho(cx, cy))
                        put(('coupling', nm), rho(cx, cz))
                        put(('ha_ctrl', nm), partial(cx, cy, cz))
                        if nm != BLIND:
                            put(('ha_blind', nm), partial(cx, cy, cb))
                if np.ptp(S_true[m]) > 0:                          # the spread-level contexts
                    cs = crank(S_true[m])
                    put(('coupling', 'S_true'), rho(cs, cz))       # does treated spread follow control spread?
                    for nm in SUBS:
                        put(('sa_ctrl', nm), partial(crank(S[nm][m]), cs, cz))
            return {k: float(np.mean(v)) for k, v in acc.items()}

        point = stats(full)

        # --- reproduction 2: this routine's plain HA equals the scorer's machinery on all five submissions --------
        for nm in SUBS:
            if abs(point[('ha', nm)] - s16.mra_pre(R[nm], pre_full)) > 1e-12:
                sys.exit('FATAL: the plain HA of %s here differs from score_step16_tahoe.mra_pre' % nm)

        # --- bootstrap: the scorer's resamples, replicate for replicate -------------------------------------------
        gs = [np.where(drugs == u)[0] for u in np.unique(drugs)]
        rng = np.random.RandomState(0)
        B = {k: np.full(N_BOOT, np.nan) for k in point}
        for b in range(N_BOOT):
            sel = np.concatenate([gs[g] for g in rng.randint(0, len(gs), len(gs))])
            got = stats(sel)
            for k in point:
                B[k][b] = got.get(k, np.nan)
        con = {k: s16.verdict_of(point[k], B[k]) for k in point}

        # --- reproduction 3: same resamples as the scorer (its Q3 intervals, to 1e-9) ------------------------------
        for nm in ('ridge_S7', 'ridge_train', BLIND):
            want = R19['contrasts']['Q3_gate_ha_%s_minus_0' % nm]['ci']
            got = con[('ha', nm)]['ci']
            if max(abs(got[0] - want[0]), abs(got[1] - want[1])) > 1e-9:
                sys.exit('FATAL: the HA interval of %s (%s) does not reproduce the scorer (%s); different resamples'
                         % (nm, got, want))
        print('[reproduction] bootstrap intervals of the three additive shifts equal the scorer\'s Q3 intervals to '
              '1e-9 (same resamples)')

        # --- report ------------------------------------------------------------------------------------------------
        print('\ncoupling to the control spread (descriptive; within-context Spearman, mean over contexts)')
        print('  %-34s %s' % ('R_true  vs S_ctrl', s16.fmt(con[('coupling', 'R_true')])))
        print('  %-34s %s' % ('S_true  vs S_ctrl', s16.fmt(con[('coupling', 'S_true')])))
        for nm in SUBS:
            print('  %-34s %+.4f' % ('R_pred %s vs S_ctrl' % nm, point[('coupling', nm)]))

        print('\nHA, plain (seed-averaged prediction; pre-registered value in brackets)')
        for nm in SUBS:
            print('  %-14s %+.4f   [record %+.4f]' % (nm, point[('ha', nm)], float(R19['metrics'][nm]['ha'])))

        for key, title in (('ha_ctrl', 'HA | S_ctrl          (denominator route removed)'),
                           ('ha_blind', 'HA | mean_response   (composition route removed)'),
                           ('sa_ctrl', 'SA | S_ctrl          (spread itself, no ratio, beyond the control spread)')):
            print('\n' + title)
            for nm in SUBS:
                if (key, nm) in con:
                    print('  %-14s %s' % (nm, s16.fmt(con[(key, nm)])))

        # --- the fixed rules ---------------------------------------------------------------------------------------
        coupled = con[('coupling', 'R_true')]['verdict'] != 'includes zero'
        beyond_spread = con[('ha_ctrl', BLIND)]['verdict'] == 'above zero'
        decide = {}
        for a in ARMS:
            v = [con[(k, a)]['verdict'] for k in ('ha_ctrl', 'ha_blind', 'sa_ctrl')]
            decide[a] = ('beyond the control cells' if all(x == 'above zero' for x in v)
                         else 'not separable from the control route on this measure')
        print('\nR1  R_true coupled to its denominator: %s' % ('YES' if coupled else 'no'))
        print('R3  the control reaches HA through more than its spread (mean_response HA|ctrl above zero): %s'
              % ('YES -- HA|blind is the operative test' if beyond_spread else 'no'))
        for a in ARMS:
            print('R2  %-8s %s' % (a, decide[a]))

        results[split] = dict(
            n_conditions=n, n_contexts=n_ctx,
            point={'%s|%s' % k: v for k, v in point.items()},
            contrasts={'%s|%s' % k: c for k, c in con.items()},
            ha_record={nm: float(R19['metrics'][nm]['ha']) for nm in SUBS},
            rules=dict(R1_r_true_coupled=bool(coupled), R3_control_beyond_spread=bool(beyond_spread),
                       R2=decide))
        for k, c in con.items():
            rows.append(dict(split=split, measure=k[0], submission=k[1], value=c['diff'],
                             ci_lo=c['ci'][0], ci_hi=c['ci'][1], verdict=c['verdict']))
        print('[split] %s done (%d s)' % (split, time.time() - T0))

    out = dict(script=ME, exploratory=True, final=final, splits=splits,
               interval='97.5% structure-clustered percentile bootstrap (the Step 19 scorer\'s resamples)',
               n_boot=N_BOOT, min_context=MIN_CTX, inputs=dict(record=REC, per_condition=PCF),
               per_split=results, commit=commit, seconds=round(time.time() - T0, 1))
    json.dump(out, open(OUT, 'w'), indent=1)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'measure', 'submission', 'value', 'ci_lo', 'ci_hi', 'verdict'])
        w.writeheader()
        w.writerows(rows)
    print('\nwrote %s and %s  (%d s)' % (OUT, OUT_CSV, time.time() - T0))


if __name__ == '__main__':
    main()
