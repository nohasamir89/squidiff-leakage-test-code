#!/usr/bin/env python
"""Step 19 scoring: does the model predict how a compound changes the SPREAD of a cell population?

Pre-registration: PREREG_step19_heterogeneity.md, committed before this file existed. This script does only what that
document says, by the Step 16/18 code paths. No arrays are produced: it is a further read of the per-cell populations
stored by the Step 18 evaluation (the only ones that carry the conditioning control cells, `ctrl`).

  1. guards: the pre-registration, this scorer and the scorers it imports committed and clean; outputs absent;
  2. the Step 18 arrays loaded through score_step18_tahoe.load_eval18, which hash-checks each against its provenance;
  3. the stored real cells and the conditioning control cells asserted identical across both arms and all seeds
     (prereg section 9.4): the control draw is seeded per condition, so this must hold and the comparison is invalid
     otherwise;
  4. Step 18's control tracking reproduced to 1e-9 as proof the same arrays are being read (prereg section 9.3);
  5. the measure (prereg section 3): spread S(X) = mean distance of a population's cells to their own centroid;
     R = S(treated) / S(control); heterogeneity agreement HA = the Step 16 section 4.4 context machinery with R in
     place of the response norms -- Spearman within each (line, dose) context of >= 10 conditions, mean over contexts,
     per seed then averaged;
  6. the four Step 15b shift baselines rebuilt by the Step 15b code path and SCORED ON THE SAME MEASURE: their cells
     are the conditioning control cells plus the shift, clipped at zero, so control_paste gives R = 1 exactly
     (asserted) and the three additive shifts give R near 1 only up to clipping, which is measured, not assumed;
  7. the replicate ceiling: on the plate 6 / plate 14 pairs, the partner copy's measured R as the prediction;
  8. the section-4 endpoints and the section-6 verdict (structure-clustered percentile bootstrap, 5,000 replicates,
     RandomState(0), 97.5 % per split, HA recomputed per replicate on the resampled structures).

Usage: python -u score_step19_heterogeneity.py --splits drug_split_1     (development run first; no --splits = final)
Outputs: tahoe_results/step19_heterogeneity[_development].json, step19_long[_development].csv,
         step19_per_condition[_development].csv. Never overwrites.
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15                      # noqa: E402  loaders, constants, bootstrap, verdict
import score_step15c_tahoe as s15c                    # noqa: E402  load_vectors, close, contrast
import score_step16_tahoe as s16                      # noqa: E402  rebuild, prep_ctx, mra_pre, ctx_split, verdict_of, fmt
import score_step18_tahoe as s18                      # noqa: E402  load_eval18, tracking

PRE = 'PREREG_step19_heterogeneity.md'
PC15, PC15C = s16.PC15, s16.PC15C
REC15B, REC15C, REC16 = s16.REC15B, s16.REC15C, s18.REC16
REC18 = os.path.join(s15.RES, 'step18_analysis.json')
OTREC = s18.OTREC
NEW, OLD = 'ot_pair', 'recipe'
ARMS = (OLD, NEW)
SHIFTS = tuple(s15.BASELINES)                          # ridge_S7, ridge_train, mean_response, control_paste
MIN_CTX, N_BOOT = s16.MIN_CTX, s16.N_BOOT
T0 = time.time()


def spread(X):
    """Mean Euclidean distance of a population's cells to their own centroid (prereg section 3)."""
    X = np.asarray(X, np.float64)
    if X.shape[0] < 2:
        return np.nan
    return float(np.linalg.norm(X - X.mean(axis=0, keepdims=True), axis=1).mean())


def spreads(cells):
    return np.array([spread(X) for X in cells], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=list(s15.ALL_SPLITS))
    args = ap.parse_args()
    splits = list(args.splits)
    for s in splits:
        if s not in s15.ALL_SPLITS:
            sys.exit('FATAL: unknown split %s' % s)
    final = sorted(splits) == sorted(s15.ALL_SPLITS)
    suffix = '' if final else '_development'
    OUT = os.path.join(s15.RES, 'step19_heterogeneity%s.json' % suffix)
    OUT_CSV = os.path.join(s15.RES, 'step19_long%s.csv' % suffix)
    OUT_PC = os.path.join(s15.RES, 'step19_per_condition%s.csv' % suffix)
    for q in (OUT, OUT_CSV, OUT_PC):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)

    # --- guards -------------------------------------------------------------------------------------------------
    if open(PRE).read().count('\n## ') < 9:
        sys.exit('FATAL: prereg incomplete')
    tracked = ('%s score_step19_heterogeneity.py score_step18_tahoe.py score_step16_tahoe.py score_step15_tahoe.py '
               'score_step15c_tahoe.py score_step15d_utility.py score_step15e_pooled.py step14_metrics.py '
               'Squidiff/scrna_datasets.py %s %s %s %s %s %s %s %s'
               % (PRE, PC15, PC15C, REC15B, REC15C, REC16, REC18, OTREC, s15.RECORD))
    if subprocess.run('git ls-files --error-unmatch ' + tracked, shell=True, capture_output=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers or records not committed')
    if subprocess.run('git diff --quiet -- ' + tracked, shell=True).returncode != 0:
        sys.exit('FATAL: prereg, scorers or records uncommitted')
    record = json.load(open(s15.RECORD))
    otrec = json.load(open(OTREC))
    rec18 = json.load(open(REC18))
    if not rec18.get('final'):
        sys.exit('FATAL: the Step 18 record is not the final one')
    commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    print('[prereg] committed; scorer committed; commit %s; splits %s (%s)' % (commit[:7], splits, 'final' if final else 'development'))
    s15c.self_check()
    s16.mra_self_check()
    pc15 = pd.read_csv(PC15)

    results, inputs, rows, pc_rows = {}, {}, [], []
    for split in splits:
        print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
        k = split.split('_')[-1]
        rec_split = record['splits'][k]
        n_drugs = len(rec_split['held_structures'])
        ot_sha = otrec['written'][split]['sha256']

        ref = s15c.load_vectors(split, OLD, 0)
        cond, n = ref['cond'], len(ref['cond'])
        if n != int(rec_split['n_test_conditions']):
            sys.exit('FATAL: %d conditions, record expects %d' % (n, rec_split['n_test_conditions']))
        true_mean, C = ref['true'], ref['C']
        Y = true_mean - C
        pc = pc15[pc15['split'] == split].reset_index(drop=True)
        if not np.array_equal(pc['condition'].astype(str).values, np.array([str(x) for x in cond])):
            sys.exit('FATAL: the Step 15b per-condition record lists different conditions')
        lines, plates = pc['line'].astype(str).values, pc['plate'].astype(str).values
        drugs, doses = pc['structure'].astype(str).values, pc['dose_nM'].astype(float).values
        lv = np.unique(lines)
        if len(np.unique(drugs)) != n_drugs:
            sys.exit('FATAL: %d structures, record expects %d' % (len(np.unique(drugs)), n_drugs))
        print('[design] %d conditions | %d held-out structures | %d lines | cells per condition up to %d'
              % (n, n_drugs, len(lv), s15.CAP))

        # --- the Step 18 arrays: spreads, tracking; real cells and ctrl asserted shared (prereg 9.3, 9.4) ---------
        S_pred = {a: [] for a in ARMS}
        ctrl_cells = real_cells = None                  # kept from the first run: the shared populations
        track, track_null, inp = {}, {}, {}
        # ONE RandomState(0) per split, advanced by the arm/seed loop in the same order as Step 18 section 4.4, so the
        # permutation null reproduces Step 18's recorded null as well as its tracking (a fresh seed per call would not)
        rng_null = np.random.RandomState(0)
        for kind, arm in (('recipe', OLD), ('ot', NEW)):
            tr_, tn_, inp[arm] = [], [], []
            for sd in range(s15.N_SEEDS):
                v = s18.load_eval18(split, sd, kind, ot_sha)
                if not np.array_equal(v['cond'], cond):
                    sys.exit('FATAL: condition order differs in %s' % v['tag'])
                if ctrl_cells is None:                   # the harness draws both with a per-condition seed
                    ctrl_cells = [np.asarray(x, np.float32) for x in v['ctrl']]
                    real_cells = [np.asarray(x, np.float32) for x in v['true_cells']]
                    S_ctrl, S_true = spreads(ctrl_cells), spreads(real_cells)
                else:
                    for i in range(n):
                        if not np.array_equal(np.asarray(v['ctrl'][i], np.float32), ctrl_cells[i]):
                            sys.exit('FATAL: the conditioning control cells of %s differ from the first run at '
                                     'condition %d; the control draw is seeded per condition, so they must be '
                                     'identical across arms and seeds and the comparison is invalid otherwise' % (v['tag'], i))
                        if not np.array_equal(np.asarray(v['true_cells'][i], np.float32), real_cells[i]):
                            sys.exit('FATAL: the stored real cells of %s differ from the first run at condition %d'
                                     % (v['tag'], i))
                S_pred[arm].append(spreads(v['pred']))
                tr_.append(s18.tracking(v['pred'], v['ctrl']))
                tn_.append(s18.tracking(v['pred'], v['ctrl'], rng_null))
                inp[arm].append(v['rec'])
                del v
            track[arm], track_null[arm] = tr_, tn_
        print('[shared] the conditioning control cells and the stored real cells are identical across both arms and all %d seeds'
              % s15.N_SEEDS)
        if not np.all(np.isfinite(S_true)) or not np.all(np.isfinite(S_ctrl)) or float(np.min(S_ctrl)) <= 0:
            sys.exit('FATAL: non-finite or non-positive spreads in the measured or control populations')

        # --- reproduction: Step 18's tracking, to 1e-9 (prereg section 9.3) --------------------------------------
        # Averaged exactly as Step 18 averages it -- over seeds per condition, then over conditions -- so that a
        # condition whose tracking is undefined in one seed cannot make this guard fire for a legitimate reason.
        for arm in ARMS:
            m18 = rec18['per_split'][split]['metrics'][arm]
            for label, got, want in (
                    ('tracking', float(np.nanmean(np.nanmean(np.vstack(track[arm]), axis=0))),
                     float(m18['tracking'])),
                    ('its permutation null', float(np.nanmean(np.nanmean(np.vstack(track_null[arm]), axis=0))),
                     float(m18['tracking_null']))):
                if abs(got - want) > 1e-9:
                    sys.exit('FATAL: %s of %s (%.9f) does not reproduce the Step 18 record (%.9f)'
                             % (label, arm, got, want))
            per_seed = [float(np.nanmean(x)) for x in track[arm]]
            for sd, (got, want) in enumerate(zip(per_seed, m18['tracking_per_seed'])):
                if abs(got - float(want)) > 1e-9:
                    sys.exit('FATAL: tracking of %s seed %d (%.9f) does not reproduce the Step 18 record (%.9f)'
                             % (arm, sd, got, float(want)))
        print('[reproduction] control tracking of both arms, per seed and pooled, and the permutation null equal the '
              'Step 18 record to 1e-9')

        # --- the shift baselines on the same measure (prereg section 5) ------------------------------------------
        deltas, _train = s16.rebuild(split, lines, drugs, doses, lv, Y)
        for bname, D in deltas.items():
            j, d = s15.thesis_per_cond(true_mean, C + D, C)
            if not s15c.close(j, pc['%s_de_overlap' % bname].values) or not s15c.close(d, pc['%s_dpc50' % bname].values):
                sys.exit('FATAL: %s does not reproduce the Step 15b per-condition record' % bname)
        print('[reproduction] DE-overlap and DPC-50 of the four shift baselines equal the Step 15b record to 1e-9')
        S_shift = {}                                   # the shifts act on the SAME control cells the arms were given
        for bname, D in deltas.items():
            S_shift[bname] = np.array([spread(np.clip(np.asarray(ctrl_cells[i], np.float64) + D[i], 0, None))
                                       for i in range(n)])
        if np.abs(S_shift['control_paste'] - S_ctrl).max() > 1e-9:
            sys.exit('FATAL: control_paste spread differs from the control spread; it pastes the cells unchanged')
        del ctrl_cells, real_cells

        # --- R, contexts, HA ------------------------------------------------------------------------------------
        R_true = S_true / S_ctrl
        R_pred = {a: [S / S_ctrl for S in S_pred[a]] for a in ARMS}
        for bname in SHIFTS:
            R_pred[bname] = [S_shift[bname] / S_ctrl]
        if np.abs(R_pred['control_paste'][0] - 1.0).max() > 1e-9:
            sys.exit('FATAL: control_paste R is not 1; the zero point of prereg section 5 is broken')
        clip_var = {b: float(np.std(R_pred[b][0])) for b in SHIFTS}
        print('[zero point] control_paste R = 1 to 1e-9; additive-shift R spread from clipping: %s'
              % {b: round(clip_var[b], 5) for b in SHIFTS})

        ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
        ctx_u = np.unique(ctx_key)
        contexts = [np.where(ctx_key == u)[0] for u in ctx_u]
        ctx_id = np.array([int(np.where(ctx_u == ctx_key[i])[0][0]) for i in range(n)])
        n_ctx = len(contexts)
        full = np.arange(n)

        keyed = {}
        for i, c in enumerate(cond):
            key, plate = s15.parse_cond(c)
            if plate in s15.REP_PLATES:
                keyed.setdefault(key, {})[plate] = i
        pairs = [(v['6'], v['14']) for v in keyed.values() if '6' in v and '14' in v]
        partner = np.full(n, -1)
        for i6, i14 in pairs:
            partner[i6], partner[i14] = i14, i6
        rep_ix = np.where(partner >= 0)[0]
        rep_R = np.zeros(n)
        rep_R[rep_ix] = R_true[partner[rep_ix]]

        def ha_all(sel):
            pre = s16.prep_ctx(R_true, s16.ctx_split(ctx_id, n_ctx, sel))
            out = {nm: float(np.mean([s16.mra_pre(x, pre) for x in R_pred[nm]])) for nm in R_pred}
            s_ = sel[partner[sel] >= 0]
            out['ceiling'] = (s16.mra_pre(rep_R, s16.prep_ctx(R_true, s16.ctx_split(ctx_id, n_ctx, s_)))
                              if len(s_) else np.nan)
            return out

        pre_full = s16.prep_ctx(R_true, s16.ctx_split(ctx_id, n_ctx, full))
        ha_seed = {nm: [s16.mra_pre(x, pre_full) for x in R_pred[nm]] for nm in R_pred}
        ha = {nm: float(np.mean(v_)) for nm, v_ in ha_seed.items()}
        if abs(ha['control_paste']) > 1e-12:
            sys.exit('FATAL: control_paste HA is %.3g, expected 0 by definition (constant R)' % ha['control_paste'])
        ha_ceil = ha_all(full)['ceiling']
        n_ctx_rep = sum(1 for c in range(n_ctx) if (ctx_id[rep_ix] == c).sum() >= MIN_CTX)
        print('[ha] %d contexts | replicate ceiling %s over %d contexts (%d pairs)'
              % (n_ctx, 'n/a' if not np.isfinite(ha_ceil) else '%.4f' % ha_ceil, n_ctx_rep, len(pairs)))
        for nm in list(SHIFTS) + list(ARMS):
            print('[ha] %-14s %+.4f | per-seed %s' % (nm, ha[nm], [round(x, 4) for x in ha_seed[nm]]))

        # --- descriptive: how much compounds actually change spread, and is the size right -----------------------
        frac_up = float(np.mean(R_true > 1))
        ratio_abs = {a: float(np.mean([np.nanmean(S / S_true) for S in S_pred[a]])) for a in ARMS}
        print('[spread] R_true mean %.4f median %.4f sd %.4f | %.1f %% of conditions more varied than control'
              % (R_true.mean(), np.median(R_true), R_true.std(), 100 * frac_up))
        for a in ARMS:
            print('[spread] %-8s predicted/measured spread %.4f | R_pred median %.4f'
                  % (a, ratio_abs[a], float(np.median(np.mean(R_pred[a], axis=0)))))

        # --- endpoints (prereg section 4) -------------------------------------------------------------------------
        rng = np.random.RandomState(0)
        gs = [np.where(drugs == u)[0] for u in np.unique(drugs)]
        BH = {nm: np.full(N_BOOT, np.nan) for nm in R_pred}
        BC = np.full(N_BOOT, np.nan)
        for b in range(N_BOOT):
            sel = np.concatenate([gs[g] for g in rng.randint(0, len(gs), len(gs))])
            got = ha_all(sel)
            for nm in R_pred:
                BH[nm][b] = got[nm]
            BC[b] = got['ceiling']
        con = {}
        con['Q1_primary_ha_ot_pair_minus_0'] = s16.verdict_of(ha[NEW], BH[NEW])
        con['Q2_coprimary_ha_ot_pair_minus_recipe'] = s16.verdict_of(ha[NEW] - ha[OLD], BH[NEW] - BH[OLD])
        for b in SHIFTS:
            if b == 'control_paste':
                continue                                            # identically 0 by construction; asserted above
            con['Q3_gate_ha_%s_minus_0' % b] = s16.verdict_of(ha[b], BH[b])
        con['secondary_ha_recipe_minus_0'] = s16.verdict_of(ha[OLD], BH[OLD])
        if np.isfinite(ha_ceil):
            con['secondary_ha_ceiling_minus_0'] = s16.verdict_of(ha_ceil, BC)
        for key, c in con.items():
            print('%-42s %s' % (key, s16.fmt(c)))
        print('[bootstrap] HA: %d replicates (%d s)' % (N_BOOT, time.time() - T0))

        # --- exploratory (prereg section 4) -----------------------------------------------------------------------
        learn = pc['learnability'].values.astype(float)
        rho_l, p_l = spearmanr(learn, R_true)
        # measured response size: the length of the condition's real mean shift away from its vehicle control, taken
        # from the arrays (Y = true_mean - C) rather than from a column of the Step 15b record, which does not carry it
        resp = np.linalg.norm(Y, axis=1)
        rho_s, p_s = spearmanr(resp, R_true)
        print('exploratory  R_true vs learnability      Spearman %+.4f (p %.2e)' % (rho_l, p_l))
        print('exploratory  R_true vs measured response Spearman %+.4f (p %.2e)' % (rho_s, p_s))

        metrics = {nm: dict(ha=ha[nm], ha_per_seed=ha_seed[nm]) for nm in R_pred}
        for a in ARMS:
            metrics[a]['pred_over_measured_spread'] = ratio_abs[a]
            metrics[a]['tracking'] = float(np.nanmean(np.nanmean(np.vstack(track[a]), axis=0)))
            metrics[a]['tracking_null'] = float(np.nanmean(np.nanmean(np.vstack(track_null[a]), axis=0)))
        results[split] = dict(n_conditions=n, n_structures=n_drugs, n_contexts=n_ctx, min_context=MIN_CTX,
                              metrics=metrics, clip_variation=clip_var,
                              replicate=dict(ha=ha_ceil if np.isfinite(ha_ceil) else None,
                                             ha_n_contexts=n_ctx_rep, n_pairs=len(pairs)),
                              r_true=dict(mean=float(R_true.mean()), median=float(np.median(R_true)),
                                          sd=float(R_true.std()), frac_above_one=frac_up),
                              exploratory=dict(r_true_vs_learnability=[float(rho_l), float(p_l)],
                                               r_true_vs_response_size=[float(rho_s), float(p_s)]),
                              contrasts=con)
        inputs[split] = inp
        for nm in R_pred:
            rows.append(dict(split=split, level='metric', key='ha', submission=nm, value=ha[nm], ci_lo='', ci_hi='', verdict=''))
        for key, c in con.items():
            rows.append(dict(split=split, level='contrast', key=key, submission='', value=c['diff'],
                             ci_lo=c['ci'][0], ci_hi=c['ci'][1], verdict=c['verdict']))
        for i in range(n):
            r = dict(split=split, condition=cond[i], line=lines[i], plate=plates[i], structure=drugs[i],
                     dose_nM=doses[i], learnability=learn[i], response_size=resp[i], spread_true=S_true[i],
                     spread_ctrl=S_ctrl[i], R_true=R_true[i])
            for nm in R_pred:
                r['%s_R_pred' % nm] = float(np.mean([x[i] for x in R_pred[nm]]))
            pc_rows.append(r)

    # --- verdict (prereg section 6), only when all three splits are present ---------------------------------------
    verdict = None
    if final:
        q1 = {s: results[s]['contrasts']['Q1_primary_ha_ot_pair_minus_0']['verdict'] for s in splits}
        q2 = {s: results[s]['contrasts']['Q2_coprimary_ha_ot_pair_minus_recipe']['verdict'] for s in splits}
        q3 = {s: {k: v['verdict'] for k, v in results[s]['contrasts'].items() if k.startswith('Q3_')} for s in splits}
        above1 = [v == 'above zero' for v in q1.values()]
        if all(above1) and all(v == 'above zero' for v in q2.values()):
            verdict = '1 SPREAD PREDICTED, OT BETTER'
        elif all(above1):
            verdict = '2 SPREAD PREDICTED'
        elif all(v == 'includes zero' for v in q1.values()):
            verdict = '5 UNRESOLVED'
        elif any(v == 'below zero' for v in q1.values()):
            verdict = '3 REVERSED'
        elif sum(above1) in (1, 2):
            verdict = '4 PARTIAL'
        else:
            sys.exit('FATAL: the Q1 verdicts %s match no category of prereg section 6; refusing to guess' % q1)
        shifts_live = [(s, k) for s in splits for k, v in q3[s].items() if v == 'above zero']
        print('\n' + '=' * 72)
        print('PRE-REGISTERED VERDICT (section 6): %s' % verdict)
        print('  Q1 ha ot_pair - 0 per split:        %s' % q1)
        print('  Q2 ha ot_pair - recipe per split:   %s' % q2)
        if shifts_live:
            print('  NOTE: the linear references are NOT inert on this measure (%s); prereg section 6 requires the'
                  ' section-1 framing to be withdrawn in the findings.' % shifts_live)
        else:
            print('  Q3: no linear reference resolves above zero; the section-1 framing stands.')
        print('  Section 7 expected: 2 SPREAD PREDICTED, with Q2 including zero.')
        print('=' * 72)

    out = dict(prereg=PRE, final=final, verdict=verdict, splits=splits, interval='97.5% structure-clustered percentile bootstrap',
               n_boot=N_BOOT, n_seeds=s15.N_SEEDS, min_context=MIN_CTX, per_split=results, inputs=inputs,
               records=dict(step15b=REC15B, step15c=REC15C, step16=REC16, step18=REC18, data_record=s15.RECORD),
               commit=commit, seconds=round(time.time() - T0, 1))
    json.dump(out, open(OUT, 'w'), indent=1)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'level', 'key', 'submission', 'value', 'ci_lo', 'ci_hi', 'verdict'])
        w.writeheader()
        w.writerows(rows)
    pd.DataFrame(pc_rows).to_csv(OUT_PC, index=False)
    print('\nwrote %s, %s and %s  (%d s)' % (OUT, OUT_CSV, OUT_PC, time.time() - T0))


if __name__ == '__main__':
    main()
