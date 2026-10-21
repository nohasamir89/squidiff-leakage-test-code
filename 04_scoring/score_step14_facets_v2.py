#!/usr/bin/env python3
"""Step 14, facets v2: the six VCC-style facets recomputed under Amendment 2 (PREREG_step14_amendment2_facets_v2.md).

Everything the pre-registered question Q1 gates - the DE-overlap and energy contrasts of the shift baselines
against the unguided 50 NFE model, their verdict, and the thesis's three metrics - stays in the first run's
record (leakage_results/step14_analysis.json, commit c354314) and is NOT recomputed here. This script recomputes
only the descriptive facets of Q2, with the five departures from the cell-eval2 procedure that the audit of
2026-09-23 found corrected:
  1. Wilcoxon p-values without a continuity correction (step14_metrics_v2._mannwhitney_p; scanpy's convention,
     reproduced to 1e-15 on synthetic data);
  2. the fold-change NMAE replicate anchor takes its gate and denominator from the FULL reference table
     (step14_metrics_v2.lfc_nmae_anchor);
  3. a facet whose replicate anchor does not lie on the better side of the baseline on a cell line has NO scaled
     value on that line (cell-eval2 refuses such a scale); it is reported as undefined with the reason;
  4. the 0 point of the scale is the challenge's "dispersed" emission of the mean response: the condition's paired
     vehicle cells resampled with replacement to the condition's cell count and scaled gene-wise, in normalised-count
     space, by the line's mean-response profile over its control profile (cell_eval2/baseline.py); the additive
     mean-response shift of the first run is kept as an ordinary submission;
  5. a context's overall is the mean of its six scaled facets and the split's overall the mean over its lines
     (cell-eval2's order); the shift baselines' cells are cast to float32 before the DE table (the control-paste
     reach artefact of the first run).
Inputs, matching, gate, ridge reproduction, replicate half-splits and every raw facet definition are those of
score_step14.py. Cross-checks against the first run: PDS and the expression-error ratio of every submission must
equal the first record to 1e-9 (no test statistic enters them); the change in every DE facet is printed.
Writes leakage_results/step14_facets_v2.json and leakage_results/step14_long_v2.csv. Every number is printed.
"""
import os, sys, json, subprocess, hashlib, time, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
import anndata as ad
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from Squidiff.scrna_datasets import Drug_dose_encoder
import step14_metrics_v2 as vm

RES = 'leakage_results'
DATA = os.environ.get('S14_DATA', '/scratch/ni2071/Diffusion/sciplex/datasets_corrected')
PRE, AMEND = 'PREREG_step14_shift_baselines_vcc_scoring.md', 'PREREG_step14_amendment2_facets_v2.md'
V1 = os.path.join(RES, 'step14_analysis.json')
OUT, OUT_CSV = os.path.join(RES, 'step14_facets_v2.json'), os.path.join(RES, 'step14_long_v2.csv')
SPLITS = {'drug_split_1': 37, 'drug_split_0': 37, 'drug_split_2': 38}
N_SEEDS, K, N_BOOT, ALPHA = 5, 50, 5000, 1.0
LO, HI = 1.25, 98.75
MIN_CELLS, CAP, N_REP = 20, 200, 5
COMPARATOR = 'fm_50'
HIGHER = dict(pds=True, mse=False, fidelity=True, reach=True, sig_jaccard=True, lfc_nmae=False)
ZERO_POINT = 'mean_response_dispersed'
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
for q in (V1, PRE, AMEND):
    if not os.path.isfile(q): sys.exit('FATAL: missing %s' % q)
CODE = 'step14_metrics_v2.py score_step14_facets_v2.py'
if subprocess.run('git diff --quiet -- %s %s %s' % (PRE, AMEND, CODE), shell=True).returncode != 0: sys.exit('FATAL: amendment or scoring code uncommitted')
if subprocess.run('git ls-files --error-unmatch %s %s %s' % (AMEND, V1, CODE), shell=True, capture_output=True).returncode != 0: sys.exit('FATAL: amendment, first record or scoring code not committed')
S1 = json.load(open(V1))
if S1.get('verdict') != 'PARTIAL' or S1.get('comparator') != COMPARATOR: sys.exit('FATAL: the first record is not the one this amendment refers to')
print('[amendment] committed; first record committed; scoring code committed')
rng = np.random.RandomState(0)
T0 = time.time()

def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): m.update(c)
    return m.hexdigest()

def jac(a, b):
    i = len(np.intersect1d(a, b)); return i / (len(a) + len(b) - i)

def thesis_per_cond(true, B, C):
    n = len(true); j = np.full(n, np.nan); d = np.full(n, np.nan)
    for i in range(n):
        td = true[i] - C[i]; p = B[i] - C[i]
        g = np.argsort(np.abs(td))[-K:]
        if np.abs(p).std() > 1e-12: j[i] = jac(g, np.argsort(np.abs(p))[-K:])
        if td[g].std() > 1e-12 and p[g].std() > 1e-12: d[i] = pearsonr(p[g], td[g])[0]
    return j, d

def verdict(lo, hi):
    return 'above zero' if lo > 0 else 'below zero' if hi < 0 else 'includes zero'

def dense(X):
    return np.asarray(X.todense() if hasattr(X, 'todense') else X, dtype=np.float32)

# ------------------------------------------------------------------ arm registry (identical to score_step14.py)
def registry(split):
    if split == 'drug_split_1':
        R = {'adaln_ddim_1000': ('sw_ddim_seed_%d_n1000', 'provenance/s6sw_sw_ddim_seed_%d_n1000.json', dict(arm='ddim', steps=1000)),
             'fm_20': ('sw_fm_gaussian_seed_%d_n20', 'provenance/s6sw_sw_fm_gaussian_seed_%d_n20.json', dict(arm='fm_gaussian', steps=20)),
             'fm_50': ('sw_fm_gaussian_seed_%d_n50', 'provenance/s6sw_sw_fm_gaussian_seed_%d_n50.json', dict(arm='fm_gaussian', steps=50)),
             'guidance_w2': ('s12_cfg_w2_seed_%d', 'provenance/s12ev_s12_cfg_w2_seed_%d.json', dict(guidance_w=2.0)),
             'attn_cross': ('s13_drug_split_1_attn_cross_seed_%d', 'provenance/s13ev_s13_drug_split_1_attn_cross_seed_%d.json', dict(arm='attn_cross')),
             'mlp_ctrl': ('s13_drug_split_1_mlp_ctrl_seed_%d', 'provenance/s13ev_s13_drug_split_1_mlp_ctrl_seed_%d.json', dict(arm='mlp_ctrl'))}
        extras = {'s9_pathway': ('s9_pathway_seed_%d', 'provenance/s9ev_s9_pathway_seed_%d.json', {}),
                  's9_both': ('s9_both_seed_%d', 'provenance/s9ev_s9_both_seed_%d.json', {}),
                  's10_chemvae': ('s10_chemvae_seed_%d', 'provenance/s10ev_s10_chemvae_seed_%d.json', {}),
                  's10_fp_chemvae': ('s10_fp_chemvae_seed_%d', 'provenance/s10ev_s10_fp_chemvae_seed_%d.json', {})}
        base = ('step5_base_seed_%d', 'provenance/s5eval_step5_base_seed_%d.json', dict(arm='base'))
    else:
        s = split
        R = {'adaln_ddim_1000': ('s6b_%s_ddim_seed_%%d' % s, 'provenance/s6bev_s6b_%s_ddim_seed_%%d.json' % s, dict(arm='ddim', split=s)),
             'fm_20': ('s6b_%s_fm_gaussian_seed_%%d' % s, 'provenance/s6bev_s6b_%s_fm_gaussian_seed_%%d.json' % s, dict(arm='fm_gaussian', split=s, nfe=20)),
             'fm_50': ('s11_%s_fm_gaussian_nfe50_seed_%%d' % s, 'provenance/s11ev_s11_%s_fm_gaussian_nfe50_seed_%%d.json' % s, dict(split=s, nfe=50)),
             'guidance_w2': ('s12b_%s_w2_seed_%%d' % s, 'provenance/s12bev_s12b_%s_w2_seed_%%d.json' % s, dict(split=s, guidance_w=2.0)),
             'attn_cross': ('s13b_%s_attn_cross_seed_%%d' % s, 'provenance/s13bev_s13b_%s_attn_cross_seed_%%d.json' % s, dict(arm='attn_cross', split=s)),
             'mlp_ctrl': ('s13b_%s_mlp_ctrl_seed_%%d' % s, 'provenance/s13bev_s13b_%s_mlp_ctrl_seed_%%d.json' % s, dict(arm='mlp_ctrl', split=s))}
        extras = {}
        base = ('repl_%s_base_seed_%%d' % s, 'provenance/repleval_repl_%s_base_seed_%%d.json' % s, dict(arm='base', split=s))
    return R, extras, base

def check_prov(pv, name, checks):
    for k, v in checks.items():
        if k not in pv: print('  [note] %s: provenance has no key %r (expected %r)' % (name, k, v)); continue
        got = pv[k]
        ok = (abs(float(got) - float(v)) < 1e-9) if isinstance(v, (int, float)) else (str(got) == str(v))
        if not ok: sys.exit('FATAL: %s provenance %s=%r, expected %r' % (name, k, got, v))

def load_vectors(name, pj, checks, need_pc=True):
    f = os.path.join(RES, 'vectors_%s.npz' % name); fpc = os.path.join(RES, 'vectors_%s_pc.npz' % name)
    for q in ([f, pj] + ([fpc] if need_pc else [])):
        if not os.path.isfile(q): sys.exit('FATAL: missing %s' % q)
    pv = json.load(open(pj)); check_prov(pv, name, checks)
    if pv.get('vectors_sha256') != sha(f): sys.exit('FATAL: %s vectors hash differs from provenance' % name)
    if need_pc and pv.get('per_cell_sha256') != sha(fpc): sys.exit('FATAL: %s per-cell hash differs from provenance' % name)
    v = np.load(f, allow_pickle=True)
    out = dict(cond=np.array([str(x) for x in v['cond']]), true=np.asarray(v['true'], np.float64), B=np.asarray(v['B'], np.float64), C=np.asarray(v['C'], np.float64),
               rec=dict(tag=name, vectors_sha256=pv.get('vectors_sha256'), per_cell_sha256=pv.get('per_cell_sha256') if need_pc else None))
    if need_pc:
        p = np.load(fpc, allow_pickle=True)
        out['pred'] = [np.asarray(a, np.float32) for a in p['pred']]; out['true_cells'] = [np.asarray(a, np.float32) for a in p['true']]
        if len(out['pred']) != len(out['cond']): sys.exit('FATAL: %s per-cell length differs from vectors' % name)
    return out

# ------------------------------------------------------------------ facets of one submission
def de_facets(sub_tables, ref_tables, gene_names):
    out = {k: np.full(len(ref_tables), np.nan) for k in ('fidelity', 'reach', 'sig_jaccard', 'lfc_nmae')}
    for i, (r, s) in enumerate(zip(ref_tables, sub_tables)):
        if r is None or s is None: continue
        f = vm.direction_fidelity(r, s); out['fidelity'][i] = np.nan if f is None else f
        f = vm.direction_reach(r, s, gene_names); out['reach'][i] = np.nan if f is None else f
        out['sig_jaccard'][i] = vm.sig_jaccard(r, s)
        f = vm.lfc_nmae(r, s); out['lfc_nmae'][i] = np.nan if f is None else f
    return out

def profile_facets(B, true, C, lines):
    pds = np.full(len(B), np.nan); mse = {}
    for ln in sorted(set(lines)):
        ix = np.where(lines == ln)[0]
        pds[ix] = vm.pds_cosine(B[ix] - C[ix], true[ix] - C[ix])
        num = float(((B[ix] - true[ix]) ** 2).sum()); den = float(((true[ix] - C[ix]) ** 2).sum())
        mse[ln] = num / den if den > 0 else np.nan
    return pds, mse

def boot(x, drugs, n_drugs, stat=np.nanmean):
    ix = np.where(np.isfinite(x))[0]
    gs = [ix[drugs[ix] == u] for u in np.unique(drugs[ix])]
    if len(gs) != n_drugs: sys.exit('FATAL: %d drug clusters, expected %d' % (len(gs), n_drugs))
    o = np.array([stat(x[np.concatenate([gs[k] for k in rng.randint(0, len(gs), len(gs))])]) for _ in range(N_BOOT)])
    return float(np.percentile(o, LO)), float(np.percentile(o, HI))

def contrast(x_b, x_a, drugs, n_drugs):
    diff = x_b - x_a
    if np.isfinite(diff).sum() < len(diff) // 2:
        return dict(diff=None, ci=None, verdict='undefined (metric not defined for this submission)', half_width=None)
    m = float(np.nanmean(diff)); lo, hi = boot(diff, drugs, n_drugs)
    return dict(diff=m, ci=[lo, hi], verdict=verdict(lo, hi), half_width=float((hi - lo) / 2))

def scaled(u, b, r, higher):
    """(u - b) / (r - b) with cell-eval2's refusal: the anchor must lie on the better side of the baseline."""
    if any(v is None or not np.isfinite(v) for v in (u, b, r)): return np.nan, 'undefined: a value is missing'
    if (r <= b) if higher else (r >= b): return np.nan, 'degenerate: replicate %.4f not on the better side of the baseline %.4f' % (r, b)
    return float((u - b) / (r - b)), 'ok'

# ------------------------------------------------------------------ per split
results, inputs, gate, rows = {}, {}, {}, []
for split, n_drugs in SPLITS.items():
    print('\n' + '#' * 72 + '\n# %s\n' % split + '#' * 72)
    R, extras, base = registry(split)
    arms = {}
    for name, (vec, pj, checks) in list(R.items()) + list(extras.items()):
        arms[name] = [load_vectors(vec % s, pj % s, checks) for s in range(N_SEEDS)]
    base_v = [load_vectors(base[0] % s, base[1] % s, base[2], need_pc=False) for s in range(N_SEEDS)]
    cond = arms[COMPARATOR][0]['cond']; n = len(cond)
    for name, L in list(arms.items()) + [('base', base_v)]:
        for v in L:
            if not np.array_equal(v['cond'], cond): sys.exit('FATAL: condition order differs in %s' % v['rec']['tag'])
            if np.abs(v['C'] - arms[COMPARATOR][0]['C']).max() > 1e-5: sys.exit('FATAL: paired-control means differ in %s' % v['rec']['tag'])
    inputs[split] = {name: [v['rec'] for v in L] for name, L in list(arms.items()) + [('base', base_v)]}
    # the inputs must be the first run's inputs, hash for hash
    for name, L in inputs[split].items():
        prev = S1['inputs'][split].get(name)
        if prev is None or [r['vectors_sha256'] for r in prev] != [r['vectors_sha256'] for r in L]: sys.exit('FATAL: %s inputs differ from the first record on %s' % (name, split))
    true_mean = arms[COMPARATOR][0]['true']; C = arms[COMPARATOR][0]['C']
    for name, L in arms.items():
        for v in L:
            if np.abs(v['true'] - true_mean).max() > 1e-5: sys.exit('FATAL: true means differ in %s' % v['rec']['tag'])

    # --- data and the harness draw, matched with repeats allowed (as score_step14.py)
    te = ad.read_h5ad('%s/sci_plex_test_%s.h5ad' % (DATA, split)); tc = ad.read_h5ad('%s/sci_plex_test_%s_control.h5ad' % (DATA, split))
    if not np.array_equal(te.obs_names, tc.obs_names): sys.exit('FATAL: test and control files are not row-aligned')
    genes = np.array(te.var_names); Xte = dense(te.X); Xtc = dense(tc.X)
    g_all = te.obs['Group'].astype(str).values; smi_all = te.obs['SMILES'].astype(str).values
    dose_all = te.obs['dose'].astype(float).values; line_all = te.obs['cell_name'].astype(str).values
    drugs = np.array([smi_all[g_all == c][0] for c in cond]); lines = np.array([line_all[g_all == c][0] for c in cond])
    if len(np.unique(drugs)) != n_drugs: sys.exit('FATAL: %d drugs, expected %d' % (len(np.unique(drugs)), n_drugs))
    real = []; paired = []; rs = np.random.RandomState(0); recovered = exact = 0
    for i, c in enumerate(cond):
        ix = np.where(g_all == c)[0]
        stored = arms[COMPARATOR][0]['true_cells'][i]
        lookup = {Xte[j].tobytes(): j for j in ix}
        hit = [lookup.get(np.asarray(r, np.float32).tobytes()) for r in stored]
        if all(h is not None for h in hit):
            sel = np.array(sorted(set(hit))); recovered += 1; exact += 1
        else:
            D = cdist(np.asarray(stored, np.float64), Xte[ix].astype(np.float64), metric='chebyshev')
            if (D.min(axis=1) < 1e-5).all(): sel = np.array(sorted(set(ix[np.argmin(D, axis=1)].tolist()))); recovered += 1
            else: sel = np.sort(rs.choice(ix, CAP, replace=False)) if len(ix) > CAP else ix
        real.append(Xte[sel]); paired.append(Xtc[sel])
    n_real = np.array([len(a) for a in real]); gated = n_real >= MIN_CELLS
    g1 = S1['per_split'][split]['gate']
    if int(gated.sum()) != g1['n_gated'] or int(recovered) != g1['conditions_with_harness_draw_recovered']: sys.exit('FATAL: gate or matching differs from the first record on %s' % split)
    gate[split] = dict(n_conditions=int(n), n_gated=int(gated.sum()), recovered=int(recovered), recovered_exact=int(exact))
    print('[design] %d conditions | %d drugs | DE gate >= %d distinct cells: %d pass | harness draw recovered in %d conditions (%d exact) | as the first record' % (n, n_drugs, MIN_CELLS, gated.sum(), recovered, exact))
    ref_tab = [vm.de_table(real[i], paired[i]) if gated[i] else None for i in range(n)]

    # --- the ridges, exactly as score_step14.py (Step 7 reproduction re-asserted)
    FP = np.array([Drug_dose_encoder([smi_all[g_all == c][0]], [float(np.median(dose_all[g_all == c]))], comb_num=1)[0] for c in cond], dtype=np.float64)
    lv = sorted(set(lines)); Lmat = np.array([[float(l == u) for u in lv] for l in lines])
    Xfeat = np.hstack([FP, Lmat, np.concatenate([FP * Lmat[:, [k]] for k in range(len(lv))], axis=1)])
    Y = true_mean - C
    v7 = ('vectors_step6_fm_gaussian_seed_0.npz', 'provenance/s6eval_step6_fm_gaussian_seed_0.json') if split == 'drug_split_1' \
         else ('vectors_s6b_%s_fm_gaussian_seed_0.npz' % split, 'provenance/s6bev_s6b_%s_fm_gaussian_seed_0.json' % split)
    f7, e7 = os.path.join(RES, v7[0]), v7[1]
    if json.load(open(e7)).get('vectors_sha256') != sha(f7): sys.exit('FATAL: %s changed since Step 7' % f7)
    with np.load(f7, allow_pickle=True) as v:
        Y7 = np.asarray(v['true']) - np.asarray(v['C']); cond7 = np.array([str(x) for x in v['cond']])
    if not np.array_equal(cond7, cond): sys.exit('FATAL: Step 7 target condition order differs on %s' % split)
    FP7 = np.array([Drug_dose_encoder([smi_all[g_all == c][0]], [float(np.median(dose_all[g_all == c]))], comb_num=1)[0] for c in cond])
    X7 = np.hstack([FP7, Lmat, np.concatenate([FP7 * Lmat[:, [k]] for k in range(len(lv))], axis=1)])
    P7 = np.zeros_like(Y7)
    for a_, b_ in GroupKFold(n_splits=5).split(X7, Y7, groups=drugs):
        P7[b_] = Ridge(alpha=ALPHA).fit(X7[a_], Y7[a_]).predict(X7[b_])
    d7 = np.full(n, np.nan)
    for i in range(n):
        d = np.argsort(np.abs(Y7[i]))[-K:]
        if Y7[i][d].std() > 1e-12 and P7[i][d].std() > 1e-12: d7[i] = pearsonr(P7[i][d], Y7[i][d])[0]
    P7 = P7.astype(np.float64)
    rec_med = float(json.load(open(os.path.join(RES, 'step7_interaction_baseline.json')))['per_split'][split]['marginal']['interaction'])
    if abs(float(np.nanmedian(d7)) - rec_med) > 1e-9: sys.exit('FATAL: ridge-S7 DPC-50 median %.9f differs from the Step 7 record %.9f' % (np.nanmedian(d7), rec_med))
    print('[ridge-S7] reproduced: marginal median DPC-50 %.6f (Step 7 record %.6f)' % (np.nanmedian(d7), rec_med))
    tr = ad.read_h5ad('%s/sci_plex_train_%s.h5ad' % (DATA, split)); trc = ad.read_h5ad('%s/sci_plex_train_%s_control.h5ad' % (DATA, split))
    if not np.array_equal(tr.obs_names, trc.obs_names): sys.exit('FATAL: train and control files are not row-aligned')
    gtr = tr.obs['Group'].astype(str).values; utr = np.unique(gtr); Xtr = dense(tr.X); Xtrc = dense(trc.X)
    idx = {c: np.where(gtr == c)[0] for c in utr}
    Ytr = np.array([Xtr[idx[c]].mean(axis=0) - Xtrc[idx[c]].mean(axis=0) for c in utr], dtype=np.float64)
    smi_tr = tr.obs['SMILES'].astype(str).values; dose_tr = tr.obs['dose'].astype(float).values; line_tr = tr.obs['cell_name'].astype(str).values
    if set(smi_tr) & set(drugs): sys.exit('FATAL: a held-out compound appears in the training file')
    FPtr = np.array([Drug_dose_encoder([smi_tr[idx[c]][0]], [float(np.median(dose_tr[idx[c]]))], comb_num=1)[0] for c in utr], dtype=np.float64)
    Ltr = np.array([[float(line_tr[idx[c]][0] == u) for u in lv] for c in utr])
    Xtrf = np.hstack([FPtr, Ltr, np.concatenate([FPtr * Ltr[:, [k]] for k in range(len(lv))], axis=1)])
    Ptr = Ridge(alpha=ALPHA).fit(Xtrf, Ytr).predict(Xfeat)
    del tr, trc, Xtr, Xtrc
    deltas = {'ridge_S7': P7, 'ridge_train': Ptr, 'control_paste': np.zeros_like(Y)}
    mr = np.zeros_like(Y)
    for ln in lv:
        ix = np.where(lines == ln)[0]; mr[ix] = Y[ix].mean(axis=0)[None, :]
    deltas['mean_response'] = mr

    # --- submissions: cells per condition and the profile B. Shift baselines as the first run (cells cast to float32,
    #     the dtype fix); the dispersed mean response as cell_eval2/baseline.py emits it
    subs = {}
    for bname, D in deltas.items():
        cells = [np.clip(paired[i] + D[i][None, :], 0, None).astype(np.float32) for i in range(n)]
        j, d = thesis_per_cond(true_mean, C + D, C)
        subs[bname] = [dict(B=C + D, cells=cells, jac=j, dpc=d)]
    emit_rng = np.random.default_rng(0); B_disp = np.zeros_like(Y); cells_disp = []; emit_diag = {}
    for ln in lv:
        ix = np.where(lines == ln)[0]
        prof = np.mean([np.expm1(real[i].astype(np.float64)).mean(axis=0) for i in ix], axis=0)       # the line's mean response, normalised-count space, equal weight per condition
        ctrl = np.mean([np.expm1(paired[i].astype(np.float64)).mean(axis=0) for i in ix], axis=0)     # the line's control profile from the same paired vehicle cells
        zero = ctrl <= 0
        scale = np.where(zero, 0.0, prof / np.where(zero, 1.0, ctrl))
        emit_diag[ln] = dict(n_genes_control_zero=int(zero.sum()), r_max=float(scale.max()), r_median_nonzero=float(np.median(scale[scale > 0])) if (scale > 0).any() else 0.0)
        for i in ix:
            draw = emit_rng.integers(0, len(paired[i]), size=len(paired[i]))
            cells_disp.append((i, np.log1p(np.expm1(paired[i][draw].astype(np.float64)) * scale[None, :]).astype(np.float32)))
    cells_disp = [c for _, c in sorted(cells_disp, key=lambda t: t[0])]
    for i in range(n): B_disp[i] = cells_disp[i].astype(np.float64).mean(axis=0) - C[i]
    j, d = thesis_per_cond(true_mean, C + B_disp, C)
    subs[ZERO_POINT] = [dict(B=C + B_disp, cells=cells_disp, jac=j, dpc=d)]
    print('[dispersed] mean-response emission per line: %s' % json.dumps({ln: {k: round(v, 4) for k, v in dd.items()} for ln, dd in emit_diag.items()}))
    for name, L in arms.items():
        subs[name] = [dict(B=v['B'], cells=v['pred'], **dict(zip(('jac', 'dpc'), thesis_per_cond(v['true'], v['B'], v['C'])))) for v in L]
    subs['base'] = [dict(B=v['B'], cells=None, **dict(zip(('jac', 'dpc'), thesis_per_cond(v['true'], v['B'], v['C'])))) for v in base_v]
    for name in subs:
        print('[thesis] %-24s DE-overlap %.4f  DPC-50 mean %.4f  median %.4f' % (name, np.mean([np.nanmean(x['jac']) for x in subs[name]]),
              np.mean([np.nanmean(x['dpc']) for x in subs[name]]), np.median(np.nanmean(np.vstack([x['dpc'] for x in subs[name]]), axis=0))))

    # --- raw facets per submission, per line (seeds averaged)
    def score_submission(S):
        vals = []
        for s_ in S:
            pds, mse = profile_facets(s_['B'], true_mean, C, lines)
            if s_['cells'] is not None:
                tabs = [vm.de_table(s_['cells'][i], paired[i]) if gated[i] else None for i in range(n)]
                de = de_facets(tabs, ref_tab, genes)
            else:
                de = {k: np.full(n, np.nan) for k in ('fidelity', 'reach', 'sig_jaccard', 'lfc_nmae')}
            per_line = {ln: dict(pds=float(np.nanmean(pds[np.where(lines == ln)[0]])), mse=float(mse[ln]),
                                 **{k: (float(np.nanmean(de[k][np.where(lines == ln)[0]])) if np.isfinite(de[k][np.where(lines == ln)[0]]).any() else np.nan) for k in de}) for ln in lv}
            vals.append(dict(per_line=per_line, per_cond=dict(pds=pds, **de)))
        return vals
    facet = {name: score_submission(S) for name, S in subs.items()}

    # --- replicate anchor: five seeded half-splits; the NMAE anchor from the full reference gate (correction 2)
    rep_vals = []
    for r in range(N_REP):
        rr = np.random.RandomState(r)
        halfA, halfB, vA, vB = [], [], [], []
        for i in range(n):
            perm = rr.permutation(len(real[i])); h = len(perm) // 2
            halfA.append(real[i][perm[:h]]); halfB.append(real[i][perm[h:2 * h]])
        for i in range(n):
            perm = rr.permutation(len(paired[i])); h = len(perm) // 2
            vA.append(paired[i][perm[:h]]); vB.append(paired[i][perm[h:2 * h]])
        profA = np.array([a.mean(axis=0) for a in halfA], np.float64); profB = np.array([b.mean(axis=0) for b in halfB], np.float64)
        cA = np.array([a.mean(axis=0) for a in vA], np.float64); cB = np.array([b.mean(axis=0) for b in vB], np.float64)
        pds = np.full(n, np.nan); mse = {}
        for ln in lv:
            ix = np.where(lines == ln)[0]
            pds[ix] = vm.pds_cosine(profA[ix] - cA[ix], profB[ix] - cB[ix])
            num = float(((profA[ix] - profB[ix]) ** 2).sum()); den = float(((profB[ix] - cB[ix]) ** 2).sum()); mse[ln] = num / den if den > 0 else np.nan
        tabB = [vm.de_table(halfB[i], vB[i]) if gated[i] else None for i in range(n)]
        tabA = [vm.de_table(halfA[i], vA[i]) if gated[i] else None for i in range(n)]
        de = de_facets(tabA, tabB, genes)
        nm = np.full(n, np.nan)
        for i in range(n):
            if gated[i]:
                f = vm.lfc_nmae_anchor(ref_tab[i], tabA[i], tabB[i]); nm[i] = np.nan if f is None else f
        de['lfc_nmae'] = nm
        rep_vals.append({ln: dict(pds=float(np.nanmean(pds[np.where(lines == ln)[0]])), mse=float(mse[ln]),
                                  **{k: (float(np.nanmean(de[k][np.where(lines == ln)[0]])) if np.isfinite(de[k][np.where(lines == ln)[0]]).any() else np.nan) for k in de}) for ln in lv})
    replicate = {ln: {f: float(np.mean([rv[ln][f] for rv in rep_vals])) for f in vm.FACETS} for ln in lv}
    print('[replicate] per line: %s' % json.dumps({ln: {f: round(replicate[ln][f], 4) for f in vm.FACETS} for ln in lv}))

    # --- raw and scaled per line; per-line overall; split = mean over lines (correction 5); refusal (correction 3)
    facets_out = {}
    bl = {ln: {f: float(np.nanmean([v['per_line'][ln][f] for v in facet[ZERO_POINT]])) for f in vm.FACETS} for ln in lv}
    for name in subs:
        raw = {ln: {f: float(np.nanmean([v['per_line'][ln][f] for v in facet[name]])) for f in vm.FACETS} for ln in lv}
        sc, status, overall_line = {}, {}, {}
        for ln in lv:
            sc[ln], status[ln] = {}, {}
            for f in vm.FACETS:
                s_, st = scaled(raw[ln][f], bl[ln][f], replicate[ln][f], HIGHER[f])
                if f == 'mse' and np.isfinite(s_): s_ = float(np.clip(s_, 0, 1))
                if f == 'lfc_nmae' and np.isfinite(s_): s_ = float(max(s_, -6.0))
                sc[ln][f], status[ln][f] = s_, st
            vals6 = [sc[ln][f] for f in vm.FACETS]
            overall_line[ln] = float(np.mean(vals6)) if all(np.isfinite(vals6)) else np.nan
        split_raw = {f: float(np.nanmean([raw[ln][f] for ln in lv])) for f in vm.FACETS}
        split_sc = {f: (float(np.nanmean([sc[ln][f] for ln in lv])) if any(np.isfinite(sc[ln][f]) for ln in lv) else np.nan) for f in vm.FACETS}
        fin_over = [overall_line[ln] for ln in lv if np.isfinite(overall_line[ln])]
        split_sc['overall'] = float(np.mean(fin_over)) if fin_over else np.nan
        split_sc['overall_lines'] = [ln for ln in lv if np.isfinite(overall_line[ln])]
        facets_out[name] = dict(raw_per_line=raw, scaled_per_line=sc, scale_status_per_line=status, overall_per_line=overall_line, raw=split_raw, scaled=split_sc)
        for ln in lv:
            for f in vm.FACETS:
                rows.append(dict(split=split, line=ln, submission=name, facet=f, kind='raw', value=raw[ln][f], scale_status=''))
                rows.append(dict(split=split, line=ln, submission=name, facet=f, kind='scaled', value=sc[ln][f], scale_status=status[ln][f]))
            rows.append(dict(split=split, line=ln, submission=name, facet='overall', kind='scaled', value=overall_line[ln], scale_status=''))
        print('[facets] %-24s raw  %s' % (name, '  '.join('%s %.4f' % (f, split_raw[f]) for f in vm.FACETS)))
        print('         %-24s scaled %s  overall %+.3f (lines %s)' % ('', '  '.join('%s %+.3f' % (f, split_sc[f]) for f in vm.FACETS), split_sc['overall'], ','.join(split_sc['overall_lines'])))
    for ln in lv:
        for f in vm.FACETS: rows.append(dict(split=split, line=ln, submission='replicate', facet=f, kind='raw', value=replicate[ln][f], scale_status=''))
    degen = {ln: {f: facets_out[COMPARATOR]['scale_status_per_line'][ln][f] for f in vm.FACETS if facets_out[COMPARATOR]['scale_status_per_line'][ln][f] != 'ok'} for ln in lv}
    print('[scale] baseline (dispersed mean response) per line: %s' % json.dumps({ln: {f: round(bl[ln][f], 4) for f in vm.FACETS} for ln in lv}))
    print('[scale] refused (no usable replicate scale): %s' % json.dumps(degen))

    # --- cross-checks against the first record and the facet contrasts of Amendment 2
    F1 = S1['per_split'][split]['facets']; delta_v1 = {}
    for name in subs:
        if name not in F1: continue
        for f in ('pds', 'mse'):
            for ln in lv:
                a, b = facets_out[name]['raw_per_line'][ln][f], F1[name]['raw_per_line'][ln][f]
                if (a == a) != (b == b) or (a == a and abs(a - b) > 1e-9): sys.exit('FATAL: %s %s on %s/%s differs from the first record (%r vs %r)' % (name, f, split, ln, a, b))
        for f in ('fidelity', 'reach', 'sig_jaccard', 'lfc_nmae'):
            dd = [abs(facets_out[name]['raw_per_line'][ln][f] - F1[name]['raw_per_line'][ln][f]) for ln in lv if np.isfinite(facets_out[name]['raw_per_line'][ln][f]) and np.isfinite(F1[name]['raw_per_line'][ln][f])]
            delta_v1.setdefault(f, {})[name] = float(max(dd)) if dd else None
    print('[vs first record] PDS and expression error equal to 1e-9 for every submission; max |change| in the DE facets per line: %s'
          % json.dumps({f: {k: (round(v, 4) if v is not None else None) for k, v in dd.items()} for f, dd in delta_v1.items()}))
    res = dict(gate=gate[split], facets=facets_out, replicate=replicate, baseline_per_line=bl, emission=emit_diag, facet_contrasts={}, vs_first_record_max_abs_change=delta_v1,
               thesis_metrics={name: dict(de_overlap=float(np.mean([np.nanmean(x['jac']) for x in subs[name]])), dpc_mean=float(np.mean([np.nanmean(x['dpc']) for x in subs[name]])),
                                          dpc_median=float(np.median(np.nanmean(np.vstack([x['dpc'] for x in subs[name]]), axis=0)))) for name in subs})
    fS7 = facet['ridge_S7'][0]['per_cond']; fFM = {k: np.nanmean(np.vstack([v['per_cond'][k] for v in facet[COMPARATOR]]), axis=0) for k in fS7}
    for f in ('pds', 'fidelity', 'reach', 'sig_jaccard', 'lfc_nmae'):
        c = contrast(fS7[f], fFM[f], drugs, n_drugs); res['facet_contrasts'][f] = c
        print('facet       %-12s ridge-S7 minus unguided 50 NFE          %s' % (f, ('%+.5f  [%+.5f, %+.5f] -> %s' % (c['diff'], *c['ci'], c['verdict'])) if c['diff'] is not None else c['verdict']))
    results[split] = res

# ------------------------------------------------------------------ write
import csv
commit = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
out = dict(version='facets_v2', amendment=AMEND, amendment_sha256=sha(AMEND), first_record=V1, first_record_sha256=sha(V1), zero_point=ZERO_POINT, comparator=COMPARATOR,
           interval='97.5% percentile per split, drug-clustered', n_boot=N_BOOT, min_cells=MIN_CELLS, n_replicate_splits=N_REP, per_split=results, inputs=inputs,
           metrics_module_sha256=sha('step14_metrics_v2.py'), commit=commit, seconds=int(time.time() - T0))
json.dump(out, open(OUT, 'w'), indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else (float(o) if isinstance(o, np.floating) else str(o)))
with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['split', 'line', 'submission', 'facet', 'kind', 'value', 'scale_status']); w.writeheader()
    for r in rows: w.writerow({k: ('' if (isinstance(r[k], float) and not np.isfinite(r[k])) else r[k]) for k in w.fieldnames})
print('\nwrote %s and %s  (%d s)' % (OUT, OUT_CSV, int(time.time() - T0)))
