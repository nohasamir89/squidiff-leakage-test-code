#!/usr/bin/env python3
"""Step 14 scoring under PREREG_step14_shift_baselines_vcc_scoring.md.

For each of the three compound splits:
  1. loads every generative arm with stored per-cell predictions (five seeds each), verifying every
     vectors / per-cell file against its provenance hash, plus base Squidiff's mean vectors;
  2. builds the four shift baselines - ridge-S7 (Step 7's within-held-out interaction ridge, reproduced
     with Step 7's procedure and asserted equal to the record), ridge-train (the same ridge fitted on the
     training split's condition deltas), mean-response (the challenge's 0 point) and control-paste - each
     as the condition's paired vehicle cells (cap 200, RandomState(0)) plus the delta, clipped at 0, with
     the profile-level vector C + delta. Amendment 1 of the pre-registration: every profile effect is taken
     against the condition's paired-control mean C, and every DE comparison group is the condition's paired
     vehicle cells - the origin the thesis's own metrics use - so the matched controls cannot leak identity;
  3. builds the replicate anchor (five seeded half-splits of the real cells and of the paired vehicle cells);
  4. scores the thesis's three metrics (DE-overlap Jaccard on the true top-50 |delta| genes, energy
     distance U-statistic, DPC-50) exactly as every earlier step, and the six VCC-style facets of
     step14_metrics.py for every submission, per cell line;
  5. computes the pre-registered contrasts (ridge-S7 minus unguided 50 NFE on DE-overlap and energy,
     97.5% drug-clustered percentile bootstrap, 5000 replicates, RandomState(0)) and the secondaries,
     applies section 6 mechanically across the three splits, and writes
     leakage_results/step14_analysis.json and leakage_results/step14_long.csv.
It stops if any input is missing, any hash differs from its record, or the ridge-S7 DPC-50 differs
from the Step 7 record. Every number it uses is printed.
"""
import os, sys, json, subprocess, hashlib, time, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from Squidiff.scrna_datasets import Drug_dose_encoder
import step14_metrics as vm

RES = 'leakage_results'
DATA = os.environ.get('S14_DATA', '/scratch/ni2071/Diffusion/sciplex/datasets_corrected')
PRE = 'PREREG_step14_shift_baselines_vcc_scoring.md'
OUT, OUT_CSV = os.path.join(RES, 'step14_analysis.json'), os.path.join(RES, 'step14_long.csv')
OUT_PC = os.path.join(RES, 'step14_per_condition.csv')   # Step 28: per-condition thesis metrics (additive)
SPLITS = {'drug_split_1': 37, 'drug_split_0': 37, 'drug_split_2': 38}          # held-out compounds per split
N_SEEDS, K, N_BOOT, ALPHA = 5, 50, 5000, 1.0
LO, HI = 1.25, 98.75                                                          # 97.5% per split
MIN_CELLS, CAP, N_REP = 20, 200, 5
COMPARATOR = 'fm_50'
BASELINES = ('ridge_S7', 'ridge_train', 'mean_response', 'control_paste')
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
if open(PRE).read().count('\n## ') < 9: sys.exit('FATAL: prereg incomplete')
if subprocess.run('git diff --quiet -- %s step14_metrics.py score_step14.py' % PRE, shell=True).returncode != 0: sys.exit('FATAL: prereg or scoring code uncommitted')
if subprocess.run('git ls-files --error-unmatch step14_metrics.py score_step14.py', shell=True, capture_output=True).returncode != 0: sys.exit('FATAL: scoring code not committed')
print('[prereg] committed; scoring code committed')
rng = np.random.RandomState(0)
T0 = time.time()

# ------------------------------------------------------------------ the thesis's metrics, verbatim from the Step 13 analyses
def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): m.update(c)
    return m.hexdigest()

def jac(a, b):
    i = len(np.intersect1d(a, b)); return i / (len(a) + len(b) - i)

def thesis_per_cond(true, B, C):
    """DE-overlap Jaccard and DPC-50 per condition from mean vectors (true, predicted B, paired control C)."""
    n = len(true); j = np.full(n, np.nan); d = np.full(n, np.nan)
    for i in range(n):
        td = true[i] - C[i]; p = B[i] - C[i]
        g = np.argsort(np.abs(td))[-K:]
        if np.abs(p).std() > 1e-12: j[i] = jac(g, np.argsort(np.abs(p))[-K:])
        if td[g].std() > 1e-12 and p[g].std() > 1e-12: d[i] = pearsonr(p[g], td[g])[0]
    return j, d

def energy(X, Y):
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    return 2 * cdist(X, Y).mean() - pdist(X).mean() - pdist(Y).mean()

def verdict(lo, hi):
    return 'above zero' if lo > 0 else 'below zero' if hi < 0 else 'includes zero'

def dense(X):
    return np.asarray(X.todense() if hasattr(X, 'todense') else X, dtype=np.float32)

# ------------------------------------------------------------------ arm registry: (vectors name, provenance path, checks) per split
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
    """A provenance key that exists must carry the expected value; keys absent from a record are reported, not assumed."""
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
               rec=dict(tag=name, vectors_sha256=pv.get('vectors_sha256'), per_cell_sha256=pv.get('per_cell_sha256') if need_pc else None, checkpoint_sha256=pv.get('checkpoint_sha256')))
    if need_pc:
        p = np.load(fpc, allow_pickle=True)
        out['pred'] = [np.asarray(a, np.float32) for a in p['pred']]; out['true_cells'] = [np.asarray(a, np.float32) for a in p['true']]
        if len(out['pred']) != len(out['cond']): sys.exit('FATAL: %s per-cell length differs from vectors' % name)
    return out

# ------------------------------------------------------------------ scoring of one submission (cells per condition + profile-level vectors)
def de_facets(sub_tables, ref_tables, gene_names):
    """Per-condition values of the four DE facets; NaN where the spec drops the condition."""
    out = {k: np.full(len(ref_tables), np.nan) for k in ('fidelity', 'reach', 'sig_jaccard', 'lfc_nmae')}
    for i, (r, s) in enumerate(zip(ref_tables, sub_tables)):
        if r is None or s is None: continue
        f = vm.direction_fidelity(r, s); out['fidelity'][i] = np.nan if f is None else f
        f = vm.direction_reach(r, s, gene_names); out['reach'][i] = np.nan if f is None else f
        out['sig_jaccard'][i] = vm.sig_jaccard(r, s)
        f = vm.lfc_nmae(r, s); out['lfc_nmae'][i] = np.nan if f is None else f
    return out

def profile_facets(B, true, C, lines):
    """PDS per condition (ranked within its cell line) and the MSE ratio per line, from profile-level vectors; effects
    are B - C and true - C (amendment 1), and the MSE denominator is sum ||true - C||^2."""
    pds = np.full(len(B), np.nan); mse = {}
    for ln in sorted(set(lines)):
        ix = np.where(lines == ln)[0]
        pds[ix] = vm.pds_cosine(B[ix] - C[ix], true[ix] - C[ix])
        num = float(((B[ix] - true[ix]) ** 2).sum()); den = float(((true[ix] - C[ix]) ** 2).sum())
        mse[ln] = num / den if den > 0 else np.nan
    return pds, mse

def boot(x, drugs, n_drugs, stat=np.nanmean):
    """Drug-clustered percentile bootstrap of stat over the finite entries of x (97.5%)."""
    ix = np.where(np.isfinite(x))[0]
    gs = [ix[drugs[ix] == u] for u in np.unique(drugs[ix])]
    if len(gs) != n_drugs: sys.exit('FATAL: %d drug clusters, expected %d' % (len(gs), n_drugs))
    o = np.array([stat(x[np.concatenate([gs[k] for k in rng.randint(0, len(gs), len(gs))])]) for _ in range(N_BOOT)])
    return float(np.percentile(o, LO)), float(np.percentile(o, HI))

def contrast(x_b, x_a, drugs, n_drugs):
    diff = x_b - x_a
    if np.isfinite(diff).sum() < len(diff) // 2:              # e.g. the zero-shift baseline has no top-50 set, so its DE-overlap is undefined
        return dict(diff=None, ci=None, verdict='undefined (metric not defined for this submission)', half_width=None)
    m = float(np.nanmean(diff)); lo, hi = boot(diff, drugs, n_drugs)
    return dict(diff=m, ci=[lo, hi], verdict=verdict(lo, hi), half_width=float((hi - lo) / 2))

# ------------------------------------------------------------------ per split
results, inputs, gate, rows, pc_rows = {}, {}, {}, [], []    # pc_rows: Step 28 per-condition record
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
    true_mean = arms[COMPARATOR][0]['true']; C = arms[COMPARATOR][0]['C']
    for name, L in arms.items():
        for v in L:
            if np.abs(v['true'] - true_mean).max() > 1e-5: sys.exit('FATAL: true means differ in %s' % v['rec']['tag'])

    # --- data: test cells, paired vehicle cells, vehicle pool per line, gene names
    te = ad.read_h5ad('%s/sci_plex_test_%s.h5ad' % (DATA, split)); tc = ad.read_h5ad('%s/sci_plex_test_%s_control.h5ad' % (DATA, split))
    if not np.array_equal(te.obs_names, tc.obs_names): sys.exit('FATAL: test and control files are not row-aligned')
    genes = np.array(te.var_names); Xte = dense(te.X); Xtc = dense(tc.X)
    g_all = te.obs['Group'].astype(str).values; smi_all = te.obs['SMILES'].astype(str).values
    dose_all = te.obs['dose'].astype(float).values; line_all = te.obs['cell_name'].astype(str).values
    drugs = np.array([smi_all[g_all == c][0] for c in cond]); lines = np.array([line_all[g_all == c][0] for c in cond])
    if len(np.unique(drugs)) != n_drugs: sys.exit('FATAL: %d drugs, expected %d' % (len(np.unique(drugs)), n_drugs))
    # real cells per condition and THEIR paired vehicles: the harness's own draw of real cells (stored per cell) is
    # matched back to the test file row by row, so the vehicles are exactly the cells the arms were conditioned on;
    # a condition whose stored cells cannot all be matched falls back to a seeded draw and is counted
    # The harness draws each condition's real cells WITH replacement (seeded, identical across arms), so its stored draw
    # can hold a cell more than once. Matching allows repeats; the DE facets and the cell gate use the DISTINCT matched
    # cells and their paired vehicles; energy distance keeps the harness's draw as stored, exactly as every arm was scored.
    real = []; paired = []; rs = np.random.RandomState(0); recovered = exact = tolerant = with_dup = 0; n_stored = []
    for i, c in enumerate(cond):
        ix = np.where(g_all == c)[0]
        stored = arms[COMPARATOR][0]['true_cells'][i]; n_stored.append(len(stored))
        lookup = {Xte[j].tobytes(): j for j in ix}
        hit = [lookup.get(np.asarray(r, np.float32).tobytes()) for r in stored]
        if all(h is not None for h in hit):
            sel = np.array(sorted(set(hit))); recovered += 1; exact += 1
        else:                                                    # nearest test row within 1e-5 (max-abs)
            D = cdist(np.asarray(stored, np.float64), Xte[ix].astype(np.float64), metric='chebyshev')
            if (D.min(axis=1) < 1e-5).all(): sel = np.array(sorted(set(ix[np.argmin(D, axis=1)].tolist()))); recovered += 1; tolerant += 1
            else: sel = np.sort(rs.choice(ix, CAP, replace=False)) if len(ix) > CAP else ix
        if len(sel) < len(stored): with_dup += 1
        real.append(Xte[sel]); paired.append(Xtc[sel])
    n_real = np.array([len(a) for a in real]); gated = n_real >= MIN_CELLS
    gate[split] = dict(n_conditions=int(n), n_gated=int(gated.sum()), distinct_cells_min=int(n_real.min()), distinct_cells_max=int(n_real.max()), stored_cells_min=int(min(n_stored)), stored_cells_max=int(max(n_stored)),
                       conditions_with_harness_draw_recovered=int(recovered), recovered_exact=int(exact), recovered_within_1e5=int(tolerant), conditions_with_duplicate_stored_cells=int(with_dup))
    print('[design] %d conditions | %d drugs | lines %s | stored cells per condition %d-%d, distinct %d-%d | DE gate >= %d distinct cells: %d pass | harness draw recovered in %d conditions (%d exact, %d within 1e-5), seeded draw for %d | conditions with duplicate stored cells %d'
          % (n, n_drugs, sorted(set(str(x) for x in lines)), min(n_stored), max(n_stored), n_real.min(), n_real.max(), MIN_CELLS, gated.sum(), recovered, exact, tolerant, n - recovered, with_dup))

    # --- reference DE tables (real cells vs the line's vehicle pool)
    ref_tab = [vm.de_table(real[i], paired[i]) if gated[i] else None for i in range(n)]

    # --- the ridges. Step 7's, reproduced; and the like-for-like ridge on the training split's condition deltas
    FP = np.array([Drug_dose_encoder([smi_all[g_all == c][0]], [float(np.median(dose_all[g_all == c]))], comb_num=1)[0] for c in cond], dtype=np.float64)
    lv = sorted(set(lines)); Lmat = np.array([[float(l == u) for u in lv] for l in lines])
    Xfeat = np.hstack([FP, Lmat, np.concatenate([FP * Lmat[:, [k]] for k in range(len(lv))], axis=1)])
    Y = true_mean - C
    # Step 7's target file (the same one analyse_step13 / analyse_step13b reproduced from), hash-checked; the ridge is
    # fitted on ITS true - C so the reproduction is exact, and its predictions are the ridge-S7 delta
    v7 = ('vectors_step6_fm_gaussian_seed_0.npz', 'provenance/s6eval_step6_fm_gaussian_seed_0.json') if split == 'drug_split_1' \
         else ('vectors_s6b_%s_fm_gaussian_seed_0.npz' % split, 'provenance/s6bev_s6b_%s_fm_gaussian_seed_0.json' % split)
    f7, e7 = os.path.join(RES, v7[0]), v7[1]
    for q in (f7, e7):
        if not os.path.isfile(q): sys.exit('FATAL: missing ' + q)
    if json.load(open(e7)).get('vectors_sha256') != sha(f7): sys.exit('FATAL: %s changed since Step 7' % f7)
    with np.load(f7, allow_pickle=True) as v:
        Y7 = np.asarray(v['true']) - np.asarray(v['C']); cond7 = np.array([str(x) for x in v['cond']])   # float32, as analyse_step13 computes it
    if not np.array_equal(cond7, cond): sys.exit('FATAL: Step 7 target condition order differs from the arms on %s' % split)
    FP7 = np.array([Drug_dose_encoder([smi_all[g_all == c][0]], [float(np.median(dose_all[g_all == c]))], comb_num=1)[0] for c in cond])
    X7 = np.hstack([FP7, Lmat, np.concatenate([FP7 * Lmat[:, [k]] for k in range(len(lv))], axis=1)])
    P7 = np.zeros_like(Y7)                                        # float32 store, as analyse_step13's ridge_pred
    for a_, b_ in GroupKFold(n_splits=5).split(X7, Y7, groups=drugs):
        P7[b_] = Ridge(alpha=ALPHA).fit(X7[a_], Y7[a_]).predict(X7[b_])
    d7 = np.full(n, np.nan)
    for i in range(n):
        d = np.argsort(np.abs(Y7[i]))[-K:]
        if Y7[i][d].std() > 1e-12 and P7[i][d].std() > 1e-12: d7[i] = pearsonr(P7[i][d], Y7[i][d])[0]
    P7 = P7.astype(np.float64); Y7 = Y7.astype(np.float64)
    s7 = json.load(open(os.path.join(RES, 'step7_interaction_baseline.json')))
    rec_med = float(s7['per_split'][split]['marginal']['interaction'])
    if abs(float(np.nanmedian(d7)) - rec_med) > 1e-9: sys.exit('FATAL: ridge-S7 DPC-50 median %.9f differs from the Step 7 record %.9f' % (np.nanmedian(d7), rec_med))
    print('[ridge-S7] reproduced from %s: marginal median DPC-50 %.6f (Step 7 record %.6f); max |true - C| difference between that file and the 50 NFE file %.2e'
          % (v7[0], np.nanmedian(d7), rec_med, np.abs(Y7 - Y).max()))
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
    print('[ridge-train] fitted on %d training conditions (%d compounds), predicting %d held-out conditions' % (len(utr), len(set(smi_tr)), n))
    del tr, trc, Xtr, Xtrc

    # --- shift baselines: profile vectors C + delta; cells = paired vehicles + delta, clipped at 0
    deltas = {'ridge_S7': P7, 'ridge_train': Ptr, 'control_paste': np.zeros_like(Y)}
    mr = np.zeros_like(Y)
    for ln in lv:
        ix = np.where(lines == ln)[0]; mr[ix] = Y[ix].mean(axis=0)[None, :]
    deltas['mean_response'] = mr
    subs = {}                                                  # name -> list over "seeds" of dict(B, cells, per-cond thesis metrics)
    real_for_energy = arms[COMPARATOR][0]['true_cells']
    for bname, D in deltas.items():
        cells = [np.clip(paired[i] + D[i][None, :], 0, None) for i in range(n)]
        clipped = float(np.mean([(paired[i] + D[i][None, :] < 0).mean() for i in range(n)]))
        j, d = thesis_per_cond(true_mean, C + D, C)
        en = np.array([energy(cells[i], real_for_energy[i]) for i in range(n)])
        subs[bname] = [dict(B=C + D, cells=cells, jac=j, dpc=d, en=en)]
        print('[baseline] %-14s DE-overlap %.4f  energy %.4f  DPC-50 mean %.4f  median %.4f  (entries clipped at 0: %.4f%%)' % (bname, np.nanmean(j), en.mean(), np.nanmean(d), np.nanmedian(d), 100 * clipped))
    for name, L in arms.items():
        subs[name] = []
        for v in L:
            j, d = thesis_per_cond(v['true'], v['B'], v['C']); en = np.array([energy(v['pred'][i], v['true_cells'][i]) for i in range(n)])
            subs[name].append(dict(B=v['B'], cells=v['pred'], jac=j, dpc=d, en=en))
        print('[arm] %-14s DE-overlap %.4f  energy %.4f  DPC-50 mean %.4f  median %.4f' % (name, np.mean([np.nanmean(x['jac']) for x in subs[name]]), np.mean([x['en'].mean() for x in subs[name]]),
              np.mean([np.nanmean(x['dpc']) for x in subs[name]]), np.median(np.nanmean(np.vstack([x['dpc'] for x in subs[name]]), axis=0))))
    subs['base'] = [dict(B=v['B'], cells=None, jac=thesis_per_cond(v['true'], v['B'], v['C'])[0], dpc=thesis_per_cond(v['true'], v['B'], v['C'])[1], en=np.full(n, np.nan)) for v in base_v]

    # --- VCC-style facets per submission: raw per-line values (mean over that line's conditions; MSE as the line ratio)
    def score_submission(S, cells_ok=True):
        vals = []
        for s_ in S:
            pds, mse = profile_facets(s_['B'], true_mean, C, lines)
            if cells_ok and s_['cells'] is not None:
                tabs = [vm.de_table(s_['cells'][i], paired[i]) if gated[i] else None for i in range(n)]
                de = de_facets(tabs, ref_tab, genes)
            else:
                de = {k: np.full(n, np.nan) for k in ('fidelity', 'reach', 'sig_jaccard', 'lfc_nmae')}
            per_line = {}
            for ln in lv:
                ix = np.where(lines == ln)[0]
                per_line[ln] = dict(pds=float(np.nanmean(pds[ix])), mse=float(mse[ln]), **{k: float(np.nanmean(de[k][ix])) if np.isfinite(de[k][ix]).any() else np.nan for k in de})
            vals.append(dict(per_line=per_line, per_cond=dict(pds=pds, **de)))
        return vals
    facet = {name: score_submission(S) for name, S in subs.items()}
    # --- replicate anchor: half-splits of the real cells and of the vehicle pool, five seeds
    rep_vals = []
    for r in range(N_REP):
        rr = np.random.RandomState(r)
        halfA, halfB = [], []
        for i in range(n):
            perm = rr.permutation(len(real[i])); h = len(perm) // 2
            halfA.append(real[i][perm[:h]]); halfB.append(real[i][perm[h:2 * h]])
        vA, vB = [], []
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
        rep_vals.append({ln: dict(pds=float(np.nanmean(pds[np.where(lines == ln)[0]])), mse=float(mse[ln]),
                                  **{k: float(np.nanmean(de[k][np.where(lines == ln)[0]])) for k in de}) for ln in lv})
    replicate = {ln: {f: float(np.mean([rv[ln][f] for rv in rep_vals])) for f in vm.FACETS} for ln in lv}
    print('[replicate] per line: %s' % json.dumps({ln: {f: round(replicate[ln][f], 4) for f in vm.FACETS} for ln in lv}))

    # --- raw and scaled facets; seeds averaged per submission
    facets_out = {}
    for name in subs:
        raw = {ln: {f: float(np.nanmean([v['per_line'][ln][f] for v in facet[name]])) for f in vm.FACETS} for ln in lv}
        bl = {ln: {f: float(np.nanmean([v['per_line'][ln][f] for v in facet['mean_response']])) for f in vm.FACETS} for ln in lv}
        sc = {}
        for ln in lv:
            sc[ln] = {}
            for f in vm.FACETS:
                s_ = vm.scaled(raw[ln][f], bl[ln][f], replicate[ln][f])
                if f == 'mse' and np.isfinite(s_): s_ = float(np.clip(s_, 0, 1))
                if f == 'lfc_nmae' and np.isfinite(s_): s_ = float(max(s_, -6.0))
                sc[ln][f] = s_
        split_raw = {f: float(np.nanmean([raw[ln][f] for ln in lv])) for f in vm.FACETS}
        split_sc = {f: float(np.nanmean([sc[ln][f] for ln in lv])) for f in vm.FACETS}
        split_sc['overall'] = float(np.mean([split_sc[f] for f in vm.FACETS]))      # NaN when any facet is undefined (base Squidiff)
        facets_out[name] = dict(raw_per_line=raw, scaled_per_line=sc, raw=split_raw, scaled=split_sc)
        for ln in lv:
            for f in vm.FACETS:
                rows.append(dict(split=split, line=ln, submission=name, facet=f, kind='raw', value=raw[ln][f])); rows.append(dict(split=split, line=ln, submission=name, facet=f, kind='scaled', value=sc[ln][f]))
        print('[facets] %-14s raw  %s' % (name, '  '.join('%s %.4f' % (f, split_raw[f]) for f in vm.FACETS)))
        print('         %-14s scaled %s  overall %.3f' % ('', '  '.join('%s %+.3f' % (f, split_sc[f]) for f in vm.FACETS), split_sc['overall']))
    for ln in lv:
        for f in vm.FACETS: rows.append(dict(split=split, line=ln, submission='replicate', facet=f, kind='raw', value=replicate[ln][f]))

    # --- the pre-registered contrasts (thesis metrics), seed-mean per condition for arms
    cm = lambda name, k: np.nanmean(np.vstack([x[k] for x in subs[name]]), axis=0)
    # Step 28 (salt-twin sensitivity): record each submission's seed-averaged thesis metrics per condition,
    # with the condition's cell line and structure (SMILES), so sci-Plex gene identity can be re-aggregated
    # with the parent-twin compounds removed. This only appends to pc_rows; nothing above is altered, and the
    # gate after re-running is that step14_long.csv stays byte-identical.
    for _nm in subs:
        _j, _e, _d = cm(_nm, 'jac'), cm(_nm, 'en'), cm(_nm, 'dpc')
        for _i in range(n):
            pc_rows.append(dict(split=split, condition=str(cond[_i]), line=lines[_i], structure=drugs[_i],
                                submission=_nm, de_overlap=float(_j[_i]),
                                energy=float(_e[_i]) if np.isfinite(_e[_i]) else None,
                                dpc50=float(_d[_i])))
    res = dict(gate=gate[split], ridge_S7_dpc_median=float(np.nanmedian(d7)), step7_record=rec_med, contrasts={}, facets=facets_out, replicate=replicate,
               thesis_metrics={name: dict(de_overlap=float(np.nanmean(cm(name, 'jac'))), energy=float(np.nanmean(cm(name, 'en'))) if np.isfinite(cm(name, 'en')).any() else None,
                                          dpc_mean=float(np.nanmean(cm(name, 'dpc'))), dpc_median=float(np.nanmedian(cm(name, 'dpc')))) for name in subs})
    pj = contrast(cm('ridge_S7', 'jac'), cm(COMPARATOR, 'jac'), drugs, n_drugs); pe = contrast(cm('ridge_S7', 'en'), cm(COMPARATOR, 'en'), drugs, n_drugs)
    res['contrasts']['primary_de_ridgeS7_minus_fm50'] = pj; res['contrasts']['coprimary_energy_ridgeS7_minus_fm50'] = pe
    print('\nPRIMARY     DE-overlap, ridge-S7 minus unguided 50 NFE   %+.5f  97.5%% CI [%+.5f, %+.5f] -> %s' % (pj['diff'], *pj['ci'], pj['verdict']))
    print('CO-PRIMARY  energy,     ridge-S7 minus unguided 50 NFE   %+.5f  97.5%% CI [%+.5f, %+.5f] -> %s' % (pe['diff'], *pe['ci'], pe['verdict']))
    def fmt(c):
        return '%+.5f  [%+.5f, %+.5f] -> %s' % (c['diff'], *c['ci'], c['verdict']) if c['diff'] is not None else c['verdict']
    for bname in ('ridge_train', 'mean_response', 'control_paste'):
        for k, lab in (('jac', 'DE-overlap'), ('en', 'energy')):
            c = contrast(cm(bname, k), cm(COMPARATOR, k), drugs, n_drugs); res['contrasts']['secondary_%s_%s_minus_fm50' % (k, bname)] = c
            print('secondary   %-10s %-13s minus unguided 50 NFE   %s' % (lab, bname, fmt(c)))
    for k, lab in (('jac', 'DE-overlap'), ('en', 'energy')):
        c = contrast(cm('ridge_S7', k), cm('guidance_w2', k), drugs, n_drugs); res['contrasts']['secondary_%s_ridgeS7_minus_guidance_w2' % k] = c
        print('secondary   %-10s ridge-S7 minus guidance w = 2           %s' % (lab, fmt(c)))
    # facet-level intervals for ridge-S7 minus fm_50 (per-condition facets; MSE by the difference of line-ratio means under compound resampling)
    res['facet_contrasts'] = {}
    fS7 = facet['ridge_S7'][0]['per_cond']; fFM = {k: np.nanmean(np.vstack([v['per_cond'][k] for v in facet[COMPARATOR]]), axis=0) for k in fS7}
    for f in ('pds', 'fidelity', 'reach', 'sig_jaccard', 'lfc_nmae'):
        c = contrast(fS7[f], fFM[f], drugs, n_drugs); res['facet_contrasts'][f] = c
        print('facet       %-12s ridge-S7 minus unguided 50 NFE          %s' % (f, fmt(c)))
    results[split] = res

# ------------------------------------------------------------------ verdict across the three splits (section 6)
de_v = [results[s]['contrasts']['primary_de_ridgeS7_minus_fm50']['verdict'] for s in SPLITS]
en_v = [results[s]['contrasts']['coprimary_energy_ridgeS7_minus_fm50']['verdict'] for s in SPLITS]
if all(v == 'above zero' for v in de_v) and all(v != 'above zero' for v in en_v): cat = 'LINEAR SHIFT AHEAD'
elif all(v == 'below zero' for v in de_v) and all(v == 'above zero' for v in en_v): cat = 'GENERATIVE ADVANTAGE'
elif (all(v == 'above zero' for v in de_v) and all(v == 'above zero' for v in en_v)) or (all(v == 'below zero' for v in de_v) and all(v != 'above zero' for v in en_v)): cat = 'TRADE'
elif all(v == 'includes zero' for v in de_v) and all(v == 'includes zero' for v in en_v): cat = 'UNRESOLVED'
else: cat = 'PARTIAL'
print('\n' + '=' * 72 + '\nPRE-REGISTERED VERDICT: %s\n  primary (DE-overlap) per split: %s\n  co-primary (energy) per split: %s\n' % (cat, dict(zip(SPLITS, de_v)), dict(zip(SPLITS, en_v))) +
      '  Section 7 expected TRADE: ridge shift above on DE-overlap (splits 0 and 1), worse on energy on all three.\n' + '=' * 72)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
pd.DataFrame(pc_rows).to_csv(OUT_PC, index=False)   # Step 28 per-condition record
json.dump(dict(prereg=PRE, verdict=cat, interval='97.5% percentile per split, drug-clustered', n_boot=N_BOOT, min_cells=MIN_CELLS, cap=CAP, n_replicate_splits=N_REP,
               comparator=COMPARATOR, per_split=results, inputs=inputs, metrics_module_sha256=sha('step14_metrics.py'),
               commit=subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip(), seconds=round(time.time() - T0)),
          open(OUT, 'w'), indent=2, allow_nan=True)
print('\nwrote %s and %s  (%.0f s)' % (OUT, OUT_CSV, time.time() - T0))
