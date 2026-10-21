#!/usr/bin/env python3
"""PRE-REGISTERED Step 13 analysis. PREREG_step13_attention.md, section 8.

Written and committed BEFORE any Step 13 checkpoint exists. Refuses to run if any
input is missing, if a provenance record disagrees with its filename (arm, memory
size, parameter count, guidance), or if a vectors hash disagrees with its record.

Development split drug_split_1, 443 conditions, 37 held-out compounds. Four arms
(attn_fp, attn_mem, attn_cross, mlp_ctrl; five seeds each) against the reference:
the Step 6 checkpoints at 50 NFE (Step 6 sweep vectors, sw_fm_gaussian_seed_k_n50).

Contrasts: mean over conditions of seed-mean per-condition values, drug-clustered
percentile bootstrap over the 37 compounds, 5000 replicates, RandomState(0), 95%.
  PRIMARY (gated, per attention arm)      DE-overlap Jaccard, arm minus reference
  ATTRIBUTION (gated, per attention arm)  DE-overlap Jaccard, arm minus mlp_ctrl
  CO-PRIMARY (gated, per attention arm)   energy distance, arm minus reference (higher = worse)
  SECONDARY                               mlp_ctrl minus reference (both endpoints);
                                          DPC-50 paired median vs the Step 7 interaction ridge,
                                          reproduced here with Step 7's exact procedure;
                                          DPC-50 mean, B-C, magnitude for every arm;
                                          attn_mem memory-attention entropy on held-out compounds
Categories (section 5), per attention arm, applied mechanically:
  ATTENTION HELPS  DE above zero vs reference AND vs mlp_ctrl AND energy not above zero
  CAPACITY         DE above zero vs reference, not above zero vs mlp_ctrl
  TRADE-OFF        DE above zero vs reference AND energy above zero (whatever the control)
  UNRESOLVED       DE vs reference includes zero
  HARMS            DE vs reference below zero
"""
import os, sys, json, subprocess, hashlib
import numpy as np
import anndata as ad
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from Squidiff.scrna_datasets import Drug_dose_encoder

RES = 'leakage_results'
DATA = '/scratch/ni2071/Diffusion/sciplex/datasets_corrected'
SPLIT, N_EXPECT, N_DRUGS = 'drug_split_1', 443, 37
N_SEEDS, K, N_BOOT, ALPHA = 5, 50, 5000, 1.0
LO, HI = 2.5, 97.5                        # 95% interval, development split
ARMS = ('attn_fp', 'attn_mem', 'attn_cross', 'mlp_ctrl'); ATT = ('attn_fp', 'attn_mem', 'attn_cross')
MEMSZ = {'attn_fp': 0, 'attn_mem': 112, 'attn_cross': 0, 'mlp_ctrl': 0}
EXPECT_PARAMS = {'attn_fp': 58360260, 'attn_mem': 58383108, 'attn_cross': 58532804, 'mlp_ctrl': 58532300}
OUT = os.path.join(RES, 'step13_analysis.json')
PRE = 'PREREG_step13_attention.md'
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
if open(PRE).read().count('\n## ') < 9: sys.exit('FATAL: prereg incomplete')
if subprocess.run('git diff --quiet -- %s' % PRE, shell=True).returncode != 0: sys.exit('FATAL: prereg uncommitted')
print('[prereg] committed')
rng = np.random.RandomState(0)

def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): m.update(c)
    return m.hexdigest()

def jac(a, b):
    i = len(np.intersect1d(a, b)); return i / (len(a) + len(b) - i)

def per_cond(f):
    v = np.load(f, allow_pickle=True)
    true, B, C = [np.asarray(v[k], dtype=np.float64) for k in ('true', 'B', 'C')]
    cond = np.array([str(x) for x in v['cond']])
    n = len(true)
    j = np.full(n, np.nan); d = np.full(n, np.nan); bc = np.full(n, np.nan); mag = np.full(n, np.nan); magt = np.full(n, np.nan)
    for i in range(n):
        td = true[i] - C[i]; p = B[i] - C[i]
        g = np.argsort(np.abs(td))[-K:]
        if np.abs(p).std() > 1e-12: j[i] = jac(g, np.argsort(np.abs(p))[-K:])
        if td[g].std() > 1e-12 and p[g].std() > 1e-12: d[i] = pearsonr(p[g], td[g])[0]
        if true[i][g].std() > 1e-12 and B[i][g].std() > 1e-12 and C[i][g].std() > 1e-12:
            bc[i] = pearsonr(B[i][g], true[i][g])[0] - pearsonr(C[i][g], true[i][g])[0]
        mag[i] = np.abs(p).mean(); magt[i] = np.abs(td).mean()
    return dict(jac=j, dpc=d, bc=bc, mag=mag, magt=magt, cond=cond)

def energy_per_cond(f_pc):
    p = np.load(f_pc, allow_pickle=True); out = []
    for a, b in zip(p['pred'], p['true']):
        X = np.asarray(a, dtype=np.float64); Y = np.asarray(b, dtype=np.float64)
        if X.shape != Y.shape: sys.exit('FATAL: per-cell shape mismatch in %s' % f_pc)
        out.append(2 * cdist(X, Y).mean() - pdist(X).mean() - pdist(Y).mean())
    return np.array(out)

def boot(x, drugs, stat=np.mean):
    """Drug-clustered percentile bootstrap of stat over the finite entries of x."""
    ix = np.where(np.isfinite(x))[0]
    gs = [ix[drugs[ix] == u] for u in np.unique(drugs[ix])]
    if len(gs) != N_DRUGS: sys.exit('FATAL: %d drug clusters, expected %d' % (len(gs), N_DRUGS))
    o = np.array([stat(x[np.concatenate([gs[k] for k in rng.randint(0, len(gs), len(gs))])]) for _ in range(N_BOOT)])
    return float(np.percentile(o, LO)), float(np.percentile(o, HI))

def verdict(lo, hi):
    return 'above zero' if lo > 0 else 'below zero' if hi < 0 else 'includes zero'

def load_arm(name_of_seed, prov_of_seed, check, ref_cond):
    per = {k: [] for k in ('jac', 'dpc', 'bc', 'mag', 'magt', 'en')}; files = []
    for s in range(N_SEEDS):
        name = name_of_seed(s)
        f = os.path.join(RES, 'vectors_%s.npz' % name); fpc = os.path.join(RES, 'vectors_%s_pc.npz' % name); pj = prov_of_seed(s)
        for q in (f, fpc, pj):
            if not os.path.isfile(q): sys.exit('FATAL: missing %s' % q)
        pv = json.load(open(pj)); check(pv, name)
        if pv.get('vectors_sha256') != sha(f): sys.exit('FATAL: %s vectors hash differs from provenance' % name)
        if pv.get('per_cell_sha256') != sha(fpc): sys.exit('FATAL: %s per-cell hash differs from provenance' % name)
        r = per_cond(f)
        if len(r['cond']) != N_EXPECT: sys.exit('FATAL: %s has %d conditions, expected %d' % (name, len(r['cond']), N_EXPECT))
        if ref_cond[0] is None: ref_cond[0] = r['cond']
        elif not np.array_equal(r['cond'], ref_cond[0]): sys.exit('FATAL: condition order differs in %s' % name)
        for k in ('jac', 'dpc', 'bc', 'mag', 'magt'): per[k].append(r[k])
        per['en'].append(energy_per_cond(fpc))
        files.append(dict(tag=name, vectors_sha256=pv['vectors_sha256'], per_cell_sha256=pv['per_cell_sha256'],
                          checkpoint_sha256=pv.get('checkpoint_sha256'), checkpoint=pv.get('checkpoint')))
    for k in ('jac', 'en'):
        if not np.all(np.isfinite(np.vstack(per[k]))): sys.exit('FATAL: non-finite %s in %s' % (k, name_of_seed(0)))
    return (dict(cond_mean={k: np.nanmean(np.vstack(per[k]), axis=0) for k in per},
                 ckpt_mean={k: np.array([np.nanmean(x) for x in per[k]]) for k in per}), files)

# ------------------------------------------------------------------ load everything
M = {}; inputs = {}; ref_cond = [None]
def check_ref(pv, name):
    if pv.get('arm') != 'fm_gaussian' or int(pv.get('steps', 0)) != 50: sys.exit('FATAL: %s is not fm_gaussian at 50 steps' % name)
M['ref'], inputs['ref'] = load_arm(lambda s: 'sw_fm_gaussian_seed_%d_n50' % s,
                                   lambda s: os.path.join('provenance', 's6sw_sw_fm_gaussian_seed_%d_n50.json' % s), check_ref, ref_cond)
for arm in ARMS:
    def check_arm(pv, name, arm=arm):
        if pv.get('encoder_arm') != arm or pv.get('arm') != arm: sys.exit('FATAL: %s provenance arm=%r/%r, expected %s' % (name, pv.get('arm'), pv.get('encoder_arm'), arm))
        if pv.get('split') != SPLIT: sys.exit('FATAL: %s provenance split=%r' % (name, pv.get('split')))
        if int(pv.get('memory_size', -1)) != MEMSZ[arm]: sys.exit('FATAL: %s memory_size=%r, expected %d' % (name, pv.get('memory_size'), MEMSZ[arm]))
        if int(pv.get('n_params', -1)) != EXPECT_PARAMS[arm]: sys.exit('FATAL: %s n_params=%r, expected %d' % (name, pv.get('n_params'), EXPECT_PARAMS[arm]))
        if pv.get('guidance_w') is not None: sys.exit('FATAL: %s was evaluated with guidance' % name)
        if float(pv.get('cond_drop', 1.0)) != 0.0: sys.exit('FATAL: %s cond_drop=%r' % (name, pv.get('cond_drop')))
    M[arm], inputs[arm] = load_arm(lambda s, a=arm: 's13_%s_%s_seed_%d' % (SPLIT, a, s),
                                   lambda s, a=arm: os.path.join('provenance', 's13ev_s13_%s_%s_seed_%d.json' % (SPLIT, a, s)), check_arm, ref_cond)
cond = ref_cond[0]
a = ad.read_h5ad('%s/sci_plex_test_%s.h5ad' % (DATA, SPLIT), backed='r')
g = a.obs['Group'].astype(str).values; smi_all = a.obs['SMILES'].astype(str).values
dose_all = a.obs['dose'].astype(float).values; line_all = a.obs['cell_name'].astype(str).values
drugs = np.array([smi_all[g == c][0] for c in cond])
if len(np.unique(drugs)) != N_DRUGS: sys.exit('FATAL: %d drugs, expected %d' % (len(np.unique(drugs)), N_DRUGS))
print('[design] %d conditions | %d drugs | reference = Step 6 checkpoints at 50 NFE' % (len(cond), N_DRUGS))
for tag in ('ref',) + ARMS:
    c = M[tag]['ckpt_mean']
    print('[load] %-10s DE-overlap %.4f  energy %.4f  DPC-50 %.4f  B-C %+.4f  |B-C| %.5f  |true-C| %.5f'
          % (tag, c['jac'].mean(), c['en'].mean(), c['dpc'].mean(), c['bc'].mean(), c['mag'].mean(), c['magt'].mean()))
# magnitude gate carried from Step 10, per arm against the reference
for arm in ARMS:
    ratio = M[arm]['ckpt_mean']['mag'].mean() / M['ref']['ckpt_mean']['mag'].mean()
    if not (0.5 <= ratio <= 2.0): sys.exit('FATAL: %s |B-C| is %.2fx the reference; pairing or checkpoint fault' % (arm, ratio))
    print('[sanity] %-10s |B-C| / reference: %.3f' % (arm, ratio))

def contrast(b, a_, key):
    diff = M[b]['cond_mean'][key] - M[a_]['cond_mean'][key]
    m = float(np.nanmean(diff)); lo, hi = boot(diff, drugs)
    return dict(diff=m, ci=[lo, hi], verdict=verdict(lo, hi), half_width=float((hi - lo) / 2))

# ------------------------------------------------------------------ gated contrasts and categories
res = {'arm_means': {t: {k: float(M[t]['ckpt_mean'][k].mean()) for k in ('jac', 'en', 'dpc', 'bc', 'mag', 'magt')} for t in M}}
ctrl = {k: contrast('mlp_ctrl', 'ref', k) for k in ('jac', 'en', 'dpc')}
res['control_minus_reference'] = ctrl
print('\nSECONDARY  mlp_ctrl minus reference:  DE %+.5f [%+.5f,%+.5f] %-13s | energy %+.5f [%+.5f,%+.5f] %-13s | DPC %+.5f'
      % (ctrl['jac']['diff'], *ctrl['jac']['ci'], ctrl['jac']['verdict'], ctrl['en']['diff'], *ctrl['en']['ci'], ctrl['en']['verdict'], ctrl['dpc']['diff']))
if ctrl['jac']['verdict'] == 'above zero':
    print('  NOTE: the parameter-matched control alone beats the reference; every arm comparison below is read in that light (PREREG section 5).')
res['arms'] = {}; cats = {}
for arm in ATT:
    pj = contrast(arm, 'ref', 'jac'); pe = contrast(arm, 'ref', 'en'); pc = contrast(arm, 'mlp_ctrl', 'jac')
    if pj['verdict'] == 'below zero': cat = 'HARMS'
    elif pj['verdict'] == 'includes zero': cat = 'UNRESOLVED'
    elif pe['verdict'] == 'above zero': cat = 'TRADE-OFF'
    elif pc['verdict'] == 'above zero': cat = 'ATTENTION HELPS'
    else: cat = 'CAPACITY'
    cats[arm] = cat
    extra = {k: contrast(arm, 'ref', k) for k in ('dpc', 'bc')}
    res['arms'][arm] = dict(de_overlap_minus_reference=pj, energy_minus_reference=pe, de_overlap_minus_control=pc,
                            dpc_minus_reference=extra['dpc'], bc_minus_reference=extra['bc'], category=cat)
    print('\n%s' % arm)
    print('  PRIMARY      DE-overlap, arm minus reference   %+.5f  95%% CI [%+.5f, %+.5f]  -> %s   (half-width %.5f)' % (pj['diff'], *pj['ci'], pj['verdict'], pj['half_width']))
    print('  ATTRIBUTION  DE-overlap, arm minus mlp_ctrl    %+.5f  95%% CI [%+.5f, %+.5f]  -> %s' % (pc['diff'], *pc['ci'], pc['verdict']))
    print('  CO-PRIMARY   energy, arm minus reference       %+.5f  95%% CI [%+.5f, %+.5f]  -> %s   (%+.0f%% of reference)'
          % (pe['diff'], *pe['ci'], pe['verdict'], 100 * pe['diff'] / res['arm_means']['ref']['en']))
    print('  reported     DPC-50 mean %+.5f | B-C %+.5f | magnitude ratio %.3f' % (extra['dpc']['diff'], extra['bc']['diff'], res['arm_means'][arm]['mag'] / res['arm_means'][arm]['magt']))
    print('  CATEGORY: %s' % cat)

# ------------------------------------------------------------------ DPC-50 against the interaction ridge, Step 7's exact procedure
s7 = json.load(open(os.path.join(RES, 'step7_interaction_baseline.json')))
v0 = os.path.join(RES, 'vectors_step6_fm_gaussian_seed_0.npz'); e0 = os.path.join('provenance', 's6eval_step6_fm_gaussian_seed_0.json')
for p in (v0, e0):
    if not os.path.isfile(p): sys.exit('FATAL: missing ' + p)
rec = json.load(open(e0))
if rec.get('vectors_sha256') != sha(v0): sys.exit('FATAL: %s changed since Step 7' % v0)
with np.load(v0, allow_pickle=True) as v:
    Y7 = np.asarray(v['true']) - np.asarray(v['C']); cond7 = np.array([str(x) for x in v['cond']])
if not np.array_equal(cond7, cond): sys.exit('FATAL: Step 7 condition order differs from the Step 13 vectors')
smi7 = {c: smi_all[g == c][0] for c in cond}; dose7 = {c: float(np.median(dose_all[g == c])) for c in cond}; line7 = {c: line_all[g == c][0] for c in cond}
FP = np.array([Drug_dose_encoder([smi7[c]], [dose7[c]], comb_num=1)[0] for c in cond])
lv = sorted(set(line7.values())); L = np.array([[float(line7[c] == u) for u in lv] for c in cond])
INT = np.concatenate([FP * L[:, [k]] for k in range(L.shape[1])], axis=1)
def dpc_vec(P, Y):
    out = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        d = np.argsort(np.abs(Y[i]))[-K:]
        if Y[i][d].std() > 1e-12 and P[i][d].std() > 1e-12: out[i] = pearsonr(P[i][d], Y[i][d])[0]
    return out
def ridge_pred(X):
    P = np.zeros_like(Y7)
    for a_, b_ in GroupKFold(n_splits=5).split(X, Y7, groups=drugs):
        P[b_] = Ridge(alpha=ALPHA).fit(X[a_], Y7[a_]).predict(X[b_])
    return P
d_inter = dpc_vec(ridge_pred(np.hstack([FP, L, INT])), Y7)
rec_med = float(s7['per_split'][SPLIT]['marginal']['interaction'])
if abs(float(np.nanmedian(d_inter)) - rec_med) > 1e-9:
    sys.exit('FATAL: interaction ridge reproduced at %.6f, Step 7 recorded %.6f' % (np.nanmedian(d_inter), rec_med))
print('\n[ridge] interaction ridge reproduced: marginal median DPC-50 %.4f (Step 7 record %.4f)' % (np.nanmedian(d_inter), rec_med))
res['dpc_vs_interaction_ridge'] = {}
for tag in ('ref',) + ARMS:
    diff = M[tag]['cond_mean']['dpc'] - d_inter; ok = np.isfinite(diff)
    med = float(np.median(diff[ok])); lo, hi = boot(diff, drugs, stat=np.median)
    row = dict(paired_median=med, ci=[lo, hi], verdict=verdict(lo, hi), arm_marginal_median=float(np.nanmedian(M[tag]['cond_mean']['dpc'])), frac_positive=float(np.mean(diff[ok] > 0)))
    res['dpc_vs_interaction_ridge'][tag] = row
    print('  PAIRED DPC-50 %-10s minus interaction ridge  %+.4f  95%% CI [%+.4f, %+.4f]  -> %s   (marginal median %.4f)' % (tag, med, lo, hi, row['verdict'], row['arm_marginal_median']))

# ------------------------------------------------------------------ attn_mem: how many seen compounds does an unseen drug borrow from (descriptive)
try:
    import torch as th
    from Squidiff.drug_attention import CompoundMemoryAttention
    ents = []; maxw = []
    for f in inputs['attn_mem']:
        sd = th.load(f['checkpoint'], map_location='cpu', weights_only=True)
        mod = CompoundMemoryAttention(int(sd['encoder.arm.keys_fp'].shape[0]))
        mod.load_state_dict({k[len('encoder.arm.'):]: v for k, v in sd.items() if k.startswith('encoder.arm.')})
        mod.eval()
        with th.no_grad():
            q = mod.q(th.tensor(FP, dtype=th.float32)); k_ = mod.k(mod.keys_fp)
            att = th.softmax(q @ k_.t() / np.sqrt(mod.d), dim=-1).numpy()
        H = -(att * np.log(att + 1e-12)).sum(1)
        ents.append(H.mean()); maxw.append(att.max(1).mean())
    res['attn_mem_memory_attention'] = dict(mean_entropy_nats=float(np.mean(ents)), effective_compounds=float(np.exp(np.mean(ents))),
                                           mean_max_weight=float(np.mean(maxw)), memory_size=MEMSZ['attn_mem'])
    print('\n[attn_mem] memory attention on the 443 held-out conditions: entropy %.3f nats = %.1f effective compounds of %d; mean max weight %.3f'
          % (np.mean(ents), np.exp(np.mean(ents)), MEMSZ['attn_mem'], np.mean(maxw)))
except Exception as e:
    print('\n[attn_mem] memory-attention descriptor not computed: %s' % e); res['attn_mem_memory_attention'] = None

# ------------------------------------------------------------------ verdicts
print('\n' + '=' * 72)
for arm in ATT: print('PRE-REGISTERED CATEGORY  %-10s : %s' % (arm, cats[arm]))
print('  Section 6 expected: attn_fp UNRESOLVED; attn_mem UNRESOLVED or CAPACITY; attn_cross UNRESOLVED (small positive point estimate);\n'
      '  mlp_ctrl unresolved against the reference; every arm below the interaction ridge on paired DPC-50.')
print('=' * 72)
json.dump(dict(prereg=PRE, split=SPLIT, categories=cats, interval='95% percentile, drug-clustered', n_boot=N_BOOT,
               statistic='mean over conditions of seed-mean per-condition values (DPC vs ridge: paired median)',
               reference='Step 6 checkpoints at 50 NFE (sw_fm_gaussian_seed_k_n50)', results=res, inputs=inputs,
               commit=subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()),
          open(OUT, 'w'), indent=2, allow_nan=False)
print('\nwrote %s' % OUT)
