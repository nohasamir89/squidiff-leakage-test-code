#!/usr/bin/env python3
"""Step 7: the drug x cell-line interaction baseline. PREREG_step7_interaction_baseline.md.

NOT BLIND for drug_split_1, whose values are reproduced in section 2 of the
pre-registration. The replication on splits 0 and 2 and every paired contrast
were fixed in advance and had not been computed.

Estimand: median of paired per-condition differences, drug-clustered bootstrap.
The difference of marginal medians is printed alongside but is NOT the estimand;
it reversed sign against the paired form twice in this project.
"""
import os, sys, json, hashlib, subprocess
import numpy as np, scanpy as sc
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
from Squidiff.scrna_datasets import Drug_dose_encoder

ROOT = os.path.dirname(os.path.abspath(__file__))
RES, PROV = os.path.join(ROOT,'leakage_results'), os.path.join(ROOT,'provenance')
DATA = '/scratch/ni2071/Diffusion/sciplex/datasets_corrected'
OUT = os.path.join(RES, 'step7_interaction_baseline.json')
SPLITS = {'drug_split_0': (444,37), 'drug_split_1': (443,37), 'drug_split_2': (455,38)}
K, N_SEEDS, N_BOOT, SEED, ALPHA = 50, 5, 5000, 0, 1.0
rng = np.random.RandomState(SEED)

if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
PRE = os.path.join(ROOT,'PREREG_step7_interaction_baseline.md')
if open(PRE).read().count('\n## ') < 7: sys.exit('FATAL: prereg incomplete')
if subprocess.run('git diff --quiet -- %s' % PRE, shell=True).returncode != 0:
    sys.exit('FATAL: prereg uncommitted')
print('[prereg] Step 7 present and committed (NOT blind for drug_split_1)')

def sha(p):
    m=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): m.update(c)
    return m.hexdigest()

def fm_vector(sp, seed):
    if sp == 'drug_split_1':
        vp = os.path.join(RES,'vectors_step6_fm_gaussian_seed_%d.npz'%seed)
        ep = os.path.join(PROV,'s6eval_step6_fm_gaussian_seed_%d.json'%seed)
    else:
        vp = os.path.join(RES,'vectors_s6b_%s_fm_gaussian_seed_%d.npz'%(sp,seed))
        ep = os.path.join(PROV,'s6bev_s6b_%s_fm_gaussian_seed_%d.json'%(sp,seed))
    for p in (vp, ep):
        if not os.path.isfile(p): sys.exit('FATAL: missing '+p)
    rec = json.load(open(ep))
    if str(rec.get('arm')) != 'fm_gaussian': sys.exit('FATAL: %s arm=%r'%(vp,rec.get('arm')))
    if int(rec.get('nfe',20)) != 20: sys.exit('FATAL: %s nfe=%r'%(vp,rec.get('nfe')))
    if rec.get('vectors_sha256') != sha(vp): sys.exit('FATAL: %s changed'%vp)
    return vp

def dpc_vec(P, Y):
    out = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        d = np.argsort(np.abs(Y[i]))[-K:]
        if Y[i][d].std()>1e-12 and P[i][d].std()>1e-12:
            out[i] = pearsonr(P[i][d], Y[i][d])[0]
    return out

def cluster_boot(x, drugs, n_expected):
    ix = np.where(np.isfinite(x))[0]
    groups = [ix[drugs[ix]==u] for u in np.unique(drugs[ix])]
    if len(groups) != n_expected:
        sys.exit('FATAL: %d drug clusters, expected %d' % (len(groups), n_expected))
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.randint(0, len(groups), len(groups))
        out[b] = np.median(x[np.concatenate([groups[k] for k in pick])])
    return tuple(float(q) for q in np.percentile(out, [2.5, 97.5]))

results = {}
for sp, (n_want, d_want) in SPLITS.items():
    print('\n' + '='*72 + '\n%s\n' % sp + '='*72)
    v0 = fm_vector(sp, 0)
    with np.load(v0, allow_pickle=True) as v:
        Y = np.asarray(v['true']) - np.asarray(v['C'])
        cond = np.array([str(x) for x in v['cond']])
    if len(cond) != n_want: sys.exit('FATAL: %d conditions, expected %d' % (len(cond), n_want))
    ad = sc.read_h5ad('%s/sci_plex_test_%s.h5ad' % (DATA, sp))
    g = ad.obs['Group'].astype(str).values
    smi  = {c: ad.obs['SMILES'].astype(str).values[g==c][0] for c in cond}
    dose = {c: float(np.median(ad.obs['dose'].astype(float).values[g==c])) for c in cond}
    line = {c: ad.obs['cell_name'].astype(str).values[g==c][0] for c in cond}
    drugs = np.array([smi[c] for c in cond])
    if len(np.unique(drugs)) != d_want:
        sys.exit('FATAL: %d drugs, expected %d' % (len(np.unique(drugs)), d_want))
    FP = np.array([Drug_dose_encoder([smi[c]],[dose[c]],comb_num=1)[0] for c in cond])
    lv = sorted(set(line.values()))
    L = np.array([[float(line[c]==u) for u in lv] for c in cond])
    INT = np.concatenate([FP*L[:,[k]] for k in range(L.shape[1])], axis=1)

    def ridge_pred(X):
        P = np.zeros_like(Y)
        for a,b in GroupKFold(n_splits=5).split(X, Y, groups=drugs):
            P[b] = Ridge(alpha=ALPHA).fit(X[a], Y[a]).predict(X[b])
        return P
    d_plain = dpc_vec(ridge_pred(FP), Y)
    d_inter = dpc_vec(ridge_pred(np.hstack([FP, L, INT])), Y)

    per_seed = []
    for s in range(N_SEEDS):
        with np.load(fm_vector(sp, s), allow_pickle=True) as v:
            c2 = np.array([str(x) for x in v['cond']])
            if not np.array_equal(c2, cond): sys.exit('FATAL: condition order differs')
            per_seed.append(dpc_vec(np.asarray(v['B']) - np.asarray(v['C']), Y))
    d_fm = np.nanmean(np.vstack(per_seed), axis=0)

    print('  marginal medians (NOT the estimand):  plain %.4f | interaction %.4f | FM %.4f'
          % (np.nanmedian(d_plain), np.nanmedian(d_inter), np.nanmedian(d_fm)))
    row = {}
    for lab, diff in (('interaction - plain ridge', d_inter - d_plain),
                      ('flow matching - interaction', d_fm - d_inter)):
        ok = np.isfinite(diff)
        med = float(np.median(diff[ok]))
        lo, hi = cluster_boot(diff, drugs, d_want)
        verdict = ('above zero' if lo > 0 else 'below zero' if hi < 0 else 'includes zero')
        print('  PAIRED  %-28s %+.4f  95%% CI [%+.4f, %+.4f]  -> %s'
              % (lab, med, lo, hi, verdict))
        row[lab] = dict(paired_median=med, ci=[lo,hi], verdict=verdict,
                        frac_positive=float(np.mean(diff[ok] > 0)))
    row.update(marginal=dict(plain=float(np.nanmedian(d_plain)),
                             interaction=float(np.nanmedian(d_inter)),
                             flow_matching=float(np.nanmedian(d_fm))),
               n_conditions=int(len(cond)), n_drugs=int(d_want))
    results[sp] = row

def overall(key):
    a = [s for s in SPLITS if results[s][key]['ci'][0] > 0]
    b = [s for s in SPLITS if results[s][key]['ci'][1] < 0]
    if len(a) == 3: return 'RESOLVED ABOVE ZERO on all three'
    if len(b) == 3: return 'RESOLVED BELOW ZERO on all three'
    if a or b: return 'MIXED (above: %s | below: %s)' % (a or 'none', b or 'none')
    return 'UNRESOLVED on all three'
print('\n' + '='*72)
print('VERDICT  interaction - plain ridge   : %s' % overall('interaction - plain ridge'))
print('VERDICT  flow matching - interaction : %s' % overall('flow matching - interaction'))
print('\n  Both directions were recorded as expected in section 5 before computation.')
print('  UNRESOLVED is not equivalence; no margin was justified.')
print('='*72)
g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
json.dump(dict(per_split=results,
               verdicts={k: overall(k) for k in ('interaction - plain ridge',
                                                 'flow matching - interaction')},
               prereg='PREREG_step7_interaction_baseline.md', blind=False,
               feature_set='FP + line + FP x line', alpha=ALPHA,
               bootstrap=dict(unit='drug', n_replicates=N_BOOT, seed=SEED),
               commit=g('git rev-parse HEAD'), script_sha256=sha(os.path.abspath(__file__))),
          open(OUT+'.tmp','w'), indent=2, allow_nan=False)
os.replace(OUT+'.tmp', OUT); print('\nwrote %s' % OUT)
