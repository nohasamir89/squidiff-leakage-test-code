#!/usr/bin/env python3
"""Step 8 section 4 sanity arm. PREREG_step8_sample_regime.md.

Can a LINEAR network trained by AdamW match ridge's closed-form solution on
identical features and folds? If not, the MLP deficit is confounded with
optimisation and every conclusion about model class is downgraded to "under
this training procedure" - fixed in advance in section 6.

Runs BEFORE the learning curve, because a failure here invalidates it.
"""
import os, sys, json, hashlib, subprocess
import numpy as np, scanpy as sc, torch as th, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
from Squidiff.scrna_datasets import Drug_dose_encoder

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'leakage_results')
DATA = '/scratch/ni2071/Diffusion/sciplex/datasets_corrected'
OUT = os.path.join(RES, 'step8_linear_check_v2.json')
K, TOL = 50, 0.02
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
PRE = os.path.join(ROOT, 'PREREG_step8_sample_regime.md')
if open(PRE).read().count('\n## ') < 8: sys.exit('FATAL: prereg incomplete')
if subprocess.run('git diff --quiet -- %s' % PRE, shell=True).returncode != 0:
    sys.exit('FATAL: prereg uncommitted')
print('[prereg] Step 8 present and committed')

def sha(p):
    m = hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): m.update(c)
    return m.hexdigest()

def dpc(P, Y):
    o = []
    for i in range(len(Y)):
        d = np.argsort(np.abs(Y[i]))[-K:]
        if Y[i][d].std() > 1e-12 and P[i][d].std() > 1e-12:
            o.append(pearsonr(P[i][d], Y[i][d])[0])
    return float(np.median(o))

def build(sp):
    f = ('leakage_results/vectors_step6_fm_gaussian_seed_0.npz' if sp == 'drug_split_1'
         else 'leakage_results/vectors_s6b_%s_fm_gaussian_seed_0.npz' % sp)
    if not os.path.isfile(f): sys.exit('FATAL: missing ' + f)
    v = np.load(f, allow_pickle=True)
    Y = (np.asarray(v['true']) - np.asarray(v['C'])).astype(np.float32)
    cond = np.array([str(x) for x in v['cond']])
    ad = sc.read_h5ad('%s/sci_plex_test_%s.h5ad' % (DATA, sp))
    g = ad.obs['Group'].astype(str).values
    smi  = {c: ad.obs['SMILES'].astype(str).values[g==c][0] for c in cond}
    dose = {c: float(np.median(ad.obs['dose'].astype(float).values[g==c])) for c in cond}
    line = {c: ad.obs['cell_name'].astype(str).values[g==c][0] for c in cond}
    drugs = np.array([smi[c] for c in cond])
    FP = np.array([Drug_dose_encoder([smi[c]],[dose[c]],comb_num=1)[0] for c in cond],
                  dtype=np.float32)
    lv = sorted(set(line.values()))
    L = np.array([[float(line[c]==u) for u in lv] for c in cond], dtype=np.float32)
    X = np.hstack([FP, L, np.concatenate([FP*L[:,[k]] for k in range(len(lv))], axis=1)])
    return X.astype(np.float32), Y, drugs

def linear_net(X, Y, drugs, wd, epochs, lr, seed=0):
    th.manual_seed(seed); P = np.zeros_like(Y)
    for a, b in GroupKFold(n_splits=5).split(X, Y, groups=drugs):
        xa, ya, xb = th.tensor(X[a]), th.tensor(Y[a]), th.tensor(X[b])
        # Drop columns constant in the TRAINING fold before standardising.
        # 2,560 of 4,099 are constant per fold; dividing a non-zero TEST value
        # by 1e-6 produced |z| up to 4e6 on 84 of 96 test rows and inflated
        # test MSE to 7.5e9. Ridge was unaffected: sklearn does not standardise.
        keep = xa.std(0) > 1e-9
        xa, xb = xa[:, keep], xb[:, keep]
        mu, sd = xa.mean(0, keepdim=True), xa.std(0, keepdim=True) + 1e-6
        m = nn.Linear(int(keep.sum()), Y.shape[1])
        opt = th.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad(); (((m((xa-mu)/sd) - ya)**2).mean()).backward(); opt.step()
        m.eval()
        with th.no_grad(): P[b] = m((xb-mu)/sd).numpy()
    return dpc(P, Y)

# Every split selected wd=0.1, ep=4000, lr=0.01 - the epoch ceiling of the
# previous grid, which is the signature of not having converged. Extended.
GRID = [(wd, ep, lr) for wd in (1e-2, 1e-1, 1.0)
                     for ep in (4000, 16000, 64000)
                     for lr in (1e-2, 3e-2)]
results = {}
print('\n%-14s %9s %11s %s' % ('split', 'ridge', 'linear net', 'best setting'))
for sp in ('drug_split_0', 'drug_split_1', 'drug_split_2'):
    X, Y, drugs = build(sp)
    P = np.zeros_like(Y)
    for a, b in GroupKFold(n_splits=5).split(X, Y, groups=drugs):
        P[b] = Ridge(alpha=1.0).fit(X[a], Y[a]).predict(X[b])
    r = dpc(P, Y)
    scores = {}
    for wd, ep, lr in GRID:
        scores['wd%g_ep%d_lr%g' % (wd, ep, lr)] = linear_net(X, Y, drugs, wd, ep, lr)
    bk = max(scores, key=scores.get); bv = scores[bk]
    print('%-14s %9.4f %11.4f  %s  gap %+.4f' % (sp, r, bv, bk, bv - r))
    results[sp] = dict(ridge=r, linear_best=bv, best_setting=bk, gap=bv - r,
                       within_tolerance=bool(abs(bv - r) <= TOL), all_scores=scores,
                       n_conditions=int(len(Y)), n_drugs=int(len(np.unique(drugs))))

ok = sum(results[s]['within_tolerance'] for s in results)
print('\n' + '='*70)
if ok == 3:
    verdict = 'HARNESS SOUND'
    note = ('A linear network matches ridge within %.2f on all three splits, so the '
            'nonlinear arms\' deficit is about model class rather than optimisation.' % TOL)
elif ok == 0:
    verdict = 'OPTIMISATION CONFOUND'
    note = ('The linear network does NOT reach ridge on any split. The MLP deficit is '
            'confounded with training procedure, and every Step 8 conclusion is '
            'downgraded to "under this training procedure". Section 6 fixed this in '
            'advance.')
else:
    verdict = 'MIXED (%d of 3 within tolerance)' % ok
    note = 'Reported per split; conclusions qualified on the splits that failed.'
print('SANITY VERDICT: %s' % verdict); print('  %s' % note); print('='*70)
g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
json.dump(dict(verdict=verdict, note=note, tolerance=TOL, per_split=results,
               prereg='PREREG_step8_sample_regime.md section 4',
               commit=g('git rev-parse HEAD'), script_sha256=sha(os.path.abspath(__file__))),
          open(OUT+'.tmp','w'), indent=2, allow_nan=False)
os.replace(OUT+'.tmp', OUT); print('\nwrote %s' % OUT)
