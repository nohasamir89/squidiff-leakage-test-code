#!/usr/bin/env python3
"""PRE-REGISTERED Step 9 analysis. PREREG_step10_chemvae_stochastic.md.

PRIMARY, gated: DPC-50, median of paired per-condition differences, arm C
(fingerprint + ChemicalVAE) minus arm A (fingerprint), drug-clustered bootstrap.
Matches Step 6c and amendment 12 so the three are comparable.

SECONDARY: B minus A, with amendment 8's caveat that arm B's entire drug
representation is a single non-zero scalar; DE-overlap; energy distance; and
each arm against the pathway-ridge reference of 0.2133.
"""
import os, sys, json, hashlib, subprocess
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist, pdist
import scanpy as sc

RES, K, N_SEEDS, N_BOOT, SEED = 'leakage_results', 50, 5, 5000, 0
OUT = os.path.join(RES, 'step10_analysis.json')
ARMS = ('fingerprint_s10', 'fp_chemvae', 'chemvae')
rng = np.random.RandomState(SEED)
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
PRE = 'PREREG_step10_chemvae_stochastic.md'
if open(PRE).read().count('\n## ') < 9: sys.exit('FATAL: prereg incomplete')
if subprocess.run('git diff --quiet -- %s' % PRE, shell=True).returncode != 0:
    sys.exit('FATAL: prereg uncommitted')
print('[prereg] committed')

def sha(p):
    m = hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): m.update(c)
    return m.hexdigest()
def jac(a,b):
    i=len(np.intersect1d(a,b)); return i/(len(a)+len(b)-i)
def per_cond(f):
    v=np.load(f, allow_pickle=True)
    true,B,C=np.asarray(v['true']),np.asarray(v['B']),np.asarray(v['C'])
    cond=np.array([str(x) for x in v['cond']])
    d=np.full(len(true),np.nan); j=np.full(len(true),np.nan)
    for i in range(len(true)):
        td=true[i]-C[i]; g=np.argsort(np.abs(td))[-K:]; p=(B[i]-C[i])
        if td[g].std()>1e-12 and p[g].std()>1e-12: d[i]=pearsonr(p[g],td[g])[0]
        if np.abs(p).std()>1e-12: j[i]=jac(g,np.argsort(np.abs(p))[-K:])
    return d,j,cond
def energy(f):
    p=np.load(f.replace('.npz','_pc.npz'), allow_pickle=True)
    o=[]
    for a,b in zip(p['pred'],p['true']):
        X=np.asarray(a,dtype=np.float64); Y=np.asarray(b,dtype=np.float64)
        o.append(2*cdist(X,Y).mean()-pdist(X).mean()-pdist(Y).mean())
    return float(np.mean(o))
def boot(x,drugs,n):
    ix=np.where(np.isfinite(x))[0]
    gs=[ix[drugs[ix]==u] for u in np.unique(drugs[ix])]
    if len(gs)!=n: sys.exit('FATAL: %d clusters, expected %d' % (len(gs),n))
    out=np.array([np.median(x[np.concatenate([gs[k] for k in rng.randint(0,len(gs),len(gs))])])
                  for _ in range(N_BOOT)])
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5))

D='/scratch/ni2071/Diffusion/sciplex/datasets_corrected'
ad=sc.read_h5ad('%s/sci_plex_test_drug_split_1.h5ad'%D); g=ad.obs['Group'].astype(str).values
DPC,JAC,EN,ref={},{},{},None
for arm in ARMS:
    ds,js,es=[],[],[]
    for s in range(N_SEEDS):
        f=os.path.join(RES,'vectors_s10_%s_seed_%d.npz'%(arm,s))
        if not os.path.isfile(f): sys.exit('FATAL: missing '+f)
        d,j,cond=per_cond(f)
        if ref is None: ref=cond
        elif not np.array_equal(cond,ref): sys.exit('FATAL: condition order differs')
        ds.append(d); js.append(j); es.append(energy(f))
    DPC[arm]=np.nanmean(np.vstack(ds),axis=0); JAC[arm]=np.array([np.nanmean(x) for x in js])
    EN[arm]=np.array(es)
drugs=np.array([ad.obs['SMILES'].astype(str).values[g==c][0] for c in ref])
nd=len(np.unique(drugs))
print('[design] %d conditions | %d drugs' % (len(ref), nd))

print('\n'+'='*72+'\nPRIMARY (gated): DPC-50, arm B minus arm A\n'+'='*72)
res={}
for lab,a0,a1,gated in (('B - A  (fp_chemvae minus fingerprint)','fingerprint_s10','fp_chemvae',True),
                        ('C - A  (chemvae minus fingerprint)','fingerprint_s10','chemvae',False)):
    diff=DPC[a1]-DPC[a0]; ok=np.isfinite(diff)
    med=float(np.median(diff[ok])); lo,hi=boot(diff,drugs,nd)
    v=('above zero' if lo>0 else 'below zero' if hi<0 else 'includes zero')
    print('  %-36s %+.4f  95%% CI [%+.4f, %+.4f]  -> %s%s'
          % (lab,med,lo,hi,v,'' if gated else '   [SECONDARY]'))
    res[lab]=dict(paired_median=med,ci=[lo,hi],verdict=v,gated=gated,
                  frac_positive=float(np.mean(diff[ok]>0)))
print('\n  marginal medians (NOT the estimand): ' +
      '  '.join('%s %.4f'%(a,np.nanmedian(DPC[a])) for a in ARMS))
print('\nSECONDARY, not gated')
for a in ARMS:
    print('  %-12s DE-overlap %.4f  energy %.5f' % (a, JAC[a].mean(), EN[a].mean()))
print('\n  ridge reference on this split: fingerprint 0.1769 | ChemicalVAE 0.0183 | pathway 0.2251')
print('  arm C REMOVES the fingerprint, so a null there confounds the new')
print('  representation with the removal; arm B is the gated contrast.')
gg=res['B - A  (fp_chemvae minus fingerprint)']
print('\n'+'='*72)
print('PRE-REGISTERED VERDICT: %s' % {'above zero':'STOCHASTIC CHEMICALVAE IMPROVES DPC-50',
      'below zero':'FINGERPRINT ALONE IS BETTER'}.get(gg['verdict'],'UNRESOLVED'))
print('  Section 7 recorded UNRESOLVED as expected.')
print('  UNRESOLVED is not equivalence; no margin was justified.')
print('='*72)
gc=lambda c: subprocess.run(c,shell=True,capture_output=True,text=True).stdout.strip()
json.dump(dict(contrasts=res, dpc_marginal={a:float(np.nanmedian(DPC[a])) for a in ARMS},
               de_overlap={a:list(map(float,JAC[a])) for a in ARMS},
               energy={a:list(map(float,EN[a])) for a in ARMS},
               n_conditions=len(ref), n_drugs=nd, prereg=PRE,
               commit=gc('git rev-parse HEAD'), script_sha256=sha(os.path.abspath(__file__))),
          open(OUT+'.tmp','w'),indent=2,allow_nan=False)
os.replace(OUT+'.tmp',OUT); print('\nwrote %s'%OUT)
