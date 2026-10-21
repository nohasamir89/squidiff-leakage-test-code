"""PRE-REGISTERED Step 6b confirmation. PREREG_step6b_confirmation.md.

PRIMARY, gated: raw DE-overlap Jaccard, fm_gaussian minus ddim, assessed
SEPARATELY on each split at the 97.5% level. P2 is computed and reported but
NOT gated: it was unresolved for this contrast in Step 6, so there is no
distributional finding to confirm.
"""
import os, sys, json, hashlib, subprocess
import numpy as np
from itertools import combinations
from scipy.stats import hypergeom, t as tdist
from scipy.spatial.distance import cdist, pdist

RES, K, N_SEEDS, ALPHA = 'leakage_results', 50, 5, 0.9875   # 97.5% two-sided
SPLITS = ('drug_split_0', 'drug_split_2')
ARMS = ('ddim', 'fm_gaussian')
OUT = os.path.join(RES, 'step6b_analysis.json')
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
PRE = 'PREREG_step6b_confirmation.md'
if open(PRE).read().count('\n## ') < 8: sys.exit('FATAL: prereg incomplete')
if subprocess.run('git diff --quiet -- %s' % PRE, shell=True).returncode != 0:
    sys.exit('FATAL: prereg uncommitted')
print('[prereg] committed')

def sha(p):
    m = hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): m.update(c)
    return m.hexdigest()
def jac(a,b):
    i = len(np.intersect1d(a,b)); return i/(len(a)+len(b)-i)
def cs_ref(M, k=K):
    i = np.arange(0,k+1); p = hypergeom.pmf(i,M,k,k)
    return float(np.sum(p*i/(2*k-i)))
def welch(x,y):
    d = y.mean()-x.mean(); vx,vy,n = x.var(ddof=1),y.var(ddof=1),N_SEEDS
    se = np.sqrt(vx/n+vy/n)
    df = (vx/n+vy/n)**2/((vx/n)**2/(n-1)+(vy/n)**2/(n-1))
    h = tdist.ppf(ALPHA, df)*se
    return float(d), float(d-h), float(d+h), float(df), float(se)
def perm_p(x,y):
    a = np.concatenate([x,y]); obs = abs(y.mean()-x.mean()); hits=tot=0
    for idx in combinations(range(len(a)), N_SEEDS):
        m = np.zeros(len(a),bool); m[list(idx)]=True
        if abs(a[~m].mean()-a[m].mean()) >= obs-1e-12: hits+=1
        tot+=1
    return hits/tot, tot
def energy_u(X,Y):
    return float(2*cdist(X,Y).mean() - pdist(X).mean() - pdist(Y).mean())

res, verdicts = {}, {}
for sp in SPLITS:
    P1, P2, ref = {}, {}, None
    for arm in ARMS:
        j, e = [], []
        for s in range(N_SEEDS):
            f = os.path.join(RES,'vectors_s6b_%s_%s_seed_%d.npz'%(sp,arm,s))
            v = np.load(f, allow_pickle=True)
            true,B,C = np.asarray(v['true']),np.asarray(v['B']),np.asarray(v['C'])
            cond = np.array([str(x) for x in v['cond']])
            if ref is None: ref=(cond,true,C)
            elif not np.array_equal(cond,ref[0]): sys.exit('FATAL: cond order differs in '+sp)
            vals=[]
            for i in range(len(true)):
                d = np.argsort(np.abs(true[i]-C[i]))[-K:]
                p = B[i]-C[i]
                if np.abs(p).std()>1e-12: vals.append(jac(d,np.argsort(np.abs(p))[-K:]))
            if len(vals)!=len(true): sys.exit('FATAL: %d of %d conditions scored'%(len(vals),len(true)))
            j.append(np.mean(vals))
            pc = np.load(f.replace('.npz','_pc.npz'), allow_pickle=True)
            e.append(np.mean([energy_u(np.asarray(a,dtype=np.float64),
                                       np.asarray(b,dtype=np.float64))
                              for a,b in zip(pc['pred'],pc['true'])]))
        P1[arm], P2[arm] = np.array(j), np.array(e)
    M_s = int(((ref[1]-ref[2])!=0).any(axis=0).sum()); c_s = cs_ref(M_s)
    d,lo,hi,df,se = welch(P1['ddim'],P1['fm_gaussian'])
    p,na = perm_p(P1['ddim'],P1['fm_gaussian'])
    d2,lo2,hi2,_,_ = welch(P2['ddim'],P2['fm_gaussian'])
    resolved = lo>0
    print('\n'+'='*72); print('%s  (%d conditions, M_s %d, c_s %.6f)'%(sp,len(ref[0]),M_s,c_s)); print('='*72)
    for a in ARMS:
        print('  %-13s P1 %s | mean %.5f  SD %.5f'%(a,' '.join('%.5f'%x for x in P1[a]),P1[a].mean(),P1[a].std(ddof=1)))
    print('  GATED  fm - ddim  %+.5f  97.5%% CI [%+.5f, %+.5f]  df %.2f  p %.4f  -> %s'
          % (d,lo,hi,df,p,'resolved above zero' if resolved else ('resolved below zero' if hi<0 else 'includes zero')))
    for a in ARMS:
        print('  %-13s P2 %s | mean %.5f'%(a,' '.join('%.5f'%x for x in P2[a]),P2[a].mean()))
    print('  reported, NOT gated: P2 fm - ddim %+.5f  CI [%+.5f, %+.5f]  (lower is better)'%(d2,lo2,hi2))
    res[sp] = dict(p1={a:list(map(float,P1[a])) for a in ARMS},
                   p2={a:list(map(float,P2[a])) for a in ARMS},
                   gated=dict(delta=d,ci=[lo,hi],welch_df=df,se=se,perm_p=p,n_alloc=na,
                              resolved_above_zero=bool(resolved)),
                   p2_contrast=dict(delta=d2,ci=[lo2,hi2]),
                   n_conditions=len(ref[0]),M_s=M_s,c_s=c_s)
    verdicts[sp] = bool(resolved)

both = all(verdicts.values()); any_neg = any(res[s]['gated']['ci'][1] < 0 for s in SPLITS)
v = ('FULL CONFIRMATION' if both else
     'DIRECTIONAL CONTRADICTION' if any_neg else
     'PARTIAL CONFIRMATION' if any(verdicts.values()) else 'NO CONFIRMATION')
print('\n'+'='*72); print('PRE-REGISTERED VERDICT: %s'%v)
print('  resolved above zero: %s'%{k:verdicts[k] for k in SPLITS})
print('  Step 6 development effect was +0.03102; MDE stated in advance 0.01727.')
print('  Categories 3 and 4 do not establish that flow matching fails to improve')
print('  DE-overlap, only that this design did not resolve it. No equivalence claimed.')
print('='*72)
g=lambda c: subprocess.run(c,shell=True,capture_output=True,text=True).stdout.strip()
json.dump(dict(verdict=v,per_split=res,prereg=PRE,alpha_two_sided=0.025,
               commit=g('git rev-parse HEAD'),script_sha256=sha(os.path.abspath(__file__))),
          open(OUT+'.tmp','w'),indent=2,allow_nan=False)
os.replace(OUT+'.tmp',OUT); print('\nwrote %s'%OUT)
