#!/usr/bin/env python3
"""Is the linear network converging at all?

Compares TRAINING loss against ridge's closed-form solution on one fold. If the
network's training MSE stays well above ridge's, it has not converged and the
Step 8 OPTIMISATION CONFOUND is a convergence failure, not a regularisation
choice. Weight decay is set to ZERO so nothing but optimisation is tested.
"""
import os, sys, json, subprocess
import numpy as np, scanpy as sc, torch as th, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from Squidiff.scrna_datasets import Drug_dose_encoder

D='/scratch/ni2071/Diffusion/sciplex/datasets_corrected'
OUT='leakage_results/step8_convergence.json'
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
v=np.load('leakage_results/vectors_step6_fm_gaussian_seed_0.npz',allow_pickle=True)
Y=(np.asarray(v['true'])-np.asarray(v['C'])).astype(np.float32)
cond=np.array([str(x) for x in v['cond']])
ad=sc.read_h5ad('%s/sci_plex_test_drug_split_1.h5ad'%D); g=ad.obs['Group'].astype(str).values
smi={c:ad.obs['SMILES'].astype(str).values[g==c][0] for c in cond}
dose={c:float(np.median(ad.obs['dose'].astype(float).values[g==c])) for c in cond}
line={c:ad.obs['cell_name'].astype(str).values[g==c][0] for c in cond}
drugs=np.array([smi[c] for c in cond])
FP=np.array([Drug_dose_encoder([smi[c]],[dose[c]],comb_num=1)[0] for c in cond],dtype=np.float32)
lv=sorted(set(line.values()))
L=np.array([[float(line[c]==u) for u in lv] for c in cond],dtype=np.float32)
X=np.hstack([FP,L,np.concatenate([FP*L[:,[k]] for k in range(len(lv))],axis=1)]).astype(np.float32)
a,b=next(iter(GroupKFold(n_splits=5).split(X,Y,groups=drugs)))
R=Ridge(alpha=1.0).fit(X[a],Y[a])
r_tr=float(((R.predict(X[a])-Y[a])**2).mean()); r_te=float(((R.predict(X[b])-Y[b])**2).mean())
print('ridge          train MSE %.6f   test MSE %.6f' % (r_tr, r_te))
print('              (weight decay is ZERO below: only optimisation is tested)\n')
xa,ya,xb,yb=th.tensor(X[a]),th.tensor(Y[a]),th.tensor(X[b]),th.tensor(Y[b])
keep=xa.std(0)>1e-9
xa,xb=xa[:,keep],xb[:,keep]
mu,sd=xa.mean(0,keepdim=True),xa.std(0,keepdim=True)+1e-6
za,zb=(xa-mu)/sd,(xb-mu)/sd
rec={'ridge_train':r_tr,'ridge_test':r_te,'runs':{}}
for lr in (1e-3,1e-2,1e-1):
    th.manual_seed(0); m=nn.Linear(int(keep.sum()),Y.shape[1])
    opt=th.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.0)
    hist=[]
    for e in range(20001):
        opt.zero_grad(); l=((m(za)-ya)**2).mean(); l.backward(); opt.step()
        if e % 2000 == 0 or e == 20000:
            with th.no_grad(): t=float(((m(zb)-yb)**2).mean())
            hist.append((e,float(l),t))
            print('  lr %-6g epoch %6d  train %.6f  test %.6f  train/ridge %.2fx'
                  % (lr,e,l,t,l/r_tr))
    rec['runs']['lr%g'%lr]=hist
    print()
g=lambda c: subprocess.run(c,shell=True,capture_output=True,text=True).stdout.strip()
rec['commit']=g('git rev-parse HEAD')
json.dump(rec,open(OUT,'w'),indent=2)
print('wrote %s' % OUT)
