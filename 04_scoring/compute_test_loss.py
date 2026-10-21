#!/usr/bin/env python3
"""EXPLORATORY: the diffusion objective evaluated on held-out compounds.

No test loss exists in the stored artifacts - the evaluation harness scores by
sampling and DE-overlap, never by the training objective. Intermediate model
weights were not kept either, so only ONE value per checkpoint is recoverable,
not a curve.

Each arm's loss is computed with ITS OWN objective: DDIM on epsilon-prediction,
flow matching on velocity. The two are therefore NOT comparable to each other in
value - only each arm's train-versus-test gap is interpretable.
"""
import os, sys, json
import numpy as np, torch as th, scanpy as sc
sys.argv=['x']
from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
from Squidiff.scrna_datasets import prepared_data
import sample_squidiff

DATA='/scratch/ni2071/Diffusion/sciplex/datasets_corrected'
OUT='leakage_results/test_loss_v2.json'
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
TRAIN_FINAL={'ddim':0.0116,'fm_gaussian':0.0515}   # last logged, step 90,000

def batch_loss(ck, split, part, n_batches=40, bs=64, seed=0):
    s=sample_squidiff.sampler(ck, 200, 200, True)
    # The OBJECTIVE is not recoverable from the weights - Step 6 recorded this:
    # the architectures are identical and arm identity lives in a sidecar. The
    # sampler detects adaLN from the state dict but defaults to the DDIM
    # objective, so a flow-matching checkpoint was previously scored with
    # q_sample and epsilon-prediction, giving 4.19 against a logged 0.0515.
    if 'fm_gaussian' in ck:
        from Squidiff.flow_matching import FlowMatchingWrapper
        s.diffusion = FlowMatchingWrapper(source='gaussian', ot=False)
        print('  [objective] FlowMatchingWrapper(source=gaussian)')
    else:
        print('  [objective] DDIM (from the sampler)')
    dl=prepared_data(data_dir='%s/sci_plex_%s_%s.h5ad'%(DATA,part,split),
                     control_data_dir='%s/sci_plex_%s_%s_control.h5ad'%(DATA,part,split),
                     batch_size=bs, use_drug_structure=True, comb_num=1)
    th.manual_seed(seed); losses=[]
    for i,b in enumerate(dl):
        if i>=n_batches: break
        x=b['feature'].float().cuda()
        kw={'drug_dose':b['drug_dose'].float().cuda(),
            'control_feature':b['control_feature'].float().cuda()}
        t=th.randint(0, 1000, (x.shape[0],), device=x.device).long()
        with th.no_grad():
            out=s.diffusion.training_losses(s.model, x, t, model_kwargs=kw)
        losses.append(float(out['loss'].mean()))
    return float(np.mean(losses)), float(np.std(losses))

res={}
for split in ('drug_split_0','drug_split_2'):
    for arm in ('ddim','fm_gaussian'):
        for seed in range(5):
            ck='step6b/%s_%s/seed_%d/model_0.9999.pt' % (split,arm,seed)
            if not os.path.isfile(ck): continue
            tr,_=batch_loss(ck,split,'train')
            te,_=batch_loss(ck,split,'test')
            k='%s_%s_seed%d'%(split,arm,seed)
            res[k]=dict(train=tr,test=te,ratio=te/tr if tr>0 else np.nan)
            print('%-34s train %.5f  test %.5f  ratio %.3f' % (k,tr,te,te/tr))
json.dump(dict(analysis='EXPLORATORY, not pre-registered',
               note='one value per checkpoint; intermediate weights were not kept so no '
                    'curve is recoverable. Each arm uses its own objective and the two '
                    'arms are not comparable in value.',
               train_final_logged=TRAIN_FINAL, per_checkpoint=res),
          open(OUT,'w'),indent=2)
print('\nwrote %s' % OUT)
