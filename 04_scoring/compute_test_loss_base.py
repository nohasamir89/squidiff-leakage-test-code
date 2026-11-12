#!/usr/bin/env python3
"""
compute_test_loss_base.py - the training objective of the ORIGINAL model evaluated on held-out compounds
(sci-Plex, splits 0 and 2, five seeds each), by exactly the procedure of compute_test_loss.py (commit f67c8ce).

Purpose
-------
Results 1 ("The failure is not due to overfitting") rests on two sci-Plex models with adaptive normalisation
(leakage_results/test_loss_v2.json, the Step 6b checkpoints); the original model itself was not evaluated. This
script evaluates the original model's own training objective (DDIM epsilon-prediction mean squared error, as in
training) on 2,560 cells of held-out compounds and, by the same procedure, on 2,560 training cells, for the ten
original-model checkpoints that the Step 4 replication scored. EXPLORATORY, like test_loss_v2.json: it changes no
pre-registered result and is reported as a description of the objective, not as a test.

Logic
-----
0. Guards (the script stops on any failure and writes nothing):
   - the output record does not exist yet (no overwrite);
   - this script and compute_test_loss.py are tracked by git and unmodified (results only from committed code);
   - a CUDA device is present (the procedure being reproduced ran on GPU).
1. Checkpoint identity. No checkpoint path is typed in. For "<split>/base/seed<k>" and "<split>/adaln/seed<k>",
   leakage_results/replication_analysis.json records a checkpoint_sha256 and the evaluation provenance file of that
   run; the provenance names the checkpoint the evaluation read and its sha256. The path is taken from the
   provenance, and the file's sha256 must equal both records: these are exactly the checkpoints whose evaluations the
   replication and Step 14 report (the base checkpoints are in <ROOT>/replication_base/, per the provenance).
   History: the first run (job 19298983, 11 Nov 2026) used a typed path, replication_failed_16930893_boolparse/, and
   stopped at this check because those files differ from the recorded ones; nothing was written.
2. Architecture. Each original-model checkpoint is compared with the replication's adaptive-normalisation checkpoint
   of the same split and seed (found and checked as in step 1). The adaptive-normalisation layers are the keys present
   only in that checkpoint; the original-model checkpoint must contain none of them (that set must be non-empty, so
   the two really differ) and must have fewer parameters. Keys present only in the original checkpoint (for example
   affine weights of normalisation layers that adaptive normalisation replaces) are allowed and recorded, as are
   shared keys whose shapes differ; the sha256 check above already fixes which file is scored, so this step only
   confirms that the file is not an adaptive-normalisation model. Key and parameter counts are printed and recorded.
3. Reproduction gate. batch_loss below is copied verbatim from compute_test_loss.py (same sampler call, data files,
   40 batches of 64 cells, torch.manual_seed(0) before the loader is iterated, diffusion time uniform over 1,000
   steps, no gradients). Before any original-model checkpoint is scored, it recomputes the first entry of the
   committed record (drug_split_0_ddim_seed0: Step 6b adaptive normalisation + DDIM, seed 0) and must reproduce its
   train and test values within 2e-5 (the values are about 0.01; GPU float reductions can differ in the last digits
   between devices). A failure means the procedure or the code has changed since f67c8ce; the script stops.
4. For each split (drug_split_0, drug_split_2) and seed (0-4): training loss, held-out loss, ratio held-out/training.
5. Record leakage_results/test_loss_base.json: per checkpoint (train, test, ratio, sha256, key and parameter counts);
   per split the mean and SD over seeds of train and test, the mean ratio and its range over seeds; the gate values;
   the environment (git commit, torch, device). A table is printed at the end.

Run on a GPU node from the Squidiff_ms worktree root, through run_test_loss_base.sh, after both files are committed.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch as th

sys.argv = ['x']                      # as in compute_test_loss.py: the Squidiff modules parse sys.argv on import
from Squidiff.scrna_datasets import prepared_data   # noqa: E402
import sample_squidiff                               # noqa: E402

# ------------------------------------------------------------------ paths
ROOT = '/scratch/ni2071/Diffusion/Squidiff_git/Squidiff'          # main checkout that holds the checkpoints
DATA = '/scratch/ni2071/Diffusion/sciplex/datasets_corrected'      # identical to compute_test_loss.py
OUT = 'leakage_results/test_loss_base.json'
REF = 'leakage_results/test_loss_v2.json'                          # committed record of compute_test_loss.py
REPL = 'leakage_results/replication_analysis.json'                 # Step 4 replication record (checkpoint hashes)
GATE_KEY = 'drug_split_0_ddim_seed0'
GATE_CK = ROOT + '/step6b/drug_split_0_ddim/seed_0/model_0.9999.pt'  # the path compute_test_loss.py read (cwd ROOT)
GATE_TOL = 2e-5
SPLITS = ('drug_split_0', 'drug_split_2')
SEEDS = range(5)


def fail(msg):
    sys.exit('FATAL: ' + msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(['git'] + list(args), capture_output=True, text=True).stdout.strip()


def prefixes(keys):
    """First two name components of each state-dict key, to show which layers differ (at most 8, sorted)."""
    return sorted({'.'.join(k.split('.')[:2]) for k in keys})[:8]


def all_keys(obj, prefix):
    """Every dict key anywhere in a nested JSON object that starts with `prefix` (to report what the record holds)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith(prefix):
                out.append(k)
            out += all_keys(v, prefix)
    elif isinstance(obj, list):
        for v in obj:
            out += all_keys(v, prefix)
    return out


def recorded_checkpoint(repl, key):
    """Path and sha256 of the checkpoint that the replication record lists under `key` (e.g. 'drug_split_0/base/seed0').
    The record gives the checkpoint's sha256 and the evaluation's provenance file; the provenance gives the path the
    evaluation read and the same sha256. Both records must agree, and the file on disk must have that sha256."""
    rec = find_key(repl, key)
    if not isinstance(rec, dict) or 'checkpoint_sha256' not in rec or 'eval_provenance' not in rec:
        split = key.split('/')[0]
        fail('%s has no checkpoint_sha256 / eval_provenance under %r; keys for this split: %s'
             % (REPL, key, sorted(set(all_keys(repl, split + '/')))))
    prov_path = rec['eval_provenance']
    if not os.path.isfile(prov_path):
        fail('missing provenance file %s' % prov_path)
    prov = json.load(open(prov_path))
    if 'checkpoint' not in prov or 'checkpoint_sha256' not in prov:
        fail('%s does not name the checkpoint and its sha256' % prov_path)
    if prov['checkpoint_sha256'] != rec['checkpoint_sha256']:
        fail('%s: provenance sha256 %s differs from the replication record %s'
             % (key, prov['checkpoint_sha256'], rec['checkpoint_sha256']))
    path = prov['checkpoint']
    if not os.path.isfile(path):
        fail('missing checkpoint %s (named in %s)' % (path, prov_path))
    h = sha256(path)
    if h != rec['checkpoint_sha256']:
        fail('%s sha256 %s differs from the replication record %s' % (path, h, rec['checkpoint_sha256']))
    return path, h


def find_key(obj, key):
    """Return the value stored under `key` anywhere in a nested JSON object (first match), else None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


# ------------------------------------------------------------------ the loss: verbatim from compute_test_loss.py
def batch_loss(ck, split, part, n_batches=40, bs=64, seed=0):
    s = sample_squidiff.sampler(ck, 200, 200, True)
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
    dl = prepared_data(data_dir='%s/sci_plex_%s_%s.h5ad' % (DATA, part, split),
                       control_data_dir='%s/sci_plex_%s_%s_control.h5ad' % (DATA, part, split),
                       batch_size=bs, use_drug_structure=True, comb_num=1)
    th.manual_seed(seed)
    losses = []
    for i, b in enumerate(dl):
        if i >= n_batches:
            break
        x = b['feature'].float().cuda()
        kw = {'drug_dose': b['drug_dose'].float().cuda(),
              'control_feature': b['control_feature'].float().cuda()}
        t = th.randint(0, 1000, (x.shape[0],), device=x.device).long()
        with th.no_grad():
            out = s.diffusion.training_losses(s.model, x, t, model_kwargs=kw)
        losses.append(float(out['loss'].mean()))
    return float(np.mean(losses)), float(np.std(losses))


def main():
    # ---------------------------------------------------------------- 0. guards
    if os.path.exists(OUT):
        fail('%s exists; nothing is overwritten' % OUT)
    for f in ('compute_test_loss_base.py', 'compute_test_loss.py', REF, REPL):
        if not os.path.isfile(f):
            fail('missing %s (run from the Squidiff_ms worktree root)' % f)
    for f in ('compute_test_loss_base.py', 'compute_test_loss.py'):
        if subprocess.run(['git', 'ls-files', '--error-unmatch', f], capture_output=True).returncode != 0:
            fail('%s is not tracked by git; commit it before running' % f)
        if git('status', '--porcelain', '--', f):
            fail('%s has uncommitted changes; commit it before running' % f)
    if not th.cuda.is_available():
        fail('no CUDA device')
    device = th.cuda.get_device_name(0)
    print('[env] commit %s  torch %s  device %s' % (git('rev-parse', '--short', 'HEAD'), th.__version__, device))

    ref = json.load(open(REF))['per_checkpoint']
    repl = json.load(open(REPL))

    # ---------------------------------------------------------------- 1-2. identity and architecture of every checkpoint
    meta = {}
    for split in SPLITS:
        for seed in SEEDS:
            ck, h = recorded_checkpoint(repl, '%s/base/seed%d' % (split, seed))        # the original model
            ad, h_ad = recorded_checkpoint(repl, '%s/adaln/seed%d' % (split, seed))    # its adaLN twin
            sb = th.load(ck, map_location='cpu')
            sa = th.load(ad, map_location='cpu')
            kb, ka = set(sb), set(sa)
            only_a = ka - kb                                  # the adaptive-normalisation layers
            only_b = kb - ka                                  # e.g. affine norm weights that adaLN replaces (allowed)
            if not only_a:
                fail('%s: no key is specific to the adaLN checkpoint; the two models do not differ' % ck)
            nb = int(sum(v.numel() for v in sb.values()))
            na = int(sum(v.numel() for v in sa.values()))
            if not nb < na:
                fail('%s: %d parameters, not fewer than the adaLN checkpoint (%d)' % (ck, nb, na))
            shape_diff = sorted(k for k in kb & ka if tuple(sb[k].shape) != tuple(sa[k].shape))
            meta[(split, seed)] = dict(checkpoint=ck, sha256=h, n_keys=len(kb), n_params=nb,
                                       adaln_checkpoint=ad, adaln_sha256=h_ad,
                                       adaln_n_keys=len(ka), adaln_n_params=na,
                                       adaln_only_keys=len(only_a), adaln_only_prefixes=prefixes(only_a),
                                       base_only_keys=len(only_b), base_only_prefixes=prefixes(only_b),
                                       shared_keys_shape_differs=len(shape_diff))
            print('[identity] %s seed %d  sha256 = replication record  params %d < %d  adaLN-only keys %d '
                  '(%s)  base-only keys %d  shared keys with other shape %d'
                  % (split, seed, nb, na, len(only_a), ', '.join(prefixes(only_a)[:3]), len(only_b),
                     len(shape_diff)))

    # ---------------------------------------------------------------- 3. reproduction gate
    if GATE_KEY not in ref:
        fail('%s has no entry %s' % (REF, GATE_KEY))
    if not os.path.isfile(GATE_CK):
        fail('missing gate checkpoint %s' % GATE_CK)
    print('[gate] recomputing %s from %s' % (GATE_KEY, GATE_CK))
    g_tr, _ = batch_loss(GATE_CK, 'drug_split_0', 'train')
    g_te, _ = batch_loss(GATE_CK, 'drug_split_0', 'test')
    d_tr, d_te = abs(g_tr - ref[GATE_KEY]['train']), abs(g_te - ref[GATE_KEY]['test'])
    print('[gate] train %.8f (record %.8f, diff %.2e)  test %.8f (record %.8f, diff %.2e)'
          % (g_tr, ref[GATE_KEY]['train'], d_tr, g_te, ref[GATE_KEY]['test'], d_te))
    if d_tr > GATE_TOL or d_te > GATE_TOL:
        fail('reproduction gate failed (tolerance %.0e): the procedure differs from compute_test_loss.py' % GATE_TOL)
    print('[gate] PASS')

    # ---------------------------------------------------------------- 4. the original model
    per = {}
    for split in SPLITS:
        for seed in SEEDS:
            ck = meta[(split, seed)]['checkpoint']
            print('[score] %s base seed %d' % (split, seed))
            tr, tr_sd = batch_loss(ck, split, 'train')
            te, te_sd = batch_loss(ck, split, 'test')
            k = '%s_base_seed%d' % (split, seed)
            per[k] = dict(train=tr, test=te, ratio=te / tr if tr > 0 else float('nan'),
                          train_batch_sd=tr_sd, test_batch_sd=te_sd, **meta[(split, seed)])
            print('%-28s train %.5f  test %.5f  ratio %.3f' % (k, tr, te, te / tr))

    # ---------------------------------------------------------------- 5. summary and record
    summary = {}
    for split in SPLITS:
        rows = [per['%s_base_seed%d' % (split, s)] for s in SEEDS]
        tr = np.array([r['train'] for r in rows])
        te = np.array([r['test'] for r in rows])
        ra = np.array([r['ratio'] for r in rows])
        # both SD conventions are recorded (ddof 1 = sample SD over the five seeds; ddof 0 = numpy's default), so a
        # table can use whichever convention the earlier record used without recomputing
        summary[split] = dict(train_mean=float(tr.mean()), train_sd=float(tr.std(ddof=1)),
                              train_sd_ddof0=float(tr.std(ddof=0)),
                              test_mean=float(te.mean()), test_sd=float(te.std(ddof=1)),
                              test_sd_ddof0=float(te.std(ddof=0)),
                              ratio_mean=float(ra.mean()), ratio_min=float(ra.min()), ratio_max=float(ra.max()))
    rec = dict(analysis='EXPLORATORY, not pre-registered',
               note=('The original model (no adaptive normalisation, DDIM objective) scored with the procedure of '
                     'compute_test_loss.py (f67c8ce): 40 batches of 64 cells, seed 0, uniform diffusion time, from the '
                     'training and from the test file of each split. One value per checkpoint; ratio = test / train. '
                     'The checkpoints are those of the Step 4 replication (sha256 equal to replication_analysis.json).'),
               gate=dict(key=GATE_KEY, checkpoint=GATE_CK, train=g_tr, test=g_te,
                         record_train=ref[GATE_KEY]['train'], record_test=ref[GATE_KEY]['test'], tolerance=GATE_TOL),
               environment=dict(commit=git('rev-parse', 'HEAD'), torch=th.__version__, device=device),
               per_checkpoint=per, summary=summary)
    with open(OUT, 'w') as f:
        json.dump(rec, f, indent=2)

    print('\n%-14s %-22s %-22s %-10s %s' % ('split', 'train mean (SD)', 'test mean (SD)', 'ratio', 'range over seeds'))
    for split in SPLITS:
        s = summary[split]
        print('%-14s %.5f (%.5f)      %.5f (%.5f)      %.3f      %.3f - %.3f'
              % (split, s['train_mean'], s['train_sd'], s['test_mean'], s['test_sd'],
                 s['ratio_mean'], s['ratio_min'], s['ratio_max']))
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
