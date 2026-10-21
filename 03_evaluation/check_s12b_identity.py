#!/usr/bin/env python3
"""Step 12b identity check, PREREG_step12b_confirmation.md section 3.

Same check as check_s12_identity.py, per split: the guided code path at w=1
must reproduce the unguided path to within 1e-6 on condition means for every
seed BEFORE any w>1 sample is drawn on that split. Writes
provenance/s12b_identity_<split>_seed_<k>.json with "pass": true/false;
eval_step12b.sh tasks with w>1 refuse to start unless their split's and seed's
file says true.
"""
import os, sys, json, hashlib
import numpy as np

RES = 'leakage_results'; TOL = 1e-6; N_SEEDS = 5
SPLITS = ('drug_split_0', 'drug_split_2')
def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): m.update(c)
    return m.hexdigest()

all_ok = True
for split in SPLITS:
    for k in range(N_SEEDS):
        na, nb = 's12b_%s_wnone_seed_%d' % (split, k), 's12b_%s_w1_seed_%d' % (split, k)
        fa, fb = [os.path.join(RES, 'vectors_%s.npz' % n) for n in (na, nb)]
        pa, pb = [os.path.join('provenance', 's12bev_%s.json' % n) for n in (na, nb)]
        for q in (fa, fb, pa, pb):
            if not os.path.isfile(q): sys.exit('FATAL: missing %s' % q)
        ja, jb = json.load(open(pa)), json.load(open(pb))
        if ja.get('guidance_w') is not None: sys.exit('FATAL: %s provenance guidance_w=%r, expected None' % (na, ja.get('guidance_w')))
        if jb.get('guidance_w') is None or abs(float(jb['guidance_w']) - 1.0) > 1e-12:
            sys.exit('FATAL: %s provenance guidance_w=%r, expected 1.0' % (nb, jb.get('guidance_w')))
        if ja.get('checkpoint_sha256') != jb.get('checkpoint_sha256'):
            sys.exit('FATAL: %s seed %d compares different checkpoints' % (split, k))
        if ja.get('split') != split or jb.get('split') != split:
            sys.exit('FATAL: %s seed %d provenance split mismatch' % (split, k))
        for f, j in ((fa, ja), (fb, jb)):
            if j.get('vectors_sha256') != sha(f): sys.exit('FATAL: %s hash differs from its provenance' % f)
        A, B = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
        if set(A.files) != set(B.files): sys.exit('FATAL: %s seed %d key sets differ' % (split, k))
        ca = np.array([str(x) for x in A['cond']]); cb = np.array([str(x) for x in B['cond']])
        if not np.array_equal(ca, cb): sys.exit('FATAL: %s seed %d condition order differs' % (split, k))
        worst = {}
        for key in A.files:
            x, y = np.asarray(A[key]), np.asarray(B[key])
            if x.dtype.kind in 'fc' and y.dtype.kind in 'fc':
                if x.shape != y.shape: sys.exit('FATAL: %s seed %d key %s shape differs' % (split, k, key))
                worst[key] = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
        PA = np.load(fa.replace('.npz', '_pc.npz'), allow_pickle=True); PB = np.load(fb.replace('.npz', '_pc.npz'), allow_pickle=True)
        pc_worst = 0.0
        for u, v in zip(PA['pred'], PB['pred']):
            u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
            if u.shape != v.shape: sys.exit('FATAL: %s seed %d per-cell shapes differ' % (split, k))
            pc_worst = max(pc_worst, float(np.abs(u - v).max()))
        sampled = max(worst.get(key, 0.0) for key in ('A', 'B'))
        fixed_identical = all(worst.get(key, 0.0) == 0.0 for key in ('true', 'C'))
        ok = bool(sampled < TOL and fixed_identical)
        all_ok &= ok
        rec = dict(split=split, seed=k, tolerance=TOL, max_abs_diff_condition_means=worst,
                   max_abs_diff_sampled=sampled, true_and_C_bitwise_identical=fixed_identical,
                   per_cell_max_abs_diff=pc_worst, unguided=na, guided_w1=nb,
                   checkpoint_sha256=ja.get('checkpoint_sha256'), **{'pass': ok})
        json.dump(rec, open('provenance/s12b_identity_%s_seed_%d.json' % (split, k), 'w'), indent=2)
        print('%s seed %d  %s  condition-mean max|diff| A/B %.2e  (true, C identical: %s)  per-cell max|diff| %.2e'
              % (split, k, 'PASS' if ok else 'FAIL', sampled, fixed_identical, pc_worst))
print('IDENTITY CHECK: %s' % ('PASS for all seeds on both splits; w>1 tasks may run' if all_ok else 'FAILED - do not run w>1 tasks'))
sys.exit(0 if all_ok else 1)
