#!/usr/bin/env python3
"""Step 12 identity check, PREREG_step12_guidance.md section 3.

The guided code path at w=1 computes v_null + 1*(v_drug - v_null), which equals
v_drug up to floating-point rounding. Its stored condition-mean vectors must
therefore reproduce the unguided code path's vectors to within 1e-6 for every
seed BEFORE any w>1 sample is drawn. A wiring fault (wrong embedding, wrong
split of the [z | z_null] pair, wrong sign) produces differences of order 1e-2.

Compares, per seed: provenance (same checkpoint hash; guidance_w None vs 1.0),
condition order, every float array in the condition-mean file (A, B are the
sampled quantities; true and C do not depend on sampling and must be bitwise
identical), and the per-cell predictions. Writes
provenance/s12_identity_seed_<k>.json with "pass": true/false; eval_step12.sh
tasks 15-44 refuse to start unless their seed's file says true.
"""
import os, sys, json, hashlib
import numpy as np

RES = 'leakage_results'; TOL = 1e-6; N_SEEDS = 5
def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): m.update(c)
    return m.hexdigest()

all_ok = True
for k in range(N_SEEDS):
    na, nb = 's12_cfg_wnone_seed_%d' % k, 's12_cfg_w1_seed_%d' % k
    fa, fb = [os.path.join(RES, 'vectors_%s.npz' % n) for n in (na, nb)]
    pa, pb = [os.path.join('provenance', 's12ev_%s.json' % n) for n in (na, nb)]
    for q in (fa, fb, pa, pb):
        if not os.path.isfile(q): sys.exit('FATAL: missing %s' % q)
    ja, jb = json.load(open(pa)), json.load(open(pb))
    if ja.get('guidance_w') is not None: sys.exit('FATAL: %s provenance guidance_w=%r, expected None' % (na, ja.get('guidance_w')))
    if jb.get('guidance_w') is None or abs(float(jb['guidance_w']) - 1.0) > 1e-12:
        sys.exit('FATAL: %s provenance guidance_w=%r, expected 1.0' % (nb, jb.get('guidance_w')))
    if ja.get('checkpoint_sha256') != jb.get('checkpoint_sha256'):
        sys.exit('FATAL: seed %d compares different checkpoints' % k)
    for f, j in ((fa, ja), (fb, jb)):
        if j.get('vectors_sha256') != sha(f): sys.exit('FATAL: %s hash differs from its provenance' % f)
    A, B = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
    if set(A.files) != set(B.files): sys.exit('FATAL: seed %d key sets differ: %s vs %s' % (k, sorted(A.files), sorted(B.files)))
    ca = np.array([str(x) for x in A['cond']]); cb = np.array([str(x) for x in B['cond']])
    if not np.array_equal(ca, cb): sys.exit('FATAL: seed %d condition order differs' % k)
    worst = {}
    for key in A.files:
        x, y = np.asarray(A[key]), np.asarray(B[key])
        if x.dtype.kind in 'fc' and y.dtype.kind in 'fc':
            if x.shape != y.shape: sys.exit('FATAL: seed %d key %s shape differs' % (k, key))
            worst[key] = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
    PA, PB = np.load(fa.replace('.npz', '_pc.npz'), allow_pickle=True), np.load(fb.replace('.npz', '_pc.npz'), allow_pickle=True)
    pc_worst = 0.0
    for u, v in zip(PA['pred'], PB['pred']):
        u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        if u.shape != v.shape: sys.exit('FATAL: seed %d per-cell shapes differ' % k)
        pc_worst = max(pc_worst, float(np.abs(u - v).max()))
    sampled = max(worst.get(key, 0.0) for key in ('A', 'B'))
    fixed_identical = all(worst.get(key, 0.0) == 0.0 for key in ('true', 'C'))
    ok = bool(sampled < TOL and fixed_identical)
    all_ok &= ok
    rec = dict(seed=k, tolerance=TOL, max_abs_diff_condition_means=worst,
               max_abs_diff_sampled=sampled, true_and_C_bitwise_identical=fixed_identical,
               per_cell_max_abs_diff=pc_worst, unguided=na, guided_w1=nb,
               checkpoint_sha256=ja.get('checkpoint_sha256'), **{'pass': ok})
    json.dump(rec, open('provenance/s12_identity_seed_%d.json' % k, 'w'), indent=2)
    print('seed %d  %s  condition-mean max|diff| A/B %.2e  (true, C identical: %s)  per-cell max|diff| %.2e'
          % (k, 'PASS' if ok else 'FAIL', sampled, fixed_identical, pc_worst))
print('IDENTITY CHECK: %s' % ('PASS for all %d seeds; w>1 tasks may run' % N_SEEDS if all_ok else 'FAILED - do not run w>1 tasks'))
sys.exit(0 if all_ok else 1)
