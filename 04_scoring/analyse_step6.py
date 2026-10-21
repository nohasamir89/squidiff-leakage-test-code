"""PRE-REGISTERED Step 6 analysis. PREREG_step6_flow_matching.md.

TWO CO-PRIMARY endpoints, each gated at the 97.5% level (Bonferroni over two):
  P1 raw DE-overlap Jaccard, arithmetic mean over conditions
  P2 energy distance between predicted and true perturbed populations

Two gated contrasts per endpoint: B minus A (objective) and C minus B (source).
Unpaired Welch, five per arm, plus the exhaustive 252-allocation permutation.

P1's MDE at 80% power was stated in advance as ~0.0173. NO prior SD exists for
P2, so its power was disclosed in advance as UNKNOWN; an unresolved P2 result
must not be read as evidence of no effect.
"""
import os, sys, json, hashlib, subprocess
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr, hypergeom, t as tdist

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, 'leakage_results')
PROV = os.path.join(ROOT, 'provenance')
OUT  = os.path.join(RES, 'step6_analysis.json')
ARMS = ('ddim', 'fm_gaussian', 'fm_control_ot')
K, N_SEEDS, N_COND, ALPHA = 50, 5, 443, 0.975

if os.path.exists(OUT): sys.exit('FATAL: %s exists; refusing to overwrite' % OUT)
PRE = os.path.join(ROOT, 'PREREG_step6_flow_matching.md')
n_sec = open(PRE).read().count('\n## ')
if n_sec < 10: sys.exit('FATAL: prereg has %d sections, expected at least 10' % n_sec)
if subprocess.run('git diff --quiet -- PREREG_step6_flow_matching.md',
                  shell=True).returncode != 0:
    sys.exit('FATAL: prereg has uncommitted changes')
print('[prereg] %d sections, committed' % n_sec)

def sha(p):
    m = hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): m.update(c)
    return m.hexdigest()

def jac(a, b):
    i = len(np.intersect1d(a, b)); return i / (len(a) + len(b) - i)

def cs_reference(M, k=K):
    if M <= k: sys.exit('FATAL: M_s=%d must exceed K' % M)
    i = np.arange(0, k+1); p = hypergeom.pmf(i, M, k, k)
    if abs(p.sum()-1.0) > 1e-9: sys.exit('FATAL: pmf sums to %.12f' % p.sum())
    return float(np.sum(p * i / (2*k - i)))

def welch(x, y, alpha=ALPHA):
    d = y.mean() - x.mean()
    vx, vy, n = x.var(ddof=1), y.var(ddof=1), N_SEEDS
    se = np.sqrt(vx/n + vy/n)
    if not np.isfinite(se) or se <= 0: sys.exit('FATAL: Welch SE non-positive')
    df = (vx/n + vy/n)**2 / ((vx/n)**2/(n-1) + (vy/n)**2/(n-1))
    h = tdist.ppf(alpha, df) * se
    return float(d), float(d-h), float(d+h), float(df), float(se)

def perm_p(x, y):
    allv = np.concatenate([x, y]); obs = abs(y.mean() - x.mean()); hits = tot = 0
    for idx in combinations(range(len(allv)), N_SEEDS):
        m = np.zeros(len(allv), bool); m[list(idx)] = True
        if abs(allv[~m].mean() - allv[m].mean()) >= obs - 1e-12: hits += 1
        tot += 1
    return hits/tot, tot

# ---------------- P1: raw DE-overlap Jaccard ------------------------------
def per_condition_jaccard(f):
    with np.load(f, allow_pickle=True) as v:
        true, B, C = np.asarray(v['true']), np.asarray(v['B']), np.asarray(v['C'])
        cond = np.array([str(x) for x in v['cond']])
    for nm, a in (('true', true), ('B', B), ('C', C)):
        if not np.all(np.isfinite(a)): sys.exit('FATAL: non-finite %s in %s' % (nm, f))
    out = np.full(len(true), np.nan)
    for i in range(len(true)):
        td = true[i] - C[i]
        d = np.argsort(np.abs(td))[-K:]
        pd_ = B[i] - C[i]
        if np.abs(pd_).std() > 1e-12:
            out[i] = jac(d, np.argsort(np.abs(pd_))[-K:])
    return out, cond, true, C

store, paths, ref = {}, {}, None
for arm in ARMS:
    ps = [os.path.join(RES, 'vectors_step6_%s_seed_%d.npz' % (arm, s)) for s in range(N_SEEDS)]
    for p in ps:
        if not os.path.isfile(p): sys.exit('FATAL: missing ' + p)
    store[arm] = [per_condition_jaccard(p) for p in ps]; paths[arm] = ps
for arm in ARMS:
    for k, (J, cond, true, C) in enumerate(store[arm]):
        if ref is None: ref = (cond, true, C)
        else:
            if not np.array_equal(cond, ref[0]): sys.exit('FATAL: condition order differs')
            if not np.array_equal(true, ref[1]): sys.exit('FATAL: true differs, %s seed %d' % (arm, k))
            if not np.array_equal(C, ref[2]): sys.exit('FATAL: control mean differs')
        if int((~np.isfinite(J)).sum()): sys.exit('FATAL: non-finite Jaccard, %s seed %d' % (arm, k))
if len(ref[0]) != N_COND: sys.exit('FATAL: %d conditions' % len(ref[0]))
M_s = int(((ref[1] - ref[2]) != 0).any(axis=0).sum())
c_s = cs_reference(M_s)
P1 = {a: np.array([np.mean(r[0]) for r in store[a]]) for a in ARMS}
print('[design] %d conditions | M_s %d | c_s %.6f' % (N_COND, M_s, c_s))

# ---------------- P2: energy distance, read from the frozen artifact -------
p2f = os.path.join(RES, 'step6_p2_energy.json')
if not os.path.isfile(p2f): sys.exit('FATAL: run compute_p2.py first')
P2J = json.load(open(p2f))
if P2J['n_conditions'] != N_COND: sys.exit('FATAL: P2 condition count mismatch')
P2 = {a: np.array([P2J['records']['%s/%d' % (a, s)]['p2_u'] for s in range(N_SEEDS)])
      for a in ARMS}
print('[P2] estimator: %s' % P2J['estimator'])

# ---------------- gated contrasts -----------------------------------------
# P1 is a Jaccard: HIGHER is better. P2 is an energy distance: LOWER is better.
# `favours` records which arm a positive raw difference favours, so the verdict
# cannot silently invert.
CONTRASTS = [('B - A  (objective)', 'ddim', 'fm_gaussian'),
             ('C - B  (source)',    'fm_gaussian', 'fm_control_ot')]
ENDPOINTS = [('P1 DE-overlap Jaccard', P1, 'higher'),
             ('P2 energy distance',    P2, 'lower')]

results = {}
for ename, vals, better in ENDPOINTS:
    print('\n' + '=' * 74)
    print('%s   (%s is better)   gated at the %.1f%% level' % (ename, better, 100*ALPHA))
    print('=' * 74)
    for a in ARMS:
        print('  %-15s %s | mean %.5f  SD %.5f'
              % (a, ' '.join('%.5f' % v for v in vals[a]),
                 vals[a].mean(), vals[a].std(ddof=1)))
    results[ename] = {}
    for cname, a0, a1 in CONTRASTS:
        d, lo, hi, df, se = welch(vals[a0], vals[a1])
        p, n_alloc = perm_p(vals[a0], vals[a1])
        if better == 'higher':
            favours = 'the second arm' if d > 0 else 'the first arm'
            resolved = (lo > 0) or (hi < 0)
            direction = ('second arm better' if lo > 0 else
                         'first arm better' if hi < 0 else 'unresolved')
        else:
            favours = 'the first arm' if d > 0 else 'the second arm'
            resolved = (lo > 0) or (hi < 0)
            direction = ('first arm better' if lo > 0 else
                         'second arm better' if hi < 0 else 'unresolved')
        print('  %-20s %+.5f  CI [%+.5f, %+.5f]  df %.2f  p %.4f  -> %s'
              % (cname, d, lo, hi, df, p, direction))
        results[ename][cname] = dict(delta=d, ci=[lo, hi], welch_df=df, se=se,
                                     perm_p=p, n_alloc=n_alloc,
                                     direction=direction, resolved=bool(resolved),
                                     better=better, favours=favours)

# ---------------- joint decision rule -------------------------------------
print('\n' + '=' * 74)
print('JOINT DECISION, section 4: both co-primary endpoints, each at 97.5%')
print('=' * 74)
verdicts = {}
for cname, _, _ in CONTRASTS:
    d1 = results['P1 DE-overlap Jaccard'][cname]
    d2 = results['P2 energy distance'][cname]
    s1, s2 = d1['direction'], d2['direction']
    if s1 == s2 and s1 != 'unresolved':
        v = 'CONCORDANT: %s on both endpoints' % s1
    elif 'unresolved' in (s1, s2):
        other = s1 if s2 == 'unresolved' else s2
        v = ('UNRESOLVED on both' if s1 == s2 == 'unresolved'
             else 'PARTIAL: %s on one endpoint, unresolved on the other' % other)
    else:
        v = 'DISCORDANT: %s on P1 but %s on P2' % (s1, s2)
    verdicts[cname] = v
    print('  %-20s %s' % (cname, v))
print('\n  Neither endpoint alone constitutes the gate. A discordant result is')
print('  reported as such and NOT resolved by preferring one endpoint.')
print('  P1 MDE at 80%% power was stated in advance as ~0.0173.')
print('  P2 power was disclosed in advance as UNKNOWN; an unresolved P2 is not')
print('  evidence of no effect.')

g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
json.dump(dict(prereg='PREREG_step6_flow_matching.md', alpha=ALPHA,
               n_conditions=N_COND, M_s=M_s, c_s=c_s,
               p1={a: list(map(float, P1[a])) for a in ARMS},
               p2={a: list(map(float, P2[a])) for a in ARMS},
               contrasts=results, joint_verdicts=verdicts,
               p2_source=os.path.basename(p2f), p2_sha256=sha(p2f),
               commit=g('git rev-parse HEAD'),
               script_sha256=sha(os.path.abspath(__file__)),
               inputs={a: [sha(p) for p in paths[a]] for a in ARMS}),
          open(OUT + '.tmp', 'w'), indent=2, allow_nan=False)
os.replace(OUT + '.tmp', OUT)
print('\nwrote %s' % OUT)
