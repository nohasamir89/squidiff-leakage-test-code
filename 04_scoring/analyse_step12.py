#!/usr/bin/env python3
"""PRE-REGISTERED Step 12 analysis. PREREG_step12_guidance.md, section 8.

Written and committed BEFORE any guided vector exists. Refuses to run if any
expected vector file is missing or any provenance record's guidance_w disagrees
with the file it belongs to.

PRIMARY, gated: DE-overlap Jaccard, guided at w=2 minus unguided (w=1) on the
SAME five dropout-trained checkpoints, paired per condition. Per-condition
values are the mean over the five seeds (the convention of Steps 6b and 11);
the contrast is the mean over conditions of the per-condition difference, with
a drug-clustered percentile bootstrap over the split's compounds, 5000
replicates, RandomState(0), 95% interval (2.5 / 97.5 percentiles).

CO-PRIMARY, gated: energy distance, amendment-13 off-diagonal U-statistic per
condition (compute_p2.py), summarised by the unweighted mean over conditions
(Step 6 convention), same contrast, same bootstrap. Higher is worse.

Outcome categories (section 5), applied mechanically:
  GUIDANCE HELPS  DE-overlap interval above zero AND energy interval not above zero
  TRADE-OFF       DE-overlap interval above zero AND energy interval above zero
  UNRESOLVED      DE-overlap interval includes zero
  GUIDANCE HARMS  DE-overlap interval below zero

SECONDARY, reported and not gated: the w sweep {1, 1.5, 2, 3, 5, 8} against w=1
on both endpoints; the dropout effect (unguided dropout arm minus the Step 10
fingerprint_s10 reference, both at 50 NFE); DPC-50 and B-C at every w; the
EXPLORATORY half-trajectory variant (w=2, guidance only for t>=0.5); and the
predicted-delta magnitude at every w, which is what guidance is expected to
change first.

SENSITIVITY (section 4): the minimum detectable effect is computed here, after
the fact, by the Step 6b formula adapted to the paired design - the five
per-checkpoint paired differences give SE = SD/sqrt(5) with 4 degrees of
freedom, and MDE at 80% power is (t_0.975,4 + t_0.80,4) * SE. Reported next to
the result and not used to interpret it.

The unguided baseline of the gated contrast is the guided code path at w=1
(tag w1). The unguided code path (tag wnone) is the identity check's reference
and is reported alongside; the two must agree to 1e-6 (check_s12_identity.py).
"""
import os, sys, json, subprocess, hashlib
import numpy as np
import anndata as ad
from scipy.stats import pearsonr, t as student_t
from scipy.spatial.distance import cdist, pdist

RES = 'leakage_results'
DATA = '/scratch/ni2071/Diffusion/sciplex/datasets_corrected'
SPLIT = 'drug_split_1'; N_COND = 443; N_SEEDS = 5
K = 50; N_BOOT = 5000
OUT = os.path.join(RES, 'step12_analysis.json')
PRE = 'PREREG_step12_guidance.md'
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
if open(PRE).read().count('\n## ') < 9: sys.exit('FATAL: prereg incomplete')
if subprocess.run('git diff --quiet -- %s' % PRE, shell=True).returncode != 0:
    sys.exit('FATAL: prereg uncommitted')
print('[prereg] committed')
rng = np.random.RandomState(0)

# tag -> (arm key, expected guidance_w in provenance, expected guide_t_min)
ARMS = {
    'wnone': ('cfg', None, 0.0),   # unguided code path
    'w1':    ('cfg', 1.0,  0.0),   # guided path, w=1: the gated baseline
    'ref':   ('ref', None, 0.0),   # fingerprint_s10, no dropout, 50 NFE
    'w1p5':  ('cfg', 1.5,  0.0),
    'w2':    ('cfg', 2.0,  0.0),   # GATED
    'w3':    ('cfg', 3.0,  0.0),
    'w5':    ('cfg', 5.0,  0.0),
    'w8':    ('cfg', 8.0,  0.0),
    'w2h':   ('cfg', 2.0,  0.5),   # EXPLORATORY: guidance only for t >= 0.5
}
SWEEP = ('w1', 'w1p5', 'w2', 'w3', 'w5', 'w8')

def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): m.update(c)
    return m.hexdigest()

def jac(a, b):
    i = len(np.intersect1d(a, b)); return i / (len(a) + len(b) - i)

def per_cond(f):
    """Per-condition DE-overlap, DPC-50, B-C and predicted-delta magnitude from one vectors file."""
    v = np.load(f, allow_pickle=True)
    true, B, C = [np.asarray(v[k], dtype=np.float64) for k in ('true', 'B', 'C')]
    cond = np.array([str(x) for x in v['cond']])
    n = len(true)
    j = np.full(n, np.nan); d = np.full(n, np.nan); bc = np.full(n, np.nan)
    mag = np.full(n, np.nan); magt = np.full(n, np.nan)
    for i in range(n):
        td = true[i] - C[i]; p = B[i] - C[i]
        g = np.argsort(np.abs(td))[-K:]                       # true top-K DEGs
        if np.abs(p).std() > 1e-12: j[i] = jac(g, np.argsort(np.abs(p))[-K:])
        if td[g].std() > 1e-12 and p[g].std() > 1e-12: d[i] = pearsonr(p[g], td[g])[0]
        if true[i][g].std() > 1e-12 and B[i][g].std() > 1e-12 and C[i][g].std() > 1e-12:
            bc[i] = pearsonr(B[i][g], true[i][g])[0] - pearsonr(C[i][g], true[i][g])[0]
        mag[i] = np.abs(p).mean(); magt[i] = np.abs(td).mean()
    return dict(jac=j, dpc=d, bc=bc, mag=mag, magt=magt, cond=cond)

def energy_per_cond(f_pc):
    """Amendment-13 U-statistic per condition (compute_p2.py), signed, not clamped, no square root."""
    p = np.load(f_pc, allow_pickle=True)
    out = []
    for a, b in zip(p['pred'], p['true']):
        X = np.asarray(a, dtype=np.float64); Y = np.asarray(b, dtype=np.float64)
        if X.shape != Y.shape: sys.exit('FATAL: per-cell shape mismatch in %s' % f_pc)
        out.append(2 * cdist(X, Y).mean() - pdist(X).mean() - pdist(Y).mean())
    return np.array(out)

def boot(x, drugs, lo=2.5, hi=97.5):
    """Drug-clustered percentile bootstrap of the mean of x over conditions (95% interval)."""
    ix = np.where(np.isfinite(x))[0]
    gs = [ix[drugs[ix] == u] for u in np.unique(drugs[ix])]
    o = np.array([np.mean(x[np.concatenate([gs[k] for k in rng.randint(0, len(gs), len(gs))])])
                  for _ in range(N_BOOT)])
    return float(np.percentile(o, lo)), float(np.percentile(o, hi))

def verdict(lo, hi):
    return 'above zero' if lo > 0 else 'below zero' if hi < 0 else 'includes zero'

# ---- load every arm, checking provenance against the filename -------------
M, prov, ref_cond = {}, {}, None
for tag, (armk, gw, tmin) in ARMS.items():
    per = {k: [] for k in ('jac', 'dpc', 'bc', 'mag', 'magt', 'en')}
    files = []
    for s in range(N_SEEDS):
        name = 's12_%s_%s_seed_%d' % (armk, tag, s)
        f = os.path.join(RES, 'vectors_%s.npz' % name); fpc = os.path.join(RES, 'vectors_%s_pc.npz' % name)
        pj = os.path.join('provenance', 's12ev_%s.json' % name)
        for q in (f, fpc, pj):
            if not os.path.isfile(q): sys.exit('FATAL: missing %s' % q)
        pv = json.load(open(pj))
        got_w = pv.get('guidance_w'); got_t = float(pv.get('guide_t_min', 0.0))
        if (gw is None and got_w is not None) or (gw is not None and (got_w is None or abs(float(got_w) - gw) > 1e-9)):
            sys.exit('FATAL: %s provenance guidance_w=%r, filename implies %r' % (name, got_w, gw))
        if abs(got_t - tmin) > 1e-9: sys.exit('FATAL: %s provenance guide_t_min=%r, expected %r' % (name, got_t, tmin))
        if pv.get('vectors_sha256') != sha(f): sys.exit('FATAL: %s vectors hash differs from provenance' % name)
        r = per_cond(f)
        if len(r['cond']) != N_COND: sys.exit('FATAL: %s has %d conditions' % (name, len(r['cond'])))
        if ref_cond is None: ref_cond = r['cond']
        elif not np.array_equal(r['cond'], ref_cond): sys.exit('FATAL: condition order differs in %s' % name)
        for k in ('jac', 'dpc', 'bc', 'mag', 'magt'): per[k].append(r[k])
        per['en'].append(energy_per_cond(fpc))
        files.append(dict(tag=name, vectors=f, vectors_sha256=sha(f), per_cell_sha256=sha(fpc),
                          checkpoint_sha256=pv.get('checkpoint_sha256'), guidance_w=got_w, guide_t_min=got_t))
    for k in per:
        arr = np.vstack(per[k])
        if k in ('jac', 'en') and not np.all(np.isfinite(arr)):
            sys.exit('FATAL: non-finite %s in arm %s' % (k, tag))
    M[tag] = dict(cond_mean={k: np.nanmean(np.vstack(per[k]), axis=0) for k in per},
                  ckpt_mean={k: np.array([np.nanmean(x) for x in per[k]]) for k in per})
    prov[tag] = files
    print('[load] %-6s %s  DE-overlap %.4f  energy %.4f  DPC-50 %.4f  B-C %+.4f  |B-C| %.5f  |true-C| %.5f'
          % (tag, 'w=%s t>=%.1f' % (gw, tmin) if gw is not None else 'unguided',
             M[tag]['ckpt_mean']['jac'].mean(), M[tag]['ckpt_mean']['en'].mean(),
             M[tag]['ckpt_mean']['dpc'].mean(), M[tag]['ckpt_mean']['bc'].mean(),
             M[tag]['ckpt_mean']['mag'].mean(), M[tag]['ckpt_mean']['magt'].mean()))

# ---- design: drug clusters from the test file ----------------------------
a = ad.read_h5ad('%s/sci_plex_test_%s.h5ad' % (DATA, SPLIT), backed='r')
g = a.obs['Group'].astype(str).values; smi = a.obs['SMILES'].astype(str).values
drugs = np.array([smi[g == c][0] for c in ref_cond])
print('[design] %d conditions | %d drugs' % (len(ref_cond), len(np.unique(drugs))))

# ---- sanity gate carried from Step 10: unguided prediction magnitude ------
ref_mag = M['ref']['ckpt_mean']['mag'].mean()
for tag in ('wnone', 'w1'):
    ratio = M[tag]['ckpt_mean']['mag'].mean() / ref_mag
    if not (0.5 <= ratio <= 2.0):
        sys.exit('FATAL: unguided %s predicted |delta| is %.2fx the reference arm; pairing or checkpoint fault' % (tag, ratio))
print('[sanity] unguided |B-C| / reference |B-C|: wnone %.3f  w1 %.3f' %
      (M['wnone']['ckpt_mean']['mag'].mean() / ref_mag, M['w1']['ckpt_mean']['mag'].mean() / ref_mag))

def contrast(b, a_, key):
    diff = M[b]['cond_mean'][key] - M[a_]['cond_mean'][key]
    m = float(np.nanmean(diff)); lo, hi = boot(diff, drugs)
    return dict(diff=m, ci=[lo, hi], verdict=verdict(lo, hi))

res = {'primary': {}, 'secondary': {}, 'exploratory': {}, 'arm_means': {}}
for tag in ARMS:
    res['arm_means'][tag] = {k: float(M[tag]['ckpt_mean'][k].mean()) for k in ('jac', 'en', 'dpc', 'bc', 'mag', 'magt')}
    res['arm_means'][tag]['jac_seed_sd'] = float(M[tag]['ckpt_mean']['jac'].std(ddof=1))

print('\n' + '=' * 72 + '\nPRIMARY (gated): DE-overlap, w=2 minus w=1, same checkpoints\n' + '=' * 72)
pj = contrast('w2', 'w1', 'jac'); res['primary']['de_overlap_w2_minus_w1'] = pj
print('  %+.5f  95%% CI [%+.5f, %+.5f]  -> %s' % (pj['diff'], *pj['ci'], pj['verdict']))
print('\nCO-PRIMARY (gated): energy distance (U-statistic), w=2 minus w=1; higher = worse')
pe = contrast('w2', 'w1', 'en'); res['primary']['energy_w2_minus_w1'] = pe
print('  %+.5f  95%% CI [%+.5f, %+.5f]  -> %s' % (pe['diff'], *pe['ci'], pe['verdict']))

# MDE, Step 6b formula adapted to the paired design (reported, not used)
d_seed = M['w2']['ckpt_mean']['jac'] - M['w1']['ckpt_mean']['jac']
se = float(d_seed.std(ddof=1) / np.sqrt(N_SEEDS))
mde = float((student_t.ppf(0.975, N_SEEDS - 1) + student_t.ppf(0.80, N_SEEDS - 1)) * se)
res['primary']['mde_80_paired'] = dict(value=mde, se=se, df=N_SEEDS - 1,
                                       formula='(t_0.975,4 + t_0.80,4) * SD(paired per-checkpoint diff)/sqrt(5)')
print('  MDE at 80%% power (paired, Step 6b formula): %.5f   [reported, not used]' % mde)

if pj['verdict'] == 'above zero':
    cat = 'TRADE-OFF' if pe['verdict'] == 'above zero' else 'GUIDANCE HELPS'
elif pj['verdict'] == 'below zero': cat = 'GUIDANCE HARMS'
else: cat = 'UNRESOLVED'
res['primary']['category'] = cat
print('\n' + '=' * 72 + '\nPRE-REGISTERED VERDICT: %s\n' % cat +
      '  Section 6 expected UNRESOLVED with energy distance rising in w.\n' + '=' * 72)

print('\nSECONDARY (not gated): sweep against w=1, both endpoints and DPC-50 / B-C')
for tag in SWEEP:
    if tag == 'w1': continue
    row = {k: contrast(tag, 'w1', k) for k in ('jac', 'en', 'dpc', 'bc')}
    res['secondary']['%s_minus_w1' % tag] = row
    print('  %-5s DE %+.5f [%+.5f,%+.5f] %-13s | energy %+.5f [%+.5f,%+.5f] %-13s | DPC %+.5f | B-C %+.5f | |B-C| %.5f'
          % (tag, row['jac']['diff'], *row['jac']['ci'], row['jac']['verdict'],
             row['en']['diff'], *row['en']['ci'], row['en']['verdict'],
             row['dpc']['diff'], row['bc']['diff'], res['arm_means'][tag]['mag']))

print('\nSECONDARY (not gated): dropout itself, unguided dropout arm minus fingerprint_s10 reference, 50 NFE')
for tag in ('w1', 'wnone'):
    row = {k: contrast(tag, 'ref', k) for k in ('jac', 'en', 'dpc')}
    res['secondary']['%s_minus_ref' % tag] = row
    print('  %-5s DE %+.5f [%+.5f,%+.5f] %-13s | energy %+.5f [%+.5f,%+.5f] %-13s | DPC %+.5f'
          % (tag, row['jac']['diff'], *row['jac']['ci'], row['jac']['verdict'],
             row['en']['diff'], *row['en']['ci'], row['en']['verdict'], row['dpc']['diff']))
row = {k: contrast('w1', 'wnone', k) for k in ('jac', 'en')}
res['secondary']['w1_minus_wnone_identity'] = row
print('  identity: w1 minus wnone  DE %+.2e  energy %+.2e  (must be ~0)' % (row['jac']['diff'], row['en']['diff']))

print('\nEXPLORATORY: w=2 with guidance only for t >= 0.5, against w=1 and against w=2')
for other in ('w1', 'w2'):
    row = {k: contrast('w2h', other, k) for k in ('jac', 'en')}
    res['exploratory']['w2h_minus_%s' % other] = row
    print('  vs %-3s DE %+.5f [%+.5f,%+.5f] %-13s | energy %+.5f [%+.5f,%+.5f] %s'
          % (other, row['jac']['diff'], *row['jac']['ci'], row['jac']['verdict'],
             row['en']['diff'], *row['en']['ci'], row['en']['verdict']))

json.dump(dict(prereg=PRE, split=SPLIT, n_conditions=N_COND, n_drugs=int(len(np.unique(drugs))),
               n_boot=N_BOOT, interval='95% percentile, drug-clustered', statistic='mean over conditions of seed-mean per-condition values',
               results=res, inputs=prov,
               commit=subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()),
          open(OUT, 'w'), indent=2, allow_nan=False)
print('\nwrote %s' % OUT)
