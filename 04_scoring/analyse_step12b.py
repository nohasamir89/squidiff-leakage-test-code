#!/usr/bin/env python3
"""PRE-REGISTERED Step 12b analysis. PREREG_step12b_confirmation.md, section 8.

Written and committed BEFORE any dropout-trained model exists on drug_split_0
or drug_split_2. Refuses to run if any input is missing, if any provenance
record's guidance weight disagrees with its filename, or if any vectors hash
disagrees with its record.

PRIMARY, gated: DE-overlap Jaccard, w=2 minus w=1 on the SAME five dropout-
trained checkpoints, per split, mean over conditions of seed-mean per-condition
differences, drug-clustered percentile bootstrap over the split's compounds,
5000 replicates, RandomState(0), at the 97.5% level per split (1.25 / 98.75
percentiles) - the level Steps 6b and 11 used for two-split confirmation.

CO-PRIMARY, gated: energy distance, amendment-13 U-statistic per condition,
mean over conditions, same contrast, same level. Higher is worse.

Outcome categories (section 5), applied mechanically across the two splits:
  CONFIRMS       DE-overlap above zero on BOTH splits AND energy not above zero on both
  TRADE-OFF      DE-overlap above zero on both, energy above zero on either
  PARTIAL        DE-overlap above zero on exactly one split
  NOT CONFIRMED  DE-overlap unresolved on both
  REVERSED       DE-overlap below zero on either split
"Energy not above zero" is unresolved, not evidence of no cost.

SECONDARY, not gated: dropout itself (unguided dropout arm at w=1 minus the
Step 11 reference vectors: Step 6b checkpoints at 50 NFE, no dropout); DPC-50,
B-C and predicted-delta magnitude at every weight; EXPLORATORY w=3 and the
half-trajectory w=2, each against w=1 and against w=2.

SENSITIVITY: paired MDE per split by the Step 6b formula for the paired design,
(t_0.975,4 + t_0.80,4) * SD(paired per-checkpoint diff)/sqrt(5), reported and
not used to interpret the result.
"""
import os, sys, json, subprocess, hashlib
import numpy as np
import anndata as ad
from scipy.stats import pearsonr, t as student_t
from scipy.spatial.distance import cdist, pdist

RES = 'leakage_results'
DATA = '/scratch/ni2071/Diffusion/sciplex/datasets_corrected'
SPLITS = {'drug_split_0': 444, 'drug_split_2': 455}
N_SEEDS = 5; K = 50; N_BOOT = 5000
LO, HI = 1.25, 98.75                      # 97.5% interval per split
OUT = os.path.join(RES, 'step12b_analysis.json')
PRE = 'PREREG_step12b_confirmation.md'
if os.path.exists(OUT): sys.exit('FATAL: %s exists' % OUT)
if open(PRE).read().count('\n## ') < 9: sys.exit('FATAL: prereg incomplete')
if subprocess.run('git diff --quiet -- %s' % PRE, shell=True).returncode != 0:
    sys.exit('FATAL: prereg uncommitted')
print('[prereg] committed')
rng = np.random.RandomState(0)

# tag -> (expected guidance_w, expected guide_t_min)
ARMS = {'wnone': (None, 0.0), 'w1': (1.0, 0.0), 'w2': (2.0, 0.0), 'w3': (3.0, 0.0), 'w2h': (2.0, 0.5)}

def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): m.update(c)
    return m.hexdigest()

def jac(a, b):
    i = len(np.intersect1d(a, b)); return i / (len(a) + len(b) - i)

def per_cond(f):
    v = np.load(f, allow_pickle=True)
    true, B, C = [np.asarray(v[k], dtype=np.float64) for k in ('true', 'B', 'C')]
    cond = np.array([str(x) for x in v['cond']])
    n = len(true)
    j = np.full(n, np.nan); d = np.full(n, np.nan); bc = np.full(n, np.nan)
    mag = np.full(n, np.nan); magt = np.full(n, np.nan)
    for i in range(n):
        td = true[i] - C[i]; p = B[i] - C[i]
        g = np.argsort(np.abs(td))[-K:]
        if np.abs(p).std() > 1e-12: j[i] = jac(g, np.argsort(np.abs(p))[-K:])
        if td[g].std() > 1e-12 and p[g].std() > 1e-12: d[i] = pearsonr(p[g], td[g])[0]
        if true[i][g].std() > 1e-12 and B[i][g].std() > 1e-12 and C[i][g].std() > 1e-12:
            bc[i] = pearsonr(B[i][g], true[i][g])[0] - pearsonr(C[i][g], true[i][g])[0]
        mag[i] = np.abs(p).mean(); magt[i] = np.abs(td).mean()
    return dict(jac=j, dpc=d, bc=bc, mag=mag, magt=magt, cond=cond)

def energy_per_cond(f_pc):
    p = np.load(f_pc, allow_pickle=True)
    out = []
    for a, b in zip(p['pred'], p['true']):
        X = np.asarray(a, dtype=np.float64); Y = np.asarray(b, dtype=np.float64)
        if X.shape != Y.shape: sys.exit('FATAL: per-cell shape mismatch in %s' % f_pc)
        out.append(2 * cdist(X, Y).mean() - pdist(X).mean() - pdist(Y).mean())
    return np.array(out)

def boot(x, drugs):
    ix = np.where(np.isfinite(x))[0]
    gs = [ix[drugs[ix] == u] for u in np.unique(drugs[ix])]
    o = np.array([np.mean(x[np.concatenate([gs[k] for k in rng.randint(0, len(gs), len(gs))])])
                  for _ in range(N_BOOT)])
    return float(np.percentile(o, LO)), float(np.percentile(o, HI))

def verdict(lo, hi):
    return 'above zero' if lo > 0 else 'below zero' if hi < 0 else 'includes zero'

def load_arm(split, name_of_seed, prov_of_seed, gw, tmin, n_expect, ref_cond):
    """Load five seeds of one arm; returns cond-mean and per-checkpoint-mean dicts, file records, cond order."""
    per = {k: [] for k in ('jac', 'dpc', 'bc', 'mag', 'magt', 'en')}; files = []
    for s in range(N_SEEDS):
        name = name_of_seed(s)
        f = os.path.join(RES, 'vectors_%s.npz' % name); fpc = os.path.join(RES, 'vectors_%s_pc.npz' % name)
        pj = prov_of_seed(s)
        for q in (f, fpc, pj):
            if not os.path.isfile(q): sys.exit('FATAL: missing %s' % q)
        pv = json.load(open(pj))
        got_w = pv.get('guidance_w'); got_t = float(pv.get('guide_t_min', 0.0))
        if (gw is None and got_w is not None) or (gw is not None and (got_w is None or abs(float(got_w) - gw) > 1e-9)):
            sys.exit('FATAL: %s provenance guidance_w=%r, expected %r' % (name, got_w, gw))
        if abs(got_t - tmin) > 1e-9: sys.exit('FATAL: %s provenance guide_t_min=%r, expected %r' % (name, got_t, tmin))
        if pv.get('split') != split: sys.exit('FATAL: %s provenance split=%r, expected %s' % (name, pv.get('split'), split))
        if pv.get('vectors_sha256') != sha(f): sys.exit('FATAL: %s vectors hash differs from provenance' % name)
        r = per_cond(f)
        if len(r['cond']) != n_expect: sys.exit('FATAL: %s has %d conditions, expected %d' % (name, len(r['cond']), n_expect))
        if ref_cond[0] is None: ref_cond[0] = r['cond']
        elif not np.array_equal(r['cond'], ref_cond[0]): sys.exit('FATAL: condition order differs in %s' % name)
        for k in ('jac', 'dpc', 'bc', 'mag', 'magt'): per[k].append(r[k])
        per['en'].append(energy_per_cond(fpc))
        files.append(dict(tag=name, vectors_sha256=sha(f), per_cell_sha256=sha(fpc),
                          checkpoint_sha256=pv.get('checkpoint_sha256'), guidance_w=got_w, guide_t_min=got_t))
    for k in ('jac', 'en'):
        if not np.all(np.isfinite(np.vstack(per[k]))): sys.exit('FATAL: non-finite %s in %s' % (k, name_of_seed(0)))
    return (dict(cond_mean={k: np.nanmean(np.vstack(per[k]), axis=0) for k in per},
                 ckpt_mean={k: np.array([np.nanmean(x) for x in per[k]]) for k in per}), files)

results = {}; inputs = {}; gate = {}
for split, n_expect in SPLITS.items():
    print('\n' + '#' * 72 + '\n# %s  (%d conditions)\n' % (split, n_expect) + '#' * 72)
    M = {}; ref_cond = [None]
    for tag, (gw, tmin) in ARMS.items():
        M[tag], inputs['%s/%s' % (split, tag)] = load_arm(
            split, lambda s, t=tag: 's12b_%s_%s_seed_%d' % (split, t, s),
            lambda s, t=tag: os.path.join('provenance', 's12bev_s12b_%s_%s_seed_%d.json' % (split, t, s)),
            gw, tmin, n_expect, ref_cond)
    # Step 11 reference: Step 6b checkpoints at 50 NFE, no dropout, no guidance
    M['ref'], inputs['%s/ref' % split] = load_arm(
        split, lambda s: 's11_%s_fm_gaussian_nfe50_seed_%d' % (split, s),
        lambda s: os.path.join('provenance', 's11ev_s11_%s_fm_gaussian_nfe50_seed_%d.json' % (split, s)),
        None, 0.0, n_expect, ref_cond)
    cond = ref_cond[0]
    a = ad.read_h5ad('%s/sci_plex_test_%s.h5ad' % (DATA, split), backed='r')
    g = a.obs['Group'].astype(str).values; smi = a.obs['SMILES'].astype(str).values
    drugs = np.array([smi[g == c][0] for c in cond])
    print('[design] %d conditions | %d drugs' % (len(cond), len(np.unique(drugs))))
    for tag in ('ref', 'wnone', 'w1', 'w2', 'w3', 'w2h'):
        c = M[tag]['ckpt_mean']
        print('[load] %-6s DE-overlap %.4f  energy %.4f  DPC-50 %.4f  B-C %+.4f  |B-C| %.5f  |true-C| %.5f'
              % (tag, c['jac'].mean(), c['en'].mean(), c['dpc'].mean(), c['bc'].mean(), c['mag'].mean(), c['magt'].mean()))
    # sanity gate carried from Step 10: unguided magnitude against the reference
    ratio = M['w1']['ckpt_mean']['mag'].mean() / M['ref']['ckpt_mean']['mag'].mean()
    if not (0.5 <= ratio <= 2.0): sys.exit('FATAL: %s unguided |B-C| is %.2fx the reference; pairing or checkpoint fault' % (split, ratio))
    print('[sanity] unguided |B-C| / reference |B-C|: %.3f' % ratio)

    def contrast(b, a_, key):
        diff = M[b]['cond_mean'][key] - M[a_]['cond_mean'][key]
        m = float(np.nanmean(diff)); lo, hi = boot(diff, drugs)
        return dict(diff=m, ci=[lo, hi], verdict=verdict(lo, hi))

    res = {'arm_means': {t: {k: float(M[t]['ckpt_mean'][k].mean()) for k in ('jac', 'en', 'dpc', 'bc', 'mag', 'magt')} for t in M}}
    pj = contrast('w2', 'w1', 'jac'); pe = contrast('w2', 'w1', 'en')
    d_seed = M['w2']['ckpt_mean']['jac'] - M['w1']['ckpt_mean']['jac']
    se = float(d_seed.std(ddof=1) / np.sqrt(N_SEEDS))
    mde = float((student_t.ppf(0.975, N_SEEDS - 1) + student_t.ppf(0.80, N_SEEDS - 1)) * se)
    res['primary'] = dict(de_overlap_w2_minus_w1=pj, energy_w2_minus_w1=pe,
                          mde_80_paired=dict(value=mde, se=se, df=N_SEEDS - 1))
    print('\nPRIMARY (gated): DE-overlap, w=2 minus w=1     %+.5f  97.5%% CI [%+.5f, %+.5f]  -> %s' % (pj['diff'], *pj['ci'], pj['verdict']))
    print('CO-PRIMARY (gated): energy, w=2 minus w=1       %+.5f  97.5%% CI [%+.5f, %+.5f]  -> %s' % (pe['diff'], *pe['ci'], pe['verdict']))
    print('  paired MDE at 80%% power: %.5f   [reported, not used]' % mde)
    gate[split] = (pj['verdict'], pe['verdict'])
    res['secondary'] = {}
    row = {k: contrast('w1', 'ref', k) for k in ('jac', 'en', 'dpc')}
    res['secondary']['w1_minus_step11_reference'] = row
    print('SECONDARY dropout itself, w=1 minus Step 11 reference:  DE %+.5f [%+.5f,%+.5f] %s | energy %+.5f [%+.5f,%+.5f] %s | DPC %+.5f'
          % (row['jac']['diff'], *row['jac']['ci'], row['jac']['verdict'], row['en']['diff'], *row['en']['ci'], row['en']['verdict'], row['dpc']['diff']))
    row = {k: contrast('w1', 'wnone', k) for k in ('jac', 'en')}
    res['secondary']['w1_minus_wnone_identity'] = row
    print('  identity: w1 minus wnone  DE %+.2e  energy %+.2e' % (row['jac']['diff'], row['en']['diff']))
    res['exploratory'] = {}
    for tag in ('w3', 'w2h'):
        for other in ('w1', 'w2'):
            row = {k: contrast(tag, other, k) for k in ('jac', 'en', 'dpc', 'bc')}
            res['exploratory']['%s_minus_%s' % (tag, other)] = row
            print('EXPLORATORY %-4s vs %-3s  DE %+.5f [%+.5f,%+.5f] %-13s | energy %+.5f [%+.5f,%+.5f] %-13s | DPC %+.5f | B-C %+.5f'
                  % (tag, other, row['jac']['diff'], *row['jac']['ci'], row['jac']['verdict'],
                     row['en']['diff'], *row['en']['ci'], row['en']['verdict'], row['dpc']['diff'], row['bc']['diff']))
    results[split] = res

de = [gate[s][0] for s in SPLITS]; en = [gate[s][1] for s in SPLITS]
if any(v == 'below zero' for v in de): cat = 'REVERSED'
elif all(v == 'above zero' for v in de): cat = 'TRADE-OFF' if any(v == 'above zero' for v in en) else 'CONFIRMS'
elif any(v == 'above zero' for v in de): cat = 'PARTIAL'
else: cat = 'NOT CONFIRMED'
print('\n' + '=' * 72 + '\nPRE-REGISTERED VERDICT: %s\n' % cat +
      '  Section 6 expected CONFIRMS with energy unresolved. Development effect was +0.00854.\n' + '=' * 72)
json.dump(dict(prereg=PRE, verdict=cat, interval='97.5% percentile per split, drug-clustered', n_boot=N_BOOT,
               statistic='mean over conditions of seed-mean per-condition values',
               per_split=results, inputs=inputs,
               commit=subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()),
          open(OUT, 'w'), indent=2, allow_nan=False)
print('\nwrote %s' % OUT)
