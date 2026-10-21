#!/usr/bin/env python3
"""
Does the drug fingerprint influence the model's output at all?
(Step 3 causal test. GPU needed - decodes 3 arms per condition.)

==============================================================================
WHY THIS EXISTS
==============================================================================
Rank-based specificity on BH-significant conditions gave z = -0.27 (A549),
-1.96 (K562), -1.84 (MCF7) against per-regime permutation nulls: weak, partial
drug specificity, not clean collapse. That is a correlational result. This is
the causal one: replace the fingerprint and see whether the output moves.

If predictions are unchanged when the drug identity is swapped, no fingerprint
information reaches the decoder, and gene-level attention (Steps 5-6) cannot
help. If predictions change but rank does not improve, the information arrives
and is used badly - a different problem with a different fix.

==============================================================================
DESIGN
==============================================================================
Three arms, identical in every respect except the fingerprint fed to the encoder:
  REAL     the condition's own (SMILES, dose) fingerprint          [reproduces B]
  SHUFFLED a DIFFERENT drug's fingerprint at the SAME dose         [primary test]
  ZERO     an all-zero fingerprint vector                          [secondary]

SHUFFLED is primary. It is a derangement over unique SMILES (no drug maps to
itself), with dose held fixed, so only drug identity changes. The fingerprint
distribution is preserved exactly, so a change in output cannot be attributed
to feeding the encoder an unfamiliar input.

ZERO is secondary and must be read with caution: an all-zero fingerprint is
out-of-distribution for an encoder trained only on real rFCFP vectors, so a
change there could reflect OOD behaviour rather than loss of information.

PAIRING: all three arms use the same torch seed per condition
(torch.manual_seed(i_cond)) immediately before each decode, so DDIM noise is
identical across arms and cancels in any within-condition comparison. Without
this, decoder noise (SD ~0.015 unseeded) would swamp the effect.

DETERMINISM CHECK: the REAL arm should reproduce vectors_seed_0.npz['B'] to
float32 precision, since the harness that produced it used the same per-condition
seeding, the same RandomState(0) cell subsample and the same RandomState(i_cond)
control draw. Any mismatch means the arms are not comparable to prior results
and the run should be discarded.

==============================================================================
DATA ASSUMPTIONS
==============================================================================
* Model: seedvar/seed_0/model_0.9999.pt (Step 1 seed model, drug_split_1).
* Panel 200 genes, log1p, panel mean expression ~0.021.
* Encoder (MLPModel.py:201) computes x_start = concat([control_feature,
  drug_dose]) - it ASSIGNS, so under use_drug_structure=True the encoder input
  is exactly control expression (200) + fingerprint (1024) = 1224. Confirmed by
  the trained first-layer shape (2048, 1224). The fingerprint is therefore 84%
  of the encoder's input dimensions.
* drug_split_1 test: 37 held-out drugs x 4 doses (10/100/1000/10000 nM,
  NANOMOLAR) x 3 lines; 443 conditions with >= 6 cells.
* RULE 7 DOES NOT APPLY HERE. Verified from vectors_seed_0.npz: B ranges
  0-2.89 (mean 0.042), true 0-2.68 (mean 0.029), C 0-1.26 (mean 0.026) - all
  the same scale. sample_squidiff.py contains no log1p/clip/expm1, and
  leakage_test_base_sciplex_array.py applies none. The sci-Plex decoder emits
  log1p-scale output directly. Rule 7 derives from the GBM 2000-gene run where
  output was 12-77 against a data mean of 0.092, i.e. from a model that failed
  to fit. Applying it here would corrupt every arm.
* Control mean is per cell line, never pooled.

==============================================================================
WHAT IS REPORTED
==============================================================================
1. DETERMINISM: max |REAL - saved B| per condition. Must be ~0.
2. OUTPUT SENSITIVITY: per-condition Pearson r between the REAL delta and the
   SHUFFLED delta. r ~ 1.0 means swapping the drug changed nothing -> the
   fingerprint does not reach the output. r << 1 means it does.
   Also reports relative L2 change ||REAL - SHUF|| / ||REAL||.
3. Same for ZERO (secondary, OOD caveat above).
4. B-C on top-50 DEGs (Pearson) per arm, so the ablation is on the same scale
   as the headline result.

==============================================================================
POINTS FOR REVIEW
==============================================================================
(a) DERANGEMENT. Built over unique SMILES by rejection sampling until no drug
    maps to itself, dose held fixed. Is a single derangement enough, or should
    several be drawn and averaged? A single draw could be unlucky (e.g. mapping
    many drugs to structurally similar ones).
(b) SENSITIVITY MEASURE. Per-condition Pearson between REAL and SHUFFLED deltas
    is primary. Is correlation right, or should it be relative L2 (also
    reported), or cosine on uncentred deltas?
(c) INTERPRETATION OF INTERMEDIATE r. If r ~ 0.9, the fingerprint has a small
    but real effect. Is there a principled threshold, or is this necessarily
    descriptive?
(d) ZERO ARM. Included as secondary with an OOD caveat. Worth reporting at all,
    or does its presence invite over-interpretation?
==============================================================================
"""
import os, sys, json
import numpy as np
import torch
import scanpy as sc
from scipy.stats import pearsonr
import sample_squidiff
from Squidiff.scrna_datasets import Drug_dose_encoder

ROOT  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(ROOT, 'leakage_results')
DATA  = '/scratch/ni2071/Diffusion/sciplex/datasets_corrected'
# Optional argv: [1] checkpoint path, [2] output tag, [3] vectors tag for the
# determinism check. Defaults reproduce the original base-model invocation.
CKPT    = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'seedvar', 'seed_0', 'model_0.9999.pt')
ABL_TAG = sys.argv[2] if len(sys.argv) > 2 else 'drug_split_1'
VEC_TAG = sys.argv[3] if len(sys.argv) > 3 else 'seed_0'
SPLIT, GENE_SIZE, MAX_CELLS, SEED = 'drug_split_1', 200, 200, 0
rng = np.random.RandomState(SEED)

def dense(x): return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)

if not os.path.exists(CKPT): sys.exit('FATAL: checkpoint not found: ' + CKPT)

tr = sc.read_h5ad(f'{DATA}/sci_plex_test_{SPLIT}.h5ad')
ct = sc.read_h5ad(f'{DATA}/sci_plex_test_{SPLIT}_control.h5ad')
tX, cX = dense(tr.X), dense(ct.X)
g   = tr.obs['Group'].astype(str).values
smi = tr.obs['SMILES'].astype(str).values
dos = tr.obs['dose'].astype(float).values
tl  = tr.obs['cell_name'].astype(str).values
cl  = ct.obs['cell_name'].astype(str).values
pools = {L: cX[cl == L] for L in np.unique(cl)}

# ---- derangement over unique SMILES, dose held fixed -----------------------
uniq = sorted(set(smi))
ns = len(uniq)
for _ in range(10000):
    perm = rng.permutation(ns)
    if not np.any(perm == np.arange(ns)): break
else:
    sys.exit('FATAL: could not build derangement')
smap = {uniq[i]: uniq[perm[i]] for i in range(ns)}
assert all(smap[k] != k for k in uniq), 'derangement sanity check failed'
print('derangement over %d unique SMILES; no drug maps to itself' % ns)
fp_sim = []   # cosine between each drug's fingerprint and its deranged partner

sampler = sample_squidiff.sampler(CKPT, GENE_SIZE, GENE_SIZE, True)
print('model: ' + CKPT)

def decode(z):
    out = []
    for s in range(0, z.shape[0], 256):
        out.append(sampler.pred(z[s:s+256], gene_size=GENE_SIZE).detach().cpu().numpy())
    return np.concatenate(out, 0)

conds = np.unique(g)
recs, VR, VS, VZ, VT, VC = [], [], [], [], [], []
for i_cond, cond in enumerate(conds):
    m = g == cond
    if m.sum() <= 5: continue
    line = tl[m][0]
    pool = pools.get(line)
    if pool is None or len(pool) == 0: continue
    real = tX[m]; real_mean = real.mean(0)
    idx = np.arange(real.shape[0])
    if len(idx) > MAX_CELLS:
        idx = np.random.RandomState(0).choice(idx, MAX_CELLS, replace=False)
    n = len(idx)
    cidx = np.random.RandomState(i_cond).choice(len(pool), n, replace=(len(pool) < n))
    cf = torch.tensor(pool[cidx], dtype=torch.float32).to('cuda')
    dummy = torch.zeros((n, GENE_SIZE), dtype=torch.float32).to('cuda')

    s0, d0 = smi[m][0], float(np.median(dos[m]))
    fp_r = Drug_dose_encoder([s0],        [d0], comb_num=1)[0]
    fp_s = Drug_dose_encoder([smap[s0]],  [d0], comb_num=1)[0]
    arms = {}
    for tag, fpv in (('REAL', fp_r), ('SHUF', fp_s), ('ZERO', np.zeros_like(fp_r))):
        fp = torch.tensor(np.repeat(fpv[None, :], n, 0), dtype=torch.float32).to('cuda')
        with torch.no_grad():
            z = sampler.model.encoder(dummy, None, fp, cf)
        torch.manual_seed(i_cond); torch.cuda.manual_seed_all(i_cond)
        arms[tag] = decode(z).mean(0)

    predC = pool.mean(0)
    VR.append(arms['REAL']); VS.append(arms['SHUF']); VZ.append(arms['ZERO'])
    VT.append(real_mean);    VC.append(predC)
    _c = float(np.dot(fp_r, fp_s) / (np.linalg.norm(fp_r) * np.linalg.norm(fp_s) + 1e-12))
    fp_sim.append(_c)
    recs.append(dict(cond=str(cond), line=str(line), smiles=s0, swapped=smap[s0],
                     dose=d0, n=int(n), fp_cosine=_c))
    if (i_cond + 1) % 50 == 0:
        print('  %d/%d' % (i_cond + 1, len(conds)), flush=True)

VR, VS, VZ = np.array(VR), np.array(VS), np.array(VZ)
VT, VC     = np.array(VT), np.array(VC)
N = len(recs)
print('\nscored %d conditions' % N)

# ---- 1. determinism vs saved vectors ---------------------------------------
sv = os.path.join(RES, 'vectors_%s.npz' % VEC_TAG)
if os.path.exists(sv):
    d = np.load(sv, allow_pickle=True)
    if d['B'].shape == VR.shape:
        diff = np.abs(d['B'] - VR)
        print('\n=== determinism: REAL arm vs saved B ===')
        print('  max abs diff %.3e | mean abs diff %.3e' % (diff.max(), diff.mean()))
        if diff.max() > 1e-5:
            sys.exit('FATAL: REAL arm does not reproduce saved B (max %.3e > 1e-5). '
                     'Arms are not comparable to prior results; discard this run.' % diff.max())
        print('  PASS (< 1e-5)')
    else:
        print('\n[warn] saved B shape %s != REAL %s, skipping check' % (d['B'].shape, VR.shape))

# ---- 2/3. output sensitivity ------------------------------------------------
DR, DS, DZ = VR - VC, VS - VC, VZ - VC
def sens(DA, DB):
    r = np.array([pearsonr(DA[i], DB[i])[0] for i in range(N)])
    l2 = np.linalg.norm(DA - DB, axis=1) / (np.linalg.norm(DA, axis=1) + 1e-12)
    return r, l2

print('\n=== output sensitivity to fingerprint ===')
print('  %-24s %9s %9s %11s' % ('comparison', 'r (mean)', 'r (med)', 'relL2 (med)'))
for tag, DB in (('REAL vs SHUFFLED', DS), ('REAL vs ZERO (OOD)', DZ)):
    r, l2 = sens(DR, DB)
    print('  %-24s %9.4f %9.4f %11.4f' % (tag, np.nanmean(r), np.nanmedian(r), np.median(l2)))
r_shuf, _l2 = sens(DR, DS)
print('  95th pct of r(REAL,SHUF): %.4f   (if > 0.95, negligible for almost every condition)'
      % np.nanpercentile(r_shuf, 95))
print('\n=== reference anchors (measured, not assumed) ===')
_ln = np.array([r['line'] for r in recs])
for L in np.unique(_ln):
    ix = np.where(_ln == L)[0]
    if len(ix) < 5: continue
    Cc = np.corrcoef(DR[ix]); up = np.triu_indices_from(Cc, k=1)
    print('  %-6s  r(REAL,SHUF) within line %.4f  |  cross-condition r %.4f  <- full-effect anchor'
          % (L, np.nanmean(r_shuf[ix]), np.nanmean(Cc[up])))
print('  fingerprint cosine to deranged partner: median %.4f  max %.4f'
      % (np.median(fp_sim), np.max(fp_sim)))
print('  (high cosine would mean the shuffle paired structurally similar drugs,')
print('   weakening the test)')
print('\n  r ~ 1.0 => swapping the drug changed nothing: fingerprint does not')
print('  reach the output. r << 1 => it does, and the failure is downstream.')

# ---- 4. B-C on DEGs per arm -------------------------------------------------
print('\n=== B-C on top-50 DEGs (Pearson), per arm ===')
print('  %-10s %9s %9s %9s' % ('arm', 'mean', 'median', 'frac>0'))
for tag, V in (('REAL', VR), ('SHUF', VS), ('ZERO', VZ)):
    vals = []
    for i in range(N):
        deg = np.argsort(np.abs(VT[i] - VC[i]))[-50:]
        b = pearsonr(V[i][deg], VT[i][deg])[0]
        c = pearsonr(VC[i][deg], VT[i][deg])[0]
        vals.append(b - c)
    v = np.array(vals)
    print('  %-10s %9.4f %9.4f %9.3f' % (tag, np.nanmean(v), np.nanmedian(v), np.nanmean(v > 0)))

np.savez_compressed(os.path.join(RES, 'fingerprint_ablation_%s.npz' % ABL_TAG),
                    real=VR, shuf=VS, zero=VZ, true=VT, ctrl=VC,
                    cond=np.array([r['cond'] for r in recs]),
                    line=np.array([r['line'] for r in recs]))
json.dump({'records': recs, 'shuffle_map': smap, 'seed': SEED,
           'fp_cosine_median': float(np.median(fp_sim))},
          open(os.path.join(RES, 'fingerprint_ablation_meta_%s.json' % ABL_TAG), 'w'), indent=2)
print('\nwrote leakage_results/fingerprint_ablation_%s.npz' % ABL_TAG)
