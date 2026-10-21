#!/usr/bin/env python3
"""
Where is drug identity lost: encoder or decoder?
Same estimator at both layers, so no normalisation choice to argue about.
(Step 4 rationale, final form. CPU only, seconds.)

==============================================================================
WHY THIS EXISTS
==============================================================================
Two earlier normalisations disagreed by an order of magnitude on the same data:
  relative L2       latent/output ratio 1.04x, 1.04x, 1.34x  (signal passes
                    through roughly unchanged)
  share-of-anchor   latent/output ratio 7x, 23x, 5x          (decoder discards
                    ~90% of it)

Both are artifacts of their denominators:
  - The OUTPUT anchor (cross-condition 1-r, 0.53-0.80) includes DDIM decode
    noise and per-condition control-cell draws. The LATENT anchor (0.045-0.137)
    does not, because the encoder is deterministic. So the output denominator
    is inflated by noise and its share deflated.
  - Latent relL2 divides by ||z_real||, which retains a large shared component
    (cross-condition r ~ 0.90). Output relL2 divides by ||delta||, from which
    the control mean has already been subtracted. Not parallel quantities.

This replaces both with one estimator applied identically at each layer:
eta^2 for drug identity across 443 conditions with 37 drug levels. Because the
factor, level count and n are identical at both layers, eta^2's upward bias is
the same on both sides and cancels in the ratio. The permutation nulls verify
that empirically rather than assuming it.

==============================================================================
DATA ASSUMPTIONS
==============================================================================
* zsem_sensitivity.npz: 443 conditions x 60 dims (z_real), plus cond/line/drug.
  Encoder output on seedvar/seed_0, averaged over up to 200 cells per
  condition. The encoder is deterministic given its inputs.
* fingerprint_ablation_drug_split_1.npz: 443 x 200, arms real and ctrl. Same
  checkpoint, same control draws (RandomState(i_cond)), same condition order -
  asserted below.
* UNIT IS THE CONDITION, not the cell. 443 observations, 37 drugs, so ~12
  conditions per drug (4 doses x 3 cell lines).
* ASYMMETRY CORRECTED: the output is a DELTA (per-cell-line control mean
  subtracted); the latent is not baseline-corrected. Both are therefore also
  reported after centring within cell line, which is the like-for-like
  comparison and is PRIMARY. The uncorrected pair is shown for transparency.
* drug_split_1: 37 held-out drugs x 4 doses (NANOMOLAR 10/100/1000/10000)
  x 3 lines (A549, K562, MCF7).

==============================================================================
WHAT IS COMPUTED
==============================================================================
Per dimension, one-way eta^2 = SS_between / SS_total, aggregated two ways:
  mean over dimensions   - each dimension counts equally
  variance-weighted      - sum(SS_between)/sum(SS_total), the multivariate
                           analogue, dominated by high-variance dimensions
                           [PRIMARY: the 60 latent dims have unequal scales]

Factors: drug (37 levels), cell line (3), dose (4).

TWO PERMUTATION NULLS. eta^2 has substantial upward bias under the null with 37
levels and 443 observations (roughly (k-1)/(n-1) = 36/442 = 0.081):
  global      - drug labels shuffled across all conditions
  stratified  - drug labels permuted WITHIN each dose x cell-line stratum, so
                dose response and line identity are held fixed and only drug
                identity is randomised. Each stratum holds ~37 conditions, one
                per drug. This is the stricter null and the primary one.

==============================================================================
REVIEW OUTCOMES (external statistical review)
==============================================================================
(a) Bias cancellation confirmed: one-way eta^2's null bias depends on level
    count k and sample size n, not on the dimensionality of the observation
    vector. Identical at both layers, so it cancels in the ratio. Verified
    empirically by the nulls below.
(b) Within-line centring is the correct equaliser; a control-only latent would
    require a second encoder pass and would mix batch noise into the latent.
(c) Variance-weighted is the headline; dim-mean is the robustness figure.
(d) Treating conditions as observations does conflate dose with drug identity,
    but symmetrically across layers. A dose-factor eta^2 is reported so the
    magnitude is visible, and the stratified null removes the conflation.
(e) Stratified null added per review.
==============================================================================
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, 'leakage_results')
SPLIT, N_PERM, SEED = 'drug_split_1', 500, 0
# Optional argv: [1] zsem suffix, [2] ablation tag, [3] output suffix.
# Defaults reproduce the base-model invocation.
if len(sys.argv) > 4:
    sys.exit('usage: eta_latent_vs_output.py [zsem_suffix] [abl_tag] [out_suffix]')
ZS_TAG  = sys.argv[1] if len(sys.argv) > 1 else ''
ABL_TAG = sys.argv[2] if len(sys.argv) > 2 else 'drug_split_1'
OUT_TAG = sys.argv[3] if len(sys.argv) > 3 else ''
_META = 'fingerprint_ablation_meta.json' if ABL_TAG == 'drug_split_1' \
        else 'fingerprint_ablation_meta_%s.json' % ABL_TAG
rng = np.random.RandomState(SEED)

z  = np.load(os.path.join(RES, 'zsem_sensitivity%s.npz' % ZS_TAG), allow_pickle=True)
ab = np.load(os.path.join(RES, 'fingerprint_ablation_%s.npz' % ABL_TAG), allow_pickle=True)

conds = np.array([str(x) for x in z['cond']])
if not np.array_equal(conds, np.array([str(x) for x in ab['cond']])):
    sys.exit('FATAL: condition order differs between latent and output artifacts')
lines = np.array([str(x) for x in z['line']])
drugs = np.array([str(x) for x in z['drug']])

LAT = np.asarray(z['z_real'], dtype=np.float64)
OUT = np.asarray(ab['real'] - ab['ctrl'], dtype=np.float64)
n = len(conds)
print('aligned on %d conditions | latent %d dims | output %d dims'
      % (n, LAT.shape[1], OUT.shape[1]))
print('factor levels: drug %d, cell line %d' % (len(np.unique(drugs)), len(np.unique(lines))))

meta = json.load(open(os.path.join(RES, _META)))
recs = meta['records'] if isinstance(meta, dict) else meta
dmap = {r['cond']: r['dose'] for r in recs}
doses = np.array([str(dmap[c]) for c in conds])
print('dose levels: %d  (%s)' % (len(np.unique(doses)), ', '.join(sorted(set(doses)))))

def centre_within(X, labels):
    Xc = X.copy()
    for lv in np.unique(labels):
        m = labels == lv
        Xc[m] -= Xc[m].mean(0)
    return Xc

def eta2(X, labels):
    """Returns (per-dim eta^2 array, variance-weighted scalar)."""
    gm = X.mean(0)
    sst = ((X - gm) ** 2).sum(0)
    ssb = np.zeros(X.shape[1])
    for lv in np.unique(labels):
        m = labels == lv
        if m.sum() < 2: continue
        ssb += m.sum() * (X[m].mean(0) - gm) ** 2
    per = np.divide(ssb, sst, out=np.zeros_like(ssb), where=sst > 1e-12)
    vw = ssb.sum() / sst.sum() if sst.sum() > 1e-12 else np.nan
    return per, float(vw)

def null_eta(X, labels, n_perm=N_PERM):
    vals = np.empty(n_perm)
    for i in range(n_perm):
        _, vw = eta2(X, labels[rng.permutation(len(labels))])
        vals[i] = vw
    return float(vals.mean()), float(vals.std(ddof=1)), float(np.percentile(vals, 97.5))

strata = np.array([d + '|' + l for d, l in zip(doses, lines)])

def null_eta_strat(X, labels, n_perm=N_PERM):
    """Stricter null: permute drug labels WITHIN each dose x cell-line stratum,
    holding dose response and line identity fixed."""
    vals = np.empty(n_perm)
    for i in range(n_perm):
        perm = labels.copy()
        for st in np.unique(strata):
            m = strata == st
            perm[m] = rng.permutation(labels[m])
        _, vw = eta2(X, perm)
        vals[i] = vw
    return float(vals.mean()), float(vals.std(ddof=1))

LATc, OUTc = centre_within(LAT, lines), centre_within(OUT, lines)

print('\n=== eta^2 by factor, same estimator at both layers ===')
print('  %-28s %11s %11s %11s %11s' % ('', 'lat mean', 'lat var-wtd', 'out mean', 'out var-wtd'))
rows = {}
for lab, fac in (('drug (37 lv)', drugs), ('cell line (3 lv)', lines), ('dose (4 lv)', doses)):
    for tag, A, B in (('raw', LAT, OUT), ('line-centred', LATc, OUTc)):
        if tag == 'line-centred' and lab == 'cell line (3 lv)': continue
        pl, vl = eta2(A, fac); po, vo = eta2(B, fac)
        print('  %-28s %11.4f %11.4f %11.4f %11.4f'
              % ('%s, %s' % (lab, tag), pl.mean(), vl, po.mean(), vo))
        rows['%s|%s' % (lab, tag)] = dict(lat_mean=float(pl.mean()), lat_vw=vl,
                                          out_mean=float(po.mean()), out_vw=vo)

nulls = {}   # release fix (21 Oct 2026): the nulls and excess ratios were printed only; now recorded too
print('\n=== global permutation null (drug labels shuffled everywhere) ===')
for tag, A, B in (('raw', LAT, OUT), ('line-centred', LATc, OUTc)):
    ml, sl, ql = null_eta(A, drugs)
    mo, so, qo = null_eta(B, drugs)
    _, vl = eta2(A, drugs); _, vo = eta2(B, drugs)
    exl, exo = vl - ml, vo - mo
    nulls['global|%s' % tag] = dict(latent_obs=vl, latent_null_mean=ml, latent_null_sd=sl, latent_null_p975=ql,
                                    output_obs=vo, output_null_mean=mo, output_null_sd=so, output_null_p975=qo,
                                    excess_latent=exl, excess_output=exo,
                                    excess_ratio_latent_over_output=(exl / exo if exo > 1e-9 else None))
    print('  %-14s latent obs %.4f null %.4f+-%.4f (97.5pct %.4f) -> excess %+.4f'
          % (tag, vl, ml, sl, ql, exl))
    print('  %-14s output obs %.4f null %.4f+-%.4f (97.5pct %.4f) -> excess %+.4f'
          % ('', vo, mo, so, qo, exo))
    print('  %-14s excess ratio latent/output: %.2fx' % ('', exl / exo if exo > 1e-9 else np.nan))

print('\n=== stratified null (drugs permuted within dose x line) [PRIMARY] ===')
print('  %d strata, median size %.0f' % (len(np.unique(strata)),
      np.median([int((strata == st).sum()) for st in np.unique(strata)])))
for tag, A, B in (('raw', LAT, OUT), ('line-centred', LATc, OUTc)):
    ml, sl = null_eta_strat(A, drugs)
    mo, so = null_eta_strat(B, drugs)
    _, vl = eta2(A, drugs); _, vo = eta2(B, drugs)
    exl, exo = vl - ml, vo - mo
    nulls['stratified|%s' % tag] = dict(latent_obs=vl, latent_null_mean=ml, latent_null_sd=sl,
                                        output_obs=vo, output_null_mean=mo, output_null_sd=so,
                                        excess_latent=exl, excess_output=exo,
                                        excess_ratio_latent_over_output=(exl / exo if exo > 1e-9 else None))
    print('  %-14s latent obs %.4f null %.4f+-%.4f -> excess %+.4f' % (tag, vl, ml, sl, exl))
    print('  %-14s output obs %.4f null %.4f+-%.4f -> excess %+.4f' % ('', vo, mo, so, exo))
    print('  %-14s excess ratio latent/output: %.2fx' % ('', exl / exo if exo > 1e-9 else np.nan))

print('\n=== per cell line, drug factor (variance-weighted) ===')
print('  %-8s %5s %11s %11s %9s' % ('line', 'n', 'latent', 'output', 'ratio'))
for L in np.unique(lines):
    m = lines == L
    _, vl = eta2(LAT[m], drugs[m]); _, vo = eta2(OUT[m], drugs[m])
    print('  %-8s %5d %11.4f %11.4f %9.2fx' % (L, m.sum(), vl, vo, vl / vo if vo > 1e-9 else np.nan))

rows['permutation_nulls'] = dict(n_perm=N_PERM, seed=SEED, **nulls)   # added key; the eta^2 rows above are unchanged
json.dump(rows, open(os.path.join(RES, 'eta_latent_vs_output%s.json' % OUT_TAG), 'w'), indent=2)
print('\nwrote leakage_results/eta_latent_vs_output%s.json' % OUT_TAG)
