"""Step 14, facets v2 (Amendment 2): the six VCC-style facets, adapted to the sci-Plex held-out sets as PREREG_step14
section 4 states, with two corrections against the cell-eval2 0.16.0 source made after the first run:
  * the Wilcoxon rank-sum p-value is the tie-corrected normal approximation WITHOUT a continuity correction, as
    scanpy's rank_genes_groups(method="wilcoxon", tie_correct=True) computes it (cell_eval2/de_compute.py); v1 added
    scipy's continuity correction;
  * lfc_nmae_anchor() implements the replicate estimator of the fold-change NMAE anchor (cell_eval2/lfc_nmae_ref.py):
    gate and denominator from the FULL reference table, only the two fold-change vectors from the halves.
Everything else is byte-identical to step14_metrics.py.

Definitions follow cell-eval2 0.15.0, docs/vcc2026_metrics/vcc2026-metrics-brief.md (2026-08-19), sections 1-7.
Adaptations (all forced by the data, none a choice among alternatives):
  * gene space: the split's 200 highly variable genes; values are the stored library-size-normalised
    log1p expression; the DE test and fold changes use expm1 of those values (normalised counts);
  * a "context" is one cell line; its perturbation panel is the line's held-out conditions;
  * group profiles for PDS / MSE are means of the stored log1p values over cells (no count pseudobulk);
    effects are taken against the line's vehicle-pool profile on both sides;
  * expression error is the plain ratio sum ||b_hat - b||^2 / sum ||b - b_ctrl||^2 (no jackknife term);
  * no target-gene exclusion (a compound has no target gene in the panel); no CPM gate;
  * conditions with fewer than MIN_CELLS real cells are excluded from the four DE facets and from the
    replicate anchor for every submission alike (the gate reads the reference alone).
Everything else - Wilcoxon rank-sum two-sided, BH within each condition at alpha 0.05, lfc with
epsilon 1e-9, adjudicability, fidelity F = k / max(n_pred, n_real), reach with purity floor 0.9,
significant-set Jaccard with empty union = 1, lfc NMAE with a gate of >= 10 genes - is as specified.
No number produced here is comparable to a challenge leaderboard entry.
"""
import numpy as np
from scipy.stats import rankdata, norm

ALPHA_BH = 0.05
EPS_LFC = 1e-9
P0_REACH = 0.9
MIN_GATE_NMAE = 10


# ------------------------------------------------------------------ the differential-expression table (spec section 3)
def _mannwhitney_p(X, Y):
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U) p-values for every column of X (n1 x G) against Y (n2 x G):
    normal approximation to the rank sum with tie correction and no continuity correction (scanpy's convention)."""
    n1, n2 = X.shape[0], Y.shape[0]
    Z = np.vstack([X, Y]).astype(np.float64)
    R = rankdata(Z, axis=0)                                   # average ranks per column
    U1 = R[:n1].sum(axis=0) - n1 * (n1 + 1) / 2.0            # U statistic of X
    mu = n1 * n2 / 2.0
    N = n1 + n2
    # tie correction per column: sum over tie groups of (t^3 - t)
    tie = np.zeros(Z.shape[1])
    Zs = np.sort(Z, axis=0)
    for g in range(Z.shape[1]):
        _, counts = np.unique(Zs[:, g], return_counts=True)
        tie[g] = np.sum(counts.astype(np.float64) ** 3 - counts)
    sigma = np.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1))))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.abs(U1 - mu) / sigma                           # no continuity correction (scanpy's convention)
        z = np.where(sigma > 0, z, 0.0)
        p = 2.0 * norm.sf(z)
    p = np.where(sigma > 0, p, 1.0)                           # a constant column (all ties) is not significant
    return np.minimum(p, 1.0)


def _bh(p):
    """Benjamini-Hochberg adjusted p-values over one condition's genes (the spec's per-perturbation FDR scope)."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1.0)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(adj, 1.0)
    return out


def de_table(cells_log, ctrl_log):
    """DE table of one group against the vehicle pool. cells_log, ctrl_log: (n x G) stored log1p values.
    Returns dict(p, padj, sig, lfc) over the G genes; lfc = log2((mean_pert + eps) / (mean_ctrl + eps)) on expm1 values."""
    Xc, Yc = np.expm1(cells_log), np.expm1(ctrl_log)
    p = _mannwhitney_p(Xc, Yc)
    padj = _bh(p)
    lfc = np.log2((Xc.mean(axis=0) + EPS_LFC) / (Yc.mean(axis=0) + EPS_LFC))
    return dict(p=p, padj=padj, sig=padj < ALPHA_BH, lfc=lfc)


# ------------------------------------------------------------------ the four DE facets (spec sections 4-7), per condition
def adjudicable(ref):
    lfc = ref['lfc']
    return np.isfinite(lfc) & (lfc != 0) | np.isinf(lfc)      # defined and non-zero; infinite carries a direction


def direction_fidelity(ref, sub):
    """F = k / max(n_pred, n_real); None when the perturbation is dropped (0/0)."""
    adj = adjudicable(ref)
    n_real = int(ref['sig'].sum())
    call = sub['sig'] & adj
    n_pred = int(call.sum())
    k = int((call & (np.sign(sub['lfc']) == np.sign(ref['lfc']))).sum())
    if n_pred == 0 and n_real == 0: return None
    if n_pred == 0: return 0.0
    if n_real == 0: return k / n_pred
    return k / max(n_pred, n_real)


def direction_reach(ref, sub, gene_names):
    """R = k* / n_real, k* the deepest prefix of the submission's ranking of the reference's significant genes
    whose directional purity is >= P0 (counting adjudicable genes only). None when the pool is empty."""
    adj = adjudicable(ref)
    pool = np.where(ref['sig'])[0]
    n_real = len(pool)
    if n_real == 0: return None
    # submission's confidence order: significant first, then ascending padj, ascending p, descending |lfc|, gene name
    keys = [(0 if sub['sig'][g] else 1, sub['padj'][g], sub['p'][g], -abs(sub['lfc'][g]) if np.isfinite(sub['lfc'][g]) else 0.0, str(gene_names[g])) for g in pool]
    order = [pool[i] for i in sorted(range(n_real), key=lambda i: keys[i])]
    match = []
    for g in order:
        if not adj[g]: continue
        s_ = sub['lfc'][g]                                     # a null / NaN prediction has no direction and is a miss
        match.append((not np.isnan(s_)) and np.sign(s_) == np.sign(ref['lfc'][g]))
    if not match: return 0.0
    cum = np.cumsum(np.array(match, dtype=float))
    purity = cum / (np.arange(len(match)) + 1.0)
    ok = np.where(purity >= P0_REACH)[0]
    k_star = int(ok[-1] + 1) if len(ok) else 0                # deepest, not first, prefix clearing the floor
    return k_star / n_real


def sig_jaccard(ref, sub):
    a, b = ref['sig'], sub['sig']
    u = int((a | b).sum())
    return 1.0 if u == 0 else int((a & b).sum()) / u


def lfc_nmae(ref, sub):
    """NMAE over the reference's significant genes with finite lfc; None when the gate holds < MIN_GATE_NMAE genes
    or the denominator is zero / non-finite. A missing or non-finite predicted lfc is read as zero."""
    gate = ref['sig'] & np.isfinite(ref['lfc'])
    if int(gate.sum()) < MIN_GATE_NMAE: return None
    r = ref['lfc'][gate]
    s = np.where(np.isfinite(sub['lfc'][gate]), sub['lfc'][gate], 0.0)
    den = np.abs(r).sum()
    if not np.isfinite(den) or den == 0: return None
    return float(np.abs(s - r).sum() / den)


def lfc_nmae_anchor(ref_full, tab_a, tab_b):
    """Replicate anchor of the fold-change NMAE for one condition (cell_eval2/lfc_nmae_ref.py, nmae_ref_raw):
    gate S and denominator from the FULL reference table, numerator |lfc_A - lfc_B| over S from the two half tables
    (each half against its own vehicle half). None under the same gate rule as lfc_nmae (< MIN_GATE_NMAE genes, or a
    zero / non-finite denominator); a non-finite half fold change is read as zero, as a submission's would be."""
    gate = ref_full['sig'] & np.isfinite(ref_full['lfc'])
    if int(gate.sum()) < MIN_GATE_NMAE: return None
    r = ref_full['lfc'][gate]
    a = np.where(np.isfinite(tab_a['lfc'][gate]), tab_a['lfc'][gate], 0.0)
    b = np.where(np.isfinite(tab_b['lfc'][gate]), tab_b['lfc'][gate], 0.0)
    den = np.abs(r).sum()
    if not np.isfinite(den) or den == 0: return None
    return float(np.abs(a - b).sum() / den)


# ------------------------------------------------------------------ the two profile facets (spec sections 1-2), per context
def pds_cosine(pred_eff, true_eff):
    """pred_eff, true_eff: (n x G) predicted and measured effects of one context's n conditions (profile minus the
    vehicle-pool profile). PDS_p = 1 - k_p / (n - 1), k_p the rank of the correct match by cosine distance, ties averaged;
    a zero-norm vector has distance 1 to everything."""
    def unit(M):
        nrm = np.linalg.norm(M, axis=1, keepdims=True)
        return np.where(nrm > 0, M / np.where(nrm > 0, nrm, 1.0), 0.0), (nrm[:, 0] > 0)
    P, okp = unit(pred_eff); T, okt = unit(true_eff)
    D = 1.0 - P @ T.T                                          # d[p, q]
    D[~okp, :] = 1.0; D[:, ~okt] = 1.0
    n = D.shape[0]
    out = np.empty(n)
    for p in range(n):
        d = D[p]; own = d[p]
        k = np.sum(d < own) + 0.5 * (np.sum(d == own) - 1)
        out[p] = 1.0 - k / (n - 1)
    return out


def expr_mse_ratio(pred_prof, true_prof, ctrl_prof):
    """Panel ratio sum_p ||b_hat_p - b_p||^2 / sum_p ||b_p - b_ctrl||^2 for one context (no per-condition value)."""
    num = float(((pred_prof - true_prof) ** 2).sum())
    den = float(((true_prof - ctrl_prof[None, :]) ** 2).sum())
    return num / den if den > 0 else np.nan


FACETS = ('pds', 'mse', 'fidelity', 'reach', 'sig_jaccard', 'lfc_nmae')
HIGHER_IS_BETTER = dict(pds=True, mse=False, fidelity=True, reach=True, sig_jaccard=True, lfc_nmae=False)


def scaled(u, b, r):
    """s = (u - b) / (r - b): 0 at the mean-response baseline, 1 at the replicate; unclamped except as the spec states."""
    if any(v is None or not np.isfinite(v) for v in (u, b, r)) or r == b: return np.nan
    return float((u - b) / (r - b))
