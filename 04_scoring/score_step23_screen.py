#!/usr/bin/env python3
"""
score_step23_screen.py - Step 23, Stage 1: a model-free screen of compound representations, on the development
split (drug_split_1) only.

Purpose
-------
Step 23 asks whether chemical information that the model's fingerprint does not carry (atom identity, the
arrangement of pharmacophore features across the molecule, whole-molecule physicochemical properties) says more
about how an unseen compound changes the cells. Training the model costs 18 GPU runs per representation, so this
screen comes first and costs minutes of CPU. For every held-out condition, the measured responses of the k training
compounds most similar to the held-out compound (same cell line, same dose) are averaged and offered as the
prediction (k-nearest-neighbour transfer). The prediction is scored with the paper's own measures on the paper's own
candidate sets. A representation that knows more about the response finds neighbours whose responses point the right
way more often. Step 23 also asks whether the drug's measured potency (strongest protein-target pChEMBL) says how
strong the response will be; potency carries no direction, so it is screened on response size instead (Qp): does it
rank the measured size beyond what the recipe already predicts? Rules fixed in advance decide whether a chemical
representation, potency, both or neither go on to the model tests (Stage 2).

Pre-registration: PREREG_step23_compound_features.md. This script does only what its sections 2-5 and 8 say.

Logic
-----
1. Guards (prereg 8.1): the pre-registration, this script, the scorers it imports, the Step 15b and 15c
   per-condition records and the Step 23 feature tables are committed and unchanged against HEAD (staged edits
   count as changes); the feature tables' recorded commit is an ancestor of HEAD and the features script is unchanged
   since that commit; no output exists yet. Only drug_split_1 is ever read: there is no option for another split.
2. Inputs (prereg 2): the Step 15b mean-vector files of the recipe and base arms (hash-checked against provenance by
   score_step15c_tahoe.load_vectors); the Step 15b per-condition record for the annotations; the test file's obs
   (annotations only); the Step 15a training file and its row-aligned control file; the feature tables.
3. Identity checks (prereg 8.2-8.3): every structure in the files maps to exactly one structure id through the data
   record; the held-out ids equal the record's held_structures; the as-encoded fingerprint equals the model's own
   encoder for all 372 structures, and equals the dose-scaled input of every held-out condition.
4. Reproduction and checks (prereg 8.4-8.8), before any Step 23 value is printed: the ridge-train shift rebuilt here
   from the training files, and the recipe, reproduce the Step 15c per-condition PDS to 1e-9; the response-size
   agreement (MRA) implemented here reproduces the Step 16 values; the bootstrap reproduces Step 15c's split-1
   interval; the fast bootstrap equals written-out resamples; the partial Spearman equals the closed formula.
5. Scored set (prereg 2): held-out structures whose parent molecule (InChIKey connectivity block) also occurs among
   the training structures are not scored - their nearest neighbour would be themselves. They stay in every
   candidate set as competitors, so the candidate sets are exactly Step 15c's.
6. Representations (prereg 3): Tanimoto on fingerprints, minus Euclidean distance on z-scored numbers (training
   statistics); two-block representations combine the two similarity rankings with equal weight (mean rank).
7. Transfer (prereg 4.1): the unweighted mean measured response of the first k eligible training structures (k = 10
   primary, 5 secondary), eligible = having a training condition at the same line and dose.
8. Scores (prereg 4.2-4.5): PDS (Step 15c), MRA (Step 16 section 4.4), DE-overlap and DPC-50 (Step 15b code; point
   values only); Qp: potency against measured response size - MRA, and the partial Spearman beyond the recipe's own
   predicted size (the gate of the potency route).
9. Intervals (prereg 4.6) and the gated contrasts and both selection routes (prereg 5): the chemistry route
   (rules 5.1-5.3) and the potency route (rule 5.4).
Outputs (never overwritten): tahoe_results/step23_screen.json, step23_screen_per_condition.csv,
step23_screen_summary.csv.

Run on a compute node (no network needed), from the worktree root, through run_step23_screen.sh:
    sbatch run_step23_screen.sh
"""
import csv
import json
import os
import subprocess
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from rdkit import RDLogger
from sklearn.linear_model import Ridge

RDLogger.DisableLog('rdApp.*')                          # RDKit deprecation notices only; no effect on any value
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_step15_tahoe as s15      # noqa: E402  Step 15b scorer: paths, loaders, ridge features, metrics
import score_step15c_tahoe as s15c    # noqa: E402  Step 15c scorer: provenance-checked vectors, PDS on candidate sets

PRE = 'PREREG_step23_compound_features.md'
SPLIT = 'drug_split_1'                                  # the development split; no other split is ever read here
FEAT = os.path.join(s15.RES, 'step23_features')
FEAT_FILES = [os.path.join(FEAT, f) for f in ('structures.csv', 'fingerprints.npz', 'manifest.json')]
PC15C = os.path.join(s15.RES, 'step15c_pds_per_condition.csv')   # Step 15c final per-condition record
OUT_JSON = os.path.join(s15.RES, 'step23_screen.json')
OUT_PC = os.path.join(s15.RES, 'step23_screen_per_condition.csv')
OUT_SUM = os.path.join(s15.RES, 'step23_screen_summary.csv')
K_MAIN, K_SEC = 10, 5                                   # prereg 4.1: neighbours, primary and secondary; not tuned
N_BOOT, LO, HI = s15.N_BOOT, s15.LO, s15.HI             # 5,000 replicates; 1.25th and 98.75th percentiles (97.5 %)
FOLD = 1024                                             # R4: 2D pharmacophore bits folded to the Stage 2 input width
MIN_SECTIONS = 10                                       # the pre-registration has at least ten '## ' sections
T0 = time.time()

# Published values this implementation must reproduce before any Step 23 value is printed (prereg 8.5 and 8.6).
REFERENCE = {
    'source_mra': 'step16_findings.md section 4.4, drug_split_1, printed to four decimals',
    'mra': {'ridge_S7': -0.0499, 'ridge_train': 0.1537, 'base_mean': -0.0755,
            'recipe_seeds': [0.3112, 0.3237, 0.3265], 'recipe_mean': 0.3205},
    'mra_tol': 5e-5 + 1e-9,
    'source_boot': 'step15c_findings.md section 1, drug_split_1, Q1 recipe PDS - 0.5, printed to five decimals',
    'boot_q1': {'diff': 0.22234, 'lo': 0.18829, 'hi': 0.25411},
    'boot_tol': 5e-6 + 1e-9,
}
TEST_REFERENCE = os.environ.get('S23_TEST_REFERENCE')  # synthetic test only: replaces REFERENCE, written to the record

# Representations (prereg 3). Each is one block or two blocks; the first block of a pair breaks remaining ties.
REPS = [
    ('R1', 'FCFP4 as encoded (the model input)', ['fa']),
    ('R2', 'FCFP4 of the parent', ['fp']),
    ('R3', 'FCFP4 parent + ECFP4 parent', ['fp', 'ep']),
    ('R4', 'FCFP4 parent + 2D pharmacophores (1,024 bits)', ['fp', 'phf']),
    ('R5', 'FCFP4 parent + 49 numbers', ['fp', 'num']),
    ('E1', 'ECFP4 parent alone', ['ep']),
    ('E2', '2D pharmacophore fingerprint alone', ['ph']),
    ('E3', '49 numbers alone', ['num']),
    ('E3a', '16 descriptors alone', ['desc']),
    ('E3b', '25 atom and bond counts alone', ['ab']),
    ('E3c', '8 pharmacophore counts alone', ['pc']),
    ('E4', 'FCFP4 parent + 2D pharmacophores unfolded', ['fp', 'ph']),
    ('X1', 'response profile in the other lines (reference)', ['resp']),
    ('X2', 'FCFP4 parent + response profile (method check)', ['fp', 'resp']),
]
GATED = [('P', 'R1', None), ('Q1a', 'R2', 'R1'), ('Q1b', 'R3', 'R2'), ('Q1c', 'R4', 'R2'), ('Q1d', 'R5', 'R2')]
POTENCY = ['pchembl_protein_max', 'pchembl_mechanism_max', 'pchembl_cellline_median']
POTENCY_GATED = 'pchembl_protein_max'                  # prereg 2: the only potency summary allowed as a model input


def say(msg):
    print(msg, flush=True)


def die(msg):
    sys.exit('FATAL: ' + msg)


# ------------------------------------------------------------------ guards
def git(*a):
    return subprocess.run(['git'] + list(a), capture_output=True, text=True)


def committed_and_clean(paths, what):
    """Tracked, and identical to HEAD in the working tree (a staged but uncommitted edit also fails)."""
    r = git('ls-files', '--error-unmatch', *paths)
    if r.returncode:
        die('%s not committed: %s' % (what, r.stderr.strip()))
    if git('diff', '--quiet', 'HEAD', '--', *paths).returncode:
        die('%s differ from the last commit (commit them first)' % what)


# ------------------------------------------------------------------ similarities and neighbour order (prereg 3)
def tanimoto(A, B):
    """Tanimoto similarity between the rows of two 0/1 matrices; 0 where both rows are empty."""
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    inter = A @ B.T                                          # counts are exact in float32 (far below 2**24)
    den = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.where(den > 0, inter / np.where(den > 0, den, 1.0), 0.0).astype(np.float64)


def order_single(sim):
    """Training structures from most to least similar; ties go to the lower training index (sorted structure id)."""
    return np.lexsort((np.arange(len(sim)), -sim))


def order_pair(sim_a, sim_b):
    """Two blocks with equal weight: rank by each similarity (1 = most similar, ties averaged), order by the mean of
    the two ranks; remaining ties go to the higher first-block similarity, then to the lower training index."""
    ra = rankdata(-sim_a, method='average')
    rb = rankdata(-sim_b, method='average')
    return np.lexsort((np.arange(len(sim_a)), -sim_a, (ra + rb) / 2.0))


# ------------------------------------------------------------------ response-size agreement (Step 16 section 4.4)
def spearman_point(a, b):
    """Spearman correlation = Pearson correlation of average ranks; 0 when either side is constant."""
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def mra_point(pn, tn, ctx_code, idx):
    """MRA: within each (line, dose) context, Spearman between predicted and measured response size; mean over the
    contexts present among idx. pn may be any per-condition size predictor (a norm, or a potency value for Qp)."""
    cc = ctx_code[idx]
    return float(np.mean([spearman_point(pn[idx[cc == c]], tn[idx[cc == c]]) for c in np.unique(cc)]))


def weighted_ranks(v, W):
    """Average ranks of v when item j appears W[b, j] times in replicate b (vectorised over replicates). Copies of an
    item, and items with equal values, share one average rank - exactly what rankdata gives on the expanded list."""
    o = np.argsort(v, kind='mergesort')
    vs = v[o]
    new = np.r_[True, vs[1:] != vs[:-1]]                   # start of each group of equal values
    starts = np.flatnonzero(new)
    gid = np.cumsum(new) - 1                               # group of each sorted position
    GS = np.add.reduceat(W[:, o], starts, axis=1)          # weight of each group per replicate
    rank_g = np.cumsum(GS, axis=1) - GS + (GS + 1.0) / 2.0
    R = np.empty_like(W, dtype=np.float64)
    R[:, o] = rank_g[:, gid]
    return R


def weighted_pearson(Rp, Rt, W):
    """Pearson correlation per replicate of two rank matrices with item weights W; 0 if a side is constant, NaN if the
    replicate drew nothing in this context."""
    sw = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mp = (W * Rp).sum(1) / sw
        mt = (W * Rt).sum(1) / sw
        dp = Rp - mp[:, None]
        dt = Rt - mt[:, None]
        cov = (W * dp * dt).sum(1)
        vp = (W * dp * dp).sum(1)
        vt = (W * dt * dt).sum(1)
        rho = np.where((vp > 1e-9) & (vt > 1e-9), cov / np.sqrt(np.maximum(vp * vt, 1e-300)), 0.0)
    rho[sw == 0] = np.nan
    return rho


def weighted_partial(Rx, Ry, Rz, W):
    """Partial correlation per replicate of rank matrices Rx and Ry given Rz, with item weights W: the weighted Pearson
    correlation of the residuals of Rx and Ry after a weighted least-squares fit on Rz (the residual construction of
    Step 19). 0 if a residual is (near) constant; NaN if the replicate drew nothing in this context."""
    sw = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        def centred(R):
            return R - ((W * R).sum(1) / sw)[:, None]
        dx, dy, dz = centred(Rx), centred(Ry), centred(Rz)
        vz = (W * dz * dz).sum(1)
        okz = vz > 1e-12
        bx = np.where(okz, (W * dx * dz).sum(1) / np.where(okz, vz, 1.0), 0.0)   # slope of x-ranks on z-ranks
        by = np.where(okz, (W * dy * dz).sum(1) / np.where(okz, vz, 1.0), 0.0)   # slope of y-ranks on z-ranks
        ex = dx - bx[:, None] * dz                                               # what z does not explain
        ey = dy - by[:, None] * dz
        vx0, vy0 = (W * dx * dx).sum(1), (W * dy * dy).sum(1)
        vx, vy = (W * ex * ex).sum(1), (W * ey * ey).sum(1)
        ok = (vx0 > 0) & (vy0 > 0) & (vx > 1e-10 * vx0) & (vy > 1e-10 * vy0)
        rho = np.where(ok, (W * ex * ey).sum(1) / np.sqrt(np.where(ok, vx * vy, 1.0)), 0.0)
    rho[sw == 0] = np.nan
    return rho


def partial_point(x, y, z, ctx_code, idx):
    """Partial Spearman of x and y given z within each (line, dose) context present among idx (average ranks, residual
    construction), mean over the contexts. Duplicated indices are ranked as ties, as in a bootstrap resample."""
    cc = ctx_code[idx]
    vals = []
    for c in np.unique(cc):
        e = idx[cc == c]
        R = [rankdata(np.asarray(v)[e])[None, :] for v in (x, y, z)]
        vals.append(float(weighted_partial(R[0], R[1], R[2], np.ones((1, len(e))))[0]))
    return float(np.mean(vals))


def partial_closed(x, y, z):
    """The textbook closed formula from the three Spearman correlations; used only to check partial_point."""
    rxy, rxz, ryz = spearman_point(x, y), spearman_point(x, z), spearman_point(y, z)
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


# ------------------------------------------------------------------ bootstrap (prereg 4.6)
class Boot:
    """Structure-clustered percentile bootstrap over one set of conditions. Clusters are the structures (SMILES) in the
    set, in sorted order; the draws are those of score_step15_tahoe.boot (rng.randint(0, G, G) per replicate) from a
    fresh RandomState(0), so every interval of one set uses the same 5,000 resamples."""

    def __init__(self, idx, clusters, n_total):
        self.idx = np.asarray(idx)
        lab = np.asarray(clusters)[self.idx]
        self.labels = np.unique(lab)
        self.G = len(self.labels)
        self.member = np.searchsorted(self.labels, lab)              # cluster number of each condition in the set
        self.groups = [self.idx[self.member == g] for g in range(self.G)]
        self.mem_full = np.full(n_total, -1)
        self.mem_full[self.idx] = self.member
        rng = np.random.RandomState(0)
        self.D = np.array([rng.randint(0, self.G, self.G) for _ in range(N_BOOT)])
        self.cnt = np.zeros((N_BOOT, self.G))                      # how often each cluster is drawn, per replicate
        np.add.at(self.cnt, (np.repeat(np.arange(N_BOOT), self.G), self.D.ravel()), 1.0)

    def expand(self, b):
        """Replicate b written out literally: the conditions of every drawn cluster, in draw order (for the checks)."""
        return np.concatenate([self.groups[g] for g in self.D[b]])

    def mean_reps(self, x):
        """Replicate means of per-condition values x over the set; the values are fixed under resampling (Step 15c)."""
        v = np.asarray(x, np.float64)[self.idx]
        if not np.isfinite(v).all():
            die('non-finite per-condition value inside a scored set')
        S = np.bincount(self.member, weights=v, minlength=self.G)
        N = np.bincount(self.member, minlength=self.G).astype(np.float64)
        return float(v.mean()), (self.cnt @ S) / (self.cnt @ N)

    def mra_reps(self, preds, tn, ctx_code):
        """Replicate MRA of each predictor (a list of per-condition size arrays; a list of arrays inside the list is
        a multi-seed predictor whose MRA is averaged over seeds), recomputed per replicate (Step 16). Returns the
        point values and an (N_BOOT, n_predictors) array."""
        flat, owner = [], []
        for j, p in enumerate(preds):
            for q in (p if isinstance(p, list) else [p]):
                flat.append(np.asarray(q, np.float64))
                owner.append(j)
        owner = np.array(owner)
        nb = self.cnt.shape[0]
        acc = np.zeros((nb, len(flat)))
        ncx = np.zeros((nb, len(flat)))
        acc1 = np.zeros(len(flat))
        cc = ctx_code[self.idx]
        contexts = np.unique(cc)
        for c in contexts:
            items = self.idx[cc == c]
            W = self.cnt[:, self.mem_full[items]]
            W1 = np.ones((1, len(items)))
            Rt, Rt1 = weighted_ranks(tn[items], W), weighted_ranks(tn[items], W1)
            for f, pn in enumerate(flat):
                rho = weighted_pearson(weighted_ranks(pn[items], W), Rt, W)
                ok = np.isfinite(rho)
                acc[ok, f] += rho[ok]
                ncx[ok, f] += 1
                acc1[f] += weighted_pearson(weighted_ranks(pn[items], W1), Rt1, W1)[0]
        reps_flat = acc / ncx
        point_flat = acc1 / len(contexts)
        reps = np.column_stack([reps_flat[:, owner == j].mean(1) for j in range(len(preds))])
        point = np.array([point_flat[owner == j].mean() for j in range(len(preds))])
        return point, reps

    def partial_reps(self, x, y, z, ctx_code):
        """Replicate partial Spearman of x and y given z (mean over contexts), recomputed per replicate (prereg 4.4).
        Returns the point value and an (N_BOOT,) array."""
        nb = self.cnt.shape[0]
        acc, ncx, acc1 = np.zeros(nb), np.zeros(nb), 0.0
        cc = ctx_code[self.idx]
        contexts = np.unique(cc)
        for c in contexts:
            items = self.idx[cc == c]
            W = self.cnt[:, self.mem_full[items]]
            W1 = np.ones((1, len(items)))
            rho = weighted_partial(weighted_ranks(x[items], W), weighted_ranks(y[items], W), weighted_ranks(z[items], W), W)
            ok = np.isfinite(rho)
            acc[ok] += rho[ok]
            ncx[ok] += 1
            acc1 += weighted_partial(weighted_ranks(x[items], W1), weighted_ranks(y[items], W1),
                                     weighted_ranks(z[items], W1), W1)[0]
        return float(acc1 / len(contexts)), acc / ncx


def interval(point, reps):
    """Percentile interval (1.25th, 98.75th) of the replicate values; verdict against zero as score_step15_tahoe."""
    lo, hi = float(np.percentile(reps, LO)), float(np.percentile(reps, HI))
    return dict(diff=float(point), ci=[lo, hi], verdict=s15.verdict(lo, hi), half_width=float((hi - lo) / 2))


def fmt(c):
    return '%+.5f [%+.5f, %+.5f] %s' % (c['diff'], c['ci'][0], c['ci'][1], c['verdict'])


# ------------------------------------------------------------------ main
def main():
    if len(sys.argv) > 1:
        die('this script takes no options (prereg 9: one run, development split only)')
    for q in (OUT_JSON, OUT_PC, OUT_SUM):
        if os.path.exists(q):
            die('%s exists; refusing to overwrite a record' % q)
    if not os.path.isfile(PRE) or open(PRE).read().count('\n## ') < MIN_SECTIONS:
        die('pre-registration missing or incomplete')
    committed_and_clean([PRE, 'score_step23_screen.py', 'score_step15_tahoe.py', 'score_step15c_tahoe.py',
                         'step14_metrics.py', 'step23_compound_features.py'], 'pre-registration or scripts')
    committed_and_clean([s15c.PC15, PC15C, s15.RECORD] + FEAT_FILES, 'records or feature tables')
    man = json.load(open(os.path.join(FEAT, 'manifest.json')))
    fcommit = man.get('commit') or ''
    if not fcommit:
        die('the feature tables record no commit (built with --no-git-check?); rebuild them from a committed script')
    if git('merge-base', '--is-ancestor', fcommit, 'HEAD').returncode:
        die('the feature tables were built at %s, which is not an ancestor of HEAD' % fcommit[:7])
    if git('diff', '--quiet', fcommit, 'HEAD', '--', 'step23_compound_features.py').returncode:
        die('step23_compound_features.py changed since the feature tables were built (%s)' % fcommit[:7])
    commit = git('rev-parse', 'HEAD').stdout.strip()
    ref = REFERENCE
    if TEST_REFERENCE:
        ref = json.load(open(TEST_REFERENCE))
        say('[WARNING] S23_TEST_REFERENCE set: reproduction targets read from %s (synthetic test only)' % TEST_REFERENCE)
    say('[guards] pre-registration, scripts, records and feature tables committed and clean; commit %s; '
        'feature tables built at %s; split %s (development only)' % (commit[:7], fcommit[:7], SPLIT))
    s15c.self_check()

    # ---------------------------------------------------------------- vectors (recipe, base; three seeds each)
    arms = {a: [s15c.load_vectors(SPLIT, a, s) for s in range(s15.N_SEEDS)] for a in ('recipe', 'base')}
    ref0 = arms['recipe'][0]
    cond, n = ref0['cond'], len(ref0['cond'])
    for a, L in arms.items():
        for v in L:
            if not np.array_equal(v['cond'], cond) or np.abs(v['C'] - ref0['C']).max() > 1e-5 \
                    or np.abs(v['true'] - ref0['true']).max() > 1e-5:
                die('condition order, control means or true means differ in %s' % v['tag'])
    true_mean, C = ref0['true'], ref0['C']
    Y = true_mean - C                                          # measured mean response per held-out condition
    tn = np.linalg.norm(Y, axis=1)                             # measured response size
    record = json.load(open(s15.RECORD))
    rec_split = record['splits'][SPLIT.split('_')[-1]]
    if n != int(rec_split['n_test_conditions']):
        die('%d conditions, the data record expects %d' % (n, rec_split['n_test_conditions']))

    # ---------------------------------------------------------------- annotations (Step 15b record, test file obs)
    pc15 = pd.read_csv(s15c.PC15)
    pc = pc15[pc15['split'] == SPLIT].reset_index(drop=True)
    if len(pc) != n or not np.array_equal(pc['condition'].astype(str).values, cond):
        die('the Step 15b per-condition record does not list the split-1 conditions in the vectors order')
    lines = pc['line'].astype(str).values
    plates = pc['plate'].astype(str).values
    drugs = pc['structure'].astype(str).values               # the study structure (SMILES) of each condition
    doses = pc['dose_nM'].astype(float).values
    lv = sorted(set(lines))
    te = ad.read_h5ad('%s/tahoe_test_%s.h5ad' % (s15.DATA, SPLIT), backed='r')
    idx_te = s15.condition_index(te.obs['Group'].astype(str).values)
    smi_all = te.obs['SMILES'].astype(str).values
    dose_all = te.obs['dose'].astype(float).values
    line_all = te.obs['cell_name'].astype(str).values
    for i, c in enumerate(cond):
        ix = idx_te[c]
        if smi_all[ix[0]] != drugs[i] or line_all[ix[0]] != lines[i] or s15.parse_cond(c)[1] != plates[i] \
                or abs(float(np.median(dose_all[ix])) - doses[i]) > 1e-9:
            die('annotation of %s differs between the test file and the Step 15b record' % c)
    te.file.close()

    # ---------------------------------------------------------------- structure identities (prereg 8.3)
    dr = record['drugs']
    smi2sid, sid2smi, sid2names = {}, {}, {}
    for name, smi in dr['canonical_smiles'].items():
        sid = dr['structure'][name]
        if sid2smi.setdefault(sid, smi) != smi:
            die('the data record gives one structure two canonical SMILES (%s)' % name)
        for form in {smi, dr.get('table_smiles', {}).get(name, smi)}:     # canonical, or the table form
            if smi2sid.setdefault(form, sid) != sid:
                die('the data record maps one SMILES to two structures (%s)' % name)
        sid2names.setdefault(sid, []).append(name)
    miss = sorted(set(drugs) - set(smi2sid))
    if miss:
        die('%d held-out SMILES are not SMILES of the data record, e.g. %s' % (len(miss), miss[:2]))
    sid_c = np.array([smi2sid[s] for s in drugs])              # structure id of each held-out condition
    held = sorted(set(sid_c))
    if held != sorted(rec_split['held_structures']):
        die('held-out structure ids differ from the data record held_structures')
    say('[design] %d held-out conditions | %d held-out structures | %d lines | doses %s'
        % (n, len(held), len(lv), sorted(set('%g' % d for d in doses), key=float)))

    # ---------------------------------------------------------------- feature tables (prereg 2, 3)
    fz = np.load(os.path.join(FEAT, 'fingerprints.npz'))
    fsid = [str(x) for x in fz['structure_id']]
    st = pd.read_csv(os.path.join(FEAT, 'structures.csv'), dtype={'structure_id': str, 'parent_inchikey_block': str})
    if fsid != list(st['structure_id']) or fsid != sorted(sid2smi):
        die('feature tables and data record do not list the same structures in the same order')
    frow = {s: j for j, s in enumerate(fsid)}
    FA, FPp, EP, PH = (np.asarray(fz[k]) for k in ('fcfp4_asis', 'fcfp4_parent', 'ecfp4_parent', 'pharm2d_parent'))
    enc = np.array([s15.Drug_dose_encoder([sid2smi[s]], [9.0], comb_num=1)[0] for s in fsid], dtype=np.float64)
    if np.abs(enc - FA).max() > 1e-6:
        die('fcfp4_asis differs from the model encoder for %d structures' % int((np.abs(enc - FA) > 1e-6).any(1).sum()))
    FPh = np.array([s15.Drug_dose_encoder([drugs[i]], [doses[i]], comb_num=1)[0] for i in range(n)], dtype=np.float64)
    if np.abs(FPh - FA[[frow[s] for s in sid_c]] * np.log10(doses + 1.0)[:, None]).max() > 1e-5:
        die('the dose-scaled input of a held-out condition is not fcfp4_asis x log10(dose + 1)')
    say('[identity] every held-out SMILES maps to one structure id; held-out ids equal the record; fcfp4_asis equals '
        'the model encoder for all %d structures and the dose-scaled input of every held-out condition' % len(fsid))
    r_, c_ = np.nonzero(PH)
    PHF = np.zeros((PH.shape[0], FOLD), np.uint8)
    PHF[r_, c_ % FOLD] = 1                                     # R4: bit i -> i mod 1,024 (OR); E2 and E4 use all bits
    block = dict(zip(st['structure_id'], st['parent_inchikey_block']))

    # ---------------------------------------------------------------- training responses (Step 15c code path)
    tr = ad.read_h5ad('%s/tahoe_train_%s.h5ad' % (s15.DATA, SPLIT))
    trc = ad.read_h5ad('%s/tahoe_train_%s_control.h5ad' % (s15.DATA, SPLIT))
    if tr.n_obs != trc.n_obs or not np.array_equal(tr.obs['Group'].astype(str).values, trc.obs['Group'].astype(str).values):
        die('train and control files are not row-aligned')
    gtr, smi_tr = tr.obs['Group'].astype(str).values, tr.obs['SMILES'].astype(str).values
    dose_tr, line_tr = tr.obs['dose'].astype(float).values, tr.obs['cell_name'].astype(str).values
    if set(smi_tr) & set(drugs):
        die('a held-out structure appears in the training file')
    codes, utr = pd.factorize(pd.Series(gtr), sort=False)
    Xtr, Xtrc = s15.dense(tr.X), s15.dense(trc.X)
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(len(utr)))
    cnt = np.bincount(codes, minlength=len(utr)).astype(np.float64)
    Ytr = np.add.reduceat((Xtr[order].astype(np.float64) - Xtrc[order].astype(np.float64)), bounds, axis=0) / cnt[:, None]
    first = order[bounds]
    dose_sorted = dose_tr[order]
    ends = np.append(bounds[1:], len(order))
    dose_med = np.array([float(np.median(dose_sorted[bounds[q]:ends[q]])) for q in range(len(utr))])
    del tr, trc, Xtr, Xtrc
    miss = sorted(set(smi_tr[first]) - set(smi2sid))
    if miss:
        die('%d training SMILES are not SMILES of the data record, e.g. %s' % (len(miss), miss[:2]))
    tsid_q = np.array([smi2sid[smi_tr[j]] for j in first])     # structure id of each training condition
    tline_q = line_tr[first]

    # ---------------------------------------------------------------- PDS candidate sets (Step 15c section 4)
    ctx_key = np.array(['%s|%.6f' % (lines[i], doses[i]) for i in range(n)])
    ctx_list = sorted(set(ctx_key))
    cpos = {c: j for j, c in enumerate(ctx_list)}
    ctx_code = np.array([cpos[c] for c in ctx_key])
    contexts = [np.where(ctx_key == u)[0] for u in np.unique(ctx_key)]
    cand = {}
    for members in contexts:
        for p in members:
            cand[p] = np.array([q for q in members if q == p or drugs[q] != drugs[p]])
    n_cand = np.array([len(cand[p]) for p in range(n)])

    # ---------------------------------------------------------------- reproduction (prereg 8.4-8.6)
    FPtr = np.array([s15.Drug_dose_encoder([smi_tr[j]], [dose_med[q]], comb_num=1)[0] for q, j in enumerate(first)],
                    dtype=np.float64)
    X = s15.features(FPh, lines, lv)
    Ptr = Ridge(alpha=s15.ALPHA).fit(s15.features(FPtr, tline_q, lv), Ytr).predict(X)
    P7 = s15.ridge_s7(X, Y, drugs)
    pds_ridge_train = s15c.pds_sets(Ptr, Y, contexts, cand)
    pds_recipe_seeds = [s15c.pds_sets(v['B'] - v['C'], Y, contexts, cand) for v in arms['recipe']]
    pds_recipe = np.nanmean(np.vstack(pds_recipe_seeds), axis=0)
    p15c = pd.read_csv(PC15C)
    p15c = p15c[p15c['split'] == SPLIT].reset_index(drop=True)
    if len(p15c) != n or not np.array_equal(p15c['condition'].astype(str).values, cond):
        die('the Step 15c per-condition record does not list the split-1 conditions in the vectors order')
    if not s15c.close(pds_ridge_train, p15c['ridge_train_pds'].values) or not s15c.close(pds_recipe, p15c['recipe_pds'].values):
        die('ridge-train or recipe PDS does not reproduce the Step 15c per-condition record to 1e-9')
    if not np.array_equal(n_cand, p15c['n_candidates'].values):
        die('candidate-set sizes differ from the Step 15c record')
    say('[reproduction] ridge-train rebuilt from the training files and the recipe reproduce the Step 15c per-condition '
        'PDS to 1e-9 on all %d conditions (candidate sets %d-%d)' % (n, n_cand.min(), n_cand.max()))
    allidx = np.arange(n)
    mra_chk = {'ridge_S7': mra_point(np.linalg.norm(P7, axis=1), tn, ctx_code, allidx),
               'ridge_train': mra_point(np.linalg.norm(Ptr, axis=1), tn, ctx_code, allidx),
               'recipe_seeds': [mra_point(np.linalg.norm(v['B'] - v['C'], axis=1), tn, ctx_code, allidx) for v in arms['recipe']],
               'base_seeds': [mra_point(np.linalg.norm(v['B'] - v['C'], axis=1), tn, ctx_code, allidx) for v in arms['base']]}
    mra_chk['recipe_mean'] = float(np.mean(mra_chk['recipe_seeds']))
    mra_chk['base_mean'] = float(np.mean(mra_chk['base_seeds']))
    tol = ref['mra_tol']
    pairs = [('ridge_S7', mra_chk['ridge_S7'], ref['mra']['ridge_S7']),
             ('ridge_train', mra_chk['ridge_train'], ref['mra']['ridge_train']),
             ('recipe_mean', mra_chk['recipe_mean'], ref['mra']['recipe_mean']),
             ('base_mean', mra_chk['base_mean'], ref['mra']['base_mean'])] + \
        [('recipe_seed_%d' % s, mra_chk['recipe_seeds'][s], ref['mra']['recipe_seeds'][s]) for s in range(s15.N_SEEDS)]
    bad = [(k, a, b) for k, a, b in pairs if abs(a - b) > tol]
    if bad:
        die('MRA does not reproduce %s: %s' % (ref['source_mra'], ['%s %.6f vs %.4f' % x for x in bad]))
    say('[reproduction] MRA reproduces %s: ridge-S7 %+.4f, ridge-train %+.4f, recipe %+.4f (seeds %s), base %+.4f'
        % (ref['source_mra'], mra_chk['ridge_S7'], mra_chk['ridge_train'], mra_chk['recipe_mean'],
           ' '.join('%.4f' % x for x in mra_chk['recipe_seeds']), mra_chk['base_mean']))
    b_all = Boot(allidx, drugs, n)
    pt, reps = b_all.mean_reps(pds_recipe - 0.5)
    q1 = interval(pt, reps)
    btol = ref['boot_tol']
    if abs(q1['diff'] - ref['boot_q1']['diff']) > btol or abs(q1['ci'][0] - ref['boot_q1']['lo']) > btol \
            or abs(q1['ci'][1] - ref['boot_q1']['hi']) > btol:
        die('the bootstrap does not reproduce %s: got %s' % (ref['source_boot'], fmt(q1)))
    say('[reproduction] bootstrap reproduces %s: %s' % (ref['source_boot'], fmt(q1)))
    # internal check: the vectorised replicate statistics equal the literal ones computed on written-out resamples
    pn_chk = np.linalg.norm(Ptr, axis=1)
    pchk, rchk = b_all.mra_reps([pn_chk], tn, ctx_code)
    if abs(pchk[0] - mra_chk['ridge_train']) > 1e-10:
        die('the vectorised MRA differs from the plain MRA (%.12f vs %.12f)' % (pchk[0], mra_chk['ridge_train']))
    _, mchk = b_all.mean_reps(pds_recipe)
    for b in range(3):
        e = b_all.expand(b)
        if abs(mra_point(pn_chk, tn, ctx_code, e) - rchk[b, 0]) > 1e-10 or abs(pds_recipe[e].mean() - mchk[b]) > 1e-12:
            die('the vectorised bootstrap differs from the literal resample %d' % b)
    say('[check] the vectorised bootstrap equals the literal resamples (mean and MRA, replicates 0-2)')
    # partial Spearman (prereg 4.4, 8.8), checked on quantities that are not Step 23 results: the ridge-train size
    # against the measured size, beyond the recipe's predicted size (mean of the three seeds' lengths)
    rec_size = np.mean([np.linalg.norm(v['B'] - v['C'], axis=1) for v in arms['recipe']], axis=0)
    n_cmp = 0
    for c in np.unique(ctx_code):
        e = allidx[ctx_code == c]
        if abs(spearman_point(pn_chk[e], rec_size[e])) < 0.99 and abs(spearman_point(tn[e], rec_size[e])) < 0.99:
            if abs(partial_closed(pn_chk[e], tn[e], rec_size[e]) - partial_point(pn_chk, tn, rec_size, ctx_code, e)) > 1e-9:
                die('the partial Spearman (residual construction) differs from the closed formula in a context')
            n_cmp += 1
    ppt_chk, preps_chk = b_all.partial_reps(pn_chk, tn, rec_size, ctx_code)
    if abs(ppt_chk - partial_point(pn_chk, tn, rec_size, ctx_code, allidx)) > 1e-10:
        die('the vectorised partial Spearman differs from the plain one')
    for b in range(3):
        if abs(partial_point(pn_chk, tn, rec_size, ctx_code, b_all.expand(b)) - preps_chk[b]) > 1e-10:
            die('the vectorised partial Spearman differs from the literal resample %d' % b)
    say('[check] partial Spearman: the residual construction equals the closed formula on %d well-conditioned contexts; '
        'the vectorised replicates equal the literal resamples (replicates 0-2)' % n_cmp)

    # ---------------------------------------------------------------- training responses per (structure, context)
    tsids = sorted(set(tsid_q))
    tpos = {s: j for j, s in enumerate(tsids)}
    n_t, G = len(tsids), Y.shape[1]
    TDs = np.zeros((len(ctx_list), n_t, G))
    TDn = np.zeros((len(ctx_list), n_t))
    for q in range(len(utr)):
        c = cpos.get('%s|%.6f' % (tline_q[q], dose_med[q]))
        if c is None:
            continue                                           # a training context with no held-out condition
        TDs[c, tpos[tsid_q[q]]] += Ytr[q]
        TDn[c, tpos[tsid_q[q]]] += 1                           # plate copies are averaged below
    avail = TDn > 0
    with np.errstate(invalid='ignore'):
        TDm = TDs / TDn[:, :, None]                            # NaN where a training structure lacks the context
    if avail.sum(1).min() < K_MAIN:
        die('a held-out context has fewer than %d training structures' % K_MAIN)
    trow = [frow[s] for s in tsids]

    # ---------------------------------------------------------------- scored set: no salt or hydrate twins (prereg 2)
    tblocks = {block[s] for s in tsids}
    twins = sorted(s for s in held if block[s] in tblocks)
    scored = np.array([s not in twins for s in sid_c])
    S_idx = np.where(scored)[0]
    say('[scored set] %d of %d held-out structures have their parent molecule among the %d training structures and '
        'are not scored: %s' % (len(twins), len(held), n_t, '; '.join('%s (%s)' % (s, ', '.join(sid2names[s])) for s in twins)))
    say('[scored set] %d conditions of %d structures scored; candidate sets unchanged (Step 15c)'
        % (len(S_idx), len(held) - len(twins)))
    if not np.isfinite(pds_recipe[S_idx]).all():
        die('a scored condition has no PDS (fewer than %d candidates)' % s15c.MIN_CAND)

    # ---------------------------------------------------------------- similarities (prereg 3)
    hsids = held
    hpos = {s: j for j, s in enumerate(hsids)}
    hrow = [frow[s] for s in hsids]
    cols = {'desc': list(man['descriptors']), 'ab': list(man['atom_bond_columns']),
            'pc': list(man['pharmacophore_count_columns'])}
    cols['num'] = cols['desc'] + cols['ab'] + cols['pc']
    M = st[cols['num']].apply(pd.to_numeric, errors='coerce').to_numpy(np.float64)
    med = np.nanmedian(M[trow], axis=0)
    if np.isnan(med).any():
        die('a numeric column has no value among the training structures')
    n_imputed = int(np.isnan(M).sum())
    M = np.where(np.isnan(M), med[None, :], M)                 # missing -> training median
    mu, sd = M[trow].mean(0), M[trow].std(0)
    Z = np.where(sd[None, :] > 0, (M - mu[None, :]) / np.where(sd > 0, sd, 1.0)[None, :], 0.0)
    cidx = {k: [cols['num'].index(x) for x in v] for k, v in cols.items()}
    SIM = {'fa': tanimoto(FA[hrow], FA[trow]), 'fp': tanimoto(FPp[hrow], FPp[trow]),
           'ep': tanimoto(EP[hrow], EP[trow]), 'ph': tanimoto(PH[hrow], PH[trow]),
           'phf': tanimoto(PHF[hrow], PHF[trow])}
    for k in ('num', 'desc', 'ab', 'pc'):
        SIM[k] = -cdist(Z[hrow][:, cidx[k]], Z[trow][:, cidx[k]])
    say('[features] %d training structures as neighbours; numbers: %d columns (%d constant among training '
        'structures), %d missing values set to the training median' % (n_t, len(cols['num']), int((sd == 0).sum()), n_imputed))

    # X1: response profile in the other lines, shared context response removed (reference, never a predictor)
    HDs = np.zeros((len(hsids), len(ctx_list), G))
    HDn = np.zeros((len(hsids), len(ctx_list)))
    for i in range(n):
        HDs[hpos[sid_c[i]], ctx_code[i]] += Y[i]
        HDn[hpos[sid_c[i]], ctx_code[i]] += 1
    with np.errstate(invalid='ignore'):
        HD = HDs / HDn[:, :, None]
    Mc = np.nanmean(TDm, axis=1)                               # mean training response per context
    HDc, TDc = HD - Mc[None], TDm - Mc[:, None]
    ctx_line = np.array([c.split('|')[0] for c in ctx_list])
    SIMR = {}
    for L in lv:
        cm = ctx_line != L                                     # never the line being predicted
        T = TDc[cm].transpose(1, 0, 2).reshape(n_t, -1)
        vt = np.isfinite(T)
        Tz = np.where(vt, T, 0.0)                              # missing entries contribute nothing
        for h in range(len(hsids)):
            x = HDc[h, cm].reshape(-1)
            vx = np.isfinite(x)
            x0 = np.where(vx, x, 0.0)
            num = Tz @ x0                                      # cosine over the entries present on both sides
            dx = vt.astype(np.float64) @ (x0 * x0)
            dt = (Tz * Tz) @ vx.astype(np.float64)
            den = np.sqrt(dx * dt)
            SIMR[(h, L)] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)

    # ---------------------------------------------------------------- transfer (prereg 4.1)
    def order_for(blocks, h, L):
        sims = [SIMR[(h, L)] if b == 'resp' else SIM[b][h] for b in blocks]
        return order_single(sims[0]) if len(sims) == 1 else order_pair(sims[0], sims[1])

    preds, nbrs = {}, {}
    for code, label, blocks in REPS:
        cache = {}
        for k in (K_MAIN, K_SEC):
            P = np.zeros((n, G))
            nb = []
            for i in range(n):
                h = hpos[sid_c[i]]
                key = (h, lines[i]) if 'resp' in blocks else h
                if key not in cache:
                    cache[key] = order_for(blocks, h, lines[i])
                o = cache[key]
                el = o[avail[ctx_code[i]][o]][:k]
                if len(el) < k:
                    die('fewer than %d eligible neighbours' % k)
                P[i] = TDm[ctx_code[i], el].mean(0)
                nb.append(el)
            preds[(code, k)] = P
            nbrs[(code, k)] = nb

    # ---------------------------------------------------------------- scores (prereg 4.2-4.5)
    pds = {key: s15c.pds_sets(P, Y, contexts, cand) for key, P in preds.items()}
    norms = {key: np.linalg.norm(P, axis=1) for key, P in preds.items()}
    b_S = Boot(S_idx, drugs, n)
    res = {'pds': {}, 'mra': {}, 'de_overlap': {}, 'dpc50': {}}
    for (code, k), P in preds.items():
        j, d = s15.thesis_per_cond(true_mean, C + P, C)
        res['de_overlap']['%s_k%d' % (code, k)] = float(np.nanmean(j[S_idx]))
        res['dpc50']['%s_k%d' % (code, k)] = float(np.nanmean(d[S_idx]))
        res['pds']['%s_k%d' % (code, k)] = float(np.mean(pds[(code, k)][S_idx]))
    rec_norm_seeds = [np.linalg.norm(v['B'] - v['C'], axis=1) for v in arms['recipe']]
    keys_k = {k: [(code, k) for code, _, _ in REPS] for k in (K_MAIN, K_SEC)}
    mra_pt = {}
    for k in (K_MAIN, K_SEC):
        for key in keys_k[k]:
            mra_pt[key] = mra_point(norms[key], tn, ctx_code, S_idx)
            res['mra']['%s_k%d' % key] = mra_pt[key]
    recipe_S = {'pds': float(np.mean(pds_recipe[S_idx])),
                'mra': float(np.mean([mra_point(x, tn, ctx_code, S_idx) for x in rec_norm_seeds]))}
    twin_idx = np.where(~scored)[0]
    twin_R1 = float(np.mean(pds[('R1', K_MAIN)][twin_idx])) if len(twin_idx) else None
    twin_recipe = float(np.mean(pds_recipe[twin_idx])) if len(twin_idx) else None

    # ---------------------------------------------------------------- intervals and contrasts (prereg 4.6, 5)
    con = {}

    def pds_contrast(name, a, b, k):
        xa = pds[(a, k)]
        xb = pds[(b, k)] if b else np.full(n, 0.5)
        pt_, reps_ = b_S.mean_reps(xa - xb)
        con[name] = interval(pt_, reps_)
        return con[name]

    say('\n' + '=' * 100 + '\nSTAGE 1 SCREEN, %s, %d scored conditions, %d structures; PDS (chance 0.5); 97.5 %% '
        'structure-clustered intervals\n' % (SPLIT, len(S_idx), b_S.G) + '=' * 100)
    for code, label, _ in REPS:
        say('  %-4s %-58s PDS k=10 %.4f  k=5 %.4f | MRA k=10 %+.4f | DE-overlap %.4f | DPC-50 %.4f'
            % (code, label, res['pds']['%s_k10' % code], res['pds']['%s_k5' % code], res['mra']['%s_k10' % code],
               res['de_overlap']['%s_k10' % code], res['dpc50']['%s_k10' % code]))
    say('  recipe on the scored set: PDS %.4f, MRA %+.4f (seed mean)' % (recipe_S['pds'], recipe_S['mra']))
    if twin_R1 is not None:
        say('  not scored (twins, %d conditions): R1 PDS %.4f, recipe PDS %.4f (descriptive)' % (len(twin_idx), twin_R1, twin_recipe))
    say('\nGATED (k = 10, PDS):')
    for name, a, b in GATED:
        c_ = pds_contrast('gated_%s' % name, a, b, K_MAIN)
        say('  %-4s %-10s %s' % (name, '%s - %s' % (a, b or '0.5'), fmt(c_)))
    say('\nsecondary (k = 5, PDS):')
    for name, a, b in GATED:
        say('  %-4s %-10s %s' % (name, '%s - %s' % (a, b or '0.5'), fmt(pds_contrast('k5_%s' % name, a, b, K_SEC))))
    say('\nexploratory and references (k = 10, PDS):')
    for code in ('E1', 'E2', 'E3', 'E3a', 'E3b', 'E3c', 'E4', 'X2'):
        say('  %-4s - R2      %s' % (code, fmt(pds_contrast('pds_%s_minus_R2' % code, code, 'R2', K_MAIN))))
    say('  X1   - 0.5     %s' % fmt(pds_contrast('pds_X1_minus_0.5', 'X1', None, K_MAIN)))
    say('  E4   - R4      %s   (unfolded minus folded)' % fmt(pds_contrast('pds_E4_minus_R4', 'E4', 'R4', K_MAIN)))

    # MRA (secondary): each gated representation against 0, and the gated differences, recomputed per replicate
    mcodes = ['R1', 'R2', 'R3', 'R4', 'R5']
    mpt, mreps = b_S.mra_reps([norms[(c_, K_MAIN)] for c_ in mcodes] + [rec_norm_seeds], tn, ctx_code)
    say('\nsecondary (k = 10, MRA, recomputed per replicate):')
    for j, c_ in enumerate(mcodes):
        con['mra_%s' % c_] = interval(mpt[j], mreps[:, j])
        say('  %-4s - 0       %s' % (c_, fmt(con['mra_%s' % c_])))
    for name, a, b in GATED[1:]:
        ja, jb = mcodes.index(a), mcodes.index(b)
        con['mra_%s' % name] = interval(mpt[ja] - mpt[jb], mreps[:, ja] - mreps[:, jb])
        say('  %-4s %-10s %s' % (name, '%s - %s' % (a, b), fmt(con['mra_%s' % name])))
    con['mra_recipe'] = interval(mpt[-1], mreps[:, -1])
    say('  recipe - 0     %s' % fmt(con['mra_recipe']))

    # Qp: potency against measured response size (prereg 4.4). Within a (line, dose) context the dose is fixed, so
    # ranking by pChEMBL equals ranking by the occupancy dose / (dose + IC50) that Stage 2 would give the model.
    pot = {}
    say('\nQp: potency against measured response size, on the scored conditions of structures with a value. '
        '(a) MRA; (b) partial Spearman beyond the recipe\'s predicted size - gated for the protein-target value:')
    for col in POTENCY:
        vals = pd.to_numeric(st[col], errors='coerce').to_numpy(np.float64)
        pv = np.array([vals[frow[s]] for s in sid_c])
        q_idx = S_idx[np.isfinite(pv[S_idx])]
        bq = Boot(q_idx, drugs, n)
        qpt, qreps = bq.mra_reps([np.nan_to_num(pv), rec_norm_seeds], tn, ctx_code)
        ppt, preps = bq.partial_reps(np.nan_to_num(pv), tn, rec_size, ctx_code)
        pot[col] = dict(gated=(col == POTENCY_GATED), n_structures=int(bq.G), n_conditions=int(len(q_idx)),
                        mra=interval(qpt[0], qreps[:, 0]), recipe_mra_same_conditions=interval(qpt[1], qreps[:, 1]),
                        partial_beyond_recipe=interval(ppt, preps))
        say('  %-24s %2d structures, %4d conditions%s\n      (a) MRA %s | recipe on the same conditions %+.4f\n'
            '      (b) partial %s'
            % (col, bq.G, len(q_idx), '  [GATED, Qp]' if col == POTENCY_GATED else '', fmt(pot[col]['mra']), qpt[1],
               fmt(pot[col]['partial_beyond_recipe'])))

    # ---------------------------------------------------------------- selection rule (prereg 5)
    v = {name: con['gated_%s' % name]['verdict'] for name, _, _ in GATED}
    blocks_pass = [(name, con['gated_%s' % name]['diff']) for name in ('Q1b', 'Q1c', 'Q1d') if v[name] == 'above zero']
    what = {'Q1b': 'R3 (ECFP4 parent)', 'Q1c': 'R4 (2D pharmacophores, 1,024 bits)', 'Q1d': 'R5 (49 numbers)'}
    if blocks_pass:
        best = max(blocks_pass, key=lambda t: t[1])[0]
        decision = dict(rule='5.1', stage2=True, selected=what[best], arms=['chem', 'chem_shuf'])
        text = 'Stage 2 runs with %s: arms chem and chem_shuf (prereg 6).' % what[best]
    elif v['Q1a'] == 'above zero':
        decision = dict(rule='5.2', stage2=True, selected='R2 (FCFP4 of the parent)', arms=['chem'])
        text = 'Stage 2 runs with the parent fingerprint alone: arm chem, no shuffled control (prereg 6).'
    else:
        decision = dict(rule='5.3', stage2=False, selected=None, arms=[],
                        reading='no added information detected by a probe that detects the fingerprint\'s own'
                        if v['P'] == 'above zero' else 'screen uninformative: the probe does not read the fingerprint\'s information')
        text = 'The chemistry arms are not run. Reading: %s.' % decision['reading']
    qp_v = pot[POTENCY_GATED]['partial_beyond_recipe']['verdict']
    pdec = dict(rule='5.4', gated='Qp: partial Spearman of %s beyond the recipe\'s predicted size' % POTENCY_GATED,
                verdict=qp_v, stage2=(qp_v == 'above zero'), arms=(['pot', 'pot_shuf'] if qp_v == 'above zero' else []))
    ptext = ('Stage 2 also runs the potency arms pot and pot_shuf (prereg 6.2).' if pdec['stage2']
             else 'The potency arms are not run.')
    say('\n' + '=' * 100 + '\nPRE-REGISTERED SELECTION (section 5)\n'
        '  chemistry route, rule %s: P %s | Q1a %s | Q1b %s | Q1c %s | Q1d %s\n    %s\n'
        '  potency route, rule 5.4: Qp %s\n    %s\n'
        % (decision['rule'], v['P'], v['Q1a'], v['Q1b'], v['Q1c'], v['Q1d'], text, qp_v, ptext)
        + '  Section 7 expected: rule 5.3 with P above zero (no chemistry arms); Qp includes zero (no potency arms).\n'
        + '=' * 100)

    # ---------------------------------------------------------------- records (built in memory, then written together)
    rows = []
    for (code, k) in preds:
        for metric in ('pds', 'mra', 'de_overlap', 'dpc50'):
            rows.append(dict(kind='value', name=code, k=k, metric=metric, value=res[metric]['%s_k%d' % (code, k)],
                             lo='', hi='', verdict=''))
    for name, c_ in con.items():
        rows.append(dict(kind='interval', name=name, k='', metric='pds' if not name.startswith('mra') else 'mra',
                         value=c_['diff'], lo=c_['ci'][0], hi=c_['ci'][1], verdict=c_['verdict']))
    for col, d_ in pot.items():
        for part, metric in (('mra', 'mra'), ('recipe_mra_same_conditions', 'mra'), ('partial_beyond_recipe', 'partial')):
            rows.append(dict(kind='interval', name='Qp_%s_%s' % (col, part), k='', metric=metric, value=d_[part]['diff'],
                             lo=d_[part]['ci'][0], hi=d_[part]['ci'][1], verdict=d_[part]['verdict']))
    pcols = {col: pd.to_numeric(st[col], errors='coerce').to_numpy(np.float64) for col in POTENCY}
    fields = ['split', 'condition', 'line', 'plate', 'structure', 'structure_id', 'dose_nM', 'scored', 'n_candidates',
              'measured_size', 'recipe_pds', 'ridge_train_pds'] + POTENCY + \
             ['%s_k%d_%s' % (code, k, m) for code, _, _ in REPS for k in (K_MAIN, K_SEC) for m in ('pds', 'size')] + \
             ['%s_k10_neighbours' % code for code in ('R1', 'R2', 'R3', 'R4', 'R5')]
    pc_rows = []
    for i in range(n):
        r = dict(split=SPLIT, condition=cond[i], line=lines[i], plate=plates[i], structure=drugs[i],
                 structure_id=sid_c[i], dose_nM=doses[i], scored=int(scored[i]), n_candidates=int(n_cand[i]),
                 measured_size=tn[i], recipe_pds=pds_recipe[i], ridge_train_pds=pds_ridge_train[i])
        for col in POTENCY:
            r[col] = pcols[col][frow[sid_c[i]]]
        for code, _, _ in REPS:
            for k in (K_MAIN, K_SEC):
                r['%s_k%d_pds' % (code, k)] = pds[(code, k)][i]
                r['%s_k%d_size' % (code, k)] = norms[(code, k)][i]
        for code in ('R1', 'R2', 'R3', 'R4', 'R5'):
            r['%s_k10_neighbours' % code] = ' '.join(tsids[j] for j in nbrs[(code, K_MAIN)][i])
        pc_rows.append(r)
    json_text = json.dumps(dict(
        prereg=PRE, split=SPLIT, commit=commit, feature_commit=fcommit, k=[K_MAIN, K_SEC], n_boot=N_BOOT,
        interval='97.5% percentile, structure-clustered, fresh RandomState(0) per scored set',
        reference=ref, test_reference_override=TEST_REFERENCE,
        reproduction=dict(mra=mra_chk, bootstrap_q1=q1),
        design=dict(n_conditions=int(n), n_structures=len(held), n_training_structures=int(n_t),
                    twins=[dict(structure_id=s, names=sid2names[s], parent_block=block[s]) for s in twins],
                    n_scored_conditions=int(len(S_idx)), n_scored_structures=int(b_S.G),
                    candidates_per_condition=[int(n_cand.min()), int(n_cand.max())],
                    numeric_columns=len(cols['num']), numeric_constant=int((sd == 0).sum()), numeric_imputed=n_imputed),
        representations={c_: dict(label=l_, blocks=b_) for c_, l_, b_ in REPS},
        values=res, recipe_on_scored_set=recipe_S, twins_descriptive=dict(R1_pds=twin_R1, recipe_pds=twin_recipe),
        contrasts=con, qp_potency=pot, selection=dict(chemistry=decision, potency=pdec),
        inputs={'vectors': {a: [v['rec'] for v in L] for a, L in arms.items()},
                'feature_manifest_sha256': s15.sha(os.path.join(FEAT, 'manifest.json')),
                'fingerprints_sha256': s15.sha(os.path.join(FEAT, 'fingerprints.npz')),
                'structures_sha256': s15.sha(os.path.join(FEAT, 'structures.csv')),
                'step15c_per_condition_sha256': s15.sha(PC15C)},
        seconds=int(round(time.time() - T0))), indent=2, allow_nan=True)
    with open(OUT_SUM, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['kind', 'name', 'k', 'metric', 'value', 'lo', 'hi', 'verdict'], lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    with open(OUT_PC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(pc_rows)
    with open(OUT_JSON, 'w') as f:
        f.write(json_text)
    say('\nwrote %s, %s and %s  (%.0f s)' % (OUT_JSON, OUT_PC, OUT_SUM, time.time() - T0))

if __name__ == '__main__':
    main()
