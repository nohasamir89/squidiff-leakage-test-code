"""Cross-checks of every statistical primitive in the scorers against an
independent reference.

Design (and the sentence to cite in Methods): each hand-written statistical
routine is compared with (a) a reference implementation from scipy or
scikit-learn, or (b) an analytical value, on inputs that include the awkward
cases real data produce (ties, zeros, constant vectors, repeated rows as a
bootstrap draw makes). A test fails if the difference exceeds floating-point
noise (1e-12 unless stated), so these tests detect wrong formulas, not only
crashes.

Why the repo's own functions are imported rather than re-typed: a test that
re-implements the formula tests the re-implementation. These tests import
``step14_metrics``, ``score_step15_tahoe``, ``score_step16_tahoe``,
``score_step19_heterogeneity`` and ``diag_step19_control_route`` exactly as
the scorers use them (see conftest.py for how that works without torch).

Every number below is deterministic: ``rng`` is a fixed-seed generator.
"""
import ast

import numpy as np
import pytest
from scipy import stats
from scipy.spatial.distance import cdist, pdist   # noqa: F401  (kept for readers tracing the energy formula)
from sklearn.metrics import r2_score

import diag_step19_control_route as d19
import score_step15_tahoe as s15
import score_step16_tahoe as s16
import score_step19_heterogeneity as s19
import step14_metrics as vm

TOL = 1e-12


# =========================================================================== differential expression
def test_mannwhitney_matches_scipy_on_sparse_tied_data(rng):
    """Per-gene two-sided Wilcoxon rank-sum (Mann-Whitney U) with tie and
    continuity corrections must equal scipy's asymptotic method.

    Poisson(0.4)/Poisson(0.6) counts give the heavy ties (many exact zeros)
    that log1p single-cell expression has, which is where a hand-written tie
    correction most often goes wrong."""
    X = rng.poisson(0.4, size=(40, 200)).astype(float)
    Y = rng.poisson(0.6, size=(60, 200)).astype(float)
    p_repo = vm._mannwhitney_p(X, Y)
    p_ref = stats.mannwhitneyu(X, Y, alternative="two-sided", method="asymptotic", axis=0).pvalue
    assert np.max(np.abs(p_repo - p_ref)) < TOL


def test_mannwhitney_matches_scipy_on_continuous_data(rng):
    """Same comparison without ties (the tie-correction term must vanish cleanly)."""
    X = rng.normal(size=(25, 150))
    Y = rng.normal(0.3, 1.0, size=(35, 150))
    p_repo = vm._mannwhitney_p(X, Y)
    p_ref = stats.mannwhitneyu(X, Y, alternative="two-sided", method="asymptotic", axis=0).pvalue
    assert np.max(np.abs(p_repo - p_ref)) < TOL


def test_mannwhitney_constant_gene_is_not_significant():
    """A gene that is zero in every cell has no evidence either way: p must be exactly 1,
    never NaN (a NaN would silently drop the gene from the BH correction)."""
    p = vm._mannwhitney_p(np.zeros((30, 5)), np.zeros((20, 5)))
    assert np.all(p == 1.0)


def test_benjamini_hochberg_matches_scipy(rng):
    """BH adjusted p-values (step-up, monotone) must equal scipy.stats.false_discovery_control.
    Half the cases are rounded to create tied p-values, the usual trap for cummin-based code."""
    worst = 0.0
    for _ in range(50):
        p = rng.uniform(size=int(rng.integers(5, 400)))
        if rng.uniform() < 0.5:
            p = np.round(p, 2)
        ref = stats.false_discovery_control(p, method="bh")
        worst = max(worst, float(np.max(np.abs(vm._bh(p) - ref))))
    assert worst < TOL


# =========================================================================== gene identity (Jaccard)
def test_expected_random_jaccard_is_0144():
    """The dashed 'chance' line in Figs 1b/2a: two random sets of 50 genes out of 200.
    E[I/(100-I)] with I ~ Hypergeometric(N=200, K=50, n=50) = 0.143918... The manuscript
    writes 0.144; the committed fig1b_values.csv stores the exact value."""
    hg = stats.hypergeom(200, 50, 50)
    i = np.arange(0, 51)
    expect = float(np.sum(hg.pmf(i) * (i / (100.0 - i))))
    assert abs(expect - 0.143918) < 1e-6
    assert round(expect, 3) == 0.144


def test_jaccard_known_overlap():
    """25 shared genes between two 50-gene sets -> 25 / (50 + 50 - 25) = 1/3."""
    assert abs(s15.jac(np.arange(50), np.arange(25, 75)) - 1 / 3) < 1e-15


def test_thesis_per_cond_selects_true_top50_and_uses_pearson():
    """Gene identity and response correlation are computed on the 50 genes with the largest
    measured |change|. Constructed case: true change non-zero on genes 0-49, predicted change
    non-zero on genes 25-74 -> Jaccard 1/3; DPC must equal scipy's Pearson r on genes 0-49."""
    G = 200
    true_d = np.zeros(G); true_d[:50] = np.linspace(2, 3, 50)
    pred_d = np.zeros(G); pred_d[25:75] = np.linspace(2, 3, 50)
    C = np.zeros((1, G))                                       # paired-control mean: zero -> deltas are the vectors themselves
    j, d = s15.thesis_per_cond(true_d[None, :], pred_d[None, :], C)
    g = np.argsort(np.abs(true_d))[-50:]
    assert abs(j[0] - 1 / 3) < 1e-15
    assert abs(d[0] - stats.pearsonr(pred_d[g], true_d[g])[0]) < 1e-15


# =========================================================================== population distance (energy)
def _energy_loops(X, Y):
    """Independent O(n^2) implementation of the Szekely-Rizzo energy distance with
    U-statistic within-sample terms (self-pairs excluded), as the Methods define it."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, m = len(X), len(Y)
    xy = sum(np.linalg.norm(X[i] - Y[j]) for i in range(n) for j in range(m)) / (n * m)
    xx = sum(np.linalg.norm(X[i] - X[j]) for i in range(n) for j in range(i + 1, n)) / (n * (n - 1) / 2)
    yy = sum(np.linalg.norm(Y[i] - Y[j]) for i in range(m) for j in range(i + 1, m)) / (m * (m - 1) / 2)
    return 2 * xy - xx - yy


def test_energy_matches_independent_implementation(rng):
    X = rng.normal(size=(25, 8)); Y = rng.normal(0.5, 1.3, size=(30, 8))
    assert abs(s15.energy(X, Y) - _energy_loops(X, Y)) < 1e-10


def test_energy_matches_analytic_value_for_shifted_normals(rng):
    """Known-answer test. For X ~ N(0,1) and Y ~ N(delta,1) in one dimension,
    E|X-Y| = sqrt(2) * ( sqrt(2/pi) exp(-delta^2/4) + (delta/sqrt 2)(2 Phi(delta/sqrt 2) - 1) )
    and E|X-X'| = E|Y-Y'| = 2/sqrt(pi), so energy = 2 E|X-Y| - 4/sqrt(pi).
    With n = 4000 per sample the Monte-Carlo error is about 0.01; tolerance 0.03."""
    delta = 1.0
    s2 = np.sqrt(2.0)
    exy = s2 * (np.sqrt(2 / np.pi) * np.exp(-delta ** 2 / 4) + (delta / s2) * (2 * stats.norm.cdf(delta / s2) - 1))
    analytic = 2 * exy - 4 / np.sqrt(np.pi)
    X = rng.normal(size=(4000, 1)); Y = rng.normal(delta, 1.0, size=(4000, 1))
    assert abs(s15.energy(X, Y) - analytic) < 0.03


# =========================================================================== compound discrimination (PDS)
def _pds_naive(P, T):
    """Independent PDS: cosine-distance ranks via scipy.stats.rankdata (ties averaged),
    PDS = 1 - k/(n-1) with k = number of candidates ranked at or above the true match."""
    n = len(P)
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            a, b = np.linalg.norm(P[i]), np.linalg.norm(T[j])
            D[i, j] = 1.0 if (a == 0 or b == 0) else 1.0 - float(P[i] @ T[j]) / (a * b)
    return np.array([1.0 - (stats.rankdata(D[i], method="average")[i] - 1.0) / (n - 1) for i in range(n)])


def test_pds_matches_independent_implementation(rng):
    A = rng.normal(size=(30, 40)); B = A + rng.normal(scale=0.05, size=A.shape)
    assert np.max(np.abs(vm.pds_cosine(B, A) - _pds_naive(B, A))) < TOL


def test_pds_perfect_prediction_scores_one(rng):
    A = rng.normal(size=(30, 40))
    assert np.all(vm.pds_cosine(A, A) == 1.0)


def test_pds_chance_level_is_half(rng):
    """Unrelated predictions must average 0.5 (the chance line in Fig 5a)."""
    sims = [float(np.mean(vm.pds_cosine(rng.normal(size=(60, 30)), rng.normal(size=(60, 30))))) for _ in range(40)]
    assert abs(np.mean(sims) - 0.5) < 0.01


# =========================================================================== response-magnitude agreement (MRA)
def test_fast_rank_matches_scipy_rankdata(rng):
    """fast_rank is the vectorised rank used inside 5,000 bootstrap replicates; it must give
    scipy's average ranks on ties and on repeated entries (as a bootstrap draw produces)."""
    worst = 0.0
    for _ in range(200):
        x = rng.choice(rng.normal(size=10), size=int(rng.integers(3, 80)))
        worst = max(worst, float(np.max(np.abs(s16.fast_rank(x) - stats.rankdata(x)))))
    assert worst < TOL


def test_mra_matches_scipy_spearman_per_context(rng):
    """MRA = mean over contexts (cell line x dose) of the Spearman correlation between predicted
    and measured response norms; contexts below MIN_CTX conditions are dropped and a constant
    prediction scores 0 (the documented convention that puts the compound-blind average at 0)."""
    worst = 0.0
    for _ in range(50):
        n, k = 120, 5
        ctx = rng.integers(0, k, n)
        t = rng.normal(size=n); p = rng.normal(size=n)
        got = s16.mra_pre(p, s16.prep_ctx(t, s16.ctx_split(ctx, k, np.arange(n))))
        rhos = []
        for c in range(k):
            m = np.where(ctx == c)[0]
            if len(m) >= s16.MIN_CTX and np.ptp(t[m]) > 0:
                rhos.append(0.0 if np.ptp(p[m]) == 0 else stats.spearmanr(p[m], t[m])[0])
        worst = max(worst, abs(got - float(np.mean(rhos))))
    assert worst < TOL


def test_repo_mra_self_check_runs():
    """score_step16_tahoe ships its own self-check (fast_rank vs rankdata, MRA vs the Step 15e
    routine). It calls sys.exit on failure, so a clean return is the assertion."""
    s16.mra_self_check()


# =========================================================================== spread agreement and its decomposition
def test_spread_hand_case():
    """Spread = mean Euclidean distance of cells to their centroid: two points at distance 2 -> 1.0."""
    assert abs(s19.spread(np.array([[0.0, 0.0], [2.0, 0.0]])) - 1.0) < 1e-15


def test_partial_spearman_matches_closed_formula(rng):
    """The exploratory decomposition of spread agreement removes the shared vehicle-spread
    denominator by a partial Spearman correlation (correlation of rank residuals). It must equal
    the textbook formula (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2)) on the ranks."""
    worst = 0.0
    for _ in range(100):
        n = int(rng.integers(12, 60))
        z = rng.normal(size=n); x = 0.6 * z + rng.normal(size=n); y = 0.5 * z + rng.normal(size=n)
        got = d19.partial(d19.crank(x), d19.crank(y), d19.crank(z))
        rxy, rxz, ryz = stats.spearmanr(x, y)[0], stats.spearmanr(x, z)[0], stats.spearmanr(y, z)[0]
        ref = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        worst = max(worst, abs(got - ref))
    assert worst < 1e-9


# =========================================================================== eta-squared (compound-identity localisation)
def _eta2_from_script(layout):
    """eta_latent_vs_output.py runs at import (it is a script), so its eta2() is extracted from
    the source with ast and compiled on its own. Nothing else from the script is executed."""
    src = (layout.scoring / "eta_latent_vs_output.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "eta2")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<eta2>", "exec"), ns)
    return ns["eta2"]


def test_eta2_matches_one_way_anova_identity(rng, layout):
    """eta^2 = SS_between / SS_total per dimension; the variance-weighted version is
    sum(SS_between) / sum(SS_total). Checked against the sums of squares computed directly."""
    eta2 = _eta2_from_script(layout)
    X = rng.normal(size=(120, 7)); lab = rng.integers(0, 6, 120).astype(str)
    per, vw = eta2(X, lab)
    ref = []
    for g in range(X.shape[1]):
        v = X[:, g]; gm = v.mean()
        ssb = sum((v[lab == u].mean() - gm) ** 2 * (lab == u).sum() for u in np.unique(lab))
        ref.append(ssb / ((v - gm) ** 2).sum())
    assert np.max(np.abs(per - np.array(ref))) < TOL
    assert 0.0 <= vw <= 1.0


# =========================================================================== harness r2
def test_harness_r2_formula_matches_sklearn(rng):
    """leakage_test_base_*.py and eval_gbm_sensitivity.py define r2 locally as
    1 - SS_res/SS_tot with a = measured, b = predicted; that is sklearn's r2_score(y_true=a, y_pred=b)."""
    def harness_r2(a, b):
        ss_res = np.sum((a - b) ** 2); ss_tot = np.sum((a - a.mean()) ** 2)
        return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    a = rng.normal(size=300); b = a + rng.normal(scale=0.5, size=300)
    assert abs(harness_r2(a, b) - r2_score(a, b)) < TOL


# =========================================================================== bootstrap mechanics
def test_cluster_bootstrap_percentiles_are_975_percent():
    """The pre-registered interval is 97.5% percentile per split: LO/HI must be 1.25 and 98.75."""
    assert (s15.LO, s15.HI) == (1.25, 98.75)
    assert s15.N_BOOT == 5000


def test_verdict_rule():
    """'above zero' iff the whole interval is above 0, 'below zero' iff whole interval below, else includes."""
    assert s15.verdict(0.01, 0.05) == "above zero"
    assert s15.verdict(-0.05, -0.01) == "below zero"
    assert s15.verdict(-0.01, 0.01) == "includes zero"
    assert s15.verdict(0.0, 0.01) == "includes zero"      # a bound exactly at zero does not exclude it


@pytest.mark.slow
def test_cluster_bootstrap_coverage_simulation(rng):
    """Optional, slower (~1 min): with compound-clustered data and a known mean difference, the
    97.5% structure-clustered percentile interval should contain the truth in roughly 97.5% of
    simulated datasets. 200 simulations give a standard error of about 1.1 points, so the check
    is loose (>= 93%); it guards against gross errors such as resampling conditions instead of
    compounds. Run with: pytest -m slow"""
    truth, n_drugs, per_drug = 0.05, 60, 12
    hits = 0
    n_sim = 200
    for _ in range(n_sim):
        drug_eff = rng.normal(truth, 0.05, size=n_drugs)                        # between-compound variation
        x = np.repeat(drug_eff, per_drug) + rng.normal(0, 0.05, size=n_drugs * per_drug)
        drugs = np.repeat(np.arange(n_drugs), per_drug)
        s15.rng = np.random.RandomState(int(rng.integers(0, 2**31 - 1)))          # the scorer's module-level generator
        lo, hi = s15.boot(x, drugs, n_drugs)
        hits += (lo <= truth <= hi)
    assert hits / n_sim >= 0.93
