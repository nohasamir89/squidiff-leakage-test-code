# Step 14, Amendment 2: the six VCC-style facets recomputed against the cell-eval2 source (facets v2)

**Written after the results of the first run were seen** (2026-09-23; first record `leakage_results/step14_analysis.json`,
commit c354314; findings `step14_findings.md`, commit 37a660b). It is therefore kept as a separate document and
`PREREG_step14_shift_baselines_vcc_scoring.md` is not edited. It changes nothing that Q1 gates: the primary and
co-primary contrasts, their verdict (PARTIAL), the secondary contrasts and the thesis's three metrics stand as
recorded. It changes only how the descriptive facets of Q2 are computed, and it records why.

Author: Noha Samir Ismail.

## 1. What prompted it

PREREG §4 states that the six facets follow `cell-eval2` 0.15.0 (`vcc2026-metrics-brief.md`, 2026-08-19) with the
listed adaptations. An audit of the first run against the cell-eval2 source (the 0.16.0 code imported from
ArcInstitute/cell_eval2, files `de_compute.py`, `lfc_nmae_ref.py`, `anchor.py`, `baseline.py`, `score.py`,
`scoring.py`, `metrics/direction.py`, `configs/vcc2026.yaml`) found that the six facet formulas were implemented as
specified but that five points of the surrounding procedure were not the challenge's, and none of the five was
forced by the data. They are listed in §2 with the correction each receives. The adaptations of PREREG §4 and
Amendment 1 that are forced by the data (200 highly variable genes of stored log1p values instead of 18,533 genes
of raw counts; profiles as means of stored values instead of count pseudobulk; no jackknife sampling correction and
no ρ cap on the expression error, both of which need raw counts and per-cell totals; no 5-CPM gate; no target-gene
exclusion; the paired-control origin; the ≥ 20-cell gate) are unchanged.

## 2. Corrections

| # | item | cell-eval2 (source) | first run | facets v2 |
|---|---|---|---|---|
| 1 | Wilcoxon p-value | scanpy `rank_genes_groups(method="wilcoxon", tie_correct=True)`: tie-corrected normal approximation, no continuity correction (`de_compute.py`) | tie correction plus scipy's continuity correction | continuity correction removed (`step14_metrics_v2._mannwhitney_p`); reproduces scanpy's p-values to 1e-15 on synthetic data with ties |
| 2 | fold-change NMAE replicate anchor | gate and denominator from the full reference table, only the two fold-change vectors from the halves (`lfc_nmae_ref.py`, brief §7) | gate and denominator from the half table | as the source (`step14_metrics_v2.lfc_nmae_anchor`) |
| 3 | degenerate scale | a member whose replicate does not lie on the better side of the baseline is refused; no scaled value is produced (`score.py`, `scoring.denominator`) | scored, giving inverted values | refused: the cell is undefined and carries the reason; a line without all six scaled facets has no overall |
| 4 | the 0 point | the mean response emitted as cells: control cells resampled with replacement to the group's size and scaled gene-wise, in count space, by the mean-response profile over the control profile (`baseline.py`, "dispersed") | each paired vehicle cell plus the mean delta in log space, clipped at 0 | the dispersed emission on the condition's paired vehicle cells, in normalised-count space (`expm1` of the stored values), scaled by the line's mean-response profile over the line's control profile, both equal-weight means over the line's conditions; the additive shift of the first run is kept as an ordinary submission (`mean_response`) because the secondary contrasts were pre-registered on it |
| 5 | aggregation | a context's score is the mean of its six scaled facets; the overall is the mean over contexts (brief §0) | mean over lines per facet, then over facets | per line first, then over lines; per-facet split values remain means over the lines with a defined value |
| 6 | dtype | — | the shift baselines' cells were float64 copies of float32 vehicle cells, so control-paste's fold change was rounding noise with a random sign (reach 0.26–0.44 instead of 0) | cells cast to float32 before the DE table |

Nothing else changes: inputs and their hashes, the matching of the harness draw, the ≥ 20-cell gate, the Step 7
reproduction guard, the replicate half-splits (five seeds, halves of floor(n/2), each half against its own vehicle
half), the four raw facet definitions, the profile-level facets (PDS, expression-error ratio), the clamps (MSE to
[0, 1], NMAE floored at −6, the rest unclamped), and the facet contrasts (ridge-S7 minus unguided 50 NFE, 97.5%
drug-clustered percentile bootstrap, 5000 replicates, `RandomState(0)`).

## 3. What the recomputation can and cannot change

- The PDS and expression-error ratio of every submission contain no test statistic and must equal the first record
  to 1e-9; the script stops otherwise.
- The four DE facets of every submission change only through correction 1 (a few borderline significance calls),
  and the control-paste reach through correction 6. The change per facet and submission is printed and recorded.
- The replicate anchor of the NMAE changes through correction 2, and is now defined wherever the metric is defined.
- The 0 point of every facet changes through correction 4. The scaled values and the overall therefore change for
  every submission, and some cells become undefined through correction 3.
- Expectations are not re-registered: PREREG §7's expectations for the facets were audited in `step14_findings.md`
  §7 against the first run and are not re-scored against v2.

## 4. Analysis plan

`score_step14_facets_v2.py` with `step14_metrics_v2.py`, committed before the run, reading the first record's
inputs (asserted hash for hash), writing `leakage_results/step14_facets_v2.json` and
`leakage_results/step14_long_v2.csv`, printing every number. The first record is not modified. `step14_findings.md`
§6 and §9, Figures 20–21 and the leaderboard-style table are regenerated from the v2 record with a dated note;
the first run's facet tables are kept in the findings as superseded.
