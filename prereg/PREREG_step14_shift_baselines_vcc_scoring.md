# Pre-registration: Step 14, linear-shift baselines and a six-facet (VCC-style) scoring of every arm

**BLIND for every quantity this step computes: no shift baseline has ever been scored on
DE-overlap, energy distance or any VCC-style facet, and no arm of this thesis has ever been
scored on a VCC-style facet or against a replicate anchor. NOT BLIND for the DPC-50 of the
Step 7 ridge, which this step reproduces by construction (section 2).**

Author: Noha Samir Ismail. Date: 2026-09-22.

## 1. Questions

Steps 7–13b established that a linear model with fingerprint × cell-line features recovers
more of the compound-specific signal (DPC-50) than any generative arm, while the generative
arms improved on DE-overlap and energy distance along the ladder. The two families have never
been scored on the same metrics: the ridge has no DE-overlap or energy-distance record, and no
arm has an absolute reference point beyond the chance floor.

**Q1. Does the generative model add anything over a linear shift of the control cells?** A
linear shift is each vehicle cell of a cell line plus the ridge-predicted mean delta for the
condition — a "submission" with the ridge's compound-specific direction and the control
distribution's shape. Scored on the thesis's three metrics against the ladder's unguided 50 NFE
rung.

**Q2. Where does every arm sit between "no compound-specific information" and "an
experimental replicate"?** The 2026 Virtual Cell Challenge scores six facets, each placed on a
scale whose 0 is the context's mean perturbation response and whose 1 is the reference
compared with itself (`cell-eval2` 0.15.0, specification of 2026-08-19). The same six facets,
adapted as section 4 states, are computed for every arm with stored per-cell predictions and
for the shift baselines. These are descriptive; nothing is gated on them.

## 2. What is known, and its status

- Step 7 (`leakage_results/step7_interaction_baseline.json`): interaction ridge (FP + line +
  FP × line, alpha 1.0), fitted *within the held-out set* by compound-grouped 5-fold CV,
  marginal median DPC-50 0.2473 / 0.1884 / 0.0816 on splits 0 / 1 / 2; flow matching at
  20 NFE 0.0888 / 0.0924 / 0.0626. The ridge-shift baseline of this step has, by construction,
  exactly the Step 7 ridge's per-condition DPC-50 (DPC-50 depends only on the condition's mean
  delta), and the analysis asserts that equality to 1e-9 as its reproduction check.
- The Step 7 ridge learns from the held-out compounds' own responses (about 30 compounds to
  predict 7); the generative arms learn from the 112 training compounds. A like-for-like ridge
  fitted on the training compounds' condition means is added here (section 3) so that Q1 is
  answered against both.
- Ladder levels on DE-overlap (Figures 17–18): unguided 50 NFE 0.6100 / 0.6124 / 0.6106
  (splits 1 / 0 / 2); guidance w = 2 0.6172 / 0.6275 / 0.6222. Energy distance, unguided
  50 NFE: 0.1795 / 0.0483 / 0.1178.
- Diagnosis carried into this step (Step 13 §8 as corrected, Step 13b §10): the conditioning
  pathway transmits a small fraction of the compound-specific signal whatever the drug
  representation. Q1 asks whether the generative model's remaining advantage — the shape of
  the cell distribution — outweighs that.

## 3. Submissions and reference points, per split

Every submission is a set of predicted cells per held-out condition (cell line × compound ×
dose), scored against the real cells of that condition and the vehicle cells of that line.
"Vehicle cells" are the source file's `dose == 0` cells of the line, restricted to the split
file's 200 genes (the same cells that supplied the arms' paired controls).

**Generative arms** (stored per-cell predictions; five checkpoints each, pooled per condition
by arithmetic mean of per-condition metric values, as every earlier step):

| arm | split 1 files | splits 0 / 2 files |
|---|---|---|
| adaLN DDIM, 1000 steps | `sw_ddim_seed_k_n1000` | `s6b_<split>_ddim_seed_k` |
| flow matching, 20 NFE | `s6pc_fm_gaussian_seed_k` | `s6b_<split>_fm_gaussian_seed_k` |
| flow matching, 50 NFE (unguided; **the comparator for Q1**) | `sw_fm_gaussian_seed_k_n50` | `s11_<split>_fm_gaussian_nfe50_seed_k` |
| guidance w = 2 | `s12_cfg_w2_seed_k` | `s12b_<split>_w2_seed_k` |
| cross-attention (`attn_cross`) | `s13_drug_split_1_attn_cross_seed_k` | `s13b_<split>_attn_cross_seed_k` |
| matched control (`mlp_ctrl`) | `s13_drug_split_1_mlp_ctrl_seed_k` | `s13b_<split>_mlp_ctrl_seed_k` |

Base Squidiff has no per-cell record and is scored only on the two profile-level facets (PDS,
expression error) from its stored mean vectors, marked as such. Split-1-only extras (Step 9
pathway / both; Step 10 ChemVAE arms) are scored and reported as descriptive rows.

**Shift baselines** — each vehicle cell of the line plus a per-condition delta, clipped at 0
as the arms' predictions are (`clamp_min = 0`), the number of shifted cells per condition equal
to the condition's number of real cells (cap 200), the vehicle cells sampled once per line with
`RandomState(0)` and shared by all four baselines:

| baseline | delta |
|---|---|
| **ridge-S7** | Step 7 interaction ridge, reproduced with Step 7's exact procedure (within-held-out compound-grouped 5-fold CV, alpha 1.0) |
| **ridge-train** (like-for-like) | interaction ridge, same features and alpha, fitted on the training split's conditions — delta = mean treated minus mean paired control per training condition, from the split's train and train-control files — and applied to the held-out conditions |
| **mean-response** (the challenge's 0 point) | per line, the equal-weight mean over the line's held-out conditions of (condition mean − vehicle mean); the same delta for every condition of the line; transductive by construction, as in the challenge |
| **control-paste** (zero shift) | delta = 0; the thesis's control-mean baseline |

**Replicate anchor** (the challenge's 1 point): per condition, the real cells are split into two
disjoint halves, and per line the vehicle cells likewise; one half is scored as a submission
against the other with its own vehicle half; five splits, seeds 0–4; the anchor is the mean.

## 4. Metrics

**The thesis's three**, exactly as in every earlier step: DE-overlap Jaccard (true top-50
|delta| genes), energy distance (amendment-13 U-statistic on per-cell data), DPC-50 (Pearson on
the true top-50 genes; mean over conditions and paired condition median).

**Six VCC-style facets**, definitions from `cell-eval2` 0.15.0 §§1–7, with these adaptations,
each forced by the data and none a choice among alternatives:

| item | challenge | here |
|---|---|---|
| gene space | 18,533 genes, raw counts | the split's 200 highly variable genes, library-size-normalised log1p values, as stored |
| context | one cell line, 300 CRISPRi perturbations | one cell line, its held-out conditions (compound × dose; 37–38 compounds × 4 doses, minus absent conditions) |
| group profile (PDS, MSE) | log1p of counts summed over cells and normalised to 5 × 10⁴ | mean of the stored log1p values over the group's cells; effect = profile − vehicle profile of the line |
| PDS | cosine rank among 300 measured effects | cosine rank among the line's measured condition effects; `1 − k/(n − 1)`, ties averaged; 0.5 = no information |
| expression error | unbiased, capped, normalised (jackknife corrections) | Σ‖b̂ − b‖² / Σ‖b − b_ctrl‖² over the line's conditions, no sampling-noise correction (stated; the corrections are defined for counts) |
| DE test | Wilcoxon rank-sum, two-sided, on counts per 10⁶; 5-CPM gate on control cells | Wilcoxon rank-sum, two-sided, on `expm1` of the stored values (library-size-normalised counts); no expression gate (all 200 genes are highly variable by construction) |
| significance | BH within each perturbation, α = 0.05 | identical |
| fold change | log₂ of mean normalised counts, ε = 10⁻⁹ | identical, on `expm1` of the stored values |
| target-gene exclusion | own target gene removed; all 300 removed for PDS | none (no target gene exists for a compound) |
| direction fidelity, reach (P₀ = 0.9), significant-set Jaccard, fold-change NMAE (gate ≥ 10 genes) | §§4–7 | identical |
| cell-count gate | 400 cells per perturbation | conditions with fewer than 20 real cells are excluded from the four DE facets and from the replicate anchor, for every submission alike; the count is reported per split |
| scaled score | `s = (u − b)/(r − b)` per context, mean over six | identical, per line; the split's value is the mean over its three lines |

No number produced here is comparable to a challenge leaderboard entry, and none will be
placed beside one.

## 5. Endpoints

All intervals: drug-clustered percentile bootstrap over the split's compounds, 5000
replicates, `RandomState(0)`, **97.5% per split, on all three splits**. There is no development
/ confirmation distinction in this step because nothing in it is chosen on any split: the
ridge's alpha is Step 7's, the shift constructions and gates are fixed above, the metrics are a
published specification, and no checkpoint is trained. Every split is therefore a confirmation
split for this step, and the categories below require agreement across all three.

**Primary, gated (Q1):** DE-overlap Jaccard, ridge-S7 shift minus unguided 50 NFE, per split.

**Co-primary, gated (Q1):** energy distance, ridge-S7 shift minus unguided 50 NFE, per split.
Higher is worse.

**Secondary, reported, not gated:** the same two contrasts for ridge-train, mean-response and
control-paste against unguided 50 NFE, and for ridge-S7 against guidance w = 2; DPC-50 for
every submission (the ridge-S7 shift reproduces Step 7's to 1e-9 or the analysis stops); the
six VCC-style facets, raw and scaled, for every submission, per line and per split, with
97.5% intervals over compounds on each raw facet for the contrast ridge-S7 minus unguided
50 NFE; the count of conditions passing the cell gate.

**Sensitivity.** The DE-overlap contrasts of Steps 11–13b between generative arms resolved at
half-widths of about 0.003; a contrast between a shift baseline and an arm has never been
computed, its variance is unknown, and no power is claimed.

## 6. Outcome categories, fixed now

1. **LINEAR SHIFT AHEAD** — primary above zero on all three splits and co-primary not above
   zero on all three. The ridge shift beats the generative model on the genes it ranks and
   costs nothing in distributional fidelity: the ladder's DE-overlap gains are recovered by a
   linear model, and the thesis says so.
2. **GENERATIVE ADVANTAGE** — primary below zero on all three and co-primary above zero on
   all three. The generative model ranks genes better and shapes the distribution better than
   a linear shift, despite losing on DPC-50.
3. **TRADE** — primary above zero on all three and co-primary above zero on all three (the
   shift ranks genes better but the generative distribution is closer), or the reverse. The two
   families win different facets; reported per facet, neither declared better.
4. **PARTIAL** — either contrast resolved on one or two splits only. Reported per split.
5. **UNRESOLVED** — both contrasts include zero on all three splits. Not equivalence.

## 7. Expectation, recorded in advance

**I expect TRADE**, in the direction: ridge-S7 shift above unguided 50 NFE on DE-overlap on
splits 0 and 1 — the ridge's better delta direction should carry into the top-50 ranking —
with split 2, where the ridge's own DPC-50 is lowest (0.0816), the one most likely to be
unresolved; and the shift *worse* on energy distance on all three splits, because it translates
the vehicle distribution rigidly while the generative model reshapes it. If the shift is not
worse on energy, the generative model's remaining claim is gone and the thesis says so. I
expect ridge-train below ridge-S7 on DPC-50 and on DE-overlap (fewer in-distribution
conditions to learn from), but I do not know the direction with confidence and record it as
uncertain. On the VCC-style facets I expect every generative arm's raw PDS within 0.05 of the
0.5 no-information point, the ridge shifts above it, and every scaled score far below 1 on
every facet, with fidelity and reach the lowest.

## 8. What this cannot establish

- **Comparability with the challenge.** Different data, gene space, perturbation type and
  parameters; the facets are borrowed for their construction, not for their numbers.
- **That the mean-response baseline is reachable.** It is transductive, as in the challenge;
  it bounds triviality, not skill.
- **That the within-held-out ridge is the right comparator.** It uses the held-out compounds'
  own responses; ridge-train is the like-for-like version and both are reported.
- **Anything about single-cell shape beyond energy distance and the six facets.**
- **A new gated claim on the VCC-style facets.** They are descriptive here.
- **Independence from earlier use of the splits.** Sixth use of splits 0 and 2 and the
  development split; no model choice is made, which is why all three are used.

## 9. Analysis plan and provenance

`score_step14.py`, committed before it is run, reads the per-cell files and vectors of every
submission in section 3 with their provenance hashes, the split's test / train / control files
and the source file's vehicle cells, builds the four shift baselines and the replicate anchor,
computes every metric of section 4, applies section 6 mechanically, prints every number it
uses, and writes `leakage_results/step14_analysis.json` and `leakage_results/step14_long.csv`
(one row per submission × split × line × facet × value). It stops if any input is missing, if
any hash differs from its record, or if the ridge-S7 shift's DPC-50 differs from the Step 7
record.

- Step 7 record: `leakage_results/step7_interaction_baseline.json`; procedure `analyse_step7.py`.
- Metric specification: `cell-eval2` 0.15.0, `docs/vcc2026_metrics/vcc2026-metrics-brief.md`
  (2026-08-19), ArcInstitute/cell-eval2; the adaptations of section 4 are this document's.
- Findings this step revises if category 1 or 2 is reached: Step 13 §8, Step 13b §10.
- Pre-registration commit: [hash]. Scoring script commit: [hash].

---

## 10. Amendment 1, written before any real data were scored (2026-09-23)

`score_step14.py` was first run on a synthetic environment (invented numbers, the real files never
read). That run showed a flaw in section 4 as written. The challenge measures effects against a
pooled control because all of a context's perturbations share it. Here every condition has its own
*matched* vehicle cells (`paired_control_index`), and every arm's predictions were generated from
exactly those cells. Taking the profile effects against the line's vehicle pool lets the matched
controls' deviation from the pool enter both the reference profile and every submission's profile,
so even a zero-shift submission discriminates conditions (PDS far above 0.5 on the synthetic data).
The thesis's own metrics already avoid this by taking both deltas against the paired-control mean
`C`. The facets are therefore aligned with that origin, and the source file's vehicle pool is not
used at all:

| item | section 4 said | amended to |
|---|---|---|
| origin of profile effects (PDS, MSE) | the line's vehicle-pool profile | the condition's paired-control mean `C`, for the reference and every submission alike |
| DE comparison group (Wilcoxon), both sides | the line's vehicle pool | the condition's paired vehicle cells (the same draw of ≤ 200 that the shift baselines are built from) |
| replicate anchor | halves of the real cells and of the vehicle pool | halves of the real cells and of the condition's paired vehicle cells; each half against its own vehicle half |
| shift-baseline cells | vehicle cells of the line, sampled per line | the condition's paired vehicle cells (they are vehicle cells of the line, selected by the pairing) |

With this origin the zero-shift baseline has a zero effect vector for every condition and scores
PDS 0.5 exactly, the specification's stated value for a control-pasting submission. Nothing else in
sections 3–9 changes; the categories, comparator, level and expectation stand.
