# Pre-registration, Step 19: does the model predict how a compound changes the SPREAD of a cell population?

Written 2026-10-11, after Step 18 closed (`step18_findings.md`, commit da39c38) and before any Step 19 code exists.
No arrays are produced by this step: it is a further read of the per-cell populations already stored and hash-pinned
by the Step 18 evaluation. CPU only.

## 1. Why this question

Every endpoint the thesis has used so far scores the *average* response of a condition (which genes move, how large
the shift is, which compound it was) or the *distance* between two whole populations (energy). None asks whether the
model gets the **shape** of the treated population right — specifically, whether a compound makes the cells more alike
or more varied, and whether the model predicts that per compound.

This matters for two reasons.

**It is something a linear reference structurally cannot do.** Every linear baseline in this thesis adds one fixed
vector to the matched vehicle cells. Translating a cloud of points does not change its spread, so a shift baseline
predicts "this compound changes the spread by a factor of exactly 1" for every compound, always. (Clipping at zero
perturbs this slightly; §5 measures that rather than assuming it away.) A generative model draws each cell
separately and can predict a change. If the model tracks real changes in spread, that is a capability no linear
reference in the thesis can have — unlike every endpoint used so far, where the linear references are competitive or
ahead.

**It decides whether Step 18's fidelity gain is worth optimising.** Step 18 showed OT-paired training produces cell
populations closer to the real treated cells (energy −0.068 to −0.077, all three splits), at a cost on the
gene-level endpoints, with the generated cell following its conditioning control cell (tracking +0.30 against a null
of +0.000). Two readings fit those numbers: the model learned to produce *correctly* shaped populations, or it
learned to copy the control population's shape, which is realistic but not compound-specific. The first predicts
that `ot_pair` tracks real per-compound changes in spread better than the recipe; the second predicts it does not,
because a copied spread carries no compound information. This step separates them before any further GPU is spent.

## 2. Data and scope

The per-cell files written by the Step 18 evaluation, which store for every held-out condition the generated cells,
the real treated cells and **the conditioning control cells** (`ctrl`), hash-pinned in `provenance/s18ev_*` and
`provenance/s18rv_*`. Only the Step 18 arrays carry `ctrl`, so the arms are:

- **`ot_pair`** — the OT-paired arm, 3 splits × 3 seeds (`s18_<split>_ot_pair_seed_<k>`)
- **recipe** — the Step 15b recipe re-evaluated under the Step 18 harness, 3 splits × 3 seeds
  (`s18r_<split>_recipe_seed_<k>`), which reproduces the Step 15b arrays bitwise (Step 18 §2)

Three splits (`drug_split_1` development, `drug_split_0` and `drug_split_2` confirmation), 2,899 / 3,115 / 2,853
conditions, up to 200 cells per population. Base Squidiff, guidance, attn_cross and mlp_ctrl are **out of scope**:
their evaluations predate the `ctrl` storage, and re-running them is GPU work this step is designed to avoid.

## 3. The measure

For a population of cells `X` (rows = cells, columns = the 200 genes), **spread** is the mean Euclidean distance of
its cells to their own centroid:

    S(X) = mean_i || x_i - mean(X) ||

For each condition this gives `S_true` (real treated cells), `S_pred` (generated cells) and `S_ctrl` (the
conditioning control cells, identical across arms for a given condition and seed by construction of the harness).

The **spread change** of a condition is the ratio

    R = S(treated) / S(control)

computed as `R_true = S_true / S_ctrl` from the data and `R_pred = S_pred / S_ctrl` from the model. R > 1 means the
compound made the population more varied than its vehicle control; R < 1 means more uniform.

**Heterogeneity agreement (HA).** Within each (cell line, dose) context holding at least 10 conditions, the Spearman
correlation between `R_pred` and `R_true` across that context's conditions; HA is the mean over contexts. Contexts
are formed exactly as in Step 16 §4.4 (the MRA machinery), so the context construction, the ≥10 threshold and the
mean-over-contexts rule are inherited from a tested implementation rather than chosen here. Per arm, HA is computed
per seed and then averaged over seeds.

Contexts are defined within a split, which keeps every comparison inside the held-out compounds of that split.

## 4. Endpoints

Intervals are 97.5 % structure-clustered percentile bootstrap intervals over the split's held-out structures (5,000
replicates, `RandomState(0)`), as in Steps 14–18. All three endpoints are evaluated per split.

- **Q1, primary (gated): `ot_pair` HA − 0.** Does the OT-paired model track real per-compound changes in spread at
  all? Above zero on all three splits is the condition for any claim in this step.
- **Q2, co-primary (gated): `ot_pair` HA − recipe HA.** Does OT pairing improve it? This is the question that
  decides whether the Step 18 mechanism is worth optimising.
- **Q3, gate (interpretation): the linear references' HA.** The four shift baselines of Step 15b (ridge-S7,
  ridge-train, compound-blind mean, control-paste), rebuilt from the same control cells by the Step 15b code path.
  For the claim in §1 to hold, these must not resolve above zero. If any does, the "a shift cannot do this"
  statement is false as applied to these data and the step reports that instead.

**Secondary, reported, not gated:** recipe HA − 0; the absolute HA of every submission; the mean, median and spread of
`R_true` per split (how much compounds actually change spread on this panel); the fraction of conditions with
R_true > 1; `S_pred / S_true` per arm (whether the generated spread is right in absolute terms, not only in rank).

**Exploratory, pre-declared:** the Spearman between a condition's HA residual and its learnability (Step 15b §6), and
between R_true and the measured response size — whether spread changes most where the compound does most.

## 5. The replicate ceiling and the zero point

**Ceiling.** On the conditions present on both plate 6 and plate 14 at the top dose (259 / 299 / 246 pairs, as in
Steps 15b–16), the plate-14 copy's measured `R` is offered as a prediction of the plate-6 copy's and scored by the
same HA definition, and vice versa, averaged. This is how well a repeat of the experiment predicts itself, and it
bounds what any model could achieve while the data are this noisy. It is computed on 14 contexts against the arms'
42, so it is reported beside the arms and **never used as a gate** — the Step 16 §9 lesson, where a like-for-like
ceiling comparison dissolved a claim built on mismatched context sets.

**Zero point.** `control_paste` returns the vehicle cells unchanged, so `S_pred = S_ctrl` and `R_pred = 1` for every
condition; the Spearman of a constant is undefined and the scorer records HA = 0 for it by definition, asserting
that its `R_pred` is 1 to within 1e-9. The three additive shifts clip at zero, which changes spread slightly, so
their `R_pred` is **not** exactly 1; the scorer measures the resulting variation and reports it rather than assuming
it is negligible. This is the honest form of the §1 claim.

## 6. Verdict categories, applied mechanically

Precedence is fixed here and coded in the scorer's docstring before it runs: 1 → 2 → 5 → 3 → 4.

1. **SPREAD PREDICTED, OT BETTER** — Q1 above zero on all three splits and Q2 above zero on all three.
2. **SPREAD PREDICTED** — Q1 above zero on all three splits; Q2 not above zero on at least one.
3. **REVERSED** — Q1 below zero on any split.
4. **PARTIAL** — Q1 above zero on one or two splits.
5. **UNRESOLVED** — Q1 includes zero on all three splits.

If Q3 resolves above zero for any linear reference on any split, the verdict is reported with the clause "the linear
references are not inert on this measure", and the §1 framing is withdrawn in the findings, whatever the category.

## 7. Expectation, recorded before the run

I expect **category 2, SPREAD PREDICTED**: Q1 above zero on all three splits at an HA of roughly 0.1 to 0.3, and Q2
including zero — that is, both generative arms track real spread changes somewhat, and OT pairing is not clearly
better at it than the recipe.

My reasoning, stated so it can be audited against the outcome: the generative arms do carry compound-specific
direction (Step 15c, PDS 0.71–0.72 against linear references at chance), so some spread signal is plausible; but
Step 18's tracking result says OT pairing's realism comes from following the conditioning control cell, and a
followed control cell carries the *control's* spread, not the compound's effect on it. If that is the whole story,
OT pairing should be no better here, and could be worse.

I also expect the ceiling to be low — the Tahoe authors report between-plate agreement of 0.29 for their own
heterogeneity measure (Zhang et al., Cell 2026), computed differently and on more cells, so a modest ceiling here
would be unsurprising and must be reported beside any result rather than after it.

## 8. What this step cannot establish

1. **That the model is biologically right about heterogeneity.** HA is a rank agreement across conditions within a
   context; it says nothing about whether the predicted spread is correct in size. `S_pred / S_true` is reported for
   that and is not gated.
2. **Anything about the arms without stored control cells** (base, guidance, attn_cross, mlp_ctrl) — §2.
3. **Anything about Gini or cell-cycle phenotypes.** Gini requires clustering with a resolution parameter, and
   cell-cycle scoring requires enough cell-cycle genes on the 200-gene panel; neither is attempted here, and both
   would need their own pre-registration and, for the cell cycle, a feasibility check on the panel first.
4. **A causal reading of Q2.** If OT pairing is better, this step does not separate "it predicts spread changes"
   from "it inherits a spread that happens to correlate with the truth".
5. **Seventh use of the Tahoe confirmation splits** (Steps 15b, 15c, 15d, 15e, 16, 18, 19). No model choice is made
   here; the development split is scored alone first and the confirmation splits only after.
6. **Power.** With 42 contexts per split and ≥10 conditions each, an HA below roughly 0.05 is unlikely to resolve;
   "unresolved" will not be read as "equivalent".

## 9. Procedure and discipline

1. This document is committed before `score_step19_heterogeneity.py` exists.
2. The scorer is committed before it is run, refuses to run from a modified working tree, and refuses unless this
   pre-registration is committed and has at least 9 sections.
3. It re-checks every per-cell array against its `provenance/s18ev_*` / `s18rv_*` hash before reading, and
   reproduces Step 18's tracking numbers to 1e-9 as proof it is reading the same arrays.
4. It asserts the `ctrl` populations are identical across the two arms for each (condition, seed), since both were
   drawn by the same seeded harness draw — the comparison is invalid otherwise.
5. The development split (`drug_split_1`) is scored alone and its log read before the confirmation splits are
   scored. No endpoint, interval, context rule or category is changed after any result is seen; a correction before
   the run is added as a dated paragraph inside an existing section, never by rewriting one.
6. Records: `tahoe_results/step19_heterogeneity.json`, `step19_long.csv`, `step19_per_condition.csv` and their
   `_development` counterparts, with the scoring logs under `tahoe_results/step19_logs/`.
7. Findings go in `step19_findings.md` with an expectation audit against §7.
