# Pre-registration: Step 6, flow matching

**Written and committed BEFORE any flow-matching model is trained.** No Step 6
checkpoint exists and no flow-matching code has been run. Blind prospective
pre-registration.

Author: Noha Samir Ismail. Date: 2026-08-06.

## 1. Question

Steps 4, 4b and 5 established, on the same backbone:

- adaLN improves DE-overlap and replicates on two held-out splits (+0.0487)
- the gain is NOT attributable to semantic conditioning (two blind tests, both
  unresolved)
- gene-level mixing after the output layer gives no resolved improvement
- the model still loses to fingerprint ridge regression

Across those experiments a pattern appeared that is more general than any
architecture: **the denoising training loss is a poor guide to the biological
endpoint.** Base and GeneAttention differ by 0.9% in final training loss and
47% in DE-overlap. Within the GeneAttention arm, five checkpoints identical in
logged loss span a tenfold range in signal above the chance reference.

Step 6 asks two questions:

**Q1 (performance).** Does replacing the 1000-step DDIM denoising objective
with conditional flow matching, holding the network fixed, change DE-overlap or
distributional accuracy?

**Q2 (transferable).** Does the loss/endpoint dissociation persist under a
different generative objective? If it does, it is a property of the task rather
than of denoising diffusion.

Q2 is the reason this experiment is worth running even if Q1 is null.

## 2. Arms

Three arms, five independently trained checkpoints each, all trained fresh
under one commit on `drug_split_1`. Fifteen checkpoints.

| arm | objective | source p0 | coupling |
|---|---|---|---|
| **A. DDIM** | epsilon-prediction, 1000 steps | Gaussian | n/a |
| **B. FM-gauss** | conditional flow matching | Gaussian | independent |
| **C. FM-control** | conditional flow matching | control cells | minibatch OT |

The network is IDENTICAL in all three: 200 -> 2048 input layer, three
`MLPBlock`s at 2048 with semantic adaLN modulation, 2048 -> 200 output layer,
58,178,180 parameters. No architectural change is made in Step 6.

**Why three arms rather than two.** B minus A isolates the change of objective
with the source held Gaussian. C minus B isolates the change of source and
coupling with the objective held fixed. A two-arm design would confound them.

Arm A is retrained from scratch rather than reusing existing adaLN
checkpoints, so that all three share a commit and no cross-revision caveat
arises. Existing adaLN checkpoints will be reported alongside as an incidental
replication check, not as an arm.

## 3. What changes, and what does not

This is NOT a one-line change. The following differ between A and B/C:

| component | arm A | arms B and C |
|---|---|---|
| network input | noisy expression at discrete timestep | interpolated state x_t at continuous t |
| regression target | epsilon | conditional velocity |
| time sampling | discrete t in [0, 999] | t ~ Uniform[0, 1] |
| sampler | DDIM, 1000 steps | ODE integration, pre-specified NFE |
| output constraint | `pred_xstart.clamp(0,)` at every step | softplus/clamp at t=1 only |

**Path and target.** Straight-line interpolation x_t = (1-t)*x_0 + t*x_1 with
conditional velocity target v = x_1 - x_0. Chosen over a Gaussian diffusion
path because straight paths are the basis of the low-NFE claim and because the
target is constant along the path.

**Sampler.** Euler at **20 function evaluations** for the gated comparison,
fixed in advance. A step-count sweep over {1, 5, 10, 20, 50} and a Heun solver
check are SECONDARY and cannot alter the gated verdict.

**Non-negativity.** Expression is log1p-transformed and non-negative. Arms B
and C apply a clamp at the final integration step only. Arm A retains the
existing per-step `clamp(0,)`. This is an unavoidable asymmetry: it is intrinsic
to the two samplers, not a choice, and is recorded as a limitation rather than
controlled.

**Implementation.** `torchcfm` (MIT licence) supplies the conditional flow
matcher and the OT coupling. Only its loss and coupling classes are used; the
network, the encoder, the data pipeline and the evaluation harness are ours and
unchanged.

## 4. Endpoints

**Two co-primary endpoints**, both gated, with multiplicity handled by
splitting alpha: each is tested at the **97.5% level** (Bonferroni over two).

**P1. Raw DE-overlap Jaccard**, arithmetic mean of condition-level values over
the 443 eligible conditions of `drug_split_1`. Comparable with every prior
step. Chance reference c_s = 0.206228, the exact hypergeometric expectation
E[I/(100-I)] over the 147 globally ever-active genes.

**P2. Energy distance** between the predicted and true perturbed cell
populations, per condition, averaged over conditions. Pre-registered because
the flow-matching literature reports its clearest gains on distributional
rather than mean-expression metrics; omitting it would leave the experiment
underpowered against the effect flow matching is most likely to produce.
Including it in advance prevents post-hoc metric selection.

**Gated contrasts.** For each endpoint: B minus A, and C minus B. Unpaired
two-sample Welch, five per arm, Welch-Satterthwaite df. Exhaustive two-sided
5-vs-5 permutation over all 252 allocations as support; its floor is 2/252 =
0.0079.

**Sensitivity, stated in advance.** Measured within-arm SD on P1 in prior steps
is 0.0071 to 0.0117. At SD = 0.008 the SE of a difference is 0.0051, the 97.5%
half-width is 0.0130, and the MDE at 80% power is approximately **0.0173**. A
difference smaller than roughly 0.017 on P1 will not be resolved.

**No prior SD exists for P2.** Power for the energy-distance endpoint is
therefore UNKNOWN in advance. This is disclosed rather than estimated. The
observed SD will be reported, and an unresolved P2 result will not be described
as evidence of no effect.

**Missingness.** Every condition-level value must be finite for all fifteen
checkpoints. The analysis asserts identical condition labels and order across
arms and ABORTS on any non-finite value. Conditions and seeds are never
dropped.

## 5. Secondary, not gated

- **Q2, the dissociation analysis.** For all fifteen checkpoints: final
  training loss at full precision (recomputed by forward pass, not read from
  three-significant-figure logs) against DE-overlap. Reported as within-arm and
  between-arm Spearman correlations, with the caveat that five points per arm
  cannot support a within-arm claim on its own. The comparison of interest is
  whether the loss-endpoint relationship differs between objective A and
  objectives B/C.
- Step-count sweep {1, 5, 10, 20, 50} and Heun solver check for arms B and C.
- Wall-clock sampling time per condition, all arms.
- B-C and DPC-50 for all arms.
- Ridge and control-mean baselines on the same conditions, always reported.
- Incidental replication: existing adaLN checkpoints versus arm A.

## 6. Outcome categories, fixed now

Decided per endpoint. The first applicable category applies.

1. **Flow matching improves the endpoint** - the B-minus-A interval lies
   entirely above zero (P1) or entirely favours B (P2). Reported per endpoint;
   an improvement on P2 alone is a distributional result and must NOT be
   described as improved prediction accuracy.
2. **Control-source improves over Gaussian-source** - the C-minus-B interval
   excludes zero. Attributes the gain to the source and coupling, not the
   objective.
3. **Flow matching degrades the endpoint** - interval entirely on the
   unfavourable side. Reported as observed.
4. **UNRESOLVED** - the interval includes zero. NOT equivalence. No margin is
   justified in advance and none may be claimed afterwards.

**A separate criterion for the sampling-efficiency claim**, which is not a
statistical test: flow matching is recorded as more efficient if arms B or C
reach a DE-overlap within the P1 MDE of arm A at 20 NFE or fewer, against arm
A's 1000. This is a descriptive engineering comparison and is labelled as such.

**No arm may be added after seeing these results.** If flow matching
underperforms, that is the result; a fourth arm with different hyperparameters
would be a post-hoc rescue.

## 7. Verification required before training

To be performed and recorded before any Step 6 checkpoint is trained.

- **Arm A reproduces the existing adaLN result.** A retrained arm A must fall
  within the seed range of the existing adaLN checkpoints (0.5455 to 0.5679 on
  drug_split_1). A departure means the training path changed and must be
  explained before proceeding.
- **The FM loss is implemented correctly.** On a synthetic two-Gaussian
  problem with a known optimal transport map, the trained velocity field must
  recover the analytic solution to within a stated tolerance. This is a unit
  test of the objective, independent of the biology.
- **Velocity target verified by finite difference.** For the straight-line
  path, (x_{t+h} - x_t)/h must equal the supplied target to numerical
  precision.
- **Arms B and C are architecturally identical to A.** Parameter count exactly
  58,178,180; state-dict keys identical; only the loss and sampler differ.
- **Non-negativity holds at output.** Sampled expression must satisfy x >= 0
  for every condition in all arms.
- **Evaluation harness unchanged.** `leakage_test_base_sciplex_array.py` and
  `sample_squidiff.py` must be byte-identical to the Step 5 commit for arm A.
  Flow-matching sampling uses a SEPARATE entry point; the existing harness is
  not edited, so arm A remains comparable to Steps 4, 4b and 5.
- **Provenance per run:** resolved flags, constructed parameter count, git
  commit, node, GPU read from torch, data and script hashes, checkpoint hash,
  and the architecture detected from the saved state dict.

**Arm identity is recoverable** from the training configuration and the sampler
used, but NOT from the weights alone: arms A, B and C have identical
architectures. As in Step 4b, arm identity therefore rests on the recorded
configuration plus the frozen code, and a sidecar recording the objective and
source is written beside each checkpoint.

## 8. Splits

Training and both gated analyses use `drug_split_1` only.

`drug_split_0` and `drug_split_2` are NOT used and NOT spent. They remain
reserved for confirmation of any resolved Step 6 finding, under a separate
pre-registration.

**Disclosure:** those splits are not untouched. They carried the Step 4
replication, so a future use would be a SECOND use. They did not inform any
Step 6 design decision, with one qualification: semantic adaLN was chosen as
the shared backbone because it is the best-performing architecture to date, and
that judgement drew partly on those splits.

## 9. What this cannot establish

- **Whether flow matching beats ridge regression.** Prior work (Ahlmann-Eltze,
  Huber and Anders, Nature Methods 2025) found that deep perturbation models
  did not outperform linear and no-change baselines on comparable data, and
  amendment 12 found no evidence that adaLN exceeds ridge on the confirmation
  splits. Ridge and control-mean are reported for every arm; beating them is
  not the pre-registered criterion.
- **Whether a different flow-matching variant would differ.** One path
  (straight-line), one solver family, one NFE for the gated test, one
  parameterisation.
- **Anything about generalisation.** One split, and the development one.
- **Whether the loss/endpoint dissociation is causal.** Q2 is observational
  across fifteen checkpoints; five per arm cannot support a within-arm claim,
  and the between-arm comparison is descriptive.
- **Whether a larger or gene-token architecture would behave differently.** The
  network is deliberately held fixed so that the objective is the only variable.
- The per-step versus final-step clamp asymmetry between arms A and B/C is
  intrinsic to the two samplers and is not controlled.

---

## 10. Amendment, 2026-08-17: two training commits, verified equivalent

Section 3 requires all three arms to train under one commit. They did not.

- Arm A (ddim), five seeds, commit `8e11f73`
- Arms B and C, five seeds each, commit `f4a49f8`

**Why.** The first submission ran all fifteen tasks. All five arm-A runs
completed; all ten flow-matching runs failed after 88 seconds with
`KeyError: 'x_start'`. `MLPModel.forward` reads `model_kwargs['x_start']` to
build z_sem, and arm A's `GaussianDiffusion.training_losses` inserts it at
`diffusion.py:756`, but `FlowMatchingWrapper` received it positionally and
never added it back. The fix touched `Squidiff/flow_matching.py`; the ten runs
were then repeated. Arm A was not retrained.

**Why this is not a cross-version confound.** `git diff 8e11f73 f4a49f8`
changes exactly two files: `Squidiff/flow_matching.py`, which arm A never
imports, and `train_step6.sh`, the launch wrapper. Every file determining arm
A's training path - `train_squidiff.py`, `Squidiff/train_util.py`,
`Squidiff/MLPModel.py`, `Squidiff/script_util.py`, `Squidiff/diffusion.py` -
is byte-identical between the two commits, verified by
`git diff --name-only` returning empty over those paths. Arm A therefore
trained under exactly the code it would have trained under at `f4a49f8`.

**Recorded rather than corrected.** Retraining arm A under `f4a49f8` would cost
about two hours and change nothing measurable. The deviation is disclosed here
so that a reader encountering two commits in the provenance records finds the
reason and the verification rather than an unexplained discrepancy.

**Also recorded.** The x_start defect was not caught by the synthetic
two-Gaussian test in section 7, because that test hand-built a kwargs dict that
already contained `x_start`. The corrected verification passes a
TrainLoop-shaped `micro_cond` with no `x_start` key, and gives per-sample
losses of 1.1656, 2.1082 and 1.8956 for arms A, B and C. A unit test that
constructs its own inputs cannot detect a missing input.

---

## 11. Amendment, 2026-08-17: Protocol A for arm C

**Written after arm C's five evaluations failed.** Disclosed as such: this is
not a blind decision.

### What happened

Arm C integrates its ODE from control cells. The evaluation harness runs three
protocols per condition:

- **A** conditions the encoder on TREATED cells, to detect leakage
- **B** conditions on CONTROL cells, the genuine prediction
- **C** is the control mean, no model

Protocol A has no control cells by definition, so arm C had no defined starting
point and all five runs raised `_fm_x0 is unset` at the first Protocol A decode.
This was not anticipated when section 3 was written.

### Decision

**Arm C's Protocol A integrates from the treated cells it is conditioned on;
Protocol B integrates from the control cells it is conditioned on.** Each
protocol starts from whatever it conditions on.

### Why not the alternatives

*Skip Protocol A for arm C.* Protocol A is load-bearing in the frozen harness -
the vector store, the metrics loop, the leakage gap, the summary table and a
final length assertion all require it. Skipping it means editing
`leakage_test_base_sciplex_array.py`, which would break arm A's byte-identical
comparability with Steps 4, 4b and 5. That is a caveat on everything, to avoid
a caveat on one secondary diagnostic.

*Give arm C's Protocol A a Gaussian start.* Then Protocol A would be arm B's
and the A-B difference would confound source with conditioning in a way that is
harder to state than the chosen option.

### What this costs

**Arm C's A-B is not comparable to the other arms'.** For arms A and B, A-B
varies the CONDITIONING with the source held fixed, and measures leakage of
treated information through the encoder. For arm C it varies conditioning AND
source together, so it is not a leakage measure. Arm C's A-B must not be placed
in the same table as the others without this qualification.

**The gated endpoints are unaffected.** Both co-primary endpoints in section 4
use Protocol B, which is unchanged: arm C's Protocol B integrates from control
cells exactly as section 3 specifies.

### Verification

The second `setattr` for `cfA` is inert for every non-arm-C evaluation, by the
same argument and the same test as the first: two runs of a Step 5 checkpoint on
one node must agree bitwise, as demonstrated in `CORRECTION_cn270.md`.

---

## 12. Amendment, 2026-08-17: recovering the per-cell P2 artifact

**Technical-recovery amendment. No endpoint is being redefined.**

### What happened

P1 was computed and inspected: raw DE-overlap Jaccard, arm means 0.5548
(ddim), 0.5858 (fm_gaussian), 0.5859 (fm_control_ot).

P2 could not be computed. The evaluator averages each condition's predicted
cells before serialisation - `predB = decode(zB).mean(0)` - so every stored
array is (443, 200) condition means. Energy distance requires the predicted and
observed cell POPULATIONS within each condition. The necessary intermediate was
discarded.

**No per-cell P2 value has been generated or inspected.**

### What is NOT changing

The P2 definition, its aggregation, its uncertainty procedure, the 97.5% level,
and the joint decision rule of section 4 are unchanged. Only artifact retention
is modified.

A condition-level substitute was considered and REJECTED: with one predicted
mean and one true mean per condition, energy distance degenerates to a scaled
Euclidean distance and carries no within-condition distributional information.
It would not be a lower-resolution P2; it would be a different estimand, and
adopting it after seeing a favourable P1 would lower the evidential bar.

### Status until recovery

**P1 is positive; the pre-registered co-primary analysis is incomplete.** The
joint gate of section 4 is NOT declared until P2 has been computed under its
original definition.

### Recovery

An opt-in `--save_per_cell` mode stores the per-cell prediction array
immediately before the existing mean is taken. Default behaviour is unchanged.
The sampler, model calls, cell counts, and control- and treated-cell selection
are untouched, so RNG consumption is identical and the recovery run must
reproduce the existing condition means bitwise.

The original P1 artifacts remain frozen and are not replaced. Recovery outputs
are written under separate filenames and verified against the frozen means
before P2 is computed.

---

## 13. Amendment, 2026-08-17: P2 estimator and summary, fixed before computation

**No P2 value has been computed or inspected.** Amendment 12 recovered the
per-cell artifact but did not fix the finite-sample estimator or the treatment
of unequal cell counts. Both are fixed here.

### Summary across conditions

P2 for checkpoint s is the **unweighted arithmetic mean** of the per-condition
energy statistic over all 443 eligible conditions. Every condition carries
weight 1/443 whether it holds 7 cells or 200. Cell counts range from 7 to 200,
mean 119.6.

The estimand is **the expected distributional error for a uniformly selected
experimental condition**. Cell-count weighting would estimate the expected
error for a uniformly selected observed CELL - a different question, which
would emphasise whichever drugs, doses and cell lines happened to yield more
cells. A minimum-count restriction would change the estimand to a post-selected
subset and could preferentially discard biologically difficult conditions.
Neither is adopted.

### Per-condition estimator

The **off-diagonal U-statistic**:

    E_U = (2/nm) * sum_ij ||X_i - Y_j||
        - (1/(n(n-1))) * sum_{i!=i'} ||X_i - X_i'||
        - (1/(m(m-1))) * sum_{j!=j'} ||Y_j - Y_j'||

**Why this rather than the ordinary V-statistic.** The V-statistic includes n
zero self-distances and has expectation

    E(P,Q) + (1/n) E||X-X'|| + (1/m) E||Y-Y'||

so its upward bias grows as n falls AND depends on each predicted population's
dispersion. Equal cell counts across arms therefore do NOT make the bias
cancel. The three arms differ measurably in predicted dispersion - mean
|predicted delta| relative to true is 0.83, 0.73 and 0.74 for ddim,
fm_gaussian and fm_control_ot - so the lower-dispersion flow-matching arms
would receive smaller upward bias, favouring them at precisely the conditions
with fewest cells. This was the reasoning error in the earlier justification,
which claimed cancellation.

**Signed values are retained.** The unbiased statistic can be slightly negative
at finite sample size. It is not clamped to zero and no square root is taken.
All conditions have at least two cells in both samples, so the estimator is
defined throughout.

### Supplementary, non-gated, fixed now

1. **Cell-count quartile stratification.** Conditions split into quartiles by
   n_c alone; the unweighted arm contrast reported within each. Consistent arm
   ordering across quartiles is the robustness result.
2. **Estimator sensitivity.** The V-statistic reported alongside the
   U-statistic, with the U-statistic identified as primary.
3. **Count-dependence diagnostic.** Per-condition arm difference against n_c,
   with a descriptive Spearman correlation. Diagnostic only, not a test.

Thresholded summaries at n >= 20 and n >= 50 may be reported as exploratory.
No threshold may be selected after seeing which is favourable.

### Inferential boundary

The retained cells improve estimation of each condition's discrepancy. They are
NOT independent replicates for the architecture comparison. The chain remains:
energy statistic per condition, unweighted mean across 443 conditions, one P2
scalar per checkpoint, seed-level contrasts at the pre-registered 97.5% level,
and the section 4 joint decision rule. Cells are not bootstrapped as if the
study had thousands of model replicates.
