# Pre-registration: Step 8, is the bottleneck the data regime?

**NOT BLIND, and section 2 says exactly what was already seen.** An exploratory
comparison across all three splits preceded this document. What has not been
computed is the learning curve, any architecture beyond one feedforward family,
or any inferential interval on these differences.

Author: Noha Samir Ismail. Date: 2026-08-21.

## 1. Question

Ahlmann-Eltze, Huber and Anders (Nature Methods 2025) found that deep
perturbation models do not outperform linear baselines. Steps 1 through 7 of
this thesis reproduce that on sci-Plex3. **This step asks WHY.**

The hypothesis: **the effective sample size is the number of held-out
COMPOUNDS, roughly 37, not the 443 conditions and not the 290,888 cells.** At
that scale a penalised linear model beats a neural network fitting the same
chemistry-to-response map, and no architectural change repairs it.

## 2. What was already observed

Identical features, identical compound-grouped folds, identical target:

| split | ridge | MLP-64 | MLP-256 | MLP-1024 |
|---|---|---|---|---|
| drug_split_0 | 0.2473 | 0.0151 | 0.0282 | 0.0506 |
| drug_split_1 | 0.1883 | 0.0218 | 0.0442 | 0.0619 |
| drug_split_2 | 0.0816 | 0.0112 | 0.0132 | 0.0370 |

On drug_split_1, twelve regularisation settings spanning weight decay 0.01 to
10 and 25 to 400 epochs all produced DPC-50 between 0.027 and 0.060.

**Limits of that evidence.** One feedforward family, one optimiser, three
widths, three to five seeds, no intervals, and no test of the mechanism. It
shows THIS network family loses; it does not show networks lose, and it does
not show WHY.

## 3. Primary: the learning curve

For each split, subsample the training compounds to
**n in {8, 16, 24, 32, all}** and fit both models on each subset, evaluating on
the same held-out folds. Ten random compound subsets per n, drawn with
`RandomState(0)`, identical subsets for both models.

**Primary quantity:** the ridge-minus-MLP gap in DPC-50 as a function of n.

**Why this and not another architecture comparison.** If the deficit SHRINKS as
compounds increase, the sample-size explanation is supported and the trend
bounds where the two would meet. If it is FLAT, the explanation is wrong and
the MLP's failure is something else. This tests the mechanism; adding
architectures only broadens the observation.

**Gated criterion:** the mechanism is supported if the ridge-minus-MLP gap at
n = 8 exceeds the gap at n = all, with 95% percentile intervals from a
drug-clustered bootstrap that exclude zero, on at least two of three splits.

## 4. Secondary: architecture breadth

Same features and folds, at full n only, reported without gating:

- feedforward MLP, widths 64 / 256 / 1024 (the exploratory family)
- residual MLP, two blocks, width 256
- wide-shallow: single hidden layer, width 4096
- linear network with no nonlinearity, as a sanity check that the harness can
  reproduce ridge-like behaviour

That last arm matters: if a linear network trained by AdamW cannot match
ridge's closed-form solution, the deficit is partly an OPTIMISATION artefact
rather than a statement about model class, and the whole finding weakens. It is
included precisely because it can undermine the hypothesis.

## 5. Secondary: seeds and intervals

Five seeds per configuration, and a drug-clustered percentile bootstrap over
compounds, 5000 replicates, `RandomState(0)` - matching every other interval in
this thesis. The exploratory run reported point estimates only.

## 6. Outcome categories, fixed now

On the primary learning curve:

1. **SAMPLE-SIZE MECHANISM SUPPORTED** - the gap narrows with n on at least two
   splits, intervals excluding zero. The finding becomes: deep models lose here
   because the compound count is small, with a measured trend.
2. **GAP IS FLAT** - no reliable narrowing. The sample-size explanation is NOT
   supported, and the MLP's deficit needs another explanation. This must be
   reported as a failed hypothesis, not quietly reframed.
3. **GAP WIDENS WITH n** - reported as observed; would suggest the MLP
   overfits more as data grows, which would be surprising and worth stating.
4. **MIXED** - resolves differently across splits. Reported per split with no
   pooling.

On the secondary linear-network check:

- **If the linear network matches ridge** (within 0.02), the harness is sound
  and the nonlinear arms' deficit is about model class.
- **If it does NOT match ridge**, the deficit is confounded with optimisation
  and every conclusion in section 6 is downgraded to "under this training
  procedure". This is stated in advance so the outcome cannot be explained away.

## 7. Expectation, recorded in advance

**The gap is expected to narrow with n**, because the hypothesis predicts it.
**The linear network is expected to approach ridge**, since it is fitting the
same function class by gradient descent.

If the linear network does NOT approach ridge, I expect that to be the more
informative result, and it would mean the exploratory finding was substantially
an optimisation artefact.

## 8. What this cannot establish

- That networks cannot work on perturbation prediction generally. Four
  architecture families on one dataset, one panel, one metric.
- Where the curves would cross. Extrapolating beyond 37 compounds is not
  supported by data that stops at 37.
- Anything about DE-overlap, on which flow matching DID improve and replicate.
  This is a DPC-50 analysis.
- That more compounds would fix Squidiff. The learning curve concerns a
  supervised MLP on fixed features, not the full generative model.
- Anything causal about the published model's design.
