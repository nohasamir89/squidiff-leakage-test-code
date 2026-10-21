# Pre-registration: Step 5, gene-level mixing modules

**Written and committed BEFORE any Step 5 model is trained.** No Step 5
checkpoint exists. This is a blind prospective pre-registration.

Author: Noha Samir Ismail. Date: 2026-08-05.

## 1. Question

Step 4 established that adaptive modulation improves DE-overlap, replicated on
two held-out splits. Step 4b established that the gain is NOT attributable to
conditioning that modulation on the semantic latent: a parameter-matched
timestep-driven arm reproduced it, and the incremental contrast was UNRESOLVED
(-0.0015, CI [-0.0123, +0.0093]).

Step 5 asks a different question: does **input-dependent gene-gene mixing** add
anything beyond **static learned mixing of the same size**?

## 2. Where the module attaches, and why it has to

`MLPModel` maps 200 genes -> 2048 via a DENSE `input_layer`, runs three
`MLPBlock`s entirely in 2048-d, then maps 2048 -> 200 via `output_layer`. No
hidden unit corresponds to a gene, and 2048 does not factor into 200 tokens, so
there is no point inside the block stack where gene tokens exist.

The modules therefore attach AFTER `output_layer`, as a residual refinement of
the predicted 200-gene profile. This is a constraint of the architecture, not a
preference.

## 3. Arms

Four arms, five seeds each, **all trained fresh under one commit** on
`drug_split_1`. Existing `seedvar/` and `adaln/` checkpoints are NOT reused,
even though they load with `strict=True`, because they predate the
`gene_module` plumbing and reusing them would reintroduce the cross-revision
limitation recorded in amendment 13.

| arm | flag | parameters | vs adaLN |
|---|---|---|---|
| base | `use_adaln` off | 54,221,060 | - |
| semantic adaLN | `use_adaln True` | 58,178,180 | reference |
| + GeneAttention | `gene_module attention` | 61,434,501 | +3,256,321 (+5.597%) |
| + GeneMLP | `gene_module mlp` | 61,434,500 | +3,256,320 (+5.597%) |

Arm mismatch: **1 parameter**. Counts are MEASURED from constructed models, not
estimated. For scale, adaLN itself added 3,957,120 (+7.3%).

**GeneAttention**: 200 genes as tokens, learned per-gene embedding (required -
without it the module is permutation-equivariant and cannot represent gene
identity), embed_dim 512, 16 heads, FFN expansion 4, zero-initialised output
projection, residual.

**GeneMLP**: 200 -> 8120 -> 200 with GELU, zero-initialised second layer,
residual. The bottleneck 8120 solves 401b + 200 = 3,256,321.

**What the control controls for.** Both allow all-to-all gene interaction and
are parameter matched. The attention module's mixing weights are INPUT
DEPENDENT; the MLP's are fixed once trained. The contrast isolates dynamic
content-dependent mixing from static learned mixing, NOT interaction versus no
interaction.

## 4. Endpoint and analysis

**Primary, and the only gated quantity:** raw DE-overlap Jaccard,
**GeneAttention minus GeneMLP**, on `drug_split_1`.

Jaccard is set overlap between the predicted and true top-50 DEG sets. With
both sets of size 50 the union is 100 - I, so J = I / (100 - I). It is NOT B-C,
which is a difference of Pearson correlations, and NOT DPC-50, which is a
correlation of deltas. These have been conflated repeatedly in planning
documents; the gated endpoint is Jaccard.

**Seed summary:** ARITHMETIC MEAN of condition-level raw Jaccard over the 443
eligible conditions, matching section 4 of the replication pre-registration.
Not a median.

**Test:** unpaired two-sample Welch 95% CI, five checkpoints per arm, with df
from the Welch-Satterthwaite formula (approximately 8 when arm variances are
similar). Arms are UNPAIRED: a shared numeric seed does not align two
independently trained models once the parameterisations differ.

**Support:** exhaustive two-sided 5-vs-5 label permutation over all C(10,5) =
252 allocations. Minimum attainable p is 2/252 = 0.0079.

**Chance reference:** c_s recomputed for `drug_split_1` as the exact
hypergeometric expectation E[I/(2K-I)] over its ever-active gene set. At
M_s = 147 this is 0.206228. The correction is positive affine, so the gate is
identical on raw or corrected Jaccard; raw is used for the test and both are
reported.

**Sensitivity.** Measured within-arm SDs on this endpoint are 0.0071 to 0.0088.
At SD = 0.0077, SE of the difference is 0.0049, the 95% half-width is 0.0113,
and the MDE at 80% power is **0.0156**. A difference smaller than roughly 0.016
will not be resolved by this design. This is stated in advance so that an
unresolved result is not later described as evidence of no effect.

**Missingness.** Every condition-level Jaccard must be finite for all 20
checkpoints. The analysis asserts identical condition labels and order across
all arms and ABORTS on any non-finite value. Conditions and seeds are never
dropped post hoc.

## 5. Secondary, not gated

Reported with intervals, unadjusted for multiplicity, and explicitly
exploratory:

- GeneAttention minus semantic adaLN
- GeneMLP minus semantic adaLN
- semantic adaLN minus base
- B-C and DPC-50 for all four arms

The base arm exists so that "does gene-level mixing help at all" has a
same-commit answer. Without it, a null primary contrast could not be
distinguished from two modules that both degrade the model.

## 6. Outcome categories, fixed now

Decided on the primary contrast, then qualified by the base comparisons. The
first applicable category applies.

1. **Both modules underperform semantic adaLN** - if both
   attention-minus-adaLN and MLP-minus-adaLN have 95% CIs entirely below zero,
   the primary contrast is reported but NOT interpreted as informative about
   attention: it would compare two interventions that both harm the model.
   Gene-level mixing after the output layer is reported as unhelpful at this
   scale.

2. **Attention exceeds MLP** - the attention-minus-MLP CI lies entirely above
   zero. Supports an incremental benefit from input-dependent mixing at matched
   capacity. Does NOT establish that either module improves on adaLN; that is
   read from the secondary contrasts.

3. **MLP exceeds attention** - CI entirely below zero. Reported as observed;
   would indicate the attention inductive bias is harmful here.

4. **UNRESOLVED** - the CI includes zero and category 1 does not apply. This is
   NOT equivalence. No equivalence margin is justified in advance, so none may
   be claimed afterwards; any future equivalence claim requires a prospectively
   defined margin and a procedure such as two one-sided tests.

No further arm may be introduced after seeing these results to rescue a
preferred reading.

## 7. Verification required before training

Already performed and recorded; re-checked at launch.

- Plumbing inert at `gene_module='none'`: adaLN constructs to exactly
  58,178,180 parameters, zero `gene_module` keys in its state dict, and
  existing checkpoints load with `strict=True`. Confirmed.
- Parameter counts MEASURED: 61,434,501 and 61,434,500, mismatch 1.
- Identity at initialisation: loading adaLN weights into either augmented model
  gives max abs output difference exactly 0.000e+00. Confirmed.
- Gene identity represented: permuting input genes does NOT simply permute the
  attention output. Confirmed. Without this the module is
  permutation-equivariant and the hypothesis is untestable.
- Flag type: `gene_module` is a STRING defaulting to `'none'`, never `None`.
  `train_squidiff.py` uses `type=type(value)`, so a `None` default would give
  `type=NoneType` and reject every value - the divergence behind the
  `bool("False")` fault.

Each run records before training: resolved flags, constructed parameter count,
number of `gene_module` tensors. After training: checkpoint hash and the
architecture detected from the saved state dict. All to `provenance/` with git
commit, node, GPU from torch, and data and script hashes.

Arm identity IS recoverable here, unlike Step 4b: the four architectures differ
in parameter count and state-dict keys, so no mode sidecar is needed.

## 8. Splits

Training and the gated analysis use `drug_split_1` only.

`drug_split_0` and `drug_split_2` are NOT used and NOT spent. They are reserved
for confirmation of any resolved finding, under a separate pre-registration.

Disclosure: they are not untouched. They carried the Step 4 replication, 20
checkpoints trained and evaluated, so a future use is a SECOND use. They did
not inform any Step 5 design decision, with one qualification: semantic adaLN
was chosen as the gating for all arms because it is the best-performing
architecture to date, and that judgement drew partly on those splits. The
information path is small but not zero.

## 9. What this cannot establish

- Whether gene-level mixing helps at another scale. Both modules add 5.6%; a
  null result is specific to that budget.
- Whether attaching before `input_layer` would differ. Post-output is forced by
  the architecture but is not the only conceivable position.
- Anything about generalisation. One split, and the development one.
- Whether either arm beats ridge, which amendment 12 found no evidence for.
- The absolute quality of any arm. All four are expected to stay below the
  expression-only heuristic on DE-overlap and below control-mean parity on B-C.
