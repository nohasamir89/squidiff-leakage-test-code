# Pre-registration: Step 6b, confirmation of the flow-matching result

**Written and committed BEFORE any Step 6b model is trained.** No checkpoint
exists on `drug_split_0` or `drug_split_2` for either arm under any
flow-matching objective.

Author: Noha Samir Ismail. Date: 2026-08-19.

## 1. What is being confirmed

Step 6 found, on the development split `drug_split_1`, under the joint rule of
its section 4:

- **B minus A PARTIAL.** P1 raw DE-overlap Jaccard **+0.03102**, 97.5% CI
  [+0.02152, +0.04053], complete seed separation, exhaustive permutation p at
  its 0.0079 floor, 1.8 times the pre-registered MDE. P2 energy distance
  UNRESOLVED at -0.00264, CI [-0.01146, +0.00618].
- **C minus B PARTIAL.** P1 unresolved at +0.00009. P2 **+0.09695**, CI
  [+0.07568, +0.11822], against the control-cell source.

Step 6b confirms the **B minus A P1 finding** on two held-out splits. Nothing
else is under confirmation: P2 was unresolved for that contrast, so there is no
P2 finding to confirm, and the C minus B result is a negative that does not
require replication before being reported as such.

## 2. Arms

Two arms, five independently trained checkpoints each, on **each** of
`drug_split_0` and `drug_split_2`. Twenty checkpoints total, all trained fresh
under one commit.

| arm | objective | source | sampler |
|---|---|---|---|
| **A. DDIM** | epsilon-prediction | Gaussian | DDIM, 1000 steps |
| **B. FM-gauss** | conditional flow matching | Gaussian | Euler ODE, 20 NFE |

The network is identical in both: 58,178,180 parameters, semantic adaLN,
unchanged from Steps 4 through 6.

**`fm_control_ot` is NOT carried forward.** Its Step 6 result was P1 unresolved
and P2 strongly adverse, so confirming it would spend held-out splits on a
resolved negative.

**The sampler settings replicate the Step 6 gated contrast exactly**: flow
matching at 20 NFE against DDIM at 1000. A post-hoc NFE sweep (section 5 below)
found other configurations, but the confirmation must test the comparison that
was gated, not one selected afterwards.

## 3. Disclosure: this is a second use of these splits

`drug_split_0` and `drug_split_2` are NOT untouched. They carried the Step 4
replication: 20 checkpoints trained and evaluated, producing the confirmed
adaLN result. This is their **second** use.

They did not inform any Step 6 or Step 6b design decision, with one
qualification: semantic adaLN was chosen as the shared backbone because it is
the best-performing architecture to date, and that judgement drew partly on
these splits. The information path is small but not zero.

No further confirmation split exists in the current six-split design. If Step 6b
resolves, there is no third split available and any further confirmation would
require a new data partition.

## 4. Endpoints and analysis

**Primary, and the only gated quantity:** raw DE-overlap Jaccard, arm B minus
arm A, arithmetic mean of condition-level values, assessed **separately** on
each split. Unpaired two-sample Welch, five per arm, Welch-Satterthwaite df.
Exhaustive two-sided 5-vs-5 permutation over 252 allocations as support, floor
2/252 = 0.0079.

**Level.** 97.5% per split, matching Step 6 and holding the family-wise rate
across the two splits at approximately 5%.

**Co-primary P2 is NOT gated here.** P2 was unresolved for B minus A in Step 6,
so there is no distributional finding to confirm. Energy distance is computed
on both splits under the amendment-13 estimator and **reported**, because a
resolved adverse P2 on a confirmation split would be material even though it is
not gated. It cannot alter the section 5 verdict.

**Chance reference** c_s is recomputed per split as the exact hypergeometric
expectation E[I/(2K-I)] over that split's ever-active gene count. drug_split_0
and drug_split_2 differ from drug_split_1 in condition count (444 and 455
against 443) and drug count (37 and 38 against 37).

**Sensitivity, stated in advance.** At the larger observed within-arm SD of
0.0075, the SE of a difference is 0.00474, the 97.5% half-width is 0.01305, and
the MDE at 80% power is **0.01727**. The Step 6 effect of +0.03102 is 1.8 times
that MDE, so this design is adequately powered for the effect being confirmed.

**Missingness.** Every condition-level Jaccard must be finite for all ten
checkpoints of a split. The analysis asserts identical condition labels and
order within each split and ABORTS on any non-finite value. Conditions and
seeds are never dropped.

## 5. Outcome categories, fixed now

Assessed on the primary endpoint, per split, then combined. The first
applicable category applies.

1. **FULL CONFIRMATION** - the B minus A interval lies entirely above zero on
   BOTH splits.
2. **DIRECTIONAL CONTRADICTION** - the interval lies entirely below zero on
   either split. Reported as such; not folded into partial confirmation.
3. **PARTIAL CONFIRMATION** - resolved above zero on one split, includes zero
   on the other. Reported as such, with no pooling across splits to rescue it.
4. **NO CONFIRMATION** - includes zero on both splits.

Categories 3 and 4 do NOT establish that flow matching fails to improve
DE-overlap; they establish that this design did not resolve it on held-out
splits. No equivalence margin is justified and none may be claimed.

**No arm, split, NFE or endpoint may be added after seeing these results.**

## 6. Disclosure: what the post-hoc sweep found, and why it is not confirmed

After the Step 6 gated result, a step-count sweep was run. Its flow-matching
half was pre-registered in Step 6 section 3; its **DDIM half was not** and is
recorded as post-hoc.

| configuration | P1 | P2 |
|---|---|---|
| ddim 1000 NFE | 0.55477 | 0.18886 |
| ddim 250 | 0.55222 | 0.18942 |
| ddim 50 | 0.53798 | 0.19361 |
| ddim 20 | 0.51477 | 0.20351 |
| **fm 50** | **0.61000** | **0.17951** |
| fm 20 | 0.58579 | 0.18622 |
| fm 10 | 0.52758 | 0.20213 |
| fm 5 | 0.46445 | 0.24705 |

Three observations, all secondary and non-gated:

1. DDIM at 1000 steps is its BEST configuration; it loses 0.040 on P1 by 20
   steps. The Step 6 gated contrast therefore compared flow matching against
   DDIM at its strongest, which is the conservative direction.
2. At matched 20 NFE the P1 gap is +0.07102, more than double the gated
   +0.03102. The advantage is not a sampling-budget artefact.
3. `fm_gaussian` at 50 NFE is the best configuration observed anywhere in this
   project on both endpoints, at one twentieth of DDIM's sampling cost.

**Step 6b confirms the 20-NFE comparison, not the 50-NFE one**, because 50 was
identified by inspecting a sweep. The 50-NFE configuration will be evaluated on
the confirmation splits and reported as EXPLORATORY, clearly labelled as
post-hoc selected, and cannot contribute to the section 5 verdict. Promoting it
to the gated comparison would be selecting the configuration after seeing which
performed best.

**Flow matching degrades sharply below 20 NFE** on this data with Euler
integration: 0.5276 at 10 steps and 0.4644 at 5. The few-step claims in the
flow-matching literature do not transfer here without path straightening or a
higher-order solver, neither of which was tested.

## 7. Verification required before training

- **Arm A must reproduce the Step 4 replication range on each split.** Step 4
  recorded adaLN at 0.5453-0.5806 on drug_split_0 and 0.5529-0.5620 on
  drug_split_2. A retrained arm A falling outside its split's range means the
  training path changed and must be explained before the comparison is read.
- Both arms constructed at exactly 58,178,180 parameters.
- Resolved flags asserted before training: `use_adaln` Boolean True,
  `fm_source` string `none` or `gaussian`.
- Arm identity is NOT recoverable from the weights - the arms are
  architecturally identical - so an objective sidecar is written beside each
  checkpoint, as in Step 6.
- Evaluation sources byte-identical to the Step 6 commit, verified by hash.
- Node recorded per evaluation. cn270 differs from cn274 by 4.232e-06 on
  Protocol A and 6.080e-06 on Protocol B, demonstrated in CORRECTION_cn270.md;
  cross-node comparisons must be labelled.

## 8. What this cannot establish

- **Whether flow matching beats ridge regression.** Amendment 12 of the Step 4
  pre-registration found no evidence that adaLN exceeds ridge on these splits,
  and ridge is reported alongside every arm here without being the criterion.
- Whether a different path, solver, or NFE would differ. One path, one solver,
  one gated NFE.
- Anything about the control-cell source, which is not carried forward.
- Whether the improvement holds beyond sci-Plex3 or beyond a 200-gene panel.
- The absolute quality of either arm. Both are expected to remain below the
  drug-agnostic expression-only heuristic on DE-overlap.
