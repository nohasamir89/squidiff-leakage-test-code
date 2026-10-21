# Pre-registration: Step 11, confirming the 50-NFE configuration

**NOT BLIND.** The value being confirmed is reproduced in section 2. What has
not been computed is any 50-NFE or 100-NFE evaluation on `drug_split_0` or
`drug_split_2`, for either arm.

Author: Noha Samir Ismail. Date: 2026-09-14.

## 1. Question

Step 6 gated its comparison at **20 NFE**, fixed before any run. The sweep over
{1, 5, 10, 20, 50} sampling steps was also pre-registered, as a secondary
endpoint. That sweep found flow matching scoring higher at 50 steps than at the
gated 20.

**Does the 50-NFE advantage replicate on held-out compounds?**

## 2. What was observed, and its status

DE-overlap Jaccard on `drug_split_1`, the development split, from the
pre-registered Step 6 sweep:

| NFE | flow matching | DDIM |
|---|---|---|
| 5 | 0.4644 | - |
| 10 | 0.5276 | - |
| 20 | **0.5858** (gated) | 0.5148 |
| 50 | **0.6100** | 0.5380 |
| 250 | - | 0.5522 |
| 1000 | - | **0.5548** (gated) |

**Status of 0.6100.** It comes from a sweep that was planned in advance, so it
is not a post-hoc discovery in origin. But 20 NFE was the gated configuration,
fixed before any data existed, and 50 was not. Carrying a sweep value forward as
a headline without confirmation is the pattern that pre-registration exists to
prevent, which is why this document exists.

**The DDIM half of that sweep WAS post-hoc.** It was added after seeing the flow
matching values. DDIM at 1000 is its best configuration and remains the
comparator here.

## 3. The intervention

**No new training.** Step 6b's checkpoints already exist: 20 models, two arms,
five seeds each, on `drug_split_0` and `drug_split_2`, trained under commit
`75fe161` and evaluated only at 20 NFE.

They are re-evaluated at 50 NFE. Nothing about the models changes; only the
number of ODE steps at sampling time.

| | |
|---|---|
| arms | DDIM, Gaussian-source flow matching |
| splits | `drug_split_0`, `drug_split_2` (both held out, both used twice before) |
| seeds | 5 per arm per split |
| evaluations | 20 |

**DDIM is evaluated at 1000 NFE**, its own best setting, not at 50. Evaluating
DDIM at 50 would compare flow matching at its best against DDIM at a crippled
setting. The Step 6 sweep showed DDIM losing 0.040 between 1000 and 20 steps.

## 4. Secondary and EXPLORATORY: 100 NFE

100 NFE was **not** in the Step 6 sweep, whose ceiling of 50 was fixed in
advance. It is added here, and the reason is recorded before any run: the sweep
is monotonic up to its boundary and 50 was the endpoint, so whether the gain
continues is unknown rather than answered.

**This is an extension of a search past the boundary that was set, prompted by
the endpoint looking favourable.** That is exactly the pattern pre-registration
guards against. It is therefore labelled EXPLORATORY, reported separately from
the confirmation, and cannot be used to revise the gated claim.

Diminishing returns are expected: gains across 5, 10, 20, 50 are +0.063, +0.058,
+0.024. An ODE solver converges toward the true trajectory rather than improving
without limit.

## 5. Endpoints

**Primary, gated: DE-overlap Jaccard**, flow matching at 50 NFE minus DDIM at
1000 NFE, per split, with a drug-clustered percentile bootstrap over each
split's compounds, 5000 replicates, `RandomState(0)`, at the **97.5% level per
split** — matching Step 6b exactly so the two are directly comparable.

DE-overlap is gated rather than DPC-50 because that is the endpoint on which
flow matching improved and confirmed in Step 6b, and the endpoint the sweep
concerns. DPC-50 is reported and not gated.

**Secondary, reported and not gated:** DPC-50; energy distance under the
amendment-13 U-statistic; the 100-NFE arm from section 4; and the 50-NFE minus
20-NFE difference within flow matching, which isolates the sampling change from
the objective change.

**Sensitivity.** Step 6b's minimum detectable effect at this design was 0.0173.
The observed 50-versus-20 difference on the development split is +0.0242, so the
design has power for an effect of that size if it replicates. The 50-NFE minus
DDIM difference on development is +0.0552.

## 6. Outcome categories, fixed now

1. **50 NFE CONFIRMS** — flow matching at 50 beats DDIM at 1000 with intervals
   above zero on BOTH held-out splits. The configuration becomes reportable as a
   confirmed result alongside the 20-NFE finding.
2. **PARTIAL** — resolved on one split only. Reported per split with no pooling,
   and not described as confirmed.
3. **NOT CONFIRMED** — unresolved on both. The 0.6100 value stands as a
   development-split observation and nothing more.
4. **REVERSED** — resolved against flow matching on either split. Reported as
   observed.

**On the 50-versus-20 comparison within flow matching:** if that difference does
not resolve on either held-out split, the sampling-step advantage is a
development-split artefact, whatever the DDIM comparison shows.

## 7. Expectation, recorded in advance

**I expect CONFIRMS.** Step 6b's 20-NFE contrast replicated on both splits with
effects within 0.004 of the development value, and the 50-NFE advantage is
larger than the 20-NFE one. The mechanism is also straightforward: more ODE
steps approximate the learned trajectory more closely.

**I expect the 100-NFE gain to be small**, under +0.01, on the diminishing-
returns trend in section 4.

If 50 NFE does NOT confirm while 20 NFE did, that would be surprising and would
need explaining rather than reporting.

## 8. What this cannot establish

- **That 50 is optimal.** The sweep stopped at 50 by design; 100 is exploratory
  and anything beyond is untested.
- **A fair DDIM comparison at matched cost.** DDIM is evaluated at 1000 NFE, its
  best setting. At matched 20 NFE the Step 6 gap was +0.071, larger still, but
  that is a different comparison and is not gated here.
- **Anything about training.** No model is retrained; only the sampler changes.
- **Independence from Steps 4 and 6b.** Splits 0 and 2 have now been used three
  times — Step 4 replication, Step 6b confirmation, and this. Each use erodes
  their value as held-out data, and no unspent split remains.
- **Anything about DPC-50 as a gated claim**, on which every model in this
  thesis loses to a linear baseline.
