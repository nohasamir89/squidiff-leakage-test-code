# Pre-registration: Step 9, conditioning on mechanism instead of chemistry

**NOT BLIND for the ridge diagnostics, which are reproduced in section 2.
BLIND for everything about the model**: no Squidiff or flow-matching checkpoint
has ever been trained on a pathway-conditioned input, and no such value exists
in any artifact.

Author: Noha Samir Ismail. Date: 2026-09-09.

## 1. Question

Every architectural intervention in this thesis - adaLN in Step 4, gene mixing
in Step 5, flow matching in Step 6 - operated downstream of a fixed drug
representation: a 1024-bit Morgan fingerprint, unchanged from the published
model.

Ridge diagnostics now indicate that representation is the weaker of two
available options. **Does conditioning the model on mechanism of action instead
of chemical structure improve prediction?**

## 2. What the diagnostics showed

DPC-50, ridge, compound-grouped folds, each feature set at its own best alpha:

| split | pathway x line | fingerprint x line | target x line |
|---|---|---|---|
| drug_split_0 | **0.2649** | 0.2473 | 0.2639 |
| drug_split_1 | **0.2133** | 0.1893 | 0.1696 |
| drug_split_2 | **0.1287** | 0.0823 | 0.1012 |

Pathway wins on all three with roughly 20 categories against 4,099 features.
The margin is largest on drug_split_2, where every method performs worst.

**Disclosure.** Nine feature sets were tried on drug_split_1 and the best was
carried forward before the other two splits were touched. That is selection.
The replication is reported above, but the family was chosen after looking.

**These are ridge results.** Nothing here establishes that the same swap helps
the generative model. This project has repeatedly shown that improvements in one
place do not transfer: gene mixing helped nothing, the capacity control
reproduced the whole adaLN gain, and flow matching improved DE-overlap while
losing on DPC-50.

## 3. The intervention

`Drug_dose_encoder` currently returns a 1024-bit fingerprint concatenated with
dose. Step 9 replaces the fingerprint block with a one-hot `pathway_level_2`
encoding, dose handling unchanged.

**Arms**, five seeds each, one commit, `drug_split_1`:

| arm | drug representation | width |
|---|---|---|
| **A. fingerprint** | Morgan 1024 + dose | as published |
| **B. pathway** | pathway_level_2 one-hot + dose | ~20 + dose |

Both arms use **Gaussian-source flow matching at 20 NFE**, the configuration
confirmed in Step 6b. The objective, sampler, backbone and every hyperparameter
are identical; only the drug encoding differs.

Arm A is retrained rather than reused, so both arms share a commit and the
comparison carries no cross-revision caveat.

**Parameter count will differ**, because the encoder's input width changes.
That is intrinsic to the intervention and is not controlled; a capacity-matched
control is NOT included, and the limitation is recorded in section 7.

## 4. Endpoints

**Primary, and the only gated quantity: DPC-50**, median of paired
per-condition differences, arm B minus arm A, with a drug-clustered percentile
bootstrap over the split's 37 compounds, 5000 replicates, `RandomState(0)`.

DPC-50 is chosen deliberately over DE-overlap. The diagnostics that motivate
this step are DPC-50 results, and DPC-50 is the metric on which every model in
this thesis loses to ridge. DE-overlap is where flow matching already succeeds,
and gating there would test the wrong thing.

The five checkpoints per arm are pooled per condition by arithmetic mean before
differencing, matching Step 6c and amendment 12 so the three are comparable.

**Secondary, reported and not gated:** raw DE-overlap Jaccard, energy distance
under the amendment-13 U-statistic, and both arms against the pathway-ridge
baseline of 0.2133.

**Sensitivity.** No prior SD exists for a DPC-50 arm difference under this
estimator, so power is UNKNOWN in advance and is disclosed rather than
estimated. An unresolved result will not be described as evidence of no effect.

## 5. Outcome categories, fixed now

1. **PATHWAY CONDITIONING IMPROVES DPC-50** - the interval lies entirely above
   zero. The drug representation was a bottleneck in the model, not only in
   ridge.
2. **FINGERPRINT CONDITIONING IS BETTER** - interval entirely below zero.
   Reported as observed; would mean the ridge finding does not transfer.
3. **UNRESOLVED** - interval includes zero. NOT equivalence. Given this
   project's history, this is the single most likely outcome and is recorded as
   such here rather than explained afterwards.

**Neither arm is expected to reach the pathway-ridge baseline of 0.2133.** If
one does, that would be the first model in this thesis to match a linear
baseline on DPC-50, and it would warrant its own confirmation on splits 0 and 2
under a separate pre-registration.

## 6. Expectation, recorded in advance

**I expect UNRESOLVED or a small positive effect.** The ridge margin is +0.024
on this split, and improvements have repeatedly failed to transfer from
diagnostic to model in this project. A large positive effect would be
surprising and would need its own confirmation before being reported as such.

## 7. What this cannot establish

- **Capacity is not controlled.** Arm B's encoder is far narrower, so a null
  result confounds representation with capacity, and a positive result could in
  principle be a regularisation effect. A capacity-matched arm is the obvious
  follow-up and is not run here.
- **Pathway labels are coarse and shared.** Roughly 20 categories over 37
  compounds, so a held-out drug's class is almost always represented in
  training. That is legitimate transfer but it is a different kind of
  generalisation from chemical similarity, and it would not extend to a
  compound whose mechanism is unannotated or novel.
- **One split.** Confirmation on splits 0 and 2 requires a separate
  pre-registration, and those splits have already been used twice.
- Nothing about DE-overlap, on which flow matching already improves and
  replicates.
- Nothing about whether a learned mechanistic embedding would do better than a
  one-hot label.

---

## 8. Amendment, written before any Step 9 training

Verification of the encoder revealed something section 3 did not anticipate.
Dose enters MULTIPLICATIVELY in `Drug_dose_encoder`:
`fingerprint * log10(dose + 1)`. Reproducing that faithfully for a one-hot
pathway means arm B's entire drug representation is **a single non-zero scalar**
in a 55-wide vector, against 13 non-zero bits for the fingerprint.

That is a sharper reduction than the ridge diagnostic faced. Ridge received
pathway and cell line as separate blocks with independent coefficients plus an
explicit crossing; the model receives one scalar and must construct z_sem from
it, with cell-line information arriving separately through control expression.

**A null result for arm B would therefore confound the REPRESENTATION with the
SPARSITY of its presentation**, which is not what section 1 asks.

**A third arm is added, before any training:**

| arm | drug representation | width |
|---|---|---|
| A. fingerprint | Morgan 1024, dose-scaled | 1024 |
| B. pathway | pathway one-hot, dose-scaled | 55 |
| **C. both** | **fingerprint concatenated with pathway** | **1079** |

Arm C keeps the published input intact and ADDS the mechanistic signal, so it
tests whether pathway information helps at all without removing anything. It is
the arm most likely to isolate the question, and it is added now rather than
after seeing B fail.

**The gated contrast becomes C minus A**, with B minus A reported as secondary.
Section 5's outcome categories apply to C minus A; B minus A is reported with
the sparsity caveat attached and cannot on its own establish that mechanism
conditioning fails.

### Measured parameter counts, before training

| arm | drug_dimension | parameters | difference from A |
|---|---|---|---|
| A fingerprint | 1024 | 58,178,180 | - |
| B pathway | 55 | 56,193,668 | -1,984,512 (-3.41%) |
| C both | 1079 | 58,290,820 | +112,640 (+0.19%) |

The gated contrast C minus A differs by **0.19%**, so capacity is very nearly
matched there and section 7's confound applies chiefly to the secondary B minus
A contrast, where the encoder is 3.41% smaller. This was measured before any
Step 9 training and is recorded here rather than reported afterwards.

---

## 9. Amendment, 2026-09-09: two training commits, arm A provably unaffected

Section 3 requires all arms to train under one commit. They did not.

- Arm A (fingerprint), five seeds, commit `6b5b76a`
- Arms B and C, five seeds each, commit `11b55f7`

**Why.** The first submission set `args['drug_dimension']` at line 107 while
`create_model_and_diffusion` runs at line 74, so every model was constructed
with a 1224-wide first layer regardless of encoding. Arm A's data matched and
trained normally; arms B and C died at the first forward pass with
`mat1 and mat2 shapes cannot be multiplied`. The assignment was moved above
model construction and the ten affected runs repeated. Arm A was not retrained.

**Why arm A is unaffected, and this is stronger than an unused-code-path
argument.** For `drug_encoding='fingerprint'` the computed `drug_dimension` is
1024, which is exactly the default the model would have taken before the fix.
The two values are IDENTICAL, not merely on different paths, so the reordering
cannot have changed arm A's construction. Confirmed by the state-dict counts:
each arm has exactly one size across its five seeds - 58,186,374 / 56,201,862 /
58,299,014 - and the logged first-layer widths are 1224, 255 and 1279, one per
arm.

**Also recorded.** The pre-training guard block written into `train_step9.sh`,
which would have caught this before any GPU time, did not execute. Two safety
nets were bypassed: that guard by not running, and a constructed-width
assertion by not existing. A width check against
`gene_size + drug_dimension` should be standard in any future arm that changes
an input dimension.

**All fifteen runs trained on cn270.** Evaluation must pin cn274 to stay
comparable with Steps 6 and 6b, per `CORRECTION_cn270.md`.
