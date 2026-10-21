# Pre-registration: Step 10, stochastic ChemicalVAE conditioning

**NOT BLIND for the ridge diagnostic, reproduced in section 2. BLIND for the
model**: no Squidiff or flow-matching checkpoint has been trained on a
ChemicalVAE input, and no such value exists in any artifact.

Author: Noha Samir Ismail. Date: 2026-09-13.

## 1. Hypothesis

PerturbNet (Yu et al., Mol Syst Biol 21:960-982, 2025) state a specific
rationale for encoding drugs with a variational autoencoder rather than a
deterministic fingerprint:

> "Given that a chemical screen experiment usually contains a limited number of
> distinct drug treatments, the use of stochastic representations of the drug
> treatments prevents possible model overfitting."

**The hypothesis: a stochastic ChemicalVAE embedding acts as data augmentation
and regularises training, so the model generalises better to unseen compounds
than it does with a fixed Morgan fingerprint.** Their training set has 152
compounds; ours has 109 after exclusions, so the regime their argument concerns
is ours.

This is the one claim the ridge diagnostic in section 2 cannot test. Ridge fits
once on a fixed feature matrix and cannot see a representation that varies
across epochs.

## 2. What the diagnostics showed, and why they do not settle it

DPC-50 under ridge with compound-held-out folds, each feature set at its own
best alpha:

| split | ChemicalVAE x line | fingerprint x line | pathway x line |
|---|---|---|---|
| drug_split_0 | 0.1247 | 0.2331 | 0.2555 |
| drug_split_1 | **0.0183** | **0.1769** | 0.2251 |
| drug_split_2 | 0.0498 | 0.0827 | 0.1534 |

**ChemicalVAE loses to the fingerprint on all three splits**, and to a
20-category pathway annotation by a wide margin. RDKit physicochemical
descriptors scored lower still (0.011 to 0.094). Four chemical representations
now order consistently: mechanism annotation, then Morgan bits, then
ChemicalVAE, then descriptors.

**Why this does not settle the hypothesis.** Each ridge fit used ONE embedding
per compound. The claim concerns a representation that is resampled during
training, which is a property of the optimisation, not of the feature matrix.
Ridge is silent on it.

**Why the prior is nonetheless unfavourable.** Step 9 tested the analogous
contrast: pathway annotation beat the fingerprint substantially under ridge
(+0.05) and produced +0.0023, CI [-0.0027, +0.0087], in the model. A
representation that HELPED ridge did not transfer. This one does not help ridge.

## 3. The intervention

A pool of **64 ChemicalVAE draws per compound**, precomputed with their
pretrained weights (`model_params_525.pt`, 249,455 ZINC molecules), their
`smiles_to_hot` tokeniser with their default 35-character list, and their
standardisation using their published `mu.npy` and `std.npy`. Latent dimension
196.

`__getitem__` draws one of the 64 at random on every access, so a cell receives
a different embedding each time it is seen.

**Measured variability, before training:** within-compound standard deviation is
0.1322 against 0.1060 between compounds - **12.5%**. Draws of one molecule are
distinguishable from each other and clearly closer to each other than to another
molecule. The mechanism exists and is not degenerate.

**DISCLOSED DEPARTURE from their code.** PerturbNet calls the VAE once per
training CELL and fits ridge once, so a per-cell draw and a pool are equivalent
there. For a model trained over many epochs they are not: a fixed per-cell draw
is noise added once and frozen, while a pool resampled at each access is the
augmentation their rationale describes. The pool is used deliberately, in
service of their stated mechanism rather than their implementation.

### Arms, five seeds each, one commit, drug_split_1

| arm | drug representation | width |
|---|---|---|
| **A. fingerprint** | Morgan 1024, dose-scaled | 1024 |
| **B. fp_chemvae** (gated) | fingerprint + ChemicalVAE, concatenated | 1220 |
| C. chemvae | ChemicalVAE alone | 196 |

All arms use Gaussian-source flow matching at 20 NFE, the Step 6b configuration.

**Arm A is retrained, not reused.** Step 9's fingerprint checkpoints trained on
112 compounds; this experiment trains on 109 (section 4). Reusing them would
compare arms trained on different data.

## 4. Exclusions, applied identically to every arm

183 of 188 sci-Plex compounds encode. The five that do not: two whose canonical
SMILES exceed 120 characters, two containing sodium (`a` is outside their
35-character list), and one `nan`. Their reported figure is 180 of 188.

On `drug_split_1`:

| | compounds | cells | unencodable |
|---|---|---|---|
| train | 112 | 172,060 | **3 compounds, 4,577 cells (2.7%)** |
| test | 37 | 55,546 | **0** |

**Training excludes those 3 compounds from ALL THREE ARMS**, including the
fingerprint control, so every arm sees identical data. The encoder raises on an
unmapped SMILES rather than returning zeros, because a zero vector is
indistinguishable from a zero-dose control and would corrupt training silently.

**The test split is unaffected.** All 443 conditions are retained, so the
endpoints remain directly comparable to Steps 6, 6b, 9.

## 5. Endpoints

**Primary, gated: DPC-50**, median of paired per-condition differences, arm B
minus arm A, drug-clustered percentile bootstrap over the split's 37 compounds,
5000 replicates, `RandomState(0)`. Five checkpoints per arm pooled per condition
by arithmetic mean before differencing, matching Step 6c and Step 9.

DPC-50 is gated rather than DE-overlap for the same reason as Step 9: it is the
metric the diagnostics concern and the one every model in this thesis loses on.

**Secondary, reported and not gated:** arm C minus arm A, with the caveat that
arm C removes the fingerprint so a null confounds the representation with the
removal; DE-overlap Jaccard; energy distance under the amendment-13 U-statistic.

**Sensitivity.** No prior SD exists for a DPC-50 arm difference under a
stochastic encoder, so power is UNKNOWN in advance and disclosed rather than
estimated. An unresolved result will not be described as evidence of no effect.

## 6. Outcome categories, fixed now

1. **STOCHASTIC CHEMICALVAE IMPROVES DPC-50** - the interval for B minus A lies
   entirely above zero. The augmentation mechanism operates, and a
   representation that loses under ridge nonetheless helps a model trained over
   many epochs. This would be the first result in this thesis where the ridge
   diagnostic and the model disagree in that direction.
2. **FINGERPRINT ALONE IS BETTER** - interval entirely below zero. Adding the
   embedding actively hurts.
3. **UNRESOLVED** - interval includes zero. NOT equivalence.
4. Arm C is interpreted only alongside B, never alone.

## 7. Expectation, recorded in advance

**UNRESOLVED.** ChemicalVAE scores 0.0183 against the fingerprint's 0.1769 under
ridge on this split, and Step 9's analogous contrast returned +0.0023 with an
interval containing zero for a representation that DID help ridge. Both point
the same way.

A resolved positive would be a genuinely surprising result and would warrant its
own confirmation on a held-out split before being reported as one.

## 8. What this cannot establish

- **That stochastic conditioning fails in general.** One pool size (64), one
  latent (196), one dataset, one split.
- **That their per-cell scheme would behave as the pool does.** The two differ,
  and only the pool is tested (section 3).
- **Anything about PerturbNet as a model.** Only its drug representation is
  borrowed; the cINN, scVI decoder and cell-state network are not involved.
- **Anything about the four excluded training compounds** or the five
  unencodable ones.
- **That a larger pool would behave the same.** 64 was fixed in advance with no
  principled justification beyond memory and coverage.
- Nothing about DE-overlap, on which flow matching already improves and
  replicates; this is a DPC-50 analysis by design.

### Measured parameter counts, before training

| arm | drug_dimension | first layer | parameters | vs control |
|---|---|---|---|---|
| A. fingerprint_s10 | 1024 | 1224 | 58,178,180 | - |
| B. fp_chemvae (gated) | 1220 | 1420 | 58,579,588 | +401,408 (+0.69%) |
| C. chemvae | 196 | 396 | 56,482,436 | -1,695,744 (-2.91%) |

The gated contrast differs by 0.69% in capacity, so section 8's confound applies
chiefly to the secondary arm C, which is 2.91% smaller. Measured before any
Step 10 training and recorded here rather than reported afterwards.

Arm A's count is identical to Step 6 and Step 9's fingerprint arms
(58,178,180), confirming the `fingerprint_s10` alias changes only which cells
are loaded, not how the model is built.

---

## 9. Amendment, written after training and before any evaluation

Section 3 specifies the stochastic encoder for training but does not say how it
behaves at EVALUATION. The encoder draws one of 64 pooled embeddings per call,
so an unspecified evaluation would give each condition one arbitrary draw and
the reported score would partly reflect which draws came up.

**Fixed now: predictions are averaged over 16 independent draws per condition,
then scored once.** The rationale: the stochastic encoding is a training-time
regulariser, and at inference the quantity of interest is the model's estimate,
not a single sample from it. Sixteen is fixed in advance with no principled
justification beyond variance reduction at acceptable cost.

Two alternatives were considered and rejected BEFORE any Step 10 evaluation
value existed: a single fixed-seed draw, which matches training but makes the
score a lottery; and the pool mean, which is deterministic but furthest from the
representation the model was trained on. Neither was computed.

Arm A is unaffected - the Morgan fingerprint is deterministic - so its
evaluation is identical to Steps 6, 6b and 9.

**Reported as a secondary diagnostic:** the spread across the 16 draws for arm B,
as a measure of how much the prediction depends on the sampled representation.
