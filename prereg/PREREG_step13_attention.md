# Pre-registration: Step 13, attention on the drug condition

**NOT BLIND.** This is a development-split step on `drug_split_1`, whose every
previous result (Steps 1–12) is known. Nothing below has been run: no attention
arm exists, no checkpoint, no vector. Confirmation of anything found here is a
separate pre-registration (Step 13b) on `drug_split_0` and `drug_split_2`, which
would be their fifth use.

Author: Noha Samir Ismail. Date: 2026-09-21.

## 1. Question

Every architecture change so far acted on the sampler (adaLN, Step 4), the
objective (flow matching, Step 6), the integration (50 NFE, Step 11) or the
sampling-time conditioning strength (guidance, Step 12). None changed how the
encoder *reads the drug*. The encoder is a three-layer MLP on the concatenation
`[control feature (200), fingerprint × log10(dose + 1) (1024)]`, so the drug and
the cell state are mixed additively by the first layer.

**Does an attention mechanism on the drug condition raise held-out-compound
DE-overlap beyond what the same number of extra parameters raises it?**

Three attention mechanisms are tested, each answering a different sub-question,
against the current encoder and against a parameter-matched non-attention control.

## 2. What is known, and its status

| fact | source |
|---|---|
| The flow-matching objective is met on held-out compounds (test/train loss ratio 1.06–1.11) | `test_loss_exploratory.md` |
| Ridge on the same fingerprint beats the model on DPC-50; ridge with drug × cell-line interaction features beats plain ridge: on `drug_split_1` paired median +0.0451 [+0.0245, +0.0720], marginal medians 0.1884 (interaction), 0.1155 (plain), 0.0924 (flow matching) | `leakage_results/step7_analysis.txt`, `step7_interaction_baseline.json` |
| The adaLN gain was not attributable to the drug signal: the time-only control matched it (0.558 vs 0.557; adaLN − time-only −0.002 [−0.012, +0.009]) | `control_analysis.json` (Step 4b) |
| Attention over genes collapsed (GeneAttention 0.267 against adaLN 0.557) | `step5_analysis.json` |
| Guidance at w = 2 raises DE-overlap on all three splits with a fidelity cost that resolved on `drug_split_0` (TRADE-OFF) | `step12b_analysis.json` |
| Reference level on this split at 50 NFE: 0.6100 (Step 6 checkpoints, sweep), 0.6095 (Step 12 reference, different checkpoints) | `all_experiments_long.csv`, `step12_analysis.json` |

The bottleneck this evidence points at is not encoder capacity but what a
fingerprint of an unseen compound carries, and the interaction between compound
and cell line. That fixes the expectation in section 6.

**Precedent, fixed before any result (literature check of 2026-09-21; the
primary papers are to be read before the thesis cites them).** Self-attention
over drug substructures as a drug encoder is established prior art in bulk
drug-sensitivity prediction (PaccMann, Manica et al., *Mol. Pharmaceutics*
2019; DeepTTA, Jiang et al., *Brief. Bioinform.* 2022). Attention between a drug
and a cell-line representation exists in bulk IC50 models: PaccMann's
"contextual attention" re-weights drug tokens in the context of the cell's
expression, and TransCDR (Xia et al., *BMC Biology* 2024) implements
drug-queries-cell and cell-queries-drug cross-attention; both predict a scalar
sensitivity, not an expression profile. Single-cell perturbation models compose
drug and cell latents additively (CPA, chemCPA, biolord, GEARS, TxPert) or
condition a generative model by concatenation or guidance (Squidiff, scDiffusion,
PRnet); where attention appears it is among genes with a perturbation token
(scGPT), over condition tokens without the cell state (CellFlow), or among cells
(STATE). The 2026 diffusion transformers CrossDiT and PertDiff cross-attend to
drug and control-expression conditions with the noisy expression tokens or a text
description as the query — the reverse direction of arm `attn_cross`. No
published model in which the drug embedding is the query over tokens made from
the control-cell expression, inside a generative single-cell model, was found; no
compound-memory attention for expression prediction was found. The novelty claim
this step can support is therefore narrow: a role-swapped, generative-setting
analogue of PaccMann's contextual attention, tested against a parameter-matched
control. Nothing wider is claimed whatever the outcome.

## 3. The intervention

**Common form.** Each arm is a module that maps the drug input (and, for the
cross-attention arm, the control feature) to a residual `r` of width 1024; the
unchanged encoder MLP then receives `drug_dose + r`. The module's last projection
is initialised to zero (weight and bias), so at step 0 every arm is bitwise the
reference model. `encoder.fc1.weight` keeps its 2048 × 1224 shape, which both
samplers use to identify the drug encoding; arm identity is read from the
module's state-dict keys. Widths are fixed here and not tuned: d = 64, 4 heads,
one attention layer, 16 cell tokens, a 256-token budget for active bits.

| arm | module (`Squidiff/drug_attention.py`) | what it tests | added parameters |
|---|---|---|---|
| `mlp` (reference) | none; `r = 0` | — | 0 |
| `attn_fp` | `FingerprintSelfAttention`: a 1024 × 64 embedding of bit identity; tokens = the active bits, each scaled by the log10(dose + 1) the fingerprint already carries; one transformer layer (4 heads, feed-forward 256, no dropout) with padding mask; masked mean over the active tokens; zero-init 64 → 1024. A row with no active bits gives `r = 0` by construction. | reading substructures in combination | 182,080 |
| `attn_mem` | `CompoundMemoryAttention`: one slot per training compound of the split (M = 112 on `drug_split_1`); key = learned projection (1024 → 64) of that compound's *unscaled* FCFP4 bits, stored as a checkpoint buffer; value = a free 64-vector per compound; query = projection of the input `drug_dose`; softmax over the M keys; zero-init 64 → 1024. | representing an unseen compound through the seen ones it resembles (the neural form of a fingerprint kernel) | 197,760 + 64 M = 204,928 |
| `attn_cross` | `DrugCellCrossAttention`: the control feature (200) projected to 16 tokens × 64; the drug (1024 → 64) is a single query over them (multi-head attention, 4 heads); zero-init 64 → 1024. The first multiplicative drug × cell-state term in the encoder. | modelling the drug × cell-line interaction | 354,624 |
| `mlp_ctrl` (control) | `MatchedMLPResidual`: 1024 → 264 → 64 → 1024, ReLU, zero-init last layer; no attention; parameters within 0.15% of `attn_cross`. | extra parameters plus a new pathway, without attention (Step 4b's lesson) | 354,120 |

Whole-model parameter counts, asserted at construction and recorded in every
provenance file: reference 58,178,180; `attn_fp` 58,360,260; `attn_mem`
58,383,108; `attn_cross` 58,532,804; `mlp_ctrl` 58,532,300.

**Memory arm, leakage guard.** The key table is built by the training script
from the unique SMILES of the split's *training* file only (fingerprint at dose
9, for which log10(dose + 1) = 1 exactly, i.e. the unscaled bits). The script
asserts that this set is disjoint from the unique SMILES of the split's test
file, records M and the sha256 of the sorted training-SMILES list in the
provenance, and refuses to run otherwise. A held-out compound is never a slot.

**Training.** Five seeds per arm, 20 runs, on `drug_split_1`: the Step 11/12b
recipe without condition dropout — Gaussian-source flow matching, semantic adaLN,
`--drug_encoding fingerprint` (full construction), 100,000 steps, uniform time
sampler, lr 1e-4, batch 64, EMA 0.9999, `cond_drop` 0, at the commit of this
document plus the implementation commits listed in section 9 before any run.
Guidance is out of scope for this step: two of the arms do not return `r = 0`
for a hidden condition, so the null condition of Step 12 is not defined for them.

**Reference.** The five Step 6 checkpoints sampled at 50 NFE, already evaluated
in the Step 6 NFE sweep with per-cell files
(`leakage_results/vectors_sw_fm_gaussian_seed_<k>_n50.npz`, DE-overlap 0.6100,
energy 0.1795). No new reference runs. Same data construction, same objective,
same NFE, same evaluation harness; different checkpoints, so every contrast in
this step is between checkpoints, as Step 12's dropout secondary was.

**Evaluation.** 50 NFE Euler on cn274, per-cell storage on, 443 conditions, 37
held-out compounds; the post-checks of `eval_step12b.sh` (condition count,
finite, non-negative, provenance with hashes) plus: the arm inferred from the
checkpoint equals the arm the task expects; for `attn_mem`, the buffer's M
equals the provenance's M and the buffer's row set equals the fingerprints of
the recorded training SMILES.

**Implementation assertions, part of the run:** parameter count per arm as
above; at construction the module's residual is exactly zero on a random batch
(identity to the reference at step 0); the active-bit budget (256) is never
exceeded on any batch, training or evaluation (hard failure, never truncation);
the Step 10 magnitude gate (mean |B − C| of each arm within [0.5, 2] × the
reference's).

## 4. Endpoints

All contrasts: mean over conditions of seed-mean per-condition values,
drug-clustered percentile bootstrap over the 37 held-out compounds, 5000
replicates, `RandomState(0)`, **95%** (development split).

**Primary, gated, per attention arm:** DE-overlap Jaccard, arm minus reference.

**Attribution, gated, per attention arm:** DE-overlap Jaccard, arm minus
`mlp_ctrl`. An attention arm is credited only if it beats the control as well as
the reference.

**Co-primary, gated, per attention arm:** energy distance (amendment-13
U-statistic, mean over conditions), arm minus reference. Higher is worse.

**Secondary, reported, not gated:** `mlp_ctrl` minus reference on both
endpoints (is capacity alone worth anything); DPC-50 as the paired condition
median of each arm next to the Step 7 values (interaction ridge 0.1884, plain
0.1155, flow matching 0.0924, marginal medians) with the paired arm − interaction
ridge difference and its interval; DPC-50 mean, B − C and predicted-delta
magnitude for every arm; for `attn_mem`, the entropy of the memory attention on
held-out compounds (how many training compounds an unseen drug borrows from),
descriptive.

**Sensitivity.** Step 12's between-checkpoint contrast on this split (dropout
arm minus reference, same design as here) had a DE-overlap interval half-width
of 0.0018; effects below about 0.003 will not resolve with five seeds, and this
is stated so that "unresolved" is read against it.

**Multiplicity.** Seven gated contrasts at 95% on one split. This is a screen:
any arm placed in category 1 below is a hypothesis for Step 13b at 97.5% per
split on the other two splits, not a claim. No category is reported as a
finding from this document alone.

**Escalation, fixed now.** The thesis's standing comparator is the interaction
ridge on DPC-50. If any arm's paired DPC-50 difference against the interaction
ridge resolves above zero on this split, Step 13b gates that contrast as well, at
97.5% on both confirmation splits, and only a confirmation there may change the
thesis's headline. A DE-overlap gain alone, however large, leaves the headline
as it is and is reported as a narrowing of the gap.

## 5. Outcome categories, fixed now, applied per attention arm

1. **ATTENTION HELPS** — DE-overlap above zero against the reference AND above
   zero against `mlp_ctrl` AND energy not above zero against the reference.
2. **CAPACITY** — DE-overlap above zero against the reference but not above zero
   against `mlp_ctrl`. Reported as a capacity effect, not an attention effect.
3. **TRADE-OFF** — DE-overlap above zero against the reference AND energy above
   zero against the reference (whatever the control comparison).
4. **UNRESOLVED** — DE-overlap against the reference includes zero.
5. **HARMS** — DE-overlap against the reference below zero.

"Not above zero" for energy is unresolved, not equivalence. If `mlp_ctrl` itself
lands above zero against the reference, that is reported first: it means the
reference encoder was under-parameterised for this task, and every arm's
comparison to the reference is read in that light.

## 6. Expectation, recorded in advance

`attn_fp`: **UNRESOLVED** — a better reading of the same 1024 bits does not add
information about an unseen compound. `attn_mem`: **UNRESOLVED or CAPACITY** —
borrowing from similar seen compounds is what ridge already does, and ridge's
DPC-50 advantage over the model is not a reading advantage. `attn_cross`: the
only arm with a mechanism the evidence supports; I expect a small positive
DE-overlap point estimate, and with the sensitivity stated I expect it to be
**UNRESOLVED** rather than category 1. `mlp_ctrl`: UNRESOLVED against the
reference. On DPC-50 I expect every arm to remain below the interaction ridge
(paired difference resolved below zero), i.e. the thesis's headline unchanged.

If `attn_cross` resolves above both the reference and the control, that is the
first neural arm to recover interaction signal, and Step 13b follows. If
`attn_fp` or `attn_mem` resolves in category 1, the expectation was wrong on the
bottleneck and that needs explaining rather than reporting.

## 7. What this cannot establish

- **Confirmation.** One split, the development split, seven contrasts.
- **That the widths chosen are the right ones.** d = 64, 4 heads, one layer, 16
  cell tokens are fixed a priori; no sweep, no tuning; a null result is a null
  for this configuration.
- **Anything about guidance on these arms** (out of scope, section 3).
- **That an unresolved arm has no effect.** The sensitivity is about 0.003.
- **Anything about DDIM, other NFE, or the Step 10 data construction.**
- **A claim about ridge.** The DPC-50 comparison is reported; the interaction
  ridge remains the standing comparator unless an arm resolves above it.

## 8. Analysis plan, fixed now

`analyse_step13.py`, committed before any checkpoint exists, reads the 20 arm
vector files and the 5 reference files, checks every provenance record's arm and
parameter count against its filename and every vectors hash against its record,
computes the contrasts of section 4 per arm, applies section 5 mechanically,
prints every number it uses, and writes `leakage_results/step13_analysis.json`
with commit and input hashes. It refuses to run if any input is missing.

## 9. Provenance

- Module: `Squidiff/drug_attention.py` — [commit].
- Encoder, construction path and samplers: `Squidiff/MLPModel.py`,
  `Squidiff/script_util.py`, `train_squidiff.py`, `sample_squidiff.py` — [commit].
- Training script `train_step13.sh`, evaluation `eval_step13.sh`, analysis
  `analyse_step13.py` — [commit].
- Reference vectors: Step 6 checkpoints at 50 NFE, `provenance/` records of the
  Step 6 sweep; DE-overlap and energy values in `all_experiments_long.csv`.
- Step 7 comparator: `leakage_results/step7_interaction_baseline.json`.
- Pre-registration commit: [hash].
