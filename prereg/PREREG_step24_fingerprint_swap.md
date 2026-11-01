# Step 24 — does the rewired model's compound discrimination require the right fingerprint? (pre-registration)

Written 2026-11-01, before any Step 24 quantity existed. Evaluation only: no model is trained, and the arms use the
Step 15b recipe checkpoints unchanged. After this file is committed, changes are made only as dated paragraphs, each
written before the data it concerns are read.

## 1. Question

Step 15c reported that the rewired recipe identifies a held-out compound from its predicted response: PDS 0.7223 /
0.7181 / 0.7142 on splits 1 / 0 / 2, against 0.4859–0.5400 for the fingerprint ridges, which do not resolve against
chance. That is the manuscript's central positive claim about the conditioning pathway.

Three read-outs now fail to extract drug-specific signal from the same fingerprint: the ridges (Steps 7, 15c), a
feedforward network on identical features (`ridge_vs_mlp_v2.json`, response correlation 0.11–0.21 against the ridge's
0.08–0.25) and nearest-neighbour transfer over five chemical representations (Step 23, PDS 0.5077–0.5404, the
positive control itself unresolved). Step 22's external recogniser did not confirm compound-specific signatures in
the generated cells. The base model is known to be insensitive to compound identity: swapping a drug's fingerprint
for another's moves its predicted response by 2–8 % of the available span, and a blank fingerprint does as much
damage as a wrong one (`fp_sensitivity_vs_signal.txt`, sci-Plex, 443 conditions, 37 drugs).

The direct test of the claim has not been run: does the recipe still discriminate when it is given the wrong
compound? This step runs it. It is the ablation already performed on the base model, applied to the rewired model's
Tahoe predictions and scored with the measure the claim is stated in.

## 2. Design

**Arms** (development split `drug_split_1`; the Step 15b recipe checkpoints, seeds 0–2, unchanged):

- **`fpshuf`** — every held-out structure is handed another held-out structure's SMILES for encoding, under one fixed
  derangement of the split's 75 held-out structures (no structure keeps its own). Dose scaling is applied to the
  substituted fingerprint exactly as to the real one, so dose information is untouched.
- **`fpzero`** — the fingerprint is all zeros at every condition. This separates *wrong* compound information from
  *no* compound information, as the base-model ablation did, and it is the null embedding the guided sampler already
  builds internally, so it is a within-model reference rather than a new construction.
- **`recipe_repro`** — one run (seed 0) with the substitution switched off, to show the patch is output-neutral.

Nothing else changes: the same control cells, the same real-cell draw, 50-step Euler sampling without guidance, the
200-cell cap, the same condition order.

**Why evaluation only.** The compound encoder is a function from SMILES and dose to a 1,024-dimensional vector. A
substituted fingerprint has the same width, so the checkpoint, the width-inferred encoding name and every launcher
assertion are unaffected. The question — does the trained model's discrimination depend on receiving the right
compound — is answered at sampling time. A model *trained* on shuffled fingerprints answers a different question
(whether such a model could still discriminate) and is not run here.

**What is read.** Only `drug_split_1`: its Step 15a test and paired control files, the Step 15b recipe checkpoints
and their provenance, the Step 15b and 15c per-condition records, and the Step 23 records for the salt-twin flags.
Nothing from splits 0 and 2.

## 3. The substitution

Built once, before any arm runs, and recorded (`tahoe_results/step24/permutation.json`): the split's 75 held-out
structure ids in sorted order, a derangement drawn with `RandomState(0)` by rejection (redraw while any structure maps
to itself or would receive fingerprint bits identical to its own), the resulting structure-to-structure map, each
structure's own and substituted SMILES, and the sha256 of the map. The SMILES are the strings the test file carries,
which are the strings the harness hands to the encoder, so a substituted compound receives exactly the fingerprint the
other compound receives in its own evaluation. The map is keyed by structure, not by condition, so a compound
receives one consistent wrong fingerprint across all its conditions, lines, doses and plates — the Step 16
convention for a shuffled control.

Recorded with it, for interpretation rather than as a gate: the FCFP4 Tanimoto similarity between each structure's
own and substituted fingerprint (median, range), so a derangement that happens to pair similar compounds is visible.

## 4. Measures

Every measure is the committed code already used for these predictions, on the Step 15c candidate sets:

- **PDS** (`score_step15c_tahoe.pds_sets`): the predicted change ranked by cosine distance against the measured
  changes of the held-out conditions of the same line at the same dose, the other conditions of the same structure
  removed. Chance 0.5.
- **DE-overlap, DPC-50** (`score_step15_tahoe.thesis_per_cond`) and **energy distance** (`score_step15_tahoe.energy`,
  on the stored per-cell populations).
- **MRA** (Step 16 §4.4): within each (line, dose) context, the Spearman correlation between predicted and measured
  response size, averaged over the 42 contexts.

Arms are scored per seed and averaged per condition. Intervals: 97.5 % structure-clustered percentile bootstrap,
5,000 replicates, a fresh `RandomState(0)` per scored set, as Step 23.

**Scored set.** All 2,899 held-out conditions. The 8 salt-twin structures of Step 23 are *not* excluded here: the
question is about the model's use of its own input, and both arms see the same conditions, so the comparison is
internal. Their 384 conditions are also reported separately.

## 5. Gated contrasts and verdict

Primary and co-primary, on PDS:

- **Q1 primary** — recipe − `fpshuf`. Above zero means the discrimination requires the right compound.
- **Q2 co-primary** — `fpshuf` − 0.5. Above zero means the wrong-fingerprint model still discriminates, which can
  only come from something other than compound identity.
- **Q3 reference** — `fpzero` − 0.5, `fpshuf` − `fpzero`, and recipe − `fpzero`.

Secondary, not gated: the same three arm differences (recipe − `fpshuf`, recipe − `fpzero`, `fpshuf` − `fpzero`) on
DE-overlap, DPC-50, energy and MRA, and each arm's MRA, all with intervals; PDS per dose; and PDS on the salt-twin
conditions (descriptive).

**Verdict**, first match:

1. **DISCRIMINATION IS COMPOUND-SPECIFIC** — Q1 above zero and Q2 includes zero or below. The claim stands as
   written: the recipe needs the right compound, and without it falls to chance.
2. **PARTLY COMPOUND-SPECIFIC** — Q1 above zero and Q2 above zero. The recipe uses the compound, but part of the
   reported 0.72 is reachable without it; the manuscript must report the wrong-fingerprint level beside it.
3. **NOT SHOWN TO BE COMPOUND-SPECIFIC** — Q1 includes zero or lies below zero, and Q2 above zero. The reported
   discrimination does not require the right compound. Results 3 and the abstract must be rewritten, and Step 15c's
   reading revised. (Q1 below zero would mean the wrong fingerprint discriminates better than the right one; it is
   reported as such, and the verdict is the same.)
4. **UNINTERPRETABLE** — Q1 includes zero or lies below zero, and Q2 includes zero or below (both arms near chance on
   this scored set), which would contradict the committed Step 15c record and means a fault, not a result.

Under 1 and 2 the size of Q1 is reported as the amount of discrimination attributable to compound identity.

## 6. Expectations (written before any Step 24 value)

Q1 above zero, Q2 above zero: outcome 2, PARTLY COMPOUND-SPECIFIC. The reasoning: PDS scores the *direction* of the
predicted change by cosine, and a prediction conditioned on this condition's own vehicle cells carries line, dose and
plate structure that is shared with the measured change, so some discrimination should survive the substitution. I
expect `fpshuf` in the 0.55–0.65 range and Q1 between +0.05 and +0.15, with `fpzero` close to `fpshuf` (the base
model's ablation found wrong ≈ blank). I record now that I am not confident in these numbers: the three failed
read-outs of §1 would also fit outcome 3, and the honest position before the run is that outcomes 2 and 3 are both
live.

## 7. Guards and reproduction checks (before any Step 24 value is printed)

1. This file, the patch, the launcher and the scorer committed and identical to HEAD; the permutation record present
   and its sha256 matching; no output exists. Only `drug_split_1` is read.
2. The permutation is a derangement over exactly the record's 75 held-out structures.
3. **`recipe_repro` reproduces the committed Step 15b arrays bitwise** — `cond`, `true`, `C`, the stored real cells
   and `B` — with max |dB| = 0, as the Step 18 harness change did. Until this passes, no shuffled number is read.
4. Every arm's `cond`, `true`, `C` and stored real cells are identical to the recipe's, so only the prediction differs.
5. The recipe's per-condition PDS, DE-overlap, DPC-50 and energy rebuilt here reproduce the Step 15c and 15b records
   to 1e-9, its MRA the Step 16 values to four decimals, and the bootstrap the Step 15c split-1 interval.
6. In every `fpshuf` run the fingerprint actually handed to the model differs from the real one for all 75
   structures (asserted inside the patched encoder, counted and logged), and in `fpzero` it is exactly zero.
7. In every `fpshuf` and `fpzero` run the predicted mean differs from the recipe's same-seed prediction at every
   condition. An unchanged condition means the substitution did not reach the model there; the scorer stops for
   inspection rather than score it.

## 8. Scripts, records, run order

1. Committed before any run: this file; `make_step24_permutation.py`; `port_harness_step24.py` and the harness it
   generates, `leakage_test_fm_step24.py` (identical to the Step 15b Tahoe harness except the encoder shim, which the
   port script verifies byte for byte); `eval_step24_tahoe.sh`; `score_step24_tahoe.py` and its job script
   `run_step24_score.sh`.
2. The permutation record is built and committed.
3. `recipe_repro` runs and its bitwise check passes (guard 3).
4. The six arm runs (`fpshuf` and `fpzero`, seeds 0–2), then scoring, then the records committed.
5. Findings in `step24_findings.md`. No rerun with changed settings; a fault fixed after a run is reported as such,
   with both runs kept.

## 9. Limits

- One split. A result here is a statement about the development split's 75 held-out compounds, and the confirmation
  splits would be a separate, pre-registered extension.
- Evaluation only: it tests what the trained recipe does with a substituted input, not what a model trained on
  substituted inputs would learn.
- One derangement. A different pairing would give slightly different numbers; the Tanimoto record of §3 shows how
  similar the substituted compounds happen to be, but no interval covers the choice of permutation.
- `fpzero` is outside the training distribution for an unguided arm (cond_drop 0 means the recipe never saw a blank
  fingerprint in training), so it is a reference, not a fair arm.
- PDS scores direction only, as Step 15c §9 states. A null on Q1 would not mean the model ignores the compound in
  every respect, only that this measure does not require it.
