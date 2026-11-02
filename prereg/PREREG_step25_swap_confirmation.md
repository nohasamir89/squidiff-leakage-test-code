# Step 25 — does the fingerprint-swap null replicate, and does the base model behave the same way? (pre-registration)

Written 2026-11-01, after the Step 24 development-split result was read and before any Step 25 quantity existed.
Evaluation only: no model is trained, and every arm uses the Step 15b checkpoints unchanged. After this file is
committed, changes are made only as dated paragraphs, each written before the data it concerns are read.

## 1. Question

Step 24 (`step24_findings.md`, records at 35af542) found on the development split `drug_split_1` that the rewired
recipe's compound discrimination does not require the compound: PDS 0.7223 with its own fingerprint, 0.7160 with
another held-out compound's, 0.7360 with none. Q1 (recipe − `fpshuf`) was +0.00632 [−0.01531, +0.02882], including
zero; Q2 (`fpshuf` − 0.5) was +0.21601 [+0.18619, +0.24554], above zero. Pre-registered verdict: 3, NOT SHOWN TO BE
COMPOUND-SPECIFIC.

Two things are missing before that can be stated in the manuscript.

**One split.** Step 24 §9 limits the result to the development split and says the confirmation splits would be a
separate pre-registered extension. This is that extension. The development/confirmation design exists because a
development reading has twice failed to replicate in this project (Step 15d layer A; Step 16 §5).

**The base model was never swapped.** Step 15b reports the recipe above the base model on the thesis's three metrics
on all three splits, and Step 15c reports PDS 0.6415 / 0.6444 / 0.6549 for base against 0.7223 / 0.7181 / 0.7142 for
the recipe. Step 24 ran recipe checkpoints only, so it says nothing about whether the base model's discrimination
needs the compound either, and therefore nothing about what the recipe-over-base gap measures. On sci-Plex, swapping
a drug's fingerprint moved the base model's predicted response by 2–8 % of the available span and a blank fingerprint
did as much damage as a wrong one (`fp_sensitivity_vs_signal.txt`, 443 conditions, 37 drugs), but that was a
sensitivity measurement on a different dataset, not a PDS test.

## 2. Design

**Arms.** The Step 24 arms, unchanged in construction (`PREREG_step24_fingerprint_swap.md` §2 and §3), extended in
two directions:

| arm | checkpoint | splits | seeds | S24_MODE |
|---|---|---|---|---|
| `recipe_repro` | Step 15b recipe | 0, 2 | 0 | off |
| `recipe_fpshuf` | Step 15b recipe | 0, 2 | 0–2 | shuf |
| `recipe_fpzero` | Step 15b recipe | 0, 2 | 0–2 | zero |
| `base_repro` | Step 15b base | 1, 0, 2 | 0 | off |
| `base_fpshuf` | Step 15b base | 1, 0, 2 | 0–2 | shuf |
| `base_fpzero` | Step 15b base | 1, 0, 2 | 0–2 | zero |

35 runs. Split 1's recipe arms are not re-run; the committed Step 24 records are read.

**The base model needs its own port.** The Step 24 shim sits in `leakage_test_fm_step24.py`, which the flow-matching
harness generates. The base model is evaluated through `leakage_test_base_tahoe.py`, which imports
`Drug_dose_encoder` itself, so the same substitution block is applied there by a second port script
(`port_harness_base_step25.py`), verified byte for byte against its source in the same way. The base model is a DDIM
sampler without NFE or guidance, exactly as Step 15b ran it; nothing about the sampler changes.

**One permutation per split.** The held-out structures differ between splits, so
`make_step24_permutation.py --split drug_split_0` and `--split drug_split_2` each build their own derangement under
`RandomState(0)` with the same rejection rule (no structure keeps its own SMILES or receives identical fingerprint
bits), keyed on the SMILES strings the split's test file carries. Each record's sha256 is committed before any run of
that split and re-checked by the launcher, the harness and the scorer. The split-1 permutation (sha256
69ff344f2ce3…) is reused unchanged for the base arms of split 1.

**What is read.** For each split: its Step 15a test and paired control files, the Step 15b recipe and base
checkpoints and their provenance, and the Step 15b and 15c per-condition records. The condition and structure counts
of splits 0 and 2 are read from `step15_data_record.json` at run time and reported in the findings; they are not
asserted here.

## 3. Measures

Exactly the Step 24 measures (`PREREG_step24_fingerprint_swap.md` §4), on each split's own Step 15c candidate sets:
PDS (`score_step15c_tahoe.pds_sets`, chance 0.5), DE-overlap and DPC-50 (`score_step15_tahoe.thesis_per_cond`),
energy distance (`score_step15_tahoe.energy`), and MRA (Step 16 §4.4). Arms are scored per seed and averaged per
condition. All held-out conditions of each split are scored; the salt-twin structures are included and reported
apart, as in Step 24.

Intervals: 97.5 % structure-clustered percentile bootstrap, 5,000 replicates, a fresh `RandomState(0)` per scored set
of conditions, per split, as Steps 15c, 23 and 24.

## 4. Gated contrasts

Per split, on PDS, for each checkpoint arm (recipe and base):

- **Q1** — arm − its `fpshuf`. Above zero means that arm's discrimination requires the right compound.
- **Q2** — that arm's `fpshuf` − 0.5. Above zero means the wrong-fingerprint model still discriminates.

Across arms, on PDS, per split:

- **Q3** — (recipe − base) − (`recipe_fpshuf` − `base_fpshuf`). This asks whether the recipe-over-base gap depends on
  either model receiving the right compound. Above zero means the gap shrinks when both are given wrong compounds,
  so part of the gap is compound-specific. Including zero means the gap is the same with wrong compounds as with
  right ones, and is therefore not about compound identity.

Reference, not gated: each arm's `fpzero` − 0.5, `fpshuf` − `fpzero`, arm − `fpzero`; the same six arm differences on
DE-overlap, DPC-50, energy and MRA; PDS per dose; PDS on the salt-twin conditions (descriptive).

## 5. Verdicts

**Verdict A, replication of Step 24 (recipe arms, splits 0 and 2), first match:**

1. **NULL REPLICATED** — on both confirmation splits, Q1 includes zero or lies below zero, and Q2 is above zero. The
   Step 24 reading holds across all three splits and is stated in the manuscript without split-specific hedging.
2. **NULL PARTLY REPLICATED** — the pattern of outcome 1 on one confirmation split, with Q1 above zero on the other.
   The manuscript reports the split-by-split values and claims no general result.
3. **NULL NOT REPLICATED** — Q1 above zero on both confirmation splits. The Step 24 development result does not
   generalise; Step 24 is reported as a development-split finding that failed confirmation, with both results kept.
4. **UNINTERPRETABLE** — Q2 includes zero or lies below zero on either confirmation split, which would contradict the
   committed Step 15c records for that split and means a fault, not a result.

**Verdict B, the base model (base arms, all three splits):** the same four categories applied to the base arms, read
across the three splits, with the development split treated as development and the other two as confirmation.

**Verdict C, the recipe-over-base gap:** Q3 above zero on all three splits means part of the gap is attributable to
compound identity, and its size is reported as that part. Q3 including zero on all three splits means the gap is not
about compound identity, and every statement of the form "the recipe discriminates compounds better than the base
model" is rewritten as "the recipe discriminates conditions better than the base model". Mixed outcomes are reported
split by split with no general claim.

## 6. Expectations (written before any Step 25 value)

- **Verdict A: NULL REPLICATED.** Q1 including zero on both confirmation splits, Q2 in the +0.17 to +0.25 range. The
  reasoning: the Step 24 mechanism — the shared control mean and the condition's own plate vehicle cells — is a
  property of the harness and the data layout, not of the split, and the three splits' recipe PDS values sit within
  0.008 of each other. Moderately confident.
- **Verdict B: NULL REPLICATED for base too**, with `base_fpshuf` close to base's own 0.64–0.65. The sci-Plex
  sensitivity result and three failed fingerprint read-outs all point this way. Confident in direction, not in size.
- **Verdict C: Q3 includes zero on all three splits.** If neither model needs the compound, the gap between them
  cannot be about the compound. If instead Q3 is above zero, the honest reading is that the recipe uses the compound
  a little where the base model does not, and Step 24's Q1 was too noisy on one split to see it; I record now that
  this would change the manuscript's claim and that I do not expect it.
- I was wrong in Step 24's §6 expectations (I predicted outcome 2 and `fpshuf` at 0.55–0.65, and got outcome 3 and
  0.7160). That failure is the reason the expectations above are stated with explicit confidence levels.

## 7. Guards and reproduction checks (before any Step 25 value is printed)

1. This file, both port scripts, the generated harnesses, the launcher and the scorer committed and identical to
   HEAD; each split's permutation record present with its sha256 matching; no output exists.
2. Each permutation is a derangement over exactly that split's held-out structures from the data record, and its
   SMILES are the strings that split's test file carries.
3. **Per split and per checkpoint arm, the reproduction run with the substitution switched off equals the committed
   Step 15b arrays bitwise** — every array the Step 15b file stores, and every stored real and predicted cell. Until
   a split's reproduction passes, no swapped number from that split is read. This gate is what makes the base port
   trustworthy, since the base harness is being modified for the first time here.
4. Every arm shares the conditions, measured means, control means and real cells of the Step 15b run of its own
   checkpoint, split and seed, bitwise.
5. Every `fpshuf` and `fpzero` prediction differs from its own arm's same-seed Step 15b prediction at every
   condition.
6. In every `fpshuf` run all of that split's held-out structures are substituted under the committed permutation, the
   encoder-call counter is above zero, and in `fpzero` the fingerprint is exactly zero.
7. For each split and each checkpoint arm (recipe and base), the arm's own-fingerprint per-condition PDS, DE-overlap,
   DPC-50 and energy rebuilt here reproduce the committed Step 15c (`<arm>_pds`) and Step 15b (`<arm>_de_overlap`,
   `<arm>_dpc50`, `<arm>_energy`) per-condition records to 1e-9, and the bootstrap of that arm's PDS − 0.5 reproduces
   the Step 15c published interval for that split and arm (Step 15c findings §1 recipe: +0.22234 [+0.18829, +0.25411]
   / +0.21811 [+0.17822, +0.25899] / +0.21420 [+0.18261, +0.24477] for splits 1 / 0 / 2; §3 base: +0.14150 [+0.10358,
   +0.18063] / +0.14441 [+0.10481, +0.18450] / +0.15488 [+0.11017, +0.19767]). MRA is checked to four decimals against
   Step 16 where a committed four-decimal value exists (the development-split recipe, 0.3112 / 0.3237 / 0.3265) and is
   otherwise verified internally — the vectorised replicate MRA equals the plain per-context computation to 1e-10 —
   and reported for comparison; the exact per-condition reproduction above carries the guarantee.

## 8. Scripts, records, run order

1. Committed before any run: this file; `port_harness_base_step25.py` and the harness it generates,
   `leakage_test_base_step25.py`; `eval_step25_tahoe.sh`; `score_step25_tahoe.py` and `run_step25_score.sh`. The
   Step 24 scripts are reused unchanged for the recipe arms and the flow-matching harness.
2. The two new permutation records (splits 0 and 2) are built by the committed `make_step24_permutation.py` and
   committed.
3. The five reproduction runs (`recipe_repro` on splits 0 and 2; `base_repro` on all three) run first and their
   bitwise checks pass.
4. The 30 arm runs, then scoring, then the records committed.
5. Findings in `step25_findings.md`. No rerun with changed settings; a fault fixed after a run is reported as such,
   with both runs kept.

## 9. Limits

- Evaluation only: this tests what two trained models do with a substituted input, not what a model trained without
  compound information would learn.
- One derangement per split; no interval covers the choice of pairing. The recorded Tanimoto similarities show how
  chemically alike the substituted compounds happen to be.
- `fpzero` is outside the training distribution for both models (neither was trained with condition dropout), so it
  is a reference, not a fair arm.
- PDS scores direction only. A null on Q1 says that this measure does not need the compound, not that either model
  ignores the compound in every respect.
- This step does not identify the source of the discrimination. That is Step 26, which is pre-registered separately
  and reads the same predictions on the CPU.
- The base model's cells are known to fail on scale and spread (Step 15b §9); its PDS is a direction-only measure and
  carries no claim about its cell populations.
