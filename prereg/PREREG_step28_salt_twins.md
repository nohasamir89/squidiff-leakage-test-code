# Step 28 — sensitivity of the reported results to held-out compounds whose salt or hydrate twin is in training (pre-registration)

Written 2026-11-09, after the Step 27 records and findings were committed (research `1765382`, package `04f5873`)
and before any Step 28 quantity was computed. Evaluation only, CPU only, no model is run and no cell is re-sampled:
every quantity is a re-aggregation of the committed per-condition records with a fixed set of conditions removed.
After this file is committed, changes are made only as dated paragraphs, each written before the data it concerns
are read.

## 1. Question

The compound splits define a "structure" as the RDKit canonical SMILES with stereochemistry removed (Methods,
Compound splits). Counter-ions, solvent molecules and water were kept in those SMILES, so one active molecule
supplied as two salts counts as two structures and the two can fall on opposite sides of a split. `check_split_parent_overlap.py`
(committed with its record `leakage_results/split_parent_overlap.json`, 25 Oct) matched compounds by the parent
molecule — the first block of the standard InChIKey of the largest organic fragment — and found held-out structures
whose parent also occurs in the same split's training set:

| dataset | split 1 (dev) | split 0 | split 2 |
|---|---|---|---|
| Tahoe-100M (of 75 held-out structures) | 8 | 4 | 9 |
| sci-Plex (of 37 / 37 / 38 held-out compounds) | 0 | 1 (12 test conditions) | 1 (12 test conditions) |

The twins' fingerprints, as the model encodes them, have a median Tanimoto of 0.97 across the 22 parent-sharing
pairs of the Tahoe compound list, so for those compounds the model saw a near-identical input during training. A
reader may therefore ask whether any reported result is carried by compounds that were not truly unseen. Step 28
answers that question for the results that could be affected by it, and only those.

**Which results could be affected.** Knowing the compound can only inflate a quantity that depends on the compound.
Three groups of results are therefore treated differently, and the reasoning is part of the registered design:

1. **Results where the model loses** (Sections 1–2: the original model below the unaltered vehicle cells and below
   every linear reference; the rewired model below the compound-blind average on Tahoe gene identity). Leakage can
   only help the model, so removing the twins can only widen these gaps. These are recomputed, because the cost is
   one more column, but no verdict can turn on them.
2. **The discrimination score and its audit** (Section 5). Steps 24–26 found the score unchanged when the compound
   is replaced by another held-out compound's fingerprint or removed entirely (gain from the correct compound
   +0.006 / +0.017 / +0.024, every interval including zero), so the score does not use the compound and knowing it
   cannot inflate the score. These are recomputed to foreclose the question, not because the argument needs them.
3. **The one result that depends on the compound and is positive**: sci-Plex gene identity, where the rewired model
   reaches 0.61 on every split, above the cross-validated ridge (0.55–0.57) and the compound-blind average
   (0.53–0.55). This is where the question has force. Note that sci-Plex split 1 contains no parent twins at all, so
   the published 0.61 on that split is already free of this route; Step 28 measures the other two splits against it.

## 2. Inputs

Committed per-condition records only; no `.h5ad` file, checkpoint or array is read, and no sampler is run.

- `tahoe_results/step15_model_per_condition.csv` (gene identity, population distance, response correlation, every
  arm and reference, 8,867 rows over three splits)
- `tahoe_results/step15c_pds_per_condition.csv` (cross-plate discrimination, published construction)
- `tahoe_results/step25_swap_per_condition.csv`, `step26_source_per_condition.csv`,
  `step27_within_plate_per_condition.csv` (the audit arms, their schemes and the within-plate restriction)
- `tahoe_results/step16_per_condition.csv` (response-magnitude agreement, now Supplementary Note 4)
- `leakage_results/split_parent_overlap.json` (the exclusion list) and `tahoe_results/step15_data_record.json`
  (structure ids ↔ compound names)
- sci-Plex: `leakage_results/step14_analysis.json` and whichever per-condition record the Step 14 scorer wrote. If
  no per-condition record exists, the sci-Plex arm instead re-runs the committed `score_step14.py` with an exclusion
  list passed as an argument, and is gated by §7.3.

Every file above must be committed and unchanged; the scorer refuses to run otherwise.

## 3. Exclusion rule (fixed before any recomputation)

A held-out structure is **excluded on a split** if and only if it appears in that split's
`held_out_with_parent_in_training` list of `leakage_results/split_parent_overlap.json`, as committed. The lists are
copied into the scorer as literal constants and checked against the record at run time; a mismatch stops the step.

Tahoe structure ids: split 1 — S0012, S0052, S0077, S0142, S0190, S0216, S0317, S0326; split 0 — S0058, S0141,
S0204, S0287; split 2 — S0011, S0095, S0137, S0139, S0191, S0241, S0258, S0343, S0345. sci-Plex: the two SMILES
named in the record (one on split 0, one on split 2; none on split 1).

**What exclusion means.** The conditions of an excluded structure are removed from the set over which each quantity
is averaged, and from the bootstrap clusters. Candidate sets are **not** rebuilt: for the discrimination score, the
other conditions' candidates are measured changes of real cells, not predictions, so a twin among the candidates is
not a route by which training leaked into a prediction, and rebuilding the sets would change every condition's score
and make the comparison with the published values impossible. This choice is registered here with its reason; the
alternative is not run and is named in §9.

The number of conditions excluded per split is not yet known for Tahoe (the overlap record lists structures, not
conditions); the scorer prints it, and it is reported whatever it is.

## 4. Quantities

Per split, recomputed on the retained conditions, beside the published value and the difference:

**A. Gene-level and population agreement (Tahoe, Step 15b record)** — gene identity, population distance and
response correlation for: `base` (original model), `recipe` (rewired model), `ridge_train`, `ridge_S7`,
`mean_response` (compound-blind average), `control_paste` (unaltered vehicle cells).

**B. Gene-level agreement (sci-Plex, Step 14 record)** — gene identity and response correlation for the same six
entries. This is group 3 of §1.

**C. Cross-plate discrimination (Step 15c construction, and the Step 25–26 arms)** — `recipe`, `base`, their
`fpshuf` and `fpzero` arms, `ctrl_line`, `ctrl_sample`, `duplicate`, under S-orig and S-split.

**D. Within-plate discrimination and the plate residual (Step 27)** — `recipe`, `base`, their `fpshuf` arms,
`ctrl_line`, `duplicate` under S-split; the measured plate residual (`true`) and its `ctrl_sample_h2` null.

**E. Response-magnitude agreement (Tahoe, Step 16 record)** — `recipe` and the references, since Supplementary
Note 4 reports it.

**Contrasts** are the ones the manuscript states, recomputed on the retained conditions and paired per condition
where they were paired before: rewired − original; rewired − each linear reference; rewired − compound-blind
average; `recipe` − `recipe_fpshuf` (the compound contrast of Steps 24–26); `recipe` − 0.5 and `base` − 0.5 within
plate; measured plate residual − 0.

**Intervals** are the same as the published ones: 97.5 % structure-clustered percentile bootstrap, 5,000 replicates,
fresh `RandomState(0)` per scored set (`score_step23_screen.Boot`), with the excluded structures absent from the
cluster list. Intervals are expected to be slightly wider than the published ones because there are fewer clusters.

## 5. Verdicts

**Verdict K (does any reported conclusion change?)**, read per quantity and then overall:

A conclusion **changes** if, after exclusion,

1. an interval that excluded zero (or chance, for discrimination) now includes it, or an interval that included zero
   now excludes it; or
2. an ordering the manuscript states reverses — specifically: rewired above every linear reference on sci-Plex gene
   identity; rewired below the compound-blind average on Tahoe gene identity; the original model below the unaltered
   vehicle cells on population distance; the cell-line vehicle average above the rewired model on cross-plate
   discrimination; both models at chance within plate.

Outcomes:

1. **NO CHANGE** — no quantity in §4 meets either criterion. The manuscript's conclusions stand as written; the
   Methods state the overlap counts and point to the supplementary note.
2. **CHANGE CONFINED TO MAGNITUDES** — no ordering reverses and no interval crosses, but some point estimate moves
   by more than the published interval's half-width. Reported as such; the affected numbers are restated with both
   values.
3. **A CONCLUSION CHANGES** — any quantity meets criterion 1 or 2. The affected claim is rewritten in the main text
   with the twin-free value as the primary number and the published value beside it, and the Abstract is revised if
   the claim is in it.
4. **UNINTERPRETABLE** — any reproduction gate of §7 fails.

**Verdict L (descriptive, committed):** whether the excluded compounds scored above or below the retained ones on
each quantity. If leakage were operating, the excluded compounds should score better than the rest; if they score
the same or worse, that is evidence against this route independent of verdict K. Reported as the difference of means
with its interval, for the rewired model on sci-Plex gene identity and Tahoe gene identity, and for the cross-plate
discrimination score.

## 6. Expectations (written before any Step 28 value exists)

- Verdict K outcome 1 (NO CHANGE) — confident. The fractions removed are 5–12 % of structures on Tahoe and one
  compound of 37–38 on sci-Plex, and the gaps the manuscript claims are much larger than the movement a tenth of the
  clusters can produce.
- sci-Plex gene identity of the rewired model: within 0.01 of 0.61 on splits 0 and 2 — moderately confident, and
  split 1 is unchanged by construction (no twins), which is the check that matters.
- Cross-plate discrimination: within 0.01 of the published values on every arm — confident, because the score does
  not use the compound at all (Steps 24–26).
- Within-plate discrimination: unchanged in verdict (intervals still include chance) — confident, though the
  intervals will widen.
- Verdict L: the excluded compounds score **no better** than the retained ones on sci-Plex gene identity — moderate
  confidence only. A salt twin in training gives the model a near-identical fingerprint, so a positive difference
  would not be surprising; if it appears without changing verdict K, it is reported as a measured effect that is too
  small to matter at this scale.
- Intervals wider than published by roughly the square root of the cluster ratio (a few per cent) — confident.

## 7. Guards and reproduction checks (before any Step 28 value is printed)

1. This file, `score_step28_salt_twins.py` and `run_step28_score.sh` committed and identical to HEAD; the Step 27
   records and findings committed; `check_split_parent_overlap.py` and `leakage_results/split_parent_overlap.json`
   committed; no Step 28 output exists.
2. **Empty-exclusion reproduction.** With an empty exclusion list, every quantity of §4 reproduces its published
   value to 1e-9 and every interval replays to 5e-6. This is the gate that proves the re-aggregation is the same
   arithmetic as the original scorers; it runs first and a failure stops the step. For the sci-Plex arm taken
   through a re-run of `score_step14.py`, the same gate applies to its outputs.
3. **Exclusion-list integrity.** The literal lists in the scorer equal the committed record; every excluded
   structure id exists in the Step 15 data record; the counts are 8 / 4 / 9 (Tahoe, splits 1 / 0 / 2) and 0 / 1 / 1
   (sci-Plex, splits 1 / 0 / 2); any mismatch stops the step.
4. **Row integrity.** Retained rows are a strict subset of the published rows, in the same order, with the same
   `condition` values; retained + excluded = published for every split and quantity.
5. Outputs written once (`step28_salt_twins.json`, `step28_salt_twins_per_quantity.csv`,
   `step28_salt_twins_summary.csv`), never overwritten; log to `tahoe_results/step28_logs/`.

## 8. Scripts, records, run order

1. Committed before any run: this file, `score_step28_salt_twins.py` (imports the committed Step 15, 15c, 23, 25,
   26 and 27 scorers for their constants and their bootstrap; reimplements nothing), `run_step28_score.sh`.
2. Synthetic test before the real run: three scenarios on a mock set of 36 structures per split — (a) excluded
   compounds identical in distribution to the retained ones, (b) excluded compounds planted far better than the
   retained ones, enough to move a mean past an interval, (c) excluded compounds planted worse — recovered as
   verdict K outcome 1, outcome 3 and outcome 1 respectively, with verdict L positive only in (b); plus six negative
   tests (existing output, uncommitted script, tampered exclusion list, tampered published value, a structure id not
   in the data record, retained + excluded ≠ published) each stopping the step.
3. One CPU job, minutes. Findings in `step28_findings.md`; records committed to the three repositories as Steps
   24–27 were.
4. No rerun with changed settings. A fault fixed after a run is reported as such, with both runs kept.

## 9. Limits

- The parent key matches salts, hydrates, solvates, charge states and stereoisomers of one molecule. It does **not**
  match close analogues: a compound differing by one methyl group is a different parent and is not excluded, though
  its fingerprint may be nearly as similar. This step bounds one specific route, not chemical similarity in general;
  the structure-based split itself is the control for that, and a nearest-neighbour similarity analysis is not run.
- Candidate sets are not rebuilt (§3), so the discrimination quantities answer "do the twin compounds' own scores
  inflate the average?" and not "would the score change if twins were absent from the screen?". The second question
  would require re-scoring from the arrays and is not asked.
- Exclusion removes 5–12 % of Tahoe structures and one sci-Plex compound, so intervals widen and the step's power to
  detect a small change is lower than the published analysis's. A verdict of NO CHANGE bounds the effect; it does not
  prove the effect is zero, and the bound is stated with the verdict.
- sci-Plex splits 0 and 2 lose 12 conditions each of roughly 450; split 1 loses none. The sci-Plex arm therefore
  rests on a comparison with a split that was already clean, which is a stronger argument than the exclusion itself.
- No model is retrained. The question "what would the model have learned had the twins never been in training?" is
  not answered by this step and is not claimed to be.
- The overlap counts were known before this file was written (the record is from 25 Oct); the per-condition counts,
  every recomputed value and every difference are not. §6 says what is expected before they exist.

## 10. Amendment, 2026-11-09 (before any Step 28 value was computed)

Group E of §4 (response-magnitude agreement) is withdrawn from this step. Response-magnitude agreement is a rank
correlation computed across the conditions of a context, not a per-condition quantity, so removing conditions
changes the ranking itself rather than the average of a per-condition value, and the question of §1 — whether the
affected compounds inflate a reported average — is not well posed for it. It is also not a compound-dependent
result: the manuscript reports the same ranking with another compound's fingerprint or none (0.21–0.36), so the
leakage route of §1 cannot be operating on it. §4 groups A, C and D are unchanged, and §5's verdicts are read over
them. Supplementary Note 4 is unaffected and states its own references.

## 11. Amendment, 2026-11-09 (after synthetic runs 19237778 and 19238263; before any real Step 28 value was computed)

§7.2 gate 2 stated an absolute tolerance of 5e-6 between the interval from `score_step23_screen.Boot` and the one
from `score_step15_tahoe.boot`. The synthetic test showed this to be the wrong kind of tolerance. The two are the
same estimator on the same draws, but they sum in different orders — `Boot.mean_reps` forms each replicate as the
drawn clusters' sums over their member counts, `s15.boot` averages the drawn values written out — and
`np.percentile`'s interpolation between order statistics amplifies the resulting ~1e-16 per-value difference to
~2–5e-5 on intervals ~0.017 wide (0.3 % of the width; run 19238263). A genuine mismatch of draws or clusters would
move an interval by something of the order of its own width. Gate 2 therefore requires agreement to 1 % of each
interval's width (floor 5e-6), and the largest difference observed is written to the output
(`gate2_largest_difference`, per split) rather than discarded. Gate 1 (1e-9) is unchanged. §8.2 also expected
scenario (c), the excluded compounds planted worse than the rest, to give verdict K1; the test expects K3, because
the verdict asks whether a reported value changes, not whether it improves; a shift of the same size in either
direction moves the retained mean. The run that confirms or refutes this is reported in `step28_findings.md`.

## 12. Amendment, 2026-11-09 (after synthetic run 19238650; before any real Step 28 value was computed) — withdraws §11

§11 attributed the gate-2 difference to summation order and loosened the tolerance to 1 % of the interval width.
Run 19238650 showed differences of 0.5–2.7e-3 on intervals 0.02–0.08 wide, far too large for rounding, while the
first column tested (`recipe_de_overlap`) matched to better than 5e-6 in every run and only the later columns
drifted. That is the signature of a shared generator: `score_step15_tahoe.boot` draws from the module-level
`s15.rng`, which every call advances, so only the first call after import starts from `RandomState(0)`;
`score_step23_screen.Boot` makes a fresh `RandomState(0)` per set. Step 27 resets the same generator before its
replay for the same reason. Gate 2 therefore resets `s15.rng = RandomState(0)` before each `s15.boot` call and
requires agreement to the absolute 5e-6 that §7.2 first stated; §11's relative tolerance is withdrawn. Consequence
for reporting: the "published" intervals in the Step 28 table are recomputed from a fresh seed and so can differ
from the manuscript's printed digits by the Monte-Carlo noise of a redraw (of the order of 1 % of the width);
the published and twin-free sets in the table use the same draws, so their comparison is internally consistent,
and the findings state this.

## 13. Amendment, 2026-11-09 (after real run 19239333 failed at gate 1; before any Step 28 value was computed)

The first real run stopped at gate 1 because tahoe_results/step15_model_long.csv carries, besides the per-line
rows, a split-level row under the line label ALL (the manuscript's headline value, aggregated over every condition
of the split), which the gate tried to select as a cell line. Gate 1 now recognises ALL and tests it over every
condition of the split - a strengthening, since the headline numbers are now checked to 1e-9, not only the per-line
breakdown - and skips, with a printed note, any long-record line label that has no conditions in the per-condition
file (a tissue aggregate, if present), which cannot be reproduced from the per-condition rows. No value changed;
the gate reaches further than section 7.2 described. The scorer committed in c56f235 had this gate-1 defect; the
corrected file supersedes it.

## 14. Amendment, 2026-11-09 (after real run 19239924 passed gate 1 but matched no exclusion; before any Step 28 value was computed)

Section 3 said the `structure` column is "the id on some steps and the SMILES on others" and named two SMILES
fields. The Step 15b per-condition record in fact carries `table_smiles` (RDKit-canonicalised from the data table),
which differs from `canonical_smiles`; resolving the excluded structure ids through `canonical_smiles` matched none.
The resolver now tries three forms of the identifier - structure id, `table_smiles`, `canonical_smiles` - and uses
the first that covers every excluded structure, resolving each id through all of its compound names so that a study
structure supplied under two salt names is matched in full. It still stops, naming how many each form matched, if
no form covers all of them. This selects the join; it does not change which structures are excluded (section 3's
list is unchanged). Gate 1 had already reproduced all 1,125 published values including the split-level headline
rows at this commit, so the aggregation is unaffected.
