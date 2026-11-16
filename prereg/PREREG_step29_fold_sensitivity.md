# Step 29 — do the comparisons with the cross-validated ridge depend on the fold assignment? (pre-registration)

Written 2026-11-14, after the repetition of the Note 1 analysis and the rebuilt Supplementary Table S10 were
committed (research `2cca4b6`, package `10e0c16`) and before any Step 29 quantity was computed. CPU only. No
generative model is retrained or re-sampled. After this file is committed, changes are made only as dated
paragraphs, each written before the data it concerns are read.

## 1. Question

The cross-validated ridge is fitted within the held-out compounds, with five folds grouped by compound. The folds
come from scikit-learn's `GroupKFold`, which deals the compounds to the folds in the order of their number of
conditions. Almost all compounds have the same number of conditions, and numpy orders such ties differently on
processors with and without AVX-512 instructions. The same code therefore gives different folds on different
compute nodes (found on 13 and 14 Nov 2026; `leakage_results/diag_note1_ridge_19363649.json`).

All sci-Plex analyses of the paper used one and the same assignment, because each scoring script first had to
reproduce the median response correlation of the Step 7 record (to 1e-9 in `score_step14.py`,
`score_step14_facets_v2.py`, `score_step17_sciplex_mra.py`, `analyse_step13.py` and `analyse_step13b.py`; to
0.005 in `ridge_vs_mlp_v2.py`, whose ridge the Table S10 script later matched to 1e-4). Another assignment
would have stopped them. That assignment is written out in `pathway_vs_fingerprint_v2.py` and stored in its
record.

The reference moves with the assignment. In the Note 1 record (`leakage_results/pathway_vs_fingerprint_v2.json`)
its response correlation on sci-Plex (median over conditions) is 0.188, 0.247 and 0.082 on Splits 1, 0 and 2 with the
folds of the paper. Over 200 random assignments the median is 0.177, 0.230 and 0.097, and the 2.5th to
97.5th percentile is 0.149–0.211, 0.203–0.257 and 0.073–0.118. The intervals of the paper do not include this source of variation.

Step 29 asks: does any statement of the paper that compares something with the cross-validated ridge depend on
the fold assignment?

## 2. Fold assignments

- **Of the paper.** sci-Plex: the assignment stored in the Note 1 record. Tahoe-100M: not yet known; it is
  identified first (section 7, guard 4).
- **Random.** 200 per split, by the rule and with the seeds of the Note 1 record (seeds
  20261113 to 20261312). The compounds are put in random order
  (`numpy.random.RandomState(seed).permutation`), sorted by their number of conditions, largest first, with a
  stable sort, and each is given to the fold that holds the fewest conditions so far; five folds. This is how
  `GroupKFold` builds its folds, with a random order in place of the order of ties. On sci-Plex these are the
  200 assignments already stored in the Note 1 record. On Tahoe-100M the same rule and seeds are
  applied to the 75 held-out structures of each split.
- **Fewer on Tahoe-100M only if needed.** If a timing test shows that 200 assignments would take more
  than 24 hours, 100 are used (seeds 20261113 to 20261212). The number is written here as a dated
  line before the job is submitted.
- **Feedforward network (item A7).** The first 50 assignments (seeds 20261113 to 20261162), because
  every assignment needs 45 networks to be trained (three splits, three seeds, five folds).

## 3. What is recomputed

For every assignment only the cross-validated ridge is refitted, by the procedure of the paper (same features,
penalty 1, same arithmetic). This gives a predicted change for every held-out condition, and from it: gene
identity, response correlation (mean and median over conditions) and response-magnitude agreement; on Tahoe-100M
also population distance (the vehicle cells shifted by the predicted change) and the discrimination score.

The generative models, the training-set ridge and the compound-blind average do not depend on the folds. Their
values are read from the committed per-condition records. The feedforward network of Results 3 is itself fitted
with the folds, so it is refitted together with the ridge (width 64, three seeds, weight decay 10, 400 epochs, as
in `ridge_vs_mlp_v2.json`).

Intervals are those of the paper: 97.5 % percentile bootstrap per split, 5,000 replicates, resampling held-out
compounds (structures on Tahoe-100M). They are drawn with a fresh `RandomState(0)` for every contrast and every
assignment, so that all assignments meet the same resamples. A committed interval was drawn from a generator
shared between contrasts and can differ from the recomputed one in the third decimal (as recorded in
`step28_sciplex.json`); both are printed for the folds of the paper.

## 4. Statements examined

Each item is judged split by split. Its **verdict** is what the paper states: a direction (above or below) and,
where the paper gives an interval, whether the interval excludes zero. Values of table rows are given for Splits
1, 0 and 2; an asterisk marks an interval that excludes zero.

### A. sci-Plex (443, 444 and 455 conditions of 37, 37 and 38 held-out compounds on Splits 1, 0 and 2)

| Item | Statement of the paper | What is counted |
|---|---|---|
| A1 | Results 2: “The best model reached 0.61 on every split, above both the cross-validated ridge (0.55–0.57) and the compound-blind average (0.53–0.55).” The same claim is made in the Abstract, the Introduction, Results 4, the Discussion and Supplementary Note 7. Table S5c, rewired model − cross-validated ridge: `+0.056 [+0.035, +0.072]*; +0.044 [+0.034, +0.054]*; +0.057 [+0.044, +0.070]*` | Gene identity, rewired model minus ridge: positive, with the interval above zero. |
| A2 | Supplementary Note 7: “The rewired model’s gene identity remained 0.61 on both splits, above the cross-validated ridge (by 0.045 and 0.056) and the compound-blind average (by 0.059 and 0.077), each interval excluding zero.” | The same on Splits 0 and 2 without the 12 conditions of the compound whose salt twin is in training: positive, with the interval above zero. |
| A3 | Results 1: “This was below the compound-blind per-line average (0.53–0.55) and below a ridge regression on the compound's chemical fingerprint (a vector of bits that records which substructures the molecule contains, the same encoding the model receives), the cell line and their interaction (0.55–0.57), and response correlation was close to zero.” Restated in the Discussion. | Gene identity, original model minus ridge: negative. |
| A4 | Results 2: “On the correlation between predicted and measured change across the 50 most responsive genes, averaged over conditions, every generative model on sci-Plex scored below both ridges and below the compound-blind average (Supplementary Table S6).” | Response correlation, mean over conditions: the highest value among the generative models of Table S6 is below the ridge. |
| A5 | Caption of Table S6: “On the median over conditions the same holds on splits 1 and 0, but on split 2 the model with cross-attention and its parameter-matched control lie above the three references.” | Response correlation, median over conditions. Splits 1 and 0: the highest value among the generative models is below the ridge. Split 2: the model with cross-attention and its control are both above the ridge. |
| A6 | Supplementary Note 4: “On sci-Plex the rewired model reached 0.24–0.33, but there the linear references also had positive estimates, resolved on one split, and the rewired model did not differ from the training-set ridge.” Table S16b, cross-validated ridge: `+0.278 [+0.031, +0.481]*; +0.127 [−0.020, +0.262]; +0.239 [−0.008, +0.451]` | Response-magnitude agreement of the ridge: positive on each split; interval above zero on Split 1; interval includes zero on Splits 0 and 2. |
| A7 | Results 3: “Nor was the information itself the limit: a feedforward network given exactly the features of the linear references (fingerprint, cell line and their interaction), fitted within the held-out compounds like the cross-validated ridge, reached a response correlation of 0.11–0.21 against 0.08–0.25 for the ridge, close to the ridge on two splits and below it on one (exploratory; weight decay, training length and width chosen on the development split; Supplementary Table S10).” | Response correlation (median over conditions), network minus ridge, both fitted with the same folds. With the folds of the paper the difference is +0.005, −0.036 and +0.027 on Splits 1, 0 and 2 (`ridge_vs_mlp_v2.json`). “Below it on one”: negative on Split 0. “Close to the ridge on two splits” has no threshold in the paper and gets no verdict; the range of the difference on Splits 1 and 2 is reported. |

### B. Tahoe-100M (three splits, 75 held-out structures each)

| Item | Statement of the paper | What is counted |
|---|---|---|
| B1 | Results 2: “On Tahoe, the compound-blind average reached a gene-identity score of 0.50–0.51, above both the best model of this study (0.46; Section 4) and the cross-validated ridge (0.47–0.48) on all three splits, and level with the replicate (0.48–0.51) (Fig. 2a; Supplementary Table S5).” | Gene identity, compound-blind average minus ridge: positive. |
| B2 | Results 1: “Gene identity was 0.27–0.29 across splits: above the 0.14 expected for 50 genes chosen at random from 200, but below every reference (Fig. 1b; Section 2).” Restated in the Discussion. | Gene identity, original model minus ridge: negative. |
| B3 | Results 5: “The ridge regressions scored 0.49–0.54, not different from chance (Fig. 6a; Supplementary Table S15).” Restated in the Abstract and the Discussion. Table S15b, cross-validated ridge − 0.5: `+0.004 [−0.055, +0.062]; −0.008 [−0.057, +0.044]; −0.014 [−0.071, +0.042]` | Discrimination score of the ridge minus 0.5: interval includes zero. |
| B4 | Supplementary Note 4: “The original model did not rank conditions (−0.11 to −0.06), the ridges did so on at most one split, and the line-mean prediction, whose change has one size per plate, did not (−0.04 to +0.04) (Supplementary Fig. S2; Supplementary Table S16).” Table S16a, cross-validated ridge: `−0.050 [−0.202, +0.109]; +0.004 [−0.167, +0.154]; +0.094 [−0.046, +0.228]` | Response-magnitude agreement of the ridge: interval includes zero. |

### C. Rows of the supplementary tables that no sentence repeats

| Item | Row of the paper | What is counted |
|---|---|---|
| C1 | Table S5c, gene identity on Tahoe-100M, rewired model − cross-validated ridge: `−0.013 [−0.027, +0.001]; −0.005 [−0.014, +0.005]; −0.009 [−0.019, +0.001]` | Interval includes zero. |
| C2 | Table S5c, population distance on Tahoe-100M, rewired model − cross-validated ridge: `+0.065 [+0.020, +0.114]*; +0.021 [−0.060, +0.079]; +0.056 [+0.018, +0.095]*` | Positive; interval above zero on Splits 1 and 2; interval includes zero on Split 0. |
| C3 | Table S15b, discrimination score, rewired model − cross-validated ridge: `+0.218 [+0.149, +0.289]*; +0.227 [+0.170, +0.282]*; +0.228 [+0.165, +0.289]*` | Positive, with the interval above zero. |
| C4 | Table S16b, response magnitude on sci-Plex, rewired model − cross-validated ridge: `+0.056 [−0.170, +0.278]; +0.114 [−0.039, +0.283]; +0.031 [−0.128, +0.192]` | Interval includes zero. |

The values of the ridge itself in Tables S5a, S5b, S6, S10e, S15a and S16 (parts a and b) are reported with their
range (section 6).

## 5. Rule

An item is **robust** on a split if at least 95 % of the random assignments give the verdict of the paper: at
least 190 of 200 (95 of 100; 48 of 50 for A7). An item is robust if it is robust on every split it names. The
assignment of the paper is reported beside the random ones and is not counted among them.

The threshold of 95 % is a convention fixed here. It is not an error rate.

## 6. What follows

- **In every case** the fold assignment of the paper stays the reported one. No assignment is chosen for its
  result, and no value of the paper is replaced by a value from another assignment. Methods states that the folds
  are fixed and listed in the code, and why. One new supplementary table gives, for every item and for every value
  of the ridge named in section 4: the value with the folds of the paper, the median and the 2.5th to 97.5th
  percentile over the random assignments, the number of assignments at or below the value of the paper, and the
  number of assignments that give the verdict of the paper.
- **A sentence (items A and B) that is robust** stays as it is.
- **A sentence that is not robust** is reworded to say what was found, with the count. If A1 is not robust, the
  Abstract, the Introduction, Results 2 and 4, the Discussion and Supplementary Note 7 are changed together, and
  the change log records it as a change of a headline result.
- **A table row (items C) that is not robust** gets a footnote with the count.
- **A7:** the sentence of Results 3 is completed with the range of the difference, whatever it is.

## 7. Guards (before any Step 29 value is printed)

1. This file, the scorers and every record they read are committed and unchanged; no output exists.
2. sci-Plex: the 200 assignments built by the scorer equal those stored in the Note 1 record.
3. sci-Plex, folds of the paper. The ridge reproduces the Step 7 record exactly; the per-condition gene identity
   and response correlation of the cross-validated ridge in `leakage_results/step14_per_condition.csv` to 1e-9;
   its response-magnitude agreement in `step17_mra.json` to 1e-9; and the contrasts of Supplementary Note 7 in the
   Step 28 record to 1e-9. The means of every model in the per-condition record reproduce `step14_analysis.json`
   to 1e-9. The network reproduces `ridge_vs_mlp_v2.json` to 0.005 (the arithmetic of the training library can
   differ between nodes); the difference is printed.
4. Tahoe-100M, the folds of the paper (a separate job that only reads). The assignment is the one that reproduces
   the committed per-condition gene identity and response correlation of the cross-validated ridge
   (`tahoe_results/step15_model_per_condition.csv`) to 1e-9. The candidates are the three orders of ties found on
   sci-Plex: numpy's default sort without AVX-512, the same with AVX-512, and a stable sort. If none reproduces
   the record, the step stops and this file is amended before anything else is read. The assignment found is
   written to `tahoe_results/step29_tahoe_folds.json`.
5. Tahoe-100M, folds of the paper. Population distance, discrimination score and response-magnitude agreement of
   the cross-validated ridge reproduce the committed records of Steps 15b, 15c and 16 to 1e-9.
6. If a guard fails, nothing is written.

## 8. What is already known, and expectations

Written before any Step 29 value exists.

- **Already known.** The spread of the ridge's response correlation on sci-Plex (median over conditions) is in the
  Note 1 record (section 1). With the folds of the paper the ridge lies in the upper part of that spread on
  Splits 1 and 0 and in the lower part on Split 2: 152, 167 and 25 of the 200 random assignments
  give a value at or below it. The outcome of A5 and the ridge side of A7 can be worked out from that record.
  These two are listed for completeness and are not blind.
- **Not known.** Gene identity, the mean over conditions, response magnitude, the network under other folds, and
  everything on Tahoe-100M.
- **Expectations.** A1 to A3, B1 and B2 keep their direction. A4 holds on Splits 1 and 0; on Split 2 the
  difference is small and the outcome is open. A5 is not robust on Split 2. A6 and C1, whose intervals have a bound
  close to zero, may not be robust. B3 is robust.

## 9. Scripts, records, order

1. `score_step29_sciplex.py` with `run_step29_sciplex.sh`: items A1 to A7 and C4. Writes
   `leakage_results/step29_fold_sensitivity_sciplex.json` and `step29_fold_sensitivity_sciplex_per_assignment.csv`.
2. `diag_step29_tahoe_folds.py` with its job file: guard 4.
3. `score_step29_tahoe.py` with `run_step29_tahoe.sh`: items B1 to B4 and C1 to C3. Writes
   `tahoe_results/step29_fold_sensitivity_tahoe.json` and `step29_fold_sensitivity_tahoe_per_assignment.csv`. Its
   details (timing, number of assignments) are added here as a dated paragraph before it is submitted.
4. `step29_findings.md`; the new supplementary table, built by a table script from the two records; the Methods
   sentence; and the rewording that section 6 requires. Text is changed only with the author's approval.

Every script refuses to run if a tracked file is modified or an output exists, and prints every number it uses.

## 10. Limits

- One rule for the random assignments: balanced folds of five, as `GroupKFold` makes them. Other numbers of folds
  and unbalanced folds are not examined.
- The penalty stays 1. Step 29 says nothing about a tuned ridge.
- Not recomputed: the rows of the cross-validated ridge in Table S17 (spread) and Table S18 (recognition by an
  external method, which would need the external software to be run once per assignment), the rows by dose and
  for the conditions with a plate replicate in Table S16, and the fractions of entries set to zero in Table S9.
  No sentence of the paper rests on them: the range of fractions quoted in Results 2 is set by the other two
  references, between which the cross-validated ridge lies on every split of both datasets. Each of these rows
  gets a footnote that it holds for the fold assignment of the paper.
- Population distance is recomputed on Tahoe-100M only. For sci-Plex the paper reports no population distance
  of the references.
- Supplementary Note 2 is not written yet. Its values of the cross-validated ridge are for the fold assignment of
  the paper, and the note will say so.
- The network is refitted for 50 assignments, not 200.

## Dated paragraph, 15 Nov 2026 (written before any Step 29 value was printed)

Two runs of `score_step29_sciplex.py` (commit `ce1ed24`) stopped at guard 3f and wrote nothing. No value of
section 4 was printed in either. The first, in an interactive session on cn274 (Intel Xeon Gold 6548N), is not
used. The second, job 19381505 on cn224 (AMD EPYC 7742, 8 threads), passed guards 1, 2 and 3a to 3e: the Step 7
record with a difference of zero on the three splits, and the per-condition rows of the ridge to 1.1e-16 or
better. Guard 3f:

- Split 1: network 0.192102, record 0.192911, difference -0.000809; per seed 0.1918, 0.1980, 0.1865 against 0.1918, 0.1980, 0.1890.
- Split 0: network 0.211194, record 0.211194, difference +0.000000; per seed 0.1927, 0.2314, 0.2095 against 0.1927, 0.2314, 0.2095.
- Split 2: network 0.101199, record 0.108165, difference -0.006967; per seed 0.1024, 0.1066, 0.0946 against 0.1025, 0.1149, 0.1072.

The trainings of Split 0 are identical to the record in all three seeds, as are two seeds of Split 1. So the
folds and their numbering are those of the record. The record was written with 16 threads of the training
library and this run used 8; that is the one known difference, and a training that leaves the path of the
record ends at another value. Change: the network of guard 3f is trained with 16 threads, set in the script
(`NET_THREADS`). The ridge keeps the 8 threads of the job that wrote the per-condition record. Nothing else
changes: tolerance 0.005, items, rule and assignments as above. If guard 3f still fails with 16 threads, the run
stops again and a further dated paragraph follows before anything else is done.

Consequence for item A7, stated now: the networks of the random assignments are trained on one thread each, so
a single training can differ from what 16 threads would give by an amount of this size (largest seen in one
seed: about 0.013). Item A7 is read with that in mind, and the per-assignment table shows how close each
difference is to zero.

## Dated paragraph, 16 Nov 2026 (written before any Step 29 value of Tahoe-100M was computed)

**Guard 4 is met.** `diag_step29_tahoe_folds.py` (commit `4796dc2`), run on cn274 (Intel Xeon Gold 6548N, with AVX-512,
so that all three candidate orders could be made), found one candidate on each split that reproduces the per-condition
record of the cross-validated ridge: numpy's default sort without AVX-512. Largest differences: 5.6e-17 for gene
identity and 3.0e-12, 6.6e-12 and 6.3e-12 for response correlation on Splits 1, 0 and 2; no condition differs. The
other two candidates differ on every condition. The assignment is stored in `tahoe_results/step29_tahoe_folds.json`
(sha256 `a6e36a3f1de054704526ec0ad10b92c83226765e65ea244b93db645fee7bdd82`, commit `00f61e7`). No Step 29 quantity of
section 4 was computed by that script.

**Number of assignments: 200.** In the guard-4 run one refit of the ridge on one split (five fits) with the measures
of that script took about 7 seconds on 8 threads. With the further measures of the Tahoe scorer, 200 assignments on
three splits stay far below 24 hours. The reduction to 100 is not used.

**How `score_step29_tahoe.py` runs** (details of section 9, item 3):

- The folds of the paper are scored in the main process with 8 linear-algebra threads, and guard 5 is applied to that
  run. The random assignments are scored in worker processes with one thread each; their values can differ from an
  8-thread run in the last digits (order 1e-12), which cannot change a count.
- The values of the rewired model, the original model and the compound-blind average are read from the committed
  per-condition records of Steps 15b and 15c; they are not recomputed. Guard 5 covers the ridge: its per-condition
  gene identity, response correlation and population distance (Step 15b), its per-condition discrimination score
  (Step 15c), and its response-magnitude agreement with its interval (Step 16), each to 1e-9.
- Population distance of the ridge: the term that involves only the measured cells does not depend on the folds and is
  computed once per condition.
- The interval of response-magnitude agreement was drawn from a fresh generator in the paper as well, so it must be
  reproduced exactly (it is part of guard 5). The intervals of gene identity, population distance and the
  discrimination score were drawn in the paper from a generator shared between contrasts; as fixed in section 3, they
  are recomputed from a fresh `RandomState(0)`, and the committed interval is printed beside the recomputed one.
- If, with the folds of the paper and the fresh generator, an item does not give the verdict the paper states (possible
  where a bound of the committed interval is within about 0.001 of zero, as for item C1 on Splits 1 and 2), the run
  does not stop: this is printed and recorded, and the count is still the number of random assignments that give the
  verdict the paper states. It is reported in the findings.
- The job runs on the standard compute nodes (AMD EPYC 7742, no AVX-512), where the records of the paper were written.
