# Step 28 findings — sensitivity of the reported Tahoe-100M results to held-out compounds with a salt/hydrate twin in training

Pre-registration `PREREG_step28_salt_twins.md` (committed `ac52736`, amendments 10–14). Scorer, runner and synthetic
test committed `c56f235`; gate-1 ALL-row fix `24b2ec2`; resolver fix `2e7164e`. Real run job 19240209 at commit
`2e7164e`, 119.9 s, node cn035. Records: `tahoe_results/step28_salt_twins.json`, `_per_quantity.csv`, `_summary.csv`;
log `tahoe_results/step28_logs/s28score_19240209.log`. Evaluation only: no model run, no cell re-sampled; every
number is a re-aggregation of committed per-condition records with the affected conditions removed.

## Verdict

**K3 A CONCLUSION CHANGES**, by the pre-registered rule (§5): 4 interval crossings, 0 contrast sign reversals, 0
point estimates moved by more than half the published interval width. Read literally, and reported literally here:
the four crossings are described in full below, and **none of them is a conclusion the manuscript states**. Every
ordering the manuscript asserts holds on all three splits; the four crossings are intervals that already touched
their reference in the published analysis and tipped across it as the bootstrap clusters thinned by 6–14 %.

**L (did the excluded compounds score better?)** — No. If a salt twin in training inflated the model, the excluded
compounds should score above the retained ones. On cross-plate discrimination the rewired model scored the excluded
compounds **worse** on two of three splits and near chance on the third (excluded − retained −0.0476, −0.0557,
+0.0187 on splits 1 / 0 / 2); on gene identity −0.0175, +0.0117, −0.0048. The leakage route the analysis was built
to detect is not operating.

## Reproduction (the gates that make the re-aggregation trustworthy)

- **Gate 1 (aggregation).** 1,125 published values reproduced to 1e-9 — every `(split, line, submission, metric)`
  row of `step15_model_long.csv`, including all 81 split-level ALL rows (the manuscript's headline gene identity,
  population distance and response correlation). 9 long-record rows had no per-condition column; 0 lines were
  absent from the per-condition record. The published numbers are therefore a mean over conditions, and this script
  reads the right column for each quantity.
- **Gate 2 (interval equivalence).** `score_step23_screen.Boot` + percentiles equals `score_step15_tahoe.boot`
  (with `s15.rng` reset to `RandomState(0)` per call) to ≤ 4.4e-16 — machine precision — on all three splits. The
  two bootstrap implementations are identical once the shared module-level generator is reset (amendments 11–12).

## Exclusions

| split | conditions | excluded structures | excluded conditions | retained |
|---|---|---|---|---|
| 1 (dev) | 2,899 | 8 | 384 (13.2 %) | 2,515 |
| 0 | 3,115 | 4 | 178 (5.7 %) | 2,937 |
| 2 | 2,853 | 9 | 406 (14.2 %) | 2,447 |

Matched through `table_smiles` (the form the per-condition `structure` column carries; amendment 14). The excluded
structures carry more conditions than the split average (~48 vs ~39), which is why the condition fraction exceeds
the structure fraction.

## The four interval crossings (verdict K3), in full

1. **Split 0, cross-plate discrimination, `recon_recipe` (linemean scheme), vs chance.** [0.4976, 0.6116] →
   [0.5009, 0.6223]; lower bound moved 2 thousandths above 0.5. A reconstruction control arm, not a reported
   quantity.
2. **Split 0, cross-plate discrimination, `ctrl_sample` (flip scheme), vs chance.** [0.4737, 0.4983] →
   [0.4738, 0.5003]; upper bound moved 2 thousandths above 0.5. A no-model control arm.
3. **Split 2, gene identity, rewired − cross-validated ridge, vs zero.** [−0.0187, +0.0003] → [−0.0218, −0.0014];
   it was sitting on zero (upper bound +0.0003) and now excludes it, slightly below. The cross-validated ridge is
   an upper reference (it uses the held-out responses), which the manuscript does not claim the model beats; the
   crossing makes the model mildly more clearly below it, overturning no stated claim.
4. **Split 2, population distance, rewired − unaltered vehicle cells, vs zero.** [+0.0033, +0.0791] →
   [−0.0113, +0.0699]; it was sitting on zero (lower bound +0.0033) and now includes it. The manuscript states the
   rewired model's populations are no closer than the vehicle cells "and farther on one split"; this crossing
   softens that negative-for-the-model result on split 2, overturning no stated claim.

All four are confirmation-split arms (0 and 2), none on the development split; all four had a published interval
whose nearer bound was within ±0.003 of the reference; the point estimates moved by 0.001–0.015.

## The stated orderings, twin-free (all hold on all three splits; splits 1 / 0 / 2)

- **Rewired > original (gene identity):** +0.1990 [+0.188, +0.210], +0.1753 [+0.165, +0.187], +0.1681
  [+0.157, +0.179] — all exclude zero.
- **Rewired > training-set ridge (gene identity, the fair linear baseline):** +0.0375 [+0.027, +0.049],
  +0.0265 [+0.017, +0.035], +0.0353 [+0.024, +0.046] — all exclude zero.
- **Rewired < compound-blind average on Tahoe (gene identity, the honest negative result):** −0.0426
  [−0.050, −0.035], −0.0401 [−0.048, −0.032], −0.0502 [−0.059, −0.042] — all exclude zero.
- **Response correlation, rewired > original:** +0.307 / +0.256 / +0.260 — all exclude zero.

The published point estimates and their twin-free counterparts differ by at most ~0.015, and the intervals overlap
throughout.

## Expectations that were wrong, recorded rather than adjusted

- §6 predicted verdict K1 with confidence. The rule returned K3. The prediction was wrong about the label but right
  about the substance: no stated conclusion changed, and the four crossings are boundary intervals. The lesson is
  that a criterion phrased as "any interval crosses" fires on arms that carry no claim; the finding is reported with
  that distinction made explicit rather than by relabelling the verdict.
- §8.2 predicted synthetic scenario (c) would give outcome 1 (K1). It gives K3, because a shift of equal size in
  either direction moves the retained mean; the verdict asks whether a reported value changes, not whether it
  improves. The synthetic test was amended to expect K3 (amendment 11) before the real run.

## Still to run

The sci-Plex arm (group B of §4 — gene identity on sci-Plex, the one positive compound-dependent result) is a
separate script, because `score_step14.py` writes no per-condition record; it re-runs that scorer with an exclusion
list. The substantive answer for sci-Plex does not depend on it: sci-Plex split 1 contains no held-out compound
whose parent molecule is in training, and the rewired model reaches 0.61 on that split, above every linear
reference, so the one positive compound-dependent result already holds on a split from which this route is absent
by construction (Supplementary Note 7). The group-B recompute will confirm the two affected splits numerically.

## Synthetic-test history (all committed with the scorer, `c56f235`)

- 19237778 — login-node reaping of the subprocess during a slow scipy/anndata import; moved to a compute node.
- 19238263 — exposed that `score_step15_tahoe.boot` returns `(lo, hi)`, not `(point, lo, hi)`.
- 19238650 — exposed the shared module-level generator (gate-2 differences of 0.3–3 % of the interval width);
  diagnosed and reset per call, tolerance restored to 5e-6.
- 19239021, 19239640 — 8 of 8 (three scenarios K1 / K3 / K3; five guards each stopping on their own FATAL).
