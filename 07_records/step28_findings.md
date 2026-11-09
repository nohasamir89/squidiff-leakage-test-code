# Step 28 findings — sensitivity of the reported results (Tahoe-100M and sci-Plex) to held-out compounds with a salt/hydrate twin in training

Pre-registration `PREREG_step28_salt_twins.md` (committed `ac52736`, amendments 10–16; 15–16 cover sci-Plex).
Scorer, runner and synthetic test committed `c56f235`; gate-1 ALL-row fix `24b2ec2`; resolver fix `2e7164e`. Real run job 19240209 at commit
`2e7164e`, 119.9 s, node cn035. Records: `tahoe_results/step28_salt_twins.json`, `_per_quantity.csv`, `_summary.csv`;
log `tahoe_results/step28_logs/s28score_19240209.log`. Evaluation only: no model run, no cell re-sampled; every
number is a re-aggregation of committed per-condition records with the affected conditions removed. Sections
"Verdict" to "Expectations that were wrong" are Tahoe-100M; the sci-Plex arm (group B) has its own section below.

## Verdict — Tahoe-100M

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

## sci-Plex (group B; amendments 15–16)

**Question.** The paper's one positive, compound-dependent result is sci-Plex gene identity (the rewired model 0.61
on every split, above every linear reference). On sci-Plex one held-out compound on split 0 and one on split 2
(tofacitinib, 12 conditions each) have their parent molecule in training; split 1 has none. Does removing them
change a reported value or ordering, and did the model score that compound better than the others?

**Provenance.**
- `score_step14.py` wrote no per-condition record. An additive patch (`patch_step14_add_percond.py`, md5
  `d017464193ee44d873bc495ec8d0afd6`; amendment 15; commit `e4df318`) added `leakage_results/step14_per_condition.csv`
  (per condition and submission: the seed-averaged gene identity, energy and response correlation, with the
  condition's cell line and structure = test-file SMILES), and `score_step14.py` was re-run once for it (job 19288685,
  `run_step14_rescore.sh`).
- That re-run's gate was written as "`step14_long.csv` byte-identical to its committed version". It was not: the
  largest difference was 4.6e-13, floating-point noise from multi-threaded linear algebra in the ridge fits.
  Amendment 16(a) restated the gate as agreement to 1e-9 **after the re-run had been seen** — a post hoc change,
  declared here. The committed `step14_long.csv` and `step14_analysis.json` were restored to their committed
  versions; only the per-condition record is used from the re-run.
- Scorer `score_step28_sciplex.py` (md5 `ee84e35f80bf3f1a1f5ab43fcd363cd2`) and runner `run_step28_sciplex.sh` (md5
  `f193e372a6362ab24f075c8c94f27512`), committed with amendment 16(b) in `e42c8c3`; real run job 19289646 at
  `e42c8c3`. Records: `leakage_results/step28_sciplex.json`, `leakage_results/step28_sciplex_per_quantity.csv`; log
  `tahoe_results/step28_logs/s28sciplex_19289646.log`.

**Gates.**
- Gate 1 (aggregation): every committed sci-Plex thesis metric (gene identity, energy, response correlation mean
  and median) and every committed contrast point of `step14_analysis.json` was recomputed from the per-condition
  record to 1e-9 — 63 / 47 / 47 values on splits 1 / 0 / 2. Comparator, read from the record: `fm_50`, the rewired
  model; gene identity 0.6100 / 0.6124 / 0.6106.
- Interval replay (descriptive, amendment 16(b)): the committed Step 14 contrast intervals and this scorer's
  full-set recomputation differ by at most 0.00239 over 21 contrasts, because `score_step14.py` draws all its
  contrasts from one shared random generator. Every interval below is this scorer's (fresh `RandomState(0)` per
  set, the same for the full, retained and excluded sets), so all-compounds and twin-free are compared like for like.

**Exclusions.** Exact-SMILES match to the overlap record; exactly 12 conditions per compound, as required.

| split | conditions | compounds | excluded | retained |
|---|---|---|---|---|
| 1 (dev) | 443 | 37 | none (free of the route by construction) | 443 |
| 0 | 444 | 37 | 1 compound, 12 conditions (2.7 %) | 432 |
| 2 | 455 | 38 | 1 compound, 12 conditions (2.6 %) | 443 |

**Verdict K: K3 A CONCLUSION CHANGES** — 1 interval crossing, 0 contrast sign reversals, 0 values moved by more than
half the published interval width. Reported literally:

- **The crossing.** Split 2, response correlation, rewired minus the compound-blind average (`mean_response`):
  −0.0619 [−0.1244, −0.0032] with all compounds → −0.0511 [−0.1094, +0.0043] twin-free; it now includes zero. Both
  bounds moved up, towards the model. The route would push the other way (a compound inflated by its twin, once
  removed, lowers the model's mean), so this crossing is not the route operating.
- **It touches a stated sentence**, unlike the four Tahoe crossings: the Discussion says that on sci-Plex the
  compound-blind average has the higher response correlation. That still holds on every split (twin-free −0.0511 on
  split 2); Supplementary Note 7 now reports it with this interval.
- **A separate imprecision found while checking that sentence.** As first written ("it outscored every generative
  model on response correlation") it is true for the condition mean but not for the condition median on split 2,
  where cross-attention (0.0979) and matched control (0.0967) are above the compound-blind average (0.0858) (Step 14
  record). The Discussion now reads "its response correlation, averaged over conditions, was higher than that of
  every generative model" (manuscript rev 167); dated corrections were added to step14_findings §5 and §10.

**All-compounds and twin-free values, splits 0 and 2** (97.5 % structure-clustered intervals; split 1 is unchanged
by construction; contrasts against control-paste are undefined, it has no top-50 set)

| split | quantity | rewired, or rewired minus … | all compounds | twin-free |
|---|---|---|---|---|
| 0 | gene identity | rewired | 0.6124 [0.6061, 0.6192] | 0.6125 [0.6060, 0.6194] |
| 0 | response correlation | rewired | 0.0698 [0.0253, 0.1157] | 0.0736 [0.0256, 0.1189] |
| 2 | gene identity | rewired | 0.6106 [0.6011, 0.6199] | 0.6095 [0.6001, 0.6189] |
| 2 | response correlation | rewired | 0.0691 [0.0000, 0.1536] | 0.0689 [−0.0018, 0.1525] |
| 0 | gene identity | − original | +0.1077 [+0.0972, +0.1188] | +0.1081 [+0.0975, +0.1193] |
| 0 | gene identity | − training-set ridge | +0.0755 [+0.0655, +0.0854] | +0.0768 [+0.0665, +0.0863] |
| 0 | gene identity | − cross-validated ridge | +0.0439 [+0.0338, +0.0535] | +0.0449 [+0.0347, +0.0544] |
| 0 | gene identity | − compound-blind average | +0.0586 [+0.0521, +0.0651] | +0.0588 [+0.0523, +0.0655] |
| 0 | response correlation | − original | +0.0601 [+0.0328, +0.0872] | +0.0624 [+0.0342, +0.0898] |
| 0 | response correlation | − training-set ridge | −0.0536 [−0.1029, −0.0057] | −0.0544 [−0.1029, −0.0064] |
| 0 | response correlation | − cross-validated ridge | −0.1325 [−0.2006, −0.0604] | −0.1254 [−0.1948, −0.0529] |
| 0 | response correlation | − compound-blind average | −0.1807 [−0.2474, −0.1160] | −0.1732 [−0.2411, −0.1074] |
| 2 | gene identity | − original | +0.1198 [+0.1073, +0.1321] | +0.1185 [+0.1061, +0.1313] |
| 2 | gene identity | − training-set ridge | +0.0633 [+0.0490, +0.0768] | +0.0626 [+0.0487, +0.0765] |
| 2 | gene identity | − cross-validated ridge | +0.0570 [+0.0445, +0.0692] | +0.0563 [+0.0438, +0.0685] |
| 2 | gene identity | − compound-blind average | +0.0781 [+0.0666, +0.0893] | +0.0773 [+0.0657, +0.0888] |
| 2 | response correlation | − original | +0.0183 [−0.0064, +0.0450] | +0.0196 [−0.0052, +0.0452] |
| 2 | response correlation | − training-set ridge | −0.0520 [−0.1024, −0.0032] | −0.0602 [−0.1069, −0.0151] |
| 2 | response correlation | − cross-validated ridge | −0.0291 [−0.0996, +0.0439] | −0.0477 [−0.1092, +0.0085] |
| 2 | response correlation | − compound-blind average | −0.0619 [−0.1244, −0.0032] | −0.0511 [−0.1094, +0.0043] **← the crossing** |

Gene identity: every contrast excludes zero with all compounds and twin-free, and nothing moved by more than
0.0013. The ordering the paper states — the rewired model above the original model, both ridges and the
compound-blind average — holds on both splits without the compound.

One descriptive boundary outside the verdict-K rule (which covers contrasts only): the rewired model's own
response correlation on split 2 has a lower bound of 0.0000 with all compounds and −0.0018 twin-free. The manuscript
states nothing about it.

**Verdict L — gene identity, excluded minus retained** (pre-registered)

| arm | split 0 | split 2 |
|---|---|---|
| rewired (`fm_50`) | −0.0056 | +0.0401 |
| original (`base`) | +0.0112 | −0.0072 |
| training-set ridge (`ridge_train`) | +0.0439 | +0.0137 |
| cross-validated ridge (`ridge_S7`) | +0.0302 | +0.0127 |
| compound-blind average (`mean_response`) | +0.0042 | +0.0093 |

On split 2 the rewired model scored the excluded compound 0.040 above the retained ones — the direction the route
would produce; on split 0 it scored it 0.006 below. One compound is one bootstrap cluster, so the structure-clustered
interval cannot test either difference. Its effect on the headline is 0.0011 (split 2 gene identity 0.6106 →
0.6095).

**Post hoc context** (added after this result was seen; not pre-registered). Per-compound gene identity (mean over
the compound's conditions): the excluded compound among all held-out compounds of its split. First printed by an
inline diagnostic on 2026-11-09 (numbers below); recorded by `diag_step28_sciplex_rank.py` in
`leakage_results/step28_sciplex_posthoc_rank.json`, which must reproduce these printed values before it writes.

| split | quantity | excluded | compounds: mean (sd) | rank (1 = highest) |
|---|---|---|---|---|
| 0 | rewired | 0.6069 | 0.6124 (0.0179) | 24 of 37 |
| 0 | compound-blind average | 0.5579 | 0.5538 (0.0127) | 14 of 37 |
| 0 | rewired − compound-blind | +0.0491 | +0.0586 (0.0181) | 26 of 37 |
| 2 | rewired | 0.6496 | 0.6106 (0.0267) | 4 of 38 |
| 2 | compound-blind average | 0.5415 | 0.5324 (0.0273) | 9 of 38 |
| 2 | rewired − compound-blind | +0.1081 | +0.0782 (0.0323) | 7 of 38 |

Reading. Part of the split-2 difference belongs to the compound, not the model: the compound-blind average, which
carries no compound information and cannot use a twin, also places it above most compounds (9th of 38; verdict L
+0.0093). The rewired model's own gain over that average on this compound (+0.108) is about one standard deviation
above the typical compound and is exceeded by six of the 37 other compounds, none of which has a twin in training.
On split 0, where a form of the same molecule is also held out with its twin in training, it ranks 26th of 37 on the
same gain. One compound per split cannot exclude a small effect of the route; what the data show is that the
split-2 value is ordinary among compounds without a twin and is not repeated on split 0.

**Expectations (§6), checked against the result**
- "Verdict K outcome 1 (NO CHANGE) — confident": wrong again — K3, from one response-correlation interval that moved
  in the model's favour.
- "sci-Plex gene identity of the rewired model: within 0.01 of 0.61 on splits 0 and 2 — moderately confident": right
  (0.6125, 0.6095).
- "Verdict L: the excluded compounds score no better than the retained ones on sci-Plex gene identity — moderate
  confidence only … a positive difference would not be surprising; if it appears without changing verdict K, it is
  reported as a measured effect that is too small to matter at this scale": right on split 0 (−0.0056), wrong on
  split 2 (+0.0401). Verdict K did change, but through response correlation, not gene identity (every gene-identity
  contrast moved by ≤ 0.0013). The +0.040 is reported as a measured effect in Supplementary Note 7; it moves the
  split-2 headline by 0.0011.

## Synthetic-test history (all committed with the scorer, `c56f235`)

- 19237778 — login-node reaping of the subprocess during a slow scipy/anndata import; moved to a compute node.
- 19238263 — exposed that `score_step15_tahoe.boot` returns `(lo, hi)`, not `(point, lo, hi)`.
- 19238650 — exposed the shared module-level generator (gate-2 differences of 0.3–3 % of the interval width);
  diagnosed and reset per call, tolerance restored to 5e-6.
- 19239021, 19239640 — 8 of 8 (three scenarios K1 / K3 / K3; five guards each stopping on their own FATAL).
