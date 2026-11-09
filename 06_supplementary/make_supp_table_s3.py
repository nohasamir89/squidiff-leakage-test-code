#!/usr/bin/env python3
"""
make_supp_table_s3.py - assemble Supplementary Table S3 (glioblastoma: gene-panel sweep and reproduction of the
published analysis) from the records on the corrected scale.

Purpose
-------
Supplementary Table S3 holds the values behind Fig. 1d (gene-panel sweep) and Fig. 1e (reproduction of the
published glioblastoma analysis), plus the vehicle-mean correlation quoted in Results 1. The earlier table was built
from the old-scale records (the first evaluation took log1p of the decoded values a second time). This script
rebuilds it from the re-evaluation of 24 Oct 2026 only, so that every value in the table comes from a record.

Logic
-----
1. Read the records (nothing else; no model, no data matrix):
     SWEEP   gene_sensitivity_SEEDED_v2.csv  one row per (panel size, cell type), 4 decimals; written by
             eval_gbm_sensitivity_v2.py via run_gbm_seeded_reeval_v2.sh (job 18932187), decoder seed 42.
     VALUES  figures/fig1e_values_v2.csv     per-compound and reconstruction correlations at 200 and 2,000 genes,
             parsed from the two evaluation logs by plot_fig1e_v2.R (research commit 7dbbcab).
     LOGS    eval_200_v2_18927698.out, eval_v2_18927697.out  the two evaluation logs themselves (3 decimals).
2. Check the records before anything is written (the script stops on the first failure):
     a. the sweep has exactly the ten sizes 100..1,000 x two cell types, no duplicates;
     b. recon_ok, which the evaluation writes once per size (1 only if both cell types reach 0.70) on both rows,
        is the same on the two rows and agrees with the two reconstruction values;
     c. BminusC_R_deg equals B_R_deg - C_R_deg within rounding of the three 4-decimal values;
     d. the zero fraction is the same for both cell types at each size (it is a property of the panel);
     e. the values file has 24 rows (2 sizes x [5 compounds x 2 cell types + 2 reconstructions]), every value in
        [0, 1] and at most 3 decimals, and every value equals the same number parsed from the log again here;
     f. the mean of the five compounds equals the mean the log printed, within 0.001 (rounding only);
     g. every value the manuscript quotes from these records (typed below in EXPECTED, as printed in the text on
        10 Nov 2026) is reproduced at its printed precision.
3. Write the table at the precision of the records: sweep values exactly as stored (4 decimals, copied as text, so
   nothing is re-rounded), log values with 3 decimals, means as printed in the logs. The only typed-in numbers
   are the two published values (He et al., Squidiff, Nat. Methods 2026: 0.912 tumour cells, 0.915
   oligodendrocytes), the same two plot_fig1e.R and plot_fig1e_v2.R use.
4. Outputs, relative to the directory the script is run from (the Squidiff_ms worktree root):
     supplementary_tables/supp_table_S3.csv   long format: part, n_genes, cell_type, quantity, value, source
     supplementary_tables/supp_table_S3.md    caption, three tables (a-c), notes and sources with sha256 (first 16 hex digits)

Run (login node is fine: standard library only)
---
    cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
    python3 make_supp_table_s3.py
"""
import csv
import hashlib
import os
import re
import sys

# ------------------------------------------------------------------ inputs (paths as on Jubail)
SWEEP = "/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/gene_sensitivity_SEEDED_v2.csv"
VALUES = "figures/fig1e_values_v2.csv"                      # relative to the worktree root (committed, 7dbbcab)
LOGS = {200: "/scratch/ni2071/Diffusion/gbm_run_200/logs/eval_200_v2_18927698.out",
        2000: "/scratch/ni2071/Diffusion/gbm_run/logs/eval_v2_18927697.out"}
OUT_DIR = "supplementary_tables"

SIZES = list(range(100, 1001, 100))                         # the ten panel sizes of the sweep
TYPES = ["tumor", "oligodendrocyte"]                        # cell-type keys as written in the records
TYPE_LABEL = {"tumor": "Tumour cells", "oligodendrocyte": "Oligodendrocytes"}
VALUES_TYPE = {"Tumour cells": "tumor", "Oligodendrocytes": "oligodendrocyte"}   # fig1e_values_v2.csv labels
COMPOUNDS = ["panobinostat", "RO4929097", "tazemetostat", "ispinesib", "ANA-12"]  # order of the logs
FLOOR = 0.70                                                # reconstruction gate used in the study
PUBLISHED = {"tumor": "0.912", "oligodendrocyte": "0.915"}  # He et al. 2026; the only typed-in numbers

# ------------------------------------------------------------------ values the manuscript quotes from these records
# Typed from the manuscript text as it stands on 10 Nov 2026 (Results 1, Methods, Supplementary Note 3 entry),
# each with the number of decimals printed there. Used ONLY as checks (step g); nothing here enters the table.
EXPECTED = {
    "interpreted_sizes": [100, 200, 600, 700, 800],          # Results 1: "Five sizes (100, 200, 600, 700 and 800)"
    "failing_sizes": [300, 400, 500, 900, 1000],             # Results 1: "Five sizes (300, 400, 500, 900 and 1,000)"
    "failing_minima": [("0.61", 300), ("0.42", 400), ("0.66", 500), ("0.45", 900)],   # "...was 0.61, 0.42, 0.66, 0.45"
    "failing_minimum_1000_abs_below": 0.05,                  # "...and approximately zero" (tolerance chosen here)
    "tumour_margin_range": ("0.06", "0.24"),                 # Results 1: "by 0.06 to 0.24"
    "zeros_percent_range": (93, 94),                         # Results 1: "93-94 % zeros"
    "vehicle_mean_all_200": {"tumor": "0.74", "oligodendrocyte": "0.93"},   # Results 1, added 10 Nov 2026
    "sweep_recon_200": {"tumor": "0.982", "oligodendrocyte": "0.997"},       # Note 3 entry
    "sweep_pred_all_200": {"tumor": "0.80", "oligodendrocyte": "0.93"},      # Note 3 entry
    "repro_recon_200": {"tumor": "0.990", "oligodendrocyte": "0.996"},       # Note 3 entry
    "repro_mean_200": {"tumor": "0.84", "oligodendrocyte": "0.93"},          # Results 1: "0.93 ... 0.84"
    "repro_recon_2000_range": ("0.43", "0.51"),              # Results 1: "fell to 0.43-0.51"
    "pred_diff_200": {"tumor": "0.04"},                      # Methods: "differed by 0.04 in tumour cells"
    "pred_diff_200_below": {"oligodendrocyte": 0.01},        # Methods: "by less than 0.01 in oligodendrocytes"
}


def fail(msg):
    sys.exit("FATAL: " + msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decimals(s):
    """Number of decimals written in a numeric string, e.g. '0.9816' -> 4."""
    return len(s.split(".")[1]) if "." in s else 0


def matches(record, printed, record_decimals=None):
    """True if a recorded value is consistent with a value printed at fewer decimals.
    The recorded value is itself rounded (half a unit of its last decimal, h), so the true value lies within
    record +/- h; the printed value is accepted if some value in that range rounds to it, i.e.
    |record - printed| <= half a unit of the printed precision + h (a small epsilon guards float noise)."""
    rd = decimals(record) if record_decimals is None else record_decimals
    pd = decimals(printed)
    h = 0.5 * 10 ** (-rd)
    return abs(float(record) - float(printed)) <= 0.5 * 10 ** (-pd) + h + 1e-12


# ------------------------------------------------------------------ 1. read the sweep record
for p in [SWEEP, VALUES] + list(LOGS.values()):
    if not os.path.isfile(p):
        fail("missing input %s" % p)

with open(SWEEP, newline="") as f:
    sweep_rows = list(csv.DictReader(f))
need = ["n_hvg", "cell_type", "frac_zero", "recon_R_vehicle", "recon_ok", "B_R_all", "C_R_all",
        "B_R_deg", "C_R_deg", "BminusC_R_deg"]
missing = [c for c in need if c not in sweep_rows[0]]
if missing:
    fail("sweep record lacks columns %s" % missing)

# index by (size, cell type); values are kept as the strings written in the record
S = {}
for r in sweep_rows:
    key = (int(r["n_hvg"]), r["cell_type"])
    if key in S:
        fail("duplicate sweep row %s" % (key,))                               # check a
    S[key] = r
if sorted(S) != sorted((n, t) for n in SIZES for t in TYPES):
    fail("sweep sizes/cell types are not exactly 100..1,000 x %s: %s" % (TYPES, sorted(S)))   # check a

for n in SIZES:                                                               # check b
    # the evaluation script writes ONE recon_ok per size (1 only if BOTH cell types reach the floor) on both rows
    flags = {int(S[(n, t)]["recon_ok"]) for t in TYPES}
    if len(flags) != 1:
        fail("recon_ok differs between the two cell types at %d genes" % n)
    both_pass = all(float(S[(n, t)]["recon_R_vehicle"]) >= FLOOR for t in TYPES)
    if bool(flags.pop()) != both_pass:
        fail("recon_ok at %d genes disagrees with the two reconstruction values" % n)
for (n, t), r in S.items():
    diff = float(r["B_R_deg"]) - float(r["C_R_deg"])
    if abs(diff - float(r["BminusC_R_deg"])) > 1.5e-4 + 1e-12:              # check c (three 4-decimal roundings)
        fail("BminusC_R_deg != B_R_deg - C_R_deg at %d %s (%s vs %.5f)" % (n, t, r["BminusC_R_deg"], diff))
for n in SIZES:
    if S[(n, "tumor")]["frac_zero"] != S[(n, "oligodendrocyte")]["frac_zero"]:   # check d
        fail("zero fraction differs between cell types at %d genes" % n)
print("[sweep] %s: 20 rows, gate, margin and zero-fraction checks passed (sha256 %s)" % (SWEEP, sha256(SWEEP)[:12]))

interpreted = [n for n in SIZES if all(int(S[(n, t)]["recon_ok"]) == 1 for t in TYPES)]

# ------------------------------------------------------------------ 2. read the reproduction values and the logs
with open(VALUES, newline="") as f:
    val_rows = list(csv.DictReader(f))
if len(val_rows) != 24:                                                       # check e
    fail("%s has %d rows, expected 24" % (VALUES, len(val_rows)))
V = {}                                                                        # (n_genes, type, compound) -> float
for r in val_rows:
    t = VALUES_TYPE.get(r["cell_type"])
    if t is None:
        fail("unknown cell type %r in %s" % (r["cell_type"], VALUES))
    x = float(r["r"])
    if not (0.0 <= x <= 1.0):
        fail("value outside [0, 1] in %s: %s" % (VALUES, r))
    if abs(round(x, 3) - x) > 1e-12:
        fail("value with more than 3 decimals in %s: %s" % (VALUES, r))
    V[(int(r["n_genes"]), t, r["compound"])] = x


def parse_log(path):
    """Per-compound R, reconstruction R and the printed means, with the regexes of plot_fig1e_v2.R."""
    per, rec, mean = {}, {}, {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^\s*(tumor|oligodendrocyte)\s+(\S+)\s+(-?[0-9]\.[0-9]{3})", line)
            if m:
                per[(m.group(1), m.group(2))] = m.group(3)
                continue
            m = re.match(r"^\s*(tumor|oligodendrocyte)\s*:\s*recon R = (-?[0-9.]+)", line)
            if m:
                rec[m.group(1)] = m.group(2)
                continue
            m = re.match(r"^\s*(tumor|oligodendrocyte)\s*:\s*R = (-?[0-9.]+)", line)
            if m:
                mean[m.group(1)] = m.group(2)
    if len(per) != 10 or len(rec) != 2 or len(mean) != 2:
        fail("%s: parsed %d per-compound, %d reconstruction, %d mean rows (expected 10, 2, 2)"
             % (path, len(per), len(rec), len(mean)))
    return per, rec, mean


L = {n: parse_log(p) for n, p in LOGS.items()}
for n, (per, rec, mean) in L.items():
    for t in TYPES:
        for c in COMPOUNDS:                                                   # check e: values file == log
            if (t, c) not in per or (n, t, c) not in V:
                fail("compound %s / %s missing at %d genes" % (c, t, n))
            if abs(float(per[(t, c)]) - V[(n, t, c)]) > 1e-9:
                fail("values file and log disagree: %d %s %s (%s vs %s)" % (n, t, c, V[(n, t, c)], per[(t, c)]))
        if abs(float(rec[t]) - V[(n, t, "vehicle reconstruction")]) > 1e-9:
            fail("values file and log disagree on reconstruction: %d %s" % (n, t))
        m5 = sum(float(per[(t, c)]) for c in COMPOUNDS) / 5.0                 # check f
        if abs(m5 - float(mean[t])) > 0.001 + 1e-12:
            fail("mean of the five compounds %.4f differs from the log's %s at %d %s" % (m5, mean[t], n, t))
print("[logs] values file equals both logs; compound means equal the logs' own means within 0.001")

# ------------------------------------------------------------------ 3. check the values the manuscript quotes (g)
E = EXPECTED
problems = []
if interpreted != E["interpreted_sizes"]:
    problems.append("interpreted sizes %s != %s" % (interpreted, E["interpreted_sizes"]))
failing = [n for n in SIZES if n not in interpreted]
if failing != E["failing_sizes"]:
    problems.append("failing sizes %s != %s" % (failing, E["failing_sizes"]))
for printed, n in E["failing_minima"]:
    lo = min((S[(n, t)]["recon_R_vehicle"] for t in TYPES), key=float)       # the lower of the two, as a number
    if not matches(lo, printed):
        problems.append("lower reconstruction at %d genes %s, manuscript %s" % (n, lo, printed))
lo1000 = min((S[(1000, t)]["recon_R_vehicle"] for t in TYPES), key=float)
if abs(float(lo1000)) >= E["failing_minimum_1000_abs_below"]:
    problems.append("lower reconstruction at 1,000 genes %s is not approximately zero" % lo1000)
tm = [S[(n, "tumor")]["BminusC_R_deg"] for n in interpreted]
tmin, tmax = min(tm, key=float), max(tm, key=float)
if not (matches(tmin, E["tumour_margin_range"][0]) and matches(tmax, E["tumour_margin_range"][1])):
    problems.append("tumour margin range %s-%s, manuscript %s-%s" % (tmin, tmax, *E["tumour_margin_range"]))
zeros = [float(S[(n, "tumor")]["frac_zero"]) * 100 for n in SIZES]
zr = (int(round(min(zeros))), int(round(max(zeros))))
if zr != E["zeros_percent_range"]:
    problems.append("zero entries %d-%d %%, manuscript %d-%d %%" % (zr + E["zeros_percent_range"]))
for t in TYPES:
    if not matches(S[(200, t)]["C_R_all"], E["vehicle_mean_all_200"][t]):
        problems.append("vehicle mean, all genes, 200 %s: %s vs %s" % (t, S[(200, t)]["C_R_all"], E["vehicle_mean_all_200"][t]))
    if not matches(S[(200, t)]["recon_R_vehicle"], E["sweep_recon_200"][t]):
        problems.append("sweep reconstruction 200 %s: %s vs %s" % (t, S[(200, t)]["recon_R_vehicle"], E["sweep_recon_200"][t]))
    if not matches(S[(200, t)]["B_R_all"], E["sweep_pred_all_200"][t]):
        problems.append("sweep prediction 200 %s: %s vs %s" % (t, S[(200, t)]["B_R_all"], E["sweep_pred_all_200"][t]))
    per200, rec200, mean200 = L[200]
    if not matches(rec200[t], E["repro_recon_200"][t]):
        problems.append("reproduction reconstruction 200 %s: %s vs %s" % (t, rec200[t], E["repro_recon_200"][t]))
    if not matches(mean200[t], E["repro_mean_200"][t]):
        problems.append("reproduction mean 200 %s: %s vs %s" % (t, mean200[t], E["repro_mean_200"][t]))
rec2000 = L[2000][1]
r2 = sorted((rec2000[t] for t in TYPES), key=float)
if not (matches(r2[0], E["repro_recon_2000_range"][0]) and matches(r2[1], E["repro_recon_2000_range"][1])):
    problems.append("2,000-gene reconstruction %s-%s vs %s-%s" % (r2[0], r2[1], *E["repro_recon_2000_range"]))
d_t = abs(float(L[200][2]["tumor"]) - float(S[(200, "tumor")]["B_R_all"]))
if abs(d_t - float(E["pred_diff_200"]["tumor"])) > 0.005 + 0.0006:           # 0.04 printed; inputs 3 and 4 decimals
    problems.append("tumour prediction difference between the two 200-gene models %.4f vs 0.04" % d_t)
d_o = abs(float(L[200][2]["oligodendrocyte"]) - float(S[(200, "oligodendrocyte")]["B_R_all"]))
if not d_o < E["pred_diff_200_below"]["oligodendrocyte"]:
    problems.append("oligodendrocyte prediction difference %.4f is not below 0.01" % d_o)
if problems:
    for p in problems:
        print("  MISMATCH:", p)
    fail("%d value(s) quoted in the manuscript are not reproduced by the records; nothing written" % len(problems))
print("[manuscript] every quoted value is reproduced at its printed precision")

# ------------------------------------------------------------------ 4. write the table
os.makedirs(OUT_DIR, exist_ok=True)
src_sweep = "gene_sensitivity_SEEDED_v2.csv"
src_log = {200: "eval_200_v2_18927698.out", 2000: "eval_v2_18927697.out"}
f3 = lambda x: "%.3f" % x                                                     # log values have 3 decimals exactly


def tm(s):
    """Typographic minus for the markdown table only (the CSV keeps the record's ASCII text)."""
    return "−" + s[1:] if s.startswith("-") else s

long_rows = []
for n in SIZES:
    long_rows.append(("a", n, "", "fraction of zero entries", S[(n, "tumor")]["frac_zero"], src_sweep))
    for t in TYPES:
        long_rows.append(("a", n, TYPE_LABEL[t], "vehicle reconstruction", S[(n, t)]["recon_R_vehicle"], src_sweep))
    long_rows.append(("a", n, "", "interpreted (>= 0.70 in both cell types)", "yes" if n in interpreted else "no", src_sweep))
    for t in TYPES:
        for q, col in (("prediction, all genes", "B_R_all"), ("vehicle mean, all genes", "C_R_all"),
                       ("prediction, 50 most responsive genes", "B_R_deg"),
                       ("vehicle mean, 50 most responsive genes", "C_R_deg"), ("margin (Fig. 1d)", "BminusC_R_deg")):
            long_rows.append(("b", n, TYPE_LABEL[t], q, S[(n, t)][col], src_sweep))
for n in (200, 2000):
    per, rec, mean = L[n]
    for t in TYPES:
        for c in COMPOUNDS:
            long_rows.append(("c", n, TYPE_LABEL[t], c, f3(V[(n, t, c)]), src_log[n]))
        long_rows.append(("c", n, TYPE_LABEL[t], "mean over the five compounds (as printed in the log)", mean[t], src_log[n]))
        long_rows.append(("c", n, TYPE_LABEL[t], "vehicle reconstruction", rec[t], src_log[n]))
for t in TYPES:
    long_rows.append(("c", 200, TYPE_LABEL[t], "published value (He et al.)", PUBLISHED[t], "He et al. 2026"))

csv_path = os.path.join(OUT_DIR, "supp_table_S3.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["part", "n_genes", "cell_type", "quantity", "value", "source"])
    w.writerows(long_rows)

# ---- markdown: three tables ----
md = []
md.append("# Supplementary Table S3 — glioblastoma: gene-panel sweep and reproduction of the published analysis\n")
md.append("All correlations are Pearson correlations between mean expression profiles on the log scale, from the "
          "re-evaluation on the corrected scale (24 October 2026). Vehicle reconstruction: between the mean expression "
          "of reconstructed and of measured vehicle cells, over all genes of the panel. Prediction: between the predicted "
          "and measured mean expression of the treated cells, with the compound direction estimated in myeloid cells and "
          "added to the encoded vehicle cells of the held-out cell type; averaged over the five held-out compounds. "
          "Vehicle mean: the same correlation with the vehicle mean of the held-out cell type used as the prediction. "
          "50 most responsive genes: for each compound, the 50 genes with the largest absolute difference between the "
          "treated and vehicle means. Margin = prediction minus vehicle mean on those genes (Fig. 1d, bottom). A size "
          "was interpreted only if vehicle reconstruction reached 0.70 in both cell types. The sweep model at 200 genes "
          "(a, b) and the model of the reproduction (c) were trained independently, on gene panels constructed "
          "independently (Supplementary Note 3). Values are given at the precision of the records: four decimals in "
          "the sweep record, three in the evaluation logs.\n")
md.append("**a | Gene-panel sweep: panel and vehicle reconstruction (Fig. 1d, top).**\n")
md.append("| Genes in the panel | Fraction of zero entries | Vehicle reconstruction, tumour cells | "
          "Vehicle reconstruction, oligodendrocytes | Interpreted |")
md.append("|---|---|---|---|---|")
for n in SIZES:
    md.append("| %s | %s | %s | %s | %s |" % ("{:,}".format(n), S[(n, "tumor")]["frac_zero"],
                                           tm(S[(n, "tumor")]["recon_R_vehicle"]),
                                           tm(S[(n, "oligodendrocyte")]["recon_R_vehicle"]),
                                           "yes" if n in interpreted else "no"))
md.append("")
md.append("**b | Gene-panel sweep: prediction against the vehicle mean (Fig. 1d, bottom; Results 1).**\n")
md.append("| Genes in the panel | Cell type | Prediction, all genes | Vehicle mean, all genes | "
          "Prediction, 50 most responsive genes | Vehicle mean, 50 most responsive genes | Margin |")
md.append("|---|---|---|---|---|---|---|")
for n in SIZES:
    for t in TYPES:
        r = S[(n, t)]
        md.append("| %s | %s | %s | %s | %s | %s | %s |" % ("{:,}".format(n), TYPE_LABEL[t], tm(r["B_R_all"]),
                                                            tm(r["C_R_all"]), tm(r["B_R_deg"]), tm(r["C_R_deg"]),
                                                            tm(r["BminusC_R_deg"])))
md.append("")
md.append("**c | Reproduction of the published analysis (Fig. 1e).** Correlation between predicted and measured mean "
          "expression of the treated cells, over all genes of the panel, per held-out compound.\n")
md.append("| | Tumour cells, 200 genes | Tumour cells, 2,000 genes | Oligodendrocytes, 200 genes | Oligodendrocytes, 2,000 genes |")
md.append("|---|---|---|---|---|")
for c in COMPOUNDS:
    md.append("| %s | %s | %s | %s | %s |" % (c, f3(V[(200, "tumor", c)]), f3(V[(2000, "tumor", c)]),
                                           f3(V[(200, "oligodendrocyte", c)]), f3(V[(2000, "oligodendrocyte", c)])))
md.append("| Mean over the five compounds | %s | %s | %s | %s |" % (L[200][2]["tumor"], L[2000][2]["tumor"],
                                                              L[200][2]["oligodendrocyte"], L[2000][2]["oligodendrocyte"]))
md.append("| Published value (He et al.) | %s | — | %s | — |" % (PUBLISHED["tumor"], PUBLISHED["oligodendrocyte"]))
md.append("| Vehicle reconstruction | %s | %s | %s | %s |" % (L[200][1]["tumor"], L[2000][1]["tumor"],
                                                        L[200][1]["oligodendrocyte"], L[2000][1]["oligodendrocyte"]))
md.append("")
md.append("Sources: (a, b) %s (job 18932187; eval_gbm_sensitivity_v2.py via run_gbm_seeded_reeval_v2.sh; decoder seed "
          "42), sha256 %s; (c) %s (sha256 %s) and %s (sha256 %s), through figures/fig1e_values_v2.csv (plot_fig1e_v2.R, "
          "research commit 7dbbcab), sha256 %s. Means in (c) are those printed in the logs, computed there from the "
          "unrounded per-compound values. Published values: He et al. (Squidiff, Nat. Methods 2026). Assembled by "
          "make_supp_table_s3.py."
          % (src_sweep, sha256(SWEEP)[:16], src_log[200], sha256(LOGS[200])[:16], src_log[2000], sha256(LOGS[2000])[:16],
             sha256(VALUES)[:16]))                                            # first 16 hex digits of each sha256
md_path = os.path.join(OUT_DIR, "supp_table_S3.md")
with open(md_path, "w") as f:
    f.write("\n".join(md) + "\n")

print("[written] %s (%d rows) and %s" % (csv_path, len(long_rows), md_path))
print("interpreted sizes: %s; tumour margin at these sizes %s to %s"
      % (", ".join(str(n) for n in interpreted), tmin, tmax))
