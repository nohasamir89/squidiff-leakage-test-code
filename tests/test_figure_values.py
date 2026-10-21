"""Every figure value must be re-derivable from the committed result records.

How it works: the extraction logic of each figure script (plot_fig1b.py and
plot_fig2.R ... plot_fig5.R, plot_fig1e.R) is reimplemented here in Python,
run against the committed records in 07_records/ (or tahoe_results/ and
leakage_results/ in the repository), and compared row by row with the
committed fig*_values.csv tables that feed the supplementary tables.

Why this matters: the figure scripts are the last step between a result record
and a number a reader sees. A transcription error there would not be caught by
any other test. On 21 Oct 2026 this comparison matched all 297 committed rows
with a worst numeric discrepancy of 5e-14 (R's 15-significant-digit CSV
printing); the tolerance below is 1e-12 relative, i.e. "equal to the record".

Nothing numeric is typed in: the only constants are the R scripts' own label
strings, copied verbatim so the row keys line up.
"""
import csv
import json
import math
import re

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

REL = 1e-12
ABS = 1e-12
SPLITS = ["drug_split_1", "drug_split_0", "drug_split_2"]
SPLIT_MAP = {"drug_split_1": "Split 1", "drug_split_0": "Split 0", "drug_split_2": "Split 2"}


# --------------------------------------------------------------------------- helpers
def jload(path):
    """json with NaN/Infinity tokens mapped to None, exactly as the R helper reads them as null."""
    with open(path) as f:
        return json.load(f, parse_constant=lambda c: None)


def read_committed(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(s):
    s = s.strip()
    return None if s in ("NA", "") else float(s)


def compare(committed, recomputed, keyfields, numfields, strfields):
    """Row-by-row comparison in file order. Returns a list of mismatch descriptions (empty = pass)."""
    bad = []
    if len(committed) != len(recomputed):
        bad.append("row count: committed %d vs recomputed %d" % (len(committed), len(recomputed)))
    for i in range(min(len(committed), len(recomputed))):
        com, rec = committed[i], recomputed[i]
        key = "|".join(str(com.get(k, "")) for k in keyfields)
        for k in keyfields + strfields:                      # the row identity itself must agree
            if com.get(k, "") != str(rec.get(k, "")):
                bad.append("%s: %s committed %r recomputed %r" % (key, k, com.get(k, ""), rec.get(k, "")))
        for k in numfields:
            cv, rv = _num(com.get(k, "NA")), rec.get(k, None)
            if cv is None and rv is None:
                continue
            if cv is None or rv is None or abs(cv - rv) > max(ABS, REL * abs(cv)):
                bad.append("%s: %s committed %r recomputed %r" % (key, k, cv, rv))
    return bad


def chance_jaccard():
    """R: sum(dhyper(i, 50, 150, 50) * i / (100 - i))  ==  scipy hypergeom(M=200, n=50, N=50)."""
    i = np.arange(0, 51)
    return float(np.sum(hypergeom(200, 50, 50).pmf(i) * i / (100 - i)))


# --------------------------------------------------------------------------- Fig 1b
def test_fig1b_values(layout):
    rows = []
    ROWS = [("base", "Original model"), ("ridge_train", "Like-for-like ridge"),
            ("ridge_S7", "Cross-validated ridge"), ("mean_response", "Compound-blind average"),
            ("replicate", "Plate replicate")]
    lab = dict(ROWS)
    d = pd.read_csv(layout.record("step15_model_long.csv"))
    d = d[(d["line"] == "ALL") & (d["metric"] == "de_overlap")]
    tahoe = {}
    for split in SPLITS:
        for key, _ in ROWS[:-1]:
            v = d[(d["split"] == split) & (d["submission"] == key)]["value"]
            assert len(v) == 1, (split, key, len(v))
            tahoe[(split, key)] = float(v.iloc[0])
    a = jload(layout.record("step15_model_analysis.json"))
    for split in SPLITS:
        tahoe[(split, "replicate")] = float(a["per_split"][split]["replicate"]["de_overlap"])
    b = jload(layout.record("step14_analysis.json"))
    sciplex = {}
    for split in SPLITS:
        tm = b["per_split"][split]["thesis_metrics"]
        for key, _ in ROWS[:-1]:
            sciplex[(split, key)] = float(tm[key]["de_overlap"])
    for dataset, vals in [("Tahoe-100M", tahoe), ("sci-Plex", sciplex)]:
        for (split, key), v in vals.items():
            rows.append(dict(dataset=dataset, split=split, submission=key, label=lab[key], gene_identity=v))
    rows.append(dict(dataset="both", split="all", submission="chance", label="Chance (50 of 200)",
                     gene_identity=chance_jaccard()))
    bad = compare(read_committed(layout.figvalue("fig1b_values.csv")), rows,
                  ["dataset", "split", "submission"], ["gene_identity"], ["label"])
    assert not bad, "\n".join(bad)


def test_fig1b_two_committed_copies_identical(layout):
    """fig1b_values.csv is committed twice (supplementary table and the extractor's own output)."""
    paths = [d / "fig1b_values.csv" for d in layout.figvalues if (d / "fig1b_values.csv").exists()]
    if len(paths) < 2:
        pytest.skip("only one copy present in this layout")
    assert paths[0].read_bytes() == paths[1].read_bytes()


# --------------------------------------------------------------------------- Fig 2
def test_fig2_values(layout):
    rows = []
    row_labels = dict(base="Original model", rewired="Rewired model", ridge_train="Training-set ridge",
                      ridge_S7="Cross-validated ridge", mean_response="Compound-blind average",
                      control_paste="Unaltered vehicle cells", replicate="Plate replicate")
    tahoe_keys = dict(base="base", rewired="recipe", ridge_train="ridge_train", ridge_S7="ridge_S7",
                      mean_response="mean_response", control_paste="control_paste")
    sciplex_keys = dict(base="base", rewired="fm_50", ridge_train="ridge_train", ridge_S7="ridge_S7",
                        mean_response="mean_response")

    def add(panel, dataset, split, key, source_key, metric, value):
        assert value is not None and math.isfinite(value), (panel, dataset, split, key, metric, value)
        rows.append(dict(panel=panel, dataset=dataset, split=split, submission=key, source_key=source_key,
                         label=row_labels[key], metric=metric, value=float(value)))

    long = pd.read_csv(layout.record("step15_model_long.csv"))
    long = long[long["line"] == "ALL"]
    for split in SPLITS:
        for key, src in tahoe_keys.items():
            for metric in ("de_overlap", "energy"):
                if key == "control_paste" and metric == "de_overlap":
                    continue                                   # undefined for the vehicle cells (no top-50 set)
                v = long[(long["split"] == split) & (long["submission"] == src) & (long["metric"] == metric)]["value"]
                assert len(v) == 1
                add("a" if metric == "de_overlap" else "b", "Tahoe-100M", split, key, src, metric, float(v.iloc[0]))
    aj = jload(layout.record("step15_model_analysis.json"))
    for split in SPLITS:
        rep = aj["per_split"][split]["replicate"]
        add("a", "Tahoe-100M", split, "replicate", "replicate", "de_overlap", rep["de_overlap"])
        add("b", "Tahoe-100M", split, "replicate", "replicate", "energy", rep["energy"])
    bj = jload(layout.record("step14_analysis.json"))
    for split in SPLITS:
        tm = bj["per_split"][split]["thesis_metrics"]
        for key, src in sciplex_keys.items():
            add("a", "sci-Plex", split, key, src, "de_overlap", tm[src]["de_overlap"])
    rows.append(dict(panel="a", dataset="both", split="all", submission="chance", source_key="computed",
                     label="Chance (50 of 200)", metric="de_overlap", value=chance_jaccard()))
    bad = compare(read_committed(layout.figvalue("fig2_values.csv")), rows,
                  ["panel", "dataset", "split", "submission", "metric"], ["value"], ["source_key", "label"])
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------- Fig 3
def test_fig3_values(layout):
    rows = []

    def add(panel, intervention, quantity, split, level, point, lo, hi, source):
        assert all(v is not None and math.isfinite(v) for v in (point, lo, hi))
        assert lo <= point <= hi, (intervention, quantity, point, lo, hi)
        rows.append(dict(panel=panel, intervention=intervention, quantity=quantity, split=split, level=level,
                         point=float(point), lo=float(lo), hi=float(hi), source=source))

    def pick(dct, pattern):
        ks = [k for k in dct if pattern in k]
        assert len(ks) == 1, (pattern, ks)
        return dct[ks[0]]

    RC = "Response correlation\n(paired median difference)"
    GI = "Gene identity\n(mean difference)"
    s9 = jload(layout.record("step9_analysis.json"))
    c9 = pick(s9["contrasts"], "both minus fingerprint")
    add("b", "Mechanism-of-action labels", RC, "Split 1", "95 %", c9["paired_median"], c9["ci"][0], c9["ci"][1], "step9_analysis.json")
    s10 = jload(layout.record("step10_analysis.json"))
    c10b = pick(s10["contrasts"], "fp_chemvae minus fingerprint")
    k_alone = [k for k in s10["contrasts"] if "chemvae minus fingerprint" in k and "fp_chemvae" not in k]
    assert len(k_alone) == 1
    c10a = s10["contrasts"][k_alone[0]]
    add("b", "Learned chemical embedding", RC, "Split 1", "95 %", c10a["paired_median"], c10a["ci"][0], c10a["ci"][1], "step10_analysis.json")
    add("b", "Chemical embedding + fingerprint", RC, "Split 1", "95 %", c10b["paired_median"], c10b["ci"][0], c10b["ci"][1], "step10_analysis.json")
    s13 = jload(layout.record("step13_analysis.json"))
    arms13 = s13["results"]["arms"]
    for arm, lab in [("attn_fp", "Fingerprint self-attention"), ("attn_mem", "Attention over similar training compounds"),
                     ("attn_cross", "Cross-attention, compound × cell state")]:
        x = arms13[arm]["de_overlap_minus_reference"]
        add("b", lab, GI, "Split 1", "95 %", x["diff"], x["ci"][0], x["ci"][1], "step13_analysis.json")
    Q1, Q2, Q3 = "Gene identity\nvs rewired model", "Gene identity\nvs matched control", "Population distance\nvs rewired model"
    x = arms13["attn_cross"]
    for q, key in ((Q1, "de_overlap_minus_reference"), (Q2, "de_overlap_minus_control"), (Q3, "energy_minus_reference")):
        add("c", "Cross-attention", q, "Split 1", "95 %", x[key]["diff"], x[key]["ci"][0], x[key]["ci"][1], "step13_analysis.json")
    s13b = jload(layout.record("step13b_analysis.json"))
    for sp in ("drug_split_0", "drug_split_2"):
        p = s13b["per_split"][sp]["primary"]
        for q, key in ((Q1, "de_overlap_minus_reference"), (Q2, "de_overlap_minus_control"), (Q3, "energy_minus_reference")):
            add("c", "Cross-attention", q, SPLIT_MAP[sp], "97.5 %", p[key]["diff"], p[key]["ci"][0], p[key]["ci"][1], "step13b_analysis.json")
    for r in rows:
        r["excludes_zero"] = "TRUE" if (r["lo"] > 0 or r["hi"] < 0) else "FALSE"
    bad = compare(read_committed(layout.figvalue("fig3_values.csv")), rows,
                  ["panel", "intervention", "quantity", "split"], ["point", "lo", "hi"], ["level", "source", "excludes_zero"])
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------- Fig 4
def test_fig4_values(layout):
    rows = []

    def add(panel, facet, x, split, point, lo, hi, extra=None, source=""):
        assert all(v is not None and math.isfinite(v) for v in (point, lo, hi))
        assert lo <= point <= hi
        rows.append(dict(panel=panel, facet=facet, x=x, split=split, point=float(point), lo=float(lo), hi=float(hi),
                         extra=extra, source=source))

    def ctr(contrasts, key):
        assert contrasts.get(key) is not None, "missing contrast " + key
        return contrasts[key]

    long = pd.read_csv(layout.record("step15_model_long.csv"))
    long = long[(long["line"] == "ALL") & (long["submission"].isin(["base", "recipe"])) &
                (long["metric"].isin(["de_overlap", "dpc_mean", "energy"]))]
    assert len(long) == 2 * 3 * 3
    B = dict(de_overlap="Gene identity\n(Jaccard, top-50 genes)", dpc_mean="Response correlation\n(Pearson r, top-50 genes)",
             energy="Population distance\n(energy; lower is better)")
    for _, r in long.iterrows():                               # R keeps the file order of the filtered rows
        model = "Original\nmodel" if r["submission"] == "base" else "Rewired\nmodel"
        add("b", B[r["metric"]], model, SPLIT_MAP[r["split"]], r["value"], r["value"], r["value"], source="step15_model_long.csv")
    s18 = jload(layout.record("step18_analysis.json"))
    ph = jload(layout.record("posthoc_ot_energy_vs_references.json"))
    C = dict(en="Population distance\nvs random pairing", en_cp="Population distance\nvs vehicle cells",
             de="Gene identity", pds="Compound\ndiscrimination", tr="Cell tracking")
    for sp in SPLITS:
        cc = s18["per_split"][sp]["contrasts"]
        for facet, key in [(C["en"], "Q1_primary_energy_ot_pair_minus_recipe"), (C["de"], "Q2_gate_de_ot_pair_minus_recipe"),
                           (C["pds"], "Q3_gate_pds_ot_pair_minus_recipe"), (C["tr"], "exploratory_tracking_ot_pair_minus_recipe")]:
            x = ctr(cc, key)
            add("c", facet, "OT", SPLIT_MAP[sp], x["diff"], x["ci"][0], x["ci"][1], source="step18_analysis.json")
    for r in ph["results"]:
        if r["contrast"] == "ot_pair - control_paste":
            add("c", C["en_cp"], "OT", SPLIT_MAP[r["split"]], r["point"], r["lo"], r["hi"], source="posthoc_ot_energy_vs_references.json")
    s20 = jload(layout.record("step20_analysis_development.json"))
    d20 = s20["per_split"]["drug_split_1"]
    D = dict(en="Population distance", de="Gene identity", pds="Compound\ndiscrimination", tr="Cell tracking")
    set_lab = dict(mb4="m = 4", mb8="m = 8", mb32="m = 32")
    for s in ("mb4", "mb8", "mb32"):
        cc = d20["contrasts"][s]
        f = d20["f_cos"][s]
        assert f is not None and math.isfinite(f)
        for facet, key in [(D["en"], "Q1_energy_minus_recipe"), (D["de"], "Q2_de_minus_recipe"),
                           (D["pds"], "Q3_pds_minus_recipe"), (D["tr"], "secondary_tracking_minus_recipe")]:
            x = ctr(cc, key)
            add("d", facet, set_lab[s], "Split 1", x["diff"], x["ci"][0], x["ci"][1], extra=f, source="step20_analysis_development.json")
    if d20["contrasts"].get("ot_pair") is not None:
        cc, src = d20["contrasts"]["ot_pair"], "step20_analysis_development.json"
        k = dict(en="Q1_energy_minus_recipe", de="Q2_de_minus_recipe", pds="Q3_pds_minus_recipe", tr="secondary_tracking_minus_recipe")
    else:
        cc, src = s18["per_split"]["drug_split_1"]["contrasts"], "step18_analysis.json"
        k = dict(en="Q1_primary_energy_ot_pair_minus_recipe", de="Q2_gate_de_ot_pair_minus_recipe",
                 pds="Q3_gate_pds_ot_pair_minus_recipe", tr="exploratory_tracking_ot_pair_minus_recipe")
    for m in k:
        x = ctr(cc, k[m])
        add("d", D[m], "exact", "Split 1", x["diff"], x["ci"][0], x["ci"][1], extra=1, source=src)
    s21 = jload(layout.record("step21_analysis.json"))
    E = dict(en="Population distance\n(dashed: margin)", pds="Compound discrimination", de="Gene identity")
    dev = s21["development_descriptive"]
    x = ctr(dev["contrasts"], "Q1_energy_mb32_minus_ot_pair")
    add("e", E["en"], "mb32", "Split 1", x["diff"], x["ci"][0], x["ci"][1], extra=dev["delta"], source="step21_analysis.json (development_descriptive)")
    x = ctr(dev["contrasts"], "Q2_pds_mb32_minus_ot_pair")
    add("e", E["pds"], "mb32", "Split 1", x["diff"], x["ci"][0], x["ci"][1], source="step21_analysis.json (development_descriptive)")
    x = ctr(dev["contrasts"], "secondary_de_mb32_minus_ot_pair")
    add("e", E["de"], "mb32", "Split 1", x["diff"], x["ci"][0], x["ci"][1], source="step21_analysis.json (development_descriptive)")
    for sp in ("drug_split_0", "drug_split_2"):
        ps = s21["per_split"][sp]
        x = ctr(ps["contrasts"], "Q1_energy_mb32_minus_ot_pair")
        add("e", E["en"], "mb32", SPLIT_MAP[sp], x["diff"], x["ci"][0], x["ci"][1], extra=ps["delta"], source="step21_analysis.json")
        x = ctr(ps["contrasts"], "Q2_pds_mb32_minus_ot_pair")
        add("e", E["pds"], "mb32", SPLIT_MAP[sp], x["diff"], x["ci"][0], x["ci"][1], source="step21_analysis.json")
        x = ctr(ps["contrasts"], "secondary_de_mb32_minus_ot_pair")
        add("e", E["de"], "mb32", SPLIT_MAP[sp], x["diff"], x["ci"][0], x["ci"][1], source="step21_analysis.json")
    for r in rows:
        r["excludes_zero"] = "TRUE" if (r["lo"] > 0 or r["hi"] < 0) else "FALSE"
    bad = compare(read_committed(layout.figvalue("fig4_values.csv")), rows,
                  ["panel", "facet", "x", "split"], ["point", "lo", "hi", "extra"], ["source", "excludes_zero"])
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------- Fig 5
def test_fig5_values(layout):
    rows = []

    def add(panel, facet, row, split, point, lo=None, hi=None, chance=None, source=""):
        assert point is not None and math.isfinite(point)
        if lo is not None:
            assert lo <= point <= hi, (facet, row, split, point, lo, hi)
        rows.append(dict(panel=panel, facet=facet, row=row, split=split, point=float(point), lo=lo, hi=hi,
                         chance=chance, source=source))

    def find_contrast(contrasts, pattern):
        ks = [k for k in contrasts if re.search(pattern, k)]
        assert len(ks) == 1, (pattern, ks)
        return contrasts[ks[0]]

    s15c = jload(layout.record("step15c_pds.json"))
    A_ROWS = [("base", "Original model"), ("recipe", "Rewired model"), ("guidance_w2", "Rewired model + guidance"),
              ("ridge_train", "Training-set ridge"), ("ridge_S7", "Cross-validated ridge"), ("mean_response", "Compound-blind average")]
    for sp in SPLITS:
        ps = s15c["per_split"][sp]
        for s, lab in A_ROWS:
            v = ps["metrics"][s]["pds_mean"]
            x = find_contrast(ps["contrasts"], "pds_" + s + r"_minus_0\.5$")
            add("a", "Compound discrimination", lab, SPLIT_MAP[sp], v, 0.5 + x["ci"][0], 0.5 + x["ci"][1], chance=0.5, source="step15c_pds.json")
        add("a", "Compound discrimination", "Plate replicate", SPLIT_MAP[sp], ps["replicate"]["pds_mean"], chance=0.5, source="step15c_pds.json (replicate)")
    B_ROWS = [("base", "Original model"), ("rewired", "Rewired model"), ("ridge_train", "Training-set ridge"), ("ridge_S7", "Cross-validated ridge")]
    s16 = jload(layout.record("step16_analysis.json"))
    for sp in SPLITS:
        ps = s16["per_split"][sp]
        for s, lab in B_ROWS:
            key = "recipe" if s == "rewired" else s
            x = find_contrast(ps["contrasts"], "mra_" + key + "_minus_0$")
            add("b", "Tahoe-100M", lab, SPLIT_MAP[sp], ps["metrics"][key]["mra"], x["ci"][0], x["ci"][1], chance=0, source="step16_analysis.json")
        add("b", "Tahoe-100M", "Plate replicate (14 contexts)", SPLIT_MAP[sp], ps["replicate"]["mra"], chance=0, source="step16_analysis.json (replicate)")
    s17 = jload(layout.record("step17_mra.json"))
    for sp in SPLITS:
        ps = s17["per_split"][sp]
        for s, lab in B_ROWS:
            key = "fm_50" if s == "rewired" else s
            x = find_contrast(ps["contrasts"], "^(Q1_primary|secondary)_mra_" + key + "_minus_0$")
            add("b", "sci-Plex", lab, SPLIT_MAP[sp], ps["metrics"][key]["mra"], x["ci"][0], x["ci"][1], chance=0, source="step17_mra.json")
        add("b", "sci-Plex", "Split-half anchor", SPLIT_MAP[sp], ps["anchor"]["mra"], chance=0, source="step17_mra.json (anchor)")
    s19 = jload(layout.record("step19_heterogeneity.json"))
    C_ROWS = [("recipe", "Rewired model"), ("ot_pair", "Rewired model, OT pairing"), ("mean_response", "Compound-blind average"),
              ("ridge_train", "Training-set ridge"), ("ridge_S7", "Cross-validated ridge")]
    for sp in SPLITS:
        ps = s19["per_split"][sp]
        for s, lab in C_ROWS:
            x = find_contrast(ps["contrasts"], "ha_" + s + "_minus_0$")
            add("c", "Spread agreement", lab, SPLIT_MAP[sp], ps["metrics"][s]["ha"], x["ci"][0], x["ci"][1], chance=0, source="step19_heterogeneity.json")
        x = find_contrast(ps["contrasts"], "ha_ceiling_minus_0$")
        add("c", "Spread agreement", "Plate replicate (14 contexts)", SPLIT_MAP[sp], ps["replicate"]["ha"], x["ci"][0], x["ci"][1], chance=0, source="step19_heterogeneity.json (replicate)")
    s22 = jload(layout.record("step22_analysis.json"))
    D_ROWS = [("real_half", "Measured cells (other half)"), ("recipe", "Rewired model"), ("ot_pair", "Rewired model, OT pairing"),
              ("mb32", "Rewired model, OT pairing (m = 32)"), ("mean_response", "Compound-blind average"),
              ("control_paste", "Unaltered vehicle cells"), ("ridge_train", "Training-set ridge"), ("ridge_S7", "Cross-validated ridge")]
    D1, D2 = "Recognised (% of query populations)", "No compound assigned (%)"
    for sp in SPLITS:
        g = s22["per_split"][sp]["rates"]["group"]
        for s, lab in D_ROWS:
            r = g[s]
            add("d", D1, lab, SPLIT_MAP[sp], 100 * r["hit"], chance=100 * r["chance"], source="step22_analysis.json (rates.group)")
            add("d", D2, lab, SPLIT_MAP[sp], 100 * r["unassigned"], source="step22_analysis.json (rates.group)")
    for r in rows:
        if r["lo"] is not None and math.isfinite(r["lo"]):
            ch = r["chance"] if r["chance"] is not None else float("nan")
            r["excludes"] = "TRUE" if (r["lo"] > ch or r["hi"] < ch) else "FALSE"
        else:
            r["excludes"] = "NA"
        r["chance"] = None if r["chance"] is None else float(r["chance"])
    bad = compare(read_committed(layout.figvalue("fig5_values.csv")), rows,
                  ["panel", "facet", "row", "split"], ["point", "lo", "hi", "chance"], ["source", "excludes"])
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------- Fig 1e (GBM logs)
def _parse_gbm_log(path, n_genes):
    """plot_fig1e.R's two regexes, applied to the raw SLURM log."""
    type_lab = dict(tumor="Tumour cells", oligodendrocyte="Oligodendrocytes")
    per_re = re.compile(r"^\s*(tumor|oligodendrocyte)\s+(\S+)\s+(-?[0-9]\.[0-9]{3})")
    rec_re = re.compile(r"^\s*(tumor|oligodendrocyte)\s*:\s*recon R = (-?[0-9.]+)")
    per, rec = [], []
    with open(path) as f:
        for line in f:
            m = per_re.match(line)
            if m:
                per.append(dict(panel="Prediction of held-out compounds", n_genes=str(n_genes), cell_type=type_lab[m.group(1)],
                                compound=m.group(2), r=float(m.group(3))))
                continue
            m = rec_re.match(line)
            if m:
                rec.append(dict(panel="Vehicle reconstruction", n_genes=str(n_genes), cell_type=type_lab[m.group(1)],
                                compound="vehicle reconstruction", r=float(m.group(2))))
    assert len(per) == 10 and len(rec) == 2, (path, len(per), len(rec))
    return per + rec


def test_fig1e_values_from_raw_logs(layout):
    if not (layout.gbm_log_200.exists() and layout.gbm_log_2000.exists()):
        pytest.skip("GBM SLURM logs not present in this layout")
    rows = _parse_gbm_log(layout.gbm_log_200, 200) + _parse_gbm_log(layout.gbm_log_2000, 2000)
    bad = compare(read_committed(layout.figvalue("fig1e_values.csv")), rows, ["panel", "n_genes", "cell_type", "compound"], ["r"], [])
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------- Figs 1c, 1d (printed annotations)
def test_fig1c_annotations_regression(layout):
    """Golden test for the ratios printed in Fig 1c (original model, Split 1, seed 0). The values
    were derived independently from the committed per-condition CSV on 21 Oct 2026 and match the
    manuscript (2.5x spread, 1.7x library size, 27% vs 89% zeros, 3.3x displacement)."""
    d = pd.read_csv(layout.record("step15_diag_cells_drug_split_1_seed_0_per_condition.csv"))
    b = d[d["arm"] == "base"]
    assert len(b) == 2899                                           # held-out conditions of Split 1
    m = b[["xx", "yy", "lib_pred", "lib_true", "zero_pred", "zero_true", "d_mean_pred_true", "d_true_ctrl"]].mean()
    assert round(m["xx"] / m["yy"], 1) == 2.5
    assert round(m["lib_pred"] / m["lib_true"], 1) == 1.7
    assert (round(100 * m["zero_pred"]), round(100 * m["zero_true"])) == (27, 89)
    assert round(m["d_mean_pred_true"] / m["d_true_ctrl"], 1) == 3.3


def test_fig1d_record_structure_and_gate(layout):
    """gene_sensitivity_SEEDED.csv must hold exactly one row per (panel size, cell type), and the
    interpreted sizes (vehicle reconstruction >= 0.70 in BOTH cell types) must be the six the
    manuscript reports: 100, 200, 500, 600, 700, 800."""
    if not layout.gbm_sens_csv.exists():
        pytest.skip("gene_sensitivity_SEEDED.csv not present in this layout")
    d = pd.read_csv(layout.gbm_sens_csv)
    assert not d.duplicated(["n_hvg", "cell_type"]).any()
    assert sorted(d["n_hvg"].unique()) == list(range(100, 1001, 100))
    ok = d.groupby("n_hvg")["recon_R_vehicle"].agg(lambda s: bool((s >= 0.70).all()))
    assert sorted(k for k, v in ok.items() if v) == [100, 200, 500, 600, 700, 800]
    flag = d.groupby("n_hvg")["recon_ok"].agg(lambda s: bool((s == 1).all()))
    assert all(flag[k] == ok[k] for k in ok.index)                 # the committed per-size flag agrees
