#!/usr/bin/env python3
"""
step23_chembl_targets.py — what KIND of bioactivity the ChEMBL records are, and does each drug have a potency
value on its own mechanism target?

Purpose
-------
step23_chembl_coverage.py found 50,473 human IC50 / Ki / Kd records with a pChEMBL value for 326 of the 372
Tahoe structures. Its most widely shared "targets" are cancer cell lines (A549, MCF7, HCT-116, HT-29, ...) and
safety-panel proteins (BSEP, hERG, CYP3A4), not the drugs' mechanism targets. Those are different kinds of
information and must be separated before any feature is defined. Still feature preparation: reads compound
identities and ChEMBL data only, no Tahoe expression or response data.

Logic
-----
1. Target type. Every target id in activities.csv is looked up in ChEMBL's target table (batched): SINGLE PROTEIN,
   PROTEIN COMPLEX, PROTEIN FAMILY, CELL-LINE, ORGANISM, TISSUE, ...
2. Mechanism targets. ChEMBL's curated drug-mechanism table is queried for every ChEMBL molecule of each structure
   (all salt forms and their parents, from the coverage cache): target id, action type, mechanism text.
3. Per structure (written to structure_potency.csv):
     - protein-level records: number of protein targets with a pChEMBL value; maximum pChEMBL over them
     - mechanism: mechanism targets; maximum pChEMBL measured on one of them (exact target id match)
     - cell-line records: number of cell lines; median pChEMBL over cell lines (growth inhibition / cytotoxicity)
     - the Tahoe cell lines among them (matched on the name with spaces, dashes and case removed)
4. summary_targets.json: records and structures per target type; coverage of each per-structure quantity.

Run on a Jubail LOGIN node (needs www.ebi.ac.uk), squidiff environment, after step23_chembl_coverage.py:
    cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
    python step23_chembl_targets.py
    python step23_chembl_targets.py --mock --dir <mock dir>     # offline structure test only
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

API = "https://www.ebi.ac.uk/chembl/api/data"
PROTEIN_TYPES = {"SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY", "PROTEIN COMPLEX GROUP",
                 "CHIMERIC PROTEIN", "PROTEIN-PROTEIN INTERACTION", "SELECTIVITY GROUP", "PROTEIN NUCLEIC-ACID COMPLEX"}
TAHOE_LINES = ["A-172", "H4", "SW 1088", "KATO III", "SNU-1", "COLO 205", "HCT15", "HT-29", "LS 180", "LoVo",
               "RKO", "SW1417", "SW48", "SW480"]
norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
TAHOE_NORM = {norm(x): x for x in TAHOE_LINES}


def http_get_json(path, mock, tries=5):
    if mock:
        return mock_api(path)
    url = path if path.startswith("http") else (
        "https://www.ebi.ac.uk" + path if path.startswith("/chembl/") else API + path)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            time.sleep(0.15)
            return data
        except Exception as e:
            if k == tries - 1:
                raise SystemExit(f"FATAL: ChEMBL request failed after {tries} tries: {url}\n{e}")
            time.sleep(2 ** k)


def paged(first, key, mock):
    out, nxt = [], first
    while nxt:
        page = http_get_json(nxt, mock)
        out.extend(page.get(key, []))
        nxt = page.get("page_meta", {}).get("next")
    return out


def batched(ids, n):
    ids = sorted(ids)
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def mock_api(path):
    """Canned answers in the API's shapes, for the offline test. Never a result."""
    q = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
    if path.startswith("/target"):
        ids = q["target_chembl_id__in"][0].split(",")
        return {"targets": [{"target_chembl_id": t, "pref_name": ("HT-29" if t.endswith("1") else f"protein {t}"),
                             "target_type": ("CELL-LINE" if t.endswith("1") else "SINGLE PROTEIN"),
                             "organism": "Homo sapiens"} for t in ids], "page_meta": {"next": None}}
    if path.startswith("/mechanism"):
        ids = q["molecule_chembl_id__in"][0].split(",")
        return {"mechanisms": [{"molecule_chembl_id": m, "parent_molecule_chembl_id": m, "target_chembl_id": "CHEMBL_T2",
                                "action_type": "INHIBITOR", "mechanism_of_action": "mock inhibitor"} for m in ids[:3]],
                "page_meta": {"next": None}}
    raise SystemExit(f"mock: unknown path {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tahoe_results/step23_chembl", help="output folder of step23_chembl_coverage.py")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    acts = list(csv.DictReader(open(os.path.join(a.dir, "activities.csv"))))
    caches = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.dir, "cache", "*.json")))]
    print(f"[input] {len(acts)} activity records, {len(caches)} structures")

    # 1. target types (batched lookups)
    tids = {r["target_chembl_id"] for r in acts if r["target_chembl_id"]}
    targets = {}
    for chunk in batched(tids, 100):
        q = urllib.parse.urlencode({"target_chembl_id__in": ",".join(chunk),
                                    "only": "target_chembl_id,pref_name,target_type,organism", "limit": 1000})
        for t in paged(f"/target.json?{q}", "targets", a.mock):
            targets[t["target_chembl_id"]] = t
    missing = tids - set(targets)
    print(f"[targets] {len(targets)} of {len(tids)} target ids typed; untyped: {len(missing)}")
    with open(os.path.join(a.dir, "targets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target_chembl_id", "pref_name", "target_type", "organism"], lineterminator="\n")
        w.writeheader(); w.writerows(targets[t] for t in sorted(targets))
    ttype = lambda t: targets.get(t, {}).get("target_type") or "UNTYPED"

    # 2. mechanism targets for every ChEMBL molecule of each structure (salt forms and parents)
    mol_to_struct = defaultdict(set)
    for c in caches:
        for m in c["molecules"]:
            mol_to_struct[m["molecule_chembl_id"]].add(c["structure_id"])
        for p in c["parents"]:
            mol_to_struct[p].add(c["structure_id"])
    mech_rows = []
    for chunk in batched(mol_to_struct, 50):
        q = urllib.parse.urlencode({"molecule_chembl_id__in": ",".join(chunk),
                                    "only": "molecule_chembl_id,parent_molecule_chembl_id,target_chembl_id,"
                                            "action_type,mechanism_of_action", "limit": 1000})
        for m in paged(f"/mechanism.json?{q}", "mechanisms", a.mock):
            for sid in mol_to_struct.get(m["molecule_chembl_id"], ()):
                mech_rows.append({"structure_id": sid, **{k: m.get(k) for k in
                                  ("molecule_chembl_id", "target_chembl_id", "action_type", "mechanism_of_action")}})
    with open(os.path.join(a.dir, "mechanisms.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["structure_id", "molecule_chembl_id", "target_chembl_id", "action_type",
                                          "mechanism_of_action"], lineterminator="\n")
        w.writeheader(); w.writerows(mech_rows)
    mech = defaultdict(set)
    for r in mech_rows:
        if r["target_chembl_id"]:
            mech[r["structure_id"]].add(r["target_chembl_id"])

    # 3. per-structure quantities
    by_s = defaultdict(list)
    for r in acts:
        by_s[r["structure_id"]].append(r)
    rows, tahoe_hits = [], Counter()
    for c in sorted(caches, key=lambda c: c["structure_id"]):
        sid = c["structure_id"]
        rs = [r for r in by_s.get(sid, []) if r["target_chembl_id"] and r["pchembl_value"]]
        prot = [r for r in rs if ttype(r["target_chembl_id"]) in PROTEIN_TYPES]
        cell = [r for r in rs if ttype(r["target_chembl_id"]) == "CELL-LINE"]
        on_mech = [float(r["pchembl_value"]) for r in prot if r["target_chembl_id"] in mech.get(sid, ())]
        cell_by_line = defaultdict(list)
        for r in cell:
            cell_by_line[targets[r["target_chembl_id"]]["pref_name"]].append(float(r["pchembl_value"]))
        tahoe = sorted({TAHOE_NORM[norm(n)] for n in cell_by_line if norm(n) in TAHOE_NORM})
        for t in tahoe:
            tahoe_hits[t] += 1
        line_medians = [statistics.median(v) for v in cell_by_line.values()]
        rows.append({
            "structure_id": sid, "names": " | ".join(c["names"]),
            "n_protein_targets": len({r["target_chembl_id"] for r in prot}),
            "max_protein_pchembl": max((float(r["pchembl_value"]) for r in prot), default=""),
            "mechanism_targets": " ".join(sorted(mech.get(sid, ()))),
            "max_pchembl_on_mechanism_target": max(on_mech, default=""),
            "n_cell_lines": len(cell_by_line),
            "median_cell_line_pchembl": statistics.median(line_medians) if line_medians else "",
            "tahoe_lines_tested": " | ".join(tahoe),
        })
    with open(os.path.join(a.dir, "structure_potency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)

    # 4. summary
    rec_by_type = Counter(ttype(r["target_chembl_id"]) for r in acts if r["target_chembl_id"])
    struct_by_type = defaultdict(set)
    for r in acts:
        if r["target_chembl_id"]:
            struct_by_type[ttype(r["target_chembl_id"])].add(r["structure_id"])
    has = lambda k: sum(1 for r in rows if r[k] not in ("", 0, "0"))
    summary = {
        "records_by_target_type": dict(rec_by_type.most_common()),
        "structures_by_target_type": {k: len(v) for k, v in sorted(struct_by_type.items(), key=lambda kv: -len(kv[1]))},
        "n_structures": len(rows),
        "structures_with_protein_potency": has("max_protein_pchembl"),
        "structures_with_mechanism_annotation": has("mechanism_targets"),
        "structures_with_potency_on_mechanism_target": has("max_pchembl_on_mechanism_target"),
        "structures_with_cell_line_potency": has("median_cell_line_pchembl"),
        "structures_tested_in_a_tahoe_line": has("tahoe_lines_tested"),
        "structures_per_tahoe_line": dict(tahoe_hits.most_common()),
        "untyped_target_ids": len(missing),
    }
    json.dump(summary, open(os.path.join(a.dir, "summary_targets.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[written] {a.dir}/targets.csv, mechanisms.csv, structure_potency.csv, summary_targets.json")


if __name__ == "__main__":
    main()
