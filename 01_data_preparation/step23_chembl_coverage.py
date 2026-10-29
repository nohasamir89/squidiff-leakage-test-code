#!/usr/bin/env python3
"""
step23_chembl_coverage.py — how much measured bioactivity (IC50, Ki, Kd) exists in ChEMBL for the Tahoe compounds,
and is stereochemistry recoverable from the screen's own SMILES?

Purpose
-------
Feature preparation for a possible Step 23 (richer compound representation). Before any compound feature is
pre-registered, we need to know whether it exists for enough of the 372 structures. This script reads ONLY
compound identities (the Step 15 data record); it reads no expression or response data, so running it before a
pre-registration is safe.

Logic
-----
1. Compounds. For every drug name in tahoe_results/step15_data_record.json: its structure id, the study's
   canonical SMILES (stereo-stripped, may carry a counter-ion), the screen's table SMILES and PubChem CID.
2. Identity key. The parent fragment of the canonical SMILES (largest organic fragment, so ".Cl" etc. are
   dropped) -> standard InChIKey -> its first block (14 characters = atom connectivity). Stereo-isomers and salt
   forms share this block, which matches the study's definition of a structure (stereo removed).
3. ChEMBL molecules. All ChEMBL molecules whose standard InChIKey starts with that block; their parent ChEMBL
   ids (molecule_hierarchy.parent_chembl_id) are pooled per structure.
4. ChEMBL activities. Every activity of those parents with standard_type IC50, Ki or Kd, a pChEMBL value
   (ChEMBL's quality-controlled -log10 molar potency), and target organism Homo sapiens.
5. Records written (nothing else is changed):
     <out>/cache/<structure_id>.json   raw API answers per structure (a re-run skips finished structures)
     <out>/structures.csv              one row per structure: ids, ChEMBL parents, record and target counts
     <out>/activities.csv              one row per activity record
     <out>/summary.json                ChEMBL release, date, coverage numbers, stereo check
   A self-test on imatinib (connectivity block KTUFNOKKBVMGRW must find CHEMBL941) runs first, so a changed API
   stops the script before any compound is queried.

Run (on a Jubail LOGIN node, which can reach www.ebi.ac.uk; the squidiff environment has RDKit):
    cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
    python step23_chembl_coverage.py
    python step23_chembl_coverage.py --mock --record mock_record.json --out /tmp/mock_out   # offline test only
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
API = "https://www.ebi.ac.uk/chembl/api/data"
TYPES = ["IC50", "Ki", "Kd"]
ACT_FIELDS = ["activity_id", "molecule_chembl_id", "parent_molecule_chembl_id", "target_chembl_id",
              "target_pref_name", "target_organism", "standard_type", "standard_relation", "standard_value",
              "standard_units", "pchembl_value", "assay_chembl_id", "assay_type", "data_validity_comment",
              "document_year"]
SELF_TEST = ("KTUFNOKKBVMGRW", "CHEMBL941")     # imatinib: connectivity block -> parent ChEMBL id


# ----------------------------------------------------------------------------- HTTP (real or mock)
def http_get_json(path_or_url, mock=False, tries=5):
    """GET a ChEMBL API path (or the relative 'next' URL of a page) and return the decoded JSON."""
    if mock:
        return mock_api(path_or_url)
    url = path_or_url if path_or_url.startswith("http") else (
        "https://www.ebi.ac.uk" + path_or_url if path_or_url.startswith("/chembl/") else API + path_or_url)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            time.sleep(0.15)                       # polite rate: well under the API's limits
            return data
        except Exception as e:                     # transient network / 5xx: back off and retry
            if k == tries - 1:
                raise SystemExit(f"FATAL: ChEMBL request failed after {tries} tries: {url}\n{e}")
            time.sleep(2 ** k)


def paged(first_path, key, mock=False):
    """All records of a paginated ChEMBL list endpoint (follows page_meta.next)."""
    out, nxt = [], first_path
    while nxt:
        page = http_get_json(nxt, mock)
        out.extend(page.get(key, []))
        nxt = page.get("page_meta", {}).get("next")
    return out


def mock_api(path):
    """Canned answers with the real API's shapes, for the offline structure test (--mock). Never a result."""
    if "status" in path:
        return {"chembl_db_version": "ChEMBL_MOCK", "chembl_release_date": "2000-01-01", "status": "UP"}
    if path.startswith("/molecule"):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
        block = q["molecule_structures__standard_inchi_key__startswith"][0]
        if block == SELF_TEST[0]:
            return {"molecules": [{"molecule_chembl_id": "CHEMBL941", "pref_name": "IMATINIB", "max_phase": "4.0",
                                   "molecule_hierarchy": {"parent_chembl_id": "CHEMBL941"}}],
                    "page_meta": {"next": None, "total_count": 1}}
        h = sum(map(ord, block))
        if h % 5 == 0:                                   # some structures absent from ChEMBL
            return {"molecules": [], "page_meta": {"next": None, "total_count": 0}}
        return {"molecules": [{"molecule_chembl_id": f"CHEMBL{h}", "pref_name": None, "max_phase": None,
                               "molecule_hierarchy": {"parent_chembl_id": f"CHEMBL{h}"}}],
                "page_meta": {"next": None, "total_count": 1}}
    if "/activity" in path:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
        pid = q["parent_molecule_chembl_id__in"][0].split(",")[0]
        off = int(q.get("offset", ["0"])[0])
        recs = [{"activity_id": off + i, "molecule_chembl_id": pid, "parent_molecule_chembl_id": pid,
                 "target_chembl_id": f"CHEMBL_T{(off + i) % 7}", "target_pref_name": f"target {(off + i) % 7}",
                 "target_organism": "Homo sapiens", "standard_type": TYPES[(off + i) % 3], "standard_relation": "=",
                 "standard_value": "10", "standard_units": "nM", "pchembl_value": "8.0", "assay_chembl_id": "A",
                 "assay_type": "B", "data_validity_comment": None, "document_year": 2020} for i in range(3)]
        nxt = None if off else path + "&offset=3"
        return {"activities": recs, "page_meta": {"next": nxt, "total_count": 6}}
    raise SystemExit(f"mock: unknown path {path}")


# ----------------------------------------------------------------------------- ChEMBL queries
def molecules_for_block(block, mock):
    q = urllib.parse.urlencode({"molecule_structures__standard_inchi_key__startswith": block,
                                "only": "molecule_chembl_id,pref_name,max_phase,molecule_hierarchy",
                                "limit": 1000})
    return paged(f"/molecule.json?{q}", "molecules", mock)


def activities_for_parents(parents, mock):
    if not parents:
        return []
    q = urllib.parse.urlencode({"parent_molecule_chembl_id__in": ",".join(sorted(parents)),
                                "standard_type__in": ",".join(TYPES),
                                "pchembl_value__isnull": "false",
                                "target_organism": "Homo sapiens",
                                "only": ",".join(ACT_FIELDS),
                                "limit": 1000})
    return paged(f"/activity.json?{q}", "activities", mock)


def parents_of(mols):
    ids = set()
    for m in mols:
        h = m.get("molecule_hierarchy") or {}
        ids.add(h.get("parent_chembl_id") or m["molecule_chembl_id"])
    return ids


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="tahoe_results/step15_data_record.json")
    ap.add_argument("--out", default="tahoe_results/step23_chembl")
    ap.add_argument("--mock", action="store_true", help="offline structure test with canned API answers")
    a = ap.parse_args()

    # 1. compounds from the data record (identities only)
    drugs = json.load(open(a.record))["drugs"]
    names = sorted(drugs["canonical_smiles"])
    by_struct = defaultdict(list)
    for n in names:
        by_struct[drugs["structure"][n]].append(n)
    print(f"[record] {len(names)} drug names, {len(by_struct)} structures")

    # stereo check on the screen's own SMILES (can chirality be recovered?)
    table = drugs.get("table_smiles", {})
    stereo_names = [n for n in names if any(c in str(table.get(n, "")) for c in "@/\\")]
    print(f"[stereo] names whose table SMILES carries stereo marks: {len(stereo_names)} of {len(names)}")

    # 2. identity key per structure
    lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    key = {}
    for sid, ns in by_struct.items():
        m = Chem.MolFromSmiles(drugs["canonical_smiles"][ns[0]])
        if m is None:
            raise SystemExit(f"FATAL: RDKit cannot parse the SMILES of {ns[0]}")
        key[sid] = Chem.MolToInchiKey(lfc.choose(m)).split("-")[0]
    if len(set(key.values())) != len(key):
        dup = [k for k, c in Counter(key.values()).items() if c > 1]
        print(f"[warn] {len(dup)} connectivity blocks shared by more than one structure: {dup}")

    # 3. ChEMBL release and self-test (stop early if the API changed)
    status = http_get_json("/status.json", a.mock)
    print(f"[chembl] {status.get('chembl_db_version')} released {status.get('chembl_release_date')}")
    st = parents_of(molecules_for_block(SELF_TEST[0], a.mock))
    if SELF_TEST[1] not in st:
        raise SystemExit(f"FATAL: self-test failed: block {SELF_TEST[0]} returned {sorted(st)}, expected {SELF_TEST[1]}")
    print("[self-test] imatinib found: the InChIKey prefix query works")

    # 4. per structure, cached
    cache = os.path.join(a.out, "cache")
    os.makedirs(cache, exist_ok=True)
    for i, sid in enumerate(sorted(by_struct)):
        cf = os.path.join(cache, f"{sid}.json")
        if os.path.exists(cf):
            continue
        mols = molecules_for_block(key[sid], a.mock)
        par = parents_of(mols)
        acts = activities_for_parents(par, a.mock)
        json.dump({"structure_id": sid, "names": by_struct[sid], "inchikey_block": key[sid],
                   "molecules": mols, "parents": sorted(par), "activities": acts}, open(cf, "w"))
        if (i + 1) % 25 == 0 or i + 1 == len(by_struct):
            print(f"  {i + 1}/{len(by_struct)} structures queried")

    # 5. tables and summary
    srows, arows = [], []
    target_structs = defaultdict(set)
    target_name = {}
    n_no_target = 0                      # records whose target id came back empty: counted, not used as a target
    for sid in sorted(by_struct):
        c = json.load(open(os.path.join(cache, f"{sid}.json")))
        acts = c["activities"]
        tcount = Counter(x["standard_type"] for x in acts)
        targets = {x["target_chembl_id"] for x in acts if x.get("target_chembl_id")}
        n_no_target += sum(1 for x in acts if not x.get("target_chembl_id"))
        for x in acts:
            if x.get("target_chembl_id"):
                target_name.setdefault(x["target_chembl_id"], x.get("target_pref_name"))
        for t in targets:
            target_structs[t].add(sid)
        phases = [float(m["max_phase"]) for m in c["molecules"] if m.get("max_phase") not in (None, "")]
        srows.append({"structure_id": sid, "names": " | ".join(c["names"]), "inchikey_block": c["inchikey_block"],
                      "chembl_parents": " ".join(c["parents"]), "max_phase": max(phases) if phases else "",
                      "n_records": len(acts), "n_IC50": tcount.get("IC50", 0), "n_Ki": tcount.get("Ki", 0),
                      "n_Kd": tcount.get("Kd", 0), "n_targets": len(targets)})
        for x in acts:
            arows.append({"structure_id": sid, **{f: x.get(f) for f in ACT_FIELDS}})

    with open(os.path.join(a.out, "structures.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]), lineterminator="\n"); w.writeheader(); w.writerows(srows)
    with open(os.path.join(a.out, "activities.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["structure_id"] + ACT_FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(arows)

    n = len(srows)
    has = lambda f: sum(1 for r in srows if f(r))
    shared = sorted(((len(s), t) for t, s in target_structs.items()), reverse=True)
    summary = {
        "chembl_release": status.get("chembl_db_version"), "chembl_release_date": status.get("chembl_release_date"),
        "queried_utc": datetime.now(timezone.utc).isoformat(), "mock": a.mock,
        "filters": {"standard_type": TYPES, "pchembl_value": "not null", "target_organism": "Homo sapiens",
                    "identity": "parent fragment, InChIKey connectivity block, ChEMBL parents pooled"},
        "n_structures": n,
        "n_parent_molecules": len({r["inchikey_block"] for r in srows}),
        "parent_molecules_with_any_record": len({r["inchikey_block"] for r in srows if r["n_records"] > 0}),
        "structures_in_chembl": has(lambda r: r["chembl_parents"]),
        "structures_with_any_record": has(lambda r: r["n_records"] > 0),
        "structures_with_10plus_records": has(lambda r: r["n_records"] >= 10),
        "structures_with_IC50": has(lambda r: r["n_IC50"] > 0),
        "structures_with_Ki": has(lambda r: r["n_Ki"] > 0),
        "structures_with_Kd": has(lambda r: r["n_Kd"] > 0),
        "n_records": len(arows), "records_without_target_id": n_no_target,
        "n_distinct_targets": len(target_structs),
        "targets_shared_by_10plus_structures": sum(1 for c, _ in shared if c >= 10),
        "targets_shared_by_30plus_structures": sum(1 for c, _ in shared if c >= 30),
        "top_targets_by_structures": [{"target": t, "name": target_name.get(t), "n_structures": c}
                                      for c, t in shared[:25]],
        "stereo_marked_table_smiles": len(stereo_names),
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "top_targets_by_structures"}, indent=2))
    print("top targets (number of structures with records):")
    for c, t in shared[:25]:
        print(f"    {c:4d}  {t:14s} {target_name.get(t)}")
    print(f"[written] {a.out}/structures.csv, activities.csv, summary.json")


if __name__ == "__main__":
    main()
