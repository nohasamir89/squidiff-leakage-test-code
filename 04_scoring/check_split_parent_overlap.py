#!/usr/bin/env python3
"""
check_split_parent_overlap.py — are any held-out compounds present in training as a different salt, hydrate or
solvate of the same molecule?

Purpose
-------
The compound splits hold out "structures" defined by RDKit canonical SMILES with stereochemistry removed. Counter-
ions and solvent molecules were kept in those SMILES, so the same active molecule supplied as, for example, the free
base and the hydrochloride counts as two structures, and the two can fall on opposite sides of a split. In the
Tahoe-100M list, 22 pairs of study structures share their parent molecule (largest organic fragment; InChIKey
connectivity block), and their fingerprints, as the model encodes them, have a median Tanimoto of 0.97. This script
measures, for every split of both datasets, which held-out compounds have such a twin in the training set.

Logic
-----
Parent key of a compound = first block (14 characters) of the standard InChIKey of its largest organic fragment
(rdMolStandardize.LargestFragmentChooser(preferOrganic=True)). Same key = same molecule up to salt, solvate,
hydrate, charge state and stereochemistry.
  Tahoe-100M: compounds, structures and the held-out structure list of each split come from the Step 15 data
              record; the training set of a split is every structure not held out in it.
  sci-Plex:   compounds of each split are the unique SMILES in the obs of the train and test files.
For each split: held-out compounds whose parent key also occurs in that split's training set, with the number of
held-out conditions they contribute (sci-Plex: from the test file; Tahoe: printed where the record allows).
Reads compound identities and split membership only; no expression or response values are read.

Output: leakage_results/split_parent_overlap.json (and a printed report).

Run from the repository worktree, squidiff environment:
    python check_split_parent_overlap.py
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
LFC = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
SPLITS = ["drug_split_0", "drug_split_1", "drug_split_2"]


def parent_key(smiles, strict=True):
    """Parent-molecule key; with strict=False an unparseable or empty SMILES (e.g. a vehicle row) gives None."""
    m = Chem.MolFromSmiles(smiles) if smiles not in ("", "nan", "None") else None
    if m is None or m.GetNumAtoms() == 0:
        if strict:
            raise SystemExit(f"FATAL: RDKit cannot parse {smiles!r}")
        return None
    return Chem.MolToInchiKey(LFC.choose(m)).split("-")[0]


def find_heldout_lists(obj, path=""):
    """Every list of strings inside the record's split entry, with its path (the held-out list is among them)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += find_heldout_lists(v, f"{path}/{k}")
    elif isinstance(obj, list) and obj and all(isinstance(x, str) for x in obj):
        out.append((path, obj))
    return out


def tahoe(record_path):
    rec = json.load(open(record_path))
    drugs = rec["drugs"]
    names = sorted(drugs["canonical_smiles"])
    struct_of = drugs["structure"]
    smiles_of_struct = {}
    for n in names:
        smiles_of_struct.setdefault(struct_of[n], drugs["canonical_smiles"][n])
    key = {s: parent_key(smi) for s, smi in smiles_of_struct.items()}
    all_structs = set(key)
    sid_re = re.compile(r"^S\d{4}$")

    result = {}
    for sp in SPLITS:
        # the Step 15 record keys its splits by index ('0', '1', '2'); accept either form
        entry = None
        if isinstance(rec["splits"], dict):
            entry = rec["splits"].get(sp, rec["splits"].get(sp.rsplit("_", 1)[1]))
        if entry is None:
            raise SystemExit(f"FATAL: split {sp} not in record['splits']; keys: {list(rec['splits'])[:10]}")
        cands = find_heldout_lists(entry, sp)
        held = None
        for path, lst in cands:                       # a list of structure ids, or of drug names, of ~75 entries
            if all(sid_re.match(x) for x in lst) and 50 <= len(set(lst)) <= 100:
                held, how = set(lst), f"{path} (structure ids)"; break
            if all(x in struct_of for x in lst) and 50 <= len({struct_of[x] for x in lst}) <= 100:
                held, how = {struct_of[x] for x in lst}, f"{path} (drug names)"; break
        if held is None:
            print(f"[{sp}] could not identify the held-out list; lists found:")
            for path, lst in cands:
                print(f"    {path}: {len(lst)} items, e.g. {lst[:3]}")
            raise SystemExit("FATAL: paste this output so the held-out field can be named explicitly")
        train = all_structs - held
        train_keys = defaultdict(set)
        for s in train:
            train_keys[key[s]].add(s)
        leaked = sorted(s for s in held if key[s] in train_keys)
        result[sp] = {"held_out_structures": len(held), "source_field": how,
                      "held_out_with_parent_in_training": [
                          {"structure_id": s, "names": [n for n in names if struct_of[n] == s],
                           "training_twins": sorted(train_keys[key[s]]),
                           "twin_names": [n for n in names if struct_of[n] in train_keys[key[s]]]} for s in leaked]}
        print(f"[Tahoe-100M {sp}] {len(leaked)} of {len(held)} held-out structures have their parent molecule in "
              f"training (field {how})")
        for x in result[sp]["held_out_with_parent_in_training"]:
            print(f"      {x['structure_id']} {x['names']}  <- training: {x['twin_names']}")
    return result


def sciplex(data_dir):
    import anndata as ad
    result = {}
    for sp in SPLITS:
        obs = {}
        for part in ("train", "test"):
            f = os.path.join(data_dir, f"sci_plex_{part}_{sp}.h5ad")
            a = ad.read_h5ad(f, backed="r")
            obs[part] = a.obs[["SMILES", "Group"]].astype(str).copy()
            a.file.close()
        tr_keys, skipped = defaultdict(set), set()
        for smi in obs["train"]["SMILES"].unique():
            k = parent_key(smi, strict=False)
            if k is None:
                skipped.add(smi); continue
            tr_keys[k].add(smi)
        test = obs["test"]
        per_smi = test.groupby("SMILES")["Group"].nunique()
        leaked = []
        for smi, n_cond in per_smi.items():
            k = parent_key(smi, strict=False)
            if k is None:
                skipped.add(smi); continue
            if k in tr_keys:
                leaked.append({"smiles": smi, "n_test_conditions": int(n_cond),
                               "training_twins": sorted(tr_keys[k]),
                               "identical_smiles_in_training": smi in tr_keys[k]})
        n_cond_total = int(test["Group"].nunique())
        n_cond_leaked = sum(x["n_test_conditions"] for x in leaked)
        result[sp] = {"held_out_compounds": int(len(per_smi)), "test_conditions": n_cond_total,
                      "held_out_with_parent_in_training": leaked,
                      "test_conditions_affected": n_cond_leaked}
        if skipped:
            print(f"[sci-Plex {sp}] SMILES not parsed (vehicle or empty), skipped: {sorted(skipped)}")
        result[sp]["unparsed_smiles_skipped"] = sorted(skipped)
        print(f"[sci-Plex {sp}] {len(leaked)} of {len(per_smi)} held-out compounds have their parent molecule in "
              f"training ({n_cond_leaked} of {n_cond_total} test conditions)")
        for x in leaked:
            print(f"      {x['smiles']}  <- training: {x['training_twins']}"
                  + ("  [IDENTICAL SMILES]" if x["identical_smiles_in_training"] else ""))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="tahoe_results/step15_data_record.json")
    ap.add_argument("--sciplex_dir", default="/scratch/ni2071/Diffusion/sciplex/datasets_corrected")
    ap.add_argument("--out", default="leakage_results/split_parent_overlap.json")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"FATAL: {a.out} exists; refusing to overwrite a record")
    res = {"what": "held-out compounds whose parent molecule (largest organic fragment, InChIKey connectivity "
                   "block) also occurs in the training set of the same split",
           "tahoe": tahoe(a.record), "sciplex": sciplex(a.sciplex_dir),
           "written_utc": datetime.now(timezone.utc).isoformat()}
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[written] {a.out}")


if __name__ == "__main__":
    main()
