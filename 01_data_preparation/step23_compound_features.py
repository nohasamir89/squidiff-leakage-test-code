#!/usr/bin/env python3
"""
step23_compound_features.py — compound feature tables for the 372 Tahoe structures (Step 23 preparation).

Purpose
-------
Build, once and for every structure, the compound representations a Step 23 screen will compare, from chemical
structure and from the ChEMBL potency summaries already gathered. Reads compound identities only (the Step 15 data
record) and the ChEMBL outputs of step23_chembl_coverage.py / step23_chembl_targets.py. No Tahoe expression or
response data are read, so the tables can be built before the screen is pre-registered.

Logic
-----
Per structure (the study's unit: canonical SMILES with stereochemistry removed, counter-ions kept):
  parent              largest organic fragment (counter-ions and solvent removed), as in the split-overlap check
  fcfp4_asis          1,024-bit FCFP4 on the study SMILES exactly as the model encodes it (counter-ions included).
                      Checked bit for bit against the repository's own Drug_dose_encoder at dose 9 nM
                      (log10(9 + 1) = 1, so the encoder returns the unscaled bits), whenever that import works.
  fcfp4_parent        the same fingerprint on the parent (salt-free)
  ecfp4_parent        1,024-bit ECFP4 on the parent: atom-identity invariants (element, degree, hydrogens, charge,
                      ring membership) in place of FCFP4's pharmacophore classes
  descriptors         a fixed list (DESCRIPTORS below) on the parent: lipophilicity, hydrogen bonding, size and
                      shape, polarity and partial charges, net charge, element counts, hybridisation, ring sizes,
                      bond types. Values RDKit cannot compute are left empty and counted.
  pharmacophores      (a) counts of pharmacophore features on the parent, from RDKit's BaseFeatures definitions:
                      donor, acceptor, negative and positive ionisable, aromatic, hydrophobe, lumped hydrophobe,
                      zinc binder; (b) the 2D pharmacophore fingerprint of Gobbi and Poppinger (RDKit Pharm2D):
                      every pair and triplet of pharmacophore features with the bond-count distances between them,
                      i.e. how the features are arranged across the whole molecule. FCFP4 already encodes
                      pharmacophore classes, but only within two bonds of each atom. 3D pharmacophores are not
                      built: the study SMILES carry no stereochemistry, so conformers would rest on arbitrary
                      choices.
  potency             from tahoe_results/step23_chembl/structure_potency.csv (ChEMBL_37): strongest pChEMBL on any
                      protein target, on a curated mechanism target, and the median growth-inhibition pChEMBL over
                      cell lines; empty where ChEMBL has no value (no imputation here).
Outputs (tahoe_results/step23_features/):
  structures.csv      one row per structure: ids, names, parent key and SMILES, descriptors, potency summaries
  fingerprints.npz    structure_id order + fcfp4_asis, fcfp4_parent, ecfp4_parent (uint8, 372 x 1024) and
                      pharm2d_parent (uint8, 372 x the Gobbi signature size)
  manifest.json       RDKit version, commit, input sha256, descriptor list, missing-value counts, encoder check
Refuses to run if tracked files are modified (the commit it records must be the code that ran) or if the output
folder already exists.

Run (squidiff environment; CPU only, seconds):
    cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
    python step23_compound_features.py
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import rdkit
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem, ChemicalFeatures, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

RDLogger.DisableLog("rdApp.*")
LFC = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
PHARM_FAMILIES = ["Donor", "Acceptor", "NegIonizable", "PosIonizable", "Aromatic", "Hydrophobe",
                  "LumpedHydrophobe", "ZnBinder"]
PHARM2D_SIZE = Gobbi_Pharm2D.factory.GetSigSize()

# RDKit descriptors, grouped as discussed (lipophilicity, H-bonding, size/shape, polarity/electronic)
DESCRIPTORS = ["MolLogP", "MolMR",                                               # lipophilicity
               "NumHDonors", "NumHAcceptors",                                    # hydrogen bonding
               "MolWt", "HeavyAtomCount", "NumRotatableBonds", "RingCount",      # size and shape
               "NumAromaticRings", "FractionCSP3", "LabuteASA",
               "TPSA", "MaxPartialCharge", "MinPartialCharge",                   # polarity and electronic
               "MaxEStateIndex", "MinEStateIndex"]
ELEMENTS = ["N", "O", "S", "P", "F", "Cl", "Br", "I", "B"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bits(mol, features):
    """1,024-bit Morgan fingerprint of radius 2; features=True gives FCFP4, False gives ECFP4."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useFeatures=features)
    a = np.zeros((1024,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a


def pharmacophore_counts(m):
    """Number of pharmacophore features of each family (RDKit BaseFeatures.fdef definitions)."""
    fam = [f.GetFamily() for f in FEATURE_FACTORY.GetFeaturesForMol(m)]
    return {f"pharm_n_{k}": fam.count(k) for k in PHARM_FAMILIES}


def pharm2d_bits(m):
    """Gobbi-Poppinger 2D pharmacophore fingerprint: feature pairs and triplets with bond-count distances."""
    sv = Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory)
    a = np.zeros((PHARM2D_SIZE,), dtype=np.uint8)
    on = list(sv.GetOnBits())
    if on:
        a[on] = 1
    return a


def atom_bond_counts(m):
    """Atom- and bond-level properties summarised per molecule (element, charge, hybridisation, rings, bonds)."""
    f = {}
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    for e in ELEMENTS:
        f[f"n_{e}"] = sym.count(e)
    f["net_formal_charge"] = Chem.GetFormalCharge(m)
    f["n_pos_charged_atoms"] = sum(a.GetFormalCharge() > 0 for a in m.GetAtoms())
    f["n_neg_charged_atoms"] = sum(a.GetFormalCharge() < 0 for a in m.GetAtoms())
    hyb = [str(a.GetHybridization()) for a in m.GetAtoms()]
    for h in ("SP", "SP2", "SP3"):
        f[f"n_{h.lower()}_atoms"] = hyb.count(h)
    sizes = [len(r) for r in m.GetRingInfo().AtomRings()]
    for k in (3, 4, 5, 6, 7):
        f[f"n_rings_size_{k}"] = sizes.count(k)
    f["n_rings_size_8plus"] = sum(s >= 8 for s in sizes)
    bt = [str(b.GetBondType()) for b in m.GetBonds()]
    for t in ("DOUBLE", "TRIPLE", "AROMATIC"):
        f[f"n_{t.lower()}_bonds"] = bt.count(t)
    f["n_conjugated_bonds"] = sum(b.GetIsConjugated() for b in m.GetBonds())
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="tahoe_results/step15_data_record.json")
    ap.add_argument("--potency", default="tahoe_results/step23_chembl/structure_potency.csv")
    ap.add_argument("--out", default="tahoe_results/step23_features")
    ap.add_argument("--no-git-check", action="store_true", help="offline test only")
    a = ap.parse_args()

    if os.path.exists(a.out):
        sys.exit(f"FATAL: {a.out} exists; refusing to overwrite a record")
    commit = ""
    if not a.no_git_check:
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            sys.exit("FATAL: tracked files are modified; commit first\n" + dirty)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    desc_fn = dict(Descriptors._descList)
    absent = [d for d in DESCRIPTORS if d not in desc_fn]
    if absent:
        sys.exit(f"FATAL: RDKit {rdkit.__version__} lacks descriptors {absent}")

    # ---- structures and names from the data record (identities only) ----------------------------
    drugs = json.load(open(a.record))["drugs"]
    names_of = {}
    for n in sorted(drugs["canonical_smiles"]):
        names_of.setdefault(drugs["structure"][n], []).append(n)
    sids = sorted(names_of)
    print(f"[record] {sum(len(v) for v in names_of.values())} names, {len(sids)} structures")

    # ---- potency summaries (ChEMBL_37), merged as they are ---------------------------------------
    pot = {r["structure_id"]: r for r in csv.DictReader(open(a.potency))}
    if set(pot) != set(sids):
        sys.exit("FATAL: structure_potency.csv does not cover exactly the record's structures")

    rows, fa, fp, ep, ph = [], [], [], [], []
    for sid in sids:
        smi = drugs["canonical_smiles"][names_of[sid][0]]
        mol = Chem.MolFromSmiles(smi)
        par = LFC.choose(mol)
        row = {"structure_id": sid, "names": " | ".join(names_of[sid]),
               "parent_inchikey_block": Chem.MolToInchiKey(par).split("-")[0],
               "parent_smiles": Chem.MolToSmiles(par), "n_fragments_study_smiles": len(Chem.GetMolFrags(mol))}
        for d in DESCRIPTORS:
            try:
                v = float(desc_fn[d](par))
                row[d] = v if np.isfinite(v) else ""
            except Exception:
                row[d] = ""
        row.update(atom_bond_counts(par))
        row.update(pharmacophore_counts(par))
        p = pot[sid]
        row["pchembl_protein_max"] = p["max_protein_pchembl"]
        row["pchembl_mechanism_max"] = p["max_pchembl_on_mechanism_target"]
        row["pchembl_cellline_median"] = p["median_cell_line_pchembl"]
        row["n_protein_targets"] = p["n_protein_targets"]
        rows.append(row)
        fa.append(bits(mol, True)); fp.append(bits(par, True)); ep.append(bits(par, False))
        ph.append(pharm2d_bits(par))
    fa, fp, ep, ph = np.array(fa), np.array(fp), np.array(ep), np.array(ph)

    # ---- the as-encoded fingerprint must equal the model's own encoder ---------------------------
    enc_check = "not run (repository encoder could not be imported)"
    Drug_dose_encoder = None
    try:                                   # only the import may fail quietly; a failed comparison stops the run
        sys.path.insert(0, os.getcwd())
        from Squidiff.scrna_datasets import Drug_dose_encoder
    except Exception as e:
        print(f"[warn] encoder check skipped, import failed: {type(e).__name__}: {e}")
    if Drug_dose_encoder is not None:
        ref = np.array([Drug_dose_encoder([drugs["canonical_smiles"][names_of[s][0]]], [9.0], comb_num=1)[0]
                        for s in sids])
        n_diff = int((np.abs(ref - fa) > 1e-6).any(axis=1).sum())
        if n_diff:
            sys.exit(f"FATAL: fcfp4_asis differs from Drug_dose_encoder for {n_diff} structures")
        enc_check = "identical to Squidiff.scrna_datasets.Drug_dose_encoder at dose 9 nM for all structures"
    print(f"[encoder] {enc_check}")

    # ---- write ------------------------------------------------------------------------------------
    os.makedirs(a.out)
    with open(os.path.join(a.out, "structures.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(os.path.join(a.out, "fingerprints.npz"), structure_id=np.array(sids),
                        fcfp4_asis=fa, fcfp4_parent=fp, ecfp4_parent=ep, pharm2d_parent=ph)
    missing = {k: sum(1 for r in rows if r[k] == "") for k in rows[0] if k not in ("structure_id", "names")}
    changed = int((fa != fp).any(axis=1).sum())
    manifest = {
        "rdkit": rdkit.__version__, "commit": commit, "written_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {a.record: sha256(a.record), a.potency: sha256(a.potency)},
        "n_structures": len(sids), "n_parent_molecules": len({r["parent_inchikey_block"] for r in rows}),
        "descriptors": DESCRIPTORS, "atom_bond_columns": list(atom_bond_counts(Chem.MolFromSmiles("C")).keys()),
        "pharmacophore_count_columns": [f"pharm_n_{k}" for k in PHARM_FAMILIES],
        "pharm2d": {"factory": "Gobbi_Pharm2D (RDKit Pharm2D)", "signature_size": int(PHARM2D_SIZE),
                    "on_bits_per_structure_median": int(np.median(ph.sum(1))),
                    "structures_with_no_on_bits": int((ph.sum(1) == 0).sum())},
        "potency_columns": ["pchembl_protein_max", "pchembl_mechanism_max", "pchembl_cellline_median",
                            "n_protein_targets"],
        "missing_values": {k: v for k, v in missing.items() if v},
        "structures_whose_fingerprint_changes_without_counter_ions": changed,
        "encoder_check": enc_check,
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(f"[features] {len(DESCRIPTORS)} descriptors + {len(manifest['atom_bond_columns'])} atom/bond counts "
          f"+ {len(PHARM_FAMILIES)} pharmacophore counts + 4 fingerprints + 4 potency columns")
    print(f"[pharm2d] {PHARM2D_SIZE} possible bits; median {manifest['pharm2d']['on_bits_per_structure_median']} on per "
          f"structure; {manifest['pharm2d']['structures_with_no_on_bits']} structures with none")
    print(f"[salt-free] fingerprint changes for {changed} structures when counter-ions are removed")
    print(f"[missing] {manifest['missing_values']}")
    print(f"[written] {a.out}/structures.csv, fingerprints.npz, manifest.json")


if __name__ == "__main__":
    main()
