#!/usr/bin/env python3
"""
make_step24_permutation.py - the fixed compound substitution used by the Step 24 `fpshuf` arm.

Purpose
-------
Step 24 asks whether the rewired recipe still identifies a held-out compound when it is handed a different
compound's fingerprint. That requires one substitution, fixed and recorded before any arm runs, so that the sampling
runs cannot influence it and so that a reader can reconstruct exactly which compound stood in for which. The map is
keyed by STRUCTURE, not by condition: a compound receives one consistent wrong fingerprint across all its
conditions, lines, doses and plates, which is the convention of the Step 16 shuffled control. Pre-registration
PREREG_step24_fingerprint_swap.md section 3.

Logic
-----
1. Reads the Step 15 data record and takes the held-out structure ids of the split, sorted.
2. Reads the SMILES column of the split's test file (obs only, no expression values). That column is what the
   harness hands to the encoder, so the substitution is keyed by exactly those strings. Each string is mapped to its
   structure id through every SMILES form the record knows (canonical and table form, all names); the script stops
   unless the test file carries exactly the held-out structures, each under one string.
3. Encodes each structure's string with the repository's own encoder at a dose that leaves the bits unscaled
   (9 nM: log10(9 + 1) = 1), giving the 1,024 bits the model receives before dose scaling.
4. Draws a derangement of the structure ids with RandomState(0) by rejection: permute, and redraw while any
   structure maps to itself or would receive bits identical to its own (two different structures can share all
   1,024 bits; such a pairing would not change the input). Rejection keeps every admissible pairing equally likely;
   the number of draws is recorded.
5. Records, for interpretation only, the FCFP4 Tanimoto similarity between each structure's own bits and the bits
   substituted for it: a derangement that happened to pair chemically similar compounds would show a high median.
6. Writes the map, both strings per structure (own, substituted), the similarities, the draw count and a sha256 over
   the map itself (sorted "from>to" pairs, newline-joined), which the harness patch, the launcher and the scorer all
   re-check. Refuses to overwrite an existing record.

Run (squidiff environment, CPU, under a minute), from the worktree root:
    python make_step24_permutation.py --split drug_split_1
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

DATA_DEFAULT = os.environ.get('S15_DATA', '/scratch/ni2071/Diffusion/tahoe/datasets_step15')   # as score_step15_tahoe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="tahoe_results/step15_data_record.json")
    ap.add_argument("--split", default="drug_split_1")
    ap.add_argument("--data", default=DATA_DEFAULT, help="folder holding tahoe_test_<split>.h5ad")
    ap.add_argument("--out", default="tahoe_results/step24/permutation.json")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("FATAL: %s exists; refusing to overwrite a record" % a.out)

    # ---- 1. the held-out structures of the split
    rec = json.load(open(a.record))
    key = a.split.split("_")[-1]
    if key not in rec["splits"]:
        sys.exit("FATAL: the data record has no split %r" % a.split)
    sids = sorted(rec["splits"][key]["held_structures"])
    if len(sids) < 3:
        sys.exit("FATAL: a derangement needs at least 3 structures, found %d" % len(sids))

    # every SMILES form the record knows, mapped to its structure id; the names sharing each structure
    dr = rec["drugs"]
    form2sid, canon, names = {}, {}, {}
    for nm, s in dr["canonical_smiles"].items():
        sid = dr["structure"][nm]
        if canon.setdefault(sid, s) != s:
            sys.exit("FATAL: structure %s has two canonical SMILES (%s)" % (sid, nm))
        for form in {s, dr.get("table_smiles", {}).get(nm, s)}:
            if form2sid.setdefault(form, sid) != sid:
                sys.exit("FATAL: the record maps one SMILES to two structures (%s)" % nm)
        names.setdefault(sid, []).append(nm)

    # ---- 2. the strings the harness hands to the encoder: the test file's SMILES column
    import anndata as ad
    test_file = os.path.join(a.data, "tahoe_test_%s.h5ad" % a.split)
    if not os.path.isfile(test_file):
        sys.exit("FATAL: missing %s" % test_file)
    te = ad.read_h5ad(test_file, backed="r")
    file_strings = sorted(set(te.obs["SMILES"].astype(str).values))
    te.file.close()
    by_sid = {}
    for s in file_strings:
        if s not in form2sid:
            sys.exit("FATAL: a test-file SMILES is not a SMILES of the data record: %s" % s[:80])
        by_sid.setdefault(form2sid[s], set()).add(s)
    if sorted(by_sid) != sids:
        sys.exit("FATAL: the test file carries %d structures, the record holds %d held-out (%s)"
                 % (len(by_sid), len(sids), sorted(set(by_sid) ^ set(sids))[:3]))
    multi = sorted(s for s, v in by_sid.items() if len(v) != 1)
    if multi:
        sys.exit("FATAL: %d structures appear under more than one SMILES string in the test file, e.g. %s"
                 % (len(multi), multi[:3]))
    own = {s: next(iter(by_sid[s])) for s in sids}
    n_canon = sum(own[s] == canon[s] for s in sids)
    print("[test file] %s: %d distinct SMILES strings, one per held-out structure; %d of %d equal the record's "
          "canonical SMILES" % (test_file, len(file_strings), n_canon, len(sids)))

    # ---- 3. the bits each structure receives (dose 9 nM leaves them unscaled)
    sys.path.insert(0, os.getcwd())
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')                     # RDKit deprecation notices only; no effect on any bit
    from Squidiff.scrna_datasets import Drug_dose_encoder
    raw = {s: np.asarray(Drug_dose_encoder([own[s]], [9.0], comb_num=1)[0], dtype=np.float64) for s in sids}
    for s in sids:
        if not np.all(np.isin(raw[s], (0.0, 1.0))):
            sys.exit("FATAL: the encoder at 9 nM did not return 0/1 bits for %s" % s)
        if raw[s].sum() == 0:
            sys.exit("FATAL: structure %s encodes to an empty fingerprint" % s)
    bits = {s: raw[s] > 0.5 for s in sids}
    same = [(x, y) for i, x in enumerate(sids) for y in sids[i + 1:] if np.array_equal(bits[x], bits[y])]
    if same:
        print("[bits] %d pair(s) of different held-out structures share all 1,024 bits: %s"
              % (len(same), ", ".join("%s=%s" % p for p in same)))

    # ---- 4. derangement by rejection
    rng = np.random.RandomState(0)
    idx = np.arange(len(sids))
    draws = 0
    while True:
        draws += 1
        if draws > 100000:
            sys.exit("FATAL: no admissible derangement after 100000 draws")
        p = rng.permutation(len(sids))
        if np.any(p == idx):
            continue                                  # a structure would keep its own SMILES
        if any(np.array_equal(bits[sids[i]], bits[sids[p[i]]]) for i in idx):
            continue                                  # a structure would receive bits identical to its own
        break
    mapping = {sids[i]: sids[p[i]] for i in idx}
    if any(k == v for k, v in mapping.items()) or sorted(mapping.values()) != sids:
        sys.exit("FATAL: the map is not a derangement of the held-out structures")
    digest = hashlib.sha256("\n".join("%s>%s" % (k, mapping[k]) for k in sids).encode()).hexdigest()

    # ---- 5. Tanimoto between own and substituted bits (interpretation only)
    sim = {}
    for s in sids:
        x, y = bits[s], bits[mapping[s]]
        sim[s] = round(float((x & y).sum() / (x | y).sum()), 4)
    vals = np.array([sim[s] for s in sids])
    stats = dict(median=round(float(np.median(vals)), 4), min=round(float(vals.min()), 4),
                 max=round(float(vals.max()), 4), above_0_5=int((vals > 0.5).sum()))

    # ---- 6. the record
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = dict(prereg="PREREG_step24_fingerprint_swap.md", split=a.split, n_structures=len(sids),
               structures=sids, draws_until_derangement=draws,
               rng="RandomState(0), rejection: redraw while a structure maps to itself or to identical bits",
               map=mapping, sha256=digest,
               smiles_source=dict(test_file=test_file, column="SMILES", distinct_strings=len(file_strings),
                                  equal_to_record_canonical=n_canon),
               smiles={s: dict(own=own[s], substituted=own[mapping[s]], names=names[s],
                               own_is_record_canonical=bool(own[s] == canon[s])) for s in sids},
               identical_bit_pairs=[list(q) for q in same],
               tanimoto_own_vs_substituted=sim, tanimoto_summary=stats)
    json.dump(out, open(a.out, "w"), indent=2)
    print("[permutation] %d held-out structures of %s, admissible derangement after %d draw(s)"
          % (len(sids), a.split, draws))
    print("[tanimoto] own vs substituted FCFP4: median %.4f, range %.4f-%.4f, above 0.5: %d"
          % (stats["median"], stats["min"], stats["max"], stats["above_0_5"]))
    print("[sha256] %s" % digest)
    print("[written] %s" % a.out)


if __name__ == "__main__":
    main()
