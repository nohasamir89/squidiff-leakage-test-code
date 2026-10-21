#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_sciplex_stage1.py
======================
STAGE 1 preprocessing for the Squidiff sci-Plex3 reproduction, run in TWO modes
so you can both (a) reproduce the authors' exact pipeline and (b) build a clean,
correctly-transformed baseline -- and quantify the difference between them.

Runs on the PRnet-processed file already downloaded & verified:
    /scratch/ni2071/Diffusion/sciplex/Sci_Plex.h5ad   (already log1p-normalized)

TWO MODES (written to separate directories, nothing overwrites):
  faithful  : reproduces the authors' prep_sciplex_data.ipynb exactly, INCLUDING
              the redundant second transform. The authors call
                  sc.pp.normalize_total(adata)   # NOTE: no target_sum -> MEDIAN
                  sc.pp.log1p(adata)
              on this already-log1p file. We replicate that verbatim (median, not
              1e4) so the run is a genuine reproduction of what they published.
  corrected : skips the redundant normalization (the file is already log1p), so
              downstream training sees correctly-scaled data. This is your real
              baseline.

WHY THIS SCRIPT AND NOT THE SHORTER ONE:
  The key object Squidiff consumes is NOT a pool of vehicle cells. For every
  treated cell i, the model needs ITS paired vehicle cell (given by
  obs['paired_control_index']) as a ROW-ALIGNED control. The authors' training
  command reads a companion file per split:
        --data_path         datasets/sci_plex_train_<split>.h5ad
        --control_data_path  datasets/sci_plex_train_<split>_control.h5ad
  where control row i is the partner of treated row i. This script builds and
  writes those *_control.h5ad companions. A per-cell-line vehicle pool would
  silently break the model's control conditioning, so we do not do that.

DECISIONS PINNED TO SOURCE (each is a defensible line for the thesis):
  - Controls identified by dose == 0 (== product_name 'control'; 6,464 cells).
    NOT dose_val, which is a rescaled dose (vehicle==1.0, never 0). The file's
    'control' flag column is uniformly 0 and non-informative (documented quirk).
  - 200 highly variable genes (authors' notebook: n_top_genes=200; model built
    with gene_size=200). HVGs selected on the full matrix, per the authors.
  - Splits used: drug_split_{0,1,2} (unseen-drug) and random_split_{0,1,2}
    (random). These are the six behind Fig. 3i (n=3 each). PRnet's other split
    columns are deliberately not used.

USAGE:
    python prep_sciplex_stage1.py            # runs both modes
    python prep_sciplex_stage1.py --mode corrected   # just one, if you prefer
"""

import argparse
import os
import numpy as np
import anndata as ad
import scanpy as sc

SRC = "/scratch/ni2071/Diffusion/sciplex/Sci_Plex.h5ad"
BASE_OUT = "/scratch/ni2071/Diffusion/sciplex"
N_TOP_GENES = 200
SPLITS = [f"drug_split_{i}" for i in range(3)] + [f"random_split_{i}" for i in range(3)]


def slim(adata_subset):
    """Minimal treated AnnData carrying the fields the training code reads."""
    out = ad.AnnData(adata_subset.X.copy())
    out.obs_names = adata_subset.obs_names
    out.var_names = adata_subset.var_names
    out.obs["Group"] = adata_subset.obs["cov_drug_dose_name"].values
    out.obs["dose"] = adata_subset.obs["dose"].values
    out.obs["SMILES"] = adata_subset.obs["SMILES"].values
    # cell_name drives Squidiff's per-cell-line embedding (AnnDataDataset checks
    # it against CELL_LINE_ORDER). sci-Plex values: A549/K562/MCF7, taken from
    # the source 'cell_type' column.
    out.obs["cell_name"] = adata_subset.obs["cell_type"].values
    return out


def run_mode(mode):
    outd = os.path.join(BASE_OUT, f"datasets_{mode}")
    os.makedirs(outd, exist_ok=True)
    print(f"\n=== MODE: {mode.upper()}  ->  {outd} ===")

    a = sc.read_h5ad(SRC)
    print(f"  loaded {a.n_obs:,} cells x {a.n_vars:,} genes  (X max={a.X.max():.4f})")

    # --- transform handling: the ONLY difference between the two modes ---
    if mode == "faithful":
        # verbatim reproduction of the authors' notebook (NO target_sum -> median)
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        print(f"  [faithful] re-applied normalize_total(median)+log1p -> X max={a.X.max():.4f}")
    else:
        print(f"  [corrected] skipped renormalization (already log1p) -> X max={a.X.max():.4f}")

    # --- HVG(200) on the full matrix, per the authors' notebook ---
    sc.pp.highly_variable_genes(a, n_top_genes=N_TOP_GENES)
    a = a[:, a.var.highly_variable].copy()
    print(f"  selected {a.n_vars} HVGs")

    # --- matched control, ROW-ALIGNED to every cell via paired_control_index ---
    # EMPIRICAL FINDING (verified on the file, documented for the thesis):
    # paired_control_index holds the obs_name (cell barcode) of each cell's
    # matched vehicle -- NOT an integer row position. The barcodes happen to be
    # digit-strings (e.g. '143150') because obs_names are stringified row ids,
    # which is why an earlier astype(int)+positional read landed on the wrong
    # cells (0% vehicles). obs_names are unique here, so name lookup is exact.
    # We map each barcode -> its row position via a name index, then gather
    # positionally with numpy (robust to AnnData's name/position indexing quirks).
    name_to_pos = {n: i for i, n in enumerate(a.obs_names)}
    pci_names = a.obs["paired_control_index"].astype(str).values
    pci_pos = np.fromiter((name_to_pos[n] for n in pci_names), dtype=int, count=len(pci_names))

    # SANITY (defensible safeguard): every paired control must be a vehicle.
    # VEHICLES are identified by dose == 0 (equivalently product_name=='control';
    # both give the same 6,464 cells). NOTE: do NOT use 'dose_val' -- that column
    # is a RESCALED dose taking values {0.001,0.01,0.1,1.0} where vehicle==1.0,
    # so 'dose_val==0' is never true. Verified empirically on the file.
    ctrl_dose = a.obs["dose"].values[pci_pos]
    n_bad = int((ctrl_dose != 0).sum())
    assert n_bad == 0, (
        f"{n_bad} paired_control_index entries resolve to NON-vehicle cells "
        f"(dose != 0). The control pairing is corrupt; stop and inspect."
    )

    # Build the row-aligned control matrix: control_all[i] is the matched
    # vehicle of a[i]. numpy positional gather on X.
    control_X = a.X[pci_pos].copy()
    control_all = ad.AnnData(control_X)
    control_all.var_names = a.var_names
    control_all.obs_names = a.obs_names            # align to treated rows by position
    # The control file is ALSO loaded through AnnDataDataset, which requires
    # 'cell_name'. Each control is the matched vehicle of a treated cell, so it
    # belongs to the SAME cell line as that treated cell -> use the treated
    # cell's own cell_type (row-aligned), not the vehicle donor's.
    control_all.obs["cell_name"] = a.obs["cell_type"].values
    # Group/dose/SMILES are read from the TREATED file, but AnnDataDataset builds
    # them from whichever adata it is given; controls carry benign placeholders so
    # the loader's column lookups succeed if it inspects them.
    control_all.obs["Group"] = a.obs["cov_drug_dose_name"].values
    control_all.obs["dose"] = a.obs["dose"].values
    control_all.obs["SMILES"] = a.obs["SMILES"].values
    print(f"  [check] all {control_all.n_obs:,} paired controls are vehicle cells (dose==0)")

    # --- treated cells only (exclude vehicles from the treated files) ---
    is_treated = (a.obs["dose"] != 0).values   # vehicles are dose==0 (not dose_val)

    # --- write train/test treated + row-aligned control companions per split ---
    for col in SPLITS:
        split_col = a.obs[col].astype(str).values
        for part in ("train", "test"):
            mask = (split_col == part) & is_treated
            n = int(mask.sum())
            treated = slim(a[mask])
            control = control_all[mask].copy()   # SAME rows, SAME order -> aligned
            t_path = os.path.join(outd, f"sci_plex_{part}_{col}.h5ad")
            c_path = os.path.join(outd, f"sci_plex_{part}_{col}_control.h5ad")
            treated.write_h5ad(t_path)
            control.write_h5ad(c_path)
            print(f"  {col:16s} {part:5s}: {n:7d} cells  "
                  f"(treated+control written)  test_drugs="
                  f"{a.obs.loc[mask,'product_name'].nunique() if part=='test' else '-'}")

    print(f"  [ok] {mode} written to {outd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["faithful", "corrected", "both"], default="both")
    args = ap.parse_args()
    modes = ["corrected", "faithful"] if args.mode == "both" else [args.mode]
    for m in modes:
        run_mode(m)
    print("\n[done] Stage 1 complete.")
    print("       Compare X max: faithful should be visibly smaller-scaled than")
    print("       corrected if the double-transform materially changed the data.")


if __name__ == "__main__":
    main()