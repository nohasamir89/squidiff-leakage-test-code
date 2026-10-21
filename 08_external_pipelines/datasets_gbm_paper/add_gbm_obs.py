#!/usr/bin/env python
"""
Add the obs columns the Squidiff loader requires (cell_name, Group) to the
already-written GBM h5ad files. Avoids re-running the full preprocessing.

The loader needs:
  - cell_name : drives cell_id integer (3 cell types -> n_cell_lines=3)
  - Group     : carried column. NOTE: with num_classes=None the model NEVER
                reads Group (MLPModel.forward line ~229), so its value does not
                affect training. We set it to a per-drug int purely so the column
                exists and is available for bookkeeping / the Delta-z eval.
"""
import scanpy as sc

DATA_DIR = "/scratch/ni2071/Diffusion/datasets_gbm_paper"
FILES = ["gbm_train.h5ad", "gbm_test.h5ad", "gbm_control_to.h5ad"]

DRUG_TO_INT = {
    'vehicle': 0, 'etoposide': 1, 'panobinostat': 2, 'RO4929097': 3,
    'tazemetostat': 4, 'ispinesib': 5, 'ANA-12': 6,
}

for f in FILES:
    path = f"{DATA_DIR}/{f}"
    a = sc.read_h5ad(path)
    a.obs['cell_name'] = a.obs['cell_type'].astype(str)
    a.obs['Group']     = a.obs['drug'].map(DRUG_TO_INT).astype(int)
    a.write(path)
    print(f"{f}: n={a.n_obs}, genes={a.n_vars}, "
          f"cell_name set ({a.obs['cell_name'].nunique()} types), Group set")
    print("   cell types:", dict(a.obs['cell_name'].value_counts()))

print("\ngene_size for training config:", sc.read_h5ad(f'{DATA_DIR}/gbm_train.h5ad').n_vars)